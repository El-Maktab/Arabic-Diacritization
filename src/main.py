from xml.parsers.expat import model
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import re

PAD = -1

class ArabicDataset(Dataset):
    def __init__(self):

        self.arabic_letters = np.load(
            '../data/utils/arabic_letters.pkl', allow_pickle=True)
        self.diacritics = np.load(
            '../data/utils/diacritics.pkl', allow_pickle=True)
        self.punctuations = {".", "،", ":", "؛", "؟", "!", '"', "-"}

        self.valid_chars = set(self.arabic_letters).union(
            set(self.diacritics)).union(self.punctuations).union({" "})

        self.char2id = {char: id for id, char in enumerate(self.arabic_letters)}
        self.char2id[" "] = len(self.arabic_letters)
        self.id2char = {id: char for char, id in self.char2id.items()}
        self.diacritic2id = np.load(
            '../data/utils/diacritic2id.pkl', allow_pickle=True)
        self.id2diacritic = {id: diacritic for diacritic,
                             id in self.diacritic2id.items()}

        self.train_data_Y = self.load_data('../data/train.txt')
        self.train_data_X = self.train_data_Y.copy()
        for diacritic, id in self.diacritic2id.items():
            self.train_data_X = np.char.replace(
                self.train_data_X, diacritic, '')

        encoded_train_dataX = []
        for sentence in self.train_data_X:
            encoded_train_dataX.append([self.char2id[char] for char in sentence if char in self.char2id])

        encoded_train_dataY = []
        for sentence in self.train_data_Y:
            encoded_train_dataY.append(self.extract_diacritics(sentence))

        max_sentence_len = max(len(sentence) for sentence in encoded_train_dataX)
        padded_train_dataX = np.full((len(encoded_train_dataX), max_sentence_len), PAD, dtype=np.int64)
        padded_train_dataY = np.full((len(encoded_train_dataY), max_sentence_len), PAD, dtype=np.int64)
                
        self.train_data_X = torch.tensor(padded_train_dataX, dtype=torch.int)
        self.train_data_Y = torch.tensor(padded_train_dataY, dtype=torch.int)
                
        self.val_data_Y = self.load_data('../data/val.txt')
        self.val_data_X = self.val_data_Y.copy()
        for diacritic, id in self.diacritic2id.items():
            self.val_data_X = np.char.replace(
                self.val_data_X, diacritic, '')
        
        encoded_val_dataX = []
        for sentence in self.val_data_X:
            encoded_val_dataX.append([self.char2id[char] for char in sentence if char in self.char2id])
        
        encoded_val_dataY = []
        for sentence in self.val_data_Y:
            encoded_val_dataY.append(self.extract_diacritics(sentence))
        
        max_sentence_len = max(len(sentence) for sentence in encoded_val_dataX)
        padded_val_dataX = np.full((len(encoded_val_dataX), max_sentence_len), PAD, dtype=np.int64)
        padded_val_dataY = np.full((len(encoded_val_dataY), max_sentence_len), PAD, dtype=np.int64)
        
        self.val_data_X = torch.tensor(padded_val_dataX, dtype=torch.int)
        self.val_data_Y = torch.tensor(padded_val_dataY, dtype=torch.int)

    def __len__(self):
        return len(self.train_data_X), len(self.val_data_X)

    def __getitem__(self, idx):
        return self.train_data_X[idx], self.val_data_X[idx]

    def load_data(self, file_p):
        data = []
        with open(file_p, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Remove invalid characters
                    line = re.sub(
                        f'[^{re.escape("".join(self.valid_chars))}]', '', line)
                    # Normalize spaces
                    line = re.sub(r'\s+', ' ', line)
                    # Split into sentences based on punctuation
                    sentences = re.split(
                        f'[{re.escape("".join(self.punctuations))}]', line)
                    sentences = [s.strip() for s in sentences if s.strip()]
                    data.extend(sentences)

        data = np.array(data)
        return data
    
    def extract_diacritics(self, sentence):
        result = []
        i = 0
        n = len(sentence)
        on_char = False

        while i < n:
            ch = sentence[i]

            if ch in self.diacritics:
                # check if next char forms a stacked diacritic
                if i+1 < n and sentence[i+1] in self.diacritics:
                    combined = ch + sentence[i+1]
                    if combined in self.diacritic2id:
                        result.append(self.diacritic2id[combined])
                        i += 2
                        continue
                result.append(self.diacritic2id[ch])
                on_char = False
            else:
                if on_char:
                    result.append(self.diacritic2id[''])
                on_char = True

            i += 1

        return result


class ArabicModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(ArabicModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output


def main():
    dataset = ArabicDataset()
    model = ArabicModel(
        vocab_size=len(dataset.char2id),
        embedding_dim=128,
        hidden_dim=256,
        output_dim=len(dataset.diacritic2id)
    )
    
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(5):
        print(f"Epoch {epoch+1}/5")
        # Training and validation logic would go here
        model.train()
        for i in range(len(dataset.train_data_X)):
            inputs = dataset.train_data_X[i]
            targets = dataset.train_data_Y[i]
            outputs = model(inputs.unsqueeze(0))
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0
        total_accuracy = 0
        for i in range(len(dataset.val_data_X)):
            input = dataset.val_data_X[i]
            target = dataset.val_data_Y[i]
            output = model(input.unsqueeze(0))
            output = output.view(-1, output.size(-1))
            target = target.view(-1)
            loss = criterion(output, targets_flat)
            prediction = output.argmax(dim=1)
            correct_predictions = (prediction == target).sum().item()
            actual_predictions = target.size(0)
            accuracy = (correct_predictions / actual_predictions) * 100
            total_loss += loss.item()
            total_accuracy += accuracy

        avg_loss = total_loss / len(dataset.val_data_X)
        avg_accuracy = total_accuracy / len(dataset.val_data_X)
        print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy:.2f}%")


if __name__ == "__main__":
    main()
