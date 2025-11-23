import torch
from torch.utils.data import Dataset
import numpy as np
import re


class ArabicDataset(Dataset):
    def __init__(self):

        self.arabic_letters = np.load(
            '../data/utils/arabic_letters.pkl', allow_pickle=True)
        self.diacritics = np.load(
            '../data/utils/diacritics.pkl', allow_pickle=True)
        self.punctuations = {".", "،", ":", "؛", "؟", "!", '"', "-"}

        self.valid_chars = set(self.arabic_letters).union(
            set(self.diacritics)).union(self.punctuations).union({" "})

        self.char2id = {char: i for i, char in enumerate(self.arabic_letters)}
        self.id2char = {i: char for i, char in enumerate(self.arabic_letters)}
        self.diacritic2id = np.load(
            '../data/utils/diacritic2id.pkl', allow_pickle=True)
        self.id2diacritic = {id: diacritic for diacritic,
                             id in self.diacritic2id.items()}

        self.train_data = self.load_data('../data/train.txt')
        self.val_data = self.load_data('../data/val.txt')

    def __len__(self):
        return len(self.train_data), len(self.val_data)

    def __getitem__(self, idx):
        return self.train_data[idx], self.val_data[idx]

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


def main():
    dataset = ArabicDataset()
    for i in range(5):
        print(f"Train Sample {i}: {dataset.train_data[i]}")


if __name__ == "__main__":
    main()
