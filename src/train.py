from tqdm import tqdm  # type: ignore
import torch
import torch.nn as nn
from config import (
    DEVICE, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, HIDDEN_DIM, EMBEDDING_DIM,
    SRC_DIR, PAD, CHAR2ID, DIACRITIC2ID
)
from models import ArabicModel
from dataset import ArabicDataset
from torch.utils.data import DataLoader
import os


def train(model: ArabicModel, train_dataset: ArabicDataset):

    save_path = os.path.join(SRC_DIR, f"../models/{model.name()}.pth")

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch in range(NUM_EPOCHS):
        total_correct = 0
        total_tokens = 0
        epoch_loss = 0

        model.train()
        for train_X, train_Y in tqdm(train_data_loader, desc=f"Training Epoch {epoch + 1}"):
            train_X = train_X.to(DEVICE)
            train_Y = train_Y.to(DEVICE)

            output = model(train_X)

            loss = criterion(output.view(-1, output.size(-1)),
                             train_Y.view(-1))
            epoch_loss += loss.item()

            mask = (train_Y != PAD)
            prediction = output.argmax(dim=-1)
            total_correct += ((prediction == train_Y) & mask).sum().item()
            total_tokens += mask.sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_acc = (total_correct / total_tokens) * 100
        print(
            f'Epochs: {epoch + 1} | Train Loss: {epoch_loss} \
            | Train Accuracy: {epoch_acc}\n')

    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    train_dataset = ArabicDataset(os.path.join(SRC_DIR, f"../data/train.txt"))
    model = ArabicModel(
        vocab_size=len(CHAR2ID),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=len(DIACRITIC2ID),
        PAD=PAD
    )

    train(model, train_dataset)
