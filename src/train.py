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


def train(model: ArabicModel, train_dataset: ArabicDataset, val_dataset: ArabicDataset):

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    save_path = os.path.join(SRC_DIR, f"../models/{model.name()}.pth")

    best_val_accuracy = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(NUM_EPOCHS):
        model.train()
        for X, Y in tqdm(train_data_loader, desc=f"Training Epoch {epoch+1}"):
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)

            outputs = model(X)
            loss = criterion(outputs.view(-1, outputs.size(-1)), Y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_correct = 0
        total_tokens = 0
        model.eval()
        for X, Y in tqdm(val_data_loader, desc=f"Validating Epoch {epoch+1}"):
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)

            with torch.no_grad():
                outputs = model(X)

            predictions = outputs.argmax(dim=-1)
            mask = (Y != PAD)

            total_correct += ((predictions == Y) & mask).sum().item()
            total_tokens += mask.sum().item()

        accuracy = total_correct / total_tokens * 100
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(model.state_dict(), save_path)

    print(f"Validation Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    train_dataset = ArabicDataset(os.path.join(SRC_DIR, f"../data/train.txt"))
    val_dataset = ArabicDataset(os.path.join(SRC_DIR, f"../data/val.txt"))
    model = ArabicModel(
        vocab_size=len(CHAR2ID),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=len(DIACRITIC2ID),
        PAD=PAD
    ).to(DEVICE)

    train(model, train_dataset, val_dataset)
