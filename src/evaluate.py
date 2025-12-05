from tqdm import tqdm  # type: ignore
import torch
from config import (
    DEVICE, BATCH_SIZE, HIDDEN_DIM, EMBEDDING_DIM,
    SRC_DIR, SPACE, PAD, CHAR2ID, DIACRITIC2ID
)
from models import generate_model
from dataset import generate_dataset
from torch.utils.data import DataLoader
import os
import sys


def evaluate(model: torch.nn.Module, val_dataset: torch.utils.data.Dataset):

    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    if torch.cuda.is_available():
        model = model.cuda()

    total_correct_without_ending = 0
    total_tokens_without_ending = 0
    total_correct_ending = 0
    total_tokens_ending = 0
    total_correct = 0
    total_tokens = 0
    model.eval()
    with torch.no_grad():

        for val_X, val_Y in tqdm(val_data_loader):
            val_X = val_X.to(DEVICE)
            val_Y = val_Y.to(DEVICE)

            output = model(val_X)
            prediction = output.argmax(dim=-1)

            padding_mask = (val_Y == PAD)
            shifted = torch.roll(val_X, shifts=-1, dims=1)
            end_of_word_mask = (shifted == SPACE) | (shifted == PAD)

            last_char_mask = end_of_word_mask & (~padding_mask)
            rest_of_word_mask = (~end_of_word_mask) & (~padding_mask)
            everything_mask = ~padding_mask

            total_correct_ending += ((prediction == val_Y)
                                     & last_char_mask).sum().item()
            total_tokens_ending += last_char_mask.sum().item()

            total_correct_without_ending += ((prediction == val_Y) &
                                             rest_of_word_mask).sum().item()
            total_tokens_without_ending += rest_of_word_mask.sum().item()

            total_correct += ((prediction == val_Y) &
                              everything_mask).sum().item()
            total_tokens += everything_mask.sum().item()

        val_accuracy = (total_correct / total_tokens) * 100
        val_accuracy_without_ending = (total_correct_without_ending /
                                       total_tokens_without_ending) * 100
        val_accuracy_ending = (total_correct_ending /
                               total_tokens_ending) * 100
        print(
            f"Validation Accuracy (Overall): {val_accuracy:.2f}%\n" +
            f"Validation Accuracy (Without Last Character): {val_accuracy_without_ending:.2f}%\n" +
            f"Validation Accuracy (Last Character): {val_accuracy_ending:.2f}%\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <model_name>")
        sys.exit(1)

    if not os.path.exists(os.path.join(SRC_DIR, f"../models/{sys.argv[1]}.pth")):
        print(f"Model '{sys.argv[1]}' does not exist in models directory.")
        sys.exit(1)

    model_path = os.path.join(SRC_DIR, f"../models/{sys.argv[1]}.pth")

    val_dataset = generate_dataset(
        "ArabicDataset", os.path.join(SRC_DIR, f"../data/val.txt"))
    model = generate_model(
        model_name="LSTMArabicModel",
        vocab_size=len(CHAR2ID),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=len(DIACRITIC2ID),
        PAD=PAD
    )
    model_state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(model_state_dict)

    evaluate(model, val_dataset)
