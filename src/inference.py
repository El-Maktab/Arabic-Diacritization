import torch
from config import (
    DEVICE, EMBEDDING_DIM, HIDDEN_DIM, SRC_DIR,
    PAD, CHAR2ID, ID2CHAR, DIACRITIC2ID, ID2DIACRITIC
)
from models import ArabicModel
import os
import sys


def predict(model, encoded_sentence):
    input_tensor = torch.tensor(
        [encoded_sentence], dtype=torch.int64).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
    return outputs.argmax(dim=-1).squeeze(0).cpu().numpy()


def infer(model_path, input_path):

    model = ArabicModel(
        vocab_size=len(CHAR2ID),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=len(DIACRITIC2ID),
        PAD=PAD
    ).to(DEVICE)

    model_state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(model_state_dict)

    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = f.readlines()

    output_list = []

    model.eval()
    for sentence in input_data:
        encoded_sentence = [CHAR2ID[char]
                            for char in sentence if char in CHAR2ID]

        predictions = predict(model, encoded_sentence)

        diacritized_sentence = ""
        for char_id, diacritic_id in zip(encoded_sentence, predictions):
            char = ID2CHAR[char_id]
            diacritic = ID2DIACRITIC[diacritic_id]
            diacritized_sentence += char + diacritic

        output_list.append(diacritized_sentence)

    with open(output_path, 'w', encoding='utf-8') as f:
        for line in output_list:
            f.write(line + '\n')


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python inference.py <model_name> <input_file> <output_file>")
        sys.exit(1)

    if not os.path.exists(os.path.join(SRC_DIR, f"../models/{sys.argv[1]}.pth")):
        print(f"Model '{sys.argv[1]}' does not exist in models directory.")
        sys.exit(1)
    if not os.path.exists(os.path.join(SRC_DIR, f"../input/{sys.argv[2]}")):
        print(f"Input file '{sys.argv[2]}' does not exist in input directory.")
        sys.exit(1)

    model_path = os.path.join(SRC_DIR, f"../models/{sys.argv[1]}.pth")
    input_path = os.path.join(SRC_DIR, f"../input/{sys.argv[2]}")
    output_path = os.path.join(SRC_DIR, f"../output/{sys.argv[3]}")

    infer(model_path, input_path)
