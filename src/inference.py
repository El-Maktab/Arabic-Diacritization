import torch
from config import (
    DEVICE,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    SRC_DIR,
    PAD,
    ARABIC_LETTERS,
    CHAR2ID,
    DIACRITIC2ID,
    ID2DIACRITIC,
)
from models import generate_model
import os
import csv
import sys
from collections import defaultdict


def predict(model, encoded_sentence):
    input_tensor = torch.tensor([encoded_sentence], dtype=torch.int64).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
    return outputs.argmax(dim=-1).squeeze(0).cpu().numpy()


def infer(model, model_path, input_path, output_path, text_path=None):
    """
    Run inference on input data and generate diacritized output.

    Args:
        model: The model to use for inference
        model_path: Path to model weights
        input_path: Path to input file (CSV or TXT)
        output_path: Path to output file
        text_path: Path to text file (required for CSV input to get full context)
    """
    model_state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(model_state_dict)

    # Check if input is CSV format (submission format) or text format
    is_csv_input = input_path.endswith(".csv")

    if is_csv_input:
        # Read CSV with id,line_number,letter format (may have case_ending column)
        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Build a set of IDs we need to output predictions for
        target_ids = {int(row["id"]) for row in rows}

        # Read the text file to get full sentences for context
        if text_path is None:
            text_path = os.path.join(
                os.path.dirname(input_path), "dataset_no_diacritics.txt"
            )

        with open(text_path, "r", encoding="utf-8") as f:
            input_lines = f.readlines()

        # Store predictions with their IDs
        output_csv = [["ID", "label"]]
        output_list = []
        current_id = 0  # Global character ID counter

        model.eval()
        for sentence in input_lines:
            encoded_sentence = [CHAR2ID[char] for char in sentence if char in CHAR2ID]

            if len(encoded_sentence) == 0:
                output_list.append("")
                continue

            predictions = predict(model, encoded_sentence)

            diacritized_sentence = ""
            pred_idx = 0
            for char in sentence:
                if char in CHAR2ID:
                    diacritic_id = int(predictions[pred_idx])
                    diacritic = ID2DIACRITIC[diacritic_id]
                    pred_idx += 1

                    # Only output for Arabic letters that are in target IDs
                    if char in ARABIC_LETTERS:
                        if current_id in target_ids:
                            output_csv.append([current_id, diacritic_id])
                        current_id += 1

                    diacritized_sentence += char + diacritic
                else:
                    diacritized_sentence += char

            output_list.append(diacritized_sentence.strip())
    else:
        # Original text file format
        with open(input_path, "r", encoding="utf-8") as f:
            input_data = f.readlines()

        output_list = []
        output_csv = [["ID", "label"]]
        current_id = 0

        model.eval()
        for sentence in input_data:
            encoded_sentence = [CHAR2ID[char] for char in sentence if char in CHAR2ID]

            if len(encoded_sentence) == 0:
                output_list.append("")
                continue

            predictions = predict(model, encoded_sentence)

            diacritized_sentence = ""
            pred_idx = 0
            for char in sentence:
                if char in CHAR2ID:
                    diacritic_id = int(predictions[pred_idx])
                    diacritic = ID2DIACRITIC[diacritic_id]
                    pred_idx += 1
                    if char in ARABIC_LETTERS:
                        output_csv.append([current_id, diacritic_id])
                        current_id += 1
                    diacritized_sentence += char + diacritic
                else:
                    diacritized_sentence += char

            output_list.append(diacritized_sentence.strip())

    # Write diacritized text output
    with open(output_path, "w", encoding="utf-8") as f:
        for line in output_list:
            f.write(line + "\n")

    # Write CSV submission output
    output_path_csv = os.path.splitext(output_path)[0] + ".csv"
    with open(output_path_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(output_csv)


if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print(
            "Usage: python inference.py <model_name> <input_file> <output_file> [text_file]"
        )
        print("  input_file: Can be .txt or .csv (with id,line_number,letter columns)")
        print(
            "  text_file: Optional. Required for CSV input to get full sentence context."
        )
        print(
            "             Defaults to 'dataset_no_diacritics.txt' in input directory."
        )
        print("")
        print("Examples:")
        print("  python inference.py ArabicModel test_no_diacritics.csv output.txt")
        print(
            "  python inference.py ArabicModel test_no_diacritics_ce.csv output_ce.txt dataset_no_diacritics.txt"
        )
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

    # Optional text file path for CSV input
    text_path = None
    if len(sys.argv) == 5:
        text_path = os.path.join(SRC_DIR, f"../input/{sys.argv[4]}")
        if not os.path.exists(text_path):
            print(f"Text file '{sys.argv[4]}' does not exist in input directory.")
            sys.exit(1)

    model = generate_model(
        model_name="LSTMArabicModel",
        vocab_size=len(CHAR2ID),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=len(DIACRITIC2ID),
        PAD=PAD,
    ).to(DEVICE)

    infer(model, model_path, input_path, output_path, text_path)
