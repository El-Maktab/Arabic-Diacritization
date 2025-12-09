import os
import pickle
from dataset import generate_dataset
from models import generate_model
from config import SRC_DIR, CHAR2ID, DIACRITIC2ID, PAD, SPACE

def tensor_to_sequences(tensor):
    sequences = []
    for row in tensor:
        seq = [int(x) for x in row if int(x) != PAD]
        sequences.append(seq)
    return sequences

if __name__ == "__main__":
    # Load validation data
    val_dataset = generate_dataset("ArabicDataset", os.path.join(SRC_DIR, "../data/val.txt"))
    X_val = tensor_to_sequences(val_dataset.data_X)
    Y_val = tensor_to_sequences(val_dataset.data_Y)

    # Load trained HMM
    model_path = os.path.join(SRC_DIR, "../models/hmm_model.pkl")
    with open(model_path, "rb") as f:
        hmm_model = pickle.load(f)
    print(f"HMM model loaded from {model_path}")

    # Evaluation metrics
    total_correct = 0
    total_tokens = 0
    total_correct_ending = 0
    total_tokens_ending = 0
    total_correct_without_ending = 0
    total_tokens_without_ending = 0

    for obs_seq, true_seq in zip(X_val, Y_val):
        pred_seq = hmm_model.viterbi(obs_seq)

        # Determine end-of-word positions
        end_of_word_mask = []
        for i, o in enumerate(obs_seq):
            if i + 1 < len(obs_seq):
                end_of_word_mask.append(obs_seq[i + 1] == SPACE)
            else:
                end_of_word_mask.append(True)
        end_of_word_mask = [bool(x) for x in end_of_word_mask]

        for i, (pred, true) in enumerate(zip(pred_seq, true_seq)):
            if true == PAD:
                continue

            total_tokens += 1
            if pred == true:
                total_correct += 1

            if end_of_word_mask[i]:
                total_tokens_ending += 1
                if pred == true:
                    total_correct_ending += 1
            else:
                total_tokens_without_ending += 1
                if pred == true:
                    total_correct_without_ending += 1

    val_accuracy = total_correct / total_tokens * 100 if total_tokens else 0
    val_accuracy_ending = total_correct_ending / total_tokens_ending * 100 if total_tokens_ending else 0
    val_accuracy_without_ending = total_correct_without_ending / total_tokens_without_ending * 100 if total_tokens_without_ending else 0

    print(
        f"Validation Accuracy (Overall): {val_accuracy:.2f}%\n" +
        f"Validation Accuracy (Without Last Character): {val_accuracy_without_ending:.2f}%\n" +
        f"Validation Accuracy (Last Character): {val_accuracy_ending:.2f}%\n"
    )