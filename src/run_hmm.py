import os
import pickle
from dataset import generate_dataset
from models import generate_model
from config import SRC_DIR, CHAR2ID, DIACRITIC2ID, PAD

def tensor_to_sequences(tensor):
    sequences = []
    for row in tensor:
        seq = [int(x) for x in row if int(x) != PAD]
        sequences.append(seq)
    return sequences

if __name__ == "__main__":
    # Load training data
    train_dataset = generate_dataset("ArabicDataset", os.path.join(SRC_DIR, "../data/train.txt"))
    X_train = tensor_to_sequences(train_dataset.data_X)
    Y_train = tensor_to_sequences(train_dataset.data_Y)

    # Initialize HMM
    hmm_model = generate_model(
        model_name="HMMArabicModel",
        num_states=len(DIACRITIC2ID),
        num_observations=len(CHAR2ID),
        pad_state_id=PAD
    )

    # Fit HMM
    print("Fitting HMM...")
    hmm_model.fit(X_train, Y_train)
    print("HMM fitted.")

    # Save the trained model
    model_path = os.path.join(SRC_DIR, "../models/hmm_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(hmm_model, f)

    print(f"HMM model saved to {model_path}")