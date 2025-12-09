import torch
import numpy as np
import os
from typing import Any

# Configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SRC_DIR = os.path.dirname(__file__)

# Global registries
DATASET_REGISTRY: dict[str, Any] = {}
MODEL_REGISTRY: dict[str, Any] = {}

# Model hyperparameters
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
NUM_LAYERS = 5
DROPOUT = 0.4

# Data parameters
ARABIC_LETTERS = sorted(
    np.load('../data/utils/arabic_letters.pkl', allow_pickle=True))
DIACRITICS = sorted(np.load(
    '../data/utils/diacritics.pkl', allow_pickle=True))
PUNCTUATIONS = {".", "،", ":", "؛", "؟", "!", '"', "-"}

VALID_CHARS = set(ARABIC_LETTERS).union(
    set(DIACRITICS)).union(PUNCTUATIONS).union({" "})

CHAR2ID = {char: id for id, char in enumerate(ARABIC_LETTERS)}
CHAR2ID[" "] = len(ARABIC_LETTERS)
CHAR2ID["<PAD>"] = len(ARABIC_LETTERS) + 1
PAD = CHAR2ID["<PAD>"]
SPACE = CHAR2ID[" "]
ID2CHAR = {id: char for char, id in CHAR2ID.items()}

DIACRITIC2ID = np.load('../data/utils/diacritic2id.pkl', allow_pickle=True)
ID2DIACRITIC = {id: diacritic for diacritic, id in DIACRITIC2ID.items()}
