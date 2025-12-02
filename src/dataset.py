import torch
from torch.utils.data import Dataset
import numpy as np
import re
from config import (
    VALID_CHARS, PUNCTUATIONS, DIACRITICS,
    PAD, CHAR2ID, DIACRITIC2ID
)
from typing import List


class ArabicDataset(Dataset):
    def __init__(self, file_path: str):
        self.data_X, self.data_Y = self.generate_tensor_data(file_path)

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, idx):
        return self.data_X[idx], self.data_Y[idx]

    def load_data(self, file_path: str):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    line = re.sub(
                        f'[^{re.escape("".join(VALID_CHARS))}]', '', line)
                    line = re.sub(r'\s+', ' ', line)
                    sentences = re.split(
                        f'[{re.escape("".join(PUNCTUATIONS))}]', line)
                    sentences = [s.strip() for s in sentences if s.strip()]
                    data.extend(sentences)

        return np.array(data)

    def extract_text_without_diacritics(self, dataY):
        dataX = dataY.copy()
        for diacritic, _ in DIACRITIC2ID.items():
            dataX = np.char.replace(
                dataX, diacritic, '')
        return dataX

    def encode_data(self, dataX: List[str], dataY: List[str]):
        encoded_data_X = []
        for sentence in dataX:
            encoded_data_X.append([CHAR2ID[char]
                                   for char in sentence if char in CHAR2ID])
        encoded_data_Y = []
        for sentence in dataY:
            encoded_data_Y.append(self.extract_diacritics(sentence))

        max_sentence_len = max(len(sentence) for sentence in encoded_data_X)
        padded_dataX = np.full(
            (len(encoded_data_X), max_sentence_len), PAD, dtype=np.int64)
        for i, seq in enumerate(encoded_data_X):
            padded_dataX[i, :len(seq)] = seq

        padded_dataY = np.full(
            (len(encoded_data_Y), max_sentence_len), PAD, dtype=np.int64)
        for i, seq in enumerate(encoded_data_Y):
            padded_dataY[i, :len(seq)] = seq

        return padded_dataX, padded_dataY

    def generate_tensor_data(self, data_path: str):
        data_Y = self.load_data(data_path)
        data_X = self.extract_text_without_diacritics(data_Y)

        encoded_data_X, encoded_data_Y = self.encode_data(data_X, data_Y)
        data_X = torch.tensor(
            encoded_data_X, dtype=torch.int64)
        data_Y = torch.tensor(
            encoded_data_Y, dtype=torch.int64)

        return data_X, data_Y

    def extract_diacritics(self, sentence: str):
        result = []
        i = 0
        n = len(sentence)
        on_char = False

        while i < n:
            ch = sentence[i]
            if ch in DIACRITICS:
                on_char = False
                # check if next char forms a stacked diacritic
                if i+1 < n and sentence[i+1] in DIACRITICS:
                    combined = ch + sentence[i+1]
                    if combined in DIACRITIC2ID:
                        result.append(DIACRITIC2ID[combined])
                        i += 2
                        continue
                result.append(DIACRITIC2ID[ch])
            elif ch in CHAR2ID:
                if on_char:
                    result.append(DIACRITIC2ID[''])
                on_char = True
            i += 1
        if on_char:
            result.append(DIACRITIC2ID[''])
        return result
