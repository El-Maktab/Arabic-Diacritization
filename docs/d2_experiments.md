## Experiment Results

| Exp # | Accuracy | CE Accuracy | Word LSTM Layers | Char LSTM Layers | Word Hidden | Char Hidden | Dropout | Word Dropout | Batch Size | LR    | Max Words | Stride | Notes    |
| ----- | -------- | ----------- | ---------------- | ---------------- | ----------- | ----------- | ------- | ------------ | ---------- | ----- | --------- | ------ | -------- |
| 1     | 96.1     | 92.8        | 2                | 2                | 256         | 512         | 0.25    | 0.2          | 32         | 0.002 | 10        | 5      | Achieved low in training set  |
| 2     | 95.0     | 93.0        | 2                | 2                | 256         | 512         | 0.25    | 0.0          | 32         | 0.002 | 10        | 5      | Tried to remove the word dropout |
| 3     | 96.87    | 95          | 2                | 3                | 256         | 512         | 0.25    | 0.2          | 32         | 0.002 | 10        | 5      | Better on training set |
| 4     | 81.0     | -           | 2                | 5                | 256         | 512         | 0.25    | 0.2          | 32         | 0.002 | 10        | 5      | Over fitted |
| 5     | 96.98    | 94.77       | 2                | 3                | 256         | 512         | 0.25    | 0.2          | 128        | 0.002 | 20        | 15     | Baseline |

## Notes

- **CE (Case Endings)**: Diacritics at word boundaries, determined by syntax
- **DER**: Diacritic Error Rate
- **WER**: Word Error Rate (any error in word counts as wrong)
- All experiments use Adam optimizer with ReduceLROnPlateau scheduler
- Early stopping patience: 3 epochs
