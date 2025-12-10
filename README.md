# Arabic-Diacritization

A system that takes an Arabic sentence and produces the same sentence after restoring the missing diacritics.

---

## Project Structure

```
Arabic-Diacritization/
├── models/              # Trained model weights
├── input/               # Input files (text or CSV)
├── output/              # Generated output files
├── src/                 # Source code and notebooks
│   ├── inference.py     # Pipeline A inference script
│   ├── d2-paper-implementation.ipynb  # Pipeline B notebook
│   └── ...
└── data/                # Training data
```

---

## Pipeline A: Character-Level Models

### How to Run Inference

```bash
cd src
python inference.py <model_name> <input_file.csv|input_file.txt> <output_file> [<context_file.txt>]
```

### Examples

**With text file input:**

```bash
python inference.py ArabicBiLSTMModel input.txt output.txt
```

**With CSV file input (for Kaggle submission):**

```bash
python inference.py ArabicBiLSTMModel test_no_diacritics.csv output.txt
```

**With case-ending CSV (specify text file for context):**

```bash
python inference.py ArabicBiLSTMModel test_no_diacritics_ce.csv output_ce.txt dataset_no_diacritics.txt
```

### Output

- `output.txt` - Diacritized text
- `output.csv` - Submission file with `ID,label` columns

---

## Pipeline B: Hierarchical D2 Model

The D2 model uses a notebook for training and inference.

### How to Run with Pre-trained Model

1. Open `src/d2-paper-implementation.ipynb` in Jupyter/VS Code

2. In the **CONFIG cell**, set:

```python
CONFIG = {
    "RUNNING_IN_KAGGLE": False,
    ...
    "TRAIN": {
        "NUM_EPOCHS": 0,  # Set to 0 to skip training
        ...
    },
    ...
}
```

3. Set the resume path to load your trained model:

```python
RESUME_PATH = "d2_model_best.pth"  # Or any saved checkpoint
```

4. Run all cells - it will:
   - Load the pre-trained model
   - Skip training (since epochs = 0)
   - Run evaluation and generate predictions

---
