# Pipeline Architectures

We implemented **two main pipeline architectures**:

1. **Pipeline A — Character-Level Sequential Models**
2. **Pipeline B — Hierarchical Word + Character Model with Attention**

---

# **Pipeline A — Character-Level Encoder + Diacritic Classifier**

This pipeline treats the task as a sequence labeling problem at the character level.

---

## **1. Data Preprocessing**

### **1.1 Data Cleaning**

We removed all characters outside the allowed vocabulary:

- Arabic letters
- Arabic diacritics
- Spaces
- Standard punctuation
- Padding symbol

Any invalid tokens (HTML tags, English characters, digits, unsupported Unicode symbols) were stripped.

### **1.2 Sentence Tokenization**

We split each sentence into smaller segments based on punctuation to reduce excessively long sequences and stabilize LSTM gradients.

### **1.3 Character/Diacritic Separation**

Each Arabic character was decomposed into:

- **Base letter**
- **Associated diacritic** (or PAD if none)

This enabled independent modeling of diacritics as categorical labels.

### **1.4 Encoding**

We performed:

- **Character encoding:** `letter → ID`
- **Diacritic encoding:** `diacritic → ID`
- **Sequence padding:** fixed-length representation for batching

---

## **2. Feature Extraction**

We implemented and evaluated several feature families:

### **2.1 Trainable Embeddings (BiLSTM / BiLSTM-CRF / CNN-CRF)**

- Learned jointly with the model during training
- Embedding dimension tuned experimentally (128–256)

### **2.2 Skip-gram Word Embeddings (CRF Model Only)**

Skip-gram was used in the pure CRF model to provide word-level distributional semantics.

Hyperparameters:

- Embedding dim: 100
- Window size: 5
- Workers: 4
- Min count: 1
- Epochs: 10

### **2.3 CNN Character Encoder** _(for CNN-CRF model)_

We extracted n-gram-like morphological features using a shallow CNN:

- Kernel sizes: (2, 3, 4)
- Filters per kernel: configurable
- ReLU activation + max-pooling

### **2.4 HMM Features**

For the HMM model, we computed:

- **Initial probabilities**
- **Transition probabilities** between diacritics
- **Emission probabilities** `p(letter | diacritic)`

These were estimated directly from the training corpus.

---

## **3. Models**

We trained four main models under Pipeline A.

---

### **3.1 Bidirectional LSTM**

A standard sequence labeling model using only trainable embeddings.

#### **Hyperparameters**

- Embedding dim: 128
- Hidden dim: 256
- Layers: 3
- Dropout: 0.2
- Batch size: 32
- Epochs: 5
- Learning rate: 0.001

---

### **3.2 Conditional Random Field (CRF)**

Uses skip-gram embeddings + handcrafted contextual features.

##### **Skip-gram hyperparameters**

- Embedding dim: 100
- Window size: 5
- Min count: 1
- SG = 1 (skip-gram)
- Epochs: 10

#### **CRF hyperparameters**

- Algorithm: L-BFGS
- C1: 0.1 (L1 regularization)
- C2: 0.1 (L2 regularization)
- Max iterations: 100
- Context window: ±2 characters

---

### **3.3 CNN-CRF Model**

Combines CNN character encoder + skip-gram embeddings + CRF inference.

### **Characteristics**

- CNN extracts morphological features
- CRF ensures valid label transitions

### **Skip-gram & CRF hyperparameters are the same as the other model**

### **CNN hyperparameters**

Character embedding dimension: 30
Number of convolution filters: 50
Kernel sizes: 2, 3, 4
CNN output dimension: 150
Batch size: 512

---

### **3.4 Bidirectional LSTM-CRF**

This hybrid model adds a CRF decoding layer on top of the BiLSTM output, allowing explicit modeling of dependencies between consecutive diacritics.

#### **Hyperparameters**

Same as BiLSTM **except**:

- Batch size: 128
- Epochs: 6

This model ultimately achieved the best performance.

---

### **3.5 Hidden Markov Model (HMM)**

A baseline generative model trained using:

- Initial probability estimation
- Transition matrix
- Emission matrix

Inference performed using Viterbi decoding.

---

# **4. Pipeline B — Hierarchical Word-Level + Character-Level Model with Attention**

## **4.1 Overview**

---

## **4.2 Preprocessing Differences**

---

## **4.3 Feature Extraction**

---

## **4.4 Model Architecture**

---

## **5. Evaluation**

## **5.1 Accuracy Evaluation Template**

### **Accuracy (Case Ending)**

| Model      | Accuracy |
| ---------- | -------- |
| BiLSTM     | 95.69    |
| CRF        | 83.82    |
| CNN-CRF    | 85.17    |
| BiLSTM-CRF | 95.75    |
| HMM        | 45.81    |

---

### **Overall Performance**

| Model      | Accuracy |
| ---------- | -------- |
| BiLSTM     | 97.73    |
| CRF        | 82.33    |
| CNN-CRF    | 83.67    |
| BiLSTM-CRF | 97.80    |
| HMM        | 61.41    |

## **6. Final Model Selection**

We selected the **Bidirectional LSTM-CRF** model as our final submission.

## **Reasons for selection**

- It achieved the **highest accuracy** among all tested models.
- It obtained the **lowest Diacritic Error Rate (DER)**.
- CRF decoding significantly reduced invalid or unlikely diacritic sequences.
- Better generalization to rare diacritics and complex morphological patterns.
