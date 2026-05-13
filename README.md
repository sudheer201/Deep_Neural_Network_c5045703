# Optimizing Story Diversity in Visual Storytelling using Reinforcement Learning

## 📌 Overview

This project addresses the task of predicting the **position (1–5)** of a sentence within a story sequence.
The problem is formulated as a **5-class classification task**, where each input text corresponds to a specific narrative position.

---

## 📂 Dataset

We use the **StoryReasoning dataset** as specified in the reassessment task. 

Each story consists of 5 ordered elements. Each element contains:

* A text sentence
* (Image and metadata available but **not used**)

### ✅ Modality Chosen

**Text modality only (LSTM-based model)**
➡️ This follows the requirement: *only ONE modality must be used* 

---

## ⚙️ Dataset Construction

* Extracted sentences from each story
* Assigned labels:

  * 1 → First position
  * 2 → Second position
  * 3 → Third position
  * 4 → Fourth position
  * 5 → Fifth position
* Converted labels to (0–4) for training
* Split dataset into:

  * **80% training**
  * **20% validation**

---

## 🧠 Model Architecture

A TensorFlow-based LSTM model:

* Embedding Layer
* LSTM Layer
* Dropout Layer
* Dense Layer (ReLU)
* Output Layer (Softmax, 5 classes)

### Training Details

* Loss Function: `Sparse Categorical Crossentropy`
* Optimizer: `Adam`
* Batch Size: 32
* Epochs: 10

---

## 📊 Experiments (5 Required Variations)

Each experiment modifies **only one parameter**:

| Experiment   | Modification               |
| ------------ | -------------------------- |
| baseline     | Default parameters         |
| more_units   | LSTM units increased (256) |
| less_units   | LSTM units reduced (64)    |
| high_dropout | Dropout increased (0.5)    |
| no_dropout   | Dropout removed (0.0)      |

---

## 📈 Results

| Experiment   | Train Loss | Val Loss   | Train Acc  | Val Acc    |
| ------------ | ---------- | ---------- | ---------- | ---------- |
| baseline     | 0.7843     | 0.6572     | 0.6005     | 0.5979     |
| more_units   | 0.6599     | 0.6646     | 0.5993     | 0.6027     |
| less_units   | **0.2798** | **0.2786** | **0.7986** | **0.8054** |
| high_dropout | 0.3573     | 0.2856     | 0.8019     | 0.7925     |
| no_dropout   | 1.6096     | 1.6097     | 0.2000     | 0.2001     |

---

## 🔍 Key Observations

* ✅ **Best Performance:** `less_units`

  * Highest validation accuracy (**80.54%**)
  * Lowest loss → better generalization

* ⚠️ **Overfitting Indicator:** `high_dropout`

  * Slight drop in validation accuracy

* ❌ **Worst Model:** `no_dropout`

  * Accuracy ≈ random guessing (20%)
  * Indicates severe underfitting

---

## 📉 Outputs Generated

* Training & validation loss curves (saved in `results/figures/`)
* Results table (saved in `results/tables/results.csv`)
* Predictions using `model.predict()`

---

## 🔮 Predictions Example

The trained model predicts story positions:

Example:
Text: *"She opened the letter with trembling hands."*
Predicted Position: **2**

---

## 📐 Accuracy Calculation

Accuracy is computed as:

```
Accuracy = correct_predictions / total_predictions
```

(as required in the task specification) 

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/train.py
```

---

## ✅ Addressing Feedback

The previous submission lacked:

* ❌ Predictions
* ❌ Training details
* ❌ Evidence of results

This updated version includes:

* ✔ Model predictions
* ✔ Training & validation loss per epoch
* ✔ Accuracy computation
* ✔ 5 controlled experiments
* ✔ Saved plots and results

---

## 🎯 Conclusion

The model successfully learns **temporal structure in stories** using text.
Reducing model complexity (less_units) improved generalization, while removing regularization (no_dropout) severely degraded performance.

---

## 📌 Future Work

* Apply **BiLSTM or Attention mechanisms**
* Explore **Transformer-based models (BERT)**
* Extend to **reinforcement learning for story diversity optimization**

---
