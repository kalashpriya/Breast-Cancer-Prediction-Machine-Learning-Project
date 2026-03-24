# *🧬 Breast Cancer Classification*

This repository contains a dataset and classification analysis for predicting breast cancer diagnosis (malignant vs. benign) using machine learning models trained on the **Breast Cancer Wisconsin (Diagnostic) Dataset**.

---

## 📊 Dataset Overview

- **Rows:** 569  
- **Columns:** 31  
- **Size:** 137.9 KB  

### Features
| Column                   | Type    | Description |
|---------------------------|---------|-------------|
| mean radius              | float64 | Mean of cell radius |
| mean texture             | float64 | Mean of cell texture |
| mean perimeter           | float64 | Mean of cell perimeter |
| ...                      | float64 | Other computed features (30 total) |
| target                   | int64   | Diagnosis (0 = Malignant, 1 = Benign) |

---

## 📈 Summary Statistics

| Metric | Mean | Std | Min | 25% | 50% | 75% | Max |
|--------|------|-----|-----|-----|-----|-----|-----|
| mean radius | 14.13 | 3.52 | 6.98 | 11.70 | 13.37 | 15.78 | 28.11 |
| mean texture | 19.29 | 4.30 | 9.71 | 16.17 | 18.84 | 21.80 | 39.28 |
| mean perimeter | 91.97 | 24.30 | 43.79 | 75.17 | 86.24 | 104.10 | 188.50 |
| target | 0.63 | 0.48 | 0 | 0 | 1 | 1 | 1 |

- **Target distribution:**  
  - Malignant (0): ~37%  
  - Benign (1): ~63%  
- **Missing Values:** None  

---

## 🤖 Classification Model

A supervised machine learning classifier was trained to predict tumor diagnosis.

### Predicted Probabilities (first 5 samples)
[5.888e-08, 0.99999, 0.00641, 0.53350, 6.525e-10]

### Confusion Matrix
[[41  1] [ 1 71]]

### Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Malignant) | 0.98 | 0.98 | 0.98 | 42 |
| 1 (Benign)    | 0.99 | 0.99 | 0.99 | 72 |

- **Accuracy:** 0.9825  
- **Macro Avg:** Precision 0.98, Recall 0.98, F1-Score 0.98  
- **Weighted Avg:** Precision 0.98, Recall 0.98, F1-Score 0.98  

---

## 🚀 Project Goals
- Explore breast cancer dataset features.  
- Build classification models to predict diagnosis.  
- Evaluate model performance using standard metrics.  

---

## 📂 Structure
- `load_breast_cancer` → In-built toy datasets provided by scikit-learn. (imported at line 6)
- `Confusion Matrix Heatmap` → Visualization of Confusion Matrix
- `code.py` → Final code (run on Colab or VS Code)
- `README.md` → Project documentation.

---

## 📝 Notes
- Dataset has **no missing values**.  
- Model achieves **98.25% accuracy**.  
- Balanced performance across both malignant and benign cases.  
- Further improvements can be made using feature selection, ensemble methods, or deep learning.  

---

## 📌 Author
**Kalash Priya**  
IILM University, Greater Noida
