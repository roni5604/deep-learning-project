
# Face Expression Recognition Project

A project to classify facial expressions using multiple approaches:
- Baseline (majority class)
- Logistic Regression
- Linear Regression (for classification, not typical)
- Deeper MLP (multi-layer perceptron)
- Advanced CNN (with data augmentation)

---

## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Running the Scripts](#running-the-scripts)
6. [Expected Outputs](#expected-outputs)
7. [Troubleshooting](#troubleshooting)

---

## Overview

- **Goal**: Detect a person’s facial expression from grayscale images (48×48).
- **Dataset**: [Kaggle - Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset).
- **Splits**: 70% train, 15% validation, 15% test.
- **Models**:
  1. **Baseline** (predict majority class).
  2. **Logistic Regression** with scaling.
  3. **Linear Regression** (experimental).
  4. **Deeper MLP** (two hidden layers + dropout).
  5. **Advanced CNN** (augmentations + dropout).

---

## Requirements

- **Python** >= 3.8
- **PyTorch**, **torchvision**
- **scikit-learn**, **numpy**, **Pillow**, **matplotlib**, etc.
- **Kaggle** (optional) for dataset download.

The versions are listed in `requirements.txt`.

---

## Installation

1. **Clone or download**:
   ```bash
   git clone https://github.com/your-username/face_expression_project.git
   cd face_expression_project
   ```
2. **Create virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   On Windows:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **(Optional) Kaggle**:
   - Place `kaggle.json` in `~/.kaggle/`.
   - `pip install kaggle`.

---

## Dataset Preparation

1. **Download** (optional):
   ```bash
   python scripts/dataset_download.py
   ```
2. **Prepare** (merge old train/val, re-split):
   ```bash
   python scripts/prepare_dataset.py
   ```
   Creates `new_train`, `new_validation`, and `test` directories under `data/face-expression-recognition-dataset/images/images/`.

---

## Running the Scripts

In order (or selectively):

1. **Baseline** (majority class):
   ```bash
   python baseline.py
   ```
2. **Logistic Regression**:
   ```bash
   python logistic_regression.py
   ```
3. **Linear Regression** (classification by rounding outputs):
   ```bash
   python linear_regression.py
   ```
4. **Deeper MLP**:
   ```bash
   python basic_nn.py
   ```
5. **Advanced CNN** (data augmentation):
   ```bash
   python advanced_cnn.py
   ```

**Optional**: Summarize results via:
```bash
python final_summary.py
```

---

## Expected Outputs

Below are the **final** validation metrics from the latest run:

1. **Baseline**  
   - Accuracy: 0.2504  
   - Precision: 0.0358  
   - Recall: 0.1429  

2. **Logistic Regression**  
   - Accuracy: 0.3518  
   - Precision: 0.3286  
   - Recall: 0.3032  

3. **Linear Regression**  
   - Accuracy: 0.2207  
   - Precision: 0.2207  
   - Recall: 0.1542  

4. **Deeper MLP**  
   - Accuracy: 0.4548  
   - Precision: 0.4717  
   - Recall: 0.4163  

5. **Advanced CNN**  
   - Accuracy: 0.5597  
   - Precision: 0.5698  
   - Recall: 0.4804  

---

## Troubleshooting

- **ConvergenceWarning**: Increase `max_iter` for logistic regression, or standardize features more thoroughly.
- **Low Accuracy**: Increase epochs, tweak batch size, or refine architecture.
- **No GPU**: Code runs on CPU but might be slow.  
- **Overfitting**: Increase dropout/weight decay, add more augmentation.  
- **Underfitting**: Use deeper models, more epochs, or different LR schedules.

Refer to `final_report.md` for deeper analysis of each model’s performance and potential future improvements.
