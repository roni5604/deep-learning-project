
# Face Expression Recognition Project

A comprehensive project demonstrating baseline, logistic regression, a deeper MLP (multi-layer perceptron), and an advanced CNN (with data augmentation) to classify facial expressions.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Running the Code](#running-the-code)
6. [Expected Outputs](#expected-outputs)
7. [Troubleshooting](#troubleshooting)

---

## Project Overview

- **Goal**: Classify images into one of several facial expression classes (e.g., happy, sad, angry, etc.).  
- **Models Implemented**:
  1. **Baseline** (majority class).
  2. **Logistic Regression**.
  3. **Deeper MLP** (two hidden layers + dropout).
  4. **Advanced CNN** (with data augmentation).

The dataset comprises grayscale images (48×48 pixels). Data is split into **train**, **validation**, and **test** sets (70%–15%–15%).

---

## Requirements

- **Python** >= 3.8
- **Pip** for dependencies
- **PyTorch** (CPU or GPU)
- **scikit-learn**, **numpy**, **Pillow**, **matplotlib**, etc.
- **Kaggle** (optional) if you want to automatically download the dataset.

You can see the specific package versions in `requirements.txt`.

---

## Installation

1. **Clone/Download** this repository or place these files in a dedicated folder:
   ```bash
   git clone https://github.com/roni5604/face_expression_project.git
   cd face_expression_project
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   On Windows, use:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```
   > For macOS M1/M2, check [PyTorch official docs](https://pytorch.org) for a compatible build if needed.

4. **(Optional) Kaggle Setup** if you plan to use the Kaggle API:
   - Place `kaggle.json` in `~/.kaggle/` with the right permissions.
   - `pip install kaggle`.

---

## Dataset Preparation

1. **Download the dataset** from Kaggle (optional):
   ```bash
   python scripts/dataset_download.py
   ```
   This will download and unzip into `data/face-expression-recognition-dataset/`.

2. **Prepare the dataset**:
   ```bash
   python scripts/prepare_dataset.py
   ```
   - Merges original `train` and `validation` sets from the Kaggle folder.
   - Re-splits them into **new_train**, **new_validation**, and **test**.

You can skip the above steps if your data is already in the correct location.

---

## Running the Code

Execute scripts in this order (or selectively) to see each model’s performance:

1. **Baseline** (Majority Class):
   ```bash
   python baseline.py
   ```
2. **Logistic Regression**:
   ```bash
   python logistic_regression.py
   ```
3. **Deeper MLP** (two hidden layers + dropout):
   ```bash
   python basic_nn.py
   ```
4. **Advanced CNN** (with data augmentation):
   ```bash
   python advanced_cnn.py
   ```
5. **Summary** (optional):
   ```bash
   python final_summary.py
   ```
   - Prints a concise table of final metrics.

---

## Expected Outputs

Below is an **example** of final validation metrics you might see (actual values can vary):

1. **Baseline**  
   - Accuracy: 0.2504  
   - Precision: 0.0358  
   - Recall: 0.1429  

2. **Logistic Regression**  
   - Accuracy: 0.3518  
   - Precision: 0.3286  
   - Recall: 0.3032  

3. **Deeper MLP**  
   - Accuracy: 0.3766  
   - Precision: 0.2933  
   - Recall: 0.3069  

4. **Advanced CNN** (Augment)  
   - Accuracy: 0.5437  
   - Precision: 0.5883  
   - Recall: 0.4619  

When you run the models, the console prints intermediate epoch logs (for MLP and CNN) and final metrics on the validation set.

---

## Troubleshooting

1. **ConvergenceWarning** (Logistic Regression):  
   - Increase `max_iter` or normalize data more thoroughly.  
2. **Low Performance**:
   - Try increasing epochs, adjusting learning rates, or using GPU.  
   - Check if your data is placed in the correct folders.  
3. **Slow Training**:
   - Use a GPU if available (`torch.cuda.is_available()`).  
   - Reduce batch size for memory constraints, or optimize your network.

**For deeper explanations** of each step, see the project’s `final_report.md`, which includes a detailed analysis of results, improvements, and future directions.

