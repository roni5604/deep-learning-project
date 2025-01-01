"""
linear_regression.py

Implements a naive linear regression approach for classification,
mapping pixel intensities -> numeric label.
Not recommended for classification, but shown here as requested.
"""

import os
import glob
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

def load_data_flat(data_dir, img_size=(48, 48)):
    """
    Loads images from subfolders of data_dir, flattens them, returns X (features) and y (labels).
    We'll map classes to numeric labels (0,1,2,...) to handle with LinearRegression.
    """
    X, y = [], []
    class_names = []

    if not os.path.exists(data_dir):
        print(f"[ERROR] Directory not found: {data_dir}")
        return np.array(X), np.array(y), class_names

    # Gather subfolders = classes
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    subdirs.sort()

    class_names = subdirs  # keep an index => class mapping
    class_to_idx = {subdir: idx for idx, subdir in enumerate(subdirs)}

    for c in subdirs:
        class_dir = os.path.join(data_dir, c)
        image_files = glob.glob(os.path.join(class_dir, "*"))
        for img_path in image_files:
            with Image.open(img_path).convert('L') as img:
                img = img.resize(img_size)
                arr = np.array(img, dtype=np.float32).reshape(-1)
                arr /= 255.0
                X.append(arr)
                y.append(class_to_idx[c])

    X = np.array(X)
    y = np.array(y)
    return X, y, class_names

def main():
    print("\n=== Linear Regression for Facial Expression (Experimental) ===")

    train_dir = "data/face-expression-recognition-dataset/images/images/new_train"
    val_dir   = "data/face-expression-recognition-dataset/images/images/new_validation"

    print(f"Loading training data from: {train_dir}")
    X_train, y_train, train_classes = load_data_flat(train_dir)
    print(f"Loading validation data from: {val_dir}")
    X_val, y_val, val_classes = load_data_flat(val_dir)

    # Just ensure classes are consistent
    if train_classes != val_classes:
        print("[WARNING] Class mismatch between train/val directories. Proceeding anyway.")

    # Build model
    print("Fitting linear regression model (Note: this is not typical for classification).")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict continuous outputs, then round them to the nearest integer
    preds_val_float = model.predict(X_val)
    preds_val_rounded = np.rint(preds_val_float).astype(int)

    # Clip predictions so they are in valid range
    preds_val_clipped = np.clip(preds_val_rounded, 0, len(train_classes) - 1)

    # Compare with ground truth
    accuracy  = accuracy_score(y_val, preds_val_clipped)
    precision = precision_score(y_val, preds_val_clipped, average='macro', zero_division=0)
    recall    = recall_score(y_val, preds_val_clipped, average='macro', zero_division=0)

    print("\n=== Linear Regression Metrics on Validation Set ===")
    print(f" - Accuracy:  {accuracy:.4f}")
    print(f" - Precision: {precision:.4f}")
    print(f" - Recall:    {recall:.4f}")

    print("\n(As expected, using linear regression for classification is sub-optimal.)")

if __name__ == "__main__":
    main()
