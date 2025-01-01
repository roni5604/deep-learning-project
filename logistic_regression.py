import os
import glob
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score


def load_data_flat(data_dir, img_size=(48, 48)):
    """
    Loads all images from each subfolder of data_dir,
    flattens them, and returns X (features) and y (labels).
    """
    X, y = [], []

    if not os.path.exists(data_dir):
        print(f"[ERROR] Directory not found: {data_dir}")
        return np.array(X), np.array(y)

    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for c in classes:
        class_dir = os.path.join(data_dir, c)
        image_files = glob.glob(os.path.join(class_dir, "*"))
        for img_path in image_files:
            with Image.open(img_path).convert('L') as img:
                img = img.resize(img_size)
                arr = np.array(img, dtype=np.float32).reshape(-1)
                X.append(arr)
                y.append(c)

    X = np.array(X)
    y = np.array(y)
    # Normalize pixel values [0, 255] -> [0,1]
    X /= 255.0
    return X, y


def main():
    print("\n=== Logistic Regression (Multi-Class) ===")
    train_dir = "data/face-expression-recognition-dataset/images/images/new_train"
    val_dir = "data/face-expression-recognition-dataset/images/images/new_validation"

    print(f"Loading flattened data from:\n - Train: {train_dir}\n - Val: {val_dir}")
    X_train, y_train = load_data_flat(train_dir)
    X_val, y_val = load_data_flat(val_dir)

    if X_train.size == 0 or X_val.size == 0:
        print("[ERROR] Could not load data properly. Exiting.")
        return

    print(f"Training shape: {X_train.shape}, Validation shape: {X_val.shape}")
    print("Fitting logistic regression model (softmax)...")

    # Increase max_iter to help convergence
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000)
    model.fit(X_train, y_train)

    print("Predicting on validation set...")
    y_pred = model.predict(X_val)

    # Compute metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_val, y_pred, average='macro', zero_division=0)

    print("\nLogistic Regression Metrics on Validation Set:")
    print(f" - Accuracy:  {accuracy:.4f}")
    print(f" - Precision: {precision:.4f}")
    print(f" - Recall:    {recall:.4f}")


if __name__ == "__main__":
    main()
