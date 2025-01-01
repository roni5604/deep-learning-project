import os
import glob
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score

def load_labels(data_dir):
    """
    Loads labels (subfolder names) for every image found in `data_dir`.
    """
    labels = []
    if not os.path.exists(data_dir):
        print(f"[ERROR] Directory not found: {data_dir}")
        return labels

    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    for c in classes:
        class_dir = os.path.join(data_dir, c)
        image_files = glob.glob(os.path.join(class_dir, "*"))
        for _ in image_files:
            labels.append(c)
    return labels

def main():
    print("\n=== Baseline Model: Majority Class ===")
    train_dir = "data/face-expression-recognition-dataset/images/images/new_train"
    val_dir   = "data/face-expression-recognition-dataset/images/images/new_validation"

    print(f"Loading training labels from: {train_dir}")
    train_labels = load_labels(train_dir)
    print(f"Loading validation labels from: {val_dir}")
    val_labels   = load_labels(val_dir)

    if not train_labels or not val_labels:
        print("[ERROR] Could not load labels. Exiting baseline.")
        return

    # Determine majority class in train set
    counter = Counter(train_labels)
    majority_class, majority_count = counter.most_common(1)[0]

    print(f"Most common class in training set: '{majority_class}' "
          f"(appears {majority_count} times).")

    # Predict majority class for all validation images
    predictions = [majority_class] * len(val_labels)

    # Calculate metrics
    accuracy  = accuracy_score(val_labels, predictions)
    precision = precision_score(val_labels, predictions, average='macro', zero_division=0)
    recall    = recall_score(val_labels, predictions, average='macro', zero_division=0)

    print(f"\nBaseline Metrics on Validation Set:")
    print(f" - Accuracy:  {accuracy:.4f}")
    print(f" - Precision: {precision:.4f}")
    print(f" - Recall:    {recall:.4f}")

if __name__ == "__main__":
    main()
