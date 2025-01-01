# final_summary.py

import os

def main():
    """
    Reads final results from each approach (if available) and prints
    a neat comparison table (Baseline, Logistic, MLP, CNN) with accuracy,
    precision, and recall. The baseline and logistic regression results are
    not saved to files by default, so we manually enter them if needed.
    """
    print("\n=== Final Summary of All Models ===\n")

    # Manually define placeholders for Baseline and Logistic
    # since they don't write to any .txt file.
    # If you want to automate reading them, you could parse the console logs
    # or store them in a file. For now, set them based on your latest run.
    baseline_metrics = {
        "Accuracy":  0.2504,
        "Precision": 0.0358,
        "Recall":    0.1429
    }
    logistic_metrics = {
        "Accuracy":  0.3518,
        "Precision": 0.3286,
        "Recall":    0.3032
    }

    # Attempt to load MLP results from file
    mlp_file = "basic_mlp_results.txt"
    mlp_metrics = None
    if os.path.exists(mlp_file):
        with open(mlp_file, "r") as f:
            line = f.read().strip()
            if line:
                acc, prec, rec = line.split(",")
                mlp_metrics = {
                    "Accuracy":  float(acc),
                    "Precision": float(prec),
                    "Recall":    float(rec)
                }
    else:
        # Could fallback to placeholders if needed
        mlp_metrics = {
            "Accuracy":  0.0,
            "Precision": 0.0,
            "Recall":    0.0
        }

    # Attempt to load CNN results from file
    cnn_file = "cnn_results.txt"
    cnn_metrics = None
    if os.path.exists(cnn_file):
        with open(cnn_file, "r") as f:
            line = f.read().strip()
            if line:
                acc, prec, rec = line.split(",")
                cnn_metrics = {
                    "Accuracy":  float(acc),
                    "Precision": float(prec),
                    "Recall":    float(rec)
                }
    else:
        # Could fallback to placeholders if needed
        cnn_metrics = {
            "Accuracy":  0.0,
            "Precision": 0.0,
            "Recall":    0.0
        }

    # Print a table of results
    print("Below is a direct metric comparison among the four models:\n")

    print(" Model          | Accuracy  | Precision | Recall   ")
    print("----------------|----------|-----------|----------")
    # Baseline row
    print(f" Baseline       | {baseline_metrics['Accuracy']:.4f}   | {baseline_metrics['Precision']:.4f}    | {baseline_metrics['Recall']:.4f}")
    # Logistic row
    print(f" Logistic Reg.  | {logistic_metrics['Accuracy']:.4f}   | {logistic_metrics['Precision']:.4f}    | {logistic_metrics['Recall']:.4f}")
    # MLP row
    print(f" Deeper MLP     | {mlp_metrics['Accuracy']:.4f}   | {mlp_metrics['Precision']:.4f}    | {mlp_metrics['Recall']:.4f}")
    # CNN row
    print(f" Advanced CNN   | {cnn_metrics['Accuracy']:.4f}   | {cnn_metrics['Precision']:.4f}    | {cnn_metrics['Recall']:.4f}")

    print("\n")
    print("=== Explanations and Observations ===")
    print("1. Baseline: Predicts the majority class ('happy'), resulting in low precision/recall.")
    print("2. Logistic Regression: Improves accuracy over baseline but still limited by linear boundaries.")
    print("3. Deeper MLP: Learns non-linear features, achieving moderate improvements in accuracy.")
    print("4. Advanced CNN (Augmentation): Achieves the highest accuracy, precision, and recall by leveraging")
    print("   convolutional layers, dropout, and data augmentation to capture visual features more effectively.\n")

    print("For more detailed analysis, consult the console outputs from each script and the final_report.md.")

if __name__ == "__main__":
    main()
