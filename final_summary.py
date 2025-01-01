# final_summary.py

def main():
    """
    Summarizes final metrics in a neat table for:
    Baseline, Logistic Regression, Linear Regression, Deeper MLP, Advanced CNN.
    Adjust values below to match your final outputs.
    """
    print("\n=== Final Summary of All Models ===\n")

    # Hardcode final metrics from your logs:
    baseline =  {"Accuracy": 0.2504, "Precision": 0.0358, "Recall": 0.1429}
    logistic =  {"Accuracy": 0.3518, "Precision": 0.3286, "Recall": 0.3032}
    linear   =  {"Accuracy": 0.2207, "Precision": 0.2207, "Recall": 0.1542}
    mlp      =  {"Accuracy": 0.4548, "Precision": 0.4717, "Recall": 0.4163}
    cnn      =  {"Accuracy": 0.5597, "Precision": 0.5698, "Recall": 0.4804}

    # Print table
    print(" Model              | Accuracy  | Precision | Recall   ")
    print("--------------------|----------|-----------|----------")
    print(f" Baseline           | {baseline['Accuracy']:.4f}   | {baseline['Precision']:.4f}    | {baseline['Recall']:.4f}")
    print(f" Logistic Regr.     | {logistic['Accuracy']:.4f}   | {logistic['Precision']:.4f}    | {logistic['Recall']:.4f}")
    print(f" Linear Regr.       | {linear['Accuracy']:.4f}   | {linear['Precision']:.4f}    | {linear['Recall']:.4f}")
    print(f" Deeper MLP         | {mlp['Accuracy']:.4f}   | {mlp['Precision']:.4f}    | {mlp['Recall']:.4f}")
    print(f" Advanced CNN       | {cnn['Accuracy']:.4f}   | {cnn['Precision']:.4f}    | {cnn['Recall']:.4f}")

    print("\n=== Observations ===")
    print("1) Baseline: Minimal performance by predicting majority class.")
    print("2) Logistic: Improved from baseline, still limited by linear boundaries.")
    print("3) Linear Regr.: Not ideal for classification, as expected.")
    print("4) Deeper MLP: Gains from non-linear layers and dropout.")
    print("5) Advanced CNN: Best performance with convolution layers, augmentation, and dropout.\n")

if __name__ == "__main__":
    main()
