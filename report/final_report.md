
# Final Report

## 1. Introduction
This project classifies facial expressions (angry, disgust, fear, happy, neutral, sad, surprise) from grayscale images. We implemented five methods:

1. **Baseline** (majority-class prediction)
2. **Logistic Regression** (multi-class softmax)
3. **Linear Regression** (experimental classification)
4. **Deeper MLP** with dropout
5. **Advanced CNN** with data augmentation

## 2. Dataset Description
- **Source**: [Kaggle Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset).
- **Splits**: 70% train, 15% validation, 15% test (re-split via `prepare_dataset.py`).
- **Features**: 48×48 pixel grayscale images, numeric intensities in `[0,255]`.
- **Labels**: 7 emotion classes.

## 3. Baseline Execution
- **Method**: Always predict “happy” (the majority class).
- **Metrics** (Validation):
  - Accuracy: 0.2504
  - Precision: 0.0358
  - Recall: 0.1429

## 4. Logistic Regression Execution
- **Model**: Multinomial logistic regression with `lbfgs`.
- **Improvements**:
  - Feature scaling (`StandardScaler`).
  - `max_iter=2000`.
- **Metrics** (Validation):
  - Accuracy: 0.3518
  - Precision: 0.3286
  - Recall: 0.3032

## 5. Linear Regression Execution
- **Model**: LinearRegression from scikit-learn, rounding continuous outputs.
- **Note**: Not typical for classification.  
- **Metrics** (Validation):
  - Accuracy: 0.2207
  - Precision: 0.2207
  - Recall: 0.1542

## 6. Deeper MLP Execution
- **Architecture**:
  - Two hidden layers (256→128) with ReLU + dropout (p=0.3).
  - Standard scaling of pixel features.
  - Adam optimizer, optional StepLR scheduler.
- **Metrics** (Validation):
  - Accuracy: 0.4548
  - Precision: 0.4717
  - Recall: 0.4163

## 7. Advanced CNN Execution
- **Architecture**:
  - 3 convolutional blocks (32→64→128), each with max pooling + dropout2d.
  - 2 fully connected layers (fc1=256, fc2=7).
  - Data augmentation (random flips, rotation).
  - Weight decay (1e-4), Adam, StepLR for LR scheduling.
- **Metrics** (Validation):
  - Accuracy: 0.5597
  - Precision: 0.5698
  - Recall: 0.4804

## 8. Comparison
| Model            | Accuracy | Precision | Recall  |
|------------------|---------:|----------:|--------:|
| Baseline         | 0.2504   | 0.0358    | 0.1429  |
| Logistic         | 0.3518   | 0.3286    | 0.3032  |
| Linear Regress.  | 0.2207   | 0.2207    | 0.1542  |
| Deeper MLP       | 0.4548   | 0.4717    | 0.4163  |
| Advanced CNN     | 0.5597   | 0.5698    | 0.4804  |

**Observations**:
- Baseline is simplest, minimal performance.
- Logistic Regression improves significantly.
- Linear Regression is sub-optimal for classification.
- Deeper MLP further improves performance with non-linear layers.
- Advanced CNN yields the best accuracy, precision, and recall, benefiting from convolutional feature extraction and augmentation.

## 9. Improvements & Future Work
1. **Additional Epochs**: The MLP and CNN may improve further with more training epochs.
2. **Hyperparameter Tuning**: Adjust dropout rates, hidden sizes, learning rates.
3. **Data Augmentation**: More variety (RandomCrop, brightness changes).
4. **Regularization**: Weight decay, advanced schedulers, or early stopping.
5. **Larger CNN**: Additional conv layers might push performance higher.

## 10. Conclusion
This project shows progressive improvements across five methods:
- Baseline → Logistic → Linear → MLP → CNN
with the **Advanced CNN** achieving ~55.97% accuracy on validation. Further tuning and architectural changes can lead to even better results.


