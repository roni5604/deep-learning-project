
# Final Report

## 1. Goal and Dataset Description

- **Goal**: Classify facial expressions (angry, disgust, fear, happy, neutral, sad, surprise).
- **Dataset**: Kaggle (jonathanoheix/face-expression-recognition-dataset).
- **Features**: Grayscale pixels, resized to 48×48.
- **Label**: Emotion class.
- **Problem Type**: Multi-class classification.
- **Split**: 70% train, 15% validation, 15% test.

---

## 2. Baseline Execution

- **Method**: Predict the majority class (happy).
- **Validation Metrics**:
  - Accuracy: 0.2504
  - Precision: 0.0358
  - Recall: 0.1429
- **Interpretation**: Low overall metrics, but a starting reference.

---

## 3. Logistic Regression Execution

1. **Flattened images** → 2304 features.
2. **Softmax** for multi-class.
3. **Validation Metrics**:
   - Accuracy: 0.3518
   - Precision: 0.3286
   - Recall: 0.3032
4. **Comparison**: Improved over baseline. Convergence warning suggests more tuning (max_iter, normalization) is possible.

---

## 4. Deeper MLP Execution

1. **Architecture**:
   - Two hidden layers: 256 and 128 units, ReLU activations.
   - Dropout (p=0.3).
   - Final softmax output (7 classes).
2. **Hyperparameters**:
   - 15 epochs, lr=0.001, batch_size=64.
3. **Validation Metrics**:
   - Accuracy: 0.3766
   - Precision: 0.2933
   - Recall: 0.3069
4. **Analysis**: 
   - Outperforms baseline/logistic. 
   - Non-linear layers + dropout help, but limited improvement indicates more advanced architectures may be needed.

---

## 5. Advanced CNN Execution

1. **Architecture**:
   - Two convolution blocks (Conv+ReLU+Pool).
   - Dropout on feature maps.
   - Fully connected layers (256 → 128 → final 7).
2. **Data Augmentation**:
   - Random horizontal flips (p=0.5).
   - Random rotation ±10°.
   - Applies only during training.
3. **Validation Metrics**:
   - Accuracy: 0.5437
   - Precision: 0.5883
   - Recall: 0.4619
4. **Comparison**: 
   - Significantly higher accuracy and precision than MLP.
   - Suggests CNN is better at extracting meaningful image features.

---

## 6. Improvements and Corrections

### Initial Attempts:
- Simple MLP or logistic regression lead to 35%–37% accuracy range.

### Overfitting / Underfitting:
- CNN with no dropout or augmentation → overfitting risk.
- The final CNN uses dropout + random transforms, improving generalization.

### Hyperparameters:
- 15 epochs, Adam optimizer (lr=0.001).
- Could further adjust hidden layers, dropout probability, learning rate scheduling.

---

## 7. Conclusion

- **Baseline**: 25% accuracy, naive approach.
- **Logistic Regression**: ~35% accuracy.
- **Deeper MLP**: ~37.66% accuracy.
- **CNN** (Augment): ~54.37% accuracy.

This progression shows how advanced architectures and data augmentation can significantly increase performance on image classification tasks.

**Next Steps** might include:
- Even deeper CNN or specialized architectures (ResNet, VGG).
- More extensive data augmentation.
- Hyperparameter tuning or early stopping.

---

## 8. References

- **Kaggle Dataset**: [jonathanoheix/face-expression-recognition-dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
- **PyTorch**: [https://pytorch.org](https://pytorch.org)
- **scikit-learn**: [https://scikit-learn.org](https://scikit-learn.org)
- **Deep Learning Book** by Ian Goodfellow, Yoshua Bengio, Aaron Courville.


