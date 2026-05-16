# Model Evaluation Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Implement a Python function that calculates the F1-score given precision and recall values.**

**Answer:**

```python
def calculate_f1(precision, recall):
    """Calculate F1-score from precision and recall."""
    if precision + recall == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# Example usage
precision = 0.8
recall = 0.7
f1 = calculate_f1(precision, recall)
print(f"F1-score: {f1:.3f}")  # Output: 0.747
```

### Interview Tip
F1 is the harmonic mean, which penalizes extreme values.

---

## Question 2
- [ ] Done

**Write a Python script to compute the Confusion Matrix for a two-class problem.**

**Answer:**

```python
def confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix manually."""
    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    tn = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    return [[tn, fp], [fn, tp]]

# Using sklearn
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)
```

### Interview Tip
Know the layout: [[TN, FP], [FN, TP]].

---

## Question 3
- [ ] Done

**Develop a Python function to perform k-fold cross-validation on a dataset.**

**Answer:**

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def kfold_cv(X, y, model, k=5):
    """Perform k-fold cross-validation."""
    scores = cross_val_score(model, X, y, cv=k, scoring='accuracy')
    print(f"Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")
    return scores

# Example
model = RandomForestClassifier()
scores = kfold_cv(X, y, model, k=5)
```

### Interview Tip
Use stratified K-fold for classification.

---

## Question 4
- [ ] Done

**Simulate overfitting in a machine learning model, and show how to detect it with a validation curve.**

**Answer:**

```python
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_validation_curve(X, y, model, param_name, param_range):
    """Plot validation curve to detect overfitting."""
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name,
        param_range=param_range, cv=5
    )
    
    plt.plot(param_range, train_scores.mean(axis=1), label='Train')
    plt.plot(param_range, val_scores.mean(axis=1), label='Validation')
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.legend()
    plt.show()

# Overfitting: Gap between train and validation
```

### Interview Tip
Large gap between curves indicates overfitting.

---

## Question 5
- [ ] Done

**Write code to draw an ROC curve and calculate AUC for a given set of predictions and true labels.**

**Answer:**

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc(y_true, y_proba):
    """Plot ROC curve and calculate AUC."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    return roc_auc

# Usage with probabilities
y_proba = model.predict_proba(X_test)[:, 1]
auc_score = plot_roc(y_test, y_proba)
```

### Interview Tip
Use predict_proba, not predict, for ROC curves.

---

## Question 6
- [ ] Done

**Code a Python function that uses StratifiedKFold cross-validation on an imbalanced dataset.**

**Answer:**

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

def stratified_cv(X, y, model, k=5):
    """Perform stratified k-fold CV for imbalanced data."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
    print(f"Mean F1: {scores.mean():.3f}")
    return scores

# Stratified preserves class distribution in each fold
```

### Interview Tip
Always use stratified for imbalanced classification.

---

## Question 7
- [ ] Done

**Implement a Python program to plot learning curves for a given estimator.**

**Answer:**

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(model, X, y):
    """Plot learning curves to diagnose model."""
    sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    plt.plot(sizes, train_scores.mean(axis=1), label='Train')
    plt.plot(sizes, val_scores.mean(axis=1), label='Validation')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

# Converging curves = good; gap = overfitting
```

### Interview Tip
Flat validation curve suggests more data won't help.

---

## Question 8
- [ ] Done

**Simulate and evaluate model performance with Monte Carlo cross-validation using Python.**

**Answer:**

```python
from sklearn.model_selection import ShuffleSplit, cross_val_score

def monte_carlo_cv(X, y, model, n_splits=10, test_size=0.2):
    """Monte Carlo (repeated random) cross-validation."""
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    scores = cross_val_score(model, X, y, cv=ss)
    print(f"Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")
    return scores

# More robust than single train-test split
```

### Interview Tip
Monte Carlo CV is good for variance estimation.

---

## Question 9
- [ ] Done

**Create a Python function to calculate specificity and sensitivity from a given confusion matrix.**

**Answer:**

```python
def calc_sensitivity_specificity(cm):
    """Calculate sensitivity (recall) and specificity."""
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn)  # True Positive Rate
    specificity = tn / (tn + fp)  # True Negative Rate
    
    return sensitivity, specificity

# Example
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
sens, spec = calc_sensitivity_specificity(cm)
print(f"Sensitivity: {sens:.3f}, Specificity: {spec:.3f}")
```

### Interview Tip
Sensitivity = Recall; Specificity = 1 - FPR.

---

## Question 10
- [ ] Done

**Provide a Python script to compare two models using t-tests and report statistical significance.**

**Answer:**

```python
from scipy import stats
from sklearn.model_selection import cross_val_score

def compare_models(model1, model2, X, y, k=10):
    """Compare two models using paired t-test."""
    scores1 = cross_val_score(model1, X, y, cv=k)
    scores2 = cross_val_score(model2, X, y, cv=k)
    
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    
    print(f"Model 1: {scores1.mean():.3f} ± {scores1.std():.3f}")
    print(f"Model 2: {scores2.mean():.3f} ± {scores2.std():.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Significant: {p_value < 0.05}")
    
    return p_value

# Use paired t-test on same CV folds
```

### Interview Tip
Use paired test since same data is used for both models.

---

