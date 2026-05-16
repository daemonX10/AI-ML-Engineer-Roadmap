# Light Gbm Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Implement a basic LightGBM model for a binary classification problem using Python.**

**Answer:**

```python
import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set parameters
params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'verbose': -1
}

# Train model
model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=20)]
)

# Predict probabilities
y_pred_proba = model.predict(X_test)

# Convert to binary predictions
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

**Alternative: Sklearn API**
```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
          callbacks=[lgb.early_stopping(20)])
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
```

---

## Question 2
- [ ] Done

**Write a script to perform grid search hyperparameter tuning for a LightGBM model in Python.**

**Answer:**

```python
import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define parameter grid
param_grid = {
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 7, -1]
}

# Create LightGBM classifier
model = lgb.LGBMClassifier(
    objective='binary',
    verbose=-1
)

# Grid search with cross-validation
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    verbose=1,
    n_jobs=-1
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Results
print("Best Parameters:", grid_search.best_params_)
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
print(f"Test AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

**For Larger Search Space (RandomizedSearchCV):**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_distributions = {
    'num_leaves': randint(10, 150),
    'learning_rate': uniform(0.01, 0.2),
    'n_estimators': randint(50, 500)
}

random_search = RandomizedSearchCV(model, param_distributions, 
                                   n_iter=50, cv=5, scoring='roc_auc')
random_search.fit(X_train, y_train)
```

---

## Question 3
- [ ] Done

**Code a LightGBM regression model with custom evaluation metrics in Python.**

**Answer:**

```python
import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create sample data
X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create datasets
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Custom evaluation metric: Mean Absolute Percentage Error
def mape_metric(preds, data):
    labels = data.get_label()
    # Avoid division by zero
    mask = labels != 0
    mape = np.mean(np.abs((labels[mask] - preds[mask]) / labels[mask])) * 100
    return 'mape', mape, False  # False = lower is better

# Custom metric: Root Mean Squared Log Error
def rmsle_metric(preds, data):
    labels = data.get_label()
    # Handle negative predictions
    preds = np.maximum(preds, 0)
    log_diff = np.log1p(preds) - np.log1p(labels)
    rmsle = np.sqrt(np.mean(log_diff ** 2))
    return 'rmsle', rmsle, False

# Parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',  # Built-in metric
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}

# Train with custom metrics
model = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[valid_data],
    feval=[mape_metric, rmsle_metric],  # Add custom metrics
    callbacks=[lgb.early_stopping(30)]
)

# Predict
predictions = model.predict(X_test)
print(f"RMSE: {np.sqrt(np.mean((y_test - predictions)**2)):.4f}")
```

---

## Question 4
- [ ] Done

**Demonstrate feature importance extraction from a trained LightGBM model.**

**Answer:**

```python
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Create sample data with feature names
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
feature_names = [f'feature_{i}' for i in range(10)]

# Train model
train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
params = {'objective': 'binary', 'num_leaves': 31, 'verbose': -1}
model = lgb.train(params, train_data, num_boost_round=100)

# Method 1: Get importance as array
importance_gain = model.feature_importance(importance_type='gain')
importance_split = model.feature_importance(importance_type='split')

print("Feature Importance (Gain):")
for name, imp in sorted(zip(feature_names, importance_gain), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.2f}")

# Method 2: Plot importance
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

lgb.plot_importance(model, importance_type='gain', ax=axes[0], 
                    title='Feature Importance (Gain)')
lgb.plot_importance(model, importance_type='split', ax=axes[1], 
                    title='Feature Importance (Split)')

plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# Method 3: Get as dictionary
importance_dict = dict(zip(feature_names, importance_gain))
print("\nTop 5 Features:", dict(sorted(importance_dict.items(), 
                                       key=lambda x: -x[1])[:5]))
```

**Importance Types:**
- **gain**: Total gain from splits using this feature (how much it improves predictions)
- **split**: Number of times feature is used in splits (how often it's used)

---

## Question 5
- [ ] Done

**Create a Python function that uses LightGBM for cross-validation on a given dataset.**

**Answer:**

```python
import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

def lightgbm_cv(X, y, params, n_folds=5, num_boost_round=1000, 
                early_stopping_rounds=50, stratified=True):
    """
    Perform cross-validation with LightGBM
    
    Returns:
    - cv_results: dict with mean and std of metrics
    - models: list of trained models
    """
    train_data = lgb.Dataset(X, label=y)
    
    # Using LightGBM's built-in CV
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=num_boost_round,
        nfold=n_folds,
        stratified=stratified,
        callbacks=[lgb.early_stopping(early_stopping_rounds)],
        return_cvbooster=True
    )
    
    return cv_results

# Example usage
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}

# Run CV
results = lightgbm_cv(X, y, params, n_folds=5)

# Get metric name
metric_name = 'valid auc-mean'
print(f"Best CV Score: {max(results[metric_name]):.4f}")
print(f"Best Iteration: {np.argmax(results[metric_name])}")

# Manual CV with sklearn
def manual_lightgbm_cv(X, y, params, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(params, train_data, valid_sets=[val_data],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        
        preds = model.predict(X_val)
        from sklearn.metrics import roc_auc_score
        scores.append(roc_auc_score(y_val, preds))
    
    return np.mean(scores), np.std(scores)
```

---

## Question 6
- [ ] Done

**Optimize a LightGBM model using early stopping criteria with Python's lightgbm package.**

**Answer:**

```python
import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create data
X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Create datasets
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Parameters
params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}

# Train with early stopping using callbacks
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,          # High max iterations
    valid_sets=[valid_data],
    valid_names=['validation'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),  # Stop if no improvement for 50 rounds
        lgb.log_evaluation(period=20)             # Print every 20 iterations
    ]
)

# Best iteration is automatically tracked
print(f"Best iteration: {model.best_iteration}")
print(f"Best score: {model.best_score['validation']['auc']:.4f}")

# Predictions use best iteration by default
predictions = model.predict(X_val)

# Sklearn API with early stopping
from lightgbm import LGBMClassifier

model_sklearn = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31
)

model_sklearn.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(20)]
)

print(f"Best iteration (sklearn): {model_sklearn.best_iteration_}")
```

---

## Question 7
- [ ] Done

**Implement a multi-class classification using LightGBM and evaluate it using the appropriate metrics.**

**Answer:**

```python
import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, log_loss)

# Create multi-class data
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=15, n_classes=4, 
                           n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create datasets
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test)

# Parameters for multi-class
params = {
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}

# Train model
model = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(30)]
)

# Predict probabilities (shape: n_samples x n_classes)
y_pred_proba = model.predict(X_test)

# Get class predictions
y_pred = np.argmax(y_pred_proba, axis=1)

# Evaluation
print("=" * 50)
print("Multi-Class Classification Evaluation")
print("=" * 50)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Log Loss: {log_loss(y_test, y_pred_proba):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

**Key Metrics for Multi-Class:**
- **Accuracy**: Overall correct predictions
- **Log Loss**: Penalizes confident wrong predictions
- **Per-class Precision/Recall**: Performance per category
- **Macro/Micro F1**: Aggregated F1 scores

---

