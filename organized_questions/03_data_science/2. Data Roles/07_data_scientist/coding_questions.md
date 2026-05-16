# Data Scientist Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Describe the concept of Gradient Boosting and its popular implementations.**

**Answer:**

### Definition
Gradient Boosting builds an ensemble of weak learners (typically decision trees) sequentially, where each new tree corrects the errors of the previous ones by fitting to the residuals.

### Algorithm Steps
1. Initialize with a constant prediction (mean for regression)
2. Compute residuals (pseudo-residuals)
3. Fit a tree to the residuals
4. Add tree to ensemble with learning rate
5. Repeat until convergence

### Mathematical Formulation
$$F_{m}(x) = F_{m-1}(x) + \gamma_m h_m(x)$$

Where:
- $F_m(x)$ = prediction at step m
- $\gamma_m$ = learning rate
- $h_m(x)$ = weak learner

### Popular Implementations

| Library | Key Features |
|---------|--------------|
| XGBoost | Regularization, parallel processing, missing values |
| LightGBM | Leaf-wise growth, histogram-based, faster |
| CatBoost | Categorical features, ordered boosting |

### Python Code Example
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Gradient Boosting model
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# XGBoost example
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
xgb_model.fit(X_train, y_train)
```

### Key Hyperparameters
- **n_estimators**: Number of boosting rounds
- **learning_rate**: Shrinkage to prevent overfitting
- **max_depth**: Tree depth (controls complexity)
- **subsample**: Row sampling ratio
- **colsample_bytree**: Feature sampling ratio

### Interview Tip
Know the difference between bagging (Random Forest) and boosting (Gradient Boosting):
- Bagging: Parallel, reduces variance
- Boosting: Sequential, reduces bias

---

