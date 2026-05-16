# Decision Tree Interview Questions - Scenario_Based Questions

## Question 1
- [ ] Done

**How would you approach a real-world problem requiring a Decision Tree model?**

**Answer:**

A systematic approach ensures robust model development from problem understanding to deployment. Here's a practical end-to-end workflow.

**Step 1: Problem Understanding**
- Define objective: Classification or Regression?
- Identify success metrics: Accuracy, F1, RMSE, business KPIs
- Understand constraints: Interpretability requirements, latency limits

**Step 2: Data Exploration & Preparation**

```python
import pandas as pd
import numpy as np

# Load and explore
df = pd.read_csv('data.csv')
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Handle missing values
df.fillna(df.median(), inplace=True)  # or use imputer

# Encode categoricals (required for sklearn trees)
from sklearn.preprocessing import LabelEncoder
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

**Step 3: Baseline Model**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Simple baseline
baseline = DecisionTreeClassifier(max_depth=5, random_state=42)
baseline.fit(X_train, y_train)
print(classification_report(y_test, baseline.predict(X_test)))
```

**Step 4: Hyperparameter Tuning**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'ccp_alpha': [0, 0.001, 0.01, 0.1]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid, cv=5, scoring='f1', n_jobs=-1
)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
```

**Step 5: Model Evaluation**

```python
from sklearn.metrics import confusion_matrix, roc_auc_score

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print(f"AUC: {roc_auc_score(y_test, y_prob):.3f}")
```

**Step 6: Interpret & Validate**

```python
from sklearn.tree import plot_tree, export_text

# Visualize
plot_tree(best_model, feature_names=feature_names, filled=True)

# Feature importance
for name, imp in sorted(zip(feature_names, best_model.feature_importances_), 
                        key=lambda x: -x[1]):
    print(f"{name}: {imp:.3f}")
```

**Step 7: Consider Ensemble (if needed)**

If single tree performance insufficient:
- Random Forest for variance reduction
- Gradient Boosting for higher accuracy
- XGBoost/LightGBM for production

**Step 8: Deploy**
- Save model with joblib/pickle
- Create prediction API
- Monitor performance in production

---

## Question 2
- [ ] Done

**Imagine you have a highly imbalanced dataset, how would you fine-tune a Decision Tree to handle it?**

**Answer:**

Imbalanced datasets (e.g., 95% negative, 5% positive) cause Decision Trees to be biased toward the majority class. Several techniques address this at data, algorithm, and evaluation levels.

**Strategies:**

**1. Class Weighting (Algorithm Level):**

```python
from sklearn.tree import DecisionTreeClassifier

# Option A: Balanced weights (inverse of class frequency)
clf = DecisionTreeClassifier(class_weight='balanced')

# Option B: Custom weights
clf = DecisionTreeClassifier(class_weight={0: 1, 1: 10})  # 10x weight for minority

clf.fit(X_train, y_train)
```

**2. Resampling (Data Level):**

```python
# Oversampling minority class (SMOTE)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Undersampling majority class
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# Combination: SMOTE + Tomek Links
from imblearn.combine import SMOTETomek
smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
```

**3. Threshold Adjustment:**

```python
# Default threshold is 0.5, adjust for imbalanced data
y_prob = clf.predict_proba(X_test)[:, 1]

# Lower threshold to catch more positives
threshold = 0.3
y_pred = (y_prob >= threshold).astype(int)
```

**4. Appropriate Evaluation Metrics:**

```python
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)

# DON'T use accuracy for imbalanced data!
# Instead use:
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1: {f1_score(y_test, y_pred):.3f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")
print(f"AUC-PR: {average_precision_score(y_test, y_prob):.3f}")
```

**5. Hyperparameter Tuning with Appropriate Scoring:**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [1, 5, 10],
    'class_weight': ['balanced', {0: 1, 1: 5}, {0: 1, 1: 10}]
}

# Use F1 or AUC for scoring, NOT accuracy
grid = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid,
    cv=5,
    scoring='f1'  # or 'roc_auc'
)
```

**6. Stratified Sampling:**

```python
from sklearn.model_selection import StratifiedKFold

# Ensures class ratio preserved in each fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Summary Table:**

| Technique | When to Use |
|-----------|-------------|
| class_weight='balanced' | First thing to try, simple |
| SMOTE | Moderate imbalance, enough minority samples |
| Undersampling | Very large dataset, willing to lose data |
| Threshold tuning | Need to control precision-recall tradeoff |
| AUC-PR metric | Very severe imbalance (< 5% minority) |

---

## Question 3
- [ ] Done

**Discuss how you would apply a Decision Tree for a time-series prediction problem.**

**Answer:**

Decision Trees are not designed for time-series data but can be adapted through feature engineering. The key is transforming temporal data into tabular format with lag features, rolling statistics, and time-based features.

**Why Decision Trees Struggle with Time-Series:**
- No inherent notion of sequence/order
- Cannot extrapolate beyond training range
- Ignores temporal dependencies
- Makes axis-parallel predictions (step functions)

**Adaptation Approach:**

**Step 1: Create Lag Features**

```python
import pandas as pd

def create_lag_features(df, target_col, lags):
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

# Example
df = create_lag_features(df, 'sales', lags=[1, 7, 14, 30])
# sales_lag_1: yesterday's sales
# sales_lag_7: sales 1 week ago
```

**Step 2: Create Rolling Statistics**

```python
def create_rolling_features(df, target_col, windows):
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
    return df

df = create_rolling_features(df, 'sales', windows=[7, 14, 30])
```

**Step 3: Extract Time-Based Features**

```python
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_holiday'] = df['date'].isin(holiday_dates).astype(int)
```

**Step 4: Train-Test Split (Temporal)**

```python
# IMPORTANT: Time-based split, not random!
train_end = '2023-06-30'
X_train = df[df['date'] <= train_end].drop(['date', 'target'], axis=1)
X_test = df[df['date'] > train_end].drop(['date', 'target'], axis=1)
y_train = df[df['date'] <= train_end]['target']
y_test = df[df['date'] > train_end]['target']
```

**Step 5: Model Training**

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Drop rows with NaN from lag features
X_train = X_train.dropna()

# Use Random Forest for better time-series performance
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
```

**Better Alternatives:**
| Method | When to Use |
|--------|-------------|
| ARIMA/SARIMA | Strong seasonality, linear patterns |
| Prophet | Business forecasting, holidays |
| LSTM/RNN | Long dependencies, complex patterns |
| XGBoost | With lag features, strong baseline |
| Decision Tree | Simple baseline, interpretability needed |

**Key Considerations:**
- Use TimeSeriesSplit for cross-validation
- Don't shuffle data (preserves order)
- Include sufficient lag depth
- Monitor for concept drift

---
