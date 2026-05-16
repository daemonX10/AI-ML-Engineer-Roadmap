# Data Analyst Interview Questions - General Questions

## Question 1
- [ ] Done

**How do you handle categorical variables in machine learning?**

**Answer:**

### Definition
Categorical variables are non-numeric data representing categories or labels. Handling them involves converting them into numerical format that ML algorithms can process, using encoding techniques based on the variable type (nominal vs ordinal) and model requirements.

### Core Concepts
- **Nominal Variables**: No inherent order (e.g., color: red, blue, green)
- **Ordinal Variables**: Natural order exists (e.g., rating: low, medium, high)
- **Cardinality**: Number of unique categories affects encoding choice
- **Label Encoding**: Assigns integers (0, 1, 2...) to categories
- **One-Hot Encoding**: Creates binary columns for each category
- **Target Encoding**: Replaces category with mean of target variable

### Encoding Techniques Summary
| Technique | Use Case | Pros | Cons |
|-----------|----------|------|------|
| Label Encoding | Ordinal data, Tree models | Simple, memory efficient | Implies false ordering |
| One-Hot Encoding | Nominal data, Linear models | No ordering assumption | High dimensionality |
| Target Encoding | High cardinality | Reduces dimensions | Risk of data leakage |
| Frequency Encoding | When count matters | Simple | Collision for same frequency |

### Python Code Example
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Sample data
df = pd.DataFrame({'color': ['red', 'blue', 'green', 'red', 'blue']})

# 1. Label Encoding (for ordinal or tree-based models)
le = LabelEncoder()
df['color_label'] = le.fit_transform(df['color'])
# Output: red=2, blue=0, green=1

# 2. One-Hot Encoding (for nominal data)
df_onehot = pd.get_dummies(df['color'], prefix='color')
# Output: color_blue, color_green, color_red columns with 0/1 values

# 3. Target Encoding (for high cardinality)
# Replace category with mean of target
target = [100, 200, 150, 120, 180]
df['target'] = target
df['color_target_enc'] = df.groupby('color')['target'].transform('mean')
```

### Practical Relevance
- **Tree-based models** (Random Forest, XGBoost): Can use Label Encoding directly
- **Linear models** (Logistic Regression, SVM): Require One-Hot Encoding
- **High cardinality features**: Use Target Encoding or Feature Hashing
- **Neural Networks**: Embedding layers for categorical features

### Interview Tips
- Always ask about cardinality before choosing encoding method
- One-Hot creates multicollinearity - drop one column for linear models
- Target Encoding must be computed on training data only to avoid leakage
- For tree-based models, label encoding is often sufficient and faster

---

## Question 2
- [ ] Done

**How do you evaluate the performance of a regression model?**

**Answer:**

### Definition
Regression model evaluation involves measuring how well the model's predictions match actual continuous values using metrics that quantify prediction error. Key metrics include MAE, MSE, RMSE, and R² score, each providing different insights into model performance.

### Core Concepts
- **Residual**: Difference between actual and predicted value $(y_i - \hat{y}_i)$
- **Scale-dependent metrics**: MAE, MSE, RMSE (in original units)
- **Scale-independent metrics**: R², Adjusted R², MAPE (percentage/ratio)
- **Outlier sensitivity**: MSE/RMSE penalize large errors more than MAE

### Mathematical Formulation

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MAE | $\frac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|$ | Average absolute error |
| MSE | $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ | Average squared error |
| RMSE | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | Root of MSE (same units as y) |
| R² | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Proportion of variance explained |

### Intuition
- **MAE**: Robust to outliers, gives equal weight to all errors
- **RMSE**: Penalizes large errors heavily due to squaring
- **R²**: 1 = perfect fit, 0 = model predicts mean, <0 = worse than mean
- **Adjusted R²**: Penalizes adding irrelevant features

### Python Code Example
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Actual vs Predicted values
y_true = [3, 5, 2.5, 7, 4]
y_pred = [2.8, 5.2, 2.1, 7.5, 3.8]

# Calculate metrics
mae = mean_absolute_error(y_true, y_pred)      # Average error magnitude
mse = mean_squared_error(y_true, y_pred)        # Squared error (penalizes outliers)
rmse = np.sqrt(mse)                             # Same units as target
r2 = r2_score(y_true, y_pred)                   # Variance explained (0-1)

print(f"MAE: {mae:.3f}")   # ~0.34
print(f"RMSE: {rmse:.3f}") # ~0.39
print(f"R²: {r2:.3f}")     # ~0.96
```

### Practical Relevance
- **MAE**: Use when outliers should not dominate (e.g., delivery time prediction)
- **RMSE**: Use when large errors are particularly undesirable (e.g., financial forecasting)
- **R²**: Use for comparing models on same dataset
- **MAPE**: Use when relative error matters (e.g., sales forecasting)

### Interview Tips
- R² can be negative if model is worse than predicting mean
- RMSE is always >= MAE; larger gap indicates more outliers
- Always use cross-validation for reliable metric estimates
- For business context, translate metrics to business impact

---

## Question 3
- [ ] Done

**How do you handle imbalanced datasets in classification problems?**

**Answer:**

### Definition
Imbalanced datasets occur when class distribution is skewed (e.g., 95% class A, 5% class B). Handling involves resampling techniques, algorithm modifications, or evaluation metric changes to ensure the model learns minority class patterns effectively.

### Core Concepts
- **Imbalance Ratio**: Ratio of majority to minority class samples
- **Resampling**: Modify training data distribution
- **Cost-sensitive Learning**: Assign higher penalty to minority class errors
- **Threshold Tuning**: Adjust decision boundary for prediction

### Techniques Overview

| Category | Technique | Description |
|----------|-----------|-------------|
| Undersampling | Random | Remove majority class samples |
| Oversampling | Random | Duplicate minority class samples |
| Oversampling | SMOTE | Generate synthetic minority samples |
| Algorithm-level | Class Weights | Penalize misclassification of minority |
| Ensemble | BalancedRandomForest | Combines undersampling with bagging |

### Algorithm: SMOTE (Synthetic Minority Oversampling)
1. Select a minority sample
2. Find its k-nearest neighbors (k=5 typically)
3. Randomly select one neighbor
4. Create synthetic sample along the line between them
5. Repeat until desired balance achieved

### Python Code Example
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Create imbalanced dataset (90:10 ratio)
X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Method 1: SMOTE oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Method 2: Class weights in model
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate with appropriate metrics
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))  # Check precision, recall, F1
```

### Practical Relevance
- **Fraud Detection**: 0.1% fraud vs 99.9% legitimate
- **Disease Diagnosis**: Rare diseases
- **Churn Prediction**: Small percentage of customers churn
- **Anomaly Detection**: Very few anomalies

### Evaluation Metrics for Imbalanced Data
- **Precision-Recall Curve**: Better than ROC for imbalanced data
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-PR**: Area under precision-recall curve
- **Avoid accuracy**: Can be misleading (99% accuracy but 0% recall on minority)

### Interview Tips
- SMOTE should only be applied to training data, never test data
- Class weights is often simpler and effective as a first approach
- Undersampling may lose important information
- Combine multiple techniques: SMOTE + Tomek Links
- Always evaluate using class-specific metrics (recall for minority class)

---

