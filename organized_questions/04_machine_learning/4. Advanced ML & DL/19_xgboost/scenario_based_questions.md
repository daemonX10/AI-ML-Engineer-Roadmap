# XGBoost Interview Questions - Scenario-Based Questions

## Question 1
- [ ] Done

**Suppose you have a dataset with a mixture of categorical and continuous features. How would you preprocess the data before training an XGBoost model?**

### Answer

**Definition:**
Mixed data requires encoding categoricals (since XGBoost doesn't natively handle them in most versions) while preserving continuous features, with appropriate handling based on cardinality.

**Preprocessing Pipeline:**

```python
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Sample data
df = pd.DataFrame({
    'age': [25, 35, 45, 55],
    'income': [50000, 75000, 100000, 150000],
    'city': ['NYC', 'LA', 'Chicago', 'NYC'],
    'education': ['High School', 'Bachelor', 'Master', 'PhD'],
    'target': [0, 1, 1, 0]
})

# Identify column types
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['number']).columns.drop('target').tolist()

print(f"Categorical: {categorical_cols}")
print(f"Numerical: {numerical_cols}")
```

**Strategy by Cardinality:**

```python
def preprocess_mixed_data(df, target_col, cat_cols, num_cols):
    """Preprocess mixed data for XGBoost."""
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Separate by cardinality
    low_cardinality = [c for c in cat_cols if X[c].nunique() < 10]
    high_cardinality = [c for c in cat_cols if X[c].nunique() >= 10]
    
    # Method 1: ColumnTransformer
    preprocessor = ColumnTransformer([
        # One-hot for low cardinality
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False), low_cardinality),
        # Label encoding for high cardinality (or target encoding)
        ('label', 'passthrough', high_cardinality),  # Handle separately
        # Passthrough numerical (XGBoost doesn't need scaling)
        ('num', 'passthrough', num_cols)
    ])
    
    return preprocessor

# Alternative: Manual encoding
def encode_categoricals(df, cat_cols):
    """Encode categorical columns."""
    df_encoded = df.copy()
    
    for col in cat_cols:
        cardinality = df[col].nunique()
        
        if cardinality <= 10:
            # One-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded.drop(col, axis=1, inplace=True)
        else:
            # Label encoding (or target encoding)
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
    
    return df_encoded
```

**Full Pipeline:**

```python
from sklearn.model_selection import train_test_split

# Prepare data
X = df.drop('target', axis=1)
y = df['target']

# Encode
X_encoded = encode_categoricals(X, categorical_cols)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
model.fit(X_train, y_train)
```

**Native Categorical Support (XGBoost 1.5+):**

```python
# Enable native categorical support
X['city'] = X['city'].astype('category')
X['education'] = X['education'].astype('category')

model = xgb.XGBClassifier(
    tree_method='hist',
    enable_categorical=True
)
model.fit(X, y)
```

**Summary:**

| Cardinality | Encoding Method |
|-------------|-----------------|
| Low (< 10) | One-hot encoding |
| Medium (10-100) | Label or target encoding |
| High (100+) | Target encoding |
| Native support | Use `enable_categorical=True` |

---

## Question 2
- [ ] Done

**In a scenario where model interpretability is crucial, how would you justify the use of XGBoost?**

### Answer

**Definition:**
While XGBoost is not as interpretable as linear models, it offers multiple interpretation techniques (feature importance, SHAP, partial dependence) that can provide sufficient transparency for many applications.

**Interpretability Tools:**

**1. Feature Importance:**
```python
import xgboost as xgb
import matplotlib.pyplot as plt

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Plot importance
xgb.plot_importance(model, importance_type='gain', max_num_features=15)
plt.title('Feature Importance (Gain)')
plt.tight_layout()
plt.show()
```

**2. SHAP Values (Gold Standard):**
```python
import shap

# Create explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global interpretation: Which features matter most?
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Local interpretation: Why this prediction?
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# Feature interaction
shap.dependence_plot('feature_name', shap_values, X_test)
```

**3. Partial Dependence Plots:**
```python
from sklearn.inspection import PartialDependenceDisplay

# Show how feature affects prediction
PartialDependenceDisplay.from_estimator(
    model, X_test, features=['feature1', 'feature2'],
    kind='both'  # Shows both average and individual
)
plt.show()
```

**4. Individual Prediction Explanations:**
```python
def explain_prediction(model, X_sample, feature_names):
    """Generate human-readable explanation."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Get top contributing features
    contributions = pd.DataFrame({
        'feature': feature_names,
        'value': X_sample.values[0],
        'contribution': shap_values[0]
    }).sort_values('contribution', key=abs, ascending=False)
    
    print("Top factors for this prediction:")
    for _, row in contributions.head(5).iterrows():
        direction = "increases" if row['contribution'] > 0 else "decreases"
        print(f"  • {row['feature']}={row['value']:.2f} → {direction} probability by {abs(row['contribution']):.3f}")
    
    return contributions

# Example usage
explain_prediction(model, X_test.iloc[[0]], feature_names)
```

**Justification Arguments:**

| Concern | Response |
|---------|----------|
| "It's a black box" | SHAP provides complete explanations at global and local levels |
| "Can't explain to stakeholders" | Feature importance + SHAP force plots are visual and intuitive |
| "Regulatory requirements" | Document top features, monotonicity constraints available |
| "Need to audit decisions" | Every prediction can be decomposed into feature contributions |

**Comparison with Linear Models:**

| Aspect | Linear Model | XGBoost + SHAP |
|--------|--------------|----------------|
| Global importance | Coefficients | SHAP summary |
| Local explanation | Coefficient × value | SHAP values |
| Interactions | Manual feature engineering | Captured automatically |
| Performance | Often lower | Often higher |

**Interview Point:**
"XGBoost with SHAP is my preferred approach when both accuracy and interpretability matter. SHAP provides mathematically grounded explanations that satisfy most regulatory and business requirements."

---

## Question 3
- [ ] Done

**You're tasked with predicting customer churn. How would you go about applying XGBoost to solve this problem?**

### Answer

**Definition:**
Customer churn prediction involves building a binary classifier to identify customers likely to leave. XGBoost is well-suited due to its handling of mixed features, class imbalance, and interpretability via feature importance.

**Complete Workflow:**

```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, roc_auc_score, 
                            precision_recall_curve, confusion_matrix)
import matplotlib.pyplot as plt

# 1. LOAD AND EXPLORE DATA
df = pd.read_csv('customer_churn.csv')
print(f"Shape: {df.shape}")
print(f"Churn rate: {df['churn'].mean():.2%}")

# 2. FEATURE ENGINEERING
def create_churn_features(df):
    """Create features for churn prediction."""
    features = df.copy()
    
    # Tenure features
    features['tenure_months'] = features['tenure']
    features['is_new_customer'] = (features['tenure'] < 6).astype(int)
    
    # Usage features
    features['avg_monthly_charges'] = features['total_charges'] / (features['tenure'] + 1)
    features['charge_ratio'] = features['monthly_charges'] / features['avg_monthly_charges']
    
    # Interaction features
    features['services_count'] = features[['phone', 'internet', 'streaming']].sum(axis=1)
    
    # Engagement score
    features['engagement_score'] = features['support_tickets'] + features['login_frequency']
    
    return features

df = create_churn_features(df)

# 3. PREPARE DATA
feature_cols = [c for c in df.columns if c not in ['customer_id', 'churn']]
X = df[feature_cols]
y = df['churn']

# Handle categoricals
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. HANDLE IMBALANCE
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale pos weight: {scale_pos_weight:.2f}")

# 5. TRAIN MODEL WITH CROSS-VALIDATION
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

model = xgb.XGBClassifier(
    n_estimators=500,
    early_stopping_rounds=50,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    random_state=42
)

# Note: For proper CV with early stopping, use custom loop
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_params = {}  # From GridSearchCV or manual tuning

# 6. FINAL MODEL
final_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=50,
    eval_metric='auc',
    random_state=42
)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=20
)

# 7. EVALUATE
y_proba = final_model.predict_proba(X_test)[:, 1]
y_pred = final_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

# 8. FEATURE IMPORTANCE
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Churn Predictors:")
print(importance_df.head(10))

# 9. FIND OPTIMAL THRESHOLD
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal threshold: {optimal_threshold:.3f}")
print(f"At this threshold - Precision: {precision[optimal_idx]:.3f}, Recall: {recall[optimal_idx]:.3f}")
```

**Key Considerations:**
- Class imbalance: Use scale_pos_weight
- Feature engineering: Create behavioral features
- Threshold tuning: Optimize for business metric (cost of FN vs FP)
- Interpretability: Use SHAP for customer-level explanations

---

## Question 4
- [ ] Done

**Imagine you're developing a recommendation system. Explain how you might utilize XGBoost in this context.**

### Answer

**Definition:**
XGBoost can power recommendation systems by framing recommendations as either a ranking problem (rank items by predicted relevance) or a classification/regression problem (predict ratings or click probability).

**Approaches Using XGBoost:**

**1. Click-Through Rate (CTR) Prediction:**
- Frame as binary classification: will the user click/interact?
- Features: user demographics, item attributes, context (time, device), interaction history
- XGBoost predicts P(click), items ranked by probability

**2. Rating Prediction:**
- Frame as regression: predict user rating for an item
- Use collaborative filtering features + content-based features
- XGBoost learns non-linear feature interactions

**3. Learning to Rank:**
- Use XGBoost's `rank:ndcg` objective
- Group items by user, rank by predicted relevance

**Feature Engineering for Recommendations:**
```python
features = {
    # User features
    'user_age': ...,
    'user_avg_rating': ...,
    'user_activity_count': ...,
    
    # Item features
    'item_popularity': ...,
    'item_avg_rating': ...,
    'item_category': ...,
    
    # Interaction features
    'user_item_category_affinity': ...,
    'time_since_last_interaction': ...,
    'user_item_price_ratio': ...,
}
```

**Advantages of XGBoost for Recommendations:**

| Advantage | Description |
|-----------|-------------|
| **Feature flexibility** | Handles mixed feature types natively |
| **Non-linear interactions** | Captures complex user-item relationships |
| **Fast inference** | Suitable for real-time recommendations |
| **Interpretability** | SHAP explains why items are recommended |
| **Cold-start handling** | Content-based features work for new items |

**Limitations:**
- Doesn't learn embeddings like deep learning models
- Requires manual feature engineering
- Less effective for sequential behavior patterns

**Interview Tip:** In practice, XGBoost is often used as the ranking stage in a two-stage recommendation pipeline: a candidate generation model (e.g., ALS, neural retrieval) selects top-N candidates, then XGBoost re-ranks them using rich features.

---

