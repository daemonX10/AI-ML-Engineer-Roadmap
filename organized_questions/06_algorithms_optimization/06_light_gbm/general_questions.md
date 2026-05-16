# Light Gbm Interview Questions - General Questions

## Question 1
- [ ] Done

**How do Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) contribute to LightGBM's performance?**

**Answer:**

**GOSS (Gradient-based One-Side Sampling):**

**Problem:** Large datasets slow down training.
**Solution:** Sample data intelligently based on gradients.

**How It Works:**
1. Keep all instances with large gradients (well-learned = low gradient, poorly-learned = high gradient)
2. Randomly sample from instances with small gradients
3. Multiply sampled instances by weight to compensate

```
Large gradients (top a%): Keep all
Small gradients (remaining): Random sample b%
Weight for sampled: (1-a)/b
```

**Benefits:**
- Reduces data by 70-90%
- Preserves information (large gradient samples are most important for learning)
- Maintains accuracy while speeding training

---

**EFB (Exclusive Feature Bundling):**

**Problem:** High-dimensional sparse features slow training.
**Solution:** Bundle mutually exclusive features together.

**How It Works:**
1. Identify features that rarely have non-zero values simultaneously
2. Bundle them into single feature
3. Use offset to distinguish original features

**Example:**
```
Feature A: [0, 1, 0, 0, 2]
Feature B: [3, 0, 0, 4, 0]
Bundled:   [3, 1, 0, 4, 2]  # Add offset for B
```

**Benefits:**
- Reduces feature count significantly for sparse data
- Especially useful for one-hot encoded features
- O(#features) → O(#bundles)

---

**Combined Impact:**
- 10-20x speedup on large, sparse datasets
- Minimal accuracy loss

---

## Question 2
- [ ] Done

**What preprocessing steps would you recommend when preparing data for LightGBM?**

**Answer:**

**Minimal Preprocessing Required:**
LightGBM handles many issues natively, reducing preprocessing needs.

---

**Recommended Preprocessing:**

| Step | Recommendation |
|------|----------------|
| **Missing values** | Leave as NaN (LightGBM handles natively) |
| **Categorical features** | Convert to `category` dtype |
| **Numerical features** | No scaling needed (tree-based) |
| **Outliers** | Generally robust, but check extreme cases |

---

**What to Do:**

**1. Categorical Features:**
```python
for col in categorical_cols:
    df[col] = df[col].astype('category')
```

**2. Handle Target Variable:**
- Classification: Ensure labels are 0, 1, 2, ...
- Regression: Check for outliers

**3. Remove Identifier Columns:**
```python
df = df.drop(['id', 'timestamp'], axis=1)
```

**4. Handle High Cardinality:**
- Features with many unique values may need binning
- Or use `cat_l2`, `cat_smooth` parameters

---

**What NOT to Do:**

| Don't | Reason |
|-------|--------|
| One-hot encode | LightGBM handles categories natively |
| Scale features | Trees are scale-invariant |
| Fill missing values | Native handling is better |
| Remove outliers aggressively | Trees are robust |

---

**Final Checklist:**
```python
# Prepare data
X = df.drop('target', axis=1)
y = df['target']

# Create dataset with categorical features
train_data = lgb.Dataset(
    X, label=y,
    categorical_feature=categorical_cols
)
```

---

## Question 3
- [ ] Done

**In what scenarios would you prefer LightGBM over other machine learning algorithms?**

**Answer:**

**Prefer LightGBM When:**

| Scenario | Reason |
|----------|--------|
| **Large datasets (>100K rows)** | Fast training, memory efficient |
| **Tabular data** | Trees excel on structured data |
| **Mixed feature types** | Handles numerical + categorical |
| **Fast iteration needed** | Quick training enables experimentation |
| **Kaggle competitions** | State-of-the-art for tabular |
| **Production with latency constraints** | Fast inference |
| **Missing values common** | Native handling |

---

**When to Choose Other Algorithms:**

| Scenario | Better Choice |
|----------|---------------|
| Small data (<1K) | Logistic Regression, SVM |
| Image/text/audio | Deep Learning (CNN, Transformer) |
| Interpretability critical | Linear models, Decision Trees |
| Sequential/time dependencies | RNN, LSTM, Transformers |
| Need confidence intervals | Bayesian methods |
| Very high cardinality categories | CatBoost |
| Overfitting concerns | XGBoost (depth-wise), Random Forest |

---

**LightGBM vs Alternatives:**

| vs Algorithm | Choose LightGBM if |
|--------------|-------------------|
| vs XGBoost | Need speed, large data |
| vs CatBoost | Speed priority over categorical handling |
| vs Random Forest | Need boosting (often higher accuracy) |
| vs Neural Networks | Tabular data, less tuning desired |
| vs Linear Models | Nonlinear relationships expected |

**Rule of Thumb:**
For structured/tabular data with reasonable size, LightGBM is often the best starting point.

---

## Question 4
- [ ] Done

**How do you approach hyperparameter optimization for a LightGBM model?**

**Answer:**

**Key Hyperparameters to Tune:**

| Category | Parameters |
|----------|-----------|
| Tree structure | `num_leaves`, `max_depth`, `min_data_in_leaf` |
| Regularization | `lambda_l1`, `lambda_l2`, `min_gain_to_split` |
| Sampling | `bagging_fraction`, `feature_fraction` |
| Learning | `learning_rate`, `num_iterations` |

---

**Tuning Strategies:**

**1. Grid Search**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'num_leaves': [31, 63, 127],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [-1, 5, 10]
}

grid = GridSearchCV(lgb.LGBMClassifier(), param_grid, cv=5)
grid.fit(X, y)
```

**2. Random Search (Better for many params)**
```python
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions, n_iter=50, cv=5)
```

**3. Optuna (Recommended)**
```python
import optuna

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0)
    }
    cv_result = lgb.cv(params, train_data, nfold=5)
    return cv_result['auc-mean'][-1]

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

---

**Tuning Order:**
1. Fix `learning_rate=0.1`, tune tree params
2. Tune regularization
3. Lower learning rate, increase trees with early stopping

---

## Question 5
- [ ] Done

**Which metrics can you use to evaluate the performance of a LightGBM model?**

**Answer:**

**Classification Metrics:**

| Metric | Parameter | Use Case |
|--------|-----------|----------|
| **Binary Log Loss** | `binary_logloss` | Binary classification |
| **AUC-ROC** | `auc` | Binary, ranking ability |
| **Accuracy** | `binary_error` | Balanced classes |
| **Multi-class Log Loss** | `multi_logloss` | Multi-class |
| **Multi-class Error** | `multi_error` | Multi-class accuracy |

**Regression Metrics:**

| Metric | Parameter | Use Case |
|--------|-----------|----------|
| **RMSE** | `rmse` | Standard regression |
| **MSE** | `mse` | Penalize large errors |
| **MAE** | `mae` | Robust to outliers |
| **MAPE** | `mape` | Percentage errors |
| **Huber** | `huber` | Robust regression |

**Ranking Metrics:**

| Metric | Parameter | Use Case |
|--------|-----------|----------|
| **NDCG** | `ndcg` | Learning to rank |
| **MAP** | `map` | Precision-based ranking |

---

**Usage:**
```python
params = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],  # Multiple metrics
    'first_metric_only': False
}

# Cross-validation
cv_results = lgb.cv(params, train_data, nfold=5, 
                    metrics='auc', stratified=True)
```

**Custom Metric:**
```python
def custom_metric(preds, data):
    labels = data.get_label()
    score = my_custom_score(labels, preds)
    return 'custom_name', score, True  # True = higher is better
```

---

## Question 6
- [ ] Done

**How can LightGBM be applied to ranking problems, and what parameters are important in this context?**

**Answer:**

**LightGBM Ranking Application:**
LightGBM supports learning-to-rank for search engines, recommendations, and document retrieval.

---

**Data Format:**
```python
# Group by query (user/search session)
# Each group has items with relevance scores

train_data = lgb.Dataset(
    X_train,
    label=y_train,  # Relevance: 0, 1, 2, ... (higher = more relevant)
    group=group_sizes  # [n_items_query1, n_items_query2, ...]
)
```

---

**Important Parameters:**

| Parameter | Description |
|-----------|-------------|
| `objective` | `'lambdarank'` or `'rank_xendcg'` |
| `metric` | `'ndcg'`, `'map'` |
| `ndcg_eval_at` | Positions to evaluate (e.g., [1, 3, 5, 10]) |
| `label_gain` | Gain for each relevance level |
| `lambdarank_truncation_level` | Number of positions to consider |

---

**Training Example:**
```python
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [5, 10],
    'num_leaves': 31,
    'learning_rate': 0.05,
    'min_data_in_leaf': 20,
    'lambdarank_truncation_level': 30
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=1000,
    early_stopping_rounds=50
)
```

---

**Ranking Objectives:**

| Objective | Description |
|-----------|-------------|
| `lambdarank` | Optimizes NDCG directly |
| `rank_xendcg` | Cross-entropy approach to NDCG |

**Prediction:** Returns relevance scores to rank items within each query.

---

## Question 7
- [ ] Done

**Detail how LightGBM can be used for multiclass classification problems.**

**Answer:**

**Native Multiclass Support:**
LightGBM handles multiclass classification using one-vs-all or softmax approaches.

---

**Setup:**
```python
params = {
    'objective': 'multiclass',
    'num_class': 3,  # Number of classes
    'metric': 'multi_logloss'
}

train_data = lgb.Dataset(X_train, label=y_train)  # Labels: 0, 1, 2, ...
model = lgb.train(params, train_data, num_boost_round=100)
```

---

**Objectives:**

| Objective | Description |
|-----------|-------------|
| `multiclass` | Softmax, outputs probabilities |
| `multiclassova` | One-vs-all, binary classifiers |

---

**Prediction:**
```python
# Returns probability matrix: (n_samples, n_classes)
probs = model.predict(X_test)
# Shape: (n_samples, num_class)

# Get class predictions
predictions = np.argmax(probs, axis=1)
```

---

**Evaluation Metrics:**

| Metric | Description |
|--------|-------------|
| `multi_logloss` | Cross-entropy loss |
| `multi_error` | Classification error rate |

---

**Sklearn API:**
```python
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

model = LGBMClassifier(
    objective='multiclass',
    num_class=3,
    n_estimators=100
)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(classification_report(y_test, preds))
```

**Note:** Labels must be integers starting from 0: [0, 1, 2, ..., num_class-1]

---

## Question 8
- [ ] Done

**Outline your approach to building a fraud detection system using LightGBM.**

**Answer:**

**Fraud Detection Challenges:**
- Highly imbalanced data (fraud ~0.1-1%)
- Need real-time prediction
- Evolving fraud patterns

---

**Approach:**

**1. Data Preparation**
```python
# Features
features = ['transaction_amount', 'merchant_category', 'time_since_last_tx',
            'location_distance', 'device_type', 'historical_avg_amount']

# Handle imbalance
fraud_weight = len(y[y==0]) / len(y[y==1])  # e.g., 100
```

**2. Model Configuration**
```python
params = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'is_unbalance': True,        # Handle imbalance
    # OR
    'scale_pos_weight': fraud_weight,  # Weight positive class
    
    'num_leaves': 63,
    'max_depth': 8,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}
```

**3. Training with Early Stopping**
```python
model = lgb.train(
    params, train_data,
    valid_sets=[valid_data],
    num_boost_round=1000,
    early_stopping_rounds=50
)
```

**4. Threshold Optimization**
```python
# Optimize for precision-recall trade-off
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, probs)
# Choose threshold based on business requirements
```

**5. Evaluation**
- AUC-PR (better than AUC-ROC for imbalanced)
- Precision at fixed recall
- Business metrics (false positive cost)

**6. Real-time Serving**
- Export model, use fast inference
- Monitor for concept drift

---

## Question 9
- [ ] Done

**How can the interpretability of LightGBM be improved while maintaining its performance?**

**Answer:**

**Interpretability Strategies:**

**1. Feature Importance Analysis**
```python
# Built-in importance
importance = model.feature_importance(importance_type='gain')

# Visualize
lgb.plot_importance(model, max_num_features=15)
```

**2. SHAP Values (Best for Explanation)**
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Global explanation
shap.summary_plot(shap_values, X)

# Local explanation (single prediction)
shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0])
```

**3. Partial Dependence Plots**
```python
from sklearn.inspection import PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(model, X, features=['feature1', 'feature2'])
```

**4. Monotonic Constraints**
```python
# Force feature to have monotonic relationship
params = {
    'monotone_constraints': [1, -1, 0]  # 1=increasing, -1=decreasing, 0=none
}
```
- Improves interpretability without hurting performance

**5. Feature Interaction Constraints**
```python
params = {
    'interaction_constraints': [[0, 1], [2, 3]]  # Limit which features can interact
}
```

**6. Simpler Model Settings**
```python
params = {
    'max_depth': 5,      # Shallower trees
    'num_leaves': 15,    # Fewer leaves
}
```

---

**Balance Interpretability vs Performance:**

| Approach | Interpretability | Performance Impact |
|----------|------------------|-------------------|
| SHAP | High | None |
| Monotonic constraints | Medium | Slight |
| Shallower trees | High | Some loss |

---

## Question 10
- [ ] Done

**Consider the implications of adversarial examples on LightGBM models and how you would protect against them.**

**Answer:**

**Adversarial Examples in Tree Models:**
Adversarial examples are inputs intentionally modified to cause misclassification. While more studied in neural networks, tree models are also vulnerable.

---

**Types of Attacks:**

| Attack | Description |
|--------|-------------|
| **Evasion attacks** | Modify input at test time to avoid detection |
| **Data poisoning** | Inject malicious training data |
| **Model extraction** | Query model to steal/replicate it |

---

**Vulnerabilities in LightGBM:**
- Decision boundaries are discrete (split points)
- Small perturbations near boundaries can flip predictions
- Feature importance reveals attack surface

---

**Defense Strategies:**

**1. Adversarial Training**
```python
# Generate adversarial samples, add to training
X_train_aug = np.vstack([X_train, X_adversarial])
y_train_aug = np.hstack([y_train, y_adversarial])
```

**2. Input Validation**
```python
# Detect suspicious inputs
def validate_input(x, training_stats):
    for feature, value in enumerate(x):
        if value < training_stats['min'][feature] - threshold:
            return False  # Anomalous
    return True
```

**3. Ensemble Defenses**
- Train multiple models with different seeds
- Require agreement for high-stakes decisions

**4. Feature Preprocessing**
- Discretization reduces attack surface
- Feature selection removes easily manipulated features

**5. Robust Training**
```python
params = {
    'lambda_l1': 1.0,  # Regularization
    'lambda_l2': 1.0,
    'min_data_in_leaf': 50  # More data per decision
}
```

**6. Monitoring**
- Track prediction confidence distributions
- Alert on unusual input patterns

---

## Question 11
- [ ] Done

**Explore the possibility of combining LightGBM with neural networks in a hybrid model for complex tasks.**

**Answer:**

**Hybrid Model Approaches:**

**1. Neural Network Embeddings + LightGBM**
```python
# Train NN on complex features (text, images)
# Extract embeddings
embeddings = nn_model.get_embeddings(raw_data)

# Combine with tabular features
X_combined = np.hstack([tabular_features, embeddings])

# Train LightGBM on combined features
model = lgb.train(params, lgb.Dataset(X_combined, y))
```

**Use case:** Product recommendations with image + behavior data

**2. Stacking Ensemble**
```python
# Level 1: Multiple base models
nn_preds = nn_model.predict(X)
lgbm_preds = lgbm_model.predict(X)

# Level 2: Meta-learner
meta_features = np.column_stack([nn_preds, lgbm_preds])
final_model = LGBMClassifier().fit(meta_features, y)
```

**3. Residual Learning**
```python
# NN captures complex patterns
nn_preds = nn_model.predict(X)

# LightGBM learns residuals
residuals = y - nn_preds
lgbm_model = lgb.train(params, lgb.Dataset(X, residuals))

# Final prediction
final_pred = nn_preds + lgbm_model.predict(X)
```

**4. Feature Engineering with NN**
- Use autoencoder to create compressed features
- Train LightGBM on latent representations

---

**Benefits of Hybrid:**

| Aspect | NN Strength | LightGBM Strength |
|--------|-------------|-------------------|
| Unstructured data | Excellent | Poor |
| Tabular data | Moderate | Excellent |
| Training speed | Slow | Fast |
| Interpretability | Low | Higher |

**Recommendation:** Use NNs for feature extraction from complex data, LightGBM for final prediction on combined features.

---

