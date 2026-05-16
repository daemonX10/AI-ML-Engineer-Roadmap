# Light Gbm Interview Questions - Scenario_Based Questions

## Question 1
- [ ] Done

**Discuss the types of tree learners available in LightGBM.**

**Answer:**

**Tree Learners (tree_learner parameter):**

**1. Serial (Default for single machine)**
```python
params = {'tree_learner': 'serial'}
```
- Standard single-machine training
- All features and data on one node
- Best for: small to medium datasets

**2. Feature Parallel**
```python
params = {'tree_learner': 'feature'}
```
- Distributes features across workers
- Each worker finds best split for its features
- Communication: share split decisions
- Best for: many features, few workers

**3. Data Parallel**
```python
params = {'tree_learner': 'data'}
```
- Distributes data (rows) across workers
- Each worker builds local histogram
- Communication: aggregate histograms
- Best for: many rows, distributed training

**4. Voting Parallel**
```python
params = {'tree_learner': 'voting'}
```
- Hybrid of data and feature parallel
- Workers vote on top features
- Only aggregate histograms for winning features
- Best for: large-scale distributed training, reduces communication

---

**Comparison:**

| Learner | Best For | Communication Cost |
|---------|----------|-------------------|
| Serial | Single machine | None |
| Feature | Many features | Low (split points) |
| Data | Many rows | High (histograms) |
| Voting | Very large scale | Medium (voting + selected histograms) |

**Usage Example:**
```python
params = {
    'tree_learner': 'data',
    'num_machines': 4,
    'local_listen_port': 12345
}
```

---

## Question 2
- [ ] Done

**How would you tune the number of leaves or maximum depth of trees in LightGBM?**

**Answer:**

**Relationship Between Parameters:**
- `num_leaves`: Maximum number of leaves per tree (LightGBM primary control)
- `max_depth`: Maximum depth of tree (secondary constraint)

**Rule of thumb:** num_leaves ≤ 2^max_depth

---

**Tuning Approach:**

**1. Start with num_leaves (Primary)**
```python
# Default = 31, try range based on data size
num_leaves_options = [15, 31, 63, 127, 255]
```

**2. Use max_depth as safety limit**
```python
params = {
    'num_leaves': 31,
    'max_depth': 6  # Prevents overly deep trees
}
```

---

**Guidelines by Dataset Size:**

| Dataset Size | num_leaves | max_depth |
|--------------|------------|-----------|
| Small (<10K) | 15-31 | 4-6 |
| Medium (10K-100K) | 31-63 | 6-8 |
| Large (>100K) | 63-255 | 8-12 or -1 |

---

**Tuning Process:**
```python
import optuna

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
    }
    # Ensure consistency
    if params['num_leaves'] > 2**params['max_depth']:
        params['num_leaves'] = 2**params['max_depth']
    
    cv_result = lgb.cv(params, train_data, nfold=5)
    return cv_result['auc-mean'][-1]
```

**Trade-offs:**

| Setting | Effect |
|---------|--------|
| More leaves | Higher accuracy, risk overfitting |
| Fewer leaves | Lower accuracy, better generalization |
| Deep trees | Capture complex patterns |
| Shallow trees | Simpler, more robust |

---

## Question 3
- [ ] Done

**Discuss the impact of using a large versus small bagging_fraction in LightGBM.**

**Answer:**

**Definition:**
`bagging_fraction` (or `subsample`) controls what fraction of training data is used to train each tree. Value between 0 and 1.

---

**Impact Comparison:**

| Aspect | Large (0.9-1.0) | Small (0.5-0.7) |
|--------|-----------------|-----------------|
| Data per tree | More data | Less data |
| Variance | Lower | Higher |
| Overfitting risk | Higher | Lower |
| Training speed | Slower | Faster |
| Bias | Lower | Slightly higher |

---

**Detailed Effects:**

**Large bagging_fraction (0.9-1.0):**
- Each tree sees almost all data
- Trees are more similar to each other
- Lower randomness, may overfit
- Better for small datasets

**Small bagging_fraction (0.5-0.7):**
- Each tree sees only a subset
- More diverse trees
- Regularization effect (reduces overfitting)
- Better for large datasets

---

**Usage:**
```python
params = {
    'bagging_fraction': 0.8,  # Use 80% of data per tree
    'bagging_freq': 5         # Apply bagging every 5 iterations
}
```

**Note:** Must set `bagging_freq > 0` for bagging_fraction to take effect.

---

**Recommended Settings:**

| Scenario | bagging_fraction |
|----------|-----------------|
| Small data, underfitting | 0.9-1.0 |
| Large data, default | 0.8 |
| Overfitting concerns | 0.6-0.7 |
| Very large data | 0.5-0.6 |

**Combined with feature_fraction:**
```python
params = {
    'bagging_fraction': 0.8,
    'feature_fraction': 0.8  # Also subsample features
}
```

---

## Question 4
- [ ] Done

**Discuss the support of weight-based sampling in LightGBM.**

**Answer:**

**Weight-Based Sampling Support:**
LightGBM supports instance weights for various purposes including handling imbalanced data and importance weighting.

---

**Methods:**

**1. Instance Weights**
```python
# Assign weights to each sample
train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
```

**Use cases:**
- Imbalanced classes (higher weight for minority)
- Recent data more important
- Business importance of samples

**2. Class Weights (for imbalanced data)**
```python
params = {
    'is_unbalance': True  # Auto-weight based on class frequency
}
# OR
params = {
    'scale_pos_weight': 10  # Weight for positive class
}
```

**3. GOSS (Gradient-based One-Side Sampling)**
```python
params = {
    'boosting_type': 'goss',
    'top_rate': 0.2,      # Keep top 20% by gradient
    'other_rate': 0.1     # Sample 10% of rest
}
```
- Weights samples by gradient magnitude
- Automatic, not manual

---

**Example: Custom Weights for Imbalanced Data**
```python
# Calculate weights
weights = np.where(y_train == 1, 
                   len(y_train) / (2 * sum(y_train)),
                   len(y_train) / (2 * (len(y_train) - sum(y_train))))

train_data = lgb.Dataset(X_train, label=y_train, weight=weights)
```

---

**Impact of Weights:**
- Loss is weighted: $L = \sum_i w_i \cdot l(y_i, \hat{y}_i)$
- Higher weight samples influence splits more
- Affects tree structure and predictions

**Note:** Use `sample_weight` in sklearn API:
```python
model.fit(X_train, y_train, sample_weight=weights)
```

---

## Question 5
- [ ] Done

**How would you use LightGBM to predict customer churn based on usage data?**

**Answer:**

**Problem Setup:**
Binary classification - predict if customer will churn (1) or stay (0)

---

**Feature Engineering:**

| Feature Category | Examples |
|-----------------|----------|
| Usage patterns | Logins/month, feature usage, session duration |
| Engagement | Days since last activity, support tickets |
| Financial | Monthly spend, payment delays |
| Demographics | Account age, plan type |
| Trend features | Usage change (this month vs last 3 months) |

---

**Model Building:**

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

# Handle imbalance
params = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'is_unbalance': True,  # Weight minority class
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}

# Train with early stopping
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
valid_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols)

model = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=1000,
    early_stopping_rounds=50
)

# Predict churn probability
churn_prob = model.predict(X_test)
```

---

**Evaluation:**
```python
from sklearn.metrics import roc_auc_score, precision_recall_curve

auc = roc_auc_score(y_test, churn_prob)
```

**Business Application:**
- Rank customers by churn probability
- Target high-risk customers with retention offers
- Analyze feature importance for actionable insights

---

## Question 6
- [ ] Done

**Discuss how LightGBM could be utilized in a high-frequency trading algorithm.**

**Answer:**

**HFT Requirements:**
- Ultra-low latency prediction (<1ms)
- High accuracy on price direction
- Real-time feature computation
- Continuous model updates

---

**LightGBM Suitability:**

| Requirement | LightGBM Capability |
|-------------|---------------------|
| Fast inference | Very fast (tree traversal) |
| Accuracy | State-of-the-art for tabular |
| Feature handling | Native categorical, missing values |
| Model size | Small, cache-efficient |

---

**Feature Engineering:**

| Category | Features |
|----------|----------|
| Price features | Returns, momentum, volatility |
| Order book | Bid-ask spread, depth imbalance |
| Technical | Moving averages, RSI, MACD |
| Microstructure | Trade volume, tick frequency |
| Cross-asset | Correlated instruments |

---

**Implementation:**

```python
params = {
    'objective': 'binary',  # or 'regression' for price prediction
    'num_leaves': 15,       # Keep small for speed
    'max_depth': 4,
    'learning_rate': 0.1,
    'num_iterations': 50,   # Fewer trees for speed
}

# Train offline, deploy for inference
model = lgb.train(params, train_data)
model.save_model('trading_model.txt')

# Fast inference in production
def predict(features):
    return model.predict([features])[0]
```

---

**Latency Optimization:**
- Fewer, shallower trees
- Minimal features
- Pre-compile model (ONNX, Treelite)
- Keep model in memory

**Challenges:**
- Non-stationarity requires frequent retraining
- Overfitting to market regimes
- Need robust risk management beyond predictions

---

## Question 7
- [ ] Done

**Propose a methodology for using LightGBM to detect anomalies in time-series sensor data.**

**Answer:**

**Approach: Supervised Anomaly Detection**
Frame as binary classification (normal vs anomaly) or regression (predict normal, flag deviations).

---

**Methodology:**

**1. Feature Engineering (Time-Series)**

| Feature Type | Examples |
|-------------|----------|
| Statistical | Rolling mean, std, min, max |
| Temporal | Hour, day of week, season |
| Lag features | Value at t-1, t-2, ..., t-n |
| Rate of change | First/second derivatives |
| Frequency | FFT components, spectral features |

```python
# Create features
df['rolling_mean'] = df['sensor'].rolling(window=10).mean()
df['rolling_std'] = df['sensor'].rolling(window=10).std()
df['lag_1'] = df['sensor'].shift(1)
df['diff'] = df['sensor'].diff()
```

**2. Model Training**

```python
# If labeled data available
params = {
    'objective': 'binary',
    'is_unbalance': True,  # Anomalies are rare
    'num_leaves': 31,
    'learning_rate': 0.05
}

model = lgb.train(params, train_data, 
                  valid_sets=[valid_data],
                  early_stopping_rounds=50)
```

**3. Alternative: Regression-Based Anomaly**
```python
# Predict expected value, flag large residuals
params = {'objective': 'regression'}
model = lgb.train(params, train_data)

predictions = model.predict(X_test)
residuals = np.abs(y_test - predictions)
anomalies = residuals > threshold  # e.g., 3 std
```

---

**Evaluation:**
- Precision/Recall (anomalies are rare)
- F1-score
- Time-to-detection

**Deployment:**
- Sliding window feature extraction
- Real-time prediction
- Alert system with threshold tuning

---

## Question 8
- [ ] Done

**Discuss the current research trends and advancements in the field of gradient boosting and LightGBM.**

**Answer:**

**Current Research Trends:**

**1. AutoML and Hyperparameter Optimization**
- Automatic tuning with Optuna, FLAML
- Meta-learning for hyperparameter transfer
- Neural Architecture Search adapted for trees

**2. Interpretability and Explainability**
- SHAP integration improvements
- Causal inference with gradient boosting
- Attention-like mechanisms for feature importance

**3. Handling Specific Data Types**
- Better categorical feature handling
- Time-series native support
- Graph-structured data integration

**4. Federated Learning**
- Privacy-preserving distributed training
- Secure multi-party computation for GBMs
- Training across organizations without data sharing

**5. Neural Network + GBM Hybrids**
- TabNet, NODE (differentiable trees)
- GBM for tabular + NN for embeddings
- End-to-end differentiable boosting

**6. Efficiency Improvements**
- GPU training optimization
- Quantization for faster inference
- Model compression techniques

---

**Recent Advancements:**

| Advancement | Description |
|-------------|-------------|
| **GPU training** | Faster training on large datasets |
| **Treelite/ONNX** | Model compilation for fast inference |
| **Monotonic constraints** | Enforcing domain knowledge |
| **Custom objectives** | Flexible loss functions |

---

**Emerging Directions:**
- Self-supervised pre-training for tabular data
- Few-shot learning with gradient boosting
- Continual/lifelong learning frameworks
- Uncertainty quantification (confidence intervals)

**Competition:** LightGBM vs XGBoost vs CatBoost continues driving improvements in speed, accuracy, and usability.

---

