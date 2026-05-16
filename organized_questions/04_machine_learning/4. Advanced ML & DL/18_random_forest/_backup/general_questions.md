# Random Forest Interview Questions - General Questions

## Question 1
- [ ] Done

**How do you handle missing values in a Random Forest model?**

### Answer

**Definition:**
Random Forest can handle missing values through surrogate splits (built-in), imputation before training, or using implementations that natively support NaN values.

**Approaches:**

**1. Surrogate Splits (CART-based):**
- During training, find alternative splits that mimic primary split
- When primary feature is missing, use surrogate
- Not all implementations support this

**2. Pre-imputation:**
```python
from sklearn.impute import SimpleImputer
import numpy as np

# Median imputation (recommended for RF)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Or use RF-based imputation (iterative)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=42)
X_imputed = imputer.fit_transform(X)
```

**3. Missing Indicator Feature:**
```python
# Add binary column indicating missingness
X['feature_missing'] = X['feature'].isna().astype(int)
X['feature'] = X['feature'].fillna(X['feature'].median())
```

**4. Native NaN Support:**
```python
# Some implementations handle NaN natively
import lightgbm as lgb
# LightGBM handles NaN automatically

# HistGradientBoosting in sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
```

**Best Practice:**
- Small missing %: Median/mode imputation
- Large missing %: Add missing indicator + impute
- Many features missing: Use iterative imputation

---

## Question 2
- [ ] Done

**Can Random Forest be used for both classification and regression tasks?**

### Answer

**Definition:**
Yes, Random Forest works for both classification (RandomForestClassifier) and regression (RandomForestRegressor). The difference lies in how predictions are aggregated and how splits are evaluated.

**Classification vs Regression:**

| Aspect | Classification | Regression |
|--------|----------------|------------|
| **Target** | Categorical (classes) | Continuous |
| **Split Criterion** | Gini impurity, Entropy | MSE, MAE |
| **Aggregation** | Majority voting | Mean/Median |
| **Output** | Class label + probabilities | Numeric value |

**Classification:**
```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(
    n_estimators=100,
    criterion='gini'  # or 'entropy'
)
rf_clf.fit(X_train, y_train)
predictions = rf_clf.predict(X_test)
probabilities = rf_clf.predict_proba(X_test)
```

**Regression:**
```python
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(
    n_estimators=100,
    criterion='squared_error'  # or 'absolute_error'
)
rf_reg.fit(X_train, y_train)
predictions = rf_reg.predict(X_test)
```

**How Regression Aggregation Works:**
$$\hat{y} = \frac{1}{B}\sum_{b=1}^{B} T_b(x)$$

Each tree predicts a value, final prediction is the average.

**Multi-output:**
Both support multi-output (multiple targets simultaneously).

---

## Question 3
- [ ] Done

**Compare Random Forest with Gradient Boosting Machine (GBM).**

### Answer

**Definition:**
Both are tree-based ensembles, but Random Forest uses bagging (parallel independent trees) while GBM uses boosting (sequential trees correcting previous errors).

**Comparison Table:**

| Aspect | Random Forest | Gradient Boosting |
|--------|---------------|-------------------|
| **Method** | Bagging | Boosting |
| **Tree Training** | Parallel, independent | Sequential, dependent |
| **Tree Depth** | Deep (fully grown) | Shallow (weak learners) |
| **What it Reduces** | Variance | Bias |
| **Learning Rate** | N/A | Critical parameter |
| **Overfitting Risk** | Lower | Higher |
| **Training Speed** | Faster (parallelizable) | Slower (sequential) |
| **Noise Sensitivity** | Robust | Sensitive |
| **Hyperparameter Tuning** | Easier | More sensitive |

**When to Use Each:**

| Scenario | Recommended |
|----------|-------------|
| Noisy data | Random Forest |
| Clean, structured data | GBM |
| Quick baseline | Random Forest |
| Maximum accuracy needed | GBM (tuned) |
| Limited tuning time | Random Forest |
| Imbalanced classes | Either with proper handling |

**Mathematical Difference:**

**Random Forest:**
$$\hat{f}(x) = \frac{1}{B}\sum_{b=1}^{B} T_b(x) \quad \text{(average)}$$

**GBM:**
$$\hat{f}_m(x) = \hat{f}_{m-1}(x) + \gamma_m T_m(x) \quad \text{(additive)}$$

**Interview Point:**
"RF for robustness and speed; GBM for squeezing maximum accuracy when data is clean and I have time to tune."

---

## Question 4
- [ ] Done

**How do you determine the number of trees to use in a Random Forest?**

### Answer

**Definition:**
The optimal number of trees balances performance and computational cost. Generally, more trees improve accuracy with diminishing returns. Use OOB error or cross-validation to find the point where adding trees stops helping.

**Methods to Determine:**

**1. OOB Error Curve:**
```python
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

oob_errors = []
tree_range = range(10, 500, 20)

for n_trees in tree_range:
    rf = RandomForestClassifier(n_estimators=n_trees, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)
    oob_errors.append(1 - rf.oob_score_)

plt.plot(tree_range, oob_errors)
plt.xlabel('Number of Trees')
plt.ylabel('OOB Error')
plt.title('OOB Error vs Number of Trees')
# Choose where curve flattens
```

**2. Cross-Validation:**
```python
from sklearn.model_selection import cross_val_score

for n_trees in [50, 100, 200, 300, 500]:
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    scores = cross_val_score(rf, X, y, cv=5)
    print(f"n_trees={n_trees}: {scores.mean():.4f} ± {scores.std():.4f}")
```

**Rules of Thumb:**
- Start: 100-200 trees
- Add more until improvement < 0.1%
- Typical range: 100-1000
- More features/complex data → more trees

**Tradeoffs:**

| More Trees | Fewer Trees |
|------------|-------------|
| Better accuracy | Faster training |
| Slower training/prediction | May underfit |
| More memory | Less memory |
| Diminishing returns | Might miss patterns |

**Practical Advice:**
- Production with latency constraints: Fewer trees (50-100)
- Accuracy critical: More trees (300-500+)
- Default starting point: 100 trees

---

## Question 5
- [ ] Done

**How can Random Forest be used for feature selection?**

### Answer

**Definition:**
Random Forest provides feature importance scores that can be used to select the most predictive features. This can be done using Gini importance, permutation importance, or recursive feature elimination.

**Methods:**

**1. Gini/MDI Importance:**
```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Select top features
top_features = importance_df.head(20)['feature'].tolist()
```

**2. Permutation Importance (Recommended):**
```python
from sklearn.inspection import permutation_importance

perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': perm_imp.importances_mean,
    'std': perm_imp.importances_std
}).sort_values('importance', ascending=False)

# Select features with positive importance
selected = importance_df[importance_df['importance'] > 0]['feature'].tolist()
```

**3. Recursive Feature Elimination:**
```python
from sklearn.feature_selection import RFE

rf = RandomForestClassifier(n_estimators=100)
rfe = RFE(estimator=rf, n_features_to_select=10, step=1)
rfe.fit(X_train, y_train)

selected_features = [f for f, s in zip(feature_names, rfe.support_) if s]
```

**4. SelectFromModel:**
```python
from sklearn.feature_selection import SelectFromModel

rf = RandomForestClassifier(n_estimators=100)
selector = SelectFromModel(rf, threshold='median')
X_selected = selector.fit_transform(X_train, y_train)
```

**Best Practice:**
- Use permutation importance (unbiased)
- Validate selected features on held-out data
- Check for correlated features (importance may split)

---

## Question 6
- [ ] Done

**How do you measure the performance of a Random Forest model?**

### Answer

**Definition:**
Performance is measured using task-appropriate metrics: accuracy, precision, recall, F1, AUC-ROC for classification; MSE, RMSE, MAE, R² for regression. OOB error provides quick internal validation.

**Classification Metrics:**

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

# Train and predict
rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# Metrics
print(f"OOB Accuracy: {rf.oob_score_:.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
print(classification_report(y_test, y_pred))
```

**Regression Metrics:**

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

rf = RandomForestRegressor(n_estimators=100, oob_score=True)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"OOB R²: {rf.oob_score_:.4f}")
print(f"Test R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
```

**Metric Selection Guide:**

| Task | Metric | When to Use |
|------|--------|-------------|
| Balanced Classification | Accuracy, F1 | Classes roughly equal |
| Imbalanced Classification | Precision, Recall, AUC | Rare event detection |
| Regression | RMSE | Penalize large errors |
| Regression | MAE | Robust to outliers |
| Ranking | AUC-ROC | Probability calibration matters |

---

## Question 7
- [ ] Done

**Can Random Forest handle time series data? If so, how?**

### Answer

**Definition:**
Random Forest can be adapted for time series through feature engineering (lag features, rolling statistics) but requires careful handling to avoid data leakage. It doesn't natively handle temporal dependencies.

**Challenges:**
- RF doesn't understand time ordering
- Standard CV causes data leakage
- Cannot extrapolate trends

**Adaptation Approaches:**

**1. Feature Engineering:**
```python
import pandas as pd

def create_time_features(df, target_col, lags=[1,2,3,7]):
    """Create lag and rolling features"""
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling statistics
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window).mean()
        df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window).std()
    
    # Date features
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    
    return df.dropna()
```

**2. Time Series CV (Avoid Leakage):**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
```

**3. Recursive Forecasting:**
```python
# Multi-step forecasting
predictions = []
for step in range(forecast_horizon):
    X_current = create_features(history)
    pred = rf.predict(X_current.iloc[[-1]])[0]
    predictions.append(pred)
    history = history.append(pred)  # Update history with prediction
```

**Limitations:**
- Cannot extrapolate (e.g., trending data)
- Consider: ARIMA, Prophet, LSTM for pure time series

---

## Question 8
- [ ] Done

**How do you deal with categorical variables in Random Forest?**

### Answer

**Definition:**
Random Forest requires numerical inputs. Categorical variables must be encoded using label encoding (ordinal), one-hot encoding (nominal), or target encoding. Scikit-learn RF doesn't handle categories natively, unlike some other implementations.

**Encoding Methods:**

**1. One-Hot Encoding (Low Cardinality):**
```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# For pandas
df_encoded = pd.get_dummies(df, columns=['category_col'])

# For sklearn pipeline
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(df[['category_col']])
```

**2. Label Encoding (Ordinal):**
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category_col'])
# Note: Implies ordering - OK for trees, but not semantically correct
```

**3. Target Encoding (High Cardinality):**
```python
# Mean target value per category
target_means = df.groupby('category_col')['target'].mean()
df['category_encoded'] = df['category_col'].map(target_means)

# Use with caution: potential data leakage
# Better: use within CV folds
```

**4. Ordinal Encoding (Ordered Categories):**
```python
from sklearn.preprocessing import OrdinalEncoder

# For ordered categories like ['low', 'medium', 'high']
encoder = OrdinalEncoder(categories=[['low', 'medium', 'high']])
df['encoded'] = encoder.fit_transform(df[['ordered_category']])
```

**Best Practices:**

| Cardinality | Recommended Encoding |
|-------------|---------------------|
| Low (< 10 levels) | One-Hot |
| Medium (10-100) | Label or Target |
| High (100+) | Target or Embedding |

**Native Category Support:**
```python
# LightGBM handles categories natively
import lightgbm as lgb
lgb_clf = lgb.LGBMClassifier()
# Specify categorical features directly
```

---

## Question 9
- [ ] Done

**What practices should be followed to scale Random Forest for big data?**

### Answer

**Definition:**
Scaling Random Forest for big data involves parallelization, distributed computing frameworks, subsampling strategies, and optimized implementations.

**Scaling Strategies:**

**1. Parallelization (Single Machine):**
```python
from sklearn.ensemble import RandomForestClassifier

# Use all CPU cores
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
```

**2. Distributed Computing (Spark):**
```python
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(
    numTrees=100,
    maxDepth=10,
    featureSubsetStrategy='sqrt'
)
model = rf.fit(train_df)
```

**3. Subsampling:**
```python
# Use max_samples parameter (sklearn 0.22+)
rf = RandomForestClassifier(
    n_estimators=100,
    max_samples=0.1,  # Use 10% of data per tree
    n_jobs=-1
)
```

**4. GPU Acceleration:**
```python
# RAPIDS cuML
from cuml.ensemble import RandomForestClassifier as cuRF

rf_gpu = cuRF(n_estimators=100)
rf_gpu.fit(X_gpu, y_gpu)
```

**5. Efficient Data Formats:**
```python
# Use memory-efficient formats
import numpy as np

# Convert to float32 (half the memory of float64)
X = X.astype(np.float32)

# Use sparse matrices if data is sparse
from scipy import sparse
X_sparse = sparse.csr_matrix(X)
```

**Scaling Checklist:**

| Technique | When to Use |
|-----------|-------------|
| n_jobs=-1 | Always (single machine) |
| max_samples | Data > 100K rows |
| Spark/Dask | Data > 10M rows |
| GPU (cuML) | Need speed, have GPU |
| Fewer trees | Latency constraints |
| Reduced max_depth | Memory constraints |

**Memory Optimization:**
- Limit max_depth
- Increase min_samples_leaf
- Use fewer trees
- Reduce features (feature selection first)

---

## Question 10
- [ ] Done

**What model validation techniques would you apply for a Random Forest algorithm?**

### Answer

**Definition:**
Model validation ensures Random Forest generalizes well. Techniques include OOB error, k-fold cross-validation, holdout validation, and time-based splits for temporal data.

**Validation Techniques:**

**1. OOB Error (Built-in):**
```python
rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X, y)
print(f"OOB Score: {rf.oob_score_:.4f}")
# Approximately equal to LOOCV, no separate set needed
```

**2. K-Fold Cross-Validation:**
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
print(f"CV Score: {scores.mean():.4f} ± {scores.std():.4f}")
```

**3. Train-Validation-Test Split:**
```python
from sklearn.model_selection import train_test_split

# First split: train+val vs test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
# Second split: train vs val
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)

rf.fit(X_train, y_train)
val_score = rf.score(X_val, y_val)  # For hyperparameter tuning
test_score = rf.score(X_test, y_test)  # Final evaluation
```

**4. Time Series Split:**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(rf, X, y, cv=tscv)
```

**5. Nested Cross-Validation (Hyperparameter Tuning):**
```python
from sklearn.model_selection import GridSearchCV, cross_val_score

inner_cv = StratifiedKFold(n_splits=3)
outer_cv = StratifiedKFold(n_splits=5)

param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search = GridSearchCV(rf, param_grid, cv=inner_cv)
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv)
```

**Validation Strategy Selection:**

| Scenario | Recommended |
|----------|-------------|
| Quick assessment | OOB error |
| Thorough evaluation | 5-fold CV |
| Time series | TimeSeriesSplit |
| Hyperparameter tuning | Nested CV |
| Large dataset | Single holdout |

---

## Question 11
- [ ] Done

**How do you tune a Random Forest model's hyperparameters systematically?**

### Answer

**Definition:**
Systematic hyperparameter tuning involves grid search, random search, or Bayesian optimization to find optimal values for n_estimators, max_depth, max_features, and other parameters.

**Key Parameters to Tune:**
```
n_estimators: [100, 200, 300, 500]
max_depth: [None, 10, 20, 30]
max_features: ['sqrt', 'log2', 0.3, 0.5]
min_samples_split: [2, 5, 10]
min_samples_leaf: [1, 2, 4]
```

**1. Grid Search:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

**2. Random Search (Faster):**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None] + list(range(5, 30)),
    'max_features': uniform(0.1, 0.9),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    rf, param_dist, n_iter=50, cv=5, scoring='f1', n_jobs=-1, random_state=42
)
random_search.fit(X_train, y_train)
```

**3. Bayesian Optimization (Efficient):**
```python
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

search_space = {
    'n_estimators': Integer(100, 500),
    'max_depth': Integer(5, 30),
    'max_features': Real(0.1, 1.0),
    'min_samples_leaf': Integer(1, 10)
}

bayes_search = BayesSearchCV(
    rf, search_space, n_iter=50, cv=5, scoring='f1', n_jobs=-1
)
bayes_search.fit(X_train, y_train)
```

**Tuning Strategy:**
1. Start with random search (broad exploration)
2. Narrow range, use grid search (fine tuning)
3. Or use Bayesian optimization throughout

**Time Efficiency:**
- Grid: $O(n^k)$ where n=values, k=params
- Random: O(iterations)
- Bayesian: O(iterations) but smarter sampling

---

## Question 12
- [ ] Done

**How is Random Forest used in the analysis of genomic and bioinformatics data?**

### Answer

**Definition:**
Random Forest is widely used in bioinformatics for gene selection, disease classification, and variant prioritization due to its ability to handle high-dimensional data (many genes, few samples) and provide feature importance.

**Applications:**

**1. Gene Expression Classification:**
```python
# Classify cancer types based on gene expression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# X: samples × genes (e.g., 200 samples × 20,000 genes)
rf = RandomForestClassifier(n_estimators=500, max_features='sqrt', random_state=42)
rf.fit(X_train, y_train)  # y = cancer subtype

# Gene importance for biomarker discovery
gene_importance = pd.DataFrame({
    'gene': gene_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

**2. SNP-based Disease Risk:**
```python
# Predict disease from genetic variants
# Features: SNP genotypes (0, 1, 2 copies of minor allele)
rf = RandomForestClassifier(n_estimators=1000, max_depth=10)
rf.fit(X_snp, y_disease)

# Identify risk variants
risk_snps = importance_df[importance_df['importance'] > threshold]
```

**3. Protein Function Prediction:**
- Features: Sequence-derived, structural features
- Target: Protein function class
- RF handles mixed feature types well

**Why RF for Bioinformatics:**

| Advantage | Relevance |
|-----------|-----------|
| High-dimensional data | Many genes, few samples (n << p) |
| Feature importance | Biomarker discovery |
| No feature scaling | Diverse feature types |
| OOB error | No need for separate test set |
| Handles interactions | Gene-gene interactions |

**Challenges & Solutions:**

```python
# Class imbalance (rare diseases)
rf = RandomForestClassifier(class_weight='balanced')

# High dimensionality
# Pre-filter genes by variance or differential expression
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.1)
X_filtered = selector.fit_transform(X)
```

---

## Question 13
- [ ] Done

**What role does Random Forest play in complex systems like self-driving cars or high-frequency trading algorithms?**

### Answer

**Definition:**
Random Forest serves as a robust component in complex systems for tasks like object classification, signal processing, and decision-making, though often combined with or replaced by deep learning in critical real-time applications.

**Self-Driving Cars:**

**Applications:**
- LiDAR point cloud classification (road vs obstacle)
- Sensor fusion decisions
- Intent prediction (pedestrian behavior)
- Non-critical subsystems

```python
# Example: Classify LiDAR segments
features = ['height', 'intensity', 'point_density', 'shape_features']
rf = RandomForestClassifier(n_estimators=100, max_depth=10)
# Target: obstacle_type (vehicle, pedestrian, cyclist, static)
```

**Limitations in Self-Driving:**
- Deep learning (CNNs) preferred for image recognition
- Real-time constraints (RF prediction can be slow with many trees)
- Typically used in non-critical or offline analysis

**High-Frequency Trading:**

**Applications:**
- Feature importance for signal discovery
- Regime classification (trending/mean-reverting)
- Risk model components
- Backtesting/research phase

```python
# Example: Predict price direction
features = ['momentum', 'volatility', 'volume_imbalance', 'spread', 'microstructure_features']
rf = RandomForestClassifier(n_estimators=200, max_depth=5)  # Shallow for speed
# Target: price_direction_next_tick
```

**HFT Considerations:**

| Factor | Consideration |
|--------|---------------|
| Latency | Fewer trees, shallow depth |
| Overfitting | Market regimes change; need regularization |
| Feature engineering | Critical for alpha generation |
| Non-stationarity | Frequent retraining needed |

**Why RF in Production Systems:**

| Advantage | Application |
|-----------|-------------|
| Robustness | Noisy sensor data |
| Interpretability | Regulatory requirements |
| No scaling needed | Fast preprocessing |
| OOB validation | Quick model assessment |

**Reality Check:**
- Deep learning dominates perception in autonomous vehicles
- RF often used in research/prototyping, simpler subsystems
- In HFT, speed often trumps RF (linear models, hardware solutions)

**Interview Point:**
"RF is valuable for rapid prototyping, feature selection, and interpretable models in complex systems, but production often requires specialized solutions optimized for latency or accuracy."
