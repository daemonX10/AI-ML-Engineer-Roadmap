# Scikit-Learn Interview Questions - General Questions

## Question 1
- [ ] Done

**How do you handle missing values in a dataset using Scikit-Learn ?**

### SimpleImputer Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `mean` | Replace with column mean | Numeric, normal distribution |
| `median` | Replace with column median | Numeric, with outliers |
| `most_frequent` | Replace with mode | Categorical |
| `constant` | Replace with specified value | Custom fill |

### Code Example
```python
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np

X = np.array([[1, 2], [np.nan, 3], [7, np.nan]])

# Mean imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Median imputation (robust to outliers)
imputer = SimpleImputer(strategy='median')

# KNN imputation (uses similar samples)
imputer = KNNImputer(n_neighbors=3)
X_imputed = imputer.fit_transform(X)
```

### Best Practice: Use in Pipeline
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
```

---

## Question 2
- [ ] Done

**How do you perform feature scaling in Scikit-Learn?**

### Scaling Methods

| Scaler | Formula | Use Case |
|--------|---------|----------|
| `StandardScaler` | (x - mean) / std | Most algorithms, normal dist |
| `MinMaxScaler` | (x - min) / (max - min) | Fixed range [0, 1] |
| `RobustScaler` | (x - median) / IQR | Data with outliers |
| `MaxAbsScaler` | x / max(abs(x)) | Sparse data |

### Code Example
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler: mean=0, std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same parameters!

# MinMaxScaler: [0, 1] range
scaler = MinMaxScaler(feature_range=(0, 1))

# RobustScaler: resistant to outliers
scaler = RobustScaler()
```

### Important: Fit only on training data!
```python
# CORRECT
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# WRONG (data leakage)
scaler.fit(X)  # Don't fit on all data!
```

---

## Question 3
- [ ] Done

**How do you encode categorical variables using Scikit-Learn ?**

### Encoding Methods

| Encoder | Output | Use Case |
|---------|--------|----------|
| `OneHotEncoder` | Binary columns | Nominal (no order) |
| `OrdinalEncoder` | Integer codes | Ordinal (has order) |
| `LabelEncoder` | Integer codes | Target variable |

### Code Example
```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

# One-Hot Encoding (for features)
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[['city']])

# Ordinal Encoding (for ordered categories)
encoder = OrdinalEncoder(categories=[['small', 'medium', 'large']])
X_encoded = encoder.fit_transform(X[['size']])

# Label Encoding (for target variable)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
```

### In ColumnTransformer
```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(), categorical_cols)
])
```

---

## Question 4
- [ ] Done

**How do you split data for training and testing?**

### train_test_split
```python
from sklearn.model_selection import train_test_split

# Basic split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # Reproducibility
    stratify=y          # Maintain class proportions
)

# With validation set
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)
# Result: 60% train, 20% val, 20% test
```

### Cross-Validation (Better)
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"Mean accuracy: {scores.mean():.4f}")
```

---

## Question 5
- [ ] Done

**How do you perform feature selection?**

### Feature Selection Methods

| Method | Type | Use Case |
|--------|------|----------|
| `SelectKBest` | Filter | Quick, based on statistics |
| `RFE` | Wrapper | Uses model performance |
| `SelectFromModel` | Embedded | Uses feature importances |

### Code Example
```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier

# Method 1: SelectKBest (statistical test)
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
print(f"Selected features: {selector.get_support()}")

# Method 2: RFE (Recursive Feature Elimination)
model = RandomForestClassifier()
rfe = RFE(model, n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)

# Method 3: SelectFromModel (based on importance)
model = RandomForestClassifier().fit(X, y)
selector = SelectFromModel(model, prefit=True)
X_selected = selector.transform(X)

# Get feature importances
importances = model.feature_importances_
```

---

## Question 6
- [ ] Done

**How do you save and load a trained model?**

### Using joblib (Recommended)
```python
import joblib

# Save model
joblib.dump(model, 'model.joblib')

# Load model
loaded_model = joblib.load('model.joblib')
predictions = loaded_model.predict(X_new)
```

### Using pickle
```python
import pickle

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

### Save Complete Pipeline
```python
from sklearn.pipeline import Pipeline

# Create and train pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)

# Save entire pipeline (includes preprocessing!)
joblib.dump(pipeline, 'full_pipeline.joblib')

# Load and use directly on raw data
pipeline = joblib.load('full_pipeline.joblib')
predictions = pipeline.predict(X_new)  # Handles scaling automatically
```

---

## Question 7
- [ ] Done

**How do you implement custom transformers?**

### Creating Custom Transformer
```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomScaler(BaseEstimator, TransformerMixin):
    """Custom transformer that scales to [0, 100] range."""
    
    def __init__(self, multiply_by=100):
        self.multiply_by = multiply_by
    
    def fit(self, X, y=None):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self
    
    def transform(self, X):
        X_scaled = (X - self.min_) / (self.max_ - self.min_)
        return X_scaled * self.multiply_by

# Use in pipeline
pipeline = Pipeline([
    ('custom_scaler', CustomScaler(multiply_by=100)),
    ('model', LogisticRegression())
])
```

### FunctionTransformer (Simple Cases)
```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# Wrap any function
log_transformer = FunctionTransformer(np.log1p)
X_log = log_transformer.fit_transform(X)
```

---

## Question 8
- [ ] Done

**How do you handle multiclass classification?**

### Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| One-vs-Rest (OvR) | N binary classifiers | Default for most |
| One-vs-One (OvO) | N(N-1)/2 classifiers | SVM with many classes |
| Native | Algorithm handles natively | Decision Trees, RF |

### Code Example
```python
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC

# One-vs-Rest
ovr = OneVsRestClassifier(SVC(kernel='linear'))
ovr.fit(X_train, y_train)

# One-vs-One
ovo = OneVsOneClassifier(SVC(kernel='rbf'))
ovo.fit(X_train, y_train)

# Native multiclass (no wrapper needed)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()  # Handles multiclass natively
rf.fit(X_train, y_train)
```

### Multiclass Metrics
```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
# Shows precision, recall, F1 for each class
```

---

## Question 9
- [ ] Done

**How do you implement ensemble methods?**

### Ensemble Types

| Type | Method | Key Parameter |
|------|--------|---------------|
| **Bagging** | Random Forest | `n_estimators` |
| **Boosting** | GradientBoosting, AdaBoost | `learning_rate` |
| **Voting** | VotingClassifier | `voting='hard'/'soft'` |
| **Stacking** | StackingClassifier | `final_estimator` |

### Code Example
```python
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression

# Bagging: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Boosting: Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

# Voting: Combine multiple models
voting = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier()),
        ('lr', LogisticRegression())
    ],
    voting='soft'  # Use probabilities
)

# Stacking: Use meta-learner
stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier())
    ],
    final_estimator=LogisticRegression()
)
```

---

## Question 10
- [ ] Done

**How do you perform dimensionality reduction?**

### Methods

| Method | Type | Use Case |
|--------|------|----------|
| PCA | Linear | General reduction |
| t-SNE | Non-linear | Visualization |
| LDA | Supervised | Classification |

### Code Example
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# PCA: Reduce to n components
pca = PCA(n_components=0.95)  # Keep 95% variance
X_reduced = pca.fit_transform(X)
print(f"Components kept: {pca.n_components_}")

# Explained variance
print(f"Variance ratio: {pca.explained_variance_ratio_}")

# t-SNE: For visualization (2D)
tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(X)

# LDA: Supervised dimensionality reduction
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
```

### PCA in Pipeline
```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50)),
    ('model', LogisticRegression())
])
```


---

## Question 11
- [ ] Done

**What preprocessing steps would you take before inputting data into a machine learning algorithm?**

**Answer:**

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# === Essential Preprocessing Steps ===

# 1. Handle Missing Values
imputer_num = SimpleImputer(strategy='median')      # or mean, most_frequent
imputer_cat = SimpleImputer(strategy='most_frequent')

# 2. Encode Categorical Variables
encoder = OneHotEncoder(drop='first', sparse_output=False)  # Nominal
label_enc = LabelEncoder()                                   # Ordinal / target

# 3. Scale/Normalize Features
scaler = StandardScaler()          # Z-score: mean=0, std=1
# MinMaxScaler()                   # [0, 1] range
# RobustScaler()                   # Uses median/IQR (robust to outliers)

# 4. Feature Engineering & Selection
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)

# === Complete Pipeline ===
numeric_features = ['age', 'income', 'score']
categorical_features = ['city', 'gender']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first'))
    ]), categorical_features)
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)
```

| Step | Why | Scikit-Learn |
|------|-----|--------------|
| Missing values | Algorithms can't handle NaN | `SimpleImputer`, `KNNImputer` |
| Encoding | ML needs numeric input | `OneHotEncoder`, `OrdinalEncoder` |
| Scaling | Equal feature contribution | `StandardScaler`, `MinMaxScaler` |
| Outliers | Distort model training | `RobustScaler`, IQR filtering |
| Feature selection | Reduce dimensionality | `SelectKBest`, `RFE` |

> **Interview Tip:** Always use a **Pipeline** to avoid data leakage — preprocessing must be fit on training data only, then applied to test data.

---

## Question 12
- [ ] Done

**How do you perform cross-validation using Scikit-Learn?**

**Answer:**

```python
from sklearn.model_selection import (
    cross_val_score, cross_validate, KFold, StratifiedKFold,
    LeaveOneOut, RepeatedKFold
)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

# === Method 1: cross_val_score (simplest) ===
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

# === Method 2: cross_validate (multiple metrics) ===
results = cross_validate(model, X, y, cv=5,
    scoring=['accuracy', 'f1', 'roc_auc'],
    return_train_score=True)
print(f"Test Acc: {results['test_accuracy'].mean():.3f}")
print(f"Test F1:  {results['test_f1'].mean():.3f}")

# === Method 3: Custom CV strategies ===

# K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Stratified K-Fold (preserves class distribution)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Leave-One-Out
loo = LeaveOneOut()

# Repeated K-Fold (more robust estimate)
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

# === Method 4: Manual loop (when you need predictions) ===
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(model, X, y, cv=5)  # Out-of-fold predictions
```

| CV Strategy | Use Case |
|-------------|----------|
| `KFold` | General purpose |
| `StratifiedKFold` | Classification (preserves class ratio) |
| `LeaveOneOut` | Small datasets |
| `TimeSeriesSplit` | Time-series (no future leakage) |
| `GroupKFold` | Grouped data (no group in both train/test) |

> **Interview Tip:** Always use **StratifiedKFold** for classification. Use **TimeSeriesSplit** for temporal data. Cross-validation gives a more reliable performance estimate than a single train/test split.

---

## Question 13
- [ ] Done

**What metrics can be used in Scikit-Learn to assess the performance of a regression model versus a classification model?**

**Answer:**

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, log_loss
)

print(classification_report(y_true, y_pred))

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
auc = roc_auc_score(y_true, y_prob)  # Needs probabilities
cm = confusion_matrix(y_true, y_pred)
```

### Regression Metrics

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)

mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)
```

| Classification | Best For | Regression | Best For |
|---------------|----------|------------|----------|
| Accuracy | Balanced classes | MSE/RMSE | General purpose |
| Precision | Minimize false positives | MAE | Robust to outliers |
| Recall | Minimize false negatives | R² | Explained variance |
| F1 | Imbalanced classes | MAPE | Relative error |
| AUC-ROC | Ranking quality | Explained Variance | Model quality |
| Log Loss | Probability calibration | | |

> **Interview Tip:** For imbalanced classification, **never use accuracy** — use F1, precision-recall AUC, or balanced accuracy. For regression with outliers, prefer **MAE** over **MSE**.

---

## Question 14
- [ ] Done

**How are hyperparameters tuned in Scikit-Learn?**

**Answer:**

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform

model = RandomForestClassifier()

# === Method 1: Grid Search (exhaustive) ===
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    model, param_grid,
    cv=5, scoring='f1',
    n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")
best_model = grid_search.best_estimator_

# === Method 2: Randomized Search (faster) ===
param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 30),
    'min_samples_split': randint(2, 20),
    'max_features': uniform(0.1, 0.9)
}

random_search = RandomizedSearchCV(
    model, param_distributions,
    n_iter=100, cv=5, scoring='f1',
    n_jobs=-1, random_state=42
)
random_search.fit(X_train, y_train)

# === Method 3: Halving Search (efficient) ===
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

halving_search = HalvingGridSearchCV(
    model, param_grid, cv=5,
    factor=3, scoring='f1'
)
halving_search.fit(X_train, y_train)
```

| Method | Combinations | Speed | Best For |
|--------|-------------|-------|----------|
| GridSearchCV | All combos | Slowest | Small param space |
| RandomizedSearchCV | Random subset | Faster | Large param space |
| HalvingGridSearchCV | Progressive | Fastest | Very large space |
| Bayesian (Optuna) | Guided | Smart | Complex models |

> **Interview Tip:** Start with **RandomizedSearchCV** (faster), narrow down, then use **GridSearchCV** on a smaller range. For deep learning, use **Optuna** or **Ray Tune** for Bayesian optimization.

---

## Question 15
- [ ] Done

**How do you monitor the performance of a Scikit-Learn model in production?**

**Answer:**

### Monitoring Strategy

| Aspect | What to Monitor | Tools |
|--------|----------------|-------|
| **Data drift** | Feature distribution changes | Evidently AI, Great Expectations |
| **Model drift** | Prediction accuracy degradation | MLflow, Weights & Biases |
| **Concept drift** | Relationship between features & target changes | Custom statistical tests |
| **Performance** | Latency, throughput | Prometheus, Grafana |
| **Infrastructure** | Memory, CPU usage | CloudWatch, DataDog |

```python
import numpy as np
from scipy import stats

# === 1. Data Drift Detection ===
def detect_data_drift(reference_data, new_data, threshold=0.05):
    """KS test to detect distribution shift."""
    drift_results = {}
    for col in reference_data.columns:
        stat, p_value = stats.ks_2samp(reference_data[col], new_data[col])
        drift_results[col] = {
            'statistic': stat,
            'p_value': p_value,
            'drift_detected': p_value < threshold
        }
    return drift_results

# === 2. Performance Monitoring ===
def monitor_predictions(model, X_new, y_true=None):
    y_pred = model.predict(X_new)
    y_prob = model.predict_proba(X_new)[:, 1]
    
    metrics = {
        'prediction_mean': np.mean(y_pred),
        'prediction_std': np.std(y_pred),
        'confidence_mean': np.mean(np.max(model.predict_proba(X_new), axis=1))
    }
    
    if y_true is not None:
        from sklearn.metrics import accuracy_score, f1_score
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred)
    return metrics

# === 3. Model Versioning with MLflow ===
import mlflow
mlflow.sklearn.log_model(model, "model")
mlflow.log_metrics({"accuracy": 0.95, "f1": 0.92})
```

> **Interview Tip:** The three types of drift: **data drift** (input distribution changes), **concept drift** (relationship X→y changes), **model drift** (performance degrades). Automated retraining pipelines trigger when drift exceeds thresholds.

---

## Question 16
- [ ] Done

**What recent advancements in machine learning are not yet fully supported by Scikit-Learn?**

**Answer:**

| Advancement | Scikit-Learn Status | Alternative |
|------------|--------------------|--------------|
| **Deep learning** | Not supported | TensorFlow, PyTorch |
| **GPU acceleration** | Minimal (experimental) | cuML (RAPIDS), XGBoost GPU |
| **Transformers / LLMs** | Not supported | HuggingFace, OpenAI |
| **Reinforcement learning** | Not supported | Stable Baselines3, RLlib |
| **Graph neural networks** | Not supported | PyG, DGL |
| **AutoML** | Limited (HalvingSearch) | Auto-sklearn, FLAML, H2O |
| **Online learning** | `partial_fit` (limited) | River, Vowpal Wabbit |
| **Federated learning** | Not supported | PySyft, Flower |
| **Explainability** | Basic (feature importance) | SHAP, LIME, Captum |
| **Probabilistic models** | Limited | PyMC, Stan, GPyTorch |
| **Time-series** | No dedicated support | Prophet, statsmodels, Darts |
| **Large-scale distributed** | Not supported | Spark MLlib, Dask-ML |

### Why Scikit-Learn Remains Relevant

1. **Best for classical ML** — random forests, SVMs, clustering, preprocessing
2. **Consistent API** — `fit/predict/transform` pattern adopted by all libraries
3. **Production-ready pipelines** — `Pipeline` + `ColumnTransformer`
4. **Excellent documentation** and community
5. **Lightweight** — no GPU dependencies

> **Interview Tip:** Scikit-Learn excels at **classical ML** and **preprocessing**. For deep learning, NLP, or GPU-intensive tasks, it's combined with specialized libraries. Its API design (`fit/predict`) is the de facto standard across the ecosystem.

---

## Question 17
- [ ] Done

**What role do libraries like joblib play in the context of Scikit-Learn?**

**Answer:**

**joblib** is the backbone library for **parallelism** and **model persistence** in Scikit-Learn.

### 1. Model Persistence (Save/Load)

```python
import joblib
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier().fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.joblib')          # Faster than pickle for NumPy arrays
joblib.dump(model, 'model.joblib.gz', compress=3)  # Compressed

# Load model
model = joblib.load('model.joblib')
y_pred = model.predict(X_test)

# vs pickle (joblib is 10-100× faster for large models)
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### 2. Parallel Computing

```python
from joblib import Parallel, delayed

# Parallel processing
def process_item(item):
    return item ** 2

results = Parallel(n_jobs=-1)(  # -1 = all CPUs
    delayed(process_item)(i) for i in range(1000)
)

# Scikit-Learn uses joblib internally:
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)  # Parallel CV

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)  # Parallel trees
```

### 3. Memory Caching

```python
from joblib import Memory
memory = Memory('./cache_dir', verbose=0)

@memory.cache
def expensive_computation(data):
    # Results are cached to disk
    return heavy_processing(data)

# Used in Scikit-Learn pipelines
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('preprocessing', preprocessor),
    ('model', classifier)
], memory='./cache_dir')  # Cache transform results
```

| Feature | joblib | pickle |
|---------|--------|--------|
| Speed (large arrays) | 10-100× faster | Slower |
| Compression | Built-in | Manual |
| Parallel computing | Yes (`Parallel`) | No |
| Memory caching | Yes (`Memory`) | No |
| NumPy optimization | Yes | No |

> **Interview Tip:** Use `joblib.dump()` over `pickle` for Scikit-Learn models — it's optimized for NumPy arrays. The `n_jobs=-1` parameter in Scikit-Learn uses joblib for parallelism.
