# Python ML Interview Questions - Scenario-Based Questions

## Question 1
- [ ] Done

**Discuss the difference between a list , a tuple , and a set in Python**

### Definition
Lists, tuples, and sets are Python's core collection data structures, each with distinct characteristics for mutability, ordering, and uniqueness.

### Comparison Table

| Feature | List | Tuple | Set |
|---------|------|-------|-----|
| Syntax | `[1, 2, 3]` | `(1, 2, 3)` | `{1, 2, 3}` |
| Mutable | Yes | No | Yes |
| Ordered | Yes | Yes | No (Python 3.7+ maintains insertion order) |
| Duplicates | Allowed | Allowed | Not Allowed |
| Indexing | Yes | Yes | No |
| Hashable | No | Yes | No (but elements must be hashable) |
| Use Case | General purpose collection | Fixed data, dictionary keys | Unique elements, membership testing |

### When to Use Each

**List** - When you need:
- Ordered collection that can change
- Allow duplicates
- Index-based access

```python
features = ['age', 'income', 'score']
features.append('education')  # Modifiable
```

**Tuple** - When you need:
- Immutable data (coordinates, RGB values)
- Dictionary keys
- Return multiple values from function
- Memory efficiency

```python
point = (10, 20)  # Coordinates
rgb = (255, 128, 0)  # Color
# point[0] = 5  # Error! Immutable
```

**Set** - When you need:
- Unique elements only
- Fast membership testing O(1)
- Set operations (union, intersection)

```python
unique_labels = {1, 2, 3, 2, 1}  # Becomes {1, 2, 3}
print(2 in unique_labels)  # O(1) lookup
```

### ML Use Case
```python
# List: Store features
features = ['f1', 'f2', 'f3']

# Tuple: Return train/test split
def split_data(X, y):
    return X_train, X_test, y_train, y_test  # Returns tuple

# Set: Find unique classes
unique_classes = set(y_labels)
```

---

## Question 2
- [ ] Done

**Discuss the usage of *args and **kwargs in function definitions.**

### Definition
- `*args`: Allows passing variable number of **positional arguments** (collected as tuple)
- `**kwargs`: Allows passing variable number of **keyword arguments** (collected as dictionary)

### Code Examples

```python
# *args example
def sum_all(*args):
    """Accept any number of positional arguments."""
    print(f"args is a tuple: {args}")
    return sum(args)

result = sum_all(1, 2, 3, 4, 5)
# args is a tuple: (1, 2, 3, 4, 5)
# result = 15


# **kwargs example
def create_model(**kwargs):
    """Accept any number of keyword arguments."""
    print(f"kwargs is a dict: {kwargs}")
    for key, value in kwargs.items():
        print(f"  {key} = {value}")

create_model(learning_rate=0.01, epochs=100, batch_size=32)
# kwargs is a dict: {'learning_rate': 0.01, 'epochs': 100, 'batch_size': 32}


# Combined example
def flexible_function(required, *args, default=10, **kwargs):
    """Shows argument order: required -> *args -> defaults -> **kwargs"""
    print(f"required: {required}")
    print(f"args: {args}")
    print(f"default: {default}")
    print(f"kwargs: {kwargs}")

flexible_function("must have", 1, 2, 3, default=20, extra="value")
```

### ML Use Case
```python
# Wrapper function for model training
def train_model(model_class, X, y, *preprocessing_steps, **hyperparams):
    """
    model_class: The ML model class
    *preprocessing_steps: Variable preprocessing functions
    **hyperparams: Model hyperparameters
    """
    # Apply preprocessing
    for step in preprocessing_steps:
        X = step(X)
    
    # Create and train model with hyperparameters
    model = model_class(**hyperparams)
    model.fit(X, y)
    return model
```

---

## Question 3
- [ ] Done

**Discuss the benefits of using Jupyter Notebooks for machine learning projects.**

### Key Benefits

**1. Interactive Development**
- Execute code cell by cell
- See immediate output/visualizations
- Experiment and iterate quickly

**2. Documentation + Code Together**
- Markdown cells for explanations
- Code cells for implementation
- Creates reproducible research

**3. Visualization Integration**
- Inline plots with Matplotlib/Seaborn
- Interactive widgets
- Rich output (images, HTML, LaTeX)

**4. Exploratory Data Analysis (EDA)**
- Display DataFrames directly
- Quick statistical summaries
- Iterative data exploration

**5. Prototyping and Experimentation**
- Test different approaches quickly
- Share results with stakeholders
- Export to various formats (HTML, PDF, Python script)

### Limitations
- Version control is difficult (JSON format)
- Not ideal for production code
- Can lead to hidden state issues
- Transition to .py files for deployment

---

## Question 4
- [ ] Done

**Discuss the use of pipelines in Scikit-learn for streamlining preprocessing steps.**

### Definition
A Pipeline chains multiple preprocessing steps and a model into a single object, ensuring consistent data flow and preventing data leakage.

### Benefits
1. **Prevents Data Leakage**: Fit preprocessing only on training data
2. **Cleaner Code**: Single object instead of multiple steps
3. **Easy Cross-Validation**: CV properly handles preprocessing
4. **Simpler Deployment**: One object to save and load

### Code Example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Define column types
numeric_features = ['age', 'income']
categorical_features = ['city', 'gender']

# Create preprocessing pipelines for each type
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine into ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create full pipeline with model
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Use the pipeline
full_pipeline.fit(X_train, y_train)
predictions = full_pipeline.predict(X_test)
```

### Interview Tip
Always mention that pipelines prevent data leakage - preprocessing is fit only on training folds during cross-validation.

---

## Question 5
- [ ] Done

**Discuss how ensemble methods work and give an example where they might be useful.**

### Definition
Ensemble methods combine multiple models to create a stronger predictor. The key insight: diverse models make different errors that can cancel out.

### Types of Ensemble Methods

| Method | How It Works | Reduces |
|--------|-------------|---------|
| **Bagging** | Train models on random subsets (with replacement) | Variance |
| **Boosting** | Train sequentially, focus on previous errors | Bias |
| **Stacking** | Use meta-model to combine base model predictions | Both |

### Scenario: Credit Card Fraud Detection

**Why Ensemble Works Here:**
- High-stakes decision (false negatives are costly)
- Complex patterns in data
- Need robust predictions

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

# Create diverse base models
models = [
    ('lr', LogisticRegression()),
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier(n_estimators=100))
]

# Combine with voting
ensemble = VotingClassifier(estimators=models, voting='soft')
ensemble.fit(X_train, y_train)

# Ensemble often outperforms individual models
print(f"Ensemble Accuracy: {ensemble.score(X_test, y_test):.3f}")
```

---

## Question 6
- [ ] Done

**How would you assess a model’s performance ? Mention at least three metrics**

### Classification Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| Accuracy | $(TP + TN) / Total$ | Balanced classes |
| Precision | $TP / (TP + FP)$ | High cost of false positive |
| Recall | $TP / (TP + FN)$ | High cost of false negative |
| F1-Score | $2 \times \frac{P \times R}{P + R}$ | Imbalanced data |
| ROC-AUC | Area under TPR vs FPR curve | Threshold-independent |

### Regression Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| MSE | $\frac{1}{n}\sum(y - \hat{y})^2$ | Penalizes large errors |
| MAE | $\frac{1}{n}\sum abs(y - \hat{y})$ | Robust to outliers |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | Variance explained |

### Code Example
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
print(f"Precision: {precision_score(y_true, y_pred):.3f}")
print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
print(f"F1:        {f1_score(y_true, y_pred):.3f}")
```

### Interview Tip
Always ask: "What is the business cost of different types of errors?" This determines which metric to prioritize.


---

## Question 7
- [ ] Done

**Discuss the differences between supervised and unsupervised learning evaluation.**

**Answer:**

### Fundamental Difference

| Aspect | Supervised Learning | Unsupervised Learning |
|--------|--------------------|-----------------------|
| **Labels** | Ground truth available | No ground truth |
| **Goal** | Predict correct output | Discover hidden patterns |
| **Evaluation** | Compare predictions to labels | Measure internal structure quality |

### Supervised Learning Metrics

| Task | Metrics |
|------|---------|
| **Classification** | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC |
| **Regression** | MSE, RMSE, MAE, R², MAPE |
| **Validation** | Cross-validation, train/test split, learning curves |

### Unsupervised Learning Metrics

| Metric | Type | What It Measures |
|--------|------|------------------|
| **Silhouette Score** | Internal | Cluster cohesion vs. separation (-1 to 1) |
| **Inertia (WCSS)** | Internal | Within-cluster sum of squares |
| **Davies-Bouldin Index** | Internal | Ratio of within-cluster to between-cluster distances |
| **Calinski-Harabasz** | Internal | Ratio of between-cluster to within-cluster variance |
| **Adjusted Rand Index** | External | Agreement with true labels (if available) |
| **NMI** | External | Mutual information between clusters and true labels |

### Code Comparison

```python
from sklearn.metrics import (accuracy_score, silhouette_score,
                              adjusted_rand_score, davies_bouldin_score)

# Supervised
print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")

# Unsupervised (internal — no labels needed)
print(f"Silhouette: {silhouette_score(X, cluster_labels):.3f}")
print(f"Davies-Bouldin: {davies_bouldin_score(X, cluster_labels):.3f}")

# Unsupervised (external — if labels exist)
print(f"Adjusted Rand: {adjusted_rand_score(y_true, cluster_labels):.3f}")
```

### Elbow Method for Clustering

```python
from sklearn.cluster import KMeans

inertias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    inertias.append(km.inertia_)
# Plot inertias → find "elbow" point
```

> **Interview Tip:** Unsupervised evaluation is inherently harder because there’s no "correct answer." Combining **internal metrics** (silhouette) with **domain expertise** (do clusters make business sense?) is the best approach.

---

## Question 8
- [ ] Done

**How would you approach feature selection in a large dataset?**

**Answer:**

### Feature Selection Methods

| Category | Method | Description |
|----------|--------|-------------|
| **Filter** | Correlation | Remove features highly correlated with each other |
| **Filter** | Variance Threshold | Drop near-zero variance features |
| **Filter** | Chi-squared / ANOVA | Statistical test for relevance to target |
| **Wrapper** | Recursive Feature Elimination (RFE) | Iteratively remove weakest features |
| **Wrapper** | Forward/Backward Selection | Add/remove features greedily |
| **Embedded** | Lasso (L1) | Regularization shrinks coefficients to zero |
| **Embedded** | Tree-based importance | Feature importance from Random Forest/XGBoost |

### Implementation

```python
import pandas as pd
import numpy as np
from sklearn.feature_selection import (VarianceThreshold, SelectKBest,
                                       f_classif, RFE, mutual_info_classif)
from sklearn.ensemble import RandomForestClassifier

# 1. Remove low-variance features
selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X)

# 2. Remove highly correlated features
corr_matrix = pd.DataFrame(X).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]

# 3. Statistical selection (top k features)
selector = SelectKBest(f_classif, k=20)
X_best = selector.fit_transform(X, y)
selected_features = np.array(feature_names)[selector.get_support()]

# 4. Mutual Information
mi_scores = mutual_info_classif(X, y)
mi_ranking = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)

# 5. Tree-based importance
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=feature_names)
top_features = importances.nlargest(20).index.tolist()

# 6. RFE (Recursive Feature Elimination)
rfe = RFE(estimator=rf, n_features_to_select=15, step=5)
rfe.fit(X, y)
rfe_features = np.array(feature_names)[rfe.support_]

# 7. L1 Regularization (Lasso)
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5).fit(X, y)
lasso_features = np.array(feature_names)[lasso.coef_ != 0]
```

### Decision Flow

```
Start → Remove zero-variance → Remove high correlation (>0.95)
  → Choose method:
      Small dataset? → RFE (thorough but slow)
      Large dataset? → Tree importance (fast)
      Linear model?  → Lasso L1 (automatic)
```

> **Interview Tip:** Use **multiple methods** and take the intersection of selected features for robustness. Always evaluate the final model with and without feature selection to confirm improvement.

---

## Question 9
- [ ] Done

**Discuss strategies for dealing with imbalanced datasets.**

**Answer:**

### The Problem

Imbalanced datasets have a skewed class distribution (e.g., 98% negative, 2% positive), causing models to predict the majority class and achieve high accuracy while failing on the minority class.

### Strategies

| Strategy | Level | Description |
|----------|-------|-------------|
| **SMOTE** | Data | Synthetic Minority Oversampling |
| **Random Undersampling** | Data | Remove majority class samples |
| **Random Oversampling** | Data | Duplicate minority class samples |
| **Class Weights** | Algorithm | Penalize misclassifying minority class more |
| **Threshold Tuning** | Post-hoc | Adjust decision threshold from 0.5 |
| **Ensemble** | Algorithm | BalancedRandomForest, EasyEnsemble |
| **Anomaly Detection** | Algorithm | Treat minority as anomaly (Isolation Forest) |
| **Cost-Sensitive Learning** | Algorithm | Different misclassification costs |
| **Collect More Data** | Data | Best solution if possible |

### Implementation

```python
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. SMOTE (most popular)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# 2. Class Weights (no resampling needed)
model = RandomForestClassifier(
    class_weight='balanced',  # auto-adjusts weights inversely proportional to frequency
    n_estimators=200
)
model.fit(X_train, y_train)

# 3. Threshold Tuning
y_proba = model.predict_proba(X_test)[:, 1]

# Find optimal threshold using F1
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
optimal_threshold = thresholds[np.argmax(f1_scores)]
y_pred_tuned = (y_proba >= optimal_threshold).astype(int)

# 4. Combined approach (SMOTE + Tomek links)
combined = SMOTETomek(random_state=42)
X_res, y_res = combined.fit_resample(X_train, y_train)

# 5. Always evaluate with proper metrics
print(classification_report(y_test, y_pred_tuned))
# Use ROC-AUC or PR-AUC, NOT accuracy
```

### Method Selection Guide

```
Imbalance ratio < 1:10  → Class weights
Imbalance ratio 1:10-1:100 → SMOTE + class weights
Imbalance ratio > 1:100 → Anomaly detection approach
```

> **Interview Tip:** SMOTE should only be applied to **training data** (never test/validation). Always evaluate with **Precision-Recall AUC** instead of accuracy on imbalanced datasets.

---

## Question 10
- [ ] Done

**Discuss the importance of model persistence and demonstrate how to save and load models in Python.**

**Answer:**

### Why Model Persistence?

| Reason | Description |
|--------|------------|
| **Deployment** | Serve predictions without retraining |
| **Reproducibility** | Reload exact model for auditing |
| **Version control** | Track model versions over time |
| **Efficiency** | Training takes hours; loading takes seconds |
| **Sharing** | Distribute trained models to team/clients |

### Methods for Saving Models

```python
import joblib
import pickle

# ========== 1. Joblib (Recommended for Scikit-learn) ==========
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# Save
joblib.dump(model, 'model_rf.joblib')

# Load
loaded_model = joblib.load('model_rf.joblib')
y_pred = loaded_model.predict(X_test)

# ========== 2. Pickle (Python standard library) ==========
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# ========== 3. TensorFlow/Keras ==========
import tensorflow as tf

# Save entire model (architecture + weights + optimizer)
model.save('my_model.h5')
# or SavedModel format
model.save('saved_model_dir')

# Load
loaded = tf.keras.models.load_model('my_model.h5')

# ========== 4. PyTorch ==========
import torch

# Save state dict (recommended)
torch.save(model.state_dict(), 'model_weights.pth')

# Load
model = MyModel()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# ========== 5. ONNX (Cross-framework) ==========
import onnx
from skl2onnx import convert_sklearn

onnx_model = convert_sklearn(model, initial_types=[('input', FloatTensorType([None, n_features]))])
onnx.save_model(onnx_model, 'model.onnx')
```

### Save Entire Pipeline

```python
# Save model + preprocessor together
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'full_pipeline.joblib')  # saves everything
```

### Comparison

| Method | Pros | Cons |
|--------|------|------|
| **Joblib** | Best for large NumPy arrays (sklearn) | Python-only |
| **Pickle** | Standard library, works everywhere | Security risk (untrusted sources) |
| **HDF5/SavedModel** | TF standard, includes architecture | TF-specific |
| **ONNX** | Cross-framework, production-ready | Extra conversion step |

> **Interview Tip:** Always save the **preprocessing pipeline** along with the model. Version your models with timestamps: `model_v2_2024-01-15.joblib`. Mention **MLflow** for production model registry and tracking.

---

## Question 11
- [ ] Done

**Discuss the impact of the GIL (Global Interpreter Lock) on Python concurrency in machine learning applications.**

**Answer:**

### What is the GIL?

The **Global Interpreter Lock (GIL)** is a mutex in CPython that allows only **one thread to execute Python bytecode at a time**, even on multi-core systems.

### Impact on ML

| Scenario | GIL Impact | Solution |
|----------|-----------|----------|
| **NumPy operations** | Minimal — NumPy releases GIL for C-level ops | Use NumPy vectorized ops |
| **Scikit-learn** | Minimal — `n_jobs=-1` uses multiprocessing | Set `n_jobs=-1` |
| **Custom Python loops** | Severe — threads don’t parallelize | Use `multiprocessing` |
| **I/O operations** | No impact — GIL released during I/O | Use `threading` |
| **Deep learning** | No impact — GPU ops are outside GIL | Use CUDA/GPU |

### How ML Libraries Bypass the GIL

```python
# 1. NumPy/SciPy — C extensions release GIL
import numpy as np
a = np.random.randn(10000, 10000)
b = a @ a.T  # GIL released, uses BLAS threads

# 2. Scikit-learn — Multiprocessing via joblib
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=-1)  # spawns separate processes

# 3. Multiprocessing — Bypass GIL with separate processes
from multiprocessing import Pool

def train_single_model(params):
    model = SomeModel(**params)
    return model.fit(X, y)

with Pool(4) as pool:
    results = pool.map(train_single_model, param_list)

# 4. Threading — Useful for I/O-bound tasks
from concurrent.futures import ThreadPoolExecutor

def load_image(path):
    return cv2.imread(path)  # I/O releases GIL

with ThreadPoolExecutor(max_workers=8) as executor:
    images = list(executor.map(load_image, image_paths))
```

### Summary

```
CPU-bound Python code  → GIL is a bottleneck → Use multiprocessing
CPU-bound C extensions  → GIL released       → NumPy, SciPy, Cython
I/O-bound tasks        → GIL released       → Use threading
GPU operations         → GIL irrelevant     → CUDA handles parallelism
```

> **Interview Tip:** The GIL is a CPython implementation detail, not a Python language limitation. Most ML libraries (NumPy, TensorFlow, PyTorch) bypass it through C extensions and multiprocessing. Python 3.13+ introduces a **free-threaded mode** (PEP 703) that removes the GIL experimentally.

---

## Question 12
- [ ] Done

**Discuss the role of the collections module in managing data structures for machine learning.**

**Answer:**

Python’s `collections` module provides specialized high-performance data structures beyond built-in types.

### Key Classes for ML

| Class | Purpose | ML Use Case |
|-------|---------|------------|
| **Counter** | Count element frequencies | Class distribution, word frequency |
| **defaultdict** | Dict with default values | Feature grouping, adjacency lists |
| **OrderedDict** | Dict preserving insertion order | Feature ordering, experiment tracking |
| **deque** | Fast append/pop from both ends | Sliding window, replay buffer (RL) |
| **namedtuple** | Lightweight immutable records | Data records, configuration |

### Code Examples

```python
from collections import Counter, defaultdict, deque, namedtuple

# 1. Counter — Class distribution analysis
y_labels = [0, 1, 0, 0, 1, 1, 2, 2, 0, 0]
class_dist = Counter(y_labels)
print(class_dist)  # Counter({0: 5, 1: 3, 2: 2})
print(class_dist.most_common(2))  # [(0, 5), (1, 3)]

# Check imbalance ratio
total = sum(class_dist.values())
for cls, count in class_dist.items():
    print(f"Class {cls}: {count/total:.2%}")

# 2. Counter — Word frequency (NLP)
from collections import Counter
words = "the cat sat on the mat the cat".split()
word_freq = Counter(words)
print(word_freq.most_common(3))  # [('the', 3), ('cat', 2), ('sat', 1)]

# 3. defaultdict — Group features by type
feature_types = defaultdict(list)
for feature, dtype in zip(feature_names, dtypes):
    feature_types[dtype].append(feature)
# {'float64': ['age', 'income'], 'object': ['city', 'gender']}

# 4. deque — Replay buffer for RL
replay_buffer = deque(maxlen=10000)
replay_buffer.append((state, action, reward, next_state, done))
batch = [replay_buffer[i] for i in np.random.choice(len(replay_buffer), 32)]

# 5. deque — Moving average (training loss)
from collections import deque
recent_losses = deque(maxlen=100)
for epoch in range(1000):
    loss = train_one_epoch()
    recent_losses.append(loss)
    avg_loss = sum(recent_losses) / len(recent_losses)

# 6. namedtuple — Experiment tracking
Experiment = namedtuple('Experiment', ['model', 'params', 'accuracy', 'f1'])
results = [
    Experiment('RF', {'n_estimators': 100}, 0.92, 0.89),
    Experiment('XGB', {'max_depth': 5}, 0.95, 0.93),
]
best = max(results, key=lambda x: x.f1)
```

> **Interview Tip:** `Counter` is the fastest way to analyze class distributions. `deque` with `maxlen` is essential for implementing fixed-size buffers in reinforcement learning and streaming applications.

---

## Question 13
- [ ] Done

**Discuss various options for deploying a machine learning model in Python.**

**Answer:**

### Deployment Options

| Method | Best For | Latency | Scalability |
|--------|----------|---------|-------------|
| **REST API (Flask/FastAPI)** | Real-time predictions | Low | Medium |
| **Batch Processing** | Periodic bulk predictions | N/A | High |
| **Serverless (AWS Lambda)** | Infrequent predictions | Cold start | Auto-scaling |
| **Containerized (Docker + K8s)** | Production systems | Low | Very high |
| **Edge Deployment** | Mobile/IoT devices | Very low | Per-device |
| **Streaming (Kafka + Spark)** | Real-time data streams | Medium | Very high |
| **MLaaS** | Quick prototyping | Varies | Managed |

### Implementation Examples

```python
# ========== 1. FastAPI (Recommended) ==========
from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()
model = joblib.load('model.joblib')

class PredictionRequest(BaseModel):
    features: list[float]

@app.post('/predict')
def predict(request: PredictionRequest):
    X = np.array(request.features).reshape(1, -1)
    prediction = model.predict(X)
    probability = model.predict_proba(X).max()
    return {'prediction': int(prediction[0]), 'confidence': float(probability)}

# Run: uvicorn app:app --host 0.0.0.0 --port 8000

# ========== 2. Batch Processing ==========
import pandas as pd

def batch_predict(input_path, output_path):
    df = pd.read_csv(input_path)
    model = joblib.load('model.joblib')
    df['prediction'] = model.predict(df[feature_columns])
    df.to_csv(output_path, index=False)

# Schedule with cron or Airflow

# ========== 3. Docker Deployment ==========
# Dockerfile
# FROM python:3.10-slim
# COPY requirements.txt model.joblib app.py ./
# RUN pip install -r requirements.txt
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# ========== 4. Edge Deployment (TensorFlow Lite) ==========
import tensorflow as tf

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Deployment Checklist

```
☐ Model serialization (joblib/ONNX/TFLite)
☐ Input validation & error handling
☐ API endpoint with health check
☐ Containerization (Docker)
☐ CI/CD pipeline
☐ Monitoring & logging
☐ A/B testing setup
☐ Model versioning
☐ Auto-scaling configuration
```

> **Interview Tip:** Mention **FastAPI over Flask** for ML (async support, auto-docs, type validation). For production, discuss **model monitoring** (data drift, performance degradation) and **shadow deployment** for safe rollouts.

---

## Question 14
- [ ] Done

**Discuss strategies for effective logging and monitoring in machine-learning applications.**

**Answer:**

### Why Logging & Monitoring for ML?

ML models degrade over time due to **data drift**, **concept drift**, and **infrastructure issues**. Unlike traditional software, ML failures are often silent (model returns predictions, just wrong ones).

### Logging Strategy

```python
import logging
import json
from datetime import datetime

# Structured logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ml_service')

# Log predictions with metadata
def log_prediction(request_id, features, prediction, confidence, latency_ms):
    log_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'request_id': request_id,
        'input_features': features,
        'prediction': prediction,
        'confidence': confidence,
        'latency_ms': latency_ms,
        'model_version': 'v2.1'
    }
    logger.info(json.dumps(log_data))
```

### Monitoring Dimensions

| Dimension | What to Monitor | Tools |
|-----------|----------------|-------|
| **Model Performance** | Accuracy, F1, drift over time | MLflow, Evidently |
| **Data Quality** | Missing values, schema violations, distribution shift | Great Expectations, Evidently |
| **Infrastructure** | CPU, memory, GPU utilization, latency | Prometheus, Grafana |
| **Predictions** | Prediction distribution, confidence scores | Custom dashboards |
| **Business Metrics** | Revenue impact, user engagement | Analytics platforms |

### Experiment Tracking with MLflow

```python
import mlflow

mlflow.set_experiment("churn_prediction")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "xgboost")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 5)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### Data Drift Detection

```python
from scipy.stats import ks_2samp

def detect_drift(reference_data, production_data, threshold=0.05):
    drift_report = {}
    for col in reference_data.columns:
        stat, p_value = ks_2samp(reference_data[col], production_data[col])
        drift_report[col] = {
            'statistic': stat,
            'p_value': p_value,
            'drift_detected': p_value < threshold
        }
    return drift_report
```

### Alerting Checklist

| Alert | Trigger |
|-------|---------|
| Model latency | > 500ms p95 |
| Prediction drift | Distribution shift from training |
| Error rate | > 1% failed predictions |
| Data quality | Missing values > 5% |
| Model staleness | Last retrain > 30 days |

> **Interview Tip:** Mention the **ML monitoring stack**: MLflow (experiments), Evidently (data/model drift), Prometheus + Grafana (infrastructure), and PagerDuty (alerting). The key difference from traditional monitoring is tracking **data drift** and **prediction quality**.

---

## Question 15
- [ ] Done

**Discuss the implications of quantum computing on machine learning, with a Python perspective.**

**Answer:**

### Quantum Computing & ML (QML)

Quantum computing uses **qubits** (which can be in superposition of 0 and 1 simultaneously) and **entanglement** to potentially solve certain computational problems exponentially faster than classical computers.

### Potential Impact on ML

| Area | Quantum Advantage | Status |
|------|-------------------|--------|
| **Optimization** | Faster convergence on complex loss landscapes | Research |
| **Sampling** | Efficient sampling from complex distributions | Promising |
| **Linear Algebra** | Quantum speedup for matrix operations (HHL) | Theoretical |
| **Kernel Methods** | Quantum kernels for SVM | Research |
| **Feature Spaces** | Map data to high-dimensional quantum spaces | Experimental |
| **Generative Models** | Quantum Boltzmann machines | Research |

### Python Quantum ML Libraries

```python
# ========== 1. Qiskit (IBM) ==========
from qiskit import QuantumCircuit
from qiskit_machine_learning.algorithms import QSVC
from qiskit.circuit.library import ZZFeatureMap

# Quantum Support Vector Classifier
feature_map = ZZFeatureMap(feature_dimension=2, reps=2)
qsvc = QSVC(feature_map=feature_map)
qsvc.fit(X_train, y_train)
accuracy = qsvc.score(X_test, y_test)

# ========== 2. PennyLane (Xanadu) ==========
import pennylane as qml
from pennylane import numpy as np

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def quantum_circuit(params, x):
    qml.RX(x[0], wires=0)
    qml.RY(x[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.Rot(*params[:3], wires=0)
    return qml.expval(qml.PauliZ(0))

# ========== 3. Cirq (Google) ==========
import cirq
qubit = cirq.GridQubit(0, 0)
circuit = cirq.Circuit(cirq.H(qubit), cirq.measure(qubit))
```

### Current Limitations

| Challenge | Description |
|-----------|------------|
| **Noise** | Current quantum computers are error-prone (NISQ era) |
| **Qubit count** | Limited qubits (100-1000) vs. millions needed |
| **Decoherence** | Quantum states decay quickly |
| **No proven speedup** | QML advantages not yet demonstrated at scale |
| **Data loading** | Encoding classical data into quantum states is expensive |

### Timeline

```
Now (NISQ era)     → Small experiments, hybrid quantum-classical
5-10 years          → Error-corrected qubits, practical QML
10+ years           → Large-scale quantum advantage for ML
```

> **Interview Tip:** Be honest — quantum ML is mostly **research-stage**. The near-term value is in **hybrid quantum-classical** approaches where quantum circuits handle specific subroutines. Mention **variational quantum circuits** as the most promising near-term QML approach.

---

## Question 16
- [ ] Done

**Discuss the integration of big data technologies with Python in machine learning projects.**

**Answer:**

### Why Big Data + Python?

When datasets exceed single-machine memory (>10GB), traditional Pandas/Scikit-learn workflows break down. Big data technologies enable distributed processing.

### Python Big Data Ecosystem

| Technology | Purpose | Python API |
|-----------|---------|------------|
| **Apache Spark** | Distributed computing | PySpark |
| **Dask** | Parallel Pandas/NumPy | dask.dataframe, dask.array |
| **Apache Kafka** | Real-time streaming | kafka-python, Faust |
| **Hadoop HDFS** | Distributed storage | hdfs, PyArrow |
| **Apache Airflow** | Workflow orchestration | Python-native |
| **Ray** | Distributed ML training | ray.train, ray.tune |
| **Vaex** | Out-of-core DataFrames | vaex (lazy evaluation) |
| **Polars** | Fast DataFrame library | polars (Rust-backed) |

### Integration Examples

```python
# ========== 1. PySpark for Large-Scale ML ==========
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName('ML').getOrCreate()
df = spark.read.parquet('hdfs:///data/large_dataset.parquet')

assembler = VectorAssembler(inputCols=['f1', 'f2', 'f3'], outputCol='features')
rf = RandomForestClassifier(numTrees=100, labelCol='label')
pipeline = Pipeline(stages=[assembler, rf])
model = pipeline.fit(df)

# ========== 2. Dask for Parallel Processing ==========
import dask.dataframe as dd

ddf = dd.read_csv('data_*.csv')  # reads multiple files in parallel
result = ddf.groupby('category').agg({'value': 'mean'}).compute()

# Dask-ML for distributed training
from dask_ml.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5)
search.fit(X_dask, y_dask)

# ========== 3. Ray for Distributed Training ==========
import ray
from ray import tune

ray.init()

def train_fn(config):
    model = XGBClassifier(**config)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    tune.report(accuracy=accuracy)

analysis = tune.run(train_fn, config={
    'max_depth': tune.choice([3, 5, 7, 10]),
    'n_estimators': tune.choice([100, 200, 500])
})
```

### Decision Guide

```
Data size < 10 GB    → Pandas + Scikit-learn
Data size 10-100 GB  → Dask or Polars
Data size > 100 GB   → PySpark on cluster
Real-time streaming   → Kafka + Spark Streaming
Distributed training  → Ray or Horovod
```

> **Interview Tip:** The key shift is from **single-machine** (Pandas) to **distributed** (PySpark/Dask). Mention that Spark MLlib provides distributed versions of common algorithms, and that **feature stores** (Feast, Tecton) bridge big data pipelines with ML models.

## Question 17
- [ ] Done

**Describe steps to take when a model performs well on the training data but poorly on new data**

### Diagnosing and Fixing Overfitting

When a model performs well on training data but poorly on new (test/validation) data, this is **overfitting** — the model has memorized noise rather than learning generalizable patterns.

### Step-by-Step Investigation

```python
import numpy as np
from sklearn.model_selection import cross_val_score, learning_curve
import matplotlib.pyplot as plt

# Step 1: Confirm overfitting with learning curves
def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
    plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
    plt.xlabel('Training Size'); plt.ylabel('Score')
    plt.legend(); plt.title('Learning Curve'); plt.show()

# Step 2: Cross-validation to measure generalization gap
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
train_score = model.fit(X_train, y_train).score(X_train, y_train)
cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
print(f"Train: {train_score:.3f}, CV: {cv_score:.3f}, Gap: {train_score - cv_score:.3f}")
```

### Remediation Strategies

| Strategy | Technique | Example |
|----------|-----------|--------|
| **More data** | Collect or augment | Data augmentation, SMOTE |
| **Reduce complexity** | Simpler model | Fewer layers, shallower trees |
| **Regularization** | Penalize large weights | L1/L2, dropout, early stopping |
| **Feature selection** | Remove noise features | Mutual information, RFE |
| **Ensemble methods** | Average multiple models | Bagging, boosting |
| **Cross-validation** | Better evaluation | K-fold, stratified CV |

```python
# Step 3: Apply regularization
from sklearn.linear_model import LogisticRegression

# Before (overfitting)
model_overfit = LogisticRegression(C=1e6, max_iter=1000)  # very low regularization

# After (regularized)
model_reg = LogisticRegression(C=0.1, penalty='l2', max_iter=1000)

# Step 4: Feature selection
from sklearn.feature_selection import SelectKBest, mutual_info_classif
selector = SelectKBest(mutual_info_classif, k=10)
X_selected = selector.fit_transform(X_train, y_train)
```

> **Interview Tip:** The core issue is the **bias-variance tradeoff**. Overfitting = low bias, high variance. Address it by increasing bias (simpler model, regularization) or reducing variance (more data, ensembles). Always plot learning curves first to confirm the diagnosis.

---

## Question 18
- [ ] Done

**Explain the use of regularization in linear models and provide a Python example**

### Regularization in Linear Models

Regularization adds a **penalty term** to the loss function to prevent overfitting by discouraging overly complex models (large coefficient values).

### Types of Regularization

| Type | Penalty Term | Effect | Model Name |
|------|-------------|--------|------------|
| **L1 (Lasso)** | $\lambda \sum \|w_i\|$ | Drives coefficients to exactly zero (feature selection) | `Lasso` |
| **L2 (Ridge)** | $\lambda \sum w_i^2$ | Shrinks coefficients toward zero (never exactly) | `Ridge` |
| **Elastic Net** | $\lambda_1 \sum \|w_i\| + \lambda_2 \sum w_i^2$ | Combines L1 and L2 benefits | `ElasticNet` |

### Python Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Generate data with many features (some irrelevant)
np.random.seed(42)
X = np.random.randn(100, 20)  # 20 features
y = X[:, 0] * 3 + X[:, 1] * (-2) + X[:, 2] * 1.5 + np.random.randn(100) * 0.5

# Compare models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=1.0),
    'Lasso (L1)': Lasso(alpha=0.1),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
}

for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
    pipe.fit(X, y)
    n_nonzero = np.sum(np.abs(pipe.named_steps['model'].coef_) > 1e-6)
    print(f"{name:25s} | R²: {scores.mean():.3f} ± {scores.std():.3f} | Non-zero coefs: {n_nonzero}")

# Output:
# Linear Regression         | R²: 0.89 ± 0.05 | Non-zero coefs: 20
# Ridge (L2)                | R²: 0.91 ± 0.04 | Non-zero coefs: 20
# Lasso (L1)                | R²: 0.93 ± 0.03 | Non-zero coefs: 3  ← feature selection!
# Elastic Net               | R²: 0.92 ± 0.03 | Non-zero coefs: 5

# Tune regularization strength with cross-validation
from sklearn.linear_model import RidgeCV, LassoCV
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]).fit(X, y)
print(f"Best Ridge alpha: {ridge_cv.alpha_}")
```

### When to Use Each
- **Ridge**: When all features are potentially relevant (correlated features)
- **Lasso**: When you suspect many features are irrelevant (automatic feature selection)
- **Elastic Net**: When features are correlated and you want feature selection

> **Interview Tip:** Always **scale features** before regularization (StandardScaler), since the penalty term is sensitive to feature magnitudes. Use `RidgeCV`/`LassoCV` for automatic hyperparameter tuning.

---

## Question 19
- [ ] Done

**What are the advantages of using Stochastic Gradient Descent over standard Gradient Descent ?**

### SGD vs. Batch Gradient Descent

| Aspect | Batch GD | Stochastic GD (SGD) | Mini-Batch GD |
|--------|----------|---------------------|---------------|
| **Per-step data** | Entire dataset | 1 random sample | Batch of 32-256 |
| **Speed per step** | Slow | Very fast | Fast |
| **Convergence** | Smooth, stable | Noisy, oscillating | Balanced |
| **Memory** | Full dataset in RAM | O(1) | O(batch_size) |
| **Local minima** | Can get stuck | Noise helps escape | Moderate escape |
| **Scalability** | Poor for large data | Excellent | Excellent |

### Key Advantages of SGD

1. **Scalability**: Processes one sample at a time — works with datasets that don't fit in memory
2. **Faster convergence (wall-clock)**: Updates weights much more frequently
3. **Escapes local minima**: Noise in gradient estimates helps explore the loss landscape
4. **Online learning**: Can learn from streaming data incrementally
5. **Regularization effect**: The inherent noise acts as implicit regularization

### Python Example

```python
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.datasets import make_classification
import time

X, y = make_classification(n_samples=100000, n_features=50, random_state=42)

# Batch approach
start = time.time()
lr = LogisticRegression(max_iter=1000).fit(X, y)
print(f"Logistic Regression: {time.time()-start:.2f}s, Acc: {lr.score(X, y):.4f}")

# SGD approach
start = time.time()
sgd = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42).fit(X, y)
print(f"SGD Classifier:      {time.time()-start:.2f}s, Acc: {sgd.score(X, y):.4f}")

# SGD with partial_fit for online learning
sgd_online = SGDClassifier(loss='log_loss')
for i in range(0, len(X), 1000):  # process in chunks
    sgd_online.partial_fit(X[i:i+1000], y[i:i+1000], classes=[0, 1])
```

### SGD Variants (Modern Optimizers)
| Optimizer | Key Idea |
|-----------|----------|
| **Momentum** | Accumulates past gradients for smoother updates |
| **RMSProp** | Adapts learning rate per parameter |
| **Adam** | Combines Momentum + RMSProp (default for deep learning) |
| **AdaGrad** | Larger updates for infrequent features |

> **Interview Tip:** In practice, **mini-batch SGD** (not pure SGD) is used because it leverages GPU parallelism. Modern deep learning always uses **Adam** or **AdamW** (Adam with weight decay). Mention that learning rate scheduling (warmup, cosine annealing) is critical for SGD convergence.

---

## Question 20
- [ ] Done

**Describe a situation where a machine learning model might fail, and how you would investigate the issue using Python**

### Common ML Failure Scenarios

| Failure Mode | Symptom | Root Cause |
|-------------|---------|------------|
| **Data drift** | Accuracy drops over time | Production data differs from training data |
| **Label leakage** | Unrealistically high accuracy | Target information leaks into features |
| **Class imbalance** | High accuracy, low recall for minority class | Model predicts majority class always |
| **Feature corruption** | Sudden performance drop | Missing values, schema changes, encoding errors |
| **Concept drift** | Gradual degradation | Underlying relationships change |

### Investigation Workflow with Python

```python
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Scenario: Model deployed for fraud detection, suddenly missing real fraud cases

# Step 1: Check data distributions (data drift)
def check_data_drift(train_df, prod_df, features):
    from scipy.stats import ks_2samp
    for col in features:
        stat, p_val = ks_2samp(train_df[col].dropna(), prod_df[col].dropna())
        if p_val < 0.05:
            print(f"⚠ DRIFT detected in '{col}': KS stat={stat:.3f}, p={p_val:.4f}")

# Step 2: Check for missing/corrupted features
def check_data_quality(df):
    print("Missing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("\nData types:")
    print(df.dtypes)
    print("\nUnique values per column:")
    print(df.nunique())

# Step 3: Evaluate per-class performance
def diagnose_predictions(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# Step 4: Check feature importance changes
def compare_feature_importance(model_old, model_new, feature_names):
    imp_old = model_old.feature_importances_
    imp_new = model_new.feature_importances_
    changes = pd.DataFrame({
        'feature': feature_names,
        'old_importance': imp_old,
        'new_importance': imp_new,
        'change': imp_new - imp_old
    }).sort_values('change', key=abs, ascending=False)
    print(changes.head(10))

# Step 5: Analyze error cases
def analyze_errors(X_test, y_true, y_pred):
    errors = X_test[y_true != y_pred]
    correct = X_test[y_true == y_pred]
    print("Error cases statistics:")
    print(errors.describe())
    print("\nCorrect cases statistics:")
    print(correct.describe())
```

### Systematic Debugging Checklist
1. **Verify data pipeline** — schema changes, missing values, encoding issues
2. **Compare distributions** — training vs. production data (KS test, PSI)
3. **Evaluate per-segment** — break down metrics by subgroups
4. **Check temporal patterns** — plot accuracy over time windows
5. **Retrain on recent data** — test if concept drift is the cause

> **Interview Tip:** Structure your answer as a systematic investigation: Data → Features → Model → Evaluation. Mention **monitoring tools** like Evidently AI, WhyLabs, or Great Expectations for production model monitoring.

---

## Question 21
- [ ] Done

**What are Python’s profiling tools and how do they assist in optimizing machine learning code ?**

### Python Profiling Tools for ML

| Tool | Type | Best For |
|------|------|----------|
| **`cProfile`** | CPU profiling | Function-level time analysis |
| **`line_profiler`** | Line-by-line profiling | Pinpointing slow lines |
| **`memory_profiler`** | Memory profiling | Finding memory leaks |
| **`py-spy`** | Sampling profiler | Production profiling (low overhead) |
| **`timeit`** | Benchmarking | Comparing small code snippets |
| **`%%time` / `%%timeit`** | Jupyter magic | Quick notebook benchmarks |
| **`scalene`** | CPU + Memory + GPU | All-in-one ML profiling |

### Usage Examples

```python
# 1. cProfile - overall function profiling
import cProfile
import pstats

def train_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=10000, n_features=50)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model.score(X, y)

cProfile.run('train_model()', sort='cumulative')

# 2. line_profiler - line-by-line analysis
# pip install line_profiler
# In Jupyter: %load_ext line_profiler
# %lprun -f train_model train_model()

# 3. memory_profiler - track memory usage
# pip install memory_profiler
from memory_profiler import profile

@profile
def memory_intensive():
    import numpy as np
    a = np.random.randn(10000, 10000)  # ~800 MB
    b = a @ a.T
    return b.mean()

# 4. timeit for benchmarking
import timeit
setup = "import numpy as np; a = np.random.randn(10000)"
print(timeit.timeit("np.sum(a)", setup=setup, number=1000))
print(timeit.timeit("sum(a)", setup=setup, number=1000))  # Much slower
```

> **Interview Tip:** Start with `cProfile` to find the bottleneck function, then use `line_profiler` to find the exact slow line. For ML pipelines, the bottleneck is usually **data loading** or **feature engineering**, not model training.

---

## Question 22
- [ ] Done

**Explain how unit tests and integration tests ensure the correctness of your machine learning code**

### Testing Strategy for ML Code

| Test Type | What It Tests | Example |
|-----------|--------------|--------|
| **Unit tests** | Individual functions in isolation | Feature engineering, data transforms |
| **Integration tests** | Components working together | Full pipeline: load → preprocess → train → predict |
| **Regression tests** | Model performance doesn't degrade | Accuracy stays above threshold |
| **Data validation** | Input data quality | Schema, ranges, distributions |

### Unit Tests for ML

```python
import pytest
import numpy as np
import pandas as pd

# ---- Feature Engineering Tests ----
def normalize(arr):
    """Min-max normalize an array."""
    return (arr - arr.min()) / (arr.max() - arr.min())

def test_normalize_basic():
    result = normalize(np.array([1, 2, 3, 4, 5]))
    assert result.min() == 0.0
    assert result.max() == 1.0
    assert len(result) == 5

def test_normalize_constant():
    """Edge case: all values identical."""
    with pytest.raises((ZeroDivisionError, RuntimeWarning)):
        normalize(np.array([5, 5, 5]))

def test_normalize_negative():
    result = normalize(np.array([-10, 0, 10]))
    np.testing.assert_array_almost_equal(result, [0, 0.5, 1.0])

# ---- Data Processing Tests ----
def clean_data(df):
    df = df.dropna(subset=['target'])
    df['age'] = df['age'].clip(0, 120)
    return df

def test_clean_data_removes_null_targets():
    df = pd.DataFrame({'age': [25, 30], 'target': [1, None]})
    result = clean_data(df)
    assert len(result) == 1
    assert result['target'].isna().sum() == 0

def test_clean_data_clips_age():
    df = pd.DataFrame({'age': [-5, 25, 150], 'target': [1, 0, 1]})
    result = clean_data(df)
    assert result['age'].min() >= 0
    assert result['age'].max() <= 120
```

### Integration Tests for ML Pipelines

```python
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def test_full_pipeline():
    """Test that the full pipeline runs end-to-end."""
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])
    
    pipeline.fit(X[:150], y[:150])
    predictions = pipeline.predict(X[150:])
    
    # Check output shape and type
    assert len(predictions) == 50
    assert set(predictions).issubset({0, 1})
    
    # Check minimum acceptable performance
    accuracy = (predictions == y[150:]).mean()
    assert accuracy > 0.6, f"Accuracy too low: {accuracy}"

def test_model_reproducibility():
    """Same data + same seed = same results."""
    X, y = make_classification(n_samples=100, random_state=42)
    
    model1 = LogisticRegression(random_state=42).fit(X, y)
    model2 = LogisticRegression(random_state=42).fit(X, y)
    
    np.testing.assert_array_equal(model1.predict(X), model2.predict(X))

def test_prediction_shape():
    """Model output has expected dimensionality."""
    X, y = make_classification(n_samples=100, n_classes=3, 
                                n_informative=5, random_state=42)
    model = LogisticRegression(max_iter=500).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (100, 3)
    np.testing.assert_almost_equal(proba.sum(axis=1), 1.0)  # probabilities sum to 1
```

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only fast unit tests
pytest tests/unit/ -v -x  # stop on first failure
```

> **Interview Tip:** ML testing extends beyond traditional software testing. Emphasize testing **data quality** (Great Expectations), **model performance** (regression tests on metrics), and **prediction consistency** (same input → same output). In production, add **monitoring tests** for data drift and model staleness.

---
