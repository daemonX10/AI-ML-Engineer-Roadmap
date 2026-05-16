# Scikit-Learn Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Implement K-Means clustering with Scikit-Learn.**

### Solution
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

# Generate sample data
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

# 1. Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Apply K-Means
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 3. Evaluate
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.3f}")
print(f"Cluster centers:\n{kmeans.cluster_centers_}")

# 4. Elbow method to find optimal k
def find_optimal_k(X, max_k=10):
    inertias = []
    silhouettes = []
    
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels_))
    
    return inertias, silhouettes

inertias, silhouettes = find_optimal_k(X_scaled)
optimal_k = np.argmax(silhouettes) + 2
print(f"Optimal k: {optimal_k}")
```

---

## Question 2
- [ ] Done

**Implement a complete classification pipeline with preprocessing.**

### Solution
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np

# Sample data
data = {
    'age': [25, 30, np.nan, 45, 35],
    'salary': [50000, 60000, 55000, np.nan, 70000],
    'city': ['NY', 'LA', 'NY', 'LA', 'SF'],
    'target': [0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

X = df.drop('target', axis=1)
y = df['target']

# Define column types
numeric_features = ['age', 'salary']
categorical_features = ['city']

# Create transformers
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine with ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train and evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipeline.fit(X_train, y_train)

print(f"Test accuracy: {pipeline.score(X_test, y_test):.4f}")
```

---

## Question 3
- [ ] Done

**Implement hyperparameter tuning with GridSearchCV.**

### Solution
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from scipy.stats import randint

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Define model
rf = RandomForestClassifier(random_state=42)

# GridSearchCV: exhaustive search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    rf, param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")

# RandomizedSearchCV: faster for large search spaces
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    rf, param_dist,
    n_iter=50,  # Number of random combinations
    cv=5,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
```

---

## Question 4
- [ ] Done

**Implement cross-validation with multiple metrics.**

### Solution
```python
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score

# Define custom scorers
scorers = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted'),
    'roc_auc': 'roc_auc'
}

# Stratified K-Fold (maintains class proportions)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validate with multiple metrics
model = RandomForestClassifier(random_state=42)
results = cross_validate(
    model, X, y,
    cv=cv,
    scoring=scorers,
    return_train_score=True
)

# Print results
for metric in scorers.keys():
    train_score = results[f'train_{metric}'].mean()
    test_score = results[f'test_{metric}'].mean()
    print(f"{metric}: Train={train_score:.4f}, Test={test_score:.4f}")
```

---

## Question 5
- [ ] Done

**Implement a custom transformer for log transformation.**

### Solution
```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply log transformation to skewed features."""
    
    def __init__(self, columns=None, threshold=1.0):
        self.columns = columns
        self.threshold = threshold
        
    def fit(self, X, y=None):
        if self.columns is None:
            # Auto-detect skewed columns
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                skewness = X.skew()
                self.columns_ = skewness[skewness.abs() > self.threshold].index.tolist()
            else:
                self.columns_ = list(range(X.shape[1]))
        else:
            self.columns_ = self.columns
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns_:
            if isinstance(X_copy, pd.DataFrame):
                X_copy[col] = np.log1p(X_copy[col])
            else:
                X_copy[:, col] = np.log1p(X_copy[:, col])
        return X_copy

# Use in pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('log', LogTransformer(threshold=0.5)),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
```

---

## Question 6
- [ ] Done

**Implement model evaluation with confusion matrix and ROC curve.**

### Solution
```python
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_recall_curve
)
import matplotlib.pyplot as plt
import numpy as np

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)

plt.subplot(1, 2, 2)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.tight_layout()
plt.show()
```

---

## Question 7
- [ ] Done

**Implement feature importance analysis.**

### Solution
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Method 1: Built-in feature importance (tree-based)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature Ranking (Built-in):")
for i, idx in enumerate(indices[:10]):
    print(f"{i+1}. Feature {idx}: {importances[idx]:.4f}")

# Method 2: Permutation Importance (model-agnostic)
perm_importance = permutation_importance(
    model, X_test, y_test, 
    n_repeats=10, 
    random_state=42
)

print("\nPermutation Importance:")
sorted_idx = perm_importance.importances_mean.argsort()[::-1]
for idx in sorted_idx[:10]:
    print(f"Feature {idx}: {perm_importance.importances_mean[idx]:.4f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.barh(range(10), importances[indices[:10]])
plt.yticks(range(10), [f'Feature {i}' for i in indices[:10]])
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.tight_layout()
plt.show()
```

---

## Question 8
- [ ] Done

**Implement train-test split with stratification for imbalanced data.**

### Solution
```python
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from collections import Counter

# Check class distribution
print(f"Class distribution: {Counter(y)}")

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,  # Maintains class proportions
    random_state=42
)

print(f"Train distribution: {Counter(y_train)}")
print(f"Test distribution: {Counter(y_test)}")

# Stratified cross-validation
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier(class_weight='balanced', random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')

print(f"Cross-val F1 scores: {scores}")
print(f"Mean F1: {scores.mean():.4f}")
```

---

## Question 9
- [ ] Done

**Implement a regression model with regularization.**

### Solution
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Generate regression data
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Compare regularized models
models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet()
}

for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: RMSE={rmse:.4f}, R²={r2:.4f}")

# Tune regularization strength
param_grid = {'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

print(f"\nBest alpha: {grid.best_params_}")
print(f"Best CV RMSE: {np.sqrt(-grid.best_score_):.4f}")
```

---

## Question 10
- [ ] Done

**Implement learning curve analysis to diagnose bias/variance.**

### Solution
```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    
    plt.plot(train_sizes, val_mean, 's-', color='red', label='Validation')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Diagnose
    gap = train_mean[-1] - val_mean[-1]
    if gap > 0.1:
        print("⚠️ High variance (overfitting): Try more data, simpler model, regularization")
    elif val_mean[-1] < 0.7:
        print("⚠️ High bias (underfitting): Try more features, complex model")
    else:
        print("✅ Good fit!")
    
    plt.show()

# Use
model = RandomForestClassifier(random_state=42)
plot_learning_curve(model, X, y)
```


---

## Question 11
- [ ] Done

**Write a Python script using Scikit-Learn to train and evaluate a logistic regression model.**

**Answer:**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer

# 1. Load data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Build pipeline (preprocessing + model)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
])

# 4. Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 5. Train on full training set
pipeline.fit(X_train, y_train)

# 6. Evaluate
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# 7. Feature importance (coefficients)
model = pipeline.named_steps['model']
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)
print(f"\nTop 5 Features:\n{coef_df.head()}")
```

> **Interview Tip:** Always use a **Pipeline** to prevent data leakage. Use **stratify** in train_test_split for classification to preserve class distribution.

---

## Question 12
- [ ] Done

**Implement feature extraction from text using Scikit-Learn’s CountVectorizer or TfidfVectorizer**

**Answer:**

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    "Machine learning is great",
    "Deep learning is a subset of machine learning",
    "Natural language processing uses machine learning",
    "Deep learning models are powerful"
]

# === CountVectorizer: Bag of Words ===
count_vec = CountVectorizer(
    max_features=100,        # Keep top 100 words
    stop_words='english',    # Remove common words
    ngram_range=(1, 2),      # Unigrams + bigrams
    min_df=2,                # Appear in at least 2 docs
    max_df=0.9               # Appear in at most 90% of docs
)

X_count = count_vec.fit_transform(corpus)  # Sparse matrix
print(f"Shape: {X_count.shape}")           # (4, num_features)
print(f"Feature names: {count_vec.get_feature_names_out()}")
print(f"Dense:\n{X_count.toarray()}")

# === TfidfVectorizer: TF-IDF (preferred for ML) ===
tfidf_vec = TfidfVectorizer(
    max_features=100,
    stop_words='english',
    ngram_range=(1, 2),
    sublinear_tf=True,       # Use 1 + log(tf) instead of tf
    norm='l2'                # L2 normalize rows
)

X_tfidf = tfidf_vec.fit_transform(corpus)
print(f"\nTF-IDF Shape: {X_tfidf.shape}")
print(f"TF-IDF Dense:\n{X_tfidf.toarray().round(3)}")

# === Full Text Classification Pipeline ===
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('classifier', MultinomialNB())
])

# Assuming texts and labels
# scores = cross_val_score(text_pipeline, texts, labels, cv=5, scoring='f1')

# === Custom tokenization ===
import re
def custom_tokenizer(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text.split()

vec = TfidfVectorizer(tokenizer=custom_tokenizer)
```

| Feature | CountVectorizer | TfidfVectorizer |
|---------|----------------|------------------|
| Output | Raw word counts | TF-IDF weights |
| Value range | Integers (0, 1, 2...) | Floats [0, 1] |
| Common words | High values | Downweighted |
| Best for | Short texts, Naive Bayes | General text classification |

> **Interview Tip:** **TfidfVectorizer** is generally preferred over CountVectorizer because it downweights common words. Use `sublinear_tf=True` for better performance and `ngram_range=(1, 2)` to capture phrases.

---

## Question 13
- [ ] Done

**Normalize a given dataset using Scikit-Learn’s preprocessing module, then train and test a Naive Bayes classifier**

**Answer:**

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === Option A: MinMaxScaler + MultinomialNB ===
# MultinomialNB requires non-negative values
pipeline_mnb = Pipeline([
    ('scaler', MinMaxScaler()),        # Scale to [0, 1]
    ('classifier', MultinomialNB())
])
pipeline_mnb.fit(X_train, y_train)
y_pred_mnb = pipeline_mnb.predict(X_test)
print(f"MultinomialNB Accuracy: {accuracy_score(y_test, y_pred_mnb):.3f}")

# === Option B: StandardScaler + GaussianNB ===
pipeline_gnb = Pipeline([
    ('scaler', StandardScaler()),      # Z-score normalization
    ('classifier', GaussianNB())
])
pipeline_gnb.fit(X_train, y_train)
y_pred_gnb = pipeline_gnb.predict(X_test)
print(f"GaussianNB Accuracy: {accuracy_score(y_test, y_pred_gnb):.3f}")

# 3. Detailed evaluation
print("\nGaussianNB Report:")
print(classification_report(y_test, y_pred_gnb, target_names=iris.target_names))

# 4. Predict probabilities
y_prob = pipeline_gnb.predict_proba(X_test)
print(f"Confidence for first sample: {y_prob[0].round(3)}")
```

| NB Variant | Assumes | Normalization | Use Case |
|-----------|---------|---------------|---------|
| GaussianNB | Gaussian features | StandardScaler | Continuous |
| MultinomialNB | Count-like features | MinMaxScaler (≥0) | Text, counts |
| BernoulliNB | Binary features | Binarizer | Binary features |

> **Interview Tip:** **GaussianNB** doesn't technically need scaling (it fits mean/variance per feature), but scaling helps when combined in pipelines. **MultinomialNB** requires non-negative inputs, so use `MinMaxScaler`.

---

## Question 14
- [ ] Done

**Use Scikit-Learn to visualize the decision boundary of a SVM with a non-linear kernel.**

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# 1. Generate non-linear data
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 2. Train SVM with RBF kernel
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X, y)

# 3. Visualize decision boundary
def plot_decision_boundary(model, X, y, title=""):
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k', s=50)
    
    # Highlight support vectors
    sv = model.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], s=200, facecolors='none',
                edgecolors='black', linewidths=2, label='Support Vectors')
    
    plt.title(title)
    plt.legend()
    plt.show()

plot_decision_boundary(svm, X, y, title='SVM with RBF Kernel')

# 4. Compare kernels
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, kernel in zip(axes, ['linear', 'poly', 'rbf']):
    model = SVC(kernel=kernel, C=1.0, degree=3)
    model.fit(X, y)
    ax.set_title(f'{kernel.upper()} kernel (acc={model.score(X,y):.2f})')
    # Plot on each axis...
plt.tight_layout()
plt.show()
```

| Kernel | Use Case | Key Parameter |
|--------|----------|---------------|
| `linear` | Linearly separable | `C` |
| `poly` | Polynomial boundaries | `C`, `degree` |
| `rbf` | Complex, non-linear | `C`, `gamma` |
| `sigmoid` | Neural network-like | `C`, `gamma` |

> **Interview Tip:** `meshgrid` + `predict` on the grid creates the decision surface. `gamma` controls the RBF’s influence radius — low gamma = smoother boundary, high gamma = complex/overfitting.

---

## Question 15
- [ ] Done

**Implement dimensionality reduction using PCA with Scikit-Learn and visualize the result.**

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 1. Load and scale data
iris = load_iris()
X = StandardScaler().fit_transform(iris.data)  # Always scale before PCA
y = iris.target

# 2. Apply PCA — reduce from 4D to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"Original shape: {iris.data.shape}")       # (150, 4)
print(f"Reduced shape: {X_pca.shape}")             # (150, 2)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")  # [0.729, 0.228]
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.1%}")  # 95.8%

# 3. Visualize 2D projection
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                      cmap='viridis', edgecolors='k', s=50)
plt.colorbar(scatter, label='Class')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA: Iris Dataset (4D → 2D)')
plt.legend(handles=scatter.legend_elements()[0],
           labels=iris.target_names.tolist())
plt.show()

# 4. Scree plot — choose number of components
pca_full = PCA().fit(X)
plt.figure(figsize=(8, 4))
plt.bar(range(1, 5), pca_full.explained_variance_ratio_, alpha=0.7, label='Individual')
plt.step(range(1, 5), np.cumsum(pca_full.explained_variance_ratio_),
         where='mid', label='Cumulative')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.legend()
plt.show()

# 5. PCA with threshold
pca_95 = PCA(n_components=0.95)  # Keep components for 95% variance
X_reduced = pca_95.fit_transform(X)
print(f"Components needed for 95% variance: {pca_95.n_components_}")
```

> **Interview Tip:** Always **StandardScaler** before PCA (PCA is sensitive to scale). Use the **scree plot** to choose components. `n_components=0.95` automatically selects enough components for 95% variance.

---

## Question 16
- [ ] Done

**Create a clustering analysis on a dataset using Scikit-Learn’s DBSCAN method**

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_blobs
from sklearn.metrics import silhouette_score

# 1. Generate non-linear data (DBSCAN excels here)
X, y_true = make_moons(n_samples=300, noise=0.1, random_state=42)
X = StandardScaler().fit_transform(X)

# 2. Apply DBSCAN
dbscan = DBSCAN(
    eps=0.3,          # Maximum distance between neighbors
    min_samples=5,    # Minimum points to form a cluster
    metric='euclidean'
)
labels = dbscan.fit_predict(X)

# 3. Analyze results
n_clusters = len(set(labels) - {-1})    # Exclude noise label (-1)
n_noise = list(labels).count(-1)
print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise} ({n_noise/len(X):.1%})")

# Silhouette score (exclude noise)
mask = labels != -1
if len(set(labels[mask])) > 1:
    sil = silhouette_score(X[mask], labels[mask])
    print(f"Silhouette Score: {sil:.3f}")

# 4. Visualize
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10',
                      edgecolors='k', s=50)
noise_mask = labels == -1
plt.scatter(X[noise_mask, 0], X[noise_mask, 1], c='red', marker='x',
            s=100, label=f'Noise ({n_noise})', linewidths=2)
plt.title(f'DBSCAN: {n_clusters} clusters, eps={0.3}, min_samples={5}')
plt.legend()
plt.show()

# 5. Tune eps with k-distance plot
from sklearn.neighbors import NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(X)
distances, _ = neighbors.kneighbors(X)
distances = np.sort(distances[:, -1])  # k-th nearest neighbor distance

plt.figure(figsize=(8, 4))
plt.plot(distances)
plt.xlabel('Points (sorted)')
plt.ylabel(f'{5}-th nearest neighbor distance')
plt.title('K-Distance Plot (elbow = optimal eps)')
plt.show()
```

| Parameter | Effect | How to Choose |
|-----------|--------|---------------|
| `eps` | Neighborhood radius | K-distance elbow plot |
| `min_samples` | Min cluster density | Rule of thumb: 2 × dimensions |

| DBSCAN vs K-Means | DBSCAN | K-Means |
|-------------------|--------|---------|
| Shape | Arbitrary | Spherical |
| Num clusters | Automatic | Must specify k |
| Outliers | Detected as noise | Assigned to clusters |
| Scalability | O(n²) or O(n log n) | O(nk) |

> **Interview Tip:** DBSCAN is ideal for **non-spherical clusters** and **outlier detection**. Use the **k-distance plot** to find optimal `eps` value (look for the elbow).

---

## Question 17
- [ ] Done

**How do you save a trained Scikit-Learn model to disk and load it back for later use?**

**Answer:**

```python
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Train a model
model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])
model.fit(X_train, y_train)

# === Method 1: joblib (RECOMMENDED for Scikit-Learn) ===
# Save
joblib.dump(model, 'model.joblib')

# Save compressed (smaller file)
joblib.dump(model, 'model.joblib.gz', compress=3)

# Load
loaded_model = joblib.load('model.joblib')
y_pred = loaded_model.predict(X_test)

# === Method 2: pickle ===
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# === Method 3: ONNX (cross-platform deployment) ===
# pip install skl2onnx onnxruntime
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)
with open('model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

# === Best Practices ===
# Save metadata alongside model
import json
metadata = {
    'sklearn_version': '1.3.0',
    'features': list(feature_names),
    'target': 'label',
    'accuracy': 0.95,
    'training_date': '2024-01-15'
}
joblib.dump({'model': model, 'metadata': metadata}, 'model_with_meta.joblib')
```

| Method | Speed | Size | Cross-platform | Best For |
|--------|-------|------|----------------|----------|
| joblib | Fastest | Compact | Python only | Development, Scikit-Learn |
| pickle | Fast | Larger | Python only | General Python objects |
| ONNX | N/A | Small | Yes (any lang) | Production deployment |

> **Interview Tip:** Always save the **Scikit-Learn version** with the model — models may be incompatible across versions. Use `joblib` for Scikit-Learn (optimized for large NumPy arrays). For production, convert to **ONNX** for language-agnostic deployment.

## Question 18
- [ ] Done

**Create a Python function that uses Scikit-Learn to perform a k-fold cross-validation on a dataset**

### Solution
```python
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def kfold_cross_validation(X, y, model=None, n_splits=5, scoring='accuracy',
                           stratified=True, random_state=42):
    \"\"\"Perform k-fold cross-validation and return detailed results.\"\"\"
    if model is None:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=random_state))
        ])
    
    cv = (StratifiedKFold if stratified else KFold)(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    
    # Multiple metrics support
    if isinstance(scoring, list):
        results = cross_validate(model, X, y, cv=cv, scoring=scoring,
                                 return_train_score=True, n_jobs=-1)
        for metric in scoring:
            train = results[f'train_{metric}']
            test = results[f'test_{metric}']
            print(f"{metric}: Train={train.mean():.4f}±{train.std():.4f}, "
                  f"Test={test.mean():.4f}±{test.std():.4f}")
        return results
    
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    for i, s in enumerate(scores):
        print(f"Fold {i+1}: {s:.4f}")
    print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")
    return {'scores': scores, 'mean': scores.mean(), 'std': scores.std()}

# Usage
X, y = load_iris(return_X_y=True)
result = kfold_cross_validation(X, y, n_splits=5)
result = kfold_cross_validation(X, y, scoring=['accuracy', 'f1_weighted'])
```

> **Interview Tip:** Always use **StratifiedKFold** for classification to maintain class proportions. Use `cross_validate()` for multiple metrics or train scores.

---

## Question 19
- [ ] Done

**Demonstrate how to use Scikit-Learn’s Pipeline to combine preprocessing and model training steps**

### Solution

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# Sample data with mixed types
data = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 35, 28, 50, np.nan, 40, 33],
    'salary': [50000, 60000, 55000, np.nan, 70000, 48000, 90000, 62000, np.nan, 58000],
    'city': ['NY', 'LA', 'NY', 'LA', 'SF', 'NY', 'LA', 'SF', 'NY', 'LA'],
    'target': [0, 1, 0, 1, 1, 0, 1, 1, 0, 1]
})
X = data.drop('target', axis=1)
y = data['target']

# Preprocessing pipelines for different column types
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, ['age', 'salary']),
    ('cat', categorical_transformer, ['city'])
])

# Full pipeline: preprocessing + model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
pipeline.fit(X_train, y_train)
print(f"Accuracy: {pipeline.score(X_test, y_test):.4f}")

# Hyperparameter tuning through the pipeline
param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None]
}
grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
```

> **Interview Tip:** Pipelines prevent data leakage by fitting preprocessing only on training data. Use `double_underscore__` notation to tune any nested step in `GridSearchCV`.

---

## Question 20
- [ ] Done

**Write a Python function that uses Scikit-Learn’s RandomForestClassifier and performs a grid search to find the best hyperparameters**

### Solution

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint

def rf_grid_search(X, y, search_type='grid', cv=5, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=random_state))
    ])

    if search_type == 'grid':
        param_grid = {
            'rf__n_estimators': [50, 100, 200],
            'rf__max_depth': [5, 10, 20, None],
            'rf__min_samples_split': [2, 5, 10],
            'rf__max_features': ['sqrt', 'log2']
        }
        search = GridSearchCV(pipeline, param_grid, cv=cv,
                              scoring='f1_weighted', n_jobs=-1, verbose=1)
    else:
        param_dist = {
            'rf__n_estimators': randint(50, 500),
            'rf__max_depth': randint(3, 50),
            'rf__min_samples_split': randint(2, 20),
            'rf__max_features': ['sqrt', 'log2', None]
        }
        search = RandomizedSearchCV(pipeline, param_dist, n_iter=100, cv=cv,
                                     scoring='f1_weighted', n_jobs=-1,
                                     random_state=random_state)

    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.4f}")
    print(classification_report(y_test, search.predict(X_test)))
    return search.best_estimator_, search

# Usage
X, y = load_breast_cancer(return_X_y=True)
best_model, results = rf_grid_search(X, y, search_type='random')
```

> **Interview Tip:** Start with **RandomizedSearchCV** for a broad search, then narrow down with **GridSearchCV**. Always wrap models in pipelines to prevent data leakage.

---
