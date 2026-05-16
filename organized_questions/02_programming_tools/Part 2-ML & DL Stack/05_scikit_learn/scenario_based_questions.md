# Scikit-Learn Interview Questions - Scenario-Based Questions

## Question 1
- [ ] Done

**Your model achieves 95% accuracy on training data but only 70% on test data. What do you do?**

### Answer

**Problem**: Overfitting (model memorized training data)

### Diagnostic Steps
```python
from sklearn.model_selection import learning_curve
import numpy as np

# 1. Plot learning curves
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)

gap = train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1]
print(f"Train-Val Gap: {gap:.2f}")  # Large gap = overfitting
```

### Solutions

| Strategy | Implementation |
|----------|----------------|
| **Regularization** | Add L1/L2 penalty |
| **Reduce complexity** | Decrease tree depth, fewer features |
| **More data** | Increase training samples |
| **Cross-validation** | Use k-fold to detect overfitting early |
| **Early stopping** | Stop training when validation loss increases |
| **Dropout/Pruning** | For neural networks/trees |

```python
from sklearn.ensemble import RandomForestClassifier

# Before (overfit)
model = RandomForestClassifier(max_depth=None, min_samples_split=2)

# After (regularized)
model = RandomForestClassifier(
    max_depth=10,           # Limit tree depth
    min_samples_split=10,   # Require more samples to split
    min_samples_leaf=5,     # Minimum samples in leaf
    max_features='sqrt',    # Limit features per tree
    n_estimators=100
)
```

---

## Question 2
- [ ] Done

**You have a dataset with 90% Class A and 10% Class B. How do you handle this imbalance?**

### Answer

### Strategies

| Approach | When to Use |
|----------|-------------|
| **Class weights** | Quick fix, no data modification |
| **Oversampling (SMOTE)** | Small dataset, need more minority samples |
| **Undersampling** | Very large dataset, can afford to lose data |
| **Threshold tuning** | Need control over precision/recall tradeoff |
| **Different metric** | Use F1, AUC instead of accuracy |

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE  # pip install imbalanced-learn

# Method 1: Class weights
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Method 2: SMOTE oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model.fit(X_resampled, y_resampled)

# Method 3: Threshold tuning
y_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.3  # Lower threshold to catch more minority class
y_pred = (y_proba >= threshold).astype(int)

# Always use appropriate metrics
print(classification_report(y_test, y_pred))
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
```

---

## Question 3
- [ ] Done

**You need to compare multiple models and select the best one. What's your approach?**

### Answer

### Comparison Framework
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import pandas as pd
import time

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(probability=True)
}

# Compare with cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    start = time.time()
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    duration = time.time() - start
    
    results.append({
        'Model': name,
        'Mean AUC': scores.mean(),
        'Std': scores.std(),
        'Time (s)': duration
    })

# Display results
df_results = pd.DataFrame(results).sort_values('Mean AUC', ascending=False)
print(df_results)

# Select best model
best_model_name = df_results.iloc[0]['Model']
print(f"\nBest model: {best_model_name}")
```

### Selection Criteria
| Criteria | Consideration |
|----------|---------------|
| **Performance** | AUC, F1, Accuracy |
| **Training time** | Important for large datasets |
| **Inference speed** | Critical for real-time systems |
| **Interpretability** | Required for regulated industries |
| **Scalability** | How model handles more data |

---

## Question 4
- [ ] Done

**Your dataset has 1000 features but only 500 samples. How do you approach this?**

### Answer

**Problem**: High-dimensional data (p >> n) causes overfitting

### Solutions
```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.pipeline import Pipeline

# Method 1: Filter-based selection (fastest)
selector = SelectKBest(f_classif, k=100)
X_selected = selector.fit_transform(X, y)

# Method 2: L1 Regularization (embedded)
lasso = LassoCV(cv=5)
lasso.fit(X, y)
important_features = X[:, lasso.coef_ != 0]

# Method 3: PCA (dimensionality reduction)
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X)
print(f"Reduced from {X.shape[1]} to {X_pca.shape[1]} features")

# Method 4: Recursive Feature Elimination
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=50)
X_rfe = rfe.fit_transform(X, y)

# Complete pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=100)),
    ('model', LogisticRegression(penalty='l2'))
])
```

### Approach Summary
| Step | Action |
|------|--------|
| 1 | Remove low-variance features |
| 2 | Remove highly correlated features |
| 3 | Use regularization (L1/L2) |
| 4 | Apply feature selection or PCA |
| 5 | Use simpler models |

---

## Question 5
- [ ] Done

**You need to deploy a trained model. What steps do you follow?**

### Answer

### Deployment Checklist
```python
import joblib
import pickle

# 1. Save the trained model
joblib.dump(model, 'model.joblib')  # Preferred for sklearn

# 2. Save the preprocessing pipeline
joblib.dump(pipeline, 'full_pipeline.joblib')

# 3. Load and use
loaded_pipeline = joblib.load('full_pipeline.joblib')
predictions = loaded_pipeline.predict(new_data)

# 4. Version tracking
import sklearn
metadata = {
    'sklearn_version': sklearn.__version__,
    'model_type': type(model).__name__,
    'training_date': '2024-01-15',
    'features': list(feature_names),
    'metrics': {'accuracy': 0.95, 'f1': 0.92}
}
joblib.dump(metadata, 'model_metadata.joblib')
```

### Production Considerations
| Aspect | Best Practice |
|--------|---------------|
| **Serialization** | Use `joblib` for sklearn models |
| **Versioning** | Track sklearn version |
| **Input validation** | Validate feature types/ranges |
| **Monitoring** | Track prediction distributions |
| **Fallback** | Have a backup model ready |
| **Testing** | Test predictions before deployment |

---

## Question 6
- [ ] Done

**Your model's performance degrades over time in production. How do you handle this?**

### Answer

**Problem**: Model drift (data distribution changes)

### Detection
```python
from scipy.stats import ks_2samp
import numpy as np

def detect_drift(reference_data, production_data, threshold=0.05):
    """Detect feature drift using KS test"""
    drifted_features = []
    
    for col in range(reference_data.shape[1]):
        stat, p_value = ks_2samp(
            reference_data[:, col],
            production_data[:, col]
        )
        if p_value < threshold:
            drifted_features.append(col)
    
    return drifted_features

# Monitor predictions
def monitor_predictions(model, new_data, baseline_predictions):
    new_preds = model.predict_proba(new_data)[:, 1]
    
    # Check prediction distribution shift
    stat, p_value = ks_2samp(baseline_predictions, new_preds)
    
    if p_value < 0.05:
        print("⚠️ Prediction drift detected!")
        return True
    return False
```

### Solutions
| Strategy | When to Apply |
|----------|---------------|
| **Retrain periodically** | Scheduled (weekly/monthly) |
| **Online learning** | Continuous data stream |
| **Feature store** | Keep features consistent |
| **A/B testing** | Compare old vs new model |
| **Monitoring dashboard** | Track metrics in real-time |

---

## Question 7
- [ ] Done

**You're asked to explain model predictions to business stakeholders. How do you do it?**

### Answer

### Interpretability Techniques
```python
from sklearn.inspection import permutation_importance
import numpy as np

# 1. Feature Importance (tree-based models)
importances = model.feature_importances_
top_features = np.argsort(importances)[::-1][:10]
print("Top 10 important features:")
for i, idx in enumerate(top_features):
    print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

# 2. Permutation Importance (model-agnostic)
perm_imp = permutation_importance(model, X_test, y_test, n_repeats=10)

# 3. SHAP values (pip install shap)
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:100])

# Summary plot
shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names)

# Individual prediction explanation
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][0],
    X_test[0],
    feature_names=feature_names
)
```

### Explanation Framework
| Audience | Explanation Type |
|----------|------------------|
| **Executives** | High-level impact, ROI |
| **Business analysts** | Feature importance ranking |
| **Data scientists** | SHAP values, partial dependence |
| **Regulators** | Full model documentation |

---

## Question 8
- [ ] Done

**You need to handle missing values in production data differently from training. How?**

### Answer

### Robust Imputation Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer

# Training: fit imputer on training data
class RobustImputer:
    def __init__(self):
        self.imputers = {}
        self.medians = {}
        
    def fit(self, X, y=None):
        for col in range(X.shape[1]):
            self.medians[col] = np.nanmedian(X[:, col])
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in range(X.shape[1]):
            mask = np.isnan(X_copy[:, col])
            X_copy[mask, col] = self.medians[col]
        return X_copy

# Production-ready pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

# Fit on training data
pipeline.fit(X_train, y_train)

# Save imputer statistics
imputer_stats = {
    'medians': pipeline.named_steps['imputer'].statistics_,
    'scaler_mean': pipeline.named_steps['scaler'].mean_,
    'scaler_std': pipeline.named_steps['scaler'].scale_
}
```

### Key Principles
| Principle | Reason |
|-----------|--------|
| **Fit on training only** | Prevent data leakage |
| **Save statistics** | Apply same values in production |
| **Handle unexpected nulls** | Production data may have more missing |
| **Log missing patterns** | Monitor data quality |

---

## Question 9
- [ ] Done

**Your manager asks for a quick baseline model before building a complex one. What do you do?**

### Answer

### Quick Baseline Strategy
```python
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import time

# Level 1: Dummy baseline (random/majority)
dummy = DummyClassifier(strategy='most_frequent')
dummy_scores = cross_val_score(dummy, X, y, cv=5, scoring='accuracy')
print(f"Dummy baseline: {dummy_scores.mean():.4f}")

# Level 2: Simple linear model
start = time.time()
lr = LogisticRegression(max_iter=1000)
lr_scores = cross_val_score(lr, X, y, cv=5, scoring='accuracy')
lr_time = time.time() - start
print(f"Logistic Regression: {lr_scores.mean():.4f} ({lr_time:.2f}s)")

# Level 3: Moderate complexity
start = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
rf_time = time.time() - start
print(f"Random Forest: {rf_scores.mean():.4f} ({rf_time:.2f}s)")

# Report
improvement = rf_scores.mean() - lr_scores.mean()
print(f"\nImprovement over baseline: {improvement:.4f}")
print(f"Worth adding complexity: {improvement > 0.02}")
```

### Baseline Hierarchy
| Level | Model | Use Case |
|-------|-------|----------|
| 0 | Dummy (random) | Establish minimum |
| 1 | Linear (LR/Ridge) | Fast, interpretable |
| 2 | Tree-based (RF) | Non-linear patterns |
| 3 | Boosting (XGBoost) | Best performance |
| 4 | Deep Learning | Complex patterns |

---

## Question 10
- [ ] Done

**You discover that two features are highly correlated (0.95). Should you remove one?**

### Answer

### Decision Framework
```python
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# Calculate correlations
corr_matrix = pd.DataFrame(X, columns=feature_names).corr()

# Find highly correlated pairs
def find_correlated_pairs(corr_matrix, threshold=0.9):
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    return pairs

correlated_pairs = find_correlated_pairs(corr_matrix, threshold=0.95)

# Decide which to keep based on target relationship
mi_scores = mutual_info_classif(X, y)

for feat1, feat2, corr in correlated_pairs:
    idx1 = feature_names.index(feat1)
    idx2 = feature_names.index(feat2)
    
    if mi_scores[idx1] > mi_scores[idx2]:
        print(f"Keep {feat1}, remove {feat2}")
    else:
        print(f"Keep {feat2}, remove {feat1}")
```

### Considerations
| Factor | Action |
|--------|--------|
| **Model type** | Trees handle correlation well |
| **Interpretability** | Remove for clearer coefficients |
| **Computational cost** | Remove to speed up |
| **Business meaning** | Keep if both meaningful |
| **Target relationship** | Keep the more predictive one |

**Rule of thumb**: Remove for linear models, keep for tree-based models

## Question 11
- [ ] Done

**How would you approach building a recommendation system using Scikit-Learn ?**

### Answer

### Approach

| Method | Description | Best For |
|--------|-------------|----------|
| **Content-Based** | Recommend items similar to user's past preferences | Known item features |
| **Collaborative Filtering** | Use user-item interaction matrix | Many users, sparse data |
| **Hybrid** | Combine content + collaborative | Production systems |

```python
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.neighbors import NearestNeighbors
import numpy as np

# === Method 1: Content-Based (using item features) ===
item_descriptions = ["Action movie with explosions", "Romantic comedy", "Thriller with suspense"]
tfidf = TfidfVectorizer(stop_words='english')
item_features = tfidf.fit_transform(item_descriptions)
similarity_matrix = cosine_similarity(item_features)
# Recommend items most similar to a given item

# === Method 2: Collaborative Filtering (Matrix Factorization) ===
# user_item_matrix: rows=users, cols=items, values=ratings
user_item_matrix = np.array([[5, 3, 0, 1], [4, 0, 0, 1], [1, 1, 0, 5], [0, 0, 5, 4]])

# SVD for matrix factorization
svd = TruncatedSVD(n_components=2, random_state=42)
user_factors = svd.fit_transform(user_item_matrix)
item_factors = svd.components_
predicted_ratings = user_factors @ item_factors  # Reconstruct

# === Method 3: KNN-Based ===
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(user_item_matrix)
distances, indices = knn.kneighbors(user_item_matrix[0:1])
# indices = most similar users
```

> **Interview Tip:** Scikit-Learn excels at content-based filtering (TF-IDF + cosine similarity) and basic matrix factorization (SVD). For large-scale collaborative filtering, mention specialized libraries like **Surprise** or **implicit**.

---

## Question 12
- [ ] Done

**Discuss the steps you would take to diagnose and solve performance issues in a machine learning model built with Scikit-Learn**

### Answer

### Diagnostic Workflow

| Step | Check | Tool |
|------|-------|------|
| 1 | **Data quality** | Missing values, class imbalance, outliers |
| 2 | **Underfitting vs overfitting** | Learning curves |
| 3 | **Feature relevance** | Feature importance, RFE |
| 4 | **Hyperparameters** | GridSearchCV, validation curves |
| 5 | **Model selection** | Compare multiple algorithms |
| 6 | **Error analysis** | Confusion matrix, misclassified samples |

```python
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Step 1: Check for underfitting/overfitting via learning curves
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)
gap = train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1]
if gap > 0.1:
    print("Overfitting: simplify model, add regularization, get more data")
elif val_scores.mean(axis=1)[-1] < 0.7:
    print("Underfitting: more features, complex model, less regularization")

# Step 2: Validation curve to find optimal hyperparameter
train_scores, val_scores = validation_curve(
    model, X, y, param_name='max_depth', param_range=[3, 5, 10, 20, None], cv=5
)

# Step 3: Error analysis
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
# Examine which classes are confused

# Step 4: Feature importance
importances = model.feature_importances_
low_importance = np.where(importances < 0.01)[0]
print(f"Low-value features: {low_importance}")

# Step 5: Try better algorithms
from sklearn.ensemble import GradientBoostingClassifier
for name, m in [('RF', RandomForestClassifier()), ('GB', GradientBoostingClassifier())]:
    scores = cross_val_score(m, X, y, cv=5)
    print(f"{name}: {scores.mean():.4f}")
```

> **Interview Tip:** Always start with **learning curves** to diagnose bias vs variance. Then systematically check data quality, features, hyperparameters, and model choice. Document each step and its outcome.

---

## Question 13
- [ ] Done

**Propose a pipeline for processing and analyzing textual data from social media platforms using Scikit-Learn’s tools**

### Answer

### Pipeline Architecture

| Stage | Component | Purpose |
|-------|-----------|---------|
| 1 | **Text Cleaning** | Remove URLs, mentions, emojis, special chars |
| 2 | **Tokenization** | Split text into words/n-grams |
| 3 | **Vectorization** | Convert text to numerical features (TF-IDF) |
| 4 | **Feature Engineering** | Add sentiment score, text length, hashtag count |
| 5 | **Classification** | Train model (Logistic Regression, SVM, or ensemble) |

```python
import re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Custom text cleaner
class SocialMediaCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        cleaned = []
        for text in X:
            text = re.sub(r'http\S+|www\S+', '', text)         # Remove URLs
            text = re.sub(r'@\w+', '', text)                     # Remove mentions
            text = re.sub(r'#(\w+)', r'\1', text)               # Remove # but keep word
            text = re.sub(r'[^a-zA-Z\s]', '', text)              # Keep only letters
            text = text.lower().strip()
            cleaned.append(text)
        return cleaned

# Custom feature extractor for metadata
class TextMetaFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        features = []
        for text in X:
            features.append([
                len(text),                                    # Text length
                len(text.split()),                             # Word count
                text.count('#'),                               # Hashtag count
                text.count('!'),                               # Exclamation count
                sum(1 for c in text if c.isupper()) / max(len(text), 1),  # Caps ratio
            ])
        return np.array(features)

# Full pipeline
pipeline = Pipeline([
    ('cleaner', SocialMediaCleaner()),
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                               sublinear_tf=True, stop_words='english')),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# Train and evaluate
# scores = cross_val_score(pipeline, texts, labels, cv=5, scoring='f1_weighted')
# print(f"F1: {scores.mean():.4f}")
```

> **Interview Tip:** Social media text is noisy—always clean URLs, mentions, and special characters first. **TF-IDF with n-grams** works surprisingly well for social media classification. For production, consider pre-trained embeddings (BERT) instead of TF-IDF.

---
