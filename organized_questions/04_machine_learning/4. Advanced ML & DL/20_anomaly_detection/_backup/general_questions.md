# Anomaly Detection Interview Questions - General Questions

---

## Question 1: How do you handle high-dimensional data in anomaly detection?
- [ ] Done

### Answer

**Challenge**: Curse of dimensionality makes distance-based methods ineffective.

**Why High Dimensions Are Problematic**:

```
As dimensions increase:
- All points become equidistant
- Distance metrics lose discriminative power
- Density estimation becomes unreliable
- Computational cost explodes

Distance concentration:
lim(d→∞) [d_max - d_min] / d_min → 0
```

**Solutions**:

| Approach | Method | When to Use |
|----------|--------|-------------|
| **Dimensionality Reduction** | PCA, t-SNE, UMAP | Preserve structure |
| **Feature Selection** | Filter, Wrapper, Embedded | Remove irrelevant features |
| **Subspace Methods** | Feature bagging, random projections | High-d data |
| **Specialized Algorithms** | Isolation Forest, LOF variants | Native high-d support |

**Strategy Comparison**:

```
Dimensionality Reduction:
High-D → PCA/UMAP → Low-D → Standard anomaly detection
                      ↓
                   Retain key variance

Subspace Methods:
High-D → Multiple random subspaces → Detect in each → Aggregate
              ↓                           ↓
         Random projections          Ensemble voting
```

**Python Implementation**:

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.random_projection import GaussianRandomProjection

def high_dim_anomaly_detection(X, method='pca', target_dim=10):
    """
    Handle high-dimensional anomaly detection.
    """
    if method == 'pca':
        # Reduce dimensions with PCA
        pca = PCA(n_components=target_dim)
        X_reduced = pca.fit_transform(X)
        
        # Detect in reduced space
        detector = IsolationForest(contamination=0.1)
        return detector.fit_predict(X_reduced)
    
    elif method == 'subspace':
        # Feature bagging: multiple random subspaces
        n_subspaces = 10
        n_features_per_subspace = min(target_dim, X.shape[1] // 2)
        
        all_predictions = []
        for _ in range(n_subspaces):
            # Random feature subset
            features = np.random.choice(X.shape[1], n_features_per_subspace, replace=False)
            X_sub = X[:, features]
            
            detector = IsolationForest(contamination=0.1)
            predictions = detector.fit_predict(X_sub)
            all_predictions.append(predictions)
        
        # Majority vote
        all_predictions = np.array(all_predictions)
        return np.sign(all_predictions.sum(axis=0))
    
    elif method == 'random_projection':
        # Johnson-Lindenstrauss projection
        projector = GaussianRandomProjection(n_components=target_dim)
        X_projected = projector.fit_transform(X)
        
        detector = IsolationForest(contamination=0.1)
        return detector.fit_predict(X_projected)
```

**Best Practices**:

1. Always visualize reduced data to verify structure preservation
2. Use Isolation Forest for native high-d handling
3. Consider domain knowledge for feature selection
4. Ensemble subspace methods for robustness

**Interview Tip**: Mention that Isolation Forest handles high dimensions better than distance-based methods because it uses random splits, not distances.

---

## Question 2: What preprocessing steps are important before applying anomaly detection algorithms?
- [ ] Done

### Answer

**Essential Preprocessing Pipeline**:

```
Raw Data
    ↓
1. Handle Missing Values
    ↓
2. Remove/Fix Data Errors
    ↓
3. Feature Scaling
    ↓
4. Handle Categorical Variables
    ↓
5. Feature Engineering
    ↓
6. Dimensionality Reduction (if needed)
    ↓
Preprocessed Data → Anomaly Detection
```

**Detailed Steps**:

| Step | Methods | Importance for AD |
|------|---------|-------------------|
| **Missing Values** | Imputation, deletion | Missing patterns may be anomalies |
| **Scaling** | StandardScaler, MinMaxScaler | Distance-based methods require it |
| **Encoding** | One-hot, target encoding | Handle categorical features |
| **Outlier Handling** | Winsorization, capping | Extreme values affect scaling |

**Python Implementation**:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer

class AnomalyDetectionPreprocessor:
    def __init__(self, scaling='robust', impute_strategy='median'):
        self.scaling = scaling
        self.impute_strategy = impute_strategy
        self.scaler = None
        self.imputer = None
    
    def fit_transform(self, X):
        """Full preprocessing pipeline."""
        X = X.copy()
        
        # 1. Handle missing values
        if self.impute_strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
        else:
            self.imputer = SimpleImputer(strategy=self.impute_strategy)
        X = self.imputer.fit_transform(X)
        
        # 2. Feature scaling
        if self.scaling == 'robust':
            # Robust to outliers
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        return X
    
    def transform(self, X):
        """Transform new data."""
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        return X


def handle_categorical_for_anomaly(df, categorical_cols):
    """Handle categorical variables for anomaly detection."""
    df_processed = df.copy()
    
    for col in categorical_cols:
        # Frequency encoding (useful for rare categories as anomalies)
        freq = df[col].value_counts(normalize=True)
        df_processed[f'{col}_freq'] = df[col].map(freq)
        
        # Rare category flag
        rare_threshold = 0.01
        df_processed[f'{col}_is_rare'] = (df_processed[f'{col}_freq'] < rare_threshold).astype(int)
    
    return df_processed
```

**Scaling Choice Impact**:

| Scaler | Formula | When to Use |
|--------|---------|-------------|
| StandardScaler | $(x-\mu)/\sigma$ | Gaussian-like data |
| RobustScaler | $(x-Q2)/(Q3-Q1)$ | Data with outliers |
| MinMaxScaler | $(x-min)/(max-min)$ | Bounded features |

**Common Mistakes to Avoid**:

1. ❌ Fitting scaler on test data (data leakage)
2. ❌ Removing outliers before anomaly detection (might remove true anomalies!)
3. ❌ Using StandardScaler when outliers present
4. ❌ Ignoring missing value patterns

**Interview Tip**: RobustScaler is preferred for anomaly detection preprocessing because StandardScaler is sensitive to the very outliers you're trying to detect.

---

## Question 3: How do you select the threshold for flagging anomalies using a given method?
- [ ] Done

### Answer

**Threshold Selection Approaches**:

| Approach | Method | When to Use |
|----------|--------|-------------|
| **Statistical** | Percentile, standard deviations | Known contamination rate |
| **Domain-based** | Business rules, expert knowledge | Clear operational limits |
| **Validation-based** | Optimize metric on labeled data | Labels available |
| **Visual** | Elbow method, score distribution | Exploratory analysis |

**Statistical Methods**:

```python
import numpy as np
from sklearn.ensemble import IsolationForest

def threshold_by_percentile(scores, contamination=0.05):
    """Set threshold at percentile based on expected contamination."""
    threshold = np.percentile(scores, contamination * 100)
    return threshold

def threshold_by_std(scores, n_std=3):
    """Set threshold at n standard deviations from mean."""
    threshold = scores.mean() - n_std * scores.std()
    return threshold

def threshold_by_iqr(scores, k=1.5):
    """Set threshold using IQR method."""
    q1, q3 = np.percentile(scores, [25, 75])
    iqr = q3 - q1
    threshold = q1 - k * iqr
    return threshold
```

**Validation-Based Selection**:

```python
from sklearn.metrics import f1_score, precision_recall_curve
import numpy as np

def optimize_threshold(y_true, scores, metric='f1'):
    """Find optimal threshold using labeled data."""
    # Try different thresholds
    thresholds = np.percentile(scores, np.linspace(1, 20, 100))
    
    best_threshold = None
    best_score = -np.inf
    
    for thresh in thresholds:
        y_pred = (scores < thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = (y_true[y_pred == 1] == 1).mean()
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score

def precision_at_k(y_true, scores, k):
    """Precision when flagging top-k anomalies."""
    top_k_indices = np.argsort(scores)[:k]
    return y_true[top_k_indices].mean()
```

**Visual Elbow Method**:

```python
import matplotlib.pyplot as plt
import numpy as np

def elbow_threshold(scores):
    """Find threshold using elbow in sorted scores."""
    sorted_scores = np.sort(scores)
    
    # Calculate second derivative (curvature)
    first_diff = np.diff(sorted_scores)
    second_diff = np.diff(first_diff)
    
    # Elbow at maximum curvature
    elbow_idx = np.argmax(np.abs(second_diff)) + 1
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(sorted_scores)
    plt.axvline(elbow_idx, color='r', linestyle='--', label='Elbow')
    plt.title('Sorted Anomaly Scores')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(scores, bins=50)
    plt.axvline(sorted_scores[elbow_idx], color='r', linestyle='--')
    plt.title('Score Distribution')
    plt.show()
    
    return sorted_scores[elbow_idx]
```

**Business-Driven Threshold**:

```
Consider:
- Cost of false positive (unnecessary investigation)
- Cost of false negative (missed anomaly)
- Operational capacity (how many alerts can team handle?)

If FP cost << FN cost: Lower threshold (more sensitive)
If FP cost >> FN cost: Higher threshold (more specific)
```

**Interview Tip**: Always ask about the cost of false positives vs. false negatives - the threshold should be optimized for business impact, not just statistical measures.

---

## Question 4: What metrics would you use to evaluate the performance of an anomaly detection model?
- [ ] Done

### Answer

**Key Challenge**: Extreme class imbalance (anomalies are rare).

**Metrics Overview**:

| Metric | Formula | Best For |
|--------|---------|----------|
| **Precision** | TP / (TP + FP) | When FP cost is high |
| **Recall** | TP / (TP + FN) | When FN cost is high |
| **F1 Score** | 2 × (P × R) / (P + R) | Balance P and R |
| **AUC-ROC** | Area under ROC curve | Overall ranking ability |
| **AUC-PR** | Area under PR curve | Imbalanced data |
| **Precision@K** | Precision in top K | Limited review capacity |

**Why Accuracy Fails**:

```
Example: 1% anomaly rate
Predict all normal → 99% accuracy! (but useless)

Confusion Matrix:
                 Predicted
                 Normal  Anomaly
Actual  Normal    990      0
        Anomaly    10      0

Accuracy = 99%, but missed ALL anomalies
```

**Python Implementation**:

```python
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt

def evaluate_anomaly_detector(y_true, y_pred, scores=None):
    """
    Comprehensive evaluation of anomaly detection.
    
    Args:
        y_true: Ground truth (1 = anomaly, 0 = normal)
        y_pred: Binary predictions
        scores: Continuous anomaly scores (optional)
    """
    results = {}
    
    # Basic metrics
    results['precision'] = precision_score(y_true, y_pred)
    results['recall'] = recall_score(y_true, y_pred)
    results['f1'] = f1_score(y_true, y_pred)
    
    if scores is not None:
        # ROC-AUC and PR-AUC
        results['roc_auc'] = roc_auc_score(y_true, scores)
        results['pr_auc'] = average_precision_score(y_true, scores)
        
        # Precision at K
        k_values = [10, 50, 100]
        for k in k_values:
            top_k = np.argsort(scores)[-k:]  # Top k highest scores
            results[f'precision@{k}'] = y_true[top_k].mean()
    
    return results

def plot_evaluation_curves(y_true, scores):
    """Plot ROC and PR curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)
    axes[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    axes[1].plot(recall, precision, label=f'AP = {pr_auc:.3f}')
    axes[1].axhline(y=y_true.mean(), color='k', linestyle='--', 
                    label=f'Baseline = {y_true.mean():.3f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
```

**Metric Selection Guide**:

| Scenario | Primary Metric | Reason |
|----------|----------------|--------|
| Fraud detection | Recall, then Precision | Missing fraud is costly |
| Manufacturing QC | Precision@K | Limited inspection capacity |
| Medical screening | High Recall | Don't miss disease |
| Spam filtering | F1 Score | Balance user experience |

**PR-AUC vs ROC-AUC**:

```
Use PR-AUC when:
- Class imbalance is severe (< 5% anomalies)
- You care more about positive class performance

Use ROC-AUC when:
- Classes are more balanced
- You want to evaluate overall ranking
```

**Interview Tip**: For imbalanced anomaly detection, PR-AUC is more informative than ROC-AUC because it focuses on the minority class performance.

---

## Question 5: How can you ensure your anomaly detection model is not overfitting?
- [ ] Done

### Answer

**Overfitting in Anomaly Detection**:

```
Overfitting symptoms:
- Model memorizes training data noise
- Flags normal variations as anomalies (false positives)
- Misses true anomalies similar to training noise
```

**Prevention Strategies**:

| Strategy | Implementation | Effect |
|----------|----------------|--------|
| **Cross-validation** | K-fold on normal data | Estimate generalization |
| **Holdout set** | Temporal split | Test on future data |
| **Regularization** | L1/L2 penalties | Simplify model |
| **Early stopping** | Monitor validation loss | Prevent overtraining |
| **Ensemble** | Multiple models | Reduce variance |

**Validation Approaches**:

```python
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, KFold

def cross_validate_anomaly_detector(X, detector_class, n_splits=5):
    """
    Cross-validation for unsupervised anomaly detection.
    Evaluates consistency of scores across folds.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_scores = []
    score_correlations = []
    
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        
        # Train on fold
        detector = detector_class()
        detector.fit(X_train)
        
        # Score test set
        scores = detector.decision_function(X_test)
        all_scores.append((test_idx, scores))
    
    # Check consistency: do different folds rank same points similarly?
    # (Good model should be consistent)
    return all_scores

def temporal_validation(X, timestamps, detector, train_ratio=0.7):
    """
    Temporal validation for time-series anomaly detection.
    Train on past, test on future.
    """
    n = len(X)
    train_size = int(n * train_ratio)
    
    # Sort by time
    sorted_idx = np.argsort(timestamps)
    X_sorted = X[sorted_idx]
    
    X_train = X_sorted[:train_size]
    X_test = X_sorted[train_size:]
    
    detector.fit(X_train)
    scores_train = detector.decision_function(X_train)
    scores_test = detector.decision_function(X_test)
    
    # Check for distribution shift
    print(f"Train score mean: {scores_train.mean():.4f}")
    print(f"Test score mean: {scores_test.mean():.4f}")
    
    return scores_train, scores_test
```

**Regularization in Autoencoders**:

```python
import tensorflow as tf

def regularized_autoencoder(input_dim, encoding_dim, l2_lambda=0.01, dropout_rate=0.3):
    """Build autoencoder with regularization to prevent overfitting."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', 
                             kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
                             input_shape=(input_dim,)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(encoding_dim, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(input_dim, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

**Signs of Overfitting**:

| Sign | Diagnosis |
|------|-----------|
| Train error << Test error | Clear overfitting |
| Unstable across folds | High variance |
| Sensitive to noise | Model too complex |
| Too many anomalies flagged | Learned noise as anomalies |

**Interview Tip**: For anomaly detection, consistency check across different subsets is key - an overfit model will give very different results on different samples.

---

## Question 6: How can anomaly detection models be updated over time as new data comes in?
- [ ] Done

### Answer

**Challenge**: Normal behavior evolves (concept drift), requiring model updates.

**Update Strategies**:

| Strategy | Approach | When to Use |
|----------|----------|-------------|
| **Periodic retraining** | Full retrain on recent data | Batch processing |
| **Incremental learning** | Update with new samples | Streaming data |
| **Sliding window** | Train on recent N samples | Time-based drift |
| **Ensemble replacement** | Replace oldest sub-models | Continuous learning |

**Concept Drift Types**:

```
Sudden Drift:          Gradual Drift:         Recurring Drift:
    │                      │                      │
────┼────┐            ────/                  ────┼────┐
         │               /                   │   │    │
         └────        ──/                    └───┼────┘
                                                 │
```

**Python Implementation**:

```python
import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest

class OnlineAnomalyDetector:
    """Anomaly detector with online updates."""
    
    def __init__(self, window_size=10000, update_frequency=1000):
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.data_buffer = deque(maxlen=window_size)
        self.model = None
        self.samples_since_update = 0
    
    def partial_fit(self, X_new):
        """Update model with new data."""
        # Add to buffer
        for x in X_new:
            self.data_buffer.append(x)
        
        self.samples_since_update += len(X_new)
        
        # Retrain if enough new samples
        if self.samples_since_update >= self.update_frequency:
            self._retrain()
            self.samples_since_update = 0
    
    def _retrain(self):
        """Retrain model on current buffer."""
        X_train = np.array(list(self.data_buffer))
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.model.fit(X_train)
    
    def predict(self, X):
        """Predict anomalies."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def score_samples(self, X):
        """Get anomaly scores."""
        return self.model.decision_function(X)


class EnsembleOnlineDetector:
    """Ensemble of models trained on different time windows."""
    
    def __init__(self, n_models=5, window_size=5000):
        self.n_models = n_models
        self.window_size = window_size
        self.models = []
        self.model_ages = []
    
    def update(self, X_new):
        """Add new model, remove oldest if needed."""
        # Train new model on recent data
        new_model = IsolationForest(contamination=0.1)
        new_model.fit(X_new)
        
        self.models.append(new_model)
        self.model_ages.append(0)
        
        # Remove oldest model if exceeds limit
        if len(self.models) > self.n_models:
            self.models.pop(0)
            self.model_ages.pop(0)
        
        # Age all models
        self.model_ages = [age + 1 for age in self.model_ages]
    
    def predict(self, X):
        """Ensemble prediction with recency weighting."""
        if not self.models:
            raise ValueError("No models trained")
        
        # Weight by recency (newer models have higher weight)
        weights = [1 / (age + 1) for age in self.model_ages]
        weights = np.array(weights) / sum(weights)
        
        predictions = np.zeros(len(X))
        for model, weight in zip(self.models, weights):
            pred = model.predict(X)
            predictions += weight * pred
        
        return np.sign(predictions)
```

**Drift Detection**:

```python
def detect_drift(old_scores, new_scores, threshold=0.05):
    """
    Detect if score distribution has shifted significantly.
    Uses Kolmogorov-Smirnov test.
    """
    from scipy import stats
    
    statistic, p_value = stats.ks_2samp(old_scores, new_scores)
    
    drift_detected = p_value < threshold
    
    return drift_detected, p_value

def adaptive_update(detector, X_new, old_scores, significance=0.05):
    """Update model only if drift detected."""
    new_scores = detector.score_samples(X_new)
    
    drift, p_value = detect_drift(old_scores, new_scores, significance)
    
    if drift:
        print(f"Drift detected (p={p_value:.4f}), retraining...")
        detector.fit(X_new)
    else:
        print(f"No significant drift (p={p_value:.4f})")
    
    return drift
```

**Interview Tip**: Mention the trade-off between model freshness and stability - frequent updates catch drift but may introduce noise.

---

## Question 7: How is DBSCAN clustering used for anomaly detection?
- [ ] Done

### Answer

**Core Idea**: DBSCAN labels sparse points as "noise" - these are natural anomalies.

**DBSCAN Concepts**:

```
Core Point:    ≥ minPts neighbors within ε
Border Point:  < minPts but neighbor of core
Noise Point:   Neither → ANOMALY

Visual:
    •──•──•     Core points (dense)
    │  │  │
    •──•──•
        │
        ∘       Border point
        
        
        *       Noise point (ANOMALY)
```

**Parameters**:

| Parameter | Description | Effect on Anomaly Detection |
|-----------|-------------|----------------------------|
| **eps (ε)** | Neighborhood radius | Larger → fewer anomalies |
| **min_samples** | Min neighbors for core | Larger → more anomalies |

**Python Implementation**:

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

def dbscan_anomaly_detection(X, eps=0.5, min_samples=5):
    """
    Detect anomalies using DBSCAN.
    
    Args:
        X: Feature matrix
        eps: Maximum distance for neighborhood
        min_samples: Minimum points for core status
    
    Returns:
        anomalies: Boolean mask (True = anomaly)
        labels: Cluster labels (-1 = noise/anomaly)
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    # Noise points (-1) are anomalies
    anomalies = labels == -1
    
    return anomalies, labels


def tune_dbscan_eps(X, min_samples=5):
    """
    Tune eps parameter using k-distance graph.
    """
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt
    
    # Find k-nearest neighbor distances
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    
    # Sort distances to min_samples-th neighbor
    k_distances = np.sort(distances[:, min_samples - 1])
    
    # Plot k-distance graph
    plt.figure(figsize=(10, 5))
    plt.plot(k_distances)
    plt.xlabel('Points (sorted)')
    plt.ylabel(f'{min_samples}-NN Distance')
    plt.title('K-Distance Graph (elbow = good eps)')
    plt.grid(True)
    plt.show()
    
    # Suggest eps at elbow (maximum curvature)
    gradient = np.gradient(k_distances)
    elbow_idx = np.argmax(gradient)
    suggested_eps = k_distances[elbow_idx]
    
    return suggested_eps
```

**Advantages for Anomaly Detection**:

| Advantage | Explanation |
|-----------|-------------|
| No contamination assumption | Doesn't require specifying anomaly rate |
| Handles varying densities | With appropriate parameters |
| Discovers clusters | Simultaneously clusters and finds anomalies |
| No cluster count needed | Unlike K-means |

**Disadvantages**:

| Disadvantage | Mitigation |
|--------------|------------|
| Sensitive to eps | Use k-distance graph |
| Uniform density assumption | Use HDBSCAN for varying densities |
| High dimensions issues | Reduce dimensions first |

**HDBSCAN Alternative**:

```python
import hdbscan

def hdbscan_anomaly_detection(X, min_cluster_size=15, min_samples=5):
    """
    Hierarchical DBSCAN handles varying densities better.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    labels = clusterer.fit_predict(X)
    
    # Get outlier scores (probability of being outlier)
    outlier_scores = clusterer.outlier_scores_
    
    return labels == -1, outlier_scores
```

**Interview Tip**: DBSCAN's main advantage for anomaly detection is that it doesn't require specifying the contamination rate upfront - it discovers anomalies organically as noise points.

---

## Question 8: Present a framework for detecting anomalies in social media trend data
- [ ] Done

### Answer

**Framework Overview**:

```
Social Media Trend Anomaly Detection Framework

┌─────────────────────────────────────────────────────────┐
│                    DATA COLLECTION                       │
│  • Post volumes, engagement rates, sentiment scores      │
│  • Hashtag frequencies, user activity patterns           │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌───────────────────────────┴─────────────────────────────┐
│                    PREPROCESSING                         │
│  • Time-series aggregation (hourly/daily)               │
│  • Seasonal decomposition                                │
│  • Feature engineering                                   │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌───────────────────────────┴─────────────────────────────┐
│                  ANOMALY DETECTION                       │
│  • Trend anomalies (unexpected spikes/drops)            │
│  • Behavioral anomalies (bot detection)                  │
│  • Content anomalies (unusual topics)                    │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌───────────────────────────┴─────────────────────────────┐
│                    ALERT & ACTION                        │
│  • Real-time notifications                               │
│  • Investigation dashboard                               │
│  • Automated responses                                   │
└─────────────────────────────────────────────────────────┘
```

**Types of Social Media Anomalies**:

| Anomaly Type | Description | Detection Method |
|--------------|-------------|------------------|
| **Viral spike** | Sudden trend explosion | Time-series threshold |
| **Bot activity** | Coordinated fake engagement | Behavioral clustering |
| **Sentiment shift** | Unusual opinion change | NLP + statistical |
| **Trending manipulation** | Artificial trend inflation | Pattern analysis |

**Python Implementation**:

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

class SocialMediaAnomalyDetector:
    """Framework for detecting anomalies in social media trends."""
    
    def __init__(self, baseline_window=168):  # 1 week of hourly data
        self.baseline_window = baseline_window
        self.trend_detector = None
        self.behavior_detector = None
    
    def preprocess_trend_data(self, df):
        """
        Preprocess raw social media data.
        
        Expected columns: timestamp, post_count, engagement, sentiment
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Resample to hourly
        df_hourly = df.resample('1H').agg({
            'post_count': 'sum',
            'engagement': 'mean',
            'sentiment': 'mean'
        }).fillna(method='ffill')
        
        # Add time features
        df_hourly['hour'] = df_hourly.index.hour
        df_hourly['dayofweek'] = df_hourly.index.dayofweek
        df_hourly['is_weekend'] = df_hourly['dayofweek'].isin([5, 6]).astype(int)
        
        # Rolling statistics
        df_hourly['post_count_rolling_mean'] = df_hourly['post_count'].rolling(24).mean()
        df_hourly['post_count_rolling_std'] = df_hourly['post_count'].rolling(24).std()
        
        # Deviation from rolling baseline
        df_hourly['post_count_zscore'] = (
            (df_hourly['post_count'] - df_hourly['post_count_rolling_mean']) /
            (df_hourly['post_count_rolling_std'] + 1e-10)
        )
        
        return df_hourly.dropna()
    
    def detect_trend_anomalies(self, df, threshold=3):
        """
        Detect unusual spikes/drops in trend metrics.
        """
        anomalies = {}
        
        # Z-score based detection for each metric
        for col in ['post_count', 'engagement', 'sentiment']:
            col_zscore = f'{col}_zscore' if f'{col}_zscore' in df.columns else None
            if col_zscore:
                anomalies[col] = np.abs(df[col_zscore]) > threshold
            else:
                # Calculate z-score
                z = stats.zscore(df[col])
                anomalies[col] = np.abs(z) > threshold
        
        return pd.DataFrame(anomalies, index=df.index)
    
    def detect_viral_events(self, df, growth_threshold=5):
        """
        Detect viral events based on rapid growth.
        """
        df = df.copy()
        
        # Hour-over-hour growth rate
        df['growth_rate'] = df['post_count'].pct_change()
        
        # Viral = growth rate > threshold AND absolute volume significant
        volume_threshold = df['post_count'].quantile(0.75)
        
        viral = (
            (df['growth_rate'] > growth_threshold) &
            (df['post_count'] > volume_threshold)
        )
        
        return viral
    
    def detect_bot_patterns(self, user_activity_df):
        """
        Detect bot-like behavior patterns.
        
        Features: posting_frequency, time_regularity, content_similarity
        """
        features = user_activity_df[['posting_frequency', 'time_regularity', 
                                     'content_similarity', 'follower_ratio']]
        
        # Isolation Forest for behavioral anomalies
        self.behavior_detector = IsolationForest(contamination=0.05)
        predictions = self.behavior_detector.fit_predict(features)
        
        return predictions == -1  # True = suspected bot
    
    def detect_sentiment_anomalies(self, df, window=24):
        """
        Detect unusual sentiment shifts.
        """
        df = df.copy()
        
        # Sentiment change rate
        df['sentiment_change'] = df['sentiment'].diff(window)
        
        # Anomaly if sudden large shift
        threshold = 2 * df['sentiment_change'].std()
        anomalies = np.abs(df['sentiment_change']) > threshold
        
        return anomalies


# Real-time monitoring example
class RealTimeSocialMonitor:
    """Real-time social media trend monitoring."""
    
    def __init__(self, detector, alert_callback=None):
        self.detector = detector
        self.alert_callback = alert_callback or print
        self.buffer = []
        self.buffer_size = 100
    
    def process_event(self, event):
        """Process single social media event."""
        self.buffer.append(event)
        
        if len(self.buffer) >= self.buffer_size:
            # Convert buffer to DataFrame and detect
            df = pd.DataFrame(self.buffer)
            df_processed = self.detector.preprocess_trend_data(df)
            
            # Check for anomalies in latest data
            anomalies = self.detector.detect_trend_anomalies(df_processed.tail(10))
            
            if anomalies.any().any():
                self.alert_callback(f"ALERT: Anomaly detected at {df_processed.index[-1]}")
                self.alert_callback(f"Details: {anomalies[anomalies.any(axis=1)]}")
            
            # Keep only recent data in buffer
            self.buffer = self.buffer[-50:]
```

**Alert Prioritization**:

| Anomaly Type | Priority | Action |
|--------------|----------|--------|
| Viral negative sentiment | High | Immediate PR response |
| Bot attack detected | High | Security team alert |
| Unusual engagement spike | Medium | Marketing opportunity |
| Gradual trend decline | Low | Strategic review |

**Interview Tip**: Emphasize the need for domain context - what's anomalous on social media depends on the platform, topic, and time (e.g., election night vs. normal day).

---

## Question 9: How can transfer learning be applied to anomaly detection in different domains?
- [ ] Done

### Answer

**Concept**: Leverage knowledge from a source domain with labeled anomalies to detect anomalies in a target domain with limited/no labels.

**Transfer Learning Scenarios**:

| Scenario | Source | Target | Transfer Method |
|----------|--------|--------|-----------------|
| **Cross-domain** | Manufacturing defects | Medical imaging | Feature transfer |
| **Cross-task** | Fraud detection | Money laundering | Model adaptation |
| **Cross-time** | Historical patterns | Current data | Incremental learning |

**Approaches**:

```
1. Feature-based Transfer:
   Source domain → Feature extractor → Shared features → Target detector
   
2. Instance-based Transfer:
   Select relevant source samples → Weight by similarity → Train on weighted data
   
3. Model-based Transfer:
   Pre-trained model → Fine-tune on target → Detect anomalies
```

**Python Implementation**:

```python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest
import tensorflow as tf

class TransferAnomalyDetector:
    """Transfer learning for anomaly detection across domains."""
    
    def __init__(self, feature_dim=64):
        self.feature_dim = feature_dim
        self.feature_extractor = None
        self.detector = None
    
    def build_feature_extractor(self, input_dim):
        """Build neural network feature extractor."""
        self.feature_extractor = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(self.feature_dim, activation='relu')
        ])
        return self.feature_extractor
    
    def train_on_source(self, X_source, y_source):
        """
        Train feature extractor on source domain with labels.
        """
        input_dim = X_source.shape[1]
        
        # Build full classifier for source
        model = tf.keras.Sequential([
            self.build_feature_extractor(input_dim),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_source, y_source, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        
        print(f"Source domain training complete")
        return self
    
    def transfer_to_target(self, X_target, freeze_features=True):
        """
        Transfer to target domain (unsupervised or few-shot).
        """
        # Extract features using pre-trained extractor
        target_features = self.feature_extractor.predict(X_target)
        
        # Train anomaly detector on target features
        self.detector = IsolationForest(contamination=0.1)
        self.detector.fit(target_features)
        
        return self
    
    def detect(self, X):
        """Detect anomalies in target domain."""
        features = self.feature_extractor.predict(X)
        return self.detector.predict(features)


class DomainAdaptiveDetector:
    """Domain adaptation for anomaly detection using MMD."""
    
    def __init__(self, kernel_bandwidth=1.0):
        self.kernel_bandwidth = kernel_bandwidth
    
    def compute_mmd(self, X_source, X_target):
        """
        Compute Maximum Mean Discrepancy between domains.
        """
        def rbf_kernel(X, Y, bandwidth):
            XX = np.sum(X ** 2, axis=1, keepdims=True)
            YY = np.sum(Y ** 2, axis=1, keepdims=True)
            distances = XX + YY.T - 2 * np.dot(X, Y.T)
            return np.exp(-distances / (2 * bandwidth ** 2))
        
        K_ss = rbf_kernel(X_source, X_source, self.kernel_bandwidth)
        K_tt = rbf_kernel(X_target, X_target, self.kernel_bandwidth)
        K_st = rbf_kernel(X_source, X_target, self.kernel_bandwidth)
        
        mmd = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
        return mmd
    
    def select_transferable_samples(self, X_source, X_target, n_samples=100):
        """
        Select source samples most similar to target domain.
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Find nearest target samples for each source sample
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(X_target)
        
        distances, _ = nn.kneighbors(X_source)
        mean_distances = distances.mean(axis=1)
        
        # Select source samples closest to target
        selected_idx = np.argsort(mean_distances)[:n_samples]
        
        return X_source[selected_idx]


# Pre-trained anomaly detection (like using ImageNet features)
class PretrainedFeatureAnomalyDetector:
    """Use pre-trained deep learning features for anomaly detection."""
    
    def __init__(self, model_name='resnet50'):
        from tensorflow.keras.applications import ResNet50
        
        # Load pre-trained model without top layers
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.feature_extractor = base_model
        self.detector = None
    
    def extract_features(self, images):
        """Extract features from images."""
        from tensorflow.keras.applications.resnet50 import preprocess_input
        
        images_processed = preprocess_input(images)
        features = self.feature_extractor.predict(images_processed)
        return features
    
    def fit(self, images):
        """Fit anomaly detector on extracted features."""
        features = self.extract_features(images)
        self.detector = IsolationForest(contamination=0.1)
        self.detector.fit(features)
        return self
    
    def predict(self, images):
        """Detect anomalies in new images."""
        features = self.extract_features(images)
        return self.detector.predict(features)
```

**When Transfer Learning Helps**:

| Condition | Benefit |
|-----------|---------|
| Similar feature spaces | Direct feature transfer |
| Shared anomaly patterns | Model transfer |
| Limited target labels | Leverage source labels |
| Domain shift is moderate | Adaptation techniques |

**Challenges**:

| Challenge | Mitigation |
|-----------|------------|
| Negative transfer | Measure domain similarity first |
| Different anomaly types | Focus on shared patterns |
| Feature mismatch | Domain adaptation techniques |

**Interview Tip**: Transfer learning for anomaly detection is powerful when the underlying patterns of "normal" and "anomalous" share characteristics across domains.

---
