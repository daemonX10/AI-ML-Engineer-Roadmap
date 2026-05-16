# Anomaly Detection Interview Questions - Coding Questions

---

## Question 1: Write a Python function to identify outliers in a dataset using the IQR (Interquartile Range) method
- [ ] Done

### Complete Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def detect_outliers_iqr(data, columns=None, k=1.5, return_bounds=False):
    """
    Detect outliers using the IQR (Interquartile Range) method.
    
    The IQR method identifies outliers as points falling below Q1 - k*IQR
    or above Q3 + k*IQR, where IQR = Q3 - Q1.
    
    Args:
        data: numpy array, pandas Series, or DataFrame
        columns: list of columns to check (for DataFrame)
        k: IQR multiplier (default 1.5 for outliers, 3.0 for extreme outliers)
        return_bounds: whether to return the computed bounds
    
    Returns:
        outlier_mask: boolean mask where True indicates outlier
        bounds (optional): dict with lower and upper bounds
    """
    # Handle different input types
    if isinstance(data, pd.DataFrame):
        return _detect_outliers_dataframe(data, columns, k, return_bounds)
    elif isinstance(data, pd.Series):
        data = data.values
    
    data = np.array(data).flatten()
    
    # Calculate quartiles
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    # Calculate bounds
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    # Identify outliers
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    
    if return_bounds:
        bounds = {
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        return outlier_mask, bounds
    
    return outlier_mask


def _detect_outliers_dataframe(df, columns=None, k=1.5, return_bounds=False):
    """Handle DataFrame input."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outlier_mask = pd.DataFrame(False, index=df.index, columns=columns)
    bounds = {}
    
    for col in columns:
        mask, col_bounds = detect_outliers_iqr(df[col], k=k, return_bounds=True)
        outlier_mask[col] = mask
        bounds[col] = col_bounds
    
    if return_bounds:
        return outlier_mask, bounds
    return outlier_mask


def remove_outliers_iqr(data, columns=None, k=1.5):
    """
    Remove outliers from dataset using IQR method.
    
    Args:
        data: pandas DataFrame or numpy array
        columns: columns to check for outliers
        k: IQR multiplier
    
    Returns:
        Cleaned data with outliers removed
    """
    if isinstance(data, pd.DataFrame):
        outlier_mask = detect_outliers_iqr(data, columns, k)
        # Remove rows where ANY column has outlier
        rows_with_outliers = outlier_mask.any(axis=1)
        return data[~rows_with_outliers].copy()
    else:
        outlier_mask = detect_outliers_iqr(data, k=k)
        return data[~outlier_mask]


def cap_outliers_iqr(data, columns=None, k=1.5):
    """
    Cap (winsorize) outliers to the boundary values.
    
    Args:
        data: pandas DataFrame or numpy array
        columns: columns to cap
        k: IQR multiplier
    
    Returns:
        Data with outliers capped to bounds
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            _, bounds = detect_outliers_iqr(df[col], k=k, return_bounds=True)
            df[col] = df[col].clip(lower=bounds['lower_bound'], upper=bounds['upper_bound'])
        
        return df
    else:
        data = np.array(data).copy()
        _, bounds = detect_outliers_iqr(data, k=k, return_bounds=True)
        return np.clip(data, bounds['lower_bound'], bounds['upper_bound'])


def visualize_outliers(data, column=None, k=1.5):
    """Visualize outliers using box plot and histogram."""
    if isinstance(data, pd.DataFrame) and column is not None:
        values = data[column].values
        title = f'Outlier Analysis: {column}'
    else:
        values = np.array(data).flatten()
        title = 'Outlier Analysis'
    
    outliers, bounds = detect_outliers_iqr(values, k=k, return_bounds=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    axes[0].boxplot(values, vert=True)
    axes[0].axhline(bounds['lower_bound'], color='r', linestyle='--', label=f'Lower bound: {bounds["lower_bound"]:.2f}')
    axes[0].axhline(bounds['upper_bound'], color='r', linestyle='--', label=f'Upper bound: {bounds["upper_bound"]:.2f}')
    axes[0].set_title('Box Plot')
    axes[0].legend()
    
    # Histogram
    axes[1].hist(values, bins=50, alpha=0.7, label='All data')
    axes[1].hist(values[outliers], bins=50, alpha=0.7, color='red', label=f'Outliers ({outliers.sum()})')
    axes[1].axvline(bounds['lower_bound'], color='r', linestyle='--')
    axes[1].axvline(bounds['upper_bound'], color='r', linestyle='--')
    axes[1].set_title('Distribution with Outliers Highlighted')
    axes[1].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    print(f"\nOutlier Statistics:")
    print(f"  Total points: {len(values)}")
    print(f"  Outliers: {outliers.sum()} ({100*outliers.mean():.2f}%)")
    print(f"  Q1: {bounds['q1']:.4f}")
    print(f"  Q3: {bounds['q3']:.4f}")
    print(f"  IQR: {bounds['iqr']:.4f}")
    print(f"  Lower bound: {bounds['lower_bound']:.4f}")
    print(f"  Upper bound: {bounds['upper_bound']:.4f}")


# Example usage
if __name__ == "__main__":
    # Generate sample data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(100, 15, 1000)
    outliers = np.array([20, 30, 180, 200, 250])
    data = np.concatenate([normal_data, outliers])
    
    # Detect outliers
    outlier_mask = detect_outliers_iqr(data)
    print(f"Detected {outlier_mask.sum()} outliers")
    
    # With bounds
    outlier_mask, bounds = detect_outliers_iqr(data, return_bounds=True)
    print(f"Bounds: [{bounds['lower_bound']:.2f}, {bounds['upper_bound']:.2f}]")
    
    # Visualize
    visualize_outliers(data)
    
    # DataFrame example
    df = pd.DataFrame({
        'A': np.concatenate([np.random.normal(50, 10, 100), [100, 5]]),
        'B': np.concatenate([np.random.normal(30, 5, 100), [60, 10]])
    })
    
    outlier_df = detect_outliers_iqr(df)
    print(f"\nDataFrame outliers:\n{outlier_df.sum()}")
```

---

## Question 2: Implement a k-NN algorithm to detect anomalies in a two-dimensional dataset
- [ ] Done

### Complete Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

class KNNAnomalyDetector:
    """
    k-Nearest Neighbors based anomaly detection.
    
    Anomaly score is based on distance to k-th nearest neighbor
    or average distance to k nearest neighbors.
    """
    
    def __init__(self, k=5, method='kth', contamination=0.1):
        """
        Args:
            k: Number of neighbors
            method: 'kth' (distance to k-th neighbor) or 'mean' (average distance)
            contamination: Expected proportion of anomalies
        """
        self.k = k
        self.method = method
        self.contamination = contamination
        self.nn_model = None
        self.threshold = None
        self.X_train = None
    
    def fit(self, X):
        """
        Fit the k-NN model on training data.
        
        Args:
            X: Training data (n_samples, n_features)
        """
        self.X_train = np.array(X)
        self.nn_model = NearestNeighbors(n_neighbors=self.k + 1)  # +1 to exclude self
        self.nn_model.fit(self.X_train)
        
        # Compute threshold from training data
        scores = self.score_samples(self.X_train)
        self.threshold = np.percentile(scores, 100 * (1 - self.contamination))
        
        return self
    
    def score_samples(self, X):
        """
        Compute anomaly scores for samples.
        
        Higher score = more anomalous
        """
        X = np.array(X)
        distances, indices = self.nn_model.kneighbors(X)
        
        # Exclude self if point is in training data
        distances = distances[:, 1:]  # Remove distance to self (0)
        
        if self.method == 'kth':
            # Distance to k-th nearest neighbor
            scores = distances[:, -1]
        elif self.method == 'mean':
            # Average distance to k nearest neighbors
            scores = distances.mean(axis=1)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return scores
    
    def predict(self, X):
        """
        Predict anomalies.
        
        Returns:
            -1 for anomalies, 1 for normal
        """
        scores = self.score_samples(X)
        predictions = np.where(scores > self.threshold, -1, 1)
        return predictions
    
    def fit_predict(self, X):
        """Fit and predict on same data."""
        self.fit(X)
        return self.predict(X)


def knn_anomaly_detection_manual(X, k=5, method='kth'):
    """
    Manual implementation without sklearn for educational purposes.
    
    Args:
        X: Data array (n_samples, n_features)
        k: Number of neighbors
        method: 'kth' or 'mean'
    
    Returns:
        anomaly_scores: Score for each point
    """
    X = np.array(X)
    n_samples = len(X)
    
    # Compute pairwise distances
    distances = cdist(X, X, metric='euclidean')
    
    # Set diagonal to infinity (exclude self)
    np.fill_diagonal(distances, np.inf)
    
    # Sort distances for each point
    sorted_distances = np.sort(distances, axis=1)
    
    # Get k nearest neighbor distances
    k_distances = sorted_distances[:, :k]
    
    if method == 'kth':
        scores = k_distances[:, -1]  # k-th neighbor distance
    elif method == 'mean':
        scores = k_distances.mean(axis=1)  # average distance
    
    return scores


def visualize_knn_anomalies(X, k=5, contamination=0.1):
    """
    Visualize k-NN anomaly detection results for 2D data.
    """
    detector = KNNAnomalyDetector(k=k, contamination=contamination)
    predictions = detector.fit_predict(X)
    scores = detector.score_samples(X)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Data with anomalies highlighted
    normal_mask = predictions == 1
    axes[0].scatter(X[normal_mask, 0], X[normal_mask, 1], c='blue', alpha=0.6, label='Normal')
    axes[0].scatter(X[~normal_mask, 0], X[~normal_mask, 1], c='red', s=100, marker='x', label='Anomaly')
    axes[0].set_title(f'k-NN Anomaly Detection (k={k})')
    axes[0].legend()
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    # Plot 2: Anomaly scores as heatmap
    scatter = axes[1].scatter(X[:, 0], X[:, 1], c=scores, cmap='YlOrRd', alpha=0.7)
    plt.colorbar(scatter, ax=axes[1], label='Anomaly Score')
    axes[1].set_title('Anomaly Scores')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    
    # Plot 3: Score distribution
    axes[2].hist(scores, bins=50, alpha=0.7)
    axes[2].axvline(detector.threshold, color='r', linestyle='--', 
                   label=f'Threshold: {detector.threshold:.3f}')
    axes[2].set_title('Score Distribution')
    axes[2].set_xlabel('Anomaly Score')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Detected {(predictions == -1).sum()} anomalies out of {len(X)} points")
    return predictions, scores


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate 2D data with clusters and outliers
    n_normal = 300
    n_outliers = 15
    
    # Normal data: two clusters
    cluster1 = np.random.randn(n_normal // 2, 2) + [2, 2]
    cluster2 = np.random.randn(n_normal // 2, 2) + [-2, -2]
    
    # Outliers: scattered around
    outliers = np.random.uniform(-6, 6, (n_outliers, 2))
    
    X = np.vstack([cluster1, cluster2, outliers])
    
    # Detect anomalies
    predictions, scores = visualize_knn_anomalies(X, k=10, contamination=0.05)
    
    # Compare methods
    print("\nComparing k-NN methods:")
    for method in ['kth', 'mean']:
        detector = KNNAnomalyDetector(k=10, method=method, contamination=0.05)
        preds = detector.fit_predict(X)
        print(f"  {method}: {(preds == -1).sum()} anomalies detected")
```

---

## Question 3: Code an example of using the Isolation Forest algorithm with scikit-learn
- [ ] Done

### Complete Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def isolation_forest_detection(X, contamination=0.1, n_estimators=100, 
                               max_samples='auto', random_state=42):
    """
    Perform anomaly detection using Isolation Forest.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        contamination: Expected proportion of anomalies
        n_estimators: Number of trees in the forest
        max_samples: Number of samples to draw for each tree
        random_state: Random seed
    
    Returns:
        predictions: -1 for anomalies, 1 for normal
        scores: Anomaly scores (lower = more anomalous)
    """
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1
    )
    
    predictions = iso_forest.fit_predict(X)
    scores = iso_forest.decision_function(X)
    
    return predictions, scores, iso_forest


def isolation_forest_pipeline(X_train, X_test=None, y_test=None, 
                             contamination=0.1, scale=True):
    """
    Complete Isolation Forest pipeline with preprocessing and evaluation.
    
    Args:
        X_train: Training data
        X_test: Test data (optional)
        y_test: True labels for test data (optional, for evaluation)
        contamination: Expected anomaly proportion
        scale: Whether to standardize features
    
    Returns:
        results: Dictionary with model, predictions, and metrics
    """
    results = {}
    
    # Preprocessing
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        results['scaler'] = scaler
    else:
        X_train_scaled = X_train
    
    # Train Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    iso_forest.fit(X_train_scaled)
    results['model'] = iso_forest
    
    # Training predictions
    train_predictions = iso_forest.predict(X_train_scaled)
    train_scores = iso_forest.decision_function(X_train_scaled)
    results['train_predictions'] = train_predictions
    results['train_scores'] = train_scores
    
    print(f"Training set: {(train_predictions == -1).sum()} anomalies detected "
          f"out of {len(X_train)} samples")
    
    # Test set evaluation if provided
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test) if scale else X_test
        test_predictions = iso_forest.predict(X_test_scaled)
        test_scores = iso_forest.decision_function(X_test_scaled)
        
        results['test_predictions'] = test_predictions
        results['test_scores'] = test_scores
        
        print(f"Test set: {(test_predictions == -1).sum()} anomalies detected "
              f"out of {len(X_test)} samples")
        
        # Evaluate if labels provided
        if y_test is not None:
            # Convert to same format (1 = anomaly, 0 = normal)
            y_pred = (test_predictions == -1).astype(int)
            
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
            
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            
            # ROC-AUC (use negative scores since lower = anomaly)
            auc = roc_auc_score(y_test, -test_scores)
            print(f"\nROC-AUC: {auc:.4f}")
            results['auc'] = auc
    
    return results


def visualize_isolation_forest(X, predictions, scores, feature_names=None):
    """
    Visualize Isolation Forest results.
    
    Args:
        X: Feature matrix
        predictions: Anomaly predictions (-1 or 1)
        scores: Decision function scores
        feature_names: Names of features
    """
    n_features = X.shape[1]
    
    if n_features == 2:
        # 2D visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Scatter with anomalies
        normal_mask = predictions == 1
        axes[0].scatter(X[normal_mask, 0], X[normal_mask, 1], 
                       c='blue', alpha=0.6, label='Normal')
        axes[0].scatter(X[~normal_mask, 0], X[~normal_mask, 1], 
                       c='red', s=100, marker='x', label='Anomaly')
        axes[0].set_title('Isolation Forest Predictions')
        axes[0].legend()
        
        # Plot 2: Decision boundary
        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
            np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
        )
        
        # This requires refitting - simplified version
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(X)
        Z = iso_forest.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        axes[1].contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.5)
        axes[1].contour(xx, yy, Z, levels=[0], colors='red', linewidths=2)
        axes[1].scatter(X[:, 0], X[:, 1], c='black', s=10)
        axes[1].set_title('Decision Boundary (red = threshold)')
        
        # Plot 3: Score distribution
        axes[2].hist(scores, bins=50, alpha=0.7)
        axes[2].axvline(0, color='r', linestyle='--', label='Threshold')
        axes[2].set_title('Anomaly Score Distribution')
        axes[2].legend()
        
    else:
        # High-dimensional: show score distribution and top features
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Score distribution
        axes[0].hist(scores[predictions == 1], bins=30, alpha=0.7, label='Normal')
        axes[0].hist(scores[predictions == -1], bins=30, alpha=0.7, label='Anomaly')
        axes[0].axvline(0, color='r', linestyle='--')
        axes[0].set_title('Score Distribution by Class')
        axes[0].legend()
        
        # Feature comparison
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]
        
        normal_means = X[predictions == 1].mean(axis=0)
        anomaly_means = X[predictions == -1].mean(axis=0)
        
        x_pos = np.arange(min(10, n_features))
        width = 0.35
        
        axes[1].bar(x_pos - width/2, normal_means[:10], width, label='Normal')
        axes[1].bar(x_pos + width/2, anomaly_means[:10], width, label='Anomaly')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(feature_names[:10], rotation=45)
        axes[1].set_title('Feature Means: Normal vs Anomaly')
        axes[1].legend()
    
    plt.tight_layout()
    plt.show()


def tune_isolation_forest(X, y=None, contamination_range=None):
    """
    Tune Isolation Forest hyperparameters.
    
    Args:
        X: Feature matrix
        y: True labels (optional)
        contamination_range: Range of contamination values to try
    
    Returns:
        best_params: Best hyperparameters found
    """
    if contamination_range is None:
        contamination_range = [0.01, 0.05, 0.1, 0.15, 0.2]
    
    results = []
    
    for contamination in contamination_range:
        for n_estimators in [50, 100, 200]:
            iso_forest = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=42
            )
            
            predictions = iso_forest.fit_predict(X)
            scores = iso_forest.decision_function(X)
            
            result = {
                'contamination': contamination,
                'n_estimators': n_estimators,
                'n_anomalies': (predictions == -1).sum()
            }
            
            if y is not None:
                y_pred = (predictions == -1).astype(int)
                result['auc'] = roc_auc_score(y, -scores)
            
            results.append(result)
    
    results_df = pd.DataFrame(results)
    
    if y is not None:
        best_idx = results_df['auc'].idxmax()
        print(f"Best AUC: {results_df.loc[best_idx, 'auc']:.4f}")
    else:
        print("No labels provided - showing all results")
    
    print(results_df.to_string())
    
    return results_df


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate synthetic data
    n_normal = 500
    n_outliers = 25
    
    # Normal data
    X_normal = np.random.randn(n_normal, 2) * 2 + [5, 5]
    
    # Outliers
    X_outliers = np.random.uniform(-5, 15, (n_outliers, 2))
    
    X = np.vstack([X_normal, X_outliers])
    y = np.array([0] * n_normal + [1] * n_outliers)
    
    # Basic detection
    print("Basic Isolation Forest Detection:")
    predictions, scores, model = isolation_forest_detection(X, contamination=0.05)
    
    # Visualize
    visualize_isolation_forest(X, predictions, scores)
    
    # Full pipeline with evaluation
    print("\nFull Pipeline with Evaluation:")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    results = isolation_forest_pipeline(X_train, X_test, y_test, contamination=0.05)
```

---

## Question 4: Simulate a dataset with outliers and demonstrate how PCA can be used to detect these points
- [ ] Done

### Complete Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats

def generate_data_with_outliers(n_samples=500, n_features=5, n_outliers=25, 
                                outlier_type='random', random_state=42):
    """
    Generate synthetic data with injected outliers.
    
    Args:
        n_samples: Number of normal samples
        n_features: Number of features
        n_outliers: Number of outliers to inject
        outlier_type: 'random', 'shift', or 'correlation_break'
        random_state: Random seed
    
    Returns:
        X: Combined data
        y: Labels (0=normal, 1=outlier)
    """
    np.random.seed(random_state)
    
    # Generate correlated normal data
    mean = np.zeros(n_features)
    cov = np.eye(n_features)
    # Add correlation structure
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                cov[i, j] = 0.5 ** abs(i - j)
    
    X_normal = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Generate outliers based on type
    if outlier_type == 'random':
        X_outliers = np.random.uniform(
            X_normal.min() - 3 * X_normal.std(),
            X_normal.max() + 3 * X_normal.std(),
            (n_outliers, n_features)
        )
    elif outlier_type == 'shift':
        X_outliers = np.random.multivariate_normal(mean + 5, cov * 0.5, n_outliers)
    elif outlier_type == 'correlation_break':
        # Outliers that break correlation structure
        X_outliers = np.random.randn(n_outliers, n_features) * 3
    
    X = np.vstack([X_normal, X_outliers])
    y = np.array([0] * n_samples + [1] * n_outliers)
    
    return X, y


class PCAAnomalyDetector:
    """
    PCA-based anomaly detection using reconstruction error and T² statistic.
    """
    
    def __init__(self, n_components=None, variance_threshold=0.95):
        """
        Args:
            n_components: Number of PCs to retain (None = auto based on variance)
            variance_threshold: Cumulative variance to retain if n_components is None
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca = None
        self.scaler = StandardScaler()
        self.q_threshold = None
        self.t2_threshold = None
    
    def fit(self, X):
        """Fit PCA model on normal data."""
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        if self.n_components is None:
            # Auto-select based on variance
            pca_full = PCA()
            pca_full.fit(X_scaled)
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            self.n_components = np.argmax(cumsum >= self.variance_threshold) + 1
        
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_scaled)
        
        # Compute thresholds from training data
        q_scores, t2_scores = self._compute_scores(X_scaled)
        self.q_threshold = np.percentile(q_scores, 95)
        self.t2_threshold = np.percentile(t2_scores, 95)
        
        print(f"PCA fitted with {self.n_components} components")
        print(f"Explained variance: {self.pca.explained_variance_ratio_.sum():.2%}")
        
        return self
    
    def _compute_scores(self, X_scaled):
        """Compute Q and T² statistics."""
        # Project and reconstruct
        X_transformed = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        
        # Q-statistic (Squared Prediction Error)
        reconstruction_error = X_scaled - X_reconstructed
        q_scores = np.sum(reconstruction_error ** 2, axis=1)
        
        # T² statistic (Hotelling's T-squared)
        t2_scores = np.sum(
            (X_transformed ** 2) / self.pca.explained_variance_, 
            axis=1
        )
        
        return q_scores, t2_scores
    
    def detect(self, X):
        """
        Detect anomalies in new data.
        
        Returns:
            predictions: Boolean mask (True = anomaly)
            q_scores: Q-statistic values
            t2_scores: T²-statistic values
        """
        X_scaled = self.scaler.transform(X)
        q_scores, t2_scores = self._compute_scores(X_scaled)
        
        # Anomaly if either score exceeds threshold
        q_anomaly = q_scores > self.q_threshold
        t2_anomaly = t2_scores > self.t2_threshold
        predictions = q_anomaly | t2_anomaly
        
        return predictions, q_scores, t2_scores
    
    def contribution_analysis(self, X, sample_idx):
        """
        Analyze which features contribute most to anomaly for a sample.
        """
        X_scaled = self.scaler.transform(X)
        sample = X_scaled[sample_idx:sample_idx+1]
        
        # Reconstruction
        transformed = self.pca.transform(sample)
        reconstructed = self.pca.inverse_transform(transformed)
        
        # Per-feature reconstruction error
        feature_errors = (sample - reconstructed) ** 2
        
        return feature_errors.flatten()


def visualize_pca_anomaly_detection(X, y, detector):
    """
    Comprehensive visualization of PCA anomaly detection.
    """
    predictions, q_scores, t2_scores = detector.detect(X)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: PCA projection with true labels
    X_scaled = detector.scaler.transform(X)
    X_pca = detector.pca.transform(X_scaled)
    
    axes[0, 0].scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], 
                       c='blue', alpha=0.6, label='Normal')
    axes[0, 0].scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], 
                       c='red', s=100, marker='x', label='Outlier')
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    axes[0, 0].set_title('PCA Projection (True Labels)')
    axes[0, 0].legend()
    
    # Plot 2: PCA projection with predictions
    axes[0, 1].scatter(X_pca[~predictions, 0], X_pca[~predictions, 1], 
                       c='blue', alpha=0.6, label='Predicted Normal')
    axes[0, 1].scatter(X_pca[predictions, 0], X_pca[predictions, 1], 
                       c='red', s=100, marker='x', label='Predicted Anomaly')
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    axes[0, 1].set_title('PCA Projection (Predictions)')
    axes[0, 1].legend()
    
    # Plot 3: Q vs T² plot
    axes[0, 2].scatter(t2_scores[y == 0], q_scores[y == 0], 
                       c='blue', alpha=0.6, label='Normal')
    axes[0, 2].scatter(t2_scores[y == 1], q_scores[y == 1], 
                       c='red', s=100, marker='x', label='Outlier')
    axes[0, 2].axhline(detector.q_threshold, color='r', linestyle='--', alpha=0.5)
    axes[0, 2].axvline(detector.t2_threshold, color='r', linestyle='--', alpha=0.5)
    axes[0, 2].set_xlabel('T² Statistic')
    axes[0, 2].set_ylabel('Q Statistic')
    axes[0, 2].set_title('Q vs T² (red lines = thresholds)')
    axes[0, 2].legend()
    
    # Plot 4: Q-statistic distribution
    axes[1, 0].hist(q_scores[y == 0], bins=30, alpha=0.7, label='Normal')
    axes[1, 0].hist(q_scores[y == 1], bins=30, alpha=0.7, label='Outlier')
    axes[1, 0].axvline(detector.q_threshold, color='r', linestyle='--', 
                       label=f'Threshold: {detector.q_threshold:.2f}')
    axes[1, 0].set_xlabel('Q Statistic')
    axes[1, 0].set_title('Q-Statistic Distribution')
    axes[1, 0].legend()
    
    # Plot 5: T²-statistic distribution
    axes[1, 1].hist(t2_scores[y == 0], bins=30, alpha=0.7, label='Normal')
    axes[1, 1].hist(t2_scores[y == 1], bins=30, alpha=0.7, label='Outlier')
    axes[1, 1].axvline(detector.t2_threshold, color='r', linestyle='--',
                       label=f'Threshold: {detector.t2_threshold:.2f}')
    axes[1, 1].set_xlabel('T² Statistic')
    axes[1, 1].set_title('T²-Statistic Distribution')
    axes[1, 1].legend()
    
    # Plot 6: Explained variance
    variance_ratio = detector.pca.explained_variance_ratio_
    cumsum = np.cumsum(variance_ratio)
    
    axes[1, 2].bar(range(1, len(variance_ratio) + 1), variance_ratio, alpha=0.7, label='Individual')
    axes[1, 2].plot(range(1, len(cumsum) + 1), cumsum, 'r-o', label='Cumulative')
    axes[1, 2].set_xlabel('Principal Component')
    axes[1, 2].set_ylabel('Variance Explained')
    axes[1, 2].set_title('Explained Variance Ratio')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    print("\nDetection Results:")
    print(f"True outliers: {y.sum()}")
    print(f"Detected anomalies: {predictions.sum()}")
    print(f"True positives: {((predictions) & (y == 1)).sum()}")
    print(f"False positives: {((predictions) & (y == 0)).sum()}")


# Example usage
if __name__ == "__main__":
    # Generate data
    X, y = generate_data_with_outliers(
        n_samples=500, 
        n_features=10, 
        n_outliers=25,
        outlier_type='correlation_break'
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Outliers: {y.sum()}")
    
    # Fit PCA anomaly detector
    detector = PCAAnomalyDetector(variance_threshold=0.90)
    detector.fit(X[y == 0])  # Fit only on normal data
    
    # Detect anomalies
    predictions, q_scores, t2_scores = detector.detect(X)
    
    # Visualize
    visualize_pca_anomaly_detection(X, y, detector)
```

---

## Question 5: Use TensorFlow/Keras to build a simple autoencoder for anomaly detection on a sample dataset
- [ ] Done

### Complete Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

class AutoencoderAnomalyDetector:
    """
    Autoencoder-based anomaly detection using TensorFlow/Keras.
    """
    
    def __init__(self, input_dim, encoding_dim=32, hidden_layers=[64, 32]):
        """
        Args:
            input_dim: Number of input features
            encoding_dim: Dimension of the bottleneck layer
            hidden_layers: List of hidden layer sizes for encoder
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.model = None
        self.encoder = None
        self.decoder = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.history = None
    
    def _build_model(self):
        """Build the autoencoder architecture."""
        # Encoder
        encoder_input = keras.layers.Input(shape=(self.input_dim,))
        x = encoder_input
        
        for units in self.hidden_layers:
            x = keras.layers.Dense(units, activation='relu')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)
        
        # Bottleneck
        encoded = keras.layers.Dense(self.encoding_dim, activation='relu', name='encoding')(x)
        
        # Decoder
        x = encoded
        for units in reversed(self.hidden_layers):
            x = keras.layers.Dense(units, activation='relu')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)
        
        decoded = keras.layers.Dense(self.input_dim, activation='linear')(x)
        
        # Full autoencoder
        self.model = keras.Model(encoder_input, decoded, name='autoencoder')
        
        # Encoder only (for embeddings)
        self.encoder = keras.Model(encoder_input, encoded, name='encoder')
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse'
        )
        
        return self.model
    
    def fit(self, X_normal, epochs=100, batch_size=32, validation_split=0.1, verbose=1):
        """
        Train autoencoder on normal data only.
        
        Args:
            X_normal: Training data (assumed to be normal/non-anomalous)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction for validation
            verbose: Verbosity level
        """
        # Scale data
        X_scaled = self.scaler.fit_transform(X_normal)
        
        # Build model
        self._build_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train (reconstruction target = input)
        self.history = self.model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Compute threshold from training reconstruction errors
        train_reconstructions = self.model.predict(X_scaled, verbose=0)
        train_errors = self._compute_reconstruction_error(X_scaled, train_reconstructions)
        
        # Set threshold at 95th or 99th percentile
        self.threshold = np.percentile(train_errors, 95)
        
        print(f"\nTraining complete. Threshold set at: {self.threshold:.6f}")
        
        return self
    
    def _compute_reconstruction_error(self, original, reconstructed):
        """Compute reconstruction error (MSE per sample)."""
        return np.mean((original - reconstructed) ** 2, axis=1)
    
    def predict(self, X, return_scores=False):
        """
        Predict anomalies.
        
        Returns:
            predictions: Boolean mask (True = anomaly)
            scores (optional): Reconstruction errors
        """
        X_scaled = self.scaler.transform(X)
        reconstructions = self.model.predict(X_scaled, verbose=0)
        errors = self._compute_reconstruction_error(X_scaled, reconstructions)
        
        predictions = errors > self.threshold
        
        if return_scores:
            return predictions, errors
        return predictions
    
    def evaluate(self, X, y_true):
        """
        Evaluate detector performance.
        
        Args:
            X: Test data
            y_true: True labels (1 = anomaly, 0 = normal)
        """
        predictions, scores = self.predict(X, return_scores=True)
        
        print("Classification Report:")
        print(classification_report(y_true, predictions.astype(int), 
                                   target_names=['Normal', 'Anomaly']))
        
        # ROC-AUC
        auc = roc_auc_score(y_true, scores)
        print(f"ROC-AUC: {auc:.4f}")
        
        return {
            'predictions': predictions,
            'scores': scores,
            'auc': auc
        }
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Training History')
        axes[0].legend()
        
        # Learning rate (if available)
        if 'lr' in self.history.history:
            axes[1].plot(self.history.history['lr'])
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('Learning Rate Schedule')
        
        plt.tight_layout()
        plt.show()


def visualize_autoencoder_results(X, y, detector):
    """Comprehensive visualization of autoencoder anomaly detection."""
    predictions, scores = detector.predict(X, return_scores=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Reconstruction error distribution
    axes[0, 0].hist(scores[y == 0], bins=50, alpha=0.7, label='Normal', density=True)
    axes[0, 0].hist(scores[y == 1], bins=50, alpha=0.7, label='Anomaly', density=True)
    axes[0, 0].axvline(detector.threshold, color='r', linestyle='--', 
                       label=f'Threshold: {detector.threshold:.4f}')
    axes[0, 0].set_xlabel('Reconstruction Error')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Reconstruction Error Distribution')
    axes[0, 0].legend()
    
    # Plot 2: Scatter plot of first two features
    if X.shape[1] >= 2:
        axes[0, 1].scatter(X[y == 0, 0], X[y == 0, 1], 
                          c='blue', alpha=0.6, label='True Normal')
        axes[0, 1].scatter(X[y == 1, 0], X[y == 1, 1], 
                          c='red', s=100, marker='x', label='True Anomaly')
        axes[0, 1].set_xlabel('Feature 1')
        axes[0, 1].set_ylabel('Feature 2')
        axes[0, 1].set_title('Data Distribution (First 2 Features)')
        axes[0, 1].legend()
    
    # Plot 3: ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, scores)
    auc = roc_auc_score(y, scores)
    
    axes[1, 0].plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    axes[1, 0].plot([0, 1], [0, 1], 'k--')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curve')
    axes[1, 0].legend()
    
    # Plot 4: Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y, scores)
    
    axes[1, 1].plot(recall, precision)
    axes[1, 1].axhline(y.mean(), color='r', linestyle='--', label=f'Baseline: {y.mean():.3f}')
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_title('Precision-Recall Curve')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Generate synthetic data
    n_samples = 1000
    n_features = 20
    n_anomalies = 50
    
    # Normal data (multivariate normal)
    X_normal = np.random.randn(n_samples, n_features)
    
    # Anomalies (different distribution)
    X_anomaly = np.random.randn(n_anomalies, n_features) * 3 + 5
    
    # Combine
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * n_samples + [1] * n_anomalies)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X, y = X[shuffle_idx], y[shuffle_idx]
    
    # Split (train only on normal data)
    X_train = X[y == 0][:800]  # 800 normal samples for training
    X_test = X
    y_test = y
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)} ({y_test.sum()} anomalies)")
    
    # Build and train autoencoder
    detector = AutoencoderAnomalyDetector(
        input_dim=n_features,
        encoding_dim=8,
        hidden_layers=[64, 32]
    )
    
    detector.fit(X_train, epochs=100, verbose=1)
    
    # Plot training history
    detector.plot_training_history()
    
    # Evaluate
    results = detector.evaluate(X_test, y_test)
    
    # Visualize
    visualize_autoencoder_results(X_test, y_test, detector)
```

---

## Question 6: Write an SQL query to spot potential anomalies in a transaction table based on statistical z-scores
- [ ] Done

### Complete Implementation

```sql
-- Anomaly Detection using Z-Scores in SQL
-- Database: PostgreSQL (with adaptations for other databases noted)

-- Sample table structure
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id SERIAL PRIMARY KEY,
    customer_id INT NOT NULL,
    amount DECIMAL(12, 2) NOT NULL,
    transaction_date TIMESTAMP NOT NULL,
    merchant_category VARCHAR(50),
    channel VARCHAR(20),
    location VARCHAR(100)
);

-- ============================================
-- Method 1: Global Z-Score Anomaly Detection
-- ============================================

-- Calculate z-scores for all transactions
WITH stats AS (
    SELECT 
        AVG(amount) AS mean_amount,
        STDDEV(amount) AS std_amount
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '30 days'
),
z_scores AS (
    SELECT 
        t.*,
        (t.amount - s.mean_amount) / NULLIF(s.std_amount, 0) AS z_score
    FROM transactions t
    CROSS JOIN stats s
    WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '30 days'
)
SELECT 
    transaction_id,
    customer_id,
    amount,
    transaction_date,
    z_score,
    CASE 
        WHEN ABS(z_score) > 3 THEN 'HIGH_ANOMALY'
        WHEN ABS(z_score) > 2 THEN 'MEDIUM_ANOMALY'
        ELSE 'NORMAL'
    END AS anomaly_flag
FROM z_scores
WHERE ABS(z_score) > 2
ORDER BY ABS(z_score) DESC;


-- ============================================
-- Method 2: Customer-Specific Z-Score
-- ============================================

-- Anomalies relative to each customer's spending pattern
WITH customer_stats AS (
    SELECT 
        customer_id,
        AVG(amount) AS mean_amount,
        STDDEV(amount) AS std_amount,
        COUNT(*) AS txn_count
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY customer_id
    HAVING COUNT(*) >= 10  -- Minimum transactions for reliable stats
),
customer_z_scores AS (
    SELECT 
        t.*,
        cs.mean_amount AS customer_avg,
        cs.std_amount AS customer_std,
        (t.amount - cs.mean_amount) / NULLIF(cs.std_amount, 0) AS customer_z_score
    FROM transactions t
    JOIN customer_stats cs ON t.customer_id = cs.customer_id
    WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '7 days'
)
SELECT 
    transaction_id,
    customer_id,
    amount,
    customer_avg,
    customer_std,
    customer_z_score,
    transaction_date,
    CASE 
        WHEN customer_z_score > 3 THEN 'UNUSUALLY_HIGH'
        WHEN customer_z_score < -3 THEN 'UNUSUALLY_LOW'
        ELSE 'NORMAL'
    END AS anomaly_type
FROM customer_z_scores
WHERE ABS(customer_z_score) > 3
ORDER BY customer_z_score DESC;


-- ============================================
-- Method 3: Category-Specific Z-Score
-- ============================================

-- Anomalies relative to merchant category norms
WITH category_stats AS (
    SELECT 
        merchant_category,
        AVG(amount) AS category_avg,
        STDDEV(amount) AS category_std,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount) AS q1,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) AS q3
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY merchant_category
),
category_anomalies AS (
    SELECT 
        t.*,
        cs.category_avg,
        cs.category_std,
        cs.q1,
        cs.q3,
        cs.q3 - cs.q1 AS iqr,
        (t.amount - cs.category_avg) / NULLIF(cs.category_std, 0) AS category_z_score,
        CASE 
            WHEN t.amount < cs.q1 - 1.5 * (cs.q3 - cs.q1) OR
                 t.amount > cs.q3 + 1.5 * (cs.q3 - cs.q1)
            THEN TRUE
            ELSE FALSE
        END AS iqr_outlier
    FROM transactions t
    JOIN category_stats cs ON t.merchant_category = cs.merchant_category
    WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '7 days'
)
SELECT 
    transaction_id,
    customer_id,
    amount,
    merchant_category,
    category_avg,
    category_z_score,
    iqr_outlier,
    transaction_date
FROM category_anomalies
WHERE ABS(category_z_score) > 3 OR iqr_outlier = TRUE
ORDER BY ABS(category_z_score) DESC;


-- ============================================
-- Method 4: Time-Based Anomaly Detection
-- ============================================

-- Detect unusual transaction frequency or amount by time
WITH hourly_stats AS (
    SELECT 
        customer_id,
        EXTRACT(HOUR FROM transaction_date) AS hour,
        COUNT(*) AS txn_count,
        SUM(amount) AS total_amount
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '24 hours'
    GROUP BY customer_id, EXTRACT(HOUR FROM transaction_date)
),
customer_hourly_baseline AS (
    SELECT 
        customer_id,
        EXTRACT(HOUR FROM transaction_date) AS hour,
        AVG(COUNT(*)) OVER (PARTITION BY customer_id) AS avg_hourly_txns,
        STDDEV(COUNT(*)) OVER (PARTITION BY customer_id) AS std_hourly_txns
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY customer_id, EXTRACT(HOUR FROM transaction_date)
)
SELECT 
    hs.customer_id,
    hs.hour,
    hs.txn_count,
    hs.total_amount,
    chb.avg_hourly_txns,
    (hs.txn_count - chb.avg_hourly_txns) / NULLIF(chb.std_hourly_txns, 0) AS velocity_z_score
FROM hourly_stats hs
JOIN customer_hourly_baseline chb 
    ON hs.customer_id = chb.customer_id AND hs.hour = chb.hour
WHERE (hs.txn_count - chb.avg_hourly_txns) / NULLIF(chb.std_hourly_txns, 0) > 3;


-- ============================================
-- Method 5: Combined Multi-Factor Anomaly Score
-- ============================================

-- Comprehensive anomaly detection combining multiple factors
WITH global_stats AS (
    SELECT 
        AVG(amount) AS global_mean,
        STDDEV(amount) AS global_std
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '90 days'
),
customer_stats AS (
    SELECT 
        customer_id,
        AVG(amount) AS cust_mean,
        STDDEV(amount) AS cust_std,
        COUNT(*) AS cust_txn_count
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY customer_id
),
recent_velocity AS (
    SELECT 
        customer_id,
        COUNT(*) AS recent_txn_count,
        SUM(amount) AS recent_total
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '24 hours'
    GROUP BY customer_id
),
anomaly_scores AS (
    SELECT 
        t.transaction_id,
        t.customer_id,
        t.amount,
        t.transaction_date,
        t.merchant_category,
        -- Global z-score
        (t.amount - gs.global_mean) / NULLIF(gs.global_std, 0) AS global_z,
        -- Customer z-score
        CASE WHEN cs.cust_std > 0 
             THEN (t.amount - cs.cust_mean) / cs.cust_std
             ELSE 0 
        END AS customer_z,
        -- Velocity factor
        COALESCE(rv.recent_txn_count, 0) AS velocity_24h,
        -- Combined anomaly score (weighted)
        (
            0.3 * ABS((t.amount - gs.global_mean) / NULLIF(gs.global_std, 0)) +
            0.5 * ABS(CASE WHEN cs.cust_std > 0 
                           THEN (t.amount - cs.cust_mean) / cs.cust_std
                           ELSE 0 END) +
            0.2 * COALESCE(rv.recent_txn_count, 0) / 10.0
        ) AS combined_score
    FROM transactions t
    CROSS JOIN global_stats gs
    LEFT JOIN customer_stats cs ON t.customer_id = cs.customer_id
    LEFT JOIN recent_velocity rv ON t.customer_id = rv.customer_id
    WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '1 day'
)
SELECT 
    transaction_id,
    customer_id,
    amount,
    transaction_date,
    merchant_category,
    ROUND(global_z::numeric, 2) AS global_z_score,
    ROUND(customer_z::numeric, 2) AS customer_z_score,
    velocity_24h,
    ROUND(combined_score::numeric, 2) AS anomaly_score,
    CASE 
        WHEN combined_score > 5 THEN 'CRITICAL'
        WHEN combined_score > 3 THEN 'HIGH'
        WHEN combined_score > 2 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS risk_level
FROM anomaly_scores
WHERE combined_score > 2
ORDER BY combined_score DESC
LIMIT 100;


-- ============================================
-- Create View for Real-Time Monitoring
-- ============================================

CREATE OR REPLACE VIEW v_transaction_anomalies AS
WITH stats AS (
    SELECT 
        customer_id,
        AVG(amount) AS mean_amount,
        STDDEV(amount) AS std_amount
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY customer_id
)
SELECT 
    t.transaction_id,
    t.customer_id,
    t.amount,
    t.transaction_date,
    s.mean_amount AS customer_avg,
    ROUND(((t.amount - s.mean_amount) / NULLIF(s.std_amount, 0))::numeric, 2) AS z_score,
    CASE 
        WHEN ABS((t.amount - s.mean_amount) / NULLIF(s.std_amount, 0)) > 3 THEN 'ANOMALY'
        ELSE 'NORMAL'
    END AS status
FROM transactions t
JOIN stats s ON t.customer_id = s.customer_id
WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '1 day';

-- Query the view
SELECT * FROM v_transaction_anomalies WHERE status = 'ANOMALY';
```

---

## Question 7: Implement a simple version of the Local Outlier Factor algorithm in Python
- [ ] Done

### Complete Implementation

```python
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class LocalOutlierFactor:
    """
    Local Outlier Factor (LOF) implementation from scratch.
    
    LOF compares the local density of a point with the local densities
    of its neighbors. Points with substantially lower density than their
    neighbors are considered outliers.
    """
    
    def __init__(self, k=5, contamination=0.1):
        """
        Args:
            k: Number of neighbors (minPts parameter)
            contamination: Expected proportion of outliers
        """
        self.k = k
        self.contamination = contamination
        self.X = None
        self.distances = None
        self.k_distances = None
        self.lrd = None
        self.lof_scores = None
    
    def fit(self, X):
        """
        Compute LOF scores for all points.
        
        Args:
            X: Data array (n_samples, n_features)
        """
        self.X = np.array(X)
        n_samples = len(self.X)
        
        # Step 1: Compute pairwise distances
        self.distances = cdist(self.X, self.X, metric='euclidean')
        
        # Set diagonal to infinity to exclude self
        np.fill_diagonal(self.distances, np.inf)
        
        # Step 2: Find k nearest neighbors for each point
        self.neighbor_indices = np.argsort(self.distances, axis=1)[:, :self.k]
        
        # Step 3: Compute k-distance for each point
        # k-distance(p) = distance to the k-th nearest neighbor
        sorted_distances = np.sort(self.distances, axis=1)
        self.k_distances = sorted_distances[:, self.k - 1]
        
        # Step 4: Compute reachability distances
        # reach-dist(p, o) = max(k-distance(o), d(p, o))
        self.reach_distances = np.zeros_like(self.distances)
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    self.reach_distances[i, j] = max(
                        self.k_distances[j], 
                        self.distances[i, j]
                    )
        
        # Step 5: Compute Local Reachability Density (LRD)
        # lrd(p) = 1 / (avg reachability distance to k neighbors)
        self.lrd = np.zeros(n_samples)
        for i in range(n_samples):
            neighbors = self.neighbor_indices[i]
            avg_reach_dist = np.mean([self.reach_distances[i, j] for j in neighbors])
            self.lrd[i] = 1.0 / (avg_reach_dist + 1e-10)  # Avoid division by zero
        
        # Step 6: Compute LOF scores
        # LOF(p) = avg(lrd(neighbors) / lrd(p))
        self.lof_scores = np.zeros(n_samples)
        for i in range(n_samples):
            neighbors = self.neighbor_indices[i]
            neighbor_lrds = self.lrd[neighbors]
            self.lof_scores[i] = np.mean(neighbor_lrds) / (self.lrd[i] + 1e-10)
        
        return self
    
    def predict(self):
        """
        Predict outliers based on LOF scores.
        
        Returns:
            predictions: 1 for inliers, -1 for outliers
        """
        if self.lof_scores is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Threshold based on contamination
        threshold = np.percentile(self.lof_scores, 100 * (1 - self.contamination))
        
        predictions = np.where(self.lof_scores > threshold, -1, 1)
        return predictions
    
    def fit_predict(self, X):
        """Fit and predict in one step."""
        self.fit(X)
        return self.predict()
    
    def get_scores(self):
        """Return LOF scores (higher = more anomalous)."""
        return self.lof_scores
    
    def explain_point(self, idx):
        """
        Explain why a point is considered anomalous.
        
        Args:
            idx: Index of the point
        
        Returns:
            Explanation dictionary
        """
        neighbors = self.neighbor_indices[idx]
        
        return {
            'point_idx': idx,
            'lof_score': self.lof_scores[idx],
            'lrd': self.lrd[idx],
            'k_distance': self.k_distances[idx],
            'neighbors': neighbors.tolist(),
            'neighbor_lrds': self.lrd[neighbors].tolist(),
            'neighbor_distances': self.distances[idx, neighbors].tolist(),
            'is_outlier': self.lof_scores[idx] > np.percentile(
                self.lof_scores, 100 * (1 - self.contamination)
            )
        }


def visualize_lof(X, lof_model, true_labels=None):
    """
    Visualize LOF results for 2D data.
    """
    predictions = lof_model.predict()
    scores = lof_model.get_scores()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Predictions
    inlier_mask = predictions == 1
    axes[0].scatter(X[inlier_mask, 0], X[inlier_mask, 1], 
                   c='blue', alpha=0.6, label='Inlier')
    axes[0].scatter(X[~inlier_mask, 0], X[~inlier_mask, 1], 
                   c='red', s=100, marker='x', label='Outlier')
    axes[0].set_title('LOF Predictions')
    axes[0].legend()
    
    # Plot 2: LOF scores as colors
    scatter = axes[1].scatter(X[:, 0], X[:, 1], c=scores, cmap='YlOrRd', alpha=0.7)
    plt.colorbar(scatter, ax=axes[1], label='LOF Score')
    axes[1].set_title('LOF Scores')
    
    # Plot 3: Score distribution
    axes[2].hist(scores, bins=50, alpha=0.7)
    threshold = np.percentile(scores, 100 * (1 - lof_model.contamination))
    axes[2].axvline(threshold, color='r', linestyle='--', 
                   label=f'Threshold: {threshold:.2f}')
    axes[2].set_xlabel('LOF Score')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('LOF Score Distribution')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Total points: {len(X)}")
    print(f"Detected outliers: {(predictions == -1).sum()}")
    print(f"Min LOF: {scores.min():.4f}")
    print(f"Max LOF: {scores.max():.4f}")
    print(f"Mean LOF: {scores.mean():.4f}")


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate data with outliers
    n_inliers = 200
    n_outliers = 10
    
    # Dense cluster
    X_inliers = np.random.randn(n_inliers, 2) * 0.5 + [2, 2]
    
    # Outliers
    X_outliers = np.random.uniform(-2, 6, (n_outliers, 2))
    
    X = np.vstack([X_inliers, X_outliers])
    y = np.array([0] * n_inliers + [1] * n_outliers)
    
    # Fit LOF
    lof = LocalOutlierFactor(k=10, contamination=0.05)
    predictions = lof.fit_predict(X)
    
    # Visualize
    visualize_lof(X, lof)
    
    # Explain a specific outlier
    outlier_indices = np.where(predictions == -1)[0]
    if len(outlier_indices) > 0:
        explanation = lof.explain_point(outlier_indices[0])
        print("\nOutlier Explanation:")
        for key, value in explanation.items():
            print(f"  {key}: {value}")
```

---

## Question 8: Create a Python script using pandas that flags outliers in time-series data based on moving averages
- [ ] Done

### Complete Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class TimeSeriesAnomalyDetector:
    """
    Time-series anomaly detection using moving averages and statistical methods.
    """
    
    def __init__(self, window_size=24, n_std=3):
        """
        Args:
            window_size: Size of the moving window
            n_std: Number of standard deviations for threshold
        """
        self.window_size = window_size
        self.n_std = n_std
    
    def detect_moving_average_anomalies(self, series):
        """
        Detect anomalies based on deviation from moving average.
        
        Args:
            series: pandas Series with datetime index
        
        Returns:
            DataFrame with anomaly analysis
        """
        df = pd.DataFrame({'value': series})
        
        # Calculate moving statistics
        df['rolling_mean'] = df['value'].rolling(window=self.window_size, center=True).mean()
        df['rolling_std'] = df['value'].rolling(window=self.window_size, center=True).std()
        
        # Calculate bounds
        df['upper_bound'] = df['rolling_mean'] + self.n_std * df['rolling_std']
        df['lower_bound'] = df['rolling_mean'] - self.n_std * df['rolling_std']
        
        # Calculate z-score
        df['z_score'] = (df['value'] - df['rolling_mean']) / df['rolling_std']
        
        # Flag anomalies
        df['is_anomaly'] = (df['value'] > df['upper_bound']) | (df['value'] < df['lower_bound'])
        
        return df
    
    def detect_ewm_anomalies(self, series, span=24):
        """
        Detect anomalies using Exponential Weighted Moving Average.
        
        Args:
            series: pandas Series
            span: Span for EWMA
        
        Returns:
            DataFrame with anomaly analysis
        """
        df = pd.DataFrame({'value': series})
        
        # EWMA mean and std
        df['ewm_mean'] = df['value'].ewm(span=span).mean()
        df['ewm_std'] = df['value'].ewm(span=span).std()
        
        # Bounds
        df['upper_bound'] = df['ewm_mean'] + self.n_std * df['ewm_std']
        df['lower_bound'] = df['ewm_mean'] - self.n_std * df['ewm_std']
        
        # Z-score
        df['z_score'] = (df['value'] - df['ewm_mean']) / df['ewm_std']
        
        # Anomalies
        df['is_anomaly'] = (df['value'] > df['upper_bound']) | (df['value'] < df['lower_bound'])
        
        return df
    
    def detect_seasonal_anomalies(self, series, period=24):
        """
        Detect anomalies considering seasonal patterns.
        
        Args:
            series: pandas Series with datetime index
            period: Seasonal period (e.g., 24 for hourly data with daily pattern)
        
        Returns:
            DataFrame with anomaly analysis
        """
        df = pd.DataFrame({'value': series})
        
        if hasattr(series.index, 'hour'):
            df['hour'] = series.index.hour
        else:
            df['hour'] = np.arange(len(series)) % period
        
        # Calculate seasonal baseline
        seasonal_stats = df.groupby('hour')['value'].agg(['mean', 'std']).reset_index()
        seasonal_stats.columns = ['hour', 'seasonal_mean', 'seasonal_std']
        
        df = df.merge(seasonal_stats, on='hour')
        
        # Bounds
        df['upper_bound'] = df['seasonal_mean'] + self.n_std * df['seasonal_std']
        df['lower_bound'] = df['seasonal_mean'] - self.n_std * df['seasonal_std']
        
        # Z-score
        df['z_score'] = (df['value'] - df['seasonal_mean']) / df['seasonal_std']
        
        # Anomalies
        df['is_anomaly'] = (df['value'] > df['upper_bound']) | (df['value'] < df['lower_bound'])
        
        # Restore index
        df.index = series.index
        
        return df


def detect_level_shift(series, window=50, threshold=3):
    """
    Detect level shifts (sudden changes in mean) in time series.
    
    Args:
        series: pandas Series
        window: Window size for calculating statistics
        threshold: Z-score threshold
    
    Returns:
        DataFrame with level shift detection
    """
    df = pd.DataFrame({'value': series})
    
    # Calculate difference between consecutive windows
    df['left_mean'] = df['value'].rolling(window=window).mean()
    df['right_mean'] = df['value'].shift(-window).rolling(window=window).mean()
    
    df['mean_diff'] = np.abs(df['right_mean'] - df['left_mean'])
    df['pooled_std'] = df['value'].rolling(window=2*window).std()
    
    df['level_shift_score'] = df['mean_diff'] / df['pooled_std']
    df['is_level_shift'] = df['level_shift_score'] > threshold
    
    return df


def visualize_time_series_anomalies(df, title="Time Series Anomaly Detection"):
    """
    Visualize time series with detected anomalies.
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Original series with bounds
    axes[0].plot(df.index, df['value'], 'b-', alpha=0.7, label='Value')
    axes[0].plot(df.index, df['rolling_mean'], 'g-', label='Moving Average')
    axes[0].fill_between(df.index, df['lower_bound'], df['upper_bound'], 
                         alpha=0.2, color='green', label='Normal Range')
    
    # Mark anomalies
    anomalies = df[df['is_anomaly']]
    axes[0].scatter(anomalies.index, anomalies['value'], 
                   c='red', s=100, marker='x', label='Anomaly', zorder=5)
    
    axes[0].set_title(title)
    axes[0].legend(loc='upper right')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    
    # Plot 2: Z-scores
    axes[1].plot(df.index, df['z_score'], 'b-', alpha=0.7)
    axes[1].axhline(y=3, color='r', linestyle='--', label='Threshold (±3σ)')
    axes[1].axhline(y=-3, color='r', linestyle='--')
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[1].fill_between(df.index, -3, 3, alpha=0.1, color='green')
    
    axes[1].scatter(anomalies.index, anomalies['z_score'], 
                   c='red', s=100, marker='x', zorder=5)
    
    axes[1].set_title('Z-Score Over Time')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Z-Score')
    axes[1].legend()
    
    # Plot 3: Z-score distribution
    axes[2].hist(df['z_score'].dropna(), bins=50, alpha=0.7, density=True)
    
    # Overlay normal distribution
    x = np.linspace(-5, 5, 100)
    axes[2].plot(x, stats.norm.pdf(x), 'r-', label='Standard Normal')
    
    axes[2].axvline(3, color='r', linestyle='--')
    axes[2].axvline(-3, color='r', linestyle='--')
    axes[2].set_title('Z-Score Distribution')
    axes[2].set_xlabel('Z-Score')
    axes[2].set_ylabel('Density')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Anomaly Detection Summary")
    print(f"{'='*50}")
    print(f"Total observations: {len(df)}")
    print(f"Anomalies detected: {df['is_anomaly'].sum()}")
    print(f"Anomaly rate: {100 * df['is_anomaly'].mean():.2f}%")


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate synthetic time series with anomalies
    n_points = 1000
    
    # Base signal: trend + seasonality + noise
    t = np.arange(n_points)
    trend = 0.01 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 24)  # Daily pattern
    noise = np.random.randn(n_points) * 2
    
    values = 100 + trend + seasonal + noise
    
    # Inject anomalies
    anomaly_indices = np.random.choice(n_points, 20, replace=False)
    values[anomaly_indices] += np.random.choice([-1, 1], 20) * np.random.uniform(15, 25, 20)
    
    # Create time series
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='H')
    series = pd.Series(values, index=dates, name='sensor_reading')
    
    # Detect anomalies
    detector = TimeSeriesAnomalyDetector(window_size=48, n_std=3)
    result_df = detector.detect_moving_average_anomalies(series)
    
    # Visualize
    visualize_time_series_anomalies(result_df, "Sensor Reading Anomaly Detection")
    
    # Show detected anomalies
    print("\nDetected Anomalies:")
    anomalies = result_df[result_df['is_anomaly']][['value', 'rolling_mean', 'z_score']]
    print(anomalies.head(10))
```

---

## Question 9: Write a Python function that flags anomalies in a dataset by evaluating cluster compactness after running K-means
- [ ] Done

### Complete Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

class KMeansAnomalyDetector:
    """
    Anomaly detection based on K-means clustering.
    
    Points that are far from their cluster centroid or in sparse
    clusters are flagged as potential anomalies.
    """
    
    def __init__(self, n_clusters=5, contamination=0.1):
        """
        Args:
            n_clusters: Number of clusters for K-means
            contamination: Expected proportion of anomalies
        """
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.kmeans = None
        self.scaler = StandardScaler()
        self.thresholds = {}
    
    def fit(self, X):
        """
        Fit K-means and compute cluster statistics.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        """
        # Scale data
        self.X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels = self.kmeans.fit_predict(self.X_scaled)
        
        # Compute cluster statistics
        self._compute_cluster_stats()
        
        return self
    
    def _compute_cluster_stats(self):
        """Compute statistics for each cluster."""
        self.cluster_stats = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.labels == cluster_id
            cluster_points = self.X_scaled[cluster_mask]
            centroid = self.kmeans.cluster_centers_[cluster_id]
            
            # Distances to centroid
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            
            self.cluster_stats[cluster_id] = {
                'size': cluster_mask.sum(),
                'centroid': centroid,
                'mean_distance': distances.mean(),
                'std_distance': distances.std(),
                'max_distance': distances.max(),
                'compactness': distances.mean()  # Lower = more compact
            }
    
    def compute_anomaly_scores(self, X=None):
        """
        Compute anomaly scores for all points.
        
        Higher score = more anomalous
        
        Returns:
            Dictionary with different anomaly scores
        """
        if X is None:
            X_scaled = self.X_scaled
            labels = self.labels
        else:
            X_scaled = self.scaler.transform(X)
            labels = self.kmeans.predict(X_scaled)
        
        n_samples = len(X_scaled)
        
        # Score 1: Distance to assigned centroid
        centroid_distances = np.zeros(n_samples)
        for i, (point, label) in enumerate(zip(X_scaled, labels)):
            centroid = self.kmeans.cluster_centers_[label]
            centroid_distances[i] = np.linalg.norm(point - centroid)
        
        # Score 2: Relative distance (z-score within cluster)
        relative_distances = np.zeros(n_samples)
        for i, (dist, label) in enumerate(zip(centroid_distances, labels)):
            stats = self.cluster_stats[label]
            z = (dist - stats['mean_distance']) / (stats['std_distance'] + 1e-10)
            relative_distances[i] = z
        
        # Score 3: Cluster size penalty (small clusters are suspicious)
        size_penalty = np.zeros(n_samples)
        total_size = sum(s['size'] for s in self.cluster_stats.values())
        for i, label in enumerate(labels):
            cluster_size = self.cluster_stats[label]['size']
            expected_size = total_size / self.n_clusters
            size_penalty[i] = expected_size / (cluster_size + 1)  # Higher for small clusters
        
        # Score 4: Distance to nearest different centroid
        other_centroid_distances = np.zeros(n_samples)
        for i, (point, label) in enumerate(zip(X_scaled, labels)):
            distances_to_centroids = np.linalg.norm(
                self.kmeans.cluster_centers_ - point, axis=1
            )
            distances_to_centroids[label] = np.inf  # Exclude own centroid
            other_centroid_distances[i] = distances_to_centroids.min()
        
        # Silhouette-like score: (b - a) / max(a, b)
        # a = distance to own centroid, b = distance to nearest other centroid
        silhouette_scores = (other_centroid_distances - centroid_distances) / (
            np.maximum(centroid_distances, other_centroid_distances) + 1e-10
        )
        
        # Combined score
        combined_score = (
            0.4 * (centroid_distances / centroid_distances.max()) +
            0.3 * (relative_distances / (np.abs(relative_distances).max() + 1e-10)) +
            0.2 * (size_penalty / size_penalty.max()) +
            0.1 * (1 - silhouette_scores)  # Lower silhouette = more anomalous
        )
        
        return {
            'centroid_distance': centroid_distances,
            'relative_distance': relative_distances,
            'size_penalty': size_penalty,
            'silhouette': silhouette_scores,
            'combined': combined_score,
            'labels': labels
        }
    
    def predict(self, X=None):
        """
        Predict anomalies.
        
        Returns:
            predictions: Boolean mask (True = anomaly)
        """
        scores = self.compute_anomaly_scores(X)
        combined = scores['combined']
        
        threshold = np.percentile(combined, 100 * (1 - self.contamination))
        predictions = combined > threshold
        
        return predictions, scores
    
    def find_optimal_clusters(self, X, k_range=range(2, 15)):
        """
        Find optimal number of clusters using elbow method and silhouette.
        """
        X_scaled = self.scaler.fit_transform(X)
        
        inertias = []
        silhouettes = []
        
        from sklearn.metrics import silhouette_score
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            if k > 1:
                silhouettes.append(silhouette_score(X_scaled, labels))
            else:
                silhouettes.append(0)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(list(k_range), inertias, 'b-o')
        axes[0].set_xlabel('Number of Clusters')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method')
        
        axes[1].plot(list(k_range), silhouettes, 'g-o')
        axes[1].set_xlabel('Number of Clusters')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Analysis')
        
        plt.tight_layout()
        plt.show()
        
        # Recommend k with highest silhouette
        best_k = list(k_range)[np.argmax(silhouettes)]
        print(f"Recommended k: {best_k} (silhouette: {max(silhouettes):.4f})")
        
        return best_k


def visualize_kmeans_anomalies(X, detector, true_labels=None):
    """Visualize K-means anomaly detection results."""
    predictions, scores = detector.predict()
    labels = scores['labels']
    
    # Reduce to 2D if needed
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(detector.X_scaled)
        centroids_2d = pca.transform(detector.kmeans.cluster_centers_)
    else:
        X_2d = detector.X_scaled
        centroids_2d = detector.kmeans.cluster_centers_
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Clusters with anomalies
    scatter = axes[0, 0].scatter(X_2d[:, 0], X_2d[:, 1], c=labels, 
                                  cmap='tab10', alpha=0.6)
    axes[0, 0].scatter(X_2d[predictions, 0], X_2d[predictions, 1], 
                       facecolors='none', edgecolors='red', s=100, 
                       linewidths=2, label='Anomaly')
    axes[0, 0].scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
                       c='black', marker='X', s=200, label='Centroid')
    axes[0, 0].set_title('Clusters with Anomalies')
    axes[0, 0].legend()
    
    # Plot 2: Anomaly scores
    scatter = axes[0, 1].scatter(X_2d[:, 0], X_2d[:, 1], 
                                  c=scores['combined'], cmap='YlOrRd', alpha=0.7)
    plt.colorbar(scatter, ax=axes[0, 1], label='Anomaly Score')
    axes[0, 1].set_title('Combined Anomaly Scores')
    
    # Plot 3: Distance to centroid
    scatter = axes[1, 0].scatter(X_2d[:, 0], X_2d[:, 1], 
                                  c=scores['centroid_distance'], cmap='YlOrRd', alpha=0.7)
    plt.colorbar(scatter, ax=axes[1, 0], label='Distance')
    axes[1, 0].set_title('Distance to Centroid')
    
    # Plot 4: Cluster statistics
    cluster_ids = list(detector.cluster_stats.keys())
    sizes = [detector.cluster_stats[c]['size'] for c in cluster_ids]
    compactness = [detector.cluster_stats[c]['compactness'] for c in cluster_ids]
    
    x_pos = np.arange(len(cluster_ids))
    width = 0.35
    
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x_pos - width/2, sizes, width, label='Size', color='blue', alpha=0.7)
    bars2 = ax2.bar(x_pos + width/2, compactness, width, label='Compactness', color='green', alpha=0.7)
    
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Cluster Size', color='blue')
    ax2.set_ylabel('Compactness (Mean Distance)', color='green')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(cluster_ids)
    ax1.set_title('Cluster Statistics')
    
    fig.legend([bars1, bars2], ['Size', 'Compactness'], loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nCluster Statistics:")
    for cid, stats in detector.cluster_stats.items():
        print(f"  Cluster {cid}: Size={stats['size']}, "
              f"Compactness={stats['compactness']:.4f}")
    
    print(f"\nAnomaly Summary:")
    print(f"  Total points: {len(predictions)}")
    print(f"  Anomalies detected: {predictions.sum()}")


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate clustered data with outliers
    n_samples = 500
    
    # Three clusters
    cluster1 = np.random.randn(200, 2) * 0.5 + [0, 0]
    cluster2 = np.random.randn(150, 2) * 0.8 + [4, 4]
    cluster3 = np.random.randn(100, 2) * 0.3 + [-3, 3]
    
    # Outliers
    outliers = np.random.uniform(-5, 8, (50, 2))
    
    X = np.vstack([cluster1, cluster2, cluster3, outliers])
    y = np.array([0]*200 + [0]*150 + [0]*100 + [1]*50)
    
    # Find optimal k
    detector = KMeansAnomalyDetector()
    optimal_k = detector.find_optimal_clusters(X)
    
    # Detect anomalies
    detector = KMeansAnomalyDetector(n_clusters=optimal_k, contamination=0.1)
    detector.fit(X)
    
    # Visualize
    visualize_kmeans_anomalies(X, detector)
```

---

