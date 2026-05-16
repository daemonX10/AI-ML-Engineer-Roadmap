# Cluster Analysis Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Implement the K-means clustering algorithm from scratch in Python.**

### Answer

**Algorithm Steps:**
1. Initialize k random centroids from data points
2. Assign each point to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat until convergence

**Code:**
```python
import numpy as np

def kmeans_scratch(X, k, max_iters=100):
    """K-means from scratch"""
    n_samples = X.shape[0]
    
    # Step 1: Random initialization
    random_idx = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_idx].copy()
    
    for _ in range(max_iters):
        # Step 2: Assignment - find nearest centroid for each point
        distances = np.zeros((n_samples, k))
        for i in range(k):
            distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)
        labels = np.argmin(distances, axis=1)
        
        # Step 3: Update - recalculate centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            if np.sum(labels == i) > 0:
                new_centroids[i] = X[labels == i].mean(axis=0)
            else:
                new_centroids[i] = centroids[i]
        
        # Step 4: Check convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return labels, centroids

# Usage
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
labels, centroids = kmeans_scratch(X, k=4)
```

---

## Question 2
- [ ] Done

**Write a Python script that uses hierarchical clustering to group data and visualizes the resulting dendrogram.**

### Answer

**Pipeline:**
1. Load data
2. Compute linkage matrix
3. Plot dendrogram
4. Cut at desired height for clusters

**Code:**
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.datasets import make_blobs

# Step 1: Generate data
X, _ = make_blobs(n_samples=50, centers=4, random_state=42)

# Step 2: Compute linkage matrix
# Methods: 'ward', 'complete', 'average', 'single'
Z = linkage(X, method='ward')

# Step 3: Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(Z, leaf_rotation=90, leaf_font_size=8)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.axhline(y=15, color='r', linestyle='--', label='Cut threshold')
plt.legend()
plt.tight_layout()
plt.show()

# Step 4: Get cluster labels by cutting dendrogram
labels = fcluster(Z, t=15, criterion='distance')
# Or by number of clusters
labels = fcluster(Z, t=4, criterion='maxclust')
```

---

## Question 3
- [ ] Done

**Use scikit-learn to perform DBSCAN clustering on a given dataset and plot the clusters.**

### Answer

**Pipeline:**
1. Load and scale data
2. Apply DBSCAN
3. Visualize clusters and noise

**Code:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

# Step 1: Generate and scale data
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# Step 2: Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Step 3: Visualize
plt.figure(figsize=(10, 6))

# Plot clusters
unique_labels = set(labels)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    if label == -1:
        color = 'black'  # Noise points
        marker = 'x'
        label_name = 'Noise'
    else:
        marker = 'o'
        label_name = f'Cluster {label}'
    
    mask = labels == label
    plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1], 
                c=[color], marker=marker, s=50, label=label_name)

plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

print(f"Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"Noise points: {sum(labels == -1)}")
```

---

## Question 4
- [ ] Done

**Create a Python function to calculate silhouette scores for different numbers of clusters in a dataset.**

### Answer

**Pipeline:**
1. Loop through k values (2 to max_k)
2. Fit K-Means for each k
3. Calculate silhouette score
4. Plot and return optimal k

**Code:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

def find_optimal_k_silhouette(X, max_k=10):
    """Find optimal k using silhouette scores"""
    k_values = range(2, max_k + 1)
    scores = []
    
    for k in k_values:
        # Fit K-Means
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # Calculate silhouette score
        score = silhouette_score(X, labels)
        scores.append(score)
        print(f"k={k}: Silhouette Score = {score:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for Optimal k')
    plt.grid(True)
    plt.show()
    
    # Find optimal k
    optimal_k = k_values[np.argmax(scores)]
    print(f"\nOptimal k = {optimal_k}")
    return optimal_k, scores

# Usage
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
optimal_k, scores = find_optimal_k_silhouette(X, max_k=8)
```

---

## Question 5
- [ ] Done

**Implement a Gaussian Mixture Model clustering with scikit-learn and visualize the results.**

### Answer

**Pipeline:**
1. Fit GMM to data
2. Get cluster assignments
3. Plot data with cluster colors
4. Visualize Gaussian ellipses

**Code:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse

def plot_gmm(gmm, X, labels):
    """Plot GMM clusters with ellipses"""
    plt.figure(figsize=(10, 8))
    
    # Plot points
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=40, alpha=0.7)
    
    # Plot ellipses for each component
    for i, (mean, cov) in enumerate(zip(gmm.means_, gmm.covariances_)):
        # Eigenvalue decomposition for ellipse
        v, w = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(w[1, 0], w[0, 0]))
        
        # Plot 1, 2, 3 sigma ellipses
        for n_std in [1, 2, 3]:
            width, height = 2 * n_std * np.sqrt(v)
            ellipse = Ellipse(mean, width, height, angle=angle,
                            fill=False, edgecolor='red', linewidth=2, alpha=0.5)
            plt.gca().add_patch(ellipse)
        
        # Plot centroid
        plt.scatter(*mean, c='red', s=100, marker='x', linewidth=3)
    
    plt.title('Gaussian Mixture Model Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Generate data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

# Fit GMM
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
labels = gmm.fit_predict(X)

# Visualize
plot_gmm(gmm, X, labels)

# Soft probabilities
probs = gmm.predict_proba(X)
print(f"Sample probabilities:\n{probs[:3]}")
```

---

## Question 6
- [ ] Done

**Develop a Python script to run and compare multiple clustering algorithms on the same dataset.**

### Answer

**Pipeline:**
1. Create datasets (blobs, moons)
2. Define algorithms
3. Run each algorithm on each dataset
4. Plot comparison grid

**Code:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_moons

# Step 1: Create datasets
datasets = {
    'Blobs': make_blobs(n_samples=200, centers=4, random_state=42),
    'Moons': make_moons(n_samples=200, noise=0.05, random_state=42)
}

# Step 2: Define algorithms
def get_algorithms(n_clusters):
    return {
        'K-Means': KMeans(n_clusters=n_clusters, n_init=10, random_state=42),
        'GMM': GaussianMixture(n_components=n_clusters, random_state=42),
        'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
        'DBSCAN': DBSCAN(eps=0.3, min_samples=5)
    }

# Step 3: Plot comparison
fig, axes = plt.subplots(len(datasets), 4, figsize=(16, 8))

for i, (data_name, (X, _)) in enumerate(datasets.items()):
    X_scaled = StandardScaler().fit_transform(X)
    n_clusters = 4 if data_name == 'Blobs' else 2
    algorithms = get_algorithms(n_clusters)
    
    for j, (algo_name, algo) in enumerate(algorithms.items()):
        ax = axes[i, j]
        
        # Fit and predict
        if hasattr(algo, 'fit_predict'):
            labels = algo.fit_predict(X_scaled)
        else:
            labels = algo.fit(X_scaled).predict(X_scaled)
        
        # Plot
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=20)
        ax.set_title(f'{algo_name}\non {data_name}')
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()
```

---

## Question 7
- [ ] Done

**Write a Python function to normalize and scale data before clustering.**

### Answer

**Pipeline:**
1. Choose scaling method
2. Fit scaler to data
3. Transform data
4. Return scaled data

**Code:**
```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def scale_for_clustering(X, method='standard'):
    """
    Scale data for clustering algorithms.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
    method : str, one of 'standard', 'minmax', 'robust'
    
    Returns:
    --------
    X_scaled : scaled data
    scaler : fitted scaler object
    """
    scalers = {
        'standard': StandardScaler(),      # Mean=0, Std=1
        'minmax': MinMaxScaler(),          # Range [0, 1]
        'robust': RobustScaler()           # Robust to outliers
    }
    
    if method not in scalers:
        raise ValueError(f"Method must be one of {list(scalers.keys())}")
    
    scaler = scalers[method]
    X_scaled = scaler.fit_transform(X)
    
    print(f"Scaling method: {method}")
    print(f"Original range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"Scaled range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
    
    return X_scaled, scaler

# Usage example
sample_data = np.array([
    [10, 1000],
    [12, 1200],
    [8, 900],
    [200, 10],
    [210, 15]
], dtype=float)

print("Original:\n", sample_data)
X_scaled, scaler = scale_for_clustering(sample_data, method='standard')
print("\nScaled:\n", X_scaled)
```

---

## Question 8
- [ ] Done

**Implement a custom distance metric and use it in a clustering algorithm within scikit-learn.**

### Answer

**Pipeline:**
1. Define custom distance function
2. Pass to algorithm via metric parameter
3. Fit and predict

**Code:**
```python
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Custom distance: Weighted Euclidean
def weighted_euclidean(x, y, weights=None):
    """Weighted Euclidean distance"""
    if weights is None:
        weights = np.ones(len(x))
    return np.sqrt(np.sum(weights * (x - y)**2))

# Create wrapper for sklearn (needs only x, y)
def create_weighted_metric(weights):
    """Create weighted metric function"""
    def metric(x, y):
        return np.sqrt(np.sum(weights * (x - y)**2))
    return metric

# Generate data
X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

# Feature weights: first feature twice as important
weights = np.array([2.0, 1.0])
custom_metric = create_weighted_metric(weights)

# Method 1: DBSCAN with custom metric
dbscan = DBSCAN(eps=2.0, min_samples=5, metric=custom_metric)
labels_custom = dbscan.fit_predict(X)

# Method 2: Using precomputed distance matrix
distance_matrix = pairwise_distances(X, metric=custom_metric)
dbscan_precomputed = DBSCAN(eps=2.0, min_samples=5, metric='precomputed')
labels_precomputed = dbscan_precomputed.fit_predict(distance_matrix)

# Compare with standard Euclidean
dbscan_standard = DBSCAN(eps=1.5, min_samples=5)
labels_standard = dbscan_standard.fit_predict(X)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_standard, cmap='viridis')
axes[0].set_title('Standard Euclidean')
axes[1].scatter(X[:, 0], X[:, 1], c=labels_custom, cmap='viridis')
axes[1].set_title('Weighted Euclidean (2:1)')
plt.show()
```

---
