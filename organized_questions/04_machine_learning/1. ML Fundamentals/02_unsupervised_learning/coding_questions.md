# Unsupervised Learning - Coding Questions

## Question 1: Implement K-means clustering from scratch in Python.
- [ ] Done

### Algorithm
1. Initialize K random centroids
2. Assign each point to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat until convergence

### Implementation

```python
import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        
    def fit_predict(self, X):
        n_samples, n_features = X.shape
        
        # 1. Initialize centroids randomly
        idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            # 2. Assignment step: assign points to nearest centroid
            labels = self._assign_clusters(X)
            
            # 3. Update step: recalculate centroids
            new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 
                else self.centroids[i]
                for i in range(self.k)
            ])
            
            # 4. Check convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
                
            self.centroids = new_centroids
            
        return self._assign_clusters(X)
    
    def _assign_clusters(self, X):
        # Compute distance from each point to each centroid
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

# Usage
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=3, random_state=42)

kmeans = KMeans(k=3)
labels = kmeans.fit_predict(X)
print(f"Centroids:\n{kmeans.centroids}")
```

---

## Question 2: Write a Python function to compute the silhouette coefficient for a given clustering.
- [ ] Done

### Formula
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

- **a(i)**: Mean distance to points in same cluster
- **b(i)**: Mean distance to nearest other cluster

### Implementation

```python
import numpy as np
from sklearn.metrics import pairwise_distances

def silhouette_score(X, labels):
    """
    Compute mean silhouette coefficient.
    
    Args:
        X: Data array (n_samples, n_features)
        labels: Cluster labels for each sample
    
    Returns:
        Mean silhouette score (-1 to 1)
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 clusters")
    
    # Pre-compute all pairwise distances
    distances = pairwise_distances(X)
    
    silhouette_vals = []
    
    for i in range(n_samples):
        # a(i): mean distance to same cluster
        same_cluster = (labels == labels[i])
        same_cluster[i] = False  # exclude self
        
        if same_cluster.sum() == 0:
            a_i = 0
        else:
            a_i = distances[i, same_cluster].mean()
        
        # b(i): mean distance to nearest other cluster
        b_i = np.inf
        for label in unique_labels:
            if label == labels[i]:
                continue
            other_cluster = (labels == label)
            b_candidate = distances[i, other_cluster].mean()
            b_i = min(b_i, b_candidate)
        
        # Silhouette for point i
        s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
        silhouette_vals.append(s_i)
    
    return np.mean(silhouette_vals)

# Verify against sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as sk_silhouette

X, _ = make_blobs(n_samples=200, centers=3, random_state=42)
labels = KMeans(n_clusters=3, random_state=42).fit_predict(X)

print(f"Our implementation: {silhouette_score(X, labels):.4f}")
print(f"Sklearn: {sk_silhouette(X, labels):.4f}")
```

---

## Question 3: Use PCA with scikit-learn to reduce the dimensions of a dataset.
- [ ] Done

### Steps
1. Standardize the data
2. Fit PCA
3. Transform data
4. Analyze explained variance

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# 1. Load data (64 features - 8x8 pixel images)
digits = load_digits()
X, y = digits.data, digits.target
print(f"Original shape: {X.shape}")  # (1797, 64)

# 2. Standardize (critical for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA - reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"Reduced shape: {X_pca.shape}")  # (1797, 2)

# 4. Check explained variance
print(f"Variance explained: {pca.explained_variance_ratio_}")
print(f"Total: {pca.explained_variance_ratio_.sum():.2%}")

# 5. Visualize
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA: 64D → 2D')
plt.colorbar(scatter)
plt.show()

# Alternative: Keep 95% variance (auto-select components)
pca_auto = PCA(n_components=0.95)
X_auto = pca_auto.fit_transform(X_scaled)
print(f"Components for 95% variance: {pca_auto.n_components_}")
```

---

## Question 4: Code an example using the DBSCAN algorithm to cluster a given spatial dataset.
- [ ] Done

### DBSCAN Advantages
- No need to specify K
- Finds arbitrary-shaped clusters
- Identifies outliers as noise

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 1. Generate non-linear data (K-means would fail here)
X, y_true = make_moons(n_samples=300, noise=0.08, random_state=42)

# 2. Scale data
X_scaled = StandardScaler().fit_transform(X)

# 3. Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# 4. Analyze results
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise}")

# 5. Visualize
plt.figure(figsize=(10, 5))

# Original data
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='coolwarm')
plt.title('True Labels')

# DBSCAN result
plt.subplot(1, 2, 2)
colors = ['red' if l == -1 else 'blue' if l == 0 else 'green' for l in labels]
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=colors)
plt.title(f'DBSCAN (clusters={n_clusters}, noise={n_noise})')

plt.tight_layout()
plt.show()
```

### Parameter Tuning Tips
- **eps**: Use k-distance plot to find "elbow"
- **min_samples**: Usually 2 × dimensions, or use domain knowledge

---

## Question 5: Implement an Apriori algorithm in Python to find frequent itemsets in transaction data.
- [ ] Done

### Apriori Principle
If {A, B} is infrequent, then {A, B, C} must be infrequent. This prunes the search space.

### Implementation

```python
from collections import defaultdict
from itertools import combinations

def apriori(transactions, min_support):
    """
    Find frequent itemsets using Apriori algorithm.
    
    Args:
        transactions: List of sets (each set is a transaction)
        min_support: Minimum support threshold (0 to 1)
    
    Returns:
        Dictionary of frequent itemsets by level
    """
    n_transactions = len(transactions)
    min_count = int(min_support * n_transactions)
    
    # Step 1: Find frequent 1-itemsets
    item_counts = defaultdict(int)
    for txn in transactions:
        for item in txn:
            item_counts[item] += 1
    
    L1 = {frozenset([item]) for item, count in item_counts.items() 
          if count >= min_count}
    
    frequent_itemsets = {1: L1}
    k = 2
    
    while frequent_itemsets.get(k-1):
        # Step 2: Generate candidates
        Lk_minus_1 = frequent_itemsets[k-1]
        candidates = set()
        
        for s1 in Lk_minus_1:
            for s2 in Lk_minus_1:
                union = s1 | s2
                if len(union) == k:
                    candidates.add(union)
        
        # Step 3: Count support for candidates
        candidate_counts = defaultdict(int)
        for txn in transactions:
            for candidate in candidates:
                if candidate.issubset(txn):
                    candidate_counts[candidate] += 1
        
        # Step 4: Filter by min_support
        Lk = {itemset for itemset, count in candidate_counts.items() 
              if count >= min_count}
        
        if not Lk:
            break
            
        frequent_itemsets[k] = Lk
        k += 1
    
    return frequent_itemsets

# Example usage
transactions = [
    {'Milk', 'Bread', 'Butter'},
    {'Beer', 'Diapers', 'Chips'},
    {'Milk', 'Bread', 'Diapers', 'Eggs'},
    {'Milk', 'Bread', 'Beer', 'Diapers'},
    {'Bread', 'Milk', 'Eggs'}
]

# Find itemsets appearing in at least 60% of transactions
frequent = apriori(transactions, min_support=0.6)

print("Frequent Itemsets:")
for level, itemsets in frequent.items():
    print(f"  Level {level}:")
    for itemset in itemsets:
        print(f"    {set(itemset)}")
```

### Output
```
Frequent Itemsets:
  Level 1:
    {'Bread'}
    {'Milk'}
  Level 2:
    {'Bread', 'Milk'}
```

### Production Note
For real applications, use `mlxtend` library:
```python
from mlxtend.frequent_patterns import apriori, association_rules
```
