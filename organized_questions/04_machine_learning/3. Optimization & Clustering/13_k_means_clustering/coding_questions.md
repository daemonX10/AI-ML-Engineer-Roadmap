# K Means Clustering Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Implement a basic K-Means Clustering algorithm from scratch using Python.**

**Answer:**

```python
import numpy as np

def kmeans_scratch(X, k, max_iters=100):
    """
    K-Means from scratch
    
    Pipeline:
    1. Initialize centroids randomly
    2. Assign points to nearest centroid
    3. Update centroids as mean of assigned points
    4. Repeat until convergence
    
    Output: labels, centroids
    """
    n_samples, n_features = X.shape
    
    # Step 1: Random initialization
    random_idx = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_idx]
    
    for iteration in range(max_iters):
        # Step 2: Assign points to nearest centroid
        distances = np.zeros((n_samples, k))
        for i in range(k):
            distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)
        labels = np.argmin(distances, axis=1)
        
        # Step 3: Update centroids
        new_centroids = np.zeros((k, n_features))
        for i in range(k):
            points_in_cluster = X[labels == i]
            if len(points_in_cluster) > 0:
                new_centroids[i] = points_in_cluster.mean(axis=0)
            else:
                new_centroids[i] = centroids[i]  # Keep old if empty
        
        # Step 4: Check convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return labels, centroids

# Usage
np.random.seed(42)
X = np.random.randn(100, 2)
labels, centroids = kmeans_scratch(X, k=3)
print(f"Cluster labels: {np.unique(labels)}")
print(f"Centroids shape: {centroids.shape}")
```

---

## Question 2
- [ ] Done

**Write a function in Python that determines the best value of k (number of clusters) using the Elbow Method.**

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_optimal_k_elbow(X, k_range=range(1, 11)):
    """
    Elbow Method to find optimal K
    
    Pipeline:
    1. Run K-Means for each k value
    2. Record inertia (WCSS) for each k
    3. Plot k vs inertia
    4. Visual: find elbow point
    
    Output: Plot showing elbow curve
    """
    inertias = []
    
    # Step 1-2: Run K-Means and record inertia
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Step 3: Plot
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (WCSS)')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True)
    plt.show()
    
    return list(k_range), inertias

# Usage
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
k_values, inertias = find_optimal_k_elbow(X)

# Print inertias for reference
for k, inertia in zip(k_values, inertias):
    print(f"K={k}: Inertia={inertia:.2f}")
```

---

## Question 3
- [ ] Done

**Given a dataset, apply feature scaling and run K-Means Clustering using scikit-learn.**

**Answer:**

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

def kmeans_with_scaling(X, n_clusters=3):
    """
    K-Means with proper scaling
    
    Pipeline:
    1. Scale features (StandardScaler)
    2. Apply K-Means
    3. Return labels and model
    
    Output: labels, pipeline (for new predictions)
    """
    # Method 1: Using Pipeline (Recommended)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
    ])
    
    labels = pipeline.fit_predict(X)
    
    return labels, pipeline

# Usage with sample data
np.random.seed(42)
X = np.array([
    [25, 50000],   # age, salary
    [30, 60000],
    [35, 55000],
    [50, 100000],
    [55, 120000],
    [60, 110000]
])

labels, pipeline = kmeans_with_scaling(X, n_clusters=2)
print(f"Cluster labels: {labels}")

# Access components
scaler = pipeline.named_steps['scaler']
kmeans = pipeline.named_steps['kmeans']
print(f"Inertia: {kmeans.inertia_:.2f}")

# Predict new data (automatically scaled)
new_data = np.array([[40, 75000]])
new_label = pipeline.predict(new_data)
print(f"New point cluster: {new_label[0]}")
```

---

## Question 4
- [ ] Done

**Create a Python script to visualize the results of K-Means Clustering on a 2D dataset.**

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def visualize_kmeans(X, n_clusters=3):
    """
    Visualize K-Means clustering
    
    Pipeline:
    1. Fit K-Means
    2. Get labels and centroids
    3. Plot points colored by cluster
    4. Plot centroids with markers
    
    Output: Scatter plot with clusters and centroids
    """
    # Step 1: Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    # Step 2-4: Visualize
    plt.figure(figsize=(10, 6))
    
    # Plot points colored by cluster
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, 
                         cmap='viridis', alpha=0.6, s=50)
    
    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], 
               c='red', marker='X', s=200, edgecolors='black',
               linewidths=2, label='Centroids')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'K-Means Clustering (K={n_clusters})')
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return labels, centroids

# Usage
X, true_labels = make_blobs(n_samples=300, centers=4, 
                            cluster_std=0.6, random_state=42)
labels, centroids = visualize_kmeans(X, n_clusters=4)
print(f"Centroid locations:\n{centroids}")
```

---

## Question 5
- [ ] Done

**Script a program to compare the performance of different initialization methods for centroids.**

**Answer:**

```python
import numpy as np
import time
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def compare_init_methods(X, n_clusters=4, n_runs=10):
    """
    Compare K-Means initialization methods
    
    Pipeline:
    1. Run K-Means with random init
    2. Run K-Means with k-means++ init
    3. Compare: inertia, iterations, time
    
    Output: Comparison table
    """
    results = {'random': [], 'k-means++': []}
    
    for init_method in ['random', 'k-means++']:
        inertias = []
        iterations = []
        times = []
        
        for seed in range(n_runs):
            start = time.time()
            
            kmeans = KMeans(
                n_clusters=n_clusters,
                init=init_method,
                n_init=1,  # Single run per seed
                random_state=seed
            )
            kmeans.fit(X)
            
            elapsed = time.time() - start
            
            inertias.append(kmeans.inertia_)
            iterations.append(kmeans.n_iter_)
            times.append(elapsed)
        
        results[init_method] = {
            'avg_inertia': np.mean(inertias),
            'std_inertia': np.std(inertias),
            'avg_iterations': np.mean(iterations),
            'avg_time': np.mean(times) * 1000  # ms
        }
    
    # Print comparison
    print("Initialization Method Comparison")
    print("=" * 50)
    for method, stats in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Avg Inertia: {stats['avg_inertia']:.2f} ± {stats['std_inertia']:.2f}")
        print(f"  Avg Iterations: {stats['avg_iterations']:.1f}")
        print(f"  Avg Time: {stats['avg_time']:.2f} ms")
    
    return results

# Usage
X, _ = make_blobs(n_samples=1000, centers=5, random_state=42)
results = compare_init_methods(X, n_clusters=5)
```

---

## Question 6
- [ ] Done

**Write code to compute the silhouette coefficient for evaluating the clustering quality.**

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

def evaluate_with_silhouette(X, k_range=range(2, 11)):
    """
    Evaluate clustering using Silhouette Score
    
    Pipeline:
    1. Run K-Means for each k
    2. Calculate silhouette score
    3. Find optimal k (max silhouette)
    4. Plot scores
    
    Output: Optimal k, silhouette scores
    """
    silhouette_scores = []
    
    # Step 1-2: Run K-Means and calculate silhouette
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        print(f"K={k}: Silhouette Score = {score:.4f}")
    
    # Step 3: Find optimal k
    optimal_k = list(k_range)[np.argmax(silhouette_scores)]
    print(f"\nOptimal K = {optimal_k}")
    
    # Step 4: Plot
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, silhouette_scores, 'bo-', linewidth=2)
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs K')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return optimal_k, silhouette_scores

# Usage
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
optimal_k, scores = evaluate_with_silhouette(X)

# Detailed silhouette analysis for optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(X)
sample_scores = silhouette_samples(X, labels)
print(f"\nMin silhouette: {sample_scores.min():.4f}")
print(f"Max silhouette: {sample_scores.max():.4f}")
```

---

## Question 7
- [ ] Done

**Implement a mini-batch K-Means clustering using Python.**

**Answer:**

```python
import numpy as np
import time
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans

def compare_kmeans_minibatch(X, n_clusters=5, batch_size=100):
    """
    Compare standard K-Means vs Mini-Batch K-Means
    
    Pipeline:
    1. Run standard K-Means (for comparison)
    2. Run Mini-Batch K-Means
    3. Compare time and quality
    
    Output: Comparison of both methods
    """
    results = {}
    
    # Standard K-Means
    start = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_km = kmeans.fit_predict(X)
    time_km = time.time() - start
    results['KMeans'] = {
        'time': time_km,
        'inertia': kmeans.inertia_,
        'labels': labels_km
    }
    
    # Mini-Batch K-Means
    start = time.time()
    mbkmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=42,
        n_init=10
    )
    labels_mb = mbkmeans.fit_predict(X)
    time_mb = time.time() - start
    results['MiniBatchKMeans'] = {
        'time': time_mb,
        'inertia': mbkmeans.inertia_,
        'labels': labels_mb
    }
    
    # Print comparison
    print("K-Means vs Mini-Batch K-Means")
    print("=" * 40)
    print(f"{'Method':<20} {'Time (s)':<12} {'Inertia'}")
    print("-" * 40)
    for method, stats in results.items():
        print(f"{method:<20} {stats['time']:.4f}       {stats['inertia']:.2f}")
    
    speedup = results['KMeans']['time'] / results['MiniBatchKMeans']['time']
    print(f"\nSpeedup: {speedup:.2f}x")
    
    return results

# Usage with large dataset
X, _ = make_blobs(n_samples=50000, centers=10, random_state=42)
results = compare_kmeans_minibatch(X, n_clusters=10, batch_size=1000)
```

---

## Question 8
- [ ] Done

**Write a Python function to identify the centroid of a new data point in an existing K-Means model.**

**Answer:**

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def predict_cluster_for_new_point(model, scaler, new_point):
    """
    Predict cluster for new data point
    
    Pipeline:
    1. Scale new point (using fitted scaler)
    2. Predict cluster using model
    3. Get distance to assigned centroid
    4. Return cluster label and centroid
    
    Output: cluster label, centroid, distance
    """
    # Ensure 2D array
    new_point = np.array(new_point).reshape(1, -1)
    
    # Step 1: Scale (if scaler provided)
    if scaler is not None:
        new_point_scaled = scaler.transform(new_point)
    else:
        new_point_scaled = new_point
    
    # Step 2: Predict cluster
    cluster_label = model.predict(new_point_scaled)[0]
    
    # Step 3: Get centroid and distance
    centroid = model.cluster_centers_[cluster_label]
    distance = np.linalg.norm(new_point_scaled - centroid)
    
    return cluster_label, centroid, distance

# Train model
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Predict for new point
new_point = [1.5, 2.0]
cluster, centroid, dist = predict_cluster_for_new_point(kmeans, scaler, new_point)

print(f"New point: {new_point}")
print(f"Assigned cluster: {cluster}")
print(f"Centroid: {centroid}")
print(f"Distance to centroid: {dist:.4f}")

# Batch prediction
new_points = [[1.5, 2.0], [5.0, 5.0], [-2.0, -3.0]]
for point in new_points:
    cluster, _, dist = predict_cluster_for_new_point(kmeans, scaler, point)
    print(f"Point {point} -> Cluster {cluster} (dist={dist:.2f})")
```

---

## Question 9
- [ ] Done

**Using Pandas and Python, clean and prepare a real-world dataset for K-Means Clustering.**

**Answer:**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

def prepare_data_for_kmeans(df, numeric_cols, categorical_cols=None):
    """
    Clean and prepare data for K-Means
    
    Pipeline:
    1. Handle missing values
    2. Remove outliers
    3. Encode categoricals (optional)
    4. Scale features
    
    Output: Cleaned and scaled data ready for K-Means
    """
    df_clean = df.copy()
    
    # Step 1: Handle missing values
    print(f"Missing values before: {df_clean[numeric_cols].isnull().sum().sum()}")
    imputer = SimpleImputer(strategy='median')
    df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    print(f"Missing values after: {df_clean[numeric_cols].isnull().sum().sum()}")
    
    # Step 2: Remove outliers (IQR method)
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    
    print(f"Rows after outlier removal: {len(df_clean)}")
    
    # Step 3: Encode categoricals (one-hot)
    if categorical_cols:
        df_clean = pd.get_dummies(df_clean, columns=categorical_cols)
    
    # Step 4: Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[numeric_cols])
    
    return X_scaled, df_clean, scaler

# Example usage with synthetic data
np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.randint(18, 70, 500),
    'income': np.random.randint(20000, 150000, 500),
    'spending_score': np.random.randint(1, 100, 500),
    'category': np.random.choice(['A', 'B', 'C'], 500)
})

# Add some missing values
df.loc[np.random.choice(500, 20), 'income'] = np.nan

# Prepare data
numeric_cols = ['age', 'income', 'spending_score']
X_scaled, df_clean, scaler = prepare_data_for_kmeans(df, numeric_cols)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
print("\nCluster Profiles:")
print(df_clean.groupby('cluster')[numeric_cols].mean())
```

---

## Question 10
- [ ] Done

**Create a multi-dimensional K-Means clustering example and visualize it using PCA for dimensionality reduction.**

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def kmeans_with_pca_visualization(X, n_clusters=4):
    """
    K-Means on high-D data with PCA visualization
    
    Pipeline:
    1. Scale features
    2. Apply K-Means on full data
    3. Reduce to 2D using PCA
    4. Visualize clusters in 2D space
    
    Output: Visualization and cluster analysis
    """
    # Step 1: Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 2: K-Means on full data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Step 3: PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    
    print(f"Original dimensions: {X.shape[1]}")
    print(f"Variance explained by 2 PCs: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Step 4: Visualize
    plt.figure(figsize=(10, 6))
    
    # Plot points
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                         cmap='viridis', alpha=0.6, s=50)
    
    # Plot centroids
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               c='red', marker='X', s=200, edgecolors='black',
               linewidths=2, label='Centroids')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title(f'K-Means Clustering (K={n_clusters}) - PCA Projection')
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return labels, kmeans, pca

# Create high-dimensional data
X, true_labels = make_blobs(
    n_samples=500, 
    n_features=10,  # 10 dimensions
    centers=5, 
    random_state=42
)

print(f"Data shape: {X.shape}")
labels, kmeans, pca = kmeans_with_pca_visualization(X, n_clusters=5)

# Cluster distribution
print("\nCluster distribution:")
unique, counts = np.unique(labels, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count} points")
```

---
