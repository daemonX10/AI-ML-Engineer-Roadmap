# Curse Of Dimensionality Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Implement PCA in Python from scratch and apply it to a high-dimensional dataset.**

**Answer:**

```python
import numpy as np

def pca_from_scratch(X, n_components):
    """
    PCA implementation from scratch
    1. Center data
    2. Compute covariance
    3. Get eigenvectors
    4. Project
    """
    # Step 1: Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Step 2: Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Step 4: Sort by eigenvalue (descending)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    eigenvalues = eigenvalues[sorted_idx]
    
    # Step 5: Select top k components
    components = eigenvectors[:, :n_components]
    
    # Step 6: Project data
    X_reduced = X_centered @ components
    
    # Explained variance ratio
    explained_var = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return X_reduced, explained_var

# Example usage
np.random.seed(42)
X = np.random.randn(100, 50)  # 100 samples, 50 features

X_pca, var_ratio = pca_from_scratch(X, n_components=5)
print(f"Reduced shape: {X_pca.shape}")  # (100, 5)
print(f"Explained variance: {var_ratio}")
```

---

## Question 2
- [ ] Done

**Write a Python function that selects the top k features based on mutual information with the target variable.**

**Answer:**

```python
from sklearn.feature_selection import mutual_info_classif
import numpy as np

def select_top_k_features_mi(X, y, k=10):
    """
    Select top k features using mutual information
    Higher MI = more informative feature
    """
    # Compute mutual information for each feature
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Get indices of top k features
    top_k_idx = np.argsort(mi_scores)[::-1][:k]
    
    # Select features
    X_selected = X[:, top_k_idx]
    
    # Return selected features and their scores
    return X_selected, top_k_idx, mi_scores[top_k_idx]

# Example usage
from sklearn.datasets import make_classification

# Create dataset with informative and noise features
X, y = make_classification(n_samples=500, n_features=50, 
                           n_informative=10, n_redundant=5,
                           random_state=42)

X_selected, indices, scores = select_top_k_features_mi(X, y, k=10)
print(f"Original shape: {X.shape}")      # (500, 50)
print(f"Selected shape: {X_selected.shape}")  # (500, 10)
print(f"Top feature indices: {indices}")
print(f"MI scores: {scores.round(3)}")
```

**Key Point:** Mutual information captures non-linear relationships, unlike correlation.

---

## Question 3
- [ ] Done

**Code a Python script that performs recursive feature elimination to reduce the dimensionality of the dataset.**

**Answer:**

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Create high-dimensional dataset
X, y = make_classification(n_samples=200, n_features=30, 
                           n_informative=5, random_state=42)

# Base estimator (must have coef_ or feature_importances_)
estimator = LogisticRegression(max_iter=1000)

# RFE: Recursively remove weakest features
rfe = RFE(estimator=estimator, 
          n_features_to_select=5,  # Keep top 5
          step=1)                   # Remove 1 feature at a time

# Fit RFE
rfe.fit(X, y)

# Results
print(f"Selected features: {rfe.support_}")  # Boolean mask
print(f"Feature ranking: {rfe.ranking_}")    # 1 = selected
print(f"Number selected: {rfe.n_features_}")

# Transform data
X_reduced = rfe.transform(X)
print(f"Original: {X.shape} → Reduced: {X_reduced.shape}")

# Get selected feature indices
selected_idx = [i for i, s in enumerate(rfe.support_) if s]
print(f"Selected indices: {selected_idx}")
```

**How RFE Works:**
1. Train model on all features
2. Remove feature with smallest coefficient
3. Repeat until k features remain

**Tip:** Use RFECV for automatic selection of optimal k.

---

## Question 4
- [ ] Done

**Create a visualization of the nearest neighbors of a point in a high-dimensional space after applying t-SNE.**

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_digits

# Load high-dimensional data (64 features)
digits = load_digits()
X, y = digits.data, digits.target

# Find nearest neighbors in original high-D space
k = 5
nn = NearestNeighbors(n_neighbors=k+1)  # +1 for the point itself
nn.fit(X)

# Pick a query point
query_idx = 0
distances, neighbor_idx = nn.kneighbors([X[query_idx]])
neighbor_idx = neighbor_idx[0][1:]  # Exclude self

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = tsne.fit_transform(X)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.5, s=20)
plt.scatter(X_2d[query_idx, 0], X_2d[query_idx, 1], 
            c='red', s=200, marker='*', label='Query point')
plt.scatter(X_2d[neighbor_idx, 0], X_2d[neighbor_idx, 1], 
            c='black', s=100, marker='x', label=f'Top {k} neighbors (high-D)')
plt.legend()
plt.title('t-SNE with High-D Nearest Neighbors Highlighted')
plt.show()
```

**Insight:** High-D neighbors may not be close in t-SNE plot (distances don't transfer).

---

## Question 5
- [ ] Done

**Demonstrate the use of L1 regularization in a logistic regression model on a high-dimensional dataset using scikit-learn.**

**Answer:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# Create high-dimensional dataset (many noisy features)
X, y = make_classification(n_samples=500, n_features=100, 
                           n_informative=10, n_redundant=10,
                           random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# L1 regularization (Lasso) - creates sparse weights
model_l1 = LogisticRegression(penalty='l1', solver='saga', 
                               C=0.1, max_iter=1000)
model_l1.fit(X_train, y_train)

# Count non-zero coefficients
n_nonzero = np.sum(model_l1.coef_ != 0)
print(f"Non-zero coefficients: {n_nonzero} / 100")

# Compare with L2 (no sparsity)
model_l2 = LogisticRegression(penalty='l2', C=0.1, max_iter=1000)
model_l2.fit(X_train, y_train)

print(f"\nL1 Test Accuracy: {model_l1.score(X_test, y_test):.3f}")
print(f"L2 Test Accuracy: {model_l2.score(X_test, y_test):.3f}")

# Show which features L1 selected
selected_features = np.where(model_l1.coef_[0] != 0)[0]
print(f"\nL1 selected features: {selected_features}")
```

**Key Point:** L1 automatically selects ~10-20 features, ignoring noise. C controls regularization strength (smaller = more sparsity).

---

