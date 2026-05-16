# Pca Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Write a Python function to perform PCA from scratch using NumPy.**

### Answer

**Pipeline:**
1. Center the data (subtract mean)
2. Compute covariance matrix
3. Compute eigenvalues and eigenvectors
4. Sort by eigenvalue descending
5. Select top k eigenvectors
6. Project data

```python
import numpy as np

def pca_from_scratch(X, n_components):
    """
    X: Input data (n_samples, n_features)
    n_components: Number of components to keep
    Returns: Transformed data, components, explained variance ratio
    """
    # Step 1: Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    # Step 2: Compute covariance matrix
    n_samples = X.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
    
    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Step 4: Sort by eigenvalue (descending)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    # Step 5: Select top k eigenvectors
    components = eigenvectors[:, :n_components]
    
    # Step 6: Project data
    X_transformed = X_centered @ components
    
    # Explained variance ratio
    explained_var_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return X_transformed, components, explained_var_ratio

# Usage
X = np.random.randn(100, 5)
X_pca, components, var_ratio = pca_from_scratch(X, n_components=2)
print(f"Shape: {X_pca.shape}")
print(f"Variance explained: {var_ratio}")
```

---

## Question 2
- [ ] Done

**Use scikit-learn to apply PCA on a high-dimensional dataset and interpret the results.**

### Answer

**Pipeline:**
1. Load data
2. Standardize features
3. Apply PCA
4. Analyze explained variance
5. Visualize results

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load high-dimensional data (64 features)
digits = load_digits()
X, y = digits.data, digits.target
print(f"Original shape: {X.shape}")  # (1797, 64)

# Step 2: Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_scaled)
print(f"Reduced shape: {X_pca.shape}")

# Step 4: Analyze results
print(f"Components needed for 95%: {pca.n_components_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# Step 5: Visualize
plt.figure(figsize=(10, 4))

# Scree plot
plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca.explained_variance_ratio_)+1), 
        pca.explained_variance_ratio_)
plt.xlabel('Component')
plt.ylabel('Variance Ratio')
plt.title('Scree Plot')

# 2D visualization
plt.subplot(1, 2, 2)
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=5)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
plt.title('PCA 2D Projection')
plt.colorbar()

plt.tight_layout()
plt.show()
```

**Interpretation:**
- Number of components for 95% variance shows intrinsic dimensionality
- Scree plot shows importance of each component
- 2D plot shows class separation

---

## Question 3
- [ ] Done

**Code a Python script to visualize the eigenfaces from a given set of facial images dataset using PCA.**

### Answer

**Pipeline:**
1. Load face dataset
2. Compute mean face
3. Apply PCA
4. Reshape components to image dimensions
5. Display eigenfaces

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

# Step 1: Load face data
faces = fetch_olivetti_faces()
X = faces.data  # (400, 4096) - 400 images, 64x64 pixels
image_shape = (64, 64)
print(f"Data shape: {X.shape}")

# Step 2: Compute mean face
mean_face = X.mean(axis=0)

# Step 3: Apply PCA
n_components = 16
pca = PCA(n_components=n_components)
pca.fit(X)

# Step 4: Get eigenfaces (components reshaped to images)
eigenfaces = pca.components_.reshape((n_components, *image_shape))

# Step 5: Visualize
fig, axes = plt.subplots(3, 6, figsize=(12, 6))

# Mean face
axes[0, 0].imshow(mean_face.reshape(image_shape), cmap='gray')
axes[0, 0].set_title('Mean Face')
axes[0, 0].axis('off')

# Eigenfaces
for i, (eigenface, ax) in enumerate(zip(eigenfaces, axes.flat[1:17])):
    ax.imshow(eigenface, cmap='gray')
    ax.set_title(f'EF {i+1}\n({pca.explained_variance_ratio_[i]:.1%})')
    ax.axis('off')

# Hide remaining axes
for ax in axes.flat[17:]:
    ax.axis('off')

plt.suptitle('Eigenfaces from PCA')
plt.tight_layout()
plt.show()

print(f"Variance explained by {n_components} components: "
      f"{sum(pca.explained_variance_ratio_):.1%}")
```

---

## Question 4
- [ ] Done

**Implement PCA for feature extraction before applying a machine learning model.**

### Answer

**Pipeline:**
1. Split data
2. Standardize (fit on train)
3. Apply PCA (fit on train)
4. Train model on PCA features
5. Evaluate

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load and split data
data = load_breast_cancer()
X, y = data.data, data.target
print(f"Original features: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 2: Standardize (fit on train only!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Apply PCA (fit on train only!)
pca = PCA(n_components=0.95)  # Keep 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"Reduced features: {X_train_pca.shape[1]}")

# Step 4: Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with PCA: {accuracy:.4f}")

# Compare without PCA
model_no_pca = LogisticRegression(max_iter=1000)
model_no_pca.fit(X_train_scaled, y_train)
y_pred_no_pca = model_no_pca.predict(X_test_scaled)
print(f"Accuracy without PCA: {accuracy_score(y_test, y_pred_no_pca):.4f}")
```

**Key Points:**
- Always fit scaler and PCA on training data only
- Use transform (not fit_transform) on test data
- Compare with/without PCA to validate benefit

---

## Question 5
- [ ] Done

**Implement PCA in TensorFlow or PyTorch and compare the results with scikit-learn's implementation.**

### Answer

**Pipeline:**
1. Implement PCA using SVD in PyTorch
2. Apply sklearn PCA
3. Compare results

```python
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 10)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_components = 3

# --- PyTorch PCA ---
def pca_pytorch(X, n_components):
    """PCA using PyTorch SVD"""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Center data
    mean = X_tensor.mean(dim=0)
    X_centered = X_tensor - mean
    
    # SVD
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
    
    # Principal components are rows of Vt (columns of V)
    components = Vt[:n_components, :]
    
    # Project data
    X_transformed = X_centered @ components.T
    
    # Explained variance
    var_explained = (S[:n_components] ** 2) / (X.shape[0] - 1)
    total_var = (S ** 2).sum() / (X.shape[0] - 1)
    var_ratio = var_explained / total_var
    
    return X_transformed.numpy(), components.numpy(), var_ratio.numpy()

X_pytorch, comp_pytorch, var_pytorch = pca_pytorch(X_scaled, n_components)

# --- Sklearn PCA ---
pca_sklearn = PCA(n_components=n_components)
X_sklearn = pca_sklearn.fit_transform(X_scaled)

# --- Compare Results ---
print("=== Comparison ===")
print(f"PyTorch variance ratio: {var_pytorch}")
print(f"Sklearn variance ratio: {pca_sklearn.explained_variance_ratio_}")

# Check if results match (allow sign flip)
correlation = np.abs(np.corrcoef(X_pytorch.T, X_sklearn.T))
print(f"\nCorrelation between components:")
for i in range(n_components):
    print(f"  PC{i+1}: {correlation[i, n_components+i]:.6f}")
```

**Note:** PCA components may have flipped signs (both +v and -v are valid eigenvectors).

---

## Question 6
- [ ] Done

**Demonstrate how to choose an optimal number of dimensions with PCA in Python using the "elbow method".**

### Answer

**Pipeline:**
1. Fit PCA with all components
2. Plot explained variance (scree plot)
3. Plot cumulative variance
4. Identify elbow visually

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
X, y = load_digits(return_X_y=True)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA with all components
pca = PCA()
pca.fit(X_scaled)

# Get variance data
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Scree plot (elbow method)
axes[0].plot(range(1, len(explained_var)+1), explained_var, 'bo-')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Scree Plot (Elbow Method)')
axes[0].axhline(y=0.01, color='r', linestyle='--', label='1% threshold')
axes[0].legend()
axes[0].set_xlim(0, 30)

# Cumulative variance
axes[1].plot(range(1, len(cumulative_var)+1), cumulative_var, 'go-')
axes[1].axhline(y=0.90, color='r', linestyle='--', label='90%')
axes[1].axhline(y=0.95, color='orange', linestyle='--', label='95%')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative Variance')
axes[1].legend()
axes[1].set_xlim(0, 50)

plt.tight_layout()
plt.show()

# Find optimal k programmatically
k_90 = np.argmax(cumulative_var >= 0.90) + 1
k_95 = np.argmax(cumulative_var >= 0.95) + 1

print(f"Components for 90% variance: {k_90}")
print(f"Components for 95% variance: {k_95}")
print(f"Original features: {X.shape[1]}")
```

---

## Question 7
- [ ] Done

**Write a Python script to automatically remove outliers before performing PCA.**

### Answer

**Pipeline:**
1. Detect outliers using IQR or z-score
2. Remove outliers
3. Apply PCA
4. Compare results

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Generate data with outliers
np.random.seed(42)
X = np.random.randn(200, 5)
# Add outliers
X[0] = [10, 10, 10, 10, 10]
X[1] = [-10, -10, -10, -10, -10]
X[2] = [15, -15, 0, 20, -20]

def remove_outliers_iqr(X, factor=1.5):
    """Remove outliers using IQR method"""
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    
    # Keep rows where ALL features are within bounds
    mask = np.all((X >= lower) & (X <= upper), axis=1)
    return X[mask], mask

def remove_outliers_zscore(X, threshold=3):
    """Remove outliers using z-score method"""
    z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
    mask = np.all(z_scores < threshold, axis=1)
    return X[mask], mask

# Remove outliers
X_clean_iqr, mask_iqr = remove_outliers_iqr(X)
X_clean_zscore, mask_zscore = remove_outliers_zscore(X)

print(f"Original samples: {X.shape[0]}")
print(f"After IQR removal: {X_clean_iqr.shape[0]}")
print(f"After Z-score removal: {X_clean_zscore.shape[0]}")

# Compare PCA with and without outliers
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

datasets = [
    (X, "With Outliers"),
    (X_clean_iqr, "After IQR Removal"),
    (X_clean_zscore, "After Z-score Removal")
]

for ax, (data, title) in zip(axes, datasets):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title(f'{title}\n(n={len(data)})')

plt.tight_layout()
plt.show()
```

---

## Question 8
- [ ] Done

**Create a synthetic dataset and show the effect of PCA on classification accuracy using a machine learning algorithm before and after PCA.**

### Answer

**Pipeline:**
1. Create high-dimensional synthetic data
2. Train classifier without PCA
3. Train classifier with different PCA dimensions
4. Compare accuracies

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Step 1: Create synthetic high-dimensional data
X, y = make_classification(
    n_samples=1000,
    n_features=100,      # High dimensional
    n_informative=10,    # Only 10 are useful
    n_redundant=20,      # 20 are combinations of informative
    n_clusters_per_class=2,
    random_state=42
)

print(f"Data shape: {X.shape}")
print(f"Informative features: 10")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 2: Baseline (no PCA)
pipeline_no_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])
scores_no_pca = cross_val_score(pipeline_no_pca, X_train, y_train, cv=5)
print(f"\nNo PCA - Accuracy: {scores_no_pca.mean():.4f} (+/- {scores_no_pca.std():.4f})")

# Step 3: With different PCA dimensions
n_components_list = [5, 10, 20, 30, 50, 70, 90]
results = []

for n_comp in n_components_list:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_comp)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    results.append((n_comp, scores.mean(), scores.std()))
    print(f"PCA({n_comp:2d}) - Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Step 4: Plot results
plt.figure(figsize=(10, 5))
n_comps, means, stds = zip(*results)

plt.errorbar(n_comps, means, yerr=stds, marker='o', capsize=5, label='With PCA')
plt.axhline(y=scores_no_pca.mean(), color='r', linestyle='--', 
            label=f'No PCA ({scores_no_pca.mean():.4f})')
plt.axvline(x=10, color='g', linestyle=':', label='True informative features')

plt.xlabel('Number of PCA Components')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Effect of PCA on Classification Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n=== Insights ===")
print("- Accuracy peaks around true informative dimension (10)")
print("- Too few components: underfitting (lost information)")
print("- Too many components: noise included")
```

---
