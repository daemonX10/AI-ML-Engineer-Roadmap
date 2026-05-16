# Dimensionality Reduction Interview Questions - General Questions

## Question 1
- [ ] Done

**How can dimensionality reduction prevent overfitting?**

### Answer

**Definition:**  
Dimensionality reduction prevents overfitting by reducing the number of parameters the model needs to learn, eliminating noise and redundant features, and forcing the model to focus on the most important patterns in the data.

**How It Prevents Overfitting:**

| Mechanism | Explanation |
|-----------|-------------|
| **Fewer Parameters** | Less capacity to memorize noise |
| **Noise Removal** | Low-variance components often contain noise |
| **Regularization Effect** | Constrains model complexity implicitly |
| **Better Generalization** | Captures underlying structure, not artifacts |
| **Sample-to-Feature Ratio** | Improves when features reduced |

**Mathematical Intuition:**
- Overfitting risk ∝ (number of features / number of samples)
- Reducing d while keeping n constant → lower risk
- VC dimension decreases with fewer features

**When It Helps Most:**
- High-dimensional data (d >> n)
- Many correlated features
- Noisy features present
- Small training set

**Python Example:**
```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Without PCA - may overfit with many features
model = LogisticRegression()
score_original = cross_val_score(model, X, y, cv=5).mean()

# With PCA - reduces overfitting
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
score_reduced = cross_val_score(model, X_reduced, y, cv=5).mean()

# score_reduced often > score_original when d is large
```

---

## Question 2
- [ ] Done

**When would you use dimensionality reduction in the machine learning pipeline?**

### Answer

**Definition:**  
Use dimensionality reduction when facing high-dimensional data, computational constraints, multicollinearity, visualization needs, or when the curse of dimensionality affects model performance. It typically comes after preprocessing and before model training.

**When to Use:**

| Scenario | Reason |
|----------|--------|
| d >> n (features > samples) | Prevent overfitting |
| Training too slow | Reduce computational cost |
| Visualization needed | Reduce to 2D/3D |
| Multicollinearity present | Remove redundant features |
| Distance-based algorithms | Make distances meaningful |
| Noise in data | Remove noisy dimensions |

**Pipeline Position:**
```
Data → Clean → Handle Missing → Scale → Dim Reduction → Model → Evaluate
```

**When NOT to Use:**
- Tree-based models (handle high dims well)
- Need interpretable features
- All features carry unique information
- d is already small

**Python Pipeline Example:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

pipeline = Pipeline([
    ('scaler', StandardScaler()),        # Step 1: Scale
    ('pca', PCA(n_components=0.95)),     # Step 2: Reduce
    ('classifier', SVC())                 # Step 3: Model
])

pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
```

---

## Question 3
- [ ] Done

**Can dimensionality reduction be reversed? Why or why not?**

### Answer

**Definition:**  
Dimensionality reduction can be partially reversed for some methods (like PCA) through inverse transformation, but information is permanently lost. Non-linear methods like t-SNE have no inverse, and feature selection cannot recover discarded features.

**Reversibility by Method:**

| Method | Reversible? | Reason |
|--------|-------------|--------|
| **PCA** | Partial | Can reconstruct, but loses variance in dropped components |
| **LDA** | Partial | Similar to PCA, information lost |
| **t-SNE** | No | No inverse function exists |
| **UMAP** | Partial | Has inverse_transform but approximate |
| **Feature Selection** | No | Original features discarded |
| **Autoencoders** | Yes (by design) | Decoder reconstructs input |

**PCA Reconstruction:**
```
Forward:  Z = X × W_k          (project to k dims)
Inverse:  X̂ = Z × W_k^T        (reconstruct)
Loss:     X - X̂               (information lost)
```

**Mathematical Explanation:**
- PCA keeps top k eigenvectors
- Discarded (d-k) components are lost forever
- Reconstruction error = Σᵢ₌ₖ₊₁ᵈ λᵢ (sum of discarded eigenvalues)

**Python Example:**
```python
from sklearn.decomposition import PCA
import numpy as np

# Forward transformation
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)  # d dims → 2 dims

# Inverse transformation (partial reconstruction)
X_reconstructed = pca.inverse_transform(X_reduced)  # 2 dims → d dims

# Reconstruction error (information lost)
reconstruction_error = np.mean((X - X_reconstructed) ** 2)
print(f"MSE: {reconstruction_error}")  # Non-zero = info lost
```

**Key Point:** The more components you keep, the better the reconstruction, but true original data cannot be perfectly recovered.

---
