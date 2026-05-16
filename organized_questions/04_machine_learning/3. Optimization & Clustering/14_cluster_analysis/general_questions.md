# Cluster Analysis Interview Questions - General Questions

## Question 1
- [ ] Done

**What preprocessing steps would you suggest before performing cluster analysis?**

### Answer

**Definition:**
Preprocessing prepares data for clustering by handling missing values, scaling features, encoding categories, reducing dimensions, and treating outliers - essential for distance-based algorithms.

**Key Preprocessing Steps:**

| Step | Purpose | Methods |
|------|---------|---------|
| Missing Values | Algorithms need complete data | Imputation (mean/median), Deletion |
| Feature Scaling | Equalize feature contributions | StandardScaler, MinMaxScaler |
| Categorical Encoding | Convert to numeric | One-Hot Encoding, Label Encoding |
| Dimensionality Reduction | Handle high-D | PCA, t-SNE |
| Outlier Treatment | Prevent distortion | Removal, Robust algorithms |

**Python Pipeline:**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),  # Optional
])
X_preprocessed = pipeline.fit_transform(X)
```

**Critical Points:**
- **Scaling is mandatory** for distance-based algorithms (K-Means, DBSCAN)
- StandardScaler preferred over MinMaxScaler (less outlier sensitive)
- Reduce dimensions for high-D data (> 20 features)

---

## Question 2
- [ ] Done

**How might you address missing values in a dataset before clustering?**

### Answer

**Definition:**
Missing values must be handled before clustering since most algorithms can't process incomplete data. Strategies range from simple deletion to sophisticated imputation methods.

**Strategies:**

| Method | Description | Pros/Cons |
|--------|-------------|-----------|
| **Listwise Deletion** | Remove rows with missing values | Simple but loses data |
| **Mean/Median Imputation** | Replace with column mean/median | Fast but reduces variance |
| **Mode Imputation** | For categorical - use most frequent | Simple but may bias |
| **k-NN Imputation** | Use k nearest neighbors' values | Better accuracy, slower |
| **MICE** | Multiple imputation by chained equations | Best accuracy, complex |

**Python Examples:**
```python
from sklearn.impute import SimpleImputer, KNNImputer

# Simple imputation
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# k-NN imputation (better accuracy)
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(X)
```

**Treat Missingness as Feature:**
```python
import numpy as np

# Create indicator column
X['feature_missing'] = X['feature'].isna().astype(int)
# Then impute the original
X['feature'] = X['feature'].fillna(X['feature'].median())
```

**Best Practice:**
Start with median imputation, use k-NN for important applications.

---

## Question 3
- [ ] Done

**How can the elbow method help in selecting the optimal number of clusters?**

### Answer

**Definition:**
The Elbow Method plots Within-Cluster Sum of Squares (WCSS/Inertia) against different k values. The "elbow" point where the rate of decrease sharply slows indicates optimal k.

**Procedure:**
1. Run K-Means for k = 1, 2, 3, ..., 10
2. Calculate WCSS for each k
3. Plot k vs WCSS
4. Find the "elbow" - point of diminishing returns

**Python Code:**
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(K, wcss, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method')
plt.show()
```

**Interpretation:**
```
WCSS
  |
  |●
  | ●
  |  ●
  |   ●●●●●●  ← Elbow here (k=4)
  |_____________ k
      1 2 3 4 5 6 7
```

**Limitations:**
- Elbow often not clear/ambiguous
- Subjective interpretation
- Use with Silhouette Score for confirmation

---
