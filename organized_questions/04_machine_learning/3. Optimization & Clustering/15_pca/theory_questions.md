# Pca Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What is Principal Component Analysis (PCA)?**

### Answer

**Definition:**
PCA is an unsupervised linear dimensionality reduction technique that transforms a dataset with many correlated features into a smaller set of uncorrelated variables called principal components. These components are ordered by the variance they capture, with PC1 capturing the most variance.

**Core Concepts:**
- Finds orthogonal axes (principal components) aligned with directions of maximum variance
- Eigenvalues represent variance captured by each component
- Eigenvectors define the direction of principal components
- Covariance matrix captures relationships between features

**Mathematical Formulation:**
$$C = \frac{1}{n-1} X^T X$$
$$Cv = \lambda v$$
Where $C$ = covariance matrix, $v$ = eigenvector (principal component), $\lambda$ = eigenvalue (variance)

**Intuition:**
Imagine a 3D cloud of data points. PCA finds the best 2D plane to project this cloud such that the "shadow" preserves maximum spread of points.

**Practical Relevance:**
- Dimensionality reduction for faster model training
- Data visualization (reduce to 2D/3D)
- Noise reduction by discarding low-variance components
- Feature decorrelation for multicollinearity issues

**Algorithm Steps:**
1. Standardize data (mean=0, std=1)
2. Compute covariance matrix
3. Compute eigenvalues and eigenvectors
4. Sort eigenvectors by eigenvalues (descending)
5. Select top k eigenvectors
6. Project data: $X_{new} = X \cdot V_k$

---

## Question 2
- [ ] Done

**Can you explain the concept of eigenvalues and eigenvectors in PCA?**

### Answer

**Definition:**
Eigenvectors are the directions of principal components (new axes). Eigenvalues are scalar values representing the variance captured by each corresponding eigenvector.

**Core Concepts:**
- Eigenvector: Non-zero vector that only gets scaled (not rotated) when multiplied by a matrix
- Eigenvalue: The scaling factor for that eigenvector
- Eigenvectors of covariance matrix = Principal component directions
- Eigenvalues = Variance explained by each component

**Mathematical Formulation:**
$$Cv = \lambda v$$
Where:
- $C$ = Covariance matrix (p × p)
- $v$ = Eigenvector (direction)
- $\lambda$ = Eigenvalue (magnitude/variance)

**Intuition:**
Think of eigenvectors as the natural axes of data spread. The largest eigenvalue points to where data varies most. It's like finding the longest and shortest axes of an ellipse.

**Practical Relevance:**
- Eigenvector with highest eigenvalue → PC1 (most important direction)
- Sort eigenvalues descending to rank component importance
- Explained variance ratio = $\lambda_i / \sum \lambda$

**Algorithm Steps:**
1. Compute covariance matrix $C$
2. Solve characteristic equation: $det(C - \lambda I) = 0$
3. Find eigenvalues $\lambda_1, \lambda_2, ..., \lambda_p$
4. For each $\lambda_i$, solve $(C - \lambda_i I)v = 0$ to get eigenvector $v_i$
5. Eigenvector with largest $\lambda$ = PC1

---

## Question 3
- [ ] Done

**Describe the role of the covariance matrix in PCA.**

### Answer

**Definition:**
The covariance matrix is a square matrix that captures how features vary together. It is the foundation from which principal components are derived through eigen-decomposition.

**Core Concepts:**
- Diagonal elements = Variance of each feature
- Off-diagonal elements = Covariance between feature pairs
- Positive covariance: features increase/decrease together
- Negative covariance: one increases as other decreases
- Near-zero covariance: features are linearly independent

**Mathematical Formulation:**
$$C = \frac{1}{n-1} X^T X$$
For features $A$ and $B$:
$$Cov(A,B) = \frac{\sum(A_i - \bar{A})(B_i - \bar{B})}{n-1}$$

**Intuition:**
The covariance matrix defines the shape and orientation of the data cloud. PCA finds the principal axes of this elliptical cloud by decomposing this matrix.

**Practical Relevance:**
- Eigenvectors of covariance matrix = Principal components
- Eigenvalues = Variance along each principal component
- On standardized data, covariance matrix = correlation matrix

**Key Points:**
- Data must be centered (subtract mean) before computing covariance
- Data should be standardized if features have different scales
- Symmetric matrix: $C = C^T$

---

## Question 4
- [ ] Done

**What is the variance explained by a principal component?**

### Answer

**Definition:**
The variance explained by a principal component is the eigenvalue associated with that component. It measures how much of the total data variability is captured by that single component.

**Core Concepts:**
- Eigenvalue = Variance along that principal component
- Explained Variance Ratio = Proportion of total variance
- Cumulative Explained Variance = Sum of ratios for top k components
- Total variance = Sum of all eigenvalues

**Mathematical Formulation:**
$$\text{Explained Variance Ratio (PC}_i) = \frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j}$$

$$\text{Cumulative Variance (k components)} = \sum_{i=1}^{k} \frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j}$$

**Intuition:**
If PC1 explains 60% variance and PC2 explains 25%, keeping both retains 85% of the original information in just 2 dimensions.

**Practical Relevance:**
- Decide number of components: Keep enough to explain 90-95% variance
- Scree plot: Visualize eigenvalues to find "elbow" point
- Components with low variance often represent noise

**Example:**
If eigenvalues are [4.0, 2.0, 0.5, 0.3, 0.2]:
- Total variance = 7.0
- PC1 explains 4.0/7.0 = 57%
- PC1 + PC2 explains (4.0+2.0)/7.0 = 86%

---

## Question 5
- [ ] Done

**How does scaling of features affect PCA?**

### Answer

**Definition:**
Feature scaling critically affects PCA because PCA is a variance-maximizing algorithm. Features with larger scales will have larger variances and dominate the principal components, leading to biased results.

**Core Concepts:**
- PCA identifies directions of maximum variance
- Unscaled features with large ranges dominate PC1
- Standardization puts all features on equal footing
- Standardized PCA = Eigen-decomposition of correlation matrix

**Mathematical Formulation:**
Standardization (Z-score):
$$z = \frac{x - \mu}{\sigma}$$

After standardization: mean = 0, std = 1 for all features

**Intuition:**
Consider features: Age (20-70) vs Income (30,000-150,000). Without scaling, income's variance dominates. PC1 will align almost entirely with income axis, ignoring age.

**Practical Relevance:**
- Always standardize unless features share same units and scale matters
- Use `StandardScaler` before PCA in sklearn
- Exception: When intentional to let certain features dominate

**Interview Tip:**
"Should you always scale before PCA?" → Yes, unless all features are in same units AND you want variance to reflect importance.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

---

## Question 6
- [ ] Done

**What is the difference between PCA and Factor Analysis?**

### Answer

**Definition:**
PCA is a mathematical data transformation technique that maximizes variance. Factor Analysis is a statistical model that assumes observed variables are caused by underlying latent factors.

**Core Concepts:**

| Aspect | PCA | Factor Analysis |
|--------|-----|-----------------|
| Goal | Account for total variance | Explain common variance via latent factors |
| Model | No model (algorithmic) | Latent variable model: X = Loading × Factor + Error |
| Variance | Decomposes total variance | Separates common and unique variance |
| Direction | PC = f(Variables) | Variable = f(Factors) + Error |
| Output | Orthogonal components | Factor loadings, factors may be rotated |

**Intuition:**
- **PCA**: "How can I best summarize this data with fewer variables?"
- **FA**: "What hidden factors are causing these variables to be correlated?"

**Example:**
For test scores (math, logic, reading, vocabulary):
- **PCA**: Creates weighted combinations to summarize data
- **FA**: Discovers latent factors like "Quantitative Ability" and "Verbal Ability"

**When to Use:**
- **PCA**: Dimensionality reduction, preprocessing, visualization
- **FA**: Understanding underlying constructs, survey development, psychology research

---

## Question 7
- [ ] Done

**Can you explain the Singular Value Decomposition (SVD) and its relationship with PCA?**

### Answer

**Definition:**
SVD factorizes any matrix X into three matrices: $X = U\Sigma V^T$. SVD is the numerically stable method used to compute PCA in practice. The right-singular vectors (V) are the principal components.

**Core Concepts:**
- $U$: Left-singular vectors (n × n orthogonal matrix)
- $\Sigma$: Diagonal matrix of singular values
- $V$: Right-singular vectors (p × p orthogonal matrix) = **Principal Components**
- Singular values relate to eigenvalues: $\lambda_i = \sigma_i^2 / (n-1)$

**Mathematical Formulation:**
$$X = U\Sigma V^T$$

Connection to PCA:
$$C = \frac{1}{n-1}X^TX = \frac{1}{n-1}V\Sigma^T\Sigma V^T$$

Therefore:
- Columns of $V$ = Eigenvectors of covariance matrix = Principal components
- $\sigma_i^2 / (n-1)$ = Eigenvalues (variance explained)
- Projected data (scores) = $U\Sigma$

**Intuition:**
SVD is like breaking down any transformation into three steps: rotation → scaling → rotation. For PCA, we care about the final rotation (V) and the scaling (Σ).

**Why SVD for PCA:**
- **Numerical stability**: Avoids computing $X^TX$ which can lose precision
- **Efficiency**: Faster for datasets with more features than samples
- **sklearn default**: Uses SVD internally for PCA

---

## Question 8
- [ ] Done

**What is meant by 'loading' in the context of PCA?**

### Answer

**Definition:**
Loadings are the correlations (or weights) between original variables and principal components. They indicate how much each original feature contributes to a particular component.

**Core Concepts:**
- High loading (close to ±1): Strong relationship with component
- Low loading (close to 0): Weak relationship
- For standardized data: Loadings ≈ Eigenvector values
- Used to interpret meaning of abstract components

**Mathematical Formulation:**
$$\text{Loading}(j, i) = v_i(j) \times \sqrt{\lambda_i} / s_j$$

For standardized data (s=1):
$$\text{Loading} \approx \text{Eigenvector values}$$

**Intuition:**
If PC1 has high positive loadings for "horsepower" and "engine_size", and negative loading for "mpg", we can interpret PC1 as a "Performance/Power" component.

**Loadings vs Eigenvectors:**
- **Eigenvectors**: Mathematical weights for linear combination
- **Loadings**: Correlation-based interpretation of relationships

**Practical Relevance:**
- Interpret what each component represents
- Identify important features (high loadings on top components)
- Create meaningful labels for abstract components

**Example Interpretation:**
| Feature | PC1 Loading | PC2 Loading |
|---------|-------------|-------------|
| Income | 0.85 | 0.10 |
| Education | 0.80 | 0.15 |
| Age | 0.20 | 0.90 |

PC1 = "Socioeconomic Status", PC2 = "Life Stage"

---

## Question 9
- [ ] Done

**Explain the process of eigenvalue decomposition in PCA.**

### Answer

**Definition:**
Eigenvalue decomposition breaks down the covariance matrix into eigenvectors (directions) and eigenvalues (magnitudes). This reveals the fundamental variance structure needed for PCA.

**Core Concepts:**
- Covariance matrix is symmetric → guaranteed real eigenvalues
- Eigenvectors are orthogonal to each other
- Eigenvalue equation: $Cv = \lambda v$
- Number of eigenvalue-eigenvector pairs = Number of features (p)

**Mathematical Formulation:**

**Step 1:** Compute covariance matrix $C$ (p × p)

**Step 2:** Solve characteristic equation:
$$det(C - \lambda I) = 0$$

**Step 3:** Find eigenvalues $\lambda_1, \lambda_2, ..., \lambda_p$

**Step 4:** For each $\lambda_i$, solve:
$$(C - \lambda_i I)v_i = 0$$

**Step 5:** Result: p eigenvector-eigenvalue pairs

**Algorithm Steps to Remember:**
1. Center data (subtract mean)
2. Compute covariance matrix C
3. Find eigenvalues via characteristic polynomial
4. Find eigenvector for each eigenvalue
5. Sort pairs by eigenvalue (descending)
6. Top k eigenvectors = Top k principal components

**Intuition:**
The equation $Cv = \lambda v$ means: when covariance matrix acts on eigenvector, it only scales it by $\lambda$, not rotates. These are the "natural" directions of the data.

---

## Question 10
- [ ] Done

**What are the limitations of PCA when it comes to handling non-linear relationships?**

### Answer

**Definition:**
Standard PCA is a linear algorithm that assumes data structure lies in a linear subspace. It cannot discover or represent non-linear relationships among variables.

**Core Concepts:**
- PCA finds linear projections (straight axes)
- Non-linear manifolds (curves, spirals) get flattened incorrectly
- Variance maximization only works for linear patterns
- Data on circles, Swiss rolls, S-curves → PCA fails

**Intuition:**
Imagine data points arranged in a spiral (Swiss roll). Any linear projection will flatten and overlap the points, losing the spiral structure entirely.

**Example:**
Data arranged in a circle has no linear direction of high variance. PCA produces two components with equal variance and misses the circular pattern.

**Solutions/Alternatives:**

| Method | Description |
|--------|-------------|
| **Kernel PCA** | Uses kernel trick to handle non-linearity implicitly |
| **t-SNE** | Preserves local neighborhood structure (visualization) |
| **UMAP** | Similar to t-SNE but faster and preserves more global structure |
| **Isomap** | Preserves geodesic distances on manifold |
| **Autoencoders** | Neural networks for non-linear embeddings |

**Interview Tip:**
When asked about PCA limitations → Non-linearity is the primary answer. Mention Kernel PCA as the direct extension.

---

## Question 11
- [ ] Done

**Explain the curse of dimensionality and how PCA can help to mitigate it.**

### Answer

**Definition:**
The curse of dimensionality refers to problems that arise in high-dimensional spaces: data becomes sparse, distances become meaningless, and models overfit. PCA mitigates this by reducing dimensions while preserving important information.

**Problems of High Dimensionality:**
- **Data sparsity**: Volume grows exponentially; fixed data points become sparse
- **Distance concentration**: All distances become similar; k-NN fails
- **Overfitting**: More features than samples → models learn noise
- **Computational cost**: Processing time increases with dimensions

**How PCA Helps:**

| Problem | PCA Solution |
|---------|--------------|
| Sparsity | Reduces dimensions, data becomes denser |
| Distance issues | Fewer dimensions = meaningful distances |
| Overfitting | Fewer features = simpler models |
| Noise | Discards low-variance components (often noise) |
| Multicollinearity | Creates uncorrelated components |

**Intuition:**
With 1000 features and 500 samples, data is very sparse in 1000D space. PCA compresses to 50 dimensions while keeping 95% of information → denser, more learnable representation.

**Practical Relevance:**
- Preprocessing step before distance-based algorithms (k-NN, k-Means)
- Reduces training time
- Improves model generalization

**Interview Tip:**
Always mention PCA removes redundant information (correlated features) and noise (low-variance components).

---

## Question 12
- [ ] Done

**How does PCA handle missing values in the data?**

### Answer

**Definition:**
Standard PCA cannot handle missing values directly. The covariance matrix computation and SVD require complete data. Missing values must be handled in preprocessing.

**Core Concepts:**
- PCA operations need complete numerical matrix
- NaN values cause algorithm to fail
- Preprocessing responsibility falls on user

**Strategies to Handle Missing Values:**

| Strategy | Description | Drawback |
|----------|-------------|----------|
| **Deletion** | Remove rows/columns with missing values | Data loss, potential bias |
| **Mean/Median Imputation** | Replace missing with column mean/median | Reduces variance, weakens correlations |
| **k-NN Imputation** | Use k nearest neighbors to estimate | More accurate but computationally expensive |
| **Probabilistic PCA** | EM algorithm handles missing values internally | More complex implementation |

**Best Practice Pipeline:**
```python
# 1. Split data
X_train, X_test = train_test_split(X)

# 2. Fit imputer on training only
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)

# 3. Transform test with same imputer
X_test_imputed = imputer.transform(X_test)

# 4. Apply PCA
pca = PCA(n_components=k)
X_train_pca = pca.fit_transform(X_train_imputed)
X_test_pca = pca.transform(X_test_imputed)
```

**Interview Tip:**
Mention that imputation should be fit on training data only to avoid data leakage.

---

## Question 13
- [ ] Done

**What is the difference between PCA and t-SNE for dimensionality reduction?**

### Answer

**Definition:**
PCA is a linear technique that preserves global variance. t-SNE is a non-linear technique that preserves local neighborhood structure, primarily used for visualization.

**Core Comparison:**

| Aspect | PCA | t-SNE |
|--------|-----|-------|
| **Goal** | Preserve global variance | Preserve local structure |
| **Linearity** | Linear transformation | Non-linear embedding |
| **Output interpretation** | Distances meaningful, axes interpretable | Distances NOT meaningful, axes NOT interpretable |
| **Determinism** | Deterministic | Stochastic (results vary) |
| **Speed** | Fast, scalable | Slow (O(n²) or O(n log n)) |
| **Use case** | Preprocessing, feature engineering | Visualization only |

**Intuition:**
- **PCA**: Finds the best "shadow" that maximizes spread
- **t-SNE**: Keeps neighbors close, doesn't care about global distances

**Critical Difference:**
- PCA: If points A and B are far apart in original space, they're far in PCA space
- t-SNE: Only guarantees nearby points stay nearby; far points may appear anywhere

**Best Practices:**
- Do NOT use t-SNE for clustering (creates false clusters)
- Use PCA before t-SNE: Reduce to 50D with PCA, then t-SNE to 2D
- For ML preprocessing: Use PCA (not t-SNE)
- For visualization of clusters: Use t-SNE

**Interview Tip:**
t-SNE cluster sizes and between-cluster distances are NOT meaningful!

---

## Question 14
- [ ] Done

**Explain how PCA can be used as a noise reduction technique.**

### Answer

**Definition:**
PCA reduces noise based on the assumption that signal has high variance while noise has low variance. By reconstructing data using only high-variance components, noise is filtered out.

**Core Concepts:**
- Signal → High variance components (dominant patterns)
- Noise → Low variance components (random perturbations)
- Reconstruction with top k components discards noise
- Denoised data = inverse_transform of transformed data

**Process:**
1. Apply PCA to noisy data
2. Keep only top k components (high variance)
3. Transform to k dimensions
4. Inverse transform back to original space
5. Result: Denoised approximation

**Mathematical Formulation:**
$$X_{denoised} = X_{transformed} \cdot V_k^T + \mu$$

Where $V_k$ = top k eigenvectors

**Python Example:**
```python
from sklearn.decomposition import PCA

# Keep components explaining 95% variance
pca = PCA(n_components=0.95)
X_transformed = pca.fit_transform(X_noisy)

# Reconstruct denoised data
X_denoised = pca.inverse_transform(X_transformed)
```

**Limitations:**
- Assumes low variance = noise (not always true)
- Important subtle information in low-variance components may be lost
- Lossy process: reconstructed ≠ original clean data

**Interview Tip:**
This works best when noise is uniformly distributed across all directions while signal is concentrated in few directions.

---

## Question 15
- [ ] Done

**Describe how you would apply PCA for visualization purposes.**

### Answer

**Definition:**
PCA reduces high-dimensional data to 2 or 3 dimensions for plotting, creating a projection that preserves maximum variance and allows visual inspection of data structure.

**Step-by-Step Procedure:**
1. **Load data**: High-dimensional dataset (e.g., 100 features)
2. **Standardize**: Scale features to mean=0, std=1
3. **Apply PCA**: Set n_components=2 (or 3 for 3D)
4. **Plot**: Scatter plot with PC1 on x-axis, PC2 on y-axis
5. **Color by labels**: If available, color points by class

**Python Example:**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. PCA to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 3. Plot
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA Visualization')
plt.show()
```

**Best Practices:**
- Include explained variance % in axis labels
- Color points by target variable to see class separation
- Check cumulative variance: if PC1+PC2 < 50%, plot may be misleading
- Use 3D plot for additional perspective if PC3 has significant variance

**What to Look For:**
- Clusters → Natural groupings in data
- Overlapping classes → Difficult classification problem
- Outliers → Isolated points

---

## Question 16
- [ ] Done

**What are the advantages and drawbacks of kernel PCA compared to linear PCA?**

### Answer

**Definition:**
Kernel PCA extends PCA to handle non-linear data by using the kernel trick to implicitly map data to a higher-dimensional space where non-linear patterns become linear.

**Advantages of Kernel PCA:**

| Advantage | Description |
|-----------|-------------|
| Handles non-linearity | Captures curves, spirals, concentric circles |
| Powerful representation | Finds structure linear PCA misses |
| Kernel flexibility | RBF, polynomial, sigmoid kernels available |

**Drawbacks of Kernel PCA:**

| Drawback | Description |
|----------|-------------|
| Computational cost | O(n²) or O(n³) vs O(p) for linear PCA |
| Parameter tuning | Must choose kernel and tune γ, degree, etc. |
| No interpretability | Components exist in implicit high-dim space |
| No direct reconstruction | Pre-image problem for inverse transform |
| Memory intensive | Stores n×n kernel matrix |

**Common Kernels:**
- **RBF (Gaussian)**: $K(x,y) = exp(-\gamma ||x-y||^2)$ — Most flexible
- **Polynomial**: $K(x,y) = (x^Ty + c)^d$ — Polynomial relationships
- **Sigmoid**: $K(x,y) = tanh(\alpha x^Ty + c)$ — Similar to neural networks

**When to Use:**
- **Linear PCA**: Fast baseline, interpretable, large datasets
- **Kernel PCA**: Known non-linear structure, can afford computation, visualization of complex manifolds

**Interview Tip:**
"How to choose kernel?" → Start with RBF (most versatile), tune γ via cross-validation on downstream task.

---

## Question 17
- [ ] Done

**What are the best practices in visualizing the components obtained from PCA?**

### Answer

**1. 2D Scatter Plot of Components:**
- Plot data on PC1 (x-axis) vs PC2 (y-axis)
- Include explained variance % in axis labels
- Color points by target variable/class

**2. Scree Plot:**
- Bar/line plot of eigenvalues (descending order)
- Shows variance captured by each component
- Look for "elbow" to decide how many components to keep
- Overlay cumulative variance curve

**3. Biplot:**
- Combines sample scores + feature loadings
- Points = samples, Arrows = features
- Arrow length = feature influence
- Arrow direction = correlation with PCs
- Angle between arrows ≈ correlation between features

**4. Loading Plot:**
- Bar chart of loadings for each PC
- Shows which features define each component
- Helps interpret PC meaning

**Best Practices:**
```python
# Include variance in labels
plt.xlabel(f'PC1 ({var_ratio[0]:.1%} variance)')
plt.ylabel(f'PC2 ({var_ratio[1]:.1%} variance)')

# Color by target
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis')

# Add legend for classes
plt.legend()
```

**Interpretation Tips:**
- If PC1+PC2 < 50% variance → 2D plot may be misleading
- Close points in plot = similar samples
- Clusters suggest natural groupings

---

## Question 18
- [ ] Done

**Explain how Incremental PCA differs from standard PCA and when you would use it.**

### Answer

**Definition:**
Incremental PCA (IPCA) processes data in mini-batches instead of loading entire dataset in memory. It provides an approximate but memory-efficient solution for large datasets.

**Comparison:**

| Aspect | Standard PCA | Incremental PCA |
|--------|--------------|-----------------|
| Memory | Full dataset in RAM | Only batch in RAM |
| Computation | Single exact SVD | Iterative partial updates |
| Result | Exact | Approximation |
| Scalability | Limited by RAM | Handles any size |
| API | `fit_transform()` | `partial_fit()` per batch |

**How IPCA Works:**
1. Initialize model
2. For each mini-batch:
   - Call `partial_fit(batch)`
   - Model updates component estimates
3. Final result approximates full PCA

**When to Use:**
- **Big Data**: Dataset larger than RAM (100GB+)
- **Streaming Data**: Data arrives continuously
- **Online Learning**: Model updates as new data arrives

**Python Example:**
```python
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=10, batch_size=1000)

# Process data in chunks
for chunk in data_generator:
    ipca.partial_fit(chunk)

# Transform new data
X_transformed = ipca.transform(X_new)
```

**Trade-off:**
Larger batch size → Better approximation but more memory.

---

## Question 19
- [ ] Done

**How does Generalized PCA differ from standard PCA and what are its applications?**

### Answer

**Definition:**
Generalized PCA (GPCA) extends PCA to handle data from non-Gaussian distributions in the exponential family (Bernoulli, Poisson, Gamma). Standard PCA assumes Gaussian data with squared error loss.

**Key Differences:**

| Aspect | Standard PCA | Generalized PCA |
|--------|--------------|-----------------|
| Data assumption | Gaussian | Exponential family |
| Loss function | Squared error | Log-likelihood of distribution |
| Relationship | Linear | Link function (logit, log, etc.) |
| Best for | Continuous data | Binary, count, positive data |

**Applications:**

| Data Type | Distribution | Example |
|-----------|--------------|---------|
| Binary (0/1) | Bernoulli | Survey responses, click data |
| Counts | Poisson | Word counts, event frequencies |
| Positive continuous | Gamma | Financial amounts, time durations |

**Intuition:**
Standard PCA minimizing squared error is like maximizing Gaussian likelihood. For binary data (0/1), using logistic loss (Bernoulli likelihood) makes more statistical sense.

**Example: Logistic PCA (for binary data)**
- User interaction data: clicked/not clicked
- Standard PCA treats 0/1 as continuous → poor fit
- Logistic PCA models binary nature properly

**When to Use:**
- Data clearly non-Gaussian
- Features are binary or counts
- Standard PCA gives poor reconstruction

---

## Question 20
- [ ] Done

**Define PCA and its optimization objective.**

### Answer

**Definition:**
PCA is an unsupervised linear transformation that projects data onto a lower-dimensional subspace while preserving maximum variance. It has two equivalent optimization objectives.

**Optimization Objectives:**

**1. Variance Maximization:**
Find orthogonal unit vectors that maximize projected variance.
$$\max_{v} v^T C v \quad \text{subject to } v^T v = 1$$

**2. Reconstruction Error Minimization:**
Find subspace that minimizes average squared distance between original and reconstructed points.
$$\min_{V_k} \sum_i ||x_i - \hat{x}_i||^2$$

**Mathematical Proof of Equivalence:**
Using Pythagorean theorem:
$$\text{Total Variance} = \text{Projected Variance} + \text{Reconstruction Error}$$

Since total variance is constant:
- Maximizing projected variance = Minimizing reconstruction error
- Both lead to same solution: top k eigenvectors of covariance matrix

**Intuition:**
- Variance view: "Capture maximum spread"
- Reconstruction view: "Minimize information loss"
- Both find the same optimal subspace

---

## Question 21
- [ ] Done

**Show link between PCA and eigen-decomposition of covariance.**

### Answer

**The Link:**
Principal components ARE the eigenvectors of the covariance matrix. This arises from the variance maximization objective.

**Mathematical Derivation:**

**Step 1:** Objective - Maximize variance of projection
$$\text{Var}(Xv) = v^T C v$$

**Step 2:** Constraint - v is unit vector
$$v^T v = 1$$

**Step 3:** Lagrangian
$$L(v, \lambda) = v^T C v - \lambda(v^T v - 1)$$

**Step 4:** Derivative = 0
$$\frac{\partial L}{\partial v} = 2Cv - 2\lambda v = 0$$
$$Cv = \lambda v$$

**This IS the eigenvalue equation!**

**Step 5:** Finding maximum variance
Substituting $Cv = \lambda v$ back:
$$\text{Variance} = v^T C v = v^T (\lambda v) = \lambda$$

**Conclusion:**
- Variance along eigenvector = its eigenvalue
- To maximize variance → choose eigenvector with largest eigenvalue
- PC1 = eigenvector with largest λ₁
- PC2 = eigenvector with second largest λ₂
- And so on...

---

## Question 22
- [ ] Done

**Explain variance maximization vs. reconstruction error minimization.**

### Answer

**Variance Maximization View:**
- **Goal**: Preserve maximum information (measured as variance)
- **Objective**: Find axes where data is most "spread out"
- **Intuition**: Capture the dominant patterns of variation

$$\max \text{Var}(Xv) = \max v^T C v$$

**Reconstruction Error Minimization View:**
- **Goal**: Best approximate original data from low dimensions
- **Objective**: Minimize distance between original and reconstructed points
- **Intuition**: Find the most faithful compression

$$\min \sum_i ||x_i - \hat{x}_i||^2$$

**Why They're Equivalent:**

For a data point x projected onto subspace:
- $d_1^2$ = variance of projection (preserved)
- $d_2^2$ = reconstruction error (lost)

By Pythagorean theorem:
$$||x||^2 = d_1^2 + d_2^2$$
$$\text{Total} = \text{Preserved} + \text{Lost}$$

Since total is fixed:
$$\max(\text{Preserved}) \equiv \min(\text{Lost})$$

**Both objectives lead to the same solution: eigenvectors of covariance matrix.**

---

## Question 23
- [ ] Done

**Derive PCA via Singular Value Decomposition.**

### Answer

**SVD Definition:**
Any matrix X (n×p) can be factorized as:
$$X = U \Sigma V^T$$

- $U$: n×n orthogonal (left-singular vectors)
- $\Sigma$: n×p diagonal (singular values σᵢ)
- $V$: p×p orthogonal (right-singular vectors)

**Derivation:**

**Step 1:** Covariance matrix
$$C = \frac{1}{n-1} X^T X$$

**Step 2:** Substitute SVD
$$C = \frac{1}{n-1} (U\Sigma V^T)^T (U\Sigma V^T)$$
$$= \frac{1}{n-1} V \Sigma^T U^T U \Sigma V^T$$

Since $U^T U = I$:
$$C = \frac{1}{n-1} V (\Sigma^T \Sigma) V^T$$

**Step 3:** Identify eigen-decomposition
This has form $C = V \Lambda V^T$ where:
- **Columns of V** = Eigenvectors = **Principal Components**
- **Eigenvalues**: $\lambda_i = \sigma_i^2 / (n-1)$

**Step 4:** Projected data (scores)
$$T = XV = U\Sigma V^T V = U\Sigma$$

**Summary:**
- Principal components = Right-singular vectors (V)
- Variance explained = $\sigma_i^2 / (n-1)$
- Scores = $U\Sigma$

**Why SVD for PCA:**
- Numerically stable (avoids computing $X^TX$)
- More efficient for large matrices

---

## Question 24
- [ ] Done

**Discuss importance of data centering before PCA.**

### Answer

**Definition:**
Data centering means subtracting the mean of each feature. It is MANDATORY for PCA to produce correct results.

**Why Centering is Essential:**

**Problem without centering:**
- PCA finds directions of maximum variance
- Variance is defined relative to the mean
- Without centering, PCA uses origin (0,0,...) as reference
- PC1 will point FROM origin TO data center (wrong!)

**Example:**
Data cluster at coordinates (100, 100). Without centering:
- PC1 points toward (100, 100) from origin
- This captures LOCATION, not variance structure
- Completely meaningless component!

**Mathematical Reason:**
$$\text{Cov}(A,B) = E[(A - \mu_A)(B - \mu_B)]$$

Covariance formula inherently involves mean subtraction. Computing $X^TX$ only gives covariance when X is centered.

**Practical Impact:**
- First PC becomes meaningless without centering
- All subsequent PCs (orthogonal to first) also wrong
- Entire analysis invalid

**Key Point:**
`StandardScaler` includes centering by default. Always center before PCA.

```python
# Centering only
X_centered = X - X.mean(axis=0)

# Full standardization (recommended)
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
```

---

## Question 25
- [ ] Done

**Why does scaling features change component loadings?**

### Answer

**Definition:**
Scaling changes feature variances, which changes their influence on PCA. Since loadings represent feature contributions to components, they change when relative variances change.

**Core Logic:**

**Before Scaling:**
- Income: variance = 1,000,000 (large range)
- Children: variance = 0.5 (small range)
- PCA dominated by income
- PC1 loading: Income ≈ 1.0, Children ≈ 0.0

**After Standardization:**
- Income: variance = 1
- Children: variance = 1
- Equal influence on PCA
- Loadings reflect correlation structure, not scale

**Why This Happens:**
1. PCA maximizes variance
2. High-variance features dominate
3. Scaling equalizes variances
4. Loadings now based on correlations, not arbitrary units

**Mathematical View:**

| Condition | Loadings Reflect |
|-----------|------------------|
| Unscaled | Variance + Correlations |
| Scaled | Correlations only |

**Practical Implication:**
- Unscaled: Feature measured in larger units dominates
- Scaled: Features contribute based on actual relationships

**Interview Tip:**
"PCA on standardized data is equivalent to eigen-decomposition of the correlation matrix instead of covariance matrix."

---

## Question 26
- [ ] Done

**Explain meaning of scree plot and elbow criteria.**

### Answer

**Scree Plot:**
A line/bar plot showing eigenvalues (variance) for each component in descending order.

- **X-axis**: Component number (1, 2, 3, ...)
- **Y-axis**: Eigenvalue (variance explained)
- Always downward sloping (sorted by importance)

**Elbow Criterion:**
Heuristic to find the natural cutoff point where adding more components provides diminishing returns.

**Interpretation:**
- **Cliff** (steep part): Components capturing significant variance (keep these)
- **Scree** (flat part): Components capturing little variance (discard these)
- **Elbow**: Point where cliff meets scree (cutoff)

**Visual Example:**
```
Variance
   |
   |*
   | *
   |  *
   |   *
   |    *---*---*---*  ← Elbow here
   +----------------→ Component
     1  2  3  4  5  6
```

Keep components before the elbow (1-3 in this example).

**Limitations:**
- Elbow often not sharp/clear
- Subjective interpretation
- Use with other methods (cumulative variance, Kaiser rule)

**Python:**
```python
plt.plot(range(1, len(pca.explained_variance_)+1), 
         pca.explained_variance_, 'bo-')
plt.xlabel('Component')
plt.ylabel('Eigenvalue')
```

---

## Question 27
- [ ] Done

**How to decide number of components via Kaiser rule.**

### Answer

**Kaiser Rule (Kaiser-Guttman Criterion):**
Retain only components with eigenvalue > 1.0

**Rationale:**
- Applied to standardized data only
- Each standardized feature has variance = 1
- Total variance = number of features (p)
- Average eigenvalue = p/p = 1
- Component with λ < 1 explains less than one original variable
- Such components are not worth keeping

**How to Apply:**
1. Standardize data
2. Perform PCA
3. Count eigenvalues > 1.0
4. Keep that many components

**Example:**
Eigenvalues: [3.2, 1.8, 1.1, 0.7, 0.5, 0.4, 0.3]
- λ > 1: Components 1, 2, 3
- Kaiser rule: Keep 3 components

**Advantages:**
- Simple, objective, easy to apply
- Quick first estimate

**Limitations:**
- 1.0 cutoff is arbitrary
- May keep too few (small datasets) or too many (large datasets)
- Not data-driven like scree plot
- Best used as starting point, not final decision

**Best Practice:**
Combine Kaiser rule with scree plot and cumulative variance for robust decision.

---

## Question 28
- [ ] Done

**Describe cumulative explained variance ratio.**

### Answer

**Definition:**
The cumulative explained variance ratio is the sum of explained variance ratios for the top k components. It shows what percentage of total information is retained.

**Calculation:**

Individual ratio:
$$\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j}$$

Cumulative ratio:
$$\text{Cumulative EVR}(k) = \sum_{i=1}^{k} \text{EVR}_i$$

**Example:**
| Component | Eigenvalue | EVR | Cumulative EVR |
|-----------|------------|-----|----------------|
| PC1 | 4.0 | 57% | 57% |
| PC2 | 2.0 | 29% | 86% |
| PC3 | 0.5 | 7% | 93% |
| PC4 | 0.3 | 4% | 97% |
| PC5 | 0.2 | 3% | 100% |

To retain 90% variance → Keep 3 components

**How to Use:**
1. Set target threshold (90%, 95%, 99%)
2. Find smallest k where cumulative EVR ≥ threshold
3. Keep k components

**Python:**
```python
pca = PCA()
pca.fit(X_scaled)
cumsum = np.cumsum(pca.explained_variance_ratio_)
k = np.argmax(cumsum >= 0.95) + 1  # +1 for 0-indexing
```

**Advantages:**
- Objective criterion
- Interpretable ("keeping 95% of information")
- More robust than elbow method

---

## Question 29
- [ ] Done

**Explain whitening transformation and its risks.**

### Answer

**Definition:**
Whitening transforms PCA output to have unit variance in all components. The result has identity covariance matrix (uncorrelated + equal variance).

**Process:**
1. Perform PCA: Get scores T
2. Scale by eigenvalues: $T_{white} = T / \sqrt{\lambda}$

**Result:**
- Each component: mean=0, variance=1
- Components uncorrelated
- Covariance matrix = Identity

**Benefits:**
- Spherical data distribution
- Useful for some algorithms (ICA, some neural networks)
- Features on exactly same scale

**Risks:**

| Risk | Explanation |
|------|-------------|
| **Information loss** | Destroys relative importance (all variance = 1) |
| **Noise amplification** | Low-variance components (noise) scaled UP to variance=1 |
| **Loss of interpretation** | Can't identify which components were important |

**Example of Noise Amplification:**
- PC1: variance=100 (signal) → scaled to 1
- PC10: variance=0.01 (noise) → scaled to 1 (100x amplification!)

**When to Use:**
- Specific algorithms requiring spherical data (ICA)
- NOT for general dimensionality reduction

**Interview Tip:**
Whitening destroys the "principal" nature of PCA—all components become equally important, even noise.

---

## Question 30
- [ ] Done

**Compare PCA with Factor Analysis.**

### Answer

(See Question 6 for detailed comparison)

**Quick Summary:**

| Aspect | PCA | Factor Analysis |
|--------|-----|-----------------|
| Type | Transformation | Statistical model |
| Goal | Maximize variance | Find latent factors |
| Model | PC = f(Variables) | Variable = f(Factors) + Error |
| Variance | Total variance | Common + Unique variance |
| Output | Components (orthogonal) | Factors (rotatable) |
| Use | Dimensionality reduction | Understand latent structure |

**Key Insight:**
- PCA: "How to summarize?"
- FA: "What's causing correlations?"

---

## Question 31
- [ ] Done

**Describe kernel PCA and nonlinear embeddings.**

### Answer

**Definition:**
Kernel PCA uses the kernel trick to perform PCA in a high-dimensional space implicitly, enabling capture of non-linear relationships.

**How It Works:**
1. Compute kernel matrix K (n×n): $K_{ij} = K(x_i, x_j)$
2. Center the kernel matrix
3. Eigen-decompose K
4. Project data using eigenvectors

**Common Kernels:**
- **RBF**: $K(x,y) = \exp(-\gamma||x-y||^2)$
- **Polynomial**: $K(x,y) = (x^Ty + c)^d$
- **Sigmoid**: $K(x,y) = \tanh(\alpha x^Ty + c)$

**Non-linear Embedding:**
A low-dimensional representation that preserves non-linear manifold structure.

**Other Non-linear Methods:**
- t-SNE: Local structure preservation
- UMAP: Faster, preserves more global structure
- Isomap: Geodesic distance preservation
- Autoencoders: Neural network approach

**Trade-offs:**
- Kernel PCA: O(n²) complexity, needs parameter tuning
- t-SNE/UMAP: Better for visualization, not for preprocessing

---

## Question 32
- [ ] Done

**Discuss incremental PCA for streaming data.**

### Answer

(See Question 20 for detailed explanation)

**Key Points for Streaming:**
- Data arrives continuously
- Cannot store all data
- Model must update incrementally

**Process:**
```python
ipca = IncrementalPCA(n_components=k)

# As data streams in:
while new_batch_arrives:
    batch = get_next_batch()
    ipca.partial_fit(batch)
```

**Considerations:**
- **Concept drift**: If data distribution changes, old components become stale
- **Batch size**: Larger = better approximation, more memory
- **Stationary assumption**: Standard IPCA assumes stable distribution

**Use Cases:**
- Real-time sensor data analysis
- Network traffic monitoring
- Online recommendation systems

---

## Question 33
- [ ] Done

**Explain randomized SVD acceleration.**

### Answer

**Definition:**
Randomized SVD is a probabilistic algorithm that computes approximate SVD much faster than exact methods, especially for large matrices when only top k components are needed.

**Core Idea:**
Find a smaller subspace that captures most of the matrix's action, then do exact SVD on that smaller space.

**Algorithm Steps:**
1. Generate random matrix R
2. Compute Y = XR (random projection)
3. Find orthonormal basis Q for Y (QR decomposition)
4. Project: B = Q^T X (smaller matrix)
5. Compute exact SVD of small B
6. Reconstruct approximate SVD of X

**Complexity:**
- Exact SVD: O(np²) or O(n²p)
- Randomized: O(npk) where k << min(n,p)

**When to Use:**
- High-dimensional data (many features)
- Only need top k components
- Speed matters more than exact precision

**In sklearn:**
```python
pca = PCA(n_components=50, svd_solver='randomized')
```

**Accuracy:**
Very close to exact PCA for most practical purposes.

---

## Question 34
- [ ] Done

**Describe robust PCA and handling outliers.**

### Answer

**Problem:**
Standard PCA is highly sensitive to outliers. Single extreme points can skew principal components.

**Robust PCA Approach:**
Decompose data matrix D into:
$$D = L + S$$
- $L$: Low-rank matrix (clean data structure)
- $S$: Sparse matrix (outliers/errors)

**Objective:**
Find L with minimum rank and S with maximum sparsity.

**How It Handles Outliers:**
- Outliers get captured in sparse matrix S
- Low-rank matrix L contains clean structure
- PCA on L is robust to original outliers

**Applications:**
- **Video surveillance**: Background (L) vs moving objects (S)
- **Image denoising**: Clean image (L) vs corrupted pixels (S)
- **Data cleaning**: Identify and separate outliers

**Alternative Approaches:**
1. Remove outliers before PCA (using IQR, z-score)
2. Use median instead of mean for centering
3. Winsorize extreme values
4. Use robust covariance estimation (Minimum Covariance Determinant)

**Interview Tip:**
Standard PCA objective uses squared error—outliers have disproportionate influence. Robust PCA uses different objective (nuclear norm + L1 norm).

---

## Question 35
- [ ] Done

**Explain projecting new samples into PCA space.**

### Answer

**Definition:**
After training PCA on training data, new samples must be transformed using the SAME learned parameters (mean, scale, components).

**Critical Rule:**
FIT on training data, TRANSFORM on new data. Never refit!

**Process:**
```python
# 1. Fit on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=k)
X_train_pca = pca.fit_transform(X_train_scaled)

# 2. Transform new data (NO fit!)
X_new_scaled = scaler.transform(X_new)  # NOT fit_transform
X_new_pca = pca.transform(X_new_scaled)  # NOT fit_transform
```

**Why This Matters:**
- New data must be in same coordinate system as training
- Refitting creates different transformation
- Model trained on old PCA space won't work on new PCA space

**Mathematical Operation:**
$$X_{new,pca} = (X_{new} - \mu_{train}) \cdot V$$

Where $\mu_{train}$ and $V$ are learned from training data.

**Common Mistake:**
Using `fit_transform()` on test data → Creates incompatible transformation!

---

## Question 36
- [ ] Done

**Discuss PCA for missing-value imputation.**

### Answer

**Concept:**
Use PCA reconstruction to estimate missing values based on data correlation structure.

**Iterative PCA Imputation Algorithm:**
1. **Initialize**: Fill missing with column means
2. **Iterate**:
   - Perform PCA with k components
   - Reconstruct data via inverse transform
   - Replace only originally missing values with reconstructed values
   - Keep original values unchanged
3. **Converge**: Stop when imputed values stabilize

**Why It Works:**
- PCA captures correlation patterns
- Reconstruction uses relationships between features
- Missing values estimated from correlated present features

**Advantages over Mean Imputation:**
- Uses multivariate relationships
- More accurate estimates
- Preserves correlation structure

**Limitations:**
- Computationally expensive
- Assumes low-rank structure
- May not work for random missing patterns

**Implementation:**
```python
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10)
X_imputed = imputer.fit_transform(X_with_missing)
```

---

## Question 37
- [ ] Done

**Compare PCA vs. autoencoders for dimensionality reduction.**

### Answer

**Comparison:**

| Aspect | PCA | Autoencoder |
|--------|-----|-------------|
| Linearity | Linear only | Non-linear possible |
| Training | Algebraic (no iteration) | Backpropagation |
| Speed | Fast | Slow to train |
| Parameters | None (just k) | Many hyperparameters |
| Interpretability | Loadings interpretable | Black box |
| Data requirement | Works on small data | Needs large data |
| Result | Deterministic | May vary |

**When to Use PCA:**
- Fast baseline
- Linear relationships sufficient
- Small to medium data
- Interpretability needed

**When to Use Autoencoder:**
- Complex non-linear patterns
- PCA gives poor results
- Large dataset available
- Can invest in tuning

**Interesting Connection:**
Linear autoencoder (no activation functions) with single hidden layer learns the same subspace as PCA.

**Practical Approach:**
1. Start with PCA (baseline)
2. If model performance poor → try autoencoder
3. Compare reconstruction error and downstream task performance

---

## Question 38
- [ ] Done

**Explain biplot interpretation.**

### Answer

**Definition:**
A biplot displays both sample scores (points) and feature loadings (arrows) in the same PCA plot.

**Components:**
- **Points**: Data samples positioned by PC1, PC2 scores
- **Arrows**: Original features showing loadings on PC1, PC2

**How to Interpret:**

**1. Points (Samples):**
- Close points = similar samples
- Clusters = natural groupings

**2. Arrows (Features):**
- **Length**: Feature's influence (longer = more important)
- **Direction**: Correlation with PCs

**3. Angle Between Arrows:**
- 0° (parallel): Highly positively correlated
- 180° (opposite): Highly negatively correlated
- 90° (perpendicular): Uncorrelated

**4. Sample-Feature Relationship:**
- Sample far along arrow direction = high value for that feature
- Project sample onto arrow to estimate feature value

**Example Interpretation:**
If "income" arrow points right and a sample is far right:
→ That sample likely has high income

**Python:**
```python
# Plot scores
plt.scatter(scores[:,0], scores[:,1])

# Plot loadings as arrows
for i, feature in enumerate(features):
    plt.arrow(0, 0, loadings[i,0], loadings[i,1])
    plt.text(loadings[i,0], loadings[i,1], feature)
```

**Best For:**
Understanding relationships between samples AND features simultaneously.

---

## Question 39
- [ ] Done

**Derive the PCA from the optimization perspective, i.e., minimization of reconstruction error.**

### Answer

**Objective:**
Find k-dimensional subspace minimizing reconstruction error.

**Setup:**
- $V_k$: Orthonormal basis (p × k)
- Projection: $\hat{x} = V_k V_k^T x$
- Error: $||x - \hat{x}||^2$

**Derivation:**

$$J = \sum_i ||x_i - V_k V_k^T x_i||^2$$

Expanding:
$$= \sum_i (x_i^T x_i - x_i^T V_k V_k^T x_i)$$

First term is constant. Minimizing J = Maximizing:
$$\sum_i x_i^T V_k V_k^T x_i = Tr(V_k^T X^T X V_k)$$

**Result:**
Maximize $Tr(V_k^T C V_k)$ subject to $V_k^T V_k = I$

**Solution:** Columns of $V_k$ = top k eigenvectors of C

---

## Question 40
- [ ] Done

**How do you determine the number of principal components to use?**

### Answer

**Methods:**

| Method | Approach |
|--------|----------|
| **Cumulative Variance** | Keep k components for 90-95% variance |
| **Scree Plot** | Find elbow where curve flattens |
| **Kaiser Rule** | Keep components with eigenvalue > 1 |
| **Cross-validation** | Tune k for best downstream model performance |

**Recommended Approach:**
1. Plot cumulative variance
2. Check scree plot for elbow
3. If for ML model, cross-validate k

**Python:**
```python
cumsum = np.cumsum(pca.explained_variance_ratio_)
k = np.argmax(cumsum >= 0.95) + 1
```

---

## Question 41
- [ ] Done

**Provide examples of how PCA can be used in image processing.**

### Answer

**1. Facial Recognition (Eigenfaces):**
- Flatten face images to vectors
- PCA finds principal "eigenfaces"
- Represent any face as combination of eigenfaces
- Match new faces in low-dimensional space

**2. Image Compression:**
- Divide image into blocks
- PCA on block collection
- Store only top component scores
- Reconstruct with minimal quality loss

**3. Hyperspectral Imaging:**
- Hundreds of spectral bands (highly correlated)
- PCA reduces to few components
- Enables efficient classification

**4. Handwriting Recognition:**
- Extract principal components of digit variations
- Use as features for classification

---

## Question 42
- [ ] Done

**Can PCA be applied to categorical data? Why or why not?**

### Answer

**Short Answer:**
Standard PCA is NOT appropriate for categorical data.

**Why Not:**
- PCA assumes continuous, Gaussian-like data
- Covariance/correlation meaningless for categories
- Distance metrics don't apply to categories
- One-hot encoding creates sparse, high-dimensional data

**Alternatives:**

| Method | Description |
|--------|-------------|
| **MCA (Multiple Correspondence Analysis)** | PCA equivalent for categorical data |
| **FAMD** | Mixed data (categorical + numerical) |
| **Categorical PCA (CATPCA)** | Optimal scaling for categories |

**If You Must Use PCA:**
- One-hot encode categories
- But results may be misleading
- Better to use MCA

---

## Question 43
- [ ] Done

**How can PCA be parallelized to handle very large datasets?**

### Answer

**Approaches:**

**1. Incremental PCA:**
- Process mini-batches sequentially
- Update components iteratively
- Memory efficient

**2. Randomized SVD:**
- Use random projections
- Approximate top k components
- Much faster for large matrices

**3. Distributed PCA:**
- Split data across nodes
- Compute local covariances
- Aggregate and decompose

**4. Parallel SVD Libraries:**
- Use GPU-accelerated libraries (cuML, RAPIDS)
- Distributed computing (Spark MLlib)

**sklearn Options:**
```python
# Randomized (faster)
PCA(n_components=k, svd_solver='randomized')

# Incremental (memory efficient)
IncrementalPCA(n_components=k, batch_size=1000)
```

---

## Question 44
- [ ] Done

**Compare the use of PCA to select features with other feature selection methods.**

### Answer

**Key Distinction:**
PCA is feature EXTRACTION (creates new features), not feature SELECTION (picks original features).

**Comparison:**

| Aspect | PCA | Feature Selection |
|--------|-----|-------------------|
| Output | New synthetic features | Subset of original features |
| Interpretability | Less interpretable | Original feature meaning preserved |
| Supervision | Unsupervised | Can be supervised |
| Information | Combines all features | Discards features |

**Feature Selection Methods:**
- **Filter**: Correlation, mutual information, chi-square
- **Wrapper**: RFE, forward/backward selection
- **Embedded**: L1 regularization, tree importance

**When to Use PCA:**
- Multicollinearity present
- Want decorrelated features
- Interpretability less important

**When to Use Feature Selection:**
- Need interpretable features
- Domain knowledge suggests specific features matter
- Regulatory requirements for explainability

---

## Question 45
- [ ] Done

**What recent advancements have been made concerning PCA for big data?**

### Answer

**Key Advancements:**

**1. Randomized PCA:**
- Probabilistic approximation
- O(npk) instead of O(np²)
- Near-exact results

**2. Streaming/Online PCA:**
- Processes data incrementally
- Handles concept drift
- Memory independent of data size

**3. GPU Acceleration:**
- RAPIDS cuML library
- Orders of magnitude faster
- Handles larger datasets

**4. Distributed Implementations:**
- Spark MLlib PCA
- Handles TB-scale data
- Parallel covariance computation

**5. Sparse PCA:**
- Produces sparse loadings
- More interpretable components
- Useful for high-dimensional data

**6. Robust PCA:**
- Handles outliers
- Separates low-rank structure from sparse noise

---

## Question 46
- [ ] Done

**Discuss the importance of the trace of a matrix in the context of PCA.**

### Answer

**Definition:**
The trace of a matrix is the sum of its diagonal elements. In PCA context, the trace of the covariance matrix equals the total variance of the dataset.

**Why It Matters:**

**1. Total Variance:**
$$Tr(C) = \sum_{i=1}^{p} \lambda_i = \text{Total Variance}$$

The trace of covariance matrix = sum of all eigenvalues = total variance.

**2. Variance Preservation:**
When selecting k components, the preserved variance is:
$$\text{Preserved} = \sum_{i=1}^{k} \lambda_i$$

Explained variance ratio = Preserved / Tr(C)

**3. Reconstruction Error:**
$$\text{Reconstruction Error} = Tr(C) - \sum_{i=1}^{k} \lambda_i$$

**Practical Implication:**
- Trace remains constant regardless of PCA transformation
- Helps verify PCA implementation correctness
- Used in deriving reconstruction error formulas

**Interview Point:**
"The trace tells us the total information budget. PCA redistributes this into components, with top components getting larger shares."

---

## Question 47
- [ ] Done

**Discuss the application of PCA in feature engineering.**

### Answer

**How PCA Helps Feature Engineering:**

**1. Decorrelation:**
- Original features often correlated
- PCA produces uncorrelated components
- Benefits: Removes multicollinearity, helps linear models

**2. Noise Filtering:**
- Keep top components, discard low-variance ones
- Low-variance components often capture noise
- Result: Cleaner features for downstream models

**3. Dimensionality Reduction:**
- Reduce 1000 features to 50 components (95% variance)
- Faster training, less storage
- Prevents overfitting

**4. Creating New Features:**
- PC scores as new engineered features
- Can combine with original features
- Components capture underlying patterns

**Scenario Example:**
Customer dataset with 100 purchase features:
1. Apply PCA → First 3 components
2. PC1 = "Overall spending" (high loadings on all purchase types)
3. PC2 = "Online vs Offline preference"
4. PC3 = "Luxury vs Necessity preference"
5. Use these 3 interpretable components for segmentation

**Caution:**
PCA is unsupervised—components maximize variance, not predictive power. Validate with downstream task performance.

---

## Question 48
- [ ] Done

**Discuss how Randomized PCA is used and its benefits over traditional PCA.**

### Answer

**What is Randomized PCA?**
A probabilistic algorithm that computes approximate SVD using random projections, much faster than exact methods for large matrices.

**How It Works:**

1. **Random Projection**: Project data onto random low-dimensional subspace
2. **Find Basis**: Compute orthonormal basis of projected space
3. **Small SVD**: Do exact SVD on smaller projected matrix
4. **Reconstruct**: Approximate original SVD from smaller decomposition

**Benefits Over Traditional PCA:**

| Aspect | Traditional PCA | Randomized PCA |
|--------|-----------------|----------------|
| **Complexity** | O(np²) or O(n²p) | O(npk) |
| **Memory** | Full matrix operations | Lower memory footprint |
| **Speed** | Slow for large data | 10-100x faster |
| **Accuracy** | Exact | Near-exact for top k |
| **When Faster** | Always exact | k << min(n,p) |

**When to Use Randomized PCA:**
- Large datasets (millions of rows)
- High-dimensional data (thousands of features)
- Only need top k components (k << p)
- Speed matters more than exact precision

**sklearn Usage:**
```python
from sklearn.decomposition import PCA

# Randomized (fast)
pca_fast = PCA(n_components=50, svd_solver='randomized')

# Auto-select (sklearn chooses best)
pca_auto = PCA(n_components=50, svd_solver='auto')

# Full exact (slower)
pca_exact = PCA(n_components=50, svd_solver='full')
```

**Practical Note:**
sklearn defaults to 'auto' which uses randomized for large data with small k. Results are typically indistinguishable from exact PCA.

---

## Question 49
- [ ] Done

**Discuss how robust PCA attempts to handle outliers and its practical implications.**

### Answer

**Problem:**
Standard PCA is highly sensitive to outliers due to squared error objective.

**Robust PCA Approach:**
Decompose data matrix D into:
$$D = L + S$$

- $L$: Low-rank matrix (true data structure)
- $S$: Sparse matrix (outliers/corruptions)

**Optimization Objective:**
$$\min_{L,S} ||L||_* + \lambda ||S||_1 \quad \text{s.t.} \quad D = L + S$$

- $||L||_*$: Nuclear norm (encourages low rank)
- $||S||_1$: L1 norm (encourages sparsity)

**How It Handles Outliers:**
1. Outliers captured in sparse matrix S
2. Low-rank L contains clean structure
3. PCA on L is robust to original outliers

**Practical Implications:**

| Implication | Description |
|-------------|-------------|
| **Automatic outlier detection** | S matrix identifies corrupted entries |
| **Better components** | L gives clean structure |
| **No manual preprocessing** | Algorithm handles outliers internally |
| **Higher computation** | Slower than standard PCA |

**Applications:**
- **Video surveillance**: Background (L) vs moving objects (S)
- **Image denoising**: Clean image (L) vs corrupted pixels (S)
- **Data cleaning**: Identify and separate outliers

**Implementation:**
```python
# Not in sklearn, but available in specialized packages
# Pseudocode concept:
from robust_pca import RobustPCA

rpca = RobustPCA()
L, S = rpca.fit_transform(X)

# L contains clean low-rank approximation
# S contains sparse outliers
standard_pca.fit(L)  # PCA on clean data
```

**Trade-offs:**
- More robust results
- Higher computational cost
- Requires tuning λ parameter
- Best for data with gross errors, not just noisy data

---
