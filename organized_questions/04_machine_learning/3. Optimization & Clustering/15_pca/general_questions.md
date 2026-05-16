# Pca Interview Questions - General Questions

## Question 1
- [ ] Done

**How is PCA used for dimensionality reduction?**

### Answer

**Definition:**
PCA reduces dimensions by transforming correlated features into a smaller set of uncorrelated principal components, keeping those that capture maximum variance.

**Process:**
1. Standardize data
2. Compute covariance matrix
3. Find eigenvectors (directions) and eigenvalues (variance)
4. Sort by eigenvalue, select top k
5. Project: $X_{new} = X \cdot V_k$

**Example:**
100 features → PCA → Keep 10 components capturing 95% variance

**Practical Benefits:**
- Faster model training
- Reduced storage
- Combat overfitting
- Enable visualization (reduce to 2D/3D)

---

## Question 2
- [ ] Done

**Why is PCA considered an unsupervised technique?**

### Answer

**Definition:**
PCA is unsupervised because it uses only input features (X) without any target labels (y). It finds structure based purely on variance within the data.

**Key Points:**
- No labels used in computation
- Objective: maximize variance (data-internal goal)
- Doesn't optimize for prediction
- Contrast: LDA uses labels to maximize class separation

**Implication:**
PCA can be used as preprocessing for both supervised and unsupervised downstream tasks.

---
