# Linear Algebra Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What is a vector and how is it used in machine learning?**

### Answer

**Definition:**  
A vector is an ordered collection of numbers (scalars) that represents a point or direction in n-dimensional space. In ML, vectors encode features, weights, and data samples.

**Core Concepts:**
- A vector in ℝⁿ has n components: **v** = [v₁, v₂, ..., vₙ]
- Column vector (n×1) vs Row vector (1×n)
- Magnitude (norm): ||**v**|| = √(v₁² + v₂² + ... + vₙ²)
- Direction: unit vector **v̂** = **v** / ||**v**||

**Mathematical Formulation:**
$$\mathbf{v} \in \mathbb{R}^n, \quad \mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

**Intuition:**
- Geometrically: an arrow from origin to a point in space
- Each dimension represents a feature (e.g., age, income, height)
- A data sample with 100 features is a vector in ℝ¹⁰⁰

**ML Applications:**
- **Feature vectors**: each data point is a vector of features
- **Word embeddings**: words as dense vectors (Word2Vec, GloVe)
- **Weight vectors**: model parameters in linear models
- **Gradient vectors**: direction of steepest ascent in optimization

**Python Example:**
```python
import numpy as np

# Feature vector for a data sample
feature_vector = np.array([25, 50000, 5.8])  # age, income, height

# Compute magnitude (L2 norm)
magnitude = np.linalg.norm(feature_vector)

# Normalize to unit vector
unit_vector = feature_vector / magnitude

# Dot product (similarity measure)
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot(v1, v2)  # 32
```

**Interview Tips:**
- Always clarify column vs row vector convention
- Know the difference between L1, L2, and L∞ norms
- Vectors are the foundation—everything in ML builds on them

---

## Question 2
- [ ] Done

**Explain the difference between a scalar and a vector.**

### Answer

**Definition:**  
A **scalar** is a single numerical value (0-dimensional), while a **vector** is an ordered array of scalars (1-dimensional) representing magnitude and direction.

**Core Concepts:**

| Property | Scalar | Vector |
|----------|--------|--------|
| Dimension | 0-D (single value) | 1-D (array of values) |
| Notation | lowercase italic: *a*, *λ* | bold/arrow: **v**, v⃗ |
| Example | temperature = 25°C | position = [3, 4, 5] |
| Operations | basic arithmetic | dot product, cross product, norms |

**Mathematical Formulation:**
$$\text{Scalar: } \alpha \in \mathbb{R}$$
$$\text{Vector: } \mathbf{v} \in \mathbb{R}^n = [v_1, v_2, \ldots, v_n]$$

**Intuition:**
- Scalar: "how much" (magnitude only) — e.g., speed = 60 km/h
- Vector: "how much + which way" — e.g., velocity = 60 km/h North

**ML Applications:**
- **Scalars**: learning rate (α), loss value, bias term, regularization coefficient (λ)
- **Vectors**: feature vectors, weight vectors, gradient vectors, embedding vectors

**Python Example:**
```python
import numpy as np

# Scalar
learning_rate = 0.01  # single float value
loss = 0.25

# Vector
weights = np.array([0.5, -0.3, 0.8])  # weight vector
features = np.array([1.0, 2.0, 3.0])  # feature vector

# Scalar-vector multiplication
scaled_weights = learning_rate * weights

# Scalar from vector operation (dot product)
prediction = np.dot(weights, features)  # returns scalar: 2.3
```

**Interview Tips:**
- Scalar operations are element-wise; vector operations have geometric meaning
- In tensor terminology: scalar is rank-0, vector is rank-1
- A scalar can be viewed as a 1×1 matrix or a vector of length 1

---

## Question 3
- [ ] Done

**What is a matrix and why is it central to linear algebra?**

### Answer

**Definition:**  
A matrix is a 2-dimensional rectangular array of numbers arranged in rows and columns. It is the fundamental data structure for representing linear transformations, systems of equations, and datasets.

**Core Concepts:**
- Dimensions: m×n matrix has m rows and n columns
- Element access: Aᵢⱼ = element at row i, column j
- Square matrix: m = n
- Types: diagonal, symmetric, orthogonal, sparse, identity

**Mathematical Formulation:**
$$A \in \mathbb{R}^{m \times n} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

**Why Central to Linear Algebra:**
- **Represents linear transformations**: rotation, scaling, projection
- **Encodes systems of equations**: Ax = b
- **Stores datasets**: rows = samples, columns = features
- **Enables decomposition**: SVD, eigendecomposition, LU, QR

**Intuition:**
- A matrix transforms one vector space to another
- Think of it as a "function" that maps input vectors to output vectors
- Dataset of 1000 samples with 50 features = 1000×50 matrix

**ML Applications:**
- **Data matrix**: X ∈ ℝⁿˣᵈ (n samples, d features)
- **Weight matrices**: layers in neural networks
- **Covariance matrix**: captures feature relationships
- **Adjacency matrix**: graph neural networks

**Python Example:**
```python
import numpy as np

# Create a matrix (dataset: 3 samples, 4 features)
X = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

print(X.shape)  # (3, 4)

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = A @ B  # or np.matmul(A, B)

# Transpose
A_T = A.T

# Inverse (if exists)
A_inv = np.linalg.inv(A)
```

**Interview Tips:**
- Always state dimensions when discussing matrices
- Matrix multiplication is NOT commutative: AB ≠ BA (generally)
- Know when inverse exists (non-singular, det ≠ 0)

---

## Question 4
- [ ] Done

**Explain the concept of a tensor in the context of machine learning.**

### Answer

**Definition:**  
A tensor is a multi-dimensional array that generalizes scalars (0-D), vectors (1-D), and matrices (2-D) to arbitrary dimensions. In ML, tensors are the fundamental data structure for representing and processing data.

**Core Concepts:**
- **Rank/Order**: number of dimensions (axes)
  - Rank 0: Scalar (e.g., 5)
  - Rank 1: Vector (e.g., [1, 2, 3])
  - Rank 2: Matrix (e.g., 3×3 grid)
  - Rank 3+: Higher-order tensor (e.g., RGB image, video)
- **Shape**: size along each dimension
- **Axes**: individual dimensions of the tensor

**Mathematical Formulation:**
$$\mathcal{T} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_n}$$

Example: RGB image tensor
$$\mathcal{I} \in \mathbb{R}^{H \times W \times 3}$$

**Intuition:**
| Tensor Rank | Example | Shape |
|-------------|---------|-------|
| 0 (Scalar) | Single pixel intensity | () |
| 1 (Vector) | Audio signal | (samples,) |
| 2 (Matrix) | Grayscale image | (H, W) |
| 3 | Color image | (H, W, 3) |
| 4 | Batch of images | (batch, H, W, 3) |
| 5 | Video batch | (batch, frames, H, W, 3) |

**ML Applications:**
- **Deep Learning**: all data flows as tensors through neural networks
- **CNNs**: 4-D tensors (batch, channels, height, width)
- **RNNs/Transformers**: 3-D tensors (batch, sequence, features)
- **Tensor decomposition**: dimensionality reduction, recommender systems

**Python Example:**
```python
import numpy as np
import torch

# NumPy tensors
scalar = np.array(5)                           # rank 0
vector = np.array([1, 2, 3])                   # rank 1, shape (3,)
matrix = np.array([[1, 2], [3, 4]])            # rank 2, shape (2, 2)
tensor_3d = np.random.randn(32, 224, 224)      # rank 3 (batch of grayscale)
tensor_4d = np.random.randn(32, 224, 224, 3)   # rank 4 (batch of RGB)

# PyTorch tensors (GPU-accelerated)
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape)   # torch.Size([2, 2])
print(x.dim())   # 2 (rank)

# Common operations
x_reshaped = x.view(4)           # reshape
x_transposed = x.permute(1, 0)   # swap axes
```

**Interview Tips:**
- "Tensor" in ML frameworks ≠ strict mathematical tensor (covariance properties)
- Know the difference: NumPy arrays vs PyTorch/TensorFlow tensors (GPU, autograd)
- Shape manipulation (reshape, squeeze, unsqueeze) is critical for debugging

---

## Question 5
- [ ] Done

**What are the properties of matrix multiplication?**

### Answer

**Definition:**  
Matrix multiplication combines two matrices A (m×n) and B (n×p) to produce C (m×p), where each element Cᵢⱼ is the dot product of row i of A and column j of B.

**Core Properties:**

| Property | Formula | Notes |
|----------|---------|-------|
| **Non-commutative** | AB ≠ BA | Order matters! |
| **Associative** | (AB)C = A(BC) | Grouping doesn't matter |
| **Distributive** | A(B + C) = AB + AC | Over addition |
| **Scalar associative** | α(AB) = (αA)B = A(αB) | Scalars factor out |
| **Transpose** | (AB)ᵀ = BᵀAᵀ | Reverse order rule |
| **Identity** | AI = IA = A | I is multiplicative identity |
| **Inverse** | (AB)⁻¹ = B⁻¹A⁻¹ | Reverse order for inverse |

**Mathematical Formulation:**
$$C = AB \quad \text{where} \quad C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

Dimensions: $(m \times n) \cdot (n \times p) = (m \times p)$

**Intuition:**
- Matrix multiplication = composition of linear transformations
- If A rotates and B scales, then AB rotates then scales (right to left)
- Think: "inner dimensions must match, outer dimensions give result"

**ML Applications:**
- **Forward pass**: output = Wx + b (weight matrix × input)
- **Batch processing**: multiply data matrix by weight matrix
- **Attention mechanism**: QKᵀ (query-key dot products)
- **Layer composition**: multiple transformations in one operation

**Python Example:**
```python
import numpy as np

A = np.array([[1, 2], [3, 4]])      # 2×2
B = np.array([[5, 6], [7, 8]])      # 2×2

# Matrix multiplication (3 equivalent ways)
C1 = A @ B                # Recommended (Python 3.5+)
C2 = np.matmul(A, B)      # Explicit function
C3 = np.dot(A, B)         # Works for 2-D arrays

# Verify non-commutativity
print(A @ B)
# [[19 22]
#  [43 50]]

print(B @ A)
# [[23 34]
#  [31 46]]

# Transpose property: (AB)^T = B^T @ A^T
assert np.allclose((A @ B).T, B.T @ A.T)

# Dimension mismatch example
X = np.random.randn(100, 50)   # 100 samples, 50 features
W = np.random.randn(50, 10)    # 50 inputs, 10 outputs
Y = X @ W                       # (100, 10) - valid
# W @ X would fail: (50, 10) @ (100, 50) - inner dims don't match
```

**Algorithm to Remember:**
1. Check dimensions: (m×n) × (n×p) → result is (m×p)
2. For each element Cᵢⱼ: dot product of row i of A with column j of B
3. Complexity: O(mnp) for naive multiplication

**Interview Tips:**
- Always verify dimensions before multiplying
- AB ≠ BA is the most common pitfall
- For element-wise multiplication, use `*` (Hadamard product), not `@`

---

## Question 6
- [ ] Done

**Explain the dot product of two vectors and its significance in machine learning.**

### Answer

**Definition:**  
The dot product (inner product) of two vectors is a scalar value computed by summing the element-wise products. It measures the similarity between vectors and projects one vector onto another.

**Core Concepts:**
- Also called: inner product, scalar product
- Requires vectors of same dimension
- Result is always a scalar
- Commutative: **a** · **b** = **b** · **a**

**Mathematical Formulation:**

Algebraic form:
$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1b_1 + a_2b_2 + \cdots + a_nb_n$$

Geometric form:
$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$$

where θ is the angle between the vectors.

**Intuition:**
- **Similarity measure**: larger dot product → more similar direction
- **Projection**: a · b = |a| × (projection of b onto a's direction)
- Sign interpretation:
  - Positive: vectors point in similar direction (θ < 90°)
  - Zero: vectors are orthogonal (θ = 90°)
  - Negative: vectors point in opposite directions (θ > 90°)

**ML Applications:**

| Application | Usage |
|-------------|-------|
| **Linear models** | prediction = **w** · **x** + b |
| **Cosine similarity** | sim = (a · b) / (‖a‖‖b‖) |
| **Attention mechanism** | query · key for relevance scores |
| **Neural networks** | weighted sum of inputs |
| **SVM** | kernel trick relies on dot products |
| **Recommendation** | user · item embeddings |

**Python Example:**
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product (3 ways)
dot1 = np.dot(a, b)         # 32
dot2 = a @ b                 # 32
dot3 = np.sum(a * b)         # 32 (manual)

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Check orthogonality
v1 = np.array([1, 0])
v2 = np.array([0, 1])
print(np.dot(v1, v2))  # 0 → orthogonal

# Linear model prediction
weights = np.array([0.5, -0.3, 0.8])
features = np.array([2.0, 1.0, 3.0])
bias = 0.1
prediction = np.dot(weights, features) + bias  # 0.5*2 + (-0.3)*1 + 0.8*3 + 0.1 = 3.2
```

**Interview Tips:**
- Dot product is the foundation of linear models: y = **w**ᵀ**x**
- Cosine similarity normalizes dot product to [-1, 1] range
- In deep learning, attention = softmax(QKᵀ) uses dot products
- Orthogonal vectors (dot product = 0) are linearly independent

---

## Question 7
- [ ] Done

**What is the cross product of vectors and when is it used?**

### Answer

**Definition:**  
The cross product is a binary operation on two vectors in 3D space that produces a third vector perpendicular to both input vectors. Unlike the dot product (scalar), the cross product returns a vector.

**Core Concepts:**
- Only defined in 3D (and 7D, mathematically)
- Result is orthogonal to both input vectors
- **Anti-commutative**: **a** × **b** = −(**b** × **a**)
- Magnitude = area of parallelogram formed by the two vectors

**Mathematical Formulation:**
$$\mathbf{a} \times \mathbf{b} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \end{vmatrix}$$

$$= (a_2b_3 - a_3b_2)\mathbf{i} - (a_1b_3 - a_3b_1)\mathbf{j} + (a_1b_2 - a_2b_1)\mathbf{k}$$

Magnitude:
$$\|\mathbf{a} \times \mathbf{b}\| = \|\mathbf{a}\| \|\mathbf{b}\| \sin\theta$$

**Intuition:**
- **Right-hand rule**: curl fingers from **a** to **b**, thumb points in direction of **a** × **b**
- If **a** and **b** are parallel → cross product = **0** (zero vector)
- Magnitude represents the area of the parallelogram spanned by **a** and **b**

**Properties:**
| Property | Formula |
|----------|---------|
| Anti-commutative | **a** × **b** = −**b** × **a** |
| Distributive | **a** × (**b** + **c**) = **a** × **b** + **a** × **c** |
| Scalar | (k**a**) × **b** = k(**a** × **b**) |
| Self-cross | **a** × **a** = **0** |

**ML/Applications:**
- **Computer graphics**: calculating surface normals
- **3D transformations**: rotation axes
- **Physics simulations**: torque, angular momentum
- **Robotics**: orientation and pose estimation
- **Point cloud processing**: normal estimation

**Python Example:**
```python
import numpy as np

a = np.array([1, 0, 0])
b = np.array([0, 1, 0])

# Cross product
c = np.cross(a, b)
print(c)  # [0, 0, 1] - perpendicular to both

# Verify orthogonality
print(np.dot(a, c))  # 0
print(np.dot(b, c))  # 0

# Area of parallelogram
v1 = np.array([3, 0, 0])
v2 = np.array([0, 4, 0])
area = np.linalg.norm(np.cross(v1, v2))  # 12

# Surface normal calculation (common in graphics)
def compute_normal(p1, p2, p3):
    """Compute normal vector for a triangle defined by 3 points."""
    edge1 = p2 - p1
    edge2 = p3 - p1
    normal = np.cross(edge1, edge2)
    return normal / np.linalg.norm(normal)  # unit normal
```

**Interview Tips:**
- Cross product is **not** commonly used in standard ML algorithms
- Important for 3D vision, graphics, and robotics ML applications
- Remember: dot product → scalar (similarity), cross product → vector (perpendicular)
- Zero cross product means vectors are parallel (linearly dependent in 3D)

---

## Question 8
- [ ] Done

**What is the determinant of a matrix and what information does it provide?**

### Answer

**Definition:**  
The determinant is a scalar value computed from a square matrix that encodes information about the matrix's invertibility, volume scaling factor, and linear transformation properties.

**Core Concepts:**
- Only defined for **square matrices** (n×n)
- det(A) = 0 → matrix is **singular** (non-invertible)
- det(A) ≠ 0 → matrix is **invertible**
- Represents signed volume scaling of the transformation

**Mathematical Formulation:**

For 2×2 matrix:
$$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

For 3×3 matrix (expansion along first row):
$$\det(A) = a_{11}(a_{22}a_{33} - a_{23}a_{32}) - a_{12}(a_{21}a_{33} - a_{23}a_{31}) + a_{13}(a_{21}a_{32} - a_{22}a_{31})$$

General formula (Leibniz):
$$\det(A) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^{n} a_{i,\sigma(i)}$$

**Key Properties:**
| Property | Formula |
|----------|---------|
| Product rule | det(AB) = det(A) · det(B) |
| Transpose | det(Aᵀ) = det(A) |
| Inverse | det(A⁻¹) = 1/det(A) |
| Scalar multiple | det(kA) = kⁿ det(A) for n×n matrix |
| Row swap | Changes sign of determinant |
| Triangular matrix | Product of diagonal elements |

**Intuition:**
- **2D**: Area scaling factor of parallelogram
- **3D**: Volume scaling factor of parallelepiped
- **Sign**: Positive = preserves orientation, Negative = flips orientation
- **Zero determinant**: Transformation collapses space (loses dimension)

**ML Applications:**
- **Multivariate Gaussian**: normalization constant uses det(Σ)
- **Covariance matrix**: det = 0 means features are linearly dependent
- **PCA**: eigenvalues relate to determinant
- **Numerical stability**: near-zero determinant indicates ill-conditioning

**Python Example:**
```python
import numpy as np

# 2x2 determinant
A = np.array([[4, 2], [3, 1]])
det_A = np.linalg.det(A)  # 4*1 - 2*3 = -2

# Check invertibility
if np.abs(det_A) > 1e-10:
    A_inv = np.linalg.inv(A)
    print("Matrix is invertible")
else:
    print("Matrix is singular")

# Singular matrix (det = 0)
B = np.array([[1, 2], [2, 4]])  # Row 2 = 2 * Row 1
det_B = np.linalg.det(B)  # ≈ 0

# Determinant of covariance matrix
X = np.random.randn(100, 3)
cov = np.cov(X.T)
det_cov = np.linalg.det(cov)  # Volume of uncertainty ellipsoid

# Multivariate Gaussian PDF (uses determinant)
def multivariate_gaussian_pdf(x, mean, cov):
    n = len(x)
    diff = x - mean
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    norm_const = 1 / (np.sqrt((2*np.pi)**n * det))
    exponent = -0.5 * diff.T @ inv @ diff
    return norm_const * np.exp(exponent)
```

**Algorithm: 2×2 Determinant**
1. Given matrix [[a, b], [c, d]]
2. Compute: ad - bc
3. Return result

**Interview Tips:**
- det = 0 is the key indicator of singularity
- For large matrices, use LU decomposition to compute determinant (O(n³))
- Determinant is sensitive to scaling: det(kA) = kⁿ·det(A)
- In ML, we often use log-determinant for numerical stability

---

## Question 9
- [ ] Done

**Can you explain what an eigenvector and eigenvalue are?**

### Answer

**Definition:**  
An eigenvector of a square matrix A is a non-zero vector **v** that, when transformed by A, only gets scaled (not rotated). The eigenvalue λ is the corresponding scaling factor. They satisfy: A**v** = λ**v**.

**Core Concepts:**
- Eigen = "own" or "characteristic" (German)
- Eigenvectors define invariant directions under transformation
- Eigenvalues indicate stretch/compression along those directions
- An n×n matrix has at most n eigenvalues (counting multiplicity)

**Mathematical Formulation:**

Eigenvalue equation:
$$A\mathbf{v} = \lambda\mathbf{v}$$

Characteristic equation (to find eigenvalues):
$$\det(A - \lambda I) = 0$$

Once λ is found, solve for **v**:
$$(A - \lambda I)\mathbf{v} = \mathbf{0}$$

**Intuition:**
- Most vectors change direction when transformed by A
- Eigenvectors are special: they only stretch/shrink, keeping their direction
- Eigenvalue > 1: stretch | Eigenvalue < 1: compress | Eigenvalue < 0: flip

**Visual Example (2D):**
```
Original vector v    →    After transformation Av
       ↑                         ↑↑ (same direction, scaled by λ)
```

**Properties:**
| Property | Description |
|----------|-------------|
| Sum of eigenvalues | = trace(A) |
| Product of eigenvalues | = det(A) |
| Symmetric matrix | Real eigenvalues, orthogonal eigenvectors |
| Distinct eigenvalues | Linearly independent eigenvectors |

**ML Applications:**
- **PCA**: Eigenvectors of covariance matrix = principal components
- **PageRank**: Dominant eigenvector of link matrix
- **Spectral clustering**: Eigenvectors of Laplacian matrix
- **Stability analysis**: Eigenvalues determine convergence in optimization
- **Graph Neural Networks**: Eigenvalues of adjacency/Laplacian matrices

**Python Example:**
```python
import numpy as np

A = np.array([[4, 2],
              [1, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)   # [5. 2.]
print("Eigenvectors:\n", eigenvectors)

# Verify: A @ v = lambda * v
v = eigenvectors[:, 0]  # First eigenvector
lam = eigenvalues[0]    # First eigenvalue
print(np.allclose(A @ v, lam * v))  # True

# PCA example: eigenvectors of covariance matrix
X = np.random.randn(100, 5)  # 100 samples, 5 features
X_centered = X - X.mean(axis=0)
cov_matrix = np.cov(X_centered.T)

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # eigh for symmetric
# Sort by eigenvalue (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# First principal component
pc1 = eigenvectors[:, 0]
```

**Algorithm: Finding Eigenvalues (Conceptual)**
1. Form characteristic polynomial: det(A - λI) = 0
2. Solve polynomial for λ values
3. For each λ, solve (A - λI)**v** = **0** for eigenvector **v**

**Interview Tips:**
- For symmetric matrices, use `np.linalg.eigh()` (faster, guaranteed real)
- Eigendecomposition: A = VΛV⁻¹ (V = eigenvector matrix, Λ = diagonal eigenvalues)
- Power iteration finds the dominant eigenvector (used in PageRank)
- PCA = eigendecomposition of covariance matrix (or SVD of data matrix)

---

## Question 10
- [ ] Done

**How is the trace of a matrix defined and what is its relevance?**

### Answer

**Definition:**  
The trace of a square matrix is the sum of its diagonal elements. It provides a simple scalar summary of a matrix and has important connections to eigenvalues and matrix operations.

**Core Concepts:**
- Only defined for square matrices (n×n)
- Invariant under cyclic permutations
- Sum of eigenvalues equals trace
- Computationally efficient: O(n)

**Mathematical Formulation:**
$$\text{tr}(A) = \sum_{i=1}^{n} a_{ii} = a_{11} + a_{22} + \cdots + a_{nn}$$

Eigenvalue relationship:
$$\text{tr}(A) = \sum_{i=1}^{n} \lambda_i$$

**Key Properties:**
| Property | Formula |
|----------|---------|
| Linearity | tr(A + B) = tr(A) + tr(B) |
| Scalar | tr(cA) = c · tr(A) |
| Transpose | tr(Aᵀ) = tr(A) |
| Cyclic | tr(ABC) = tr(CAB) = tr(BCA) |
| Product | tr(AB) = tr(BA) |
| Similarity | tr(P⁻¹AP) = tr(A) |

**Intuition:**
- Trace captures the "total scaling" along principal axes
- For covariance matrix: trace = total variance
- Invariant under basis change (similarity transformation)

**ML Applications:**
- **Total variance**: tr(Σ) = sum of variances of all features
- **Frobenius norm**: ||A||_F = √tr(AᵀA)
- **Nuclear norm**: sum of singular values (related to trace)
- **Loss functions**: trace appears in matrix-based losses
- **Regularization**: trace norm for low-rank matrix completion

**Python Example:**
```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Compute trace
trace_A = np.trace(A)  # 1 + 5 + 9 = 15

# Verify: trace = sum of eigenvalues
eigenvalues = np.linalg.eigvals(A)
print(np.isclose(np.sum(eigenvalues), trace_A))  # True (within numerical precision)

# Total variance from covariance matrix
X = np.random.randn(100, 5)
cov = np.cov(X.T)
total_variance = np.trace(cov)

# Frobenius norm using trace
frobenius_norm = np.sqrt(np.trace(A.T @ A))
# Equivalent to:
frobenius_norm_direct = np.linalg.norm(A, 'fro')

# Cyclic property verification
B = np.random.randn(3, 3)
C = np.random.randn(3, 3)
print(np.isclose(np.trace(A @ B @ C), np.trace(C @ A @ B)))  # True
```

**Interview Tips:**
- tr(A) = sum of eigenvalues; det(A) = product of eigenvalues
- Cyclic property is useful for simplifying gradient derivations
- trace(AᵀA) = sum of squared elements (related to Frobenius norm)
- In ML: trace often appears when deriving matrix calculus gradients

---

## Question 11
- [ ] Done

**What is a diagonal matrix and how is it used in linear algebra?**

### Answer

**Definition:**  
A diagonal matrix is a square matrix where all off-diagonal elements are zero. Only elements on the main diagonal (where row index equals column index) can be non-zero.

**Core Concepts:**
- Notation: D = diag(d₁, d₂, ..., dₙ)
- Extremely efficient for computation
- Eigenvalues = diagonal elements
- Represents independent scaling along each axis

**Mathematical Formulation:**
$$D = \begin{pmatrix} d_1 & 0 & \cdots & 0 \\ 0 & d_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & d_n \end{pmatrix}$$

where $D_{ij} = 0$ for $i \neq j$

**Key Properties:**
| Property | Formula/Description |
|----------|---------------------|
| Multiplication | D₁D₂ = diag(d₁·e₁, d₂·e₂, ...) |
| Inverse | D⁻¹ = diag(1/d₁, 1/d₂, ...) if all dᵢ ≠ 0 |
| Power | Dᵏ = diag(d₁ᵏ, d₂ᵏ, ...) |
| Determinant | det(D) = d₁ · d₂ · ... · dₙ |
| Trace | tr(D) = d₁ + d₂ + ... + dₙ |
| Eigenvalues | λᵢ = dᵢ |

**Intuition:**
- Diagonal matrix = independent scaling per dimension
- No "mixing" between dimensions
- diag(2, 3) scales x by 2 and y by 3 independently

**ML Applications:**
- **Eigendecomposition**: A = VDV⁻¹ (D is diagonal of eigenvalues)
- **SVD**: Σ matrix is diagonal (singular values)
- **Covariance**: Diagonal covariance = independent features
- **Regularization**: Ridge regression adds λI (scaled identity)
- **Batch normalization**: Scaling by diagonal matrix

**Python Example:**
```python
import numpy as np

# Create diagonal matrix
d = np.array([2, 3, 4])
D = np.diag(d)
# [[2, 0, 0],
#  [0, 3, 0],
#  [0, 0, 4]]

# Extract diagonal from matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
diagonal_elements = np.diag(A)  # [1, 5, 9]

# Efficient diagonal operations
# Instead of D @ x, use element-wise: d * x
x = np.array([1, 2, 3])
result = d * x  # [2, 6, 12] - much faster than matrix multiply

# Inverse of diagonal matrix
D_inv = np.diag(1 / d)  # diag(0.5, 0.333, 0.25)

# Power of diagonal matrix
D_squared = np.diag(d ** 2)  # diag(4, 9, 16)

# Diagonal covariance (independent features)
variances = np.array([1.0, 2.0, 0.5])
cov_diagonal = np.diag(variances)
# Sample from independent Gaussian
samples = np.random.randn(100, 3) * np.sqrt(variances)
```

**Interview Tips:**
- Diagonal matrices are computationally cheap: O(n) instead of O(n³) for inverse
- Diagonalization goal: transform A into D form for easier computation
- Identity matrix is a special diagonal matrix: I = diag(1, 1, ..., 1)
- Sparse diagonal matrices are memory efficient

---

## Question 12
- [ ] Done

**Explain the properties of an identity matrix.**

### Answer

**Definition:**  
The identity matrix (I) is a square diagonal matrix with 1s on the main diagonal and 0s elsewhere. It is the multiplicative identity for matrices, meaning AI = IA = A for any compatible matrix A.

**Core Concepts:**
- Notation: Iₙ for n×n identity matrix
- Acts like "1" in matrix multiplication
- Leaves vectors unchanged: I**v** = **v**
- Every vector is an eigenvector with eigenvalue 1

**Mathematical Formulation:**
$$I_n = \begin{pmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{pmatrix}$$

Formally: $I_{ij} = \delta_{ij}$ (Kronecker delta)

**Key Properties:**
| Property | Formula |
|----------|---------|
| Multiplicative identity | AI = IA = A |
| Self-inverse | I⁻¹ = I |
| Idempotent | Iⁿ = I |
| Determinant | det(I) = 1 |
| Trace | tr(Iₙ) = n |
| Eigenvalues | All eigenvalues = 1 |
| Transpose | Iᵀ = I |
| Orthogonal | IᵀI = IIᵀ = I |

**Intuition:**
- Identity transformation: does nothing
- Multiplying by I = "no transformation"
- Like multiplying a number by 1

**ML Applications:**
- **Regularization**: Ridge regression adds λI to prevent overfitting
  - (XᵀX + λI)⁻¹Xᵀy
- **Initialization**: Weight matrices sometimes initialized near identity
- **Residual connections**: Skip connections add identity mapping
- **Eigenvalue problems**: (A - λI)**v** = 0

**Python Example:**
```python
import numpy as np

# Create identity matrix
I = np.eye(3)  # 3x3 identity
# [[1., 0., 0.],
#  [0., 1., 0.],
#  [0., 0., 1.]]

# Also works:
I_alt = np.identity(3)

# Verify multiplicative identity
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(np.allclose(A @ I, A))  # True
print(np.allclose(I @ A, A))  # True

# Vector unchanged
v = np.array([1, 2, 3])
print(np.allclose(I @ v, v))  # True

# Ridge regression example
X = np.random.randn(100, 5)
y = np.random.randn(100)
lambda_reg = 0.1

# Normal equation with regularization
XtX = X.T @ X
XtX_regularized = XtX + lambda_reg * np.eye(5)
weights = np.linalg.inv(XtX_regularized) @ X.T @ y

# Residual connection (identity shortcut)
def residual_block(x, transform_fn):
    return x + transform_fn(x)  # Identity + learned transformation
```

**Interview Tips:**
- Identity matrix makes (XᵀX + λI) invertible even when XᵀX is singular
- In neural networks, residual connections approximate identity at initialization
- det(I) = 1 means identity preserves volume
- Know how to quickly create identity matrices in NumPy/PyTorch

---

## Question 13
- [ ] Done

**What is a unit vector and how do you find it?**

### Answer

**Definition:**  
A unit vector is a vector with magnitude (norm) equal to 1. It represents pure direction without magnitude. Any non-zero vector can be converted to a unit vector by dividing by its norm—a process called normalization.

**Core Concepts:**
- Magnitude: ||**û**|| = 1
- Preserves direction, removes magnitude
- Standard basis vectors (ê₁, ê₂, ...) are unit vectors
- Notation: **û** or **v̂** (hat symbol)

**Mathematical Formulation:**

Unit vector from **v**:
$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|} = \frac{\mathbf{v}}{\sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}}$$

Verification:
$$\|\hat{\mathbf{v}}\| = \left\|\frac{\mathbf{v}}{\|\mathbf{v}\|}\right\| = \frac{\|\mathbf{v}\|}{\|\mathbf{v}\|} = 1$$

**Standard Basis Unit Vectors:**
- 2D: **î** = [1, 0], **ĵ** = [0, 1]
- 3D: **î** = [1, 0, 0], **ĵ** = [0, 1, 0], **k̂** = [0, 0, 1]

**Intuition:**
- Unit vector answers "which direction?" without "how far?"
- All unit vectors lie on the unit sphere/circle
- Normalization = projecting onto the unit sphere

**ML Applications:**
- **Cosine similarity**: uses unit vectors for direction comparison
- **Word embeddings**: often normalized for similarity computation
- **Gradient normalization**: gradient clipping by norm
- **Batch normalization**: related to normalizing activations
- **Weight normalization**: decouple direction from magnitude

**Python Example:**
```python
import numpy as np

# Create a vector
v = np.array([3, 4])

# Compute magnitude (L2 norm)
magnitude = np.linalg.norm(v)  # 5.0

# Normalize to unit vector
unit_v = v / magnitude  # [0.6, 0.8]

# Verify magnitude is 1
print(np.linalg.norm(unit_v))  # 1.0

# Utility function for normalization
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v  # Handle zero vector
    return v / norm

# Batch normalization (normalize multiple vectors)
vectors = np.array([[3, 4], [1, 0], [5, 12]])
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
unit_vectors = vectors / norms

# Cosine similarity using unit vectors
def cosine_similarity(a, b):
    a_unit = a / np.linalg.norm(a)
    b_unit = b / np.linalg.norm(b)
    return np.dot(a_unit, b_unit)  # Dot product of unit vectors

# Standard basis vectors
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])
```

**Algorithm: Normalize a Vector**
1. Input: vector **v** = [v₁, v₂, ..., vₙ]
2. Compute norm: ||**v**|| = √(v₁² + v₂² + ... + vₙ²)
3. If norm = 0, return **v** (or handle as error)
4. Return **û** = **v** / ||**v**||

**Interview Tips:**
- Zero vector cannot be normalized (division by zero)
- Cosine similarity = dot product of unit vectors
- L2 normalization is most common, but L1 exists too
- Unit vectors are essential for directional comparisons

---

## Question 14
- [ ] Done

**Explain the concept of an orthogonal matrix.**

### Answer

**Definition:**  
An orthogonal matrix Q is a square matrix whose columns (and rows) are orthonormal vectors. It satisfies QᵀQ = QQᵀ = I, meaning its transpose equals its inverse: Qᵀ = Q⁻¹.

**Core Concepts:**
- Columns are mutually orthogonal unit vectors
- Rows are also mutually orthogonal unit vectors
- Preserves lengths (norms) and angles
- Represents rotations and reflections

**Mathematical Formulation:**

Definition:
$$Q^T Q = Q Q^T = I \quad \Rightarrow \quad Q^{-1} = Q^T$$

Column orthonormality:
$$\mathbf{q}_i^T \mathbf{q}_j = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

**Key Properties:**
| Property | Formula/Description |
|----------|---------------------|
| Inverse = Transpose | Q⁻¹ = Qᵀ |
| Determinant | det(Q) = ±1 |
| Length preservation | ||Q**x**|| = ||**x**|| |
| Angle preservation | ⟨Q**x**, Q**y**⟩ = ⟨**x**, **y**⟩ |
| Eigenvalues | |λ| = 1 (on unit circle) |
| Product | Q₁Q₂ is orthogonal |

**Types:**
- **det(Q) = +1**: Rotation (proper orthogonal)
- **det(Q) = -1**: Reflection (improper orthogonal)

**Intuition:**
- Orthogonal transformation = rigid motion (no stretching/shearing)
- Think: rotation matrix, reflection matrix
- Preserves the "shape" of data, just reorients it

**ML Applications:**
- **QR decomposition**: Q is orthogonal, R is upper triangular
- **PCA**: Eigenvectors of symmetric matrix form orthogonal basis
- **SVD**: U and V matrices are orthogonal
- **Orthogonal initialization**: Prevents gradient vanishing/exploding in RNNs
- **Whitening**: Transform data to have orthogonal (uncorrelated) features

**Python Example:**
```python
import numpy as np
from scipy.stats import ortho_group

# Rotation matrix (2D) - orthogonal
theta = np.pi / 4  # 45 degrees
Q_rotation = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

# Verify orthogonality
print(np.allclose(Q_rotation.T @ Q_rotation, np.eye(2)))  # True
print(np.allclose(Q_rotation @ Q_rotation.T, np.eye(2)))  # True

# Verify Q^T = Q^-1
print(np.allclose(Q_rotation.T, np.linalg.inv(Q_rotation)))  # True

# Determinant = ±1
print(np.linalg.det(Q_rotation))  # 1.0 (rotation)

# Length preservation
x = np.array([3, 4])
Qx = Q_rotation @ x
print(np.linalg.norm(x))   # 5.0
print(np.linalg.norm(Qx))  # 5.0

# Generate random orthogonal matrix
Q_random = ortho_group.rvs(dim=3)  # 3x3 random orthogonal

# QR decomposition
A = np.random.randn(4, 3)
Q, R = np.linalg.qr(A)
print(np.allclose(Q.T @ Q, np.eye(3)))  # True (Q is orthogonal)
```

**Interview Tips:**
- Orthogonal matrices are numerically stable (condition number = 1)
- Computing Q⁻¹ is cheap: just transpose!
- Orthogonal weight matrices help with gradient flow in deep networks
- In higher dimensions, orthogonal matrices generalize rotations

---

## Question 15
- [ ] Done

**What is the rank of a matrix and why is it important?**

### Answer

**Definition:**  
The rank of a matrix is the maximum number of linearly independent rows (or equivalently, columns). It represents the dimensionality of the vector space spanned by the matrix's rows or columns.

**Core Concepts:**
- rank(A) = number of linearly independent rows = number of linearly independent columns
- 0 ≤ rank(A) ≤ min(m, n) for m×n matrix
- Full rank: rank = min(m, n)
- Rank-deficient: rank < min(m, n)

**Mathematical Formulation:**
$$\text{rank}(A) = \dim(\text{column space}) = \dim(\text{row space})$$

Rank-Nullity Theorem:
$$\text{rank}(A) + \text{nullity}(A) = n \quad \text{(number of columns)}$$

**Rank and Matrix Properties:**
| Condition | Implication |
|-----------|-------------|
| rank(A) = n (full column rank) | Columns are linearly independent |
| rank(A) = m (full row rank) | Rows are linearly independent |
| rank(A) = min(m,n) | Full rank |
| rank(A) < min(m,n) | Rank deficient, redundant information |
| rank(A) = 0 | Zero matrix |

**Intuition:**
- Rank = "effective dimensionality" of the data
- Low rank = redundancy, correlation, or missing information
- Full rank square matrix → invertible
- Rank measures how much "information" the matrix contains

**ML Applications:**
- **Linear regression**: XᵀX invertible requires X to have full column rank
- **Multicollinearity**: Rank-deficient design matrix → unstable solutions
- **PCA**: Rank = number of non-zero principal components
- **Low-rank approximation**: Compression, noise reduction
- **Matrix completion**: Recommender systems assume low-rank structure

**Python Example:**
```python
import numpy as np

# Full rank matrix
A = np.array([[1, 2], [3, 4]])
rank_A = np.linalg.matrix_rank(A)  # 2 (full rank)

# Rank-deficient matrix (row 2 = 2 * row 1)
B = np.array([[1, 2], [2, 4]])
rank_B = np.linalg.matrix_rank(B)  # 1

# Rank of data matrix
X = np.random.randn(100, 10)  # 100 samples, 10 features
rank_X = np.linalg.matrix_rank(X)  # likely 10 (full column rank)

# Adding redundant feature reduces effective rank
X_redundant = np.column_stack([X, X[:, 0] * 2])  # duplicate feature
rank_redundant = np.linalg.matrix_rank(X_redundant)  # still 10

# Check if design matrix has full rank (for linear regression)
def check_full_rank(X):
    n_samples, n_features = X.shape
    rank = np.linalg.matrix_rank(X)
    if rank < n_features:
        print(f"Warning: Rank {rank} < {n_features} features. Multicollinearity present.")
    return rank == n_features

# Low-rank approximation via SVD
def low_rank_approx(A, k):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

A_approx = low_rank_approx(A, k=1)  # Rank-1 approximation
```

**Algorithm: Computing Rank (via Row Echelon Form)**
1. Apply Gaussian elimination to get row echelon form
2. Count non-zero rows
3. That count = rank

**Interview Tips:**
- rank(AB) ≤ min(rank(A), rank(B))
- rank(A) = rank(Aᵀ) = rank(AᵀA) = rank(AAᵀ)
- Rank = number of non-zero singular values
- For numerical stability, use SVD to compute rank (np.linalg.matrix_rank uses SVD internally)

---

## Question 16
- [ ] Done

**What is the method of Gaussian elimination?**

### Answer

**Definition:**  
Gaussian elimination is an algorithm for solving systems of linear equations by transforming the augmented matrix into row echelon form (REF) through elementary row operations. Back-substitution then yields the solution.

**Core Concepts:**
- Systematic approach to solve A**x** = **b**
- Uses three elementary row operations
- Produces upper triangular form
- Foundation for LU decomposition

**Elementary Row Operations:**
1. **Swap** two rows: Rᵢ ↔ Rⱼ
2. **Scale** a row: Rᵢ → c·Rᵢ (c ≠ 0)
3. **Add** multiple of one row to another: Rᵢ → Rᵢ + c·Rⱼ

**Mathematical Formulation:**

System: A**x** = **b**

Augmented matrix: [A | **b**]

Goal: Transform to row echelon form:
$$\begin{pmatrix} * & * & * & | & * \\ 0 & * & * & | & * \\ 0 & 0 & * & | & * \end{pmatrix}$$

**Algorithm Steps:**
```
1. Form augmented matrix [A | b]
2. For each column j from 1 to n:
   a. Find pivot: largest |a_ij| in column j (rows i ≥ j)
   b. Swap current row with pivot row (partial pivoting)
   c. For each row i below pivot:
      - Compute multiplier: m = a_ij / a_jj
      - Subtract: R_i = R_i - m * R_j
3. Back-substitution:
   - Solve for x_n from last equation
   - Substitute back to find x_{n-1}, ..., x_1
```

**Example:**
Solve:
```
2x + y - z = 8
-3x - y + 2z = -11
-2x + y + 2z = -3
```

$$\begin{pmatrix} 2 & 1 & -1 & | & 8 \\ -3 & -1 & 2 & | & -11 \\ -2 & 1 & 2 & | & -3 \end{pmatrix} \rightarrow \begin{pmatrix} 2 & 1 & -1 & | & 8 \\ 0 & 0.5 & 0.5 & | & 1 \\ 0 & 0 & 1 & | & -1 \end{pmatrix}$$

Back-substitute: z = -1, y = 3, x = 2

**Complexity:** O(n³) for n×n system

**Python Example:**

```python
import numpy as np

def gaussian_elimination(A, b):
    n = len(b)

    # augmented matrix
    Ab = np.c_[A.astype(float), b.astype(float)]

    # forward elimination with pivoting
    for i in range(n):
        # ---- pivot (row swap) ----
        max_row = i
        for k in range(i + 1, n):
            if abs(Ab[k, i]) > abs(Ab[max_row, i]):
                max_row = k

        Ab[i], Ab[max_row] = Ab[max_row].copy(), Ab[i].copy()

        # ---- eliminate below ----
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j] = Ab[j] - factor * Ab[i]

    # back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i + 1, n):
            s += Ab[i, j] * x[j]
        x[i] = (Ab[i, -1] - s) / Ab[i, i]

    return x
```

---

## Test

```python
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])

b = np.array([8, -11, -3])

x = gaussian_elimination(A, b)
print(x)                # [ 2.  3. -1.]
print(np.allclose(A @ x, b))  # True
```


**ML Applications:**
- **Linear regression**: Solving normal equations (XᵀX)w = Xᵀy
- **Basis for LU decomposition**: Used in many numerical libraries
- **Matrix inversion**: Apply to [A | I] to get [I | A⁻¹]

**Interview Tips:**
- Partial pivoting improves numerical stability
- Gaussian elimination is O(n³), same as matrix inversion
- For sparse matrices, specialized methods are more efficient
- In practice, use `np.linalg.solve()` which is optimized

---

## Question 17
- [ ] Done

**Explain the concept of linear dependence and independence.**

### Answer

**Definition:**  
Vectors are **linearly independent** if no vector can be written as a linear combination of the others. Equivalently, the only solution to c₁**v**₁ + c₂**v**₂ + ... + cₙ**v**ₙ = **0** is c₁ = c₂ = ... = cₙ = 0. Otherwise, they are **linearly dependent**.

**Core Concepts:**
- Independence: Each vector adds new "direction" to the span
- Dependence: At least one vector is redundant (expressible from others)
- n vectors in ℝⁿ are independent iff their matrix has full rank
- Maximum independent vectors in ℝⁿ = n

**Mathematical Formulation:**

Vectors {**v**₁, **v**₂, ..., **v**ₖ} are linearly independent iff:
$$\sum_{i=1}^{k} c_i \mathbf{v}_i = \mathbf{0} \implies c_1 = c_2 = \cdots = c_k = 0$$

Matrix test: Form matrix V = [**v**₁ | **v**₂ | ... | **v**ₖ]
- Independent: rank(V) = k
- Dependent: rank(V) < k

**Intuition:**
| Concept | 2D Visualization |
|---------|------------------|
| Independent | Vectors point in different directions (not collinear) |
| Dependent | Vectors lie on the same line (one is scalar multiple) |

**Examples:**
```
Independent: v₁ = [1, 0], v₂ = [0, 1]  → span entire ℝ²
Dependent:   v₁ = [1, 2], v₂ = [2, 4]  → v₂ = 2·v₁
```

**Tests for Independence:**
1. **Determinant test** (square matrix): det ≠ 0 → independent
2. **Rank test**: rank = number of vectors → independent
3. **Row reduction**: No zero rows after REF → independent

**ML Applications:**
- **Feature selection**: Remove linearly dependent features
- **Multicollinearity**: Dependent features cause unstable coefficients
- **Basis vectors**: Independent vectors form a basis
- **Dimensionality reduction**: Identify independent directions (PCA)
- **Regularization**: Needed when features are dependent

**Python Example:**
```python
import numpy as np

# Linearly independent vectors
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])
V_indep = np.column_stack([v1, v2, v3])
print(np.linalg.matrix_rank(V_indep))  # 3 (full rank = independent)

# Linearly dependent vectors
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])  # v2 = 2 * v1
v3 = np.array([1, 1, 1])
V_dep = np.column_stack([v1, v2, v3])
print(np.linalg.matrix_rank(V_dep))  # 2 (rank < 3 = dependent)

# Function to check linear independence
def are_independent(vectors):
    """Check if a list of vectors are linearly independent."""
    V = np.column_stack(vectors)
    return np.linalg.matrix_rank(V) == len(vectors)

print(are_independent([v1, v3]))  # True
print(are_independent([v1, v2]))  # False

# Detect multicollinearity in features
def check_multicollinearity(X, threshold=1e-10):
    """Check for multicollinearity in feature matrix X."""
    n_features = X.shape[1]
    rank = np.linalg.matrix_rank(X)
    if rank < n_features:
        print(f"Multicollinearity detected: rank={rank}, features={n_features}")
        return True
    return False

# Practical: correlation matrix check
X = np.random.randn(100, 5)
X = np.column_stack([X, X[:, 0] + X[:, 1]])  # Add dependent column
check_multicollinearity(X)  # Multicollinearity detected
```

**Algorithm: Test for Linear Independence**
1. Arrange vectors as columns of matrix V
2. Compute rank(V)
3. If rank = number of vectors → independent
4. If rank < number of vectors → dependent

**Interview Tips:**
- Any set containing the zero vector is automatically dependent
- In ℝⁿ, more than n vectors are always dependent
- Orthogonal vectors are always independent
- Condition number measures "how close" to dependent

---

## Question 18
- [ ] Done

**What is the meaning of the solution space of a system of linear equations?**

### Answer

**Definition:**  
The solution space (or solution set) is the set of all vectors **x** that satisfy the system A**x** = **b**. For homogeneous systems (A**x** = **0**), the solution space is called the **null space** or **kernel**, which forms a vector subspace.

**Core Concepts:**
- Homogeneous system: A**x** = **0** (always has at least trivial solution **x** = **0**)
- Non-homogeneous system: A**x** = **b** (may have no solution)
- Solution space structure depends on rank and dimensions

**Types of Solutions:**
| Condition | Number of Solutions |
|-----------|-------------------|
| rank(A) = rank([A\|b]) = n | Unique solution |
| rank(A) = rank([A\|b]) < n | Infinite solutions (n - rank free variables) |
| rank(A) < rank([A\|b]) | No solution (inconsistent) |

**Mathematical Formulation:**

For A**x** = **b**:
$$\text{Solution space} = \{\mathbf{x} \in \mathbb{R}^n : A\mathbf{x} = \mathbf{b}\}$$

General solution structure:
$$\mathbf{x} = \mathbf{x}_p + \mathbf{x}_h$$
- **x**ₚ = particular solution (any solution to A**x** = **b**)
- **x**ₕ = homogeneous solution (from null space of A)

**Null Space (Kernel):**
$$\text{Null}(A) = \ker(A) = \{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$$

Dimension of null space = n - rank(A) (by rank-nullity theorem)

**Intuition:**
- Unique solution: lines/planes intersect at exactly one point
- Infinite solutions: lines/planes overlap (coincide)
- No solution: lines/planes are parallel (never meet)

**Python Example:**
```python
import numpy as np
from scipy.linalg import null_space

# System with unique solution
A1 = np.array([[2, 1], [1, 3]])
b1 = np.array([4, 5])
x1 = np.linalg.solve(A1, b1)
print("Unique solution:", x1)  # [1. 2.]

# System with infinite solutions (underdetermined)
A2 = np.array([[1, 2, 3], [4, 5, 6]])  # 2 equations, 3 unknowns
b2 = np.array([1, 2])
# Find particular solution using least-squares
x_particular, residuals, rank, s = np.linalg.lstsq(A2, b2, rcond=None)

# Find null space (homogeneous solutions)
null = null_space(A2)
print("Null space basis:\n", null)
# General solution: x = x_particular + t * null (for any scalar t)

# System with no solution (inconsistent)
A3 = np.array([[1, 1], [1, 1]])  # Parallel lines
b3 = np.array([1, 2])  # Different intercepts
# np.linalg.solve(A3, b3) would raise LinAlgError

# Check consistency
def analyze_system(A, b):
    """Analyze the solution space of Ax = b."""
    n = A.shape[1]
    rank_A = np.linalg.matrix_rank(A)
    rank_Ab = np.linalg.matrix_rank(np.column_stack([A, b]))
    
    if rank_A < rank_Ab:
        return "No solution (inconsistent)"
    elif rank_A == n:
        return "Unique solution"
    else:
        free_vars = n - rank_A
        return f"Infinite solutions ({free_vars} free variables)"

print(analyze_system(A1, b1))  # Unique solution
print(analyze_system(A2, b2))  # Infinite solutions (1 free variable)
print(analyze_system(A3, b3))  # No solution (inconsistent)
```

**ML Applications:**
- **Linear regression**: Solving X**w** = **y** (often overdetermined → least squares)
- **Underdetermined systems**: Compressed sensing, sparse recovery
- **Null space**: Features in null space don't affect prediction
- **Regularization**: Handles infinite solution cases by adding constraints

**Interview Tips:**
- Overdetermined (more equations than unknowns): usually no exact solution → use least squares
- Underdetermined (more unknowns than equations): infinite solutions → use regularization
- Rank-nullity theorem: rank(A) + dim(null(A)) = n

---

## Question 19
- [ ] Done

**Describe the conditions for consistency in linear equations.**

### Answer

**Definition:**  
A system of linear equations A**x** = **b** is **consistent** if it has at least one solution (either unique or infinitely many). It is **inconsistent** if no solution exists.

**Core Concepts:**
- Consistency depends on the relationship between A and **b**
- **b** must lie in the column space of A
- Rank comparison determines consistency

**Consistency Conditions:**

**Rouché–Capelli Theorem:**
$$\text{System is consistent} \iff \text{rank}(A) = \text{rank}([A | \mathbf{b}])$$

| Condition | Result |
|-----------|--------|
| rank(A) = rank([A\|b]) | Consistent |
| rank(A) < rank([A\|b]) | Inconsistent |

**Types of Consistent Systems:**
| Condition | Solution Type |
|-----------|---------------|
| rank(A) = rank([A\|b]) = n | Unique solution |
| rank(A) = rank([A\|b]) < n | Infinite solutions |

**Mathematical Formulation:**

Geometric interpretation:
$$A\mathbf{x} = \mathbf{b} \text{ is consistent} \iff \mathbf{b} \in \text{Col}(A)$$

where Col(A) is the column space of A.

**Intuition:**
- **Consistent**: The target **b** can be "reached" by some combination of A's columns
- **Inconsistent**: **b** lies outside what A's columns can span
- 2D example: Two parallel lines (inconsistent) vs two intersecting lines (consistent)

**Visual Examples:**
```
Consistent (unique):     Consistent (infinite):    Inconsistent:
    \  /                      ___                     ___
     \/                      ___                     ___
  One point              Overlapping lines        Parallel lines
```

**Python Example:**
```python
import numpy as np

def check_consistency(A, b):
    """
    Check if the system Ax = b is consistent and determine solution type.
    """
    n_cols = A.shape[1]
    
    # Compute ranks
    rank_A = np.linalg.matrix_rank(A)
    Ab = np.column_stack([A, b])
    rank_Ab = np.linalg.matrix_rank(Ab)
    
    if rank_A < rank_Ab:
        return "Inconsistent (no solution)"
    elif rank_A == rank_Ab == n_cols:
        return "Consistent (unique solution)"
    else:
        free_vars = n_cols - rank_A
        return f"Consistent (infinite solutions, {free_vars} free variables)"

# Example 1: Unique solution
A1 = np.array([[1, 2], [3, 4]])
b1 = np.array([5, 6])
print(check_consistency(A1, b1))  # Consistent (unique solution)

# Example 2: Infinite solutions
A2 = np.array([[1, 2], [2, 4]])  # Row 2 = 2 * Row 1
b2 = np.array([3, 6])            # b2 = 2 * b1 (consistent)
print(check_consistency(A2, b2))  # Consistent (infinite solutions)

# Example 3: Inconsistent
A3 = np.array([[1, 2], [2, 4]])  # Row 2 = 2 * Row 1
b3 = np.array([3, 7])            # b3 ≠ 2 * b1 (inconsistent!)
print(check_consistency(A3, b3))  # Inconsistent (no solution)

# Alternative: Check if b is in column space
def is_in_column_space(A, b):
    """Check if b lies in the column space of A."""
    rank_A = np.linalg.matrix_rank(A)
    rank_Ab = np.linalg.matrix_rank(np.column_stack([A, b]))
    return rank_A == rank_Ab

# Practical: Least squares handles inconsistent systems
x_lstsq, residuals, _, _ = np.linalg.lstsq(A3, b3, rcond=None)
# x_lstsq is the "best" solution minimizing ||Ax - b||²
```

**ML Applications:**
- **Overdetermined systems**: Linear regression (m equations, n unknowns, m > n)
  - Usually inconsistent → solve via least squares
- **Data fitting**: Finding best-fit line/plane when exact fit impossible
- **Feature engineering**: Ensuring design matrix leads to consistent system

**Interview Tips:**
- Rank([A|b]) > Rank(A) means **b** has a component outside Col(A)
- In ML, we rarely have exactly consistent systems (noise in data)
- Least squares finds the closest solution for inconsistent systems
- Always check for multicollinearity when rank(A) < n

---

## Question 20
- [ ] Done

**Explain the LU decomposition of a matrix.**

### Answer

**Definition:**  
LU decomposition factors a square matrix A into the product of a lower triangular matrix L and an upper triangular matrix U, such that A = LU. This enables efficient solving of linear systems.

**Core Concepts:**
- L = Lower triangular (zeros above diagonal, 1s on diagonal typically)
- U = Upper triangular (zeros below diagonal)
- Based on Gaussian elimination
- Enables solving A**x** = **b** in two easy steps

**Mathematical Formulation:**
$$A = LU$$

$$\begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 \\ l_{21} & 1 & 0 \\ l_{31} & l_{32} & 1 \end{pmatrix} \begin{pmatrix} u_{11} & u_{12} & u_{13} \\ 0 & u_{22} & u_{23} \\ 0 & 0 & u_{33} \end{pmatrix}$$

**With Pivoting (PLU):**
$$PA = LU$$
where P is a permutation matrix for numerical stability.

**Solving A**x** = **b** using LU:**
1. Decompose: A = LU
2. Solve L**y** = **b** (forward substitution)
3. Solve U**x** = **y** (back substitution)

**Algorithm Steps:**
```
For k = 1 to n:
    U[k,k:n] = A[k,k:n]  # First row of remaining submatrix
    L[k:n,k] = A[k:n,k] / U[k,k]  # First column divided by pivot
    A[k+1:n,k+1:n] -= L[k+1:n,k] * U[k,k+1:n]  # Update remaining
```

**Complexity:**
- Decomposition: O(n³/3)
- Each solve: O(n²)
- Advantage: Decompose once, solve for multiple **b** vectors efficiently

**Why LU is Useful:**
| Scenario | Benefit |
|----------|---------|
| Multiple systems with same A | Decompose once, solve many |
| Computing determinant | det(A) = det(L)·det(U) = ∏u_ii |
| Matrix inversion | Solve A·col_i = e_i for each column |

**Python Example:**
```python
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve

A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]], dtype=float)
b = np.array([4, 10, 24], dtype=float)

# Method 1: Full LU decomposition (P, L, U)
P, L, U = lu(A)
print("L:\n", L)
print("U:\n", U)
print("P @ L @ U:\n", P @ L @ U)  # Should equal A

# Verify: PA = LU
print(np.allclose(P @ A, L @ U))  # True

# Method 2: Compact LU for solving systems
lu_factored, piv = lu_factor(A)

# Solve multiple systems efficiently
b1 = np.array([4, 10, 24])
b2 = np.array([1, 2, 3])
x1 = lu_solve((lu_factored, piv), b1)
x2 = lu_solve((lu_factored, piv), b2)

# Verify solution
print(np.allclose(A @ x1, b1))  # True

# Determinant using LU
det_A = np.prod(np.diag(lu_factored)) * (-1)**(np.sum(piv != np.arange(len(piv))))
print("det(A):", det_A)

# Manual forward/back substitution
def solve_lower(L, b):
    """Forward substitution for Ly = b"""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y

def solve_upper(U, y):
    """Back substitution for Ux = y"""
    n = len(y)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x
```

**ML Applications:**
- **Linear regression**: Solving normal equations efficiently
- **Gaussian processes**: Efficient covariance matrix operations
- **Determinant computation**: For multivariate Gaussian likelihood
- **Preconditioning**: Accelerating iterative solvers

**Interview Tips:**
- LU fails if pivot is zero → use PLU (pivoted LU)
- Cholesky is preferred for symmetric positive definite matrices
- LU is 2x faster than computing A⁻¹ explicitly
- For sparse matrices, sparse LU variants exist (e.g., SuperLU)

---

## Question 21
- [ ] Done

**What are singular or ill-conditioned matrices?**

### Answer

**Definition:**  
A square matrix is **singular** (or degenerate) if it does not have an inverse. Equivalently, its determinant is zero, and it maps some non-zero vectors to the zero vector.

**Core Concepts:**
- Singular ⟺ det(A) = 0
- Singular ⟺ A is not invertible
- Singular ⟺ rank(A) < n (rank-deficient)
- Singular ⟺ A has eigenvalue λ = 0

**Equivalent Conditions:**
| Condition | Singular | Non-singular |
|-----------|----------|--------------|
| Determinant | det(A) = 0 | det(A) ≠ 0 |
| Rank | rank(A) < n | rank(A) = n |
| Null space | Non-trivial (≠ {0}) | Only {0} |
| Eigenvalues | Has λ = 0 | All λ ≠ 0 |
| Columns/Rows | Linearly dependent | Linearly independent |
| System A**x**=**b** | May have 0 or ∞ solutions | Unique solution |

**Mathematical Formulation:**

A matrix A is singular if:
$$\det(A) = 0$$

Equivalently:
$$\exists \mathbf{v} \neq \mathbf{0} \text{ such that } A\mathbf{v} = \mathbf{0}$$

**Intuition:**
- Singular matrix "collapses" space (loses at least one dimension)
- Information is lost—transformation cannot be reversed
- Think: projecting 3D onto a plane (can't recover the original 3D point)

**Causes of Singularity:**
1. Duplicate rows/columns (linear dependence)
2. Row/column of zeros
3. Rows/columns are scalar multiples
4. Multicollinearity in data

**Python Example:**
```python
import numpy as np

# Non-singular matrix
A = np.array([[1, 2], [3, 4]])
print("det(A):", np.linalg.det(A))  # -2.0 (non-zero)
print("A is invertible:", np.linalg.det(A) != 0)  # True
A_inv = np.linalg.inv(A)  # Works fine

# Singular matrix
B = np.array([[1, 2], [2, 4]])  # Row 2 = 2 * Row 1
print("det(B):", np.linalg.det(B))  # 0.0 (or very small due to floating point)
print("rank(B):", np.linalg.matrix_rank(B))  # 1 (< 2)
# np.linalg.inv(B) raises LinAlgError: Singular matrix

# Near-singular (ill-conditioned) matrix
C = np.array([[1, 2], [1.0001, 2.0002]])
print("det(C):", np.linalg.det(C))  # Very small (~1e-4)
print("Condition number:", np.linalg.cond(C))  # Very large

# Check singularity safely
def is_singular(A, tol=1e-10):
    """Check if matrix is singular."""
    return np.abs(np.linalg.det(A)) < tol

# Better method: use condition number
def is_ill_conditioned(A, threshold=1e10):
    """Check if matrix is numerically ill-conditioned."""
    return np.linalg.cond(A) > threshold

# Handling singular matrices in ML
# Option 1: Add regularization
X = np.random.randn(100, 10)
X = np.column_stack([X, X[:, 0]])  # Add duplicate column (singular XᵀX)

lambda_reg = 0.01
XtX = X.T @ X
XtX_reg = XtX + lambda_reg * np.eye(XtX.shape[0])  # Now invertible!

# Option 2: Use pseudoinverse
X_pinv = np.linalg.pinv(X)  # Works even when X is rank-deficient
```

**ML Applications:**
- **Multicollinearity**: Correlated features cause singular XᵀX
- **Ridge regression**: Adds λI to prevent singularity
- **Pseudoinverse**: Moore-Penrose inverse handles singular matrices
- **Numerical stability**: Near-singular matrices cause large errors

**Interview Tips:**
- Singular matrices are dangerous: small errors get amplified
- Condition number quantifies "how close" to singular (higher = worse)
- Regularization is the standard fix for singular/near-singular systems
- Use pseudoinverse (`np.linalg.pinv`) instead of inverse when unsure

---

## Question 22
- [ ] Done

**What is the Singular Value Decomposition (SVD) and its applications in machine learning?**

### Answer

**Definition:**  
SVD decomposes any m×n matrix A into three matrices: A = UΣVᵀ, where U and V are orthogonal matrices and Σ is a diagonal matrix of singular values. It is the most general and powerful matrix decomposition.

**Core Concepts:**
- Works for **any** matrix (rectangular, singular, complex)
- U ∈ ℝᵐˣᵐ: Left singular vectors (orthogonal)
- Σ ∈ ℝᵐˣⁿ: Singular values on diagonal (non-negative, sorted σ₁ ≥ σ₂ ≥ ... ≥ 0)
- V ∈ ℝⁿˣⁿ: Right singular vectors (orthogonal)

**Mathematical Formulation:**
$$A = U \Sigma V^T$$

$$A = \begin{pmatrix} | & | & \\ \mathbf{u}_1 & \mathbf{u}_2 & \cdots \\ | & | & \end{pmatrix} \begin{pmatrix} \sigma_1 & & \\ & \sigma_2 & \\ & & \ddots \end{pmatrix} \begin{pmatrix} - & \mathbf{v}_1^T & - \\ - & \mathbf{v}_2^T & - \\ & \vdots & \end{pmatrix}$$

As a sum of rank-1 matrices:
$$A = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

**Relationships:**
- Columns of U: eigenvectors of AAᵀ
- Columns of V: eigenvectors of AᵀA
- Singular values: σᵢ = √λᵢ (square root of eigenvalues)

**Key Properties:**
| Property | Formula |
|----------|---------|
| Rank | Number of non-zero singular values |
| Frobenius norm | ‖A‖_F = √(Σσᵢ²) |
| 2-norm | ‖A‖₂ = σ₁ (largest singular value) |
| Condition number | κ(A) = σ₁/σᵣ |
| Pseudoinverse | A⁺ = VΣ⁺Uᵀ |

**Low-Rank Approximation (Eckart-Young Theorem):**
Best rank-k approximation:
$$A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T = U_k \Sigma_k V_k^T$$

**Python Example:**
```python
import numpy as np

# Create a matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

# Full SVD
U, s, Vt = np.linalg.svd(A, full_matrices=True)
print("U shape:", U.shape)   # (4, 4)
print("s:", s)                # Singular values
print("Vt shape:", Vt.shape) # (3, 3)

# Compact SVD (more efficient)
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print("U shape:", U.shape)   # (4, 3)

# Reconstruct A
Sigma = np.diag(s)
A_reconstructed = U @ Sigma @ Vt
print(np.allclose(A, A_reconstructed))  # True

# Low-rank approximation (rank k)
def low_rank_approx(A, k):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

A_rank1 = low_rank_approx(A, k=1)
A_rank2 = low_rank_approx(A, k=2)

# Compute approximation error
error = np.linalg.norm(A - A_rank2, 'fro')
print(f"Rank-2 approximation error: {error}")

# Image compression example
def compress_image(image, k):
    """Compress grayscale image using SVD."""
    U, s, Vt = np.linalg.svd(image, full_matrices=False)
    compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    return compressed

# Pseudoinverse via SVD
def pseudo_inverse(A):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_inv = np.array([1/si if si > 1e-10 else 0 for si in s])
    return Vt.T @ np.diag(s_inv) @ U.T
```

**ML Applications:**
| Application | How SVD is Used |
|-------------|-----------------|
| **PCA** | SVD of centered data gives principal components |
| **LSA/LSI** | Dimensionality reduction for text |
| **Recommender Systems** | Matrix factorization (Netflix prize) |
| **Image Compression** | Keep top-k singular values |
| **Noise Reduction** | Remove small singular values |
| **Pseudoinverse** | Solve least squares via SVD |
| **Data Whitening** | Decorrelate features |

**Algorithm: Computing SVD (Conceptual)**
1. Compute AᵀA and AAᵀ
2. Find eigenvectors of AᵀA → V
3. Find eigenvectors of AAᵀ → U
4. Singular values σᵢ = √(eigenvalue of AᵀA)

**Interview Tips:**
- SVD always exists; eigendecomposition doesn't (need square matrix)
- For symmetric positive definite: SVD = eigendecomposition
- Truncated SVD is core of PCA, LSA, and matrix completion
- NumPy's `svd` returns Vᵀ, not V (common source of bugs!)

---

## Question 23
- [ ] Done

**Explain the concept of matrix factorization.**

### Answer

**Definition:**  
Matrix factorization (decomposition) expresses a matrix as a product of two or more matrices with special properties. It simplifies complex operations, reveals structure, and enables efficient algorithms.

**Core Concepts:**
- Break complex matrix into simpler components
- Each factorization reveals different properties
- Enables efficient computation, compression, and analysis
- Foundation of many ML algorithms

**Common Matrix Factorizations:**

| Factorization | Form | Requirements | Use Cases |
|---------------|------|--------------|-----------|
| **LU** | A = LU | Square | Solving linear systems |
| **Cholesky** | A = LLᵀ | Symmetric positive definite | Gaussian sampling, optimization |
| **QR** | A = QR | Any | Least squares, eigenvalues |
| **Eigendecomposition** | A = VΛV⁻¹ | Square, diagonalizable | PCA (symmetric), stability |
| **SVD** | A = UΣVᵀ | Any | PCA, compression, recommenders |
| **NMF** | A ≈ WH | Non-negative | Topic modeling, image parts |

**Mathematical Formulation:**

**Eigendecomposition** (square, diagonalizable):
$$A = V \Lambda V^{-1}$$

**Cholesky** (symmetric positive definite):
$$A = L L^T$$

**QR Decomposition** (any matrix):
$$A = QR \quad (Q \text{ orthogonal}, R \text{ upper triangular})$$

**Non-negative Matrix Factorization**:
$$A \approx WH \quad (W, H \geq 0)$$

**Intuition:**
- **LU/Cholesky**: Efficient system solving (triangular = easy)
- **QR**: Numerical stability for least squares
- **SVD**: Reveals rank, enables compression
- **NMF**: Parts-based representation (additivity, interpretability)

**Python Example:**
```python
import numpy as np
from scipy.linalg import lu, cholesky, qr
from sklearn.decomposition import NMF

A = np.array([[4, 2], [2, 3]], dtype=float)

# LU Decomposition
P, L, U = lu(A)
print("LU: P @ L @ U =\n", P @ L @ U)

# Cholesky (for positive definite)
L_chol = cholesky(A, lower=True)
print("Cholesky: L @ L.T =\n", L_chol @ L_chol.T)

# QR Decomposition
Q, R = qr(A)
print("QR: Q @ R =\n", Q @ R)

# SVD
U, s, Vt = np.linalg.svd(A)
print("SVD: U @ diag(s) @ Vt =\n", U @ np.diag(s) @ Vt)

# Eigendecomposition (symmetric)
eigenvalues, V = np.linalg.eigh(A)
Lambda = np.diag(eigenvalues)
print("Eigen: V @ Λ @ V.T =\n", V @ Lambda @ V.T)

# Non-negative Matrix Factorization
V_nmf = np.abs(np.random.randn(100, 50))  # Non-negative data
nmf = NMF(n_components=10, random_state=42)
W = nmf.fit_transform(V_nmf)
H = nmf.components_
print("NMF approximation error:", np.linalg.norm(V_nmf - W @ H, 'fro'))
```

**ML Applications:**
| Factorization | ML Application |
|---------------|----------------|
| **SVD** | PCA, LSA, image compression |
| **NMF** | Topic modeling, face parts, audio source separation |
| **Cholesky** | Gaussian process sampling, covariance operations |
| **QR** | Numerically stable linear regression |
| **Matrix Factorization** | Collaborative filtering (Netflix, Spotify) |

**Recommender Systems Example:**
```python
# User-Item rating matrix R ≈ U @ V.T
# U: user latent factors (n_users × k)
# V: item latent factors (n_items × k)

from sklearn.decomposition import TruncatedSVD

# Ratings matrix (users × items)
R = np.array([[5, 3, 0, 1],
              [4, 0, 0, 1],
              [1, 1, 0, 5],
              [0, 0, 5, 4]])

svd = TruncatedSVD(n_components=2)
U = svd.fit_transform(R)
V = svd.components_.T
R_approx = U @ V.T  # Predicted ratings (including for 0s)
```

**Interview Tips:**
- Choose factorization based on matrix properties and goal
- SVD is most general; Cholesky is fastest for SPD matrices
- NMF produces interpretable, non-negative factors
- Matrix factorization is the backbone of collaborative filtering

---

## Question 24
- [ ] Done

**What is a linear transformation in linear algebra?**

### Answer

**Definition:**  
A linear transformation (linear map) is a function T: V → W between vector spaces that preserves vector addition and scalar multiplication. Every linear transformation can be represented by a matrix.

**Core Concepts:**
- Maps vectors from one space to another
- Preserves linear structure (lines remain lines, origin stays fixed)
- Fully determined by its action on basis vectors
- Matrix multiplication IS a linear transformation

**Defining Properties:**
For all vectors **u**, **v** and scalar c:
$$T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v}) \quad \text{(Additivity)}$$
$$T(c\mathbf{u}) = c \cdot T(\mathbf{u}) \quad \text{(Homogeneity)}$$

Combined:
$$T(\alpha\mathbf{u} + \beta\mathbf{v}) = \alpha T(\mathbf{u}) + \beta T(\mathbf{v})$$

**Matrix Representation:**
If T: ℝⁿ → ℝᵐ is linear, there exists a unique m×n matrix A such that:
$$T(\mathbf{x}) = A\mathbf{x}$$

**Common Linear Transformations:**
| Transformation | Matrix (2D) | Effect |
|----------------|-------------|--------|
| **Identity** | [[1,0],[0,1]] | No change |
| **Scaling** | [[s,0],[0,s]] | Uniform scaling by s |
| **Rotation** (θ) | [[cos θ, -sin θ],[sin θ, cos θ]] | Rotate by angle θ |
| **Reflection** (x-axis) | [[1,0],[0,-1]] | Flip over x-axis |
| **Shear** | [[1,k],[0,1]] | Horizontal shear |
| **Projection** (x-axis) | [[1,0],[0,0]] | Project onto x-axis |

**Non-Linear Examples (NOT linear transformations):**
- Translation: T(**x**) = **x** + **b** (doesn't preserve origin)
- Polynomial: T(x) = x²
- Normalization: T(**x**) = **x**/‖**x**‖

**Python Example:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Define transformation matrices
def rotation_matrix(theta):
    """2D rotation by theta radians."""
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

def scaling_matrix(sx, sy):
    """2D scaling."""
    return np.array([[sx, 0], [0, sy]])

def shear_matrix(kx, ky):
    """2D shear."""
    return np.array([[1, kx], [ky, 1]])

def reflection_x():
    """Reflect over x-axis."""
    return np.array([[1, 0], [0, -1]])

# Apply transformation
v = np.array([1, 0])

# Rotation by 90 degrees
R = rotation_matrix(np.pi / 2)
v_rotated = R @ v  # [0, 1]

# Composition of transformations
# First scale, then rotate
S = scaling_matrix(2, 0.5)
T_composed = R @ S  # Note: applied right to left

# Verify linearity
u = np.array([1, 2])
w = np.array([3, 1])
alpha, beta = 2, 3

# T(αu + βv) = αT(u) + βT(v)
lhs = R @ (alpha * u + beta * w)
rhs = alpha * (R @ u) + beta * (R @ w)
print(np.allclose(lhs, rhs))  # True

# Neural network layer is a linear transformation (before activation)
def linear_layer(X, W, b):
    """X @ W + b is affine, X @ W alone is linear."""
    return X @ W  # Linear transformation

# Visualize transformation
def visualize_transform(A, title="Linear Transformation"):
    # Original unit square
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
    # Transform
    transformed = A @ square
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(square[0], square[1], 'b-', linewidth=2)
    plt.title("Original")
    plt.axis('equal')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(transformed[0], transformed[1], 'r-', linewidth=2)
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.show()
```

**ML Applications:**
- **Neural network layers**: y = Wx (before activation)
- **PCA**: Projects data onto principal components
- **Feature extraction**: Transforming input space
- **Dimensionality reduction**: Mapping to lower dimensions
- **Attention**: Query-key-value projections

**Interview Tips:**
- Matrix = linear transformation (one-to-one correspondence)
- Composition of linear maps = matrix multiplication
- Affine transformation = linear + translation (not strictly linear)
- Neural networks alternate linear transforms with non-linear activations

---

## Question 25
- [ ] Done

**Describe the kernel and image of a linear transformation.**

### Answer

**Definition:**  
For a linear transformation T: V → W (or matrix A):
- **Kernel (Null Space)**: The set of all vectors that map to zero: ker(T) = {**v** | T(**v**) = **0**}
- **Image (Range/Column Space)**: The set of all possible outputs: im(T) = {T(**v**) | **v** ∈ V}

**Core Concepts:**
- Kernel measures what information is "lost" by the transformation
- Image measures what outputs are "reachable"
- Both are vector subspaces
- Connected by the rank-nullity theorem

**Mathematical Formulation:**

**Kernel (Null Space):**
$$\ker(A) = \text{Null}(A) = \{\mathbf{x} \in \mathbb{R}^n : A\mathbf{x} = \mathbf{0}\}$$

**Image (Column Space):**
$$\text{im}(A) = \text{Col}(A) = \{A\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\} = \text{span of columns of } A$$

**Rank-Nullity Theorem:**
$$\dim(\ker(A)) + \dim(\text{im}(A)) = n$$
$$\text{nullity}(A) + \text{rank}(A) = n$$

**Intuition:**
| Concept | Meaning |
|---------|---------|
| **Kernel** | Vectors "collapsed" to zero (lost information) |
| **Image** | All possible output vectors (reachable space) |
| **dim(kernel) = 0** | Injective (one-to-one), no information loss |
| **dim(image) = m** | Surjective (onto), all outputs reachable |

**Visual (2D → 2D):**
```
Kernel = {0} only     vs.     Kernel = a line
     ↓                              ↓
Full rank (invertible)    Rank-deficient (line collapses to 0)
```

**Python Example:**
```python
import numpy as np
from scipy.linalg import null_space

# Matrix transformation
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# 1. KERNEL (Null Space)
# Find vectors x such that Ax = 0
kernel = null_space(A)
print("Kernel basis:\n", kernel)
print("Kernel dimension (nullity):", kernel.shape[1])

# Verify: A @ kernel_vector ≈ 0
if kernel.size > 0:
    print("Verify Ax = 0:", np.allclose(A @ kernel[:, 0], 0))

# 2. IMAGE (Column Space)
# The column space is spanned by the columns of A
# Its dimension = rank(A)
rank = np.linalg.matrix_rank(A)
print("Image dimension (rank):", rank)

# Verify rank-nullity theorem
n = A.shape[1]  # Number of columns
nullity = kernel.shape[1]
print(f"Rank ({rank}) + Nullity ({nullity}) = {rank + nullity} = n ({n})")

# Check if vector b is in the image (column space)
def is_in_image(A, b):
    """Check if b is in the column space of A."""
    rank_A = np.linalg.matrix_rank(A)
    rank_Ab = np.linalg.matrix_rank(np.column_stack([A, b]))
    return rank_A == rank_Ab

b1 = np.array([1, 2, 3])  # In column space
b2 = np.array([1, 0, 0])  # May or may not be
print("b1 in image:", is_in_image(A, b1))

# Projection onto column space
def project_onto_image(A, b):
    """Project b onto column space of A."""
    # A @ (A^+ @ b) gives projection
    return A @ np.linalg.lstsq(A, b, rcond=None)[0]

# Full rank example (trivial kernel)
B = np.array([[1, 0], [0, 1], [1, 1]])  # 3x2, rank 2
kernel_B = null_space(B)
print("Kernel of full-rank B:", kernel_B.shape)  # (2, 0) - empty

# Practical: Features in kernel don't affect prediction
# If w is in kernel of X^T, then X @ w = 0
# Adding any kernel vector to solution doesn't change predictions
```

**ML Applications:**
| Concept | ML Relevance |
|---------|--------------|
| **Kernel = {0}** | All features are useful (no redundancy) |
| **Non-trivial kernel** | Redundant features exist |
| **Image** | Representable outputs of model |
| **Rank-deficient** | Multicollinearity, need regularization |

**Interview Tips:**
- ker(A) = {**0**} iff A has full column rank (injective)
- im(A) = ℝᵐ iff A has full row rank (surjective)
- For square matrix: ker(A) = {**0**} ⟺ A is invertible
- Null space vectors represent "directions of no change" in predictions

---

## Question 26
- [ ] Done

**How does change of basis affect matrix representation of linear transformations?**

### Answer

**Definition:**  
A change of basis transforms the coordinate representation of vectors from one basis to another. It allows the same vector to be expressed in different coordinate systems, often simplifying computations or revealing structure.

**Core Concepts:**
- A basis is a set of linearly independent vectors that span the space
- Same vector, different coordinates depending on basis
- Change of basis matrix converts coordinates between bases
- Diagonal/simpler forms often exist in the "right" basis

**Mathematical Formulation:**

Given two bases B = {**b**₁, ..., **b**ₙ} and B' = {**b'**₁, ..., **b'**ₙ}:

**Change of basis matrix P** from B to B':
$$[\mathbf{v}]_{B'} = P^{-1} [\mathbf{v}]_B$$

where P is the matrix whose columns are the old basis vectors expressed in the new basis.

**Similarity Transformation:**
If A represents a linear transformation in basis B, then in basis B':
$$A' = P^{-1} A P$$

**Intuition:**
- Coordinates are just numbers relative to a chosen basis
- A vector doesn't change; only its representation does
- Some bases make matrices diagonal (eigenvalue basis)
- Choosing the right basis simplifies problems

**Example:**
```
Standard basis: [3, 4]
New basis B = {[1,1], [1,-1]}
Same vector in B: [3.5, -0.5] (different numbers, same point!)
```

**Python Example:**
```python
import numpy as np

# Standard basis coordinates
v_std = np.array([3, 4])

# New basis vectors (as columns)
B = np.array([[1, 1],
              [1, -1]])

# Change of basis: standard → B
# v_B = B^(-1) @ v_std
B_inv = np.linalg.inv(B)
v_B = B_inv @ v_std
print("Coordinates in new basis:", v_B)  # [3.5, -0.5]

# Verify: B @ v_B should give back v_std
print("Reconstructed:", B @ v_B)  # [3, 4]

# Similarity transformation: Change basis for a matrix
A = np.array([[4, 1],
              [2, 3]])

# Find eigenvector basis (diagonalizing basis)
eigenvalues, P = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)  # [5, 2]

# A in eigenvector basis (should be diagonal)
A_diag = np.linalg.inv(P) @ A @ P
print("A in eigenvector basis:\n", A_diag)
# [[5, 0],
#  [0, 2]]  (approximately, up to numerical precision)

# PCA as change of basis
from sklearn.decomposition import PCA

X = np.random.randn(100, 3)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)  # X in principal component basis

# The principal components ARE the new basis vectors
new_basis = pca.components_  # Shape: (2, 3)

# Transform back
X_reconstructed = pca.inverse_transform(X_pca)
```

**ML Applications:**
| Application | Change of Basis |
|-------------|-----------------|
| **PCA** | Standard basis → principal component basis |
| **Whitening** | Correlated → uncorrelated features |
| **Eigendecomposition** | Original → eigenvector basis (diagonal form) |
| **Word embeddings** | Words → dense vector representation |
| **Fourier transform** | Time domain → frequency domain |

**Why It Matters:**
- **Diagonalization**: Diagonal matrices are easy (powers, exp, etc.)
- **Sparsity**: Some bases yield sparse representations
- **Interpretability**: Principal components may have meaning
- **Computation**: Right basis can speed up algorithms

**Interview Tips:**
- PCA = finding a basis where covariance is diagonal
- Eigendecomposition A = PDP⁻¹ is a change of basis
- Not all matrices are diagonalizable (need n independent eigenvectors)
- Similar matrices (P⁻¹AP = B) have same eigenvalues

---

## Question 27
- [ ] Done

**Describe the role of linear algebra in neural network computations.**

### Answer

**Definition:**  
Neural networks are fundamentally composed of linear algebraic operations (matrix multiplications, additions) interleaved with non-linear activations. Linear algebra enables efficient representation, computation, and optimization of neural networks.

**Core Components:**

| Component | Linear Algebra Operation |
|-----------|-------------------------|
| Dense/Linear layer | y = Wx + b (matrix-vector product) |
| Batch processing | Y = XW + B (matrix-matrix product) |
| Convolution | Can be expressed as matrix multiplication (im2col) |
| Attention | QKᵀ, softmax, weighted sum of V |
| Backpropagation | Chain rule via Jacobian matrices |
| Optimization | Gradient vectors, Hessian matrices |

**Mathematical Formulation:**

**Forward Pass (single layer):**
$$\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})$$

**Batch forward pass:**
$$H = \sigma(XW^T + \mathbf{1}\mathbf{b}^T)$$

**Multi-layer network:**
$$\mathbf{y} = \sigma_L(W_L \sigma_{L-1}(W_{L-1} \cdots \sigma_1(W_1\mathbf{x} + \mathbf{b}_1) \cdots + \mathbf{b}_{L-1}) + \mathbf{b}_L)$$

**Attention mechanism:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Key Linear Algebra Concepts in NNs:**

| Concept | Role in Neural Networks |
|---------|------------------------|
| Matrix multiplication | Core computation of every layer |
| Transpose | Weight sharing, gradient computation |
| Hadamard product | Gating mechanisms (LSTM, GRU) |
| Outer product | Attention, rank-1 updates |
| Norms | Regularization (L1, L2), normalization |
| Eigenvalues | Stability analysis, exploding/vanishing gradients |
| SVD | Weight initialization, compression |

**Python Example:**
```python
import numpy as np

# Simple neural network forward pass
class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros(output_dim)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, X):
        """Forward pass using matrix operations."""
        # Layer 1: Linear + ReLU
        self.z1 = X @ self.W1 + self.b1   # (batch, hidden)
        self.a1 = self.relu(self.z1)
        
        # Layer 2: Linear
        self.z2 = self.a1 @ self.W2 + self.b2  # (batch, output)
        return self.z2
    
    def backward(self, X, y, y_pred, lr=0.01):
        """Backward pass - all linear algebra!"""
        batch_size = X.shape[0]
        
        # Output layer gradient
        dz2 = y_pred - y  # (batch, output)
        dW2 = self.a1.T @ dz2 / batch_size  # (hidden, output)
        db2 = np.mean(dz2, axis=0)
        
        # Hidden layer gradient
        da1 = dz2 @ self.W2.T  # (batch, hidden)
        dz1 = da1 * (self.z1 > 0)  # ReLU gradient (Hadamard)
        dW1 = X.T @ dz1 / batch_size  # (input, hidden)
        db1 = np.mean(dz1, axis=0)
        
        # Update weights
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

# Attention mechanism
def scaled_dot_product_attention(Q, K, V):
    """
    Q: (batch, seq_len, d_k)
    K: (batch, seq_len, d_k)
    V: (batch, seq_len, d_v)
    """
    d_k = Q.shape[-1]
    
    # Attention scores: QK^T
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)  # (batch, seq, seq)
    
    # Softmax
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    
    # Weighted sum of values
    output = attention_weights @ V  # (batch, seq, d_v)
    return output, attention_weights

# Batch normalization (uses mean and variance computations)
def batch_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
```

**GPU Acceleration:**
- Matrix operations are highly parallelizable
- GPUs have thousands of cores optimized for linear algebra
- Libraries (cuBLAS, cuDNN) provide optimized implementations

**Interview Tips:**
- Without non-linearities, deep networks collapse to single linear transform
- Eigenvalue analysis reveals gradient flow properties
- Orthogonal weight initialization helps with gradient flow
- Transformer attention is fundamentally matrix multiplication

---

## Question 28
- [ ] Done

**Explain how the SVD is used in recommendation systems.**

### Answer

**Definition:**  
Recommendation systems use linear algebra—specifically matrix factorization—to model user-item interactions. A user-item rating matrix is decomposed into latent factor matrices, enabling prediction of missing ratings and discovery of similar items/users.

**Core Concepts:**
- User-Item matrix R: rows = users, columns = items, entries = ratings
- Matrix factorization: R ≈ UV^T (low-rank approximation)
- Latent factors capture hidden features (genre preference, item quality)
- SVD and variants (ALS, NMF) are primary techniques

**Mathematical Formulation:**

**Basic Matrix Factorization:**
$$R \approx U V^T$$
- R ∈ ℝᵐˣⁿ: Rating matrix (m users, n items)
- U ∈ ℝᵐˣᵏ: User latent factors
- V ∈ ℝⁿˣᵏ: Item latent factors
- k << min(m, n): Number of latent factors

**Predicted rating:**
$$\hat{r}_{ui} = \mathbf{u}_u^T \mathbf{v}_i = \sum_{f=1}^{k} u_{uf} \cdot v_{if}$$

**Loss function (with regularization):**
$$\min_{U,V} \sum_{(u,i) \in \text{observed}} (r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda(\|U\|_F^2 + \|V\|_F^2)$$

**Intuition:**
- Each user is a vector in k-dimensional "preference space"
- Each item is a vector in k-dimensional "feature space"
- Dot product of user and item vectors = predicted affinity
- Similar users/items have similar vectors (close in latent space)

**Key Techniques:**

| Method | Description | Use Case |
|--------|-------------|----------|
| **SVD** | A = UΣVᵀ, keep top-k singular values | Dense matrices |
| **Truncated SVD** | Efficient for sparse matrices | Large-scale |
| **ALS** | Alternating Least Squares | Parallel computation |
| **NMF** | Non-negative factors | Interpretable features |
| **SVD++** | Adds implicit feedback | Netflix Prize winner |

**Python Example:**
```python
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD, NMF

# User-Item rating matrix (0 = missing)
R = np.array([
    [5, 3, 0, 1, 4],
    [4, 0, 0, 1, 0],
    [1, 1, 0, 5, 0],
    [0, 0, 5, 4, 0],
    [0, 1, 5, 4, 0],
])

# Method 1: SVD-based recommendation
def svd_recommend(R, k=2):
    """SVD-based collaborative filtering."""
    # Center the ratings (subtract mean)
    user_mean = np.nanmean(np.where(R > 0, R, np.nan), axis=1)
    R_centered = R - user_mean[:, np.newaxis]
    R_centered = np.nan_to_num(R_centered)
    
    # SVD decomposition
    U, sigma, Vt = svds(R_centered, k=k)
    
    # Reconstruct and add back mean
    R_pred = U @ np.diag(sigma) @ Vt + user_mean[:, np.newaxis]
    return R_pred

R_pred = svd_recommend(R, k=2)
print("Predicted ratings:\n", np.round(R_pred, 1))

# Method 2: NMF-based recommendation
def nmf_recommend(R, k=2, max_iter=200):
    """NMF-based collaborative filtering."""
    # Replace 0s with small value for NMF
    R_filled = np.where(R == 0, 0.1, R)
    
    nmf = NMF(n_components=k, max_iter=max_iter, random_state=42)
    U = nmf.fit_transform(R_filled)
    V = nmf.components_
    
    return U @ V

# Method 3: Alternating Least Squares (ALS)
def als_recommend(R, k=2, lambda_reg=0.1, n_iter=20):
    """ALS matrix factorization."""
    m, n = R.shape
    mask = R > 0  # Observed entries
    
    # Initialize factors
    U = np.random.randn(m, k) * 0.01
    V = np.random.randn(n, k) * 0.01
    
    for _ in range(n_iter):
        # Fix V, solve for U
        for u in range(m):
            observed_items = np.where(mask[u, :])[0]
            if len(observed_items) > 0:
                V_obs = V[observed_items, :]
                R_obs = R[u, observed_items]
                U[u, :] = np.linalg.solve(
                    V_obs.T @ V_obs + lambda_reg * np.eye(k),
                    V_obs.T @ R_obs
                )
        
        # Fix U, solve for V
        for i in range(n):
            observed_users = np.where(mask[:, i])[0]
            if len(observed_users) > 0:
                U_obs = U[observed_users, :]
                R_obs = R[observed_users, i]
                V[i, :] = np.linalg.solve(
                    U_obs.T @ U_obs + lambda_reg * np.eye(k),
                    U_obs.T @ R_obs
                )
    
    return U @ V.T

R_als = als_recommend(R, k=3)
print("ALS predictions:\n", np.round(R_als, 1))

# Find similar items using cosine similarity
def find_similar_items(V, item_idx, top_n=3):
    """Find items similar to item_idx using latent factors."""
    item_vec = V[item_idx]
    similarities = V @ item_vec / (np.linalg.norm(V, axis=1) * np.linalg.norm(item_vec))
    similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
    return similar_indices, similarities[similar_indices]
```

**ML Applications:**
- Netflix, Amazon, Spotify recommendations
- Ad targeting (user-ad interaction matrix)
- Content-based filtering (item-feature matrix)
- Implicit feedback (views, clicks, purchases)

**Interview Tips:**
- Cold start problem: New users/items have no interactions
- Implicit vs explicit feedback requires different loss functions
- Real systems combine collaborative filtering with content-based features
- Scalability: Use ALS (parallelizable) or SGD for large matrices

---

## Question 29
- [ ] Done

**Explain how you would preprocess data to be used in linear algebra computations.**

### Answer

**Definition:**  
Data preprocessing transforms raw data into a suitable format for ML models. Linear algebra operations—scaling, normalization, centering, whitening, and dimensionality reduction—form the backbone of these transformations.

**Core Preprocessing Operations:**

| Operation | Linear Algebra | Purpose |
|-----------|---------------|---------|
| **Centering** | X - μ | Zero-mean data |
| **Standardization** | (X - μ) / σ | Unit variance |
| **Min-Max Scaling** | (X - min) / (max - min) | Scale to [0, 1] |
| **Whitening** | ΛV^T(X - μ)^T | Decorrelate + unit variance |
| **PCA** | Project onto eigenvectors | Dimensionality reduction |
| **One-hot encoding** | Expand to binary vectors | Categorical features |

**Mathematical Formulation:**

**Standardization (Z-score):**
$$X_{scaled} = \frac{X - \mu}{\sigma}$$

**Whitening (ZCA):**
$$X_{white} = (X - \mu) \Sigma^{-1/2}$$
where Σ is the covariance matrix.

**PCA Transformation:**
$$X_{pca} = (X - \mu) V_k$$
where V_k contains top-k eigenvectors.

**Python Example:**
```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# Sample data
X = np.array([
    [1, 200, 0.5],
    [2, 300, 0.7],
    [3, 400, 0.3],
    [4, 500, 0.9],
    [5, 600, 0.6]
])

# 1. CENTERING (subtract mean)
X_centered = X - X.mean(axis=0)
print("Centered mean:", X_centered.mean(axis=0))  # [0, 0, 0]

# 2. STANDARDIZATION (zero mean, unit variance)
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
print("Standardized std:", X_standardized.std(axis=0))  # [1, 1, 1]

# Manual standardization (linear algebra)
mean = X.mean(axis=0)
std = X.std(axis=0)
X_std_manual = (X - mean) / std

# 3. MIN-MAX SCALING
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X)
print("MinMax range:", X_minmax.min(axis=0), X_minmax.max(axis=0))  # [0,0,0], [1,1,1]

# 4. WHITENING (decorrelate and standardize)
def whiten(X):
    """ZCA whitening."""
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    
    # Eigendecomposition of covariance
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Whitening matrix
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + 1e-5))
    W = eigenvectors @ D_inv_sqrt @ eigenvectors.T
    
    return X_centered @ W

X_whitened = whiten(X)
print("Whitened covariance:\n", np.round(np.cov(X_whitened.T), 2))  # ≈ Identity

# 5. PCA TRANSFORMATION
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("Explained variance ratio:", pca.explained_variance_ratio_)

# 6. BATCH NORMALIZATION (neural networks)
def batch_norm_layer(X, gamma, beta, eps=1e-5):
    """Batch normalization: linear transformation."""
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    X_norm = (X - mean) / np.sqrt(var + eps)
    return gamma * X_norm + beta  # Scale and shift

# 7. L2 NORMALIZATION (row-wise unit vectors)
def l2_normalize(X):
    """Normalize each sample to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms

X_l2 = l2_normalize(X)
print("L2 norms:", np.linalg.norm(X_l2, axis=1))  # All 1.0
```

**Why Preprocessing Matters:**
| Issue | Solution | Benefit |
|-------|----------|---------|
| Features on different scales | Standardization | Equal feature importance |
| Correlated features | Whitening/PCA | Decorrelation, stability |
| Sparse high-dimensional | Dimensionality reduction | Efficiency, noise removal |
| Non-Gaussian | Various transforms | Better model assumptions |

**ML Applications:**
- **Neural networks**: Batch normalization, layer normalization
- **Distance-based**: K-means, KNN require scaling
- **Gradient descent**: Converges faster with normalized features
- **Regularization**: L2 penalty interpretation depends on scale

**Interview Tips:**
- Always standardize for regularized models (Ridge, Lasso, SVM)
- Fit scaler on training data only; transform both train and test
- PCA requires centered data; sklearn does this automatically
- Whitening is stronger than standardization (removes correlation)

---

## Question 30
- [ ] Done

**Describe ways to find the rank of a matrix effectively.**

### Answer

**Definition:**  
The rank of a matrix is the number of linearly independent rows (or columns). Efficient computation uses SVD, QR decomposition, or row reduction, with SVD being the most numerically stable approach.

**Methods to Compute Rank:**

| Method | Complexity | Stability | Description |
|--------|------------|-----------|-------------|
| **SVD** | O(mn·min(m,n)) | Best | Count non-zero singular values |
| **QR (pivoted)** | O(mn²) | Good | Count non-zero diagonal in R |
| **Gaussian Elimination** | O(n³) | Moderate | Count pivots |
| **Eigenvalues** | O(n³) | For square | Count non-zero eigenvalues |

**Mathematical Formulation:**

**Via SVD:**
$$\text{rank}(A) = \#\{i : \sigma_i > \epsilon\}$$
where ε is a tolerance threshold.

**Numerical tolerance:**
$$\epsilon = \max(m,n) \cdot \|A\|_2 \cdot \epsilon_{machine}$$

**Python Example:**
```python
import numpy as np
from scipy.linalg import qr

# Test matrix
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

# Method 1: NumPy's matrix_rank (uses SVD internally)
rank_numpy = np.linalg.matrix_rank(A)
print(f"NumPy rank: {rank_numpy}")  # 2

# Method 2: SVD-based (manual)
def rank_via_svd(A, tol=None):
    """Compute rank by counting significant singular values."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    if tol is None:
        tol = max(A.shape) * np.finfo(A.dtype).eps * s[0]
    
    rank = np.sum(s > tol)
    return rank, s

rank_svd, singular_values = rank_via_svd(A)
print(f"SVD rank: {rank_svd}")
print(f"Singular values: {singular_values}")  # [25.4, 1.3, ~0]

# Method 3: QR with column pivoting
def rank_via_qr(A, tol=None):
    """Compute rank using pivoted QR decomposition."""
    Q, R, P = qr(A, pivoting=True)
    
    diag_R = np.abs(np.diag(R))
    if tol is None:
        tol = max(A.shape) * np.finfo(A.dtype).eps * diag_R[0]
    
    rank = np.sum(diag_R > tol)
    return rank

rank_qr = rank_via_qr(A)
print(f"QR rank: {rank_qr}")

# Method 4: For very large sparse matrices
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

def rank_sparse(A_sparse, k=None):
    """Estimate rank of sparse matrix using truncated SVD."""
    if k is None:
        k = min(A_sparse.shape) - 1
    
    U, s, Vt = svds(A_sparse, k=k)
    tol = max(A_sparse.shape) * np.finfo(float).eps * s[0]
    return np.sum(s > tol)

# Performance comparison
import time

sizes = [100, 500, 1000]
for n in sizes:
    A_test = np.random.randn(n, n)
    
    start = time.time()
    rank = np.linalg.matrix_rank(A_test)
    elapsed = time.time() - start
    print(f"Size {n}x{n}: rank={rank}, time={elapsed:.4f}s")
```

**Practical Considerations:**

| Scenario | Best Method |
|----------|-------------|
| General dense matrix | `np.linalg.matrix_rank()` (SVD) |
| Sparse matrix | Truncated SVD or iterative methods |
| Need factorization | QR with pivoting |
| Real-time/embedded | Incremental rank estimation |

**Interview Tips:**
- Rank is numerically sensitive; use appropriate tolerance
- SVD is gold standard for stability
- For sparse matrices, use iterative methods (Lanczos, randomized SVD)
- Rank deficiency often indicates multicollinearity or data issues

---

## Question 31
- [ ] Done

**Explain the difference between Cholesky and LU decomposition.**

### Answer

**Definition:**  
Both are matrix factorizations into triangular matrices, but Cholesky is specialized for symmetric positive definite (SPD) matrices, while LU works for general square matrices.

**Comparison:**

| Aspect | LU Decomposition | Cholesky Decomposition |
|--------|------------------|----------------------|
| **Form** | A = LU (or PA = LU) | A = LLᵀ (or A = LDLᵀ) |
| **Matrix requirement** | Any square matrix | Symmetric positive definite |
| **Uniqueness** | Not unique without pivoting | Unique (for SPD) |
| **Complexity** | O(n³/3) | O(n³/6) — **2x faster** |
| **Storage** | 2 triangular matrices | 1 triangular matrix |
| **Numerical stability** | Needs pivoting | Inherently stable |

**Mathematical Formulation:**

**LU Decomposition:**
$$A = LU \quad \text{or} \quad PA = LU$$
- L: Lower triangular (1s on diagonal)
- U: Upper triangular
- P: Permutation matrix (for pivoting)

**Cholesky Decomposition:**
$$A = LL^T$$
- L: Lower triangular with positive diagonal
- Works only if A is SPD (xᵀAx > 0 for all x ≠ 0)

**When to Use Which:**

| Matrix Type | Use |
|-------------|-----|
| General square | LU |
| Symmetric positive definite | **Cholesky** (preferred) |
| Symmetric indefinite | LDLᵀ |
| Rectangular | QR or SVD |

**Python Example:**
```python
import numpy as np
from scipy.linalg import lu, cholesky, cho_factor, cho_solve

# General matrix (LU decomposition)
A_general = np.array([[2, 1, 1],
                       [4, 3, 3],
                       [8, 7, 9]], dtype=float)

P, L, U = lu(A_general)
print("LU: P @ L @ U =\n", P @ L @ U)

# Symmetric positive definite matrix (Cholesky)
A_spd = np.array([[4, 2, 2],
                  [2, 5, 1],
                  [2, 1, 6]], dtype=float)

# Verify SPD: all eigenvalues positive
eigenvalues = np.linalg.eigvalsh(A_spd)
print("Eigenvalues (all positive?):", eigenvalues)  # [2.27, 4.38, 8.35]

# Cholesky decomposition
L_chol = cholesky(A_spd, lower=True)
print("Cholesky L:\n", L_chol)
print("L @ L.T =\n", L_chol @ L_chol.T)

# Solving linear systems
b = np.array([1, 2, 3], dtype=float)

# Using LU
from scipy.linalg import lu_factor, lu_solve
lu_factored, piv = lu_factor(A_general)
x_lu = lu_solve((lu_factored, piv), b)

# Using Cholesky (faster for SPD)
c, low = cho_factor(A_spd, lower=True)
x_chol = cho_solve((c, low), b)

# Timing comparison
import time

n = 1000
A_large = np.random.randn(n, n)
A_spd_large = A_large @ A_large.T + np.eye(n)  # Ensure SPD

# LU timing
start = time.time()
lu_factor(A_spd_large)
lu_time = time.time() - start

# Cholesky timing
start = time.time()
cholesky(A_spd_large)
chol_time = time.time() - start

print(f"LU time: {lu_time:.4f}s")
print(f"Cholesky time: {chol_time:.4f}s")
print(f"Cholesky is {lu_time/chol_time:.1f}x faster")

# Multivariate Gaussian sampling (uses Cholesky)
def sample_multivariate_normal(mean, cov, n_samples):
    """Sample from N(mean, cov) using Cholesky."""
    L = cholesky(cov, lower=True)
    z = np.random.randn(n_samples, len(mean))
    return mean + z @ L.T

# Example
mean = np.array([0, 0])
cov = np.array([[1, 0.8], [0.8, 1]])
samples = sample_multivariate_normal(mean, cov, 1000)
```

**ML Applications:**

| Application | Decomposition | Reason |
|-------------|---------------|--------|
| Covariance operations | Cholesky | Covariance is SPD |
| Gaussian Process | Cholesky | Kernel matrix is SPD |
| Kalman Filter | Cholesky | Covariance updates |
| General linear systems | LU | Works for any matrix |
| Determinant computation | Both | det = product of diagonal |

**Interview Tips:**
- Cholesky fails if matrix is not SPD (use this as a test for SPD)
- Log-determinant: log|A| = 2·Σlog(Lᵢᵢ) for Cholesky
- For nearly singular SPD, add small diagonal (jitter)
- Cholesky is numerically more stable for SPD matrices

---

## Question 32
- [ ] Done

**Describe a scenario where linear algebra could be used to improve model accuracy.**

### Answer

**Definition:**  
Linear algebra techniques can improve model accuracy by addressing numerical instability, multicollinearity, noise, and high dimensionality. Key techniques include regularization, PCA, whitening, and proper matrix conditioning.

**Scenarios and Solutions:**

| Problem | Linear Algebra Solution | Improvement |
|---------|------------------------|-------------|
| Multicollinearity | Ridge regression (add λI) | Stable coefficients |
| High dimensionality | PCA / SVD | Remove noise, reduce overfitting |
| Correlated features | Whitening | Better gradient descent |
| Ill-conditioned matrices | Regularization | Numerical stability |
| Noise in data | Low-rank approximation | Denoise signals |

**Detailed Scenario: Multicollinearity in Linear Regression**

**Problem:** Features are highly correlated → XᵀX is near-singular → unstable, high-variance coefficients.

**Solution:** Ridge regression adds λI to regularize.

$$\hat{\mathbf{w}}_{ridge} = (X^TX + \lambda I)^{-1}X^T\mathbf{y}$$

**Python Example:**
```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

np.random.seed(42)

# Create data with multicollinearity
n_samples = 100
X1 = np.random.randn(n_samples)
X2 = X1 + 0.01 * np.random.randn(n_samples)  # X2 ≈ X1 (highly correlated)
X3 = np.random.randn(n_samples)
X = np.column_stack([X1, X2, X3])
y = 2*X1 + 3*X3 + np.random.randn(n_samples) * 0.5

# Check condition number (high = ill-conditioned)
XtX = X.T @ X
cond_number = np.linalg.cond(XtX)
print(f"Condition number: {cond_number:.0f}")  # Very high!

# Problem: Standard linear regression has unstable coefficients
lr = LinearRegression()
lr.fit(X, y)
print(f"OLS coefficients: {lr.coef_}")  # Unstable, high variance

# Solution 1: Ridge regression (regularization)
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
print(f"Ridge coefficients: {ridge.coef_}")  # More stable

# Solution 2: PCA to remove multicollinearity
pca = PCA(n_components=2)  # Reduce to 2 components
X_pca = pca.fit_transform(X)
lr_pca = LinearRegression()
lr_pca.fit(X_pca, y)
print(f"PCA condition number: {np.linalg.cond(X_pca.T @ X_pca):.2f}")  # Much lower

# Compare cross-validation scores
from sklearn.pipeline import Pipeline

pipelines = {
    'OLS': Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())]),
    'Ridge': Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))]),
    'PCA+LR': Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2)), ('lr', LinearRegression())])
}

for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X, y, cv=5)
    print(f"{name}: Mean CV Score = {scores.mean():.3f} ± {scores.std():.3f}")
```

**Scenario 2: Denoising with Low-Rank Approximation**

```python
# Noisy data matrix
true_signal = np.outer(np.sin(np.linspace(0, 4*np.pi, 50)), 
                        np.cos(np.linspace(0, 2*np.pi, 30)))
noise = 0.5 * np.random.randn(50, 30)
noisy_data = true_signal + noise

# Low-rank approximation via SVD
def denoise_svd(X, k):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

denoised = denoise_svd(noisy_data, k=2)

# Measure improvement
print(f"Noise MSE: {np.mean((noisy_data - true_signal)**2):.4f}")
print(f"Denoised MSE: {np.mean((denoised - true_signal)**2):.4f}")
```

**Scenario 3: Whitening for Better Optimization**

```python
# Correlated features slow down gradient descent
X_correlated = np.random.randn(1000, 2) @ np.array([[1, 0.95], [0.95, 1]])

# Whitening transformation
def whiten_data(X):
    mean = X.mean(axis=0)
    X_centered = X - mean
    cov = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    whitening_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    return X_centered @ whitening_matrix

X_whitened = whiten_data(X_correlated)
print(f"Original covariance:\n{np.cov(X_correlated.T)}")
print(f"Whitened covariance:\n{np.round(np.cov(X_whitened.T), 4)}")  # ≈ Identity
```

**Interview Tips:**
- High condition number → numerical instability → regularize
- PCA removes noise in trailing components
- Whitening improves optimization landscape (spherical contours)
- Always check eigenvalue spectrum of covariance for issues

---

## Question 33
- [ ] Done

**What are sparse matrices and how are they efficiently represented and used?**

### Answer

**Definition:**  
A sparse matrix is a matrix where most elements are zero. Sparse representations store only non-zero values and their positions, dramatically reducing memory and enabling efficient operations for large-scale problems.

**Core Concepts:**
- **Sparsity ratio**: (# zeros) / (total elements)
- A matrix is considered sparse when sparsity > 90-99%
- Dense storage: O(m×n), Sparse storage: O(nnz) where nnz = non-zeros
- Special algorithms exploit sparsity for speed

**Common Sparse Formats:**

| Format | Name | Best For | Storage |
|--------|------|----------|---------|
| **COO** | Coordinate | Building sparse matrices | 3×nnz |
| **CSR** | Compressed Sparse Row | Row slicing, matrix-vector | 2×nnz + m |
| **CSC** | Compressed Sparse Column | Column slicing | 2×nnz + n |
| **LIL** | List of Lists | Incremental construction | Variable |
| **DOK** | Dictionary of Keys | Random access | Variable |

**Mathematical Formulation:**

**COO format** stores: (row_indices, col_indices, values)

**CSR format** stores:
- `data`: non-zero values (length nnz)
- `indices`: column indices (length nnz)  
- `indptr`: row pointers (length m+1)

For row i, non-zeros are at `data[indptr[i]:indptr[i+1]]`

**Python Example:**
```python
import numpy as np
from scipy import sparse
import time

# Create a sparse matrix (many zeros)
dense = np.array([
    [1, 0, 0, 0, 2],
    [0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0],
    [4, 0, 0, 5, 0]
])

# Convert to sparse formats
coo = sparse.coo_matrix(dense)
csr = sparse.csr_matrix(dense)
csc = sparse.csc_matrix(dense)

print(f"Dense size: {dense.nbytes} bytes")
print(f"CSR size: {csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes} bytes")

# Sparsity
nnz = csr.nnz
total = csr.shape[0] * csr.shape[1]
sparsity = 1 - nnz / total
print(f"Sparsity: {sparsity:.1%}")  # 75%

# Building sparse matrices efficiently
# Method 1: COO (best for construction)
rows = [0, 0, 1, 3, 3]
cols = [0, 4, 2, 0, 3]
data = [1, 2, 3, 4, 5]
coo_built = sparse.coo_matrix((data, (rows, cols)), shape=(4, 5))

# Method 2: LIL (good for incremental)
lil = sparse.lil_matrix((4, 5))
lil[0, 0] = 1
lil[0, 4] = 2
lil[1, 2] = 3
csr_from_lil = lil.tocsr()

# Large-scale example
n = 10000
density = 0.01  # 1% non-zero

# Create random sparse matrix
sparse_large = sparse.random(n, n, density=density, format='csr')
dense_large = sparse_large.toarray()

print(f"Dense memory: {dense_large.nbytes / 1e6:.1f} MB")
print(f"Sparse memory: {(sparse_large.data.nbytes + sparse_large.indices.nbytes + sparse_large.indptr.nbytes) / 1e6:.2f} MB")

# Performance comparison: Matrix-vector multiplication
x = np.random.randn(n)

start = time.time()
y_dense = dense_large @ x
dense_time = time.time() - start

start = time.time()
y_sparse = sparse_large @ x
sparse_time = time.time() - start

print(f"Dense time: {dense_time:.4f}s")
print(f"Sparse time: {sparse_time:.4f}s")
print(f"Speedup: {dense_time/sparse_time:.1f}x")

# Sparse linear systems
from scipy.sparse.linalg import spsolve, cg

A_sparse = sparse.random(1000, 1000, density=0.01, format='csr')
A_sparse = A_sparse + sparse.eye(1000) * 10  # Make diagonally dominant
b = np.random.randn(1000)

# Direct solver
x_direct = spsolve(A_sparse, b)

# Iterative solver (conjugate gradient)
x_iterative, info = cg(A_sparse, b, tol=1e-6)
```

**ML Applications:**

| Application | Why Sparse |
|-------------|------------|
| **NLP / Text** | Bag-of-words, TF-IDF (vocabulary >> document) |
| **Recommender systems** | User-item matrices (most items unrated) |
| **Graph ML** | Adjacency matrices (few connections) |
| **One-hot encoding** | High cardinality categoricals |
| **Feature hashing** | Dimensionality reduction |

**Best Practices:**
1. Use COO/LIL for construction, CSR/CSC for computation
2. Avoid dense operations on sparse matrices
3. Use sparse-aware algorithms (`scipy.sparse.linalg`)
4. Consider approximate methods for very large sparse problems

**Interview Tips:**
- CSR for row-wise operations (ML common), CSC for column-wise
- Sparse + dense = dense (be careful!)
- Iterative solvers (CG, GMRES) often better than direct for sparse
- Graph Laplacian and adjacency matrices are naturally sparse

---

## Question 34
- [ ] Done

**Explain how tensor operations are vital in algorithms working with higher-dimensional data.**

### Answer

**Definition:**  
Tensor operations extend matrix algebra to multi-dimensional arrays, enabling efficient representation and computation on complex data like images, videos, sequences, and batched neural network operations.

**Core Concepts:**
- Tensors generalize: scalar (0D) → vector (1D) → matrix (2D) → tensor (3D+)
- Deep learning frameworks are built around tensor operations
- Operations: element-wise, reduction, contraction, broadcasting
- GPU parallelization relies on tensor structure

**Common Tensor Operations:**

| Operation | Description | Example |
|-----------|-------------|---------|
| **Reshape** | Change dimensions | (32, 28, 28) → (32, 784) |
| **Transpose/Permute** | Swap axes | (B, H, W, C) → (B, C, H, W) |
| **Broadcasting** | Expand dimensions | (10, 1) + (1, 20) → (10, 20) |
| **Reduction** | Sum/mean over axes | (B, H, W) → (B,) |
| **Einsum** | Einstein summation | Flexible contractions |
| **Batch matmul** | Batched matrix multiply | (B, N, M) @ (B, M, K) |

**Mathematical Formulation:**

**Tensor contraction (generalized matrix multiply):**
$$C_{ijk} = \sum_l A_{ijl} B_{lk}$$

**Einsum notation:**
$$\text{einsum}('ijk,kl->ijl', A, B)$$

**Python Example:**
```python
import numpy as np
import torch

# 1. IMAGE DATA (4D tensor: batch, height, width, channels)
batch_images = np.random.randn(32, 224, 224, 3)  # BHWC format
print(f"Image batch shape: {batch_images.shape}")

# Convert to PyTorch format (BCHW)
images_torch = np.transpose(batch_images, (0, 3, 1, 2))
print(f"PyTorch format: {images_torch.shape}")  # (32, 3, 224, 224)

# 2. SEQUENCE DATA (3D tensor: batch, sequence, features)
batch_sequences = np.random.randn(16, 100, 512)  # B, T, D
print(f"Sequence batch shape: {batch_sequences.shape}")

# 3. ATTENTION MECHANISM (batch matrix multiplication)
B, H, T, D = 8, 12, 64, 64  # batch, heads, tokens, dim
Q = torch.randn(B, H, T, D)  # Queries
K = torch.randn(B, H, T, D)  # Keys
V = torch.randn(B, H, T, D)  # Values

# Attention: softmax(QK^T / sqrt(d)) @ V
attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(D)
attention_weights = torch.softmax(attention_scores, dim=-1)
attention_output = torch.matmul(attention_weights, V)
print(f"Attention output: {attention_output.shape}")  # (8, 12, 64, 64)

# 4. EINSUM - Powerful tensor operation
# Matrix multiplication
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C = np.einsum('ij,jk->ik', A, B)  # Equivalent to A @ B

# Batch matrix multiplication
A_batch = np.random.randn(10, 3, 4)
B_batch = np.random.randn(10, 4, 5)
C_batch = np.einsum('bij,bjk->bik', A_batch, B_batch)

# Dot product of each row
X = np.random.randn(100, 50)
dot_products = np.einsum('ij,ij->i', X, X)  # Row-wise squared norms

# 5. BROADCASTING
a = np.random.randn(5, 1, 3)
b = np.random.randn(1, 4, 3)
c = a + b  # Broadcasting: (5, 1, 3) + (1, 4, 3) → (5, 4, 3)
print(f"Broadcast result: {c.shape}")

# 6. CONVOLUTION as tensor operation
def conv2d_manual(image, kernel):
    """Simplified 2D convolution (no padding, stride=1)."""
    H, W = image.shape
    kH, kW = kernel.shape
    out_H, out_W = H - kH + 1, W - kW + 1
    
    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            # Element-wise multiply and sum (tensor contraction)
            output[i, j] = np.sum(image[i:i+kH, j:j+kW] * kernel)
    return output

# 7. TENSOR DECOMPOSITION (higher-order SVD)
from scipy.linalg import svd

def tensor_unfold(tensor, mode):
    """Unfold tensor along specified mode."""
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

# Tucker decomposition (conceptual)
def hosvd(tensor):
    """Higher-order SVD for 3D tensor."""
    U0, _, _ = svd(tensor_unfold(tensor, 0), full_matrices=False)
    U1, _, _ = svd(tensor_unfold(tensor, 1), full_matrices=False)
    U2, _, _ = svd(tensor_unfold(tensor, 2), full_matrices=False)
    
    # Core tensor
    core = np.einsum('ijk,ia,jb,kc->abc', tensor, U0.T, U1.T, U2.T)
    return core, [U0, U1, U2]
```

**ML Applications:**

| Domain | Tensor Structure | Operations |
|--------|------------------|------------|
| **CNNs** | (B, C, H, W) | Convolution, pooling |
| **RNNs/Transformers** | (B, T, D) | Attention, sequential |
| **Video** | (B, T, H, W, C) | 3D convolution |
| **Graph NNs** | (N, N), (N, F) | Adjacency ops |
| **Tensor decomposition** | (I, J, K, ...) | CP, Tucker, TT |

**Interview Tips:**
- Know tensor shapes for common architectures (CNN, Transformer)
- Einsum is powerful and appears in research code frequently
- Broadcasting rules: align right, expand 1s
- GPU acceleration requires proper tensor layout (contiguous memory)

---

## Question 35
- [ ] Done

**What is the role of linear algebra in time series analysis?**

### Answer

**Definition:**  
Linear algebra underpins time series analysis through state-space models, autoregressive formulations, spectral analysis (Fourier/wavelets), and matrix representations of temporal dependencies.

**Core Applications:**

| Technique | Linear Algebra Role |
|-----------|---------------------|
| **AR models** | Coefficient matrices, Yule-Walker equations |
| **State-space models** | Transition matrices, Kalman filter |
| **Fourier analysis** | DFT as matrix multiplication |
| **PCA/SVD** | Dimensionality reduction of multivariate TS |
| **Hankel matrices** | Embedding for SSA, DMD |
| **Covariance estimation** | Toeplitz structure |

**Mathematical Formulations:**

**AR(p) model in matrix form:**
$$\mathbf{y} = X\boldsymbol{\phi} + \boldsymbol{\epsilon}$$

where X is a Toeplitz-like matrix of lagged values.

**State-space model:**
$$\mathbf{x}_{t+1} = A\mathbf{x}_t + B\mathbf{u}_t + \mathbf{w}_t$$
$$\mathbf{y}_t = C\mathbf{x}_t + D\mathbf{u}_t + \mathbf{v}_t$$

**Discrete Fourier Transform:**
$$\mathbf{X} = F\mathbf{x}$$

where F is the DFT matrix with $F_{jk} = e^{-2\pi i jk/N}$.

**Python Example:**
```python
import numpy as np
from scipy import linalg
from scipy.fft import fft, ifft

# Generate time series
np.random.seed(42)
T = 200
t = np.arange(T)
signal = np.sin(2*np.pi*t/20) + 0.5*np.sin(2*np.pi*t/50) + 0.3*np.random.randn(T)

# 1. AUTOREGRESSIVE MODEL (Yule-Walker)
def fit_ar_yule_walker(y, p):
    """Fit AR(p) model using Yule-Walker equations."""
    # Autocorrelation
    r = np.correlate(y, y, mode='full')[len(y)-1:]
    r = r[:p+1] / r[0]
    
    # Toeplitz matrix
    R = linalg.toeplitz(r[:p])
    
    # Solve Yule-Walker: R @ phi = r[1:p+1]
    phi = linalg.solve(R, r[1:p+1])
    
    return phi

ar_coeffs = fit_ar_yule_walker(signal, p=5)
print(f"AR(5) coefficients: {ar_coeffs}")

# 2. HANKEL MATRIX (for SSA, Dynamic Mode Decomposition)
def hankel_matrix(y, L):
    """Build Hankel matrix from time series."""
    N = len(y)
    K = N - L + 1
    H = np.zeros((L, K))
    for i in range(L):
        H[i, :] = y[i:i+K]
    return H

H = hankel_matrix(signal, L=50)
print(f"Hankel matrix shape: {H.shape}")

# SVD of Hankel matrix (Singular Spectrum Analysis)
U, s, Vt = np.linalg.svd(H, full_matrices=False)
print(f"Top 5 singular values: {s[:5]}")

# Reconstruct with top components
k = 3
H_reconstructed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# 3. FOURIER TRANSFORM (DFT as matrix multiplication)
def dft_matrix(N):
    """Create DFT matrix."""
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-2j * np.pi * k * n / N)
    return W

# DFT using matrix
F = dft_matrix(len(signal))
X_dft = F @ signal

# Equivalent to numpy fft
X_fft = fft(signal)
print(f"DFT matches FFT: {np.allclose(X_dft, X_fft)}")

# 4. STATE-SPACE MODEL (Kalman Filter)
class KalmanFilter:
    def __init__(self, A, C, Q, R, x0, P0):
        self.A = A  # State transition
        self.C = C  # Observation
        self.Q = Q  # Process noise covariance
        self.R = R  # Observation noise covariance
        self.x = x0  # State estimate
        self.P = P0  # Estimate covariance
    
    def predict(self):
        """Predict step: x = Ax, P = APA' + Q"""
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
    
    def update(self, y):
        """Update step using observation y."""
        # Kalman gain: K = PC'(CPC' + R)^{-1}
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        
        # Update state and covariance
        innovation = y - self.C @ self.x
        self.x = self.x + K @ innovation
        self.P = (np.eye(len(self.x)) - K @ self.C) @ self.P
        
        return self.x.copy()

# Example: Tracking with Kalman filter
A = np.array([[1, 1], [0, 1]])  # Position + velocity
C = np.array([[1, 0]])          # Observe position only
Q = 0.1 * np.eye(2)
R = np.array([[1.0]])
x0 = np.array([0, 0])
P0 = np.eye(2)

kf = KalmanFilter(A, C, Q, R, x0, P0)

# 5. MULTIVARIATE TIME SERIES (VAR model)
def fit_var(Y, p=1):
    """Fit VAR(p) model: Y_t = A_1 Y_{t-1} + ... + A_p Y_{t-p} + e"""
    T, n = Y.shape
    
    # Build lagged matrix
    X = np.zeros((T-p, n*p))
    for i in range(p):
        X[:, i*n:(i+1)*n] = Y[p-1-i:T-1-i, :]
    
    Y_target = Y[p:, :]
    
    # OLS: A = (X'X)^{-1} X'Y
    A = np.linalg.lstsq(X, Y_target, rcond=None)[0]
    
    return A.reshape(p, n, n)
```

**ML Applications:**

| Application | Linear Algebra Technique |
|-------------|-------------------------|
| **Forecasting** | AR/VAR coefficient estimation |
| **Anomaly detection** | PCA on sliding windows |
| **Signal separation** | SVD of Hankel (SSA) |
| **State estimation** | Kalman filter (linear algebra core) |
| **Frequency analysis** | DFT, spectral decomposition |

**Interview Tips:**
- Kalman filter is optimal for linear Gaussian state-space models
- Hankel matrix SVD = Singular Spectrum Analysis (SSA)
- Toeplitz structure in autocorrelation enables fast algorithms
- DFT/FFT is the most important algorithm in signal processing

---
