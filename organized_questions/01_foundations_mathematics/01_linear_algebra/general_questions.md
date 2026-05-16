# Linear Algebra Interview Questions - General Questions

## Question 1
- [ ] Done

**How do you perform matrix addition and subtraction?**

### Answer

**Definition:**  
Matrix addition and subtraction are element-wise operations between two matrices of identical dimensions. Each element in the result is the sum (or difference) of corresponding elements.

**Core Concepts:**
- Matrices must have the **same dimensions** (m×n)
- Operations are performed element-by-element
- Commutative: A + B = B + A
- Associative: (A + B) + C = A + (B + C)

**Mathematical Formulation:**

Addition:
$$(A + B)_{ij} = A_{ij} + B_{ij}$$

Subtraction:
$$(A - B)_{ij} = A_{ij} - B_{ij}$$

**Example:**
$$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} + \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} = \begin{pmatrix} 6 & 8 \\ 10 & 12 \end{pmatrix}$$

**Python Example:**
```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Addition
C_add = A + B
# [[6, 8], [10, 12]]

# Subtraction
C_sub = A - B
# [[-4, -4], [-4, -4]]

# Manual implementation
def matrix_add(A, B):
    if A.shape != B.shape:
        raise ValueError("Matrices must have same dimensions")
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
```

**ML Applications:**
- Gradient accumulation in mini-batch training
- Residual connections: output = F(x) + x
- Combining multiple feature matrices

**Interview Tips:**
- Dimension mismatch is a common error—always check shapes
- Broadcasting in NumPy can auto-expand dimensions (be careful!)
- Subtraction is not commutative: A - B ≠ B - A

---

## Question 2
- [ ] Done

**Define the transpose of a matrix.**

### Answer

**Definition:**  
The transpose of a matrix A, denoted Aᵀ, is obtained by interchanging its rows and columns. Element at position (i, j) becomes element at position (j, i).

**Core Concepts:**
- Original: m×n → Transpose: n×m
- Diagonal elements remain in place
- (Aᵀ)ᵀ = A
- Symmetric matrix: A = Aᵀ

**Mathematical Formulation:**
$$(A^T)_{ij} = A_{ji}$$

**Example:**
$$A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix} \quad \Rightarrow \quad A^T = \begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}$$

**Key Properties:**
| Property | Formula |
|----------|---------|
| Double transpose | (Aᵀ)ᵀ = A |
| Sum | (A + B)ᵀ = Aᵀ + Bᵀ |
| Product | (AB)ᵀ = BᵀAᵀ (reverse order!) |
| Scalar | (cA)ᵀ = cAᵀ |
| Inverse | (A⁻¹)ᵀ = (Aᵀ)⁻¹ |

**Python Example:**
```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Transpose
A_T = A.T           # Shape: (3, 2)
A_T = A.transpose() # Equivalent
A_T = np.transpose(A)  # Equivalent

# Verify property: (AB)^T = B^T @ A^T
B = np.array([[1, 2],
              [3, 4],
              [5, 6]])
AB = A @ B
assert np.allclose((A @ B).T, B.T @ A.T)

# Create symmetric matrix
S = A @ A.T  # Always symmetric
print(np.allclose(S, S.T))  # True
```

**ML Applications:**
- **Covariance matrix**: Σ = XᵀX / n
- **Gradient computation**: ∂L/∂W often involves transposes
- **Weight sharing**: CNNs use transposed convolutions
- **Normal equations**: (XᵀX)⁻¹Xᵀy

**Interview Tips:**
- (AB)ᵀ = BᵀAᵀ — the order reverses (very common mistake!)
- Row vector becomes column vector after transpose
- XᵀX is always symmetric and positive semi-definite

---

## Question 3
- [ ] Done

**How do you calculate the norm of a vector and what does it represent?**

### Answer

**Definition:**  
A vector norm measures the "size" or "length" of a vector. Different norms capture different notions of magnitude, with L2 (Euclidean) being most common.

**Core Concepts:**
- Norm is always non-negative
- ||**v**|| = 0 iff **v** = **0**
- Triangle inequality: ||**u** + **v**|| ≤ ||**u**|| + ||**v**||
- Homogeneity: ||c**v**|| = |c| · ||**v**||

**Common Norms:**

| Norm | Formula | Interpretation |
|------|---------|----------------|
| **L1** (Manhattan) | Σ\|vᵢ\| | Sum of absolute values |
| **L2** (Euclidean) | √(Σvᵢ²) | Straight-line distance |
| **L∞** (Max) | max(\|vᵢ\|) | Largest absolute component |
| **Lp** (General) | (Σ\|vᵢ\|ᵖ)^(1/p) | Generalized norm |

**Mathematical Formulation:**

$$\|\mathbf{v}\|_1 = \sum_{i=1}^{n} |v_i|$$

$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2} = \sqrt{\mathbf{v}^T\mathbf{v}}$$

$$\|\mathbf{v}\|_\infty = \max_i |v_i|$$

**Python Example:**
```python
import numpy as np

v = np.array([3, -4, 5])

# L1 norm (Manhattan)
l1 = np.linalg.norm(v, ord=1)  # |3| + |-4| + |5| = 12
l1_manual = np.sum(np.abs(v))

# L2 norm (Euclidean) - default
l2 = np.linalg.norm(v)         # sqrt(9 + 16 + 25) = sqrt(50) ≈ 7.07
l2_manual = np.sqrt(np.dot(v, v))

# L-infinity norm (Max)
linf = np.linalg.norm(v, ord=np.inf)  # max(3, 4, 5) = 5

# Normalize to unit vector
unit_v = v / np.linalg.norm(v)
print(np.linalg.norm(unit_v))  # 1.0

# Matrix norms
A = np.array([[1, 2], [3, 4]])
frobenius = np.linalg.norm(A, 'fro')  # sqrt(sum of squares)
spectral = np.linalg.norm(A, 2)       # Largest singular value
```

**ML Applications:**
| Application | Norm Used |
|-------------|-----------|
| **L2 Regularization** (Ridge) | ||**w**||₂² |
| **L1 Regularization** (Lasso) | ||**w**||₁ |
| **Gradient clipping** | ||∇||₂ |
| **Distance metrics** | ||**x** - **y**||₂ |
| **Batch normalization** | ||**x**||₂ |

**Interview Tips:**
- L1 promotes sparsity (many zeros), L2 promotes small weights
- L2 is differentiable everywhere; L1 is not differentiable at 0
- Frobenius norm for matrices = L2 norm of flattened matrix
- Spectral norm = largest singular value (important for Lipschitz)

---

## Question 4
- [ ] Done

**Define the concept of orthogonality in linear algebra.**

### Answer

**Definition:**  
Two vectors are orthogonal if their dot product is zero, meaning they are perpendicular in geometric terms. Orthogonal sets form the foundation for many decompositions and are crucial for numerical stability.

**Core Concepts:**
- **Orthogonal vectors**: **u** · **v** = 0
- **Orthonormal vectors**: Orthogonal + unit length (||**u**|| = 1)
- **Orthogonal matrix**: Q where QᵀQ = QQᵀ = I
- **Orthogonal complement**: All vectors perpendicular to a subspace

**Mathematical Formulation:**

Vectors orthogonal:
$$\mathbf{u} \perp \mathbf{v} \iff \mathbf{u}^T\mathbf{v} = 0 \iff \langle\mathbf{u}, \mathbf{v}\rangle = 0$$

Orthonormal set {**q**₁, **q**₂, ..., **q**ₙ}:
$$\mathbf{q}_i^T\mathbf{q}_j = \delta_{ij} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$$

**Orthogonal Projection:**
$$\text{proj}_\mathbf{u}(\mathbf{v}) = \frac{\mathbf{u}^T\mathbf{v}}{\mathbf{u}^T\mathbf{u}}\mathbf{u}$$

**Python Example:**
```python
import numpy as np

# Check orthogonality
u = np.array([1, 0, 0])
v = np.array([0, 1, 0])
w = np.array([1, 1, 0])

print(np.dot(u, v))  # 0 → orthogonal
print(np.dot(u, w))  # 1 → not orthogonal

# Orthonormal vectors (standard basis)
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

# Verify orthonormality
Q = np.column_stack([e1, e2, e3])
print(np.allclose(Q.T @ Q, np.eye(3)))  # True

# Gram-Schmidt orthogonalization
def gram_schmidt(V):
    """Orthonormalize columns of V."""
    Q = np.zeros_like(V, dtype=float)
    for i in range(V.shape[1]):
        q = V[:, i].astype(float)
        for j in range(i):
            q -= np.dot(Q[:, j], V[:, i]) * Q[:, j]
        Q[:, i] = q / np.linalg.norm(q)
    return Q

# Example
V = np.array([[1, 1], [1, 0], [0, 1]], dtype=float)
Q = gram_schmidt(V)
print("Q^T Q:\n", Q.T @ Q)  # ≈ Identity

# Orthogonal projection
def project(v, u):
    """Project v onto u."""
    return (np.dot(u, v) / np.dot(u, u)) * u

# Project onto subspace spanned by orthonormal columns of Q
def project_subspace(v, Q):
    """Project v onto column space of orthonormal Q."""
    return Q @ (Q.T @ v)
```

**ML Applications:**
- **PCA**: Principal components are orthogonal
- **QR decomposition**: Q is orthogonal
- **Orthogonal weight initialization**: Helps gradient flow
- **Decorrelation**: Whitening produces orthogonal features
- **Regularization**: Orthogonal constraints in neural networks

**Interview Tips:**
- Orthogonal vectors are always linearly independent
- Orthogonal matrices preserve lengths and angles
- Gram-Schmidt is numerically unstable; use QR decomposition instead
- In high dimensions, random vectors are nearly orthogonal

---

## Question 5
- [ ] Done

**Define what a symmetric matrix is.**

### Answer

**Definition:**  
A symmetric matrix is a square matrix that equals its own transpose: A = Aᵀ. This means element Aᵢⱼ = Aⱼᵢ for all i, j—the matrix is mirrored across its main diagonal.

**Core Concepts:**
- Only square matrices can be symmetric
- Diagonal elements can be any value
- Off-diagonal elements are mirrored: Aᵢⱼ = Aⱼᵢ
- Real symmetric matrices have real eigenvalues

**Mathematical Formulation:**
$$A = A^T \iff A_{ij} = A_{ji} \quad \forall i, j$$

**Key Properties:**
| Property | Description |
|----------|-------------|
| Eigenvalues | Always real |
| Eigenvectors | Orthogonal (for distinct eigenvalues) |
| Diagonalizable | A = QΛQᵀ (Q orthogonal) |
| XᵀAX | = (XᵀAX)ᵀ (quadratic form is scalar) |

**Special Symmetric Matrices:**
- **Covariance matrix**: Σ = E[(X - μ)(X - μ)ᵀ]
- **Gram matrix**: G = XᵀX
- **Kernel matrix**: K where Kᵢⱼ = k(xᵢ, xⱼ)
- **Laplacian matrix**: L = D - A (graph theory)

**Python Example:**
```python
import numpy as np

# Check symmetry
A = np.array([[1, 2, 3],
              [2, 5, 6],
              [3, 6, 9]])

is_symmetric = np.allclose(A, A.T)
print(f"Is symmetric: {is_symmetric}")  # True

# Create symmetric matrix from any matrix
B = np.random.randn(3, 3)
B_symmetric = (B + B.T) / 2
print(np.allclose(B_symmetric, B_symmetric.T))  # True

# Covariance matrix is always symmetric
X = np.random.randn(100, 5)
cov = np.cov(X.T)
print(f"Covariance symmetric: {np.allclose(cov, cov.T)}")  # True

# Gram matrix X^T X is symmetric
G = X.T @ X
print(f"Gram symmetric: {np.allclose(G, G.T)}")  # True

# Eigendecomposition of symmetric matrix
eigenvalues, eigenvectors = np.linalg.eigh(A)  # eigh for symmetric
print(f"Eigenvalues (real): {eigenvalues}")

# Verify orthogonal eigenvectors
Q = eigenvectors
print(f"Q^T Q ≈ I: {np.allclose(Q.T @ Q, np.eye(3))}")

# Reconstruct: A = Q Λ Q^T
Lambda = np.diag(eigenvalues)
A_reconstructed = Q @ Lambda @ Q.T
print(f"Reconstruction correct: {np.allclose(A, A_reconstructed)}")
```

**ML Applications:**
- **PCA**: Eigendecomposition of symmetric covariance matrix
- **Kernel methods**: Kernel matrix is symmetric
- **Graph algorithms**: Laplacian matrix is symmetric
- **Optimization**: Hessian matrix (second derivatives) is symmetric
- **Covariance estimation**: Sample covariance is symmetric

**Interview Tips:**
- Use `np.linalg.eigh()` for symmetric matrices (faster, more stable)
- Symmetric + positive definite → Cholesky decomposition exists
- Skew-symmetric: A = -Aᵀ (Aᵢⱼ = -Aⱼᵢ)
- Any matrix can be decomposed: A = (A + Aᵀ)/2 + (A - Aᵀ)/2 (symmetric + skew)

---

## Question 6
- [ ] Done

**Define positive definiteness of a matrix.**

### Answer

**Definition:**  
A symmetric matrix A is positive definite (PD) if **x**ᵀA**x** > 0 for all non-zero vectors **x**. Positive semi-definite (PSD) allows **x**ᵀA**x** ≥ 0. These matrices have non-negative eigenvalues and arise naturally in ML.

**Core Concepts:**
- Must be symmetric (for real matrices)
- All eigenvalues > 0 (PD) or ≥ 0 (PSD)
- Cholesky decomposition exists: A = LLᵀ
- Represents valid covariance matrices

**Types:**
| Type | Condition | Eigenvalues |
|------|-----------|-------------|
| Positive definite (PD) | **x**ᵀA**x** > 0, ∀**x** ≠ 0 | All λᵢ > 0 |
| Positive semi-definite (PSD) | **x**ᵀA**x** ≥ 0, ∀**x** | All λᵢ ≥ 0 |
| Negative definite | **x**ᵀA**x** < 0, ∀**x** ≠ 0 | All λᵢ < 0 |
| Indefinite | Mixed signs possible | Mixed λᵢ |

**Tests for Positive Definiteness:**
1. All eigenvalues > 0
2. All leading principal minors > 0 (Sylvester's criterion)
3. Cholesky decomposition succeeds
4. det(A) > 0 and A₁₁ > 0 (necessary but not sufficient alone)

**Mathematical Formulation:**

Positive definite:
$$\mathbf{x}^T A \mathbf{x} > 0 \quad \forall \mathbf{x} \neq \mathbf{0}$$

Quadratic form expansion (2×2):
$$\begin{pmatrix} x_1 & x_2 \end{pmatrix} \begin{pmatrix} a & b \\ b & c \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = ax_1^2 + 2bx_1x_2 + cx_2^2$$

**Python Example:**
```python
import numpy as np
from scipy.linalg import cholesky

# Positive definite matrix
A = np.array([[4, 2],
              [2, 3]])

# Test 1: Eigenvalues all positive
eigenvalues = np.linalg.eigvalsh(A)
is_pd_eigen = np.all(eigenvalues > 0)
print(f"Eigenvalues: {eigenvalues}")  # [1.76, 5.24] - all positive
print(f"PD by eigenvalues: {is_pd_eigen}")

# Test 2: Cholesky succeeds
try:
    L = cholesky(A, lower=True)
    print("Cholesky successful → PD")
    print(f"L:\n{L}")
    print(f"L @ L.T:\n{L @ L.T}")  # Reconstructs A
except np.linalg.LinAlgError:
    print("Cholesky failed → Not PD")

# Test 3: Quadratic form test
def is_positive_definite(A, n_samples=1000):
    """Monte Carlo test for positive definiteness."""
    for _ in range(n_samples):
        x = np.random.randn(A.shape[0])
        if x.T @ A @ x <= 0:
            return False
    return True

# Creating PD matrices
# Method 1: A^T A is always PSD
B = np.random.randn(3, 3)
PSD = B.T @ B  # Positive semi-definite

# Method 2: Add diagonal to make PD
PD = B.T @ B + 0.1 * np.eye(3)  # Positive definite

# Covariance matrix (always PSD)
X = np.random.randn(100, 5)
cov = np.cov(X.T)
print(f"Covariance eigenvalues: {np.linalg.eigvalsh(cov)}")  # All ≥ 0

# Regularization to ensure PD
def make_positive_definite(A, eps=1e-6):
    """Add small diagonal to ensure positive definiteness."""
    return A + eps * np.eye(A.shape[0])
```

**ML Applications:**
- **Covariance matrices**: Must be PSD
- **Kernel matrices**: Valid kernels produce PSD Gram matrices
- **Optimization**: Hessian PD at minimum (convex)
- **Gaussian processes**: Kernel matrix must be PSD
- **Regularization**: λI ensures PD (Ridge regression)

**Interview Tips:**
- Covariance matrices are PSD by construction
- Near-PSD matrices (numerical issues) → add small diagonal (jitter)
- PD ⟹ invertible (all eigenvalues non-zero)
- **x**ᵀA**x** is the "energy" associated with state **x**

---

## Question 7
- [ ] Done

**How do you represent a system of linear equations using matrices?**

### Answer

**Definition:**  
A system of linear equations can be compactly written as A**x** = **b**, where A is the coefficient matrix, **x** is the unknown vector, and **b** is the constants vector. This matrix form enables systematic solution methods.

**Core Concepts:**
- Each row of A corresponds to one equation
- Each column of A corresponds to one variable
- Matrix form enables computational solutions
- Existence/uniqueness depends on rank

**Mathematical Formulation:**

System of equations:
$$\begin{cases} a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\ a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\ \vdots \\ a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m \end{cases}$$

Matrix form:
$$\underbrace{\begin{pmatrix} a_{11} & \cdots & a_{1n} \\ \vdots & \ddots & \vdots \\ a_{m1} & \cdots & a_{mn} \end{pmatrix}}_{A} \underbrace{\begin{pmatrix} x_1 \\ \vdots \\ x_n \end{pmatrix}}_{\mathbf{x}} = \underbrace{\begin{pmatrix} b_1 \\ \vdots \\ b_m \end{pmatrix}}_{\mathbf{b}}$$

**Solution Methods:**
| Method | When to Use | Complexity |
|--------|-------------|------------|
| Direct inverse | Small, invertible A | O(n³) |
| LU decomposition | Square, multiple b vectors | O(n³) |
| QR decomposition | Rectangular, least squares | O(mn²) |
| Iterative (CG, GMRES) | Large, sparse | O(n²) per iteration |

**Python Example:**
```python
import numpy as np

# System: 2x + y = 5
#         x + 3y = 7

A = np.array([[2, 1],
              [1, 3]])
b = np.array([5, 7])

# Method 1: np.linalg.solve (recommended)
x = np.linalg.solve(A, b)
print(f"Solution: {x}")  # [1.6, 1.8]

# Verify
print(f"Ax = {A @ x}")  # Should equal b

# Method 2: Using inverse (not recommended for large systems)
x_inv = np.linalg.inv(A) @ b

# Method 3: Least squares (for overdetermined systems)
# 3 equations, 2 unknowns
A_over = np.array([[1, 1],
                   [2, 1],
                   [1, 2]])
b_over = np.array([3, 4, 4])

x_lstsq, residuals, rank, s = np.linalg.lstsq(A_over, b_over, rcond=None)
print(f"Least squares solution: {x_lstsq}")

# Method 4: For sparse systems
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

A_sparse = csr_matrix(A)
x_sparse = spsolve(A_sparse, b)

# Linear regression as system of equations
# Normal equations: (X^T X) w = X^T y
X = np.random.randn(100, 5)
y = np.random.randn(100)

XtX = X.T @ X
Xty = X.T @ y
w = np.linalg.solve(XtX, Xty)

# Equivalent to
w_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"Solutions match: {np.allclose(w, w_lstsq)}")
```

**ML Applications:**
- **Linear regression**: Solve X**w** = **y** (or normal equations)
- **Neural network training**: Solve for weights
- **Kalman filter**: State estimation via linear systems
- **Least squares**: Overdetermined systems in curve fitting

**Interview Tips:**
- Use `np.linalg.solve()` not `np.linalg.inv() @` (more stable, faster)
- Overdetermined (m > n): Use least squares
- Underdetermined (m < n): Infinite solutions, need regularization
- Check condition number for numerical stability

---

## Question 8
- [ ] Done

**Define and differentiate between homogenous and non-homogenous systems.**

### Answer

**Definition:**  
A **homogeneous system** has the form A**x** = **0** (zero vector on right side). A **non-homogeneous system** has A**x** = **b** where **b** ≠ **0**. Homogeneous systems always have at least the trivial solution (**x** = **0**).

**Core Concepts:**

| Aspect | Homogeneous (A**x** = **0**) | Non-homogeneous (A**x** = **b**) |
|--------|------------------------------|----------------------------------|
| Trivial solution | Always exists (**x** = **0**) | May not exist |
| Non-trivial solutions | Exist if rank(A) < n | Depend on rank conditions |
| Solution set | Vector subspace (null space) | Affine subspace |
| Consistency | Always consistent | May be inconsistent |

**Mathematical Formulation:**

**Homogeneous:**
$$A\mathbf{x} = \mathbf{0}$$
- Solutions form the **null space** of A: Null(A) = {**x** : A**x** = **0**}
- Dimension = n - rank(A)

**Non-homogeneous:**
$$A\mathbf{x} = \mathbf{b}$$
- General solution: **x** = **x**ₚ + **x**ₕ
  - **x**ₚ = particular solution
  - **x**ₕ = any homogeneous solution

**Solution Structure:**
```
Non-homogeneous solution = Particular solution + Null space

If x_p solves Ax = b, then all solutions are:
x = x_p + x_h where x_h ∈ Null(A)
```

**Python Example:**
```python
import numpy as np
from scipy.linalg import null_space

# Homogeneous system: Ax = 0
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Find null space (all solutions to Ax = 0)
null_A = null_space(A)
print(f"Null space dimension: {null_A.shape[1]}")  # 1

# Verify: A @ null_vector ≈ 0
if null_A.size > 0:
    print(f"A @ null_vector: {A @ null_A[:, 0]}")  # ≈ [0, 0, 0]

# Trivial solution always works
x_trivial = np.zeros(3)
print(f"A @ 0: {A @ x_trivial}")  # [0, 0, 0]

# Non-homogeneous system: Ax = b
A_nh = np.array([[2, 1],
                 [1, 3]])
b = np.array([5, 7])

# Particular solution
x_particular = np.linalg.solve(A_nh, b)
print(f"Particular solution: {x_particular}")

# For this system, null space is trivial (unique solution)
null_A_nh = null_space(A_nh)
print(f"Null space dimension: {null_A_nh.shape[1]}")  # 0

# Example with infinite solutions
A_inf = np.array([[1, 2, 3],
                  [2, 4, 6]])  # Row 2 = 2 * Row 1
b_inf = np.array([6, 12])     # Consistent

# General solution = particular + null space
x_part, _, _, _ = np.linalg.lstsq(A_inf, b_inf, rcond=None)
null_A_inf = null_space(A_inf)

print(f"Particular: {x_part}")
print(f"Null space basis:\n{null_A_inf}")

# Any solution: x = x_part + c1*null[:,0] + c2*null[:,1]
c1, c2 = 2, -1
x_general = x_part + c1 * null_A_inf[:, 0] + c2 * null_A_inf[:, 1]
print(f"Verify: A @ x_general = {A_inf @ x_general}")  # Should ≈ b_inf
```

**ML Applications:**
- **Regularization**: Without regularization, weight space has null space
- **Kernel trick**: Kernel null space affects solution
- **Feature engineering**: Detecting redundant features (non-trivial null space)
- **Underdetermined systems**: Compressed sensing exploits structure

**Interview Tips:**
- Homogeneous systems always have **x** = **0**; question is whether other solutions exist
- Non-trivial homogeneous solutions exist iff rank(A) < n
- General solution = particular + homogeneous is key insight
- Regularization "picks" one solution from infinite possibilities

---

## Question 9
- [ ] Done

**How do you compute the inverse of a matrix and when is it possible?**

### Answer

**Definition:**  
The inverse of a square matrix A, denoted A⁻¹, is the matrix such that AA⁻¹ = A⁻¹A = I. A matrix is invertible (non-singular) if and only if its determinant is non-zero.

**Core Concepts:**
- Only **square matrices** can have inverses
- Inverse exists iff det(A) ≠ 0
- Inverse exists iff rank(A) = n (full rank)
- Inverse exists iff all eigenvalues ≠ 0

**Conditions for Invertibility:**
| Condition | Invertible | Singular |
|-----------|------------|----------|
| Determinant | det(A) ≠ 0 | det(A) = 0 |
| Rank | rank(A) = n | rank(A) < n |
| Eigenvalues | All λᵢ ≠ 0 | Some λᵢ = 0 |
| Null space | {0} only | Non-trivial |
| Columns | Linearly independent | Linearly dependent |

**Mathematical Formulation:**

**2×2 Matrix Inverse:**
$$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix} \quad \Rightarrow \quad A^{-1} = \frac{1}{ad-bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

**Properties:**
- (A⁻¹)⁻¹ = A
- (AB)⁻¹ = B⁻¹A⁻¹ (reverse order)
- (Aᵀ)⁻¹ = (A⁻¹)ᵀ
- (cA)⁻¹ = (1/c)A⁻¹

**Computation Methods:**
1. **Gauss-Jordan elimination**: [A | I] → [I | A⁻¹]
2. **LU decomposition**: Solve AU = I column by column
3. **Adjugate method**: A⁻¹ = adj(A)/det(A)

**Python Example:**
```python
import numpy as np

A = np.array([[4, 7],
              [2, 6]])

# Check if invertible
det_A = np.linalg.det(A)
print(f"Determinant: {det_A}")  # 10 (non-zero → invertible)

if np.abs(det_A) > 1e-10:
    # Method 1: NumPy inverse
    A_inv = np.linalg.inv(A)
    print(f"Inverse:\n{A_inv}")
    
    # Verify: A @ A_inv = I
    print(f"A @ A_inv:\n{A @ A_inv}")

# 2x2 inverse formula (manual)
def inverse_2x2(A):
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    det = a*d - b*c
    if np.abs(det) < 1e-10:
        raise ValueError("Matrix is singular")
    return np.array([[d, -b], [-c, a]]) / det

A_inv_manual = inverse_2x2(A)
print(f"Manual inverse matches: {np.allclose(A_inv, A_inv_manual)}")

# Singular matrix (no inverse)
B = np.array([[1, 2],
              [2, 4]])  # Row 2 = 2 * Row 1
print(f"Singular det: {np.linalg.det(B)}")  # ≈ 0

# Use pseudoinverse for singular matrices
B_pinv = np.linalg.pinv(B)
print(f"Pseudoinverse:\n{B_pinv}")

# Solving Ax = b: use solve, not inverse
b = np.array([1, 2])

# Good (numerically stable)
x = np.linalg.solve(A, b)

# Bad (less stable, slower)
x_bad = np.linalg.inv(A) @ b

# Check condition number
cond = np.linalg.cond(A)
print(f"Condition number: {cond}")  # Lower is better

# Near-singular matrix (ill-conditioned)
C = np.array([[1, 1],
              [1, 1.0001]])
print(f"Near-singular condition: {np.linalg.cond(C)}")  # Very large!
```

**Algorithm: Gauss-Jordan Inverse**
1. Create augmented matrix [A | I]
2. Apply row operations to transform A into I
3. The right side becomes A⁻¹
4. If A cannot become I, matrix is singular

**ML Applications:**
- **Normal equations**: w = (XᵀX)⁻¹Xᵀy
- **Covariance inverse**: Precision matrix Σ⁻¹
- **Kalman filter**: Innovation covariance inverse
- **Newton's method**: Hessian inverse for optimization

**Interview Tips:**
- **Never use inverse directly** to solve Ax = b; use `np.linalg.solve()`
- Pseudoinverse (A⁺) works for any matrix, including singular
- High condition number → numerical instability → use regularization
- Computing inverse is O(n³), same as solving one system, but solving is more stable

---

## Question 10
- [ ] Done

**How do you perform QR decomposition?**

### Answer

**Definition:**  
QR decomposition factors a matrix A into Q (orthogonal) and R (upper triangular) such that A = QR. It is numerically stable and fundamental for solving least squares problems and eigenvalue computation.

**Core Concepts:**
- Q: Orthogonal matrix (QᵀQ = I)
- R: Upper triangular matrix
- Works for any m×n matrix (m ≥ n)
- More numerically stable than normal equations

**Mathematical Formulation:**
$$A = QR$$

For A ∈ ℝᵐˣⁿ (m ≥ n):
- Full QR: Q ∈ ℝᵐˣᵐ, R ∈ ℝᵐˣⁿ
- Reduced QR: Q ∈ ℝᵐˣⁿ, R ∈ ℝⁿˣⁿ

**Methods:**

| Method | Description | Stability |
|--------|-------------|-----------|
| **Gram-Schmidt** | Orthogonalize columns sequentially | Poor |
| **Modified Gram-Schmidt** | More stable variant | Moderate |
| **Householder** | Reflections to zero below diagonal | Best |
| **Givens** | Rotations to zero elements | Good for sparse |

**Algorithm: Modified Gram-Schmidt**
```
For j = 1 to n:
    q_j = a_j
    For i = 1 to j-1:
        r_ij = q_i^T a_j
        q_j = q_j - r_ij * q_i
    r_jj = ||q_j||
    q_j = q_j / r_jj
```

**Python Example:**
```python
import numpy as np

A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

# NumPy QR decomposition
Q, R = np.linalg.qr(A)
print("Q:\n", Q)
print("R:\n", R)

# Verify
print("Q @ R:\n", Q @ R)  # Should equal A
print("Q^T @ Q:\n", Q.T @ Q)  # Should be identity

# Modified Gram-Schmidt implementation
def modified_gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    
    return Q, R

Q_mgs, R_mgs = modified_gram_schmidt(A)
print("MGS matches NumPy:", np.allclose(np.abs(Q), np.abs(Q_mgs)))

# Solving least squares via QR (more stable than normal equations)
def solve_least_squares_qr(A, b):
    """Solve min ||Ax - b||² using QR decomposition."""
    Q, R = np.linalg.qr(A)
    # Ax = b → QRx = b → Rx = Q^T b
    return np.linalg.solve(R, Q.T @ b)

# Overdetermined system
A_over = np.array([[1, 1],
                   [1, 2],
                   [1, 3]], dtype=float)
b = np.array([1, 2, 2], dtype=float)

x_qr = solve_least_squares_qr(A_over, b)
x_lstsq = np.linalg.lstsq(A_over, b, rcond=None)[0]
print(f"QR solution: {x_qr}")
print(f"Solutions match: {np.allclose(x_qr, x_lstsq)}")

# QR for computing eigenvalues (QR algorithm)
def qr_algorithm(A, n_iter=50):
    """Find eigenvalues using QR iteration."""
    Ak = A.copy()
    for _ in range(n_iter):
        Q, R = np.linalg.qr(Ak)
        Ak = R @ Q
    return np.diag(Ak)  # Eigenvalues on diagonal

eigenvalues_qr = qr_algorithm(A)
eigenvalues_true = np.linalg.eigvalsh(A)
print(f"QR eigenvalues: {sorted(eigenvalues_qr)}")
print(f"True eigenvalues: {sorted(eigenvalues_true)}")
```

**ML Applications:**
- **Linear regression**: Solve least squares via QR (more stable)
- **Eigenvalue computation**: QR algorithm
- **Orthogonalization**: Neural network weight orthogonalization
- **PCA**: QR can be used as part of SVD computation

**Interview Tips:**
- QR is more stable than (XᵀX)⁻¹Xᵀy for least squares
- Householder QR is the default in NumPy (most stable)
- QR algorithm iteratively finds eigenvalues
- For tall matrices, reduced QR is more memory efficient

---

## Question 11
- [ ] Done

**How can you represent linear transformation using a matrix?**

### Answer

**Definition:**  
Every linear transformation T: ℝⁿ → ℝᵐ can be uniquely represented by an m×n matrix A, where T(**x**) = A**x**. The matrix is constructed by applying T to each standard basis vector.

**Core Concepts:**
- Linear transformation ↔ Matrix (one-to-one correspondence)
- Matrix columns = transformed basis vectors
- Composition of transformations = matrix multiplication
- The matrix depends on the choice of basis

**Mathematical Formulation:**

Construction of matrix A for transformation T:
$$A = \begin{pmatrix} | & | & & | \\ T(\mathbf{e}_1) & T(\mathbf{e}_2) & \cdots & T(\mathbf{e}_n) \\ | & | & & | \end{pmatrix}$$

where **e**ᵢ are standard basis vectors.

**Common Transformations:**

| Transformation | Matrix (2D) | Effect |
|----------------|-------------|--------|
| **Scaling** | [[sₓ, 0], [0, sᵧ]] | Scale by sₓ, sᵧ |
| **Rotation** (θ) | [[cos θ, -sin θ], [sin θ, cos θ]] | Rotate by θ |
| **Reflection** (x-axis) | [[1, 0], [0, -1]] | Flip vertically |
| **Shear** (horizontal) | [[1, k], [0, 1]] | Shear by k |
| **Projection** (x-axis) | [[1, 0], [0, 0]] | Project onto x-axis |

**Python Example:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Rotation transformation
def rotation_matrix(theta):
    """Return 2D rotation matrix for angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

# Scaling transformation
def scaling_matrix(sx, sy):
    return np.array([[sx, 0],
                     [0, sy]])

# Apply transformation to points
def transform(points, matrix):
    """Transform points (2xN) using transformation matrix."""
    return matrix @ points

# Create a square
square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]])

# Apply transformations
theta = np.pi / 4  # 45 degrees
R = rotation_matrix(theta)
S = scaling_matrix(2, 0.5)

square_rotated = transform(square, R)
square_scaled = transform(square, S)
square_both = transform(square, R @ S)  # Composition

# Construct matrix from transformation
def construct_matrix(T, n):
    """Construct matrix for transformation T: R^n -> R^m."""
    columns = []
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        columns.append(T(e_i))
    return np.column_stack(columns)

# Example: Define a transformation
def my_transform(x):
    """T(x, y) = (2x + y, x - y)"""
    return np.array([2*x[0] + x[1], x[0] - x[1]])

A = construct_matrix(my_transform, 2)
print(f"Matrix for transformation:\n{A}")
# [[2, 1],
#  [1, -1]]

# Verify: A @ [1, 1] should equal T([1, 1])
x = np.array([1, 1])
print(f"A @ x: {A @ x}")
print(f"T(x): {my_transform(x)}")

# Neural network layer as linear transformation
def linear_layer(x, W, b=None):
    """y = Wx + b (affine transformation)"""
    y = W @ x
    if b is not None:
        y = y + b  # b makes it affine, not linear
    return y

# Weight matrix IS the transformation matrix
W = np.random.randn(10, 5)  # Transform from R^5 to R^10
x = np.random.randn(5)
y = linear_layer(x, W)
print(f"Input dim: {x.shape}, Output dim: {y.shape}")
```

**Intuition:**
- Matrix A "encodes" what T does to basis vectors
- Each column shows where a basis vector lands
- Matrix multiplication = applying the transformation

**ML Applications:**
- **Neural network layers**: y = Wx (before activation)
- **Feature transformation**: Map features to new space
- **PCA**: Transform to principal component space
- **Attention**: Query, key, value projections

**Interview Tips:**
- Matrix multiplication is function composition (right to left)
- The same transformation has different matrices in different bases
- Affine transformation (Wx + b) is not linear (doesn't preserve origin)
- Knowing this connection helps understand what weight matrices "do"

---

## Question 12
- [ ] Done

**How is linear regression related to linear algebra?**

### Answer

**Definition:**  
Linear regression finds weights **w** that minimize ||X**w** - **y**||². The solution involves solving linear systems, matrix decompositions, and projections—all core linear algebra operations.

**Core Concepts:**
- Model: y = X**w** + ε
- Objective: min ||X**w** - **y**||²
- Solution: Normal equations, QR, or SVD
- Geometric view: Project **y** onto column space of X

**Mathematical Formulation:**

**Normal Equations:**
$$X^TX\mathbf{w} = X^T\mathbf{y}$$
$$\mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}$$

**Geometric Interpretation:**
The prediction ŷ = X**w** is the orthogonal projection of **y** onto Col(X).

**Ridge Regression (Regularized):**
$$\mathbf{w} = (X^TX + \lambda I)^{-1}X^T\mathbf{y}$$

**Solution Methods:**

| Method | Formula | When to Use |
|--------|---------|-------------|
| Normal equations | (XᵀX)⁻¹Xᵀy | Small n, well-conditioned |
| QR decomposition | R⁻¹Qᵀy | General, numerically stable |
| SVD | VΣ⁻¹Uᵀy | Ill-conditioned, regularization |
| Gradient descent | Iterative | Large-scale, online |

**Python Example:**
```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

np.random.seed(42)

# Generate data
n_samples, n_features = 100, 5
X = np.random.randn(n_samples, n_features)
true_w = np.array([1, -2, 3, -4, 5])
y = X @ true_w + 0.5 * np.random.randn(n_samples)

# Method 1: Normal equations
XtX = X.T @ X
Xty = X.T @ y
w_normal = np.linalg.solve(XtX, Xty)
print(f"Normal equations: {w_normal}")

# Method 2: QR decomposition (more stable)
Q, R = np.linalg.qr(X)
w_qr = np.linalg.solve(R, Q.T @ y)
print(f"QR method: {w_qr}")

# Method 3: SVD (most stable, handles rank-deficiency)
U, s, Vt = np.linalg.svd(X, full_matrices=False)
w_svd = Vt.T @ np.diag(1/s) @ U.T @ y
print(f"SVD method: {w_svd}")

# Method 4: Pseudoinverse
w_pinv = np.linalg.pinv(X) @ y
print(f"Pseudoinverse: {w_pinv}")

# Method 5: sklearn
lr = LinearRegression(fit_intercept=False)
lr.fit(X, y)
print(f"sklearn: {lr.coef_}")

# All methods give same result
print(f"All close: {np.allclose(w_normal, w_qr) and np.allclose(w_qr, w_svd)}")

# Ridge regression (regularization)
lambda_reg = 1.0
w_ridge = np.linalg.solve(XtX + lambda_reg * np.eye(n_features), Xty)

ridge = Ridge(alpha=lambda_reg, fit_intercept=False)
ridge.fit(X, y)
print(f"Ridge (manual): {w_ridge}")
print(f"Ridge (sklearn): {ridge.coef_}")

# Geometric interpretation: projection
y_hat = X @ w_normal  # Projection of y onto Col(X)
residual = y - y_hat   # Residual is orthogonal to Col(X)

# Verify orthogonality: X^T @ residual ≈ 0
print(f"X^T @ residual ≈ 0: {np.allclose(X.T @ residual, 0)}")
```

**Key Linear Algebra Insights:**
| Concept | Role in Linear Regression |
|---------|---------------------------|
| Matrix multiplication | X**w** computes predictions |
| Transpose | Xᵀ appears in normal equations |
| Inverse | (XᵀX)⁻¹ solves for weights |
| Projection | ŷ = X(XᵀX)⁻¹Xᵀy |
| Rank | rank(X) < n causes issues |
| Condition number | High κ(XᵀX) → unstable |

**Interview Tips:**
- Never compute (XᵀX)⁻¹ directly; use `np.linalg.solve()` or decompositions
- Ridge regression makes XᵀX + λI always invertible
- SVD-based solution handles rank-deficiency gracefully
- The residual **y** - X**w** is orthogonal to all columns of X

---

## Question 13
- [ ] Done

**How do eigenvalues and eigenvectors apply to Principal Component Analysis (PCA)?**

### Answer

**Definition:**  
PCA finds orthogonal directions (principal components) that maximize variance in data. These directions are the eigenvectors of the covariance matrix, and the eigenvalues represent the variance along each direction.

**Core Concepts:**
- Covariance matrix Σ is symmetric → real eigenvalues, orthogonal eigenvectors
- Eigenvectors = principal component directions
- Eigenvalues = variance explained along each direction
- Sort by eigenvalue (descending) to rank importance

**Mathematical Formulation:**

**Covariance matrix:**
$$\Sigma = \frac{1}{n-1}X_c^T X_c$$
where Xc is centered data (mean subtracted).

**Eigendecomposition:**
$$\Sigma \mathbf{v}_i = \lambda_i \mathbf{v}_i$$

**PCA transformation:**
$$Z = X_c V_k$$
where V_k contains top-k eigenvectors.

**Variance explained:**
$$\text{Explained variance ratio}_i = \frac{\lambda_i}{\sum_j \lambda_j}$$

**Connection to SVD:**
If X = UΣVᵀ, then:
- Columns of V = eigenvectors of XᵀX (principal components)
- Σ² / (n-1) = eigenvalues of covariance matrix

**Python Example:**
```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Generate correlated data
n_samples = 1000
X = np.random.randn(n_samples, 3)
# Introduce correlation
X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n_samples)
X[:, 2] = 0.5 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * np.random.randn(n_samples)

# Center the data
X_centered = X - X.mean(axis=0)

# Method 1: Eigendecomposition of covariance matrix
cov_matrix = np.cov(X_centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort by eigenvalue (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Eigenvalues (variance along each PC):", eigenvalues)
print("Explained variance ratio:", eigenvalues / eigenvalues.sum())

# Transform data
X_pca_manual = X_centered @ eigenvectors

# Method 2: SVD (numerically more stable)
U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
X_pca_svd = U * s  # Or: X_centered @ Vt.T

# Eigenvalues from singular values
eigenvalues_from_svd = (s ** 2) / (n_samples - 1)
print("Eigenvalues from SVD:", eigenvalues_from_svd)

# Method 3: sklearn PCA
pca = PCA(n_components=3)
X_pca_sklearn = pca.fit_transform(X_centered)

print("\nExplained variance ratio (sklearn):", pca.explained_variance_ratio_)
print("Principal components (eigenvectors):\n", pca.components_)

# Verify all methods match
print("\nMethods match:", np.allclose(np.abs(X_pca_manual), np.abs(X_pca_sklearn)))

# Dimensionality reduction: keep top 2 PCs
n_components = 2
X_reduced = X_centered @ eigenvectors[:, :n_components]
print(f"\nOriginal shape: {X.shape}, Reduced shape: {X_reduced.shape}")

# Reconstruction
X_reconstructed = X_reduced @ eigenvectors[:, :n_components].T + X.mean(axis=0)
reconstruction_error = np.mean((X - X_reconstructed) ** 2)
print(f"Reconstruction MSE: {reconstruction_error:.4f}")

# Cumulative explained variance
cumsum = np.cumsum(eigenvalues / eigenvalues.sum())
print(f"Cumulative variance: {cumsum}")
# Choose n_components where cumsum > 0.95 for 95% variance
```

**Algorithm: PCA via Eigendecomposition**
1. Center data: X_c = X - mean(X)
2. Compute covariance: Σ = X_cᵀ X_c / (n-1)
3. Eigendecompose: Σ = VΛVᵀ
4. Sort eigenvectors by eigenvalues (descending)
5. Project: Z = X_c V_k (top k eigenvectors)

**ML Applications:**
| Application | How PCA Helps |
|-------------|---------------|
| **Dimensionality reduction** | Reduce features while preserving variance |
| **Visualization** | Project to 2D/3D for plotting |
| **Noise reduction** | Remove low-variance components |
| **Feature extraction** | Decorrelated features |
| **Preprocessing** | Before clustering, classification |

**Interview Tips:**
- Use SVD for PCA in practice (numerically stable)
- PCA assumes linear relationships; use kernel PCA for non-linear
- Standardize features first if scales differ
- sklearn's PCA uses SVD internally, not eigendecomposition

---

## Question 14
- [ ] Done

**What would you consider when choosing a library for linear algebra operations?**

### Answer

**Definition:**  
Choosing the right linear algebra library depends on factors like performance requirements, hardware availability (CPU vs GPU), ease of use, numerical stability, and integration with ML frameworks. The right choice can significantly impact both development speed and runtime performance.

**Core Considerations:**
- **Performance**: BLAS/LAPACK optimization, GPU support
- **Numerical stability**: Handling ill-conditioned matrices
- **Memory efficiency**: Sparse matrix support, in-place operations
- **Ecosystem integration**: ML framework compatibility
- **Development speed**: API simplicity, debugging support

**Library Comparison:**

| Library | Strengths | Best For | Limitations |
|---------|-----------|----------|-------------|
| **NumPy** | Easy API, widespread, good for prototyping | Small-medium matrices, CPU | No GPU, limited parallelism |
| **SciPy** | Sparse matrices, specialized decompositions | Sparse data, scientific computing | Still CPU-only |
| **PyTorch** | GPU support, autograd, DL integration | Deep learning, batched ops | Overhead for simple tasks |
| **TensorFlow** | Distributed computing, production deployment | Large-scale ML, TPU support | Complex API |
| **JAX** | JIT compilation, autograd, XLA optimization | Research, custom gradients | Learning curve |
| **CuPy** | NumPy API on GPU | Drop-in GPU acceleration | NVIDIA-only |
| **Intel MKL** | Highly optimized for Intel CPUs | Production CPU systems | Vendor lock-in |

**Mathematical Considerations:**

**Condition number** (numerical stability):
$$\kappa(A) = \|A\| \cdot \|A^{-1}\|$$
High κ → need double precision or pivoting strategies.

**Computational complexity:**
- Matrix multiply: O(n³) or O(n^{2.37}) with Strassen
- SVD: O(min(mn², m²n)) for m×n matrix
- Sparse operations: O(nnz) where nnz = non-zero elements

**Python Example:**
```python
import numpy as np
import time
from scipy import sparse
from scipy.sparse.linalg import svds

# Function to benchmark operations
def benchmark(name, func, iterations=10):
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    print(f"{name}: {np.mean(times)*1000:.2f} ms ± {np.std(times)*1000:.2f} ms")

# Dense vs Sparse comparison
n = 1000
density = 0.01  # 1% non-zero

# Dense matrix
A_dense = np.random.randn(n, n)
A_dense[np.random.rand(n, n) > density] = 0

# Sparse matrix (same data)
A_sparse = sparse.csr_matrix(A_dense)

print(f"Dense memory: {A_dense.nbytes / 1e6:.2f} MB")
print(f"Sparse memory: {(A_sparse.data.nbytes + A_sparse.indices.nbytes + A_sparse.indptr.nbytes) / 1e6:.2f} MB")

# Benchmark matrix-vector multiplication
x = np.random.randn(n)
benchmark("Dense matmul", lambda: A_dense @ x)
benchmark("Sparse matmul", lambda: A_sparse @ x)

# SVD comparison for sparse
k = 10  # Top k singular values
benchmark("Dense SVD (top-k)", lambda: np.linalg.svd(A_dense, full_matrices=False)[1][:k])
benchmark("Sparse SVD (top-k)", lambda: svds(A_sparse, k=k, return_singular_vectors=False))

# Check numerical stability
def condition_number_analysis(A):
    """Analyze matrix condition for numerical stability."""
    cond = np.linalg.cond(A)
    if cond < 1e3:
        stability = "Well-conditioned"
    elif cond < 1e6:
        stability = "Moderate - use double precision"
    elif cond < 1e12:
        stability = "Ill-conditioned - consider regularization"
    else:
        stability = "Severely ill-conditioned - use SVD/pseudoinverse"
    return cond, stability

# Example ill-conditioned matrix
A_good = np.random.randn(100, 100)
A_bad = np.vander(np.linspace(0.1, 1, 20))  # Vandermonde matrices are ill-conditioned

print(f"\nRandom matrix: κ = {condition_number_analysis(A_good)[0]:.2e} ({condition_number_analysis(A_good)[1]})")
print(f"Vandermonde: κ = {condition_number_analysis(A_bad)[0]:.2e} ({condition_number_analysis(A_bad)[1]})")

# GPU example (pseudo-code for reference)
"""
# CuPy (NumPy API on GPU)
import cupy as cp
A_gpu = cp.array(A_dense)
x_gpu = cp.array(x)
result_gpu = A_gpu @ x_gpu

# PyTorch GPU
import torch
A_torch = torch.tensor(A_dense, device='cuda')
x_torch = torch.tensor(x, device='cuda')
result_torch = A_torch @ x_torch
"""

# Library selection decision tree
decision_tree = """
Library Selection Guide:
========================
1. Is your data sparse (>90% zeros)?
   YES → SciPy sparse or PyTorch sparse
   NO → Continue

2. Do you need GPU acceleration?
   YES → CuPy (simple), PyTorch/JAX (ML), TensorFlow (production)
   NO → Continue

3. Do you need automatic differentiation?
   YES → PyTorch, JAX, or TensorFlow
   NO → NumPy/SciPy

4. Is it production deployment?
   YES → Consider Intel MKL, TensorFlow Serving
   NO → NumPy is usually fine
"""
print(decision_tree)
```

**Algorithm: Library Selection Process**
1. Profile your data: size, sparsity, batch requirements
2. Identify hardware: CPU-only, NVIDIA GPU, TPU, distributed
3. Check ML framework requirements (PyTorch, TensorFlow, sklearn)
4. Benchmark critical operations with realistic data sizes
5. Consider maintenance and team expertise

**ML Applications:**
| Scenario | Recommended Libraries |
|----------|----------------------|
| **Prototyping** | NumPy + SciPy |
| **Sparse NLP features** | SciPy sparse |
| **Deep learning training** | PyTorch/TensorFlow |
| **Research with custom ops** | JAX |
| **Production inference** | ONNX Runtime, TensorRT |

**Interview Tips:**
- Know when NumPy is sufficient vs when to use specialized libraries
- Understand sparse matrix formats (CSR, CSC, COO) and their trade-offs
- Be aware of BLAS backends (OpenBLAS, MKL) that accelerate NumPy
- Mention memory vs speed trade-offs for large matrices

---

## Question 15
- [ ] Done

**How do you ensure numerical stability when performing matrix computations?**

### Answer

**Definition:**  
Numerical stability refers to algorithms that produce accurate results even with finite precision arithmetic (floating-point). Unstable algorithms amplify rounding errors, leading to catastrophic loss of precision, especially for ill-conditioned problems.

**Core Concepts:**
- **Condition number**: Sensitivity of output to input perturbations
- **Forward stability**: Small errors in output relative to exact computation
- **Backward stability**: Result is exact for slightly perturbed input
- **Catastrophic cancellation**: Loss of precision when subtracting similar numbers

**Mathematical Formulation:**

**Condition number:**
$$\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{max}}{\sigma_{min}}$$

**Error amplification:**
$$\frac{\|\Delta \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(A) \frac{\|\Delta \mathbf{b}\|}{\|\mathbf{b}\|}$$

**Machine epsilon** (float64):
$$\epsilon \approx 2.2 \times 10^{-16}$$

**Digits of accuracy lost:**
$$\text{digits lost} \approx \log_{10}(\kappa(A))$$

**Stability Strategies:**

| Problem | Unstable Approach | Stable Approach |
|---------|-------------------|-----------------|
| **Solve Ax = b** | A⁻¹b (explicit inverse) | LU with pivoting, QR |
| **Least squares** | (AᵀA)⁻¹Aᵀb (normal equations) | QR decomposition, SVD |
| **Eigenvalues** | det(A - λI) = 0 | QR algorithm |
| **Matrix exponential** | Taylor series | Padé approximation + scaling |

**Python Example:**
```python
import numpy as np
from scipy import linalg

np.set_printoptions(precision=15)

# ========================================
# 1. CONDITION NUMBER AND ERROR ANALYSIS
# ========================================

def analyze_stability(A, b):
    """Analyze numerical stability of solving Ax = b."""
    cond = np.linalg.cond(A)
    machine_eps = np.finfo(float).eps
    
    # Solve the system
    x = np.linalg.solve(A, b)
    
    # Check residual
    residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    
    # Estimated relative error bound
    error_bound = cond * machine_eps
    
    print(f"Condition number: {cond:.2e}")
    print(f"Machine epsilon: {machine_eps:.2e}")
    print(f"Relative residual: {residual:.2e}")
    print(f"Error bound: {error_bound:.2e}")
    print(f"Digits of accuracy: ~{max(0, 16 - np.log10(cond)):.1f}")
    
    return x, cond

# Well-conditioned system
A_good = np.array([[4, 1], [1, 3]], dtype=float)
b = np.array([1, 2], dtype=float)
print("=== Well-conditioned system ===")
x_good, _ = analyze_stability(A_good, b)

# Ill-conditioned system (Hilbert matrix)
n = 10
A_bad = linalg.hilbert(n)  # Notoriously ill-conditioned
b_bad = np.ones(n)
print("\n=== Hilbert matrix (ill-conditioned) ===")
x_bad, _ = analyze_stability(A_bad, b_bad)

# ========================================
# 2. STABLE VS UNSTABLE ALGORITHMS
# ========================================

print("\n=== Least Squares: Normal Equations vs QR ===")

# Create a problem with collinear columns
np.random.seed(42)
m, n = 100, 5
A = np.random.randn(m, n)
A[:, 4] = A[:, 3] + 1e-10 * np.random.randn(m)  # Near-collinear
b = np.random.randn(m)

# Method 1: Normal equations (UNSTABLE for ill-conditioned A)
try:
    ATA = A.T @ A
    ATb = A.T @ b
    x_normal = np.linalg.solve(ATA, ATb)
    print(f"Normal equations κ(AᵀA) = {np.linalg.cond(ATA):.2e}")
except np.linalg.LinAlgError:
    print("Normal equations failed!")
    x_normal = None

# Method 2: QR decomposition (STABLE)
Q, R = np.linalg.qr(A)
x_qr = linalg.solve_triangular(R, Q.T @ b)

# Method 3: SVD with truncation (MOST STABLE)
U, s, Vt = np.linalg.svd(A, full_matrices=False)
threshold = 1e-10 * s[0]  # Truncate small singular values
s_inv = np.where(s > threshold, 1/s, 0)
x_svd = (Vt.T * s_inv) @ (U.T @ b)

# Compare residuals
if x_normal is not None:
    print(f"Normal eq. residual: {np.linalg.norm(A @ x_normal - b):.6e}")
print(f"QR residual: {np.linalg.norm(A @ x_qr - b):.6e}")
print(f"SVD residual: {np.linalg.norm(A @ x_svd - b):.6e}")

# ========================================
# 3. PIVOTING FOR STABILITY
# ========================================

print("\n=== LU Decomposition with/without Pivoting ===")

# Matrix where pivoting matters
A_pivot = np.array([[1e-20, 1], [1, 1]], dtype=float)
b_pivot = np.array([1, 2], dtype=float)

# Without pivoting (if we could - numpy always pivots)
# Simulating what would happen
print(f"Without pivoting, 1/a[0,0] = {1/A_pivot[0,0]:.2e} → overflow risk")

# With partial pivoting (standard)
P, L, U = linalg.lu(A_pivot)
x_lu = np.linalg.solve(A_pivot, b_pivot)
print(f"With pivoting solution: {x_lu}")
print(f"Residual: {np.linalg.norm(A_pivot @ x_lu - b_pivot):.2e}")

# ========================================
# 4. CATASTROPHIC CANCELLATION
# ========================================

print("\n=== Catastrophic Cancellation Example ===")

def quadratic_unstable(a, b, c):
    """Standard formula - unstable for b² >> 4ac."""
    disc = np.sqrt(b**2 - 4*a*c)
    x1 = (-b + disc) / (2*a)
    x2 = (-b - disc) / (2*a)
    return x1, x2

def quadratic_stable(a, b, c):
    """Stable formula using alternative form."""
    disc = np.sqrt(b**2 - 4*a*c)
    if b >= 0:
        x1 = (-b - disc) / (2*a)
        x2 = (2*c) / (-b - disc)  # Avoids subtraction
    else:
        x1 = (-b + disc) / (2*a)
        x2 = (2*c) / (-b + disc)
    return x1, x2

# Case where b² >> 4ac
a, b, c = 1, 1e8, 1
x1_u, x2_u = quadratic_unstable(a, b, c)
x1_s, x2_s = quadratic_stable(a, b, c)

print(f"Unstable: x1={x1_u:.15e}, x2={x2_u:.15e}")
print(f"Stable:   x1={x1_s:.15e}, x2={x2_s:.15e}")
print(f"True x2 ≈ -c/b = {-c/b:.15e}")

# ========================================
# 5. REGULARIZATION FOR STABILITY
# ========================================

print("\n=== Tikhonov Regularization ===")

# Ill-conditioned least squares
A_ill = linalg.hilbert(8)[:, :6]
b_ill = np.ones(8)

# Unregularized (unstable)
x_unreg = np.linalg.lstsq(A_ill, b_ill, rcond=None)[0]

# Regularized (stable)
lambda_reg = 1e-6
x_reg = np.linalg.solve(A_ill.T @ A_ill + lambda_reg * np.eye(6), A_ill.T @ b_ill)

print(f"Unregularized ||x||: {np.linalg.norm(x_unreg):.2e}")
print(f"Regularized ||x||: {np.linalg.norm(x_reg):.2e}")
```

**Algorithm: Numerically Stable Least Squares**
1. Compute thin SVD: A = UΣVᵀ
2. Set threshold τ = ε · σ_max (ε = machine precision)
3. For each σᵢ: σᵢ⁺ = 1/σᵢ if σᵢ > τ, else 0
4. Compute x = V · diag(σ⁺) · Uᵀ · b

**Best Practices:**
| Practice | Rationale |
|----------|-----------|
| **Use QR/SVD, not normal equations** | κ(AᵀA) = κ(A)² |
| **Always use pivoting in LU** | Prevents division by small numbers |
| **Check condition number first** | Know if problem is solvable |
| **Use double precision** | 16 digits vs 7 (single) |
| **Regularize ill-conditioned problems** | Ridge regression, truncated SVD |

**Interview Tips:**
- Mention condition number as key metric
- Know why normal equations are unstable (squaring condition number)
- Explain pivoting's role in LU decomposition
- Discuss trade-offs: stability vs computational cost
- Real-world: sklearn's `LinearRegression` uses SVD for stability

---

## Question 16
- [ ] Done

**How do graph theory and linear algebra intersect in machine learning?**

### Answer

**Definition:**  
Graphs are naturally represented as matrices (adjacency, Laplacian), enabling powerful linear algebra techniques for graph analysis. This intersection powers graph neural networks, spectral clustering, PageRank, and network analysis in ML.

**Core Concepts:**
- **Adjacency matrix** (A): A[i,j] = 1 if edge from i to j
- **Degree matrix** (D): Diagonal with D[i,i] = degree of node i
- **Laplacian matrix** (L): L = D - A (unnormalized)
- **Normalized Laplacian**: L_norm = I - D^(-1/2) A D^(-1/2)
- **Graph Fourier Transform**: Eigendecomposition of Laplacian

**Mathematical Formulation:**

**Adjacency matrix:**
$$A_{ij} = \begin{cases} w_{ij} & \text{if edge } (i,j) \text{ exists} \\ 0 & \text{otherwise} \end{cases}$$

**Graph Laplacian:**
$$L = D - A$$

**Normalized Laplacian:**
$$L_{norm} = I - D^{-1/2}AD^{-1/2}$$

**Spectral properties:**
- L is positive semi-definite (all eigenvalues ≥ 0)
- Number of zero eigenvalues = number of connected components
- Second smallest eigenvalue (Fiedler value) = algebraic connectivity

**Random walk transition:**
$$P = D^{-1}A \quad (\text{row stochastic})$$

**Python Example:**
```python
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import networkx as nx

np.random.seed(42)

# ========================================
# 1. GRAPH MATRIX REPRESENTATIONS
# ========================================

# Create a simple graph
edges = [(0,1), (0,2), (1,2), (1,3), (2,3), (3,4), (4,5), (5,6), (4,6)]
n_nodes = 7

# Adjacency matrix
A = np.zeros((n_nodes, n_nodes))
for i, j in edges:
    A[i, j] = 1
    A[j, i] = 1  # Undirected

# Degree matrix
D = np.diag(A.sum(axis=1))

# Graph Laplacian
L = D - A

# Normalized Laplacian
D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D)))
L_norm = np.eye(n_nodes) - D_inv_sqrt @ A @ D_inv_sqrt

print("Adjacency matrix A:\n", A)
print("\nDegree matrix diagonal:", np.diag(D))
print("\nLaplacian L:\n", L)

# ========================================
# 2. SPECTRAL CLUSTERING
# ========================================

print("\n=== Spectral Clustering ===")

# Generate two clusters with connecting bridge
from sklearn.datasets import make_moons
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans

X, y_true = make_moons(n_samples=200, noise=0.05, random_state=42)

# Build k-NN graph
k = 10
A_knn = kneighbors_graph(X, k, mode='connectivity', include_self=False)
A_knn = 0.5 * (A_knn + A_knn.T)  # Make symmetric

# Compute normalized Laplacian
D_knn = sparse.diags(np.array(A_knn.sum(axis=1)).flatten())
D_inv_sqrt = sparse.diags(1 / np.sqrt(np.array(A_knn.sum(axis=1)).flatten() + 1e-10))
L_norm_knn = sparse.eye(A_knn.shape[0]) - D_inv_sqrt @ A_knn @ D_inv_sqrt

# Get bottom k eigenvectors (excluding λ=0)
n_clusters = 2
eigenvalues, eigenvectors = eigsh(L_norm_knn, k=n_clusters+1, which='SM')
print(f"Smallest eigenvalues: {eigenvalues}")

# Use eigenvectors for clustering (skip first which is constant)
embedding = eigenvectors[:, 1:n_clusters+1]

# Normalize rows
embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

# Cluster in spectral embedding space
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(embedding)

# Compare with true labels
from sklearn.metrics import adjusted_rand_score
print(f"Spectral clustering ARI: {adjusted_rand_score(y_true, labels):.3f}")

# ========================================
# 3. PAGERANK ALGORITHM
# ========================================

print("\n=== PageRank ===")

def pagerank(A, damping=0.85, max_iter=100, tol=1e-6):
    """Compute PageRank using power iteration."""
    n = A.shape[0]
    
    # Row normalize (transition matrix)
    out_degree = A.sum(axis=1)
    out_degree[out_degree == 0] = 1  # Handle dangling nodes
    P = A / out_degree[:, np.newaxis]
    
    # PageRank iteration
    # r = d * P^T @ r + (1-d) / n
    r = np.ones(n) / n
    
    for i in range(max_iter):
        r_new = damping * (P.T @ r) + (1 - damping) / n
        if np.linalg.norm(r_new - r) < tol:
            print(f"Converged in {i+1} iterations")
            break
        r = r_new
    
    return r

# Simple web graph
web_edges = [(0,1), (0,2), (1,2), (2,0), (2,3), (3,0)]
A_web = np.zeros((4, 4))
for i, j in web_edges:
    A_web[i, j] = 1

ranks = pagerank(A_web)
print(f"PageRank scores: {ranks}")
print(f"Most important node: {np.argmax(ranks)}")

# Compare with networkx
G = nx.DiGraph()
G.add_edges_from(web_edges)
nx_ranks = nx.pagerank(G, alpha=0.85)
print(f"NetworkX PageRank: {list(nx_ranks.values())}")

# ========================================
# 4. GRAPH NEURAL NETWORK PROPAGATION
# ========================================

print("\n=== GNN Message Passing ===")

def gcn_layer(A, X, W):
    """
    Graph Convolutional Network layer.
    H = σ(Ã X W) where Ã = D̃^(-1/2) Â D̃^(-1/2)
    Â = A + I (add self-loops)
    """
    n = A.shape[0]
    
    # Add self-loops
    A_hat = A + np.eye(n)
    
    # Normalize
    D_hat = np.diag(A_hat.sum(axis=1))
    D_hat_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D_hat)))
    A_norm = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt
    
    # Propagate and transform
    H = A_norm @ X @ W
    
    # ReLU activation
    return np.maximum(0, H)

# Node features
n_nodes = 7
n_features = 4
n_hidden = 3

X = np.random.randn(n_nodes, n_features)
W = np.random.randn(n_features, n_hidden) * 0.1

# One GCN layer
H = gcn_layer(A, X, W)
print(f"Input shape: {X.shape}")
print(f"Output shape after GCN layer: {H.shape}")

# ========================================
# 5. GRAPH PROPERTIES FROM EIGENVALUES
# ========================================

print("\n=== Spectral Graph Properties ===")

eigenvalues_L, eigenvectors_L = np.linalg.eigh(L)

print(f"Laplacian eigenvalues: {eigenvalues_L}")
print(f"Number of connected components: {np.sum(eigenvalues_L < 1e-10)}")
print(f"Algebraic connectivity (Fiedler value): {eigenvalues_L[1]:.4f}")

# Fiedler vector for graph partitioning
fiedler_vector = eigenvectors_L[:, 1]
partition = fiedler_vector > 0
print(f"Fiedler vector: {fiedler_vector}")
print(f"Graph partition: {partition.astype(int)}")
```

**Algorithm: Spectral Clustering**
1. Build similarity graph (k-NN or ε-neighborhood)
2. Compute normalized Laplacian: L = I - D^(-1/2)AD^(-1/2)
3. Find k smallest eigenvectors of L
4. Form matrix U ∈ R^(n×k) from eigenvectors
5. Normalize rows of U to unit length
6. Apply k-means to rows of U

**ML Applications:**
| Application | Linear Algebra Technique |
|-------------|--------------------------|
| **Spectral clustering** | Laplacian eigenvectors |
| **PageRank / link analysis** | Power iteration on transition matrix |
| **Graph neural networks** | Normalized adjacency multiplication |
| **Community detection** | Modularity matrix eigendecomposition |
| **Node embeddings** | Random walk transition matrices |
| **Knowledge graphs** | Tensor factorization |

**Key Connections:**
- GCN: Smoothing via Laplacian → feature propagation
- DeepWalk/Node2Vec: Random walk → implicit matrix factorization
- Attention in GAT: Weighted adjacency with learned weights

**Interview Tips:**
- Know the relationship: L = D - A and its normalized forms
- Explain why Laplacian eigenvectors reveal cluster structure
- Connect GCN to spectral graph convolutions
- Mention that GNN aggregation is matrix multiplication
- Know PageRank is finding principal eigenvector of modified adjacency

---

## Question 17
- [ ] Done

**How do you evaluate and choose dimensionality reduction techniques?**

### Answer

**Definition:**  
Dimensionality reduction techniques reduce the number of features while preserving important structure. Choosing the right technique depends on data characteristics (linear vs non-linear), task requirements (visualization vs preprocessing), and computational constraints.

**Core Concepts:**
- **Linear methods**: PCA, Factor Analysis, LDA (preserve global structure)
- **Non-linear methods**: t-SNE, UMAP, Autoencoders (preserve local structure)
- **Supervised vs unsupervised**: LDA uses labels; PCA doesn't
- **Global vs local preservation**: PCA (global variance); t-SNE (local neighborhoods)

**Mathematical Comparison:**

| Method | Objective | Mathematical Formulation |
|--------|-----------|--------------------------|
| **PCA** | Max variance | $\max_W \text{Var}(XW) = W^T \Sigma W$ |
| **LDA** | Max class separation | $\max_W \frac{W^T S_B W}{W^T S_W W}$ |
| **t-SNE** | Preserve neighborhoods | $\min \sum_i KL(P_i \| Q_i)$ |
| **UMAP** | Preserve fuzzy topology | Cross-entropy on fuzzy sets |

**PCA variance:**
$$\text{maximize } \mathbf{w}^T \Sigma \mathbf{w} \text{ subject to } \|\mathbf{w}\|=1$$

**LDA objective (Fisher's criterion):**
$$J(\mathbf{w}) = \frac{\mathbf{w}^T S_B \mathbf{w}}{\mathbf{w}^T S_W \mathbf{w}}$$

where S_B = between-class scatter, S_W = within-class scatter.

**t-SNE probability:**
$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$

**Python Example:**
```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
import time

np.random.seed(42)

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target
X_scaled = StandardScaler().fit_transform(X)

print(f"Original dimensions: {X.shape}")
print(f"Classes: {len(np.unique(y))}")

# ========================================
# 1. COMPARISON OF METHODS
# ========================================

methods = {}

# PCA
start = time.time()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
methods['PCA'] = {
    'embedding': X_pca,
    'time': time.time() - start,
    'variance_explained': pca.explained_variance_ratio_.sum()
}

# t-SNE
start = time.time()
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)
methods['t-SNE'] = {
    'embedding': X_tsne,
    'time': time.time() - start
}

# LDA (supervised)
start = time.time()
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)
methods['LDA'] = {
    'embedding': X_lda,
    'time': time.time() - start,
    'variance_explained': lda.explained_variance_ratio_.sum()
}

# Try UMAP if available
try:
    import umap
    start = time.time()
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    methods['UMAP'] = {
        'embedding': X_umap,
        'time': time.time() - start
    }
except ImportError:
    print("UMAP not installed")

# ========================================
# 2. EVALUATION METRICS
# ========================================

print("\n=== Evaluation Metrics ===")

def evaluate_embedding(X_embed, y, name):
    """Evaluate quality of embedding."""
    # Silhouette score (cluster separation)
    silhouette = silhouette_score(X_embed, y)
    
    # Trustworthiness (neighborhood preservation) - manual implementation
    # How well local neighborhoods are preserved
    from sklearn.manifold import trustworthiness
    trust = trustworthiness(X_scaled, X_embed, n_neighbors=15)
    
    # Classification accuracy in reduced space
    knn = KNeighborsClassifier(n_neighbors=5)
    acc = cross_val_score(knn, X_embed, y, cv=5).mean()
    
    print(f"{name}:")
    print(f"  Silhouette Score: {silhouette:.3f}")
    print(f"  Trustworthiness: {trust:.3f}")
    print(f"  5-NN Accuracy: {acc:.3f}")
    print(f"  Computation Time: {methods[name]['time']:.2f}s")
    if 'variance_explained' in methods[name]:
        print(f"  Variance Explained: {methods[name]['variance_explained']:.3f}")
    
    return {'silhouette': silhouette, 'trust': trust, 'accuracy': acc}

results = {}
for name in methods:
    results[name] = evaluate_embedding(methods[name]['embedding'], y, name)

# ========================================
# 3. CHOOSING NUMBER OF COMPONENTS (PCA)
# ========================================

print("\n=== PCA: Choosing n_components ===")

pca_full = PCA().fit(X_scaled)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)

# Find n_components for different variance thresholds
thresholds = [0.80, 0.90, 0.95, 0.99]
for thresh in thresholds:
    n_comp = np.argmax(cumsum >= thresh) + 1
    print(f"  {thresh*100:.0f}% variance: {n_comp} components")

# Kaiser criterion: eigenvalue > 1
eigenvalues = pca_full.explained_variance_
n_kaiser = np.sum(eigenvalues > 1)
print(f"  Kaiser criterion (λ > 1): {n_kaiser} components")

# ========================================
# 4. LINEAR VS NON-LINEAR DATA
# ========================================

print("\n=== Linear vs Non-linear Data ===")

# Generate Swiss Roll (inherently non-linear)
X_swiss, color_swiss = make_swiss_roll(n_samples=1000, random_state=42)

# PCA on Swiss Roll
pca_swiss = PCA(n_components=2)
X_swiss_pca = pca_swiss.fit_transform(X_swiss)

# t-SNE on Swiss Roll
tsne_swiss = TSNE(n_components=2, random_state=42)
X_swiss_tsne = tsne_swiss.fit_transform(X_swiss)

# Measure structure preservation
# For Swiss Roll, we know the true 2D structure (t parameter)
# Good method should preserve ordering along the roll

def measure_ordering_preservation(X_embed, color):
    """Check if embedding preserves the roll ordering."""
    from scipy.stats import spearmanr
    # Use distance to centroid in embedding vs original color
    centroid = X_embed.mean(axis=0)
    dist_to_centroid = np.linalg.norm(X_embed - centroid, axis=1)
    corr, _ = spearmanr(dist_to_centroid, color)
    return abs(corr)

print(f"PCA ordering preservation: {measure_ordering_preservation(X_swiss_pca, color_swiss):.3f}")
print(f"t-SNE ordering preservation: {measure_ordering_preservation(X_swiss_tsne, color_swiss):.3f}")

# ========================================
# 5. DECISION FRAMEWORK
# ========================================

decision_guide = """
=== Dimensionality Reduction Decision Guide ===

1. What is your goal?
   - Visualization (2D/3D) → t-SNE, UMAP
   - Preprocessing for ML → PCA, LDA
   - Feature extraction → PCA, Autoencoders
   - Noise reduction → PCA (remove low-variance components)

2. Is your data linear or non-linear?
   - Linear relationships → PCA
   - Non-linear manifolds → t-SNE, UMAP, Kernel PCA

3. Do you have labels?
   - No → PCA, t-SNE, UMAP
   - Yes → LDA (if linear), UMAP supervised

4. What's your computational budget?
   - Fast → PCA O(min(n²d, nd²))
   - Slow OK → t-SNE O(n²), UMAP O(n^1.14)

5. Need interpretability?
   - Yes → PCA (loadings interpretable)
   - No → t-SNE, UMAP, Autoencoders

6. Need to transform new data?
   - Yes → PCA, LDA, UMAP
   - No → t-SNE (no out-of-sample)
"""
print(decision_guide)

# ========================================
# 6. RECONSTRUCTION ERROR (for PCA)
# ========================================

print("=== Reconstruction Error Analysis ===")

for n_comp in [5, 10, 20, 30, 50]:
    pca_n = PCA(n_components=n_comp)
    X_reduced = pca_n.fit_transform(X_scaled)
    X_reconstructed = pca_n.inverse_transform(X_reduced)
    mse = np.mean((X_scaled - X_reconstructed) ** 2)
    print(f"n_components={n_comp}: MSE={mse:.4f}, Variance={pca_n.explained_variance_ratio_.sum():.3f}")
```

**Algorithm: Method Selection**
1. Determine goal (visualization vs preprocessing)
2. Check data linearity (plot, try both)
3. Consider label availability
4. Evaluate computational constraints
5. Test multiple methods with metrics
6. Validate on downstream task

**Comparison Summary:**

| Criterion | PCA | LDA | t-SNE | UMAP |
|-----------|-----|-----|-------|------|
| **Supervision** | No | Yes | No | Both |
| **Linearity** | Linear | Linear | Non-linear | Non-linear |
| **Scalability** | Excellent | Good | Poor | Good |
| **New data** | Yes | Yes | No* | Yes |
| **Interpretable** | Yes | Yes | No | No |
| **Global structure** | ✓ | ✓ | ✗ | Partial |
| **Local structure** | ✗ | ✗ | ✓ | ✓ |

*t-SNE has parametric variants for new data

**Interview Tips:**
- Know trade-offs: PCA (linear, fast, interpretable) vs t-SNE (non-linear, slow, visual)
- Explain variance explained vs trustworthiness
- Mention t-SNE hyperparameters (perplexity) affect results significantly
- Know UMAP advantages: faster than t-SNE, preserves more global structure
- Real practice: Try PCA first, use t-SNE/UMAP for visualization

---

## Question 18
- [ ] Done

**Design a linear algebra solution for a collaborative filtering problem in a movie recommendation system.**

### Answer

**Definition:**  
Collaborative filtering predicts user preferences by finding patterns in user-item interactions. Matrix factorization decomposes the sparse ratings matrix R ≈ UV^T, where U captures user latent factors and V captures item latent factors, enabling predictions for unobserved entries.

**Core Concepts:**
- **Ratings matrix R**: m users × n items (sparse, mostly missing)
- **Latent factors**: Hidden features explaining preferences (e.g., genre affinity)
- **Matrix factorization**: R ≈ UV^T where U ∈ R^(m×k), V ∈ R^(n×k)
- **SVD-based methods**: Truncated SVD on imputed or centered matrix
- **Optimization**: Minimize reconstruction error on known ratings

**Mathematical Formulation:**

**Basic matrix factorization:**
$$\min_{U, V} \sum_{(i,j) \in \Omega} (R_{ij} - \mathbf{u}_i^T \mathbf{v}_j)^2 + \lambda(\|U\|_F^2 + \|V\|_F^2)$$

where Ω = set of observed ratings, λ = regularization.

**With biases:**
$$\hat{R}_{ij} = \mu + b_i + c_j + \mathbf{u}_i^T \mathbf{v}_j$$

- μ = global mean rating
- b_i = user i's bias (tendency to rate high/low)
- c_j = item j's bias (generally liked/disliked)

**SVD approach:**
$$R \approx U_k \Sigma_k V_k^T$$

**Prediction:**
$$\hat{r}_{ij} = \mathbf{u}_i^T \mathbf{v}_j$$

**Python Example:**
```python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# ========================================
# 1. CREATE SYNTHETIC MOVIE RATINGS DATA
# ========================================

n_users = 500
n_movies = 200
n_ratings = 10000  # Sparse: ~10% filled

# True latent factors (hidden)
k_true = 5
U_true = np.random.randn(n_users, k_true)
V_true = np.random.randn(n_movies, k_true)
R_true = U_true @ V_true.T

# Add noise and clip to 1-5 rating scale
R_true = np.clip(R_true + 0.5 * np.random.randn(n_users, n_movies), 1, 5)

# Create sparse observations (simulating missing data)
observed_mask = np.zeros((n_users, n_movies), dtype=bool)
indices = np.random.choice(n_users * n_movies, n_ratings, replace=False)
observed_mask.flat[indices] = True

R_observed = np.where(observed_mask, R_true, np.nan)
print(f"Ratings matrix shape: {R_observed.shape}")
print(f"Sparsity: {100 * (1 - n_ratings / (n_users * n_movies)):.1f}% missing")

# ========================================
# 2. SVD-BASED COLLABORATIVE FILTERING
# ========================================

print("\n=== SVD-based Approach ===")

def svd_collaborative_filtering(R, k, method='mean_impute'):
    """
    SVD-based collaborative filtering.
    
    Args:
        R: Ratings matrix with np.nan for missing values
        k: Number of latent factors
        method: How to handle missing values
    """
    mask = ~np.isnan(R)
    
    if method == 'mean_impute':
        # Impute with global mean
        global_mean = np.nanmean(R)
        R_filled = np.where(np.isnan(R), global_mean, R)
        
        # Center the data
        user_means = np.nanmean(R, axis=1, keepdims=True)
        user_means = np.nan_to_num(user_means, nan=global_mean)
        R_centered = R_filled - user_means
        
    elif method == 'zero_impute':
        R_filled = np.nan_to_num(R, nan=0)
        R_centered = R_filled
        user_means = np.zeros((R.shape[0], 1))
    
    # Truncated SVD
    U, sigma, Vt = svds(csr_matrix(R_centered), k=k)
    
    # Reconstruct
    R_pred = U @ np.diag(sigma) @ Vt + user_means
    R_pred = np.clip(R_pred, 1, 5)
    
    return R_pred, U @ np.diag(np.sqrt(sigma)), (np.diag(np.sqrt(sigma)) @ Vt).T

R_pred_svd, U_svd, V_svd = svd_collaborative_filtering(R_observed, k=10)

# Evaluate on held-out ratings
test_mask = observed_mask & (np.random.rand(n_users, n_movies) > 0.8)  # 20% test
train_mask = observed_mask & ~test_mask

rmse_svd = np.sqrt(mean_squared_error(
    R_true[test_mask],
    R_pred_svd[test_mask]
))
print(f"SVD RMSE: {rmse_svd:.4f}")

# ========================================
# 3. ALTERNATING LEAST SQUARES (ALS)
# ========================================

print("\n=== Alternating Least Squares ===")

def als_collaborative_filtering(R, k, lambda_reg=0.1, n_iter=20):
    """
    ALS matrix factorization for collaborative filtering.
    
    Alternates between:
    - Fix V, solve for U: u_i = (V^T V + λI)^{-1} V^T r_i
    - Fix U, solve for V: v_j = (U^T U + λI)^{-1} U^T r_j
    """
    n_users, n_items = R.shape
    mask = ~np.isnan(R)
    R_filled = np.nan_to_num(R, nan=0)
    
    # Initialize factors randomly
    U = np.random.randn(n_users, k) * 0.1
    V = np.random.randn(n_items, k) * 0.1
    
    for iteration in range(n_iter):
        # Update U (user factors)
        VTV = V.T @ V + lambda_reg * np.eye(k)
        for i in range(n_users):
            # Only use observed ratings for user i
            rated_items = mask[i, :]
            if rated_items.sum() > 0:
                V_i = V[rated_items, :]
                r_i = R_filled[i, rated_items]
                U[i, :] = np.linalg.solve(V_i.T @ V_i + lambda_reg * np.eye(k), V_i.T @ r_i)
        
        # Update V (item factors)
        UTU = U.T @ U + lambda_reg * np.eye(k)
        for j in range(n_items):
            # Only use observed ratings for item j
            rated_users = mask[:, j]
            if rated_users.sum() > 0:
                U_j = U[rated_users, :]
                r_j = R_filled[rated_users, j]
                V[j, :] = np.linalg.solve(U_j.T @ U_j + lambda_reg * np.eye(k), U_j.T @ r_j)
        
        # Compute loss
        R_pred = U @ V.T
        loss = np.sum((R_filled[mask] - R_pred[mask])**2)
        loss += lambda_reg * (np.sum(U**2) + np.sum(V**2))
        
        if iteration % 5 == 0:
            rmse = np.sqrt(mean_squared_error(R_filled[mask], R_pred[mask]))
            print(f"Iteration {iteration}: Loss = {loss:.2f}, RMSE = {rmse:.4f}")
    
    return np.clip(U @ V.T, 1, 5), U, V

R_pred_als, U_als, V_als = als_collaborative_filtering(R_observed, k=10, lambda_reg=0.1, n_iter=20)

rmse_als = np.sqrt(mean_squared_error(R_true[test_mask], R_pred_als[test_mask]))
print(f"ALS RMSE: {rmse_als:.4f}")

# ========================================
# 4. STOCHASTIC GRADIENT DESCENT (SGD)
# ========================================

print("\n=== SGD Matrix Factorization ===")

def sgd_matrix_factorization(R, k, learning_rate=0.01, lambda_reg=0.1, n_epochs=50):
    """
    SGD for matrix factorization with biases.
    
    r_ij = mu + b_i + c_j + u_i^T v_j
    """
    n_users, n_items = R.shape
    mask = ~np.isnan(R)
    
    # Global mean
    mu = np.nanmean(R)
    
    # Initialize
    U = np.random.randn(n_users, k) * 0.1
    V = np.random.randn(n_items, k) * 0.1
    b = np.zeros(n_users)  # User biases
    c = np.zeros(n_items)  # Item biases
    
    # Get observed indices
    user_idx, item_idx = np.where(mask)
    ratings = R[mask]
    
    for epoch in range(n_epochs):
        # Shuffle
        perm = np.random.permutation(len(ratings))
        
        total_loss = 0
        for idx in perm:
            i, j = user_idx[idx], item_idx[idx]
            r_ij = ratings[idx]
            
            # Prediction
            pred = mu + b[i] + c[j] + U[i, :] @ V[j, :]
            error = r_ij - pred
            total_loss += error**2
            
            # Update biases
            b[i] += learning_rate * (error - lambda_reg * b[i])
            c[j] += learning_rate * (error - lambda_reg * c[j])
            
            # Update factors
            U_old = U[i, :].copy()
            U[i, :] += learning_rate * (error * V[j, :] - lambda_reg * U[i, :])
            V[j, :] += learning_rate * (error * U_old - lambda_reg * V[j, :])
        
        if epoch % 10 == 0:
            rmse = np.sqrt(total_loss / len(ratings))
            print(f"Epoch {epoch}: RMSE = {rmse:.4f}")
    
    # Final predictions
    R_pred = mu + b[:, np.newaxis] + c[np.newaxis, :] + U @ V.T
    return np.clip(R_pred, 1, 5), U, V, mu, b, c

R_pred_sgd, U_sgd, V_sgd, mu, b, c = sgd_matrix_factorization(
    R_observed, k=10, learning_rate=0.01, lambda_reg=0.1, n_epochs=50
)

rmse_sgd = np.sqrt(mean_squared_error(R_true[test_mask], R_pred_sgd[test_mask]))
print(f"SGD with biases RMSE: {rmse_sgd:.4f}")

# ========================================
# 5. MAKING RECOMMENDATIONS
# ========================================

print("\n=== Generating Recommendations ===")

def get_top_recommendations(user_id, R_pred, R_observed, n=5):
    """Get top N recommendations for a user."""
    # Find unrated movies
    unrated = np.isnan(R_observed[user_id, :])
    
    # Get predictions for unrated movies
    predictions = R_pred[user_id, :]
    predictions[~unrated] = -np.inf  # Exclude already rated
    
    # Top N
    top_indices = np.argsort(predictions)[::-1][:n]
    top_scores = predictions[top_indices]
    
    return list(zip(top_indices, top_scores))

# Example recommendations
user_id = 0
recommendations = get_top_recommendations(user_id, R_pred_sgd, R_observed, n=5)
print(f"Top 5 recommendations for user {user_id}:")
for movie_id, score in recommendations:
    print(f"  Movie {movie_id}: Predicted rating = {score:.2f}")

# ========================================
# 6. FINDING SIMILAR ITEMS
# ========================================

print("\n=== Item Similarity (using latent factors) ===")

def find_similar_items(item_id, V, n=5):
    """Find similar items using cosine similarity in latent space."""
    item_vector = V[item_id, :]
    
    # Cosine similarity
    similarities = V @ item_vector / (np.linalg.norm(V, axis=1) * np.linalg.norm(item_vector) + 1e-10)
    
    # Top N (excluding itself)
    similarities[item_id] = -np.inf
    top_indices = np.argsort(similarities)[::-1][:n]
    
    return list(zip(top_indices, similarities[top_indices]))

similar_movies = find_similar_items(0, V_sgd, n=5)
print(f"Movies similar to Movie 0:")
for movie_id, sim in similar_movies:
    print(f"  Movie {movie_id}: Similarity = {sim:.3f}")

# Compare methods
print("\n=== Method Comparison ===")
print(f"SVD RMSE:       {rmse_svd:.4f}")
print(f"ALS RMSE:       {rmse_als:.4f}")
print(f"SGD+bias RMSE:  {rmse_sgd:.4f}")
```

**Algorithm: ALS for Matrix Factorization**
1. Initialize U, V randomly
2. Repeat until convergence:
   - Fix V, update each u_i: u_i = (V_Ω^T V_Ω + λI)^(-1) V_Ω^T r_i
   - Fix U, update each v_j: v_j = (U_Ω^T U_Ω + λI)^(-1) U_Ω^T r_j
3. Predict: r̂_ij = u_i^T v_j

**Method Comparison:**

| Method | Pros | Cons | Use When |
|--------|------|------|----------|
| **SVD** | Fast, closed-form | Needs imputation | Dense-ish data |
| **ALS** | Parallelizable, handles implicit | Memory intensive | Large-scale, implicit feedback |
| **SGD** | Memory efficient, online | Slower convergence | Streaming data |
| **NMF** | Non-negative, interpretable | Local optima | When non-negativity matters |

**ML Applications:**
| Application | Linear Algebra Technique |
|-------------|--------------------------|
| **Netflix Prize** | SVD++, matrix factorization |
| **Implicit feedback** | Weighted ALS |
| **Cold start** | Hybrid with content features |
| **Real-time updates** | Online SGD |

**Interview Tips:**
- Know the difference between explicit (ratings) and implicit (clicks) feedback
- Explain the cold-start problem and hybrid approaches
- Mention SVD++ which incorporates implicit feedback
- Know ALS is preferred for implicit feedback (Spotify, etc.)
- Real systems: Add biases, use BPR loss for ranking
