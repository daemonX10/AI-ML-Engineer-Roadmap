# Linear Algebra Interview Questions - Coding Questions

## Question 1
- [x] Done

**Write code to add, subtract, and multiply two matrices without using external libraries.**

### Answer

**Definition:**  
Matrix operations (add, subtract, multiply) are element-wise or row-column computations on 2D arrays. Addition/subtraction require same dimensions; multiplication requires inner dimensions to match.

**Core Concepts:**
- Addition/Subtraction: Element-wise, requires same shape (m×n)
- Multiplication: (m×n) × (n×p) → (m×p), each element = dot product of row and column
- Time: O(mn) for add/sub, O(mnp) for multiply

**Mathematical Formulation:**
$$(A \pm B)_{ij} = A_{ij} \pm B_{ij}$$
$$(AB)_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}$$

**Python Code:**
```python
# Pipeline: Check dimensions → Initialize result → Fill element-by-element

def matrix_add(A, B):
    rows, cols = len(A), len(A[0])
    result = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] + B[i][j]
    return result

def matrix_subtract(A, B):
    rows, cols = len(A), len(A[0])
    result = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] - B[i][j]
    return result

def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    # Result: rows_A x cols_B
    result = [[0] * cols_B for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

# Test
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = [[1, 2, 3], [4, 5, 6]]  # 2x3

print("A + B:", matrix_add(A, B))        # [[6,8], [10,12]]
print("A - B:", matrix_subtract(A, B))   # [[-4,-4], [-4,-4]]
print("A × B:", matrix_multiply(A, B))   # [[19,22], [43,50]]
print("A × C:", matrix_multiply(A, C))   # [[9,12,15], [19,26,33]]
```

**Algorithm Steps (Multiplication):**
1. Check: cols(A) == rows(B)
2. Initialize result matrix of size rows(A) × cols(B)
3. For each (i, j) in result:
   - Compute sum of A[i][k] × B[k][j] for all k
4. Return result

**Complexity:**
| Operation | Time | Space |
|-----------|------|-------|
| Add/Sub | O(mn) | O(mn) |
| Multiply | O(mnp) | O(mp) |

**Interview Tips:**
- Always check dimension compatibility first
- Multiply: "row i dot column j"
- Know that Strassen's algorithm exists (O(n^2.81)) but rarely asked to implement

---

## Question 2
- [ ] Done

**Implement a function to calculate the transpose of a given matrix.**

### Answer

**Definition:**  
Transpose swaps rows and columns of a matrix. If A is m×n, then A^T is n×m where (A^T)_ij = A_ji.

**Core Concepts:**
- Row i becomes column i
- Shape changes from (m×n) to (n×m)
- For square matrices: can do in-place by swapping A[i][j] with A[j][i]
- Property: (A^T)^T = A, (AB)^T = B^T A^T

**Mathematical Formulation:**
$$A^T_{ij} = A_{ji}$$

**Python Code:**
```python
# Pipeline: Get dimensions → Create result (cols×rows) → Swap positions

def transpose(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    
    # Result has swapped dimensions
    result = [[0] * rows for _ in range(cols)]
    
    for i in range(rows):
        for j in range(cols):
            result[j][i] = matrix[i][j]
    return result

# One-liner version (using zip)
def transpose_zip(matrix):
    return [list(row) for row in zip(*matrix)]

# Test
A = [[1, 2, 3],
     [4, 5, 6]]  # 2x3

print("Original:", A)
print("Transpose:", transpose(A))  # [[1,4], [2,5], [3,6]] - now 3x2
```

**Algorithm Steps:**
1. Get rows (m) and cols (n)
2. Create new matrix of size n×m
3. For each (i, j): set result[j][i] = matrix[i][j]
4. Return result

**Complexity:**
- Time: O(mn)
- Space: O(mn) for new matrix, O(1) for in-place (square only)

**Interview Tips:**
- Know the `zip(*matrix)` trick for one-liner
- Remember: (AB)^T = B^T A^T (order reverses!)
- Symmetric matrix: A = A^T (e.g., covariance matrix)
```python
# ========================================
# SPECIAL CASES
# ========================================

# Row vector to column vector
row_vec = [[1, 2, 3]]
col_vec = transpose(row_vec)
print("\nRow vector:", row_vec)
print("Column vector:", col_vec)

# Column vector to row vector
col_vec2 = [[1], [2], [3]]
row_vec2 = transpose(col_vec2)
print("\nColumn vector:", col_vec2)
print("Row vector:", row_vec2)

# Check symmetric matrix
def is_symmetric(matrix):
    """Check if a matrix is symmetric (A = A^T)."""
    n = len(matrix)
    if n != len(matrix[0]):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True

symmetric = [[1, 2, 3],
             [2, 4, 5],
             [3, 5, 6]]

print("\nIs symmetric:", is_symmetric(symmetric))
print("Is B symmetric:", is_symmetric(B))
```

**Algorithm: Matrix Transpose**
1. Get dimensions: rows = m, cols = n
2. Create new matrix of size n×m
3. For each position (i, j) in original:
4.     Set result[j][i] = matrix[i][j]
5. Return transposed matrix

**Complexity Analysis:**
| Approach | Time | Space |
|----------|------|-------|
| Standard | O(mn) | O(mn) |
| In-place (square) | O(n²) | O(1) |
| Pythonic (zip) | O(mn) | O(mn) |

**Interview Tips:**
- Know in-place is only possible for square matrices
- Mention the `zip(*matrix)` trick in Python
- Remember: (AB)^T = B^T A^T (order reverses!)
- For symmetric matrices, A = A^T (useful for covariance matrices)

---

## Question 3
- [ ] Done

**Code to find the determinant of a matrix using recursion.**

### Answer

**Definition:**  
The determinant is a scalar value computed from a square matrix. It indicates if a matrix is invertible (det ≠ 0) and represents the scaling factor of the linear transformation.

**Core Concepts:**
- Base case: 1×1 → det = element, 2×2 → det = ad - bc
- Recursive: Cofactor expansion along first row
- Cofactor C_ij = (-1)^(i+j) × det(minor)
- Minor = submatrix with row i and col j removed

**Mathematical Formulation:**

2×2: $\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$

n×n (cofactor expansion along row 0):
$$\det(A) = \sum_{j=0}^{n-1} (-1)^{j} \cdot a_{0j} \cdot \det(M_{0j})$$

**Python Code:**
```python
# Pipeline: Base cases → Get minor → Recursive cofactor expansion

def get_minor(matrix, row, col):
    """Remove row and col to get minor matrix."""
    return [[matrix[i][j] for j in range(len(matrix)) if j != col]
            for i in range(len(matrix)) if i != row]

def determinant(matrix):
    n = len(matrix)
    
    # Base cases
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    # Cofactor expansion along first row
    det = 0
    for j in range(n):
        minor = get_minor(matrix, 0, j)
        cofactor = ((-1) ** j) * determinant(minor)
        det += matrix[0][j] * cofactor
    
    return det

# Test
A = [[1, 2], [3, 4]]
print("2x2 det:", determinant(A))  # 1*4 - 2*3 = -2

B = [[6, 1, 1],
     [4, -2, 5],
     [2, 8, 7]]
print("3x3 det:", determinant(B))  # -306
```

**Algorithm Steps:**
1. If 1×1: return element
2. If 2×2: return ad - bc
3. For j = 0 to n-1:
   - Get minor M_0j (remove row 0, col j)
   - Add (-1)^j × A[0][j] × det(M_0j)
4. Return sum

**Complexity:**
- Time: O(n!) - very slow for large n
- Space: O(n²) for recursion stack
- Use LU decomposition O(n³) in practice

**Properties to Remember:**
- det(AB) = det(A) × det(B)
- det(A^T) = det(A)
- det(A⁻¹) = 1/det(A)
- Row swap → sign flips

**Interview Tips:**
- Know 2×2 formula by heart: ad - bc
- Mention LU decomposition as efficient alternative
- det = 0 means singular (no inverse)

---

## Question 4
- [ ] Done

**Develop a Python function to compute the inverse of a matrix.**

### Answer

**Definition:**  
The inverse of matrix A is A⁻¹ where AA⁻¹ = I. Exists only if det(A) ≠ 0. Used to solve Ax = b as x = A⁻¹b.

**Core Concepts:**
- Only square matrices can have inverses
- det(A) must be non-zero
- Gauss-Jordan: Augment [A|I], row reduce to [I|A⁻¹]
- 2×2 formula: swap diagonal, negate off-diagonal, divide by det

**Mathematical Formulation:**

2×2 inverse:
$$A^{-1} = \frac{1}{ad-bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

Gauss-Jordan: $[A | I] \xrightarrow{\text{row ops}} [I | A^{-1}]$

**Python Code:**
```python
# Pipeline: Augment [A|I] → Row reduce → Extract inverse

def inverse(matrix):
    n = len(matrix)
    
    # Create augmented matrix [A | I]
    aug = [[float(matrix[i][j]) for j in range(n)] + 
           [1.0 if i == j else 0.0 for j in range(n)]
           for i in range(n)]
    
    # Forward elimination + back substitution (Gauss-Jordan)
    for col in range(n):
        # Find pivot (partial pivoting)
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]
        
        # Check for singular matrix
        if abs(aug[col][col]) < 1e-10:
            raise ValueError("Matrix is singular")
        
        # Scale pivot row to make pivot = 1
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot
        
        # Eliminate all other rows
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]
    
    # Extract inverse (right half)
    return [[aug[i][j + n] for j in range(n)] for i in range(n)]

# Test
A = [[4, 7], [2, 6]]
A_inv = inverse(A)
print("A⁻¹:", A_inv)
# Verify: A × A⁻¹ should be identity
```

**Algorithm Steps (Gauss-Jordan):**
1. Create augmented matrix [A | I]
2. For each column i:
   - Find pivot (largest element)
   - Swap rows if needed
   - Scale row to make pivot = 1
   - Eliminate: make all other rows have 0 in column i
3. Right half is A⁻¹

**Complexity:**
- Time: O(n³)
- Space: O(n²)

**Interview Tips:**
- Know 2×2 formula by heart
- Avoid computing inverse directly; solve Ax=b instead
- det = 0 means no inverse exists
- Use `np.linalg.solve(A, b)` not `np.linalg.inv(A) @ b`

---

## Question 5
- [ ] Done

**Write an algorithm to perform eigenvalue and eigenvector decomposition.**

### Answer

**Definition:**  
Eigendecomposition finds scalars λ (eigenvalues) and vectors v (eigenvectors) where Av = λv. The eigenvalue scales the eigenvector; the eigenvector's direction remains unchanged under transformation A.

**Core Concepts:**
- Eigenvalue equation: Av = λv
- Characteristic polynomial: det(A - λI) = 0
- n×n matrix has n eigenvalues (may repeat or be complex)
- Symmetric matrices → real eigenvalues, orthogonal eigenvectors

**Mathematical Formulation:**

$$A\mathbf{v} = \lambda\mathbf{v}$$
$$\det(A - \lambda I) = 0$$

For 2×2: $\lambda = \frac{\text{trace} \pm \sqrt{\text{trace}^2 - 4\det}}{2}$

**Python Code:**
```python
# Pipeline: Power iteration for dominant eigenvalue
# (Full eigendecomposition typically uses NumPy)

import math

def power_iteration(A, num_iterations=100):
    """Find dominant eigenvalue and eigenvector."""
    n = len(A)
    
    # Start with random vector
    v = [1.0] * n
    
    for _ in range(num_iterations):
        # Multiply: w = Av
        w = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
        
        # Normalize
        norm = math.sqrt(sum(x**2 for x in w))
        v = [x / norm for x in w]
        
        # Eigenvalue estimate (Rayleigh quotient)
        Av = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
        eigenvalue = sum(Av[i] * v[i] for i in range(n))
    
    return eigenvalue, v

# Using NumPy (practical approach)
import numpy as np

A = np.array([[4, 2], [1, 3]])

# All eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# For symmetric matrices (faster, more stable)
A_sym = np.array([[4, 2], [2, 3]])
eigenvalues, eigenvectors = np.linalg.eigh(A_sym)
print("Symmetric eigenvalues:", eigenvalues)
```

**Algorithm: Power Iteration**
1. Start with random vector v
2. Repeat:
   - Compute w = Av
   - Normalize: v = w / ||w||
   - λ = v^T A v (Rayleigh quotient)
3. Converges to largest |λ| and its eigenvector

**Complexity:**
| Method | Time | Notes |
|--------|------|-------|
| Power iteration | O(n² × iter) | Dominant eigenvalue only |
| NumPy eig | O(n³) | All eigenvalues |
| NumPy eigh | O(n³) | Symmetric (faster) |

**Interview Tips:**
- Power iteration gives largest eigenvalue only
- Use `np.linalg.eigh` for symmetric matrices
- Eigenvalues of symmetric matrices are always real
- PCA uses eigendecomposition of covariance matrix

---

## Question 6
- [ ] Done

**Create a Python script to solve a system of linear equations using NumPy.**

### Answer

**Definition:**  
Solve Ax = b where A is coefficient matrix, x is unknowns, b is constants. NumPy provides efficient methods using LU decomposition internally.

**Core Concepts:**
- Square system (n×n): unique solution if det(A) ≠ 0
- Overdetermined (m > n): least squares solution
- `np.linalg.solve`: preferred for square systems
- `np.linalg.lstsq`: works for all cases

**Mathematical Formulation:**
$$A\mathbf{x} = \mathbf{b}$$

Square: $\mathbf{x} = A^{-1}\mathbf{b}$

Overdetermined: $\mathbf{x} = (A^T A)^{-1} A^T \mathbf{b}$

**Python Code:**
```python
import numpy as np

# Pipeline: Setup A and b → Use np.linalg.solve → Verify

# Example: 2x + y = 5, x + 3y = 6
A = np.array([[2, 1],
              [1, 3]], dtype=float)
b = np.array([5, 6], dtype=float)

# Method 1: np.linalg.solve (preferred)
x = np.linalg.solve(A, b)
print("Solution:", x)  # [1.8, 1.4]

# Verify: Ax should equal b
print("Verification Ax:", A @ x)
print("Original b:", b)

# Method 2: Least squares (works for overdetermined too)
x_lstsq, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
print("Lstsq solution:", x_lstsq)

# Overdetermined example (3 equations, 2 unknowns)
A_over = np.array([[1, 1], [2, 1], [1, 2]], dtype=float)
b_over = np.array([3, 4, 4], dtype=float)
x_over = np.linalg.lstsq(A_over, b_over, rcond=None)[0]
print("Overdetermined:", x_over)
```

**Method Selection:**
| System | Method | Function |
|--------|--------|----------|
| Square | Direct | `np.linalg.solve` |
| Overdetermined | Least squares | `np.linalg.lstsq` |
| Underdetermined | Min norm | `np.linalg.lstsq` |

**Interview Tips:**
- Always use `solve` over `inv(A) @ b` (more stable, faster)
- `lstsq` returns (solution, residuals, rank, singular values)
- Check condition number for ill-conditioned systems

---

## Question 7
- [ ] Done

**Implement a function to calculate the L2 norm of a vector.**

### Answer

**Definition:**  
L2 norm (Euclidean norm) is the "length" of a vector, computed as the square root of sum of squared elements. It's the most common norm in ML for regularization and distance.

**Core Concepts:**
- L2 norm = √(Σxᵢ²) = distance from origin
- Used in ridge regression, gradient clipping, distance metrics
- Unit vector: v / ||v||₂

**Mathematical Formulation:**
$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2} = \sqrt{\mathbf{x}^T \mathbf{x}}$$

**Python Code:**
```python
import math

# Pipeline: Square each element → Sum → Square root

def l2_norm(vector):
    return math.sqrt(sum(x**2 for x in vector))

def normalize(vector):
    """Convert to unit vector."""
    norm = l2_norm(vector)
    return [x / norm for x in vector]

# Test
v = [3, 4]
print("L2 norm:", l2_norm(v))  # 5 (3-4-5 triangle)
print("Normalized:", normalize(v))  # [0.6, 0.8]

# Using NumPy
import numpy as np
v_np = np.array([3, 4])
print("NumPy norm:", np.linalg.norm(v_np))

# Different norms
print("L1 norm:", np.linalg.norm(v_np, ord=1))   # |3|+|4| = 7
print("L2 norm:", np.linalg.norm(v_np, ord=2))   # 5
print("L∞ norm:", np.linalg.norm(v_np, ord=np.inf))  # max(3,4) = 4
```

**Algorithm Steps:**
1. For each element xᵢ, compute xᵢ²
2. Sum all squared values
3. Return √sum

**Complexity:** O(n) time, O(1) space

**ML Applications:**
| Use Case | Role of L2 Norm |
|----------|-----------------|
| Ridge regression | Regularization: λ||w||₂² |
| Gradient clipping | Clip if ||∇|| > threshold |
| Cosine similarity | Normalize before dot product |
| K-NN | Distance metric |

**Interview Tips:**
- L2 is differentiable everywhere (L1 is not at 0)
- Ridge uses ||w||² (squared), not ||w||
- `np.linalg.norm(x)` defaults to L2

---

## Question 8
- [ ] Done

**Write a program to verify if a given square matrix is orthogonal.**

### Answer

**Definition:**  
A matrix Q is orthogonal if Q^T Q = I (transpose equals inverse). Orthogonal matrices preserve lengths and angles, representing rotations or reflections.

**Core Concepts:**
- Q^T Q = QQ^T = I
- Q^(-1) = Q^T (inverse is just transpose)
- det(Q) = ±1 (rotation: +1, reflection: -1)
- Columns (and rows) are orthonormal vectors

**Mathematical Formulation:**
$$Q^T Q = I$$
$$|\det(Q)| = 1$$

**Python Code:**
```python
import numpy as np

# Pipeline: Compute Q^T Q → Check if equals identity

def is_orthogonal(Q, tol=1e-10):
    """Check if matrix Q is orthogonal."""
    Q = np.array(Q, dtype=float)
    n = Q.shape[0]
    
    # Compute Q^T Q
    QTQ = Q.T @ Q
    
    # Check if Q^T Q ≈ I
    I = np.eye(n)
    return np.allclose(QTQ, I, atol=tol)

# Test: Rotation matrix (should be orthogonal)
import math
theta = math.pi / 4  # 45 degrees
R = np.array([[math.cos(theta), -math.sin(theta)],
              [math.sin(theta), math.cos(theta)]])

print("Rotation matrix:")
print(R)
print("Is orthogonal:", is_orthogonal(R))
print("det(R):", np.linalg.det(R))  # Should be 1

# Test: Random matrix (not orthogonal)
A = np.array([[1, 2], [3, 4]])
print("\nRandom matrix is orthogonal:", is_orthogonal(A))

# Test: Identity (trivially orthogonal)
I = np.eye(3)
print("Identity is orthogonal:", is_orthogonal(I))
```

**Algorithm Steps:**
1. Compute Q^T Q
2. Check if diagonal elements ≈ 1
3. Check if off-diagonal elements ≈ 0
4. Return True if all checks pass

**Properties:**
| Property | Formula |
|----------|---------|
| Definition | Q^T Q = I |
| Inverse | Q^(-1) = Q^T |
| Determinant | det(Q) = ±1 |
| Preserves length | ||Qx|| = ||x|| |

**Interview Tips:**
- Q^T = Q^(-1) is the key property
- Rotation: det = +1, Reflection: det = -1
- In deep learning, orthogonal initialization helps gradient flow
- QR decomposition gives orthogonal Q

---

## Question 9
- [ ] Done

**How would you implement a linear algebra-based algorithm to classify text documents?**

### Answer

**Definition:**  
Text classification using linear algebra involves: (1) representing documents as TF-IDF vectors, (2) optionally reducing dimensions with SVD/LSA, (3) using linear classifiers. Core operations are matrix multiplication and SVD.

**Core Concepts:**
- **TF-IDF**: Term frequency × inverse document frequency weighting
- **LSA (Latent Semantic Analysis)**: Truncated SVD to find semantic patterns
- **Cosine similarity**: Compare document vectors
- **Linear classifier**: logistic regression on vectorized documents

**Mathematical Formulation:**

TF-IDF: $\text{tfidf}(t, d) = \text{tf}(t, d) \times \log\frac{N}{\text{df}(t)}$

LSA: $X \approx U_k \Sigma_k V_k^T$ (truncated SVD)

**Python Code:**
```python
# Pipeline: Documents → TF-IDF vectors → (Optional: LSA) → Classifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

# Sample documents
documents = [
    "machine learning neural networks deep learning",
    "python programming software development",
    "football game team championship score",
    "basketball player points rebounds"
]
labels = [0, 0, 1, 1]  # 0: Tech, 1: Sports

# Method 1: Simple TF-IDF + Logistic Regression
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(documents)

clf = LogisticRegression()
clf.fit(X_tfidf, labels)

# Predict new document
new_doc = ["neural network training gradient"]
X_new = vectorizer.transform(new_doc)
print("Prediction:", clf.predict(X_new))  # 0 (Tech)

# Method 2: With LSA for dimensionality reduction
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svd', TruncatedSVD(n_components=2)),  # Reduce to 2 dims
    ('clf', LogisticRegression())
])

pipeline.fit(documents, labels)
print("Pipeline prediction:", pipeline.predict(new_doc))

# Manual cosine similarity
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Compare documents using TF-IDF vectors
X_dense = X_tfidf.toarray()
print("Similarity doc0-doc1:", cosine_similarity(X_dense[0], X_dense[1]))
print("Similarity doc0-doc2:", cosine_similarity(X_dense[0], X_dense[2]))
```

**Pipeline Steps:**
1. **Tokenize**: Split text into words
2. **Vectorize**: Convert to TF-IDF vectors (sparse matrix)
3. **Reduce** (optional): Apply truncated SVD (LSA)
4. **Train**: Fit logistic regression on vectors
5. **Predict**: Transform new doc, classify

**Linear Algebra Operations:**
| Step | Operation |
|------|-----------|
| TF-IDF | Sparse matrix construction |
| LSA | Truncated SVD decomposition |
| Prediction | Matrix-vector multiplication |
| Similarity | Dot product (cosine) |

**Complexity:**
- TF-IDF: O(n × avg_words)
- Truncated SVD: O(mnk) for m terms, n docs, k components
- Logistic regression: O(iterations × features × samples)

**Interview Tips:**
- TF-IDF captures word importance (rare words = more important)
- LSA/SVD captures semantic relationships
- Modern: Use word embeddings (Word2Vec) or transformers (BERT)
- Know that this is "bag of words" (loses word order)