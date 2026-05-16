# NumPy Interview Questions - Coding Questions

## Question 1
- [ ] Done

**How do you create a NumPy array from a regular Python list ?**

### Solution
```python
import numpy as np

# From 1D list
py_list = [1, 2, 3, 4, 5]
arr_1d = np.array(py_list)
print(arr_1d)  # [1 2 3 4 5]

# From 2D list (nested)
py_list_2d = [[1, 2, 3],
              [4, 5, 6]]
arr_2d = np.array(py_list_2d)
print(arr_2d)
# [[1 2 3]
#  [4 5 6]]

# With specific dtype
arr_float = np.array([1, 2, 3], dtype=np.float32)
print(arr_float)  # [1. 2. 3.]
```

---

## Question 2
- [ ] Done

**How do you reshape a NumPy array?**

### Solution
```python
import numpy as np

arr = np.arange(12)  # [0, 1, 2, ..., 11]

# Reshape to 3x4
arr_3x4 = arr.reshape(3, 4)
print(arr_3x4)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Use -1 for automatic dimension calculation
arr_auto = arr.reshape(4, -1)  # 4 rows, auto calculate cols (3)
print(arr_auto.shape)  # (4, 3)

# Flatten back to 1D
arr_flat = arr_3x4.flatten()    # Returns copy
arr_ravel = arr_3x4.ravel()     # Returns view (if possible)
```

### Important
- Total elements must match: `3 × 4 = 12`
- Use `-1` for one dimension to auto-calculate

---

## Question 3
- [ ] Done

**How do you slice and index NumPy arrays?**

### Solution
```python
import numpy as np

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# --- Basic Slicing (returns VIEW) ---
# arr[row_start:row_end, col_start:col_end]
print(arr[0, :])      # First row: [1 2 3 4]
print(arr[:, 0])      # First column: [1 5 9]
print(arr[:2, 1:3])   # First 2 rows, cols 1-2
# [[2 3]
#  [6 7]]

# --- Boolean Indexing (returns COPY) ---
mask = arr > 5
print(arr[mask])      # [6 7 8 9 10 11 12]

# Conditional replacement
arr_copy = arr.copy()
arr_copy[arr_copy > 10] = 10  # Cap at 10

# --- Integer Array Indexing (returns COPY) ---
indices = [0, 2]
print(arr[indices])   # Rows 0 and 2
```

### Key Difference
- **Basic slicing → View** (modifications affect original)
- **Boolean/Integer indexing → Copy** (independent)

---

## Question 4
- [ ] Done

**How do you perform matrix multiplication using NumPy ?**

### Solution
```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])    # Shape: (2, 2)

B = np.array([[5, 6],
              [7, 8]])    # Shape: (2, 2)

# Method 1: @ operator (recommended)
C = A @ B

# Method 2: np.dot()
C = np.dot(A, B)

# Method 3: np.matmul()
C = np.matmul(A, B)

print(C)
# [[19 22]
#  [43 50]]

# --- For vectors ---
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Dot product (scalar result)
dot = np.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32
```

### Warning: Don't confuse with element-wise multiplication!
```python
print(A * B)  # Element-wise: [[ 5 12], [21 32]]
print(A @ B)  # Matrix mult:  [[19 22], [43 50]]
```

---

## Question 5
- [ ] Done

**How do you concatenate and stack arrays?**

### Solution
```python
import numpy as np

a = np.array([[1, 2],
              [3, 4]])
b = np.array([[5, 6],
              [7, 8]])

# --- Concatenate (join along existing axis) ---
# Vertically (axis=0) - adds rows
vert = np.concatenate([a, b], axis=0)
print(vert.shape)  # (4, 2)

# Horizontally (axis=1) - adds columns
horiz = np.concatenate([a, b], axis=1)
print(horiz.shape)  # (2, 4)

# --- Stack (creates new axis) ---
# Stacks along new first axis
stacked = np.stack([a, b], axis=0)
print(stacked.shape)  # (2, 2, 2)

# --- Shortcuts ---
np.vstack([a, b])  # Same as concatenate axis=0
np.hstack([a, b])  # Same as concatenate axis=1
```

---

## Question 6
- [ ] Done

**How do you generate random numbers with NumPy?**

### Solution
```python
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# --- Uniform distribution [0, 1) ---
uniform = np.random.rand(3, 4)      # Shape (3, 4)

# --- Normal distribution (mean=0, std=1) ---
normal = np.random.randn(3, 4)      # Shape (3, 4)

# --- Integers ---
integers = np.random.randint(0, 10, size=(3, 4))  # [0, 10)

# --- From specific distribution ---
# Normal with custom mean and std
custom_normal = np.random.normal(loc=50, scale=10, size=(100,))

# Uniform with custom range
custom_uniform = np.random.uniform(low=5, high=15, size=(100,))

# --- Shuffle and choice ---
arr = np.arange(10)
np.random.shuffle(arr)              # In-place shuffle
sample = np.random.choice(arr, 3)   # Random sample of 3
```

### New Generator API (Recommended)
```python
rng = np.random.default_rng(seed=42)
uniform = rng.random((3, 4))
normal = rng.standard_normal((3, 4))
integers = rng.integers(0, 10, size=(3, 4))
```

---

## Question 7
- [ ] Done

**How do you find unique values and their counts?**

### Solution
```python
import numpy as np

arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

# --- Unique values ---
unique = np.unique(arr)
print(unique)  # [1 2 3 4]

# --- Unique with counts ---
unique, counts = np.unique(arr, return_counts=True)
print(dict(zip(unique, counts)))  # {1: 1, 2: 2, 3: 3, 4: 4}

# --- Unique with indices ---
unique, indices = np.unique(arr, return_index=True)
print(indices)  # [0 1 3 6] - first occurrence indices
```

---

## Question 8
- [ ] Done

**How do you normalize an array in NumPy ?**

### Solution
```python
import numpy as np

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# --- Sort (returns copy) ---
sorted_arr = np.sort(arr)
print(sorted_arr)  # [1 1 2 3 4 5 6 9]

# --- In-place sort ---
arr.sort()  # Modifies original

# --- Get sorted indices (argsort) ---
arr = np.array([3, 1, 4, 1, 5])
indices = np.argsort(arr)
print(indices)         # [1 3 0 2 4]
print(arr[indices])    # Sorted: [1 1 3 4 5]

# --- 2D array sorting ---
arr_2d = np.array([[3, 1, 2],
                   [6, 4, 5]])

# Sort each row
print(np.sort(arr_2d, axis=1))
# [[1 2 3]
#  [4 5 6]]

# Sort each column
print(np.sort(arr_2d, axis=0))
# [[3 1 2]
#  [6 4 5]]
```

---

## Question 9
- [ ] Done

**How do you implement standardization (Z-score normalization)?**

### Solution
```python
import numpy as np

# Sample data: 5 samples, 3 features
X = np.array([[1, 200, 0.5],
              [2, 300, 0.8],
              [3, 250, 0.6],
              [4, 350, 0.9],
              [5, 400, 0.7]])

# Calculate mean and std for each feature (column)
mean = X.mean(axis=0)
std = X.std(axis=0)

# Standardize: Z = (X - mean) / std
X_standardized = (X - mean) / std

print(f"Mean before: {X.mean(axis=0)}")
print(f"Mean after: {X_standardized.mean(axis=0).round(2)}")  # [0, 0, 0]
print(f"Std after: {X_standardized.std(axis=0).round(2)}")    # [1, 1, 1]
```

---

## Question 10
- [ ] Done

**How do you implement Min-Max normalization?**

### Solution
```python
import numpy as np

X = np.array([[1, 200],
              [2, 300],
              [3, 400],
              [4, 500],
              [5, 600]])

# Calculate min and max per feature
X_min = X.min(axis=0)
X_max = X.max(axis=0)

# Normalize to [0, 1]
X_normalized = (X - X_min) / (X_max - X_min)

print(X_normalized)
# [[0.   0.  ]
#  [0.25 0.25]
#  [0.5  0.5 ]
#  [0.75 0.75]
#  [1.   1.  ]]

# Verify range is [0, 1]
print(f"Min: {X_normalized.min(axis=0)}")  # [0, 0]
print(f"Max: {X_normalized.max(axis=0)}")  # [1, 1]
```

---

## Question 11
- [ ] Done

**How do you compute the dot product and cross product?**

### Solution
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# --- Dot Product (scalar) ---
# Formula: a·b = a1*b1 + a2*b2 + a3*b3
dot = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32
print(f"Dot product: {dot}")

# Alternative: a @ b

# --- Cross Product (vector, 3D only) ---
# Returns vector perpendicular to both inputs
cross = np.cross(a, b)
print(f"Cross product: {cross}")  # [-3, 6, -3]
```

---

## Question 12
- [ ] Done

**How do you compute eigenvalues and eigenvectors?**

### Solution
```python
import numpy as np

A = np.array([[4, 2],
              [1, 3]])

# Get eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")     # [5. 2.]
print(f"Eigenvectors:\n{eigenvectors}")

# Verify: A @ v = λ * v
v = eigenvectors[:, 0]  # First eigenvector
lambda_1 = eigenvalues[0]

print(f"A @ v: {A @ v}")
print(f"λ * v: {lambda_1 * v}")
# Both should be equal
```

### Use Case in ML
- PCA (Principal Component Analysis) uses eigendecomposition
- Eigenvectors of covariance matrix = principal components

---

## Question 13
- [ ] Done

**Explain how to invert a matrix in NumPy.**

**Answer:**

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])

# Method 1: np.linalg.inv()
A_inv = np.linalg.inv(A)
# [[-2.0,  1.0],
#  [ 1.5, -0.5]]

# Verify: A @ A_inv ≈ Identity
print(np.allclose(A @ A_inv, np.eye(2)))  # True

# Method 2: Pseudo-inverse (works for non-square & singular matrices)
A_pinv = np.linalg.pinv(A)  # Moore-Penrose pseudo-inverse

# Method 3: Solve system Ax = b (preferred over computing inverse)
b = np.array([5, 6])
x = np.linalg.solve(A, b)  # More numerically stable than inv(A) @ b

# Check if matrix is invertible
det = np.linalg.det(A)
if abs(det) > 1e-10:
    A_inv = np.linalg.inv(A)
else:
    print("Matrix is singular")
```

| Function | Use Case |
|----------|----------|
| `np.linalg.inv()` | Square, non-singular matrices |
| `np.linalg.pinv()` | Non-square or singular (pseudo-inverse) |
| `np.linalg.solve()` | Solving Ax=b (faster, more stable) |

> **Interview Tip:** In practice, **never invert** a matrix to solve $Ax = b$. Use `np.linalg.solve(A, b)` instead — it's 2-3× faster and more numerically stable.

---

## Question 14
- [ ] Done

**How do you calculate the determinant of a matrix?**

**Answer:**

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])

# Calculate determinant
det = np.linalg.det(A)  # -2.0  (1*4 - 2*3)

# 3x3 matrix
B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.linalg.det(B)  # 0.0 (singular matrix)

# Use cases
# 1. Check if matrix is invertible
if abs(np.linalg.det(A)) > 1e-10:
    print("Invertible")

# 2. Multivariate Gaussian: det of covariance matrix
cov = np.cov(np.random.randn(3, 100))
log_det = np.linalg.slogdet(cov)  # Returns (sign, log|det|)
# More numerically stable for large matrices
sign, logdet = np.linalg.slogdet(cov)
actual_det = sign * np.exp(logdet)
```

| Function | Description |
|----------|------------|
| `np.linalg.det(A)` | Determinant (may overflow for large matrices) |
| `np.linalg.slogdet(A)` | Sign and log-determinant (numerically stable) |

> **Interview Tip:** For ML, the determinant is key in **multivariate Gaussian** distributions and **covariance matrix** analysis. Use `slogdet()` for large matrices to avoid numerical overflow.

---

## Question 15
- [ ] Done

**Describe how you would flatten a multi-dimensional array.**

**Answer:**

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

# Method 1: flatten() — returns a COPY
flat = arr.flatten()         # [1, 2, 3, 4, 5, 6]
flat_f = arr.flatten('F')    # [1, 4, 2, 5, 3, 6] (column-major / Fortran order)

# Method 2: ravel() — returns a VIEW (no copy, faster)
raveled = arr.ravel()        # [1, 2, 3, 4, 5, 6]
raveled[0] = 99              # Modifies original arr!

# Method 3: reshape(-1)
reshaped = arr.reshape(-1)   # [1, 2, 3, 4, 5, 6]

# Method 4: np.ndarray.flat (iterator)
for val in arr.flat:
    print(val)               # 1, 2, 3, 4, 5, 6
```

| Method | Returns | Modifies Original? | Speed |
|--------|---------|--------------------|---------|
| `flatten()` | Copy | No | Slower |
| `ravel()` | View (usually) | Yes | Fastest |
| `reshape(-1)` | View (if possible) | Maybe | Fast |
| `.flat` | Iterator | N/A | For looping |

### Order Options

| Order | Description | Example |
|-------|-------------|--------|
| `'C'` (default) | Row-major | [1,2,3,4,5,6] |
| `'F'` | Column-major | [1,4,2,5,3,6] |
| `'A'` | Follow memory layout | Depends on storage |

> **Interview Tip:** Use `ravel()` for performance (no copy). Use `flatten()` when you need an independent copy. The key distinction is **view vs. copy**.

---

## Question 16
- [ ] Done

**How can you reverse an array in NumPy?**

**Answer:**

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Method 1: Slicing with step -1 (most common)
reversed_arr = arr[::-1]       # [5, 4, 3, 2, 1]  — returns a view

# Method 2: np.flip()
reversed_arr = np.flip(arr)    # [5, 4, 3, 2, 1]

# Method 3: np.flipud() / np.fliplr() for 2D
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
np.flipud(arr2d)  # Flip up-down (reverse rows)
# [[4, 5, 6],
#  [1, 2, 3]]

np.fliplr(arr2d)  # Flip left-right (reverse columns)
# [[3, 2, 1],
#  [6, 5, 4]]

# Reverse along specific axis
np.flip(arr2d, axis=0)  # Same as flipud
np.flip(arr2d, axis=1)  # Same as fliplr
np.flip(arr2d)          # Flip all axes

# Reverse in-place (without creating a new array)
arr[:] = arr[::-1]  # In-place reversal
```

| Method | Dimension | Returns |
|--------|-----------|--------|
| `arr[::-1]` | 1D | View |
| `np.flip(arr)` | Any | View |
| `np.flipud(arr)` | 2D+ | Flip rows (up-down) |
| `np.fliplr(arr)` | 2D+ | Flip columns (left-right) |

> **Interview Tip:** `[::-1]` returns a **view** (no copy), making it extremely fast. `np.flip()` is more readable and works on any axis.

---

## Question 17
- [ ] Done

**How do you apply a conditional filter to a NumPy array?**

**Answer:**

```python
import numpy as np

arr = np.array([10, 25, 30, 5, 15, 40, 8])

# Method 1: Boolean indexing (most common)
result = arr[arr > 15]           # [25, 30, 40]
result = arr[arr % 2 == 0]       # [10, 30, 40, 8]

# Method 2: Multiple conditions (use & | ~ with parentheses)
result = arr[(arr > 10) & (arr < 35)]   # [25, 30, 15]
result = arr[(arr < 10) | (arr > 30)]   # [5, 40, 8]
result = arr[~(arr > 20)]               # [10, 5, 15, 8]

# Method 3: np.where() — conditional replacement
np.where(arr > 15, arr, 0)       # [0, 25, 30, 0, 0, 40, 0]
np.where(arr > 15, 'high', 'low')  # ['low','high','high','low','low','high','low']

# Method 4: np.extract()
np.extract(arr > 15, arr)        # [25, 30, 40]

# Method 5: np.select() — multiple conditions
conditions = [arr < 10, arr < 30, arr >= 30]
choices = ['low', 'medium', 'high']
np.select(conditions, choices, default='unknown')
# ['medium','medium','high','low','medium','high','low']

# 2D array filtering
arr2d = np.array([[1, 2], [3, 4], [5, 6]])
row_mask = arr2d[:, 0] > 2      # Filter rows where first column > 2
arr2d[row_mask]                  # [[3, 4], [5, 6]]
```

> **Interview Tip:** Boolean indexing is NumPy's most powerful filtering tool. Always use **parentheses** around each condition with `&` / `|` operators (they have higher precedence than comparison operators).

---

## Question 18
- [ ] Done

**How can you compute percentiles with NumPy?**

**Answer:**

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Single percentile
np.percentile(arr, 50)    # 55.0 (median)
np.percentile(arr, 25)    # 32.5 (Q1)
np.percentile(arr, 75)    # 77.5 (Q3)

# Multiple percentiles at once
np.percentile(arr, [25, 50, 75])  # [32.5, 55.0, 77.5]

# Quantile (0 to 1 range instead of 0 to 100)
np.quantile(arr, 0.5)    # 55.0
np.quantile(arr, [0.25, 0.5, 0.75])  # [32.5, 55.0, 77.5]

# Different interpolation methods
np.percentile(arr, 30, interpolation='linear')   # Default
np.percentile(arr, 30, interpolation='lower')    # 30
np.percentile(arr, 30, interpolation='higher')   # 40
np.percentile(arr, 30, interpolation='nearest')  # 30
np.percentile(arr, 30, interpolation='midpoint') # 35.0

# Along axis (2D)
arr2d = np.array([[10, 20, 30], [40, 50, 60]])
np.percentile(arr2d, 50, axis=0)   # [25, 35, 45] per column
np.percentile(arr2d, 50, axis=1)   # [20, 50] per row

# NaN-safe version
arr_nan = np.array([1.0, np.nan, 3.0, 4.0])
np.nanpercentile(arr_nan, 50)  # 3.0 (ignores NaN)
np.nanquantile(arr_nan, 0.5)   # 3.0

# IQR for outlier detection
Q1 = np.percentile(arr, 25)
Q3 = np.percentile(arr, 75)
IQR = Q3 - Q1
outliers = arr[(arr < Q1 - 1.5*IQR) | (arr > Q3 + 1.5*IQR)]
```

> **Interview Tip:** `percentile` uses 0-100 scale; `quantile` uses 0-1 scale. The IQR method (Q1/Q3 ± 1.5×IQR) is a classic outlier detection technique using percentiles.

---

## Question 19
- [ ] Done

**How do you calculate the correlation coefficient using NumPy?**

**Answer:**

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Method 1: np.corrcoef() — Pearson correlation matrix
corr_matrix = np.corrcoef(x, y)
# [[1.0,  0.8165],
#  [0.8165, 1.0]]
corr_xy = corr_matrix[0, 1]  # 0.8165

# Method 2: Manual calculation
def pearson_corr(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    return numerator / denominator

# Method 3: Multiple variables — correlation matrix
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]).T
corr_all = np.corrcoef(data)  # 3x3 correlation matrix

# Covariance (related)
cov_matrix = np.cov(x, y)
# cov_xy = cov_matrix[0, 1]
# Relationship: corr = cov(x,y) / (std(x) * std(y))

# Rank correlation (Spearman) — manual with NumPy
def spearman_corr(x, y):
    rank_x = np.argsort(np.argsort(x)).astype(float)
    rank_y = np.argsort(np.argsort(y)).astype(float)
    return np.corrcoef(rank_x, rank_y)[0, 1]
```

| Function | Returns |
|----------|--------|
| `np.corrcoef(x, y)` | Pearson correlation matrix |
| `np.cov(x, y)` | Covariance matrix |
| Relationship | $r = \frac{cov(x,y)}{\sigma_x \sigma_y}$ |

> **Interview Tip:** `np.corrcoef` returns a **matrix**, not a scalar. The correlation coefficient is at `[0, 1]` or `[1, 0]`. For Spearman or Kendall, use `scipy.stats.spearmanr()`.

---

## Question 20
- [ ] Done

**Explain the use of the np.cumsum() and np.cumprod() functions.**

**Answer:**

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Cumulative Sum
np.cumsum(arr)   # [1, 3, 6, 10, 15]
# [1, 1+2, 1+2+3, 1+2+3+4, 1+2+3+4+5]

# Cumulative Product
np.cumprod(arr)  # [1, 2, 6, 24, 120]
# [1, 1*2, 1*2*3, 1*2*3*4, 1*2*3*4*5] = factorials!

# 2D array with axis
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
np.cumsum(arr2d)          # [1, 3, 6, 10, 15, 21] (flattened)
np.cumsum(arr2d, axis=0)  # [[1,2,3],[5,7,9]] (down columns)
np.cumsum(arr2d, axis=1)  # [[1,3,6],[4,9,15]] (across rows)

# Use cases in ML/Data Science:

# 1. Running total / cumulative distribution
data = np.array([100, 200, 150, 300])
cumulative_revenue = np.cumsum(data)  # [100, 300, 450, 750]

# 2. Cumulative probability (CDF)
probs = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
cdf = np.cumsum(probs)  # [0.1, 0.3, 0.6, 0.85, 1.0]

# 3. Compound returns (finance)
daily_returns = np.array([0.01, -0.02, 0.015, 0.005])
cumulative_return = np.cumprod(1 + daily_returns)  # [1.01, 0.9898, 1.0046, 1.0097]

# 4. NaN-safe version
arr_nan = np.array([1.0, np.nan, 3.0])
np.nancumsum(arr_nan)   # [1.0, 1.0, 4.0]
np.nancumprod(arr_nan)  # [1.0, 1.0, 3.0]
```

> **Interview Tip:** `cumsum` is used for **running totals** and **CDF construction**. `cumprod` is used for **compound growth rates** in finance. Both are vectorized and much faster than Python loops.

---

## Question 21
- [ ] Done

**Describe the process for creating a structured array in NumPy.**

**Answer:**

A **structured array** stores heterogeneous data types in named fields — like a table/record.

```python
import numpy as np

# Method 1: Define dtype with list of tuples
dt = np.dtype([('name', 'U20'), ('age', 'i4'), ('salary', 'f8')])
employees = np.array([
    ('Alice', 30, 75000.0),
    ('Bob', 25, 65000.0),
    ('Charlie', 35, 90000.0)
], dtype=dt)

# Access fields
print(employees['name'])     # ['Alice', 'Bob', 'Charlie']
print(employees['salary'])   # [75000., 65000., 90000.]
print(employees[0])          # ('Alice', 30, 75000.)

# Method 2: Dictionary format
dt2 = np.dtype({'names': ['x', 'y', 'z'],
                'formats': ['f4', 'f4', 'f4']})
points = np.zeros(5, dtype=dt2)
points['x'] = [1, 2, 3, 4, 5]

# Method 3: Comma-separated string shorthand
dt3 = np.dtype('U10, i4, f8')  # auto-named f0, f1, f2

# Nested structured array
dt_nested = np.dtype([
    ('name', 'U20'),
    ('position', [('x', 'f4'), ('y', 'f4')])
])
objects = np.array([('obj1', (1.0, 2.0))], dtype=dt_nested)
print(objects['position']['x'])  # [1.0]

# Filtering and sorting
high_salary = employees[employees['salary'] > 70000]
sorted_emp = np.sort(employees, order='salary')  # Sort by salary
```

| dtype Code | Type | Example |
|-----------|------|--------|
| `'i4'` | 32-bit int | age |
| `'f8'` | 64-bit float | salary |
| `'U20'` | Unicode string (20 chars) | name |
| `'b1'` | Boolean | flag |
| `'S10'` | Byte string (10 bytes) | label |

> **Interview Tip:** Structured arrays are useful for reading CSV/binary data with mixed types. For complex tabular analysis, Pandas DataFrames are more practical.

---

## Question 22
- [ ] Done

**How do you save and load NumPy arrays to and from disk?**

**Answer:**

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
mat = np.random.rand(3, 3)

# === Binary Format (NumPy native) ===

# Single array: .npy
np.save('array.npy', arr)
loaded = np.load('array.npy')  # Exact copy

# Multiple arrays: .npz
np.savez('arrays.npz', data=arr, matrix=mat)
loaded = np.load('arrays.npz')
print(loaded['data'])    # arr
print(loaded['matrix'])  # mat

# Compressed .npz (smaller file)
np.savez_compressed('arrays_compressed.npz', data=arr, matrix=mat)

# === Text Format ===

# CSV / space-delimited
np.savetxt('data.csv', mat, delimiter=',', header='col1,col2,col3', fmt='%.4f')
loaded = np.loadtxt('data.csv', delimiter=',')

# With missing values
loaded = np.genfromtxt('data.csv', delimiter=',', filling_values=0.0)

# === Memory-mapped files (for huge arrays) ===
# Creates file-backed array — doesn't load into RAM
mmap = np.memmap('huge_array.dat', dtype='float64', mode='w+', shape=(10000, 10000))
mmap[:] = np.random.rand(10000, 10000)
mmap.flush()  # Write to disk

# Load memory-mapped
mmap_read = np.memmap('huge_array.dat', dtype='float64', mode='r', shape=(10000, 10000))
```

| Method | Format | Speed | File Size | Use Case |
|--------|--------|-------|-----------|----------|
| `save/load` | .npy (binary) | Fastest | Small | Single array |
| `savez` | .npz (binary) | Fast | Small | Multiple arrays |
| `savez_compressed` | .npz (compressed) | Medium | Smallest | Storage-limited |
| `savetxt/loadtxt` | .csv/.txt | Slowest | Largest | Human-readable |
| `memmap` | Raw binary | Instant | Raw size | Huge datasets |

> **Interview Tip:** Use `.npy/.npz` for fast internal storage. Use `.csv` for interoperability. Use **memmap** for datasets larger than RAM.

---

## Question 23
- [ ] Done

**Write a NumPy code to create a 3x3 identity matrix.**

**Answer:**

```python
import numpy as np

# Method 1: np.eye() (most common)
I = np.eye(3)
# [[1., 0., 0.],
#  [0., 1., 0.],
#  [0., 0., 1.]]

# Method 2: np.identity()
I = np.identity(3)

# Method 3: np.diag()
I = np.diag([1, 1, 1])  # or np.diag(np.ones(3))

# Offset diagonal with np.eye()
np.eye(3, k=1)   # Upper diagonal
# [[0., 1., 0.],
#  [0., 0., 1.],
#  [0., 0., 0.]]

np.eye(3, k=-1)  # Lower diagonal

# Non-square identity-like
np.eye(3, 4)  # 3x4 with ones on main diagonal

# Integer identity
np.eye(3, dtype=int)

# Verify
A = np.random.rand(3, 3)
np.allclose(A @ np.eye(3), A)  # True — AI = A
```

| Function | Description |
|----------|------------|
| `np.eye(N)` | N×N identity, supports offset `k` |
| `np.identity(N)` | N×N identity (always square) |
| `np.diag(v)` | Diagonal matrix from vector |

> **Interview Tip:** `np.eye()` is more flexible (supports non-square and offset diagonals). `np.identity()` is strictly square.

---

## Question 24
- [ ] Done

**Code a function in NumPy to compute the moving average of a 1D array.**

**Answer:**

```python
import numpy as np

# Method 1: np.convolve() (simplest)
def moving_average(arr, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(arr, kernel, mode='valid')

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(moving_average(arr, 3))  # [2., 3., 4., 5., 6., 7., 8., 9.]

# Method 2: Cumulative sum (fastest for large arrays)
def moving_average_cumsum(arr, window_size):
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

# Method 3: Stride tricks (no copy, memory-efficient)
def moving_average_stride(arr, window_size):
    windows = np.lib.stride_tricks.sliding_window_view(arr, window_size)
    return np.mean(windows, axis=1)

# Method 4: With padding (same output length)
def moving_average_padded(arr, window_size):
    pad_size = window_size // 2
    padded = np.pad(arr.astype(float), pad_size, mode='edge')
    kernel = np.ones(window_size) / window_size
    result = np.convolve(padded, kernel, mode='valid')
    return result[:len(arr)]

# Performance comparison (10M elements)
# convolve:  ~50ms
# cumsum:    ~20ms  (fastest)
# stride:    ~80ms
```

> **Interview Tip:** The **cumsum method** is the fastest with O(n) complexity regardless of window size. `np.convolve` is the cleanest. For 2D data (images), use `scipy.ndimage.uniform_filter`.

---

## Question 25
- [ ] Done

**Generate a 2D NumPy array of random integers and normalize it between 0 and 1.**

**Answer:**

```python
import numpy as np

# Generate random 2D array
arr = np.random.randint(0, 100, size=(4, 5))
print("Original:")
print(arr)
# e.g., [[23, 87, 45, 12, 67],
#        [91, 34, 56, 78, 3], ...]

# Method 1: Min-Max normalization (global)
normalized = (arr - arr.min()) / (arr.max() - arr.min())
# All values now between 0.0 and 1.0

# Method 2: Normalize per column
col_min = arr.min(axis=0)
col_max = arr.max(axis=0)
norm_cols = (arr - col_min) / (col_max - col_min)

# Method 3: Normalize per row
row_min = arr.min(axis=1, keepdims=True)
row_max = arr.max(axis=1, keepdims=True)
norm_rows = (arr - row_min) / (row_max - row_min)

# Method 4: Z-score standardization
standardized = (arr - arr.mean()) / arr.std()

# Safe normalization (handle zero range)
def safe_normalize(arr):
    arr = arr.astype(np.float64)
    min_val = arr.min()
    max_val = arr.max()
    range_val = max_val - min_val
    if range_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / range_val

print("Normalized:")
print(normalized)
print(f"Min: {normalized.min()}, Max: {normalized.max()}")
# Min: 0.0, Max: 1.0
```

> **Interview Tip:** Always cast to float before normalizing (`astype(float)`) to avoid integer division. Handle edge case where min == max (constant array).

---

## Question 26
- [ ] Done

**Create a NumPy code snippet to extract all odd numbers from an array.**

**Answer:**

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Method 1: Boolean indexing with modulo
odds = arr[arr % 2 != 0]  # [1, 3, 5, 7, 9]

# Method 2: Using bitwise AND (faster)
odds = arr[arr & 1 == 1]  # [1, 3, 5, 7, 9]

# Method 3: np.extract()
odds = np.extract(arr % 2 != 0, arr)  # [1, 3, 5, 7, 9]

# Method 4: np.where() — get indices
odd_indices = np.where(arr % 2 != 0)[0]  # [0, 2, 4, 6, 8]
odds = arr[odd_indices]

# Even numbers
evens = arr[arr % 2 == 0]  # [2, 4, 6, 8, 10]

# 2D array — extract odd elements
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
odds_2d = arr2d[arr2d % 2 != 0]  # [1, 3, 5, 7, 9] (flattened)

# Replace even with 0, keep odd
result = np.where(arr % 2 != 0, arr, 0)  # [1, 0, 3, 0, 5, 0, 7, 0, 9, 0]
```

> **Interview Tip:** Boolean indexing (`arr[arr % 2 != 0]`) is the most Pythonic and efficient approach. The bitwise `& 1` check is slightly faster for large arrays.

---

## Question 27
- [ ] Done

**Implement a routine to calculate the outer product of two vectors in NumPy.**

**Answer:**

The **outer product** of vectors $\mathbf{a}$ (m×1) and $\mathbf{b}$ (n×1) produces an m×n matrix where element $(i,j) = a_i \times b_j$.

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6, 7])

# Method 1: np.outer() (most readable)
result = np.outer(a, b)
# [[ 4,  5,  6,  7],
#  [ 8, 10, 12, 14],
#  [12, 15, 18, 21]]

# Method 2: Broadcasting with reshape
result = a[:, np.newaxis] * b[np.newaxis, :]
# Same: a.reshape(-1, 1) * b.reshape(1, -1)

# Method 3: np.einsum()
result = np.einsum('i,j->ij', a, b)

# Method 4: Matrix multiplication
result = a.reshape(-1, 1) @ b.reshape(1, -1)

# Applications in ML:

# 1. Rank-1 update in gradient descent
weights_update = np.outer(error, input_vector)  # Weight gradient

# 2. Covariance matrix (outer product of deviations)
x = np.random.randn(100, 3)
x_centered = x - x.mean(axis=0)
cov_manual = np.dot(x_centered.T, x_centered) / (len(x) - 1)

# 3. Attention mechanism (simplified)
query = np.random.randn(5)
key = np.random.randn(5)
attention_score = np.outer(query, key)  # QK^T pattern
```

> **Interview Tip:** The outer product creates a **rank-1 matrix**. It's fundamental in ML for **weight updates**, **attention mechanisms**, and **covariance estimation**.

---

## Question 28
- [ ] Done

**Write a NumPy program to create a checkerboard 8x8 matrix using the tile function.**

**Answer:**

```python
import numpy as np

# Method 1: np.tile() (as asked)
block = np.array([[0, 1], [1, 0]])  # 2x2 base pattern
checkerboard = np.tile(block, (4, 4))  # Tile 4x4 times = 8x8
print(checkerboard)
# [[0, 1, 0, 1, 0, 1, 0, 1],
#  [1, 0, 1, 0, 1, 0, 1, 0],
#  [0, 1, 0, 1, 0, 1, 0, 1],
#  ...
#  [1, 0, 1, 0, 1, 0, 1, 0]]

# Method 2: Slicing approach
checkerboard = np.zeros((8, 8), dtype=int)
checkerboard[1::2, ::2] = 1   # Odd rows, even cols
checkerboard[::2, 1::2] = 1   # Even rows, odd cols

# Method 3: Index arithmetic
i, j = np.indices((8, 8))
checkerboard = (i + j) % 2

# Method 4: np.fromfunction
checkerboard = np.fromfunction(lambda i, j: (i + j) % 2, (8, 8), dtype=int)

# Verify
print(checkerboard.shape)  # (8, 8)
print(np.sum(checkerboard))  # 32 (half of 64)
```

> **Interview Tip:** `np.tile(A, reps)` repeats array `A` according to `reps`. It's useful for creating repeating patterns, expanding arrays for broadcasting, and data augmentation.

---

## Question 29
- [ ] Done

**Code a NumPy snippet to create a border around an existing array.**

**Answer:**

```python
import numpy as np

arr = np.ones((3, 3))
print("Original:")
print(arr)
# [[1., 1., 1.],
#  [1., 1., 1.],
#  [1., 1., 1.]]

# Method 1: np.pad() (most flexible)
bordered = np.pad(arr, pad_width=1, mode='constant', constant_values=0)
print("With border:")
print(bordered)
# [[0., 0., 0., 0., 0.],
#  [0., 1., 1., 1., 0.],
#  [0., 1., 1., 1., 0.],
#  [0., 1., 1., 1., 0.],
#  [0., 0., 0., 0., 0.]]

# Asymmetric padding
np.pad(arr, ((1, 2), (3, 4)), mode='constant', constant_values=0)
# 1 row top, 2 bottom, 3 cols left, 4 right

# Method 2: Manual approach
rows, cols = arr.shape
bordered = np.zeros((rows + 2, cols + 2))
bordered[1:-1, 1:-1] = arr

# Method 3: Different padding modes
np.pad(arr, 1, mode='edge')       # Replicate edge values
np.pad(arr, 1, mode='reflect')    # Mirror reflection
np.pad(arr, 1, mode='wrap')       # Circular wrapping
np.pad(arr, 1, mode='linear_ramp', end_values=0)  # Linear fade
np.pad(arr, 1, mode='mean')       # Fill with mean

# Custom border value
bordered = np.pad(arr, pad_width=2, mode='constant', constant_values=9)
```

| Mode | Description | Use Case |
|------|-------------|----------|
| `constant` | Fill with a value | Zero-padding in CNNs |
| `edge` | Replicate border | Image processing |
| `reflect` | Mirror | Signal processing |
| `wrap` | Circular | Periodic data |

> **Interview Tip:** `np.pad` with `constant` mode is how **zero-padding** works in convolutional neural networks to preserve spatial dimensions.

---

## Question 30
- [ ] Done

**Write a function to compute the convolution of two matrices in NumPy.**

**Answer:**

```python
import numpy as np
from scipy.signal import convolve2d, fftconvolve

# Method 1: Manual 2D convolution
def convolve2d_manual(image, kernel):
    """2D convolution (valid mode)"""
    ih, iw = image.shape
    kh, kw = kernel.shape
    oh, ow = ih - kh + 1, iw - kw + 1
    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    return output

# Method 2: Stride tricks (vectorized, fast)
def convolve2d_vectorized(image, kernel):
    kh, kw = kernel.shape
    windows = np.lib.stride_tricks.sliding_window_view(image, (kh, kw))
    return np.einsum('ijkl,kl->ij', windows, kernel)

# Method 3: scipy (production use)
image = np.random.rand(10, 10)
kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # Sobel edge detection

result_valid = convolve2d(image, kernel, mode='valid')  # Output smaller
result_same = convolve2d(image, kernel, mode='same')    # Output same size
result_full = convolve2d(image, kernel, mode='full')    # Output larger

# Method 4: FFT-based (fastest for large kernels)
result_fft = fftconvolve(image, kernel, mode='same')

# Common kernels for image processing
blur_kernel = np.ones((3, 3)) / 9           # Box blur
sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])  # Sharpen
edge_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])  # Edge detect

# Test
image = np.random.rand(5, 5)
kernel = np.array([[1, 0], [0, -1]])
print(convolve2d_manual(image, kernel))
print(convolve2d_vectorized(image, kernel))
```

> **Interview Tip:** Convolution is the core operation in **CNNs**. For small kernels, direct computation is fine. For large kernels, **FFT-based convolution** is O(n log n) vs O(n²).

---

## Question 31
- [ ] Done

**Implement a script that computes the Fibonacci sequence using a NumPy matrix.**

**Answer:**

The Fibonacci sequence can be computed via matrix exponentiation:

$$\begin{pmatrix} F_{n+1} \\ F_n \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}^n \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$

```python
import numpy as np

# Method 1: Matrix exponentiation
def fibonacci_matrix(n):
    """Compute nth Fibonacci number using matrix power."""
    if n <= 0:
        return 0
    F = np.array([[1, 1], [1, 0]], dtype=np.int64)
    result = np.linalg.matrix_power(F, n)
    return result[0, 1]

# Single values
for i in range(10):
    print(fibonacci_matrix(i), end=' ')  # 0 1 1 2 3 5 8 13 21 34

# Method 2: Generate full sequence with matrix multiplication
def fibonacci_sequence(n):
    """Generate first n Fibonacci numbers."""
    F = np.array([[1, 1], [1, 0]], dtype=np.int64)
    result = [0, 1]
    state = np.array([1, 0], dtype=np.int64)
    for _ in range(2, n):
        state = F @ state
        result.append(state[0])
    return np.array(result)

print(fibonacci_sequence(15))
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

# Method 3: Golden ratio (continuous, approximate)
def fib_golden(n):
    phi = (1 + np.sqrt(5)) / 2
    return np.round((phi**n - (-phi)**(-n)) / np.sqrt(5)).astype(int)

fib_array = np.array([fib_golden(i) for i in range(15)])
```

> **Interview Tip:** Matrix exponentiation computes the $n$-th Fibonacci number in $O(\log n)$ time using fast matrix power. This technique generalizes to any linear recurrence.

---

## Question 32
- [ ] Done

**Write a code to replace all elements greater than a certain threshold in a NumPy array with a specific value.**

**Answer:**

```python
import numpy as np

arr = np.array([10, 25, 30, 5, 15, 40, 8, 50])
threshold = 20
replacement = -1

# Method 1: Boolean indexing (in-place modification)
arr_copy = arr.copy()
arr_copy[arr_copy > threshold] = replacement
print(arr_copy)  # [10, -1, -1, 5, 15, -1, 8, -1]

# Method 2: np.where() (creates new array)
result = np.where(arr > threshold, replacement, arr)
print(result)  # [10, -1, -1, 5, 15, -1, 8, -1]

# Method 3: np.clip() — cap at threshold
clipped = np.clip(arr, a_min=None, a_max=threshold)
print(clipped)  # [10, 20, 20, 5, 15, 20, 8, 20]

# Method 4: np.putmask() (in-place)
arr_copy2 = arr.copy()
np.putmask(arr_copy2, arr_copy2 > threshold, replacement)

# 2D array
arr2d = np.array([[10, 25], [30, 5]])
arr2d[arr2d > threshold] = replacement
# [[10, -1], [-1, 5]]

# Multiple conditions
result = np.where((arr > 10) & (arr < 40), 0, arr)
# [10, 0, 0, 5, 0, 40, 8, 50]

# Replace with different values based on condition
result = np.select(
    [arr < 10, arr < 30, arr >= 30],
    ['low', 'medium', 'high']
)
```

> **Interview Tip:** Boolean indexing modifies **in-place**. `np.where` creates a **new array**. `np.clip` is specifically designed for **capping values** at a threshold.

---

## Question 33
- [ ] Done

**Implement an efficient rolling window calculation for a 1D array using NumPy.**

**Answer:**

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
window = 3

# Method 1: sliding_window_view (NumPy 1.20+, recommended)
from numpy.lib.stride_tricks import sliding_window_view
windows = sliding_window_view(arr, window)
print(windows)
# [[1, 2, 3],
#  [2, 3, 4],
#  [3, 4, 5],
#  ...]

# Apply any aggregation
rolling_mean = np.mean(windows, axis=1)    # [2., 3., 4., ...]
rolling_std = np.std(windows, axis=1)
rolling_max = np.max(windows, axis=1)

# Method 2: as_strided (pre-1.20, advanced)
from numpy.lib.stride_tricks import as_strided
def rolling_window(arr, window_size):
    shape = (arr.shape[0] - window_size + 1, window_size)
    strides = (arr.strides[0], arr.strides[0])
    return as_strided(arr, shape=shape, strides=strides)

windows = rolling_window(arr, 3)

# Method 3: Cumsum trick (fastest for sum/mean)
def rolling_sum_cumsum(arr, window_size):
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    return cumsum[window_size:] - cumsum[:-window_size]

rolling_mean = rolling_sum_cumsum(arr, 3) / 3

# Method 4: 2D rolling window
def rolling_window_2d(arr, window_h, window_w):
    return sliding_window_view(arr, (window_h, window_w))

arr2d = np.arange(25).reshape(5, 5)
windows_2d = rolling_window_2d(arr2d, 3, 3)  # Shape: (3, 3, 3, 3)

# Performance (1M elements, window=100):
# sliding_window_view: ~5ms  (view, no copy)
# cumsum trick:        ~3ms  (fastest for sum)
# manual loop:         ~500ms
```

> **Interview Tip:** `sliding_window_view` creates a **view** (zero memory overhead). The cumsum trick is the fastest for sum/mean. These are critical for **time-series feature engineering**.

---

## Question 34
- [ ] Done

**Explain how you would implement gradient descent optimization with NumPy.**

**Answer:**

```python
import numpy as np

# === Linear Regression with Gradient Descent ===
def gradient_descent_linear(X, y, lr=0.01, epochs=1000):
    """Vanilla gradient descent for linear regression."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    losses = []
    
    for epoch in range(epochs):
        # Forward pass: predictions
        y_pred = X @ weights + bias
        
        # Loss: MSE
        loss = np.mean((y_pred - y) ** 2)
        losses.append(loss)
        
        # Gradients
        dw = (2/n_samples) * X.T @ (y_pred - y)   # dL/dw
        db = (2/n_samples) * np.sum(y_pred - y)    # dL/db
        
        # Update parameters
        weights -= lr * dw
        bias -= lr * db
    
    return weights, bias, losses

# === Stochastic Gradient Descent (SGD) ===
def sgd(X, y, lr=0.01, epochs=100, batch_size=32):
    n_samples, n_features = X.shape
    weights = np.random.randn(n_features) * 0.01
    bias = 0
    
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = indices[start:start+batch_size]
            X_batch, y_batch = X[idx], y[idx]
            
            y_pred = X_batch @ weights + bias
            dw = (2/len(idx)) * X_batch.T @ (y_pred - y_batch)
            db = (2/len(idx)) * np.sum(y_pred - y_batch)
            
            weights -= lr * dw
            bias -= lr * db
    return weights, bias

# === Adam Optimizer ===
def adam(X, y, lr=0.001, epochs=1000, beta1=0.9, beta2=0.999, eps=1e-8):
    n, d = X.shape
    w = np.zeros(d)
    b = 0
    m_w, v_w = np.zeros(d), np.zeros(d)  # Moments for weights
    m_b, v_b = 0, 0                       # Moments for bias
    
    for t in range(1, epochs + 1):
        y_pred = X @ w + b
        dw = (2/n) * X.T @ (y_pred - y)
        db = (2/n) * np.sum(y_pred - y)
        
        # Update moments
        m_w = beta1 * m_w + (1 - beta1) * dw
        v_w = beta2 * v_w + (1 - beta2) * dw**2
        m_b = beta1 * m_b + (1 - beta1) * db
        v_b = beta2 * v_b + (1 - beta2) * db**2
        
        # Bias correction
        m_w_hat = m_w / (1 - beta1**t)
        v_w_hat = v_w / (1 - beta2**t)
        m_b_hat = m_b / (1 - beta1**t)
        v_b_hat = v_b / (1 - beta2**t)
        
        w -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
        b -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
    return w, b

# Usage
np.random.seed(42)
X = np.random.randn(100, 3)
y = X @ np.array([2, -1, 0.5]) + 3 + np.random.randn(100) * 0.1
w, b, losses = gradient_descent_linear(X, y, lr=0.1, epochs=500)
print(f"Weights: {w}, Bias: {b:.2f}")  # Close to [2, -1, 0.5], 3.0
```

> **Interview Tip:** Know three variants: **Batch GD** (full dataset), **SGD** (random samples), **Mini-batch** (batches of 32-128). **Adam** is the default optimizer in deep learning due to adaptive learning rates.

## Question 35
- [ ] Done

**How do you concatenate two arrays in NumPy ?**

### Array Concatenation in NumPy

```python
import numpy as np

# 1D concatenation
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = np.concatenate([a, b])
print(result)  # [1 2 3 4 5 6]

# 2D concatenation along rows (axis=0, default)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

row_concat = np.concatenate([A, B], axis=0)
print(row_concat)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]  → shape (4, 2)

# 2D concatenation along columns (axis=1)
col_concat = np.concatenate([A, B], axis=1)
print(col_concat)
# [[1 2 5 6]
#  [3 4 7 8]]  → shape (2, 4)
```

### Concatenation Methods Comparison

| Method | Description | Equivalent |
|--------|-------------|------------|
| `np.concatenate([a,b], axis=0)` | General-purpose join | Flexible axis |
| `np.vstack([a,b])` | Vertical stack (row-wise) | `concatenate` axis=0 |
| `np.hstack([a,b])` | Horizontal stack (column-wise) | `concatenate` axis=1 |
| `np.dstack([a,b])` | Depth stack (3rd axis) | `concatenate` axis=2 |
| `np.append(a, b)` | Append (flattens by default) | Less efficient |

```python
# Multiple arrays at once
c = np.array([7, 8, 9])
result = np.concatenate([a, b, c])
print(result)  # [1 2 3 4 5 6 7 8 9]

# Important: arrays must match in all dimensions except the concatenation axis
A = np.ones((3, 4))   # 3×4
B = np.zeros((2, 4))  # 2×4
result = np.concatenate([A, B], axis=0)  # OK → 5×4
# np.concatenate([A, B], axis=1)  # ERROR: mismatch on axis 0 (3 vs 2)
```

> **Interview Tip:** `np.concatenate` is the most flexible. Use `vstack`/`hstack` for readability. For building datasets row by row, prefer collecting in a list and calling `np.array()` once — repeated concatenation is O(n²).

---

## Question 36
- [ ] Done

**How do you calculate the eigenvalues and eigenvectors of a matrix in NumPy ?**

### Eigenvalues and Eigenvectors

For a square matrix $A$, eigenvalues $\lambda$ and eigenvectors $v$ satisfy: $Av = \lambda v$

```python
import numpy as np

# Symmetric matrix (common in ML: covariance matrices)
A = np.array([[4, 2],
              [2, 3]])

# Method 1: np.linalg.eig (general matrices)
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")     # [5. 2.]
print(f"Eigenvectors:\n{eigenvectors}")  # columns are eigenvectors

# Method 2: np.linalg.eigh (symmetric/Hermitian — faster, numerically stable)
eigenvalues, eigenvectors = np.linalg.eigh(A)  # sorted ascending
print(f"Eigenvalues: {eigenvalues}")     # [2. 5.]

# Verify: A @ v = lambda * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lv = eigenvalues[i] * v
    av = A @ v
    print(f"λ={eigenvalues[i]:.1f}: A@v = {av}, λ*v = {lv}")  # should match
```

### ML Applications

```python
# PCA using eigendecomposition
def pca_eigen(X, n_components=2):
    """PCA via eigendecomposition of covariance matrix."""
    X_centered = X - X.mean(axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by largest eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Project data
    components = eigenvectors[:, :n_components]
    X_pca = X_centered @ components
    
    explained_variance = eigenvalues[:n_components] / eigenvalues.sum()
    return X_pca, explained_variance

X = np.random.randn(100, 5)
X_pca, var_explained = pca_eigen(X, n_components=2)
print(f"Variance explained: {var_explained}")
```

### Key Functions
| Function | Use Case |
|----------|----------|
| `np.linalg.eig(A)` | General square matrices |
| `np.linalg.eigh(A)` | Symmetric matrices (PCA, covariance) |
| `np.linalg.eigvals(A)` | Eigenvalues only (faster) |
| `np.linalg.svd(A)` | SVD (related, works on non-square) |

> **Interview Tip:** Use `eigh` for symmetric matrices (covariance, kernel matrices) — it's faster and returns sorted real eigenvalues. For large sparse matrices, use `scipy.sparse.linalg.eigsh`. Eigendecomposition is the foundation of **PCA**, **spectral clustering**, and **graph algorithms**.

---

## Question 37
- [ ] Done

**Explain how to generate random data with NumPy**

### Random Data Generation in NumPy

### Modern API (`np.random.default_rng` — recommended)

```python
import numpy as np

rng = np.random.default_rng(seed=42)  # reproducible

# Uniform distribution [0, 1)
rng.random(5)                        # 5 random floats
rng.random((3, 4))                   # 3×4 matrix

# Uniform integers
rng.integers(0, 10, size=5)          # 5 integers in [0, 10)
rng.integers(1, 7, size=(3, 3))      # 3×3 dice rolls

# Normal (Gaussian) distribution
rng.normal(loc=0, scale=1, size=1000)     # standard normal
rng.normal(loc=100, scale=15, size=100)   # IQ-like distribution

# Other distributions
rng.uniform(low=0, high=1, size=10)       # uniform [low, high)
rng.exponential(scale=1.0, size=10)       # exponential
rng.poisson(lam=5, size=10)               # Poisson
rng.binomial(n=10, p=0.5, size=10)        # binomial
rng.choice([1, 2, 3, 4], size=3)          # random selection
```

### Legacy API (`np.random.*` — still common)

```python
np.random.seed(42)
np.random.rand(3, 4)           # uniform [0, 1), shape (3, 4)
np.random.randn(3, 4)          # standard normal, shape (3, 4)
np.random.randint(0, 10, (3,)) # integers in [0, 10)
np.random.choice(5, 3)         # 3 choices from [0, 5)
np.random.shuffle(arr)         # in-place shuffle
np.random.permutation(10)      # shuffled range
```

### ML Use Cases

```python
rng = np.random.default_rng(42)

# Generate synthetic classification data
n_samples = 1000
X = np.vstack([
    rng.normal(loc=[0, 0], scale=1, size=(n_samples//2, 2)),  # class 0
    rng.normal(loc=[3, 3], scale=1, size=(n_samples//2, 2)),  # class 1
])
y = np.array([0] * (n_samples//2) + [1] * (n_samples//2))

# Weight initialization for neural networks
def xavier_init(fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out))

# Bootstrap sampling
def bootstrap_sample(X, y):
    idx = rng.integers(0, len(X), size=len(X))  # sample with replacement
    return X[idx], y[idx]
```

> **Interview Tip:** Always use `np.random.default_rng(seed)` for reproducibility. The legacy `np.random.seed()` uses global state (not thread-safe). For ML, random generation is critical for **weight initialization**, **data augmentation**, **dropout**, and **cross-validation splits**.

---

## Question 38
- [ ] Done

**How do you stack multiple arrays vertically and horizontally ?**

### Stacking Arrays in NumPy

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])

# Vertical stacking (row-wise) — np.vstack
result = np.vstack([a, b, c])
print(result)
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]  → shape (3, 3)

# Horizontal stacking (column-wise) — np.hstack
result = np.hstack([a, b, c])
print(result)
# [1 2 3 4 5 6 7 8 9]  → shape (9,)

# For 2D arrays
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.vstack([A, B]))
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]  → shape (4, 2)

print(np.hstack([A, B]))
# [[1 2 5 6]
#  [3 4 7 8]]  → shape (2, 4)
```

### All Stacking Methods

| Method | Axis | Input 1D | Input 2D |
|--------|------|----------|----------|
| `np.vstack` | Rows (axis=0) | Stacks as rows | Appends rows |
| `np.hstack` | Columns (axis=1) | Concatenates flat | Appends columns |
| `np.dstack` | Depth (axis=2) | 3D stack | 3D stack |
| `np.column_stack` | Columns | Treats 1D as columns | Same as hstack |
| `np.row_stack` | Rows | Same as vstack | Same as vstack |
| `np.stack` | New axis | Creates new dimension | Creates new dimension |

```python
# np.stack creates a NEW dimension
result = np.stack([a, b, c], axis=0)  # shape (3, 3) — like vstack
result = np.stack([a, b, c], axis=1)  # shape (3, 3) — transposed

# column_stack: treats 1D arrays as columns
result = np.column_stack([a, b, c])
print(result)
# [[1 4 7]
#  [2 5 8]
#  [3 6 9]]  → shape (3, 3)

# ML example: stack features into a feature matrix
age = np.array([25, 30, 35])
income = np.array([50000, 60000, 70000])
score = np.array([0.8, 0.6, 0.9])

X = np.column_stack([age, income, score])
print(X.shape)  # (3, 3) — 3 samples, 3 features
```

> **Interview Tip:** `vstack`/`hstack` are most commonly used. Remember that `np.stack` creates a **new dimension** (useful for batching), while `vstack`/`hstack` concatenate along existing dimensions. For building feature matrices, `np.column_stack` is the most intuitive.

---
