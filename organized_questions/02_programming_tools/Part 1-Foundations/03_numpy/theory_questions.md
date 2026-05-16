# NumPy Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What is NumPy , and why is it important in Machine Learning ?**

### Definition
NumPy (Numerical Python) is the foundational library for numerical computing in Python. Its core feature is the **ndarray** - a fast, memory-efficient multi-dimensional array object.

### Why NumPy is Important for ML

**1. Performance**
- **Vectorization**: Operations on entire arrays without Python loops
- **C Backend**: Operations implemented in optimized C/Fortran code
- **10-100x faster** than Python lists for numerical operations

**2. Universal Data Format**
- All ML libraries (Pandas, Scikit-learn, TensorFlow, PyTorch) use NumPy arrays
- Dataset → 2D array (samples × features)
- Image → 3D array (height × width × channels)
- Model weights → ndarray

**3. Mathematical Tools**
- Linear algebra (matrix multiplication, SVD, eigenvalues)
- Statistics (mean, std, variance)
- Random number generation (weight initialization)

---

## Question 2
- [ ] Done

**Explain how NumPy arrays are different from Python lists**

### Comparison Table

| Feature | Python List | NumPy Array |
|---------|-------------|-------------|
| Data Types | Heterogeneous (mixed) | Homogeneous (same type) |
| Memory | Pointers to objects | Contiguous block |
| Performance | Slow (Python loops) | Fast (vectorized C) |
| Functionality | General purpose | Numerical computing |

### Code Example
```python
import numpy as np
import sys

# Memory comparison
py_list = list(range(1000))
np_array = np.arange(1000)

print(f"List size: {sys.getsizeof(py_list)} bytes")   # ~9000 bytes
print(f"Array size: {np_array.nbytes} bytes")          # ~8000 bytes

# Performance: NumPy is 10-100x faster
# Python: [x*2 for x in py_list]  
# NumPy:  np_array * 2  (vectorized)
```

---

## Question 3
- [ ] Done

**What are the main attributes of a NumPy ndarray?**

### Key Attributes

| Attribute | Description | Example |
|-----------|-------------|---------|
| `ndim` | Number of dimensions | `2` for matrix |
| `shape` | Size of each dimension | `(3, 4)` = 3 rows, 4 cols |
| `size` | Total elements | `12` (3 × 4) |
| `dtype` | Data type | `float64`, `int32` |
| `itemsize` | Bytes per element | `8` for float64 |
| `nbytes` | Total bytes | `96` (12 × 8) |
| `T` | Transpose | Swaps axes |

### Code Example
```python
import numpy as np

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8]], dtype=np.float32)

print(f"ndim: {arr.ndim}")        # 2
print(f"shape: {arr.shape}")      # (2, 4)
print(f"size: {arr.size}")        # 8
print(f"dtype: {arr.dtype}")      # float32
print(f"itemsize: {arr.itemsize}")  # 4 bytes
print(f"nbytes: {arr.nbytes}")    # 32 bytes
```

---

## Question 4
- [ ] Done

**Explain the concept of broadcasting in NumPy.**

### Definition
Broadcasting allows operations on arrays of different shapes by "stretching" smaller arrays to match larger ones (virtually, without copying data).

### Broadcasting Rules
Compare shapes from right to left. Dimensions are compatible if:
1. They are equal, OR
2. One of them is 1

### Examples
```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])  # Shape (3, 3)

b = np.array([10, 20, 30])  # Shape (3,)

# b is broadcast across each row
result = A + b
# [[11, 22, 33],
#  [14, 25, 36],
#  [17, 28, 39]]

# Column vector broadcast
c = np.array([[100], [200], [300]])  # Shape (3, 1)
result2 = A + c  # c broadcast across columns
```

### Shape Compatibility
```
A: (3, 3)       A: (4, 3)       A: (3, 3)
b: (3,)    ✓    b: (3,)    ✓    b: (4,)    ✗
```

---

## Question 5
- [ ] Done

**What are the data types supported by NumPy arrays ?**

### Common dtypes

| Category | Types | Notes |
|----------|-------|-------|
| Integers | `int8`, `int16`, `int32`, `int64` | Signed |
| Unsigned | `uint8`, `uint16`, `uint32`, `uint64` | Non-negative (images use uint8) |
| Floats | `float16`, `float32`, `float64` | float32 for GPU/ML |
| Boolean | `bool_` | True/False |
| Complex | `complex64`, `complex128` | Real + imaginary |

### Code Example
```python
import numpy as np

# Specify dtype at creation
arr_int8 = np.array([1, 2, 3], dtype=np.int8)
arr_float32 = np.array([1.0, 2.0], dtype=np.float32)

# Type casting
arr = np.array([1, 2, 3])
arr_float = arr.astype(np.float64)
```

### ML Tip
Use `float32` instead of `float64` to save memory and speed up GPU training.

---

## Question 6
- [ ] Done

**What is the difference between a deep copy and a shallow copy in NumPy ?**

### Comparison

| Type | Creates New Data? | Affects Original? |
|------|------------------|-------------------|
| Assignment (`b = a`) | No | Yes (same object) |
| View/Slice (`b = a[1:3]`) | No | Yes (shares data) |
| Copy (`b = a.copy()`) | Yes | No (independent) |

### Code Example
```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])

# View (shallow copy) - shares memory
view = a[1:4]
view[0] = 99
print(a)  # [1, 99, 3, 4, 5] - original changed!

# Copy (deep copy) - independent
a = np.array([1, 2, 3, 4, 5])
copy = a.copy()
copy[0] = 77
print(a)  # [1, 2, 3, 4, 5] - original unchanged
```

### Check if View
```python
print(view.base is a)  # True = view
print(copy.base)       # None = copy
```

---

## Question 7
- [ ] Done

**What are universal functions ( ufuncs ) in NumPy ?**

### Definition
Ufuncs are vectorized functions that operate element-by-element on arrays, implemented in fast C code.

### Types

| Type | Examples | Description |
|------|----------|-------------|
| Unary | `np.sqrt()`, `np.exp()`, `np.sin()` | One input array |
| Binary | `np.add()`, `np.multiply()`, `np.maximum()` | Two input arrays |

### Code Example
```python
import numpy as np

a = np.array([1, 4, 9, 16])

# Unary ufuncs
print(np.sqrt(a))  # [1., 2., 3., 4.]
print(np.exp(a))   # Exponential

# Binary ufuncs
b = np.array([1, 2, 3, 4])
print(np.add(a, b))      # Same as a + b
print(np.maximum(a, b))  # Element-wise max
```

---

## Question 8
- [ ] Done

**What is the use of the axis parameter in NumPy functions?**

### Definition
`axis` specifies which dimension to operate along. That dimension is "collapsed."

### For 2D Arrays
- `axis=0`: Down rows (column-wise) → result has shape (n_cols,)
- `axis=1`: Across columns (row-wise) → result has shape (n_rows,)
- `axis=None`: All elements → scalar

### Code Example
```python
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])  # Shape (2, 3)

print(arr.sum())         # 21 (all elements)
print(arr.sum(axis=0))   # [5, 7, 9] (sum each column)
print(arr.sum(axis=1))   # [6, 15] (sum each row)

# Mean per column (useful for feature normalization)
print(arr.mean(axis=0))  # [2.5, 3.5, 4.5]
```

### ML Use Case
```python
# Normalize features (subtract column mean)
X_normalized = X - X.mean(axis=0)
```



---

## Question 9
- [ ] Done

**Explain the use of slicing and indexing with NumPy arrays.**

### Overview
Slicing and indexing are fundamental mechanisms for accessing and modifying elements in NumPy arrays. Unlike Python lists, NumPy supports **advanced indexing** techniques that enable powerful data selection without explicit loops.

### Types of Indexing

| Type | Syntax | Description |
|------|--------|-------------|
| Basic Indexing | `a[0]`, `a[1, 2]` | Access single element by position |
| Slicing | `a[start:stop:step]` | Extract sub-array (returns a **view**) |
| Boolean Indexing | `a[a > 5]` | Filter elements by condition (returns a **copy**) |
| Fancy Indexing | `a[[0, 3, 5]]` | Select elements by index array (returns a **copy**) |
| Ellipsis | `a[..., 0]` | Shortcut for multiple `:` across dimensions |

### Key Concepts
- **Slicing returns a view** — modifying the slice modifies the original array
- **Boolean and fancy indexing return copies** — modifications do not affect the original
- **Negative indexing** works the same as Python lists (`a[-1]` = last element)
- Multi-dimensional arrays use comma-separated indices: `a[row, col]`

### Code Example
```python
import numpy as np

arr = np.array([[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]])

# Basic indexing
print(arr[0, 1])        # 20

# Slicing (rows 0-1, all columns)
print(arr[:2, :])       # [[10, 20, 30], [40, 50, 60]]

# Step slicing
print(arr[::2, :])      # [[10, 20, 30], [70, 80, 90]]

# Boolean indexing
print(arr[arr > 50])    # [60, 70, 80, 90]

# Fancy indexing
print(arr[[0, 2], :])   # [[10, 20, 30], [70, 80, 90]]

# Combining: rows where column 0 > 30
print(arr[arr[:, 0] > 30])  # [[40, 50, 60], [70, 80, 90]]
```

### Interview Tip
Always mention the **view vs. copy** distinction — slicing creates a view (shared memory), while boolean and fancy indexing create copies. This is a common source of bugs and a frequent interview question.

---

## Question 10
- [ ] Done

**What is the purpose of the NumPy histogram function?**

### Definition
`np.histogram()` computes the frequency distribution of data by dividing values into bins and counting how many fall into each bin. Unlike Matplotlib's `plt.hist()`, NumPy's version **only computes** the histogram — it does not plot it.

### Function Signature
```python
np.histogram(a, bins=10, range=None, density=False, weights=None)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `a` | Input data (flattened automatically) |
| `bins` | Number of bins (int) or bin edges (array) |
| `range` | `(min, max)` range of bins |
| `density` | If `True`, returns probability density instead of counts |
| `weights` | Array of weights for each element |

### Returns
- `counts` — array of frequency values (length = number of bins)
- `bin_edges` — array of bin boundaries (length = bins + 1)

### Code Example
```python
import numpy as np

data = np.random.randn(1000)  # 1000 standard normal values

# Basic histogram
counts, bin_edges = np.histogram(data, bins=10)
print(f"Counts: {counts}")
print(f"Bin edges: {bin_edges}")

# Probability density
density, edges = np.histogram(data, bins=20, density=True)

# Custom bin edges
counts, edges = np.histogram(data, bins=[-3, -1, 0, 1, 3])

# Related: np.histogram2d() for 2D data
# Related: np.histogramdd() for N-dimensional data
```

### Interview Tip
Emphasize that `np.histogram()` is a **computation-only** function — it returns arrays, not a plot. For visualization, pass the result to Matplotlib. This separation makes it useful in data pipelines where plotting is not needed.

---

## Question 11
- [ ] Done

**What is the difference between `np.var()` and `np.std()`?**

### Definition
Both functions measure the **spread** (dispersion) of data, but they differ in scale:
- `np.var()` — computes the **variance** (average of squared deviations)
- `np.std()` — computes the **standard deviation** (square root of variance)

### Mathematical Formulas

| Function | Formula | Unit |
|----------|---------|------|
| `np.var()` | $\sigma^2 = \frac{1}{N} \sum_{i=1}^{N}(x_i - \bar{x})^2$ | Squared units |
| `np.std()` | $\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N}(x_i - \bar{x})^2}$ | Same units as data |

### Key Parameter: `ddof`
- `ddof=0` (default) → **Population** variance/std (divides by N)
- `ddof=1` → **Sample** variance/std (divides by N−1, Bessel's correction)

### Code Example
```python
import numpy as np

data = np.array([4, 8, 6, 5, 3, 7, 9, 2])

# Population (default ddof=0)
print(f"Variance: {np.var(data)}")      # 5.25
print(f"Std Dev:  {np.std(data)}")      # 2.2913...

# Sample (ddof=1) — used in statistics
print(f"Sample Var: {np.var(data, ddof=1)}")  # 6.0
print(f"Sample Std: {np.std(data, ddof=1)}")  # 2.4494...

# Relationship
print(np.std(data) == np.sqrt(np.var(data)))  # True

# Along axis
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(np.std(arr, axis=0))  # Per-column std: [1.5, 1.5, 1.5]
print(np.std(arr, axis=1))  # Per-row std: [0.816, 0.816]
```

### Interview Tip
Always mention `ddof` — the default `ddof=0` gives population statistics, but for ML cross-validation and statistical tests you often need `ddof=1` for unbiased sample estimates. Standard deviation is preferred for interpretation because it is in the **same units** as the original data.

---

## Question 12
- [ ] Done

**What is the concept of vectorization in NumPy?**

### Definition
Vectorization is the technique of replacing explicit Python loops with **array-level operations** that are executed internally by optimized C/Fortran code. This leverages SIMD (Single Instruction, Multiple Data) instructions and contiguous memory access for massive speed gains.

### Why Vectorization Is Fast

| Factor | Python Loop | Vectorized NumPy |
|--------|-------------|-------------------|
| Interpreter overhead | Per-element | Single call |
| Type checking | Every iteration | Once |
| Memory access | Scattered (objects) | Contiguous (buffer) |
| CPU optimization | None | SIMD / cache-friendly |
| Typical speedup | 1x (baseline) | 10–100x |

### Code Example
```python
import numpy as np
import time

arr = np.random.rand(1_000_000)

# ---- Non-vectorized (slow) ----
start = time.time()
result = [x ** 2 + 2 * x + 1 for x in arr]
print(f"Loop: {time.time() - start:.4f}s")

# ---- Vectorized (fast) ----
start = time.time()
result = arr ** 2 + 2 * arr + 1
print(f"Vectorized: {time.time() - start:.4f}s")

# Common vectorized patterns in ML
X = np.random.rand(1000, 10)
w = np.random.rand(10)

# Dot product (linear layer)
y = X @ w                     # Matrix-vector multiplication

# Softmax (vectorized)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
```

### Interview Tip
Emphasize that vectorization is the **#1 performance principle** in NumPy. If you find yourself writing a `for` loop over array elements, there is almost always a vectorized alternative. Mention that ML operations like dot products, activation functions, and loss computations are all vectorized.

---

## Question 13
- [ ] Done

**Explain the term “ stride ” in the context of NumPy arrays**

### Definition
Strides are the **number of bytes** that must be skipped in memory to move to the next element along each dimension of an ndarray. They define how the flat, contiguous memory buffer is interpreted as a multi-dimensional array.

### How Strides Work
For an array with shape `(rows, cols)` and `itemsize` bytes per element:
- **Stride for axis 0 (rows):** `cols × itemsize` bytes to jump to the next row
- **Stride for axis 1 (cols):** `itemsize` bytes to jump to the next column

### Memory Layout: C-order vs. Fortran-order

| Order | Name | Row/Col Major | Strides (3×4, int32) |
|-------|------|---------------|----------------------|
| C (default) | Row-major | Rows contiguous | `(16, 4)` |
| F | Column-major | Columns contiguous | `(4, 12)` |

### Code Example
```python
import numpy as np

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [7, 8, 9, 10]], dtype=np.int32)

print(f"Shape: {arr.shape}")        # (3, 4)
print(f"Strides: {arr.strides}")    # (16, 4)
print(f"Itemsize: {arr.itemsize}")  # 4 bytes

# Stride for axis 0 = 4 cols × 4 bytes = 16
# Stride for axis 1 = 1 element × 4 bytes = 4

# Fortran-order array
arr_f = np.asfortranarray(arr)
print(f"F-order strides: {arr_f.strides}")  # (4, 12)

# Transposing swaps strides (no data copy!)
print(f"Transpose strides: {arr.T.strides}")  # (4, 16)
```

### Interview Tip
Strides explain why operations like **transpose** and **slicing** are O(1) — they only change the stride/shape metadata without copying data. Understanding strides is key to understanding NumPy's memory model and performance characteristics.

---

## Question 14
- [ ] Done

**How does NumPy handle data types to optimize memory use?**

### Overview
NumPy optimizes memory by enforcing **homogeneous data types** within each array. Every element occupies the exact same number of bytes, stored in a contiguous memory block. This eliminates the per-object overhead of Python and enables fine-grained control over memory consumption.

### Memory Optimization Strategies

| Strategy | Description | Example |
|----------|-------------|--------|
| Dtype selection | Choose smallest sufficient type | `float32` vs `float64` (halves memory) |
| Downcasting | Convert to smaller type after creation | `arr.astype(np.float16)` |
| Structured arrays | Mix types in a single array | Named fields with individual dtypes |
| Memory-mapped files | Load portions of large files on demand | `np.memmap()` |
| Views | Reinterpret data without copying | `arr.view(np.uint8)` |

### Memory Comparison by dtype

| dtype | Bytes/Element | 1M Elements |
|-------|---------------|-------------|
| `float64` | 8 | 8 MB |
| `float32` | 4 | 4 MB |
| `float16` | 2 | 2 MB |
| `int8` | 1 | 1 MB |
| `bool_` | 1 | 1 MB |

### Code Example
```python
import numpy as np

# Default float64 (8 bytes per element)
arr64 = np.random.rand(1_000_000)
print(f"float64: {arr64.nbytes / 1e6:.1f} MB")  # 8.0 MB

# Downcast to float32 (4 bytes)
arr32 = arr64.astype(np.float32)
print(f"float32: {arr32.nbytes / 1e6:.1f} MB")  # 4.0 MB

# Specify dtype at creation
arr16 = np.zeros(1_000_000, dtype=np.float16)
print(f"float16: {arr16.nbytes / 1e6:.1f} MB")  # 2.0 MB

# Structured array (like a record)
dt = np.dtype([('name', 'U10'), ('age', np.int8), ('score', np.float32)])
records = np.array([('Alice', 25, 95.5), ('Bob', 30, 88.0)], dtype=dt)
print(records['age'])  # [25, 30]

# Check and minimize
print(np.iinfo(np.int8))   # min=-128, max=127
print(np.finfo(np.float16))  # precision and range
```

### Interview Tip
In ML, always prefer `float32` over `float64` — GPUs are optimized for 32-bit, and models rarely benefit from 64-bit precision. For image data, `uint8` (0–255) saves 8x memory over `float64`. Mention `np.can_cast()` and `np.promote_types()` for safe type conversions.

---

## Question 15
- [ ] Done

**What are NumPy strides , and how do they affect array manipulation ?**

### Definition
Strides are a tuple of integers indicating the **number of bytes to step** in each dimension when traversing an array. They are a core part of NumPy's memory model and determine how the underlying flat data buffer is viewed as a multi-dimensional structure.

### How Strides Affect Array Manipulation

| Operation | Effect on Strides | Data Copied? |
|-----------|-------------------|--------------|
| Slicing `a[::2]` | Strides doubled | No (view) |
| Transpose `a.T` | Strides reversed | No (view) |
| Reshape (compatible) | Strides recalculated | No (view) |
| Reshape (incompatible) | N/A | Yes (copy) |
| Flatten `.ravel()` | Simplified to 1D | Maybe (view if C-contiguous) |
| `.flatten()` | Simplified to 1D | Always (copy) |

### Code Example
```python
import numpy as np

arr = np.arange(12, dtype=np.int32).reshape(3, 4)
print(f"Shape: {arr.shape}")      # (3, 4)
print(f"Strides: {arr.strides}")  # (16, 4)

# Slicing every other element doubles the stride
sliced = arr[:, ::2]
print(f"Sliced shape: {sliced.shape}")    # (3, 2)
print(f"Sliced strides: {sliced.strides}")  # (16, 8)

# Transpose reverses strides
transposed = arr.T
print(f"Transposed strides: {transposed.strides}")  # (4, 16)

# Check contiguity
print(f"C-contiguous: {arr.flags['C_CONTIGUOUS']}")      # True
print(f"T C-contiguous: {transposed.flags['C_CONTIGUOUS']}")  # False

# Reshape (view when possible)
reshaped = arr.reshape(4, 3)
print(f"Reshaped strides: {reshaped.strides}")  # (12, 4)
print(f"Is view: {reshaped.base is arr}")       # True
```

### Interview Tip
Strides explain **why NumPy is fast**: operations like transpose, slicing, and reshaping change only metadata (shape + strides), not data. However, non-contiguous arrays (e.g., after transposing) can be slower for subsequent computations because they break cache locality. Use `.copy()` to make a contiguous copy when performance matters.

---

## Question 16
- [ ] Done

**Explain the concept and use of masked arrays in NumPy.**

### Definition
Masked arrays (`numpy.ma` module) are arrays with an attached Boolean **mask** that marks certain elements as invalid or missing. Operations on masked arrays automatically **ignore masked elements**, making them ideal for handling incomplete or corrupted data.

### Key Concepts
- A masked array has two components: **data** (the values) and **mask** (Boolean array)
- `mask=True` means the element is **invalid/hidden**
- `mask=False` means the element is **valid/used**
- All NumPy operations respect the mask automatically

### Common Use Cases

| Use Case | Description |
|----------|-------------|
| Missing data | Sensor readings with gaps |
| Invalid values | Division by zero, NaN, outliers |
| Conditional exclusion | Ignoring specific values in statistics |
| Image processing | Excluding regions of interest |

### Code Example
```python
import numpy as np
import numpy.ma as ma

# Create a masked array
data = np.array([1, 2, -999, 4, 5, -999, 7])
mask = (data == -999)  # Mask sentinel values

masked_arr = ma.masked_array(data, mask=mask)
print(masked_arr)       # [1 2 -- 4 5 -- 7]

# Operations ignore masked elements
print(masked_arr.mean())   # 3.8 (only valid elements)
print(masked_arr.sum())    # 19
print(masked_arr.count())  # 5 (number of valid elements)

# Convenience constructors
arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
masked_nan = ma.masked_invalid(arr)   # Mask NaN/Inf
print(masked_nan.mean())              # 3.0

# Mask values greater than a threshold
masked_gt = ma.masked_greater(np.array([1, 5, 3, 8, 2]), 4)
print(masked_gt)  # [1 -- 3 -- 2]

# Fill masked values for export
filled = masked_arr.filled(fill_value=0)
print(filled)  # [1 2 0 4 5 0 7]
```

### Interview Tip
Masked arrays are the NumPy-native way to handle missing data (before Pandas `NaN`). Mention that they are especially useful in scientific computing where sentinel values (e.g., `-999`) indicate missing readings. The key advantage is that **all statistical functions automatically skip masked values** without any extra code.

---

## Question 17
- [ ] Done

**What are the functions available for padding arrays in NumPy?**

### Overview
NumPy provides `np.pad()` as the primary function for extending arrays by adding elements along their edges. Padding is essential in image processing (convolutions), signal processing, and preparing data for neural networks.

### `np.pad()` Signature
```python
np.pad(array, pad_width, mode='constant', **kwargs)
```

### Padding Modes

| Mode | Description | Example (1D: `[1, 2, 3]`, pad=2) |
|------|-------------|-----------------------------------|
| `constant` | Fill with a constant value (default 0) | `[0, 0, 1, 2, 3, 0, 0]` |
| `edge` | Repeat edge values | `[1, 1, 1, 2, 3, 3, 3]` |
| `reflect` | Mirror reflection (excludes edge) | `[3, 2, 1, 2, 3, 2, 1]` |
| `symmetric` | Mirror reflection (includes edge) | `[2, 1, 1, 2, 3, 3, 2]` |
| `wrap` | Circular/periodic wrapping | `[2, 3, 1, 2, 3, 1, 2]` |
| `linear_ramp` | Linear ramp to end value | `[0, 0.5, 1, 2, 3, 1.5, 0]` |
| `mean` | Fill with mean of array | `[2, 2, 1, 2, 3, 2, 2]` |
| `maximum` | Fill with max of array | `[3, 3, 1, 2, 3, 3, 3]` |
| `minimum` | Fill with min of array | `[1, 1, 1, 2, 3, 1, 1]` |

### Code Example
```python
import numpy as np

arr = np.array([[1, 2], [3, 4]])

# Zero padding (common in CNNs)
padded = np.pad(arr, pad_width=1, mode='constant', constant_values=0)
print(padded)
# [[0 0 0 0]
#  [0 1 2 0]
#  [0 3 4 0]
#  [0 0 0 0]]

# Different padding per axis: ((top, bottom), (left, right))
padded2 = np.pad(arr, ((1, 0), (0, 2)), mode='edge')
print(padded2)
# [[1 2 2 2]
#  [1 2 2 2]
#  [3 4 4 4]]

# Reflect padding (used in image processing)
padded3 = np.pad(arr, 1, mode='reflect')
print(padded3)
# [[4 3 4 3]
#  [2 1 2 1]
#  [4 3 4 3]
#  [2 1 2 1]]
```

### Interview Tip
Padding is critical in **convolutional neural networks** (CNNs) to preserve spatial dimensions. Zero padding (`mode='constant'`) is the most common. Mention that `pad_width` can be specified per-axis as a tuple of `(before, after)` pairs for asymmetric padding.

---

## Question 18
- [ ] Done

**Describe how you can use NumPy for simulating Monte Carlo experiments.**

### Definition
Monte Carlo simulation uses **repeated random sampling** to estimate mathematical quantities that may be difficult to compute analytically. NumPy's fast random number generation and vectorized operations make it ideal for running millions of simulations efficiently.

### Key NumPy Functions for Monte Carlo

| Function | Purpose |
|----------|--------|
| `np.random.rand()` | Uniform random numbers [0, 1) |
| `np.random.randn()` | Standard normal distribution |
| `np.random.choice()` | Random selection from array |
| `np.random.binomial()` | Binomial distribution |
| `np.cumsum()` | Cumulative sum (e.g., random walks) |
| `np.mean()`, `np.std()` | Aggregate simulation results |

### Code Example
```python
import numpy as np

# --- Example 1: Estimate Pi ---
n_points = 1_000_000
x = np.random.rand(n_points)
y = np.random.rand(n_points)

# Points inside unit circle: x^2 + y^2 <= 1
inside_circle = (x**2 + y**2) <= 1.0
pi_estimate = 4 * np.mean(inside_circle)
print(f"Pi ≈ {pi_estimate:.4f}")  # ~3.1416

# --- Example 2: Stock Price Simulation (Geometric Brownian Motion) ---
np.random.seed(42)
S0 = 100        # Initial price
mu = 0.05       # Expected return
sigma = 0.2     # Volatility
T = 1.0         # 1 year
n_steps = 252   # Trading days
n_simulations = 10_000

dt = T / n_steps
returns = np.random.normal(mu * dt, sigma * np.sqrt(dt),
                           size=(n_steps, n_simulations))
price_paths = S0 * np.exp(np.cumsum(returns, axis=0))

final_prices = price_paths[-1]
print(f"Mean final price: ${np.mean(final_prices):.2f}")
print(f"5th percentile (VaR): ${np.percentile(final_prices, 5):.2f}")

# --- Example 3: Coin Flip ---
n_flips = 10_000
results = np.random.choice(['H', 'T'], size=n_flips)
print(f"Heads probability: {np.mean(results == 'H'):.4f}")  # ~0.5
```

### Interview Tip
Monte Carlo is widely asked in finance and ML interviews. Emphasize that NumPy enables **vectorized simulation** — generating all random samples at once as arrays rather than looping — which is orders of magnitude faster than Python loops. Mention practical uses: estimating integrals, option pricing, Bayesian inference, and reinforcement learning.

---

## Question 19
- [ ] Done

**Explain how to resolve the MemoryError when working with very large arrays in NumPy.**

### Overview
A `MemoryError` occurs when NumPy tries to allocate more memory than the system has available. This is common with large datasets in ML. There are several strategies to handle and prevent it.

### Strategies to Resolve MemoryError

| Strategy | Description | Memory Savings |
|----------|-------------|---------------|
| Use smaller dtypes | `float32` instead of `float64` | 50% |
| Memory-mapped files | Load only needed portions from disk | Up to 99% |
| Chunked processing | Process data in batches | Proportional to chunk size |
| Sparse arrays | Store only non-zero elements | Depends on sparsity |
| In-place operations | Avoid creating intermediate copies | ~50% per operation |
| Generators | Stream data instead of loading all | Nearly 100% |
| 64-bit Python | Access >4 GB RAM | Removes 32-bit limit |

### Code Example
```python
import numpy as np

# --- Strategy 1: Use smaller dtypes ---
# BAD: 8 GB for 1 billion float64
# arr = np.zeros(1_000_000_000, dtype=np.float64)

# GOOD: 4 GB for 1 billion float32
arr = np.zeros(1_000_000_000, dtype=np.float32)

# --- Strategy 2: Memory-mapped files ---
# Create a memory-mapped file (data lives on disk)
mmap_arr = np.memmap('large_data.dat', dtype=np.float32,
                     mode='w+', shape=(100_000, 1_000))
mmap_arr[0:100] = np.random.rand(100, 1_000)  # Write chunk
chunk = mmap_arr[0:100]  # Read chunk (only this part in RAM)
del mmap_arr  # Flush to disk

# --- Strategy 3: Chunked processing ---
def process_in_chunks(filepath, chunk_size=10_000):
    data = np.memmap(filepath, dtype=np.float32, mode='r',
                     shape=(1_000_000, 100))
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        results.append(chunk.mean(axis=0))
    return np.mean(results, axis=0)

# --- Strategy 4: In-place operations ---
a = np.random.rand(10_000_000)
np.multiply(a, 2, out=a)   # In-place, no extra memory
np.add(a, 1, out=a)        # In-place

# --- Strategy 5: Delete intermediate arrays ---
temp = np.random.rand(1_000_000)
result = np.sum(temp)
del temp  # Free memory immediately
```

### Interview Tip
The most practical answers are **dtype reduction** (immediate 50% savings), **memory-mapped files** (for datasets larger than RAM), and **chunked processing**. Mention that `np.memmap()` is NumPy's built-in solution for out-of-core computing, and that libraries like Dask extend this pattern for distributed arrays.

---

## Question 20
- [ ] Done

**What are NumPy “polynomial” objects and how are they used?**

### Definition
NumPy provides classes and functions for creating, evaluating, and manipulating **polynomials**. The modern API is in `numpy.polynomial`, while the legacy `np.poly1d` class is still widely used.

### Two APIs

| API | Module | Status |
|-----|--------|--------|
| Legacy | `np.poly1d`, `np.polyfit`, `np.polyval` | Still supported, simpler |
| Modern | `numpy.polynomial.polynomial` | Recommended, more stable numerically |

### Common Operations

| Operation | Legacy | Modern |
|-----------|--------|--------|
| Create | `np.poly1d([1, -3, 2])` | `Polynomial([2, -3, 1])` |
| Evaluate | `p(5)` | `p(5)` |
| Fit data | `np.polyfit(x, y, deg)` | `Polynomial.fit(x, y, deg)` |
| Roots | `p.roots` | `p.roots()` |
| Derivative | `p.deriv()` | `p.deriv()` |
| Integral | `p.integ()` | `p.integ()` |

### Code Example
```python
import numpy as np
from numpy.polynomial import Polynomial

# --- Legacy API: np.poly1d ---
# Coefficients: 2x^2 - 3x + 1
p = np.poly1d([2, -3, 1])
print(p)         # 2 x^2 - 3 x + 1
print(p(5))      # 36
print(p.roots)   # [1.0, 0.5]
print(p.deriv())  # 4 x - 3

# Polynomial fitting
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 3, 7, 13, 21])
coeffs = np.polyfit(x, y, deg=2)  # Fit quadratic
fit_fn = np.poly1d(coeffs)
print(f"Fitted: {fit_fn}")  # ~1x^2 + 1x + 1

# --- Modern API: numpy.polynomial ---
p2 = Polynomial([1, -3, 2])  # Note: coeffs in ascending order
print(p2(5))       # 36
print(p2.roots())  # [0.5, 1.0]

# Fit with modern API
fit = Polynomial.fit(x, y, deg=2)
print(fit.convert().coef)  # Coefficients in standard form
```

### Interview Tip
Polynomial fitting (`polyfit`) is a form of **regression** — fitting a degree-N polynomial minimizes least-squares error. Mention that the modern `numpy.polynomial` API uses **ascending coefficient order** (opposite of `poly1d`), and is numerically more stable for high-degree fits due to internal scaling.

---

## Question 21
- [ ] Done

**How does the internal C-API contribute to NumPy’s performance ?**

### Overview
NumPy's performance comes from its **C-language core**. The Python interface is a thin wrapper around highly optimized C (and some Fortran) code. The C-API allows:
1. Array operations to bypass the Python interpreter
2. Direct memory manipulation without Python object overhead
3. Integration with optimized BLAS/LAPACK libraries

### Architecture

```
┌─────────────────────────────────────┐
│   Python Code: np.dot(A, B)       │
├─────────────────────────────────────┤
│  NumPy C-API (type checking,      │
│  dispatch, memory management)     │
├─────────────────────────────────────┤
│  BLAS/LAPACK (linear algebra)      │
│  or NumPy’s own C loops            │
├─────────────────────────────────────┤
│  CPU: SIMD, Cache, Multi-thread    │
└─────────────────────────────────────┘
```

### Key Performance Mechanisms

| Mechanism | Description |
|-----------|-------------|
| **No interpreter loop** | C loops replace Python `for` loops |
| **Contiguous memory** | Arrays stored as flat C buffers (cache-friendly) |
| **Type specialization** | Compiled code for each dtype (no runtime type checks) |
| **BLAS/LAPACK** | Industry-standard optimized math (MKL, OpenBLAS) |
| **SIMD instructions** | SSE/AVX process 4–8 floats per CPU cycle |
| **Buffer protocol** | Zero-copy data sharing between C extensions |
| **Ufunc machinery** | Generic iterator for element-wise ops across any shape |

### Code Example
```python
import numpy as np

# All of these execute C code internally:
a = np.random.rand(1000, 1000)
b = np.random.rand(1000, 1000)

# Matrix multiply → dispatched to BLAS (e.g., OpenBLAS, MKL)
c = a @ b

# Element-wise ops → NumPy C loops with SIMD
d = np.sqrt(a) + np.exp(b)

# Check which BLAS is linked
np.show_config()

# Writing custom C extensions via the C-API
# - Create custom ufuncs
# - Expose C/C++ code to NumPy arrays
# - Libraries like Cython, pybind11, cffi use the C-API
```

### Interview Tip
The C-API is what makes NumPy the **foundation of Python's scientific stack**. Key points: (1) Python overhead is paid once per operation, not per element; (2) BLAS libraries like Intel MKL can parallelize linear algebra automatically; (3) the C-API enables other libraries (SciPy, Pandas, Scikit-learn) to operate directly on NumPy memory without copying.

---

## Question 22
- [ ] Done

**Explain the concept of a stride trick in NumPy.**

### Definition
Stride tricks use `np.lib.stride_tricks.as_strided()` to create **custom views** of array memory by manually specifying shape and strides. This enables creating sliding windows, tiling, and other advanced views **without copying data**, achieving O(1) memory overhead.

### How It Works
`as_strided(x, shape, strides)` reinterprets the same memory buffer with a new shape and stride pattern. The key insight is that overlapping regions of memory can be viewed as separate elements.

### Common Use Cases

| Use Case | Description |
|----------|-------------|
| Sliding windows | Extract overlapping sub-arrays for convolution |
| Rolling statistics | Compute rolling mean/std without loops |
| Toeplitz matrices | Create structured matrices for signal processing |
| Patch extraction | Extract image patches for CNNs |
| Block views | Split array into non-overlapping blocks |

### Code Example
```python
import numpy as np
from numpy.lib.stride_tricks import as_strided, sliding_window_view

arr = np.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# --- Sliding window using as_strided ---
window_size = 3
new_shape = (len(arr) - window_size + 1, window_size)  # (8, 3)
new_strides = (arr.strides[0], arr.strides[0])         # (8, 8) for int64

windows = as_strided(arr, shape=new_shape, strides=new_strides)
print(windows)
# [[0 1 2]
#  [1 2 3]
#  [2 3 4]
#  ...]

# --- Modern alternative (NumPy 1.20+): sliding_window_view ---
windows2 = sliding_window_view(arr, window_shape=3)
print(windows2)  # Same result, safer API

# --- Rolling mean ---
rolling_mean = sliding_window_view(arr.astype(float), 3).mean(axis=1)
print(rolling_mean)  # [1., 2., 3., 4., 5., 6., 7., 8.]

# --- 2D sliding windows (image patches) ---
image = np.arange(16).reshape(4, 4)
patches = sliding_window_view(image, (2, 2))
print(patches.shape)  # (3, 3, 2, 2) — 3x3 grid of 2x2 patches
```

### Interview Tip
`as_strided` is powerful but **dangerous** — incorrect strides can read arbitrary memory. Always prefer `sliding_window_view()` (NumPy 1.20+) for safety. The key interview point is that stride tricks create views with **zero memory overhead**, which is crucial for operations like convolution where naive copying would be prohibitively expensive.

---

## Question 23
- [ ] Done

**What is the role of the NumPy nditer object?**

### Definition
`np.nditer` is a **multi-dimensional iterator** that provides efficient element-wise iteration over arrays. It handles broadcasting, multiple arrays, and different memory layouts automatically, and serves as the Python-level equivalent of NumPy's internal C iteration machinery.

### Why Use nditer Instead of Regular Loops

| Feature | Regular Loop | nditer |
|---------|-------------|--------|
| Multi-array iteration | Manual shape matching | Automatic broadcasting |
| Memory order | Always C-order | Respects actual layout |
| Output allocation | Manual | Built-in `op_flags` |
| C-API bridge | No | Direct mapping to C iterators |
| Read/Write control | No checking | Explicit `op_flags` |

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `flags` | `['multi_index']`, `['external_loop']`, `['buffered']` |
| `op_flags` | `['readonly']`, `['readwrite']`, `['writeonly']` |
| `order` | `'C'` (row-major) or `'F'` (column-major) |
| `casting` | Type casting rules: `'no'`, `'same_kind'`, `'unsafe'` |

### Code Example
```python
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# --- Basic iteration ---
for x in np.nditer(arr):
    print(x, end=' ')  # 1 2 3 4 5 6

# --- Multi-array iteration with broadcasting ---
a = np.arange(3)       # [0, 1, 2]
b = np.arange(6).reshape(2, 3)  # [[0,1,2],[3,4,5]]

for x, y in np.nditer([a, b]):
    print(f"{x}+{y}={x+y}", end='  ')
# 0+0=0  1+1=2  2+2=4  0+3=3  1+4=5  2+5=7

# --- Modifying array elements ---
arr2 = np.array([1, 2, 3])
with np.nditer(arr2, op_flags=['readwrite']) as it:
    for x in it:
        x[...] = x * 2  # In-place modification
print(arr2)  # [2, 4, 6]

# --- Tracking multi-index ---
it = np.nditer(arr, flags=['multi_index'])
while not it.finished:
    print(f"Index {it.multi_index}: {it[0]}")
    it.iternext()
# Index (0, 0): 1
# Index (0, 1): 2  ...etc

# --- External loop for efficiency ---
for x in np.nditer(arr, flags=['external_loop'], order='C'):
    print(x)  # [1 2 3 4 5 6] (as one chunk)
```

### Interview Tip
`nditer` is rarely used in everyday NumPy code (vectorized operations are preferred), but it is essential for **writing custom ufuncs** and understanding how NumPy's internal iteration works. The key point is that `x[...]` (ellipsis assignment) must be used for in-place modification, not `x = value`.

---

## Question 24
- [ ] Done

**Explain how NumPy integrates with other Python libraries like Pandas and Matplotlib.**

### Overview
NumPy is the **foundation of Python's data science stack**. Its ndarray is the universal data exchange format, and its C-level memory layout allows other libraries to operate on NumPy data **without copying**.

### Integration Map

| Library | How It Uses NumPy |
|---------|-------------------|
| **Pandas** | DataFrame/Series backed by NumPy arrays; `.values` / `.to_numpy()` returns ndarray |
| **Matplotlib** | Accepts NumPy arrays directly for plotting; internal computations use NumPy |
| **Scikit-learn** | All inputs/outputs are NumPy arrays; fit/predict expect ndarrays |
| **SciPy** | Extends NumPy with advanced math (optimization, interpolation, sparse matrices) |
| **TensorFlow** | `tf.constant(np_array)` creates tensors; `.numpy()` converts back |
| **PyTorch** | `torch.from_numpy(arr)` shares memory; `.numpy()` converts back |
| **OpenCV** | Images are NumPy arrays (`uint8` with shape H×W×C) |
| **PIL/Pillow** | `np.array(image)` converts; `Image.fromarray(arr)` converts back |

### Code Example
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- NumPy ↔ Pandas ---
arr = np.array([[1, 2, 3], [4, 5, 6]])
df = pd.DataFrame(arr, columns=['a', 'b', 'c'])

# DataFrame to NumPy
np_data = df.to_numpy()           # Recommended
np_data2 = df.values              # Legacy
np_series = df['a'].to_numpy()    # Single column

# NumPy operations on Pandas data
print(np.mean(df['a']))  # Works directly

# --- NumPy ↔ Matplotlib ---
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.plot(x, y)  # Accepts NumPy arrays directly
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.show()

# Histogram from NumPy data
data = np.random.randn(10000)
plt.hist(data, bins=50, density=True)
plt.show()

# --- NumPy ↔ Scikit-learn ---
from sklearn.linear_model import LinearRegression

X = np.random.rand(100, 3)  # 100 samples, 3 features
y = np.random.rand(100)
model = LinearRegression().fit(X, y)
predictions = model.predict(X)  # Returns NumPy array
```

### Interview Tip
The key concept is **zero-copy interoperability** — NumPy's array protocol and buffer protocol allow libraries to share memory without duplication. Mention that this is why converting between NumPy, Pandas, and PyTorch is nearly free when data types are compatible.

---

## Question 25
- [ ] Done

**Describe how NumPy can be used with JAX for accelerated machine learning computation.**

### Overview
JAX is Google's library that provides a **NumPy-compatible API** with automatic differentiation, GPU/TPU acceleration, and JIT compilation. JAX's `jax.numpy` is designed as a near drop-in replacement for `numpy`, making it easy to convert existing NumPy code to run on accelerators.

### NumPy vs JAX Comparison

| Feature | NumPy | JAX (`jax.numpy`) |
|---------|-------|--------------------|
| API | `np.sum()`, `np.dot()` | `jnp.sum()`, `jnp.dot()` — same API |
| Hardware | CPU only | CPU, GPU, TPU |
| Auto-differentiation | No | `jax.grad()` — automatic gradients |
| JIT compilation | No | `jax.jit()` — XLA compiler |
| Vectorization | Manual | `jax.vmap()` — auto-vectorize |
| Parallelism | No | `jax.pmap()` — multi-device |
| Mutability | Mutable arrays | **Immutable** arrays (functional style) |
| Random numbers | `np.random.rand()` | Explicit PRNG keys (`jax.random.PRNGKey`) |

### Code Example
```python
import numpy as np
import jax
import jax.numpy as jnp

# --- Drop-in replacement ---
# NumPy version
def numpy_fn(x):
    return np.sum(np.sin(x) ** 2)

# JAX version (nearly identical)
def jax_fn(x):
    return jnp.sum(jnp.sin(x) ** 2)

# --- Automatic differentiation ---
grad_fn = jax.grad(jax_fn)  # Gradient function
x = jnp.array([1.0, 2.0, 3.0])
print(grad_fn(x))  # Gradients w.r.t. x

# --- JIT compilation for speed ---
@jax.jit
def fast_fn(x):
    return jnp.dot(x, x.T) + jnp.exp(x)

result = fast_fn(jnp.ones((1000, 1000)))  # Compiled & runs on GPU

# --- Converting between NumPy and JAX ---
np_array = np.random.rand(1000)
jax_array = jnp.array(np_array)       # NumPy → JAX (copies to device)
np_back = np.asarray(jax_array)        # JAX → NumPy

# --- Vectorized mapping (auto-batching) ---
def single_example_loss(x, y):
    return jnp.sum((x - y) ** 2)

# Automatically vectorize over a batch dimension
batch_loss = jax.vmap(single_example_loss)
X = jnp.ones((32, 10))  # Batch of 32
Y = jnp.zeros((32, 10))
losses = batch_loss(X, Y)  # Shape (32,) — one loss per example

# --- Practical ML: gradient descent ---
def mse_loss(params, X, y):
    predictions = X @ params
    return jnp.mean((predictions - y) ** 2)

params = jnp.zeros(10)
X = jnp.array(np.random.rand(100, 10))
y = jnp.array(np.random.rand(100))

for _ in range(100):
    grads = jax.grad(mse_loss)(params, X, y)
    params = params - 0.1 * grads  # SGD step
```

### Interview Tip
JAX's key selling point is that it is **NumPy with superpowers**: same API but with `jit` (speed), `grad` (derivatives), `vmap` (batching), and `pmap` (parallelism). Mention that JAX arrays are **immutable** (functional programming paradigm), which enables compiler optimizations but requires a different mindset from NumPy. JAX is heavily used at Google for ML research and powers libraries like Flax and Optax.

---

## Question 26
- [ ] Done

**Why is NumPy more efficient for numerical computations than pure Python ?**

### NumPy vs. Pure Python Performance

### Core Reasons

| Factor | Pure Python | NumPy |
|--------|------------|-------|
| **Storage** | Objects on heap (28 bytes per int) | Contiguous C array (8 bytes per float64) |
| **Loops** | Python interpreter loop (slow) | Pre-compiled C/Fortran loops |
| **Type checking** | Every operation checks types | Homogeneous array, one check |
| **Memory layout** | Scattered pointers | Contiguous block (cache-friendly) |
| **Vectorization** | Not available | SIMD instructions (SSE, AVX) |
| **BLAS/LAPACK** | Not used | Optimized linear algebra libraries |

### Performance Demonstration

```python
import numpy as np
import time

n = 10_000_000

# Pure Python
python_list = list(range(n))
start = time.time()
result = sum(x * x for x in python_list)
print(f"Python: {time.time()-start:.3f}s")

# NumPy
np_array = np.arange(n)
start = time.time()
result = np.sum(np_array ** 2)
print(f"NumPy:  {time.time()-start:.3f}s")

# NumPy is typically 50-100x faster for this operation
```

### Why is NumPy Faster?

```
Python list:  [ptr] → PyObject(type=int, val=1)
              [ptr] → PyObject(type=int, val=2)   ← scattered in memory
              [ptr] → PyObject(type=int, val=3)

NumPy array:  [1][2][3][4][5][6][7][8]  ← contiguous block. CPU cache loves this!
              ^^^^^^^^^^^^^^^^^^^^^^^^
              raw bytes, no Python overhead
```

1. **No per-element overhead**: NumPy stores raw numbers; Python wraps each in a PyObject (28+ bytes)
2. **Vectorized C loops**: Operations run in compiled C code, not interpreted Python
3. **CPU cache efficiency**: Contiguous memory means fewer cache misses
4. **SIMD parallelism**: Modern CPUs process 4-8 numbers simultaneously via AVX instructions
5. **Optimized libraries**: BLAS/LAPACK for linear algebra (tuned for specific CPU)

### Memory Comparison
```python
import sys
# Python list of 1000 ints: ~28,000 bytes (28 per object + pointers)
# NumPy array of 1000 int64: ~8,000 bytes (8 per element, contiguous)

python_size = sys.getsizeof(list(range(1000))) + 1000 * sys.getsizeof(1)
numpy_size = np.arange(1000).nbytes
print(f"Python: {python_size} bytes, NumPy: {numpy_size} bytes")  # ~3.5x difference
```

> **Interview Tip:** The three key words are: **contiguous memory**, **vectorized C operations**, and **no Python object overhead**. NumPy essentially moves the loop from Python to C, processes data in bulk, and leverages CPU hardware (SIMD, cache). This is why "avoid Python loops" is the #1 NumPy performance rule.

---

## Question 27
- [ ] Done

**Discuss the performance benefits of using NumPy’s in-place operations**

### In-Place Operations in NumPy

In-place operations modify an array **without creating a copy**, saving memory and improving speed.

### In-Place vs. Out-of-Place

```python
import numpy as np

a = np.ones(10_000_000)

# Out-of-place: creates a NEW array (2x memory)
b = a + 1       # allocates new array, copies result

# In-place: modifies existing array (no extra memory)
a += 1          # modifies a directly
a *= 2          # modifies a directly
np.add(a, 1, out=a)  # explicit in-place with 'out' parameter
```

### Benefits

| Benefit | Explanation |
|---------|------------|
| **Memory saving** | No temporary array allocated |
| **Speed** | No memory allocation + no copy = faster |
| **Cache friendly** | Data stays in the same location |
| **Reduced GC pressure** | Fewer objects for garbage collector |

### In-Place Methods

```python
# Augmented assignment operators
a += b        # np.add(a, b, out=a)
a -= b        # np.subtract(a, b, out=a)
a *= b        # np.multiply(a, b, out=a)

# Using 'out' parameter (works with any ufunc)
np.sqrt(a, out=a)
np.exp(a, out=a)
np.clip(a, 0, 1, out=a)

# In-place array modification
a.fill(0)              # fill with value
a[a < 0] = 0           # conditional in-place
```

### Cautions

```python
# Caution 1: dtype casting may fail in-place
a = np.array([1, 2, 3], dtype=np.int32)
# a += 0.5  # ERROR: can't cast float to int in-place

# Caution 2: views share data
b = a[::2]  # b is a VIEW of a
b += 10     # this also modifies a!
```

> **Interview Tip:** Use in-place operations in **tight loops** and **memory-constrained** scenarios (GPU training, large datasets). The `out=` parameter is especially useful in neural network forward passes where you can reuse pre-allocated buffers.

---

## Question 28
- [ ] Done

**Discuss the use of NumPy for operations on polynomials**

### Polynomial Operations in NumPy

NumPy provides two APIs for polynomial operations: the **legacy** `np.poly1d`/`np.polyfit` and the **modern** `np.polynomial` module.

### Modern API (Recommended)

```python
import numpy as np
from numpy.polynomial import polynomial as P

# Define polynomial: 1 + 2x + 3x² (coefficients in ascending order)
coeffs = [1, 2, 3]  # lowest degree first

# Evaluate at specific points
x = np.array([0, 1, 2, 3])
y = P.polyval(x, coeffs)
print(y)  # [1, 6, 17, 34]  (1 + 2*x + 3*x²)

# Polynomial arithmetic
p1 = [1, 2]      # 1 + 2x
p2 = [3, 0, 1]   # 3 + x²

sum_poly = P.polyadd(p1, p2)   # 4 + 2x + x²
prod_poly = P.polymul(p1, p2)  # 3 + 6x + x² + 2x³
deriv = P.polyder(coeffs)      # [2, 6]  (derivative: 2 + 6x)
integral = P.polyint(coeffs)   # [0, 1, 1, 1]  (integral)

# Find roots
roots = P.polyroots([6, -5, 1])  # x² - 5x + 6 = 0 → [2, 3]
```

### Polynomial Fitting (Regression)

```python
# Generate noisy data
x = np.linspace(0, 10, 100)
y_true = 2 + 3 * x - 0.5 * x**2
y_noisy = y_true + np.random.randn(100) * 2

# Fit polynomial (modern API)
coeffs_fit = P.polyfit(x, y_noisy, deg=2)
print(f"Fitted: {coeffs_fit}")  # close to [2, 3, -0.5]

# Evaluate fitted polynomial
y_pred = P.polyval(x, coeffs_fit)

# Legacy API (still common)
coeffs_legacy = np.polyfit(x, y_noisy, deg=2)  # highest degree first!
p = np.poly1d(coeffs_legacy)
print(p)  # prints polynomial equation
print(p(5))  # evaluate at x=5
```

### ML Applications

```python
# 1. Polynomial feature expansion for regression
def polynomial_features(X, degree=3):
    """Create polynomial features for regression."""
    features = [np.ones(len(X))]  # bias term
    for d in range(1, degree + 1):
        features.append(X ** d)
    return np.column_stack(features)

X = np.array([1, 2, 3, 4, 5])
X_poly = polynomial_features(X, degree=3)
# [[1, 1, 1, 1], [1, 2, 4, 8], [1, 3, 9, 27], ...]

# 2. Learning rate schedules (polynomial decay)
def poly_decay(initial_lr, step, total_steps, power=2):
    return initial_lr * (1 - step / total_steps) ** power
```

### Legacy vs. Modern API
| Feature | Legacy (`np.poly1d`) | Modern (`np.polynomial`) |
|---------|---------------------|-------------------------|
| Coefficient order | Highest degree first | Lowest degree first |
| Numerical stability | Poor for high degrees | Better (uses different basis) |
| API consistency | Ad hoc | Unified across types |
| Types | Only power series | Chebyshev, Legendre, Hermite, etc. |

> **Interview Tip:** Use the **modern `np.polynomial`** API for numerical stability. For ML, polynomial features are a simple way to capture non-linear relationships in linear models (polynomial regression). Be aware of **overfitting** with high-degree polynomials — use regularization.

---
