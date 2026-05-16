# NumPy Interview Questions - General Questions

## Question 1
- [ ] Done

**How do you inspect the shape and size of a NumPy array?**

### Key Attributes

| Attribute | Returns | Example for (3, 4) array |
|-----------|---------|--------------------------|
| `arr.shape` | Tuple of dimensions | `(3, 4)` |
| `arr.size` | Total elements | `12` |
| `arr.ndim` | Number of dimensions | `2` |

### Code Example
```python
import numpy as np

arr = np.arange(24).reshape(2, 3, 4)  # 3D array

print(f"Shape: {arr.shape}")     # (2, 3, 4)
print(f"Size: {arr.size}")       # 24
print(f"Dimensions: {arr.ndim}") # 3
```

---

## Question 2
- [ ] Done

**How do you perform element-wise operations in NumPy?**

### Definition
Element-wise operations apply to each element independently. Use arithmetic operators or ufuncs.

### Code Example
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

# Using operators (element-wise)
print(a + b)   # [11, 22, 33]
print(a * b)   # [10, 40, 90]
print(a ** 2)  # [1, 4, 9]

# Using ufuncs
print(np.add(a, b))      # [11, 22, 33]
print(np.sqrt(a))        # [1.0, 1.41, 1.73]
print(np.exp(a))         # [2.72, 7.39, 20.09]

# With scalar (broadcasting)
print(a * 10)  # [10, 20, 30]
```

---

## Question 3
- [ ] Done

**How do you compute the mean , median , and standard deviation with NumPy ?**

### Key Functions

| Function | Description |
|----------|-------------|
| `np.mean(arr)` | Arithmetic average |
| `np.median(arr)` | Middle value (robust to outliers) |
| `np.std(arr)` | Standard deviation |

### Code Example
```python
import numpy as np

data = np.array([[1, 2, 3],
                 [4, 5, 6]])

# Overall statistics
print(f"Mean: {np.mean(data)}")      # 3.5
print(f"Median: {np.median(data)}")  # 3.5
print(f"Std: {np.std(data)}")        # 1.71

# Axis-wise (important for ML)
print(f"Mean per column: {data.mean(axis=0)}")  # [2.5, 3.5, 4.5]
print(f"Mean per row: {data.mean(axis=1)}")     # [2.0, 5.0]
```

### ML Use Case
Feature normalization: `X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)`

---

## Question 4
- [ ] Done

**Why is NumPy more efficient than pure Python for numerical computations?**

### Four Key Reasons

| Reason | Python List | NumPy Array |
|--------|-------------|-------------|
| **Implementation** | Python interpreter | Compiled C/Fortran |
| **Operations** | Explicit for loops | Vectorized (single call to C) |
| **Memory** | Pointers to objects | Contiguous block |
| **Type Checking** | Every operation | Once per array |

### Performance Example
```python
import numpy as np
import time

size = 1_000_000
py_list = list(range(size))
np_arr = np.arange(size)

# Python: ~200ms
start = time.time()
[x * 2 for x in py_list]
print(f"Python: {time.time() - start:.4f}s")

# NumPy: ~2ms (100x faster!)
start = time.time()
np_arr * 2
print(f"NumPy: {time.time() - start:.4f}s")
```

---

## Question 5
- [ ] Done

**How do you check the memory size of a NumPy array?**

### Key Attributes

| Attribute | Description |
|-----------|-------------|
| `arr.nbytes` | Total bytes for data |
| `arr.itemsize` | Bytes per element |
| `arr.size` | Number of elements |

### Code Example
```python
import numpy as np

arr = np.zeros((1000, 1000), dtype=np.float64)

print(f"Total bytes: {arr.nbytes}")           # 8,000,000
print(f"Bytes per element: {arr.itemsize}")   # 8
print(f"Elements: {arr.size}")                # 1,000,000

# Verify: nbytes = size × itemsize
assert arr.nbytes == arr.size * arr.itemsize

# Convert to MB
print(f"Memory: {arr.nbytes / (1024**2):.2f} MB")  # 7.63 MB
```

### Memory Optimization Tip
```python
# float64 (default): 8 bytes
arr64 = np.zeros(1000000, dtype=np.float64)
print(f"float64: {arr64.nbytes / 1e6:.1f} MB")  # 8.0 MB

# float32: 4 bytes (half the memory!)
arr32 = np.zeros(1000000, dtype=np.float32)
print(f"float32: {arr32.nbytes / 1e6:.1f} MB")  # 4.0 MB
```

---

## Question 6
- [ ] Done

**How do you handle NaN or infinite values in a NumPy array ?**

### Detection Functions

| Function | Detects |
|----------|---------|
| `np.isnan(arr)` | NaN values |
| `np.isinf(arr)` | Infinite values |
| `np.isfinite(arr)` | Finite (not NaN or inf) |

### NaN-Safe Functions

| Standard | NaN-Safe | Description |
|----------|----------|-------------|
| `np.mean()` | `np.nanmean()` | Mean ignoring NaN |
| `np.sum()` | `np.nansum()` | Sum treating NaN as 0 |
| `np.std()` | `np.nanstd()` | Std ignoring NaN |
| `np.min()` | `np.nanmin()` | Min ignoring NaN |

### Code Example
```python
import numpy as np

arr = np.array([1.0, 2.0, np.nan, 4.0, np.inf])

# Detection
print(np.isnan(arr))     # [False, False, True, False, False]
print(np.isnan(arr).sum())  # Count: 1

# NaN-safe computation
print(np.mean(arr))      # nan (propagates)
print(np.nanmean(arr))   # 2.33 (ignores NaN)

# Replacement
arr_clean = arr.copy()
arr_clean[np.isnan(arr_clean)] = 0  # Replace NaN with 0
# OR
arr_clean = np.nan_to_num(arr)      # Replaces NaN→0, inf→large number
```

### Important
```python
# DON'T use == to check for NaN
np.nan == np.nan  # Returns False!

# DO use np.isnan()
np.isnan(np.nan)  # Returns True
```


---

## Question 7
- [ ] Done

**How do you create a record array in NumPy?**

**Answer:**

A **record array** (recarray) is a structured array that allows field access using attribute syntax (`arr.field`) instead of dictionary syntax (`arr['field']`).

```python
import numpy as np

# Method 1: np.rec.array from tuples
data = np.rec.array([
    ('Alice', 25, 85.5),
    ('Bob', 30, 92.0),
    ('Charlie', 28, 78.3)
], dtype=[('name', 'U10'), ('age', 'i4'), ('score', 'f8')])

print(data.name)    # ['Alice' 'Bob' 'Charlie']  — attribute access
print(data.age)     # [25 30 28]
print(data[0])      # ('Alice', 25, 85.5)

# Method 2: np.rec.fromarrays
names = np.array(['Alice', 'Bob', 'Charlie'])
ages = np.array([25, 30, 28])
scores = np.array([85.5, 92.0, 78.3])
data = np.rec.fromarrays([names, ages, scores], names='name,age,score')

# Method 3: Structured array + view
dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('score', 'f8')])
arr = np.array([('Alice', 25, 85.5)], dtype=dt)
rec = arr.view(np.recarray)
print(rec.name)  # attribute access works
```

| Feature | Structured Array | Record Array |
|---------|------------------|--------------|
| Access | `arr['field']` | `arr.field` (attribute) |
| Performance | Slightly faster | Slight overhead |
| Creation | `np.array(..., dtype=dt)` | `np.rec.array(...)` |

> **Interview Tip:** Record arrays are convenient for tabular data with named columns. For large-scale tabular work, prefer **Pandas DataFrames** instead.

---

## Question 8
- [ ] Done

**How can NumPy be used for audio signal processing?**

**Answer:**

NumPy provides the mathematical foundation for audio signal processing through its array operations and FFT module.

### Key Operations

```python
import numpy as np
from scipy.io import wavfile

# 1. Load audio data
sr, audio = wavfile.read('audio.wav')  # sr = sample rate
audio = audio.astype(np.float32) / 32768.0  # Normalize 16-bit to [-1, 1]

# 2. Fourier Transform — frequency analysis
fft_result = np.fft.fft(audio)
freqs = np.fft.fftfreq(len(audio), d=1/sr)
magnitude = np.abs(fft_result)
phase = np.angle(fft_result)

# 3. Filter frequencies (e.g., low-pass filter at 1000 Hz)
fft_filtered = fft_result.copy()
fft_filtered[np.abs(freqs) > 1000] = 0
filtered_audio = np.real(np.fft.ifft(fft_filtered))

# 4. Generate audio signals
t = np.linspace(0, 1, sr)  # 1 second
sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz (A note)

# 5. Spectrogram (short-time FFT)
window_size = 1024
hop = 512
frames = np.lib.stride_tricks.sliding_window_view(audio, window_size)[::hop]
spectrogram = np.abs(np.fft.rfft(frames * np.hanning(window_size), axis=1))

# 6. Energy / RMS
rms = np.sqrt(np.mean(audio**2))
energy_frames = np.array([np.sum(f**2) for f in frames])
```

| Task | NumPy Function |
|------|---------------|
| FFT | `np.fft.fft()`, `np.fft.rfft()` |
| Frequency axis | `np.fft.fftfreq()` |
| Windowing | `np.hanning()`, `np.hamming()` |
| Convolution | `np.convolve()` |
| Cross-correlation | `np.correlate()` |

> **Interview Tip:** NumPy handles raw signal math. For production audio ML, use **librosa** (built on NumPy) for mel spectrograms, MFCCs, and other audio features.

---

## Question 9
- [ ] Done

**What methods are there in NumPy to deal with missing data?**

**Answer:**

NumPy uses `np.nan` (Not a Number, float type) to represent missing data.

### Detection

```python
import numpy as np

arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])

# Detect NaN
np.isnan(arr)                    # [False, True, False, True, False]
np.any(np.isnan(arr))            # True
np.sum(np.isnan(arr))            # 2 (count of NaN)
np.where(np.isnan(arr))          # (array([1, 3]),)  — indices
```

### NaN-safe Aggregations

```python
# Standard functions propagate NaN
np.mean(arr)      # nan
np.sum(arr)       # nan

# NaN-ignoring versions
np.nanmean(arr)   # 3.0
np.nansum(arr)    # 9.0
np.nanstd(arr)    # 1.633
np.nanmin(arr)    # 1.0
np.nanmax(arr)    # 5.0
np.nanmedian(arr) # 3.0
np.nanpercentile(arr, 50)  # 3.0
```

### Removal & Replacement

```python
# Remove NaN values
clean = arr[~np.isnan(arr)]  # [1.0, 3.0, 5.0]

# Replace with a value (imputation)
arr_filled = np.where(np.isnan(arr), 0, arr)                    # Replace with 0
arr_mean = np.where(np.isnan(arr), np.nanmean(arr), arr)        # Replace with mean
np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)         # Replace NaN & inf

# Forward fill (carry last valid value)
def forward_fill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(len(arr)), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]
```

### Masked Arrays

```python
masked = np.ma.masked_invalid(arr)  # masks NaN and inf
np.ma.mean(masked)                  # 3.0 — ignores masked values
```

> **Interview Tip:** NumPy's NaN handling is limited to float arrays (integers can't hold NaN). For robust missing data handling with mixed types, use **Pandas** (`pd.isna()`, `.fillna()`, `.dropna()`).

---

## Question 10
- [ ] Done

**How do you find unique values and their counts in a NumPy array?**

**Answer:**

```python
import numpy as np

arr = np.array([3, 1, 2, 3, 1, 3, 2, 4, 1])

# Basic unique values (sorted)
np.unique(arr)  # [1, 2, 3, 4]

# Unique values with counts
values, counts = np.unique(arr, return_counts=True)
# values: [1, 2, 3, 4],  counts: [3, 2, 3, 1]

# Display as dictionary
dict(zip(values, counts))  # {1: 3, 2: 2, 3: 3, 4: 1}

# Unique with indices
values, indices = np.unique(arr, return_index=True)         # first occurrence index
values, inverse = np.unique(arr, return_inverse=True)       # map back to original

# All at once
values, idx, inv, cnt = np.unique(arr, return_index=True,
                                   return_inverse=True,
                                   return_counts=True)

# Reconstruct original: arr == values[inv]  → True

# 2D array — unique rows
arr2d = np.array([[1,2],[3,4],[1,2],[5,6]])
np.unique(arr2d, axis=0)  # [[1,2],[3,4],[5,6]]

# Most frequent value (mode)
values, counts = np.unique(arr, return_counts=True)
mode = values[np.argmax(counts)]  # 1 or 3 (both have count 3)
```

| Parameter | Returns |
|-----------|--------|
| `return_counts` | Count of each unique value |
| `return_index` | Index of first occurrence |
| `return_inverse` | Indices to reconstruct original from unique |
| `axis=0` | Unique rows (2D) |

> **Interview Tip:** `np.unique` always returns **sorted** values. For unsorted order-preserving unique, use `pd.unique()` or `np.unique` with `return_index` and sort by index.

---

## Question 11
- [ ] Done

**How can you use NumPy arrays with Cython for performance optimization?**

**Answer:**

**Cython** compiles Python-like code to C, and with **typed memoryviews** it can access NumPy array data at C speed (bypassing Python overhead).

### Basic Example

```cython
# fast_ops.pyx
import numpy as np
cimport numpy as cnp
from cython cimport boundscheck, wraparound

@boundscheck(False)   # Disable bounds checking for speed
@wraparound(False)    # Disable negative indexing
def fast_sum(cnp.ndarray[cnp.float64_t, ndim=1] arr):
    cdef int i, n = arr.shape[0]
    cdef double total = 0.0
    for i in range(n):
        total += arr[i]
    return total
```

### Modern Typed Memoryview (Preferred)

```cython
# modern_ops.pyx
from cython cimport boundscheck, wraparound

@boundscheck(False)
@wraparound(False)
def pairwise_distance(double[:, :] X):
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    cdef double[:, :] D = np.zeros((n, n))
    cdef int i, j, k
    cdef double tmp, diff
    for i in range(n):
        for j in range(i+1, n):
            tmp = 0.0
            for k in range(d):
                diff = X[i, k] - X[j, k]
                tmp += diff * diff
            D[i, j] = D[j, i] = tmp ** 0.5
    return np.asarray(D)
```

### Setup & Compilation

```python
# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("fast_ops.pyx"),
    include_dirs=[np.get_include()]
)
# Build: python setup.py build_ext --inplace
```

### Performance Comparison

| Approach | Relative Speed |
|----------|---------------|
| Pure Python loop | 1× (baseline) |
| NumPy vectorized | ~50-100× |
| Cython (untyped) | ~10-30× |
| Cython (typed memoryview) | ~200-500× |
| Cython + parallel (prange) | ~500-2000× |

> **Interview Tip:** Cython gives C-level speed for element-wise loops that can't be vectorized with NumPy. For simpler cases, try **Numba** (`@jit`) first — zero compilation step needed.
