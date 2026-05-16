# Linear Algebra Interview Questions - Scenario_Based Questions

## Question 1
- [ ] Done

**Discuss the importance of linear algebra in optimization problems, such as gradient descent.**

### Answer

**Definition:**  
Optimization in ML finds parameters that minimize a loss function. Linear algebra provides the mathematical framework for computing gradients, understanding convergence, and analyzing optimization landscapes.

**Core Concepts:**
- **Gradient**: Vector of partial derivatives pointing to steepest ascent
- **Hessian**: Matrix of second derivatives (curvature information)
- **Convexity**: Hessian is positive semi-definite → unique minimum
- **Condition number**: Ratio of largest/smallest eigenvalues → affects convergence speed

**Mathematical Formulation:**

Gradient descent update:
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla L(\mathbf{w}_t)$$

Newton's method (uses Hessian):
$$\mathbf{w}_{t+1} = \mathbf{w}_t - H^{-1} \nabla L(\mathbf{w}_t)$$

**Linear Algebra in Optimization:**

| Concept | Role in Optimization |
|---------|---------------------|
| Gradient (∇L) | Direction of steepest descent |
| Hessian (H) | Curvature, convergence analysis |
| Eigenvalues of H | Learning rate bounds, condition number |
| Matrix inversion | Newton's method, natural gradient |
| SVD/QR | Numerical stability in least squares |

**Python Example:**
```python
import numpy as np

# Gradient descent for linear regression
def gradient_descent(X, y, lr=0.01, epochs=1000):
    n, d = X.shape
    w = np.zeros(d)
    
    for _ in range(epochs):
        # Gradient: ∇L = (1/n) * X^T (Xw - y)
        gradient = X.T @ (X @ w - y) / n
        w = w - lr * gradient
    
    return w

# Condition number affects convergence
X = np.random.randn(100, 5)
cond_number = np.linalg.cond(X.T @ X)
print(f"Condition number: {cond_number:.2f}")
# Higher → slower convergence, need smaller learning rate

# Hessian eigenvalues determine optimal learning rate
H = X.T @ X / len(X)  # Hessian for MSE
eigenvalues = np.linalg.eigvalsh(H)
max_lr = 2 / np.max(eigenvalues)  # Stability bound
print(f"Max stable learning rate: {max_lr:.4f}")
```

**Interview Tips:**
- Gradient is a vector, Hessian is a matrix
- Condition number = max eigenvalue / min eigenvalue
- Poor conditioning → use adaptive methods (Adam, RMSprop)
- Know that matrix operations dominate computational cost in deep learning


---

## Question 2
- [ ] Done

**How would you handle large-scale matrix operations efficiently in terms of memory and computation?**

### Answer

**Definition:**  
Large-scale matrix operations require strategies to reduce memory footprint and computation time. Key techniques include sparse representations, chunking, approximations, and leveraging specialized hardware.

**Core Strategies:**

| Strategy | When to Use | Memory Savings |
|----------|-------------|----------------|
| **Sparse matrices** | Most entries are zero | O(nnz) vs O(n²) |
| **Chunking/batching** | Matrix doesn't fit in RAM | Process in pieces |
| **Low-rank approximation** | Approximate is acceptable | O(nk) vs O(n²) |
| **Out-of-core** | Larger than memory | Stream from disk |
| **GPU acceleration** | Dense operations | Parallel computation |

**Python Example:**
```python
import numpy as np
from scipy import sparse

# 1. SPARSE MATRICES (most common)
# Dense: 10000x10000 = 800MB
# Sparse with 0.1% non-zero: ~0.8MB
n = 10000
density = 0.001
data = np.random.randn(int(n * n * density))
rows = np.random.randint(0, n, len(data))
cols = np.random.randint(0, n, len(data))

A_sparse = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
print(f"Sparse storage: {A_sparse.data.nbytes / 1e6:.2f} MB")

# Sparse matrix-vector multiplication
x = np.random.randn(n)
y = A_sparse @ x  # Efficient O(nnz) operation

# 2. CHUNKED PROCESSING
def chunked_matmul(A, B, chunk_size=1000):
    """Multiply large matrices in chunks."""
    n = A.shape[0]
    C = np.zeros((n, B.shape[1]))
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        C[i:end] = A[i:end] @ B
    return C

# 3. LOW-RANK APPROXIMATION (Truncated SVD)
from scipy.sparse.linalg import svds
U, s, Vt = svds(A_sparse, k=50)  # Keep top 50 components
# Approximate: A ≈ U @ diag(s) @ Vt
# Storage: O(n*k) instead of O(n²)

# 4. MEMORY-MAPPED FILES (out-of-core)
# For arrays larger than RAM
large_matrix = np.memmap('large_matrix.dat', dtype='float32', 
                         mode='w+', shape=(50000, 50000))
```

**Hardware Considerations:**
- **GPU**: Use cupy, PyTorch, or TensorFlow for dense operations
- **Distributed**: Dask, Spark for cluster computing
- **BLAS libraries**: NumPy uses optimized BLAS (OpenBLAS, MKL)

**Interview Tips:**
- Always ask: "Is the matrix sparse?" before choosing approach
- CSR format for row operations, CSC for column operations
- Low-rank ≈ truncated SVD is the workhorse for approximations
- Memory bandwidth often bottleneck, not compute



---

## Question 3
- [ ] Done

**Propose a method for dimensionality reduction using linear algebra techniques.**

### Answer

**Definition:**  
Dimensionality reduction projects high-dimensional data to lower dimensions while preserving important structure. The main linear algebra technique is **PCA** (Principal Component Analysis) using eigendecomposition or SVD.

**Core Methods:**

| Method | Technique | Preserves |
|--------|-----------|-----------|
| **PCA** | Eigendecomposition of covariance | Variance |
| **SVD** | Singular value decomposition | Same as PCA |
| **LDA** | Eigendecomposition of scatter matrices | Class separation |
| **Random Projection** | Random matrix multiplication | Pairwise distances |

**PCA Algorithm:**

1. Center data: $X_c = X - \mu$
2. Compute covariance: $C = \frac{1}{n} X_c^T X_c$
3. Eigendecompose: $C = V \Lambda V^T$
4. Select top k eigenvectors
5. Project: $X_{reduced} = X_c V_k$

**Mathematical Formulation:**
$$X_{reduced} = X_c V_k \quad \text{where } V_k = [\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_k]$$

Variance preserved by k components:
$$\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$$

**Python Example:**
```python
import numpy as np
from sklearn.decomposition import PCA

# Manual PCA using eigendecomposition
def pca_manual(X, n_components):
    # Center data
    X_centered = X - X.mean(axis=0)
    
    # Covariance matrix
    cov = np.cov(X_centered.T)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Project to lower dimension
    V_k = eigenvectors[:, :n_components]
    X_reduced = X_centered @ V_k
    
    # Variance explained
    var_explained = eigenvalues[:n_components].sum() / eigenvalues.sum()
    
    return X_reduced, var_explained

# Example usage
X = np.random.randn(100, 50)  # 100 samples, 50 features
X_reduced, var_exp = pca_manual(X, n_components=10)
print(f"Reduced shape: {X_reduced.shape}")  # (100, 10)
print(f"Variance explained: {var_exp:.2%}")

# Using sklearn (recommended)
pca = PCA(n_components=0.95)  # Keep 95% variance
X_sklearn = pca.fit_transform(X)
print(f"Components for 95% variance: {pca.n_components_}")
```

**Choosing Number of Components:**
- **Elbow method**: Plot cumulative variance, find elbow
- **Kaiser criterion**: Keep eigenvalues > 1
- **Variance threshold**: Keep components for 95% variance

**Interview Tips:**
- PCA finds directions of maximum variance
- SVD on centered data gives same result as PCA
- PCA assumes linear relationships; use kernel PCA for non-linear
- Always standardize features before PCA if scales differ


---

## Question 4
- [ ] Done

**How would you use matrices to model relational data in databases?**

### Answer

**Definition:**  
Relational data (entities and their relationships) can be represented as matrices: adjacency matrices for graphs, incidence matrices for bipartite relations, and tensor factorization for multi-relational data.

**Matrix Representations:**

| Data Type | Matrix Representation |
|-----------|----------------------|
| Graph | Adjacency matrix A[i,j] = edge weight |
| Bipartite (user-item) | Interaction matrix R[u,i] = rating |
| Multi-relational | Tensor A[i,j,k] = relation k between i,j |

**Applications:**

1. **Recommender Systems**: User-item matrix factorization
2. **Knowledge Graphs**: Entity-relation-entity tensors
3. **Social Networks**: Adjacency matrix analysis
4. **Link Prediction**: Matrix completion

**Python Example:**
```python
import numpy as np

# 1. ADJACENCY MATRIX (Graph)
# 4 nodes: A-B-C-D with edges A-B, B-C, C-D, A-D
adj_matrix = np.array([
    [0, 1, 0, 1],  # A connects to B, D
    [1, 0, 1, 0],  # B connects to A, C
    [0, 1, 0, 1],  # C connects to B, D
    [1, 0, 1, 0]   # D connects to A, C
])

# Find connected nodes (2-hop neighbors)
two_hop = adj_matrix @ adj_matrix
print("2-hop connections:\n", two_hop)

# 2. USER-ITEM MATRIX (Recommendations)
# Users × Items ratings matrix
ratings = np.array([
    [5, 3, 0, 1],   # User 0
    [4, 0, 0, 1],   # User 1  
    [1, 1, 0, 5],   # User 2
    [0, 0, 5, 4],   # User 3
])

# Matrix factorization: R ≈ U @ V.T
def matrix_factorization(R, k=2, steps=1000, lr=0.01):
    n_users, n_items = R.shape
    U = np.random.rand(n_users, k)
    V = np.random.rand(n_items, k)
    
    mask = R > 0  # Only observed entries
    
    for _ in range(steps):
        pred = U @ V.T
        error = (R - pred) * mask
        U += lr * (error @ V)
        V += lr * (error.T @ U)
    
    return U, V

U, V = matrix_factorization(ratings, k=2)
predicted = U @ V.T
print("Predicted ratings:\n", predicted.round(1))

# 3. LINK PREDICTION (will nodes connect?)
# Use adjacency matrix powers and similarity
def common_neighbors(adj, i, j):
    """Score = number of common neighbors."""
    return adj[i] @ adj[j]  # Dot product of rows

score = common_neighbors(adj_matrix, 0, 2)  # A and C
print(f"Common neighbors A-C: {score}")
```

**Key Operations:**
- **A²[i,j]**: Number of 2-hop paths from i to j
- **A @ A.T**: Co-occurrence matrix
- **SVD of R**: Latent factor model

**Interview Tips:**
- Adjacency matrix eigenvalues reveal graph structure
- Matrix factorization = collaborative filtering
- Sparse matrices essential for real-world scale
- Know difference between explicit (ratings) and implicit (clicks) feedback


---

## Question 5
- [ ] Done

**Discuss how to apply linear algebra to image processing tasks.**

### Answer

**Definition:**  
Images are matrices (grayscale) or 3D tensors (color). Linear algebra operations enable filtering, compression, feature extraction, and transformations through matrix operations.

**Image as Matrix:**
- Grayscale: H × W matrix, values 0-255
- Color: H × W × 3 tensor (RGB channels)
- Batch: N × H × W × C tensor

**Key Operations:**

| Task | Linear Algebra Operation |
|------|-------------------------|
| Filtering/blur | Convolution (matrix multiplication) |
| Compression | SVD/truncated SVD |
| Edge detection | Derivative kernels |
| Rotation/scaling | Transformation matrices |
| Feature extraction | PCA on image patches |

**Python Example:**
```python
import numpy as np
from scipy import ndimage

# Create sample grayscale image (matrix)
image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

# 1. CONVOLUTION (Filtering)
# Blur kernel (averaging)
blur_kernel = np.ones((3, 3)) / 9
blurred = ndimage.convolve(image.astype(float), blur_kernel)

# Edge detection (Sobel)
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = sobel_x.T
edges_x = ndimage.convolve(image.astype(float), sobel_x)
edges_y = ndimage.convolve(image.astype(float), sobel_y)
edges = np.sqrt(edges_x**2 + edges_y**2)

# 2. IMAGE COMPRESSION (SVD)
def compress_image(img, k):
    """Compress using top k singular values."""
    U, s, Vt = np.linalg.svd(img.astype(float), full_matrices=False)
    # Reconstruct with k components
    compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    return np.clip(compressed, 0, 255).astype(np.uint8)

compressed = compress_image(image, k=20)  # Keep 20 components
compression_ratio = (100 * 100) / (100 * 20 + 20 + 20 * 100)
print(f"Compression ratio: {compression_ratio:.1f}x")

# 3. TRANSFORMATION MATRIX (Rotation)
def rotation_matrix(theta):
    """2D rotation matrix."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

theta = np.pi / 4  # 45 degrees
R = rotation_matrix(theta)

# 4. PCA FOR FACE RECOGNITION (Eigenfaces)
# Stack images as rows, apply PCA
def eigenfaces(images, n_components=50):
    """images: N x (H*W) matrix."""
    mean_face = images.mean(axis=0)
    centered = images - mean_face
    
    # Covariance and eigendecomposition
    cov = centered.T @ centered / len(images)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Top k eigenvectors (eigenfaces)
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    return eigenvectors[:, idx], mean_face
```

**Convolution as Matrix Multiplication:**
```
[y1]   [k1 k2 k3  0  0] [x1]
[y2] = [ 0 k1 k2 k3  0] [x2]
[y3]   [ 0  0 k1 k2 k3] [x3]
                        [x4]
                        [x5]
```

**Interview Tips:**
- Images are just matrices—all matrix operations apply
- SVD compression trades quality for storage
- Convolution in CNNs is linear operation (matrix multiplication)
- Know that deep learning frameworks use im2col to convert conv to matmul


---

## Question 6
- [ ] Done

**Discuss the role of linear algebra in deep learning, specifically in training convolutional neural networks.**

### Answer

**Definition:**  
Deep learning is fundamentally matrix operations: forward pass = matrix multiplications + activations, backward pass = transposed matrix multiplications. CNNs use specialized convolution operations that are implemented as matrix multiplications.

**Linear Algebra in Neural Networks:**

| Component | Linear Algebra Operation |
|-----------|-------------------------|
| Dense layer | y = Wx + b (matrix-vector multiply) |
| Convolution | im2col + matrix multiply |
| Batch norm | Element-wise scaling (diagonal matrix) |
| Attention | QKᵀ softmax (matrix products) |
| Backprop | Transposed weight matrices |

**Forward Pass:**
$$\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})$$

**Backward Pass (Gradient):**
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \mathbf{h}} \mathbf{x}^T$$
$$\frac{\partial L}{\partial \mathbf{x}} = W^T \frac{\partial L}{\partial \mathbf{h}}$$

**Python Example:**
```python
import numpy as np

# Simple neural network layer (forward + backward)
class DenseLayer:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(out_features, in_features) * 0.01
        self.b = np.zeros(out_features)
    
    def forward(self, x):
        self.x = x  # Cache for backward
        return self.W @ x + self.b  # Matrix-vector multiply
    
    def backward(self, grad_output):
        # Gradient w.r.t. weights: outer product
        self.grad_W = np.outer(grad_output, self.x)
        self.grad_b = grad_output
        # Gradient w.r.t. input: transpose multiply
        return self.W.T @ grad_output

# Convolution as matrix multiplication (im2col)
def conv2d_as_matmul(image, kernel):
    """Convolution via im2col transformation."""
    h, w = image.shape
    kh, kw = kernel.shape
    out_h, out_w = h - kh + 1, w - kw + 1
    
    # Extract patches and flatten to columns
    patches = np.zeros((kh * kw, out_h * out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i:i+kh, j:j+kw]
            patches[:, i * out_w + j] = patch.flatten()
    
    # Convolution = matrix multiply
    kernel_flat = kernel.flatten()
    output_flat = kernel_flat @ patches
    
    return output_flat.reshape(out_h, out_w)

# Batch processing: X is (batch, features)
X = np.random.randn(32, 784)  # 32 samples, 784 features
W = np.random.randn(128, 784)  # 128 output neurons
b = np.zeros(128)

# Forward: Y = XW^T + b
Y = X @ W.T + b  # Shape: (32, 128)
```

**Why Matrix Operations Dominate:**
- Highly parallelizable on GPUs (CUDA, cuBLAS)
- Optimized BLAS libraries for CPUs
- Memory-efficient with batching

**Interview Tips:**
- Convolution IS matrix multiplication (via im2col)
- Backprop uses transposed weights: W^T
- Attention mechanism: softmax(QK^T/√d)V
- Batch normalization: learns diagonal scaling matrix
- Weight initialization matters for gradient flow (eigenvalue spectrum)



---

## Question 7
- [ ] Done

**Propose strategies to visualize high-dimensional data using linear algebra techniques.**

### Answer

**Definition:**  
Visualization requires projecting high-dimensional data to 2D or 3D. Linear techniques (PCA, MDS) preserve global structure; non-linear (t-SNE, UMAP) preserve local neighborhoods.

**Linear Techniques:**

| Method | Preserves | Best For |
|--------|-----------|----------|
| **PCA** | Maximum variance | Global structure, outliers |
| **MDS** | Pairwise distances | Distance-based data |
| **LDA** | Class separation | Supervised visualization |

**Non-Linear (for comparison):**
- **t-SNE**: Local neighborhoods (clusters)
- **UMAP**: Both local and global structure

**Python Example:**
```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# High-dimensional data
np.random.seed(42)
n_samples = 300
X = np.random.randn(n_samples, 50)  # 50 features

# Add cluster structure
X[:100] += 3  # Cluster 1
X[100:200] -= 3  # Cluster 2
labels = np.array([0]*100 + [1]*100 + [2]*100)

# 1. PCA - Project to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# Manual PCA
def pca_2d(X):
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Top 2 eigenvectors
    idx = np.argsort(eigenvalues)[::-1][:2]
    V = eigenvectors[:, idx]
    return X_centered @ V

X_pca_manual = pca_2d(X)

# 2. MDS - Preserve distances
mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X)

# 3. LDA - Supervised, maximize class separation
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, labels)

# Compare projections
print(f"PCA shape: {X_pca.shape}")
print(f"MDS shape: {X_mds.shape}")
print(f"LDA shape: {X_lda.shape}")

# Biplots: Show feature contributions in PCA
def biplot_info(pca, feature_names=None):
    """Get feature loadings for biplot."""
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    return loadings  # Plot as arrows in 2D
```

**Choosing a Method:**
```
Want global structure? → PCA
Have labels? → LDA  
Care about clusters? → t-SNE/UMAP
Need interpretability? → PCA (can trace back to features)
```

**Interview Tips:**
- PCA is linear; can't capture non-linear manifolds
- t-SNE distances are NOT meaningful—only clusters matter
- LDA requires labels; PCA is unsupervised
- Always report variance explained for PCA
- For very high dimensions, use randomized PCA for speed



---

---

## Question 8
- [ ] Done

**Discuss an approach for optimizing memory usage in matrix computations for a large-scale machine learning application.**

### Answer

**Definition:**  
Memory optimization for large matrices involves choosing appropriate data structures, using lazy evaluation, streaming data, and leveraging hardware-specific memory hierarchies.

**Key Strategies:**

| Strategy | Technique | Savings |
|----------|-----------|---------|
| **Sparse format** | CSR, CSC, COO | O(nnz) vs O(n²) |
| **Lower precision** | float16, int8 | 2-4x memory |
| **Chunking** | Process in batches | Fits in RAM |
| **Memory mapping** | np.memmap | Larger than RAM |
| **Gradient checkpointing** | Recompute vs store | Trade compute for memory |
| **In-place operations** | Modify existing arrays | No extra allocation |

**Python Example:**
```python
import numpy as np
from scipy import sparse

# 1. SPARSE MATRICES
# Dense 10000x10000 float64 = 800 MB
# Sparse with 1% density = ~8 MB
n = 10000
A_dense = np.random.randn(n, n)  # 800 MB
A_sparse = sparse.random(n, n, density=0.01, format='csr')
print(f"Sparse memory: {A_sparse.data.nbytes / 1e6:.1f} MB")

# 2. LOWER PRECISION
A_fp32 = A_dense.astype(np.float32)  # 400 MB
A_fp16 = A_dense.astype(np.float16)  # 200 MB
print(f"fp64: {A_dense.nbytes/1e6:.0f} MB")
print(f"fp32: {A_fp32.nbytes/1e6:.0f} MB")
print(f"fp16: {A_fp16.nbytes/1e6:.0f} MB")

# 3. MEMORY-MAPPED FILES (larger than RAM)
# Data stays on disk, loaded on demand
mmap_array = np.memmap('large_matrix.dat', dtype='float32',
                       mode='w+', shape=(50000, 10000))
# Access like normal array, but memory-efficient
mmap_array[0:1000] = np.random.randn(1000, 10000).astype('float32')

# 4. CHUNKED PROCESSING
def chunked_mean(filename, chunk_size=1000):
    """Compute mean without loading entire file."""
    mmap = np.memmap(filename, dtype='float32', mode='r', 
                     shape=(50000, 10000))
    total = 0
    count = 0
    for i in range(0, 50000, chunk_size):
        chunk = mmap[i:i+chunk_size]
        total += chunk.sum()
        count += chunk.size
    return total / count

# 5. IN-PLACE OPERATIONS
A = np.random.randn(1000, 1000)
# Bad: creates copy
B = A * 2  # New allocation
# Good: in-place
A *= 2  # No new allocation
np.multiply(A, 2, out=A)  # Explicit in-place

# 6. GENERATORS FOR STREAMING
def batch_generator(X, batch_size=32):
    """Yield batches without loading all data."""
    n = len(X)
    for i in range(0, n, batch_size):
        yield X[i:i+batch_size]
```

**Deep Learning Specific:**
```python
# Gradient checkpointing: trade compute for memory
# Instead of storing all activations, recompute during backward pass

# Mixed precision training (PyTorch)
# model.half()  # Convert to fp16
# Use torch.cuda.amp for automatic mixed precision
```

**Memory Hierarchy:**
```
Registers → L1 Cache → L2 Cache → RAM → SSD → HDD
Fastest/smallest ←――――――――――――――――――→ Slowest/largest
```

**Interview Tips:**
- Always profile memory first (memory_profiler, tracemalloc)
- Sparse only helps if density < ~10%
- fp16 can cause numerical issues—use mixed precision
- GPU memory is limited; batch size affects memory
- Know that modern frameworks handle much of this automatically

---

## Question 9
- [ ] Done

**Explain how you would use linear algebra to clean and preprocess a dataset.**

---

## Question 10
- [ ] Done

**Given a dataset, determine if PCA would be beneficial and justify your approach.**

