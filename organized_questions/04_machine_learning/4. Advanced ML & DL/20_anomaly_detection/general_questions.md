# Anomaly Detection Interview Questions - General Questions

---

## Question 1: What is anomaly detection?
- [ ] Done

### Answer

**Definition**: Anomaly detection (also called outlier detection) is the identification of data points, observations, or patterns that deviate significantly from the expected behavior or the majority of the data.

**Core Concepts**:

| Term | Description |
|------|-------------|
| Anomaly | Data point significantly different from normal observations |
| Normal Behavior | Expected pattern established from training data |
| Anomaly Score | Numerical measure of how anomalous a point is |

**Mathematical Formulation**:

For a data point $x$ in dataset $D$:
$$\text{Anomaly Score}(x) = f(x) \quad \text{where } f: \mathbb{R}^d \rightarrow \mathbb{R}$$

A point is flagged as anomaly if:
$$f(x) > \tau \quad \text{(threshold)}$$

**Types of Learning Approaches**:

```
Anomaly Detection Approaches
├── Supervised (labeled anomalies available)
├── Semi-supervised (only normal data for training)
└── Unsupervised (no labels, assume anomalies are rare)
```

**Interview Tip**: Emphasize that anomaly detection assumes anomalies are rare and different - this distinguishes it from binary classification where both classes may be equally represented.

---

## Question 2: What are the main types of anomalies in data?
- [ ] Done

### Answer

**Three Main Types**:

| Type | Description | Example |
|------|-------------|---------|
| **Point Anomalies** | Single data point deviating from the rest | Unusually large transaction |
| **Contextual Anomalies** | Normal in one context, anomalous in another | High temperature in summer vs winter |
| **Collective Anomalies** | Group of points anomalous together | Sequence of failed login attempts |

**Visual Representation**:

```
Point Anomaly:        Contextual Anomaly:       Collective Anomaly:
                      
    *                     * (summer)            [* * * * *] ← anomalous
                                                   sequence
  • • •                 • • • •               
 • • • •               • • • •                • • • • • • •
• • • • •             • * (winter)            • • • • • • •
```

**Contextual Anomaly Components**:
- **Contextual attributes**: Define the context (e.g., time, location)
- **Behavioral attributes**: Actual values being evaluated

**Mathematical Example** (Contextual):
$$x_t \text{ is anomalous if } |x_t - \mu_{\text{context}}| > k \cdot \sigma_{\text{context}}$$

**Interview Tip**: Always ask about the context of the data - the same value might be normal or anomalous depending on the situation.

---

## Question 3: How does anomaly detection differ from noise removal?
- [ ] Done

### Answer

**Key Differences**:

| Aspect | Anomaly Detection | Noise Removal |
|--------|-------------------|---------------|
| **Goal** | Find interesting deviations | Remove unwanted variations |
| **Value** | Anomalies are valuable | Noise is discarded |
| **Interpretation** | Anomalies carry meaning | Noise is random error |
| **Action** | Investigate anomalies | Filter out noise |

**Conceptual Difference**:

```
Signal Decomposition:
Data = True Signal + Noise + Anomalies
         ↓            ↓         ↓
       Keep       Remove     Detect!
```

**Example**:
- **Noise**: Sensor measurement jitter (±0.1°C random fluctuation)
- **Anomaly**: Sudden temperature spike indicating equipment failure

**Mathematical Perspective**:

Noise follows expected distribution:
$$\epsilon \sim \mathcal{N}(0, \sigma^2_{\text{known}})$$

Anomalies violate expected distribution:
$$P(x|\text{normal}) < \tau$$

**Interview Tip**: A key insight is that noise is expected and modeled, while anomalies are unexpected and interesting.

---

## Question 4: Explain the concepts of outliers and their impact on dataset
- [ ] Done

### Answer

**Definition**: Outliers are observations that lie at an abnormal distance from other values in a dataset.

**Types of Outliers by Cause**:

| Type | Description | Action |
|------|-------------|--------|
| **Data Entry Errors** | Typos, measurement mistakes | Correct or remove |
| **Measurement Errors** | Instrument malfunction | Investigate and correct |
| **Experimental Errors** | Procedure deviation | Document and decide |
| **Natural Outliers** | Genuine extreme values | Keep and analyze |

**Impact on Statistical Measures**:

```python
# Example impact
normal_data = [10, 12, 11, 13, 12, 11, 10, 12]
with_outlier = [10, 12, 11, 13, 12, 11, 10, 100]  # outlier: 100

# Mean: 11.4 vs 22.4 (heavily affected)
# Median: 11.5 vs 11.5 (robust)
# Std Dev: 1.0 vs 30.7 (heavily affected)
```

**Impact on ML Models**:

| Model Type | Impact of Outliers |
|------------|-------------------|
| Linear Regression | Severely distorts slope |
| K-Means | Pulls centroids toward outliers |
| PCA | Distorts principal components |
| Tree-based | Relatively robust |

**Mathematical Impact** (Least Squares):
$$\hat{\beta} = (X^TX)^{-1}X^Ty$$

Single outlier can dramatically change $\hat{\beta}$ due to squared error penalty.

**Interview Tip**: Always discuss whether outliers should be removed, transformed, or kept based on domain knowledge and analysis goals.

---

## Question 5: What is the difference between supervised and unsupervised anomaly detection?
- [ ] Done

### Answer

**Comparison Table**:

| Aspect | Supervised | Unsupervised |
|--------|------------|--------------|
| **Training Data** | Labeled normal + anomalies | Unlabeled data only |
| **Assumption** | Labels are accurate | Anomalies are rare |
| **Approach** | Classification problem | Density/distance based |
| **Accuracy** | Generally higher | May miss novel anomalies |
| **Practical Use** | When labels available | Most real scenarios |

**Semi-Supervised** (Important Third Category):
- Train only on normal data
- Flag deviations as anomalies
- Most practical in many domains

**Visual Comparison**:

```
Supervised:                 Unsupervised:
                           
Training:                  Training:
• = normal, × = anomaly    • = data (no labels)
                           
• • • ×                    • • • •
• • × •                    • • • •
× • • •                    • • • •
                           Assumes: rare points = anomalies

Test: Classify new point   Test: Score by density/distance
```

**Algorithm Examples**:

| Type | Algorithms |
|------|------------|
| Supervised | Random Forest, SVM, Neural Networks |
| Semi-supervised | One-Class SVM, Autoencoders |
| Unsupervised | Isolation Forest, LOF, DBSCAN |

**Interview Tip**: In practice, labeled anomaly data is rare and expensive. Semi-supervised and unsupervised methods are most common.

---

## Question 6: What are some real-world applications of anomaly detection?
- [ ] Done

### Answer

**Major Application Domains**:

| Domain | Application | Anomaly Type |
|--------|-------------|--------------|
| **Cybersecurity** | Intrusion detection | Unusual network traffic |
| **Finance** | Fraud detection | Suspicious transactions |
| **Healthcare** | Disease detection | Abnormal vital signs |
| **Manufacturing** | Quality control | Defective products |
| **IT Operations** | System monitoring | Server failures |

**Detailed Examples**:

```
1. Credit Card Fraud:
   Normal: $50 coffee shop, $200 grocery
   Anomaly: $5000 electronics in foreign country
   
2. Network Security:
   Normal: 100 requests/min from IP
   Anomaly: 10,000 requests/min (DDoS)
   
3. Medical Diagnosis:
   Normal: Heart rate 60-100 bpm
   Anomaly: Sudden spike to 180 bpm
   
4. Manufacturing:
   Normal: Part dimensions within ±0.1mm
   Anomaly: Part 2mm off specification
```

**Business Impact**:

| Application | Cost of Missing Anomaly |
|-------------|------------------------|
| Fraud Detection | Financial loss |
| Medical Diagnosis | Patient harm |
| Predictive Maintenance | Equipment damage |
| Security | Data breach |

**Interview Tip**: Tailor your anomaly detection approach to the domain - false positive/negative costs vary dramatically across applications.

---

## Question 7: What is the role of statistics in anomaly detection?
- [ ] Done

### Answer

**Statistical Foundation**:

Statistics provides the theoretical basis for defining "normal" and measuring deviation from it.

**Key Statistical Concepts**:

| Concept | Role in Anomaly Detection |
|---------|---------------------------|
| **Distribution** | Model normal behavior |
| **Mean/Variance** | Define center and spread |
| **Percentiles** | Set thresholds |
| **Hypothesis Testing** | Formal anomaly decisions |

**Statistical Approach Flow**:

```
Data → Fit Distribution → Calculate Probability → Flag if P < threshold
         ↓
    Estimate μ, σ
         ↓
    P(x|μ,σ) for new point
         ↓
    If P(x) < 0.01 → Anomaly
```

**Mathematical Framework**:

For Gaussian assumption:
$$P(x|\mu,\sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

Anomaly if:
$$P(x|\mu,\sigma) < \epsilon \quad \text{or equivalently} \quad |x - \mu| > k\sigma$$

**Limitations**:
- Assumes known distribution (often Gaussian)
- Struggles with multimodal or complex distributions
- High-dimensional data challenges (curse of dimensionality)

**Interview Tip**: Statistical methods are interpretable but make assumptions. Always validate distributional assumptions before applying.

---

## Question 8: How do you handle high-dimensional data in anomaly detection?
- [ ] Done

### Answer

**Challenge**: Curse of dimensionality makes distance-based methods ineffective.

**Why High Dimensions Are Problematic**:

```
As dimensions increase:
- All points become equidistant
- Distance metrics lose discriminative power
- Density estimation becomes unreliable
- Computational cost explodes

Distance concentration:
lim(d→∞) [d_max - d_min] / d_min → 0
```

**Solutions**:

| Approach | Method | When to Use |
|----------|--------|-------------|
| **Dimensionality Reduction** | PCA, t-SNE, UMAP | Preserve structure |
| **Feature Selection** | Filter, Wrapper, Embedded | Remove irrelevant features |
| **Subspace Methods** | Feature bagging, random projections | High-d data |
| **Specialized Algorithms** | Isolation Forest, LOF variants | Native high-d support |

**Strategy Comparison**:

```
Dimensionality Reduction:
High-D → PCA/UMAP → Low-D → Standard anomaly detection
                      ↓
                   Retain key variance

Subspace Methods:
High-D → Multiple random subspaces → Detect in each → Aggregate
              ↓                           ↓
         Random projections          Ensemble voting
```

**Python Implementation**:

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.random_projection import GaussianRandomProjection

def high_dim_anomaly_detection(X, method='pca', target_dim=10):
    """
    Handle high-dimensional anomaly detection.
    """
    if method == 'pca':
        # Reduce dimensions with PCA
        pca = PCA(n_components=target_dim)
        X_reduced = pca.fit_transform(X)
        
        # Detect in reduced space
        detector = IsolationForest(contamination=0.1)
        return detector.fit_predict(X_reduced)
    
    elif method == 'subspace':
        # Feature bagging: multiple random subspaces
        n_subspaces = 10
        n_features_per_subspace = min(target_dim, X.shape[1] // 2)
        
        all_predictions = []
        for _ in range(n_subspaces):
            # Random feature subset
            features = np.random.choice(X.shape[1], n_features_per_subspace, replace=False)
            X_sub = X[:, features]
            
            detector = IsolationForest(contamination=0.1)
            predictions = detector.fit_predict(X_sub)
            all_predictions.append(predictions)
        
        # Majority vote
        all_predictions = np.array(all_predictions)
        return np.sign(all_predictions.sum(axis=0))
    
    elif method == 'random_projection':
        # Johnson-Lindenstrauss projection
        projector = GaussianRandomProjection(n_components=target_dim)
        X_projected = projector.fit_transform(X)
        
        detector = IsolationForest(contamination=0.1)
        return detector.fit_predict(X_projected)
```

**Best Practices**:

1. Always visualize reduced data to verify structure preservation
2. Use Isolation Forest for native high-d handling
3. Consider domain knowledge for feature selection
4. Ensemble subspace methods for robustness

**Interview Tip**: Mention that Isolation Forest handles high dimensions better than distance-based methods because it uses random splits, not distances.

---

## Question 9: What are some common statistical methods for anomaly detection?
- [ ] Done

### Answer

**Common Methods**:

| Method | Approach | Best For |
|--------|----------|----------|
| **Z-Score** | Standard deviations from mean | Univariate, Gaussian |
| **IQR** | Quartile-based bounds | Robust to outliers |
| **Grubbs' Test** | Hypothesis testing | Single outlier |
| **Mahalanobis Distance** | Multivariate distance | Correlated features |

**Z-Score Method**:
$$z = \frac{x - \mu}{\sigma}$$

Anomaly if $|z| > 3$ (3-sigma rule)

**IQR Method**:
$$\text{Lower} = Q_1 - 1.5 \times IQR$$
$$\text{Upper} = Q_3 + 1.5 \times IQR$$

**Mahalanobis Distance** (Multivariate):
$$D_M(x) = \sqrt{(x-\mu)^T \Sigma^{-1} (x-\mu)}$$

**Comparison**:

```python
import numpy as np
from scipy import stats

def z_score_anomalies(data, threshold=3):
    z = np.abs(stats.zscore(data))
    return z > threshold

def iqr_anomalies(data, k=1.5):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    return (data < q1 - k*iqr) | (data > q3 + k*iqr)
```

**Method Selection**:

| Data Characteristic | Recommended Method |
|--------------------|-------------------|
| Gaussian, no outliers | Z-Score |
| Non-Gaussian, heavy tails | IQR |
| Multivariate, correlated | Mahalanobis |
| Unknown distribution | Non-parametric (IQR) |

**Interview Tip**: IQR is more robust than Z-score because quartiles are less sensitive to extreme values than mean and standard deviation.

---

## Question 10: Explain the working principle of k-NN in anomaly detection
- [ ] Done

### Answer

**Core Idea**: Points far from their k nearest neighbors are anomalies.

**Algorithm**:

```
1. For each point x:
   a. Find k nearest neighbors
   b. Calculate distance to k-th neighbor (or average distance)
   c. Assign anomaly score = distance
2. Flag points with high scores as anomalies
```

**Distance Metrics**:

| Metric | Formula | Use Case |
|--------|---------|----------|
| Euclidean | $\sqrt{\sum(x_i-y_i)^2}$ | Continuous features |
| Manhattan | $\sum|x_i-y_i|$ | Sparse data |
| Minkowski | $(\sum|x_i-y_i|^p)^{1/p}$ | General purpose |

**Anomaly Score Variants**:

1. **Distance to k-th neighbor**: $d_k(x)$
2. **Average distance to k neighbors**: $\frac{1}{k}\sum_{i=1}^k d_i(x)$
3. **Average distance from k neighbors**: Consider reverse neighbors

**Visual Example**:

```
Normal point (dense region):     Anomaly (isolated):
    • •                              
   • x • ← k=3 neighbors close      x ← k=3 neighbors far
    • •                            
  • • • •                         • • • • •
                                  • • • • •
```

**Python Implementation**:

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

def knn_anomaly_scores(X, k=5):
    nn = NearestNeighbors(n_neighbors=k+1)  # +1 excludes self
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    return distances[:, -1]  # Distance to k-th neighbor
```

**Choosing k**:
- Small k: Sensitive to local variations
- Large k: Smoother but may miss local anomalies
- Rule of thumb: $k \approx \sqrt{n}$

**Interview Tip**: k-NN is simple and intuitive but computationally expensive for large datasets ($O(n^2)$ naive).

---

## Question 11: Describe how cluster analysis can be used for detecting anomalies
- [ ] Done

### Answer

**Core Approaches**:

| Approach | Anomaly Definition |
|----------|-------------------|
| **Distance-based** | Far from nearest cluster centroid |
| **Density-based** | In sparse regions (DBSCAN noise points) |
| **Size-based** | In very small clusters |

**K-Means Based Detection**:

```
1. Cluster data using K-Means
2. For each point, calculate distance to assigned centroid
3. Points with large distances are anomalies
```

**DBSCAN for Anomaly Detection**:

```
Core Point: ≥ minPts neighbors within ε
Border Point: < minPts but neighbor of core
Noise Point: Neither → ANOMALY
```

**Visual Representation**:

```
K-Means approach:          DBSCAN approach:
                           
    ○ ○                       • •
   ○ C ○    * anomaly        • • •   * noise (anomaly)
    ○ ○                       • •
                    *                        *
   ○ ○ ○
  ○ C ○                      • • •
   ○ ○                        • •
```

**Python Implementation**:

```python
from sklearn.cluster import KMeans, DBSCAN
import numpy as np

def kmeans_anomaly_detection(X, n_clusters=5, threshold_percentile=95):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X)
    
    # Distance to assigned centroid
    distances = np.min(kmeans.transform(X), axis=1)
    threshold = np.percentile(distances, threshold_percentile)
    
    return distances > threshold

def dbscan_anomaly_detection(X, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    return labels == -1  # Noise points are anomalies
```

**Interview Tip**: DBSCAN naturally identifies anomalies as noise points without needing to specify anomaly proportion.

---

## Question 12: Explain how the Isolation Forest algorithm works
- [ ] Done

### Answer

**Core Insight**: Anomalies are easier to isolate (separate) than normal points.

**Algorithm**:

```
Building Trees:
1. Randomly select a feature
2. Randomly select split value between min and max
3. Recursively partition until each point is isolated
4. Record path length (number of splits) to isolate each point

Scoring:
- Shorter path = easier to isolate = more anomalous
- Longer path = harder to isolate = more normal
```

**Path Length Intuition**:

```
Normal point (dense):        Anomaly (isolated):
      |                           |
     / \                         / \
    /   \                       *   \
   / \   \                          / \
  / \ /\  /\                       / \ /\
 *                              (many more splits needed)
 
Path length: 5                 Path length: 1
Score: Low (normal)            Score: High (anomaly)
```

**Anomaly Score Formula**:

$$s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}$$

Where:
- $h(x)$ = path length for point x
- $E[h(x)]$ = average path length over all trees
- $c(n)$ = average path length in unsuccessful BST search

**Score Interpretation**:
- $s \approx 1$: Definitely anomaly
- $s \approx 0.5$: Normal point
- $s \approx 0$: Very normal (but rare)

**Python Implementation**:

```python
from sklearn.ensemble import IsolationForest
import numpy as np

def isolation_forest_detection(X, contamination=0.1):
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    predictions = iso_forest.fit_predict(X)
    scores = iso_forest.decision_function(X)
    
    return predictions == -1, scores  # -1 = anomaly
```

**Advantages**:
- Linear time complexity: $O(n \log n)$
- No distance calculations
- Handles high dimensions well
- Few hyperparameters

**Interview Tip**: Isolation Forest is often the go-to algorithm for anomaly detection due to efficiency and effectiveness.

---

## Question 13: Explain the concept of a Z-Score and how it is used in anomaly detection
- [ ] Done

### Answer

**Definition**: Z-score measures how many standard deviations a point is from the mean.

**Formula**:
$$z = \frac{x - \mu}{\sigma}$$

**Interpretation**:

| Z-Score | Interpretation | % of Data (Normal) |
|---------|----------------|-------------------|
| \|z\| < 1 | Normal | 68.3% |
| \|z\| < 2 | Somewhat unusual | 95.4% |
| \|z\| < 3 | Unusual | 99.7% |
| \|z\| > 3 | **Anomaly** | 0.3% |

**Visual Representation**:

```
Normal Distribution:
                    
          ▄▄▄▄
        ▄██████▄
      ▄██████████▄
    ▄██████████████▄
  ▄████████████████████▄
──┴──┴──┴──┴──┴──┴──┴──┴──
 -3σ -2σ -1σ  μ  +1σ +2σ +3σ
  │                       │
  └─── Anomaly zone ──────┘
```

**Python Implementation**:

```python
import numpy as np
from scipy import stats

def z_score_anomaly_detection(data, threshold=3):
    """
    Detect anomalies using Z-score method.
    
    Args:
        data: 1D array of values
        threshold: Z-score threshold (default 3)
    
    Returns:
        Boolean array (True = anomaly)
    """
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold

# Robust version using median and MAD
def robust_z_score(data, threshold=3.5):
    """Modified Z-score using median absolute deviation."""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z = 0.6745 * (data - median) / mad
    return np.abs(modified_z) > threshold
```

**Limitations**:
- Assumes Gaussian distribution
- Sensitive to outliers (mean and std affected)
- Not suitable for multimodal distributions

**Robust Alternative** (Modified Z-Score):
$$M_i = \frac{0.6745(x_i - \tilde{x})}{MAD}$$

Where MAD = Median Absolute Deviation

**Interview Tip**: Use modified Z-score when data may already contain outliers that would skew mean and standard deviation.

---

## Question 14: Describe the autoencoder approach for anomaly detection in neural networks
- [ ] Done

### Answer

**Core Idea**: Train autoencoder on normal data; anomalies have high reconstruction error.

**Architecture**:

```
Input → Encoder → Bottleneck → Decoder → Reconstruction
  x        ↓          z           ↓          x̂
         compress            decompress
         
Anomaly Score = ||x - x̂||²
```

**Mathematical Formulation**:

Encoder: $z = f_\theta(x)$
Decoder: $\hat{x} = g_\phi(z)$

Loss (reconstruction error):
$$\mathcal{L} = \frac{1}{n}\sum_{i=1}^n ||x_i - g_\phi(f_\theta(x_i))||^2$$

**Why It Works for Anomaly Detection**:

```
Normal data:                 Anomaly:
x ──[Encoder]──> z           x ──[Encoder]──> z
        │                            │
   (learned                    (not learned
    pattern)                    pattern)
        │                            │
z ──[Decoder]──> x̂           z ──[Decoder]──> x̂
        │                            │
    x ≈ x̂                       x ≠ x̂
    (low error)                (HIGH ERROR!)
```

**Python Implementation**:

```python
import tensorflow as tf
from tensorflow import keras

def build_autoencoder(input_dim, encoding_dim=32):
    # Encoder
    encoder = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(encoding_dim, activation='relu')
    ])
    
    # Decoder
    decoder = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(encoding_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(input_dim, activation='sigmoid')
    ])
    
    # Autoencoder
    autoencoder = keras.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

def detect_anomalies(autoencoder, X_train, X_test, threshold_percentile=95):
    # Train on normal data only
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, verbose=0)
    
    # Calculate reconstruction errors
    reconstructions = autoencoder.predict(X_test)
    errors = np.mean(np.square(X_test - reconstructions), axis=1)
    
    # Set threshold from training data errors
    train_recon = autoencoder.predict(X_train)
    train_errors = np.mean(np.square(X_train - train_recon), axis=1)
    threshold = np.percentile(train_errors, threshold_percentile)
    
    return errors > threshold
```

**Variants**:
- **Variational Autoencoder (VAE)**: Probabilistic latent space
- **Denoising Autoencoder**: Trained to reconstruct from noisy input
- **Sparse Autoencoder**: Regularized latent representation

**Interview Tip**: Autoencoders are powerful for high-dimensional data (images, sequences) where traditional methods fail.

---

## Question 15: How does Principal Component Analysis (PCA) help in identifying anomalies?
- [ ] Done

### Answer

**Core Idea**: Anomalies have high reconstruction error when projected onto principal components.

**Two Approaches**:

| Approach | Method | Intuition |
|----------|--------|-----------|
| **Major PC** | Distance in PC space | Anomalies far from center |
| **Minor PC** | Reconstruction error | Anomalies violate correlation structure |

**Mathematical Framework**:

PCA decomposition:
$$X = T P^T + E$$

Where:
- $T$ = scores (projections onto PCs)
- $P$ = loadings (principal components)
- $E$ = residual (reconstruction error)

**Anomaly Scores**:

1. **Hotelling's T² (Major Components)**:
$$T^2 = \sum_{i=1}^k \frac{t_i^2}{\lambda_i}$$

2. **Q-statistic/SPE (Residual)**:
$$Q = ||e||^2 = ||x - \hat{x}||^2$$

**Visual Intuition**:

```
PC2 ↑
    │         * (T² high - far from center)
    │    
    │  ○ ○ ○
    │ ○ ○ ○ ○
    │  ○ ○ ○
    │________→ PC1
    
    * (Q high - off the plane)
      ↓
    ○ ○ ○ (normal data lies on PC1-PC2 plane)
```

**Python Implementation**:

```python
from sklearn.decomposition import PCA
import numpy as np

def pca_anomaly_detection(X, n_components=2, threshold_percentile=95):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    # Transform and reconstruct
    X_transformed = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    
    # Reconstruction error (Q-statistic)
    reconstruction_error = np.sum((X - X_reconstructed) ** 2, axis=1)
    
    # T² statistic
    t_squared = np.sum(X_transformed ** 2 / pca.explained_variance_, axis=1)
    
    # Combine scores
    threshold_q = np.percentile(reconstruction_error, threshold_percentile)
    threshold_t = np.percentile(t_squared, threshold_percentile)
    
    anomalies = (reconstruction_error > threshold_q) | (t_squared > threshold_t)
    return anomalies, reconstruction_error, t_squared
```

**When to Use**:
- High-dimensional data with correlated features
- When anomalies violate normal correlation structure
- Process monitoring in manufacturing

**Interview Tip**: Choosing number of components is crucial - too few misses variance, too many includes noise.

---

## Question 16: What are the benefits and drawbacks of using Gaussian Mixture Models for anomaly detection?
- [ ] Done

### Answer

**How GMM Works for Anomaly Detection**:

```
1. Fit GMM to data: P(x) = Σ πₖ N(x|μₖ, Σₖ)
2. Calculate probability density for each point
3. Low probability points = anomalies
```

**Benefits**:

| Benefit | Description |
|---------|-------------|
| **Flexible distributions** | Models multimodal data |
| **Soft clustering** | Probabilistic assignments |
| **Density estimation** | Provides probability scores |
| **Handles elliptical clusters** | Full covariance matrices |

**Drawbacks**:

| Drawback | Description |
|----------|-------------|
| **Requires k selection** | Number of components unknown |
| **Sensitive to initialization** | EM can find local optima |
| **Assumes Gaussian components** | May not fit all data |
| **Struggles high dimensions** | Covariance estimation issues |

**Mathematical Formulation**:

$$P(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$$

Anomaly score:
$$s(x) = -\log P(x)$$

**Python Implementation**:

```python
from sklearn.mixture import GaussianMixture
import numpy as np

def gmm_anomaly_detection(X, n_components=3, threshold_percentile=5):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=42
    )
    gmm.fit(X)
    
    # Log probability density
    log_probs = gmm.score_samples(X)
    
    # Low probability = anomaly
    threshold = np.percentile(log_probs, threshold_percentile)
    
    return log_probs < threshold, log_probs
```

**Comparison with Other Methods**:

| Aspect | GMM | K-Means | Isolation Forest |
|--------|-----|---------|------------------|
| Output | Probability | Distance | Anomaly score |
| Cluster shape | Elliptical | Spherical | N/A |
| Interpretability | High | High | Low |
| Scalability | Medium | High | High |

**Interview Tip**: GMM is excellent when data has clear cluster structure but you need probabilistic anomaly scores rather than hard decisions.

---

