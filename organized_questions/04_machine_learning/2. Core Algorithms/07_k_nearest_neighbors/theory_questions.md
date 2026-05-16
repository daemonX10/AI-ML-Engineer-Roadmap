# K Nearest Neighbors Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What is K-Nearest Neighbors (K-NN) in the context of machine learning?**

### Answer

K-Nearest Neighbors (K-NN) is a simple, non-parametric, instance-based, lazy learning algorithm used for classification and regression. It makes no assumptions about data distribution, memorizes the entire training dataset, and defers computation until prediction time—finding K closest data points to classify or predict based on their labels.

**Core Concepts:**
- **Non-parametric**: No assumptions about underlying data distribution
- **Instance-based**: Memorizes entire dataset instead of learning a function
- **Lazy learning**: No training phase; all computation at prediction time
- **Distance-based**: Uses distance metrics to find similar points

**Mathematical Formulation:**

For a query point $x_q$, find K nearest neighbors from training set:
$$\hat{y} = \text{mode}(y_{n_1}, y_{n_2}, ..., y_{n_K}) \quad \text{(Classification)}$$
$$\hat{y} = \frac{1}{K}\sum_{i=1}^{K} y_{n_i} \quad \text{(Regression)}$$

**Intuition:**
"You are who your friends are" — a point is classified based on the majority class of its nearest neighbors.

**Practical Relevance:**
- Recommendation systems
- Image recognition (baseline)
- Anomaly detection
- Medical diagnosis

**Algorithm Steps:**
1. Store all training data
2. For new point, calculate distance to all training points
3. Select K nearest neighbors
4. Majority vote (classification) or average (regression)

---

## Question 2
- [ ] Done

**How does the K-NN algorithm work for classification problems?**

### Answer

For classification, K-NN predicts the class of a new data point by finding K nearest neighbors in the training set and assigning the class that appears most frequently among them (majority vote). The algorithm stores data during training and performs all distance calculations at prediction time.

**Core Concepts:**
- Majority voting among K neighbors
- Distance calculation to all training points
- Hyperparameters: K value and distance metric
- No explicit model building

**Algorithm Steps:**
1. **Choose K**: Select number of neighbors (typically odd for binary classification)
2. **Choose distance metric**: Usually Euclidean distance
3. **Store training data**: No computation during "training"
4. **For prediction**:
   - Calculate distance from new point to all training points
   - Sort distances and select K smallest
   - Count class labels among K neighbors
   - Assign majority class to new point

**Python Code:**
```python
import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_classify(X_train, y_train, x_new, k=3):
    # Step 1: Calculate distances to all training points
    distances = [euclidean_distance(x_new, x) for x in X_train]
    
    # Step 2: Get indices of K nearest neighbors
    k_indices = np.argsort(distances)[:k]
    
    # Step 3: Get labels of K nearest neighbors
    k_labels = [y_train[i] for i in k_indices]
    
    # Step 4: Majority vote
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]
```

**Interview Tip:**
- Small K → High variance, sensitive to noise (overfitting)
- Large K → High bias, smooth boundary (underfitting)

---

## Question 3
- [ ] Done

**Explain how K-NN can be used for regression.**

### Answer

For regression, K-NN predicts a continuous value by finding K nearest neighbors and returning the average (or weighted average) of their target values. The process of finding neighbors is identical to classification, only the final aggregation step differs—averaging instead of voting.

**Core Concepts:**
- Average of K neighbors' target values instead of majority vote
- Weighted K-NN: closer neighbors contribute more (weight = 1/distance)
- Same distance calculation as classification

**Mathematical Formulation:**

Simple Average:
$$\hat{y} = \frac{1}{K}\sum_{i=1}^{K} y_{n_i}$$

Weighted Average:
$$\hat{y} = \frac{\sum_{i=1}^{K} w_i \cdot y_{n_i}}{\sum_{i=1}^{K} w_i} \quad \text{where } w_i = \frac{1}{d_i}$$

**Python Code:**
```python
import numpy as np

def knn_regression(X_train, y_train, x_new, k=3):
    # Calculate distances
    distances = [np.sqrt(np.sum((x_new - x) ** 2)) for x in X_train]
    
    # Get K nearest indices
    k_indices = np.argsort(distances)[:k]
    
    # Get target values of K neighbors
    k_values = [y_train[i] for i in k_indices]
    
    # Return average
    return np.mean(k_values)
```

**Real-world Example:**
Predicting house price: Find 3 most similar houses → prices are $300K, $320K, $340K → predicted price = $320K

---

## Question 4
- [ ] Done

**How does the choice of distance metric affect the K-NN algorithm's performance?**

### Answer

The distance metric defines what "closeness" means in feature space—wrong choice leads to poor neighbor selection. Different metrics suit different data types: Euclidean for continuous data, Manhattan for high dimensions, Hamming for categorical data. The metric directly shapes the decision boundary.

**Common Distance Metrics:**

| Metric | Formula | Best For |
|--------|---------|----------|
| Euclidean (L2) | $\sqrt{\sum(x_i - y_i)^2}$ | Low-dimensional continuous data |
| Manhattan (L1) | $\sum|x_i - y_i|$ | High-dimensional data, distinct features |
| Minkowski | $(\sum|x_i - y_i|^p)^{1/p}$ | Tunable (p=1: Manhattan, p=2: Euclidean) |
| Hamming | Count of different positions | Categorical/binary data |
| Cosine | $1 - \frac{x \cdot y}{||x|| \cdot ||y||}$ | Text data, NLP |

**Key Effects:**
- **Euclidean**: Sensitive to outliers (squares differences), works well with scaled data
- **Manhattan**: More robust to outliers, better in high dimensions
- **Cosine**: Measures direction, ignores magnitude (good for text)

**Practical Tip:**
- Treat distance metric as a hyperparameter
- Tune using cross-validation
- Always scale features before using Euclidean/Manhattan

---

## Question 5
- [ ] Done

**What are the effects of feature scaling on the K-NN algorithm?**

### Answer

Feature scaling is mandatory for K-NN. Since K-NN uses distance metrics, features with larger scales dominate the distance calculation and effectively ignore smaller-scale features. Without scaling, the algorithm becomes biased toward high-magnitude features regardless of their actual importance.

**The Problem:**
- Age (0-100) vs Income (50,000-200,000)
- Income will dominate distance calculation
- Age becomes negligible in neighbor selection

**Scaling Methods:**

| Method | Formula | Result |
|--------|---------|--------|
| Min-Max Scaling | $(x - min)/(max - min)$ | Values in [0, 1] |
| Standardization | $(x - \mu)/\sigma$ | Mean=0, Std=1 |

**Key Points:**
- Standardization preferred (robust to outliers)
- Fit scaler on training data only
- Transform both train and test with same scaler

**Python Code:**
```python
from sklearn.preprocessing import StandardScaler

# Fit on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data with same scaler
X_test_scaled = scaler.transform(X_test)
```

**Interview Tip:**
Never skip scaling for K-NN — it's a mandatory preprocessing step, not optional.

---

## Question 6
- [ ] Done

**How does K-NN handle multi-class problems?**

### Answer

K-NN handles multi-class classification naturally without modification. The algorithm finds K nearest neighbors and assigns the class with the highest count among them (plurality vote). No restructuring like one-vs-rest is needed—it directly supports C > 2 classes.

**Process:**
1. Find K nearest neighbors
2. Count occurrences of each class among K neighbors
3. Assign class with maximum count

**Example:**
- K = 7, Classes = {Apple, Orange, Banana}
- Neighbors: 4 Apple, 2 Orange, 1 Banana
- Prediction: Apple (majority)

**Handling Ties:**
- Choose odd K (reduces but doesn't eliminate ties for multi-class)
- Use weighted K-NN (sum weights by class)
- Reduce K by 1 and re-vote
- Random selection among tied classes

**Python Code:**
```python
from sklearn.neighbors import KNeighborsClassifier

# Works directly for multi-class
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train, y_train)  # y_train can have multiple classes
predictions = knn.predict(X_test)
```

**Interview Tip:**
Weighted K-NN (`weights='distance'`) naturally handles ties better since exact weight ties are rare.

---

## Question 7
- [ ] Done

**Can K-NN be used for feature selection? If yes, explain how.**

### Answer

Yes, K-NN can be used for feature selection via wrapper methods. The approach treats feature selection as a search problem where K-NN's cross-validated performance evaluates each feature subset. The subset yielding the best K-NN accuracy is selected as optimal features.

**Wrapper Method Process:**
1. Generate candidate feature subsets
2. Train K-NN on each subset
3. Evaluate using cross-validation
4. Select subset with best performance

**Forward Selection with K-NN:**
1. Start with empty feature set
2. Add each feature one by one, evaluate K-NN performance
3. Keep feature that improves performance most
4. Repeat until no improvement

**Why Important for K-NN:**
- K-NN is sensitive to irrelevant features
- Irrelevant features add noise to distance calculations
- Noise can overwhelm signal from important features
- Feature selection provides cleaner distance metric

**Python Code:**
```python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
sfs = SequentialFeatureSelector(knn, n_features_to_select=5, 
                                 direction='forward', cv=5)
sfs.fit(X_train, y_train)

# Get selected feature indices
selected_features = sfs.get_support()
```

**Drawback:** Computationally expensive — requires training K-NN many times.

---

## Question 8
- [ ] Done

**What are the differences between weighted K-NN and standard K-NN?**

### Answer

Standard K-NN gives equal votes to all K neighbors, while weighted K-NN weights each neighbor's vote by inverse of its distance—closer neighbors have more influence. Weighted K-NN is more robust, less sensitive to K choice, and naturally handles ties.

**Key Differences:**

| Aspect | Standard K-NN | Weighted K-NN |
|--------|---------------|---------------|
| Voting | Equal (democratic) | Distance-weighted |
| Weight formula | All = 1 | $w_i = 1/d_i$ |
| Close neighbor influence | Same as distant | Much higher |
| Sensitivity to K | High | Low |
| Tie handling | Can tie | Rarely ties |

**Mathematical Formulation:**

Standard Classification: $\hat{y} = \text{mode}(y_1, ..., y_K)$

Weighted Classification: $\hat{y} = \arg\max_c \sum_{i: y_i=c} \frac{1}{d_i}$

Weighted Regression: $\hat{y} = \frac{\sum w_i \cdot y_i}{\sum w_i}$

**When to Use Weighted K-NN:**
- Noisy data (down-weights distant/noisy neighbors)
- When choice of K is uncertain
- When you want more robust predictions

**Python Code:**
```python
from sklearn.neighbors import KNeighborsClassifier

# Standard K-NN
knn_standard = KNeighborsClassifier(n_neighbors=5, weights='uniform')

# Weighted K-NN
knn_weighted = KNeighborsClassifier(n_neighbors=5, weights='distance')
```

**Interview Tip:**
`weights='distance'` is almost always a good default choice.

---

## Question 9
- [ ] Done

**How does the curse of dimensionality affect K-NN, and how can it be mitigated?**

### Answer

In high-dimensional spaces, all points become nearly equidistant, making "nearest neighbor" meaningless. Data becomes sparse, neighborhoods empty, and irrelevant features add noise to distance calculations. This fundamentally breaks K-NN's core assumption that close points have similar labels.

**Effects on K-NN:**
- **Distance loses meaning**: Nearest and farthest neighbors have similar distances
- **Data sparsity**: Fixed samples spread thin; neighborhoods become empty
- **Noise dominance**: Irrelevant dimensions overwhelm informative ones

**Mathematical Insight:**
As dimensions increase, for random points:
$$\frac{d_{max} - d_{min}}{d_{min}} \rightarrow 0$$

**Mitigation Strategies:**

| Method | Approach | When to Use |
|--------|----------|-------------|
| Feature Selection | Select relevant features only | When interpretability needed |
| PCA | Project to lower dimensions | General dimensionality reduction |
| Autoencoders | Learn compressed representation | Non-linear relationships |

**Python Code:**
```python
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Reduce dimensions before K-NN
pipeline = Pipeline([
    ('pca', PCA(n_components=10)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
pipeline.fit(X_train, y_train)
```

**Interview Tip:**
Never apply K-NN directly to high-dimensional data — always reduce dimensions first.

---

## Question 10
- [ ] Done

**What is the role of data normalization in K-NN, and how is it performed?**

### Answer

Data normalization ensures all features contribute equally to distance calculations. Without it, features with larger scales dominate the metric, effectively ignoring smaller-scale features. K-NN requires scaling as a mandatory preprocessing step—not optional.

**The Problem:**
- Feature A: Age (0-100)
- Feature B: Salary (50,000-200,000)
- Salary difference dominates distance → Age ignored

**Normalization Methods:**

| Method | Formula | Range | When to Use |
|--------|---------|-------|-------------|
| Min-Max | $(x-min)/(max-min)$ | [0, 1] | Known bounds, no outliers |
| Standardization | $(x-\mu)/\sigma$ | Mean=0, Std=1 | Preferred for K-NN |

**Key Rules:**
1. Fit scaler on training data only
2. Transform both train and test with same scaler
3. Never fit on test data (data leakage)

**Python Code:**
```python
from sklearn.preprocessing import StandardScaler

# Correct approach
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit + Transform
X_test_scaled = scaler.transform(X_test)        # Transform only

# Now apply K-NN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
predictions = knn.predict(X_test_scaled)
```

**Interview Tip:**
Standardization is preferred over Min-Max because it's less sensitive to outliers.

---

## Question 11
- [ ] Done

**Describe the process of cross-validation in the context of tuning K-NN's hyperparameters.**

### Answer

Cross-validation systematically evaluates K-NN performance across different hyperparameter values (K, distance metric, weights) using held-out validation folds. GridSearchCV tests all combinations and selects the hyperparameters yielding the best average cross-validated score.

**Hyperparameters to Tune:**
- K: Number of neighbors (1, 3, 5, 7, ...)
- Distance metric: euclidean, manhattan
- Weights: uniform, distance

**Process:**
1. Define hyperparameter grid
2. For each combination:
   - Perform k-fold CV (split data into k folds)
   - Train on k-1 folds, validate on 1 fold
   - Average performance across all folds
3. Select combination with best average score
4. Train final model on full training data

**Python Code:**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Grid search with 5-fold CV
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Best parameters
print(f"Best K: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_
```

**Interview Tip:**
Always use odd K values for binary classification to avoid ties.

---

## Question 12
- [ ] Done

**What is a kd-tree, and how can it be used to optimize K-NN?**

### Answer

A kd-tree (k-dimensional tree) is a space-partitioning binary tree that organizes points by recursively splitting along coordinate axes at medians. It optimizes K-NN by enabling pruning of search space—reducing average complexity from O(N) brute-force to O(log N).

**Building a kd-tree:**
1. Select axis with highest variance
2. Find median along that axis
3. Split: left child (< median), right child (≥ median)
4. Recurse for each child, cycling through axes

**Search Optimization:**
1. Traverse to leaf containing query point (fast)
2. Use leaf points as initial neighbor candidates
3. Backtrack: at each parent, check if other branch could contain closer neighbors
4. **Pruning**: If distance to split plane > distance to current farthest neighbor, skip entire branch

**Complexity:**

| Operation | Brute Force | kd-tree |
|-----------|-------------|---------|
| Build | O(1) | O(N log N) |
| Search | O(N × d) | O(d × log N) |

**Python Code:**
```python
from sklearn.neighbors import KNeighborsClassifier

# Use kd-tree algorithm
knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
knn.fit(X_train, y_train)
```

**Limitation:**
Performance degrades in high dimensions (d > 20) — use ball tree or approximate methods instead.

---

## Question 13
- [ ] Done

**Compare K-NN to decision trees. What are the key differences in their algorithms?**

### Answer

K-NN is a lazy, instance-based learner that stores data and computes at prediction time; Decision Tree is an eager learner that builds a rule-based model during training. K-NN has slow predictions but fast training; Decision Tree has fast predictions but slower training.

**Key Differences:**

| Aspect | K-NN | Decision Tree |
|--------|------|---------------|
| Learning Type | Lazy, instance-based | Eager, model-based |
| Training | Fast (stores data) | Slower (builds tree) |
| Prediction | Slow (computes distances) | Fast (traverses tree) |
| Model | Entire dataset | Tree of if-then rules |
| Feature Scaling | Required | Not required |
| Decision Boundary | Complex, local, non-linear | Axis-parallel, "stair-step" |
| Interpretability | Moderate | High (visualizable) |
| Handles Irrelevant Features | Poorly | Well (feature importance) |

**Decision Boundary Visualization:**
- **K-NN**: Smooth, locally-defined boundaries following data density
- **Decision Tree**: Rectangular regions with horizontal/vertical splits

**When to Choose:**
- **K-NN**: When local patterns matter, small to medium data
- **Decision Tree**: When interpretability needed, mixed feature types

**Interview Tip:**
Decision trees don't need feature scaling because they split features one at a time (relative ordering matters, not magnitude).

---

## Question 14
- [ ] Done

**Explain how K-NN can be adapted for time-series prediction.**

### Answer

K-NN is adapted for time-series by creating a lagged feature space—transforming the series into (X, y) pairs where X is a window of past values and y is the next value. K-NN then finds similar historical patterns and averages their outcomes for prediction.

**Adaptation Process:**

**Step 1: Create Lagged Dataset**
- Choose window size (lag) p
- Features: p consecutive past values
- Target: next time step value

**Example:** Series = [10, 12, 15, 14, 18, 20], lag = 3
| X (Window) | y (Target) |
|------------|------------|
| [10, 12, 15] | 14 |
| [12, 15, 14] | 18 |
| [15, 14, 18] | 20 |

**Step 2: Apply K-NN Regression**
- Find K windows most similar to current window
- Average their target values

**Python Code:**
```python
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

def create_lagged_features(series, lag):
    X, y = [], []
    for i in range(lag, len(series)):
        X.append(series[i-lag:i])
        y.append(series[i])
    return np.array(X), np.array(y)

# Create features
series = [10, 12, 15, 14, 18, 20, 22, 25]
X, y = create_lagged_features(series, lag=3)

# Train and predict
knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(X[:-1], y[:-1])
next_value = knn.predict([X[-1]])  # Predict next
```

**Key Considerations:**
- Tune both lag (p) and K via time-series CV
- Consider making data stationary (differencing)

---

## Question 15
- [ ] Done

**Describe a scenario where you would use K-NN for image classification.**

### Answer

K-NN for image classification works best as a baseline or for content-based image retrieval. The approach extracts feature vectors (from CNN or hand-crafted descriptors), stores them, and classifies new images by finding K most similar images and taking majority vote on their labels.

**Scenario: Content-Based Image Retrieval System**

**Step 1: Feature Extraction** (Critical step)
- Raw pixels won't work (curse of dimensionality)
- Use pre-trained CNN (ResNet, VGG) as feature extractor
- Extract features from penultimate layer (e.g., 2048-dim vector)

**Step 2: Store Features**
- Store all image feature vectors with labels
- This is the K-NN "model"

**Step 3: Prediction**
- Extract features from query image
- Find K nearest feature vectors
- Majority vote for classification OR return K images for retrieval

**Python Code:**
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Load pre-trained model (remove top layer)
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(images):
    preprocessed = preprocess_input(images)
    return base_model.predict(preprocessed)

# Extract features for all training images
train_features = extract_features(train_images)

# Train K-NN on features
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn.fit(train_features, train_labels)

# Classify new image
new_features = extract_features(new_image)
prediction = knn.predict(new_features)
```

**Why K-NN Works Here:**
- Interpretable: Can show which images were neighbors
- Easy to add new images without retraining

---

## Question 16
- [ ] Done

**In a retail context, explain how K-NN could be used for customer segmentation.**

### Answer

**Important Clarification:** K-NN is a supervised learning algorithm requiring labeled data. Customer segmentation is unsupervised (discovering groups without labels). The correct algorithm is K-Means clustering, not K-NN. However, K-NN can be used after segmentation to classify new customers into existing segments.

**Correct Approach: K-Means for Segmentation**

**Step 1: Feature Engineering (RFM Analysis)**
- Recency: Days since last purchase
- Frequency: Total number of purchases
- Monetary: Total amount spent

**Step 2: Preprocessing**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_data)
```

**Step 3: K-Means Clustering**
```python
from sklearn.cluster import KMeans

# Find optimal K using elbow method
kmeans = KMeans(n_clusters=4, random_state=42)
segments = kmeans.fit_predict(rfm_scaled)
```

**Step 4: Profile Segments**
- Segment 1: "High-Value Loyalists"
- Segment 2: "At-Risk Churners"
- Segment 3: "New Customers"
- Segment 4: "Occasional Buyers"

**Where K-NN Fits:**
After segments are defined, use K-NN to classify NEW customers:
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(rfm_scaled, segments)  # segments as labels
new_customer_segment = knn.predict(new_customer_rfm)
```

**Interview Tip:**
Clarifying supervised vs unsupervised distinction shows strong fundamentals.

---

## Question 17
- [ ] Done

**Explain how the K-NN algorithm can be parallelized. What are the challenges and benefits?**

### Answer

K-NN's distance calculations are "embarrassingly parallel"—each distance computation is independent. Data parallelism distributes training data across processors; each computes local K neighbors, then results are merged to find global K nearest neighbors.

**Parallelization Approach:**

**Data Parallelism:**
1. Partition training data across P processors
2. Broadcast query point to all processors
3. Each processor finds local top-K from its partition
4. Merge local results → find global top-K

**Benefits:**
- Massive speedup (up to P times faster)
- Makes K-NN viable for large datasets
- Enables real-time predictions at scale

**Challenges:**
- **Communication overhead**: Broadcasting query and aggregating results
- **Memory requirements**: Each processor needs its data partition
- **Load balancing**: Unequal partitions cause idle processors
- **Aggregation bottleneck**: Merging local results

**Python Code (Scikit-learn):**
```python
from sklearn.neighbors import KNeighborsClassifier

# Use all CPU cores
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)  # Parallelized
```

**Tools for Large-Scale:**
- Scikit-learn: `n_jobs=-1` for multi-core
- Apache Spark: MLlib's K-NN
- RAPIDS cuML: GPU-accelerated K-NN

**Interview Tip:**
The prediction phase is parallelizable; the "training" (storing data) is trivially parallelizable.

---

## Question 18
- [ ] Done

**What are the trends and future advancements in the field of K-NN and its applications?**

### Answer

Key trends focus on overcoming K-NN's limitations: Approximate Nearest Neighbor (ANN) search for scalability, deep metric learning for better similarity measures, GPU acceleration for speed, and integration with deep learning architectures for modern applications.

**Major Advancements:**

**1. Approximate Nearest Neighbor (ANN)**
- Trade small accuracy loss for huge speed gain
- **LSH**: Hash similar items to same bucket
- **HNSW**: Graph-based search (state-of-the-art)
- Libraries: FAISS (Meta), Annoy (Spotify), ScaNN (Google)

**2. Deep Metric Learning**
- Learn embedding space where similar items are close
- Siamese networks, Triplet networks
- Foundation of face recognition, image retrieval

**3. GPU Acceleration**
- NVIDIA cuML (RAPIDS): Orders of magnitude faster
- Enables billion-scale nearest neighbor search

**4. K-NN + Deep Learning**
- K-NN as final layer (replaces softmax)
- Graph Neural Networks: K-NN-like message passing
- Retrieval-augmented models

**Practical Applications:**
- Large-scale recommendation systems
- Real-time image/video retrieval
- Semantic search (embeddings + ANN)
- RAG (Retrieval Augmented Generation) in LLMs

**Python Code (FAISS for ANN):**
```python
import faiss

# Build index
d = features.shape[1]  # dimension
index = faiss.IndexFlatL2(d)
index.add(features)

# Search (much faster than brute force)
k = 5
distances, indices = index.search(query_features, k)
```

---

## Question 19
- [ ] Done

**What are the mathematical foundations and theoretical aspects of K-NN algorithm?**

### Answer

K-NN operates in a metric space with distance functions satisfying mathematical properties. The Cover-Hart theorem provides theoretical justification: with infinite samples, 1-NN error is at most twice the Bayes optimal error. The decision boundary is defined by Voronoi tessellation.

**Mathematical Foundations:**

**1. Metric Space Properties**
Distance function $d(x, y)$ must satisfy:
- Non-negativity: $d(x, y) \geq 0$
- Identity: $d(x, y) = 0 \iff x = y$
- Symmetry: $d(x, y) = d(y, x)$
- Triangle inequality: $d(x, z) \leq d(x, y) + d(y, z)$

**2. Voronoi Tessellation (for 1-NN)**
- Space divided into regions, one per training point
- Each region contains all points closer to that training point than any other
- Decision boundaries are edges between cells

**3. Cover-Hart Theorem (1967)**
$$R_{1NN} \leq 2 \cdot R^* \left(1 - \frac{1}{C}\right)$$

Where:
- $R_{1NN}$: 1-NN error rate
- $R^*$: Bayes optimal error rate
- $C$: Number of classes

**Implication:** 1-NN error ≤ 2 × Bayes error (asymptotically)

**4. Convergence of K-NN**
As $N \rightarrow \infty$ and $K \rightarrow \infty$ (but $K/N \rightarrow 0$):
$$R_{KNN} \rightarrow R^*$$
K-NN converges to Bayes optimal!

**Theoretical Caveat:**
These guarantees assume meaningful distances—fails in high dimensions (curse of dimensionality).

---

## Question 20
- [ ] Done

**How do you choose the optimal value of K in K-NN algorithm?**

### Answer

The optimal K balances bias-variance tradeoff: small K = high variance (overfitting), large K = high bias (underfitting). Use cross-validation to find K with best validation score. Rule of thumb: $K = \sqrt{N}$, but always validate empirically.

**Bias-Variance Tradeoff:**

| K Value | Bias | Variance | Risk |
|---------|------|----------|------|
| Small (1-3) | Low | High | Overfitting, noise-sensitive |
| Large (> √N) | High | Low | Underfitting, over-smoothing |

**Methods to Choose K:**

**1. Rule of Thumb**
$$K = \sqrt{N}$$
Starting point, not guaranteed optimal.

**2. Cross-Validation (Recommended)**
```python
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Test range of K values (odd for binary classification)
k_range = range(1, 31, 2)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Find best K
best_k = k_range[np.argmax(cv_scores)]
print(f"Optimal K: {best_k}")
```

**3. GridSearchCV (Complete Approach)**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': range(1, 31, 2)}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
print(f"Best K: {grid.best_params_['n_neighbors']}")
```

**Best Practices:**
- Always use odd K for binary classification (avoid ties)
- Plot validation score vs K to visualize
- Consider weighted K-NN to reduce K sensitivity

---

## Question 21
- [ ] Done

**What are the different distance metrics used in K-NN and their applications?**

### Answer

Different distance metrics suit different data types: Euclidean for continuous low-dimensional data, Manhattan for high dimensions, Cosine for text/embeddings, Hamming for categorical data. The metric defines "similarity" and directly affects which neighbors are selected.

**Distance Metrics:**

| Metric | Formula | Best Application |
|--------|---------|------------------|
| Euclidean (L2) | $\sqrt{\sum(x_i - y_i)^2}$ | Low-dim continuous data |
| Manhattan (L1) | $\sum|x_i - y_i|$ | High-dim, distinct features |
| Minkowski | $(\sum|x_i - y_i|^p)^{1/p}$ | Tunable (p=1: L1, p=2: L2) |
| Cosine | $1 - \frac{x \cdot y}{\|x\| \|y\|}$ | Text, embeddings, NLP |
| Hamming | Count of mismatches | Categorical, binary data |
| Chebyshev (L∞) | $\max|x_i - y_i|$ | When max difference matters |

**When to Use Which:**
- **Euclidean**: Default for scaled numerical data
- **Manhattan**: High dimensions, robust to outliers
- **Cosine**: When direction matters, not magnitude (documents)
- **Hamming**: Categorical features

**Python Code:**
```python
from sklearn.neighbors import KNeighborsClassifier

# Different metrics
knn_euclidean = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_manhattan = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn_cosine = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn_minkowski = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=3)
```

**Interview Tip:**
Treat metric as hyperparameter — tune via cross-validation alongside K.

---

## Question 22
- [ ] Done

**How does the curse of dimensionality affect K-NN performance?**

### Answer

In high dimensions: (1) all distances become similar—nearest neighbor is almost as far as farthest, (2) data becomes sparse—neighborhoods are empty, (3) irrelevant features add noise that overwhelms signal. This breaks K-NN's core assumption that close points share labels.

**Three Key Effects:**

**1. Distance Concentration**
$$\lim_{d \rightarrow \infty} \frac{d_{max} - d_{min}}{d_{min}} \rightarrow 0$$
- All pairwise distances converge to same value
- "Nearest" becomes meaningless

**2. Data Sparsity**
- Volume grows exponentially: $V \propto r^d$
- Fixed N samples spread extremely thin
- K nearest neighbors may be very far away

**3. Noise Dominance**
- Many features are irrelevant
- Noise in irrelevant dimensions overwhelms signal
- Wrong neighbors get selected

**Practical Impact:**
- Random neighbor selection
- Poor classification accuracy
- Model no better than random guessing

**Solutions:**

| Method | Action |
|--------|--------|
| Feature Selection | Keep only relevant features |
| PCA | Project to lower dimensions |
| t-SNE/UMAP | Non-linear dimensionality reduction |
| Autoencoders | Learn compressed representation |

**Rule of Thumb:**
K-NN typically fails when $d > 20$. Always apply dimensionality reduction for high-dimensional data.

---

## Question 23
- [ ] Done

**What are the computational complexity considerations for K-NN algorithm?**

### Answer

K-NN has inverted complexity: O(1) training (just stores data) but O(N×d) prediction (calculates all distances). This makes K-NN fast to train but slow to predict—opposite of most ML algorithms. Optimization structures like kd-trees reduce prediction to O(d×log N).

**Complexity Analysis:**

| Phase | Brute Force | kd-tree | Ball Tree |
|-------|-------------|---------|-----------|
| Training | O(1) or O(N×d) | O(N×d×log N) | O(N×d×log N) |
| Prediction | O(N×d) | O(d×log N)* | O(d×log N)* |
| Space | O(N×d) | O(N×d) | O(N×d) |

*Average case; degrades in high dimensions

**Where:**
- N = number of training samples
- d = number of dimensions
- k = number of neighbors

**Bottleneck Analysis:**
```
Single prediction: O(N×d) distance calculations + O(N×log k) sorting
```

**Why This Matters:**
- 1M training points, 100 features → 100M operations per prediction
- Not suitable for real-time applications without optimization

**Optimization Strategies:**

| Strategy | Speedup | Trade-off |
|----------|---------|-----------|
| kd-tree | O(log N) | Fails in high-d |
| Ball tree | O(log N) | Better for high-d |
| LSH/HNSW | Sub-linear | Approximate results |
| GPU (cuML) | 10-100x | Hardware requirement |

**Python Code:**
```python
from sklearn.neighbors import KNeighborsClassifier

# Choose algorithm
knn = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='auto'  # 'ball_tree', 'kd_tree', 'brute'
)
```

---

## Question 24
- [ ] Done

**How do you implement efficient nearest neighbor search algorithms?**

### Answer

Efficient NN search uses space-partitioning structures (kd-tree, ball tree) to prune search space, or approximate methods (LSH, HNSW) for massive scale. The key is avoiding brute-force O(N) search by eliminating regions that cannot contain nearer neighbors.

**Key Algorithms:**

**1. kd-tree**
- Best for: d < 20
- Build: Recursively split at median along each axis
- Search: Traverse to leaf, backtrack with pruning
```python
knn = KNeighborsClassifier(algorithm='kd_tree')
```

**2. Ball Tree**
- Best for: Higher dimensions, arbitrary metrics
- Build: Partition into nested hyperspheres
- Search: Use triangle inequality for pruning
```python
knn = KNeighborsClassifier(algorithm='ball_tree')
```

**3. Approximate Methods (for massive scale)**

| Method | Description | Library |
|--------|-------------|---------|
| LSH | Hash similar items to same bucket | sklearn |
| HNSW | Hierarchical graph navigation | FAISS |
| IVF | Inverted file index | FAISS |

**FAISS Example (Billion-scale):**
```python
import faiss

# Build index
d = 128  # dimension
index = faiss.IndexFlatL2(d)
index.add(training_vectors)

# Fast search
k = 5
distances, indices = index.search(query_vectors, k)
```

**Scikit-learn Implementation:**
```python
from sklearn.neighbors import NearestNeighbors

# Let sklearn choose best algorithm
nn = NearestNeighbors(n_neighbors=5, algorithm='auto')
nn.fit(X_train)

# Find neighbors
distances, indices = nn.kneighbors(X_query)
```

**Decision Guide:**
- Small data, low-d: kd-tree
- Higher-d: ball tree
- Millions of points: FAISS/Annoy (approximate)

---

## Question 25
- [ ] Done

**What are K-D trees and how do they optimize K-NN search?**

### Answer

K-D tree is a binary tree that recursively partitions space by splitting at median along each axis. It optimizes K-NN by enabling branch pruning during search—if a hyperplane is farther than current best neighbor, entire branch is skipped. Reduces O(N) to O(log N) average case.

**Building K-D Tree:**
1. Select axis (highest variance or cycle through)
2. Find median value along axis
3. Split: left child < median, right child ≥ median
4. Recurse for each child

**Visual Example (2D):**
```
       [7,2]          <- Split on x=7
      /     \
   [5,4]    [9,6]     <- Split on y
   /  \     /  \
 [2,3] [4,7] [8,1] [9,9]
```

**Search Algorithm:**
1. Traverse to leaf containing query point
2. Set leaf points as initial candidates
3. Backtrack up tree
4. **Pruning check**: If distance to split plane > distance to current farthest neighbor → skip branch
5. Otherwise, explore other branch

**Pruning Condition:**
```
if |query[split_axis] - split_value| > distance_to_kth_neighbor:
    skip entire branch
```

**Complexity:**

| Operation | Complexity |
|-----------|------------|
| Build | O(N × d × log N) |
| Search (avg) | O(d × log N) |
| Search (worst) | O(N × d) |

**Limitation:**
Performance degrades when d > 20 (curse of dimensionality makes pruning ineffective).

**Python:**
```python
from sklearn.neighbors import KDTree

tree = KDTree(X_train)
distances, indices = tree.query(X_test, k=5)
```

---

## Question 26
- [ ] Done

**How do ball trees improve K-NN performance for high-dimensional data?**

### Answer

Ball trees partition space into nested hyperspheres instead of axis-aligned rectangles (kd-trees). They use triangle inequality for pruning and work with any distance metric satisfying this property. More effective than kd-trees when d > 20 due to tighter bounding regions.

**Building Ball Tree:**
1. Create root ball containing all points (center + radius)
2. Partition points into two clusters
3. Create child balls enclosing each cluster
4. Recurse until leaf size threshold

**Why Better for High Dimensions:**
- kd-trees: Axis-aligned rectangles become elongated, poor pruning
- Ball trees: Hyperspheres adapt to data shape, tighter bounds

**Search with Pruning:**
```
if distance(query, ball_center) - ball_radius > current_farthest_neighbor:
    skip entire ball (no closer points possible)
```

**Triangle Inequality:**
$$d(q, p) \geq d(q, c) - d(c, p) \geq d(q, c) - r$$

Where: q = query, c = ball center, r = radius, p = any point in ball

**Comparison:**

| Aspect | kd-tree | Ball Tree |
|--------|---------|-----------|
| Partition shape | Rectangles | Hyperspheres |
| Best for | d < 20 | d > 20 |
| Metric restriction | Mainly Euclidean | Any metric with triangle inequality |
| Pruning efficiency | Degrades in high-d | More robust |

**Python Code:**
```python
from sklearn.neighbors import BallTree, KNeighborsClassifier

# Direct ball tree
tree = BallTree(X_train, metric='euclidean')
dist, idx = tree.query(X_test, k=5)

# Via K-NN classifier
knn = KNeighborsClassifier(algorithm='ball_tree', metric='manhattan')
```

---

## Question 27
- [ ] Done

**What is locality-sensitive hashing (LSH) and its role in approximate K-NN?**

### Answer

LSH uses hash functions that map similar items to the same bucket with high probability. For approximate K-NN, you only search within the query's bucket(s) instead of entire dataset. Trades small accuracy loss for massive speed gain—enables sub-linear search time on billion-scale data.

**Core Principle:**
- If points are close → high probability of same hash
- If points are far → low probability of same hash

**How LSH Works:**

**1. Hash Table Construction:**
- Create multiple hash tables with LSH functions
- Hash all training points into buckets

**2. Search:**
- Hash query point
- Retrieve candidates from matching buckets
- Brute-force search only among candidates

**Why "Approximate":**
- True neighbor might hash to different bucket
- Multiple hash tables increase recall
- Tunable accuracy-speed tradeoff

**Common LSH Variants:**

| Variant | Distance Metric | Application |
|---------|-----------------|-------------|
| Random Projection | Cosine | Text, embeddings |
| p-stable | Euclidean | General |
| MinHash | Jaccard | Document similarity |

**Python Code:**
```python
from sklearn.neighbors import LSHForest  # Deprecated but illustrative

# Modern approach: Use FAISS
import faiss

d = 128  # dimension
nlist = 100  # number of clusters

# LSH-like index
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)
index.train(training_data)
index.add(training_data)

# Search
index.nprobe = 10  # search 10 nearest clusters
D, I = index.search(query, k=5)
```

**Use Cases:**
- Large-scale image retrieval
- Duplicate detection
- Recommendation systems

---

## Question 28
- [ ] Done

**How do you handle missing values in K-NN algorithms?**

### Answer

K-NN cannot compute distances with missing values. Solutions: (1) delete rows/columns with missing data, (2) impute with mean/median before K-NN, or (3) use KNNImputer which leverages K-NN itself to impute based on similar samples. KNNImputer is the most principled approach.

**Strategies:**

| Method | Approach | When to Use |
|--------|----------|-------------|
| Deletion | Remove rows with missing values | Very few missing values |
| Mean/Median Imputation | Fill with column mean/median | Simple, quick |
| KNNImputer | Use neighbors to impute | Best accuracy |

**KNNImputer Process:**
1. For sample with missing value in feature F
2. Find K nearest neighbors using non-missing features
3. Impute F as mean of neighbors' F values

**Python Code:**
```python
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Impute missing values
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Step 2: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Step 3: Train K-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y_train)
```

**Why KNNImputer is Best:**
- Uses local data structure (not global mean)
- Preserves relationships between features
- Consistent with K-NN's instance-based philosophy

**Interview Tip:**
Always impute BEFORE scaling, and fit imputer only on training data.

---

## Question 29
- [ ] Done

**What are weighted K-NN algorithms and when should you use them?**

### Answer

Weighted K-NN assigns importance to neighbors based on distance—closer neighbors get higher weights (typically 1/distance). Use it when you want robust predictions, reduced sensitivity to K choice, natural tie-breaking, and when you suspect closer neighbors are more reliable predictors.

**Weighting Scheme:**
$$w_i = \frac{1}{d_i} \quad \text{or} \quad w_i = \frac{1}{d_i^2}$$

**Classification:**
$$\hat{y} = \arg\max_c \sum_{i: y_i = c} w_i$$

**Regression:**
$$\hat{y} = \frac{\sum_{i=1}^{K} w_i \cdot y_i}{\sum_{i=1}^{K} w_i}$$

**When to Use Weighted K-NN:**

| Scenario | Why Weighted Helps |
|----------|-------------------|
| Noisy data | Down-weights distant noisy neighbors |
| Uncertain K | Less sensitive to K choice |
| Ties in voting | Weights rarely tie exactly |
| Varying density | Closer = more reliable |

**Python Code:**
```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Classification
knn_clf = KNeighborsClassifier(n_neighbors=5, weights='distance')

# Regression  
knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')

# Custom weight function
def custom_weights(distances):
    return 1 / (distances + 0.001)  # Add small constant to avoid division by zero

knn_custom = KNeighborsClassifier(n_neighbors=5, weights=custom_weights)
```

**Best Practice:**
Default to `weights='distance'` — it's almost always equal or better than uniform weights.

---

## Question 30
- [ ] Done

**How do you implement K-NN for categorical and mixed data types?**

### Answer

For categorical data, use Hamming distance (counts mismatches). For mixed data, use Gower's distance which handles numerical and categorical features differently then combines them. Practical approach: one-hot encode categorical features + scale numerical features, then use standard distance metrics.

**Distance Metrics by Data Type:**

| Data Type | Metric | Calculation |
|-----------|--------|-------------|
| Numerical | Euclidean/Manhattan | Standard formula |
| Categorical | Hamming | Count of differences |
| Mixed | Gower's | Weighted combination |

**Gower's Distance:**
$$d_{gower} = \frac{\sum_k w_k \cdot d_k}{\sum_k w_k}$$

Where:
- Numerical: $d_k = \frac{|x_k - y_k|}{range_k}$
- Categorical: $d_k = 0$ if same, $1$ if different

**Practical Approach (Most Common):**
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Define transformers
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'income']),
    ('cat', OneHotEncoder(), ['gender', 'city'])
])

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

**Using Gower's Distance:**
```python
import gower
# pip install gower

distance_matrix = gower.gower_matrix(df)
```

**Interview Tip:**
One-hot encoding + scaling is the most practical and widely used approach.

---

## Question 31
- [ ] Done

**What is the role of feature selection and dimensionality reduction in K-NN?**

### Answer

Feature selection and dimensionality reduction are critical for K-NN: they mitigate the curse of dimensionality (makes distance meaningful), remove irrelevant features (reduces noise in distance calculation), and improve computational efficiency (fewer features = faster distance calculation).

**Why Essential for K-NN:**

| Problem | How DR Helps |
|---------|--------------|
| Curse of dimensionality | Creates meaningful distances |
| Irrelevant features | Removes noise from distance metric |
| Computational cost | Fewer dimensions = faster |

**Two Approaches:**

**1. Feature Selection (Keep Original Features)**
- Maintain interpretability
- Methods: RFE, Lasso, Mutual Information
```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif

selector = SelectKBest(mutual_info_classif, k=10)
X_selected = selector.fit_transform(X_train, y_train)
```

**2. Feature Extraction (Create New Features)**
- Capture variance in fewer dimensions
- Methods: PCA, Autoencoders
```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('pca', PCA(n_components=10)),
    ('scaler', StandardScaler()),  # PCA output may need scaling
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
```

**Decision Guide:**
- Interpretability needed → Feature Selection
- Maximum performance → PCA or Autoencoders
- Very high dimensions → Always reduce before K-NN

**Interview Tip:**
For K-NN, dimensionality reduction is not optimization—it's essential for correctness.

---

## Question 32
- [ ] Done

**How do you handle imbalanced datasets with K-NN algorithm?**

### Answer

K-NN is biased toward majority class because majority neighbors dominate voting. Solutions: (1) use weighted K-NN to give closer minority neighbors more influence, (2) resample data with SMOTE/undersampling, (3) adjust K carefully via stratified CV, (4) use class-weighted voting.

**The Problem:**
- Majority class is denser in feature space
- K neighbors statistically more likely from majority class
- Model predicts majority class most of the time

**Strategies:**

| Strategy | How It Helps |
|----------|--------------|
| Weighted K-NN | Close minority neighbors get high weights |
| SMOTE | Creates synthetic minority samples |
| Undersampling | Balances by reducing majority |
| Stratified CV | Ensures proper evaluation |

**Python Code:**
```python
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Option 1: Weighted K-NN (simplest)
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

# Option 2: SMOTE + K-NN
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_resampled, y_resampled)

# Always use stratified CV for imbalanced data
cv = StratifiedKFold(n_splits=5)
scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='f1')
```

**Recommended Approach:**
1. First try `weights='distance'`
2. If still poor, apply SMOTE to training data
3. Always use F1-score or AUC-ROC, not accuracy

---

## Question 33
- [ ] Done

**What are ensemble methods for K-NN and their advantages?**

### Answer

Ensemble K-NN combines multiple K-NN models via bagging (different data subsets) or random subspace method (different feature subsets). Advantages: reduces variance, improves stability, handles curse of dimensionality, and produces more robust predictions than single K-NN.

**Ensemble Approaches:**

**1. Bagging with K-NN**
- Train N K-NN models on bootstrap samples
- Final prediction: majority vote
- Reduces variance, smooths decision boundary

**2. Random Subspace Method (Feature Bagging)**
- Each K-NN uses random subset of features
- Effective for high-dimensional data
- Each model works in lower-dimensional space

**3. Boosting (Less Common)**
- Sequential models focus on errors
- Risk: K-NN is low-bias, can overfit quickly

**Python Code (Bagging):**
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Bagging with K-NN base estimator
bagged_knn = BaggingClassifier(
    estimator=KNeighborsClassifier(n_neighbors=5),
    n_estimators=10,
    max_samples=0.8,       # 80% of samples per model
    max_features=0.8,      # 80% of features (random subspace)
    bootstrap=True,
    n_jobs=-1
)

bagged_knn.fit(X_train, y_train)
predictions = bagged_knn.predict(X_test)
```

**Advantages:**
- Reduces K-NN's high variance
- More stable predictions
- Handles noisy data better
- Feature bagging mitigates curse of dimensionality

**Interview Tip:**
Random subspace method is especially effective for K-NN in high dimensions.

---

## Question 34
- [ ] Done

**How do you implement cross-validation for K-NN model selection?**

### Answer

Use GridSearchCV with a Pipeline (scaler + K-NN) to prevent data leakage. Define parameter grid for n_neighbors, weights, and metric. Use StratifiedKFold for classification. The pipeline ensures scaling is fit only on training folds.

**Implementation Steps:**
1. Create Pipeline (scaler → K-NN)
2. Define hyperparameter grid
3. Use StratifiedKFold for CV
4. Run GridSearchCV
5. Evaluate best model on test set

**Python Code:**
```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np

# Step 1: Create Pipeline (prevents data leakage)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Step 2: Define parameter grid
param_grid = {
    'knn__n_neighbors': np.arange(1, 31, 2),  # Odd values
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}

# Step 3: Set up cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Step 4: Run GridSearchCV
grid_search = GridSearchCV(
    pipeline, param_grid, cv=cv, 
    scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Step 5: Results
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Evaluate on test set
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
```

**Why Pipeline is Critical:**
- Scaler fit on each training fold only
- No data leakage from test/validation folds

---

## Question 35
- [ ] Done

**What is adaptive K-NN and how does it improve performance?**

### Answer

Adaptive K-NN modifies parameters based on local data characteristics: (1) variable K—adapts neighborhood size based on local density, (2) metric learning—learns optimal distance function from data. Most powerful is LMNN (Large Margin Nearest Neighbor) which learns a task-specific metric.

**Adaptive Approaches:**

**1. Adaptive K (Variable Neighborhood)**
- Sparse regions: larger K for stability
- Dense regions: smaller K for precision
- Adapts based on local density

**2. Metric Learning (Most Powerful)**
- Learn distance function optimized for classification
- Pull same-class neighbors closer
- Push different-class points apart

**LMNN (Large Margin Nearest Neighbor):**
- Learns Mahalanobis distance: $d(x,y) = \sqrt{(x-y)^T M (x-y)}$
- Optimizes M to separate classes
- Automatically weights features by importance

**Python Code (Metric Learning):**
```python
# pip install metric-learn
from metric_learn import LMNN
from sklearn.neighbors import KNeighborsClassifier

# Learn optimal metric
lmnn = LMNN(k=3, learn_rate=1e-6)
lmnn.fit(X_train, y_train)

# Transform data to learned space
X_train_transformed = lmnn.transform(X_train)
X_test_transformed = lmnn.transform(X_test)

# Apply K-NN in learned space
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_transformed, y_train)
```

**Improvements:**
- Better accuracy (learned metric is task-specific)
- Automatic feature weighting
- Accounts for feature correlations

---

## Question 36
- [ ] Done

**How do you handle noise and outliers in K-NN algorithms?**

### Answer

K-NN is sensitive to noise/outliers, especially with small K. Solutions: (1) increase K for more robust voting, (2) use weighted K-NN to down-weight distant outliers, (3) preprocess with outlier detection, (4) use Edited Nearest Neighbors (ENN) to clean training data.

**Impact of Noise/Outliers:**
- Small K: single noisy neighbor can flip prediction
- Jagged, unreliable decision boundary
- Outliers become "wrong" neighbors

**Strategies:**

| Method | How It Helps |
|--------|--------------|
| Larger K | Noisy point gets outvoted |
| Weighted K-NN | Distant outliers get low weight |
| ENN cleaning | Removes noisy training points |
| Outlier detection | Remove before training |

**Edited Nearest Neighbors (ENN):**
```python
from imblearn.under_sampling import EditedNearestNeighbours

# Remove points misclassified by their neighbors
enn = EditedNearestNeighbours(n_neighbors=3)
X_clean, y_clean = enn.fit_resample(X_train, y_train)

# Train K-NN on cleaned data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_clean, y_clean)
```

**Outlier Detection Preprocessing:**
```python
from sklearn.ensemble import IsolationForest

# Detect outliers
iso = IsolationForest(contamination=0.05)
outlier_mask = iso.fit_predict(X_train) == 1

# Train on clean data
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train[outlier_mask], y_train[outlier_mask])
```

**Recommended Approach:**
1. Use `weights='distance'` (always helps)
2. Tune K via cross-validation
3. If noise severe, apply ENN cleaning

---

## Question 37
- [ ] Done

**What are the memory optimization techniques for large-scale K-NN?**

### Answer

K-NN stores entire dataset as "model"—memory-intensive. Optimization techniques: (1) data reduction via prototyping/CNN, (2) product quantization compresses vectors, (3) lower precision storage (float32→float16), (4) distributed storage across cluster nodes.

**Techniques:**

| Technique | Memory Reduction | Trade-off |
|-----------|------------------|-----------|
| Prototype Selection | Store only centroids | Some accuracy loss |
| Product Quantization | 4-32x compression | Approximate results |
| Lower Precision | 2-4x reduction | Minor precision loss |
| Distributed | Scales horizontally | Communication overhead |

**1. Prototype Selection (Condensed NN):**
- Store only essential training points
- Points near decision boundary matter most

**2. Product Quantization (PQ):**
- Split vector into subvectors
- Quantize each subvector separately
- Store only cluster IDs
```python
import faiss

d = 128  # dimension
m = 8    # subquantizers
bits = 8 # bits per subquantizer

# Create PQ index (compressed)
index = faiss.IndexPQ(d, m, bits)
index.train(training_data)
index.add(training_data)
```

**3. Lower Precision:**
```python
import numpy as np

# Convert to float16
X_train_compressed = X_train.astype(np.float16)  # 50% memory
```

**4. Distributed (Spark):**
- Partition data across worker nodes
- Each node handles local search
- Aggregate results

---

## Question 38
- [ ] Done

**How do you implement distributed and parallel K-NN algorithms?**

### Answer

Distributed K-NN partitions training data across cluster nodes. For prediction: broadcast query to all nodes, each finds local K nearest, aggregate results to find global K nearest. Uses MapReduce pattern—embarrassingly parallel since distance calculations are independent.

**Distributed Strategy:**

**Step 1: Data Partition**
- Split training data across M worker nodes
- Each node holds ~N/M samples

**Step 2: Broadcast Query**
- Send query point to all workers

**Step 3: Map (Parallel)**
- Each worker finds local top-K neighbors
- Returns K candidates with distances

**Step 4: Reduce (Aggregate)**
- Collect M×K candidates at driver
- Find global top-K from candidates
- Make final prediction

**Python Code (Conceptual):**
```python
# Using Spark (pseudo-code)
from pyspark import SparkContext

sc = SparkContext()

# Partition training data
training_rdd = sc.parallelize(training_data, numSlices=100)

def find_local_k_nearest(partition, query, k):
    """Find K nearest in local partition"""
    distances = [(euclidean(query, x), label) for x, label in partition]
    return sorted(distances)[:k]

# Broadcast query and find neighbors
query_broadcast = sc.broadcast(query_point)
local_neighbors = training_rdd.mapPartitions(
    lambda p: find_local_k_nearest(p, query_broadcast.value, k)
)

# Aggregate and find global K nearest
global_neighbors = local_neighbors.takeOrdered(k, key=lambda x: x[0])
```

**Tools:**
- Apache Spark MLlib
- Dask (Python parallel)
- scikit-learn: `n_jobs=-1` (multi-core)

---

## Question 39
- [ ] Done

**What is the role of K-NN in collaborative filtering and recommendation systems?**

### Answer

K-NN is the foundation of memory-based collaborative filtering. User-based: find K similar users, recommend what they liked. Item-based: find K similar items to what user liked. Uses cosine similarity or Pearson correlation as distance metric on rating vectors.

**Two Approaches:**

**1. User-Based Collaborative Filtering**
- Concept: "Similar users like similar items"
- Process: Find K users with similar rating patterns
- Recommend items those users rated highly

**2. Item-Based Collaborative Filtering**
- Concept: "User will like items similar to ones they liked"
- Process: Find K items similar to user's liked items
- More stable, commonly used (Amazon)

**Example (User-Based):**
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# User-Item rating matrix (rows=users, cols=items)
ratings = np.array([
    [5, 3, 0, 1],  # User 0
    [4, 0, 0, 1],  # User 1
    [1, 1, 0, 5],  # User 2
    [0, 0, 5, 4],  # User 3 (target user)
])

# Find similar users for User 3
user_similarity = cosine_similarity(ratings)
target_user = 3

# Get K=2 most similar users (excluding self)
similar_users = np.argsort(user_similarity[target_user])[-3:-1]

# Predict rating for item 1 (unrated by user 3)
item_idx = 1
neighbor_ratings = ratings[similar_users, item_idx]
weights = user_similarity[target_user, similar_users]
predicted_rating = np.dot(weights, neighbor_ratings) / weights.sum()
```

**Advantages:**
- Simple and interpretable
- Easy to add new users/items

**Limitations:**
- Sparsity problem
- Cold start for new users

---

## Question 40
- [ ] Done

**How do you use K-NN for anomaly detection and outlier identification?**

### Answer

Anomalies are isolated points with sparse neighborhoods. K-NN detects them by measuring: (1) distance to K-th nearest neighbor—large distance indicates anomaly, or (2) average distance to K neighbors. Points with distances exceeding threshold are flagged as outliers.

**Core Idea:**
- Normal points: dense neighborhood, small distances
- Anomalies: sparse neighborhood, large distances

**Methods:**

**1. K-th Neighbor Distance**
- Compute distance to K-th nearest neighbor
- High distance → likely anomaly
```python
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=5)
nn.fit(X)
distances, _ = nn.kneighbors(X)

# K-th neighbor distance (last column)
kth_distances = distances[:, -1]

# Flag anomalies (top 5%)
threshold = np.percentile(kth_distances, 95)
anomalies = kth_distances > threshold
```

**2. Average K-Neighbor Distance**
```python
avg_distances = distances.mean(axis=1)
anomalies = avg_distances > np.percentile(avg_distances, 95)
```

**3. Local Outlier Factor (LOF)**
- Compares local density to neighbors' density
- More sophisticated than simple distance
```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
outlier_labels = lof.fit_predict(X)  # -1 = outlier, 1 = inlier
```

**Use Cases:**
- Fraud detection
- Network intrusion detection
- Manufacturing defect detection

---

## Question 41
- [ ] Done

**What is local outlier factor (LOF) and its relationship to K-NN?**

### Answer

LOF is a K-NN based anomaly detection algorithm that measures local density deviation. Unlike simple K-th neighbor distance, LOF compares a point's density to its neighbors' densities. LOF > 1 means point is less dense than neighbors (outlier); LOF ≈ 1 means similar density (inlier).

**How LOF Works:**

**Step 1: Calculate k-distance and reachability distance**
- k-distance(A) = distance to K-th nearest neighbor
- reachability-distance(A, B) = max(k-distance(B), actual-distance(A, B))

**Step 2: Calculate Local Reachability Density (LRD)**
$$LRD(A) = \frac{1}{\frac{\sum_{B \in N_k(A)} reach\text{-}dist(A, B)}{|N_k(A)|}}$$

**Step 3: Calculate LOF**
$$LOF(A) = \frac{\sum_{B \in N_k(A)} \frac{LRD(B)}{LRD(A)}}{|N_k(A)|}$$

**Interpretation:**
- LOF ≈ 1: Normal point (similar density to neighbors)
- LOF > 1: Less dense than neighbors (potential outlier)
- LOF >> 1: Much less dense (strong outlier)

**Python Code:**
```python
from sklearn.neighbors import LocalOutlierFactor

# Detect outliers
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
outlier_labels = lof.fit_predict(X)  # -1 = outlier, 1 = inlier

# Get LOF scores
lof_scores = -lof.negative_outlier_factor_
print(f"Outliers: {sum(outlier_labels == -1)}")
```

**Advantage over Simple K-NN Distance:**
- Adapts to varying local densities
- Dense cluster next to sparse region: identifies sparse points correctly
- Simple distance would flag all sparse region points

---

## Question 42
- [ ] Done

**How do you implement K-NN for time-series classification and forecasting?**

### Answer

For time-series, transform sequential data into supervised format using lagged features (sliding window). Classification: predict class labels for sequences. Forecasting: predict next value(s). Key: use time-based train/test split, not random shuffle.

**Time-Series Classification:**
- Input: Sequence of values
- Output: Class label (e.g., "normal", "anomaly")
- Approach: Use entire sequence or extract features

```python
# Classify based on sequence features
from sklearn.neighbors import KNeighborsClassifier

# Example: ECG heartbeat classification
# Each sample is a 187-point sequence
X_sequences = ...  # Shape: (n_samples, 187)
y_labels = ...     # Shape: (n_samples,)

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_sequences, y_train)
predictions = knn.predict(X_test_sequences)
```

**Time-Series Forecasting:**
```python
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

def create_sequences(series, window_size):
    X, y = [], []
    for i in range(window_size, len(series)):
        X.append(series[i-window_size:i])
        y.append(series[i])
    return np.array(X), np.array(y)

# Create lagged features
window = 10
X, y = create_sequences(time_series, window)

# Time-based split (NO SHUFFLE!)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# K-NN Regression
knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn.fit(X_train, y_train)
forecasts = knn.predict(X_test)
```

**Important Considerations:**
- Never shuffle time-series data
- Use time-series cross-validation
- Consider DTW (Dynamic Time Warping) distance for sequences

---

## Question 43
- [ ] Done

**What are the challenges of K-NN for streaming and online learning?**

### Answer

K-NN faces significant challenges in streaming/online settings because it stores all training data and has O(n) prediction complexity. As data streams continuously, memory grows unbounded and predictions slow down. Additionally, data distributions may change over time (concept drift).

**Key Challenges:**

| Challenge | Description |
|-----------|-------------|
| **Memory Growth** | Must store all seen instances → unbounded memory |
| **Slow Prediction** | O(n) distance calculations → slows as n increases |
| **Concept Drift** | Data distribution changes; old data becomes irrelevant |
| **No Model Update** | Traditional K-NN doesn't "learn" incrementally |
| **Imbalanced Arrival** | Classes may arrive unevenly in stream |

**Solutions:**

**1. Sliding Window:**
```python
from collections import deque

class StreamingKNN:
    def __init__(self, k=5, window_size=1000):
        self.k = k
        self.window = deque(maxlen=window_size)  # Fixed memory
    
    def partial_fit(self, X, y):
        self.window.append((X, y))
    
    def predict(self, X_query):
        # Find K nearest in window only
        distances = [np.linalg.norm(X_query - x) for x, y in self.window]
        k_nearest_idx = np.argsort(distances)[:self.k]
        k_labels = [list(self.window)[i][1] for i in k_nearest_idx]
        return Counter(k_labels).most_common(1)[0][0]
```

**2. Instance Selection:** Keep only representative instances (condensed nearest neighbor)

**3. Approximate Methods:** Use LSH or HNSW for O(1) approximate queries

**Interview Tip:** Mention sliding window as the simplest practical solution for streaming K-NN.

---

## Question 44
- [ ] Done

**How do you handle concept drift in K-NN models?**

### Answer

Concept drift occurs when the data distribution changes over time (e.g., customer preferences shift, fraud patterns evolve). K-NN is affected because old stored instances may no longer represent current patterns. Handle drift by forgetting old data or weighting recent data more heavily.

**Types of Concept Drift:**
- **Sudden**: Abrupt change (e.g., new policy)
- **Gradual**: Slow transition over time
- **Recurring**: Patterns that repeat (seasonal)
- **Incremental**: Small continuous changes

**Strategies for K-NN:**

**1. Sliding Window (Time-Based Forgetting):**
```python
# Keep only last N instances or last T time period
window_size = 1000
training_data = training_data[-window_size:]
```

**2. Time-Weighted Distance:**
```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def time_weighted_knn(X_train, y_train, timestamps, X_query, k=5, decay=0.1):
    """More recent instances have higher weight"""
    current_time = timestamps.max()
    time_weights = np.exp(-decay * (current_time - timestamps))
    
    # Compute distances
    distances = np.linalg.norm(X_train - X_query, axis=1)
    
    # Get K nearest
    k_nearest_idx = np.argsort(distances)[:k]
    
    # Weighted voting using time weights
    weighted_votes = {}
    for idx in k_nearest_idx:
        label = y_train[idx]
        weight = time_weights[idx] / distances[idx]
        weighted_votes[label] = weighted_votes.get(label, 0) + weight
    
    return max(weighted_votes, key=weighted_votes.get)
```

**3. Drift Detection + Model Reset:**
- Monitor prediction accuracy
- If accuracy drops significantly, clear old data and rebuild

**Interview Tip:** Sliding window is the most practical; time-weighting is more sophisticated.

---

## Question 45
- [ ] Done

**What is the role of K-NN in semi-supervised learning?**

### Answer

In semi-supervised learning, we have few labeled samples and many unlabeled samples. K-NN helps by propagating labels from labeled to unlabeled points through neighborhood relationships—if an unlabeled point's neighbors are mostly class A, it's likely class A. This leverages the assumption that nearby points share the same label.

**Why K-NN Suits Semi-Supervised Learning:**
- Naturally captures local structure
- No parametric assumptions
- Can propagate labels iteratively

**Approaches:**

**1. Self-Training with K-NN:**
```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def self_training_knn(X_labeled, y_labeled, X_unlabeled, k=5, confidence=0.8):
    knn = KNeighborsClassifier(n_neighbors=k)
    
    while len(X_unlabeled) > 0:
        # Train on current labeled data
        knn.fit(X_labeled, y_labeled)
        
        # Get predictions and probabilities for unlabeled
        probs = knn.predict_proba(X_unlabeled)
        max_probs = probs.max(axis=1)
        predictions = knn.predict(X_unlabeled)
        
        # Add high-confidence predictions to labeled set
        confident_mask = max_probs >= confidence
        if not confident_mask.any():
            break
            
        X_labeled = np.vstack([X_labeled, X_unlabeled[confident_mask]])
        y_labeled = np.concatenate([y_labeled, predictions[confident_mask]])
        X_unlabeled = X_unlabeled[~confident_mask]
    
    return knn.fit(X_labeled, y_labeled)
```

**2. Label Propagation (Graph-Based):**
- Build K-NN graph connecting all points
- Propagate labels through edges
- See Question 46 for implementation

**Key Assumption:** Smoothness assumption—nearby points have same labels.

---

## Question 46
- [ ] Done

**How do you implement label propagation with K-NN?**

### Answer

Label propagation builds a K-NN graph where nodes are all data points (labeled + unlabeled) and edges connect K nearest neighbors. Labels spread iteratively from labeled to unlabeled nodes based on edge weights until convergence.

**Algorithm:**
1. Build K-NN graph (symmetric or mutual)
2. Initialize: labeled points keep their labels; unlabeled get uniform distribution
3. Propagate: Each unlabeled node updates its label distribution based on neighbors
4. Repeat until convergence
5. Clamp: Reset labeled nodes to original labels after each iteration

**Python Implementation:**
```python
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import numpy as np

# -1 indicates unlabeled samples
y_train = np.array([0, 1, -1, -1, -1, 0, -1, 1, -1, -1])

# Label Propagation with K-NN kernel
label_prop = LabelPropagation(
    kernel='knn',
    n_neighbors=5,
    max_iter=1000
)
label_prop.fit(X_train, y_train)

# Get propagated labels for all points
predicted_labels = label_prop.transduction_
```

**Label Spreading (Softer Version):**
```python
# More robust to noise
label_spread = LabelSpreading(
    kernel='knn',
    n_neighbors=7,
    alpha=0.2  # 0.2 = keep 20% of original, 80% from neighbors
)
label_spread.fit(X_train, y_train)
```

**Key Difference:**
- **Label Propagation**: Hard clamping (labeled points never change)
- **Label Spreading**: Soft clamping (allows some change, more robust)

**Interview Tip:** Use `LabelSpreading` for noisy labels, `LabelPropagation` when labels are reliable.

---

## Question 47
- [ ] Done

**What are metric learning techniques for improving K-NN performance?**

### Answer

Metric learning learns a distance function (metric) that brings same-class points closer and pushes different-class points farther apart. Instead of using fixed Euclidean distance, we learn a transformation matrix that optimizes K-NN performance for the specific dataset.

**Why Metric Learning Helps K-NN:**
- Default Euclidean treats all features equally
- Some features may be irrelevant or have wrong scale
- Learned metric adapts to data structure

**Common Techniques:**

| Method | Idea | Loss Function |
|--------|------|---------------|
| **LMNN** | Large Margin NN | Pull same-class, push different-class outside margin |
| **NCA** | Neighborhood Component Analysis | Maximize leave-one-out K-NN accuracy |
| **ITML** | Information Theoretic | Learn Mahalanobis distance with constraints |
| **Siamese Networks** | Neural network pairs | Contrastive loss |

**LMNN (Large Margin Nearest Neighbors):**
```python
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.pipeline import Pipeline

# NCA learns a linear transformation
nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

# Pipeline: Transform then classify
pipeline = Pipeline([
    ('nca', nca),
    ('knn', knn)
])

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
```

**Metric Learning with metric-learn library:**
```python
# pip install metric-learn
from metric_learn import LMNN

lmnn = LMNN(k=3, learn_rate=1e-6)
lmnn.fit(X_train, y_train)
X_train_transformed = lmnn.transform(X_train)
X_test_transformed = lmnn.transform(X_test)

# Now use standard K-NN on transformed space
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_transformed, y_train)
```

**Interview Tip:** NCA is built into sklearn; LMNN requires metric-learn library.

---

## Question 48
- [ ] Done

**How do you learn optimal distance functions for K-NN?**

### Answer

Learning optimal distance functions means finding a transformation (linear or non-linear) such that distances in the transformed space better reflect semantic similarity. The goal: same-class points become closer, different-class points become farther apart.

**Approaches:**

**1. Linear Transformation (Mahalanobis Distance):**
Learn matrix M such that: $d(x, y) = \sqrt{(x-y)^T M (x-y)}$

```python
from sklearn.neighbors import NeighborhoodComponentsAnalysis

# NCA learns optimal linear transformation
nca = NeighborhoodComponentsAnalysis(n_components=None)
nca.fit(X_train, y_train)

# Transform data
X_transformed = nca.transform(X_train)
# Now Euclidean distance in transformed space = learned metric
```

**2. Contrastive Learning (Siamese Networks):**
```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def create_siamese_network(input_shape):
    # Shared encoder
    input_layer = layers.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(input_layer)
    x = layers.Dense(64, activation='relu')(x)
    embedding = layers.Dense(32)(x)  # Embedding space
    
    return Model(input_layer, embedding)

# Train with contrastive loss
# Similar pairs → distance close to 0
# Dissimilar pairs → distance > margin
```

**3. Triplet Loss:**
- Anchor (a), Positive (p), Negative (n)
- Loss: $\max(0, d(a,p) - d(a,n) + \text{margin})$

**When to Use:**
- When default Euclidean performs poorly
- When you have enough labeled data for learning
- When feature importance varies significantly

**Interview Tip:** NCA is simplest; Siamese/Triplet networks for complex data (images, text).

---

## Question 49
- [ ] Done

**What is the Mahalanobis distance and its application in K-NN?**

### Answer

Mahalanobis distance accounts for feature correlations and different variances by using the inverse covariance matrix. It measures distance in terms of "how many standard deviations away" rather than raw units. For K-NN, it handles correlated features better than Euclidean.

**Formula:**
$$d_{Mahalanobis}(x, y) = \sqrt{(x - y)^T \Sigma^{-1} (x - y)}$$

Where $\Sigma$ is the covariance matrix.

**Why It Matters:**
- Euclidean assumes features are independent and equally scaled
- Mahalanobis handles:
  - Different feature scales (no need for StandardScaler)
  - Correlated features (adjusts for redundancy)

**Visual Intuition:**
- Euclidean: Circular distance contours
- Mahalanobis: Elliptical contours aligned with data distribution

**Python Implementation:**
```python
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import KNeighborsClassifier

# Compute covariance matrix from training data
cov_matrix = np.cov(X_train.T)
inv_cov = np.linalg.inv(cov_matrix)

# Custom Mahalanobis distance function
def mahalanobis_distance(x, y):
    diff = x - y
    return np.sqrt(diff @ inv_cov @ diff)

# Use with sklearn (pass precomputed distances or use callable)
# Alternative: Transform data first
from scipy.linalg import sqrtm
transform_matrix = sqrtm(inv_cov)
X_train_transformed = X_train @ transform_matrix
X_test_transformed = X_test @ transform_matrix

# Now Euclidean on transformed = Mahalanobis on original
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_transformed, y_train)
```

**When to Use:**
- Features have different scales and correlations
- Data is multivariate normal
- You want automatic handling of correlations

**Limitation:** Requires invertible covariance matrix (fails if n_samples < n_features).

---

## Question 50
- [ ] Done

**How do you implement K-NN for multi-label classification problems?**

### Answer

Multi-label classification means each sample can belong to multiple classes simultaneously (e.g., a movie tagged as "Comedy" AND "Romance"). K-NN adapts by collecting all labels from K neighbors and using various strategies to determine output labels.

**Approaches:**

**1. MLkNN (Multi-Label K-NN):**
- For each label, estimate probability based on K neighbors
- Use Bayesian approach: P(label | count of neighbors with that label)

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
import numpy as np

# Method 1: Multi-Output Wrapper
# y_train shape: (n_samples, n_labels) - binary matrix
knn = KNeighborsClassifier(n_neighbors=5)
multi_knn = MultiOutputClassifier(knn)
multi_knn.fit(X_train, y_train)
predictions = multi_knn.predict(X_test)  # Shape: (n_samples, n_labels)
```

**2. Using skmultilearn Library:**
```python
# pip install scikit-multilearn
from skmultilearn.adapt import MLkNN

mlknn = MLkNN(k=5, s=1.0)  # s = smoothing parameter
mlknn.fit(X_train, y_train)
predictions = mlknn.predict(X_test)
```

**3. Manual Implementation:**
```python
def multi_label_knn(X_train, y_train, X_query, k=5, threshold=0.5):
    """
    y_train: binary matrix (n_samples, n_labels)
    Returns: binary vector of predicted labels
    """
    distances = np.linalg.norm(X_train - X_query, axis=1)
    k_nearest_idx = np.argsort(distances)[:k]
    
    # Average labels of K neighbors
    neighbor_labels = y_train[k_nearest_idx]
    label_probs = neighbor_labels.mean(axis=0)
    
    # Threshold to get final labels
    return (label_probs >= threshold).astype(int)
```

**Evaluation Metrics:** Hamming loss, subset accuracy, micro/macro F1.

---

## Question 51
- [ ] Done

**What are fuzzy K-NN algorithms and their advantages?**

### Answer

Fuzzy K-NN assigns class membership degrees (probabilities) rather than hard labels. Each training point has fuzzy membership to each class, and predictions are based on weighted membership from K neighbors. This provides soft predictions and handles overlapping class boundaries better.

**Difference from Standard K-NN:**
- Standard: Each point belongs to exactly one class
- Fuzzy: Each point has membership degree to all classes (sum = 1)

**Fuzzy K-NN Algorithm:**
1. Initialize memberships (1 for true class, or calculate from neighborhood)
2. For query point, find K neighbors
3. Compute membership to each class as weighted sum of neighbor memberships

**Formula:**
$$u_i(x) = \frac{\sum_{j=1}^{K} u_{ij} \cdot (1/d_j^{2/(m-1)})}{\sum_{j=1}^{K} (1/d_j^{2/(m-1)})}$$

Where:
- $u_i(x)$ = membership of x to class i
- $u_{ij}$ = membership of neighbor j to class i
- $d_j$ = distance to neighbor j
- $m$ = fuzziness parameter (typically 2)

**Python Implementation:**
```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

class FuzzyKNN:
    def __init__(self, k=5, m=2):
        self.k = k
        self.m = m  # Fuzziness parameter
    
    def fit(self, X, y):
        self.X_train = X
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        # Initialize memberships (crisp: 1 for true class)
        self.memberships = np.zeros((len(y), self.n_classes))
        for i, c in enumerate(self.classes):
            self.memberships[y == c, i] = 1.0
        
        self.nn = NearestNeighbors(n_neighbors=self.k)
        self.nn.fit(X)
    
    def predict_proba(self, X):
        distances, indices = self.nn.kneighbors(X)
        distances = np.clip(distances, 1e-10, None)  # Avoid div by 0
        
        # Weight = 1 / d^(2/(m-1))
        weights = 1 / (distances ** (2 / (self.m - 1)))
        
        proba = np.zeros((len(X), self.n_classes))
        for i in range(len(X)):
            neighbor_memberships = self.memberships[indices[i]]
            weighted_sum = (neighbor_memberships.T * weights[i]).sum(axis=1)
            proba[i] = weighted_sum / weights[i].sum()
        
        return proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes[proba.argmax(axis=1)]
```

**Advantages:**
- Provides probability estimates naturally
- Handles overlapping classes better
- More robust to noisy labels

---

## Question 52
- [ ] Done

**How do you handle uncertainty quantification in K-NN predictions?**

### Answer

K-NN naturally provides uncertainty through the distribution of neighbor labels. If 5/5 neighbors are class A, high confidence. If 3/5 are class A and 2/5 are class B, lower confidence. Quantify uncertainty using class proportions, distance-weighted votes, or entropy.

**Methods for Uncertainty Quantification:**

**1. Class Proportion (Simplest):**
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# predict_proba gives class proportions among K neighbors
probabilities = knn.predict_proba(X_test)
# High max probability = high confidence
confidence = probabilities.max(axis=1)
uncertainty = 1 - confidence
```

**2. Entropy-Based Uncertainty:**
```python
import numpy as np
from scipy.stats import entropy

def knn_entropy_uncertainty(knn, X):
    proba = knn.predict_proba(X)
    # Entropy: higher = more uncertain
    uncertainties = entropy(proba.T)  
    return uncertainties

# Normalize to [0, 1]
max_entropy = np.log(len(knn.classes_))
normalized_uncertainty = uncertainties / max_entropy
```

**3. Distance-Based Uncertainty:**
```python
from sklearn.neighbors import NearestNeighbors

def distance_uncertainty(X_train, X_query, k=5):
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_train)
    distances, _ = nn.kneighbors(X_query)
    
    # Mean distance to neighbors: higher = more uncertain
    mean_distances = distances.mean(axis=1)
    return mean_distances
```

**4. Combining Both:**
```python
def combined_uncertainty(knn, X_train, X_query):
    # Label uncertainty (entropy)
    proba = knn.predict_proba(X_query)
    label_uncertainty = entropy(proba.T) / np.log(len(knn.classes_))
    
    # Distance uncertainty
    nn = NearestNeighbors(n_neighbors=knn.n_neighbors)
    nn.fit(X_train)
    distances, _ = nn.kneighbors(X_query)
    dist_uncertainty = distances.mean(axis=1)
    dist_uncertainty = (dist_uncertainty - dist_uncertainty.min()) / (dist_uncertainty.max() - dist_uncertainty.min())
    
    # Combine
    return 0.5 * label_uncertainty + 0.5 * dist_uncertainty
```

**Use Cases:**
- Flag low-confidence predictions for human review
- Selective prediction (abstain when uncertain)
- Active learning (query most uncertain points)

---

## Question 53
- [ ] Done

**What is the role of K-NN in instance-based learning and case-based reasoning?**

### Answer

K-NN is the foundational algorithm for instance-based learning (IBL), where models store training instances and generalize at prediction time. Case-based reasoning (CBR) extends this to complex problem-solving by retrieving, adapting, and reusing past solutions (cases) for new problems.

**Instance-Based Learning Characteristics:**
- **Lazy learning**: No training phase; all work at prediction time
- **Memory-based**: Stores all (or selected) training instances
- **Local approximation**: Decision based on local neighborhood

**K-NN as Core IBL Algorithm:**
```
Instance-Based Learning Family:
├── K-NN (basic)
├── Weighted K-NN
├── Locally Weighted Learning
├── Radius Neighbors
└── Condensed/Edited NN
```

**Case-Based Reasoning (CBR) Cycle:**
1. **Retrieve**: Find similar past cases (K-NN step)
2. **Reuse**: Apply past solution to new problem
3. **Revise**: Adapt solution if needed
4. **Retain**: Store new case for future use

**Example: Help Desk CBR System**
```python
# Case = (problem_features, solution)
# K-NN retrieves similar past problems

def cbr_help_desk(problem_features, case_base, k=3):
    """
    problem_features: vector describing customer issue
    case_base: [(features, solution), ...]
    """
    # RETRIEVE: Find K similar cases
    X_cases = np.array([c[0] for c in case_base])
    distances = np.linalg.norm(X_cases - problem_features, axis=1)
    k_nearest_idx = np.argsort(distances)[:k]
    
    # REUSE: Return solutions from similar cases
    similar_solutions = [case_base[i][1] for i in k_nearest_idx]
    
    return similar_solutions  # Human reviews and adapts

# Usage
case_base = [
    ([1, 0, 1], "Reset password via email"),
    ([0, 1, 1], "Check payment method"),
    ([1, 1, 0], "Clear browser cache"),
]
solutions = cbr_help_desk([1, 0, 0], case_base, k=2)
```

**Key Difference:**
- **K-NN**: Direct prediction (majority vote)
- **CBR**: Retrieve + Adapt (human or rule-based adaptation)

---

## Question 54
- [ ] Done

**How do you implement K-NN for text classification and NLP tasks?**

### Answer

For text classification, convert text documents to numerical vectors using TF-IDF or embeddings, then apply K-NN with cosine similarity. The key is proper text preprocessing and choosing the right representation.

**Pipeline:**
1. Text preprocessing (cleaning, tokenization)
2. Vectorization (TF-IDF, embeddings)
3. K-NN with appropriate metric (cosine)

**Method 1: TF-IDF + K-NN**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Sample data
texts = ["I love this movie", "Terrible film", "Great acting", "Boring plot"]
labels = [1, 0, 1, 0]  # 1=positive, 0=negative

# Create pipeline
text_knn = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2)  # Unigrams and bigrams
    )),
    ('knn', KNeighborsClassifier(
        n_neighbors=5,
        metric='cosine'  # Crucial for text!
    ))
])

text_knn.fit(texts, labels)
predictions = text_knn.predict(["Amazing movie!"])
```

**Method 2: Word Embeddings + K-NN**
```python
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier

# Use pre-trained sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert texts to embeddings
train_embeddings = model.encode(train_texts)
test_embeddings = model.encode(test_texts)

# K-NN on embeddings
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn.fit(train_embeddings, train_labels)
predictions = knn.predict(test_embeddings)
```

**Best Practices:**
- Always use **cosine similarity** for text (not Euclidean)
- TF-IDF for traditional NLP; embeddings for semantic tasks
- Tune max_features to control dimensionality

---

## Question 55
- [ ] Done

**What are the considerations for K-NN in image recognition and computer vision?**

### Answer

Raw pixel K-NN fails due to curse of dimensionality (images = very high dimensions). Success requires: (1) feature extraction using CNNs, (2) dimensionality reduction, (3) approximate NN for scalability.

**Key Considerations:**

| Consideration | Problem | Solution |
|---------------|---------|----------|
| High Dimensionality | 224×224×3 = 150K features | CNN feature extraction |
| Semantic Gap | Similar pixels ≠ similar content | Learned representations |
| Scalability | Millions of images | ANN (FAISS, Annoy) |
| Translation Invariance | Shifted images look different | CNN features handle this |

**Recommended Pipeline:**
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 1. Load pre-trained CNN (remove classification head)
feature_extractor = ResNet50(
    weights='imagenet',
    include_top=False,
    pooling='avg'  # Global average pooling → 2048-dim vector
)

# 2. Extract features
def extract_features(images):
    preprocessed = preprocess_input(images)
    return feature_extractor.predict(preprocessed)

train_features = extract_features(train_images)  # Shape: (n, 2048)
test_features = extract_features(test_images)

# 3. K-NN on CNN features
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn.fit(train_features, train_labels)
predictions = knn.predict(test_features)
```

**For Large-Scale Image Search (FAISS):**
```python
import faiss

# Build index for fast search
d = train_features.shape[1]  # 2048
index = faiss.IndexFlatIP(d)  # Inner product (for cosine, normalize first)

# Normalize for cosine similarity
faiss.normalize_L2(train_features)
index.add(train_features)

# Search
faiss.normalize_L2(query_features)
distances, indices = index.search(query_features, k=5)
```

**Interview Tip:** Always extract CNN features first; never use raw pixels with K-NN.

---

## Question 56
- [ ] Done

**How do you use K-NN for feature selection and wrapper methods?**

### Answer

K-NN can be used as the classifier inside a wrapper-based feature selection method. The wrapper evaluates feature subsets by training K-NN on each subset and measuring performance. Features that improve K-NN accuracy are kept; irrelevant features are removed.

**Wrapper Method Process:**
1. Start with all features or empty set
2. Add/remove features iteratively
3. Train K-NN on current feature subset
4. Evaluate using cross-validation
5. Keep subset with best performance

**Forward Selection with K-NN:**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def forward_selection_knn(X, y, k=5, max_features=None):
    n_features = X.shape[1]
    selected = []
    remaining = list(range(n_features))
    best_score = 0
    
    if max_features is None:
        max_features = n_features
    
    while remaining and len(selected) < max_features:
        scores = []
        for f in remaining:
            current_features = selected + [f]
            knn = KNeighborsClassifier(n_neighbors=k)
            score = cross_val_score(
                knn, X[:, current_features], y, cv=5
            ).mean()
            scores.append((f, score))
        
        best_feature, score = max(scores, key=lambda x: x[1])
        
        if score > best_score:
            best_score = score
            selected.append(best_feature)
            remaining.remove(best_feature)
        else:
            break  # No improvement
    
    return selected, best_score

selected_features, accuracy = forward_selection_knn(X, y, k=5)
print(f"Selected features: {selected_features}, Accuracy: {accuracy:.4f}")
```

**Using sklearn's SequentialFeatureSelector:**
```python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
sfs = SequentialFeatureSelector(
    knn,
    n_features_to_select=5,
    direction='forward',  # or 'backward'
    cv=5
)
sfs.fit(X, y)
selected_mask = sfs.get_support()
X_selected = X[:, selected_mask]
```

**Why K-NN for Feature Selection:**
- Sensitive to irrelevant features (good indicator)
- Non-parametric (no assumptions)
- Simple and interpretable wrapper

---

## Question 57
- [ ] Done

**What is the relationship between K-NN and kernel methods?**

### Answer

K-NN and kernel methods are related through the concept of local weighting. K-NN can be viewed as using a "K-nearest kernel"—a kernel that is 1 for K nearest points and 0 elsewhere. Kernel regression generalizes this with smooth weighting based on distance.

**Connection:**

| Aspect | K-NN | Kernel Methods |
|--------|------|----------------|
| Weighting | Uniform for K nearest, 0 elsewhere | Smooth, distance-based decay |
| Kernel Shape | "Box" kernel (discontinuous) | Gaussian, Epanechnikov (smooth) |
| Bandwidth | Adaptive (K determines reach) | Fixed (bandwidth parameter h) |

**K-NN as a Kernel:**
- K-NN uses: $K(x, x_i) = 1$ if $x_i$ in K-nearest, else 0
- Kernel regression uses: $K(x, x_i) = \exp(-||x - x_i||^2 / 2h^2)$

**Kernel-Weighted K-NN (Nadaraya-Watson):**
```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def kernel_weighted_knn(X_train, y_train, X_query, k=5, bandwidth=1.0):
    """Combine K-NN with Gaussian kernel weighting"""
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_train)
    distances, indices = nn.kneighbors(X_query)
    
    predictions = []
    for i in range(len(X_query)):
        # Gaussian kernel weights
        weights = np.exp(-distances[i]**2 / (2 * bandwidth**2))
        weights /= weights.sum()
        
        # Weighted average (regression) or vote (classification)
        pred = np.average(y_train[indices[i]], weights=weights)
        predictions.append(pred)
    
    return np.array(predictions)
```

**sklearn Kernel Density + Classification:**
```python
from sklearn.neighbors import KernelDensity, KNeighborsClassifier

# KDE estimates density, can be used for classification
kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
kde.fit(X_class_0)
log_density = kde.score_samples(X_test)
```

**Key Insight:**
- K-NN: Adaptive bandwidth (reaches until K points found)
- Kernel methods: Fixed bandwidth (may include 0 or 100 points)

---

## Question 58
- [ ] Done

**How do you implement K-NN for graph-structured data?**

### Answer

For graph-structured data (molecules, social networks), standard Euclidean distance doesn't work. Use graph kernels or graph neural network embeddings to measure similarity between graphs, then apply K-NN on these representations.

**Approaches:**

**1. Graph Kernels (Traditional):**
- Weisfeiler-Lehman kernel
- Random walk kernel
- Graphlet kernel

**2. Graph Neural Network Embeddings (Modern):**
- Use GNN to embed graphs as vectors
- Apply standard K-NN on embeddings

**Using grakel for Graph Kernels:**
```python
# pip install grakel
from grakel import GraphKernel
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# graphs = list of (adjacency_matrix, node_labels)
# y = labels for each graph

# Weisfeiler-Lehman kernel
wl_kernel = GraphKernel(kernel="WL", normalize=True)
K = wl_kernel.fit_transform(graphs)  # Kernel matrix

# K-NN with precomputed kernel (convert to distance)
# For normalized kernels: d(x,y)² = K(x,x) + K(y,y) - 2K(x,y) = 2 - 2K(x,y)
distances = np.sqrt(2 - 2*K)

knn = KNeighborsClassifier(n_neighbors=5, metric='precomputed')
knn.fit(distances, y)
```

**Using GNN Embeddings:**
```python
# Conceptual example with PyTorch Geometric
import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.neighbors import KNeighborsClassifier

class GraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)  # Graph-level embedding

# Extract embeddings for all graphs
embeddings = encoder(data.x, data.edge_index, data.batch).detach().numpy()

# K-NN on graph embeddings
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(embeddings, labels)
```

**Applications:**
- Molecule similarity (drug discovery)
- Social network analysis
- Code analysis (AST graphs)

---

## Question 59
- [ ] Done

**What are the privacy-preserving techniques for K-NN algorithms?**

### Answer

Privacy-preserving K-NN allows finding nearest neighbors without exposing raw data. Key techniques include secure computation protocols (homomorphic encryption, secure multi-party computation) and differential privacy (adding noise to protect individual records).

**Privacy Concerns:**
- K-NN stores all training data
- Neighbors reveal information about training points
- Membership inference attacks possible

**Techniques:**

**1. Differential Privacy:**
Add noise to distances or results to prevent identifying specific training points.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def dp_knn(X_train, y_train, X_query, k=5, epsilon=1.0):
    """Differentially private K-NN using exponential mechanism"""
    nn = NearestNeighbors(n_neighbors=len(X_train))
    nn.fit(X_train)
    distances, indices = nn.kneighbors(X_query)
    
    predictions = []
    for i in range(len(X_query)):
        # Add Laplace noise to distances
        noisy_distances = distances[i] + np.random.laplace(0, 1/epsilon, len(distances[i]))
        
        # Select K nearest based on noisy distances
        k_nearest = np.argsort(noisy_distances)[:k]
        labels = y_train[indices[i][k_nearest]]
        
        # Majority vote
        values, counts = np.unique(labels, return_counts=True)
        predictions.append(values[np.argmax(counts)])
    
    return np.array(predictions)
```

**2. Secure Multi-Party Computation (Conceptual):**
- Data split across multiple parties
- Compute distances without revealing data
- Uses cryptographic protocols

**3. Homomorphic Encryption:**
- Encrypt data; compute on encrypted values
- Slow but mathematically secure

**4. Federated K-NN:**
- Keep data on local devices
- Share only aggregated results

**Trade-offs:**
| Technique | Privacy | Accuracy | Speed |
|-----------|---------|----------|-------|
| Differential Privacy | Medium | Reduced | Fast |
| Homomorphic Encryption | High | Exact | Very Slow |
| Secure MPC | High | Exact | Slow |

---

## Question 60
- [ ] Done

**How do you implement secure K-NN computation in federated learning?**

### Answer

Federated K-NN keeps data on local devices/servers. Each node computes local distances to a query, and only K nearest candidates (with distances) are sent to a coordinator. The coordinator selects global K nearest without seeing raw features.

**Federated K-NN Architecture:**
```
Query → Coordinator → Broadcasts to all nodes
                   ↓
        Node 1: Local K-NN → Top K candidates
        Node 2: Local K-NN → Top K candidates
        Node 3: Local K-NN → Top K candidates
                   ↓
Coordinator: Aggregate and select global top K
                   ↓
              Final prediction
```

**Implementation:**
```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

class FederatedKNNNode:
    """Each node holds its local data"""
    def __init__(self, X_local, y_local, k=5):
        self.X = X_local
        self.y = y_local
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=min(k, len(X_local)))
        self.nn.fit(X_local)
    
    def get_local_candidates(self, X_query):
        """Return local top-K candidates (distances, labels only)"""
        distances, indices = self.nn.kneighbors(X_query)
        labels = self.y[indices]
        return distances, labels  # Don't send raw features!

class FederatedKNNCoordinator:
    def __init__(self, nodes, k=5):
        self.nodes = nodes
        self.k = k
    
    def predict(self, X_query):
        all_distances = []
        all_labels = []
        
        # Collect candidates from all nodes
        for node in self.nodes:
            distances, labels = node.get_local_candidates(X_query)
            all_distances.append(distances)
            all_labels.append(labels)
        
        # Aggregate
        all_distances = np.hstack(all_distances)
        all_labels = np.hstack(all_labels)
        
        # Select global top K
        predictions = []
        for i in range(len(X_query)):
            k_nearest_idx = np.argsort(all_distances[i])[:self.k]
            k_labels = all_labels[i][k_nearest_idx]
            values, counts = np.unique(k_labels, return_counts=True)
            predictions.append(values[np.argmax(counts)])
        
        return np.array(predictions)

# Usage
nodes = [
    FederatedKNNNode(X_node1, y_node1, k=5),
    FederatedKNNNode(X_node2, y_node2, k=5),
    FederatedKNNNode(X_node3, y_node3, k=5),
]
coordinator = FederatedKNNCoordinator(nodes, k=5)
predictions = coordinator.predict(X_query)
```

**Privacy Considerations:**
- Only distances and labels are shared
- Raw features stay on local nodes
- Can add differential privacy noise to distances

---

## Question 61
- [ ] Done

**What are approximate nearest neighbor algorithms and their trade-offs?**

### Answer

Approximate Nearest Neighbor (ANN) algorithms sacrifice exact accuracy for massive speed improvements. Instead of guaranteeing the true K nearest neighbors, they find neighbors that are "close enough" with high probability. Essential for large-scale applications (millions+ points).

**Common ANN Algorithms:**

| Algorithm | Method | Best For |
|-----------|--------|----------|
| **LSH** | Hash similar items to same bucket | High dimensions |
| **HNSW** | Hierarchical navigable graphs | General purpose, fast |
| **Annoy** | Random projection trees | Read-heavy workloads |
| **IVF** | Cluster-based search | Very large datasets |
| **PQ** | Product quantization | Memory-limited |

**Trade-offs:**

| Metric | Exact K-NN | ANN |
|--------|-----------|-----|
| **Accuracy** | 100% recall | 90-99% typical |
| **Speed** | O(n) | O(log n) or O(1) |
| **Memory** | O(n×d) | Often compressed |
| **Index Build** | None | Can be slow |

**FAISS Example:**
```python
import faiss
import numpy as np

# Data: 1M vectors, 128 dimensions
n, d = 1_000_000, 128
X = np.random.random((n, d)).astype('float32')

# Exact K-NN (slow)
index_flat = faiss.IndexFlatL2(d)
index_flat.add(X)

# ANN with IVF (fast, approximate)
nlist = 100  # Number of clusters
quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
index_ivf.train(X)
index_ivf.add(X)
index_ivf.nprobe = 10  # Search 10 clusters (trade-off parameter)

# Query
query = np.random.random((1, d)).astype('float32')
distances, indices = index_ivf.search(query, k=5)
```

**Recall vs Speed Trade-off:**
- Higher `nprobe` → better recall, slower
- Lower `nprobe` → worse recall, faster
- Typical: 95% recall at 100× speedup

---

## Question 62
- [ ] Done

**How do you evaluate and validate K-NN model performance?**

### Answer

Evaluate K-NN using standard classification/regression metrics with cross-validation. Key: use stratified K-fold for classification, never use the same data for training and testing (K-NN memorizes!).

**Classification Metrics:**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

knn = KNeighborsClassifier(n_neighbors=5)

# Cross-validation (ALWAYS use for K-NN)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# After final model selection
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
```

**Regression Metrics:**
```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)
y_pred = knn_reg.predict(X_test)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

**Validation Curve for K Selection:**
```python
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

k_range = range(1, 31)
train_scores, test_scores = validation_curve(
    KNeighborsClassifier(), X, y,
    param_name='n_neighbors',
    param_range=k_range, cv=5
)

plt.plot(k_range, train_scores.mean(axis=1), label='Training')
plt.plot(k_range, test_scores.mean(axis=1), label='Validation')
plt.xlabel('K'); plt.ylabel('Accuracy')
plt.legend(); plt.show()
```

**Key Point:** Always validate K-NN with held-out data or CV—never report training accuracy (it's meaningless for K-NN).

---

## Question 63
- [ ] Done

**What are the interpretability aspects of K-NN algorithms?**

### Answer

K-NN is inherently interpretable because predictions come directly from training examples. You can explain any prediction by showing "these are the K most similar cases and their outcomes." This local, example-based explanation is intuitive for non-technical users.

**Interpretability Advantages:**

| Aspect | K-NN Interpretability |
|--------|----------------------|
| **Local explanations** | Show actual similar cases |
| **No hidden logic** | Distance → neighbors → vote |
| **Debugging** | Inspect neighbors for errors |
| **Trust building** | Users see evidence |

**Explaining a Prediction:**
```python
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Fit model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Explain a single prediction
def explain_prediction(knn, X_train, y_train, x_query, feature_names=None):
    distances, indices = knn.kneighbors([x_query])
    
    print("Prediction:", knn.predict([x_query])[0])
    print("\nEvidence (5 most similar cases):")
    print("-" * 50)
    
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        print(f"\nNeighbor {i+1}: Distance = {dist:.4f}")
        print(f"  Label: {y_train[idx]}")
        if feature_names:
            for j, name in enumerate(feature_names):
                print(f"  {name}: {X_train[idx][j]:.2f}")

# Usage
explain_prediction(knn, X_train, y_train, X_test[0], 
                   feature_names=['age', 'income', 'score'])
```

**Output Example:**
```
Prediction: 1

Evidence (5 most similar cases):
--------------------------------------------------
Neighbor 1: Distance = 0.2341
  Label: 1
  age: 35.00, income: 55000.00, score: 720.00

Neighbor 2: Distance = 0.3012
  Label: 1
  ...
```

**Interview Tip:** K-NN is one of the most interpretable ML algorithms—explain predictions by showing neighbors.

---

## Question 64
- [ ] Done

**How do you explain K-NN predictions and decision boundaries?**

### Answer

Explain K-NN predictions by: (1) showing the K nearest neighbors and their labels, (2) displaying the voting process, (3) visualizing decision boundaries (2D/3D). Decision boundaries in K-NN are Voronoi-like regions that follow local data patterns.

**1. Neighbor-Based Explanation:**
```python
def detailed_explanation(knn, X_train, y_train, x_query):
    distances, indices = knn.kneighbors([x_query])
    neighbor_labels = y_train[indices[0]]
    
    # Count votes
    from collections import Counter
    votes = Counter(neighbor_labels)
    
    print("Query point neighbors:")
    for i, (idx, dist, label) in enumerate(zip(indices[0], distances[0], neighbor_labels)):
        print(f"  {i+1}. Label={label}, Distance={dist:.3f}")
    
    print(f"\nVote counts: {dict(votes)}")
    print(f"Prediction: {votes.most_common(1)[0][0]}")
    print(f"Confidence: {votes.most_common(1)[0][1] / len(neighbor_labels):.0%}")
```

**2. Decision Boundary Visualization:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def plot_decision_boundary(X, y, k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    
    # Create mesh grid
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict for each point in mesh
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    plt.title(f'K-NN Decision Boundary (K={k})')
    plt.show()

plot_decision_boundary(X_train, y_train, k=5)
```

**3. Boundary Characteristics:**
- **K=1**: Voronoi tessellation (very complex)
- **Larger K**: Smoother boundaries
- Boundaries follow local data density

**Interview Tip:** Unlike linear models with global boundaries, K-NN creates local, data-driven boundaries.

---

## Question 65
- [ ] Done

**What is the role of K-NN in active learning and query strategies?**

### Answer

In active learning, K-NN helps identify which unlabeled points to query (label) next. The idea: query points where K-NN is most uncertain (neighbors disagree) or points in sparse regions. This maximizes learning from minimal labels.

**Active Learning with K-NN:**

**1. Uncertainty Sampling:**
Query points where neighbor labels are most mixed.

```python
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import entropy
import numpy as np

def uncertainty_sampling_knn(X_labeled, y_labeled, X_unlabeled, k=5, n_queries=10):
    """Select most uncertain unlabeled points for labeling"""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_labeled, y_labeled)
    
    # Get prediction probabilities for unlabeled
    probas = knn.predict_proba(X_unlabeled)
    
    # Calculate uncertainty (entropy)
    uncertainties = entropy(probas.T)
    
    # Select top-N most uncertain
    query_indices = np.argsort(uncertainties)[-n_queries:]
    return query_indices

# Usage
queries_idx = uncertainty_sampling_knn(X_labeled, y_labeled, X_pool)
# Human labels X_pool[queries_idx], add to training set
```

**2. Density-Weighted Uncertainty:**
Prefer uncertain points in dense regions (more representative).

```python
from sklearn.neighbors import NearestNeighbors

def density_weighted_sampling(X_labeled, y_labeled, X_unlabeled, k=5, n_queries=10):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_labeled, y_labeled)
    
    # Uncertainty
    probas = knn.predict_proba(X_unlabeled)
    uncertainties = entropy(probas.T)
    
    # Density (inverse of average distance to neighbors)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_unlabeled)
    distances, _ = nn.kneighbors(X_unlabeled)
    densities = 1 / (distances.mean(axis=1) + 1e-10)
    
    # Combined score
    scores = uncertainties * densities
    
    return np.argsort(scores)[-n_queries:]
```

**Why K-NN for Active Learning:**
- Natural uncertainty measure (neighbor label variance)
- No retraining needed (just add points)
- Works with any distance metric

---

## Question 66
- [ ] Done

**How do you implement K-NN for hierarchical classification?**

### Answer

Hierarchical classification organizes classes in a tree structure (e.g., Animal → Mammal → Dog). K-NN can classify hierarchically by: (1) top-down approach (classify at each level), (2) flat approach with hierarchical loss, or (3) using hierarchy-aware distances.

**Hierarchical Structure Example:**
```
Animal
├── Mammal
│   ├── Dog
│   └── Cat
└── Bird
    ├── Eagle
    └── Sparrow
```

**Top-Down K-NN:**
```python
from sklearn.neighbors import KNeighborsClassifier

class HierarchicalKNN:
    def __init__(self, hierarchy, k=5):
        """
        hierarchy: dict mapping parent → list of children
        e.g., {'root': ['Animal'], 'Animal': ['Mammal', 'Bird'], ...}
        """
        self.hierarchy = hierarchy
        self.k = k
        self.classifiers = {}  # One K-NN per internal node
    
    def fit(self, X, y_hierarchical):
        """y_hierarchical: list of paths like ['Animal', 'Mammal', 'Dog']"""
        # Train classifier for each internal node
        for parent, children in self.hierarchy.items():
            if len(children) <= 1:
                continue
            
            # Get samples belonging to this subtree
            mask = [parent in path for path in y_hierarchical]
            X_sub = X[mask]
            
            # Labels are direct children
            y_sub = [self._get_child(path, parent) for path in y_hierarchical if parent in path]
            
            self.classifiers[parent] = KNeighborsClassifier(n_neighbors=self.k)
            self.classifiers[parent].fit(X_sub, y_sub)
    
    def _get_child(self, path, parent):
        idx = path.index(parent)
        return path[idx + 1] if idx + 1 < len(path) else path[-1]
    
    def predict(self, X):
        predictions = []
        for x in X:
            path = ['root']
            node = 'root'
            while node in self.classifiers:
                child = self.classifiers[node].predict([x])[0]
                path.append(child)
                node = child
            predictions.append(path)
        return predictions
```

**Simpler Flat Approach:**
```python
# Classify at leaf level, infer hierarchy
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_leaf_labels)  # y = ['Dog', 'Cat', 'Eagle', ...]
prediction = knn.predict(X_test)

# Map leaf to full path
leaf_to_path = {
    'Dog': ['Animal', 'Mammal', 'Dog'],
    'Cat': ['Animal', 'Mammal', 'Cat'],
    'Eagle': ['Animal', 'Bird', 'Eagle'],
}
full_path = leaf_to_path[prediction[0]]
```

**Interview Tip:** Top-down is more consistent with hierarchy but may propagate errors.

---

## Question 67
- [ ] Done

**What are the considerations for K-NN in real-time and edge computing?**

### Answer

Real-time and edge computing require K-NN to be fast and memory-efficient on limited hardware. Key considerations: reduce dataset size, use approximate methods, optimize data structures, and consider model compression.

**Challenges:**
| Challenge | Edge/Real-Time Impact |
|-----------|----------------------|
| Memory | Limited RAM on devices |
| Latency | Must respond in milliseconds |
| Power | Battery constraints |
| Updates | Need to handle new data |

**Optimization Strategies:**

**1. Instance Selection (Reduce Dataset):**
```python
from sklearn.neighbors import KNeighborsClassifier

def condensed_nearest_neighbor(X, y):
    """Keep only boundary points"""
    store_X, store_y = [X[0]], [y[0]]
    
    for i in range(1, len(X)):
        # Classify with current store
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(np.array(store_X), np.array(store_y))
        pred = knn.predict([X[i]])[0]
        
        # Add if misclassified (boundary point)
        if pred != y[i]:
            store_X.append(X[i])
            store_y.append(y[i])
    
    return np.array(store_X), np.array(store_y)

X_condensed, y_condensed = condensed_nearest_neighbor(X, y)
print(f"Reduced from {len(X)} to {len(X_condensed)} points")
```

**2. Use Efficient Data Structures:**
```python
# Ball tree is memory-efficient
from sklearn.neighbors import BallTree

tree = BallTree(X_train, leaf_size=40)  # Adjust leaf_size for speed/memory
distances, indices = tree.query(X_query, k=5)
```

**3. Quantization:**
```python
import numpy as np

# Reduce precision (32-bit → 16-bit)
X_train_quantized = X_train.astype(np.float16)
# Even smaller: 8-bit integers
X_train_int8 = ((X_train - X_train.min()) / (X_train.max() - X_train.min()) * 255).astype(np.uint8)
```

**4. Approximate Methods (HNSW for fast queries):**
```python
import hnswlib

# Build index (done offline)
index = hnswlib.Index(space='l2', dim=d)
index.init_index(max_elements=n, ef_construction=200, M=16)
index.add_items(X_train)

# Fast query (on device)
index.set_ef(50)  # Trade-off: lower = faster, less accurate
labels, distances = index.knn_query(X_query, k=5)
```

---

## Question 68
- [ ] Done

**How do you optimize K-NN for mobile and IoT devices?**

### Answer

Mobile/IoT optimization focuses on minimizing memory footprint, inference time, and power consumption. Techniques: prototype selection, dimensionality reduction, quantization, and using lightweight ANN libraries.

**Optimization Pipeline:**
```
Full Dataset → Instance Selection → Dimensionality Reduction → Quantization → Lightweight Index
```

**1. Prototype Selection (KMeans Centroids):**
```python
from sklearn.cluster import KMeans
import numpy as np

def create_prototypes(X, y, n_prototypes_per_class=10):
    """Replace dataset with cluster centroids"""
    prototypes_X, prototypes_y = [], []
    
    for label in np.unique(y):
        X_class = X[y == label]
        n_proto = min(n_prototypes_per_class, len(X_class))
        
        kmeans = KMeans(n_clusters=n_proto, random_state=42)
        kmeans.fit(X_class)
        
        prototypes_X.extend(kmeans.cluster_centers_)
        prototypes_y.extend([label] * n_proto)
    
    return np.array(prototypes_X), np.array(prototypes_y)

# Reduce 10,000 points to 100 prototypes
X_proto, y_proto = create_prototypes(X_train, y_train, n_prototypes_per_class=10)
```

**2. Dimensionality Reduction:**
```python
from sklearn.decomposition import PCA

# Reduce to essential dimensions
pca = PCA(n_components=20)
X_reduced = pca.fit_transform(X_train)

# Save PCA model for mobile deployment
import joblib
joblib.dump(pca, 'pca_model.pkl')
```

**3. Full Mobile Optimization Pipeline:**
```python
def create_mobile_knn(X_train, y_train, target_size_mb=1):
    """Create memory-optimized K-NN for mobile"""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    
    # Step 1: Reduce dimensions
    n_components = min(32, X_train.shape[1])
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_train)
    
    # Step 2: Prototype selection
    n_prototypes = int(target_size_mb * 1024 * 1024 / (n_components * 2))  # 2 bytes per float16
    prototypes_per_class = n_prototypes // len(np.unique(y_train))
    X_proto, y_proto = create_prototypes(X_reduced, y_train, prototypes_per_class)
    
    # Step 3: Quantize
    X_proto = X_proto.astype(np.float16)
    
    return pca, X_proto, y_proto

# Deploy: ~1MB model
pca, X_mobile, y_mobile = create_mobile_knn(X_train, y_train, target_size_mb=1)
```

**4. Use Lightweight Libraries:**
- **ONNX Runtime**: Optimized inference
- **TensorFlow Lite**: Mobile deployment
- **Core ML**: iOS devices

---

## Question 69
- [ ] Done

**What is the role of K-NN in ensemble learning and stacking?**

### Answer

K-NN contributes to ensembles as a diverse base learner—its local, non-parametric nature complements global models like linear regression or trees. In stacking, K-NN predictions become meta-features that capture local neighborhood information.

**K-NN in Ensemble Methods:**

**1. Voting Ensemble:**
```python
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Combine diverse models
ensemble = VotingClassifier(
    estimators=[
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('lr', LogisticRegression()),
        ('dt', DecisionTreeClassifier())
    ],
    voting='soft'  # Use probabilities
)
ensemble.fit(X_train, y_train)
```

**2. Stacking with K-NN:**
```python
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# K-NN as base learner
stacking = StackingClassifier(
    estimators=[
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(probability=True)),
    ],
    final_estimator=LogisticRegression(),
    cv=5
)
stacking.fit(X_train, y_train)
```

**3. Manual Stacking (K-NN Features):**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict

# Generate K-NN meta-features
knn = KNeighborsClassifier(n_neighbors=5)
knn_proba = cross_val_predict(knn, X_train, y_train, cv=5, method='predict_proba')

# Stack with original features
X_stacked = np.hstack([X_train, knn_proba])

# Train meta-learner
from sklearn.ensemble import GradientBoostingClassifier
meta_learner = GradientBoostingClassifier()
meta_learner.fit(X_stacked, y_train)
```

**Why K-NN is Valuable in Ensembles:**
- Non-parametric (different error patterns than parametric models)
- Local (captures neighborhood structure)
- Provides probability estimates

---

## Question 70
- [ ] Done

**How do you combine K-NN with other machine learning algorithms?**

### Answer

K-NN combines with other algorithms through: (1) feature augmentation (K-NN distances/predictions as features), (2) cascading (K-NN for hard cases), (3) hybrid predictions (weighted combination), (4) K-NN for post-processing.

**1. K-NN Features for Other Models:**
```python
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

def add_knn_features(X_train, y_train, X_test, k=5):
    """Add K-NN based features"""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_train)
    
    # Features for training data (use leave-one-out)
    train_features = []
    for i in range(len(X_train)):
        mask = np.ones(len(X_train), dtype=bool)
        mask[i] = False
        nn_temp = NearestNeighbors(n_neighbors=k)
        nn_temp.fit(X_train[mask])
        dists, _ = nn_temp.kneighbors([X_train[i]])
        train_features.append([dists.mean(), dists.std()])
    
    # Features for test data
    test_dists, _ = nn.kneighbors(X_test)
    test_features = np.column_stack([test_dists.mean(axis=1), test_dists.std(axis=1)])
    
    return np.column_stack([X_train, train_features]), np.column_stack([X_test, test_features])

X_train_aug, X_test_aug = add_knn_features(X_train, y_train, X_test)
gb = GradientBoostingClassifier()
gb.fit(X_train_aug, y_train)
```

**2. Cascading (K-NN for Uncertain Cases):**
```python
def cascaded_classifier(X_train, y_train, X_test, confidence_threshold=0.8):
    """Fast model first, K-NN for hard cases"""
    from sklearn.linear_model import LogisticRegression
    
    # Stage 1: Fast model
    fast_model = LogisticRegression()
    fast_model.fit(X_train, y_train)
    probas = fast_model.predict_proba(X_test)
    
    # Stage 2: K-NN for uncertain predictions
    confident_mask = probas.max(axis=1) >= confidence_threshold
    predictions = fast_model.predict(X_test)
    
    if not confident_mask.all():
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        predictions[~confident_mask] = knn.predict(X_test[~confident_mask])
    
    return predictions
```

**3. K-NN + Neural Network (Embedding + K-NN):**
```python
# Use neural network to create embeddings, then K-NN for classification
from tensorflow.keras.models import Model

# Extract embeddings from penultimate layer
embedding_model = Model(inputs=nn_model.input, outputs=nn_model.layers[-2].output)
embeddings = embedding_model.predict(X)

# K-NN on learned embeddings
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(embeddings_train, y_train)
```

---

## Question 71
- [ ] Done

**What are the challenges of K-NN for very large datasets (big data)?**

### Answer

K-NN faces severe scalability issues with big data: O(n) prediction time, O(n×d) memory, and no parallelization in naive implementation. Solutions include approximate methods, distributed computing, and data reduction.

**Big Data Challenges:**

| Challenge | Impact |
|-----------|--------|
| **Memory** | Can't fit 1B points in RAM |
| **Latency** | O(n) distance calculations per query |
| **Index Building** | kd-tree fails in high dimensions |
| **Updates** | Adding new data is expensive |

**Solutions:**

**1. Approximate Nearest Neighbors (FAISS):**
```python
import faiss
import numpy as np

# For 100M vectors, 128 dimensions
n, d = 100_000_000, 128

# IVF index with product quantization
nlist = 10000  # Number of clusters
m = 8  # Number of subquantizers
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

# Train on subset
sample = X[:1_000_000]
index.train(sample)

# Add in batches
batch_size = 1_000_000
for i in range(0, n, batch_size):
    index.add(X[i:i+batch_size])

# Search
index.nprobe = 100
distances, indices = index.search(query, k=5)
```

**2. Data Reduction (Random Sampling + Clustering):**
```python
from sklearn.cluster import MiniBatchKMeans

# Reduce 100M points to 1M representatives
def reduce_dataset(X, y, target_size=1_000_000):
    if len(X) <= target_size:
        return X, y
    
    # Stratified sampling
    from sklearn.model_selection import train_test_split
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=target_size, stratify=y, random_state=42
    )
    return X_sample, y_sample
```

**3. Distributed K-NN (Spark):**
```python
# Conceptual PySpark example
from pyspark.ml.feature import BucketedRandomProjectionLSH

# LSH for approximate K-NN
brp = BucketedRandomProjectionLSH(
    inputCol="features", outputCol="hashes", bucketLength=2.0, numHashTables=3
)
model = brp.fit(training_df)
similar = model.approxNearestNeighbors(training_df, query_vector, k=5)
```

**Interview Tip:** FAISS + IVF-PQ can handle billions of vectors; use for any production big data K-NN.

---

## Question 72
- [ ] Done

**How do you implement K-NN using MapReduce and distributed computing?**

### Answer

Distributed K-NN partitions data across nodes. Each node finds local K-nearest candidates, then a reducer merges results to find global K-nearest. MapReduce naturally parallelizes the distance computation bottleneck.

**MapReduce K-NN Algorithm:**

```
Map Phase:
  - Each mapper has a partition of training data
  - For each query, compute distances to local data
  - Emit (query_id, local_top_k)

Reduce Phase:
  - Collect all local_top_k for each query
  - Merge and select global top K
  - Output final K neighbors
```

**PySpark Implementation:**
```python
from pyspark import SparkContext
from pyspark.sql import SparkSession
import numpy as np

spark = SparkSession.builder.appName("KNN").getOrCreate()
sc = spark.sparkContext

def distributed_knn(X_train_rdd, y_train_rdd, X_query, k=5):
    """
    X_train_rdd: RDD of (index, feature_vector)
    y_train_rdd: RDD of (index, label)
    """
    query_broadcast = sc.broadcast(X_query)
    
    def local_knn(partition):
        """Find local K-nearest for each query point"""
        query = query_broadcast.value
        local_data = list(partition)
        
        results = []
        for q_idx, q_point in enumerate(query):
            distances = []
            for train_idx, train_point in local_data:
                dist = np.linalg.norm(q_point - train_point)
                distances.append((train_idx, dist))
            
            # Local top K
            local_topk = sorted(distances, key=lambda x: x[1])[:k]
            for train_idx, dist in local_topk:
                results.append((q_idx, (train_idx, dist)))
        
        return iter(results)
    
    # Map phase: local K-NN per partition
    local_results = X_train_rdd.mapPartitions(local_knn)
    
    # Reduce phase: merge local results
    def merge_topk(candidates):
        sorted_candidates = sorted(candidates, key=lambda x: x[1])[:k]
        return sorted_candidates
    
    global_knn = local_results.groupByKey().mapValues(merge_topk)
    
    return global_knn.collect()

# Usage
X_train_rdd = sc.parallelize(list(enumerate(X_train)), numSlices=100)
result = distributed_knn(X_train_rdd, y_train_rdd, X_query, k=5)
```

**Optimizations:**
- **LSH Partitioning**: Hash similar points to same partition
- **Block Broadcasting**: Broadcast query points to all nodes
- **Combiners**: Reduce network traffic by local aggregation

**Interview Tip:** Key insight is that K-NN is embarrassingly parallel—each query is independent.

---

## Question 73
- [ ] Done

**What is the role of K-NN in transfer learning and domain adaptation?**

### Answer

K-NN can leverage transfer learning through learned embeddings: use a pre-trained model (from source domain) to extract features, then apply K-NN for classification in the target domain. For domain adaptation, weight or filter neighbors based on domain similarity.

**Transfer Learning with K-NN:**

**1. Feature Transfer (Most Common):**
```python
from tensorflow.keras.applications import ResNet50
from sklearn.neighbors import KNeighborsClassifier

# Pre-trained on ImageNet (source domain)
feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Extract features for target domain (e.g., medical images)
def extract_features(images):
    return feature_extractor.predict(images)

target_features = extract_features(target_images)

# K-NN on transferred features
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn.fit(target_features[:100], target_labels[:100])  # Few labeled examples
predictions = knn.predict(target_features[100:])
```

**2. Domain Adaptation with K-NN:**
```python
def domain_adaptive_knn(X_source, y_source, X_target, X_query, k=5):
    """Weight source neighbors by domain similarity"""
    from sklearn.neighbors import NearestNeighbors
    
    # Find neighbors from source domain
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_source)
    distances, indices = nn.kneighbors(X_query)
    
    # Weight by similarity to target domain
    target_centroid = X_target.mean(axis=0)
    domain_weights = []
    for idx in indices[0]:
        # Closer to target distribution = higher weight
        source_point = X_source[idx]
        domain_dist = np.linalg.norm(source_point - target_centroid)
        domain_weights.append(1 / (domain_dist + 1e-10))
    
    # Weighted voting
    labels = y_source[indices[0]]
    weighted_votes = {}
    for label, weight in zip(labels, domain_weights):
        weighted_votes[label] = weighted_votes.get(label, 0) + weight
    
    return max(weighted_votes, key=weighted_votes.get)
```

**3. Few-Shot Learning with K-NN:**
K-NN is ideal for few-shot scenarios—no retraining needed, just add support examples.

```python
# Few-shot: 5-way 5-shot classification
support_features = feature_extractor(support_images)  # 25 examples
query_features = feature_extractor(query_images)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(support_features, support_labels)
predictions = knn.predict(query_features)
```

---

## Question 74
- [ ] Done

**How do you handle K-NN for multi-modal and heterogeneous data?**

### Answer

Multi-modal data combines different types (text + images, numerical + categorical). K-NN handles this by: (1) computing separate distances per modality, (2) embedding each modality to common space, or (3) using specialized multi-modal metrics.

**Approaches:**

**1. Combined Distance (Weighted Sum):**
```python
import numpy as np
from sklearn.preprocessing import StandardScaler

def multi_modal_distance(x1, x2, modality_weights):
    """
    x1, x2: tuples of (numerical_features, text_embedding, image_embedding)
    modality_weights: [w_num, w_text, w_image]
    """
    total_dist = 0
    
    # Numerical: Euclidean
    num_dist = np.linalg.norm(x1[0] - x2[0])
    total_dist += modality_weights[0] * num_dist
    
    # Text: Cosine distance
    text_dist = 1 - np.dot(x1[1], x2[1]) / (np.linalg.norm(x1[1]) * np.linalg.norm(x2[1]))
    total_dist += modality_weights[1] * text_dist
    
    # Image: Euclidean on embeddings
    img_dist = np.linalg.norm(x1[2] - x2[2])
    total_dist += modality_weights[2] * img_dist
    
    return total_dist
```

**2. Joint Embedding Space:**
```python
from sentence_transformers import SentenceTransformer
from tensorflow.keras.applications import ResNet50
import numpy as np

class MultiModalEncoder:
    def __init__(self):
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.image_encoder = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    def encode(self, texts, images, numerical):
        # Encode each modality
        text_emb = self.text_encoder.encode(texts)
        image_emb = self.image_encoder.predict(images)
        
        # Normalize and concatenate
        text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
        image_emb = image_emb / np.linalg.norm(image_emb, axis=1, keepdims=True)
        numerical = StandardScaler().fit_transform(numerical)
        
        return np.hstack([text_emb, image_emb, numerical])

# Usage
encoder = MultiModalEncoder()
X_combined = encoder.encode(texts, images, numerical_features)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_combined, labels)
```

**3. Gower Distance for Heterogeneous Data:**
```python
import gower
from sklearn.neighbors import NearestNeighbors

# Gower handles mixed types automatically
# Columns can be numeric, categorical, or binary
gower_distances = gower.gower_matrix(df)

# K-NN with precomputed distances
nn = NearestNeighbors(n_neighbors=5, metric='precomputed')
nn.fit(gower_distances)
```

**Interview Tip:** Gower distance is the go-to for mixed data types; for embeddings, concatenate normalized vectors.

---

## Question 75
- [ ] Done

**What are the fairness and bias considerations in K-NN algorithms?**

### Answer

K-NN can perpetuate or amplify biases present in training data. If certain groups are underrepresented or historically disadvantaged, K-NN will learn these patterns. Key concerns: representation bias, feature bias, and disparate impact.

**Bias Sources in K-NN:**

| Source | Description | Example |
|--------|-------------|---------|
| **Sampling bias** | Underrepresented groups | Fewer samples from minorities |
| **Feature bias** | Features correlate with protected attributes | Zip code → race |
| **Historical bias** | Past discrimination in labels | Biased hiring decisions |
| **Proximity bias** | Disadvantaged groups clustered | Redlining effects |

**Fairness Metrics:**
```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def calculate_fairness_metrics(knn, X_test, y_test, sensitive_attr):
    """
    sensitive_attr: binary array (0=group A, 1=group B)
    """
    predictions = knn.predict(X_test)
    
    # Separate by group
    group_a_mask = sensitive_attr == 0
    group_b_mask = sensitive_attr == 1
    
    # Demographic Parity: P(Y=1|A) = P(Y=1|B)
    rate_a = predictions[group_a_mask].mean()
    rate_b = predictions[group_b_mask].mean()
    demographic_parity_diff = abs(rate_a - rate_b)
    
    # Equal Opportunity: TPR_A = TPR_B
    tpr_a = (predictions[group_a_mask] & (y_test[group_a_mask] == 1)).sum() / (y_test[group_a_mask] == 1).sum()
    tpr_b = (predictions[group_b_mask] & (y_test[group_b_mask] == 1)).sum() / (y_test[group_b_mask] == 1).sum()
    equal_opportunity_diff = abs(tpr_a - tpr_b)
    
    return {
        'demographic_parity_diff': demographic_parity_diff,
        'equal_opportunity_diff': equal_opportunity_diff
    }
```

**Mitigation Strategies:**

**1. Pre-processing (Rebalancing):**
```python
from imblearn.over_sampling import SMOTE

# Oversample underrepresented groups
smote = SMOTE()
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
```

**2. In-processing (Fair Distance):**
```python
def fair_distance(x1, x2, sensitive_features):
    """Exclude sensitive features from distance"""
    non_sensitive_idx = [i for i in range(len(x1)) if i not in sensitive_features]
    return np.linalg.norm(x1[non_sensitive_idx] - x2[non_sensitive_idx])
```

**3. Post-processing (Threshold Adjustment):**
```python
# Different thresholds per group to equalize rates
def fair_predict(knn, X, sensitive_attr, threshold_a=0.5, threshold_b=0.5):
    probas = knn.predict_proba(X)[:, 1]
    predictions = np.zeros(len(X))
    predictions[sensitive_attr == 0] = (probas[sensitive_attr == 0] >= threshold_a).astype(int)
    predictions[sensitive_attr == 1] = (probas[sensitive_attr == 1] >= threshold_b).astype(int)
    return predictions
```

---

## Question 76
- [ ] Done

**How do you implement K-NN for survival analysis and censored data?**

### Answer

Survival analysis predicts time-to-event (e.g., death, churn) with censored data (event not observed yet). K-NN adapts by using survival-specific metrics (e.g., weighted Kaplan-Meier) on neighbors' survival times rather than simple voting.

**Key Concepts:**
- **Censored**: Event not observed (patient still alive, customer still active)
- **Survival function**: P(T > t) = probability of surviving past time t

**K-NN Kaplan-Meier Estimator:**
```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
from lifelines import KaplanMeierFitter

class SurvivalKNN:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, times, events):
        """
        X: features
        times: observed times
        events: 1=event occurred, 0=censored
        """
        self.X = X
        self.times = times
        self.events = events
        self.nn = NearestNeighbors(n_neighbors=self.k)
        self.nn.fit(X)
    
    def predict_survival(self, X_query, time_points):
        """Predict survival probability at given time points"""
        distances, indices = self.nn.kneighbors(X_query)
        
        survival_probs = []
        for i in range(len(X_query)):
            # Get neighbors' survival data
            neighbor_times = self.times[indices[i]]
            neighbor_events = self.events[indices[i]]
            
            # Fit Kaplan-Meier on neighbors
            kmf = KaplanMeierFitter()
            kmf.fit(neighbor_times, neighbor_events)
            
            # Predict at specified time points
            probs = kmf.survival_function_at_times(time_points).values
            survival_probs.append(probs)
        
        return np.array(survival_probs)
    
    def predict_median_survival(self, X_query):
        """Predict median survival time"""
        distances, indices = self.nn.kneighbors(X_query)
        
        medians = []
        for i in range(len(X_query)):
            neighbor_times = self.times[indices[i]]
            neighbor_events = self.events[indices[i]]
            
            kmf = KaplanMeierFitter()
            kmf.fit(neighbor_times, neighbor_events)
            medians.append(kmf.median_survival_time_)
        
        return np.array(medians)

# Usage
sknn = SurvivalKNN(k=10)
sknn.fit(X_train, times_train, events_train)
survival_probs = sknn.predict_survival(X_test, time_points=[12, 24, 36])
```

**Using scikit-survival:**
```python
# pip install scikit-survival
from sksurv.neighbors import KNeighborsSurvivalAnalysis

# y must be structured array: (event, time)
y = np.array([(e, t) for e, t in zip(events, times)], 
             dtype=[('event', bool), ('time', float)])

knn_surv = KNeighborsSurvivalAnalysis(n_neighbors=5)
knn_surv.fit(X_train, y_train)
survival_funcs = knn_surv.predict_survival_function(X_test)
```

---

## Question 77
- [ ] Done

**What is the relationship between K-NN and prototype-based learning?**

### Answer

K-NN stores all training instances; prototype-based learning stores only representative points (prototypes). Prototypes reduce memory and speed up prediction while maintaining accuracy. K-NN can be seen as prototype-based learning with all points as prototypes.

**Comparison:**

| Aspect | K-NN | Prototype-Based |
|--------|------|-----------------|
| Storage | All N points | M prototypes (M << N) |
| Memory | O(N × d) | O(M × d) |
| Prediction | O(N) | O(M) |
| Accuracy | Baseline | Similar if prototypes chosen well |

**Prototype Generation Methods:**

**1. Cluster Centroids:**
```python
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def create_prototypes_kmeans(X, y, n_prototypes_per_class=10):
    prototypes_X, prototypes_y = [], []
    
    for label in np.unique(y):
        X_class = X[y == label]
        k = min(n_prototypes_per_class, len(X_class))
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_class)
        
        prototypes_X.extend(kmeans.cluster_centers_)
        prototypes_y.extend([label] * k)
    
    return np.array(prototypes_X), np.array(prototypes_y)

X_proto, y_proto = create_prototypes_kmeans(X_train, y_train, n_prototypes_per_class=20)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_proto, y_proto)  # Much faster predictions!
```

**2. Condensed Nearest Neighbor (CNN):**
Keep only misclassified boundary points.

```python
def condensed_nn(X, y):
    """Keep boundary points only"""
    store_X, store_y = [X[0]], [y[0]]
    
    changed = True
    while changed:
        changed = False
        for i in range(len(X)):
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(np.array(store_X), np.array(store_y))
            
            if knn.predict([X[i]])[0] != y[i]:
                store_X.append(X[i])
                store_y.append(y[i])
                changed = True
    
    return np.array(store_X), np.array(store_y)
```

**Interview Tip:** Prototypes reduce K-NN from O(N) to O(M); use KMeans for simple prototype generation.

---

## Question 78
- [ ] Done

**How do you implement learning vector quantization (LVQ) with K-NN?**

### Answer

LVQ (Learning Vector Quantization) learns prototype positions through gradient-like updates: move prototypes toward same-class points and away from different-class points. Unlike K-Means, LVQ uses labels to optimize classification boundaries.

**LVQ Algorithm:**
1. Initialize prototypes (e.g., random samples per class)
2. For each training point:
   - Find nearest prototype
   - If same class: move prototype closer
   - If different class: move prototype away
3. Repeat until convergence

**LVQ Implementation:**
```python
import numpy as np

class LVQ:
    def __init__(self, n_prototypes_per_class=1, learning_rate=0.1, max_iter=100):
        self.n_proto = n_prototypes_per_class
        self.lr = learning_rate
        self.max_iter = max_iter
    
    def fit(self, X, y):
        classes = np.unique(y)
        
        # Initialize prototypes randomly from data
        self.prototypes = []
        self.proto_labels = []
        for c in classes:
            X_class = X[y == c]
            indices = np.random.choice(len(X_class), self.n_proto, replace=False)
            self.prototypes.extend(X_class[indices])
            self.proto_labels.extend([c] * self.n_proto)
        
        self.prototypes = np.array(self.prototypes)
        self.proto_labels = np.array(self.proto_labels)
        
        # Training loop
        for epoch in range(self.max_iter):
            for i in np.random.permutation(len(X)):
                # Find nearest prototype
                distances = np.linalg.norm(self.prototypes - X[i], axis=1)
                nearest_idx = np.argmin(distances)
                
                # Update rule
                if self.proto_labels[nearest_idx] == y[i]:
                    # Same class: attract
                    self.prototypes[nearest_idx] += self.lr * (X[i] - self.prototypes[nearest_idx])
                else:
                    # Different class: repel
                    self.prototypes[nearest_idx] -= self.lr * (X[i] - self.prototypes[nearest_idx])
            
            # Decay learning rate
            self.lr *= 0.99
        
        return self
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.prototypes - x, axis=1)
            nearest_idx = np.argmin(distances)
            predictions.append(self.proto_labels[nearest_idx])
        return np.array(predictions)

# Usage
lvq = LVQ(n_prototypes_per_class=5, learning_rate=0.1, max_iter=100)
lvq.fit(X_train, y_train)
predictions = lvq.predict(X_test)
```

**LVQ vs K-Means:**
- K-Means: Unsupervised, optimizes cluster compactness
- LVQ: Supervised, optimizes classification boundaries

---

## Question 79
- [ ] Done

**What are the advances in neural K-NN and differentiable nearest neighbors?**

### Answer

Neural K-NN makes the nearest neighbor operation differentiable, allowing end-to-end training with neural networks. Key advances include soft attention over neighbors, differentiable sorting, and neural memory networks that learn what to store and retrieve.

**Key Advances:**

**1. Soft/Differentiable K-NN:**
Replace hard K-nearest selection with soft attention over all points.

```python
import torch
import torch.nn.functional as F

def soft_knn(query, keys, values, temperature=1.0):
    """
    Differentiable K-NN using attention
    query: (batch, d)
    keys: (n, d) - training features
    values: (n, c) - one-hot labels
    """
    # Compute distances
    distances = torch.cdist(query, keys)  # (batch, n)
    
    # Soft attention weights (closer = higher weight)
    weights = F.softmax(-distances / temperature, dim=1)  # (batch, n)
    
    # Weighted combination of values
    output = torch.matmul(weights, values)  # (batch, c)
    
    return output  # Soft class probabilities

# As temperature → 0, approaches hard K-NN
```

**2. Neural Memory Networks:**
```python
import torch.nn as nn

class NeuralKNN(nn.Module):
    def __init__(self, input_dim, memory_size, embedding_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, embedding_dim)
        
        # Learnable memory keys and values
        self.memory_keys = nn.Parameter(torch.randn(memory_size, embedding_dim))
        self.memory_values = nn.Parameter(torch.randn(memory_size, embedding_dim))
    
    def forward(self, x):
        # Encode query
        query = self.encoder(x)
        
        # Attention over memory
        scores = torch.matmul(query, self.memory_keys.T)
        weights = F.softmax(scores, dim=-1)
        
        # Retrieve from memory
        output = torch.matmul(weights, self.memory_values)
        return output
```

**3. Differentiable Sorting (for Top-K):**
- SoftSort, NeuralSort algorithms
- Allow gradients through ranking operations

**Applications:**
- Few-shot learning with metric learning
- Retrieval-augmented models
- Memory-augmented neural networks

---

## Question 80
- [ ] Done

**How do you integrate K-NN with deep learning architectures?**

### Answer

K-NN integrates with deep learning by using neural networks to learn embeddings where K-NN works well. The network learns to map inputs to a space where similar items are close. Common patterns: Siamese networks, triplet networks, and retrieval-augmented models.

**Integration Patterns:**

**1. CNN Embedding + K-NN (Two-Stage):**
```python
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier

class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

# Train embedding network with contrastive loss
# Then use K-NN on embeddings
model = EmbeddingNet(input_dim, 128)
embeddings = model(X_train).detach().numpy()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(embeddings, y_train)
```

**2. End-to-End Differentiable K-NN:**
```python
class DifferentiableKNN(nn.Module):
    def __init__(self, encoder, support_X, support_y, k=5, temperature=0.1):
        super().__init__()
        self.encoder = encoder
        self.register_buffer('support_X', support_X)
        self.register_buffer('support_y', F.one_hot(support_y).float())
        self.k = k
        self.temperature = temperature
    
    def forward(self, x):
        # Encode query and support
        query_emb = self.encoder(x)
        support_emb = self.encoder(self.support_X)
        
        # Soft K-NN
        distances = torch.cdist(query_emb, support_emb)
        weights = F.softmax(-distances / self.temperature, dim=1)
        
        return torch.matmul(weights, self.support_y)
```

**3. Retrieval-Augmented Generation (RAG-style):**
```python
class RetrievalAugmentedModel(nn.Module):
    def __init__(self, encoder, memory_bank, k=5):
        super().__init__()
        self.encoder = encoder
        self.memory_keys = memory_bank['embeddings']
        self.memory_values = memory_bank['data']
        self.k = k
    
    def forward(self, x):
        query_emb = self.encoder(x)
        
        # Retrieve K nearest from memory
        distances = torch.cdist(query_emb, self.memory_keys)
        _, top_k_indices = distances.topk(self.k, largest=False)
        
        # Use retrieved context
        retrieved = self.memory_values[top_k_indices]
        
        # Combine query with retrieved context
        augmented = torch.cat([query_emb.unsqueeze(1).expand(-1, self.k, -1), retrieved], dim=-1)
        return augmented
```

**Interview Tip:** Modern approach: Train neural network to create good embeddings, then use K-NN for interpretable predictions.

---

## Question 81
- [ ] Done

**What are the considerations for K-NN model deployment and productionization?**

### Answer

Deploying K-NN in production requires addressing: memory footprint (storing all data), latency requirements, update mechanisms, scaling strategy, and monitoring. Unlike parametric models, K-NN needs special consideration for data storage and retrieval.

**Deployment Checklist:**

| Consideration | Challenge | Solution |
|---------------|-----------|----------|
| **Memory** | Store N×d floats | Prototype selection, quantization |
| **Latency** | O(N) per query | ANN indexes (FAISS, HNSW) |
| **Updates** | Add new training data | Online/incremental indexes |
| **Scaling** | Multiple users | Horizontal scaling, caching |
| **Monitoring** | Detect drift | Track neighbor distributions |

**Production Architecture:**
```python
import faiss
import numpy as np
import pickle

class ProductionKNN:
    def __init__(self, n_neighbors=5):
        self.k = n_neighbors
        self.index = None
        self.labels = None
        
    def build_index(self, X, y):
        """Build optimized index for production"""
        d = X.shape[1]
        
        # Use IVF for large datasets
        nlist = min(int(np.sqrt(len(X))), 1000)
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist)
        
        # Train and add
        self.index.train(X.astype('float32'))
        self.index.add(X.astype('float32'))
        self.labels = y
        
        # Set search parameters
        self.index.nprobe = min(nlist // 10, 100)
    
    def save(self, path):
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/labels.pkl", 'wb') as f:
            pickle.dump(self.labels, f)
    
    def load(self, path):
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/labels.pkl", 'rb') as f:
            self.labels = pickle.load(f)
    
    def predict(self, X):
        distances, indices = self.index.search(X.astype('float32'), self.k)
        predictions = []
        for idx in indices:
            neighbor_labels = self.labels[idx]
            pred = np.bincount(neighbor_labels).argmax()
            predictions.append(pred)
        return np.array(predictions)

# Deploy
model = ProductionKNN(n_neighbors=5)
model.build_index(X_train, y_train)
model.save('model_artifacts/')
```

**REST API Example:**
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
model = ProductionKNN()
model.load('model_artifacts/')

@app.route('/predict', methods=['POST'])
def predict():
    data = np.array(request.json['features']).astype('float32')
    prediction = model.predict(data.reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})
```

---

## Question 82
- [ ] Done

**How do you monitor and maintain K-NN models in production environments?**

### Answer

Monitor K-NN in production by tracking: prediction latency, neighbor distance distributions (detect drift), accuracy on labeled feedback, and memory usage. Set up alerts for anomalies and schedule regular retraining.

**Key Monitoring Metrics:**

**1. Performance Metrics:**
```python
import time
import numpy as np
from collections import deque

class KNNMonitor:
    def __init__(self, window_size=1000):
        self.latencies = deque(maxlen=window_size)
        self.distances = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
    
    def log_prediction(self, latency, distances, prediction):
        self.latencies.append(latency)
        self.distances.append(distances.mean())
        self.predictions.append(prediction)
    
    def get_metrics(self):
        return {
            'avg_latency_ms': np.mean(self.latencies) * 1000,
            'p99_latency_ms': np.percentile(self.latencies, 99) * 1000,
            'avg_neighbor_distance': np.mean(self.distances),
            'distance_std': np.std(self.distances),
        }
    
    def check_alerts(self, thresholds):
        metrics = self.get_metrics()
        alerts = []
        
        if metrics['avg_latency_ms'] > thresholds['max_latency_ms']:
            alerts.append(f"High latency: {metrics['avg_latency_ms']:.2f}ms")
        
        if metrics['avg_neighbor_distance'] > thresholds['max_avg_distance']:
            alerts.append(f"Potential drift: avg distance {metrics['avg_neighbor_distance']:.4f}")
        
        return alerts

# Usage in prediction
monitor = KNNMonitor()

def monitored_predict(model, X):
    start = time.time()
    distances, indices = model.index.search(X, model.k)
    prediction = model.predict(X)
    latency = time.time() - start
    
    monitor.log_prediction(latency, distances, prediction)
    return prediction
```

**2. Drift Detection:**
```python
def detect_data_drift(baseline_distances, current_distances, threshold=0.1):
    """Compare neighbor distance distributions"""
    from scipy.stats import ks_2samp
    
    stat, p_value = ks_2samp(baseline_distances, current_distances)
    
    if p_value < threshold:
        return True, f"Drift detected (KS stat: {stat:.4f}, p-value: {p_value:.4f})"
    return False, "No drift detected"
```

**3. Retraining Triggers:**
- Accuracy drops below threshold
- Distance distribution shifts significantly
- New data volume exceeds threshold
- Scheduled (daily/weekly)

---

## Question 83
- [ ] Done

**What is model versioning and A/B testing for K-NN algorithms?**

### Answer

Version K-NN models by storing: training data snapshot, index file, hyperparameters, and feature preprocessing. A/B test by routing traffic between versions and comparing metrics (accuracy, latency, business KPIs).

**Model Versioning:**
```python
import os
import json
import faiss
import hashlib
from datetime import datetime

class VersionedKNN:
    def __init__(self, base_path='models/'):
        self.base_path = base_path
    
    def save_version(self, index, labels, metadata):
        """Save model with version info"""
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_path = os.path.join(self.base_path, version)
        os.makedirs(version_path, exist_ok=True)
        
        # Save index and labels
        faiss.write_index(index, f"{version_path}/index.faiss")
        np.save(f"{version_path}/labels.npy", labels)
        
        # Save metadata
        metadata['version'] = version
        metadata['created_at'] = datetime.now().isoformat()
        with open(f"{version_path}/metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        return version
    
    def load_version(self, version):
        version_path = os.path.join(self.base_path, version)
        index = faiss.read_index(f"{version_path}/index.faiss")
        labels = np.load(f"{version_path}/labels.npy")
        with open(f"{version_path}/metadata.json") as f:
            metadata = json.load(f)
        return index, labels, metadata
    
    def list_versions(self):
        return sorted(os.listdir(self.base_path), reverse=True)
```

**A/B Testing:**
```python
import random

class ABTestRouter:
    def __init__(self, model_a, model_b, traffic_split=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.results_a = []
        self.results_b = []
    
    def predict(self, X, user_id=None):
        """Route prediction to model A or B"""
        # Consistent routing per user
        if user_id:
            use_a = hash(user_id) % 100 < self.traffic_split * 100
        else:
            use_a = random.random() < self.traffic_split
        
        if use_a:
            pred = self.model_a.predict(X)
            self.results_a.append({'prediction': pred, 'timestamp': time.time()})
            return pred, 'A'
        else:
            pred = self.model_b.predict(X)
            self.results_b.append({'prediction': pred, 'timestamp': time.time()})
            return pred, 'B'
    
    def get_statistics(self):
        return {
            'model_a_calls': len(self.results_a),
            'model_b_calls': len(self.results_b),
        }
```

---

## Question 84
- [ ] Done

**How do you handle K-NN for continual learning and lifelong learning?**

### Answer

Continual/lifelong learning requires K-NN to adapt to new data without forgetting old patterns. Strategies: sliding window (forget old), reservoir sampling (maintain representative subset), or expanding memory with consolidation.

**Challenges:**
- Memory grows unboundedly
- Old data may become irrelevant (concept drift)
- Need to maintain class balance

**Strategies:**

**1. Sliding Window:**
```python
from collections import deque

class SlidingWindowKNN:
    def __init__(self, window_size=10000, k=5):
        self.window_size = window_size
        self.k = k
        self.X_buffer = deque(maxlen=window_size)
        self.y_buffer = deque(maxlen=window_size)
    
    def partial_fit(self, X_new, y_new):
        """Add new data, automatically forget old"""
        for x, y in zip(X_new, y_new):
            self.X_buffer.append(x)
            self.y_buffer.append(y)
    
    def predict(self, X):
        X_train = np.array(self.X_buffer)
        y_train = np.array(self.y_buffer)
        
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(X_train, y_train)
        return knn.predict(X)
```

**2. Reservoir Sampling (Fixed Memory):**
```python
import random

class ReservoirKNN:
    def __init__(self, reservoir_size=10000, k=5):
        self.reservoir_size = reservoir_size
        self.k = k
        self.X_reservoir = []
        self.y_reservoir = []
        self.n_seen = 0
    
    def partial_fit(self, X_new, y_new):
        """Maintain representative sample via reservoir sampling"""
        for x, y in zip(X_new, y_new):
            self.n_seen += 1
            
            if len(self.X_reservoir) < self.reservoir_size:
                self.X_reservoir.append(x)
                self.y_reservoir.append(y)
            else:
                # Replace with decreasing probability
                j = random.randint(0, self.n_seen - 1)
                if j < self.reservoir_size:
                    self.X_reservoir[j] = x
                    self.y_reservoir[j] = y
```

**3. Class-Balanced Memory:**
```python
class BalancedMemoryKNN:
    def __init__(self, samples_per_class=1000, k=5):
        self.samples_per_class = samples_per_class
        self.k = k
        self.memory = {}  # {class: deque of samples}
    
    def partial_fit(self, X_new, y_new):
        for x, y in zip(X_new, y_new):
            if y not in self.memory:
                self.memory[y] = deque(maxlen=self.samples_per_class)
            self.memory[y].append(x)
```

---

## Question 85
- [ ] Done

**What are the emerging hardware accelerations for K-NN computations?**

### Answer

K-NN benefits from hardware acceleration due to its parallelizable distance computations. Key accelerators: GPUs (massive parallelism), TPUs (matrix operations), FPGAs (custom logic), and specialized vector processors (for similarity search).

**Hardware Options:**

| Hardware | Strength | Best For |
|----------|----------|----------|
| **GPU** | Massive parallelism | Batch queries, high throughput |
| **TPU** | Matrix operations | Large-scale similarity |
| **FPGA** | Custom logic, low latency | Real-time, edge deployment |
| **CPU + SIMD** | Vector instructions | Moderate scale, general purpose |

**GPU Acceleration with FAISS:**
```python
import faiss
import numpy as np

# Check GPU availability
ngpus = faiss.get_num_gpus()
print(f"Number of GPUs: {ngpus}")

# Create GPU index
d = 128  # dimension
n = 1000000  # database size

# CPU index
cpu_index = faiss.IndexFlatL2(d)

# Move to GPU
gpu_resource = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(gpu_resource, 0, cpu_index)

# Add data and search on GPU
gpu_index.add(X_database.astype('float32'))
distances, indices = gpu_index.search(X_query.astype('float32'), k=5)
```

**Multi-GPU:**
```python
# Use all available GPUs
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
```

**CPU SIMD Optimization:**
```python
# FAISS automatically uses AVX2/AVX-512 instructions
# Check compilation options
print(faiss.get_compile_options())

# For optimal CPU performance, ensure:
# - OpenMP enabled (parallel processing)
# - AVX2/AVX-512 support
# - Memory alignment
```

**Interview Tip:** FAISS with GPU can achieve 100x speedup over CPU for large-scale K-NN.

---

## Question 86
- [ ] Done

**How do you implement K-NN using GPU and specialized hardware?**

### Answer

GPU K-NN leverages thousands of parallel cores for distance computation. Libraries like FAISS, cuML, and PyTorch enable efficient GPU implementation. Key: batch queries together and keep data on GPU memory.

**Method 1: FAISS GPU:**
```python
import faiss
import numpy as np

def gpu_knn_faiss(X_train, X_query, k=5):
    d = X_train.shape[1]
    
    # Create GPU resources
    res = faiss.StandardGpuResources()
    
    # GPU index with IVF for large datasets
    nlist = 1024
    quantizer = faiss.IndexFlatL2(d)
    cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist)
    
    # Move to GPU
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    
    # Train and add
    gpu_index.train(X_train.astype('float32'))
    gpu_index.add(X_train.astype('float32'))
    
    # Search
    gpu_index.nprobe = 64
    distances, indices = gpu_index.search(X_query.astype('float32'), k)
    
    return distances, indices
```

**Method 2: cuML (RAPIDS):**
```python
# pip install cuml-cu11  (CUDA 11)
from cuml.neighbors import NearestNeighbors
import cupy as cp

# Data on GPU
X_train_gpu = cp.array(X_train)
X_query_gpu = cp.array(X_query)

# cuML K-NN
nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
nn.fit(X_train_gpu)
distances, indices = nn.kneighbors(X_query_gpu)

# Convert back to numpy if needed
distances = cp.asnumpy(distances)
indices = cp.asnumpy(indices)
```

**Method 3: PyTorch (Custom):**
```python
import torch

def pytorch_knn(X_train, X_query, k=5, device='cuda'):
    X_train = torch.tensor(X_train, device=device, dtype=torch.float32)
    X_query = torch.tensor(X_query, device=device, dtype=torch.float32)
    
    # Compute all pairwise distances (batched for memory)
    batch_size = 1000
    all_indices = []
    all_distances = []
    
    for i in range(0, len(X_query), batch_size):
        batch = X_query[i:i+batch_size]
        
        # Efficient distance computation
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        a_sq = (batch ** 2).sum(dim=1, keepdim=True)
        b_sq = (X_train ** 2).sum(dim=1).unsqueeze(0)
        ab = torch.mm(batch, X_train.T)
        distances = a_sq + b_sq - 2 * ab
        
        # Get top-k
        top_k = distances.topk(k, largest=False)
        all_distances.append(top_k.values.cpu())
        all_indices.append(top_k.indices.cpu())
    
    return torch.cat(all_distances), torch.cat(all_indices)
```

**Performance Tips:**
- Keep data on GPU (avoid CPU↔GPU transfers)
- Batch queries together
- Use float16 for 2x memory savings

---

## Question 87
- [ ] Done

**What is the role of K-NN in AutoML and automated algorithm selection?**

### Answer

In AutoML, K-NN serves as: (1) a baseline algorithm to beat, (2) a meta-learner for algorithm selection (find similar datasets, use their best algorithms), and (3) part of the search space for hyperparameter optimization.

**K-NN as Baseline:**
- Simple, no hyperparameters to tune (except K)
- Any AutoML system should beat K-NN
- Used in benchmarking

**K-NN for Meta-Learning (Algorithm Selection):**
```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class MetaLearningSelector:
    """Select best algorithm based on similar datasets"""
    
    def __init__(self, k=3):
        self.k = k
        self.meta_features = []  # Dataset characteristics
        self.best_algorithms = []  # What worked best
    
    def extract_meta_features(self, X, y):
        """Compute dataset meta-features"""
        return np.array([
            X.shape[0],  # Number of samples
            X.shape[1],  # Number of features
            len(np.unique(y)),  # Number of classes
            X.mean(),  # Mean value
            X.std(),  # Standard deviation
            np.corrcoef(X.T).mean(),  # Feature correlation
        ])
    
    def fit(self, datasets_info):
        """
        datasets_info: list of (X, y, best_algo_name)
        """
        for X, y, best_algo in datasets_info:
            meta_feat = self.extract_meta_features(X, y)
            self.meta_features.append(meta_feat)
            self.best_algorithms.append(best_algo)
        
        self.meta_features = np.array(self.meta_features)
        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        self.knn.fit(self.meta_features, self.best_algorithms)
    
    def recommend(self, X_new, y_new):
        """Recommend algorithm for new dataset"""
        meta_feat = self.extract_meta_features(X_new, y_new)
        return self.knn.predict([meta_feat])[0]

# Usage
selector = MetaLearningSelector(k=3)
selector.fit(historical_datasets)
recommended_algo = selector.recommend(new_X, new_y)
```

**K-NN in AutoML Search Space:**
```python
# Example with sklearn's HalvingGridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.neighbors import KNeighborsClassifier

param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'cosine'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree'],
}

search = HalvingGridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    factor=3
)
search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
```

---

## Question 88
- [ ] Done

**How do you handle K-NN for multi-objective optimization problems?**

### Answer

Multi-objective K-NN predicts multiple targets simultaneously or helps find Pareto-optimal solutions. For multi-output prediction, use `KNeighborsRegressor` with `MultiOutputRegressor`. For optimization, K-NN can guide search by identifying promising regions.

**1. Multi-Output K-NN:**
```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor

# y has multiple columns (objectives)
# y shape: (n_samples, n_objectives)

knn = KNeighborsRegressor(n_neighbors=5)
multi_knn = MultiOutputRegressor(knn)
multi_knn.fit(X_train, y_train)

# Predict all objectives
predictions = multi_knn.predict(X_test)  # Shape: (n_test, n_objectives)
```

**2. K-NN for Pareto Front Approximation:**
```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def knn_pareto_search(X_candidates, objectives_func, k=5, n_iterations=100):
    """
    Use K-NN to guide multi-objective optimization
    """
    evaluated_X = []
    evaluated_Y = []  # Objective values
    
    # Initial random samples
    initial_idx = np.random.choice(len(X_candidates), size=10, replace=False)
    for idx in initial_idx:
        y = objectives_func(X_candidates[idx])
        evaluated_X.append(X_candidates[idx])
        evaluated_Y.append(y)
    
    for _ in range(n_iterations):
        # Find K-NN of Pareto-optimal points
        pareto_idx = get_pareto_front_indices(np.array(evaluated_Y))
        pareto_X = np.array(evaluated_X)[pareto_idx]
        
        # Find neighbors of Pareto points in candidate space
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_candidates)
        _, neighbor_idx = nn.kneighbors(pareto_X)
        
        # Evaluate promising candidates
        for idx in neighbor_idx.flatten():
            if X_candidates[idx] not in evaluated_X:
                y = objectives_func(X_candidates[idx])
                evaluated_X.append(X_candidates[idx])
                evaluated_Y.append(y)
                break
    
    return np.array(evaluated_X), np.array(evaluated_Y)

def get_pareto_front_indices(Y):
    """Return indices of Pareto-optimal points"""
    is_pareto = np.ones(len(Y), dtype=bool)
    for i, y in enumerate(Y):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(Y[is_pareto] < y, axis=1)
            is_pareto[i] = True
    return np.where(is_pareto)[0]
```

**3. Weighted Scalarization:**
```python
def weighted_knn_prediction(knn_models, X, weights):
    """
    Combine predictions with objective weights
    knn_models: list of K-NN models, one per objective
    weights: importance weights for each objective
    """
    predictions = np.array([knn.predict(X) for knn in knn_models])
    weighted_sum = np.average(predictions, axis=0, weights=weights)
    return weighted_sum
```

---

## Question 89
- [ ] Done

**What are the research frontiers and open challenges in K-NN algorithms?**

### Answer

Current research frontiers in K-NN include: scalability to billions of points, streaming/online learning, learned index structures, privacy-preserving computation, and integration with deep learning. These address K-NN's fundamental limitations.

**Open Challenges and Research Directions:**

| Challenge | Current State | Research Direction |
|-----------|---------------|-------------------|
| **Billion-scale search** | ANN works but lossy | Better accuracy-speed trade-offs |
| **High dimensions** | Curse of dimensionality | Learned metrics, subspace methods |
| **Streaming data** | Sliding window | Efficient incremental indexes |
| **Privacy** | Slow cryptographic methods | Efficient secure computation |
| **Interpretability** | Good but limited | Counterfactual explanations |

**1. Learned Index Structures:**
Replace hand-crafted indexes (kd-tree, LSH) with neural networks that learn optimal data organization.

```python
# Conceptual: Neural network predicts which partition to search
class LearnedIndex:
    def __init__(self, n_partitions):
        self.partition_predictor = NeuralNetwork()
        self.partitions = [[] for _ in range(n_partitions)]
    
    def query(self, x):
        # NN predicts which partition(s) to search
        partition_probs = self.partition_predictor(x)
        top_partitions = partition_probs.topk(3)
        
        # Search only predicted partitions
        candidates = []
        for p in top_partitions:
            candidates.extend(self.partitions[p])
        return find_knn_in(candidates, x)
```

**2. Differentiable K-NN:**
Make K-NN fully differentiable for end-to-end learning.

**3. Federated K-NN:**
Privacy-preserving K-NN across distributed data without centralization.

**4. K-NN for Non-Euclidean Data:**
Graphs, manifolds, sequences with learned distance functions.

**5. Adaptive K Selection:**
Automatically determine optimal K per query based on local data density.

**Interview Tip:** Mention FAISS, HNSW, and learned indexes as cutting-edge solutions.

---

## Question 90
- [ ] Done

**How do you implement K-NN for reinforcement learning and policy learning?**

### Answer

K-NN in RL stores (state, action, reward) experiences and retrieves similar past states to inform action selection. This is called episodic memory or model-free planning—predict outcomes by looking up similar past situations.

**Applications:**

**1. K-NN Policy (Action Selection):**
```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

class KNNPolicy:
    """Select actions based on similar past states"""
    
    def __init__(self, k=5):
        self.k = k
        self.states = []
        self.actions = []
        self.rewards = []
    
    def store_experience(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def select_action(self, state, exploration=0.1):
        if len(self.states) < self.k or np.random.random() < exploration:
            return np.random.choice(self.n_actions)  # Explore
        
        # Find K similar states
        nn = NearestNeighbors(n_neighbors=self.k)
        nn.fit(np.array(self.states))
        distances, indices = nn.kneighbors([state])
        
        # Weight by reward and distance
        weights = 1 / (distances[0] + 1e-10)
        neighbor_actions = [self.actions[i] for i in indices[0]]
        neighbor_rewards = [self.rewards[i] for i in indices[0]]
        
        # Select action with highest weighted reward
        action_values = {}
        for a, r, w in zip(neighbor_actions, neighbor_rewards, weights):
            if a not in action_values:
                action_values[a] = []
            action_values[a].append(r * w)
        
        best_action = max(action_values, key=lambda a: np.mean(action_values[a]))
        return best_action
```

**2. K-NN Value Function:**
```python
class KNNValueFunction:
    """Estimate state values from similar past states"""
    
    def __init__(self, k=5, gamma=0.99):
        self.k = k
        self.gamma = gamma
        self.states = []
        self.returns = []  # Monte Carlo returns
    
    def update(self, trajectory):
        """Store trajectory with computed returns"""
        G = 0
        for state, reward in reversed(trajectory):
            G = reward + self.gamma * G
            self.states.append(state)
            self.returns.append(G)
    
    def estimate(self, state):
        if len(self.states) < self.k:
            return 0
        
        nn = NearestNeighbors(n_neighbors=self.k)
        nn.fit(np.array(self.states))
        distances, indices = nn.kneighbors([state])
        
        # Weighted average of neighbor returns
        weights = 1 / (distances[0] + 1e-10)
        neighbor_returns = [self.returns[i] for i in indices[0]]
        
        return np.average(neighbor_returns, weights=weights)
```

**3. Model-Based Planning with K-NN:**
```python
def knn_planning(current_state, experience_buffer, k=5, horizon=10):
    """Plan by simulating with K-NN transition model"""
    simulated_rewards = []
    
    for action in possible_actions:
        state = current_state
        total_reward = 0
        
        for t in range(horizon):
            # Find similar (state, action) pairs
            sa = np.concatenate([state, [action]])
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(experience_buffer['state_actions'])
            _, indices = nn.kneighbors([sa])
            
            # Predict next state and reward from neighbors
            next_states = experience_buffer['next_states'][indices[0]]
            rewards = experience_buffer['rewards'][indices[0]]
            
            state = next_states.mean(axis=0)  # Average next state
            total_reward += rewards.mean()
        
        simulated_rewards.append(total_reward)
    
    return possible_actions[np.argmax(simulated_rewards)]
```

---

## Question 91
- [ ] Done

**What are the theoretical guarantees and convergence properties of K-NN?**

### Answer

K-NN is universally consistent: as n→∞ and k→∞ with k/n→0, K-NN's error converges to the Bayes optimal error rate. This means K-NN will eventually achieve the best possible accuracy given enough data. Convergence rate depends on data dimensionality and smoothness.

**Key Theoretical Results:**

**1. Universal Consistency (Stone, 1977):**
If $k \to \infty$ and $k/n \to 0$ as $n \to \infty$, then:
$$P(error_{KNN}) \to P(error_{Bayes})$$

**2. Convergence Rate:**
For smooth decision boundaries in d dimensions:
$$E[error_{KNN}] - error_{Bayes} = O(n^{-4/(d+4)})$$

This shows the curse of dimensionality—convergence slows exponentially with dimension.

**3. 1-NN Error Bound (Cover & Hart, 1967):**
$$error_{Bayes} \leq error_{1-NN} \leq 2 \cdot error_{Bayes}(1 - error_{Bayes})$$

For small Bayes error, 1-NN error is at most twice optimal.

**Practical Implications:**
```python
import numpy as np
import matplotlib.pyplot as plt

def theoretical_convergence(n_samples, k, d):
    """
    Approximate convergence rate
    Error ≈ C * (k/n)^(2/d) + C' * (1/k)
    """
    # Bias term (decreases with k)
    bias = 1 / np.sqrt(k)
    
    # Variance term (increases with k)
    variance = np.sqrt(k / n_samples)
    
    # Curse of dimensionality factor
    dim_factor = (k / n_samples) ** (2 / (d + 4))
    
    return bias + variance + dim_factor

# Visualize
ns = np.logspace(2, 6, 50).astype(int)
for d in [2, 10, 50]:
    errors = [theoretical_convergence(n, k=int(np.sqrt(n)), d=d) for n in ns]
    plt.plot(ns, errors, label=f'd={d}')

plt.xscale('log'); plt.xlabel('n'); plt.ylabel('Excess Error')
plt.legend(); plt.title('K-NN Convergence Rate')
plt.show()
```

**Optimal K Selection (Theory):**
$$k^* \approx n^{4/(d+4)}$$

For d=10: $k^* \approx n^{0.29}$, so for n=10,000: $k^* \approx 20$

---

## Question 92
- [ ] Done

**How do you analyze the sample complexity and generalization bounds for K-NN?**

### Answer

Sample complexity asks: how many samples n are needed to achieve error ε? For K-NN, this depends heavily on dimension d. Generalization bounds show how training error relates to test error—K-NN's leave-one-out error is a good estimate.

**Sample Complexity:**

To achieve excess error ε above Bayes optimal:
$$n = O\left(\frac{1}{\epsilon^{(d+4)/2}}\right)$$

**Example:** For d=10 and ε=0.01:
- Need n ≈ (1/0.01)^7 = 10^14 samples
- This illustrates why K-NN struggles in high dimensions

**Generalization Bound (PAC Learning):**
```python
import numpy as np

def knn_generalization_bound(n, k, delta=0.05):
    """
    With probability 1-delta, test error is bounded by:
    train_error + O(sqrt(k log(n) / n) + sqrt(log(1/delta)/n))
    """
    # Estimation error
    estimation = np.sqrt(k * np.log(n) / n)
    
    # Approximation error  
    approximation = 1 / np.sqrt(k)
    
    # Confidence term
    confidence = np.sqrt(np.log(1/delta) / n)
    
    bound = estimation + approximation + confidence
    return bound

# Example
n, k = 10000, 10
bound = knn_generalization_bound(n, k)
print(f"Generalization bound: {bound:.4f}")
```

**Leave-One-Out Error Analysis:**
```python
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# LOO is almost unbiased for K-NN
knn = KNeighborsClassifier(n_neighbors=5)
loo_scores = cross_val_score(knn, X, y, cv=LeaveOneOut())

loo_error = 1 - loo_scores.mean()
print(f"LOO Error: {loo_error:.4f}")
# This closely estimates true generalization error
```

**Key Insight:** K-NN's LOO error is nearly unbiased (unlike many other algorithms), making it a reliable performance estimate.

---

## Question 93
- [ ] Done

**What is the relationship between K-NN and Bayesian non-parametric methods?**

### Answer

K-NN and Bayesian non-parametrics share the principle that model complexity grows with data. K-NN stores all points (model = data); Bayesian non-parametrics use infinite-dimensional priors. Both avoid fixed parametric assumptions and adapt to data complexity.

**Connections:**

| Aspect | K-NN | Bayesian Non-Parametric |
|--------|------|-------------------------|
| Complexity | Grows with n | Grows with n |
| Model | All training points | Posterior over functions |
| Inference | Distance-based | Probabilistic |
| Uncertainty | Neighbor variance | Full posterior |

**1. K-NN as Kernel Density Estimation:**
K-NN implicitly estimates local density, related to kernel density estimation (KDE).

```python
from sklearn.neighbors import KernelDensity, KNeighborsClassifier

# KDE is Bayesian non-parametric density estimation
kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
kde.fit(X_class_0)
log_density = kde.score_samples(X_test)

# K-NN classification uses similar local density ideas
```

**2. Gaussian Process vs K-NN:**
```python
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier

# GP: Full Bayesian non-parametric (with uncertainty)
gp = GaussianProcessClassifier()
gp.fit(X_train, y_train)
proba, std = gp.predict_proba(X_test), gp.predict(X_test)

# K-NN: Simpler, point estimate
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
proba = knn.predict_proba(X_test)  # No uncertainty on uncertainty
```

**3. Dirichlet Process Mixture vs K-NN:**
```python
# Conceptual comparison
# DPM: Infinite mixture model, clusters grow with data
# K-NN: Uses local neighborhoods, implicitly adapts to local density

# Both are "non-parametric" in the sense that
# model complexity is not fixed a priori
```

**Key Difference:**
- K-NN: Point estimates, no explicit uncertainty quantification
- Bayesian NP: Full posterior distributions, principled uncertainty

**Interview Tip:** Both are non-parametric; Bayesian methods provide principled uncertainty while K-NN is simpler and faster.

---

## Question 94
- [ ] Done

**How do you implement K-NN for causal inference and treatment effect estimation?**

### Answer

K-NN estimates treatment effects by matching treated and control units with similar characteristics. For each treated unit, find K similar untreated units (controls) and compare outcomes. This is nearest-neighbor matching for causal inference.

**Matching Estimator:**
```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

class NNMatchingEstimator:
    """Estimate Average Treatment Effect using K-NN matching"""
    
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, treatment, outcome):
        """
        X: covariates
        treatment: binary (0=control, 1=treated)
        outcome: observed outcome
        """
        self.X = X
        self.treatment = treatment
        self.outcome = outcome
        
        # Separate treated and control
        self.X_treated = X[treatment == 1]
        self.X_control = X[treatment == 0]
        self.Y_treated = outcome[treatment == 1]
        self.Y_control = outcome[treatment == 0]
    
    def estimate_ate(self):
        """Estimate Average Treatment Effect"""
        # Match each treated to K controls
        nn_control = NearestNeighbors(n_neighbors=self.k)
        nn_control.fit(self.X_control)
        
        nn_treated = NearestNeighbors(n_neighbors=self.k)
        nn_treated.fit(self.X_treated)
        
        # ATT: Average Treatment effect on Treated
        _, control_matches = nn_control.kneighbors(self.X_treated)
        counterfactual_control = self.Y_control[control_matches].mean(axis=1)
        att = (self.Y_treated - counterfactual_control).mean()
        
        # ATC: Average Treatment effect on Control
        _, treated_matches = nn_treated.kneighbors(self.X_control)
        counterfactual_treated = self.Y_treated[treated_matches].mean(axis=1)
        atc = (counterfactual_treated - self.Y_control).mean()
        
        # ATE: Weighted average
        n_treated = len(self.X_treated)
        n_control = len(self.X_control)
        ate = (n_treated * att + n_control * atc) / (n_treated + n_control)
        
        return {'ATE': ate, 'ATT': att, 'ATC': atc}

# Usage
estimator = NNMatchingEstimator(k=5)
estimator.fit(X, treatment, outcome)
effects = estimator.estimate_ate()
print(f"Average Treatment Effect: {effects['ATE']:.4f}")
```

**Individual Treatment Effect (ITE):**
```python
def estimate_ite(X, treatment, outcome, X_new, k=5):
    """Estimate treatment effect for a new individual"""
    X_control = X[treatment == 0]
    Y_control = outcome[treatment == 0]
    X_treated = X[treatment == 1]
    Y_treated = outcome[treatment == 1]
    
    nn_control = NearestNeighbors(n_neighbors=k)
    nn_control.fit(X_control)
    nn_treated = NearestNeighbors(n_neighbors=k)
    nn_treated.fit(X_treated)
    
    _, control_idx = nn_control.kneighbors([X_new])
    _, treated_idx = nn_treated.kneighbors([X_new])
    
    Y0_estimate = Y_control[control_idx[0]].mean()  # Without treatment
    Y1_estimate = Y_treated[treated_idx[0]].mean()  # With treatment
    
    ite = Y1_estimate - Y0_estimate
    return ite
```

**Key Assumption:** No unmeasured confounders (treatment assignment ignorable given X).

---

## Question 95
- [ ] Done

**What are the considerations for K-NN in federated and decentralized learning?**

### Answer

Federated K-NN keeps data on local devices. Challenges: can't compute global distances without sharing data. Solutions: share local top-K candidates only, use secure aggregation, or distribute index across nodes.

**Federated K-NN Protocol:**
```
1. Query broadcast to all nodes
2. Each node computes local K nearest (with distances)
3. Nodes send only top-K candidates (not raw data)
4. Aggregator selects global K nearest from candidates
5. Return prediction
```

**Implementation:**
```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

class FederatedKNNNode:
    def __init__(self, X_local, y_local, k=5):
        self.X = X_local
        self.y = y_local
        self.k = min(k, len(X_local))
        self.nn = NearestNeighbors(n_neighbors=self.k)
        self.nn.fit(X_local)
    
    def get_candidates(self, X_query):
        """Return local candidates without raw features"""
        distances, indices = self.nn.kneighbors(X_query)
        labels = self.y[indices]
        # Send only (distance, label) - not features
        return distances, labels

class FederatedKNNAggregator:
    def __init__(self, nodes, k=5):
        self.nodes = nodes
        self.k = k
    
    def predict(self, X_query):
        # Collect from all nodes
        all_distances = []
        all_labels = []
        
        for node in self.nodes:
            dists, labels = node.get_candidates(X_query)
            all_distances.append(dists)
            all_labels.append(labels)
        
        # Aggregate
        all_distances = np.concatenate(all_distances, axis=1)
        all_labels = np.concatenate(all_labels, axis=1)
        
        predictions = []
        for i in range(len(X_query)):
            top_k_idx = np.argsort(all_distances[i])[:self.k]
            k_labels = all_labels[i, top_k_idx]
            pred = np.bincount(k_labels.astype(int)).argmax()
            predictions.append(pred)
        
        return np.array(predictions)
```

**Privacy Considerations:**
- Distances can leak information → add noise
- Labels reveal class distribution → use secure aggregation
- Query patterns → add dummy queries

**Secure Aggregation:**
```python
def secure_aggregate_predictions(node_predictions, k):
    """Aggregate without revealing individual node outputs"""
    # In practice, use cryptographic secure aggregation
    # This is a simplified version
    
    # Each node adds random mask
    # Masks cancel out when aggregated
    aggregated = sum(node_predictions)  # Masks cancel
    return aggregated
```

---

## Question 96
- [ ] Done

**How do you handle K-NN for adversarial robustness and security?**

### Answer

K-NN is vulnerable to adversarial attacks: small input perturbations can change nearest neighbors and predictions. Defenses include robust distance metrics, neighbor verification, and adversarial training.

**Attack Types:**

| Attack | Description | K-NN Vulnerability |
|--------|-------------|-------------------|
| **Evasion** | Perturb test input | Change nearest neighbors |
| **Poisoning** | Add malicious training points | Influence predictions |
| **Membership Inference** | Detect if point in training | Neighbors reveal membership |

**1. Evasion Attack Example:**
```python
import numpy as np

def adversarial_perturbation(knn, x, target_class, epsilon=0.1):
    """Generate adversarial example to change prediction"""
    # Find nearest point of target class
    target_mask = knn.classes_ == target_class
    X_target = knn._fit_X[knn._y == target_class]
    
    nearest_target = X_target[np.argmin(np.linalg.norm(X_target - x, axis=1))]
    
    # Move toward target
    direction = nearest_target - x
    direction = direction / np.linalg.norm(direction)
    
    x_adv = x + epsilon * direction
    return x_adv
```

**2. Defense: Robust Distance Metric:**
```python
def robust_knn_predict(X_train, y_train, x_query, k=5, n_random=10):
    """Average predictions over random projections"""
    d = X_train.shape[1]
    predictions = []
    
    for _ in range(n_random):
        # Random projection
        proj = np.random.randn(d, d // 2)
        X_proj = X_train @ proj
        x_proj = x_query @ proj
        
        # K-NN on projected space
        distances = np.linalg.norm(X_proj - x_proj, axis=1)
        k_nearest = np.argsort(distances)[:k]
        pred = np.bincount(y_train[k_nearest]).argmax()
        predictions.append(pred)
    
    # Majority vote over projections
    return np.bincount(predictions).argmax()
```

**3. Defense: Neighbor Verification:**
```python
def verified_knn_predict(knn, x_query, X_train, y_train, k=5, threshold=0.1):
    """Verify neighbors are not adversarial"""
    distances, indices = knn.kneighbors([x_query])
    
    # Check for anomalous distances
    median_dist = np.median(distances[0])
    
    # Filter outlier neighbors
    valid_mask = distances[0] < median_dist * (1 + threshold)
    valid_indices = indices[0][valid_mask]
    
    if len(valid_indices) == 0:
        return -1  # Abstain
    
    return np.bincount(y_train[valid_indices]).argmax()
```

**4. Poisoning Defense:**
```python
def sanitize_training_data(X, y, contamination=0.1):
    """Remove potential poisoning points using LOF"""
    from sklearn.neighbors import LocalOutlierFactor
    
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    outlier_mask = lof.fit_predict(X) == 1  # 1 = inlier
    
    return X[outlier_mask], y[outlier_mask]
```

---

## Question 97
- [ ] Done

**What is the integration of K-NN with probabilistic and generative models?**

### Answer

K-NN integrates with probabilistic models through: (1) K-NN for density estimation in generative models, (2) Bayesian K-NN with posterior over predictions, (3) K-NN as likelihood in probabilistic frameworks.

**1. K-NN Density Estimation:**
```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def knn_density_estimate(X_train, X_query, k=5):
    """
    Estimate density: p(x) ∝ k / (n * V_k)
    where V_k is volume of hypersphere containing K neighbors
    """
    n, d = X_train.shape
    
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_train)
    distances, _ = nn.kneighbors(X_query)
    
    # Volume of d-dimensional ball
    r_k = distances[:, -1]  # Distance to k-th neighbor
    volume = (np.pi ** (d/2) / np.math.gamma(d/2 + 1)) * (r_k ** d)
    
    density = k / (n * volume)
    return density
```

**2. Generative Classification with K-NN:**
```python
def generative_knn_classify(X_train, y_train, X_query, k=5):
    """
    Use Bayes rule: P(y|x) ∝ P(x|y) * P(y)
    Estimate P(x|y) using K-NN density within each class
    """
    classes = np.unique(y_train)
    n = len(X_train)
    
    posteriors = []
    for x in X_query:
        class_probs = []
        for c in classes:
            X_class = X_train[y_train == c]
            n_class = len(X_class)
            
            # P(x|y=c) from K-NN density
            likelihood = knn_density_estimate(X_class, [x], k)[0]
            
            # P(y=c) = prior
            prior = n_class / n
            
            # P(y=c|x) ∝ likelihood * prior
            class_probs.append(likelihood * prior)
        
        # Normalize
        class_probs = np.array(class_probs)
        posteriors.append(class_probs / class_probs.sum())
    
    return np.array(posteriors)
```

**3. Bayesian K-NN:**
```python
def bayesian_knn_predict(X_train, y_train, x_query, k=5, prior_alpha=1.0):
    """
    Bayesian treatment with Dirichlet prior on class proportions
    """
    from scipy.special import digamma
    
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_train)
    _, indices = nn.kneighbors([x_query])
    
    neighbor_labels = y_train[indices[0]]
    classes = np.unique(y_train)
    
    # Count neighbors per class
    counts = np.array([np.sum(neighbor_labels == c) for c in classes])
    
    # Posterior with Dirichlet prior: Dir(alpha + counts)
    posterior_alpha = prior_alpha + counts
    
    # Expected probabilities
    expected_probs = posterior_alpha / posterior_alpha.sum()
    
    # Uncertainty (variance of Dirichlet)
    total = posterior_alpha.sum()
    variance = posterior_alpha * (total - posterior_alpha) / (total**2 * (total + 1))
    
    return expected_probs, np.sqrt(variance)
```

---

## Question 98
- [ ] Done

**How do you implement K-NN for few-shot and zero-shot learning scenarios?**

### Answer

K-NN excels at few-shot learning because it doesn't need training—just provide few labeled examples and classify by similarity. For zero-shot, combine K-NN with attribute/text embeddings to classify unseen classes.

**Few-Shot K-NN:**
```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def few_shot_knn(support_X, support_y, query_X, k=1):
    """
    N-way K-shot classification
    support_X: (N_classes * K_shots, d) support examples
    support_y: (N_classes * K_shots,) labels
    query_X: (n_queries, d) queries to classify
    """
    # For few-shot, often k=1 works best
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(support_X, support_y)
    return knn.predict(query_X)

# Example: 5-way 5-shot
N_classes, K_shots = 5, 5
# support_X has 25 examples, query_X has test examples
predictions = few_shot_knn(support_X, support_y, query_X, k=1)
```

**Prototypical Networks (Few-Shot with Prototypes):**
```python
def prototypical_few_shot(support_X, support_y, query_X):
    """
    Compute class prototypes and classify queries
    """
    classes = np.unique(support_y)
    
    # Compute prototype (mean) for each class
    prototypes = []
    for c in classes:
        prototype = support_X[support_y == c].mean(axis=0)
        prototypes.append(prototype)
    prototypes = np.array(prototypes)
    
    # Classify queries by nearest prototype
    distances = np.linalg.norm(query_X[:, None] - prototypes, axis=2)
    predictions = classes[distances.argmin(axis=1)]
    
    return predictions
```

**Zero-Shot with Attribute Embeddings:**
```python
def zero_shot_knn(train_X, train_attrs, test_X, new_class_attrs, k=5):
    """
    Classify into new classes using attribute descriptions
    
    train_X: (n, d) training features
    train_attrs: (n, a) attribute vectors for training classes
    test_X: (m, d) test features
    new_class_attrs: (c, a) attributes for new classes
    """
    # Learn mapping: features → attributes
    from sklearn.linear_model import Ridge
    
    attr_predictor = Ridge()
    attr_predictor.fit(train_X, train_attrs)
    
    # Predict attributes for test samples
    predicted_attrs = attr_predictor.predict(test_X)
    
    # K-NN in attribute space to new classes
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(new_class_attrs)
    _, class_indices = nn.kneighbors(predicted_attrs)
    
    return class_indices.flatten()

# Example: Classify animals into new species using attributes
# Attributes: [has_fur, has_wings, lives_in_water, ...]
```

**Text-Based Zero-Shot:**
```python
from sentence_transformers import SentenceTransformer

def text_zero_shot(image_embeddings, class_descriptions):
    """
    Zero-shot using text descriptions of classes
    """
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Embed class descriptions
    class_embeddings = text_encoder.encode(class_descriptions)
    
    # K-NN between image and text embeddings
    # (Assumes joint embedding space like CLIP)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1, metric='cosine')
    nn.fit(class_embeddings)
    _, indices = nn.kneighbors(image_embeddings)
    
    return indices.flatten()
```

---

## Question 99
- [ ] Done

**What are the ethical considerations and responsible AI practices for K-NN?**

### Answer

K-NN raises ethical concerns around: data privacy (stores all training data), fairness (reflects biases in training data), transparency (explanation via neighbors), and accountability (whose data drives decisions). Responsible use requires addressing these proactively.

**Key Ethical Considerations:**

| Issue | K-NN Concern | Mitigation |
|-------|--------------|------------|
| **Privacy** | Stores all personal data | Federated K-NN, differential privacy |
| **Fairness** | Reflects historical biases | Audit, rebalancing, fair metrics |
| **Transparency** | Neighbors can be shown | But may reveal sensitive info |
| **Accountability** | Prediction depends on specific individuals | Document data provenance |

**1. Privacy Protection:**
```python
def privacy_aware_knn(X_train, y_train, x_query, k=5, epsilon=1.0):
    """K-NN with differential privacy"""
    from sklearn.neighbors import NearestNeighbors
    
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_train)
    distances, indices = nn.kneighbors([x_query])
    
    # Add Laplace noise to distances
    noisy_distances = distances + np.random.laplace(0, 1/epsilon, distances.shape)
    
    # Re-rank with noisy distances
    noisy_order = np.argsort(noisy_distances[0])
    final_indices = indices[0][noisy_order[:k]]
    
    # Never reveal actual neighbors to user
    return y_train[final_indices]
```

**2. Fairness Auditing:**
```python
def fairness_audit_knn(knn, X_test, y_test, sensitive_attr):
    """Audit K-NN for disparate impact"""
    predictions = knn.predict(X_test)
    
    groups = np.unique(sensitive_attr)
    metrics = {}
    
    for g in groups:
        mask = sensitive_attr == g
        accuracy = (predictions[mask] == y_test[mask]).mean()
        positive_rate = predictions[mask].mean()
        metrics[f'group_{g}_accuracy'] = accuracy
        metrics[f'group_{g}_positive_rate'] = positive_rate
    
    # Check disparate impact
    rates = [metrics[f'group_{g}_positive_rate'] for g in groups]
    disparate_impact = min(rates) / max(rates)
    metrics['disparate_impact_ratio'] = disparate_impact
    
    # 80% rule: ratio should be > 0.8
    metrics['passes_80_percent_rule'] = disparate_impact > 0.8
    
    return metrics
```

**3. Responsible Explanation:**
```python
def responsible_explain(knn, X_train, x_query, k=5, anonymize=True):
    """Explain prediction while protecting privacy"""
    distances, indices = knn.kneighbors([x_query])
    
    if anonymize:
        # Don't reveal actual neighbors
        explanation = {
            'n_neighbors': k,
            'avg_distance': distances.mean(),
            'class_distribution': dict(zip(*np.unique(
                knn._y[indices[0]], return_counts=True
            ))),
            'confidence': knn.predict_proba([x_query]).max()
        }
    else:
        # Full explanation (only if appropriate)
        explanation = {
            'neighbors': X_train[indices[0]],
            'neighbor_labels': knn._y[indices[0]],
            'distances': distances[0]
        }
    
    return explanation
```

**Best Practices:**
- Document data sources and potential biases
- Regular fairness audits
- Privacy impact assessments
- User consent for data used as neighbors
- Right to explanation (show similar cases)

---

## Question 100
- [ ] Done

**What are the best practices for K-NN algorithm selection and implementation?**

### Answer

Best practices for K-NN cover: preprocessing (always scale), K selection (cross-validate), algorithm choice (kd-tree vs ball tree vs brute), metric selection (based on data type), and production considerations (approximate methods for scale).

**Complete K-NN Implementation Checklist:**

**1. Data Preprocessing:**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# Always scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle missing values
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X_scaled)
```

**2. K Selection:**
```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

# Cross-validate to find optimal K
param_grid = {'n_neighbors': range(1, 31, 2)}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=cv,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)
best_k = grid_search.best_params_['n_neighbors']
```

**3. Algorithm Selection:**
```python
# Based on dataset characteristics
n_samples, n_features = X.shape

if n_features < 20:
    algorithm = 'kd_tree'  # Fast for low dimensions
elif n_features < 100:
    algorithm = 'ball_tree'  # Better for higher dimensions
else:
    algorithm = 'brute'  # Or use approximate methods

knn = KNeighborsClassifier(
    n_neighbors=best_k,
    algorithm=algorithm,
    weights='distance',  # Usually better than uniform
    metric='euclidean'  # Or 'cosine' for text/sparse
)
```

**4. Production Pipeline:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Complete pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        algorithm='auto',
        n_jobs=-1  # Parallel
    ))
])

pipeline.fit(X_train, y_train)
```

**5. For Large Scale:**
```python
import faiss
import numpy as np

def production_knn_large_scale(X_train, y_train, k=5):
    """For millions of samples"""
    d = X_train.shape[1]
    
    # Use IVF index
    nlist = min(int(np.sqrt(len(X_train))), 1000)
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    
    index.train(X_train.astype('float32'))
    index.add(X_train.astype('float32'))
    index.nprobe = 10
    
    return index, y_train
```

**Decision Framework:**

| Dataset Size | Dimensions | Recommendation |
|--------------|------------|----------------|
| < 10K | Any | sklearn brute/kd_tree |
| 10K - 1M | < 100 | sklearn ball_tree |
| 10K - 1M | > 100 | FAISS IVF |
| > 1M | Any | FAISS with GPU |

**Summary of Best Practices:**
1. ✅ Always scale features
2. ✅ Cross-validate K (try 1-30, odd numbers)
3. ✅ Use distance weighting
4. ✅ Choose appropriate metric for data type
5. ✅ Use approximate methods for >100K samples
6. ✅ Consider prototype selection to reduce memory
7. ✅ Monitor performance in production

---

## Question 101
- [ ] Done

**Discuss the impact of imbalanced datasets on the K-NN algorithm.**

### Answer

**Impact:**
Imbalanced datasets severely bias K-NN toward the majority class. Since K-NN uses majority voting among K neighbors, and majority class samples are more densely distributed, they dominate the neighborhood of most query points. Result: model predicts majority class almost always, especially for minority class samples.

**Why This Happens:**
- Majority class has higher density in feature space
- More majority samples available to be "nearest"
- Minority class gets outvoted even when query is actually minority

**Example:**
- 95% Class A, 5% Class B
- K=5: Neighbors likely 5 Class A → predicts A even for true B samples

**Solutions (in order of preference):**

**1. Weighted K-NN (First Try)**
```python
# Closer minority neighbors get high weight
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
```
- Why: Close minority neighbor can outvote distant majority

**2. SMOTE (Synthetic Oversampling)**
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
```
- Creates synthetic minority samples

**3. Undersampling Majority**
- Randomly remove majority samples
- Risk: Losing important information

**4. Stratified K Selection**
- Use stratified CV to find optimal K
- Smaller K may help detect local minority patterns

**Evaluation:**
- Never use accuracy alone (misleading)
- Use F1-score, Precision, Recall, AUC-ROC
- Focus on minority class recall

---

## Question 102
- [ ] Done

**How would you explain the concept of locality-sensitive hashing and its relation to K-NN?**

### Answer

**What is LSH:**
Locality-Sensitive Hashing (LSH) is a technique for approximate nearest neighbor search. It uses special hash functions where similar items have high probability of same hash, while dissimilar items have low probability. Instead of searching entire dataset, you only search within matching hash buckets.

**Core Principle:**
$$P(h(x) = h(y) | x \text{ close to } y) >> P(h(x) = h(y) | x \text{ far from } y)$$

**Relation to K-NN:**

| Aspect | Exact K-NN | K-NN with LSH |
|--------|------------|---------------|
| Search | All N points | Only bucket candidates |
| Complexity | O(N) | Sub-linear (nearly O(1)) |
| Accuracy | Exact neighbors | Approximate (may miss some) |
| Scalability | Poor for large N | Excellent |

**How LSH Works:**

**Building Phase:**
1. Create multiple hash tables with LSH functions
2. Hash all training points into buckets
3. Similar points likely share buckets

**Search Phase:**
1. Hash query point
2. Retrieve candidates from matching buckets
3. Brute-force search only among candidates
4. Return K nearest from candidates

**Trade-off:**
- More hash tables → Higher recall, slower
- Fewer hash tables → Lower recall, faster
- Tunable based on accuracy requirements

**When to Use:**
- Millions+ of data points
- Real-time latency requirements
- Acceptable to miss some true neighbors

**Example Applications:**
- Large-scale image retrieval
- Document near-duplicate detection
- Recommendation candidate generation

**Key Insight:**
LSH makes K-NN practical for massive datasets by sacrificing perfect accuracy for dramatic speed improvement.

---

## Question 103
- [ ] Done

**Discuss how missing values in the dataset affect K-NN and how you would handle them.**

### Answer

**Impact of Missing Values:**
K-NN cannot compute distances when feature values are missing. Distance calculation requires all feature values. Missing values cause: (1) complete failure if not handled, (2) incorrect distances if imputed poorly, (3) biased neighbor selection.

**Why K-NN is Particularly Affected:**
- Distance metrics need all dimensions
- Cannot simply ignore missing features (changes distance scale)
- Every prediction requires complete feature vectors

**Handling Strategies (Best to Worst):**

**1. KNNImputer (Recommended)**
- Uses K-NN to impute missing values
- Finds K similar complete samples
- Imputes with mean of their values
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```
**Why Best:** Uses local data structure, preserves relationships

**2. Mean/Median Imputation**
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_train)
```
**Limitation:** Ignores relationships between features

**3. Delete Rows (if very few missing)**
```python
df_clean = df.dropna()
```
**Limitation:** Loses data, may introduce bias

**4. Delete Features (if many missing)**
- Remove feature if >50% missing
- Only if feature not critical

**Pipeline Order:**
1. Impute missing values (fit on train only)
2. Scale features
3. Train K-NN

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
```

**Key Point:** Always impute before scaling, and fit imputer only on training data.

---

## Question 104
- [ ] Done

**Discuss how bootstrap aggregating (bagging) can improve the performance of K-NN.**

### Answer

**Why K-NN Benefits from Bagging:**
K-NN has high variance, especially with small K—predictions are sensitive to specific training samples and noise. Bagging reduces variance by training multiple K-NN models on different bootstrap samples and averaging their predictions.

**How Bagging Works with K-NN:**

**Process:**
1. Create N bootstrap samples (random sampling with replacement)
2. Train separate K-NN on each sample
3. For prediction: get vote from each K-NN
4. Final prediction: majority vote (classification) or average (regression)

**Why It Helps:**
- Single K-NN sensitive to noisy neighbors
- Bagged K-NN: noisy sample affects only some models
- Averaging smooths out noise effects
- More stable, robust predictions

**Implementation:**
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Single K-NN (high variance)
knn_single = KNeighborsClassifier(n_neighbors=5)

# Bagged K-NN (reduced variance)
bagged_knn = BaggingClassifier(
    estimator=KNeighborsClassifier(n_neighbors=5),
    n_estimators=10,      # Number of K-NN models
    max_samples=0.8,      # 80% of samples per model
    bootstrap=True,       # Sample with replacement
    n_jobs=-1
)

bagged_knn.fit(X_train_scaled, y_train)
predictions = bagged_knn.predict(X_test_scaled)
```

**Random Subspace (Feature Bagging):**
Even more effective for K-NN in high dimensions:
```python
bagged_knn = BaggingClassifier(
    estimator=KNeighborsClassifier(n_neighbors=5),
    n_estimators=10,
    max_features=0.7,     # 70% of features per model
    bootstrap_features=True
)
```
**Why:** Each K-NN works in lower-dimensional subspace where distances are more meaningful.

**Benefits:**
- Reduced variance → more stable predictions
- Better handling of noise and outliers
- Mitigates curse of dimensionality (with feature bagging)

**Trade-off:**
- Increased computation (N models instead of 1)
- Slightly reduced interpretability

---

## Question 105
- [ ] Done

**Discuss the role of approximate nearest neighbor search in scaling K-NN for big data.**

### Answer

**The Scaling Problem:**
Exact K-NN has O(N×d) prediction complexity—infeasible for millions/billions of points. For 1 billion points, even millisecond-per-distance means hours per query. Approximate Nearest Neighbor (ANN) trades small accuracy loss for massive speedup.

**Why ANN is Essential:**

| Dataset Size | Exact K-NN | ANN (HNSW) |
|--------------|------------|------------|
| 1 Million | ~1 second | ~1 ms |
| 100 Million | ~100 seconds | ~5 ms |
| 1 Billion | Infeasible | ~10 ms |

**Key ANN Algorithms:**

**1. Locality-Sensitive Hashing (LSH)**
- Hash similar items to same bucket
- Search only within buckets

**2. HNSW (Hierarchical Navigable Small World)**
- Graph-based: nodes connected to similar nodes
- Multi-layer hierarchy for fast navigation
- State-of-the-art for recall vs speed

**3. IVF (Inverted File Index)**
- Cluster data into partitions
- Search only nearest partitions

**4. Product Quantization (PQ)**
- Compress vectors for memory efficiency
- Approximate distance from compressed codes

**Trade-off: Recall vs Speed**
```
Recall = (True neighbors found) / (K neighbors requested)
```
- Higher recall → slower, more accurate
- Lower recall → faster, may miss some neighbors

**Implementation with FAISS:**
```python
import faiss
import numpy as np

# Data: 1 million 128-dim vectors
n_data = 1_000_000
d = 128
data = np.random.random((n_data, d)).astype('float32')

# Build HNSW index (state-of-the-art)
index = faiss.IndexHNSWFlat(d, 32)  # 32 connections per node
index.add(data)

# Fast search
k = 10
query = np.random.random((1, d)).astype('float32')
distances, indices = index.search(query, k)  # ~1 ms for 1M points!
```

**Production Systems Using ANN:**
- Spotify (Annoy): Music recommendations
- Meta (FAISS): Image/video search
- Pinecone, Milvus: Vector databases
- OpenAI (FAISS): Embedding retrieval for RAG

**Key Insight:**
For big data K-NN, the question isn't "exact vs approximate" but "which ANN method and what recall level." Modern ANN achieves 95%+ recall with orders of magnitude speedup.

---
