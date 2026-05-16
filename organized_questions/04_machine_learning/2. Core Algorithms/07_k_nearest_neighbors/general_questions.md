# K Nearest Neighbors Interview Questions - General Questions

## Question 1
- [ ] Done

**What does the 'K' in K-NN stand for, and how do you choose its value?**

### Answer

'K' is the number of nearest neighbors considered for voting/averaging. Choose K via cross-validation: test range of values, select K with best validation score. Rule of thumb: $K = \sqrt{N}$. Always use odd K for binary classification to avoid ties.

**K Selection Guidelines:**

| K Value | Effect | Risk |
|---------|--------|------|
| Small (1-3) | Flexible boundary | Overfitting, noise-sensitive |
| Large (>√N) | Smooth boundary | Underfitting |
| Optimal | Balances both | Found via CV |

**Selection Process:**
```python
from sklearn.model_selection import cross_val_score
import numpy as np

# Test range of K values
k_range = range(1, 31, 2)  # Odd values
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_score = cross_val_score(knn, X_train, y_train, cv=5).mean()
    scores.append(cv_score)

# Best K
best_k = k_range[np.argmax(scores)]
```

**Interview Tips:**
- Binary classification → always odd K
- Start with $K = \sqrt{N}$
- Smaller K = more variance, larger K = more bias

---

## Question 2
- [ ] Done

**List the pros and cons of using the K-NN algorithm.**

### Answer

**Pros:**
- **Simple and intuitive**: Easy to understand and implement
- **No training phase**: Just stores data (lazy learning)
- **Non-parametric**: No assumptions about data distribution
- **Naturally handles multi-class**: No modification needed
- **Adaptable**: Works for classification and regression
- **Interpretable**: Can show which neighbors influenced decision

**Cons:**
- **Slow prediction**: O(N×d) for each prediction
- **Memory intensive**: Stores entire training set
- **Curse of dimensionality**: Fails in high dimensions
- **Sensitive to scale**: Requires feature scaling
- **Sensitive to irrelevant features**: Noise in distance calculation
- **Struggles with imbalanced data**: Majority class dominates
- **No feature importance**: Doesn't indicate which features matter

**Summary Table:**

| Aspect | Pro | Con |
|--------|-----|-----|
| Training | O(1) - instant | N/A |
| Prediction | Simple voting | O(N×d) - slow |
| Memory | N/A | Stores all data |
| High dimensions | N/A | Performance degrades |
| Interpretability | Shows neighbors | No feature weights |

**When to Use K-NN:**
- Small to medium datasets
- Low to moderate dimensions
- When interpretability of neighbors matters
- As a baseline model

---

## Question 3
- [ ] Done

**In what kind of situations is K-NN not an ideal choice?**

### Answer

Avoid K-NN when: (1) high-dimensional data (curse of dimensionality), (2) large datasets (slow prediction), (3) real-time/low-latency requirements, (4) highly imbalanced classes, (5) many irrelevant features, (6) limited memory for storing data.

**K-NN is NOT ideal for:**

| Situation | Why Problematic | Better Alternative |
|-----------|-----------------|-------------------|
| High dimensions (d > 20) | Distances become meaningless | Neural networks, tree-based |
| Large N (millions) | O(N) prediction too slow | Use ANN or different algorithm |
| Real-time systems | Latency too high | Pre-trained models |
| Imbalanced data | Majority class dominates | Tree-based with class weights |
| Many irrelevant features | Noise overwhelms signal | Feature selection first |
| Memory constraints | Stores entire dataset | Model-based algorithms |
| Streaming data | Cannot incrementally update | Online learning algorithms |

**Rule of Thumb:**
- If d > 20: Reduce dimensions first or use different algorithm
- If N > 100K: Consider ANN methods or different algorithm
- If prediction latency matters: Pre-compute or use eager learners

**Better Alternatives:**
- Large data: Random Forest, XGBoost
- High dimensions: Neural Networks
- Interpretability needed: Decision Trees
- Speed critical: Logistic Regression, Naive Bayes

---

## Question 4
- [ ] Done

**Explore the differences between K-NN and Radius Neighbors.**

### Answer

K-NN uses fixed K neighbors regardless of distance; Radius Neighbors uses all points within fixed radius r. K-NN has consistent neighbor count but varying "reach"; Radius has consistent reach but varying neighbor count (can be zero in sparse regions).

**Key Differences:**

| Aspect | K-NN | Radius Neighbors |
|--------|------|------------------|
| Parameter | K (neighbor count) | r (radius) |
| Neighbor count | Always K | Variable (0 to N) |
| Sparse regions | Reaches far to find K | May find no neighbors |
| Dense regions | Uses only K nearest | Uses all within radius |
| Consistency | Fixed count | Fixed distance |

**Radius Neighbors Behavior:**
- Dense region: Many neighbors contribute (more stable)
- Sparse region: Few/no neighbors (can fail to predict)

**Python Code:**
```python
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

# K-NN: Always uses 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Radius: Uses all within radius 1.0
radius_nn = RadiusNeighborsClassifier(radius=1.0, outlier_label='most_frequent')
```

**When to Use Which:**
- **K-NN**: Most cases, more robust
- **Radius Neighbors**: When local density matters, or need density-aware predictions

**Caveat:**
Radius Neighbors can fail if no points within radius. Use `outlier_label` parameter to handle.

---

## Question 5
- [ ] Done

**If you have a large dataset, how can you make K-NN's computation faster?**

### Answer

Speed up K-NN with: (1) kd-tree/ball tree for O(log N) search, (2) Approximate Nearest Neighbors (FAISS, Annoy), (3) dimensionality reduction, (4) data sampling/prototyping, (5) parallelization (n_jobs=-1), (6) GPU acceleration (cuML).

**Optimization Strategies:**

| Method | Speedup | Trade-off |
|--------|---------|-----------|
| kd-tree | O(log N) | Fails d > 20 |
| Ball tree | O(log N) | Better for high d |
| ANN (FAISS) | Sub-linear | Approximate |
| PCA | Reduces d | Some info loss |
| Prototype selection | Reduces N | Some accuracy loss |
| GPU (cuML) | 10-100x | Hardware needed |

**Python Code:**
```python
# 1. Use optimized algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', n_jobs=-1)

# 2. Reduce dimensions first
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('pca', PCA(n_components=20)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

# 3. For massive scale: FAISS
import faiss
d = features.shape[1]
index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 100)
index.train(features)
index.add(features)
D, I = index.search(query, k=5)  # Fast approximate search
```

**Decision Guide:**
- N < 10K: Brute force is fine
- N < 100K: kd-tree/ball tree
- N > 100K: FAISS or other ANN

---

## Question 6
- [ ] Done

**How do you assess the similarity between instances in K-NN?**

### Answer

Similarity is assessed via distance metrics. Common choices: Euclidean (straight-line), Manhattan (city-block), Cosine (angle between vectors), Hamming (categorical mismatches). Choice depends on data type and problem semantics. Lower distance = higher similarity.

**Distance Metrics:**

| Metric | Formula | Use Case |
|--------|---------|----------|
| Euclidean | $\sqrt{\sum(x_i-y_i)^2}$ | Continuous, low-d |
| Manhattan | $\sum\|x_i-y_i\|$ | High-d, robust to outliers |
| Cosine | $1 - \frac{x \cdot y}{\|x\|\|y\|}$ | Text, embeddings |
| Hamming | Count of differences | Categorical |
| Minkowski | $(\sum\|x_i-y_i\|^p)^{1/p}$ | Tunable (p parameter) |

**Similarity vs Distance:**
$$\text{Similarity} \propto \frac{1}{\text{Distance}}$$

**Python Code:**
```python
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import numpy as np

x = np.array([[1, 2, 3]])
y = np.array([[4, 5, 6]])

# Distance (lower = more similar)
euclidean_dist = euclidean_distances(x, y)

# Similarity (higher = more similar)
cosine_sim = cosine_similarity(x, y)

# In K-NN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(metric='euclidean')  # or 'manhattan', 'cosine'
```

**Selection Guide:**
- Numerical continuous: Euclidean (after scaling)
- High dimensions: Manhattan
- Text/embeddings: Cosine
- Categorical: Hamming or Gower

---

## Question 7
- [ ] Done

**Outline strategies you would use to select an appropriate distance metric for K-NN.**

### Answer

Strategy: (1) consider data type—numerical vs categorical vs mixed, (2) consider problem semantics—magnitude vs direction matters, (3) treat metric as hyperparameter and tune via cross-validation, (4) consider dimensionality—Manhattan better for high-d.

**Selection Framework:**

**Step 1: Assess Data Type**
| Data Type | Recommended Metric |
|-----------|-------------------|
| Numerical continuous | Euclidean, Manhattan |
| High-dimensional numerical | Manhattan |
| Text/embeddings | Cosine |
| Categorical | Hamming |
| Mixed | Gower or one-hot + Euclidean |

**Step 2: Consider Semantics**
- Magnitude matters: Euclidean/Manhattan
- Direction matters: Cosine
- Binary features: Jaccard

**Step 3: Tune via Cross-Validation**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

param_grid = {
    'n_neighbors': [3, 5, 7],
    'metric': ['euclidean', 'manhattan', 'cosine', 'chebyshev']
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

print(f"Best metric: {grid.best_params_['metric']}")
```

**Step 4: Validate Choice**
- Check if selected metric aligns with domain knowledge
- Ensure scaling is applied for Euclidean/Manhattan

**Interview Tip:**
Always mention that metric choice is problem-dependent and should be validated empirically.

---

## Question 8
- [ ] Done

**How can ensemble methods be used in conjunction with K-NN?**

### Answer

Main approaches: (1) Bagging—train multiple K-NN on bootstrap samples, average predictions for lower variance, (2) Random Subspace—each K-NN uses random feature subset, handles high-d well, (3) Stacking—use K-NN as base learner combined with other models.

**Ensemble Approaches:**

**1. Bagging with K-NN**
- Multiple K-NN on different data samples
- Majority vote reduces variance
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

bagged_knn = BaggingClassifier(
    estimator=KNeighborsClassifier(n_neighbors=5),
    n_estimators=10,
    max_samples=0.8,
    n_jobs=-1
)
```

**2. Random Subspace (Feature Bagging)**
- Each K-NN uses random features
- Effective for high-dimensional data
```python
bagged_knn = BaggingClassifier(
    estimator=KNeighborsClassifier(n_neighbors=5),
    n_estimators=10,
    max_features=0.7,  # 70% of features per model
    bootstrap_features=True
)
```

**3. Stacking with K-NN**
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

stacking = StackingClassifier(
    estimators=[
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('rf', RandomForestClassifier())
    ],
    final_estimator=LogisticRegression()
)
```

**Benefits:**
- Reduces K-NN's variance
- More robust predictions
- Random subspace handles curse of dimensionality

---

## Question 9
- [ ] Done

**Explore the use of K-NN for outlier detection and the rationale behind it.**

### Answer

Rationale: Outliers have sparse neighborhoods—their nearest neighbors are far away. K-NN measures distance to K-th neighbor; if large, point is likely an outlier. More sophisticated: Local Outlier Factor (LOF) compares local density to neighbors' density for relative outlierness.

**K-NN for Outlier Detection:**

**Simple Approach: K-th Neighbor Distance**
- Compute distance to K-th nearest neighbor
- Large distance → isolated → outlier

**Advanced: Local Outlier Factor (LOF)**
- Compares point's density to its neighbors' density
- LOF > 1: Less dense than neighbors (outlier)
- LOF ≈ 1: Similar density (inlier)

**Python Code:**
```python
# Method 1: Simple K-th neighbor distance
from sklearn.neighbors import NearestNeighbors
import numpy as np

nn = NearestNeighbors(n_neighbors=5)
nn.fit(X)
distances, _ = nn.kneighbors(X)

# Flag points with large K-th neighbor distance
threshold = np.percentile(distances[:, -1], 95)
outliers = distances[:, -1] > threshold

# Method 2: Local Outlier Factor
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
predictions = lof.fit_predict(X)  # -1 = outlier, 1 = inlier

# Get outlier scores
outlier_scores = -lof.negative_outlier_factor_
```

**Why K-NN Works for Outliers:**
- Distance-based: outliers are naturally far from others
- Local approach: adapts to varying densities
- No assumptions about data distribution

**Use Cases:** Fraud detection, network intrusion, defect detection

---

## Question 10
- [ ] Done

**Compare and contrast the use of K-NN in a supervised context versus its use in unsupervised learning (e.g., clustering).**

### Answer

Supervised K-NN: Uses labeled data, finds K neighbors, predicts via voting (classification) or averaging (regression). Unsupervised: K-NN concept used for density estimation, outlier detection, or as similarity measure in clustering—no labels required.

**Comparison:**

| Aspect | Supervised K-NN | Unsupervised K-NN Use |
|--------|-----------------|----------------------|
| Labels | Required | Not used |
| Goal | Predict class/value | Find structure/outliers |
| Output | Classification/Regression | Density, similarity |
| Example | KNeighborsClassifier | LOF, KNNImputer |

**Supervised K-NN:**
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)  # Needs labels
prediction = knn.predict(X_test)
```

**Unsupervised Applications:**

**1. Outlier Detection (LOF):**
```python
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=20)
outliers = lof.fit_predict(X)  # No labels
```

**2. Imputation:**
```python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)  # No labels
```

**3. As similarity for clustering:**
- Build K-NN graph
- Use for spectral clustering

**Key Distinction:**
- K-NN classifier: Supervised prediction algorithm
- K-NN as tool: Distance/similarity calculation for various tasks

**Note:** K-Means (clustering) is different from K-NN despite similar names.

---

## Question 11
- [ ] Done

**Summarize the main ideas of a few recent research papers on improving the K-NN algorithm.**

### Answer

Recent K-NN research focuses on: (1) HNSW (graph-based ANN) for billion-scale search, (2) Deep Metric Learning (Siamese/Triplet networks) for learned similarity, (3) Neural K-NN (differentiable neighbors), (4) GPU-accelerated implementations for real-time performance.

**Key Research Directions:**

**1. HNSW (Hierarchical Navigable Small World)**
- Multi-layer graph structure for ANN search
- Achieves near-perfect recall with sub-millisecond latency
- Industry standard (FAISS, Milvus use it)

**2. Deep Metric Learning**
- Learn embedding where similar items are close
- Siamese Networks: Twin networks compare pairs
- Triplet Networks: Learn anchor-positive-negative relationships
- Applications: Face recognition, image retrieval

**3. Neural K-NN / Differentiable KNN**
- Make K-NN differentiable for end-to-end learning
- Replace softmax with K-NN retrieval in neural nets
- Improves robustness and calibration

**4. Product Quantization Advances**
- Better compression with Optimized PQ
- Enables billion-scale search on single GPU

**5. Hardware-Optimized K-NN**
- RAPIDS cuML: GPU-accelerated K-NN
- Custom FPGA implementations for edge devices

**Practical Impact:**
- Billion-scale vector search (Pinecone, Milvus)
- Real-time recommendation systems
- RAG (Retrieval Augmented Generation) in LLMs

**Interview Tip:**
Mention that modern K-NN research is driven by embedding search and RAG applications.

---

## Question 12
- [ ] Done

**How can K-NN be combined with deep learning techniques?**

### Answer

Combinations: (1) Use CNN/transformers as feature extractors, apply K-NN on embeddings, (2) Deep metric learning—learn embedding space optimized for K-NN, (3) Replace softmax classifier with K-NN retrieval, (4) K-NN graph as input to Graph Neural Networks.

**Integration Approaches:**

**1. Deep Features + K-NN**
- Extract features with pretrained CNN
- Apply K-NN on learned embeddings
```python
from tensorflow.keras.applications import ResNet50

# Feature extractor
encoder = ResNet50(weights='imagenet', include_top=False, pooling='avg')
features = encoder.predict(images)

# K-NN on deep features
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn.fit(features, labels)
```

**2. Deep Metric Learning**
- Train network to learn optimal embedding
- Siamese/Triplet loss pulls similar items together
- Then apply K-NN in learned space

**3. K-NN as Classification Layer**
- Replace softmax with K-NN retrieval
- More robust, better calibration
- Used in few-shot learning

**4. K-NN Graph for GNNs**
- Build K-NN graph from point cloud
- Feed to Graph Neural Network

**Applications:**
- Image retrieval (deep features + ANN)
- Face recognition (metric learning)
- Few-shot learning (prototype networks)
- RAG in LLMs (embedding search)

**Why Combine?**
- Deep learning: Learns good representations
- K-NN: Simple, interpretable classification
- Best of both worlds

---
