# Unsupervised Learning - General Questions

## Question 1: What is unsupervised learning and how does it differ from supervised learning?
- [ ] Done

### Definition
Unsupervised learning is a paradigm where algorithms are trained on **unlabeled data** to discover hidden patterns, structures, and relationships without a "teacher" providing correct outputs.

### Key Differences

| Feature | Supervised Learning | Unsupervised Learning |
|---------|--------------------|-----------------------|
| **Input Data** | Labeled (X, y) | Unlabeled (X only) |
| **Goal** | Prediction | Discovery of patterns |
| **Feedback** | Loss function compares predictions to true labels | No direct feedback; learns from data structure |
| **Tasks** | Classification, Regression | Clustering, Dimensionality Reduction, Association Rules |
| **Evaluation** | Objective (accuracy, RMSE) | More subjective (silhouette score) |

### Analogy
- **Supervised**: Learning with flashcards (question + answer)
- **Unsupervised**: Sorting a box of mixed Lego bricks into logical groups

---

## Question 2: Name the main types of problems addressed by unsupervised learning.
- [ ] Done

### The Four Main Problem Types

| Problem Type | Goal | Key Question | Algorithms |
|-------------|------|--------------|------------|
| **Clustering** | Group similar data points | "What natural groupings exist?" | K-means, DBSCAN, Hierarchical, GMM |
| **Dimensionality Reduction** | Reduce features while preserving info | "Can we represent this more simply?" | PCA, t-SNE, Autoencoders |
| **Association Rule Mining** | Find co-occurrence patterns | "What items occur together?" | Apriori, FP-Growth |
| **Anomaly Detection** | Identify rare, unusual points | "Does this look unusual?" | Isolation Forest, LOF, Autoencoders |

### Examples

- **Clustering**: Customer segmentation, document grouping
- **Dimensionality Reduction**: Visualization, feature engineering
- **Association Rules**: Market basket analysis, web usage mining
- **Anomaly Detection**: Fraud detection, network intrusion

---

## Question 3: Explain the concept of dimensionality reduction and why it's important.
- [ ] Done

### Definition
Dimensionality reduction transforms high-dimensional data into lower-dimensional space while retaining meaningful information.

### Why It's Important

| Reason | Explanation |
|--------|-------------|
| **Curse of Dimensionality** | High dimensions → sparse data → poor model performance |
| **Reduce Overfitting** | Fewer features = simpler model = better generalization |
| **Computational Efficiency** | Less data to process = faster training |
| **Visualization** | Reduce to 2D/3D for human understanding |
| **Remove Redundancy** | Eliminate correlated/collinear features |

### Two Approaches
1. **Feature Selection**: Keep subset of original features
2. **Feature Extraction**: Create new features by combining originals (e.g., PCA)

---

## Question 4: What is clustering, and how can it be used to gain insights into data?
- [ ] Done

### Definition
Clustering groups data points into subsets (clusters) where:
- Points **within** a cluster are similar
- Points **across** clusters are dissimilar

### Business Insights from Clustering

| Application | How It's Used | Insight |
|-------------|---------------|---------|
| **Customer Segmentation** | Cluster by purchase behavior | "High-spenders" vs "Bargain hunters" |
| **Anomaly Detection** | Points not in any cluster | Fraud, outliers, errors |
| **Document Grouping** | Cluster text by topic | Organize news, route support tickets |
| **Image Segmentation** | Cluster pixels by color/position | Separate objects from background |

---

## Question 5: Can you discuss the differences between hard and soft clustering?
- [ ] Done

### Definition

| | Hard Clustering | Soft Clustering |
|---|----------------|-----------------|
| **Assignment** | Each point belongs to exactly one cluster | Each point has probability of belonging to each cluster |
| **Output** | Single label per point | Vector of probabilities per point |
| **Example** | K-means, DBSCAN | GMM, Fuzzy C-Means |

### Example Output

**Hard (K-means)**:
- Point A → Cluster 1

**Soft (GMM)**:
- Point A → 70% Cluster 1, 30% Cluster 2

### When to Use Each

| Use Hard When | Use Soft When |
|---------------|---------------|
| Clear, actionable segmentation needed | Clusters naturally overlap |
| Each customer gets one marketing campaign | Need to represent uncertainty |
| Simple interpretation required | Mixed membership is realistic (e.g., genres) |

### Analogy
- **Hard**: Mail goes into one mailbox only
- **Soft**: Song is "70% rock, 20% pop, 10% jazz"

---

## Question 6: What preprocessing steps are suggested before performing unsupervised learning?
- [ ] Done

### Essential Preprocessing Pipeline

**1. Feature Scaling** (Critical for distance-based algorithms)
- Standardization: mean=0, std=1 (most common)
- Min-Max: scale to [0,1]

**2. Handle Missing Values**
- Impute: median (numerical), mode (categorical)
- Advanced: KNN imputation, iterative imputation

**3. Handle Categorical Variables**
- One-hot encoding for nominal features

**4. Dimensionality Reduction**
- PCA after scaling
- Remove low-variance or highly correlated features

**5. Handle Outliers**
- Detect and decide: remove, cap, or use robust algorithm (DBSCAN)

### Typical Pipeline for K-means
```
Missing Values → One-Hot Encode → Scale → PCA → K-means
```

---

## Question 7: How do you handle missing values in an unsupervised learning context?
- [ ] Done

### Strategies by Complexity

**1. Deletion**
- Remove rows with missing values (if < 1-2%)
- Remove columns with > 60% missing

**2. Simple Imputation**
| Feature Type | Method |
|-------------|--------|
| Numerical | Median (robust to outliers) |
| Categorical | Mode (most frequent) |

**3. Advanced Imputation**

**KNN Imputation**:
- Find k similar points based on other features
- Impute from neighbors' values
- Good for unsupervised (uses local structure)

**Iterative Imputation**:
- Treat each feature as target, predict from others
- Iterate until convergence
- Most accurate but slowest

### Recommended Workflow
1. Analyze missingness pattern
2. Start with median/mode baseline
3. Try KNN or iterative for better results
4. Always scale data after imputation

---

## Question 8: Describe the steps you would take to scale and normalize data for clustering.
- [ ] Done

### Why Scale?
Distance-based algorithms (K-means, DBSCAN) are dominated by features with larger scales.

### Steps

**1. Separate Features**
- Identify numerical features to scale
- Handle categorical separately (one-hot encoding)

**2. Choose Scaling Method**

| Method | Formula | Result | When to Use |
|--------|---------|--------|-------------|
| **Standardization** | (x - μ) / σ | mean=0, std=1 | Most common, robust |
| **Min-Max** | (x - min) / (max - min) | [0, 1] | Sensitive to outliers |

**3. Apply Scaling**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Important**: Fit scaler on training data only, then transform test data.

---

## Question 9: Discuss how you could evaluate the performance of a clustering algorithm.
- [ ] Done

### Without Ground Truth (Real-World Scenario)

**Internal Validation Indices**

| Index | Measures | Better |
|-------|----------|--------|
| **Silhouette** | Cohesion vs separation | Higher (max 1) |
| **Davies-Bouldin** | Cluster similarity | Lower |
| **Calinski-Harabasz** | Variance ratio | Higher |

### With Ground Truth (Benchmarking)

**External Validation Indices**

| Index | Description | Perfect Score |
|-------|-------------|---------------|
| **Adjusted Rand Index** | Agreement corrected for chance | 1.0 |
| **Normalized Mutual Information** | Mutual information normalized | 1.0 |
| **V-measure** | Harmonic mean of homogeneity & completeness | 1.0 |

### Practical Strategy

1. **Quantitative**: Use silhouette score to guide K selection
2. **Qualitative**: Analyze cluster characteristics
   - Plot feature distributions per cluster
   - Use domain knowledge to validate meaningfulness
3. **Business Validation**: Do clusters make sense? Are they actionable?

**Key Point**: High silhouette score ≠ useful clusters. Always validate with domain expertise.

---

## Question 10: Explain the importance of feature selection in unsupervised learning.
- [ ] Done

### Why It Matters

| Benefit | Explanation |
|---------|-------------|
| **Better Clusters** | Irrelevant features add noise, obscure true structure |
| **Curse of Dimensionality** | Fewer features = denser, more meaningful space |
| **Interpretability** | Clusters from 5 features easier to explain than 100 |
| **Computational Efficiency** | Faster training |

### Methods for Unsupervised Feature Selection

1. **Variance Threshold**: Remove low-variance features
2. **Correlation Analysis**: Remove highly correlated (redundant) features
3. **PCA Loadings**: Identify most influential original features
4. **Domain Knowledge**: Expert judgment on relevance

---

## Question 11: How would you implement clustering on a large, distributed dataset?
- [ ] Done

### Approach: Use Apache Spark MLlib

For datasets that don't fit in memory, use distributed computing frameworks like Spark.

### Implementation

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# 1. Initialize Spark
spark = SparkSession.builder.appName("DistributedClustering").getOrCreate()

# 2. Load data from distributed storage
df = spark.read.csv("s3://bucket/large_data.csv", header=True, inferSchema=True)

# 3. Assemble features into vector column
feature_cols = ["col1", "col2", "col3"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
df = assembler.transform(df)

# 4. Scale features
scaler = StandardScaler(inputCol="features_raw", outputCol="features")
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# 5. Run distributed K-means
kmeans = KMeans(featuresCol="features", k=5, seed=42)
model = kmeans.fit(df)

# 6. Get predictions
predictions = model.transform(df)

# 7. Evaluate with silhouette score
evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette Score: {silhouette:.4f}")
```

### Key Points
- Spark distributes data across cluster nodes
- MLlib provides parallel K-means, Bisecting K-means, LDA
- Only 2 passes over data needed

---

## Question 12: Describe a scenario where unsupervised learning could add value to a business process.
- [ ] Done

### Scenario: E-commerce Customer Segmentation

**Problem**: One-size-fits-all marketing is inefficient

**Solution**: K-means clustering on customer behavior

**Features Used (RFM)**:
- Recency: Days since last purchase
- Frequency: Number of purchases
- Monetary: Total spend

**Process**:
1. Scale features
2. Run K-means, use silhouette score for optimal K
3. Analyze cluster characteristics

**Resulting Segments**:
- **Champions**: High frequency, high value, recent
- **At-Risk**: High past value, but long since purchase
- **Bargain Hunters**: Medium frequency, buys only on sale

**Business Action**:
- Champions → Loyalty rewards
- At-Risk → Re-engagement campaign
- Bargain Hunters → Sale notifications

---

## Question 13: Discuss how unsupervised learning can be used in image segmentation.
- [ ] Done

### Core Idea
Treat each pixel as a data point → Cluster pixels → Segments = Clusters

### Feature Representation

| Feature | Description |
|---------|-------------|
| **Color** | (R, G, B) per pixel |
| **Spatial** | (x, y) coordinates |
| **Combined** | (R, G, B, x, y) for color + location |

### Algorithm Choice

**K-means**:
- Choose K (number of segments)
- Cluster by color/position
- Replace pixels with centroid color
- Good for color quantization

**DBSCAN / Mean-Shift**:
- No K needed
- Better for irregular shapes
- Can identify varying density regions

### Example: Background Removal
1. Represent pixels by RGB
2. K-means with K=2
3. One cluster = foreground, one = background
4. Create mask to separate

### Limitations
- Simple approach; deep learning (U-Net) is state-of-the-art for complex segmentation
- Works well for color-based separation, less for semantic understanding

---

## Question 14: Explain how recommendation systems utilize unsupervised learning techniques.
- [ ] Done

### Collaborative Filtering (Unsupervised)

**Core Idea**: "Users who liked similar items will like similar items in future"

**Data**: User-item interaction matrix (ratings, purchases, clicks)

### Types

**1. Memory-Based**
- **User-Based**: Find similar users → recommend what they liked
- **Item-Based**: Find similar items to what user liked

**2. Model-Based (Matrix Factorization)**
- Decompose sparse user-item matrix into: **User-Factor** × **Item-Factor**
- Discovers latent factors (genres, styles) automatically
- Predict rating = dot product of user and item factor vectors

### Why It's Unsupervised
- No explicit labels
- Learns "similarity" and "latent tastes" from patterns in interaction data

---

## Question 15: How can unsupervised learning be applied to anomaly detection?
- [ ] Done

### Core Idea
Learn what "normal" looks like → Flag points that don't fit

### Approaches

**1. Clustering-Based**
- **DBSCAN**: Points labeled "noise" are anomalies
- **K-means**: Points far from their centroid are suspicious

**2. Density-Based**
- **Local Outlier Factor (LOF)**: Compares local density to neighbors
- Low density relative to neighbors = outlier

**3. Reconstruction-Based (Autoencoders)**
- Train on normal data only
- High reconstruction error = anomaly (never seen before)

**4. Isolation-Based**
- **Isolation Forest**: Anomalies are easier to isolate
- Fewer random splits needed = more anomalous

### Why Unsupervised?
No labeled anomaly examples needed - learns "normal" from data structure.
