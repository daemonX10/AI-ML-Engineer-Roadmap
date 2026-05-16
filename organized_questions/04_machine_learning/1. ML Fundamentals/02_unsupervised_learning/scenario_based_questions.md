# Unsupervised Learning - Scenario-Based Questions

## Question 1: Propose an unsupervised learning strategy to segment customers for targeted marketing.
- [ ] Done

### Strategy Pipeline

**Step 1: Feature Engineering (360° Customer View)**

| Category | Features |
|----------|----------|
| **RFM** | Recency, Frequency, Monetary value |
| **Demographic** | Age, location, tenure |
| **Behavioral** | Session length, categories viewed, discount usage |

**Step 2: Preprocessing**
```python
# Handle missing values
X = X.fillna(X.median())

# Scale features
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

# Optional: PCA if many features
```

**Step 3: Clustering**
```python
from sklearn.cluster import KMeans

# Find optimal K
for k in range(2, 11):
    score = silhouette_score(X_scaled, KMeans(n_clusters=k).fit_predict(X_scaled))
    
# Train final model
kmeans = KMeans(n_clusters=optimal_k)
labels = kmeans.fit_predict(X_scaled)
```

**Step 4: Cluster Profiling**
- Calculate mean features per cluster
- Create personas:

| Cluster | Profile | Action |
|---------|---------|--------|
| 1 | Champions: High RFM | Loyalty rewards |
| 2 | At-Risk: High past value, low recency | Re-engagement campaign |
| 3 | Bargain Hunters: High discount affinity | Sale notifications |

**Step 5: Deploy & Monitor**
- Assign new customers to clusters
- Track conversion rates per segment
- Retrain periodically

---

## Question 2: How would you use clustering to inform feature creation in a supervised learning task?
- [ ] Done

### The Approach
Use cluster labels as a new feature for supervised model.

### Implementation

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Fit clustering on training data only
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_train_scaled)

# 3. Create cluster feature for both sets
X_train['cluster_id'] = kmeans.predict(X_train_scaled)
X_test['cluster_id'] = kmeans.predict(X_test_scaled)

# 4. One-hot encode (avoid false ordinal relationship)
encoder = OneHotEncoder(sparse=False)
cluster_features_train = encoder.fit_transform(X_train[['cluster_id']])
cluster_features_test = encoder.transform(X_test[['cluster_id']])

# 5. Concatenate with original features
X_train_augmented = np.hstack([X_train_scaled, cluster_features_train])
```

### Why It Works
- Cluster ID captures complex non-linear interactions
- Acts as high-level summary of feature relationships
- Especially helpful for linear models (adds non-linearity)

### Alternative: Distance Features
Create features for distance to each centroid:
```python
distances = kmeans.transform(X_scaled)  # (n_samples, k)
```

---

## Question 3: Design an approach to group similar documents using unsupervised learning.
- [ ] Done

### Pipeline

**Step 1: Text Preprocessing**
```python
text = text.lower()                    # Lowercase
text = remove_punctuation(text)        # Clean
tokens = tokenize(text)                # Split into words
tokens = remove_stopwords(tokens)      # Remove "the", "a", etc.
tokens = lemmatize(tokens)             # "running" → "run"
```

**Step 2: Vectorization (TF-IDF)**
- Captures word importance (frequent in doc, rare overall)
- Output: Document-term matrix

**Step 3: Dimensionality Reduction (Optional)**
- Apply LSA (SVD on TF-IDF) or PCA
- Reduce to ~100-300 dimensions

**Step 4: Clustering**
- K-means with Elbow/Silhouette for optimal K
- Or DBSCAN if K unknown

**Step 5: Interpretation**
- Analyze top TF-IDF words per cluster
- Assign topic labels: "Sports", "Politics", "Tech"

---

## Question 4: Discuss a framework for detecting communities in social networks via unsupervised learning.
- [ ] Done

### Framework

**Step 1: Graph Representation**
- Nodes = Users
- Edges = Connections (friends, follows, interactions)
- Store as adjacency matrix or list

**Step 2: Choose Algorithm**

**Option A: Louvain Method (Recommended)**
- Maximizes modularity (within-community density vs between)
- Fast, scalable to millions of nodes
- Automatically determines number of communities

**Option B: Spectral Clustering**
- Compute Laplacian eigenvectors
- Project nodes to low-D space
- Run K-means on projections
- Requires specifying K

```python
import networkx as nx
from community import community_louvain  # pip install python-louvain

# Create graph
G = nx.Graph()
G.add_edges_from(edges)

# Detect communities
partition = community_louvain.best_partition(G)

# Analyze
n_communities = len(set(partition.values()))
modularity = community_louvain.modularity(partition, G)
print(f"Communities: {n_communities}, Modularity: {modularity:.3f}")
```

**Step 3: Evaluation**
- **Modularity Score**: Higher = better community structure
- **Analyze characteristics**: Common attributes within communities

**Step 4: Interpretation**
- Label communities by dominant attributes
- "Python Developers in London", "Marketing Professionals"

---

## Question 5: Explain how unsupervised learning could assist in identifying patterns in genomic data.
- [ ] Done

### Answer

Unsupervised learning is extremely valuable in genomics because biological data is high-dimensional, complex, and often lacks clear labels.

**Key Applications:**

| Application | Technique | Purpose |
|-------------|-----------|---------|
| **Gene Expression Clustering** | K-means, Hierarchical Clustering | Group genes with similar expression profiles to identify co-regulated genes |
| **Patient Subtyping** | NMF, Consensus Clustering | Discover disease subtypes (e.g., cancer subtypes) from molecular data |
| **Dimensionality Reduction** | PCA, t-SNE, UMAP | Visualize high-dimensional genomic data (thousands of genes) in 2D/3D |
| **Variant Detection** | Autoencoders, GMMs | Identify rare genetic variants or mutations |
| **Pathway Discovery** | Network Clustering | Find functional modules in gene interaction networks |

**Workflow Example:**
```
Raw Gene Expression Data (20,000+ genes)
    → Preprocessing (normalization, filtering)
    → Dimensionality Reduction (PCA to ~50 components)
    → Clustering (K-means or DBSCAN)
    → Biological Interpretation (pathway enrichment analysis)
```

**Why Unsupervised Over Supervised?**
- **Labels are scarce**: Most genomic data lacks ground-truth annotations
- **Discovery-driven**: The goal is to find *unknown* patterns, not predict known ones
- **Heterogeneity**: Diseases like cancer have subtypes not yet fully catalogued
- **High dimensionality**: Thousands of features with few samples (p >> n)

**Real-World Impact:**
- The Cancer Genome Atlas (TCGA) used unsupervised clustering to identify molecular subtypes of breast cancer (Luminal A, Luminal B, HER2-enriched, Basal-like)
- Single-cell RNA sequencing uses UMAP + Leiden clustering to identify novel cell types
