# Cluster Analysis Interview Questions - Scenario_Based Questions

## Question 1
- [ ] Done

**Discuss feature selection techniques appropriate for cluster analysis.**

### Answer

**Challenge:** No target variable → standard feature importance doesn't apply

**Techniques:**

**1. Filter Methods (Fast, Algorithm-Independent):**

| Technique | Logic |
|-----------|-------|
| Variance Threshold | Remove near-constant features (no separation power) |
| Correlation Filter | Remove highly correlated features (redundant) |
| Laplacian Score | Keep features preserving local structure |

```python
from sklearn.feature_selection import VarianceThreshold

# Remove features with <10% variance
selector = VarianceThreshold(threshold=0.1)
X_filtered = selector.fit_transform(X)
```

**2. Wrapper Methods (Expensive, Algorithm-Specific):**
```
For each feature subset:
    1. Run clustering
    2. Evaluate (silhouette score)
    3. Keep subset with best score

Methods: Forward selection, Backward elimination
```

**3. Embedded Methods (Built into Algorithm):**
- **Subspace Clustering (CLIQUE):** Finds features defining each cluster
- **Regularized Clustering:** L1 penalty drives irrelevant features to zero

**Practical Approach:**
```python
# Step 1: Remove low-variance features
from sklearn.feature_selection import VarianceThreshold
X = VarianceThreshold(threshold=0.01).fit_transform(X)

# Step 2: Remove highly correlated (>0.95)
corr_matrix = pd.DataFrame(X).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]

# Step 3: Dimensionality reduction
from sklearn.decomposition import PCA
X_reduced = PCA(n_components=10).fit_transform(X)
```

---

## Question 2
- [ ] Done

**How would you apply cluster analysis for customer segmentation in a retail business?**

### Answer

**Step-by-Step Strategy:**

**Step 1: Define Business Objective**
```
Goal: Segment customers for targeted marketing
Outcome: 3-5 actionable segments with distinct strategies
```

**Step 2: Feature Engineering (RFM Framework)**
```python
# Calculate RFM features
df['Recency'] = (today - df['last_purchase_date']).dt.days
df['Frequency'] = df.groupby('customer_id')['order_id'].transform('count')
df['Monetary'] = df.groupby('customer_id')['amount'].transform('sum')
```

**Step 3: Preprocessing**
```python
from sklearn.preprocessing import StandardScaler

features = df[['Recency', 'Frequency', 'Monetary']]
X = StandardScaler().fit_transform(features)
```

**Step 4: Determine k and Cluster**
```python
# Elbow + Silhouette to find k
# Then cluster
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42)
df['Segment'] = kmeans.fit_predict(X)
```

**Step 5: Profile Segments**
```python
profile = df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()
print(profile)
```

**Step 6: Name and Act on Segments**

| Segment | Profile | Action |
|---------|---------|--------|
| Champions | Low R, High F, High M | VIP rewards, exclusive access |
| At-Risk | High R, High F, High M | Win-back campaigns |
| New Customers | Low R, Low F, Low M | Onboarding, second purchase incentives |
| Hibernating | High R, Low F, Low M | Reactivation offers |

**Deliverable:**
- Segment membership for each customer
- Segment profiles with averages
- Recommended actions per segment

---

## Question 3
- [ ] Done

**Discuss how cluster analysis can be leveraged for image segmentation.**

### Answer

**Concept:**
Treat each pixel as a data point with features (color, position). Cluster pixels to group similar regions.

**Feature Space:**
```
Each pixel → feature vector:
- Color: (L, A, B) or (R, G, B)  [3D]
- Position: (x, y)               [2D]
- Combined: (L, A, B, x, y)      [5D]
```

**Algorithm Choices:**

| Algorithm | Best For |
|-----------|----------|
| K-Means | Fast, spherical regions, color quantization |
| Mean Shift | Unknown # of regions, irregular shapes |
| DBSCAN | Noise handling, arbitrary shapes |

**Implementation:**
```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Load image
img = np.array(Image.open('image.jpg'))
h, w, c = img.shape

# Create feature matrix (L, A, B, x, y)
from skimage.color import rgb2lab
img_lab = rgb2lab(img)

# Add position features
xx, yy = np.meshgrid(np.arange(w), np.arange(h))
features = np.concatenate([
    img_lab.reshape(-1, 3),
    xx.reshape(-1, 1),
    yy.reshape(-1, 1)
], axis=1)

# Scale features
X = StandardScaler().fit_transform(features)

# Cluster
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X)

# Reshape to image
segmented = labels.reshape(h, w)
```

**Key Consideration:**
Relative scaling between color and position determines whether clustering emphasizes color similarity or spatial proximity.

---

## Question 4
- [ ] Done

**Propose a clustering strategy for identifying similar regions in geographical data.**

### Answer

**Strategy: DBSCAN with Haversine Distance**

**Why DBSCAN?**
- Geographical clusters are density-based (hotspots)
- Arbitrary shapes (coastlines, city boundaries)
- Automatic outlier detection
- No need to specify number of regions

**Why Haversine?**
- Correct distance on Earth's surface
- Euclidean on lat/lon is WRONG (Earth is not flat)

**Implementation:**
```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

# Data: crime locations
locations = df[['latitude', 'longitude']].values

# DBSCAN with Haversine
# eps in radians: 0.5 km ≈ 0.5/6371 radians
eps_km = 0.5  # 500 meters
eps_rad = eps_km / 6371.0

dbscan = DBSCAN(
    eps=eps_rad,
    min_samples=10,
    metric='haversine'  # For lat/lon in radians
)

# Convert to radians
locations_rad = np.radians(locations)
labels = dbscan.fit_predict(locations_rad)

# Visualize on map
import folium
m = folium.Map(location=[lat_center, lon_center], zoom_start=12)
for (lat, lon), label in zip(locations, labels):
    color = 'red' if label == -1 else f'#{label*12345:06x}'[:7]
    folium.CircleMarker([lat, lon], radius=3, color=color).add_to(m)
m.save('clusters.html')
```

**Including Attributes:**
```python
# Combine spatial + non-spatial features
# Scale separately, weight appropriately
spatial = StandardScaler().fit_transform(locations)
attributes = StandardScaler().fit_transform(df[['crime_type_encoded', 'time_of_day']])

# Combine with weighting (spatial more important)
features = np.hstack([spatial * 2, attributes])
```

---

## Question 5
- [ ] Done

**Discuss a potential framework for analyzing social network connectivity using clustering.**

### Answer

**Goal:** Find communities (groups of densely connected users)

**Framework:**

**Step 1: Build Graph**
```python
import networkx as nx

G = nx.Graph()
# Add edges from connections data
for user1, user2, weight in connections:
    G.add_edge(user1, user2, weight=weight)
```

**Step 2: Choose Community Detection Algorithm**

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| Louvain | Modularity optimization, hierarchical | Large networks, fast |
| Spectral | Graph Laplacian eigenvectors | Medium networks |
| Girvan-Newman | Edge betweenness removal | Small networks |
| Label Propagation | Node label spreading | Very large networks |

**Step 3: Detect Communities**
```python
import community as community_louvain

# Louvain method (fast, good quality)
partition = community_louvain.best_partition(G)

# Assign labels
for node, community_id in partition.items():
    G.nodes[node]['community'] = community_id

# Count communities
n_communities = len(set(partition.values()))
print(f"Found {n_communities} communities")
```

**Step 4: Analyze Communities**
```python
# Community sizes
from collections import Counter
sizes = Counter(partition.values())

# Within-community density
for comm_id in set(partition.values()):
    members = [n for n, c in partition.items() if c == comm_id]
    subgraph = G.subgraph(members)
    density = nx.density(subgraph)
    print(f"Community {comm_id}: {len(members)} members, density={density:.3f}")
```

**Step 5: Profile Communities**
- Analyze shared interests, demographics
- Identify influencers (high degree within community)
- Find bridge nodes (connect communities)

**Use Cases:** Content targeting, influence analysis, fake account detection

---

## Question 6
- [ ] Done

**How would you approach clustering time-series data, such as stock market prices or weather patterns?**

### Answer

**Challenge:** Standard distance metrics fail on time-series (phase shifts, different scales)

**Two Main Approaches:**

**Approach 1: Shape-Based (DTW Distance)**
```python
from scipy.spatial.distance import cdist
from dtw import dtw  # pip install dtw-python
from sklearn.cluster import AgglomerativeClustering

def dtw_distance(ts1, ts2):
    """Dynamic Time Warping distance"""
    d, _, _, _ = dtw(ts1, ts2, dist=lambda x, y: abs(x - y))
    return d

# Compute pairwise DTW distance matrix
n = len(time_series)
distance_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        d = dtw_distance(time_series[i], time_series[j])
        distance_matrix[i, j] = d
        distance_matrix[j, i] = d

# Cluster with precomputed distances
clustering = AgglomerativeClustering(
    n_clusters=5,
    metric='precomputed',
    linkage='average'
)
labels = clustering.fit_predict(distance_matrix)
```

**Approach 2: Feature-Based**
```python
from tsfresh import extract_features

# Extract features from each time series
features = []
for ts in time_series:
    stats = {
        'mean': np.mean(ts),
        'std': np.std(ts),
        'trend': np.polyfit(range(len(ts)), ts, 1)[0],
        'autocorr': np.corrcoef(ts[:-1], ts[1:])[0,1],
        'max': np.max(ts),
        'min': np.min(ts)
    }
    features.append(stats)

X = pd.DataFrame(features)
X_scaled = StandardScaler().fit_transform(X)

# Standard clustering on features
labels = KMeans(n_clusters=5).fit_predict(X_scaled)
```

**When to Use Which:**

| Approach | Best For |
|----------|----------|
| DTW (shape-based) | Exact shape matching, phase-invariant |
| Feature-based | Interpretable characteristics (trend, volatility) |

**Hybrid:** Use DTW as one of several features

---

## Question 7
- [ ] Done

**Describe how you would use clustering for organizing a large set of documents into topics.**

### Answer

**Definition:**
Document clustering (topic modeling) groups text documents by thematic similarity. It involves converting text to numerical vectors, then applying clustering to discover natural topic groupings.

**Step-by-Step Procedure:**

**1. Text Preprocessing:**
```
- Tokenization: Split into words
- Lowercase: Convert to lowercase
- Stop words: Remove "the", "is", "a"
- Lemmatization: "running" → "run"
- Punctuation removal
```

**2. Feature Extraction (Vectorization):**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = vectorizer.fit_transform(documents)
```
TF-IDF: Words important in document but rare overall get high scores.

**3. Dimensionality Reduction (Optional but Recommended):**
```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=100)  # LSA
X_reduced = svd.fit_transform(X_tfidf)
```

**4. Clustering:**
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_reduced)
```

**5. Topic Interpretation:**
```python
# Find top words per cluster
for cluster_id in range(5):
    center = kmeans.cluster_centers_[cluster_id]
    top_word_indices = center.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_word_indices]
    print(f"Topic {cluster_id}: {top_words}")
```

**Best Practice:**
Use Cosine similarity (inherent in normalized TF-IDF) rather than Euclidean for text.

---

## Question 8
- [ ] Done

**Explain how you would employ cluster analysis in a recommendation system.**

### Answer

**Definition:**
Clustering enables collaborative filtering in recommendation systems by grouping similar users or items. Users receive recommendations based on what similar users in their cluster liked.

**Two Approaches:**

**1. User-Based Clustering:**
```
Goal: Group users with similar tastes

Steps:
1. Create user-item matrix (rows=users, cols=items, values=ratings)
2. Cluster users based on rating patterns
3. For target user, find their cluster
4. Recommend items popular in that cluster but unseen by user
```

**2. Item-Based Clustering:**
```
Goal: Group similar items

Steps:
1. Create item-user matrix (transpose)
2. Cluster items rated similarly by same users
3. If user likes item X, recommend other items in X's cluster
```

**Python Example (User-Based):**
```python
from sklearn.cluster import KMeans

# user_item_matrix: rows=users, cols=items
kmeans = KMeans(n_clusters=5, random_state=42)
user_clusters = kmeans.fit_predict(user_item_matrix)

# For user_id, find their cluster and recommend
user_cluster = user_clusters[user_id]
similar_users = np.where(user_clusters == user_cluster)[0]
# Aggregate ratings from similar users for unseen items
```

**Handling Sparsity:**
- User-item matrices are very sparse
- Use matrix factorization (SVD) to create dense embeddings first
- Then cluster the dense vectors

**Cold Start Problem:**
New users/items can't be clustered due to lack of data - use content-based fallback.

---
