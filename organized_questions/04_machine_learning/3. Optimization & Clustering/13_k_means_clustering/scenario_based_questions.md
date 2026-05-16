# K Means Clustering Interview Questions - Scenario_Based Questions

## Question 1
- [ ] Done

**Discuss how you would use K-Means Clustering for image compression.**

**Answer:**

**Scenario:** A 1000x1000 RGB image uses 24 bits per pixel (16.7M colors possible). You need to reduce file size while maintaining visual quality.

**Approach: Color Quantization with K-Means**

Reduce 16.7M possible colors to K representative colors.

**How It Works:**
```
1. Reshape image: (1000, 1000, 3) → (1M, 3) pixels
2. Cluster pixels by RGB values with K-Means
3. Replace each pixel with its centroid color
4. Store: K centroids + index per pixel
```

**Compression Math:**
```
Original: 1M pixels × 24 bits = 24 Mbits
Compressed (K=16): 
  - Indices: 1M × 4 bits = 4 Mbits
  - Codebook: 16 × 24 bits = 384 bits
  - Total: ~4 Mbits → 6x compression!
```

**Implementation:**
```python
import cv2
import numpy as np
from sklearn.cluster import KMeans

def compress_image(image, n_colors=16):
    # Reshape to pixel list
    pixels = image.reshape(-1, 3)
    
    # Cluster colors
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    labels = kmeans.fit_predict(pixels)
    
    # Replace with centroid colors
    compressed = kmeans.cluster_centers_[labels]
    compressed = compressed.reshape(image.shape).astype(np.uint8)
    
    return compressed, kmeans.cluster_centers_

# Usage
img = cv2.imread('photo.jpg')
compressed, palette = compress_image(img, n_colors=32)
```

**Trade-off:**
- More colors (K) → Better quality, less compression
- Fewer colors (K) → More compression, posterization artifacts

---

## Question 2
- [ ] Done

**How would you apply K-Means clustering for anomaly detection?**

**Answer:**

**Scenario:** Detect fraudulent transactions in banking data. Normal transactions should cluster together; anomalies should be far from any cluster center.

**Approach: Distance-Based Anomaly Detection**

Points far from ALL centroids are potential anomalies.

**Method:**
```
1. Cluster data with K-Means
2. For each point, calculate distance to its centroid
3. Points with distance > threshold are anomalies
4. Threshold: percentile-based or domain-defined
```

**Implementation:**
```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def detect_anomalies(X, n_clusters=5, percentile=95):
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    # Calculate distances to assigned centroid
    distances = np.min(kmeans.transform(X_scaled), axis=1)
    
    # Define threshold
    threshold = np.percentile(distances, percentile)
    
    # Flag anomalies
    anomalies = distances > threshold
    
    return anomalies, distances, threshold

# Usage
anomalies, distances, threshold = detect_anomalies(transactions)
print(f"Detected {anomalies.sum()} anomalies")
```

**Alternative: Small Cluster Method**
- Anomalies may form their own tiny clusters
- Flag clusters with very few members

**Considerations:**
| Aspect | Recommendation |
|--------|---------------|
| Threshold | 95th-99th percentile or domain-specific |
| K selection | Use silhouette, avoid too many clusters |
| Better alternatives | DBSCAN naturally labels noise, Isolation Forest |

**Interview Note:**
"K-Means anomaly detection is simple but effective. For production, I'd also consider Isolation Forest or DBSCAN which are designed for anomaly detection."

---

## Question 3
- [ ] Done

**How would you leverage K-Means clustering in designing a content delivery network?**

**Answer:**

**Scenario:** Design a CDN with K edge servers to minimize latency for millions of globally distributed users.

**Problem Formulation:**
- Users = data points (lat, long, request volume)
- Edge servers = centroids
- Goal: Minimize total distance users travel to reach nearest server

**This is exactly what K-Means optimizes!**

**Approach:**
```
1. Collect user locations (weighted by traffic volume)
2. Run K-Means with K = number of servers
3. Centroids = optimal server locations
4. Each user connects to nearest server (cluster assignment)
```

**Implementation:**
```python
import numpy as np
from sklearn.cluster import KMeans

def optimize_cdn_locations(user_locations, traffic_weights, n_servers):
    """
    user_locations: [[lat, long], ...]
    traffic_weights: importance of each user
    n_servers: number of edge servers to place
    """
    # Weight users by traffic (repeat high-traffic locations)
    weighted_locations = np.repeat(
        user_locations, 
        (traffic_weights / traffic_weights.min()).astype(int),
        axis=0
    )
    
    # K-Means to find optimal server locations
    kmeans = KMeans(n_clusters=n_servers, random_state=42)
    kmeans.fit(weighted_locations)
    
    server_locations = kmeans.cluster_centers_
    user_assignments = kmeans.predict(user_locations)
    
    return server_locations, user_assignments

# Usage
server_locs, assignments = optimize_cdn_locations(
    user_locations, traffic_weights, n_servers=10
)
```

**Considerations:**
- Weight by traffic volume (high-traffic areas get priority)
- Consider network topology, not just geographic distance
- Real CDNs add constraints: data center availability, costs

**Result:**
Servers placed to minimize total weighted user-to-server distance → optimal latency.

---

## Question 4
- [ ] Done

**How can K-Means be applied to segment customers in a retail business?**

**Answer:**

Use RFM (Recency, Frequency, Monetary) features or behavioral data. Steps: (1) Extract features (purchase history, demographics, behavior); (2) Scale features; (3) Find optimal K (elbow/silhouette); (4) Cluster customers; (5) Profile segments; (6) Develop segment-specific strategies. Result: actionable customer segments for targeted marketing.

**RFM Framework:**
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create RFM features
rfm = df.groupby('customer_id').agg({
    'order_date': lambda x: (today - x.max()).days,  # Recency
    'order_id': 'count',                              # Frequency
    'amount': 'sum'                                   # Monetary
}).rename(columns={
    'order_date': 'recency',
    'order_id': 'frequency',
    'amount': 'monetary'
})

# Scale and cluster
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

kmeans = KMeans(n_clusters=4, random_state=42)
rfm['segment'] = kmeans.fit_predict(rfm_scaled)
```

**Typical Segments:**

| Segment | Profile | Strategy |
|---------|---------|----------|
| Champions | High R, F, M | Loyalty rewards |
| At-Risk | Low R, High F, M | Win-back campaigns |
| New | High R, Low F | Onboarding |
| Lost | Low R, F, M | Re-engagement or let go |

---

## Question 5
- [ ] Done

**Outline a strategy to cluster documents based on their textual content using K-Means.**

**Answer:**

Pipeline: (1) **Preprocess text** (lowercase, remove stopwords, lemmatize); (2) **Vectorize** using TF-IDF (captures term importance); (3) **Optional: reduce dimensions** (TruncatedSVD/LSA); (4) **Normalize vectors** (L2 for cosine similarity); (5) **Apply K-Means**; (6) **Analyze cluster keywords**. Normalization makes Euclidean K-Means approximate cosine-based clustering.

**Full Pipeline:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        max_df=0.5,
        min_df=2
    )),
    ('svd', TruncatedSVD(n_components=100)),  # LSA
    ('normalizer', Normalizer()),              # For cosine
    ('kmeans', KMeans(n_clusters=5))
])

labels = pipeline.fit_predict(documents)
```

**Analyze Clusters:**
```python
# Get top terms per cluster
terms = tfidf.get_feature_names_out()
for i, centroid in enumerate(kmeans.cluster_centers_):
    top_idx = centroid.argsort()[-10:][::-1]
    print(f"Cluster {i}: {[terms[j] for j in top_idx]}")
```

**Key Points:**
- TF-IDF > raw counts (handles term importance)
- Normalize for cosine-like similarity
- LSA helps with synonymy and noise

---

## Question 6
- [ ] Done

**Provide an example of using K-Means clustering for market trend analysis.**

**Answer:**

Use K-Means to cluster stocks or products based on performance patterns. Example: Cluster stocks by returns, volatility, and sector exposure to identify market regimes or investment groups. This helps portfolio construction, risk management, and identifying similar assets. Can also cluster time periods to identify market regimes.

**Stock Clustering Example:**
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Features for each stock
stock_features = pd.DataFrame({
    'avg_return': returns.mean(),
    'volatility': returns.std(),
    'beta': betas,
    'momentum': momentum_scores
})

# Scale and cluster
scaler = StandardScaler()
X_scaled = scaler.fit_transform(stock_features)

kmeans = KMeans(n_clusters=4, random_state=42)
stock_features['cluster'] = kmeans.fit_predict(X_scaled)
```

**Resulting Clusters:**

| Cluster | Profile | Interpretation |
|---------|---------|---------------|
| 0 | Low return, low volatility | Defensive stocks |
| 1 | High return, high volatility | Growth stocks |
| 2 | Negative return, high beta | Distressed |
| 3 | Moderate return, low beta | Value stocks |

**Applications:**
- Portfolio diversification (select from different clusters)
- Regime detection (cluster time periods)
- Peer comparison (find similar companies)

---

## Question 7
- [ ] Done

**Describe a scenario where K-Means clustering was effectively applied to solve a real-world problem.**

**Answer:**

**Customer Segmentation in E-commerce:**

An e-commerce company wants to personalize marketing campaigns. Using K-Means on customer features (purchase frequency, average order value, product categories, recency), they segment customers into distinct groups like "high-value loyal", "bargain hunters", "new customers", enabling targeted promotions that increased conversion by 25%.

**How K-Means Was Applied:**

```
1. Data Collection: Customer transaction history
2. Feature Engineering:
   - RFM: Recency, Frequency, Monetary value
   - Category preferences, average basket size
3. Preprocessing: Standardize features (StandardScaler)
4. Find K: Elbow method + Silhouette → K=5
5. Cluster: K-Means on scaled features
6. Profile: Analyze each cluster's characteristics
7. Action: Tailor marketing per segment
```

**Resulting Segments:**
- Cluster 0: VIP customers → Premium offers
- Cluster 1: Price-sensitive → Discount campaigns
- Cluster 2: New users → Onboarding emails
- Cluster 3: Dormant → Re-engagement
- Cluster 4: One-time buyers → Loyalty incentives

**Other Real-World Applications:**
- Image compression (color quantization)
- Document/news article clustering
- Fraud detection (unusual transaction patterns)
- Medical diagnosis (patient grouping)

---

## Question 8
- [ ] Done

**Explain how you can apply K-Means Clustering to the problem of load balancing in distributed computing.**

**Answer:**

K-Means can optimize load balancing by clustering tasks/requests based on resource requirements (CPU, memory, latency), then assigning each cluster to a server. This groups similar workloads together, ensuring servers handle homogeneous tasks efficiently and load is distributed based on actual resource patterns rather than round-robin.

**How It Works:**

```
1. Feature Extraction: For each task/request:
   - CPU usage, memory requirement
   - Expected execution time
   - Data locality, I/O intensity

2. Cluster Tasks: K-Means with K = number of servers
   - Similar tasks grouped together

3. Assign Clusters to Servers:
   - Each cluster → one server
   - Server specializes in that workload type

4. Route New Requests:
   - Compute features of new request
   - Find nearest centroid
   - Route to corresponding server
```

**Benefits:**
- Homogeneous workloads per server → better caching
- Predictable resource allocation
- Reduced context switching
- Better capacity planning

**Considerations:**
- Re-cluster periodically as workload changes
- Handle cluster imbalance (some clusters larger)
- Combine with traditional load metrics

---

## Question 9
- [ ] Done

**Describe a project where K-Means contributed to improving recommendation systems.**

**Answer:**

K-Means improves recommendations by clustering users with similar preferences, then recommending items popular within a user's cluster. This is "cluster-based collaborative filtering." For a streaming service, users clustered by viewing history receive recommendations based on what similar users in their cluster enjoyed, solving the cold-start problem for new items.

**Implementation Approach:**

```
1. Build User Feature Vectors:
   - Genre preferences (% watched per genre)
   - Average watch time, completion rate
   - Rating patterns

2. Cluster Users: K-Means on user vectors
   - Each cluster = user segment with similar taste

3. Generate Recommendations:
   - For user U in cluster C:
   - Find top-rated items in cluster C
   - Recommend unseen items popular in C

4. Cold Start Solution:
   - New user → assign to nearest cluster
   - Recommend cluster favorites immediately
```

**Benefits:**
- Reduces computation (cluster-level vs user-level)
- Handles sparsity (cluster averages are denser)
- Interpretable segments ("action lovers", "documentary fans")
- Enables serendipitous discovery within taste cluster

**Hybrid Approach:**
Combine K-Means clustering with matrix factorization: cluster users, then apply collaborative filtering within each cluster for refined recommendations.

---
