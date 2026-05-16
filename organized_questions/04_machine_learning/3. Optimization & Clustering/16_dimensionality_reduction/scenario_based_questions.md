# Dimensionality Reduction Interview Questions - Scenario_Based Questions

## Question 1
- [ ] Done

**How would you use dimensionality reduction for a dataset containing thousands of features, such as gene expression data?**

### Answer

**Scenario Context:**  
Gene expression data typically has 10,000-50,000 genes (features) but only hundreds to thousands of samples. This extreme d >> n scenario requires careful handling to avoid curse of dimensionality while preserving biological signal.

**Approach:**

**Step 1: Understand the Data**
- How many samples (n)? How many genes (d)?
- What's the task? (Classification, clustering, biomarker discovery)
- Is interpretability needed? (Gene names matter for biologists)

**Step 2: Preprocessing**
```python
# Log transform (common for gene expression)
X_log = np.log2(X + 1)

# Filter low-variance genes
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.1)
X_filtered = selector.fit_transform(X_log)

# Standardize
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X_filtered)
```

**Step 3: Feature Selection (if interpretability needed)**
```python
# Statistical test per gene
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=500)
X_selected = selector.fit_transform(X_scaled, y)

# Get selected gene names
selected_genes = gene_names[selector.get_support()]
```

**Step 4: Dimensionality Reduction Strategy**

| Goal | Method | Reason |
|------|--------|--------|
| Preprocessing for ML | PCA (50-100 components) | Fast, reduces noise |
| Visualization | t-SNE/UMAP after PCA | Reveals clusters |
| Biomarker discovery | Sparse PCA, Lasso | Selects actual genes |
| Classification | LDA | Uses labels |

**Recommended Pipeline:**
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Step 1: Filter to top variance genes
n_top_genes = 2000
gene_variance = X_scaled.var(axis=0)
top_gene_idx = np.argsort(gene_variance)[-n_top_genes:]
X_top = X_scaled[:, top_gene_idx]

# Step 2: PCA for noise reduction
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_top)
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# Step 3: For ML - use PCA output
from sklearn.svm import SVC
model = SVC()
model.fit(X_pca, y)

# Step 4: For visualization - use UMAP/t-SNE on PCA
reducer = umap.UMAP(n_components=2, n_neighbors=15)
X_viz = reducer.fit_transform(X_pca)

plt.scatter(X_viz[:, 0], X_viz[:, 1], c=y, cmap='tab10')
plt.title('Gene Expression: Sample Clusters')
```

**Specific Considerations for Gene Data:**

| Consideration | Solution |
|---------------|----------|
| Batch effects | Batch correction before DR |
| Sparse data (scRNA-seq) | Use methods for sparse data |
| Biological interpretation | Keep gene names trackable |
| Multiple omics | Integrated analysis methods |

**Complete Pipeline Summary:**
```
Raw Expression
     ↓
Log Transform
     ↓
Filter Low-Variance Genes (keep top 2000-5000)
     ↓
Standardize
     ↓
PCA (50-100 components) for ML tasks
     ↓
UMAP/t-SNE for visualization
```

---

## Question 2
- [ ] Done

**Discuss your approach to reduce dimensionality for text data before performing sentiment analysis.**

### Answer

**Scenario Context:**  
Text data has extremely high dimensionality (vocabulary size 10K-100K+). Effective reduction requires text-specific methods that preserve semantic meaning while reducing feature space for sentiment classification.

**Approach Strategy:**

| Stage | Options |
|-------|---------|
| **Text Representation** | BoW, TF-IDF, Word Embeddings, Transformers |
| **Dimensionality Reduction** | Feature selection, LSA/SVD, Embeddings |
| **Sentiment Model** | Logistic Regression, SVM, Neural Network |

**Method 1: TF-IDF + Truncated SVD (LSA)**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

# TF-IDF creates sparse high-dimensional matrix
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

# Truncated SVD (LSA) reduces dimensions
svd = TruncatedSVD(n_components=100)

# Pipeline
text_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('svd', svd)
])

X_reduced = text_pipeline.fit_transform(texts)
# Shape: (n_docs, 100) - much smaller than (n_docs, 10000)
```

**Method 2: Pre-trained Word Embeddings**
```python
import numpy as np

# Load pre-trained embeddings (Word2Vec, GloVe, FastText)
def get_document_vector(text, word2vec_model):
    words = text.split()
    word_vectors = [word2vec_model[w] for w in words if w in word2vec_model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Average word vectors
    return np.zeros(300)  # 300-dim embedding

X_embeddings = np.array([get_document_vector(text, w2v) for text in texts])
# Shape: (n_docs, 300)
```

**Method 3: Sentence Transformers (Modern Approach)**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
X_embeddings = model.encode(texts)
# Shape: (n_docs, 384) - semantically rich
```

**Comparison of Methods:**

| Method | Dimensions | Semantic | Speed | Accuracy |
|--------|------------|----------|-------|----------|
| TF-IDF + SVD | 100-300 | Low | Fast | Good |
| Word2Vec avg | 100-300 | Medium | Fast | Good |
| Sentence Transformers | 384-768 | High | Medium | Best |
| BERT fine-tuned | 768 | Highest | Slow | Best |

**Complete Pipeline:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Method 1: Traditional
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(train_texts)

svd = TruncatedSVD(n_components=100)
X_reduced = svd.fit_transform(X_tfidf)

# Train sentiment classifier
X_train, X_test, y_train, y_test = train_test_split(X_reduced, labels)
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Method 2: Modern (Sentence Transformers)
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer('all-MiniLM-L6-v2')
X_embeddings = encoder.encode(texts)
# Already low-dimensional (384), may not need further reduction

clf = LogisticRegression()
clf.fit(X_embeddings[train_idx], labels[train_idx])
```

**Key Decisions:**

| Decision | Guidance |
|----------|----------|
| BoW vs Embeddings | Embeddings better for semantic tasks |
| SVD components | 100-300 usually sufficient |
| Pre-trained vs train | Pre-trained usually better for sentiment |
| Further reduction? | Often not needed with embeddings |

**Recommendation:**
- For quick baseline: TF-IDF + SVD
- For best results: Sentence Transformers (already reduced and semantic)

---

## Question 3
- [ ] Done

**How could dimensionality reduction be applied effectively when visualizing high-dimensional data?**

### Answer

**Definition:**  
For visualization, reduce data to 2D or 3D using appropriate methods: PCA for global linear structure, t-SNE for local cluster structure, UMAP for both global and local preservation. Choice depends on what patterns you want to reveal.

**Method Selection for Visualization:**

| Method | Best For | Preserves |
|--------|----------|-----------|
| **PCA** | Quick overview, linear relationships | Global variance |
| **t-SNE** | Cluster visualization | Local neighborhoods |
| **UMAP** | General purpose, balanced | Global + local |
| **MDS** | Distance preservation | Pairwise distances |

**Best Practices:**

1. **Preprocess First:**
   - Scale features before any method
   - For large data: PCA to 50 dims, then t-SNE/UMAP

2. **Choose Method Based on Goal:**
   - Explore clusters → t-SNE or UMAP
   - Understand feature relationships → PCA biplot
   - Quick check → PCA

3. **Parameter Tuning (t-SNE):**
   - Perplexity: 5-50 (try multiple values)
   - Iterations: ≥1000

**Python Example:**
```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Preprocessing
X_scaled = StandardScaler().fit_transform(X)

# Method 1: PCA (fast, global)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Method 2: t-SNE (slow, local clusters)
# First reduce with PCA for speed
X_pca_50 = PCA(n_components=50).fit_transform(X_scaled)
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_pca_50)

# Method 3: UMAP (fast, balanced)
reducer = umap.UMAP(n_components=2, n_neighbors=15)
X_umap = reducer.fit_transform(X_scaled)

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, data, title in zip(axes, [X_pca, X_tsne, X_umap], 
                           ['PCA', 't-SNE', 'UMAP']):
    ax.scatter(data[:, 0], data[:, 1], c=y, cmap='viridis', s=5)
    ax.set_title(title)
plt.show()
```

**Interpretation Tips:**
- t-SNE/UMAP: Cluster distance not meaningful, only cluster existence
- PCA: Direction matters, can interpret via loadings
- Always show multiple views, don't rely on single method

---

## Question 4
- [ ] Done

**Explain the process you would follow to select features for a predictive model in a marketing dataset.**

### Answer

**Definition:**  
Feature selection for marketing involves a systematic process: understanding business context, exploratory analysis, removing irrelevant/redundant features, applying statistical tests, using model-based selection, and validating with cross-validation—always keeping interpretability in mind for stakeholder communication.

**Step-by-Step Process:**

**Step 1: Understand Business Context**
- What is the target? (conversion, churn, CLV)
- What features are actionable?
- What features are available at prediction time?
- Any regulatory constraints? (GDPR, fairness)

**Step 2: Initial Data Exploration**
```python
# Check data types, missing values, cardinality
df.info()
df.describe()
df.isnull().sum()
```

**Step 3: Remove Obvious Irrelevant Features**
- IDs, timestamps (unless engineered)
- Features with single value (zero variance)
- Features with >50% missing values
- Leakage features (derived from target)

**Step 4: Handle Multicollinearity**
```python
import seaborn as sns
# Correlation matrix
corr = df.corr()
# Remove one of highly correlated pairs (>0.9)
```

**Step 5: Statistical Feature Selection**
```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# For categorical target (classification)
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_mask = selector.get_support()
selected_features = X.columns[selected_mask]
```

**Step 6: Model-Based Selection**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Tree-based importance
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)

# Select features above median importance
selector = SelectFromModel(rf, threshold='median')
X_selected = selector.fit_transform(X, y)
```

**Step 7: Recursive Feature Elimination (Optional)**
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

rfe = RFE(LogisticRegression(), n_features_to_select=15)
rfe.fit(X, y)
selected_features = X.columns[rfe.support_]
```

**Step 8: Validate with Cross-Validation**
```python
from sklearn.model_selection import cross_val_score

# Compare performance with all vs selected features
score_all = cross_val_score(model, X, y, cv=5).mean()
score_selected = cross_val_score(model, X_selected, y, cv=5).mean()
```

**Marketing-Specific Considerations:**

| Feature Type | Selection Approach |
|--------------|-------------------|
| Demographics | Keep interpretable ones |
| Behavioral | Check recency (RFM) |
| Campaign history | Avoid leakage |
| External data | Verify availability at inference |

**Final Feature Selection Strategy:**
1. Start with domain knowledge
2. Apply multiple selection methods
3. Intersect selected features from different methods
4. Validate improvement in CV score
5. Ensure interpretability for business stakeholders

**Interview Tip:** Emphasize interpretability for marketing—stakeholders need to understand why customers convert.

---

## Question 5
- [ ] Done

**What are some potential pitfalls when applying dimensionality reduction to time-series data?**

### Answer

**Definition:**  
Time-series dimensionality reduction has unique challenges including destroying temporal dependencies, data leakage from future values, loss of sequential structure, and inappropriateness of standard methods that assume i.i.d. samples.

**Key Pitfalls:**

| Pitfall | Description |
|---------|-------------|
| **Temporal Structure Loss** | PCA/t-SNE ignore time ordering |
| **Data Leakage** | Fitting on future data contaminates model |
| **Non-stationarity** | Time-varying statistics break assumptions |
| **Autocorrelation** | Standard methods assume independence |
| **Sequence Length Variation** | Different length series hard to handle |

**Detailed Explanation:**

**1. Loss of Temporal Dependencies:**
- Standard PCA treats each time point as independent feature
- Destroys lag relationships, trends, seasonality
- Solution: Use time-aware methods (Dynamic PCA, wavelet decomposition)

**2. Data Leakage:**
```python
# WRONG: Fit scaler/PCA on entire dataset
pca = PCA().fit(full_data)  # Leaks future info into past

# CORRECT: Fit only on training (past) data
pca = PCA().fit(train_data)
test_transformed = pca.transform(test_data)
```

**3. Non-stationarity Issues:**
- Mean and variance change over time
- Covariance matrix computed over all time is meaningless
- Solution: Difference the series, use rolling windows

**4. Autocorrelation Violation:**
- PCA assumes uncorrelated observations
- Time series have serial correlation
- Eigenvalue estimates become biased

**5. Sequence-specific Challenges:**

| Issue | Standard DR Problem | Time Series Context |
|-------|---------------------|---------------------|
| Alignment | N/A | Different length sequences |
| Lag features | N/A | Must preserve lag structure |
| Trend | Treated as feature | Should be modeled separately |

**Better Approaches for Time Series:**

1. **Time-aware feature engineering:**
   - Extract features: mean, std, trend, seasonality, autocorrelation
   - Apply DR to extracted features

2. **Dynamic PCA:**
   - Extends PCA to capture lagged covariances
   - Preserves temporal dynamics

3. **Recurrent Autoencoders:**
   - LSTM/GRU autoencoders for sequence compression
   - Preserves temporal structure

4. **Wavelets/Fourier:**
   - Transform to frequency domain
   - Reduce high-frequency components

**Python Example - Safe Approach:**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Time-series cross-validation approach
def reduce_time_series(train, test):
    # Fit only on training data
    scaler = StandardScaler()
    scaler.fit(train)
    
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    
    pca = PCA(n_components=0.95)
    pca.fit(train_scaled)  # Only train!
    
    return pca.transform(train_scaled), pca.transform(test_scaled)

# Feature extraction approach (better)
def extract_ts_features(series):
    return {
        'mean': series.mean(),
        'std': series.std(),
        'trend': np.polyfit(range(len(series)), series, 1)[0],
        'autocorr_1': series.autocorr(lag=1)
    }
```

**Interview Tip:** Always mention the leakage risk and that you would use time-aware cross-validation (walk-forward or expanding window).

---
