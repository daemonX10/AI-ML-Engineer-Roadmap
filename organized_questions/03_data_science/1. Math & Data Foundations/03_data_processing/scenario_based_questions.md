# Data Processing Interview Questions - Scenario-Based Questions

## Question 1
- [ ] Done

**Discuss various dimensionality reduction techniques besides PCA and LDA.**

**Answer:**

Beyond PCA and LDA, several dimensionality reduction techniques address different data characteristics and use cases. The choice depends on data structure, whether labels exist, and whether non-linear relationships need to be captured.

**Techniques Overview:**

| Technique | Type | Preserves | Best For |
|-----------|------|-----------|----------|
| **t-SNE** | Non-linear, unsupervised | Local structure | Visualization of clusters |
| **UMAP** | Non-linear, unsupervised | Local + global | Visualization, faster than t-SNE |
| **Autoencoders** | Non-linear, unsupervised | Learned features | Complex patterns |
| **Factor Analysis** | Linear, unsupervised | Latent factors | Finding underlying factors |
| **ICA** | Linear, unsupervised | Independent signals | Signal separation |
| **NMF** | Non-linear, unsupervised | Non-negative parts | Images, topic modeling |
| **Isomap** | Non-linear, unsupervised | Geodesic distances | Manifold learning |

**When to Use Each:**

- **t-SNE/UMAP:** Visualization, exploring cluster structure
- **Autoencoders:** When PCA doesn't capture patterns, deep feature learning
- **ICA:** Separating mixed signals (audio, EEG)
- **NMF:** When features should be non-negative (images, text counts)

**Key Consideration:**
- Linear methods (PCA, FA): Fast, interpretable, assumes linear relationships
- Non-linear methods (t-SNE, UMAP): Captures complex patterns, less interpretable

---

## Question 2
- [ ] Done

**How would you handle a categorical feature with a large number of levels?**

**Answer:**

High-cardinality categorical features (many unique values) require special handling because one-hot encoding creates too many columns, causing dimensionality issues and sparsity.

**Approach (Decision Tree):**

```
Number of unique values?
├── < 15: One-hot encoding OK
├── 15-100: Consider target encoding or frequency encoding
└── > 100: Use embeddings, feature hashing, or grouping
```

**Strategies by Situation:**

| Strategy | When to Use | Example |
|----------|-------------|---------|
| **Group rare categories** | Clear frequency distribution | Combine cities with <100 occurrences |
| **Target encoding** | Strong category-target relationship | Replace with mean of target |
| **Frequency encoding** | Frequency is informative | Replace with category count |
| **Feature hashing** | Very high cardinality, streaming | Hash to fixed buckets |
| **Embeddings** | Neural networks, semantic meaning | Learn dense vectors |

**Implementation Example:**
```python
# Strategy 1: Group rare categories
top_cats = df['city'].value_counts().nlargest(50).index
df['city_grouped'] = df['city'].where(df['city'].isin(top_cats), 'Other')

# Strategy 2: Target encoding with smoothing
global_mean = df['target'].mean()
category_means = df.groupby('city')['target'].agg(['mean', 'count'])
smoothing = 10
encoded = (category_means['count'] * category_means['mean'] + 
           smoothing * global_mean) / (category_means['count'] + smoothing)
df['city_encoded'] = df['city'].map(encoded)
```

---

## Question 3
- [ ] Done

**Discuss the advantages of using a data preprocessing pipeline.**

**Answer:**

A preprocessing pipeline encapsulates all transformation steps into a single, reproducible workflow that prevents data leakage and ensures consistency between training and inference.

**Advantages:**

| Advantage | Explanation |
|-----------|-------------|
| **Prevents data leakage** | Fit only on training data, apply to all |
| **Reproducibility** | Same transformations applied consistently |
| **Simplified deployment** | Single object to save and deploy |
| **Cleaner code** | No scattered preprocessing code |
| **Easy experimentation** | Swap components easily |
| **Cross-validation compatible** | Integrates with GridSearchCV |

**Without Pipeline (Problems):**
```python
# Bad: Fitting on all data causes leakage
scaler.fit(X_all)  # Leaks test info into training
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**With Pipeline (Correct):**
```python
# Good: Pipeline handles fit/transform correctly
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipeline.fit(X_train, y_train)  # Fits scaler on train only
pipeline.predict(X_test)        # Transforms test using train stats
```

**Deployment Benefit:**
```python
# Save entire pipeline
import joblib
joblib.dump(pipeline, 'model_pipeline.pkl')

# Load and use in production
pipeline = joblib.load('model_pipeline.pkl')
predictions = pipeline.predict(new_data)  # All preprocessing included
```

---

## Question 4
- [ ] Done

**Discuss the approaches to handle outliers in your data.**

**Answer:**

Outlier handling depends on whether outliers are errors (remove), genuine extreme values (cap/transform), or informative signals (keep). Understanding the cause determines the approach.

**Detection Methods:**

| Method | Approach | Threshold |
|--------|----------|-----------|
| **Z-score** | Standard deviations from mean | |z| > 3 |
| **IQR** | Distance from quartiles | < Q1-1.5×IQR or > Q3+1.5×IQR |
| **Percentile** | Extreme percentiles | < 1st or > 99th percentile |
| **Isolation Forest** | Anomaly detection model | Anomaly score |

**Handling Strategies:**

| Strategy | When to Use | Implementation |
|----------|-------------|----------------|
| **Remove** | Clear data errors | Drop rows |
| **Cap (Winsorize)** | Keep info, limit impact | Replace with threshold |
| **Transform** | Reduce scale impact | Log, square root |
| **Impute** | Treat as missing | Replace with median |
| **Separate model** | Outliers are different population | Model separately |
| **Robust methods** | Can't remove | Use median, MAD |

**Practical Approach:**
```python
import numpy as np

def handle_outliers(df, column, method='iqr', action='cap'):
    """Handle outliers with specified method and action."""
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        lower = mean - 3 * std
        upper = mean + 3 * std
    
    if action == 'cap':
        df[column] = df[column].clip(lower, upper)
    elif action == 'remove':
        df = df[(df[column] >= lower) & (df[column] <= upper)]
    
    return df
```

**Key Principle:** Investigate before removing. Outliers may be:
- Errors → Remove
- Valid extremes → Cap or transform
- Valuable signals (fraud) → Keep and flag

---

## Question 5
- [ ] Done

**You are given a dataset with several categorical features; describe your approach to preprocessing it for a machine learning model.**

**Answer:**

**Systematic Approach:**

**Step 1: Understand Categories**
```python
# For each categorical column
for col in categorical_columns:
    print(f"\n{col}:")
    print(f"  Unique values: {df[col].nunique()}")
    print(f"  Missing: {df[col].isnull().sum()}")
    print(f"  Top categories:\n{df[col].value_counts().head()}")
```

**Step 2: Classify Category Types**
| Type | Characteristics | Encoding |
|------|----------------|----------|
| **Binary** | 2 values | Label encode (0, 1) |
| **Nominal (low)** | < 10-15 unordered values | One-hot encode |
| **Nominal (high)** | Many unordered values | Target/frequency encode |
| **Ordinal** | Ordered values | Ordinal encode with correct order |

**Step 3: Handle Missing Values**
```python
# Categorical missing → "Unknown" or mode
df[cat_col] = df[cat_col].fillna('Unknown')
# Or impute with mode
df[cat_col] = df[cat_col].fillna(df[cat_col].mode()[0])
```

**Step 4: Apply Encoding**
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Group columns by encoding type
low_cardinality = ['gender', 'city']  # One-hot
high_cardinality = ['zip_code']       # Target encode
ordinal = ['education']               # Ordinal encode

preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), low_cardinality),
    ('ordinal', OrdinalEncoder(categories=[['High School', 'Bachelor', 'Master', 'PhD']]), 
     ordinal)
])
```

**Step 5: Validate**
- Check encoded output dimensions
- Verify no unexpected NaN
- Test on sample before full dataset

---

## Question 6
- [ ] Done

**How would you approach preprocessing a dataset that you know little about?**

**Answer:**

**Systematic Exploration Framework:**

**Phase 1: Initial Exploration**
```python
# Basic info
print(df.shape)
print(df.dtypes)
print(df.describe())
print(df.isnull().sum())

# Sample data
print(df.head(10))
print(df.sample(10))
```

**Phase 2: Data Quality Assessment**
| Check | What to Look For |
|-------|------------------|
| Missing values | Pattern, percentage per column |
| Duplicates | Exact matches, near-duplicates |
| Data types | Mismatched types (dates as strings) |
| Value ranges | Invalid values (negative age) |
| Distributions | Skewness, outliers |
| Cardinality | Unique values per column |

**Phase 3: Develop Hypotheses**
```python
# Correlations
df.corr()

# Target relationship (if supervised)
df.groupby(categorical_col)[target].mean()
```

**Phase 4: Iterative Preprocessing**
1. Start with minimal preprocessing
2. Train baseline model
3. Analyze errors
4. Add preprocessing steps
5. Evaluate improvement

**Checklist for Unknown Data:**
- [ ] Identify ID columns (don't use as features)
- [ ] Separate target variable
- [ ] Identify datetime columns
- [ ] Check for leakage columns (created after target)
- [ ] Validate train/test similarity

---

## Question 7
- [ ] Done

**How would you process and clean a large dataset that doesn't fit in memory?**

**Answer:**

**Strategies for Out-of-Memory Data:**

| Approach | When to Use | Tool |
|----------|-------------|------|
| **Chunked processing** | Sequential operations | pandas chunks |
| **Lazy evaluation** | Complex pipelines | Dask, Polars |
| **Database processing** | SQL operations | SQLite, PostgreSQL |
| **Distributed computing** | Very large scale | PySpark |
| **Sampling** | Exploration, prototyping | random sample |

**Approach 1: Chunked Processing**
```python
import pandas as pd

# Process in chunks
chunk_size = 100000
cleaned_chunks = []

for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Clean chunk
    chunk = chunk.dropna()
    chunk = chunk[chunk['value'] > 0]
    cleaned_chunks.append(chunk)

# Combine if fits in memory, otherwise write to disk
df_cleaned = pd.concat(cleaned_chunks)
```

**Approach 2: Dask for Larger-than-Memory**
```python
import dask.dataframe as dd

# Lazy loading
df = dd.read_csv('large_file.csv')

# Preprocessing (lazy)
df = df.dropna()
df['new_col'] = df['col1'] + df['col2']

# Execute and save
df.to_csv('cleaned_*.csv')  # Writes in parts
```

**Approach 3: Efficient Memory Usage**
```python
# Optimize dtypes
dtypes = {
    'category_col': 'category',
    'small_int': 'int16',
    'float_col': 'float32'
}
df = pd.read_csv('file.csv', dtype=dtypes)

# Load only needed columns
df = pd.read_csv('file.csv', usecols=['col1', 'col2', 'target'])
```

**Key Principles:**
- Process incrementally, not all at once
- Write intermediate results to disk
- Use appropriate data types
- Consider database for complex joins

---

## Question 8
- [ ] Done

**How would you handle varying scales of features in a clustering problem?**

**Answer:**

**Why It Matters:**
Clustering algorithms (K-Means, hierarchical) use distance measures that are scale-sensitive. Features with larger ranges dominate distance calculations, making scaling essential.

**Example Problem:**
```
Feature 1 (Age): Range 0-100
Feature 2 (Income): Range 0-500,000
Distance = sqrt((age_diff)² + (income_diff)²)
Income dominates → Age has negligible impact
```

**Solution Approaches:**

| Scenario | Scaling Method | Reason |
|----------|---------------|--------|
| General case | StandardScaler | Mean=0, Std=1 |
| Outliers present | RobustScaler | Uses median/IQR |
| Need bounded range | MinMaxScaler | [0, 1] range |
| Sparse data | MaxAbsScaler | Preserves sparsity |

**Implementation:**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Always scale before clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Now cluster
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X_scaled)

# Inverse transform cluster centers for interpretation
centers_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)
```

**Additional Considerations:**
- **After scaling:** All features contribute equally
- **Domain knowledge:** Some features may deserve more weight
- **Evaluation:** Compare clustering with/without scaling
- **Mixed types:** Handle categorical separately (Gower distance, k-prototypes)

---

## Question 9
- [ ] Done

**Describe the steps you would take to preprocess a dataset for a recommender system.**

**Answer:**

Preprocessing for recommender systems involves handling sparse user-item interaction data, encoding user/item features, normalizing ratings, and creating appropriate train/test splits that respect temporal ordering.

**Step-by-Step Approach:**

**1. Data Collection & Understanding**
- User-item interaction matrix (ratings, clicks, purchases)
- User features (demographics, preferences)
- Item features (categories, descriptions, metadata)
- Check sparsity: typically 95-99% of user-item pairs are missing

**2. Handle Interaction Data**

| Task | Method |
|------|--------|
| **Explicit feedback** (ratings) | Normalize ratings per user (mean-center) |
| **Implicit feedback** (clicks/purchases) | Convert to binary or confidence scores |
| **Missing values** | NOT random — absence ≠ dislike |
| **Duplicates** | Aggregate (average rating, total clicks) |

**3. Feature Engineering**

```python
# User features
user_features = {
    'avg_rating': user_ratings.mean(),
    'num_ratings': user_ratings.count(),
    'rating_std': user_ratings.std(),
    'active_days': (last_activity - first_activity).days
}

# Item features
item_features = {
    'avg_rating': item_ratings.mean(),
    'num_ratings': item_ratings.count(),
    'popularity': item_interactions.count(),
    'category_encoded': one_hot(item_category)
}
```

**4. Train/Test Split**
- **Temporal split**: Train on past, test on future (most realistic)
- **Leave-one-out**: Hold out last interaction per user
- **Never random split**: Would cause data leakage

**5. Normalize Ratings**
```python
# Mean-center per user (removes user bias)
user_means = ratings.groupby('user_id')['rating'].transform('mean')
ratings['normalized'] = ratings['rating'] - user_means
```

**6. Handle Cold Start**
- New users: Use demographic-based recommendations initially
- New items: Use content-based features until interactions accumulate

**Key Considerations:**
- Respect temporal ordering in all preprocessing
- Handle popularity bias (popular items dominate)
- Scale features for distance-based methods
- Consider implicit vs explicit feedback differently

---

## Question 10
- [ ] Done

**How would you sanitize and validate user input data in a production environment?**

**Answer:**

**Production Data Validation Framework:**

**Layer 1: Schema Validation**
```python
from pydantic import BaseModel, validator, Field
from typing import Optional

class UserInput(BaseModel):
    age: int = Field(ge=0, le=120)
    email: str
    income: Optional[float] = Field(ge=0)
    
    @validator('email')
    def valid_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('age', 'income', pre=True)
    def convert_to_number(cls, v):
        if isinstance(v, str):
            return float(v.replace(',', ''))
        return v
```

**Layer 2: Business Rules**
```python
def validate_business_rules(data):
    errors = []
    
    # Cross-field validation
    if data.get('start_date') > data.get('end_date'):
        errors.append("Start date must be before end date")
    
    # Range checks
    if data.get('quantity') < 0:
        errors.append("Quantity cannot be negative")
    
    return errors
```

**Layer 3: Sanitization**
```python
import re

def sanitize_input(data):
    # Remove potential SQL injection
    if isinstance(data, str):
        data = re.sub(r'[;\'"\\]', '', data)
    
    # Normalize whitespace
    data = ' '.join(data.split())
    
    # Strip HTML tags
    data = re.sub(r'<[^>]+>', '', data)
    
    return data
```

**Layer 4: Monitoring & Logging**
```python
def process_input(raw_data):
    try:
        validated = UserInput(**raw_data)
        log_successful_validation(validated)
        return validated
    except ValidationError as e:
        log_validation_failure(raw_data, e)
        raise HTTPException(status_code=400, detail=str(e))
```

**Key Principles:**
- Never trust user input
- Validate early, fail fast
- Log all validation failures
- Have fallback/default values
- Rate limit to prevent abuse

---

## Question 11
- [ ] Done

**Discuss the process of cleaning and preprocessing real-time streaming data.**

**Answer:**

**Streaming Data Challenges:**
- Data arrives continuously
- Cannot look at future data
- Must process with low latency
- Handle late/out-of-order data

**Architecture:**
```
Data Source → Ingestion → Validation → Transformation → Model/Storage
                ↓
           Dead Letter Queue (failed records)
```

**Processing Strategy:**

| Component | Implementation |
|-----------|----------------|
| **Validation** | Schema check, type coercion |
| **Missing values** | Use last known value or default |
| **Normalization** | Online statistics (running mean/std) |
| **Feature engineering** | Window-based aggregations |
| **Anomaly detection** | Real-time outlier flagging |

**Online Normalization:**
```python
class OnlineScaler:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0
    
    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)
    
    @property
    def std(self):
        return (self.M2 / self.n) ** 0.5 if self.n > 1 else 1
    
    def transform(self, x):
        return (x - self.mean) / self.std
```

**Windowed Features:**
```python
from collections import deque

class StreamingFeatures:
    def __init__(self, window_size=100):
        self.window = deque(maxlen=window_size)
    
    def update(self, value):
        self.window.append(value)
    
    def get_features(self):
        return {
            'mean': sum(self.window) / len(self.window),
            'min': min(self.window),
            'max': max(self.window),
            'count': len(self.window)
        }
```

**Tools:** Apache Kafka, Apache Flink, Spark Streaming

---

## Question 12
- [ ] Done

**Discuss the concept of embeddings in collaborative filtering.**

**Answer:**

**What are Embeddings:**
Embeddings are dense, learned vector representations that capture latent characteristics. In collaborative filtering, user and item embeddings represent preferences and attributes in a shared latent space.

**How It Works:**
```
User ID → User Embedding (e.g., 50 dimensions)
Item ID → Item Embedding (e.g., 50 dimensions)
Prediction = User Embedding · Item Embedding (dot product)
```

**Why Embeddings Work:**
- Similar users have similar embedding vectors
- Similar items have similar embedding vectors
- Interaction = compatibility in latent space

**Preprocessing for Embeddings:**

| Step | Purpose |
|------|---------|
| **ID mapping** | Convert user/item IDs to consecutive integers |
| **Interaction matrix** | User-item ratings/interactions |
| **Negative sampling** | Generate non-interactions for training |
| **Normalization** | Scale ratings to consistent range |

**Implementation:**
```python
import torch.nn as nn

class CollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=50):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        return (user_emb * item_emb).sum(dim=1)
```

**Benefits over Traditional Methods:**
- Handles sparsity better
- Captures complex patterns
- Generalizes to cold-start (with side features)
- Can visualize similar users/items

---

## Question 13
- [ ] Done

**Discuss research trends aimed at handling very large and high-dimensional datasets.**

**Answer:**

**Current Research Trends:**

| Trend | Description | Application |
|-------|-------------|-------------|
| **Efficient Transformers** | Sparse attention, linear attention | Long sequences, large datasets |
| **Neural Architecture Search** | Automate model design | Find efficient architectures |
| **Knowledge Distillation** | Compress large models | Deploy smaller models |
| **Federated Learning** | Train across distributed data | Privacy-preserving |
| **Self-Supervised Learning** | Learn from unlabeled data | Reduce labeling needs |
| **Random Projections** | Efficient dimensionality reduction | Very high dimensions |

**Scalable Preprocessing Techniques:**

| Technique | Purpose |
|-----------|---------|
| **Feature Hashing** | Fixed-size encoding for any cardinality |
| **Locality-Sensitive Hashing** | Approximate nearest neighbors |
| **Sketch-based Methods** | Approximate statistics from streams |
| **Quantization** | Reduce precision for efficiency |

**Hardware-Aware Approaches:**
- Mixed-precision training (FP16)
- Gradient checkpointing
- Model parallelism
- Data parallelism with efficient communication

**Key Direction:** Moving from "more data, bigger models" to "efficient, targeted learning" with:
- Better data selection (active learning)
- More efficient representations
- Hardware-algorithm co-design

---

## Question 14
- [ ] Done

**Discuss the importance of transparency in data preprocessing.**

**Answer:**

**Why Transparency Matters:**

| Stakeholder | Need for Transparency |
|-------------|----------------------|
| **Data Scientists** | Reproduce results, debug issues |
| **Engineers** | Deploy and maintain pipelines |
| **Business** | Understand model decisions |
| **Regulators** | Audit and compliance |
| **Users** | Trust and fairness |

**Transparency Practices:**

| Practice | Implementation |
|----------|----------------|
| **Documentation** | Record all transformation decisions |
| **Version control** | Track changes to preprocessing code |
| **Logging** | Log parameter values, statistics |
| **Data lineage** | Track data from source to model |
| **Reproducibility** | Same input → same output |

**Documentation Template:**
```markdown
## Preprocessing Pipeline v2.3

### Data Sources
- Customer transactions: database.customers
- Product catalog: s3://bucket/products.csv

### Transformations
1. **Missing Values**
   - Income: Median imputation (median=$52,000 from training)
   - Category: Filled with "Unknown"
   - Rationale: <5% missing, MCAR pattern

2. **Outliers**
   - Income: Capped at 99th percentile ($200,000)
   - Rationale: Extreme values were data entry errors

3. **Encoding**
   - City: Target encoding with smoothing=10
   - Gender: One-hot encoding

### Validation
- Row retention: 97.3%
- No leakage verification: ✓
```

**Benefits:**
- Easier debugging when issues arise
- Regulatory compliance (GDPR, financial audits)
- Knowledge transfer between team members
- Builds trust with stakeholders
