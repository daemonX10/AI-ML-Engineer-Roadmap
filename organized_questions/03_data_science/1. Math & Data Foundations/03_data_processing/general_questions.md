# Data Processing Interview Questions - General Questions

## Question 1
- [ ] Done

**Why is data cleaning essential before model training?**

**Answer:**

Data cleaning removes errors, inconsistencies, and noise that would otherwise propagate through the model, leading to poor predictions. Clean data ensures the model learns genuine patterns rather than artifacts of bad data quality ("Garbage In, Garbage Out").

**Why It's Essential:**
- Most ML algorithms cannot handle missing values
- Outliers can skew model parameters
- Inconsistent data leads to incorrect pattern learning
- Duplicates cause overfitting to repeated samples

**Common Cleaning Tasks:**
- Handle missing values
- Remove/correct duplicates
- Fix inconsistent formatting
- Handle outliers
- Correct data types

---

## Question 2
- [ ] Done

**How do you handle missing data within a dataset?**

**Answer:**

Missing data handling depends on the amount, pattern, and mechanism of missingness. Options include deletion (if minimal and random), imputation (fill with estimated values), or using algorithms that handle missing values natively.

**Decision Framework:**

| Missing % | Pattern | Approach |
|-----------|---------|----------|
| < 5% | MCAR | Deletion often OK |
| 5-20% | MAR | Imputation (mean, KNN, regression) |
| > 20% | Any | Advanced imputation or indicator variable |

**Imputation Methods:**
- **Simple:** Mean, median, mode
- **Advanced:** KNN, iterative imputation (MICE)
- **Model-based:** Predict missing values using other features

**Python Example:**
```python
from sklearn.impute import SimpleImputer, KNNImputer

# Simple imputation
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(X)
```

---

## Question 3
- [ ] Done

**When would you recommend using regression imputation?**

**Answer:**

Regression imputation is recommended when missing values have relationships with other features that can be modeled. It predicts missing values using a regression model trained on complete cases, capturing feature correlations better than simple mean/median.

**When to Use:**
- Features are correlated with missing variable
- Missing data is MAR (Missing at Random)
- Sufficient complete cases for training regression
- Want to preserve feature relationships

**When NOT to Use:**
- Features are independent (no predictive power)
- Very few complete cases
- High percentage of missing data

**Python Example:**
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Uses regression to predict missing values iteratively
imputer = IterativeImputer(max_iter=10, random_state=42)
X_imputed = imputer.fit_transform(X)
```

**Caution:** Can underestimate variance (all imputed values on regression line).

---

## Question 4
- [ ] Done

**How do missing values impact machine learning models?**

**Answer:**

Missing values directly affect model training and predictions. Most algorithms cannot process NaN values and will error. Those that run may produce biased results if missingness is not random, leading to poor generalization.

**Impact by Algorithm:**

| Algorithm | Impact |
|-----------|--------|
| **Linear/Logistic Regression** | Cannot handle; requires preprocessing |
| **Decision Trees** | Can handle (treat missing as category) |
| **XGBoost/LightGBM** | Native handling; learns missing patterns |
| **Neural Networks** | Cannot handle; requires preprocessing |
| **KNN** | Cannot compute distances with missing |

**Broader Impacts:**
- **Bias:** If missingness relates to target
- **Reduced power:** Deletion reduces sample size
- **Wrong correlations:** Imputation can distort relationships

**Best Practice:**
- Understand why data is missing (MCAR, MAR, MNAR)
- Choose appropriate handling method
- Create indicator features for missingness patterns

---

## Question 5
- [ ] Done

**How do principal component analysis (PCA) and linear discriminant analysis (LDA) differ?**

**Answer:**

PCA is an unsupervised dimensionality reduction technique that maximizes variance, finding directions of greatest data spread. LDA is supervised, maximizing class separability by finding directions that best separate predefined classes.

**Comparison:**

| Aspect | PCA | LDA |
|--------|-----|-----|
| **Type** | Unsupervised | Supervised |
| **Objective** | Maximize variance | Maximize class separation |
| **Uses labels** | No | Yes |
| **Max components** | min(n_features, n_samples) | n_classes - 1 |
| **Use case** | General dimensionality reduction | Classification preprocessing |

**Mathematical Objective:**
- **PCA:** Maximize $Var(w^T X)$
- **LDA:** Maximize $\frac{w^T S_B w}{w^T S_W w}$ (between-class / within-class variance)

**When to Use:**
- **PCA:** Visualization, noise reduction, unsupervised tasks
- **LDA:** Classification tasks, when labels available

---

## Question 6
- [ ] Done

**Why is feature engineering critical in model performance?**

**Answer:**

Feature engineering creates informative representations from raw data that make patterns easier for algorithms to learn. Good features can dramatically improve model performance, sometimes more than algorithm choice, by encoding domain knowledge and revealing hidden relationships.

**Why It's Critical:**
- Raw data often not in optimal form for learning
- Captures domain knowledge algorithms can't discover
- Can make simple models perform like complex ones
- Reduces need for complex architectures

**Types of Feature Engineering:**
- **Transformation:** Log, square root, binning
- **Combination:** Ratios, interactions, polynomials
- **Aggregation:** Count, sum, mean by group
- **Time-based:** Lag, rolling statistics, seasonality
- **Domain-specific:** Custom features from expertise

**Example:**
```python
# Raw: timestamp, amount
# Engineered:
df['hour'] = df['timestamp'].dt.hour
df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
df['amount_log'] = np.log1p(df['amount'])
df['amount_per_hour'] = df.groupby('hour')['amount'].transform('mean')
```

---

## Question 7
- [ ] Done

**How do you design and select features for a machine learning model?**

**Answer:**

Feature design involves creating features from domain knowledge and data exploration. Feature selection identifies the most predictive subset, removing irrelevant or redundant features to improve model performance and reduce complexity.

**Feature Design Process:**
1. Understand domain and business problem
2. Explore data (distributions, correlations)
3. Create hypotheses about predictive signals
4. Engineer features based on hypotheses
5. Validate feature utility

**Feature Selection Methods:**

| Type | Method | Description |
|------|--------|-------------|
| **Filter** | Correlation, Chi-square | Independent of model |
| **Wrapper** | RFE, Forward Selection | Uses model performance |
| **Embedded** | Lasso, Tree importance | Built into training |

**Python Example:**
```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# Filter method
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Wrapper method (RFE)
from sklearn.ensemble import RandomForestClassifier
rfe = RFE(RandomForestClassifier(), n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)
```

---

## Question 8
- [ ] Done

**When should you apply Z-score normalization?**

**Answer:**

Z-score normalization (standardization) transforms features to have mean=0 and std=1. Apply it when algorithms assume normally distributed features, for gradient-based optimization, or when comparing features with different units.

**When to Apply:**
- Algorithms assuming normal distribution (e.g., Gaussian Naive Bayes)
- Gradient descent optimization (linear regression, neural networks)
- Distance-based algorithms (KNN, SVM, K-Means)
- PCA (standardize before applying)
- Features have different units/scales

**When NOT to Apply:**
- Tree-based models (scale-invariant)
- Data has meaningful bounds (use Min-Max instead)
- Sparse data (would destroy sparsity)

**Formula:**
$$z = \frac{x - \mu}{\sigma}$$

**Python Example:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use train statistics!
```

---

## Question 9
- [ ] Done

**How do you decide which feature scaling method to use?**

**Answer:**

Choose scaling method based on data distribution, presence of outliers, algorithm requirements, and whether bounded output is needed. Different methods suit different scenarios.

**Decision Guide:**

| Scenario | Method | Reason |
|----------|--------|--------|
| Normal distribution | StandardScaler | Assumes normality |
| Outliers present | RobustScaler | Uses median/IQR |
| Need bounded [0,1] | MinMaxScaler | Neural networks, images |
| Sparse data | MaxAbsScaler | Preserves sparsity |
| Heavy-tailed | Log transform | Reduce skewness first |

**Quick Decision Tree:**
```
Is data sparse? → MaxAbsScaler
    ↓ No
Are there outliers? → RobustScaler
    ↓ No
Need bounded range? → MinMaxScaler
    ↓ No
Default → StandardScaler
```

**Python Example:**
```python
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, 
                                   RobustScaler, MaxAbsScaler)

# Check for outliers
if df['feature'].quantile(0.99) > 3 * df['feature'].std():
    scaler = RobustScaler()
else:
    scaler = StandardScaler()
```

---

## Question 10
- [ ] Done

**Compare and contrast standardization vs normalization.**

**Answer:**

Standardization (Z-score) centers data around mean=0 with std=1. Normalization (Min-Max) scales data to a fixed range [0,1]. Standardization is unbounded and handles outliers better; normalization provides bounded output.

**Comparison:**

| Aspect | Standardization | Normalization |
|--------|-----------------|---------------|
| **Formula** | $(x - \mu)/\sigma$ | $(x - min)/(max - min)$ |
| **Output range** | Unbounded (~[-3, 3]) | [0, 1] |
| **Center** | Mean = 0 | Not centered |
| **Outlier sensitivity** | Moderate | High |
| **Preserves distribution** | Yes | Yes |
| **Use case** | Most algorithms | Neural networks, bounded requirements |

**When to Use:**
- **Standardization:** SVM, PCA, Logistic Regression, when outliers exist
- **Normalization:** Neural networks (sigmoid/tanh), image pixels, when bounds needed

---

## Question 11
- [ ] Done

**Why do you need to convert categorical data into numerical format?**

**Answer:**

Machine learning algorithms perform mathematical operations (distances, gradients, matrix multiplications) that require numerical inputs. Categorical data (text labels) cannot be used directly in these computations, so conversion to numbers is essential.

**Reasons:**
- Algorithms compute distances (KNN, K-Means) — need numbers
- Gradient descent optimizes numerical parameters
- Matrix operations require numerical matrices
- Statistical measures (mean, variance) need numbers

**Encoding Methods:**
- **Label Encoding:** Assign integers (ordered categories)
- **One-Hot Encoding:** Binary columns (unordered categories)
- **Target Encoding:** Replace with target statistics

**Important Consideration:**
- Wrong encoding choice can mislead models
- Label encoding implies order (use for ordinal data)
- One-hot prevents false ordinal relationships

---

## Question 12
- [ ] Done

**What special considerations are there when processing time-series data?**

**Answer:**

Time-series data has temporal dependencies requiring special handling to prevent data leakage, preserve order, and capture time-based patterns. Standard random splits and transformations can violate causality.

**Special Considerations:**

| Aspect | Consideration |
|--------|---------------|
| **Train/Test Split** | Must be chronological (no future data in training) |
| **Validation** | Use time-based CV (e.g., TimeSeriesSplit) |
| **Feature Engineering** | Lag features, rolling statistics (only past data) |
| **Scaling** | Fit only on training period |
| **Stationarity** | Check and achieve (differencing, detrending) |
| **Seasonality** | Capture or remove periodic patterns |

**Python Example:**
```python
from sklearn.model_selection import TimeSeriesSplit

# Time-based cross-validation
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    # Train always before test chronologically
    X_train, X_test = X[train_idx], X[test_idx]

# Lag features (only past values)
df['lag_1'] = df['value'].shift(1)
df['rolling_mean_7'] = df['value'].rolling(7).mean()
```

---

## Question 13
- [ ] Done

**How do you handle seasonality in time-series data?**

**Answer:**

Seasonality handling involves identifying periodic patterns (daily, weekly, yearly) and either removing them for stationary modeling or encoding them as features. Methods include seasonal decomposition, differencing, and Fourier features.

**Handling Methods:**

| Method | Description | Use Case |
|--------|-------------|----------|
| **Seasonal differencing** | $y_t - y_{t-s}$ | Remove seasonal pattern |
| **Decomposition** | Extract trend + seasonal + residual | Understand components |
| **Fourier features** | Sin/cos terms for seasonality | Capture smooth patterns |
| **Seasonal dummies** | One-hot for season | Capture discrete seasons |

**Python Example:**
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decomposition
result = seasonal_decompose(df['value'], model='additive', period=12)
trend = result.trend
seasonal = result.seasonal
residual = result.resid

# Seasonal differencing
df['seasonal_diff'] = df['value'] - df['value'].shift(12)  # Monthly data

# Fourier features for yearly seasonality
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

---

## Question 14
- [ ] Done

**How do you incorporate custom transformers within a preprocessing pipeline?**

**Answer:**

Custom transformers inherit from sklearn's BaseEstimator and TransformerMixin, implementing fit() and transform() methods. They integrate seamlessly into Pipeline for reproducible preprocessing.

**Template:**
```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, parameter=None):
        self.parameter = parameter
    
    def fit(self, X, y=None):
        # Learn parameters from training data
        self.learned_param_ = X.mean()  # Example
        return self
    
    def transform(self, X):
        # Apply transformation
        return X - self.learned_param_
```

**Integration in Pipeline:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('custom', CustomTransformer(parameter=10)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

---

## Question 15
- [ ] Done

**How do you preprocess text data for natural language processing?**

**Answer:**

Text preprocessing converts raw text into numerical features suitable for ML. Steps include cleaning, tokenization, normalization, and vectorization to create structured numerical representations.

**Preprocessing Pipeline:**

| Step | Description | Example |
|------|-------------|---------|
| **Lowercasing** | Standardize case | "Hello" → "hello" |
| **Cleaning** | Remove special chars, HTML | Remove <tags>, @mentions |
| **Tokenization** | Split into words/subwords | "hello world" → ["hello", "world"] |
| **Stopword removal** | Remove common words | Remove "the", "is", "a" |
| **Stemming/Lemmatization** | Reduce to root form | "running" → "run" |
| **Vectorization** | Convert to numbers | TF-IDF, embeddings |

**Python Example:**
```python
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters
    text = ' '.join(text.split())  # Remove extra whitespace
    return text

# Clean and vectorize
texts_clean = [preprocess_text(t) for t in texts]
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(texts_clean)
```

---

## Question 16
- [ ] Done

**How do you deal with a large vocabulary size in text data?**

**Answer:**

Large vocabularies increase dimensionality and computation. Strategies include limiting vocabulary size, using subword tokenization, hashing, or dense embeddings to manage dimensionality while preserving information.

**Strategies:**

| Method | Description | Trade-off |
|--------|-------------|-----------|
| **Max features** | Keep top N frequent words | May lose rare but important terms |
| **Min document frequency** | Remove rare words | Loses rare terms |
| **Subword tokenization** | BPE, WordPiece | Smaller vocab, handles OOV |
| **Feature hashing** | Hash words to fixed buckets | Collisions possible |
| **Embeddings** | Dense vectors per word | Fixed dimension regardless of vocab |

**Python Example:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Limit vocabulary
vectorizer = TfidfVectorizer(
    max_features=10000,      # Top 10K words
    min_df=5,                # Appears in at least 5 docs
    max_df=0.95              # Not in more than 95% of docs
)

# Feature hashing (no vocabulary needed)
from sklearn.feature_extraction.text import HashingVectorizer
hasher = HashingVectorizer(n_features=10000)
```

---

## Question 17
- [ ] Done

**What preprocessing steps are commonly applied to image data?**

**Answer:**

Image preprocessing prepares pixel data for models through resizing, normalization, and optional augmentation. Steps ensure consistent input dimensions, appropriate value ranges, and sufficient training variety.

**Common Steps:**

| Step | Description | Purpose |
|------|-------------|---------|
| **Resizing** | Scale to model input size | Consistent dimensions |
| **Normalization** | Scale pixels [0,1] or standardize | Stable training |
| **Channel ordering** | RGB/BGR, channels-first/last | Framework compatibility |
| **Augmentation** | Random transforms | Increase training data |
| **Cropping** | Center or random crop | Focus on content |

**Python Example:**
```python
from torchvision import transforms

# Standard preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # Converts to [0,1] and channels-first
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225]
    )
])

# With augmentation (training only)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## Question 18
- [ ] Done

**How do you identify and resolve data inconsistencies?**

**Answer:**

Data inconsistencies are variations in representing the same information (formatting, spelling, units). Identification involves profiling and pattern analysis; resolution involves standardization and validation rules.

**Common Inconsistencies:**

| Type | Example | Resolution |
|------|---------|------------|
| **Format** | "2023-01-15" vs "01/15/2023" | Standardize date format |
| **Case** | "New York" vs "new york" | Lowercase or title case |
| **Spelling** | "colour" vs "color" | Map to standard |
| **Units** | meters vs feet | Convert to single unit |
| **Abbreviations** | "St." vs "Street" | Expand or standardize |

**Identification Methods:**
```python
# Profile data
df['column'].value_counts()  # See variations
df['column'].nunique()       # Count unique values

# Find similar strings (fuzzy matching)
from fuzzywuzzy import fuzz
# Check if entries are similar but not exact
```

**Resolution:**
```python
# Standardize formatting
df['date'] = pd.to_datetime(df['date'])
df['text'] = df['text'].str.lower().str.strip()

# Map variations to standard
mapping = {'NYC': 'New York', 'NY': 'New York', 'New York City': 'New York'}
df['city'] = df['city'].replace(mapping)
```

---

## Question 19
- [ ] Done

**How do you verify the correctness of the data after cleaning?**

**Answer:**

Data verification involves systematic checks to ensure cleaning operations preserved data integrity, removed issues, and didn't introduce new problems. Use assertions, statistical checks, and manual sampling.

**Verification Methods:**

| Check | Purpose | Implementation |
|-------|---------|----------------|
| **Row count** | No unexpected loss | Compare before/after |
| **Null check** | Missing values handled | df.isnull().sum() |
| **Range validation** | Values in expected bounds | assert (df['age'] >= 0).all() |
| **Distribution** | Shape preserved | Compare histograms |
| **Unique counts** | Categories as expected | df['col'].nunique() |

**Python Example:**
```python
def verify_cleaning(df_before, df_after, config):
    checks = []
    
    # 1. Row count within acceptable range
    row_change = len(df_after) / len(df_before)
    checks.append(('row_retention', row_change >= config['min_retention']))
    
    # 2. No missing values in required columns
    for col in config['required_cols']:
        checks.append((f'{col}_no_nulls', df_after[col].isnull().sum() == 0))
    
    # 3. Values in valid range
    if 'age' in df_after.columns:
        checks.append(('age_valid', df_after['age'].between(0, 120).all()))
    
    # Report
    for check, passed in checks:
        print(f"{check}: {'✓' if passed else '✗'}")
    
    return all(passed for _, passed in checks)
```

---

## Question 20
- [ ] Done

**What tools and libraries do you prefer for data preprocessing in Python?**

**Answer:**

Python's data preprocessing ecosystem includes pandas for data manipulation, scikit-learn for ML preprocessing, and specialized libraries for specific data types. Choice depends on data size, type, and workflow requirements.

**Core Libraries:**

| Library | Use Case |
|---------|----------|
| **pandas** | Data manipulation, cleaning, exploration |
| **NumPy** | Numerical operations, array processing |
| **scikit-learn** | ML preprocessing (scaling, encoding, imputation) |
| **category_encoders** | Advanced categorical encoding |
| **imbalanced-learn** | Handling imbalanced data |

**Domain-Specific:**

| Data Type | Libraries |
|-----------|-----------|
| **Text** | NLTK, spaCy, transformers |
| **Images** | PIL, OpenCV, torchvision, albumentations |
| **Time-series** | tsfresh, pandas |
| **Geospatial** | geopandas, shapely |

**For Scale:**

| Size | Library |
|------|---------|
| **< 1GB** | pandas |
| **1-100GB** | Dask, Polars |
| **> 100GB** | PySpark |

---

## Question 21
- [ ] Done

**How do you keep track of different preprocessing and feature engineering steps you have tested?**

**Answer:**

Track preprocessing experiments using version control, experiment tracking tools, and reproducible pipelines. Document transformations, parameters, and results to compare approaches and ensure reproducibility.

**Tracking Methods:**

| Method | Tool | Purpose |
|--------|------|---------|
| **Code versioning** | Git | Track code changes |
| **Experiment tracking** | MLflow, W&B, Neptune | Log parameters, metrics |
| **Pipeline serialization** | joblib, pickle | Save fitted transformers |
| **Config files** | YAML, JSON | Document parameters |
| **Notebooks** | Jupyter + Git | Document exploration |

**Example with MLflow:**
```python
import mlflow

with mlflow.start_run():
    # Log preprocessing parameters
    mlflow.log_params({
        'imputer_strategy': 'median',
        'scaler': 'standard',
        'encoding': 'one_hot'
    })
    
    # Log artifacts
    mlflow.sklearn.log_model(pipeline, 'preprocessing_pipeline')
    
    # Log metrics
    mlflow.log_metric('null_count_after', df.isnull().sum().sum())
```

---

## Question 22
- [ ] Done

**How can deep learning be used for feature extraction in unstructured data?**

**Answer:**

Deep learning models automatically learn hierarchical representations from raw unstructured data (images, text, audio). Pre-trained models serve as feature extractors, producing dense embeddings that capture semantic meaning for downstream tasks.

**Feature Extraction by Data Type:**

| Data | Model | Output |
|------|-------|--------|
| **Images** | ResNet, VGG, EfficientNet | Feature vectors (e.g., 2048-dim) |
| **Text** | BERT, GPT, Sentence-BERT | Contextualized embeddings (768-dim) |
| **Audio** | Wav2Vec, VGGish | Audio embeddings |

**How It Works:**
1. Use pre-trained model trained on large dataset
2. Remove final classification layer
3. Pass data through model
4. Extract intermediate representations

**Python Example (Image Features):**
```python
from torchvision import models, transforms
import torch

# Load pre-trained model
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classifier
model.eval()

# Extract features
with torch.no_grad():
    features = model(image_tensor)  # Shape: (batch, 2048)
```

---

## Question 23
- [ ] Done

**How is unsupervised learning used for preprocessing and feature extraction?**

**Answer:**

Unsupervised learning discovers structure in data without labels, enabling dimensionality reduction, clustering-based features, and anomaly detection during preprocessing. It's valuable when labeled data is scarce or for data exploration.

**Applications:**

| Technique | Use in Preprocessing |
|-----------|---------------------|
| **PCA** | Dimensionality reduction, noise removal |
| **Autoencoders** | Learn compressed representations |
| **K-Means** | Create cluster-based features |
| **DBSCAN** | Anomaly/outlier detection |
| **t-SNE/UMAP** | Visualization, embedding features |

**Example (Cluster Features):**
```python
from sklearn.cluster import KMeans

# Add cluster membership as feature
kmeans = KMeans(n_clusters=5)
df['cluster'] = kmeans.fit_predict(X)

# Add distance to cluster centers as features
distances = kmeans.transform(X)  # Distance to each center
for i in range(5):
    df[f'dist_cluster_{i}'] = distances[:, i]
```

**Autoencoder for Feature Learning:**
```python
# Train autoencoder on unlabeled data
# Use encoder part to transform data to lower dimension
encoded_features = encoder.predict(X)
```

---

## Question 24
- [ ] Done

**How do advances in hardware (like GPUs, TPUs) influence data processing techniques?**

**Answer:**

Hardware advances enable processing larger datasets faster, running complex preprocessing in parallel, and using deep learning for feature extraction. GPU/TPU acceleration makes previously impractical techniques viable at scale.

**Impact Areas:**

| Aspect | Impact |
|--------|--------|
| **Batch processing** | Larger batches processed in parallel |
| **Deep learning features** | Neural feature extractors now practical |
| **Real-time processing** | Low-latency preprocessing possible |
| **Data augmentation** | On-the-fly augmentation during training |
| **Large-scale transformations** | Matrix operations accelerated |

**Hardware-Accelerated Libraries:**

| Library | Description |
|---------|-------------|
| **RAPIDS (cuDF)** | GPU-accelerated pandas |
| **cuML** | GPU-accelerated scikit-learn |
| **DALI** | GPU data loading and augmentation |
| **tf.data, DataLoader** | Optimized data pipelines |

**Example (GPU DataFrame):**
```python
import cudf  # GPU pandas

# Load and process on GPU
gdf = cudf.read_csv('large_file.csv')
gdf = gdf.fillna(gdf.mean())  # GPU-accelerated
```

---

## Question 25
- [ ] Done

**How can preprocessing steps impact data bias in your models?**

**Answer:**

Preprocessing can introduce, amplify, or mitigate bias depending on how missing values, outliers, and categories are handled. Biased preprocessing leads to models that perform unfairly across different demographic groups.

**How Preprocessing Introduces Bias:**

| Step | Potential Bias |
|------|----------------|
| **Mean imputation** | Different means across groups affect imputed values |
| **Outlier removal** | May disproportionately remove certain groups |
| **Category grouping** | Lumping minorities into "Other" |
| **Feature selection** | May remove features protecting fairness |
| **Sampling** | Under/over-representation of groups |

**Mitigation Strategies:**
- Analyze preprocessing impact per demographic group
- Use group-aware imputation
- Check distributions before and after preprocessing
- Include fairness metrics in evaluation

**Example Check:**
```python
# Check imputation impact by group
for group in df['demographic'].unique():
    subset = df[df['demographic'] == group]
    missing_rate = subset['income'].isnull().mean()
    imputed_mean = subset['income'].mean()
    print(f"{group}: {missing_rate:.2%} missing, mean={imputed_mean:.2f}")
```

---

## Question 26
- [ ] Done

**What measures can be taken to prevent introducing bias during data cleaning?**

**Answer:**

Preventing bias requires awareness of how cleaning decisions affect different groups, using group-stratified approaches, documenting decisions, and validating that cleaning preserves fairness metrics.

**Prevention Measures:**

| Measure | Implementation |
|---------|----------------|
| **Stratified analysis** | Analyze cleaning impact per group |
| **Group-aware imputation** | Impute within groups, not globally |
| **Document decisions** | Record why data was removed |
| **Balance validation** | Check class balance after cleaning |
| **Fairness metrics** | Track demographic parity |

**Best Practices:**
```python
# Group-aware imputation
for group in df['demographic'].unique():
    mask = df['demographic'] == group
    group_median = df.loc[mask, 'income'].median()
    df.loc[mask & df['income'].isna(), 'income'] = group_median

# Validate balance
print(df['demographic'].value_counts(normalize=True))
print(df.groupby('demographic')['target'].mean())
```

---

## Question 27
- [ ] Done

**How do data preprocessing requirements differ between industries like finance, healthcare, and retail?**

**Answer:**

Each industry has unique data characteristics, regulatory requirements, and business priorities that shape preprocessing approaches. Understanding domain context is essential for effective preprocessing.

**Industry-Specific Requirements:**

| Industry | Key Considerations |
|----------|-------------------|
| **Finance** | High precision, fraud detection, regulatory compliance, time-series focus |
| **Healthcare** | Privacy (HIPAA), missing data common, imbalanced (rare diseases), clinical validation |
| **Retail** | High volume, real-time, seasonality, customer segmentation |

**Detailed Comparison:**

| Aspect | Finance | Healthcare | Retail |
|--------|---------|------------|--------|
| **Missing data** | Must handle carefully (financial impact) | Common (patient dropout) | Often ignorable |
| **Outliers** | Critical (fraud signals) | May be meaningful (rare conditions) | Filter noise |
| **Time sensitivity** | Real-time for trading | Batch OK | Near real-time |
| **Regulations** | SOX, PCI-DSS | HIPAA, FDA | GDPR for customers |
| **Scale** | Moderate, high frequency | Moderate | Very high volume |

---

## Question 28
- [ ] Done

**How do data privacy regulations affect data preprocessing in sensitive fields?**

**Answer:**

Regulations like GDPR, HIPAA, and CCPA mandate how personal data must be handled during preprocessing. This includes anonymization, consent tracking, data minimization, and audit trails for all transformations.

**Key Requirements:**

| Regulation | Requirements |
|------------|--------------|
| **GDPR** | Consent, right to deletion, data minimization, anonymization |
| **HIPAA** | De-identification of PHI, access controls, audit logs |
| **CCPA** | Disclosure, opt-out rights, data handling transparency |

**Preprocessing Implications:**

| Requirement | Implementation |
|-------------|----------------|
| **Anonymization** | Remove/hash identifiers before processing |
| **Data minimization** | Only process necessary features |
| **Audit trail** | Log all transformations |
| **Secure processing** | Encryption, access control |

**Techniques:**
- **K-anonymity:** Ensure each record matches k-1 others
- **Differential privacy:** Add noise to preserve privacy
- **Pseudonymization:** Replace identifiers with pseudonyms

**Example:**
```python
import hashlib

# Pseudonymize identifier
df['user_id_hash'] = df['user_id'].apply(
    lambda x: hashlib.sha256(x.encode()).hexdigest()
)
df = df.drop('user_id', axis=1)  # Remove original
```
