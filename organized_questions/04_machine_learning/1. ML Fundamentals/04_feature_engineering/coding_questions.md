# Feature Engineering Interview Questions - Coding Questions

## Question 1: Write a function to normalize a feature vector using Min-Max normalization.
- [ ] Done

### Answer

**Min-Max Normalization Formula:**

$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

This scales features to a range of [0, 1].

**Implementation:**

```python
import numpy as np

def min_max_normalize(feature_vector):
    """
    Normalize a feature vector using Min-Max normalization.
    
    Args:
        feature_vector: numpy array or list of numerical values
        
    Returns:
        Normalized array with values in range [0, 1]
    """
    feature_vector = np.array(feature_vector, dtype=float)
    
    min_val = np.min(feature_vector)
    max_val = np.max(feature_vector)
    
    # Handle edge case where all values are the same
    if max_val - min_val == 0:
        return np.zeros_like(feature_vector)
    
    normalized = (feature_vector - min_val) / (max_val - min_val)
    return normalized


# Example usage
data = [10, 20, 30, 40, 50]
normalized_data = min_max_normalize(data)
print(f"Original: {data}")
print(f"Normalized: {normalized_data}")
# Output: [0.0, 0.25, 0.5, 0.75, 1.0]
```

**Scikit-learn Alternative:**

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# For 2D data (required by sklearn)
data = np.array([[10], [20], [30], [40], [50]])
scaler = MinMaxScaler()
normalized = scaler.fit_transform(data)
print(normalized.flatten())  # [0.0, 0.25, 0.5, 0.75, 1.0]
```

**Custom Range Normalization:**

```python
def min_max_normalize_range(feature_vector, new_min=0, new_max=1):
    """
    Normalize to custom range [new_min, new_max]
    """
    feature_vector = np.array(feature_vector, dtype=float)
    min_val = np.min(feature_vector)
    max_val = np.max(feature_vector)
    
    if max_val - min_val == 0:
        return np.full_like(feature_vector, (new_min + new_max) / 2)
    
    normalized = (feature_vector - min_val) / (max_val - min_val)
    return normalized * (new_max - new_min) + new_min

# Scale to [-1, 1]
scaled = min_max_normalize_range([10, 20, 30, 40, 50], -1, 1)
print(scaled)  # [-1.0, -0.5, 0.0, 0.5, 1.0]
```

---

## Question 2: Write a Python function to perform one-hot encoding on a categorical feature.
- [ ] Done

### Answer

**One-Hot Encoding:**
Converts categorical values into binary vectors where only one element is "hot" (1) at a time.

**Implementation from Scratch:**

```python
import numpy as np
import pandas as pd

def one_hot_encode(categorical_feature):
    """
    Perform one-hot encoding on a categorical feature.
    
    Args:
        categorical_feature: list or array of categorical values
        
    Returns:
        DataFrame with one-hot encoded columns
    """
    # Get unique categories
    categories = sorted(set(categorical_feature))
    
    # Create encoding
    encoded = []
    for value in categorical_feature:
        row = [1 if value == cat else 0 for cat in categories]
        encoded.append(row)
    
    # Return as DataFrame for readability
    return pd.DataFrame(encoded, columns=categories)


# Example usage
colors = ['red', 'green', 'blue', 'red', 'green']
encoded = one_hot_encode(colors)
print(encoded)
#    blue  green  red
# 0     0      0    1
# 1     0      1    0
# 2     1      0    0
# 3     0      0    1
# 4     0      1    0
```

**Using Pandas:**

```python
import pandas as pd

def one_hot_encode_pandas(df, column, drop_first=False):
    """
    One-hot encode a column using pandas.
    
    Args:
        df: DataFrame
        column: column name to encode
        drop_first: whether to drop first category (avoid multicollinearity)
        
    Returns:
        DataFrame with encoded columns
    """
    return pd.get_dummies(df, columns=[column], drop_first=drop_first)

# Example
df = pd.DataFrame({'color': ['red', 'green', 'blue', 'red']})
encoded_df = one_hot_encode_pandas(df, 'color')
print(encoded_df)
```

**Using Scikit-learn:**

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def one_hot_encode_sklearn(categorical_feature):
    """
    One-hot encoding using sklearn.
    Handles unknown categories in transform.
    """
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    # Reshape for sklearn
    data = np.array(categorical_feature).reshape(-1, 1)
    encoded = encoder.fit_transform(data)
    
    # Get category names
    categories = encoder.categories_[0]
    return pd.DataFrame(encoded, columns=categories)

# Example
colors = ['red', 'green', 'blue', 'red']
encoded = one_hot_encode_sklearn(colors)
print(encoded)
```

---

## Question 3: Write a Python function to perform PCA on a dataset and reduce its dimensions.
- [ ] Done

### Answer

**PCA (Principal Component Analysis):**
Transforms data to a lower-dimensional space by finding directions of maximum variance.

**Implementation from Scratch:**

```python
import numpy as np

def pca_from_scratch(X, n_components):
    """
    Perform PCA from scratch.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        n_components: number of principal components to keep
        
    Returns:
        Transformed data of shape (n_samples, n_components)
    """
    # Step 1: Center the data (subtract mean)
    X_centered = X - np.mean(X, axis=0)
    
    # Step 2: Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Step 4: Sort eigenvectors by eigenvalues (descending)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Step 5: Select top n_components eigenvectors
    components = sorted_eigenvectors[:, :n_components]
    
    # Step 6: Transform data
    X_transformed = np.dot(X_centered, components)
    
    # Calculate explained variance ratio
    explained_variance = eigenvalues[sorted_indices][:n_components]
    explained_variance_ratio = explained_variance / np.sum(eigenvalues)
    
    return X_transformed, explained_variance_ratio


# Example usage
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data

X_pca, variance_ratio = pca_from_scratch(X, n_components=2)
print(f"Shape: {X.shape} -> {X_pca.shape}")
print(f"Explained variance ratio: {variance_ratio}")
```

**Using Scikit-learn:**

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

def perform_pca(X, n_components=None, variance_threshold=0.95):
    """
    Perform PCA with optional variance threshold.
    
    Args:
        X: Feature matrix
        n_components: Number of components (if None, use variance_threshold)
        variance_threshold: Minimum cumulative explained variance
        
    Returns:
        Transformed data, PCA object
    """
    # Standardize data first (important!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA
    if n_components is None:
        pca = PCA()
        pca.fit(X_scaled)
        
        # Find number of components for variance threshold
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Final PCA with selected components
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X_scaled)
    
    print(f"Components: {n_components}")
    print(f"Explained variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    return X_transformed, pca


# Example
from sklearn.datasets import load_iris
iris = load_iris()
X_pca, pca = perform_pca(iris.data, variance_threshold=0.95)
print(f"Original shape: {iris.data.shape}")
print(f"Reduced shape: {X_pca.shape}")
```

**Visualization:**

```python
import matplotlib.pyplot as plt

def plot_pca_2d(X_pca, y, title="PCA Visualization"):
    """Plot 2D PCA results."""
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(title)
    plt.colorbar(scatter)
    plt.show()
```

---

## Question 4: Use Scikit-learn to select the top k features using SelectKBest.
- [ ] Done

### Answer

**SelectKBest:**
Selects features based on univariate statistical tests.

**Implementation:**

```python
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2, mutual_info_classif
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

def select_k_best_features(X, y, k=5, score_func=f_classif, feature_names=None):
    """
    Select top k features using SelectKBest.
    
    Args:
        X: Feature matrix
        y: Target vector
        k: Number of features to select
        score_func: Scoring function (f_classif, chi2, mutual_info_classif)
        feature_names: Optional list of feature names
        
    Returns:
        Selected features, selector object, scores DataFrame
    """
    # Initialize selector
    selector = SelectKBest(score_func=score_func, k=k)
    
    # Fit and transform
    X_selected = selector.fit_transform(X, y)
    
    # Get scores and feature names
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Create scores DataFrame
    scores_df = pd.DataFrame({
        'Feature': feature_names,
        'Score': selector.scores_,
        'P-value': selector.pvalues_ if hasattr(selector, 'pvalues_') else None,
        'Selected': selector.get_support()
    }).sort_values('Score', ascending=False)
    
    # Get selected feature names
    selected_features = scores_df[scores_df['Selected']]['Feature'].tolist()
    
    return X_selected, selector, scores_df, selected_features


# Example with Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

X_selected, selector, scores_df, selected_features = select_k_best_features(
    X, y, k=2, 
    score_func=f_classif,
    feature_names=iris.feature_names
)

print("Feature Scores:")
print(scores_df)
print(f"\nSelected features: {selected_features}")
print(f"Shape: {X.shape} -> {X_selected.shape}")
```

**Different Score Functions:**

```python
# For classification:
# - f_classif: ANOVA F-statistic (for continuous features)
# - chi2: Chi-squared (for non-negative features)
# - mutual_info_classif: Mutual information (captures non-linear relationships)

# For regression:
# - f_regression: F-statistic
# - mutual_info_regression: Mutual information

from sklearn.feature_selection import f_regression, mutual_info_regression

# Classification example
selector_clf = SelectKBest(score_func=mutual_info_classif, k=3)
X_selected_clf = selector_clf.fit_transform(X, y)

# Regression example (if y were continuous)
# selector_reg = SelectKBest(score_func=f_regression, k=3)
# X_selected_reg = selector_reg.fit_transform(X, y_continuous)
```

**Pipeline Integration:**

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Create pipeline with feature selection
pipeline = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif, k=2)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Cross-validate
scores = cross_val_score(pipeline, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

---

## Question 5: Write a function to impute missing values in a dataset using mean, median, or mode.
- [ ] Done

### Answer

**Implementation:**

```python
import numpy as np
import pandas as pd
from scipy import stats

def impute_missing_values(df, strategy='mean', columns=None):
    """
    Impute missing values using specified strategy.
    
    Args:
        df: DataFrame with missing values
        strategy: 'mean', 'median', 'mode', or 'constant'
        columns: Specific columns to impute (default: all)
        
    Returns:
        DataFrame with imputed values, imputation statistics
    """
    df_imputed = df.copy()
    imputation_stats = {}
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if df[col].isnull().sum() == 0:
            continue
            
        if strategy == 'mean':
            if df[col].dtype in ['float64', 'int64']:
                fill_value = df[col].mean()
            else:
                fill_value = df[col].mode()[0]
        elif strategy == 'median':
            if df[col].dtype in ['float64', 'int64']:
                fill_value = df[col].median()
            else:
                fill_value = df[col].mode()[0]
        elif strategy == 'mode':
            fill_value = df[col].mode()[0]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        df_imputed[col].fillna(fill_value, inplace=True)
        imputation_stats[col] = {
            'strategy': strategy,
            'fill_value': fill_value,
            'missing_count': df[col].isnull().sum()
        }
    
    return df_imputed, imputation_stats


# Example usage
df = pd.DataFrame({
    'age': [25, 30, np.nan, 35, 40, np.nan],
    'salary': [50000, np.nan, 60000, np.nan, 70000, 80000],
    'city': ['NY', 'LA', np.nan, 'NY', 'SF', 'LA']
})

print("Original DataFrame:")
print(df)
print(f"\nMissing values:\n{df.isnull().sum()}")

# Impute with different strategies
df_mean, stats_mean = impute_missing_values(df, strategy='mean')
print("\nMean imputation:")
print(df_mean)

df_median, stats_median = impute_missing_values(df, strategy='median')
print("\nMedian imputation:")
print(df_median)
```

**Using Scikit-learn:**

```python
from sklearn.impute import SimpleImputer
import numpy as np

def sklearn_impute(X, strategy='mean'):
    """
    Impute using sklearn SimpleImputer.
    
    Args:
        X: 2D array or DataFrame
        strategy: 'mean', 'median', 'most_frequent', 'constant'
        
    Returns:
        Imputed array
    """
    imputer = SimpleImputer(strategy=strategy)
    return imputer.fit_transform(X)


# Separate imputers for numeric and categorical
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_imputation_pipeline(numeric_features, categorical_features):
    """Create preprocessing pipeline with appropriate imputers."""
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat', SimpleImputer(strategy='most_frequent'), categorical_features)
        ])
    
    return preprocessor
```

**Advanced: KNN Imputation:**

```python
from sklearn.impute import KNNImputer

def knn_impute(X, n_neighbors=5):
    """
    Impute missing values using KNN.
    Better for data where values depend on neighboring samples.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    return imputer.fit_transform(X)

# Example
X_with_nan = np.array([[1, 2, np.nan], [3, np.nan, 6], [7, 8, 9]])
X_imputed = knn_impute(X_with_nan, n_neighbors=2)
print(X_imputed)
```

---

## Question 6: Write a SQL query to create derived feature columns from existing columns.
- [ ] Done

### Answer

**Common Derived Features in SQL:**

```sql
-- Create derived features for customer analysis
SELECT 
    customer_id,
    first_name,
    last_name,
    
    -- Date features
    DATEDIFF(CURDATE(), date_of_birth) / 365 AS age,
    DATEDIFF(CURDATE(), account_created_date) AS account_age_days,
    YEAR(last_purchase_date) AS last_purchase_year,
    MONTH(last_purchase_date) AS last_purchase_month,
    DAYOFWEEK(last_purchase_date) AS last_purchase_dow,
    
    -- Aggregated features
    total_purchases,
    total_amount,
    
    -- Ratio features
    total_amount / NULLIF(total_purchases, 0) AS avg_purchase_value,
    returns_count / NULLIF(total_purchases, 0) AS return_rate,
    
    -- Recency features
    DATEDIFF(CURDATE(), last_purchase_date) AS days_since_last_purchase,
    
    -- Binary flags
    CASE WHEN email_verified = 1 THEN 1 ELSE 0 END AS is_verified,
    CASE WHEN total_purchases >= 10 THEN 1 ELSE 0 END AS is_frequent_buyer,
    CASE WHEN DATEDIFF(CURDATE(), last_purchase_date) <= 30 THEN 1 ELSE 0 END AS is_active,
    
    -- Binned features
    CASE 
        WHEN total_amount < 100 THEN 'Low'
        WHEN total_amount < 500 THEN 'Medium'
        WHEN total_amount < 1000 THEN 'High'
        ELSE 'VIP'
    END AS customer_segment
    
FROM customers;
```

**RFM Features (Recency, Frequency, Monetary):**

```sql
-- Create RFM features for customer segmentation
WITH customer_metrics AS (
    SELECT 
        customer_id,
        DATEDIFF(CURDATE(), MAX(order_date)) AS recency,
        COUNT(DISTINCT order_id) AS frequency,
        SUM(order_total) AS monetary
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR)
    GROUP BY customer_id
),
rfm_scores AS (
    SELECT 
        customer_id,
        recency,
        frequency,
        monetary,
        NTILE(5) OVER (ORDER BY recency DESC) AS r_score,
        NTILE(5) OVER (ORDER BY frequency ASC) AS f_score,
        NTILE(5) OVER (ORDER BY monetary ASC) AS m_score
    FROM customer_metrics
)
SELECT 
    customer_id,
    recency,
    frequency,
    monetary,
    r_score,
    f_score,
    m_score,
    CONCAT(r_score, f_score, m_score) AS rfm_segment,
    r_score + f_score + m_score AS rfm_score
FROM rfm_scores;
```

**Lag and Rolling Features (SQL Server/PostgreSQL):**

```sql
-- Time series features with window functions
SELECT 
    date,
    product_id,
    sales,
    
    -- Lag features
    LAG(sales, 1) OVER (PARTITION BY product_id ORDER BY date) AS sales_lag_1,
    LAG(sales, 7) OVER (PARTITION BY product_id ORDER BY date) AS sales_lag_7,
    
    -- Lead features (if applicable)
    LEAD(sales, 1) OVER (PARTITION BY product_id ORDER BY date) AS sales_next_day,
    
    -- Rolling averages
    AVG(sales) OVER (
        PARTITION BY product_id 
        ORDER BY date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS sales_rolling_avg_7d,
    
    -- Cumulative sum
    SUM(sales) OVER (
        PARTITION BY product_id 
        ORDER BY date
    ) AS sales_cumsum,
    
    -- Percent change
    (sales - LAG(sales, 1) OVER (PARTITION BY product_id ORDER BY date)) * 100.0 / 
        NULLIF(LAG(sales, 1) OVER (PARTITION BY product_id ORDER BY date), 0) AS pct_change
    
FROM daily_sales
ORDER BY product_id, date;
```

---

## Question 7: Implement feature hashing in Python to handle high-cardinality categorical features.
- [ ] Done

### Answer

**Feature Hashing (Hashing Trick):**
Maps high-cardinality categorical features to a fixed-size feature vector using hash functions.

**Implementation from Scratch:**

```python
import numpy as np
import mmh3  # MurmurHash3

def feature_hashing(values, n_features=100, signed=True):
    """
    Implement feature hashing from scratch.
    
    Args:
        values: List of categorical values
        n_features: Size of the hashed feature vector
        signed: Whether to use signed hash (reduces collisions)
        
    Returns:
        Hashed feature matrix
    """
    n_samples = len(values)
    hashed_matrix = np.zeros((n_samples, n_features))
    
    for i, value in enumerate(values):
        # Hash the value to get the index
        hash_index = mmh3.hash(str(value)) % n_features
        
        if signed:
            # Use second hash for sign
            sign = 1 if mmh3.hash(str(value), seed=42) % 2 == 0 else -1
            hashed_matrix[i, hash_index] = sign
        else:
            hashed_matrix[i, hash_index] = 1
    
    return hashed_matrix


# Example usage
categories = ['user_123', 'user_456', 'user_789', 'user_123', 'user_999']
hashed = feature_hashing(categories, n_features=10)
print("Hashed features shape:", hashed.shape)
print(hashed)
```

**Using Scikit-learn:**

```python
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer
import pandas as pd

def sklearn_feature_hashing(df, columns, n_features=100):
    """
    Feature hashing using sklearn FeatureHasher.
    
    Args:
        df: DataFrame
        columns: Columns to hash
        n_features: Number of features per column
        
    Returns:
        Hashed feature matrix
    """
    # Create list of dictionaries
    records = df[columns].to_dict(orient='records')
    
    # Apply feature hashing
    hasher = FeatureHasher(n_features=n_features, input_type='dict')
    hashed_features = hasher.transform(records)
    
    return hashed_features.toarray()


# Example
df = pd.DataFrame({
    'user_id': ['user_123', 'user_456', 'user_789', 'user_123'],
    'product_id': ['prod_A', 'prod_B', 'prod_A', 'prod_C']
})

hashed = sklearn_feature_hashing(df, ['user_id', 'product_id'], n_features=20)
print("Hashed shape:", hashed.shape)
```

**For Text Data:**

```python
from sklearn.feature_extraction.text import HashingVectorizer

def hash_text_features(texts, n_features=1000):
    """
    Hash text data using HashingVectorizer.
    Memory-efficient for large vocabularies.
    """
    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=True,  # Reduces collision impact
        norm='l2'  # Normalize vectors
    )
    return vectorizer.fit_transform(texts)


# Example
texts = [
    "machine learning is great",
    "deep learning and neural networks",
    "feature engineering for ML"
]
hashed_text = hash_text_features(texts, n_features=50)
print("Hashed text shape:", hashed_text.shape)
```

**Advantages:**
- Fixed memory regardless of cardinality
- No need to maintain vocabulary
- Handles unseen categories
- Fast transformation

**Disadvantages:**
- Hash collisions (different values → same index)
- Not invertible (can't recover original values)
- Harder to interpret

---

## Question 8: Write a function to identify and remove highly correlated features from a dataset.
- [ ] Done

### Answer

**Implementation:**

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def remove_correlated_features(df, threshold=0.9, target_col=None):
    """
    Remove highly correlated features from dataset.
    
    Args:
        df: DataFrame with features
        threshold: Correlation threshold (0-1)
        target_col: Target column name (excluded from removal)
        
    Returns:
        Cleaned DataFrame, list of removed features
    """
    # Exclude target column if specified
    if target_col:
        feature_df = df.drop(columns=[target_col])
    else:
        feature_df = df.copy()
    
    # Only numeric columns
    numeric_df = feature_df.select_dtypes(include=[np.number])
    
    # Compute correlation matrix
    corr_matrix = numeric_df.corr().abs()
    
    # Get upper triangle of correlation matrix
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features with correlation above threshold
    to_remove = []
    for column in upper_triangle.columns:
        correlated = upper_triangle[column][upper_triangle[column] > threshold]
        if not correlated.empty:
            to_remove.append(column)
    
    # Remove duplicates
    to_remove = list(set(to_remove))
    
    # Create cleaned DataFrame
    df_cleaned = df.drop(columns=to_remove)
    
    return df_cleaned, to_remove


def visualize_correlations(df, threshold=0.9):
    """Visualize correlation matrix with threshold."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                center=0, fmt='.2f', linewidths=0.5)
    plt.title(f'Correlation Matrix (threshold={threshold})')
    plt.tight_layout()
    plt.show()


# Example usage
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Remove correlated features
df_cleaned, removed = remove_correlated_features(df, threshold=0.9)

print(f"Original features: {len(df.columns)}")
print(f"Removed features: {len(removed)}")
print(f"Remaining features: {len(df_cleaned.columns)}")
print(f"\nRemoved: {removed[:10]}...")  # Show first 10
```

**Smarter Version (Keep Feature with Higher Target Correlation):**

```python
def smart_remove_correlated(df, target_col, threshold=0.9):
    """
    Remove correlated features, keeping the one with
    higher correlation to target.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Compute feature-feature correlations
    corr_matrix = X.corr().abs()
    
    # Compute feature-target correlations
    target_corr = X.corrwith(y).abs()
    
    # Find pairs above threshold
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_remove = set()
    for i in range(len(upper.columns)):
        for j in range(i+1, len(upper.columns)):
            if upper.iloc[i, j] > threshold:
                col_i = upper.columns[i]
                col_j = upper.columns[j]
                
                # Remove the one with lower target correlation
                if target_corr[col_i] < target_corr[col_j]:
                    to_remove.add(col_i)
                else:
                    to_remove.add(col_j)
    
    df_cleaned = df.drop(columns=list(to_remove))
    return df_cleaned, list(to_remove)
```

---

## Question 9: Implement polynomial feature generation in Python.
- [ ] Done

### Answer

**Polynomial Features:**
Creates new features as polynomial combinations of existing features.

**Implementation from Scratch:**

```python
import numpy as np
from itertools import combinations_with_replacement

def polynomial_features_scratch(X, degree=2, include_bias=True, interaction_only=False):
    """
    Generate polynomial features from scratch.
    
    Args:
        X: Input array (n_samples, n_features)
        degree: Maximum polynomial degree
        include_bias: Whether to include bias term (all 1s)
        interaction_only: Only include interaction terms, not powers
        
    Returns:
        Array with polynomial features
    """
    X = np.array(X)
    n_samples, n_features = X.shape
    
    features = []
    feature_names = []
    
    # Bias term
    if include_bias:
        features.append(np.ones(n_samples))
        feature_names.append('1')
    
    # Generate polynomial combinations
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            if interaction_only and len(set(combo)) != len(combo):
                continue  # Skip non-interaction terms like x1^2
            
            feature = np.prod([X[:, i] for i in combo], axis=0)
            features.append(feature)
            
            # Create name
            name = ' * '.join([f'x{i}' for i in combo])
            feature_names.append(name)
    
    return np.column_stack(features), feature_names


# Example
X = np.array([[1, 2], [3, 4], [5, 6]])
X_poly, names = polynomial_features_scratch(X, degree=2)
print("Feature names:", names)
print("Shape:", X_poly.shape)
print(X_poly)
```

**Using Scikit-learn:**

```python
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

def generate_polynomial_features(X, degree=2, interaction_only=False, include_bias=False):
    """
    Generate polynomial features using sklearn.
    
    Args:
        X: Input array or DataFrame
        degree: Maximum polynomial degree
        interaction_only: Only interaction terms
        include_bias: Include bias column
        
    Returns:
        Transformed array, feature names
    """
    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=include_bias
    )
    
    X_poly = poly.fit_transform(X)
    
    # Get feature names
    if hasattr(X, 'columns'):
        feature_names = poly.get_feature_names_out(X.columns)
    else:
        feature_names = poly.get_feature_names_out()
    
    return X_poly, feature_names


# Example
df = pd.DataFrame({
    'age': [25, 30, 35],
    'income': [50000, 60000, 70000]
})

X_poly, names = generate_polynomial_features(df, degree=2)
print("Feature names:", names)
print("Shape:", X_poly.shape)
print(pd.DataFrame(X_poly, columns=names))
```

**Pipeline Integration:**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', Ridge(alpha=1.0))
])

# Fit and evaluate
# scores = cross_val_score(pipeline, X, y, cv=5)
```

---

## Question 10: Write a function to scale features using standardization and MinMax scaling.
- [ ] Done

### Answer

**Implementation:**

```python
import numpy as np
import pandas as pd

class FeatureScaler:
    """Custom feature scaler supporting multiple scaling methods."""
    
    def __init__(self, method='standard'):
        """
        Args:
            method: 'standard' (z-score), 'minmax', 'robust', or 'maxabs'
        """
        self.method = method
        self.params = {}
    
    def fit(self, X):
        """Compute scaling parameters."""
        X = np.array(X)
        
        if self.method == 'standard':
            self.params['mean'] = np.mean(X, axis=0)
            self.params['std'] = np.std(X, axis=0)
            # Handle zero std
            self.params['std'][self.params['std'] == 0] = 1
            
        elif self.method == 'minmax':
            self.params['min'] = np.min(X, axis=0)
            self.params['max'] = np.max(X, axis=0)
            # Handle constant features
            range_val = self.params['max'] - self.params['min']
            range_val[range_val == 0] = 1
            self.params['range'] = range_val
            
        elif self.method == 'robust':
            self.params['median'] = np.median(X, axis=0)
            self.params['iqr'] = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
            self.params['iqr'][self.params['iqr'] == 0] = 1
            
        elif self.method == 'maxabs':
            self.params['max_abs'] = np.max(np.abs(X), axis=0)
            self.params['max_abs'][self.params['max_abs'] == 0] = 1
        
        return self
    
    def transform(self, X):
        """Apply scaling transformation."""
        X = np.array(X)
        
        if self.method == 'standard':
            return (X - self.params['mean']) / self.params['std']
            
        elif self.method == 'minmax':
            return (X - self.params['min']) / self.params['range']
            
        elif self.method == 'robust':
            return (X - self.params['median']) / self.params['iqr']
            
        elif self.method == 'maxabs':
            return X / self.params['max_abs']
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """Reverse the scaling."""
        X_scaled = np.array(X_scaled)
        
        if self.method == 'standard':
            return X_scaled * self.params['std'] + self.params['mean']
            
        elif self.method == 'minmax':
            return X_scaled * self.params['range'] + self.params['min']
            
        elif self.method == 'robust':
            return X_scaled * self.params['iqr'] + self.params['median']
            
        elif self.method == 'maxabs':
            return X_scaled * self.params['max_abs']


# Example usage
X = np.array([[1, 10, 100],
              [2, 20, 200],
              [3, 30, 300],
              [4, 40, 400]])

# Standard scaling
standard_scaler = FeatureScaler(method='standard')
X_standard = standard_scaler.fit_transform(X)
print("Standard Scaling:")
print(X_standard)
print(f"Mean: {X_standard.mean(axis=0)}")  # Should be ~0
print(f"Std: {X_standard.std(axis=0)}")    # Should be ~1

# MinMax scaling
minmax_scaler = FeatureScaler(method='minmax')
X_minmax = minmax_scaler.fit_transform(X)
print("\nMinMax Scaling:")
print(X_minmax)
print(f"Min: {X_minmax.min(axis=0)}")  # Should be 0
print(f"Max: {X_minmax.max(axis=0)}")  # Should be 1
```

**Using Scikit-learn:**

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import pandas as pd

def scale_features(df, method='standard', columns=None):
    """
    Scale features using sklearn scalers.
    
    Args:
        df: DataFrame
        method: 'standard', 'minmax', 'robust', 'maxabs'
        columns: Columns to scale (default: all numeric)
        
    Returns:
        Scaled DataFrame, scaler object
    """
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
        'maxabs': MaxAbsScaler()
    }
    
    scaler = scalers.get(method)
    if scaler is None:
        raise ValueError(f"Unknown method: {method}")
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    
    return df_scaled, scaler


# Example
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [30000, 50000, 70000, 90000, 110000],
    'score': [0.5, 0.6, 0.7, 0.8, 0.9]
})

df_standard, _ = scale_features(df, method='standard')
df_minmax, _ = scale_features(df, method='minmax')

print("Original:")
print(df)
print("\nStandard Scaled:")
print(df_standard)
print("\nMinMax Scaled:")
print(df_minmax)
```

**Comparison of Scaling Methods:**

| Method | Formula | Range | Use Case |
|--------|---------|-------|----------|
| **Standard** | $(x - \mu) / \sigma$ | Unbounded | Most algorithms |
| **MinMax** | $(x - min) / (max - min)$ | [0, 1] | Neural networks |
| **Robust** | $(x - median) / IQR$ | Unbounded | Outlier-prone data |
| **MaxAbs** | $x / max(\|x\|)$ | [-1, 1] | Sparse data |

---
