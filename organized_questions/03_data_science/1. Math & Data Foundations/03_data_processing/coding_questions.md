# Data Processing Interview Questions - Coding Questions

## Question 1
- [ ] Done

**How do you implement a data processing pipeline with scikit-learn?**

**Answer:**

A scikit-learn Pipeline chains preprocessing steps and models into a single object that ensures consistent transformations and prevents data leakage. It fits on training data and applies the same transformations to test data.

**Pipeline Structure:**
```
Raw Data → Imputer → Scaler → Encoder → Model → Predictions
```

**Python Code:**
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Define column types
num_features = ['age', 'income']
cat_features = ['city', 'gender']

# Numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine pipelines
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Full pipeline with model
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])

# Fit and predict
full_pipeline.fit(X_train, y_train)
predictions = full_pipeline.predict(X_test)
```

**Key Benefits:**
- Prevents data leakage (fit only on train)
- Reproducible transformations
- Easy to save/deploy entire pipeline

---

## Question 2
- [ ] Done

**Write a Python function to replace missing values with the median in a dataset.**

**Answer:**

**What it does:** Finds columns with missing values, calculates median from available data, fills missing values with that median.

**Python Code:**
```python
import pandas as pd
import numpy as np

def replace_with_median(df, columns=None):
    """
    Replace missing values with column median.
    
    Parameters:
    - df: pandas DataFrame
    - columns: list of columns to process (default: all numeric)
    
    Returns:
    - DataFrame with missing values filled
    """
    df = df.copy()
    
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            print(f"{col}: filled {df[col].isnull().sum()} values with median {median_value}")
    
    return df

# Usage
df = pd.DataFrame({
    'age': [25, 30, np.nan, 35, np.nan],
    'income': [50000, np.nan, 60000, 55000, 70000]
})

df_filled = replace_with_median(df)
print(df_filled)
```

**Output:**
```
age: filled 2 values with median 30.0
income: filled 1 values with median 57500.0
   age   income
0  25.0  50000.0
1  30.0  57500.0
2  30.0  60000.0
3  35.0  55000.0
4  30.0  70000.0
```

---

## Question 3
- [ ] Done

**Implement min-max scaling on a given dataset without using any libraries.**

**Answer:**

**Formula:** $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$

**What it does:** Finds min and max of each column, transforms values to [0, 1] range.

**Python Code:**
```python
def min_max_scale(data):
    """
    Min-max scaling from scratch.
    
    Parameters:
    - data: 2D list or list of lists
    
    Returns:
    - scaled_data: normalized data in [0, 1]
    - params: dict with min/max for each column (for inverse transform)
    """
    # Get dimensions
    n_rows = len(data)
    n_cols = len(data[0])
    
    # Calculate min and max for each column
    params = {}
    for j in range(n_cols):
        column = [data[i][j] for i in range(n_rows)]
        params[j] = {
            'min': min(column),
            'max': max(column)
        }
    
    # Scale the data
    scaled_data = []
    for i in range(n_rows):
        scaled_row = []
        for j in range(n_cols):
            min_val = params[j]['min']
            max_val = params[j]['max']
            
            if max_val - min_val == 0:
                scaled_val = 0  # Handle constant column
            else:
                scaled_val = (data[i][j] - min_val) / (max_val - min_val)
            
            scaled_row.append(scaled_val)
        scaled_data.append(scaled_row)
    
    return scaled_data, params

# Usage
data = [
    [10, 100],
    [20, 200],
    [30, 300],
    [40, 400],
    [50, 500]
]

scaled, params = min_max_scale(data)

print("Original:", data)
print("Scaled:", scaled)
# Output: [[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [1.0, 1.0]]
```

---

## Question 4
- [ ] Done

**Create a function to encode categorical variables using one-hot encoding in Python.**

**Answer:**

**What it does:** Gets unique categories, creates binary columns for each category, marks 1 for presence and 0 for absence.

**Python Code:**
```python
def one_hot_encode(data, column_index):
    """
    One-hot encode a categorical column from scratch.
    
    Parameters:
    - data: 2D list
    - column_index: index of column to encode
    
    Returns:
    - encoded_data: data with one-hot columns replacing original
    - categories: list of unique categories
    """
    n_rows = len(data)
    
    # Get unique categories
    categories = list(set(row[column_index] for row in data))
    categories.sort()  # For consistent ordering
    
    # Create encoded data
    encoded_data = []
    for row in data:
        new_row = []
        
        # Add columns before the encoded column
        for j in range(column_index):
            new_row.append(row[j])
        
        # Add one-hot encoded columns
        original_value = row[column_index]
        for cat in categories:
            new_row.append(1 if original_value == cat else 0)
        
        # Add columns after the encoded column
        for j in range(column_index + 1, len(row)):
            new_row.append(row[j])
        
        encoded_data.append(new_row)
    
    return encoded_data, categories

# Usage
data = [
    ['Red', 10],
    ['Blue', 20],
    ['Green', 30],
    ['Red', 40],
    ['Blue', 50]
]

encoded, cats = one_hot_encode(data, column_index=0)
print("Categories:", cats)
print("Encoded data:")
for row in encoded:
    print(row)

# Output:
# Categories: ['Blue', 'Green', 'Red']
# [0, 0, 1, 10]  # Red
# [1, 0, 0, 20]  # Blue
# [0, 1, 0, 30]  # Green
# [0, 0, 1, 40]  # Red
# [1, 0, 0, 50]  # Blue
```

---

## Question 5
- [ ] Done

**Use sklearn to set up a preprocessing pipeline with feature scaling and PCA.**

**Answer:**

**Pipeline flow:** Raw Data → Scaling → PCA → Reduced Dimensions

**Python Code:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load data
data = load_iris()
X = data.data
y = data.target

# Create pipeline
# Step 1: Scale features (required before PCA)
# Step 2: Reduce dimensions with PCA
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2))  # Reduce to 2 dimensions
])

# Fit and transform
X_transformed = pipeline.fit_transform(X)

print(f"Original shape: {X.shape}")       # (150, 4)
print(f"Transformed shape: {X_transformed.shape}")  # (150, 2)

# Check explained variance
print(f"Explained variance ratio: {pipeline.named_steps['pca'].explained_variance_ratio_}")

# For train/test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit on train, transform both
X_train_pca = pipeline.fit_transform(X_train)
X_test_pca = pipeline.transform(X_test)  # Uses same transformation
```

**Important:** Always scale before PCA because PCA is sensitive to feature scales.

---

## Question 6
- [ ] Done

**Write an SQL query to clean and preprocess a dataset with null values and outliers.**

**Answer:**

**SQL Code:**
```sql
-- Step 1: Create cleaned table handling nulls and outliers

-- View null counts
SELECT 
    COUNT(*) - COUNT(age) AS age_nulls,
    COUNT(*) - COUNT(income) AS income_nulls,
    COUNT(*) - COUNT(city) AS city_nulls
FROM customers;

-- Calculate statistics for outlier detection
WITH stats AS (
    SELECT 
        AVG(income) AS mean_income,
        STDDEV(income) AS std_income,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY age) AS median_age
    FROM customers
    WHERE income IS NOT NULL
),

-- Clean data
cleaned_data AS (
    SELECT 
        customer_id,
        -- Handle null age with median
        COALESCE(age, (SELECT median_age FROM stats)) AS age,
        -- Handle null income with mean, cap outliers at 3 std
        CASE 
            WHEN income IS NULL THEN (SELECT mean_income FROM stats)
            WHEN income > (SELECT mean_income + 3 * std_income FROM stats) 
                THEN (SELECT mean_income + 3 * std_income FROM stats)
            WHEN income < (SELECT mean_income - 3 * std_income FROM stats) 
                THEN (SELECT mean_income - 3 * std_income FROM stats)
            ELSE income
        END AS income,
        -- Handle null city with 'Unknown'
        COALESCE(city, 'Unknown') AS city
    FROM customers
)

SELECT * FROM cleaned_data;

-- Remove duplicates
SELECT DISTINCT * FROM cleaned_data;

-- Or keep first occurrence
SELECT * FROM (
    SELECT *, 
           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at) AS rn
    FROM customers
) t
WHERE rn = 1;
```

---

## Question 7
- [ ] Done

**Code a Python script to automatically detect and resolve duplicates in a dataset.**

**Answer:**

**What it does:** Identifies exact and fuzzy duplicates, provides options to keep first, last, or remove all.

**Python Code:**
```python
import pandas as pd

def detect_and_resolve_duplicates(df, subset=None, keep='first', 
                                   report=True):
    """
    Detect and resolve duplicate records.
    
    Parameters:
    - df: DataFrame
    - subset: columns to check for duplicates (None = all columns)
    - keep: 'first', 'last', or False (remove all duplicates)
    - report: print duplicate summary
    
    Returns:
    - cleaned DataFrame
    """
    original_count = len(df)
    
    # Detect duplicates
    duplicates = df.duplicated(subset=subset, keep=False)
    duplicate_count = duplicates.sum()
    
    if report:
        print(f"Total rows: {original_count}")
        print(f"Duplicate rows: {duplicate_count}")
        
        if duplicate_count > 0:
            print("\nSample duplicates:")
            print(df[duplicates].head(10))
    
    # Remove duplicates
    df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
    
    if report:
        print(f"\nRows after cleaning: {len(df_cleaned)}")
        print(f"Rows removed: {original_count - len(df_cleaned)}")
    
    return df_cleaned


def find_fuzzy_duplicates(df, column, threshold=80):
    """
    Find near-duplicate strings using fuzzy matching.
    """
    from fuzzywuzzy import fuzz
    
    values = df[column].unique()
    fuzzy_matches = []
    
    for i, val1 in enumerate(values):
        for val2 in values[i+1:]:
            similarity = fuzz.ratio(str(val1), str(val2))
            if similarity >= threshold:
                fuzzy_matches.append({
                    'value1': val1,
                    'value2': val2,
                    'similarity': similarity
                })
    
    return pd.DataFrame(fuzzy_matches)


# Usage
df = pd.DataFrame({
    'id': [1, 2, 2, 3, 4, 4],
    'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David'],
    'value': [100, 200, 200, 300, 400, 400]
})

# Exact duplicates
df_clean = detect_and_resolve_duplicates(df, keep='first')

# Duplicates based on specific columns
df_clean = detect_and_resolve_duplicates(df, subset=['id'], keep='first')
```

**Output:**
```
Total rows: 6
Duplicate rows: 4
Rows after cleaning: 4
Rows removed: 2
```

---

## Question 8
- [ ] Done

**Implement a time-series rolling window feature extraction in pandas.**

**Answer:**

**What it does:** Creates rolling statistics (mean, std, min, max) over a sliding window, maintaining temporal order.

**Python Code:**
```python
import pandas as pd
import numpy as np

def create_rolling_features(df, column, windows=[7, 14, 30]):
    """
    Create rolling window features for time-series.
    
    Parameters:
    - df: DataFrame with datetime index or date column
    - column: column to create features from
    - windows: list of window sizes
    
    Returns:
    - DataFrame with new features
    """
    df = df.copy()
    
    for window in windows:
        # Rolling mean
        df[f'{column}_rolling_mean_{window}'] = (
            df[column].rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling std
        df[f'{column}_rolling_std_{window}'] = (
            df[column].rolling(window=window, min_periods=1).std()
        )
        
        # Rolling min
        df[f'{column}_rolling_min_{window}'] = (
            df[column].rolling(window=window, min_periods=1).min()
        )
        
        # Rolling max
        df[f'{column}_rolling_max_{window}'] = (
            df[column].rolling(window=window, min_periods=1).max()
        )
        
        # Exponential moving average
        df[f'{column}_ewm_{window}'] = (
            df[column].ewm(span=window).mean()
        )
    
    # Lag features
    for lag in [1, 7, 14]:
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    
    # Percent change
    df[f'{column}_pct_change'] = df[column].pct_change()
    
    return df


# Usage
dates = pd.date_range('2023-01-01', periods=50, freq='D')
df = pd.DataFrame({
    'date': dates,
    'sales': np.random.randint(100, 500, 50)
})

df = create_rolling_features(df, 'sales', windows=[7, 14])
print(df.head(10))
```

**Output columns:**
- sales_rolling_mean_7, sales_rolling_std_7, etc.
- sales_lag_1, sales_lag_7, sales_lag_14
- sales_pct_change

---

## Question 9
- [ ] Done

**Write a Python function to perform sentiment encoding on text data.**

**Answer:**

**What it does:** Analyzes text sentiment and creates numerical features (polarity score, subjectivity, positive/negative indicators).

**Python Code:**
```python
from textblob import TextBlob
import pandas as pd

def sentiment_encoding(texts):
    """
    Encode text data with sentiment features.
    
    Parameters:
    - texts: list of text strings
    
    Returns:
    - DataFrame with sentiment features
    """
    results = []
    
    for text in texts:
        blob = TextBlob(str(text))
        
        # Sentiment polarity: -1 (negative) to 1 (positive)
        polarity = blob.sentiment.polarity
        
        # Subjectivity: 0 (objective) to 1 (subjective)
        subjectivity = blob.sentiment.subjectivity
        
        results.append({
            'text': text,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'is_positive': 1 if polarity > 0 else 0,
            'is_negative': 1 if polarity < 0 else 0,
            'is_neutral': 1 if polarity == 0 else 0,
            'sentiment_category': (
                'positive' if polarity > 0.1 
                else 'negative' if polarity < -0.1 
                else 'neutral'
            )
        })
    
    return pd.DataFrame(results)


# Simple version without library
def simple_sentiment(text, positive_words, negative_words):
    """
    Basic sentiment based on word counting.
    """
    words = text.lower().split()
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    
    total = pos_count + neg_count
    if total == 0:
        return 0
    return (pos_count - neg_count) / total


# Usage
texts = [
    "I love this product! It's amazing!",
    "Terrible experience, very disappointed",
    "The product is okay, nothing special",
    "Best purchase ever, highly recommend!"
]

df_sentiment = sentiment_encoding(texts)
print(df_sentiment)
```

**Output:**
```
                                      text  polarity  subjectivity  is_positive  sentiment_category
0       I love this product! It's amazing!      0.65          0.80            1            positive
1  Terrible experience, very disappointed     -0.60          0.90            0            negative
2     The product is okay, nothing special      0.10          0.50            1             neutral
3    Best purchase ever, highly recommend!      0.75          0.60            1            positive
```

---

## Question 10
- [ ] Done

**Perform image augmentation on a batch of images using TensorFlow or PyTorch.**

**Answer:**

**What it does:** Applies random transformations (flip, rotate, crop, color changes) to images to increase training data diversity.

**PyTorch Version:**
```python
import torch
from torchvision import transforms
from PIL import Image

# Define augmentation pipeline
train_augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# For inference (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply to image
def augment_image(image_path):
    image = Image.open(image_path).convert('RGB')
    augmented = train_augmentation(image)
    return augmented

# Apply to batch
def augment_batch(image_paths):
    return torch.stack([augment_image(p) for p in image_paths])
```

**TensorFlow Version:**
```python
import tensorflow as tf

def create_augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

# Apply during training
augmentation = create_augmentation_layer()

def preprocess_and_augment(image, label, training=True):
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0  # Normalize to [0, 1]
    
    if training:
        image = augmentation(image)
    
    return image, label
```

---

## Question 11
- [ ] Done

**Implement a custom transformer in sklearn that adds a new feature calculated from existing ones.**

**Answer:**

**What it does:** Creates custom preprocessing step that integrates into sklearn Pipeline, computing derived features from existing columns.

**Python Code:**
```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create derived features.
    """
    
    def __init__(self, add_ratios=True, add_interactions=True):
        self.add_ratios = add_ratios
        self.add_interactions = add_interactions
    
    def fit(self, X, y=None):
        # Store column names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        return self
    
    def transform(self, X):
        # Convert to array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = X.copy()
        new_features = [X]
        
        # Add ratio features
        if self.add_ratios and X.shape[1] >= 2:
            # Ratio of first two columns
            ratio = X[:, 0] / (X[:, 1] + 1e-8)  # Avoid division by zero
            new_features.append(ratio.reshape(-1, 1))
        
        # Add interaction features
        if self.add_interactions and X.shape[1] >= 2:
            # Product of first two columns
            interaction = X[:, 0] * X[:, 1]
            new_features.append(interaction.reshape(-1, 1))
        
        # Add polynomial features (squared)
        squared = X[:, 0] ** 2
        new_features.append(squared.reshape(-1, 1))
        
        return np.hstack(new_features)
    
    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_)
        if self.add_ratios:
            names.append('ratio_0_1')
        if self.add_interactions:
            names.append('interaction_0_1')
        names.append('squared_0')
        return names


# Usage in pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('feature_engineer', FeatureEngineer(add_ratios=True, add_interactions=True)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Example
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 0, 1, 1])

pipeline.fit(X, y)
print(f"Original features: 2")
print(f"After engineering: {pipeline.named_steps['feature_engineer'].transform(X).shape[1]}")
```

**Output:**
```
Original features: 2
After engineering: 5  # original(2) + ratio(1) + interaction(1) + squared(1)
```
