# Data Mining Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Write a SQL query that selects the top 3 most frequent purchasers from a sales table.**

**Answer:**

**Problem:** Find the top 3 customers with the most purchase transactions.

**Approach:**
1. Group by customer_id
2. Count transactions per customer
3. Order by count descending
4. Limit to top 3

```sql
-- Table: sales (customer_id, product_id, amount, purchase_date)

SELECT 
    customer_id,
    COUNT(*) AS purchase_count
FROM 
    sales
GROUP BY 
    customer_id
ORDER BY 
    purchase_count DESC
LIMIT 3;
```

**Alternative with Customer Name (JOIN):**
```sql
SELECT 
    c.customer_id,
    c.customer_name,
    COUNT(s.transaction_id) AS purchase_count
FROM 
    customers c
JOIN 
    sales s ON c.customer_id = s.customer_id
GROUP BY 
    c.customer_id, c.customer_name
ORDER BY 
    purchase_count DESC
LIMIT 3;
```

**For databases without LIMIT (SQL Server):**
```sql
SELECT TOP 3
    customer_id,
    COUNT(*) AS purchase_count
FROM 
    sales
GROUP BY 
    customer_id
ORDER BY 
    purchase_count DESC;
```

**Output Example:**
| customer_id | purchase_count |
|-------------|----------------|
| C101 | 45 |
| C203 | 38 |
| C157 | 32 |

---

## Question 2
- [ ] Done

**Implement the Apriori algorithm in Python to generate association rules from a transaction dataset.**

**Answer:**

**Problem:** Find frequent itemsets and generate association rules.

**Pipeline:**
1. Load transaction data
2. Find frequent itemsets (support >= min_support)
3. Generate rules (confidence >= min_confidence)
4. Return rules with support, confidence, lift

```python
from itertools import combinations
from collections import defaultdict

def get_support(itemset, transactions):
    """Calculate support of an itemset."""
    count = sum(1 for t in transactions if itemset.issubset(t))
    return count / len(transactions)

def get_frequent_itemsets(transactions, min_support):
    """Find all frequent itemsets using Apriori."""
    # Get unique items
    items = set(item for t in transactions for item in t)
    
    # Start with 1-itemsets
    frequent = {}
    k = 1
    current_itemsets = [frozenset([item]) for item in items]
    
    while current_itemsets:
        # Filter by support
        freq_k = {}
        for itemset in current_itemsets:
            sup = get_support(itemset, transactions)
            if sup >= min_support:
                freq_k[itemset] = sup
        
        frequent.update(freq_k)
        
        # Generate (k+1)-itemsets from k-itemsets
        k += 1
        items_k = list(freq_k.keys())
        current_itemsets = []
        
        for i in range(len(items_k)):
            for j in range(i+1, len(items_k)):
                union = items_k[i] | items_k[j]
                if len(union) == k and union not in current_itemsets:
                    current_itemsets.append(union)
    
    return frequent

def generate_rules(frequent_itemsets, transactions, min_confidence):
    """Generate association rules from frequent itemsets."""
    rules = []
    
    for itemset, support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
            
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                
                ant_support = get_support(antecedent, transactions)
                confidence = support / ant_support
                
                if confidence >= min_confidence:
                    cons_support = get_support(consequent, transactions)
                    lift = confidence / cons_support
                    
                    rules.append({
                        'antecedent': set(antecedent),
                        'consequent': set(consequent),
                        'support': round(support, 3),
                        'confidence': round(confidence, 3),
                        'lift': round(lift, 3)
                    })
    
    return rules

# Example usage
transactions = [
    {'bread', 'milk'},
    {'bread', 'butter', 'milk'},
    {'bread', 'butter'},
    {'butter', 'milk'},
    {'bread', 'milk', 'butter', 'eggs'}
]

# Convert to frozensets for processing
transactions = [frozenset(t) for t in transactions]

# Find frequent itemsets
frequent = get_frequent_itemsets(transactions, min_support=0.4)
print("Frequent Itemsets:", frequent)

# Generate rules
rules = generate_rules(frequent, transactions, min_confidence=0.6)
for rule in rules:
    print(f"{rule['antecedent']} -> {rule['consequent']}: "
          f"sup={rule['support']}, conf={rule['confidence']}, lift={rule['lift']}")
```

---

## Question 3
- [ ] Done

**Create a Python function to normalize a vector using Min-Max normalization.**

**Answer:**

**Problem:** Scale values to range [0, 1] using Min-Max normalization.

**Formula:** $x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$

**Pipeline:**
1. Find min and max of vector
2. Apply formula to each element
3. Handle edge case (all same values)

```python
def min_max_normalize(vector):
    """
    Normalize vector to [0, 1] range using Min-Max normalization.
    
    Args:
        vector: List or array of numeric values
    
    Returns:
        List of normalized values
    """
    min_val = min(vector)
    max_val = max(vector)
    
    # Handle edge case: all values are same
    if max_val == min_val:
        return [0.0] * len(vector)
    
    # Apply Min-Max formula
    normalized = [(x - min_val) / (max_val - min_val) for x in vector]
    
    return normalized


# Example usage
data = [10, 20, 30, 40, 50]
normalized = min_max_normalize(data)

print("Original:", data)
print("Normalized:", normalized)
# Output: [0.0, 0.25, 0.5, 0.75, 1.0]

# Verify
print(f"Min: {min(normalized)}, Max: {max(normalized)}")
# Output: Min: 0.0, Max: 1.0
```

**NumPy Version (More Efficient):**
```python
import numpy as np

def min_max_normalize_numpy(arr):
    """Min-Max normalization using NumPy."""
    arr = np.array(arr)
    min_val, max_val = arr.min(), arr.max()
    
    if max_val == min_val:
        return np.zeros_like(arr, dtype=float)
    
    return (arr - min_val) / (max_val - min_val)

# Example
data = np.array([10, 20, 30, 40, 50])
print(min_max_normalize_numpy(data))
```

**Custom Range [a, b]:**
```python
def min_max_scale(vector, a=0, b=1):
    """Scale to custom range [a, b]."""
    min_val, max_val = min(vector), max(vector)
    if max_val == min_val:
        return [(a + b) / 2] * len(vector)
    return [a + (x - min_val) * (b - a) / (max_val - min_val) for x in vector]
```

---

## Question 4
- [ ] Done

**Use scikit-learn to perform k-means clustering on a sample multi-dimensional dataset.**

**Answer:**

**Problem:** Cluster multi-dimensional data into k groups.

**Pipeline:**
1. Create/load dataset
2. Preprocess (scale features)
3. Apply K-Means
4. Evaluate and visualize

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Step 1: Create sample dataset
X, y_true = make_blobs(n_samples=300, centers=4, 
                        n_features=2, random_state=42)

# Step 2: Scale features (important for distance-based algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Step 4: Get results
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_

print(f"Cluster Labels: {np.unique(labels)}")
print(f"Inertia (WCSS): {inertia:.2f}")
print(f"Centroids:\n{centroids}")

# Step 5: Visualize (for 2D data)
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], 
            c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
```

**Finding Optimal k (Elbow Method):**
```python
# Elbow method to find optimal k
inertias = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

**Silhouette Score Evaluation:**
```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {score:.3f}")
# Score ranges from -1 to 1; higher is better
```

---

## Question 5
- [ ] Done

**Write a Python script to preprocess text data, including tokenization and stemming.**

**Answer:**

**Problem:** Clean and normalize text for mining.

**Pipeline:**
1. Convert to lowercase
2. Remove punctuation/special characters
3. Tokenize (split into words)
4. Remove stopwords
5. Apply stemming/lemmatization

```python
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data (run once)
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocess text: lowercase, remove punctuation, 
    tokenize, remove stopwords, and stem.
    """
    # Step 1: Lowercase
    text = text.lower()
    
    # Step 2: Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Step 3: Tokenize
    tokens = word_tokenize(text)
    
    # Step 4: Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # Step 5: Stemming
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(t) for t in tokens]
    
    return stemmed

# Example usage
sample_text = """Data mining is the process of discovering patterns 
                 in large datasets. It involves methods at the 
                 intersection of machine learning and statistics."""

processed = preprocess_text(sample_text)
print("Original:", sample_text)
print("Processed:", processed)
```

**Alternative with Lemmatization:**
```python
def preprocess_with_lemma(text):
    """Preprocess with lemmatization instead of stemming."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # Lemmatization (more accurate than stemming)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    
    return lemmatized

# Compare outputs
print("Stemmed:", preprocess_text("running studies"))
# Output: ['run', 'studi']

print("Lemmatized:", preprocess_with_lemma("running studies"))
# Output: ['running', 'study']
```

**Simple Version (No NLTK):**
```python
def simple_preprocess(text):
    """Basic preprocessing without external libraries."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stopwords = {'the', 'is', 'in', 'of', 'and', 'a', 'to', 'it', 'at'}
    tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
    return tokens
```

---

## Question 6
- [ ] Done

**Implement a decision tree classifier from scratch in Python.**

**Answer:**

**Problem:** Build a decision tree for classification using Gini impurity.

**Pipeline:**
1. Calculate Gini impurity
2. Find best split (feature + threshold)
3. Recursively build tree
4. Predict by traversing tree

```python
import numpy as np
from collections import Counter

class DecisionTreeNode:
    """Node in decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, 
                 right=None, value=None):
        self.feature = feature      # Feature index to split on
        self.threshold = threshold  # Threshold value
        self.left = left            # Left subtree
        self.right = right          # Right subtree
        self.value = value          # Leaf value (class label)

class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def gini(self, y):
        """Calculate Gini impurity."""
        counts = Counter(y)
        n = len(y)
        return 1 - sum((c/n)**2 for c in counts.values())
    
    def find_best_split(self, X, y):
        """Find best feature and threshold to split on."""
        best_gain = -1
        best_feature, best_threshold = None, None
        
        n_samples, n_features = X.shape
        parent_gini = self.gini(y)
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                # Split data
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue
                
                # Calculate weighted Gini
                left_gini = self.gini(y[left_mask])
                right_gini = self.gini(y[right_mask])
                
                n_left, n_right = sum(left_mask), sum(right_mask)
                weighted_gini = (n_left * left_gini + n_right * right_gini) / n_samples
                
                gain = parent_gini - weighted_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        """Recursively build the tree."""
        n_samples = len(y)
        n_classes = len(set(y))
        
        # Stopping conditions
        if (depth >= self.max_depth or 
            n_classes == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)
        
        # Find best split
        feature, threshold = self.find_best_split(X, y)
        
        if feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)
        
        # Split and recurse
        left_mask = X[:, feature] <= threshold
        left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self.build_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return DecisionTreeNode(feature, threshold, left, right)
    
    def fit(self, X, y):
        """Train the tree."""
        self.root = self.build_tree(np.array(X), np.array(y))
        return self
    
    def predict_one(self, x, node):
        """Predict single sample."""
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        return self.predict_one(x, node.right)
    
    def predict(self, X):
        """Predict multiple samples."""
        return [self.predict_one(x, self.root) for x in np.array(X)]

# Example usage
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
accuracy = sum(p == t for p, t in zip(predictions, y_test)) / len(y_test)
print(f"Accuracy: {accuracy:.2f}")
```

---

## Question 7
- [ ] Done

**Write a Python function that calculates the Gini index for a given data split in a decision tree.**

**Answer:**

**Problem:** Calculate Gini impurity for evaluating decision tree splits.

**Formula:** $Gini = 1 - \sum_{i=1}^{c} p_i^2$

Where $p_i$ is the proportion of class $i$ in the node.

**Pipeline:**
1. Calculate Gini for each child node
2. Calculate weighted average based on sample sizes
3. Return Gini index for the split

```python
from collections import Counter

def gini_impurity(y):
    """
    Calculate Gini impurity for a single node.
    
    Args:
        y: List of class labels
    
    Returns:
        Gini impurity value (0 = pure, 0.5 = max impurity for binary)
    """
    if len(y) == 0:
        return 0
    
    counts = Counter(y)
    n = len(y)
    
    # Gini = 1 - sum(p_i^2)
    gini = 1 - sum((count / n) ** 2 for count in counts.values())
    
    return gini


def gini_split(y_left, y_right):
    """
    Calculate weighted Gini index for a split.
    
    Args:
        y_left: Labels in left child
        y_right: Labels in right child
    
    Returns:
        Weighted Gini index for the split
    """
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right
    
    if n_total == 0:
        return 0
    
    # Weighted average of child Gini impurities
    gini_left = gini_impurity(y_left)
    gini_right = gini_impurity(y_right)
    
    weighted_gini = (n_left / n_total) * gini_left + \
                    (n_right / n_total) * gini_right
    
    return weighted_gini


def information_gain(y_parent, y_left, y_right):
    """Calculate information gain from a split."""
    parent_gini = gini_impurity(y_parent)
    split_gini = gini_split(y_left, y_right)
    return parent_gini - split_gini


# Example usage
# Pure node (all same class)
y_pure = [1, 1, 1, 1]
print(f"Pure node Gini: {gini_impurity(y_pure)}")
# Output: 0.0

# Impure node (mixed classes)
y_mixed = [1, 1, 0, 0]
print(f"Mixed node Gini: {gini_impurity(y_mixed)}")
# Output: 0.5

# Evaluate a split
y_parent = [1, 1, 1, 0, 0, 0]
y_left = [1, 1, 1]    # Pure left
y_right = [0, 0, 0]   # Pure right

print(f"Parent Gini: {gini_impurity(y_parent):.3f}")
print(f"Split Gini: {gini_split(y_left, y_right):.3f}")
print(f"Information Gain: {information_gain(y_parent, y_left, y_right):.3f}")
# Output: Parent: 0.5, Split: 0.0, Gain: 0.5 (perfect split!)
```

**Multi-class Example:**
```python
# 3 classes
y_multiclass = ['A', 'A', 'B', 'B', 'C', 'C']
print(f"3-class Gini: {gini_impurity(y_multiclass):.3f}")
# Output: 0.667 (max impurity for 3 equal classes)
```

---

## Question 8
- [ ] Done

**Use Pandas and NumPy to process and clean a dataset, handling missing values and outliers.**

**Answer:**

**Problem:** Clean a dataset by handling missing values and outliers.

**Pipeline:**
1. Load and inspect data
2. Handle missing values (imputation/deletion)
3. Detect and handle outliers (IQR method)
4. Verify cleaned data

```python
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Clean dataset: handle missing values and outliers.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    print(f"Original shape: {df.shape}")
    
    # Step 1: Inspect missing values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    # Step 2: Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                # Numeric: fill with median
                df[col].fillna(df[col].median(), inplace=True)
            else:
                # Categorical: fill with mode
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    print(f"\nMissing after imputation: {df.isnull().sum().sum()}")
    
    # Step 3: Handle outliers using IQR method
    print("\n--- Outlier Detection (IQR) ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        
        if outliers > 0:
            print(f"{col}: {outliers} outliers")
            # Cap outliers (Winsorization)
            df[col] = df[col].clip(lower, upper)
    
    print(f"\nFinal shape: {df.shape}")
    return df


# Example usage with sample data
np.random.seed(42)

# Create sample dataset with issues
data = {
    'age': [25, 30, np.nan, 35, 40, 150, 28, 33, np.nan, 31],
    'salary': [50000, 60000, 55000, np.nan, 70000, 65000, 52000, 58000, 200000, 62000],
    'department': ['IT', 'HR', 'IT', np.nan, 'Finance', 'HR', 'IT', 'Finance', 'HR', 'IT']
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)

# Clean the dataset
df_cleaned = clean_dataset(df)
print("\nCleaned Data:")
print(df_cleaned)
```

**Additional Cleaning Functions:**
```python
def remove_duplicates(df):
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicates")
    return df

def standardize_column(df, col):
    """Z-score standardization."""
    df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def detect_outliers_zscore(df, col, threshold=3):
    """Detect outliers using Z-score."""
    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
    return df[z_scores > threshold]
```

**Complete Preprocessing Pipeline:**
```python
def preprocess_pipeline(df):
    """Full preprocessing pipeline."""
    df = df.copy()
    df = remove_duplicates(df)
    df = clean_dataset(df)
    
    # Encode categoricals
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols)
    
    return df
```

---

