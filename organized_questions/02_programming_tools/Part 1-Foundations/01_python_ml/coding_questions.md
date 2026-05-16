# Python ML Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Give an example of how to implement a gradient descent algorithm in Python.**

### Definition
Gradient descent is an optimization algorithm that iteratively updates parameters by moving in the direction of steepest descent (negative gradient) to minimize a loss function.

### Algorithm Steps
1. Initialize parameters (weights) randomly
2. Compute predictions using current weights
3. Calculate loss (error)
4. Compute gradient of loss with respect to weights
5. Update weights: `w = w - learning_rate * gradient`
6. Repeat until convergence

### Mathematical Formulation
For linear regression with MSE loss:
- Prediction: $\hat{y} = X \cdot w$
- Loss: $L = \frac{1}{n} \sum (y - \hat{y})^2$
- Gradient: $\nabla L = -\frac{2}{n} X^T (y - \hat{y})$
- Update: $w = w - \alpha \cdot \nabla L$

### Python Implementation

```python
import numpy as np

def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    """
    Simple gradient descent for linear regression.
    
    Pipeline:
    1. Add bias term to X
    2. Initialize weights to zeros
    3. Loop: predict → compute gradient → update weights
    4. Return final weights
    """
    # Step 1: Add bias column (column of 1s)
    n_samples = X.shape[0]
    X_b = np.c_[np.ones((n_samples, 1)), X]  # Shape: (n_samples, n_features+1)
    
    # Step 2: Initialize weights to zeros
    n_features = X_b.shape[1]
    weights = np.zeros(n_features)
    
    # Step 3: Gradient descent loop
    for epoch in range(epochs):
        # Predict
        y_pred = X_b @ weights  # Matrix multiplication
        
        # Compute error
        error = y_pred - y
        
        # Compute gradient: (2/n) * X^T * error
        gradient = (2 / n_samples) * (X_b.T @ error)
        
        # Update weights
        weights = weights - learning_rate * gradient
        
        # Optional: Print loss every 100 epochs
        if epoch % 100 == 0:
            loss = np.mean(error ** 2)
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    return weights


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)  # 100 samples, 1 feature
    y = 4 + 3 * X.flatten() + np.random.randn(100) * 0.5  # y = 4 + 3x + noise
    
    # Run gradient descent
    weights = gradient_descent(X, y, learning_rate=0.1, epochs=1000)
    
    print(f"\nLearned weights: bias = {weights[0]:.4f}, slope = {weights[1]:.4f}")
    print("Expected: bias ≈ 4, slope ≈ 3")
```

### Output
```
Epoch 0: Loss = 24.1234
Epoch 100: Loss = 0.2567
...
Epoch 900: Loss = 0.2341

Learned weights: bias = 3.9876, slope = 3.0123
Expected: bias ≈ 4, slope ≈ 3
```

### Interview Tips
- Know the difference: Batch GD (all data), Stochastic GD (1 sample), Mini-batch GD (subset)
- Learning rate too high → divergence; too low → slow convergence
- Gradient descent finds local minimum (for convex loss, it's global)

---

## Question 2
- [ ] Done

**Implement K-Means clustering from scratch in Python.**

### Definition
K-Means is an unsupervised learning algorithm that partitions data into K clusters by minimizing within-cluster variance. Each cluster is represented by its centroid (mean).

### Algorithm Steps
1. Initialize K cluster centroids randomly (or using K-Means++)
2. **Assignment Step**: Assign each point to nearest centroid using Euclidean distance
3. **Update Step**: Recompute centroids as mean of all points in each cluster
4. Repeat steps 2-3 until convergence (centroids don't change or max iterations reached)
5. Return final cluster assignments and centroids

### Mathematical Formulation

**Objective Function** (minimize within-cluster sum of squares):
$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$

Where:
- $K$ = number of clusters
- $C_k$ = set of points in cluster $k$
- $\mu_k$ = centroid of cluster $k$
- $||x_i - \mu_k||^2$ = squared Euclidean distance

**Assignment Step**:
$$c_i = \arg\min_k ||x_i - \mu_k||^2$$

Assign point $x_i$ to cluster with nearest centroid.

**Update Step**:
$$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$

Update centroid as mean of all points in cluster.

### Python Implementation

```python
import numpy as np

def kmeans(X, k, max_iters=100):
    """
    K-Means clustering implementation.
    
    Pipeline:
    1. Randomly initialize centroids
    2. Loop: assign points → update centroids
    3. Return cluster labels and centroids
    """
    n_samples, n_features = X.shape
    
    # Step 1: Randomly pick k data points as initial centroids
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]
    
    for iteration in range(max_iters):
        # Step 2: Assign each point to nearest centroid
        # Calculate distances from each point to each centroid
        distances = np.zeros((n_samples, k))
        for i in range(k):
            distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)
        
        # Get cluster assignment (index of minimum distance)
        labels = np.argmin(distances, axis=1)
        
        # Step 3: Update centroids
        new_centroids = np.zeros((k, n_features))
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                new_centroids[i] = centroids[i]  # Keep old if empty
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            print(f"Converged at iteration {iteration}")
            break
        
        centroids = new_centroids
    
    return labels, centroids


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate sample data (3 clusters)
    cluster1 = np.random.randn(50, 2) + [0, 0]
    cluster2 = np.random.randn(50, 2) + [5, 5]
    cluster3 = np.random.randn(50, 2) + [10, 0]
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # Run K-Means
    labels, centroids = kmeans(X, k=3)
    
    print(f"Centroids:\n{centroids}")
    print(f"Cluster sizes: {[np.sum(labels == i) for i in range(3)]}")
```

### Interview Tips
- K-Means is sensitive to initial centroid placement → use K-Means++ for better initialization
- Assumes spherical clusters with similar sizes
- Choose K using elbow method or silhouette score
- Computational complexity: O(n × K × i × d) where n=samples, K=clusters, i=iterations, d=dimensions
- Not guaranteed to find global optimum (local minimum)

---

## Question 3
- [ ] Done

**Implement train-test split function from scratch.**

### Definition
Train-test split is a data partitioning technique that divides a dataset into two subsets: training set (for model learning) and test set (for model evaluation). This prevents overfitting and provides unbiased performance estimates.

### Algorithm Steps
1. Set random seed for reproducibility (optional)
2. Generate array of shuffled indices from 0 to n-1
3. Calculate split point based on test_size ratio
4. Split indices into train and test portions
5. Use indices to partition X and y arrays
6. Return X_train, X_test, y_train, y_test

### Mathematical Formulation

Given dataset $(X, y)$ with $n$ samples:
- Test set size: $n_{test} = \lfloor n \times \text{test\_size} \rfloor$
- Train set size: $n_{train} = n - n_{test}$

**Random permutation**: $\pi: \{0, 1, ..., n-1\} \rightarrow \{0, 1, ..., n-1\}$

**Split**:
- Test indices: $I_{test} = \{\pi(0), \pi(1), ..., \pi(n_{test}-1)\}$
- Train indices: $I_{train} = \{\pi(n_{test}), ..., \pi(n-1)\}$

### Python Implementation

```python
import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split data into training and test sets.
    
    Pipeline:
    1. Set random seed for reproducibility
    2. Generate shuffled indices
    3. Calculate split point
    4. Split arrays using indices
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Get number of samples
    n_samples = len(X)
    
    # Create shuffled indices
    indices = np.random.permutation(n_samples)
    
    # Calculate split point
    test_count = int(n_samples * test_size)
    
    # Split indices
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    # Return split arrays
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


# Example usage
X = np.arange(100).reshape(50, 2)
y = np.arange(50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
```

### Interview Tips
- Common split ratios: 80-20, 70-30, 90-10 (depends on dataset size)
- For small datasets, use K-fold cross-validation instead
- Always set random_state for reproducible results
- Stratified split preserves class distribution (important for imbalanced data)
- Temporal data requires time-based split (not random)

---

## Question 4
- [ ] Done

**Implement standardization (Z-score normalization) from scratch.**

### Definition
Standardization (Z-score normalization) transforms features to have zero mean and unit variance. This is crucial for algorithms that are sensitive to feature scales (e.g., gradient descent, SVM, KNN).

### Algorithm Steps

**Training Phase (fit)**:
1. Calculate mean $\mu$ for each feature across all training samples
2. Calculate standard deviation $\sigma$ for each feature
3. Store $\mu$ and $\sigma$ for later use
4. Handle zero std (avoid division by zero): set $\sigma = 1$ if $\sigma = 0$

**Transformation Phase (transform)**:
1. Subtract mean: $x' = x - \mu$
2. Divide by standard deviation: $z = \frac{x'}{\sigma}$
3. Return standardized features

### Mathematical Formulation

**Z-score transformation**:
$$z = \frac{x - \mu}{\sigma}$$

Where:
- $x$ = original feature value
- $\mu = \frac{1}{n} \sum_{i=1}^{n} x_i$ = mean
- $\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2}$ = standard deviation
- $z$ = standardized value

**Properties after standardization**:
- Mean: $E[z] = 0$
- Standard deviation: $\text{std}(z) = 1$
- Distribution shape preserved (not normalized to 0-1 range)

### Python Implementation

```python
import numpy as np

class StandardScaler:
    """
    Standardizes features by removing mean and scaling to unit variance.
    
    Pipeline:
    fit() → Learn mean and std from training data
    transform() → Apply transformation
    fit_transform() → Do both
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X):
        """Learn mean and std from data."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero
        self.std[self.std == 0] = 1
        return self
    
    def transform(self, X):
        """Apply standardization."""
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


# Example usage
X = np.array([[1, 100], [2, 200], [3, 300], [4, 400], [5, 500]])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Original data:")
print(X)
print("\nScaled data (mean=0, std=1):")
print(X_scaled)
print(f"\nMean of scaled: {X_scaled.mean(axis=0)}")
print(f"Std of scaled: {X_scaled.std(axis=0)}")
```

### Interview Tips
- **Standardization vs Normalization**: Standardization → mean=0, std=1; Normalization → scale to [0,1]
- Always fit on training data only, then transform both train and test
- Required for: SVM, KNN, PCA, gradient descent, neural networks
- Not required for: tree-based models (decision trees, random forest, XGBoost)
- Sensitive to outliers (consider robust scaling for outlier-heavy data)

---

## Question 5
- [ ] Done

**Implement a simple decision tree classifier (for binary classification).**

### Definition
A decision tree is a supervised learning algorithm that recursively splits data based on feature values to create a tree structure. Each internal node represents a decision rule, and leaf nodes represent class predictions.

### Algorithm Steps

**Training (Recursive Tree Building)**:
1. **Base cases** (stop splitting):
   - Maximum depth reached
   - Node is pure (all samples have same label)
   - Too few samples to split
2. **Find best split**:
   - For each feature and threshold
   - Calculate information gain (or Gini reduction)
   - Select split with maximum gain
3. **Partition data** into left (≤ threshold) and right (> threshold) subsets
4. **Recursively build** left and right subtrees
5. **Return tree structure** (dict with feature, threshold, children)

**Prediction (Tree Traversal)**:
1. Start at root node
2. Compare sample's feature value with node's threshold
3. Go left if ≤ threshold, right otherwise
4. Repeat until reaching leaf node
5. Return leaf's prediction

### Mathematical Formulation

**Gini Impurity** (measures node impurity):
$$\text{Gini}(p) = 2p(1-p) = 1 - p^2 - (1-p)^2$$

Where $p$ = proportion of positive class samples.

**For multi-class** (C classes):
$$\text{Gini} = 1 - \sum_{i=1}^{C} p_i^2$$

**Information Gain** (reduction in impurity after split):
$$\text{Gain} = \text{Gini}_{\text{parent}} - \left(\frac{n_L}{n}\text{Gini}_L + \frac{n_R}{n}\text{Gini}_R\right)$$

Where:
- $n$ = total samples
- $n_L, n_R$ = samples in left/right child
- $\text{Gini}_L, \text{Gini}_R$ = impurity of children

**Alternative: Entropy** (Shannon entropy):
$$H(p) = -p\log_2(p) - (1-p)\log_2(1-p)$$

### Python Implementation

```python
import numpy as np

class SimpleDecisionTree:
    """
    A simple binary decision tree classifier.
    
    Pipeline:
    1. Find best split (feature + threshold) using Gini impurity
    2. Recursively build left and right subtrees
    3. Predict by traversing tree
    """
    
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None
    
    def gini(self, y):
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0
        p = np.mean(y)
        return 2 * p * (1 - p)
    
    def find_best_split(self, X, y):
        """Find best feature and threshold to split on."""
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        parent_gini = self.gini(y)
        n_samples = len(y)
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                
                # Calculate weighted Gini after split
                left_gini = self.gini(y[left_mask])
                right_gini = self.gini(y[right_mask])
                
                weighted_gini = (left_mask.sum() / n_samples * left_gini +
                               right_mask.sum() / n_samples * right_gini)
                
                gain = parent_gini - weighted_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        """Recursively build the tree."""
        # Stopping conditions
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < 2:
            return {"leaf": True, "prediction": int(np.round(np.mean(y)))}
        
        feature, threshold = self.find_best_split(X, y)
        
        if feature is None:
            return {"leaf": True, "prediction": int(np.round(np.mean(y)))}
        
        left_mask = X[:, feature] <= threshold
        
        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self.build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self.build_tree(X[~left_mask], y[~left_mask], depth + 1)
        }
    
    def fit(self, X, y):
        """Train the decision tree."""
        self.tree = self.build_tree(X, y)
        return self
    
    def predict_one(self, x, node):
        """Predict for a single sample."""
        if node["leaf"]:
            return node["prediction"]
        
        if x[node["feature"]] <= node["threshold"]:
            return self.predict_one(x, node["left"])
        else:
            return self.predict_one(x, node["right"])
    
    def predict(self, X):
        """Predict for all samples."""
        return np.array([self.predict_one(x, self.tree) for x in X])


# Example usage
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=4, random_state=42)
X_train, X_test = X[:160], X[160:]
y_train, y_test = y[:160], y[160:]

tree = SimpleDecisionTree(max_depth=3)
tree.fit(X_train, y_train)

predictions = tree.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2f}")
```

### Interview Tips
- **Advantages**: Interpretable, handles non-linear relationships, no feature scaling needed
- **Disadvantages**: Prone to overfitting, unstable (small data changes → different tree)
- **Hyperparameters**: max_depth, min_samples_split, min_samples_leaf, max_features
- **Splitting criteria**: Gini (faster), Entropy (more balanced), Classification Error
- **Pruning**: Pre-pruning (early stopping) vs Post-pruning (build full tree, then prune)
- **Extensions**: Random Forest (ensemble of trees), Gradient Boosting

---

## Question 6
- [x] Done

**Implement One-Hot Encoding from scratch.**

### Definition
One-hot encoding converts categorical variables into binary vector representation. Each category becomes a separate binary feature, with exactly one '1' and rest '0's. Essential for feeding categorical data into ML algorithms.

### Algorithm Steps
1. **Extract unique categories** from input array
2. **Create mapping**: category → index (0 to n_categories-1)
3. **Initialize binary matrix**: shape (n_samples, n_categories) with zeros
4. **Set bits**: For each sample, set corresponding category index to 1
5. **Return**: encoded matrix and category labels

### Mathematical Formulation

Given categorical variable $x$ with $C$ unique categories $\{c_1, c_2, ..., c_C\}$:

**Encoding function**:
$$\text{OneHot}(x_i) = [e_1, e_2, ..., e_C]$$

Where:
$$e_j = \begin{cases} 
1 & \text{if } x_i = c_j \\
0 & \text{otherwise}
\end{cases}$$

**Properties**:
- $\sum_{j=1}^{C} e_j = 1$ (exactly one '1' per sample)
- Dimension: $n \times C$ matrix for $n$ samples
- Sparse representation (mostly zeros)

**Example**:
Colors = {red, blue, green} → Red = [1, 0, 0], Blue = [0, 1, 0], Green = [0, 0, 1]

### Python Implementation

```python
import numpy as np

def one_hot_encode(data):
    """
    Convert categorical array to one-hot encoded matrix.
    
    Pipeline:
    1. Find unique categories
    2. Create mapping: category → index
    3. Create binary matrix
    """
    # Get unique categories
    categories = np.unique(data)
    n_samples = len(data)
    n_categories = len(categories)
    
    # Create category to index mapping
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    # Create one-hot matrix
    one_hot = np.zeros((n_samples, n_categories))
    
    for i, value in enumerate(data):
        one_hot[i, cat_to_idx[value]] = 1
    
    return one_hot, categories


# Example usage
colors = np.array(["red", "blue", "green", "red", "green", "blue"])

encoded, categories = one_hot_encode(colors)

print(f"Categories: {categories}")
print(f"Original: {colors}")
print(f"Encoded:\n{encoded}")
```

### Output
```
Categories: ['blue' 'green' 'red']
Original: ['red' 'blue' 'green' 'red' 'green' 'blue']
Encoded:
[[0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]]
```

### Interview Tips
- **When to use**: For nominal categorical variables (no order: colors, countries)
- **Avoid for ordinal**: Use label encoding for ordinal data (small < medium < large)
- **Dummy variable trap**: Drop one column to avoid multicollinearity (n-1 encoding)
- **High cardinality problem**: Many categories → too many features → use target encoding or embeddings
- **Alternatives**: Label encoding, target encoding, frequency encoding, embeddings (for neural networks)
- **Libraries**: `pd.get_dummies()`, `sklearn.preprocessing.OneHotEncoder`

---

## Question 7
- [ ] Done

**Implement accuracy, precision, recall, and F1-score from scratch.**

### Definition
Classification metrics evaluate model performance for binary/multi-class classification tasks. Each metric focuses on different aspects:
- **Accuracy**: Overall correctness
- **Precision**: Correctness of positive predictions
- **Recall**: Coverage of actual positives
- **F1-score**: Harmonic mean balancing precision and recall

### Algorithm Steps

**Step 1: Calculate Confusion Matrix Components**
- True Positives (TP): Correctly predicted positive
- True Negatives (TN): Correctly predicted negative
- False Positives (FP): Incorrectly predicted positive (Type I error)
- False Negatives (FN): Incorrectly predicted negative (Type II error)

**Step 2: Calculate Metrics**
1. **Accuracy**: $\frac{TP + TN}{\text{Total}}$
2. **Precision**: $\frac{TP}{TP + FP}$ (of predicted positives, how many are correct?)
3. **Recall**: $\frac{TP}{TP + FN}$ (of actual positives, how many did we catch?)
4. **F1**: Harmonic mean of precision and recall

### Mathematical Formulation

**Confusion Matrix**:
```
                Predicted
              Pos    Neg
Actual  Pos | TP  |  FN  |
        Neg | FP  |  TN  |
```

**Metrics**:

1. **Accuracy**:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

2. **Precision** (Positive Predictive Value):
$$\text{Precision} = \frac{TP}{TP + FP}$$

3. **Recall** (Sensitivity, True Positive Rate):
$$\text{Recall} = \frac{TP}{TP + FN}$$

4. **F1-Score** (Harmonic Mean):
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

5. **Specificity** (True Negative Rate):
$$\text{Specificity} = \frac{TN}{TN + FP}$$

6. **F-Beta Score** (Generalized F1):
$$F_\beta = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}}$$

Where $\beta$ controls precision-recall trade-off:
- $\beta < 1$: Favor precision
- $\beta > 1$: Favor recall

### Python Implementation

```python
import numpy as np

def confusion_matrix_values(y_true, y_pred):
    """Calculate TP, TN, FP, FN for binary classification."""
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN


def accuracy(y_true, y_pred):
    """Calculate accuracy."""
    TP, TN, FP, FN = confusion_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)


def precision(y_true, y_pred):
    """Calculate precision."""
    TP, TN, FP, FN = confusion_matrix_values(y_true, y_pred)
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)


def recall(y_true, y_pred):
    """Calculate recall (sensitivity)."""
    TP, TN, FP, FN = confusion_matrix_values(y_true, y_pred)
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)


def f1_score(y_true, y_pred):
    """Calculate F1 score."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0
    return 2 * (p * r) / (p + r)


# Example usage
y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 0, 1, 0])

print(f"Accuracy:  {accuracy(y_true, y_pred):.2f}")
print(f"Precision: {precision(y_true, y_pred):.2f}")
print(f"Recall:    {recall(y_true, y_pred):.2f}")
print(f"F1 Score:  {f1_score(y_true, y_pred):.2f}")
```

### Interview Tips

**When to use each metric**:
- **Accuracy**: Balanced datasets, equal cost for all errors
- **Precision**: When False Positives are costly (spam detection, fraud detection)
- **Recall**: When False Negatives are costly (cancer detection, security threats)
- **F1-Score**: Imbalanced datasets, balance precision and recall

**Common pitfalls**:
- Accuracy paradox: 99% accuracy meaningless if 99% of data is one class
- Precision-Recall tradeoff: Improving one often hurts the other
- Use ROC-AUC for threshold-independent evaluation

**Multi-class extensions**:
- Macro-average: Average metrics across classes (equal weight)
- Micro-average: Aggregate TP/FP/FN, then calculate (weight by class size)
- Weighted-average: Weight by class support


---

## Question 8
- [ ] Done

**Write a Python function that normalizes an array of data to the range [0, 1].**

**Answer:**

### Min-Max Normalization

## Formula: $x_{norm} = \frac{ x - x_{min} }{ x_{max} - x_{min}}$

```python
import numpy as np

def normalize(data):
    """Normalize array to [0, 1] range using min-max scaling."""
    data = np.array(data, dtype=np.float64)
    min_val = np.min(data)
    max_val = np.max(data)
    
    if max_val == min_val:
        return np.zeros_like(data)  # avoid division by zero
    
    return (data - min_val) / (max_val - min_val)

# Example
arr = np.array([10, 20, 30, 40, 50])
print(normalize(arr))  # [0.0, 0.25, 0.5, 0.75, 1.0]
```

### Using Scikit-learn

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_normalized = scaler.fit_transform(X.reshape(-1, 1))

# Inverse transform
X_original = scaler.inverse_transform(X_normalized)
```

### Column-wise Normalization (2D)

```python
def normalize_2d(data):
    """Normalize each column independently."""
    data = np.array(data, dtype=np.float64)
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # avoid division by zero
    return (data - mins) / ranges
```

> **Interview Tip:** Always fit the scaler on **training data only** and use `transform()` on test data to prevent data leakage.

---

## Question 9
- [ ] Done

**Construct a Python class structure for a simple perceptron model.**

**Answer:**

```python
import numpy as np

class Perceptron:
    """Single-layer perceptron for binary classification."""
    
    def __init__(self, learning_rate=0.01, n_epochs=100):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.errors_ = []  # track errors per epoch
    
    def activation(self, x):
        """Step function: returns 1 if x >= 0, else 0."""
        return np.where(x >= 0, 1, 0)
    
    def fit(self, X, y):
        """Train the perceptron."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for epoch in range(self.n_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                # Forward pass
                linear_output = np.dot(xi, self.weights) + self.bias
                y_pred = self.activation(linear_output)
                
                # Update rule
                update = self.lr * (yi - y_pred)
                self.weights += update * xi
                self.bias += update
                errors += int(update != 0.0)
            
            self.errors_.append(errors)
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break
        
        return self
    
    def predict(self, X):
        """Predict class labels."""
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)
    
    def score(self, X, y):
        """Return accuracy."""
        return np.mean(self.predict(X) == y)

# Usage
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=200, n_features=2,
                           n_redundant=0, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Perceptron(learning_rate=0.01, n_epochs=100)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2f}")
```

### Key Concepts

| Component | Role |
|-----------|------|
| **Weights** | Learned importance of each feature |
| **Bias** | Shifts the decision boundary |
| **Step function** | Threshold activation (0 or 1) |
| **Update rule** | `w += lr * (y_true - y_pred) * x` |

> **Interview Tip:** The perceptron **only converges for linearly separable data**. For non-linear problems, use multi-layer perceptrons (MLPs) with non-linear activations.

---

## Question 10
- [ ] Done

**Create a Python script that performs linear regression on a dataset using NumPy.**

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionNumPy:
    """Linear regression using the Normal Equation and Gradient Descent."""
    
    def __init__(self):
        self.weights = None
        self.bias = None
    
    # --- Method 1: Normal Equation (Closed-form) ---
    def fit_normal(self, X, y):
        """Solve: w = (X^T X)^(-1) X^T y"""
        X_b = np.c_[np.ones(X.shape[0]), X]  # add bias column
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        self.bias = theta[0]
        self.weights = theta[1:]
        return self
    
    # --- Method 2: Gradient Descent ---
    def fit_gd(self, X, y, lr=0.01, epochs=1000):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []
        
        for _ in range(epochs):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y
            
            # Gradients
            dw = (1/n_samples) * (X.T @ error)
            db = (1/n_samples) * np.sum(error)
            
            # Update
            self.weights -= lr * dw
            self.bias -= lr * db
            
            # Track MSE
            mse = np.mean(error ** 2)
            self.losses.append(mse)
        
        return self
    
    def predict(self, X):
        return X @ self.weights + self.bias
    
    def r2_score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# --- Demo ---
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.squeeze() + np.random.randn(100) * 0.5

model = LinearRegressionNumPy()
model.fit_normal(X, y)
print(f"Normal Eq  -> w={model.weights[0]:.3f}, b={model.bias:.3f}, R2={model.r2_score(X, y):.4f}")

model2 = LinearRegressionNumPy()
model2.fit_gd(X, y, lr=0.1, epochs=1000)
print(f"Grad Desc  -> w={model2.weights[0]:.3f}, b={model2.bias:.3f}, R2={model2.r2_score(X, y):.4f}")
```

### Comparison

| Method | Pros | Cons |
|--------|------|------|
| **Normal Equation** | Exact solution, no hyperparameters | $O(n^3)$ — slow for large datasets |
| **Gradient Descent** | Scales to large data | Requires tuning lr, epochs |

> **Interview Tip:** Normal equation fails when $X^T X$ is singular — use `np.linalg.pinv` (pseudo-inverse) instead of `np.linalg.inv`.

---

## Question 11
- [ ] Done

**Write a function that optimizes a given cost function using gradient descent.**

**Answer:**

```python
import numpy as np

def gradient_descent(cost_fn, grad_fn, x0, lr=0.01, epochs=1000, tol=1e-8):
    """
    General-purpose gradient descent optimizer.
    
    Args:
        cost_fn: Function to minimize f(x)
        grad_fn: Gradient of cost function df/dx
        x0: Initial guess (numpy array)
        lr: Learning rate
        epochs: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        x_opt: Optimized parameters
        history: List of (cost, x) per iteration
    """
    x = np.array(x0, dtype=np.float64)
    history = []
    
    for i in range(epochs):
        cost = cost_fn(x)
        grad = grad_fn(x)
        history.append((cost, x.copy()))
        
        # Update step
        x = x - lr * grad
        
        # Convergence check
        if np.linalg.norm(grad) < tol:
            print(f"Converged at iteration {i+1}")
            break
    
    return x, history

# --- Example 1: Minimize f(x) = x^2 + 4x + 4 ---
cost_fn = lambda x: x[0]**2 + 4*x[0] + 4
grad_fn = lambda x: np.array([2*x[0] + 4])

x_opt, hist = gradient_descent(cost_fn, grad_fn, x0=[10.0], lr=0.1)
print(f"Minimum at x = {x_opt[0]:.4f}, f(x) = {cost_fn(x_opt):.4f}")
# Output: Minimum at x = -2.0000, f(x) = 0.0000

# --- Example 2: Rosenbrock function (2D) ---
def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    dx0 = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    dx1 = 200*(x[1] - x[0]**2)
    return np.array([dx0, dx1])

x_opt, hist = gradient_descent(rosenbrock, rosenbrock_grad,
                                x0=[0.0, 0.0], lr=0.001, epochs=50000)
print(f"Minimum at x = {x_opt}")
```

### Gradient Descent Variants

| Variant | Update Rule | Use Case |
|---------|-------------|----------|
| **Batch GD** | Full dataset gradient | Small datasets |
| **Stochastic GD** | Single sample gradient | Large datasets, online learning |
| **Mini-batch GD** | Batch of samples | Most common in deep learning |
| **Momentum** | Accumulate velocity | Escape local minima |
| **Adam** | Adaptive lr + momentum | Default for deep learning |

> **Interview Tip:** Mention the importance of **learning rate scheduling** (decay, warm-up) and that **Adam optimizer** is the most widely used in practice because it adapts the learning rate per parameter.

---

## Question 12
- [ ] Done

**Use Pandas to read a CSV file , clean the data, and prepare it for analysis**

**Answer:**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ========== 1. READ DATA ==========
df = pd.read_csv('data.csv')

# Quick exploration
print(df.shape)
print(df.info())
print(df.describe())
print(df.head())

# ========== 2. CLEAN DATA ==========

# 2a. Remove duplicates
print(f"Duplicates: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

# 2b. Handle missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Numeric columns: fill with median
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical columns: fill with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# 2c. Fix data types
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# 2d. Remove outliers (IQR method)
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

for col in num_cols:
    df = remove_outliers(df, col)

# ========== 3. PREPARE FOR ANALYSIS ==========

# 3a. Encode categorical variables
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 3b. Feature scaling
scaler = StandardScaler()
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

# 3c. Split features and target
X = df_encoded.drop(columns=['target'])
y = df_encoded['target']

print(f"\nFinal shape: {X.shape}")
print(f"Features: {list(X.columns)}")
```

### Data Cleaning Checklist

| Step | Method | Purpose |
|------|--------|--------|
| Duplicates | `drop_duplicates()` | Remove redundant rows |
| Missing values | `fillna()`, `dropna()` | Handle NaNs |
| Data types | `astype()`, `to_datetime()` | Ensure correct types |
| Outliers | IQR, Z-score | Remove extreme values |
| Encoding | `get_dummies()`, `LabelEncoder` | Convert categorical to numeric |
| Scaling | `StandardScaler`, `MinMaxScaler` | Normalize feature ranges |

> **Interview Tip:** Always check `df.info()` and `df.describe()` first. Use **pipelines** (`sklearn.pipeline.Pipeline`) to chain preprocessing steps and prevent data leakage.

---

## Question 13
- [ ] Done

**Develop a Python script that automates the process of hyperparameter tuning using grid search.**

**Answer:**

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Grid Search ==========
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,                    # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,               # use all CPU cores
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

# Results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score:   {grid_search.best_score_:.4f}")
print(f"Test Score:      {grid_search.score(X_test, y_test):.4f}")

# ========== Randomized Search (faster) ==========
param_dist = {
    'n_estimators': np.arange(50, 500, 50),
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': np.arange(2, 20),
    'min_samples_leaf': np.arange(1, 10),
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,               # number of random combinations
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
print(f"Best Parameters: {random_search.best_params_}")
```

### Comparison of Tuning Methods

| Method | Pros | Cons |
|--------|------|------|
| **Grid Search** | Exhaustive, finds best in grid | Slow with many params |
| **Random Search** | Faster, covers more space | May miss optimum |
| **Bayesian (Optuna)** | Intelligent search, efficient | Extra dependency |
| **Halving Grid** | Progressive elimination | Requires sklearn 0.24+ |

> **Interview Tip:** Grid search with $k$ parameters of $n$ values each tests $n^k$ combinations. Use **RandomizedSearchCV** first to narrow the range, then **GridSearchCV** to fine-tune.

---

## Question 14
- [ ] Done

**Explain the concept of a neural network , and how you would implement one in Python**

**Answer:**

A **neural network** is a computational model inspired by biological neurons, consisting of interconnected layers that learn to map inputs to outputs through weighted connections.

### Architecture

```
Input Layer    Hidden Layer(s)    Output Layer
  [x1] ----\    [h1] ----\
  [x2] -----\-> [h2] -----\-> [y]
  [x3] ----/    [h3] ----/
```

### From-Scratch Implementation

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        """
        layers: list of layer sizes, e.g., [2, 4, 1]
        """
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.5
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_deriv(self, a):
        return a * (1 - a)
    
    def forward(self, X):
        self.activations = [X]
        for w, b in zip(self.weights, self.biases):
            z = self.activations[-1] @ w + b
            a = self.sigmoid(z)
            self.activations.append(a)
        return self.activations[-1]
    
    def backward(self, y, lr=0.01):
        m = y.shape[0]
        deltas = [None] * len(self.weights)
        
        # Output layer error
        error = self.activations[-1] - y
        deltas[-1] = error * self.sigmoid_deriv(self.activations[-1])
        
        # Hidden layer errors (backpropagate)
        for i in range(len(self.weights) - 2, -1, -1):
            error = deltas[i+1] @ self.weights[i+1].T
            deltas[i] = error * self.sigmoid_deriv(self.activations[i+1])
        
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= lr * (self.activations[i].T @ deltas[i]) / m
            self.biases[i] -= lr * np.mean(deltas[i], axis=0, keepdims=True)
    
    def train(self, X, y, epochs=1000, lr=0.1):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(y, lr)
            if epoch % 200 == 0:
                loss = np.mean((y - output)**2)
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

# XOR Problem
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 4, 1])
nn.train(X, y, epochs=5000, lr=1.0)
print(nn.forward(X).round(2))
```

> **Interview Tip:** Understand **forward propagation** (compute predictions) and **backpropagation** (compute gradients). In practice, use frameworks like PyTorch or TensorFlow instead of from-scratch implementations.

---

## Question 15
- [ ] Done

**Discuss reinforcement learning and its implementation challenges.**

**Answer:**

**Reinforcement Learning (RL)** is a learning paradigm where an **agent** learns to make decisions by interacting with an **environment** to maximize cumulative **reward**.

### Core Components

| Component | Description |
|-----------|------------|
| **Agent** | The learner/decision maker |
| **Environment** | The world the agent interacts with |
| **State (s)** | Current situation of the agent |
| **Action (a)** | What the agent can do |
| **Reward (r)** | Feedback signal from the environment |
| **Policy (π)** | Strategy mapping states to actions |
| **Value function V(s)** | Expected cumulative reward from state s |
| **Q-function Q(s,a)** | Expected reward for taking action a in state s |

### Simple Q-Learning Implementation

```python
import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=1.0):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = lr            # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.q_table.shape[1] - 1)  # explore
        return np.argmax(self.q_table[state])  # exploit
    
    def update(self, state, action, reward, next_state, done):
        best_next = np.max(self.q_table[next_state]) if not done else 0
        target = reward + self.gamma * best_next
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])
        
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

### Implementation Challenges

| Challenge | Description | Solution |
|-----------|-------------|----------|
| **Exploration vs. Exploitation** | Balance trying new actions vs. using known good ones | Epsilon-greedy, UCB, entropy regularization |
| **Sparse rewards** | Reward only at episode end | Reward shaping, curiosity-driven exploration |
| **Credit assignment** | Which actions caused the reward? | Temporal difference learning, eligibility traces |
| **Sample efficiency** | Millions of interactions needed | Experience replay, model-based RL |
| **Stability** | Q-values can diverge | Target networks, gradient clipping |
| **Continuous actions** | Q-table can’t handle continuous spaces | Policy gradient methods (PPO, SAC) |

> **Interview Tip:** Know the difference between **value-based** (DQN), **policy-based** (REINFORCE), and **actor-critic** (A3C, PPO) methods. PPO is the current industry standard for most RL tasks.

---

## Question 16
- [ ] Done

**What is transfer learning , and how can you implement it using Python libraries ?**

**Answer:**

**Transfer learning** reuses a model trained on one task (source) as the starting point for a different task (target), leveraging learned feature representations.

### Why Transfer Learning?

| Benefit | Description |
|---------|------------|
| **Less data needed** | Pre-trained features reduce data requirements |
| **Faster training** | Only fine-tune top layers |
| **Better performance** | Leverages knowledge from large datasets |
| **Practical** | ImageNet models transfer well to most vision tasks |

### Implementation with TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Step 1: Load pre-trained model (without top classification layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Step 2: Freeze base model layers
base_model.trainable = False

# Step 3: Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)  # 10 classes

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train only the new layers
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Step 5: Fine-tune (unfreeze some base layers)
base_model.trainable = True
for layer in base_model.layers[:-20]:  # freeze all except last 20
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # low lr
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_split=0.2)
```

### Implementation with PyTorch

```python
import torch
import torchvision.models as models
import torch.nn as nn

# Load pre-trained model
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10)
)

# Only new layers will be updated
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

### Transfer Learning Strategies

| Strategy | When to Use |
|----------|------------|
| **Feature extraction** (freeze all) | Small dataset, similar domain |
| **Fine-tuning** (unfreeze top layers) | Medium dataset, slightly different domain |
| **Full fine-tuning** (unfreeze all) | Large dataset, different domain |

> **Interview Tip:** Transfer learning works best when source and target tasks share similar low-level features. For NLP, use **Hugging Face Transformers** with pre-trained BERT/GPT models.

---

## Question 17
- [ ] Done

**How do you implement a recommendation system using Python?**

**Answer:**

### Types of Recommendation Systems

| Type | Method | Example |
|------|--------|--------|
| **Collaborative Filtering** | User-user or item-item similarity | "Users like you also liked..." |
| **Content-Based** | Item feature similarity | "Because you watched action movies..." |
| **Hybrid** | Combines both approaches | Netflix, Spotify |
| **Matrix Factorization** | SVD, NMF to find latent factors | Sparse rating matrices |

### Collaborative Filtering with Surprise Library

```python
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import cross_validate
import pandas as pd

# Prepare data
ratings = pd.DataFrame({
    'user_id':  [1,1,1,2,2,3,3,3],
    'item_id':  [1,2,3,1,3,2,3,4],
    'rating':   [5,3,4,4,2,1,5,4]
})

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

# SVD-based collaborative filtering
model = SVD(n_factors=50, n_epochs=20, lr_all=0.005)
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

# Train and predict
trainset = data.build_full_trainset()
model.fit(trainset)
prediction = model.predict(uid=1, iid=4)  # predict user 1's rating for item 4
print(f"Predicted rating: {prediction.est:.2f}")
```

### Content-Based Filtering (Cosine Similarity)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Item descriptions
items = pd.DataFrame({
    'title': ['Action Movie 1', 'Romance Film', 'Action Movie 2', 'Comedy Show'],
    'description': [
        'explosive car chase fighting',
        'love story romantic dinner',
        'martial arts fighting action',
        'funny jokes humor'
    ]
})

# TF-IDF + Cosine Similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(items['description'])
similarity = cosine_similarity(tfidf_matrix)

# Recommend similar items
def recommend(title, n=2):
    idx = items[items['title'] == title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    return [items.iloc[i[0]]['title'] for i in scores]

print(recommend('Action Movie 1'))  # ['Action Movie 2']
```

> **Interview Tip:** Collaborative filtering suffers from the **cold-start problem** (new users/items have no history). Hybrid systems solve this by combining content features with collaborative signals.

---

## Question 18
- [ ] Done

**How would you develop a spam detection system using Python?**

**Answer:**

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import re

# ========== 1. Load & Preprocess ==========
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)       # remove non-alpha
    text = re.sub(r'\s+', ' ', text).strip()    # normalize whitespace
    return text

df['text'] = df['text'].apply(preprocess)

# ========== 2. Split Data ==========
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# ========== 3. Build Pipeline ==========
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                              stop_words='english')),
    ('classifier', MultinomialNB(alpha=0.1))
])

# ========== 4. Train & Evaluate ==========
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
print(confusion_matrix(y_test, y_pred))

# ========== 5. Predict New Messages ==========
new_msgs = [
    "Congratulations! You won a free iPhone. Click here now!",
    "Hey, are we still meeting for lunch tomorrow?"
]
for msg, pred in zip(new_msgs, pipeline.predict(new_msgs)):
    print(f"{'SPAM' if pred else 'HAM'}: {msg}")
```

### Pipeline Overview

```
Raw Text → Preprocessing → TF-IDF Vectorization → Classifier → Spam/Ham
```

### Model Comparison for Spam Detection

| Model | Pros | Accuracy |
|-------|------|----------|
| **Naive Bayes** | Fast, works well with text | ~97% |
| **Logistic Regression** | Interpretable coefficients | ~97% |
| **SVM** | Good with high-dimensional TF-IDF | ~98% |
| **BERT fine-tuned** | State-of-the-art accuracy | ~99% |

> **Interview Tip:** Naive Bayes is the classic baseline for text classification. Mention **TF-IDF** over bag-of-words (it penalizes common words) and **stratified splits** for imbalanced spam/ham ratios.

---

## Question 19
- [ ] Done

**Describe the steps to design a Python system that predicts house prices based on multiple features.**

**Answer:**

### End-to-End Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ========== 1. LOAD & EXPLORE ==========
df = pd.read_csv('house_prices.csv')
print(df.describe())
print(df.corr()['price'].sort_values(ascending=False))

# ========== 2. FEATURE ENGINEERING ==========
df['age'] = 2024 - df['year_built']
df['price_per_sqft'] = df['price'] / df['sqft']
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

# ========== 3. DEFINE FEATURES ==========
numeric_features = ['sqft', 'bedrooms', 'bathrooms', 'age', 'lot_size']
categorical_features = ['neighborhood', 'condition']
target = 'price'

X = df[numeric_features + categorical_features]
y = df[target]

# ========== 4. PREPROCESSING PIPELINE ==========
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42))
])

# ========== 5. TRAIN & EVALUATE ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5,
                            scoring='neg_mean_squared_error')
print(f"CV RMSE: {np.sqrt(-cv_scores.mean()):.2f}")

# Fit and test
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"Test MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Test R2:   {r2_score(y_test, y_pred):.4f}")
```

### Key Design Steps

| Step | Details |
|------|---------|
| **EDA** | Correlations, distributions, scatter plots |
| **Feature Engineering** | Domain knowledge (age, ratios, interactions) |
| **Preprocessing** | Scale numerics, encode categoricals |
| **Model Selection** | Try Linear Reg, RF, XGBoost, compare CV scores |
| **Evaluation** | RMSE, MAE, R² on held-out test set |
| **Deployment** | Save pipeline with `joblib`, serve via API |

> **Interview Tip:** Use **log-transform** on skewed prices (`np.log1p(y)`) to improve linear model performance. Mention **feature importance** from tree models to identify key price drivers.

---

## Question 20
- [ ] Done

**Explain how you would create a sentiment analysis model with Python.**

**Answer:**

### Approach 1: Traditional ML (TF-IDF + Classifier)

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)          # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)       # keep only letters
    return text.strip()

# Load data (e.g., IMDB reviews)
df = pd.read_csv('reviews.csv')
df['clean_text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['sentiment'], test_size=0.2, random_state=42)

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('classifier', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))
```

### Approach 2: Deep Learning with Transformers (State-of-the-art)

```python
from transformers import pipeline as hf_pipeline

# Zero-shot (no training needed)
sentiment = hf_pipeline('sentiment-analysis')
results = sentiment([
    "This movie was absolutely fantastic!",
    "Terrible experience, would not recommend.",
    "It was okay, nothing special."
])
for r in results:
    print(f"{r['label']}: {r['score']:.4f}")
```

### Approach 3: Fine-tuned BERT

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Prepare dataset
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

dataset = Dataset.from_pandas(df[['clean_text', 'label']].rename(columns={'clean_text': 'text'}))
dataset = dataset.map(tokenize, batched=True)

# Train
training_args = TrainingArguments(
    output_dir='./results', num_train_epochs=3,
    per_device_train_batch_size=16, evaluation_strategy='epoch')

trainer = Trainer(model=model, args=training_args,
                  train_dataset=dataset, eval_dataset=dataset)
trainer.train()
```

### Model Comparison

| Method | Accuracy | Speed | Data Needed |
|--------|----------|-------|-------------|
| **TF-IDF + LR** | ~88% | Very fast | Medium |
| **LSTM/GRU** | ~90% | Medium | Medium-Large |
| **BERT fine-tuned** | ~94% | Slow | Small-Medium |
| **GPT zero-shot** | ~90% | Fast (no training) | None |

> **Interview Tip:** Start with TF-IDF + Logistic Regression as a **strong baseline**, then try transformers. Mention **VADER** for rule-based sentiment (no training) and **Hugging Face** as the go-to library.

---

## Question 21
- [ ] Done

**How would you build and deploy a machine-learning model for predicting customer churn?**

**Answer:**

### End-to-End Churn Prediction System

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

# ========== 1. FEATURE ENGINEERING ==========
df = pd.read_csv('telecom_churn.csv')

# Create features
df['avg_monthly_charge'] = df['total_charges'] / (df['tenure'] + 1)
df['contract_value'] = df['monthly_charges'] * df['tenure']
df['has_multiple_services'] = (df[['internet', 'phone', 'tv']].sum(axis=1) > 1).astype(int)

# ========== 2. PREPROCESSING ==========
# Encode categoricals
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('churn', axis=1)
y = df_encoded['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Handle imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# ========== 3. MODEL TRAINING ==========
model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train_res)

# ========== 4. EVALUATION ==========
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances.nlargest(10))

# ========== 5. DEPLOYMENT ==========
joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# FastAPI endpoint
# app.py
from fastapi import FastAPI
app = FastAPI()

@app.post('/predict')
def predict_churn(features: dict):
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    X = scaler.transform([list(features.values())])
    prob = model.predict_proba(X)[0][1]
    return {'churn_probability': round(prob, 4),
            'prediction': 'Churn' if prob > 0.5 else 'Stay'}
```

### Key Churn Features

| Feature | Impact |
|---------|--------|
| Tenure | Lower tenure = higher churn |
| Contract type | Month-to-month = highest churn |
| Monthly charges | Higher charges = higher churn |
| Customer support calls | More complaints = higher churn |

> **Interview Tip:** Churn is typically an **imbalanced problem** (5-20% churn rate). Always use **SMOTE** or **class weights**, evaluate with **ROC-AUC** instead of accuracy, and mention **retention cost analysis** for setting the optimal threshold.

---

## Question 22
- [ ] Done

**Discuss the development of a system to classify images using Python.**

**Answer:**

### Image Classification Pipeline

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ========== 1. DATA LOADING ==========
# Directory structure: train/class1/, train/class2/, etc.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    'data/train', target_size=(224, 224), batch_size=32,
    class_mode='categorical', subset='training')

val_gen = train_datagen.flow_from_directory(
    'data/train', target_size=(224, 224), batch_size=32,
    class_mode='categorical', subset='validation')

# ========== 2. MODEL (Transfer Learning) ==========
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # freeze pre-trained weights

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ========== 3. TRAINING ==========
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

history = model.fit(train_gen, epochs=30, validation_data=val_gen, callbacks=callbacks)

# ========== 4. PREDICTION ==========
import numpy as np
from tensorflow.keras.preprocessing import image

img = image.load_img('test_image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
class_names = list(train_gen.class_indices.keys())
print(f"Predicted: {class_names[np.argmax(prediction)]}")
```

### Architecture Choices

| Model | Parameters | Accuracy (ImageNet) | Speed |
|-------|-----------|---------------------|-------|
| **MobileNetV2** | 3.4M | 71.8% | Very fast |
| **ResNet50** | 25.6M | 76.1% | Medium |
| **EfficientNet-B0** | 5.3M | 77.1% | Fast |
| **VGG16** | 138M | 71.3% | Slow |

> **Interview Tip:** Always use **transfer learning** with pre-trained models for image classification. Mention **data augmentation** to prevent overfitting and **MobileNet** for edge/mobile deployment.

---

## Question 23
- [ ] Done

**Propose a method for detecting fraudulent transactions with Python-based machine learning.**

**Answer:**

### Fraud Detection Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE

# ========== 1. FEATURE ENGINEERING ==========
df = pd.read_csv('transactions.csv')

# Time-based features
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['is_weekend'] = pd.to_datetime(df['timestamp']).dt.dayofweek >= 5

# Aggregation features (per user)
user_stats = df.groupby('user_id')['amount'].agg(['mean', 'std', 'max']).reset_index()
user_stats.columns = ['user_id', 'avg_amount', 'std_amount', 'max_amount']
df = df.merge(user_stats, on='user_id')

# Deviation from normal behavior
df['amount_deviation'] = (df['amount'] - df['avg_amount']) / (df['std_amount'] + 1e-8)
df['is_high_amount'] = (df['amount'] > df['avg_amount'] * 3).astype(int)

# ========== 2. HANDLE EXTREME IMBALANCE ==========
# Fraud is typically 0.1-2% of transactions
print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")

X = df[['amount', 'hour', 'is_weekend', 'amount_deviation', 'is_high_amount']]
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE for training set only
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# ========== 3. SUPERVISED MODEL ==========
xgb_model = XGBClassifier(
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    max_depth=6, learning_rate=0.1, n_estimators=200,
    eval_metric='aucpr', random_state=42)

xgb_model.fit(X_train_scaled, y_train_res)
y_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

# ========== 4. EVALUATION (Precision-Recall, not accuracy!) ==========
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)
print(f"PR-AUC: {pr_auc:.4f}")

# Optimize threshold for business needs
optimal_idx = np.argmax(2 * precision * recall / (precision + recall + 1e-8))
optimal_threshold = thresholds[optimal_idx]
y_pred = (y_proba >= optimal_threshold).astype(int)
print(classification_report(y_test, y_pred))
```

### Approach Comparison

| Method | Type | Pros |
|--------|------|------|
| **XGBoost/LightGBM** | Supervised | Best overall performance |
| **Isolation Forest** | Unsupervised | No labels needed, detects anomalies |
| **Autoencoders** | Deep learning | Learn normal patterns, flag deviations |
| **Rule-based** | Heuristic | Transparent, complementary |

> **Interview Tip:** Never use **accuracy** for fraud detection (99.9% accuracy by predicting all non-fraud). Use **Precision-Recall AUC**, and discuss the **cost of false positives** (blocking legitimate users) vs. **false negatives** (missing fraud).

---

## Question 24
- [ ] Done

**Create a Python generator that yields batches of data from a large dataset.**

**Answer:**

```python
import numpy as np

# ========== Method 1: NumPy Array Batching ==========
def batch_generator(X, y, batch_size=32, shuffle=True):
    """
    Generator that yields batches from arrays.
    
    Args:
        X: Features array (n_samples, n_features)
        y: Labels array (n_samples,)
        batch_size: Number of samples per batch
        shuffle: Randomize order each epoch
    
    Yields:
        (X_batch, y_batch) tuples
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]

# Usage
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

for epoch in range(3):
    for X_batch, y_batch in batch_generator(X, y, batch_size=64):
        # model.train_on_batch(X_batch, y_batch)
        pass

# ========== Method 2: CSV File Batching (Large Files) ==========
import pandas as pd

def csv_batch_generator(filepath, batch_size=1000):
    """Read large CSV files in chunks without loading into memory."""
    for chunk in pd.read_csv(filepath, chunksize=batch_size):
        X = chunk.drop('target', axis=1).values
        y = chunk['target'].values
        yield X, y

# Usage
for X_batch, y_batch in csv_batch_generator('large_data.csv', batch_size=5000):
    model.partial_fit(X_batch, y_batch)

# ========== Method 3: Infinite Generator (for training loops) ==========
def infinite_batch_generator(X, y, batch_size=32):
    """Infinitely yields batches (cycles through data)."""
    n = len(X)
    while True:
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield X[indices[start:end]], y[indices[start:end]]

# Usage with Keras
# model.fit(infinite_batch_generator(X, y, 32), steps_per_epoch=len(X)//32, epochs=10)
```

### Why Use Generators?

| Benefit | Description |
|---------|------------|
| **Memory efficient** | Only one batch in memory at a time |
| **Large datasets** | Process files larger than RAM |
| **Lazy evaluation** | Data loaded on-demand |
| **Streaming** | Works with real-time data |

> **Interview Tip:** Python generators use `yield` (lazy evaluation) vs. `return` (eager). For deep learning, PyTorch's `DataLoader` and TensorFlow's `tf.data.Dataset` are optimized versions of this concept.

---

## Question 25
- [ ] Done

**Implement a convolutional neural network using PyTorch or TensorFlow in Python.**

**Answer:**

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ========== 1. Define CNN Architecture ==========
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ========== 2. Data Loading ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# ========== 3. Training Loop ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Evaluate
    model.eval()
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            correct += (output.argmax(1) == y_batch).sum().item()
    
    accuracy = correct / len(test_dataset)
    print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={accuracy:.4f}")
```

### CNN Layer Explanation

| Layer | Purpose | Output Shape |
|-------|---------|-------------|
| **Conv2d** | Extract spatial features (edges, textures) | (batch, channels, H, W) |
| **ReLU** | Non-linear activation | Same |
| **MaxPool2d** | Downsample, translation invariance | (batch, channels, H/2, W/2) |
| **Flatten** | Reshape for dense layers | (batch, features) |
| **Linear** | Classification decision | (batch, num_classes) |
| **Dropout** | Regularization (prevent overfitting) | Same |

> **Interview Tip:** Know the formula for output size: $O = \frac{I - K + 2P}{S} + 1$ where I=input size, K=kernel size, P=padding, S=stride.

---

## Question 26
- [ ] Done

**Develop a Python function that uses genetic algorithms to optimize a simple problem.**

**Answer:**

```python
import numpy as np
import random

class GeneticAlgorithm:
    """
    Genetic Algorithm to maximize a fitness function.
    Example: Find x that maximizes f(x) = -x^2 + 10x + 5
    """
    
    def __init__(self, pop_size=50, gene_length=10, mutation_rate=0.01, crossover_rate=0.8):
        self.pop_size = pop_size
        self.gene_length = gene_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def initialize_population(self):
        """Random binary strings."""
        return np.random.randint(0, 2, size=(self.pop_size, self.gene_length))
    
    def decode(self, chromosome):
        """Convert binary chromosome to decimal value in [0, 20]."""
        decimal = int(''.join(map(str, chromosome)), 2)
        return decimal * 20 / (2**self.gene_length - 1)
    
    def fitness(self, x):
        """Function to maximize: f(x) = -x^2 + 10x + 5."""
        return -x**2 + 10*x + 5
    
    def selection(self, population, fitness_scores):
        """Tournament selection."""
        selected = []
        for _ in range(self.pop_size):
            i, j = random.sample(range(self.pop_size), 2)
            winner = i if fitness_scores[i] > fitness_scores[j] else j
            selected.append(population[winner].copy())
        return np.array(selected)
    
    def crossover(self, parent1, parent2):
        """Single-point crossover."""
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.gene_length - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, chromosome):
        """Bit-flip mutation."""
        for i in range(self.gene_length):
            if random.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        return chromosome
    
    def evolve(self, generations=100):
        population = self.initialize_population()
        best_solution = None
        best_fitness = float('-inf')
        
        for gen in range(generations):
            # Evaluate fitness
            decoded = [self.decode(chrom) for chrom in population]
            fitness_scores = [self.fitness(x) for x in decoded]
            
            # Track best
            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_solution = decoded[gen_best_idx]
            
            if gen % 20 == 0:
                print(f"Gen {gen}: Best x={best_solution:.4f}, f(x)={best_fitness:.4f}")
            
            # Selection, Crossover, Mutation
            selected = self.selection(population, fitness_scores)
            new_pop = []
            for i in range(0, self.pop_size, 2):
                c1, c2 = self.crossover(selected[i], selected[min(i+1, self.pop_size-1)])
                new_pop.extend([self.mutate(c1), self.mutate(c2)])
            population = np.array(new_pop[:self.pop_size])
        
        print(f"\nOptimal: x={best_solution:.4f}, f(x)={best_fitness:.4f}")
        return best_solution, best_fitness

# Run
ga = GeneticAlgorithm(pop_size=50, gene_length=16, mutation_rate=0.02)
ga.evolve(generations=100)
# Expected: x=5.0, f(x)=30.0
```

### GA Components

| Component | Analogy | Role |
|-----------|---------|------|
| **Population** | Species | Set of candidate solutions |
| **Chromosome** | DNA | Encoded solution |
| **Fitness** | Survival | Quality measure |
| **Selection** | Natural selection | Pick best individuals |
| **Crossover** | Reproduction | Combine parent genes |
| **Mutation** | Random variation | Maintain diversity |

> **Interview Tip:** GAs are useful when the search space is large, non-differentiable, or has many local optima. For differentiable functions, gradient-based methods are more efficient.

---

## Question 27
- [ ] Done

**Code a Python simulation that compares different optimization techniques on a fixed dataset.**

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt

# ========== Target Function: Rosenbrock ==========
def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    dx0 = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    dx1 = 200*(x[1] - x[0]**2)
    return np.array([dx0, dx1])

# ========== 1. Vanilla Gradient Descent ==========
def gradient_descent(x0, lr=0.001, epochs=5000):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for _ in range(epochs):
        x -= lr * rosenbrock_grad(x)
        history.append(x.copy())
    return np.array(history)

# ========== 2. Momentum ==========
def momentum_gd(x0, lr=0.001, beta=0.9, epochs=5000):
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    history = [x.copy()]
    for _ in range(epochs):
        v = beta * v - lr * rosenbrock_grad(x)
        x += v
        history.append(x.copy())
    return np.array(history)

# ========== 3. Adam ==========
def adam(x0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, epochs=5000):
    x = np.array(x0, dtype=float)
    m, v = np.zeros_like(x), np.zeros_like(x)
    history = [x.copy()]
    for t in range(1, epochs + 1):
        g = rosenbrock_grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(x.copy())
    return np.array(history)

# ========== 4. RMSprop ==========
def rmsprop(x0, lr=0.001, beta=0.9, eps=1e-8, epochs=5000):
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    history = [x.copy()]
    for _ in range(epochs):
        g = rosenbrock_grad(x)
        v = beta * v + (1 - beta) * g**2
        x -= lr * g / (np.sqrt(v) + eps)
        history.append(x.copy())
    return np.array(history)

# ========== Run Comparison ==========
x0 = [-1.0, 1.0]
results = {
    'GD':       gradient_descent(x0),
    'Momentum': momentum_gd(x0),
    'Adam':     adam(x0),
    'RMSprop':  rmsprop(x0)
}

# Print final values
for name, hist in results.items():
    final = hist[-1]
    print(f"{name:10s}: x={final}, f(x)={rosenbrock(final):.6f}")

# Plot convergence
plt.figure(figsize=(10, 6))
for name, hist in results.items():
    losses = [rosenbrock(h) for h in hist]
    plt.plot(losses[:500], label=name)
plt.xlabel('Iteration'); plt.ylabel('Loss'); plt.yscale('log')
plt.title('Optimization Comparison on Rosenbrock Function')
plt.legend(); plt.grid(True); plt.show()
```

### Comparison Summary

| Optimizer | Convergence | Hyperparameters | Best For |
|-----------|-------------|-----------------|----------|
| **Vanilla GD** | Slow | lr only | Simple, convex |
| **Momentum** | Faster | lr, beta | Escaping local minima |
| **RMSprop** | Good | lr, beta | Non-stationary problems |
| **Adam** | Best overall | lr, beta1, beta2 | Default choice for DL |

> **Interview Tip:** Adam combines momentum + RMSprop and is the default optimizer for most deep learning tasks. However, **SGD with momentum** can sometimes generalize better with proper tuning.

---

## Question 28
- [ ] Done

**Write a Python script that visualizes decision boundaries for a classification model.**

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def plot_decision_boundary(model, X, y, title='Decision Boundary', ax=None):
    """
    Visualize the decision boundary of a 2D classifier.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create mesh grid
    h = 0.02  # step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and data points
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black', s=30)
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    return ax

# ========== Compare Multiple Models ==========
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'SVM (RBF)': SVC(kernel='rbf', gamma='scale'),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (name, model) in zip(axes.ravel(), models.items()):
    model.fit(X, y)
    plot_decision_boundary(model, X, y, title=name, ax=ax)
    acc = model.score(X, y)
    ax.set_title(f"{name} (acc={acc:.2f})")

plt.tight_layout()
plt.savefig('decision_boundaries.png', dpi=150)
plt.show()
```

### Interpretation Guide

| Boundary Shape | Model Type |
|---------------|------------|
| **Linear** | Logistic Regression, Linear SVM |
| **Smooth curves** | SVM (RBF), Neural Networks |
| **Irregular/jagged** | KNN, Decision Trees (overfitting risk) |
| **Rectangular** | Decision Trees |

> **Interview Tip:** Decision boundary visualization only works for 2D features. For high-dimensional data, use **PCA or t-SNE** to reduce to 2D first, then plot boundaries.

---

## Question 29
- [ ] Done

**Create a Python implementation of the A* search algorithm for pathfinding on a grid.**

**Answer:**

```python
import heapq
import numpy as np

class AStarPathfinder:
    def __init__(self, grid):
        """
        grid: 2D numpy array where 0=walkable, 1=obstacle
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
    
    def heuristic(self, a, b):
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, node):
        """Get valid adjacent cells (4-directional)."""
        r, c = node
        neighbors = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] == 0:
                neighbors.append((nr, nc))
        return neighbors
    
    def find_path(self, start, goal):
        """
        A* search from start to goal.
        Returns: list of (row, col) tuples representing the path, or None if no path.
        """
        # Priority queue: (f_score, node)
        open_set = [(0, start)]
        came_from = {}
        
        g_score = {start: 0}                           # cost from start
        f_score = {start: self.heuristic(start, goal)} # estimated total cost
        visited = set()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1  # cost of 1 per step
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # no path found
    
    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]
    
    def visualize(self, path=None):
        """Print grid with path marked."""
        display = self.grid.astype(str)
        display[display == '0'] = '.'
        display[display == '1'] = '#'
        if path:
            for r, c in path:
                display[r][c] = '*'
            display[path[0][0]][path[0][1]] = 'S'
            display[path[-1][0]][path[-1][1]] = 'G'
        for row in display:
            print(' '.join(row))

# ========== Usage ==========
grid = np.array([
    [0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

pathfinder = AStarPathfinder(grid)
path = pathfinder.find_path((0, 0), (4, 4))

if path:
    print(f"Path found ({len(path)} steps): {path}")
    pathfinder.visualize(path)
else:
    print("No path found!")
```

### A* vs Other Search Algorithms

| Algorithm | Optimal | Complete | Time Complexity |
|-----------|---------|----------|-----------------|
| **A*** | Yes (admissible h) | Yes | $O(b^d)$ |
| **Dijkstra** | Yes | Yes | $O(V \log V)$ |
| **BFS** | Yes (unweighted) | Yes | $O(V + E)$ |
| **DFS** | No | No | $O(V + E)$ |
| **Greedy Best-First** | No | No | $O(b^d)$ |

> **Interview Tip:** A* is optimal when the heuristic is **admissible** (never overestimates). Manhattan distance is admissible for grid movement without diagonals; Euclidean distance for diagonal movement.

---

## Question 30
- [ ] Done

**Implement a simple reinforcement learning agent that learns to play a basic game.**

**Answer:**

```python
import numpy as np
import random

# ========== Simple Grid World Environment ==========
class GridWorld:
    """
    4x4 grid. Agent starts at (0,0), goal at (3,3).
    Actions: 0=Up, 1=Down, 2=Left, 3=Right
    Reward: +10 at goal, -1 per step, -5 for hitting wall
    """
    def __init__(self, size=4):
        self.size = size
        self.state = (0, 0)
        self.goal = (size-1, size-1)
        self.actions = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}
    
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        dr, dc = self.actions[action]
        new_r = self.state[0] + dr
        new_c = self.state[1] + dc
        
        # Check boundaries
        if 0 <= new_r < self.size and 0 <= new_c < self.size:
            self.state = (new_r, new_c)
            if self.state == self.goal:
                return self.state, 10, True   # reached goal
            return self.state, -1, False       # normal step
        else:
            return self.state, -5, False       # hit wall

# ========== Q-Learning Agent ==========
class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=1.0):
        self.q_table = np.zeros((n_states, n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # explore
        return np.argmax(self.q_table[state[0], state[1]])  # exploit
    
    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state[0], state[1], action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state[0], next_state[1]])
        
        self.q_table[state[0], state[1], action] += self.lr * (target - current_q)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ========== Training Loop ==========
env = GridWorld(size=4)
agent = QLearningAgent(n_states=4, n_actions=4)

episodes = 1000
rewards_history = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    
    for step in range(100):  # max steps per episode
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    agent.decay_epsilon()
    rewards_history.append(total_reward)
    
    if (ep + 1) % 200 == 0:
        avg_reward = np.mean(rewards_history[-100:])
        print(f"Episode {ep+1}: Avg Reward={avg_reward:.2f}, Epsilon={agent.epsilon:.3f}")

# ========== Test Learned Policy ==========
state = env.reset()
print("\nLearned path:")
for _ in range(20):
    action = np.argmax(agent.q_table[state[0], state[1]])
    action_name = ['Up', 'Down', 'Left', 'Right'][action]
    next_state, reward, done = env.step(action)
    print(f"  {state} -> {action_name} -> {next_state}")
    state = next_state
    if done:
        print("  Goal reached!")
        break
```

> **Interview Tip:** Q-Learning is **off-policy** (learns optimal policy regardless of behavior policy). For continuous state spaces, use **Deep Q-Networks (DQN)** which replace the Q-table with a neural network.

---

## Question 31
- [ ] Done

**Use a Python library to perform time-series forecasting on stock market data.**

**Answer:**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ========== 1. LOAD & PREPARE DATA ==========
# Using yfinance to download stock data
import yfinance as yf

df = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
df = df[['Close']].copy()
df.columns = ['price']

# ========== 2. FEATURE ENGINEERING ==========
# Create lagged features
for lag in [1, 3, 7, 14, 30]:
    df[f'lag_{lag}'] = df['price'].shift(lag)

# Rolling statistics
df['rolling_mean_7'] = df['price'].rolling(7).mean()
df['rolling_std_7'] = df['price'].rolling(7).std()
df['rolling_mean_30'] = df['price'].rolling(30).mean()

df.dropna(inplace=True)

# ========== 3. TRAIN/TEST SPLIT (time-based) ==========
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

features = [c for c in df.columns if c != 'price']
X_train, y_train = train[features], train['price']
X_test, y_test = test[features], test['price']

# ========== 4. MODEL: XGBoost ==========
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# ========== 5. MODEL: LSTM (Deep Learning) ==========
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Prepare sequences
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['price']].values)

X_seq, y_seq = create_sequences(scaled, seq_length=30)
split = int(len(X_seq) * 0.8)
X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
y_train_seq, y_test_seq = y_seq[:split], y_seq[split:]

lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(30, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=0)
```

### Method Comparison

| Method | Pros | Cons |
|--------|------|------|
| **ARIMA/SARIMA** | Statistical, interpretable | Linear only, manual parameter tuning |
| **Prophet (Facebook)** | Easy to use, handles holidays | Less accurate for volatile data |
| **XGBoost** | Feature engineering flexibility | Requires manual lag features |
| **LSTM** | Captures long-term dependencies | Slow training, needs lots of data |

> **Interview Tip:** Always split time-series data **chronologically** (never random split). Mention **walk-forward validation** instead of k-fold CV, and discuss the **efficient market hypothesis** as a caveat for stock prediction.

---

## Question 32
- [ ] Done

**What is federated learning , and how can Python be used to implement it?**

**Answer:**

**Federated Learning (FL)** is a distributed ML approach where models are trained across multiple devices/servers holding local data, without exchanging raw data.

### How It Works

```
      Central Server
      [Global Model]
       /    |    \
      v     v     v
  Client1  Client2  Client3
  [Local   [Local   [Local
   Data]    Data]    Data]
      \     |     /
       v    v    v
    Aggregate Model Updates
    (FedAvg: average weights)
```

### Simplified Implementation

```python
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import copy

# ========== Simulate Federated Learning ==========

def create_client_data(n_clients=5, n_samples=200):
    """Simulate data distributed across clients."""
    X, y = make_classification(n_samples=n_samples * n_clients,
                               n_features=10, random_state=42)
    client_data = []
    for i in range(n_clients):
        start = i * n_samples
        end = start + n_samples
        client_data.append((X[start:end], y[start:end]))
    return client_data

def federated_averaging(global_weights, client_weights_list):
    """FedAvg: Average model weights from all clients."""
    avg_weights = np.mean(client_weights_list, axis=0)
    return avg_weights

def train_federated(n_clients=5, n_rounds=10):
    """Federated learning training loop."""
    client_data = create_client_data(n_clients)
    
    # Initialize global model
    global_model = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True)
    # Initial fit to set dimensions
    X_init, y_init = client_data[0]
    global_model.fit(X_init[:10], y_init[:10])
    
    for round_num in range(n_rounds):
        client_weights = []
        
        for client_id, (X_client, y_client) in enumerate(client_data):
            # Client receives global model
            local_model = copy.deepcopy(global_model)
            
            # Local training (multiple epochs)
            for _ in range(5):
                local_model.partial_fit(X_client, y_client)
            
            client_weights.append(local_model.coef_.copy())
        
        # Server aggregates weights (FedAvg)
        avg_weights = federated_averaging(global_model.coef_, client_weights)
        global_model.coef_ = avg_weights
        
        # Evaluate
        all_X = np.vstack([d[0] for d in client_data])
        all_y = np.concatenate([d[1] for d in client_data])
        acc = accuracy_score(all_y, global_model.predict(all_X))
        print(f"Round {round_num+1}: Global Accuracy = {acc:.4f}")
    
    return global_model

model = train_federated(n_clients=5, n_rounds=10)
```

### Key Concepts

| Concept | Description |
|---------|------------|
| **FedAvg** | Most common aggregation (average weights) |
| **Data privacy** | Raw data never leaves the device |
| **Non-IID data** | Clients may have different data distributions |
| **Communication** | Only model updates (not data) are sent |
| **Differential Privacy** | Add noise to updates for extra privacy |

### Production Frameworks

| Framework | By | Description |
|-----------|-----|------------|
| **PySyft** | OpenMined | Privacy-preserving FL in PyTorch |
| **TensorFlow Federated** | Google | FL simulation and deployment |
| **Flower** | Adap | Framework-agnostic FL |
| **FATE** | WeBank | Industrial FL platform |

> **Interview Tip:** Federated learning is critical for **healthcare** (patient data stays in hospitals), **mobile keyboards** (Google Gboard), and **financial institutions** (data regulation compliance). Mention challenges: communication overhead, non-IID data, and Byzantine fault tolerance.

## Question 33
- [ ] Done

**Implement the k-means clustering algorithm using only standard Python libraries**

### K-Means Clustering from Scratch

```python
import random
import math

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def kmeans(data, k, max_iters=100, tol=1e-4):
    """
    K-Means clustering algorithm.
    
    Args:
        data: list of data points (each point is a list/tuple of features)
        k: number of clusters
        max_iters: maximum iterations
        tol: convergence tolerance
    
    Returns:
        centroids: final cluster centers
        assignments: cluster label for each data point
    """
    n_features = len(data[0])
    
    # Step 1: Randomly initialize centroids
    centroids = random.sample(data, k)
    
    for iteration in range(max_iters):
        # Step 2: Assign each point to nearest centroid
        assignments = []
        for point in data:
            distances = [euclidean_distance(point, c) for c in centroids]
            assignments.append(distances.index(min(distances)))
        
        # Step 3: Recompute centroids
        new_centroids = []
        for cluster_id in range(k):
            cluster_points = [data[i] for i in range(len(data)) if assignments[i] == cluster_id]
            if cluster_points:
                centroid = [sum(dim) / len(cluster_points) for dim in zip(*cluster_points)]
            else:
                centroid = random.choice(data)  # reinitialize empty cluster
            new_centroids.append(centroid)
        
        # Step 4: Check convergence
        shift = sum(euclidean_distance(old, new) for old, new in zip(centroids, new_centroids))
        centroids = new_centroids
        if shift < tol:
            break
    
    return centroids, assignments

# Usage
data = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]
centroids, labels = kmeans(data, k=2)
print(f"Centroids: {centroids}")
print(f"Labels: {labels}")
```

### Key Steps
| Step | Description |
|------|-------------|
| **Initialize** | Randomly select k data points as initial centroids |
| **Assign** | Assign each point to the nearest centroid |
| **Update** | Recompute centroid as mean of assigned points |
| **Converge** | Repeat until centroids stop moving (or max iterations) |

> **Interview Tip:** Mention K-Means++ initialization (choosing initial centroids that are far apart) to avoid poor convergence. Also discuss the **elbow method** for choosing k by plotting inertia vs. k.

---

## Question 34
- [ ] Done

**Implement a decision tree from scratch in Python**

### Decision Tree Classifier from Scratch

```python
from collections import Counter
import math

def entropy(labels):
    """Calculate entropy of a label list."""
    counts = Counter(labels)
    total = len(labels)
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)

def information_gain(data, labels, feature_idx, threshold):
    """Calculate information gain for a binary split."""
    parent_entropy = entropy(labels)
    
    left_mask = [d[feature_idx] <= threshold for d in data]
    left_labels = [l for l, m in zip(labels, left_mask) if m]
    right_labels = [l for l, m in zip(labels, left_mask) if not m]
    
    if not left_labels or not right_labels:
        return 0
    
    n = len(labels)
    child_entropy = (len(left_labels) / n) * entropy(left_labels) + \
                    (len(right_labels) / n) * entropy(right_labels)
    return parent_entropy - child_entropy

def best_split(data, labels):
    """Find the best feature and threshold to split on."""
    best_gain = 0
    best_feature, best_threshold = None, None
    n_features = len(data[0])
    
    for feature_idx in range(n_features):
        values = sorted(set(d[feature_idx] for d in data))
        thresholds = [(values[i] + values[i+1]) / 2 for i in range(len(values) - 1)]
        
        for threshold in thresholds:
            gain = information_gain(data, labels, feature_idx, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
    
    return best_feature, best_threshold, best_gain

def build_tree(data, labels, depth=0, max_depth=5):
    """Recursively build a decision tree."""
    # Base cases
    if len(set(labels)) == 1:
        return {"leaf": True, "class": labels[0]}
    if depth >= max_depth or len(data) < 2:
        return {"leaf": True, "class": Counter(labels).most_common(1)[0][0]}
    
    feature, threshold, gain = best_split(data, labels)
    if gain == 0:
        return {"leaf": True, "class": Counter(labels).most_common(1)[0][0]}
    
    left_data, left_labels, right_data, right_labels = [], [], [], []
    for d, l in zip(data, labels):
        if d[feature] <= threshold:
            left_data.append(d); left_labels.append(l)
        else:
            right_data.append(d); right_labels.append(l)
    
    return {
        "leaf": False,
        "feature": feature,
        "threshold": threshold,
        "left": build_tree(left_data, left_labels, depth + 1, max_depth),
        "right": build_tree(right_data, right_labels, depth + 1, max_depth),
    }

def predict_one(tree, sample):
    """Predict class for a single sample."""
    if tree["leaf"]:
        return tree["class"]
    if sample[tree["feature"]] <= tree["threshold"]:
        return predict_one(tree["left"], sample)
    return predict_one(tree["right"], sample)

# Usage
data = [[2, 3], [1, 1], [4, 5], [6, 7], [5, 4], [3, 2]]
labels = [0, 0, 1, 1, 1, 0]
tree = build_tree(data, labels, max_depth=3)
print(predict_one(tree, [3, 4]))  # Predicts class
```

### Splitting Criteria
| Criterion | Formula | Use Case |
|-----------|---------|----------|
| **Entropy** | $-\sum p_i \log_2(p_i)$ | ID3, C4.5 |
| **Gini Impurity** | $1 - \sum p_i^2$ | CART (sklearn default) |
| **Information Gain** | Parent entropy − weighted child entropy | Feature selection |

> **Interview Tip:** Discuss **pruning** (pre-pruning via max_depth, min_samples; post-pruning via cost-complexity). Mention that sklearn's `DecisionTreeClassifier` uses optimized C code with Gini impurity by default.

---

## Question 35
- [ ] Done

**Write a Python function to split a dataset into training and testing sets**

### Train-Test Split from Scratch

```python
import random

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split dataset into training and testing sets.
    
    Args:
        X: feature data (list of lists)
        y: labels (list)
        test_size: fraction of data for testing (0 to 1)
        random_state: seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        random.seed(random_state)
    
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    
    split_idx = int(n * (1 - test_size))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test

# Usage
X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
y = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
# Train size: 8, Test size: 2
```

### Stratified Split (preserves class distribution)

```python
def stratified_split(X, y, test_size=0.2, random_state=None):
    """Split while maintaining class proportions."""
    if random_state is not None:
        random.seed(random_state)
    
    # Group indices by class
    class_indices = {}
    for i, label in enumerate(y):
        class_indices.setdefault(label, []).append(i)
    
    train_indices, test_indices = [], []
    for label, indices in class_indices.items():
        random.shuffle(indices)
        split = int(len(indices) * (1 - test_size))
        train_indices.extend(indices[:split])
        test_indices.extend(indices[split:])
    
    random.shuffle(train_indices)
    random.shuffle(test_indices)
    
    return ([X[i] for i in train_indices], [X[i] for i in test_indices],
            [y[i] for i in train_indices], [y[i] for i in test_indices])
```

### Comparison with sklearn
```python
from sklearn.model_selection import train_test_split  # The standard way
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                     stratify=y, random_state=42)
```

> **Interview Tip:** Always mention **stratified splitting** for imbalanced datasets. For time series, use **chronological splits** (no shuffling). For small datasets, prefer **k-fold cross-validation** over a single train-test split.

---
