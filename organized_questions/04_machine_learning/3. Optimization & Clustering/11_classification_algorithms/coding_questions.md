# Classification Algorithms Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Implement a logistic regression model from scratch using Python.**

### Answer

**Definition:**
Logistic Regression predicts binary outcomes using the sigmoid function to map linear combinations to probabilities [0,1]. Implementation involves forward pass (prediction), loss calculation (binary cross-entropy), and gradient descent optimization.

**Key Components:**
- Sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- Loss: $J = -\frac{1}{m}\sum[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$
- Gradients: $\frac{\partial J}{\partial w} = \frac{1}{m}X^T(\hat{y} - y)$

**Python Implementation:**
```python
import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iter):
            # Forward pass
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict_proba(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# Test the implementation
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.3f}")
```

**Interview Tips:**
- Always mention sigmoid clipping to prevent overflow
- Explain gradient derivation if asked
- Know vectorized vs loop implementation
- Mention regularization extension (add λw to gradient)

---

## Question 2
- [ ] Done

**Write a function that calculates the Gini impurity for a given dataset in a Decision Tree.**

### Answer

**Definition:**
Gini impurity measures the probability of incorrectly classifying a randomly chosen element. It ranges from 0 (pure) to 0.5 (binary) or higher (multiclass). Decision trees use it to select optimal splits.

**Formula:**
$$Gini = 1 - \sum_{i=1}^{C} p_i^2$$

Where $p_i$ is the proportion of class $i$ in the node.

**Python Implementation:**
```python
import numpy as np
from collections import Counter

def gini_impurity(y):
    """
    Calculate Gini impurity for a node.
    
    Args:
        y: Array of class labels
    Returns:
        Gini impurity value (0 = pure, 0.5 = max for binary)
    """
    if len(y) == 0:
        return 0
    
    # Count class frequencies
    counter = Counter(y)
    n_samples = len(y)
    
    # Calculate Gini: 1 - sum(p_i^2)
    gini = 1.0
    for count in counter.values():
        p = count / n_samples
        gini -= p ** 2
    
    return gini

def weighted_gini(y_left, y_right):
    """
    Calculate weighted Gini for a split.
    """
    n_total = len(y_left) + len(y_right)
    if n_total == 0:
        return 0
    
    w_left = len(y_left) / n_total
    w_right = len(y_right) / n_total
    
    return w_left * gini_impurity(y_left) + w_right * gini_impurity(y_right)

def information_gain_gini(y_parent, y_left, y_right):
    """
    Calculate information gain using Gini impurity.
    """
    return gini_impurity(y_parent) - weighted_gini(y_left, y_right)

# Example usage
y_pure = np.array([1, 1, 1, 1, 1])
y_mixed = np.array([1, 1, 0, 0, 1])
y_balanced = np.array([1, 1, 0, 0])

print(f"Pure node Gini: {gini_impurity(y_pure):.3f}")      # 0.0
print(f"Mixed node Gini: {gini_impurity(y_mixed):.3f}")    # 0.48
print(f"Balanced node Gini: {gini_impurity(y_balanced):.3f}")  # 0.5

# Split evaluation
y_parent = np.array([1, 1, 1, 0, 0, 0])
y_left = np.array([1, 1, 1])  # Pure
y_right = np.array([0, 0, 0])  # Pure
print(f"Information gain: {information_gain_gini(y_parent, y_left, y_right):.3f}")  # 0.5
```

**Comparison with Entropy:**

| Metric | Formula | Range |
|--------|---------|-------|
| Gini | $1 - \sum p_i^2$ | 0 to 0.5 (binary) |
| Entropy | $-\sum p_i \log_2(p_i)$ | 0 to 1 (binary) |

**Interview Tips:**
- Gini is computationally faster (no log)
- Both usually give similar results
- Gini prefers larger partitions, Entropy prefers balanced
- Know that sklearn uses Gini by default

---

## Question 3
- [ ] Done

**Code a Support Vector Machine using scikit-learn to classify data from a toy dataset.**

### Answer

**Definition:**
SVM finds the optimal hyperplane that maximizes the margin between classes. It uses kernel trick for non-linear boundaries and support vectors (closest points) define the decision boundary.

**Python Implementation:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# 1. Linear SVM on linearly separable data
X_linear, y_linear = make_classification(n_samples=200, n_features=2, 
                                          n_redundant=0, n_informative=2,
                                          random_state=42, n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(X_linear, y_linear, test_size=0.2)

# Scale features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)
print(f"Linear SVM Accuracy: {svm_linear.score(X_test_scaled, y_test):.3f}")

# 2. Non-linear SVM on moons dataset
X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons, test_size=0.2)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RBF kernel for non-linear boundaries
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train_scaled, y_train)
print(f"RBF SVM Accuracy: {svm_rbf.score(X_test_scaled, y_test):.3f}")

# 3. Hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1, 1],
    'kernel': ['rbf', 'poly']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# 4. Visualize decision boundary
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black')
    plt.title(title)
    plt.show()

# plot_decision_boundary(svm_rbf, X_train_scaled, y_train, "SVM RBF Kernel")
```

**Key Parameters:**

| Parameter | Description | Effect |
|-----------|-------------|--------|
| C | Regularization | High = low bias, low = low variance |
| kernel | Transformation | linear, rbf, poly, sigmoid |
| gamma | RBF width | High = complex boundary |

**Interview Tips:**
- Always scale features before SVM
- RBF kernel works for most non-linear problems
- C controls margin softness (higher = harder margin)
- Support vectors are the only points that matter

---

## Question 4
- [ ] Done

**Create a k-NN classifier in Python and test its performance on a sample dataset.**

### Answer

**Definition:**
k-NN classifies by majority vote of k nearest neighbors. It's a lazy learner (no training), distance-based, and sensitive to k choice and feature scaling.

**Python Implementation:**
```python
import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# From scratch implementation
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):
        # Calculate distances to all training points
        distances = [self.euclidean_distance(x, x_train) 
                    for x_train in self.X_train]
        
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        
        # Majority vote
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

# Load and prepare data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test scratch implementation
knn_scratch = KNNClassifier(k=5)
knn_scratch.fit(X_train_scaled, y_train)
y_pred_scratch = knn_scratch.predict(X_test_scaled)
print(f"Scratch KNN Accuracy: {np.mean(y_pred_scratch == y_test):.3f}")

# Sklearn implementation
knn_sklearn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_sklearn.fit(X_train_scaled, y_train)
y_pred = knn_sklearn.predict(X_test_scaled)
print(f"Sklearn KNN Accuracy: {knn_sklearn.score(X_test_scaled, y_test):.3f}")

# Find optimal k using cross-validation
k_values = range(1, 21)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal k: {optimal_k}, CV Score: {max(cv_scores):.3f}")

# Classification report
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

**Key Considerations:**

| Aspect | Recommendation |
|--------|----------------|
| k value | Odd number, use CV to find optimal |
| Scaling | Always scale (distance-based) |
| Distance | Euclidean, Manhattan, Minkowski |
| Weighting | 'uniform' or 'distance' |

**Interview Tips:**
- k=1 overfits, large k underfits
- Scaling is critical for distance metrics
- Curse of dimensionality affects high-d data
- Memory-intensive (stores all training data)

---

## Question 5
- [ ] Done

**Use a Boosting algorithm to improve the accuracy of a weak classifier on a dataset.**

### Answer

**Definition:**
Boosting sequentially trains weak learners, each focusing on errors of previous ones. Final prediction is weighted vote. Popular algorithms: AdaBoost, Gradient Boosting, XGBoost.

**Concept:**
```
Iteration 1: Train weak learner → Find errors → Increase weights
Iteration 2: Train on reweighted data → Find errors → Update weights
...
Final: Weighted combination of all weak learners
```

**Python Implementation:**
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Create dataset
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 1. Weak classifier baseline (depth=1 tree = decision stump)
weak_clf = DecisionTreeClassifier(max_depth=1)
weak_clf.fit(X_train, y_train)
print(f"Weak Classifier Accuracy: {weak_clf.score(X_test, y_test):.3f}")

# 2. AdaBoost
ada_boost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada_boost.fit(X_train, y_train)
print(f"AdaBoost Accuracy: {ada_boost.score(X_test, y_test):.3f}")

# 3. Gradient Boosting
gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb_clf.fit(X_train, y_train)
print(f"Gradient Boosting Accuracy: {gb_clf.score(X_test, y_test):.3f}")

# 4. XGBoost
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_clf.fit(X_train, y_train)
print(f"XGBoost Accuracy: {xgb_clf.score(X_test, y_test):.3f}")

# 5. Feature importance from XGBoost
feature_importance = xgb_clf.feature_importances_
top_features = np.argsort(feature_importance)[-5:]
print(f"Top 5 important features: {top_features}")

# Show improvement
print(f"\nImprovement: {weak_clf.score(X_test, y_test):.3f} → {xgb_clf.score(X_test, y_test):.3f}")
```

**Comparison:**

| Algorithm | Key Idea | Strength |
|-----------|----------|----------|
| AdaBoost | Reweight samples | Simple, interpretable |
| Gradient Boosting | Fit residuals | Handles complex patterns |
| XGBoost | Regularized GB | Fast, handles missing |
| LightGBM | Leaf-wise growth | Very fast, large data |

**Interview Tips:**
- Boosting reduces bias (vs Bagging reduces variance)
- Learning rate controls contribution of each tree
- Early stopping prevents overfitting
- XGBoost/LightGBM are production standards

---

## Question 6
- [ ] Done

**Implement a function for feature scaling and normalization in preparation for classification.**

### Answer

**Definition:**
Feature scaling transforms features to similar ranges, crucial for distance-based and gradient-based algorithms. Standardization (z-score) and Min-Max normalization are most common.

**Scaling Methods:**

| Method | Formula | Range | Use Case |
|--------|---------|-------|----------|
| Standardization | $(x - \mu) / \sigma$ | ~ [-3, 3] | Most algorithms |
| Min-Max | $(x - min) / (max - min)$ | [0, 1] | Neural networks |
| Robust | $(x - median) / IQR$ | Variable | Outliers present |
| MaxAbs | $x / max(\|x\|)$ | [-1, 1] | Sparse data |

**Python Implementation:**
```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# From scratch implementations
def standardize(X):
    """Z-score normalization: (x - mean) / std"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return (X - mean) / std, mean, std

def min_max_scale(X, feature_range=(0, 1)):
    """Min-Max scaling to [0, 1] or custom range"""
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1  # Avoid division by zero
    
    X_scaled = (X - min_val) / range_val
    
    # Scale to custom range
    min_r, max_r = feature_range
    X_scaled = X_scaled * (max_r - min_r) + min_r
    return X_scaled, min_val, max_val

def robust_scale(X):
    """Robust scaling using median and IQR"""
    median = np.median(X, axis=0)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1
    return (X - median) / iqr, median, iqr

# Example usage
np.random.seed(42)
X = np.random.randn(100, 3) * [10, 100, 1000] + [5, 50, 500]  # Different scales

# From scratch
X_std, mean, std = standardize(X)
X_mm, min_v, max_v = min_max_scale(X)
X_robust, med, iqr = robust_scale(X)

print("Original ranges:")
print(f"  Feature 1: [{X[:, 0].min():.1f}, {X[:, 0].max():.1f}]")
print(f"  Feature 2: [{X[:, 1].min():.1f}, {X[:, 1].max():.1f}]")
print(f"  Feature 3: [{X[:, 2].min():.1f}, {X[:, 2].max():.1f}]")

print("\nAfter Standardization (mean=0, std=1):")
print(f"  Mean: {X_std.mean(axis=0).round(3)}")
print(f"  Std: {X_std.std(axis=0).round(3)}")

print("\nAfter Min-Max [0, 1]:")
print(f"  Min: {X_mm.min(axis=0).round(3)}")
print(f"  Max: {X_mm.max(axis=0).round(3)}")

# Sklearn comparison
scaler_std = StandardScaler()
scaler_mm = MinMaxScaler()
scaler_robust = RobustScaler()

X_std_sk = scaler_std.fit_transform(X)
X_mm_sk = scaler_mm.fit_transform(X)
X_robust_sk = scaler_robust.fit_transform(X)

# Important: Transform test data using train statistics
X_test = np.random.randn(20, 3) * [10, 100, 1000] + [5, 50, 500]
X_test_scaled = scaler_std.transform(X_test)  # Use fit from training!
```

**When to Use What:**

| Algorithm | Recommended Scaling |
|-----------|---------------------|
| SVM, KNN | StandardScaler |
| Neural Networks | MinMaxScaler [0,1] or [-1,1] |
| Tree-based | None needed |
| With outliers | RobustScaler |

**Interview Tips:**
- Always fit on train, transform on test
- Tree-based models don't need scaling
- Standardization is safest default choice
- Check for outliers before choosing method

---

## Question 7
- [ ] Done

**Develop a Python script that visualizes the decision boundary of a given classification model.**

### Answer

**Definition:**
Decision boundary visualization shows how a classifier separates classes in feature space. It helps understand model complexity, overfitting, and compare algorithms visually.

**Python Implementation:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def plot_decision_boundary(model, X, y, title="Decision Boundary", ax=None):
    """
    Plot decision boundary for a 2D classification problem.
    
    Args:
        model: Fitted classifier
        X: Feature matrix (n_samples, 2)
        y: Labels
        title: Plot title
        ax: Matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create mesh grid
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    ax.contour(xx, yy, Z, colors='black', linewidths=0.5)
    
    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, 
                         edgecolors='black', s=50)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    
    return ax

def compare_classifiers(X, y, classifiers, titles):
    """Compare decision boundaries of multiple classifiers."""
    n_classifiers = len(classifiers)
    fig, axes = plt.subplots(1, n_classifiers, figsize=(5*n_classifiers, 4))
    
    if n_classifiers == 1:
        axes = [axes]
    
    for ax, clf, title in zip(axes, classifiers, titles):
        clf.fit(X, y)
        plot_decision_boundary(clf, X, y, title, ax)
    
    plt.tight_layout()
    plt.show()

# Example: Compare classifiers on different datasets
# Dataset 1: Linearly separable
X_linear, y_linear = make_classification(n_samples=200, n_features=2, 
                                          n_redundant=0, n_informative=2,
                                          random_state=42, n_clusters_per_class=1)

# Dataset 2: Non-linear (moons)
X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)

# Dataset 3: Circles
X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)

# Scale data
scaler = StandardScaler()
X_linear = scaler.fit_transform(X_linear)
X_moons = scaler.fit_transform(X_moons)
X_circles = scaler.fit_transform(X_circles)

# Define classifiers
classifiers = [
    LogisticRegression(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(n_estimators=10),
    SVC(kernel='rbf', gamma='scale'),
    KNeighborsClassifier(n_neighbors=5)
]

titles = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM (RBF)', 'KNN (k=5)']

# Visualize on moons dataset
print("Decision Boundaries on Moons Dataset:")
compare_classifiers(X_moons, y_moons, classifiers, titles)

# Single model visualization with probability
def plot_with_probability(model, X, y, title):
    """Plot decision boundary with probability contours."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot probability contours
    contour = ax.contourf(xx, yy, Z, levels=20, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.colorbar(contour, ax=ax, label='Probability')
    
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, 
               edgecolors='black', s=50)
    ax.set_title(title)
    plt.show()

# Example with probability
lr = LogisticRegression()
lr.fit(X_moons, y_moons)
# plot_with_probability(lr, X_moons, y_moons, "Logistic Regression Probability")
```

**Interview Tips:**
- Meshgrid creates the grid for plotting
- Lower step size (h) = smoother boundary, slower
- Compare models visually to explain overfitting
- Mention complexity vs decision boundary smoothness

---
