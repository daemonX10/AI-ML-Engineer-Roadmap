# Decision Tree Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Implement a basic Decision Tree algorithm from scratch in Python.**

**Answer:**

```python
import numpy as np
from collections import Counter

class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def gini(self, y):
        """Calculate Gini impurity."""
        counts = Counter(y)
        n = len(y)
        return 1 - sum((count / n) ** 2 for count in counts.values())
    
    def best_split(self, X, y):
        """Find best feature and threshold to split on."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
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
                
                # Calculate gain
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
            # Return leaf node with majority class
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}
        
        # Find best split
        feature, threshold = self.best_split(X, y)
        
        if feature is None:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}
        
        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # Build subtrees
        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X, y):
        """Train the decision tree."""
        self.tree = self.build_tree(np.array(X), np.array(y))
        return self
    
    def predict_sample(self, x, node):
        """Predict single sample."""
        if node['leaf']:
            return node['class']
        
        if x[node['feature']] <= node['threshold']:
            return self.predict_sample(x, node['left'])
        return self.predict_sample(x, node['right'])
    
    def predict(self, X):
        """Predict for multiple samples."""
        return np.array([self.predict_sample(x, self.tree) for x in np.array(X)])


# Usage Example
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Load data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

---

## Question 2
- [ ] Done

**Write a Python function to compute Gini impurity given a dataset.**

**Answer:**

```python
import numpy as np
from collections import Counter

def gini_impurity(y):
    """
    Calculate Gini impurity for a set of labels.
    
    Gini = 1 - sum(p_i^2)
    
    Parameters:
        y: array-like, class labels
    
    Returns:
        float: Gini impurity (0 = pure, 0.5 = max impurity for binary)
    """
    if len(y) == 0:
        return 0
    
    counts = Counter(y)
    n = len(y)
    
    gini = 1 - sum((count / n) ** 2 for count in counts.values())
    return gini


def gini_split(y_left, y_right):
    """
    Calculate weighted Gini impurity for a split.
    
    Parameters:
        y_left: labels in left child
        y_right: labels in right child
    
    Returns:
        float: weighted average Gini impurity
    """
    n_total = len(y_left) + len(y_right)
    
    if n_total == 0:
        return 0
    
    weighted_gini = (
        (len(y_left) / n_total) * gini_impurity(y_left) +
        (len(y_right) / n_total) * gini_impurity(y_right)
    )
    return weighted_gini


def gini_gain(y_parent, y_left, y_right):
    """
    Calculate Gini gain from a split.
    
    Gain = Gini(parent) - weighted_Gini(children)
    
    Parameters:
        y_parent: labels before split
        y_left: labels in left child
        y_right: labels in right child
    
    Returns:
        float: Gini gain (higher = better split)
    """
    parent_gini = gini_impurity(y_parent)
    children_gini = gini_split(y_left, y_right)
    return parent_gini - children_gini


# Usage Examples
if __name__ == "__main__":
    # Example 1: Pure node (all same class)
    pure = [0, 0, 0, 0, 0]
    print(f"Pure node Gini: {gini_impurity(pure):.3f}")  # 0.0
    
    # Example 2: Maximum impurity (50-50 split)
    max_impure = [0, 0, 1, 1]
    print(f"50-50 split Gini: {gini_impurity(max_impure):.3f}")  # 0.5
    
    # Example 3: Multiclass
    multiclass = [0, 0, 0, 1, 1, 2]
    print(f"Multiclass Gini: {gini_impurity(multiclass):.3f}")  # 0.611
    
    # Example 4: Split evaluation
    parent = [0, 0, 0, 1, 1, 1]
    left = [0, 0, 0]  # Pure
    right = [1, 1, 1]  # Pure
    print(f"Perfect split gain: {gini_gain(parent, left, right):.3f}")  # 0.5
    
    # Example 5: Bad split
    bad_left = [0, 0, 1, 1]
    bad_right = [0, 1]
    print(f"Bad split gain: {gini_gain(parent, bad_left, bad_right):.3f}")  # ~0.0
```

**Output:**
```
Pure node Gini: 0.000
50-50 split Gini: 0.500
Multiclass Gini: 0.611
Perfect split gain: 0.500
Bad split gain: 0.000
```

---

## Question 3
- [ ] Done

**Create a Python script to visualize a Decision Tree using graphviz.**

**Answer:**

```python
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import load_iris
import graphviz

# Load and train
X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names
class_names = load_iris().target_names

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

# Method 1: Export to Graphviz and render
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,              # Color by class
    rounded=True,             # Rounded boxes
    special_characters=True,  # Allow special chars
    proportion=True,          # Show proportions
    precision=2               # Decimal precision
)

# Render and save
graph = graphviz.Source(dot_data)
graph.render("iris_tree", format="png", cleanup=True)
print("Saved: iris_tree.png")


# Method 2: Using sklearn's plot_tree (no graphviz needed)
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    fontsize=10,
    rounded=True
)
plt.tight_layout()
plt.savefig("iris_tree_matplotlib.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved: iris_tree_matplotlib.png")


# Method 3: Text representation (no visualization library needed)
from sklearn.tree import export_text

tree_rules = export_text(clf, feature_names=list(feature_names))
print("\nTree Rules:")
print(tree_rules)


# Method 4: Interactive visualization with dtreeviz (if installed)
# pip install dtreeviz
try:
    from dtreeviz.trees import dtreeviz
    
    viz = dtreeviz(
        clf,
        X, y,
        feature_names=list(feature_names),
        class_names=list(class_names),
        title="Iris Decision Tree"
    )
    viz.save("iris_tree_dtreeviz.svg")
    print("Saved: iris_tree_dtreeviz.svg")
except ImportError:
    print("dtreeviz not installed. Install with: pip install dtreeviz")
```

**Output (Text Rules):**
```
Tree Rules:
|--- petal length (cm) <= 2.45
|   |--- class: setosa
|--- petal length (cm) >  2.45
|   |--- petal width (cm) <= 1.75
|   |   |--- petal length (cm) <= 4.95
|   |   |   |--- class: versicolor
|   |   |--- petal length (cm) >  4.95
|   |   |   |--- class: virginica
|   |--- petal width (cm) >  1.75
|   |   |--- class: virginica
```

**Installation:**
```bash
pip install graphviz
# Also install Graphviz system package:
# Windows: choco install graphviz
# Mac: brew install graphviz
# Linux: apt-get install graphviz
```

---

## Question 4
- [ ] Done

**Using scikit-learn, train a Decision Tree classifier on a sample dataset and evaluate its performance.**

**Answer:**

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt

# Step 1: Load data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
print(f"Dataset shape: {X.shape}")
print(f"Classes: {data.target_names}")

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Train Decision Tree
clf = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
clf.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Step 5: Evaluate performance
print("\n=== Model Evaluation ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.3f}")

# Step 6: Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Step 7: Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Step 8: Cross-validation
cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Step 9: Feature importance
print("\nTop 10 Important Features:")
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1][:10]
for i, idx in enumerate(indices):
    print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")

# Step 10: Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(10), importances[indices][::-1])
plt.yticks(range(10), [feature_names[i] for i in indices[::-1]])
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
```

**Output:**
```
Dataset shape: (569, 30)
Classes: ['malignant' 'benign']

=== Model Evaluation ===
Accuracy:  0.947
Precision: 0.958
Recall:    0.958
F1 Score:  0.958
AUC-ROC:   0.943

Confusion Matrix:
[[40  3]
 [ 3 68]]

Classification Report:
              precision    recall  f1-score   support
   malignant       0.93      0.93      0.93        43
      benign       0.96      0.96      0.96        71
    accuracy                           0.95       114

5-Fold CV Accuracy: 0.921 ± 0.022
```

---

## Question 5
- [ ] Done

**Implement a recursive binary splitting algorithm for a regression Decision Tree.**

**Answer:**

```python
import numpy as np

class RegressionTree:
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    
    def mse(self, y):
        """Calculate Mean Squared Error (variance) for a node."""
        if len(y) == 0:
            return 0
        return np.var(y) * len(y)  # Total squared error
    
    def find_best_split(self, X, y):
        """Find best feature and threshold using MSE reduction."""
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feature in range(n_features):
            # Get unique values as potential thresholds
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                # Split data
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                # Check min_samples_leaf constraint
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                # Calculate total MSE after split
                left_mse = self.mse(y[left_mask])
                right_mse = self.mse(y[right_mask])
                total_mse = left_mse + right_mse
                
                if total_mse < best_mse:
                    best_mse = total_mse
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        """Recursively build the regression tree."""
        n_samples = len(y)
        
        # Stopping conditions
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split):
            # Return leaf with mean value
            return {'leaf': True, 'value': np.mean(y)}
        
        # Find best split
        feature, threshold = self.find_best_split(X, y)
        
        if feature is None:
            return {'leaf': True, 'value': np.mean(y)}
        
        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # Recursive binary splitting
        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X, y):
        """Train the regression tree."""
        self.tree = self.build_tree(np.array(X), np.array(y))
        return self
    
    def predict_sample(self, x, node):
        """Predict single sample."""
        if node['leaf']:
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self.predict_sample(x, node['left'])
        return self.predict_sample(x, node['right'])
    
    def predict(self, X):
        """Predict for multiple samples."""
        return np.array([self.predict_sample(x, self.tree) for x in np.array(X)])


# Usage Example
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Generate sample data
    X, y = make_regression(n_samples=200, n_features=4, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate
    reg = RegressionTree(max_depth=5, min_samples_leaf=5)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    print(f"MSE:  {mean_squared_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R²:   {r2_score(y_test, y_pred):.3f}")
```

**Output:**
```
MSE:  1245.67
RMSE: 35.29
R²:   0.892
```

---

## Question 6
- [ ] Done

**Write a function in Python that prunes a Decision Tree to avoid overfitting.**

**Answer:**

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

def cost_complexity_pruning(X_train, y_train, X_val, y_val):
    """
    Find optimal pruning parameter (ccp_alpha) using validation set.
    
    Returns:
        best_alpha: optimal complexity parameter
        pruned_tree: Decision tree with optimal pruning
    """
    # Step 1: Get the cost complexity pruning path
    clf = DecisionTreeClassifier(random_state=42)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    
    # Step 2: Train trees for each alpha value
    trees = []
    for alpha in ccp_alphas:
        tree = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
        tree.fit(X_train, y_train)
        trees.append(tree)
    
    # Step 3: Evaluate on validation set
    val_scores = [tree.score(X_val, y_val) for tree in trees]
    train_scores = [tree.score(X_train, y_train) for tree in trees]
    
    # Step 4: Find best alpha
    best_idx = np.argmax(val_scores)
    best_alpha = ccp_alphas[best_idx]
    
    return best_alpha, trees[best_idx], ccp_alphas, train_scores, val_scores


def prune_with_cv(X, y, cv=5):
    """
    Find optimal pruning using cross-validation.
    
    Returns:
        best_alpha: optimal complexity parameter
        pruned_tree: final pruned tree trained on all data
    """
    # Get pruning path
    clf = DecisionTreeClassifier(random_state=42)
    path = clf.cost_complexity_pruning_path(X, y)
    ccp_alphas = path.ccp_alphas
    
    # Cross-validate for each alpha
    mean_scores = []
    std_scores = []
    
    for alpha in ccp_alphas:
        tree = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
        scores = cross_val_score(tree, X, y, cv=cv, scoring='accuracy')
        mean_scores.append(scores.mean())
        std_scores.append(scores.std())
    
    # Best alpha
    best_idx = np.argmax(mean_scores)
    best_alpha = ccp_alphas[best_idx]
    
    # Train final model
    pruned_tree = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
    pruned_tree.fit(X, y)
    
    return best_alpha, pruned_tree, ccp_alphas, mean_scores


# Main execution
if __name__ == "__main__":
    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    
    # Method 1: Using validation set
    print("=== Method 1: Validation Set Pruning ===")
    best_alpha, pruned_tree, alphas, train_scores, val_scores = cost_complexity_pruning(
        X_train, y_train, X_val, y_val
    )
    print(f"Best alpha: {best_alpha:.6f}")
    print(f"Tree depth: {pruned_tree.get_depth()}")
    print(f"Num leaves: {pruned_tree.get_n_leaves()}")
    print(f"Test accuracy: {pruned_tree.score(X_test, y_test):.3f}")
    
    # Compare with unpruned
    unpruned = DecisionTreeClassifier(random_state=42)
    unpruned.fit(X_train, y_train)
    print(f"\nUnpruned depth: {unpruned.get_depth()}")
    print(f"Unpruned leaves: {unpruned.get_n_leaves()}")
    print(f"Unpruned test accuracy: {unpruned.score(X_test, y_test):.3f}")
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(alphas, train_scores, label='Train')
    plt.plot(alphas, val_scores, label='Validation')
    plt.axvline(best_alpha, color='r', linestyle='--', label=f'Best α={best_alpha:.4f}')
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Cost Complexity Pruning')
    plt.savefig('pruning_curve.png')
    plt.show()
```

**Output:**
```
=== Method 1: Validation Set Pruning ===
Best alpha: 0.008547
Tree depth: 4
Num leaves: 9
Test accuracy: 0.956

Unpruned depth: 7
Unpruned leaves: 17
Unpruned test accuracy: 0.939
```

---

## Question 7
- [ ] Done

**Code a Python function to calculate feature importance from a trained Decision Tree.**

**Answer:**

```python
import numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def calculate_feature_importance(tree, n_features):
    """
    Calculate feature importance from a sklearn Decision Tree.
    
    Importance = total weighted impurity reduction for each feature
    normalized to sum to 1.
    
    Parameters:
        tree: sklearn tree object (clf.tree_)
        n_features: number of features
    
    Returns:
        numpy array of feature importances
    """
    # Get tree structure
    n_nodes = tree.node_count
    feature = tree.feature
    threshold = tree.threshold
    impurity = tree.impurity
    n_node_samples = tree.n_node_samples
    children_left = tree.children_left
    children_right = tree.children_right
    
    # Calculate importance for each node
    importances = np.zeros(n_features)
    
    for node_id in range(n_nodes):
        # Skip leaf nodes
        if children_left[node_id] == -1:
            continue
        
        # Get feature used at this node
        feat = feature[node_id]
        
        # Calculate weighted impurity decrease
        left_child = children_left[node_id]
        right_child = children_right[node_id]
        
        n_parent = n_node_samples[node_id]
        n_left = n_node_samples[left_child]
        n_right = n_node_samples[right_child]
        
        impurity_decrease = (
            n_parent * impurity[node_id] -
            n_left * impurity[left_child] -
            n_right * impurity[right_child]
        )
        
        importances[feat] += impurity_decrease
    
    # Normalize to sum to 1
    total = importances.sum()
    if total > 0:
        importances = importances / total
    
    return importances


def get_feature_importance_dict(clf, feature_names):
    """
    Return feature importance as a sorted dictionary.
    """
    importances = clf.feature_importances_
    return dict(sorted(
        zip(feature_names, importances),
        key=lambda x: -x[1]
    ))


def plot_feature_importance(clf, feature_names, top_n=None):
    """
    Plot feature importance as horizontal bar chart.
    """
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    if top_n:
        indices = indices[:top_n]
    
    plt.figure(figsize=(10, len(indices) * 0.4))
    plt.barh(range(len(indices)), importances[indices][::-1])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices[::-1]])
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.tight_layout()
    return plt


# Main execution
if __name__ == "__main__":
    # Load data and train
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X, y)
    
    # Method 1: Built-in sklearn
    print("=== sklearn feature_importances_ ===")
    for name, imp in zip(feature_names, clf.feature_importances_):
        print(f"{name}: {imp:.4f}")
    
    # Method 2: Manual calculation
    print("\n=== Manual calculation ===")
    manual_imp = calculate_feature_importance(clf.tree_, X.shape[1])
    for name, imp in zip(feature_names, manual_imp):
        print(f"{name}: {imp:.4f}")
    
    # Verify they match
    print(f"\nMatch: {np.allclose(clf.feature_importances_, manual_imp)}")
    
    # Method 3: As dictionary
    print("\n=== Sorted Dictionary ===")
    imp_dict = get_feature_importance_dict(clf, feature_names)
    for name, imp in imp_dict.items():
        print(f"{name}: {imp:.4f}")
    
    # Plot
    plot_feature_importance(clf, feature_names)
    plt.savefig('feature_importance.png')
    plt.show()
```

**Output:**
```
=== sklearn feature_importances_ ===
sepal length (cm): 0.0000
sepal width (cm): 0.0000
petal length (cm): 0.4271
petal width (cm): 0.5729

=== Manual calculation ===
sepal length (cm): 0.0000
sepal width (cm): 0.0000
petal length (cm): 0.4271
petal width (cm): 0.5729

Match: True

=== Sorted Dictionary ===
petal width (cm): 0.5729
petal length (cm): 0.4271
sepal length (cm): 0.0000
sepal width (cm): 0.0000
```

---

## Question 8
- [ ] Done

**Use cross-validation in Python to determine the optimal depth for a Decision Tree.**

**Answer:**

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt

# Load data
X, y = load_breast_cancer(return_X_y=True)

# Method 1: Manual cross-validation loop
print("=== Method 1: Manual CV Loop ===")
depths = range(1, 20)
cv_means = []
cv_stds = []
train_scores = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    
    # Cross-validation scores
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    cv_means.append(cv_scores.mean())
    cv_stds.append(cv_scores.std())
    
    # Training score
    clf.fit(X, y)
    train_scores.append(clf.score(X, y))

# Find optimal depth
optimal_depth = depths[np.argmax(cv_means)]
best_cv_score = max(cv_means)
print(f"Optimal depth: {optimal_depth}")
print(f"Best CV accuracy: {best_cv_score:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, 'o-', label='Training')
plt.plot(depths, cv_means, 'o-', label='CV Mean')
plt.fill_between(
    depths,
    np.array(cv_means) - np.array(cv_stds),
    np.array(cv_means) + np.array(cv_stds),
    alpha=0.2
)
plt.axvline(optimal_depth, color='r', linestyle='--', label=f'Optimal depth={optimal_depth}')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Finding Optimal Tree Depth with Cross-Validation')
plt.grid(True)
plt.savefig('optimal_depth_cv.png')
plt.show()


# Method 2: Using GridSearchCV
print("\n=== Method 2: GridSearchCV ===")
param_grid = {'max_depth': list(range(1, 20))}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    return_train_score=True
)
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Detailed results
results = grid_search.cv_results_
print("\nAll results:")
for depth, mean, std in zip(
    results['param_max_depth'],
    results['mean_test_score'],
    results['std_test_score']
):
    print(f"  depth={depth}: {mean:.4f} ± {std:.4f}")


# Method 3: GridSearchCV with multiple parameters
print("\n=== Method 3: Full Hyperparameter Tuning ===")
param_grid_full = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_full = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid_full,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_full.fit(X, y)

print(f"Best parameters: {grid_full.best_params_}")
print(f"Best CV score: {grid_full.best_score_:.4f}")

# Final model
best_model = grid_full.best_estimator_
print(f"Final tree depth: {best_model.get_depth()}")
print(f"Final tree leaves: {best_model.get_n_leaves()}")
```

**Output:**
```
=== Method 1: Manual CV Loop ===
Optimal depth: 4
Best CV accuracy: 0.9314

=== Method 2: GridSearchCV ===
Best parameters: {'max_depth': 4}
Best CV score: 0.9314

=== Method 3: Full Hyperparameter Tuning ===
Best parameters: {'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 2}
Best CV score: 0.9367
Final tree depth: 5
Final tree leaves: 11
```

---

## Question 9
- [ ] Done

**Build a Random Forest model in Python and compare its performance with a single Decision Tree.**

**Answer:**

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import time

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=== Single Decision Tree ===")
# Train Decision Tree
start = time.time()
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
dt_train_time = time.time() - start

# Evaluate
dt_train_score = dt.score(X_train, y_train)
dt_test_score = dt.score(X_test, y_test)
dt_cv_scores = cross_val_score(dt, X_train, y_train, cv=5)

print(f"Training time: {dt_train_time:.4f}s")
print(f"Train accuracy: {dt_train_score:.4f}")
print(f"Test accuracy: {dt_test_score:.4f}")
print(f"CV accuracy: {dt_cv_scores.mean():.4f} ± {dt_cv_scores.std():.4f}")
print(f"Tree depth: {dt.get_depth()}")
print(f"Num leaves: {dt.get_n_leaves()}")


print("\n=== Random Forest ===")
# Train Random Forest
start = time.time()
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_train_time = time.time() - start

# Evaluate
rf_train_score = rf.score(X_train, y_train)
rf_test_score = rf.score(X_test, y_test)
rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=5)

print(f"Training time: {rf_train_time:.4f}s")
print(f"Train accuracy: {rf_train_score:.4f}")
print(f"Test accuracy: {rf_test_score:.4f}")
print(f"CV accuracy: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
print(f"Num trees: {rf.n_estimators}")


print("\n=== Comparison Summary ===")
comparison = {
    'Metric': ['Train Accuracy', 'Test Accuracy', 'CV Accuracy', 'Training Time (s)'],
    'Decision Tree': [dt_train_score, dt_test_score, dt_cv_scores.mean(), dt_train_time],
    'Random Forest': [rf_train_score, rf_test_score, rf_cv_scores.mean(), rf_train_time]
}

for i, metric in enumerate(comparison['Metric']):
    dt_val = comparison['Decision Tree'][i]
    rf_val = comparison['Random Forest'][i]
    print(f"{metric}: DT={dt_val:.4f}, RF={rf_val:.4f}")


# Variance comparison (run multiple times with different seeds)
print("\n=== Variance Comparison ===")
dt_scores = []
rf_scores = []

for seed in range(10):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    dt_temp = DecisionTreeClassifier(max_depth=10, random_state=seed)
    dt_temp.fit(X_tr, y_tr)
    dt_scores.append(dt_temp.score(X_te, y_te))
    
    rf_temp = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed)
    rf_temp.fit(X_tr, y_tr)
    rf_scores.append(rf_temp.score(X_te, y_te))

print(f"DT variance: {np.std(dt_scores):.4f} (mean: {np.mean(dt_scores):.4f})")
print(f"RF variance: {np.std(rf_scores):.4f} (mean: {np.mean(rf_scores):.4f})")


# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar chart
metrics = ['Train', 'Test', 'CV']
dt_vals = [dt_train_score, dt_test_score, dt_cv_scores.mean()]
rf_vals = [rf_train_score, rf_test_score, rf_cv_scores.mean()]
x = np.arange(len(metrics))
width = 0.35

axes[0].bar(x - width/2, dt_vals, width, label='Decision Tree')
axes[0].bar(x + width/2, rf_vals, width, label='Random Forest')
axes[0].set_ylabel('Accuracy')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics)
axes[0].legend()
axes[0].set_title('Performance Comparison')

# Variance box plot
axes[1].boxplot([dt_scores, rf_scores], labels=['Decision Tree', 'Random Forest'])
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Variance Across Data Splits')

plt.tight_layout()
plt.savefig('dt_vs_rf_comparison.png')
plt.show()
```

**Output:**
```
=== Single Decision Tree ===
Training time: 0.0035s
Train accuracy: 1.0000
Test accuracy: 0.9386
CV accuracy: 0.9143 ± 0.0323
Tree depth: 10
Num leaves: 27

=== Random Forest ===
Training time: 0.1567s
Train accuracy: 1.0000
Test accuracy: 0.9649
CV accuracy: 0.9582 ± 0.0186
Num trees: 100

=== Comparison Summary ===
Train Accuracy: DT=1.0000, RF=1.0000
Test Accuracy: DT=0.9386, RF=0.9649
CV Accuracy: DT=0.9143, RF=0.9582
Training Time (s): DT=0.0035, RF=0.1567

=== Variance Comparison ===
DT variance: 0.0298 (mean: 0.9254)
RF variance: 0.0142 (mean: 0.9596)
```

**Key Insight:** Random Forest has higher accuracy AND lower variance than a single Decision Tree.

---

## Question 10
- [ ] Done

**Implement a simple version of the AdaBoost algorithm with Decision Trees in Python.**

**Answer:**

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SimpleAdaBoost:
    """
    Simple AdaBoost implementation with Decision Tree stumps.
    """
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []       # Weak learners
        self.alphas = []       # Learner weights
        
    def fit(self, X, y):
        """
        Train AdaBoost classifier.
        
        Parameters:
            X: feature matrix (n_samples, n_features)
            y: labels (should be -1 or 1)
        """
        n_samples = X.shape[0]
        
        # Initialize uniform weights
        weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            # Step 1: Train weak learner (stump) on weighted data
            stump = DecisionTreeClassifier(max_depth=1)  # Stump
            stump.fit(X, y, sample_weight=weights)
            
            # Step 2: Make predictions
            predictions = stump.predict(X)
            
            # Step 3: Calculate weighted error
            incorrect = predictions != y
            error = np.sum(weights * incorrect) / np.sum(weights)
            
            # Avoid division by zero
            error = np.clip(error, 1e-10, 1 - 1e-10)
            
            # Step 4: Calculate alpha (learner weight)
            alpha = 0.5 * np.log((1 - error) / error)
            
            # Step 5: Update sample weights
            weights = weights * np.exp(-alpha * y * predictions)
            weights = weights / np.sum(weights)  # Normalize
            
            # Store learner and its weight
            self.stumps.append(stump)
            self.alphas.append(alpha)
        
        return self
    
    def predict(self, X):
        """
        Make predictions using weighted voting.
        """
        # Sum of weighted predictions
        stump_preds = np.array([
            alpha * stump.predict(X) 
            for stump, alpha in zip(self.stumps, self.alphas)
        ])
        
        # Final prediction: sign of weighted sum
        return np.sign(np.sum(stump_preds, axis=0)).astype(int)
    
    def score(self, X, y):
        """Calculate accuracy."""
        return accuracy_score(y, self.predict(X))


# Usage Example
if __name__ == "__main__":
    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    
    # Convert labels to -1 and 1 (required for AdaBoost)
    y = 2 * y - 1  # 0,1 -> -1,1
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train custom AdaBoost
    print("=== Custom AdaBoost ===")
    ada = SimpleAdaBoost(n_estimators=50)
    ada.fit(X_train, y_train)
    
    print(f"Train accuracy: {ada.score(X_train, y_train):.4f}")
    print(f"Test accuracy: {ada.score(X_test, y_test):.4f}")
    print(f"Number of stumps: {len(ada.stumps)}")
    
    # Compare with sklearn
    print("\n=== sklearn AdaBoostClassifier ===")
    from sklearn.ensemble import AdaBoostClassifier
    
    sklearn_ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        random_state=42
    )
    sklearn_ada.fit(X_train, y_train)
    
    print(f"Train accuracy: {sklearn_ada.score(X_train, y_train):.4f}")
    print(f"Test accuracy: {sklearn_ada.score(X_test, y_test):.4f}")
    
    # Compare with single tree
    print("\n=== Single Decision Tree ===")
    single_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    single_tree.fit(X_train, y_train)
    print(f"Train accuracy: {single_tree.score(X_train, y_train):.4f}")
    print(f"Test accuracy: {single_tree.score(X_test, y_test):.4f}")
    
    # Plot alpha values
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(ada.alphas)), ada.alphas)
    plt.xlabel('Stump Number')
    plt.ylabel('Alpha (Weight)')
    plt.title('AdaBoost Learner Weights')
    plt.savefig('adaboost_alphas.png')
    plt.show()
```

**Output:**
```
=== Custom AdaBoost ===
Train accuracy: 0.9868
Test accuracy: 0.9649

=== sklearn AdaBoostClassifier ===
Train accuracy: 0.9890
Test accuracy: 0.9649

=== Single Decision Tree ===
Train accuracy: 0.9890
Test accuracy: 0.9386
```

**Key Points:**
- Each stump is a weak learner (max_depth=1)
- Misclassified samples get higher weights
- Alpha determines how much each stump contributes
- Final prediction = sign(Σ alpha_i * prediction_i)
