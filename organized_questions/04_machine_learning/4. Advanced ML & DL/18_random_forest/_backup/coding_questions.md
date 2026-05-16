# Random Forest Interview Questions - Coding Questions

## Question 1
- [ ] Done

**What is bagging, and how is it implemented in a Random Forest?**

### Answer

**Definition:**
Bagging (Bootstrap Aggregating) is an ensemble technique that trains multiple models on different bootstrap samples (random samples with replacement) and aggregates their predictions. Random Forest implements bagging with decision trees plus feature randomness.

**How Bagging Works:**

```
Original Dataset (n samples)
        ↓
[Bootstrap Sample 1] → [Model 1] → Prediction 1
[Bootstrap Sample 2] → [Model 2] → Prediction 2
[Bootstrap Sample 3] → [Model 3] → Prediction 3
        ...
[Bootstrap Sample B] → [Model B] → Prediction B
        ↓
    Aggregate: Vote (classification) / Average (regression)
        ↓
    Final Prediction
```

**Implementation from Scratch:**

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

class SimpleBaggingClassifier:
    def __init__(self, n_estimators=10, max_features=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.trees = []
        self.feature_indices = []
    
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def _get_feature_subset(self, n_features):
        if self.max_features is None:
            return np.arange(n_features)
        m = min(self.max_features, n_features)
        return np.random.choice(n_features, size=m, replace=False)
    
    def fit(self, X, y):
        self.trees = []
        self.feature_indices = []
        n_features = X.shape[1]
        
        for _ in range(self.n_estimators):
            # Bootstrap sample
            X_boot, y_boot = self._bootstrap_sample(X, y)
            
            # Feature subset (Random Forest style)
            feat_idx = self._get_feature_subset(n_features)
            self.feature_indices.append(feat_idx)
            
            # Train tree on subset
            tree = DecisionTreeClassifier()
            tree.fit(X_boot[:, feat_idx], y_boot)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        # Get predictions from all trees
        predictions = np.array([
            tree.predict(X[:, feat_idx])
            for tree, feat_idx in zip(self.trees, self.feature_indices)
        ])
        
        # Majority vote
        final_predictions = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            most_common = Counter(votes).most_common(1)[0][0]
            final_predictions.append(most_common)
        
        return np.array(final_predictions)

# Usage
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

bagging = SimpleBaggingClassifier(n_estimators=50, max_features=5)
bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

**Random Forest Adds:**
- Feature randomness at each split (not just per tree)
- Typically deeper trees
- OOB error estimation

---

## Question 2
- [ ] Done

**What are some common implementation challenges with Random Forest?**

### Answer

**Definition:**
Common challenges include handling memory constraints with large forests, slow prediction times, dealing with categorical features, and ensuring reproducibility across runs.

**Challenges and Solutions:**

**1. Memory Issues:**
```python
# Problem: Large forests consume significant memory
# Solution: Limit tree complexity

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,           # Limit depth
    min_samples_leaf=10,    # Larger leaves
    max_leaf_nodes=100,     # Limit total leaves
    n_jobs=-1
)

# Or use incremental training (not native to sklearn)
# Consider LightGBM or XGBoost for memory efficiency
```

**2. Slow Prediction:**
```python
# Problem: Querying many trees is slow
# Solutions:

# a) Fewer trees
rf = RandomForestClassifier(n_estimators=50)

# b) Compile model
import treelite
import treelite_runtime

# Convert to compiled model
model = treelite.sklearn.import_model(rf)
model.export_lib(toolchain='gcc', libpath='./mymodel.so')

# c) Parallel prediction
predictions = rf.predict(X_test)  # Already parallel with n_jobs=-1
```

**3. Categorical Variables:**
```python
# Problem: sklearn RF doesn't handle categories natively
# Solutions:

# a) One-hot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numerical_cols)
])

# b) Use category-native implementation
import lightgbm as lgb
lgb_model = lgb.LGBMClassifier()
# Specify categorical features directly
```

**4. Reproducibility:**
```python
# Problem: Results vary between runs
# Solution: Set random_state everywhere

import numpy as np
np.random.seed(42)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,  # Critical for reproducibility
    n_jobs=1          # n_jobs > 1 can affect reproducibility
)
```

**5. Handling Imbalanced Data:**
```python
# Problem: Majority class dominates
# Solutions:

# a) Class weights
rf = RandomForestClassifier(class_weight='balanced')

# b) Balanced Random Forest
from imblearn.ensemble import BalancedRandomForestClassifier
brf = BalancedRandomForestClassifier(n_estimators=100)
```

**6. Feature Importance Bias:**
```python
# Problem: Gini importance biased toward high-cardinality features
# Solution: Use permutation importance

from sklearn.inspection import permutation_importance

perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10)
# Use perm_imp.importances_mean instead of rf.feature_importances_
```

---

## Question 3
- [ ] Done

**Write a Python code to train a Random Forest Classifier using scikit-learn on a given dataset.**

### Answer

**Complete Training Pipeline:**

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score)
import matplotlib.pyplot as plt

# 1. Load and prepare data
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# 4. Evaluate
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print("\n=== Model Performance ===")
print(f"OOB Score: {rf.oob_score_:.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 5. Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"\nCV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 6. Feature importance
importance_df = pd.DataFrame({
    'feature': data.feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
print(importance_df.head(10))

# 7. Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

**Output:**
```
Dataset shape: (569, 30)
Class distribution: [212 357]

=== Model Performance ===
OOB Score: 0.9648
Test Accuracy: 0.9649
AUC-ROC: 0.9945

Classification Report:
              precision    recall  f1-score   support
   malignant       0.95      0.95      0.95        43
      benign       0.97      0.97      0.97        71
    accuracy                           0.96       114
```

---

## Question 4
- [ ] Done

**Create a function that computes the OOB error for a Random Forest model.**

### Answer

**Definition:**
OOB (Out-of-Bag) error is computed using samples not included in each tree's bootstrap sample. Each sample's prediction comes only from trees that didn't train on it.

**Implementation:**

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def compute_oob_error_manual(X, y, n_estimators=100, random_state=42):
    """
    Manually compute OOB error to understand the process.
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    
    # Store OOB predictions for each sample
    oob_predictions = np.zeros((n_samples, len(np.unique(y))))
    oob_counts = np.zeros(n_samples)
    
    from sklearn.tree import DecisionTreeClassifier
    
    for i in range(n_estimators):
        # Create bootstrap sample
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        oob_indices = list(set(range(n_samples)) - set(bootstrap_indices))
        
        # Train tree on bootstrap sample
        tree = DecisionTreeClassifier(random_state=random_state + i)
        tree.fit(X[bootstrap_indices], y[bootstrap_indices])
        
        # Predict on OOB samples
        if len(oob_indices) > 0:
            oob_proba = tree.predict_proba(X[oob_indices])
            
            # Handle case where tree hasn't seen all classes
            for j, class_label in enumerate(tree.classes_):
                oob_predictions[oob_indices, class_label] += oob_proba[:, j]
            oob_counts[oob_indices] += 1
    
    # Compute OOB predictions (majority vote)
    valid_samples = oob_counts > 0
    oob_predictions_final = np.argmax(oob_predictions[valid_samples], axis=1)
    
    # Compute OOB error
    oob_error = 1 - np.mean(oob_predictions_final == y[valid_samples])
    
    return oob_error, oob_counts

def compute_oob_error_sklearn(X, y, n_estimators=100, random_state=42):
    """
    Compute OOB error using sklearn's built-in functionality.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        oob_score=True,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X, y)
    
    oob_error = 1 - rf.oob_score_
    oob_decision = rf.oob_decision_function_  # Probability for each class
    
    return oob_error, oob_decision

def plot_oob_error_vs_trees(X, y, max_trees=300, step=20):
    """
    Plot OOB error as a function of number of trees.
    """
    import matplotlib.pyplot as plt
    
    tree_counts = list(range(step, max_trees + 1, step))
    oob_errors = []
    
    for n_trees in tree_counts:
        rf = RandomForestClassifier(n_estimators=n_trees, oob_score=True, random_state=42)
        rf.fit(X, y)
        oob_errors.append(1 - rf.oob_score_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(tree_counts, oob_errors, 'b-', marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('OOB Error')
    plt.title('OOB Error vs Number of Trees')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return tree_counts, oob_errors


# Example usage
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=10, random_state=42)

# Method 1: Manual computation
oob_error_manual, oob_counts = compute_oob_error_manual(X, y)
print(f"Manual OOB Error: {oob_error_manual:.4f}")
print(f"Avg OOB count per sample: {oob_counts.mean():.1f}")

# Method 2: sklearn built-in
oob_error_sklearn, _ = compute_oob_error_sklearn(X, y)
print(f"Sklearn OOB Error: {oob_error_sklearn:.4f}")

# Plot OOB error vs trees
# tree_counts, errors = plot_oob_error_vs_trees(X, y)
```

**Key Points:**
- ~36.8% of samples are OOB for each tree
- OOB error ≈ Leave-One-Out CV error
- Use `oob_score=True` in sklearn for automatic computation

---

## Question 5
- [ ] Done

**Write Python code that selects the most important features using a trained Random Forest model.**

### Answer

**Multiple Feature Selection Methods:**

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=30, 
                           n_informative=10, n_redundant=5, random_state=42)
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)


# Method 1: Gini/MDI Importance (Built-in)
def select_features_mdi(rf, feature_names, top_k=10):
    """Select top features using Mean Decrease in Impurity."""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    selected = importance_df.head(top_k)['feature'].tolist()
    return selected, importance_df

selected_mdi, importance_mdi = select_features_mdi(rf, feature_names, top_k=10)
print("Top 10 features (MDI):", selected_mdi[:5])


# Method 2: Permutation Importance (Recommended)
def select_features_permutation(rf, X_test, y_test, feature_names, top_k=10):
    """Select top features using permutation importance."""
    perm_imp = permutation_importance(rf, X_test, y_test, 
                                       n_repeats=10, random_state=42, n_jobs=-1)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_imp.importances_mean,
        'std': perm_imp.importances_std
    }).sort_values('importance', ascending=False)
    
    selected = importance_df.head(top_k)['feature'].tolist()
    return selected, importance_df

selected_perm, importance_perm = select_features_permutation(
    rf, X_test, y_test, feature_names, top_k=10
)
print("Top 10 features (Permutation):", selected_perm[:5])


# Method 3: SelectFromModel (Threshold-based)
def select_features_threshold(rf, X_train, feature_names, threshold='median'):
    """Select features above importance threshold."""
    selector = SelectFromModel(rf, threshold=threshold, prefit=True)
    X_selected = selector.transform(X_train)
    
    selected_mask = selector.get_support()
    selected_features = [f for f, s in zip(feature_names, selected_mask) if s]
    
    return selected_features, X_selected

selected_thresh, X_selected = select_features_threshold(rf, X_train, feature_names)
print(f"Features above median: {len(selected_thresh)}")


# Method 4: Recursive Feature Elimination
def select_features_rfe(X_train, y_train, feature_names, n_features=10):
    """Select features using RFE with RF."""
    rf_rfe = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rf_rfe, n_features_to_select=n_features, step=1)
    rfe.fit(X_train, y_train)
    
    selected_mask = rfe.support_
    selected_features = [f for f, s in zip(feature_names, selected_mask) if s]
    rankings = dict(zip(feature_names, rfe.ranking_))
    
    return selected_features, rankings

selected_rfe, rankings = select_features_rfe(X_train, y_train, feature_names, n_features=10)
print("Top 10 features (RFE):", selected_rfe[:5])


# Compare feature importance methods
def plot_importance_comparison(importance_mdi, importance_perm, top_k=15):
    """Compare MDI vs Permutation importance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # MDI
    top_mdi = importance_mdi.head(top_k)
    axes[0].barh(top_mdi['feature'], top_mdi['importance'])
    axes[0].set_xlabel('Importance')
    axes[0].set_title('MDI (Gini) Importance')
    axes[0].invert_yaxis()
    
    # Permutation
    top_perm = importance_perm.head(top_k)
    axes[1].barh(top_perm['feature'], top_perm['importance'])
    axes[1].errorbar(top_perm['importance'], range(len(top_perm)), 
                     xerr=top_perm['std'], fmt='none', c='black', capsize=3)
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Permutation Importance')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()

# plot_importance_comparison(importance_mdi, importance_perm)


# Evaluate model with selected features
def evaluate_feature_selection(X_train, X_test, y_train, y_test, 
                               selected_features, feature_names):
    """Train model on selected features and compare."""
    # Get indices of selected features
    selected_idx = [feature_names.index(f) for f in selected_features]
    
    # Train on selected features
    rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selected.fit(X_train[:, selected_idx], y_train)
    
    # Train on all features
    rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_all.fit(X_train, y_train)
    
    print(f"\nAccuracy with all features: {rf_all.score(X_test, y_test):.4f}")
    print(f"Accuracy with {len(selected_features)} features: "
          f"{rf_selected.score(X_test[:, selected_idx], y_test):.4f}")

evaluate_feature_selection(X_train, X_test, y_train, y_test, selected_perm, feature_names)
```

---

## Question 6
- [ ] Done

**Implement from scratch a simplified version of the Random Forest algorithm in Python.**

### Answer

**Complete Implementation:**

```python
import numpy as np
from collections import Counter

class DecisionTreeFromScratch:
    """Simplified Decision Tree for use in Random Forest."""
    
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None
    
    def _gini(self, y):
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _split(self, X, y, feature, threshold):
        """Split data based on feature and threshold."""
        left_mask = X[:, feature] <= threshold
        return X[left_mask], X[~left_mask], y[left_mask], y[~left_mask]
    
    def _best_split(self, X, y):
        """Find best split considering random feature subset."""
        n_samples, n_features = X.shape
        
        if n_samples < self.min_samples_split:
            return None, None
        
        # Random feature subset (key RF component)
        if self.max_features:
            feature_indices = np.random.choice(
                n_features, size=min(self.max_features, n_features), replace=False
            )
        else:
            feature_indices = np.arange(n_features)
        
        best_gain = -1
        best_feature, best_threshold = None, None
        current_gini = self._gini(y)
        
        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                _, _, y_left, y_right = self._split(X, y, feature, threshold)
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                # Calculate information gain
                n_left, n_right = len(y_left), len(y_right)
                weighted_gini = (n_left * self._gini(y_left) + 
                                n_right * self._gini(y_right)) / n_samples
                gain = current_gini - weighted_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the tree."""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1 or n_samples < self.min_samples_split:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}
        
        # Find best split
        feature, threshold = self._best_split(X, y)
        
        if feature is None:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}
        
        # Split and recurse
        X_left, X_right, y_left, y_right = self._split(X, y, feature, threshold)
        
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X_left, y_left, depth + 1),
            'right': self._build_tree(X_right, y_right, depth + 1)
        }
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, node):
        if node['leaf']:
            return node['class']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])


class RandomForestFromScratch:
    """Random Forest Classifier from scratch."""
    
    def __init__(self, n_estimators=100, max_depth=None, 
                 min_samples_split=2, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.oob_score_ = None
    
    def _bootstrap_sample(self, X, y):
        """Create bootstrap sample and identify OOB indices."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        oob_indices = list(set(range(n_samples)) - set(indices))
        return X[indices], y[indices], oob_indices
    
    def _get_max_features(self, n_features):
        """Convert max_features parameter to integer."""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        elif isinstance(self.max_features, int):
            return self.max_features
        else:
            return n_features
    
    def fit(self, X, y):
        if self.random_state:
            np.random.seed(self.random_state)
        
        self.trees = []
        n_features = X.shape[1]
        max_features = self._get_max_features(n_features)
        
        # For OOB score calculation
        n_samples = X.shape[0]
        oob_predictions = np.zeros((n_samples, len(np.unique(y))))
        oob_counts = np.zeros(n_samples)
        
        for i in range(self.n_estimators):
            # Bootstrap sample
            X_boot, y_boot, oob_indices = self._bootstrap_sample(X, y)
            
            # Train tree
            tree = DecisionTreeFromScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
            # OOB predictions
            if len(oob_indices) > 0:
                oob_pred = tree.predict(X[oob_indices])
                for j, idx in enumerate(oob_indices):
                    oob_predictions[idx, oob_pred[j]] += 1
                    oob_counts[idx] += 1
        
        # Calculate OOB score
        valid = oob_counts > 0
        if np.any(valid):
            oob_final = np.argmax(oob_predictions[valid], axis=1)
            self.oob_score_ = np.mean(oob_final == y[valid])
        
        return self
    
    def predict(self, X):
        """Predict using majority vote."""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        final_predictions = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            most_common = Counter(votes).most_common(1)[0][0]
            final_predictions.append(most_common)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        n_samples = X.shape[0]
        n_classes = len(np.unique(predictions))
        
        probas = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            votes = predictions[:, i]
            for cls in votes:
                probas[i, cls] += 1
            probas[i] /= len(self.trees)
        
        return probas


# Test implementation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train custom Random Forest
rf_custom = RandomForestFromScratch(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',
    random_state=42
)
rf_custom.fit(X_train, y_train)
y_pred_custom = rf_custom.predict(X_test)

print(f"Custom RF Accuracy: {accuracy_score(y_test, y_pred_custom):.4f}")
print(f"Custom RF OOB Score: {rf_custom.oob_score_:.4f}")

# Compare with sklearn
from sklearn.ensemble import RandomForestClassifier
rf_sklearn = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                     random_state=42, oob_score=True)
rf_sklearn.fit(X_train, y_train)
y_pred_sklearn = rf_sklearn.predict(X_test)

print(f"Sklearn RF Accuracy: {accuracy_score(y_test, y_pred_sklearn):.4f}")
print(f"Sklearn RF OOB Score: {rf_sklearn.oob_score_:.4f}")
```

---

## Question 7
- [ ] Done

**Write a function to visualize an individual decision tree from a Random Forest in Python.**

### Answer

**Multiple Visualization Methods:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree, export_text, export_graphviz
from sklearn.datasets import load_iris
import graphviz

# Train a Random Forest
iris = load_iris()
X, y = iris.data, iris.target

rf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
rf.fit(X, y)


# Method 1: Using matplotlib (plot_tree)
def visualize_tree_matplotlib(rf, tree_index=0, feature_names=None, 
                              class_names=None, figsize=(20, 10)):
    """Visualize a tree using matplotlib."""
    tree = rf.estimators_[tree_index]
    
    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(tree, 
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              ax=ax,
              fontsize=10)
    plt.title(f'Decision Tree #{tree_index + 1} from Random Forest')
    plt.tight_layout()
    plt.show()
    
    return fig

# Visualize first tree
fig = visualize_tree_matplotlib(rf, tree_index=0, 
                                feature_names=iris.feature_names,
                                class_names=iris.target_names)


# Method 2: Text representation
def visualize_tree_text(rf, tree_index=0, feature_names=None):
    """Get text representation of tree."""
    tree = rf.estimators_[tree_index]
    text_repr = export_text(tree, feature_names=feature_names)
    print(f"Tree #{tree_index + 1}:")
    print(text_repr)
    return text_repr

text = visualize_tree_text(rf, tree_index=0, feature_names=iris.feature_names)


# Method 3: Using Graphviz (higher quality)
def visualize_tree_graphviz(rf, tree_index=0, feature_names=None,
                            class_names=None, filename='tree'):
    """Visualize tree using Graphviz (requires graphviz installed)."""
    tree = rf.estimators_[tree_index]
    
    dot_data = export_graphviz(
        tree,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    
    graph = graphviz.Source(dot_data)
    graph.render(filename, format='png', cleanup=True)
    print(f"Tree saved to {filename}.png")
    
    return graph

# Uncomment to use (requires graphviz)
# graph = visualize_tree_graphviz(rf, tree_index=0, 
#                                  feature_names=iris.feature_names,
#                                  class_names=list(iris.target_names))


# Method 4: Interactive visualization with dtreeviz
def visualize_tree_dtreeviz(rf, X, y, tree_index=0, feature_names=None,
                            class_names=None):
    """Visualize tree using dtreeviz (requires dtreeviz installed)."""
    try:
        from dtreeviz.trees import dtreeviz
        
        tree = rf.estimators_[tree_index]
        viz = dtreeviz(tree, X, y,
                      feature_names=feature_names,
                      target_name='class',
                      class_names=class_names)
        return viz
    except ImportError:
        print("dtreeviz not installed. Install with: pip install dtreeviz")
        return None


# Function to compare multiple trees
def compare_trees(rf, tree_indices, feature_names=None, class_names=None):
    """Visualize multiple trees side by side."""
    n_trees = len(tree_indices)
    fig, axes = plt.subplots(1, n_trees, figsize=(8 * n_trees, 8))
    
    if n_trees == 1:
        axes = [axes]
    
    for ax, idx in zip(axes, tree_indices):
        tree = rf.estimators_[idx]
        plot_tree(tree,
                  feature_names=feature_names,
                  class_names=class_names,
                  filled=True,
                  rounded=True,
                  ax=ax,
                  fontsize=8)
        ax.set_title(f'Tree #{idx + 1}')
    
    plt.tight_layout()
    plt.show()

# Compare first 3 trees
compare_trees(rf, [0, 1, 2], 
              feature_names=iris.feature_names,
              class_names=list(iris.target_names))


# Function to analyze tree structure
def analyze_tree_structure(rf, tree_index=0):
    """Analyze structure of a tree in the forest."""
    tree = rf.estimators_[tree_index].tree_
    
    print(f"Tree #{tree_index + 1} Structure:")
    print(f"  - Total nodes: {tree.node_count}")
    print(f"  - Max depth: {tree.max_depth}")
    print(f"  - Number of leaves: {tree.n_leaves}")
    print(f"  - Number of features used: {len(np.unique(tree.feature[tree.feature >= 0]))}")
    
    # Features used
    features_used = tree.feature[tree.feature >= 0]
    feature_counts = np.bincount(features_used, minlength=tree.n_features)
    
    return {
        'node_count': tree.node_count,
        'max_depth': tree.max_depth,
        'n_leaves': tree.n_leaves,
        'feature_usage': feature_counts
    }

# Analyze first tree
structure = analyze_tree_structure(rf, tree_index=0)
```

**Output:**
```
Tree #1 Structure:
  - Total nodes: 15
  - Max depth: 4
  - Number of leaves: 8
  - Number of features used: 3
```

**Tips:**
- Use `max_depth=3-5` for readable visualizations
- For large trees, use text representation
- Graphviz produces highest quality images
- dtreeviz adds data distribution visualizations
