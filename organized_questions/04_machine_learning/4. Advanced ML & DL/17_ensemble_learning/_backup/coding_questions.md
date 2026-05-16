# Ensemble Learning Interview Questions - Coding Questions

## Question 1: Can you implement ensemble models with imbalanced datasets? If yes, how?
- [ ] Done

### Definition
Yes, ensemble methods can handle imbalanced data through class weighting, resampling techniques (SMOTE, undersampling), cost-sensitive learning, or specialized algorithms like BalancedRandomForest.

### Approach 1: Class Weights
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Create imbalanced data (1:10 ratio)
X, y = make_classification(n_samples=10000, weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest with balanced class weights
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Automatically adjusts weights
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Approach 2: SMOTE + Ensemble
```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Pipeline: SMOTE then Random Forest
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Approach 3: Balanced Random Forest
```python
from imblearn.ensemble import BalancedRandomForestClassifier

# Built-in balanced sampling
brf = BalancedRandomForestClassifier(
    n_estimators=100,
    sampling_strategy='all',  # Balance all classes
    random_state=42
)
brf.fit(X_train, y_train)

y_pred = brf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Key Points
- Use F1-score, precision-recall, not just accuracy
- `class_weight='balanced'` is simplest approach
- SMOTE creates synthetic minority samples
- BalancedRandomForest undersamples majority per tree

---

## Question 2: Implement a simple bagging classifier in Python using decision trees as base learners
- [ ] Done

### Algorithm Steps
1. Create B bootstrap samples from training data
2. Train a decision tree on each sample
3. For prediction: majority vote from all trees

### Implementation
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from collections import Counter

class SimpleBaggingClassifier:
    def __init__(self, n_estimators=10, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.trees = []
        
    def _create_bootstrap_sample(self, X, y):
        """Create bootstrap sample (sampling with replacement)"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        """Train ensemble on bootstrap samples"""
        np.random.seed(self.random_state)
        self.trees = []
        
        for _ in range(self.n_estimators):
            # Create bootstrap sample
            X_boot, y_boot = self._create_bootstrap_sample(X, y)
            
            # Train decision tree
            tree = DecisionTreeClassifier()
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """Predict using majority voting"""
        # Get predictions from all trees
        all_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Majority vote for each sample
        final_predictions = []
        for i in range(X.shape[0]):
            votes = all_predictions[:, i]
            majority = Counter(votes).most_common(1)[0][0]
            final_predictions.append(majority)
        
        return np.array(final_predictions)


# Test the implementation
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train bagging classifier
bagging = SimpleBaggingClassifier(n_estimators=50, random_state=42)
bagging.fit(X_train, y_train)

# Evaluate
accuracy = np.mean(bagging.predict(X_test) == y_test)
print(f"Bagging Accuracy: {accuracy:.4f}")

# Compare with single tree
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
single_accuracy = np.mean(single_tree.predict(X_test) == y_test)
print(f"Single Tree Accuracy: {single_accuracy:.4f}")
```

### Output
```
Bagging Accuracy: 0.9150
Single Tree Accuracy: 0.8500
```

---

## Question 3: Write a Python script to perform K-fold cross-validation on a Random Forest model
- [ ] Done

### Implementation
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, make_scorer

# Load data
data = load_iris()
X, y = data.data, data.target

# Create Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Method 1: Simple cross_val_score
scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')

print("Method 1: cross_val_score")
print(f"CV Scores: {scores}")
print(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Method 2: Manual K-Fold (more control)
print("\nMethod 2: Manual K-Fold")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

fold_scores = []
for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    score = accuracy_score(y_val, y_pred)
    fold_scores.append(score)
    
    print(f"Fold {fold+1}: Accuracy = {score:.4f}")

print(f"\nMean: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

# Method 3: Stratified K-Fold (preserves class distribution)
from sklearn.model_selection import StratifiedKFold

print("\nMethod 3: Stratified K-Fold")
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_scores = cross_val_score(rf, X, y, cv=stratified_kfold, scoring='accuracy')
print(f"Mean Accuracy: {stratified_scores.mean():.4f} (+/- {stratified_scores.std():.4f})")
```

### Key Points
- Use `StratifiedKFold` for classification (preserves class ratios)
- Set `shuffle=True` for random fold assignment
- Report mean ± std for proper comparison

---

## Question 4: Create a stacking ensemble of classifiers using scikit-learn and evaluate its performance
- [ ] Done

### Implementation
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base learners (Level 0)
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ('svc', SVC(kernel='rbf', probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# Define meta-learner (Level 1)
meta_learner = LogisticRegression()

# Create stacking classifier
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,  # Cross-validation for generating meta-features
    stack_method='auto'  # Uses predict_proba if available
)

# Train stacking ensemble
stacking_clf.fit(X_train, y_train)

# Evaluate
y_pred = stacking_clf.predict(X_test)
print("Stacking Ensemble Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Compare with individual models
print("\nIndividual Model Performance:")
for name, model in base_learners:
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: {acc:.4f}")

# Cross-validation score
cv_scores = cross_val_score(stacking_clf, X_train, y_train, cv=5)
print(f"\nStacking CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

### Key Points
- Use diverse base learners (different algorithms)
- Simple meta-learner (Logistic Regression) to avoid overfitting
- `cv=5` generates out-of-fold predictions for level-1 training

---

## Question 5: Code a Boosting algorithm from scratch using Python
- [ ] Done

### AdaBoost Implementation
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class SimpleAdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.stump_weights = []
        
    def fit(self, X, y):
        """Train AdaBoost on data"""
        n_samples = X.shape[0]
        
        # Convert labels to {-1, +1}
        y_converted = np.where(y == 0, -1, 1)
        
        # Initialize sample weights uniformly
        sample_weights = np.ones(n_samples) / n_samples
        
        self.stumps = []
        self.stump_weights = []
        
        for _ in range(self.n_estimators):
            # Train weak learner (decision stump)
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y_converted, sample_weight=sample_weights)
            
            # Get predictions
            predictions = stump.predict(X)
            
            # Calculate weighted error
            incorrect = predictions != y_converted
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)
            
            # Avoid division by zero
            error = np.clip(error, 1e-10, 1 - 1e-10)
            
            # Calculate stump weight (alpha)
            alpha = 0.5 * np.log((1 - error) / error)
            
            # Update sample weights
            sample_weights *= np.exp(-alpha * y_converted * predictions)
            sample_weights /= np.sum(sample_weights)  # Normalize
            
            # Store stump and its weight
            self.stumps.append(stump)
            self.stump_weights.append(alpha)
        
        return self
    
    def predict(self, X):
        """Predict using weighted vote of all stumps"""
        # Get weighted predictions from all stumps
        stump_predictions = np.array([
            alpha * stump.predict(X) 
            for stump, alpha in zip(self.stumps, self.stump_weights)
        ])
        
        # Sum weighted predictions and take sign
        final_predictions = np.sign(np.sum(stump_predictions, axis=0))
        
        # Convert back to {0, 1}
        return np.where(final_predictions == -1, 0, 1)


# Test implementation
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train AdaBoost
adaboost = SimpleAdaBoost(n_estimators=50)
adaboost.fit(X_train, y_train)

# Evaluate
accuracy = np.mean(adaboost.predict(X_test) == y_test)
print(f"AdaBoost from scratch: {accuracy:.4f}")

# Compare with sklearn
from sklearn.ensemble import AdaBoostClassifier
sklearn_ada = AdaBoostClassifier(n_estimators=50, random_state=42)
sklearn_ada.fit(X_train, y_train)
sklearn_acc = np.mean(sklearn_ada.predict(X_test) == y_test)
print(f"Sklearn AdaBoost: {sklearn_acc:.4f}")
```

---

## Question 6: Use XGBoost in Python to train and fine-tune a model on a given dataset
- [ ] Done

### Implementation
```python
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint, uniform

# Create dataset
X, y = make_classification(n_samples=2000, n_features=20, n_informative=15,
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Step 1: Basic XGBoost model
print("Step 1: Basic Model")
basic_model = xgb.XGBClassifier(
    n_estimators=100,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
basic_model.fit(X_train, y_train)
print(f"Basic Accuracy: {accuracy_score(y_test, basic_model.predict(X_test)):.4f}")

# Step 2: XGBoost with early stopping
print("\nStep 2: With Early Stopping")
early_stop_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=50
)
early_stop_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
print(f"Best iteration: {early_stop_model.best_iteration}")
print(f"Accuracy: {accuracy_score(y_test, early_stop_model.predict(X_test)):.4f}")

# Step 3: Hyperparameter tuning with RandomizedSearchCV
print("\nStep 3: Hyperparameter Tuning")
param_distributions = {
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'n_estimators': randint(50, 300),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 2)
}

random_search = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    param_distributions=param_distributions,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)

print(f"Best params: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, random_search.predict(X_test)):.4f}")

# Step 4: Final model with best parameters
print("\nStep 4: Final Tuned Model")
final_model = xgb.XGBClassifier(
    **random_search.best_params_,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
final_model.fit(X_train, y_train)
print(f"Final Test Accuracy: {accuracy_score(y_test, final_model.predict(X_test)):.4f}")

# Feature importance
print("\nTop 5 Important Features:")
importance = final_model.feature_importances_
for i in np.argsort(importance)[-5:][::-1]:
    print(f"  Feature {i}: {importance[i]:.4f}")
```

---

## Question 7: Implement feature bagging in Python to see its effect on a classification problem
- [ ] Done

### Implementation
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from collections import Counter

class FeatureBaggingClassifier:
    def __init__(self, n_estimators=50, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []
        
    def _get_n_features(self, n_total_features):
        """Determine number of features to sample"""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_total_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_total_features))
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_total_features)
        elif isinstance(self.max_features, int):
            return self.max_features
        return n_total_features
    
    def fit(self, X, y):
        """Train trees on random feature subsets"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        n_select = self._get_n_features(n_features)
        
        self.trees = []
        self.feature_indices = []
        
        for _ in range(self.n_estimators):
            # Random feature selection
            selected_features = np.random.choice(n_features, size=n_select, replace=False)
            self.feature_indices.append(selected_features)
            
            # Train tree on selected features
            tree = DecisionTreeClassifier()
            tree.fit(X[:, selected_features], y)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """Predict using majority voting"""
        all_predictions = []
        
        for tree, features in zip(self.trees, self.feature_indices):
            pred = tree.predict(X[:, features])
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        # Majority vote
        final_predictions = []
        for i in range(X.shape[0]):
            votes = all_predictions[:, i]
            majority = Counter(votes).most_common(1)[0][0]
            final_predictions.append(majority)
        
        return np.array(final_predictions)


# Test with high-dimensional data
X, y = make_classification(n_samples=1000, n_features=50, n_informative=10,
                           n_redundant=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compare: Single tree vs Feature bagging
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
single_acc = np.mean(single_tree.predict(X_test) == y_test)

feature_bag = FeatureBaggingClassifier(n_estimators=50, max_features='sqrt', random_state=42)
feature_bag.fit(X_train, y_train)
bag_acc = np.mean(feature_bag.predict(X_test) == y_test)

print(f"Single Tree Accuracy: {single_acc:.4f}")
print(f"Feature Bagging Accuracy: {bag_acc:.4f}")
print(f"Improvement: {(bag_acc - single_acc) * 100:.2f}%")
```

### Key Points
- Feature bagging helps with high-dimensional data
- `sqrt` features per tree is common for classification
- Creates diversity even without bootstrap sampling

---

## Question 8: Develop a voting ensemble classifier in Python with different weighting strategies for base learners
- [ ] Done

### Implementation
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base classifiers
clf1 = LogisticRegression(max_iter=1000, random_state=42)
clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
clf3 = GradientBoostingClassifier(n_estimators=50, random_state=42)
clf4 = SVC(kernel='rbf', probability=True, random_state=42)

# Strategy 1: Hard Voting (Majority)
print("Strategy 1: Hard Voting")
hard_voting = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3), ('svc', clf4)],
    voting='hard'
)
hard_voting.fit(X_train, y_train)
print(f"Accuracy: {accuracy_score(y_test, hard_voting.predict(X_test)):.4f}")

# Strategy 2: Soft Voting (Average Probabilities)
print("\nStrategy 2: Soft Voting")
soft_voting = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3), ('svc', clf4)],
    voting='soft'
)
soft_voting.fit(X_train, y_train)
print(f"Accuracy: {accuracy_score(y_test, soft_voting.predict(X_test)):.4f}")

# Strategy 3: Weighted Voting (Based on CV performance)
print("\nStrategy 3: Weighted Voting (CV-based weights)")

# Calculate CV scores for each model
models = [('lr', clf1), ('rf', clf2), ('gb', clf3), ('svc', clf4)]
cv_scores = []
for name, model in models:
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    cv_scores.append(score)
    print(f"  {name} CV Score: {score:.4f}")

# Normalize weights
weights = [s / sum(cv_scores) for s in cv_scores]
print(f"  Weights: {[f'{w:.3f}' for w in weights]}")

weighted_voting = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3), ('svc', clf4)],
    voting='soft',
    weights=weights
)
weighted_voting.fit(X_train, y_train)
print(f"Weighted Accuracy: {accuracy_score(y_test, weighted_voting.predict(X_test)):.4f}")

# Strategy 4: Custom weights (emphasize best performers)
print("\nStrategy 4: Custom Weights [1, 2, 3, 2]")
custom_voting = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3), ('svc', clf4)],
    voting='soft',
    weights=[1, 2, 3, 2]  # Higher weight for Gradient Boosting
)
custom_voting.fit(X_train, y_train)
print(f"Accuracy: {accuracy_score(y_test, custom_voting.predict(X_test)):.4f}")

# Compare all strategies
print("\n--- Summary ---")
print(f"Hard Voting:     {accuracy_score(y_test, hard_voting.predict(X_test)):.4f}")
print(f"Soft Voting:     {accuracy_score(y_test, soft_voting.predict(X_test)):.4f}")
print(f"Weighted (CV):   {accuracy_score(y_test, weighted_voting.predict(X_test)):.4f}")
print(f"Custom Weights:  {accuracy_score(y_test, custom_voting.predict(X_test)):.4f}")
```

---

## Question 9: Simulate overfitting in an ensemble model and implement a method to reduce it
- [ ] Done

### Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create dataset with noise to encourage overfitting
X, y = make_classification(n_samples=500, n_features=20, n_informative=5,
                           n_redundant=10, n_clusters_per_class=2,
                           flip_y=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Part 1: Create an overfitting ensemble
print("Part 1: Overfitting Model")
overfit_model = GradientBoostingClassifier(
    n_estimators=500,       # Too many trees
    max_depth=10,           # Too deep
    learning_rate=1.0,      # Too high learning rate
    min_samples_leaf=1,     # No regularization
    random_state=42
)
overfit_model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, overfit_model.predict(X_train))
test_acc = accuracy_score(y_test, overfit_model.predict(X_test))
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Overfitting Gap: {train_acc - test_acc:.4f}")

# Part 2: Reduce overfitting with regularization
print("\nPart 2: Regularized Model")
regularized_model = GradientBoostingClassifier(
    n_estimators=100,       # Fewer trees
    max_depth=3,            # Shallow trees
    learning_rate=0.1,      # Lower learning rate
    min_samples_leaf=5,     # More samples per leaf
    subsample=0.8,          # Row subsampling
    random_state=42
)
regularized_model.fit(X_train, y_train)

train_acc_reg = accuracy_score(y_train, regularized_model.predict(X_train))
test_acc_reg = accuracy_score(y_test, regularized_model.predict(X_test))
print(f"Training Accuracy: {train_acc_reg:.4f}")
print(f"Test Accuracy: {test_acc_reg:.4f}")
print(f"Overfitting Gap: {train_acc_reg - test_acc_reg:.4f}")

# Part 3: Early Stopping
print("\nPart 3: Early Stopping")
X_train_es, X_val, y_train_es, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Track validation error during training
train_errors = []
val_errors = []
n_est_range = range(1, 301, 10)

for n_est in n_est_range:
    model = GradientBoostingClassifier(
        n_estimators=n_est,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train_es, y_train_es)
    train_errors.append(1 - accuracy_score(y_train_es, model.predict(X_train_es)))
    val_errors.append(1 - accuracy_score(y_val, model.predict(X_val)))

# Find best n_estimators
best_idx = np.argmin(val_errors)
best_n = list(n_est_range)[best_idx]
print(f"Best n_estimators: {best_n}")

# Train final model with early stopping point
final_model = GradientBoostingClassifier(
    n_estimators=best_n,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
final_model.fit(X_train, y_train)
print(f"Test Accuracy with Early Stopping: {accuracy_score(y_test, final_model.predict(X_test)):.4f}")

# Summary
print("\n--- Summary ---")
print(f"Overfit Model Test Accuracy: {test_acc:.4f}")
print(f"Regularized Model Test Accuracy: {test_acc_reg:.4f}")
print(f"Early Stopping Test Accuracy: {accuracy_score(y_test, final_model.predict(X_test)):.4f}")
```

---

## Question 10: Demonstrate the use of out-of-bag samples to estimate model accuracy in Random Forest using Python
- [ ] Done

### Implementation
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Create dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest with OOB scoring enabled
rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,  # Enable OOB scoring
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Get OOB score
oob_accuracy = rf.oob_score_
print(f"OOB Accuracy: {oob_accuracy:.4f}")

# Compare with actual test accuracy
test_accuracy = accuracy_score(y_test, rf.predict(X_test))
print(f"Test Accuracy: {test_accuracy:.4f}")

# Compare with cross-validation
cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train, y_train, cv=5
)
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# OOB predictions per sample
print("\nOOB Details:")
print(f"OOB Decision Function Shape: {rf.oob_decision_function_.shape}")
print(f"Each training sample has probability estimates from trees that didn't train on it")

# Manual verification of OOB concept
print("\n--- OOB Concept Verification ---")
# For each bootstrap sample, ~37% of data is left out
# These left-out samples are used for that tree's OOB prediction

n_samples = X_train.shape[0]
n_trees = rf.n_estimators

# Count how many trees each sample is OOB for
oob_counts = np.zeros(n_samples)
for i, tree in enumerate(rf.estimators_):
    # Get indices used in bootstrap sample for this tree
    # (Random Forest doesn't expose this directly, so this is conceptual)
    pass

print(f"On average, each sample is OOB for ~{0.368 * n_trees:.0f} trees")
print(f"This is enough to get a reliable prediction estimate")

# Compare OOB error at different n_estimators
print("\n--- OOB vs n_estimators ---")
for n_trees in [10, 50, 100, 200, 500]:
    rf_temp = RandomForestClassifier(n_estimators=n_trees, oob_score=True, random_state=42)
    rf_temp.fit(X_train, y_train)
    print(f"n_estimators={n_trees:3d}: OOB={rf_temp.oob_score_:.4f}, Test={accuracy_score(y_test, rf_temp.predict(X_test)):.4f}")
```

### Key Points
- OOB score is a "free" validation estimate
- Typically very close to CV or test accuracy
- ~37% of data is OOB for each tree
- More trees → more reliable OOB estimates

---

## Question 11: Write a Python routine to identify the least important features in a Gradient Boosting model
- [ ] Done

### Implementation
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Create dataset with informative and noisy features
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=5,      # Only 5 actually useful
    n_redundant=5,        # 5 redundant (correlated with informative)
    n_repeated=0,
    n_clusters_per_class=2,
    random_state=42
)

# Create feature names
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)

# Method 1: Built-in Feature Importance (Gini/Impurity-based)
print("Method 1: Built-in Feature Importance")
feature_importance_builtin = pd.DataFrame({
    'feature': feature_names,
    'importance': gb.feature_importances_
}).sort_values('importance', ascending=True)

print("\nLeast Important Features (Built-in):")
print(feature_importance_builtin.head(5).to_string(index=False))

# Method 2: Permutation Importance (More Reliable)
print("\nMethod 2: Permutation Importance")
perm_importance = permutation_importance(
    gb, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

feature_importance_perm = pd.DataFrame({
    'feature': feature_names,
    'importance': perm_importance.importances_mean,
    'std': perm_importance.importances_std
}).sort_values('importance', ascending=True)

print("\nLeast Important Features (Permutation):")
print(feature_importance_perm.head(5).to_string(index=False))

# Method 3: Identify features with near-zero or negative importance
print("\nMethod 3: Features to Consider Removing")
threshold = 0.01

# From built-in
weak_builtin = feature_importance_builtin[
    feature_importance_builtin['importance'] < threshold
]['feature'].tolist()
print(f"Near-zero importance (built-in): {weak_builtin}")

# From permutation (negative = removing helps)
weak_perm = feature_importance_perm[
    feature_importance_perm['importance'] <= 0
]['feature'].tolist()
print(f"Negative/zero importance (permutation): {weak_perm}")

# Function to identify least important features
def get_least_important_features(model, X, y, feature_names, n_least=5, method='permutation'):
    """
    Identify least important features in a model
    
    Parameters:
    -----------
    model: fitted sklearn model
    X, y: test data
    feature_names: list of feature names
    n_least: number of least important features to return
    method: 'builtin' or 'permutation'
    
    Returns:
    --------
    DataFrame with least important features
    """
    if method == 'builtin':
        importance = model.feature_importances_
        std = np.zeros_like(importance)
    else:
        perm_imp = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        importance = perm_imp.importances_mean
        std = perm_imp.importances_std
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'std': std
    }).sort_values('importance', ascending=True)
    
    return df.head(n_least)

# Use the function
print("\n--- Using Helper Function ---")
least_important = get_least_important_features(gb, X_test, y_test, feature_names, n_least=5)
print(least_important.to_string(index=False))
```

---

## Question 12: Implement ensemble learning to improve accuracy on a multi-class classification problem
- [ ] Done

### Implementation
```python
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load multi-class dataset
digits = load_digits()  # 10 classes (digits 0-9)
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

# Individual models
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

print("\n--- Individual Model Performance ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: {acc:.4f}")

# Ensemble 1: Voting Classifier
print("\n--- Ensemble: Soft Voting ---")
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=5000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
voting_acc = accuracy_score(y_test, voting_clf.predict(X_test))
print(f"Voting Ensemble Accuracy: {voting_acc:.4f}")

# Ensemble 2: Stacking Classifier
print("\n--- Ensemble: Stacking ---")
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ],
    final_estimator=LogisticRegression(max_iter=5000),
    cv=5
)
stacking_clf.fit(X_train, y_train)
stacking_acc = accuracy_score(y_test, stacking_clf.predict(X_test))
print(f"Stacking Ensemble Accuracy: {stacking_acc:.4f}")

# Best individual model
best_individual = max([(name, accuracy_score(y_test, model.predict(X_test))) 
                       for name, model in models.items()], key=lambda x: x[1])

print("\n--- Summary ---")
print(f"Best Individual Model: {best_individual[0]} ({best_individual[1]:.4f})")
print(f"Voting Ensemble: {voting_acc:.4f}")
print(f"Stacking Ensemble: {stacking_acc:.4f}")
print(f"Improvement over best individual: {max(voting_acc, stacking_acc) - best_individual[1]:.4f}")

# Classification report for best ensemble
print("\n--- Classification Report (Best Ensemble) ---")
best_ensemble = voting_clf if voting_acc > stacking_acc else stacking_clf
print(classification_report(y_test, best_ensemble.predict(X_test)))
```

---

## Question 13: Using scikit-learn, compare the performance of a single decision tree and a Random Forest on the same dataset
- [ ] Done

### Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

# Create dataset
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("="*60)
print("Comparison: Decision Tree vs Random Forest")
print("="*60)

# Decision Tree
print("\n--- Single Decision Tree ---")
dt = DecisionTreeClassifier(random_state=42)

start_time = time.time()
dt.fit(X_train, y_train)
dt_train_time = time.time() - start_time

dt_train_acc = accuracy_score(y_train, dt.predict(X_train))
dt_test_acc = accuracy_score(y_test, dt.predict(X_test))
dt_cv_scores = cross_val_score(dt, X_train, y_train, cv=5)

print(f"Training Time: {dt_train_time:.4f}s")
print(f"Training Accuracy: {dt_train_acc:.4f}")
print(f"Test Accuracy: {dt_test_acc:.4f}")
print(f"CV Accuracy: {dt_cv_scores.mean():.4f} (+/- {dt_cv_scores.std():.4f})")
print(f"Overfitting Gap: {dt_train_acc - dt_test_acc:.4f}")

# Random Forest
print("\n--- Random Forest (100 trees) ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

start_time = time.time()
rf.fit(X_train, y_train)
rf_train_time = time.time() - start_time

rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))
rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=5)

print(f"Training Time: {rf_train_time:.4f}s")
print(f"Training Accuracy: {rf_train_acc:.4f}")
print(f"Test Accuracy: {rf_test_acc:.4f}")
print(f"CV Accuracy: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")
print(f"Overfitting Gap: {rf_train_acc - rf_test_acc:.4f}")

# Summary comparison
print("\n" + "="*60)
print("Summary Comparison")
print("="*60)
print(f"{'Metric':<25} {'Decision Tree':<15} {'Random Forest':<15}")
print("-"*55)
print(f"{'Training Time (s)':<25} {dt_train_time:<15.4f} {rf_train_time:<15.4f}")
print(f"{'Training Accuracy':<25} {dt_train_acc:<15.4f} {rf_train_acc:<15.4f}")
print(f"{'Test Accuracy':<25} {dt_test_acc:<15.4f} {rf_test_acc:<15.4f}")
print(f"{'CV Accuracy':<25} {dt_cv_scores.mean():<15.4f} {rf_cv_scores.mean():<15.4f}")
print(f"{'CV Std':<25} {dt_cv_scores.std():<15.4f} {rf_cv_scores.std():<15.4f}")
print(f"{'Overfitting Gap':<25} {dt_train_acc - dt_test_acc:<15.4f} {rf_train_acc - rf_test_acc:<15.4f}")

# Key observations
print("\n--- Key Observations ---")
print(f"1. Random Forest improves test accuracy by: {rf_test_acc - dt_test_acc:.4f}")
print(f"2. Random Forest reduces variance (lower CV std)")
print(f"3. Random Forest has smaller overfitting gap")
print(f"4. Trade-off: Random Forest takes {rf_train_time/dt_train_time:.1f}x longer to train")
```

---

## Question 14: Build an ensemble model that combines predictions from a neural network and a boosting classifier
- [ ] Done

### Implementation
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

# Create dataset
X, y = make_classification(n_samples=2000, n_features=20, n_informative=15,
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale for neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Neural Network (MLP)
print("Training Neural Network...")
nn = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    max_iter=500,
    random_state=42
)
nn.fit(X_train_scaled, y_train)
nn_acc = accuracy_score(y_test, nn.predict(X_test_scaled))
print(f"Neural Network Accuracy: {nn_acc:.4f}")

# Model 2: Gradient Boosting
print("\nTraining Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)  # GBM doesn't need scaling
gb_acc = accuracy_score(y_test, gb.predict(X_test))
print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")

# Ensemble: Combine NN and GBM
class NNBoostingEnsemble:
    def __init__(self, nn_model, gb_model, scaler, weights=(0.5, 0.5)):
        self.nn = nn_model
        self.gb = gb_model
        self.scaler = scaler
        self.weights = weights
    
    def predict_proba(self, X):
        # NN needs scaled input
        X_scaled = self.scaler.transform(X)
        nn_proba = self.nn.predict_proba(X_scaled)
        gb_proba = self.gb.predict_proba(X)
        
        # Weighted average
        ensemble_proba = (self.weights[0] * nn_proba + 
                         self.weights[1] * gb_proba)
        return ensemble_proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# Create ensemble with equal weights
ensemble_equal = NNBoostingEnsemble(nn, gb, scaler, weights=(0.5, 0.5))
ensemble_equal_acc = accuracy_score(y_test, ensemble_equal.predict(X_test))
print(f"\nEnsemble (50-50) Accuracy: {ensemble_equal_acc:.4f}")

# Try different weight combinations
print("\n--- Weight Tuning ---")
best_acc = 0
best_weights = None

for w_nn in np.arange(0.1, 1.0, 0.1):
    w_gb = 1 - w_nn
    ensemble = NNBoostingEnsemble(nn, gb, scaler, weights=(w_nn, w_gb))
    acc = accuracy_score(y_test, ensemble.predict(X_test))
    print(f"NN={w_nn:.1f}, GB={w_gb:.1f}: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_weights = (w_nn, w_gb)

print(f"\nBest Weights: NN={best_weights[0]:.1f}, GB={best_weights[1]:.1f}")
print(f"Best Ensemble Accuracy: {best_acc:.4f}")

# Final comparison
print("\n--- Summary ---")
print(f"Neural Network alone: {nn_acc:.4f}")
print(f"Gradient Boosting alone: {gb_acc:.4f}")
print(f"Best Ensemble: {best_acc:.4f}")
print(f"Improvement over best individual: {best_acc - max(nn_acc, gb_acc):.4f}")
```

---

## Question 15: Create a weighted ensemble that dynamically adjusts weights based on the performance of each learner
- [ ] Done

### Implementation
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class DynamicWeightedEnsemble:
    def __init__(self, models, window_size=100):
        """
        Ensemble with dynamically adjusted weights based on recent performance
        
        Parameters:
        -----------
        models: dict of {name: model} 
        window_size: number of recent samples to consider for weight update
        """
        self.models = models
        self.window_size = window_size
        self.weights = {name: 1/len(models) for name in models}
        self.recent_correct = {name: [] for name in models}
        
    def fit(self, X, y):
        """Train all base models"""
        for name, model in self.models.items():
            model.fit(X, y)
        return self
    
    def update_weights(self, X_batch, y_batch):
        """Update weights based on performance on new batch"""
        for name, model in self.models.items():
            predictions = model.predict(X_batch)
            correct = (predictions == y_batch).astype(int)
            self.recent_correct[name].extend(correct)
            
            # Keep only recent samples
            if len(self.recent_correct[name]) > self.window_size:
                self.recent_correct[name] = self.recent_correct[name][-self.window_size:]
        
        # Calculate new weights based on recent accuracy
        accuracies = {}
        for name in self.models:
            if len(self.recent_correct[name]) > 0:
                accuracies[name] = np.mean(self.recent_correct[name])
            else:
                accuracies[name] = 0.5  # Default
        
        # Convert to weights (softmax-like)
        total = sum(accuracies.values())
        if total > 0:
            self.weights = {name: acc/total for name, acc in accuracies.items()}
    
    def predict_proba(self, X):
        """Weighted probability prediction"""
        probas = []
        weights_list = []
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
            else:
                # Convert predictions to pseudo-probabilities
                pred = model.predict(X)
                proba = np.zeros((len(X), 2))
                proba[range(len(X)), pred] = 1
            
            probas.append(proba)
            weights_list.append(self.weights[name])
        
        # Weighted average
        weighted_proba = sum(w * p for w, p in zip(weights_list, probas))
        return weighted_proba
    
    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def get_weights(self):
        """Return current weights"""
        return self.weights.copy()


# Demonstration
X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split for simulation
X_initial, X_stream, y_initial, y_stream = train_test_split(
    X_train, y_train, test_size=0.5, random_state=42
)

# Define models
models = {
    'LR': LogisticRegression(max_iter=1000, random_state=42),
    'RF': RandomForestClassifier(n_estimators=50, random_state=42),
    'GB': GradientBoostingClassifier(n_estimators=50, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Create dynamic ensemble
ensemble = DynamicWeightedEnsemble(models, window_size=100)

# Initial training
ensemble.fit(X_initial, y_initial)

print("Initial Weights:", {k: f"{v:.3f}" for k, v in ensemble.get_weights().items()})
print(f"Initial Test Accuracy: {accuracy_score(y_test, ensemble.predict(X_test)):.4f}")

# Simulate streaming data and dynamic weight updates
batch_size = 50
for i in range(0, len(X_stream), batch_size):
    X_batch = X_stream[i:i+batch_size]
    y_batch = y_stream[i:i+batch_size]
    
    # Update weights based on batch performance
    ensemble.update_weights(X_batch, y_batch)

print("\nAfter processing streaming data:")
print("Updated Weights:", {k: f"{v:.3f}" for k, v in ensemble.get_weights().items()})
print(f"Final Test Accuracy: {accuracy_score(y_test, ensemble.predict(X_test)):.4f}")

# Compare with individual models
print("\nIndividual Model Performance:")
for name, model in models.items():
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  {name}: {acc:.4f} (weight: {ensemble.weights[name]:.3f})")
```

---

## Question 16: Develop a mechanism to periodically retrain an ensemble model with new streaming data
- [ ] Done

### Implementation
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import deque
import time

class StreamingEnsemble:
    def __init__(self, base_models, buffer_size=1000, retrain_threshold=500):
        """
        Ensemble that retrains periodically with new data
        
        Parameters:
        -----------
        base_models: list of (name, model_class, params) tuples
        buffer_size: max samples to keep for retraining
        retrain_threshold: number of new samples before retraining
        """
        self.model_specs = base_models
        self.models = {}
        self.buffer_X = deque(maxlen=buffer_size)
        self.buffer_y = deque(maxlen=buffer_size)
        self.samples_since_retrain = 0
        self.retrain_threshold = retrain_threshold
        self.retrain_count = 0
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Create fresh model instances"""
        for name, model_class, params in self.model_specs:
            self.models[name] = model_class(**params)
    
    def initial_fit(self, X, y):
        """Initial training on historical data"""
        # Add to buffer
        for xi, yi in zip(X, y):
            self.buffer_X.append(xi)
            self.buffer_y.append(yi)
        
        # Train all models
        X_arr = np.array(self.buffer_X)
        y_arr = np.array(self.buffer_y)
        
        for name, model in self.models.items():
            model.fit(X_arr, y_arr)
        
        print(f"Initial training on {len(X)} samples")
        return self
    
    def add_data(self, X_new, y_new):
        """Add new data and retrain if threshold reached"""
        # Add to buffer
        for xi, yi in zip(X_new, y_new):
            self.buffer_X.append(xi)
            self.buffer_y.append(yi)
        
        self.samples_since_retrain += len(X_new)
        
        # Check if retraining needed
        if self.samples_since_retrain >= self.retrain_threshold:
            self._retrain()
    
    def _retrain(self):
        """Retrain all models on buffered data"""
        X_arr = np.array(self.buffer_X)
        y_arr = np.array(self.buffer_y)
        
        # Reinitialize and retrain
        self._initialize_models()
        for name, model in self.models.items():
            model.fit(X_arr, y_arr)
        
        self.retrain_count += 1
        self.samples_since_retrain = 0
        print(f"Retrained (count: {self.retrain_count}) on {len(X_arr)} samples")
    
    def predict(self, X):
        """Ensemble prediction (majority voting)"""
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        # Majority vote
        final = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            final.append(np.bincount(votes.astype(int)).argmax())
        
        return np.array(final)


# Simulation
np.random.seed(42)

# Define models
model_specs = [
    ('RF', RandomForestClassifier, {'n_estimators': 50, 'random_state': 42}),
    ('GB', GradientBoostingClassifier, {'n_estimators': 50, 'random_state': 42}),
    ('LR', LogisticRegression, {'max_iter': 1000, 'random_state': 42})
]

# Create ensemble
ensemble = StreamingEnsemble(
    base_models=model_specs,
    buffer_size=1000,
    retrain_threshold=200
)

# Generate initial data
X_initial, y_initial = make_classification(n_samples=500, n_features=20, random_state=42)
ensemble.initial_fit(X_initial, y_initial)

# Generate test set
X_test, y_test = make_classification(n_samples=200, n_features=20, random_state=123)

# Initial accuracy
print(f"\nInitial Test Accuracy: {accuracy_score(y_test, ensemble.predict(X_test)):.4f}")

# Simulate streaming data
print("\n--- Streaming Data Simulation ---")
for batch in range(5):
    # Generate new batch (could have different distribution)
    X_new, y_new = make_classification(
        n_samples=100, 
        n_features=20, 
        random_state=42 + batch
    )
    
    # Add to ensemble
    ensemble.add_data(X_new, y_new)
    
    # Check accuracy
    acc = accuracy_score(y_test, ensemble.predict(X_test))
    print(f"Batch {batch+1}: Accuracy = {acc:.4f}, Buffer size = {len(ensemble.buffer_X)}")

print(f"\nTotal retrains: {ensemble.retrain_count}")
```

---

## Question 17: Write a script in Python that utilizes early stopping with gradient boosting methods
- [ ] Done

### Implementation
```python
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Create dataset
X, y = make_classification(n_samples=5000, n_features=20, n_informative=15,
                           random_state=42)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Dataset splits:")
print(f"  Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# Method 1: XGBoost Early Stopping
print("\n" + "="*50)
print("Method 1: XGBoost with Early Stopping")
print("="*50)

xgb_model = xgb.XGBClassifier(
    n_estimators=1000,          # High number, early stopping will find optimal
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=50     # Stop if no improvement for 50 rounds
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"Best iteration: {xgb_model.best_iteration}")
print(f"Best score: {xgb_model.best_score:.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, xgb_model.predict(X_test)):.4f}")

# Method 2: LightGBM Early Stopping
print("\n" + "="*50)
print("Method 2: LightGBM with Early Stopping")
print("="*50)

lgb_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbose=-1
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=0)  # Suppress output
    ]
)

print(f"Best iteration: {lgb_model.best_iteration_}")
print(f"Test Accuracy: {accuracy_score(y_test, lgb_model.predict(X_test)):.4f}")

# Method 3: Sklearn GradientBoosting (Manual Early Stopping)
print("\n" + "="*50)
print("Method 3: Sklearn GB with Manual Early Stopping")
print("="*50)

# Train with staged_predict for manual early stopping
gb_model = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train, y_train)

# Find best iteration using staged_predict
val_scores = []
for i, y_pred in enumerate(gb_model.staged_predict(X_val)):
    val_scores.append(accuracy_score(y_val, y_pred))

best_n_estimators = np.argmax(val_scores) + 1
print(f"Best n_estimators: {best_n_estimators}")
print(f"Best validation accuracy: {max(val_scores):.4f}")

# Retrain with optimal n_estimators
gb_final = GradientBoostingClassifier(
    n_estimators=best_n_estimators,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_final.fit(X_train, y_train)
print(f"Test Accuracy: {accuracy_score(y_test, gb_final.predict(X_test)):.4f}")

# Summary
print("\n" + "="*50)
print("Summary: Early Stopping Results")
print("="*50)
print(f"XGBoost:  {xgb_model.best_iteration} iterations, Test Acc: {accuracy_score(y_test, xgb_model.predict(X_test)):.4f}")
print(f"LightGBM: {lgb_model.best_iteration_} iterations, Test Acc: {accuracy_score(y_test, lgb_model.predict(X_test)):.4f}")
print(f"Sklearn:  {best_n_estimators} iterations, Test Acc: {accuracy_score(y_test, gb_final.predict(X_test)):.4f}")
```

---

## Question 18: Create an end-to-end pipeline for training, validating, and selecting the best ensemble setup automatically
- [ ] Done

### Implementation
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier, StackingClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class AutoEnsembleSelector:
    def __init__(self, cv=5, verbose=True):
        self.cv = cv
        self.verbose = verbose
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.best_name = None
        
    def _log(self, message):
        if self.verbose:
            print(message)
    
    def fit(self, X_train, y_train, X_test, y_test):
        """Automatically find best ensemble configuration"""
        
        self._log("="*60)
        self._log("Auto Ensemble Selection Pipeline")
        self._log("="*60)
        
        # Step 1: Evaluate individual models
        self._log("\nStep 1: Evaluating Individual Models")
        self._log("-"*40)
        
        individual_models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
        }
        
        individual_scores = {}
        for name, model in individual_models.items():
            cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv)
            individual_scores[name] = cv_scores.mean()
            self._log(f"  {name}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Select top 4 models for ensembling
        top_models = sorted(individual_scores.items(), key=lambda x: x[1], reverse=True)[:4]
        self._log(f"\nTop 4 models: {[m[0] for m in top_models]}")
        
        # Step 2: Build ensemble configurations
        self._log("\nStep 2: Building Ensemble Configurations")
        self._log("-"*40)
        
        top_model_list = [(name, individual_models[name]) for name, _ in top_models]
        
        ensemble_configs = {
            'VotingHard': VotingClassifier(
                estimators=top_model_list,
                voting='hard'
            ),
            'VotingSoft': VotingClassifier(
                estimators=top_model_list,
                voting='soft'
            ),
            'Stacking_LR': StackingClassifier(
                estimators=top_model_list,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=3
            ),
            'Stacking_RF': StackingClassifier(
                estimators=top_model_list,
                final_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
                cv=3
            )
        }
        
        # Step 3: Evaluate ensembles
        self._log("\nStep 3: Evaluating Ensemble Configurations")
        self._log("-"*40)
        
        for name, ensemble in ensemble_configs.items():
            cv_scores = cross_val_score(ensemble, X_train, y_train, cv=self.cv)
            self.results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': ensemble
            }
            self._log(f"  {name}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Add individual models to results
        for name, score in individual_scores.items():
            self.results[name] = {
                'cv_mean': score,
                'cv_std': 0,  # Already computed above
                'model': individual_models[name]
            }
        
        # Step 4: Select best model
        self._log("\nStep 4: Selecting Best Model")
        self._log("-"*40)
        
        best_name = max(self.results, key=lambda x: self.results[x]['cv_mean'])
        self.best_name = best_name
        self.best_score = self.results[best_name]['cv_mean']
        self.best_model = self.results[best_name]['model']
        
        self._log(f"Best model: {best_name} (CV: {self.best_score:.4f})")
        
        # Step 5: Final training and evaluation
        self._log("\nStep 5: Final Training and Evaluation")
        self._log("-"*40)
        
        self.best_model.fit(X_train, y_train)
        test_accuracy = accuracy_score(y_test, self.best_model.predict(X_test))
        
        self._log(f"Final Test Accuracy: {test_accuracy:.4f}")
        
        return self
    
    def get_best_model(self):
        return self.best_model
    
    def get_results_summary(self):
        return {name: res['cv_mean'] for name, res in self.results.items()}


# Run the pipeline
X, y = make_classification(n_samples=2000, n_features=20, n_informative=15,
                           n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

selector = AutoEnsembleSelector(cv=5, verbose=True)
selector.fit(X_train, y_train, X_test, y_test)

# Get best model for deployment
best_model = selector.get_best_model()
print(f"\nBest model selected: {selector.best_name}")
```

---

## Question 19: Script a solution for an imbalanced classification problem using ensemble learning with proper sampling techniques
- [ ] Done

### Implementation
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Create highly imbalanced dataset (1:20 ratio)
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=10,
    weights=[0.95, 0.05],  # 95% class 0, 5% class 1
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                     stratify=y, random_state=42)

print("Class Distribution:")
print(f"  Training: {np.bincount(y_train)}")
print(f"  Test: {np.bincount(y_test)}")
print(f"  Imbalance Ratio: 1:{np.bincount(y_train)[0]//np.bincount(y_train)[1]}")

# Define evaluation function
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """Train and evaluate model with multiple metrics"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n{name}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    return {'f1': f1, 'auc': auc}

print("\n" + "="*60)
print("Comparing Different Approaches for Imbalanced Data")
print("="*60)

results = {}

# Approach 1: Baseline (No handling)
print("\n--- Approach 1: Baseline (No handling) ---")
baseline = RandomForestClassifier(n_estimators=100, random_state=42)
results['Baseline'] = evaluate_model('Baseline RF', baseline, X_train, y_train, X_test, y_test)

# Approach 2: Class Weights
print("\n--- Approach 2: Class Weights ---")
weighted = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
results['ClassWeight'] = evaluate_model('Weighted RF', weighted, X_train, y_train, X_test, y_test)

# Approach 3: SMOTE + Random Forest
print("\n--- Approach 3: SMOTE + Random Forest ---")
smote_pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])
results['SMOTE'] = evaluate_model('SMOTE + RF', smote_pipe, X_train, y_train, X_test, y_test)

# Approach 4: SMOTE-Tomek (Combination)
print("\n--- Approach 4: SMOTE-Tomek ---")
smotetomek_pipe = Pipeline([
    ('smotetomek', SMOTETomek(random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])
results['SMOTETomek'] = evaluate_model('SMOTE-Tomek + RF', smotetomek_pipe, X_train, y_train, X_test, y_test)

# Approach 5: Balanced Random Forest
print("\n--- Approach 5: Balanced Random Forest ---")
brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
results['BalancedRF'] = evaluate_model('Balanced RF', brf, X_train, y_train, X_test, y_test)

# Approach 6: Easy Ensemble
print("\n--- Approach 6: Easy Ensemble ---")
ee = EasyEnsembleClassifier(n_estimators=10, random_state=42)
results['EasyEnsemble'] = evaluate_model('Easy Ensemble', ee, X_train, y_train, X_test, y_test)

# Summary
print("\n" + "="*60)
print("Summary of Results")
print("="*60)
print(f"{'Method':<20} {'F1 Score':<12} {'AUC-ROC':<12}")
print("-"*44)
for name, metrics in results.items():
    print(f"{name:<20} {metrics['f1']:<12.4f} {metrics['auc']:<12.4f}")

# Best approach
best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
best_auc = max(results.items(), key=lambda x: x[1]['auc'])
print(f"\nBest by F1: {best_f1[0]} ({best_f1[1]['f1']:.4f})")
print(f"Best by AUC: {best_auc[0]} ({best_auc[1]['auc']:.4f})")
```

---

## Question 20: Generate a synthetic dataset with Python and apply different ensemble learning models to compare their generalization capabilities
- [ ] Done

### Implementation
```python
import numpy as np
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, BaggingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def generate_datasets():
    """Generate various synthetic datasets"""
    datasets = {}
    
    # Dataset 1: Linear separable
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               n_redundant=5, n_clusters_per_class=1, random_state=42)
    datasets['Linear'] = (X, y)
    
    # Dataset 2: Non-linear (moons)
    X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
    datasets['Moons'] = (X, y)
    
    # Dataset 3: Non-linear (circles)
    X, y = make_circles(n_samples=1000, noise=0.2, factor=0.5, random_state=42)
    datasets['Circles'] = (X, y)
    
    # Dataset 4: High dimensional sparse
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=10,
                               n_redundant=10, n_clusters_per_class=2, random_state=42)
    datasets['HighDim'] = (X, y)
    
    # Dataset 5: Noisy
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=5,
                               n_redundant=5, flip_y=0.2, random_state=42)
    datasets['Noisy'] = (X, y)
    
    return datasets

def get_models():
    """Define ensemble models to compare"""
    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'Bagging': BaggingClassifier(n_estimators=100, random_state=42),
        'Voting': VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=1000)),
                ('rf', RandomForestClassifier(n_estimators=50)),
                ('gb', GradientBoostingClassifier(n_estimators=50))
            ],
            voting='soft'
        )
    }
    return models

# Generate datasets
print("Generating Synthetic Datasets...")
datasets = generate_datasets()
models = get_models()

# Results storage
results = {model_name: {} for model_name in models}

print("\n" + "="*70)
print("Comparing Ensemble Models Across Different Dataset Types")
print("="*70)

for dataset_name, (X, y) in datasets.items():
    print(f"\n--- Dataset: {dataset_name} ---")
    print(f"    Shape: {X.shape}, Classes: {np.unique(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    for model_name, model in models.items():
        # Cross-validation score (generalization)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # Train and test
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        results[model_name][dataset_name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train': train_acc,
            'test': test_acc,
            'overfit_gap': train_acc - test_acc
        }

# Print results table
print("\n" + "="*70)
print("Results Summary: Test Accuracy")
print("="*70)

# Header
header = f"{'Model':<18}"
for ds_name in datasets:
    header += f"{ds_name:<12}"
print(header)
print("-"*70)

# Rows
for model_name in models:
    row = f"{model_name:<18}"
    for ds_name in datasets:
        acc = results[model_name][ds_name]['test']
        row += f"{acc:<12.4f}"
    print(row)

# Generalization Gap Analysis
print("\n" + "="*70)
print("Overfitting Analysis: Train - Test Accuracy Gap")
print("="*70)

header = f"{'Model':<18}"
for ds_name in datasets:
    header += f"{ds_name:<12}"
print(header)
print("-"*70)

for model_name in models:
    row = f"{model_name:<18}"
    for ds_name in datasets:
        gap = results[model_name][ds_name]['overfit_gap']
        row += f"{gap:<12.4f}"
    print(row)

# Best model per dataset
print("\n" + "="*70)
print("Best Model per Dataset (by Test Accuracy)")
print("="*70)
for ds_name in datasets:
    best_model = max(models.keys(), 
                     key=lambda m: results[m][ds_name]['test'])
    best_acc = results[best_model][ds_name]['test']
    print(f"{ds_name}: {best_model} ({best_acc:.4f})")
```

---

## Question 21: Implement a collaborative filtering recommendation system using a stack of matrix factorization models as an ensemble
- [ ] Done

### Implementation
```python
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class SimpleMF:
    """Simple Matrix Factorization using SVD"""
    def __init__(self, n_factors=20, random_state=None):
        self.n_factors = n_factors
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        
    def fit(self, ratings_matrix):
        """Fit matrix factorization"""
        self.global_mean = np.nanmean(ratings_matrix[ratings_matrix > 0])
        
        # Fill missing with global mean for SVD
        filled_matrix = ratings_matrix.copy()
        filled_matrix[filled_matrix == 0] = self.global_mean
        
        # Center the data
        centered = filled_matrix - self.global_mean
        
        # SVD
        U, sigma, Vt = svds(centered, k=min(self.n_factors, min(centered.shape)-1))
        
        self.user_factors = U
        self.sigma = sigma
        self.item_factors = Vt.T
        
        return self
    
    def predict(self, user_idx, item_idx):
        """Predict rating for user-item pair"""
        pred = self.global_mean + np.dot(
            self.user_factors[user_idx] * self.sigma,
            self.item_factors[item_idx]
        )
        return np.clip(pred, 1, 5)  # Clip to valid rating range
    
    def predict_all(self, ratings_matrix):
        """Predict all ratings"""
        predictions = self.global_mean + np.dot(
            self.user_factors * self.sigma,
            self.item_factors.T
        )
        return np.clip(predictions, 1, 5)


class MFEnsemble:
    """Ensemble of Matrix Factorization models"""
    def __init__(self, n_models=5, factors_range=(10, 50)):
        self.n_models = n_models
        self.factors_range = factors_range
        self.models = []
        self.weights = None
        
    def fit(self, ratings_matrix, val_matrix=None):
        """Train ensemble of MF models with different configurations"""
        # Create models with different n_factors
        factors_list = np.linspace(
            self.factors_range[0], 
            self.factors_range[1], 
            self.n_models
        ).astype(int)
        
        self.models = []
        val_errors = []
        
        for i, n_factors in enumerate(factors_list):
            model = SimpleMF(n_factors=n_factors, random_state=42+i)
            model.fit(ratings_matrix)
            self.models.append(model)
            
            # Calculate validation error if validation data provided
            if val_matrix is not None:
                predictions = model.predict_all(ratings_matrix)
                mask = val_matrix > 0
                if mask.sum() > 0:
                    error = np.sqrt(mean_squared_error(
                        val_matrix[mask], 
                        predictions[mask]
                    ))
                    val_errors.append(error)
                    print(f"Model {i+1} (factors={n_factors}): RMSE = {error:.4f}")
        
        # Calculate weights based on validation performance
        if val_errors:
            # Inverse error weighting
            inverse_errors = [1/(e + 0.001) for e in val_errors]
            total = sum(inverse_errors)
            self.weights = [ie/total for ie in inverse_errors]
        else:
            self.weights = [1/self.n_models] * self.n_models
        
        print(f"\nModel weights: {[f'{w:.3f}' for w in self.weights]}")
        return self
    
    def predict(self, user_idx, item_idx):
        """Ensemble prediction for single user-item pair"""
        predictions = [model.predict(user_idx, item_idx) for model in self.models]
        return np.average(predictions, weights=self.weights)
    
    def predict_all(self, ratings_matrix):
        """Ensemble predictions for all user-item pairs"""
        all_predictions = [model.predict_all(ratings_matrix) for model in self.models]
        weighted_pred = np.average(all_predictions, axis=0, weights=self.weights)
        return weighted_pred
    
    def recommend(self, user_idx, ratings_matrix, n_items=5):
        """Recommend top N items for a user"""
        predictions = self.predict_all(ratings_matrix)
        user_predictions = predictions[user_idx]
        
        # Exclude already rated items
        already_rated = ratings_matrix[user_idx] > 0
        user_predictions[already_rated] = -np.inf
        
        # Get top N
        top_items = np.argsort(user_predictions)[-n_items:][::-1]
        top_scores = user_predictions[top_items]
        
        return list(zip(top_items, top_scores))


# Generate synthetic ratings data
np.random.seed(42)

n_users = 100
n_items = 50

# Create sparse ratings matrix (most entries are 0 = not rated)
ratings = np.zeros((n_users, n_items))

# Fill ~20% of entries with ratings
for u in range(n_users):
    n_rated = np.random.randint(5, 15)  # Each user rates 5-15 items
    rated_items = np.random.choice(n_items, n_rated, replace=False)
    ratings[u, rated_items] = np.random.randint(1, 6, n_rated)

print("Synthetic Ratings Matrix:")
print(f"  Shape: {ratings.shape}")
print(f"  Sparsity: {(ratings == 0).sum() / ratings.size * 100:.1f}%")
print(f"  Rating range: {ratings[ratings > 0].min():.0f} - {ratings[ratings > 0].max():.0f}")

# Create train/validation split
train_matrix = ratings.copy()
val_matrix = np.zeros_like(ratings)

# Hold out some ratings for validation
for u in range(n_users):
    rated_items = np.where(ratings[u] > 0)[0]
    if len(rated_items) > 2:
        val_items = np.random.choice(rated_items, size=min(2, len(rated_items)//2), replace=False)
        val_matrix[u, val_items] = ratings[u, val_items]
        train_matrix[u, val_items] = 0

print(f"\nTraining ratings: {(train_matrix > 0).sum()}")
print(f"Validation ratings: {(val_matrix > 0).sum()}")

# Train ensemble
print("\n" + "="*50)
print("Training MF Ensemble")
print("="*50)

ensemble = MFEnsemble(n_models=5, factors_range=(5, 30))
ensemble.fit(train_matrix, val_matrix)

# Evaluate ensemble
predictions = ensemble.predict_all(train_matrix)
val_mask = val_matrix > 0

ensemble_rmse = np.sqrt(mean_squared_error(val_matrix[val_mask], predictions[val_mask]))
print(f"\nEnsemble Validation RMSE: {ensemble_rmse:.4f}")

# Compare with single model
single_model = SimpleMF(n_factors=20)
single_model.fit(train_matrix)
single_predictions = single_model.predict_all(train_matrix)
single_rmse = np.sqrt(mean_squared_error(val_matrix[val_mask], single_predictions[val_mask]))
print(f"Single Model (factors=20) RMSE: {single_rmse:.4f}")
print(f"Improvement: {(single_rmse - ensemble_rmse) / single_rmse * 100:.2f}%")

# Demonstrate recommendations
print("\n" + "="*50)
print("Sample Recommendations for User 0")
print("="*50)

user_id = 0
recommendations = ensemble.recommend(user_id, train_matrix, n_items=5)

print(f"\nUser {user_id}'s past ratings:")
past_items = np.where(train_matrix[user_id] > 0)[0]
for item in past_items[:5]:
    print(f"  Item {item}: {train_matrix[user_id, item]:.0f} stars")

print(f"\nTop 5 Recommendations:")
for item, score in recommendations:
    print(f"  Item {item}: Predicted {score:.2f} stars")
```

---
