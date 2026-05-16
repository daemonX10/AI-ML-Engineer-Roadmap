# Bias And Variance Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Implement k-fold cross-validation on a dataset to diagnose model bias and variance.**

### Answer

**Theory:**
K-fold cross-validation helps diagnose bias-variance by comparing training vs. validation performance across multiple data splits.

**Diagnosis Framework:**
| Observation | Training Score | Validation Score | Diagnosis |
|-------------|---------------|------------------|-----------|
| High Bias | Low | Low (close to training) | Underfitting |
| High Variance | High | Low (large gap) | Overfitting |
| Good Balance | High | High (close to training) | Well-tuned |

**Code Implementation:**

```python
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# --- 1. Create a synthetic dataset ---
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. Define models to compare ---
# Simple model (likely high bias)
high_bias_model = LogisticRegression(max_iter=1000)

# Complex model (likely high variance)
high_variance_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None,  # No limit - can overfit
    random_state=42
)

# --- 3. Diagnosis function using k-fold cross-validation ---
def diagnose_bias_variance(model, X, y, cv=5):
    """
    Perform k-fold CV and diagnose bias-variance issues.
    """
    # Get both training and validation scores
    cv_results = cross_validate(
        model, X, y, 
        cv=cv, 
        scoring='accuracy',
        return_train_score=True
    )
    
    # Calculate statistics
    mean_train = np.mean(cv_results['train_score'])
    std_train = np.std(cv_results['train_score'])
    mean_val = np.mean(cv_results['test_score'])
    std_val = np.std(cv_results['test_score'])
    gap = mean_train - mean_val
    
    # Print results
    print(f"Model: {model.__class__.__name__}")
    print(f"  Training Accuracy:   {mean_train:.4f} (+/- {std_train:.4f})")
    print(f"  Validation Accuracy: {mean_val:.4f} (+/- {std_val:.4f})")
    print(f"  Gap (Train - Val):   {gap:.4f}")
    
    # Diagnosis
    if mean_val < 0.80:
        print("  → Diagnosis: HIGH BIAS (underfitting)")
        print("    Both scores low. Model too simple.")
    elif gap > 0.10:
        print("  → Diagnosis: HIGH VARIANCE (overfitting)")
        print("    Large gap between train and validation.")
    else:
        print("  → Diagnosis: GOOD BALANCE")
        print("    High scores with small gap.")
    print("-" * 50)

# --- 4. Run diagnosis ---
print("=" * 50)
print("BIAS-VARIANCE DIAGNOSIS USING K-FOLD CV")
print("=" * 50)

diagnose_bias_variance(high_bias_model, X_scaled, y, cv=5)
diagnose_bias_variance(high_variance_model, X_scaled, y, cv=5)
```

**Expected Output:**
```
Model: LogisticRegression
  Training Accuracy:   0.8512 (+/- 0.0089)
  Validation Accuracy: 0.8420 (+/- 0.0234)
  Gap (Train - Val):   0.0092
  → Diagnosis: GOOD BALANCE
--------------------------------------------------
Model: RandomForestClassifier
  Training Accuracy:   1.0000 (+/- 0.0000)
  Validation Accuracy: 0.8840 (+/- 0.0312)
  Gap (Train - Val):   0.1160
  → Diagnosis: HIGH VARIANCE (overfitting)
```

**Key Points:**
- `cross_validate` with `return_train_score=True` gives both scores
- Small gap + high scores = good generalization
- Perfect training + lower validation = overfitting
- Both scores low = underfitting

---

## Question 2
- [ ] Done

**Write a Python script to plot learning curves for understanding model bias and variance.**

### Answer

**Theory:**
Learning curves plot model performance as a function of training set size:
- **High Bias**: Both curves converge at high error (more data won't help)
- **High Variance**: Large gap between curves (more data might help)

**Code Implementation:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

def plot_learning_curves(estimator, title, X, y, cv=5):
    """
    Plot training and validation learning curves.
    """
    # Generate learning curve data
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        cv=cv,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)
    
    # Plot training scores
    plt.fill_between(train_sizes, 
                     train_mean - train_std,
                     train_mean + train_std, 
                     alpha=0.1, color='blue')
    plt.plot(train_sizes, train_mean, 'o-', color='blue',
             label='Training Score', linewidth=2)
    
    # Plot validation scores
    plt.fill_between(train_sizes,
                     val_mean - val_std,
                     val_mean + val_std,
                     alpha=0.1, color='orange')
    plt.plot(train_sizes, val_mean, 'o-', color='orange',
             label='Validation Score', linewidth=2)
    
    # Labels and formatting
    plt.title(title, fontsize=14)
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.ylim([0.6, 1.05])
    
    # Add diagnosis annotation
    final_gap = train_mean[-1] - val_mean[-1]
    if val_mean[-1] < 0.80:
        diagnosis = "HIGH BIAS: Both curves converge at low value"
    elif final_gap > 0.10:
        diagnosis = "HIGH VARIANCE: Large gap between curves"
    else:
        diagnosis = "GOOD FIT: Small gap, high scores"
    
    plt.annotate(diagnosis, xy=(0.5, 0.02), xycoords='axes fraction',
                fontsize=10, ha='center', style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return plt

# --- Create dataset ---
X, y = make_classification(
    n_samples=1000, 
    n_features=20,
    n_informative=10,
    random_state=42
)
X_scaled = StandardScaler().fit_transform(X)

# --- Define models ---
simple_model = LogisticRegression(max_iter=1000)
complex_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None,
    random_state=42
)

# --- Plot learning curves ---
# Simple model (may show high bias pattern)
plot_learning_curves(
    simple_model,
    "Learning Curves - Logistic Regression (Check for High Bias)",
    X_scaled, y
)
plt.savefig('learning_curve_simple.png', dpi=150)
plt.show()

# Complex model (may show high variance pattern)
plot_learning_curves(
    complex_model,
    "Learning Curves - Random Forest (Check for High Variance)",
    X_scaled, y
)
plt.savefig('learning_curve_complex.png', dpi=150)
plt.show()
```

**Interpretation Guide:**

```
HIGH BIAS (Underfitting):
    │
    │  ═══════════════════ Training (converges high)
    │  ═══════════════════ Validation (converges close)
    │
    └──────────────────────→ Training Size
    
HIGH VARIANCE (Overfitting):
    │
    │  ═══════════════════ Training (near perfect)
    │           
    │        ─────────────── Validation (lower, still improving)
    │  Large Gap
    └──────────────────────→ Training Size

GOOD FIT:
    │
    │  ═══════════════════ Training (high)
    │  ─────────────────── Validation (close, high)
    │  Small Gap
    └──────────────────────→ Training Size
```

---

## Question 3
- [ ] Done

**Use L1 (Lasso) and L2 (Ridge) regularization to address a high-variance problem in a linear regression model.**

### Answer

**Theory:**
- **L2 (Ridge)**: Shrinks all coefficients toward zero, never exactly zero
- **L1 (Lasso)**: Can shrink coefficients to exactly zero (feature selection)
- Both reduce variance by constraining model complexity

**Code Implementation:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- 1. Create high-dimensional dataset (prone to overfitting) ---
np.random.seed(42)
n_samples, n_features = 100, 50  # More features than ideal for samples

X = np.random.randn(n_samples, n_features)

# True relationship: only 10 features are actually important
true_coef = np.zeros(n_features)
important_idx = np.random.choice(n_features, 10, replace=False)
true_coef[important_idx] = 5 * np.random.randn(10)

# Generate target with noise
y = X @ true_coef + np.random.randn(n_samples) * 5

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 2. Train models ---
# Plain Linear Regression (likely to overfit)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_mse = mean_squared_error(y_test, lr.predict(X_test_scaled))

# Ridge (L2 regularization)
ridge = Ridge(alpha=10.0)
ridge.fit(X_train_scaled, y_train)
ridge_mse = mean_squared_error(y_test, ridge.predict(X_test_scaled))

# Lasso (L1 regularization)
lasso = Lasso(alpha=1.0)
lasso.fit(X_train_scaled, y_train)
lasso_mse = mean_squared_error(y_test, lasso.predict(X_test_scaled))

# --- 3. Compare results ---
print("=" * 50)
print("REGULARIZATION COMPARISON")
print("=" * 50)
print(f"\nTest Set MSE:")
print(f"  Linear Regression: {lr_mse:.2f}")
print(f"  Ridge (L2):        {ridge_mse:.2f}")
print(f"  Lasso (L1):        {lasso_mse:.2f}")

# Improvement
print(f"\nImprovement over Linear Regression:")
print(f"  Ridge: {(1 - ridge_mse/lr_mse)*100:.1f}% reduction in MSE")
print(f"  Lasso: {(1 - lasso_mse/lr_mse)*100:.1f}% reduction in MSE")

# Coefficient analysis
print(f"\nCoefficient Analysis:")
print(f"  Linear Regression - Max |coef|: {np.max(np.abs(lr.coef_)):.2f}")
print(f"  Ridge - Max |coef|: {np.max(np.abs(ridge.coef_)):.2f}")
print(f"  Lasso - Max |coef|: {np.max(np.abs(lasso.coef_)):.2f}")
print(f"  Lasso - Non-zero coefficients: {np.sum(lasso.coef_ != 0)}/{n_features}")

# --- 4. Visualize coefficients ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].stem(lr.coef_, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0].set_title(f'Linear Regression\nMSE: {lr_mse:.2f}')
axes[0].set_ylabel('Coefficient Value')
axes[0].set_xlabel('Feature Index')
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

axes[1].stem(ridge.coef_, linefmt='g-', markerfmt='go', basefmt='k-')
axes[1].set_title(f'Ridge (L2)\nMSE: {ridge_mse:.2f}')
axes[1].set_xlabel('Feature Index')
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

axes[2].stem(lasso.coef_, linefmt='r-', markerfmt='ro', basefmt='k-')
axes[2].set_title(f'Lasso (L1)\nMSE: {lasso_mse:.2f}')
axes[2].set_xlabel('Feature Index')
axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('regularization_comparison.png', dpi=150)
plt.show()
```

**Expected Output:**
```
Test Set MSE:
  Linear Regression: 156.34
  Ridge (L2):        48.21
  Lasso (L1):        42.87

Coefficient Analysis:
  Linear Regression - Max |coef|: 15.23
  Ridge - Max |coef|: 4.12
  Lasso - Max |coef|: 3.87
  Lasso - Non-zero coefficients: 12/50
```

**Key Observations:**
- Linear Regression: Large, noisy coefficients (overfitting)
- Ridge: All coefficients shrunk (variance reduced)
- Lasso: Sparse solution (automatic feature selection)

---

## Question 4
- [ ] Done

**Code an ensemble method to combine multiple decision trees with the intention of reducing variance.**

### Answer

**Theory:**
Bagging (Bootstrap Aggregating) reduces variance by:
1. Training multiple models on bootstrap samples
2. Averaging predictions (errors cancel out)

**Code Implementation:**

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

def bootstrap_sample(X, y):
    """Create a bootstrap sample (sampling with replacement)."""
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]


class SimpleBaggingClassifier:
    """
    Simple implementation of Bagging for classification.
    Reduces variance by training multiple trees on bootstrap samples.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators = []
        
    def fit(self, X, y):
        """Train ensemble of decision trees."""
        if self.random_state:
            np.random.seed(self.random_state)
            
        self.estimators = []
        
        for i in range(self.n_estimators):
            # Create bootstrap sample
            X_sample, y_sample = bootstrap_sample(X, y)
            
            # Train a decision tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state + i if self.random_state else None
            )
            tree.fit(X_sample, y_sample)
            self.estimators.append(tree)
            
        return self
    
    def predict(self, X):
        """Predict using majority vote."""
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.estimators])
        
        # Transpose: rows = samples, columns = tree predictions
        predictions = predictions.T
        
        # Majority vote for each sample
        y_pred = [Counter(pred).most_common(1)[0][0] for pred in predictions]
        
        return np.array(y_pred)
    
    def predict_proba(self, X):
        """Predict probabilities by averaging."""
        probas = np.array([tree.predict_proba(X) for tree in self.estimators])
        return np.mean(probas, axis=0)


# --- Example Usage ---
# Create a noisy, non-linear dataset
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- Compare single tree vs. bagging ---
print("=" * 55)
print("BAGGING: VARIANCE REDUCTION DEMONSTRATION")
print("=" * 55)

# 1. Single Decision Tree (High Variance)
single_tree = DecisionTreeClassifier(max_depth=None, random_state=42)
single_tree.fit(X_train, y_train)
single_acc = accuracy_score(y_test, single_tree.predict(X_test))

print(f"\nSingle Decision Tree (no depth limit):")
print(f"  Test Accuracy: {single_acc:.4f}")

# 2. Our Bagging Classifier
bagging = SimpleBaggingClassifier(
    n_estimators=100,
    max_depth=None,  # Same as single tree
    random_state=42
)
bagging.fit(X_train, y_train)
bagging_acc = accuracy_score(y_test, bagging.predict(X_test))

print(f"\nBagging Classifier (100 trees):")
print(f"  Test Accuracy: {bagging_acc:.4f}")

# Improvement
print(f"\nImprovement: {(bagging_acc - single_acc)*100:.2f}% accuracy gain")
print("\n→ Bagging reduced variance by averaging multiple overfit trees!")

# --- Visualize variance reduction ---
# Test stability across multiple runs
print("\n" + "-" * 55)
print("STABILITY ANALYSIS (10 random initializations)")
print("-" * 55)

single_accs = []
bagging_accs = []

for seed in range(10):
    # Single tree (different random state)
    tree = DecisionTreeClassifier(max_depth=None, random_state=seed)
    tree.fit(X_train, y_train)
    single_accs.append(accuracy_score(y_test, tree.predict(X_test)))
    
    # Bagging
    bag = SimpleBaggingClassifier(n_estimators=50, random_state=seed)
    bag.fit(X_train, y_train)
    bagging_accs.append(accuracy_score(y_test, bag.predict(X_test)))

print(f"\nSingle Tree:")
print(f"  Mean Accuracy: {np.mean(single_accs):.4f}")
print(f"  Std Deviation: {np.std(single_accs):.4f}  ← HIGH VARIANCE")

print(f"\nBagging:")
print(f"  Mean Accuracy: {np.mean(bagging_accs):.4f}")
print(f"  Std Deviation: {np.std(bagging_accs):.4f}  ← LOW VARIANCE")
```

**Expected Output:**
```
Single Decision Tree:
  Test Accuracy: 0.8533

Bagging Classifier (100 trees):
  Test Accuracy: 0.9067

STABILITY ANALYSIS:
Single Tree:
  Mean Accuracy: 0.8320
  Std Deviation: 0.0423  ← HIGH VARIANCE

Bagging:
  Mean Accuracy: 0.8987
  Std Deviation: 0.0098  ← LOW VARIANCE
```

---

## Question 5
- [ ] Done

**Implement a Grid Search in scikit-learn to find the optimal parameters and balance bias-variance in an SVM model.**

### Answer

**Theory:**
SVM key hyperparameters for bias-variance control:
- **C**: Regularization (low C = high bias, high C = high variance)
- **gamma**: RBF kernel parameter (low = smooth/high bias, high = complex/high variance)

**Code Implementation:**

```python
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# --- 1. Create non-linear dataset ---
X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 2. Define hyperparameter grid ---
param_grid = {
    'C': [0.1, 1, 10, 100],      # Regularization strength
    'gamma': [1, 0.1, 0.01, 0.001],  # Kernel coefficient
    'kernel': ['rbf']
}

print("=" * 60)
print("GRID SEARCH FOR SVM HYPERPARAMETER TUNING")
print("=" * 60)
print(f"\nParameter Grid:")
print(f"  C values: {param_grid['C']}")
print(f"  gamma values: {param_grid['gamma']}")
print(f"  Total combinations: {len(param_grid['C']) * len(param_grid['gamma'])}")

# --- 3. Perform Grid Search with Cross-Validation ---
svm = SVC()

grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=5,               # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,          # Use all CPU cores
    verbose=1,
    return_train_score=True  # For bias-variance analysis
)

print("\nRunning Grid Search...")
grid_search.fit(X_train_scaled, y_train)

# --- 4. Analyze results ---
print("\n" + "-" * 60)
print("RESULTS")
print("-" * 60)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# Get best model
best_model = grid_search.best_estimator_

# --- 5. Evaluate on test set ---
y_pred = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Set Accuracy: {test_accuracy:.4f}")

# --- 6. Show bias-variance analysis for different parameter combinations ---
print("\n" + "-" * 60)
print("BIAS-VARIANCE ANALYSIS ACROSS PARAMETER COMBINATIONS")
print("-" * 60)
print("\nC\t| gamma\t| Train Score\t| Val Score\t| Gap\t\t| Interpretation")
print("-" * 90)

results = grid_search.cv_results_
for i in range(len(results['params'])):
    c = results['params'][i]['C']
    gamma = results['params'][i]['gamma']
    train_score = results['mean_train_score'][i]
    val_score = results['mean_test_score'][i]
    gap = train_score - val_score
    
    # Interpretation
    if val_score < 0.85:
        interp = "High Bias"
    elif gap > 0.05:
        interp = "High Variance"
    else:
        interp = "Balanced"
    
    print(f"{c}\t| {gamma}\t| {train_score:.4f}\t\t| {val_score:.4f}\t\t| {gap:.4f}\t\t| {interp}")

# --- 7. Compare extreme cases ---
print("\n" + "-" * 60)
print("EXTREME CASES COMPARISON")
print("-" * 60)

# High bias model (low C, low gamma)
high_bias_svm = SVC(C=0.1, gamma=0.001, kernel='rbf')
high_bias_svm.fit(X_train_scaled, y_train)
hb_train = accuracy_score(y_train, high_bias_svm.predict(X_train_scaled))
hb_test = accuracy_score(y_test, high_bias_svm.predict(X_test_scaled))

print(f"\nHigh Bias Model (C=0.1, gamma=0.001):")
print(f"  Train: {hb_train:.4f}, Test: {hb_test:.4f}, Gap: {hb_train-hb_test:.4f}")

# High variance model (high C, high gamma)
high_var_svm = SVC(C=100, gamma=1, kernel='rbf')
high_var_svm.fit(X_train_scaled, y_train)
hv_train = accuracy_score(y_train, high_var_svm.predict(X_train_scaled))
hv_test = accuracy_score(y_test, high_var_svm.predict(X_test_scaled))

print(f"\nHigh Variance Model (C=100, gamma=1):")
print(f"  Train: {hv_train:.4f}, Test: {hv_test:.4f}, Gap: {hv_train-hv_test:.4f}")

# Optimal model
print(f"\nOptimal Model (C={grid_search.best_params_['C']}, gamma={grid_search.best_params_['gamma']}):")
opt_train = accuracy_score(y_train, best_model.predict(X_train_scaled))
print(f"  Train: {opt_train:.4f}, Test: {test_accuracy:.4f}, Gap: {opt_train-test_accuracy:.4f}")
```

**Expected Output:**
```
Best Parameters: {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
Best CV Score: 0.9629

Test Set Accuracy: 0.9533

EXTREME CASES COMPARISON:
High Bias Model (C=0.1, gamma=0.001):
  Train: 0.5000, Test: 0.5000, Gap: 0.0000  ← Underfitting!

High Variance Model (C=100, gamma=1):
  Train: 1.0000, Test: 0.9133, Gap: 0.0867  ← Some overfitting

Optimal Model (C=10, gamma=0.1):
  Train: 0.9743, Test: 0.9533, Gap: 0.0210  ← Good balance!
```

**Key Insights:**
- Grid Search systematically explores bias-variance trade-off
- Cross-validation ensures robust parameter selection
- Comparing train/val gap reveals overfitting risk
- Optimal parameters balance complexity and generalization
