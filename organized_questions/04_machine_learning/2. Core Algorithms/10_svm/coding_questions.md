# Svm Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Implement a basic linear SVM from scratch using Python.**

### Answer

**Pipeline:**
1. Initialize weights and bias
2. Compute hinge loss and gradient
3. Update weights using gradient descent
4. Repeat until convergence

**Code:**
```python
import numpy as np

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param  # Regularization
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Convert labels to -1, 1
        y_ = np.where(y <= 0, -1, 1)
        
        # Initialize weights
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Check if point satisfies margin condition
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
                    # Only regularization gradient
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Regularization + hinge loss gradient
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * (-y_[idx])
    
    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)

# Usage
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate data
X, y = make_blobs(n_samples=200, centers=2, random_state=42)
y = np.where(y == 0, -1, 1)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
svm = LinearSVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X_train, y_train)

# Predict
predictions = svm.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2f}")
```

**Key Points:**
- Hinge loss: max(0, 1 - y(wx + b))
- Gradient when violated: w -= lr * (λw - yx)
- Gradient when satisfied: w -= lr * (λw)
- λ controls regularization strength

---

## Question 2
- [ ] Done

**Write a Python function to select an optimal C parameter for an SVM using cross-validation.**

### Answer

**Pipeline:**
1. Define range of C values to test
2. Use GridSearchCV or manual cross-validation
3. Fit SVM for each C value
4. Select C with best CV score

**Code:**
```python
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load and prepare data
X, y = load_iris(return_X_y=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Method 1: GridSearchCV (Recommended)
def find_best_C_gridsearch(X, y):
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    
    svm = SVC(kernel='rbf')
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    
    print(f"Best C: {grid_search.best_params_['C']}")
    print(f"Best Score: {grid_search.best_score_:.4f}")
    return grid_search.best_params_['C']

# Method 2: Manual Cross-Validation
def find_best_C_manual(X, y):
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    best_C = None
    best_score = 0
    
    for C in C_values:
        svm = SVC(kernel='rbf', C=C)
        scores = cross_val_score(svm, X, y, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        
        print(f"C={C}: Mean Accuracy = {mean_score:.4f} (+/- {scores.std():.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_C = C
    
    print(f"\nBest C: {best_C} with accuracy: {best_score:.4f}")
    return best_C

# Usage
best_C = find_best_C_gridsearch(X_scaled, y)
```

**Key Points:**
- Always scale features before SVM
- Use logarithmic scale for C values (0.001, 0.01, 0.1, 1, 10, ...)
- Higher C = stricter margin, potential overfitting
- Lower C = wider margin, potential underfitting

---

## Question 3
- [ ] Done

**Code an SVM model in scikit-learn to classify text data using TF-IDF features.**

### Answer

**Pipeline:**
1. Load text data
2. Convert text to TF-IDF features
3. Train LinearSVC (best for text)
4. Evaluate on test set

**Code:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Sample text data
texts = [
    "I love this movie, it was fantastic",
    "Great film, highly recommend",
    "Terrible movie, waste of time",
    "Awful film, very disappointing",
    "Amazing story and acting",
    "Boring and predictable plot",
    "Best movie I have ever seen",
    "Worst experience ever"
]
labels = [1, 1, 0, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42
)

# Method 1: Using Pipeline (Recommended)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('svm', LinearSVC(C=1.0, max_iter=10000))
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print(classification_report(y_test, predictions))

# Method 2: Step by step
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

svm = LinearSVC(C=1.0)
svm.fit(X_train_tfidf, y_train)
predictions = svm.predict(X_test_tfidf)

# Predict new text
new_text = ["This movie was absolutely wonderful"]
new_pred = pipeline.predict(new_text)
print(f"Prediction for new text: {'Positive' if new_pred[0] == 1 else 'Negative'}")
```

**Key Points:**
- Use LinearSVC for text (faster than SVC with linear kernel)
- TF-IDF handles sparse, high-dimensional text data well
- ngram_range=(1,2) captures word pairs
- Linear kernel works best for text classification

---

## Question 4
- [ ] Done

**Develop a multi-class SVM classifier on a given dataset using the one-vs-one strategy.**

### Answer

**Pipeline:**
1. Load multi-class dataset
2. Configure SVC with one-vs-one (default)
3. Train and evaluate
4. Show class predictions

**Code:**
```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load multi-class data
X, y = load_iris(return_X_y=True)
class_names = ['setosa', 'versicolor', 'virginica']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVC uses One-vs-One by default for multi-class
# For 3 classes: trains 3*(3-1)/2 = 3 binary classifiers
svm_ovo = SVC(kernel='rbf', C=1.0, decision_function_shape='ovo')
svm_ovo.fit(X_train, y_train)

# Predictions
y_pred = svm_ovo.predict(X_test)

# Evaluation
print("One-vs-One SVM Results:")
print(f"Accuracy: {np.mean(y_pred == y_test):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Compare with One-vs-Rest
svm_ovr = SVC(kernel='rbf', C=1.0, decision_function_shape='ovr')
svm_ovr.fit(X_train, y_train)
y_pred_ovr = svm_ovr.predict(X_test)
print(f"\nOne-vs-Rest Accuracy: {np.mean(y_pred_ovr == y_test):.2f}")

# Number of support vectors per class
print(f"\nSupport vectors per class: {svm_ovo.n_support_}")
```

**Key Points:**
- OvO trains k(k-1)/2 classifiers for k classes
- OvO is sklearn SVC default
- For many classes, OvR may be faster
- `decision_function_shape='ovo'` or `'ovr'` to switch

---

## Question 5
- [ ] Done

**Use Python to demonstrate the impact of different kernels on SVM decision boundaries with a 2D dataset.**

### Answer

**Pipeline:**
1. Create 2D dataset
2. Train SVM with different kernels
3. Plot decision boundaries
4. Compare visually

**Code:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# Create non-linear 2D data
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define kernels to compare
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# Function to plot decision boundary
def plot_decision_boundary(ax, clf, X, y, title):
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Predict on mesh
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='black')
    ax.set_title(f'{title}\nAccuracy: {clf.score(X, y):.2f}')

# Plot all kernels
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, kernel in enumerate(kernels):
    if kernel == 'poly':
        clf = SVC(kernel=kernel, degree=3, C=1.0)
    elif kernel == 'sigmoid':
        clf = SVC(kernel=kernel, C=1.0, gamma='scale')
    else:
        clf = SVC(kernel=kernel, C=1.0)
    
    clf.fit(X, y)
    plot_decision_boundary(axes[idx], clf, X, y, f'{kernel.upper()} Kernel')

plt.tight_layout()
plt.savefig('svm_kernels_comparison.png')
plt.show()
```

**Key Observations:**
- **Linear**: Straight line boundary, fails for non-linear data
- **Polynomial**: Curved boundary, degree controls complexity
- **RBF**: Flexible, captures complex patterns
- **Sigmoid**: Similar to neural network, less common

**Key Points:**
- RBF is default and works well for most non-linear problems
- Linear is fastest but limited to linear separation
- Always visualize boundaries when possible for 2D data

---

## Question 6
- [ ] Done

**Implement an SVM in Python using a stochastic gradient descent approach.**

### Answer

**Pipeline:**
1. Use SGDClassifier with hinge loss
2. This is equivalent to linear SVM with SGD optimization
3. Supports partial_fit for online learning

**Code:**
```python
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SGD-based SVM (hinge loss = SVM)
sgd_svm = SGDClassifier(
    loss='hinge',           # Hinge loss = SVM
    penalty='l2',           # L2 regularization
    alpha=0.0001,           # Regularization strength (1/C)
    max_iter=1000,
    tol=1e-3,
    random_state=42
)

# Train
sgd_svm.fit(X_train, y_train)

# Evaluate
y_pred = sgd_svm.predict(X_test)
print(f"SGD-SVM Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Online learning with partial_fit
sgd_svm_online = SGDClassifier(loss='hinge', random_state=42)

# Simulate streaming data
batch_size = 100
classes = np.unique(y)

for i in range(0, len(X_train), batch_size):
    X_batch = X_train[i:i+batch_size]
    y_batch = y_train[i:i+batch_size]
    sgd_svm_online.partial_fit(X_batch, y_batch, classes=classes)

print(f"Online SGD-SVM Accuracy: {accuracy_score(y_test, sgd_svm_online.predict(X_test)):.4f}")

# Access weights and bias
print(f"\nWeight shape: {sgd_svm.coef_.shape}")
print(f"Bias: {sgd_svm.intercept_[0]:.4f}")
```

**Key Points:**
- `loss='hinge'` makes SGDClassifier equivalent to linear SVM
- Much faster than SVC for large datasets
- Supports `partial_fit` for online/incremental learning
- `alpha` = regularization (equivalent to 1/C in SVC)
- Only works for linear kernel

**Comparison:**
| Aspect | SVC | SGDClassifier |
|--------|-----|---------------|
| Kernel support | All | Linear only |
| Large datasets | Slow | Fast |
| Online learning | No | Yes (partial_fit) |

---

## Question 7
- [ ] Done

**Write a script to visualize support vectors in a trained SVM model.**

### Answer

**Pipeline:**
1. Train SVM on 2D data
2. Extract support vectors from model
3. Plot data points with support vectors highlighted
4. Show decision boundary and margins

**Code:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Create 2D data for visualization
X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.5)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)

# Extract support vectors
support_vectors = svm.support_vectors_
support_indices = svm.support_

print(f"Number of support vectors: {len(support_vectors)}")
print(f"Support vectors per class: {svm.n_support_}")

# Plotting
plt.figure(figsize=(10, 8))

# Plot all data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k', label='Data points')

# Highlight support vectors
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
            s=200, facecolors='none', edgecolors='green', linewidths=2,
            label=f'Support Vectors (n={len(support_vectors)})')

# Plot decision boundary and margins
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid
xx = np.linspace(xlim[0], xlim[1], 50)
yy = np.linspace(ylim[0], ylim[1], 50)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function(xy).reshape(XX.shape)

# Plot decision boundary (Z=0) and margins (Z=-1, Z=1)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], 
           linestyles=['--', '-', '--'], alpha=0.7)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary with Support Vectors')
plt.legend()
plt.savefig('support_vectors_visualization.png')
plt.show()

# Print support vector details
print("\nSupport Vector Details:")
for i, sv in enumerate(support_vectors):
    print(f"SV {i+1}: {sv}, Class: {y[support_indices[i]]}")
```

**Key Points:**
- `svm.support_vectors_`: Actual support vector coordinates
- `svm.support_`: Indices of support vectors in training data
- `svm.n_support_`: Number of SVs per class
- `svm.dual_coef_`: Alpha values (Lagrange multipliers × y)
- Solid line = decision boundary, dashed lines = margins

---

## Question 8
- [ ] Done

**Create a Python function for grid search optimization to find the best kernel and its parameters for an SVM.**

### Answer

**Pipeline:**
1. Define parameter grid (kernel + kernel-specific params)
2. Use GridSearchCV with cross-validation
3. Find best combination
4. Return best model

**Code:**
```python
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def find_best_svm(X, y, cv=5):
    """
    Find best kernel and parameters for SVM using grid search.
    Returns: best model, best params, best score
    """
    # Define parameter grid for different kernels
    param_grid = [
        # Linear kernel
        {
            'kernel': ['linear'],
            'C': [0.1, 1, 10, 100]
        },
        # RBF kernel
        {
            'kernel': ['rbf'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1]
        },
        # Polynomial kernel
        {
            'kernel': ['poly'],
            'C': [0.1, 1, 10],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto']
        }
    ]
    
    # Grid search
    svm = SVC()
    grid_search = GridSearchCV(
        svm, 
        param_grid, 
        cv=cv, 
        scoring='accuracy',
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    # Results
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

# Usage
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Find best SVM
best_model, best_params, best_score = find_best_svm(X_train_scaled, y_train)

# Evaluate on test set
test_accuracy = best_model.score(X_test_scaled, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Show top 5 parameter combinations
import pandas as pd
results_df = pd.DataFrame(grid_search.cv_results_)
top5 = results_df.nsmallest(5, 'rank_test_score')[['params', 'mean_test_score', 'std_test_score']]
print("\nTop 5 Parameter Combinations:")
print(top5)
```

**Key Points:**
- Use list of dicts for different kernels with different params
- `n_jobs=-1` parallelizes search
- Always scale before grid search
- Consider RandomizedSearchCV for large parameter spaces
- RBF kernel often wins for non-linear problems

---

## Question 9
- [ ] Done

**How would you implement an anomaly detection system using a one-class SVM?**

### Answer

**Pipeline:**
1. Train on normal data only (no anomalies)
2. One-Class SVM learns boundary around normal points
3. New points outside boundary = anomalies
4. Returns +1 for normal, -1 for anomaly

**Code:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Generate normal data (training - only normal samples)
np.random.seed(42)
X_normal = np.random.randn(200, 2) * 0.5 + [2, 2]

# Generate test data (mix of normal and anomalies)
X_test_normal = np.random.randn(50, 2) * 0.5 + [2, 2]
X_anomalies = np.random.uniform(low=-2, high=6, size=(20, 2))
X_test = np.vstack([X_test_normal, X_anomalies])
y_test = np.array([1]*50 + [-1]*20)  # 1=normal, -1=anomaly

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_normal)
X_test_scaled = scaler.transform(X_test)

# Train One-Class SVM
oc_svm = OneClassSVM(
    kernel='rbf',
    gamma='scale',    # Controls boundary tightness
    nu=0.05           # Expected fraction of outliers (~5%)
)
oc_svm.fit(X_train_scaled)

# Predict
y_pred = oc_svm.predict(X_test_scaled)

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix

print("Anomaly Detection Results:")
print(classification_report(y_test, y_pred, target_names=['Anomaly', 'Normal']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualization
plt.figure(figsize=(10, 8))

# Create mesh for decision boundary
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = oc_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap='Blues_r', alpha=0.5)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c='blue', s=30, label='Training (Normal)')
plt.scatter(X_test_scaled[y_pred==1, 0], X_test_scaled[y_pred==1, 1], 
            c='green', s=50, marker='o', label='Predicted Normal')
plt.scatter(X_test_scaled[y_pred==-1, 0], X_test_scaled[y_pred==-1, 1], 
            c='red', s=50, marker='x', label='Predicted Anomaly')
plt.legend()
plt.title('One-Class SVM Anomaly Detection')
plt.savefig('anomaly_detection.png')
plt.show()

# Get anomaly scores
scores = oc_svm.decision_function(X_test_scaled)
print("\nSample Anomaly Scores (negative = more anomalous):")
print(f"Normal samples (first 5): {scores[:5]}")
print(f"Anomaly samples (last 5): {scores[-5:]}")
```

**Key Points:**
- **nu parameter**: Upper bound on fraction of training errors and lower bound on fraction of support vectors (set to expected anomaly rate)
- **gamma**: Controls boundary tightness (high = tight, overfitting risk)
- Train ONLY on normal data
- Output: +1 (normal), -1 (anomaly)
- `decision_function`: Distance to boundary (negative = anomaly)

**Applications:**
- Fraud detection
- Network intrusion detection
- Manufacturing defect detection
- Medical anomaly detection
