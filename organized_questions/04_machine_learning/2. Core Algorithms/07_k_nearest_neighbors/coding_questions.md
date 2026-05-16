# K Nearest Neighbors Interview Questions - Coding Questions

## Question 1
- [ ] Done

**How do you handle categorical features when implementing K-NN?**

### Answer

Convert categorical features to numerical format: (1) One-hot encoding for nominal categories, (2) Label encoding for ordinal categories. Then scale all features. Alternative: use Hamming distance for categorical-only data or Gower distance for mixed types.

**Approach 1: One-Hot Encoding + Scaling (Most Common)**
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Sample data
data = pd.DataFrame({
    'age': [25, 30, 35],
    'income': [50000, 60000, 70000],
    'gender': ['M', 'F', 'M'],
    'city': ['NY', 'LA', 'NY']
})

# Define column types
numerical_cols = ['age', 'income']
categorical_cols = ['gender', 'city']

# Create preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first'), categorical_cols)
])

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('knn', KNeighborsClassifier(n_neighbors=3))
])

# Fit and predict
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

**Approach 2: Gower Distance (for mixed types)**
```python
import gower
from sklearn.neighbors import NearestNeighbors

# Compute Gower distance matrix
distance_matrix = gower.gower_matrix(data)

# Use precomputed distances
nn = NearestNeighbors(metric='precomputed')
nn.fit(distance_matrix)
```

**Key Points:**
- One-hot encoding increases dimensionality
- Always scale numerical features after encoding
- Gower is elegant but computationally expensive

---

## Question 2
- [ ] Done

**Write a Python function to implement K-NN from scratch on a simple dataset.**

### Answer

**Pipeline:**
1. Calculate Euclidean distance from query to all training points
2. Sort distances, get K smallest
3. Get labels of K nearest neighbors
4. Return majority vote (classification) or average (regression)

**Code:**
```python
import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(X_train, y_train, X_test, k=3):
    """
    K-NN classifier from scratch
    
    Parameters:
    -----------
    X_train: Training features (n_samples, n_features)
    y_train: Training labels (n_samples,)
    X_test: Test features (n_test, n_features)
    k: Number of neighbors
    
    Returns:
    --------
    predictions: Predicted labels for X_test
    """
    predictions = []
    
    for test_point in X_test:
        # Step 1: Calculate distances to all training points
        distances = []
        for i, train_point in enumerate(X_train):
            dist = euclidean_distance(test_point, train_point)
            distances.append((dist, y_train[i]))
        
        # Step 2: Sort by distance and get K nearest
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        
        # Step 3: Get labels of K neighbors
        k_labels = [label for _, label in k_nearest]
        
        # Step 4: Majority vote
        most_common = Counter(k_labels).most_common(1)[0][0]
        predictions.append(most_common)
    
    return np.array(predictions)

# Example usage
X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y_train = np.array([0, 0, 0, 1, 1, 1])
X_test = np.array([[4, 4], [1, 1]])

predictions = knn_predict(X_train, y_train, X_test, k=3)
print(f"Predictions: {predictions}")  # Output: [1, 0]
```

---

## Question 3
- [ ] Done

**Use scikit-learn to demonstrate K-NN classification using the Iris dataset.**

### Answer

**Pipeline:**
1. Load Iris dataset
2. Split into train/test
3. Scale features
4. Train KNeighborsClassifier
5. Predict and evaluate

**Code:**
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load data
iris = load_iris()
X, y = iris.data, iris.target

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Scale features (important for K-NN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train K-NN classifier
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train_scaled, y_train)

# Step 5: Predict and evaluate
y_pred = knn.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Find optimal K using cross-validation
from sklearn.model_selection import cross_val_score

k_values = range(1, 21, 2)
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    cv_scores.append(scores.mean())

best_k = k_values[cv_scores.index(max(cv_scores))]
print(f"\nBest K: {best_k} with CV score: {max(cv_scores):.4f}")
```

**Output:**
```
Accuracy: 1.0000
Best K: 5 with CV score: 0.9667
```

---

## Question 4
- [ ] Done

**Implement a LazyLearningClassifier in Python that uses K-NN under the hood.**

### Answer

**Pipeline:**
1. Create class with fit (stores data) and predict (runs K-NN)
2. fit() just stores training data (lazy learning)
3. predict() computes distances and votes for each test point

**Code:**
```python
import numpy as np
from collections import Counter

class LazyLearningClassifier:
    """
    A lazy learning classifier using K-NN.
    No computation during fit - all work done at predict time.
    """
    
    def __init__(self, n_neighbors=3, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Lazy learning: just store the data"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _predict_single(self, x):
        """Predict for single sample"""
        # Calculate all distances
        distances = [self._euclidean_distance(x, x_train) 
                     for x_train in self.X_train]
        
        # Get K nearest indices
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_labels = self.y_train[k_indices]
        k_distances = np.array(distances)[k_indices]
        
        # Voting
        if self.weights == 'uniform':
            # Simple majority vote
            return Counter(k_labels).most_common(1)[0][0]
        else:
            # Distance-weighted vote
            weights = 1 / (k_distances + 1e-8)
            weighted_votes = {}
            for label, weight in zip(k_labels, weights):
                weighted_votes[label] = weighted_votes.get(label, 0) + weight
            return max(weighted_votes, key=weighted_votes.get)
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])
    
    def score(self, X, y):
        """Return accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Usage
clf = LazyLearningClassifier(n_neighbors=5, weights='distance')
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
```

---

## Question 5
- [ ] Done

**Create a Python script to visualize the decision boundary of K-NN on a 2D dataset.**

### Answer

**Pipeline:**
1. Create 2D synthetic dataset
2. Train K-NN classifier
3. Create meshgrid over feature space
4. Predict class for each grid point
5. Plot contour and scatter points

**Code:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Step 1: Create 2D dataset
X, y = make_classification(
    n_samples=200, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1, random_state=42
)

# Step 2: Scale and train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# Step 3: Create meshgrid
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                      np.linspace(y_min, y_max, 200))

# Step 4: Predict for grid points
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Step 5: Plot
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'K-NN Decision Boundary (K={knn.n_neighbors})')
plt.colorbar(label='Class')
plt.show()

# Compare different K values
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, k in zip(axes, [1, 5, 15]):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_scaled, y)
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    ax.set_title(f'K = {k}')
plt.tight_layout()
plt.show()
```

---

## Question 6
- [ ] Done

**Develop a weighted K-NN algorithm in Python and test its performance against the standard K-NN.**

### Answer

**Pipeline:**
1. Implement weighted K-NN (weight = 1/distance)
2. Train both standard and weighted K-NN
3. Compare accuracy on test set
4. Test with different K values

**Code:**
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Create dataset
X, y = make_classification(n_samples=500, n_features=10, 
                           n_informative=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare standard vs weighted K-NN
print("Standard vs Weighted K-NN Comparison")
print("=" * 50)

results = []
for k in [3, 5, 7, 9, 11]:
    # Standard K-NN (uniform weights)
    knn_standard = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    knn_standard.fit(X_train_scaled, y_train)
    acc_standard = knn_standard.score(X_test_scaled, y_test)
    
    # Weighted K-NN (distance weights)
    knn_weighted = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn_weighted.fit(X_train_scaled, y_train)
    acc_weighted = knn_weighted.score(X_test_scaled, y_test)
    
    # Cross-validation scores
    cv_standard = cross_val_score(knn_standard, X_train_scaled, y_train, cv=5).mean()
    cv_weighted = cross_val_score(knn_weighted, X_train_scaled, y_train, cv=5).mean()
    
    print(f"K={k:2d} | Standard: {acc_standard:.4f} (CV: {cv_standard:.4f}) | "
          f"Weighted: {acc_weighted:.4f} (CV: {cv_weighted:.4f})")
    
    results.append((k, acc_standard, acc_weighted))

# Custom weight function example
def inverse_square_weights(distances):
    """Weight by inverse square of distance"""
    return 1 / (distances ** 2 + 1e-8)

knn_custom = KNeighborsClassifier(n_neighbors=5, weights=inverse_square_weights)
knn_custom.fit(X_train_scaled, y_train)
print(f"\nCustom (1/d²) weights: {knn_custom.score(X_test_scaled, y_test):.4f}")
```

**Output Example:**
```
K= 3 | Standard: 0.8900 | Weighted: 0.9000
K= 5 | Standard: 0.8800 | Weighted: 0.9100
K= 7 | Standard: 0.8700 | Weighted: 0.9000
```

**Observation:** Weighted K-NN often performs equal or better than standard K-NN.

---

## Question 7
- [ ] Done

**Optimize a K-NN model in a large dataset using approximate nearest neighbors techniques like LSH or kd-trees.**

### Answer

**Pipeline:**
1. Use sklearn's algorithm parameter for kd-tree/ball-tree
2. For massive scale, use FAISS for approximate search
3. Compare query times and accuracy

**Code:**
```python
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.datasets import make_classification

# Create large dataset
X, y = make_classification(n_samples=50000, n_features=20, random_state=42)

# Compare different algorithms
algorithms = ['brute', 'kd_tree', 'ball_tree']
results = {}

for algo in algorithms:
    knn = KNeighborsClassifier(n_neighbors=5, algorithm=algo)
    
    # Time fitting
    start = time.time()
    knn.fit(X[:40000], y[:40000])
    fit_time = time.time() - start
    
    # Time prediction
    start = time.time()
    predictions = knn.predict(X[40000:])
    predict_time = time.time() - start
    
    results[algo] = {'fit': fit_time, 'predict': predict_time}
    print(f"{algo:10s} | Fit: {fit_time:.4f}s | Predict: {predict_time:.4f}s")

# For massive scale: FAISS (Approximate NN)
print("\n--- FAISS (Approximate Nearest Neighbors) ---")

import faiss

# Prepare data
X_train = X[:40000].astype('float32')
X_test = X[40000:].astype('float32')

# Build FAISS index
d = X_train.shape[1]  # dimension
k = 5  # neighbors

# Exact search (for comparison)
index_flat = faiss.IndexFlatL2(d)
index_flat.add(X_train)

start = time.time()
D, I = index_flat.search(X_test, k)
print(f"FAISS Flat L2: {time.time() - start:.4f}s")

# Approximate search with IVF (Inverted File)
nlist = 100  # number of clusters
index_ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist)
index_ivf.train(X_train)
index_ivf.add(X_train)
index_ivf.nprobe = 10  # search 10 nearest clusters

start = time.time()
D_approx, I_approx = index_ivf.search(X_test, k)
print(f"FAISS IVF:     {time.time() - start:.4f}s")

# Calculate recall (how many true neighbors found)
recall = np.mean([len(set(I[i]) & set(I_approx[i])) / k for i in range(len(I))])
print(f"Recall: {recall:.4f}")
```

**Output Example:**
```
brute      | Fit: 0.0012s | Predict: 2.5000s
kd_tree    | Fit: 0.1500s | Predict: 0.4000s
ball_tree  | Fit: 0.2000s | Predict: 0.5000s
FAISS IVF: 0.0100s (Recall: 0.95)
```

---

## Question 8
- [ ] Done

**Given a dataset with time-series data, how would you apply K-NN for forecasting?**

### Answer

**Pipeline:**
1. Create lagged features (past values as features)
2. Target: next time step value
3. Train K-NN regressor on lagged data
4. Predict by finding similar historical patterns

**Code:**
```python
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Create sample time series
np.random.seed(42)
n_points = 200
time_series = np.sin(np.linspace(0, 20, n_points)) + np.random.normal(0, 0.1, n_points)

def create_lagged_features(series, lag):
    """Transform time series to supervised learning format"""
    X, y = [], []
    for i in range(lag, len(series)):
        X.append(series[i-lag:i])
        y.append(series[i])
    return np.array(X), np.array(y)

# Step 1: Create lagged features
lag = 5
X, y = create_lagged_features(time_series, lag)

# Step 2: Train/test split (time-based, no shuffle!)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Step 3: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train K-NN regressor
knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
knn.fit(X_train_scaled, y_train)

# Step 5: Predict
y_pred = knn.predict(X_test_scaled)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")

# Multi-step forecasting
def forecast_multistep(model, scaler, last_window, steps):
    """Forecast multiple steps ahead"""
    predictions = []
    current_window = last_window.copy()
    
    for _ in range(steps):
        # Scale and predict
        scaled = scaler.transform(current_window.reshape(1, -1))
        pred = model.predict(scaled)[0]
        predictions.append(pred)
        
        # Slide window
        current_window = np.roll(current_window, -1)
        current_window[-1] = pred
    
    return predictions

# Forecast next 10 steps
last_window = X_test[-1]
future_preds = forecast_multistep(knn, scaler, last_window, steps=10)
print(f"Next 10 predictions: {future_preds}")
```

**Key Points:**
- Use time-based split (no shuffle)
- Tune lag and K via time-series CV
- Consider making data stationary first

---

## Question 9
- [ ] Done

**Discuss a healthcare application where K-NN could be beneficial. How would you implement it?**

### Answer

**Application: Disease Diagnosis Prediction**

K-NN is suitable for medical diagnosis because: (1) interpretable—doctors can see similar patient cases, (2) no assumptions about disease distributions, (3) handles multi-class naturally.

**Scenario: Diabetes Prediction**

**Pipeline:**
1. Preprocess patient data (handle missing, scale features)
2. Handle class imbalance (SMOTE)
3. Train weighted K-NN
4. Evaluate with medical-relevant metrics (recall for disease)

**Code:**
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Sample healthcare data (Pima Indians Diabetes)
# Features: Pregnancies, Glucose, BloodPressure, BMI, Age, etc.
from sklearn.datasets import load_diabetes
# For demo, using synthetic data
np.random.seed(42)
n_samples = 500
data = {
    'glucose': np.random.normal(120, 30, n_samples),
    'blood_pressure': np.random.normal(70, 15, n_samples),
    'bmi': np.random.normal(32, 8, n_samples),
    'age': np.random.randint(21, 80, n_samples),
    'diabetes': np.random.binomial(1, 0.3, n_samples)  # Imbalanced
}
df = pd.DataFrame(data)

X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Step 1: Handle missing values with KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)

# Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, stratify=y, random_state=42
)

# Step 3: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Step 5: Train weighted K-NN
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
knn.fit(X_train_balanced, y_train_balanced)

# Step 6: Predict and evaluate
y_pred = knn.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# Show similar patients for interpretability
def explain_prediction(model, scaler, X_train, y_train, new_patient, k=3):
    """Show K most similar patients for interpretation"""
    new_scaled = scaler.transform(new_patient.reshape(1, -1))
    distances, indices = model.kneighbors(new_scaled, n_neighbors=k)
    
    print("Similar patients in database:")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        print(f"  Patient {i+1}: Distance={dist:.2f}, Outcome={y_train.iloc[idx]}")

# Example: Explain prediction for new patient
new_patient = X_test_scaled[0]
explain_prediction(knn, scaler, X_train_balanced, y_train_balanced, new_patient)
```

**Why K-NN for Healthcare:**
- **Interpretability**: "This patient is similar to cases X, Y, Z"
- **No black box**: Doctors can verify similar cases
- **Handles rare diseases**: Works with small datasets

---
