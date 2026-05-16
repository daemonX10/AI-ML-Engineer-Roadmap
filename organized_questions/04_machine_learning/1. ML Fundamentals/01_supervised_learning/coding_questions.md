# Supervised Learning Interview Questions - Coding Questions

---

## Question 1: Implement Linear Regression from Scratch
- [ ] Done

**Definition:**  
Linear regression finds the best-fit line y = wx + b by minimizing Mean Squared Error. We use Gradient Descent to iteratively update weights and bias in the direction that reduces error.

**Algorithm Steps:**
1. Initialize weights (w) and bias (b) to zeros
2. For each iteration:
   - Predict: y_pred = X @ w + b
   - Compute gradients: dw = (1/N) * X.T @ (y_pred - y), db = (1/N) * sum(y_pred - y)
   - Update: w = w - lr * dw, b = b - lr * db
3. Return learned w and b

**Python Code:**
```python
import numpy as np

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Step 1: Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Step 2: Gradient descent
        for _ in range(self.n_iter):
            # Forward pass: prediction
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# Usage
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 6, 8, 10])  # y = 2x

model = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

print(f"Weight: {model.weights[0]:.4f}")  # Should be ~2
print(f"Bias: {model.bias:.4f}")          # Should be ~0
print(f"Prediction for X=6: {model.predict([[6]])[0]:.4f}")  # Should be ~12
```

**Key Points:**
- Vectorized operations with NumPy for efficiency
- Learning rate controls step size
- More iterations = better convergence (if lr appropriate)

---

## Question 2: Logistic Regression Classifier with Scikit-learn
- [ ] Done

**Definition:**  
Logistic Regression predicts probability of binary outcome using sigmoid function. Scikit-learn provides optimized implementation with regularization options.

**Pipeline:**
1. Load/prepare data
2. Split into train/test
3. Scale features (important for convergence)
4. Train LogisticRegression
5. Evaluate with appropriate metrics

**Python Code:**
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Create sample data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                           n_classes=2, random_state=42)

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler!

# Step 4: Train model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Access coefficients
print(f"\nCoefficients shape: {model.coef_.shape}")
print(f"Intercept: {model.intercept_[0]:.4f}")
```

**Interview Tip:** Always scale features for logistic regression; fit scaler on train only.

---

## Question 3: Decision Tree Classification and Visualization
- [ ] Done

**Definition:**  
Decision trees learn if-else rules from data. Visualization shows the learned decision rules at each node, making them highly interpretable.

**Pipeline:**
1. Load dataset (Iris is classic)
2. Train DecisionTreeClassifier
3. Visualize with plot_tree

**Python Code:**
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Load data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Step 2: Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 3: Train decision tree
tree = DecisionTreeClassifier(
    max_depth=3,           # Limit depth to prevent overfitting
    min_samples_split=5,   # Minimum samples to split a node
    random_state=42
)
tree.fit(X_train, y_train)

# Step 4: Evaluate
print(f"Train Accuracy: {tree.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {tree.score(X_test, y_test):.4f}")

# Step 5: Visualize
plt.figure(figsize=(15, 10))
plot_tree(
    tree,
    filled=True,              # Color nodes by class
    feature_names=feature_names,
    class_names=class_names,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree for Iris Classification")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150)
plt.show()

# Feature importance
print("\nFeature Importances:")
for name, importance in zip(feature_names, tree.feature_importances_):
    print(f"  {name}: {importance:.4f}")
```

**Reading the Tree:**
- Each node shows: split condition, gini impurity, samples, class distribution
- Left branch: condition is True
- Right branch: condition is False
- Deeper color = more pure

---

## Question 4: Feedforward Neural Network with Keras
- [ ] Done

**Definition:**  
A feedforward (dense) neural network passes data through layers of neurons with activation functions. Keras provides high-level API for building and training.

**Pipeline:**
1. Load and preprocess data (normalize)
2. Define model architecture (Sequential)
3. Compile (optimizer, loss, metrics)
4. Train (fit)
5. Evaluate

**Python Code:**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Step 1: Load and preprocess data
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Step 2: Define model
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),    # 28x28 -> 784
    layers.Dense(128, activation='relu'),     # Hidden layer
    layers.Dropout(0.2),                      # Regularization
    layers.Dense(64, activation='relu'),      # Hidden layer
    layers.Dense(10)                          # Output (10 classes, logits)
])

# Step 3: Compile
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# Step 4: Train
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Step 5: Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Make prediction
probability_model = keras.Sequential([model, layers.Softmax()])
predictions = probability_model.predict(x_test[:5])

for i in range(5):
    pred_class = np.argmax(predictions[i])
    true_class = y_test[i]
    print(f"Predicted: {class_names[pred_class]}, Actual: {class_names[true_class]}")
```

**Key Points:**
- `from_logits=True` because output layer has no activation
- Dropout for regularization
- `SparseCategoricalCrossentropy` for integer labels

---

## Question 5: Compute F1 Score from Confusion Matrix
- [ ] Done

**Definition:**  
F1 Score = 2 * (Precision * Recall) / (Precision + Recall). Given a confusion matrix, extract TP, FP, FN for each class and compute metrics.

**Formulas:**
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 * Precision * Recall / (Precision + Recall)

**Python Code:**
```python
import numpy as np

def compute_f1_from_confusion_matrix(cm):
    """
    Compute precision, recall, F1 for each class from confusion matrix.
    
    Args:
        cm: numpy array, shape (n_classes, n_classes)
            cm[i, j] = samples with true label i predicted as j
    
    Returns:
        dict with per-class and averaged metrics
    """
    n_classes = cm.shape[0]
    
    precision_list = []
    recall_list = []
    f1_list = []
    support_list = []
    
    for i in range(n_classes):
        # True Positives: diagonal element
        TP = cm[i, i]
        
        # False Positives: column sum minus TP (predicted as i, but not i)
        FP = np.sum(cm[:, i]) - TP
        
        # False Negatives: row sum minus TP (actually i, but not predicted as i)
        FN = np.sum(cm[i, :]) - TP
        
        # Support: actual samples in class i
        support = TP + FN
        
        # Calculate metrics (handle division by zero)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        support_list.append(support)
        
        print(f"Class {i}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    # Macro average (unweighted mean)
    macro_f1 = np.mean(f1_list)
    
    # Weighted average (weighted by support)
    total_support = np.sum(support_list)
    weighted_f1 = np.sum([f1 * s for f1, s in zip(f1_list, support_list)]) / total_support
    
    print(f"\nMacro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    return {
        'precision': precision_list,
        'recall': recall_list,
        'f1': f1_list,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }


# Example usage
cm = np.array([
    [50, 2, 3],    # Class 0: 50 correct, 2 predicted as 1, 3 predicted as 2
    [5, 45, 5],    # Class 1: 5 predicted as 0, 45 correct, 5 predicted as 2
    [2, 8, 40]     # Class 2: 2 predicted as 0, 8 predicted as 1, 40 correct
])

print("Confusion Matrix:")
print(cm)
print()

metrics = compute_f1_from_confusion_matrix(cm)

# Verify with sklearn
print("\n--- Sklearn Verification ---")
from sklearn.metrics import classification_report

# Reconstruct labels from confusion matrix
y_true = [0]*55 + [1]*55 + [2]*50
y_pred = ([0]*50 + [1]*2 + [2]*3 + 
          [0]*5 + [1]*45 + [2]*5 + 
          [0]*2 + [1]*8 + [2]*40)

print(classification_report(y_true, y_pred))
```

**Output Structure:**
```
Class 0: Precision=0.8772, Recall=0.9091, F1=0.8929
Class 1: Precision=0.8182, Recall=0.8182, F1=0.8182
Class 2: Precision=0.8333, Recall=0.8000, F1=0.8163

Macro F1: 0.8425
Weighted F1: 0.8437
```

**Key Points:**
- TP is diagonal element cm[i,i]
- FP is column sum minus TP
- FN is row sum minus TP
- Handle division by zero for edge cases

---
