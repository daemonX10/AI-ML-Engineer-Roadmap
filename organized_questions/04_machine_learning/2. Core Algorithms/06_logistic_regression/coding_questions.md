# Logistic Regression Interview Questions - Coding Questions

## Question 1
- [ ] Done

**How would you implement class-weighting in logistic regression?**

### Answer

**Definition:**
Class weighting adjusts the loss function to penalize misclassifications of minority classes more heavily. This addresses class imbalance without modifying the data.

**Implementation:**
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Create imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], 
                           n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Method 1: Automatic balanced weighting
model_balanced = LogisticRegression(class_weight='balanced')
model_balanced.fit(X_train, y_train)

# Method 2: Manual class weights
# Weight = n_samples / (n_classes * n_samples_per_class)
class_counts = np.bincount(y_train)
weights = {0: len(y_train) / (2 * class_counts[0]),
           1: len(y_train) / (2 * class_counts[1])}
model_manual = LogisticRegression(class_weight=weights)
model_manual.fit(X_train, y_train)

# Method 3: Custom weights based on cost
# If false negative costs 10x more than false positive
model_custom = LogisticRegression(class_weight={0: 1, 1: 10})
model_custom.fit(X_train, y_train)

# Compare results
print("Balanced weights:")
print(classification_report(y_test, model_balanced.predict(X_test)))

print("Manual weights:")
print(classification_report(y_test, model_manual.predict(X_test)))
```

**Interview Tip:**
Use `class_weight='balanced'` for automatic weight calculation. For business-driven weighting (different misclassification costs), use manual weights based on cost ratios.

---

## Question 2
- [ ] Done

**Code a basic logistic regression model from scratch using Numpy.**

### Answer

**Implementation:**
```python
import numpy as np

class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def sigmoid(self, z):
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y, y_pred):
        # Binary cross-entropy loss
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_pred)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Track loss
            if i % 100 == 0:
                loss = self.compute_loss(y, y_pred)
                self.losses.append(loss)
    
    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# Usage
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Generate data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features (important for gradient descent)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegressionFromScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

**Interview Tip:**
Key components to explain: sigmoid function, binary cross-entropy loss, gradient computation, and the importance of feature scaling for gradient descent convergence.

---

## Question 3
- [ ] Done

**Implement data standardization for a logistic regression model in Python.**

### Answer

**Implementation:**
```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Method 1: StandardScaler (z-score normalization)
# Transforms to mean=0, std=1
scaler_standard = StandardScaler()
X_train_standard = scaler_standard.fit_transform(X_train)
X_test_standard = scaler_standard.transform(X_test)

# Method 2: MinMaxScaler (scales to [0, 1])
scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

# Method 3: RobustScaler (robust to outliers, uses median/IQR)
scaler_robust = RobustScaler()
X_train_robust = scaler_robust.fit_transform(X_train)
X_test_robust = scaler_robust.transform(X_test)

# Best Practice: Use Pipeline to avoid data leakage
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression())
])

# Cross-validation with pipeline (scaling done properly within each fold)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Fit and evaluate
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.3f}")

# Manual standardization (educational)
def standardize_manual(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std  # Use train stats!
    return X_train_scaled, X_test_scaled
```

**Why Standardization Matters:**
- Features on same scale → fair coefficient comparison
- Faster convergence for gradient-based optimization
- Regularization affects all features equally

**Interview Tip:**
Always fit scaler on training data only, then transform both train and test. Using a Pipeline ensures this automatically and prevents data leakage.

---

## Question 4
- [ ] Done

**Write a Python function to calculate the AUC-ROC curve for a logistic regression model.**

### Answer

**Implementation:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

def calculate_and_plot_roc(model, X_test, y_test, plot=True):
    """
    Calculate AUC-ROC and optionally plot the curve.
    
    Parameters:
    -----------
    model : fitted classifier with predict_proba method
    X_test : test features
    y_test : true labels
    plot : whether to display the plot
    
    Returns:
    --------
    dict with fpr, tpr, thresholds, and auc_score
    """
    # Get predicted probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    # Calculate AUC
    auc_score = roc_auc_score(y_test, y_proba)
    
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.fill_between(fpr, tpr, alpha=0.3)
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.show()
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc_score': auc_score,
        'optimal_threshold': optimal_threshold
    }

# Manual ROC calculation (educational)
def calculate_roc_manual(y_true, y_scores):
    """Calculate ROC curve from scratch."""
    thresholds = np.sort(np.unique(y_scores))[::-1]
    tpr_list, fpr_list = [], []
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return np.array(fpr_list), np.array(tpr_list), thresholds

# Usage
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

results = calculate_and_plot_roc(model, X_test, y_test)
print(f"AUC: {results['auc_score']:.3f}")
print(f"Optimal Threshold: {results['optimal_threshold']:.3f}")
```

**Interview Tip:**
AUC-ROC measures ranking ability across all thresholds. An AUC of 0.5 means random guessing, 1.0 is perfect. For imbalanced data, also check AUC-PR.

---

## Question 5
- [ ] Done

**Given a dataset with categorical features, perform one-hot encoding and fit a logistic regression model using scikit-learn.**

### Answer

**Implementation:**
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Create sample dataset
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.randint(18, 70, 1000),
    'income': np.random.normal(50000, 15000, 1000),
    'gender': np.random.choice(['Male', 'Female'], 1000),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], 1000),
    'purchased': np.random.randint(0, 2, 1000)
})

# Define feature types
categorical_features = ['gender', 'education', 'city']
numerical_features = ['age', 'income']

X = data[categorical_features + numerical_features]
y = data['purchased']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

# Complete pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Fit and predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))

# Get feature names after encoding
feature_names = (numerical_features + 
                 list(pipeline.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .get_feature_names_out(categorical_features)))

# Get coefficients
coefficients = pipeline.named_steps['classifier'].coef_[0]

# Create coefficient dataframe
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Odds_Ratio': np.exp(coefficients)
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nFeature Coefficients:")
print(coef_df)
```

**Interview Tip:**
Use `drop='first'` to avoid multicollinearity. The dropped category becomes the reference level for interpretation. Use `handle_unknown='ignore'` for production robustness.

---

## Question 6
- [ ] Done

**Create a Python script that tunes the regularization strength (C value) for a logistic regression model using cross-validation.**

### Answer

**Implementation:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                           n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Method 1: LogisticRegressionCV (built-in cross-validation)
lr_cv = LogisticRegressionCV(
    Cs=np.logspace(-4, 4, 20),  # Range of C values
    cv=5,
    scoring='roc_auc',
    max_iter=1000,
    random_state=42
)
lr_cv.fit(X_train, y_train)
print(f"Method 1 - Best C: {lr_cv.C_[0]:.4f}")
print(f"Test Score: {lr_cv.score(X_test, y_test):.3f}")

# Method 2: GridSearchCV with Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000))
])

param_grid = {
    'lr__C': np.logspace(-4, 4, 20),
    'lr__penalty': ['l1', 'l2'],
    'lr__solver': ['saga']  # Supports both L1 and L2
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\nMethod 2 - Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.3f}")
print(f"Test Score: {grid_search.score(X_test, y_test):.3f}")

# Visualize C value impact
C_values = np.logspace(-4, 4, 20)
train_scores = []
cv_scores = []

for C in C_values:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
    cv_scores.append(cv_score)

plt.figure(figsize=(10, 6))
plt.semilogx(C_values, train_scores, 'b-', label='Train Score')
plt.semilogx(C_values, cv_scores, 'r-', label='CV Score')
plt.axvline(grid_search.best_params_['lr__C'], color='g', linestyle='--', 
            label=f'Best C = {grid_search.best_params_["lr__C"]:.4f}')
plt.xlabel('C (Regularization Strength)')
plt.ylabel('Accuracy')
plt.title('Effect of Regularization on Model Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Interview Tip:**
C is the inverse of regularization strength (higher C = less regularization). Use `LogisticRegressionCV` for quick tuning, `GridSearchCV` when also tuning other parameters.

---

## Question 7
- [ ] Done

**Write a Python function to interpret and output the model coefficients of a logistic regression in terms of odds ratios.**

### Answer

**Implementation:**
```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

def interpret_coefficients(model, feature_names, X=None, y=None):
    """
    Interpret logistic regression coefficients as odds ratios.
    
    Parameters:
    -----------
    model : fitted LogisticRegression model
    feature_names : list of feature names
    X, y : optional, for confidence interval calculation via statsmodels
    
    Returns:
    --------
    DataFrame with coefficients, odds ratios, and interpretation
    """
    # Basic interpretation
    coefficients = model.coef_[0]
    odds_ratios = np.exp(coefficients)
    
    result = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Odds_Ratio': odds_ratios,
        'Effect': ['Increases odds' if c > 0 else 'Decreases odds' for c in coefficients],
        'Percent_Change': [(or_ - 1) * 100 for or_ in odds_ratios]
    })
    
    # Add interpretation
    def interpret_or(or_):
        if or_ > 1:
            return f"{(or_ - 1) * 100:.1f}% increase in odds"
        else:
            return f"{(1 - or_) * 100:.1f}% decrease in odds"
    
    result['Interpretation'] = result['Odds_Ratio'].apply(interpret_or)
    
    return result.sort_values('Coefficient', key=abs, ascending=False)

def interpret_with_ci(X, y, feature_names, alpha=0.05):
    """
    Get odds ratios with confidence intervals using statsmodels.
    """
    # Fit with statsmodels for CI
    X_const = sm.add_constant(X)
    model = sm.Logit(y, X_const).fit(disp=0)
    
    # Get confidence intervals
    conf_int = model.conf_int(alpha=alpha)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Feature': ['Intercept'] + list(feature_names),
        'Coefficient': model.params,
        'Std_Error': model.bse,
        'z_value': model.tvalues,
        'p_value': model.pvalues,
        'Odds_Ratio': np.exp(model.params),
        'OR_CI_Lower': np.exp(conf_int[0]),
        'OR_CI_Upper': np.exp(conf_int[1])
    })
    
    # Add significance
    results['Significant'] = results['p_value'] < alpha
    
    return results

# Usage
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate data
X, y = make_classification(n_samples=1000, n_features=5, n_informative=4, 
                           random_state=42)
feature_names = [f'Feature_{i}' for i in range(5)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Fit sklearn model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Basic interpretation
print("=== Basic Interpretation ===")
print(interpret_coefficients(model, feature_names))

# With confidence intervals
print("\n=== With Confidence Intervals ===")
print(interpret_with_ci(X_train_scaled, y_train, feature_names))
```

**Interview Tip:**
Odds ratio of 1.5 means "50% increase in odds for one unit increase in the feature." Always standardize features first so coefficients are comparable across features.

---

## Question 8
- [ ] Done

**Develop a logistic regression model that handles class imbalance with weighted classes in scikit-learn.**

### Answer

**Implementation:**
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Create highly imbalanced dataset
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=10,
    n_classes=2,
    weights=[0.95, 0.05],  # 95% class 0, 5% class 1
    random_state=42
)

print(f"Class distribution: {np.bincount(y)}")
print(f"Imbalance ratio: {np.bincount(y)[0] / np.bincount(y)[1]:.1f}:1")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Model 1: No weighting (baseline)
model_baseline = LogisticRegression(max_iter=1000)
model_baseline.fit(X_train, y_train)
y_pred_baseline = model_baseline.predict(X_test)

print("\n=== Baseline (No Weighting) ===")
print(classification_report(y_test, y_pred_baseline))

# Model 2: Balanced class weights
model_balanced = LogisticRegression(class_weight='balanced', max_iter=1000)
model_balanced.fit(X_train, y_train)
y_pred_balanced = model_balanced.predict(X_test)

print("\n=== Balanced Class Weights ===")
print(classification_report(y_test, y_pred_balanced))

# Model 3: Custom weights (cost-sensitive)
# If false negative costs 20x more than false positive
cost_ratio = 20
model_custom = LogisticRegression(class_weight={0: 1, 1: cost_ratio}, max_iter=1000)
model_custom.fit(X_train, y_train)
y_pred_custom = model_custom.predict(X_test)

print(f"\n=== Custom Weights (1:{cost_ratio}) ===")
print(classification_report(y_test, y_pred_custom))

# Model 4: Optimal weights via cross-validation
def find_optimal_weight(X_train, y_train, weight_range, metric='f1'):
    """Find optimal class weight via CV."""
    from sklearn.metrics import make_scorer, f1_score
    
    best_weight = 1
    best_score = 0
    
    for weight in weight_range:
        model = LogisticRegression(class_weight={0: 1, 1: weight}, max_iter=1000)
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=StratifiedKFold(5), 
            scoring='f1'
        )
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_weight = weight
    
    return best_weight, best_score

weight_range = np.arange(1, 50, 2)
optimal_weight, optimal_score = find_optimal_weight(X_train, y_train, weight_range)
print(f"\nOptimal weight for minority class: {optimal_weight}")
print(f"Best CV F1 score: {optimal_score:.3f}")

# Compare models
models = {
    'Baseline': model_baseline,
    'Balanced': model_balanced,
    f'Custom (1:{cost_ratio})': model_custom
}

print("\n=== Model Comparison ===")
print(f"{'Model':<20} {'Recall(1)':<12} {'Precision(1)':<14} {'F1(1)':<10} {'AUC-ROC':<10}")
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    print(f"{name:<20} {report['1']['recall']:<12.3f} {report['1']['precision']:<14.3f} "
          f"{report['1']['f1-score']:<10.3f} {auc:<10.3f}")
```

**Interview Tip:**
`class_weight='balanced'` is a good starting point. For business applications, use custom weights based on the cost of different types of errors.

---

## Question 9
- [ ] Done

**Implement a multi-class logistic regression model in TensorFlow/Keras.**

### Answer

**Implementation:**
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to one-hot encoding for training
n_classes = len(np.unique(y))
y_train_onehot = keras.utils.to_categorical(y_train, n_classes)
y_test_onehot = keras.utils.to_categorical(y_test, n_classes)

# Method 1: Simple Dense Layer (equivalent to multinomial logistic regression)
model_simple = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(n_classes, activation='softmax')
])

model_simple.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model_simple.fit(
    X_train, y_train_onehot,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Evaluate
y_pred_proba = model_simple.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

print("=== Simple Softmax Model ===")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Method 2: With regularization (equivalent to regularized logistic regression)
model_regularized = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(
        n_classes, 
        activation='softmax',
        kernel_regularizer=keras.regularizers.l2(0.01)
    )
])

model_regularized.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_regularized.fit(
    X_train, y_train_onehot,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Get coefficients (weights)
weights, biases = model_regularized.layers[0].get_weights()
print("\n=== Model Weights (Coefficients) ===")
print(f"Shape: {weights.shape}")  # (n_features, n_classes)

# Method 3: Using sparse categorical crossentropy (no one-hot needed)
model_sparse = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(n_classes, activation='softmax')
])

model_sparse.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Use original labels
    metrics=['accuracy']
)

model_sparse.fit(
    X_train, y_train,  # Original labels, not one-hot
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Custom training loop (for more control)
class LogisticRegressionTF(keras.Model):
    def __init__(self, n_classes):
        super().__init__()
        self.dense = keras.layers.Dense(n_classes, activation='softmax')
    
    def call(self, inputs):
        return self.dense(inputs)

# Usage
model_custom = LogisticRegressionTF(n_classes)
model_custom.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model_custom.fit(X_train, y_train, epochs=100, verbose=0)

print("\n=== Custom Model ===")
print(f"Test Accuracy: {model_custom.evaluate(X_test, y_test, verbose=0)[1]:.3f}")
```

**Interview Tip:**
A single Dense layer with softmax activation is mathematically equivalent to multinomial logistic regression. Use `sparse_categorical_crossentropy` when labels are integers.

---

## Question 10
- [ ] Done

**Code a Python function to perform stepwise regression using the logistic regression model.**

### Answer

**Implementation:**
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm

def forward_stepwise_selection(X, y, feature_names, scoring='roc_auc', cv=5, threshold=0.05):
    """
    Forward stepwise selection for logistic regression.
    
    Parameters:
    -----------
    X : feature matrix
    y : target variable
    feature_names : list of feature names
    scoring : metric for cross-validation
    cv : number of CV folds
    threshold : p-value threshold for inclusion
    
    Returns:
    --------
    List of selected features
    """
    selected = []
    remaining = list(range(X.shape[1]))
    current_score = 0
    
    while remaining:
        scores = []
        for feature in remaining:
            candidate = selected + [feature]
            X_subset = X[:, candidate]
            
            # Use cross-validation score
            model = LogisticRegression(max_iter=1000)
            score = cross_val_score(model, X_subset, y, cv=cv, scoring=scoring).mean()
            scores.append((feature, score))
        
        # Find best feature
        best_feature, best_score = max(scores, key=lambda x: x[1])
        
        # Check if improvement is significant
        if best_score > current_score + 0.001:  # Minimum improvement
            selected.append(best_feature)
            remaining.remove(best_feature)
            current_score = best_score
            print(f"Added {feature_names[best_feature]}: CV Score = {best_score:.4f}")
        else:
            break
    
    return [feature_names[i] for i in selected], selected

def backward_stepwise_selection(X, y, feature_names, scoring='roc_auc', cv=5):
    """
    Backward stepwise selection for logistic regression.
    """
    selected = list(range(X.shape[1]))
    
    while len(selected) > 1:
        scores = []
        for feature in selected:
            candidate = [f for f in selected if f != feature]
            X_subset = X[:, candidate]
            
            model = LogisticRegression(max_iter=1000)
            score = cross_val_score(model, X_subset, y, cv=cv, scoring=scoring).mean()
            scores.append((feature, score))
        
        # Find feature whose removal causes least harm
        worst_feature, best_score_without = max(scores, key=lambda x: x[1])
        
        # Current score with all features
        model = LogisticRegression(max_iter=1000)
        current_score = cross_val_score(model, X[:, selected], y, cv=cv, scoring=scoring).mean()
        
        # Remove if it improves or doesn't significantly hurt
        if best_score_without >= current_score - 0.005:
            selected.remove(worst_feature)
            print(f"Removed {feature_names[worst_feature]}: CV Score = {best_score_without:.4f}")
        else:
            break
    
    return [feature_names[i] for i in selected], selected

def stepwise_with_pvalues(X, y, feature_names, p_enter=0.05, p_remove=0.10):
    """
    Stepwise selection based on p-values using statsmodels.
    """
    included = []
    
    while True:
        changed = False
        
        # Forward step
        excluded = [i for i in range(X.shape[1]) if i not in included]
        best_pval = p_enter
        best_feature = None
        
        for feature in excluded:
            features = included + [feature]
            X_subset = sm.add_constant(X[:, features])
            model = sm.Logit(y, X_subset).fit(disp=0)
            
            # Get p-value for new feature (last one)
            pval = model.pvalues[-1]
            if pval < best_pval:
                best_pval = pval
                best_feature = feature
        
        if best_feature is not None:
            included.append(best_feature)
            changed = True
            print(f"Added {feature_names[best_feature]}: p-value = {best_pval:.4f}")
        
        # Backward step
        if len(included) > 1:
            X_subset = sm.add_constant(X[:, included])
            model = sm.Logit(y, X_subset).fit(disp=0)
            pvalues = model.pvalues[1:]  # Exclude intercept
            
            # Find worst feature
            worst_idx = np.argmax(pvalues)
            if pvalues[worst_idx] > p_remove:
                removed_feature = included.pop(worst_idx)
                changed = True
                print(f"Removed {feature_names[removed_feature]}: p-value = {pvalues[worst_idx]:.4f}")
        
        if not changed:
            break
    
    return [feature_names[i] for i in included], included

# Usage
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=500, n_features=15, n_informative=5, 
                           n_redundant=5, random_state=42)
feature_names = [f'Feature_{i}' for i in range(15)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("=== Forward Stepwise Selection ===")
selected_forward, indices_forward = forward_stepwise_selection(
    X_train, y_train, feature_names
)
print(f"\nSelected features: {selected_forward}")

print("\n=== Backward Stepwise Selection ===")
selected_backward, indices_backward = backward_stepwise_selection(
    X_train, y_train, feature_names
)
print(f"\nSelected features: {selected_backward}")

print("\n=== P-value Based Stepwise ===")
selected_pval, indices_pval = stepwise_with_pvalues(
    X_train, y_train, feature_names
)
print(f"\nSelected features: {selected_pval}")
```

**Interview Tip:**
Stepwise selection can overfit and doesn't handle multicollinearity well. Prefer L1 regularization for automatic feature selection in modern practice.

---

## Question 11
- [ ] Done

**Implement a logistic regression model with polynomial features using scikit-learn's Pipeline.**

### Answer

**Implementation:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.datasets import make_moons, make_circles

# Create non-linearly separable data
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with polynomial features
def create_polynomial_lr_pipeline(degree=2):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, C=1.0))
    ])

# Compare different polynomial degrees
degrees = [1, 2, 3, 4, 5]
results = []

for degree in degrees:
    pipeline = create_polynomial_lr_pipeline(degree)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
    
    pipeline.fit(X_train, y_train)
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    results.append({
        'degree': degree,
        'cv_auc': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_acc': train_score,
        'test_acc': test_score,
        'n_features': pipeline.named_steps['poly'].n_output_features_
    })
    
    print(f"Degree {degree}: CV AUC = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}, "
          f"Features = {pipeline.named_steps['poly'].n_output_features_}")

# Full pipeline with regularization tuning
full_pipeline = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000))
])

param_grid = {
    'poly__degree': [2, 3, 4],
    'lr__C': [0.01, 0.1, 1, 10],
    'lr__penalty': ['l1', 'l2'],
    'lr__solver': ['saga']
}

grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.3f}")
print(f"Test Score: {grid_search.score(X_test, y_test):.3f}")

# Visualize decision boundaries
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
    plt.title(title)

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, degree in enumerate([1, 2, 3]):
    plt.subplot(1, 3, i+1)
    pipeline = create_polynomial_lr_pipeline(degree)
    pipeline.fit(X_train, y_train)
    plot_decision_boundary(pipeline, X, y, f'Degree {degree}')

plt.tight_layout()
plt.show()

# Final model with interaction features only
pipeline_interactions = Pipeline([
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000))
])

pipeline_interactions.fit(X_train, y_train)
print(f"\nInteraction only features: {pipeline_interactions.named_steps['poly'].n_output_features_}")
print(f"Test Accuracy: {pipeline_interactions.score(X_test, y_test):.3f}")
```

**Interview Tip:**
Polynomial features capture non-linear relationships but increase model complexity exponentially. Use regularization (lower C) with higher degrees to prevent overfitting. `interaction_only=True` limits features to interactions only.

---
