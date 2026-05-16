# Supervised Learning Interview Questions - Theory Questions

---

## Linear Models and General Techniques

### Question 1: Describe Linear Regression
- [ ] Done

**Definition:**  
Linear Regression is a supervised algorithm that models the relationship between input features and a continuous target as a linear function. It finds the best-fit line/hyperplane by minimizing the sum of squared errors (Ordinary Least Squares).

**Mathematical Formulation:**
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon$$

**Matrix Form:**
$$\hat{y} = X\beta$$

**Normal Equation (Closed-form solution):**
$$\hat{\beta} = (X^TX)^{-1}X^Ty$$

**Core Concepts:**
- $\beta_0$: Intercept (y when all x = 0)
- $\beta_i$: Coefficient (change in y per unit change in $x_i$)
- $\epsilon$: Error term (noise)

**Assumptions:**
1. Linearity between X and y
2. Independence of errors
3. Homoscedasticity (constant error variance)
4. Normality of errors

**Python Code Example:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
```

---

### Question 2: Difference between Simple and Multiple Linear Regression
- [ ] Done

**Definition:**  
**Simple Linear Regression** uses one input feature to predict the target, modeling a straight line. **Multiple Linear Regression** uses two or more features, modeling a hyperplane in multi-dimensional space.

**Mathematical Formulation:**

| Type | Equation |
|------|----------|
| Simple | $y = \beta_0 + \beta_1 x$ |
| Multiple | $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n$ |

**Key Differences:**

| Aspect | Simple | Multiple |
|--------|--------|----------|
| Features | 1 | 2+ |
| Geometry | 2D line | Hyperplane |
| Interpretation | Direct relationship | Holding others constant |
| Complexity | Simple | Multicollinearity concerns |

**Interpretation Difference:**
- Simple: $\beta_1$ = change in y for 1-unit change in x
- Multiple: $\beta_i$ = change in y for 1-unit change in $x_i$, **holding all other features constant**

**Example:**
- Simple: Predict exam score from hours studied
- Multiple: Predict exam score from hours studied + attendance + previous quiz score

---

### Question 3: What is Logistic Regression and when is it used?
- [ ] Done

**Definition:**  
Logistic Regression is a classification algorithm that predicts the probability of a binary outcome by applying the sigmoid function to a linear combination of features. Despite its name, it's used for classification, not regression.

**Mathematical Formulation:**
$$z = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n$$
$$P(y=1|X) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

**Decision Rule:**
- If $P(y=1) \geq 0.5$ → Predict class 1
- If $P(y=1) < 0.5$ → Predict class 0

**Core Concepts:**
- Sigmoid squashes output to [0, 1] → interpretable as probability
- Learns a linear decision boundary
- Uses cross-entropy (log loss) for training

**When to Use:**
- Binary classification problems
- Spam detection, disease diagnosis, credit approval, churn prediction
- When probability estimates are needed
- As a baseline model

**Python Code Example:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

**Interview Tip:** Coefficients are interpretable - positive coefficient increases probability of class 1.

---

### Question 4: How does Ridge Regression prevent Overfitting?
- [ ] Done

**Definition:**  
Ridge Regression prevents overfitting by adding an L2 penalty (sum of squared weights) to the loss function. This forces the model to keep coefficients small, reducing model complexity and sensitivity to training data noise.

**Mathematical Formulation:**
$$\text{Ridge Loss} = \sum(y_i - \hat{y}_i)^2 + \alpha \sum \beta_j^2$$

**How it Works:**
1. Large weights → high penalty → model avoids them
2. All coefficients shrink toward zero (but never exactly zero)
3. Reduces variance at cost of slight bias increase
4. Handles multicollinearity by stabilizing coefficient estimates

**Effect of $\alpha$ (regularization strength):**
- $\alpha = 0$: Equivalent to ordinary linear regression
- $\alpha \rightarrow \infty$: All coefficients shrink to zero (underfitting)
- Optimal $\alpha$: Found via cross-validation

**Python Code Example:**
```python
from sklearn.linear_model import Ridge, RidgeCV

# Ridge with cross-validation to find best alpha
alphas = [0.1, 1.0, 10.0, 100.0]
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)

print(f"Best alpha: {ridge_cv.alpha_}")
print(f"Coefficients: {ridge_cv.coef_}")
```

**Interview Tip:** Ridge is also called "weight decay" in deep learning.

---

### Question 5: Describe Lasso Regression and its Unique Property
- [ ] Done

**Definition:**  
Lasso Regression uses L1 regularization (sum of absolute weights) which has the unique property of forcing some coefficients to exactly zero, performing **automatic feature selection** and producing sparse models.

**Mathematical Formulation:**
$$\text{Lasso Loss} = \sum(y_i - \hat{y}_i)^2 + \alpha \sum |\beta_j|$$

**Unique Property - Feature Selection:**
- L2 (Ridge): Shrinks all coefficients, none become exactly zero
- L1 (Lasso): Can force coefficients to exactly zero
- Result: Automatically removes irrelevant features

**When to Use Lasso vs Ridge:**

| Use Lasso | Use Ridge |
|-----------|-----------|
| Many features, suspect many irrelevant | Most features are relevant |
| Want feature selection | Multicollinearity concern |
| Want interpretable sparse model | Want stable predictions |

**Python Code Example:**
```python
from sklearn.linear_model import Lasso
import numpy as np

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# See which features were selected (non-zero coefficients)
selected_features = np.where(lasso.coef_ != 0)[0]
print(f"Selected features: {selected_features}")
print(f"Number of features used: {np.sum(lasso.coef_ != 0)}")
```

**Interview Tip:** Elastic Net combines L1 and L2 for best of both worlds.

---

### Question 6: Explain the Principle of Support Vector Machine (SVM)
- [ ] Done

**Definition:**  
SVM finds the optimal hyperplane that maximizes the margin (distance) between classes. The decision boundary is defined by support vectors (nearest points to the boundary). For non-linear data, the kernel trick projects data to higher dimensions where it becomes linearly separable.

**Core Concepts:**
- **Hyperplane:** Decision boundary separating classes
- **Margin:** Distance between hyperplane and nearest points
- **Support Vectors:** Points on the margin boundary (define the hyperplane)
- **Kernel Trick:** Implicit mapping to higher dimensions for non-linear separation

**Mathematical Formulation:**
$$\text{Maximize } \frac{2}{||w||} \quad \text{(margin)}$$
$$\text{Subject to } y_i(w \cdot x_i + b) \geq 1$$

**Common Kernels:**
- Linear: $K(x, x') = x \cdot x'$
- Polynomial: $K(x, x') = (x \cdot x' + c)^d$
- RBF (Gaussian): $K(x, x') = e^{-\gamma||x-x'||^2}$

**Python Code Example:**
```python
from sklearn.svm import SVC

# Linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)

# RBF Kernel for non-linear data
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train, y_train)

print(f"Number of support vectors: {len(svm_rbf.support_vectors_)}")
```

**Intuition:** Find the widest street (margin) that separates two neighborhoods (classes).

---

### Question 7: Handling Categorical Variables
- [ ] Done

**Definition:**  
Categorical variables must be converted to numerical format for most ML models. The encoding method depends on whether the variable is **nominal** (no order) or **ordinal** (has order).

**Encoding Methods:**

| Variable Type | Method | Description |
|---------------|--------|-------------|
| **Nominal** (no order) | One-Hot Encoding | Create binary column per category |
| **Ordinal** (has order) | Label Encoding | Assign integers based on rank |
| **High Cardinality** | Target Encoding | Replace with mean of target |
| **High Cardinality** | Feature Hashing | Hash to fixed-size vector |

**Python Code Example:**
```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

# One-Hot Encoding (nominal)
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = ohe.fit_transform(df[['color']])  # color: red, blue, green

# Label Encoding (ordinal)
le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])  # HS=0, BS=1, MS=2

# Pandas get_dummies (quick one-hot)
df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True)
```

**Interview Tip:** Never use Label Encoding for nominal variables - model assumes false ordering.

---

### Question 8: What are Ensemble Methods?
- [ ] Done

**Definition:**  
Ensemble methods combine predictions from multiple individual models (base learners) to produce a final prediction. They improve performance by reducing variance (bagging), reducing bias (boosting), or leveraging diverse model strengths (stacking).

**Core Concepts:**

| Method | How it Works | Goal | Example |
|--------|--------------|------|---------|
| **Bagging** | Train models on bootstrap samples, aggregate | Reduce variance | Random Forest |
| **Boosting** | Train sequentially, focus on errors | Reduce bias | XGBoost, AdaBoost |
| **Stacking** | Train meta-model on base model predictions | Combine strengths | Blend diverse models |

**Why They Work:**
- Averaging reduces noise and variance
- Sequential correction reduces bias
- Diverse models capture different patterns

**Python Code Example:**
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

# Simple voting ensemble
ensemble = VotingClassifier(estimators=[
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier(n_estimators=100)),
    ('lr', LogisticRegression())
], voting='soft')  # 'soft' uses probabilities

ensemble.fit(X_train, y_train)
print(f"Ensemble Accuracy: {ensemble.score(X_test, y_test):.4f}")
```

**Interview Tip:** Ensemble methods (XGBoost, LightGBM) win most Kaggle competitions.

---

### Question 9: Difference between Bagging and Boosting
- [ ] Done

**Definition:**  
**Bagging** trains models independently on random data subsets and averages predictions (reduces variance). **Boosting** trains models sequentially where each model corrects errors of previous ones (reduces bias).

**Key Differences:**

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| **Goal** | Reduce variance | Reduce bias |
| **Training** | Parallel (independent) | Sequential (dependent) |
| **Data Sampling** | Bootstrap samples | Weighted samples |
| **Aggregation** | Simple average/vote | Weighted sum |
| **Base Models** | High variance (deep trees) | High bias (shallow trees) |
| **Examples** | Random Forest | AdaBoost, XGBoost, LightGBM |

**Bagging Process:**
1. Create N bootstrap samples (random sampling with replacement)
2. Train N models independently
3. Aggregate: Vote (classification) or Average (regression)

**Boosting Process:**
1. Train model on full data
2. Identify misclassified/high-error samples
3. Train next model focusing on errors
4. Combine with weighted sum

**Python Code Example:**
```python
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    bootstrap=True
)

# Boosting
boosting = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # weak learner
    n_estimators=100
)
```

---

### Question 10: Explain Feature Scaling and its Importance
- [ ] Done

**Definition:**  
Feature scaling transforms numerical features to a similar scale/range. It's critical for gradient-based algorithms (convergence speed) and distance-based algorithms (equal feature contribution), but not needed for tree-based models.

**Common Techniques:**

| Method | Formula | Result |
|--------|---------|--------|
| **Standardization** | $\frac{x - \mu}{\sigma}$ | Mean=0, Std=1 |
| **Min-Max Scaling** | $\frac{x - x_{min}}{x_{max} - x_{min}}$ | Range [0, 1] |

**When Feature Scaling is Critical:**
- Gradient Descent: Unscaled features cause slow, inefficient convergence
- KNN, SVM, K-Means: Distance calculations dominated by large-scale features
- Neural Networks: Activations and gradients behave better with scaled inputs

**When NOT Needed:**
- Tree-based models (Decision Trees, Random Forest, XGBoost)

**Python Code Example:**
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization (most common)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train only
X_test_scaled = scaler.transform(X_test)        # transform test

# Min-Max Scaling
minmax = MinMaxScaler()
X_train_norm = minmax.fit_transform(X_train)
X_test_norm = minmax.transform(X_test)
```

**Interview Tip:** Always fit scaler on training data only, then transform both train and test to prevent data leakage.

---

## Decision Trees and Random Forests

### Question 11: How is a Decision Tree Constructed?
- [ ] Done

**Definition:**  
A decision tree is built using recursive partitioning. At each node, the algorithm finds the best feature and split point that maximizes purity (minimizes impurity) in the resulting child nodes. This process repeats until stopping criteria are met.

**Algorithm Steps (CART):**
1. Start with entire dataset at root node
2. For each feature and each possible split point:
   - Calculate impurity metric (Gini or Entropy) of child nodes
3. Select split with lowest weighted impurity (or highest information gain)
4. Create child nodes based on split rule
5. Recursively repeat for each child node
6. Stop when: node is pure, max depth reached, or min samples threshold

**Stopping Criteria:**
- Node is pure (all samples same class)
- Maximum depth reached
- Minimum samples per node
- No improvement possible

**Python Code Example:**
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Create and train
tree = DecisionTreeClassifier(max_depth=3, min_samples_split=5)
tree.fit(X_train, y_train)

# Visualize
plt.figure(figsize=(15, 8))
plot_tree(tree, filled=True, feature_names=feature_names)
plt.show()
```

---

### Question 12: Pros and Cons of Decision Trees
- [ ] Done

**Definition:**  
Decision trees are interpretable models that recursively partition data based on feature thresholds. They're easy to understand but prone to overfitting and instability.

**Pros:**
| Advantage | Explanation |
|-----------|-------------|
| High Interpretability | Can visualize and explain decisions |
| No Feature Scaling | Not sensitive to scale of features |
| Handles Mixed Data | Works with numerical and categorical |
| Non-linear | Captures complex relationships |
| Fast Inference | Simple if-else rules |

**Cons:**
| Disadvantage | Explanation |
|--------------|-------------|
| Prone to Overfitting | Deep trees memorize training data |
| High Variance | Small data changes → different trees |
| Greedy Algorithm | Locally optimal, not globally optimal |
| Axis-parallel Boundaries | Stair-step decision boundaries |
| Bias to High-cardinality Features | More split options = more likely selected |

**Interview Tip:** Decision trees are building blocks for Random Forest and Gradient Boosting which overcome their weaknesses.

---

### Question 13: Explain Gini Impurity and Information Gain
- [ ] Done

**Definition:**  
**Gini Impurity** measures probability of misclassifying a randomly chosen element. **Information Gain** (based on Entropy) measures reduction in uncertainty after a split. Both guide the tree to find the best splits.

**Mathematical Formulation:**

**Gini Impurity:**
$$Gini = 1 - \sum_{i=1}^{C} p_i^2$$

**Entropy:**
$$Entropy = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

**Information Gain:**
$$IG = Entropy_{parent} - \sum \frac{N_{child}}{N_{parent}} \cdot Entropy_{child}$$

**Interpretation:**
| Metric | Pure Node | Maximum Impurity (binary) |
|--------|-----------|---------------------------|
| Gini | 0 | 0.5 |
| Entropy | 0 | 1.0 |

**Python Code Example:**
```python
import numpy as np

def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-10))

# Example
y = [0, 0, 1, 1, 1]
print(f"Gini: {gini(y):.4f}")
print(f"Entropy: {entropy(y):.4f}")
```

**Interview Tip:** Gini is computationally faster (no log). CART uses Gini; ID3/C4.5 use Entropy.

---

### Question 14: How Random Forest Improves on Decision Trees
- [ ] Done

**Definition:**  
Random Forest improves on single decision trees by using two sources of randomness: **Bootstrap Aggregating (Bagging)** trains trees on random data samples, and **Feature Randomness** considers random feature subsets at each split. This creates decorrelated trees whose averaged predictions reduce variance.

**Two Key Innovations:**

| Technique | How It Works | Effect |
|-----------|--------------|--------|
| **Bootstrap Sampling** | Each tree trained on random sample with replacement | Different trees see different data |
| **Feature Randomness** | At each split, only random subset of features considered | Decorrelates trees |

**How Predictions are Made:**
- Classification: Majority vote across all trees
- Regression: Average of all tree predictions

**Python Code Example:**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_features='sqrt',   # Features considered per split
    max_depth=10,
    random_state=42
)
rf.fit(X_train, y_train)

print(f"Accuracy: {rf.score(X_test, y_test):.4f}")
print(f"Feature Importances: {rf.feature_importances_}")
```

**Interview Tip:** Random Forest reduces variance significantly but loses interpretability of single tree.

---

### Question 15: What is Feature Importance?
- [ ] Done

**Definition:**  
Feature importance scores indicate how useful each feature was for making predictions. In tree-based models, it's calculated as the total reduction in impurity (Gini/Entropy) brought by that feature across all splits, averaged over all trees.

**Calculation Process:**
1. For each split using feature $f$: Calculate impurity reduction
2. Sum all impurity reductions for feature $f$ across tree
3. For Random Forest: Average across all trees
4. Normalize so all importances sum to 1

**Alternative Methods:**
- **Permutation Importance:** Shuffle feature values, measure performance drop
- **SHAP Values:** Game-theoretic approach to feature attribution

**Python Code Example:**
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Feature importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head(10))

# Permutation importance (more robust)
from sklearn.inspection import permutation_importance
perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10)
```

---

## Neural Networks and Deep Learning

### Question 16: Basic Components of a Neural Network
- [ ] Done

**Definition:**  
A neural network consists of **neurons** organized in **layers** (input, hidden, output), connected by **weights**. Each neuron computes a weighted sum of inputs, adds a **bias**, and applies an **activation function** to introduce non-linearity.

**Components:**

| Component | Description |
|-----------|-------------|
| **Neurons** | Basic processing units |
| **Input Layer** | Receives raw features (no computation) |
| **Hidden Layers** | Learn feature representations |
| **Output Layer** | Produces final prediction |
| **Weights (w)** | Learnable connection strengths |
| **Biases (b)** | Learnable shift parameters |
| **Activation Function** | Introduces non-linearity |

**Mathematical Formulation:**
$$z = \sum_{i} w_i x_i + b$$
$$a = \sigma(z)$$

Where $\sigma$ is the activation function.

**Python Code Example:**
```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)  # weights + bias
        self.activation = nn.ReLU()                        # activation
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x
```

---

### Question 17: Role of Activation Functions
- [ ] Done

**Definition:**  
Activation functions introduce non-linearity into neural networks. Without them, any deep network would collapse to a single linear transformation and couldn't learn complex patterns.

**Why Non-linearity is Essential:**
- Stack of linear functions = still linear
- Without activation: Deep network ≡ Linear regression
- Non-linearity enables learning complex decision boundaries

**Common Activation Functions:**

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| **ReLU** | $\max(0, z)$ | [0, ∞) | Default for hidden layers |
| **Sigmoid** | $\frac{1}{1+e^{-z}}$ | [0, 1] | Binary output layer |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | [-1, 1] | Zero-centered |
| **Softmax** | $\frac{e^{z_i}}{\sum e^{z_j}}$ | [0, 1] | Multi-class output |
| **Leaky ReLU** | $\max(\alpha z, z)$ | (-∞, ∞) | Avoids dying ReLU |

**Problems:**
- Sigmoid/Tanh: Vanishing gradient problem
- ReLU: Dying ReLU (neurons stuck at 0)

**Interview Tip:** ReLU is default choice for hidden layers; Sigmoid for binary output; Softmax for multi-class output.

---

### Question 18: Preventing Overfitting in Neural Networks
- [ ] Done

**Definition:**  
Neural networks are especially prone to overfitting due to their high capacity. Key techniques include Dropout, weight regularization, data augmentation, early stopping, and batch normalization.

**Techniques:**

| Technique | How It Works |
|-----------|--------------|
| **Dropout** | Randomly set neurons to 0 during training |
| **L2 Regularization (Weight Decay)** | Add penalty for large weights |
| **Data Augmentation** | Create modified training samples |
| **Early Stopping** | Stop when validation loss plateaus |
| **Batch Normalization** | Normalize layer inputs (slight regularization) |
| **Reduce Architecture** | Fewer layers/neurons |
| **Transfer Learning** | Start from pretrained weights |

**Python Code Example (Keras):**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),  # L2
    BatchNormalization(),
    Dropout(0.5),  # 50% dropout
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X, y, validation_split=0.2, callbacks=[early_stop], epochs=100)
```

---

### Question 19: Discuss backpropagation and its significance in neural networks.
- [ ] Done

### Definition
Backpropagation is the algorithm used to compute gradients of the loss function with respect to all weights in a neural network, enabling gradient descent optimization.

### Why It Matters

1. **Enables Training**: Without backprop, we couldn't train multi-layer networks efficiently
2. **Computational Efficiency**: Uses chain rule to compute all gradients in one backward pass
3. **Foundation of Deep Learning**: Made modern neural networks possible

### Core Mechanism

**Forward Pass:**
```
Input → Hidden Layers → Output → Loss
```

**Backward Pass:**
```
Loss → Gradients flow backward → Update all weights
```

### Mathematical Essence (Chain Rule)

For a weight $w$ in layer $l$:

$$\frac{\partial L}{\partial w^{(l)}} = \frac{\partial L}{\partial a^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial a^{(L-1)}} \cdots \frac{\partial a^{(l+1)}}{\partial a^{(l)}} \cdot \frac{\partial a^{(l)}}{\partial w^{(l)}}$$

### Key Steps

1. **Forward Pass**: Compute activations layer by layer, store intermediate values
2. **Compute Output Loss**: Calculate loss at final layer
3. **Backward Pass**: Propagate error gradients from output to input layer
4. **Weight Update**: $w_{new} = w_{old} - \eta \cdot \nabla_w L$

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Vanishing Gradients | Sigmoid/tanh squash gradients | Use ReLU, skip connections |
| Exploding Gradients | Gradients multiply and grow | Gradient clipping, batch norm |

---

### Question 20: Shallow vs Deep Neural Networks
- [ ] Done

**Definition:**  
A **shallow network** has one hidden layer; a **deep network** has two or more. Depth enables hierarchical feature learning where early layers learn simple patterns and later layers learn complex abstractions.

**Key Differences:**

| Aspect | Shallow (1 hidden layer) | Deep (2+ hidden layers) |
|--------|--------------------------|-------------------------|
| Feature Learning | Single abstraction level | Hierarchical features |
| Capacity | Limited | High |
| Efficiency | May need huge width | More parameter efficient |
| Use Cases | Simple patterns | Images, text, speech |

**Hierarchical Feature Learning (Image Example):**
- Layer 1: Edges, corners
- Layer 2: Textures, patterns
- Layer 3: Object parts (eyes, wheels)
- Layer 4: Full objects (faces, cars)

**Challenges with Depth:**
- Vanishing/Exploding gradients
- Solutions: ReLU, Batch Normalization, Residual connections

**Interview Tip:** Universal Approximation Theorem says shallow networks can approximate any function, but deep networks do it more efficiently.

---

## Evaluation Metrics

### Question 21: Confusion Matrix, Precision, Recall, F1 Score
- [ ] Done

**Definition:**  
A **confusion matrix** summarizes classification results into TP, TN, FP, FN. **Precision** measures accuracy of positive predictions. **Recall** measures coverage of actual positives. **F1 Score** is the harmonic mean balancing both.

**Confusion Matrix:**
|  | Predicted Positive | Predicted Negative |
|--|--------------------|--------------------|
| **Actual Positive** | TP (True Positive) | FN (False Negative) |
| **Actual Negative** | FP (False Positive) | TN (True Negative) |

**Metrics:**
$$Precision = \frac{TP}{TP + FP}$$ (Of predicted positive, how many correct?)

$$Recall = \frac{TP}{TP + FN}$$ (Of actual positive, how many found?)

$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

**When to Prioritize:**
- **Precision:** When FP is costly (spam filter)
- **Recall:** When FN is costly (disease detection)

**Python Code Example:**
```python
from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Full report
print(classification_report(y_test, y_pred))
```

---

### Question 22: Accuracy and Why It's Not Always the Best Metric
- [ ] Done

**Definition:**  
Accuracy = (Correct Predictions) / (Total Predictions). It's misleading for **imbalanced datasets** because a model predicting only the majority class achieves high accuracy while being useless.

**Formula:**
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**The Problem:**

| Dataset | Class Distribution | Naive Model Strategy | Accuracy |
|---------|-------------------|---------------------|----------|
| Fraud Detection | 99% legit, 1% fraud | Always predict "legit" | 99% |
| Disease Diagnosis | 95% healthy, 5% sick | Always predict "healthy" | 95% |

**Better Metrics for Imbalanced Data:**
- **Precision:** Of predicted positive, how many correct?
- **Recall:** Of actual positive, how many found?
- **F1 Score:** Harmonic mean of precision and recall
- **AUC-ROC:** Discrimination ability across thresholds
- **AUPRC:** Precision-Recall curve (best for severe imbalance)

**Interview Tip:** Always ask about class distribution before choosing metrics.

---

### Question 23: ROC Curve and AUC
- [ ] Done

**Definition:**  
**ROC curve** plots True Positive Rate (Recall) vs False Positive Rate at various classification thresholds. **AUC** (Area Under Curve) summarizes performance as a single value between 0 and 1, where 1 is perfect and 0.5 is random.

**Mathematical Formulation:**
$$TPR = \frac{TP}{TP + FN}$$ (Same as Recall)
$$FPR = \frac{FP}{FP + TN}$$

**Interpretation:**
| AUC Value | Interpretation |
|-----------|----------------|
| 1.0 | Perfect classifier |
| 0.9 - 1.0 | Excellent |
| 0.8 - 0.9 | Good |
| 0.5 | Random guess |
| < 0.5 | Worse than random |

**Probabilistic Meaning:**  
AUC = Probability that randomly chosen positive sample is ranked higher than randomly chosen negative sample.

**Python Code Example:**
```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

**Interview Tip:** AUC is threshold-independent and works well for imbalanced datasets.

---

### Question 24: Compare RMSE and MAE
- [ ] Done

**Definition:**  
Both are regression error metrics. **MAE** (Mean Absolute Error) treats all errors equally. **RMSE** (Root Mean Squared Error) penalizes large errors more heavily due to squaring.

**Formulas:**
$$MAE = \frac{1}{N}\sum|y_i - \hat{y}_i|$$
$$RMSE = \sqrt{\frac{1}{N}\sum(y_i - \hat{y}_i)^2}$$

**Comparison:**

| Aspect | MAE | RMSE |
|--------|-----|------|
| Error Penalty | Linear | Quadratic |
| Outlier Sensitivity | Robust | Sensitive |
| Interpretation | "Average error magnitude" | "Std dev of errors" |
| Differentiability | Not at zero | Everywhere |

**When to Use:**
- **RMSE:** Large errors are especially bad (critical systems)
- **MAE:** Outliers present, want robust metric

**Python Code Example:**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
```

---

### Question 25: When to Use MAPE
- [ ] Done

**Definition:**  
MAPE (Mean Absolute Percentage Error) measures average percentage error. Use it when you need **relative error** that's easy to explain to stakeholders or when comparing forecasts across different scales.

**Formula:**
$$MAPE = \frac{100\%}{N}\sum\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

**When to Use:**
- Business stakeholders need intuitive interpretation ("5% error")
- Comparing forecasts across different scales (sales of different products)

**When to AVOID:**
- Actual values can be **zero** (division by zero)
- Data has values near zero (inflates percentage)
- **Asymmetric penalty:** Under-predictions penalized less than over-predictions

**Alternative:** sMAPE (Symmetric MAPE) uses average of actual and predicted in denominator.

**Python Code Example:**
```python
import numpy as np

def mape(y_true, y_pred):
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

print(f"MAPE: {mape(y_true, y_pred):.2f}%")
```

---

## Model Tuning and Optimization

### Question 26: Importance of Hyperparameter Tuning
- [ ] Done

### Definition
Hyperparameter tuning is the process of finding optimal values for model parameters that are set before training (not learned from data).

### Why It's Critical

1. **Performance**: Right hyperparameters can dramatically improve accuracy
2. **Generalization**: Prevents overfitting/underfitting
3. **Efficiency**: Wrong settings waste compute time

### Types of Hyperparameters

| Model Type | Common Hyperparameters |
|------------|----------------------|
| Neural Networks | Learning rate, batch size, layers, neurons, dropout |
| Decision Trees | max_depth, min_samples_split, min_samples_leaf |
| SVM | C (regularization), kernel type, gamma |
| Random Forest | n_estimators, max_features, max_depth |

### Tuning Methods

| Method | How It Works | Pros/Cons |
|--------|--------------|-----------|
| **Grid Search** | Try all combinations | Thorough but expensive |
| **Random Search** | Sample random combinations | More efficient, good coverage |
| **Bayesian Optimization** | Use past results to guide search | Smart, efficient, complex setup |

### Python Example

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search
param_grid = {'max_depth': [3, 5, 7], 'n_estimators': [50, 100, 200]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

# Random Search (more efficient)
from scipy.stats import randint
param_dist = {'max_depth': randint(3, 10), 'n_estimators': randint(50, 300)}
random_search = RandomizedSearchCV(RandomForestClassifier(), param_dist, n_iter=20, cv=5)
random_search.fit(X_train, y_train)
```

### Best Practices

1. Always use cross-validation during tuning
2. Start with coarse search, then refine
3. Focus on most impactful hyperparameters first
4. Keep a held-out test set for final evaluation

---

### Question 27: Grid Search for Hyperparameter Optimization
- [ ] Done

**Definition:**  
Grid Search exhaustively evaluates all combinations of specified hyperparameter values using cross-validation. It's guaranteed to find the best combination within the grid but is computationally expensive.

**Process:**
1. Define grid of hyperparameter values to try
2. For each combination:
   - Train model with those hyperparameters
   - Evaluate using k-fold cross-validation
3. Select combination with best CV score

**Python Code Example:**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Define hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

# Grid search with 5-fold CV
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_
```

**Limitation:** Number of combinations grows exponentially with hyperparameters.

---

### Question 28: Random Search vs Grid Search
- [ ] Done

**Definition:**  
Random Search samples random hyperparameter combinations from specified distributions instead of exhaustively searching a grid. It's more efficient because it explores the search space more broadly with fewer evaluations.

**Key Differences:**

| Aspect | Grid Search | Random Search |
|--------|-------------|---------------|
| Search Strategy | Exhaustive | Random sampling |
| Computational Cost | Exponential | Fixed (n_iter) |
| Coverage | Only grid points | Entire distribution |
| Efficiency | Wastes time on unimportant params | Explores broadly |

**Why Random Search is Often Better:**
- Some hyperparameters are more important than others
- Random search explores more values of important parameters
- Same compute budget → better results

**Python Code Example:**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define distributions
param_distributions = {
    'C': uniform(0.1, 100),
    'gamma': uniform(0.001, 1),
    'kernel': ['rbf', 'linear']
}

# Random search with 50 iterations
random_search = RandomizedSearchCV(
    SVC(), param_distributions, 
    n_iter=50, cv=5, random_state=42
)
random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
```

---

### Question 29: Role of Learning Rate in Neural Network Convergence
- [ ] Done

### Definition
The learning rate (η) controls the step size when updating weights during gradient descent. It determines how much to adjust weights based on the computed gradient.

### Weight Update Rule

$$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$$

### Impact on Training

| Learning Rate | Behavior | Outcome |
|--------------|----------|---------|
| **Too High** | Large steps, overshoots minimum | Divergence, loss explodes |
| **Too Low** | Tiny steps, slow progress | Very slow convergence, may get stuck |
| **Optimal** | Balanced steps | Fast, stable convergence |

### Visual Intuition

```
Too High:    ↗↘↗↘ (oscillating, never converges)
Too Low:     →→→→→→→→→... (takes forever)
Just Right:  →→→→ ✓ (smooth descent to minimum)
```

### Learning Rate Schedules

| Schedule | Description | Use Case |
|----------|-------------|----------|
| **Constant** | Same rate throughout | Simple, works for small problems |
| **Step Decay** | Reduce by factor every N epochs | Common, easy to implement |
| **Exponential Decay** | $\eta_t = \eta_0 \cdot e^{-kt}$ | Smooth reduction |
| **Cosine Annealing** | Follows cosine curve | Popular in deep learning |
| **Warmup** | Start low, increase, then decay | Stabilizes training initially |

### Adaptive Methods

| Optimizer | Learning Rate Behavior |
|-----------|----------------------|
| **Adam** | Per-parameter adaptive rates |
| **RMSprop** | Adapts based on recent gradient magnitudes |
| **AdaGrad** | Decreases rate for frequent features |

### Practical Tips

1. **Start with 0.001** for Adam, **0.01** for SGD
2. Use learning rate finder: increase rate until loss explodes
3. Reduce rate if validation loss plateaus
4. Use adaptive optimizers (Adam) for most cases

---

### Question 30: What is Early Stopping?
- [ ] Done

**Definition:**  
Early stopping monitors validation loss during training and halts when it stops improving for a specified number of epochs (patience). It prevents overfitting by stopping before the model memorizes training noise.

**Process:**
1. Split data into train and validation sets
2. Train model, evaluate on validation after each epoch
3. Track best validation score
4. If no improvement for `patience` epochs → stop
5. Use model from best validation epoch

**Python Code Example (Keras):**
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,              # Stop after 10 epochs without improvement
    restore_best_weights=True # Keep best model, not last
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=1000,  # Large number, early stopping will halt
    callbacks=[early_stop]
)
```

**Interview Tip:** Early stopping is a form of regularization that saves training time.

---

## Algorithm-Specific Questions

### Question 31: Handling Imbalanced Datasets
- [ ] Done

**Definition:**  
Imbalanced datasets have skewed class distributions. Solutions include appropriate metrics, resampling techniques, and algorithm-level modifications like class weights.

**Three-Pronged Strategy:**

**1. Choose Right Metrics:**
- F1-Score, Precision-Recall AUC (not accuracy)

**2. Data-Level Techniques:**

| Technique | Method |
|-----------|--------|
| **Oversampling** | Duplicate/SMOTE minority class |
| **Undersampling** | Remove majority class samples |

**3. Algorithm-Level Techniques:**

| Technique | Method |
|-----------|--------|
| **Class Weights** | Penalize minority misclassification more |
| **Threshold Adjustment** | Lower decision threshold |

**Python Code Example:**
```python
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# SMOTE oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Class weights
rf = RandomForestClassifier(class_weight='balanced')  # Auto-adjust
rf.fit(X_train, y_train)

# Or manual weights
rf = RandomForestClassifier(class_weight={0: 1, 1: 10})  # 10x penalty for class 1
```

**Interview Tip:** Never resample test set - it must reflect real distribution.

---

### Question 32: SVMs with Non-linear Kernels
- [ ] Done

**Definition:**  
When data isn't linearly separable, SVMs use the **kernel trick** to implicitly map data to higher-dimensional space where a linear boundary becomes possible. The RBF kernel is most common, enabling complex, curved decision boundaries.

**The Kernel Trick:**
- Compute dot products in high-dimensional space without explicit transformation
- Only depends on dot products, not actual coordinates
- Computationally efficient

**Common Kernels:**

| Kernel | Formula | Use Case |
|--------|---------|----------|
| Linear | $K(x, y) = x^T y$ | Linearly separable data |
| Polynomial | $(x^T y + c)^d$ | Polynomial boundaries |
| RBF (Gaussian) | $e^{-\gamma||x-y||^2}$ | Complex, localized boundaries |

**RBF Parameters:**
- $\gamma$: Controls influence radius (high γ → complex boundary)
- $C$: Regularization (low C → smoother boundary)

**Python Code Example:**
```python
from sklearn.svm import SVC

# RBF Kernel SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)

# Tune with grid search
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
```

---

### Question 33: How Decision Trees are Pruned
- [ ] Done

### Why Prune?

An unpruned tree overfits by memorizing training noise. Pruning removes unnecessary branches to improve generalization.

### Two Types of Pruning

#### 1. Pre-Pruning (Early Stopping)

**How**: Stop tree growth before full depth

| Parameter | Effect |
|-----------|--------|
| `max_depth` | Limits tree depth |
| `min_samples_split` | Minimum samples needed to split a node |
| `min_samples_leaf` | Minimum samples in leaf nodes |
| `min_impurity_decrease` | Minimum impurity reduction for split |

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5
)
```

**Pros**: Computationally efficient
**Cons**: May stop too early (greedy)

#### 2. Post-Pruning (Cost-Complexity Pruning)

**How**: Grow full tree, then prune back

**Process**:
1. Build complete tree
2. Calculate cost-complexity: $R_\alpha(T) = R(T) + \alpha |T|$
   - $R(T)$ = misclassification cost
   - $|T|$ = number of leaves
   - $\alpha$ = complexity penalty
3. Prune subtrees where removing improves cross-validation score

```python
# Find optimal alpha
path = tree.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas

# Cross-validate to find best alpha
best_tree = DecisionTreeClassifier(ccp_alpha=best_alpha)
```

**Pros**: Considers full tree structure, often better results
**Cons**: More computationally expensive

### Practical Recommendation

Use pre-pruning via hyperparameter tuning with cross-validation - simpler and usually sufficient.

---

### Question 34: AdaBoost Algorithm
- [ ] Done

**Definition:**  
AdaBoost (Adaptive Boosting) sequentially trains weak learners (typically stumps), giving more weight to misclassified samples after each iteration. The final prediction is a weighted vote where better-performing models have higher influence.

**Algorithm Steps:**
1. **Initialize:** Equal weights for all samples: $w_i = 1/N$
2. **For m = 1 to M:**
   - Train weak learner $h_m$ on weighted data
   - Calculate weighted error: $\epsilon_m = \sum w_i \cdot \mathbb{1}(h_m(x_i) \neq y_i)$
   - Calculate model weight: $\alpha_m = \frac{1}{2}\ln\frac{1-\epsilon_m}{\epsilon_m}$
   - Update sample weights: Increase for misclassified, decrease for correct
   - Normalize weights
3. **Final prediction:** $H(x) = \text{sign}\left(\sum_{m=1}^{M} \alpha_m h_m(x)\right)$

**Python Code Example:**
```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# AdaBoost with decision stumps
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # stump
    n_estimators=50,
    learning_rate=1.0
)
ada.fit(X_train, y_train)
```

---

### Question 35: Gradient Boosting vs AdaBoost
- [ ] Done

**Definition:**  
Both are boosting methods that train models sequentially. **AdaBoost** reweights samples based on misclassification. **Gradient Boosting** fits each new model to the residual errors of the current ensemble, making it more flexible with any differentiable loss function.

**Key Differences:**

| Aspect | AdaBoost | Gradient Boosting |
|--------|----------|-------------------|
| Error Correction | Re-weight samples | Fit residuals |
| Focus | Misclassified points | Prediction errors |
| Loss Function | Exponential | Any differentiable |
| Flexibility | Limited | Highly flexible |
| Modern Variants | Less used | XGBoost, LightGBM, CatBoost |

**Gradient Boosting Process:**
1. Start with initial prediction (mean)
2. Calculate residuals: $r = y - \hat{y}$
3. Fit new tree to residuals
4. Update: $\hat{y} = \hat{y} + \eta \cdot \text{new\_tree}(x)$
5. Repeat

**Python Code Example:**
```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
gb.fit(X_train, y_train)
```

---

## Dataset Handling

### Question 36: Handling Missing Data
- [ ] Done

**Definition:**  
Missing data handling depends on the mechanism (MCAR, MAR, MNAR) and percentage missing. Strategies include deletion, simple imputation (mean/median/mode), or advanced imputation (KNN, iterative models).

**Handling Strategies:**

| Strategy | When to Use | Pros/Cons |
|----------|-------------|-----------|
| **Delete rows** | Low % missing, MCAR | Simple, may lose data |
| **Delete columns** | >50% missing | Simple, may lose info |
| **Mean/Median** | Low %, numerical | Fast, reduces variance |
| **Mode** | Categorical | Simple |
| **KNN Imputation** | Structured data | Considers neighbors |
| **Iterative Imputer** | Complex patterns | Most accurate |

**Python Code Example:**
```python
from sklearn.impute import SimpleImputer, KNNImputer

# Simple imputation
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_train)

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed_knn = knn_imputer.fit_transform(X_train)

# For tree models: can use indicator for missingness
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='constant', fill_value=-999)
```

**Interview Tip:** Fit imputer on training data only; transform both train and test.

---

### Question 37: Dataset Preparation Steps
- [ ] Done

**Definition:**  
Dataset preparation involves systematic cleaning, engineering, transformation, and splitting of data before model training. Proper preparation prevents data leakage and ensures model reliability.

**Step-by-Step Pipeline:**

| Step | Actions |
|------|---------|
| **1. Data Cleaning** | Handle missing values, fix inconsistencies, remove duplicates, handle outliers |
| **2. Feature Engineering** | Create new features, combine/decompose existing ones |
| **3. Feature Transformation** | Encode categoricals (one-hot, label), scale numericals |
| **4. Feature Selection** | Remove irrelevant/redundant features |
| **5. Data Splitting** | Split into train/validation/test |

**Python Code Example:**
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define preprocessing
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Full pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier())
])
```

---

### Question 38: Handling Textual Data in Supervised Learning
- [ ] Done

### Step 1: Text Preprocessing

```python
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def preprocess(text):
    text = text.lower()                           # Lowercase
    text = re.sub(r'[^\w\s]', '', text)          # Remove punctuation
    tokens = text.split()                         # Tokenize
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)
```

### Step 2: Vectorization Methods

| Method | Description | When to Use |
|--------|-------------|-------------|
| **CountVectorizer** | Word counts | Simple baseline |
| **TF-IDF** | Weighted by importance | Strong baseline for most tasks |
| **Word2Vec/GloVe** | Dense word embeddings | Need semantic similarity |
| **BERT** | Contextual embeddings | State-of-the-art performance |

### Approach Comparison

#### Traditional (Fast, Simple)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(texts)
model = LogisticRegression()
model.fit(X_tfidf, y)
```

#### Deep Learning (Best Performance)

```python
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# Fine-tune on your labeled data
```

### My Strategy

1. **Always start with TF-IDF + Logistic Regression** as baseline
2. If not sufficient, move to **fine-tuned BERT** for state-of-the-art results
3. BERT understands context ("not good" vs "good") while TF-IDF doesn't

---

### Question 39: When is Dimensionality Reduction Useful?
- [ ] Done

**Definition:**  
Dimensionality reduction reduces the number of features to combat curse of dimensionality, reduce overfitting, speed up training, handle multicollinearity, and enable visualization.

**When to Use:**
- Too many features relative to samples
- Features are highly correlated
- Need to visualize high-dimensional data
- Training is too slow

**Methods:**

| Category | Technique | Description |
|----------|-----------|-------------|
| **Feature Selection** | Filter (correlation) | Rank features statistically |
| **Feature Selection** | Wrapper (RFE) | Search feature subsets |
| **Feature Selection** | Embedded (Lasso) | Built into model |
| **Feature Extraction** | PCA | Linear, maximize variance |
| **Feature Extraction** | t-SNE, UMAP | Non-linear, for visualization |
| **Feature Extraction** | Autoencoder | Neural network compression |

**Python Code Example:**
```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# PCA - feature extraction
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X)

# SelectKBest - feature selection
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
```

---

### Question 40: Data Augmentation Techniques
- [ ] Done

**Definition:**  
Data augmentation artificially increases training data by creating modified copies of existing samples. This regularizes the model, prevents overfitting, and improves generalization.

**By Data Type:**

| Data Type | Techniques |
|-----------|------------|
| **Images** | Rotation, flip, crop, color jitter, Cutout, Mixup |
| **Text** | Synonym replacement, back-translation, random deletion |
| **Tabular** | SMOTE (for imbalanced), noise injection |

**Image Augmentation Example:**
```python
from torchvision import transforms

augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

**SMOTE for Imbalanced Data:**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

---

## Advanced Topics

### Question 41: Transfer Learning in Supervised Models
- [ ] Done

### Definition
Transfer learning reuses a model trained on a large general dataset as a starting point for a specific task with limited data.

### Why It Matters

| Benefit | Explanation |
|---------|-------------|
| **Overcome Data Scarcity** | Build accurate models with limited labeled data |
| **Faster Training** | Pre-trained weights converge quickly |
| **Better Performance** | Leverages knowledge from massive datasets |
| **Regularization** | Pre-trained features prevent overfitting |

### Two Main Approaches

#### 1. Feature Extraction
- Freeze pre-trained layers
- Only train new classification head
- Use when: very limited data

#### 2. Fine-Tuning
- Unfreeze some/all pre-trained layers
- Train end-to-end with low learning rate
- Use when: more data, need task-specific adaptation

### Examples

| Domain | Pre-trained Model | Trained On | Fine-tune For |
|--------|------------------|------------|---------------|
| **Vision** | ResNet, EfficientNet | ImageNet (1.2M images) | Medical imaging, defect detection |
| **NLP** | BERT, GPT | Web text corpus | Sentiment, Q&A, classification |

### Code Example (Vision)

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load pre-trained model without top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy')
```

### Key Point
Transfer learning is now the **default approach** in deep learning, not an optimization.

---

### Question 42: End-to-End Learning
- [ ] Done

**Definition:**  
End-to-end learning trains a single neural network to directly map raw inputs to final outputs, automatically learning all intermediate representations. It eliminates manual feature engineering but requires large data and compute.

**Traditional vs End-to-End:**

| Traditional Pipeline | End-to-End |
|---------------------|------------|
| Raw data → Manual features → Simple model | Raw data → Deep network → Output |
| Requires domain expertise | Learns features automatically |
| Modular, interpretable | Black box |
| Works with less data | Needs lots of data |

**Example (Image Classification):**
- Traditional: Extract SIFT features → Train SVM
- End-to-End: Feed raw pixels → CNN → Class prediction

**Advantages:**
- No manual feature engineering
- Can learn optimal features for the task
- Simpler pipeline

**Disadvantages:**
- Requires massive data
- Computationally expensive
- Less interpretable

---

### Question 43: Multitask Learning
- [ ] Done

**Definition:**  
Multitask learning trains a single model to perform multiple related tasks simultaneously, sharing representations in common layers. This acts as regularization, improves generalization, and is more efficient than separate models.

**Architecture:**
```
Input → Shared Layers → Task 1 Head → Output 1
                     → Task 2 Head → Output 2
```

**Loss Function:**
$$\mathcal{L}_{total} = w_1 \cdot \mathcal{L}_{task1} + w_2 \cdot \mathcal{L}_{task2}$$

**Benefits:**
- Regularization through shared representations
- Implicit data augmentation
- Faster learning from related signals
- Reduced computational cost

**Example (Self-driving car):**
Single model with shared CNN backbone:
- Task 1: Road segmentation
- Task 2: Object detection
- Task 3: Depth estimation

---

### Question 44: Framing RL as Supervised Learning
- [ ] Done

**Definition:**  
Reinforcement Learning can be framed as supervised learning through **Imitation Learning** (predict expert actions from states) or **Value Function Fitting** (predict Q-values using regression).

**Method 1: Imitation Learning (Behavioral Cloning)**
- Input (X): States observed by expert
- Label (y): Actions expert took
- Problem: Classification (discrete actions) or Regression (continuous actions)

**Method 2: Q-Learning as Regression**
- Input (X): State-action pairs
- Target (y): Calculated target Q-value = r + gamma * max Q(s', a')
- Problem: Regression

**Python Code Example (Behavioral Cloning):**
```python
# Expert demonstrations: (state, action) pairs
states = expert_states      # observations
actions = expert_actions    # what expert did

# Train supervised classifier to mimic expert
from sklearn.ensemble import RandomForestClassifier
policy = RandomForestClassifier()
policy.fit(states, actions)

# Agent uses learned policy
predicted_action = policy.predict(new_state)
```

**Limitation:** Imitation learning can't exceed expert performance.

---

### Question 45: Role of Attention Mechanisms
- [ ] Done

**Definition:**  
Attention mechanisms allow neural networks to dynamically focus on relevant parts of input when generating output. They compute weighted combinations of input elements, where weights indicate importance for the current task.

**The Problem Before Attention:**
- Entire input compressed to fixed-size vector (bottleneck)
- Lost information for long sequences

**How Attention Works:**
1. **Query:** Current decoder state asks "what's relevant?"
2. **Keys:** Each input element provides a key
3. **Scores:** Compute similarity (Query x Keys)
4. **Weights:** Softmax to get attention distribution
5. **Context:** Weighted sum of Values

**Mathematical Formulation:**
$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Benefits:**
- Handles long sequences
- Interpretable (visualize attention weights)
- Foundation of Transformers (BERT, GPT)

**Self-Attention:** Query, Key, Value all come from same sequence - each element attends to all others.

**Interview Tip:** Transformers replaced RNNs/LSTMs by using only attention (no recurrence).

---
