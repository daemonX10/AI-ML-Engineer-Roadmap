# Classification Algorithms Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What is classification in the context of machine learning?**

### Answer

**Definition:**
Classification is a supervised learning task where the model learns to map input features to discrete categorical output labels. Given a training set of (feature, label) pairs, the algorithm learns a decision function that predicts the class of new, unseen instances.

**Core Concepts:**
- Supervised learning: requires labeled training data
- Output is categorical (discrete classes), not continuous
- Binary classification: 2 classes (spam/not spam)
- Multi-class classification: more than 2 classes (digit recognition 0-9)
- Decision boundary separates different classes in feature space

**Mathematical Formulation:**
$$f: X \rightarrow Y$$
Where $X \in \mathbb{R}^d$ (feature space), $Y \in \{c_1, c_2, ..., c_k\}$ (class labels)

Goal: Find $\hat{f}$ that minimizes prediction error on unseen data

**Intuition:**
Think of classification as drawing boundaries in space to separate different groups. Like sorting emails into inbox vs spam based on keywords, sender, etc.

**Practical Relevance:**
- Email spam detection
- Medical diagnosis (disease/no disease)
- Image recognition (cat/dog/bird)
- Fraud detection
- Sentiment analysis

**Python Code Example:**
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
X, y = load_iris(return_X_y=True)

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```

**Interview Tips:**
- Distinguish classification from regression (discrete vs continuous output)
- Know when to use binary vs multi-class approaches
- Understand that classification outputs can be hard labels or probabilities

---

## Question 2
- [ ] Done

**How does a classification algorithm learn from data?**

### Answer

**Definition:**
A classification algorithm learns by iteratively adjusting its parameters to minimize the difference between predicted and actual class labels on training data. It optimizes a loss function using training examples to find the best decision boundary that generalizes to unseen data.

**Core Concepts:**
- Training data: labeled examples $(x_i, y_i)$
- Loss function: measures prediction error
- Optimization: gradient descent, closed-form solutions
- Generalization: ability to predict correctly on new data
- Iterative updates until convergence or stopping criteria

**Mathematical Formulation:**
$$\hat{\theta} = \arg\min_{\theta} \sum_{i=1}^{n} L(y_i, f(x_i; \theta)) + \lambda R(\theta)$$

Where:
- $L$ = loss function (cross-entropy, hinge loss)
- $f(x_i; \theta)$ = model prediction
- $R(\theta)$ = regularization term
- $\lambda$ = regularization strength

**Learning Steps (Algorithm):**
1. Initialize model parameters randomly
2. Forward pass: compute predictions for training batch
3. Calculate loss between predictions and true labels
4. Backward pass: compute gradients of loss w.r.t. parameters
5. Update parameters: $\theta = \theta - \eta \nabla L$
6. Repeat steps 2-5 until convergence

**Intuition:**
Like learning to distinguish apples from oranges - you see many examples, note the patterns (color, shape, size), and adjust your mental rules until you can accurately classify new fruits.

**Python Code Example:**
```python
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# SGD learns iteratively
model = SGDClassifier(loss='log_loss', max_iter=1000, learning_rate='optimal')
model.fit(X_train, y_train)

print(f"Train Accuracy: {model.score(X_train, y_train):.2f}")
print(f"Test Accuracy: {model.score(X_test, y_test):.2f}")
```

**Interview Tips:**
- Understand difference between batch, mini-batch, and stochastic gradient descent
- Know that different algorithms have different learning mechanisms
- Be ready to explain overfitting vs underfitting

---

## Question 3
- [ ] Done

**What is the role of a loss function in classification algorithms?**

### Answer

**Definition:**
A loss function quantifies how far the model's predictions are from the actual labels. It provides a single scalar value that the optimization algorithm minimizes during training, guiding the model to learn the correct mapping from features to classes.

**Core Concepts:**
- Measures prediction error
- Differentiable for gradient-based optimization
- Different loss functions suit different problems
- Lower loss = better predictions
- Training minimizes average loss over all samples

**Common Classification Loss Functions:**

| Loss Function | Formula | Use Case |
|--------------|---------|----------|
| Cross-Entropy | $-\sum y_i \log(\hat{y}_i)$ | Probabilistic outputs |
| Hinge Loss | $\max(0, 1 - y \cdot \hat{y})$ | SVM, margin-based |
| 0-1 Loss | $\mathbb{1}[y \neq \hat{y}]$ | Evaluation only |

**Mathematical Formulation:**
Binary Cross-Entropy Loss:
$$L = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

**Intuition:**
Loss function is like a scorecard that tells the model "how wrong" it is. High loss = very wrong predictions, low loss = predictions close to truth.

**Python Code Example:**
```python
import numpy as np

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

y_true = np.array([1, 0, 1, 1])
y_pred_good = np.array([0.9, 0.1, 0.8, 0.7])
y_pred_bad = np.array([0.1, 0.9, 0.2, 0.3])

print(f"Good predictions loss: {cross_entropy_loss(y_true, y_pred_good):.4f}")
print(f"Bad predictions loss: {cross_entropy_loss(y_true, y_pred_bad):.4f}")
```

**Interview Tips:**
- Cross-entropy penalizes confident wrong predictions heavily
- 0-1 loss is not differentiable, so we use surrogate losses
- Loss function choice impacts what the model optimizes for

---

## Question 4
- [ ] Done

**What are the differences between generative and discriminative models?**

### Answer

**Definition:**
**Discriminative models** learn the decision boundary directly by modeling $P(y|x)$. **Generative models** learn the data distribution $P(x|y)$ and $P(y)$, then use Bayes' theorem to compute $P(y|x)$.

**Core Concepts:**

| Aspect | Discriminative | Generative |
|--------|---------------|------------|
| Models | $P(y|x)$ directly | $P(x|y)$ and $P(y)$ |
| Focus | Decision boundary | Data distribution |
| Examples | Logistic Regression, SVM | Naive Bayes, GMM |
| Data needed | Less data usually | More data |

**Mathematical Formulation:**
Generative uses Bayes' theorem:
$$P(y|x) = \frac{P(x|y) \cdot P(y)}{P(x)}$$

**Intuition:**
- **Discriminative**: Learn what separates cats from dogs
- **Generative**: Learn what cats look like and what dogs look like separately

**Python Code Example:**
```python
from sklearn.linear_model import LogisticRegression  # Discriminative
from sklearn.naive_bayes import GaussianNB  # Generative
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=500, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

disc_model = LogisticRegression().fit(X_train, y_train)
gen_model = GaussianNB().fit(X_train, y_train)

print(f"Logistic Regression: {disc_model.score(X_test, y_test):.2f}")
print(f"Naive Bayes: {gen_model.score(X_test, y_test):.2f}")
```

**Interview Tips:**
- Generative models can generate synthetic data
- Discriminative usually has better classification accuracy
- Generative handles missing data better

---

## Question 5
- [ ] Done

**Explain the concept of decision boundaries in classification.**

### Answer

**Definition:**
A decision boundary is the hypersurface in feature space that separates different classes. Points on one side belong to one class, points on the other side belong to another class.

**Core Concepts:**
- Linear boundary: straight line (2D), hyperplane (higher-D)
- Non-linear boundary: curves, complex shapes
- Margin: distance between boundary and nearest points
- Points near boundary are harder to classify

**Mathematical Formulation:**
Linear decision boundary: $w^T x + b = 0$

Classification:
- Class 1 if $w^T x + b > 0$
- Class 0 if $w^T x + b < 0$

**Algorithm Types and Boundaries:**

| Algorithm | Boundary Type |
|-----------|--------------|
| Logistic Regression | Linear |
| SVM with RBF | Non-linear |
| Decision Tree | Axis-parallel |
| Neural Network | Complex non-linear |

**Python Code Example:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC

X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
model = SVC(kernel='rbf', gamma=2).fit(X, y)

# Plot
xx, yy = np.meshgrid(np.linspace(-2, 3, 100), np.linspace(-1.5, 2, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k')
plt.title("Non-linear Decision Boundary")
plt.show()
```

**Interview Tips:**
- More complex boundaries need more data to avoid overfitting
- Kernel trick enables non-linear boundaries with linear algorithms

---

## Question 6
- [ ] Done

**What is the "Curse of Dimensionality" and how does it affect classification?**

### Answer

**Definition:**
The Curse of Dimensionality refers to phenomena arising in high-dimensional spaces. As dimensions increase, data becomes sparse, distances lose meaning, and exponentially more data is needed for statistical significance.

**Core Concepts:**
- Data becomes sparse in high dimensions
- Distance metrics lose meaning
- Need exponentially more data
- Overfitting risk increases
- k-NN particularly affected

**Effects on Classification:**

| Problem | Description |
|---------|-------------|
| Distance concentration | All distances become similar |
| Empty space | Most of high-d space is empty |
| Overfitting | Too many parameters, too little data |

**Practical Solutions:**
- Dimensionality reduction: PCA, t-SNE
- Feature selection
- L1 regularization for sparse solutions
- Use tree-based algorithms (more robust)

**Python Code Example:**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

dimensions = [5, 20, 50, 100, 200]

for d in dimensions:
    X, y = make_classification(n_samples=500, n_features=d, 
                                n_informative=5, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X, y, cv=5)
    print(f"Dimensions: {d:3d} | Accuracy: {scores.mean():.3f}")
```

**Interview Tips:**
- k-NN particularly suffers in high dimensions
- Rule of thumb: need samples >> features
- Tree-based methods are more robust

---

## Question 7
- [ ] Done

**Briefly describe the working principle of Logistic Regression.**

### Answer

**Definition:**
Logistic Regression is a linear classifier that models the probability of a binary outcome using the sigmoid function. It finds a linear combination of features and outputs probabilities between 0 and 1.

**Core Concepts:**
- Linear model with sigmoid activation
- Outputs probability $P(y=1|x)$
- Decision boundary is linear
- Loss: Binary Cross-Entropy

**Mathematical Formulation:**
Linear combination: $z = w^T x + b$

Sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$

Probability: $P(y=1|x) = \sigma(w^T x + b)$

**Algorithm Steps:**
1. Initialize weights
2. Compute $z = w^T x + b$
3. Apply sigmoid: $\hat{y} = \sigma(z)$
4. Compute cross-entropy loss
5. Update weights via gradient descent
6. Repeat until convergence

**Python Code Example:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=5000).fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test):.2f}")
print(f"Sample probabilities: {model.predict_proba(X_test)[:3, 1]}")
```

**Interview Tips:**
- Logistic regression is for classification, NOT regression
- Can extend to multi-class using One-vs-Rest or Softmax
- Regularization (L1/L2) prevents overfitting

---

## Question 8
- [ ] Done

**Explain the concept of Support Vector Machines (SVM).**

### Answer

**Definition:**
SVM is a classifier that finds the optimal hyperplane maximizing the margin between classes. It focuses on support vectors (points closest to boundary) and uses kernel trick for non-linear classification.

**Core Concepts:**
- Maximum margin classifier
- Support vectors: points on margin boundary
- Soft margin: allows some misclassification (C parameter)
- Kernel trick: maps to higher dimensions

**Mathematical Formulation:**
Optimization:
$$\min_{w,b} \frac{1}{2}||w||^2 + C\sum_{i}\xi_i$$

Subject to: $y_i(w^T x_i + b) \geq 1 - \xi_i$

**Key Hyperparameters:**
- **C**: Regularization (high C = less margin)
- **kernel**: 'linear', 'rbf', 'poly'
- **gamma**: controls RBF kernel influence

**Python Code Example:**
```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X_train_scaled, y_train)
print(f"Accuracy: {svm.score(X_test_scaled, y_test):.2f}")
```

**Interview Tips:**
- Always scale features before SVM
- Large C = complex boundary, small C = simpler
- SVM doesn't output probabilities natively

---

## Question 9
- [ ] Done

**What is the Naive Bayes classifier and how does it work?**

### Answer

**Definition:**
Naive Bayes is a probabilistic classifier based on Bayes' theorem with a "naive" assumption that all features are conditionally independent given the class. It assigns the class with highest posterior probability.

**Core Concepts:**
- Generative model: learns $P(x|y)$
- Naive assumption: features are independent
- Fast training and prediction
- Works well with high-dimensional data

**Mathematical Formulation:**
$$P(y|x_1,...,x_n) \propto P(y) \prod_{i=1}^{n} P(x_i|y)$$

**Types:**
| Type | Use Case |
|------|----------|
| Gaussian NB | Real-valued features |
| Multinomial NB | Text (word counts) |
| Bernoulli NB | Binary features |

**Python Code Example:**
```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GaussianNB().fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2f}")
```

**Interview Tips:**
- Independence assumption rarely holds but still works well
- Very fast - good baseline for text classification
- Use Laplace smoothing for zero probabilities

---

## Question 10
- [ ] Done

**Describe how a Decision Tree works in classification tasks.**

### Answer

**Definition:**
A Decision Tree recursively splits data based on feature values to create homogeneous subsets. Each internal node tests a feature, each branch represents a decision, and each leaf represents a class prediction.

**Core Concepts:**
- Recursive binary splitting
- Split criteria: Gini impurity, Entropy
- Greedy algorithm
- No feature scaling needed

**Mathematical Formulation:**
Gini: $Gini = 1 - \sum_{i=1}^{c} p_i^2$

Entropy: $Entropy = -\sum_{i=1}^{c} p_i \log_2(p_i)$

Information Gain: $IG = Entropy(parent) - \sum \frac{n_j}{n} Entropy(child_j)$

**Algorithm Steps:**
1. Start at root with all data
2. For each feature, evaluate split points
3. Select split that maximizes information gain
4. Create child nodes
5. Recursively repeat
6. Stop when criteria met

**Python Code Example:**
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tree = DecisionTreeClassifier(max_depth=3, criterion='gini')
tree.fit(X_train, y_train)

print(f"Accuracy: {tree.score(X_test, y_test):.2f}")
```

**Interview Tips:**
- Overfits easily - use max_depth, min_samples_leaf
- Gini is faster, Entropy slightly better theoretically
- Foundation for Random Forest and Gradient Boosting

---

## Question 11
- [ ] Done

**What is a Random Forest and why is it often more effective than a single Decision Tree?**

### Answer

**Definition:**
Random Forest is an ensemble method that builds multiple decision trees using bootstrap sampling and random feature subsets, then aggregates predictions via majority voting.

**Why Better Than Single Tree:**

| Single Tree | Random Forest |
|------------|---------------|
| High variance | Low variance |
| Overfits easily | Resistant to overfitting |
| Unstable | Stable predictions |

**Core Concepts:**
- Bagging: bootstrap samples
- Feature randomness at each split
- Majority voting
- Feature subset: $m = \sqrt{d}$

**Algorithm Steps:**
1. Create n bootstrap samples
2. For each sample, train decision tree with random feature subsets
3. For prediction, each tree votes, majority wins

**Python Code Example:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

print(f"Accuracy: {rf.score(X_test, y_test):.2f}")
```

**Interview Tips:**
- n_estimators: more trees = better (diminishing returns after ~100)
- OOB error: built-in validation
- Random Forest rarely overfits

---

## Question 12
- [ ] Done

**Explain what Gradient Boosting Machines (GBM) are and how they work.**

### Answer

**Definition:**
Gradient Boosting builds trees sequentially, where each new tree corrects the errors (residuals) of the previous ensemble. It uses gradient descent to minimize a loss function.

**Core Concepts:**
- Sequential ensemble
- Each tree fits residuals
- Learning rate controls contribution
- Additive model

**Mathematical Formulation:**
$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

Where $h_m$ = tree fitted to residuals

**Bagging vs Boosting:**

| Bagging | Boosting |
|---------|----------|
| Parallel | Sequential |
| Reduces variance | Reduces bias |
| Harder to overfit | Can overfit |

**Python Code Example:**
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gbm.fit(X_train, y_train)

print(f"Accuracy: {gbm.score(X_test, y_test):.2f}")
```

**Interview Tips:**
- Lower learning_rate + more trees = better generalization
- Use early stopping
- XGBoost/LightGBM preferred for large datasets

---

## Question 13
- [ ] Done

**How does the k-Nearest Neighbours (k-NN) algorithm classify data points?**

### Answer

**Definition:**
k-NN classifies a new point by finding the k closest training samples and assigning the majority class among those neighbors. It's instance-based with no explicit training phase.

**Core Concepts:**
- Instance-based learning
- Distance metric: Euclidean, Manhattan
- k = number of neighbors
- Majority voting
- Sensitive to feature scaling

**Mathematical Formulation:**
Euclidean distance: $d(x, x') = \sqrt{\sum(x_i - x'_i)^2}$

Prediction: $\hat{y} = \text{mode}(y_{i_1}, ..., y_{i_k})$

**Choosing k:**
- Small k: high variance, noisy
- Large k: smoother, may miss patterns
- Odd k: avoids ties

**Python Code Example:**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features (important!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)
print(f"Accuracy: {knn.score(X_test_scaled, y_test):.2f}")
```

**Interview Tips:**
- Always scale features
- O(n) prediction time
- Suffers from curse of dimensionality

---

## Question 14
- [ ] Done

**What are Artificial Neural Networks and how can they be used for classification tasks?**

### Answer

**Definition:**
ANNs are computational models with interconnected layers of neurons. For classification, they learn hierarchical features and map inputs to class probabilities using forward propagation, with weights optimized via backpropagation.

**Core Concepts:**
- Layers: Input -> Hidden -> Output
- Activation: ReLU, Sigmoid, Softmax
- Forward propagation
- Backpropagation for gradients
- Cross-entropy loss

**Mathematical Formulation:**
Neuron: $z = w^T x + b, \quad a = \sigma(z)$

Softmax: $\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$

**Algorithm Steps:**
1. Initialize weights
2. Forward pass
3. Compute loss
4. Backward pass (gradients)
5. Update weights
6. Repeat for epochs

**Python Code Example:**
```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=500)
nn.fit(X_train_scaled, y_train)

print(f"Accuracy: {nn.score(X_test_scaled, y_test):.2f}")
```

**Interview Tips:**
- ReLU for hidden, Softmax for multi-class output
- Dropout prevents overfitting
- For tabular data, gradient boosting often wins

---

## Question 15
- [ ] Done

**What is cross-validation and why is it important?**

### Answer

**Definition:**
Cross-validation evaluates model performance by splitting data into multiple train-test folds, training on k-1 folds and testing on the remaining, then averaging results.

**Core Concepts:**
- Reduces variance in estimation
- Better utilizes limited data
- Detects overfitting
- k=5 or k=10 common

**Types:**
| Type | Use Case |
|------|----------|
| k-Fold | General purpose |
| Stratified k-Fold | Imbalanced data |
| Leave-One-Out | Very small datasets |
| Time Series Split | Sequential data |

**Python Code Example:**
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = LogisticRegression(max_iter=1000)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

**Interview Tips:**
- Always use stratified k-fold for imbalanced data
- Never tune hyperparameters on test set
- Use nested CV for unbiased model selection

---

## Question 16
- [ ] Done

**Explain the concept of hyperparameter tuning in the context of classification models.**

### Answer

**Definition:**
Hyperparameter tuning finds optimal configuration settings that control the learning process. Unlike parameters learned from data, hyperparameters are selected through systematic search and validation.

**Core Concepts:**
- Parameters: learned from data
- Hyperparameters: set before training
- Search methods: Grid, Random, Bayesian
- Use CV for validation

**Common Hyperparameters:**
| Algorithm | Key Hyperparameters |
|-----------|-------------------|
| SVM | C, kernel, gamma |
| Random Forest | n_estimators, max_depth |
| Neural Network | learning_rate, layers |

**Python Code Example:**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
```

**Interview Tips:**
- Random search often as good as grid search, much faster
- Use coarse-to-fine search
- Hyperparameter tuning on test set = data leakage

---

## Question 17
- [ ] Done

**What is model ensemble and how can it improve classification performance?**

### Answer

**Definition:**
Model ensemble combines multiple base models to produce a single superior prediction. By aggregating diverse models, ensembles reduce variance, bias, or both.

**Types:**
| Method | Approach | Reduces |
|--------|----------|---------|
| Bagging | Parallel, bootstrap | Variance |
| Boosting | Sequential, errors | Bias |
| Stacking | Meta-learner | Both |
| Voting | Majority vote | Variance |

**Mathematical Formulation:**
Hard Voting: $\hat{y} = \text{mode}(h_1(x), h_2(x), ..., h_n(x))$

**Python Code Example:**
```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

X, y = load_iris(return_X_y=True)

models = [
    ('lr', LogisticRegression(max_iter=200)),
    ('dt', DecisionTreeClassifier(max_depth=5)),
    ('svm', SVC(probability=True))
]

voting_clf = VotingClassifier(estimators=models, voting='soft')
scores = cross_val_score(voting_clf, X, y, cv=5)
print(f"Voting Ensemble: {scores.mean():.3f}")
```

**Interview Tips:**
- Diversity is key
- Soft voting usually better than hard voting
- Diminishing returns after ~5-10 models

---

## Question 18
- [ ] Done

**Explain the "Kernel trick" in SVMs and why it is useful.**

### Answer

**Definition:**
The kernel trick computes dot products in a high-dimensional feature space without explicitly transforming data. It allows SVMs to find non-linear decision boundaries by implicitly mapping data to higher dimensions.

**Core Concepts:**
- Maps data implicitly to high-D space
- Avoids computational cost of explicit transformation
- Enables non-linear classification

**Mathematical Formulation:**
Kernel function: $K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$

**Common Kernels:**
| Kernel | Formula | Use Case |
|--------|---------|----------|
| Linear | $x^T y$ | Linearly separable |
| RBF | $e^{-\gamma\|x-y\|^2}$ | General non-linear |
| Polynomial | $(x^T y + c)^d$ | Polynomial boundaries |

**Intuition:**
2D data not linearly separable (circle inside circle) can be "lifted" to 3D where a plane separates them. Kernel trick does this without computing actual 3D coordinates.

**Python Code Example:**
```python
from sklearn.svm import SVC
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=200, noise=0.1, factor=0.3)

linear_svm = SVC(kernel='linear').fit(X, y)
rbf_svm = SVC(kernel='rbf', gamma=2).fit(X, y)

print(f"Linear: {linear_svm.score(X, y):.2f}")
print(f"RBF: {rbf_svm.score(X, y):.2f}")
```

**Interview Tips:**
- RBF is safe default
- gamma controls kernel reach
- Kernel must satisfy Mercer's condition

---

## Question 19
- [ ] Done

**What are the advantages of using ensemble methods like Bagging and Boosting?**

### Answer

**Definition:**
Ensemble methods combine multiple models to improve prediction. Bagging reduces variance through parallel training; Boosting reduces bias through sequential error correction.

**Bagging Advantages:**
- Reduces variance (averaging effect)
- Handles overfitting
- Parallelizable
- Example: Random Forest

**Boosting Advantages:**
- Reduces bias
- Focuses on hard examples
- Often achieves highest accuracy
- Example: XGBoost, LightGBM

**Comparison:**
| Aspect | Bagging | Boosting |
|--------|---------|----------|
| Training | Parallel | Sequential |
| Reduces | Variance | Bias |
| Overfitting | Resistant | Can overfit |
| Base models | Independent | Dependent |

**Python Code Example:**
```python
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)

bagging = BaggingClassifier(n_estimators=50)
boosting = AdaBoostClassifier(n_estimators=50)

print(f"Bagging: {cross_val_score(bagging, X, y, cv=5).mean():.3f}")
print(f"Boosting: {cross_val_score(boosting, X, y, cv=5).mean():.3f}")
```

**Interview Tips:**
- Bagging works best when base model overfits
- Boosting works best when base model underfits
- Both improve stability and accuracy

---

## Question 20
- [ ] Done

**What are the challenges of using Neural Networks for classification problems?**

### Answer

**Definition:**
Neural networks face several challenges including data requirements, computational cost, interpretability, and optimization difficulties.

**Key Challenges:**

| Challenge | Description |
|-----------|-------------|
| Data hunger | Need large labeled datasets |
| Computational cost | Requires GPU, long training |
| Interpretability | Black box, hard to explain |
| Overfitting | Easy to overfit with many params |
| Hyperparameter tuning | Many choices (layers, neurons, lr) |
| Vanishing gradients | Deep networks hard to train |

**Solutions:**
- **Data**: Data augmentation, transfer learning
- **Overfitting**: Dropout, regularization, early stopping
- **Interpretability**: LIME, SHAP, attention
- **Gradients**: ReLU, batch normalization, residual connections

**Python Code Example:**
```python
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
import numpy as np

# Neural nets need more data
# Dropout equivalent in sklearn: alpha regularization
model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    alpha=0.01,  # L2 regularization
    early_stopping=True,
    validation_fraction=0.1
)
```

**Interview Tips:**
- For tabular data, gradient boosting often beats NNs
- Deep learning shines for images, text, speech
- Always try simpler models first

---

## Question 21
- [ ] Done

**What is one-class classification and in what scenarios is it used?**

### Answer

**Definition:**
One-class classification trains on only one class (normal) and learns to identify whether new samples belong to that class or are anomalies. Used when negative examples are rare or hard to define.

**Core Concepts:**
- Train only on positive/normal class
- Learn boundary around normal data
- New points outside boundary = anomaly
- Also called novelty/anomaly detection

**Use Cases:**
| Scenario | Normal Class | Anomaly |
|----------|-------------|---------|
| Fraud detection | Legitimate transactions | Fraudulent |
| Network intrusion | Normal traffic | Attacks |
| Medical diagnosis | Healthy patients | Rare diseases |
| Manufacturing | Normal products | Defects |

**Algorithms:**
- One-Class SVM
- Isolation Forest
- Local Outlier Factor (LOF)
- Autoencoders

**Python Code Example:**
```python
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import numpy as np

# Normal data (only positive class)
X_train = np.random.randn(100, 2)

# Test data (includes anomalies)
X_test = np.vstack([np.random.randn(50, 2), np.random.randn(10, 2) + 5])

# One-Class SVM
oc_svm = OneClassSVM(kernel='rbf', nu=0.1).fit(X_train)
predictions = oc_svm.predict(X_test)  # 1=normal, -1=anomaly

print(f"Detected anomalies: {sum(predictions == -1)}")
```

**Interview Tips:**
- Use when anomalies are rare or undefined
- nu parameter controls expected outlier fraction
- Isolation Forest often works better for high-dim data

---

## Question 22
- [ ] Done

**Explain how semi-supervised learning can be used for classification tasks.**

### Answer

**Definition:**
Semi-supervised learning uses both labeled and unlabeled data for training. When labeled data is scarce but unlabeled data is abundant, it leverages unlabeled data's structure to improve classification.

**Core Concepts:**
- Small labeled dataset + large unlabeled dataset
- Unlabeled data reveals data distribution
- Smoothness assumption: nearby points have same label
- Cluster assumption: points in same cluster share labels

**Techniques:**
| Method | Description |
|--------|-------------|
| Self-training | Train on labeled, predict unlabeled, retrain |
| Label propagation | Spread labels through graph |
| Co-training | Two views train each other |
| Pseudo-labeling | Use confident predictions as labels |

**When to Use:**
- Labeling is expensive (medical imaging)
- Lots of unlabeled data available
- Data has clear cluster structure

**Python Code Example:**
```python
from sklearn.semi_supervised import LabelSpreading
import numpy as np

# Some labeled, most unlabeled (-1)
X = np.random.randn(100, 2)
y = np.array([0]*10 + [1]*10 + [-1]*80)  # 80 unlabeled

model = LabelSpreading(kernel='rbf')
model.fit(X, y)

# Now can predict all points
predictions = model.transduction_
print(f"Propagated labels for {sum(y == -1)} unlabeled points")
```

**Interview Tips:**
- Assumes unlabeled data comes from same distribution
- Can hurt performance if assumption violated
- Combine with active learning for best results

---

## Question 23
- [ ] Done

**Explain the concept of transfer learning and its relevance to classification.**

### Answer

**Definition:**
Transfer learning uses knowledge from a model trained on one task/domain to improve learning on a related task/domain. Instead of training from scratch, leverage pre-trained features.

**Core Concepts:**
- Source task: where knowledge comes from
- Target task: where knowledge is applied
- Features learned on large dataset transfer to smaller
- Reduces data and compute requirements

**Types:**
| Type | Description |
|------|-------------|
| Feature extraction | Use pre-trained model as fixed feature extractor |
| Fine-tuning | Retrain some/all layers on new data |
| Domain adaptation | Same task, different domains |

**When to Use:**
- Limited labeled data in target domain
- Source and target tasks are related
- Pre-trained model available

**Python Code Example:**
```python
# Conceptual example (PyTorch-style)
# from torchvision import models
# 
# # Load pre-trained model
# model = models.resnet18(pretrained=True)
# 
# # Freeze early layers
# for param in model.parameters():
#     param.requires_grad = False
# 
# # Replace final layer for new task
# model.fc = nn.Linear(512, num_classes)

# Sklearn example with pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Feature extraction then classification
pipeline = Pipeline([
    ('features', PCA(n_components=50)),  # "transfer" dimensionality reduction
    ('classifier', LogisticRegression())
])
```

**Interview Tips:**
- Very effective for images (ImageNet pre-training)
- Fine-tune with small learning rate
- Works best when source and target are related

---

## Question 24
- [ ] Done

**What are adversarial examples and how do they affect classification models?**

### Answer

**Definition:**
Adversarial examples are inputs intentionally modified with small, often imperceptible perturbations that cause classifiers to make incorrect predictions with high confidence.

**Core Concepts:**
- Small perturbation -> large prediction change
- Often imperceptible to humans
- Exploit model's decision boundaries
- Transferable across models

**Attack Types:**
| Attack | Description |
|--------|-------------|
| FGSM | Fast Gradient Sign Method - single step |
| PGD | Projected Gradient Descent - iterative |
| C&W | Carlini & Wagner - optimization-based |

**Impact:**
- Security threat (autonomous vehicles, medical)
- Reveals model brittleness
- Challenge model robustness claims

**Mathematical Formulation:**
FGSM: $x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L(x, y))$

**Defenses:**
- Adversarial training: train on adversarial examples
- Input preprocessing
- Certified defenses
- Model ensembles

**Python Code Example:**
```python
import numpy as np

def fgsm_attack(model, x, y, epsilon=0.1):
    """Conceptual FGSM attack"""
    # Get gradient of loss w.r.t. input
    # grad = compute_gradient(model, x, y)
    
    # Perturb in direction of gradient sign
    # x_adv = x + epsilon * np.sign(grad)
    
    # Clip to valid range
    # x_adv = np.clip(x_adv, 0, 1)
    pass

# In practice, use libraries like CleverHans, ART
```

**Interview Tips:**
- Adversarial examples highlight model limitations
- Robustness vs accuracy trade-off
- Active research area in ML security

---

## Question 25
- [ ] Done

**What are the theoretical foundations of statistical learning theory for classification?**

### Answer

**Definition:**
Statistical learning theory provides mathematical framework for understanding generalization - how well a model trained on finite data performs on unseen data. Key concepts include PAC learning, VC dimension, and bias-variance tradeoff.

**Core Concepts:**

**1. PAC Learning (Probably Approximately Correct):**
- Algorithm can learn concept with high probability (1-δ)
- Error at most ε given sufficient samples

**2. Generalization Bound:**
$$R(h) \leq \hat{R}(h) + \sqrt{\frac{VC(H) \log(2n/VC(H)) + \log(2/\delta)}{n}}$$

Where:
- $R(h)$ = true risk
- $\hat{R}(h)$ = empirical risk
- $VC(H)$ = VC dimension
- $n$ = sample size

**3. Bias-Variance Tradeoff:**
$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

**Key Theorems:**
- No Free Lunch: No universal best algorithm
- VC Dimension bounds: complexity vs generalization

**Interview Tips:**
- Understand that more complex models (higher VC) need more data
- Regularization controls model complexity
- Theory guides practical model selection

---

## Question 26
- [ ] Done

**How do you analyze the sample complexity and generalization bounds for classifiers?**

### Answer

**Definition:**
Sample complexity is the minimum number of training samples needed to achieve a specified generalization error with high probability. Generalization bounds quantify the gap between training and test error.

**Key Concepts:**

**Sample Complexity:**
$$n \geq O\left(\frac{VC(H) + \log(1/\delta)}{\epsilon^2}\right)$$

Where:
- $n$ = samples needed
- $VC(H)$ = hypothesis class complexity
- $\epsilon$ = error tolerance
- $\delta$ = failure probability

**Generalization Gap:**
$$|R_{true}(h) - R_{train}(h)| \leq O\left(\sqrt{\frac{VC(H)}{n}}\right)$$

**Practical Implications:**
| Factor | Effect on Sample Complexity |
|--------|---------------------------|
| More features | Need more samples |
| Complex model | Need more samples |
| Tighter bounds | Need more samples |

**Rules of Thumb:**
- 10x samples per feature
- 5-10x samples per class
- Validation curves reveal if more data helps

**Interview Tips:**
- More complex models need more data
- Regularization reduces effective complexity
- Learning curves diagnose data sufficiency

---

## Question 27
- [ ] Done

**What is the VC-dimension and its role in classification algorithm analysis?**

### Answer

**Definition:**
VC (Vapnik-Chervonenkis) dimension is the maximum number of points that can be shattered (perfectly separated in all possible labelings) by a hypothesis class. It measures the capacity/complexity of a classifier.

**Core Concepts:**
- Shattering: classify all 2^n labelings correctly
- Higher VC = more complex model
- Connects to generalization bounds
- Finite VC -> learnable

**Examples:**
| Model | VC Dimension |
|-------|-------------|
| Line in 2D | 3 |
| Hyperplane in R^d | d + 1 |
| Linear classifier | d + 1 (d features) |
| Decision stump | 2 |
| k-NN (k=1) | Infinite |

**Generalization Bound:**
$$P[\text{error}] \leq \hat{\text{error}} + O\left(\sqrt{\frac{VC \cdot \log(n/VC) + \log(1/\delta)}{n}}\right)$$

**Intuition:**
- 3 non-collinear points can be shattered by a line
- 4 points (XOR) cannot be shattered by a line
- So VC dimension of lines in 2D = 3

**Interview Tips:**
- VC dimension independent of data distribution
- Regularization effectively reduces VC dimension
- Practical models often have manageable VC

---

## Question 28
- [ ] Done

**How do you implement ensemble classification methods effectively?**

### Answer

**Definition:**
Effective ensemble implementation requires diverse base models, proper aggregation, and avoiding data leakage. Key strategies include bagging, boosting, stacking, and voting.

**Implementation Steps:**

1. **Choose base models** (diverse, complementary)
2. **Training strategy** (parallel vs sequential)
3. **Aggregation method** (voting, averaging, stacking)
4. **Validation** (proper CV to avoid leakage)

**Best Practices:**
| Practice | Reason |
|----------|--------|
| Diverse models | Different errors cancel |
| Proper CV for stacking | Avoid overfitting |
| Feature subset | Increase diversity |
| Hyperparameter diversity | Different perspectives |

**Python Code Example:**
```python
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Diverse base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=50)),
    ('svm', SVC(probability=True)),
    ('lr', LogisticRegression())
]

# Voting
voting = VotingClassifier(estimators=base_models, voting='soft')

# Stacking (uses CV internally)
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5
)
```

**Interview Tips:**
- Diversity is more important than individual accuracy
- Stacking needs proper CV (uses out-of-fold predictions)
- Diminishing returns after 5-10 models

---

## Question 29
- [ ] Done

**What are voting classifiers and their different voting strategies?**

### Answer

**Definition:**
Voting classifiers combine predictions from multiple models through voting. Hard voting uses majority class labels; soft voting averages class probabilities.

**Voting Strategies:**

| Strategy | Method | When to Use |
|----------|--------|-------------|
| Hard voting | Majority of predicted labels | Models output labels only |
| Soft voting | Average probabilities | Models output probabilities |
| Weighted voting | Weighted average | Models have different reliability |

**Mathematical Formulation:**
Hard: $\hat{y} = \text{mode}(h_1(x), ..., h_n(x))$

Soft: $\hat{y} = \arg\max_c \frac{1}{n}\sum_i P_i(c|x)$

Weighted: $\hat{y} = \arg\max_c \sum_i w_i \cdot P_i(c|x)$

**Python Code Example:**
```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

models = [
    ('lr', LogisticRegression()),
    ('dt', DecisionTreeClassifier()),
    ('nb', GaussianNB())
]

# Hard voting
hard_vote = VotingClassifier(estimators=models, voting='hard')

# Soft voting (requires probability outputs)
soft_vote = VotingClassifier(estimators=models, voting='soft')

# Weighted soft voting
weighted_vote = VotingClassifier(
    estimators=models, 
    voting='soft',
    weights=[2, 1, 1]  # LR weighted higher
)
```

**Interview Tips:**
- Soft voting usually better (uses confidence)
- All models must support predict_proba for soft voting
- Weight based on CV performance

---

## Question 30
- [ ] Done

**How do you implement stacking and blending for classification?**

### Answer

**Definition:**
Stacking trains a meta-learner on base model predictions. Blending is similar but uses a holdout set instead of cross-validation. Both leverage strengths of different models.

**Stacking Steps:**
1. Split data into K folds
2. For each fold, train base models on K-1 folds
3. Predict on held-out fold -> meta-features
4. Stack meta-features from all folds
5. Train meta-learner on meta-features

**Blending Steps:**
1. Split into train/holdout
2. Train base models on train
3. Predict on holdout -> meta-features
4. Train meta-learner on holdout predictions

**Comparison:**
| Aspect | Stacking | Blending |
|--------|----------|----------|
| Data usage | All data via CV | Loses holdout |
| Leakage risk | Lower with proper CV | Simpler, less risk |
| Implementation | Complex | Simple |

**Python Code Example:**
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Stacking with sklearn
base_models = [
    ('rf', RandomForestClassifier(n_estimators=50)),
    ('svm', SVC(probability=True))
]

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5,  # Cross-validation for meta-features
    stack_method='predict_proba'
)

# Manual blending
# X_train, X_blend, y_train, y_blend = train_test_split(X, y, test_size=0.3)
# Train base on X_train, predict on X_blend
# Train meta-learner on blend predictions
```

**Interview Tips:**
- Use CV in stacking to avoid leakage
- Meta-learner should be simple (logistic regression)
- Blending is easier but wastes data

---

## Question 31
- [ ] Done

**What is cascading classification and when is it useful?**

### Answer

**Definition:**
Cascading classification uses a sequence of classifiers where each stage filters out "easy" samples, passing only difficult ones to the next stage. Early stages are fast/simple; later stages are more complex.

**Core Concepts:**
- Sequential decision making
- Early rejection of easy cases
- Progressively complex classifiers
- Reduces computation on average

**Use Cases:**
| Application | Stage 1 | Stage 2 |
|-------------|---------|---------|
| Face detection | Haar features | CNN |
| Spam filtering | Rule-based | ML model |
| Medical screening | Simple tests | Expensive tests |

**Benefits:**
- Computational efficiency
- Handle class imbalance
- Different cost at stages

**Python Code Example:**
```python
class CascadeClassifier:
    def __init__(self, models, thresholds):
        self.models = models  # List of classifiers
        self.thresholds = thresholds  # Confidence thresholds
    
    def predict(self, X):
        predictions = []
        remaining_idx = list(range(len(X)))
        
        for model, threshold in zip(self.models, self.thresholds):
            if not remaining_idx:
                break
            X_remaining = X[remaining_idx]
            proba = model.predict_proba(X_remaining)
            
            for i, idx in enumerate(remaining_idx[:]):
                if proba[i].max() >= threshold:
                    predictions.append((idx, proba[i].argmax()))
                    remaining_idx.remove(idx)
        
        # Final stage handles remaining
        return predictions
```

**Interview Tips:**
- Famous example: Viola-Jones face detection
- Balance speed vs accuracy at each stage
- Useful when classes have very different frequencies

---

## Question 32
- [ ] Done

**How do you handle hierarchical classification problems?**

### Answer

**Definition:**
Hierarchical classification deals with class labels organized in a tree/DAG structure. Predictions must respect the hierarchy - a sample predicted as "Golden Retriever" is also a "Dog" and "Animal".

**Approaches:**

| Approach | Description |
|----------|-------------|
| Flat | Ignore hierarchy, predict leaf nodes |
| Local per node | Classifier at each node |
| Local per parent | Multi-class at each internal node |
| Global | Single model respecting hierarchy |

**Challenges:**
- Error propagation (wrong parent -> wrong child)
- Inconsistent predictions
- Class imbalance at different levels

**Python Code Example:**
```python
from sklearn.linear_model import LogisticRegression

class HierarchicalClassifier:
    def __init__(self, hierarchy):
        """hierarchy: dict mapping parent -> children"""
        self.hierarchy = hierarchy
        self.classifiers = {}
    
    def fit(self, X, y_hierarchy):
        # Train classifier at each internal node
        for parent, children in self.hierarchy.items():
            mask = y_hierarchy['level1'] == parent
            if mask.sum() > 0:
                self.classifiers[parent] = LogisticRegression()
                self.classifiers[parent].fit(
                    X[mask], 
                    y_hierarchy.loc[mask, 'level2']
                )
    
    def predict(self, X):
        # Predict top-down
        # First predict level 1, then level 2 based on level 1
        pass

# Example hierarchy: Animal -> [Dog, Cat], Dog -> [Labrador, Poodle]
```

**Interview Tips:**
- Top-down: fast but error propagates
- Bottom-up: start from leaves, aggregate
- Consider hierarchical loss functions

---

## Question 33
- [ ] Done

**What are the considerations for multi-label classification algorithms?**

### Answer

**Definition:**
Multi-label classification assigns multiple non-exclusive labels to each instance. Unlike multi-class (one label) or multi-output (multiple separate outputs), labels can be correlated.

**Key Considerations:**

| Aspect | Consideration |
|--------|---------------|
| Label correlation | Labels often co-occur |
| Imbalance | Some labels rare |
| Evaluation | Need multi-label metrics |
| Scalability | Many possible label combinations |

**Approaches:**
1. **Binary Relevance**: Independent classifier per label
2. **Classifier Chains**: Sequential, use previous predictions
3. **Label Powerset**: Treat each combination as class

**Metrics:**
- Hamming loss: fraction of wrong labels
- Subset accuracy: exact match
- Micro/Macro F1

**Python Code Example:**
```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss
import numpy as np

# Multi-label data: each sample has multiple labels
X = np.random.randn(100, 10)
Y = np.random.randint(0, 2, (100, 5))  # 5 binary labels

# Binary Relevance
br = OneVsRestClassifier(LogisticRegression())
br.fit(X, Y)

# Classifier Chains (captures label dependencies)
chain = ClassifierChain(LogisticRegression(), order='random')
chain.fit(X, Y)

Y_pred = br.predict(X)
print(f"Hamming Loss: {hamming_loss(Y, Y_pred):.3f}")
```

**Interview Tips:**
- Classifier chains capture label dependencies
- Binary relevance is simple but ignores correlation
- Label powerset doesn't scale with many labels

---

## Question 34
- [ ] Done

**How do you implement binary relevance and classifier chains for multi-label problems?**

### Answer

**Binary Relevance:**
Train independent binary classifier for each label. Simple, parallelizable, but ignores label correlations.

**Classifier Chains:**
Train classifiers sequentially, each using previous labels as additional features. Captures label dependencies.

**Comparison:**
| Aspect | Binary Relevance | Classifier Chains |
|--------|-----------------|-------------------|
| Correlation | Ignores | Captures |
| Training | Parallel | Sequential |
| Order dependency | No | Yes |

**Python Code Example:**
```python
from sklearn.linear_model import LogisticRegression
import numpy as np

class BinaryRelevance:
    def __init__(self, base_classifier):
        self.base = base_classifier
        self.classifiers = []
    
    def fit(self, X, Y):
        self.classifiers = []
        for i in range(Y.shape[1]):
            clf = clone(self.base)
            clf.fit(X, Y[:, i])
            self.classifiers.append(clf)
    
    def predict(self, X):
        return np.column_stack([clf.predict(X) for clf in self.classifiers])

class ClassifierChain:
    def __init__(self, base_classifier):
        self.base = base_classifier
        self.classifiers = []
    
    def fit(self, X, Y):
        self.classifiers = []
        for i in range(Y.shape[1]):
            clf = clone(self.base)
            # Add previous labels as features
            X_extended = np.hstack([X, Y[:, :i]]) if i > 0 else X
            clf.fit(X_extended, Y[:, i])
            self.classifiers.append(clf)
    
    def predict(self, X):
        predictions = []
        for i, clf in enumerate(self.classifiers):
            X_extended = np.hstack([X] + predictions) if predictions else X
            pred = clf.predict(X_extended)
            predictions.append(pred.reshape(-1, 1))
        return np.hstack(predictions)
```

**Interview Tips:**
- Chain order affects performance - try multiple orders
- Ensemble of chains with different orders is robust
- Use classifier chain when labels are correlated

---

## Question 35
- [ ] Done

**What is label powerset transformation in multi-label classification?**

### Answer

**Definition:**
Label Powerset (LP) transforms multi-label into multi-class by treating each unique label combination as a single class. A sample with labels {A, B} becomes class "A_B".

**Core Concepts:**
- Each label combination = unique class
- Standard multi-class classifier
- Captures all label correlations
- Exponential growth with labels

**Example:**
| Labels | LP Class |
|--------|----------|
| {A} | 0 |
| {B} | 1 |
| {A, B} | 2 |
| {A, C} | 3 |
| {A, B, C} | 4 |

**Pros and Cons:**
| Pros | Cons |
|------|------|
| Captures correlations | Exponential classes (2^n) |
| Simple | Rare combinations = sparse classes |
| Standard classifier | May not see all combinations |

**Python Code Example:**
```python
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
import numpy as np

def label_powerset_transform(Y):
    """Convert multi-label to single labels"""
    # Each row -> tuple -> string -> unique id
    label_strings = ['_'.join(map(str, row)) for row in Y]
    unique_labels = list(set(label_strings))
    label_to_id = {l: i for i, l in enumerate(unique_labels)}
    return np.array([label_to_id[l] for l in label_strings]), label_to_id

# Example
Y = np.array([[1, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]])
y_lp, mapping = label_powerset_transform(Y)
print(f"Label Powerset classes: {y_lp}")
print(f"Mapping: {mapping}")

# Train standard classifier
model = LogisticRegression()
# model.fit(X, y_lp)
```

**Interview Tips:**
- Use when few labels and many samples per combination
- Doesn't scale beyond ~15 labels
- RAkEL: subset of label powerset for scalability

---

## Question 36
- [ ] Done

**How do you handle ordinal classification with ordered class labels?**

### Answer

**Definition:**
Ordinal classification handles classes with natural ordering (e.g., low < medium < high). Standard classifiers ignore this ordering; ordinal methods exploit it.

**Approaches:**

| Method | Description |
|--------|-------------|
| Regression | Treat as continuous (loses discreteness) |
| Binary decomposition | K-1 binary classifiers: y > k? |
| Ordinal encoding | Threshold model |
| Specialized loss | Penalize far predictions more |

**Binary Decomposition:**
For classes {1, 2, 3, 4}:
- Classifier 1: y > 1?
- Classifier 2: y > 2?
- Classifier 3: y > 3?

Final prediction: count "yes" answers + 1

**Python Code Example:**
```python
from sklearn.linear_model import LogisticRegression
import numpy as np

class OrdinalClassifier:
    def __init__(self, base_classifier):
        self.base = base_classifier
        self.classifiers = []
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.classifiers = []
        
        # K-1 binary classifiers
        for k in self.classes[:-1]:
            clf = clone(self.base)
            binary_y = (y > k).astype(int)
            clf.fit(X, binary_y)
            self.classifiers.append(clf)
    
    def predict(self, X):
        predictions = np.zeros(len(X))
        for i, clf in enumerate(self.classifiers):
            predictions += clf.predict(X)
        return self.classes[predictions.astype(int)]

# Example: ratings 1-5
y = np.array([1, 2, 3, 4, 5, 2, 3, 4])
# Train ordinal classifier
```

**Interview Tips:**
- MAE better than accuracy for ordinal
- Binary decomposition maintains monotonicity
- Consider weighted loss penalizing distant errors

---

## Question 37
- [ ] Done

**What are cost-sensitive classification algorithms and their applications?**

### Answer

**Definition:**
Cost-sensitive classification accounts for different misclassification costs. Instead of minimizing error rate, minimize total cost where different errors have different penalties.

**Core Concepts:**
- Cost matrix: C[i,j] = cost of predicting j when true is i
- Weight samples by misclassification cost
- Change decision threshold
- Different from class imbalance (though related)

**Cost Matrix Example (Medical):**
|  | Predict Healthy | Predict Sick |
|--|----------------|--------------|
| Actually Healthy | 0 | 10 (unnecessary treatment) |
| Actually Sick | 100 (missed diagnosis) | 0 |

**Implementation Methods:**
1. **Resampling**: Oversample high-cost classes
2. **Class weights**: Weight samples by cost
3. **Threshold moving**: Adjust decision threshold
4. **Cost-sensitive algorithms**: Built-in cost handling

**Python Code Example:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Cost-sensitive via class weights
# Higher weight = higher cost of misclassifying that class
model = LogisticRegression(class_weight={0: 1, 1: 10})  # 10x cost for missing class 1

# Or sample_weight
sample_weights = np.where(y == 1, 10, 1)
model.fit(X, y, sample_weight=sample_weights)

# Threshold adjustment for cost-sensitive prediction
def cost_sensitive_predict(model, X, cost_fp, cost_fn):
    proba = model.predict_proba(X)[:, 1]
    # Optimal threshold: cost_fp / (cost_fp + cost_fn)
    threshold = cost_fp / (cost_fp + cost_fn)
    return (proba >= threshold).astype(int)
```

**Applications:**
- Medical diagnosis (missing disease vs false alarm)
- Fraud detection (fraud cost >> investigation cost)
- Credit scoring (default cost >> rejection cost)

**Interview Tips:**
- Don't confuse with class imbalance
- Cost-based threshold: minimize expected cost
- ROC curve helps find optimal threshold

---

## Question 38
- [ ] Done

**How do you handle classification with reject option and uncertainty quantification?**

### Answer

**Definition:**
Reject option allows classifier to abstain from prediction when uncertain, rather than making a potentially wrong prediction. Uncertainty quantification measures confidence in predictions.

**Core Concepts:**
- Abstain when confidence below threshold
- Trade-off: coverage vs accuracy
- Uncertainty types: aleatoric (data noise), epistemic (model uncertainty)

**Implementation:**
| Method | Description |
|--------|-------------|
| Threshold | Reject if max probability < threshold |
| Ambiguity | Reject if top-2 probabilities close |
| Ensemble | Reject if models disagree |

**Python Code Example:**
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

class ClassifierWithReject:
    def __init__(self, base_classifier, threshold=0.7):
        self.clf = base_classifier
        self.threshold = threshold
    
    def fit(self, X, y):
        self.clf.fit(X, y)
    
    def predict_with_reject(self, X):
        proba = self.clf.predict_proba(X)
        max_proba = proba.max(axis=1)
        predictions = self.clf.predict(X)
        
        # -1 indicates rejection
        predictions[max_proba < self.threshold] = -1
        return predictions, max_proba

# Usage
clf = ClassifierWithReject(LogisticRegression(), threshold=0.8)
# clf.fit(X_train, y_train)
# preds, confidences = clf.predict_with_reject(X_test)
# rejected = (preds == -1).sum()
```

**Uncertainty Quantification:**
- **MC Dropout**: Run multiple forward passes with dropout
- **Ensemble variance**: Disagreement among models
- **Bayesian methods**: Posterior predictive distribution

**Interview Tips:**
- Reject option useful in high-stakes applications
- Balance rejection rate with accuracy improvement
- Rejected samples can go to human expert

---

## Question 39
- [ ] Done

**What are probabilistic classifiers and how do they provide uncertainty estimates?**

### Answer

**Definition:**
Probabilistic classifiers output class probabilities rather than just labels, providing uncertainty estimates about predictions. The probability reflects confidence in the prediction.

**Core Concepts:**
- Output: P(y|x) for all classes
- Well-calibrated: probability matches true frequency
- Enables risk-aware decisions
- Supports threshold tuning

**Examples:**
| Classifier | Probabilistic? |
|-----------|---------------|
| Logistic Regression | Yes (native) |
| Naive Bayes | Yes (native) |
| SVM | Needs Platt scaling |
| Random Forest | Yes (via voting) |
| Neural Network | Yes (softmax) |

**Calibration:**
- Well-calibrated: If model says 80%, it's correct 80% of time
- Overconfident: Probabilities too extreme
- Underconfident: Probabilities too moderate

**Python Code Example:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Probabilistic prediction
model = LogisticRegression().fit(X_train, y_train)
probabilities = model.predict_proba(X_test)

# Check calibration
prob_true, prob_pred = calibration_curve(y_test, probabilities[:, 1], n_bins=10)

plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], '--', label='Perfect calibration')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve')
```

**Interview Tips:**
- Probabilities enable cost-sensitive decisions
- Most classifiers need calibration
- Use calibration_curve to diagnose

---

## Question 40
- [ ] Done

**How do you implement calibration for classification probability outputs?**

### Answer

**Definition:**
Calibration adjusts predicted probabilities so they accurately reflect true likelihoods. A calibrated classifier's 70% confidence predictions should be correct ~70% of the time.

**Why Calibrate:**
- Many classifiers output poorly calibrated probabilities
- Neural networks often overconfident
- Tree ensembles often underconfident
- Critical for decision-making

**Methods:**
| Method | Description |
|--------|-------------|
| Platt Scaling | Logistic regression on logits |
| Isotonic Regression | Non-parametric monotonic mapping |
| Temperature Scaling | Single parameter scaling (NNs) |

**Implementation Steps:**
1. Train model on training set
2. Get probability predictions on validation set
3. Fit calibration method on (predictions, true labels)
4. Apply calibration to test predictions

**Python Code Example:**
```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Uncalibrated
rf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
uncalibrated_proba = rf.predict_proba(X_test)

# Calibrated with Platt scaling (sigmoid)
rf_sigmoid = CalibratedClassifierCV(rf, method='sigmoid', cv=5)
rf_sigmoid.fit(X_train, y_train)

# Calibrated with isotonic regression
rf_isotonic = CalibratedClassifierCV(rf, method='isotonic', cv=5)
rf_isotonic.fit(X_train, y_train)

calibrated_proba = rf_sigmoid.predict_proba(X_test)
```

**Interview Tips:**
- Platt: parametric, works with less data
- Isotonic: non-parametric, needs more data
- Always use held-out data for calibration

---

## Question 41
- [ ] Done

**What is Platt scaling and isotonic regression for probability calibration?**

### Answer

**Platt Scaling:**
Fits a logistic regression on the classifier's output scores to map them to calibrated probabilities.

$$P(y=1|f) = \frac{1}{1 + e^{Af + B}}$$

Where f = classifier output, A and B learned from data.

**Isotonic Regression:**
Non-parametric method that fits a monotonically increasing function mapping scores to probabilities.

**Comparison:**
| Aspect | Platt Scaling | Isotonic Regression |
|--------|--------------|-------------------|
| Parametric | Yes (2 params) | No |
| Assumption | Sigmoid shape | Monotonicity only |
| Data needed | Less | More |
| Flexibility | Low | High |
| Overfitting | Lower risk | Higher risk |

**When to Use:**
- **Platt**: Small calibration set, well-separated classes
- **Isotonic**: Large data, non-sigmoid calibration needed

**Python Code Example:**
```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.isotonic import IsotonicRegression
import numpy as np

# Platt scaling (sigmoid)
svm = SVC()
platt_calibrated = CalibratedClassifierCV(svm, method='sigmoid', cv=5)

# Isotonic regression
isotonic_calibrated = CalibratedClassifierCV(svm, method='isotonic', cv=5)

# Manual Platt scaling
def platt_scaling(scores, y_true):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(scores.reshape(-1, 1), y_true)
    return lr

# Manual isotonic
def isotonic_calibration(scores, y_true):
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(scores, y_true)
    return ir
```

**Interview Tips:**
- SVM typically needs Platt scaling
- Deep learning uses temperature scaling
- Never calibrate on training data

---

## Question 42
- [ ] Done

**How do you handle classification in non-stationary environments?**

### Answer

**Definition:**
Non-stationary environments have data distributions that change over time (concept drift). Models trained on historical data may become outdated.

**Types of Drift:**
| Type | Description |
|------|-------------|
| Sudden | Abrupt change |
| Gradual | Slow transition |
| Recurring | Patterns reappear |
| Incremental | Small continuous changes |

**Detection Methods:**
- Monitor prediction accuracy
- Statistical tests (KS, Chi-square)
- Page-Hinkley test
- ADWIN (ADaptive WINdowing)

**Handling Strategies:**
1. **Periodic retraining**: Fixed schedule
2. **Triggered retraining**: When drift detected
3. **Online learning**: Continuous updates
4. **Ensemble with weights**: Recent models weighted higher

**Python Code Example:**
```python
from sklearn.linear_model import SGDClassifier
import numpy as np

class DriftAwareClassifier:
    def __init__(self, base_model, window_size=100, drift_threshold=0.1):
        self.model = base_model
        self.window_size = window_size
        self.threshold = drift_threshold
        self.performance_window = []
    
    def partial_fit(self, X, y):
        # Update model incrementally
        self.model.partial_fit(X, y, classes=[0, 1])
    
    def detect_drift(self, accuracy):
        self.performance_window.append(accuracy)
        if len(self.performance_window) > self.window_size:
            self.performance_window.pop(0)
        
        if len(self.performance_window) >= 2:
            recent = np.mean(self.performance_window[-10:])
            historical = np.mean(self.performance_window[:-10])
            if historical - recent > self.threshold:
                return True  # Drift detected
        return False

# Use SGDClassifier for online learning
model = SGDClassifier(loss='log_loss')
# model.partial_fit(X_batch, y_batch, classes=[0, 1])
```

**Interview Tips:**
- Common in fraud detection, recommendation systems
- Balance stability vs adaptability
- Monitor both accuracy and data distribution

---

## Question 43
- [ ] Done

**What are adaptive classification algorithms for concept drift?**

### Answer

**Definition:**
Adaptive classifiers automatically detect and adapt to changes in data distribution over time, maintaining performance without manual intervention.

**Key Algorithms:**

| Algorithm | Approach |
|-----------|----------|
| ADWIN | Adaptive sliding window |
| DDM | Drift Detection Method |
| EDDM | Early Drift Detection |
| Learn++.NSE | Ensemble with dynamic weights |

**ADWIN (Adaptive Windowing):**
- Maintains variable-size window
- Detects when two sub-windows have different distributions
- Shrinks window when drift detected

**Ensemble Approaches:**
- Train new model on recent data
- Weight models by recent performance
- Remove outdated models

**Python Code Example:**
```python
from river import drift
from river import linear_model
from river import metrics

# Using river library for streaming ML
model = linear_model.LogisticRegression()
drift_detector = drift.ADWIN()
metric = metrics.Accuracy()

def process_stream(X_stream, y_stream):
    for x, y_true in zip(X_stream, y_stream):
        # Predict
        y_pred = model.predict_one(x)
        
        # Update metric
        metric.update(y_true, y_pred)
        
        # Check for drift
        drift_detector.update(int(y_pred != y_true))
        
        if drift_detector.drift_detected:
            print("Drift detected! Resetting model...")
            model = linear_model.LogisticRegression()  # Reset
        
        # Update model
        model.learn_one(x, y_true)
```

**Interview Tips:**
- ADWIN is widely used and theoretically grounded
- Ensemble methods more robust to false alarms
- Trade-off: sensitivity to drift vs stability

---

## Question 44
- [ ] Done

**How do you implement online learning algorithms for classification?**

### Answer

**Definition:**
Online learning updates the model incrementally with each new sample or mini-batch, rather than retraining on entire dataset. Essential for streaming data and large-scale learning.

**Core Concepts:**
- One sample at a time (or mini-batch)
- No storage of full dataset
- Constant memory usage
- Adapts to changing data

**Algorithms with Online Support:**
| Algorithm | sklearn Method |
|-----------|---------------|
| SGD Classifier | partial_fit |
| Perceptron | partial_fit |
| Naive Bayes | partial_fit |
| PassiveAggressive | partial_fit |

**Python Code Example:**
```python
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Online learning with SGD
model = SGDClassifier(loss='log_loss')
classes = np.array([0, 1])

# Simulate streaming data
for batch_idx in range(10):
    X_batch = np.random.randn(100, 10)
    y_batch = np.random.randint(0, 2, 100)
    
    # Incremental update
    model.partial_fit(X_batch, y_batch, classes=classes)
    
    print(f"Batch {batch_idx}: trained on {(batch_idx+1)*100} samples")

# For Naive Bayes (good for text streams)
nb = MultinomialNB()
# nb.partial_fit(X_batch, y_batch, classes=classes)
```

**Key Considerations:**
- Initial warm-up period may be unstable
- Learning rate scheduling important
- Balance plasticity vs stability

**Interview Tips:**
- Must specify classes in first partial_fit call
- SGD with constant learning rate can forget old patterns
- Combine with drift detection for best results

---

## Question 45
- [ ] Done

**What is incremental learning and its applications in classification?**

### Answer

**Definition:**
Incremental learning updates models with new data without forgetting previously learned knowledge. Unlike online learning (single samples), incremental learning can use batches and focuses on avoiding catastrophic forgetting.

**Core Concepts:**
- Learn from new data without full retraining
- Preserve knowledge from old data
- Handle new classes over time
- Memory-efficient

**Challenges:**
| Challenge | Description |
|-----------|-------------|
| Catastrophic forgetting | New data overwrites old knowledge |
| Class imbalance | New classes may dominate |
| Stability-plasticity | Balance old vs new |

**Techniques:**
1. **Replay**: Store and replay old samples
2. **Regularization**: Penalize changes to important weights
3. **Dynamic architecture**: Add capacity for new knowledge
4. **Knowledge distillation**: Use old model to guide new

**Python Code Example:**
```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class IncrementalRandomForest:
    def __init__(self, n_estimators_per_batch=10):
        self.n_per_batch = n_estimators_per_batch
        self.trees = []
    
    def partial_fit(self, X, y):
        # Train new trees on new data
        new_forest = RandomForestClassifier(n_estimators=self.n_per_batch)
        new_forest.fit(X, y)
        
        # Add to existing ensemble
        self.trees.extend(new_forest.estimators_)
    
    def predict(self, X):
        # Aggregate predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        from scipy import stats
        return stats.mode(predictions, axis=0)[0].flatten()

# Scikit-learn doesn't natively support incremental RF
# Use warm_start parameter for some control
rf = RandomForestClassifier(n_estimators=10, warm_start=True)
rf.fit(X1, y1)
rf.n_estimators = 20  # Add more trees
rf.fit(X1, y1)  # Must use same data, adds trees
```

**Applications:**
- Robotics (continuous learning)
- Personalization (user adaptation)
- Edge devices (limited memory)

---

## Question 46
- [ ] Done

**How do you handle classification with limited labeled data?**

### Answer

**Definition:**
When labeled data is scarce, standard supervised learning fails. Techniques like semi-supervised learning, transfer learning, data augmentation, and few-shot learning help.

**Strategies:**

| Strategy | Description |
|----------|-------------|
| Semi-supervised | Use unlabeled data |
| Transfer learning | Leverage pre-trained models |
| Data augmentation | Create synthetic samples |
| Active learning | Strategically label samples |
| Few-shot learning | Learn from few examples |

**Python Code Example:**
```python
# 1. Semi-supervised: Label propagation
from sklearn.semi_supervised import LabelPropagation
# Only 10% labeled
y_partial = y.copy()
y_partial[10:] = -1  # Mark as unlabeled
model = LabelPropagation().fit(X, y_partial)

# 2. Data augmentation (for images/text)
# from imgaug import augmenters as iaa
# aug = iaa.Sequential([iaa.Fliplr(0.5), iaa.Rotate((-10, 10))])

# 3. Transfer learning (conceptual)
# pretrained = load_pretrained_model()
# pretrained.freeze_layers()
# pretrained.add_new_head(num_classes)
# pretrained.fit(X_small, y_small)

# 4. Self-training
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
self_training = SelfTrainingClassifier(SVC(probability=True))
self_training.fit(X, y_partial)
```

**Interview Tips:**
- Transfer learning most effective for images/text
- Semi-supervised requires same distribution assumption
- Active learning reduces labeling cost

---

## Question 47
- [ ] Done

**What are semi-supervised classification techniques and their benefits?**

### Answer

**Definition:**
Semi-supervised learning uses both labeled and unlabeled data. Benefits: better generalization, reduced labeling cost, leverages data structure.

**Assumptions:**
1. **Smoothness**: Nearby points have same label
2. **Cluster**: Points in same cluster share labels
3. **Manifold**: Data lies on low-dimensional manifold

**Techniques:**
| Technique | Description |
|-----------|-------------|
| Self-training | Train, predict, add confident predictions |
| Label propagation | Spread labels through graph |
| Co-training | Two views train each other |
| Generative models | Model P(x,y) |

**Benefits:**
- Reduced labeling cost
- Better use of available data
- Improved generalization
- Captures data structure

**Python Code Example:**
```python
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
import numpy as np

# Create data: few labeled, many unlabeled
X = np.random.randn(1000, 10)
y = np.random.randint(0, 3, 1000)
y_semi = y.copy()
y_semi[50:] = -1  # Only 50 labeled, rest unlabeled

# Label Propagation
lp = LabelPropagation(kernel='rbf', gamma=0.25)
lp.fit(X, y_semi)
print(f"Label Propagation: {lp.score(X, y):.3f}")

# Self-Training
st = SelfTrainingClassifier(SVC(probability=True, gamma='auto'))
st.fit(X, y_semi)
print(f"Self-Training: {st.score(X, y):.3f}")
```

**Interview Tips:**
- Requires labeled and unlabeled from same distribution
- Can hurt if assumptions violated
- Combine with active learning for best results

---

## Question 48
- [ ] Done

**How do you implement self-training and co-training for classification?**

### Answer

**Self-Training:**
1. Train classifier on labeled data
2. Predict unlabeled data
3. Add confident predictions to training set
4. Repeat until convergence

**Co-Training:**
1. Split features into two views
2. Train classifier on each view
3. Each classifier labels data for the other
4. Repeat

**Python Code Example:**
```python
from sklearn.linear_model import LogisticRegression
import numpy as np

class SelfTraining:
    def __init__(self, base_classifier, threshold=0.9, max_iter=10):
        self.clf = base_classifier
        self.threshold = threshold
        self.max_iter = max_iter
    
    def fit(self, X, y):
        # y contains -1 for unlabeled
        labeled_mask = y != -1
        
        for iteration in range(self.max_iter):
            # Train on labeled
            self.clf.fit(X[labeled_mask], y[labeled_mask])
            
            # Predict unlabeled
            unlabeled_mask = ~labeled_mask
            if not unlabeled_mask.any():
                break
            
            proba = self.clf.predict_proba(X[unlabeled_mask])
            confident = proba.max(axis=1) >= self.threshold
            
            if not confident.any():
                break
            
            # Add confident predictions
            unlabeled_indices = np.where(unlabeled_mask)[0]
            confident_indices = unlabeled_indices[confident]
            y[confident_indices] = proba[confident].argmax(axis=1)
            labeled_mask[confident_indices] = True
            
            print(f"Iter {iteration}: Added {confident.sum()} samples")
        
        return self

class CoTraining:
    def __init__(self, clf1, clf2, threshold=0.9):
        self.clf1 = clf1  # For view 1
        self.clf2 = clf2  # For view 2
        self.threshold = threshold
    
    def fit(self, X1, X2, y):
        # X1, X2 are two views of the same data
        # Implementation similar to self-training
        # but clf1 labels for clf2 and vice versa
        pass
```

**Interview Tips:**
- Self-training: simple but can reinforce errors
- Co-training: requires natural feature split
- Threshold controls quality vs quantity tradeoff

---

## Question 49
- [ ] Done

**What is active learning and its role in classification with limited labels?**

### Answer

**Definition:**
Active learning selects which samples to label, maximizing information gain per label. The model queries an oracle (human) for labels on the most informative samples.

**Core Concepts:**
- Query strategy selects samples
- Human labels selected samples
- Model improves with fewer labels
- Iterative process

**Query Strategies:**
| Strategy | Select samples that... |
|----------|----------------------|
| Uncertainty | Model is least certain |
| Query-by-committee | Models disagree most |
| Expected model change | Would change model most |
| Diversity | Cover feature space |

**Python Code Example:**
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

class ActiveLearner:
    def __init__(self, base_classifier):
        self.clf = base_classifier
        self.X_labeled = None
        self.y_labeled = None
    
    def query_uncertainty(self, X_pool, n_samples=10):
        """Select most uncertain samples"""
        proba = self.clf.predict_proba(X_pool)
        uncertainty = 1 - proba.max(axis=1)
        return np.argsort(uncertainty)[-n_samples:]
    
    def query_margin(self, X_pool, n_samples=10):
        """Select samples with smallest margin between top-2 classes"""
        proba = self.clf.predict_proba(X_pool)
        sorted_proba = np.sort(proba, axis=1)
        margin = sorted_proba[:, -1] - sorted_proba[:, -2]
        return np.argsort(margin)[:n_samples]

# Usage
learner = ActiveLearner(LogisticRegression())
# Initial training on small labeled set
# learner.clf.fit(X_initial, y_initial)
# indices = learner.query_uncertainty(X_pool, n_samples=10)
# Get labels for X_pool[indices] from oracle
```

**Interview Tips:**
- Reduces labeling cost significantly
- Uncertainty sampling most common
- Combine uncertainty with diversity for best results

---

## Question 50
- [ ] Done

**How do you implement query strategies for active learning in classification?**

### Answer

**Common Query Strategies:**

| Strategy | Formula/Approach |
|----------|-----------------|
| Least confidence | $1 - \max_y P(y|x)$ |
| Margin sampling | $P(y_1|x) - P(y_2|x)$ (top 2) |
| Entropy | $-\sum P(y|x)\log P(y|x)$ |
| Query-by-Committee | Disagreement among models |

**Python Code Example:**
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class QueryStrategies:
    def __init__(self, model):
        self.model = model
    
    def least_confidence(self, X, n=10):
        proba = self.model.predict_proba(X)
        confidence = proba.max(axis=1)
        return np.argsort(confidence)[:n]
    
    def margin_sampling(self, X, n=10):
        proba = self.model.predict_proba(X)
        sorted_proba = np.sort(proba, axis=1)
        margin = sorted_proba[:, -1] - sorted_proba[:, -2]
        return np.argsort(margin)[:n]
    
    def entropy_sampling(self, X, n=10):
        proba = self.model.predict_proba(X)
        entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
        return np.argsort(entropy)[-n:]  # Highest entropy
    
    def query_by_committee(self, X, n=10, committee_size=5):
        """Ensemble disagreement"""
        # Train multiple models with different seeds
        predictions = []
        for i in range(committee_size):
            # Bootstrap sample
            idx = np.random.choice(len(self.X_train), len(self.X_train))
            clf = RandomForestClassifier(n_estimators=10, random_state=i)
            clf.fit(self.X_train[idx], self.y_train[idx])
            predictions.append(clf.predict(X))
        
        predictions = np.array(predictions)
        disagreement = np.array([len(set(predictions[:, i])) 
                                 for i in range(len(X))])
        return np.argsort(disagreement)[-n:]

# Usage
qs = QueryStrategies(RandomForestClassifier().fit(X_train, y_train))
uncertain_samples = qs.entropy_sampling(X_pool, n=10)
```

**Interview Tips:**
- Entropy most theoretically grounded
- Margin sampling efficient for binary
- QBC provides diversity naturally

---

## Question 51
- [ ] Done

**What are transfer learning approaches for classification tasks?**

### Answer

**Definition:**
Transfer learning applies knowledge from a source domain/task to improve learning in a target domain/task. Especially useful when target has limited data.

**Approaches:**
| Approach | Description |
|----------|-------------|
| Feature extraction | Use pre-trained as feature extractor |
| Fine-tuning | Retrain some/all layers |
| Domain adaptation | Align source and target distributions |
| Multi-task learning | Learn shared representations |

**When to Use:**
- Target data limited
- Source and target related
- Pre-trained model available

**Transfer Strategies:**
1. **Freeze + new head**: Keep pre-trained weights, train new classifier
2. **Gradual unfreezing**: Unfreeze layers progressively
3. **Discriminative fine-tuning**: Different learning rates per layer

**Python Code Example:**
```python
# Conceptual PyTorch example
"""
import torchvision.models as models
import torch.nn as nn

# Load pre-trained ResNet
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Train only final layer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
"""

# Sklearn: use features from one model for another
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# "Transfer" feature extraction
transfer_pipeline = Pipeline([
    ('feature_extractor', PCA(n_components=50)),  # Pre-computed on source
    ('classifier', LogisticRegression())
])
```

**Interview Tips:**
- Works best for images, text with pre-trained models
- More related domains = better transfer
- Fine-tune with small learning rate

---

## Question 52
- [ ] Done

**How do you implement domain adaptation for classification across different domains?**

### Answer

**Definition:**
Domain adaptation addresses distribution shift between source (training) and target (test) domains. Goal: train on source, perform well on target.

**Types:**
| Type | Target Labels |
|------|--------------|
| Supervised | Available |
| Semi-supervised | Few available |
| Unsupervised | None |

**Techniques:**
| Method | Approach |
|--------|----------|
| Instance reweighting | Weight source samples by similarity to target |
| Feature alignment | Make source and target features similar |
| Domain-adversarial | Learn domain-invariant features |
| Self-training | Use target pseudo-labels |

**Python Code Example:**
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

class InstanceReweighting:
    """Weight source samples by similarity to target"""
    def __init__(self, base_classifier):
        self.clf = base_classifier
    
    def compute_weights(self, X_source, X_target):
        # Train domain classifier
        X_combined = np.vstack([X_source, X_target])
        y_domain = np.array([0]*len(X_source) + [1]*len(X_target))
        
        domain_clf = LogisticRegression().fit(X_combined, y_domain)
        
        # Weight = P(target|x) / P(source|x)
        source_proba = domain_clf.predict_proba(X_source)
        weights = source_proba[:, 1] / (source_proba[:, 0] + 1e-10)
        return np.clip(weights, 0.1, 10)  # Clip extreme weights
    
    def fit(self, X_source, y_source, X_target):
        weights = self.compute_weights(X_source, X_target)
        self.clf.fit(X_source, y_source, sample_weight=weights)
        return self

# Usage
adapter = InstanceReweighting(LogisticRegression())
adapter.fit(X_source, y_source, X_target)
predictions = adapter.clf.predict(X_target)
```

**Interview Tips:**
- Domain shift common in real applications
- Unsupervised adaptation most challenging
- Deep learning uses gradient reversal for domain-adversarial

---

## Question 53
- [ ] Done

**What is few-shot learning and its applications in classification?**

### Answer

**Definition:**
Few-shot learning trains models that can classify new classes with only a few (1-5) labeled examples per class. Mimics human ability to learn from few examples.

**Terminology:**
- **N-way K-shot**: N classes, K examples each
- **Support set**: Few labeled examples
- **Query set**: Test samples

**Approaches:**
| Method | Description |
|--------|-------------|
| Metric learning | Learn similarity function |
| Meta-learning | Learn to learn |
| Data augmentation | Generate more from few |
| Transfer learning | Leverage pre-trained features |

**Algorithms:**
- Siamese Networks (similarity)
- Prototypical Networks (class prototypes)
- MAML (model-agnostic meta-learning)
- Matching Networks

**Python Code Example:**
```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class PrototypicalClassifier:
    """Simple prototypical networks implementation"""
    
    def fit(self, X_support, y_support):
        """Compute class prototypes (centroids)"""
        self.classes = np.unique(y_support)
        self.prototypes = {}
        
        for c in self.classes:
            class_samples = X_support[y_support == c]
            self.prototypes[c] = class_samples.mean(axis=0)
    
    def predict(self, X_query):
        """Classify by nearest prototype"""
        predictions = []
        for x in X_query:
            distances = {c: np.linalg.norm(x - proto) 
                        for c, proto in self.prototypes.items()}
            predictions.append(min(distances, key=distances.get))
        return np.array(predictions)

# 5-way 3-shot example
# For each of 5 classes, provide 3 examples
support_X = np.random.randn(15, 10)  # 5 classes × 3 examples
support_y = np.repeat([0, 1, 2, 3, 4], 3)

proto_clf = PrototypicalClassifier()
proto_clf.fit(support_X, support_y)
```

**Applications:**
- Image classification with rare classes
- Drug discovery (few known compounds)
- Personalization (limited user data)

---

## Question 54
- [ ] Done

**How do you implement meta-learning algorithms for classification?**

### Answer

**Definition:**
Meta-learning ("learning to learn") trains models that can quickly adapt to new tasks with few examples. The model learns a learning algorithm itself.

**Core Concepts:**
- **Task distribution**: Many related tasks
- **Inner loop**: Adapt to specific task
- **Outer loop**: Learn across tasks
- **Few-shot evaluation**: Test on new tasks

**Popular Algorithms:**
| Algorithm | Approach |
|-----------|----------|
| MAML | Learn good initialization |
| Reptile | Simplified MAML |
| ProtoNet | Learn metric space |
| MetaSGD | Learn learning rate |

**MAML Concept:**
```
For each task:
    1. Copy model parameters
    2. Take gradient steps on task
    3. Evaluate adapted model
    4. Backprop through adaptation
```

**Python Code Example:**
```python
import numpy as np

class SimpleMAML:
    """Conceptual MAML implementation"""
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
    
    def inner_loop(self, X_support, y_support, n_steps=5):
        """Adapt to task"""
        # Clone model parameters
        adapted_params = self.model.get_params().copy()
        
        for _ in range(n_steps):
            # Compute gradients on support set
            grads = self.compute_gradients(X_support, y_support, adapted_params)
            # Update adapted parameters
            adapted_params = adapted_params - self.inner_lr * grads
        
        return adapted_params
    
    def meta_train(self, tasks, n_epochs=100):
        """Train across tasks"""
        for epoch in range(n_epochs):
            meta_grads = []
            
            for X_support, y_support, X_query, y_query in tasks:
                # Inner loop: adapt to task
                adapted_params = self.inner_loop(X_support, y_support)
                
                # Compute loss on query with adapted params
                # Backprop through inner loop
                task_grads = self.compute_meta_gradients(
                    X_query, y_query, adapted_params
                )
                meta_grads.append(task_grads)
            
            # Outer loop: update base model
            avg_grads = np.mean(meta_grads, axis=0)
            self.model.update_params(-self.outer_lr * avg_grads)
```

**Interview Tips:**
- MAML is model-agnostic (works with any differentiable model)
- Requires second-order gradients (expensive)
- Reptile is first-order approximation

---

## Question 55
- [ ] Done

**What are prototypical networks and their use in few-shot classification?**

### Answer

**Definition:**
Prototypical Networks learn an embedding space where classification is performed by computing distances to class prototypes (mean embeddings of support examples).

**Core Idea:**
1. Embed all points into metric space
2. Compute prototype (centroid) for each class
3. Classify query by nearest prototype
4. Train end-to-end with episodic training

**Mathematical Formulation:**
Prototype: $c_k = \frac{1}{|S_k|}\sum_{(x_i, y_i) \in S_k} f_\phi(x_i)$

Classification: $p(y=k|x) = \frac{\exp(-d(f_\phi(x), c_k))}{\sum_j \exp(-d(f_\phi(x), c_j))}$

**Python Code Example:**
```python
import numpy as np
from sklearn.neighbors import NearestCentroid

class PrototypicalNetwork:
    def __init__(self, embedding_fn=None):
        # embedding_fn: neural network encoder
        # For simplicity, use identity (raw features)
        self.embedding_fn = embedding_fn or (lambda x: x)
        self.prototypes = None
        self.classes = None
    
    def compute_prototypes(self, X_support, y_support):
        """Compute class centroids"""
        embeddings = self.embedding_fn(X_support)
        self.classes = np.unique(y_support)
        self.prototypes = {}
        
        for c in self.classes:
            mask = y_support == c
            self.prototypes[c] = embeddings[mask].mean(axis=0)
    
    def predict(self, X_query):
        """Classify by nearest prototype"""
        embeddings = self.embedding_fn(X_query)
        predictions = []
        
        for emb in embeddings:
            distances = {c: np.linalg.norm(emb - proto) 
                        for c, proto in self.prototypes.items()}
            predictions.append(min(distances, key=distances.get))
        
        return np.array(predictions)
    
    def predict_proba(self, X_query):
        """Softmax over negative distances"""
        embeddings = self.embedding_fn(X_query)
        all_proba = []
        
        for emb in embeddings:
            neg_distances = [-np.linalg.norm(emb - self.prototypes[c]) 
                           for c in self.classes]
            exp_dist = np.exp(neg_distances)
            proba = exp_dist / exp_dist.sum()
            all_proba.append(proba)
        
        return np.array(all_proba)

# Usage: 5-way 5-shot
proto_net = PrototypicalNetwork()
proto_net.compute_prototypes(X_support, y_support)
preds = proto_net.predict(X_query)
```

**Interview Tips:**
- Simple, effective, and fast
- Distance function matters (Euclidean common)
- Training uses episodic format (many few-shot tasks)

---

## Question 56
- [ ] Done

**How do you handle classification with noisy labels?**

### Answer

**Definition:**
Noisy labels are incorrect labels in training data due to human error, crowdsourcing disagreement, or automatic labeling mistakes. Methods aim to learn despite label noise.

**Types of Noise:**
| Type | Description |
|------|-------------|
| Uniform/Random | Each label equally likely to flip |
| Class-dependent | Confusion between similar classes |
| Instance-dependent | Noise depends on sample |

**Strategies:**
| Strategy | Approach |
|----------|----------|
| Robust loss | Loss functions less sensitive to outliers |
| Sample selection | Identify and remove noisy samples |
| Label correction | Fix suspected wrong labels |
| Regularization | Prevent memorizing noise |

**Python Code Example:**
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

class NoiseRobustClassifier:
    def __init__(self, base_classifier, noise_rate=0.1):
        self.clf = base_classifier
        self.noise_rate = noise_rate
    
    def clean_by_loss(self, X, y):
        """Remove high-loss samples (likely noisy)"""
        # Train initial model
        self.clf.fit(X, y)
        
        # Compute per-sample loss (cross-entropy)
        proba = self.clf.predict_proba(X)
        losses = -np.log(proba[np.arange(len(y)), y] + 1e-10)
        
        # Keep samples with low loss
        threshold = np.percentile(losses, (1 - self.noise_rate) * 100)
        clean_mask = losses < threshold
        
        return X[clean_mask], y[clean_mask]
    
    def fit(self, X, y):
        # Clean data
        X_clean, y_clean = self.clean_by_loss(X, y)
        print(f"Removed {len(X) - len(X_clean)} noisy samples")
        
        # Retrain on clean data
        self.clf.fit(X_clean, y_clean)
        return self

# Robust losses (conceptual)
def symmetric_cross_entropy(y_true, y_pred, alpha=0.1, beta=1.0):
    """More robust to label noise"""
    ce = -y_true * np.log(y_pred + 1e-10)
    rce = -y_pred * np.log(y_true + 1e-10)
    return alpha * ce + beta * rce
```

**Interview Tips:**
- Start simple: regularization often helps
- Co-teaching: two networks teach each other
- MixUp training improves robustness

---

## Question 57
- [ ] Done

**What are robust classification methods for label noise?**

### Answer

**Definition:**
Robust methods are designed to maintain accuracy despite incorrect labels in training data. They either use noise-tolerant architectures, losses, or explicitly model the noise.

**Methods:**

| Method | Type | Approach |
|--------|------|----------|
| Symmetric CE | Loss | Robust to noise |
| Focal loss | Loss | Down-weight easy samples |
| Bootstrap loss | Loss | Use own predictions |
| Co-teaching | Training | Two networks, small-loss selection |
| MixUp | Augmentation | Convex combination of samples |
| Noise layer | Model | Explicit noise transition matrix |

**Mathematical Formulation:**
Noise Transition Matrix: $\tilde{y} = T \cdot y$

Where $T_{ij} = P(\tilde{y}=j | y=i)$

**Python Code Example:**
```python
import numpy as np

def symmetric_cross_entropy(y_true, y_pred, alpha=1.0, beta=1.0):
    """Symmetric Cross Entropy - robust to noise"""
    ce = -np.sum(y_true * np.log(y_pred + 1e-10))
    rce = -np.sum(y_pred * np.log(y_true + 1e-10))
    return alpha * ce + beta * rce

def bootstrap_loss(y_true, y_pred, beta=0.95):
    """Use mix of true label and prediction"""
    target = beta * y_true + (1 - beta) * y_pred
    return -np.sum(target * np.log(y_pred + 1e-10))

class CoTeaching:
    """Two networks select small-loss samples for each other"""
    def __init__(self, model1, model2, noise_rate=0.2):
        self.model1 = model1
        self.model2 = model2
        self.noise_rate = noise_rate
    
    def select_small_loss(self, model, X, y, keep_ratio):
        """Select samples with smallest loss"""
        proba = model.predict_proba(X)
        losses = -np.log(proba[np.arange(len(y)), y] + 1e-10)
        n_keep = int(len(X) * keep_ratio)
        indices = np.argsort(losses)[:n_keep]
        return indices
    
    def fit_epoch(self, X, y, epoch, max_epochs):
        # Gradually increase kept samples
        keep_ratio = 1 - min(self.noise_rate, epoch / max_epochs * self.noise_rate)
        
        # Model 1 selects for Model 2
        idx1 = self.select_small_loss(self.model1, X, y, keep_ratio)
        self.model2.fit(X[idx1], y[idx1])
        
        # Model 2 selects for Model 1
        idx2 = self.select_small_loss(self.model2, X, y, keep_ratio)
        self.model1.fit(X[idx2], y[idx2])
```

**Interview Tips:**
- Co-teaching is simple and effective
- MixUp provides regularization and robustness
- Estimate noise rate from data if unknown

---

## Question 58
- [ ] Done

**How do you implement classification for streaming and real-time data?**

### Answer

**Definition:**
Streaming classification processes data continuously as it arrives, making predictions and updating models in real-time with memory and time constraints.

**Requirements:**
- Process one sample/batch at a time
- Constant memory (can't store all data)
- Fast prediction and update
- Handle concept drift

**Algorithms:**
| Algorithm | Update Method |
|-----------|--------------|
| SGD Classifier | partial_fit |
| Naive Bayes | partial_fit |
| Hoeffding Tree | Incremental splits |
| Online Random Forest | Add/remove trees |

**Python Code Example:**
```python
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from collections import deque

class StreamingClassifier:
    def __init__(self, window_size=1000):
        self.model = SGDClassifier(loss='log_loss')
        self.scaler = StandardScaler()
        self.window_size = window_size
        self.buffer_X = deque(maxlen=window_size)
        self.buffer_y = deque(maxlen=window_size)
        self.initialized = False
        self.classes = None
    
    def process(self, X, y, fit=True):
        """Process streaming data"""
        # Add to buffer
        self.buffer_X.append(X)
        self.buffer_y.append(y)
        
        if not self.initialized and len(self.buffer_X) >= 100:
            # Initial training
            X_init = np.array(list(self.buffer_X))
            y_init = np.array(list(self.buffer_y))
            self.classes = np.unique(y_init)
            self.scaler.fit(X_init)
            X_scaled = self.scaler.transform(X_init)
            self.model.fit(X_scaled, y_init)
            self.initialized = True
        elif self.initialized and fit:
            # Incremental update
            X_scaled = self.scaler.transform(X.reshape(1, -1))
            self.model.partial_fit(X_scaled, [y], classes=self.classes)
    
    def predict(self, X):
        if not self.initialized:
            return None
        X_scaled = self.scaler.transform(X.reshape(1, -1))
        return self.model.predict(X_scaled)[0]

# Usage
stream_clf = StreamingClassifier()
for x, y in data_stream:
    pred = stream_clf.predict(x)
    stream_clf.process(x, y)
```

**Interview Tips:**
- Must specify classes in first partial_fit
- Consider drift detection
- Trade-off: update frequency vs stability

---

## Question 59
- [ ] Done

**What are the computational complexity considerations for large-scale classification?**

### Answer

**Definition:**
Large-scale classification must consider time and space complexity of training and prediction for millions of samples and features.

**Complexity Analysis:**

| Algorithm | Training | Prediction | Space |
|-----------|----------|------------|-------|
| Logistic Regression | O(n·d) per iter | O(d) | O(d) |
| SVM (kernel) | O(n²) to O(n³) | O(sv·d) | O(n²) |
| Decision Tree | O(n·d·log n) | O(depth) | O(nodes) |
| Random Forest | O(T·n·d·log n) | O(T·depth) | O(T·nodes) |
| k-NN | O(1) | O(n·d) | O(n·d) |

**Scaling Strategies:**
| Strategy | Approach |
|----------|----------|
| Sampling | Train on subset |
| Dimensionality reduction | PCA, feature selection |
| Approximate methods | LSH, approximate k-NN |
| Distributed | MapReduce, Spark |
| Incremental | Online learning |

**Python Code Example:**
```python
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline

# For large data: use SGD instead of closed-form
sgd = SGDClassifier(loss='log_loss', max_iter=1000)
# Much faster than LogisticRegression for large n

# Approximate kernel SVM for large data
rbf_approx = RBFSampler(n_components=100, random_state=42)
approx_svm = Pipeline([
    ('feature_map', rbf_approx),
    ('classifier', SGDClassifier(loss='hinge'))
])

# For very large data: use partial_fit
large_data_clf = SGDClassifier()
for X_batch, y_batch in data_generator:
    large_data_clf.partial_fit(X_batch, y_batch, classes=[0, 1])
```

**Interview Tips:**
- Kernel SVM doesn't scale (O(n²) memory)
- Use linear models with kernel approximation
- SGDClassifier for large datasets

---

## Question 60
- [ ] Done

**How do you implement distributed classification for very large datasets?**

### Answer

**Definition:**
Distributed classification trains models across multiple machines/nodes, enabling processing of datasets too large for a single machine. Data or model is partitioned across workers.

**Approaches:**

| Approach | Description |
|----------|-------------|
| Data parallelism | Split data across workers, each trains on subset |
| Model parallelism | Split model across workers |
| Parameter server | Central server aggregates gradients |
| AllReduce | Peer-to-peer gradient aggregation |

**Frameworks:**
- Apache Spark MLlib
- Dask-ML
- Horovod (deep learning)
- Ray

**Python Code Example:**
```python
# PySpark example
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DistributedClassification").getOrCreate()

# Load distributed data
df = spark.read.parquet("hdfs://path/to/data")

# Train distributed model
lr = LogisticRegression(maxIter=100, regParam=0.01)
model = lr.fit(df)

# Dask example
from dask_ml.linear_model import LogisticRegression as DaskLR
from dask.distributed import Client
import dask.array as da

client = Client()  # Start distributed scheduler
X_dask = da.from_array(X, chunks=(1000, -1))
y_dask = da.from_array(y, chunks=1000)

model = DaskLR()
model.fit(X_dask, y_dask)
```

**Interview Tips:**
- Data parallelism most common
- Communication overhead is bottleneck
- Gradient compression reduces bandwidth

---

## Question 61
- [ ] Done

**What is federated learning and how does it apply to classification?**

### Answer

**Definition:**
Federated learning trains models across decentralized devices/servers holding local data, without exchanging raw data. Only model updates (gradients) are shared, preserving privacy.

**Core Concepts:**
- Data stays on local devices
- Only model updates shared
- Central server aggregates updates
- Privacy-preserving by design

**Algorithm (FedAvg):**
1. Server sends global model to clients
2. Each client trains on local data
3. Clients send model updates to server
4. Server averages updates
5. Repeat

**Challenges:**
| Challenge | Description |
|-----------|-------------|
| Non-IID data | Local data distributions differ |
| Communication | Limited bandwidth |
| Heterogeneity | Different device capabilities |
| Privacy | Gradients can leak info |

**Python Code Example:**
```python
import numpy as np

class FederatedServer:
    def __init__(self, model_init):
        self.global_weights = model_init.copy()
    
    def aggregate(self, client_weights, client_sizes):
        """FedAvg: weighted average by data size"""
        total_size = sum(client_sizes)
        new_weights = np.zeros_like(self.global_weights)
        
        for weights, size in zip(client_weights, client_sizes):
            new_weights += weights * (size / total_size)
        
        self.global_weights = new_weights
        return self.global_weights

class FederatedClient:
    def __init__(self, local_data, local_labels):
        self.X = local_data
        self.y = local_labels
    
    def local_train(self, global_weights, epochs=5):
        # Initialize with global weights
        weights = global_weights.copy()
        
        # Train locally (simplified)
        for _ in range(epochs):
            # Gradient descent on local data
            pass
        
        return weights, len(self.y)

# Usage
server = FederatedServer(initial_weights)
clients = [FederatedClient(X_i, y_i) for X_i, y_i in local_datasets]

for round in range(num_rounds):
    client_updates = []
    client_sizes = []
    
    for client in clients:
        weights, size = client.local_train(server.global_weights)
        client_updates.append(weights)
        client_sizes.append(size)
    
    server.aggregate(client_updates, client_sizes)
```

**Interview Tips:**
- Google uses for keyboard prediction
- Differential privacy adds noise for extra protection
- Non-IID data is main challenge

---

## Question 62
- [ ] Done

**How do you implement privacy-preserving classification techniques?**

### Answer

**Definition:**
Privacy-preserving classification protects sensitive training data while still enabling model training and prediction. Techniques include differential privacy, secure computation, and homomorphic encryption.

**Techniques:**

| Technique | Approach |
|-----------|----------|
| Differential Privacy | Add noise to gradients/outputs |
| Secure Multi-Party | Compute on encrypted data jointly |
| Homomorphic Encryption | Compute on encrypted data |
| Federated Learning | Keep data decentralized |

**Differential Privacy:**
- Add calibrated noise to query results
- Guarantee: single record doesn't change output much
- Privacy budget (ε): smaller = more private

**Python Code Example:**
```python
import numpy as np

class DifferentiallyPrivateSGD:
    def __init__(self, epsilon=1.0, delta=1e-5, clip_norm=1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
    
    def clip_gradients(self, gradients):
        """Clip gradients to bound sensitivity"""
        norm = np.linalg.norm(gradients)
        if norm > self.clip_norm:
            gradients = gradients * (self.clip_norm / norm)
        return gradients
    
    def add_noise(self, gradients, n_samples):
        """Add Gaussian noise for DP"""
        # Noise scale based on privacy budget
        sigma = self.clip_norm * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, sigma, gradients.shape)
        return gradients + noise / n_samples
    
    def private_gradient_step(self, X, y, weights, lr=0.01):
        # Compute per-sample gradients
        gradients = self.compute_gradients(X, y, weights)
        
        # Clip each gradient
        clipped = [self.clip_gradients(g) for g in gradients]
        
        # Average and add noise
        avg_grad = np.mean(clipped, axis=0)
        private_grad = self.add_noise(avg_grad, len(X))
        
        # Update
        return weights - lr * private_grad

# Using Opacus (PyTorch DP library)
# from opacus import PrivacyEngine
# privacy_engine = PrivacyEngine()
# model, optimizer, dataloader = privacy_engine.make_private(
#     module=model, optimizer=optimizer, data_loader=dataloader,
#     noise_multiplier=1.0, max_grad_norm=1.0
# )
```

**Interview Tips:**
- ε < 1 is strong privacy
- Privacy-utility tradeoff
- Composition: privacy degrades over queries

---

## Question 63
- [ ] Done

**What is differential privacy and its application in classification?**

### Answer

**Definition:**
Differential privacy (DP) provides mathematical guarantee that an algorithm's output doesn't reveal whether any individual's data was in the training set. Adding/removing one person changes output probability by at most factor e^ε.

**Mathematical Definition:**
Algorithm M is (ε, δ)-DP if for all datasets D, D' differing in one element:
$$P[M(D) \in S] \leq e^\epsilon \cdot P[M(D') \in S] + \delta$$

**Core Concepts:**
- ε (epsilon): privacy budget, smaller = more private
- δ (delta): probability of privacy breach
- Sensitivity: max change from one record
- Noise calibrated to sensitivity

**Mechanisms:**
| Mechanism | Noise Type | Use Case |
|-----------|-----------|----------|
| Laplace | Laplace | Counting queries |
| Gaussian | Gaussian | Gradient descent |
| Exponential | Selection | Choosing best item |

**Python Code Example:**
```python
import numpy as np

def laplace_mechanism(true_answer, sensitivity, epsilon):
    """Add Laplace noise for ε-DP"""
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return true_answer + noise

def gaussian_mechanism(true_answer, sensitivity, epsilon, delta):
    """Add Gaussian noise for (ε,δ)-DP"""
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma)
    return true_answer + noise

# DP classification example
class DPLogisticRegression:
    def __init__(self, epsilon=1.0, epochs=10):
        self.epsilon = epsilon
        self.epochs = epochs
        self.weights = None
    
    def fit(self, X, y):
        n, d = X.shape
        self.weights = np.zeros(d)
        
        # Privacy budget per epoch
        eps_per_epoch = self.epsilon / self.epochs
        
        for _ in range(self.epochs):
            # Compute gradient (sensitivity bounded by clipping)
            gradient = self.compute_gradient(X, y)
            
            # Add noise
            noisy_gradient = gaussian_mechanism(
                gradient, 
                sensitivity=1.0,  # After clipping
                epsilon=eps_per_epoch,
                delta=1e-5
            )
            
            self.weights -= 0.01 * noisy_gradient
```

**Interview Tips:**
- ε = 1 is common, ε < 0.1 very private
- Composition: k queries use k*ε budget
- Advanced: moments accountant for tighter bounds

---

## Question 64
- [ ] Done

**How do you implement secure multi-party computation for classification?**

### Answer

**Definition:**
Secure Multi-Party Computation (MPC) allows multiple parties to jointly compute a function over their inputs while keeping inputs private. Each party learns only the output, not others' data.

**Core Concepts:**
- Secret sharing: split data into shares
- Garbled circuits: encrypted computation
- Oblivious transfer: receive without revealing choice
- No single party sees complete data

**Secret Sharing (Additive):**
Split value x into n shares: $x = s_1 + s_2 + ... + s_n$

Each party holds one share; need all to reconstruct.

**Python Code Example:**
```python
import numpy as np

class SecretSharing:
    def __init__(self, n_parties):
        self.n_parties = n_parties
    
    def share(self, value):
        """Split value into additive shares"""
        shares = np.random.randint(-1000, 1000, self.n_parties - 1)
        last_share = value - shares.sum()
        return list(shares) + [last_share]
    
    def reconstruct(self, shares):
        """Reconstruct value from shares"""
        return sum(shares)

class SecureDotProduct:
    """Two-party secure dot product"""
    def __init__(self):
        self.ss = SecretSharing(2)
    
    def compute(self, x_alice, x_bob):
        """Compute dot product without revealing vectors"""
        # In practice, use crypto libraries
        # Simplified illustration:
        
        # Each party masks their input
        r_a = np.random.randn(len(x_alice))
        r_b = np.random.randn(len(x_bob))
        
        # Exchange masked values
        masked_a = x_alice + r_a
        masked_b = x_bob + r_b
        
        # Compute with correction terms
        # (Real MPC uses oblivious transfer, garbled circuits)
        return np.dot(x_alice, x_bob)

# For production, use:
# - PySyft (federated + MPC)
# - MP-SPDZ
# - CrypTen (PyTorch)
```

**Applications:**
- Healthcare: multiple hospitals train on combined data
- Finance: fraud detection across banks
- Privacy-preserving ML as a service

**Interview Tips:**
- Significant computational overhead
- Communication rounds are bottleneck
- Often combined with differential privacy

---

## Question 65
- [ ] Done

**How do you build classifiers that are robust to adversarial attacks?**

### Answer

**Definition:**
Adversarial robustness ensures classifiers maintain accuracy when inputs are deliberately perturbed. Robust classifiers resist adversarial examples designed to cause misclassification.

**Attack Types:**
| Attack | Description |
|--------|-------------|
| FGSM | Fast gradient sign method |
| PGD | Projected gradient descent |
| C&W | Carlini-Wagner optimization |
| Black-box | Query-based, no gradients |

**Defense Strategies:**

| Defense | Approach |
|---------|----------|
| Adversarial training | Train on adversarial examples |
| Certified defenses | Provable guarantees |
| Input preprocessing | Denoise, compress inputs |
| Ensemble | Multiple models vote |

**Python Code Example:**
```python
import numpy as np

class AdversarialTrainer:
    def __init__(self, model, epsilon=0.3):
        self.model = model
        self.epsilon = epsilon
    
    def fgsm_attack(self, X, y):
        """Generate adversarial examples using FGSM"""
        # Compute gradient of loss w.r.t. input
        grad = self.compute_input_gradient(X, y)
        
        # Perturb in direction of gradient sign
        perturbation = self.epsilon * np.sign(grad)
        X_adv = X + perturbation
        
        # Clip to valid range
        X_adv = np.clip(X_adv, 0, 1)
        return X_adv
    
    def adversarial_train(self, X, y, epochs=10):
        for epoch in range(epochs):
            # Generate adversarial examples
            X_adv = self.fgsm_attack(X, y)
            
            # Train on mix of clean and adversarial
            X_combined = np.vstack([X, X_adv])
            y_combined = np.hstack([y, y])
            
            self.model.fit(X_combined, y_combined)
            
            # Evaluate
            clean_acc = self.model.score(X, y)
            adv_acc = self.model.score(X_adv, y)
            print(f"Epoch {epoch}: Clean={clean_acc:.3f}, Adv={adv_acc:.3f}")

# Certified defense: randomized smoothing
class RandomizedSmoothing:
    def __init__(self, base_classifier, sigma=0.25, n_samples=100):
        self.clf = base_classifier
        self.sigma = sigma
        self.n_samples = n_samples
    
    def predict(self, x):
        """Majority vote over noisy samples"""
        votes = []
        for _ in range(self.n_samples):
            noisy_x = x + np.random.normal(0, self.sigma, x.shape)
            votes.append(self.clf.predict(noisy_x.reshape(1, -1))[0])
        
        from collections import Counter
        return Counter(votes).most_common(1)[0][0]
```

**Interview Tips:**
- Adversarial training most effective but expensive
- Robustness vs accuracy tradeoff exists
- Certified defenses provide guarantees but limited

---

## Question 66
- [ ] Done

**What are certified defenses for classification models?**

### Answer

**Definition:**
Certified defenses provide provable guarantees that a classifier's prediction won't change within a certain perturbation radius. Unlike empirical defenses, they offer mathematical guarantees.

**Types:**

| Method | Guarantee |
|--------|-----------|
| Randomized Smoothing | Certified radius for ℓ2 norm |
| Interval Bound Propagation | ℓ∞ robustness |
| Lipschitz Networks | Bounded sensitivity |
| Convex Relaxations | Verification bounds |

**Randomized Smoothing:**
- Add Gaussian noise to inputs
- Majority vote over samples
- If margin large enough, prediction certified

**Mathematical Guarantee:**
If $P_A = P[\text{predict class A}] > P_B$ for all other classes B, then prediction is certified robust within radius:
$$r = \frac{\sigma}{2}(\Phi^{-1}(P_A) - \Phi^{-1}(P_B))$$

**Python Code Example:**
```python
import numpy as np
from scipy.stats import norm

class CertifiedClassifier:
    def __init__(self, base_model, sigma=0.5, n_samples=1000, alpha=0.001):
        self.model = base_model
        self.sigma = sigma
        self.n_samples = n_samples
        self.alpha = alpha  # Confidence level
    
    def certify(self, x):
        """Get prediction and certified radius"""
        # Sample predictions
        counts = {}
        for _ in range(self.n_samples):
            noisy_x = x + np.random.normal(0, self.sigma, x.shape)
            pred = self.model.predict(noisy_x.reshape(1, -1))[0]
            counts[pred] = counts.get(pred, 0) + 1
        
        # Get top two classes
        sorted_classes = sorted(counts.items(), key=lambda x: -x[1])
        top_class, top_count = sorted_classes[0]
        
        # Lower bound on probability (Clopper-Pearson)
        p_lower = self._lower_confidence_bound(top_count, self.n_samples)
        
        if p_lower > 0.5:
            # Certified radius
            radius = self.sigma * norm.ppf(p_lower)
            return top_class, radius
        else:
            return top_class, 0  # Cannot certify
    
    def _lower_confidence_bound(self, k, n):
        """Clopper-Pearson lower bound"""
        from scipy.stats import beta
        return beta.ppf(self.alpha, k, n - k + 1)

# Usage
cert_clf = CertifiedClassifier(base_model, sigma=0.5)
pred, radius = cert_clf.certify(x_test)
print(f"Prediction: {pred}, Certified radius: {radius:.3f}")
```

**Interview Tips:**
- Randomized smoothing works for any classifier
- Larger sigma = larger radius but lower accuracy
- Certified accuracy: % correct AND certified

---

## Question 67
- [ ] Done

**How do you detect and defend against adversarial examples?**

### Answer

**Definition:**
Adversarial detection identifies malicious inputs before classification, while defense methods either reject or correct adversarial examples.

**Detection Methods:**

| Method | Approach |
|--------|----------|
| Statistical | Check if input matches training distribution |
| Uncertainty | High uncertainty suggests adversarial |
| Input transformation | Inconsistent predictions after transform |
| Feature squeezing | Compare predictions at different precisions |

**Defense Methods:**
| Defense | Description |
|---------|-------------|
| Input denoising | Remove perturbations |
| Feature distillation | Train on soft labels |
| Rejection | Refuse to classify uncertain inputs |
| Ensemble voting | Multiple models must agree |

**Python Code Example:**
```python
import numpy as np
from sklearn.ensemble import IsolationForest

class AdversarialDetector:
    def __init__(self, base_model):
        self.model = base_model
        self.outlier_detector = IsolationForest(contamination=0.1)
    
    def fit_detector(self, X_train):
        """Fit outlier detector on clean training data"""
        self.outlier_detector.fit(X_train)
    
    def detect_statistical(self, X):
        """Detect based on distribution"""
        scores = self.outlier_detector.predict(X)
        return scores == -1  # -1 indicates outlier
    
    def detect_uncertainty(self, X, threshold=0.3):
        """Detect based on prediction uncertainty"""
        proba = self.model.predict_proba(X)
        max_proba = proba.max(axis=1)
        return max_proba < threshold
    
    def detect_feature_squeezing(self, X, depth=4):
        """Compare predictions at different bit depths"""
        # Original prediction
        pred_orig = self.model.predict(X)
        
        # Squeezed prediction (reduce precision)
        X_squeezed = np.round(X * (2**depth)) / (2**depth)
        pred_squeezed = self.model.predict(X_squeezed)
        
        # Different predictions suggest adversarial
        return pred_orig != pred_squeezed

class AdversarialDefender:
    def __init__(self, models):
        self.models = models  # Ensemble
    
    def ensemble_defense(self, X, agreement_threshold=0.7):
        """Require model agreement"""
        predictions = np.array([m.predict(X) for m in self.models])
        
        results = []
        for i in range(len(X)):
            votes = predictions[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            agreement = counts.max() / len(self.models)
            
            if agreement >= agreement_threshold:
                results.append(unique[counts.argmax()])
            else:
                results.append(-1)  # Reject
        
        return np.array(results)
```

**Interview Tips:**
- No single defense is foolproof
- Combine detection and defense
- Adaptive attacks can bypass detectors

---

## Question 68
- [ ] Done

**What is fairness in machine learning classification and why is it important?**

### Answer

**Definition:**
Fairness in ML ensures classification models don't discriminate against protected groups (race, gender, age). Fair models make equitable predictions across demographic groups.

**Why Important:**
- Legal compliance (GDPR, Equal Credit Opportunity Act)
- Ethical responsibility
- Trust and adoption
- Avoid perpetuating historical biases

**Fairness Definitions:**

| Definition | Requirement |
|------------|-------------|
| Demographic Parity | P(Ŷ=1) same across groups |
| Equal Opportunity | TPR same across groups |
| Equalized Odds | TPR and FPR same across groups |
| Individual Fairness | Similar individuals treated similarly |

**Tension:**
- Different fairness definitions can conflict
- Fairness vs accuracy tradeoff
- Cannot satisfy all fairness metrics simultaneously

**Python Code Example:**
```python
import numpy as np

def demographic_parity(y_pred, sensitive_attr):
    """Check if positive rate is equal across groups"""
    groups = np.unique(sensitive_attr)
    rates = {}
    
    for g in groups:
        mask = sensitive_attr == g
        rates[g] = y_pred[mask].mean()
    
    disparity = max(rates.values()) - min(rates.values())
    return rates, disparity

def equal_opportunity(y_true, y_pred, sensitive_attr):
    """Check if TPR is equal across groups"""
    groups = np.unique(sensitive_attr)
    tpr = {}
    
    for g in groups:
        mask = (sensitive_attr == g) & (y_true == 1)
        if mask.sum() > 0:
            tpr[g] = y_pred[mask].mean()
    
    disparity = max(tpr.values()) - min(tpr.values())
    return tpr, disparity

def equalized_odds(y_true, y_pred, sensitive_attr):
    """Check if TPR and FPR are equal across groups"""
    tpr, tpr_disp = equal_opportunity(y_true, y_pred, sensitive_attr)
    
    groups = np.unique(sensitive_attr)
    fpr = {}
    for g in groups:
        mask = (sensitive_attr == g) & (y_true == 0)
        if mask.sum() > 0:
            fpr[g] = y_pred[mask].mean()
    
    fpr_disp = max(fpr.values()) - min(fpr.values())
    return {'TPR': tpr, 'FPR': fpr}, max(tpr_disp, fpr_disp)

# Usage
rates, disparity = demographic_parity(y_pred, gender)
print(f"Positive rates: {rates}, Disparity: {disparity:.3f}")
```

**Interview Tips:**
- Know multiple fairness definitions
- Understand impossibility results
- Fairness is context-dependent

---

## Question 69
- [ ] Done

**How do you implement bias detection and mitigation in classification?**

### Answer

**Definition:**
Bias detection identifies unfair disparities in model predictions. Mitigation techniques reduce these disparities through preprocessing, in-processing, or post-processing methods.

**Bias Detection:**

| Metric | Measures |
|--------|----------|
| Statistical Parity Diff | Difference in positive rates |
| Disparate Impact Ratio | Ratio of positive rates |
| Equal Opportunity Diff | Difference in TPR |
| Average Odds Diff | Average of TPR and FPR diff |

**Mitigation Approaches:**

| Stage | Method |
|-------|--------|
| Pre-processing | Reweighting, sampling, removing bias |
| In-processing | Fair constraints in training |
| Post-processing | Adjust thresholds per group |

**Python Code Example:**
```python
import numpy as np

# Detection
def disparate_impact(y_pred, sensitive_attr, privileged=1):
    """Ratio of positive rates (should be > 0.8)"""
    priv_mask = sensitive_attr == privileged
    unpriv_mask = ~priv_mask
    
    rate_priv = y_pred[priv_mask].mean()
    rate_unpriv = y_pred[unpriv_mask].mean()
    
    return min(rate_priv, rate_unpriv) / max(rate_priv, rate_unpriv)

# Pre-processing: Reweighting
def compute_reweighting(y, sensitive_attr):
    """Compute sample weights for fairness"""
    weights = np.ones(len(y))
    
    for s in np.unique(sensitive_attr):
        for label in [0, 1]:
            mask = (sensitive_attr == s) & (y == label)
            # Weight inversely to group size
            expected = len(y) / 4  # 2 groups × 2 labels
            actual = mask.sum()
            weights[mask] = expected / actual
    
    return weights

# Post-processing: Threshold adjustment
class FairThresholdClassifier:
    def __init__(self, base_model):
        self.model = base_model
        self.thresholds = {}
    
    def fit_thresholds(self, X, y, sensitive_attr, target_rate):
        """Find group-specific thresholds for equal positive rate"""
        proba = self.model.predict_proba(X)[:, 1]
        
        for group in np.unique(sensitive_attr):
            mask = sensitive_attr == group
            group_proba = proba[mask]
            
            # Binary search for threshold achieving target rate
            self.thresholds[group] = np.percentile(
                group_proba, (1 - target_rate) * 100
            )
    
    def predict(self, X, sensitive_attr):
        proba = self.model.predict_proba(X)[:, 1]
        predictions = np.zeros(len(X))
        
        for group in np.unique(sensitive_attr):
            mask = sensitive_attr == group
            predictions[mask] = (proba[mask] >= self.thresholds[group]).astype(int)
        
        return predictions

# Using AIF360 (IBM Fairness library)
# from aif360.algorithms.preprocessing import Reweighing
# from aif360.algorithms.inprocessing import AdversarialDebiasing
```

**Interview Tips:**
- Pre-processing: changes data, model-agnostic
- In-processing: fair constraints, algorithm-specific
- Post-processing: changes outputs, simple but limited

---

## Question 70
- [ ] Done

**What are fairness-aware classification algorithms?**

### Answer

**Definition:**
Fairness-aware algorithms incorporate fairness constraints directly into the learning process, optimizing for both accuracy and fairness simultaneously.

**Approaches:**

| Method | Approach |
|--------|----------|
| Regularization | Add fairness penalty to loss |
| Constraints | Optimize accuracy subject to fairness |
| Adversarial | Learn features that don't predict sensitive attr |
| Meta-learning | Learn fair across different groups |

**Key Algorithms:**
- **Prejudice Remover**: Adds discrimination-aware regularization
- **Adversarial Debiasing**: Adversarial training to remove bias
- **Exponentiated Gradient**: Constrained optimization

**Python Code Example:**
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

class FairnessRegularizedLR:
    """Logistic regression with fairness regularization"""
    def __init__(self, lambda_fair=1.0):
        self.lambda_fair = lambda_fair
        self.weights = None
    
    def fit(self, X, y, sensitive_attr):
        # Add fairness term to loss
        # L = CrossEntropy + lambda * FairnessViolation
        
        # Simplified: use reweighting as approximation
        weights = self._compute_fairness_weights(y, sensitive_attr)
        
        self.model = LogisticRegression()
        self.model.fit(X, y, sample_weight=weights)
        
    def _compute_fairness_weights(self, y, s):
        """Weight samples to encourage fairness"""
        weights = np.ones(len(y))
        for group in np.unique(s):
            mask = s == group
            # Upweight minority outcomes in each group
            weights[mask] = len(y) / (2 * mask.sum())
        return weights

class AdversarialDebiasing:
    """Remove sensitive information from features"""
    def __init__(self, predictor, adversary):
        self.predictor = predictor  # Predicts y
        self.adversary = adversary  # Predicts sensitive attr
    
    def fit(self, X, y, sensitive_attr, epochs=100):
        for epoch in range(epochs):
            # Train predictor to predict y
            self.predictor.partial_fit(X, y)
            
            # Get hidden representation
            hidden = self.predictor.get_features(X)
            
            # Train adversary to predict sensitive attr from hidden
            self.adversary.partial_fit(hidden, sensitive_attr)
            
            # Update predictor to fool adversary (gradient reversal)
            adv_grad = self.adversary.get_gradient(hidden, sensitive_attr)
            self.predictor.update_adversarial(-adv_grad)

# Using fairlearn
# from fairlearn.reductions import ExponentiatedGradient, DemographicParity
# constraint = DemographicParity()
# model = ExponentiatedGradient(LogisticRegression(), constraints=constraint)
# model.fit(X, y, sensitive_features=sensitive_attr)
```

**Interview Tips:**
- fairlearn and AIF360 are standard libraries
- Trade-off between fairness and accuracy is tunable
- Choice of fairness metric depends on application

---

## Question 71
- [ ] Done

**How do you interpret and explain classification model predictions?**

### Answer

**Definition:**
Model interpretability explains why a classifier made a specific prediction, enabling trust, debugging, and regulatory compliance.

**Levels of Interpretability:**

| Level | Scope |
|-------|-------|
| Global | Overall model behavior |
| Local | Individual prediction |
| Feature | Importance of each feature |

**Methods:**

| Method | Type | Approach |
|--------|------|----------|
| LIME | Local | Fit interpretable model locally |
| SHAP | Both | Shapley values |
| Feature importance | Global | Permutation, tree-based |
| Partial Dependence | Global | Effect of single feature |

**Python Code Example:**
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 1. Feature Importance (tree-based)
importances = model.feature_importances_
print("Feature Importances:", dict(zip(feature_names, importances)))

# 2. Permutation Importance
perm_imp = permutation_importance(model, X_test, y_test, n_repeats=10)
print("Permutation Importance:", perm_imp.importances_mean)

# 3. LIME
# from lime.lime_tabular import LimeTabularExplainer
# explainer = LimeTabularExplainer(X_train, feature_names=feature_names)
# exp = explainer.explain_instance(X_test[0], model.predict_proba)
# exp.show_in_notebook()

# 4. SHAP
# import shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# 5. Partial Dependence Plot
from sklearn.inspection import PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(model, X_train, features=[0, 1])
```

**Interpretability by Model:**
| Model | Inherent Interpretability |
|-------|--------------------------|
| Linear/Logistic | High (coefficients) |
| Decision Tree | High (rules) |
| Random Forest | Medium (importance) |
| Neural Network | Low (black box) |

**Interview Tips:**
- SHAP provides theoretically grounded explanations
- LIME is model-agnostic and intuitive
- Interpretability often required for regulated industries

---

## Question 72
- [ ] Done

**What are LIME and SHAP, and how do they work for classification?**

### Answer

**LIME (Local Interpretable Model-agnostic Explanations):**
Explains individual predictions by fitting a simple interpretable model (linear) around the prediction neighborhood.

**LIME Steps:**
1. Perturb input around instance
2. Get predictions for perturbations
3. Weight by proximity to original
4. Fit linear model on perturbations
5. Use linear coefficients as explanation

**SHAP (SHapley Additive exPlanations):**
Uses game-theoretic Shapley values to attribute prediction to each feature fairly.

**SHAP Property:**
$$f(x) = \phi_0 + \sum_{i=1}^{n} \phi_i$$

Where $\phi_i$ is feature i's contribution.

**Comparison:**
| Aspect | LIME | SHAP |
|--------|------|------|
| Theory | Heuristic | Game theory |
| Consistency | No | Yes |
| Speed | Fast | Slower |
| Scope | Local | Local + Global |

**Python Code Example:**
```python
# LIME Example
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

# Create explainer
explainer = LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=['No', 'Yes'],
    mode='classification'
)

# Explain single prediction
instance = X_test[0]
exp = explainer.explain_instance(
    instance,
    model.predict_proba,
    num_features=5
)

# Get explanation
print("LIME Explanation:")
for feature, weight in exp.as_list():
    print(f"  {feature}: {weight:.3f}")

# SHAP Example
import shap

# For tree-based models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For any model
explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
shap_values = explainer.shap_values(X_test[:10])

# Visualize
shap.initjs()
# Single prediction
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test[0])
# Summary
shap.summary_plot(shap_values[1], X_test, feature_names=feature_names)
```

**Interview Tips:**
- SHAP TreeExplainer is fast for tree models
- LIME can be unstable (run multiple times)
- SHAP values sum to prediction difference from baseline

---

## Question 73
- [ ] Done

**How do you implement feature importance analysis for classifiers?**

### Answer

**Definition:**
Feature importance quantifies how much each feature contributes to model predictions, enabling feature selection, model understanding, and debugging.

**Methods:**

| Method | Approach | Model Type |
|--------|----------|-----------|
| Coefficients | Weight magnitude | Linear models |
| Gini/Entropy | Split improvement | Trees |
| Permutation | Shuffle feature, measure drop | Any |
| SHAP | Shapley values | Any |
| Drop-column | Remove feature, retrain | Any |

**Python Code Example:**
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

# 1. Linear model coefficients
lr = LogisticRegression().fit(X_train, y_train)
coef_importance = np.abs(lr.coef_[0])
print("Logistic Regression Importance:")
for name, imp in sorted(zip(feature_names, coef_importance), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.4f}")

# 2. Tree-based importance (Gini)
rf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
gini_importance = rf.feature_importances_
print("\nRandom Forest Gini Importance:")
for name, imp in sorted(zip(feature_names, gini_importance), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.4f}")

# 3. Permutation importance (most reliable)
perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=42)
print("\nPermutation Importance:")
for name, imp in sorted(zip(feature_names, perm_imp.importances_mean), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.4f}")

# 4. Drop-column importance
def drop_column_importance(model, X, y, feature_names):
    baseline = model.score(X, y)
    importances = {}
    
    for i, name in enumerate(feature_names):
        X_dropped = np.delete(X, i, axis=1)
        model_dropped = type(model)().fit(X_dropped, y)
        score_dropped = model_dropped.score(X_dropped, y)
        importances[name] = baseline - score_dropped
    
    return importances

# 5. SHAP importance
import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap_importance = np.abs(shap_values[1]).mean(axis=0)
```

**Interview Tips:**
- Permutation importance most reliable, but slow
- Gini importance biased toward high-cardinality features
- Use multiple methods and compare

---

## Question 74
- [ ] Done

**How do Convolutional Neural Networks (CNNs) perform classification?**

### Answer

**Definition:**
CNNs are neural networks designed for image classification, using convolutional layers to automatically learn spatial hierarchies of features from raw pixels.

**Architecture Components:**

| Layer | Function |
|-------|----------|
| Convolutional | Extract local features (edges, textures) |
| Pooling | Reduce spatial dimensions |
| Activation (ReLU) | Introduce non-linearity |
| Fully Connected | Classification head |
| Softmax | Output probabilities |

**How It Works:**
1. Input image (H × W × C)
2. Convolutional layers learn filters
3. Pooling reduces size
4. Flatten to vector
5. Dense layers for classification
6. Softmax for probabilities

**Python Code Example:**
```python
# PyTorch CNN
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Keras/TensorFlow
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

**Interview Tips:**
- Pooling provides translation invariance
- Deeper networks need skip connections (ResNet)
- Transfer learning from ImageNet very effective

---

## Question 75
- [ ] Done

**How do Recurrent Neural Networks (RNNs) and LSTMs perform sequence classification?**

### Answer

**Definition:**
RNNs process sequential data by maintaining hidden state across time steps. LSTMs (Long Short-Term Memory) solve the vanishing gradient problem with gating mechanisms.

**RNN Equation:**
$$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b)$$

**LSTM Gates:**
| Gate | Function |
|------|----------|
| Forget | What to discard from memory |
| Input | What new info to store |
| Output | What to output from memory |

**Sequence Classification:**
- Process each time step
- Use final hidden state (or pooled states) for classification
- Apply dense layer + softmax

**Python Code Example:**
```python
# PyTorch LSTM
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate final hidden states from both directions
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

# Keras/TensorFlow
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_length),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

**Use Cases:**
- Sentiment analysis
- Text classification
- Time series classification
- Speech recognition

**Interview Tips:**
- Bidirectional LSTM captures both past and future context
- GRU is simpler alternative to LSTM
- Transformers often outperform RNNs now

---

## Question 76
- [ ] Done

**How do Transformer models perform classification tasks?**

### Answer

**Definition:**
Transformers use self-attention mechanisms to process sequences in parallel, capturing long-range dependencies. For classification, typically use [CLS] token representation or pooled outputs.

**Key Components:**
| Component | Function |
|-----------|----------|
| Self-Attention | Relate all positions to each other |
| Multi-Head | Multiple attention patterns |
| Position Encoding | Inject sequence order |
| Feed-Forward | Transform representations |

**Self-Attention:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Classification Approach:**
1. Add [CLS] token at start
2. Process through transformer layers
3. Use [CLS] representation for classification
4. Apply classification head

**Python Code Example:**
```python
# Using Hugging Face Transformers
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize
text = "This movie was great!"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# Classify
outputs = model(**inputs)
predictions = torch.softmax(outputs.logits, dim=1)
print(f"Positive probability: {predictions[0][1]:.3f}")

# Fine-tuning
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy='epoch'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

**Popular Models:**
- BERT: Bidirectional encoding
- RoBERTa: Optimized BERT
- DistilBERT: Smaller, faster
- GPT: Generative, unidirectional

**Interview Tips:**
- Transformers are state-of-the-art for NLP
- Pre-training + fine-tuning paradigm
- Attention scores provide interpretability

---

## Question 77
- [ ] Done

**What is attention mechanism and its role in classification?**

### Answer

**Definition:**
Attention allows models to focus on relevant parts of input when making predictions. It computes weighted combinations of values based on query-key similarity.

**Types:**
| Type | Description |
|------|-------------|
| Self-attention | Attend to other positions in same sequence |
| Cross-attention | Attend to different sequence (encoder-decoder) |
| Multi-head | Multiple attention in parallel |

**Mathematical Formulation:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- Q (Query): What we're looking for
- K (Key): What we match against
- V (Value): What we retrieve

**Benefits for Classification:**
- Focus on important parts of input
- Handle variable-length inputs
- Provide interpretability (attention weights)

**Python Code Example:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention: (batch, heads, seq, dim)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return self.out(out), attention

class AttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = SelfAttention(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        attended, weights = self.attention(embedded)
        pooled = attended.mean(dim=1)
        return self.fc(pooled), weights
```

**Interview Tips:**
- Attention weights show what model focuses on
- Multi-head captures different relationships
- O(n²) complexity limits very long sequences

---

## Question 78
- [ ] Done

**How do you implement deep learning classification with PyTorch or TensorFlow?**

### Answer

**Definition:**
Deep learning frameworks provide building blocks for neural network classification: layers, loss functions, optimizers, and training loops.

**PyTorch Implementation:**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define model
class DeepClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Training
model = DeepClassifier(input_dim=20, hidden_dims=[128, 64, 32], num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

for epoch in range(100):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
```

**TensorFlow/Keras Implementation:**

```python
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Define model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(20,)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

# Train
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr]
)
```

**Interview Tips:**
- PyTorch: more control, debugging, research
- TensorFlow/Keras: faster prototyping, production
- Always use validation for early stopping

---

## Question 79
- [ ] Done

**What are the best practices for training deep classification models?**

### Answer

**Definition:**
Best practices ensure deep models train effectively, generalize well, and avoid common pitfalls like overfitting, vanishing gradients, and slow convergence.

**Key Practices:**

| Category | Practice |
|----------|----------|
| Data | Normalize inputs, augmentation |
| Architecture | Batch norm, residual connections |
| Training | Learning rate scheduling, early stopping |
| Regularization | Dropout, weight decay |
| Initialization | Xavier/He initialization |

**Python Code Example:**
```python
import torch
import torch.nn as nn

class BestPracticeClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        
        # He initialization for ReLU
        self.fc1 = nn.Linear(input_dim, 256)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)

# Training with best practices
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = BestPracticeClassifier(input_dim=100, num_classes=10)

# Weight decay (L2 regularization)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Learning rate scheduling
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Mixed precision training (faster on GPU)
scaler = torch.cuda.amp.GradScaler()

# Training loop with best practices
best_val_loss = float('inf')
patience_counter = 0
max_patience = 10

for epoch in range(100):
    model.train()
    for batch_X, batch_y in train_loader:
        with torch.cuda.amp.autocast():  # Mixed precision
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    # Validation
    val_loss = evaluate(model, val_loader)
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            break
```

**Checklist:**
- [ ] Normalize inputs (mean=0, std=1)
- [ ] Use batch normalization
- [ ] Apply dropout
- [ ] Use weight decay
- [ ] Learning rate scheduling
- [ ] Early stopping
- [ ] Gradient clipping
- [ ] Data augmentation (if applicable)

---

## Question 80
- [ ] Done

**How do you deploy classification models to production?**

### Answer

**Definition:**
Deployment makes trained classification models available for real-time predictions in production environments, requiring consideration of latency, scalability, and reliability.

**Deployment Options:**

| Option | Use Case |
|--------|----------|
| REST API | Web services, microservices |
| Batch | Large-scale offline processing |
| Edge | Mobile, IoT devices |
| Serverless | Variable load, auto-scaling |

**Python Code Example:**
```python
# Flask REST API
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model once at startup
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    X = np.array(data).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0].max()
    
    return jsonify({
        'prediction': int(prediction),
        'confidence': float(probability)
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

# FastAPI (async, faster)
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    features: list

@app.post('/predict')
async def predict(request: PredictionRequest):
    X = np.array(request.features).reshape(1, -1)
    prediction = model.predict(X)[0]
    return {'prediction': int(prediction)}

# Docker deployment
"""
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
```

**Model Serving Frameworks:**
- TensorFlow Serving
- TorchServe
- MLflow
- Seldon Core
- BentoML

**Interview Tips:**
- Separate model from serving code
- Version your models
- Add health checks and monitoring
- Consider latency vs throughput tradeoffs

---

## Question 81
- [ ] Done

**What are the considerations for model serialization and loading?**

### Answer

**Definition:**
Model serialization saves trained models to files for later use. Key considerations include format compatibility, file size, loading speed, and security.

**Serialization Options:**

| Format | Library | Pros | Cons |
|--------|---------|------|------|
| pickle/joblib | sklearn | Easy, full object | Security risk, Python-only |
| ONNX | Multiple | Cross-framework | Conversion complexity |
| SavedModel | TensorFlow | Production-ready | TF-specific |
| TorchScript | PyTorch | Optimized | PyTorch-specific |
| PMML | sklearn | Standardized | Limited support |

**Python Code Example:**
```python
import joblib
import pickle
import json
import numpy as np

# 1. Joblib (sklearn recommended)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(X_train, y_train)

joblib.dump(model, 'model.joblib')
loaded_model = joblib.load('model.joblib')

# 2. Pickle (basic)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# 3. ONNX (cross-framework)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open('model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

# 4. PyTorch
import torch
model = MyTorchModel()
model.load_state_dict(torch.load('model.pt'))  # State dict (recommended)
torch.save(model, 'full_model.pt')  # Full model

# TorchScript for production
scripted = torch.jit.script(model)
scripted.save('model_scripted.pt')

# 5. TensorFlow SavedModel
model.save('saved_model/')
loaded = tf.keras.models.load_model('saved_model/')

# Save metadata
metadata = {
    'features': feature_names,
    'classes': class_names,
    'version': '1.0.0',
    'preprocessing': {'mean': scaler.mean_.tolist(), 'std': scaler.scale_.tolist()}
}
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f)
```

**Best Practices:**
- Save preprocessing with model
- Version your models
- Include metadata (features, classes)
- Test loading before deployment

**Interview Tips:**
- Pickle has security vulnerabilities
- ONNX enables framework interoperability
- Always version your models

---

## Question 82
- [ ] Done

**How do you monitor classification model performance in production?**

### Answer

**Definition:**
Production monitoring tracks model performance, data drift, and system health to ensure classification models continue to work correctly over time.

**What to Monitor:**

| Category | Metrics |
|----------|---------|
| Performance | Accuracy, latency, throughput |
| Data | Distribution drift, missing values |
| Model | Prediction distribution, confidence |
| System | CPU, memory, errors |

**Python Code Example:**
```python
import numpy as np
from scipy import stats
import logging
from datetime import datetime

class ModelMonitor:
    def __init__(self, reference_data, reference_predictions):
        self.ref_data = reference_data
        self.ref_preds = reference_predictions
        self.logger = logging.getLogger('model_monitor')
    
    def check_data_drift(self, current_data, threshold=0.05):
        """Detect data drift using KS test"""
        drift_detected = False
        drift_features = []
        
        for i in range(current_data.shape[1]):
            stat, p_value = stats.ks_2samp(
                self.ref_data[:, i], 
                current_data[:, i]
            )
            if p_value < threshold:
                drift_detected = True
                drift_features.append(i)
        
        if drift_detected:
            self.logger.warning(f"Data drift detected in features: {drift_features}")
        
        return drift_detected, drift_features
    
    def check_prediction_drift(self, current_predictions, threshold=0.05):
        """Check if prediction distribution changed"""
        ref_dist = np.bincount(self.ref_preds, minlength=len(np.unique(self.ref_preds)))
        curr_dist = np.bincount(current_predictions, minlength=len(ref_dist))
        
        ref_dist = ref_dist / ref_dist.sum()
        curr_dist = curr_dist / curr_dist.sum()
        
        # Chi-square test
        stat, p_value = stats.chisquare(curr_dist, ref_dist)
        
        if p_value < threshold:
            self.logger.warning(f"Prediction drift detected. p-value: {p_value}")
            return True
        return False
    
    def log_prediction(self, features, prediction, confidence, latency_ms):
        """Log individual predictions for monitoring"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'prediction': int(prediction),
            'confidence': float(confidence),
            'latency_ms': latency_ms
        }
        self.logger.info(json.dumps(log_entry))

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

prediction_counter = Counter('predictions_total', 'Total predictions', ['class'])
latency_histogram = Histogram('prediction_latency_seconds', 'Prediction latency')
confidence_gauge = Gauge('prediction_confidence', 'Average prediction confidence')

@latency_histogram.time()
def predict_with_monitoring(model, X):
    prediction = model.predict(X)
    confidence = model.predict_proba(X).max()
    
    prediction_counter.labels(class=str(prediction[0])).inc()
    confidence_gauge.set(confidence)
    
    return prediction
```

**Monitoring Tools:**
- MLflow
- Evidently AI
- Grafana + Prometheus
- AWS SageMaker Model Monitor
- WhyLabs

**Interview Tips:**
- Monitor both data and model
- Set up alerts for drift
- Log predictions for debugging

---

## Question 83
- [ ] Done

**What is A/B testing for classification models?**

### Answer

**Definition:**
A/B testing compares two model versions by randomly splitting traffic and measuring which performs better on business metrics. It provides statistical evidence for model deployment decisions.

**Key Concepts:**
- Control (A): Current/baseline model
- Treatment (B): New model
- Random assignment
- Statistical significance testing

**Design Considerations:**

| Factor | Consideration |
|--------|--------------|
| Sample size | Power analysis to detect effect |
| Duration | Long enough for significance |
| Metrics | Primary and guardrail metrics |
| Randomization | User-level, session-level |

**Python Code Example:**
```python
import numpy as np
from scipy import stats

class ABTest:
    def __init__(self, control_model, treatment_model):
        self.control = control_model
        self.treatment = treatment_model
        self.control_results = []
        self.treatment_results = []
    
    def assign_variant(self, user_id):
        """Randomly assign to control or treatment"""
        np.random.seed(hash(user_id) % 2**32)
        return 'treatment' if np.random.random() < 0.5 else 'control'
    
    def predict(self, user_id, features):
        """Route to appropriate model"""
        variant = self.assign_variant(user_id)
        
        if variant == 'control':
            return self.control.predict(features), 'control'
        else:
            return self.treatment.predict(features), 'treatment'
    
    def record_outcome(self, variant, success):
        """Record conversion/success for analysis"""
        if variant == 'control':
            self.control_results.append(success)
        else:
            self.treatment_results.append(success)
    
    def analyze(self, alpha=0.05):
        """Statistical analysis of A/B test"""
        control_successes = sum(self.control_results)
        treatment_successes = sum(self.treatment_results)
        n_control = len(self.control_results)
        n_treatment = len(self.treatment_results)
        
        # Conversion rates
        control_rate = control_successes / n_control
        treatment_rate = treatment_successes / n_treatment
        
        # Two-proportion z-test
        pooled_rate = (control_successes + treatment_successes) / (n_control + n_treatment)
        se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/n_control + 1/n_treatment))
        z_stat = (treatment_rate - control_rate) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Lift
        lift = (treatment_rate - control_rate) / control_rate * 100
        
        return {
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'lift': lift,
            'p_value': p_value,
            'significant': p_value < alpha
        }

# Power analysis
def required_sample_size(baseline_rate, mde, alpha=0.05, power=0.8):
    """Calculate required sample size per variant"""
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)
    
    n = (z_alpha * np.sqrt(2 * p1 * (1-p1)) + 
         z_beta * np.sqrt(p1*(1-p1) + p2*(1-p2)))**2 / (p2 - p1)**2
    
    return int(np.ceil(n))
```

**Interview Tips:**
- Always calculate required sample size first
- Use guardrail metrics to catch regressions
- Consider multiple testing correction

---

## Question 84
- [ ] Done

**How do you implement model versioning and experiment tracking?**

### Answer

**Definition:**
Model versioning tracks different versions of models, their parameters, and performance. Experiment tracking logs all details of training runs for reproducibility and comparison.

**What to Track:**
| Category | Examples |
|----------|----------|
| Code | Git commit, notebook |
| Data | Dataset version, hash |
| Parameters | Hyperparameters, config |
| Metrics | Accuracy, loss, confusion matrix |
| Artifacts | Model files, plots |
| Environment | Dependencies, hardware |

**Python Code Example (MLflow):**
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Start tracking
mlflow.set_experiment("classification_experiment")

with mlflow.start_run(run_name="rf_baseline"):
    # Log parameters
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5
    }
    mlflow.log_params(params)
    
    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Log metrics
    mlflow.log_metrics({
        'accuracy': accuracy,
        'f1_score': f1
    })
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    
    # Register model
    mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/model",
        "classification_model"
    )

# Compare experiments
runs = mlflow.search_runs(experiment_names=["classification_experiment"])
print(runs[['run_id', 'params.n_estimators', 'metrics.accuracy']])

# Load model
model_uri = "models:/classification_model/Production"
loaded_model = mlflow.sklearn.load_model(model_uri)
```

**Tools:**
- MLflow (most popular)
- Weights & Biases (W&B)
- Neptune.ai
- DVC (Data Version Control)
- Comet.ml

**Interview Tips:**
- Reproducibility requires logging everything
- Use model registry for production models
- Tag production models clearly

---

## Question 85
- [ ] Done

**What is continuous integration and deployment (CI/CD) for ML models?**

### Answer

**Definition:**
ML CI/CD automates the process of testing, validating, and deploying machine learning models. It ensures model quality and enables rapid, reliable releases.

**Pipeline Stages:**

| Stage | Activities |
|-------|-----------|
| Data validation | Schema checks, drift detection |
| Model training | Automated retraining |
| Model validation | Performance tests, fairness checks |
| Model deployment | Container build, A/B rollout |
| Monitoring | Performance tracking, alerts |

**Python Code Example:**
```python
# GitHub Actions workflow for ML
# .github/workflows/ml_pipeline.yml
"""
name: ML Pipeline

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly retraining

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest tests/

  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Train model
        run: python train.py
      - name: Upload model
        uses: actions/upload-artifact@v2
        with:
          name: model
          path: models/

  validate:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - name: Download model
        uses: actions/download-artifact@v2
      - name: Validate model
        run: python validate.py

  deploy:
    needs: validate
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: python deploy.py
"""

# Model validation script
import json
import sys

def validate_model(model_path, test_data_path, thresholds):
    model = load_model(model_path)
    X_test, y_test = load_data(test_data_path)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'latency_p99': measure_latency(model, X_test)
    }
    
    # Check thresholds
    passed = True
    for metric, threshold in thresholds.items():
        if metrics[metric] < threshold:
            print(f"FAILED: {metric} = {metrics[metric]:.4f} < {threshold}")
            passed = False
    
    return passed, metrics

# Tests for ML
def test_model_accuracy():
    """Minimum accuracy threshold"""
    model = load_model('models/latest.pkl')
    X_test, y_test = load_test_data()
    accuracy = model.score(X_test, y_test)
    assert accuracy >= 0.85, f"Accuracy {accuracy} below threshold"

def test_model_fairness():
    """Check fairness across groups"""
    model = load_model('models/latest.pkl')
    X_test, y_test, sensitive = load_test_data_with_sensitive()
    
    y_pred = model.predict(X_test)
    di = disparate_impact(y_pred, sensitive)
    assert di >= 0.8, f"Disparate impact {di} below threshold"

def test_no_data_leakage():
    """Ensure test data not in training"""
    train_hashes = set(hash(tuple(x)) for x in X_train)
    test_hashes = set(hash(tuple(x)) for x in X_test)
    overlap = train_hashes & test_hashes
    assert len(overlap) == 0, "Data leakage detected"
```

**Tools:**
- GitHub Actions / GitLab CI
- Jenkins
- Kubeflow Pipelines
- MLflow + CI/CD
- DVC pipelines

**Interview Tips:**
- Test both code and model performance
- Automate retraining triggers
- Implement canary/gradual rollouts

---

## Question 86
- [ ] Done

**How do you handle model retraining and updates?**

### Answer

**Definition:**
Model retraining updates models with new data to maintain performance. Strategies include scheduled retraining, triggered retraining, and continuous learning.

**Retraining Strategies:**

| Strategy | Trigger | Use Case |
|----------|---------|----------|
| Scheduled | Time-based (daily/weekly) | Stable environments |
| Triggered | Performance drop, drift | Dynamic environments |
| Continuous | Each new sample | Streaming data |

**Python Code Example:**
```python
import numpy as np
from datetime import datetime

class ModelRetrainingManager:
    def __init__(self, model, performance_threshold=0.85, drift_threshold=0.05):
        self.model = model
        self.perf_threshold = performance_threshold
        self.drift_threshold = drift_threshold
        self.last_retrain = datetime.now()
        self.reference_data = None
    
    def should_retrain(self, X_current, y_current, y_pred):
        """Check if retraining is needed"""
        reasons = []
        
        # 1. Performance degradation
        current_accuracy = np.mean(y_current == y_pred)
        if current_accuracy < self.perf_threshold:
            reasons.append(f"Performance drop: {current_accuracy:.3f}")
        
        # 2. Data drift
        if self.reference_data is not None:
            drift_detected, _ = self.detect_drift(X_current)
            if drift_detected:
                reasons.append("Data drift detected")
        
        # 3. Time-based (e.g., monthly)
        days_since_retrain = (datetime.now() - self.last_retrain).days
        if days_since_retrain > 30:
            reasons.append(f"Scheduled retraining ({days_since_retrain} days)")
        
        return len(reasons) > 0, reasons
    
    def retrain(self, X_new, y_new, strategy='full'):
        """Retrain model"""
        if strategy == 'full':
            # Full retraining on all available data
            self.model.fit(X_new, y_new)
        elif strategy == 'incremental':
            # Incremental update (if supported)
            self.model.partial_fit(X_new, y_new)
        elif strategy == 'warm_start':
            # Initialize from current weights
            self.model.set_params(warm_start=True)
            self.model.fit(X_new, y_new)
        
        self.last_retrain = datetime.now()
        self.reference_data = X_new  # Update reference
        
        return self.model
    
    def safe_retrain_and_deploy(self, X_train, y_train, X_val, y_val):
        """Retrain with validation before deployment"""
        # Train new model
        new_model = clone(self.model)
        new_model.fit(X_train, y_train)
        
        # Validate
        new_score = new_model.score(X_val, y_val)
        old_score = self.model.score(X_val, y_val)
        
        # Only deploy if improved or within tolerance
        if new_score >= old_score - 0.01:
            self.model = new_model
            self.last_retrain = datetime.now()
            return True, new_score
        else:
            return False, old_score

# Automated retraining pipeline
class RetrainingPipeline:
    def __init__(self, data_loader, model_trainer, validator, deployer):
        self.data_loader = data_loader
        self.trainer = model_trainer
        self.validator = validator
        self.deployer = deployer
    
    def run(self):
        # 1. Load new data
        X_train, y_train, X_val, y_val = self.data_loader.get_latest_data()
        
        # 2. Train new model
        new_model = self.trainer.train(X_train, y_train)
        
        # 3. Validate
        metrics = self.validator.validate(new_model, X_val, y_val)
        
        # 4. Deploy if validated
        if self.validator.passes_threshold(metrics):
            self.deployer.deploy(new_model, metrics)
            return True, metrics
        
        return False, metrics
```

**Interview Tips:**
- Always validate before deployment
- Keep old model as fallback
- Log retraining decisions

---

## Question 87
- [ ] Done

**What are the emerging trends in classification algorithms?**

### Answer

**Definition:**
Emerging trends in classification include foundation models, AutoML, neural architecture search, graph neural networks, and edge deployment.

**Key Trends:**

| Trend | Description |
|-------|-------------|
| Foundation Models | Large pre-trained models (GPT, BERT) |
| AutoML | Automated model selection and tuning |
| Few-shot/Zero-shot | Classification with minimal examples |
| Graph Neural Networks | Classification on graph-structured data |
| Edge ML | On-device classification |
| Explainable AI | Interpretable models by design |

**Python Code Example:**
```python
# 1. Zero-shot classification with transformers
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
    "I love playing soccer with my friends",
    candidate_labels=["sports", "politics", "technology"]
)
print(result['labels'][0])  # sports

# 2. AutoML with auto-sklearn
from autosklearn.classification import AutoSklearnClassifier

automl = AutoSklearnClassifier(time_left_for_this_task=300)
automl.fit(X_train, y_train)
print(f"AutoML accuracy: {automl.score(X_test, y_test):.3f}")

# 3. Graph Neural Network classification
import torch
from torch_geometric.nn import GCNConv

class GCNClassifier(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# 4. Efficient inference (quantization)
import torch.quantization

model_fp32 = load_model()
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)

# 5. Continual learning
class ContinualClassifier:
    def __init__(self, base_model):
        self.model = base_model
        self.memory = []  # Experience replay buffer
    
    def learn_task(self, X, y, task_id):
        # Add to memory
        self.memory.extend(list(zip(X, y)))
        
        # Replay from memory to avoid forgetting
        if len(self.memory) > 1000:
            replay_idx = np.random.choice(len(self.memory), 1000)
            replay_data = [self.memory[i] for i in replay_idx]
            X_replay, y_replay = zip(*replay_data)
            X_combined = np.vstack([X, X_replay])
            y_combined = np.hstack([y, y_replay])
            self.model.fit(X_combined, y_combined)
```

**Interview Tips:**
- Foundation models are changing NLP classification
- AutoML makes ML accessible
- Edge deployment requires model compression

---

## Question 88
- [ ] Done

**What is Neural Architecture Search (NAS) for classification?**

### Answer

**Definition:**
NAS automatically discovers optimal neural network architectures for classification tasks, searching over possible layer types, connections, and hyperparameters.

**Search Strategies:**

| Strategy | Approach |
|----------|----------|
| Random search | Sample random architectures |
| Reinforcement learning | Learn to generate architectures |
| Evolutionary | Evolve population of architectures |
| Differentiable (DARTS) | Gradient-based search |

**Search Space:**
- Layer types (conv, dense, LSTM)
- Number of layers
- Layer width
- Skip connections
- Activation functions

**Python Code Example:**
```python
# Using keras-tuner
import keras_tuner as kt
from tensorflow import keras

def build_model(hp):
    model = keras.Sequential()
    
    # Search over number of layers
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(keras.layers.Dense(
            units=hp.Int(f'units_{i}', 32, 512, step=32),
            activation=hp.Choice('activation', ['relu', 'tanh', 'selu'])
        ))
        
        if hp.Boolean('dropout'):
            model.add(keras.layers.Dropout(hp.Float('dropout_rate', 0.1, 0.5)))
    
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Search
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,
    factor=3
)

tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=50)

# Get best model
best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters()[0]
```

**Popular NAS Methods:**
- NASNet: RL-based search
- ENAS: Efficient NAS with weight sharing
- DARTS: Differentiable architecture search
- AutoKeras: Automated Keras

**Interview Tips:**
- NAS is computationally expensive
- DARTS makes it more efficient
- Pre-searched architectures often sufficient

---

## Question 89
- [ ] Done

**How do you implement AutoML for classification tasks?**

### Answer

**Definition:**
AutoML automates the ML pipeline including data preprocessing, feature engineering, model selection, and hyperparameter tuning, making ML accessible to non-experts.

**AutoML Components:**

| Component | Automation |
|-----------|-----------|
| Data cleaning | Missing values, outliers |
| Feature engineering | Creation, selection |
| Model selection | Algorithm choice |
| Hyperparameter tuning | Optimization |
| Ensemble | Model combination |

**Python Code Example:**
```python
# 1. Auto-sklearn
from autosklearn.classification import AutoSklearnClassifier

automl = AutoSklearnClassifier(
    time_left_for_this_task=3600,  # 1 hour
    per_run_time_limit=300,
    n_jobs=-1,
    memory_limit=8192,
    ensemble_size=50
)
automl.fit(X_train, y_train)
print(f"Accuracy: {automl.score(X_test, y_test):.3f}")
print(automl.leaderboard())

# 2. TPOT
from tpot import TPOTClassifier

tpot = TPOTClassifier(
    generations=5,
    population_size=50,
    cv=5,
    verbosity=2,
    random_state=42
)
tpot.fit(X_train, y_train)
print(f"Accuracy: {tpot.score(X_test, y_test):.3f}")
tpot.export('tpot_best_pipeline.py')

# 3. H2O AutoML
import h2o
from h2o.automl import H2OAutoML

h2o.init()
train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
test = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))

aml = H2OAutoML(max_runtime_secs=3600, seed=42)
aml.train(x=feature_names, y='target', training_frame=train)

print(aml.leaderboard)
best_model = aml.leader
predictions = best_model.predict(test)

# 4. PyCaret
from pycaret.classification import *

clf = setup(data=df, target='target', session_id=42)
best = compare_models()
tuned = tune_model(best)
final = finalize_model(tuned)
```

**Tools Comparison:**
| Tool | Strengths |
|------|----------|
| Auto-sklearn | Robust, meta-learning |
| TPOT | Genetic programming |
| H2O | Distributed, production-ready |
| PyCaret | Easy to use, low-code |
| AutoGluon | Deep learning focus |

**Interview Tips:**
- AutoML is good for baselines
- Still need domain expertise
- Understand what AutoML finds

---

## Question 90
- [ ] Done

**What are Graph Neural Networks and how do they perform classification?**

### Answer

**Definition:**
Graph Neural Networks (GNNs) perform classification on graph-structured data, where nodes have features and edges represent relationships. They aggregate information from neighbors.

**Types:**

| Type | Approach |
|------|----------|
| GCN | Spectral convolution |
| GraphSAGE | Sampling + aggregation |
| GAT | Attention-based |
| GIN | Maximally expressive |

**Message Passing:**
$$h_v^{(k+1)} = \text{UPDATE}\left(h_v^{(k)}, \text{AGGREGATE}(\{h_u^{(k)}: u \in N(v)\})\right)$$

**Classification Tasks:**
- Node classification: classify each node
- Graph classification: classify entire graph
- Link prediction: predict edge existence

**Python Code Example:**
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data

# Node Classification
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = torch.nn.Linear(32, num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        return self.fc(x)

# Graph Classification
class GraphClassifier(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GATConv(num_features, 64, heads=4)
        self.conv2 = GATConv(64 * 4, 64)
        self.fc = torch.nn.Linear(64, num_classes)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # Aggregate node features
        
        return self.fc(x)

# Training
model = GCN(num_features=34, num_classes=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

**Use Cases:**
- Social networks (user classification)
- Molecules (drug discovery)
- Citation networks (paper classification)
- Knowledge graphs

**Interview Tips:**
- GNNs capture relational structure
- Over-smoothing with too many layers
- GraphSAGE scales to large graphs

---

## Question 91
- [ ] Done

**How do you handle extremely imbalanced classification with deep learning?**

### Answer

**Definition:**
Deep learning for imbalanced data requires special techniques since standard training is biased toward majority classes. Methods include resampling, loss modification, and architecture changes.

**Techniques:**

| Technique | Approach |
|-----------|----------|
| Weighted loss | Higher weight for minority |
| Focal loss | Down-weight easy examples |
| Oversampling | Generate minority samples |
| Undersampling | Reduce majority samples |
| Two-stage | Coarse then fine-grained |

**Python Code Example:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Weighted Cross-Entropy
class_counts = torch.bincount(y_train)
weights = 1.0 / class_counts.float()
weights = weights / weights.sum()

criterion = nn.CrossEntropyLoss(weight=weights)

# 2. Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        
        return focal_loss.mean()

# 3. Weighted Sampler
from torch.utils.data import WeightedRandomSampler

class_counts = torch.bincount(y_train)
sample_weights = 1.0 / class_counts[y_train].float()
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

# 4. SMOTE for features
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 5. Class-balanced batch
class BalancedBatchSampler:
    def __init__(self, labels, batch_size, samples_per_class):
        self.labels = labels
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.classes = torch.unique(labels)
        self.class_indices = {c.item(): (labels == c).nonzero().squeeze() 
                             for c in self.classes}
    
    def __iter__(self):
        for _ in range(len(self)):
            batch = []
            for c in self.classes:
                idx = self.class_indices[c.item()]
                selected = idx[torch.randperm(len(idx))[:self.samples_per_class]]
                batch.extend(selected.tolist())
            yield batch
    
    def __len__(self):
        return 1000  # Number of batches
```

**Interview Tips:**
- Focal loss is standard for detection
- Use weighted sampling over loss weighting
- Two-stage training often effective

---

## Question 92
- [ ] Done

**What is contrastive learning and its application to classification?**

### Answer

**Definition:**
Contrastive learning learns representations by bringing similar samples closer and pushing dissimilar samples apart in embedding space. Useful for self-supervised pre-training before classification.

**Core Idea:**
- Positive pairs: same sample with different augmentations
- Negative pairs: different samples
- Learn embeddings where positives are close, negatives are far

**Popular Methods:**
| Method | Approach |
|--------|----------|
| SimCLR | Simple contrastive |
| MoCo | Momentum contrast |
| BYOL | No negatives |
| SwAV | Clustering |

**Python Code Example:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveModel(nn.Module):
    def __init__(self, encoder, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        projections = self.projector(features)
        return F.normalize(projections, dim=1)

def contrastive_loss(z1, z2, temperature=0.5):
    """NT-Xent (SimCLR) loss"""
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    
    # Similarity matrix
    sim = torch.mm(z, z.t()) / temperature
    
    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, float('-inf'))
    
    # Positive pairs
    pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=z.device)
    pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, device=z.device)
    pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, device=z.device)
    
    # InfoNCE loss
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels])
    loss = F.cross_entropy(sim, labels)
    
    return loss

# Training loop for self-supervised pre-training
for batch in dataloader:
    x1 = augment(batch)  # First augmentation
    x2 = augment(batch)  # Second augmentation
    
    z1 = model(x1)
    z2 = model(x2)
    
    loss = contrastive_loss(z1, z2)
    loss.backward()
    optimizer.step()

# Fine-tune for classification
model.encoder.fc = nn.Linear(encoder.output_dim, num_classes)
# Train classification head on labeled data
```

**Interview Tips:**
- Great for limited labeled data
- Data augmentation is crucial
- Linear evaluation: train only linear classifier on frozen features

---

## Question 93
- [ ] Done

**How do you implement self-supervised learning for classification?**

### Answer

**Definition:**
Self-supervised learning creates supervised signals from unlabeled data through pretext tasks. The learned representations transfer to classification tasks.

**Pretext Tasks:**

| Task | Description |
|------|-------------|
| Rotation prediction | Predict image rotation (0°, 90°, 180°, 270°) |
| Jigsaw puzzle | Arrange shuffled patches |
| Colorization | Predict colors from grayscale |
| Masked prediction | Predict masked portions |
| Contrastive | Match augmentations |

**Python Code Example:**
```python
import torch
import torch.nn as nn
from torchvision import transforms

# 1. Rotation Prediction (Images)
class RotationPredictor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.output_dim, 4)  # 4 rotations
    
    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

def rotation_pretext_data(images):
    rotated = []
    labels = []
    for img in images:
        for i, angle in enumerate([0, 90, 180, 270]):
            rotated.append(transforms.functional.rotate(img, angle))
            labels.append(i)
    return torch.stack(rotated), torch.tensor(labels)

# 2. Masked Language Model (Text)
class MLMHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = torch.relu(x)
        x = self.layer_norm(x)
        return self.decoder(x)

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked
    inputs[masked_indices] = tokenizer.mask_token_id
    return inputs, labels

# 3. Self-supervised then supervised pipeline
class SelfSupervisedPipeline:
    def __init__(self, encoder, pretext_task):
        self.encoder = encoder
        self.pretext_task = pretext_task
    
    def pretrain(self, unlabeled_data, epochs=100):
        """Self-supervised pre-training"""
        for epoch in range(epochs):
            for batch in unlabeled_data:
                x, pseudo_labels = self.pretext_task.create_task(batch)
                predictions = self.pretext_task.model(x)
                loss = self.pretext_task.loss(predictions, pseudo_labels)
                loss.backward()
                self.optimizer.step()
    
    def finetune(self, labeled_data, num_classes, epochs=10):
        """Supervised fine-tuning"""
        classifier = nn.Linear(self.encoder.output_dim, num_classes)
        
        for epoch in range(epochs):
            for x, y in labeled_data:
                features = self.encoder(x)
                predictions = classifier(features)
                loss = F.cross_entropy(predictions, y)
                loss.backward()
                self.optimizer.step()
```

**Interview Tips:**
- Pretext task should be related to downstream task
- BERT uses masked language modeling
- Contrastive methods currently most effective

---

## Question 94
- [ ] Done

**What is knowledge distillation for classification model compression?**

### Answer

**Definition:**
Knowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model by training the student to match teacher's soft outputs (probabilities).

**Core Idea:**
- Teacher: large, accurate model
- Student: small, fast model
- Soft labels: teacher's probability distribution
- Temperature: softens probability distribution

**Distillation Loss:**
$$L = \alpha \cdot L_{CE}(y, \sigma(z_s)) + (1-\alpha) \cdot T^2 \cdot L_{KL}(\sigma(z_t/T), \sigma(z_s/T))$$

**Python Code Example:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels):
        # Hard label loss
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Soft label loss (KL divergence)
        soft_targets = F.softmax(teacher_logits / self.T, dim=1)
        soft_predictions = F.log_softmax(student_logits / self.T, dim=1)
        soft_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean')
        
        # Combined loss
        return self.alpha * hard_loss + (1 - self.alpha) * (self.T ** 2) * soft_loss

def distill(teacher, student, train_loader, epochs=10):
    teacher.eval()
    criterion = DistillationLoss(temperature=4.0, alpha=0.5)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        student.train()
        for x, y in train_loader:
            # Get teacher predictions
            with torch.no_grad():
                teacher_logits = teacher(x)
            
            # Get student predictions
            student_logits = student(x)
            
            # Compute loss
            loss = criterion(student_logits, teacher_logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}: Student accuracy = {evaluate(student):.3f}")

# Self-distillation (same architecture)
class SelfDistillation(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.teacher = copy.deepcopy(model)  # EMA updated
        self.ema_decay = 0.999
    
    def update_teacher(self):
        for t_param, s_param in zip(self.teacher.parameters(), self.model.parameters()):
            t_param.data = self.ema_decay * t_param.data + (1 - self.ema_decay) * s_param.data
```

**Benefits:**
- Smaller model size
- Faster inference
- Maintain most of teacher's accuracy
- Regularization effect

**Interview Tips:**
- Temperature controls softness of distribution
- Higher temperature reveals more structure
- Can distill ensemble into single model

---

## Question 95
- [ ] Done

**How do you implement model pruning for efficient classification?**

### Answer

**Definition:**
Model pruning removes unnecessary weights/neurons from neural networks to reduce size and computation while maintaining accuracy.

**Pruning Types:**

| Type | What's Pruned |
|------|--------------|
| Weight pruning | Individual weights |
| Neuron/Filter pruning | Entire neurons/filters |
| Structured pruning | Regular patterns |
| Unstructured pruning | Any weights |

**Python Code Example:**
```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# 1. Magnitude-based pruning
model = YourModel()

# Prune 30% of weights in a layer
prune.l1_unstructured(model.fc1, name='weight', amount=0.3)

# Check sparsity
sparsity = 100 * float(torch.sum(model.fc1.weight == 0)) / model.fc1.weight.nelement()
print(f"Sparsity: {sparsity:.1f}%")

# Make pruning permanent
prune.remove(model.fc1, 'weight')

# 2. Global pruning (across all layers)
parameters_to_prune = [
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight'),
]

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.5  # 50% total
)

# 3. Iterative pruning
class IterativePruner:
    def __init__(self, model, target_sparsity=0.9, steps=10):
        self.model = model
        self.target = target_sparsity
        self.steps = steps
    
    def prune_step(self, current_step):
        # Gradual pruning: sparsity increases each step
        sparsity = self.target * (current_step / self.steps) ** 3
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, 'weight', amount=sparsity)
    
    def train_with_pruning(self, train_loader, epochs_per_step=5):
        for step in range(self.steps):
            # Prune
            self.prune_step(step)
            
            # Fine-tune
            for epoch in range(epochs_per_step):
                train_epoch(self.model, train_loader)
            
            print(f"Step {step}: Sparsity = {self.get_sparsity():.1f}%")

# 4. Structured pruning (filter pruning)
def prune_filters(model, layer_name, prune_ratio=0.3):
    module = dict(model.named_modules())[layer_name]
    
    # Compute filter importance (L1 norm)
    importance = module.weight.abs().sum(dim=(1, 2, 3))
    n_prune = int(len(importance) * prune_ratio)
    
    # Get indices to prune
    _, indices = torch.topk(importance, n_prune, largest=False)
    
    # Create mask
    mask = torch.ones(module.weight.shape[0])
    mask[indices] = 0
    
    prune.custom_from_mask(module, 'weight', mask.view(-1, 1, 1, 1))
```

**Interview Tips:**
- Unstructured pruning needs special hardware
- Structured pruning (filters) more practical
- Prune gradually with fine-tuning

---

## Question 96
- [ ] Done

**What is quantization and how does it help classification model efficiency?**

### Answer

**Definition:**
Quantization reduces precision of model weights and activations (e.g., 32-bit float to 8-bit int), decreasing model size and enabling faster inference.

**Types:**

| Type | Description |
|------|-------------|
| Post-training | Quantize after training |
| Quantization-aware | Simulate quantization during training |
| Dynamic | Quantize activations at runtime |
| Static | Pre-compute activation ranges |

**Benefits:**
- 4x smaller model (32-bit → 8-bit)
- Faster inference (int ops faster)
- Lower memory bandwidth
- Better for edge devices

**Python Code Example:**
```python
import torch
import torch.quantization as quant

# 1. Dynamic Quantization (easiest)
model_fp32 = YourModel()
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear, torch.nn.LSTM},
    dtype=torch.qint8
)

# Check size reduction
def model_size(model):
    torch.save(model.state_dict(), "temp.pt")
    size = os.path.getsize("temp.pt") / 1e6
    os.remove("temp.pt")
    return size

print(f"FP32 size: {model_size(model_fp32):.2f} MB")
print(f"INT8 size: {model_size(model_int8):.2f} MB")

# 2. Static Quantization
model_fp32.eval()

# Fuse layers (conv + bn + relu)
model_fp32 = torch.quantization.fuse_modules(
    model_fp32, [['conv', 'bn', 'relu']]
)

# Specify quantization config
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Prepare for calibration
model_prepared = torch.quantization.prepare(model_fp32)

# Calibrate with representative data
with torch.no_grad():
    for x, _ in calibration_loader:
        model_prepared(x)

# Convert to quantized
model_int8 = torch.quantization.convert(model_prepared)

# 3. Quantization-Aware Training
model = YourModel()
model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_qat = torch.quantization.prepare_qat(model)

# Train with fake quantization
for epoch in range(epochs):
    for x, y in train_loader:
        output = model_qat(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

# Convert to actual quantized model
model_qat.eval()
model_int8 = torch.quantization.convert(model_qat)
```

**Interview Tips:**
- Dynamic quantization simplest, minimal accuracy loss
- Static quantization faster but needs calibration
- QAT recovers accuracy, best for aggressive quantization

---

## Question 97
- [ ] Done

**What are the future directions and open problems in classification?**

### Answer

**Definition:**
Classification research continues evolving with new challenges and opportunities. Key areas include efficiency, robustness, fairness, and new paradigms.

**Open Problems:**

| Area | Challenge |
|------|-----------|
| Robustness | Adversarial examples, distribution shift |
| Efficiency | Edge deployment, green AI |
| Fairness | Bias mitigation, equitable outcomes |
| Data efficiency | Few-shot, zero-shot learning |
| Interpretability | Black-box explanations |
| Continual learning | Avoiding catastrophic forgetting |

**Emerging Directions:**

1. **Foundation Models for Classification**
```python
# Zero-shot with foundation models
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier(
    "The new smartphone has amazing battery life",
    candidate_labels=["technology", "sports", "politics", "entertainment"]
)
```

2. **Neurosymbolic Classification**
```python
# Combine neural networks with logical rules
class NeurosymbolicClassifier:
    def __init__(self, neural_model, rules):
        self.neural = neural_model
        self.rules = rules  # Logical constraints
    
    def predict(self, x):
        neural_pred = self.neural(x)
        # Apply logical rules to constrain predictions
        constrained_pred = self.apply_rules(neural_pred)
        return constrained_pred
```

3. **Causal Classification**
```python
# Use causal reasoning for robust classification
# Predict based on causal features, not spurious correlations
class CausalClassifier:
    def __init__(self, causal_graph, model):
        self.graph = causal_graph
        self.model = model
    
    def fit(self, X, y, treatment):
        # Learn causal effects
        self.causal_effects = self.estimate_effects(X, y, treatment)
```

4. **Privacy-Preserving Classification**
- Federated learning at scale
- Homomorphic encryption for inference
- Differential privacy by default

5. **Sustainable AI**
- Model efficiency (compute, energy)
- Smaller models with same capability
- Transfer learning over training from scratch

**Key Takeaways:**
- Classification fundamentals remain crucial
- Deep learning dominates but simpler models still valuable
- Practical considerations (fairness, privacy, efficiency) increasingly important
- Interdisciplinary: combines ML with ethics, law, domain expertise

**Interview Tips:**
- Show awareness of current research trends
- Balance academic knowledge with practical concerns
- Understand real-world deployment challenges

---

## Summary: Key Interview Preparation Tips

1. **Master the basics first**: Questions 1-20 cover fundamental concepts
2. **Know algorithm trade-offs**: When to use which algorithm
3. **Understand evaluation**: Precision, recall, F1, ROC-AUC
4. **Handle imbalanced data**: SMOTE, class weights, threshold tuning
5. **Practical coding**: Be able to implement basic classifiers from scratch
6. **Know sklearn API**: fit, predict, predict_proba, partial_fit
7. **Real-world considerations**: Scalability, interpretability, deployment
8. **Deep learning**: CNNs, RNNs, Transformers, training best practices
9. **MLOps**: Deployment, monitoring, A/B testing, versioning
10. **Ethics**: Fairness, privacy, robustness

## Question 98
- [ ] Done

**Discuss the differences between L1 and L2 regularization in the context of Logistic Regression.**

### Answer

**Definition:**
L1 (Lasso) and L2 (Ridge) regularization add penalty terms to the loss function to prevent overfitting. L1 adds absolute weight sum (promotes sparsity), while L2 adds squared weight sum (shrinks all weights evenly). In Logistic Regression, they control model complexity differently.

**Mathematical Formulation:**

| Regularization | Loss Function |
|----------------|---------------|
| No Regularization | $J = -\sum[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$ |
| L1 (Lasso) | $J + \lambda\sum|w_i|$ |
| L2 (Ridge) | $J + \lambda\sum w_i^2$ |
| Elastic Net | $J + \lambda_1\sum|w_i| + \lambda_2\sum w_i^2$ |

**Key Differences:**

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|------------|------------|
| Penalty | $\sum\|w_i\|$ | $\sum w_i^2$ |
| Effect | Sparse weights (some = 0) | Small weights (none = 0) |
| Feature Selection | Yes, automatic | No |
| Solution | May have multiple | Unique |
| When features correlated | Picks one randomly | Keeps all, shrinks equally |
| Computation | Harder (non-differentiable at 0) | Easier (smooth) |

**Visual Intuition:**
```
L1: Diamond-shaped constraint    L2: Circle-shaped constraint
    More likely to hit corners       Hits boundary smoothly
    → Sparse solution (w=0)          → All weights small but non-zero
```

**When to Use:**

| Scenario | Recommendation |
|----------|----------------|
| Many irrelevant features | L1 (feature selection) |
| All features important | L2 (keep all) |
| Highly correlated features | L2 or Elastic Net |
| Interpretability needed | L1 (fewer features) |
| Unsure | Elastic Net (combines both) |

**Python Code Example:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# Create dataset with some irrelevant features
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=5, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# L1 Regularization (Lasso)
l1_model = LogisticRegression(penalty='l1', solver='saga', C=0.1)
l1_model.fit(X_train, y_train)
print(f"L1 - Non-zero coefficients: {np.sum(l1_model.coef_ != 0)}")
print(f"L1 Accuracy: {l1_model.score(X_test, y_test):.3f}")

# L2 Regularization (Ridge)
l2_model = LogisticRegression(penalty='l2', C=0.1)
l2_model.fit(X_train, y_train)
print(f"L2 - Non-zero coefficients: {np.sum(l2_model.coef_ != 0)}")
print(f"L2 Accuracy: {l2_model.score(X_test, y_test):.3f}")

# Elastic Net (L1 + L2)
en_model = LogisticRegression(penalty='elasticnet', solver='saga', 
                              C=0.1, l1_ratio=0.5)
en_model.fit(X_train, y_train)
print(f"Elastic Net Accuracy: {en_model.score(X_test, y_test):.3f}")

# Compare coefficient magnitudes
print(f"\nL1 coefficients: {l1_model.coef_[0][:5].round(3)}")
print(f"L2 coefficients: {l2_model.coef_[0][:5].round(3)}")
```

**Interview Tips:**
- L1 = Lasso = Feature selection = Sparse
- L2 = Ridge = Shrinkage = Keeps all features
- C in sklearn is inverse of λ (smaller C = stronger regularization)
- Elastic Net is useful when you're unsure or have correlated features
- L1 is computationally harder due to non-differentiability at 0

---
