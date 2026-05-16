# Svm Interview Questions - General Questions

## Question 1
- [ ] Done

**What kind of kernels can be used in SVM and give examples of each?**

### Answer

**Definition:**
Kernels are functions that compute the similarity between data points in a transformed feature space without explicitly computing the transformation. They enable SVM to find non-linear decision boundaries.

**Common Kernels:**

| Kernel | Formula | Parameters | Best For |
|--------|---------|------------|----------|
| **Linear** | $K(x,y) = x^Ty$ | None | High-dim, text, linearly separable |
| **Polynomial** | $K(x,y) = (x^Ty + c)^d$ | degree d, coef c | Polynomial relationships |
| **RBF (Gaussian)** | $K(x,y) = \exp(-\gamma\|x-y\|^2)$ | gamma γ | Non-linear, most common |
| **Sigmoid** | $K(x,y) = \tanh(\alpha x^Ty + c)$ | alpha, coef | Neural network-like |

**Practical Usage in sklearn:**
```python
from sklearn.svm import SVC

# Linear
svm_linear = SVC(kernel='linear')

# Polynomial (degree 3)
svm_poly = SVC(kernel='poly', degree=3, coef0=1)

# RBF (default)
svm_rbf = SVC(kernel='rbf', gamma='scale')

# Sigmoid
svm_sig = SVC(kernel='sigmoid', coef0=0)
```

**Selection Guidelines:**
- Start with linear for high-dimensional or text data
- Try RBF for general non-linear problems
- Polynomial when you know the relationship degree
- RBF is the default and works well in most cases

---

## Question 2
- [ ] Done

**How do you choose the value of the regularization parameter C in SVM?**

### Answer

**Definition:**
C is the regularization parameter that controls the trade-off between maximizing the margin and minimizing classification errors. Higher C means stricter classification (narrow margin), lower C means more tolerance for errors (wider margin).

**Impact of C:**

| C Value | Effect | Consequence |
|---------|--------|-------------|
| **High C** | Narrow margin, fewer violations | Overfitting risk, memorizes noise |
| **Low C** | Wide margin, allows violations | Underfitting risk, too simple |
| **Optimal C** | Balance | Best generalization |

**How to Choose C:**

1. **Grid Search with Cross-Validation:**
```python
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X, y)
best_C = grid.best_params_['C']
```

2. **Logarithmic Scale:** Always search in powers of 10

3. **Learning Curves:** Plot train vs validation accuracy for different C values

**Practical Guidelines:**
- Default C=1.0 is reasonable starting point
- If underfitting: Increase C
- If overfitting: Decrease C
- Always tune C together with kernel parameters (gamma for RBF)

---

## Question 3
- [ ] Done

**Can you derive the optimization problem for the soft margin SVM?**

### Answer

**Starting Point:**
Hard margin SVM requires perfect separation, which fails for:
- Non-linearly separable data
- Noisy data with outliers

**Soft Margin Formulation:**

**Step 1: Introduce Slack Variables (ξᵢ)**
Allow points to violate margin with penalty:
- ξᵢ = 0: Point correctly classified beyond margin
- 0 < ξᵢ < 1: Point within margin but correctly classified
- ξᵢ ≥ 1: Point misclassified

**Step 2: Optimization Problem**
$$\min_{w,b,\xi} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\xi_i$$

Subject to:
- $y_i(w^Tx_i + b) \geq 1 - \xi_i$ for all i
- $\xi_i \geq 0$ for all i

**Step 3: Lagrangian**
$$L = \frac{1}{2}||w||^2 + C\sum\xi_i - \sum\alpha_i[y_i(w^Tx_i+b) - 1 + \xi_i] - \sum\mu_i\xi_i$$

**Step 4: KKT Conditions (set derivatives to zero)**
- $\frac{\partial L}{\partial w} = 0 \Rightarrow w = \sum\alpha_i y_i x_i$
- $\frac{\partial L}{\partial b} = 0 \Rightarrow \sum\alpha_i y_i = 0$
- $\frac{\partial L}{\partial \xi_i} = 0 \Rightarrow \alpha_i + \mu_i = C$

**Step 5: Dual Problem**
$$\max_\alpha \sum\alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j x_i^T x_j$$

Subject to: $0 \leq \alpha_i \leq C$ and $\sum\alpha_i y_i = 0$

**Key Insight:**
The constraint $\alpha_i \leq C$ (from soft margin) bounds the influence of any single point, making the model robust to outliers.

---

## Question 4
- [ ] Done

**How do you handle categorical variables when training an SVM?**

### Answer

**Challenge:**
SVM operates on numerical features using dot products/distances. Categorical variables must be converted to numerical format.

**Methods:**

| Method | Description | When to Use |
|--------|-------------|-------------|
| **One-Hot Encoding** | Create binary column per category | Nominal categories (no order) |
| **Label Encoding** | Assign integer to each category | Ordinal categories (with order) |
| **Target Encoding** | Replace with target mean | High-cardinality categories |
| **Binary Encoding** | Binary representation of integers | Many categories |

**One-Hot Encoding (Most Common for SVM):**
```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical_columns),
    ('num', StandardScaler(), numerical_columns)
])

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('svm', SVC(kernel='rbf'))
])
```

**Important Considerations:**
1. **Scale after encoding**: One-hot creates 0/1 values, scale numerical features
2. **High cardinality**: Many categories → many features → consider target encoding
3. **Drop first**: Remove one category to avoid multicollinearity
4. **Linear kernel**: Can handle high-dimensional one-hot better

**Interview Tip:**
Always mention that you need to scale features after encoding since SVM is distance-based.

---

## Question 5
- [ ] Done

**What methods can be used to tune SVM hyperparameters?**

### Answer

**Key Hyperparameters to Tune:**
- **C**: Regularization (all kernels)
- **gamma**: RBF/poly kernel width
- **degree**: Polynomial kernel degree
- **kernel**: Type of kernel

**Tuning Methods:**

| Method | Description | When to Use |
|--------|-------------|-------------|
| **Grid Search** | Exhaustive search over parameter grid | Small parameter space |
| **Random Search** | Random sampling from parameter distributions | Large parameter space |
| **Bayesian Optimization** | Probabilistic model guides search | Expensive evaluations |
| **Halving Grid Search** | Progressive elimination | Limited compute budget |

**Grid Search (Most Common):**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

**Random Search (For Large Spaces):**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_dist = {
    'C': loguniform(0.01, 100),
    'gamma': loguniform(0.001, 10)
}

random_search = RandomizedSearchCV(SVC(), param_dist, n_iter=50, cv=5)
```

**Best Practices:**
- Use logarithmic scale for C and gamma
- Start with coarse grid, then fine-tune
- Always use cross-validation (cv=5 or cv=10)
- Consider computational budget

---

## Question 6
- [ ] Done

**How do you deal with an imbalanced dataset when using SVM?**

### Answer

**Problem:**
When one class dominates (e.g., 95% vs 5%), SVM tends to classify everything as the majority class since it minimizes overall error.

**Solutions:**

| Method | Description | Implementation |
|--------|-------------|----------------|
| **Class Weights** | Penalize minority class errors more | `class_weight='balanced'` |
| **Oversampling** | Duplicate/synthesize minority samples | SMOTE |
| **Undersampling** | Remove majority samples | Random undersampling |
| **Threshold Adjustment** | Adjust decision threshold | Use `decision_function` |

**Method 1: Class Weights (Easiest)**
```python
from sklearn.svm import SVC

# Automatically balance classes
svm = SVC(kernel='rbf', class_weight='balanced')

# Or manual weights
svm = SVC(kernel='rbf', class_weight={0: 1, 1: 10})  # 10x penalty for class 1 errors
```

**Method 2: SMOTE (Synthetic Minority Oversampling)**
```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf'))
])
```

**Method 3: Threshold Adjustment**
```python
svm = SVC(kernel='rbf', probability=False)
svm.fit(X_train, y_train)

# Get decision scores
scores = svm.decision_function(X_test)

# Adjust threshold (default is 0)
threshold = -0.5  # Move threshold toward majority class
predictions = (scores > threshold).astype(int)
```

**Evaluation Metrics for Imbalanced Data:**
- Use Precision, Recall, F1-score (not accuracy)
- Use ROC-AUC or PR-AUC
- Confusion matrix to see class-wise performance

---

## Question 7
- [ ] Done

**What metrics are commonly used to evaluate the performance of an SVM model?**

### Answer

**Classification Metrics:**

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Accuracy** | $\frac{TP+TN}{Total}$ | Balanced classes |
| **Precision** | $\frac{TP}{TP+FP}$ | Cost of false positives high |
| **Recall** | $\frac{TP}{TP+FN}$ | Cost of false negatives high |
| **F1-Score** | $\frac{2 \times P \times R}{P+R}$ | Balance precision/recall |
| **ROC-AUC** | Area under ROC curve | Compare models, threshold selection |
| **PR-AUC** | Area under Precision-Recall | Imbalanced datasets |

**Implementation:**
```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report)

# Basic metrics
print(classification_report(y_test, y_pred))

# ROC-AUC (needs probability or decision_function)
svm = SVC(probability=True)  # Enables predict_proba
svm.fit(X_train, y_train)
proba = svm.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)

# Or use decision_function (faster)
scores = svm.decision_function(X_test)
auc = roc_auc_score(y_test, scores)
```

**Metric Selection Guide:**
| Scenario | Preferred Metric |
|----------|-----------------|
| Balanced binary classification | Accuracy, F1 |
| Imbalanced classes | F1, PR-AUC, Recall |
| Multi-class | Macro/Weighted F1 |
| Medical diagnosis (don't miss disease) | Recall |
| Spam detection (minimize false alarms) | Precision |
| Model comparison | ROC-AUC |

---

## Question 8
- [ ] Done

**How can you speed up SVM training on large datasets?**

### Answer

**Challenge:**
Standard SVM (SVC) has O(n²) to O(n³) complexity, making it impractical for large datasets (n > 10,000).

**Speed-up Strategies:**

| Strategy | Approach | Speedup | Trade-off |
|----------|----------|---------|-----------|
| **LinearSVC** | Optimized linear SVM | 10-100x | Linear kernel only |
| **SGDClassifier** | Stochastic gradient descent | 100x+ | Approximate |
| **Subsampling** | Train on data subset | Proportional | Accuracy loss |
| **Kernel Approximation** | Random Fourier Features | 10-50x | Approximate kernel |
| **Data Reduction** | Dimensionality reduction | Varies | Information loss |

**Method 1: LinearSVC (Linear Kernel)**
```python
from sklearn.svm import LinearSVC
# Uses liblinear, much faster than SVC(kernel='linear')
svm = LinearSVC(C=1.0, max_iter=10000)
```

**Method 2: SGDClassifier (Scalable)**
```python
from sklearn.linear_model import SGDClassifier
# Hinge loss = SVM
svm = SGDClassifier(loss='hinge', max_iter=1000)
# Supports partial_fit for streaming data
```

**Method 3: Kernel Approximation**
```python
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

# Approximate RBF kernel
rbf_feature = RBFSampler(n_components=100, random_state=42)
X_transformed = rbf_feature.fit_transform(X)

# Use fast linear classifier
svm = SGDClassifier(loss='hinge')
svm.fit(X_transformed, y)
```

**Decision Guide:**
| Dataset Size | Recommended Approach |
|-------------|---------------------|
| n < 10,000 | SVC (any kernel) |
| 10,000 < n < 100,000 | LinearSVC or kernel approximation |
| n > 100,000 | SGDClassifier |

---

## Question 9
- [ ] Done

**What steps would you take to diagnose and solve underfitting or overfitting in an SVM model?**

### Answer

**Diagnosis:**

| Symptom | Training Acc | Validation Acc | Problem |
|---------|-------------|----------------|---------|
| High gap | High | Low | Overfitting |
| Both low | Low | Low | Underfitting |
| Both high | High | High | Good fit |

**Diagnosing Overfitting:**
- Training accuracy >> Validation accuracy
- Model memorizes noise
- Too many support vectors

**Diagnosing Underfitting:**
- Both training and validation accuracy are low
- Model too simple for data complexity
- Linear kernel on non-linear data

**Solutions:**

| Problem | Parameter Adjustment | Other Actions |
|---------|---------------------|---------------|
| **Overfitting** | Decrease C, Decrease gamma | More data, feature selection, simpler kernel |
| **Underfitting** | Increase C, Increase gamma | More features, complex kernel (RBF), polynomial features |

**Systematic Debugging Steps:**

1. **Check learning curves:**
```python
from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(
    SVC(), X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)
# Plot to visualize bias-variance trade-off
```

2. **Tune hyperparameters:**
```python
# If overfitting: reduce C and gamma
param_grid = {'C': [0.01, 0.1, 1], 'gamma': [0.001, 0.01, 0.1]}

# If underfitting: increase C and gamma
param_grid = {'C': [10, 100, 1000], 'gamma': [1, 10, 100]}
```

3. **Try different kernels:**
- Underfitting with linear → Try RBF
- Overfitting with RBF → Try linear

4. **Feature engineering:**
- Underfitting: Add polynomial features
- Overfitting: Feature selection, PCA

---

## Question 10
- [ ] Done

**What considerations should be taken into account for deploying an SVM model in production?**

### Answer

**Key Considerations:**

| Category | Consideration | Solution |
|----------|--------------|----------|
| **Scalability** | Inference time grows with support vectors | Reduce SVs, use LinearSVC |
| **Memory** | Store all support vectors | Model compression |
| **Preprocessing** | Same scaling as training | Save scaler, use Pipeline |
| **Latency** | Real-time requirements | Kernel approximation |
| **Monitoring** | Data drift detection | Track input distributions |

**Production Checklist:**

1. **Save Complete Pipeline:**
```python
from sklearn.pipeline import Pipeline
import joblib

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf'))
])
pipeline.fit(X_train, y_train)

# Save entire pipeline
joblib.dump(pipeline, 'svm_model.joblib')

# Load in production
loaded_pipeline = joblib.load('svm_model.joblib')
predictions = loaded_pipeline.predict(new_data)
```

2. **Input Validation:**
- Check feature count matches training
- Handle missing values
- Validate data types

3. **Performance Optimization:**
```python
# If inference is slow:
# Option 1: Use LinearSVC (faster)
# Option 2: Kernel approximation
# Option 3: Reduce support vectors (increase C)
```

4. **Monitoring:**
- Log predictions and confidence scores
- Detect data drift (input distribution changes)
- Track accuracy on new labeled data
- Set up alerts for anomalies

5. **Versioning:**
- Version control model files
- Track hyperparameters
- Document training data version

---

## Question 11
- [ ] Done

**In what scenarios would you use a polynomial kernel?**

### Answer

**Definition:**
Polynomial kernel: $K(x,y) = (x^Ty + c)^d$

Parameters: degree (d), coef0 (c)

**When to Use Polynomial Kernel:**

| Scenario | Reasoning |
|----------|-----------|
| Known polynomial relationship | Feature interactions of specific degree |
| Image classification | Capture edge/corner features |
| Moderate non-linearity | When linear is too simple, RBF too complex |
| NLP with word co-occurrence | Polynomial captures feature combinations |

**Polynomial Kernel Characteristics:**
- degree=1: Equivalent to linear kernel (with coef0)
- degree=2: Captures pairwise feature interactions
- degree=3+: Higher-order interactions, overfitting risk

**Comparison with RBF:**
| Aspect | Polynomial | RBF |
|--------|------------|-----|
| Parameters | degree, coef0 | gamma |
| Interpretability | Moderate (explicit degree) | Low |
| Flexibility | Controlled by degree | Very flexible |
| Overfitting risk | Lower (bounded complexity) | Higher |

**Implementation:**
```python
from sklearn.svm import SVC

# Quadratic kernel (degree 2)
svm_poly2 = SVC(kernel='poly', degree=2, coef0=1)

# Cubic kernel (degree 3)
svm_poly3 = SVC(kernel='poly', degree=3, coef0=0, gamma='scale')
```

**Practical Tip:**
- Start with degree=2 or 3
- Higher degrees often overfit
- Tune coef0 via cross-validation
- RBF often works better in practice

---

## Question 12
- [ ] Done

**How can SVM be used for sentiment analysis on social media data?**

### Answer

**Pipeline:**

1. **Data Collection:** Gather tweets/posts with sentiment labels
2. **Text Preprocessing:** Clean, tokenize, normalize
3. **Feature Extraction:** TF-IDF or embeddings
4. **Model Training:** Linear SVM (works best for text)
5. **Evaluation:** F1-score for imbalanced sentiments

**Implementation:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Sample social media data
texts = [
    "Love this product! Amazing quality",
    "Terrible service, never buying again",
    "It's okay, nothing special",
    # ... more data
]
labels = [1, 0, 0]  # 1=positive, 0=negative

# Build pipeline
sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,
        stop_words='english'
    )),
    ('svm', LinearSVC(C=1.0, class_weight='balanced'))
])

# Train
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
sentiment_pipeline.fit(X_train, y_train)

# Predict
new_tweets = ["This is absolutely fantastic!", "Worst experience ever"]
predictions = sentiment_pipeline.predict(new_tweets)
```

**Social Media Specific Considerations:**
- Handle emojis, slang, hashtags
- Remove URLs, mentions
- Handle class imbalance (more neutral often)
- Multi-class: positive/negative/neutral

**Why Linear SVM for Text:**
- High-dimensional sparse data (TF-IDF)
- Fast training and inference
- Feature weights indicate important words
- Competitive with deep learning on smaller datasets

---

## Question 13
- [ ] Done

**How can deep learning techniques be integrated with SVMs?**

### Answer

**Integration Approaches:**

| Approach | Description | Use Case |
|----------|-------------|----------|
| **CNN + SVM** | Use CNN as feature extractor, SVM as classifier | Image classification |
| **Deep Kernel Learning** | Learn kernel function with neural network | When kernels need to be adaptive |
| **SVM Loss in NN** | Use hinge loss for neural network training | Margin-based neural networks |
| **Embedding + SVM** | Use pretrained embeddings (BERT, Word2Vec) | NLP tasks |

**CNN Features + SVM (Most Common):**
```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from sklearn.svm import SVC

# Load pretrained CNN
base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Extract features
features = base_model.predict(images)

# Train SVM on CNN features
svm = SVC(kernel='rbf', C=10)
svm.fit(features_train, labels_train)
```

**BERT Embeddings + SVM:**
```python
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC

# Get BERT embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)

# Train SVM on embeddings
svm = LinearSVC(C=1.0)
svm.fit(embeddings_train, labels_train)
```

**When to Use This Hybrid:**
- Small labeled dataset (transfer learning helps)
- Need interpretability (SVM weights)
- Computational constraints (pretrained features + light SVM)
- Combining deep features with structured data

---

## Question 14
- [ ] Done

**How can domain adaptation be achieved using SVM models for transfer learning?**

### Answer

**Problem:**
Source domain (training data) differs from target domain (deployment), causing performance drop.

**Domain Adaptation Strategies for SVM:**

| Strategy | Description | Complexity |
|----------|-------------|------------|
| **Feature-based** | Find domain-invariant features | Medium |
| **Instance-based** | Reweight source samples | Low |
| **Parameter-based** | Adapt model parameters | Medium |

**Method 1: Instance Reweighting**
Assign higher weights to source samples similar to target domain:
```python
from sklearn.svm import SVC
import numpy as np

# Compute similarity weights (example: using density ratio)
weights = compute_domain_weights(X_source, X_target)

# Use sample_weight in training
svm = SVC(kernel='rbf')
svm.fit(X_source, y_source, sample_weight=weights)
```

**Method 2: Transfer Component Analysis (TCA)**
Project both domains to shared subspace:
```python
# Pseudo-code for TCA approach
# 1. Compute MMD (Maximum Mean Discrepancy) kernel
# 2. Find projection that minimizes domain difference
# 3. Train SVM on projected features
```

**Method 3: Fine-tuning with Target Labels**
If some target labels available:
```python
# Train on source
svm.fit(X_source, y_source)

# Get predictions on target
pseudo_labels = svm.predict(X_target_unlabeled)

# Combine and retrain (self-training)
X_combined = np.vstack([X_source, X_target_labeled, X_target_unlabeled])
y_combined = np.hstack([y_source, y_target_labeled, pseudo_labels])
svm.fit(X_combined, y_combined)
```

**Practical Tips:**
- Start with feature normalization across domains
- Use domain-invariant features when possible
- Deep learning features often transfer better

---

## Question 15
- [ ] Done

**How is the research on quantum machine learning potentially impacting SVM algorithms?**

### Answer

**Quantum SVM Concepts:**

Quantum computing promises exponential speedup for certain SVM operations by leveraging quantum superposition and entanglement.

**Key Quantum SVM Approaches:**

| Approach | Description | Potential Benefit |
|----------|-------------|-------------------|
| **Quantum Kernel Estimation** | Compute kernels on quantum hardware | Handle exponentially large feature spaces |
| **Quantum Feature Maps** | Map data to quantum state space | Access quantum feature spaces |
| **Variational Quantum SVM** | Hybrid classical-quantum optimization | Near-term quantum devices |

**How Quantum Helps:**
1. **Kernel Computation:** Quantum computers can compute certain kernels exponentially faster
2. **High-dimensional Data:** Natural handling of exponential feature spaces
3. **Optimization:** Potential speedup in solving SVM's quadratic program

**Current State (2025):**
- Proof-of-concept demonstrations on quantum simulators
- Limited by current quantum hardware (noise, qubit count)
- Practical advantage not yet demonstrated for real-world problems
- Active research area

**Implications:**
- Future: Quantum-enhanced SVM for specific problems
- Near-term: Hybrid classical-quantum approaches
- May enable new kernel functions not efficiently computable classically

**Interview Perspective:**
Mention awareness of quantum ML research but acknowledge it's still largely theoretical for practical applications.

---

## Question 16
- [ ] Done

**How can SVM be combined with other machine learning models to form an ensemble?**

### Answer

**Ensemble Strategies with SVM:**

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Bagging** | Train multiple SVMs on bootstrap samples | Reduce variance |
| **Boosting** | Sequentially train, focus on hard examples | Improve accuracy |
| **Stacking** | SVM as base learner or meta-learner | Combine diverse models |
| **Voting** | Combine SVM with other classifiers | Different model strengths |

**Method 1: Bagging (BaggingClassifier)**
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

bagged_svm = BaggingClassifier(
    estimator=SVC(kernel='rbf'),
    n_estimators=10,
    max_samples=0.8,
    bootstrap=True
)
bagged_svm.fit(X_train, y_train)
```

**Method 2: Voting Ensemble**
```python
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ensemble = VotingClassifier(
    estimators=[
        ('svm', SVC(kernel='rbf', probability=True)),
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('lr', LogisticRegression())
    ],
    voting='soft'  # Use probabilities
)
ensemble.fit(X_train, y_train)
```

**Method 3: Stacking (SVM as Meta-Learner)**
```python
from sklearn.ensemble import StackingClassifier

stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('knn', KNeighborsClassifier())
    ],
    final_estimator=SVC(kernel='linear'),  # SVM combines predictions
    cv=5
)
```

**When to Ensemble SVM:**
- Single SVM performance plateaus
- Need uncertainty estimates
- Combine SVM's margin-based decisions with other approaches
- Reduce variance in predictions

**Practical Considerations:**
- Ensembling increases computation
- SVM in ensemble should use `probability=True` for soft voting
- Diverse base learners work better
