# Data Scientist Interview Questions - Theory Questions

## Table of Contents

### 🎯 Machine Learning Fundamentals (Q1-Q10)
- [Q1: What is Machine Learning?](#question-1)
- [Q2: Supervised vs Unsupervised Learning](#question-2)
- [Q3: Classification vs Regression](#question-3)
- [Q4: Overfitting and Underfitting](#question-4)
- [Q5: Bias-Variance Tradeoff](#question-5)
- [Q6: Cross-Validation](#question-6)
- [Q7: Regularization](#question-7)
- [Q8: Parametric vs Non-Parametric Models](#question-8)
- [Q9: Curse of Dimensionality](#question-9)
- [Q10: Feature Engineering](#question-10)

### 🔧 Data Preprocessing & Feature Engineering (Q11-Q20)
- [Q11: Data Preprocessing](#question-11)
- [Q12: Feature Scaling vs Normalization](#question-12)
- [Q13: One-Hot Encoding](#question-13)
- [Q14: Handling Missing Values](#question-14)
- [Q15: Feature Selection](#question-15)
- [Q16: Filter, Wrapper, Embedded Methods](#question-16)
- [Q17: PCA for Dimensionality Reduction](#question-17)
- [Q18: Outlier Detection](#question-18)
- [Q19: Imputer Class in Scikit-learn](#question-19)
- [Q20: Handling Imbalanced Datasets](#question-20)

### 🤖 Supervised Learning Algorithms (Q21-Q29)
- [Q21: Linear Regression](#question-21)
- [Q22: Logistic Regression](#question-22)
- [Q23: Decision Tree](#question-23)
- [Q24: Random Forest](#question-24)
- [Q25: Support Vector Machine (SVM)](#question-25)
- [Q26: Naive Bayes](#question-26)
- [Q27: K-Nearest Neighbors (KNN)](#question-27)
- [Q28: XGBoost](#question-28)
- [Q29: Stacking in Ensemble Learning](#question-29)

### 🔍 Unsupervised Learning (Q30-Q39)
- [Q30: K-Means Clustering](#question-30)
- [Q31: Hierarchical vs Partitional Clustering](#question-31)
- [Q32: DBSCAN](#question-32)
- [Q33: Gaussian Mixture Models (GMM)](#question-33)
- [Q34: PCA in Unsupervised Learning](#question-34)
- [Q35: t-SNE](#question-35)
- [Q36: Association Rule Mining](#question-36)
- [Q37: Anomaly Detection](#question-37)
- [Q38: Self-Organizing Maps (SOM)](#question-38)
- [Q39: Latent Dirichlet Allocation (LDA)](#question-39)

### 📊 Model Evaluation & Validation (Q40-Q49)
- [Q40: Model Evaluation Purpose](#question-40)
- [Q41: Train, Validation, Test Sets](#question-41)
- [Q42: Confusion Matrix](#question-42)
- [Q43: Precision, Recall, F1-Score](#question-43)
- [Q44: ROC Curve](#question-44)
- [Q45: AUC](#question-45)
- [Q46: Mean Squared Error (MSE)](#question-46)
- [Q47: R-squared](#question-47)
- [Q48: K-Fold Cross-Validation](#question-48)
- [Q49: Stratified K-Fold](#question-49)

### 🧠 Neural Networks & Deep Learning (Q50-Q59)
- [Q50: Neural Network Components](#question-50)
- [Q51: Feedforward vs Recurrent NN](#question-51)
- [Q52: Backpropagation](#question-52)
- [Q53: Activation Functions](#question-53)
- [Q54: Deep Learning Applications](#question-54)
- [Q55: Convolutional Neural Networks (CNN)](#question-55)
- [Q56: RNN, LSTM, GRU](#question-56)
- [Q57: Autoencoders](#question-57)
- [Q58: Transfer Learning](#question-58)
- [Q59: Generative Adversarial Networks (GAN)](#question-59)

### 📝 Natural Language Processing (Q60-Q69)
- [Q60: NLP Applications](#question-60)
- [Q61: Tokenization vs Stemming](#question-61)
- [Q62: Word Embeddings](#question-62)
- [Q63: Named Entity Recognition (NER)](#question-63)
- [Q64: Sentiment Analysis](#question-64)
- [Q65: Topic Modeling](#question-65)
- [Q66: Text Classification](#question-66)
- [Q67: Language Translation](#question-67)
- [Q68: Text Summarization](#question-68)
- [Q69: Chatbots](#question-69)

### 🎬 Recommender Systems (Q70-Q79)
- [Q70: Recommender System Types](#question-70)
- [Q71: Content-Based vs Collaborative Filtering](#question-71)
- [Q72: Matrix Factorization](#question-72)
- [Q73: Cold Start Problem](#question-73)
- [Q74: Evaluation Metrics for RecSys](#question-74)
- [Q75: Hybrid Recommender Systems](#question-75)
- [Q76: ALS Algorithm](#question-76)
- [Q77: Implicit Feedback](#question-77)
- [Q78: SVD in RecSys](#question-78)
- [Q79: Diversity and Serendipity](#question-79)

### 🎮 Reinforcement Learning (Q80-Q89)
- [Q80: RL Components](#question-80)
- [Q81: Exploitation vs Exploration](#question-81)
- [Q82: Markov Decision Process (MDP)](#question-82)
- [Q83: Q-Learning](#question-83)
- [Q84: Deep Q-Networks (DQN)](#question-84)
- [Q85: Policy Gradient Methods](#question-85)
- [Q86: Actor-Critic Methods](#question-86)
- [Q87: Monte Carlo Tree Search (MCTS)](#question-87)
- [Q88: Bellman Equation](#question-88)
- [Q89: Inverse Reinforcement Learning](#question-89)

### ⚙️ Optimization & Hyperparameter Tuning (Q90-Q99)
- [Q90: Optimization Types](#question-90)
- [Q91: GD vs SGD](#question-91)
- [Q92: Learning Rate](#question-92)
- [Q93: Momentum](#question-93)
- [Q94: Hyperparameter Tuning](#question-94)
- [Q95: Grid Search](#question-95)
- [Q96: Random Search](#question-96)
- [Q97: Bayesian Optimization](#question-97)
- [Q98: Early Stopping](#question-98)
- [Q99: Learning Rate Scheduling](#question-99)

---

## Question 1
- [ ] Done

**What is Machine Learning and how does it differ from traditional programming?**

**Answer:**

### Definition
Machine Learning is a subset of Artificial Intelligence that enables computers to learn patterns from data and make predictions or decisions without being explicitly programmed for each specific task.

### Key Difference

| Aspect | Traditional Programming | Machine Learning |
|--------|------------------------|------------------|
| Approach | Input + Rules → Output | Input + Output → Learned Rules |
| Logic | Explicitly coded by developers | Discovered from data |
| Adaptability | Requires manual updates | Learns and adapts automatically |
| Complexity | Struggles with complex patterns | Handles complex, non-linear patterns |
| Example | If-else spam rules | Spam classifier learning from examples |

### Real-World Example
**Email Spam Detection:**
- **Traditional**: Write 1000s of rules like "if contains 'FREE MONEY' → spam"
- **ML**: Feed 10,000 labeled emails, algorithm learns what makes spam

### When to Use ML
- Rules are too complex to define (image recognition)
- Rules change frequently (fraud detection)
- Problem requires personalization (recommendations)
- Data is available but patterns are unknown

### Common Follow-up Questions
1. *"What types of problems are NOT suitable for ML?"* - Small datasets, need for 100% explainability, simple rule-based logic
2. *"How do you decide between ML and traditional programming?"* - Consider data availability, problem complexity, interpretability needs

### Interview Tip
Always mention that ML excels when rules are complex, unknown, or constantly changing. Give a concrete example from your experience.

---

## Question 2
- [ ] Done

**Explain the difference between Supervised Learning and Unsupervised Learning.**

**Answer:**

### Comprehensive Comparison

| Aspect | Supervised Learning | Unsupervised Learning |
|--------|--------------------|-----------------------|
| **Data** | Labeled (X, y pairs) | Unlabeled (X only) |
| **Goal** | Predict known outcomes | Discover hidden patterns |
| **Feedback** | Direct feedback via labels | No feedback |
| **Algorithms** | Linear Regression, SVM, Random Forest | K-Means, PCA, DBSCAN |
| **Evaluation** | Clear metrics (accuracy, MSE) | Harder to evaluate |
| **Cost** | Expensive (labeling required) | Cheaper (no labeling) |

### Supervised Learning Deep Dive

**Types:**
- **Classification**: Predict discrete categories (spam/not spam, disease/healthy)
- **Regression**: Predict continuous values (house price, temperature)

**How It Works:**
1. Collect labeled training data
2. Algorithm learns mapping: f(X) → y
3. Apply learned model to new data

**Example**: Predicting customer churn
- Features (X): usage patterns, tenure, complaints
- Label (y): churned (1) or retained (0)

### Unsupervised Learning Deep Dive

**Types:**
- **Clustering**: Group similar items (customer segments)
- **Dimensionality Reduction**: Compress features (PCA)
- **Association**: Find relationships (market basket analysis)
- **Anomaly Detection**: Find outliers (fraud detection)

**How It Works:**
1. Feed unlabeled data
2. Algorithm finds structure/patterns
3. Interpret discovered patterns

**Example**: Customer segmentation
- Input: Purchase history, demographics
- Output: 5 distinct customer groups (discovered, not predefined)

### Semi-Supervised Learning (Bonus)
Combines small labeled dataset with large unlabeled dataset. Useful when labeling is expensive.

### Common Follow-up Questions
1. *"When would you use unsupervised before supervised?"* - For feature engineering, data exploration, or when labels are scarce
2. *"What's self-supervised learning?"* - Creates labels from data itself (e.g., predicting next word)

### Interview Tip
Mention that real-world projects often combine both: use unsupervised for exploration/feature engineering, then supervised for prediction. Give an example of when labeling is too expensive.

---

## Question 3
- [ ] Done

**What is the difference between Classification and Regression problems?**

**Answer:**

### Comprehensive Comparison

| Aspect | Classification | Regression |
|--------|---------------|------------|
| **Output Type** | Discrete categories | Continuous values |
| **Prediction** | Class labels | Numeric quantities |
| **Examples** | Spam/Not Spam, Cat/Dog | Price, Temperature, Sales |
| **Algorithms** | Logistic Regression, SVM, RF | Linear Regression, XGBoost |
| **Loss Function** | Cross-entropy, Hinge loss | MSE, MAE, Huber |
| **Metrics** | Accuracy, Precision, Recall, F1, AUC | MSE, RMSE, MAE, R² |

### Classification Types

**Binary Classification:**
- Two classes: Yes/No, True/False
- Examples: Fraud detection, disease diagnosis
- Metrics: Accuracy, Precision, Recall, F1, AUC-ROC

**Multi-class Classification:**
- More than two mutually exclusive classes
- Examples: Digit recognition (0-9), animal species
- Metrics: Macro/Micro F1, Confusion matrix

**Multi-label Classification:**
- Multiple labels per instance
- Examples: Movie genres, article tags
- Metrics: Hamming loss, Subset accuracy

### Regression Types

**Simple Regression**: One feature → one target
**Multiple Regression**: Many features → one target
**Multivariate Regression**: Many features → multiple targets

### Converting Between Them

**Regression → Classification (Binning):**
```
Age (continuous) → Age Group (Young/Middle/Senior)
Salary → Income Bracket (Low/Medium/High)
```

**Classification → Regression:**
```
Predict probability instead of class (Logistic Regression outputs 0.73)
```

### Tricky Cases

| Problem | Seems Like | Actually Is |
|---------|-----------|-------------|
| Predict rating (1-5) | Regression | Often treated as classification (ordinal) |
| Predict age | Regression | Could be classification (age groups) |
| Predict click probability | Regression | Binary classification with probability output |

### Common Follow-up Questions
1. *"When would you treat a regression problem as classification?"* - When exact value doesn't matter, only ranges/categories
2. *"Can the same algorithm do both?"* - Yes! Decision Trees, Neural Networks, XGBoost can do both

### Interview Tip
Always clarify the business objective. "Predict house price" (regression) vs "Is house overpriced?" (classification) require different approaches. Ask about what decisions will be made with predictions.

---

## Question 4
- [ ] Done

**Describe the concept of Overfitting and Underfitting in ML models.**

**Answer:**

### Visual Understanding
```
Model Complexity →
     Low                    Optimal                   High
      |                        |                        |
 Underfitting              Just Right              Overfitting
 (High Bias)                                    (High Variance)
      |                        |                        |
  Too Simple            Generalizes Well        Too Complex
```

### Detailed Comparison

| Aspect | Underfitting | Good Fit | Overfitting |
|--------|-------------|----------|-------------|
| **Training Error** | High | Low | Very Low |
| **Test Error** | High | Low | High |
| **Gap (Train-Test)** | Small | Small | Large |
| **Model Complexity** | Too simple | Appropriate | Too complex |
| **Bias** | High | Low | Low |
| **Variance** | Low | Low | High |

### Overfitting Deep Dive

**What Happens:**
- Model memorizes training data including noise
- Learns patterns that don't generalize
- Perfect on training, fails on new data

**Signs of Overfitting:**
- Training accuracy: 99%, Test accuracy: 70%
- Complex decision boundaries
- Model changes drastically with small data changes

**Solutions:**
1. **More training data** - Harder to memorize
2. **Regularization** - L1/L2 penalties, Dropout
3. **Simpler model** - Fewer parameters, less depth
4. **Early stopping** - Stop before memorization
5. **Cross-validation** - Detect overfitting early
6. **Feature selection** - Remove noisy features
7. **Data augmentation** - Artificially increase data
8. **Ensemble methods** - Combine multiple models

### Underfitting Deep Dive

**What Happens:**
- Model too simple to capture patterns
- Misses important relationships in data
- Poor performance everywhere

**Signs of Underfitting:**
- Training accuracy: 60%, Test accuracy: 58%
- Linear model for non-linear data
- Important features have low importance

**Solutions:**
1. **More complex model** - More layers, higher degree
2. **More features** - Feature engineering
3. **Less regularization** - Reduce penalty strength
4. **Train longer** - More epochs/iterations
5. **Remove noise** - Clean data better

### Practical Detection

```python
# Learning Curves Analysis
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5
)

# If train_score >> test_score → Overfitting
# If both scores are low → Underfitting
# If both scores are high and close → Good fit
```

### Real-World Example
**Predicting House Prices:**
- **Underfitting**: Using only square footage (ignores location, bedrooms)
- **Overfitting**: Model memorizes each house's exact price, fails on new houses
- **Good fit**: Captures key patterns (size, location, condition) without memorizing

### Common Follow-up Questions
1. *"How do you detect overfitting in production?"* - Monitor model drift, compare training vs production metrics
2. *"What's the relationship to bias-variance tradeoff?"* - Underfitting=high bias, Overfitting=high variance

### Interview Tip
Always mention learning curves as a diagnostic tool. Explain that you'd plot training vs validation error across training sizes to identify the problem. Be ready to discuss specific regularization techniques you've used.

---

## Question 5
- [ ] Done

**What is the Bias-Variance Tradeoff in ML?**

**Answer:**

### The Fundamental Equation
$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

### Understanding Each Component

**Bias (Systematic Error):**
- Error from overly simplistic assumptions
- Model's tendency to consistently miss the target
- **High bias** = Underfitting
- Example: Fitting a line to curved data

**Variance (Sensitivity Error):**
- Error from sensitivity to small fluctuations in training data
- Model changes significantly with different training sets
- **High variance** = Overfitting
- Example: Complex model that fits training noise

**Irreducible Error:**
- Noise inherent in the problem
- Cannot be reduced by any model
- Comes from unmeasured variables, measurement error

### Visual Analogy: Target Shooting

```
                Low Variance          High Variance
              (Consistent shots)    (Scattered shots)
                     
Low Bias         ●●●                    ●  ●
(On Target)       ◎                   ● ◎ ●
                 ●●●                   ●   ●

High Bias        ●●●                   ●  ●
(Off Target)      ○  ◎                ○ ●◎ ●
                 ●●●                   ●   ●

◎ = Bullseye (True Value)  ● = Shots (Predictions)  ○ = Shot Cluster Center
```

### The Tradeoff in Practice

| Model Type | Bias | Variance | Example |
|------------|------|----------|----------|
| Very Simple | High | Low | Linear Regression on complex data |
| Moderate | Medium | Medium | Regularized models |
| Very Complex | Low | High | Deep neural network on small data |

### Algorithm Examples

| Algorithm | Bias | Variance | Notes |
|-----------|------|----------|-------|
| Linear Regression | High | Low | Strong assumptions |
| Decision Tree (deep) | Low | High | Fits training exactly |
| Random Forest | Low | Low | Averaging reduces variance |
| k-NN (k=1) | Low | High | Very sensitive to training data |
| k-NN (k=n) | High | Low | Always predicts mean |

### Managing the Tradeoff

**To Reduce Bias:**
- Use more complex models
- Add more features
- Reduce regularization
- Use boosting (sequentially reduces bias)

**To Reduce Variance:**
- Use simpler models
- Add more training data
- Increase regularization
- Use bagging (averaging reduces variance)
- Feature selection

### Mathematical Intuition

For a model $\hat{f}(x)$ predicting true function $f(x)$:

$$\text{Bias}[\hat{f}(x)] = E[\hat{f}(x)] - f(x)$$

$$\text{Variance}[\hat{f}(x)] = E[(\hat{f}(x) - E[\hat{f}(x)])^2]$$

### Practical Example

**Polynomial Regression:**
- Degree 1 (linear): High bias, low variance
- Degree 3: Moderate bias, moderate variance ✓
- Degree 15: Low bias, high variance

### Common Follow-up Questions
1. *"How does ensemble learning address this tradeoff?"* - Bagging reduces variance, boosting reduces bias
2. *"What's the relationship to regularization?"* - Regularization increases bias slightly to reduce variance significantly
3. *"How do you find the optimal complexity?"* - Cross-validation to find the sweet spot

### Interview Tip
Explain that the goal is NOT zero bias or zero variance, but minimizing total error. Mention that modern deep learning with proper regularization can achieve both low bias and low variance. Use the dartboard analogy to make it intuitive.

---

## Question 6
- [ ] Done

**Explain the concept of Cross-Validation and its importance in ML.**

**Answer:**

### Definition
Cross-validation is a resampling technique that evaluates model performance by training and testing on different subsets of data, providing a more reliable estimate of how the model will perform on unseen data.

### Why Cross-Validation Matters

| Problem with Single Split | How CV Solves It |
|--------------------------|------------------|
| Lucky/unlucky test set | Averages over multiple test sets |
| Wastes data | Uses all data for both training and testing |
| High variance in estimates | Provides confidence intervals |
| Can't tune + evaluate properly | Nested CV separates concerns |

### Types of Cross-Validation

**1. K-Fold Cross-Validation (Most Common)**
```
Fold 1: [Test] [Train] [Train] [Train] [Train]
Fold 2: [Train] [Test] [Train] [Train] [Train]
Fold 3: [Train] [Train] [Test] [Train] [Train]
Fold 4: [Train] [Train] [Train] [Test] [Train]
Fold 5: [Train] [Train] [Train] [Train] [Test]

Final Score = Average of 5 scores
```

**2. Stratified K-Fold (For Imbalanced Data)**
- Preserves class distribution in each fold
- Essential for classification with imbalanced classes

**3. Leave-One-Out (LOO)**
- K = N (each sample is test set once)
- Maximum data for training, but computationally expensive
- Low bias, high variance

**4. Time Series Split**
```
Fold 1: [Train] | [Test]
Fold 2: [Train Train] | [Test]
Fold 3: [Train Train Train] | [Test]
```
- Never uses future data to predict past
- Essential for temporal data

**5. Group K-Fold**
- Ensures same group (e.g., same patient) doesn't appear in both train and test
- Prevents data leakage

### Implementation

```python
from sklearn.model_selection import (
    cross_val_score, KFold, StratifiedKFold, TimeSeriesSplit
)

# Basic K-Fold
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

# Stratified K-Fold for classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)

# Time Series Split
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])
```

### Nested Cross-Validation (For Hyperparameter Tuning)

```
Outer Loop (Model Evaluation):
├─ Fold 1: [Test] ── Inner Loop (Hyperparameter Tuning)
│                    ├─ Inner Fold 1: Find best params
│                    ├─ Inner Fold 2: ...
│                    └─ Train with best params, evaluate on outer test
├─ Fold 2: ...
└─ Final: Unbiased performance estimate
```

### Choosing K

| K Value | Pros | Cons | When to Use |
|---------|------|------|-------------|
| K=5 | Good balance | - | Default choice |
| K=10 | Lower bias | More computation | Medium datasets |
| K=N (LOO) | Lowest bias | Highest variance, slow | Very small datasets |
| K=3 | Fast | Higher bias | Very large datasets |

### Common Mistakes to Avoid

1. **Data Leakage**: Preprocessing before splitting
   - ✘ `scaler.fit(X)` then split
   - ✔ `scaler.fit(X_train)` in each fold

2. **Using test set for decisions**: Should use validation set

3. **Not stratifying**: For imbalanced classification

4. **Ignoring temporal order**: For time series data

### Common Follow-up Questions
1. *"When would you NOT use cross-validation?"* - Very large datasets (single split is reliable), time constraints
2. *"How do you handle data leakage in CV?"* - Use pipelines, fit preprocessing only on training fold
3. *"What's the difference between validation and cross-validation?"* - Validation is single split, CV averages multiple splits

### Interview Tip
Always mention stratified K-fold for classification and time series split for temporal data. Explain that you use pipelines to prevent data leakage. Know nested CV for proper hyperparameter tuning evaluation.

---

## Question 7
- [ ] Done

**What is Regularization and how does it help prevent overfitting?**

**Answer:**

### Definition
Regularization is a technique that adds a penalty term to the loss function to discourage complex models, reducing overfitting by constraining model parameters.

### The Core Idea

```
Original Loss:     L(θ) = Error(predictions, actual)
Regularized Loss:  L(θ) = Error(predictions, actual) + λ × Penalty(weights)
```

The penalty term discourages large weights, forcing the model to find simpler solutions.

### Types of Regularization

**L1 Regularization (Lasso)**
$$L = \text{Loss} + \lambda \sum_{i=1}^{n} |w_i|$$

- Adds absolute value of weights
- Produces **sparse models** (many weights become exactly zero)
- Performs **automatic feature selection**
- Robust to outliers in features

**L2 Regularization (Ridge)**
$$L = \text{Loss} + \lambda \sum_{i=1}^{n} w_i^2$$

- Adds squared weights
- Produces **small but non-zero weights**
- Handles multicollinearity well
- More stable solutions

**Elastic Net (L1 + L2)**
$$L = \text{Loss} + \lambda_1 \sum |w_i| + \lambda_2 \sum w_i^2$$

- Combines benefits of both
- Good when features are correlated
- Two hyperparameters to tune

### Detailed Comparison

| Aspect | L1 (Lasso) | L2 (Ridge) | Elastic Net |
|--------|-----------|-----------|-------------|
| Penalty | $\sum \|w\|$ | $\sum w^2$ | Both |
| Sparsity | Yes (zeros) | No | Partial |
| Feature Selection | Built-in | No | Partial |
| Correlated Features | Picks one | Keeps all | Keeps groups |
| Computation | Harder | Easier | Medium |
| Use Case | High-dim, sparse | Multicollinearity | Best of both |

### Why L1 Produces Sparsity (Geometric Intuition)

```
L1 Constraint (Diamond):      L2 Constraint (Circle):
                                    
       /\                           ___
      /  \                        /     \
     /    \  ← Corners touch     |       |
     \    /    axes first        |       |
      \  /                        \_____/
       \/                              
                                 ← Smooth, touches
                                   anywhere
```

L1's diamond shape has corners on the axes, making it likely the optimal point has zero coordinates.

### Regularization in Different Algorithms

| Algorithm | L1 Version | L2 Version | Implementation |
|-----------|-----------|-----------|----------------|
| Linear Regression | Lasso | Ridge | `sklearn.linear_model` |
| Logistic Regression | - | Default | `penalty='l2'` |
| SVM | - | Default | `C` parameter (inverse) |
| Neural Networks | Weight decay | Weight decay | `kernel_regularizer` |

### Other Regularization Techniques

**Dropout (Neural Networks):**
- Randomly zero out neurons during training
- Forces redundancy, prevents co-adaptation

**Early Stopping:**
- Stop training when validation error increases
- Implicit regularization

**Data Augmentation:**
- Create variations of training data
- Acts as regularization by increasing effective data size

**Batch Normalization:**
- Normalizes layer inputs
- Has regularization effect

### Implementation

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# L2 (Ridge)
ridge = Ridge(alpha=1.0)  # alpha = λ
ridge.fit(X_train, y_train)

# L1 (Lasso)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"Features selected: {(lasso.coef_ != 0).sum()}")

# Elastic Net
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # 50% L1, 50% L2
elastic.fit(X_train, y_train)
```

### Choosing λ (Regularization Strength)

- **λ = 0**: No regularization (original model)
- **λ → ∞**: All weights → 0 (underfitting)
- **Optimal λ**: Found via cross-validation

```python
from sklearn.linear_model import RidgeCV, LassoCV

# Automatic λ selection
ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
ridge_cv.fit(X, y)
print(f"Best alpha: {ridge_cv.alpha_}")
```

### Common Follow-up Questions
1. *"When would you use L1 vs L2?"* - L1 for feature selection/interpretability, L2 for prediction when all features matter
2. *"What if λ is too high?"* - Model underfits, all weights shrink too much
3. *"How does regularization relate to bias-variance?"* - Increases bias slightly, reduces variance significantly

### Interview Tip
Explain the geometric intuition for why L1 produces sparsity. Mention that in practice, you'd use cross-validation to find optimal λ. Discuss how regularization connects to the bias-variance tradeoff (adds bias to reduce variance).

---

## Question 8
- [ ] Done

**Describe the difference between Parametric and Non-Parametric models.**

**Answer:**

### Core Concept

**Parametric Models:**
- Assume data follows a specific functional form
- Have a **fixed number of parameters** regardless of data size
- Learn parameters from data, then discard training data

**Non-Parametric Models:**
- Make **minimal assumptions** about data distribution
- Complexity **grows with data** size
- Often keep training data for predictions

### Comprehensive Comparison

| Aspect | Parametric | Non-Parametric |
|--------|-----------|----------------|
| **Parameters** | Fixed (determined by model) | Grows with data |
| **Assumptions** | Strong (e.g., linearity) | Minimal |
| **Flexibility** | Limited to assumed form | Highly flexible |
| **Training Speed** | Fast | Slower |
| **Prediction Speed** | Fast | Often slower |
| **Data Requirement** | Less data needed | More data needed |
| **Interpretability** | Often easier | Often harder |
| **Overfitting Risk** | Lower (constrained) | Higher (flexible) |
| **Memory** | Only parameters | Often stores data |

### Algorithm Examples

**Parametric Models:**

| Algorithm | Parameters | Assumption |
|-----------|-----------|------------|
| Linear Regression | Coefficients (β₀, β₁, ..., βₙ) | Linear relationship |
| Logistic Regression | Coefficients + intercept | Log-odds is linear |
| Naive Bayes | Prior + likelihood params | Feature independence |
| Linear SVM | Weight vector + bias | Linear separability |
| Neural Networks* | Weights and biases | Architecture-defined |

*Neural networks are technically parametric but can approximate any function with enough parameters.

**Non-Parametric Models:**

| Algorithm | Why Non-Parametric |
|-----------|-------------------|
| K-Nearest Neighbors | Stores all training data |
| Decision Trees | Depth grows with data complexity |
| Random Forest | Ensemble of growing trees |
| Kernel SVM | Support vectors grow with data |
| Gaussian Processes | Complexity = O(n³) |

### Practical Implications

**When to Use Parametric:**
- Limited training data
- Strong domain knowledge about relationships
- Need fast predictions
- Interpretability is important
- Computational resources are limited

**When to Use Non-Parametric:**
- Large amounts of data available
- Unknown or complex relationships
- Flexibility is more important than speed
- Fine with "black box" predictions

### Example: Predicting House Prices

**Parametric (Linear Regression):**
```
Price = β₀ + β₁(sqft) + β₂(bedrooms) + β₃(location)

- Assumes: Linear relationship
- Parameters: 4 coefficients (fixed)
- Can fail if relationship is non-linear
```

**Non-Parametric (KNN):**
```
Price = Average of k nearest houses

- Assumes: Similar houses have similar prices
- Parameters: Stores all N houses
- Adapts to any relationship shape
```

### The "Semi-Parametric" Middle Ground

Some models combine both approaches:
- **Generalized Additive Models (GAM)**: Parametric structure with non-parametric components
- **Kernel Methods**: Parametric in feature space, non-parametric in complexity

### Common Misconceptions

| Misconception | Reality |
|--------------|--------|
| "Non-parametric = no parameters" | It has parameters, just not fixed count |
| "Deep learning is non-parametric" | It's parametric (fixed architecture) |
| "Parametric is always worse" | Often better with limited data |

### Common Follow-up Questions
1. *"Is Random Forest parametric or non-parametric?"* - Non-parametric (trees can grow indefinitely)
2. *"Why would you choose parametric with big data?"* - When you have strong prior knowledge, need interpretability, or fast inference
3. *"What about neural networks?"* - Technically parametric (fixed architecture), but with enough parameters can approximate any function

### Interview Tip
The key distinction is whether model complexity is fixed or grows with data. Mention that you'd choose based on data size, domain knowledge, and interpretability requirements. Know specific examples of each type.

---

## Question 9
- [ ] Done

**What is the curse of dimensionality and how does it impact ML models?**

**Answer:**

### Definition
The "curse of dimensionality" refers to various phenomena that arise when analyzing data in high-dimensional spaces that don't occur in low-dimensional settings. As dimensions increase, the volume of the space grows so fast that available data becomes sparse.

### The Core Problem: Sparsity

```
1D: 10 points fill a line segment well
2D: 10 points are sparse in a square (need 100)
3D: 10 points are very sparse in a cube (need 1000)
nD: Need 10ⁿ points for same density!
```

**Mathematical Intuition:**
- To maintain same data density in d dimensions, need $n^d$ samples
- 10 samples per dimension → 100D needs $10^{100}$ samples (impossible!)

### Key Phenomena

**1. Distance Concentration**

As dimensions increase, all points become equidistant:

$$\lim_{d \to \infty} \frac{\text{max\_distance} - \text{min\_distance}}{\text{min\_distance}} \to 0$$

**Implication**: Distance-based methods (KNN, K-Means) fail because "nearest" becomes meaningless.

**2. Volume Concentration in Corners**

```
2D Square: Most volume is interior
10D Hypercube: 99.7% of volume is in "corners"
100D Hypercube: Almost all data near edges
```

**Implication**: Data distribution intuitions break down.

**3. Sampling Becomes Impossible**

| Dimensions | Samples Needed | Feasibility |
|------------|---------------|-------------|
| 3 | 1,000 | Easy |
| 10 | 10 billion | Hard |
| 100 | 10^100 | Impossible |

### Impact on ML Algorithms

| Algorithm | Impact | Why |
|-----------|--------|-----|
| **KNN** | Severe | Distances become meaningless |
| **K-Means** | Severe | Clustering fails in high-D |
| **Decision Trees** | Moderate | Need more splits, prone to overfit |
| **Neural Networks** | Moderate | Can learn representations |
| **Linear Models** | Lower | Regularization helps |
| **Naive Bayes** | Lower | Independence assumption helps |

### Practical Symptoms

1. **Model Performance Degrades**: Test accuracy drops as features increase
2. **Overfitting**: Model memorizes sparse training data
3. **Training Slowdown**: Computation increases with dimensions
4. **Irrelevant Features Dominate**: Noise overwhelms signal
5. **Visualization Fails**: Can't plot or understand data

### Solutions

**1. Feature Selection** (Remove irrelevant features)
```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Keep top 20 features
selector = SelectKBest(mutual_info_classif, k=20)
X_reduced = selector.fit_transform(X, y)
```

**2. Dimensionality Reduction** (Compress features)
```python
from sklearn.decomposition import PCA

# Reduce to 50 components
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)
print(f"Variance retained: {pca.explained_variance_ratio_.sum():.2%}")
```

**3. Regularization** (Penalize complexity)
```python
from sklearn.linear_model import LassoCV

# L1 regularization zeros out irrelevant features
model = LassoCV(cv=5)
model.fit(X, y)
print(f"Non-zero features: {(model.coef_ != 0).sum()}")
```

**4. Manifold Learning** (Find lower-dimensional structure)
- t-SNE, UMAP for visualization
- Data often lies on lower-dimensional manifold

**5. Collect More Data** (If possible)
- More data helps but need exponentially more

**6. Domain Knowledge** (Feature engineering)
- Create meaningful features instead of all possible features

### Real-World Example: Text Classification

**Problem**: 100,000 unique words = 100,000 dimensions

**Symptoms**:
- Most documents have only ~100 words (99.9% sparse)
- Rare words cause overfitting

**Solutions**:
- TF-IDF reduces importance of common/rare words
- Word embeddings (Word2Vec): 100,000D → 300D
- Topic models (LDA): Documents as topic mixtures

### Rule of Thumb

```
Samples needed ≥ 10 × features (minimum)
Samples needed ≥ 50 × features (comfortable)

If you have 1,000 samples:
- Max ~100 features safely
- Use regularization or feature selection beyond that
```

### Common Follow-up Questions
1. *"How does deep learning handle high dimensions?"* - Learns useful representations, dropout regularizes, but still benefits from dimensionality reduction
2. *"What's the relationship to overfitting?"* - High dimensions = more ways to overfit, less data per region
3. *"How do you know you have this problem?"* - Performance degrades with more features, large gap between train/test

### Interview Tip
Explain the distance concentration phenomenon with a concrete example. Mention that you'd use PCA or feature selection as first steps. Emphasize that more features isn't always better — the goal is finding informative, low-dimensional representations.

---

## Question 10
- [ ] Done

**Explain the concept of Feature Engineering and its significance in ML.**

**Answer:**

### Definition
Feature engineering is the process of using domain knowledge to create, transform, and select input variables (features) that make machine learning algorithms work better. It's often the difference between a mediocre model and a great one.

### Why It Matters

> "Coming up with features is difficult, time-consuming, requires expert knowledge. Applied machine learning is basically feature engineering." — Andrew Ng

| Factor | Impact |
|--------|--------|
| Algorithm choice | 2-5% accuracy improvement |
| Hyperparameter tuning | 1-3% improvement |
| **Feature engineering** | **10-50% improvement** |

### Feature Engineering Techniques

**1. Numeric Transformations**

```python
import numpy as np

# Log transform (right-skewed data)
df['log_income'] = np.log1p(df['income'])

# Square root (reduce skew)
df['sqrt_area'] = np.sqrt(df['area'])

# Binning (continuous to categorical)
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 55, 100],
                         labels=['young', 'adult', 'middle', 'senior'])

# Polynomial features
df['area_squared'] = df['area'] ** 2
df['rooms_per_area'] = df['rooms'] / df['area']
```

**2. Categorical Encoding**

| Technique | When to Use | Example |
|-----------|------------|----------|
| One-Hot | Low cardinality, no order | Colors: Red→0,1,0 |
| Label Encoding | Ordinal categories | Education: Low=1, Med=2, High=3 |
| Target Encoding | High cardinality | City → average target for city |
| Frequency Encoding | When count matters | Category → occurrence count |

```python
# Target encoding (with smoothing to prevent overfitting)
def target_encode(df, col, target, smoothing=10):
    global_mean = df[target].mean()
    agg = df.groupby(col)[target].agg(['mean', 'count'])
    smooth = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
    return df[col].map(smooth)
```

**3. Date/Time Features**

```python
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['date'].dt.dayofweek >= 5
df['hour'] = df['date'].dt.hour
df['is_business_hours'] = df['hour'].between(9, 17)

# Cyclical encoding (for circular features)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Time since event
df['days_since_signup'] = (df['current_date'] - df['signup_date']).dt.days
```

**4. Text Features**

```python
# Basic statistics
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['avg_word_length'] = df['text_length'] / df['word_count']

# Sentiment, readability
from textblob import TextBlob
df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# TF-IDF, embeddings for actual text content
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=100)
text_features = tfidf.fit_transform(df['text'])
```

**5. Aggregation Features**

```python
# Customer-level aggregations
user_features = df.groupby('user_id').agg({
    'purchase_amount': ['sum', 'mean', 'std', 'count'],
    'purchase_date': ['min', 'max'],
    'category': 'nunique'
}).reset_index()

# Rolling/window features
df['rolling_avg_7d'] = df.groupby('user_id')['amount'].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
```

**6. Interaction Features**

```python
# Multiplicative interactions
df['bedroom_bathroom_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)
df['price_per_sqft'] = df['price'] / df['sqft']

# Polynomial interactions
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X)
```

**7. Domain-Specific Features**

| Domain | Feature Examples |
|--------|------------------|
| E-commerce | Recency, Frequency, Monetary (RFM) |
| Finance | Moving averages, volatility, ratios |
| Healthcare | BMI, age-adjusted metrics |
| NLP | TF-IDF, n-grams, embeddings |
| Images | HOG, SIFT, CNN embeddings |
| Time Series | Lag features, rolling stats, seasonality |

### Feature Engineering Workflow

```
1. Understand the Problem
   ↓
2. Explore Data (EDA)
   ↓
3. Create Features
   - Domain knowledge
   - Transformations
   - Interactions
   ↓
4. Select Features
   - Correlation analysis
   - Feature importance
   - Recursive elimination
   ↓
5. Validate
   - Cross-validation
   - Check for leakage
   ↓
6. Iterate
```

### Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Data Leakage | Using future info | Careful temporal splits |
| Target Leakage | Feature derived from target | Remove or use proper encoding |
| Overfitting | Too many features | Feature selection, regularization |
| Missing Indicator | Ignoring missingness pattern | Create `is_missing` flag |

### Real-World Example: Predicting Customer Churn

**Raw Data:**
- customer_id, signup_date, last_purchase, total_spent, support_tickets

**Engineered Features:**
```python
df['days_since_signup'] = (today - df['signup_date']).dt.days
df['days_since_purchase'] = (today - df['last_purchase']).dt.days
df['purchase_frequency'] = df['num_purchases'] / df['days_since_signup']
df['avg_ticket_value'] = df['total_spent'] / df['num_purchases']
df['support_ratio'] = df['support_tickets'] / df['num_purchases']
df['is_declining'] = df['recent_purchases'] < df['early_purchases']
df['engagement_score'] = df['logins'] * df['purchase_frequency']
```

### Common Follow-up Questions
1. *"How do you prevent data leakage?"* - Never use future information, be careful with target encoding, split before creating features
2. *"How do you handle feature selection?"* - Correlation analysis, feature importance, RFE, L1 regularization
3. *"What's your feature engineering process?"* - Describe the workflow: EDA → hypotheses → create → validate → iterate

### Interview Tip
Always emphasize that feature engineering requires domain expertise. Give concrete examples from your experience. Mention that good features often matter more than sophisticated algorithms — a simple model with great features often beats a complex model with raw features.

---

## Question 11
- [ ] Done

**What is Data Preprocessing and why is it important in ML?**

**Answer:**

### Definition
Data preprocessing is the essential step of transforming raw data into a clean, suitable format for machine learning algorithms. It typically consumes 60-80% of a data scientist's time.

### Why It's Critical

> "Garbage in, garbage out" — The quality of your model is only as good as the quality of your data.

| Without Preprocessing | With Preprocessing |
|----------------------|--------------------|
| Model fails or crashes | Model runs smoothly |
| Poor predictions | Accurate predictions |
| Misleading results | Reliable results |
| Slow training | Optimized training |

### Complete Preprocessing Pipeline

```
Raw Data
    ↓
┌─────────────────────────────────────┐
│ 1. Data Cleaning                    │
│    - Handle missing values          │
│    - Remove duplicates              │
│    - Fix data types                 │
│    - Correct errors                 │
├─────────────────────────────────────┤
│ 2. Handle Outliers                  │
│    - Detect (IQR, Z-score)          │
│    - Decide: remove, cap, transform │
├─────────────────────────────────────┤
│ 3. Feature Encoding                 │
│    - Categorical → Numeric          │
│    - One-hot, Label, Target         │
├─────────────────────────────────────┤
│ 4. Feature Scaling                  │
│    - Standardization / Normalization│
├─────────────────────────────────────┤
│ 5. Feature Selection/Engineering    │
│    - Remove irrelevant features     │
│    - Create new features            │
└─────────────────────────────────────┘
    ↓
Clean Data Ready for ML
```

### Step-by-Step Implementation

**1. Data Cleaning**

```python
import pandas as pd
import numpy as np

# Check data quality
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# Fix data types
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Handle missing values (see Question 14 for details)
df['age'].fillna(df['age'].median(), inplace=True)
df['category'].fillna(df['category'].mode()[0], inplace=True)
```

**2. Handle Outliers**

```python
# IQR method
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Cap outliers (winsorization)
df['value_capped'] = df['value'].clip(lower, upper)

# Or remove
df_clean = df[(df['value'] >= lower) & (df['value'] <= upper)]
```

**3. Encode Categoricals**

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# One-hot for nominal
df = pd.get_dummies(df, columns=['color'], drop_first=True)

# Label for ordinal
size_order = {'S': 0, 'M': 1, 'L': 2, 'XL': 3}
df['size_encoded'] = df['size'].map(size_order)
```

**4. Scale Features**

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization (mean=0, std=1)
scaler = StandardScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])

# Normalization (0-1 range)
minmax = MinMaxScaler()
df[['price']] = minmax.fit_transform(df[['price']])
```

**5. Complete Pipeline with Scikit-learn**

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define column types
numeric_features = ['age', 'income', 'score']
categorical_features = ['gender', 'category']

# Create transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline with model
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Fit on training data only!
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
```

### Critical Rule: Fit on Train Only!

```
✘ WRONG:                          ✔ RIGHT:
scaler.fit(X)                     scaler.fit(X_train)
X_scaled = scaler.transform(X)    X_train_scaled = scaler.transform(X_train)
train_test_split(X_scaled)        X_test_scaled = scaler.transform(X_test)
```

**Why?** Fitting on all data causes data leakage — test set statistics influence the model.

### Algorithm-Specific Requirements

| Algorithm | Missing Values | Scaling | Encoding |
|-----------|---------------|---------|----------|
| Linear/Logistic | ✘ Handle | ✔ Required | ✔ Required |
| SVM | ✘ Handle | ✔ Required | ✔ Required |
| KNN | ✘ Handle | ✔ Required | ✔ Required |
| Decision Trees | ✔ Can handle | ✘ Not needed | ✔ Required |
| Random Forest | ✔ Can handle | ✘ Not needed | ✔ Required |
| XGBoost | ✔ Can handle | ✘ Not needed | ✔ Required |
| Neural Networks | ✘ Handle | ✔ Required | ✔ Required |

### Common Follow-up Questions
1. *"What's the order of preprocessing steps?"* - Clean → Handle outliers → Encode → Scale → Feature selection
2. *"How do you handle new categories in production?"* - Use `handle_unknown='ignore'` in OneHotEncoder
3. *"When would you NOT scale?"* - Tree-based models, when interpretability of coefficients matters

### Interview Tip
Emphasize using pipelines to prevent data leakage. Know which algorithms require scaling (distance-based, gradient-based) vs which don't (tree-based). Be prepared to discuss your preprocessing workflow on a real project.

---

## Question 12
- [ ] Done

**Explain the difference between Feature Scaling and Normalization.**

**Answer:**

### Terminology Clarification

> Note: These terms are often used interchangeably and inconsistently. Here's the most common usage:

| Term | Also Called | What It Does |
|------|------------|---------------|
| **Standardization** | Z-score normalization | Transform to mean=0, std=1 |
| **Normalization** | Min-Max scaling | Transform to range [0, 1] |

### Standardization (Z-Score)

**Formula:**
$$z = \frac{x - \mu}{\sigma}$$

**Result:** Mean = 0, Standard Deviation = 1

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Verify
print(f"Mean: {X_standardized.mean():.2f}")  # ~0
print(f"Std: {X_standardized.std():.2f}")    # ~1
```

**Properties:**
- No bounded range (can be any value)
- Preserves outliers (less sensitive but doesn't eliminate)
- Assumes approximately normal distribution

### Normalization (Min-Max Scaling)

**Formula:**
$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**Result:** All values in range [0, 1]

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Custom range [a, b]
scaler = MinMaxScaler(feature_range=(-1, 1))
```

**Properties:**
- Bounded range [0, 1]
- Very sensitive to outliers
- Good when you need bounded values

### Detailed Comparison

| Aspect | Standardization | Normalization (Min-Max) |
|--------|----------------|------------------------|
| **Formula** | (x - mean) / std | (x - min) / (max - min) |
| **Range** | Unbounded | [0, 1] |
| **Center** | Mean = 0 | Min = 0 |
| **Outlier Impact** | Less sensitive | Very sensitive |
| **Distribution** | Preserves shape | Preserves shape |
| **Interpretation** | Standard deviations from mean | Proportion of range |

### Other Scaling Methods

**Robust Scaler (For Outliers):**
$$x_{robust} = \frac{x - \text{median}}{\text{IQR}}$$

```python
from sklearn.preprocessing import RobustScaler

robust = RobustScaler()
X_robust = robust.fit_transform(X)  # Uses median and IQR
```

**MaxAbs Scaler (Sparse Data):**
$$x_{maxabs} = \frac{x}{|x_{max}|}$$

```python
from sklearn.preprocessing import MaxAbsScaler

maxabs = MaxAbsScaler()
X_maxabs = maxabs.fit_transform(X)  # Range [-1, 1], preserves sparsity
```

**Power Transformer (Non-Normal Data):**
```python
from sklearn.preprocessing import PowerTransformer

# Box-Cox (positive values only)
pt = PowerTransformer(method='box-cox')

# Yeo-Johnson (any values)
pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X)  # Makes data more Gaussian
```

### When to Use Each

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Gradient-based algorithms (NN, LR) | Standardization | Faster convergence |
| Neural Networks with sigmoid/tanh | Normalization | Match activation range |
| Distance-based (KNN, K-Means) | Either (be consistent) | Equal feature contribution |
| Data with outliers | RobustScaler | Median-based, outlier-resistant |
| Sparse data (text) | MaxAbsScaler | Preserves sparsity |
| Heavily skewed data | PowerTransformer | Normalizes distribution |
| Tree-based models | None needed | Invariant to scaling |

### Visual Example

```
Original Data:        Age: [20, 25, 30, 35, 40]   Income: [30K, 50K, 80K, 120K, 200K]

Without Scaling:      Income dominates because of larger magnitude!

Standardized:         Age: [-1.4, -0.7, 0, 0.7, 1.4]   Income: [-1.0, -0.5, 0.1, 0.6, 1.8]
                      Both contribute equally based on their distributions

Normalized [0,1]:     Age: [0, 0.25, 0.5, 0.75, 1.0]   Income: [0, 0.12, 0.29, 0.53, 1.0]
                      Both in same range
```

### Common Mistakes

| Mistake | Problem | Solution |
|---------|---------|----------|
| Fit on all data | Data leakage | Fit on train, transform both |
| Scaling target variable | Usually not needed | Only scale features |
| Scaling after split | Inconsistent scaling | Use pipeline |
| Min-Max with outliers | Squishes normal data | Use RobustScaler or handle outliers first |

### Common Follow-up Questions
1. *"Why not always normalize?"* - Outliers severely affect min/max, makes most data squished
2. *"Does scaling affect tree-based models?"* - No, they're scale-invariant (split points don't change)
3. *"Should I scale the target?"* - Generally no for classification, sometimes for regression with very large targets

### Interview Tip
Know the formulas and be able to explain when you'd choose each. Mention that standardization is the default choice because it's less sensitive to outliers. Always emphasize fitting on training data only.

---

## Question 13
- [ ] Done

**What is the purpose of One-Hot Encoding and when is it used?**

**Answer:**

### Definition
One-hot encoding converts categorical variables into binary (0/1) columns, one column per category. Each row has exactly one "1" (hot) and the rest "0"s.

### Visual Example

```
Original:                 One-Hot Encoded:

Color                     Color_Red  Color_Green  Color_Blue
Red         →             1          0            0
Green       →             0          1            0
Blue        →             0          0            1
Red         →             1          0            0
```

### Why It's Necessary

Most ML algorithms require numeric input. But naive numeric encoding creates false ordinal relationships:

```
✘ WRONG: Label Encoding for Nominal Data
Red = 0, Green = 1, Blue = 2

Problem: Model thinks Blue > Green > Red (false ordering!)
Red + Blue = Green? (mathematically 0 + 2 = 2... but that's nonsense)
```

One-hot encoding treats each category as independent binary features.

### Implementation

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Pandas (simple)
df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True)

# Scikit-learn (for pipelines)
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[['color']])

# Get feature names
feature_names = encoder.get_feature_names_out(['color'])
```

### Important Parameters

| Parameter | Purpose | Recommendation |
|-----------|---------|----------------|
| `drop='first'` | Avoid dummy variable trap | Use for linear models |
| `handle_unknown='ignore'` | Handle new categories | Use in production |
| `sparse_output=True` | Memory efficiency | Use for high cardinality |

### The Dummy Variable Trap

**Problem:** N categories create N perfectly correlated columns (sum = 1 always)

**Solution:** Drop one category (becomes the "reference")

```python
# With 3 colors, use only 2 columns
Color_Green  Color_Blue    Interpretation
0            0             → Red (reference)
1            0             → Green
0            1             → Blue
```

### When to Use One-Hot Encoding

| Use One-Hot | Don't Use One-Hot |
|-------------|-------------------|
| Nominal categories (no order) | Ordinal categories (use label) |
| Low cardinality (<10-20 categories) | High cardinality (100s of categories) |
| Linear models, SVM, NN | Tree-based models (can use label) |
| When categories are meaningful | When category frequency matters |

### High Cardinality Problem

**Issue:** 1000 categories = 1000 new columns!
- Sparse, high-dimensional data
- Curse of dimensionality
- Memory issues

**Solutions:**

| Method | Description | When to Use |
|--------|------------|-------------|
| **Target Encoding** | Replace with mean of target | Classification/Regression |
| **Frequency Encoding** | Replace with category count | When frequency matters |
| **Feature Hashing** | Hash to fixed dimensions | Very high cardinality |
| **Embeddings** | Learn dense representation | Deep learning |
| **Grouping** | Combine rare categories | Domain knowledge |

```python
# Target Encoding (with category_encoders)
from category_encoders import TargetEncoder

encoder = TargetEncoder(cols=['city'])
X_encoded = encoder.fit_transform(X, y)

# Frequency Encoding
freq = df['city'].value_counts(normalize=True)
df['city_freq'] = df['city'].map(freq)

# Group rare categories
freq = df['city'].value_counts()
rare = freq[freq < 100].index
df['city_grouped'] = df['city'].replace(rare, 'Other')
```

### Tree-Based Models: Special Case

Decision trees can handle label-encoded categoricals:

```python
# Trees can split on any value
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['color_label'] = le.fit_transform(df['color'])

# Tree will create splits like: color_label <= 1.5
# This works because trees don't assume ordering
```

**Note:** XGBoost, LightGBM, CatBoost have native categorical support.

### Comparison of Encoding Methods

| Method | Dimensionality | Ordinal Info | Target Leakage Risk |
|--------|---------------|--------------|--------------------|
| One-Hot | High | No | No |
| Label | Low (1) | Implied (false) | No |
| Target | Low (1) | No | Yes (need CV) |
| Frequency | Low (1) | No | No |
| Binary | Medium (log₂N) | Partial | No |

### Common Follow-up Questions
1. *"Why use drop_first?"* - Avoids multicollinearity in linear models (dummy trap)
2. *"How do you handle unseen categories in production?"* - Use `handle_unknown='ignore'` (encodes as all zeros)
3. *"What about CatBoost's approach?"* - Uses target statistics with ordered boosting to prevent leakage

### Interview Tip
Mention the dummy variable trap and high cardinality problem proactively. Know alternative encoding methods (target, frequency) for high-cardinality features. Emphasize that tree-based models handle categoricals differently.

---

## Question 14
- [ ] Done

**Describe the concept of Handling Missing Values in datasets.**

**Answer:**

### Understanding Missing Data Mechanisms

Before handling missing data, understand WHY it's missing:

| Type | Meaning | Example | Handling |
|------|---------|---------|----------|
| **MCAR** | Missing Completely at Random | Random sensor failures | Any method works |
| **MAR** | Missing at Random | Higher income → less likely to report age | Model-based imputation |
| **MNAR** | Missing Not at Random | Sick people skip health surveys | Domain knowledge needed |

### Detecting Missing Values

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Overall summary
print(df.isnull().sum())
print(f"Missing percentage: {df.isnull().sum() / len(df) * 100}")

# Visualize missing pattern
import missingno as msno
msno.matrix(df)
msno.heatmap(df)  # Shows correlation in missingness

# Check patterns
missing_mask = df.isnull()
print(missing_mask.corr())  # Are missing values correlated?
```

### Handling Strategies

**1. Deletion Methods**

```python
# Drop rows with ANY missing values
df_clean = df.dropna()

# Drop rows with missing in specific columns
df_clean = df.dropna(subset=['important_column'])

# Drop rows with too many missing
threshold = len(df.columns) * 0.5  # Keep rows with >50% values
df_clean = df.dropna(thresh=threshold)

# Drop columns with too much missing
df_clean = df.dropna(axis=1, thresh=len(df) * 0.7)
```

| Method | When to Use | When NOT to Use |
|--------|------------|----------------|
| Drop rows | Few missing, MCAR | Many missing, MNAR |
| Drop columns | >50% missing | Important features |

**2. Simple Imputation**

```python
from sklearn.impute import SimpleImputer

# Numeric: Mean/Median
mean_imp = SimpleImputer(strategy='mean')
median_imp = SimpleImputer(strategy='median')  # Better for skewed data

# Categorical: Mode/Constant
mode_imp = SimpleImputer(strategy='most_frequent')
const_imp = SimpleImputer(strategy='constant', fill_value='Unknown')

# Apply
df['age'] = mean_imp.fit_transform(df[['age']])
df['category'] = mode_imp.fit_transform(df[['category']])
```

| Statistic | When to Use |
|-----------|-------------|
| Mean | Symmetric distributions |
| Median | Skewed distributions, outliers |
| Mode | Categorical variables |
| Constant | When missing has meaning |

**3. Advanced Imputation**

```python
# K-Nearest Neighbors Imputation
from sklearn.impute import KNNImputer

knn_imp = KNNImputer(n_neighbors=5)
X_imputed = knn_imp.fit_transform(X)

# Iterative Imputer (MICE-like)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

iter_imp = IterativeImputer(max_iter=10, random_state=42)
X_imputed = iter_imp.fit_transform(X)
```

| Method | Pros | Cons |
|--------|------|------|
| KNN | Captures local patterns | Slow for large data |
| Iterative (MICE) | Models feature relationships | Computationally expensive |

**4. Model-Based Imputation**

```python
# Use ML to predict missing values
from sklearn.ensemble import RandomForestRegressor

# Split data
known = df[df['age'].notna()]
unknown = df[df['age'].isna()]

# Train model
features = ['income', 'education_years', 'job_category']
model = RandomForestRegressor()
model.fit(known[features], known['age'])

# Predict missing
df.loc[df['age'].isna(), 'age'] = model.predict(unknown[features])
```

**5. Missing Indicator (When Missingness Has Meaning)**

```python
# Create binary indicator
df['income_missing'] = df['income'].isna().astype(int)

# Then impute the original column
df['income'] = df['income'].fillna(df['income'].median())
```

**Use when:**
- Missing itself is informative (e.g., didn't answer = sensitive topic)
- MNAR data
- Want model to learn from missingness pattern

### Domain-Specific Strategies

| Domain | Feature | Strategy |
|--------|---------|----------|
| E-commerce | Purchase history | 0 (no purchases) |
| Healthcare | Test results | Carry forward last value |
| Time series | Sensor readings | Interpolation |
| Surveys | Income | Category "Prefer not to say" |

### Time Series Specific

```python
# Forward fill (carry last known value)
df['value'] = df['value'].fillna(method='ffill')

# Backward fill
df['value'] = df['value'].fillna(method='bfill')

# Linear interpolation
df['value'] = df['value'].interpolate(method='linear')

# Time-based interpolation
df['value'] = df['value'].interpolate(method='time')
```

### Decision Framework

```
Missing Data
    │
    ├── <5% missing?
    │       ├── Yes → Simple imputation or deletion
    │       └── No → Continue
    │
    ├── MCAR?
    │       ├── Yes → Any method works
    │       └── No → Need careful handling
    │
    ├── Missingness informative?
    │       ├── Yes → Add missing indicator
    │       └── No → Continue
    │
    └── Use advanced imputation
            (KNN, Iterative, Model-based)
```

### Common Mistakes

| Mistake | Why It's Bad | Solution |
|---------|-------------|----------|
| Impute before train/test split | Data leakage | Fit imputer on train only |
| Always use mean | Ignores data structure | Consider distribution, relationships |
| Ignoring missingness pattern | Loses information | Add missing indicator when meaningful |
| Not documenting choices | Reproducibility issues | Track imputation decisions |

### Common Follow-up Questions
1. *"How do you detect MCAR vs MAR?"* - Little's MCAR test, check if missingness correlates with other variables
2. *"What if >50% is missing?"* - Consider dropping column, or if critical, use advanced imputation + missing indicator
3. *"How do you handle missing in production?"* - Same imputer fitted on training data, handle truly new categories

### Interview Tip
First explain the missing data mechanisms (MCAR/MAR/MNAR). Show you understand that the "best" method depends on WHY data is missing. Mention missing indicators for MNAR. Always emphasize fitting imputers on training data only.

---

## Question 15
- [ ] Done

**What is Feature Selection and its techniques?**

**Answer:**

### Definition
Feature selection is the process of selecting a subset of relevant features for model building. It reduces overfitting, improves accuracy, and decreases training time by eliminating irrelevant or redundant features.

### Why Feature Selection Matters

| Benefit | Explanation |
|---------|-------------|
| **Reduces Overfitting** | Fewer features = less chance to fit noise |
| **Improves Accuracy** | Removes noisy/irrelevant features |
| **Faster Training** | Less computation |
| **Better Interpretability** | Easier to explain fewer features |
| **Reduces Data Collection Cost** | Identify truly important features |

### Three Main Categories

```
┌───────────────────────────────────────────────────────┐
│          Feature Selection Methods              │
├────────────────┬───────────────┬────────────────────┤
│     Filter     │    Wrapper    │     Embedded       │
├────────────────┼───────────────┼────────────────────┤
│ - Correlation  │ - RFE         │ - L1 (Lasso)       │
│ - Chi-squared  │ - Forward     │ - Tree importance  │
│ - Mutual Info  │ - Backward    │ - Elastic Net      │
│ - Variance     │ - Sequential  │                    │
└────────────────┴───────────────┴────────────────────┘
     Fast              Slow              Medium
  Model-agnostic    Most accurate     Part of training
```

### 1. Filter Methods (Statistical Tests)

Independent of any ML model. Fast and scalable.

```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    f_regression, VarianceThreshold
)

# Remove low variance features
var_selector = VarianceThreshold(threshold=0.1)
X_reduced = var_selector.fit_transform(X)

# Select top K features by F-statistic
f_selector = SelectKBest(f_classif, k=10)
X_reduced = f_selector.fit_transform(X, y)

# Mutual information (captures non-linear relationships)
mi_selector = SelectKBest(mutual_info_classif, k=10)
X_reduced = mi_selector.fit_transform(X, y)

# Get selected feature names
selected_mask = f_selector.get_support()
selected_features = X.columns[selected_mask].tolist()
```

**Common Filter Methods:**

| Method | Best For | Measures |
|--------|----------|----------|
| Variance Threshold | Any | Feature variance |
| Correlation | Regression | Linear relationship with target |
| Chi-squared | Classification (non-negative) | Independence from target |
| ANOVA F-test | Classification | Variance between classes |
| Mutual Information | Any | Any relationship (non-linear too) |

**Correlation Analysis:**

```python
import seaborn as sns

# Remove highly correlated features
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation > 0.9
to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
X_reduced = X.drop(columns=to_drop)

# Visualize
sns.heatmap(corr_matrix, annot=True)
```

### 2. Wrapper Methods (Search-Based)

Use model performance to evaluate feature subsets. More accurate but slower.

```python
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier

# Recursive Feature Elimination (RFE)
model = RandomForestClassifier(n_estimators=100)
rfe = RFE(estimator=model, n_features_to_select=10, step=1)
X_reduced = rfe.fit_transform(X, y)

print(f"Selected features: {X.columns[rfe.support_].tolist()}")
print(f"Feature ranking: {rfe.ranking_}")

# Sequential Feature Selection (Forward)
sfs_forward = SequentialFeatureSelector(
    model, n_features_to_select=10, direction='forward', cv=5
)
X_reduced = sfs_forward.fit_transform(X, y)

# Sequential Feature Selection (Backward)
sfs_backward = SequentialFeatureSelector(
    model, n_features_to_select=10, direction='backward', cv=5
)
X_reduced = sfs_backward.fit_transform(X, y)
```

**Comparison:**

| Method | Process | Speed |
|--------|---------|-------|
| Forward | Start empty, add best one at a time | Fast start |
| Backward | Start full, remove worst one at a time | Slow start |
| RFE | Like backward but uses feature importance | Medium |

### 3. Embedded Methods (Built into Training)

Feature selection happens during model training.

```python
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier

# L1 Regularization (Lasso)
lasso = LassoCV(cv=5)
lasso.fit(X, y)

selected = X.columns[lasso.coef_ != 0].tolist()
print(f"Non-zero features: {len(selected)}")

# Tree-based Feature Importance
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)

importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Select top features
top_features = importances.head(10)['feature'].tolist()
```

**Tree Importance Methods:**

| Method | Description |
|--------|-------------|
| Gini/Entropy Importance | Impurity decrease from splits |
| Permutation Importance | Performance drop when feature shuffled |
| SHAP Values | Game-theoretic feature contribution |

```python
# Permutation Importance (more reliable)
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10)
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': result.importances_mean
}).sort_values('importance', ascending=False)

# SHAP Values (most detailed)
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
```

### Comparison of Methods

| Aspect | Filter | Wrapper | Embedded |
|--------|--------|---------|----------|
| **Speed** | Fast | Slow | Medium |
| **Accuracy** | Good | Best | Good |
| **Overfitting Risk** | Low | High | Low |
| **Model Dependency** | No | Yes | Yes |
| **Captures Interactions** | No | Yes | Partial |
| **Scalability** | Excellent | Poor | Good |

### Practical Workflow

```
1. Remove constant/near-constant features (VarianceThreshold)
   ↓
2. Remove highly correlated features (>0.9 correlation)
   ↓
3. Apply filter method for initial reduction (MI, Chi-squared)
   ↓
4. Use embedded method (L1 or tree importance)
   ↓
5. Fine-tune with wrapper method if needed (RFE)
   ↓
6. Validate with cross-validation
```

### Common Follow-up Questions
1. *"How do you choose the number of features?"* - Cross-validation, elbow plot of performance vs. features
2. *"What's the difference between feature selection and PCA?"* - Selection keeps original features, PCA creates new ones
3. *"When would you skip feature selection?"* - Deep learning (learns representations), when all features are important

### Interview Tip
Start with fast filter methods for initial exploration, then use embedded methods (Lasso, tree importance) for final selection. Mention that wrapper methods are most accurate but computationally expensive. Always validate feature selection with cross-validation to avoid overfitting to a single train/test split.

---

## Question 16
- [ ] Done

**Explain the difference between Filter, Wrapper, and Embedded methods for Feature Selection.**

**Answer:**

### Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                    FILTER METHODS                          │
│  Features → [Statistical Test] → Ranked Features → Model   │
│                                                              │
│  Examples: Correlation, Chi-squared, Mutual Information      │
│  Speed: ★★★★★  Accuracy: ★★★                               │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                    WRAPPER METHODS                         │
│  Features → [Search + Model Training] → Best Subset        │
│       ↑___________Feedback Loop___________|                 │
│                                                              │
│  Examples: RFE, Forward Selection, Backward Elimination     │
│  Speed: ★★  Accuracy: ★★★★★                                │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                    EMBEDDED METHODS                        │
│  Features → [Model Training with Selection Built-in]       │
│                                                              │
│  Examples: Lasso (L1), Tree Feature Importance, ElasticNet  │
│  Speed: ★★★★  Accuracy: ★★★★                              │
└────────────────────────────────────────────────────────────────────┘
```

### Detailed Comparison

| Aspect | Filter | Wrapper | Embedded |
|--------|--------|---------|----------|
| **How it works** | Statistical tests | Model-based search | Built into training |
| **Speed** | Very fast | Slow | Fast-Medium |
| **Accuracy** | Good | Best | Good-Great |
| **Model-dependent** | No | Yes | Yes |
| **Captures interactions** | No | Yes | Partial |
| **Scalability** | Excellent | Poor | Good |
| **Overfitting risk** | Low | High | Low |
| **When to use** | Initial exploration | Final fine-tuning | General use |

### Filter Methods - Deep Dive

**How They Work:**
- Evaluate each feature independently using statistical measures
- Rank features by their scores
- Select top K features or those above a threshold
- Independent of any ML model

**Common Techniques:**

| Technique | Use Case | What It Measures |
|-----------|----------|------------------|
| Variance Threshold | Any | Feature variance (remove constants) |
| Pearson Correlation | Regression | Linear relationship with target |
| Spearman Correlation | Regression | Monotonic relationship |
| Chi-squared | Classification | Independence (categorical features) |
| ANOVA F-test | Classification | Mean difference between classes |
| Mutual Information | Any | Any dependency (including non-linear) |

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# ANOVA F-test for classification
selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X, y)
scores = selector.scores_

# Mutual Information (captures non-linear)
selector = SelectKBest(mutual_info_classif, k=10)
X_new = selector.fit_transform(X, y)
```

**Pros:**
- Very fast, scales to millions of features
- Model-agnostic (can use with any algorithm)
- No overfitting to training data

**Cons:**
- Ignores feature interactions
- May select redundant features
- Doesn't optimize for specific model performance

### Wrapper Methods - Deep Dive

**How They Work:**
- Use the actual ML model to evaluate feature subsets
- Search through feature combinations
- Select subset that gives best model performance

**Search Strategies:**

| Strategy | Process | Complexity |
|----------|---------|------------|
| Forward Selection | Start empty, add best feature iteratively | O(n²) |
| Backward Elimination | Start full, remove worst feature iteratively | O(n²) |
| Exhaustive Search | Try all 2ⁿ combinations | O(2ⁿ) |
| RFE | Recursive elimination based on importance | O(n²) |

```python
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=50)

# Recursive Feature Elimination
rfe = RFE(estimator=model, n_features_to_select=10, step=1)
rfe.fit(X, y)
selected_features = X.columns[rfe.support_].tolist()

# Sequential Forward Selection
sfs = SequentialFeatureSelector(model, n_features_to_select=10, 
                                 direction='forward', cv=5)
sfs.fit(X, y)
selected_features = X.columns[sfs.get_support()].tolist()
```

**Pros:**
- Considers feature interactions
- Optimizes for actual model performance
- Most accurate feature selection

**Cons:**
- Computationally expensive
- Risk of overfitting to training data
- Model-specific (selected features may not work for other models)

### Embedded Methods - Deep Dive

**How They Work:**
- Feature selection happens during model training
- The algorithm inherently performs selection
- No separate selection step needed

**Common Techniques:**

| Technique | Algorithm | How It Selects |
|-----------|-----------|----------------|
| L1 Regularization | Lasso, Logistic | Shrinks weights to exactly zero |
| Tree Importance | RF, XGBoost | Features that contribute most to splits |
| Elastic Net | Elastic Net | Combination of L1 and L2 |

```python
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier

# L1 Regularization (Lasso)
lasso = LassoCV(cv=5)
lasso.fit(X, y)
selected = X.columns[lasso.coef_ != 0].tolist()
print(f"Lasso selected {len(selected)} features")

# Tree-based importance
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.nlargest(10).index.tolist()
```

**Pros:**
- Considers feature interactions
- Efficient (selection during training)
- Less prone to overfitting than wrapper

**Cons:**
- Model-specific
- May not find optimal subset
- Different models may give different importance

### Decision Framework

```
Start
  │
  ├── Very high-dimensional (>10K features)?
  │     └── Yes → Start with Filter (fast initial reduction)
  │
  ├── Need model-agnostic selection?
  │     └── Yes → Use Filter methods only
  │
  ├── Computational budget available?
  │     ├── Limited → Embedded (Lasso, Tree importance)
  │     └── Generous → Wrapper (RFE, SFS) for final selection
  │
  └── Typical workflow:
        Filter → Embedded → Wrapper (optional fine-tuning)
```

### Common Follow-up Questions
1. *"Which is best?"* - No single best; wrapper is most accurate but slowest, filter is fastest but may miss interactions
2. *"Can you combine them?"* - Yes! Filter for initial reduction, then embedded or wrapper
3. *"What about neural networks?"* - Dropout and attention mechanisms are forms of embedded selection

### Interview Tip
Explain the tradeoff between computational cost and accuracy. Mention that in practice, you'd often combine methods: filter for initial reduction, then embedded or wrapper for fine-tuning. Be ready to explain why wrapper methods risk overfitting (they optimize for training performance).

---

## Question 17
- [ ] Done

**What is Principal Component Analysis (PCA) and its role in dimensionality reduction?**

**Answer:**

### Definition
PCA is an unsupervised linear transformation technique that converts correlated features into a smaller set of uncorrelated variables called **principal components**, while preserving as much variance (information) as possible.

### Intuition

Imagine your data as a cloud of points in 3D space:
- PCA finds the direction of maximum spread (PC1)
- Then finds the next direction of maximum spread, perpendicular to PC1 (PC2)
- And so on...

```
Original 3D Data:           After PCA (2D projection):

      o                          o
    o   o                      o   o
  o   o   o      →        o   o   o
    o   o                      o   o
      o                          o
```

### Mathematical Foundation

**1. Center the Data:**
$$X_{centered} = X - \mu$$

**2. Compute Covariance Matrix:**
$$C = \frac{1}{n-1} X_{centered}^T X_{centered}$$

**3. Find Eigenvectors and Eigenvalues:**
$$C \cdot v = \lambda \cdot v$$

- Eigenvectors (v) = directions of principal components
- Eigenvalues (λ) = amount of variance in that direction

**4. Project Data:**
$$X_{reduced} = X_{centered} \cdot W$$

Where W contains the top k eigenvectors.

### Step-by-Step Algorithm

```
1. Standardize the data (mean=0, std=1)
2. Compute covariance matrix
3. Calculate eigenvectors and eigenvalues
4. Sort eigenvectors by eigenvalues (descending)
5. Select top k eigenvectors
6. Transform data to new subspace
```

### Implementation

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. ALWAYS standardize first!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_scaled)

# 3. Analyze results
print(f"Original dimensions: {X.shape[1]}")
print(f"Reduced dimensions: {X_pca.shape[1]}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# 4. Visualize explained variance
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(range(len(pca.explained_variance_ratio_)), 
        pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         np.cumsum(pca.explained_variance_ratio_))
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.tight_layout()
```

### Choosing the Number of Components

| Method | Description | Implementation |
|--------|-------------|----------------|
| **Kaiser's Rule** | Keep components with eigenvalue > 1 | For standardized data |
| **Scree Plot** | Find "elbow" in variance plot | Visual inspection |
| **Cumulative Variance** | Keep 95% (or 99%) of variance | `n_components=0.95` |
| **Cross-Validation** | Optimize for downstream task | GridSearchCV |

```python
# Kaiser's rule (for standardized data)
pca_full = PCA()
pca_full.fit(X_scaled)
kaiser_n = (pca_full.explained_variance_ > 1).sum()

# Cumulative variance threshold
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.argmax(cumsum >= 0.95) + 1
```

### Applications of PCA

| Application | Why Use PCA |
|-------------|-------------|
| **Visualization** | Project high-D to 2D/3D for plotting |
| **Noise Reduction** | Remove low-variance components (noise) |
| **Preprocessing** | Reduce dimensionality before ML |
| **Feature Decorrelation** | Remove multicollinearity |
| **Compression** | Store data with fewer dimensions |

### Advantages and Limitations

| Advantages | Limitations |
|------------|-------------|
| Reduces dimensionality | Loses interpretability |
| Removes multicollinearity | Only captures linear relationships |
| Reduces overfitting | Sensitive to outliers |
| Fast and deterministic | Assumes importance = variance |
| Noise reduction | May lose important small-variance features |

### PCA vs Other Methods

| Method | Type | Best For |
|--------|------|----------|
| PCA | Linear | General dimensionality reduction |
| t-SNE | Non-linear | Visualization only (not preprocessing) |
| UMAP | Non-linear | Visualization + faster than t-SNE |
| Autoencoders | Non-linear | Complex non-linear patterns |
| LDA | Supervised | Classification (maximizes separability) |

### Important Considerations

**1. Always Standardize First!**
```python
# Without standardization, features with larger scales dominate
# Age (0-100) vs Income (0-1,000,000) → Income dominates!
```

**2. PCA is Sensitive to Outliers**
```python
# Use Robust PCA or remove outliers first
from sklearn.decomposition import PCA
# Or use kernel PCA for non-linear data
from sklearn.decomposition import KernelPCA
```

**3. Inverse Transform for Interpretation**
```python
# Transform back to original space (approximate)
X_reconstructed = pca.inverse_transform(X_pca)
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
```

### Real-World Example: Image Compression

```python
from sklearn.datasets import load_digits

digits = load_digits()  # 64 features (8x8 pixels)
X = digits.data

pca = PCA(n_components=20)  # 64 → 20 dimensions
X_pca = pca.fit_transform(X)

print(f"Compression: {64} → {20} dimensions")
print(f"Variance retained: {pca.explained_variance_ratio_.sum():.1%}")
# Output: ~95% variance retained with 69% fewer dimensions!
```

### Common Follow-up Questions
1. *"Why standardize before PCA?"* - PCA maximizes variance; features with larger scales would dominate
2. *"Can PCA work for non-linear data?"* - Use Kernel PCA or autoencoders instead
3. *"What's the difference between PCA and SVD?"* - PCA uses covariance matrix; SVD is more numerically stable (sklearn uses SVD internally)
4. *"Should I use PCA before or after train/test split?"* - After! Fit on train, transform both

### Interview Tip
Explain the intuition: PCA finds directions of maximum variance and projects data onto them. Always mention standardization as a prerequisite. Know how to choose the number of components (95% variance rule). Be ready to discuss limitations (linear only, interpretability loss).

---

## Question 18
- [ ] Done

**Describe the concept of Outlier Detection and its methods.**

**Answer:**

### Definition
Outliers are data points that significantly differ from other observations. They can be errors (should be removed) or genuine extreme values (should be kept and possibly analyzed).

### Types of Outliers

```
1. Point/Global Outliers      2. Contextual Outliers      3. Collective Outliers
   (Individually extreme)        (Extreme in context)        (Group is extreme)

       o ← outlier              Temperature: 30°C           ┌───────────┐
   o o o o o                    in July: Normal ✔         │ o o o o o │ ← cluster
   o o o o o                    in January: Outlier ✘     └───────────┘
   o o o o o                                                is outlier
```

### Why Outliers Matter

| Impact | Affected By |
|--------|-------------|
| Skew mean/variance | Linear regression, PCA |
| Distort distance calculations | KNN, K-Means, SVM |
| Dominate gradient updates | Neural networks |
| Usually OK | Tree-based models (robust) |

### Detection Methods

**1. Statistical Methods (Univariate)**

```python
import numpy as np
from scipy import stats

# Z-Score Method (assumes normal distribution)
z_scores = np.abs(stats.zscore(df['column']))
outliers_z = df[z_scores > 3]  # Beyond 3 standard deviations

# Modified Z-Score (robust, uses median)
def modified_zscore(data):
    median = np.median(data)
    mad = np.median(np.abs(data - median))  # Median Absolute Deviation
    modified_z = 0.6745 * (data - median) / mad
    return modified_z

mod_z = np.abs(modified_zscore(df['column']))
outliers_modz = df[mod_z > 3.5]

# IQR Method (no distribution assumption)
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = df[(df['column'] < lower_bound) | (df['column'] > upper_bound)]
```

**Comparison:**

| Method | Assumption | Threshold | Best For |
|--------|-----------|-----------|----------|
| Z-score | Normal distribution | 3 | Normally distributed data |
| Modified Z-score | None | 3.5 | Data with outliers |
| IQR | None | 1.5×IQR | General, robust |

**2. Multivariate Methods**

```python
# Mahalanobis Distance (accounts for correlations)
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

cov_matrix = np.cov(X.T)
inv_cov = np.linalg.inv(cov_matrix)
mean = X.mean(axis=0)

mahal_dist = [mahalanobis(row, mean, inv_cov) for row in X.values]
threshold = chi2.ppf(0.975, df=X.shape[1])  # 97.5% confidence
outliers = X[np.array(mahal_dist) > np.sqrt(threshold)]
```

**3. Proximity-Based Methods**

```python
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
outlier_labels = lof.fit_predict(X)  # -1 = outlier, 1 = inlier
outliers = X[outlier_labels == -1]

# Isolation Forest (fast, scalable)
iso = IsolationForest(contamination=0.1, random_state=42)
outlier_labels = iso.fit_predict(X)

# One-Class SVM
ocsvm = OneClassSVM(nu=0.1)
outlier_labels = ocsvm.fit_predict(X)
```

**4. Density-Based Methods**

```python
from sklearn.cluster import DBSCAN

# Points not in any cluster are outliers
db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X)
outliers = X[labels == -1]  # Noise points
```

### Method Comparison

| Method | Complexity | Handles Multivariate | Handles Non-Linear |
|--------|-----------|---------------------|-------------------|
| IQR | O(n log n) | No | No |
| Z-score | O(n) | No | No |
| Mahalanobis | O(n³) | Yes | No |
| LOF | O(n²) | Yes | Yes |
| Isolation Forest | O(n log n) | Yes | Yes |
| DBSCAN | O(n log n) | Yes | Yes |

### Handling Strategies

Once detected, you have several options:

| Strategy | When to Use | Implementation |
|----------|------------|----------------|
| **Remove** | Clearly errors | `df = df[~outlier_mask]` |
| **Cap/Winsorize** | Reduce impact | `df['col'].clip(lower, upper)` |
| **Transform** | Reduce skewness | `np.log1p(df['col'])` |
| **Impute** | Treat as missing | Replace with median |
| **Keep** | Genuine values | Flag for analysis |
| **Separate Model** | Different behavior | Train separately |

```python
# Winsorization (cap at percentiles)
from scipy.stats import mstats

winsorized = mstats.winsorize(df['column'], limits=[0.05, 0.05])

# Or using clip
lower = df['column'].quantile(0.05)
upper = df['column'].quantile(0.95)
df['column_capped'] = df['column'].clip(lower, upper)

# Log transform (for right-skewed data)
df['column_log'] = np.log1p(df['column'])
```

### Decision Framework

```
Outlier Detected
       │
       ├── Data entry error? (impossible value)
       │        └── Yes → Remove or correct
       │
       ├── Measurement error? (sensor malfunction)
       │        └── Yes → Remove or impute
       │
       ├── Genuine extreme? (fraud, rare event)
       │        ├── Model sensitive to outliers? → Cap or transform
       │        └── Using robust model? → Keep, maybe flag
       │
       └── Represents different population?
                └── Yes → Separate analysis or model
```

### Visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Box plot (shows IQR-based outliers)
sns.boxplot(x=df['column'])

# Scatter plot with outliers highlighted
plt.scatter(X[outlier_labels == 1, 0], X[outlier_labels == 1, 1], c='blue')
plt.scatter(X[outlier_labels == -1, 0], X[outlier_labels == -1, 1], c='red')
plt.legend(['Inliers', 'Outliers'])
```

### Common Follow-up Questions
1. *"How do you decide if an outlier is an error?"* - Domain knowledge, impossibility checks (negative age), comparison with other sources
2. *"Should you always remove outliers?"* - No! In fraud detection, outliers ARE what you're looking for
3. *"Which method is best?"* - Isolation Forest is often a good default; IQR for quick univariate check

### Interview Tip
First ask "Are these outliers or interesting data points?" Explain that in fraud detection, outliers are the target. Know multiple detection methods (IQR for simple, Isolation Forest for complex). Always consider the business context before removing data.

---

## Question 19
- [ ] Done

**What is the Imputer class in scikit-learn and how is it used?**

**Answer:**

### Overview
Scikit-learn provides several imputer classes for handling missing values. These transformers follow the fit/transform pattern, making them compatible with pipelines.

### Available Imputers

| Imputer | Description | Best For |
|---------|-------------|----------|
| `SimpleImputer` | Fill with statistic (mean, median, mode) | Basic imputation |
| `KNNImputer` | Fill using k-nearest neighbors | Multivariate relationships |
| `IterativeImputer` | Model each feature iteratively (MICE) | Complex patterns |
| `MissingIndicator` | Create binary flags for missingness | When missing has meaning |

### SimpleImputer

```python
from sklearn.impute import SimpleImputer
import numpy as np

# Create data with missing values
X = np.array([[1, 2], [np.nan, 3], [7, np.nan], [4, 5]])

# Mean imputation (default)
mean_imputer = SimpleImputer(strategy='mean')
X_mean = mean_imputer.fit_transform(X)

# Median imputation (better for skewed data)
median_imputer = SimpleImputer(strategy='median')
X_median = median_imputer.fit_transform(X)

# Mode imputation (for categorical)
mode_imputer = SimpleImputer(strategy='most_frequent')
X_mode = mode_imputer.fit_transform(X)

# Constant imputation
const_imputer = SimpleImputer(strategy='constant', fill_value=-999)
X_const = const_imputer.fit_transform(X)
```

**Strategy Options:**

| Strategy | Use Case | Handles |
|----------|----------|--------|
| `'mean'` | Numeric, symmetric distribution | float only |
| `'median'` | Numeric, skewed distribution | float only |
| `'most_frequent'` | Categorical or numeric | any dtype |
| `'constant'` | When you need specific value | any dtype |

### KNNImputer

```python
from sklearn.impute import KNNImputer

# Fill missing with average of k nearest neighbors
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
X_knn = knn_imputer.fit_transform(X)

# Parameters
# n_neighbors: Number of neighbors to use
# weights: 'uniform' or 'distance' (closer = more weight)
# metric: Distance metric (default: 'nan_euclidean')
```

**When to use:** When feature relationships matter and you have correlated features.

### IterativeImputer (MICE)

```python
# Enable experimental feature
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# Models each feature as a function of others
iter_imputer = IterativeImputer(
    estimator=BayesianRidge(),  # Can use any regressor
    max_iter=10,
    random_state=42
)
X_iter = iter_imputer.fit_transform(X)
```

**When to use:** Complex multivariate missing patterns, when relationships between features are important.

### MissingIndicator

```python
from sklearn.impute import MissingIndicator

# Create binary flags for missing values
indicator = MissingIndicator()
missing_flags = indicator.fit_transform(X)

# Combine with imputed data
from sklearn.compose import make_column_transformer

# The imputed features + missing indicators
import numpy as np
X_with_flags = np.hstack([X_imputed, missing_flags])
```

**When to use:** When the fact that data is missing is informative (MNAR).

### Using Imputers in Pipelines (Critical!)

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Define column types
numeric_features = ['age', 'income', 'score']
categorical_features = ['gender', 'city']

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline with model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Fit on training data only!
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
```

### Critical Rules

**1. Always fit on training data only!**
```python
# ✘ WRONG: Data leakage!
imputer.fit(X)  # Fits on all data including test
X_imputed = imputer.transform(X)
X_train, X_test = train_test_split(X_imputed)

# ✔ CORRECT
X_train, X_test = train_test_split(X)
imputer.fit(X_train)  # Fit only on training
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)  # Use same imputer!
```

**2. Preserve imputer for production**
```python
import joblib

# Save
joblib.dump(pipeline, 'model_pipeline.pkl')

# Load and use
loaded_pipeline = joblib.load('model_pipeline.pkl')
predictions = loaded_pipeline.predict(new_data)  # Handles missing automatically
```

### Handling New Categories in Production

```python
# What if test data has values not seen in training?

# For SimpleImputer with constant
imputer = SimpleImputer(strategy='constant', fill_value='Unknown')

# For OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')  # Creates zero vector
```

### Common Follow-up Questions
1. *"What's the difference between impute then split vs split then impute?"* - Impute then split causes data leakage (test info in imputer)
2. *"When would you use IterativeImputer?"* - When features are correlated and multivariate imputation would be more accurate
3. *"How do you choose between mean and median?"* - Median for skewed data or outliers; mean for normal distribution

### Interview Tip
Always emphasize using pipelines to prevent data leakage. Know the difference between imputer strategies and when to use each. Mention that KNNImputer and IterativeImputer capture multivariate relationships while SimpleImputer is univariate.

---

## Question 20
- [ ] Done

**Explain the concept of Handling Imbalanced Datasets in ML.**

**Answer:**

### What is Class Imbalance?

When one class significantly outnumbers another:
```
Fraud Detection:    99.9% Normal, 0.1% Fraud
Medical Diagnosis:  95% Healthy, 5% Disease
Churn Prediction:   90% Stay, 10% Churn
```

### Why It's a Problem

| Problem | Explanation |
|---------|-------------|
| Accuracy is misleading | 99% accuracy by predicting all "Normal" |
| Model ignores minority | Optimizes for majority class |
| Gradient dominated | Majority samples dominate gradient updates |
| Poor minority recall | Fails to detect the cases you care about |

**Example:** Fraud detection with 99.9% normal
- Model predicts "Normal" for everything: 99.9% accuracy!
- But catches 0% of fraud - completely useless

### Strategies Overview

```
┌───────────────────────────────────────────────────────┐
│           Handling Imbalanced Data                  │
├───────────────┬───────────────┬───────────────┬────────┤
│   Data Level   │  Algorithm   │   Threshold    │Metrics│
│                │    Level     │    Tuning      │       │
├───────────────┼───────────────┼───────────────┼────────┤
│- Oversampling  │- Class       │- Adjust        │- F1   │
│- Undersampling │  weights     │  decision      │- AUC  │
│- SMOTE         │- Cost-       │  boundary      │- PR   │
│- ADASYN        │  sensitive   │                │  AUC  │
└───────────────┴───────────────┴───────────────┴────────┘
```

### 1. Data-Level Methods

**Undersampling (Reduce Majority)**

```python
from imblearn.under_sampling import RandomUnderSampler, TomekLinks

# Random undersampling
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

# Tomek Links (removes borderline majority samples)
tomek = TomekLinks()
X_res, y_res = tomek.fit_resample(X, y)
```

| Method | Pros | Cons |
|--------|------|------|
| Random | Fast, simple | Loses information |
| Tomek Links | Cleans boundary | Slight reduction only |
| ENN | Removes noisy majority | May not balance |

**Oversampling (Increase Minority)**

```python
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

# Random oversampling (duplicates)
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

# SMOTE (creates synthetic samples)
smote = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = smote.fit_resample(X, y)

# ADASYN (focuses on harder samples)
adasyn = ADASYN(random_state=42)
X_res, y_res = adasyn.fit_resample(X, y)
```

| Method | How It Works | Best For |
|--------|-------------|----------|
| Random | Duplicates minority | Simple baseline |
| SMOTE | Interpolates between neighbors | Continuous features |
| ADASYN | More samples near boundary | Hard-to-classify areas |
| BorderlineSMOTE | SMOTE only on border samples | Cleaner synthetic data |

**SMOTE Explained:**
```
1. Pick a minority sample
2. Find its k nearest minority neighbors
3. Create synthetic sample along line to random neighbor

    o ← original
    |
    x ← synthetic (random point on line)
    |
    o ← neighbor
```

**Combination Methods:**

```python
from imblearn.combine import SMOTETomek, SMOTEENN

# SMOTE + Tomek Links
smote_tomek = SMOTETomek(random_state=42)
X_res, y_res = smote_tomek.fit_resample(X, y)

# SMOTE + Edited Nearest Neighbors
smote_enn = SMOTEENN(random_state=42)
X_res, y_res = smote_enn.fit_resample(X, y)
```

### 2. Algorithm-Level Methods

**Class Weights:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Automatic weight calculation
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X, y)

# Manual weights
lr = LogisticRegression(class_weight={0: 1, 1: 10})  # 10x penalty for missing class 1
lr.fit(X, y)

# For neural networks
import numpy as np
class_weights = {0: 1., 1: len(y[y==0]) / len(y[y==1])}
# Use in model.fit(..., class_weight=class_weights)
```

**Cost-Sensitive Learning:**

Define different misclassification costs:

```
                  Predicted
               Negative  Positive
Actual Negative   0        1      (False alarm)
       Positive   10       0      (Missed fraud - COSTLY!)
```

### 3. Threshold Tuning

```python
from sklearn.metrics import precision_recall_curve

# Get probability predictions
y_proba = model.predict_proba(X_test)[:, 1]

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]

# Use custom threshold
y_pred = (y_proba >= best_threshold).astype(int)
```

### 4. Proper Metrics

**Don't use accuracy!** Use these instead:

| Metric | Formula | Best For |
|--------|---------|----------|
| **Precision** | TP / (TP + FP) | When FP is costly |
| **Recall** | TP / (TP + FN) | When FN is costly (fraud, disease) |
| **F1** | 2×P×R / (P+R) | Balance P and R |
| **AUC-ROC** | Area under ROC | Overall discrimination |
| **PR AUC** | Area under PR curve | Better for severe imbalance |

```python
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score,
    precision_recall_curve, average_precision_score
)

# Comprehensive report
print(classification_report(y_test, y_pred))

# Key metrics
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)
```

### Complete Pipeline Example

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Pipeline with SMOTE
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(class_weight='balanced'))
])

# Use stratified CV to preserve class ratio
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')
print(f"F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

### Decision Framework

```
Imbalanced Data?
       │
       ├── Mild (80-20)? → Class weights usually sufficient
       │
       ├── Moderate (95-5)? → SMOTE + Class weights
       │
       └── Severe (99-1)? → SMOTE + Ensemble + Threshold tuning

Always:
  • Use stratified splits
  • Evaluate with F1, AUC, PR-AUC
  • Never apply SMOTE to test set!
```

### Common Mistakes

| Mistake | Problem | Solution |
|---------|---------|----------|
| SMOTE before split | Data leakage | SMOTE only on training set |
| Using accuracy | Misleading metric | Use F1, AUC, PR-AUC |
| Random oversampling | Creates exact duplicates | Use SMOTE instead |
| Ignoring cost | Not all errors equal | Use cost-sensitive learning |

### Common Follow-up Questions
1. *"When would you NOT use SMOTE?"* - High-dimensional sparse data, when synthetic samples don't make sense (text)
2. *"What's the difference between ROC-AUC and PR-AUC?"* - PR-AUC is more informative for severe imbalance
3. *"Should you balance to 50-50?"* - Not necessarily; sometimes slight imbalance reflects reality better

### Interview Tip
Start by explaining why accuracy fails for imbalanced data. Know multiple strategies (data-level and algorithm-level). Emphasize that SMOTE should only be applied to training data. Always mention appropriate evaluation metrics (F1, PR-AUC).

---

## Question 21
- [ ] Done

**What is Linear Regression and its assumptions?**

**Answer:**

### Definition
Linear Regression models the relationship between a dependent variable (y) and one or more independent variables (X) by fitting a linear equation to observed data.

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon$$

Where:
- $\beta_0$ = intercept (y when all x = 0)
- $\beta_i$ = coefficients (effect of each feature)
- $\epsilon$ = error term (noise)

### Types

| Type | Features | Equation |
|------|----------|----------|
| Simple | 1 | $y = \beta_0 + \beta_1 x$ |
| Multiple | >1 | $y = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n$ |
| Polynomial | 1 (transformed) | $y = \beta_0 + \beta_1 x + \beta_2 x^2 + ...$ |

### How It Works: Ordinary Least Squares (OLS)

**Objective:** Minimize the sum of squared residuals

$$\min \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \min \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_i))^2$$

**Closed-form Solution:**
$$\hat{\beta} = (X^T X)^{-1} X^T y$$

### The 5 Assumptions (LINE + No Multicollinearity)

| Assumption | What It Means | How to Check | Consequence if Violated |
|------------|---------------|--------------|------------------------|
| **L**inearity | Relationship is linear | Residual vs fitted plot | Biased predictions |
| **I**ndependence | Errors are independent | Durbin-Watson test | Invalid inference |
| **N**ormality | Errors are normally distributed | Q-Q plot, Shapiro-Wilk | Invalid confidence intervals |
| **E**qual variance (Homoscedasticity) | Constant error variance | Residual plot, Breusch-Pagan | Inefficient estimates |
| **No Multicollinearity** | Features aren't highly correlated | VIF < 5 or 10 | Unstable coefficients |

### Checking Assumptions

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm

# Fit model
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
residuals = y - predictions

# 1. Linearity: Residuals vs Fitted
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted (should be random)')

# 2. Independence: Durbin-Watson (1.5-2.5 is good)
from statsmodels.stats.stattools import durbin_watson
print(f"Durbin-Watson: {durbin_watson(residuals)}")

# 3. Normality: Q-Q plot
stats.probplot(residuals, dist="norm", plot=plt)

# 4. Homoscedasticity: Scale-Location plot
from statsmodels.stats.diagnostic import het_breuschpagan
_, pval, _, _ = het_breuschpagan(residuals, X)
print(f"Breusch-Pagan p-value: {pval}")

# 5. Multicollinearity: VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(f"VIF: {vif}")  # Should be < 5 (or < 10)
```

### Interpretation of Coefficients

```python
model = LinearRegression()
model.fit(X, y)

print(f"Intercept: {model.intercept_}")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
    # Interpretation: 1 unit increase in feature -> coef unit change in y
```

**Example:** Price = 50000 + 100×sqft + 5000×bedrooms
- Base price: $50,000
- Each sqft adds: $100
- Each bedroom adds: $5,000

### When Assumptions Are Violated

| Violation | Solution |
|-----------|----------|
| Non-linearity | Add polynomial features, use non-linear model |
| Non-independence | Use time series models, add lag features |
| Non-normality | Transform y (log, Box-Cox), robust regression |
| Heteroscedasticity | Weighted least squares, transform y |
| Multicollinearity | Remove features, PCA, Ridge regression |

### Regularized Linear Regression

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge (L2) - handles multicollinearity
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

# Lasso (L1) - feature selection
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
print(f"Features selected: {(lasso.coef_ != 0).sum()}")

# ElasticNet - combination
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
```

### Evaluation Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")  # Proportion of variance explained
```

### Advantages and Limitations

| Advantages | Limitations |
|------------|-------------|
| Simple, interpretable | Assumes linearity |
| Fast training | Sensitive to outliers |
| No hyperparameters (OLS) | Assumes feature independence |
| Statistical inference (p-values) | Poor with non-linear relationships |
| Feature importance from coefficients | Multicollinearity issues |

### Common Follow-up Questions
1. *"What if R² is very high on training but low on test?"* - Overfitting, use regularization
2. *"Can Linear Regression handle categorical variables?"* - Yes, with one-hot encoding
3. *"What's the difference between R² and Adjusted R²?"* - Adjusted penalizes extra features

### Interview Tip
Be ready to explain assumptions and how to check them. Know that violating assumptions doesn't always mean the model is useless — predictions may still be good, but statistical inference (p-values, confidence intervals) becomes unreliable. Mention regularization as solution for multicollinearity.

---

## Question 22
- [ ] Done

**Explain the concept of Logistic Regression and its applications.**

**Answer:**

### Definition
Despite its name, Logistic Regression is a **classification** algorithm that predicts the probability of a binary outcome using the logistic (sigmoid) function.

### The Logistic Function (Sigmoid)

$$P(y=1|x) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Where $z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n$

```
Sigmoid Curve:

    1.0 ──────────────────────┬──────
                          │
    0.5 ───────────┼──────────┤
                 │
    0.0 ────────┴───────────────────
        -∞           0           +∞
                    z
```

### How It Works

**1. Linear Combination:**
$$z = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n$$

**2. Apply Sigmoid:**
$$P(y=1) = \frac{1}{1 + e^{-z}}$$

**3. Classify:**
$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$

### Log-Odds (Logit) Interpretation

Logistic regression models the **log-odds** as a linear function:

$$\log\left(\frac{P(y=1)}{1-P(y=1)}\right) = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n$$

**Coefficient Interpretation:**
- $e^{\beta_i}$ = odds ratio
- A 1-unit increase in $x_i$ multiplies the odds by $e^{\beta_i}$

### Loss Function: Binary Cross-Entropy

$$L = -\frac{1}{n}\sum_{i=1}^{n}[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$$

Minimized using gradient descent (no closed-form solution).

### Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Basic Logistic Regression
model = LogisticRegression(
    penalty='l2',           # Regularization type
    C=1.0,                   # Inverse of regularization strength
    solver='lbfgs',          # Optimization algorithm
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities

# Evaluation
print(classification_report(y_test, y_pred))

# Coefficients (for interpretation)
for feature, coef in zip(X.columns, model.coef_[0]):
    odds_ratio = np.exp(coef)
    print(f"{feature}: coef={coef:.3f}, odds_ratio={odds_ratio:.3f}")
```

### Multiclass Logistic Regression

```python
# One-vs-Rest (OvR): K binary classifiers
model_ovr = LogisticRegression(multi_class='ovr')

# Multinomial (Softmax): Single multiclass classifier
model_softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs')
```

| Method | How It Works | When to Use |
|--------|-------------|-------------|
| One-vs-Rest | K binary classifiers | Large K, simpler |
| Multinomial | Softmax, joint probability | Better when classes related |

### Regularization in Logistic Regression

```python
# L2 regularization (default)
model_l2 = LogisticRegression(penalty='l2', C=0.1)  # Higher C = less regularization

# L1 regularization (feature selection)
model_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)

# ElasticNet
model_en = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)
```

### Applications

| Application | What It Predicts |
|-------------|------------------|
| Medical diagnosis | Disease probability |
| Credit scoring | Default probability |
| Marketing | Conversion probability |
| Fraud detection | Fraud probability |
| Churn prediction | Churn probability |

### Advantages and Limitations

| Advantages | Limitations |
|------------|-------------|
| Outputs probabilities | Assumes linear decision boundary |
| Highly interpretable | Struggles with non-linear relationships |
| Fast training & prediction | Assumes feature independence |
| Works well with regularization | Sensitive to outliers |
| No assumption about feature distribution | Requires feature scaling |

### Logistic vs Linear Regression

| Aspect | Linear Regression | Logistic Regression |
|--------|------------------|--------------------|
| Output | Continuous | Probability [0, 1] |
| Use case | Regression | Classification |
| Loss function | MSE | Cross-entropy |
| Optimization | Closed-form (OLS) | Gradient descent |
| Output range | (-∞, +∞) | (0, 1) |

### Common Follow-up Questions
1. *"Why not use Linear Regression for classification?"* - Outputs can be <0 or >1, no probability interpretation
2. *"What does the C parameter do?"* - Inverse of regularization strength; lower C = more regularization
3. *"How do you choose the threshold?"* - Based on business needs; optimize for precision/recall/F1

### Interview Tip
Emphasize that despite the name, it's for classification. Know how to interpret coefficients as odds ratios. Mention that it outputs probabilities, which is valuable for ranking or when you need calibrated scores. Be ready to explain the difference from linear regression and when to use each.

---

## Question 23
- [ ] Done

**What is Decision Tree and how does it work?**

**Answer:**

### Definition
A Decision Tree is a supervised learning algorithm that makes predictions by learning a series of if-then-else decision rules from the data. It creates a tree-like structure where:
- **Internal nodes**: Feature tests (decisions)
- **Branches**: Outcomes of tests
- **Leaf nodes**: Final predictions

### Visual Example

```
                    [Age < 30?]
                    /         \
                  Yes          No
                  /             \
          [Income > 50K?]    [Credit Good?]
          /         \         /         \
        Yes         No      Yes         No
        /            \       /            \
    [APPROVE]    [DENY]  [APPROVE]     [DENY]
```

### How It Works: Recursive Partitioning

```
1. Start with entire dataset at root
2. Find the best feature and threshold to split
3. Split data into subsets
4. Repeat recursively for each subset
5. Stop when stopping criteria met (max depth, min samples, pure node)
```

### Splitting Criteria

**For Classification:**

| Criterion | Formula | Use Case |
|-----------|---------|----------|
| Gini Impurity | $1 - \sum_{i=1}^{k} p_i^2$ | Default in sklearn |
| Entropy | $-\sum_{i=1}^{k} p_i \log_2(p_i)$ | Information gain |
| Misclassification | $1 - \max(p_i)$ | Rarely used |

**Gini Impurity Example:**
```
Node with 30 Yes, 70 No:
Gini = 1 - (0.3² + 0.7²) = 1 - 0.58 = 0.42

Pure node (all Yes): Gini = 0
Maximum impurity (50-50): Gini = 0.5
```

**For Regression:**
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

### Best Split Selection

```
For each feature:
    For each possible threshold:
        Split data
        Calculate weighted impurity of children
        Track if this is the best split seen

Best Split = Argmin(weighted impurity)

Information Gain = Parent Impurity - Weighted Child Impurity
```

### Implementation

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Classification
clf = DecisionTreeClassifier(
    criterion='gini',      # or 'entropy'
    max_depth=5,            # Maximum tree depth
    min_samples_split=10,   # Minimum samples to split a node
    min_samples_leaf=5,     # Minimum samples in a leaf
    max_features=None,      # Features to consider per split
    random_state=42
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Visualize the tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'],
          filled=True, rounded=True)
plt.show()

# Feature importance
importances = pd.Series(clf.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False))
```

### Preventing Overfitting (Pruning)

**Pre-pruning (Constraints during training):**

| Parameter | Effect |
|-----------|--------|
| `max_depth` | Limits tree depth |
| `min_samples_split` | Minimum samples to create a split |
| `min_samples_leaf` | Minimum samples in leaf nodes |
| `max_features` | Features considered per split |
| `max_leaf_nodes` | Maximum number of leaves |

**Post-pruning (Cost-complexity pruning):**

```python
# Find optimal alpha using cross-validation
path = clf.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas

# Test different alphas
scores = []
for alpha in alphas:
    clf_temp = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    clf_temp.fit(X_train, y_train)
    scores.append(clf_temp.score(X_test, y_test))

# Use best alpha
best_alpha = alphas[np.argmax(scores)]
clf_pruned = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
```

### Advantages and Limitations

| Advantages | Limitations |
|------------|-------------|
| Easy to understand & visualize | Prone to overfitting |
| No feature scaling needed | Unstable (small data changes → different tree) |
| Handles non-linear relationships | Can create biased trees with imbalanced data |
| Handles mixed feature types | Greedy algorithm (not globally optimal) |
| Built-in feature selection | Struggles with XOR-type relationships |
| Fast prediction | High variance |

### Decision Trees vs Other Algorithms

| Aspect | Decision Tree | Linear Models |
|--------|--------------|---------------|
| Decision boundary | Non-linear, axis-parallel | Linear |
| Feature scaling | Not needed | Required |
| Interpretability | High (visual) | Medium (coefficients) |
| Overfitting risk | High | Low |
| Handles interactions | Naturally | Need to specify |

### Common Follow-up Questions
1. *"Why are decision trees prone to overfitting?"* - Can create very deep trees that memorize training data
2. *"What's the difference between Gini and Entropy?"* - Mathematically similar; Gini is slightly faster, Entropy can create more balanced trees
3. *"How does Random Forest improve on Decision Trees?"* - Averages many trees to reduce variance

### Interview Tip
Explain the greedy nature of the algorithm (makes locally optimal choices). Know how to prevent overfitting with pruning. Mention that single trees are interpretable but have high variance — that's why ensembles (Random Forest, XGBoost) are preferred for accuracy. Be ready to draw a simple tree example.

---

## Question 24
- [ ] Done

**Describe the concept of Random Forest and its advantages over Decision Trees.**

**Answer:**

### Definition
Random Forest is an ensemble learning method that builds multiple decision trees and combines their predictions. It uses **bagging** (bootstrap aggregating) and **random feature selection** to create diverse trees.

### How It Works

```
┌────────────────────────────────────────────────┐
│              Training Data                       │
└────────────────┬───────────────┬───────────────┘
                │               │               │
         Bootstrap #1    Bootstrap #2    Bootstrap #3
         (with repl.)    (with repl.)    (with repl.)
                │               │               │
                ↓               ↓               ↓
        ┌─────────┐     ┌─────────┐     ┌─────────┐
        │  Tree 1  │     │  Tree 2  │     │  Tree 3  │
        │ (random  │     │ (random  │     │ (random  │
        │ features)│     │ features)│     │ features)│
        └────┬────┘     └────┬────┘     └────┬────┘
             │               │               │
             ↓               ↓               ↓
         Prediction      Prediction      Prediction
                \             │              /
                 \            │             /
                  └───────────┴──────────┘
                              │
                    ┌───────────────────┐
                    │   Aggregation      │
                    │ (Vote/Average)     │
                    └───────────────────┘
                              │
                    Final Prediction
```

### The Two Sources of Randomness

**1. Bootstrap Sampling (Row Randomness):**
- Each tree trained on random sample WITH replacement
- ~63% of data used per tree (on average)
- ~37% left out (Out-of-Bag samples)

**2. Random Feature Selection (Column Randomness):**
- Each split considers only a random subset of features
- Classification: $\sqrt{n\_features}$ by default
- Regression: $n\_features / 3$ by default

### Why It Works: Variance Reduction

If trees are uncorrelated with variance $\sigma^2$, the ensemble variance is:

$$\text{Var}(\text{ensemble}) = \frac{\sigma^2}{n\_trees}$$

Randomness makes trees different, averaging reduces variance without increasing bias.

### Implementation

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Classification
rf_clf = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=None,          # Maximum depth (None = unlimited)
    min_samples_split=2,     # Min samples to split
    min_samples_leaf=1,      # Min samples in leaf
    max_features='sqrt',     # Features per split
    bootstrap=True,          # Use bootstrap sampling
    oob_score=True,          # Calculate out-of-bag score
    n_jobs=-1,               # Use all CPUs
    random_state=42
)

rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
y_proba = rf_clf.predict_proba(X_test)

# Out-of-bag score (free validation!)
print(f"OOB Score: {rf_clf.oob_score_}")
```

### Feature Importance

```python
# Gini/MSE-based importance (built-in)
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

# Visualization
importances.head(10).plot(kind='barh', x='feature', y='importance')

# Permutation importance (more reliable)
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(rf_clf, X_test, y_test, n_repeats=10)
perm_df = pd.DataFrame({
    'feature': X.columns,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False)
```

### Hyperparameter Tuning

| Parameter | Description | Tuning Guidance |
|-----------|-------------|----------------|
| `n_estimators` | Number of trees | More is better (diminishing returns after ~100-500) |
| `max_depth` | Tree depth | Start None, reduce if overfitting |
| `min_samples_split` | Min samples to split | Increase to reduce overfitting |
| `min_samples_leaf` | Min samples in leaf | Increase to reduce overfitting |
| `max_features` | Features per split | 'sqrt' (clf), 'auto' (reg), or tune |
| `max_samples` | Bootstrap sample size | Usually leave default |

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.3]
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(rf, param_dist, n_iter=20, cv=5, 
                                    scoring='f1', random_state=42)
random_search.fit(X_train, y_train)
print(f"Best params: {random_search.best_params_}")
```

### Out-of-Bag (OOB) Score

Free cross-validation!

```
Tree 1 trained on: [1,2,3,5,5,7] → Predicts for samples 4,6,8,...
Tree 2 trained on: [2,3,4,6,6,8] → Predicts for samples 1,5,7,...

OOB prediction = Average of predictions from trees that didn't use that sample
```

### Random Forest vs Decision Tree

| Aspect | Decision Tree | Random Forest |
|--------|--------------|---------------|
| Variance | High | Low (averaged) |
| Bias | Low | Low |
| Overfitting | Prone | Resistant |
| Interpretability | High | Lower |
| Training time | Fast | Slower (but parallelizable) |
| Prediction time | Fast | Slower |

### When to Use Random Forest

**Good for:**
- First baseline model (often works well out-of-box)
- Tabular data with mixed feature types
- When interpretability matters (feature importance)
- When you need probability estimates

**Consider alternatives when:**
- Need maximum accuracy (try XGBoost/LightGBM)
- Very high-dimensional sparse data (consider linear models)
- Real-time prediction with latency constraints
- Need to deploy with minimal memory

### Common Follow-up Questions
1. *"Why is Random Forest less prone to overfitting?"* - Averaging many trees reduces variance; bootstrap creates diversity
2. *"What's the difference between bagging and boosting?"* - Bagging: parallel, reduces variance; Boosting: sequential, reduces bias
3. *"Can Random Forest capture feature interactions?"* - Yes, naturally through tree structure

### Interview Tip
Explain both sources of randomness (bootstrap + feature subsampling). Mention OOB score as free validation. Know that it's often the best "first try" algorithm for tabular data. Be ready to compare with boosting methods (XGBoost, LightGBM).

---

## Question 25
- [ ] Done

**What is Support Vector Machine (SVM) and its kernel functions?**

**Answer:**

### Definition
Support Vector Machine (SVM) is a supervised learning algorithm that finds the optimal hyperplane that maximally separates classes. It works by maximizing the **margin** — the distance between the hyperplane and the nearest data points (support vectors).

### Key Concepts

```
          Support Vectors
               ↓     ↓
    +  +      +|     |o      o  o
      +    +   |     |    o
    +   +     +|     |o     o   o
               |     |
    +    +    +|     |  o    o
               ───────
             Maximum
              Margin
               
    ├─── Margin ───┤
```

**Support Vectors:** Data points closest to the decision boundary that define the margin.

### Mathematical Formulation

**Hard Margin (linearly separable):**

$$\min_{w,b} \frac{1}{2}||w||^2$$

Subject to: $y_i(w^T x_i + b) \geq 1$ for all $i$

**Soft Margin (allows some misclassification):**

$$\min_{w,b,\xi} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\xi_i$$

Where:
- $\xi_i$ = slack variables (allow misclassification)
- $C$ = regularization parameter (penalty for misclassification)

### The C Parameter

```
Small C (C=0.01):              Large C (C=100):
Wide margin, some errors        Narrow margin, few errors

    +  o  +     |          +    |  o
  +    +   o    |        +  +   |    o
    +      o    |          +    |  o
                |               |
More regularization         Less regularization
May underfit               May overfit
```

### Kernel Functions (The Kernel Trick)

Kernels allow SVM to handle non-linearly separable data by implicitly mapping to higher dimensions.

| Kernel | Formula | Use Case |
|--------|---------|----------|
| Linear | $K(x_i, x_j) = x_i^T x_j$ | Linearly separable, high-D text |
| RBF (Gaussian) | $K(x_i, x_j) = e^{-\gamma ||x_i - x_j||^2}$ | Non-linear, default choice |
| Polynomial | $K(x_i, x_j) = (\gamma x_i^T x_j + r)^d$ | Feature interactions |
| Sigmoid | $K(x_i, x_j) = \tanh(\gamma x_i^T x_j + r)$ | Similar to neural networks |

### The RBF Kernel and Gamma

```
Small gamma:                    Large gamma:
Smooth boundary                 Complex boundary

    +++++++                       ++
   +++++++++                     + ++
  +++++|ooooo                   + + |ooo
   ++++|ooooo                    ++ |o o
    +++|ooooo                      +|ooo
                                
Underfitting risk              Overfitting risk
```

- **Low gamma:** Far points have influence, smoother boundary
- **High gamma:** Only close points matter, complex boundary

### Implementation

```python
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# IMPORTANT: Always scale features for SVM!
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(
        kernel='rbf',      # 'linear', 'poly', 'rbf', 'sigmoid'
        C=1.0,             # Regularization (higher = less regularization)
        gamma='scale',     # 'scale', 'auto', or float
        probability=True,  # Enable probability estimates
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)

# Access the SVM model
svm_model = pipeline.named_steps['svm']
print(f"Number of support vectors: {svm_model.n_support_}")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.1, 0.01],
    'svm__kernel': ['rbf', 'linear', 'poly']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
```

### SVM for Regression (SVR)

```python
from sklearn.svm import SVR

svr = SVR(
    kernel='rbf',
    C=1.0,
    epsilon=0.1  # Tube width (no penalty within epsilon)
)
svr.fit(X_train, y_train)
```

### Advantages and Limitations

| Advantages | Limitations |
|------------|-------------|
| Effective in high dimensions | Slow on large datasets O(n²) to O(n³) |
| Memory efficient (uses support vectors) | Requires feature scaling |
| Versatile (different kernels) | Sensitive to kernel/hyperparameter choice |
| Works well with clear margins | Doesn't provide probabilities directly |
| Robust to overfitting in high-D | Not suitable for very large datasets |

### When to Use SVM

**Good for:**
- Text classification (high-D, sparse)
- Image classification
- Small to medium datasets
- Binary classification with clear margins

**Avoid when:**
- Very large datasets (>100K samples)
- Noisy data with overlapping classes
- Need for probabilistic output (slow with `probability=True`)
- Need interpretability

### SVM vs Other Algorithms

| Aspect | SVM | Logistic Regression | Random Forest |
|--------|-----|--------------------|--------------|
| Decision boundary | Non-linear (kernel) | Linear | Non-linear |
| Scaling needed | Yes | Yes | No |
| Speed (large data) | Slow | Fast | Medium |
| Interpretability | Low | High | Medium |
| Probabilistic | With overhead | Native | Native |

### Common Follow-up Questions
1. *"What are support vectors?"* - Data points that lie on the margin boundary; they define the decision boundary
2. *"Why is RBF the default kernel?"* - Works well for most non-linear problems, only one hyperparameter (gamma)
3. *"When would you use linear kernel?"* - High-dimensional data (text), linearly separable data, faster training

### Interview Tip
Explain the margin maximization concept and why support vectors are important (only they affect the decision boundary). Know when to use which kernel: linear for high-D/text, RBF for general non-linear. Emphasize that scaling is mandatory. Mention the quadratic complexity issue for large datasets.

---

## Question 26
- [ ] Done

**Explain the concept of Naive Bayes algorithm and its types.**

**Answer:**

### Definition
Probabilistic classifier using Bayes theorem with independence assumption.
$$P(y|X) \propto P(y) \prod P(x_i|y)$$

### Types
- Gaussian: Continuous features
- Multinomial: Counts (text)
- Bernoulli: Binary features

### Interview Tip
Works surprisingly well despite naive assumption.

---

## Question 27
- [ ] Done

**What is K-Nearest Neighbors (KNN) algorithm and its distance metrics?**

**Answer:**

### Definition
Classifies based on majority vote of k nearest neighbors.

### Distance Metrics
- Euclidean: Most common
- Manhattan: Grid-like paths
- Minkowski: Generalized

### Considerations
- Choose k (odd for binary)
- Scale features
- Curse of dimensionality

### Interview Tip
Simple baseline, but slow for large datasets.

---

## Question 28
- [ ] Done

**What is XGBoost and its key features?**

**Answer:**

### Key Features
- Gradient boosting with regularization
- Parallel processing
- Handles missing values
- Tree pruning
- Built-in cross-validation

### Why Popular
- High accuracy
- Fast training
- Feature importance

### Interview Tip
XGBoost often wins ML competitions.

---

## Question 29
- [ ] Done

**Explain the concept of Stacking and its benefits in Ensemble Learning.**

**Answer:**

### Definition
Uses predictions from multiple models as features for a meta-model.

### Architecture
```
Model 1 ─┐
Model 2 ─┼→ Meta-model → Final prediction
Model 3 ─┘
```

### Benefits
Combines diverse models' strengths.

### Interview Tip
Use diverse base models for best results.

---

## Question 30
- [ ] Done

**What is K-Means Clustering and its objective function?**

**Answer:**

### Objective
Minimize within-cluster sum of squares:
$$\sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

### Algorithm
1. Initialize k centroids
2. Assign points to nearest centroid
3. Update centroids
4. Repeat until convergence

### Interview Tip
Use elbow method to choose k.

---

## Question 31
- [ ] Done

**Explain the difference between Hierarchical and Partitional Clustering.**

**Answer:**

### Comparison

| Hierarchical | Partitional |
|--------------|-------------|
| Tree structure | Flat clusters |
| No k needed | Specify k |
| Agglomerative/Divisive | K-Means, K-Medoids |
| Dendrogram | Direct assignment |

### Interview Tip
Hierarchical for unknown k, partitional for large data.

---

## Question 32
- [ ] Done

**What is Density-Based Spatial Clustering of Applications with Noise (DBSCAN) and its parameters?**

**Answer:**

### Parameters
- **eps**: Neighborhood radius
- **min_samples**: Minimum points for core

### Advantages
- Finds arbitrary shapes
- Handles noise
- No need to specify k

### Interview Tip
DBSCAN is great for spatial data with outliers.

---

## Question 33
- [ ] Done

**Describe the concept of Gaussian Mixture Models (GMM) and its applications.**

**Answer:**

### Definition
Probabilistic model assuming data is from mixture of Gaussians.

### Algorithm
EM (Expectation-Maximization) to fit parameters.

### Applications
- Soft clustering
- Density estimation
- Anomaly detection

### Interview Tip
GMM gives probability of cluster membership.

---

## Question 34
- [ ] Done

**What is Principal Component Analysis (PCA) and its role in unsupervised learning?**

**Answer:**

### Role in Unsupervised Learning
- Dimensionality reduction
- Visualization (2D/3D)
- Noise reduction
- Feature extraction

### When to Use
- High-dimensional data
- Multicollinearity
- Before clustering/visualization

### Interview Tip
Choose components explaining 95% variance.

---

## Question 35
- [ ] Done

**Explain the concept of t-Distributed Stochastic Neighbor Embedding (t-SNE) and its use cases.**

**Answer:**

### Definition
Non-linear dimensionality reduction for visualization.

### Use Cases
- Visualizing high-dimensional data
- Exploring clusters
- Embeddings visualization

### Limitations
- Computationally expensive
- Non-deterministic
- Distances not meaningful

### Interview Tip
Use for visualization only, not preprocessing.

---

## Question 36
- [ ] Done

**What is Association Rule Mining and its popular algorithms?**

**Answer:**

### Definition
Finds relationships between items in transactional data.

### Metrics
- Support: Frequency
- Confidence: P(B|A)
- Lift: Strength of association

### Algorithms
Apriori, FP-Growth

### Interview Tip
Market basket analysis is the classic application.

---

## Question 37
- [ ] Done

**Describe the concept of Anomaly Detection and its techniques.**

**Answer:**

### Techniques
- **Statistical**: Z-score, IQR
- **Proximity**: LOF, k-distance
- **Reconstruction**: Autoencoders
- **Isolation**: Isolation Forest

### Applications
Fraud detection, system monitoring, quality control.

### Interview Tip
Define "normal" carefully based on domain.

---

## Question 38
- [ ] Done

**What is Self-Organizing Maps (SOM) and its applications?**

**Answer:**

### Definition
Neural network that maps high-dimensional data to 2D grid.

### Applications
- Data visualization
- Clustering
- Feature extraction

### Interview Tip
SOM preserves topological relationships.

---

## Question 39
- [ ] Done

**Explain the concept of Latent Dirichlet Allocation (LDA) in topic modeling.**

**Answer:**

### Definition
Probabilistic model where:
- Documents are mixtures of topics
- Topics are mixtures of words

### Use Cases
- Topic discovery
- Document clustering
- Content recommendation

### Interview Tip
LDA is unsupervised; NMF is an alternative.

---

## Question 40
- [ ] Done

**What is the purpose of Model Evaluation and Validation in ML?**

**Answer:**

### Purpose
- Assess model performance
- Compare models
- Detect overfitting
- Estimate production performance

### Methods
Train/validation/test split, cross-validation, holdout.

### Interview Tip
Never evaluate on training data alone.

---

## Question 41
- [ ] Done

**Explain the difference between Train, Validation, and Test sets.**

**Answer:**

### Roles

| Set | Purpose |
|-----|---------|
| Train | Fit model parameters |
| Validation | Tune hyperparameters |
| Test | Final evaluation |

### Typical Split
60/20/20 or 70/15/15

### Interview Tip
Never touch test set until final evaluation.

---

## Question 42
- [ ] Done

**What is Confusion Matrix and its components?**

**Answer:**

### Definition
A Confusion Matrix is a table that visualizes the performance of a classification model by comparing predicted labels against actual labels. It forms the foundation for many classification metrics.

### Structure (Binary Classification)

```
                        Predicted
                   Negative    Positive
                 ┌──────────┬──────────┐
    Actual   Neg │    TN     │    FP     │
                 │          │ (Type I)  │
                 ├──────────┼──────────┤
             Pos │    FN     │    TP     │
                 │ (Type II) │          │
                 └──────────┴──────────┘
```

### Components Explained

| Component | Full Name | Meaning | Example (Disease Detection) |
|-----------|-----------|---------|-----------------------------|
| **TP** | True Positive | Correctly predicted positive | Sick patient correctly diagnosed |
| **TN** | True Negative | Correctly predicted negative | Healthy patient correctly cleared |
| **FP** | False Positive | Incorrectly predicted positive | Healthy patient wrongly diagnosed |
| **FN** | False Negative | Incorrectly predicted negative | Sick patient missed |

### Error Types

| Error | Also Called | Meaning | Cost Example |
|-------|-------------|---------|-------------|
| **FP** | Type I Error, False Alarm | Predicted Yes, Actually No | Unnecessary treatment |
| **FN** | Type II Error, Miss | Predicted No, Actually Yes | Missed disease, death |

### All Derived Metrics

```python
# From the confusion matrix, we can calculate:

# Accuracy: Overall correctness
Accuracy = (TP + TN) / (TP + TN + FP + FN)

# Precision: Of predicted positives, how many are correct?
Precision = TP / (TP + FP)

# Recall (Sensitivity, TPR): Of actual positives, how many did we catch?
Recall = TP / (TP + FN)

# Specificity (TNR): Of actual negatives, how many did we correctly identify?
Specificity = TN / (TN + FP)

# F1 Score: Harmonic mean of Precision and Recall
F1 = 2 * (Precision * Recall) / (Precision + Recall)

# False Positive Rate: Of actual negatives, how many did we wrongly flag?
FPR = FP / (FP + TN) = 1 - Specificity

# False Negative Rate: Of actual positives, how many did we miss?
FNR = FN / (FN + TP) = 1 - Recall
```

### Implementation

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Get confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# [[TN, FP],
#  [FN, TP]]

# Extract components
tn, fp, fn, tp = cm.ravel()
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

# Visualize
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Normalized (percentages)
cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, 
                               display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues', values_format='.2%')
```

### Multi-class Confusion Matrix

```
                    Predicted
              Cat    Dog    Bird
         ┌──────┬──────┬──────┐
Actual Cat  │  45  │   3  │   2  │
         ├──────┼──────┼──────┤
     Dog  │   5  │  40  │   5  │
         ├──────┼──────┼──────┤
     Bird │   2  │   4  │  44  │
         └──────┴──────┴──────┘

Diagonal = Correct predictions
Off-diagonal = Misclassifications
```

### Reading the Matrix: Common Patterns

| Pattern | Indication |
|---------|------------|
| High diagonal, low off-diagonal | Good model |
| High FP | Model is too aggressive (predicting too many positives) |
| High FN | Model is too conservative (missing positives) |
| Confusion between specific classes | Those classes are similar |

### Real-World Example: Spam Detection

```
                 Predicted
              Not Spam    Spam
         ┌─────────┬─────────┐
Actual Not │  950    │   10    │  ← 10 good emails in spam (annoying)
    Spam  ├─────────┼─────────┤
     Spam │   5     │   35    │  ← 5 spam in inbox (bad!)
         └─────────┴─────────┘

Accuracy = (950 + 35) / 1000 = 98.5%
Precision (Spam) = 35 / 45 = 77.8%
Recall (Spam) = 35 / 40 = 87.5%
```

### Common Follow-up Questions
1. *"Which is worse: FP or FN?"* - Depends on context! Medical: FN worse (missed disease). Spam: FP worse (lost important email)
2. *"What if classes are imbalanced?"* - Look at precision/recall, not accuracy; use normalized confusion matrix
3. *"How do you improve high FN?"* - Lower decision threshold, or use class weights

### Interview Tip
Be able to quickly derive all metrics from the confusion matrix. Know the difference between Type I and Type II errors with real examples. Explain that the "positive" class should be the one you care about detecting (disease, fraud, spam). Always ask about the relative cost of FP vs FN in business context.

---

## Question 43
- [ ] Done

**Describe the concept of Precision, Recall, and F1-Score.**

**Answer:**

### Overview

| Metric | Formula | Question It Answers |
|--------|---------|--------------------|
| **Precision** | $\frac{TP}{TP + FP}$ | "Of all predicted positives, how many are actually positive?" |
| **Recall** | $\frac{TP}{TP + FN}$ | "Of all actual positives, how many did we find?" |
| **F1-Score** | $2 \times \frac{Precision \times Recall}{Precision + Recall}$ | "What's the balance between Precision and Recall?" |

### Visual Understanding

```
┌─────────────────────────────────────────────┐
│                All Samples                  │
│  ┌───────────────────┐  ┌────────────────┐ │
│  │  Predicted Pos   │  │  Actual Positive │ │
│  │                   │  │                  │ │
│  │    ┌───────────────────┴──┐               │ │
│  │    │     TP         │ │               │ │
│  │    │                │ │               │ │
│  │    └────────────────┬───┴───────────────┘ │
│  │         FP       │   FN (missed)       │
│  └───────────────────┘                       │
│                                             │
└─────────────────────────────────────────────┘

Precision = TP / (TP + FP)   ← Out of predicted positives
Recall = TP / (TP + FN)      ← Out of actual positives
```

### Precision Deep Dive

**"When the model says YES, how often is it right?"**

$$Precision = \frac{True\ Positives}{All\ Predicted\ Positives} = \frac{TP}{TP + FP}$$

**High Precision = Low False Positive Rate**

Optimize for precision when:
- False positives are costly
- You want to be confident when predicting positive

**Examples:**
- Spam filter: Don't want to lose important emails (FP is bad)
- Recommender system: Don't want to annoy users with bad recommendations

### Recall Deep Dive

**"Of all actual positives, how many did we catch?"**

$$Recall = \frac{True\ Positives}{All\ Actual\ Positives} = \frac{TP}{TP + FN}$$

**High Recall = Low False Negative Rate**

Optimize for recall when:
- Missing positives is costly
- You want to find all positive cases

**Examples:**
- Cancer detection: Don't want to miss a sick patient (FN is bad)
- Fraud detection: Don't want to miss fraudulent transactions
- Search engines: Don't want to miss relevant documents

### The Precision-Recall Trade-off

```
   Threshold ↓       Threshold ↑
   (predict more     (predict fewer
    positives)        positives)
        ↓                 ↓
   Recall ↑           Recall ↓
   Precision ↓        Precision ↑
```

**Why the trade-off exists:**

```python
# Probabilities from model
y_proba = [0.2, 0.4, 0.5, 0.7, 0.9]

# Threshold = 0.5: Predict [0, 0, 1, 1, 1] → 3 positives
# Threshold = 0.3: Predict [0, 1, 1, 1, 1] → 4 positives (higher recall, maybe lower precision)
# Threshold = 0.8: Predict [0, 0, 0, 0, 1] → 1 positive (higher precision, lower recall)
```

### F1-Score: The Harmonic Mean

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} = \frac{2TP}{2TP + FP + FN}$$

**Why harmonic mean?**
- Penalizes extreme values
- F1 is high only if BOTH precision and recall are high
- Arithmetic mean of 1.0 and 0.0 = 0.5, but harmonic mean = 0

| Precision | Recall | Arithmetic Mean | F1 (Harmonic) |
|-----------|--------|-----------------|---------------|
| 1.0 | 0.0 | 0.50 | 0.00 |
| 0.9 | 0.1 | 0.50 | 0.18 |
| 0.8 | 0.8 | 0.80 | 0.80 |
| 0.6 | 0.6 | 0.60 | 0.60 |

### F-beta Score: Weighted Harmonic Mean

$$F_\beta = (1 + \beta^2) \times \frac{Precision \times Recall}{(\beta^2 \times Precision) + Recall}$$

| Score | β | Emphasis |
|-------|---|----------|
| F0.5 | 0.5 | Precision 2x more important |
| F1 | 1.0 | Equal importance |
| F2 | 2.0 | Recall 2x more important |

```python
from sklearn.metrics import fbeta_score

f1 = fbeta_score(y_test, y_pred, beta=1.0)
f2 = fbeta_score(y_test, y_pred, beta=2.0)   # Recall-focused
f05 = fbeta_score(y_test, y_pred, beta=0.5)  # Precision-focused
```

### Implementation

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, precision_recall_curve
)

# Individual metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Complete report
print(classification_report(y_test, y_pred))

# For multi-class
f1_macro = f1_score(y_test, y_pred, average='macro')   # Unweighted mean
f1_micro = f1_score(y_test, y_pred, average='micro')   # Global TP/FP/FN
f1_weighted = f1_score(y_test, y_pred, average='weighted')  # Weighted by support

# Precision-Recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
```

### Multiclass Averaging

| Method | How It Works | When to Use |
|--------|-------------|-------------|
| `macro` | Average across classes (equal weight) | All classes equally important |
| `micro` | Global TP/FP/FN | Large datasets |
| `weighted` | Weight by class frequency | Imbalanced classes |

### Precision-Recall Curve

```python
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AP = {ap:.2f})')
plt.show()
```

### Decision Guide: Precision vs Recall

| Scenario | Priority | Why |
|----------|----------|-----|
| Medical screening | Recall | Don't miss sick patients |
| Email spam | Precision | Don't lose important emails |
| Fraud detection | Recall | Catch all fraud |
| Criminal conviction | Precision | "Innocent until proven guilty" |
| Search engines | Recall | Show all relevant results |
| Product recommendations | Precision | Don't annoy users |

### Common Follow-up Questions
1. *"When would you optimize for F2 vs F0.5?"* - F2 when missing positives is costly (disease), F0.5 when false alarms are costly (spam)
2. *"How do you choose the right threshold?"* - Plot precision-recall curve, choose based on business requirements
3. *"What's the problem with accuracy?"* - With imbalanced data, high accuracy can mean predicting all negatives

### Interview Tip
Always ask about the business context: "What's more costly — false positives or false negatives?" This determines whether to optimize for precision or recall. Know how to adjust the threshold to trade off between them. Explain why F1 uses harmonic mean (penalizes extreme imbalance).

---

## Question 44
- [ ] Done

**What is Receiver Operating Characteristic (ROC) Curve and its interpretation?**

**Answer:**

### Definition
The ROC (Receiver Operating Characteristic) curve is a graphical plot that shows the diagnostic ability of a binary classifier as its discrimination threshold is varied. It plots:
- **Y-axis**: True Positive Rate (TPR) = Recall = Sensitivity
- **X-axis**: False Positive Rate (FPR) = 1 - Specificity

### Key Metrics

$$TPR = \frac{TP}{TP + FN} = \text{Recall}$$

$$FPR = \frac{FP}{FP + TN} = 1 - \text{Specificity}$$

### Visual Understanding

```
    1.0───────┬──────────────────────
     T  │          .............Perfect
     P  │        .
     R  │      .     Good Model
        │     .
     0.5│   .      ........... Random
        │  .     .          (diagonal)
        │ .    .
        │.   .
    0.0 │...
        └────────────────────────────
       0.0                          1.0
                    FPR
```

### Key Points on the Curve

| Point | Location | Threshold | Meaning |
|-------|----------|-----------|----------|
| (0, 0) | Bottom-left | 1.0 | Predict all negative |
| (1, 1) | Top-right | 0.0 | Predict all positive |
| (0, 1) | Top-left | Perfect | Perfect classifier |
| Diagonal | y = x | Random | Random guessing |

### How the Curve is Created

```
1. Get probability scores from model
2. Try every possible threshold (0 to 1)
3. At each threshold:
   - Calculate TPR and FPR
   - Plot the point
4. Connect all points
```

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
```

### Interpreting the Curve

| Curve Position | Model Quality |
|---------------|---------------|
| Hugs top-left corner | Excellent |
| Above diagonal | Better than random |
| On diagonal | Random guessing |
| Below diagonal | Worse than random (flip predictions!) |

### Choosing the Optimal Threshold

```python
# Youden's J statistic (maximize TPR - FPR)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"TPR at optimal: {tpr[optimal_idx]:.3f}")
print(f"FPR at optimal: {fpr[optimal_idx]:.3f}")

# Mark on plot
plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
         label=f'Optimal (threshold={optimal_threshold:.2f})')
```

### Comparing Multiple Models

```python
models = {'Logistic': lr_proba, 'RF': rf_proba, 'XGB': xgb_proba}

plt.figure(figsize=(10, 8))
for name, proba in models.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()
```

### ROC vs Precision-Recall Curve

| Aspect | ROC | Precision-Recall |
|--------|-----|------------------|
| Y-axis | TPR (Recall) | Precision |
| X-axis | FPR | Recall |
| Class imbalance | Can be misleading | More informative |
| Random baseline | Diagonal line | Depends on class ratio |
| Focus | Overall discrimination | Positive class |

**When to use which:**
- **ROC**: Balanced classes, care about both classes equally
- **PR**: Imbalanced classes, care mostly about positive class

### Multiclass ROC

```python
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# One-vs-Rest approach
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_proba = model.predict_proba(X_test)

# Calculate ROC for each class
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
```

### Advantages and Limitations

| Advantages | Limitations |
|------------|-------------|
| Threshold-independent | Misleading for imbalanced data |
| Good for comparing models | Doesn't show actual precision |
| Intuitive visualization | Only for binary (or One-vs-Rest) |
| Works with probabilities | Doesn't consider costs |

### Common Follow-up Questions
1. *"What's a good AUC value?"* - 0.5 = random, 0.7-0.8 = acceptable, 0.8-0.9 = excellent, >0.9 = outstanding
2. *"Why is ROC misleading for imbalanced data?"* - FPR denominator (TN+FP) is large, so FPR stays low even with many FP
3. *"How do you choose ROC vs PR curve?"* - PR curve when positive class is rare and important

### Interview Tip
Explain that ROC is threshold-independent — it shows performance across ALL possible thresholds. Know the relationship: TPR = Recall, FPR = 1 - Specificity. Mention that for imbalanced data, Precision-Recall curve is more informative. Be ready to explain how to find optimal threshold (Youden's J or based on business requirements).

---

## Question 45
- [ ] Done

**Explain the concept of Area Under the Curve (AUC) and its significance.**

**Answer:**

### Definition
AUC (Area Under the ROC Curve) is a single number that summarizes the overall performance of a binary classifier across all classification thresholds. It represents the probability that the model ranks a randomly chosen positive instance higher than a randomly chosen negative instance.

### Mathematical Interpretation

$$AUC = P(score(positive) > score(negative))$$

If you randomly pick one positive and one negative sample, AUC = probability the model gives a higher score to the positive.

### Value Interpretation

| AUC | Interpretation |
|-----|----------------|
| 1.0 | Perfect classifier |
| 0.9 - 1.0 | Excellent |
| 0.8 - 0.9 | Good |
| 0.7 - 0.8 | Fair/Acceptable |
| 0.6 - 0.7 | Poor |
| 0.5 | Random guessing (no discrimination) |
| < 0.5 | Worse than random (flip predictions!) |

### Visual Understanding

```
    1.0───────┬──────────────────────
     T  │██████████████████████    AUC = Shaded area
     P  │████████████████████    (under the curve)
     R  │█████████████████
        │██████████████              Perfect = 1.0
     0.5│███████████              Random = 0.5
        │████████
        │█████
        │██
    0.0 │
        └────────────────────────────
       0.0                          1.0
                    FPR
```

### Implementation

```python
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# Get probability predictions
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate AUC
auc_score = roc_auc_score(y_test, y_proba)
print(f"AUC: {auc_score:.4f}")

# Or calculate from ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_from_curve = auc(fpr, tpr)  # Same result
```

### Why AUC is Useful

| Advantage | Explanation |
|-----------|-------------|
| **Threshold-independent** | Evaluates across all thresholds |
| **Scale-independent** | Works regardless of probability calibration |
| **Single number** | Easy to compare models |
| **Intuitive** | Probability of correct ranking |

### AUC vs Accuracy

```python
# Example: Imbalanced dataset (95% negative, 5% positive)

# Model 1: Always predicts negative
# Accuracy = 95%  ✔ High!
# AUC = 0.5       ✘ Random!

# Model 2: Actually discriminates
# Accuracy = 90%  
# AUC = 0.85      ✔ Good discrimination!
```

### Gini Coefficient

Often used in credit scoring:

$$Gini = 2 \times AUC - 1$$

| AUC | Gini |
|-----|------|
| 0.5 | 0.0 (random) |
| 0.75 | 0.5 |
| 1.0 | 1.0 (perfect) |

### AUC for Multiclass

```python
# One-vs-Rest (OvR)
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# Binarize the output
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_proba = model.predict_proba(X_test)

# Macro-average
auc_macro = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')

# Weighted average
auc_weighted = roc_auc_score(y_test_bin, y_proba, average='weighted', multi_class='ovr')

# One-vs-One (OvO)
auc_ovo = roc_auc_score(y_test, y_proba, multi_class='ovo')
```

### When AUC Can Be Misleading

**1. Severe Class Imbalance:**
```
With 99% negative, 1% positive:
- Many false positives might still give low FPR
- Use Precision-Recall AUC instead
```

```python
from sklearn.metrics import average_precision_score

# Precision-Recall AUC (better for imbalanced)
pr_auc = average_precision_score(y_test, y_proba)
```

**2. Different Costs for Different Errors:**
- AUC treats all errors equally
- If FN is 10x worse than FP, AUC doesn't capture this

**3. Need for Calibrated Probabilities:**
- AUC only measures ranking, not probability quality
- Use calibration curve if probabilities need to be accurate

### Precision-Recall AUC (PR-AUC)

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

# Calculate
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AP = {pr_auc:.2f})')

# Random baseline for PR-AUC = proportion of positives
baseline = y_test.mean()
plt.axhline(y=baseline, color='r', linestyle='--', label='Random')
plt.legend()
```

### ROC-AUC vs PR-AUC

| Aspect | ROC-AUC | PR-AUC |
|--------|---------|--------|
| Random baseline | Always 0.5 | Proportion of positives |
| Imbalanced data | Can be misleading | More informative |
| Focus | Both classes | Positive class |
| When to use | Balanced data | Rare positive class |

### Common Follow-up Questions
1. *"Can AUC be less than 0.5?"* - Yes, means worse than random; flip your predictions
2. *"What if two models have same AUC?"* - Look at the curves; one might be better at specific operating points
3. *"How to handle multiclass?"* - OvR or OvO approach, macro or weighted averaging

### Interview Tip
Explain AUC as "probability of correct ranking" for intuitive understanding. Know when to use PR-AUC instead (imbalanced data). Mention that AUC doesn't tell you about probability calibration — a model can have perfect AUC but poorly calibrated probabilities. Always consider business context when choosing evaluation metrics.

---

## Question 46
- [ ] Done

**What is Mean Squared Error (MSE) and its use in regression problems?**

**Answer:**

### Formula
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

### Properties
- Always positive
- Penalizes large errors more
- Same units as y²

### Related Metrics
RMSE = √MSE (interpretable units)

### Interview Tip
Use RMSE for interpretation, MSE for optimization.

---

## Question 47
- [ ] Done

**Describe the concept of R-squared (Coefficient of Determination) and its interpretation.**

**Answer:**

### Formula
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

### Interpretation
Proportion of variance explained by model.
- 0 = No better than mean
- 1 = Perfect fit

### Limitation
Always increases with more features (use adjusted R²).

### Interview Tip
R² can be misleading; always check residuals.

---

## Question 48
- [ ] Done

**What is K-Fold Cross-Validation and its advantages?**

**Answer:**

### Process
1. Split into K folds
2. Train on K-1, test on 1
3. Repeat K times
4. Average results

### Advantages
- Uses all data for training and testing
- More reliable estimates
- Better for small datasets

### Interview Tip
K=5 or K=10 are common choices.

---

## Question 49
- [ ] Done

**Explain the concept of Stratified K-Fold Cross-Validation and its use cases.**

**Answer:**

### Definition
K-Fold that preserves class distribution in each fold.

### When to Use
- Imbalanced classes
- Classification problems
- Small datasets

### Interview Tip
Always use stratified for classification.

---

## Question 50
- [ ] Done

**What is a Neural Network and its components?**

**Answer:**

### Definition
A Neural Network (NN) is a computational model inspired by biological neurons. It consists of interconnected nodes (neurons) organized in layers that learn to map inputs to outputs through training.

### Architecture Overview

```
Input Layer        Hidden Layer(s)        Output Layer
                                          
  x₁ ○─────────○                         
       \    /  \                         
  x₂ ○──○────○────○ → ŷ
       /    \  /                         
  x₃ ○─────────○                         
                                          
 Features        Learned              Prediction
              Representations
```

### Key Components

| Component | Description | Role |
|-----------|-------------|------|
| **Neurons (Nodes)** | Computational units | Compute weighted sum + activation |
| **Weights (W)** | Connection strengths | Learned parameters |
| **Biases (b)** | Offset terms | Shift activation function |
| **Activation Functions** | Non-linear transforms | Introduce non-linearity |
| **Layers** | Groups of neurons | Hierarchical feature extraction |

### Layer Types

**1. Input Layer:**
- Receives raw features
- No computation, just passes data
- Size = number of features

**2. Hidden Layers:**
- Perform computations
- Extract increasingly abstract features
- "Deep" = many hidden layers

**3. Output Layer:**
- Produces final prediction
- Structure depends on task:
  - Regression: 1 neuron (linear)
  - Binary classification: 1 neuron (sigmoid)
  - Multiclass: N neurons (softmax)

### Single Neuron Computation

```
     x₁ ──w₁──┐
                │
     x₂ ──w₂──┼───→ z = Σ(wᵢxᵢ) + b ───→ a = f(z) ──→ output
                │         (weighted sum)       (activation)
     x₃ ──w₃──┘
            + b (bias)
```

$$z = \sum_{i=1}^{n} w_i x_i + b = W^T X + b$$
$$a = f(z) \quad \text{(activation function)}$$

### Forward Propagation

```
For each layer l:
    z[l] = W[l] · a[l-1] + b[l]
    a[l] = f(z[l])
```

```python
def forward_propagation(X, parameters):
    A = X
    for l in range(1, L + 1):
        Z = np.dot(parameters['W' + str(l)], A) + parameters['b' + str(l)]
        A = activation(Z)  # ReLU, sigmoid, etc.
    return A
```

### Implementation with Keras

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Build model
model = Sequential([
    # Input layer is implicit (input_shape)
    Dense(128, activation='relu', input_shape=(n_features,)),  # Hidden 1
    Dropout(0.3),  # Regularization
    Dense(64, activation='relu'),   # Hidden 2
    Dropout(0.3),
    Dense(32, activation='relu'),   # Hidden 3
    Dense(1, activation='sigmoid')  # Output (binary classification)
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)]
)

# Predict
y_pred = (model.predict(X_test) > 0.5).astype(int)
```

### Choosing Architecture

| Aspect | Guidelines |
|--------|------------|
| **Input size** | Number of features |
| **Output size** | 1 (regression/binary), N (multiclass) |
| **Hidden layers** | Start with 1-2, add if underfitting |
| **Neurons per layer** | Start wide, narrow toward output |
| **Activation** | ReLU for hidden, sigmoid/softmax for output |

### Output Layer by Task

| Task | Neurons | Activation | Loss Function |
|------|---------|------------|---------------|
| Regression | 1 | Linear (none) | MSE |
| Binary Classification | 1 | Sigmoid | Binary Cross-Entropy |
| Multiclass | N classes | Softmax | Categorical Cross-Entropy |
| Multi-label | N labels | Sigmoid (each) | Binary Cross-Entropy |

### Hyperparameters

| Hyperparameter | Options | Notes |
|----------------|---------|-------|
| Learning rate | 0.001, 0.01, 0.1 | Most important |
| Batch size | 32, 64, 128 | Affects training speed/stability |
| Epochs | 10-1000 | Use early stopping |
| Layers/neurons | Task-dependent | Start simple |
| Dropout rate | 0.1-0.5 | Regularization |
| Optimizer | Adam, SGD, RMSprop | Adam is good default |

### Universal Approximation Theorem

> A neural network with a single hidden layer containing a finite number of neurons can approximate any continuous function (given enough neurons).

This is why NNs are so powerful!

### Common Follow-up Questions
1. *"Why do we need hidden layers?"* - To learn non-linear relationships and hierarchical features
2. *"What makes deep learning 'deep'?"* - Multiple hidden layers that learn increasingly abstract representations
3. *"How many neurons/layers should I use?"* - Start simple, add complexity if underfitting

### Interview Tip
Know the basic math: weighted sum → activation function. Explain why non-linearity is essential (without it, multiple layers collapse to single linear transformation). Be ready to discuss how to choose architecture based on problem type (output layer, loss function). Mention overfitting prevention: dropout, early stopping, regularization.

---

## Question 51
- [ ] Done

**Explain the difference between Feedforward and Recurrent Neural Networks.**

**Answer:**

### Comparison

| Feedforward | Recurrent |
|-------------|-----------|
| One direction | Has loops |
| No memory | Hidden state |
| Fixed input | Sequences |
| Images, tabular | Text, time series |

### Interview Tip
RNNs process sequences; FFNs process fixed inputs.

---

## Question 52
- [ ] Done

**What is Backpropagation and how does it work?**

**Answer:**

### Definition
Backpropagation (backward propagation of errors) is the algorithm used to calculate gradients of the loss function with respect to each weight in the neural network. It efficiently computes all gradients using the chain rule, enabling gradient descent to update weights.

### The Big Picture

```
┌────────────────────────────────────────────────────────┐
│                  Training Loop                        │
│                                                        │
│   1. Forward Pass: X ───────────────→ ŷ               │
│        (compute predictions)                           │
│                                                        │
│   2. Compute Loss: L = loss(y, ŷ)                     │
│        (measure error)                                 │
│                                                        │
│   3. Backward Pass: ∂L/∂w ←──────────── L              │
│        (compute gradients via chain rule)              │
│                                                        │
│   4. Update Weights: w = w - α · ∂L/∂w                │
│        (gradient descent step)                         │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### The Chain Rule (Heart of Backprop)

For a composite function $L = L(y(z(w)))$:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w}$$

### Step-by-Step Example

Simple network: Input → Hidden → Output

```
Forward Pass:
x → z₁ = w₁x + b₁ → a₁ = ReLU(z₁) → z₂ = w₂a₁ + b₂ → ŷ = σ(z₂)

Loss: L = -(y log(ŷ) + (1-y) log(1-ŷ))  [Binary Cross-Entropy]

Backward Pass:
∂L/∂ŷ → ∂L/∂z₂ → ∂L/∂w₂ → ∂L/∂a₁ → ∂L/∂z₁ → ∂L/∂w₁
```

### Mathematical Derivation

**Layer 2 (Output Layer):**

$$\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_2} \cdot \frac{\partial z_2}{\partial w_2}$$

$$\frac{\partial L}{\partial w_2} = (\hat{y} - y) \cdot a_1$$

**Layer 1 (Hidden Layer):**

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}$$

$$\frac{\partial L}{\partial w_1} = (\hat{y} - y) \cdot w_2 \cdot \text{ReLU}'(z_1) \cdot x$$

### Implementation (Simplified)

```python
def backpropagation(X, y, parameters, cache):
    m = X.shape[1]  # Number of samples
    gradients = {}
    
    # Output layer gradient
    dZ2 = cache['A2'] - y  # For cross-entropy + sigmoid
    gradients['dW2'] = (1/m) * np.dot(dZ2, cache['A1'].T)
    gradients['db2'] = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # Hidden layer gradient (chain rule!)
    dA1 = np.dot(parameters['W2'].T, dZ2)
    dZ1 = dA1 * relu_derivative(cache['Z1'])
    gradients['dW1'] = (1/m) * np.dot(dZ1, X.T)
    gradients['db1'] = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return gradients

def update_parameters(parameters, gradients, learning_rate):
    for l in range(1, L + 1):
        parameters['W' + str(l)] -= learning_rate * gradients['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * gradients['db' + str(l)]
    return parameters
```

### Why Backpropagation is Efficient

**Naive Approach:** Compute each gradient separately → O(W²) for W weights

**Backpropagation:** Reuse intermediate computations → O(W)

```
Without backprop: Compute ∂L/∂w for each w independently
With backprop: Compute once, propagate backward, reuse!
```

### Common Issues

**1. Vanishing Gradients:**
- Gradients become very small in early layers
- Weights barely update
- Caused by: sigmoid/tanh activation, deep networks
- Solutions: ReLU, skip connections, proper initialization

**2. Exploding Gradients:**
- Gradients become very large
- Training becomes unstable (NaN)
- Solutions: Gradient clipping, proper initialization, batch norm

```python
# Gradient clipping
if np.linalg.norm(gradient) > max_norm:
    gradient = gradient * max_norm / np.linalg.norm(gradient)
```

### Automatic Differentiation

Modern frameworks (TensorFlow, PyTorch) compute gradients automatically:

```python
import torch

# Forward pass
output = model(X)
loss = loss_fn(output, y)

# Backward pass (automatic!)
loss.backward()  # Computes all gradients

# Gradients stored in:
for param in model.parameters():
    print(param.grad)  # ∂L/∂param
```

### Computational Graph

```
    x ───┬───→ w·x ───┬───→ w·x + b ───→ σ(...) ───→ Loss
    w ───┘           │
                   b ─┘
    
    Forward: Left → Right (compute values)
    Backward: Right → Left (compute gradients)
```

### Common Follow-up Questions
1. *"Why do we need the chain rule?"* - Because the loss is a composition of many functions; we need gradients through each
2. *"What's the difference between forward and backward pass?"* - Forward computes predictions, backward computes gradients
3. *"What causes vanishing gradients?"* - Multiplying many small gradients (especially with sigmoid/tanh)

### Interview Tip
Explain the chain rule as the key insight. Know that backprop is just an efficient way to compute all gradients in one backward pass. Mention vanishing/exploding gradients as common problems and their solutions (ReLU, proper initialization, batch norm). Modern frameworks handle this automatically, but understanding the concept is essential.

---

## Question 53
- [ ] Done

**Describe the concept of Activation Functions and their types.**

**Answer:**

### Definition
Activation functions are non-linear transformations applied to the output of each neuron. They introduce non-linearity, enabling neural networks to learn complex patterns.

### Why Non-linearity is Essential

**Without activation functions:**
$$\text{Layer 1}: z_1 = W_1 x$$
$$\text{Layer 2}: z_2 = W_2 z_1 = W_2 W_1 x = W' x$$

Multiple linear layers collapse to a single linear transformation! Cannot learn complex patterns.

### Common Activation Functions

#### 1. ReLU (Rectified Linear Unit)

$$f(x) = \max(0, x)$$

```
    │     /
    │    /
    │   /
────┴──────
    0
```

| Pros | Cons |
|------|------|
| Fast computation | "Dying ReLU" (neurons can die) |
| No vanishing gradient (positive) | Not zero-centered |
| Sparse activation | Unbounded (can explode) |

**Best for:** Hidden layers (default choice)

#### 2. Leaky ReLU

$$f(x) = \max(\alpha x, x), \quad \alpha \approx 0.01$$

```
    │     /
    │    /
  __│___/
    │
```

**Why:** Prevents dying ReLU by allowing small gradients when x < 0

#### 3. Sigmoid

$$f(x) = \frac{1}{1 + e^{-x}}$$

```
  1 ──────────┬──
    │        │
0.5 │    ────
    │   /
  0 ┴─────────────
```

| Output | 0 to 1 |
|--------|--------|
| **Pros** | Smooth, probabilistic interpretation |
| **Cons** | Vanishing gradients, not zero-centered |

**Best for:** Binary classification output layer

#### 4. Tanh (Hyperbolic Tangent)

$$f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

| Output | -1 to 1 |
|--------|--------|
| **Pros** | Zero-centered |
| **Cons** | Still has vanishing gradient |

**Best for:** RNNs (sometimes), when zero-centered output needed

#### 5. Softmax (For Multiclass Output)

$$f(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$$

- Outputs sum to 1 (probability distribution)
- **Best for:** Multiclass classification output layer

### Comparison Table

| Function | Range | Zero-Centered | Gradient | Use Case |
|----------|-------|---------------|----------|----------|
| ReLU | [0, ∞) | No | 0 or 1 | Hidden layers |
| Leaky ReLU | (-∞, ∞) | No | 0.01 or 1 | Hidden (fix dying ReLU) |
| Sigmoid | (0, 1) | No | Vanishes | Binary output |
| Tanh | (-1, 1) | Yes | Vanishes | RNNs, hidden |
| Softmax | (0, 1), sum=1 | - | - | Multiclass output |

### Derivatives (For Backpropagation)

| Function | Derivative |
|----------|------------|
| ReLU | $1$ if $x > 0$, else $0$ |
| Sigmoid | $\sigma(x)(1 - \sigma(x))$ |
| Tanh | $1 - \tanh^2(x)$ |

```python
# Derivatives in code
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)  # Max at 0.25 when x=0
```

### Modern Activation Functions

#### GELU (Gaussian Error Linear Unit)
$$f(x) = x \cdot \Phi(x)$$

- Used in Transformers (BERT, GPT)
- Smooth approximation of ReLU

#### Swish
$$f(x) = x \cdot \sigma(x)$$

- Self-gated activation
- Often outperforms ReLU

#### Mish
$$f(x) = x \cdot \tanh(\text{softplus}(x))$$

- Smooth, non-monotonic

### Implementation

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU

# Using built-in string
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(64, activation='tanh'))

# Using layer objects (more control)
model.add(Dense(64))
model.add(LeakyReLU(alpha=0.1))

# Using tf.nn functions
model.add(Dense(64, activation=tf.nn.gelu))
model.add(Dense(64, activation=tf.nn.swish))

# Output layers
model.add(Dense(1, activation='sigmoid'))   # Binary
model.add(Dense(10, activation='softmax'))  # Multiclass
model.add(Dense(1, activation='linear'))    # Regression
```

### Vanishing Gradient Problem

```
Sigmoid derivative: max = 0.25 (at x=0)

After 10 layers: 0.25^10 = 0.000001
Gradients become extremely small!
Early layers don't learn.
```

**Solutions:**
- Use ReLU (gradient = 1 for positive values)
- Skip connections (ResNets)
- Batch normalization
- Proper weight initialization

### Choosing Activation Functions

| Layer | Recommendation |
|-------|----------------|
| Hidden layers | ReLU (default), Leaky ReLU, GELU |
| Binary output | Sigmoid |
| Multiclass output | Softmax |
| Regression output | Linear (none) |
| RNNs | Tanh, LSTM uses sigmoid+tanh |
| Transformers | GELU |

### Common Follow-up Questions
1. *"Why is ReLU preferred over sigmoid?"* - No vanishing gradient (for positive values), computationally efficient
2. *"What is dying ReLU?"* - Neurons output 0 for all inputs, gradient = 0, never recover
3. *"Why use softmax for multiclass?"* - Outputs are probabilities that sum to 1

### Interview Tip
Know why non-linearity is essential (without it, deep networks collapse to linear). Explain vanishing gradient problem and why ReLU helps. Know the appropriate activation for each layer type: ReLU for hidden, sigmoid/softmax for output depending on task.

---

## Question 54
- [ ] Done

**What is Deep Learning and its applications?**

**Answer:**

### Definition
Neural networks with many layers learning hierarchical representations.

### Applications
- Computer vision
- NLP/Language models
- Speech recognition
- Game playing
- Autonomous vehicles

### Interview Tip
Deep learning excels with large unstructured data.

---

## Question 55
- [ ] Done

**Explain the concept of Convolutional Neural Networks (CNN) and their architecture.**

**Answer:**

### Architecture
Conv → ReLU → Pool → ... → Flatten → FC → Output

### Key Components
- Convolution: Learns local patterns
- Pooling: Reduces dimensions
- Feature maps: Learned representations

### Interview Tip
CNNs learn hierarchical features automatically.

---

## Question 56
- [ ] Done

**What is Recurrent Neural Networks (RNN) and their variants (LSTM, GRU)?**

**Answer:**

### RNN Issues
Vanishing/exploding gradients for long sequences.

### LSTM Gates
- Forget: What to remove
- Input: What to add
- Output: What to expose

### GRU
Simplified LSTM with 2 gates instead of 3.

### Interview Tip
LSTM/GRU solve long-range dependency problem.

---

## Question 57
- [ ] Done

**Describe the concept of Autoencoders and their use cases.**

**Answer:**

### Architecture
Encoder → Bottleneck → Decoder

### Use Cases
- Dimensionality reduction
- Denoising
- Anomaly detection
- Feature learning
- Generative models (VAE)

### Interview Tip
Autoencoders learn compressed representations.

---

## Question 58
- [ ] Done

**What is Transfer Learning and its benefits in deep learning?**

**Answer:**

### Definition
Use pre-trained model on new task.

### Benefits
- Less data needed
- Faster training
- Better performance
- Lower compute cost

### Approaches
Feature extraction (freeze) or fine-tuning (unfreeze).

### Interview Tip
Start with feature extraction, then fine-tune.

---

## Question 59
- [ ] Done

**Explain the concept of Generative Adversarial Networks (GAN) and their applications.**

**Answer:**

### Architecture
- Generator: Creates fake data
- Discriminator: Distinguishes real/fake
- Adversarial training

### Applications
- Image generation
- Style transfer
- Data augmentation
- Super-resolution

### Interview Tip
GANs are hard to train; know mode collapse.

---

## Question 60
- [ ] Done

**What is Natural Language Processing (NLP) and its applications?**

**Answer:**

### Applications
- Sentiment analysis
- Named entity recognition
- Machine translation
- Question answering
- Text summarization
- Chatbots

### Modern Approach
Transformer-based models (BERT, GPT).

### Interview Tip
Transformers have revolutionized NLP.

---

## Question 61
- [ ] Done

**Explain the difference between Tokenization and Stemming.**

**Answer:**

### Comparison

| Tokenization | Stemming |
|--------------|----------|
| Split into tokens | Reduce to root |
| Words, subwords | Word variants |
| First step | Normalization |

### Interview Tip
Subword tokenization (BPE) is modern standard.

---

## Question 62
- [ ] Done

**What is Word Embedding and its popular techniques (Word2Vec, GloVe)?**

**Answer:**

### Definition
Dense vector representations capturing semantic meaning.

### Techniques
- Word2Vec: Predictive (CBOW, Skip-gram)
- GloVe: Count-based
- FastText: Subword embeddings

### Interview Tip
Contextual embeddings (BERT) are now preferred.

---

## Question 63
- [ ] Done

**Describe the concept of Named Entity Recognition (NER) and its approaches.**

**Answer:**

### Definition
Identify and classify named entities in text.

### Entity Types
Person, Organization, Location, Date, etc.

### Approaches
- Rule-based
- CRF
- BiLSTM-CRF
- Transformers (BERT)

### Interview Tip
Know BIO tagging scheme.

---

## Question 64
- [ ] Done

**What is Sentiment Analysis and its methods?**

**Answer:**

### Levels
- Document
- Sentence
- Aspect-based

### Methods
- Lexicon-based
- ML classifiers
- Deep learning
- Transformers

### Interview Tip
Handle sarcasm and negation challenges.

---

## Question 65
- [ ] Done

**Explain the concept of Topic Modeling and its algorithms (LDA, NMF).**

**Answer:**

### Definition
Discover abstract topics in document collections.

### Algorithms
- LDA: Probabilistic
- NMF: Matrix factorization

### Output
Topics as word distributions.

### Interview Tip
LDA is generative; NMF is deterministic.

---

## Question 66
- [ ] Done

**What is Text Classification and its techniques?**

**Answer:**

### Techniques
- Bag-of-Words + ML
- TF-IDF + ML
- Deep Learning (CNN, LSTM)
- Transformers (BERT)

### Applications
Spam detection, sentiment, categorization.

### Interview Tip
BERT fine-tuning is state-of-the-art.

---

## Question 67
- [ ] Done

**Describe the concept of Language Translation and its challenges.**

**Answer:**

### Modern Approach
Sequence-to-sequence with attention (Transformers).

### Challenges
- Word order differences
- Idioms and context
- Low-resource languages
- Domain-specific terms

### Interview Tip
Know encoder-decoder architecture.

---

## Question 68
- [ ] Done

**What is Text Summarization and its types (Extractive, Abstractive)?**

**Answer:**

### Types

| Extractive | Abstractive |
|------------|-------------|
| Select sentences | Generate new text |
| Copy-paste | Paraphrase |
| Simpler | More complex |
| Graph-based, ML | Seq2seq, Transformers |

### Interview Tip
Abstractive is harder but more natural.

---

## Question 69
- [ ] Done

**Explain the concept of Chatbots and their architecture.**

**Answer:**

### Types
- Rule-based
- Retrieval-based
- Generative

### Components
- NLU: Intent/entity extraction
- Dialog management
- NLG: Response generation

### Interview Tip
Modern chatbots use LLMs (GPT, Claude).

---

## Question 70
- [ ] Done

**What is a Recommender System and its types?**

**Answer:**

### Types
- Content-based: Item features
- Collaborative filtering: User behavior
- Hybrid: Combination

### Interview Tip
Most production systems are hybrid.

---

## Question 71
- [ ] Done

**Explain the difference between Content-Based and Collaborative Filtering.**

**Answer:**

### Comparison

| Content-Based | Collaborative |
|---------------|---------------|
| Item features | User behavior |
| No cold start for users | Cold start problem |
| Limited discovery | Serendipity |

### Interview Tip
Collaborative discovers new interests.

---

## Question 72
- [ ] Done

**What is Matrix Factorization and its role in Recommender Systems?**

**Answer:**

### Definition
Decompose user-item matrix into latent factors.
$$R \approx U \times V^T$$

### Algorithms
SVD, ALS, NMF

### Interview Tip
Latent factors capture abstract preferences.

---

## Question 73
- [ ] Done

**Describe the concept of Cold Start Problem and its solutions.**

**Answer:**

### Types
- New user: No history
- New item: No ratings

### Solutions
- Content-based fallback
- Popularity-based
- Ask preferences
- Hybrid approaches

### Interview Tip
Know solutions for both user and item cold start.

---

## Question 74
- [ ] Done

**What is Evaluation Metrics for Recommender Systems (Precision, Recall, NDCG)?**

**Answer:**

### Metrics
- Precision@K: Relevant in top K
- Recall@K: Relevant found
- NDCG: Rank-aware metric
- MAP: Average precision

### Interview Tip
NDCG accounts for position; use for ranking.

---

## Question 75
- [ ] Done

**Explain the concept of Hybrid Recommender Systems and their advantages.**

**Answer:**

### Combination Strategies
- Weighted
- Switching
- Feature combination
- Cascade

### Advantages
Overcome individual method limitations.

### Interview Tip
Netflix uses hybrid approach.

---

## Question 76
- [ ] Done

**What is the Alternating Least Squares (ALS) algorithm and its use in Recommender Systems?**

**Answer:**

### How It Works
1. Fix item factors, optimize user factors
2. Fix user factors, optimize item factors
3. Alternate until convergence

### Advantages
- Parallelizable
- Handles sparse data
- Implicit feedback support

### Interview Tip
ALS is common for large-scale systems (Spark).

---

## Question 77
- [ ] Done

**Describe the concept of Implicit Feedback and its challenges.**

**Answer:**

### Examples
Clicks, views, purchases (vs explicit ratings).

### Challenges
- No negative signal
- Click ≠ like
- Position bias

### Handling
BPR, weighted loss, one-class collaborative filtering.

### Interview Tip
Most real-world data is implicit.

---

## Question 78
- [ ] Done

**What is the Singular Value Decomposition (SVD) and its application in Recommender Systems?**

**Answer:**

### Definition
Factor matrix: $R = U \Sigma V^T$

### Application
- Reduce dimensionality
- Identify latent factors
- Make predictions

### Interview Tip
SVD won Netflix Prize (with modifications).

---

## Question 79
- [ ] Done

**Explain the concept of Diversity and Serendipity in Recommender Systems.**

**Answer:**

### Definitions
- Diversity: How different recommendations are
- Serendipity: Surprising but relevant

### Importance
Avoid filter bubbles, improve user experience.

### Interview Tip
Balance accuracy with exploration.

---

## Question 80
- [ ] Done

**What is Reinforcement Learning and its components?**

**Answer:**

### Components
- Agent: Learner
- Environment: World
- State: Situation
- Action: Decision
- Reward: Feedback
- Policy: Strategy

### Interview Tip
RL learns by trial and error.

---

## Question 81
- [ ] Done

**Explain the difference between Exploitation and Exploration in Reinforcement Learning.**

**Answer:**

### Definitions
- Exploitation: Use best known action
- Exploration: Try new actions

### Trade-off
Need balance to find optimal policy.

### Methods
ε-greedy, UCB, Thompson sampling.

### Interview Tip
Too much exploration = slow learning; too much exploitation = local optima.

---

## Question 82
- [ ] Done

**What is Markov Decision Process (MDP) and its elements?**

**Answer:**

### Elements
- States (S)
- Actions (A)
- Transition probabilities P(s'|s,a)
- Rewards R(s,a,s')
- Discount factor γ

### Markov Property
Future depends only on current state.

### Interview Tip
MDP is the mathematical framework for RL.

---

## Question 83
- [ ] Done

**Describe the concept of Q-Learning and its algorithm.**

**Answer:**

### Q-Value
Expected cumulative reward for action in state.

### Update Rule
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

### Properties
Model-free, off-policy.

### Interview Tip
Q-learning learns optimal policy.

---

## Question 84
- [ ] Done

**What is Deep Q-Networks (DQN) and its improvements?**

**Answer:**

### DQN
Neural network approximates Q-function.

### Improvements
- Experience replay
- Target network
- Double DQN
- Prioritized replay

### Interview Tip
DQN made RL work for Atari games.

---

## Question 85
- [ ] Done

**Explain the concept of Policy Gradient Methods and their advantages.**

**Answer:**

### Definition
Directly optimize policy parameters.

### Advantages
- Works with continuous actions
- Naturally stochastic policies
- Simpler objective

### Algorithms
REINFORCE, PPO, A3C.

### Interview Tip
Policy gradients for continuous control.

---

## Question 86
- [ ] Done

**What is Actor-Critic Methods and their variants?**

**Answer:**

### Components
- Actor: Policy (actions)
- Critic: Value function (evaluation)

### Variants
A2C, A3C, PPO, SAC

### Advantage
Lower variance than pure policy gradient.

### Interview Tip
Actor-Critic combines policy and value learning.

---

## Question 87
- [ ] Done

**Describe the concept of Monte Carlo Tree Search (MCTS) and its applications.**

**Answer:**

### Algorithm
1. Selection
2. Expansion
3. Simulation
4. Backpropagation

### Applications
- Game playing (Go, Chess)
- Planning
- Decision making

### Interview Tip
AlphaGo used MCTS + deep learning.

---

## Question 88
- [ ] Done

**What is the Bellman Equation and its role in Reinforcement Learning?**

**Answer:**

### Equation
$$V(s) = \max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')]$$

### Role
Foundation for value-based RL methods.
Defines optimal value recursively.

### Interview Tip
Bellman equation enables dynamic programming.

---

## Question 89
- [ ] Done

**Explain the concept of Inverse Reinforcement Learning and its use cases.**

**Answer:**

### Definition
Learn reward function from expert demonstrations.

### Use Cases
- Autonomous driving
- Robotics
- Imitation learning

### Interview Tip
IRL when reward function is unknown.

---

## Question 90
- [ ] Done

**What is Optimization in ML and its types?**

**Answer:**

### Types
- Convex: Global optimum guaranteed
- Non-convex: Local optima possible

### Goal
Find parameters minimizing loss.

### Interview Tip
Deep learning is non-convex optimization.

---

## Question 91
- [ ] Done

**Explain the difference between Gradient Descent and Stochastic Gradient Descent.**

**Answer:**

### Comparison

| Batch GD | SGD |
|----------|-----|
| All data | One sample |
| Slow, accurate | Fast, noisy |
| Stable | Escapes local minima |

### Mini-batch
Best of both (32-256 samples).

### Interview Tip
Mini-batch is the standard in practice.

---

## Question 92
- [ ] Done

**What is Learning Rate and its impact on model training?**

**Answer:**

### Impact
- Too high: Diverges
- Too low: Slow convergence
- Just right: Optimal training

### Techniques
Learning rate schedules, adaptive (Adam).

### Interview Tip
Learning rate is the most important hyperparameter.

---

## Question 93
- [ ] Done

**Describe the concept of Momentum and its benefits in optimization.**

**Answer:**

### Definition
Accumulate velocity in consistent directions.
$$v_t = \beta v_{t-1} + \nabla L$$
$$\theta = \theta - \alpha v_t$$

### Benefits
- Faster convergence
- Reduces oscillation
- Escapes shallow minima

### Interview Tip
Momentum is standard in deep learning.

---

## Question 94
- [ ] Done

**What is Hyperparameter Tuning and its techniques?**

**Answer:**

### Techniques
- Grid Search
- Random Search
- Bayesian Optimization
- Hyperband
- Optuna

### Common Hyperparameters
Learning rate, batch size, layers, regularization.

### Interview Tip
Random search often better than grid search.

---

## Question 95
- [ ] Done

**Explain the concept of Grid Search and its limitations.**

**Answer:**

### Method
Exhaustively try all combinations.

### Limitations
- Exponentially expensive
- Inefficient for many hyperparameters
- Wasteful on unimportant parameters

### Interview Tip
Use grid search for few, known important params.

---

## Question 96
- [ ] Done

**What is Random Search and its advantages over Grid Search?**

**Answer:**

### Advantages
- More efficient exploration
- Better for many hyperparameters
- Finds good solutions faster

### Why Better
Explores unique values for important parameters.

### Interview Tip
Bergstra & Bengio proved random > grid.

---

## Question 97
- [ ] Done

**Describe the concept of Bayesian Optimization and its applications.**

**Answer:**

### How It Works
1. Build surrogate model (GP)
2. Acquisition function selects next point
3. Evaluate and update
4. Repeat

### Advantages
Sample-efficient for expensive evaluations.

### Interview Tip
Use for deep learning hyperparameter tuning.

---

## Question 98
- [ ] Done

**What is Early Stopping and its role in preventing overfitting?**

**Answer:**

### Method
Stop training when validation loss stops improving.

### Implementation
- Monitor validation metric
- Patience parameter
- Restore best weights

### Interview Tip
Early stopping is implicit regularization.

---

## Question 99
- [ ] Done

**Explain the concept of Learning Rate Scheduling and its types.**

**Answer:**

### Types
- Step decay: Reduce by factor
- Exponential: Continuous decay
- Cosine: Smooth decay and restart
- Warm-up: Start low, increase

### Benefits
Better convergence, fine-tuning in later epochs.

### Interview Tip
Cosine annealing with warm restarts is popular.

---

