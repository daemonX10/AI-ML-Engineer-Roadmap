# Model Evaluation Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What is model evaluation in the context of machine learning?**

**Answer:**

### Definition
Model evaluation is the systematic process of measuring how well a trained machine learning model generalizes to new, unseen data. It answers: "Will this model work in the real world?"

### Core Concepts

**Why We Evaluate:**
- Training accuracy is misleading (model memorizes training data)
- Need to estimate performance on future/unseen data
- Select the best model among candidates
- Identify overfitting or underfitting problems

**What We Measure:**
- **Generalization ability**: Performance gap between train and test
- **Prediction quality**: Accuracy, precision, recall, MSE depending on task
- **Calibration**: Are predicted probabilities reliable?
- **Robustness**: Does model work across different data slices?

### Evaluation Workflow

```
Training Data → Train Model → Validation Data → Tune Hyperparameters → Test Data → Final Score
```

**Step-by-step:**
1. Split data into train/validation/test (before any training)
2. Train model on training set only
3. Evaluate on validation set, tune hyperparameters
4. Once finalized, evaluate once on test set
5. Report test set performance as expected real-world performance

### Key Metrics by Task

| Task | Common Metrics |
|------|----------------|
| Classification | Accuracy, Precision, Recall, F1, AUC-ROC |
| Regression | MSE, RMSE, MAE, R² |
| Clustering | Silhouette Score, Davies-Bouldin Index |
| Ranking | NDCG, MAP, MRR |

### Practical Relevance
- Wrong evaluation leads to deploying bad models
- Business decisions depend on reliable performance estimates
- Helps debug model issues before production

### Interview Tip
Never use test data for any decision-making. Test set is a "sealed envelope" opened only once at the end.

---

## Question 2
- [ ] Done

**Explain the difference between training, validation, and test datasets.**

**Answer:**

### Definition
Data splitting divides your dataset into three distinct subsets, each serving a specific purpose in the model development lifecycle. This prevents information leakage and ensures honest performance estimates.

### Core Concepts

| Dataset | Purpose | When Used | Who Sees It |
|---------|---------|-----------|-------------|
| **Training** | Learn model parameters (weights) | During training | Model learns from it |
| **Validation** | Tune hyperparameters, select model | During development | Used for decisions |
| **Test** | Final unbiased performance estimate | Once, at the end | Never touched until final evaluation |

### Detailed Explanation

**Training Set (60-70%)**
- Model learns patterns from this data
- Weights/parameters are adjusted to minimize loss on this set
- Larger = better learning, but need enough for validation/test

**Validation Set (15-20%)**
- Used to compare different models or hyperparameters
- Example: Which learning rate is best? Which model architecture?
- Can be used multiple times during development
- Slight bias introduced because we optimize for validation performance

**Test Set (15-20%)**
- Simulates real-world unseen data
- Used exactly ONCE after all development is complete
- Gives unbiased estimate of how model will perform in production
- Never use for any decision-making

### Visual Workflow

```
Original Data
     │
     ├── Training Set (70%) ──→ Model learns weights
     │
     ├── Validation Set (15%) ──→ Compare models, tune hyperparameters
     │
     └── Test Set (15%) ──→ Final performance (report this number)
```

### Why Three Sets?

If only train/test:
- You tune on test set → test set becomes validation set
- No unbiased estimate remains

The validation set acts as a "buffer" protecting test set integrity.

### Python Code Example

```python
from sklearn.model_selection import train_test_split

# Step 1: Split into train+val and test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Step 2: Split train+val into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.18, random_state=42  # 0.18 of 85% ≈ 15%
)

# Result: ~70% train, ~15% val, ~15% test
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

### Common Pitfalls
1. Using test set to tune hyperparameters (data leakage)
2. Looking at test set multiple times and picking best result
3. Not shuffling data before splitting (especially time-series needs care)
4. Unequal class distribution across splits (use stratified split)

### Interview Tip
When someone says "80/20 train-test split," ask: "Where are you tuning hyperparameters?" If answer is test set, that's a red flag.

---

## Question 3
- [ ] Done

**What is cross-validation, and why is it used?**

**Answer:**

### Definition
Cross-validation is a resampling technique that uses different portions of data as training and validation sets in multiple rounds. It provides a more reliable estimate of model performance by using all data for both training and validation.

### Why Cross-Validation?
- Single train/val split is noisy (depends on which samples land where)
- Limited data makes single split unreliable
- Reduces variance of performance estimate
- Uses all data points for training AND validation (different rounds)

### K-Fold Cross-Validation (Most Common)

**Algorithm:**
```
1. Shuffle dataset randomly
2. Split into K equal-sized folds (K=5 or 10 typical)
3. For i = 1 to K:
   - Use fold i as validation set
   - Use remaining K-1 folds as training set
   - Train model, record validation score
4. Final score = average of K validation scores
5. Standard deviation = measure of estimate reliability
```

**Visual (K=5):**
```
Round 1: [VAL] [Train] [Train] [Train] [Train]  → Score₁
Round 2: [Train] [VAL] [Train] [Train] [Train]  → Score₂
Round 3: [Train] [Train] [VAL] [Train] [Train]  → Score₃
Round 4: [Train] [Train] [Train] [VAL] [Train]  → Score₄
Round 5: [Train] [Train] [Train] [Train] [VAL]  → Score₅

Final Score = (Score₁ + Score₂ + Score₃ + Score₄ + Score₅) / 5
```

### Types of Cross-Validation

| Type | When to Use |
|------|-------------|
| **K-Fold** | Standard choice, K=5 or 10 |
| **Stratified K-Fold** | Classification with imbalanced classes |
| **Leave-One-Out (LOOCV)** | Very small datasets (K=n) |
| **Time Series Split** | Time-ordered data (no future leakage) |
| **Group K-Fold** | When samples from same group must stay together |

### Mathematical Formulation

$$\text{CV Score} = \frac{1}{K} \sum_{i=1}^{K} \text{Score}_i$$

$$\text{Variance} = \frac{1}{K} \sum_{i=1}^{K} (\text{Score}_i - \text{CV Score})^2$$

### Python Code Example

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Simple cross-validation
model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"Scores per fold: {scores}")
print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")

# Stratified K-Fold for imbalanced classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
```

### Choosing K
- K=5 or K=10: Standard choice, good bias-variance tradeoff
- K=n (LOOCV): Low bias, high variance, computationally expensive
- K=3: Use when dataset is very large (computation matters)

### Common Pitfalls
1. Doing feature selection/preprocessing BEFORE cross-validation (data leakage)
2. Using regular K-Fold for imbalanced classification (use Stratified)
3. Using regular K-Fold for time-series (use TimeSeriesSplit)
4. Reporting only mean without standard deviation

### Interview Tip
"The correct way: preprocessing must happen INSIDE each fold. Otherwise, validation fold has seen information from training data through scaling/feature selection."

---

## Question 4
- [ ] Done

**Explain the concept of the ROC curve and AUC.**

**Answer:**

### Definition
**ROC (Receiver Operating Characteristic) Curve**: A plot showing the tradeoff between True Positive Rate (sensitivity) and False Positive Rate (1-specificity) at all possible classification thresholds.

**AUC (Area Under the ROC Curve)**: A single number summarizing the entire ROC curve. It represents the probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example.

### Core Concepts

**Building Blocks:**
```
                    Predicted
                 Positive  Negative
Actual Positive    TP        FN
Actual Negative    FP        TN
```

$$\text{True Positive Rate (TPR)} = \frac{TP}{TP + FN} = \text{Recall}$$

$$\text{False Positive Rate (FPR)} = \frac{FP}{FP + TN}$$

### How ROC Curve is Created

**Algorithm:**
```
1. Model outputs probability scores for each sample
2. Sort samples by probability (high to low)
3. For each unique probability as threshold:
   - Classify samples above threshold as positive
   - Calculate TPR and FPR
   - Plot point (FPR, TPR)
4. Connect all points to form ROC curve
```

**Visual:**
```
TPR (Sensitivity)
     │
  1.0├─────────────────●  ← Perfect classifier (0,1)
     │              ●
     │           ●    ← Good model curves toward top-left
     │        ●
     │     ●
     │  ●
  0.5├●─────────────────  ← Random classifier (diagonal)
     │
     └────────────────────
    0                   1.0  FPR (1-Specificity)
```

### AUC Interpretation

| AUC Value | Interpretation |
|-----------|----------------|
| 0.5 | Random guessing (no discrimination) |
| 0.6-0.7 | Poor |
| 0.7-0.8 | Fair |
| 0.8-0.9 | Good |
| 0.9-1.0 | Excellent |
| 1.0 | Perfect classifier |

**Probabilistic Interpretation:**
AUC = P(score of random positive > score of random negative)

Example: AUC = 0.85 means if you pick a random positive and random negative sample, 85% of the time the positive will have a higher predicted probability.

### Why Use ROC-AUC?

**Advantages:**
- Threshold-independent: Evaluates model across all thresholds
- Class-imbalance robust (somewhat): Uses rates, not raw counts
- Good for comparing models regardless of threshold choice

**When to Use:**
- Binary classification
- When you need to compare model ranking ability
- When threshold will be chosen later based on business needs

### Python Code Example

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get probability predictions
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1

# Calculate ROC curve points
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate AUC
auc = roc_auc_score(y_test, y_prob)

# Plot ROC curve
plt.plot(fpr, tpr, label=f'Model (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

### ROC vs Precision-Recall Curve

| ROC Curve | PR Curve |
|-----------|----------|
| Works well for balanced data | Better for imbalanced data |
| Can be optimistic with imbalance | More informative when positives are rare |
| Shows FPR which matters when negatives >> positives | Focuses on positive class performance |

### Common Pitfalls
1. Using accuracy when data is imbalanced (use AUC instead)
2. Relying only on AUC for highly imbalanced data (check PR-AUC too)
3. Forgetting AUC measures ranking, not calibration
4. Comparing AUC across different datasets (not meaningful)

### Interview Tip
"For imbalanced datasets (e.g., fraud detection with 1% fraud), Precision-Recall curve is more informative than ROC because ROC can look good even when model performs poorly on the minority class."

---

## Question 5
- [ ] Done

**What is meant by ‘overfitting’ and ‘underfitting’ in machine learning models?**

**Answer:**

### Overfitting
- Model learns noise in training data
- High training accuracy, low test accuracy
- Too complex model

### Underfitting
- Model fails to capture patterns
- Poor on both training and test
- Too simple model

### Solutions
- Overfitting: Regularization, more data, simpler model
- Underfitting: More features, complex model, longer training

### Interview Tip
Use validation curves to diagnose which problem you have.

---

## Question 6
- [ ] Done

**What is the difference between explained variance and R-squared?**

**Answer:**

### R-squared
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

### Explained Variance
$$EV = 1 - \frac{Var(y - \hat{y})}{Var(y)}$$

### Key Difference
They differ when predictions are biased.

### Interview Tip
Use adjusted R² when comparing models with different feature counts.

---

## Question 7
- [ ] Done

**Explain the use of the Mean Squared Error (MSE) in regression models.**

**Answer:**

### Formula
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

### Properties
- Always non-negative
- Penalizes large errors quadratically
- Units are squared

### Related Metrics
RMSE = √MSE (interpretable units)

### Interview Tip
Use RMSE for interpretation, MAE when outliers are concern.

---

## Question 8
- [ ] Done

**What is the distinction between macro-average and micro-average in classification metrics?**

**Answer:**

### Comparison

| Macro | Micro |
|-------|-------|
| Average per class | Aggregate all samples |
| Equal class weight | Weight by frequency |

### Interview Tip
Use macro when all classes equally important.

---

## Question 9
- [ ] Done

**What is the Brier score, and when would you use it?**

**Answer:**

### Definition
Mean squared error for probability predictions:
$$Brier = \frac{1}{n}\sum(p_i - o_i)^2$$

### Interpretation
Lower is better (0 = perfect, 1 = worst)

### Interview Tip
Use for evaluating probability calibration.

---

## Question 10
- [ ] Done

**Describe how you would use bootstrapping in model evaluation.**

**Answer:**

### Method
1. Sample with replacement from data
2. Train model on sample
3. Evaluate on out-of-bag samples
4. Repeat many times (100-1000)
5. Compute confidence intervals

### Interview Tip
Bootstrapping gives confidence intervals for metrics.

---

## Question 11
- [ ] Done

**What are the trade-offs between the different model evaluation metrics?**

**Answer:**

### Classification Trade-offs

| Metric | Pros | Cons |
|--------|------|------|
| Accuracy | Simple | Misleading for imbalanced |
| Precision | FP focus | Ignores FN |
| Recall | FN focus | Ignores FP |
| F1 | Balances P & R | Ignores TN |
| AUC | Threshold-independent | Less interpretable |

### Interview Tip
Choose metric based on business cost of errors.

---

## Question 12
- [ ] Done

**Explain the concept of p-value in the context of model evaluation.**

**Answer:**

### Definition
Probability of observing results as extreme, assuming null hypothesis is true.

### In Model Evaluation
- Compare models statistically
- McNemar's test for classifiers
- p < 0.05: Typically significant

### Interview Tip
Statistical significance ≠ practical significance.

---

## Question 13
- [ ] Done

**What is a receiver operating characteristic (ROC) curve, and what does it tell us?**

**Answer:**

### Components
- X-axis: False Positive Rate
- Y-axis: True Positive Rate
- Each point = different threshold

### Interpretation
- Diagonal = random classifier
- Upper left = perfect classifier

### Interview Tip
Use PR curves for highly imbalanced datasets.

---

## Question 14
- [ ] Done

**Describe model selection criteria based on AIC and BIC.**

**Answer:**

### Formulas
- AIC = 2k - 2ln(L)
- BIC = k·ln(n) - 2ln(L)

### Comparison
- AIC: Less penalty, favors complex models
- BIC: More penalty, favors simpler models

### Interview Tip
Lower AIC/BIC is better.

---

## Question 15
- [ ] Done

**What is the Elbow Method, and how is it used to evaluate models?**

**Answer:**

### Application
Determining optimal number of clusters in K-Means.

### Process
1. Plot k vs inertia
2. Find the "elbow" where improvement slows
3. Choose k at the elbow point

### Interview Tip
Combine with silhouette score for reliability.

---

## Question 16
- [ ] Done

**What is the best way to evaluate a recommendation system?**

**Answer:**

### Offline Metrics
- Precision@K, Recall@K
- NDCG (rank-aware)
- MAP (Mean Average Precision)

### Online Metrics
- Click-through rate
- Conversion rate

### Interview Tip
A/B testing is crucial for recommendation systems.

---

## Question 17
- [ ] Done

**Describe a method for evaluating the performance of a clustering algorithm.**

**Answer:**

### Internal Metrics (No Labels)
- Silhouette Score: Cohesion vs separation
- Davies-Bouldin Index: Lower is better
- Calinski-Harabasz: Higher is better

### Interview Tip
Use silhouette score for unsupervised evaluation.

---

## Question 18
- [ ] Done

**Describe how you would set up an A/B test to evaluate changes in a machine learning model.**

**Answer:**

### Steps
1. Define hypothesis and success metric
2. Calculate sample size needed
3. Randomize users into control/treatment
4. Deploy old model (A) and new model (B)
5. Collect data for sufficient duration
6. Analyze with statistical tests

### Interview Tip
Ensure statistical power before running test.

---

## Question 19
- [ ] Done

**How does transfer learning affect the way we evaluate models?**

**Answer:**

### Considerations
- Evaluate on target domain specifically
- Compare with domain-specific baseline
- Check for negative transfer

### Interview Tip
Pre-trained features may not transfer to very different domains.

---

## Question 20
- [ ] Done

**What are ensemble learning models, and how do their evaluation strategies differ?**

**Answer:**

### Types
- Bagging (Random Forest)
- Boosting (XGBoost)
- Stacking

### Evaluation Differences
- Out-of-bag (OOB) error for bagging
- Validation during boosting rounds
- Nested CV for stacking

### Interview Tip
OOB error is a free validation estimate for Random Forest.

---

## Question 21
- [ ] Done

**Explain adversarial validation and where it might be used.**

**Answer:**

### Definition
Train a classifier to distinguish train from test data.

### Process
1. Label train data as 0, test as 1
2. Train binary classifier
3. If AUC ≈ 0.5 → distributions similar
4. If AUC >> 0.5 → distribution shift exists

### Interview Tip
High adversarial AUC indicates potential problem.

---

## Question 22
- [ ] Done

**What is the concept of ‘model drift’, and how do you measure it?**

**Answer:**

### Types
- **Concept drift**: Relationship between X and y changes
- **Data drift**: Input distribution changes

### Detection Methods
- Monitor prediction distributions
- Track performance metrics over time
- Statistical tests (KS test, PSI)

### Interview Tip
Set up monitoring and alerting for production models.

---

