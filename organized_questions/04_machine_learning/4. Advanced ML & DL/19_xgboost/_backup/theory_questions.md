# XGBoost Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What is XGBoost and why is it considered an effective machine learning algorithm?**

### Answer

**Definition:**
XGBoost (eXtreme Gradient Boosting) is an optimized, scalable gradient boosting library that implements machine learning algorithms under the gradient boosting framework. It's known for speed, performance, and regularization capabilities.

**Core Concepts:**
- Sequential ensemble of weak learners (decision trees)
- Additive training: each tree corrects previous errors
- Second-order gradient optimization (uses Hessian)
- Built-in regularization (L1 and L2)
- Sparsity-aware split finding

**Why It's Effective:**

| Feature | Benefit |
|---------|---------|
| **Regularization** | Prevents overfitting |
| **Parallel processing** | Faster training |
| **Tree pruning** | Optimal tree structure |
| **Built-in CV** | Easy model selection |
| **Missing value handling** | No imputation needed |
| **Cache optimization** | Efficient memory usage |

**Mathematical Formulation:**
$$\mathcal{L}(\phi) = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)$$

Where:
- $l$ = loss function (e.g., squared error)
- $\Omega(f) = \gamma T + \frac{1}{2}\lambda||w||^2$ = regularization term
- $T$ = number of leaves, $w$ = leaf weights

**Practical Relevance:**
- Top performer in Kaggle competitions
- Handles tabular data exceptionally well
- Supports classification, regression, ranking

---

## Question 2
- [ ] Done

**Can you explain the differences between gradient boosting machines (GBM) and XGBoost?**

### Answer

**Definition:**
XGBoost is an optimized implementation of gradient boosting with additional features like regularization, parallel processing, and advanced tree-building algorithms that traditional GBM lacks.

**Key Differences:**

| Aspect | Traditional GBM | XGBoost |
|--------|-----------------|---------|
| **Regularization** | None | L1 + L2 regularization |
| **Gradient Order** | First-order | Second-order (Hessian) |
| **Tree Building** | Greedy | Level-wise with pruning |
| **Missing Values** | Requires imputation | Handles natively |
| **Parallelization** | Sequential | Parallel at node level |
| **Sparsity** | Not optimized | Sparsity-aware |
| **Cache** | Basic | Cache-optimized |
| **Cross-validation** | External | Built-in |

**Objective Function Comparison:**

**GBM:**
$$\mathcal{L} = \sum_i l(y_i, \hat{y}_i)$$

**XGBoost:**
$$\mathcal{L} = \sum_i l(y_i, \hat{y}_i) + \gamma T + \frac{1}{2}\lambda \sum_j w_j^2$$

**Gradient Optimization:**

**GBM:** Uses first derivative only
$$f_m(x) = f_{m-1}(x) + \nu \cdot h_m(x)$$

**XGBoost:** Uses both first and second derivatives
$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$$

Where $G$ = gradient sum, $H$ = Hessian sum.

**Interview Point:**
"XGBoost adds regularization to prevent overfitting and uses second-order gradients for better optimization. It's also engineered for speed with parallel processing and cache optimization."

---

## Question 3
- [ ] Done

**How does XGBoost handle missing or null values in the dataset?**

### Answer

**Definition:**
XGBoost handles missing values by learning the optimal default direction for missing values at each split during training. It doesn't require imputation.

**Mechanism:**

1. At each split, XGBoost tries both directions for missing values:
   - Send missing to left child
   - Send missing to right child
2. Chooses direction that maximizes gain
3. Stores optimal direction in the tree

**Algorithm (Sparsity-Aware Split Finding):**
```
For each split candidate:
    Compute gain if missing → left
    Compute gain if missing → right
    Choose direction with higher gain
    Store default direction
```

**Example:**
```python
import xgboost as xgb
import numpy as np

# Data with missing values (no imputation needed)
X = np.array([[1, 2], [np.nan, 3], [4, np.nan], [5, 6]])
y = np.array([0, 1, 0, 1])

# XGBoost handles missing automatically
model = xgb.XGBClassifier()
model.fit(X, y)  # Works without imputation

# Prediction with missing values
X_test = np.array([[np.nan, 5]])
pred = model.predict(X_test)  # Handles missing in prediction too
```

**Benefits:**
- No information loss from imputation
- Learns from missingness pattern (informative missingness)
- Consistent handling in train and test

**When to Still Impute:**
- Very high missing rate (>50%)
- Domain knowledge suggests specific imputation
- Comparing with models that require imputation

---

## Question 4
- [ ] Done

**What is meant by 'regularization' in XGBoost and how does it help in preventing overfitting?**

### Answer

**Definition:**
Regularization in XGBoost adds penalty terms to the objective function that discourage complex models, helping prevent overfitting by penalizing large leaf weights and too many leaves.

**Regularization Terms:**

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2 + \alpha \sum_{j=1}^{T} |w_j|$$

Where:
- $T$ = number of leaves (complexity penalty via $\gamma$)
- $w_j$ = weight of leaf j
- $\lambda$ = L2 regularization (Ridge)
- $\alpha$ = L1 regularization (Lasso)

**Regularization Parameters:**

| Parameter | Name | Effect |
|-----------|------|--------|
| `gamma` | min_split_loss | Minimum gain required for split |
| `lambda` | reg_lambda | L2 regularization on weights |
| `alpha` | reg_alpha | L1 regularization on weights |
| `max_depth` | - | Limits tree depth |
| `min_child_weight` | - | Minimum Hessian sum in child |

**How They Help:**

**1. gamma (γ):**
- Split only if gain > γ
- Higher γ = simpler trees (fewer splits)

**2. lambda (λ) - L2:**
- Shrinks leaf weights toward zero
- Prevents extreme predictions
- Smoother output

**3. alpha (α) - L1:**
- Can zero out some leaf weights
- Feature selection effect
- Sparser model

**Code Example:**
```python
import xgboost as xgb

# Regularized XGBoost
model = xgb.XGBClassifier(
    reg_lambda=1.0,      # L2 regularization
    reg_alpha=0.1,       # L1 regularization
    gamma=0.5,           # Min split gain
    max_depth=5,         # Depth limit
    min_child_weight=5   # Min samples per leaf
)
```

**Interview Point:**
"XGBoost's regularization is a key differentiator from GBM. I typically start with lambda=1, alpha=0, gamma=0 and increase if overfitting."

---

## Question 5
- [ ] Done

**How does XGBoost differ from Random Forests?**

### Answer

**Definition:**
XGBoost uses sequential boosting (trees correct previous errors) while Random Forest uses parallel bagging (independent trees vote). This fundamental difference leads to different strengths and use cases.

**Key Differences:**

| Aspect | XGBoost | Random Forest |
|--------|---------|---------------|
| **Method** | Boosting (sequential) | Bagging (parallel) |
| **Trees** | Shallow, dependent | Deep, independent |
| **What it Reduces** | Bias (primarily) | Variance |
| **Learning** | Additive, corrective | Independent, averaging |
| **Regularization** | Built-in (L1, L2) | Via tree structure |
| **Speed (Training)** | Slower (sequential) | Faster (parallel) |
| **Speed (Inference)** | Similar | Similar |
| **Sensitivity to Noise** | Higher | Lower |
| **Hyperparameter Sensitivity** | Higher | Lower |

**When to Use Each:**

| Scenario | Recommended |
|----------|-------------|
| Need best accuracy | XGBoost |
| Noisy data | Random Forest |
| Quick baseline | Random Forest |
| Time for tuning | XGBoost |
| Interpretability | Random Forest |
| Feature importance | Both work well |

**Mathematical Comparison:**

**Random Forest:**
$$\hat{f}(x) = \frac{1}{B}\sum_{b=1}^{B} T_b(x) \quad \text{(average)}$$

**XGBoost:**
$$\hat{f}(x) = \sum_{m=1}^{M} \eta \cdot h_m(x) \quad \text{(additive)}$$

**Practical Tips:**
- Start with Random Forest for baseline
- Try XGBoost if RF performance is insufficient
- XGBoost often wins with proper tuning
- RF is more robust out-of-the-box

---

## Question 6
- [ ] Done

**Explain the concept of gradient boosting. How does it work in the context of XGBoost?**

### Answer

**Definition:**
Gradient boosting builds an ensemble by sequentially adding trees that predict the negative gradient (residuals) of the loss function. Each new tree corrects errors made by all previous trees combined.

**Core Concept:**

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

Where:
- $F_m$ = ensemble after m trees
- $h_m$ = new tree predicting residuals
- $\eta$ = learning rate (shrinkage)

**Algorithm Steps:**

```
1. Initialize: F_0(x) = constant (e.g., mean of y)

2. For m = 1 to M:
   a. Compute pseudo-residuals:
      r_im = -[∂L(y_i, F(x_i))/∂F(x_i)] at F=F_{m-1}
   
   b. Fit tree h_m to residuals r_im
   
   c. Update model:
      F_m(x) = F_{m-1}(x) + η · h_m(x)

3. Output: F_M(x)
```

**XGBoost Enhancements:**

**1. Second-Order Approximation:**
$$\mathcal{L}^{(t)} \approx \sum_i [g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)] + \Omega(f_t)$$

Where:
- $g_i = \partial l / \partial \hat{y}$ (gradient)
- $h_i = \partial^2 l / \partial \hat{y}^2$ (Hessian)

**2. Optimal Leaf Weight:**
$$w_j^* = -\frac{G_j}{H_j + \lambda}$$

**3. Split Gain:**
$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G^2}{H+\lambda}\right] - \gamma$$

**Intuition:**
"Imagine aiming at a target. First shot hits roughly. Each subsequent shot aims at the remaining distance from the target, getting progressively closer."

---

## Question 7
- [ ] Done

**What are the loss functions used in XGBoost for regression and classification problems?**

### Answer

**Definition:**
XGBoost supports various loss functions (objectives) tailored to different tasks. The loss function determines what the model optimizes and how gradients are computed.

**Common Loss Functions:**

**Regression:**

| Objective | Formula | Use Case |
|-----------|---------|----------|
| `reg:squarederror` | $\frac{1}{2}(y - \hat{y})^2$ | Standard regression |
| `reg:squaredlogerror` | $\frac{1}{2}[\log(\hat{y}+1) - \log(y+1)]^2$ | Targets with large range |
| `reg:pseudohubererror` | Huber-like | Robust to outliers |
| `reg:absoluteerror` | $\|y - \hat{y}\|$ | MAE, outlier-robust |
| `reg:gamma` | Gamma deviance | Positive continuous targets |
| `reg:tweedie` | Tweedie deviance | Insurance claims |

**Binary Classification:**

| Objective | Formula | Use Case |
|-----------|---------|----------|
| `binary:logistic` | $y\log(p) + (1-y)\log(1-p)$ | Binary classification |
| `binary:hinge` | Hinge loss | SVM-like |

**Multi-class Classification:**

| Objective | Use Case |
|-----------|----------|
| `multi:softmax` | Returns class labels |
| `multi:softprob` | Returns probabilities |

**Code Examples:**

```python
import xgboost as xgb

# Regression
reg_model = xgb.XGBRegressor(objective='reg:squarederror')

# Binary classification
clf_model = xgb.XGBClassifier(objective='binary:logistic')

# Multi-class (3 classes)
multi_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3
)

# Custom objective (example: asymmetric loss)
def custom_loss(y_true, y_pred):
    grad = np.where(y_true > y_pred, -2*(y_true-y_pred), -0.5*(y_true-y_pred))
    hess = np.where(y_true > y_pred, 2, 0.5)
    return grad, hess

model = xgb.XGBRegressor(objective=custom_loss)
```

**Choosing Loss Function:**
- Standard tasks: Use defaults
- Outliers in regression: Use Huber or MAE
- Positive targets: Consider Gamma or Tweedie
- Class imbalance: Adjust scale_pos_weight

---

## Question 8
- [ ] Done

**How does XGBoost use tree pruning and why is it important?**

### Answer

**Definition:**
XGBoost uses "max_depth + post-pruning" strategy: it grows trees to maximum depth first, then prunes back splits that don't improve the objective by at least gamma. This is more effective than pre-pruning (stopping early).

**Pruning Mechanism:**

**1. Grow Phase:**
- Build tree to max_depth
- Evaluate all possible splits

**2. Prune Phase:**
- Starting from leaves, remove splits where:
$$\text{Gain} < \gamma$$

**Gain Formula:**
$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G^2}{H+\lambda}\right] - \gamma$$

If Gain ≤ 0, the split is pruned.

**Why Post-Pruning is Better:**

| Pre-Pruning | Post-Pruning (XGBoost) |
|-------------|------------------------|
| Stops when gain < threshold | Grows fully, then prunes |
| May miss beneficial splits | Considers deeper patterns |
| Greedy decision | Global optimization |

**Example:**
```
Pre-pruning might stop here (low immediate gain):
    [Node A] → gain=0.1 (stop)
    
But the children might have high gain:
    [Node A] → gain=0.1
       ├── [Node B] → gain=2.0
       └── [Node C] → gain=1.5

Post-pruning keeps the beneficial splits.
```

**Controlling Pruning:**

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    max_depth=6,        # Initial tree depth
    gamma=0.5,          # Min split gain (pruning threshold)
    min_child_weight=5  # Min Hessian sum in child
)
```

**Interview Point:**
"XGBoost's post-pruning allows it to discover patterns that require multiple splits to be useful, which pre-pruning would miss."

---

## Question 9
- [ ] Done

**Describe the role of shrinkage (learning rate) in XGBoost.**

### Answer

**Definition:**
Shrinkage (learning rate, eta/η) scales down the contribution of each tree, requiring more trees to reach the same solution but with better generalization. It's a form of regularization.

**Mathematical Role:**
$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

Where η ∈ (0, 1], typically 0.01-0.3.

**Effect of Learning Rate:**

| Low η (0.01-0.1) | High η (0.3-1.0) |
|------------------|------------------|
| Needs more trees | Needs fewer trees |
| Better generalization | Risk of overfitting |
| Slower training | Faster training |
| Smoother learning | Aggressive learning |
| More robust | Can overshoot optimal |

**Tradeoff with n_estimators:**
```
High η + Few trees = Fast, may overfit
Low η + Many trees = Slow, better generalization
```

**Rule of Thumb:**
- Lower η + more trees usually performs better
- But: training time increases
- Sweet spot: η = 0.1-0.3, tune n_estimators

**Code Example:**

```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# Compare different learning rates
for eta in [0.01, 0.1, 0.3]:
    model = xgb.XGBClassifier(
        learning_rate=eta,
        n_estimators=int(100 / eta),  # Compensate
        early_stopping_rounds=10
    )
    scores = cross_val_score(model, X, y, cv=5, 
                             fit_params={'eval_set': [(X, y)]})
    print(f"eta={eta}: {scores.mean():.4f}")
```

**Best Practice:**
1. Start with η = 0.1, n_estimators = 100
2. Use early stopping to find optimal n_estimators
3. Lower η if overfitting, increase n_estimators proportionally

---

## Question 10
- [ ] Done

**What are the core parameters in XGBoost that you often consider tuning?**

### Answer

**Definition:**
XGBoost has many parameters, but a core set significantly impacts performance: learning_rate, n_estimators, max_depth, subsample, colsample_bytree, and regularization parameters.

**Parameter Categories:**

**1. Tree Structure:**
| Parameter | Range | Effect |
|-----------|-------|--------|
| `max_depth` | 3-10 | Tree complexity |
| `min_child_weight` | 1-10 | Min Hessian sum in leaf |
| `gamma` | 0-5 | Min split gain |

**2. Sampling:**
| Parameter | Range | Effect |
|-----------|-------|--------|
| `subsample` | 0.5-1.0 | Row sampling ratio |
| `colsample_bytree` | 0.5-1.0 | Feature sampling per tree |
| `colsample_bylevel` | 0.5-1.0 | Feature sampling per level |

**3. Regularization:**
| Parameter | Range | Effect |
|-----------|-------|--------|
| `reg_lambda` (λ) | 0-10 | L2 regularization |
| `reg_alpha` (α) | 0-1 | L1 regularization |

**4. Learning:**
| Parameter | Range | Effect |
|-----------|-------|--------|
| `learning_rate` (η) | 0.01-0.3 | Step size shrinkage |
| `n_estimators` | 100-1000 | Number of trees |

**Tuning Strategy:**

```python
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Step 1: Tune tree structure
param_grid_1 = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5]
}

# Step 2: Tune sampling
param_grid_2 = {
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Step 3: Tune regularization
param_grid_3 = {
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 5, 10]
}

# Step 4: Lower learning rate, increase trees
final_model = xgb.XGBClassifier(
    learning_rate=0.05,
    n_estimators=500,
    early_stopping_rounds=50,
    # Best params from above steps
)
```

**Quick Start Defaults:**
```python
xgb.XGBClassifier(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1,
    reg_alpha=0
)
```

---

## Question 11
- [ ] Done

**Explain the importance of the 'max_depth' parameter in XGBoost.**

### Answer

**Definition:**
`max_depth` controls the maximum depth of each tree in the ensemble. It's one of the most important parameters for controlling model complexity and preventing overfitting.

**Impact:**

| Low max_depth (2-4) | High max_depth (8+) |
|---------------------|---------------------|
| Simple trees | Complex trees |
| High bias | Low bias |
| Low variance | High variance |
| Underfitting risk | Overfitting risk |
| Fast training | Slower training |
| Good for large datasets | Risk for small datasets |

**Mathematical Perspective:**

Number of possible leaves: up to $2^{\text{max\_depth}}$
- max_depth=3 → up to 8 leaves
- max_depth=6 → up to 64 leaves
- max_depth=10 → up to 1024 leaves

**Recommended Values:**

| Dataset Size | Recommended max_depth |
|--------------|----------------------|
| Small (<1K) | 3-4 |
| Medium (1K-100K) | 4-6 |
| Large (>100K) | 6-10 |

**Tuning Example:**

```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# Find optimal max_depth
for depth in [3, 4, 5, 6, 7, 8]:
    model = xgb.XGBClassifier(
        max_depth=depth,
        n_estimators=100,
        learning_rate=0.1
    )
    scores = cross_val_score(model, X, y, cv=5)
    print(f"max_depth={depth}: {scores.mean():.4f} ± {scores.std():.4f}")
```

**Interaction with Other Parameters:**
- Higher max_depth → need more regularization (gamma, lambda)
- Higher max_depth → can use lower n_estimators
- Lower learning_rate → can tolerate higher max_depth

**Interview Point:**
"I typically start with max_depth=5-6 and tune based on validation performance. Lower depth with more trees often generalizes better than deep trees with few rounds."

---

## Question 12
- [ ] Done

**How does the objective function affect the performance of the XGBoost model?**

### Answer

**Definition:**
The objective function defines what the model optimizes. It consists of a loss function (measuring prediction error) and a regularization term (controlling complexity). Different objectives suit different problem types.

**Objective Structure:**
$$\text{Obj}(\Theta) = L(\Theta) + \Omega(\Theta)$$

- $L(\Theta)$ = Loss function (task-dependent)
- $\Omega(\Theta)$ = Regularization (tree complexity)

**Effect on Model:**

**1. Gradient Computation:**
Different objectives produce different gradients:
- Squared error: $g_i = \hat{y}_i - y_i$
- Logistic: $g_i = \text{sigmoid}(\hat{y}_i) - y_i$

**2. Hessian Computation:**
Second derivative affects optimal leaf weights:
- Squared error: $h_i = 1$
- Logistic: $h_i = p_i(1-p_i)$

**3. Model Behavior:**

| Objective | Behavior |
|-----------|----------|
| `squarederror` | Sensitive to outliers |
| `squaredlogerror` | Better for large target ranges |
| `logistic` | Outputs probabilities |
| `hinge` | Focuses on decision boundary |

**Choosing the Right Objective:**

```python
import xgboost as xgb

# Regression with outliers
model = xgb.XGBRegressor(objective='reg:pseudohubererror')

# Binary classification
model = xgb.XGBClassifier(objective='binary:logistic')

# Ranking
model = xgb.XGBRanker(objective='rank:pairwise')

# Custom objective
def weighted_mse(y_true, y_pred):
    weights = np.where(y_true > y_pred, 2, 1)  # Penalize under-prediction
    grad = weights * (y_pred - y_true)
    hess = weights * np.ones_like(y_true)
    return grad, hess

model = xgb.XGBRegressor(objective=weighted_mse)
```

**Performance Impact:**
- Wrong objective → suboptimal learning direction
- Custom objectives → domain-specific optimization
- Matching objective to evaluation metric → better alignment

---

## Question 13
- [ ] Done

**How does the DART booster in XGBoost work and what's its use case?**

### Answer

**Definition:**
DART (Dropouts meet Multiple Additive Regression Trees) applies dropout, a technique from neural networks, to gradient boosting. It randomly drops trees when computing gradients, reducing overfitting.

**How DART Works:**

1. At each iteration, randomly select subset of existing trees to drop
2. Compute gradients using only non-dropped trees
3. Train new tree to fit these residuals
4. Add new tree back with normalization

**Key Parameters:**

| Parameter | Description |
|-----------|-------------|
| `rate_drop` | Fraction of trees to drop (0-1) |
| `skip_drop` | Probability of skipping dropout (0-1) |
| `sample_type` | 'uniform' or 'weighted' |
| `normalize_type` | 'tree' or 'forest' |

**Code Example:**

```python
import xgboost as xgb

# DART booster
model = xgb.XGBClassifier(
    booster='dart',
    rate_drop=0.1,      # Drop 10% of trees
    skip_drop=0.5,      # 50% chance to skip dropout
    n_estimators=200,
    learning_rate=0.1
)
```

**DART vs GBTree:**

| Aspect | GBTree | DART |
|--------|--------|------|
| Regularization | Via parameters | Via dropout |
| Over-specialization | Possible | Reduced |
| Training speed | Faster | Slower |
| Early stopping | Works well | Can be tricky |

**When to Use DART:**
- When gbtree overfits despite regularization
- When trees over-specialize (each tree does too specific corrections)
- Not recommended with early stopping (randomness makes validation unstable)

**Interview Point:**
"DART prevents over-specialization where later trees only fix very specific errors. It's like ensemble dropout, but I use it only when standard regularization isn't enough."

---

## Question 14
- [ ] Done

**What is the column subsampling feature in XGBoost and how does it help?**

### Answer

**Definition:**
Column subsampling (feature bagging) randomly selects a subset of features at different levels of tree building, adding randomness and reducing overfitting, similar to Random Forest's feature randomness.

**Subsampling Levels:**

| Parameter | When Applied |
|-----------|--------------|
| `colsample_bytree` | Once per tree |
| `colsample_bylevel` | At each depth level |
| `colsample_bynode` | At each split |

**Example with 10 features:**
```
colsample_bytree=0.8 → Use 8 features per tree
colsample_bylevel=0.8 → Of those 8, use 6-7 per level
colsample_bynode=0.8 → Of remaining, use subset per split
```

**Combined Effect:**
```python
# With 100 features:
# colsample_bytree=0.8 → 80 features for this tree
# colsample_bylevel=0.8 → 64 features at each level
# colsample_bynode=0.8 → 51 features considered per split
```

**Benefits:**

1. **Reduces Overfitting:**
   - Forces model to use diverse features
   - No single feature dominates

2. **Decorrelates Trees:**
   - Different trees see different feature subsets
   - Better ensemble diversity

3. **Speeds Up Training:**
   - Fewer features to evaluate per split

**Code Example:**

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    colsample_bytree=0.8,    # 80% of features per tree
    colsample_bylevel=0.8,   # 80% at each level
    colsample_bynode=1.0,    # All remaining at each node
    n_estimators=100
)
```

**Tuning Guidelines:**
- Start with colsample_bytree=0.8
- Lower values if overfitting persists
- Very low values (<0.5) may hurt accuracy
- Works well with subsample (row sampling)

---

## Question 15
- [ ] Done

**Explain the concept of 'subsample' in XGBoost and its benefits.**

### Answer

**Definition:**
`subsample` controls the fraction of training samples used to train each tree (row sampling). A value of 0.8 means each tree sees a random 80% of the training data.

**Mechanism:**
```
Training data: 1000 samples
subsample=0.8

Tree 1: Random 800 samples (samples 1-800 shuffled)
Tree 2: Different random 800 samples
Tree 3: Different random 800 samples
...
```

**Benefits:**

**1. Reduces Overfitting:**
- Each tree sees different data
- Averages out noise-fitted patterns

**2. Adds Stochasticity:**
- More variance in individual trees
- Better generalization in ensemble

**3. Faster Training:**
- Less data per tree = faster iterations
- Especially helpful for large datasets

**Comparison with Random Forest:**

| Aspect | XGBoost subsample | Random Forest bootstrap |
|--------|-------------------|------------------------|
| Default | 1.0 (no sampling) | 1.0 (with replacement) |
| Method | Without replacement | With replacement |
| Effect | Stochasticity | Bagging effect |

**Code Example:**

```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# Compare subsample values
for ss in [0.6, 0.8, 1.0]:
    model = xgb.XGBClassifier(
        subsample=ss,
        n_estimators=100,
        learning_rate=0.1
    )
    scores = cross_val_score(model, X, y, cv=5)
    print(f"subsample={ss}: {scores.mean():.4f}")
```

**Guidelines:**
- Default: 1.0 (use all data)
- If overfitting: Try 0.7-0.9
- Large datasets: Can go lower (0.5-0.7)
- Very small datasets: Keep at 1.0

**Combined with colsample:**
```python
# Stochastic Gradient Boosting
model = xgb.XGBClassifier(
    subsample=0.8,          # 80% of rows
    colsample_bytree=0.8,   # 80% of columns
    n_estimators=200
)
```

---

## Question 16
- [ ] Done

**How does XGBoost handle categorical features?**

### Answer

**Definition:**
XGBoost doesn't natively handle categorical features like CatBoost. Categorical variables must be encoded before training. However, XGBoost 1.5+ added experimental native categorical support.

**Traditional Approaches:**

**1. Label Encoding:**
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X['category_encoded'] = le.fit_transform(X['category'])

# Works for ordinal categories or when cardinality is low
```

**2. One-Hot Encoding:**
```python
import pandas as pd

X_encoded = pd.get_dummies(X, columns=['category'])

# Good for low-cardinality categoricals
# Can create many features for high cardinality
```

**3. Target Encoding:**
```python
# Mean target value per category
target_means = y.groupby(X['category']).mean()
X['category_encoded'] = X['category'].map(target_means)

# Good for high cardinality
# Use with cross-validation to avoid leakage
```

**Native Categorical Support (XGBoost 1.5+):**

```python
import xgboost as xgb
import pandas as pd

# Convert to category dtype
X['cat_col'] = X['cat_col'].astype('category')

# Enable categorical feature support
model = xgb.XGBClassifier(
    tree_method='hist',  # Required for native categorical
    enable_categorical=True
)
model.fit(X, y)
```

**Comparison:**

| Method | Pros | Cons |
|--------|------|------|
| Label Encoding | Simple, no expansion | Implies ordering |
| One-Hot | No ordering assumption | High cardinality explosion |
| Target Encoding | Handles high cardinality | Risk of leakage |
| Native (1.5+) | Optimal splits | Still experimental |

**Recommendation:**
- Low cardinality (<10): One-hot encoding
- Medium cardinality: Label or target encoding
- High cardinality: Target encoding or native support

---

## Question 17
- [ ] Done

**What is the weighted quantile sketch in XGBoost?**

### Answer

**Definition:**
Weighted quantile sketch is XGBoost's algorithm for finding approximate split points efficiently. It handles weighted instances and enables distributed/parallel computation by summarizing data distribution.

**Why It's Needed:**

Exact split finding:
- Check every unique value as potential threshold
- O(n × features) per level
- Infeasible for large datasets

Quantile sketch:
- Approximate data distribution with buckets
- Check only bucket boundaries
- O(buckets × features) per level

**How It Works:**

1. **Compute Quantiles:**
   - Divide data into approximate quantiles
   - Weight by Hessian (second derivative)

2. **Create Buckets:**
   - Each bucket represents a range of values
   - Boundaries are candidate split points

3. **Find Splits:**
   - Only evaluate splits at bucket boundaries
   - Dramatic speedup for large datasets

**Mathematical Basis:**

For weighted quantile:
$$r_k(z) = \frac{\sum_{(x,h) \in D: x<z} h}{\sum_{(x,h) \in D} h}$$

Find split points where:
$$|r_k(s_{j+1}) - r_k(s_j)| < \epsilon$$

**Parameter:**
```python
model = xgb.XGBClassifier(
    tree_method='approx',     # Use approximate algorithm
    max_bin=256               # Number of buckets (hist method)
)
```

**Tree Methods:**

| Method | Description |
|--------|-------------|
| `exact` | Enumerate all splits (small data) |
| `approx` | Quantile sketch (medium data) |
| `hist` | Histogram-based (large data, GPU) |

**Interview Point:**
"Weighted quantile sketch is key to XGBoost's scalability. It reduces split candidates from millions to hundreds while maintaining split quality."
