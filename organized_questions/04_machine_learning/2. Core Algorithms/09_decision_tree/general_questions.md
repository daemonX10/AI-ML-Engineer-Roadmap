# Decision Tree Interview Questions - General Questions

## Question 1
- [ ] Done

**Outline some limitations or disadvantages of Decision Trees.**

**Answer:**

Decision Trees have several key limitations: they tend to overfit (especially deep trees), are unstable (small data changes lead to different trees), create axis-parallel boundaries (inefficient for diagonal relationships), and single trees have high variance. They also struggle with extrapolation and can be biased toward features with many levels.

**Key Limitations:**

| Limitation | Description | Mitigation |
|------------|-------------|------------|
| **Overfitting** | Deep trees memorize training data | Pruning, max_depth |
| **High Variance** | Sensitive to data changes | Ensembles (Random Forest) |
| **Axis-Parallel Splits** | Can't capture diagonal boundaries efficiently | Oblique trees, ensembles |
| **Biased Splits** | Favor high-cardinality features | Gain Ratio, proper encoding |
| **No Extrapolation** | Predict only within training range | Poor for time-series trends |
| **Greedy Learning** | Locally optimal, not global | Ensemble methods |

**Detailed Explanations:**

**Instability:**
```
Original: Split on Age at 35
Remove one sample → Split on Age at 32
Small change → completely different tree
```

**Axis-Parallel Problem:**
Data with diagonal boundary needs many splits:
```
Efficient: One diagonal line
Tree: Staircase pattern of many axis-parallel splits
```

**Extrapolation:**
- Training data: Age 20-60 → Predictions bounded by training range
- Age 70 → Predicts same as Age 60 (cannot extrapolate)

**Interview Tip:**
Always mention that ensembles (Random Forest, GBDT) solve most single-tree limitations, showing you understand the full picture.

---

## Question 2
- [ ] Done

**Define Gini impurity and its role in Decision Trees.**

**Answer:**

Gini impurity measures the probability of incorrectly classifying a randomly chosen sample if it were labeled according to the class distribution at that node. It ranges from 0 (pure node) to 0.5 (binary, 50-50 split). CART algorithm uses Gini to select optimal splits by maximizing Gini reduction.

**Mathematical Formula:**
$$Gini(t) = 1 - \sum_{k=1}^{K} p_k^2$$

Where $p_k$ = proportion of class k samples at node t.

**Example Calculation:**

Node with 100 samples: 80 Class A, 20 Class B
$$Gini = 1 - (0.8^2 + 0.2^2) = 1 - (0.64 + 0.04) = 0.32$$

Pure node (100 Class A, 0 Class B):
$$Gini = 1 - (1^2 + 0^2) = 0$$

Maximum impurity (50-50):
$$Gini = 1 - (0.5^2 + 0.5^2) = 0.5$$

**Role in Splitting:**
$$GiniGain = Gini(parent) - \frac{n_{left}}{n}Gini(left) - \frac{n_{right}}{n}Gini(right)$$

Select split with maximum Gini Gain.

**Gini vs Entropy:**
- Both measure impurity
- Gini: Faster (no logarithm)
- Entropy: Information-theoretic meaning
- In practice: Similar results

---

## Question 3
- [ ] Done

**Can Decision Trees be used for multi-output tasks?**

**Answer:**

Yes, Decision Trees can handle multi-output tasks where multiple target variables are predicted simultaneously using a single tree structure. Instead of storing one class/value per leaf, each leaf stores a vector of predictions for all outputs.

**How It Works:**

1. **Split Criterion Modification:**
   - Impurity calculated across all outputs
   - Total impurity = sum/average of individual output impurities
   
2. **Leaf Prediction:**
   - Classification: Store mode of each output
   - Regression: Store mean of each output

**Types:**
| Task | Example | Leaf Output |
|------|---------|-------------|
| Multi-output Classification | Predict genre AND rating | [Comedy, PG-13] |
| Multi-output Regression | Predict height AND weight | [175.5, 72.3] |
| Multi-task | Mixed classification + regression | [Class_A, 45.2] |

**Sklearn Implementation:**

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Multi-output: predict 2 labels per sample
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([[0, 1], [0, 0], [1, 1], [1, 0]])  # 2 outputs

clf = DecisionTreeClassifier()
clf.fit(X, y)
print(clf.predict([[4, 5]]))  # Output: [[1, 0]]
```

**Advantages:**
- Captures output correlations
- Single model, simpler deployment
- Faster than training separate models

**Limitations:**
- Outputs must have similar feature relationships
- Complex output interactions may need separate models

---

## Question 4
- [ ] Done

**What modifications are done by the CHAID (Chi-squared Automatic Interaction Detector) algorithm in building Decision Trees?**

**Answer:**

CHAID is a Decision Tree algorithm that uses Chi-squared statistical tests to determine optimal splits. Unlike CART or ID3, it creates multi-way splits and was designed specifically for categorical data analysis.

**Key Modifications from Standard Decision Trees:**

| Feature | Standard Trees (CART) | CHAID |
|---------|----------------------|-------|
| Split Type | Binary | Multi-way |
| Criterion | Gini/Entropy | Chi-squared test |
| Variable Types | Any | Originally categorical |
| Statistical Test | None | p-value threshold |
| Category Merging | No | Yes |

**CHAID Algorithm Steps:**

1. **For each predictor variable:**
   - Create contingency table with target
   - Calculate Chi-squared statistic
   
2. **Category Merging:**
   - Merge categories that are NOT significantly different
   - Continue until all remaining categories differ significantly
   
3. **Variable Selection:**
   - Select predictor with smallest p-value (most significant)
   - Split on that variable

4. **Stopping Criteria:**
   - No significant splits available (p-value > threshold)
   - Minimum node size reached

**Chi-squared Test:**
$$\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

Where O = observed frequency, E = expected frequency.

**Advantages:**
- Multi-way splits = shallower trees
- Built-in significance testing
- Automatic category merging
- Good for market research/surveys

**Limitations:**
- Primarily for categorical targets
- Requires larger samples for significance
- Less common in ML libraries (more in SPSS/SAS)

---

## Question 5
- [ ] Done

**How is feature importance determined in the context of Decision Trees?**

**Answer:**

Feature importance in Decision Trees measures how much each feature contributes to reducing impurity (classification) or variance (regression) across all splits in the tree. Features that cause larger impurity reductions are more important.

**Calculation Method (Mean Decrease in Impurity - MDI):**

$$Importance(f) = \sum_{t \in T_f} \frac{n_t}{N} \cdot \Delta I(t)$$

Where:
- $T_f$ = set of all nodes that split on feature f
- $n_t$ = samples at node t
- $N$ = total samples
- $\Delta I(t)$ = impurity decrease at node t

**Step-by-Step:**

1. For each node where feature f is used for splitting:
   - Calculate impurity reduction = parent_impurity - weighted_child_impurities
   
2. Sum all impurity reductions for feature f

3. Normalize so all importances sum to 1

**Example:**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Train model
X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# Get feature importances
for name, imp in zip(load_iris().feature_names, clf.feature_importances_):
    print(f"{name}: {imp:.3f}")

# Output:
# sepal length (cm): 0.000
# sepal width (cm): 0.000  
# petal length (cm): 0.427
# petal width (cm): 0.573
```

**Key Properties:**
- Importances always sum to 1.0
- Features never used = 0 importance
- Higher = more important for predictions

**Caution - Bias Issues:**
- Biased toward high-cardinality features
- Correlated features split importance
- Use permutation importance for unbiased estimates

---

## Question 6
- [ ] Done

**Elaborate on how boosting techniques can be used with Decision Trees.**

**Answer:**

Boosting with Decision Trees trains a sequence of weak tree learners, where each tree corrects errors of previous trees. The final prediction combines all trees, typically via weighted voting (classification) or summation (regression).

**How Boosting + Trees Work:**

1. Train a shallow tree on data
2. Calculate errors/residuals
3. Train next tree to predict errors
4. Repeat, adding trees that focus on hard examples
5. Combine all trees for final prediction

**Major Boosting Algorithms with Trees:**

| Algorithm | How It Works | Tree Depth |
|-----------|--------------|------------|
| AdaBoost | Re-weight misclassified samples | 1 (stumps) |
| Gradient Boosting | Fit trees to negative gradient (residuals) | 3-8 |
| XGBoost | Regularized GB + 2nd order approximation | 6 (default) |
| LightGBM | Leaf-wise growth + histogram binning | No limit (31 leaves) |
| CatBoost | Ordered boosting + target statistics | 6 (default) |

**Why Shallow Trees for Boosting?**
- Each tree is a weak learner
- Deep trees → overfitting when combined
- Shallow trees (3-6 levels) work best
- Bias-variance tradeoff: many weak learners reduce bias together

**Example - Gradient Boosting:**

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

gb = GradientBoostingClassifier(
    n_estimators=100,    # Number of trees
    max_depth=3,         # Shallow trees
    learning_rate=0.1,   # Step size shrinkage
    random_state=42
)
gb.fit(X_train, y_train)
print(f"Accuracy: {gb.score(X_test, y_test):.3f}")
```

**Key Hyperparameters:**
- `n_estimators`: Number of boosting rounds
- `learning_rate`: Shrinks each tree's contribution
- `max_depth`: Tree complexity (3-6 typical)

**Boosting vs Bagging:**
- Boosting: Sequential, reduces bias
- Bagging (Random Forest): Parallel, reduces variance

---

## Question 7
- [ ] Done

**How do you determine the optimal number of splits for a Decision Tree?**

**Answer:**

The optimal number of splits (tree depth/complexity) is determined through validation techniques and hyperparameter tuning. Too few splits = underfitting; too many = overfitting.

**Methods to Determine Optimal Splits:**

**1. Cross-Validation with max_depth:**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

depths = range(1, 20)
cv_scores = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5)
    cv_scores.append(scores.mean())

optimal_depth = depths[cv_scores.index(max(cv_scores))]
print(f"Optimal depth: {optimal_depth}")
```

**2. Cost-Complexity Pruning (ccp_alpha):**

```python
from sklearn.tree import DecisionTreeClassifier

# Get cost-complexity path
clf = DecisionTreeClassifier(random_state=42)
path = clf.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas

# Find optimal alpha via cross-validation
best_alpha = None
best_score = 0
for alpha in alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    score = cross_val_score(clf, X_train, y_train, cv=5).mean()
    if score > best_score:
        best_score = score
        best_alpha = alpha
```

**3. Grid Search:**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid.fit(X, y)
print(grid.best_params_)
```

**Key Hyperparameters Controlling Splits:**
| Parameter | Effect |
|-----------|--------|
| max_depth | Maximum tree depth |
| min_samples_split | Min samples to split a node |
| min_samples_leaf | Min samples in leaf |
| max_leaf_nodes | Max number of leaves |

**Rule of Thumb:**
- Start with max_depth=5
- Increase if underfitting
- Use pruning if overfitting

---

## Question 8
- [ ] Done

**What metrics or methods do you use for validating a Decision Tree model?**

**Answer:**

Validation of Decision Trees involves using appropriate metrics for the task (classification/regression) and proper validation methods to estimate generalization performance.

**Classification Metrics:**

| Metric | Formula | Use Case |
|--------|---------|----------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Balanced classes |
| Precision | TP/(TP+FP) | Minimize false positives |
| Recall | TP/(TP+FN) | Minimize false negatives |
| F1-Score | 2×(P×R)/(P+R) | Imbalanced classes |
| AUC-ROC | Area under ROC curve | Ranking quality |
| Log Loss | -Σ[y·log(p)+(1-y)·log(1-p)] | Probability calibration |

**Regression Metrics:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MSE | Σ(y-ŷ)²/n | Average squared error |
| RMSE | √MSE | Same unit as target |
| MAE | Σ\|y-ŷ\|/n | Average absolute error |
| R² | 1 - SS_res/SS_tot | Variance explained |

**Validation Methods:**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, accuracy_score

# 1. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 2. K-Fold Cross-Validation
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

# 3. Stratified K-Fold (for imbalanced data)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Decision Tree-Specific Validation:**

1. **Learning Curves:** Check train vs validation score across training sizes
2. **Validation Curves:** Score vs hyperparameter values
3. **Tree Visualization:** Inspect splits for domain sense
4. **Feature Importance:** Verify important features match domain knowledge

**Signs of Overfitting:**
- Train accuracy >> Test accuracy
- Very deep tree
- Many leaves with few samples

---

## Question 9
- [ ] Done

**Compare and contrast the various Decision Tree algorithms (e.g., ID3, C4.5, CART, CHAID).**

**Answer:**

| Feature | ID3 | C4.5 | CART | CHAID |
|---------|-----|------|------|-------|
| **Year** | 1986 | 1993 | 1984 | 1980 |
| **Split Criterion** | Information Gain | Gain Ratio | Gini (class) / MSE (reg) | Chi-squared |
| **Split Type** | Multi-way | Multi-way | Binary | Multi-way |
| **Variable Types** | Categorical only | Both | Both | Categorical (original) |
| **Missing Values** | Cannot handle | Surrogate splits | Surrogate splits | Can handle |
| **Pruning** | None | Error-based | Cost-complexity | Pre-pruning |
| **Output Type** | Classification | Classification | Both | Classification |

**Detailed Comparison:**

**ID3 (Iterative Dichotomiser 3):**
- Uses Information Gain: $IG = H(parent) - \sum\frac{n_i}{n}H(child_i)$
- One branch per category (multi-way)
- Limitation: Biased toward high-cardinality features
- No handling for continuous or missing values

**C4.5 (Successor to ID3):**
- Uses Gain Ratio: $GR = IG / SplitInfo$
- Handles continuous variables (threshold selection)
- Handles missing values (probabilistic split)
- Error-based pruning with confidence intervals
- Creates multi-way splits for categorical features

**CART (Classification and Regression Trees):**
- Always binary splits
- Gini impurity (classification) or MSE (regression)
- Can do regression unlike ID3/C4.5
- Cost-complexity pruning with α parameter
- Most commonly used in sklearn

**CHAID (Chi-squared Automatic Interaction Detector):**
- Statistical approach using Chi-squared tests
- Merges categories that aren't significantly different
- Multi-way splits
- Built-in significance testing
- Popular in marketing/research

**When to Use:**
- **ID3**: Educational purposes only
- **C4.5**: When interpretability with multi-way splits needed
- **CART**: General ML tasks, regression, sklearn
- **CHAID**: Market research, categorical data analysis

---

## Question 10
- [ ] Done

**How do pruning strategies differ among various Decision Tree algorithms?**

**Answer:**

Pruning removes tree sections to prevent overfitting. Different algorithms use different pruning strategies based on when (pre/post) and how they prune.

**Pruning Strategy Comparison:**

| Algorithm | Pruning Type | Method | Key Parameter |
|-----------|--------------|--------|---------------|
| ID3 | None | No built-in pruning | N/A |
| C4.5 | Post-pruning | Error-based pruning (PEP) | Confidence factor |
| CART | Post-pruning | Cost-complexity (minimal cost-complexity) | α (ccp_alpha) |
| CHAID | Pre-pruning | Significance threshold | p-value cutoff |

**Detailed Pruning Methods:**

**1. ID3 - No Pruning:**
- Grows tree until pure leaves
- Requires external pruning methods

**2. C4.5 - Pessimistic Error Pruning (PEP):**
- Estimates error rate using upper confidence bound
- Replaces subtree with leaf if pruned error ≤ subtree error + SE
- Formula: $e = (E + 0.5)/N$ where E = errors, N = samples

**3. CART - Cost-Complexity Pruning:**
$$R_\alpha(T) = R(T) + \alpha \cdot |T|$$

- $R(T)$ = misclassification rate
- $|T|$ = number of terminal nodes
- $\alpha$ = complexity penalty
- Find optimal α via cross-validation

```python
# CART cost-complexity pruning in sklearn
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(ccp_alpha=0.01)  # Set complexity penalty
clf.fit(X_train, y_train)

# Find optimal alpha
path = clf.cost_complexity_pruning_path(X_train, y_train)
alphas, impurities = path.ccp_alphas, path.impurities
```

**4. CHAID - Pre-pruning via Significance:**
- Stops splitting when Chi-squared test not significant
- Merges categories with p-value > threshold (e.g., 0.05)
- Prevents overfitting before tree grows

**Pre-pruning vs Post-pruning:**
| Aspect | Pre-pruning | Post-pruning |
|--------|-------------|--------------|
| When | During growth | After full growth |
| Method | Stopping criteria | Remove subtrees |
| Risk | May stop too early | Computationally heavier |
| Examples | CHAID, min_samples_leaf | CART, C4.5 |

---

## Question 11
- [ ] Done

**What approach would you take to handle high-dimensional data when building Decision Trees?**

**Answer:**

High-dimensional data (many features) causes Decision Trees to overfit, increases computation, and makes the tree harder to interpret. Several strategies address these challenges.

**Approaches:**

**1. Feature Selection (Before Training):**

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.tree import DecisionTreeClassifier

# Method A: Univariate Selection
selector = SelectKBest(f_classif, k=20)  # Keep top 20 features
X_selected = selector.fit_transform(X, y)

# Method B: Recursive Feature Elimination
clf = DecisionTreeClassifier()
rfe = RFE(clf, n_features_to_select=20)
X_rfe = rfe.fit_transform(X, y)
```

**2. Dimensionality Reduction:**

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X)
```

**3. Random Feature Subsets (Ensemble Approach):**

```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest automatically handles high dimensions
# Each tree uses sqrt(n_features) at each split
rf = RandomForestClassifier(max_features='sqrt', n_estimators=100)
```

**4. Regularization via Pruning:**

```python
clf = DecisionTreeClassifier(
    max_depth=10,           # Limit depth
    min_samples_leaf=5,     # Require samples per leaf
    max_features='sqrt',    # Random feature subset per split
    ccp_alpha=0.01          # Cost-complexity pruning
)
```

**Strategy Summary:**

| Approach | When to Use | Advantage |
|----------|-------------|-----------|
| Feature Selection | Known irrelevant features | Removes noise |
| PCA | Correlated features | Reduces dimensions |
| Random Forest | General high-dim | Built-in feature sampling |
| Regularization | Moderate dimensions | Simple, effective |
| L1 Logistic first | Very high (1000s) | Filters to relevant features |

**Best Practice Pipeline:**
1. Remove constant/near-constant features
2. Apply variance threshold
3. Use tree-based feature importance or L1 selection
4. Train Decision Tree on reduced feature set
5. Or use Random Forest directly

**Warning:**
- Single Decision Trees are NOT recommended for very high-dimensional data
- Use Random Forest or Gradient Boosting instead
