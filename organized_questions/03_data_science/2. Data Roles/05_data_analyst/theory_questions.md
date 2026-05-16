# Data Analyst Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What is machine learning and how does it differ from traditional programming?**

**Answer:**

### Definition
Machine Learning is a subset of AI where algorithms learn patterns from data to make predictions or decisions without being explicitly programmed. Unlike traditional programming where rules are coded manually, ML systems derive rules automatically from data.

### Core Concepts
- **Traditional Programming**: Input + Rules → Output
- **Machine Learning**: Input + Output → Rules (learned model)
- **Training**: Process of learning patterns from historical data
- **Inference**: Using learned model to make predictions on new data

### Comparison

| Aspect | Traditional Programming | Machine Learning |
|--------|------------------------|------------------|
| Logic | Manually coded rules | Learned from data |
| Adaptability | Requires code changes | Learns from new data |
| Complexity | Difficult for complex patterns | Handles complex patterns well |
| Explainability | Clear and traceable | Often black-box |
| Example | If-else spam rules | Spam classifier trained on emails |

### Intuition
- **Traditional**: You teach computer exactly what to do step-by-step
- **ML**: You show computer many examples and it figures out the pattern
- **Example**: To identify cats - traditional requires coding every cat feature; ML learns from thousands of cat images

### Practical Relevance
- **Traditional**: Simple rule-based systems, calculators, deterministic algorithms
- **ML**: Image recognition, NLP, recommendation systems, fraud detection
- **Choose ML when**: Rules are complex, data is abundant, patterns change over time

---

## Question 2
- [ ] Done

**Explain the difference between supervised and unsupervised learning.**

**Answer:**

### Definition
Supervised learning trains models on labeled data (input-output pairs) to predict outcomes for new inputs. Unsupervised learning finds hidden patterns in unlabeled data without predefined outputs.

### Core Concepts

| Aspect | Supervised Learning | Unsupervised Learning |
|--------|--------------------|-----------------------|
| Data | Labeled (X, y) | Unlabeled (X only) |
| Goal | Predict output | Discover structure |
| Feedback | Direct (correct answer known) | No feedback |
| Tasks | Classification, Regression | Clustering, Dimensionality Reduction |
| Evaluation | Clear metrics (accuracy, MSE) | Indirect metrics (silhouette, inertia) |

### Examples

**Supervised:**
- Email → Spam/Not Spam (Classification)
- House Features → Price (Regression)
- Image → Cat/Dog label (Classification)

**Unsupervised:**
- Customer data → Customer segments (Clustering)
- High-dim data → 2D visualization (Dimensionality Reduction)
- Transaction data → Anomaly detection

### Intuition
- **Supervised**: Teacher provides correct answers during learning
- **Unsupervised**: Student discovers patterns without guidance
- **Semi-supervised**: Small labeled + large unlabeled data

### Practical Relevance
- **Supervised**: Most business prediction problems
- **Unsupervised**: Exploratory analysis, feature engineering, preprocessing
- **Use unsupervised when**: Labels are expensive/unavailable, exploring data structure

---

## Question 3
- [ ] Done

**What is the role of feature selection in machine learning?**

**Answer:**

### Definition
Feature selection is the process of identifying and selecting the most relevant features (variables) from the dataset that contribute most to the predictive power. It removes irrelevant, redundant, or noisy features to improve model performance and interpretability.

### Core Concepts
- **Relevance**: Feature's ability to predict target
- **Redundancy**: Features carrying duplicate information
- **Dimensionality Reduction**: Fewer features, simpler model
- **Curse of Dimensionality**: Performance degrades with too many features

### Feature Selection Methods

| Method Type | Technique | Description |
|-------------|-----------|-------------|
| Filter | Correlation, Chi-square, Mutual Information | Rank features independently |
| Wrapper | Forward/Backward Selection, RFE | Use model to evaluate subsets |
| Embedded | Lasso (L1), Tree feature importance | Selection during model training |

### Benefits
- **Reduces Overfitting**: Fewer features = less noise
- **Improves Accuracy**: Removes irrelevant features
- **Reduces Training Time**: Less computation
- **Better Interpretability**: Easier to explain

### Python Code Example
```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

# Method 1: Filter - Select K best features
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X, y)

# Method 2: Wrapper - Recursive Feature Elimination
model = RandomForestClassifier()
rfe = RFE(model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)

# Method 3: Embedded - Feature importance from tree
rf = RandomForestClassifier().fit(X, y)
importances = rf.feature_importances_  # Select top features
```

### Interview Tips
- Filter methods are fast but ignore feature interactions
- Wrapper methods are accurate but computationally expensive
- Embedded methods (L1, trees) are often the best balance
- Always validate selected features on test set

---

## Question 4
- [ ] Done

**Describe the concept of overfitting and underfitting in machine learning models.**

**Answer:**

### Definition
**Overfitting**: Model learns training data too well including noise, resulting in poor generalization to new data (low bias, high variance). **Underfitting**: Model is too simple to capture underlying patterns (high bias, low variance).

### Core Concepts

| Aspect | Underfitting | Good Fit | Overfitting |
|--------|--------------|----------|-------------|
| Training Error | High | Low | Very Low |
| Test Error | High | Low | High |
| Model Complexity | Too simple | Optimal | Too complex |
| Bias | High | Balanced | Low |
| Variance | Low | Balanced | High |

### Visual Intuition
- **Underfitting**: Straight line through curved data
- **Good Fit**: Curve that follows data trend
- **Overfitting**: Wiggly curve passing through every point

### Causes and Solutions

**Underfitting Causes:**
- Model too simple
- Too few features
- Over-regularization

**Solutions:**
- Increase model complexity
- Add more features
- Reduce regularization

**Overfitting Causes:**
- Model too complex
- Too many features
- Insufficient training data

**Solutions:**
- Simplify model
- Add regularization (L1/L2)
- Cross-validation
- Dropout (for neural networks)
- Early stopping
- Get more training data

### Python Code Example
```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

# Underfitting: degree=1 for complex data
# Good fit: degree=2-3
# Overfitting: degree=15 for small dataset

for degree in [1, 3, 15]:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    scores = cross_val_score(model, X_poly, y, cv=5)
    print(f"Degree {degree}: Train-Test gap indicates fit quality")
```

### Interview Tips
- Compare training vs validation error to diagnose
- Large gap = overfitting; both high = underfitting
- Cross-validation is key to detecting overfitting
- Regularization strength controls the trade-off

---

## Question 5
- [ ] Done

**What is cross-validation and why is it important?**

**Answer:**

### Definition
Cross-validation is a resampling technique that partitions data into multiple train-test splits to evaluate model performance reliably. It provides a more robust estimate of model generalization by averaging results across multiple folds.

### Core Concepts
- **Fold**: A subset of data used for validation
- **K-Fold CV**: Data split into K equal parts, each serves as test once
- **Holdout**: Simple single train-test split (not CV)
- **Stratified**: Maintains class distribution in each fold

### K-Fold Cross-Validation Process
1. Split data into K equal folds
2. For i = 1 to K:
   - Train on K-1 folds
   - Validate on fold i
   - Record performance score
3. Average all K scores as final estimate

### Types of Cross-Validation

| Type | Use Case |
|------|----------|
| K-Fold (K=5,10) | General purpose |
| Stratified K-Fold | Classification with imbalanced data |
| Leave-One-Out (LOO) | Small datasets |
| Time Series Split | Sequential/temporal data |
| Group K-Fold | When samples belong to groups |

### Why Important?
- **Reliable estimate**: Reduces variance in performance estimate
- **Uses all data**: Every sample used for both training and testing
- **Detects overfitting**: More robust than single split
- **Hyperparameter tuning**: Safe model selection

### Python Code Example
```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

# Basic K-Fold (K=5)
scores = cross_val_score(model, X, y, cv=5)
print(f"Mean CV Score: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# Stratified K-Fold (for classification)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)
```

### Interview Tips
- Never use test set for any decision during CV
- K=5 or K=10 are common choices (trade-off: bias vs variance)
- Stratified CV is preferred for classification
- For time series: use TimeSeriesSplit (no shuffle!)

---

## Question 6
- [ ] Done

**Explain the bias-variance tradeoff in machine learning.**

**Answer:**

### Definition
The bias-variance tradeoff describes the fundamental tension between model simplicity (bias) and sensitivity to training data (variance). Total error = Bias² + Variance + Irreducible Error. Optimal models balance both components.

### Core Concepts
- **Bias**: Error from simplifying assumptions; high bias = underfitting
- **Variance**: Error from sensitivity to training data fluctuations; high variance = overfitting
- **Irreducible Error**: Noise inherent in the data (cannot be reduced)

### Mathematical Formulation
$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

Where:
- $\text{Bias} = E[\hat{f}(x)] - f(x)$ (systematic error)
- $\text{Variance} = E[(\hat{f}(x) - E[\hat{f}(x)])^2]$ (sensitivity to training data)

### Intuition

| Model Type | Bias | Variance | Example |
|------------|------|----------|---------|
| Simple (Linear) | High | Low | Linear regression on curved data |
| Complex (Flexible) | Low | High | High-degree polynomial |
| Optimal | Balanced | Balanced | Regularized models |

### Visual Interpretation
- **High Bias**: Model predictions clustered but away from target
- **High Variance**: Model predictions scattered around target
- **Optimal**: Predictions both accurate and consistent

### Practical Examples
- **High Bias Models**: Linear Regression, Naive Bayes
- **High Variance Models**: Deep Decision Trees, KNN (k=1)
- **Balanced**: Random Forest, Regularized Neural Networks

### How to Manage
- **Reduce Bias**: More complex model, more features
- **Reduce Variance**: Regularization, more data, ensemble methods, cross-validation

### Interview Tips
- As model complexity increases: bias decreases, variance increases
- Regularization controls the trade-off
- Ensemble methods (bagging reduces variance, boosting reduces bias)
- Cross-validation helps find the sweet spot

---

## Question 7
- [ ] Done

**What is regularization and how does it help prevent overfitting?**

**Answer:**

### Definition
Regularization is a technique that adds a penalty term to the loss function to constrain model complexity. It discourages large coefficient values, forcing the model to be simpler and generalize better to unseen data.

### Core Concepts
- **Penalty Term**: Added to loss function to shrink coefficients
- **L1 (Lasso)**: Penalty = sum of absolute values → sparse solutions
- **L2 (Ridge)**: Penalty = sum of squared values → small but non-zero coefficients
- **Lambda (λ)**: Regularization strength (hyperparameter)

### Mathematical Formulation

**Original Loss**: $L = \sum(y_i - \hat{y}_i)^2$

**L1 Regularization (Lasso)**:
$$L_{L1} = \sum(y_i - \hat{y}_i)^2 + \lambda \sum|w_j|$$

**L2 Regularization (Ridge)**:
$$L_{L2} = \sum(y_i - \hat{y}_i)^2 + \lambda \sum w_j^2$$

**Elastic Net** (L1 + L2):
$$L_{EN} = \sum(y_i - \hat{y}_i)^2 + \lambda_1 \sum|w_j| + \lambda_2 \sum w_j^2$$

### Comparison

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|------------|------------|
| Penalty | $\|w\|$ | $w^2$ |
| Feature Selection | Yes (zeros out coefficients) | No (shrinks but keeps all) |
| Solution | Sparse | Dense |
| Multicollinearity | Picks one feature | Distributes weight |
| Use Case | High-dim, feature selection | Multicollinearity |

### Python Code Example
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge (L2): Shrinks coefficients
ridge = Ridge(alpha=1.0)  # alpha = lambda
ridge.fit(X_train, y_train)

# Lasso (L1): Zeros out irrelevant features
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"Non-zero coefficients: {sum(lasso.coef_ != 0)}")

# Elastic Net: Combines L1 + L2
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio: L1 vs L2 mix
elastic.fit(X_train, y_train)
```

### Interview Tips
- Higher λ → more regularization → simpler model → higher bias
- Use cross-validation to find optimal λ
- Lasso for feature selection, Ridge for multicollinearity
- Always standardize features before regularization

---

## Question 8
- [ ] Done

**Describe the difference between parametric and non-parametric models.**

**Answer:**

### Definition
**Parametric models** have a fixed number of parameters regardless of data size (e.g., linear regression). **Non-parametric models** have parameters that grow with data size, making no strong assumptions about data distribution (e.g., KNN, decision trees).

### Core Concepts

| Aspect | Parametric | Non-parametric |
|--------|------------|----------------|
| Parameters | Fixed, finite | Grows with data |
| Assumptions | Strong (functional form) | Weak/flexible |
| Complexity | Limited | Adapts to data |
| Training Speed | Fast | Can be slower |
| Prediction Speed | Fast | Can be slower |
| Data Required | Less | More |
| Overfitting Risk | Lower | Higher |

### Examples

**Parametric Models:**
- Linear Regression (params: weights for each feature)
- Logistic Regression
- Naive Bayes
- Linear SVM

**Non-parametric Models:**
- K-Nearest Neighbors (stores all training data)
- Decision Trees (depth grows with data)
- Random Forest
- Kernel SVM

### Intuition
- **Parametric**: "I assume the relationship is linear, let me find the best line"
- **Non-parametric**: "I make no assumption, let me fit whatever shape the data suggests"

### Trade-offs
**Parametric:**
- Pros: Fast, interpretable, requires less data
- Cons: May underfit if assumption is wrong

**Non-parametric:**
- Pros: Flexible, captures complex patterns
- Cons: Computationally expensive, prone to overfitting

### Practical Relevance
- Start with parametric models (baseline)
- Move to non-parametric if underfitting
- Non-parametric preferred when: complex relationships, enough data, less interpretability needed

### Interview Tips
- "Parametric" doesn't mean "no hyperparameters"
- Non-parametric still has hyperparameters (k in KNN, depth in trees)
- Neural networks are technically parametric but behave flexibly

---

## Question 9
- [ ] Done

**What is the curse of dimensionality and how does it impact machine learning?**

**Answer:**

### Definition
The curse of dimensionality refers to problems that arise when working with high-dimensional data. As dimensions increase, data becomes sparse, distances become less meaningful, and the amount of data needed to maintain statistical significance grows exponentially.

### Core Concepts
- **Sparsity**: Data points spread thin in high dimensions
- **Distance Meaningless**: All points become equidistant
- **Exponential Data Need**: Volume grows exponentially with dimensions
- **Overfitting Risk**: More features than samples leads to overfitting

### Mathematical Insight
- Volume of unit hypersphere approaches 0 as dimensions → ∞
- Ratio of distance from closest to farthest point → 1 in high dimensions
- To maintain same density: data needed ~ $O(n^d)$ where d = dimensions

### Impact on ML

| Problem | Effect |
|---------|--------|
| Distance-based methods (KNN, K-means) | Distances become similar, poor performance |
| Feature space | Data becomes sparse, hard to find patterns |
| Overfitting | Models memorize instead of generalize |
| Computation | Time and memory grow exponentially |
| Visualization | Cannot visualize high-dimensional data directly |

### Solutions
- **Feature Selection**: Remove irrelevant features
- **Dimensionality Reduction**: PCA, t-SNE, UMAP
- **Regularization**: L1 (Lasso) for automatic feature selection
- **Feature Engineering**: Create meaningful combinations
- **More Data**: If possible, collect more samples
- **Use robust algorithms**: Tree-based methods handle high dimensions better

### Python Example
```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

# Original: 1000 features, 500 samples (curse!)
# Solution 1: PCA to reduce dimensions
pca = PCA(n_components=50)  # Reduce to 50 components
X_reduced = pca.fit_transform(X)

# Solution 2: Feature selection
selector = SelectKBest(k=50)
X_selected = selector.fit_transform(X, y)
```

### Interview Tips
- Rule of thumb: Need ~10 samples per feature minimum
- KNN severely affected; tree-based models more robust
- Always consider dimensionality when choosing algorithms

---

## Question 10
- [ ] Done

**Explain the concept of model complexity and its relationship with performance.**

**Answer:**

### Definition
Model complexity refers to the flexibility and capacity of a model to learn patterns from data. It's determined by factors like number of parameters, polynomial degree, or tree depth. There's a sweet spot where complexity balances training and generalization performance.

### Core Concepts
- **Capacity**: Model's ability to fit diverse functions
- **Training Error**: Decreases with complexity
- **Validation Error**: U-shaped curve (decreases, then increases)
- **Optimal Complexity**: Point where validation error is minimized

### Relationship with Performance

```
Error
  |
  |  \                    Validation Error
  |   \      ___________/
  |    \    /
  |     \  /
  |      \/  <-- Optimal Complexity
  |       \
  |        \________ Training Error
  |__________________________ Model Complexity
     Simple           Complex
```

### Factors Affecting Complexity

| Model | Complexity Factor |
|-------|-------------------|
| Linear Regression | Number of features |
| Polynomial Regression | Degree of polynomial |
| Decision Tree | Depth, number of leaves |
| Neural Network | Layers, neurons, parameters |
| KNN | Inverse of K (K=1 most complex) |

### Trade-offs
- **Too Simple**: Underfitting, high bias, misses patterns
- **Too Complex**: Overfitting, high variance, memorizes noise
- **Optimal**: Good generalization, balanced bias-variance

### Controlling Complexity
- **Regularization**: Penalizes complexity (L1, L2)
- **Pruning**: Simplify decision trees
- **Early Stopping**: Stop training before overfitting
- **Cross-Validation**: Find optimal hyperparameters
- **Dropout**: Neural network regularization

### Interview Tips
- More data allows for more complex models
- Use validation curves to visualize complexity impact
- Occam's Razor: prefer simpler models when performance is similar
- AIC/BIC criteria penalize complexity while rewarding fit

---

## Question 11
- [ ] Done

**What is data preprocessing and why is it important in machine learning?**

**Answer:**

### Definition
Data preprocessing transforms raw data into a clean, suitable format for ML algorithms. It includes handling missing values, encoding categorical variables, scaling features, and removing outliers. Quality preprocessing directly impacts model performance.

### Core Concepts
- **Data Quality**: Garbage in = Garbage out
- **Consistency**: Uniform data format and types
- **Completeness**: Handle missing values
- **Relevance**: Remove noise and irrelevant data

### Preprocessing Steps

| Step | Purpose | Techniques |
|------|---------|------------|
| Missing Values | Handle incomplete data | Imputation, deletion |
| Encoding | Convert categorical to numeric | One-hot, Label encoding |
| Scaling | Normalize feature ranges | StandardScaler, MinMaxScaler |
| Outliers | Handle extreme values | IQR, Z-score, capping |
| Feature Engineering | Create meaningful features | Polynomial, interaction terms |
| Imbalanced Data | Balance class distribution | SMOTE, undersampling |

### Why Important?
- **Algorithm Requirements**: Many algorithms need numeric, scaled data
- **Convergence**: Gradient-based methods converge faster with scaling
- **Performance**: Clean data → better model accuracy
- **Interpretability**: Consistent features are easier to interpret
- **Avoid Bias**: Missing data handled properly avoids biased predictions

### Python Code Example
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Preprocessing pipeline
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing
    ('scaler', StandardScaler())                     # Scale features
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])
```

### Interview Tips
- Always split data BEFORE preprocessing (fit on train, transform both)
- Use pipelines to prevent data leakage
- Document all preprocessing decisions for reproducibility
- Different algorithms have different preprocessing requirements

---

## Question 12
- [ ] Done

**Explain the techniques used for handling missing data.**

**Answer:**

### Definition
Missing data handling involves strategies to deal with incomplete data points, either by removing them, imputing (filling) values, or using algorithms that naturally handle missing values. The choice depends on the amount and pattern of missingness.

### Core Concepts
- **MCAR** (Missing Completely at Random): Missingness unrelated to any data
- **MAR** (Missing at Random): Missingness related to observed data
- **MNAR** (Missing Not at Random): Missingness related to missing value itself

### Techniques Summary

| Technique | When to Use | Pros | Cons |
|-----------|-------------|------|------|
| Deletion (Listwise) | MCAR, small % missing | Simple | Loses data |
| Mean/Median Imputation | Numeric, MCAR | Simple, preserves size | Distorts distribution |
| Mode Imputation | Categorical | Simple | May create bias |
| KNN Imputation | Complex relationships | Considers neighbors | Computationally expensive |
| Iterative/MICE | MAR, complex data | Preserves relationships | Complex |
| Missing Indicator | When missingness is informative | Captures missingness pattern | Adds features |

### Decision Framework
```
Missing Data
    │
    ├── < 5% missing → Simple imputation or deletion
    │
    ├── 5-20% missing → KNN or iterative imputation
    │
    └── > 20% missing → Consider dropping feature or 
                        model-based imputation
```

### Python Code Example
```python
from sklearn.impute import SimpleImputer, KNNImputer
import pandas as pd

# Check missing values
print(df.isnull().sum())

# Method 1: Simple imputation (mean/median/mode)
imputer_mean = SimpleImputer(strategy='mean')
imputer_median = SimpleImputer(strategy='median')
df['col'] = imputer_mean.fit_transform(df[['col']])

# Method 2: KNN Imputation (uses neighbors)
knn_imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(knn_imputer.fit_transform(df))

# Method 3: Forward/Backward fill (time series)
df['col'].fillna(method='ffill', inplace=True)

# Method 4: Add missing indicator
df['col_missing'] = df['col'].isnull().astype(int)
```

### Interview Tips
- Always analyze missingness pattern before choosing technique
- Fit imputer on training data only, transform both train and test
- For tree-based models, missingness can be handled natively
- Consider multiple imputation for statistical rigor

---

## Question 13
- [ ] Done

**What is feature scaling and why is it necessary?**

**Answer:**

### Definition
Feature scaling transforms features to a similar range or distribution. It's necessary because many ML algorithms are sensitive to feature magnitudes, and features with larger scales can dominate the learning process unfairly.

### Core Concepts
- **Scale Sensitivity**: Distance-based and gradient-based algorithms affected
- **Standardization**: Transform to zero mean, unit variance
- **Normalization**: Scale to fixed range (usually 0-1)
- **Convergence**: Scaled data leads to faster optimization

### When Necessary

| Algorithm | Needs Scaling? | Reason |
|-----------|---------------|--------|
| Linear Regression | Yes (for regularized) | Regularization penalizes large coefficients |
| Logistic Regression | Yes | Gradient descent convergence |
| SVM | Yes | Distance-based, kernel computation |
| KNN | Yes | Distance calculation |
| Neural Networks | Yes | Gradient descent, activation saturation |
| Decision Trees | No | Splits based on thresholds, scale-invariant |
| Random Forest | No | Ensemble of trees |
| Naive Bayes | No | Probability-based |

### Python Code Example
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler: mean=0, std=1
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler

# MinMaxScaler: range [0, 1]
minmax = MinMaxScaler()
X_normalized = minmax.fit_transform(X_train)
```

### Interview Tips
- Always fit scaler on training data only
- Transform both train and test with same scaler
- Tree-based models don't require scaling but it doesn't hurt
- For sparse data, use MaxAbsScaler to preserve sparsity

---

## Question 14
- [ ] Done

**Describe the difference between normalization and standardization.**

**Answer:**

### Definition
**Normalization (Min-Max Scaling)**: Scales data to a fixed range [0,1]. **Standardization (Z-score)**: Transforms data to have zero mean and unit variance. Both are scaling techniques but produce different distributions.

### Mathematical Formulation

**Normalization**:
$$x_{normalized} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**Standardization**:
$$x_{standardized} = \frac{x - \mu}{\sigma}$$

### Comparison

| Aspect | Normalization | Standardization |
|--------|---------------|-----------------|
| Range | [0, 1] | Unbounded |
| Mean | Depends on data | 0 |
| Std Dev | Depends on data | 1 |
| Outlier Sensitivity | High (affects min/max) | Lower |
| Distribution | Preserves shape | Doesn't assume normal |
| Use Case | Image pixels, neural nets | Most ML algorithms |

### When to Use

**Normalization:**
- Neural networks (especially with sigmoid/tanh)
- Image processing (pixel values 0-255 → 0-1)
- When you need bounded values
- K-Nearest Neighbors (sometimes)

**Standardization:**
- When data has outliers (more robust)
- Linear models with regularization
- SVM, Logistic Regression
- When assuming normally distributed features

### Python Code Example
```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Normalization: [0, 1] range
normalizer = MinMaxScaler()
X_norm = normalizer.fit_transform(X)

# Standardization: mean=0, std=1
standardizer = StandardScaler()
X_std = standardizer.fit_transform(X)

# Verify
print(f"Normalized range: [{X_norm.min():.2f}, {X_norm.max():.2f}]")
print(f"Standardized mean: {X_std.mean():.2f}, std: {X_std.std():.2f}")
```

### Interview Tips
- Normalization is sensitive to outliers (new data outside [min,max])
- Standardization doesn't bound values, which can be an issue for some activations
- Always apply the same transformation to test data
- When in doubt, standardization is generally safer

---

## Question 15
- [ ] Done

**What is one-hot encoding and when is it used?**

**Answer:**

### Definition
One-hot encoding converts categorical variables into binary vectors where each category becomes a separate column with values 0 or 1. Only one column has value 1 (hot) at a time, indicating the category.

### Core Concepts
- **Binary Representation**: Each category → separate binary column
- **Mutual Exclusivity**: Only one column is 1 per row
- **No Ordinal Assumption**: Treats all categories equally
- **Dimensionality Increase**: n categories → n (or n-1) columns

### Example
```
Original:      One-Hot Encoded:
Color          Color_Red  Color_Blue  Color_Green
Red       →    1          0           0
Blue      →    0          1           0
Green     →    0          0           1
Red       →    1          0           0
```

### When to Use
- **Linear Models**: Logistic Regression, Linear Regression, SVM
- **Neural Networks**: As input layer
- **Nominal Variables**: No inherent order (colors, cities, categories)

### When NOT to Use
- **High Cardinality**: Thousands of categories → too many columns
- **Tree-based Models**: Label encoding often works better
- **Ordinal Variables**: Use label encoding to preserve order

### Python Code Example
```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({'color': ['red', 'blue', 'green', 'red']})

# Method 1: pandas get_dummies (simple)
df_encoded = pd.get_dummies(df['color'], prefix='color')

# Method 2: sklearn OneHotEncoder (for pipelines)
encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' avoids multicollinearity
encoded = encoder.fit_transform(df[['color']])

# With drop='first': n-1 columns (avoids dummy variable trap)
```

### Dummy Variable Trap
- With n categories, only n-1 columns needed
- Full n columns create multicollinearity (one is redundant)
- Use `drop='first'` for linear models

### Interview Tips
- Always use `drop='first'` for linear models to avoid multicollinearity
- For unseen categories in test data, use `handle_unknown='ignore'`
- Consider target encoding for high cardinality features
- Tree models don't need one-hot encoding

---

## Question 16
- [ ] Done

**Explain the concept of feature engineering and its importance.**

**Answer:**

### Definition
Feature engineering is the process of using domain knowledge to create, transform, or select features that improve model performance. It transforms raw data into features that better represent the underlying problem, often having more impact than algorithm choice.

### Core Concepts
- **Domain Knowledge**: Understanding data context to create meaningful features
- **Feature Creation**: Derive new features from existing ones
- **Feature Transformation**: Apply mathematical operations (log, square, etc.)
- **Interaction Features**: Combine multiple features

### Feature Engineering Techniques

| Technique | Example | Use Case |
|-----------|---------|----------|
| Mathematical | log(x), x², √x | Skewed distributions |
| Date/Time | day_of_week, is_weekend, hour | Time-based patterns |
| Aggregation | mean, sum, count by group | Customer transactions |
| Binning | age → age_group | Reduce noise, create categories |
| Interaction | price × quantity | Capture relationships |
| Text | word_count, sentiment | NLP features |
| Domain-specific | BMI from height/weight | Expert knowledge |

### Why Important?
- **Better than algorithm tuning**: Good features matter more than complex models
- **Captures domain knowledge**: ML can't discover what it doesn't see
- **Reduces complexity**: Simple model with good features > complex model with raw features
- **Improves interpretability**: Meaningful features are easier to explain

### Python Code Example
```python
import pandas as pd
import numpy as np

# Original data
df = pd.DataFrame({
    'price': [100, 200, 150],
    'quantity': [2, 1, 3],
    'date': pd.to_datetime(['2024-01-15', '2024-01-16', '2024-01-20'])
})

# Feature Engineering
df['total_value'] = df['price'] * df['quantity']           # Interaction
df['log_price'] = np.log1p(df['price'])                    # Transformation
df['day_of_week'] = df['date'].dt.dayofweek                # Date extraction
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # Binary feature

# Binning
df['price_category'] = pd.cut(df['price'], bins=[0, 100, 200, np.inf], 
                               labels=['low', 'medium', 'high'])
```

### Interview Tips
- Feature engineering is often the difference between good and great models
- Start simple, add complexity as needed
- Create features before doing feature selection
- Document feature engineering decisions for reproducibility

---

## Question 17
- [ ] Done

**What are some common techniques for feature extraction?**

**Answer:**

### Definition
Feature extraction transforms raw data into a reduced set of informative features, unlike feature selection which chooses from existing features. It creates new features that capture essential information while reducing dimensionality.

### Core Concepts
- **Transformation**: Creates new features from original data
- **Dimensionality Reduction**: Fewer features, preserved information
- **Representation Learning**: Learn useful features automatically
- **Domain-specific**: Different techniques for different data types

### Techniques by Data Type

| Data Type | Technique | Output |
|-----------|-----------|--------|
| Tabular | PCA, ICA | Principal components |
| Text | TF-IDF, Word2Vec, BERT | Numeric vectors |
| Images | CNN features, HOG, SIFT | Feature maps |
| Time Series | FFT, Wavelets, Statistical features | Frequency/statistical features |
| Audio | MFCC, Spectrograms | Frequency representations |

### Common Techniques Explained

**PCA (Principal Component Analysis)**
- Projects data onto orthogonal components of maximum variance
- Linear, unsupervised

**t-SNE / UMAP**
- Non-linear dimensionality reduction
- Good for visualization

**Autoencoders**
- Neural network learns compressed representation
- Non-linear, can capture complex patterns

**TF-IDF (Text)**
- Term Frequency × Inverse Document Frequency
- Weights words by importance

### Python Code Example
```python
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# Tabular: PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)  # Extract 10 principal components

# Text: TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
X_text = tfidf.fit_transform(documents)  # Extract 1000 text features

# Images: Using pre-trained CNN
# from tensorflow.keras.applications import VGG16
# model = VGG16(include_top=False)
# features = model.predict(images)
```

### Interview Tips
- Feature extraction creates new features; feature selection chooses existing ones
- PCA for tabular, CNN for images, TF-IDF/embeddings for text
- Consider whether extracted features are interpretable

---

## Question 18
- [ ] Done

**What is the purpose of dimensionality reduction techniques like PCA (Principal Component Analysis)?**

**Answer:**

### Definition
Dimensionality reduction reduces the number of features while preserving maximum information. PCA projects data onto orthogonal axes (principal components) ordered by variance explained, allowing retention of most information with fewer dimensions.

### Core Concepts
- **Variance**: PCA maximizes variance captured in each component
- **Principal Components**: New orthogonal axes, uncorrelated
- **Eigenvalues**: Amount of variance explained by each component
- **Eigenvectors**: Direction of each principal component

### PCA Algorithm Steps
1. Standardize the data (mean=0, std=1)
2. Compute covariance matrix
3. Calculate eigenvalues and eigenvectors
4. Sort eigenvectors by eigenvalues (descending)
5. Select top k eigenvectors
6. Transform data onto new k-dimensional space

### Mathematical Formulation
- Covariance Matrix: $C = \frac{1}{n-1}X^TX$
- Eigenvalue Decomposition: $Cv = \lambda v$
- Projection: $X_{reduced} = X \cdot W_k$ (where $W_k$ = top k eigenvectors)

### Purpose and Benefits
- **Reduce Overfitting**: Fewer features, less noise
- **Speed Up Training**: Less computation
- **Visualization**: Reduce to 2-3 dimensions for plotting
- **Remove Multicollinearity**: Components are uncorrelated
- **Noise Reduction**: Small components often capture noise

### Python Code Example
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Fit PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_reduced = pca.fit_transform(X)

print(f"Components kept: {pca.n_components_}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# Scree plot: visualize variance explained
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), 
         pca.explained_variance_ratio_.cumsum())
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
```

### Interview Tips
- Always standardize before PCA (features should be on same scale)
- PCA is unsupervised (doesn't use target)
- For classification, LDA might be better (uses class labels)
- Components are hard to interpret (linear combinations)

---

## Question 19
- [ ] Done

**Explain the concept of feature importance and how it can be determined.**

**Answer:**

### Definition
Feature importance measures how much each feature contributes to the model's predictions. It helps identify which variables are most influential, enabling feature selection, model interpretation, and domain insight.

### Core Concepts
- **Importance Score**: Numerical value indicating feature's contribution
- **Global Importance**: Overall importance across all predictions
- **Local Importance**: Importance for specific predictions (SHAP, LIME)
- **Model-agnostic vs Model-specific**: Some methods work with any model

### Methods to Determine Feature Importance

| Method | Type | How It Works |
|--------|------|--------------|
| Tree-based (Gini/Entropy) | Model-specific | Measures impurity reduction |
| Permutation Importance | Model-agnostic | Shuffles feature, measures accuracy drop |
| Coefficient Magnitude | Linear models | Absolute value of coefficients |
| SHAP Values | Model-agnostic | Game theory based contribution |
| Correlation | Statistical | Correlation with target |

### Tree-Based Importance (Random Forest, XGBoost)
- Measures total decrease in impurity from splits on that feature
- Weighted by number of samples reaching the node
- Fast to compute, built into the model

### Permutation Importance
- Shuffle one feature's values
- Measure decrease in model performance
- More reliable than built-in importance for correlated features

### Python Code Example
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import pandas as pd

# Train model
rf = RandomForestClassifier().fit(X_train, y_train)

# Method 1: Built-in feature importance (trees)
importance = rf.feature_importances_
feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
feature_imp = feature_imp.sort_values('importance', ascending=False)

# Method 2: Permutation importance (any model)
perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10)
print(f"Permutation importance: {perm_imp.importances_mean}")

# Method 3: For linear models - coefficient magnitude
# importance = np.abs(model.coef_)
```

### Interview Tips
- Built-in tree importance can be biased toward high-cardinality features
- Permutation importance is more reliable but slower
- SHAP provides both global and local explanations
- Always compute importance on test data, not training data

---

## Question 20
- [ ] Done

**What is linear regression and how does it work?**

**Answer:**

### Definition
Linear regression models the relationship between a dependent variable (y) and one or more independent variables (X) by fitting a linear equation. It finds the best-fit line that minimizes the sum of squared differences between predicted and actual values.

### Core Concepts
- **Hypothesis**: $\hat{y} = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n$
- **Coefficients (weights)**: Parameters learned from data
- **Intercept**: Value of y when all features are zero
- **Residual**: Difference between actual and predicted $(y_i - \hat{y}_i)$

### Mathematical Formulation

**Model**: $\hat{y} = X \cdot w + b$

**Loss Function (MSE)**:
$$J(w) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Closed-form Solution (Normal Equation)**:
$$w = (X^TX)^{-1}X^Ty$$

### How It Works
1. Initialize weights (or use closed-form solution)
2. Calculate predictions: $\hat{y} = Xw$
3. Compute error (MSE) between predicted and actual
4. Update weights to minimize error (gradient descent or closed-form)
5. Repeat until convergence

### Python Code Example
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}")
```

### Intuition
- Finds the line that minimizes the vertical distance from all points
- Coefficients represent the change in y for one unit change in x
- Positive coefficient = positive relationship

### Interview Tips
- Linear regression assumes linear relationship (check residual plots)
- Sensitive to outliers (consider robust regression)
- Check assumptions: linearity, normality of residuals, homoscedasticity

---

## Question 21
- [ ] Done

**Explain the difference between simple linear regression and multiple linear regression.**

**Answer:**

### Definition
**Simple Linear Regression**: Models relationship between one independent variable (X) and one dependent variable (y). **Multiple Linear Regression**: Models relationship between multiple independent variables (X₁, X₂, ..., Xₙ) and one dependent variable (y).

### Mathematical Formulation

**Simple Linear Regression**:
$$y = w_0 + w_1x$$

**Multiple Linear Regression**:
$$y = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n$$

### Comparison

| Aspect | Simple Linear | Multiple Linear |
|--------|---------------|-----------------|
| Features | 1 | 2 or more |
| Equation | Line (2D) | Hyperplane (n+1 D) |
| Complexity | Simpler | More complex |
| Use Case | Single predictor | Multiple predictors |
| Overfitting Risk | Lower | Higher |
| Multicollinearity | Not applicable | Must check |

### Intuition
- **Simple**: Finding best line through 2D scatter plot
- **Multiple**: Finding best plane/hyperplane through multi-dimensional space

### Example
```
Simple: House Price = β₀ + β₁ × Area

Multiple: House Price = β₀ + β₁ × Area + β₂ × Bedrooms + β₃ × Age
```

### Python Code Example
```python
from sklearn.linear_model import LinearRegression

# Simple Linear Regression (1 feature)
X_simple = df[['area']]  # Shape: (n, 1)
model_simple = LinearRegression()
model_simple.fit(X_simple, y)

# Multiple Linear Regression (multiple features)
X_multiple = df[['area', 'bedrooms', 'age']]  # Shape: (n, 3)
model_multiple = LinearRegression()
model_multiple.fit(X_multiple, y)

print(f"Simple: y = {model_simple.intercept_:.2f} + {model_simple.coef_[0]:.2f}x")
print(f"Multiple coefficients: {model_multiple.coef_}")
```

### Interview Tips
- Multiple regression must check for multicollinearity (VIF)
- Adding features doesn't always improve model (may overfit)
- Use adjusted R² to compare models with different number of features
- Feature scaling is important for interpretation in multiple regression

---

## Question 22
- [ ] Done

**What are the assumptions of linear regression?**

**Answer:**

### Definition
Linear regression requires several assumptions to produce valid, reliable estimates. Violating these assumptions can lead to biased coefficients, incorrect standard errors, and misleading conclusions.

### The 5 Key Assumptions (LINE + M)

| Assumption | Description | How to Check |
|------------|-------------|--------------|
| **L**inearity | Linear relationship between X and y | Residual vs Fitted plot |
| **I**ndependence | Observations are independent | Durbin-Watson test |
| **N**ormality | Residuals are normally distributed | Q-Q plot, Shapiro-Wilk test |
| **E**qual Variance (Homoscedasticity) | Constant variance of residuals | Residual vs Fitted plot |
| **M**ulticollinearity (absence) | Features not highly correlated | VIF (Variance Inflation Factor) |

### Detailed Explanation

**1. Linearity**
- Relationship between X and y should be linear
- Check: Residuals should show no pattern when plotted against fitted values

**2. Independence**
- Each observation independent of others
- Violation common in time series (autocorrelation)

**3. Normality of Residuals**
- Residuals should follow normal distribution
- Important for hypothesis testing and confidence intervals

**4. Homoscedasticity**
- Variance of residuals should be constant across all X values
- Heteroscedasticity: variance changes (fan shape in residual plot)

**5. No Multicollinearity**
- Independent variables should not be highly correlated
- VIF > 10 indicates problematic multicollinearity

### Python Code to Check Assumptions
```python
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats

# Fit model
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()
residuals = model.resid

# 1. Check Normality (Shapiro-Wilk)
stat, p_value = stats.shapiro(residuals)
print(f"Normality p-value: {p_value:.4f}")  # p > 0.05 = normal

# 2. Check Multicollinearity (VIF)
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(f"VIF: {vif}")  # VIF < 10 is acceptable

# 3. Durbin-Watson (Independence) - from model summary
print(f"Durbin-Watson: {sm.stats.stattools.durbin_watson(residuals)}")
# ~2 = no autocorrelation, <1 or >3 = problem
```

### Interview Tips
- Violations don't always invalidate the model but affect interpretations
- Multicollinearity affects coefficients but not predictions
- For prediction focus, assumptions matter less than for inference
- Robust regression techniques exist for violated assumptions

---

## Question 23
- [ ] Done

**What is polynomial regression and when is it used?**

**Answer:**

### Definition
Polynomial regression extends linear regression by adding polynomial terms (x², x³, etc.) to capture non-linear relationships. It's still a linear model in terms of coefficients but models curved patterns in data.

### Mathematical Formulation

**Linear**: $y = w_0 + w_1x$

**Polynomial (degree 2)**: $y = w_0 + w_1x + w_2x^2$

**Polynomial (degree n)**: $y = w_0 + w_1x + w_2x^2 + ... + w_nx^n$

### Core Concepts
- **Degree**: Highest power of polynomial (complexity control)
- **Still Linear in Coefficients**: Can use linear regression after transformation
- **Feature Transformation**: x → [x, x², x³, ...]
- **Overfitting Risk**: High degree = overfitting

### When to Use
- Data shows curved/non-linear pattern
- Linear regression gives poor fit
- Relationship is smooth but not linear
- Physical phenomena with known polynomial behavior

### Choosing Degree

| Degree | Behavior | Risk |
|--------|----------|------|
| 1 | Linear (straight line) | Underfitting |
| 2-3 | Captures curves | Usually good |
| >5 | Complex curves | Overfitting likely |

### Python Code Example
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Create polynomial features and fit
degree = 2
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=degree)),
    ('linear', LinearRegression())
])
poly_model.fit(X_train, y_train)

# Find optimal degree using cross-validation
for d in range(1, 6):
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=d)),
        ('linear', LinearRegression())
    ])
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"Degree {d}: CV MSE = {-scores.mean():.4f}")
```

### Intuition
- Degree 1: Straight line
- Degree 2: Parabola (U or inverted U)
- Degree 3: S-curve
- Higher: More wiggles, more risk of overfitting

### Interview Tips
- Use cross-validation to select optimal degree
- High degree with limited data = overfitting
- Consider regularization (Ridge/Lasso) for high-degree polynomials
- Polynomial features increase dimensionality: n features with degree d → $\binom{n+d}{d}$ features

---

## Question 24
- [ ] Done

**Explain the concept of regularization in regression models (e.g., Ridge, Lasso).**

**Answer:**

### Definition
Regularization adds a penalty term to the loss function to constrain coefficient magnitudes, preventing overfitting. Ridge (L2) shrinks coefficients toward zero, while Lasso (L1) can zero out coefficients entirely for feature selection.

### Mathematical Formulation

**Ordinary Linear Regression**:
$$\min \sum(y_i - \hat{y}_i)^2$$

**Ridge (L2)**:
$$\min \sum(y_i - \hat{y}_i)^2 + \lambda \sum w_j^2$$

**Lasso (L1)**:
$$\min \sum(y_i - \hat{y}_i)^2 + \lambda \sum |w_j|$$

### Comparison

| Aspect | Ridge (L2) | Lasso (L1) |
|--------|------------|------------|
| Penalty | Sum of squared weights | Sum of absolute weights |
| Feature Selection | No (shrinks but keeps all) | Yes (zeros out features) |
| Solution | Closed-form exists | Requires iterative methods |
| Multicollinearity | Distributes weight among correlated | Picks one, zeros others |
| When to Use | Many small effects | Sparse solutions needed |

### Effect of λ (Regularization Strength)
- λ = 0: No regularization (ordinary regression)
- λ → ∞: All coefficients → 0 (underfitting)
- Optimal λ: Found via cross-validation

### Python Code Example
```python
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV

# Ridge with cross-validation for optimal alpha
ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
ridge_cv.fit(X_train, y_train)
print(f"Best alpha: {ridge_cv.alpha_}")

# Lasso with cross-validation
lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_train, y_train)
print(f"Non-zero coefficients: {sum(lasso_cv.coef_ != 0)}")
```

### Interview Tips
- Always standardize features before regularization
- Lasso for automatic feature selection
- Ridge when all features expected to contribute
- Use ElasticNet to combine both benefits

---

## Question 25
- [ ] Done

**What is logistic regression and how does it differ from linear regression?**

**Answer:**

### Definition
Logistic regression is a classification algorithm that predicts the probability of a binary outcome using the sigmoid function. Unlike linear regression which predicts continuous values, logistic regression outputs probabilities between 0 and 1.

### Core Differences

| Aspect | Linear Regression | Logistic Regression |
|--------|------------------|---------------------|
| Output | Continuous (-∞ to +∞) | Probability (0 to 1) |
| Task | Regression | Classification |
| Function | Linear | Sigmoid |
| Loss Function | MSE | Log Loss (Cross-Entropy) |
| Interpretation | Direct prediction | Probability of class |

### Mathematical Formulation

**Linear Regression**: $y = w^Tx + b$

**Logistic Regression**: $P(y=1) = \sigma(w^Tx + b) = \frac{1}{1 + e^{-(w^Tx + b)}}$

**Decision**: Predict 1 if $P(y=1) > 0.5$, else predict 0

**Log Loss**:
$$J = -\frac{1}{n}\sum[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$$

### Sigmoid Function
- Maps any real number to (0, 1)
- S-shaped curve
- $\sigma(0) = 0.5$, $\sigma(\infty) = 1$, $\sigma(-\infty) = 0$

### Python Code Example
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict classes
y_pred = model.predict(X_test)

# Predict probabilities
y_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Coefficients: {model.coef_}")
```

### Intuition
- Linear regression: "What is the house price?"
- Logistic regression: "What is the probability the email is spam?"

### Interview Tips
- Despite the name, logistic regression is for classification
- Coefficients represent log-odds change per unit increase
- Can be extended to multi-class (One-vs-Rest, Softmax)
- Assumes linear decision boundary (in feature space)

---

## Question 26
- [ ] Done

**Describe the concept of stepwise regression and its variants.**

**Answer:**

### Definition
Stepwise regression is an automated feature selection method that iteratively adds or removes features based on statistical significance to find the best subset. It combines forward selection and backward elimination.

### Variants

| Method | Process | Description |
|--------|---------|-------------|
| Forward Selection | Start empty, add features | Add one feature at a time that improves model most |
| Backward Elimination | Start full, remove features | Remove least significant feature iteratively |
| Stepwise (Bidirectional) | Add and remove | Combination of forward and backward |

### Forward Selection Algorithm
1. Start with no features
2. For each remaining feature, fit model and compute p-value
3. Add feature with lowest p-value (if < threshold, e.g., 0.05)
4. Repeat until no significant features remain

### Backward Elimination Algorithm
1. Start with all features
2. Fit model, find feature with highest p-value
3. If p-value > threshold (e.g., 0.05), remove feature
4. Repeat until all remaining features are significant

### Python Code Example
```python
import statsmodels.api as sm

def backward_elimination(X, y, threshold=0.05):
    features = list(X.columns)
    while len(features) > 0:
        X_subset = sm.add_constant(X[features])
        model = sm.OLS(y, X_subset).fit()
        p_values = model.pvalues[1:]  # Exclude constant
        max_p = p_values.max()
        if max_p > threshold:
            feature_to_remove = p_values.idxmax()
            features.remove(feature_to_remove)
            print(f"Removed {feature_to_remove}, p-value: {max_p:.4f}")
        else:
            break
    return features

# Usage
selected_features = backward_elimination(X, y)
print(f"Selected features: {selected_features}")
```

### Limitations
- Greedy approach, may miss optimal subset
- Multiple testing problem (inflated Type I error)
- Doesn't guarantee global optimum
- Can be unstable with correlated features

### Interview Tips
- Stepwise methods are considered outdated for feature selection
- Modern alternatives: Lasso, Elastic Net, Recursive Feature Elimination
- AIC/BIC criteria better than p-values for model selection
- Cross-validation should validate final feature set

---

## Question 27
- [ ] Done

**What is Elastic Net regularization and how does it combine L1 and L2 penalties?**

**Answer:**

### Definition
Elastic Net combines L1 (Lasso) and L2 (Ridge) penalties to get the benefits of both: feature selection from L1 and coefficient stability from L2. It's particularly useful when features are correlated.

### Mathematical Formulation
$$\min \sum(y_i - \hat{y}_i)^2 + \lambda_1 \sum|w_j| + \lambda_2 \sum w_j^2$$

Or with mixing parameter:
$$\min \sum(y_i - \hat{y}_i)^2 + \alpha \left[ \rho \sum|w_j| + \frac{(1-\rho)}{2} \sum w_j^2 \right]$$

Where:
- $\alpha$ = overall regularization strength
- $\rho$ = L1 ratio (0 = pure Ridge, 1 = pure Lasso)

### Comparison

| Scenario | Best Choice |
|----------|-------------|
| Few features matter (sparse) | Lasso |
| All features matter | Ridge |
| Correlated features, some sparse | Elastic Net |
| Groups of correlated features | Elastic Net |

### Why Combine L1 and L2?

**Lasso Limitations:**
- Selects arbitrarily among correlated features
- Unstable when features > samples

**Ridge Limitations:**
- Keeps all features (no selection)
- Can't produce sparse solutions

**Elastic Net Advantages:**
- Selects groups of correlated features together
- Stable when features > samples
- Provides feature selection with stability

### Python Code Example
```python
from sklearn.linear_model import ElasticNet, ElasticNetCV

# ElasticNet with manual parameters
# l1_ratio: 0 = Ridge, 1 = Lasso, 0.5 = 50-50 mix
en = ElasticNet(alpha=0.1, l1_ratio=0.5)
en.fit(X_train, y_train)

# Cross-validation to find optimal parameters
en_cv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1], cv=5)
en_cv.fit(X_train, y_train)
print(f"Best alpha: {en_cv.alpha_:.4f}")
print(f"Best l1_ratio: {en_cv.l1_ratio_:.2f}")
print(f"Non-zero coefficients: {sum(en_cv.coef_ != 0)}")
```

### Interview Tips
- l1_ratio = 0.5 is a good starting point
- Use cross-validation to tune both alpha and l1_ratio
- Preferred over pure Lasso when features are correlated
- Standardize features before applying Elastic Net

---

## Question 28
- [ ] Done

**Explain the concept of gradient descent in the context of regression.**

**Answer:**

### Definition
Gradient descent is an iterative optimization algorithm that finds the minimum of a function by taking steps proportional to the negative gradient. In regression, it minimizes the loss function (MSE) by updating weights in the direction of steepest descent.

### Core Concepts
- **Gradient**: Vector of partial derivatives pointing to steepest increase
- **Learning Rate (α)**: Step size for each update
- **Convergence**: When loss stops decreasing significantly
- **Local vs Global Minimum**: GD may get stuck in local minima (for non-convex functions)

### Algorithm Steps
1. Initialize weights randomly (or zeros)
2. Compute predictions: $\hat{y} = Xw$
3. Calculate loss: $J = \frac{1}{n}\sum(y - \hat{y})^2$
4. Compute gradient: $\nabla J = \frac{2}{n}X^T(Xw - y)$
5. Update weights: $w = w - \alpha \nabla J$
6. Repeat until convergence

### Update Rule
$$w_{new} = w_{old} - \alpha \frac{\partial J}{\partial w}$$

### Variants

| Variant | Batch Size | Pros | Cons |
|---------|------------|------|------|
| Batch GD | All data | Stable, smooth | Slow, memory-heavy |
| Stochastic (SGD) | 1 sample | Fast, can escape local minima | Noisy, unstable |
| Mini-batch | k samples | Balanced | Hyperparameter (batch size) |

### Python Code Example
```python
import numpy as np

def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    
    for i in range(iterations):
        # Forward pass
        y_pred = np.dot(X, weights) + bias
        
        # Compute gradients
        dw = (2/n_samples) * np.dot(X.T, (y_pred - y))
        db = (2/n_samples) * np.sum(y_pred - y)
        
        # Update weights
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        # Optional: print loss every 100 iterations
        if i % 100 == 0:
            loss = np.mean((y_pred - y)**2)
            print(f"Iteration {i}, Loss: {loss:.4f}")
    
    return weights, bias
```

### Learning Rate Impact
- Too small: Slow convergence
- Too large: Overshoots, may diverge
- Just right: Fast convergence to minimum

### Interview Tips
- MSE is convex, so GD finds global minimum
- Feature scaling crucial for faster convergence
- Adam optimizer often preferred in deep learning
- Normal equation is faster for small datasets (closed-form solution)

---

## Question 29
- [ ] Done

**What is classification in machine learning?**

**Answer:**

### Definition
Classification is a supervised learning task where the model learns to assign input data to predefined discrete categories (classes). The model learns decision boundaries from labeled training data to predict class labels for new observations.

### Core Concepts
- **Classes/Labels**: Discrete categories to predict (spam/not spam, cat/dog)
- **Decision Boundary**: Separates different classes in feature space
- **Training**: Learn mapping from features to classes
- **Prediction**: Assign class label to new data

### Types of Classification

| Type | Classes | Example |
|------|---------|---------|
| Binary | 2 | Spam detection, Fraud detection |
| Multi-class | 3+ (mutually exclusive) | Digit recognition (0-9) |
| Multi-label | Multiple labels per instance | Image tagging (beach, sunset, people) |

### Common Classification Algorithms
- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Neural Networks

### Python Code Example
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)  # Class probabilities

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: Of positive predictions, how many correct
- **Recall**: Of actual positives, how many detected
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve

### Interview Tips
- Choose metric based on problem: recall for disease detection, precision for spam
- Class imbalance requires special handling
- Threshold tuning affects precision-recall trade-off
- Always evaluate on held-out test data

---

## Question 30
- [ ] Done

**Explain the difference between binary classification and multi-class classification.**

**Answer:**

### Definition
**Binary Classification**: Predicts one of two mutually exclusive classes (e.g., yes/no, spam/not spam). **Multi-class Classification**: Predicts one of three or more mutually exclusive classes (e.g., cat/dog/bird).

### Comparison

| Aspect | Binary | Multi-class |
|--------|--------|-------------|
| Classes | 2 | 3+ |
| Output | Single probability | Probability per class |
| Loss Function | Binary Cross-Entropy | Categorical Cross-Entropy |
| Activation (NN) | Sigmoid | Softmax |
| Metrics | Accuracy, AUC-ROC | Accuracy, Macro/Micro F1 |

### Multi-class Strategies

**One-vs-Rest (OvR) / One-vs-All (OvA)**
- Train N binary classifiers (one per class)
- Each classifier: "class i" vs "all other classes"
- Predict: class with highest confidence

**One-vs-One (OvO)**
- Train N(N-1)/2 binary classifiers (one per pair)
- Each classifier: "class i" vs "class j"
- Predict: class that wins most "votes"

### Mathematical Formulation

**Binary** (Sigmoid):
$$P(y=1) = \frac{1}{1 + e^{-z}}$$

**Multi-class** (Softmax):
$$P(y=k) = \frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}$$

### Python Code Example
```python
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Binary classification (native)
binary_clf = LogisticRegression()
binary_clf.fit(X_train, y_binary)  # y: 0 or 1

# Multi-class classification (native - sklearn handles automatically)
multi_clf = LogisticRegression(multi_class='multinomial')
multi_clf.fit(X_train, y_multi)  # y: 0, 1, 2, ...

# Explicit One-vs-Rest
ovr_clf = OneVsRestClassifier(LogisticRegression())
ovr_clf.fit(X_train, y_multi)

# Probabilities
print(multi_clf.predict_proba(X_test[0:1]))  # [P(0), P(1), P(2), ...]
```

### Interview Tips
- Most sklearn classifiers handle multi-class natively
- Softmax ensures probabilities sum to 1
- For highly imbalanced multi-class, consider class weights
- OvR is more common and typically sufficient

---

## Question 31
- [ ] Done

**What is logistic regression and how is it used for classification?**

**Answer:**

### Definition
Logistic regression is a linear classification algorithm that uses the sigmoid function to model the probability of class membership. Despite the name "regression," it's used for classification by thresholding the predicted probability.

### How It Works
1. Compute linear combination: $z = w^Tx + b$
2. Apply sigmoid function: $P(y=1) = \sigma(z) = \frac{1}{1 + e^{-z}}$
3. Apply threshold (default 0.5): if P > 0.5, predict class 1

### Sigmoid Function Properties
- Output range: (0, 1) → interpretable as probability
- $\sigma(0) = 0.5$
- $\sigma(+\infty) = 1$, $\sigma(-\infty) = 0$
- Smooth, differentiable → gradient descent works

### Loss Function (Log Loss / Binary Cross-Entropy)
$$J = -\frac{1}{n}\sum_{i=1}^{n}[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$$

### Coefficient Interpretation
- Coefficients represent log-odds
- $e^{w_j}$ = odds ratio for feature j
- Positive coefficient → increases probability of class 1

### Python Code Example
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict class labels
y_pred = model.predict(X_test)

# Predict probabilities
y_proba = model.predict_proba(X_test)[:, 1]  # P(class=1)

# Custom threshold
threshold = 0.3
y_pred_custom = (y_proba >= threshold).astype(int)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
```

### When to Use Logistic Regression
- Binary or multi-class classification
- Need interpretable model (coefficients explain relationships)
- Linear decision boundary is appropriate
- Baseline model for comparison

### Interview Tips
- Assumes linear decision boundary (may underfit complex data)
- Can add polynomial features for non-linear boundaries
- Regularization (L1, L2) helps prevent overfitting
- Threshold tuning is crucial for imbalanced data

---

## Question 32
- [ ] Done

**Describe the concept of decision trees and how they work.**

**Answer:**

### Definition
Decision trees are non-parametric models that make predictions by recursively splitting data based on feature values. Each internal node tests a feature, branches represent outcomes, and leaf nodes contain predictions. They mimic human decision-making.

### Core Concepts
- **Root Node**: Top node, entire dataset
- **Internal Node**: Decision point (feature test)
- **Branch**: Outcome of test
- **Leaf Node**: Final prediction (class or value)
- **Splitting Criteria**: How to choose best split

### How It Works (Algorithm)
1. Start with all data at root
2. Find best feature and threshold to split
3. Split data into child nodes
4. Repeat recursively for each child
5. Stop when: max depth reached, min samples, or pure node

### Splitting Criteria

**For Classification:**
- **Gini Impurity**: $Gini = 1 - \sum p_i^2$
- **Entropy**: $H = -\sum p_i \log_2(p_i)$
- **Information Gain**: Reduction in entropy after split

**For Regression:**
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error

### Example
```
                [Is Age > 30?]
                  /        \
               Yes          No
               /              \
        [Income > 50K?]    [Student?]
          /      \          /      \
        Buy    Don't     Buy    Don't
```

### Python Code Example
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Train decision tree
dt = DecisionTreeClassifier(max_depth=3, min_samples_split=10)
dt.fit(X_train, y_train)

# Predict
y_pred = dt.predict(X_test)

# Visualize tree
plt.figure(figsize=(15, 10))
plot_tree(dt, feature_names=feature_names, class_names=class_names, filled=True)
plt.show()

# Feature importance
importance = dt.feature_importances_
```

### Advantages and Disadvantages

| Advantages | Disadvantages |
|------------|---------------|
| Easy to interpret | Prone to overfitting |
| Handles non-linear | Unstable (small data changes) |
| No scaling needed | Biased to dominant class |
| Handles mixed data | Greedy, not global optimum |

### Interview Tips
- Prone to overfitting → use pruning or max_depth
- Ensemble methods (Random Forest) address instability
- Feature importance: total reduction in criterion from splits
- No need to scale or normalize features

---

## Question 33
- [ ] Done

**What is the random forest algorithm and its advantages?**

**Answer:**

### Definition
Random Forest is an ensemble learning method that combines multiple decision trees using bagging (bootstrap aggregating) and random feature selection. Final prediction is the majority vote (classification) or average (regression) of all trees.

### Core Concepts
- **Bagging**: Each tree trained on bootstrap sample (random sample with replacement)
- **Feature Randomness**: Each split considers random subset of features
- **Aggregation**: Combine predictions from all trees
- **Decorrelation**: Random features reduce correlation between trees

### How It Works
1. Create N bootstrap samples from training data
2. For each sample, build a decision tree:
   - At each split, consider √p random features (classification) or p/3 (regression)
   - Grow tree fully (no pruning by default)
3. Aggregate predictions:
   - Classification: Majority vote
   - Regression: Average

### Advantages

| Advantage | Explanation |
|-----------|-------------|
| Reduces Overfitting | Averaging reduces variance |
| Handles High Dimensions | Random feature selection |
| No Feature Scaling | Tree-based, scale-invariant |
| Feature Importance | Built-in importance measure |
| Handles Missing Data | Can handle via surrogate splits |
| Robust to Outliers | Trees are resistant to outliers |
| Parallelizable | Trees can be trained independently |

### Python Code Example
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Limit tree depth
    min_samples_split=5,   # Min samples to split
    random_state=42
)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)

# Feature importance
importance = rf.feature_importances_

# Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
```

### Key Hyperparameters
- `n_estimators`: More trees = better but slower (100-500)
- `max_depth`: Limit overfitting
- `min_samples_split`: Minimum samples to split a node
- `max_features`: Number of features per split ('sqrt' or 'log2')

### Interview Tips
- Random Forest = Bagging + Random Feature Selection
- OOB (Out-of-Bag) error provides free cross-validation estimate
- Generally outperforms single decision tree
- Less interpretable than single tree

---

## Question 34
- [ ] Done

**Explain the concept of support vector machines (SVM) and their kernels.**

**Answer:**

### Definition
SVM is a discriminative classifier that finds the optimal hyperplane maximizing the margin between classes. Support vectors are the data points closest to the decision boundary. Kernels enable SVM to handle non-linear decision boundaries by mapping data to higher dimensions.

### Core Concepts
- **Hyperplane**: Decision boundary separating classes
- **Margin**: Distance between hyperplane and nearest points
- **Support Vectors**: Points on the margin boundary
- **Kernel Trick**: Implicitly map data to higher dimensions

### Mathematical Formulation

**Objective**: Maximize margin = $\frac{2}{\|w\|}$

**Constraint**: $y_i(w \cdot x_i + b) \geq 1$ for all i

**Optimization**:
$$\min \frac{1}{2}\|w\|^2 \quad \text{subject to} \quad y_i(w \cdot x_i + b) \geq 1$$

### Kernel Functions

| Kernel | Formula | Use Case |
|--------|---------|----------|
| Linear | $K(x, y) = x \cdot y$ | Linearly separable data |
| Polynomial | $K(x, y) = (x \cdot y + c)^d$ | Polynomial boundaries |
| RBF (Gaussian) | $K(x, y) = e^{-\gamma\|x-y\|^2}$ | Most common, non-linear |
| Sigmoid | $K(x, y) = \tanh(\alpha x \cdot y + c)$ | Neural network-like |

### Soft Margin (C Parameter)
- C = regularization parameter
- High C: Less tolerance for misclassification, may overfit
- Low C: More tolerance, smoother boundary, may underfit

### Python Code Example
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# SVM requires scaled data
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))
])

svm_pipeline.fit(X_train, y_train)
y_pred = svm_pipeline.predict(X_test)

# For probability predictions
svm_proba = SVC(kernel='rbf', probability=True)
svm_proba.fit(X_train, y_train)
y_proba = svm_proba.predict_proba(X_test)
```

### Key Hyperparameters
- **C**: Trade-off between margin size and misclassification
- **gamma (RBF)**: Influence of single training example (high = overfitting)
- **degree (Polynomial)**: Polynomial degree

### Interview Tips
- Always scale features before SVM
- RBF kernel is good default for non-linear problems
- SVM is memory-intensive (stores support vectors)
- Not ideal for very large datasets (use LinearSVC instead)

---

## Question 35
- [ ] Done

**What is the k-nearest neighbors (KNN) algorithm and how does it work?**

**Answer:**

### Definition
KNN is a non-parametric, lazy learning algorithm that classifies a point based on the majority class of its k nearest neighbors. It stores all training data and makes predictions by computing distances to find the closest training examples.

### Core Concepts
- **Lazy Learning**: No training phase, all computation at prediction time
- **Instance-based**: Stores training data, compares new points
- **Distance Metric**: Measures similarity between points
- **K**: Number of neighbors to consider

### How It Works
1. Store all training data
2. For new point, calculate distance to all training points
3. Find K nearest neighbors
4. Classification: Majority vote among K neighbors
5. Regression: Average of K neighbors' values

### Distance Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| Euclidean | $\sqrt{\sum(x_i - y_i)^2}$ | Continuous, default |
| Manhattan | $\sum\|x_i - y_i\|$ | Grid-like paths |
| Minkowski | $(\sum\|x_i - y_i\|^p)^{1/p}$ | Generalized |
| Cosine | $1 - \frac{x \cdot y}{\|x\| \|y\|}$ | Text, high-dimensional |

### Choosing K
- K = 1: Very sensitive to noise (overfitting)
- K = n: Always predicts majority class (underfitting)
- Odd K: Avoids ties in binary classification
- Typical: K = √n or use cross-validation

### Python Code Example
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Scale features (crucial for KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_scaled, y_train)

# Predict
y_pred = knn.predict(X_test_scaled)

# Find optimal K using cross-validation
for k in range(1, 20, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    print(f"K={k}: Accuracy = {scores.mean():.3f}")
```

### Advantages and Disadvantages

| Advantages | Disadvantages |
|------------|---------------|
| Simple, intuitive | Slow prediction (large datasets) |
| No training time | Curse of dimensionality |
| Non-parametric | Sensitive to irrelevant features |
| Works for multi-class | Requires feature scaling |

### Interview Tips
- Always scale features before KNN
- Use KD-Tree or Ball-Tree for faster lookup in large datasets
- Weighted KNN: Closer neighbors have more influence
- High dimensions reduce effectiveness (curse of dimensionality)

---

## Question 36
- [ ] Done

**Describe the Naive Bayes algorithm and its assumptions.**

**Answer:**

### Definition
Naive Bayes is a probabilistic classifier based on Bayes' theorem with the "naive" assumption that all features are conditionally independent given the class. Despite this simplification, it works surprisingly well for many real-world problems, especially text classification.

### Bayes' Theorem
$$P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}$$

### Naive Assumption
Features are conditionally independent:
$$P(X|y) = P(x_1|y) \cdot P(x_2|y) \cdot ... \cdot P(x_n|y) = \prod_{i=1}^{n} P(x_i|y)$$

### Classification Rule
$$\hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i|y)$$

### Variants

| Variant | Feature Type | Distribution |
|---------|--------------|--------------|
| Gaussian NB | Continuous | Normal distribution |
| Multinomial NB | Discrete counts | Multinomial |
| Bernoulli NB | Binary | Bernoulli |

### Assumptions
1. **Feature Independence**: Features are conditionally independent given class
2. **Feature Distribution**: Each variant assumes specific distribution
3. **Equal Feature Importance**: All features contribute equally

### Python Code Example
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Gaussian NB (continuous features)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
y_proba = gnb.predict_proba(X_test)

# Multinomial NB (text classification)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(train_texts)
X_test_vec = vectorizer.transform(test_texts)

mnb = MultinomialNB()
mnb.fit(X_train_vec, y_train)
y_pred_text = mnb.predict(X_test_vec)
```

### Advantages and Disadvantages

| Advantages | Disadvantages |
|------------|---------------|
| Fast training and prediction | Independence assumption rarely true |
| Works well with high dimensions | Can be outperformed by other models |
| Good for text classification | Sensitive to feature correlations |
| Handles missing data well | Probability estimates may be poor |

### Interview Tips
- Despite "naive" assumption, works well in practice (especially text)
- Laplace smoothing handles zero probabilities
- Fast and scalable to large datasets
- Good baseline for text classification before trying complex models

---

## Question 37
- [ ] Done

**What is the difference between a hard classifier and a soft classifier?**

**Answer:**

### Definition
**Hard Classifier**: Outputs only discrete class labels (e.g., "spam" or "not spam"). **Soft Classifier**: Outputs probability or confidence scores for each class, allowing more nuanced predictions and threshold tuning.

### Comparison

| Aspect | Hard Classifier | Soft Classifier |
|--------|----------------|-----------------|
| Output | Class label (0/1) | Probability (0-1) |
| Information | Binary decision | Confidence level |
| Threshold | Fixed (usually 0.5) | Adjustable |
| Ranking | Not possible | Can rank by confidence |
| Example | SVM (default), Decision Tree | Logistic Regression, Random Forest |

### When to Use

**Hard Classification:**
- Simple yes/no decision needed
- Equal cost for all types of errors
- Don't need probability estimates

**Soft Classification:**
- Need confidence in predictions
- Imbalanced classes (adjust threshold)
- Ranking items by likelihood
- Combining predictions (ensembles)
- Different costs for FP vs FN

### Converting Between Types
- Hard → Soft: Not possible (information lost)
- Soft → Hard: Apply threshold (default 0.5)

### Python Code Example
```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Soft classifier: Logistic Regression (native probabilities)
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_proba = lr.predict_proba(X_test)[:, 1]  # Soft output
y_pred = lr.predict(X_test)               # Hard output (threshold=0.5)

# Custom threshold for imbalanced data
threshold = 0.3  # Lower threshold → more positive predictions
y_pred_custom = (y_proba >= threshold).astype(int)

# SVM: Hard by default, soft with probability=True
svm_hard = SVC(kernel='rbf')
svm_soft = SVC(kernel='rbf', probability=True)

svm_soft.fit(X_train, y_train)
y_proba_svm = svm_soft.predict_proba(X_test)  # Now soft
```

### Practical Relevance
- **Medical Diagnosis**: Soft output helps doctors understand uncertainty
- **Fraud Detection**: Adjust threshold based on cost of missed fraud
- **Recommendation**: Rank items by probability of interest
- **Ensemble**: Average probabilities from multiple models

### Interview Tips
- Most classifiers can provide soft output (predict_proba)
- Soft classifiers are more flexible and informative
- Threshold tuning is crucial for imbalanced datasets
- ROC and Precision-Recall curves require soft outputs

---

## Question 38
- [ ] Done

**Explain the concept of ensemble learning and its techniques (e.g., bagging, boosting).**

**Answer:**

### Definition
Ensemble learning combines multiple models (weak learners) to create a stronger model with better performance than any individual model. Main techniques are bagging (reduces variance), boosting (reduces bias), and stacking (combines diverse models).

### Core Concepts
- **Weak Learner**: Model slightly better than random guessing
- **Strong Learner**: Combined ensemble with good performance
- **Diversity**: Different models make different errors
- **Aggregation**: Combine predictions (voting, averaging)

### Bagging vs Boosting

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| Training | Parallel (independent) | Sequential (dependent) |
| Samples | Bootstrap (random with replacement) | Weighted (focus on errors) |
| Goal | Reduce variance | Reduce bias |
| Base Learners | Independent | Correct previous errors |
| Overfitting | Less prone | Can overfit |
| Example | Random Forest | XGBoost, AdaBoost |

### Bagging Algorithm
1. Create N bootstrap samples from training data
2. Train N independent models on each sample
3. Aggregate predictions: vote (classification) or average (regression)

### Boosting Algorithm (AdaBoost)
1. Initialize equal weights for all samples
2. Train model, identify misclassified samples
3. Increase weights of misclassified samples
4. Train next model focusing on hard examples
5. Combine models with weighted voting

### Stacking
- Train multiple diverse models (Level 0)
- Train meta-model on predictions of base models (Level 1)
- Meta-model learns how to combine base predictions

### Python Code Example
```python
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression

# Bagging: Random Forest
rf = RandomForestClassifier(n_estimators=100)

# Boosting: AdaBoost, GradientBoosting
adaboost = AdaBoostClassifier(n_estimators=100)
gboost = GradientBoostingClassifier(n_estimators=100)

# Stacking
estimators = [('rf', rf), ('ada', adaboost)]
stacking = StackingClassifier(estimators=estimators, 
                               final_estimator=LogisticRegression())
stacking.fit(X_train, y_train)
```

### Interview Tips
- Bagging for high-variance models (deep trees)
- Boosting for high-bias models (shallow trees)
- XGBoost, LightGBM are state-of-the-art boosting implementations
- Ensemble almost always outperforms single models

---

## Question 39
- [ ] Done

**What are the common evaluation metrics for classification models?**

**Answer:**

### Definition
Classification metrics measure how well a model predicts discrete class labels. Different metrics emphasize different aspects of performance (overall accuracy, class-specific performance, ranking ability).

### Core Metrics Summary

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | Positive prediction quality |
| Recall (Sensitivity) | TP/(TP+FN) | Positive class detection |
| Specificity | TN/(TN+FP) | Negative class detection |
| F1-Score | 2×(P×R)/(P+R) | Precision-Recall balance |
| AUC-ROC | Area under ROC curve | Ranking ability |

### When to Use Each Metric

| Scenario | Best Metric | Reason |
|----------|-------------|--------|
| Balanced classes | Accuracy | Fair representation |
| Imbalanced classes | F1, AUC-PR | Accuracy misleading |
| Cost of FP high | Precision | Minimize false alarms |
| Cost of FN high | Recall | Catch all positives |
| Ranking needed | AUC-ROC | Threshold-independent |

### Python Code Example
```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Individual metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")

# Complete report
print(classification_report(y_test, y_pred))
```

### Interview Tips
- Never use accuracy alone for imbalanced data
- Precision-Recall trade-off: optimizing one reduces the other
- AUC-ROC requires probability outputs
- Choose metric based on business cost of errors

---

## Question 40
- [ ] Done

**Explain the concept of confusion matrix and its components.**

**Answer:**

### Definition
A confusion matrix is a table that summarizes classification performance by showing actual vs predicted class labels. It displays True Positives, True Negatives, False Positives, and False Negatives, from which all classification metrics can be derived.

### Structure (Binary Classification)
```
                    Predicted
                  Positive  Negative
Actual Positive     TP        FN
       Negative     FP        TN
```

### Components Explained

| Component | Meaning | Example (Spam Detection) |
|-----------|---------|--------------------------|
| TP (True Positive) | Correctly predicted positive | Spam correctly marked as spam |
| TN (True Negative) | Correctly predicted negative | Ham correctly marked as ham |
| FP (False Positive) | Incorrectly predicted positive | Ham wrongly marked as spam (Type I) |
| FN (False Negative) | Incorrectly predicted negative | Spam wrongly marked as ham (Type II) |

### Derived Metrics
- **Accuracy** = (TP + TN) / Total
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **Specificity** = TN / (TN + FP)
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)

### Python Code Example
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# [[TN, FP],
#  [FN, TP]]

# Extract components
tn, fp, fn, tp = cm.ravel()
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

# Visualize
disp = ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```

### Multi-class Confusion Matrix
- Diagonal elements = correct predictions
- Off-diagonal = misclassifications
- Row = actual class, Column = predicted class

### Interview Tips
- FP (Type I Error): False alarm
- FN (Type II Error): Missed detection
- Medical: FN often more critical (missing disease)
- Spam: FP often more critical (losing important email)
- Normalize by row to see recall per class

---

## Question 41
- [ ] Done

**What is the ROC curve and how is it used to evaluate classifier performance?**

**Answer:**

### Definition
ROC (Receiver Operating Characteristic) curve plots True Positive Rate (Recall) vs False Positive Rate at various classification thresholds. AUC-ROC (Area Under Curve) summarizes overall performance: higher AUC = better classifier.

### Key Terms
- **TPR (True Positive Rate)** = TP / (TP + FN) = Recall
- **FPR (False Positive Rate)** = FP / (FP + TN) = 1 - Specificity
- **AUC**: Area under ROC curve (0.5 to 1.0)

### ROC Curve Interpretation
```
TPR (Recall)
    |     Perfect (0,1)
1.0 |         ____
    |        /
    |       / Good classifier
    |      /
0.5 |    /   Random (diagonal)
    |   /
    |  /
0.0 |_/________________
    0.0      0.5     1.0  FPR
```

### AUC Interpretation

| AUC Value | Performance |
|-----------|-------------|
| 1.0 | Perfect classifier |
| 0.9-1.0 | Excellent |
| 0.8-0.9 | Good |
| 0.7-0.8 | Fair |
| 0.5-0.7 | Poor |
| 0.5 | Random guessing |
| < 0.5 | Worse than random |

### Why Use ROC?
- **Threshold-independent**: Shows performance across all thresholds
- **Compares classifiers**: AUC provides single number comparison
- **Class balance agnostic**: Works for imbalanced data
- **Trade-off visualization**: See TPR vs FPR trade-off

### Python Code Example
```python
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

# Get probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

### ROC vs Precision-Recall Curve
- **ROC**: Better for balanced datasets
- **PR Curve**: Better for imbalanced datasets (focuses on positive class)

### Interview Tips
- AUC = probability that random positive ranks higher than random negative
- ROC can be optimistic for imbalanced data; use PR curve instead
- Optimal threshold: point closest to (0, 1) on ROC curve

---

## Question 42
- [ ] Done

**Describe the concept of precision, recall, and F1-score.**

**Answer:**

### Definition
**Precision**: Of all positive predictions, how many are correct. **Recall**: Of all actual positives, how many are detected. **F1-Score**: Harmonic mean of precision and recall, balancing both metrics.

### Mathematical Formulation

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Intuition

| Metric | Question It Answers | Analogy |
|--------|---------------------|---------|
| Precision | "When I predict positive, am I right?" | Quality of predictions |
| Recall | "Did I catch all the positives?" | Completeness of detection |
| F1 | "Balance between precision and recall?" | Overall effectiveness |

### Trade-off
- **High Precision, Low Recall**: Conservative predictions, misses many positives
- **Low Precision, High Recall**: Catches most positives, many false alarms
- **Threshold tuning**: Lower threshold → higher recall, lower precision

### When to Prioritize

| Scenario | Prioritize | Reason |
|----------|-----------|--------|
| Cancer Detection | Recall | Don't miss any cancer cases |
| Spam Detection | Precision | Don't send important emails to spam |
| Fraud Detection | Recall | Catch all fraudulent transactions |
| Search Engine | Precision | Users expect relevant results |

### Python Code Example
```python
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Calculate metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

# Precision-Recall curve (for threshold tuning)
y_proba = model.predict_proba(X_test)[:, 1]
prec, rec, thresholds = precision_recall_curve(y_test, y_proba)

plt.plot(rec, prec)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```

### Interview Tips
- F1 uses harmonic mean because it penalizes extreme values
- F1 = 0 if either precision or recall is 0
- F-beta score: weighted version (beta > 1 favors recall)
- Always report both precision and recall, not just F1

---

## Question 43
- [ ] Done

**What is the difference between micro-average and macro-average metrics?**

**Answer:**

### Definition
**Micro-average**: Aggregate TP, FP, FN across all classes, then calculate metric. Treats all samples equally. **Macro-average**: Calculate metric per class, then average. Treats all classes equally regardless of size.

### Mathematical Formulation

**Micro-average Precision**:
$$P_{micro} = \frac{\sum_{i} TP_i}{\sum_{i} (TP_i + FP_i)}$$

**Macro-average Precision**:
$$P_{macro} = \frac{1}{K}\sum_{i=1}^{K} P_i$$

### Comparison

| Aspect | Micro-average | Macro-average |
|--------|---------------|---------------|
| Aggregation | Pool all predictions first | Average per-class metrics |
| Weight | By sample count | Equal weight per class |
| Bias | Toward majority class | Equal importance to all classes |
| Imbalanced Data | Dominated by large classes | Shows minority class performance |

### Example (3 classes)
```
Class A: Precision = 0.9, 100 samples
Class B: Precision = 0.8, 50 samples
Class C: Precision = 0.5, 10 samples

Macro: (0.9 + 0.8 + 0.5) / 3 = 0.73
Micro: Weighted by samples (closer to 0.85)
```

### When to Use

| Use Case | Preferred Average |
|----------|-------------------|
| Overall performance | Micro |
| Class balance matters | Macro |
| Minority class important | Macro |
| Large dataset, many classes | Micro |
| Equal class importance | Macro |

### Python Code Example
```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Multi-class predictions
y_true = [0, 0, 1, 1, 2, 2]
y_pred = [0, 1, 1, 1, 2, 0]

# Micro-average (global)
p_micro = precision_score(y_true, y_pred, average='micro')
r_micro = recall_score(y_true, y_pred, average='micro')
f1_micro = f1_score(y_true, y_pred, average='micro')

# Macro-average (per-class mean)
p_macro = precision_score(y_true, y_pred, average='macro')
r_macro = recall_score(y_true, y_pred, average='macro')
f1_macro = f1_score(y_true, y_pred, average='macro')

# Weighted average (weighted by support)
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print(f"Micro F1: {f1_micro:.3f}")
print(f"Macro F1: {f1_macro:.3f}")
print(f"Weighted F1: {f1_weighted:.3f}")
```

### Weighted Average
- Third option: weighted by class support (number of samples)
- Compromise between micro and macro
- `average='weighted'` in sklearn

### Interview Tips
- Large gap between micro and macro indicates class imbalance issue
- Micro = Macro when classes are perfectly balanced
- Report both for complete picture in multi-class problems

---

## Question 44
- [ ] Done

**Explain the concept of stratified k-fold cross-validation.**

**Answer:**

### Definition
Stratified K-Fold CV is a cross-validation variant that maintains the same class distribution in each fold as in the original dataset. It ensures each fold is representative, especially important for imbalanced datasets.

### Why Stratified?
- **Standard K-Fold**: Random splits may have different class distributions
- **Stratified K-Fold**: Each fold has approximately same percentage of each class
- **Problem Solved**: Prevents folds with no minority class samples

### Example
```
Original data: 90% Class A, 10% Class B

Standard K-Fold (K=5):
Fold 1: 88% A, 12% B
Fold 2: 95% A, 5% B  ← Underrepresented
Fold 3: 92% A, 8% B
...

Stratified K-Fold (K=5):
Fold 1: 90% A, 10% B
Fold 2: 90% A, 10% B
Fold 3: 90% A, 10% B  ← Consistent distribution
...
```

### Algorithm Steps
1. Separate samples by class
2. For each class, shuffle and split into K folds
3. Combine corresponding folds from each class
4. Each final fold has same class distribution as original

### Python Code Example
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# Create stratified K-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Manual iteration
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    print(f"Fold {fold}: Train class dist = {y_train.mean():.2f}, Val = {y_val.mean():.2f}")

# Using cross_val_score (automatically stratified for classification)
model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
print(f"CV F1 Scores: {scores}")
print(f"Mean F1: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
```

### When to Use
- **Classification problems**: Default choice
- **Imbalanced datasets**: Essential
- **Multi-class**: Maintains all class proportions

### Related Variants
- **StratifiedGroupKFold**: Stratified + groups don't leak across folds
- **RepeatedStratifiedKFold**: Multiple repetitions with different splits

### Interview Tips
- sklearn's `cross_val_score` uses stratified by default for classifiers
- Always use stratified for classification, especially with imbalance
- For regression, standard K-Fold is typically used
- Shuffle=True ensures random distribution

---

## Question 45
- [ ] Done

**What is the purpose of a validation set in machine learning?**

**Answer:**

### Definition
A validation set is a subset of data (separate from training and test sets) used to tune hyperparameters and make model selection decisions during development. It provides unbiased evaluation before final testing.

### Three-Way Split

| Set | Purpose | Used For |
|-----|---------|----------|
| Training | Learn model parameters | Fitting weights/coefficients |
| Validation | Tune hyperparameters | Model selection, early stopping |
| Test | Final evaluation | Unbiased performance estimate |

### Why Validation Set?
- **Hyperparameter Tuning**: Can't use test set (data leakage)
- **Model Selection**: Compare different algorithms
- **Early Stopping**: Monitor validation loss to prevent overfitting
- **Unbiased Estimation**: Test set remains untouched

### Typical Split Ratios
- Small dataset: 60% train, 20% validation, 20% test
- Large dataset: 80% train, 10% validation, 10% test
- Very large: 98% train, 1% validation, 1% test

### Validation vs Cross-Validation

| Aspect | Holdout Validation | Cross-Validation |
|--------|-------------------|------------------|
| Data Usage | Fixed split | All data used |
| Variance | Higher | Lower |
| Computation | Faster | Slower |
| Best For | Large datasets | Small datasets |

### Python Code Example
```python
from sklearn.model_selection import train_test_split

# Two-step split: first test, then validation
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)  # 0.25 of 0.8 = 0.2

# Result: 60% train, 20% validation, 20% test

# Training workflow
model.fit(X_train, y_train)
val_score = model.score(X_val, y_val)  # Use for hyperparameter tuning

# Only after all tuning is done:
test_score = model.score(X_test, y_test)  # Final unbiased evaluation
```

### Common Mistakes
- Using test set for hyperparameter tuning → overly optimistic results
- Reusing validation set too many times → validation set becomes "learned"
- Not having separate test set → no true performance estimate

### Interview Tips
- Never touch test set until final evaluation
- Cross-validation can replace validation set for small data
- Deep learning often uses validation loss for early stopping
- Large companies often have dedicated held-out test sets

---

## Question 46
- [ ] Done

**What is the difference between a Type I error and a Type II error?**

**Answer:**

### Definition
**Type I Error (False Positive)**: Rejecting a true null hypothesis; predicting positive when actual is negative. **Type II Error (False Negative)**: Failing to reject a false null hypothesis; predicting negative when actual is positive.

### Summary Table

| Error Type | Also Called | Description | Example |
|------------|-------------|-------------|---------|
| Type I | False Positive, α | Detect something that isn't there | Healthy person diagnosed with disease |
| Type II | False Negative, β | Miss something that is there | Sick person diagnosed as healthy |

### In Classification Context

| | Predicted Positive | Predicted Negative |
|------|-------------------|-------------------|
| Actual Positive | TP | **FN (Type II)** |
| Actual Negative | **FP (Type I)** | TN |

### Statistical Hypothesis Testing
- **Null Hypothesis (H₀)**: Default assumption (e.g., "person is healthy")
- **Type I (α)**: Reject H₀ when H₀ is true (false alarm)
- **Type II (β)**: Accept H₀ when H₀ is false (missed detection)
- **Power = 1 - β**: Probability of correctly rejecting false H₀

### Trade-off
- Reducing Type I increases Type II (and vice versa)
- Threshold adjustment controls this trade-off
- Lower threshold → more positives → fewer Type II, more Type I

### Real-World Examples

| Domain | Type I (FP) | Type II (FN) | Critical Error |
|--------|-------------|--------------|----------------|
| Medical | Healthy → Sick | Sick → Healthy | Type II (miss disease) |
| Spam | Ham → Spam | Spam → Ham | Type I (lose important email) |
| Fire Alarm | No fire → Alarm | Fire → No alarm | Type II (miss fire) |
| Court | Innocent → Guilty | Guilty → Innocent | Type I (convict innocent) |

### Python Code Context
```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
# [[TN, FP],
#  [FN, TP]]

tn, fp, fn, tp = cm.ravel()

type_1_error = fp  # False Positives
type_2_error = fn  # False Negatives

type_1_rate = fp / (fp + tn)  # False Positive Rate
type_2_rate = fn / (fn + tp)  # False Negative Rate (1 - Recall)
```

### Interview Tips
- "Type I = False Positive" (both have "positive" connotation - claiming something)
- "Type II = False Negative" (both have "negative" connotation - missing something)
- Which error is worse depends on business context
- Precision addresses Type I; Recall addresses Type II

---

## Question 47
- [ ] Done

**Explain the concept of learning curves and their interpretation.**

**Answer:**

### Definition
Learning curves plot model performance (training and validation scores) against training set size or training iterations. They diagnose whether a model suffers from high bias (underfitting) or high variance (overfitting).

### Types of Learning Curves
1. **Performance vs Training Size**: Shows how more data affects learning
2. **Performance vs Epochs/Iterations**: Shows training progress over time

### Interpretation Patterns

**High Bias (Underfitting)**:
```
Score
  |     ___________________  Validation
  |    /
  |   /
  |  /_____________________ Training
  |_________________________ Training Size
```
- Both scores converge at low value
- More data won't help significantly
- **Solution**: More complex model, more features

**High Variance (Overfitting)**:
```
Score
  |  _____________________  Training (high)
  |
  |          ____________  Validation (gap)
  |         /
  |_________________________ Training Size
```
- Large gap between training and validation
- Training score much higher than validation
- **Solution**: More data, regularization, simpler model

**Good Fit**:
```
Score
  |      ________________  Training
  |     /________________  Validation (small gap)
  |    /
  |_________________________ Training Size
```
- Both scores high, small gap
- Curves converge

### Python Code Example
```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

# Generate learning curve data
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, 
    scoring='accuracy'
)

# Calculate mean and std
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, val_mean, label='Validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend()
plt.show()
```

### Diagnosis Summary

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Both low, converged | High bias | Increase complexity |
| Large gap | High variance | More data, regularization |
| Both high, small gap | Good fit | Keep current model |
| Neither converged | Need more data | Collect more samples |

### Interview Tips
- Learning curves are essential diagnostic tools
- Helps decide: get more data vs. change model
- Plot both training and validation, not just validation
- Consider computational cost when varying training size

---

## Question 48
- [ ] Done

**What is unsupervised learning and how does it differ from supervised learning?**

**Answer:**

### Definition
Unsupervised learning discovers hidden patterns in unlabeled data without predefined outputs. Unlike supervised learning where the model learns from input-output pairs, unsupervised learning finds structure in data through clustering, dimensionality reduction, or association.

### Core Differences

| Aspect | Supervised | Unsupervised |
|--------|-----------|--------------|
| Labels | Required (X, y) | Not required (X only) |
| Goal | Predict output | Find patterns/structure |
| Feedback | Direct error signal | No feedback |
| Tasks | Classification, Regression | Clustering, Dim. Reduction |
| Evaluation | Clear metrics (accuracy, MSE) | Indirect (silhouette, inertia) |

### Unsupervised Learning Tasks
- **Clustering**: Group similar data points (K-Means, DBSCAN)
- **Dimensionality Reduction**: Reduce features (PCA, t-SNE)
- **Anomaly Detection**: Find outliers (Isolation Forest)
- **Association**: Find rules (Apriori, Market Basket)

### Python Code Example
```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Clustering (no labels needed)
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X)  # Only X, no y

# Dimensionality Reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

### Practical Relevance
- **Customer Segmentation**: Group customers by behavior
- **Anomaly Detection**: Fraud, network intrusion
- **Feature Engineering**: Create features for supervised models
- **Exploratory Analysis**: Understand data structure

### Interview Tips
- Unsupervised is harder to evaluate (no ground truth)
- Often used as preprocessing for supervised learning
- Semi-supervised: combines both (few labels + many unlabeled)

---

## Question 49
- [ ] Done

**Explain the concept of clustering and its applications.**

**Answer:**

### Definition
Clustering is an unsupervised learning technique that groups similar data points into clusters based on feature similarity. Points within a cluster are more similar to each other than to points in other clusters.

### Core Concepts
- **Cluster**: Group of similar data points
- **Centroid**: Center point of a cluster
- **Similarity/Distance**: Measure of how close points are
- **Intra-cluster distance**: Distance within cluster (minimize)
- **Inter-cluster distance**: Distance between clusters (maximize)

### Types of Clustering

| Type | Method | Example |
|------|--------|---------|
| Partitional | Divide into K clusters | K-Means |
| Hierarchical | Build tree of clusters | Agglomerative |
| Density-based | Clusters as dense regions | DBSCAN |
| Model-based | Probabilistic models | Gaussian Mixture |

### Applications

| Domain | Application |
|--------|-------------|
| Marketing | Customer segmentation |
| Biology | Gene expression grouping |
| Image | Image segmentation, compression |
| Social Media | Community detection |
| Retail | Product categorization |
| Anomaly Detection | Outliers as small clusters |

### Python Code Example
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Evaluate clustering quality
silhouette = silhouette_score(X, labels)
print(f"Silhouette Score: {silhouette:.3f}")  # -1 to 1, higher is better
```

### Evaluation Metrics
- **Silhouette Score**: Cohesion vs separation (-1 to 1)
- **Inertia**: Within-cluster sum of squares (lower is better)
- **Davies-Bouldin Index**: Ratio of within to between cluster distances

### Interview Tips
- No "correct" number of clusters; use elbow method or silhouette
- Scale features before distance-based clustering
- DBSCAN handles arbitrary shapes; K-Means assumes spherical

---

## Question 50
- [ ] Done

**What is the k-means clustering algorithm and how does it work?**

**Answer:**

### Definition
K-Means is a partitional clustering algorithm that divides data into K clusters by iteratively assigning points to the nearest centroid and updating centroids until convergence. It minimizes within-cluster variance.

### Algorithm Steps
1. **Initialize**: Randomly select K points as initial centroids
2. **Assign**: Assign each point to nearest centroid (Euclidean distance)
3. **Update**: Recalculate centroids as mean of assigned points
4. **Repeat**: Steps 2-3 until centroids don't change (convergence)

### Mathematical Formulation
**Objective** (minimize inertia):
$$J = \sum_{i=1}^{K}\sum_{x \in C_i} \|x - \mu_i\|^2$$

Where $\mu_i$ is centroid of cluster $C_i$

### Python Code Example
```python
from sklearn.cluster import KMeans
import numpy as np

# Fit K-Means
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
kmeans.fit(X)

# Results
labels = kmeans.labels_           # Cluster assignment for each point
centroids = kmeans.cluster_centers_  # Centroid coordinates
inertia = kmeans.inertia_         # Within-cluster sum of squares

# Predict new points
new_labels = kmeans.predict(X_new)
```

### K-Means++ Initialization
- Smart initialization to avoid poor local minima
- First centroid: random
- Next centroids: probability proportional to distance from existing centroids
- Results in faster convergence and better clusters

### Limitations
- Must specify K in advance
- Assumes spherical, equal-sized clusters
- Sensitive to outliers and initialization
- Only finds convex clusters

### Interview Tips
- Always use `init='k-means++'` (default in sklearn)
- Run multiple times (`n_init=10`) to avoid local minima
- Scale features before applying K-Means
- Use elbow method to find optimal K

---

## Question 51
- [ ] Done

**Describe the difference between hierarchical and partitional clustering.**

**Answer:**

### Definition
**Partitional Clustering**: Divides data into K non-overlapping clusters in one step (e.g., K-Means). **Hierarchical Clustering**: Builds a tree (dendrogram) of nested clusters, either bottom-up (agglomerative) or top-down (divisive).

### Comparison

| Aspect | Partitional | Hierarchical |
|--------|-------------|--------------|
| Output | K flat clusters | Tree (dendrogram) |
| K required | Yes, upfront | No, cut tree at any level |
| Flexibility | Fixed K | Multiple K from one run |
| Complexity | O(n × K × iterations) | O(n² log n) or O(n³) |
| Scalability | Better for large data | Expensive for large data |
| Reversible | No | No (greedy decisions) |

### Hierarchical Types
- **Agglomerative (Bottom-up)**: Start with n clusters, merge iteratively
- **Divisive (Top-down)**: Start with 1 cluster, split iteratively

### Linkage Methods (Agglomerative)

| Linkage | Distance Between Clusters |
|---------|---------------------------|
| Single | Minimum distance between points |
| Complete | Maximum distance between points |
| Average | Average distance between all pairs |
| Ward | Minimize variance increase |

### Python Code Example
```python
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Partitional: K-Means
kmeans = KMeans(n_clusters=3)
labels_km = kmeans.fit_predict(X)

# Hierarchical: Agglomerative
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_agg = agg.fit_predict(X)

# Visualize dendrogram
Z = linkage(X, method='ward')
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()
```

### When to Use

| Use Case | Best Choice |
|----------|-------------|
| Large dataset | Partitional (K-Means) |
| Explore different K | Hierarchical |
| Need dendrogram | Hierarchical |
| Arbitrary cluster shapes | DBSCAN |

### Interview Tips
- Hierarchical is deterministic; K-Means depends on initialization
- Ward linkage works best for compact clusters
- Single linkage can create "chaining" effect
- Cut dendrogram horizontally to get K clusters

---

## Question 52
- [ ] Done

**What is the elbow method and how is it used to determine the optimal number of clusters?**

**Answer:**

### Definition
The elbow method plots within-cluster sum of squares (inertia) against number of clusters (K). The optimal K is at the "elbow" point where adding more clusters yields diminishing returns in variance reduction.

### How It Works
1. Run K-Means for K = 1, 2, 3, ..., n
2. Calculate inertia (WCSS) for each K
3. Plot K vs Inertia
4. Find the "elbow" point where curve bends sharply
5. Choose K at that point

### Mathematical Concept
**Inertia (WCSS)** = $\sum_{i=1}^{K}\sum_{x \in C_i} \|x - \mu_i\|^2$

- K=1: Maximum inertia (all points far from single centroid)
- K=n: Zero inertia (each point is its own cluster)
- Elbow: Balance between too few and too many clusters

### Python Code Example
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Calculate inertia for different K
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method')
plt.show()

# Elbow appears around K=3 (example)
```

### Limitations
- Elbow may not be clear/obvious
- Subjective interpretation
- Doesn't work well for all data shapes

### Alternative Methods

| Method | Description |
|--------|-------------|
| Silhouette Score | Measures cluster quality (-1 to 1) |
| Gap Statistic | Compares to null reference distribution |
| Davies-Bouldin Index | Ratio of within to between distances |

### Interview Tips
- Elbow method is heuristic, not definitive
- Combine with silhouette score for confirmation
- Domain knowledge often guides final K choice
- If no clear elbow, data may not have distinct clusters

---

## Question 53
- [ ] Done

**Explain the concept of dimensionality reduction and its techniques.**

**Answer:**

### Definition
Dimensionality reduction is the process of reducing the number of features in a dataset while preserving as much important information as possible. It addresses the curse of dimensionality and improves model performance.

### Core Concepts
- **Curse of Dimensionality**: Performance degrades with too many features
- **Feature Space**: High-dimensional space where each feature is a dimension
- **Information Preservation**: Keep variance/structure while reducing dimensions
- **Computational Efficiency**: Fewer features = faster training

### Techniques

| Category | Technique | Description |
|----------|-----------|-------------|
| **Feature Selection** | Filter, Wrapper, Embedded | Select subset of original features |
| **Linear Projection** | PCA, LDA | Project to lower-dimensional space |
| **Non-linear** | t-SNE, UMAP | Preserve local structure for visualization |
| **Autoencoders** | Neural network | Learn compressed representation |

### When to Use
- High-dimensional data (images, text, genomics)
- Visualization of high-D data in 2D/3D
- Removing multicollinearity
- Noise reduction

### Interview Tip
PCA for general reduction, t-SNE/UMAP for visualization, LDA when class labels available.

---

## Question 54
- [ ] Done

**What is principal component analysis (PCA) and how does it work?**

**Answer:**

### Definition
PCA is a linear dimensionality reduction technique that transforms data into a new coordinate system where the greatest variance lies on the first principal component, second greatest on the second, and so on.

### Core Concepts
- **Principal Components**: New orthogonal axes that capture maximum variance
- **Eigenvalues**: Measure variance explained by each component
- **Eigenvectors**: Directions of principal components
- **Variance Explained**: Percentage of total variance captured

### Algorithm Steps
1. Standardize the data (mean=0, std=1)
2. Compute covariance matrix
3. Calculate eigenvalues and eigenvectors
4. Sort by eigenvalue (largest first)
5. Select top k components
6. Transform data: X_new = X × W_k

### Mathematical Formulation
$$C = \frac{1}{n-1}X^T X \quad \text{(Covariance matrix)}$$
$$Cv = \lambda v \quad \text{(Eigenvalue equation)}$$

### Practical Usage
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)
print(f"Variance explained: {pca.explained_variance_ratio_}")
```

### Interview Tip
Always standardize data before PCA. Choose components that explain ~95% variance.

---

## Question 55
- [ ] Done

**Describe the concept of t-SNE (t-Distributed Stochastic Neighbor Embedding).**

**Answer:**

### Definition
t-SNE is a non-linear dimensionality reduction technique that maps high-dimensional data to 2D/3D for visualization, preserving local neighborhood structure through probability distributions.

### Core Concepts
- **Local Structure Preservation**: Similar points stay close
- **Probability Distributions**: Models pairwise similarities
- **Student-t Distribution**: Handles crowding problem in low-D space
- **Perplexity**: Controls balance between local and global structure

### How It Works
1. Compute pairwise similarities in high-D space (Gaussian)
2. Define similar probabilities in low-D space (Student-t)
3. Minimize KL-divergence between distributions
4. Use gradient descent to optimize positions

### Key Parameters
| Parameter | Effect |
|-----------|--------|
| Perplexity (5-50) | Higher = more global structure |
| Learning rate | Too low = slow; too high = unstable |
| Iterations | Usually 1000+ for convergence |

### Limitations
- Computationally expensive (O(n²))
- Not suitable for new data projection
- Results vary with random initialization
- Distances in plot are not meaningful

### vs PCA
| PCA | t-SNE |
|-----|-------|
| Linear | Non-linear |
| Global structure | Local structure |
| Fast | Slow |
| Reproducible | Stochastic |

### Interview Tip
Use t-SNE for visualization only, not for preprocessing. Run multiple times with different seeds.

---

## Question 56
- [ ] Done

**What is the difference between PCA and LDA (Linear Discriminant Analysis)?**

**Answer:**

### Definition
PCA is unsupervised and maximizes variance regardless of class labels. LDA is supervised and maximizes class separability by finding axes that best separate classes.

### Core Comparison

| Aspect | PCA | LDA |
|--------|-----|-----|
| Type | Unsupervised | Supervised |
| Objective | Maximize variance | Maximize class separation |
| Input | Features only | Features + Labels |
| Output dims | Any k ≤ features | At most (classes - 1) |
| Use case | General dimensionality reduction | Classification preprocessing |

### Mathematical Objective
- **PCA**: Maximize $\text{Var}(X^T w)$
- **LDA**: Maximize $\frac{w^T S_B w}{w^T S_W w}$ (between-class / within-class variance)

### When to Use Each
| Scenario | Choose |
|----------|--------|
| No labels available | PCA |
| Classification task | LDA |
| Visualization | PCA or t-SNE |
| Class separation important | LDA |
| More than (c-1) components needed | PCA |

### Intuition
- **PCA**: "What directions have the most spread?"
- **LDA**: "What directions best separate my classes?"

### Interview Tip
LDA is limited to (num_classes - 1) components. For binary classification, LDA gives only 1 component.

---

## Question 57
- [ ] Done

**Explain the concept of anomaly detection and its techniques.**

**Answer:**

### Definition
Anomaly detection identifies rare observations that differ significantly from the majority of data. These outliers may indicate fraud, errors, or interesting phenomena worth investigating.

### Types of Anomalies
| Type | Description | Example |
|------|-------------|---------|
| Point | Single unusual data point | Unusually large transaction |
| Contextual | Abnormal in specific context | High AC usage in winter |
| Collective | Group of related anomalies | Coordinated attack pattern |

### Techniques

**Statistical Methods:**
- Z-score: Flag points > 3 standard deviations
- IQR: Points below Q1-1.5×IQR or above Q3+1.5×IQR

**Machine Learning Methods:**
| Method | Type | Description |
|--------|------|-------------|
| Isolation Forest | Unsupervised | Isolates anomalies with fewer splits |
| One-Class SVM | Unsupervised | Learns boundary around normal data |
| LOF | Unsupervised | Compares local density to neighbors |
| Autoencoders | Deep Learning | High reconstruction error = anomaly |

### Applications
- Credit card fraud detection
- Network intrusion detection
- Manufacturing defect detection
- Health monitoring

### Interview Tip
Choose method based on data: statistical for simple cases, Isolation Forest for high-dimensional, autoencoders for complex patterns.

---

## Question 58
- [ ] Done

**What is a neural network and how does it work?**

**Answer:**

### Definition
A neural network is a computing system inspired by biological neural networks, consisting of interconnected nodes (neurons) organized in layers that learn to transform inputs into outputs through training.

### Core Components
| Component | Function |
|-----------|----------|
| Input Layer | Receives raw features |
| Hidden Layers | Learn intermediate representations |
| Output Layer | Produces final prediction |
| Weights | Learnable connection strengths |
| Biases | Learnable offset terms |
| Activation Functions | Introduce non-linearity |

### How It Works
1. **Forward Pass**: Input flows through layers
   - Each neuron: $z = \sum w_i x_i + b$
   - Apply activation: $a = \sigma(z)$
2. **Loss Calculation**: Compare prediction to target
3. **Backward Pass**: Compute gradients via backpropagation
4. **Update Weights**: Gradient descent adjusts parameters

### Mathematical Formulation
$$y = \sigma(W_n \cdot \sigma(W_{n-1} \cdot ... \sigma(W_1 \cdot x + b_1) ... + b_{n-1}) + b_n)$$

### Types
- **Feedforward**: Information flows one direction
- **CNN**: For images (convolution layers)
- **RNN/LSTM**: For sequences (recurrent connections)
- **Transformer**: Attention-based (NLP, vision)

### Interview Tip
Understand the building blocks: layers, weights, activations, loss, and backpropagation.

---

## Question 59
- [ ] Done

**Explain the concept of activation functions and their types.**

**Answer:**

### Definition
Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Without them, stacked layers would just be linear transformations.

### Common Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | (0, 1) | Binary output, gates |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | (-1, 1) | Hidden layers (centered) |
| ReLU | $\max(0, x)$ | [0, ∞) | Most common in hidden layers |
| Leaky ReLU | $\max(0.01x, x)$ | (-∞, ∞) | Avoids dying ReLU |
| Softmax | $\frac{e^{x_i}}{\sum e^{x_j}}$ | (0, 1), sum=1 | Multi-class output |

### Problems and Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Vanishing gradient | Sigmoid/Tanh saturates | Use ReLU |
| Dying ReLU | Negative inputs → 0 gradient | Leaky ReLU, PReLU |
| Exploding gradient | Large activations | Batch normalization |

### Choosing Activation Functions
- **Hidden layers**: ReLU (default), Leaky ReLU
- **Binary classification output**: Sigmoid
- **Multi-class output**: Softmax
- **Regression output**: Linear (no activation)

### Interview Tip
ReLU is the default choice for hidden layers. Know why sigmoid causes vanishing gradients.

---

## Question 60
- [ ] Done

**What is the difference between a feedforward neural network and a recurrent neural network?**

**Answer:**

### Definition
Feedforward networks pass information in one direction (input→output). Recurrent networks have loops that allow information to persist, making them suitable for sequential data.

### Core Comparison

| Aspect | Feedforward (FFN) | Recurrent (RNN) |
|--------|-------------------|-----------------|
| Information flow | One direction | Cycles/loops |
| Memory | No memory | Has hidden state |
| Input | Fixed-size | Variable-length sequences |
| Use case | Tabular, images | Text, time series, speech |

### RNN Architecture
- **Hidden state**: $h_t = \sigma(W_h h_{t-1} + W_x x_t + b)$
- Shares weights across time steps
- Output depends on current input AND previous hidden state

### RNN Variants
| Variant | Improvement |
|---------|-------------|
| LSTM | Long-term memory with gates |
| GRU | Simplified LSTM, fewer parameters |
| Bidirectional | Processes sequence both directions |

### Limitations of Vanilla RNN
- Vanishing/exploding gradients
- Difficulty learning long-range dependencies
- Solution: LSTM/GRU with gating mechanisms

### Applications
- **FFN**: Classification, regression on fixed inputs
- **RNN**: Language models, machine translation, sentiment analysis, speech recognition

### Interview Tip
Know that LSTM/GRU solve RNN's gradient problems. Transformers have largely replaced RNNs for NLP.

---

## Question 61
- [ ] Done

**Describe the concept of backpropagation and its role in training neural networks.**

**Answer:**

### Definition
Backpropagation is the algorithm used to compute gradients of the loss function with respect to each weight by applying the chain rule, enabling gradient descent to update weights and minimize error.

### Core Concepts
- **Chain Rule**: Breaks complex derivatives into simpler parts
- **Gradient Flow**: Error signals flow backward through network
- **Weight Updates**: Proportional to how much each weight contributed to error
- **Learning Rate**: Controls step size of weight updates

### Algorithm Steps
1. **Forward Pass**: Compute predictions layer by layer
2. **Compute Loss**: Compare predictions to targets
3. **Backward Pass**: Compute gradients from output to input
4. **Update Weights**: $w = w - \eta \cdot \frac{\partial L}{\partial w}$

### Mathematical Formulation
For a simple network with loss L:
$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial h} \cdot \frac{\partial h}{\partial w_1}$$

### Key Points
- Requires differentiable activation functions
- Computational graph stores intermediate values
- Efficient: computes all gradients in one backward pass
- Foundation of modern deep learning

### Interview Tip
Understand chain rule application and how gradients flow backward through layers.

---

## Question 62
- [ ] Done

**What is deep learning and how does it differ from traditional machine learning?**

**Answer:**

### Definition
Deep learning is a subset of machine learning using neural networks with many layers (deep architectures) that automatically learn hierarchical feature representations from raw data.

### Core Comparison

| Aspect | Traditional ML | Deep Learning |
|--------|---------------|---------------|
| Feature Engineering | Manual, domain expertise | Automatic, learned |
| Data Requirements | Works with less data | Needs large datasets |
| Compute | CPU-friendly | GPU-intensive |
| Interpretability | Often interpretable | Often black-box |
| Performance | Good on structured data | Excels on unstructured data |

### What Makes It "Deep"
- Multiple hidden layers (from 3 to 100+)
- Each layer learns increasingly abstract features
- Example (image): edges → shapes → parts → objects

### When to Use Deep Learning
| Use DL | Use Traditional ML |
|--------|-------------------|
| Images, video | Tabular data |
| Text, speech | Small datasets |
| Abundant data | Need interpretability |
| Complex patterns | Simple patterns |

### Popular Architectures
- **CNNs**: Computer vision
- **RNNs/LSTMs**: Sequences
- **Transformers**: NLP, vision
- **GANs**: Generative tasks

### Interview Tip
Deep learning excels with large unstructured data; traditional ML is often better for small tabular datasets.

---

## Question 63
- [ ] Done

**Explain the concept of convolutional neural networks (CNNs) and their applications.**

**Answer:**

### Definition
CNNs are specialized neural networks for processing grid-like data (images) using convolutional layers that automatically learn spatial hierarchies of features through learnable filters.

### Core Components
| Component | Function |
|-----------|----------|
| Convolutional Layer | Learns local patterns with filters |
| Pooling Layer | Reduces spatial dimensions, adds invariance |
| Stride | Filter movement step size |
| Padding | Preserves spatial dimensions |
| Fully Connected | Final classification layers |

### How Convolution Works
- Filter (kernel) slides across image
- Element-wise multiplication and sum
- Produces feature map highlighting detected patterns
- Different filters detect different features (edges, textures)

### Typical Architecture
```
Input → [Conv → ReLU → Pool] × N → Flatten → FC → Softmax → Output
```

### Applications
- Image classification (ImageNet)
- Object detection (YOLO, R-CNN)
- Facial recognition
- Medical image analysis
- Self-driving cars

### Interview Tip
Know the intuition: early layers detect simple features (edges), deeper layers detect complex features (faces).

---

## Question 64
- [ ] Done

**What is transfer learning and how is it used in deep learning?**

**Answer:**

### Definition
Transfer learning uses a model pre-trained on one task as the starting point for a different but related task, leveraging learned representations to reduce training time and data requirements.

### Core Concepts
- **Pre-trained Model**: Trained on large dataset (ImageNet, BERT)
- **Feature Extraction**: Freeze pre-trained layers, train only new layers
- **Fine-tuning**: Unfreeze some/all layers, train with small learning rate
- **Domain Adaptation**: Adjust to new domain

### Why It Works
- Low-level features (edges, textures) are universal
- Reduces need for large labeled datasets
- Faster training
- Better generalization

### Common Approaches
| Approach | Description | When to Use |
|----------|-------------|-------------|
| Feature Extraction | Freeze all, add new head | Very small dataset |
| Fine-tuning (partial) | Unfreeze top layers | Medium dataset |
| Fine-tuning (full) | Unfreeze all layers | Large dataset |

### Popular Pre-trained Models
| Domain | Models |
|--------|--------|
| Vision | ResNet, VGG, EfficientNet |
| NLP | BERT, GPT, RoBERTa |
| Speech | Wav2Vec |

### Interview Tip
Start with feature extraction, then try fine-tuning if performance is insufficient.

---

## Question 65
- [ ] Done

**Describe the concept of long short-term memory (LSTM) networks and their use cases.**

**Answer:**

### Definition
LSTM is a special type of RNN designed to learn long-range dependencies by using gating mechanisms that control information flow, solving the vanishing gradient problem.

### Core Components (Gates)
| Gate | Function | Formula |
|------|----------|---------|
| Forget Gate | Decides what to discard | $f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)$ |
| Input Gate | Decides what to store | $i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)$ |
| Output Gate | Decides what to output | $o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)$ |
| Cell State | Long-term memory highway | $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$ |

### Why LSTM Solves Vanishing Gradient
- Cell state acts as a "highway" for gradients
- Gates allow gradients to flow unchanged
- Can learn dependencies over 100s of time steps

### Use Cases
- **Language Modeling**: Predict next word
- **Machine Translation**: Seq-to-seq models
- **Speech Recognition**: Audio to text
- **Time Series**: Stock prediction, weather
- **Sentiment Analysis**: Review classification

### LSTM vs GRU
| LSTM | GRU |
|------|-----|
| 3 gates | 2 gates |
| More parameters | Fewer parameters |
| Better long-term | Faster training |

### Interview Tip
Know the purpose of each gate: forget (what to remove), input (what to add), output (what to expose).

---

## Question 66
- [ ] Done

**What is the difference between a shallow neural network and a deep neural network?**

**Answer:**

### Definition
A shallow neural network has 1-2 hidden layers, while a deep neural network has many hidden layers (3+), enabling it to learn more complex hierarchical representations.

### Core Comparison

| Aspect | Shallow NN | Deep NN |
|--------|-----------|---------|
| Hidden Layers | 1-2 | 3+ (can be 100+) |
| Feature Learning | Limited abstraction | Hierarchical features |
| Parameters | Fewer | Many more |
| Training | Easier, faster | Harder, needs tricks |
| Representational Power | Universal approx (theoretically) | Better in practice |

### Why Depth Matters
- Deeper = exponentially more expressive
- Each layer builds on previous representations
- Example: pixels → edges → shapes → parts → objects

### Challenges of Depth
| Challenge | Solution |
|-----------|----------|
| Vanishing gradients | ReLU, batch norm, residual connections |
| Overfitting | Dropout, regularization |
| Long training | GPU, better optimizers (Adam) |
| Many hyperparameters | AutoML, transfer learning |

### Universal Approximation Theorem
- Shallow network CAN approximate any function
- But may need exponentially many neurons
- Depth achieves same with fewer parameters

### Interview Tip
Depth enables efficient representation of complex functions that shallow networks require exponentially more neurons to match.

---

## Question 67
- [ ] Done

**Explain the concept of autoencoders and their applications.**

**Answer:**

### Definition
An autoencoder is a neural network that learns to compress data into a lower-dimensional representation (encoding) and then reconstruct the original input from that encoding.

### Architecture
```
Input → Encoder → Bottleneck (latent code) → Decoder → Reconstructed Output
```

### Core Concepts
- **Encoder**: Maps input to compressed representation
- **Decoder**: Maps compressed representation back to input
- **Bottleneck**: Forces network to learn essential features
- **Loss**: Reconstruction error (MSE or BCE)

### Types of Autoencoders
| Type | Description | Use Case |
|------|-------------|----------|
| Vanilla | Basic encoder-decoder | Dimensionality reduction |
| Sparse | Penalizes active neurons | Feature learning |
| Denoising | Trained on corrupted input | Noise removal |
| Variational (VAE) | Learns probability distribution | Generation |

### Applications
- **Dimensionality Reduction**: Alternative to PCA
- **Anomaly Detection**: High reconstruction error = anomaly
- **Denoising**: Remove noise from images
- **Feature Extraction**: Use bottleneck as features
- **Generative Models**: VAEs generate new samples

### Interview Tip
Key insight: autoencoders are unsupervised. They learn compressed representations without labels.

---

## Question 68
- [ ] Done

**What is natural language processing (NLP) and its applications?**

**Answer:**

### Definition
NLP is a field of AI that enables computers to understand, interpret, and generate human language in a meaningful way, bridging human communication and machine understanding.

### Core Tasks
| Task | Description |
|------|-------------|
| Tokenization | Split text into tokens |
| POS Tagging | Identify parts of speech |
| NER | Extract named entities |
| Parsing | Analyze grammatical structure |
| Sentiment Analysis | Determine opinion/emotion |
| Machine Translation | Translate between languages |

### Applications
- **Chatbots/Virtual Assistants**: Siri, Alexa, customer service
- **Search Engines**: Query understanding, ranking
- **Sentiment Analysis**: Social media monitoring, reviews
- **Machine Translation**: Google Translate
- **Text Summarization**: News, documents
- **Question Answering**: FAQ bots, search
- **Content Generation**: Writing assistance, code generation

### NLP Pipeline
```
Raw Text → Tokenization → Normalization → Feature Extraction → Model → Output
```

### Evolution
- Rule-based → Statistical → Deep Learning → Transformers (BERT, GPT)

### Interview Tip
Modern NLP is dominated by transformer-based models (BERT, GPT) that achieve state-of-the-art on most tasks.

---

## Question 69
- [ ] Done

**Explain the concept of tokenization and its techniques.**

**Answer:**

### Definition
Tokenization is the process of breaking text into smaller units (tokens) such as words, subwords, or characters, which serve as input to NLP models.

### Types of Tokenization
| Type | Example | Use Case |
|------|---------|----------|
| Word | "Hello world" → ["Hello", "world"] | Traditional NLP |
| Character | "Hi" → ["H", "i"] | Character-level models |
| Subword | "unhappiness" → ["un", "happiness"] | Modern transformers |
| Sentence | Split by sentences | Document processing |

### Subword Tokenization Methods
| Method | Description |
|--------|-------------|
| BPE (Byte Pair Encoding) | Merges frequent character pairs |
| WordPiece | Similar to BPE, used in BERT |
| SentencePiece | Language-agnostic, handles raw text |
| Unigram | Probabilistic, keeps most likely tokens |

### Challenges
- Handling unknown words (OOV)
- Preserving meaning
- Language-specific rules
- Handling punctuation, special characters

### Python Example
```python
# Simple word tokenization
text = "Hello, world!"
tokens = text.split()  # ['Hello,', 'world!']

# Using NLTK
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)  # ['Hello', ',', 'world', '!']
```

### Interview Tip
Subword tokenization (BPE, WordPiece) is preferred in modern transformers because it handles OOV and keeps vocabulary manageable.

---

## Question 70
- [ ] Done

**What is stemming and lemmatization in NLP?**

**Answer:**

### Definition
Both reduce words to their base form. Stemming uses rule-based heuristics (often crude), while lemmatization uses vocabulary and morphological analysis to return proper dictionary forms.

### Comparison

| Aspect | Stemming | Lemmatization |
|--------|----------|---------------|
| Method | Rule-based heuristics | Dictionary/morphology |
| Output | Word stem (may not be real word) | Lemma (valid dictionary word) |
| Speed | Faster | Slower |
| Accuracy | Lower | Higher |
| Example | "running" → "run" or "runn" | "running" → "run" |
| Example | "better" → "better" | "better" → "good" |

### Popular Algorithms
**Stemming:**
- Porter Stemmer (most common)
- Snowball Stemmer
- Lancaster Stemmer (aggressive)

**Lemmatization:**
- WordNet Lemmatizer
- spaCy Lemmatizer

### Python Example
```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

word = "studies"
print(stemmer.stem(word))        # studi
print(lemmatizer.lemmatize(word, pos='v'))  # study
```

### When to Use
- **Stemming**: Speed matters, search engines, simple tasks
- **Lemmatization**: Accuracy matters, understanding context

### Interview Tip
Lemmatization needs POS tags for best results. "better" as adj → "good"; as adv → "well".

---

## Question 71
- [ ] Done

**Describe the concept of word embeddings and their types (e.g., Word2Vec, GloVe).**

**Answer:**

### Definition
Word embeddings are dense vector representations of words where semantically similar words have similar vectors, capturing meaning in a continuous space.

### Why Embeddings?
- One-hot encoding: Sparse, no semantic meaning
- Embeddings: Dense, capture relationships
- "King - Man + Woman ≈ Queen"

### Popular Methods

| Method | Type | Training |
|--------|------|----------|
| Word2Vec | Predictive | Predict context/word from context |
| GloVe | Count-based | Matrix factorization of co-occurrences |
| FastText | Predictive | Uses subword information |
| ELMo | Contextual | Bidirectional LSTM |
| BERT | Contextual | Transformer, masked LM |

### Word2Vec Architectures
- **CBOW**: Predict word from context
- **Skip-gram**: Predict context from word

### Properties
- Similar words have similar vectors (cosine similarity)
- Captures analogies: man:woman :: king:queen
- Typical dimensions: 100-300

### Contextual vs Static
| Static (Word2Vec, GloVe) | Contextual (BERT, ELMo) |
|--------------------------|-------------------------|
| One vector per word | Different vector per context |
| "bank" has one meaning | "bank" differs in "river bank" vs "bank account" |

### Interview Tip
Static embeddings are simpler but contextual embeddings (BERT) handle polysemy and are state-of-the-art.

---

## Question 72
- [ ] Done

**What is the bag-of-words model and how is it used in NLP?**

**Answer:**

### Definition
Bag-of-words (BoW) represents text as an unordered collection of words, creating a vector where each dimension corresponds to a word in the vocabulary and values indicate word presence or frequency.

### How It Works
1. Build vocabulary from all documents
2. For each document, count word occurrences
3. Create vector of counts or binary indicators

### Example
```
Doc1: "I love cats"
Doc2: "I love dogs"

Vocabulary: [I, love, cats, dogs]
Doc1 vector: [1, 1, 1, 0]
Doc2 vector: [1, 1, 0, 1]
```

### Variations
| Variation | Values |
|-----------|--------|
| Binary BoW | 0 or 1 (presence) |
| Count BoW | Word frequency |
| TF-IDF | Term frequency × Inverse document frequency |

### Limitations
- Ignores word order ("dog bites man" = "man bites dog")
- High dimensionality (vocabulary size)
- Sparse vectors
- No semantic meaning

### Applications
- Text classification (spam, sentiment)
- Document similarity
- Simple search engines

### Interview Tip
BoW is a baseline. For better results, use TF-IDF or word embeddings to capture semantics.

---

## Question 73
- [ ] Done

**Explain the concept of named entity recognition (NER) and its techniques.**

**Answer:**

### Definition
NER is the task of identifying and classifying named entities (people, organizations, locations, dates, etc.) in text into predefined categories.

### Common Entity Types
| Type | Abbreviation | Example |
|------|--------------|---------|
| Person | PER | "Albert Einstein" |
| Organization | ORG | "Google" |
| Location | LOC | "New York" |
| Date | DATE | "January 2024" |
| Money | MONEY | "$100" |
| Percentage | PERCENT | "50%" |

### Approaches

| Approach | Method | Example |
|----------|--------|---------|
| Rule-based | Regular expressions, dictionaries | Phone number patterns |
| Classical ML | CRF, HMM with hand-crafted features | POS, capitalization features |
| Deep Learning | BiLSTM-CRF, CNN | Learned features |
| Transformers | BERT, RoBERTa fine-tuned | State-of-the-art |

### BIO Tagging Scheme
```
"Barack Obama was born in Hawaii"
 B-PER  I-PER  O    O    O  B-LOC
```
- B = Beginning of entity
- I = Inside entity
- O = Outside any entity

### Applications
- Information extraction
- Question answering
- Knowledge graph construction
- Document summarization

### Interview Tip
Modern NER uses transformer models fine-tuned on labeled data. Know BIO tagging scheme.

---

## Question 74
- [ ] Done

**What is sentiment analysis and how is it performed?**

**Answer:**

### Definition
Sentiment analysis determines the emotional tone (positive, negative, neutral) or opinion expressed in text, used to understand attitudes and opinions at scale.

### Types
| Type | Output | Example |
|------|--------|---------|
| Binary | Positive/Negative | Review classification |
| Multi-class | Positive/Neutral/Negative | Social media |
| Fine-grained | 1-5 stars | Product ratings |
| Aspect-based | Sentiment per aspect | "Battery good, screen bad" |

### Approaches

| Approach | Method | Pros/Cons |
|----------|--------|-----------|
| Lexicon-based | Word sentiment scores | Simple, no training; domain-specific issues |
| ML (Traditional) | BoW + SVM/Naive Bayes | Needs labeled data |
| Deep Learning | LSTM, CNN | Better accuracy |
| Transformers | BERT, RoBERTa | State-of-the-art |

### Challenges
- Sarcasm and irony: "Great, another Monday!"
- Negation: "not good" vs "good"
- Context dependency
- Domain-specific language

### Python Example
```python
from textblob import TextBlob

text = "I love this product! It's amazing."
sentiment = TextBlob(text).sentiment
print(f"Polarity: {sentiment.polarity}")  # 0.625 (positive)
```

### Applications
- Social media monitoring
- Customer feedback analysis
- Brand reputation
- Market research

### Interview Tip
Know the challenges (sarcasm, negation). Transformer-based models handle these better than simple methods.

---

## Question 75
- [ ] Done

**Describe the concept of topic modeling and its algorithms (e.g., LDA, NMF).**

**Answer:**

### Definition
Topic modeling automatically discovers abstract "topics" in a collection of documents, where each topic is a distribution over words and each document is a mixture of topics.

### Intuition
- Document = mixture of topics
- Topic = mixture of words
- "Sports article" might be 70% sports topic, 20% business, 10% politics

### Main Algorithms

**LDA (Latent Dirichlet Allocation):**
- Probabilistic generative model
- Assumes Dirichlet prior on topic distributions
- Most popular algorithm
- Hyperparameters: number of topics, alpha, beta

**NMF (Non-negative Matrix Factorization):**
- Matrix factorization approach
- Decomposes document-term matrix
- Faster than LDA
- Good for short texts

### LDA Output Example
```
Topic 1: game, team, player, score, win (Sports)
Topic 2: stock, market, company, price (Finance)
Topic 3: president, election, vote, policy (Politics)
```

### Python Example
```python
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(document_term_matrix)

# Get topics
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-10:]]
    print(f"Topic {topic_idx}: {top_words}")
```

### Applications
- Document organization
- Content recommendation
- Trend analysis
- Information retrieval

### Interview Tip
LDA is generative and probabilistic; NMF is deterministic and faster. Choose based on needs.

## Question 76
- [ ] Done

**What is the difference between rule-based and statistical approaches in NLP?**

**Answer:**

### Definition
Rule-based NLP uses hand-crafted linguistic rules and patterns, while statistical approaches learn patterns automatically from data using probability and machine learning.

### Comparison

| Aspect | Rule-Based | Statistical |
|--------|-----------|-------------|
| Knowledge Source | Human experts | Data |
| Flexibility | Rigid | Adaptive |
| Development Time | High (manual) | Lower (automated) |
| Coverage | Limited to rules | Generalizes beyond training |
| Interpretability | High | Often low |
| Maintenance | Labor-intensive | Retrain with data |

### Examples
**Rule-Based:**
- Regular expressions for email extraction
- Grammar rules for parsing
- Dictionary-based sentiment

**Statistical:**
- Machine learning classifiers
- Neural language models
- Word embeddings

### Hybrid Approaches
- Use rules for preprocessing
- Statistical models for core task
- Rules for post-processing edge cases

### Interview Tip
Modern NLP is mostly statistical/neural, but rules are valuable for specific patterns and interpretability.

---

## Question 77
- [ ] Done

**Explain the concept of language models and their applications.**

**Answer:**

### Definition
A language model is a probabilistic model that assigns probabilities to sequences of words, learning the likelihood of word sequences in a language.

### Core Concept
$$P(\text{sentence}) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1,w_2) \cdot ...$$

### Types
| Type | Method | Example |
|------|--------|---------|
| N-gram | Count-based | Bigram, trigram |
| Neural | RNN, LSTM | Character-level models |
| Transformer | Self-attention | GPT, BERT |

### Applications
- **Text Generation**: Write articles, code, stories
- **Autocomplete**: Email, search suggestions
- **Machine Translation**: Translate between languages
- **Speech Recognition**: Convert audio to text
- **Chatbots**: Conversational AI
- **Summarization**: Condense documents

### Modern LLMs
- GPT-4: Generative, autoregressive
- BERT: Bidirectional, masked LM
- T5: Text-to-text framework

### Interview Tip
Understand the difference: GPT predicts next word (autoregressive); BERT predicts masked words (bidirectional).

---

## Question 78
- [ ] Done

**What is a recommender system and its types?**

**Answer:**

### Definition
A recommender system predicts user preferences and suggests items (products, movies, content) that users are likely to find relevant or interesting.

### Main Types

| Type | Method | Example |
|------|--------|---------|
| Collaborative Filtering | User-item interactions | "Users like you also bought..." |
| Content-Based | Item features | "Similar to movies you watched..." |
| Hybrid | Combination | Netflix, Amazon |
| Knowledge-Based | Expert rules | Technical products with constraints |

### Collaborative Filtering Variants
- **User-Based**: Find similar users, recommend their items
- **Item-Based**: Find similar items to what user liked
- **Matrix Factorization**: Latent factor models (SVD)

### Content-Based Features
- Item attributes (genre, keywords)
- User profiles built from past interactions
- No need for other users' data

### Applications
- E-commerce (Amazon)
- Streaming (Netflix, Spotify)
- Social media (Facebook, TikTok)
- News and content platforms

### Interview Tip
Know the trade-offs: collaborative needs user data but discovers new interests; content-based works for new users but limits discovery.

---

## Question 79
- [ ] Done

**Explain the concept of collaborative filtering and its variants.**

**Answer:**

### Definition
Collaborative filtering recommends items based on collective user behavior, assuming users with similar past preferences will like similar items in the future.

### Core Idea
"People who agreed in the past will agree in the future"

### Variants

**1. User-Based CF:**
- Find users similar to target user
- Recommend items those users liked
- Similarity: cosine, Pearson correlation

**2. Item-Based CF:**
- Find items similar to user's liked items
- More stable than user-based
- Amazon's "customers who bought this also bought..."

**3. Model-Based CF:**
- Matrix factorization (SVD, ALS)
- Learn latent factors for users and items
- More scalable

### User-Item Matrix
```
        Item1  Item2  Item3
User1   5      3      ?
User2   4      ?      2
User3   ?      5      4
```
Goal: Predict missing ratings

### Pros and Cons
| Pros | Cons |
|------|------|
| Discovers new interests | Cold start problem |
| No item features needed | Sparsity issues |
| Domain-independent | Scalability challenges |

### Interview Tip
Item-based CF is more practical for large systems (items change less than users).

---

## Question 80
- [ ] Done

**What is content-based filtering and how does it work?**

**Answer:**

### Definition
Content-based filtering recommends items similar to those a user has previously liked, based on item features rather than other users' behavior.

### How It Works
1. Extract item features (genre, keywords, attributes)
2. Build user profile from liked item features
3. Match user profile against candidate items
4. Rank by similarity

### User Profile Construction
```
User likes: Action(5), Comedy(3), Romance(1)
Profile = weighted average of liked item features
```

### Similarity Measures
- Cosine similarity
- TF-IDF weighted
- Jaccard similarity (for sets)

### Pros and Cons
| Pros | Cons |
|------|------|
| Works for new users (no cold start for items) | Over-specialization (filter bubble) |
| Transparent recommendations | Needs good item features |
| No other users needed | Limited discovery |
| Easy to explain | Can't find unexpected interests |

### Applications
- News articles (topic matching)
- Music (audio features)
- Movies (genre, actors, director)

### Interview Tip
Content-based is great for explainability but risks creating a "filter bubble" where users never discover new interests.

---

## Question 81
- [ ] Done

**Describe the concept of matrix factorization in recommender systems.**

**Answer:**

### Definition
Matrix factorization decomposes the user-item rating matrix into lower-dimensional user and item latent factor matrices, capturing hidden features that explain ratings.

### Mathematical Formulation
$$R \approx U \times V^T$$

Where:
- R: User-item rating matrix (m × n)
- U: User latent factors (m × k)
- V: Item latent factors (n × k)
- k: Number of latent factors

### How It Works
- Latent factors capture abstract concepts (genre, mood)
- User vector: user's affinity for each factor
- Item vector: item's association with each factor
- Predicted rating: dot product of vectors

### Algorithms
| Algorithm | Description |
|-----------|-------------|
| SVD | Singular Value Decomposition |
| ALS | Alternating Least Squares |
| NMF | Non-negative Matrix Factorization |
| Funk SVD | Gradient descent optimization |

### Objective Function
$$\min_{U,V} \sum_{(i,j) \in \text{observed}} (r_{ij} - u_i^T v_j)^2 + \lambda(||U||^2 + ||V||^2)$$

### Advantages
- Handles sparsity well
- Captures complex patterns
- Scalable
- Implicit regularization

### Interview Tip
Matrix factorization won Netflix Prize. Understand latent factors conceptually (e.g., factors might capture "action-oriented" or "family-friendly").

---

## Question 82
- [ ] Done

**What are the challenges and limitations of recommender systems?**

**Answer:**

### Key Challenges

| Challenge | Description |
|-----------|-------------|
| Cold Start | New users/items have no interaction history |
| Sparsity | Most ratings are missing |
| Scalability | Millions of users and items |
| Diversity | Avoiding repetitive recommendations |
| Serendipity | Enabling unexpected discoveries |
| Filter Bubble | Users trapped in narrow content |

### Technical Limitations
- **Data Quality**: Noisy, biased, or fake ratings
- **Changing Preferences**: User tastes evolve
- **Context**: Same user wants different things at different times
- **Explainability**: Why this recommendation?

### Ethical Issues
- Privacy concerns
- Manipulation potential
- Amplifying biases
- Polarization (especially news)

### Solutions
| Problem | Solution |
|---------|----------|
| Cold Start | Hybrid systems, content-based fallback |
| Sparsity | Matrix factorization, deep learning |
| Diversity | Re-ranking, exploration |
| Filter Bubble | Diversity metrics, random exploration |

### Interview Tip
Know cold start and sparsity problems well. Common follow-up: "How would you solve cold start for new users?"

---

## Question 83
- [ ] Done

**Explain the concept of cold-start problem in recommender systems and its solutions.**

**Answer:**

### Definition
The cold-start problem occurs when the system lacks sufficient data to make recommendations for new users, new items, or new systems.

### Types
| Type | Problem | Example |
|------|---------|---------|
| New User | No interaction history | Just registered account |
| New Item | No ratings received | Newly released movie |
| New System | No data at all | Launching new platform |

### Solutions

**For New Users:**
- Ask for preferences during onboarding
- Use demographic info
- Recommend popular items initially
- Content-based on limited signals

**For New Items:**
- Use content-based features
- Promote to diverse users for initial feedback
- Bandit algorithms (exploration)

**For New Systems:**
- Import data from similar platforms
- Use knowledge-based rules
- Leverage external data (IMDb, reviews)

### Hybrid Approaches
- Combine collaborative + content-based
- Content handles cold start
- Collaborative improves with data

### Interview Tip
A good answer mentions both user and item cold start, with specific solutions for each.

---

## Question 84
- [ ] Done

**What is the difference between explicit and implicit feedback in recommender systems?**

**Answer:**

### Definition
Explicit feedback is direct user ratings (1-5 stars), while implicit feedback is inferred from user behavior (clicks, views, purchases).

### Comparison

| Aspect | Explicit | Implicit |
|--------|----------|----------|
| Collection | User actively provides | Passively observed |
| Example | Star ratings, likes | Clicks, watch time, purchases |
| Availability | Sparse (few rate) | Abundant |
| Reliability | Clear preference signal | Noisy (click ≠ like) |
| Negative Signal | Low ratings | Hard to determine (no click = dislike or unseen?) |

### Challenges with Implicit
- No negative examples (missing ≠ dislike)
- Noisy signals (accidental clicks)
- Varying engagement levels
- Position bias

### Handling Implicit Data
- Treat as positive/unlabeled (not positive/negative)
- Weight by engagement (watch time, purchases)
- Use one-class collaborative filtering
- BPR (Bayesian Personalized Ranking)

### Applications
| Explicit | Implicit |
|----------|----------|
| Movie ratings | View history |
| Product reviews | Click stream |
| Thumbs up/down | Purchase history |

### Interview Tip
Most real-world systems rely heavily on implicit feedback because explicit ratings are sparse.

---

## Question 85
- [ ] Done

**Describe the concept of evaluation metrics for recommender systems (e.g., precision, recall, NDCG).**

**Answer:**

### Ranking Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| Precision@K | Relevant items in top K / K | $\frac{|relevant \cap top_K|}{K}$ |
| Recall@K | Relevant items in top K / Total relevant | $\frac{|relevant \cap top_K|}{|relevant|}$ |
| MAP | Mean Average Precision across users | |
| NDCG | Normalized Discounted Cumulative Gain | |
| MRR | Mean Reciprocal Rank | |

### NDCG Explained
- DCG: Rewards relevant items appearing earlier
- NDCG: Normalized to [0, 1]
$$DCG_K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}$$

### Rating Prediction Metrics
| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |

### Beyond Accuracy
| Metric | Measures |
|--------|----------|
| Coverage | % of items recommended |
| Diversity | How different recommendations are |
| Novelty | How unexpected/new items are |
| Serendipity | Surprising yet relevant |

### Interview Tip
NDCG is preferred for ranking because it accounts for position. Precision@K is simpler but treats all positions equally.

---

## Question 86
- [ ] Done

**What is the role of user-item interactions in recommender systems?**

**Answer:**

### Definition
User-item interactions are the signals (explicit or implicit) that capture how users engage with items, forming the foundation for building recommendations.

### Types of Interactions

| Type | Signal Strength | Example |
|------|-----------------|---------|
| Purchase | Strong | Bought product |
| Rating | Strong | 5-star review |
| Long view | Medium | Watched full movie |
| Click | Weak | Clicked on item |
| Impression | Very weak | Item shown but not clicked |

### Interaction Matrix
```
        Item1  Item2  Item3  Item4
User1   Buy    Click  -      View
User2   -      Buy    Rate   -
User3   View   -      Click  Buy
```

### Role in Algorithms
- **Collaborative Filtering**: Core input
- **Matrix Factorization**: Entries to factorize
- **Deep Learning**: Training examples
- **Session-Based**: Sequential patterns

### Data Collection
- Server-side logging
- Client-side events
- A/B testing infrastructure

### Interview Tip
The quality and quantity of interactions directly impact recommendation quality. More interactions = better personalization.

---

## Question 87
- [ ] Done

**Explain the concept of hybrid recommender systems and their advantages.**

**Answer:**

### Definition
Hybrid recommender systems combine multiple recommendation techniques (e.g., collaborative + content-based) to overcome individual method limitations.

### Combination Strategies

| Strategy | Method |
|----------|--------|
| Weighted | Score = w₁ × CF + w₂ × Content |
| Switching | Use CF when possible, else content |
| Feature Combination | Combine features in one model |
| Cascade | First coarse filter, then fine ranking |
| Meta-level | One method's output feeds another |

### Advantages
| Benefit | Explanation |
|---------|-------------|
| Overcome cold start | Content handles new users/items |
| Better accuracy | Leverage strengths of each method |
| Increased coverage | More items can be recommended |
| Robustness | Less sensitive to individual method failures |

### Example: Netflix
- Collaborative filtering for personalization
- Content features for new shows
- Contextual signals (time, device)
- Trending/popular as fallback

### Implementation Example
```python
score = 0.6 * collaborative_score + 0.3 * content_score + 0.1 * popularity
```

### Interview Tip
Most production systems are hybrid. Know specific examples like Netflix or Spotify's approach.

---

## Question 88
- [ ] Done

**What is the role of optimization in machine learning?**

**Answer:**

### Definition
Optimization in ML is the process of finding model parameters that minimize (or maximize) an objective function, typically the loss/error between predictions and actual values.

### Core Concept
$$\theta^* = \arg\min_\theta L(\theta; X, y)$$

Where:
- θ: Model parameters (weights, biases)
- L: Loss function
- X, y: Training data

### Why Optimization Matters
- Models start with random parameters
- Training = iteratively improving parameters
- Goal: Find parameters that generalize well

### Key Components
| Component | Role |
|-----------|------|
| Objective Function | What to minimize (loss) |
| Parameters | Variables to optimize |
| Algorithm | How to search (gradient descent) |
| Constraints | Regularization, bounds |

### Common Loss Functions
| Task | Loss |
|------|------|
| Regression | MSE, MAE |
| Classification | Cross-entropy |
| Ranking | Hinge, triplet loss |

### Challenges
- Non-convex landscapes (local minima)
- Saddle points
- Overfitting
- Computational cost

### Interview Tip
Optimization is how ML models learn. Without it, we'd have no training.

---

## Question 89
- [ ] Done

**Explain the concept of gradient descent and its variants (e.g., batch, stochastic, mini-batch).**

**Answer:**

### Definition
Gradient descent is an iterative optimization algorithm that updates parameters in the direction of steepest descent of the loss function.

### Update Rule
$$\theta_{t+1} = \theta_t - \eta \cdot \nabla L(\theta_t)$$

Where:
- η: Learning rate
- ∇L: Gradient of loss

### Variants

| Variant | Data per Update | Pros | Cons |
|---------|-----------------|------|------|
| Batch (Full) | All training data | Stable, accurate gradients | Slow, memory-intensive |
| Stochastic (SGD) | 1 sample | Fast, can escape minima | Noisy, unstable |
| Mini-batch | Subset (32-256) | Balance of speed and stability | Common choice |

### Advanced Optimizers
| Optimizer | Improvement |
|-----------|-------------|
| Momentum | Accumulates velocity |
| RMSprop | Adaptive learning rates |
| Adam | Momentum + RMSprop |
| AdamW | Adam + weight decay |

### Mini-batch Advantages
- Utilizes GPU parallelism
- Regularization effect from noise
- Faster convergence than batch

### Interview Tip
Adam is the default choice. Know when to use SGD with momentum (often better generalization).

---

## Question 90
- [ ] Done

**What is the difference between a local minimum and a global minimum?**

**Answer:**

### Definition
A global minimum is the lowest point across the entire function. A local minimum is the lowest point within a neighborhood but not necessarily the overall lowest.

### Visual Intuition
```
Loss     *  (local min)
  |    /  \
  |   /    \___*  (global min)
  |  /
  | /
  +----------------------------> Parameters
```

### In Neural Networks
- Loss landscapes are highly non-convex
- Many local minima exist
- Modern insight: Most local minima are nearly as good as global
- Saddle points are bigger concern than local minima

### Escaping Local Minima
| Technique | How |
|-----------|-----|
| Momentum | Builds velocity to pass through |
| Learning rate schedules | Vary step size |
| Random restarts | Multiple initializations |
| Noise (SGD) | Stochasticity helps escape |
| Simulated annealing | Controlled randomness |

### Convex vs Non-Convex
| Convex | Non-Convex |
|--------|------------|
| Only global minimum | Multiple local minima |
| Linear regression | Neural networks |
| Guaranteed convergence | May get stuck |

### Interview Tip
For deep learning, local minima aren't the main problem—saddle points and flat regions are more challenging.

---

## Question 91
- [ ] Done

**Describe the concept of learning rate and its impact on model training.**

**Answer:**

### Definition
The learning rate (η) controls the step size of parameter updates during gradient descent. It determines how quickly or slowly a model learns.

### Impact on Training

| Learning Rate | Effect |
|---------------|--------|
| Too high | Overshoots, diverges, loss increases |
| Too low | Very slow convergence, may get stuck |
| Just right | Smooth convergence to good solution |

### Visual Effect
```
Too High:   ___     ___
           /   \___/   \  (oscillating, diverging)

Too Low:   ________...   (very slow progress)

Good:      \_______      (smooth descent)
```

### Learning Rate Schedules
| Schedule | Description |
|----------|-------------|
| Step decay | Reduce by factor every N epochs |
| Exponential | Decay exponentially |
| Cosine | Smooth cosine-based decay |
| Warm-up | Start low, increase, then decay |
| Cyclic | Oscillate between bounds |

### Finding Good Learning Rate
- Learning rate range test
- Start with 1e-3 or 1e-4 (common defaults)
- Use adaptive optimizers (Adam)

### Interview Tip
Learning rate is arguably the most important hyperparameter. If training isn't working, check learning rate first.

---

## Question 92
- [ ] Done

**What is the purpose of regularization techniques in optimization?**

**Answer:**

### Definition
Regularization adds constraints or penalties to the optimization objective to prevent overfitting and improve generalization to unseen data.

### Core Idea
$$L_{total} = L_{data} + \lambda \cdot R(\theta)$$

Where R(θ) is the regularization term.

### Common Techniques

| Technique | Method | Effect |
|-----------|--------|--------|
| L1 (Lasso) | Penalty: Σ\|w\| | Sparse weights, feature selection |
| L2 (Ridge) | Penalty: Σw² | Small weights, prevents explosion |
| Elastic Net | L1 + L2 | Best of both |
| Dropout | Random neuron zeroing | Ensemble effect |
| Early Stopping | Stop before overfitting | Implicit regularization |
| Data Augmentation | More varied training data | Reduces overfitting |

### L1 vs L2
| L1 | L2 |
|----|-----|
| Sparse solutions | Dense solutions |
| Feature selection | Feature weighting |
| Not differentiable at 0 | Smooth |

### Why Regularization Works
- Limits model complexity
- Prevents memorizing noise
- Encourages simpler solutions
- Reduces variance at cost of slight bias increase

### Interview Tip
Know L1 gives sparsity (some weights exactly zero), L2 shrinks all weights but keeps them non-zero.

---

## Question 93
- [ ] Done

**Explain the concept of hyperparameter tuning and its techniques.**

**Answer:**

### Definition
Hyperparameter tuning is the process of finding the optimal configuration of model hyperparameters (settings not learned from data) to maximize model performance.

### Hyperparameters vs Parameters
| Parameters | Hyperparameters |
|------------|-----------------|
| Learned from data | Set before training |
| Weights, biases | Learning rate, layers, regularization |
| Model adjusts | You adjust |

### Common Hyperparameters
- Learning rate
- Batch size
- Number of layers/neurons
- Regularization strength
- Number of trees (ensemble)
- Max depth (trees)

### Tuning Techniques

| Technique | Method | Pros/Cons |
|-----------|--------|-----------|
| Manual | Trial and error | Simple; slow, biased |
| Grid Search | Exhaustive grid | Complete; expensive |
| Random Search | Random sampling | Efficient; may miss optimal |
| Bayesian | Probabilistic model | Smart; complex |
| Gradient-based | Differentiate through | Fast; limited applicability |

### Validation Strategy
- Use held-out validation set or cross-validation
- NEVER tune on test set
- Track all experiments (MLflow, Weights & Biases)

### Interview Tip
Random search is often better than grid search because it explores the space more efficiently.

---

## Question 94
- [ ] Done

**What is grid search and how is it used for hyperparameter tuning?**

**Answer:**

### Definition
Grid search exhaustively evaluates all combinations of specified hyperparameter values, selecting the combination with the best validation performance.

### How It Works
1. Define parameter grid (values for each hyperparameter)
2. Generate all combinations
3. Train and evaluate model for each combination
4. Select best performing configuration

### Example
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}
# Total: 3 × 3 × 3 = 27 combinations

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```

### Pros and Cons
| Pros | Cons |
|------|------|
| Exhaustive, complete | Exponentially expensive |
| Simple to implement | Curse of dimensionality |
| Reproducible | Inefficient for continuous params |

### When to Use
- Few hyperparameters
- Small search space
- Computational resources available

### Interview Tip
Grid search guarantees finding the best in the grid, but random search is usually more efficient.

---

## Question 95
- [ ] Done

**Describe the concept of random search and its advantages over grid search.**

**Answer:**

### Definition
Random search samples hyperparameter combinations randomly from specified distributions, evaluating a fixed number of configurations.

### How It Works
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 30),
    'learning_rate': uniform(0.01, 0.2)
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,  # Number of random combinations
    cv=5
)
random_search.fit(X_train, y_train)
```

### Advantages Over Grid Search

| Advantage | Explanation |
|-----------|-------------|
| More efficient | Better coverage with fewer evaluations |
| Handles continuous params | Sample from distributions |
| Scales better | Fixed budget regardless of dimensions |
| Finds good solutions faster | Explores more of the space |

### Why Random Search Works Better
- Grid search wastes evaluations on unimportant hyperparameters
- Random search explores unique values for important parameters
- If 1 of 3 hyperparameters matters, grid search tests only a few values

### Interview Tip
Bergstra & Bengio (2012) showed random search is often significantly more efficient than grid search.

---

## Question 96
- [ ] Done

**What is Bayesian optimization and how does it work for hyperparameter tuning?**

**Answer:**

### Definition
Bayesian optimization uses a probabilistic surrogate model to predict hyperparameter performance and intelligently selects the next configuration to evaluate, balancing exploration and exploitation.

### How It Works
1. Build surrogate model (Gaussian Process) of objective function
2. Use acquisition function to select next point
3. Evaluate actual performance
4. Update surrogate model
5. Repeat

### Key Components
| Component | Purpose |
|-----------|---------|
| Surrogate Model | Predicts performance (usually GP) |
| Acquisition Function | Selects next point (EI, UCB) |
| Observed Data | Past evaluations |

### Acquisition Functions
| Function | Strategy |
|----------|----------|
| EI (Expected Improvement) | Balance exploration/exploitation |
| UCB (Upper Confidence Bound) | Optimistic under uncertainty |
| PI (Probability of Improvement) | Probability of beating best |

### Advantages
- Sample-efficient (fewer evaluations)
- Handles noise and expensive functions
- Learns from all past evaluations
- Good for expensive-to-evaluate models

### Tools
- Optuna
- Hyperopt
- Scikit-optimize
- Weights & Biases

### Python Example
```python
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    depth = trial.suggest_int('depth', 3, 10)
    # Train and return validation score
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### Interview Tip
Use Bayesian optimization when each evaluation is expensive (deep learning, large datasets). For quick experiments, random search is often sufficient.

---

