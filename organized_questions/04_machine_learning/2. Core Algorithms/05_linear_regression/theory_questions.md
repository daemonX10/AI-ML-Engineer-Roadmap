# Linear Regression Interview Questions - Theory Questions

## Question 1: What is linear regression and how is it used in predictive modeling?
- [ ] Done

### Answer

**Definition:**
Linear Regression is a fundamental supervised learning algorithm used for regression tasks. It models the relationship between one or more independent variables (features) and a continuous dependent variable (target) by fitting a linear equation.

**The Model Equation:**

$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon$$

| Term | Description |
|------|-------------|
| $y$ | Target variable to be predicted |
| $x_1, x_2, ..., x_n$ | Input features |
| $\beta_0$ | Intercept (value of y when all x are zero) |
| $\beta_1, \beta_2, ...$ | Coefficients for each feature |
| $\epsilon$ | Irreducible error term |

**How It's Used in Predictive Modeling:**

| Phase | Description |
|-------|-------------|
| **Training** | Learn optimal coefficients (β) by minimizing sum of squared errors using Ordinary Least Squares (OLS) |
| **Prediction** | Apply learned equation to new data: $\hat{y} = \beta_0 + \beta_1x_1 + ...$ |
| **Inference** | Interpret coefficients to understand feature-target relationships |

**Key Applications:**
- Price prediction (houses, stocks)
- Sales forecasting
- Risk assessment
- Demand estimation
- Understanding variable relationships

---

## Question 2: Can you explain the difference between simple linear regression and multiple linear regression?
- [ ] Done

### Answer

**Comparison Table:**

| Aspect | Simple Linear Regression | Multiple Linear Regression |
|--------|--------------------------|---------------------------|
| **Predictors** | One input feature | Two or more features |
| **Equation** | $y = \beta_0 + \beta_1x + \epsilon$ | $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \epsilon$ |
| **Geometry** | 2D line | Multi-dimensional hyperplane |
| **Interpretation** | β₁ is direct relationship | βᵢ is relationship holding others constant |
| **Complexity** | Simple, easy to visualize | Deals with multicollinearity |

**Simple Linear Regression:**
- Models relationship between ONE feature and target
- Best-fitting straight line through data points
- Example: Predicting exam score based on hours studied

**Multiple Linear Regression:**
- Models relationship between MULTIPLE features and target
- Best-fitting hyperplane in multi-dimensional space
- Example: Predicting exam score based on hours studied, attendance, and prior GPA
- Key insight: Each coefficient represents effect while holding other features constant

---

## Question 3: What is the role of the intercept term in a linear regression model?
- [ ] Done

### Answer

**Role of the Intercept (β₀):**

| Function | Description |
|----------|-------------|
| **Baseline Value** | Predicted y when all features equal zero |
| **Line Position** | Allows regression line to shift up/down on y-axis |
| **Model Flexibility** | Without it, line must pass through origin (0,0) |

**Why Include the Intercept:**
- Without intercept, model is forced through origin
- This is rarely a valid assumption
- Omitting leads to biased coefficient estimates

**When Intercept is Meaningful:**
```
Example: Sales = β₀ + β₁ × Ad_Spend
- Intercept = predicted sales when ad spend is zero
- This is a meaningful baseline (organic sales)
```

**When Intercept is NOT Meaningful:**
```
Example: Weight = β₀ + β₁ × Height
- Intercept = predicted weight when height is zero
- Nonsensical extrapolation, but still mathematically necessary
```

**Rule:** Always include the intercept unless you have strong theoretical reasons that y must be zero when all x are zero.

---

## Question 4: What are the common metrics to evaluate a linear regression model's performance?
- [ ] Done

### Answer

**Evaluation Metrics:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Average squared error (units²) |
| **RMSE** | $\sqrt{MSE}$ | Average error in original units |
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Average absolute error |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proportion of variance explained (0-1) |
| **Adjusted R²** | $1 - \frac{(1-R²)(n-1)}{n-p-1}$ | R² penalized for extra features |

**Comparison:**

| Metric | Outlier Sensitivity | Units | Use Case |
|--------|---------------------|-------|----------|
| **RMSE** | High (squares errors) | Same as target | Most common, interpretable |
| **MAE** | Low (no squaring) | Same as target | When outliers shouldn't dominate |
| **R²** | N/A | Unitless (0-1) | Comparing models, explaining fit |
| **Adjusted R²** | N/A | Unitless | Comparing models with different features |

**Key Points:**
- R² always increases with more features (even useless ones)
- Adjusted R² only increases if new feature genuinely improves model
- Use Adjusted R² for comparing models with different numbers of features

---

## Question 5: Explain the concept of homoscedasticity. Why is it important?
- [ ] Done

### Answer

**Definition:**
Homoscedasticity means "constant variance" - the variance of error terms (residuals) is constant across all levels of input features.

**Visual Comparison:**

| Pattern | What You See | Implication |
|---------|-------------|-------------|
| **Homoscedasticity** | Random scatter, uniform spread | Assumption met ✓ |
| **Heteroscedasticity** | Fan/cone shape | Assumption violated ✗ |

**Why It's Important:**

| Consequence of Violation | Impact |
|--------------------------|--------|
| **OLS still unbiased** | Coefficient estimates are still correct on average |
| **OLS not efficient** | No longer BLUE (Best Linear Unbiased Estimator) |
| **Invalid standard errors** | Standard errors are biased |
| **Unreliable hypothesis tests** | t-tests, F-tests, p-values are wrong |
| **Wrong confidence intervals** | May incorrectly conclude significance |

**How to Detect:**
- Residual vs. Fitted plot (look for fan shape)
- Breusch-Pagan test
- White's test

**How to Address:**
1. **Transform target variable** (log, square root)
2. **Use Weighted Least Squares (WLS)**
3. **Use robust standard errors** (heteroscedasticity-consistent)

---

## Question 6: What is multicollinearity and how can it affect a regression model?
- [ ] Done

### Answer

**Definition:**
Multicollinearity occurs when two or more input features are highly correlated with each other (not with the target).

**Effects on the Model:**

| Problem | Description |
|---------|-------------|
| **Unstable coefficients** | Coefficients swing wildly with small data changes |
| **Large standard errors** | Variance of estimates is inflated |
| **Difficult interpretation** | Can't isolate individual feature effects |
| **Reduced significance** | High p-values even for important features |

**What It Does NOT Affect:**
- Overall predictive accuracy
- Unbiasedness of coefficients (still unbiased, just imprecise)

**Detection Methods:**

| Method | Threshold |
|--------|-----------|
| **Correlation matrix** | \|r\| > 0.8-0.9 |
| **VIF (Variance Inflation Factor)** | VIF > 5 or 10 |

**VIF Formula:**
$$VIF_i = \frac{1}{1 - R_i^2}$$
Where $R_i^2$ is the R² from regressing feature i against all other features.

**Solutions:**
1. Remove one of the correlated features
2. Combine correlated features (e.g., PCA)
3. Use Ridge Regression (shrinks correlated coefficients together)

---

## Question 7: Describe the steps involved in preprocessing data for linear regression analysis.
- [ ] Done

### Answer

**Preprocessing Pipeline:**

| Step | Action | Why |
|------|--------|-----|
| **1. Handle Missing Values** | Impute with median/mean or drop | LR can't handle NaN |
| **2. Handle Outliers** | Detect (Z-score, IQR) and treat | LR is sensitive to outliers |
| **3. Encode Categoricals** | One-hot encoding | LR needs numerical inputs |
| **4. Check Non-linearity** | Scatter plots, transform if needed | Linearity assumption |
| **5. Feature Scaling** | Standardization/Normalization | Essential for regularization |
| **6. Check Multicollinearity** | Correlation matrix, VIF | Stable coefficients |
| **7. Data Splitting** | Train/test split | Prevent data leakage |

**Code Example:**
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numerical_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first'))
    ]), categorical_cols)
])
```

**Key Rule:** Fit preprocessing only on training data, then transform test data.

---

## Question 8: Explain the concept of data splitting into training and test sets.
- [ ] Done

### Answer

**Purpose:**
Estimate how well the model generalizes to new, unseen data.

**The Split:**

| Set | Purpose | Typical Size |
|-----|---------|--------------|
| **Training** | Learn model parameters | 70-80% |
| **Test** | Final unbiased evaluation | 20-30% |
| **Validation** (optional) | Tune hyperparameters | Part of training |

**Why It's Critical:**
- Evaluating on training data gives optimistic, misleading results
- Model may have memorized training data (overfitting)
- Test set acts as proxy for real-world data

**Three-Way Split Workflow:**
```
1. Train on training set
2. Tune hyperparameters using validation set
3. Final evaluation on test set (only once!)
```

**Code:**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Key Rule:** Never use test set for any decisions during model development.

---

## Question 9: What is cross-validation and how is it performed with linear regression?
- [ ] Done

### Answer

**Definition:**
Cross-validation is a resampling technique that provides more robust performance estimates by training and evaluating on different subsets of data.

**K-Fold Cross-Validation Process:**

```
1. Split data into k equal folds (e.g., k=5)
2. For each fold:
   - Use 1 fold as validation
   - Use remaining k-1 folds for training
   - Record performance score
3. Average all k scores for final estimate
```

**Visual (5-Fold):**
```
Fold 1: [TEST] [Train] [Train] [Train] [Train] → Score 1
Fold 2: [Train] [TEST] [Train] [Train] [Train] → Score 2
Fold 3: [Train] [Train] [TEST] [Train] [Train] → Score 3
Fold 4: [Train] [Train] [Train] [TEST] [Train] → Score 4
Fold 5: [Train] [Train] [Train] [Train] [TEST] → Score 5
                                                → Mean Score
```

**Use Cases in Linear Regression:**

| Use Case | Description |
|----------|-------------|
| **Performance Estimation** | Get reliable R²/RMSE estimate |
| **Hyperparameter Tuning** | Find optimal alpha for Ridge/Lasso |
| **Model Comparison** | Compare different feature sets |

**Code:**
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=5, scoring='r2')
print(f"Mean R²: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

---

## Question 10: Can you explain the concept of gradient descent and its importance in linear regression?
- [ ] Done

### Answer

**What is Gradient Descent?**
An iterative optimization algorithm to find the minimum of a function by following the negative gradient.

**The Analogy:**
Standing on a foggy mountain → feel the slope → step downhill → repeat until reaching valley floor.

**Update Rule:**
$$\beta_{new} = \beta_{old} - \alpha \cdot \frac{\partial Loss}{\partial \beta}$$

Where $\alpha$ is the learning rate (step size).

**Comparison: Normal Equation vs Gradient Descent**

| Aspect | Normal Equation | Gradient Descent |
|--------|-----------------|------------------|
| **Formula** | $\beta = (X^TX)^{-1}X^Ty$ | Iterative updates |
| **Complexity** | O(p³) | O(p) per iteration |
| **Small features** | Fast | Slower |
| **Large features** | Infeasible | Scalable |
| **Memory** | High | Low |

**Types of Gradient Descent:**

| Type | Batch Size | Speed | Stability |
|------|------------|-------|-----------|
| **Batch** | Entire dataset | Slow | Stable |
| **Stochastic (SGD)** | 1 sample | Fast | Noisy |
| **Mini-Batch** | Small batch (32-256) | Medium | Balanced |

**Importance:**
- Scalable to millions of features
- Foundation for training all neural networks
- Works when normal equation is computationally infeasible

---

## Question 11: What is ridge regression and how does it differ from standard linear regression?
- [ ] Done

### Answer

**Ridge Regression = Linear Regression + L2 Regularization**

**Loss Function Comparison:**

| Model | Loss Function |
|-------|---------------|
| **OLS** | $\sum(y_i - \hat{y}_i)^2$ |
| **Ridge** | $\sum(y_i - \hat{y}_i)^2 + \alpha \sum \beta_j^2$ |

The L2 penalty ($\alpha \sum \beta_j^2$) penalizes large coefficients.

**Key Differences:**

| Aspect | OLS | Ridge |
|--------|-----|-------|
| **Coefficients** | Can be large, unstable | Shrunk towards zero |
| **Overfitting** | Prone with many features | Reduced risk |
| **Multicollinearity** | Unstable coefficients | Robust and stable |
| **Bias-Variance** | Low bias, high variance | Higher bias, lower variance |
| **Feature Selection** | No | No (coefficients ≠ 0) |

**When to Use Ridge:**
- Many features with suspected multicollinearity
- All features believed to be relevant
- Prevent overfitting

**Code:**
```python
from sklearn.linear_model import Ridge, RidgeCV

# With cross-validation for alpha selection
ridge = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
ridge.fit(X_train, y_train)
print(f"Best alpha: {ridge.alpha_}")
```

---

## Question 12: Explain the concept of Lasso regression and its benefits.
- [ ] Done

### Answer

**Lasso = Linear Regression + L1 Regularization**

**Loss Function:**
$$\sum(y_i - \hat{y}_i)^2 + \alpha \sum |\beta_j|$$

The L1 penalty ($\alpha \sum |\beta_j|$) uses absolute values of coefficients.

**Key Benefit: Automatic Feature Selection**

| Regularization | Coefficient Shrinkage | Feature Selection |
|----------------|----------------------|-------------------|
| **Ridge (L2)** | Shrinks towards zero | No (never exactly 0) |
| **Lasso (L1)** | Shrinks to exactly zero | Yes (sparse model) |

**Why Lasso Creates Sparse Models:**
- L1 penalty geometry creates "corners" where coefficients become exactly zero
- Less important features are eliminated completely

**Benefits:**

| Benefit | Description |
|---------|-------------|
| **Feature Selection** | Automatically identifies important features |
| **Sparse Models** | Fewer non-zero coefficients |
| **Interpretability** | Easier to explain with fewer features |
| **Reduced Overfitting** | Regularization reduces variance |

**When to Use:**
- High-dimensional data (many features)
- Suspect many features are irrelevant
- Want interpretable, sparse model

**Code:**
```python
from sklearn.linear_model import Lasso, LassoCV

lasso = LassoCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=5)
lasso.fit(X_train, y_train)
print(f"Non-zero coefficients: {np.sum(lasso.coef_ != 0)}")
```

---

## Question 13: What is elastic net regression and in what cases would you use it?
- [ ] Done

### Answer

**Elastic Net = L1 + L2 Regularization Combined**

**Loss Function:**
$$\sum(y_i - \hat{y}_i)^2 + \alpha \left[ \rho \sum|\beta_j| + \frac{(1-\rho)}{2} \sum\beta_j^2 \right]$$

| Parameter | Description |
|-----------|-------------|
| **α (alpha)** | Overall regularization strength |
| **ρ (l1_ratio)** | Mix between L1 and L2 |
| ρ = 1 | Pure Lasso |
| ρ = 0 | Pure Ridge |
| 0 < ρ < 1 | Combination |

**When to Use Elastic Net:**

| Scenario | Why Elastic Net |
|----------|-----------------|
| **Correlated features** | Lasso selects one arbitrarily; Elastic Net selects groups |
| **p >> n** | Lasso can select at most n features; Elastic Net handles better |
| **Unsure Ridge vs Lasso** | Tune l1_ratio to find optimal mix |

**The "Grouping Effect":**
- With correlated features, Lasso picks one randomly
- Elastic Net keeps correlated features together
- More stable feature selection

**Code:**
```python
from sklearn.linear_model import ElasticNetCV

elastic_net = ElasticNetCV(
    alphas=[0.01, 0.1, 1.0],
    l1_ratio=[0.1, 0.5, 0.9],
    cv=5
)
elastic_net.fit(X_train, y_train)
print(f"Best alpha: {elastic_net.alpha_}, l1_ratio: {elastic_net.l1_ratio_}")
```

---

## Question 14: Explain the purpose of residual plots and how to interpret them.
- [ ] Done

### Answer

**What is a Residual Plot?**
Scatter plot of predicted values (x-axis) vs. residuals (y-axis).

**Residual:** $e_i = y_i - \hat{y}_i$ (actual - predicted)

**Interpretation Guide:**

| Pattern | What You See | Diagnosis | Action |
|---------|-------------|-----------|--------|
| **Random scatter** | No pattern, uniform spread | Good fit ✓ | None needed |
| **U-shape/Curve** | Systematic curve | Non-linearity | Add polynomial features |
| **Fan shape** | Spread increases/decreases | Heteroscedasticity | Transform target |
| **Clusters** | Distinct groups | Missing categorical | Add indicator variables |
| **Outliers** | Points far from 0 line | Influential points | Investigate/remove |

**Visual Examples:**

```
GOOD (Random):          BAD (Curved):           BAD (Fan):
    •  •                    •    •                  •
  •    •  •                  ••                    ••
----•--•-----            --••----••--          --•••-------
  •  •   •                ••    ••                 •••••
    •                    •        •                   ••••••
```

**Code:**
```python
import matplotlib.pyplot as plt

residuals = y_test - model.predict(X_test)
plt.scatter(model.predict(X_test), residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```

---

## Question 15: What is the adjusted R-squared, and why is it used?
- [ ] Done

### Answer

**The Problem with R²:**
R² always increases (or stays same) when adding features, even useless ones.

**Adjusted R² Formula:**
$$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

Where:
- n = number of samples
- p = number of features

**Key Difference:**

| Metric | Behavior with Added Features |
|--------|------------------------------|
| **R²** | Always increases or stays same |
| **Adjusted R²** | Only increases if feature adds genuine value |

**Example:**
```
Model A (5 features): R² = 0.85, Adj R² = 0.84
Model B (8 features): R² = 0.86, Adj R² = 0.82

→ Model A is better! Extra features in B don't add value.
```

**When to Use:**
- Comparing models with different numbers of features
- Avoiding overfitting to training data
- Model selection in multiple regression

**Interpretation:**
- Adj R² can be negative (very poor model)
- Higher is better
- Penalizes unnecessary complexity

---

## Question 16: What are leverage points and how do they affect a regression model?
- [ ] Done

### Answer

**Definition:**
A leverage point has extreme values in its input features (X values), far from the center of the data cloud.

**The Lever Analogy:**
Like a seesaw - a person far from the center has more influence on the tilt.

**Types of Leverage Points:**

| Type | Description | Effect |
|------|-------------|--------|
| **Good Leverage** | High leverage, follows trend (small residual) | Stabilizes model, reduces standard errors |
| **Bad Leverage** | High leverage, doesn't follow trend (large residual) | Dramatically distorts the regression line |

**Detection:**

| Method | Formula/Threshold |
|--------|-------------------|
| **Leverage statistic (hᵢᵢ)** | hᵢᵢ > 2(p+1)/n |
| **Visualization** | Points far on x-axis |

**Leverage vs. Outlier vs. Influential:**

| Term | Definition |
|------|------------|
| **Leverage point** | Extreme X values |
| **Outlier** | Extreme Y values (large residual) |
| **Influential point** | Changes model significantly if removed |

**Key Insight:**
A point is influential only if it has BOTH high leverage AND large residual. Use Cook's distance to measure overall influence.

---

## Question 17: Describe how you would detect and address outliers in your regression analysis.
- [ ] Done

### Answer

**Detection Methods:**

| Method | Type | Description |
|--------|------|-------------|
| **Residual plots** | Visual | Points far from y=0 line |
| **Box plots** | Visual | Univariate outliers |
| **Standardized residuals** | Statistical | \|residual\| > 3 |
| **Cook's distance** | Statistical | Dᵢ > 4/n or Dᵢ > 1 |
| **Leverage plot** | Visual | Combines leverage, residual, Cook's D |

**Treatment Decision Tree:**

```
Is it a data entry/measurement error?
├── Yes → Correct or remove
└── No → Is it a genuine extreme value?
    ├── Yes → Consider:
    │   ├── Transformation (log, sqrt)
    │   ├── Robust regression (Huber, RANSAC)
    │   └── Keep and report impact
    └── No → Remove with documentation
```

**Treatment Options:**

| Method | When to Use |
|--------|-------------|
| **Removal** | Clear error, with justification |
| **Transformation** | Skewed data (log reduces outlier impact) |
| **Winsorization** | Cap extreme values at percentiles |
| **Robust regression** | When outliers represent valid variation |

**Code:**
```python
from sklearn.linear_model import HuberRegressor

# Robust to outliers
robust_model = HuberRegressor()
robust_model.fit(X_train, y_train)
```

---

## Question 18: Explain the concept of Cook's distance.
- [ ] Done

### Answer

**Definition:**
Cook's distance measures the overall influence of a single observation on the entire regression model by combining leverage and residual size.

**What It Measures:**
How much all predicted values change when observation i is removed.

**Combines Two Factors:**

| Factor | Description |
|--------|-------------|
| **Leverage** | How extreme are the X values? |
| **Residual** | How far is the point from the regression line? |

**Key Insight:**
A point needs BOTH high leverage AND large residual to have high Cook's distance.

| Leverage | Residual | Cook's D | Interpretation |
|----------|----------|----------|----------------|
| High | Small | Low | Good leverage point |
| Low | Large | Low | Outlier with little influence |
| High | Large | **High** | **Influential - investigate!** |

**Thresholds:**

| Rule of Thumb | Threshold |
|---------------|-----------|
| Common | Dᵢ > 4/n |
| Conservative | Dᵢ > 1 |

**Code:**
```python
import statsmodels.api as sm

model = sm.OLS(y, sm.add_constant(X)).fit()
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]

# Identify influential points
influential = np.where(cooks_d > 4/len(y))[0]
```

---

## Question 19: Describe the variance inflation factor (VIF) and its significance.
- [ ] Done

### Answer

**Definition:**
VIF quantifies how much a coefficient's variance is inflated due to multicollinearity with other predictors.

**Formula:**
$$VIF_i = \frac{1}{1 - R_i^2}$$

Where $R_i^2$ is the R² from regressing feature i against all other features.

**Interpretation:**

| VIF Value | Interpretation |
|-----------|----------------|
| VIF = 1 | No correlation with other features (ideal) |
| 1 < VIF < 5 | Moderate, generally acceptable |
| 5 < VIF < 10 | High multicollinearity (concerning) |
| VIF > 10 | Severe multicollinearity (action needed) |

**Example:**
VIF = 10 means the coefficient's variance is 10× larger than if uncorrelated.

**Code:**
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                   for i in range(X.shape[1])]
print(vif_data.sort_values('VIF', ascending=False))
```

**Actions for High VIF:**
1. Remove one of the correlated features
2. Combine features (PCA)
3. Use Ridge regression

---

## Question 20: How does polynomial regression extend the linear regression model?
- [ ] Done

### Answer

**Concept:**
Polynomial regression fits non-linear relationships by creating polynomial features, then applying standard linear regression.

**Transformation:**
Original: $y = \beta_0 + \beta_1 x$

Polynomial (degree 2): $y = \beta_0 + \beta_1 x + \beta_2 x^2$

**Key Insight:**
The model is still LINEAR in the coefficients (β), just non-linear in the original feature (x).

**Implementation:**
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
poly_model.fit(X, y)
```

**Trade-offs:**

| Degree | Bias | Variance | Fit |
|--------|------|----------|-----|
| Low (1-2) | High | Low | Underfitting |
| Medium (3-4) | Medium | Medium | Good |
| High (>5) | Low | High | Overfitting |

**Best Practice:**
- Use cross-validation to select optimal degree
- Consider regularization for high degrees
- Watch for overfitting at data edges

---

## Question 21: What are generalized linear models (GLMs), and how do they relate to linear regression?
- [ ] Done

### Answer

**GLMs extend linear regression to handle different types of target variables.**

**Three Components of a GLM:**

| Component | Description | Linear Regression |
|-----------|-------------|-------------------|
| **Random** | Distribution of y | Gaussian (Normal) |
| **Systematic** | Linear predictor η = β₀ + β₁x₁ + ... | Same |
| **Link Function** | Connects η to E(y) | Identity: μ = η |

**Common GLM Family:**

| Model | Target Type | Distribution | Link Function |
|-------|-------------|--------------|---------------|
| **Linear Regression** | Continuous | Gaussian | Identity |
| **Logistic Regression** | Binary (0/1) | Bernoulli | Logit |
| **Poisson Regression** | Count (0,1,2,...) | Poisson | Log |

**Logistic Regression Example:**
$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + ...$$

**Poisson Regression Example:**
$$\log(\mu) = \beta_0 + \beta_1 x_1 + ...$$

**Key Takeaway:**
Linear regression is a special case of GLMs with Gaussian distribution and identity link.

---

## Question 22: Explain how quantile regression differs from ordinary least squares (OLS) regression.
- [ ] Done

### Answer

**Fundamental Difference:**

| Model | What It Models | Loss Function |
|-------|----------------|---------------|
| **OLS** | Conditional MEAN | Sum of squared errors |
| **Quantile** | Conditional QUANTILES | Asymmetrically weighted absolute errors |

**Why Use Quantile Regression:**

| Scenario | Why Quantile Regression |
|----------|------------------------|
| Outliers present | Median regression (50th quantile) is robust |
| Heteroscedasticity | Model different quantiles to capture spread |
| Risk analysis | 95th percentile for worst-case scenarios |

**Example:**
- Median regression: 50% of data above, 50% below
- 90th percentile: 90% of data below this line

**Comparison:**

| Aspect | OLS | Quantile |
|--------|-----|----------|
| **Output** | One line (mean) | Multiple lines (any quantile) |
| **Outlier sensitivity** | High | Low |
| **Assumptions** | Homoscedasticity | No variance assumptions |
| **Information** | Central tendency | Full distribution shape |

**Code:**
```python
from sklearn.linear_model import QuantileRegressor

# Median regression
median_model = QuantileRegressor(quantile=0.5)
median_model.fit(X, y)

# 90th percentile
upper_model = QuantileRegressor(quantile=0.9)
upper_model.fit(X, y)
```

---

## Question 23: What are mixed models, and where might you use them?
- [ ] Done

### Answer

**Definition:**
Mixed models (mixed-effects models) combine:
- **Fixed effects:** Standard regression coefficients (population-level)
- **Random effects:** Group-specific variations

**When Data is NOT Independent:**

| Scenario | Example | Random Effect |
|----------|---------|---------------|
| **Repeated measures** | Patient measurements over time | Patient-specific intercept |
| **Clustered data** | Students in schools | School-specific intercept |
| **Hierarchical data** | Employees in companies | Company-specific effects |

**Example: Medical Study**
- Fixed effect: Treatment effect (what we want to estimate)
- Random effect: Patient baseline (each patient has different starting blood pressure)

**Why Not Just OLS?**
- OLS assumes independent observations
- Observations from same patient/school are correlated
- Mixed models correctly account for this structure

**Model Structure:**
$$y_{ij} = \beta_0 + \beta_1 x_{ij} + u_j + \epsilon_{ij}$$

Where:
- $\beta_0, \beta_1$ = fixed effects
- $u_j$ = random effect for group j
- $\epsilon_{ij}$ = residual error

**Code (statsmodels):**
```python
import statsmodels.formula.api as smf

model = smf.mixedlm("score ~ treatment", data, groups=data["patient_id"])
result = model.fit()
```

---

## Question 24: What are the latest research trends in regularized regression techniques?
- [ ] Done

### Answer

**Current Research Trends:**

| Trend | Description |
|-------|-------------|
| **Group Lasso** | Select/discard groups of features together |
| **Adaptive Lasso** | Different penalties for different coefficients |
| **Sparse Additive Models** | Non-linear functions with Lasso-style sparsity |
| **Debiased Lasso** | Valid inference after Lasso selection |
| **Causal Regularization** | Double ML for treatment effect estimation |

**1. Group Lasso:**
```python
# Select entire one-hot encoded groups together
# Not just individual dummies
```

**2. Adaptive Lasso:**
- First-stage: Get initial estimates (e.g., Ridge)
- Second-stage: Penalize inversely to initial estimate size
- Benefits: Better selection consistency

**3. Double/Debiased Machine Learning:**
- Use Lasso for nuisance parameter estimation
- Valid confidence intervals for causal effects
- Handles high-dimensional confounders

**4. Graph Regularization:**
- Incorporate feature relationship structure
- Connected features get similar coefficients

**Key Direction:**
Moving from "just prediction" to valid statistical inference in high-dimensional settings.

---

## Question 25: Describe a situation where logistic regression might be preferred over linear regression.
- [ ] Done

### Answer

**When Target is Binary (0/1):**

| Aspect | Linear Regression | Logistic Regression |
|--------|------------------|---------------------|
| **Target type** | Continuous | Binary categorical |
| **Output** | Any real number | Probability [0, 1] |
| **Example target** | Price ($) | Churn (Yes/No) |

**Why Linear Regression Fails for Binary:**

1. **Unbounded predictions:** Can predict -0.5 or 1.5 (meaningless for probability)
2. **Violates assumptions:** Errors aren't normally distributed
3. **No probability interpretation**

**Example: Customer Churn Prediction**

```python
# Wrong: Linear Regression
from sklearn.linear_model import LinearRegression
# Predictions might be < 0 or > 1

# Correct: Logistic Regression
from sklearn.linear_model import LogisticRegression
# Predictions are probabilities between 0 and 1
```

**Logistic Regression Output:**
- Probability of churn (0.85 = 85% chance)
- Classification via threshold (>0.5 → Churn)

---

## Question 26: Describe a scenario where you'd have to transition from a simple to a multiple linear regression model, and the considerations you'd have to make.
- [ ] Done

### Answer

**Scenario: Predicting Student Exam Scores**

**Phase 1: Simple Linear Regression**
```python
Score = β₀ + β₁ × Hours_Studied
# R² = 0.30 → Only 30% variance explained
# Underfitting - need more features!
```

**Phase 2: Multiple Linear Regression**
```python
Score = β₀ + β₁×Hours + β₂×GPA + β₃×Attendance + ...
```

**Key Considerations:**

| Consideration | Action |
|---------------|--------|
| **Feature Selection** | Use domain knowledge, not random features |
| **Multicollinearity** | Check VIF (Hours & Assignments may correlate) |
| **Coefficient Interpretation** | Now "holding other factors constant" |
| **Evaluation Metric** | Switch to Adjusted R² |
| **Overfitting Risk** | Consider regularization |

**Interpretation Change:**
- Simple: β₁ = total effect of studying
- Multiple: β₁ = effect of studying, controlling for GPA and attendance

**Validation:**
```python
from sklearn.model_selection import cross_val_score

# Compare models
simple_cv = cross_val_score(simple_model, X_simple, y, cv=5)
multiple_cv = cross_val_score(multiple_model, X_multiple, y, cv=5)

print(f"Simple: {simple_cv.mean():.3f}")
print(f"Multiple: {multiple_cv.mean():.3f}")
```

---

## Question 27: What are the mathematical foundations and assumptions underlying linear regression?
- [ ] Done

### Answer

**The Model:**
$$y = X\beta + \epsilon$$

**Gauss-Markov Assumptions (for BLUE):**

| Assumption | Mathematical Form | Meaning |
|------------|-------------------|---------|
| **Linearity** | E(y) = Xβ | Mean is linear in X |
| **Exogeneity** | E(ε\|X) = 0 | Errors uncorrelated with features |
| **Homoscedasticity** | Var(ε\|X) = σ²I | Constant error variance |
| **No autocorrelation** | Cov(εᵢ, εⱼ) = 0 | Errors are independent |
| **Full rank** | rank(X) = p | No perfect multicollinearity |

**Additional for Inference:**
- Normality: ε ~ N(0, σ²) for valid t-tests, F-tests

**BLUE Property:**
Under Gauss-Markov assumptions, OLS estimator is:
- **B**est (minimum variance)
- **L**inear (linear function of y)
- **U**nbiased (E(β̂) = β)
- **E**stimator

---

## Question 28: How do you derive the normal equation for linear regression and when is it preferred over gradient descent?
- [ ] Done

### Answer

**Derivation:**

1. Loss function: $L = (y - X\beta)^T(y - X\beta)$

2. Expand: $L = y^Ty - 2\beta^TX^Ty + \beta^TX^TX\beta$

3. Derivative: $\frac{\partial L}{\partial \beta} = -2X^Ty + 2X^TX\beta$

4. Set to zero: $X^TX\beta = X^Ty$

5. **Normal Equation:** $\beta = (X^TX)^{-1}X^Ty$

**When to Use Each:**

| Criteria | Normal Equation | Gradient Descent |
|----------|-----------------|------------------|
| **Features (p)** | p < 10,000 | p large |
| **Samples (n)** | Any | Any |
| **Complexity** | O(p³) | O(iterations × p) |
| **Memory** | High (store X^TX) | Low |
| **Convergence** | Exact solution | Iterative |

**Rule of Thumb:**
- p < 10,000: Use Normal Equation (faster)
- p > 10,000 or sparse: Use Gradient Descent (scalable)

---

## Question 29: What is the difference between ordinary least squares (OLS) and other regression estimation methods?
- [ ] Done

### Answer

**Comparison of Estimation Methods:**

| Method | Minimizes | Assumptions | Use Case |
|--------|-----------|-------------|----------|
| **OLS** | Sum of squared errors | Gauss-Markov | Standard regression |
| **WLS** | Weighted squared errors | Known weights | Heteroscedasticity |
| **GLS** | Generalized squared errors | Known covariance | Correlated errors |
| **MLE** | Negative log-likelihood | Distributional | Known error distribution |
| **LAD** | Sum of absolute errors | Fewer | Outlier robustness |
| **Bayesian** | Posterior | Prior beliefs | Uncertainty quantification |

**OLS Properties:**
- BLUE under Gauss-Markov
- Equivalent to MLE under normality
- Simple and interpretable

**When OLS is NOT Best:**
| Problem | Better Alternative |
|---------|-------------------|
| Heteroscedasticity | WLS |
| Autocorrelated errors | GLS |
| Heavy outliers | LAD (quantile regression) |
| Prior information | Bayesian regression |
| High-dimensional | Regularized (Ridge, Lasso) |

---


---

# --- Missing Questions Restored from Source (Q34-Q30) ---

## Question 30
- [ ] Done

**How do you handle categorical variables in linear regression models?**

**Answer:**

Categorical variables must be converted to numerical form before use in linear regression.

**Encoding Methods:**

| Method | Description | When to Use |
|--------|-------------|-------------|
| **One-Hot Encoding** | Creates binary (0/1) column for each category | Nominal variables with few categories |
| **Dummy Encoding** | One-hot with one category dropped (reference) | Default for regression to avoid multicollinearity |
| **Ordinal Encoding** | Assigns integers based on order | Ordinal variables (low/medium/high) |
| **Target Encoding** | Replace category with mean of target | High-cardinality features |
| **Binary Encoding** | Binary representation of category index | High-cardinality, memory-efficient |

**The Dummy Variable Trap:**
- If a categorical variable has k categories, use k-1 dummy variables
- Including all k creates perfect multicollinearity (columns sum to 1)
- The dropped category becomes the **reference/baseline** category

**Interpretation of coefficients:**
- Each dummy coefficient represents the difference from the reference category
- Example: If "City_B = 5000", people in City B earn $5,000 more than the reference city

**Example with Python:**
```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Dummy encoding (drop_first=True avoids the trap)
dummies = pd.get_dummies(df['color'], drop_first=True)
```

---

## Question 31
- [ ] Done

**What are polynomial regression and its relationship to linear regression?**

**Answer:**

Polynomial regression models **non-linear relationships** while remaining a **linear model** in terms of its parameters.

**Key Relationship:**

| Aspect | Linear Regression | Polynomial Regression |
|--------|------------------|----------------------|
| **Equation** | y = β₀ + β₁x | y = β₀ + β₁x + β₂x² + ... + βₙxⁿ |
| **Linearity** | Linear in x and β | Non-linear in x, but **linear in β** |
| **Flexibility** | Straight line only | Can capture curves, U-shapes |
| **Overfitting Risk** | Low | Increases with degree |

**Why it's still "linear" regression:**
- The model is linear in its **parameters** (β values)
- x² is treated as a new feature; OLS still applies
- Feature transformation: [x] → [x, x², x³, ...]

**Choosing the degree:**
- Degree 1: Straight line
- Degree 2: Parabola (most common non-linear extension)
- Degree 3+: Higher flexibility but risk of overfitting
- Use **cross-validation** to select optimal degree
- **Runge's phenomenon**: Very high degrees cause wild oscillations at edges

**Implementation:**
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('reg', LinearRegression())
])
```

---

## Question 32
- [ ] Done

**How do you implement and interpret interaction terms in multiple linear regression?**

**Answer:**

Interaction terms capture how the effect of one feature **depends on the value of another** feature.

**Model with interaction:**
```
y = β₀ + β₁x₁ + β₂x₂ + β₃(x₁ × x₂) + ε
```

**Interpretation:**
- **β₁**: Effect of x₁ when x₂ = 0
- **β₂**: Effect of x₂ when x₁ = 0
- **β₃**: How much the effect of x₁ changes per unit increase in x₂ (and vice versa)

**Example:**
- Predicting salary: y = β₀ + β₁(experience) + β₂(education) + β₃(experience × education)
- β₃ > 0 means the return on experience is **higher** for more educated workers

**When to include interactions:**

| Signal | Action |
|--------|--------|
| Domain knowledge suggests synergy/moderation | Add interaction |
| Residual patterns show structure | Test interactions |
| Significant improvement in R² with interaction | Keep it |
| Non-significant β₃ (p > 0.05) | Remove interaction |

**Best practices:**
- Always include **main effects** when including their interaction
- Center variables before creating interactions to reduce multicollinearity
- Don't include too many interactions — leads to overfitting
- Use **hierarchical principle**: Include lower-order terms when higher-order terms are present

---

## Question 33
- [ ] Done

**What is regularization in linear regression and why is it important?**

**Answer:**

Regularization adds a **penalty term** to the OLS cost function that constrains the magnitude of coefficients, preventing overfitting.

**Why it's important:**

| Problem | How Regularization Helps |
|---------|------------------------|
| **Overfitting** | Constrains model complexity by shrinking coefficients |
| **Multicollinearity** | Stabilizes coefficient estimates when features are correlated |
| **High dimensionality (p > n)** | Makes the problem solvable when OLS has no unique solution |
| **Feature selection** | Lasso (L1) can drive coefficients to exactly zero |

**Types of regularization:**

| Type | Penalty Term | Effect |
|------|-------------|--------|
| **Ridge (L2)** | λΣβⱼ² | Shrinks all coefficients toward zero, never exactly zero |
| **Lasso (L1)** | λΣ|βⱼ| | Can produce exactly zero coefficients → sparse models |
| **Elastic Net** | λ₁Σ|βⱼ| + λ₂Σβⱼ² | Combines L1 and L2 benefits |

**Bias-variance perspective:**
- No regularization (λ=0): Low bias, high variance → overfitting
- Strong regularization (large λ): High bias, low variance → underfitting
- Optimal λ (via CV): Best bias-variance tradeoff → good generalization

---

## Question 34
- [ ] Done

**Explain the differences between Ridge, Lasso, and Elastic Net regression.**

**Answer:**

| Aspect | Ridge (L2) | Lasso (L1) | Elastic Net |
|--------|-----------|-----------|-------------|
| **Penalty** | λΣβⱼ² | λΣ|βⱼ| | α·λΣ|βⱼ| + (1-α)·λΣβⱼ² |
| **Coefficient shrinkage** | Toward zero, never exactly zero | Can be exactly zero | Combination |
| **Feature selection** | No (keeps all features) | Yes (sparse solutions) | Yes (grouped sparsity) |
| **Multicollinearity** | Handles well (distributes weight) | Picks one, drops others | Handles well (selects groups) |
| **Solution** | Closed-form available | No closed-form, iterative | No closed-form, iterative |
| **Geometry** | Circular constraint region | Diamond constraint region | Blend of circle and diamond |
| **Best when** | Many small effects | Few important features | Correlated feature groups |

**Key differences explained:**
- **Ridge**: Distributes coefficient weight among correlated features; good when all features contribute
- **Lasso**: Arbitrarily selects one from a group of correlated features; good for sparse models
- **Elastic Net**: Selects groups of correlated features together; best of both worlds

**Selection guide:**
1. Start with **Ridge** if you expect all features to be relevant
2. Use **Lasso** if you want automatic feature selection
3. Use **Elastic Net** when features are correlated and you want group selection

---

## Question 35
- [ ] Done

**How do you choose the optimal regularization parameter (lambda) in regularized regression?**

**Answer:**

The regularization parameter λ (alpha in sklearn) controls the tradeoff between fitting the data and keeping coefficients small.

**Methods to choose optimal λ:**

| Method | Description | Pros |
|--------|-------------|------|
| **K-Fold Cross-Validation** | Try multiple λ values, evaluate on held-out folds | Most reliable, standard approach |
| **RidgeCV / LassoCV** | Built-in efficient LOOCV implementation | Very fast, automatic |
| **Information Criteria (AIC/BIC)** | Penalize model complexity analytically | Fast, no CV needed |
| **Validation Curve** | Plot train/test error vs λ | Visual understanding |

**Cross-validation approach:**
```python
from sklearn.linear_model import LassoCV
model = LassoCV(alphas=np.logspace(-4, 4, 100), cv=5)
model.fit(X_train, y_train)
print(f"Optimal alpha: {model.alpha_}")
```

**Important considerations:**
- Search λ on a **logarithmic scale** (e.g., 10⁻⁴ to 10⁴)
- **Standardize features** before fitting (penalty is scale-dependent)
- Use the **1-SE rule**: Choose largest λ within 1 standard error of the minimum CV error
- For Elastic Net, jointly tune both λ and the L1/L2 ratio

---

## Question 36
- [ ] Done

**What is cross-validation and how is it used in linear regression model selection?**

**Answer:**

Cross-validation (CV) estimates how well a model will **generalize to unseen data** by repeatedly splitting data into training and validation sets.

**K-Fold Cross-Validation Process:**
1. Split data into K equal folds (typically K=5 or 10)
2. For each fold: train on K-1 folds, validate on the remaining fold
3. Average the K validation scores

**Uses in linear regression model selection:**

| Purpose | How CV Helps |
|---------|-------------|
| **Choosing regularization strength (λ)** | Compare CV scores across λ values |
| **Selecting features** | Compare CV scores with different feature subsets |
| **Choosing polynomial degree** | Compare CV scores for degree 1, 2, 3, ... |
| **Comparing models** | Ridge vs. Lasso vs. OLS |

**CV variants:**

| Variant | Description | Best For |
|---------|-------------|----------|
| **K-Fold** | Standard, K partitions | General default (K=5 or 10) |
| **LOOCV** | K = n (leave one out) | Small datasets, Ridge (efficient formula) |
| **Stratified K-Fold** | Preserves distribution in each fold | Imbalanced data |
| **Time-Series Split** | Respects temporal ordering | Sequential data |

**Common pitfalls:**
- **Data leakage**: Feature scaling/selection must happen inside each fold
- **Overfitting to CV**: Using test set during model selection
- **Small K**: High bias; Large K: High variance, expensive

---

## Question 37
- [ ] Done

**How do you detect and handle outliers in linear regression analysis?**

**Answer:**

**Detection methods:**

| Method | Type | How It Works | Threshold |
|--------|------|-------------|-----------|
| **Z-score** | Statistical | Standardize values | |z| > 3 |
| **IQR method** | Statistical | Q1 - 1.5×IQR to Q3 + 1.5×IQR | Outside range |
| **Cook's Distance** | Influence | Combined leverage + residual | > 4/n |
| **Studentized Residuals** | Residual-based | Leave-one-out standardized | > ±3 |
| **Leverage (hᵢᵢ)** | Feature-space | Distance from centroid | > 2(p+1)/n |
| **Box plots** | Visual | Quick visual identification | Beyond whiskers |
| **Scatter/residual plots** | Visual | Pattern recognition | Visual judgment |

**Handling strategies:**

| Strategy | When to Use |
|----------|------------|
| **Remove** | Confirmed data errors or irrelevant observations |
| **Winsorize/Cap** | Replace extreme values with percentile limits |
| **Transform** | Log, sqrt to reduce the impact of extremes |
| **Robust regression** | Use Huber, RANSAC, or Theil-Sen estimators |
| **Keep and report** | If legitimate data, run analysis with and without |

**Decision framework:**
1. Is it a **data error**? → Fix or remove
2. Is it from a **different population**? → Remove or model separately
3. Is it a **legitimate extreme**? → Consider robust methods
4. Does it **change conclusions**? → Report sensitivity analysis

---

## Question 38
- [ ] Done

**What are residual plots and how do you use them to validate regression assumptions?**

**Answer:**

Residual plots are diagnostic visualizations of the errors (residuals = actual - predicted) that help validate regression assumptions.

**Key residual plots:**

| Plot | What to Look For | Assumption Tested |
|------|-----------------|-------------------|
| **Residuals vs. Fitted** | Random scatter around zero | Linearity + Homoscedasticity |
| **Q-Q Plot** | Points on 45° diagonal | Normality of residuals |
| **Scale-Location (√|residuals| vs. fitted)** | Flat trend line | Homoscedasticity |
| **Residuals vs. Each Feature** | No patterns | Linearity per feature |
| **Residuals vs. Order** | No pattern over time | Independence |

**Pattern interpretation:**

| Pattern in Residual vs. Fitted | Problem | Solution |
|-------------------------------|---------|----------|
| **Random scatter** | No problem ✓ | None needed |
| **Funnel/fan shape** | Heteroscedasticity | WLS, log-transform target |
| **U-shape or curve** | Non-linearity | Add polynomial terms, transform features |
| **Clusters** | Missing categorical variable | Include the grouping variable |
| **Trend over time** | Autocorrelation | GLS, add time features |

**Implementation:**
```python
import matplotlib.pyplot as plt
import statsmodels.api as sm
fig = sm.graphics.plot_regress_exog(model, 'feature_name', fig=fig)
```

---

## Question 39
- [ ] Done

**How do you test for heteroscedasticity and what are the remedies?**

**Answer:**

**Heteroscedasticity** means the variance of residuals is **not constant** across levels of the predictor — it changes with the fitted values.

**Detection:**

| Test | Type | Null Hypothesis |
|------|------|----------------|
| **Breusch-Pagan test** | Statistical | Constant variance (homoscedasticity) |
| **White's test** | Statistical | Constant variance (more general) |
| **Goldfeld-Quandt test** | Statistical | Equal variance in two subgroups |
| **Residual vs. fitted plot** | Visual | Look for fan/funnel shape |

**Consequences if ignored:**
- OLS estimates are still **unbiased** but no longer **efficient** (not BLUE)
- Standard errors are **wrong** → invalid t-tests, F-tests, confidence intervals
- p-values are unreliable

**Remedies:**

| Remedy | How It Works |
|--------|-------------|
| **Weighted Least Squares (WLS)** | Weight observations inversely by variance |
| **Robust standard errors (HC)** | White's heteroscedasticity-consistent SEs |
| **Log-transform target** | Often stabilizes variance |
| **Box-Cox transformation** | Finds optimal power transformation |
| **Feasible GLS (FGLS)** | Estimate variance function, then apply GLS |

**Implementation:**
```python
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
_, p_value, _, _ = het_breuschpagan(residuals, X)
# p < 0.05 → reject null → heteroscedasticity detected
```

---

## Question 40
- [ ] Done

**What is autocorrelation in regression residuals and how do you address it?**

**Answer:**

**Autocorrelation** (serial correlation) occurs when residuals are **correlated with their lagged values** — common in time-series and spatial data.

**Detection:**

| Test | Method | Interpretation |
|------|--------|---------------|
| **Durbin-Watson (DW)** | Tests for first-order autocorrelation | DW ≈ 2: No autocorrelation; DW < 1.5: Positive; DW > 2.5: Negative |
| **Ljung-Box test** | Tests multiple lags | p < 0.05 → autocorrelation present |
| **ACF/PACF plots** | Visual | Significant spikes indicate correlated lags |
| **Residual vs. order plot** | Visual | Patterns/cycles indicate autocorrelation |

**Consequences:**
- OLS estimates remain unbiased but are **inefficient**
- Standard errors are **underestimated** → inflated t-statistics → false significance
- Predictions may appear better than they actually are

**Remedies:**

| Method | Description |
|--------|-------------|
| **Add lagged variables** | Include y(t-1), x(t-1) as features |
| **Generalized Least Squares (GLS)** | Accounts for the correlation structure |
| **Cochrane-Orcutt** | Iterative GLS procedure for AR(1) errors |
| **Newey-West standard errors** | HAC-consistent standard errors |
| **Differencing** | Use Δy = y(t) - y(t-1) as the target |
| **Time-series models** | ARIMA, ARIMAX for severe autocorrelation |

---

## Question 41
- [ ] Done

**How do you perform feature selection in linear regression models?**

**Answer:**

Feature selection identifies the most relevant predictors, improving model interpretability and reducing overfitting.

**Approaches:**

| Category | Methods | Description |
|----------|---------|-------------|
| **Filter** | Correlation, VIF, mutual information | Evaluate features independently of the model |
| **Wrapper** | Forward selection, backward elimination, RFE | Use model performance to select features |
| **Embedded** | Lasso (L1), Elastic Net | Feature selection during model training |

**Step-by-step workflow:**
1. **Remove constant/near-zero variance features**
2. **Check multicollinearity**: Remove features with VIF > 10
3. **Correlation analysis**: Remove features with |r| < 0.05 with target
4. **Lasso/Elastic Net**: Fit with CV, keep features with non-zero coefficients
5. **Validate**: Cross-validate with selected features

**Comparison:**

| Method | Pros | Cons | Complexity |
|--------|------|------|-----------|
| **Correlation filter** | Fast, simple | Ignores feature interactions | O(p) |
| **Forward selection** | Considers model fit | Greedy, expensive | O(p² × model) |
| **Backward elimination** | Starts with full model | Cannot start if p > n | O(p² × model) |
| **Lasso** | Automatic, handles p > n | May be unstable with correlations | O(model training) |
| **RFE** | Model-specific importance | Computationally expensive | O(p × model) |

---

## Question 42
- [ ] Done

**What are forward selection, backward elimination, and stepwise regression?**

**Answer:**

These are **wrapper-based** feature selection methods that iteratively add or remove features.

**Forward Selection:**
1. Start with **no features**
2. Add the feature that most improves the model (lowest p-value or highest R² increase)
3. Repeat until no feature improves the model significantly
4. Stop when all remaining features have p > threshold (e.g., 0.05)

**Backward Elimination:**
1. Start with **all features**
2. Remove the feature with the highest p-value (least significant)
3. Refit the model
4. Repeat until all remaining features have p < threshold

**Stepwise Regression:**
1. Combines forward and backward at each step
2. Add best feature, then check if any existing feature should be removed
3. More flexible but computationally expensive

**Comparison:**

| Method | Starts With | Direction | Pros | Cons |
|--------|------------|-----------|------|------|
| **Forward** | Empty model | Adds features | Works when p > n | Greedy, misses interactions |
| **Backward** | Full model | Removes features | Considers all features initially | Cannot start if p > n |
| **Stepwise** | Either | Both directions | Most flexible | Expensive, can overfit |

**Modern alternatives:** Lasso, Elastic Net, and RFECV are generally preferred over stepwise methods because they are less prone to overfitting and more computationally efficient.

---

## Question 43
- [ ] Done

**How do you handle missing values in linear regression datasets?**

**Answer:**

| Strategy | Method | When to Use |
|----------|--------|-------------|
| **Deletion** | Listwise (complete case) | MCAR, < 5% missing |
| **Simple Imputation** | Mean, median, mode | Quick baseline |
| **Model-based** | KNN, regression imputation | MAR, captures relationships |
| **Multiple Imputation** | MICE (chained equations) | MAR, gold standard for inference |
| **Indicator Method** | Add binary "is_missing" column | When missingness is informative |

**Best practices:**
1. **Analyze the pattern**: Is it MCAR (random), MAR (depends on observed data), or MNAR (depends on unobserved)?
2. **Fit imputer on training data only** — prevent data leakage
3. **Multiple imputation** creates several imputed datasets, fits model on each, pools results
4. **Never impute the target variable** — drop those rows instead
5. **Consider domain knowledge**: Some missing values have meaning (e.g., no mortgage = "not applicable")

**Implementation:**
```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# KNN Imputation
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X_train)
```

---

## Question 44
- [ ] Done

**What is the bias-variance tradeoff in the context of linear regression?**

**Answer:**

The bias-variance tradeoff describes the fundamental tension between model simplicity and flexibility.

**In linear regression context:**

| Component | Low Complexity (few features, high λ) | High Complexity (many features, low λ) |
|-----------|---------------------------------------|----------------------------------------|
| **Bias** | High (underfitting) | Low (fits training data well) |
| **Variance** | Low (stable predictions) | High (sensitive to training data) |
| **Training error** | High | Low |
| **Test error** | High initially, then decreases | Low initially, then increases |

**Mathematical relationship:**
```
Total Error = Bias² + Variance + Irreducible Noise
```

**How it applies to linear regression variants:**

| Model | Bias | Variance |
|-------|------|----------|
| **Simple OLS (few features)** | Potentially high | Low |
| **Full OLS (all features)** | Low | High |
| **Ridge (λ > 0)** | Slightly increased | Significantly reduced |
| **Lasso (λ > 0)** | Slightly increased | Significantly reduced |
| **High-degree polynomial** | Low (flexible) | Very high |

**Practical implication:**
- The goal is to minimize **total error**, not just bias or variance
- Regularization explicitly trades a small bias increase for a large variance decrease
- Cross-validation estimates the total error to find the optimal complexity

---

## Question 45
- [ ] Done

**How do you interpret confidence intervals and prediction intervals in regression?**

**Answer:**

| Aspect | Confidence Interval (CI) | Prediction Interval (PI) |
|--------|-------------------------|-------------------------|
| **Purpose** | Where the **mean response** lies | Where a **single new observation** will fall |
| **Width** | Narrower | Always wider |
| **Uncertainty** | Only parameter uncertainty | Parameter uncertainty + individual variation |
| **Formula** | ŷ ± t × SE(ŷ) | ŷ ± t × √(SE(ŷ)² + σ²) |
| **Use case** | Average house price in a neighborhood | Price of a specific house |

**Key differences:**
- **CI**: Uncertainty about the regression line position → "Where is the true mean?"
- **PI**: Uncertainty about individual predictions → "Where will the next data point fall?"
- PI is always wider because it includes both estimation error and individual noise (σ²)

**Interpretation rules:**
- A 95% CI: "We are 95% confident the true mean response lies in this interval"
- A 95% PI: "We are 95% confident a new observation will fall in this interval"
- Both intervals are **narrowest at the mean of X** and widen toward the edges

**Practical application:**
- Use **CI** when reporting expected average outcomes
- Use **PI** when making predictions for individual cases (e.g., forecasting)

---

## Question 46
- [ ] Done

**What are the differences between parametric and non-parametric regression approaches?**

**Answer:**

| Aspect | Parametric | Non-Parametric |
|--------|-----------|---------------|
| **Assumption** | Assumes functional form (e.g., linear) | No assumed functional form |
| **Model** | y = β₀ + β₁x + ε | y = f(x) + ε (f is unknown) |
| **Flexibility** | Limited by chosen form | Very flexible |
| **Interpretability** | High (coefficients have meaning) | Lower (black-box nature) |
| **Data requirements** | Works with smaller datasets | Needs more data |
| **Risk** | Model misspecification bias | Overfitting |

**Examples:**

| Parametric | Non-Parametric |
|-----------|---------------|
| Linear regression | Kernel regression |
| Polynomial regression | LOESS/LOWESS |
| Logistic regression | K-nearest neighbors regression |
| GLMs | Random Forest |
| | Gaussian Process regression |

**When to use which:**
- **Parametric**: When theory/domain knowledge suggests a specific relationship, interpretability is needed, or data is limited
- **Non-parametric**: When the relationship is complex/unknown, you have enough data, and prediction accuracy is more important than interpretability

**Semi-parametric compromise:**
- Generalized Additive Models (GAMs): y = β₀ + f₁(x₁) + f₂(x₂) + ...
- Partially linear models: y = Xβ + f(z) + ε

---

## Question 47
- [ ] Done

**How do you implement logistic regression and its relationship to linear regression?**

**Answer:**

Logistic regression is a **classification** algorithm that models the probability of a binary outcome using a linear combination of features passed through the **sigmoid function**.

**Relationship to linear regression:**

| Aspect | Linear Regression | Logistic Regression |
|--------|------------------|-------------------|
| **Target** | Continuous (ℝ) | Binary (0/1) |
| **Output** | Predicted value | Probability P(y=1) |
| **Link function** | Identity: E(y) = Xβ | Logit: log(p/(1-p)) = Xβ |
| **Loss function** | Mean Squared Error (MSE) | Binary Cross-Entropy |
| **Optimization** | OLS (closed-form) | Maximum Likelihood (iterative) |
| **Assumptions** | Normality, homoscedasticity | Independence, linearity in log-odds |

**The logistic model:**
```
P(y = 1|X) = 1 / (1 + e^(-Xβ))
log(P/(1-P)) = β₀ + β₁x₁ + ... + βₙxₙ  (this is linear!)
```

**Key insight:** Logistic regression IS linear — in the **log-odds** space. The sigmoid function squishes the linear output to [0, 1].

**Coefficient interpretation:**
- β₁ = change in **log-odds** for a unit increase in x₁
- e^β₁ = **odds ratio** — multiplicative change in odds

---

## Question 48
- [ ] Done

**What are generalized linear models (GLMs) and how do they extend linear regression?**

**Answer:**

Generalized Linear Models (GLMs) extend linear regression to handle **non-normal** response distributions through a **link function**.

**Components of a GLM:**

| Component | Description | Linear Regression | GLM |
|-----------|-------------|------------------|-----|
| **Random component** | Distribution of Y | Normal | Any exponential family |
| **Systematic component** | Linear predictor | Xβ | Xβ |
| **Link function** | Connects mean to linear predictor | Identity: E(Y) = Xβ | g(E(Y)) = Xβ |

**Common GLMs:**

| Model | Distribution | Link | Use Case |
|-------|-------------|------|----------|
| **Linear Regression** | Normal | Identity | Continuous outcomes |
| **Logistic Regression** | Binomial | Logit | Binary outcomes |
| **Poisson Regression** | Poisson | Log | Count data |
| **Gamma Regression** | Gamma | Inverse/Log | Positive continuous, skewed |
| **Negative Binomial** | Neg. Binomial | Log | Overdispersed counts |

**Why GLMs matter:**
- Linear regression assumes Y ~ Normal; GLMs relax this
- Each distribution has appropriate variance structure
- Link function ensures predictions are in valid range (e.g., log link keeps predictions positive)
- Estimated via **Iteratively Re-weighted Least Squares (IRLS)**

---

## Question 49
- [ ] Done

**How do you handle non-linear relationships in linear regression models?**

**Answer:**

Linear regression assumes a straight-line relationship, but many real relationships are non-linear. Several techniques handle this while staying within the linear regression framework.

**Approaches:**

| Method | How It Works | Example |
|--------|-------------|---------|
| **Polynomial features** | Add x², x³, etc. | Square footage vs. price (diminishing returns) |
| **Log transformation** | log(x) or log(y) | Income vs. spending (multiplicative relationships) |
| **Sqrt/power transforms** | √x, x^(1/3) | Reduce right skew |
| **Interaction terms** | x₁ × x₂ | Effect of A depends on B |
| **Piecewise/spline regression** | Different slopes in different ranges | Temperature vs. energy use |
| **Box-Cox transformation** | Data-driven power transform | Automatic optimal transform |

**Common transformation models:**

| Model | Equation | Interpretation |
|-------|----------|---------------|
| **Log-linear** | log(y) = β₀ + β₁x | 1-unit ↑ in x → β₁×100% change in y |
| **Linear-log** | y = β₀ + β₁log(x) | 1% ↑ in x → β₁/100 change in y |
| **Log-log** | log(y) = β₀ + β₁log(x) | β₁ is the elasticity: 1% ↑ in x → β₁% ↑ in y |

**Decision process:**
1. Check residual plots for non-linear patterns
2. Try log/sqrt transforms based on distribution shape
3. Add polynomial terms if curvature exists
4. Validate improvement with cross-validation

---

## Question 50
- [ ] Done

**What is robust regression and when should you use it instead of OLS?**

**Answer:**

Robust regression methods are **resistant to outliers and violations of assumptions** that would severely affect OLS estimates.

**When to use robust regression:**
- Data contains **outliers** that cannot be removed
- Residuals have **heavy tails** (non-normal)
- **Heteroscedastic** errors resistant to transformation

**Methods:**

| Method | How It Works | Pros | Cons |
|--------|-------------|------|------|
| **Huber Regression** | Squared loss for small residuals, absolute loss for large | Balanced approach | Requires threshold parameter |
| **RANSAC** | Fits model on random inlier subsets | Very robust to outliers | Non-deterministic |
| **Theil-Sen** | Median of slopes between all point pairs | 29% breakdown point | Slow for large n |
| **Least Absolute Deviations (LAD)** | Minimizes sum of absolute residuals | Median regression, robust | No closed-form solution |
| **M-estimators** | General class using robust loss functions | Flexible | Complex implementation |
| **MM-estimators** | High breakdown + high efficiency | Best of both worlds | Complex |

**Breakdown point**: The proportion of outliers a method can handle before giving arbitrarily wrong results. OLS = 0%, Theil-Sen = 29%, LMS/LTS = 50%.

**OLS vs Robust comparison:**
- OLS minimizes Σ(residual²) → one large outlier dominates
- LAD minimizes Σ|residual| → less sensitive to outlier magnitude
- Huber: hybrid approach with tunable threshold

---

## Question 51
- [ ] Done

**How do you assess model performance using different evaluation metrics in regression?**

**Answer:**

| Metric | Formula | Interpretation | Range |
|--------|---------|---------------|-------|
| **R² (Coefficient of Determination)** | 1 - SS_res/SS_tot | Proportion of variance explained | [0, 1] or negative |
| **Adjusted R²** | 1 - (1-R²)(n-1)/(n-p-1) | R² penalized for number of features | ≤ R² |
| **MSE (Mean Squared Error)** | Σ(yᵢ - ŷᵢ)²/n | Average squared error | [0, ∞) |
| **RMSE (Root MSE)** | √MSE | Error in original units | [0, ∞) |
| **MAE (Mean Absolute Error)** | Σ|yᵢ - ŷᵢ|/n | Average absolute error (robust) | [0, ∞) |
| **MAPE** | (100/n)Σ|yᵢ - ŷᵢ|/yᵢ | Percentage error | [0, ∞) |

**When to use which:**

| Metric | Best For |
|--------|---------|
| **R²** | Comparing models on same dataset |
| **Adjusted R²** | Comparing models with different numbers of features |
| **RMSE** | When large errors are especially costly |
| **MAE** | When all errors should be weighted equally (robust to outliers) |
| **MAPE** | When relative error matters more than absolute |

**Important notes:**
- R² can be negative if model is worse than predicting the mean
- RMSE ≥ MAE always (equality only if all errors are equal)
- Use cross-validated metrics for reliable comparison
- Different metrics may rank models differently — choose based on business context

---

## Question 52
- [ ] Done

**What are the computational complexity considerations for large-scale linear regression?**

**Answer:**

| Method | Time Complexity | Space Complexity | Best For |
|--------|----------------|------------------|----------|
| **Normal Equation (OLS)** | O(np² + p³) | O(np + p²) | Small to medium p |
| **QR Decomposition** | O(np²) | O(np) | Numerically stable |
| **SVD** | O(np²) | O(np) | Ill-conditioned matrices |
| **Batch Gradient Descent** | O(np × iterations) | O(np) | Large n, small-medium p |
| **SGD** | O(np) per pass | O(p) | Very large n |
| **Mini-batch GD** | O(batch × p × iters) | O(batch × p) | Large n, parallel hardware |

**Bottleneck analysis:**
- **Normal equation**: p³ matrix inversion is the bottleneck; infeasible for p > ~10,000
- **Gradient descent**: Linear in n and p; scales to millions of features/samples
- **Online/SGD**: O(p) per sample; can handle streaming data

**Scalability strategies:**

| n (samples) | p (features) | Recommended Method |
|-------------|-------------|-------------------|
| Small, small | < 10K, < 1K | Normal equation |
| Large, small | > 100K, < 1K | Normal equation or mini-batch GD |
| Small, large | < 10K, > 10K | SGD with regularization |
| Large, large | > 100K, > 10K | Distributed SGD (Spark) |

---

## Question 53
- [ ] Done

**How do you implement linear regression using gradient descent optimization?**

**Answer:**

Gradient descent is an **iterative optimization** method that finds the optimal coefficients by repeatedly updating them in the direction that reduces the loss function.

**Algorithm:**
1. Initialize coefficients β randomly or to zeros
2. Compute gradient: ∇L = -(2/n) × Xᵀ(y - Xβ)
3. Update: β ← β - α × ∇L (where α is the learning rate)
4. Repeat until convergence

**Key components:**

| Component | Description |
|-----------|-------------|
| **Learning rate (α)** | Step size; too large → diverge, too small → slow convergence |
| **Gradient** | Direction of steepest ascent in loss landscape |
| **Convergence** | Stop when gradient ≈ 0 or loss change < threshold |

**Comparison with Normal Equation:**

| Aspect | Normal Equation | Gradient Descent |
|--------|----------------|-----------------|
| **Formula** | β = (XᵀX)⁻¹Xᵀy | Iterative updates |
| **Complexity** | O(p³) | O(np × iterations) |
| **Feature scaling** | Not needed | Required for fast convergence |
| **Large n** | Works well | Preferred (SGD) |
| **Large p** | Slow (matrix inversion) | Preferred |

**Implementation:**
```python
def gradient_descent(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    beta = np.zeros(n)
    for _ in range(epochs):
        gradient = -(2/m) * X.T @ (y - X @ beta)
        beta -= lr * gradient
    return beta
```

---

## Question 54
- [ ] Done

**What are the differences between batch, mini-batch, and stochastic gradient descent for regression?**

**Answer:**

| Aspect | Batch GD | Mini-Batch GD | Stochastic GD (SGD) |
|--------|----------|--------------|-------------------|
| **Data per update** | All n samples | Random subset (batch_size) | 1 sample |
| **Gradient quality** | Exact | Noisy but reasonable | Very noisy |
| **Convergence** | Smooth, stable | Moderate noise | Oscillatory |
| **Speed per epoch** | Slow (reads all data) | Moderate | Fast |
| **Memory** | O(n × p) | O(batch × p) | O(p) |
| **Parallelizable** | Yes (vectorized) | Yes | Limited |

**When to use which:**

| Method | Best For |
|--------|---------|
| **Batch GD** | Small datasets that fit in memory; clean optima needed |
| **Mini-Batch GD** | Standard choice for medium-large datasets; GPU-friendly |
| **SGD** | Very large / streaming data; online learning scenarios |

**Learning rate considerations:**
- **Batch**: Can use fixed learning rate (stable gradients)
- **Mini-batch**: Fixed or reducing schedule
- **SGD**: Needs **decaying learning rate** (e.g., α_t = α₀/(1 + t×decay)) for convergence

**Mini-batch size selection:**
- Typical: 32, 64, 128, 256
- Smaller batches → more noise → can escape local minima (regularization effect)
- Larger batches → more stable → GPU-efficient
- Powers of 2 align with GPU memory architecture

---

## Question 55
- [ ] Done

**How do you handle high-dimensional data in linear regression (p >> n problem)?**

**Answer:**

When p (features) >> n (samples), the OLS solution is **undefined** because XᵀX is singular (not invertible). Multiple solutions exist that perfectly fit the training data but generalize poorly.

**Solutions:**

| Method | How It Addresses p >> n |
|--------|----------------------|
| **Ridge (L2)** | Makes XᵀX + λI invertible; always has a unique solution |
| **Lasso (L1)** | Selects sparse subset of features; reduces effective p |
| **Elastic Net** | Combines Ridge + Lasso; stable group selection |
| **PCA/PCR** | Reduces dimensionality before regression |
| **PLS** | Finds components maximizing covariance with target |
| **Forward selection** | Greedily adds features up to n |

**Why it's a problem:**
- **Mathematical**: XᵀX is rank n-1 at most → no unique β
- **Overfitting**: Model can perfectly fit training data (zero training error) with infinite variance
- **Interpretability**: Coefficients are arbitrary and unstable

**Practical approach:**
1. **Start with Lasso/Elastic Net** for automatic feature selection
2. **Use cross-validation** to select regularization strength
3. **Consider PCA** if features are highly correlated
4. **Domain knowledge** to pre-filter irrelevant features
5. **Feature importance** from tree-based models as initial screening

---

## Question 56
- [ ] Done

**What is the role of principal component regression (PCR) in dimensionality reduction?**

**Answer:**

Principal Component Regression (PCR) combines PCA with linear regression: first reduce dimensionality, then regress on the principal components.

**Steps:**
1. Standardize features
2. Perform PCA to get principal components (PCs)
3. Select top k PCs (explain sufficient variance)
4. Fit linear regression on the k PCs instead of original features

**Advantages:**

| Benefit | Explanation |
|---------|-------------|
| **Handles multicollinearity** | PCs are orthogonal by construction |
| **Dimensionality reduction** | Fewer features → less overfitting |
| **Handles p > n** | Can use k < n components |
| **Noise reduction** | Drops low-variance components (likely noise) |

**Limitations:**
- PCs are chosen to maximize **variance**, not **predictive power** — a PC with low variance might be highly predictive
- Loses interpretability (PCs are linear combinations of original features)
- PLS regression is often better because it considers the target variable

**PCR vs PLS:**

| Aspect | PCR | PLS |
|--------|-----|-----|
| **PC selection** | Max variance in X | Max covariance between X and y |
| **Target used?** | No (unsupervised) | Yes (supervised) |
| **Components needed** | Often more | Often fewer |
| **Predictive power** | Good | Often better |

---

## Question 57
- [ ] Done

**How do you implement partial least squares (PLS) regression and when is it useful?**

**Answer:**

Partial Least Squares (PLS) regression finds **latent components** that maximize the covariance between X and y, unlike PCA which only maximizes variance in X.

**How PLS works:**
1. Find direction in X-space that has maximum **covariance** with y
2. Project X and y onto this direction → first PLS component
3. Deflate X and y, repeat for next component
4. Regress y on the selected PLS components

**When PLS is useful:**

| Scenario | Why PLS Helps |
|----------|-------------|
| **p >> n** | Reduces features to a few supervised components |
| **Multicollinearity** | Components are orthogonal |
| **Chemometrics/spectroscopy** | Thousands of correlated wavelengths → few components |
| **Genomics** | Thousands of genes → few predictive components |

**PLS vs PCR vs Ridge:**

| Aspect | PLS | PCR | Ridge |
|--------|-----|-----|-------|
| **Supervised?** | Yes | No | Yes |
| **Handles p > n** | Yes | Yes | Yes |
| **Feature selection** | Implicit (via components) | Implicit | No selection |
| **Interpretability** | Moderate (loadings) | Low (PCs) | High (coefficients) |
| **Typical use** | Chemometrics, genomics | General dimensionality reduction | General regularization |

**Implementation:**
```python
from sklearn.cross_decomposition import PLSRegression
pls = PLSRegression(n_components=5)
pls.fit(X_train, y_train)
```

---

## Question 58
- [ ] Done

**What are bayesian approaches to linear regression and their advantages?**

**Answer:**

Bayesian linear regression treats coefficients as **random variables** with prior distributions, updated by the data to produce posterior distributions.

**Bayesian vs Frequentist:**

| Aspect | Frequentist (OLS) | Bayesian |
|--------|------------------|---------|
| **Coefficients** | Fixed unknown values | Random variables with distributions |
| **Output** | Point estimates + CI | Full posterior distribution |
| **Prior knowledge** | Not formally used | Encoded as prior distributions |
| **Uncertainty** | Confidence intervals | Credible intervals (probabilistic) |
| **Regularization** | Explicit penalty (Ridge/Lasso) | Implicit via priors |

**Key concepts:**
- **Prior**: P(β) — belief about coefficients before seeing data
  - Normal prior → equivalent to Ridge regression
  - Laplace prior → equivalent to Lasso regression
- **Likelihood**: P(y|X,β) — how well the model fits data
- **Posterior**: P(β|X,y) ∝ P(y|X,β) × P(β) — updated beliefs after seeing data

**Advantages:**
1. **Quantifies uncertainty** for each coefficient
2. **Incorporates prior knowledge** from domain experts or previous studies
3. **Automatic regularization** through priors
4. **Handles small datasets** better by leveraging prior information
5. **Prediction uncertainty** naturally propagated to predictions

**Implementation:**
```python
from sklearn.linear_model import BayesianRidge
model = BayesianRidge()
model.fit(X_train, y_train)
y_pred, y_std = model.predict(X_test, return_std=True)
```

---

## Question 59
- [ ] Done

**How do you handle time-series data in linear regression models?**

**Answer:**

Time-series data violates the **independence** assumption of standard linear regression. Special handling is required.

**Challenges:**

| Challenge | Description |
|-----------|-------------|
| **Autocorrelation** | Residuals are correlated across time |
| **Trend** | Non-stationary mean over time |
| **Seasonality** | Repeating patterns at fixed intervals |
| **Non-stationarity** | Statistical properties change over time |

**Approaches:**

| Approach | How It Works |
|----------|-------------|
| **Add time features** | Include time index, month, day-of-week as predictors |
| **Lag features** | Use y(t-1), y(t-2), x(t-1) as features |
| **Differencing** | Model Δy = y(t) - y(t-1) to remove trend |
| **Deseasonalize** | Remove seasonal component, model remainder |
| **Fourier features** | sin/cos terms for cyclical patterns |
| **GLS/Cochrane-Orcutt** | Explicitly model error correlation structure |
| **Time-series CV** | Expanding window or rolling window validation |

**Important considerations:**
- Never use standard CV (shuffles time order) — use **TimeSeriesSplit**
- Check for **stationarity** with Augmented Dickey-Fuller test
- Include **enough lag terms** to capture temporal dependencies
- Consider dedicated models (ARIMA, Prophet) for strong temporal patterns

---

## Question 60
- [ ] Done

**What is weighted least squares regression and when should you use it?**

**Answer:**

Weighted Least Squares (WLS) assigns **different weights** to observations, giving more importance to reliable data points.

**Standard OLS:** Minimize Σ(yᵢ - ŷᵢ)²
**WLS:** Minimize Σwᵢ(yᵢ - ŷᵢ)²

**When to use WLS:**

| Scenario | Weight Strategy |
|----------|----------------|
| **Heteroscedasticity** | wᵢ = 1/σᵢ² (inverse of variance) |
| **Grouped data** | wᵢ = group size |
| **Known measurement precision** | wᵢ = instrument precision |
| **Outlier downweighting** | Lower weights for outlying observations |
| **Survey data** | Sampling weights (inverse probability) |

**Process:**
1. Detect heteroscedasticity (Breusch-Pagan test, residual plots)
2. Estimate the variance function: σᵢ² = f(xᵢ)
3. Set weights: wᵢ = 1/σ̂ᵢ²
4. Fit WLS: β_WLS = (XᵀWX)⁻¹XᵀWy

**WLS vs OLS:**
- OLS is **BLUE** under homoscedasticity
- WLS is **BLUE** under heteroscedasticity (when weights are correct)
- Wrong weights can give worse results than OLS
- When true weights are unknown, use **Feasible GLS (FGLS)**: estimate variance, then apply WLS

---

## Question 61
- [ ] Done

**How do you perform hypothesis testing in linear regression (t-tests, F-tests)?**

**Answer:**

**t-tests (individual coefficients):**
- **Purpose**: Test if individual coefficient βᵢ is significantly different from zero
- **Hypotheses**: H₀: βᵢ = 0 vs H₁: βᵢ ≠ 0
- **Test statistic**: t = β̂ᵢ / SE(β̂ᵢ)
- **Interpretation**: |t| > t_critical (or p < 0.05) → reject H₀ → feature is significant

**F-test (overall model significance):**
- **Purpose**: Test if the model as a whole explains significant variance
- **Hypotheses**: H₀: β₁ = β₂ = ... = βₚ = 0 vs H₁: At least one βᵢ ≠ 0
- **Test statistic**: F = (R²/p) / ((1-R²)/(n-p-1))
- **Interpretation**: Large F (or p < 0.05) → model is significant

**Partial F-test (nested model comparison):**
- **Purpose**: Test if a subset of features contributes significantly
- **Hypotheses**: H₀: subset of β = 0
- **Test statistic**: F = ((RSS_reduced - RSS_full)/q) / (RSS_full/(n-p-1))
- Where q = number of dropped features

**Practical guidelines:**
- Always check the **F-test first** — if the model isn't significant overall, individual t-tests are misleading
- Use **adjusted p-values** (Bonferroni, BH) when testing many coefficients
- Low p-value + small coefficient = statistically significant but practically unimportant

---

## Question 62
- [ ] Done

**What are the assumptions required for valid statistical inference in linear regression?**

**Answer:**

For valid statistical inference (reliable p-values, confidence intervals, hypothesis tests), linear regression requires:

| Assumption | Requirement | If Violated |
|------------|------------|-------------|
| **Linearity** | E(Y|X) is linear in X | Biased estimates; meaningless coefficients |
| **Independence** | Observations are independent | Wrong SEs → wrong p-values |
| **Normality** | ε ~ N(0, σ²) | Invalid CIs and p-values (small samples) |
| **Homoscedasticity** | Var(ε) = σ² (constant) | OLS inefficient; wrong SEs |
| **No perfect multicollinearity** | No exact linear dependence among predictors | Cannot estimate coefficients |
| **Exogeneity** | E(ε|X) = 0 (errors uncorrelated with features) | Biased and inconsistent estimates |

**Additional considerations:**
- **No measurement error in X**: Errors-in-variables → attenuation bias
- **Correct specification**: Relevant variables included, irrelevant excluded
- **No influential outliers**: Extreme points can distort inference

**Robustness with large samples:**
- Normality becomes less critical (Central Limit Theorem)
- Heteroscedasticity can be handled with **robust standard errors (HC)**
- Independence and linearity **always** matter regardless of sample size

**Gauss-Markov conditions (for BLUE):**
- Linearity, exogeneity, homoscedasticity, no autocorrelation
- Normality not required for BLUE, only for exact inference in small samples

---

## Question 63
- [ ] Done

**How do you handle correlated errors in regression models?**

**Answer:**

Correlated errors (autocorrelation) occur when the error term at one observation is correlated with errors at other observations, common in time-series and spatial data.

**Types of correlation structures:**

| Type | Description | Example |
|------|-------------|---------|
| **AR(1)** | εₜ = ρ·εₜ₋₁ + uₜ | Most common; adjacent errors correlated |
| **AR(p)** | Depends on p previous errors | Higher-order temporal dependence |
| **MA(q)** | Moving average of past shocks | Short-memory dependence |
| **Spatial** | Nearby locations correlated | Geographic data |

**Solutions:**

| Method | Description | Best For |
|--------|-------------|---------|
| **Generalized Least Squares (GLS)** | Transform to eliminate correlation | Known correlation structure |
| **Cochrane-Orcutt** | Iterative GLS for AR(1) | First-order autocorrelation |
| **Prais-Winsten** | Modified Cochrane-Orcutt | Preserves first observation |
| **Newey-West SEs** | HAC-robust standard errors | Inference only (no efficiency gain) |
| **Differencing** | Model Δy instead of y | Unit root / trend processes |
| **Add lagged variables** | Include y(t-1) as a predictor | Dynamic relationships |

**Key principle:** GLS transforms the model to one with uncorrelated errors, then applies OLS to the transformed model. If Var(ε) = σ²Ω, then GLS uses β̂_GLS = (XᵀΩ⁻¹X)⁻¹XᵀΩ⁻¹y.

---

## Question 64
- [ ] Done

**What is instrumental variable regression and when is it needed?**

**Answer:**

Instrumental Variable (IV) regression addresses **endogeneity** — when a predictor is correlated with the error term, causing OLS to be biased and inconsistent.

**When endogeneity occurs:**

| Cause | Example |
|-------|---------|
| **Omitted variable bias** | Education → earnings, but ability affects both |
| **Measurement error** | True X measured with error → attenuation bias |
| **Simultaneity** | Price affects demand AND demand affects price |
| **Reverse causality** | Health → exercise, but exercise → health |

**Instrument requirements:**
1. **Relevance**: Z must be correlated with the endogenous X (Cov(Z, X) ≠ 0)
2. **Exclusion restriction**: Z must NOT be correlated with ε (Cov(Z, ε) = 0)
3. Z affects Y **only through** X

**Two-Stage Least Squares (2SLS):**
1. **Stage 1**: Regress endogenous X on instrument Z → get predicted X̂
2. **Stage 2**: Regress Y on X̂ (using predicted values instead of actual X)

**Classic examples:**
- **Education → Earnings**: Instrument = distance to college (affects education but not earnings directly)
- **Price → Demand**: Instrument = supply-side cost shifter
- **Smoking → Health**: Instrument = cigarette taxes

**Limitations:**
- Good instruments are hard to find
- Weak instruments → large standard errors, bias toward OLS
- Over-identification tests (Sargan/Hansen) check instrument validity

---

## Question 65
- [ ] Done

**How do you implement and interpret interaction effects in regression models?**

**Answer:**

Interaction effects occur when the effect of one predictor on the outcome **depends on the level of another predictor**.

**Model:** y = β₀ + β₁x₁ + β₂x₂ + β₃(x₁ × x₂) + ε

**Interpretation:**
- β₃ measures how the slope of x₁ changes per unit of x₂
- Effect of x₁ on y: β₁ + β₃·x₂ (varies with x₂)
- Effect of x₂ on y: β₂ + β₃·x₁ (varies with x₁)

**Types of interactions:**

| Type | Example | Model Term |
|------|---------|-----------|
| **Continuous × Continuous** | Experience × Education | x₁·x₂ |
| **Continuous × Categorical** | Salary = Experience + Gender + Experience×Gender | Allows different slopes per group |
| **Categorical × Categorical** | Region × Product type | Different effect per combination |

**Implementation and best practices:**
```python
# Create interaction term
df['exp_edu'] = df['experience'] * df['education']
# Or use sklearn
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True)
```

**Guidelines:**
- Always include **main effects** alongside interactions (hierarchical principle)
- **Center/standardize** variables before creating interactions to reduce multicollinearity
- Test significance with t-test on β₃
- Visualize with **interaction plots** (separate lines for different levels of moderator)

---

## Question 66
- [ ] Done

**What are mixed-effects models and their applications in regression analysis?**

**Answer:**

Mixed-effects models (also called multilevel or hierarchical models) include both **fixed effects** (population-level) and **random effects** (group-level variation).

**Model:** y_ij = (β₀ + u₀ⱼ) + (β₁ + u₁ⱼ)x_ij + ε_ij
- β₀, β₁ = fixed effects (population averages)
- u₀ⱼ = random intercept (group j's deviation from average intercept)
- u₁ⱼ = random slope (group j's deviation from average slope)

**When to use:**

| Scenario | Example |
|----------|---------|
| **Nested/hierarchical data** | Students within schools within districts |
| **Repeated measures** | Multiple measurements per patient over time |
| **Clustered data** | Employees within companies |
| **Longitudinal studies** | Subjects measured at multiple time points |

**Benefits:**

| Benefit | Explanation |
|---------|-------------|
| **Accounts for clustering** | Standard regression assumes independence; mixed models handle within-group correlation |
| **Shrinkage estimation** | Group estimates are pulled toward grand mean (partial pooling) |
| **Handles unbalanced data** | Groups can have different numbers of observations |
| **Appropriate standard errors** | Correct SEs for nested data structure |

**Fixed vs Random effects:**
- **Fixed effect**: Specific, replicable levels (treatment vs control)
- **Random effect**: Sample from a larger population (schools, subjects)

---

## Question 67
- [ ] Done

**How do you perform model diagnostics and residual analysis in linear regression?**

**Answer:**

Model diagnostics systematically check whether regression assumptions hold and identify potential problems.

**Diagnostic checklist:**

| Diagnostic | What to Check | Tool |
|-----------|--------------|------|
| **Linearity** | No pattern in residuals vs. fitted | Residual plot, partial regression plots |
| **Normality** | Residuals follow normal distribution | Q-Q plot, Shapiro-Wilk test |
| **Homoscedasticity** | Constant variance of residuals | Scale-location plot, Breusch-Pagan test |
| **Independence** | No autocorrelation | Durbin-Watson test, residual order plot |
| **Multicollinearity** | Features not highly correlated | VIF analysis, correlation matrix |
| **Influential points** | No single point dominates model | Cook's distance, leverage plot |
| **Outliers** | No extreme residuals | Studentized residuals, box plot |
| **Model specification** | Correct functional form | Ramsey RESET test |

**Residual analysis workflow:**
```
1. Plot residuals vs. fitted values → check linearity + homoscedasticity
2. Q-Q plot of residuals → check normality
3. Scale-location plot → confirm homoscedasticity
4. Residuals vs. leverage → identify influential points
5. ACF plot of residuals → check independence (time-series)
6. Compute VIF → check multicollinearity
```

**After diagnostics, if issues found:**
- Non-linearity → add polynomial/interaction terms
- Heteroscedasticity → WLS or transform target
- Non-normality → transform target (log, Box-Cox)
- Multicollinearity → remove features or use Ridge
- Influential points → investigate, consider robust regression

---

## Question 68
- [ ] Done

**What is the difference between prediction and inference in regression modeling?**

**Answer:**

| Aspect | Prediction | Inference |
|--------|-----------|-----------|
| **Goal** | Accurately predict ŷ for new data | Understand relationship between X and y |
| **Focus** | Minimizing prediction error | Interpreting coefficients and their significance |
| **Model complexity** | Can be complex (ensemble, neural nets) | Prefer simpler, interpretable models |
| **Evaluation** | MSE, RMSE on test data | p-values, confidence intervals, R² |
| **Feature importance** | Less critical which features are used | Central — which features matter and why |
| **Multicollinearity** | Predictions unaffected | Distorts coefficient estimates |
| **Overfitting concern** | Yes — must generalize to new data | Yes — but for different reasons (spurious significance) |

**Key insight with linear regression:**
- For **prediction**: R², RMSE on held-out data matter most; multicollinearity is acceptable if predictions are stable
- For **inference**: Coefficient estimates, p-values, CIs matter; multicollinearity is a serious problem even if R² is high

**Example:**
- Prediction: Hospital readmission risk score — accuracy matters most
- Inference: Does a new drug reduce blood pressure? — coefficient interpretation is everything

**When they conflict:**
- A complex model may predict well but be uninterpretable
- A simple model may be interpretable but predict poorly
- **Causal inference** adds another layer: requires assumptions beyond statistical association

---

## Question 69
- [ ] Done

**How do you handle seasonal patterns and trends in regression analysis?**

**Answer:**

Seasonal and trend patterns must be explicitly modeled or removed to get valid regression results.

**Handling trends:**

| Method | Description |
|--------|-------------|
| **Include time variable** | Add t as a predictor (linear trend) |
| **Polynomial trend** | Add t, t² for curved trends |
| **Differencing** | Model Δy = y(t) - y(t-1) |
| **Detrending** | Subtract estimated trend line |

**Handling seasonality:**

| Method | Description |
|--------|-------------|
| **Dummy variables** | Binary indicators for each season/month (drop one as reference) |
| **Fourier terms** | sin(2πkt/T) and cos(2πkt/T) for period T |
| **Seasonal differencing** | y(t) - y(t-T) removes seasonal component |
| **Decomposition** | STL: Trend + Seasonal + Residual, model residual |

**Combined model:**
```
y(t) = β₀ + β₁t + Σβₖ·Season_k + β_x·X(t) + ε(t)
       \_trend_/  \__seasonality__/  \_other features_/
```

**Fourier approach (for smooth cycles):**
```python
# For period T=12 (monthly seasonality)
df['sin_1'] = np.sin(2 * np.pi * df['month'] / 12)
df['cos_1'] = np.cos(2 * np.pi * df['month'] / 12)
```

**Important:** Use **TimeSeriesSplit** for CV, never random shuffle — respects temporal ordering.

---

## Question 70
- [ ] Done

**What are spline regression and local regression (LOESS) techniques?**

**Answer:**

Both are methods for modeling **non-linear relationships** flexibly within a regression framework.

**Spline Regression:**

| Aspect | Description |
|--------|-------------|
| **Concept** | Piecewise polynomials joined at **knot points** |
| **Types** | Linear splines, cubic splines, natural splines, B-splines |
| **Smoothness** | Cubic splines ensure continuous 1st and 2nd derivatives at knots |
| **Flexibility** | More knots = more flexibility (risk of overfitting) |
| **Advantage** | Local fitting — changes in one region don't affect others |

**LOESS/LOWESS (Local Regression):**

| Aspect | Description |
|--------|-------------|
| **Concept** | Fit weighted regression in a **sliding window** around each point |
| **Weights** | Nearby points get higher weight (tri-cube kernel) |
| **Parameter** | Span (fraction of data in each window): small = flexible, large = smooth |
| **Advantage** | Completely non-parametric, no knots to choose |
| **Disadvantage** | No formula, computationally expensive, O(n²) |

**Comparison:**

| Feature | Splines | LOESS |
|---------|---------|-------|
| **Model equation** | Yes | No (fitted values only) |
| **Prediction on new X** | Direct | Requires re-fitting |
| **Interpretability** | Moderate (knot-based) | Low (black-box) |
| **Speed** | Fast | Slow for large n |
| **Best for** | Modeling + prediction | Exploratory visualization |

---

## Question 71
- [ ] Done

**How do you implement regression with constraints and penalty terms?**

**Answer:**

Constrained regression imposes restrictions on coefficient values during optimization.

**Types of constraints:**

| Constraint Type | Description | Example |
|----------------|-------------|---------|
| **Box constraints** | a ≤ βᵢ ≤ b | Non-negative coefficients |
| **Equality constraints** | Σβᵢ = 1 | Portfolio weights sum to 1 |
| **Inequality constraints** | β₁ ≥ β₂ | Monotonic feature ordering |
| **Regularization penalties** | λ·||β||₁ or λ·||β||₂ | Ridge, Lasso |
| **Smoothness constraints** | Penalties on derivatives | Spline smoothing |

**Penalty-based approaches:**

| Method | Penalty | Effect |
|--------|---------|--------|
| **Ridge** | λΣβⱼ² | Shrink coefficients |
| **Lasso** | λΣ|βⱼ| | Sparse coefficients |
| **Elastic Net** | λ₁Σ|βⱼ| + λ₂Σβⱼ² | Combined |
| **Fused Lasso** | λ₁Σ|βⱼ| + λ₂Σ|βⱼ - βⱼ₋₁| | Sparsity + smoothness |
| **Group Lasso** | λΣ||β_g||₂ | Group sparsity |

**Non-negative regression:**
```python
from scipy.optimize import nnls
beta, residual = nnls(X, y)  # All coefficients ≥ 0
```

**Use cases:**
- Physics-informed models (positive relationships)
- Mixture models (weights must sum to 1)
- Monotonic constraints in dose-response curves

---

## Question 72
- [ ] Done

**What is quantile regression and how does it differ from ordinary regression?**

**Answer:**

Quantile regression models the **conditional quantile** (e.g., median, 10th percentile) instead of the conditional mean.

**OLS vs. Quantile Regression:**

| Aspect | OLS | Quantile Regression |
|--------|-----|-------------------|
| **What it models** | E(Y|X) (mean) | Q_τ(Y|X) (τ-th quantile) |
| **Loss function** | Σ(yᵢ - ŷᵢ)² | Σρ_τ(yᵢ - ŷᵢ) |
| **Sensitivity to outliers** | High (squared loss) | Low (absolute loss variant) |
| **Information** | One line (mean) | Multiple lines at different quantiles |
| **Distribution assumption** | Normality (for inference) | None |

**Check function ρ_τ(u):**
- ρ_τ(u) = τ·u if u ≥ 0, and (τ-1)·u if u < 0
- τ = 0.5 → median regression (Least Absolute Deviations)
- τ = 0.1 → 10th percentile; τ = 0.9 → 90th percentile

**When to use:**

| Scenario | Why Quantile Regression |
|----------|----------------------|
| **Heteroscedastic data** | Different variability at different levels |
| **Skewed distributions** | Mean is not representative |
| **Risk analysis** | Model extreme quantiles (5th, 95th) |
| **Outlier robustness** | Less influenced by extreme values |
| **Full distribution picture** | Understand how effects change across distribution |

**Example:** House prices — median regression is more robust than mean regression because of expensive outliers.

---

## Question 73
- [ ] Done

**How do you handle censored and truncated data in regression models?**

**Answer:**

**Censored data:** Outcome is partially observed — we know it's beyond some limit but not the exact value.
- Example: Study ends at 5 years → survival times > 5 years are right-censored

**Truncated data:** Observations outside a range are **completely excluded** from the sample.
- Example: Only studying employees earning > $50K → salaries ≤ $50K are truncated

**Handling methods:**

| Data Type | Method | Description |
|-----------|--------|-------------|
| **Censored** | **Tobit model** | Latent variable approach; MLE for censored data |
| **Censored** | **Survival models (Cox)** | Hazard-based modeling for time-to-event |
| **Censored** | **Kaplan-Meier + AFT** | Non-parametric survival curves |
| **Truncated** | **Truncated regression** | MLE adjusting for truncation point |
| **Both** | **Heckman selection model** | Two-stage: model selection + outcome |

**OLS problems with censored/truncated data:**
- **Biased estimates** — OLS ignores the censoring/truncation mechanism
- **Inconsistent** — bias doesn't decrease with more data
- **Underestimates variance** — compressed range

**Tobit model:**
```
y* = Xβ + ε           (latent variable)
y = y*  if y* > 0     (observed)
y = 0   if y* ≤ 0     (censored)
```
Estimated via Maximum Likelihood, not OLS.

---

## Question 74
- [ ] Done

**What are the considerations for linear regression in big data environments?**

**Answer:**

| Consideration | Challenge | Solution |
|---------------|-----------|----------|
| **Scalability** | Normal equation O(p³) fails for large p | SGD, mini-batch GD |
| **Memory** | Data doesn't fit in RAM | Streaming/online algorithms, out-of-core |
| **Distributed computing** | Single machine too slow | Spark MLlib, Dask distributed regression |
| **Feature engineering** | Manual feature engineering doesn't scale | Feature stores, automated feature pipelines |
| **Model updates** | Data changes frequently | Online learning, periodic retraining |
| **Regularization** | High-dimensional data (many features) | Lasso/Ridge/Elastic Net with sparse solvers |
| **Data quality** | Noise, missing values at scale | Robust preprocessing pipelines |

**Distributed linear regression:**
```
Each worker: Compute local XᵀX and Xᵀy on data partition
Coordinator: Aggregate XᵀX_total and Xᵀy_total
Solve: β = (XᵀX_total)⁻¹ · Xᵀy_total
```

**Technology stack:**

| Tool | Use Case |
|------|----------|
| **Spark MLlib** | Distributed regression on Hadoop clusters |
| **Dask** | Parallel computation with pandas-like API |
| **Ray** | Distributed computing for ML pipelines |
| **Vowpal Wabbit** | Online learning at massive scale |
| **H2O** | In-memory distributed ML |

---

## Question 75
- [ ] Done

**How do you implement distributed and parallel linear regression algorithms?**

**Answer:**

Distributed algorithms split the computation across multiple machines/cores to handle datasets too large for a single machine.

**Approaches:**

| Approach | How It Works | Best For |
|----------|-------------|---------|
| **Data parallelism** | Split rows across workers, each computes local statistics | Large n (many samples) |
| **Feature parallelism** | Split columns across workers | Large p (many features) |
| **Model parallelism** | Different parts of model on different workers | Very large models |

**Distributed OLS via MapReduce:**
```
Map phase:    Each worker i computes XᵢᵀXᵢ and Xᵢᵀyᵢ on its data partition
Reduce phase: Sum across workers: XᵀX = ΣXᵢᵀXᵢ, Xᵀy = ΣXᵢᵀyᵢ
Final:        β = (XᵀX)⁻¹ · Xᵀy
```

**Distributed SGD:**
```
1. Each worker computes gradient on its mini-batch
2. Gradients are aggregated (averaged)
3. Global model is updated
4. Updated model is broadcast to all workers
```

**Frameworks:**

| Framework | Interface | Notes |
|-----------|-----------|-------|
| **Spark MLlib** | Python/Scala/R | Industry standard; linear regression built-in |
| **Dask-ML** | Python (scikit-learn-like) | Drop-in replacement for sklearn |
| **Horovod** | TensorFlow/PyTorch | Ring-allreduce for gradient aggregation |
| **Parameter Server** | Various | Asynchronous distributed updates |

---

## Question 76
- [ ] Done

**What is online learning and adaptive linear regression for streaming data?**

**Answer:**

Online learning updates the model **incrementally** as new data arrives, without retraining from scratch.

**Batch vs Online:**

| Aspect | Batch Learning | Online Learning |
|--------|---------------|----------------|
| **Training** | All data at once | One sample (or mini-batch) at a time |
| **Data access** | Multiple passes | Single pass |
| **Adaptability** | Retrain from scratch | Continuous updating |
| **Memory** | Stores all data | Stores only model parameters |
| **Use case** | Static datasets | Streaming data, concept drift |

**Online linear regression via SGD:**
```python
from sklearn.linear_model import SGDRegressor

model = SGDRegressor(loss='squared_error')
for X_batch, y_batch in stream:
    model.partial_fit(X_batch, y_batch)
```

**Adaptive methods:**

| Method | Adaptation |
|--------|-----------|
| **SGD with decay** | Learning rate decreases over time |
| **AdaGrad** | Per-feature adaptive learning rates |
| **RMSProp** | Exponential moving average of squared gradients |
| **Adam** | Combines momentum + adaptive rates |
| **Sliding window** | Only use recent data (window size controls memory) |
| **Exponential weighting** | Weight recent data more heavily |

**Concept drift handling:**
- **Sudden drift**: Reset and retrain
- **Gradual drift**: Exponentially weighted updates
- **Recurring drift**: Ensemble of models for different regimes

---

## Question 77
- [ ] Done

**How do you handle non-linear transformations and feature engineering for regression?**

**Answer:**

Feature engineering transforms raw features to capture **non-linear patterns** within the linear regression framework.

**Common transformations:**

| Transform | When to Use | Effect |
|-----------|-------------|--------|
| **Log(x)** | Right-skewed features, multiplicative relationships | Compresses range, ~normalizes |
| **√x** | Count data, moderate skew | Mild compression |
| **x²** | U-shaped relationships | Captures quadratic patterns |
| **1/x** | Reciprocal relationships | Inverse effects |
| **Box-Cox** | Optimal power transformation (data-driven) | Best normalization |
| **Binning** | Non-monotonic relationships | Captures step-wise patterns |

**Feature engineering techniques:**

| Technique | Example |
|-----------|---------|
| **Interaction terms** | x₁ × x₂ (synergy between features) |
| **Polynomial features** | x, x², x³ (curvature) |
| **Ratio features** | price/sqft (derived metrics) |
| **Aggregation features** | Mean, std per category group |
| **Date decomposition** | Year, month, day_of_week, is_weekend |
| **Cyclical encoding** | sin(2π·month/12), cos(2π·month/12) |
| **Target encoding** | Category → mean of target (with regularization) |

**Choosing transformations:**
1. **Visual inspection**: Scatter plots of feature vs. target
2. **Residual analysis**: Non-random patterns suggest missing transforms
3. **Domain knowledge**: Known relationships (e.g., diminishing returns → log)
4. **Statistical tests**: Box-Cox finds the optimal power parameter

---

## Question 78
- [ ] Done

**What are the ethical considerations and fairness issues in regression modeling?**

**Answer:**

| Concern | Description | Impact |
|---------|-------------|--------|
| **Algorithmic bias** | Model systematically discriminates against protected groups | Unfair decisions in hiring, lending, criminal justice |
| **Proxy variables** | Features correlate with protected attributes (zip code → race) | Indirect discrimination |
| **Historical bias** | Training data reflects past discrimination | Model perpetuates inequity |
| **Feedback loops** | Predictions influence future data → reinforce bias | Amplification over time |
| **Simpson's Paradox** | Aggregated results hide subgroup disparities | Misleading conclusions |

**Fairness metrics:**

| Metric | Definition |
|--------|-----------|
| **Demographic Parity** | Prediction rates equal across groups |
| **Equalized Odds** | Equal TPR and FPR across groups |
| **Calibration** | Predicted probabilities are accurate within groups |
| **Individual Fairness** | Similar individuals get similar predictions |

**Mitigation strategies:**

| Stage | Method |
|-------|--------|
| **Pre-processing** | Re-sample or re-weight data to balance representation |
| **In-processing** | Add fairness constraints to the optimization |
| **Post-processing** | Adjust predictions to satisfy fairness criteria |
| **Feature selection** | Remove proxy variables correlated with protected attributes |
| **Transparency** | Report model performance across demographic groups |

**Best practice:** Audit regression models for disparate impact across demographic groups before deployment.

---

## Question 79
- [ ] Done

**How do you implement regression models for causal inference and treatment effects?**

**Answer:**

Standard regression measures **association**, not **causation**. Causal inference requires additional assumptions and methods.

**Key frameworks:**

| Framework | Approach |
|-----------|---------|
| **Randomized Controlled Trials (RCTs)** | Gold standard; random assignment eliminates confounding |
| **Rubin's Potential Outcomes** | Compare outcomes under treatment vs. control |
| **Structural Equation Models** | Specify causal graph, estimate structural parameters |
| **Difference-in-Differences (DiD)** | Compare before/after changes between treatment and control groups |

**Regression-based causal methods:**

| Method | Description | Requirements |
|--------|-------------|-------------|
| **Instrumental Variables (2SLS)** | Use instruments to isolate exogenous variation | Valid instrument |
| **Regression Discontinuity (RD)** | Exploit threshold-based treatment assignment | Sharp cutoff |
| **Propensity Score Matching** | Match treated/control on estimated treatment probability | Overlap, conditional independence |
| **DiD regression** | y = β₀ + β₁·Post + β₂·Treat + β₃·Post×Treat | Parallel trends assumption |
| **Control function** | Include estimated residuals from first stage | Correct first-stage specification |

**Average Treatment Effect (ATE):**
```
ATE = E[Y(1) - Y(0)]
    = E[Y | Treated] - E[Y | Control]  (only if treatment is random)
```

**Key caution:** Regression coefficients are causal ONLY when all confounders are controlled for, which is rarely verifiable from observational data.

---

## Question 80
- [ ] Done

**What is the role of regularization paths and model selection in high-dimensional regression?**

**Answer:**

The regularization path shows how coefficients change as λ varies from 0 (no regularization) to ∞ (all coefficients zero).

**What it reveals:**

| Feature of Path | Interpretation |
|----------------|---------------|
| **Coefficient magnitude** | Importance of feature at each λ |
| **Order of entry/exit** | Most important features enter first (Lasso) |
| **Stability** | Features that persist across λ values are robust |
| **Groups** | Correlated features may track together |
| **Optimal λ** | CV error minimum point on the path |

**Methods for tracing the path:**

| Method | Description | Used By |
|--------|-------------|---------|
| **LARS (Least Angle Regression)** | Computes entire Lasso path efficiently | Lasso |
| **Coordinate descent** | Iterates over λ grid with warm starts | Ridge, Lasso, Elastic Net |
| **Pathwise optimization** | Start at large λ, decrease in small steps | glmnet |

**Model selection along the path:**

| Criterion | Description |
|-----------|-------------|
| **Cross-validation** | Evaluate each λ with k-fold CV |
| **AIC/BIC** | Information criteria (penalize model complexity) |
| **1-SE rule** | Choose largest λ within 1 SE of minimum CV error |
| **Stability selection** | Select features that appear in paths across bootstrap samples |

**Implementation:**
```python
from sklearn.linear_model import lasso_path
alphas, coefs, _ = lasso_path(X, y, alphas=np.logspace(-4, 1, 100))
# Plot: x=log(alpha), y=coefficient values
```

---

## Question 81
- [ ] Done

**How do you handle regression with multiple output variables (multivariate regression)?**

**Answer:**

Multivariate regression models **multiple dependent variables** simultaneously: Y = XB + E, where Y is n×q (q targets).

**Approaches:**

| Method | Description | When to Use |
|--------|-------------|-------------|
| **Separate OLS** | Fit independent regression for each target | Targets are independent |
| **Multivariate OLS** | Joint estimation, same X for all targets | Same model structure, tests for joint effects |
| **Multi-task Lasso** | Shared sparsity pattern across targets | Same features relevant for all targets |
| **Reduced-rank regression** | Low-rank constraint on coefficient matrix | Correlated targets, dimensionality reduction |
| **CCA (Canonical Correlation)** | Find linear combinations maximizing correlation | Exploring X-Y relationships |

**Multi-task learning benefits:**
- **Shared structure**: Common features across targets improve efficiency
- **Regularization**: Learning multiple tasks simultaneously acts as regularization
- **Better with limited data**: Borrowing strength across tasks

**Implementation:**
```python
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge

# Separate models per target
model = MultiOutputRegressor(Ridge(alpha=1.0))
model.fit(X_train, Y_train)  # Y_train has multiple columns

# Or direct multivariate: sklearn Ridge supports multi-target natively
model = Ridge(alpha=1.0)
model.fit(X_train, Y_train)
```

---

## Question 82
- [ ] Done

**What are kernel methods and their applications in regression analysis?**

**Answer:**

Kernel methods enable linear regression to capture **non-linear relationships** by implicitly mapping data to a high-dimensional feature space.

**Key idea:** Instead of computing features φ(x) explicitly, use the **kernel trick**: K(xᵢ, xⱼ) = φ(xᵢ)ᵀφ(xⱼ)

**Common kernels:**

| Kernel | Formula | Best For |
|--------|---------|---------|
| **Linear** | K(x,x') = xᵀx' | Linear relationships |
| **Polynomial** | K(x,x') = (xᵀx' + c)^d | Polynomial relationships |
| **RBF (Gaussian)** | K(x,x') = exp(-γ||x-x'||²) | Complex non-linear patterns |
| **Sigmoid** | K(x,x') = tanh(αxᵀx' + c) | Neural network-like mapping |

**Kernel Ridge Regression:**
- Standard Ridge: β = (XᵀX + λI)⁻¹Xᵀy with complexity O(p³)
- Kernelized: α = (K + λI)⁻¹y with complexity O(n³)
- Prediction: ŷ = Σαᵢ K(xᵢ, x_new)

**When to use kernel regression:**

| Scenario | Use Case |
|----------|---------|
| **Non-linear patterns** | RBF kernel captures any smooth function |
| **n < p** | Kernel form is O(n³) instead of O(p³) |
| **Feature engineering alternative** | Implicitly adds infinite features (RBF) |

**Limitations:**
- O(n³) time and O(n²) space for the kernel matrix
- Harder to interpret than standard linear regression
- Hyperparameter selection (γ, degree) adds complexity

---

## Question 83
- [ ] Done

**How do you implement regression trees and their relationship to linear models?**

**Answer:**

Regression trees partition the feature space into **rectangular regions** and fit a constant (mean of y) in each region.

**How regression trees work:**
1. Select best feature and split point that minimizes MSE
2. Split data into two groups
3. Repeat recursively in each group
4. Stop when depth limit reached or minimum samples per leaf

**Comparison with linear models:**

| Aspect | Linear Regression | Regression Tree |
|--------|------------------|----------------|
| **Decision boundary** | Single hyperplane | Axis-aligned partitions |
| **Non-linearity** | Requires explicit transforms | Automatic |
| **Interactions** | Requires explicit terms | Captured naturally |
| **Interpretability** | Coefficients | Visual tree structure |
| **Extrapolation** | Extends trend beyond data | Flat prediction beyond data |
| **Smoothness** | Smooth predictions | Step-like predictions |
| **Variance** | Low | High (unstable) |

**When trees beat linear models:**
- Complex non-linear relationships with interactions
- Mixed feature types (numerical + categorical)
- Interpretable rule-based decisions needed

**When linear models beat trees:**
- True relationship is approximately linear
- Extrapolation is needed
- Smooth predictions required
- Statistical inference is important

---

## Question 84
- [ ] Done

**What is ensemble regression and how do you combine multiple linear models?**

**Answer:**

Ensemble regression combines multiple linear models to improve prediction accuracy and robustness.

**Methods:**

| Method | Description | How It Combines |
|--------|-------------|----------------|
| **Bagging** | Train multiple models on bootstrap samples | Average predictions |
| **Stacking** | Train multiple diverse models; meta-learner combines | Learned weights |
| **Blending** | Like stacking with holdout instead of CV | Simpler than stacking |
| **Model averaging** | Average predictions from different model types | Equal or weighted average |
| **BMA (Bayesian Model Averaging)** | Weight models by posterior probability | Posterior-weighted average |

**Stacking with linear models:**
```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet

estimators = [
    ('ridge', Ridge(alpha=1.0)),
    ('lasso', Lasso(alpha=0.1)),
    ('enet', ElasticNet(alpha=0.1, l1_ratio=0.5))
]
stack = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge()  # Meta-learner
)
```

**Benefits of ensembling linear models:**
- **Bias reduction**: Different models capture different aspects
- **Variance reduction**: Averaging reduces prediction variance
- **Robustness**: Less sensitive to any single model's weaknesses
- **Uncertainty**: Spread of predictions indicates model uncertainty

**Key principle:** Ensemble works best when base models are **diverse** — different regularization types, different feature subsets, different transformations.

---

## Question 85
- [ ] Done

**How do you handle regression in the presence of measurement errors?**

**Answer:**

Measurement errors in predictors (errors-in-variables) cause **attenuation bias** — coefficients are biased toward zero.

**Types of measurement error:**

| Type | Description | Effect on OLS |
|------|-------------|---------------|
| **Classical (random)** | X_observed = X_true + noise | Attenuation bias (β̂ → 0) |
| **Berkson** | X_true = X_observed + noise | No bias, but larger variance |
| **Systematic** | X_observed = a + b·X_true | Biased, direction depends on a, b |

**Consequences of measurement error in X:**
- β̂_OLS = β_true × Reliability ratio = β_true × Var(X_true)/Var(X_observed)
- Since Reliability ≤ 1, coefficient is **attenuated** (shrunken toward zero)
- R² is underestimated
- More noise → more attenuation

**Solutions:**

| Method | Description | Requirement |
|--------|-------------|-------------|
| **Instrumental Variables** | Use instrument correlated with X_true but not with error | Valid instrument |
| **SIMEX (Simulation Extrapolation)** | Simulate increasing noise, extrapolate back | Known error variance |
| **Reliability correction** | Divide β̂ by reliability ratio | Known reliability ratio |
| **Repeated measurements** | Average multiple measurements to reduce noise | Multiple measures available |
| **Deming regression** | Accounts for errors in both X and Y | Error variance ratio known |
| **Total Least Squares** | Minimizes perpendicular distances | Errors in both variables |

---

## Question 86
- [ ] Done

**What are the considerations for regression model deployment in production systems?**

**Answer:**

| Consideration | Details |
|---------------|---------|
| **Model serialization** | Save model as pickle, joblib, ONNX, or PMML |
| **API serving** | Wrap model in REST API (Flask, FastAPI, TF Serving) |
| **Latency requirements** | Linear regression is fast (<1ms); batch vs. real-time |
| **Input validation** | Schema checks, range validation, missing value handling |
| **Feature pipeline** | Same preprocessing as training (scaling, encoding) |
| **Monitoring** | Track prediction drift, input drift, model performance |
| **Versioning** | Track model versions, data versions, code versions |

**Deployment pipeline:**
```
Data → Preprocessing → Feature Engineering → Model Inference → Output
  ↓        ↓                  ↓                    ↓           ↓
Validation  Same as train    Same transforms     Version tracked  Logged
```

**Key challenges:**

| Challenge | Solution |
|-----------|---------|
| **Training-serving skew** | Use same feature pipeline code for both |
| **Data drift** | Monitor input distributions with PSI, KS test |
| **Model staleness** | Schedule retraining triggers |
| **A/B testing** | Compare new model vs. current in production |
| **Rollback** | Keep previous model version ready for quick switch |

**Best practice:** Use **MLOps frameworks** (MLflow, Kubeflow, SageMaker) for experiment tracking, model registry, and deployment automation.

---

## Question 87
- [ ] Done

**How do you monitor and maintain linear regression models in production?**

**Answer:**

Production model monitoring ensures the model continues to perform well as data and conditions change.

**What to monitor:**

| Category | Metrics | Tools |
|----------|---------|-------|
| **Prediction quality** | RMSE, MAE, R² on new labeled data | Dashboard, alerts |
| **Data drift** | Feature distributions vs. training | PSI, KS test, Evidently AI |
| **Concept drift** | Relationship between features and target changes | Performance decay tracking |
| **Operational** | Latency, throughput, error rates | APM tools (Datadog, Prometheus) |
| **Feature quality** | Missing rates, value ranges, schema violations | Great Expectations, data validation |

**Monitoring workflow:**
```
Production Data → Feature Extraction → Model Prediction → Monitoring
                                                            ↓
                                            Compare with actual outcomes
                                                            ↓
                                              Alert if degradation detected
                                                            ↓
                                              Trigger retraining pipeline
```

**Retraining strategies:**

| Strategy | When to Use |
|----------|------------|
| **Scheduled** | Retrain weekly/monthly (predictable drift) |
| **Triggered** | Retrain when performance drops below threshold |
| **Continuous** | Online learning with constant updates |
| **Champion-Challenger** | Train new model, compare with current before switching |

**Maintenance checklist:**
1. Monitor prediction distributions for anomalies
2. Compare new data distributions with training data
3. Track model accuracy using delayed ground truth labels
4. Retrain when performance degrades beyond acceptable threshold
5. Document model lineage. and version history

---

## Question 88
- [ ] Done

**What is transfer learning and domain adaptation for regression models?**

**Answer:**

Transfer learning adapts a model trained on one domain (source) to perform well on a different but related domain (target).

**In linear regression context:**

| Approach | Description |
|----------|-------------|
| **Coefficient initialization** | Use source model coefficients as starting point for target training |
| **Feature transfer** | Use feature engineering insights from source domain |
| **Domain adaptation** | Adjust for distribution shift between source and target |
| **Multi-task learning** | Train on source and target simultaneously with shared structure |

**Domain adaptation methods:**

| Method | Description |
|--------|-------------|
| **Importance weighting** | Re-weight source samples to match target distribution |
| **Feature alignment** | Transform features to reduce domain gap |
| **Fine-tuning** | Start with source model, retrain on small target data |
| **Instance selection** | Use only source samples similar to target domain |

**When transfer learning helps:**
- Target domain has **limited labeled data**
- Source and target domains share underlying structure
- Feature spaces overlap significantly
- Similar prediction tasks across domains

**Example:** Housing price model trained on City A → adapted for City B
- Start with City A coefficients
- Fine-tune on small City B dataset
- Adjust for domain-specific features (location, taxes, market)

**Limitation:** Negative transfer can occur if domains are too different — source knowledge harms target performance.

---

## Question 89
- [ ] Done

**How do you handle privacy-preserving regression and federated learning?**

**Answer:**

Privacy-preserving regression enables model training without exposing individual data points.

**Methods:**

| Method | Description | Privacy Level |
|--------|-------------|--------------|
| **Differential Privacy** | Add calibrated noise to gradient/statistics | Mathematical guarantee |
| **Federated Learning** | Train on distributed data without central collection | Data stays local |
| **Secure Multi-Party Computation** | Cryptographic protocols for collaborative learning | Strongest guarantee |
| **Homomorphic Encryption** | Compute on encrypted data | No decryption needed |
| **Data Anonymization** | Remove identifying information | Basic, often insufficient |

**Federated Linear Regression:**
```
1. Central server initializes model
2. Each client trains on local data, sends gradient updates (not data)
3. Server aggregates gradients: β_new = β_old - α × mean(∇L_i)
4. Server sends updated model back to clients
5. Repeat until convergence
```

**Differential privacy in regression:**
- Add noise to the sufficient statistics (XᵀX, Xᵀy) before solving
- Noise calibrated to sensitivity and privacy budget (ε)
- Trade-off: more privacy → more noise → less accuracy

**Practical framework:**
- **PySyft**: Python library for federated/privacy-preserving ML
- **TF Federated**: TensorFlow's federated learning framework
- **FATE**: Industrial federated learning platform

**Challenges:**
- Privacy-accuracy tradeoff: stronger privacy = noisier results
- Communication costs in federated learning
- Heterogeneous data across clients (non-IID)

---

## Question 90
- [ ] Done

**What are the interpretability and explainability challenges in complex regression models?**

**Answer:**

| Challenge | Description | Solution |
|-----------|-------------|----------|
| **Black-box perception** | Even linear models can be hard to explain with many features | Feature importance rankings, partial dependence |
| **Multicollinearity** | Correlated features make individual coefficient interpretation unreliable | VIF analysis, PCA, regularization |
| **Interaction effects** | Interactions make interpretation non-additive | Interaction plots, marginal effects |
| **Non-linear transforms** | Log, polynomial transforms complicate interpretation | Back-transform for original-scale interpretation |
| **Regularization effect** | Lasso/Ridge bias coefficients | Report both regularized and OLS estimates |

**Interpretability tools:**

| Tool | What It Shows |
|------|-------------|
| **Coefficient table** | Direct effect per feature (standardized for comparison) |
| **Partial Dependence Plots (PDP)** | Average effect of one feature on prediction |
| **SHAP values** | Contribution of each feature to individual predictions |
| **Feature importance** | Via absolute standardized coefficients or permutation importance |
| **Residual diagnostics** | Model limitations and assumption violations |

**SHAP for linear regression:**
- For linear regression, SHAP values are exactly the feature contributions: SHAP_i = β_i × (x_i - mean(x_i))
- Provides both **local** (per-prediction) and **global** (average) explanations

**Best practices:**
- Report standardized coefficients for feature comparison
- Use partial dependence for non-linear transforms
- Communicate uncertainty (confidence intervals) alongside point estimates

---

## Question 91
- [ ] Done

**How do you implement regression models for real-time prediction and scoring?**

**Answer:**

Real-time prediction requires low-latency model serving with optimized inference pipelines.

**Optimization strategies:**

| Strategy | Description | Latency Impact |
|----------|-------------|---------------|
| **Pre-compute** | Cache predictions for common inputs | <1ms (cache hit) |
| **Model simplification** | Fewer features = faster inference | Proportional to p |
| **Vectorized prediction** | Batch predictions using matrix operations | ~10x faster than loops |
| **Compiled model** | ONNX, TorchScript, C++ inference | 2-10x faster |
| **Approximate computation** | Quantization, feature hashing | Marginal accuracy loss |

**Architecture for real-time scoring:**
```
Client Request → Load Balancer → Feature Service → Model Server → Response
                                      ↓
                              Feature Store (cached)
```

**Linear regression advantages for real-time:**
- Prediction is just a **dot product**: ŷ = Xβ → O(p) per prediction
- No computational graph, no tree traversal
- Extremely fast: microseconds for typical feature counts
- Easy to deploy as a simple function (no neural network framework needed)

**Implementation:**
```python
# Fastest possible: numpy dot product
def predict(features, coefficients, intercept):
    return features @ coefficients + intercept

# Or use ONNX for cross-platform deployment
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
result = session.run(None, {"input": features})
```

---

## Question 92
- [ ] Done

**What is the role of feature importance and variable selection in regression interpretation?**

**Answer:**

Feature importance quantifies **how much each feature contributes** to the model's predictions and helps with model interpretation.

**Methods for linear regression:**

| Method | How It Works | Pros |
|--------|-------------|------|
| **Standardized coefficients** | |β_i × sd(x_i) / sd(y)| | Direct, fast |
| **t-statistic magnitude** | |β_i / SE(β_i)| | Accounts for uncertainty |
| **Reduction in R²** | R² with vs. without feature | Model-specific contribution |
| **Permutation importance** | Shuffle feature, measure performance drop | Model-agnostic, considers interactions |
| **SHAP values** | Game-theoretic attribution | Fair allocation, local + global |

**Standardized coefficient interpretation:**
- Standardize all features to zero mean, unit variance
- |β_standardized| directly ranks feature importance
- "A 1 SD increase in X_i leads to β_i SD change in y"

**Variable selection based on importance:**

| Strategy | Threshold |
|----------|-----------|
| **Statistical significance** | Keep features with p < 0.05 |
| **VIF filtering** | Remove features with VIF > 10 |
| **Lasso zero coefficients** | Remove features set to zero by L1 |
| **Cumulative importance** | Keep top features explaining 95% of importance |

**Important caveats:**
- In the presence of multicollinearity, importance is distributed among correlated features
- Different importance methods may rank features differently
- Feature importance ≠ causal importance
- Always use domain knowledge alongside statistical importance

---

## Question 93
- [ ] Done

**How do you handle regression with imbalanced or skewed target distributions?**

**Answer:**

Skewed target distributions violate the normality of residuals assumption and can lead to heteroscedasticity and poor predictions.

**Detection:**
- Histogram of y — look for asymmetry
- Skewness statistic: |skew| > 1 indicates significant skew
- Q-Q plot of residuals — deviations from diagonal

**Solutions:**

| Strategy | Method | When to Use |
|----------|--------|-------------|
| **Log transform** | y → log(y+1) | Right-skewed targets (common: prices, counts) |
| **Square root** | y → √y | Moderate right skew |
| **Box-Cox** | y → (y^λ - 1)/λ | Optimal data-driven transformation |
| **Yeo-Johnson** | Extends Box-Cox to negative values | Any distribution |
| **Quantile regression** | Model median instead of mean | Heavy tails, outliers |
| **Robust regression** | Huber, RANSAC | Outlier-resistant fitting |
| **GLM** | Use appropriate distribution (Gamma, Poisson) | Known distribution family |

**Log-transform workflow:**
```python
# Transform target
y_log = np.log1p(y)  # log(1+y) handles y=0

# Fit model on transformed target
model.fit(X_train, y_log_train)

# Predictions: back-transform
y_pred = np.expm1(model.predict(X_test))  # exp(pred) - 1
```

**Important note:** When back-transforming log predictions, the mean of exp(predictions) ≠ exp(mean of predictions). Apply **Duan's smearing estimate** or use the correction factor exp(σ²/2).

---

## Question 94
- [ ] Done

**What are the emerging trends and research directions in linear regression?**

**Answer:**

| Trend | Description |
|-------|-------------|
| **Automated ML (AutoML)** | Automated feature engineering, model selection, and hyperparameter tuning for regression |
| **Causal regression** | Integration of causal inference methods with traditional regression (Double ML, causal forests) |
| **Fairness-aware regression** | Constraints ensuring predictions are equitable across demographic groups |
| **Conformal prediction** | Distribution-free prediction intervals with guaranteed coverage |
| **High-dimensional inference** | Debiased Lasso, post-selection inference for valid p-values after variable selection |
| **Online/adaptive regression** | Models that continuously adapt to changing data distributions |
| **Privacy-preserving methods** | Differentially private regression, federated approaches |
| **Neural network integration** | Deep feature extraction + linear prediction head (interpretable last layer) |
| **Robust methods** | Advanced robust regression for contaminated/adversarial data |
| **Quantile regression forests** | Non-parametric quantile regression for uncertainty quantification |

**Notable research directions:**
- **Transfer learning for regression**: Adapting models across domains with limited target data
- **Meta-learning**: Learning to learn regression tasks quickly
- **Heterogeneous treatment effects**: Personalized causal effects using regression frameworks
- **Interpretable ML**: Using linear models as explanations for complex black-box models
- **Graph-based regression**: Incorporating network structure into linear models

**Industry trends:**
- Feature stores for consistent feature engineering
- MLOps for regression model lifecycle management
- Real-time regression scoring at scale

---

## Question 95
- [ ] Done

**How do you implement regression models for anomaly detection and outlier identification?**

**Answer:**

Linear regression can identify anomalies by analyzing **prediction residuals** — observations that deviate significantly from expected values.

**Approaches:**

| Method | How It Works | Anomaly Signal |
|--------|-------------|---------------|
| **Residual-based** | Fit regression, flag large residuals | |residual| > 3σ |
| **Cook's Distance** | Measure influence of each point | Cook's D > 4/n |
| **Studentized residuals** | Standardize residuals (leave-one-out) | |t_i| > 3 |
| **Leverage-based** | Identify unusual feature combinations | h_ii > 2(p+1)/n |
| **Prediction intervals** | Flag observations outside PI | Outside 99% PI |
| **Mahalanobis distance** | Multivariate distance from center | Chi-squared threshold |

**Workflow for anomaly detection:**
```
1. Fit regression model on "normal" training data
2. Predict on new data
3. Calculate residuals: r = y_actual - y_predicted
4. Compute standardized residuals: z = r / σ̂
5. Flag: |z| > threshold (e.g., 3) as anomalies
6. Investigate flagged observations
```

**Applications:**

| Domain | Anomaly Target |
|--------|---------------|
| **Finance** | Fraudulent transactions deviate from spending model |
| **Manufacturing** | Sensor readings deviating from process model |
| **Health** | Patient vitals deviating from expected patterns |
| **Network security** | Traffic patterns deviating from baseline model |

**Advantage of regression approach:** Provides an **explanation** — the anomaly is unusual because feature X predicts value Y but actual is Z.

---

## Question 96
- [ ] Done

**What are the best practices for end-to-end regression modeling pipelines?**

**Answer:**

An end-to-end regression pipeline covers data acquisition through model monitoring.

**Pipeline stages:**

| Stage | Components |
|-------|-----------|
| **1. Data Collection** | Data sources, APIs, databases, streaming |
| **2. Data Validation** | Schema checks, data quality, completeness |
| **3. EDA** | Distribution analysis, correlation, visualization |
| **4. Preprocessing** | Missing values, outlier handling, encoding |
| **5. Feature Engineering** | Transformations, interactions, selection |
| **6. Model Training** | Algorithm selection, hyperparameter tuning, CV |
| **7. Evaluation** | Metrics, residual analysis, assumption checks |
| **8. Deployment** | API serving, batch predictions, model registry |
| **9. Monitoring** | Drift detection, performance tracking, alerts |
| **10. Retraining** | Trigger-based or scheduled retraining |

**Best practices:**

| Practice | Details |
|----------|---------|
| **Reproducibility** | Fix random seeds, version data/code/models |
| **No data leakage** | All transforms fit on train only |
| **Pipeline automation** | Use sklearn Pipeline, MLflow, Airflow |
| **Testing** | Unit tests for transforms, integration tests for pipeline |
| **Documentation** | Model cards, feature dictionaries, decision logs |
| **Version control** | Git for code, DVC for data, MLflow for models |

**Sklearn Pipeline example:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])
pipeline.fit(X_train, y_train)
```

---

## Question 97: Can you discuss the use of spline functions in regression?
- [ ] Done

### Answer

**Definition:**
Spline regression fits piecewise polynomials connected at "knots" to model complex non-linear relationships while maintaining smoothness. It's more flexible than global polynomial regression and avoids edge instability.

**Core Concepts:**
- **Knots:** Points where polynomial pieces connect
- **Piecewise polynomials:** Different polynomial in each interval
- **Smoothness constraints:** Function and derivatives are continuous at knots
- **Cubic splines:** Most common (degree 3), ensures smooth curves

**When to Use Splines vs Polynomial:**

| Situation | Use |
|-----------|-----|
| Smooth non-linear relationship | Splines |
| Local flexibility needed | Splines |
| Simple curve, few data points | Polynomial |
| Edge stability important | Splines |

**Intuition:**
Think of a flexible ruler that bends smoothly through data points. Unlike a single polynomial that wiggles wildly, splines bend locally without affecting distant regions.

**Practical Relevance:**
- Dose-response curves in medicine
- Age-income relationships (non-linear but smooth)
- Any relationship that changes pattern at certain thresholds

**Interview Tip:**
Mention that splines avoid Runge's phenomenon (oscillation at edges) that plagues high-degree polynomials.

---

## Question 98: Discuss recent advances in optimization algorithms for linear regression
- [ ] Done

### Answer

**Definition:**
Beyond classical OLS and gradient descent, recent advances focus on scalability, sparsity, and valid inference in high-dimensional settings.

**Key Advances:**

**1. Stochastic Gradient Descent (SGD) Variants**
- **Adam, AdaGrad:** Adaptive learning rates
- **Mini-batch SGD:** Balance between batch and stochastic
- Use case: Very large datasets that don't fit in memory

**2. Coordinate Descent**
- Optimizes one coordinate (coefficient) at a time
- Very efficient for Lasso/Elastic Net
- Used in scikit-learn's Lasso implementation

**3. Proximal Gradient Methods**
- Handle non-smooth penalties (L1)
- ISTA, FISTA (Fast Iterative Shrinkage-Thresholding)

**4. Adaptive Regularization**
- **Adaptive Lasso:** Different penalties for different coefficients
- Better variable selection consistency

**5. Debiased/Double Machine Learning**
- Valid confidence intervals after Lasso selection
- Combines ML for nuisance parameters with classical inference
- Important for causal inference

**6. Distributed Optimization**
- Split data across machines
- Algorithms: ADMM, distributed SGD
- Use case: Big data environments

**Practical Relevance:**

| Scenario | Recommended Approach |
|----------|---------------------|
| Large n, small p | Normal equation or SGD |
| Large p, small n | Coordinate descent (Lasso) |
| Very large n and p | Distributed SGD |
| Causal inference | Double ML |

---

---

# --- Ridge, Lasso & ElasticNet Questions (from 33_ridge_lasso_elasticnet) ---

# Ridge / Lasso / ElasticNet Regression Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What problem does L2-regularization (Ridge) solve in ordinary least squares?**

**Answer:**

L2-regularization (Ridge) solves several critical problems with ordinary least squares:

**Problems with OLS that Ridge fixes:**

| Problem | How Ridge Helps |
|---------|----------------|
| **Multicollinearity** | OLS coefficients become unstable with correlated features; Ridge stabilizes them by adding lambda to diagonal of X'X |
| **Overfitting** | OLS minimizes training error without constraints; Ridge penalizes large coefficients, reducing variance |
| **Ill-conditioned X'X** | When X'X is near-singular, small data changes cause huge coefficient swings; Ridge makes (X'X + lambda*I) always invertible |
| **p > n** | OLS has no unique solution when features exceed samples; Ridge always has a unique solution |

**Mathematical explanation:**
- OLS: beta = (X'X)^(-1) X'y -- fails when X'X is singular
- Ridge: beta = (X'X + lambda*I)^(-1) X'y -- always invertible since eigenvalues are shifted by lambda

**Eigenvalue perspective:**
- X'X has eigenvalues d1, d2, ..., dp
- (X'X + lambda*I) has eigenvalues d1+lambda, d2+lambda, ..., dp+lambda
- Small eigenvalues (near-singularity) become (small + lambda), stabilizing the inversion
- This shrinks coefficients proportionally more in directions of low variance

---

## Question 2
- [ ] Done

**How does the cost function of Ridge regression differ from that of Lasso?**

**Answer:**

| Aspect | Ridge (L2) | Lasso (L1) |
|--------|-----------|-----------|
| **Cost function** | MSE + lambda * SUM(beta_j^2) | MSE + lambda * SUM(|beta_j|) |
| **Penalty term** | Sum of squared coefficients | Sum of absolute coefficients |
| **Gradient of penalty** | 2*lambda*beta_j (smooth) | lambda*sign(beta_j) (non-smooth at 0) |
| **Solution** | Closed-form: (X'X + lambda*I)^(-1)X'y | No closed-form; iterative optimization |

**Key mathematical differences:**
- Ridge penalty derivative = 2*lambda*beta_j --> proportional to coefficient size --> shrinks gradually, never reaches zero
- Lasso penalty has a subdifferential at zero: [-lambda, +lambda] --> can set coefficients exactly to zero

**Cost functions explicitly:**
```
Ridge: L = SUM(yi - xi'beta)^2 + lambda * SUM(beta_j^2)
Lasso: L = SUM(yi - xi'beta)^2 + lambda * SUM(|beta_j|)
```

**Practical implications:**
- Ridge: All features retained, coefficients shrunk toward zero
- Lasso: Sparse solution -- irrelevant features get exactly zero coefficients
- Ridge: O(p^3) closed-form solution
- Lasso: Requires coordinate descent or LARS (iterative)

---

## Question 3
- [ ] Done

**Explain, mathematically, why Ridge never produces exact zero coefficients.**

**Answer:**

**Mathematical proof:**

Ridge minimizes: L(beta) = ||y - X*beta||^2 + lambda * ||beta||^2

Taking the derivative and setting to zero:
```
dL/d(beta_j) = -2*X_j'*(y - X*beta) + 2*lambda*beta_j = 0
```

At the optimum:
```
beta_j = X_j'*(y - X*beta_{-j}) / (X_j'*X_j + lambda)
```

**Why it never equals zero:**
- The numerator X_j'*(y - X*beta_{-j}) is the correlation between feature j and the current residual
- The denominator (X_j'*X_j + lambda) is always positive (sum of squares + positive lambda)
- beta_j = 0 only if X_j has EXACTLY zero correlation with the residual, which is extremely unlikely in practice

**Contrast with Lasso:**
- Lasso subgradient condition: beta_j = 0 if |X_j'*(y - X*beta_{-j})| < lambda
- Lasso has a "dead zone" of width 2*lambda where coefficients ARE zero
- Ridge has NO such dead zone -- the penalty's smooth quadratic nature means the gradient always pushes coefficients toward zero but never forces them there

**Geometric intuition:** Ridge constraint region (ball) has no corners; the contour can touch the ball at any point. Lasso constraint (diamond) has corners at the axes where coefficients are zero.

---

## Question 4
- [ ] Done

**Why can Lasso be used for feature selection while Ridge usually cannot?**

**Answer:**

**Why Lasso selects features:**
- Lasso's L1 penalty |beta_j| has a **subgradient** at zero: the range [-lambda, +lambda]
- If the gradient of the loss at beta_j=0 falls within this range, the optimum IS at zero
- This creates a "thresholding" effect: weak features are set exactly to zero
- The soft-thresholding operator: beta_j = sign(z_j) * max(|z_j| - lambda, 0)

**Why Ridge cannot:**
- Ridge's L2 penalty beta_j^2 has gradient 2*lambda*beta_j at all points
- At beta_j = 0, the penalty gradient is also 0 -- it provides no "push" toward zero
- The derivative is continuous and smooth; no mechanism to lock coefficients at zero

**Geometric explanation:**

| Lasso | Ridge |
|-------|-------|
| Diamond constraint: |beta_1| + |beta_2| <= t | Ball constraint: beta_1^2 + beta_2^2 <= t |
| Diamond has **corners on axes** | Ball has **no corners** |
| Loss contours likely hit a corner | Loss contours hit smooth surface |
| Corner means one coordinate = 0 | Smooth surface means both non-zero |

**Practical use for feature selection:**
```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)
model.fit(X, y)
selected = [f for f, c in zip(features, model.coef_) if c != 0]
print(f"Selected {len(selected)} of {len(features)} features")
```

---

## Question 5
- [ ] Done

**Derive the closed-form solution for Ridge coefficients.**

**Answer:**

**Derivation:**

Ridge cost function:
```
L(beta) = (y - X*beta)'(y - X*beta) + lambda * beta'*beta
```

Expanding:
```
L = y'y - 2*beta'*X'y + beta'*X'X*beta + lambda*beta'*beta
```

Taking the gradient and setting to zero:
```
dL/d(beta) = -2*X'y + 2*X'X*beta + 2*lambda*beta = 0
             -2*X'y + 2*(X'X + lambda*I)*beta = 0
             (X'X + lambda*I)*beta = X'y
```

**Closed-form solution:**
```
beta_ridge = (X'X + lambda*I)^(-1) * X'y
```

**Key properties:**
- (X'X + lambda*I) is always invertible for lambda > 0 (all eigenvalues > lambda)
- When lambda = 0: reduces to OLS beta = (X'X)^(-1)*X'y
- When lambda -> infinity: beta -> 0 (all coefficients shrink to zero)

**SVD form (more numerically stable):**
If X = U*D*V', then:
```
beta_ridge = V * diag(d_j / (d_j^2 + lambda)) * U'y
```
- Each singular value d_j is shrunk by factor d_j/(d_j^2 + lambda)
- Small d_j (noisy directions) are shrunk more aggressively
- This is why Ridge provides variance reduction

---

## Question 6
- [ ] Done

**Describe coordinate-descent optimization for Lasso.**

**Answer:**

Coordinate descent solves Lasso by optimizing one coefficient at a time while holding others fixed.

**Algorithm:**
```
Initialize beta = 0 (or OLS estimate)
Repeat until convergence:
    For j = 1, 2, ..., p:
        1. Compute partial residual: r_j = y - X_{-j} * beta_{-j}
        2. Compute z_j = X_j' * r_j / n  (simple OLS of r_j on X_j)
        3. Apply soft-thresholding:
           beta_j = sign(z_j) * max(|z_j| - lambda, 0)
```

**Soft-thresholding operator S(z, lambda):**
```
S(z, lambda) = z - lambda    if z > lambda
             = 0             if |z| <= lambda   (THIS is why Lasso zeros out coefficients)
             = z + lambda    if z < -lambda
```

**Why coordinate descent works well for Lasso:**

| Property | Benefit |
|----------|---------|
| **Separable penalty** | L1 penalty decomposes per coordinate: SUM|beta_j| |
| **Cheap per-step** | Only updates one coefficient at a time |
| **Warm starts** | Use solution at lambda_k as initialization for lambda_{k+1} |
| **Sparse updates** | Skip coordinates already at zero (active set strategy) |
| **Convergence** | Guaranteed for convex problems (Lasso is convex) |

**Computational complexity:**
- Per full cycle through all features: O(n*p)
- Typically converges in a few cycles with warm starts
- Much faster than general-purpose solvers for sparse problems

---

## Question 7
- [ ] Done

**What is the geometric intuition behind the L1 vs L2 constraint regions?**

**Answer:**

The geometric intuition comes from viewing regularization as a **constrained optimization** problem.

**Equivalent formulations:**
- Penalized: Minimize MSE + lambda * penalty(beta)
- Constrained: Minimize MSE subject to penalty(beta) <= t

**L1 (Lasso) constraint region: DIAMOND**
```
|beta_1| + |beta_2| <= t
```
- Forms a diamond (rotated square) in 2D
- Has **sharp corners** at the coordinate axes
- In higher dimensions: polyhedron with exponentially many corners

**L2 (Ridge) constraint region: BALL**
```
beta_1^2 + beta_2^2 <= t
```
- Forms a circle (sphere) in 2D
- **Smooth** surface everywhere, no corners
- In higher dimensions: hypersphere

**Why this matters for sparsity:**
- The OLS solution is outside the constraint region
- We find where the **elliptical contours** of the MSE loss first touch the constraint region
- **Diamond**: Most likely touches at a corner (axis) where one coordinate = 0 -> **sparse solution**
- **Circle**: Touches at a smooth point where both coordinates are non-zero -> **non-sparse**

**Higher dimensions (p > 2):**
- L1 diamond has corners along every axis and every combination of axes
- Probability of hitting a corner (sparse point) increases dramatically with p
- This is why Lasso is particularly effective for high-dimensional feature selection

**Elastic Net constraint region:**
- Blend of diamond and circle -> "rounded diamond"
- Corners are softened but still present -> grouped sparsity

---

## Question 8
- [ ] Done

**How does Elastic Net combine the strengths of Ridge and Lasso?**

**Answer:**

Elastic Net combines L1 (Lasso) and L2 (Ridge) penalties:
```
L = MSE + lambda * [rho * SUM|beta_j| + (1-rho)/2 * SUM(beta_j^2)]
```
where rho (l1_ratio) controls the mix: rho=1 -> Lasso, rho=0 -> Ridge.

**How it combines strengths:**

| Property | Lasso Only | Ridge Only | Elastic Net |
|----------|-----------|-----------|-------------|
| **Sparsity** | Yes | No | Yes (from L1 part) |
| **Grouped selection** | Picks one from correlated group | Keeps all, distributes weights | Selects groups together |
| **p > n** | Selects at most n features | No limit | Can select > n features |
| **Stability** | Unstable with correlations | Stable | Stable |
| **Convexity** | Convex | Strictly convex | Strictly convex (due to L2) |

**Why the combination helps:**
1. **L2 component** removes the limitation of Lasso selecting at most n features
2. **L2 component** makes the problem strictly convex -> unique solution
3. **L1 component** provides sparsity (feature selection)
4. **L2 component** encourages correlated features to have similar coefficients before L1 selects or drops the group

**The "grouping effect":**
- For highly correlated features x_i and x_j:
  - Lasso: Arbitrarily picks one, zeros out the other
  - Ridge: Gives similar coefficients to both
  - Elastic Net: |beta_i - beta_j| is bounded -- correlated features are selected/dropped together

---

## Question 9
- [ ] Done

**When would Elastic Net outperform pure Lasso on correlated predictors?**

**Answer:**

Elastic Net outperforms pure Lasso when predictors are **correlated in groups**, because:

**Lasso's problem with correlated predictors:**
1. Among correlated features, Lasso arbitrarily picks ONE and zeros out the rest
2. Which feature gets picked varies across bootstrap samples / CV folds -> **instability**
3. Maximum number of selected features is min(n, p) -- limiting when p >> n

**Elastic Net's advantage:**

| Scenario | Lasso Behavior | Elastic Net Behavior |
|----------|---------------|---------------------|
| **Two identical features** | Picks one randomly | Assigns equal weights to both |
| **Gene expression groups** | Selects one gene per pathway | Selects pathway as a group |
| **Correlated sensors** | Picks one sensor arbitrarily | Uses multiple correlated sensors |

**Theoretical result (Zou & Hastie, 2005):**
- For Elastic Net with rho in (0,1): if x_i = x_j, then beta_i = beta_j
- For Lasso: if x_i = x_j, any split beta_i = a, beta_j = c-a is optimal -> non-unique

**Practical examples where Elastic Net wins:**
- **Genomics**: Genes in same pathway are correlated; Elastic Net selects the pathway
- **Text analysis**: Synonymous words have correlated features
- **Sensor data**: Physical proximity causes correlations
- **Financial features**: Related economic indicators

**Rule of thumb:** If you suspect predictors are correlated in meaningful groups, use Elastic Net with l1_ratio between 0.1 and 0.9 (tuned via CV).

---

## Question 10
- [ ] Done

**Define the hyper-parameters α and λ in sklearn's ElasticNet.**

**Answer:**

In sklearn's ElasticNet, the cost function is:
```
(1 / 2n) * ||y - Xb||^2 + alpha * [l1_ratio * ||b||_1 + (1-l1_ratio)/2 * ||b||_2^2]
```

**Parameters:**

| Parameter | sklearn Name | Role | Range |
|-----------|-------------|------|-------|
| **Overall penalty strength** | `alpha` | Controls total amount of regularization | [0, inf) |
| **L1/L2 mixing** | `l1_ratio` | Balance between Lasso and Ridge | [0, 1] |

**Special cases:**

| l1_ratio | alpha | Resulting Model |
|----------|-------|----------------|
| 1.0 | any | Pure Lasso (L1 only) |
| 0.0 | any | Pure Ridge (L2 only) |
| 0.5 | any | Equal mix of L1 and L2 |
| any | 0.0 | OLS (no regularization) |
| any | inf | All coefficients -> 0 |

**Important notes:**
- `alpha` in sklearn = `lambda` in statistical literature
- `l1_ratio` in sklearn = `rho` in Zou & Hastie's original paper
- glmnet (R) uses `alpha` for mixing and `lambda` for strength -- **reversed naming!**

**Practical tuning:**
```python
from sklearn.linear_model import ElasticNetCV
model = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # Mix ratios
    alphas=np.logspace(-4, 2, 50),               # Penalty strengths
    cv=5
)
model.fit(X, y)
print(f"Best alpha: {model.alpha_}, Best l1_ratio: {model.l1_ratio_}")
```

---

## Question 11
- [ ] Done

**How do you tune λ in practice? List three methods.**

**Answer:**

**Three main methods for tuning lambda:**

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **1. K-Fold Cross-Validation** | Try grid of lambda values, evaluate on held-out folds | Most reliable, standard practice | Computationally expensive |
| **2. Information Criteria (AIC/BIC)** | Penalize log-likelihood by model complexity | Fast, no data splitting needed | Assumes known effective df |
| **3. Bayesian Optimization** | Model CV error as Gaussian Process, sample smartly | Efficient for expensive models | Complex implementation |

**Additional methods:**
- **LOOCV (Leave-One-Out)**: Efficient closed-form for Ridge: LOOCV_error = (1/n) * SUM((y_i - y_hat_i) / (1 - h_ii))^2
- **Generalized Cross-Validation (GCV)**: Approximation to LOOCV without computing hat matrix
- **Validation curve**: Plot train/test error vs lambda visually

**Best practice workflow:**
1. Define lambda grid on **log scale**: logspace(-4, 4, 100)
2. Use LassoCV / RidgeCV / ElasticNetCV (built-in efficient CV)
3. Apply the **one-standard-error rule** for parsimony
4. Validate final model on a completely held-out test set

```python
from sklearn.linear_model import LassoCV
model = LassoCV(alphas=np.logspace(-4, 2, 100), cv=5)
model.fit(X_train, y_train)
```

---

## Question 12
- [ ] Done

**Describe cross-validation for choosing regularization strength.**

**Answer:**

**Process:**
1. Choose a grid of lambda values (log-spaced: 10^-4 to 10^4)
2. For each lambda:
   a. Split data into K folds
   b. For each fold k: train on K-1 folds, evaluate on fold k
   c. Average the K evaluation scores -> CV_error(lambda)
3. Select lambda that minimizes CV_error

**Implementation approaches:**

| Approach | How It Works | Speed |
|----------|-------------|-------|
| **LassoCV** | Uses coordinate descent + warm starts along lambda path | Very fast |
| **RidgeCV** | Efficient LOOCV using matrix formula | O(np^2) |
| **ElasticNetCV** | Grid over both alpha and l1_ratio | Moderate |
| **GridSearchCV** | General-purpose, wraps any estimator | Flexible but slower |

**Key considerations:**
- **Warm starts**: Use solution at lambda_k as starting point for lambda_{k+1} (coordinate descent)
- **Lambda path**: Start from lambda_max (all coefficients zero) and decrease
- **lambda_max for Lasso** = max(|X'y|) / n -- smallest lambda that gives all-zero solution
- **Standardize features** before CV (penalty is scale-dependent)
- **K=5 or K=10** is standard; LOOCV for small datasets

**Pitfalls:**
- Never select lambda on test set -- use nested CV if estimating generalization error
- Feature preprocessing must happen INSIDE the CV loop to prevent data leakage

---

## Question 13
- [ ] Done

**What is the bias–variance trade-off when increasing λ in Ridge?**

**Answer:**

**As lambda increases in Ridge:**

| lambda | Bias | Variance | Training Error | Test Error |
|--------|------|----------|---------------|------------|
| **0 (OLS)** | Lowest | Highest | Lowest | Often high (overfitting) |
| **Small** | Slightly increased | Significantly decreased | Slightly increased | Usually improved |
| **Optimal** | Moderate | Moderate | Moderate | **Minimum** |
| **Large** | High | Very low | High | High (underfitting) |
| **-> infinity** | Maximum (beta -> 0) | Zero | Maximum | Maximum |

**Mathematical perspective:**
- Ridge estimate: beta_ridge = (X'X + lambda*I)^(-1) * X'y
- As lambda increases:
  - Bias: E[beta_ridge] - beta_true != 0 (biased toward zero)
  - Variance: Var(beta_ridge) decreases (more constrained)
  - MSE = Bias^2 + Variance: initially decreases (variance reduction > bias increase), then increases

**Key insight:**
- Ridge trades a small increase in bias for a potentially large decrease in variance
- The total MSE = Bias^2 + Variance can be LOWER than OLS MSE
- This is the Gauss-Markov paradox: biased estimators can have lower MSE than the best unbiased estimator (OLS)

**Practical implication:** There almost always exists a lambda > 0 that gives lower MSE than OLS (lambda=0), especially when features are correlated or p is large relative to n.

---

## Question 14
- [ ] Done

**Show how standardizing predictors affects Ridge/Lasso solutions.**

**Answer:**

**Why standardization matters:**
- Ridge penalty: lambda * SUM(beta_j^2) -- penalizes ALL coefficients equally
- If features have different scales, a feature measured in millions (small beta) is penalized differently than one measured in fractions (large beta)
- Without standardization, the penalty is **scale-dependent** -- features with larger scales are penalized less

**Effect on Ridge:**
```
Unstandardized: feature "salary" (range 30K-200K) vs "age" (range 20-70)
- beta_salary ~ 0.001 (small because feature is large)
- beta_age ~ 500 (large because feature is small)
- Penalty: lambda * (0.001^2 + 500^2) ≈ lambda * 250000
  -> Age coefficient dominates the penalty unfairly!

Standardized: both features have mean=0, std=1
- Both coefficients on comparable scale
- Penalty treats them equally
```

**Effect on Lasso:**
- Same scale issue applies to L1: lambda * SUM|beta_j|
- Additionally, which features get zeroed out depends on scale
- A feature with a large scale may survive (small beta) while a more important small-scale feature gets eliminated

**Best practice:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)  # fit on TRAIN only
X_test_scaled = scaler.transform(X_test)  # transform test with train stats
```

**Important:** The intercept is NOT penalized (sklearn excludes it by default).

---

## Question 15
- [ ] Done

**Why does Lasso sometimes behave erratically when p ≫ n?**

**Answer:**

When p >> n (many more features than samples), Lasso can behave erratically because:

**Problem 1: At most n non-zero coefficients**
- Lasso's solution has at most n non-zero coefficients (mathematical limit)
- If there are more than n truly relevant features, Lasso cannot select them all
- This is a fundamental limitation of L1 regularization's geometry

**Problem 2: Arbitrary selection among correlated features**
- With many correlated features, Lasso picks one arbitrarily
- Different CV folds or random seeds -> different features selected
- The solution is **non-unique** when features are perfectly correlated

**Problem 3: Estimation inconsistency**
- In ultra-high dimensions, standard Lasso may not satisfy the "irrepresentable condition"
- Without this condition, Lasso cannot consistently recover the true support (set of non-zero coefficients)

**Problem 4: Instability of the solution path**
- Small changes in data can cause large changes in which features are selected
- The regularization path may show many features entering/exiting

**Solutions:**

| Solution | How It Helps |
|----------|-------------|
| **Elastic Net** | L2 part removes the n-feature limit and stabilizes selection |
| **Stability selection** | Run Lasso on many bootstrap samples, keep features selected > 50% of the time |
| **Adaptive Lasso** | Use data-driven weights to improve consistency |
| **Sure Independence Screening** | Pre-screen to reduce p before applying Lasso |

---

## Question 16
- [ ] Done

**Explain the idea of "grouped" variable selection in Elastic Net.**

**Answer:**

"Grouped" variable selection means that correlated features are selected or dropped **together** as a group, rather than one being arbitrarily chosen.

**How Elastic Net achieves this:**

The key result (Zou & Hastie, 2005): For Elastic Net with l1_ratio in (0,1):
```
|beta_i - beta_j| <= (1/lambda_2) * sqrt(2n(1 - corr(x_i, x_j))) * ||y||
```
where lambda_2 is the L2 penalty strength.

**This means:**
- Highly correlated features (corr -> 1) have similar coefficients
- The L2 penalty forces correlated features' coefficients to be close
- The L1 penalty then either selects or drops the **entire group** together

**Comparison:**

| Method | With correlated features x1, x2 (corr=0.95) |
|--------|---------------------------------------------|
| **Lasso** | beta1=3.5, beta2=0.0 (or beta1=0.0, beta2=3.5) -- picks one |
| **Ridge** | beta1=1.8, beta2=1.7 -- keeps both with similar values |
| **Elastic Net** | Either beta1=1.9, beta2=1.8 (both selected) or beta1=0, beta2=0 (both dropped) |

**Practical applications:**
- **Genomics**: Genes in the same biological pathway are correlated and should be selected together
- **Sensor arrays**: Spatially close sensors should be grouped
- **Financial indicators**: Related economic metrics should enter/exit together

**Dedicated grouped methods:** Group Lasso explicitly defines groups and applies L1 penalty at the group level.

---

## Question 17
- [ ] Done

**Compare Ridge/Lasso to subset selection in terms of computational cost.**

**Answer:**

| Method | Computational Cost | Scalability |
|--------|-------------------|-------------|
| **Best subset selection** | O(2^p) -- exponential | Infeasible for p > 30 |
| **Forward/backward stepwise** | O(p^2 * model_fit) | Feasible for moderate p |
| **Ridge** | O(np^2 + p^3) closed-form | Fast; limited by matrix inversion |
| **Lasso (coordinate descent)** | O(np * iterations * path_length) | Very scalable with warm starts |
| **LARS (for Lasso path)** | O(np * min(n,p)) | Full Lasso path efficiently |

**Detailed comparison:**

| Aspect | Subset Selection | Ridge/Lasso |
|--------|-----------------|-------------|
| **Optimization** | Combinatorial (NP-hard for best subset) | Continuous, convex optimization |
| **Solution quality** | Global optimum (if solved exactly) | Global optimum (convex) |
| **Approximation** | Greedy approximations (stepwise) | Exact (up to numerical precision) |
| **Model averaging** | No -- hard 0/1 inclusion | Smooth shrinkage (soft thresholding) |
| **Prediction** | Subset often overfits (high variance) | Regularized, lower variance |
| **Statistical properties** | Inference after selection is difficult | Well-studied theoretical properties |

**Practical bottom line:**
- Subset selection is only practical for p < 20-30
- Ridge/Lasso scale to p = millions (with sparse data structures)
- Modern consensus: Regularization is preferred over subset selection for both computational and statistical reasons

---

## Question 18
- [ ] Done

**What is the "oracle property" in the context of Lasso?**

**Answer:**

The **oracle property** means that the estimator performs as well as if an oracle had revealed the true set of non-zero coefficients in advance.

**Formally, an estimator has the oracle property if:**
1. **Consistent variable selection**: It correctly identifies ALL non-zero coefficients AND zeros out ALL truly zero coefficients, with probability approaching 1 as n -> infinity
2. **Optimal estimation rate**: The non-zero coefficients are estimated at the same rate as if OLS were applied to only the true features

**Lasso and the oracle property:**

| Condition | Lasso's Status |
|-----------|---------------|
| **Fixed p, n -> inf** | Has oracle property under **irrepresentable condition** |
| **p >> n** | Does NOT have oracle property in general |
| **Correlated features** | Often fails (irrepresentable condition violated) |

**The irrepresentable condition:**
- The correlation between irrelevant features and relevant features must be bounded:
  ||X_irrelevant' * X_relevant * sign(beta_relevant)||_inf < 1
- When violated, Lasso may include irrelevant features

**Adaptive Lasso (Zou, 2006):**
- Uses weighted L1 penalty: SUM(w_j * |beta_j|) where w_j = 1/|beta_initial_j|^gamma
- DOES have the oracle property under weaker conditions
- Assigns larger penalties to coefficients that are small in initial estimate
- Two-stage: (1) Get initial estimate (OLS or Ridge), (2) Apply weighted Lasso

---

## Question 19
- [ ] Done

**How does the elastic-net mixing parameter ρ influence sparsity?**

**Answer:**

In Elastic Net: penalty = alpha * [rho * ||beta||_1 + (1-rho)/2 * ||beta||_2^2]

where rho (l1_ratio) is the mixing parameter.

**Effect of rho on sparsity:**

| rho (l1_ratio) | L1 Weight | L2 Weight | Sparsity | Description |
|-----------------|-----------|-----------|----------|-------------|
| **1.0** | Full | None | Maximum | Pure Lasso -- most zeros |
| **0.9** | Dominant | Small | High | Near-Lasso with stabilization |
| **0.5** | Equal | Equal | Moderate | Balanced sparsity and grouping |
| **0.1** | Small | Dominant | Low | Near-Ridge with slight sparsity |
| **0.0** | None | Full | None | Pure Ridge -- no zeros |

**How rho influences feature selection:**
- **Higher rho**: More features zeroed out, sparser solution
- **Lower rho**: Fewer features zeroed out, coefficients more evenly distributed
- **The sparsity threshold**: A coefficient is zeroed out when |gradient| < alpha * rho
  - Larger rho means wider "dead zone" -> more zeros

**Interaction with alpha:**
- For fixed alpha, increasing rho increases sparsity
- For fixed rho, increasing alpha increases sparsity
- The effective L1 strength is alpha * rho
- The effective L2 strength is alpha * (1-rho)

**Practical tuning:**
- Tune rho from {0.1, 0.3, 0.5, 0.7, 0.9, 1.0} via CV
- Use ElasticNetCV which optimizes over grid of both alpha and l1_ratio

---

## Question 20
- [ ] Done

**In which scenarios can Ridge regression beat OLS even when n > p?**

**Answer:**

Ridge (lambda > 0) can beat OLS even when n > p in these scenarios:

**1. Multicollinearity:**
- When features are correlated, OLS coefficients have high variance
- Ridge shrinks and stabilizes them, reducing MSE

**2. Many weak predictors:**
- If all p features contribute weakly, OLS overfits by assigning large coefficients to noise
- Ridge shrinks all coefficients, reducing overfitting

**3. High noise-to-signal ratio:**
- When sigma^2 is large relative to the true effect sizes
- A small bias increase from Ridge is offset by a large variance reduction

**4. p close to n:**
- Even with n > p, when p/n ratio is high (e.g., 0.5-0.9), OLS overfits
- Ridge provides critical regularization in this regime

**Theoretical result (Hoerl-Kennard theorem):**
- There ALWAYS exists a lambda > 0 such that MSE(beta_ridge) < MSE(beta_OLS)
- This holds even when the model is correctly specified and n > p
- The optimal lambda depends on the signal-to-noise ratio and the eigenvalues of X'X

**Mathematical intuition:**
```
MSE(Ridge) = Bias^2 + Variance
MSE(OLS)   = 0      + Variance_OLS
```
For small lambda: Bias^2 is negligible while Variance reduction is significant -> Ridge wins.

**Practical rule:** Unless n >> p with low multicollinearity and good signal, Ridge will likely outperform OLS.

---

## Question 21
- [ ] Done

**Discuss multicollinearity and how regularization fixes it.**

**Answer:**

**Multicollinearity:** Two or more predictors are highly correlated, making it difficult to separate their individual effects.

**Effects on OLS:**

| Effect | Consequence |
|--------|------------|
| **Inflated variance** | Var(beta_j) = sigma^2 * (X'X)^(-1)_jj becomes very large |
| **Unstable coefficients** | Small data changes cause large beta swings |
| **Wide confidence intervals** | Low statistical power, non-significant p-values |
| **Sign reversals** | Coefficients may have opposite sign from true effect |
| **Poor interpretability** | Individual feature effects cannot be isolated |

**How regularization fixes multicollinearity:**

| Method | Fix Mechanism |
|--------|-------------|
| **Ridge** | Adds lambda*I to X'X; stabilizes eigenvalues -> stable beta. Distributes effect among correlated features |
| **Lasso** | Selects ONE feature from correlated group, zeros out others -> eliminates redundancy |
| **Elastic Net** | Groups correlated features together; either all selected or all dropped |

**Why Ridge works mathematically:**
```
Var(beta_ridge) = sigma^2 * (X'X + lambda*I)^(-1) * X'X * (X'X + lambda*I)^(-1)
```
- Adding lambda*I increases all eigenvalues by lambda
- Small eigenvalues (source of multicollinearity) are stabilized
- The total variance is provably less than OLS variance

**Detection:**
- VIF > 10 indicates problematic multicollinearity
- Condition number of X'X > 30 suggests ill-conditioning
- Correlation matrix: |r| > 0.8 between features

---

## Question 22
- [ ] Done

**Provide a real example where Lasso harmed model interpretability.**

**Answer:**

**Scenario: Predicting patient readmission risk using clinical features**

A hospital used Lasso to select features from 200 clinical variables. The model selected 15 features with non-zero coefficients.

**How Lasso harmed interpretability:**

| Problem | Example |
|---------|---------|
| **Arbitrary feature selection** | Selected "systolic BP" but dropped "diastolic BP" -- clinically both are important |
| **Instability across folds** | Different CV folds selected different features; doctors couldn't agree on which matter |
| **Missing known important variables** | Dropped "BMI" (correlated with waist circumference which was kept) -- counterintuitive to clinicians |
| **Scale sensitivity** | Lab values measured in different units affected which survived selection |

**Specific interpretability issues:**
1. **Clinical guidelines reference dropped features**: Doctors expected to see certain features based on medical literature
2. **Correlated feature lottery**: Among {blood glucose, HbA1c, fasting insulin}, Lasso picked one -- which one changed with each run
3. **Difficult to explain to stakeholders**: "The model uses waist circumference but not BMI" is hard to justify clinically

**Solutions that preserved interpretability:**
- Use **Elastic Net** to keep correlated clinical features together
- Use **Ridge** and report all coefficients when all features are domain-relevant
- Use **stability selection** and only trust features selected in >80% of bootstrap runs
- Apply **domain constraints**: Force inclusion of clinically mandated features

---

## Question 23
- [ ] Done

**What are potential pitfalls when using Lasso for time-series data?**

**Answer:**

**Pitfalls of Lasso with time-series data:**

| Pitfall | Description |
|---------|-------------|
| **Temporal autocorrelation** | Lasso assumes independent errors; time-series residuals are correlated -> invalid standard errors |
| **Arbitrary lag selection** | Among correlated lagged features (y_{t-1}, y_{t-2}, ...), Lasso arbitrarily picks some and drops others |
| **Non-stationarity** | Lasso doesn't account for trends, seasonality, or structural breaks |
| **Look-ahead bias** | Standard CV shuffles temporal order -> data leakage |
| **Missing temporal structure** | L1 penalty doesn't respect the ordering of lag features |

**Specific issues:**
1. **Lag selection instability**: y_{t-1} and y_{t-2} are often highly correlated; Lasso may keep y_{t-1} and drop y_{t-2} in one fold but reverse in another fold
2. **Gap creation**: Lasso might select lags 1, 3, 7 but not 2, 4-6 -> hard to interpret temporally
3. **Seasonal features**: Monthly dummies are correlated; Lasso may drop some months arbitrarily

**Solutions:**

| Solution | Description |
|----------|-------------|
| **Fused Lasso** | Adds penalty on consecutive coefficient differences: SUM|beta_{t} - beta_{t-1}| |
| **Group Lasso** | Group seasonal dummies or lag blocks |
| **Elastic Net** | L2 component stabilizes lag selection |
| **Time-series CV** | Use expanding/rolling window, never shuffle |
| **Structured penalties** | Custom penalties respecting temporal ordering |

---

## Question 24
- [ ] Done

**Show how Bayesian Ridge relates to L2-regularization.**

**Answer:**

Bayesian Ridge regression is the **Bayesian interpretation** of L2 regularization, where the Ridge penalty emerges naturally from a Gaussian prior on coefficients.

**Bayesian formulation:**
```
Prior:      beta ~ N(0, sigma_beta^2 * I)      (Gaussian prior, mean 0)
Likelihood: y|X,beta ~ N(X*beta, sigma^2 * I)  (Normal errors)
Posterior:  beta|X,y ~ N(mu_post, Sigma_post)   (Also Gaussian)
```

**Connection to Ridge:**
```
MAP estimate (mode of posterior) = Ridge solution

beta_MAP = argmax log P(beta|X,y)
         = argmax [log P(y|X,beta) + log P(beta)]
         = argmin [||y - X*beta||^2/(2*sigma^2) + ||beta||^2/(2*sigma_beta^2)]
         = argmin [||y - X*beta||^2 + lambda * ||beta||^2]

where lambda = sigma^2 / sigma_beta^2
```

**Key insight:** lambda = sigma^2 / sigma_beta^2
- **Large sigma_beta^2** (vague prior, we believe coefficients can be large) -> small lambda -> less regularization
- **Small sigma_beta^2** (strong prior toward zero) -> large lambda -> more regularization

**Bayesian Ridge advantages over standard Ridge:**
1. Automatic lambda selection via evidence maximization (empirical Bayes)
2. Full posterior distribution -> uncertainty quantification
3. Credible intervals for each coefficient
4. Posterior predictive distribution for new observations

```python
from sklearn.linear_model import BayesianRidge
model = BayesianRidge()  # Automatically tunes lambda
model.fit(X, y)
y_pred, y_std = model.predict(X_test, return_std=True)
```

---

## Question 25
- [ ] Done

**How would you extend Lasso to generalized linear models?**

**Answer:**

Extending Lasso to GLMs applies L1 regularization to generalized linear models with non-normal response distributions.

**Standard Lasso (linear):**
```
Minimize: (1/2n)||y - X*beta||^2 + lambda * ||beta||_1
```

**GLM + Lasso (penalized GLM):**
```
Minimize: -log L(beta; y, X) + lambda * ||beta||_1
```
where log L is the log-likelihood of the GLM family (Bernoulli, Poisson, Gamma, etc.)

**Common penalized GLMs:**

| GLM | Distribution | Link | Application |
|-----|-------------|------|-------------|
| **Logistic Lasso** | Binomial | Logit | Sparse classification |
| **Poisson Lasso** | Poisson | Log | Sparse count models |
| **Cox Lasso** | Survival | Partial likelihood | Sparse survival analysis |
| **Gamma Lasso** | Gamma | Log/Inverse | Sparse positive continuous |

**Implementation approach -- IRLS + Coordinate Descent:**
1. Penalized IRLS: At each iteration, form a weighted least squares problem
2. Apply coordinate descent with L1 penalty to the working response
3. This is the approach used by **glmnet**

**In Python:**
```python
from sklearn.linear_model import LogisticRegression
# L1-penalized logistic regression
model = LogisticRegression(penalty='l1', solver='saga', C=1/lambda_val)

# Using glmnet-python for general penalized GLMs
# pip install glmnet
from glmnet import LogitNet, ElasticNet
```

**Key consideration:** The intercept should NOT be penalized in GLMs (same as linear case).

---

## Question 26
- [ ] Done

**Explain warm-starts in coordinate descent for Lasso paths.**

**Answer:**

**Warm starts** use the solution at one lambda value as the **initialization** for solving at the next lambda value, dramatically speeding up the Lasso path computation.

**How it works:**
```
Lambda path: lambda_max > lambda_{k-1} > lambda_k > ... > lambda_min

Step 1: Solve Lasso at lambda_max (trivial: all beta = 0)
Step 2: Solve at lambda_{k-1}, starting from beta(lambda_max) = 0
Step 3: Solve at lambda_k, starting from beta(lambda_{k-1})  <- WARM START
...
Continue decreasing lambda, always starting from previous solution
```

**Why warm starts help:**

| Aspect | Cold Start | Warm Start |
|--------|-----------|-----------|
| **Initialization** | beta = 0 or random | beta = solution at nearby lambda |
| **Iterations needed** | Many (far from solution) | Few (very close to solution) |
| **Active set** | Must discover from scratch | Inherit from previous (most features stay zero) |
| **Total path time** | O(grid_size * full_solve) | O(grid_size * few_iterations) |

**Active set strategy (speeds it further):**
1. Most coefficients at lambda_k are zero (inherited from lambda_{k-1})
2. Only optimize over the **active set** (non-zero coefficients + a few candidates)
3. After convergence on active set, check KKT conditions for all features
4. If any violated, add to active set and re-optimize

**In practice:**
- Computing the full Lasso path over 100 lambda values with warm starts is almost as fast as solving at a SINGLE lambda value from scratch
- This is why LassoCV and glmnet are so efficient

---

## Question 27
- [ ] Done

**Compare glmnet (R) and sklearn (Python) implementations of Elastic Net.**

**Answer:**

| Aspect | glmnet (R) | sklearn (Python) |
|--------|-----------|-----------------|
| **Algorithm** | Coordinate descent with warm starts | Coordinate descent with warm starts |
| **Parameter naming** | alpha = mixing (0=Ridge, 1=Lasso); lambda = penalty strength | l1_ratio = mixing; alpha = penalty strength |
| **Default scaling** | Standardizes internally, returns original-scale coefficients | User must standardize manually (or use pipeline) |
| **Lambda sequence** | Auto-generates lambda path (lambda_max to lambda_min) | User specifies alphas list or uses CV variants |
| **Cross-validation** | cv.glmnet() with built-in lambda.min and lambda.1se | ElasticNetCV with alphas parameter |
| **Intercept fitting** | Always fits intercept (unpenalized) | fit_intercept=True by default |

**Detailed differences:**

| Feature | glmnet (R) | sklearn (Python) |
|---------|-----------|-----------------|
| **Cost function** | (1/2n)||y-Xb||^2 + lambda[alpha||b||_1 + (1-alpha)||b||_2^2/2] | (1/2n)||y-Xb||^2 + alpha[l1_ratio||b||_1 + (1-l1_ratio)||b||_2^2/2] |
| **Convergence criterion** | Fractional change in deviance | Dual gap |
| **Exact zeros** | Stored in sparse format | Returns dense coefficient array |
| **GLM support** | Full family (Gaussian, Binomial, Poisson, Cox, Multinomial) | Separate classes (Lasso, LogisticRegression, etc.) |
| **1-SE rule** | Built-in lambda.1se | Must implement manually |
| **Predict at specific lambda** | predict(model, s=lambda, newx=X) | Must refit or interpolate |
| **Speed** | Fortran backend, very fast | Cython backend, competitive |

**Tip:** When translating between the two:
- glmnet's alpha = sklearn's l1_ratio
- glmnet's lambda = sklearn's alpha

---

## Question 28
- [ ] Done

**What is "adaptive Lasso"? Describe its two-stage procedure.**

**Answer:**

**Adaptive Lasso** (Zou, 2006) uses **data-driven weights** on the L1 penalty to achieve the oracle property.

**Standard Lasso:**
```
Minimize: MSE + lambda * SUM(|beta_j|)     (equal weight on all coefficients)
```

**Adaptive Lasso:**
```
Minimize: MSE + lambda * SUM(w_j * |beta_j|)
where w_j = 1 / |beta_initial_j|^gamma,    gamma > 0
```

**Two-stage procedure:**

| Stage | Action | Details |
|-------|--------|---------|
| **Stage 1** | Get initial estimates | Fit OLS, Ridge, or another consistent estimator to get beta_initial |
| **Stage 2** | Weighted Lasso | Use w_j = 1/|beta_initial_j|^gamma as penalty weights |

**Why it works:**
- Large initial beta -> small weight -> less penalized -> likely kept
- Small initial beta -> large weight -> heavily penalized -> likely zeroed out
- This data-adaptive weighting helps consistently identify the true model

**Properties:**
1. **Oracle property**: Under mild conditions, adaptive Lasso consistently selects the true model AND estimates non-zero coefficients at the optimal rate
2. **Less restrictive conditions**: Doesn't need the "irrepresentable condition" that standard Lasso requires

**Implementation:**
```python
from sklearn.linear_model import Ridge, Lasso
# Stage 1: Initial estimate
ridge = Ridge(alpha=1.0).fit(X, y)
# Stage 2: Weighted Lasso
weights = 1.0 / (np.abs(ridge.coef_) + 1e-6)  # add epsilon for stability
X_weighted = X / weights  # Equivalent to weighted penalty
lasso = Lasso(alpha=lambda_val).fit(X_weighted, y)
beta_adaptive = lasso.coef_ / weights
```

---

## Question 29
- [ ] Done

**Discuss limitations of Lasso with highly correlated true signals.**

**Answer:**

When the true signals are **highly correlated**, Lasso has several well-documented limitations:

**Problem 1: Arbitrary selection**
- Among k perfectly correlated features, Lasso picks one and zeros out the rest
- WHICH one is picked is arbitrary and unstable
- Example: If x1, x2 are identical, Lasso assigns (a, 0) or (0, a) or any (c, a-c) -- non-unique

**Problem 2: Coefficient magnitude bias**
- The selected coefficient absorbs the combined effect, making it inflated
- If we remove x1, beta_2 changes dramatically -> instability

**Problem 3: Prediction degradation**
- Selecting only one of several correlated true signals throws away redundant but useful information
- Prediction variance increases compared to using all correlated signals

**Quantitative result:**
- If corr(x1, x2) = rho and both are truly predictive:
  - Lasso selects one, assigns it beta1 + beta2
  - Effective prediction: (beta1 + beta2) * x1 instead of beta1*x1 + beta2*x2
  - Works well only if x1 ≈ x2 (high correlation)

**Solutions:**

| Solution | How It Helps |
|----------|-------------|
| **Elastic Net** | Groups correlated signals -> selects or drops group together |
| **Group Lasso** | Explicitly define groups; L1 at group level |
| **Stability selection** | Bootstrap + Lasso; keep features stable across samples |
| **Ridge** | Distributes weight evenly (but no sparsity) |
| **Pre-clustering** | Cluster correlated features, select representative per cluster, then apply Lasso |

---

## Question 30
- [ ] Done

**Explain "cross-validation one-standard-error rule" for λ selection.**

**Answer:**

The **one-standard-error (1-SE) rule** selects the most regularized (simplest) model whose CV error is within one standard error of the minimum CV error model.

**Procedure:**
```
1. Compute CV error for each lambda: CV(lambda_k) and SE(CV(lambda_k))
2. Find lambda_min = argmin CV(lambda_k)
3. Compute threshold = CV(lambda_min) + SE(CV(lambda_min))
4. Select lambda_1se = largest lambda such that CV(lambda) <= threshold
```

**Rationale:**

| lambda_min | lambda_1se |
|-----------|-----------|
| Minimizes CV error exactly | Slightly higher CV error (within noise) |
| More complex model | Simpler, more parsimonious model |
| More features selected | Fewer features selected |
| Potentially overfit to CV noise | More robust to CV variability |

**Why it works:**
- CV errors have uncertainty (standard errors from K folds)
- Two models within 1 SE of each other are **statistically indistinguishable**
- Among equivalent models, choose the **simplest** one (Occam's razor)
- More regularization (larger lambda) -> fewer coefficients -> better interpretability

**Visual interpretation:**
```
CV Error
  |     *
  |    * *
  |   *   *
  |  *     *  ← threshold = min + 1 SE
  |  *     *
  | *       *
  |*   min   *
  +-------------- lambda
      ↑     ↑
   lambda_min  lambda_1se (simpler)
```

**In practice:**
- glmnet (R) provides both lambda.min and lambda.1se automatically
- In sklearn, manually compute: `best_se = np.std(cv_scores[best_idx]) / sqrt(K)`

---

## Question 31
- [ ] Done

**How does Ridge handle categorical variables encoded via one-hot?**

**Answer:**

One-hot encoding creates multiple binary columns for one categorical variable. Ridge handles these in specific ways:

**The issue:**
- A categorical variable with k levels -> k-1 dummy variables (or k without intercept)
- Ridge penalizes each dummy coefficient independently
- This penalizes categorical variables **more** than continuous variables (more coefficients to penalize)

**Example:**
```
"Region" with 5 levels -> 4 dummy variables
"Temperature" -> 1 continuous variable
Ridge penalty on Region: lambda * (beta_R2^2 + beta_R3^2 + beta_R4^2 + beta_R5^2)
Ridge penalty on Temperature: lambda * beta_T^2
-> Region is penalized ~4x more!
```

**Solutions:**

| Solution | Description |
|----------|-------------|
| **Group Ridge** | Apply separate penalty per group of dummies: lambda_1 * ||beta_region||^2 + lambda_2 * ||beta_temp||^2 |
| **Penalty factor adjustment** | Scale penalty for dummy groups by 1/sqrt(k-1) so total penalty per variable is comparable |
| **Target encoding** | Replace categorical with single numeric (mean of target) -- eliminates the issue |
| **Standardization** | Standardize dummies (mean=0, sd) to partially equalize penalty |

**In sklearn:**
- No built-in support for group penalties
- Workaround: Scale dummy columns by sqrt(k-1) before fitting Ridge
- Or use glmnet (R) which has `penalty.factor` argument

**Lasso's behavior:**
- May zero out some but not all dummies -> partial level elimination (hard to interpret)
- Group Lasso is specifically designed to handle this: zeros out entire groups

---

## Question 32
- [ ] Done

**Why can Lasso under-select in presence of grouped predictors?**

**Answer:**

**The problem:** When features come in groups (e.g., dummy variables for one categorical, or genes in a pathway), Lasso may select **some but not all** members of a group.

**Why Lasso under-selects in grouped predictors:**

1. **L1 geometry**: The L1 constraint has corners on individual coordinate axes, not on group-level axes
   - Solution tends to hit a corner where one feature is zero
   - Within a group of correlated features, only one may survive

2. **Correlated features compete**: Among correlated group members, Lasso sees redundancy
   - Keeping one captures most of the signal
   - Additional members add little marginal improvement relative to their penalty cost

3. **Maximum selection limit**: Lasso selects at most n features total
   - If groups are large, this limit prevents selecting all group members

**Example:**
```
Group: {gene_A1, gene_A2, gene_A3} in pathway A (corr ≈ 0.9)
Lasso result: gene_A1 = 0.5, gene_A2 = 0.0, gene_A3 = 0.0
Desired: All three selected (they're all relevant to the pathway signal)
```

**Solutions:**

| Method | Description |
|--------|-------------|
| **Group Lasso** | L1 penalty at group level: SUM||beta_group||_2 -> all-in or all-out per group |
| **Elastic Net** | L2 part encourages similar coefficients within correlated groups |
| **Sparse Group Lasso** | Group Lasso + within-group L1 -> group selection + within-group sparsity |
| **Pre-define groups** | Domain knowledge to define feature groups before modeling |

---

## Question 33
- [ ] Done

**When does Elastic Net degenerate to Ridge or Lasso?**

**Answer:**

Elastic Net cost function: MSE + alpha * [l1_ratio * ||beta||_1 + (1-l1_ratio)/2 * ||beta||_2^2]

**Degeneration conditions:**

| l1_ratio Value | Result | Mathematically |
|---------------|--------|---------------|
| **l1_ratio = 1.0** | Pure Lasso | L2 term disappears: only L1 penalty remains |
| **l1_ratio = 0.0** | Pure Ridge | L1 term disappears: only L2 penalty remains |
| **alpha = 0** | OLS | Both penalty terms disappear |
| **alpha -> infinity** | All-zero model | Both L1 and L2 force all beta -> 0 |

**Intermediate behavior:**

| l1_ratio Range | Behavior |
|---------------|----------|
| **0.0 < l1_ratio < 0.1** | Near-Ridge: minimal sparsity, strong shrinkage, grouped coefficients |
| **0.1 < l1_ratio < 0.5** | Mixed: moderate sparsity with grouping effect |
| **0.5 < l1_ratio < 0.9** | Near-Lasso: significant sparsity, some grouping stability |
| **0.9 < l1_ratio < 1.0** | Near-Lasso: strong sparsity with slight L2 stabilization |

**When does it NOT degenerate?**
- The interesting regime is l1_ratio in (0, 1) where BOTH penalties are active
- This is where Elastic Net's unique "grouped selection" emerges
- The strictly convex L2 part ensures a unique solution (unlike Lasso which may have multiple optima)

**Practical note:** Usually l1_ratio in {0.1, 0.3, 0.5, 0.7, 0.9} covers the useful range; extreme values 0 and 1 reduce to known models.

---

## Question 34
- [ ] Done

**Show how to plot the coefficient path as a function of λ.**

**Answer:**

**Coefficient path plot** shows how each coefficient evolves as lambda changes.

**Implementation:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, ridge_path

# Lasso path
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, alphas=np.logspace(-4, 1, 100))

plt.figure(figsize=(10, 6))
for i in range(coefs_lasso.shape[0]):
    plt.plot(np.log10(alphas_lasso), coefs_lasso[i], label=f'Feature {i}')
plt.xlabel('log10(lambda)')
plt.ylabel('Coefficient value')
plt.title('Lasso Coefficient Path')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.show()
```

**How to read the plot:**

| Feature of Path | Interpretation |
|----------------|---------------|
| **Coefficients at right (large lambda)** | All near zero (heavy regularization) |
| **Coefficients at left (small lambda)** | Near OLS values (minimal regularization) |
| **First feature to become non-zero** | Most important feature |
| **Features that stay zero until very small lambda** | Least important / noise features |
| **Vertical line at optimal lambda** | Selected model (from CV) |

**Ridge vs Lasso paths:**
- **Lasso**: Features enter one at a time; discrete "entry" events; path is piecewise linear
- **Ridge**: All features are always non-zero; smooth continuous shrinkage toward zero

**Use case:** Understanding feature importance ranking, identifying stable features, and selecting lambda visually.

---

## Question 35
- [ ] Done

**Discuss computational complexity of coordinate descent vs LARS.**

**Answer:**

| Algorithm | Per-Iteration Cost | Path Cost | Memory |
|-----------|-------------------|-----------|--------|
| **Coordinate Descent** | O(n) per coordinate = O(np) per full cycle | O(np * iters * grid_size) with warm starts | O(np) |
| **LARS** | O(np * min(n,p)) for full Lasso path | O(np * min(n,p)) total | O(np + p^2) |

**Detailed comparison:**

| Aspect | Coordinate Descent | LARS |
|--------|-------------------|------|
| **How it works** | Optimize one coordinate at a time, cycle through all | Add one feature at a time along equiangular direction |
| **Path computation** | Grid of lambda values with warm starts | Exact breakpoints along the path |
| **Number of steps for full path** | grid_size * few_cycles_each | Exactly min(n, p) steps (or until all features enter) |
| **Sparse data** | Excellent (skips zero features) | Not optimized for sparse data |
| **Elastic Net** | Natural extension | More complex adaptation |
| **Practical speed** | Faster with warm starts for dense data | Faster for exact path with few features |

**When to use which:**

| Scenario | Best Choice |
|----------|------------|
| Dense X, moderate p | LARS (exact path, fewer steps) |
| Sparse X, large p | Coordinate descent (exploits sparsity) |
| Elastic Net (not just Lasso) | Coordinate descent (LARS doesn't naturally extend) |
| Single lambda value | Coordinate descent |
| Full regularization path | LARS (exact breakpoints) |
| p > 10,000 | Coordinate descent with active set strategy |

---

## Question 36
- [ ] Done

**Explain how to interpret standardized coefficients after Ridge.**

**Answer:**

After fitting Ridge on **standardized** features, the standardized coefficients can be compared directly for relative importance.

**Steps:**
1. Standardize all features: z_j = (x_j - mean_j) / sd_j
2. Fit Ridge on standardized features: y = beta_0 + SUM(beta_j^std * z_j)
3. Interpret beta_j^std as: "A 1 standard deviation increase in x_j is associated with beta_j^std change in y"

**Interpretation rules:**

| Aspect | Interpretation |
|--------|---------------|
| **Magnitude** | |beta_j^std| indicates relative importance |
| **Sign** | Positive: positive association; Negative: inverse association |
| **Comparison** | Can compare across features (same scale) |
| **Shrinkage** | ALL coefficients are shrunken toward zero; rank may differ from OLS |

**Important caveats:**
- Ridge coefficients are **biased** -- they are systematically smaller than true effects
- The relative ranking may differ from OLS due to differential shrinkage (coefficients in noisy directions are shrunk more)
- With multicollinearity, Ridge **distributes** the effect among correlated features -- individual coefficient magnitudes are reduced

**Back-transforming to original scale:**
```python
# After fitting on standardized data:
beta_original_j = beta_std_j / sd_x_j
intercept_original = y_mean - SUM(beta_std_j * x_mean_j / sd_x_j)
```

**Recommendation:** Report both standardized (for comparison) and original-scale (for practical interpretation) coefficients.

---

## Question 37
- [ ] Done

**What diagnostics would you inspect after fitting a regularized model?**

**Answer:**

| Diagnostic | What to Check | How |
|-----------|--------------|-----|
| **1. CV error curve** | Is lambda optimal? Is there a clear minimum? | Plot CV error vs log(lambda) |
| **2. Coefficient path** | Are important features stable across lambda range? | Coefficient path plot |
| **3. Residual plots** | Non-linearity, heteroscedasticity still present? | Residuals vs fitted, Q-Q plot |
| **4. Selected features (Lasso)** | Are selected features sensible? Stable across folds? | Stability selection, domain check |
| **5. Non-zero count (Lasso)** | How many features survived? Too many/few? | Count non-zero coefficients |
| **6. Prediction performance** | Test set RMSE, R^2 acceptable? | Hold-out evaluation |
| **7. Coefficient magnitudes** | Any suspiciously large coefficients remaining? | Inspect coefficient table |
| **8. Feature correlations** | Did Ridge distribute well? Did Lasso pick right one? | Compare with correlation matrix |

**Post-fit workflow:**
```python
# 1. Examine CV curve
plt.plot(np.log10(model.alphas_), model.mse_path_.mean(axis=1))

# 2. Check coefficients
print(f"Non-zero: {np.sum(model.coef_ != 0)} / {len(model.coef_)}")

# 3. Residual analysis (same as OLS but with regularized fit)
residuals = y_test - model.predict(X_test)
# Q-Q plot, residual vs fitted, etc.

# 4. Compare with OLS
from sklearn.linear_model import LinearRegression
ols = LinearRegression().fit(X_train, y_train)
print(f"Ridge R2: {model.score(X_test, y_test):.4f}")
print(f"OLS R2: {ols.score(X_test, y_test):.4f}")
```

---

## Question 38
- [ ] Done

**How do Ridge/Lasso react to heteroscedastic errors?**

**Answer:**

Heteroscedastic errors (non-constant variance) affect Ridge and Lasso differently from OLS, but both can be impacted.

**Effects on regularized estimators:**

| Aspect | OLS | Ridge/Lasso |
|--------|-----|-------------|
| **Coefficient bias** | Unbiased | Biased (from regularization, not from heteroscedasticity) |
| **Coefficient variance** | Wrong SE estimates | Also wrong SE estimates |
| **Prediction** | Inefficient | Still affected by heteroscedasticity |

**Specific impacts:**
1. **Lambda selection**: CV error is estimated with higher variance -> suboptimal lambda choice
2. **Feature selection (Lasso)**: Features associated with high-variance regions may be over-penalized or under-penalized
3. **Standard errors**: Bootstrap or sandwich estimates are needed for valid inference

**Solutions:**

| Solution | Description |
|----------|-------------|
| **Log-transform target** | Often stabilizes variance |
| **Weighted regularized regression** | Weighted Ridge/Lasso with weights = 1/variance_estimate |
| **Robust standard errors** | Use HC standard errors for inference after regularized fit |
| **Square-root Lasso** | Pivotal version that's robust to unknown error variance |
| **Heteroscedasticity-aware penalties** | Weight lambdas differently per observation |

**Practical approach:**
1. Fit regularized model
2. Check residual vs fitted plot for variance patterns
3. If heteroscedastic: transform target or use weighted version
4. For inference: use bootstrap confidence intervals

---

## Question 39
- [ ] Done

**Describe the effect of strong regularization on model residuals.**

**Answer:**

**Effects of strong regularization (large lambda):**

| Aspect | Effect |
|--------|--------|
| **Coefficient magnitudes** | All coefficients close to zero (Ridge) or mostly zero (Lasso) |
| **Residuals** | Larger on average (model underfits) |
| **Residual variance** | Closer to total variance of y (model captures little signal) |
| **Residual pattern** | May show systematic patterns (e.g., curvature) indicating underfitting |
| **R^2** | Approaches zero (model not much better than predicting mean) |

**Residual analysis under strong regularization:**

| Observation | Interpretation |
|-------------|---------------|
| Residuals ~ y - mean(y) | Model is essentially predicting the mean (extreme regularization) |
| Structured patterns in residuals | Model is missing important signals; lambda too high |
| Residuals concentrated near zero | Appropriate regularization; signal captured |
| Normal Q-Q plot deviates | Not a regularization issue; inherent non-normality |

**How residuals change along the regularization path:**
```
lambda = infinity: residuals = y - mean(y)  (largest, captures no signal)
lambda = optimal:  residuals = noise        (smallest meaningful, best model)
lambda = 0 (OLS):  residuals = smallest     (may be overfitting to noise)
```

**Key diagnostic:** If increasing lambda from optimal reduces test error, you were overfitting. If increasing lambda increases test error sharply, regularization is becoming too strong.

---

## Question 40
- [ ] Done

**Explain why Lasso may pick different features across CV folds.**

**Answer:**

Lasso's feature selection can be **unstable** across CV folds because:

**Root causes:**

| Cause | Explanation |
|-------|-------------|
| **Correlated features** | Among correlated features, which gets selected depends on the specific training data in each fold |
| **Borderline features** | Features near the selection threshold (|coefficient| ~ 0) flip in/out based on data noise |
| **Small n** | With fewer observations, each fold is more different -> more variability |
| **Non-unique solution** | When features are perfectly correlated, multiple solutions with same loss exist |

**Example:**
```
Fold 1: Selects features {A, C, D, F}    (B dropped, correlated with A)
Fold 2: Selects features {B, C, D, G}    (A dropped, F dropped)
Fold 3: Selects features {A, B, C, D}    (Both A and B kept with small coefs)
```

**Quantifying instability:**
- Stability measure: Proportion of folds in which each feature is selected
- Features selected in > 80% of folds are "stable"
- Features selected in 30-70% of folds are "unstable"

**Solutions:**

| Solution | Description |
|----------|-------------|
| **Stability selection** | Subsample data many times, keep features selected > threshold (e.g., 60%) |
| **Elastic Net** | L2 component stabilizes selection of correlated features |
| **Bolasso (bootstrap-enhanced Lasso)** | Intersect support across bootstrap samples |
| **Randomized Lasso** | Add noise to penalty weights; similar to stability selection |
| **Increase n** | More data reduces fold-to-fold variability |

---

## Question 41
- [ ] Done

**Can you parallelize cross-validation for λ search? How?**

**Answer:**

Yes, CV for lambda search is **embarrassingly parallel** because evaluations at different lambda values are independent.

**Parallelization strategies:**

| Level | What's Parallelized | How |
|-------|-------------------|-----|
| **Across lambda values** | Each lambda evaluated independently | Distribute lambda grid across cores |
| **Across CV folds** | Each fold is an independent train/test split | Parallelize K fold evaluations |
| **Both** | Grid of (lambda x fold) combinations | Full grid parallelism |

**In sklearn:**
```python
from sklearn.linear_model import LassoCV
# n_jobs=-1 parallelizes across folds
model = LassoCV(alphas=np.logspace(-4, 2, 100), cv=5, n_jobs=-1)
model.fit(X, y)
```

**Parallelization with joblib:**
```python
from joblib import Parallel, delayed
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

alphas = np.logspace(-4, 2, 100)

def evaluate_alpha(alpha):
    return cross_val_score(Lasso(alpha=alpha), X, y, cv=5).mean()

results = Parallel(n_jobs=-1)(delayed(evaluate_alpha)(a) for a in alphas)
```

**Important caveats:**
- Warm-start paths are **sequential** (each lambda depends on previous solution)
- Parallel across folds but sequential along lambda path is the sweet spot
- Memory: Each worker needs a copy of X and y
- Overhead: Parallelization helps most when n*p is large or grid is fine
- LassoCV/RidgeCV use warm starts internally and may be faster than naive parallelization

---

## Question 42
- [ ] Done

**How do you handle missing values before regularized regression?**

**Answer:**

Missing values must be handled BEFORE fitting regularized regression, as sklearn doesn't accept NaN.

**Important principle:** Imputation must be fit on TRAINING data only (prevent data leakage).

**Strategies:**

| Strategy | Method | When to Use |
|----------|--------|-------------|
| **Simple imputation** | Mean, median, mode | Quick baseline; MCAR |
| **KNN imputation** | Average of K nearest neighbors | MAR; moderate missingness |
| **Iterative (MICE)** | Chained equations, iterative model-based | MAR; gold standard for inference |
| **Indicator variables** | Add binary "is_missing" column per feature | When missingness is informative |
| **Deletion** | Drop rows/columns with missing values | Very low missingness (< 5%) |

**Pipeline approach (prevents leakage):**
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

pipe = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler()),        # Must be after imputation
    ('model', LassoCV(cv=5))
])
pipe.fit(X_train, y_train)  # Imputer fit on train only
```

**Special considerations for regularized regression:**
- Imputation affects feature correlations -> affects which features Lasso selects
- Mean imputation reduces variance -> may cause features to be penalized differently
- Missing indicator features interact with regularization (may be selected/dropped)

**Best practice:** Use KNN or iterative imputation inside a pipeline with StandardScaler before the regularized model.

---

## Question 43
- [ ] Done

**Discuss the impact of outliers on Ridge and Lasso.**

**Answer:**

Outliers affect Ridge and Lasso differently than OLS, but both L1 and L2 penalties still use squared loss for the data-fitting term.

**Impact of outliers:**

| Aspect | OLS | Ridge | Lasso |
|--------|-----|-------|-------|
| **Sensitivity** | Very high (squared loss) | High (same squared loss) | High (same squared loss) |
| **Coefficient impact** | Large shift toward outlier | Reduced shift (shrinkage helps slightly) | May select features driven by outliers |
| **Feature selection** | N/A | N/A | Outliers can cause wrong features to be selected |

**Why regularization doesn't fix outliers:**
- The data-fitting term is STILL squared error: SUM(yi - xi*beta)^2
- One extreme outlier can dominate this sum regardless of regularization
- Lambda controls overfitting to noise, NOT robustness to outliers

**What can happen:**
1. **Ridge**: Outlier pulls all coefficients slightly; shrinkage provides mild protection
2. **Lasso**: Outlier may cause a feature strongly associated with the outlier to have an inflated coefficient

**Robust alternatives:**

| Method | Description |
|--------|-------------|
| **Robust Lasso** | Replace squared loss with Huber loss + L1 penalty |
| **LAD-Lasso** | Minimize SUM|residuals| + lambda*SUM|beta| |
| **Outlier detection first** | Remove/downweight outliers before fitting |
| **Winsorization** | Cap extreme values before fitting |
| **Huber regression + regularization** | Huber loss is less sensitive to outliers |

```python
from sklearn.linear_model import HuberRegressor
# Robust to outliers, includes L2 regularization
model = HuberRegressor(epsilon=1.35, alpha=0.01)
```

---

## Question 44
- [ ] Done

**Explain "post-Lasso OLS" and its benefits.**

**Answer:**

**Post-Lasso OLS** is a two-stage procedure:
1. **Stage 1 (Selection)**: Fit Lasso to select features with non-zero coefficients
2. **Stage 2 (Estimation)**: Fit ordinary OLS using ONLY the Lasso-selected features

**Why Post-Lasso OLS?**

| Issue with Lasso | How Post-Lasso OLS Fixes It |
|-----------------|---------------------------|
| **Biased coefficients** | OLS on selected features gives unbiased estimates |
| **Shrunken coefficients** | OLS doesn't shrink -> full coefficient magnitude |
| **Invalid standard errors** | OLS provides standard theory for inference |
| **Conservative predictions** | OLS predictions not systematically too small |

**Mathematical justification:**
- Lasso is good at SELECTION but biased for ESTIMATION
- OLS is good at ESTIMATION but poor at SELECTION (uses all features)
- Post-Lasso combines strengths: Lasso selects, OLS estimates

**Implementation:**
```python
from sklearn.linear_model import Lasso, LinearRegression
# Stage 1: Feature selection
lasso = Lasso(alpha=0.1).fit(X_train, y_train)
selected = np.where(lasso.coef_ != 0)[0]

# Stage 2: OLS on selected features
ols = LinearRegression().fit(X_train[:, selected], y_train)
y_pred = ols.predict(X_test[:, selected])
```

**Caveats:**
- Post-selection inference is tricky: standard p-values from Stage 2 are NOT valid (selection bias)
- Use **selective inference** or **data splitting** for valid post-selection p-values
- If Lasso selects poorly, Post-Lasso OLS will also perform poorly

---

## Question 45
- [ ] Done

**Contrast Ridge/Lasso with Principal Component Regression.**

**Answer:**

| Aspect | Ridge | Lasso | PCR (Principal Component Regression) |
|--------|-------|-------|--------------------------------------|
| **Approach** | Penalize coefficient magnitudes | Penalize |coefficient| magnitudes | Reduce dimensions with PCA, then OLS |
| **Feature selection** | No (keeps all) | Yes (sparse) | No (uses PCs, not original features) |
| **Dimensionality reduction** | Implicit (shrinkage) | Implicit (sparsity) | Explicit (k components) |
| **Multicollinearity** | Handled by lambda | Handled by selection | Eliminated (PCs are orthogonal) |
| **Interpretability** | Original features | Subset of original features | PCs (hard to interpret) |
| **Supervised?** | Yes (uses y in penalty) | Yes (uses y in penalty) | No (PCA ignores y) |

**Key differences:**

| Feature | Ridge/Lasso | PCR |
|---------|------------|-----|
| **How it reduces variance** | Shrink coefficients | Drop low-variance PCs |
| **Target consideration** | Lambda tuned to minimize prediction error | PCs chosen by X variance, not predictive power |
| **Risk** | Over/under-regularization | Important signal may be in low-variance PC (discarded!) |

**When PCR is problematic:**
- A low-variance PC could be highly predictive
- PCR discards it (based on variance only)
- Ridge/Lasso would keep it (based on prediction relevance)

**When PCR works well:**
- High-dimensional data (p >> n)
- Features are naturally structured (e.g., spectral data)
- Signal is concentrated in high-variance directions

**Better alternative:** PLS regression (supervised dimensionality reduction -- finds components maximizing covariance with y)

---

## Question 46
- [ ] Done

**What is group Lasso and how is it solved?**

**Answer:**

**Group Lasso** applies the L1 penalty at the **group level** rather than the individual feature level, enforcing all-in or all-out selection for pre-defined groups.

**Standard Lasso:** Minimize MSE + lambda * SUM_j |beta_j|
**Group Lasso:** Minimize MSE + lambda * SUM_g sqrt(p_g) * ||beta_g||_2

where:
- g indexes groups, p_g is group size
- ||beta_g||_2 = sqrt(SUM_j_in_g beta_j^2)
- sqrt(p_g) normalizes for group size

**How it works:**
- The L2 norm of each group acts as the penalty
- When ||beta_g||_2 is penalized to zero, ALL features in group g are zero
- When a group is selected, all its members have non-zero coefficients

**When to use:**

| Scenario | Groups |
|----------|--------|
| **Categorical variables** | All dummies from one categorical form a group |
| **Gene pathways** | Genes in the same biological pathway |
| **Temporal blocks** | Lagged features for the same time window |
| **Multi-level factors** | Interaction terms in a factorial design |

**Optimization:**
- Solved via block coordinate descent
- Each block update involves a group-level soft-thresholding:
  beta_g = (1 - lambda*sqrt(p_g) / ||z_g||_2)_+ * z_g
  where z_g is the OLS estimate for group g given other groups fixed

**Variants:**
- **Sparse Group Lasso**: Group Lasso + within-group L1 -> group selection + within-group sparsity
- **Overlap Group Lasso**: Allows features to belong to multiple groups

---

## Question 47
- [ ] Done

**Describe the dual formulation of Ridge regression.**

**Answer:**

The dual formulation reformulates Ridge regression from coefficient space (primal) to observation space (dual).

**Primal form (standard):**
```
Minimize: ||y - X*beta||^2 + lambda * ||beta||^2
Solution: beta = (X'X + lambda*I_p)^(-1) * X'y     [p x p matrix inversion]
```

**Dual form (kernel representation):**
```
Prediction: y_hat = X * beta = X * X'(XX' + lambda*I_n)^(-1) * y
Let K = XX' (n x n kernel matrix)
y_hat = K * (K + lambda*I_n)^(-1) * y              [n x n matrix inversion]
```

**When to use which:**

| Form | Matrix Size | Best When |
|------|------------|-----------|
| **Primal** | p x p (X'X + lambda*I) | p << n (few features, many samples) |
| **Dual** | n x n (XX' + lambda*I) | n << p (few samples, many features) |

**Connection to Kernel Ridge Regression:**
- The dual form only depends on inner products XX' = K
- We can replace K with any kernel matrix K(xi, xj) = phi(xi)' * phi(xj)
- This enables non-linear regression without explicitly computing features

**SVD connection:**
If X = U*D*V', then:
- Primal: involves V * diag(d_j^2 / (d_j^2 + lambda)) * V' (p-dimensional)
- Dual: involves U * diag(d_j^2 / (d_j^2 + lambda)) * U' (n-dimensional)
- Both give the same predictions, but computational cost differs

---

## Question 48
- [ ] Done

**Explain early stopping in gradient-descent Ridge fitting.**

**Answer:**

**Early stopping** terminates gradient descent iterations before full convergence, which acts as an implicit form of regularization equivalent to Ridge (L2).

**How it works:**
```
Initialize beta = 0
For t = 1, 2, ..., T_stop:
    beta = beta - eta * gradient(loss)
    if validation_error increases:
        stop and return beta  (early stopping)
```

**Connection to Ridge regularization:**

| Aspect | Early Stopping | Explicit Ridge |
|--------|---------------|---------------|
| **Regularization parameter** | Number of iterations T | lambda |
| **More regularization** | Fewer iterations | Larger lambda |
| **Less regularization** | More iterations | Smaller lambda |
| **At convergence** | Equivalent to OLS (lambda = 0) | lambda > 0 always |

**Mathematical result (for gradient descent with learning rate eta):**
- After T iterations: beta_T is approximately the Ridge solution with lambda = 1/(eta*T)
- Fewer iterations (small T) = larger effective lambda = more regularization
- This is known as the **implicit regularization** of gradient descent

**Why it works:**
- Gradient descent first captures the **largest eigenvalue directions** (signal)
- Later iterations fit **small eigenvalue directions** (noise)
- Stopping early prevents fitting noise = reduces variance

**Practical use:**
- Monitor validation error during training
- Stop when validation error starts increasing
- Use **patience**: stop after P consecutive increases (avoid premature stopping)
- Computationally cheaper than explicit Ridge with CV (no need to solve for multiple lambda values)

---

## Question 49
- [ ] Done

**How would you adapt Elastic Net for multinomial classification?**

**Answer:**

**Multinomial Elastic Net** extends Elastic Net from binary/regression to **multi-class classification** with K > 2 classes.

**Model:**
```
P(y = k | x) = exp(x' * beta_k) / SUM_j exp(x' * beta_j)    (softmax)

Minimize: -log-likelihood + alpha * [l1_ratio * SUM_k SUM_j |beta_kj| 
                                   + (1-l1_ratio)/2 * SUM_k SUM_j beta_kj^2]
```

**Key changes from standard Elastic Net:**
- **K-1 coefficient vectors** (one per class, one reference)
- **Softmax** link function instead of identity/logit
- **Penalty applied to all K*p coefficients**
- **Cross-entropy loss** instead of squared error

**Implementation:**
```python
from sklearn.linear_model import LogisticRegression

# Multinomial Elastic Net
model = LogisticRegression(
    penalty='elasticnet',
    solver='saga',         # Required for elastic net
    l1_ratio=0.5,
    C=1.0,                 # C = 1/alpha (inverse regularization)
    multi_class='multinomial',
    max_iter=10000
)
model.fit(X_train, y_train)
# model.coef_ shape: (K, p) -- one coefficient vector per class
```

**Feature selection behavior:**
- L1 component can zero out a feature across ALL classes (global sparsity)
- Or zero out a feature for SOME classes but not others (class-specific sparsity)

**Use case:** Gene expression classification with thousands of genes, multiple cancer types -> Elastic Net selects relevant genes while grouping correlated ones.

---

## Question 50
- [ ] Done

**Give an industry case study where Elastic Net improved performance.**

**Answer:**

**Industry case study: Gene expression analysis for drug response prediction**

**Context:** A pharmaceutical company needed to predict patient drug response from ~20,000 gene expression features with only 200 patient samples (p >> n).

**Challenge:**
- 20,000 features, 200 samples -> severe overfitting risk
- Many genes are co-regulated in pathways (highly correlated groups)
- Need interpretable results for drug mechanism understanding

**Model comparison results:**

| Model | Test R^2 | Features Used | Stability |
|-------|---------|--------------|-----------|
| **OLS** | N/A (p > n, cannot fit) | All 20,000 | N/A |
| **Ridge** | 0.42 | All 20,000 (no sparsity) | High |
| **Lasso** | 0.48 | 85 genes | Low (different genes per fold) |
| **Elastic Net (l1=0.5)** | **0.55** | 120 genes | High |

**Why Elastic Net won:**
1. **Grouped selection**: Selected entire gene pathways (5-10 genes per pathway) rather than one gene per pathway (as Lasso did)
2. **Stability**: Same gene sets selected across 10 bootstrap runs (>80% overlap vs. <40% for Lasso)
3. **Better prediction**: Using correlated gene groups captured more biological signal
4. **Interpretability**: Selected pathways matched known biology -> clinically actionable

**Other industry applications where Elastic Net excels:**
- **Finance**: Portfolio selection with correlated asset returns
- **Marketing**: Customer segmentation with correlated behavioral features
- **Manufacturing**: Sensor-based quality prediction with spatially correlated sensors
- **Climate science**: Temperature prediction from correlated atmospheric variables

---
