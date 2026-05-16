# Linear Regression - General Questions

## Question 1
- [ ] Done

**What assumptions are made in linear regression modeling?**

**Answer:**

Linear regression relies on several key assumptions (often remembered as **LINE**):

| Assumption | Description | How to Check |
|------------|-------------|--------------|
| **Linearity** | Relationship between features and target is linear | Scatter plots, residual vs. fitted plots |
| **Independence** | Observations are independent of each other | Durbin-Watson test (value ~2 = no autocorrelation) |
| **Normality** | Residuals are normally distributed | Q-Q plot, Shapiro-Wilk test |
| **Equal Variance (Homoscedasticity)** | Residuals have constant variance across all levels of predictors | Residual vs. fitted plot, Breusch-Pagan test |

**Additional assumptions:**
- **No multicollinearity**: Independent variables are not highly correlated with each other (check via VIF < 10)
- **No significant outliers**: Extreme values don't unduly influence the model (check via Cook's distance)
- **Correct model specification**: All relevant variables are included, no irrelevant ones added

**What happens when assumptions are violated:**

| Violation | Consequence | Remedy |
|-----------|-------------|--------|
| Non-linearity | Biased predictions | Polynomial terms, transformations |
| Autocorrelation | Inefficient estimates, invalid p-values | GLS, time-series models |
| Non-normality | Invalid confidence intervals | Transformations, larger sample size |
| Heteroscedasticity | Inefficient estimates | WLS, robust standard errors |

---

## Question 2
- [ ] Done

**How do you interpret the coefficients of a linear regression model?**

**Answer:**

Each coefficient β_i represents the **expected change in the target variable** for a **one-unit increase** in the corresponding feature, **holding all other features constant**.

| Component | Interpretation | Example |
|-----------|---------------|---------|
| **Intercept (β₀)** | Predicted y when all features = 0 | Base salary when experience = 0 |
| **Positive coefficient** | Feature and target move in same direction | β = 5000: Each year of experience adds $5,000 to salary |
| **Negative coefficient** | Feature and target move in opposite directions | β = -2: Each mile from city center reduces price by $2K |
| **Magnitude** | Strength of the relationship (after scaling) | Larger magnitude = stronger effect |

**Important considerations:**
- Coefficients are only directly comparable when features are **standardized** (z-scored)
- Statistical significance should be checked via **p-values** (typically p < 0.05)
- The **sign** indicates direction, the **magnitude** indicates strength
- In the presence of multicollinearity, individual coefficients become unreliable
- **Standardized coefficients** (beta weights) allow comparing relative importance across features

---

## Question 3
- [ ] Done

**How is hypothesis testing used in the context of linear regression?**

**Answer:**

Hypothesis testing determines whether the relationships found by the regression model are **statistically significant** or due to random chance.

**Key Tests:**

| Test | Hypotheses | Purpose |
|------|-----------|---------|
| **t-test (per coefficient)** | H₀: βᵢ = 0 vs H₁: βᵢ ≠ 0 | Is this individual feature significant? |
| **F-test (overall model)** | H₀: All βᵢ = 0 vs H₁: At least one βᵢ ≠ 0 | Is the model as a whole significant? |
| **Partial F-test** | H₀: Subset of βᵢ = 0 | Is a group of features jointly significant? |

**Process:**
1. **State hypotheses**: Null (no effect) vs. Alternative (effect exists)
2. **Calculate test statistic**: t-statistic = β̂ᵢ / SE(β̂ᵢ)
3. **Find p-value**: Probability of observing this statistic under H₀
4. **Decision**: If p-value < significance level (α = 0.05), reject H₀

**Practical interpretation:**
- A significant t-test (p < 0.05) means the feature likely has a real effect
- A significant F-test means the model explains more variance than a mean-only model
- Always check **confidence intervals** alongside p-values for effect size context

---

## Question 4
- [ ] Done

**What do you understand by the term "normality of residuals"?**

**Answer:**

Normality of residuals means that the **errors** (differences between actual and predicted values) follow a **normal (Gaussian) distribution** centered at zero.

**Why it matters:**
- Required for valid **confidence intervals** and **hypothesis tests** (t-tests, F-tests)
- Ensures that p-values and prediction intervals are reliable
- NOT about the distribution of the raw data — only about the residuals

**How to check:**

| Method | Type | What to Look For |
|--------|------|-----------------|
| **Q-Q Plot** | Visual | Points should fall along the 45° diagonal line |
| **Histogram of residuals** | Visual | Should look bell-shaped |
| **Shapiro-Wilk test** | Statistical | p > 0.05 → normality not rejected |
| **Kolmogorov-Smirnov test** | Statistical | p > 0.05 → normality not rejected |

**What to do if violated:**
- Apply **transformations** to the target (log, sqrt, Box-Cox)
- Remove **outliers** that skew the distribution
- Use **robust regression** methods
- With large samples (n > 30), the Central Limit Theorem makes this less critical

---

## Question 5
- [ ] Done

**How do you deal with missing values when preparing data for linear regression?**

**Answer:**

Missing values must be handled before fitting a linear regression model, as most implementations cannot handle NaN values directly.

**Strategies:**

| Strategy | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **Listwise deletion** | Missing completely at random (MCAR), small % missing | Simple, no bias if MCAR | Reduces sample size |
| **Mean/Median imputation** | Quick baseline, numerical features | Simple | Reduces variance, ignores relationships |
| **Mode imputation** | Categorical features | Simple | Ignores data structure |
| **KNN imputation** | Complex patterns in data | Captures local structure | Computationally expensive |
| **Iterative imputation (MICE)** | Multiple features with missing data | Models relationships between features | Complex, requires tuning |
| **Regression imputation** | Predictable missingness | Uses feature relationships | Can overfit imputation model |
| **Indicator variable** | Missingness is informative | Preserves missingness info | Adds extra feature |

**Best practices:**
1. First, analyze the **pattern** of missingness (MCAR, MAR, MNAR)
2. If < 5% missing and MCAR → listwise deletion is acceptable
3. For larger amounts → use **multiple imputation** (MICE) for most reliable results
4. Always impute **before** splitting, but fit imputer on **train only** to prevent data leakage

---

## Question 6
- [ ] Done

**What feature selection methods can be used prior to building a regression model?**

**Answer:**

| Category | Methods | Description |
|----------|---------|-------------|
| **Filter Methods** | Correlation analysis, mutual information, ANOVA F-test | Rank features independently of the model |
| **Wrapper Methods** | Forward selection, backward elimination, RFE | Use model performance to select features |
| **Embedded Methods** | Lasso (L1), Ridge (L2), Elastic Net | Feature selection built into model training |

**Detailed approaches:**

| Method | How It Works | Best For |
|--------|-------------|----------|
| **Correlation threshold** | Remove features with low correlation to target | Quick initial filtering |
| **VIF analysis** | Remove features with VIF > 5-10 (multicollinearity) | Reducing redundancy |
| **Forward selection** | Start empty, add features one by one based on improvement | Small feature sets |
| **Backward elimination** | Start full, remove features one by one based on p-values | Small-to-medium feature sets |
| **Lasso regularization** | L1 penalty drives coefficients to exactly zero | Automatic feature selection |
| **Recursive Feature Elimination (RFE)** | Iteratively removes least important features | Model-specific selection |

**Recommended workflow:**
1. Remove constant/near-constant features
2. Check VIF, remove highly collinear features (VIF > 10)
3. Use Lasso or RFE for further selection
4. Validate with cross-validation

---

## Question 7
- [ ] Done

**How is feature scaling relevant to linear regression?**

**Answer:**

**For standard OLS regression:** Feature scaling does **not** affect predictions, R², or p-values — only the coefficient magnitudes change. The model is mathematically equivalent.

**When scaling IS important:**

| Scenario | Why Scaling Matters |
|----------|-------------------|
| **Regularized regression (Ridge, Lasso)** | Penalty is applied equally to all coefficients; unscaled features with large ranges get penalized unfairly |
| **Gradient descent optimization** | Unscaled features cause elongated contours → slow convergence |
| **Comparing coefficient importance** | Standardized coefficients allow direct comparison of feature importance |
| **Numerical stability** | Very large/small values can cause floating-point issues |

**Common scaling methods:**

| Method | Formula | When to Use |
|--------|---------|-------------|
| **Standardization (Z-score)** | (x - μ) / σ | Default choice, preserves outlier information |
| **Min-Max scaling** | (x - min) / (max - min) | When bounded [0,1] range needed |
| **Robust scaling** | (x - median) / IQR | When data has outliers |

**Key rule:** Always fit the scaler on **training data only**, then transform both train and test sets to prevent data leakage.

---

## Question 8
- [ ] Done

**How do you address overfitting in linear regression?**

**Answer:**

Overfitting occurs when the model captures noise in the training data, leading to poor generalization.

**Strategies to address overfitting:**

| Strategy | How It Helps |
|----------|-------------|
| **Regularization (Ridge/Lasso)** | Adds penalty term to shrink coefficients → simpler model |
| **Feature selection** | Remove irrelevant features that add noise |
| **Cross-validation** | Use k-fold CV to get realistic performance estimates |
| **More training data** | More data reduces the influence of noise |
| **Reduce polynomial degree** | Lower-degree polynomials are less prone to overfitting |
| **Early stopping** | Stop gradient descent before the model memorizes noise |

**Signs of overfitting:**
- Large gap between training R² and test R²
- Very large coefficient magnitudes
- Model performs well on training data but poorly on unseen data

**Regularization comparison:**

| Method | Effect | When to Use |
|--------|--------|-------------|
| **Ridge (L2)** | Shrinks all coefficients toward zero | Many small/medium effects |
| **Lasso (L1)** | Drives some coefficients to exactly zero | Sparse models, automatic feature selection |
| **Elastic Net** | Combination of L1 and L2 | Correlated features, best of both |

---

## Question 9
- [ ] Done

**How do you use regularization to improve linear regression models?**

**Answer:**

Regularization adds a **penalty term** to the loss function to constrain coefficient magnitudes, preventing overfitting.

**Standard OLS loss:** L = Σ(yᵢ - ŷᵢ)²

**Regularized losses:**

| Method | Loss Function | Effect |
|--------|--------------|--------|
| **Ridge (L2)** | OLS + λΣβⱼ² | Shrinks all coefficients, never zeros them |
| **Lasso (L1)** | OLS + λΣ|βⱼ| | Can drive coefficients to exactly zero |
| **Elastic Net** | OLS + λ₁Σ|βⱼ| + λ₂Σβⱼ² | Combines both penalties |

**How λ (alpha) controls regularization:**
- **λ = 0**: No regularization → standard OLS
- **Small λ**: Mild regularization → slight shrinkage
- **Large λ**: Heavy regularization → coefficients approach zero
- **Optimal λ**: Found via cross-validation (e.g., RidgeCV, LassoCV)

**Practical benefits:**
- Reduces model variance at the cost of slight bias increase (bias-variance tradeoff)
- Handles multicollinearity (Ridge stabilizes coefficient estimates)
- Performs automatic feature selection (Lasso)
- Improves generalization to new data

---

## Question 10
- [ ] Done

**How can you optimize the hyperparameters of a regularized linear regression model?**

**Answer:**

The main hyperparameter is **λ (alpha)** — the regularization strength. For Elastic Net, there is also the **l1_ratio** (mixing parameter).

**Optimization methods:**

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Grid Search + CV** | Try predefined λ values, evaluate with cross-validation | Thorough, simple | Slow for large search spaces |
| **Randomized Search** | Sample λ from a distribution | Faster than grid search | May miss optimal value |
| **RidgeCV / LassoCV** | Built-in efficient CV using LOOCV or analytical shortcut | Very fast, optimized | Limited to specific models |
| **Bayesian Optimization** | Uses probabilistic model to guide search | Efficient for expensive models | More complex setup |

**Best practices:**
1. Search λ on a **logarithmic scale**: `np.logspace(-4, 4, 100)`
2. Use **k-fold cross-validation** (k=5 or 10) for evaluation
3. For Elastic Net, tune both `alpha` and `l1_ratio` jointly
4. Use **MSE** or **negative MSE** as the scoring metric
5. Always standardize features before regularized regression
6. Report results on a **held-out test set** not used during tuning

---

## Question 11
- [ ] Done

**How is influence measured in the context of linear regression?**

**Answer:**

Influence measures identify data points that disproportionately affect the regression model's fitted values or coefficients.

**Key influence measures:**

| Measure | What It Captures | Threshold |
|---------|-----------------|-----------|
| **Leverage (hᵢᵢ)** | How far a point's features are from the mean | > 2(p+1)/n |
| **Cook's Distance** | Combined effect of leverage + residual magnitude | > 4/n or > 1 |
| **DFFITS** | Change in fitted value when point i is removed | > 2√(p/n) |
| **DFBETAS** | Change in each coefficient when point i is removed | > 2/√n |
| **Studentized Residuals** | Standardized residual using leave-one-out variance | > ±3 |

**Interpretation:**
- **High leverage, low residual**: Point pulls the line toward it (may not be harmful)
- **Low leverage, high residual**: Outlier in y-direction (may or may not be influential)
- **High leverage, high residual**: Most dangerous — both extreme and poorly fit

**What to do with influential points:**
1. Investigate for **data errors** (typos, measurement issues)
2. Check if they represent a **different population**
3. Report results with and without influential points
4. Consider **robust regression** (Huber loss, RANSAC)

---

## Question 12
- [ ] Done

**How can linear regression be used for price optimization in retail?**

**Answer:**

Linear regression models the relationship between **price** and **demand/revenue**, enabling data-driven pricing decisions.

**Approach:**

| Step | Action | Details |
|------|--------|---------|
| 1 | **Data collection** | Historical sales, prices, promotions, competitor prices, seasonality |
| 2 | **Model demand** | Demand = β₀ + β₁(Price) + β₂(Competitor_Price) + β₃(Season) + ... |
| 3 | **Estimate elasticity** | Price elasticity = β₁ × (Price/Demand) |
| 4 | **Optimize** | Find price that maximizes Revenue = Price × Predicted_Demand |

**Price elasticity interpretation:**
- **|Elasticity| > 1** (elastic): Price decrease → revenue increase → lower price
- **|Elasticity| < 1** (inelastic): Price increase → revenue increase → raise price
- **|Elasticity| = 1** (unit elastic): Revenue maximized at current price

**Extensions:**
- Include **interaction terms** (Price × Season) for dynamic pricing
- Use **log-log model** for direct elasticity estimation: log(Demand) = β₀ + β₁·log(Price)
- Segment by customer type for personalized pricing
- Add competitor pricing variables for market-aware optimization

---

## Question 13
- [ ] Done

**Illustrate the process you would follow to model the relationship between advertising spend and revenue generation.**

**Answer:**

**Step-by-step workflow:**

| Phase | Steps |
|-------|-------|
| **1. Data Collection** | Gather historical data: ad spend (TV, digital, print), revenue, time period, seasonality indicators |
| **2. EDA** | Plot spend vs. revenue scatter plots, check correlations, identify trends and outliers |
| **3. Preprocessing** | Handle missing values, create lag features (ad effect delay), encode categorical variables |
| **4. Feature Engineering** | Create interaction terms (TV × Digital), diminishing returns features (log/sqrt transforms), cumulative spend |
| **5. Model Building** | Start with simple regression (Revenue ~ Total_Spend), then multiple regression with channel breakdown |
| **6. Validation** | K-fold cross-validation, check assumptions (residual plots, VIF), compare R² on train vs. test |
| **7. Interpretation** | Coefficient analysis: "Each $1K in digital ads generates $X in revenue" |
| **8. Optimization** | Use coefficients to allocate budget across channels for maximum ROI |

**Model specification:**
```
Revenue = β₀ + β₁(TV_Spend) + β₂(Digital_Spend) + β₃(Print_Spend) 
        + β₄(Season) + β₅(TV × Digital) + ε
```

**Key insights to extract:**
- Which channel has the highest ROI (largest standardized coefficient)?
- Are there synergies between channels (significant interaction terms)?
- Is there diminishing returns (log transform improves fit)?
- What is the optimal budget allocation across channels?

---

## Question 14
- [ ] Done

**Walk me through a time you diagnosed a poorly performing regression model and how you improved it.**

**Answer:**

**Scenario:** A house price prediction model had high training R² (0.92) but low test R² (0.61).

| Step | Diagnosis | Finding |
|------|-----------|---------|
| **1. Check for overfitting** | Compare train vs. test R² | Gap of 0.31 → overfitting confirmed |
| **2. Examine residual plots** | Plot residuals vs. fitted values | Fan-shaped pattern → heteroscedasticity |
| **3. Check multicollinearity** | Calculate VIF for all features | 3 features had VIF > 20 (highly collinear) |
| **4. Check normality** | Q-Q plot of residuals | Heavy right tail → target (price) is right-skewed |
| **5. Look for non-linearity** | Partial regression plots | Square footage showed curved relationship |

**Improvements applied:**

| Action | Impact |
|--------|--------|
| **Log-transform target** (price) | Fixed heteroscedasticity and non-normality; test R² → 0.72 |
| **Remove collinear features** (VIF > 10) | Reduced features from 15 to 11; more stable coefficients |
| **Add polynomial term** for sqft | Captured non-linear relationship; test R² → 0.78 |
| **Apply Ridge regularization** | Controlled remaining overfitting; test R² → 0.82 |
| **Remove outliers** (Cook's D > 4/n) | Removed 8 influential points; final test R² = 0.85 |

**Key lesson:** Systematic diagnosis using residual analysis, VIF, and assumption checks is more effective than blindly tuning hyperparameters.

---

## Question 15
- [ ] Done

**How has the field of linear regression modeling evolved with the advent of big data?**

**Answer:**

| Evolution | Traditional | Big Data Era |
|-----------|------------|-------------|
| **Scale** | Hundreds/thousands of rows | Millions/billions of rows |
| **Features** | Tens of features | Thousands+ features (p >> n scenarios) |
| **Computation** | Normal equation (closed-form) | Stochastic gradient descent, distributed computing |
| **Regularization** | Optional | Essential (Lasso/Ridge for high-dimensional data) |
| **Infrastructure** | Single machine, R/SAS | Spark MLlib, Dask, cloud computing |
| **Feature Engineering** | Manual, domain-driven | Automated (AutoML, feature stores) |
| **Model Selection** | AIC/BIC, stepwise | Cross-validation at scale, information criteria |
| **Monitoring** | One-time analysis | Continuous retraining, drift detection |

**Key developments:**
- **Online learning**: Models update incrementally with streaming data (SGD-based regression)
- **Distributed regression**: MapReduce/Spark implementations handle datasets too large for single machines
- **Sparse methods**: Lasso and Elastic Net handle millions of features efficiently
- **Automated pipelines**: End-to-end ML platforms automate preprocessing, training, deployment
- **Interpretability tools**: SHAP values, partial dependence plots complement coefficient analysis

---

## Question 16
- [ ] Done

**How can linear regression models be made more robust to non-standard data types?**

**Answer:**

| Data Type | Challenge | Solution |
|-----------|-----------|----------|
| **Categorical** | Not numeric | One-hot encoding, target encoding, ordinal encoding |
| **Text** | Unstructured | TF-IDF vectorization, word embeddings → use as numeric features |
| **Datetime** | Not directly usable | Extract components (hour, day, month), create cyclical features (sin/cos encoding) |
| **Ordinal** | Order matters but intervals unknown | Ordinal encoding, polynomial contrasts |
| **Skewed numerical** | Violates normality assumption | Log, sqrt, Box-Cox transformations |
| **Heavy-tailed data** | Outlier sensitivity | Robust regression (Huber loss, RANSAC), winsorization |
| **Spatial data** | Geographic dependencies | Include lat/lon, distance features, geographically weighted regression |
| **Hierarchical data** | Nested groups | Mixed-effects models (random intercepts/slopes) |

**Robust regression alternatives:**

| Method | How It Helps |
|--------|-------------|
| **Huber regression** | Combines squared loss (small residuals) with absolute loss (large residuals) |
| **RANSAC** | Fits model on inliers, ignores outliers |
| **Theil-Sen** | Uses median of slopes between all point pairs → resistant to outliers |
| **Quantile regression** | Models specific quantiles, not just the mean |

---

## Question 17
- [ ] Done

**What steps would you take if your linear regression model shows significant bias after deployment?**

**Answer:**

**Systematic approach to diagnosing and fixing post-deployment bias:**

| Step | Action | Details |
|------|--------|---------|
| **1. Quantify bias** | Compare predictions vs. actuals on recent data | Calculate residual mean, MAPE, direction of bias |
| **2. Check for data drift** | Compare feature distributions (train vs. production) | KS test, PSI (Population Stability Index) |
| **3. Check for concept drift** | Has the target-feature relationship changed? | Monitor coefficient stability over time |
| **4. Examine subgroup bias** | Analyze errors across segments | Are errors concentrated in specific groups? |
| **5. Review feature pipeline** | Check data preprocessing in production | Missing values handled differently? Scaling drift? |

**Remediation strategies:**

| Strategy | When to Apply |
|----------|--------------|
| **Retrain on recent data** | Data/concept drift detected |
| **Add new features** | New relevant variables available (market conditions, regulations) |
| **Recalibrate intercept** | Systematic over/under-prediction (bias offset) |
| **Use online learning** | Continuous small drift → update model incrementally |
| **Ensemble with correction model** | Residual model to correct systematic bias |
| **Switch model class** | If relationship has become non-linear → tree-based or neural models |

**Prevention for future:**
- Implement **monitoring dashboards** tracking prediction accuracy over time
- Set up **automated alerts** when error metrics exceed thresholds
- Schedule **periodic retraining** (weekly/monthly depending on domain)
- Use **sliding window** training to keep model current
