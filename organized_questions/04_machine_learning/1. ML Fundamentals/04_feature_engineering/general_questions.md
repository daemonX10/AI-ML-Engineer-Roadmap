# Feature Engineering Interview Questions - General Questions

## Question 1: List different types of features commonly used in machine learning.
- [ ] Done

### Answer

**Feature Types by Nature:**

| Type | Description | Examples |
|------|-------------|----------|
| **Numerical (Continuous)** | Real-valued numbers | Age, salary, temperature, price |
| **Numerical (Discrete)** | Integer counts | Number of children, page views |
| **Categorical (Nominal)** | Categories without order | Color, country, product type |
| **Categorical (Ordinal)** | Categories with order | Education level, rating (1-5) |
| **Binary** | Two possible values | Yes/No, Male/Female, True/False |
| **Date/Time** | Temporal information | Timestamps, dates |
| **Text** | Natural language | Reviews, descriptions, titles |

**Feature Types by Origin:**

| Type | Description | Examples |
|------|-------------|----------|
| **Raw Features** | Direct from data source | Customer age, transaction amount |
| **Derived Features** | Calculated from raw | Age groups, total spending |
| **Aggregated Features** | Summarized over groups | Average order value, max purchase |
| **Interaction Features** | Combinations of features | price × quantity, age × income |
| **Lag Features** | Past values (time series) | Sales_last_week, price_yesterday |
| **Window Features** | Rolling statistics | 7-day moving average |

**Domain-Specific Features:**

```python
# E-commerce
ecommerce_features = [
    'recency',  # Days since last purchase
    'frequency',  # Number of purchases
    'monetary',  # Total spend (RFM analysis)
    'basket_size_avg',
    'category_diversity'
]

# Time Series
time_features = [
    'hour_of_day', 'day_of_week', 'month',
    'is_weekend', 'is_holiday',
    'lag_1', 'lag_7', 'rolling_mean_7d'
]

# NLP
text_features = [
    'word_count', 'char_count',
    'avg_word_length', 'sentiment_score',
    'tfidf_vector', 'embedding'
]
```

---

## Question 2: Why is it important to understand domain knowledge while performing feature engineering?
- [ ] Done

### Answer

**Key Reasons:**

| Reason | Impact |
|--------|--------|
| **Create Meaningful Features** | Domain knowledge reveals which combinations make sense |
| **Avoid Data Leakage** | Understanding the data generation process prevents leakage |
| **Feature Validation** | Ensures engineered features are realistic |
| **Interpretability** | Creates features stakeholders can understand |
| **Efficiency** | Focus on features likely to be predictive |

**Real-World Examples:**

**1. Finance (Credit Scoring):**
```python
# Domain knowledge: Debt-to-Income ratio is crucial
df['debt_to_income'] = df['total_debt'] / df['annual_income']

# Domain knowledge: Multiple credit inquiries indicate risk
df['inquiry_rate'] = df['num_inquiries'] / df['credit_history_months']
```

**2. Healthcare:**
```python
# Domain knowledge: BMI is a standard health metric
df['bmi'] = df['weight_kg'] / (df['height_m'] ** 2)

# Domain knowledge: Age affects disease risk non-linearly
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 65, 100], 
                          labels=['young', 'middle', 'senior', 'elderly'])
```

**3. E-commerce:**
```python
# Domain knowledge: Recency, Frequency, Monetary (RFM) analysis
df['recency'] = (today - df['last_purchase_date']).dt.days
df['frequency'] = df.groupby('customer_id')['order_id'].transform('count')
df['monetary'] = df.groupby('customer_id')['amount'].transform('sum')
```

**Domain Knowledge Helps Identify:**
- Leaky features (future information)
- Proxy variables that shouldn't be used (discriminatory)
- Business rules that affect data
- Seasonal patterns and anomalies
- Data quality issues

**Without Domain Knowledge:**
- May create meaningless features
- Risk of data leakage
- Miss important domain-specific patterns
- Create features that violate business logic

---

## Question 3: How do you handle categorical variables in a dataset?
- [ ] Done

### Answer

**Encoding Methods Overview:**

| Method | When to Use | Cardinality |
|--------|-------------|-------------|
| **One-Hot Encoding** | Nominal categories, linear models | Low (<10) |
| **Label Encoding** | Ordinal categories, tree models | Any |
| **Target Encoding** | High cardinality | High (>10) |
| **Frequency Encoding** | When frequency matters | Any |
| **Binary Encoding** | Memory efficiency needed | Medium-High |
| **Embedding** | Deep learning | Very High |

**Implementation:**

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder

# 1. One-Hot Encoding
df_onehot = pd.get_dummies(df, columns=['color'], drop_first=True)

# 2. Label Encoding (for ordinal)
label_encoder = LabelEncoder()
df['education_encoded'] = label_encoder.fit_transform(df['education'])

# 3. Target Encoding (for high cardinality)
target_encoder = TargetEncoder(cols=['zip_code'])
df['zip_encoded'] = target_encoder.fit_transform(df['zip_code'], df['target'])

# 4. Frequency Encoding
freq_map = df['category'].value_counts(normalize=True).to_dict()
df['category_freq'] = df['category'].map(freq_map)

# 5. Binary Encoding
from category_encoders import BinaryEncoder
binary_encoder = BinaryEncoder(cols=['category'])
df_binary = binary_encoder.fit_transform(df)
```

**Handling Unknown Categories:**
```python
# Strategy 1: Add 'unknown' category
df['category'] = df['category'].fillna('unknown')

# Strategy 2: Use handle_unknown parameter
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
```

**Best Practices:**
1. Fit encoder on training data only
2. Handle missing values before encoding
3. Consider tree models for high cardinality (no encoding needed)
4. Watch for overfitting with target encoding (use cross-validation)

---

## Question 4: How can you use mutual information to select relevant features?
- [ ] Done

### Answer

**What is Mutual Information?**

Mutual Information (MI) measures the dependency between two variables. Unlike correlation, it can capture **any** kind of relationship (linear and non-linear).

$$I(X; Y) = H(Y) - H(Y|X)$$

Where:
- $H(Y)$ = Entropy of target
- $H(Y|X)$ = Conditional entropy of target given feature

**Key Properties:**
- MI ≥ 0 (always non-negative)
- MI = 0 means independence
- Higher MI = stronger relationship
- Captures non-linear dependencies

**Implementation:**

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression

# For Classification
selector = SelectKBest(score_func=mutual_info_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Get scores for all features
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({
    'feature': X.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print(mi_df.head(10))

# For Regression
mi_scores_reg = mutual_info_regression(X, y, random_state=42)
```

**Comparison with Correlation:**

| Aspect | Mutual Information | Correlation |
|--------|-------------------|-------------|
| **Relationship Type** | Any (linear, non-linear) | Linear only |
| **Value Range** | [0, ∞) | [-1, 1] |
| **Interpretation** | Less intuitive | More intuitive |
| **Computation** | Slower | Faster |

**Best Practices:**
```python
# Normalize numerical features before MI calculation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use multiple random states for stability
mi_scores_list = []
for seed in range(5):
    mi_scores_list.append(mutual_info_classif(X_scaled, y, random_state=seed))
mi_scores_avg = np.mean(mi_scores_list, axis=0)
```

---

## Question 5: How do you deal with missing values during feature engineering?
- [ ] Done

### Answer

**Missing Value Strategies:**

| Strategy | When to Use | Impact |
|----------|-------------|--------|
| **Deletion** | MCAR, few missing values | Loss of data |
| **Mean/Median** | Numerical, MCAR | Simple baseline |
| **Mode** | Categorical | Simple baseline |
| **KNN Imputation** | Values depend on neighbors | Better accuracy |
| **Iterative** | Complex relationships | Best accuracy |
| **Indicator Feature** | Missingness is informative | Preserves signal |

**Implementation:**

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# 1. Simple Imputation
mean_imputer = SimpleImputer(strategy='mean')
median_imputer = SimpleImputer(strategy='median')
mode_imputer = SimpleImputer(strategy='most_frequent')

# 2. KNN Imputation
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(X)

# 3. Iterative Imputation (MICE)
from sklearn.experimental import enable_iterative_imputer
iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
X_imputed = iterative_imputer.fit_transform(X)

# 4. Missing Indicator Feature (Important!)
def add_missing_indicators(df, columns):
    for col in columns:
        df[f'{col}_is_missing'] = df[col].isnull().astype(int)
    return df

df = add_missing_indicators(df, ['age', 'income', 'education'])
```

**Domain-Specific Imputation:**
```python
# Example: Fill missing values based on related features
df['income'].fillna(
    df.groupby('job_title')['income'].transform('median'), 
    inplace=True
)

# Example: Forward fill for time series
df['value'].fillna(method='ffill', inplace=True)
```

**Pipeline Integration:**
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

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
```

---

## Question 6: How do you detect and treat outliers during feature engineering?
- [ ] Done

### Answer

**Detection Methods:**

| Method | Formula/Approach | Best For |
|--------|------------------|----------|
| **Z-Score** | $z = (x - \mu) / \sigma$ | Normal distributions |
| **IQR** | $[Q1 - 1.5×IQR, Q3 + 1.5×IQR]$ | Skewed distributions |
| **MAD** | Median Absolute Deviation | Robust to outliers |
| **Isolation Forest** | ML-based | Complex patterns |
| **DBSCAN** | Density-based clustering | Multivariate outliers |

**Detection Implementation:**

```python
import numpy as np
from scipy import stats

# 1. Z-Score Method
def detect_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores > threshold].index

# 2. IQR Method
def detect_outliers_iqr(df, column, k=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    return df[(df[column] < lower) | (df[column] > upper)].index

# 3. Isolation Forest
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(df[['feature1', 'feature2']])
outlier_indices = df[outliers == -1].index
```

**Treatment Methods:**

```python
# 1. Removal (use cautiously)
df_clean = df[~df.index.isin(outlier_indices)]

# 2. Capping (Winsorization)
def cap_outliers(df, column, lower_percentile=1, upper_percentile=99):
    lower = df[column].quantile(lower_percentile / 100)
    upper = df[column].quantile(upper_percentile / 100)
    df[column] = df[column].clip(lower, upper)
    return df

# 3. Log Transformation (reduces impact)
df['amount_log'] = np.log1p(df['amount'])

# 4. Binning (converts to categories)
df['amount_bin'] = pd.qcut(df['amount'], q=10, labels=False, duplicates='drop')

# 5. Robust Scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()  # Uses median and IQR
df_scaled = scaler.fit_transform(df[['amount']])
```

**Best Practices:**
- Investigate outliers before removing (may be valid data)
- Use domain knowledge to set thresholds
- Document outlier treatment decisions
- Consider if outliers are the signal (fraud detection)

---

## Question 7: How do you evaluate the effectiveness of your engineered features?
- [ ] Done

### Answer

**Evaluation Methods:**

| Method | What It Measures | When to Use |
|--------|------------------|-------------|
| **Cross-Validation Score** | Model performance improvement | Always |
| **Feature Importance** | Individual feature contribution | Tree models |
| **Permutation Importance** | Performance drop when shuffled | Any model |
| **SHAP Values** | Feature contribution to predictions | Interpretability |
| **Learning Curves** | Overfitting risk | New features |

**Implementation:**

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap

# 1. Compare CV Scores Before/After
def evaluate_features(X_old, X_new, y, model):
    score_old = cross_val_score(model, X_old, y, cv=5).mean()
    score_new = cross_val_score(model, X_new, y, cv=5).mean()
    print(f"Before: {score_old:.4f}, After: {score_new:.4f}")
    print(f"Improvement: {(score_new - score_old) * 100:.2f}%")

# 2. Feature Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# 3. Permutation Importance
perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
perm_df = pd.DataFrame({
    'feature': X.columns,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False)

# 4. SHAP Values
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values[1], X)

# 5. Statistical Significance Test
from scipy.stats import ttest_rel
scores_old = cross_val_score(model, X_old, y, cv=5)
scores_new = cross_val_score(model, X_new, y, cv=5)
t_stat, p_value = ttest_rel(scores_new, scores_old)
print(f"p-value: {p_value:.4f}")
```

**Evaluation Checklist:**
- [ ] Does the feature improve cross-validation score?
- [ ] Is the improvement statistically significant?
- [ ] Does the feature have meaningful importance?
- [ ] Does it generalize to test set?
- [ ] Is there risk of data leakage?
- [ ] Is the feature interpretable and actionable?

---

## Question 8: How do you handle time-series data in feature engineering?
- [ ] Done

### Answer

**Key Feature Categories:**

| Category | Examples |
|----------|----------|
| **Lag Features** | value_t-1, value_t-7, value_t-30 |
| **Rolling Statistics** | rolling_mean_7d, rolling_std_30d |
| **Expanding Statistics** | cumsum, cummax, expanding_mean |
| **Date/Time Features** | hour, day_of_week, month, is_holiday |
| **Trend Features** | slope, acceleration |
| **Seasonal Features** | fourier_terms, seasonal_decomposition |

**Implementation:**

```python
import pandas as pd
import numpy as np

# 1. Lag Features
def create_lag_features(df, target_col, lags=[1, 7, 14, 30]):
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

# 2. Rolling Window Features
def create_rolling_features(df, target_col, windows=[7, 14, 30]):
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
        df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window).min()
        df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window).max()
    return df

# 3. Date/Time Features
def create_datetime_features(df, date_col):
    df['hour'] = df[date_col].dt.hour
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_month'] = df[date_col].dt.day
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    return df

# 4. Trend Features
def create_trend_features(df, target_col, window=7):
    # Simple slope
    df[f'{target_col}_slope_{window}'] = (
        df[target_col] - df[target_col].shift(window)
    ) / window
    # Percentage change
    df[f'{target_col}_pct_change_{window}'] = df[target_col].pct_change(window)
    return df

# 5. Seasonal Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df['value'], period=7)
df['trend'] = decomposition.trend
df['seasonal'] = decomposition.seasonal
df['residual'] = decomposition.resid
```

**Important: Avoid Data Leakage!**
```python
# WRONG: Uses future data
df['rolling_mean'] = df['value'].rolling(7, center=True).mean()

# CORRECT: Only uses past data
df['rolling_mean'] = df['value'].rolling(7).mean().shift(1)
```

---

## Question 9: How can you use domain knowledge to create interaction features?
- [ ] Done

### Answer

**Interaction Features:**
Interaction features capture the combined effect of two or more features that may not be apparent when looking at features individually.

**Domain-Driven Examples:**

**1. E-commerce:**
```python
# Price per unit (important for comparison)
df['price_per_unit'] = df['total_price'] / df['quantity']

# Discount percentage
df['discount_pct'] = (df['original_price'] - df['sale_price']) / df['original_price']

# Cart diversity (product type × basket size)
df['cart_diversity'] = df['n_unique_categories'] / df['n_items']
```

**2. Real Estate:**
```python
# Price per square foot (standard metric)
df['price_per_sqft'] = df['price'] / df['sqft']

# Room density
df['room_density'] = df['total_rooms'] / df['sqft'] * 1000

# Age × Renovation interaction
df['age_renovation_interaction'] = df['age'] * (1 - df['recently_renovated'])
```

**3. Finance:**
```python
# Debt-to-income ratio
df['dti'] = df['monthly_debt'] / df['monthly_income']

# Payment-to-income ratio
df['pti'] = df['monthly_payment'] / df['monthly_income']

# Credit utilization
df['credit_utilization'] = df['credit_used'] / df['credit_limit']
```

**4. Healthcare:**
```python
# BMI (standard health metric)
df['bmi'] = df['weight_kg'] / (df['height_m'] ** 2)

# Waist-to-hip ratio
df['whr'] = df['waist_cm'] / df['hip_cm']

# Age × Risk factor interactions
df['age_smoking_interaction'] = df['age'] * df['is_smoker']
```

**Automated Interaction Discovery:**
```python
from sklearn.preprocessing import PolynomialFeatures

# Create all pairwise interactions
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interactions = poly.fit_transform(X)

# Get feature names
feature_names = poly.get_feature_names_out(X.columns)
```

**Best Practices:**
- Start with domain knowledge (most valuable)
- Test interactions with tree-based models first
- Validate with cross-validation
- Watch for multicollinearity
- Keep interpretability in mind

---

## Question 10: Can you use genetic algorithms for feature engineering? If yes, how?
- [ ] Done

### Answer

**Yes! Genetic Algorithms (GA) can automate feature engineering and selection.**

**How GA Works for Feature Selection:**

```
1. Initialization: Create population of random feature subsets (binary chromosomes)
2. Fitness: Evaluate each subset using model performance (e.g., CV accuracy)
3. Selection: Select fittest individuals (best performing subsets)
4. Crossover: Combine parent chromosomes to create offspring
5. Mutation: Randomly flip bits (add/remove features)
6. Repeat: Until convergence or max generations
```

**Implementation:**

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

class GeneticFeatureSelector:
    def __init__(self, n_features, n_pop=50, n_gen=100, 
                 crossover_rate=0.8, mutation_rate=0.1):
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
    
    def initialize_population(self):
        return np.random.randint(0, 2, (self.n_pop, self.n_features))
    
    def fitness(self, chromosome, X, y):
        selected_features = np.where(chromosome == 1)[0]
        if len(selected_features) == 0:
            return 0
        X_subset = X.iloc[:, selected_features]
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        score = cross_val_score(model, X_subset, y, cv=3).mean()
        return score
    
    def selection(self, population, fitness_scores):
        # Tournament selection
        idx1, idx2 = np.random.choice(len(population), 2, replace=False)
        return population[idx1] if fitness_scores[idx1] > fitness_scores[idx2] else population[idx2]
    
    def crossover(self, parent1, parent2):
        if np.random.random() < self.crossover_rate:
            point = np.random.randint(1, self.n_features - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, chromosome):
        for i in range(len(chromosome)):
            if np.random.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        return chromosome
    
    def evolve(self, X, y):
        population = self.initialize_population()
        best_chromosome = None
        best_score = 0
        
        for gen in range(self.n_gen):
            fitness_scores = [self.fitness(chrom, X, y) for chrom in population]
            
            # Track best
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > best_score:
                best_score = fitness_scores[max_idx]
                best_chromosome = population[max_idx].copy()
            
            # Create new population
            new_population = []
            while len(new_population) < self.n_pop:
                parent1 = self.selection(population, fitness_scores)
                parent2 = self.selection(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([self.mutate(child1), self.mutate(child2)])
            
            population = np.array(new_population[:self.n_pop])
            
            if gen % 10 == 0:
                print(f"Gen {gen}: Best Score = {best_score:.4f}")
        
        return best_chromosome, best_score

# Usage
ga = GeneticFeatureSelector(n_features=X.shape[1], n_pop=50, n_gen=100)
best_features, best_score = ga.evolve(X, y)
selected_cols = X.columns[best_features == 1]
print(f"Selected {len(selected_cols)} features with score {best_score:.4f}")
```

**Advantages:**
- Can find globally optimal feature subset
- No assumptions about feature importance
- Works with any model

**Disadvantages:**
- Computationally expensive
- Requires many hyperparameters
- May overfit the selection process

**Alternatives:**
- DEAP library for more advanced GA
- Genetic Programming for feature construction
- AutoML tools (TPOT, AutoFeat)

---

---

## Question 11: How would you balance the trade-off between overfitting and underfitting while engineering features?
- [ ] Done

### Answer

**The Feature Engineering Bias-Variance Trade-off:**

| Problem | Cause | Symptoms |
|---------|-------|----------|
| **Underfitting** | Too few/simple features | High training error, high test error |
| **Overfitting** | Too many/complex features | Low training error, high test error |

**Strategy Framework:**

```
1. Start Simple → Add Complexity → Regularize → Validate
```

**Step-by-Step Approach:**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def evaluate_feature_complexity(X, y, max_degree=5):
    """
    Evaluate model performance across feature complexity levels.
    """
    results = []
    
    for degree in range(1, max_degree + 1):
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ])
        
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
        
        pipeline.fit(X, y)
        n_features = pipeline.named_steps['poly'].n_output_features_
        
        results.append({
            'degree': degree,
            'n_features': n_features,
            'cv_rmse_mean': np.sqrt(-scores.mean()),
            'cv_rmse_std': np.sqrt(-scores).std()
        })
    
    return pd.DataFrame(results)
```

**Techniques to Prevent Overfitting:**

```python
# 1. Feature Selection with Regularization
from sklearn.linear_model import LassoCV

def regularized_feature_selection(X, y, alpha_range=None):
    """Use Lasso for automatic feature selection."""
    if alpha_range is None:
        alpha_range = np.logspace(-4, 1, 50)
    
    lasso_cv = LassoCV(alphas=alpha_range, cv=5, random_state=42)
    lasso_cv.fit(X, y)
    
    # Get selected features (non-zero coefficients)
    selected_mask = lasso_cv.coef_ != 0
    selected_features = X.columns[selected_mask].tolist()
    
    print(f"Optimal alpha: {lasso_cv.alpha_:.4f}")
    print(f"Selected {len(selected_features)} of {X.shape[1]} features")
    
    return selected_features, lasso_cv

# 2. Learning Curves for Diagnosis
def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    """Plot learning curve to diagnose bias-variance."""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    
    train_rmse = np.sqrt(-train_scores.mean(axis=1))
    test_rmse = np.sqrt(-test_scores.mean(axis=1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_rmse, 'o-', label='Training RMSE')
    plt.plot(train_sizes, test_rmse, 'o-', label='Validation RMSE')
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Diagnosis
    gap = test_rmse[-1] - train_rmse[-1]
    if gap > 0.1 * test_rmse[-1]:
        print("Diagnosis: HIGH VARIANCE (Overfitting) - Reduce features or add regularization")
    elif train_rmse[-1] > 0.5:  # Adjust threshold based on problem
        print("Diagnosis: HIGH BIAS (Underfitting) - Add more features")
    else:
        print("Diagnosis: Good balance")
```

**Balanced Feature Engineering Pipeline:**

```python
def balanced_feature_engineering(X, y, target_variance_explained=0.95):
    """
    Balanced approach to feature engineering.
    """
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectFromModel
    
    # Step 1: Generate rich features
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    X_poly = poly.fit_transform(X)
    print(f"Step 1: Generated {X_poly.shape[1]} polynomial features")
    
    # Step 2: Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    # Step 3: Apply regularized selection
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_scaled, y)
    selector = SelectFromModel(lasso, prefit=True)
    X_selected = selector.transform(X_scaled)
    print(f"Step 2: Selected {X_selected.shape[1]} features via Lasso")
    
    # Step 4: Optional PCA for remaining multicollinearity
    pca = PCA(n_components=target_variance_explained)
    X_final = pca.fit_transform(X_selected)
    print(f"Step 3: PCA reduced to {X_final.shape[1]} components")
    
    return X_final
```

**Best Practices Summary:**

| Stage | Action | Purpose |
|-------|--------|---------|
| **Initial** | Start with domain-driven features | Avoid random complexity |
| **Expansion** | Add polynomial/interaction features | Increase expressiveness |
| **Selection** | Apply Lasso/RF importance | Remove noise features |
| **Validation** | Use cross-validation | Detect overfitting early |
| **Regularization** | Tune regularization strength | Control complexity |
| **Monitoring** | Plot learning curves | Diagnose bias-variance |

---
