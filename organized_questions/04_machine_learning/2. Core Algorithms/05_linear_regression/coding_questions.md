# Linear Regression Interview Questions - Coding Questions

## Question 1: Implement simple linear regression from scratch in Python
- [ ] Done

### Answer

**Definition:**
Simple linear regression finds the best-fit line $y = mx + b$ by minimizing sum of squared errors using the closed-form OLS solution.

**Core Formulas:**
- Slope: $m = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}$
- Intercept: $b = \bar{y} - m\bar{x}$

**Python Code:**
```python
import numpy as np

# Pipeline: Calculate means → Compute slope → Compute intercept → Predict

class SimpleLinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None
    
    def fit(self, X, y):
        """Learn slope and intercept from data"""
        X = X.flatten() if X.ndim > 1 else X
        
        # Step 1: Calculate means
        mean_x, mean_y = np.mean(X), np.mean(y)
        
        # Step 2: Calculate slope using OLS formula
        numerator = np.sum((X - mean_x) * (y - mean_y))
        denominator = np.sum((X - mean_x) ** 2)
        self.slope = numerator / denominator
        
        # Step 3: Calculate intercept
        self.intercept = mean_y - self.slope * mean_x
        
    def predict(self, X):
        """Predict using learned parameters"""
        X = X.flatten() if X.ndim > 1 else X
        return self.slope * X + self.intercept

# Usage
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

model = SimpleLinearRegression()
model.fit(X, y)
print(f"Slope: {model.slope:.4f}, Intercept: {model.intercept:.4f}")
# Output: Slope: 0.6000, Intercept: 2.2000
```

**Algorithm Steps:**
1. Compute mean of X and y
2. Calculate numerator: $\sum(x_i - \bar{x})(y_i - \bar{y})$
3. Calculate denominator: $\sum(x_i - \bar{x})^2$
4. Slope = numerator / denominator
5. Intercept = $\bar{y}$ - slope × $\bar{x}$

---

## Question 2: Implement a multiple linear regression model using NumPy
- [ ] Done

### Answer

**Definition:**
Multiple linear regression uses the Normal Equation $\beta = (X^TX)^{-1}X^Ty$ to find optimal coefficients for multiple features.

**Python Code:**
```python
import numpy as np

# Pipeline: Add bias column → Apply normal equation → Store coefficients

class MultipleLinearRegression:
    def __init__(self):
        self.coefficients = None
    
    def fit(self, X, y):
        """Fit using Normal Equation: β = (X'X)^(-1)X'y"""
        # Step 1: Add column of ones for intercept
        ones = np.ones((X.shape[0], 1))
        X_b = np.concatenate([ones, X], axis=1)
        
        # Step 2: Apply Normal Equation
        # β = (X'X)^(-1) X' y
        X_T_X = X_b.T @ X_b
        X_T_X_inv = np.linalg.inv(X_T_X)
        self.coefficients = X_T_X_inv @ X_b.T @ y
        
    def predict(self, X):
        """Predict: y = X_b @ β"""
        ones = np.ones((X.shape[0], 1))
        X_b = np.concatenate([ones, X], axis=1)
        return X_b @ self.coefficients

# Usage
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([5, 8, 11, 14])  # y = 2 + 1*x1 + 1.5*x2

model = MultipleLinearRegression()
model.fit(X, y)
print(f"Intercept: {model.coefficients[0]:.4f}")
print(f"Slopes: {model.coefficients[1:]}")
```

**Algorithm Steps:**
1. Add bias column (ones) to X → creates $X_b$
2. Compute $X^TX$
3. Compute inverse $(X^TX)^{-1}$
4. Compute $\beta = (X^TX)^{-1}X^Ty$
5. First coefficient is intercept, rest are slopes

---

## Question 3: Write a Python function for gradient descent in linear regression
- [ ] Done

### Answer

**Definition:**
Gradient descent iteratively updates weights by moving in the direction opposite to the gradient of the loss function.

**Update Rule:** $\beta_{new} = \beta_{old} - \alpha \cdot \nabla Loss$

**Python Code:**
```python
import numpy as np

# Pipeline: Initialize weights → Loop(predict → compute gradient → update) → Return

def gradient_descent_lr(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Perform gradient descent for linear regression.
    Returns: weights, bias
    """
    n_samples, n_features = X.shape
    
    # Step 1: Initialize parameters to zero
    weights = np.zeros(n_features)
    bias = 0
    
    # Step 2: Iterate
    for _ in range(n_iterations):
        # Step 2a: Compute predictions
        y_pred = np.dot(X, weights) + bias
        
        # Step 2b: Compute gradients
        # dL/dw = (2/n) * X' * (y_pred - y)
        # dL/db = (2/n) * sum(y_pred - y)
        dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (2 / n_samples) * np.sum(y_pred - y)
        
        # Step 2c: Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db
    
    return weights, bias

# Usage (scale features first for better convergence)
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=3, noise=10)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

weights, bias = gradient_descent_lr(X_scaled, y, learning_rate=0.1, n_iterations=1000)
print(f"Weights: {weights}")
print(f"Bias: {bias:.4f}")
```

**Algorithm Steps:**
1. Initialize weights = 0, bias = 0
2. For each iteration:
   - Predict: $\hat{y} = Xw + b$
   - Compute gradient of weights: $\frac{2}{n}X^T(\hat{y} - y)$
   - Compute gradient of bias: $\frac{2}{n}\sum(\hat{y} - y)$
   - Update: $w = w - \alpha \cdot dw$, $b = b - \alpha \cdot db$
3. Return final weights and bias

---

## Question 4: Create a Python script to calculate VIF for each predictor
- [ ] Done

### Answer

**Definition:**
VIF (Variance Inflation Factor) measures multicollinearity. $VIF_i = \frac{1}{1-R^2_i}$ where $R^2_i$ is from regressing feature $i$ against all others.

**Interpretation:** VIF > 5-10 indicates problematic multicollinearity.

**Python Code:**
```python
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Pipeline: Get features → Calculate VIF for each → Return sorted results

def calculate_vif(df):
    """
    Calculate VIF for each feature in DataFrame.
    Returns: DataFrame with feature names and VIF scores
    """
    vif_data = pd.DataFrame()
    vif_data['Feature'] = df.columns
    vif_data['VIF'] = [
        variance_inflation_factor(df.values, i) 
        for i in range(len(df.columns))
    ]
    return vif_data.sort_values('VIF', ascending=False)

# Usage
np.random.seed(42)
df = pd.DataFrame({
    'x1': np.random.randn(100),
    'x2': np.random.randn(100),
    'x3_correlated': np.random.randn(100)  # Will add correlation
})
df['x3_correlated'] = df['x1'] * 0.9 + np.random.randn(100) * 0.1  # Correlated with x1

vif_results = calculate_vif(df)
print(vif_results)

# Interpretation
for _, row in vif_results.iterrows():
    status = "HIGH - remove or combine" if row['VIF'] > 5 else "OK"
    print(f"{row['Feature']}: VIF = {row['VIF']:.2f} → {status}")
```

**Output Example:**
```
        Feature       VIF
0            x1      5.23  → HIGH
2  x3_correlated    5.18  → HIGH
1            x2      1.02  → OK
```

**Algorithm Steps:**
1. For each feature $i$:
   - Regress feature $i$ against all other features
   - Get $R^2_i$ from this regression
   - Calculate $VIF_i = \frac{1}{1-R^2_i}$
2. Sort by VIF descending
3. Flag features with VIF > 5 or 10

---

## Question 5: Code Ridge regression using scikit-learn
- [ ] Done

### Answer

**Definition:**
Ridge regression adds L2 penalty to prevent overfitting: $Loss = MSE + \alpha\sum\beta_j^2$

**Python Code:**
```python
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Pipeline: Scale data → Fit Ridge → Evaluate → (Optional) Use RidgeCV for auto-tuning

def ridge_regression(X_train, y_train, X_test, y_test, alpha=1.0):
    """
    Perform Ridge Regression with given alpha.
    Returns: model, predictions, RMSE
    """
    # Step 1: Scale features (crucial for regularization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 2: Fit Ridge model
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    
    # Step 3: Predict and evaluate
    y_pred = ridge.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return ridge, y_pred, rmse

# Usage
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=200, n_features=20, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model, preds, rmse = ridge_regression(X_train, y_train, X_test, y_test, alpha=1.0)
print(f"RMSE: {rmse:.4f}")

# Auto-tune alpha with RidgeCV
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
ridge_cv.fit(X_train_s, y_train)
print(f"Best alpha: {ridge_cv.alpha_}")
```

**Key Points:**
- Always scale features before Ridge
- Higher alpha → more shrinkage → simpler model
- Use RidgeCV to automatically find best alpha
- Ridge does NOT set coefficients to exactly zero

---

## Question 6: Use pandas to load and prepare data for linear regression
- [ ] Done

### Answer

**Definition:**
Data preparation involves: loading data, handling missing values, encoding categoricals, and splitting data.

**Python Code:**
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Pipeline: Load → Check missing → Impute → Encode categoricals → Split

def prepare_data(filepath):
    """
    Load and prepare data for linear regression.
    Returns: X_train, X_test, y_train, y_test
    """
    # Step 1: Load data
    df = pd.read_csv(filepath)
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    
    # Step 2: Handle missing values
    # Numeric: fill with median (robust to outliers)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Categorical: fill with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Step 3: One-hot encode categorical variables
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Step 4: Separate features and target
    # Assume last column or specify target
    target_col = 'target'  # Change as needed
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Step 5: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

# Example with sample data creation
df = pd.DataFrame({
    'size': [1500, 2000, np.nan, 1800, 2200],
    'bedrooms': [3, 4, 3, np.nan, 5],
    'location': ['A', 'B', 'A', np.nan, 'B'],
    'target': [300, 400, 320, 350, 450]
})
df.to_csv('sample.csv', index=False)

X_train, X_test, y_train, y_test = prepare_data('sample.csv')
print("\nTraining shape:", X_train.shape)
```

**Key Steps:**
1. Load with `pd.read_csv()`
2. Check missing: `df.isnull().sum()`
3. Impute numeric with median, categorical with mode
4. One-hot encode with `pd.get_dummies(drop_first=True)`
5. Split with `train_test_split()`

---

## Question 7: Plot residual diagrams and analyze model fit
- [ ] Done

### Answer

**Definition:**
Residual plot shows predicted values vs residuals. Good fit = random scatter around y=0. Patterns indicate assumption violations.

**Python Code:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Pipeline: Fit model → Calculate residuals → Plot → Interpret patterns

def plot_residuals(y_true, y_pred, title="Residual Plot"):
    """
    Create and display residual plot.
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.show()
    
    # Quick diagnostics
    print(f"Mean of residuals: {np.mean(residuals):.4f} (should be ~0)")
    print(f"Std of residuals: {np.std(residuals):.4f}")

# Usage
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=1, noise=10)

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plot_residuals(y, y_pred, "Good Fit - Random Scatter")

# Example of bad fit (non-linear data with linear model)
X_nonlinear = np.linspace(-3, 3, 100).reshape(-1, 1)
y_nonlinear = X_nonlinear**2 + np.random.randn(100, 1) * 0.5

model_bad = LinearRegression()
model_bad.fit(X_nonlinear, y_nonlinear)
y_pred_bad = model_bad.predict(X_nonlinear)

plot_residuals(y_nonlinear.flatten(), y_pred_bad.flatten(), "Bad Fit - Curved Pattern")
```

**Pattern Interpretation:**

| Pattern | Diagnosis | Fix |
|---------|-----------|-----|
| Random scatter | Good fit ✓ | None |
| U-shape/curve | Non-linearity | Add polynomial features |
| Fan/cone shape | Heteroscedasticity | Log transform target |
| Clusters | Missing categorical | Add indicator variable |

---

## Question 8: Write a function to compute RMSE, MAE, and R-squared
- [ ] Done

### Answer

**Definition:**
- **RMSE**: $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ — Average error in original units
- **MAE**: $\frac{1}{n}\sum|y_i - \hat{y}_i|$ — Average absolute error
- **R²**: $1 - \frac{SS_{res}}{SS_{tot}}$ — Proportion of variance explained

**Python Code:**
```python
import numpy as np

# Pipeline: Calculate errors → Compute each metric → Return dictionary

def regression_metrics(y_true, y_pred):
    """
    Calculate RMSE, MAE, and R-squared.
    Returns: dict with all metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errors = y_true - y_pred
    
    # MAE: Mean Absolute Error
    mae = np.mean(np.abs(errors))
    
    # MSE and RMSE
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

# Usage
y_true = np.array([100, 150, 200, 250, 300])
y_pred = np.array([110, 145, 190, 260, 310])

metrics = regression_metrics(y_true, y_pred)
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# Verify with sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print("\n--- Sklearn Verification ---")
print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
print(f"R2: {r2_score(y_true, y_pred):.4f}")
```

**Algorithm Steps:**
1. Calculate errors: $e_i = y_i - \hat{y}_i$
2. MAE = mean(|errors|)
3. MSE = mean(errors²), RMSE = √MSE
4. R² = 1 - (sum(errors²) / sum((y - mean(y))²))

---

## Question 9: Perform polynomial regression and plot results
- [ ] Done

### Answer

**Definition:**
Polynomial regression creates polynomial features (x, x², x³...) then fits linear regression. Still linear in coefficients, but captures non-linear patterns.

**Python Code:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Pipeline: Create poly features → Fit linear model → Predict → Plot

# Step 1: Create non-linear data
np.random.seed(42)
X = np.linspace(-3, 3, 50).reshape(-1, 1)
y = 0.5 * X**2 + X + 2 + np.random.randn(50, 1) * 0.5

# Step 2: Create pipeline (PolynomialFeatures + LinearRegression)
degree = 2
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
    ('linear', LinearRegression())
])

# Step 3: Fit model
poly_model.fit(X, y)

# Step 4: Predict for smooth curve
X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
y_pred = poly_model.predict(X_plot)

# Step 5: Plot
plt.scatter(X, y, label='Data', alpha=0.7)
plt.plot(X_plot, y_pred, color='red', label=f'Poly (degree={degree})')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Polynomial Regression')
plt.show()

print(f"Coefficients: {poly_model.named_steps['linear'].coef_}")
```

**Key Points:**
- Use Pipeline to chain PolynomialFeatures + LinearRegression
- Higher degree = more flexible but risk of overfitting
- Use cross-validation to select optimal degree

---

## Question 10: Perform cross-validation on linear regression using scikit-learn
- [ ] Done

### Answer

**Definition:**
Cross-validation splits data into k folds, trains on k-1 folds, tests on 1 fold, repeats k times. Gives robust performance estimate.

**Python Code:**
```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Pipeline: Create model → Define CV → Get scores → Analyze

# Step 1: Create data
X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)

# Step 2: Create model
model = LinearRegression()

# Step 3: Perform 5-fold cross-validation
k = 5

# Get R² scores
r2_scores = cross_val_score(model, X, y, cv=k, scoring='r2')

# Get RMSE (note: sklearn uses neg_mean_squared_error)
neg_mse = cross_val_score(model, X, y, cv=k, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-neg_mse)

# Step 4: Analyze results
print(f"R² scores per fold: {np.round(r2_scores, 4)}")
print(f"Mean R²: {r2_scores.mean():.4f} (+/- {r2_scores.std()*2:.4f})")
print(f"\nRMSE per fold: {np.round(rmse_scores, 4)}")
print(f"Mean RMSE: {rmse_scores.mean():.4f}")
```

**Output Example:**
```
R² scores per fold: [0.9234, 0.9456, 0.9123, 0.9345, 0.9278]
Mean R²: 0.9287 (+/- 0.0234)
```

**Key Points:**
- `cross_val_score` handles everything automatically
- Use `neg_mean_squared_error` and negate for MSE
- Report mean ± 2*std for confidence interval

---

## Question 11: Implement linear regression to predict Customer Lifetime Value
- [ ] Done

### Answer

**Definition:**
CLV prediction uses RFM features (Recency, Frequency, Monetary) to predict total future value of a customer.

**Python Code:**
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Pipeline: Create RFM features → Scale → Fit → Evaluate → Interpret

# Step 1: Create sample customer data
np.random.seed(42)
df = pd.DataFrame({
    'recency': np.random.randint(1, 365, 500),      # Days since last purchase
    'frequency': np.random.randint(1, 50, 500),     # Number of purchases
    'monetary': np.random.normal(50, 15, 500),      # Avg purchase value
    'tenure': np.random.randint(30, 730, 500)       # Days as customer
})

# Create target: CLV based on features
df['clv'] = (df['frequency'] * df['monetary'] * 1.5 
             - df['recency'] * 0.3 
             + df['tenure'] * 0.1
             + np.random.normal(0, 30, 500))

# Step 2: Prepare data
X = df[['recency', 'frequency', 'monetary', 'tenure']]
y = df['clv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Step 4: Fit model
model = LinearRegression()
model.fit(X_train_s, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test_s)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# Step 6: Interpret coefficients
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:.4f}")
```

**Interpretation:**
- Positive coefficient on frequency/monetary → more purchases = higher CLV
- Negative coefficient on recency → recent customers have higher CLV

---

## Question 12: Develop regularized regression for healthcare costs prediction
- [ ] Done

### Answer

**Definition:**
Use Elastic Net (L1+L2) for healthcare data with many features and potential multicollinearity. Handles feature selection automatically.

**Python Code:**
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Pipeline: Preprocess → Create full pipeline → GridSearch → Evaluate

# Step 1: Create sample healthcare data
np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.randint(20, 70, 300),
    'bmi': np.random.normal(27, 5, 300),
    'smoker': np.random.choice(['yes', 'no'], 300, p=[0.2, 0.8]),
    'region': np.random.choice(['north', 'south', 'east', 'west'], 300)
})
df['charges'] = (200*df['age'] + 300*df['bmi'] + 
                 15000*(df['smoker']=='yes') + 
                 np.random.normal(0, 1500, 300))

# Step 2: Define columns
num_cols = ['age', 'bmi']
cat_cols = ['smoker', 'region']

X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 3: Create preprocessing + model pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first'), cat_cols)
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', ElasticNet(max_iter=2000))
])

# Step 4: GridSearch for hyperparameters
param_grid = {
    'model__alpha': [0.1, 1.0, 10.0],
    'model__l1_ratio': [0.2, 0.5, 0.8]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

# Step 5: Results
print(f"Best params: {grid.best_params_}")
y_pred = grid.predict(X_test)
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
```

**Key Points:**
- Use ColumnTransformer for mixed data types
- Pipeline prevents data leakage in CV
- Elastic Net balances feature selection (L1) and stability (L2)

---

## Question 13: Perform time-series linear regression on stock data
- [ ] Done

### Answer

**Definition:**
Time-series regression uses lag features and time trends to predict future values. Key: no shuffling, chronological split.

**Python Code:**
```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Pipeline: Create lag features → Chronological split → Fit → Predict

# Step 1: Create sample stock data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=200, freq='D')
prices = 100 + np.cumsum(np.random.randn(200) * 2)  # Random walk
df = pd.DataFrame({'date': dates, 'price': prices})

# Step 2: Feature engineering - lag features
df['lag_1'] = df['price'].shift(1)   # Yesterday's price
df['lag_2'] = df['price'].shift(2)   # 2 days ago
df['rolling_mean_5'] = df['price'].shift(1).rolling(5).mean()
df['trend'] = np.arange(len(df))     # Time trend

# Drop NaN rows
df = df.dropna()

# Step 3: Chronological split (NO SHUFFLE)
train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]

features = ['lag_1', 'lag_2', 'rolling_mean_5', 'trend']
X_train, y_train = train[features], train['price']
X_test, y_test = test[features], test['price']

# Step 4: Fit and predict
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 5: Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.4f}")

# Naive baseline comparison
naive_rmse = np.sqrt(mean_squared_error(y_test, X_test['lag_1']))
print(f"Naive RMSE (yesterday's price): {naive_rmse:.4f}")
```

**Key Points:**
- Always use chronological split (no shuffling)
- Create lag features using `.shift()`
- Compare against naive baseline

---

## Question 14: Tune Elastic Net hyperparameters using GridSearch
- [ ] Done

### Answer

**Definition:**
Elastic Net has two hyperparameters: alpha (regularization strength) and l1_ratio (L1/L2 mix). GridSearchCV finds optimal combination.

**Python Code:**
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Pipeline: Scale → Define param grid → GridSearch → Get best model

# Step 1: Create data
X, y = make_regression(n_samples=300, n_features=50, n_informative=10, noise=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 2: Create pipeline with scaling
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('elasticnet', ElasticNet(max_iter=2000, random_state=42))
])

# Step 3: Define parameter grid
param_grid = {
    'elasticnet__alpha': [0.01, 0.1, 1.0, 10.0],
    'elasticnet__l1_ratio': [0.2, 0.5, 0.8, 1.0]  # 1.0 = Lasso
}

# Step 4: Run GridSearchCV
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Step 5: Results
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")

# Test set evaluation
y_pred = grid_search.predict(X_test)
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# Check sparsity
best_model = grid_search.best_estimator_.named_steps['elasticnet']
print(f"Non-zero coefficients: {np.sum(best_model.coef_ != 0)}/{len(best_model.coef_)}")
```

**Parameter Guide:**
- `alpha`: Higher = more regularization
- `l1_ratio=1.0`: Pure Lasso (sparse)
- `l1_ratio=0.0`: Pure Ridge (no sparsity)

---

## Question 15: Analyze polynomial complexity trade-off
- [ ] Done

### Answer

**Definition:**
Higher polynomial degree = lower training error but risk of overfitting. Plot train vs validation error across degrees to find optimal complexity.

**Python Code:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Pipeline: Loop degrees → Fit → Track train/val error → Plot trade-off

def analyze_poly_complexity(X, y, max_degree=10):
    """Analyze bias-variance trade-off across polynomial degrees."""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    train_errors = []
    val_errors = []
    degrees = range(1, max_degree + 1)
    
    for degree in degrees:
        # Create polynomial pipeline
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('reg', LinearRegression())
        ])
        
        # Fit and predict
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_errors.append(np.sqrt(mean_squared_error(y_train, train_pred)))
        val_errors.append(np.sqrt(mean_squared_error(y_val, val_pred)))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_errors, 'b-o', label='Training RMSE')
    plt.plot(degrees, val_errors, 'r-o', label='Validation RMSE')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('RMSE')
    plt.title('Bias-Variance Trade-off')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    best_degree = degrees[np.argmin(val_errors)]
    print(f"Best degree: {best_degree} (Val RMSE: {min(val_errors):.4f})")
    return best_degree

# Usage
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5*X.flatten()**3 - X.flatten()**2 + np.random.randn(100) * 2

best = analyze_poly_complexity(X, y, max_degree=10)
```

**Interpretation:**
- Training error always decreases with complexity
- Validation error shows U-shape: minimum at optimal degree
- Choose degree where validation error is minimum

