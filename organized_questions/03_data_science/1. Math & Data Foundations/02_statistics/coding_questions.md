
## Coding questions

59. Write Python code to calculate mean, median, and mode from a given list of numbers.

**Answer:**

```python
import numpy as np
from scipy import stats

def calculate_central_tendency(data):
    """Calculate mean, median, and mode from a list of numbers."""
    # Mean: Sum of all values / count
    mean = sum(data) / len(data)
    # Or: mean = np.mean(data)
    
    # Median: Middle value when sorted
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 0:
        median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    else:
        median = sorted_data[n//2]
    # Or: median = np.median(data)
    
    # Mode: Most frequent value
    from collections import Counter
    freq = Counter(data)
    mode = freq.most_common(1)[0][0]
    # Or: mode = stats.mode(data, keepdims=False).mode
    
    return mean, median, mode

# Example usage
data = [1, 2, 2, 3, 4, 4, 4, 5, 6]
mean, median, mode = calculate_central_tendency(data)
print(f"Mean: {mean}")      # 3.44
print(f"Median: {median}")  # 4
print(f"Mode: {mode}")      # 4
```

---

60. Generate and visualize 1,000 random points from a Normal distribution in Python.

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Set parameters
mu = 0      # Mean
sigma = 1   # Standard deviation
n = 1000    # Number of points

# Step 2: Generate random samples
samples = np.random.normal(mu, sigma, n)

# Step 3: Visualize with histogram
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')

# Step 4: Overlay theoretical PDF
x = np.linspace(-4, 4, 100)
pdf = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
plt.plot(x, pdf, 'r-', linewidth=2, label='Theoretical PDF')

# Step 5: Labels and display
plt.xlabel('Value')
plt.ylabel('Density')
plt.title(f'Normal Distribution: μ={mu}, σ={sigma}, n={n}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Verify statistics
print(f"Sample Mean: {np.mean(samples):.3f}")
print(f"Sample Std: {np.std(samples):.3f}")
```

---

61. Implement a simple linear regression model from scratch in Python.

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    """Linear Regression: y = mx + b using Ordinary Least Squares"""
    
    def __init__(self):
        self.m = None  # Slope
        self.b = None  # Intercept
    
    def fit(self, X, y):
        """
        Calculate slope and intercept using OLS formulas:
        m = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
        b = ȳ - m * x̄
        """
        n = len(X)
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # Numerator: Σ(xi - x̄)(yi - ȳ)
        numerator = sum((X[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        
        # Denominator: Σ(xi - x̄)²
        denominator = sum((X[i] - x_mean)**2 for i in range(n))
        
        self.m = numerator / denominator
        self.b = y_mean - self.m * x_mean
        
        return self
    
    def predict(self, X):
        """Predict: y = mx + b"""
        return self.m * np.array(X) + self.b
    
    def r_squared(self, X, y):
        """Calculate R-squared"""
        y_pred = self.predict(X)
        ss_res = sum((y - y_pred)**2)
        ss_tot = sum((y - np.mean(y))**2)
        return 1 - (ss_res / ss_tot)

# Example usage
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

model = SimpleLinearRegression()
model.fit(X, y)

print(f"Slope (m): {model.m:.3f}")
print(f"Intercept (b): {model.b:.3f}")
print(f"R-squared: {model.r_squared(X, y):.3f}")

# Visualize
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, model.predict(X), color='red', label=f'y = {model.m:.2f}x + {model.b:.2f}')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```

---

62. Simulate the Monty Hall problem in Python and analyze the results.

**Answer:**

```python
import random
import numpy as np

def monty_hall_simulation(n_games=10000):
    """
    Monty Hall Problem:
    - 3 doors: 1 has prize, 2 have goats
    - You pick a door
    - Host opens another door showing goat
    - You can switch or stay
    
    Question: Is switching better?
    """
    stay_wins = 0
    switch_wins = 0
    
    for _ in range(n_games):
        # Setup: randomly place prize behind one door (0, 1, or 2)
        prize_door = random.randint(0, 2)
        
        # Player picks a door randomly
        player_choice = random.randint(0, 2)
        
        # Host opens a door (not player's choice, not prize)
        available_doors = [d for d in [0, 1, 2] 
                          if d != player_choice and d != prize_door]
        host_opens = random.choice(available_doors)
        
        # Switch choice: remaining door
        switch_choice = [d for d in [0, 1, 2] 
                        if d != player_choice and d != host_opens][0]
        
        # Count wins for each strategy
        if player_choice == prize_door:
            stay_wins += 1
        if switch_choice == prize_door:
            switch_wins += 1
    
    return stay_wins / n_games, switch_wins / n_games

# Run simulation
n_games = 10000
stay_prob, switch_prob = monty_hall_simulation(n_games)

print(f"Simulated {n_games} games:")
print(f"Win probability if STAY:   {stay_prob:.2%}")    # ~33%
print(f"Win probability if SWITCH: {switch_prob:.2%}")  # ~67%
print(f"\nConclusion: Switching doubles your chances!")
```

**Intuition:** Initially 1/3 chance of correct pick. When host reveals goat, the 2/3 probability "transfers" to the other door.

---

63. Create a Python function to perform a t-test given two sample datasets.

**Answer:**

```python
import numpy as np
from scipy import stats

def perform_ttest(sample1, sample2, alpha=0.05, equal_var=True):
    """
    Perform independent two-sample t-test.
    
    H0: μ1 = μ2 (no difference)
    H1: μ1 ≠ μ2 (two-tailed)
    """
    # Calculate statistics
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
    
    # Perform t-test (Welch's if equal_var=False)
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
    
    # Decision
    significant = p_value <= alpha
    
    # Results
    print("=" * 50)
    print("Two-Sample T-Test Results")
    print("=" * 50)
    print(f"Sample 1: n={n1}, mean={mean1:.3f}, std={std1:.3f}")
    print(f"Sample 2: n={n2}, mean={mean2:.3f}, std={std2:.3f}")
    print(f"\nt-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Significance level: α = {alpha}")
    print(f"\nDecision: {'Reject H0' if significant else 'Fail to reject H0'}")
    print(f"Interpretation: {'Significant' if significant else 'No significant'} difference between groups")
    
    return t_stat, p_value, significant

# Example usage
group_A = [85, 90, 78, 92, 88, 76, 95, 89]
group_B = [75, 80, 72, 85, 78, 70, 82, 77]

t_stat, p_value, significant = perform_ttest(group_A, group_B)
```

---

64. Write a Python script to compute and graphically display a correlation matrix for a given dataset.

**Answer:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(data, method='pearson'):
    """
    Compute and visualize correlation matrix.
    
    Parameters:
    - data: DataFrame with numeric columns
    - method: 'pearson', 'spearman', or 'kendall'
    """
    # Step 1: Compute correlation matrix
    corr_matrix = data.corr(method=method)
    
    # Step 2: Create heatmap
    plt.figure(figsize=(10, 8))
    
    # Create mask for upper triangle (optional, for cleaner look)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Plot heatmap
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True,           # Show values
                fmt='.2f',            # Format to 2 decimals
                cmap='coolwarm',      # Color scheme
                center=0,             # Center colormap at 0
                square=True,
                linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    
    plt.title(f'{method.capitalize()} Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

# Example with sample data
np.random.seed(42)
data = pd.DataFrame({
    'height': np.random.normal(170, 10, 100),
    'weight': np.random.normal(70, 15, 100),
    'age': np.random.randint(20, 60, 100),
    'income': np.random.normal(50000, 15000, 100)
})

# Add correlated feature
data['BMI'] = data['weight'] / ((data['height']/100)**2)

# Plot correlation matrix
corr = plot_correlation_matrix(data)
print("\nCorrelation Matrix:\n", corr)
```

---

65. Implement the Metropolis-Hastings algorithm for a simple Bayesian inference simulation.

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt

def metropolis_hastings(target_pdf, proposal_std, n_samples, initial_value):
    """
    Metropolis-Hastings MCMC algorithm.
    
    Parameters:
    - target_pdf: Function returning probability density (unnormalized OK)
    - proposal_std: Standard deviation for proposal distribution
    - n_samples: Number of samples to generate
    - initial_value: Starting point
    
    Returns:
    - samples: Array of MCMC samples
    """
    samples = [initial_value]
    current = initial_value
    accepted = 0
    
    for _ in range(n_samples - 1):
        # Step 1: Propose new value (symmetric random walk)
        proposed = current + np.random.normal(0, proposal_std)
        
        # Step 2: Calculate acceptance probability
        # α = min(1, p(proposed) / p(current))
        acceptance_ratio = target_pdf(proposed) / target_pdf(current)
        acceptance_prob = min(1, acceptance_ratio)
        
        # Step 3: Accept or reject
        if np.random.random() < acceptance_prob:
            current = proposed
            accepted += 1
        
        samples.append(current)
    
    acceptance_rate = accepted / (n_samples - 1)
    return np.array(samples), acceptance_rate

# Example: Sample from Normal(3, 2) using MH
def target_distribution(x):
    """Target: Normal distribution with μ=3, σ=2"""
    mu, sigma = 3, 2
    return np.exp(-0.5 * ((x - mu) / sigma)**2)

# Run MCMC
n_samples = 10000
samples, acceptance_rate = metropolis_hastings(
    target_pdf=target_distribution,
    proposal_std=1.0,
    n_samples=n_samples,
    initial_value=0
)

# Discard burn-in
burn_in = 1000
samples = samples[burn_in:]

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Trace plot
axes[0].plot(samples[:500])
axes[0].set_title('Trace Plot (first 500 samples)')
axes[0].set_xlabel('Iteration')

# Histogram vs true distribution
axes[1].hist(samples, bins=50, density=True, alpha=0.7, label='MCMC samples')
x = np.linspace(-5, 11, 100)
axes[1].plot(x, (1/(2*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-3)/2)**2), 
             'r-', label='True N(3,2)')
axes[1].set_title('Posterior Estimate')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Acceptance rate: {acceptance_rate:.2%}")
print(f"Sample mean: {np.mean(samples):.3f} (true: 3)")
print(f"Sample std: {np.std(samples):.3f} (true: 2)")
```

---

66. Create a Python program that estimates Pi using a Monte Carlo simulation.

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(n_points):
    """
    Estimate π using Monte Carlo method.
    
    Logic:
    - Generate random points in unit square [0,1] x [0,1]
    - Count points inside quarter circle (x² + y² ≤ 1)
    - Ratio ≈ π/4, so π ≈ 4 × (points inside / total points)
    """
    # Step 1: Generate random points
    x = np.random.uniform(0, 1, n_points)
    y = np.random.uniform(0, 1, n_points)
    
    # Step 2: Check if inside quarter circle
    distance = x**2 + y**2
    inside = distance <= 1
    n_inside = np.sum(inside)
    
    # Step 3: Estimate π
    pi_estimate = 4 * n_inside / n_points
    
    return pi_estimate, x, y, inside

# Run simulation with increasing samples
sample_sizes = [100, 1000, 10000, 100000, 1000000]
print("Pi Estimation via Monte Carlo:")
print("-" * 40)
for n in sample_sizes:
    pi_est, _, _, _ = estimate_pi(n)
    error = abs(pi_est - np.pi)
    print(f"n = {n:>8}: π ≈ {pi_est:.6f}, error = {error:.6f}")

# Visualization with 5000 points
pi_est, x, y, inside = estimate_pi(5000)

plt.figure(figsize=(8, 8))
plt.scatter(x[inside], y[inside], c='blue', s=1, label='Inside circle')
plt.scatter(x[~inside], y[~inside], c='red', s=1, label='Outside circle')

# Draw quarter circle
theta = np.linspace(0, np.pi/2, 100)
plt.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal')
plt.title(f'Monte Carlo Pi Estimation: π ≈ {pi_est:.4f}')
plt.legend()
plt.show()
```

---

67. Write a Python code snippet for performing a Chi-squared test of independence on a contingency table.

**Answer:**

```python
import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd

def chi_squared_test(observed, row_labels=None, col_labels=None, alpha=0.05):
    """
    Perform Chi-squared test of independence.
    
    H0: Variables are independent
    H1: Variables are associated
    """
    # Convert to numpy array if needed
    observed = np.array(observed)
    
    # Perform chi-squared test
    chi2, p_value, dof, expected = chi2_contingency(observed)
    
    # Decision
    significant = p_value <= alpha
    
    # Display results
    print("=" * 50)
    print("Chi-Squared Test of Independence")
    print("=" * 50)
    
    # Show observed frequencies
    print("\nObserved Frequencies:")
    obs_df = pd.DataFrame(observed, index=row_labels, columns=col_labels)
    print(obs_df)
    
    # Show expected frequencies
    print("\nExpected Frequencies (under H0):")
    exp_df = pd.DataFrame(np.round(expected, 2), index=row_labels, columns=col_labels)
    print(exp_df)
    
    # Statistics
    print(f"\nChi-squared statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"p-value: {p_value:.4f}")
    print(f"\nAt α = {alpha}:")
    print(f"Decision: {'Reject H0' if significant else 'Fail to reject H0'}")
    print(f"Conclusion: Variables are {'NOT independent (associated)' if significant else 'independent'}")
    
    return chi2, p_value, dof, expected

# Example: Marketing A/B test
# Did version affect whether users clicked?
observed = [
    [50, 150],   # Version A: [Clicked, Not Clicked]
    [80, 120]    # Version B: [Clicked, Not Clicked]
]

chi2, p_value, dof, expected = chi_squared_test(
    observed,
    row_labels=['Version A', 'Version B'],
    col_labels=['Clicked', 'Not Clicked']
)
```

---

68. Develop a Python function to convert a non-stationary time series into a stationary one.

**Answer:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def make_stationary(series, max_diff=2, alpha=0.05):
    """
    Convert non-stationary time series to stationary using differencing.
    
    Parameters:
    - series: Time series data (pandas Series)
    - max_diff: Maximum differencing attempts
    - alpha: Significance level for ADF test
    
    Returns:
    - stationary_series: Transformed series
    - d: Number of differences applied
    """
    def check_stationarity(s):
        """Perform Augmented Dickey-Fuller test"""
        result = adfuller(s.dropna())
        return result[1] < alpha, result[1]  # (is_stationary, p_value)
    
    current_series = series.copy()
    d = 0
    
    print("Stationarity Check:")
    print("-" * 50)
    
    # Check initial stationarity
    is_stationary, p_val = check_stationarity(current_series)
    print(f"d={d}: ADF p-value = {p_val:.4f} → {'Stationary' if is_stationary else 'Non-stationary'}")
    
    # Apply differencing until stationary
    while not is_stationary and d < max_diff:
        d += 1
        current_series = current_series.diff().dropna()
        is_stationary, p_val = check_stationarity(current_series)
        print(f"d={d}: ADF p-value = {p_val:.4f} → {'Stationary' if is_stationary else 'Non-stationary'}")
    
    if is_stationary:
        print(f"\n✓ Series is stationary after {d} difference(s)")
    else:
        print(f"\n✗ Series still non-stationary after {max_diff} differences")
    
    return current_series, d

# Example: Create non-stationary series (random walk with trend)
np.random.seed(42)
n = 200
trend = np.linspace(0, 50, n)
random_walk = np.cumsum(np.random.randn(n))
non_stationary = pd.Series(trend + random_walk)

# Make stationary
stationary, d = make_stationary(non_stationary)

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

axes[0].plot(non_stationary)
axes[0].set_title('Original (Non-stationary) Series')

axes[1].plot(stationary)
axes[1].set_title(f'Stationary Series (d={d} differences)')

plt.tight_layout()
plt.show()
```

---

71. Compute unbiased sample variance and standard deviation using Bessel's correction.

**Answer:**

```python
import numpy as np

def sample_variance_std(x):
    """Return (s2, s) using Bessel's correction (n-1)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("x must contain at least two values")

    mean = np.mean(x)
    s2 = np.sum((x - mean) ** 2) / (n - 1)
    s = np.sqrt(s2)
    return float(s2), float(s)

# Example
data = [2, 4, 4, 4, 5, 5, 7, 9]
var, std = sample_variance_std(data)
print(var, std)
```

69. Write an R script to conduct an ANOVA test on a given dataset.

**Answer:**

```r
# ANOVA Test in R
# Comparing means across 3+ groups

# Step 1: Create sample data
set.seed(42)
group_A <- rnorm(30, mean=75, sd=10)
group_B <- rnorm(30, mean=80, sd=10)
group_C <- rnorm(30, mean=85, sd=10)

# Combine into data frame
data <- data.frame(
  score = c(group_A, group_B, group_C),
  group = factor(rep(c("A", "B", "C"), each=30))
)

# Step 2: Visualize data
boxplot(score ~ group, data=data,
        main="Score Distribution by Group",
        xlab="Group", ylab="Score",
        col=c("lightblue", "lightgreen", "lightyellow"))

# Step 3: Perform ANOVA
anova_result <- aov(score ~ group, data=data)
summary(anova_result)

# Step 4: Check assumptions
# Normality of residuals
shapiro.test(residuals(anova_result))

# Homogeneity of variance
bartlett.test(score ~ group, data=data)

# Step 5: Post-hoc test (if ANOVA is significant)
tukey_result <- TukeyHSD(anova_result)
print(tukey_result)
plot(tukey_result)
```

**Python Equivalent:**
```python
from scipy import stats
import pandas as pd

# Sample data
group_A = [75, 78, 80, 72, 77]
group_B = [80, 82, 85, 79, 83]
group_C = [85, 88, 90, 84, 87]

# One-way ANOVA
f_stat, p_value = stats.f_oneway(group_A, group_B, group_C)
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

---

70. Implement PCA for dimensionality reduction on a high-dimensional dataset in Python.

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

def perform_pca(X, n_components=None, variance_threshold=0.95):
    """
    Perform PCA for dimensionality reduction.
    
    Parameters:
    - X: Feature matrix
    - n_components: Number of components (if None, use variance_threshold)
    - variance_threshold: Keep components explaining this much variance
    
    Returns:
    - X_transformed: Reduced data
    - pca: Fitted PCA object
    """
    # Step 1: Standardize data (crucial for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 2: Fit PCA to find optimal components
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # Step 3: Determine number of components
    if n_components is None:
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= variance_threshold) + 1
    
    # Step 4: Fit PCA with selected components
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X_scaled)
    
    # Results
    print("=" * 50)
    print("PCA Results")
    print("=" * 50)
    print(f"Original dimensions: {X.shape[1]}")
    print(f"Reduced dimensions: {n_components}")
    print(f"Variance explained: {sum(pca.explained_variance_ratio_):.2%}")
    print("\nVariance by component:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.2%}")
    
    return X_transformed, pca

# Example: Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Perform PCA
X_pca, pca = perform_pca(X, n_components=2)

# Visualize reduced data
plt.figure(figsize=(10, 4))

# Subplot 1: Scree plot
plt.subplot(1, 2, 1)
pca_full = PCA().fit(StandardScaler().fit_transform(X))
plt.bar(range(1, len(pca_full.explained_variance_ratio_)+1), 
        pca_full.explained_variance_ratio_, alpha=0.7)
plt.plot(range(1, len(pca_full.explained_variance_ratio_)+1),
         np.cumsum(pca_full.explained_variance_ratio_), 'ro-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')

# Subplot 2: 2D projection
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA: 2D Projection')
plt.colorbar(scatter)

plt.tight_layout()
plt.show()
```

---
