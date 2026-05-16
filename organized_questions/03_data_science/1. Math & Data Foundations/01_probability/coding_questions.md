# Probability Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Write a Python function that calculates the probability of rolling a sum of 'S' on two dice.**

---

### 1. Approach
- Total outcomes = 6 × 6 = 36
- Count favorable outcomes where die1 + die2 = S
- P(sum = S) = favorable / total

### 2. Code (Writable)
```python
def probability_of_sum(S):
    if not 2 <= S <= 12:
        return 0.0
    
    total = 36  # 6 * 6
    favorable = 0
    
    for die1 in range(1, 7):
        for die2 in range(1, 7):
            if die1 + die2 == S:
                favorable += 1
    
    return favorable / total

# Test
print(f"P(sum=7) = {probability_of_sum(7):.4f}")   # 6/36 = 0.1667
print(f"P(sum=2) = {probability_of_sum(2):.4f}")   # 1/36 = 0.0278
```

### 3. Quick Reference (Byheart)
| Sum | Ways | Probability |
|-----|------|-------------|
| 2, 12 | 1 | 1/36 = 0.028 |
| 3, 11 | 2 | 2/36 = 0.056 |
| 7 | 6 | 6/36 = 0.167 |

---

## Question 2
- [ ] Done

**Implement a function that simulates a biased coin flip n times and estimates the probability of heads.**

---

### 1. Approach
- Simulate n flips using biased probability
- Count heads (represented as 1)
- Estimate P(heads) = count / n

### 2. Code (Writable)
```python
import numpy as np

def simulate_biased_coin(n_flips, prob_heads=0.7):
    # Simulate: 1=Heads, 0=Tails
    flips = np.random.choice([1, 0], size=n_flips, 
                              p=[prob_heads, 1-prob_heads])
    return np.mean(flips)  # = count / n

# Test - LLN: more flips → closer to true prob
true_p = 0.7
print(f"n=10:    {simulate_biased_coin(10, true_p):.3f}")
print(f"n=100:   {simulate_biased_coin(100, true_p):.3f}")
print(f"n=10000: {simulate_biased_coin(10000, true_p):.3f}")
```

### 3. Key Insight
Law of Large Numbers: As n increases, estimate → true probability.

---

## Question 3
- [ ] Done

**Code a Gaussian Naive Bayes classifier from scratch using Python.**

---

### 1. Approach
- **fit()**: Learn priors P(C) and Gaussian parameters (μ, σ²) per feature per class
- **predict()**: Use Bayes' theorem with "naive" independence assumption
- Use log probabilities to avoid underflow

### 2. Code (Writable)
```python
import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.priors[c] = len(X_c) / len(X)
    
    def _gaussian_pdf(self, x, mean, var):
        eps = 1e-9  # Avoid div by zero
        return np.exp(-((x - mean)**2) / (2*(var + eps))) / np.sqrt(2*np.pi*(var + eps))
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                log_prior = np.log(self.priors[c])
                log_likelihood = np.sum(np.log(self._gaussian_pdf(x, self.mean[c], self.var[c])))
                posteriors[c] = log_prior + log_likelihood
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)

# Usage
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=2, random_state=42)
gnb = GaussianNaiveBayes()
gnb.fit(X, y)
print(f"Accuracy: {np.mean(gnb.predict(X) == y):.2f}")
```

### 3. Key Points
- Use **log probabilities** to avoid numerical underflow
- **Naive assumption**: P(x₁,x₂|C) = P(x₁|C) × P(x₂|C)

---

## Question 4
- [ ] Done

**Simulate the Law of Large Numbers using Python: verify that as the number of coin tosses increases, the average of the results becomes closer to the expected value.**

---

### 1. Approach
- Simulate many coin flips (H=1, T=0)
- Calculate running average after each flip
- Visualize convergence to E[X] = 0.5

### 2. Code (Writable)
```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_lln(n_trials):
    expected = 0.5  # Fair coin: H=1, T=0
    
    # Simulate coin flips
    flips = np.random.randint(0, 2, size=n_trials)
    
    # Running average: cumsum / [1,2,3,...]
    running_avg = np.cumsum(flips) / np.arange(1, n_trials + 1)
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(running_avg, label='Sample Average')
    plt.axhline(y=expected, color='r', linestyle='--', label='E[X]=0.5')
    plt.xlabel('Number of Flips')
    plt.ylabel('Running Average')
    plt.title('LLN: Convergence to Expected Value')
    plt.xscale('log')
    plt.legend()
    plt.show()
    
    print(f"Final average: {running_avg[-1]:.4f}")

simulate_lln(100000)
```

### 3. Key Observation
- Initially: Running average fluctuates wildly
- As n → ∞: Average stabilizes at 0.5
- This IS the Law of Large Numbers

---

## Question 5
- [ ] Done

**Write a Python script that estimates the mean and variance of a dataset and plots the corresponding Gaussian distribution.**

---

### 1. Approach
- Estimate μ (mean) and σ² (variance) from data
- Plot histogram (normalized)
- Overlay fitted Gaussian PDF

### 2. Code (Writable)
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def analyze_and_plot(data):
    # Estimate parameters (MLE)
    mu = np.mean(data)
    std = np.std(data)
    
    print(f"Mean: {mu:.2f}, Std: {std:.2f}")
    
    # Histogram (density=True normalizes)
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.6, label='Data')
    
    # Fitted Gaussian
    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, norm.pdf(x, mu, std), 'k-', lw=2, label='Fitted Gaussian')
    
    plt.title(f'Gaussian Fit: μ={mu:.2f}, σ={std:.2f}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Test
data = np.random.normal(loc=50, scale=10, size=1000)
analyze_and_plot(data)
```

### 3. Key Points
- `density=True` normalizes histogram (area=1) for PDF comparison
- `norm.pdf(x, μ, σ)` gives Gaussian density
- Good fit: Gaussian curve follows histogram shape

---

