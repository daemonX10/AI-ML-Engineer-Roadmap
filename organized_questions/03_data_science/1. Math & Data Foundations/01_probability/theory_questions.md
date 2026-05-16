# Probability Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What is probability, and how is it used in machine learning?**

---

### 1. Definition
Probability quantifies the likelihood of an event occurring, expressed as a number between 0 (impossible) and 1 (certain). In ML, it provides a framework for modeling uncertainty, designing algorithms (Naive Bayes, Logistic Regression), deriving loss functions (Cross-Entropy from MLE), and evaluating model confidence.

### 2. Core Concepts
- **Sample Space (S)**: Set of all possible outcomes
- **Event (E)**: Subset of sample space we're interested in
- **P(E) = Favorable outcomes / Total outcomes**
- **Random Variable**: Maps outcomes to numerical values

### 3. Mathematical Formulation
$$P(E) = \frac{|E|}{|S|} = \frac{\text{Number of favorable outcomes}}{\text{Total possible outcomes}}$$

**Axioms of Probability:**
- $0 \leq P(E) \leq 1$
- $P(S) = 1$
- $P(A \cup B) = P(A) + P(B)$ if A and B are mutually exclusive

### 4. Intuition
Think of probability as a measure of belief or confidence. When a model outputs P(spam) = 0.95, it's saying "I'm 95% confident this is spam" rather than a hard yes/no answer.

### 5. Practical ML Applications
| Application | How Probability is Used |
|-------------|------------------------|
| Classification | Output P(class\|features), not just class label |
| Loss Functions | Cross-Entropy derived from MLE |
| Generative Models | GANs/VAEs learn data distribution p(x) |
| Uncertainty | Bayesian methods quantify model confidence |

### 6. Python Example
```python
# Spam classifier with probability output
def classify_email(features, model):
    prob_spam = model.predict_proba(features)[0, 1]
    
    # Probabilistic output allows flexible thresholding
    if prob_spam > 0.9:
        return "Spam (high confidence)"
    elif prob_spam > 0.5:
        return "Spam (low confidence)"
    else:
        return "Not spam"
```

### 7. Interview Tips
- **Probability vs Likelihood**: Probability = future events; Likelihood = how well parameters explain observed data
- **Calibration**: Model predicting 70% should be correct 70% of the time

---

## Question 2
- [ ] Done

**What is the difference between discrete and continuous probability distributions?**

---

### 1. Definition
**Discrete distributions** describe countable outcomes (coin flips, dice rolls) using a Probability Mass Function (PMF). **Continuous distributions** describe uncountable outcomes in a continuous range (height, temperature) using a Probability Density Function (PDF).

### 2. Core Concepts
| Aspect | Discrete | Continuous |
|--------|----------|------------|
| **Function** | PMF: P(X = k) | PDF: f(x) |
| **Point Probability** | P(X = k) is valid | P(X = x) = 0 always |
| **Range Probability** | Sum: Σ P(X = k) | Integral: ∫ f(x) dx |
| **Total** | Σ P(X = k) = 1 | ∫ f(x) dx = 1 |
| **Examples** | Binomial, Poisson | Normal, Exponential |

### 3. Mathematical Formulation
**Discrete (PMF):**
$$P(a \leq X \leq b) = \sum_{k=a}^{b} P(X = k)$$

**Continuous (PDF):**
$$P(a \leq X \leq b) = \int_a^b f(x)dx$$

### 4. Intuition
- **Discrete**: Like counting items - "How many heads in 10 flips?"
- **Continuous**: Like measuring - "What is the exact height?" (infinite precision impossible)
- PDF value is density, NOT probability. Probability = area under curve.

### 5. Practical ML Applications
- **Discrete**: Classification labels, count data, click/no-click
- **Continuous**: Regression targets, feature values, sensor measurements

### 6. Python Example
```python
from scipy.stats import binom, norm

# Discrete: Binomial - P(X = 5) directly
pmf_value = binom.pmf(5, n=10, p=0.5)  # = 0.246

# Continuous: Normal - P(X = 170) = 0, use range instead
# P(160 < X < 180) requires integration via CDF
prob = norm.cdf(180, 170, 10) - norm.cdf(160, 170, 10)  # ≈ 0.68
```

### 7. Common Pitfalls
- **Never interpret PDF value as probability** - density can exceed 1!
- For continuous: always calculate P(range), never P(point)

---

## Question 3
- [ ] Done

**Explain the differences between joint, marginal, and conditional probabilities.**

---

### 1. Definition
**Joint P(A,B)** = probability both A and B occur together. **Marginal P(A)** = probability of A regardless of B. **Conditional P(A|B)** = probability of A given B has occurred. These describe relationships between multiple events.

### 2. Core Concepts
| Type | Notation | Meaning |
|------|----------|---------|
| Joint | P(A, B) | Both A AND B occur |
| Marginal | P(A) | A occurs, regardless of B |
| Conditional | P(A\|B) | A occurs, GIVEN B happened |

### 3. Mathematical Formulation
**Relationships:**
- **Joint**: $P(A, B) = P(A|B) \times P(B) = P(B|A) \times P(A)$
- **Marginal**: $P(A) = P(A, B) + P(A, \neg B) = \sum_B P(A, B)$
- **Conditional**: $P(A|B) = \frac{P(A, B)}{P(B)}$

**Independence**: If A, B independent, then $P(A,B) = P(A) \times P(B)$

### 4. Intuition (Weather Example)
Let A = "Raining", B = "Heavy Traffic"
- **P(A, B)** = Prob of rain AND heavy traffic together
- **P(A)** = Overall prob of rain (ignoring traffic)
- **P(A|B)** = Prob of rain knowing traffic is heavy (likely higher than P(A))

### 5. Practical ML Applications
- **Joint**: Risk assessment - P(multiple failures)
- **Marginal**: Feature prevalence - P(spam) in dataset
- **Conditional**: Classification goal - P(Class | Features)

### 6. Python Example
```python
# Joint probability table (frequencies)
# P(Rain, Traffic) from data
joint_table = {
    ('rain', 'heavy'): 0.10,
    ('rain', 'light'): 0.05,
    ('sunny', 'heavy'): 0.15,
    ('sunny', 'light'): 0.70
}

# Marginal: P(Rain) = sum over traffic conditions
p_rain = joint_table[('rain','heavy')] + joint_table[('rain','light')]  # 0.15

# Conditional: P(Rain | Heavy Traffic)
p_heavy = joint_table[('rain','heavy')] + joint_table[('sunny','heavy')]  # 0.25
p_rain_given_heavy = joint_table[('rain','heavy')] / p_heavy  # 0.10/0.25 = 0.40
```

### 7. Common Pitfalls
- **Assuming independence**: Don't use P(A,B) = P(A)×P(B) unless verified
- **Prosecutor's fallacy**: P(A|B) ≠ P(B|A) - use Bayes' theorem to convert

---

## Question 4
- [ ] Done

**Describe Bayes' Theorem and provide an example of how it's used.**

---

### 1. Definition
Bayes' Theorem describes how to update the probability of a hypothesis based on new evidence. It connects prior beliefs, observed evidence, and updated beliefs through a mathematical formula: Posterior ∝ Likelihood × Prior.

### 2. Core Concepts
| Term | Name | Meaning |
|------|------|---------|
| P(H\|E) | **Posterior** | Updated belief after seeing evidence |
| P(E\|H) | **Likelihood** | Probability of evidence if hypothesis true |
| P(H) | **Prior** | Initial belief before evidence |
| P(E) | **Evidence** | Total probability of evidence (normalizer) |

### 3. Mathematical Formulation
$$P(H|E) = \frac{P(E|H) \times P(H)}{P(E)}$$

Where: $P(E) = P(E|H) \times P(H) + P(E|\neg H) \times P(\neg H)$

**Simplified**: Posterior ∝ Likelihood × Prior

### 4. Intuition
Bayes' theorem is about updating beliefs. Start with initial belief (prior), observe evidence, update to new belief (posterior). Strong prior + weak evidence = small update. Weak prior + strong evidence = big update.

### 5. Practical Example: Medical Diagnosis
- Disease prevalence: P(D) = 0.001 (1 in 1000)
- Test sensitivity: P(+|D) = 0.99
- False positive rate: P(+|¬D) = 0.01

**Question**: If test positive, what's P(Disease)?

### 6. Python Example
```python
def bayes_theorem(p_h, p_e_given_h, p_e_given_not_h):
    """Calculate posterior probability P(H|E)."""
    p_not_h = 1 - p_h
    p_e = (p_e_given_h * p_h) + (p_e_given_not_h * p_not_h)
    return (p_e_given_h * p_h) / p_e

# Medical test example
p_disease_given_positive = bayes_theorem(0.001, 0.99, 0.01)
print(f"P(Disease | Positive Test) = {p_disease_given_positive:.3f}")
# Output: 0.090 (only 9%!)
```

### 7. Interview Tips
- Counter-intuitive result: 99% accurate test, but only 9% chance of disease if positive. Why? Base rate (0.1%) is very low.
- **ML Applications**: Naive Bayes, Bayesian inference, A/B testing

### 8. Algorithm Steps
```
1. Define prior P(H)
2. Define likelihood P(E|H)
3. Calculate evidence P(E) = sum over all H
4. Apply formula: Posterior = (Likelihood × Prior) / Evidence
```

---

## Question 5
- [ ] Done

**What is a probability density function (PDF)?**

---

### 1. Definition
A PDF, denoted f(x), describes the probability distribution of a continuous random variable. It represents relative likelihood at each point - NOT the probability itself. Probability is obtained by integrating the PDF over a range.

### 2. Core Concepts
- PDF value at point x = density, NOT probability
- P(X = x) = 0 for any specific point (continuous)
- Probability = area under curve = integral
- PDF can exceed 1 (it's density, not probability)

### 3. Mathematical Formulation
**Properties:**
1. Non-negativity: $f(x) \geq 0$ for all x
2. Total area = 1: $\int_{-\infty}^{\infty} f(x)dx = 1$

**Probability of range:**
$$P(a \leq X \leq b) = \int_a^b f(x)dx$$

**Normal distribution PDF:**
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$$

### 4. Intuition
Think of PDF as a "density of probability" - like population density. High density = values more likely in that region. But just like you can't have a fraction of a person at a single point, you can't have probability at a single point for continuous variables.

### 5. Practical ML Applications
- Gaussian Naive Bayes: models P(feature|class) as Gaussian PDF
- Likelihood in MLE: product of PDF values
- Anomaly detection: low density = anomaly

### 6. Python Example
```python
from scipy.stats import norm

mu, sigma = 0, 1  # Standard normal

# PDF value at x=0 (NOT probability!)
density = norm.pdf(0, mu, sigma)  # = 0.3989

# For probability, integrate via CDF
prob = norm.cdf(1) - norm.cdf(-1)  # P(-1 ≤ X ≤ 1) ≈ 0.68

print(f"Density at 0: {density:.4f}")
print(f"P(-1 ≤ X ≤ 1): {prob:.4f}")
```

### 7. Common Pitfalls
- **PDF ≠ Probability**: Never say "probability is 0.4" when you mean density
- PDF > 1 is valid for narrow distributions (e.g., Uniform(0, 0.5) has PDF = 2)

---

## Question 6
- [ ] Done

**What is the role of the cumulative distribution function (CDF)?**

---

### 1. Definition
The CDF, denoted F(x), gives the probability that random variable X takes a value less than or equal to x: F(x) = P(X ≤ x). It provides a unified way to describe any distribution and calculate range probabilities.

### 2. Core Concepts
- F(x) = P(X ≤ x) — cumulative probability up to x
- Works for both discrete and continuous distributions
- Range probability: P(a < X ≤ b) = F(b) - F(a)
- Inverse CDF (quantile function) finds percentiles

### 3. Mathematical Formulation
**Properties:**
1. Range: $0 \leq F(x) \leq 1$
2. Non-decreasing: if $a < b$, then $F(a) \leq F(b)$
3. Limits: $F(-\infty) = 0$, $F(+\infty) = 1$

**Relationship with PDF:**
$$F(x) = \int_{-\infty}^{x} f(t)dt \quad \text{and} \quad f(x) = \frac{dF(x)}{dx}$$

### 4. Intuition
CDF is the "running total" of probability. At any point x, F(x) tells you what fraction of the probability mass lies to the left of x. It's like asking "what percentage of people are shorter than x?"

### 5. Practical ML Applications
- **Probability calculations**: P(a < X ≤ b) = F(b) - F(a)
- **Percentiles**: Find x where F(x) = 0.95 (95th percentile)
- **Statistical tests**: p-values computed via CDF

### 6. Python Example
```python
from scipy.stats import norm

mu, sigma = 0, 1

# CDF: P(X ≤ 1.5)
prob_less_than = norm.cdf(1.5, mu, sigma)  # ≈ 0.93

# Range probability: P(-1 < X ≤ 1)
prob_range = norm.cdf(1) - norm.cdf(-1)  # ≈ 0.68

# Inverse CDF (percentile): Find x where P(X ≤ x) = 0.95
x_95th = norm.ppf(0.95, mu, sigma)  # ≈ 1.645

print(f"P(X ≤ 1.5) = {prob_less_than:.3f}")
print(f"95th percentile = {x_95th:.3f}")
```

### 7. Interview Tips
- If CDF is not monotonically increasing, there's a bug
- CDF approach: F(b) - F(a) is cleaner than integrating PDF

---

## Question 7
- [ ] Done

**Explain the Central Limit Theorem and its significance in machine learning.**

---

### 1. Definition
The Central Limit Theorem (CLT) states that the distribution of sample means approaches a normal distribution as sample size increases, regardless of the original population's shape. This is fundamental to statistical inference.

### 2. Core Concepts
**Conditions:**
- Independent, identically distributed (i.i.d.) samples
- Sufficiently large sample size (n > 30 rule of thumb)
- Population has finite mean (μ) and variance (σ²)

**Result:**
- Sample mean distribution → Normal
- Mean of sample means = μ
- Standard Error = σ/√n

### 3. Mathematical Formulation
If $X_1, X_2, ..., X_n$ are i.i.d. with mean μ and variance σ²:
$$\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right) \text{ as } n \to \infty$$

**Standardized form:**
$$\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0, 1)$$

### 4. Intuition
No matter how weird the original data distribution (skewed, bimodal, etc.), if you take many samples and average each one, those averages will form a bell curve. This is why the normal distribution appears everywhere!

### 5. Practical ML Applications
| Application | How CLT Helps |
|-------------|---------------|
| **Normality assumptions** | Justifies Gaussian noise in regression |
| **A/B testing** | Sample mean differences are normal → t-tests work |
| **Confidence intervals** | Sample mean ± z × SE |
| **Bootstrap** | Resampling relies on CLT principles |

### 6. Python Example
```python
import numpy as np

# Non-normal population (Exponential distribution)
population = np.random.exponential(scale=2.0, size=100000)

# Take many samples and compute means
sample_means = [np.mean(np.random.choice(population, 50)) for _ in range(2000)]

# sample_means will be normally distributed despite exponential population!
print(f"Population mean: {population.mean():.2f}")
print(f"Mean of sample means: {np.mean(sample_means):.2f}")
# Histogram of sample_means will be bell-shaped
```

### 7. Interview Tips
- CLT applies to sample MEANS, not individual samples
- Original distribution can be any shape
- Larger n → narrower distribution of means (σ/√n shrinks)

---

## Question 8
- [ ] Done

**What is the Law of Large Numbers?**

---

### 1. Definition
The Law of Large Numbers (LLN) states that as the number of trials increases, the sample mean converges to the true population mean (expected value). It's the philosophical foundation for why ML works on finite data.

### 2. Core Concepts
**Two Forms:**
- **Weak LLN**: Sample mean converges in probability to E[X]
- **Strong LLN**: Sample mean converges almost surely (P = 1)

**Key Insight:** With enough data, sample statistics approximate population parameters.

### 3. Mathematical Formulation
For i.i.d. random variables $X_1, X_2, ..., X_n$ with mean μ:
$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n}X_i \xrightarrow{n \to \infty} \mu$$

**Interpretation:** As n → ∞, sample average → true average

### 4. Intuition
Flip a fair coin: first 10 flips might give 70% heads (unlucky). But after 10,000 flips, you'll get very close to 50%. Random fluctuations average out over many trials.

### 5. Practical ML Applications
| Application | How LLN is Used |
|-------------|-----------------|
| **Empirical Risk Minimization** | Training loss ≈ true expected loss |
| **Monte Carlo Methods** | Random samples approximate expectations |
| **Parameter Estimation** | Sample statistics → population parameters |
| **Cross-validation** | Average performance over folds |

### 6. Python Example
```python
import numpy as np

# Fair die: E[X] = 3.5
expected_value = 3.5
n_rolls = 10000
rolls = np.random.randint(1, 7, size=n_rolls)

# Running average converges to 3.5
running_avg = np.cumsum(rolls) / np.arange(1, n_rolls + 1)
print(f"After 10 rolls: {running_avg[9]:.2f}")
print(f"After 100 rolls: {running_avg[99]:.2f}")
print(f"After 10000 rolls: {running_avg[-1]:.4f}")  # ≈ 3.5
```

### 7. LLN vs CLT

| LLN | CLT |
|-----|-----|
| Sample mean → true mean | Sample means → Normal distribution |
| About convergence of VALUE | About SHAPE of distribution |
| Justifies learning from data | Justifies statistical inference |

### 8. Interview Tip
LLN is WHY ML works — learning from finite training data generalizes because sample risk approximates true risk.

---

## Question 9
- [ ] Done

**What are the characteristics of a Gaussian (Normal) distribution?**

---

### 1. Definition
The Gaussian/Normal distribution is a continuous probability distribution defined by mean (μ) and standard deviation (σ). It forms the famous "bell curve" and is the most important distribution in statistics due to CLT.

### 2. Core Concepts
- **Parameters**: μ (center), σ (spread), variance = σ²
- **Symmetric** around mean
- **Mean = Median = Mode**
- **Tails extend to ±∞** but decay rapidly
- **Completely specified** by just μ and σ

### 3. Mathematical Formulation
**PDF:**
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$$

**Standard Normal**: μ = 0, σ = 1
**Z-score**: $z = \frac{x - \mu}{\sigma}$ (standardization)

### 4. Empirical Rule (68-95-99.7)
| Range | Percentage of Data |
|-------|-------------------|
| μ ± 1σ | ~68% |
| μ ± 2σ | ~95% |
| μ ± 3σ | ~99.7% |

### 5. Intuition
Most natural measurements cluster around an average with symmetric spread. Heights, test scores, measurement errors — all tend to be normally distributed. Values far from the mean are increasingly rare.

### 6. Practical ML Applications
- **Linear Regression**: Assumes residuals ~ N(0, σ²)
- **Gaussian Naive Bayes**: Features ~ N(μ, σ²) per class
- **Statistical Tests**: t-tests, ANOVA assume normality
- **Regularization**: L2 penalty = Gaussian prior on weights

### 7. Python Example
```python
from scipy.stats import norm

mu, sigma = 100, 15  # e.g., IQ scores

# 68-95-99.7 rule verification
prob_1std = norm.cdf(mu+sigma, mu, sigma) - norm.cdf(mu-sigma, mu, sigma)
prob_2std = norm.cdf(mu+2*sigma, mu, sigma) - norm.cdf(mu-2*sigma, mu, sigma)

print(f"Within 1σ: {prob_1std:.2%}")  # ≈ 68%
print(f"Within 2σ: {prob_2std:.2%}")  # ≈ 95%

# Z-score
x = 130
z = (x - mu) / sigma  # z = 2.0 → 2 std above mean
```

### 8. Interview Tip
If asked "why is Normal distribution so common?" → CLT: sums of many independent factors tend toward Normal.

---

## Question 10
- [ ] Done

**Explain the utility of the Binomial distribution in machine learning.**

---

### 1. Definition
The Binomial distribution models the number of successes (k) in n independent trials, each with success probability p. It's the foundation for modeling binary outcomes in ML: clicks, conversions, correct predictions.

### 2. Core Concepts
**Conditions for Binomial:**
1. Fixed number of trials (n)
2. Each trial independent
3. Two outcomes only (success/failure)
4. Constant probability p per trial

### 3. Mathematical Formulation
**PMF:**
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

**Parameters:**
- Mean: $\mu = np$
- Variance: $\sigma^2 = np(1-p)$

### 4. Intuition
"If I flip a biased coin n times, how many heads will I get?" The Binomial gives the full probability distribution over all possible counts from 0 to n.

### 5. Practical ML Applications
| Application | Description |
|-------------|-------------|
| **CTR Modeling** | n impressions, p click rate → count of clicks |
| **A/B Testing** | Compare conversion rates between variants |
| **Classification Evaluation** | Model accuracy over n test samples |
| **Logistic Regression** | Cross-entropy loss from Bernoulli likelihood |

### 6. Python Example
```python
from scipy.stats import binom

# A/B Test: 1000 visitors, 10% conversion rate
n, p = 1000, 0.10

# P(exactly 100 conversions)
prob_100 = binom.pmf(100, n, p)

# P(at least 120 conversions) - unusual result?
prob_120_plus = 1 - binom.cdf(119, n, p)

# Expected value
expected = n * p  # = 100

print(f"Expected conversions: {expected}")
print(f"P(X ≥ 120): {prob_120_plus:.4f}")  # For significance testing
```

### 7. Related Distributions
- **Bernoulli**: Binomial with n = 1 (single trial)
- **Poisson**: Limit of Binomial when n→∞, p→0, np=λ

### 8. Interview Tip
Use Binomial when you can count BOTH successes AND failures (denominator n is known). If you only observe events (no clear n), consider Poisson instead.

---

## Question 11
- [ ] Done

**How does the Poisson distribution differ from the Binomial distribution?**

---

### 1. Definition
**Binomial** models successes in fixed n trials. **Poisson** models events in a fixed interval (time/space) with no upper bound. Poisson is used when counting events but "non-events" can't be counted.

### 2. Core Concepts

| Feature | Binomial | Poisson |
|---------|----------|---------|
| **Models** | k successes in n trials | k events in interval |
| **Trials** | Finite, fixed (n) | Infinite/uncountable |
| **Parameters** | n, p | λ (rate) |
| **Outcomes** | 0 to n | 0 to ∞ |
| **Mean** | np | λ |
| **Variance** | np(1-p) | λ |

### 3. Mathematical Formulation
**Poisson PMF:**
$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

**Poisson as Binomial limit:**
When $n \to \infty$, $p \to 0$, and $np = \lambda$:
$$\text{Binomial}(n, p) \to \text{Poisson}(\lambda)$$

### 4. Intuition
- **Binomial**: "10 clicks out of 1000 impressions" — you know both counts
- **Poisson**: "5 visitors per minute" — what's a "non-visitor"? Can't count!

### 5. When to Use Each
| Use Binomial | Use Poisson |
|--------------|-------------|
| Fixed trials, countable failures | Events in continuous interval |
| "k out of n" | "k per unit time/space" |
| Click-through rate | Website visits per hour |
| Survey responses | Customer arrivals |

### 6. Python Example
```python
from scipy.stats import binom, poisson

# Rare event approximation: n=2000, p=0.002
n, p = 2000, 0.002
lambda_val = n * p  # = 4

# Both give similar results
binom_prob = binom.pmf(5, n, p)      # Exact
poisson_prob = poisson.pmf(5, lambda_val)  # Approximation

print(f"Binomial P(X=5): {binom_prob:.4f}")
print(f"Poisson P(X=5):  {poisson_prob:.4f}")
# Both ≈ 0.156
```

### 7. Interview Tip
Poisson is simpler (1 parameter vs 2) for rare events. If n is huge and p is tiny, use Poisson with λ = np.

---

## Question 12
- [ ] Done

**What is the relevance of the Bernoulli distribution in machine learning?**

---

### 1. Definition
The Bernoulli distribution describes a single trial with two outcomes: success (1) with probability p, failure (0) with probability (1-p). It's the atomic building block for binary classification in ML.

### 2. Core Concepts
- Single trial, two outcomes
- P(X = 1) = p, P(X = 0) = 1 - p
- Mean: E[X] = p
- Variance: Var(X) = p(1-p)
- Bernoulli = Binomial with n = 1

### 3. Mathematical Formulation
**PMF:**
$$P(X = x) = p^x (1-p)^{1-x} \quad \text{for } x \in \{0, 1\}$$

### 4. Intuition
One coin flip, one experiment, one yes/no question. Did the user click? Did the patient have the disease? Did the email get opened?

### 5. Practical ML Applications

| Application | How Bernoulli is Used |
|-------------|----------------------|
| **Binary Classification** | Target y ∈ {0, 1} |
| **Logistic Regression** | Models p = P(y=1\|x) via sigmoid |
| **Bernoulli Naive Bayes** | Binary feature likelihoods |
| **Cross-Entropy Loss** | Derived from Bernoulli log-likelihood |

### 6. Connection to Logistic Regression
Logistic regression models the Bernoulli parameter p:
$$p = \sigma(w^T x) = \frac{1}{1 + e^{-w^T x}}$$

**Log-likelihood (for one sample):**
$$\log P(y|x) = y \log(p) + (1-y) \log(1-p)$$

This is exactly the negative of binary cross-entropy loss!

### 7. Python Example
```python
from scipy.stats import bernoulli
import numpy as np

# Model predicts 75% probability of positive class
p = 0.75
dist = bernoulli(p)

# PMF values
print(f"P(X=1) = {dist.pmf(1)}")  # 0.75
print(f"P(X=0) = {dist.pmf(0)}")  # 0.25

# Simulate many predictions
samples = dist.rvs(size=1000)
print(f"Observed rate: {samples.mean():.3f}")  # ≈ 0.75 (LLN)
```

### 8. Interview Tip
Cross-entropy loss = negative Bernoulli log-likelihood. This connection explains why we use cross-entropy for classification.

---

## Question 13
- [ ] Done

**In machine learning, what are Naive Bayes classifiers, and why are they 'naive'?**

---

### 1. Definition
Naive Bayes is a probabilistic classifier based on Bayes' theorem that predicts class probabilities. It's "naive" because it assumes all features are conditionally independent given the class — a simplification that's rarely true but works surprisingly well.

### 2. Core Concepts
**Goal:** Find class C that maximizes P(C | Features)

**Bayes' Theorem:**
$$P(C | F_1, ..., F_n) \propto P(C) \times P(F_1, ..., F_n | C)$$

**The "Naive" Assumption:**
$$P(F_1, ..., F_n | C) = P(F_1|C) \times P(F_2|C) \times ... \times P(F_n|C)$$

### 3. Mathematical Formulation
**Decision Rule:**
$$\hat{C} = \arg\max_C \left[ P(C) \times \prod_{i=1}^{n} P(F_i|C) \right]$$

**In log space (for numerical stability):**
$$\hat{C} = \arg\max_C \left[ \log P(C) + \sum_{i=1}^{n} \log P(F_i|C) \right]$$

### 4. Intuition
Instead of learning complex feature interactions, Naive Bayes just learns: "How often does each feature appear in each class?" Then it multiplies these simple statistics together.

### 5. Why "Naive" Still Works
1. Only needs to **rank** classes correctly, not get exact probabilities
2. Extremely **fast** to train and predict
3. Requires **little data** to estimate parameters
4. Features being correlated doesn't necessarily hurt the ranking

### 6. Variants
| Variant | Feature Type | Likelihood Model |
|---------|--------------|------------------|
| Gaussian NB | Continuous | Normal distribution |
| Multinomial NB | Counts | Multinomial distribution |
| Bernoulli NB | Binary | Bernoulli distribution |

### 7. Python Example
```python
# Conceptual spam classifier
p_spam = 0.3
p_word_free_given_spam = 0.15
p_word_money_given_spam = 0.20

# "Naive" multiplication (assumes words independent)
score_spam = p_spam * p_word_free_given_spam * p_word_money_given_spam

# In reality: "free" and "money" are correlated
# Naive Bayes treats them as 2x independent evidence
```

### 8. Common Pitfalls
- **Overconfident predictions** when features are correlated
- **Zero probability problem**: One unseen word → entire probability = 0. Fix: Laplace smoothing.

---

## Question 14
- [ ] Done

**How does logistic regression utilize probability?**

---

### 1. Definition
Logistic Regression is a classification algorithm that directly models the probability of class membership P(y=1|x) using the sigmoid function. Probability is central: it's modeled, it's the output, and it defines the training objective.

### 2. Core Concepts
- Output is probability between 0 and 1
- Uses sigmoid function to squash linear combination
- Trained via Maximum Likelihood Estimation (MLE)
- Decision threshold (typically 0.5) converts probability to class

### 3. Mathematical Formulation
**Log-odds (logit):**
$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X_1 + ... + \beta_n X_n$$

**Probability (sigmoid):**
$$p = P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + ... + \beta_n X_n)}}$$

**Loss Function (Cross-Entropy from MLE):**
$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right]$$

### 4. Intuition
Linear regression gives unbounded output. Sigmoid "squashes" it to (0, 1). The output is interpreted as "model's confidence" that the sample belongs to class 1.

### 5. Practical ML Applications
- Binary classification with interpretable coefficients
- Probability calibration (well-calibrated by design)
- Adjustable decision thresholds for cost-sensitive decisions
- Feature importance from coefficient magnitudes

### 6. Python Example
```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Trained model coefficients
intercept = -3.0
beta_credit_score = -0.01
beta_income = 0.05

# Predict for applicant
credit_score, income_k = 600, 50
z = intercept + beta_credit_score * credit_score + beta_income * income_k
prob_default = sigmoid(z)

print(f"P(default) = {prob_default:.2f}")
prediction = "Default" if prob_default > 0.5 else "No Default"
```

### 7. Interview Tips
- Logistic regression assumes **linear decision boundary** in feature space
- Sigmoid ensures output is valid probability
- Cross-entropy loss = negative log-likelihood of Bernoulli model

---

## Question 15
- [ ] Done

**What is the concept of entropy in information theory, and how does it relate to machine learning models?**

---

### 1. Definition
Entropy measures the uncertainty or "surprise" in a probability distribution. High entropy = uniform, unpredictable. Low entropy = skewed, predictable. It's used in decision trees (information gain) and as a loss function (cross-entropy).

### 2. Core Concepts
- Quantifies average "information" needed to identify an outcome
- Maximum entropy for uniform distribution
- Zero entropy when outcome is certain
- Unit: bits (when using log₂)

### 3. Mathematical Formulation
**Shannon Entropy:**
$$H(X) = -\sum_{i} P(x_i) \log_2 P(x_i)$$

**Cross-Entropy (between true p and predicted q):**
$$H(p, q) = -\sum_{x} p(x) \log q(x)$$

### 4. Intuition
| Distribution | Entropy | Interpretation |
|--------------|---------|----------------|
| Fair coin [0.5, 0.5] | 1.0 bit | Need 1 bit to encode outcome |
| Biased [0.9, 0.1] | 0.47 bits | More predictable |
| Certain [1.0, 0.0] | 0 bits | No uncertainty |
| Fair die [1/6 × 6] | 2.58 bits | More outcomes = more uncertainty |

### 5. Practical ML Applications

**Decision Trees (Information Gain):**
$$\text{Info Gain} = H(\text{parent}) - \sum \frac{n_{\text{child}}}{n} H(\text{child})$$
Split on feature that maximizes information gain (reduces entropy most).

**Cross-Entropy Loss:**
Used to train classifiers. Minimizing cross-entropy pushes predicted distribution toward true distribution.

### 6. Python Example
```python
import numpy as np

def entropy(probs):
    """Shannon entropy in bits."""
    probs = np.array([p for p in probs if p > 0])  # Avoid log(0)
    return -np.sum(probs * np.log2(probs))

print(f"Fair coin: {entropy([0.5, 0.5]):.2f} bits")      # 1.0
print(f"Biased coin: {entropy([0.9, 0.1]):.2f} bits")    # 0.47
print(f"Certain: {entropy([1.0, 0.0]):.2f} bits")        # 0.0
print(f"Fair die: {entropy([1/6]*6):.2f} bits")          # 2.58
```

### 7. Interview Tips
- **Cross-entropy ≥ entropy**: Cross-entropy equals entropy only when q = p
- **KL divergence** = Cross-entropy − Entropy = how different q is from p

---

## Question 16
- [ ] Done

**Explain the relationship between Maximum Likelihood Estimation (MLE) and probability.**

---

### 1. Definition
MLE estimates model parameters by finding values that maximize the probability (likelihood) of observing the given data. It answers: "What parameters make this data most probable?" MLE directly uses probability distributions as its objective function.

### 2. Core Concepts
- **Likelihood**: P(data | θ) — probability of data given parameters
- **MLE finds**: θ that maximizes L(θ | data)
- **Log-likelihood**: Used in practice (sums instead of products)
- **Frequentist view**: Parameters are fixed, probability is about data

### 3. Mathematical Formulation
**Likelihood Function:**
$$L(\theta | \text{data}) = P(\text{data} | \theta) = \prod_i P(x_i | \theta)$$

**MLE Objective:**
$$\hat{\theta}_{MLE} = \arg\max_\theta L(\theta) = \arg\max_\theta \sum_i \log P(x_i | \theta)$$

### 4. Intuition
Imagine you observed 8 heads in 10 coin flips. Which coin is most likely? One with p=0.5? Or p=0.8? MLE says: find p that makes your observation most probable. Answer: p=0.8.

### 5. MLE → Common Loss Functions
| Model | MLE Assumption | Resulting Loss |
|-------|---------------|----------------|
| Linear Regression | Gaussian errors | Mean Squared Error |
| Logistic Regression | Bernoulli labels | Cross-Entropy |

### 6. Python Example
```python
import numpy as np

n_heads, n_tails = 8, 2

def log_likelihood(p):
    if p <= 0 or p >= 1:
        return -np.inf
    return n_heads * np.log(p) + n_tails * np.log(1 - p)

# Find MLE
p_values = np.linspace(0.01, 0.99, 100)
mle_p = p_values[np.argmax([log_likelihood(p) for p in p_values])]
print(f"MLE estimate: {mle_p:.2f}")  # ≈ 0.8

# Analytical solution: n_heads / total = 8/10 = 0.8
```

### 7. Interview Tips
- MLE vs Bayesian: MLE gives point estimate; Bayesian gives distribution over θ
- MSE loss comes from Gaussian likelihood; Cross-entropy from Bernoulli

---

## Question 17
- [ ] Done

**Describe how to update probabilities using the concept of prior, likelihood, and posterior.**

---

### 1. Definition
Bayesian updating revises beliefs in light of new evidence. Prior (initial belief) × Likelihood (evidence strength) = Posterior (updated belief). This is the foundation of Bayesian inference and learning from data.

### 2. Core Concepts
| Term | Symbol | Role |
|------|--------|------|
| **Prior** | P(H) | Initial belief before evidence |
| **Likelihood** | P(E\|H) | How well hypothesis explains evidence |
| **Posterior** | P(H\|E) | Updated belief after evidence |
| **Evidence** | P(E) | Normalizing constant |

### 3. Mathematical Formulation
$$P(H|E) = \frac{P(E|H) \times P(H)}{P(E)}$$

**Simplified**: Posterior ∝ Likelihood × Prior

**Normalization**: $P(E) = \sum_H P(E|H) \times P(H)$

### 4. Intuition
Start with a belief (prior). See evidence. Update belief (posterior). Strong prior + weak evidence = small update. Weak prior + strong evidence = big update. Today's posterior becomes tomorrow's prior.

### 5. Algorithm Steps (Byheart)
```
1. Define hypotheses H₁, H₂, ... and assign priors P(Hᵢ)
2. Observe evidence E
3. Calculate likelihood P(E|Hᵢ) for each hypothesis
4. Compute unnormalized: P(Hᵢ) × P(E|Hᵢ)
5. Normalize: Divide each by sum of all unnormalized
6. Result: Posterior P(Hᵢ|E)
```

### 6. Python Example
```python
# Is coin fair (p=0.5) or biased (p=0.8)?
priors = {'fair': 0.7, 'biased': 0.3}

# Evidence: observed one Head
likelihoods = {'fair': 0.5, 'biased': 0.8}

# Step 4: Unnormalized posteriors
unnorm = {h: likelihoods[h] * priors[h] for h in priors}
# {'fair': 0.35, 'biased': 0.24}

# Step 5: Normalize
total = sum(unnorm.values())  # 0.59
posteriors = {h: unnorm[h] / total for h in priors}
# {'fair': 0.593, 'biased': 0.407}
```

### 7. Interview Tips
- Posterior is a **compromise** between prior and data
- With infinite data, posterior dominated by likelihood (prior washes out)

---

## Question 18
- [ ] Done

**What are p-values and confidence intervals, and how are they interpreted?**

---

### 1. Definition
**P-value**: Probability of observing data this extreme (or more) assuming null hypothesis is true. **Confidence Interval**: Range likely to contain the true parameter. Both quantify uncertainty in frequentist statistics.

### 2. Core Concepts

**P-Value:**
- Small (≤ 0.05): Data unlikely under H₀ → reject H₀
- Large (> 0.05): Data consistent with H₀ → fail to reject H₀
- **NOT** probability that H₀ is true!

**Confidence Interval (95% CI):**
- If we repeated experiment many times, 95% of CIs would contain true parameter
- **NOT** "95% probability true value is in this interval"

### 3. Mathematical Formulation
**P-value**: $p = P(\text{data this extreme} | H_0 \text{ true})$

**95% CI for mean**: $\bar{x} \pm 1.96 \times \frac{s}{\sqrt{n}}$

### 4. Intuition
- **P-value**: "How surprised should I be if there's really no effect?"
- **CI**: "What's a reasonable range for the true value?"
- If 95% CI excludes 0 → p-value < 0.05 (equivalent tests)

### 5. Practical Example
```python
from scipy import stats
import numpy as np

# A/B test: 10% vs 12.5% conversion
group_a = np.array([1]*100 + [0]*900)
group_b = np.array([1]*125 + [0]*875)

# P-value from t-test
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"P-value: {p_value:.3f}")  # ≈ 0.045

# 95% CI for difference
diff = np.mean(group_b) - np.mean(group_a)
se = np.sqrt(np.var(group_a)/len(group_a) + np.var(group_b)/len(group_b))
ci = (diff - 1.96*se, diff + 1.96*se)
print(f"95% CI: {ci}")  # Doesn't contain 0 → significant
```

### 6. Common Pitfalls
- α = 0.05 is **convention**, not universal truth
- "Not significant" ≠ "No effect" (could be low power)
- Multiple testing inflates false positives → need correction

---

## Question 19
- [ ] Done

**Describe how a probabilistic graphical model (PGM) works.**

---

### 1. Definition
A PGM uses a graph to represent complex probability distributions over multiple random variables. Nodes = variables, edges = dependencies, no edge = conditional independence. It allows compact representation and efficient inference.

### 2. Core Concepts
- **Nodes**: Random variables
- **Edges**: Dependencies (directed or undirected)
- **No edge**: Conditional independence
- **Factorization**: Joint probability = product of local terms

### 3. Types of PGMs
| Type | Edges | Factorization |
|------|-------|---------------|
| **Bayesian Network** | Directed (→) | $P(X_1,...,X_n) = \prod P(X_i | \text{Parents}(X_i))$ |
| **Markov Random Field** | Undirected (—) | $P(X) \propto \prod \psi(\text{clique})$ |

### 4. Intuition
Instead of storing a huge joint probability table (exponential in variables), PGMs factorize into small local tables. A 10-variable problem might need 2¹⁰ = 1024 entries. With PGM structure: maybe only 50 entries.

### 5. Bayesian Network Example
**Structure**: Rain → WetGrass, Sprinkler → WetGrass

**Factorization**:
$$P(R, S, W) = P(R) \times P(S) \times P(W|R,S)$$

### 6. Python Example
```python
# Conditional Probability Tables (CPTs)
p_rain = {'T': 0.2, 'F': 0.8}
p_sprinkler = {'T': 0.3, 'F': 0.7}
p_wet_given = {
    ('T','T'): {'T': 0.99, 'F': 0.01},
    ('T','F'): {'T': 0.80, 'F': 0.20},
    ('F','T'): {'T': 0.90, 'F': 0.10},
    ('F','F'): {'T': 0.00, 'F': 1.00}
}

# Joint: P(Rain=T, Sprinkler=F, Wet=T)
prob = p_rain['T'] * p_sprinkler['F'] * p_wet_given[('T','F')]['T']
# = 0.2 * 0.7 * 0.8 = 0.112
```

### 7. Key Operations & Applications
- **Inference**: P(Disease | Symptoms)
- **Learning**: Estimate CPTs from data
- **Applications**: Medical diagnosis, HMMs, image segmentation

---

## Question 20
- [ ] Done

**Explain the concepts of "Markov Chains" and how they apply to machine learning.**

---

### 1. Definition
A Markov Chain is a stochastic process where the next state depends **only on the current state**, not history. This "memorylessness" is the Markov Property. Used extensively in NLP, RL, and MCMC sampling.

### 2. Core Concepts
- **State Space**: All possible states {S₁, S₂, ..., Sₙ}
- **Transition Matrix (P)**: P[i,j] = P(next=j | current=i)
- **Markov Property**: P(Xₜ₊₁ | Xₜ, Xₜ₋₁, ...) = P(Xₜ₊₁ | Xₜ)
- **Stationary Distribution**: π where πP = π (long-run equilibrium)

### 3. Mathematical Formulation
$$P(X_{t+1} = j | X_t = i, X_{t-1}, ..., X_0) = P(X_{t+1} = j | X_t = i) = P_{ij}$$

**Rows of P sum to 1**: $\sum_j P_{ij} = 1$

### 4. Intuition
Think of weather: Tomorrow's weather depends on today (sunny→sunny likely), not last week. The transition matrix captures these one-step probabilities. Run long enough → settles into stationary distribution.

### 5. ML Applications
| Application | How Used |
|-------------|----------|
| **NLP (N-grams)** | P(word | previous word) |
| **Reinforcement Learning** | MDP = Markov Chain + actions + rewards |
| **HMMs** | Hidden states follow Markov chain |
| **MCMC** | Construct chain with desired stationary distribution |

### 6. Python Example
```python
import numpy as np

# Weather: 0=Sunny, 1=Rainy
P = np.array([
    [0.8, 0.2],  # Sunny → 80% Sunny, 20% Rainy
    [0.4, 0.6]   # Rainy → 40% Sunny, 60% Rainy
])

def simulate(start, n_steps):
    states = [start]
    for _ in range(n_steps - 1):
        states.append(np.random.choice(2, p=P[states[-1]]))
    return states

forecast = simulate(0, 10)  # Start sunny, 10 days
```

### 7. Interview Tip
Stationary distribution: Run chain forever → probability of being in each state converges regardless of start.

---

## Question 21
- [ ] Done

**What is Expectation-Maximization (EM) algorithm and how does probability play a role in it?**

---

### 1. Definition
EM is an iterative algorithm for finding MLE/MAP estimates when data has latent (unobserved) variables. Each iteration increases likelihood, converging to a (local) maximum.

### 2. Core Concepts
- **Latent Variables**: Hidden variables we can't directly observe
- **E-Step**: Compute expected value using current θ estimates
- **M-Step**: Maximize expected log-likelihood to update θ
- **Convergence**: Guaranteed to improve (or stay same) each iteration

### 3. Algorithm Steps (Byheart)
```
1. Initialize parameters θ randomly
2. E-Step: Compute P(latent | observed, θ) for all data
3. M-Step: θ_new = argmax E[log L(θ)] using E-step probabilities
4. Check convergence; if not converged, go to step 2
```

### 4. Intuition
**Analogy - Heights of men/women mixed (unknown gender):**
1. **E**: Guess probability each person is man/woman based on current mean estimates
2. **M**: Update mean height estimates using weighted average
3. **Repeat** until stable

### 5. Main Application: Gaussian Mixture Models
| Element | Role |
|---------|------|
| **Observed** | Data points x |
| **Latent** | Cluster assignments z |
| **Parameters** | Means μₖ, covariances Σₖ, weights πₖ |

### 6. Python Example (Conceptual GMM)
```python
import numpy as np

# E-Step: Compute responsibilities
def e_step(X, means, weights):
    n_clusters = len(means)
    resp = np.zeros((len(X), n_clusters))
    for k in range(n_clusters):
        resp[:, k] = weights[k] * gaussian_pdf(X, means[k])
    resp /= resp.sum(axis=1, keepdims=True)  # Normalize
    return resp  # P(cluster k | data point)

# M-Step: Update parameters
def m_step(X, resp):
    n_k = resp.sum(axis=0)  # Soft counts
    weights = n_k / len(X)
    means = (resp.T @ X) / n_k[:, None]
    return means, weights
```

### 7. Interview Tips
- EM can get stuck in local maxima → run multiple random initializations
- E-step uses probability (posterior); M-step uses maximum likelihood

---

## Question 22
- [ ] Done

**Describe how Monte Carlo simulations are used in machine learning for approximation of probabilities.**

---

### 1. Definition
Monte Carlo methods use repeated random sampling to obtain numerical results. Based on Law of Large Numbers: sample average → true expectation as N → ∞.

### 2. Core Idea
$$P(A) \approx \frac{\text{Count of samples where A occurs}}{\text{Total samples N}}$$

$$E[f(X)] \approx \frac{1}{N}\sum_{i=1}^{N} f(x_i) \text{ where } x_i \sim P(X)$$

### 3. Algorithm Steps (Byheart)
```
1. Define the quantity to estimate
2. Design sampling procedure
3. Generate N random samples
4. Compute sample statistic
5. Result: Approximation with error O(1/√N)
```

### 4. ML Applications
| Application | How Monte Carlo is Used |
|-------------|------------------------|
| **MCMC (Bayesian)** | Sample from posterior without intractable integrals |
| **Dropout at Test** | Sample multiple predictions → uncertainty estimate |
| **Reinforcement Learning** | Estimate state values by averaging returns |
| **Integration** | Approximate high-dimensional integrals |

### 5. Python Example: Estimating π
```python
import numpy as np

def estimate_pi(n_samples):
    # Random points in unit square
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    
    # Check if inside unit circle
    inside = (x**2 + y**2) <= 1
    
    # Area of circle / Area of square = π/4
    return 4 * np.mean(inside)

print(f"π ≈ {estimate_pi(100000):.5f}")  # ≈ 3.14159
```

### 6. Key Insights
- **Accuracy** ∝ √N (need 4x samples for 2x precision)
- **Variance reduction**: importance sampling, stratification
- **Foundation for MCMC**: Metropolis-Hastings, Gibbs sampling

### 7. Interview Tip
Use Monte Carlo when analytical solutions are intractable — common in high-dimensional probability spaces and Bayesian inference.

---

## Question 23
- [ ] Done

**What are the probabilistic underpinnings of Active Learning and how might they be utilized in algorithm design?**

---

### 1. Definition
Active Learning selects the most informative unlabeled samples to query for labels. "Informativeness" is measured using model uncertainty — query points the model is least certain about.

### 2. Core Concepts
- **Pool-based**: Large unlabeled pool, select best to label
- **Uncertainty Sampling**: Query where model is most uncertain
- **Goal**: Maximum performance with minimum labeled data
- **Probabilistic requirement**: Model must output probabilities

### 3. Query Strategies (Byheart)
| Strategy | Selects Sample Where | Formula |
|----------|---------------------|---------|
| **Least Confident** | max P(ŷ\|x) is lowest | $\arg\min_x \max_y P(y|x)$ |
| **Margin Sampling** | Top two classes close | $\arg\min_x [P(\hat{y}_1|x) - P(\hat{y}_2|x)]$ |
| **Entropy** | Highest uncertainty | $\arg\max_x H(P(y|x))$ |

### 4. Algorithm Steps
```
1. Start: small labeled set L, large unlabeled pool U
2. Train model M on L
3. LOOP until budget exhausted:
   a. Compute uncertainty for each x ∈ U
   b. Select x* with highest uncertainty
   c. Query oracle for label y*
   d. Move (x*, y*) from U to L
   e. Retrain M on updated L
```

### 5. Python Example
```python
import numpy as np

def active_learning_step(model, X_unlabeled, oracle):
    model.fit(X_labeled, y_labeled)
    
    # Compute entropy for each unlabeled sample
    probs = model.predict_proba(X_unlabeled)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    
    # Select most uncertain
    idx = np.argmax(entropy)
    x_query = X_unlabeled[idx]
    y_query = oracle.label(x_query)
    
    return x_query, y_query
```

### 6. Practical Applications
- **Medical imaging**: Expert labels are expensive
- **NLP annotation**: Reduce manual labeling effort
- **Any domain**: Where labeling cost >> compute cost

### 7. Interview Tips
- Model must output calibrated probabilities
- Consider batch selection (query k samples at once) for efficiency
- Trade-off: exploitation (refine decision boundary) vs exploration (cover input space)

---
