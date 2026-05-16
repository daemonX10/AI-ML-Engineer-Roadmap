# Statistics Interview Questions - General Questions

## Question 1
- [ ] Done

**What is the difference between population and sample?**

---

### 1. Definition
- **Population**: The ENTIRE set of individuals/items of interest
- **Sample**: A SUBSET drawn from the population for analysis

### 2. Why Samples?
| Challenge | Solution |
|-----------|----------|
| Too expensive to measure everyone | Sample is cost-effective |
| Population may be infinite | Sample is finite |
| Time constraints | Sample is faster |
| Destructive testing | Can't test all items |

### 3. Mathematical Notation

| Measure | Population | Sample |
|---------|------------|--------|
| Size | N | n |
| Mean | μ (mu) | x̄ (x-bar) |
| Std Dev | σ (sigma) | s |
| Variance | σ² | s² |
| Proportion | P | p̂ (p-hat) |

### 4. Key Concepts
- **Parameter**: Value describing a population (usually unknown)
- **Statistic**: Value computed from a sample (estimates parameter)
- **Sampling Error**: Difference between sample statistic and population parameter

### 5. Interview Tip
Always clarify whether you're working with a population or sample — it affects which formulas to use (n vs n-1 in variance denominator).

---

## Question 2
- [ ] Done

**Explain correlation vs causation with an example.**

---

### 1. Definition
- **Correlation**: Two variables move together (positive or negative)
- **Causation**: One variable DIRECTLY influences the other

### 2. Key Insight
**Correlation ≠ Causation**

Just because two things are correlated doesn't mean one causes the other.

### 3. Classic Examples

**Ice Cream & Drowning:**
- High correlation: Both increase in summer
- True cause: Hot weather (confounding variable)
- Ice cream doesn't cause drowning!

**Shoe Size & Reading Ability:**
- Positive correlation in children
- True cause: Age (confounding variable)
- Bigger feet don't improve reading!

### 4. Confounding Variables
A third variable that influences both X and Y:

```
Temperature (Confounder)
    ↓           ↓
Ice Cream    Drowning
  Sales      Incidents
```

### 5. How to Establish Causation

| Method | Description |
|--------|-------------|
| **Randomized Experiment** | Gold standard — randomly assign treatment |
| **A/B Testing** | Digital experiment with random assignment |
| **Natural Experiment** | Exploit naturally occurring randomization |
| **Instrumental Variables** | Use external factors affecting only X |

### 6. Interview Tip
When presented with a correlation, always ask: "Could there be a confounding variable?"

---

## Question 3
- [ ] Done

**What is sampling bias and how do you avoid it?**

---

### 1. Definition
**Sampling Bias**: When the sample is not representative of the population, leading to skewed results.

### 2. Types of Sampling Bias

| Type | Description | Example |
|------|-------------|---------|
| **Selection Bias** | Non-random selection | Surveying only gym members about exercise habits |
| **Survivorship Bias** | Only successful cases visible | Studying only successful startups |
| **Non-response Bias** | Certain groups don't respond | Phone surveys missing people who don't answer |
| **Convenience Sampling** | Using easily accessible subjects | Polling only friends |

### 3. Famous Example: 1936 Literary Digest Poll
- Predicted Landon would beat FDR
- Sampled from phone books & car registrations
- Bias: Only wealthy people had phones/cars in 1936
- Result: Completely wrong prediction

### 4. How to Avoid Sampling Bias

| Method | Description |
|--------|-------------|
| **Simple Random Sampling** | Equal probability for all |
| **Stratified Sampling** | Ensure subgroups represented |
| **Systematic Sampling** | Every kth element |
| **Cluster Sampling** | Sample entire groups |

### 5. ML Relevance
- Training data must represent deployment data
- Dataset shift: Training ≠ Production distribution
- Example: Training face recognition only on light-skinned faces

---

## Question 4
- [ ] Done

**What is the difference between Type I and Type II errors?**

---

### 1. Definition

| Error | Name | Description | Verdict |
|-------|------|-------------|---------|
| **Type I (α)** | False Positive | Reject H₀ when true | Guilty verdict for innocent |
| **Type II (β)** | False Negative | Fail to reject H₀ when false | Innocent verdict for guilty |

### 2. Medical Testing Example

| Reality | Test Positive | Test Negative |
|---------|---------------|---------------|
| **Disease Present** | ✓ True Positive | ✗ Type II (miss disease) |
| **Disease Absent** | ✗ Type I (false alarm) | ✓ True Negative |

### 3. Controlling Errors

| Error | Controlled By | Typical Value |
|-------|---------------|---------------|
| Type I (α) | Significance level | 0.05 |
| Type II (β) | Statistical power = 1-β | 0.80 |

### 4. Trade-off
- Lower α → Higher β (more conservative = more misses)
- Can't minimize both simultaneously
- Must choose based on consequence severity

### 5. Real-World Decisions

| Scenario | Worse Error | Why |
|----------|-------------|-----|
| Cancer screening | Type II | Missing cancer is dangerous |
| Spam filter | Type I | Blocking important emails is bad |
| Criminal trial | Type I | Convicting innocent is worse |
| Drug approval | Type I | Harmful drug reaching market |

### 6. Interview Tip
Always consider which error is MORE costly for the specific application.

---

## Question 5
- [ ] Done

**Explain the concept of confidence intervals.**

---

### 1. Definition
A **confidence interval (CI)** is a range of values that likely contains the true population parameter.

### 2. Correct Interpretation
"If we repeated this study many times, 95% of the computed intervals would contain the true parameter."

**NOT**: "There's a 95% probability the true value is in this interval."

### 3. Formula (for mean)
$$CI = \bar{x} \pm z_{\alpha/2} \times \frac{s}{\sqrt{n}}$$

| Component | Meaning |
|-----------|---------|
| x̄ | Sample mean |
| z | Z-score for confidence level |
| s | Sample standard deviation |
| n | Sample size |

### 4. Common Confidence Levels

| Confidence | Z-score | Width |
|------------|---------|-------|
| 90% | 1.645 | Narrower |
| 95% | 1.96 | Standard |
| 99% | 2.576 | Wider |

### 5. Width Determinants
- **Larger sample (n↑)**: Narrower CI (more precise)
- **Higher confidence**: Wider CI (more certain)
- **Higher variability (s↑)**: Wider CI

### 6. ML Application
- Model uncertainty quantification
- A/B test results: "Conversion increased by 5% ± 2%"
- If CI includes 0, effect is not statistically significant

---

## Question 6
- [ ] Done

**What is the law of large numbers?**

---

### 1. Definition
As sample size increases, the sample mean converges to the population mean.

$$\lim_{n \to \infty} \bar{X}_n = \mu$$

### 2. Two Forms

| Form | Statement |
|------|-----------|
| **Weak LLN** | Sample mean converges in probability |
| **Strong LLN** | Sample mean converges almost surely |

### 3. Example: Coin Flips

| Flips | Heads | Proportion |
|-------|-------|------------|
| 10 | 6 | 0.60 |
| 100 | 48 | 0.48 |
| 1000 | 502 | 0.502 |
| 10000 | 4987 | 0.4987 |

→ Approaches true probability (0.5) as n increases

### 4. Requirements
- Independent observations
- Identically distributed
- Finite mean and variance

### 5. NOT Gambler's Fallacy
- LLN: Long-run average converges
- Gambler's Fallacy: "I'm due for a win" (WRONG)
- Each trial is independent; past doesn't affect future

### 6. ML Applications
- Monte Carlo methods converge with more samples
- Training on more data → better estimates
- Mini-batch gradient descent averaging

---

## Question 7
- [ ] Done

**What is multicollinearity and why is it a problem?**

---

### 1. Definition
**Multicollinearity**: When independent variables in a regression model are highly correlated with each other.

### 2. Problems Caused

| Issue | Description |
|-------|-------------|
| **Unstable coefficients** | Small data changes → large coefficient changes |
| **Inflated standard errors** | Coefficients appear insignificant |
| **Interpretation difficulty** | Can't isolate individual variable effects |
| **Sign reversal** | Coefficients may have wrong sign |

### 3. Detection Methods

**Variance Inflation Factor (VIF):**
$$VIF_i = \frac{1}{1 - R_i^2}$$

| VIF | Interpretation |
|-----|----------------|
| 1 | No correlation |
| 1-5 | Moderate (acceptable) |
| 5-10 | High (concerning) |
| >10 | Severe multicollinearity |

**Correlation Matrix:**
- Look for correlations > 0.7 or < -0.7

### 4. Solutions

| Solution | How |
|----------|-----|
| Remove one variable | Drop highly correlated feature |
| Combine variables | Create composite (e.g., average) |
| PCA | Transform to uncorrelated components |
| Ridge Regression | L2 regularization handles it |

### 5. Python Check
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
```

---

## Question 8
- [ ] Done

**Explain the difference between parametric and non-parametric tests.**

---

### 1. Definition

| Type | Assumption | Examples |
|------|------------|----------|
| **Parametric** | Data follows specific distribution (usually normal) | t-test, ANOVA, Pearson |
| **Non-parametric** | No distributional assumptions | Mann-Whitney, Kruskal-Wallis, Spearman |

### 2. When to Use Each

**Use Parametric When:**
- Data is normally distributed
- Sample size is large (n > 30)
- Variance is homogeneous
- More statistical power

**Use Non-parametric When:**
- Data is skewed
- Small sample size
- Ordinal data
- Outliers present

### 3. Comparison Table

| Parametric | Non-parametric Equivalent |
|------------|---------------------------|
| Independent t-test | Mann-Whitney U |
| Paired t-test | Wilcoxon signed-rank |
| One-way ANOVA | Kruskal-Wallis |
| Pearson correlation | Spearman correlation |

### 4. Trade-offs

| Aspect | Parametric | Non-parametric |
|--------|------------|----------------|
| Power | Higher | Lower |
| Assumptions | Strict | Flexible |
| Sample size | Works with smaller n | Needs larger n for same power |

### 5. Interview Tip
Check normality first (Shapiro-Wilk test, Q-Q plot), then choose appropriate test.

---

## Question 9
- [ ] Done

**What is statistical power and how do you increase it?**

---

### 1. Definition
**Statistical Power** = Probability of correctly rejecting H₀ when it's false
= 1 - β (where β = Type II error rate)

### 2. Typical Target
- Power ≥ 0.80 (80% chance of detecting true effect)

### 3. Power Depends On

| Factor | Effect on Power |
|--------|-----------------|
| Effect size ↑ | Power ↑ |
| Sample size ↑ | Power ↑ |
| Significance level (α) ↑ | Power ↑ |
| Variance ↓ | Power ↑ |

### 4. Power Formula Components
$$Power = f(effect\ size,\ n,\ \alpha,\ \sigma)$$

### 5. How to Increase Power

| Method | How |
|--------|-----|
| Increase sample size | Most common approach |
| Use one-tailed test | If direction is known |
| Reduce measurement error | Better instruments |
| Use paired designs | Reduce variance |
| Increase α | Trade-off with Type I error |

### 6. Power Analysis (Before Experiment)
```python
from statsmodels.stats.power import TTestIndPower

power_analysis = TTestIndPower()
sample_size = power_analysis.solve_power(
    effect_size=0.5,  # Cohen's d
    power=0.8,
    alpha=0.05
)
```

### 7. Interview Tip
Always do power analysis BEFORE collecting data to ensure adequate sample size.

---

## Question 10
- [ ] Done

**What is heteroscedasticity and how do you detect it?**

---

### 1. Definition
**Heteroscedasticity**: When the variance of residuals is not constant across all levels of the independent variable.

**Opposite**: Homoscedasticity (constant variance — desired)

### 2. Visual Detection
- Residual plot: Fan or cone shape indicates heteroscedasticity
- Variance increases/decreases with fitted values

### 3. Formal Tests

| Test | Null Hypothesis |
|------|-----------------|
| **Breusch-Pagan** | Residual variance is constant |
| **White's Test** | Residual variance is constant |
| **Goldfeld-Quandt** | Variance is same in two subgroups |

### 4. Problems Caused
- OLS estimates still unbiased BUT:
  - Standard errors are wrong
  - Confidence intervals invalid
  - Hypothesis tests unreliable

### 5. Solutions

| Solution | Method |
|----------|--------|
| Weighted Least Squares (WLS) | Weight by inverse variance |
| Robust standard errors | Huber-White sandwich estimator |
| Transform dependent variable | Log, square root |
| Use different model | GLS, quantile regression |

### 6. Python Detection
```python
from statsmodels.stats.diagnostic import het_breuschpagan

bp_test = het_breuschpagan(residuals, X)
print(f"p-value: {bp_test[1]}")
# p < 0.05 → heteroscedasticity present
```

---

## Question 11
- [ ] Done

**Explain the difference between one-tailed and two-tailed tests.**

---

### 1. Definition

| Type | Hypotheses | When to Use |
|------|------------|-------------|
| **Two-tailed** | H₁: μ ≠ μ₀ | Don't know direction |
| **One-tailed** | H₁: μ > μ₀ or μ < μ₀ | Know expected direction |

### 2. Visual Difference

**Two-tailed (α = 0.05):**
- 2.5% in each tail
- Critical z = ±1.96

**One-tailed (α = 0.05):**
- 5% in one tail
- Critical z = 1.645 (or -1.645)

### 3. When to Use One-Tailed

| Use Case | Direction |
|----------|-----------|
| "Is new drug BETTER?" | Upper tail |
| "Is process FASTER?" | Lower tail |
| "Did engagement INCREASE?" | Upper tail |

### 4. Trade-offs

| Aspect | One-tailed | Two-tailed |
|--------|------------|------------|
| Power | Higher (for correct direction) | Lower |
| Critical value | Less extreme | More extreme |
| Risk | Miss opposite direction effect | None |

### 5. Interview Tip
One-tailed tests should be decided BEFORE seeing data based on prior knowledge, not after seeing results.

---

## Question 12
- [ ] Done

**What is Simpson's Paradox?**

---

### 1. Definition
A trend that appears in different groups of data **reverses** when the groups are combined.

### 2. Classic Example: UC Berkeley Admissions

**Overall:**
| Gender | Admitted | Applied | Rate |
|--------|----------|---------|------|
| Men | 8442 | 44.5% | |
| Women | 4321 | 35% | |

→ Looks like gender bias against women!

**By Department:**
| Department | Men Admit | Women Admit |
|------------|-----------|-------------|
| A | 62% | 82% |
| B | 63% | 68% |
| ... | | |

→ Women had HIGHER admission rates in most departments!

### 3. Explanation
- Women applied to more competitive departments
- Department choice was the confounding variable
- Aggregation created false impression

### 4. Lurking Variable
The hidden variable that explains the paradox:
- Department competitiveness
- Baseline rates differ by group

### 5. Key Lesson
**Always segment data by relevant confounders before drawing conclusions.**

### 6. ML Relevance
- Feature importance can flip when conditioning on other features
- Model performance can vary by subgroup
- Always check disaggregated metrics

---
