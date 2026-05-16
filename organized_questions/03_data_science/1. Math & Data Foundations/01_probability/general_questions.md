# Probability Interview Questions - General Questions

## Question 1
- [ ] Done

**Define the terms 'sample space' and 'event' in probability.**

---

### 1. Definition
**Sample Space (S or Ω)**: The set of ALL possible outcomes of a random experiment.
**Event (E)**: A **subset** of the sample space — specific outcomes we're interested in.

### 2. Examples

| Experiment | Sample Space | Example Event |
|------------|--------------|---------------|
| Rolling a die | {1, 2, 3, 4, 5, 6} | "Rolling even" = {2, 4, 6} |
| Coin flip | {H, T} | "Heads" = {H} |
| Two coin flips | {HH, HT, TH, TT} | "At least one head" = {HH, HT, TH} |

### 3. Mathematical Formulation
$$P(E) = \frac{|E|}{|S|} = \frac{\text{Favorable outcomes}}{\text{Total outcomes}}$$

### 4. Key Terms
- **Outcome**: Single element of sample space
- **Simple event**: Single outcome {5}
- **Compound event**: Multiple outcomes {2, 4, 6}

### 5. Interview Tip
Always define complete sample space first when solving probability problems — it determines what probabilities are valid.

---

## Question 2
- [ ] Done

**What does it mean for two events to be independent?**

---

### 1. Definition
Two events A and B are **independent** if the occurrence of one does not affect the probability of the other. Knowing A happened gives no information about B.

### 2. Mathematical Formulation
$$P(A \cap B) = P(A) \times P(B)$$

**Equivalently:**
- P(A | B) = P(A)
- P(B | A) = P(B)

### 3. Examples

**Independent:**
- Rolling a die and flipping a coin
- P(6 AND Heads) = (1/6) × (1/2) = 1/12

**Dependent:**
- Drawing cards WITHOUT replacement
- P(King first) = 4/52, P(King second | King first) = 3/51 ≠ 4/52

### 4. Independence vs Mutual Exclusivity

| Property | Definition | Can Co-occur? |
|----------|------------|---------------|
| **Independent** | A doesn't affect P(B) | Yes |
| **Mutually Exclusive** | A and B cannot both occur | No (P(A∩B)=0) |

**Key Insight:** Mutually exclusive events are DEPENDENT (if A happens, P(B) = 0)

### 5. ML Relevance
- Naive Bayes assumes features are conditionally independent given class
- Violated assumption → biased probability estimates but often still works

---

## Question 3
- [ ] Done

**Define expectation, variance, and covariance.**

---

### 1. Expectation E[X]
**Definition**: Long-run average value (center of mass).

| Type | Formula |
|------|---------|
| Discrete | $E[X] = \sum x \cdot P(X=x)$ |
| Continuous | $E[X] = \int x \cdot f(x) dx$ |

**Example**: Fair die → E[X] = (1+2+3+4+5+6)/6 = 3.5

### 2. Variance Var(X)
**Definition**: Measure of spread around the mean.

$$Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

**Standard Deviation**: σ = √Var(X) (same units as X)

### 3. Covariance Cov(X, Y)
**Definition**: How two variables move together.

$$Cov(X, Y) = E[(X - E[X])(Y - E[Y])]$$

| Value | Interpretation |
|-------|----------------|
| Cov > 0 | X↑ → Y↑ (positive relationship) |
| Cov < 0 | X↑ → Y↓ (inverse relationship) |
| Cov = 0 | No linear relationship |

### 4. Correlation
Normalized covariance (scale-independent):
$$\rho = \frac{Cov(X,Y)}{\sigma_X \sigma_Y} \in [-1, 1]$$

### 5. ML Applications
| Concept | Use Case |
|---------|----------|
| Expectation | Expected return, RL value functions |
| Variance | Bias-variance tradeoff, uncertainty |
| Covariance | PCA, portfolio optimization, multivariate Gaussians |

---

## Question 4
- [ ] Done

**How do probabilistic models cope with uncertainty in predictions?**

---

### 1. Core Approach
Output **probability distributions** over outcomes instead of single point predictions.

### 2. Types of Uncertainty

| Type | Description | Reducible? |
|------|-------------|------------|
| **Aleatoric** | Inherent data randomness (noise) | No |
| **Epistemic** | Model's lack of knowledge | Yes (more data) |

### 3. Methods

**a) Probability Distribution Output**
- Classification: P(class | features) not just class label
- Regression: (μ, σ) — mean and uncertainty

**b) Bayesian Inference**
- Parameters as distributions, not point values
- Wider posteriors in low-data regions

**c) Ensemble Methods**
- Train multiple models (Random Forest, MC Dropout)
- Variance across predictions = uncertainty

### 4. Python Example
```python
# Point prediction
prediction = 85  # House price: $85k

# Probabilistic prediction
prediction = {'mean': 85, 'std': 12}
# 95% CI: [85 - 2*12, 85 + 2*12] = [$61k, $109k]
```

### 5. Applications
- Self-driving cars: High uncertainty → cautious action
- Medical diagnosis: Flag uncertain cases for review
- Active learning: Query most uncertain samples

---

## Question 5
- [ ] Done

**How is probability used in Bayesian inference for machine learning?**

---

### 1. Core Idea
Treat model parameters as random variables with probability distributions, updated as data is observed.

### 2. Bayes' Theorem for ML
$$P(\theta | D) = \frac{P(D | \theta) \cdot P(\theta)}{P(D)}$$

### 3. Components

| Term | Name | Role in ML |
|------|------|------------|
| P(θ) | **Prior** | Initial belief (acts as regularization) |
| P(D\|θ) | **Likelihood** | How well θ explains data |
| P(θ\|D) | **Posterior** | Updated belief after data |
| P(D) | **Evidence** | For model comparison |

### 4. How Each is Used

| Component | ML Usage |
|-----------|----------|
| **Prior** | L2 = Gaussian prior; L1 = Laplace prior |
| **Likelihood** | Same as MLE objective |
| **Posterior** | Full uncertainty, not just point estimate |
| **Evidence** | Bayesian model selection |

### 5. Key Benefit
Posterior captures:
- **Best estimate**: posterior mean
- **Uncertainty**: posterior variance

### 6. Interview Tip
Probability encodes: prior beliefs, data generation, updated beliefs, and parameter uncertainty — all in one coherent framework.

---

## Question 6
- [ ] Done

**How can assuming independence in probabilistic models lead to inaccuracies?**

---

### 1. The Problem
Assuming independence when features are correlated misrepresents the true joint probability distribution.

### 2. How Inaccuracies Occur

**a) Double Counting Evidence**
| Scenario | Reality | Naive Bayes |
|----------|---------|-------------|
| "Viagra" + "enhancement" | Correlated (~1.5x evidence) | Treated as 2x evidence |
| Result | P(spam) = 0.90 | P(spam) = 0.999 (overconfident) |

**b) Missing Interactions**
- High temp alone: weak evidence
- Cough alone: weak evidence
- BOTH together: strong evidence (flu)
- Naive Bayes cannot capture "combination > sum of parts"

### 3. Consequences
- Overconfident predictions
- Poor probability calibration
- Wrong threshold selection

### 4. Solutions
| Solution | How it Helps |
|----------|--------------|
| Combine correlated features | Reduce redundancy |
| Add interaction terms | Capture joint effects |
| Use trees/NNs | Automatically learn interactions |
| Calibration (Platt scaling) | Fix probability estimates post-hoc |

### 5. Interview Tip
Despite violated assumptions, Naive Bayes often still picks the right class — just with wrong probability estimates.

---

## Question 7
- [ ] Done

**What strategies would you use to handle missing data in probabilistic models?**

---

### 1. Strategies (Simple → Advanced)

| Method | Description | Pros/Cons |
|--------|-------------|-----------|
| **Mean/Mode Imputation** | Replace with average | Simple; reduces variance |
| **Model-Based** | Predict missing from other features | Preserves relationships |
| **EM Algorithm** | Treat missing as latent variables | Joint optimization |
| **Multiple Imputation (MICE)** | Create M imputed datasets | Propagates uncertainty correctly |

### 2. EM for Missing Data
```
E-Step: Estimate expected value of missing data given θ
M-Step: Re-estimate θ using "completed" data
Iterate until convergence
```

### 3. Multiple Imputation (Recommended)
```
1. Create M different completed datasets (M=5)
2. Each imputation adds noise reflecting uncertainty
3. Run analysis on each dataset
4. Pool results using Rubin's rules
```

### 4. Decision Guide

| Situation | Recommended |
|-----------|-------------|
| <5% missing | Simple imputation or deletion |
| Missing at Random | MICE |
| Complex relationships | EM with GMMs |
| Need uncertainty | Multiple Imputation |

### 5. Interview Tip
Avoid listwise deletion unless minimal missing data — wastes information and can introduce bias.

---

## Question 8
- [ ] Done

**How do you determine the significance of an observed effect using probability?**

---

### 1. Framework: Hypothesis Testing

**Steps:**
1. **Formulate**: H₀ (no effect) vs H₁ (effect exists)
2. **Set α**: Significance level (usually 0.05)
3. **Compute p-value**: P(data this extreme | H₀ true)
4. **Decide**: p ≤ α → reject H₀

### 2. Interpretation

| P-value | Meaning |
|---------|---------|
| Small (≤0.05) | Data unlikely under H₀ → reject H₀ |
| Large (>0.05) | Data consistent with H₀ → fail to reject |

### 3. Python Example
```python
from scipy.stats import ttest_ind

# A/B Test: 10% vs 12.5% conversion
group_a = [1]*100 + [0]*900
group_b = [1]*125 + [0]*875

t_stat, p_value = ttest_ind(group_a, group_b)
# p_value ≈ 0.045 → significant at α=0.05
```

### 4. Common Mistakes

| Mistake | Reality |
|---------|---------|
| "P-value = P(H₀ true)" | No, it's about data given H₀ |
| "Not significant = no effect" | Could be low power |
| "Significant = important" | Statistical ≠ practical significance |

### 5. Interview Tip
Always mention effect size alongside p-value — tiny effects can be "significant" with large N.

---

## Question 9
- [ ] Done

**How do Hidden Markov Models (HMMs) use probability in sequential data modeling?**

---

### 1. Definition
HMM models sequences of observations generated by hidden states that follow a Markov chain.

### 2. Probabilistic Components

| Component | Symbol | Description |
|-----------|--------|-------------|
| **Transition** | A | P(state j at t | state i at t-1) |
| **Emission** | B | P(observation k | hidden state j) |
| **Initial** | π | P(start in state i) |

### 3. Structure
```
Hidden:    [Noun] → [Verb] → [Noun] → ...
             ↓         ↓         ↓
Observed: "Time"   "flies"   "quickly"
```

### 4. Three Fundamental Problems

| Problem | Question | Algorithm |
|---------|----------|-----------|
| **Evaluation** | P(observations \| model)? | Forward |
| **Decoding** | Most likely hidden states? | Viterbi |
| **Learning** | Best parameters? | Baum-Welch (EM) |

### 5. Applications
- **POS Tagging**: Hidden = tags, Observed = words
- **Speech Recognition**: Hidden = phonemes, Observed = audio
- **Bioinformatics**: Hidden = gene states, Observed = DNA sequence

### 6. Interview Tip
HMMs are entirely probabilistic: two stochastic processes (hidden state evolution + observation generation) working together.

---

## Question 10
- [ ] Done

**How has the advent of Quantum Computing influenced probabilistic algorithms?**

---

### 1. Quantum Fundamentals

| Concept | Description |
|---------|-------------|
| **Superposition** | Qubit in both 0 and 1: α\|0⟩ + β\|1⟩ |
| **Measurement** | Collapses to P(0)=\|α\|², P(1)=\|β\|² |
| **Entanglement** | Correlated qubits; measuring one affects other |
| **Interference** | Amplitudes can cancel or reinforce |

### 2. Impact on Probabilistic Algorithms

| Application | Quantum Advantage |
|-------------|-------------------|
| **Sampling** | Sample from complex distributions faster |
| **MCMC** | Potential Bayesian inference speedup |
| **Optimization** | Quantum annealing finds global minima |
| **Search** | Grover's: √N speedup |

### 3. Quantum ML Examples
- Quantum SVM, Quantum PCA
- Boltzmann Machine training via quantum sampling
- Exponential speedup for certain linear algebra

### 4. Current State (NISQ Era)
- Limited qubits, high noise
- Specific problems show advantage
- Hybrid classical-quantum approaches

### 5. Interview Tip
Quantum computing doesn't just speed up classical algorithms — it operates on fundamentally different (quantum) probability rules.

---

## Question 11
- [ ] Done

**What role does probability play in reinforcement learning and decision making?**

---

### 1. Probability in RL Framework

| Component | Probabilistic Element |
|-----------|----------------------|
| **Transitions** | P(s' \| s, a) — stochastic environment |
| **Policy** | π(a \| s) — probability of action given state |
| **Value** | E[Σγᵗ Rₜ] — expected cumulative reward |

### 2. Why Stochastic?
- **Environment**: Same action → different outcomes
- **Policy**: Exploration, optimal mixed strategies
- **Future**: Maximize expected value, not single outcome

### 3. MDP Components

| Element | Role |
|---------|------|
| States S | Current situation |
| Actions A | Agent choices |
| Transitions P | P(s' \| s, a) |
| Rewards R | Feedback signal |
| Policy π | π(a \| s) |

### 4. Decision Loop
```
1. Observe state s
2. Choose action a ~ π(a|s)
3. Environment transitions: s' ~ P(s'|s,a)
4. Receive reward R
5. Goal: maximize E[total future reward]
```

### 5. Interview Tip
Probability handles: uncertain environment (transitions), uncertain behavior (policy), and uncertain future (expected return).

---

## Question 12
- [ ] Done

**How do GANs (Generative Adversarial Networks) utilize probability theory?**

---

### 1. Probabilistic Goal
Learn to generate samples from true data distribution p_data(x).

### 2. The Two Players

| Player | Input | Output | Goal |
|--------|-------|--------|------|
| **Generator G** | Noise z ~ P(z) | Fake sample G(z) | Make p_g match p_data |
| **Discriminator D** | Sample x | P(x is real) | Correctly classify |

### 3. Loss Function
$$\min_G \max_D \left[ E_{x \sim p_{data}}[\log D(x)] + E_{z \sim p_z}[\log(1 - D(G(z)))] \right]$$

### 4. Equilibrium
When training succeeds:
- p_g = p_data (generator mimics real data)
- D(x) = 0.5 for all x (can't distinguish)

### 5. Probability Theory Connection
- Minimizes **Jensen-Shannon divergence** between p_g and p_data
- JS divergence measures similarity between probability distributions
- D outputs probability estimates used in loss

### 6. Summary Flow
```
z ~ P(z) → G(z) ~ p_g → D outputs P(real) → Loss optimizes distributions
```

### 7. Interview Tip
GANs are fundamentally about making one probability distribution (generator's) match another (data's).

---

