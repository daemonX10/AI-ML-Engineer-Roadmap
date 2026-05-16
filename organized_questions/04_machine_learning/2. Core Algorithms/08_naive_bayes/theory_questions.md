# Naive Bayes Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What is the Naive Bayes classifier and how does it work?**

**Answer:**

Naive Bayes is a probabilistic classifier based on Bayes' Theorem that assumes all features are conditionally independent given the class label. It calculates the posterior probability of each class for a given input and assigns the class with the highest probability. Despite its "naive" independence assumption, it works remarkably well for text classification and other high-dimensional problems.

**Core Concepts:**
- Based on Bayes' Theorem (posterior = likelihood × prior / evidence)
- "Naive" assumption: features are independent given class
- Computes P(class | features) for each class
- Predicts the class with maximum posterior probability
- Generative model (learns P(X|Y) and P(Y))

**Mathematical Formulation:**

$$P(Y|X_1, X_2, ..., X_n) = \frac{P(Y) \cdot \prod_{i=1}^{n} P(X_i|Y)}{P(X_1, X_2, ..., X_n)}$$

Since denominator is constant for all classes:
$$\hat{Y} = \arg\max_Y P(Y) \cdot \prod_{i=1}^{n} P(X_i|Y)$$

**Intuition:**
Think of a doctor diagnosing disease. Given symptoms (fever, cough), the doctor estimates which disease is most probable. Naive Bayes does this mathematically - it combines prior knowledge (disease prevalence) with evidence (symptom likelihoods) to make predictions.

**Practical Relevance:**
- Spam detection (classic use case)
- Document classification and sentiment analysis
- Real-time prediction (very fast inference)
- Medical diagnosis systems
- Works well with small training data

**Algorithm Steps:**
1. Calculate prior probability P(Y) for each class from training data
2. For each feature, calculate likelihood P(Xi|Y) for each class
3. For new instance, compute posterior for each class using Bayes formula
4. Predict class with highest posterior probability

---

## Question 2
- [ ] Done

**Explain Bayes' Theorem and how it applies to the Naive Bayes algorithm.**

**Answer:**

Bayes' Theorem describes how to update the probability of a hypothesis given new evidence. It relates posterior probability to prior probability and likelihood. In Naive Bayes, it's used to calculate the probability of a class given observed features, enabling classification by selecting the class with the highest posterior.

**Core Concepts:**
- Prior P(Y): probability of class before seeing evidence
- Likelihood P(X|Y): probability of evidence given class
- Posterior P(Y|X): probability of class after seeing evidence
- Evidence P(X): total probability of observing the features

**Mathematical Formulation:**

$$P(Y|X) = \frac{P(X|Y) \cdot P(Y)}{P(X)}$$

Where:
- P(Y|X) = Posterior (what we want)
- P(X|Y) = Likelihood (from training data)
- P(Y) = Prior (class frequency)
- P(X) = Evidence (normalizing constant)

**Intuition:**
Imagine testing for a rare disease. Bayes' Theorem tells us: even if the test is 99% accurate, a positive result might not mean you have the disease if the disease is very rare (low prior). The posterior depends on both test accuracy AND disease prevalence.

**Application in Naive Bayes:**
1. Prior P(Y) = count(class Y) / total samples
2. Likelihood P(X|Y) = product of individual feature probabilities (naive assumption)
3. Classification: pick class Y that maximizes P(Y) × P(X|Y)

**Practical Relevance:**
- Foundation of all Bayesian inference
- Enables probabilistic predictions with confidence scores
- Allows incorporating domain knowledge via priors
- Works naturally with incremental/online learning

---

## Question 3
- [ ] Done

**Can you list and describe the types of Naive Bayes classifiers?**

**Answer:**

There are three main types of Naive Bayes classifiers, each designed for different feature distributions: Gaussian (continuous), Multinomial (count-based), and Bernoulli (binary). The choice depends on the nature of your data - Gaussian for real-valued features, Multinomial for text/frequency data, and Bernoulli for binary presence/absence features.

**Types and Their Characteristics:**

| Type | Feature Type | Likelihood Distribution | Best Use Case |
|------|--------------|------------------------|---------------|
| Gaussian NB | Continuous | Normal distribution | Real-valued features |
| Multinomial NB | Discrete counts | Multinomial distribution | Text classification, word counts |
| Bernoulli NB | Binary (0/1) | Bernoulli distribution | Binary features, short texts |

**Mathematical Formulations:**

**1. Gaussian Naive Bayes:**
$$P(X_i|Y) = \frac{1}{\sqrt{2\pi\sigma_Y^2}} \exp\left(-\frac{(X_i - \mu_Y)^2}{2\sigma_Y^2}\right)$$

**2. Multinomial Naive Bayes:**
$$P(X_i|Y) = \frac{N_{Yi} + \alpha}{N_Y + \alpha \cdot n}$$
(where N_{Yi} = count of feature i in class Y)

**3. Bernoulli Naive Bayes:**
$$P(X|Y) = \prod_i P(X_i|Y)^{X_i} \cdot (1 - P(X_i|Y))^{(1-X_i)}$$

**When to Use Each:**
- **Gaussian**: Iris dataset, sensor readings, medical measurements
- **Multinomial**: Document classification, spam detection, topic modeling
- **Bernoulli**: Binary document representation, feature presence/absence

**Practical Relevance:**
- Multinomial is the go-to for NLP tasks
- Gaussian works well when features follow normal distribution
- Bernoulli is preferred when document length varies significantly

---

## Question 4
- [ ] Done

**What is the 'naive' assumption in the Naive Bayes classifier?**

**Answer:**

The "naive" assumption is that all features are conditionally independent given the class label. This means the presence or value of one feature doesn't affect another feature's probability when we know the class. While rarely true in reality, this simplification makes computation tractable and surprisingly still yields good classification results.

**Core Concepts:**
- Conditional Independence: P(X1, X2 | Y) = P(X1|Y) × P(X2|Y)
- Reduces parameter space exponentially
- Allows computing joint probability as product of marginals
- Called "naive" because real-world features are usually correlated

**Mathematical Formulation:**

Without naive assumption (intractable for high dimensions):
$$P(X_1, X_2, ..., X_n | Y)$$ requires exponential parameters

With naive assumption:
$$P(X_1, X_2, ..., X_n | Y) = \prod_{i=1}^{n} P(X_i|Y)$$

**Intuition:**
Consider spam detection with words "free" and "money". In reality, these words are correlated (both appear together in spam). But Naive Bayes assumes: knowing an email is spam, the probability of "free" appearing doesn't depend on whether "money" appears. It treats each word independently.

**Why It Still Works:**
- Classification only needs correct ranking of posteriors, not exact values
- Probability estimates may be poor, but decision boundary can still be good
- High-dimensional data often has weak feature interactions
- Errors in probability estimates may cancel out

**Interview Tip:**
Be ready to explain why Naive Bayes works despite this unrealistic assumption - the key insight is that classification accuracy doesn't require accurate probability estimation.

---

## Question 5
- [ ] Done

**How does the Naive Bayes classifier handle categorical and numerical features?**

**Answer:**

For categorical features, Naive Bayes estimates probability as frequency counts (how often each category appears per class). For numerical features, it typically assumes a probability distribution (usually Gaussian) and estimates parameters (mean, variance) from training data. Different variants handle each type: Multinomial/Bernoulli for categorical, Gaussian for numerical.

**Handling Categorical Features:**
- Count frequency of each category per class
- Apply Laplace smoothing to avoid zero probabilities
- P(feature=value | class) = (count + α) / (total + α×k)

**Handling Numerical Features:**
- Assume distribution (typically Gaussian)
- Estimate mean (μ) and variance (σ²) per class
- Use probability density function for likelihood

**Mathematical Formulation:**

**Categorical (with Laplace smoothing):**
$$P(X_i = v | Y = c) = \frac{\text{count}(X_i = v, Y = c) + \alpha}{\text{count}(Y = c) + \alpha \cdot |V|}$$

**Numerical (Gaussian):**
$$P(X_i | Y = c) = \frac{1}{\sqrt{2\pi\sigma_c^2}} \exp\left(-\frac{(X_i - \mu_c)^2}{2\sigma_c^2}\right)$$

**Practical Approaches for Mixed Data:**
1. Use separate NB variants and combine
2. Discretize numerical features into bins
3. Use mixed Naive Bayes implementations
4. One-hot encode categorical, use Gaussian NB

**Interview Tip:**
When asked about mixed feature types, mention that scikit-learn's `CategoricalNB` handles categorical features, while `GaussianNB` handles continuous. For mixed data, consider discretization or building composite models.

---

## Question 6
- [ ] Done

**Explain the concept of 'class conditional independence' in Naive Bayes.**

**Answer:**

Class conditional independence means that features are independent of each other when conditioned on (given) the class label. Once we know the class, knowing one feature value provides no information about another feature's value. This is different from marginal independence - features can be correlated overall but independent within each class.

**Core Concepts:**
- Independence is conditional on class, not unconditional
- P(X1, X2 | Y) = P(X1|Y) × P(X2|Y)
- Features may be correlated marginally but conditionally independent
- Reduces joint probability computation to product of marginals

**Mathematical Formulation:**

Conditional Independence Definition:
$$P(X_1 \perp X_2 | Y) \iff P(X_1, X_2 | Y) = P(X_1|Y) \cdot P(X_2|Y)$$

Full Naive Bayes assumption:
$$P(X_1, X_2, ..., X_n | Y) = \prod_{i=1}^{n} P(X_i|Y)$$

**Intuition:**
Consider height and weight for classifying gender. Overall, height and weight are correlated (taller people tend to weigh more). But within males only, height and weight might be less correlated. Class conditional independence says: given we know the gender, height tells us nothing extra about weight.

**Visual Understanding:**
```
Marginal: Height ←→ Weight (correlated)
Conditional on Gender:
  - Given Male: Height ⊥ Weight
  - Given Female: Height ⊥ Weight
```

**Practical Relevance:**
- Allows tractable computation in high dimensions
- Key assumption that enables Naive Bayes simplicity
- Violation degrades probability estimates but often not classification

---

## Question 7
- [ ] Done

**What are the advantages and disadvantages of using a Naive Bayes classifier?**

**Answer:**

Naive Bayes excels in simplicity, speed, and performance with small data and high dimensions (especially text). Its main drawback is the unrealistic independence assumption which leads to poor probability estimates, and it struggles when features are highly correlated. It's an excellent baseline and often the first choice for text classification.

**Advantages:**

| Advantage | Explanation |
|-----------|-------------|
| Fast training & prediction | O(n×d) training, O(d×k) prediction |
| Works with small datasets | Needs less data than discriminative models |
| Handles high dimensions | Works well even when d >> n |
| No hyperparameter tuning | Only smoothing parameter α |
| Naturally handles multi-class | Direct multi-class, no one-vs-rest needed |
| Probabilistic output | Gives class probabilities, not just labels |
| Robust to irrelevant features | Irrelevant features contribute equally to all classes |
| Incremental learning | Easy to update with new data |

**Disadvantages:**

| Disadvantage | Explanation |
|--------------|-------------|
| Independence assumption | Rarely holds in practice |
| Poor probability estimates | Probabilities are often extreme (near 0 or 1) |
| Correlated features | Performance degrades with feature correlation |
| Zero frequency problem | Requires smoothing for unseen combinations |
| Continuous features | Assumes Gaussian, may not fit actual distribution |
| Cannot learn feature interactions | Treats all features independently |

**When to Use:**
- Text classification (spam, sentiment, topic)
- When training data is limited
- When interpretability matters
- As a strong baseline before complex models

**When to Avoid:**
- Features are highly correlated
- Accurate probability estimates needed
- Feature interactions are important

---

## Question 8
- [ ] Done

**How does the Multinomial Naive Bayes classifier differ from the Gaussian Naive Bayes classifier?**

**Answer:**

Multinomial NB models feature counts/frequencies using multinomial distribution - ideal for text data with word counts. Gaussian NB assumes features follow normal distribution and uses mean/variance - suitable for continuous real-valued features. The core difference is the likelihood function: count-based probability vs. Gaussian probability density.

**Key Differences:**

| Aspect | Multinomial NB | Gaussian NB |
|--------|---------------|-------------|
| Feature type | Discrete counts | Continuous values |
| Distribution | Multinomial | Gaussian (Normal) |
| Parameters | Word frequencies | Mean (μ), Variance (σ²) |
| Use case | Text, document classification | Numeric features |
| Input format | Term frequencies, TF-IDF | Real-valued vectors |

**Mathematical Formulation:**

**Multinomial NB:**
$$P(X_i = k | Y = c) = \frac{N_{ck} + \alpha}{\sum_j N_{cj} + \alpha \cdot |V|}$$
- N_{ck} = count of feature k in class c
- |V| = vocabulary size

**Gaussian NB:**
$$P(X_i | Y = c) = \frac{1}{\sqrt{2\pi\sigma_c^2}} \exp\left(-\frac{(X_i - \mu_c)^2}{2\sigma_c^2}\right)$$
- μ_c = mean of feature for class c
- σ_c² = variance of feature for class c

**Intuition:**
- **Multinomial**: "Given this is spam, how often does 'free' appear?"
- **Gaussian**: "Given this is class A, what's the probability of height=175cm?"

**Practical Examples:**
- Multinomial: Email spam detection, news categorization
- Gaussian: Iris flower classification, medical diagnosis with continuous vitals

---

## Question 9
- [ ] Done

**Explain how a Naive Bayes classifier can be used for spam detection.**

**Answer:**

For spam detection, Naive Bayes learns the probability of each word appearing in spam vs. ham (not spam) emails. Given a new email, it calculates the posterior probability of being spam by multiplying prior spam probability with likelihoods of each word. The email is classified as spam if P(spam|words) > P(ham|words).

**Pipeline Steps:**

1. **Preprocessing**: Tokenize emails, lowercase, remove stop words, stemming
2. **Feature Extraction**: Convert to word frequency vectors (Bag of Words or TF-IDF)
3. **Training**: Calculate P(spam), P(ham), and P(word|spam), P(word|ham) for each word
4. **Prediction**: For new email, compute posteriors and compare

**Mathematical Formulation:**

$$P(spam|email) \propto P(spam) \cdot \prod_{w \in email} P(w|spam)$$

$$P(ham|email) \propto P(ham) \cdot \prod_{w \in email} P(w|ham)$$

Classify as spam if: P(spam|email) > P(ham|email)

**Example Calculation:**
```
Training data shows:
- P(spam) = 0.4, P(ham) = 0.6
- P("free"|spam) = 0.8, P("free"|ham) = 0.1
- P("meeting"|spam) = 0.1, P("meeting"|ham) = 0.6

Email: "free meeting"
P(spam|email) ∝ 0.4 × 0.8 × 0.1 = 0.032
P(ham|email) ∝ 0.6 × 0.1 × 0.6 = 0.036

Prediction: Ham (not spam)
```

**Why Naive Bayes Works Well for Spam:**
- High-dimensional sparse data (large vocabulary)
- Word independence assumption is "good enough"
- Fast training on large email datasets
- Easily updatable with new spam patterns

---

## Question 10
- [ ] Done

**How does Naive Bayes perform in terms of model interpretability compared to other classifiers?**

**Answer:**

Naive Bayes is highly interpretable because predictions can be traced to individual feature contributions through their likelihood ratios. You can directly see which features (words, attributes) push classification toward each class. This transparency makes it more interpretable than neural networks and ensemble methods, comparable to logistic regression and decision trees.

**Interpretability Features:**

| Aspect | Naive Bayes Interpretability |
|--------|------------------------------|
| Feature importance | P(feature\|class) directly shows influence |
| Prediction explanation | Product of individual probabilities |
| Class reasoning | Can show top contributing features |
| Model inspection | Parameters are human-readable probabilities |

**How to Interpret Predictions:**

1. **Likelihood Ratio**: For each feature, compute P(feature|class1)/P(feature|class2)
2. **Log-odds contribution**: Each feature adds log(P(xi|Y)) to the score
3. **Top features**: Sort features by likelihood ratio to find most discriminative

**Comparison with Other Models:**

| Model | Interpretability | Reasoning |
|-------|-----------------|-----------|
| Naive Bayes | High | Direct probability contributions |
| Logistic Regression | High | Coefficients show feature weights |
| Decision Trees | High | Clear decision rules |
| Random Forest | Medium | Feature importance available |
| SVM | Low | Support vectors hard to interpret |
| Neural Networks | Low | Black box representations |

**Practical Example:**
For spam classification, you can explain: "This email is spam because P('free'|spam)=0.8 is much higher than P('free'|ham)=0.1, contributing strongly to spam classification."

**Interview Tip:**
Emphasize that Naive Bayes gives probability scores per feature, enabling feature-level explanations - valuable in regulated industries (finance, healthcare).

---

## Question 11
- [ ] Done

**Explain how feature selection affects the performance of a Naive Bayes model.**

**Answer:**

Feature selection can significantly improve Naive Bayes by removing irrelevant and redundant features. While NB is relatively robust to irrelevant features (they contribute equally to all classes), correlated/redundant features violate the independence assumption more severely. Proper feature selection reduces dimensionality, improves accuracy, and speeds up training/inference.

**Effects of Feature Selection:**

| Effect | Impact on Naive Bayes |
|--------|----------------------|
| Remove irrelevant features | Slight improvement (reduces noise) |
| Remove redundant features | Significant improvement (less independence violation) |
| Reduce dimensionality | Faster training and prediction |
| Reduce overfitting | Better generalization |

**Feature Selection Methods for NB:**

1. **Filter Methods:**
   - Chi-square test (χ²): Measures feature-class dependence
   - Mutual Information: Information gain from feature
   - Correlation-based selection

2. **Wrapper Methods:**
   - Forward/Backward selection with NB as evaluator
   - Recursive Feature Elimination

3. **Embedded Methods:**
   - Less common for NB (no built-in regularization)

**Mathematical Intuition:**

For irrelevant feature X_irrelevant:
$$P(X_{irrelevant}|Y=c_1) \approx P(X_{irrelevant}|Y=c_2)$$

This contributes ~1 to the likelihood ratio, not affecting classification but adding computation.

**Best Practices:**
- Use Chi-square or Mutual Information for text classification
- Remove features with very low document frequency
- Remove highly correlated feature pairs (keep one)
- Use cross-validation to validate feature subset

**Interview Tip:**
Mention that unlike some models, NB doesn't automatically penalize irrelevant features, so explicit feature selection is valuable.

---

## Question 12
- [ ] Done

**Describe how you would perform parameter tuning for Naive Bayes models.**

**Answer:**

Naive Bayes has minimal hyperparameters compared to other models. The primary parameter is the smoothing factor (alpha/Laplace parameter). For Gaussian NB, you might tune variance smoothing. Tuning is done via cross-validation to find optimal alpha that balances between zero-probability handling and over-smoothing.

**Key Parameters to Tune:**

| Parameter | Model | Range | Effect |
|-----------|-------|-------|--------|
| alpha (smoothing) | Multinomial, Bernoulli | 0.0 to 1.0+ | Prevents zero probabilities |
| var_smoothing | Gaussian NB | 1e-9 to 1e-1 | Stabilizes variance estimates |
| fit_prior | All | True/False | Whether to learn priors or use uniform |
| class_prior | All | Array of priors | Manual prior specification |

**Tuning Process:**

```python
# Pipeline for tuning
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

param_grid = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
    'fit_prior': [True, False]
}

grid_search = GridSearchCV(
    MultinomialNB(),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)
best_alpha = grid_search.best_params_['alpha']
```

**Alpha Selection Guidelines:**
- alpha = 0: No smoothing (risk of zero probabilities)
- alpha = 1: Laplace smoothing (add-one, standard choice)
- alpha < 1: Lidstone smoothing (less aggressive)
- alpha > 1: Heavy smoothing (when vocabulary is very large)

**Practical Tips:**
- Start with alpha=1.0 as baseline
- For large vocabularies, try smaller alpha (0.01-0.1)
- Use stratified k-fold cross-validation
- Monitor both accuracy and probability calibration

---

## Question 13
- [ ] Done

**How does Naive Bayes handle irrelevant features in a dataset?**

**Answer:**

Naive Bayes is relatively robust to irrelevant features. An irrelevant feature has similar probability distribution across all classes, so P(feature|class1) ≈ P(feature|class2). This means the feature contributes approximately equal likelihood to all classes, effectively not influencing the final classification decision. However, many irrelevant features can still add noise.

**Mathematical Explanation:**

For irrelevant feature X_irr:
$$\frac{P(X_{irr}|Y=c_1)}{P(X_{irr}|Y=c_2)} \approx 1$$

In log-space:
$$\log P(X_{irr}|Y=c_1) - \log P(X_{irr}|Y=c_2) \approx 0$$

The irrelevant feature adds ~0 to the log-odds, not changing the classification.

**Comparison with Other Models:**

| Model | Impact of Irrelevant Features |
|-------|------------------------------|
| Naive Bayes | Minimal (adds noise, not bias) |
| Decision Trees | Can split on irrelevant features |
| KNN | Severely affected (distance distorted) |
| SVM | Moderate impact (margin affected) |
| Linear Regression | Can cause multicollinearity |

**Why Still Consider Feature Selection:**
1. Computation: More features = slower prediction
2. Noise accumulation: Many irrelevant features add cumulative noise
3. Data requirements: More features need more training data
4. Correlated irrelevant features: These do harm the independence assumption

**Best Practice:**
While NB tolerates irrelevant features better than many algorithms, removing clearly irrelevant features (very low variance, zero correlation with target) still improves performance and reduces complexity.

---

## Question 14
- [ ] Done

**Explain the Bernoulli Naive Bayes classifier and in what context it is useful.**

**Answer:**

Bernoulli Naive Bayes models binary feature occurrence (present/absent, 0/1). Unlike Multinomial NB which considers word counts, Bernoulli NB only cares whether a feature appears or not. It explicitly models both presence and absence of features. It's particularly useful for short text classification and binary feature vectors.

**Core Characteristics:**
- Features are binary (0 or 1)
- Models P(feature present | class) AND P(feature absent | class)
- Explicitly penalizes absence of important features
- Works with binary document vectors

**Mathematical Formulation:**

$$P(X|Y=c) = \prod_{i=1}^{n} P(X_i|Y=c)^{X_i} \cdot (1 - P(X_i|Y=c))^{(1-X_i)}$$

Where:
- X_i = 1 if feature present, 0 if absent
- P(X_i=1|Y=c) = probability of feature i appearing in class c

**Difference from Multinomial NB:**

| Aspect | Bernoulli NB | Multinomial NB |
|--------|--------------|----------------|
| Feature type | Binary (0/1) | Counts (0,1,2,...) |
| Considers | Presence AND absence | Only presence |
| "money" appears 5 times | Same as appearing once | Counts all 5 |
| Word not appearing | Contributes to likelihood | Ignored |

**When to Use Bernoulli NB:**
- Short texts (tweets, SMS, headlines)
- Document length varies significantly
- Binary feature vectors (has/doesn't have feature)
- When absence of features is informative

**Example Use Case:**
For sentiment analysis of tweets: "not" being absent in a positive tweet is informative - Bernoulli NB captures this, Multinomial NB doesn't.

---

## Question 15
- [ ] Done

**How does Naive Bayes deal with continuous data, and what are the challenges?**

**Answer:**

Naive Bayes handles continuous data by assuming a probability distribution for each feature (typically Gaussian). It estimates distribution parameters (mean, variance) from training data and uses the probability density function to compute likelihoods. The main challenge is that real data often doesn't follow assumed distributions, leading to poor probability estimates.

**Approaches for Continuous Features:**

| Approach | Description | Pros/Cons |
|----------|-------------|-----------|
| Gaussian NB | Assume normal distribution | Simple, may not fit data |
| Discretization | Bin continuous values | Works with any NB variant |
| Kernel Density | Non-parametric estimation | Flexible, computationally expensive |

**Gaussian Naive Bayes Formula:**

$$P(X_i|Y=c) = \frac{1}{\sqrt{2\pi\sigma_c^2}} \exp\left(-\frac{(X_i - \mu_c)^2}{2\sigma_c^2}\right)$$

**Challenges:**

1. **Distribution Mismatch:**
   - Real data may be skewed, bimodal, or heavy-tailed
   - Gaussian assumption leads to poor likelihood estimates

2. **Outliers:**
   - Outliers distort mean and variance estimates
   - Can cause very low/zero probabilities

3. **Small Sample Size:**
   - Variance estimates unreliable with few samples
   - Need variance smoothing

4. **Multi-modal Features:**
   - Single Gaussian can't model multiple peaks
   - Consider Gaussian Mixture or discretization

**Solutions:**

```python
# Discretization approach
from sklearn.preprocessing import KBinsDiscretizer
discretizer = KBinsDiscretizer(n_bins=5, strategy='quantile')
X_discrete = discretizer.fit_transform(X_continuous)
# Now use MultinomialNB or CategoricalNB
```

**Best Practices:**
- Visualize feature distributions before choosing approach
- Consider log-transform for skewed features
- Use variance smoothing in sklearn (var_smoothing parameter)
- Discretization often works better than Gaussian assumption

---

## Question 16
- [ ] Done

**Can Naive Bayes be used with kernel methods? If yes, explain how.**

**Answer:**

Yes, Kernel Naive Bayes uses kernel density estimation (KDE) instead of parametric distributions to estimate P(X|Y). Instead of assuming Gaussian distribution, KDE non-parametrically estimates the probability density from data points. This allows Naive Bayes to handle continuous features with arbitrary distributions without distribution assumptions.

**How Kernel NB Works:**

1. For each class, use KDE to estimate feature density
2. Place a kernel (usually Gaussian) at each training point
3. Sum kernels to get smooth density estimate
4. Use this density for likelihood computation

**Mathematical Formulation:**

$$P(X_i|Y=c) = \frac{1}{n_c \cdot h} \sum_{j=1}^{n_c} K\left(\frac{X_i - X_{ij}}{h}\right)$$

Where:
- K = kernel function (commonly Gaussian)
- h = bandwidth (smoothing parameter)
- n_c = number of samples in class c
- X_{ij} = j-th training sample of feature i in class c

**Common Kernels:**
- Gaussian: $K(u) = \frac{1}{\sqrt{2\pi}} e^{-u^2/2}$
- Epanechnikov: $K(u) = \frac{3}{4}(1-u^2)$ for |u| ≤ 1

**Advantages:**
- No distribution assumption needed
- Handles multi-modal, skewed distributions
- More accurate density estimation

**Disadvantages:**
- Computationally expensive (O(n) per prediction)
- Bandwidth selection is crucial
- Needs more training data

**Python Implementation:**
```python
from sklearn.neighbors import KernelDensity
# Fit KDE for each class and feature
kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
kde.fit(X_class_feature.reshape(-1, 1))
log_likelihood = kde.score_samples(X_test)
```

---

## Question 17
- [ ] Done

**Describe a practical application of Naive Bayes in medical diagnosis.**

**Answer:**

In medical diagnosis, Naive Bayes predicts disease probability given patient symptoms and test results. It calculates P(disease|symptoms) by combining disease prevalence (prior) with symptom likelihoods for each condition. It's valuable because it handles multiple symptoms, provides probability scores for informed decisions, and works well with limited medical data.

**Application Pipeline:**

1. **Training Data**: Patient records with symptoms and diagnoses
2. **Features**: Symptoms, test results, patient demographics
3. **Target**: Disease/condition (can be multi-class)
4. **Output**: Probability of each possible condition

**Example: Heart Disease Prediction**

Features:
- Age, gender, blood pressure, cholesterol level
- Symptoms: chest pain, shortness of breath
- Test results: ECG readings, stress test

**Mathematical Application:**

$$P(heart\_disease|symptoms) = \frac{P(symptoms|heart\_disease) \cdot P(heart\_disease)}{P(symptoms)}$$

**Why Naive Bayes Works in Medicine:**
- Handles missing data well (ignore missing features)
- Interpretable predictions (can explain which symptoms contributed)
- Works with small datasets common in medical research
- Probabilistic output helps doctors assess risk levels
- Can incorporate prior medical knowledge

**Real-World Examples:**
- Cancer screening (predicting malignancy from biopsy features)
- Disease outbreak prediction
- Drug response prediction
- Diagnostic decision support systems

**Limitations to Consider:**
- Independence assumption may not hold (symptoms often correlated)
- Probability calibration important for medical decisions
- Should be used as decision support, not replacement for doctors

---

## Question 18
- [ ] Done

**Explain how you would apply Naive Bayes to customer sentiment analysis from product reviews.**

**Answer:**

For sentiment analysis, Naive Bayes learns word patterns associated with positive vs. negative reviews. It tokenizes reviews into words/n-grams, calculates P(word|positive) and P(word|negative), then classifies new reviews by computing which sentiment has higher posterior probability. Multinomial NB with TF-IDF features is the standard approach.

**Implementation Pipeline:**

```
1. Data Collection → Reviews with sentiment labels
2. Preprocessing → Clean text, tokenize, remove stopwords
3. Feature Extraction → Bag of Words or TF-IDF vectors
4. Train NB → Learn word-sentiment probabilities
5. Predict → Classify new reviews as positive/negative
```

**Step-by-Step Process:**

1. **Preprocessing:**
   - Lowercase, remove punctuation
   - Handle negations ("not good" → "not_good")
   - Remove stopwords (optional)
   - Stemming/Lemmatization

2. **Feature Engineering:**
   - Unigrams and bigrams (n-grams)
   - TF-IDF weighting
   - Sentiment lexicon features (optional)

3. **Model Training:**
   - Calculate P(positive) and P(negative) from data
   - Calculate P(word|sentiment) for each word

**Code Example:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
    ('nb', MultinomialNB(alpha=0.1))
])
sentiment_pipeline.fit(reviews, labels)
predictions = sentiment_pipeline.predict(new_reviews)
```

**Challenges and Solutions:**
- Negation: Use bigrams or mark negation scope
- Sarcasm: Difficult for NB, may need additional features
- Domain-specific words: Use domain corpus for training

---

## Question 19
- [ ] Done

**What are the recent advancements in Naive Bayes for handling big data?**

**Answer:**

Key advancements include distributed implementations (Spark MLlib), incremental/online learning for streaming data, feature hashing for memory efficiency, and mini-batch training. These allow Naive Bayes to scale to billions of samples and millions of features while maintaining its simplicity and computational efficiency advantages.

**Major Advancements:**

| Advancement | Description | Benefit |
|-------------|-------------|---------|
| Distributed NB | MapReduce/Spark implementations | Scales horizontally |
| Online Learning | Incremental updates with partial_fit | Handles streaming data |
| Feature Hashing | HashingVectorizer reduces memory | Millions of features |
| Mini-batch Training | Update on batches, not full data | Memory efficient |

**Distributed Naive Bayes (MapReduce):**

```
Map Phase: For each document
  - Emit (class, feature, count) tuples
  
Reduce Phase:
  - Aggregate counts per class and feature
  - Compute final probabilities
```

**Online Learning (Streaming Data):**
```python
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
for batch_X, batch_y in data_stream:
    nb.partial_fit(batch_X, batch_y, classes=[0, 1])
```

**Feature Hashing for Large Vocabularies:**
```python
from sklearn.feature_extraction.text import HashingVectorizer

hasher = HashingVectorizer(n_features=2**18)  # Fixed memory
X = hasher.transform(documents)  # No vocabulary storage
```

**Spark MLlib Implementation:**
- Parallelizes count aggregation
- Handles terabyte-scale text data
- Native support for sparse vectors

**Why NB Scales Well:**
- Training is just counting (parallelizable)
- No iterative optimization like gradient descent
- Model size proportional to features × classes, not data size
- Prediction is O(features) regardless of training size

---

## Question 20
- [ ] Done

**How does the concept of distributional semantics enhance Naive Bayes text classification?**

**Answer:**

Distributional semantics enriches Naive Bayes by using word embeddings (Word2Vec, GloVe) to capture semantic similarity between words. Instead of treating each word independently, semantically similar words share information. This addresses vocabulary mismatch where different words express similar meanings, improving generalization especially with limited training data.

**Core Concept:**
- Traditional NB: Each word is independent, "good" ≠ "excellent"
- With distributional semantics: "good" and "excellent" are similar vectors
- Can transfer learning between semantically similar words

**Enhancement Approaches:**

1. **Document Embedding Input:**
   - Average word vectors to get document embedding
   - Use Gaussian NB on document vectors

2. **Smoothed Word Probabilities:**
   - Share probability mass between similar words
   - If "excellent" unseen, borrow from similar words like "great"

3. **Semantic Feature Expansion:**
   - Expand features to include similar words
   - "good" feature activates related semantic concepts

**Mathematical Enhancement:**

Traditional:
$$P(w|class) = \frac{count(w, class) + \alpha}{count(class) + \alpha|V|}$$

With semantic smoothing:
$$P(w|class) = \sum_{w' \in vocab} sim(w, w') \cdot P(w'|class)$$

Where sim(w, w') = cosine similarity of word embeddings

**Practical Implementation:**
```python
# Document embedding approach
import numpy as np
from gensim.models import Word2Vec

def doc_to_vector(doc, model):
    vectors = [model.wv[w] for w in doc if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

# Use GaussianNB on document vectors
```

**Benefits:**
- Better handling of synonyms and related terms
- Improved performance on small datasets
- Addresses out-of-vocabulary words

---

## Question 21
- [ ] Done

**What are the mathematical foundations and derivation of Naive Bayes classification?**

**Answer:**

Naive Bayes derives from Bayes' theorem combined with the conditional independence assumption. Starting from the goal of finding the most probable class, we apply Bayes' theorem, then simplify the joint likelihood using the naive assumption. The result is a classifier that computes class scores as products of individual feature probabilities.

**Derivation Steps:**

**Step 1: Classification Goal**
$$\hat{Y} = \arg\max_Y P(Y|X_1, X_2, ..., X_n)$$

**Step 2: Apply Bayes' Theorem**
$$P(Y|X) = \frac{P(X|Y) \cdot P(Y)}{P(X)}$$

Since P(X) is constant for all classes:
$$\hat{Y} = \arg\max_Y P(X_1, X_2, ..., X_n|Y) \cdot P(Y)$$

**Step 3: Apply Naive (Independence) Assumption**
$$P(X_1, X_2, ..., X_n|Y) = \prod_{i=1}^{n} P(X_i|Y)$$

**Step 4: Final Classification Rule**
$$\hat{Y} = \arg\max_Y P(Y) \cdot \prod_{i=1}^{n} P(X_i|Y)$$

**Step 5: Log Transform (Numerical Stability)**
$$\hat{Y} = \arg\max_Y \left[\log P(Y) + \sum_{i=1}^{n} \log P(X_i|Y)\right]$$

**Parameter Estimation (Maximum Likelihood):**

**Prior:**
$$P(Y=c) = \frac{|D_c|}{|D|}$$

**Likelihood (Categorical):**
$$P(X_i=v|Y=c) = \frac{count(X_i=v, Y=c) + \alpha}{count(Y=c) + \alpha|V_i|}$$

**Likelihood (Gaussian):**
$$\mu_c = \frac{1}{|D_c|}\sum_{x \in D_c} x, \quad \sigma_c^2 = \frac{1}{|D_c|}\sum_{x \in D_c}(x-\mu_c)^2$$

**Key Mathematical Properties:**
- Linear classifier in log-space
- Decision boundary is linear in log-likelihood ratios
- Maximum A Posteriori (MAP) estimation

---

## Question 22
- [ ] Done

**How do you handle the zero probability problem in Naive Bayes?**

**Answer:**

The zero probability problem occurs when a feature value never appears with a class in training data, making P(feature|class) = 0. Since Naive Bayes multiplies probabilities, one zero makes the entire product zero, incorrectly predicting zero probability. The solution is smoothing - adding a small count to all feature-class combinations.

**The Problem:**

If word "amazing" never appears in negative reviews:
$$P(amazing|negative) = 0$$

For new review containing "amazing":
$$P(negative|review) \propto P(negative) \times 0 \times ... = 0$$

Even if all other words suggest negative, prediction fails.

**Solutions:**

**1. Laplace Smoothing (Add-1):**
$$P(X_i=v|Y=c) = \frac{count(X_i=v, Y=c) + 1}{count(Y=c) + |V|}$$

**2. Lidstone Smoothing (Add-α):**
$$P(X_i=v|Y=c) = \frac{count(X_i=v, Y=c) + \alpha}{count(Y=c) + \alpha|V|}$$
- α < 1 gives less weight to unseen features
- α = 1 is Laplace smoothing

**3. Good-Turing Smoothing:**
- Estimates probability of unseen events from rare events
- More sophisticated but complex

**Implementation:**
```python
from sklearn.naive_bayes import MultinomialNB

# alpha parameter controls smoothing
nb = MultinomialNB(alpha=1.0)  # Laplace smoothing
nb = MultinomialNB(alpha=0.1)  # Lighter smoothing for large vocabularies
```

**Best Practices:**
- Start with α=1.0 (Laplace), adjust if needed
- For large vocabularies, smaller α (0.01-0.1) often works better
- Use cross-validation to tune α

---

## Question 23
- [ ] Done

**What is Laplace smoothing and how does it work in Naive Bayes?**

**Answer:**

Laplace smoothing (add-one smoothing) adds 1 to every feature count before calculating probabilities, ensuring no probability is ever zero. It prevents the zero-frequency problem where unseen feature-class combinations would give zero probability. The denominator is adjusted by adding the vocabulary size to maintain valid probability distribution.

**Formula:**

Without smoothing:
$$P(X_i=v|Y=c) = \frac{count(X_i=v, Y=c)}{count(Y=c)}$$

With Laplace smoothing:
$$P(X_i=v|Y=c) = \frac{count(X_i=v, Y=c) + 1}{count(Y=c) + |V|}$$

Where |V| = number of possible values for feature Xi

**Example:**

Training data has 100 spam emails, word "lottery" appears 20 times:
- Without smoothing: P("lottery"|spam) = 20/100 = 0.20

Word "unicorn" never appears in spam (0 times), vocabulary size = 10,000:
- Without smoothing: P("unicorn"|spam) = 0/100 = 0 ❌
- With Laplace: P("unicorn"|spam) = (0+1)/(100+10000) = 0.0001 ✓

**Why Denominator Changes:**

Adding |V| ensures probabilities sum to 1:
$$\sum_{v \in V} P(X_i=v|Y=c) = \sum_{v \in V} \frac{count(v,c) + 1}{count(c) + |V|} = \frac{count(c) + |V|}{count(c) + |V|} = 1$$

**Generalization - Lidstone Smoothing:**
$$P(X_i=v|Y=c) = \frac{count(X_i=v, Y=c) + \alpha}{count(Y=c) + \alpha|V|}$$

- α = 1: Laplace smoothing
- α = 0.5: Jeffreys smoothing
- α < 1: Less aggressive smoothing for large vocabularies

---

## Question 24
- [ ] Done

**How do you implement Gaussian Naive Bayes for continuous features?**

**Answer:**

Gaussian NB assumes continuous features follow normal distribution within each class. Implementation involves: (1) calculating mean and variance for each feature per class from training data, (2) using Gaussian PDF to compute likelihood P(feature|class) for new instances, (3) combining with priors to get posteriors.

**Algorithm Steps:**

1. For each class c, compute:
   - Prior: P(c) = count(class=c) / total_samples
   - For each feature i: mean μ_ic, variance σ²_ic

2. For prediction, compute:
   - P(x|c) = ∏ Gaussian(xi; μic, σ²ic)
   - Predict class with maximum P(c) × P(x|c)

**Mathematical Formulation:**

$$P(X_i|Y=c) = \frac{1}{\sqrt{2\pi\sigma_{ic}^2}} \exp\left(-\frac{(X_i - \mu_{ic})^2}{2\sigma_{ic}^2}\right)$$

**Python Implementation from Scratch:**

```python
import numpy as np

class GaussianNBFromScratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.params = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.params[c] = {
                'prior': len(X_c) / len(X),
                'mean': X_c.mean(axis=0),
                'var': X_c.var(axis=0) + 1e-9  # Smoothing
            }
        return self
    
    def _gaussian_pdf(self, x, mean, var):
        return np.exp(-0.5 * ((x - mean)**2 / var)) / np.sqrt(2 * np.pi * var)
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.params[c]['prior'])
                likelihood = np.sum(np.log(
                    self._gaussian_pdf(x, self.params[c]['mean'], self.params[c]['var'])
                ))
                posteriors.append(prior + likelihood)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)
```

**Using Scikit-learn:**
```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB(var_smoothing=1e-9)
gnb.fit(X_train, y_train)
predictions = gnb.predict(X_test)
```

---

## Question 25
- [ ] Done

**What is Multinomial Naive Bayes and when should you use it?**

**Answer:**

Multinomial NB models feature counts using multinomial distribution - it's designed for discrete count data like word frequencies in documents. Each feature represents how many times something occurs (word count, click count). It's the standard choice for text classification where features are term frequencies or TF-IDF values.

**When to Use Multinomial NB:**

| Use Case | Why Multinomial NB |
|----------|-------------------|
| Text classification | Word counts are discrete |
| Document categorization | TF-IDF features work well |
| Spam detection | Word frequency patterns |
| Topic modeling | Term distributions |

**Mathematical Formulation:**

$$P(X|Y=c) = \frac{(\sum_i X_i)!}{\prod_i X_i!} \prod_i P(w_i|c)^{X_i}$$

Simplified (ignoring constant):
$$P(X|Y=c) \propto \prod_i P(w_i|c)^{X_i}$$

Where:
- Xi = count of feature i
- P(wi|c) = probability of feature i in class c

**Parameter Estimation:**
$$P(w_i|c) = \frac{count(w_i, c) + \alpha}{\sum_j count(w_j, c) + \alpha|V|}$$

**Python Implementation:**
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# With word counts
text_clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB(alpha=1.0))
])

# With TF-IDF (recommended)
text_clf = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB(alpha=0.1))
])

text_clf.fit(train_texts, train_labels)
```

**Key Points:**
- Features must be non-negative (counts)
- Works with sparse matrices efficiently
- TF-IDF normalized values work well despite being continuous
- Alpha parameter crucial for smoothing

---

## Question 26
- [ ] Done

**How does Bernoulli Naive Bayes differ from other variants?**

**Answer:**

Bernoulli NB models binary feature occurrence (1=present, 0=absent) rather than counts or continuous values. Unlike Multinomial which only considers present features, Bernoulli explicitly models feature absence, penalizing documents where expected features are missing. This makes it suitable for short texts and binary feature vectors.

**Key Differences:**

| Aspect | Bernoulli NB | Multinomial NB | Gaussian NB |
|--------|--------------|----------------|-------------|
| Feature type | Binary (0/1) | Counts | Continuous |
| Word "free" appears 5x | Treated same as 1x | Counts all 5 | N/A |
| Missing word | Contributes (1-P) | Ignored | N/A |
| Best for | Short texts, binary | Long documents | Numeric data |

**Mathematical Difference:**

**Bernoulli:**
$$P(X|Y=c) = \prod_i P(X_i|c)^{X_i} \cdot (1-P(X_i|c))^{(1-X_i)}$$

**Multinomial:**
$$P(X|Y=c) \propto \prod_i P(w_i|c)^{X_i}$$

**Example:**
Document: "The quick fox" (binary: quick=1, lazy=0)

Bernoulli: Uses both P(quick=1|class) AND P(lazy=0|class)
Multinomial: Only uses P(quick|class), ignores lazy

**When to Choose Bernoulli:**
- Short texts (tweets, SMS)
- Feature absence is informative
- Binary feature representation preferred
- Document length varies significantly

---

## Question 27
- [ ] Done

**What are the assumptions and limitations of the Naive Bayes classifier?**

**Answer:**

The core assumption is conditional independence - features are independent given the class. Limitations include poor probability calibration, inability to model feature interactions, sensitivity to correlated features, and assumption of feature distribution (Gaussian for continuous). Despite these, NB often achieves competitive classification accuracy.

**Assumptions:**

1. **Conditional Independence:**
   - P(X1,X2|Y) = P(X1|Y) × P(X2|Y)
   - Rarely true in practice

2. **Feature Distribution:**
   - Gaussian NB: Normal distribution
   - Multinomial NB: Multinomial distribution
   - Bernoulli NB: Bernoulli distribution

3. **Equal Feature Importance:**
   - All features contribute equally to likelihood
   - No weighting mechanism

**Limitations:**

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Independence violation | Poor probability estimates | Often doesn't hurt classification |
| Correlated features | Double-counting evidence | Feature selection |
| Poor calibration | Extreme probabilities (0 or 1) | Probability calibration |
| No feature interactions | Misses XOR-like patterns | Feature engineering |
| Distribution assumptions | Likelihood mismatch | Use appropriate variant |

**Why It Still Works:**
- Classification only needs correct ranking, not exact probabilities
- High-dimensional data tends to have weaker correlations
- Simple models can have lower variance, better generalization

---

## Question 28
- [ ] Done

**How do you handle missing values in Naive Bayes classification?**

**Answer:**

Naive Bayes can naturally handle missing values by simply ignoring the missing feature when computing likelihoods. Since NB treats features independently, a missing feature doesn't provide information, so we exclude it from the product. This is equivalent to marginalizing over all possible values of the missing feature.

**Approaches:**

**1. Ignore Missing Features (Recommended):**
```python
# For each sample, only multiply likelihoods of observed features
def predict_with_missing(self, x):
    posteriors = []
    for c in self.classes:
        log_prob = np.log(self.priors[c])
        for i, val in enumerate(x):
            if not np.isnan(val):  # Only include non-missing
                log_prob += np.log(self.likelihoods[c][i][val])
        posteriors.append(log_prob)
    return self.classes[np.argmax(posteriors)]
```

**2. Imputation Before Training:**
- Mean/median imputation for continuous features
- Mode imputation for categorical features
- May introduce bias

**3. Treat Missing as Separate Category:**
- For categorical features, "missing" becomes a category
- Learn P(missing|class) from data

**Mathematical Justification:**

With missing Xi:
$$P(Y|X_1,...,X_{i-1},X_{i+1},...,X_n) \propto P(Y) \prod_{j \neq i} P(X_j|Y)$$

This is equivalent to integrating over all values of Xi (marginalization).

**Scikit-learn Handling:**
- GaussianNB doesn't natively support NaN
- Impute before fitting or use custom implementation

---

## Question 29
- [ ] Done

**What is the role of prior probabilities in Naive Bayes?**

**Answer:**

Prior probability P(Y) represents our belief about class frequency before seeing any features. It incorporates base rate information - if spam is 30% of emails, we start expecting spam less. Priors prevent the model from ignoring class imbalance and can be set from data (learned) or domain knowledge (specified).

**Mathematical Role:**

$$P(Y|X) \propto P(Y) \cdot P(X|Y)$$

The prior P(Y) scales the likelihood, affecting the decision boundary.

**Impact of Priors:**

| Scenario | Prior Effect |
|----------|--------------|
| Balanced classes | Minimal impact (all ~0.5) |
| Imbalanced classes | Shifts predictions toward majority |
| Rare but important class | May miss minority class |
| Domain knowledge available | Can override data priors |

**Setting Priors:**

```python
from sklearn.naive_bayes import MultinomialNB

# Learn from data (default)
nb = MultinomialNB(fit_prior=True)

# Uniform priors (ignore class frequency)
nb = MultinomialNB(fit_prior=False)  # Uses 1/n_classes

# Custom priors (domain knowledge)
nb = MultinomialNB(class_prior=[0.3, 0.7])
```

**When to Adjust Priors:**
- Training data doesn't reflect real-world distribution
- Want to balance precision/recall trade-off
- Domain expertise suggests different rates
- Cost-sensitive classification

**Example:**
Disease diagnosis where disease prevalence is 1%:
- Without prior: Model might predict "sick" too often
- With prior P(sick)=0.01: Properly accounts for rarity

---

## Question 30
- [ ] Done

**How do you implement feature selection for Naive Bayes classifiers?**

**Answer:**

Feature selection for NB typically uses filter methods that measure feature-class dependency. Chi-square test and Mutual Information are most common - they rank features by how informative they are for classification. Remove low-scoring features to reduce noise, improve speed, and potentially improve accuracy.

**Common Methods:**

**1. Chi-Square (χ²) Test:**
$$\chi^2 = \sum \frac{(O - E)^2}{E}$$
- Measures independence between feature and class
- Higher score = more informative feature

**2. Mutual Information:**
$$MI(X,Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$$
- Information gained about Y from knowing X

**Implementation:**

```python
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Chi-square feature selection
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('selector', SelectKBest(chi2, k=1000)),  # Top 1000 features
    ('classifier', MultinomialNB())
])

# Mutual Information
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('selector', SelectKBest(mutual_info_classif, k=1000)),
    ('classifier', MultinomialNB())
])
```

**Selection Guidelines:**
- Start with all features, then reduce and compare
- Use cross-validation to find optimal k
- Chi-square is fast for sparse text data
- Consider document frequency filtering first (min_df, max_df)

---

## Question 31
- [ ] Done

**What are the computational complexity advantages of Naive Bayes?**

**Answer:**

Naive Bayes has O(n×d) training complexity (linear in samples and features) - just counting. Prediction is O(d×k) per sample (d features, k classes). No iterative optimization needed. This makes NB extremely fast for training and inference, especially valuable for high-dimensional text data and real-time applications.

**Complexity Analysis:**

| Operation | Naive Bayes | Logistic Regression | SVM |
|-----------|-------------|--------------------|----|
| Training | O(n × d) | O(n × d × iterations) | O(n² to n³) |
| Prediction | O(d × k) | O(d × k) | O(n_sv × d) |
| Memory | O(d × k) | O(d × k) | O(n_sv × d) |

Where: n=samples, d=features, k=classes, n_sv=support vectors

**Why NB is Fast:**

1. **Training:**
   - Single pass through data
   - Just counting (no optimization)
   - Parallelizable (counts independent)

2. **Prediction:**
   - Simple multiplication (or addition in log-space)
   - No kernel computations
   - No distance calculations

**Practical Implications:**
- Train on millions of documents in seconds
- Real-time prediction for streaming data
- Works on resource-constrained devices
- Excellent for prototyping and baselines

**Benchmark Example:**
```python
# Training time comparison (typical)
# Dataset: 100,000 documents, 50,000 features
# Naive Bayes: ~1 second
# Logistic Regression: ~30 seconds
# SVM (RBF): ~10 minutes
```

---

## Question 32
- [ ] Done

**How do you handle high-dimensional data with Naive Bayes?**

**Answer:**

Naive Bayes handles high-dimensional data naturally - it scales linearly with features and doesn't suffer from curse of dimensionality like distance-based methods. Strategies include: feature hashing for memory efficiency, sparse matrix representations, feature selection to remove noise, and dimensionality reduction while preserving discriminative features.

**Why NB Excels in High Dimensions:**
- No distance computation (immune to dimension curse)
- Linear complexity O(d) in features
- Works well even when d >> n (features > samples)
- Sparse feature support (only non-zero values matter)

**Strategies:**

**1. Feature Hashing (Memory Efficient):**
```python
from sklearn.feature_extraction.text import HashingVectorizer

hasher = HashingVectorizer(n_features=2**16)  # Fixed size
X = hasher.transform(documents)  # Sparse matrix
```

**2. Sparse Representations:**
```python
from scipy.sparse import csr_matrix
# NB works directly with sparse matrices
nb.fit(X_sparse, y)  # Memory efficient
```

**3. Feature Selection:**
```python
from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(chi2, k=5000)  # Keep top 5000
X_reduced = selector.fit_transform(X, y)
```

**4. Vocabulary Pruning:**
```python
vectorizer = TfidfVectorizer(
    max_features=10000,  # Limit vocabulary
    min_df=5,            # Ignore rare words
    max_df=0.95          # Ignore too common words
)
```

**Best Practices:**
- Use sparse matrices (scipy.sparse)
- Set min_df to remove rare features
- Consider feature hashing for very large vocabularies
- Feature selection often improves accuracy too

---

## Question 33
- [ ] Done

**What is the role of Naive Bayes in text classification and NLP?**

**Answer:**

Naive Bayes is a foundational algorithm for text classification due to its speed, simplicity, and strong performance on bag-of-words representations. It naturally handles high-dimensional sparse text data, requires minimal training data, and provides interpretable results. Common applications include spam detection, sentiment analysis, topic classification, and language detection.

**Why NB Excels for Text:**

| Property | Benefit for Text |
|----------|------------------|
| High-dimensional | Vocabulary size = 10K-100K features |
| Sparse data | Most words absent in each document |
| Limited data | Needs few examples per class |
| Fast training | Process millions of documents |
| Interpretable | Show top words per class |

**Common NLP Applications:**

1. **Spam Detection** - Word patterns in spam vs. legitimate
2. **Sentiment Analysis** - Positive/negative word frequencies
3. **Topic Classification** - News categorization
4. **Language Detection** - Character n-gram patterns
5. **Author Identification** - Writing style features

**Typical Pipeline:**
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

text_classifier = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_df=0.9)),
    ('nb', MultinomialNB(alpha=0.1))
])

text_classifier.fit(train_texts, train_labels)
```

**Modern Context:**
- Still used as strong baseline
- Outperforms complex models on small datasets
- Part of ensemble systems
- Preprocessing step for neural models

---

## Question 34
- [ ] Done

**How do you implement TF-IDF with Naive Bayes for text analysis?**

**Answer:**

TF-IDF (Term Frequency-Inverse Document Frequency) weights words by their importance - frequent in document but rare across corpus. Combined with Multinomial NB, it improves classification by down-weighting common words and emphasizing distinctive terms. Implementation uses TfidfVectorizer followed by MultinomialNB in a pipeline.

**TF-IDF Formula:**

$$TF\text{-}IDF(t,d) = TF(t,d) \times IDF(t)$$

Where:
- TF(t,d) = frequency of term t in document d
- IDF(t) = log(N / df(t)), N=total docs, df=docs containing t

**Implementation:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Sample data
texts = ["great movie loved it", "terrible waste of time", ...]
labels = [1, 0, ...]  # 1=positive, 0=negative

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )),
    ('nb', MultinomialNB(alpha=0.1))
])

# Train and predict
X_train, X_test, y_train, y_test = train_test_split(texts, labels)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
accuracy = pipeline.score(X_test, y_test)
```

**Key Parameters:**
- `max_features`: Limit vocabulary size
- `ngram_range`: Include bigrams (1,2)
- `min_df`: Ignore rare terms
- `max_df`: Ignore too common terms
- `alpha`: Smoothing parameter for NB

---

## Question 35
- [ ] Done

**What are n-gram features and their use in Naive Bayes text classification?**

**Answer:**

N-grams are contiguous sequences of n words. Unigrams (n=1) are individual words; bigrams (n=2) are word pairs; trigrams (n=3) are triplets. N-grams capture local word context and phrases that unigrams miss (e.g., "not good" as bigram captures negation). Using n-grams with NB improves classification by modeling word patterns.

**N-gram Examples:**

Text: "I am not happy"
- Unigrams: ["I", "am", "not", "happy"]
- Bigrams: ["I am", "am not", "not happy"]
- Trigrams: ["I am not", "am not happy"]

**Why N-grams Help:**
- Capture negation: "not good" ≠ "good"
- Capture phrases: "New York" as single unit
- Model word order: "dog bites man" vs "man bites dog"

**Implementation:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Unigrams only
vec_uni = TfidfVectorizer(ngram_range=(1, 1))

# Unigrams + Bigrams (common choice)
vec_bi = TfidfVectorizer(ngram_range=(1, 2))

# Up to trigrams
vec_tri = TfidfVectorizer(ngram_range=(1, 3))

# Example pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
    ('nb', MultinomialNB())
])
```

**Trade-offs:**

| N-gram Range | Vocabulary Size | Context Captured | Sparsity |
|--------------|----------------|------------------|----------|
| (1,1) | Small | None | Low |
| (1,2) | Medium | Good | Medium |
| (1,3) | Large | Better | High |

**Best Practice:**
- Start with (1,2) - good balance
- Use max_features to limit vocabulary explosion
- Character n-grams for typo robustness or language detection

---

## Question 36
- [ ] Done

**How do you handle class imbalance in Naive Bayes classification?**

**Answer:**

Class imbalance can bias NB toward majority class through priors. Solutions include: adjusting priors to balance classes, oversampling minority class (SMOTE), undersampling majority class, using class weights, or adjusting decision threshold. For NB specifically, setting uniform priors or custom priors is often effective.

**Strategies:**

**1. Adjust Priors:**
```python
from sklearn.naive_bayes import MultinomialNB

# Uniform priors (ignore class frequency)
nb = MultinomialNB(fit_prior=False)

# Balanced priors
nb = MultinomialNB(class_prior=[0.5, 0.5])  # Binary case
```

**2. Resampling:**
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Oversample minority
smote = SMOTE()
X_balanced, y_balanced = smote.fit_resample(X, y)

# Undersample majority
rus = RandomUnderSampler()
X_balanced, y_balanced = rus.fit_resample(X, y)
```

**3. Threshold Adjustment:**
```python
# Get probabilities instead of predictions
probs = nb.predict_proba(X_test)

# Adjust threshold (default 0.5)
threshold = 0.3  # Lower threshold for minority class
predictions = (probs[:, 1] > threshold).astype(int)
```

**4. Cost-Sensitive Classification:**
- Manually weight priors by misclassification cost
- P(minority) = P(minority) × cost_ratio

**Evaluation Note:**
Use appropriate metrics for imbalanced data:
- Precision, Recall, F1-score
- ROC-AUC, PR-AUC
- NOT just accuracy

---

## Question 37
- [ ] Done

**What are the evaluation metrics specific to Naive Bayes performance?**

**Answer:**

Standard classification metrics apply: accuracy, precision, recall, F1-score, and confusion matrix. Additionally, since NB outputs probabilities, evaluate probability calibration using Brier score and calibration curves. Log-loss measures prediction confidence quality. For text, per-class metrics help identify which classes NB handles well.

**Key Metrics:**

| Metric | Formula | When to Use |
|--------|---------|-------------|
| Accuracy | (TP+TN)/Total | Balanced classes |
| Precision | TP/(TP+FP) | Cost of FP high |
| Recall | TP/(TP+FN) | Cost of FN high |
| F1-Score | 2×P×R/(P+R) | Balance P and R |
| Log-Loss | -Σ y log(p) | Probability quality |
| Brier Score | Σ(p-y)² | Calibration |

**Implementation:**

```python
from sklearn.metrics import (
    accuracy_score, classification_report, 
    log_loss, brier_score_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve

# Basic metrics
print(classification_report(y_test, y_pred))

# Probability-based metrics
probs = nb.predict_proba(X_test)
print(f"Log Loss: {log_loss(y_test, probs):.4f}")
print(f"Brier Score: {brier_score_loss(y_test, probs[:,1]):.4f}")

# Calibration curve (NB often poorly calibrated)
prob_true, prob_pred = calibration_curve(y_test, probs[:,1], n_bins=10)
```

**NB-Specific Considerations:**
- NB probabilities tend to be extreme (near 0 or 1)
- Consider probability calibration (Platt scaling, isotonic regression)
- Log-loss may be high even with good accuracy

---

## Question 38
- [ ] Done

**How do you implement cross-validation for Naive Bayes models?**

**Answer:**

Cross-validation for Naive Bayes uses the same approach as other classifiers - split data into k folds, train on k-1 folds, test on remaining fold, repeat k times. Since NB trains fast, cross-validation is efficient. Use stratified k-fold to maintain class distribution. Commonly used to tune smoothing parameter (alpha).

**Implementation:**

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Basic cross-validation
nb = MultinomialNB()
scores = cross_val_score(nb, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# Stratified K-Fold (recommended for imbalanced data)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(nb, X, y, cv=skf)

# Cross-validation with hyperparameter tuning
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

param_grid = {'nb__alpha': [0.01, 0.1, 0.5, 1.0, 2.0]}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)
print(f"Best alpha: {grid_search.best_params_}")
```

**Best Practices:**
- Use stratified k-fold for classification
- 5-fold or 10-fold are common choices
- For text, include vectorizer in pipeline (prevents data leakage)
- Tune alpha parameter via CV, not separate test set

---

## Question 39
- [ ] Done

**What is the relationship between Naive Bayes and logistic regression?**

**Answer:**

Both are linear classifiers with the same decision boundary form (linear in features). Key difference: NB is generative (models P(X|Y)) while Logistic Regression is discriminative (models P(Y|X) directly). NB makes independence assumptions; LR doesn't. With infinite data, LR typically wins. With limited data, NB often outperforms due to its strong inductive bias.

**Comparison:**

| Aspect | Naive Bayes | Logistic Regression |
|--------|-------------|---------------------|
| Type | Generative | Discriminative |
| Models | P(X\|Y) and P(Y) | P(Y\|X) directly |
| Assumption | Feature independence | None (implicit regularization) |
| Training | Counting | Optimization (gradient descent) |
| Speed | Very fast | Slower |
| Small data | Better | May overfit |
| Large data | LR usually wins | Usually better |
| Correlated features | Degrades performance | Handles better |

**Mathematical Connection:**

Both produce linear decision boundaries:
$$\log \frac{P(Y=1|X)}{P(Y=0|X)} = w_0 + \sum_i w_i X_i$$

For Gaussian NB with equal variance, the weights are:
$$w_i = \frac{\mu_{i,1} - \mu_{i,0}}{\sigma_i^2}$$

**When to Choose:**
- **Choose NB**: Fast training needed, small dataset, text classification
- **Choose LR**: Accurate probabilities needed, correlated features, larger dataset

---

## Question 40
- [ ] Done

**How do you handle multi-class classification with Naive Bayes?**

**Answer:**

Naive Bayes naturally handles multi-class classification without modification. It computes P(Y=c|X) for each class c and predicts the class with highest posterior. No one-vs-rest or one-vs-one needed. Training learns priors and likelihoods for each class independently, making multi-class as easy as binary.

**How It Works:**

$$\hat{Y} = \arg\max_{c \in \{1,...,K\}} P(Y=c) \cdot \prod_{i=1}^{n} P(X_i|Y=c)$$

Simply compute the posterior for each of K classes and pick the maximum.

**Implementation:**

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups

# Load multi-class text data (20 classes)
data = fetch_20newsgroups(subset='train', categories=None)

# Train - same code as binary!
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data.data)
y = data.target

nb = MultinomialNB()
nb.fit(X, y)  # Automatically handles 20 classes

# Predict
predictions = nb.predict(X_test)
probabilities = nb.predict_proba(X_test)  # Shape: (n_samples, 20)
```

**Multi-class Properties:**
- Priors: P(Y=c) learned for each class
- Likelihoods: P(Xi|Y=c) learned for each class
- Class probabilities sum to 1
- Computational cost: O(n_classes × n_features)

---

## Question 41
- [ ] Done

**What are ensemble methods for Naive Bayes and their benefits?**

**Answer:**

Ensemble methods combine multiple NB models to improve performance. Common approaches: (1) Bootstrap aggregating with feature subsets, (2) Combining different NB variants (Gaussian + Multinomial), (3) Weighted voting from NB models on different data views. Benefits include improved accuracy, robustness, and handling mixed feature types.

**Ensemble Approaches:**

**1. Feature Subspace Ensemble:**
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB

ensemble = BaggingClassifier(
    estimator=GaussianNB(),
    n_estimators=10,
    max_features=0.7,  # Use 70% of features per model
    bootstrap=True
)
ensemble.fit(X_train, y_train)
```

**2. Multi-View NB (Different Feature Types):**
```python
# Combine predictions from different NB variants
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# For continuous features
gnb = GaussianNB()
gnb.fit(X_continuous, y)

# For text features
mnb = MultinomialNB()
mnb.fit(X_text, y)

# Combine probabilities
probs_gnb = gnb.predict_proba(X_test_cont)
probs_mnb = mnb.predict_proba(X_test_text)
combined_probs = 0.5 * probs_gnb + 0.5 * probs_mnb
predictions = combined_probs.argmax(axis=1)
```

**Benefits:**
- Reduces variance from single model
- Handles mixed feature types
- More robust to noise
- Can improve probability calibration

---

## Question 42
- [ ] Done

**How do you implement Naive Bayes for spam email detection?**

**Answer:**

Spam detection uses Multinomial NB on email text. Pipeline: (1) Preprocess emails (remove HTML, lowercase, tokenize), (2) Extract TF-IDF or bag-of-words features, (3) Train NB to learn spam vs. ham word distributions, (4) Predict by comparing posteriors. Include email headers, subject line as additional features.

**Complete Implementation:**

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample data
emails = [
    "Free lottery winner click here",
    "Meeting at 3pm regarding project",
    "Claim your prize money now",
    "Please review the attached report",
    # ... more emails
]
labels = [1, 0, 1, 0]  # 1=spam, 0=ham

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    emails, labels, test_size=0.2, random_state=42
)

# Build pipeline
spam_detector = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000
    )),
    ('nb', MultinomialNB(alpha=0.1))
])

# Train
spam_detector.fit(X_train, y_train)

# Evaluate
y_pred = spam_detector.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# Predict new email
new_email = ["Congratulations! You won $1000 click to claim"]
prediction = spam_detector.predict(new_email)
probability = spam_detector.predict_proba(new_email)
```

**Key Features for Spam:**
- Suspicious words: "free", "winner", "click", "urgent"
- All caps usage
- Excessive punctuation
- URL patterns

---

## Question 43
- [ ] Done

**What is sentiment analysis using Naive Bayes classifiers?**

**Answer:**

Sentiment analysis with NB classifies text (reviews, tweets) as positive, negative, or neutral based on word patterns. It learns which words associate with each sentiment. For example, "excellent" and "love" correlate with positive, while "terrible" and "hate" correlate with negative. Multinomial NB with TF-IDF is the standard approach.

**Implementation:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Training data
reviews = [
    "This product is amazing, love it!",
    "Terrible quality, waste of money",
    "Pretty good, satisfied with purchase",
    "Worst experience ever, avoid!",
]
sentiments = [1, 0, 1, 0]  # 1=positive, 0=negative

# Build sentiment analyzer
sentiment_model = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000
    )),
    ('nb', MultinomialNB(alpha=0.1))
])

# Train
sentiment_model.fit(reviews, sentiments)

# Analyze new text
test_reviews = ["Great product, highly recommend!"]
predictions = sentiment_model.predict(test_reviews)
probabilities = sentiment_model.predict_proba(test_reviews)

print(f"Sentiment: {'Positive' if predictions[0] == 1 else 'Negative'}")
print(f"Confidence: {probabilities[0].max():.2%}")
```

**Challenges and Solutions:**
- Negation ("not good"): Use bigrams
- Sarcasm: Difficult, may need deep learning
- Context: Consider domain-specific training data

---

## Question 44
- [ ] Done

**How do you handle negation and context in Naive Bayes text classification?**

**Answer:**

Negation flips sentiment ("not good" = bad) but bag-of-words treats "not" and "good" independently. Solutions: (1) N-grams to capture "not good" as unit, (2) Negation marking - append "NOT_" to words after negation until punctuation, (3) Sentiment-aware features. These preserve negation semantics that unigrams miss.

**Approaches:**

**1. N-grams (Simple and Effective):**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Bigrams capture "not good", "not bad", etc.
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
```

**2. Negation Marking (Custom Preprocessing):**
```python
import re

def mark_negation(text):
    """Add NOT_ prefix to words following negation until punctuation."""
    negation_words = {'not', "n't", 'no', 'never', 'neither'}
    words = text.lower().split()
    result = []
    negate = False
    
    for word in words:
        if word in negation_words:
            negate = True
            result.append(word)
        elif re.search(r'[.!?]', word):
            negate = False
            result.append(word)
        elif negate:
            result.append('NOT_' + word)
        else:
            result.append(word)
    
    return ' '.join(result)

# "I do not like this movie" -> "I do not NOT_like NOT_this NOT_movie"
```

**3. Sentiment-Aware Features:**
- Use sentiment lexicons (VADER, SentiWordNet)
- Add sentiment scores as features
- Mark intensifiers ("very", "extremely")

**Impact Example:**
```
Original: "not good"
Unigrams: ["not", "good"] - may predict positive (good is positive word)
Bigrams: ["not good"] - learned as negative pattern
Negation marking: ["not", "NOT_good"] - NOT_good learned as negative
```

---

## Question 45
- [ ] Done

**What are the challenges of Naive Bayes in real-world applications?**

**Answer:**

Key challenges include: (1) Independence assumption violation with correlated features, (2) Poor probability calibration yielding extreme probabilities, (3) Zero frequency problem for unseen features, (4) Difficulty with continuous features not following assumed distribution, (5) Inability to capture feature interactions. Understanding these guides when to use NB vs. alternatives.

**Challenges and Mitigations:**

| Challenge | Impact | Mitigation |
|-----------|--------|------------|
| Feature correlation | Over-confident predictions | Feature selection, remove redundant |
| Poor calibration | Probabilities near 0/1 | Platt scaling, isotonic regression |
| Zero frequency | Zero probability for unseen | Laplace smoothing |
| Distribution mismatch | Wrong likelihood estimates | Discretize, use kernel NB |
| No interactions | Misses XOR-like patterns | Feature engineering |
| Imbalanced classes | Biased toward majority | Adjust priors, resampling |

**Probability Calibration Fix:**
```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrate NB probabilities
calibrated_nb = CalibratedClassifierCV(
    MultinomialNB(), 
    method='sigmoid'  # Platt scaling
)
calibrated_nb.fit(X_train, y_train)
```

**When NB Struggles:**
- Features are highly correlated (e.g., pixel values in images)
- Feature interactions matter (e.g., XOR problem)
- Accurate probabilities needed (e.g., ranking, bidding)

---

## Question 46
- [ ] Done

**How do you implement incremental learning with Naive Bayes?**

**Answer:**

Incremental (online) learning updates the model with new data without retraining from scratch. NB supports this naturally because training just requires count updates. Scikit-learn's `partial_fit()` method enables incremental training. This is valuable for streaming data, large datasets, or when new labeled data arrives continuously.

**Implementation:**

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

# HashingVectorizer for consistent feature space
vectorizer = HashingVectorizer(n_features=2**16)

# Initialize model
nb = MultinomialNB()

# Define all possible classes upfront
classes = np.array([0, 1])  # Must specify for partial_fit

# Simulate streaming data
data_batches = [
    (["spam message here", "legitimate email"], [1, 0]),
    (["another spam", "work meeting notes"], [1, 0]),
    # ... more batches
]

# Incremental training
for texts, labels in data_batches:
    X_batch = vectorizer.transform(texts)
    nb.partial_fit(X_batch, labels, classes=classes)

# Model continuously updated without storing all data

# Predict
X_new = vectorizer.transform(["is this spam?"])
prediction = nb.predict(X_new)
```

**Key Points:**
- Must specify `classes` parameter on first `partial_fit` call
- Use `HashingVectorizer` (fixed feature space) not `TfidfVectorizer`
- Model improves as more data arrives
- Memory efficient - no need to store all training data

---

## Question 47
- [ ] Done

**What is online Naive Bayes for streaming data classification?**

**Answer:**

Online NB processes data streams one sample (or mini-batch) at a time, updating model parameters incrementally. Unlike batch learning, it doesn't require storing all data. Parameters (class counts, feature counts) are updated with each new observation. Essential for real-time applications like live spam filtering, social media monitoring, and IoT data classification.

**Online Learning Algorithm:**

```
Initialize: class_counts = {}, feature_counts = {}

For each new sample (x, y):
    1. class_counts[y] += 1
    2. For each feature i in x:
        feature_counts[y][i] += x[i]
    3. Optionally: update priors and likelihoods
```

**Implementation:**

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

class OnlineSpamFilter:
    def __init__(self):
        self.vectorizer = HashingVectorizer(n_features=2**14)
        self.model = MultinomialNB()
        self.classes = np.array([0, 1])  # ham, spam
        self.is_fitted = False
    
    def update(self, text, label):
        """Update model with single new example."""
        X = self.vectorizer.transform([text])
        if not self.is_fitted:
            self.model.partial_fit(X, [label], classes=self.classes)
            self.is_fitted = True
        else:
            self.model.partial_fit(X, [label])
    
    def predict(self, text):
        """Classify new text."""
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]

# Usage
filter = OnlineSpamFilter()
# Stream of emails
filter.update("Free money click here", 1)  # spam
filter.update("Meeting tomorrow at 2pm", 0)  # ham
# Model learns incrementally
```

**Advantages:**
- Constant memory usage
- Adapts to changing patterns
- Real-time updates
- Handles infinite data streams

---

## Question 48
- [ ] Done

**How do you handle concept drift in Naive Bayes models?**

**Answer:**

Concept drift occurs when data distribution changes over time (e.g., new spam patterns emerge). Solutions: (1) Sliding window - train only on recent data, (2) Weighted samples - decay older data importance, (3) Drift detection - monitor performance and retrain when degradation detected, (4) Ensemble with recency weighting.

**Approaches:**

**1. Sliding Window:**
```python
from collections import deque

class SlidingWindowNB:
    def __init__(self, window_size=1000):
        self.window_X = deque(maxlen=window_size)
        self.window_y = deque(maxlen=window_size)
        self.model = MultinomialNB()
    
    def update(self, X_new, y_new):
        self.window_X.append(X_new)
        self.window_y.append(y_new)
        
        # Retrain on window
        X = np.vstack(list(self.window_X))
        y = list(self.window_y)
        self.model.fit(X, y)
```

**2. Drift Detection (Monitor Performance):**
```python
def detect_drift(recent_accuracy, historical_accuracy, threshold=0.05):
    """Detect if performance dropped significantly."""
    return (historical_accuracy - recent_accuracy) > threshold

# If drift detected, retrain or reset model
if detect_drift(recent_acc, hist_acc):
    model = MultinomialNB()  # Fresh model
    model.fit(recent_data)
```

**3. Exponential Decay Weighting:**
- Weight recent samples more heavily
- Older samples contribute less to counts

**Best Practices:**
- Monitor model accuracy over time
- Keep hold-out validation set current
- Consider ensemble of old and new models
- Log data timestamps for analysis

---

## Question 49
- [ ] Done

**What are kernel-based extensions to Naive Bayes?**

**Answer:**

Kernel Naive Bayes uses Kernel Density Estimation (KDE) instead of parametric distributions (Gaussian) to estimate P(X|Y). This allows flexible, non-parametric density estimation that can model any distribution shape. Particularly useful when continuous features don't follow Gaussian distribution (multimodal, skewed, heavy-tailed).

**Concept:**
- Standard Gaussian NB: Assumes P(X|Y) is Gaussian
- Kernel NB: Estimates P(X|Y) using KDE (sum of kernels centered at training points)

**Implementation:**

```python
from sklearn.neighbors import KernelDensity
import numpy as np

class KernelNaiveBayes:
    def __init__(self, bandwidth=0.5):
        self.bandwidth = bandwidth
        self.kde = {}  # KDE per class per feature
        self.priors = {}
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]
        
        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(X)
            self.kde[c] = []
            
            for j in range(n_features):
                kde = KernelDensity(bandwidth=self.bandwidth)
                kde.fit(X_c[:, j].reshape(-1, 1))
                self.kde[c].append(kde)
        
        return self
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                log_post = np.log(self.priors[c])
                for j, val in enumerate(x):
                    log_post += self.kde[c][j].score_samples([[val]])[0]
                posteriors.append(log_post)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)
```

**When to Use:**
- Features don't follow Gaussian distribution
- Multi-modal feature distributions
- When parametric assumptions hurt performance

---

## Question 50
- [ ] Done

**How do you implement semi-supervised learning with Naive Bayes?**

**Answer:**

Semi-supervised NB uses both labeled and unlabeled data via Expectation-Maximization (EM). Steps: (1) Initialize with labeled data, (2) E-step: predict probabilities for unlabeled data, (3) M-step: re-estimate parameters using all data weighted by probabilities, (4) Iterate until convergence. This leverages abundant unlabeled data to improve model.

**Algorithm (EM for Semi-Supervised NB):**

```
1. Train initial NB on labeled data only
2. Repeat until convergence:
   E-step: Compute P(Y|X) for unlabeled data using current model
   M-step: Re-estimate all parameters using:
           - Labeled data (hard labels)
           - Unlabeled data (soft labels = probabilities)
```

**Implementation:**

```python
import numpy as np
from sklearn.naive_bayes import MultinomialNB

def semi_supervised_nb(X_labeled, y_labeled, X_unlabeled, max_iter=10):
    # Initialize with labeled data
    nb = MultinomialNB()
    nb.fit(X_labeled, y_labeled)
    
    for iteration in range(max_iter):
        # E-step: predict probabilities for unlabeled
        probs_unlabeled = nb.predict_proba(X_unlabeled)
        
        # M-step: combine labeled and unlabeled (weighted)
        X_combined = np.vstack([X_labeled, X_unlabeled])
        
        # For labeled: one-hot, for unlabeled: soft probabilities
        y_labeled_onehot = np.eye(len(nb.classes_))[y_labeled]
        weights = np.vstack([y_labeled_onehot, probs_unlabeled])
        
        # Weighted update (simplified - full implementation needs custom NB)
        # Here we use pseudo-labels from high-confidence predictions
        pseudo_labels = nb.predict(X_unlabeled)
        confidence = probs_unlabeled.max(axis=1)
        
        # Only use high-confidence predictions
        high_conf_mask = confidence > 0.9
        X_pseudo = X_unlabeled[high_conf_mask]
        y_pseudo = pseudo_labels[high_conf_mask]
        
        # Retrain
        X_train = np.vstack([X_labeled, X_pseudo])
        y_train = np.concatenate([y_labeled, y_pseudo])
        nb.fit(X_train, y_train)
    
    return nb
```

**Use Cases:**
- Limited labeled data (expensive to annotate)
- Large amounts of unlabeled text
- Document classification with few examples

---

## Question 51
- [ ] Done

**What is the role of Naive Bayes in anomaly detection?**

**Answer:**

Naive Bayes can detect anomalies by modeling the "normal" class distribution and flagging instances with low probability as anomalies. Train NB on normal data only, then compute P(normal|x) for new instances - low probability indicates anomaly. Works well for one-class classification when anomalies are rare or undefined.

**Approaches:**

**1. Probability Threshold Method:**
```python
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Train on normal data only
nb = GaussianNB()
nb.fit(X_normal, np.zeros(len(X_normal)))  # All labeled as normal

# For new data, compute log probability
log_probs = nb.score_samples(X_test) if hasattr(nb, 'score_samples') else None

# Alternative: use predict_proba and threshold
def detect_anomaly(X, model, threshold=-10):
    log_likelihood = []
    for x in X:
        # Compute log P(x|normal)
        log_prob = 0
        for i, val in enumerate(x):
            mean = model.theta_[0, i]
            var = model.var_[0, i]
            log_prob += -0.5 * np.log(2 * np.pi * var) - (val - mean)**2 / (2 * var)
        log_likelihood.append(log_prob)
    return np.array(log_likelihood) < threshold
```

**2. Two-Class Approach (if anomaly examples exist):**
```python
# Train binary classifier
X_train = np.vstack([X_normal, X_anomaly])
y_train = np.array([0]*len(X_normal) + [1]*len(X_anomaly))

nb = GaussianNB()
nb.fit(X_train, y_train)
anomalies = nb.predict(X_test) == 1
```

**Use Cases:**
- Network intrusion detection
- Fraud detection (unusual transactions)
- Equipment failure prediction
- Quality control in manufacturing

---

## Question 52
- [ ] Done

**How do you handle hierarchical classification with Naive Bayes?**

**Answer:**

Hierarchical classification organizes classes in a tree structure (e.g., Animal → Mammal → Dog). Approaches: (1) Flat - ignore hierarchy, train single classifier, (2) Local per Node - train NB at each node, (3) Local per Level - train multi-class NB at each level, (4) Global - encode hierarchy in features.

**Example Hierarchy:**
```
Product
├── Electronics
│   ├── Phone
│   └── Laptop
└── Clothing
    ├── Shirt
    └── Pants
```

**Approach 1: Local Classifier Per Node (LCN)**
```python
from sklearn.naive_bayes import MultinomialNB

class HierarchicalNB:
    def __init__(self, hierarchy):
        self.hierarchy = hierarchy  # dict: parent -> [children]
        self.classifiers = {}  # node -> NB classifier
    
    def fit(self, X, y, node='root'):
        children = self.hierarchy.get(node, [])
        if not children:
            return
        
        # Get samples belonging to this subtree
        child_labels = self._get_all_descendants(node)
        mask = np.isin(y, child_labels)
        X_node, y_node = X[mask], y[mask]
        
        # Map to immediate children
        y_mapped = self._map_to_children(y_node, node)
        
        # Train classifier at this node
        self.classifiers[node] = MultinomialNB()
        self.classifiers[node].fit(X_node, y_mapped)
        
        # Recurse to children
        for child in children:
            self.fit(X, y, child)
    
    def predict(self, X):
        predictions = []
        for x in X:
            pred = self._predict_single(x.reshape(1, -1), 'root')
            predictions.append(pred)
        return predictions
    
    def _predict_single(self, x, node):
        children = self.hierarchy.get(node, [])
        if not children:
            return node
        pred_child = self.classifiers[node].predict(x)[0]
        return self._predict_single(x, pred_child)
```

**Advantages:**
- Exploits class relationships
- Can stop at intermediate levels
- Better for deep hierarchies

---

## Question 53
- [ ] Done

**What are the privacy-preserving techniques for Naive Bayes?**

**Answer:**

Privacy-preserving NB trains models without exposing raw data. Techniques: (1) Differential Privacy - add noise to counts/parameters, (2) Secure Multi-Party Computation - compute jointly without sharing data, (3) Homomorphic Encryption - compute on encrypted data, (4) Federated Learning - train locally, share only parameters.

**Technique 1: Differential Privacy**
```python
import numpy as np

def dp_naive_bayes_train(X, y, epsilon=1.0):
    """
    Train NB with differential privacy.
    Add Laplace noise to counts proportional to 1/epsilon.
    """
    classes = np.unique(y)
    n_features = X.shape[1]
    
    # Noisy class counts
    class_counts = {}
    for c in classes:
        true_count = np.sum(y == c)
        noise = np.random.laplace(0, 1/epsilon)
        class_counts[c] = max(1, true_count + noise)
    
    # Noisy feature counts
    feature_counts = {c: np.zeros(n_features) for c in classes}
    for c in classes:
        X_c = X[y == c]
        for j in range(n_features):
            true_sum = X_c[:, j].sum()
            noise = np.random.laplace(0, 1/epsilon)
            feature_counts[c][j] = max(0, true_sum + noise)
    
    return class_counts, feature_counts
```

**Technique 2: Federated Learning (Local Training)**
```python
def federated_nb_update(local_X, local_y):
    """Compute local sufficient statistics to share."""
    classes = np.unique(local_y)
    stats = {
        'class_counts': {c: np.sum(local_y == c) for c in classes},
        'feature_sums': {c: local_X[local_y == c].sum(axis=0) for c in classes}
    }
    return stats  # Send only aggregated stats, not raw data

def aggregate_stats(all_stats):
    """Server aggregates stats from all parties."""
    # Sum counts from all parties
    # Train global NB from aggregated stats
    pass
```

**Key Concepts:**
- Never share raw data
- Trade-off: more privacy = more noise = less accuracy
- NB is well-suited for privacy (only needs counts)

---

## Question 54
- [ ] Done

**How do you implement federated learning with Naive Bayes?**

**Answer:**

Federated NB trains on distributed data without centralizing it. Each client computes local sufficient statistics (class counts, feature sums), sends only these to server. Server aggregates to build global model. Works well because NB only needs counts, not individual samples.

**Architecture:**
```
Client 1 (Hospital A)     Client 2 (Hospital B)
      ↓ local stats            ↓ local stats
           ↘                  ↙
              Aggregation Server
                    ↓
              Global NB Model
                    ↓
           ↙                  ↘
      Client 1                Client 2
```

**Implementation:**
```python
import numpy as np

class FederatedNBClient:
    def __init__(self, client_id):
        self.client_id = client_id
    
    def compute_local_stats(self, X, y):
        """Compute sufficient statistics locally."""
        classes = np.unique(y)
        stats = {
            'n_samples': len(y),
            'class_counts': {},
            'feature_sums': {},
            'feature_sq_sums': {}  # For variance in Gaussian NB
        }
        for c in classes:
            mask = (y == c)
            stats['class_counts'][c] = mask.sum()
            stats['feature_sums'][c] = X[mask].sum(axis=0)
            stats['feature_sq_sums'][c] = (X[mask]**2).sum(axis=0)
        return stats

class FederatedNBServer:
    def aggregate(self, client_stats_list):
        """Aggregate stats from all clients."""
        all_classes = set()
        for stats in client_stats_list:
            all_classes.update(stats['class_counts'].keys())
        
        global_stats = {
            'n_samples': sum(s['n_samples'] for s in client_stats_list),
            'class_counts': {c: 0 for c in all_classes},
            'feature_sums': {c: 0 for c in all_classes},
        }
        
        for stats in client_stats_list:
            for c in stats['class_counts']:
                global_stats['class_counts'][c] += stats['class_counts'][c]
                global_stats['feature_sums'][c] += stats['feature_sums'][c]
        
        return self._build_model(global_stats)
    
    def _build_model(self, global_stats):
        # Compute priors and likelihoods from aggregated counts
        pass
```

**Benefits:**
- Data never leaves client devices
- Complies with privacy regulations (GDPR, HIPAA)
- Scalable to many clients

---

## Question 55
- [ ] Done

**What is the interpretability advantage of Naive Bayes over other classifiers?**

**Answer:**

Naive Bayes is highly interpretable because: (1) Each feature's contribution is independent and quantifiable, (2) Log-probability ratios show feature importance, (3) Predictions decompose into sum of evidence from each feature, (4) No hidden layers or complex interactions to explain. Unlike black-box models, you can explain exactly why NB made a prediction.

**Interpretability Factors:**

| Aspect | Naive Bayes | Neural Network | SVM |
|--------|-------------|----------------|-----|
| Feature contribution | Direct, additive | Hidden layers | Kernel space |
| Probability meaning | True likelihood ratios | Needs calibration | Not probabilistic |
| Global understanding | Class profiles visible | Opaque | Hard to interpret |
| Local explanation | Sum of log-probs | SHAP/LIME needed | SHAP/LIME needed |

**Example: Explaining Spam Prediction**
```python
def explain_prediction(nb, vectorizer, text):
    X = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    
    # Log probability ratios for each word
    log_prob_spam = nb.feature_log_prob_[1]  # P(word|spam)
    log_prob_ham = nb.feature_log_prob_[0]   # P(word|ham)
    log_ratios = log_prob_spam - log_prob_ham
    
    # Get words in this document
    word_indices = X.nonzero()[1]
    
    print("Evidence for SPAM:")
    contributions = []
    for idx in word_indices:
        word = feature_names[idx]
        contribution = log_ratios[idx] * X[0, idx]
        contributions.append((word, contribution))
    
    # Sort by contribution
    for word, contrib in sorted(contributions, key=lambda x: -x[1])[:5]:
        direction = "SPAM" if contrib > 0 else "HAM"
        print(f"  '{word}': {contrib:.3f} → {direction}")
```

**Output Example:**
```
Evidence for SPAM:
  'free': 2.341 → SPAM
  'click': 1.892 → SPAM
  'meeting': -1.234 → HAM
```

---

## Question 56
- [ ] Done

**How do you explain Naive Bayes predictions to stakeholders?**

**Answer:**

Explain NB as "evidence accumulation" - each feature adds or subtracts evidence for a class. Use visualizations: bar charts of feature contributions, word clouds for text, probability breakdowns. Frame in business terms: "The email was classified as spam because it contains 'free' (strong spam indicator) and 'click here' (moderate spam indicator)."

**Explanation Strategy:**

**1. Non-Technical Summary:**
```
"The model looks at each word in the email and asks: 
'How often does this word appear in spam vs. legitimate emails?'
Words like 'free' and 'winner' are strong spam signals.
The final decision combines evidence from all words."
```

**2. Visual Explanation:**
```python
import matplotlib.pyplot as plt

def visualize_prediction(word_contributions, prediction):
    words = [w for w, _ in word_contributions]
    scores = [s for _, s in word_contributions]
    colors = ['red' if s > 0 else 'green' for s in scores]
    
    plt.figure(figsize=(10, 6))
    plt.barh(words, scores, color=colors)
    plt.axvline(x=0, color='black', linestyle='-')
    plt.xlabel('Evidence (positive = spam, negative = not spam)')
    plt.title(f'Why this email was classified as {prediction}')
    plt.tight_layout()
    plt.show()
```

**3. Confidence Explanation:**
```python
def explain_confidence(nb, X):
    probs = nb.predict_proba(X)[0]
    pred_class = nb.classes_[probs.argmax()]
    confidence = probs.max()
    
    explanation = f"""
    Prediction: {pred_class}
    Confidence: {confidence:.1%}
    
    Interpretation:
    - Above 90%: Very confident
    - 70-90%: Moderately confident  
    - 50-70%: Low confidence, borderline case
    """
    return explanation
```

**Key Messages for Stakeholders:**
- Model is transparent (no black box)
- Each factor's influence is measurable
- Predictions can be audited and explained
- Easy to identify and fix mistakes

---

## Question 57
- [ ] Done

**What are feature importance measures in Naive Bayes?**

**Answer:**

Feature importance in NB can be measured by: (1) Log-probability ratio - how much feature shifts prediction between classes, (2) Mutual Information - how much feature reduces uncertainty about class, (3) Chi-square statistic - independence test between feature and class, (4) Permutation importance - accuracy drop when feature is shuffled.

**Method 1: Log-Probability Ratio**
```python
def log_prob_importance(nb):
    """Importance = |log P(feature|class1) - log P(feature|class0)|"""
    log_prob_diff = np.abs(nb.feature_log_prob_[1] - nb.feature_log_prob_[0])
    return log_prob_diff

importance = log_prob_importance(nb)
top_features = np.argsort(importance)[-10:]  # Top 10
```

**Method 2: Mutual Information**
```python
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X, y)
print("Top features by MI:", np.argsort(mi_scores)[-10:])
```

**Method 3: Chi-Square Test**
```python
from sklearn.feature_selection import chi2

chi_scores, p_values = chi2(X, y)
print("Top features by Chi2:", np.argsort(chi_scores)[-10:])
```

**Method 4: Permutation Importance**
```python
from sklearn.inspection import permutation_importance

result = permutation_importance(nb, X_test, y_test, n_repeats=10)
importance = result.importances_mean
print("Top features:", np.argsort(importance)[-10:])
```

**Visualization:**
```python
def plot_feature_importance(importance, feature_names, top_n=20):
    indices = np.argsort(importance)[-top_n:]
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importance[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title('Top Features in Naive Bayes')
    plt.show()
```

---

## Question 58
- [ ] Done

**How do you handle categorical features with high cardinality in Naive Bayes?**

**Answer:**

High cardinality (many unique values) causes sparse counts and overfitting. Solutions: (1) Increase smoothing (higher alpha), (2) Group rare categories into "other", (3) Use target encoding (replace category with class probability), (4) Hash categories to fixed buckets, (5) Hierarchical grouping (city → state → region).

**Problem:**
```
Feature: City (10,000 unique values)
Most cities appear only once → P(city|class) unreliable
```

**Solution 1: Group Rare Categories**
```python
def group_rare_categories(series, min_count=10):
    counts = series.value_counts()
    rare = counts[counts < min_count].index
    return series.replace(rare, 'OTHER')

df['city_grouped'] = group_rare_categories(df['city'], min_count=50)
```

**Solution 2: Target Encoding**
```python
def target_encode(df, col, target, smoothing=10):
    """Replace category with smoothed target mean."""
    global_mean = df[target].mean()
    stats = df.groupby(col)[target].agg(['mean', 'count'])
    
    # Smoothed mean: (count * category_mean + smoothing * global_mean) / (count + smoothing)
    stats['smoothed'] = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
    
    return df[col].map(stats['smoothed'])

df['city_encoded'] = target_encode(df, 'city', 'target')
```

**Solution 3: Feature Hashing**
```python
from sklearn.feature_extraction import FeatureHasher

hasher = FeatureHasher(n_features=100, input_type='string')
X_hashed = hasher.transform([[city] for city in df['city']])
# 10,000 cities → 100 hash buckets
```

**Solution 4: Higher Smoothing**
```python
# More smoothing for high cardinality
nb = MultinomialNB(alpha=10.0)  # Higher than default 1.0
```

**Best Practice:** Combine grouping + moderate smoothing for best results.

---

## Question 59
- [ ] Done

**What is the role of Naive Bayes in recommendation systems?**

**Answer:**

NB predicts user preference P(like|item_features) for ranking recommendations. Used in: (1) Content-based filtering - predict from item attributes, (2) Hybrid systems - combine with collaborative filtering, (3) Click prediction - estimate P(click) for ad recommendations. Fast inference makes it suitable for real-time recommendations.

**Content-Based Recommendation:**
```python
from sklearn.naive_bayes import BernoulliNB
import numpy as np

class NBRecommender:
    def __init__(self):
        self.model = BernoulliNB()
    
    def fit(self, item_features, user_ratings):
        """
        item_features: binary matrix (items × features)
        user_ratings: 1 if liked, 0 if not liked
        """
        self.model.fit(item_features, user_ratings)
    
    def recommend(self, candidate_items, top_k=10):
        """Return top-k items most likely to be liked."""
        probs = self.model.predict_proba(candidate_items)[:, 1]
        top_indices = np.argsort(probs)[-top_k:][::-1]
        return top_indices, probs[top_indices]

# Example: Movie recommendation
# Features: genres, director, actors (binary)
item_features = np.array([
    [1, 0, 1, 0],  # Action, no Comedy, Sci-Fi, no Romance
    [0, 1, 0, 1],  # Comedy, Romance
    [1, 0, 0, 0],  # Action only
])
user_likes = [1, 0, 1]  # User liked action movies

rec = NBRecommender()
rec.fit(item_features, user_likes)

# New movies to rank
new_movies = np.array([[1, 0, 1, 0], [0, 1, 1, 0]])
indices, scores = rec.recommend(new_movies)
```

**Advantages in Recommendations:**
- Fast real-time scoring
- Handles sparse features
- Works with limited user history
- Provides interpretable scores

---

## Question 60
- [ ] Done

**How do you implement Naive Bayes for image classification?**

**Answer:**

For images, use Gaussian NB with pixel intensities as features, or Bernoulli NB with binarized pixels. Better approach: extract features first (HOG, SIFT, CNN features) then apply NB. Works for simple tasks (digit recognition) but generally outperformed by CNNs for complex images.

**Approach 1: Raw Pixel Features**
```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load digit images (8x8 pixels = 64 features)
digits = load_digits()
X, y = digits.data, digits.target  # X: (samples, 64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Gaussian NB on pixel intensities
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"Accuracy: {gnb.score(X_test, y_test):.3f}")
```

**Approach 2: Binarized Pixels (Bernoulli NB)**
```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import Binarizer

# Binarize: pixel > threshold → 1, else 0
binarizer = Binarizer(threshold=8)  # Middle of 0-16 range
X_binary = binarizer.fit_transform(X)

bnb = BernoulliNB()
bnb.fit(X_binary[train_idx], y[train_idx])
```

**Approach 3: With Feature Extraction (Better)**
```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('pca', PCA(n_components=50)),  # Reduce dimensionality
    ('nb', GaussianNB())
])
pipeline.fit(X_train, y_train)
print(f"PCA + NB Accuracy: {pipeline.score(X_test, y_test):.3f}")
```

**Limitations:**
- Ignores spatial relationships (adjacent pixels)
- Independence assumption very wrong for images
- Use CNNs for serious image classification

---

## Question 61
- [ ] Done

**What are the considerations for Naive Bayes in big data environments?**

**Answer:**

NB scales well to big data because it only needs count statistics. Considerations: (1) Use streaming/online learning for data that doesn't fit in memory, (2) Parallelize count computation across workers, (3) Use sparse representations for high-dimensional data, (4) Consider distributed frameworks (Spark MLlib), (5) Monitor numerical stability with log probabilities.

**Scalability Advantages of NB:**
- Training: O(n·d) - linear in samples and features
- Prediction: O(d·k) - linear in features and classes
- Memory: Only stores counts, not data

**Strategies:**

**1. Online Learning (Streaming)**
```python
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
for batch_X, batch_y in data_stream:
    nb.partial_fit(batch_X, batch_y, classes=[0, 1])
```

**2. Sparse Matrices (High-Dimensional)**
```python
from scipy.sparse import csr_matrix

# Store only non-zero elements
X_sparse = csr_matrix(X)  # Much less memory
nb.fit(X_sparse, y)  # sklearn NB supports sparse
```

**3. Feature Hashing (Fixed Memory)**
```python
from sklearn.feature_extraction.text import HashingVectorizer

# Fixed number of features regardless of vocabulary size
vectorizer = HashingVectorizer(n_features=2**18)
```

**4. Distributed Computing (Spark)**
```python
from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes(smoothing=1.0)
model = nb.fit(training_df)  # Distributed training
```

**Key Considerations:**
- Log probabilities prevent underflow
- Aggregate counts are sufficient statistics
- Embarrassingly parallelizable

---

## Question 62
- [ ] Done

**How do you implement distributed Naive Bayes algorithms?**

**Answer:**

Distributed NB computes sufficient statistics (counts) on each worker, then aggregates centrally. Each worker computes local class counts and feature sums for its data partition. Master combines all counts to compute final model parameters. This is embarrassingly parallel because counts are additive.

**Architecture:**
```
Data Partition 1 → Worker 1 → Local Counts
Data Partition 2 → Worker 2 → Local Counts
Data Partition 3 → Worker 3 → Local Counts
                      ↓
              Aggregate Counts (sum)
                      ↓
               Final NB Model
```

**Implementation Concept:**
```python
def worker_compute_stats(X_partition, y_partition):
    """Compute local sufficient statistics."""
    classes = set(y_partition)
    stats = {
        'class_counts': {},
        'feature_sums': {}
    }
    for c in classes:
        mask = (y_partition == c)
        stats['class_counts'][c] = mask.sum()
        stats['feature_sums'][c] = X_partition[mask].sum(axis=0)
    return stats

def master_aggregate(all_worker_stats):
    """Combine stats from all workers."""
    global_class_counts = {}
    global_feature_sums = {}
    
    for stats in all_worker_stats:
        for c in stats['class_counts']:
            global_class_counts[c] = global_class_counts.get(c, 0) + stats['class_counts'][c]
            if c not in global_feature_sums:
                global_feature_sums[c] = stats['feature_sums'][c]
            else:
                global_feature_sums[c] += stats['feature_sums'][c]
    
    return build_nb_from_counts(global_class_counts, global_feature_sums)

def build_nb_from_counts(class_counts, feature_sums, alpha=1.0):
    """Build NB model from aggregated counts."""
    total = sum(class_counts.values())
    priors = {c: count/total for c, count in class_counts.items()}
    
    likelihoods = {}
    for c in class_counts:
        total_features = feature_sums[c].sum() + alpha * len(feature_sums[c])
        likelihoods[c] = (feature_sums[c] + alpha) / total_features
    
    return priors, likelihoods
```

---

## Question 63
- [ ] Done

**What is the role of Naive Bayes in MapReduce and Spark frameworks?**

**Answer:**

In MapReduce/Spark, NB training uses Map phase to compute local counts per partition, and Reduce phase to aggregate counts globally. Prediction broadcasts model to all workers for parallel scoring. Spark MLlib provides optimized NB implementation that handles distributed data efficiently.

**MapReduce Pattern:**
```
MAP:   (doc_id, (features, class)) → ((class, feature_id), count)
REDUCE: ((class, feature_id), [counts]) → ((class, feature_id), total_count)
```

**Spark MLlib Example:**
```python
from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark
spark = SparkSession.builder.appName("NB").getOrCreate()

# Load data
df = spark.read.csv("data.csv", header=True, inferSchema=True)

# Prepare features
assembler = VectorAssembler(inputCols=["f1", "f2", "f3"], outputCol="features")
df = assembler.transform(df)

# Split data
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Train Naive Bayes
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
model = nb.fit(train)

# Predict
predictions = model.transform(test)

# Evaluate
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")
```

**Spark Advantages:**
- In-memory caching
- Lazy evaluation
- Automatic parallelization
- Fault tolerance

---

## Question 64
- [ ] Done

**How do you handle memory optimization for large-scale Naive Bayes?**

**Answer:**

Memory optimization for large NB: (1) Use sparse matrices for high-dimensional data, (2) Feature hashing to bound vocabulary size, (3) Online learning to process data in batches, (4) Store only log-probabilities (not raw counts), (5) Prune low-frequency features, (6) Use compact data types (float32 vs float64).

**Optimization Techniques:**

**1. Sparse Matrix Storage**
```python
from scipy.sparse import csr_matrix
import numpy as np

# Dense: 1M docs × 100K words = 400GB (float32)
# Sparse: Only non-zeros stored = ~1GB
X_sparse = csr_matrix(X_dense)
```

**2. Feature Hashing (Bounded Memory)**
```python
from sklearn.feature_extraction.text import HashingVectorizer

# Fixed 2^20 features, regardless of vocabulary
vectorizer = HashingVectorizer(n_features=2**20, dtype=np.float32)
X = vectorizer.fit_transform(documents)
```

**3. Online Learning (Batch Processing)**
```python
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
for batch in data_iterator(batch_size=10000):
    X_batch, y_batch = process_batch(batch)
    nb.partial_fit(X_batch, y_batch, classes=[0, 1])
    # Previous batches not in memory
```

**4. Feature Pruning**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=50000,   # Cap vocabulary
    min_df=5,             # Remove rare words
    max_df=0.95,          # Remove too common words
    dtype=np.float32      # Smaller floats
)
```

**5. Compact Storage**
```python
# Store only what's needed for prediction
model_compact = {
    'log_priors': nb.class_log_prior_.astype(np.float32),
    'log_probs': nb.feature_log_prob_.astype(np.float32)
}
```

---

## Question 65
- [ ] Done

**What are the advances in neural Naive Bayes and deep learning integration?**

**Answer:**

Neural NB combines NB's probabilistic framework with neural network representations: (1) Use NB loss function in neural training, (2) Initialize neural network weights from NB parameters, (3) Use learned embeddings as NB features, (4) NB-weighted attention mechanisms. Research shows NB-inspired losses improve neural text classifiers.

**Key Advances:**

**1. NBSVM (NB + SVM Features)**
```python
def compute_nb_features(X, y):
    """Compute NB log-count ratios as features."""
    # Count features per class
    p = X[y == 1].sum(axis=0) + 1  # Positive class counts
    q = X[y == 0].sum(axis=0) + 1  # Negative class counts
    
    # Log ratio
    r = np.log((p / p.sum()) / (q / q.sum()))
    
    # Weight features by NB ratios
    return X.multiply(r)  # NB-weighted features for SVM/NN
```

**2. NB Initialization for Neural Networks**
```python
import torch.nn as nn

def init_from_nb(linear_layer, nb_model):
    """Initialize neural network weights from trained NB."""
    # Use log probability differences as initial weights
    weights = nb_model.feature_log_prob_[1] - nb_model.feature_log_prob_[0]
    linear_layer.weight.data = torch.tensor(weights).unsqueeze(0)
    linear_layer.bias.data = torch.tensor([nb_model.class_log_prior_[1] - nb_model.class_log_prior_[0]])
```

**3. Neural NB Layer**
```python
class NaiveBayesLayer(nn.Module):
    """Differentiable NB-like layer."""
    def __init__(self, vocab_size, n_classes):
        super().__init__()
        self.log_probs = nn.Parameter(torch.zeros(n_classes, vocab_size))
        self.log_prior = nn.Parameter(torch.zeros(n_classes))
    
    def forward(self, x):
        # x: (batch, vocab) - word counts or embeddings
        log_likelihood = torch.matmul(x, self.log_probs.t())
        return log_likelihood + self.log_prior
```

**Benefits:**
- NB regularization helps with small data
- Fast convergence
- Interpretable components

---

## Question 66
- [ ] Done

**How do you combine Naive Bayes with deep learning architectures?**

**Answer:**

Combine NB with deep learning by: (1) Use NB for feature weighting before neural network, (2) Ensemble NB predictions with neural outputs, (3) Use NB as initialization or regularization for neural classifier, (4) Hybrid architecture with NB layer parallel to neural layers. NB adds interpretability and works better with limited data.

**Approach 1: NB Feature Weighting + Neural Network**
```python
import torch
import torch.nn as nn

class NBWeightedNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_classes, nb_weights):
        super().__init__()
        self.nb_weights = torch.tensor(nb_weights, dtype=torch.float32)
        self.fc1 = nn.Linear(vocab_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Weight input by NB importance
        x_weighted = x * self.nb_weights
        h = self.relu(self.fc1(x_weighted))
        return self.fc2(h)
```

**Approach 2: Ensemble NB + Neural Network**
```python
class NBNeuralEnsemble:
    def __init__(self, nb_model, neural_model, nb_weight=0.3):
        self.nb = nb_model
        self.nn = neural_model
        self.w = nb_weight
    
    def predict_proba(self, X):
        nb_probs = self.nb.predict_proba(X)
        nn_probs = self.nn.predict_proba(X)
        # Weighted average
        return self.w * nb_probs + (1 - self.w) * nn_probs
```

**Approach 3: NB Regularization Loss**
```python
def nb_regularized_loss(predictions, targets, model, nb_model, lambda_nb=0.1):
    ce_loss = nn.CrossEntropyLoss()(predictions, targets)
    
    # Encourage weights to be close to NB weights
    nb_weights = torch.tensor(nb_model.feature_log_prob_[1] - nb_model.feature_log_prob_[0])
    reg_loss = torch.norm(model.fc1.weight - nb_weights)
    
    return ce_loss + lambda_nb * reg_loss
```

---

## Question 67
- [ ] Done

**What is the role of Naive Bayes in transfer learning?**

**Answer:**

NB in transfer learning: (1) Initialize target model with source NB parameters, (2) Use source NB predictions as features for target task, (3) Transfer learned priors/likelihoods and fine-tune with target data, (4) Use NB to select transferable features. NB's simple structure makes parameter transfer straightforward.

**Transfer Learning Scenarios:**

**1. Parameter Transfer (Source → Target)**
```python
def transfer_nb(source_nb, target_X, target_y, blend=0.7):
    """
    Transfer NB from source domain to target domain.
    Blend source and target statistics.
    """
    target_nb = MultinomialNB()
    target_nb.fit(target_X, target_y)
    
    # Blend parameters
    transferred_nb = MultinomialNB()
    transferred_nb.classes_ = target_nb.classes_
    transferred_nb.class_log_prior_ = (
        blend * source_nb.class_log_prior_ + 
        (1 - blend) * target_nb.class_log_prior_
    )
    transferred_nb.feature_log_prob_ = (
        blend * source_nb.feature_log_prob_ + 
        (1 - blend) * target_nb.feature_log_prob_
    )
    
    return transferred_nb
```

**2. NB Features for Transfer**
```python
def nb_transfer_features(source_nb, X):
    """
    Use source NB predictions as features for target task.
    """
    # Source domain predictions as features
    source_probs = source_nb.predict_proba(X)
    source_log_probs = source_nb.predict_log_proba(X)
    
    # Concatenate with original features
    transfer_features = np.hstack([X.toarray(), source_probs, source_log_probs])
    return transfer_features
```

**3. Cross-Domain Text Classification**
```python
# Example: Transfer from product reviews to movie reviews
source_nb = MultinomialNB()
source_nb.fit(product_review_X, product_review_y)  # Source domain

# Transfer to movie reviews with limited labels
transferred_model = transfer_nb(source_nb, movie_X_small, movie_y_small, blend=0.5)
```

**When Transfer Helps:**
- Limited target domain data
- Similar feature distributions
- Related classification tasks

---

## Question 68
- [ ] Done

**How do you handle domain adaptation with Naive Bayes classifiers?**

**Answer:**

Domain adaptation addresses distribution shift between source and target domains. For NB: (1) Re-estimate priors using target domain frequencies, (2) Use EM to adapt using unlabeled target data, (3) Instance weighting - weight source samples by similarity to target, (4) Feature transformation to align distributions.

**Problem:**
```
Source domain: Formal news articles (labeled)
Target domain: Social media posts (unlabeled)
Shift: Different vocabulary, style, class distribution
```

**Approach 1: Prior Adaptation**
```python
def adapt_priors(source_nb, target_X_unlabeled):
    """
    Adapt class priors to target domain using predictions.
    """
    # Predict target domain with source model
    target_preds = source_nb.predict(target_X_unlabeled)
    
    # Estimate target priors from predictions
    target_priors = np.bincount(target_preds) / len(target_preds)
    
    # Update model
    source_nb.class_log_prior_ = np.log(target_priors)
    return source_nb
```

**Approach 2: EM Adaptation**
```python
def em_domain_adaptation(source_nb, target_X_unlabeled, n_iter=10):
    """
    Adapt NB using EM on unlabeled target data.
    """
    model = clone(source_nb)
    
    for _ in range(n_iter):
        # E-step: soft labels from current model
        soft_labels = model.predict_proba(target_X_unlabeled)
        
        # M-step: re-estimate parameters (simplified)
        for c in range(len(model.classes_)):
            weights = soft_labels[:, c]
            # Update feature probabilities weighted by soft labels
            weighted_counts = (target_X_unlabeled.T @ weights) + 1
            model.feature_log_prob_[c] = np.log(weighted_counts / weighted_counts.sum())
    
    return model
```

**Approach 3: Instance Weighting**
```python
def instance_weighted_nb(source_X, source_y, target_X_unlabeled):
    """
    Weight source instances by similarity to target domain.
    """
    # Train density estimator on target
    from sklearn.neighbors import KernelDensity
    kde_target = KernelDensity().fit(target_X_unlabeled)
    kde_source = KernelDensity().fit(source_X)
    
    # Weight = P_target(x) / P_source(x)
    weights = np.exp(kde_target.score_samples(source_X) - kde_source.score_samples(source_X))
    weights = np.clip(weights, 0.1, 10)  # Clip extreme weights
    
    # Train weighted NB (use sample_weight in custom implementation)
    return weights
```

---

## Question 69
- [ ] Done

**What are the considerations for Naive Bayes model deployment?**

**Answer:**

Deployment considerations: (1) Serialize model and vectorizer together, (2) Handle unknown words/features gracefully, (3) Monitor prediction latency and throughput, (4) Log predictions for debugging and retraining, (5) Version control models, (6) Handle input validation and preprocessing consistency.

**Deployment Checklist:**

**1. Serialize Model + Preprocessing**
```python
import joblib
from sklearn.pipeline import Pipeline

# Always save as pipeline (preprocessing + model together)
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])
pipeline.fit(X_train, y_train)

# Save entire pipeline
joblib.dump(pipeline, 'nb_pipeline.joblib')

# Load for inference
pipeline = joblib.load('nb_pipeline.joblib')
```

**2. Handle Unknown Features**
```python
class ProductionNBClassifier:
    def __init__(self, model_path):
        self.pipeline = joblib.load(model_path)
    
    def predict(self, text):
        try:
            # Vectorizer handles unknown words (ignores them)
            prediction = self.pipeline.predict([text])[0]
            proba = self.pipeline.predict_proba([text])[0]
            return {
                'class': prediction,
                'confidence': float(max(proba)),
                'probabilities': {c: float(p) for c, p in zip(self.pipeline.classes_, proba)}
            }
        except Exception as e:
            return {'error': str(e), 'class': 'unknown'}
```

**3. Monitoring & Logging**
```python
import logging
import time

logger = logging.getLogger('nb_classifier')

def predict_with_logging(pipeline, text):
    start = time.time()
    result = pipeline.predict([text])[0]
    latency = time.time() - start
    
    logger.info(f"Input: {text[:100]}... | Prediction: {result} | Latency: {latency:.3f}s")
    return result
```

**4. Input Validation**
```python
def validate_input(text):
    if not isinstance(text, str):
        raise ValueError("Input must be string")
    if len(text) < 3:
        raise ValueError("Input too short")
    if len(text) > 10000:
        text = text[:10000]  # Truncate
    return text.strip().lower()
```

---

## Question 70
- [ ] Done

**How do you monitor and maintain Naive Bayes models in production?**

**Answer:**

Monitor: (1) Prediction distribution - detect class imbalance drift, (2) Confidence scores - flag low-confidence predictions, (3) Feature distribution - detect input drift, (4) Performance metrics - track accuracy on sampled/labeled data, (5) Latency and throughput, (6) Error rates and edge cases.

**Monitoring System:**
```python
import numpy as np
from collections import defaultdict
import time

class NBMonitor:
    def __init__(self, baseline_distribution):
        self.baseline = baseline_distribution
        self.predictions = []
        self.confidences = []
        self.latencies = []
        self.window_size = 1000
    
    def log_prediction(self, prediction, confidence, latency):
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        self.latencies.append(latency)
        
        # Rolling window
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            self.confidences.pop(0)
            self.latencies.pop(0)
    
    def check_class_drift(self, threshold=0.1):
        """Detect if prediction distribution drifted from baseline."""
        current_dist = np.bincount(self.predictions, minlength=len(self.baseline)) / len(self.predictions)
        drift = np.abs(current_dist - self.baseline).max()
        if drift > threshold:
            return f"WARNING: Class drift detected ({drift:.2%})"
        return None
    
    def check_confidence(self, min_confidence=0.6):
        """Flag low confidence predictions."""
        low_conf_rate = np.mean(np.array(self.confidences) < min_confidence)
        if low_conf_rate > 0.2:
            return f"WARNING: {low_conf_rate:.1%} predictions below {min_confidence} confidence"
        return None
    
    def get_stats(self):
        return {
            'avg_confidence': np.mean(self.confidences),
            'avg_latency_ms': np.mean(self.latencies) * 1000,
            'p99_latency_ms': np.percentile(self.latencies, 99) * 1000,
            'predictions_per_class': dict(zip(*np.unique(self.predictions, return_counts=True)))
        }
```

**Alerts to Set:**
- Accuracy drops below threshold
- Latency exceeds SLA
- Class distribution shifts significantly
- Too many low-confidence predictions
- Input feature values out of expected range

---

## Question 71
- [ ] Done

**What is A/B testing and model versioning for Naive Bayes?**

**Answer:**

A/B testing compares model versions by serving different models to different users and measuring outcomes. Model versioning tracks changes in training data, parameters, and code. For NB: test new smoothing values, feature sets, or retraining frequency. Use traffic splitting and statistical tests to measure improvement.

**A/B Testing Setup:**
```python
import random
import hashlib

class ABTestRouter:
    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.models = {'A': model_a, 'B': model_b}
        self.split_ratio = split_ratio
        self.results = {'A': [], 'B': []}
    
    def get_variant(self, user_id):
        """Deterministic assignment based on user_id."""
        hash_val = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        return 'A' if (hash_val % 100) < (self.split_ratio * 100) else 'B'
    
    def predict(self, user_id, features):
        variant = self.get_variant(user_id)
        prediction = self.models[variant].predict(features)
        return prediction, variant
    
    def log_outcome(self, variant, outcome):
        self.results[variant].append(outcome)
    
    def analyze(self):
        """Compare variants statistically."""
        from scipy.stats import ttest_ind
        a_mean = np.mean(self.results['A'])
        b_mean = np.mean(self.results['B'])
        stat, pvalue = ttest_ind(self.results['A'], self.results['B'])
        return {
            'A_mean': a_mean,
            'B_mean': b_mean,
            'improvement': (b_mean - a_mean) / a_mean,
            'p_value': pvalue,
            'significant': pvalue < 0.05
        }
```

**Model Versioning:**
```python
import joblib
import json
from datetime import datetime

def save_versioned_model(model, metadata):
    version = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    joblib.dump(model, f'models/nb_v{version}.joblib')
    
    # Save metadata
    metadata.update({
        'version': version,
        'created_at': str(datetime.now()),
        'alpha': model.alpha,
        'n_classes': len(model.classes_),
        'n_features': model.feature_log_prob_.shape[1]
    })
    with open(f'models/nb_v{version}_meta.json', 'w') as f:
        json.dump(metadata, f)
```

---

## Question 72
- [ ] Done

**How do you handle real-time inference with Naive Bayes?**

**Answer:**

NB is ideal for real-time inference due to O(d) prediction complexity. Optimize: (1) Pre-compute and cache log probabilities, (2) Use sparse operations for text, (3) Batch predictions when possible, (4) Keep model in memory, (5) Use efficient serialization. Typical latency: sub-millisecond.

**Real-Time Optimization:**
```python
import numpy as np
from scipy.sparse import csr_matrix
import time

class FastNBInference:
    """
    Optimized NB for real-time inference.
    Pre-computes everything possible.
    """
    def __init__(self, sklearn_nb, vectorizer):
        # Pre-compute and store as contiguous arrays
        self.log_priors = np.ascontiguousarray(sklearn_nb.class_log_prior_)
        self.log_probs = np.ascontiguousarray(sklearn_nb.feature_log_prob_)
        self.classes = sklearn_nb.classes_
        self.vectorizer = vectorizer
        
        # Precompute for sparse dot product
        self.log_probs_T = np.ascontiguousarray(self.log_probs.T)
    
    def predict_single(self, text):
        """Predict single sample - optimized path."""
        # Vectorize
        X = self.vectorizer.transform([text])
        
        # Compute log posteriors: X @ log_probs.T + log_priors
        log_posteriors = X @ self.log_probs_T + self.log_priors
        
        # Return class with max posterior
        return self.classes[log_posteriors.argmax()]
    
    def predict_batch(self, texts):
        """Predict batch - vectorized."""
        X = self.vectorizer.transform(texts)
        log_posteriors = X @ self.log_probs_T + self.log_priors
        return self.classes[log_posteriors.argmax(axis=1)]

# Benchmark
def benchmark(model, texts, n_iterations=1000):
    start = time.time()
    for _ in range(n_iterations):
        for text in texts:
            model.predict_single(text)
    elapsed = time.time() - start
    print(f"Avg latency: {elapsed/n_iterations/len(texts)*1000:.3f} ms per prediction")
```

**Deployment Options:**
- In-process: Model loaded in application memory
- Microservice: Dedicated prediction service (FastAPI/Flask)
- Serverless: AWS Lambda with model in container

---

## Question 73
- [ ] Done

**What are the considerations for Naive Bayes in edge computing?**

**Answer:**

Edge computing runs models on devices (phones, IoT sensors) rather than cloud. NB is well-suited due to: (1) Small model size (just parameters), (2) Low computation (multiplications only), (3) No GPU required, (4) Deterministic latency. Considerations: limited memory, no internet, power efficiency.

**Edge Deployment Advantages of NB:**

| Factor | NB | Deep Learning |
|--------|------|---------------|
| Model size | KB-MB | MB-GB |
| RAM needed | Low | High |
| CPU only | Yes | Often needs GPU |
| Latency | Microseconds | Milliseconds |
| Power | Low | High |

**Minimal NB Implementation for Edge:**
```python
import numpy as np

class EdgeNB:
    """
    Minimal NB for edge devices.
    No sklearn dependency.
    """
    def __init__(self, log_priors, log_probs, vocab):
        self.log_priors = np.array(log_priors, dtype=np.float32)
        self.log_probs = np.array(log_probs, dtype=np.float32)
        self.vocab = {w: i for i, w in enumerate(vocab)}
    
    def predict(self, tokens):
        """Predict from list of tokens."""
        log_posteriors = self.log_priors.copy()
        
        for token in tokens:
            if token in self.vocab:
                idx = self.vocab[token]
                log_posteriors += self.log_probs[:, idx]
        
        return int(np.argmax(log_posteriors))
    
    def save_minimal(self, path):
        """Save in minimal format."""
        np.savez_compressed(path,
            priors=self.log_priors,
            probs=self.log_probs,
            vocab=list(self.vocab.keys())
        )
    
    @classmethod
    def load_minimal(cls, path):
        data = np.load(path, allow_pickle=True)
        return cls(data['priors'], data['probs'], data['vocab'])
```

**Use Cases:**
- Spam filtering on phone (offline)
- Sensor anomaly detection on IoT
- Text classification on embedded devices

---

## Question 74
- [ ] Done

**How do you implement Naive Bayes for IoT and sensor data classification?**

**Answer:**

For IoT/sensor data, use Gaussian NB with continuous features (temperature, pressure, vibration). Handle streaming data with online learning (`partial_fit`). Discretize features if distribution is non-Gaussian. Common tasks: anomaly detection, activity recognition, predictive maintenance.

**IoT Classification Pipeline:**
```python
from sklearn.naive_bayes import GaussianNB
import numpy as np

class IoTSensorClassifier:
    """
    Classify sensor readings in real-time.
    Example: Predict machine state from vibration sensors.
    """
    def __init__(self, window_size=10):
        self.model = GaussianNB()
        self.window_size = window_size
        self.buffer = []
        self.is_fitted = False
    
    def extract_features(self, readings):
        """Extract statistical features from sensor window."""
        readings = np.array(readings)
        return np.array([
            readings.mean(),
            readings.std(),
            readings.min(),
            readings.max(),
            np.percentile(readings, 25),
            np.percentile(readings, 75),
            np.abs(np.diff(readings)).mean()  # Rate of change
        ])
    
    def process_reading(self, reading):
        """Process single sensor reading."""
        self.buffer.append(reading)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        if len(self.buffer) == self.window_size:
            features = self.extract_features(self.buffer)
            if self.is_fitted:
                return self.model.predict([features])[0]
        return None
    
    def train_online(self, reading, label):
        """Online training from labeled sensor data."""
        self.buffer.append(reading)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        if len(self.buffer) == self.window_size:
            features = self.extract_features(self.buffer)
            classes = [0, 1, 2]  # normal, warning, fault
            self.model.partial_fit([features], [label], classes=classes)
            self.is_fitted = True

# Example: Vibration monitoring
classifier = IoTSensorClassifier(window_size=100)

# Simulate sensor stream
for reading in sensor_stream:
    state = classifier.process_reading(reading)
    if state == 2:  # Fault detected
        alert("Machine fault detected!")
```

**Key Considerations:**
- Feature engineering from raw sensor data
- Handling missing/noisy readings
- Real-time constraints
- Battery/power limitations

---

## Question 75
- [ ] Done

**What are the fairness and bias considerations in Naive Bayes?**

**Answer:**

NB can learn and amplify biases from training data. Issues: (1) Imbalanced class representation, (2) Proxy features correlated with protected attributes (race, gender), (3) Historical bias in labels. Mitigation: audit for disparate impact, remove/modify biased features, use fairness constraints, balance training data.

**Types of Bias in NB:**

| Bias Type | Example | Detection |
|-----------|---------|-----------|  
| Label bias | Historical hiring discrimination | Audit outcomes by group |
| Feature bias | Zip code proxies for race | Feature correlation analysis |
| Representation bias | Underrepresented groups in training | Check training distribution |

**Fairness Audit:**
```python
from sklearn.metrics import confusion_matrix

def fairness_audit(y_true, y_pred, protected_attribute):
    """
    Audit model for disparate impact across groups.
    """
    groups = np.unique(protected_attribute)
    results = {}
    
    for group in groups:
        mask = (protected_attribute == group)
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        
        # Positive prediction rate
        positive_rate = y_pred_g.mean()
        
        # True positive rate (recall)
        tpr = (y_pred_g[y_true_g == 1] == 1).mean() if (y_true_g == 1).sum() > 0 else 0
        
        # False positive rate
        fpr = (y_pred_g[y_true_g == 0] == 1).mean() if (y_true_g == 0).sum() > 0 else 0
        
        results[group] = {
            'positive_rate': positive_rate,
            'true_positive_rate': tpr,
            'false_positive_rate': fpr
        }
    
    # Check disparate impact (80% rule)
    rates = [r['positive_rate'] for r in results.values()]
    disparate_impact = min(rates) / max(rates) if max(rates) > 0 else 0
    
    return {
        'group_metrics': results,
        'disparate_impact': disparate_impact,
        'fair': disparate_impact >= 0.8
    }
```

**Mitigation Strategies:**
- Remove protected attributes from features
- Reweight/resample to balance groups
- Post-hoc threshold adjustment per group
- Use fairness-aware smoothing

---

## Question 76
- [ ] Done

**How do you address algorithmic bias in Naive Bayes classifiers?**

**Answer:**

Address bias through: (1) Pre-processing - balance training data, remove biased features, (2) In-processing - modify learning algorithm with fairness constraints, (3) Post-processing - adjust thresholds per group to equalize outcomes. For NB specifically, can modify priors and likelihoods to reduce disparate impact.

**Pre-processing: Remove Biased Features**
```python
def identify_biased_features(X, protected_attr, threshold=0.3):
    """
    Find features highly correlated with protected attribute.
    """
    from scipy.stats import spearmanr
    
    biased_features = []
    for j in range(X.shape[1]):
        corr, _ = spearmanr(X[:, j], protected_attr)
        if abs(corr) > threshold:
            biased_features.append(j)
    return biased_features

# Remove biased features
biased_idx = identify_biased_features(X, gender)
X_fair = np.delete(X, biased_idx, axis=1)
```

**In-processing: Balanced Priors**
```python
def train_fair_nb(X, y, protected_attr):
    """
    Train NB with group-balanced priors.
    """
    from sklearn.naive_bayes import MultinomialNB
    
    # Compute balanced prior (equal across groups)
    groups = np.unique(protected_attr)
    balanced_prior = [0.5, 0.5]  # Force equal class priors
    
    nb = MultinomialNB(class_prior=balanced_prior)
    nb.fit(X, y)
    return nb
```

**Post-processing: Group-Specific Thresholds**
```python
def fair_predict(nb, X, protected_attr, group_thresholds):
    """
    Apply different thresholds per group to equalize positive rates.
    """
    probs = nb.predict_proba(X)[:, 1]
    predictions = np.zeros(len(X), dtype=int)
    
    for group, threshold in group_thresholds.items():
        mask = (protected_attr == group)
        predictions[mask] = (probs[mask] > threshold).astype(int)
    
    return predictions

# Example: Lower threshold for disadvantaged group
group_thresholds = {'group_A': 0.5, 'group_B': 0.4}
fair_preds = fair_predict(nb, X_test, protected_test, group_thresholds)
```

---

## Question 77
- [ ] Done

**What are the ethical implications of using Naive Bayes in decision-making?**

**Answer:**

Ethical considerations: (1) NB's simplicity may oversimplify complex decisions affecting people, (2) Independence assumption may miss important feature interactions, (3) Probability outputs may give false confidence, (4) Need human oversight for high-stakes decisions. Advantages: NB is interpretable, auditable, and explainable - supporting accountability.

**Ethical Framework for NB Deployment:**

| Principle | Consideration | NB Implication |
|-----------|---------------|----------------|
| Transparency | Explain decisions | NB is inherently explainable |
| Accountability | Track who decides | Log predictions + feature contributions |
| Fairness | Equal treatment | Audit for bias across groups |
| Privacy | Protect data | NB needs only aggregates, not individual data |
| Human oversight | Human in loop | Use NB as recommendation, not final decision |

**Ethical Deployment Guidelines:**
```python
class EthicalNBClassifier:
    def __init__(self, model, high_stakes=True):
        self.model = model
        self.high_stakes = high_stakes
        self.decision_log = []
    
    def predict_with_explanation(self, X, feature_names):
        """Provide prediction with full explanation."""
        prediction = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        
        # Explain top contributing features
        if hasattr(self.model, 'feature_log_prob_'):
            log_ratios = self.model.feature_log_prob_[1] - self.model.feature_log_prob_[0]
            contributions = X[0] * log_ratios
            top_features = np.argsort(np.abs(contributions))[-5:]
            explanation = [(feature_names[i], contributions[i]) for i in top_features]
        else:
            explanation = []
        
        result = {
            'prediction': prediction,
            'confidence': float(max(proba)),
            'explanation': explanation,
            'human_review_recommended': proba.max() < 0.7 or self.high_stakes
        }
        
        self.decision_log.append(result)
        return result
```

**High-Stakes Domains (Require Extra Care):**
- Healthcare diagnosis
- Credit/loan decisions
- Hiring and employment
- Criminal justice

---

## Question 78
- [ ] Done

**How do you implement adversarial robustness for Naive Bayes?**

**Answer:**

Adversarial attacks craft inputs to fool classifiers. For NB text classifiers, attackers insert/remove words to flip predictions. Defenses: (1) Robust features that ignore single-word changes, (2) Input sanitization (spell check, normalization), (3) Ensemble multiple models, (4) Detect anomalous inputs, (5) Use n-grams instead of single words.

**Attack Example (Text):**
```
Original:   "Free money! Click here to claim prize!" → SPAM
Adversarial: "Free money! Click here to claim prize! meeting agenda" → NOT SPAM
(Attacker adds normal words to flip prediction)
```

**Defense 1: Robust Feature Engineering**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Use n-grams - harder to attack with single words
robust_vectorizer = TfidfVectorizer(
    ngram_range=(2, 3),  # Bigrams and trigrams
    min_df=5,            # Ignore rare n-grams
    max_df=0.9,          # Ignore too common
    sublinear_tf=True    # Dampen high counts
)
```

**Defense 2: Input Sanitization**
```python
import re

def sanitize_input(text):
    """Clean input before classification."""
    # Remove excessive punctuation
    text = re.sub(r'[!@#$%^&*()]{2,}', ' ', text)
    # Remove repeated characters
    text = re.sub(r'(.){3,}', r'', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    # Optional: spell check
    return text.lower()
```

**Defense 3: Anomaly Detection**
```python
def detect_adversarial(text, model, vocab, threshold=-50):
    """
    Flag inputs with unusual word distributions.
    """
    X = vectorizer.transform([text])
    log_prob = model.predict_log_proba(X).max()
    
    # Very low probability = unusual input
    if log_prob < threshold:
        return True, "Potentially adversarial input"
    return False, None
```

**Defense 4: Ensemble Defense**
```python
def ensemble_predict(models, X):
    """Require agreement across models."""
    predictions = [m.predict(X)[0] for m in models]
    # Majority vote
    from collections import Counter
    return Counter(predictions).most_common(1)[0][0]
```

---

## Question 79
- [ ] Done

**What are the security considerations for Naive Bayes models?**

**Answer:**

Security concerns: (1) Model extraction - attackers query model to steal it, (2) Training data leakage - model memorizes sensitive data, (3) Adversarial inputs - craft inputs to evade detection, (4) Data poisoning - inject malicious training data. NB's transparency makes it easier to attack but also easier to audit.

**Security Threats:**

| Threat | Description | Mitigation |
|--------|-------------|------------|
| Model Extraction | Repeated queries to clone model | Rate limiting, query monitoring |
| Membership Inference | Detect if sample was in training | Differential privacy |
| Data Poisoning | Inject bad training data | Data validation, anomaly detection |
| Evasion | Adversarial inputs | Input sanitization, robust features |

**Model Extraction Defense:**
```python
from collections import defaultdict
import time

class SecureNBEndpoint:
    def __init__(self, model, rate_limit=100):
        self.model = model
        self.rate_limit = rate_limit
        self.query_counts = defaultdict(list)
    
    def predict(self, user_id, X):
        # Rate limiting
        now = time.time()
        self.query_counts[user_id] = [
            t for t in self.query_counts[user_id] if now - t < 3600
        ]
        
        if len(self.query_counts[user_id]) > self.rate_limit:
            raise Exception("Rate limit exceeded")
        
        self.query_counts[user_id].append(now)
        
        # Return only class, not probabilities (reduces information leakage)
        return self.model.predict(X)[0]
```

**Membership Inference Defense:**
```python
def train_with_dp(X, y, epsilon=1.0):
    """
    Train with differential privacy to prevent membership inference.
    """
    from sklearn.naive_bayes import MultinomialNB
    import numpy as np
    
    nb = MultinomialNB()
    nb.fit(X, y)
    
    # Add noise to parameters
    noise_scale = 1.0 / epsilon
    nb.feature_log_prob_ += np.random.laplace(0, noise_scale, nb.feature_log_prob_.shape)
    nb.class_log_prior_ += np.random.laplace(0, noise_scale, nb.class_log_prior_.shape)
    
    return nb
```

---

## Question 80
- [ ] Done

**How do you handle data poisoning attacks on Naive Bayes?**

**Answer:**

Data poisoning injects malicious samples into training data to manipulate model behavior. For NB: attackers add samples with specific words labeled incorrectly to shift feature probabilities. Defenses: (1) Data validation and anomaly detection, (2) Robust statistics, (3) Clean-label detection, (4) Ensemble training on data subsets.

**Attack Example:**
```
Attacker wants "discount" to not trigger spam filter.
Injects many ham emails containing "discount" → P(discount|ham) increases.
Result: Spam with "discount" evades filter.
```

**Defense 1: Data Validation**
```python
from sklearn.ensemble import IsolationForest

def detect_poisoned_samples(X, y, contamination=0.05):
    """
    Detect outliers that may be poisoned.
    """
    poisoned = []
    for c in np.unique(y):
        X_c = X[y == c]
        indices_c = np.where(y == c)[0]
        
        # Find outliers within each class
        iso = IsolationForest(contamination=contamination)
        outlier_labels = iso.fit_predict(X_c)
        
        poisoned.extend(indices_c[outlier_labels == -1])
    
    return poisoned

# Remove suspicious samples before training
poisoned_idx = detect_poisoned_samples(X_train, y_train)
X_clean = np.delete(X_train, poisoned_idx, axis=0)
y_clean = np.delete(y_train, poisoned_idx)
```

**Defense 2: Robust Training with Subsampling**
```python
def robust_ensemble_nb(X, y, n_models=10, subsample_ratio=0.7):
    """
    Train ensemble on random subsets - poisoned samples less likely in all.
    """
    models = []
    n_samples = len(y)
    
    for _ in range(n_models):
        # Random subsample
        idx = np.random.choice(n_samples, int(n_samples * subsample_ratio), replace=False)
        
        nb = MultinomialNB()
        nb.fit(X[idx], y[idx])
        models.append(nb)
    
    return models

def ensemble_predict(models, X):
    predictions = np.array([m.predict(X) for m in models])
    # Majority vote
    return np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, predictions)
```

**Defense 3: Sanitize Feature Counts**
```python
def cap_feature_influence(nb, max_ratio=10):
    """
    Limit how much any feature can influence prediction.
    """
    log_probs = nb.feature_log_prob_
    # Cap the difference between classes
    mean_log_prob = log_probs.mean(axis=0)
    for c in range(len(nb.classes_)):
        diff = log_probs[c] - mean_log_prob
        diff = np.clip(diff, -np.log(max_ratio), np.log(max_ratio))
        log_probs[c] = mean_log_prob + diff
    nb.feature_log_prob_ = log_probs
    return nb
```

---

## Question 81
- [ ] Done

**What is the role of Naive Bayes in AutoML and automated model selection?**

**Answer:**

In AutoML, NB serves as: (1) Fast baseline - quick benchmark for comparison, (2) Initial candidate in model search due to low training cost, (3) Ensemble member providing diversity, (4) Feature preprocessor (NB-weighted features). AutoML tools include NB in search space because it's fast and often competitive for text/high-dimensional data.

**NB in AutoML Pipelines:**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

def simple_automl(X, y, task_type='text'):
    """
    Simple AutoML including NB variants.
    """
    if task_type == 'text':
        candidates = [
            ('multinomial', MultinomialNB(), {'alpha': [0.01, 0.1, 1.0, 10.0]}),
            ('bernoulli', BernoulliNB(), {'alpha': [0.01, 0.1, 1.0]}),
        ]
    else:
        candidates = [
            ('gaussian', GaussianNB(), {'var_smoothing': [1e-9, 1e-7, 1e-5]}),
        ]
    
    best_score = 0
    best_model = None
    
    for name, model, params in candidates:
        grid = GridSearchCV(model, params, cv=5, scoring='f1_weighted')
        grid.fit(X, y)
        
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            print(f"New best: {name} with score {best_score:.4f}")
    
    return best_model
```

**Why AutoML Includes NB:**
- Training in milliseconds (can try many configs)
- Few hyperparameters (just alpha)
- Works with sparse high-dimensional data
- Provides interpretable baseline
- Good for quick prototyping

**Auto-sklearn / TPOT Integration:**
```python
# Auto-sklearn automatically considers NB
from autosklearn.classification import AutoSklearnClassifier

automl = AutoSklearnClassifier(time_left_for_this_task=300)
automl.fit(X_train, y_train)  # May select NB if appropriate
print(automl.leaderboard())  # Shows NB if it performed well
```

---

## Question 82
- [ ] Done

**How do you implement hyperparameter optimization for Naive Bayes?**

**Answer:**

NB has few hyperparameters: (1) alpha (smoothing) - most important, controls regularization, (2) fit_prior - whether to learn class priors, (3) class_prior - manually set priors. Use grid search or Bayesian optimization. Alpha typically ranges from 0.001 to 10, with 1.0 (Laplace) as default.

**Hyperparameter Search:**
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform
import numpy as np

# Grid Search
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    'fit_prior': [True, False]
}

grid_search = GridSearchCV(
    MultinomialNB(),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

**Bayesian Optimization:**
```python
from skopt import BayesSearchCV
from skopt.space import Real, Categorical

search_space = {
    'alpha': Real(0.001, 10.0, prior='log-uniform'),
    'fit_prior': Categorical([True, False])
}

bayes_search = BayesSearchCV(
    MultinomialNB(),
    search_space,
    n_iter=30,
    cv=5,
    scoring='f1_weighted'
)
bayes_search.fit(X_train, y_train)
```

**With Preprocessing Pipeline:**
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

param_grid = {
    'tfidf__max_features': [5000, 10000, 20000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'nb__alpha': [0.01, 0.1, 1.0]
}

grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(texts, labels)
```

---

## Question 83
- [ ] Done

**What are the emerging research directions in Naive Bayes algorithms?**

**Answer:**

Research directions: (1) Neural-NB hybrids combining probabilistic reasoning with deep learning, (2) Robust NB for adversarial settings, (3) Federated/privacy-preserving NB, (4) NB for streaming and concept drift, (5) Relaxing independence assumption (TAN, selective NB), (6) NB for few-shot learning, (7) Calibrated NB for reliable uncertainty.

**Key Research Areas:**

| Area | Focus | Example |
|------|-------|--------|
| Neural-NB | Combine with embeddings | NB loss in transformer fine-tuning |
| Robust NB | Adversarial defense | Certified robustness bounds |
| Privacy NB | Differential privacy | Federated text classification |
| Streaming NB | Concept drift | Adaptive forgetting factor |
| Few-shot NB | Limited labels | Meta-learning priors |

**1. NB with Pre-trained Embeddings:**
```python
# Research direction: Use NB on top of BERT embeddings
from transformers import BertModel, BertTokenizer

def nb_on_embeddings(texts, labels):
    # Get BERT embeddings
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    
    embeddings = []  # Extract [CLS] embeddings
    # ... embedding extraction
    
    # Apply Gaussian NB on embeddings
    nb = GaussianNB()
    nb.fit(embeddings, labels)
    return nb
```

**2. Selective Naive Bayes:**
```python
# Research: Select which independence assumptions to keep
def selective_nb(X, y):
    """
    Learn which features should remain independent vs. modeled jointly.
    """
    # Find highly correlated feature pairs
    # Model them jointly, keep rest independent
    pass
```

**3. Few-Shot NB:**
```python
def few_shot_nb(support_X, support_y, query_X, prior_model=None):
    """
    NB with meta-learned priors for few-shot learning.
    """
    nb = MultinomialNB()
    if prior_model:
        # Initialize from meta-learned priors
        nb.class_prior_ = prior_model.class_prior_
    nb.fit(support_X, support_y)
    return nb.predict(query_X)
```

---

## Question 84
- [ ] Done

**How do you implement Naive Bayes for multi-modal data classification?**

**Answer:**

Multi-modal data combines different types (text + images, audio + video). For NB: (1) Extract features from each modality, (2) Concatenate or train separate NB per modality, (3) Combine predictions via product of likelihoods or ensemble voting. Key: normalize features across modalities.

**Approach 1: Feature Concatenation**
```python
from sklearn.naive_bayes import GaussianNB
import numpy as np

class MultiModalNB:
    """
    NB for multi-modal data (e.g., text + image + metadata).
    """
    def __init__(self):
        self.model = GaussianNB()
        self.scalers = {}
    
    def extract_features(self, text_features, image_features, metadata):
        """
        Combine features from different modalities.
        """
        from sklearn.preprocessing import StandardScaler
        
        # Normalize each modality
        text_scaled = StandardScaler().fit_transform(text_features)
        image_scaled = StandardScaler().fit_transform(image_features)
        meta_scaled = StandardScaler().fit_transform(metadata)
        
        # Concatenate
        return np.hstack([text_scaled, image_scaled, meta_scaled])
    
    def fit(self, text_features, image_features, metadata, y):
        X_combined = self.extract_features(text_features, image_features, metadata)
        self.model.fit(X_combined, y)
    
    def predict(self, text_features, image_features, metadata):
        X_combined = self.extract_features(text_features, image_features, metadata)
        return self.model.predict(X_combined)
```

**Approach 2: Late Fusion (Separate Models)**
```python
class LateFusionNB:
    """
    Train separate NB per modality, combine predictions.
    """
    def __init__(self):
        self.text_nb = MultinomialNB()  # For text
        self.image_nb = GaussianNB()    # For image features
        self.weights = [0.6, 0.4]       # Modality weights
    
    def fit(self, text_X, image_X, y):
        self.text_nb.fit(text_X, y)
        self.image_nb.fit(image_X, y)
    
    def predict_proba(self, text_X, image_X):
        text_probs = self.text_nb.predict_proba(text_X)
        image_probs = self.image_nb.predict_proba(image_X)
        
        # Weighted average
        combined = self.weights[0] * text_probs + self.weights[1] * image_probs
        return combined
    
    def predict(self, text_X, image_X):
        probs = self.predict_proba(text_X, image_X)
        return self.text_nb.classes_[probs.argmax(axis=1)]
```

---

## Question 85
- [ ] Done

**What is the relationship between Naive Bayes and probabilistic graphical models?**

**Answer:**

Naive Bayes is a simple Bayesian Network (directed graphical model) where the class node is parent of all feature nodes, with no edges between features. This encodes the conditional independence assumption. More complex Bayesian networks (TAN, BAN) add edges between features to relax independence.

**Graphical Model View:**
```
Naive Bayes Structure:

         Y (Class)
       / | \ \
      /  |  \ \
     v   v   v  v
    X1  X2  X3  X4  (Features)

Encodes: X1 ⊥ X2 ⊥ X3 ⊥ X4 | Y
(All features independent given class)
```

**Tree-Augmented Naive Bayes (TAN):**
```
         Y (Class)
       / | \ \
      v  v  v  v
     X1→X2 X3→X4

Encodes: X2 depends on X1 (given Y)
         X4 depends on X3 (given Y)
```

**Comparison with Other PGMs:**
```python
# Naive Bayes as Bayesian Network
class BayesianNetworkNB:
    """
    NB viewed as Bayesian Network.
    """
    def __init__(self):
        # CPT: P(Y)
        self.prior = {}
        # CPT: P(Xi | Y) for each feature
        self.cpts = {}
    
    def fit(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        self.prior = dict(zip(classes, counts / len(y)))
        
        # Learn P(Xi | Y) for each feature
        for j in range(X.shape[1]):
            self.cpts[j] = {}
            for c in classes:
                X_c = X[y == c, j]
                # Store distribution parameters
                self.cpts[j][c] = {'mean': X_c.mean(), 'var': X_c.var() + 1e-9}
    
    def inference(self, x):
        """Compute P(Y | X=x) using Bayes rule."""
        posteriors = {}
        for c in self.prior:
            # P(Y=c) * Product of P(Xi | Y=c)
            log_prob = np.log(self.prior[c])
            for j, val in enumerate(x):
                mu, var = self.cpts[j][c]['mean'], self.cpts[j][c]['var']
                log_prob += -0.5 * np.log(2 * np.pi * var) - (val - mu)**2 / (2 * var)
            posteriors[c] = log_prob
        return posteriors
```

**Relationship:**
- NB = simplest Bayesian Network for classification
- Can extend to more complex structures (TAN, BAN, full BN)
- Trade-off: complexity vs. interpretability vs. data requirements

---

## Question 86
- [ ] Done

**How do you implement Naive Bayes for time-series classification?**

**Answer:**

For time-series, NB doesn't directly model temporal dependencies. Solutions: (1) Extract statistical features (mean, std, trend) and apply NB, (2) Use sliding windows as features, (3) Compute frequency domain features (FFT), (4) Discretize time-series into symbols (SAX) for Multinomial NB.

**Approach 1: Statistical Feature Extraction**
```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
from scipy.stats import skew, kurtosis

def extract_ts_features(time_series):
    """
    Extract statistical features from time series.
    """
    return np.array([
        np.mean(time_series),
        np.std(time_series),
        np.min(time_series),
        np.max(time_series),
        skew(time_series),
        kurtosis(time_series),
        np.percentile(time_series, 25),
        np.percentile(time_series, 75),
        np.mean(np.abs(np.diff(time_series))),  # Mean absolute change
        len(np.where(np.diff(np.sign(time_series)))[0])  # Zero crossings
    ])

# Extract features for all time series
X_features = np.array([extract_ts_features(ts) for ts in time_series_data])

# Train Gaussian NB
nb = GaussianNB()
nb.fit(X_features, labels)
```

**Approach 2: SAX (Symbolic Aggregate Approximation)**
```python
def sax_transform(time_series, n_segments=8, alphabet_size=4):
    """
    Convert time series to symbolic string for Multinomial NB.
    """
    # Normalize
    ts_norm = (time_series - np.mean(time_series)) / (np.std(time_series) + 1e-9)
    
    # PAA: Piecewise Aggregate Approximation
    segment_len = len(ts_norm) // n_segments
    paa = [np.mean(ts_norm[i*segment_len:(i+1)*segment_len]) for i in range(n_segments)]
    
    # Discretize to symbols using breakpoints
    breakpoints = [-0.67, 0, 0.67]  # For alphabet_size=4
    symbols = np.digitize(paa, breakpoints)
    
    return symbols

# Convert all time series to symbolic representation
X_sax = np.array([sax_transform(ts) for ts in time_series_data])

# Use Multinomial NB on symbol counts
from sklearn.preprocessing import OneHotEncoder
X_encoded = OneHotEncoder(sparse=False).fit_transform(X_sax)

nb = MultinomialNB()
nb.fit(X_encoded, labels)
```

**Use Cases:**
- Activity recognition from accelerometer
- ECG classification
- Stock pattern classification

---

## Question 87
- [ ] Done

**What are the considerations for Naive Bayes in reinforcement learning?**

**Answer:**

NB in RL is limited but has niche uses: (1) State classification for hierarchical RL, (2) Reward prediction from state features, (3) Model-based RL - predict next state/reward distributions, (4) Exploration via uncertainty estimation. NB's fast inference suits real-time RL, but independence assumption limits complex state modeling.

**Applications:**

**1. State Classification in Hierarchical RL**
```python
from sklearn.naive_bayes import GaussianNB

class HierarchicalRLAgent:
    """
    Use NB to classify state into high-level categories,
    then select appropriate sub-policy.
    """
    def __init__(self, sub_policies):
        self.state_classifier = GaussianNB()
        self.sub_policies = sub_policies  # Dict: category -> policy
    
    def train_classifier(self, states, categories):
        """Learn to classify states into categories."""
        self.state_classifier.fit(states, categories)
    
    def select_action(self, state):
        # Classify state
        category = self.state_classifier.predict([state])[0]
        # Use appropriate sub-policy
        return self.sub_policies[category].act(state)
```

**2. Model-Based RL: Predict Reward**
```python
class NBRewardPredictor:
    """
    Predict reward distribution given state-action.
    """
    def __init__(self):
        self.model = GaussianNB()
    
    def fit(self, states, actions, rewards):
        # Discretize rewards into bins
        reward_bins = np.digitize(rewards, bins=[-1, 0, 1])
        features = np.hstack([states, actions.reshape(-1, 1)])
        self.model.fit(features, reward_bins)
    
    def predict_reward_dist(self, state, action):
        features = np.hstack([state, [action]])
        return self.model.predict_proba([features])[0]
```

**3. Exploration via Uncertainty**
```python
def nb_exploration_bonus(nb, state, actions):
    """
    Use NB uncertainty for exploration.
    High entropy = uncertain = explore.
    """
    bonuses = []
    for action in actions:
        probs = nb.predict_proba(np.hstack([state, [action]]).reshape(1, -1))[0]
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        bonuses.append(entropy)
    return np.array(bonuses)
```

---

## Question 88
- [ ] Done

**How do you handle sequential data with Naive Bayes classifiers?**

**Answer:**

NB doesn't model sequences natively. Solutions: (1) N-gram features capture local order (bigrams, trigrams), (2) Position-weighted features, (3) Sequence kernels + NB, (4) Hidden Markov Models (HMM) which extend NB to sequences, (5) Bag-of-subsequences approach.

**Approach 1: N-gram Features**
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Character-level n-grams for sequences
vectorizer = CountVectorizer(
    analyzer='char',
    ngram_range=(2, 4)  # Bi-grams to 4-grams
)

sequences = ['ACGTACGT', 'TGCATGCA', 'AAACCCGGG']  # DNA sequences
X = vectorizer.fit_transform(sequences)

nb = MultinomialNB()
nb.fit(X, labels)
```

**Approach 2: Position-Weighted Features**
```python
def position_weighted_features(sequence, vocab, decay=0.9):
    """
    Weight features by position (recent = higher weight).
    """
    features = np.zeros(len(vocab))
    for i, item in enumerate(sequence):
        if item in vocab:
            weight = decay ** (len(sequence) - i - 1)  # Recent = higher
            features[vocab[item]] += weight
    return features

# Example
vocab = {'A': 0, 'B': 1, 'C': 2}
seq = ['A', 'B', 'A', 'C']
features = position_weighted_features(seq, vocab)
```

**Approach 3: Bag of Subsequences**
```python
def extract_subsequences(sequence, min_len=2, max_len=5):
    """
    Extract all subsequences as features.
    """
    subsequences = []
    for length in range(min_len, max_len + 1):
        for i in range(len(sequence) - length + 1):
            subseq = tuple(sequence[i:i+length])
            subsequences.append(subseq)
    return subsequences

# Convert to counts for NB
from collections import Counter

def sequence_to_features(sequences):
    all_subseqs = set()
    for seq in sequences:
        all_subseqs.update(extract_subsequences(seq))
    
    vocab = {s: i for i, s in enumerate(all_subseqs)}
    
    X = np.zeros((len(sequences), len(vocab)))
    for i, seq in enumerate(sequences):
        counts = Counter(extract_subsequences(seq))
        for subseq, count in counts.items():
            if subseq in vocab:
                X[i, vocab[subseq]] = count
    return X
```

---

## Question 89
- [ ] Done

**What is the role of Naive Bayes in causal inference?**

**Answer:**

NB is primarily correlational, not causal. However, connections exist: (1) Bayesian networks (NB's parent) can encode causal relationships, (2) NB can be used in propensity score estimation, (3) Causal discovery can determine if NB structure matches true causal model. Key: correlation ≠ causation, but NB structure can sometimes align with causal structure.

**NB and Causal Graphs:**
```
Naive Bayes assumes: Y causes all Xi

        Y (Class/Treatment)
       /|\\
      v v v v
     X1 X2 X3 X4

Causal interpretation: Disease (Y) causes symptoms (Xi)
This IS a valid causal model for medical diagnosis!
```

**When NB Aligns with Causation:**
```python
# Medical diagnosis: Disease causes symptoms
# NB structure matches causal structure

# P(Disease | Symptoms) using Bayes
# = P(Symptoms | Disease) * P(Disease) / P(Symptoms)

# If symptoms are conditionally independent given disease
# (each symptom independently caused by disease),
# then NB is causally valid!
```

**Propensity Score with NB:**
```python
def nb_propensity_score(X, treatment):
    """
    Use NB to estimate propensity score P(Treatment | X).
    Used in causal inference for matching/weighting.
    """
    from sklearn.naive_bayes import GaussianNB
    
    nb = GaussianNB()
    nb.fit(X, treatment)
    
    # Propensity score = P(Treatment=1 | X)
    propensity_scores = nb.predict_proba(X)[:, 1]
    return propensity_scores

# Use for inverse propensity weighting
def ipw_estimate(Y, treatment, propensity):
    """Inverse propensity weighted estimator."""
    treated = Y[treatment == 1] / propensity[treatment == 1]
    control = Y[treatment == 0] / (1 - propensity[treatment == 0])
    return treated.mean() - control.mean()
```

**Limitations:**
- NB learns correlations, not causation
- Independence assumption may violate causal structure
- Cannot identify confounders or interventions

---

## Question 90
- [ ] Done

**How do you implement Naive Bayes for medical diagnosis applications?**

**Answer:**

NB is well-suited for medical diagnosis because disease → symptoms matches NB's generative model. Use Gaussian NB for lab values, Bernoulli NB for binary symptoms. Key considerations: handle missing data, calibrate probabilities, provide interpretable explanations, account for rare diseases (low priors).

**Medical Diagnosis System:**
```python
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.calibration import CalibratedClassifierCV

class MedicalDiagnosisNB:
    """
    Naive Bayes for disease diagnosis from symptoms and lab values.
    """
    def __init__(self):
        self.symptom_nb = BernoulliNB()  # Binary symptoms
        self.lab_nb = GaussianNB()        # Continuous lab values
        self.diseases = None
    
    def fit(self, symptoms, lab_values, diagnoses):
        """
        Train on patient data.
        symptoms: binary matrix (patient x symptom)
        lab_values: continuous matrix (patient x lab_test)
        diagnoses: disease labels
        """
        self.diseases = np.unique(diagnoses)
        self.symptom_nb.fit(symptoms, diagnoses)
        self.lab_nb.fit(lab_values, diagnoses)
    
    def diagnose(self, symptoms, lab_values):
        """
        Combine evidence from symptoms and lab values.
        """
        # Get log probabilities from each model
        symptom_log_probs = self.symptom_nb.predict_log_proba(symptoms.reshape(1, -1))
        lab_log_probs = self.lab_nb.predict_log_proba(lab_values.reshape(1, -1))
        
        # Combine (assuming independence between symptom and lab evidence)
        combined_log_probs = symptom_log_probs + lab_log_probs
        
        # Normalize to probabilities
        probs = np.exp(combined_log_probs - combined_log_probs.max())
        probs = probs / probs.sum()
        
        return {
            'diagnosis': self.diseases[probs.argmax()],
            'confidence': float(probs.max()),
            'differential': dict(zip(self.diseases, probs[0]))
        }
    
    def explain_diagnosis(self, symptoms, symptom_names, diagnosis):
        """
        Explain which symptoms support the diagnosis.
        """
        diag_idx = np.where(self.diseases == diagnosis)[0][0]
        log_probs = self.symptom_nb.feature_log_prob_[diag_idx]
        
        evidence = []
        for i, (symptom, present) in enumerate(zip(symptom_names, symptoms)):
            if present:
                contribution = log_probs[i]
                evidence.append((symptom, contribution))
        
        return sorted(evidence, key=lambda x: -x[1])
```

**Key Considerations:**
- Calibrate probabilities for clinical use
- Handle missing lab values (imputation or ignore)
- Explain predictions to clinicians
- Account for disease prevalence (priors)

---

## Question 91
- [ ] Done

**What are the regulatory compliance considerations for Naive Bayes in healthcare?**

**Answer:**

Healthcare ML faces regulations: HIPAA (data privacy), FDA (medical device software), GDPR (EU data rights). NB advantages: interpretable (supports explainability requirements), auditable, simple validation. Requirements: document training data, validate performance on diverse populations, enable patient data access/deletion, maintain audit trails.

**Key Regulations:**

| Regulation | Requirement | NB Compliance |
|------------|-------------|---------------|
| HIPAA | Data privacy | Train on de-identified data, secure storage |
| FDA (SaMD) | Safety & efficacy | Document validation, version control |
| GDPR | Right to explanation | NB naturally explainable |
| GDPR | Data deletion | Retrain without deleted patient data |

**Compliance Implementation:**
```python
import hashlib
import json
from datetime import datetime

class CompliantMedicalNB:
    """
    HIPAA/FDA compliant NB classifier.
    """
    def __init__(self, model_version):
        self.model = MultinomialNB()
        self.version = model_version
        self.training_log = []
        self.prediction_log = []
    
    def train(self, X, y, data_source, validation_results):
        """Train with documentation for FDA compliance."""
        self.model.fit(X, y)
        
        # Document training for regulatory audit
        self.training_log.append({
            'timestamp': str(datetime.now()),
            'version': self.version,
            'data_source': data_source,
            'n_samples': len(y),
            'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
            'validation_metrics': validation_results,
            'data_hash': hashlib.sha256(str(X.tolist()).encode()).hexdigest()[:16]
        })
    
    def predict_with_audit(self, X, patient_id_hash):
        """Predict with audit trail (no PHI stored)."""
        prediction = self.model.predict(X)[0]
        confidence = self.model.predict_proba(X).max()
        
        # Log prediction (no PHI - only hashed ID)
        self.prediction_log.append({
            'timestamp': str(datetime.now()),
            'patient_hash': patient_id_hash,
            'model_version': self.version,
            'prediction': str(prediction),
            'confidence': float(confidence)
        })
        
        return prediction, confidence
    
    def export_documentation(self):
        """Export for FDA 510(k) submission."""
        return {
            'model_type': 'Naive Bayes Classifier',
            'version': self.version,
            'training_documentation': self.training_log,
            'algorithm_description': 'Probabilistic classifier using Bayes theorem...',
            'intended_use': '...',
            'limitations': '...'
        }
```

---

## Question 92
- [ ] Done

**How do you implement Naive Bayes for financial fraud detection?**

**Answer:**

NB for fraud detection predicts P(fraud|transaction_features). Challenges: extreme class imbalance (0.1% fraud), evolving fraud patterns (concept drift), real-time requirements. Solutions: adjust priors, use online learning for adaptation, optimize for precision/recall trade-off, combine with rule-based systems.

**Fraud Detection System:**
```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

class FraudDetectionNB:
    """
    Naive Bayes fraud detector with imbalance handling.
    """
    def __init__(self, fraud_cost_ratio=100):
        self.model = GaussianNB()
        self.scaler = StandardScaler()
        self.fraud_cost_ratio = fraud_cost_ratio  # Cost of missing fraud vs false alarm
        self.threshold = 0.5
    
    def fit(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Adjust priors for imbalance (or use fit_prior=False)
        fraud_rate = y.mean()
        self.model.class_prior_ = [0.5, 0.5]  # Balanced priors
        self.model.fit(X_scaled, y)
        
        # Optimize threshold for cost-sensitive classification
        self._optimize_threshold(X_scaled, y)
    
    def _optimize_threshold(self, X, y):
        """Find threshold that minimizes expected cost."""
        probs = self.model.predict_proba(X)[:, 1]
        
        best_cost = float('inf')
        best_threshold = 0.5
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            preds = (probs > threshold).astype(int)
            # Cost = false_negatives * fraud_cost + false_positives * 1
            fn = ((y == 1) & (preds == 0)).sum()
            fp = ((y == 0) & (preds == 1)).sum()
            cost = fn * self.fraud_cost_ratio + fp
            
            if cost < best_cost:
                best_cost = cost
                best_threshold = threshold
        
        self.threshold = best_threshold
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        return (probs > self.threshold).astype(int)
    
    def predict_with_risk_score(self, X):
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        return {
            'is_fraud': probs > self.threshold,
            'risk_score': probs,
            'risk_level': np.where(probs > 0.7, 'HIGH', 
                          np.where(probs > 0.3, 'MEDIUM', 'LOW'))
        }

# Example features for fraud detection
# transaction_amount, time_since_last, distance_from_home, etc.
```

**Key Techniques:**
- Cost-sensitive threshold optimization
- Real-time scoring (NB is fast)
- Online learning for evolving patterns
- Combine with rules for known fraud patterns

---

## Question 93
- [ ] Done

**What is the role of Naive Bayes in customer segmentation and marketing?**

**Answer:**

NB in marketing: (1) Predict customer response to campaigns, (2) Classify customers into segments, (3) Predict churn probability, (4) Score leads for sales prioritization. Advantages: interpretable (marketing teams understand feature importance), handles mixed feature types, works with limited conversion data.

**Marketing Applications:**

**1. Campaign Response Prediction**
```python
from sklearn.naive_bayes import GaussianNB

class CampaignResponsePredictor:
    """
    Predict customer response to marketing campaign.
    """
    def __init__(self):
        self.model = GaussianNB()
    
    def fit(self, customer_features, responded):
        """
        Features: age, income, past_purchases, email_opens, etc.
        responded: 1 if customer responded to campaign
        """
        self.model.fit(customer_features, responded)
    
    def score_customers(self, customer_features):
        """Return response probability for targeting."""
        probs = self.model.predict_proba(customer_features)[:, 1]
        return probs
    
    def get_top_targets(self, customer_features, customer_ids, top_n=1000):
        """Get top N customers most likely to respond."""
        probs = self.score_customers(customer_features)
        top_indices = np.argsort(probs)[-top_n:][::-1]
        return customer_ids[top_indices], probs[top_indices]
```

**2. Customer Churn Prediction**
```python
class ChurnPredictor:
    def __init__(self):
        self.model = GaussianNB()
    
    def fit(self, features, churned):
        """
        Features: days_since_purchase, support_tickets, 
                  usage_decline, contract_length, etc.
        """
        self.model.fit(features, churned)
    
    def identify_at_risk(self, features, customer_ids, threshold=0.7):
        """Identify customers at risk of churning."""
        probs = self.model.predict_proba(features)[:, 1]
        at_risk = probs > threshold
        return customer_ids[at_risk], probs[at_risk]
    
    def explain_risk(self, features, feature_names):
        """Explain why customer is at risk."""
        # Get feature contributions
        means_churn = self.model.theta_[1]  # Mean for churn class
        means_stay = self.model.theta_[0]   # Mean for stay class
        
        risk_factors = []
        for i, (name, val) in enumerate(zip(feature_names, features)):
            # If customer value closer to churn mean, it's a risk factor
            if abs(val - means_churn[i]) < abs(val - means_stay[i]):
                risk_factors.append(name)
        return risk_factors
```

**3. Lead Scoring**
```python
# Score leads by conversion probability
lead_probs = nb.predict_proba(lead_features)[:, 1]
lead_scores = (lead_probs * 100).astype(int)  # 0-100 score
```

---

## Question 94
- [ ] Done

**How do you handle multi-language text classification with Naive Bayes?**

**Answer:**

For multi-language classification: (1) Train separate NB per language, (2) Use language-agnostic features (character n-grams), (3) Use multilingual word embeddings, (4) First detect language then route to language-specific model, (5) Cross-lingual transfer using translation or shared vocabulary.

**Approach 1: Language Detection + Language-Specific Models**
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

class MultilingualClassifier:
    """
    Classify text in multiple languages.
    """
    def __init__(self, languages=['en', 'es', 'fr']):
        self.languages = languages
        self.lang_detector = self._build_lang_detector()
        self.classifiers = {lang: None for lang in languages}
        self.vectorizers = {lang: TfidfVectorizer() for lang in languages}
    
    def _build_lang_detector(self):
        """Simple character n-gram language detector."""
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
        detector = MultinomialNB()
        return {'vectorizer': vectorizer, 'model': detector}
    
    def fit_language_detector(self, texts, languages):
        X = self.lang_detector['vectorizer'].fit_transform(texts)
        self.lang_detector['model'].fit(X, languages)
    
    def fit(self, texts, labels, languages):
        """Train classifier for each language."""
        for lang in self.languages:
            mask = np.array(languages) == lang
            if mask.sum() > 0:
                texts_lang = [t for t, m in zip(texts, mask) if m]
                labels_lang = [l for l, m in zip(labels, mask) if m]
                
                X = self.vectorizers[lang].fit_transform(texts_lang)
                self.classifiers[lang] = MultinomialNB()
                self.classifiers[lang].fit(X, labels_lang)
    
    def predict(self, text):
        # Detect language
        lang_features = self.lang_detector['vectorizer'].transform([text])
        lang = self.lang_detector['model'].predict(lang_features)[0]
        
        # Classify with appropriate model
        if self.classifiers[lang] is not None:
            X = self.vectorizers[lang].transform([text])
            return self.classifiers[lang].predict(X)[0], lang
        else:
            return 'unknown', lang
```

**Approach 2: Character N-grams (Language-Agnostic)**
```python
# Character n-grams work across languages
vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 5),  # Character n-grams
    max_features=10000
)

# Train single model on all languages
X = vectorizer.fit_transform(all_texts)  # Mixed languages
nb = MultinomialNB()
nb.fit(X, labels)
```

**Approach 3: Cross-Lingual Transfer**
```python
# Use shared vocabulary (English + translations)
def create_bilingual_features(text, lang, translation_dict):
    """Map non-English words to English equivalents."""
    if lang != 'en':
        tokens = text.split()
        tokens = [translation_dict.get(t, t) for t in tokens]
        text = ' '.join(tokens)
    return text
```

---

## Question 95
- [ ] Done

**What are the considerations for Naive Bayes in social media analysis?**

**Answer:**

Social media challenges: informal language, slang, emojis, hashtags, short texts, evolving vocabulary. NB considerations: (1) Preprocessing for social text (handle @mentions, URLs, emojis), (2) Character n-grams for misspellings, (3) Include sentiment lexicons, (4) Handle class imbalance (most posts neutral), (5) Online learning for trending topics.

**Social Media Text Preprocessing:**
```python
import re

def preprocess_social_text(text):
    """
    Clean social media text for NB classification.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs with token
    text = re.sub(r'http\S+|www\S+', ' URL ', text)
    
    # Replace @mentions with token
    text = re.sub(r'@\w+', ' MENTION ', text)
    
    # Replace hashtags (keep the text)
    text = re.sub(r'#(\w+)', r' HASHTAG_\1 ', text)
    
    # Handle repeated characters (loooove -> love)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Handle emojis (convert to text or token)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols
        u"\U0001F680-\U0001F6FF"  # transport
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(' EMOJI ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text
```

**Social Media Sentiment Classifier:**
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class SocialMediaSentimentNB:
    def __init__(self):
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                preprocessor=preprocess_social_text,
                ngram_range=(1, 2),
                max_features=10000,
                min_df=2
            )),
            ('classifier', MultinomialNB(alpha=0.5))
        ])
    
    def fit(self, tweets, sentiments):
        self.pipeline.fit(tweets, sentiments)
    
    def predict(self, tweet):
        return self.pipeline.predict([tweet])[0]
    
    def analyze_trends(self, tweets, timestamps):
        """Analyze sentiment over time."""
        predictions = self.pipeline.predict(tweets)
        # Group by time period and compute sentiment distribution
        # ...
        return predictions
```

**Key Features for Social Media:**
- Character n-grams (handles misspellings)
- Emoji indicators
- Hashtag content
- Exclamation/question mark counts
- ALL CAPS detection

---

## Question 96
- [ ] Done

**How do you implement Naive Bayes for content recommendation systems?**

**Answer:**

NB recommends content by predicting P(user_likes|content_features). Train on user's interaction history (likes, clicks, reads). Features: content attributes (genre, author, keywords). Advantages: fast real-time recommendations, works with limited user history, provides probability scores for ranking.

**Content-Based Recommendation:**
```python
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

class ContentRecommenderNB:
    """
    Recommend content (articles, videos) based on user preferences.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.user_models = {}  # user_id -> trained NB
    
    def train_user_model(self, user_id, content_texts, user_ratings):
        """
        Train personalized model for each user.
        content_texts: list of content descriptions
        user_ratings: 1 if liked, 0 if not liked
        """
        X = self.vectorizer.fit_transform(content_texts)
        
        nb = MultinomialNB(alpha=1.0)
        nb.fit(X, user_ratings)
        
        self.user_models[user_id] = nb
    
    def recommend(self, user_id, candidate_content, top_k=10):
        """
        Recommend top-k content for user.
        """
        if user_id not in self.user_models:
            return None  # Cold start - no model for user
        
        model = self.user_models[user_id]
        X = self.vectorizer.transform(candidate_content)
        
        # Get probability of liking each content
        like_probs = model.predict_proba(X)[:, 1]
        
        # Return top-k
        top_indices = np.argsort(like_probs)[-top_k:][::-1]
        return [
            {'index': i, 'score': like_probs[i]} 
            for i in top_indices
        ]
    
    def explain_recommendation(self, user_id, content_text):
        """
        Explain why content was recommended.
        """
        model = self.user_models[user_id]
        X = self.vectorizer.transform([content_text])
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get features that contribute to "like" prediction
        log_prob_like = model.feature_log_prob_[1]
        log_prob_not = model.feature_log_prob_[0]
        importance = log_prob_like - log_prob_not
        
        # Get words in this content
        word_indices = X.nonzero()[1]
        
        reasons = []
        for idx in word_indices:
            if importance[idx] > 0:
                reasons.append((feature_names[idx], importance[idx]))
        
        return sorted(reasons, key=lambda x: -x[1])[:5]

# Example
recommender = ContentRecommenderNB()

# User's reading history
user_articles = ['tech startup raises funding', 'AI beats humans at chess', 'stock market trends']
user_liked = [1, 1, 0]  # Liked tech/AI, not finance

recommender.train_user_model('user123', user_articles, user_liked)

# Get recommendations
candidates = ['new AI model announced', 'market crash feared']
recs = recommender.recommend('user123', candidates)
```

---

## Question 97
- [ ] Done

**What is the future of Naive Bayes in the era of transformer models?**

**Answer:**

Despite transformers (BERT, GPT) dominating NLP, NB remains relevant: (1) Baseline for comparison, (2) Resource-constrained environments (edge, mobile), (3) Interpretability-critical applications, (4) Low-data scenarios, (5) Real-time high-throughput systems. NB-transformer hybrids show promise for combining interpretability with power.

**When to Choose NB vs Transformers:**

| Criterion | Choose NB | Choose Transformers |
|-----------|-----------|--------------------|
| Training data | Limited (<10K samples) | Large (>100K) |
| Compute budget | Low (CPU only) | High (GPU available) |
| Latency requirement | <1ms | >10ms acceptable |
| Interpretability | Required | Optional |
| Task complexity | Simple classification | Complex understanding |
| Deployment target | Edge/mobile | Cloud |

**NB's Enduring Value:**
```python
# 1. Baseline: Always compare transformers against NB
def evaluate_models(X_train, y_train, X_test, y_test):
    # NB baseline
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    nb_acc = nb.score(X_test, y_test)
    print(f"NB Baseline: {nb_acc:.4f}")
    
    # If transformer only marginally better, NB might be better choice
    # Consider: training cost, inference cost, interpretability

# 2. Hybrid: Use NB features in transformer
def nb_enhanced_transformer(texts, nb_model, transformer_model):
    """
    Combine NB predictions with transformer.
    """
    nb_probs = nb_model.predict_proba(texts)
    transformer_probs = transformer_model.predict_proba(texts)
    
    # Ensemble
    combined = 0.3 * nb_probs + 0.7 * transformer_probs
    return combined

# 3. Distillation: Train NB from transformer predictions
def distill_to_nb(transformer_model, unlabeled_data):
    """
    Create lightweight NB from transformer's knowledge.
    """
    # Get transformer predictions on large unlabeled corpus
    pseudo_labels = transformer_model.predict(unlabeled_data)
    
    # Train NB on pseudo-labels
    nb = MultinomialNB()
    nb.fit(unlabeled_data, pseudo_labels)
    return nb  # Deployable on edge
```

**Future Directions:**
- NB for pre-filtering before expensive transformer calls
- NB-regularized transformer training
- Transformer embeddings + NB classifier
- NB for explainability of transformer predictions

---

## Question 98
- [ ] Done

**How do you combine Naive Bayes with modern NLP techniques?**

**Answer:**

Combine NB with modern NLP: (1) Use word embeddings (Word2Vec, FastText) as features, (2) Apply NB on BERT/sentence embeddings, (3) Use NB for interpretable component in neural pipeline, (4) NB-weighted attention in transformers, (5) Ensemble NB with neural models.

**Approach 1: NB on Sentence Embeddings**
```python
from sentence_transformers import SentenceTransformer
from sklearn.naive_bayes import GaussianNB

class EmbeddingNB:
    """
    Naive Bayes on transformer embeddings.
    Combines transformer representations with NB simplicity.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.classifier = GaussianNB()
    
    def fit(self, texts, labels):
        embeddings = self.encoder.encode(texts)
        self.classifier.fit(embeddings, labels)
    
    def predict(self, texts):
        embeddings = self.encoder.encode(texts)
        return self.classifier.predict(embeddings)
    
    def predict_proba(self, texts):
        embeddings = self.encoder.encode(texts)
        return self.classifier.predict_proba(embeddings)
```

**Approach 2: Word Embedding Features**
```python
import numpy as np
from gensim.models import KeyedVectors

def text_to_embedding(text, word_vectors, method='mean'):
    """
    Convert text to embedding by averaging word vectors.
    """
    words = text.lower().split()
    vectors = []
    for word in words:
        if word in word_vectors:
            vectors.append(word_vectors[word])
    
    if not vectors:
        return np.zeros(word_vectors.vector_size)
    
    if method == 'mean':
        return np.mean(vectors, axis=0)
    elif method == 'max':
        return np.max(vectors, axis=0)

# Usage
word_vectors = KeyedVectors.load_word2vec_format('embeddings.bin', binary=True)
X_embeddings = np.array([text_to_embedding(t, word_vectors) for t in texts])

nb = GaussianNB()
nb.fit(X_embeddings, labels)
```

**Approach 3: NB-Weighted Neural Input**
```python
import torch
import torch.nn as nn

class NBWeightedBERT(nn.Module):
    """
    Weight BERT outputs by NB feature importance.
    """
    def __init__(self, bert_model, nb_weights):
        super().__init__()
        self.bert = bert_model
        self.nb_weights = nn.Parameter(torch.tensor(nb_weights), requires_grad=False)
        self.classifier = nn.Linear(768, 2)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        # Apply NB-inspired weighting to CLS token
        weighted = outputs.last_hidden_state[:, 0, :] * self.nb_weights
        return self.classifier(weighted)
```

---

## Question 99
- [ ] Done

**What are the best practices for Naive Bayes model lifecycle management?**

**Answer:**

Lifecycle management covers: (1) Development - version control experiments, (2) Training - document data and hyperparameters, (3) Validation - test on held-out data, (4) Deployment - package with preprocessing, (5) Monitoring - track drift and performance, (6) Retraining - scheduled or triggered updates, (7) Retirement - graceful deprecation.

**Complete Lifecycle Implementation:**
```python
import mlflow
import joblib
from datetime import datetime
import numpy as np

class NBModelLifecycle:
    """
    Manage complete lifecycle of NB model.
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.current_version = None
        self.model = None
        self.vectorizer = None
    
    # 1. DEVELOPMENT
    def experiment(self, X_train, y_train, X_val, y_val, alpha_values):
        """Try different hyperparameters."""
        results = []
        for alpha in alpha_values:
            nb = MultinomialNB(alpha=alpha)
            nb.fit(X_train, y_train)
            score = nb.score(X_val, y_val)
            results.append({'alpha': alpha, 'score': score})
            
            # Log to MLflow
            mlflow.log_param('alpha', alpha)
            mlflow.log_metric('val_accuracy', score)
        
        return results
    
    # 2. TRAINING
    def train(self, texts, labels, alpha=1.0):
        """Train production model with documentation."""
        self.vectorizer = TfidfVectorizer(max_features=10000)
        X = self.vectorizer.fit_transform(texts)
        
        self.model = MultinomialNB(alpha=alpha)
        self.model.fit(X, labels)
        
        self.current_version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Log training metadata
        self.metadata = {
            'version': self.current_version,
            'n_samples': len(labels),
            'n_features': X.shape[1],
            'alpha': alpha,
            'class_distribution': dict(zip(*np.unique(labels, return_counts=True))),
            'trained_at': str(datetime.now())
        }
    
    # 3. VALIDATION
    def validate(self, X_test, y_test):
        """Validate before deployment."""
        predictions = self.model.predict(X_test)
        metrics = {
            'accuracy': (predictions == y_test).mean(),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted')
        }
        self.metadata['validation_metrics'] = metrics
        return metrics
    
    # 4. DEPLOYMENT
    def deploy(self, path):
        """Save model for deployment."""
        artifact = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'metadata': self.metadata
        }
        joblib.dump(artifact, f'{path}/{self.model_name}_v{self.current_version}.joblib')
    
    # 5. MONITORING (see earlier questions)
    
    # 6. RETRAINING
    def should_retrain(self, recent_accuracy, threshold=0.05):
        """Check if model needs retraining."""
        baseline = self.metadata.get('validation_metrics', {}).get('accuracy', 1.0)
        return (baseline - recent_accuracy) > threshold
    
    # 7. RETIREMENT
    def retire(self, replacement_model):
        """Gracefully retire model."""
        # Archive current model
        # Switch traffic to replacement
        # Log retirement
        pass
```

---

## Question 100
- [ ] Done

**How do you implement end-to-end Naive Bayes classification pipelines?**

**Answer:**

End-to-end NB pipeline includes: (1) Data loading and validation, (2) Preprocessing (cleaning, normalization), (3) Feature extraction (TF-IDF, embeddings), (4) Model training with cross-validation, (5) Hyperparameter tuning, (6) Evaluation and reporting, (7) Serialization and deployment, (8) Inference API.

**Complete End-to-End Pipeline:**
```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class EndToEndNBPipeline:
    """
    Complete text classification pipeline.
    """
    def __init__(self):
        self.pipeline = None
        self.best_params = None
    
    # 1. DATA LOADING & VALIDATION
    def load_data(self, filepath):
        """Load and validate data."""
        import pandas as pd
        df = pd.read_csv(filepath)
        
        # Validation
        assert 'text' in df.columns, "Missing 'text' column"
        assert 'label' in df.columns, "Missing 'label' column"
        assert df['text'].notna().all(), "Found null texts"
        
        return df['text'].tolist(), df['label'].tolist()
    
    # 2-3. PREPROCESSING & FEATURE EXTRACTION
    def build_pipeline(self):
        """Create sklearn pipeline."""
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2),
                max_features=10000,
                min_df=2,
                max_df=0.95
            )),
            ('classifier', MultinomialNB())
        ])
        return self.pipeline
    
    # 4-5. TRAINING & HYPERPARAMETER TUNING
    def train(self, X_train, y_train, tune=True):
        """Train with optional hyperparameter tuning."""
        self.build_pipeline()
        
        if tune:
            param_grid = {
                'tfidf__max_features': [5000, 10000, 20000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'classifier__alpha': [0.01, 0.1, 1.0]
            }
            
            grid_search = GridSearchCV(
                self.pipeline, param_grid, 
                cv=5, scoring='f1_weighted', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            self.pipeline = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"Best params: {self.best_params}")
        else:
            self.pipeline.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5)
        print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # 6. EVALUATION
    def evaluate(self, X_test, y_test):
        """Comprehensive evaluation."""
        y_pred = self.pipeline.predict(X_test)
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return {
            'accuracy': (y_pred == y_test).mean(),
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    # 7. SERIALIZATION
    def save(self, filepath):
        """Save pipeline."""
        joblib.dump({
            'pipeline': self.pipeline,
            'best_params': self.best_params
        }, filepath)
    
    def load(self, filepath):
        """Load pipeline."""
        artifact = joblib.load(filepath)
        self.pipeline = artifact['pipeline']
        self.best_params = artifact['best_params']
    
    # 8. INFERENCE API
    def predict(self, text):
        """Single prediction."""
        return self.pipeline.predict([text])[0]
    
    def predict_proba(self, text):
        """Prediction with probabilities."""
        probs = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        return dict(zip(classes, probs))
    
    def predict_batch(self, texts):
        """Batch prediction."""
        return self.pipeline.predict(texts)


# Usage
if __name__ == "__main__":
    # Initialize
    pipe = EndToEndNBPipeline()
    
    # Load data
    texts, labels = pipe.load_data('data.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Train
    pipe.train(X_train, y_train, tune=True)
    
    # Evaluate
    results = pipe.evaluate(X_test, y_test)
    
    # Save
    pipe.save('model/nb_pipeline.joblib')
    
    # Inference
    prediction = pipe.predict("This is a test document")
    print(f"Prediction: {prediction}")
```

---

## Question 101
- [ ] Done

**How would you handle an imbalanced dataset when using a Naive Bayes classifier?**

**Answer:**

When dealing with imbalanced data (e.g., 95% normal, 5% fraud), standard NB will be biased toward the majority class due to prior probabilities. The solution involves multiple strategies: adjusting class priors, resampling the data, adjusting the decision threshold, or using appropriate evaluation metrics.

**Scenario Analysis:**

Suppose you have:
- 9,500 legitimate transactions (95%)
- 500 fraudulent transactions (5%)

**Problem:** NB learns P(fraud) = 0.05, biasing predictions toward legitimate.

**Solution Strategies:**

**Strategy 1: Adjust Class Priors**
```python
from sklearn.naive_bayes import MultinomialNB

# Option A: Use uniform priors
nb = MultinomialNB(fit_prior=False)  # Treats P(fraud) = P(legitimate) = 0.5

# Option B: Manually set priors
nb = MultinomialNB(class_prior=[0.5, 0.5])  # Balance the priors
```

**Strategy 2: Resample the Data**
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Oversample minority (fraud)
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Or undersample majority
rus = RandomUnderSampler(random_state=42)
X_balanced, y_balanced = rus.fit_resample(X, y)
```

**Strategy 3: Adjust Decision Threshold**
```python
# Get probabilities instead of hard predictions
probs = nb.predict_proba(X_test)[:, 1]  # P(fraud)

# Lower threshold to catch more fraud (increase recall)
threshold = 0.3  # Instead of default 0.5
predictions = (probs > threshold).astype(int)
```

**Strategy 4: Use Appropriate Metrics**
```python
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score

# Don't use accuracy! Use these instead:
print(f"F1 Score: {f1_score(y_test, y_pred)}")
print(f"ROC-AUC: {roc_auc_score(y_test, probs)}")

# Find optimal threshold using precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, probs)
```

**Recommended Approach for This Scenario:**
1. Start with balanced priors (`fit_prior=False`)
2. Use SMOTE if you have enough minority samples
3. Tune threshold based on business requirements (precision vs recall trade-off)
4. Evaluate with F1-score and ROC-AUC, not accuracy

---

## Question 102
- [ ] Done

**Discuss the impact of feature scaling on Naive Bayes classifiers.**

**Answer:**

Feature scaling (normalization, standardization) generally has **no impact on Naive Bayes** classification results, unlike many other algorithms. This is because NB computes probabilities based on feature distributions independently - scaling doesn't change the relative likelihood calculations.

**Scenario Analysis:**

Consider features with different scales:
- Feature 1: Age (20-80)
- Feature 2: Income (20,000-200,000)

**Why Scaling Doesn't Matter for NB:**

**For Gaussian NB:**
```
Original:     P(income=50000 | class) uses μ=60000, σ=20000
Standardized: P(z=-0.5 | class) uses μ=0, σ=1

Both give the same relative likelihood ranking because:
- The Gaussian PDF shape is preserved
- Only the parameters change, not the relative probabilities
```

**Mathematical Proof:**

For Gaussian NB, standardizing feature x to z = (x-μ)/σ:

Original: $P(x|c) = \frac{1}{\sigma_c\sqrt{2\pi}} e^{-\frac{(x-\mu_c)^2}{2\sigma_c^2}}$

Standardized: $P(z|c) = \frac{1}{\sqrt{2\pi}} e^{-\frac{(z-\mu_c')^2}{2}}$

The likelihood ratio between classes remains the same!

**When Scaling Might Matter:**

| Variant | Impact of Scaling |
|---------|------------------|
| Gaussian NB | None - ranking unchanged |
| Multinomial NB | N/A - expects counts, not scaled values |
| Bernoulli NB | N/A - expects binary values |

**Practical Verification:**
```python
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Without scaling
gnb = GaussianNB()
gnb.fit(X_train, y_train)
acc_original = accuracy_score(y_test, gnb.predict(X_test))

# With scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gnb_scaled = GaussianNB()
gnb_scaled.fit(X_train_scaled, y_train)
acc_scaled = accuracy_score(y_test, gnb_scaled.predict(X_test_scaled))

print(f"Original: {acc_original:.4f}")
print(f"Scaled: {acc_scaled:.4f}")
# Results will be identical!
```

**Key Takeaway:**
Unlike SVM, KNN, or Neural Networks, you don't need to scale features for Naive Bayes. Focus on other preprocessing (handling missing values, smoothing) instead.

---

## Question 103
- [ ] Done

**How can overfitting occur in Naive Bayes, and how would you prevent it?**

**Answer:**

While Naive Bayes is generally resistant to overfitting due to its strong simplicity bias, it can still overfit in specific scenarios: (1) when training data is very small, (2) when vocabulary is huge relative to data (text), (3) when smoothing is too low.

**Scenario Analysis:**

**When NB Overfits:**

1. **Rare Features Dominate:**
   - A word appears once in training, only in spam
   - Without smoothing: P(word|spam) = 1.0, P(word|ham) = 0.0
   - Model becomes overly confident on rare patterns

2. **Very Small Training Set:**
   - With only 100 samples and 10,000 features
   - Estimated probabilities are unreliable

3. **Insufficient Smoothing:**
   - Low alpha makes model memorize training patterns

**Prevention Strategies:**

**Strategy 1: Increase Smoothing (alpha)**
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

# Find optimal smoothing parameter
param_grid = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

print(f"Best alpha: {grid_search.best_params_['alpha']}")
```

**Strategy 2: Feature Selection (Reduce Vocabulary)**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

# Limit vocabulary size
vectorizer = TfidfVectorizer(
    max_features=5000,  # Cap vocabulary
    min_df=5,           # Ignore rare words
    max_df=0.95         # Ignore too common words
)

# Or use chi-square selection
selector = SelectKBest(chi2, k=1000)
X_selected = selector.fit_transform(X_tfidf, y)
```

**Strategy 3: Cross-Validation**
```python
from sklearn.model_selection import cross_val_score

# Use CV to detect overfitting
train_score = nb.score(X_train, y_train)
cv_scores = cross_val_score(nb, X_train, y_train, cv=5)

print(f"Train: {train_score:.4f}")
print(f"CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Large gap indicates overfitting
if train_score - cv_scores.mean() > 0.1:
    print("Warning: Potential overfitting detected")
```

**Strategy 4: Use Prior Regularization**
```python
# For imbalanced data, uniform priors prevent overfitting to majority
nb = MultinomialNB(fit_prior=False)
```

**Best Practices:**
- Start with alpha=1.0 (Laplace smoothing)
- Use min_df to remove rare features
- Validate with cross-validation
- Prefer simpler feature sets over large vocabularies

---

## Question 104
- [ ] Done

**Discuss improvements over standard Naive Bayes for dealing with highly correlated features.**

**Answer:**

Standard NB assumes feature independence, which fails with correlated features (e.g., "machine" and "learning" often appear together). This causes NB to double-count evidence. Solutions include: Tree-Augmented NB (TAN), feature clustering, PCA preprocessing, or Complement NB.

**Problem Demonstration:**

```
Features: "machine", "learning" (highly correlated in ML documents)

Standard NB treats them independently:
P(ML_topic | "machine learning") ∝ P("machine"|ML) × P("learning"|ML)

But "machine" and "learning" co-occur, so evidence is counted twice.
This leads to overconfident predictions.
```

**Solution 1: Tree-Augmented Naive Bayes (TAN)**

TAN allows one dependency per feature, forming a tree structure:

```
        Class
       /  |  \
      /   |   \
   F1 ←── F2 ──→ F3
   
F2 depends on both Class and F1 (one parent feature allowed)
```

```python
# Simplified TAN concept (not full implementation)
# Model: P(F1|C) × P(F2|C, F1) × P(F3|C, F2)

def tan_likelihood(features, c, dependencies):
    """
    Compute TAN likelihood with feature dependencies.
    """
    prob = 1.0
    for i, f in enumerate(features):
        if dependencies[i] is None:
            # No parent feature, standard NB
            prob *= P(f | c)
        else:
            # Has parent feature
            parent_idx = dependencies[i]
            prob *= P(f | c, features[parent_idx])
    return prob
```

**Solution 2: Feature Clustering**

Group correlated features, use one representative per group:

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import spearmanr

def cluster_correlated_features(X, threshold=0.7):
    """
    Cluster features based on correlation, keep one per cluster.
    """
    # Compute correlation matrix
    n_features = X.shape[1]
    corr_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(n_features):
            corr, _ = spearmanr(X[:, i], X[:, j])
            corr_matrix[i, j] = abs(corr)
    
    # Cluster features
    distance = 1 - corr_matrix
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1-threshold,
        affinity='precomputed',
        linkage='average'
    )
    clustering.fit(distance)
    
    # Select representative from each cluster (highest variance)
    selected_features = []
    for cluster_id in np.unique(clustering.labels_):
        cluster_features = np.where(clustering.labels_ == cluster_id)[0]
        variances = X[:, cluster_features].var(axis=0)
        best_feature = cluster_features[np.argmax(variances)]
        selected_features.append(best_feature)
    
    return selected_features
```

**Solution 3: PCA Preprocessing**

Transform features to uncorrelated principal components:

```python
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

# PCA creates uncorrelated features
pipeline = Pipeline([
    ('pca', PCA(n_components=0.95)),  # Keep 95% variance
    ('nb', GaussianNB())
])

# Now NB independence assumption holds better
pipeline.fit(X_train, y_train)
```

**Solution 4: Complement Naive Bayes**

Uses complement class to compute weights, reducing correlation impact:

```python
from sklearn.naive_bayes import ComplementNB

# ComplementNB is more robust to feature correlations
# Especially good for imbalanced text classification
cnb = ComplementNB(alpha=1.0, norm=True)
cnb.fit(X_train, y_train)
```

**Comparison:**

| Method | Pros | Cons |
|--------|------|------|
| Standard NB | Fast, simple | Overconfident with correlations |
| TAN | Models some dependencies | More complex, slower |
| Feature Clustering | Keeps NB simplicity | May lose information |
| PCA + NB | Guarantees independence | Loses interpretability |
| Complement NB | Robust, easy to use | Still assumes some independence |

**Recommendation:**
- Try Complement NB first (easy drop-in replacement)
- Use feature selection/clustering for highly redundant features
- Consider PCA if interpretability isn't crucial

---
