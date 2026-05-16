# Logistic Regression Interview Questions - General Questions

## Question 1
- [ ] Done

**How is logistic regression used for classification tasks?**

### Answer

**Definition:**
Logistic regression is a classification algorithm that models the probability of an instance belonging to a class using the sigmoid function. It outputs probabilities between 0 and 1, which are then thresholded to make binary predictions.

**How It Works:**
1. Compute linear combination: $z = \beta_0 + \beta_1 x_1 + ... + \beta_p x_p$
2. Apply sigmoid: $P(y=1|X) = \frac{1}{1 + e^{-z}}$
3. Threshold probability: predict class 1 if $P > 0.5$ (or custom threshold)

**Code Example:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict classes and probabilities
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
```

**Key Points:**
- Despite the name "regression," it's used for classification
- Outputs calibrated probabilities (well-calibrated by design)
- Decision boundary is linear in feature space
- Works for binary and multiclass (via One-vs-Rest or Multinomial)

**Interview Tip:**
Emphasize that logistic regression predicts probabilities, not just class labels. This makes it valuable when probability estimates matter (e.g., risk scoring).

---

## Question 2
- [ ] Done

**How do you interpret the coefficients of a logistic regression model?**

### Answer

**Definition:**
Coefficients in logistic regression represent the change in log-odds for a one-unit increase in the feature. Exponentiating coefficients gives odds ratios, which are more interpretable.

**Mathematical Interpretation:**

$$\log\left(\frac{P}{1-P}\right) = \beta_0 + \beta_1 x_1 + ... + \beta_p x_p$$

For coefficient $\beta_j$:
- **Log-odds change:** One unit increase in $x_j$ increases log-odds by $\beta_j$
- **Odds ratio:** $OR_j = e^{\beta_j}$

**Code Example:**
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Create interpretation table
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_[0],
    'Odds_Ratio': np.exp(model.coef_[0])
})

# Sort by absolute importance
coef_df['Abs_Coef'] = np.abs(coef_df['Coefficient'])
coef_df = coef_df.sort_values('Abs_Coef', ascending=False)
print(coef_df)
```

**Interpretation Guidelines:**

| Odds Ratio | Interpretation |
|------------|----------------|
| OR = 1 | No effect |
| OR > 1 | Increases odds of positive class |
| OR < 1 | Decreases odds of positive class |
| OR = 2 | Doubles the odds |
| OR = 0.5 | Halves the odds |

**Example:**
If `age` has coefficient 0.05, then $OR = e^{0.05} = 1.05$, meaning each year of age increases odds by 5%.

**Interview Tip:**
Always clarify whether interpreting raw coefficients (log-odds scale) or odds ratios. For non-technical stakeholders, odds ratios are more intuitive.

---

## Question 3
- [ ] Done

**How do you handle categorical variables in logistic regression?**

### Answer

**Definition:**
Categorical variables must be encoded numerically for logistic regression. The standard approach is one-hot encoding (dummy variables), creating k-1 binary features for k categories.

**Encoding Methods:**

**1. One-Hot Encoding (Most Common):**
```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

categorical_cols = ['gender', 'city', 'product_type']
numerical_cols = ['age', 'income']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
])

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('lr', LogisticRegression())
])
```

**2. Target Encoding (High Cardinality):**
```python
from sklearn.preprocessing import TargetEncoder

encoder = TargetEncoder()
X_encoded = encoder.fit_transform(X[['city']], y)
```

**Why drop='first'?**
- Avoids multicollinearity (dummy variable trap)
- With k categories, k-1 dummies fully represent the variable
- The dropped category becomes the reference level

**Interpretation:**
```python
# Each coefficient represents effect relative to reference category
# If 'gender_male' coef = 0.5, males have exp(0.5) = 1.65x higher odds than females (reference)
```

**Interview Tip:**
Always specify which category is the reference when interpreting coefficients. The choice of reference doesn't affect predictions, only interpretation.

---

## Question 4
- [ ] Done

**Can logistic regression be used for more than two classes? If so, how?**

### Answer

**Definition:**
Yes, logistic regression extends to multiclass classification via two approaches: One-vs-Rest (OvR) and Multinomial (Softmax) regression.

**1. One-vs-Rest (OvR):**
- Train K binary classifiers (one per class vs. all others)
- Predict class with highest probability
```python
from sklearn.linear_model import LogisticRegression

# OvR approach
model_ovr = LogisticRegression(multi_class='ovr')
model_ovr.fit(X_train, y_train)  # y has K classes
```

**2. Multinomial (Softmax):**
- Single model with K sets of coefficients
- Softmax function for probabilities:
$$P(y=k|X) = \frac{e^{X\beta_k}}{\sum_{j=1}^{K} e^{X\beta_j}}$$

```python
# Multinomial approach
model_multinomial = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model_multinomial.fit(X_train, y_train)
```

**Comparison:**

| Aspect | One-vs-Rest | Multinomial |
|--------|-------------|-------------|
| Models | K separate | 1 unified |
| Probabilities | May not sum to 1 | Sum to 1 |
| When better | Imbalanced classes | Balanced classes |
| Computation | Parallelizable | Single optimization |

**Code Example:**
```python
# Both approaches
model = LogisticRegression(multi_class='multinomial')
model.fit(X_train, y_train)

# Predict probabilities for all classes
proba = model.predict_proba(X_test)  # Shape: (n_samples, n_classes)
predictions = model.predict(X_test)  # Class with highest probability
```

**Interview Tip:**
Multinomial is preferred when classes are mutually exclusive and you need probabilities to sum to 1. OvR is simpler and works better with class imbalance.

---

## Question 5
- [ ] Done

**How do you evaluate a logistic regression model's performance?**

### Answer

**Definition:**
Evaluation uses classification metrics that assess prediction quality on held-out data. Key metrics include accuracy, precision, recall, F1-score, and AUC-ROC.

**Comprehensive Evaluation:**
```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
import matplotlib.pyplot as plt

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Classification report
print(classification_report(y_test, y_pred))

# Key metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")
```

**Metric Selection Guide:**

| Metric | When to Use |
|--------|-------------|
| Accuracy | Balanced classes |
| Precision | False positives costly (spam) |
| Recall | False negatives costly (disease) |
| F1-Score | Balance precision/recall |
| AUC-ROC | Overall ranking quality |
| AUC-PR | Imbalanced classes |

**ROC Curve Plot:**
```python
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

**Interview Tip:**
Always mention cross-validation for robust evaluation. Single train-test split can give misleading results due to variance.

---

## Question 6
- [ ] Done

**How do you deal with imbalanced classes in logistic regression?**

### Answer

**Definition:**
Class imbalance occurs when one class significantly outnumbers another. This can bias the model toward the majority class, reducing minority class recall.

**Solutions:**

**1. Class Weighting:**
```python
from sklearn.linear_model import LogisticRegression

# Automatic class weighting
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Manual weights
weights = {0: 1, 1: 10}  # Upweight minority class
model = LogisticRegression(class_weight=weights)
```

**2. Resampling:**
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# SMOTE oversampling
pipeline = ImbPipeline([
    ('smote', SMOTE(sampling_strategy=0.5)),
    ('lr', LogisticRegression())
])
```

**3. Threshold Adjustment:**
```python
# Lower threshold to catch more positives
y_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.3  # Default is 0.5
y_pred_adjusted = (y_proba >= threshold).astype(int)
```

**Strategy Comparison:**

| Method | Pros | Cons |
|--------|------|------|
| Class weights | Simple, no data change | May not be enough |
| SMOTE | Creates synthetic data | Can introduce noise |
| Undersampling | Reduces training time | Loses information |
| Threshold tuning | Flexible | Doesn't change model |

**Interview Tip:**
Start with `class_weight='balanced'` as it's the simplest approach. For severe imbalance (>1:100), combine with SMOTE or use anomaly detection instead.

---

## Question 7
- [ ] Done

**How can you extend logistic regression to handle ordinal outcomes?**

### Answer

**Definition:**
Ordinal regression (proportional odds model) handles ordered categories (e.g., low/medium/high) by estimating multiple thresholds while sharing coefficients across categories.

**Mathematical Formulation:**

For K ordered categories, estimate K-1 thresholds:
$$P(Y \leq k) = \frac{1}{1 + e^{-(\alpha_k - X\beta)}}$$

The proportional odds assumption: coefficients $\beta$ are the same across all threshold comparisons.

**Implementation:**
```python
# Using statsmodels
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Fit ordinal logistic regression
model = OrderedModel(y_train, X_train, distr='logit')
result = model.fit(method='bfgs')
print(result.summary())

# Predict probabilities for each category
proba = result.predict()
```

**Alternative with mord library:**
```python
from mord import LogisticAT  # Adjacent threshold model

model = LogisticAT()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**When to Use:**

| Outcome Type | Model |
|--------------|-------|
| Nominal (unordered) | Multinomial logistic |
| Ordinal (ordered) | Ordinal logistic |
| Binary | Standard logistic |

**Interview Tip:**
Ordinal regression preserves the ordering information that multinomial regression ignores. Use when categories have natural order (e.g., satisfaction ratings).

---

## Question 8
- [ ] Done

**What role do quasi-likelihood methods play in logistic regression?**

### Answer

**Definition:**
Quasi-likelihood methods estimate parameters by specifying only the mean-variance relationship rather than the full distribution. This provides robustness when the binomial variance assumption is violated (overdispersion).

**Overdispersion:**
When observed variance exceeds binomial variance:
$$\text{Var}(Y) > np(1-p)$$

**Quasi-Binomial Model:**
```python
import statsmodels.api as sm

# Quasi-binomial handles overdispersion
model = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial())
result = model.fit(scale='X2')  # Estimate scale parameter

# Check dispersion parameter
print(f"Dispersion: {result.scale}")  # >1 indicates overdispersion
```

**Adjusting Standard Errors:**
```python
# Robust (sandwich) standard errors
result_robust = model.fit(cov_type='HC1')

# Clustered standard errors
result_clustered = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
```

**When to Use:**

| Scenario | Approach |
|----------|----------|
| Standard binary data | Regular logistic |
| Overdispersion present | Quasi-binomial |
| Clustered data | GEE or mixed models |
| Unknown variance | Robust standard errors |

**Detecting Overdispersion:**
```python
# Pearson chi-square / df should be ~1
pearson_chi2 = result.pearson_chi2
df_resid = result.df_resid
dispersion = pearson_chi2 / df_resid
print(f"Dispersion parameter: {dispersion:.2f}")
```

**Interview Tip:**
Quasi-likelihood methods are useful when you're confident about the mean structure but uncertain about the variance structure. They provide valid inference even with misspecified variance.

---

## Question 9
- [ ] Done

**How can you use logistic regression for variable selection?**

### Answer

**Definition:**
Variable selection identifies the most important predictors. Logistic regression supports this through regularization (L1/Lasso), stepwise methods, and coefficient significance testing.

**1. L1 Regularization (Lasso):**
```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# L1 drives unimportant coefficients to exactly zero
model = LogisticRegression(penalty='l1', solver='saga', C=0.1)
model.fit(X_train, y_train)

# Selected features (non-zero coefficients)
selected = np.where(model.coef_[0] != 0)[0]
print(f"Selected features: {[feature_names[i] for i in selected]}")
```

**2. Recursive Feature Elimination:**
```python
from sklearn.feature_selection import RFE, RFECV

# RFE with cross-validation
selector = RFECV(
    LogisticRegression(),
    step=1,
    cv=5,
    scoring='roc_auc'
)
selector.fit(X, y)

print(f"Optimal features: {selector.n_features_}")
print(f"Selected: {[f for f, s in zip(feature_names, selector.support_) if s]}")
```

**3. Statistical Significance:**
```python
import statsmodels.api as sm

model = sm.Logit(y, sm.add_constant(X)).fit()

# Select features with p-value < 0.05
significant = model.pvalues[model.pvalues < 0.05].index.tolist()
print(f"Significant features: {significant}")
```

**Comparison:**

| Method | Pros | Cons |
|--------|------|------|
| L1 (Lasso) | Automatic, handles multicollinearity | Requires tuning C |
| RFE | Model-agnostic | Computationally expensive |
| p-values | Statistical interpretation | Multiple testing issues |

**Interview Tip:**
L1 regularization is preferred for automated variable selection. It naturally handles correlated features and doesn't require sequential hypothesis testing.

---

## Question 10
- [ ] Done

**Present an approach to predict the likelihood of a patient having a particular disease using logistic regression.**

### Answer

**Definition:**
Disease prediction using logistic regression involves building a risk model from patient features to estimate disease probability. Key considerations include clinical interpretability, calibration, and validation.

**Complete Pipeline:**
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss

# Load patient data
# Features: age, bmi, blood_pressure, cholesterol, family_history, etc.
# Target: disease (0/1)

# Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(penalty='l2', C=1.0)
model.fit(X_train_scaled, y_train)

# Evaluate
y_proba = model.predict_proba(X_test_scaled)[:, 1]
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
print(f"Brier Score: {brier_score_loss(y_test, y_proba):.3f}")
```

**Clinical Interpretability (Odds Ratios with CIs):**
```python
import statsmodels.api as sm

model_stats = sm.Logit(y_train, sm.add_constant(X_train_scaled)).fit()

# Present odds ratios
for name, coef, ci_low, ci_high in zip(
    feature_names,
    np.exp(model_stats.params[1:]),
    np.exp(model_stats.conf_int()[0][1:]),
    np.exp(model_stats.conf_int()[1][1:])
):
    print(f"{name}: OR = {coef:.2f} (95% CI: {ci_low:.2f}-{ci_high:.2f})")
```

**Calibration Check:**
```python
# Essential for clinical use - predicted 30% risk should mean 30% actually have disease
fraction_positive, mean_predicted = calibration_curve(y_test, y_proba, n_bins=10)

import matplotlib.pyplot as plt
plt.plot(mean_predicted, fraction_positive, 's-', label='Model')
plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Frequency')
plt.title('Calibration Curve')
plt.legend()
```

**Risk Score for Clinical Use:**
```python
def calculate_risk(patient_data):
    """
    Returns disease risk score for a patient
    """
    patient_scaled = scaler.transform([patient_data])
    risk_probability = model.predict_proba(patient_scaled)[0, 1]
    
    # Risk categories
    if risk_probability < 0.1:
        risk_level = "Low"
    elif risk_probability < 0.3:
        risk_level = "Moderate"
    else:
        risk_level = "High"
    
    return {
        'probability': risk_probability,
        'risk_level': risk_level,
        'recommendation': get_recommendation(risk_level)
    }
```

**Validation Requirements:**

| Requirement | Implementation |
|-------------|----------------|
| Internal validation | Cross-validation |
| External validation | Different population |
| Calibration | Calibration curve |
| Discrimination | AUC-ROC |
| Clinical utility | Decision curve analysis |

**Interview Tip:**
For medical applications, emphasize calibration (not just AUC) and external validation. Clinicians need well-calibrated probabilities to make informed decisions.

---
