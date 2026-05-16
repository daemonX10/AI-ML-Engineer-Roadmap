# Logistic Regression Interview Questions - Scenario-Based Questions

## Question 1
- [ ] Done

**How would you apply logistic regression to a marketing campaign to predict customer conversion?**

### Answer

**Definition:**
Customer conversion prediction models the probability that a prospect will take a desired action (purchase, sign-up, etc.) based on demographic, behavioral, and campaign-related features.

**End-to-End Implementation:**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Sample marketing data
data = pd.DataFrame({
    'age': np.random.randint(18, 65, 1000),
    'income': np.random.normal(50000, 20000, 1000),
    'website_visits': np.random.poisson(5, 1000),
    'email_opens': np.random.poisson(3, 1000),
    'time_on_site': np.random.exponential(10, 1000),
    'previous_purchases': np.random.poisson(1, 1000),
    'channel': np.random.choice(['Email', 'Social', 'Search', 'Direct'], 1000),
    'device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], 1000),
    'converted': np.random.binomial(1, 0.15, 1000)  # 15% conversion rate
})

# Feature engineering
data['engagement_score'] = data['website_visits'] + data['email_opens'] * 2
data['is_returning'] = (data['previous_purchases'] > 0).astype(int)

# Define features
numeric_features = ['age', 'income', 'website_visits', 'email_opens', 
                    'time_on_site', 'engagement_score']
categorical_features = ['channel', 'device', 'is_returning']

X = data[numeric_features + categorical_features]
y = data['converted']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
])

# Fit and evaluate
pipeline.fit(X_train, y_train)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
print(classification_report(y_test, pipeline.predict(X_test)))

# Feature importance (coefficients)
feature_names = numeric_features + list(
    pipeline.named_steps['prep']
    .named_transformers_['cat']
    .get_feature_names_out(categorical_features)
)
coefficients = pipeline.named_steps['clf'].coef_[0]

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Odds_Ratio': np.exp(coefficients)
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nTop Conversion Drivers:")
print(coef_df.head(10))
```

**Business Application:**

1. **Targeting:** Score all prospects, focus on high-probability
2. **Budget Allocation:** Invest more in channels with high conversion impact
3. **Personalization:** Tailor messaging based on key features
4. **Threshold Optimization:** Balance acquisition cost vs. volume

```python
# Optimal threshold for business constraint
# If cost per contact = $5, value per conversion = $100
cost_per_contact = 5
value_per_conversion = 100

# Find threshold that maximizes profit
thresholds = np.linspace(0.05, 0.5, 50)
profits = []

for thresh in thresholds:
    predicted_convert = y_proba >= thresh
    n_contacted = predicted_convert.sum()
    actual_conversions = (predicted_convert & y_test).sum()
    
    profit = actual_conversions * value_per_conversion - n_contacted * cost_per_contact
    profits.append(profit)

optimal_threshold = thresholds[np.argmax(profits)]
print(f"Optimal threshold for profit: {optimal_threshold:.3f}")
```

**Interview Tip:**
Marketing models need interpretability (odds ratios for stakeholders) and calibrated probabilities (for ROI calculations). Always present results in business terms: "Customers from social media are 2.3x more likely to convert."

---

## Question 2
- [ ] Done

**Discuss how logistic regression can be used for credit scoring in the financial industry.**

### Answer

**Definition:**
Credit scoring uses logistic regression to predict the probability of loan default, enabling lenders to make risk-based decisions on loan approval and pricing.

**Why Logistic Regression for Credit Scoring:**

| Advantage | Business Value |
|-----------|---------------|
| **Interpretability** | Regulatory requirement (explainable decisions) |
| **Calibrated Probabilities** | Direct PD (Probability of Default) estimates |
| **Stable** | Consistent predictions, low variance |
| **Coefficients as Scorecards** | Easy to deploy as point-based system |

**Implementation:**

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import roc_auc_score

# Typical credit features
credit_data = pd.DataFrame({
    'income': np.random.lognormal(10.5, 0.5, 1000),
    'debt_to_income': np.random.uniform(0.1, 0.6, 1000),
    'credit_utilization': np.random.uniform(0.1, 0.9, 1000),
    'num_credit_lines': np.random.poisson(5, 1000),
    'delinquencies_2yr': np.random.poisson(0.5, 1000),
    'credit_age_months': np.random.exponential(60, 1000),
    'inquiries_6mo': np.random.poisson(1, 1000),
    'default': np.random.binomial(1, 0.08, 1000)  # 8% default rate
})

# Feature engineering for credit
credit_data['log_income'] = np.log1p(credit_data['income'])
credit_data['debt_burden'] = credit_data['debt_to_income'] * credit_data['credit_utilization']

features = ['log_income', 'debt_to_income', 'credit_utilization', 
            'num_credit_lines', 'delinquencies_2yr', 'credit_age_months',
            'inquiries_6mo', 'debt_burden']

X = credit_data[features]
y = credit_data['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit model
model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000)
model.fit(X_train_scaled, y_train)

# Probability of Default (PD)
pd_scores = model.predict_proba(X_test_scaled)[:, 1]
print(f"AUC-ROC (Gini): {roc_auc_score(y_test, pd_scores):.3f}")
print(f"Gini Coefficient: {2 * roc_auc_score(y_test, pd_scores) - 1:.3f}")
```

**Converting to Scorecard:**
```python
def create_scorecard(model, scaler, feature_names, base_score=600, pdo=20, base_odds=50):
    """
    Convert logistic regression to points-based scorecard.
    PDO = Points to Double the Odds
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    means = scaler.mean_
    stds = scaler.scale_
    
    # Points for each feature (per unit change)
    points_per_unit = -factor * coefficients / stds
    
    scorecard = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Points_per_unit': points_per_unit
    })
    
    return scorecard, offset, factor

scorecard, offset, factor = create_scorecard(model, scaler, features)
print("\nScorecard Points:")
print(scorecard)
```

**Regulatory Considerations:**

1. **Fair Lending:** Cannot use protected attributes (race, gender)
2. **Adverse Action:** Must explain why applicant was denied
3. **Model Documentation:** Full model development documentation required
4. **Validation:** Independent validation before deployment
5. **Monitoring:** Ongoing performance monitoring required

**Interview Tip:**
Credit scoring requires Gini coefficient (2*AUC - 1) for performance measurement. Models must be explainable for regulatory compliance. Always use scaled coefficients for scorecard conversion.

---

## Question 3
- [ ] Done

**How would you use logistic regression to analyze the impact of various factors on employee attrition?**

### Answer

**Definition:**
Employee attrition modeling predicts the probability of an employee leaving the organization based on demographic, job-related, and engagement factors, enabling proactive retention strategies.

**Implementation:**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import statsmodels.api as sm

# Sample HR data
np.random.seed(42)
hr_data = pd.DataFrame({
    'age': np.random.randint(22, 60, 1000),
    'tenure_years': np.random.exponential(4, 1000),
    'salary': np.random.lognormal(10.8, 0.4, 1000),
    'satisfaction_score': np.random.uniform(1, 5, 1000),
    'performance_rating': np.random.choice([1, 2, 3, 4, 5], 1000, p=[0.05, 0.1, 0.5, 0.25, 0.1]),
    'overtime_hours': np.random.exponential(5, 1000),
    'promotions_last_5yr': np.random.poisson(0.5, 1000),
    'training_hours': np.random.exponential(20, 1000),
    'department': np.random.choice(['Sales', 'Engineering', 'HR', 'Marketing', 'Operations'], 1000),
    'manager_rating': np.random.uniform(1, 5, 1000),
    'work_from_home': np.random.choice([0, 1], 1000, p=[0.6, 0.4]),
    'left': np.random.binomial(1, 0.18, 1000)  # 18% attrition
})

# Feature engineering
hr_data['salary_ratio'] = hr_data['salary'] / hr_data.groupby('department')['salary'].transform('median')
hr_data['tenure_bucket'] = pd.cut(hr_data['tenure_years'], bins=[0, 1, 3, 5, 100], 
                                   labels=['0-1yr', '1-3yr', '3-5yr', '5+yr'])
hr_data['engagement_index'] = (hr_data['satisfaction_score'] + hr_data['manager_rating']) / 2

numeric_features = ['age', 'tenure_years', 'salary_ratio', 'satisfaction_score',
                    'overtime_hours', 'promotions_last_5yr', 'engagement_index']
categorical_features = ['department', 'tenure_bucket', 'work_from_home']

X = hr_data[numeric_features + categorical_features]
y = hr_data['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Build pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
])

# Fit and evaluate
pipeline.fit(X_train, y_train)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
print(classification_report(y_test, pipeline.predict(X_test)))

# Feature importance with confidence intervals (statsmodels)
X_train_processed = pipeline.named_steps['prep'].fit_transform(X_train)
feature_names = numeric_features + list(
    pipeline.named_steps['prep']
    .named_transformers_['cat']
    .get_feature_names_out(categorical_features)
)

X_sm = sm.add_constant(X_train_processed)
model_sm = sm.Logit(y_train, X_sm).fit(disp=0)

# Results with statistical significance
results = pd.DataFrame({
    'Feature': ['Intercept'] + feature_names,
    'Coefficient': model_sm.params,
    'Odds_Ratio': np.exp(model_sm.params),
    'p_value': model_sm.pvalues,
    'Significant': model_sm.pvalues < 0.05
})
results = results.sort_values('p_value')

print("\nSignificant Attrition Factors:")
print(results[results['Significant'] == True][['Feature', 'Odds_Ratio', 'p_value']])
```

**Key Insights Interpretation:**

```python
def interpret_for_hr(results):
    """Generate HR-friendly interpretations."""
    significant = results[results['Significant'] == True].copy()
    
    interpretations = []
    for _, row in significant.iterrows():
        feature = row['Feature']
        odds_ratio = row['Odds_Ratio']
        
        if odds_ratio > 1:
            change = f"{(odds_ratio - 1) * 100:.1f}% higher"
        else:
            change = f"{(1 - odds_ratio) * 100:.1f}% lower"
        
        interpretations.append(f"• {feature}: {change} odds of leaving")
    
    return interpretations

print("\nActionable Insights:")
for insight in interpret_for_hr(results):
    print(insight)
```

**Business Application:**

1. **Risk Scoring:** Flag high-risk employees for proactive retention
2. **Intervention Targeting:** Focus on modifiable factors (satisfaction, overtime)
3. **Budget Allocation:** Prioritize retention spend on high-value employees
4. **Policy Changes:** Address systemic issues (overtime, promotion rates)

**Interview Tip:**
For HR analytics, focus on actionable factors HR can influence (satisfaction, training, overtime) rather than fixed demographics. Present results as "X% increase in overtime is associated with Y% higher attrition odds" for stakeholder understanding.

---
