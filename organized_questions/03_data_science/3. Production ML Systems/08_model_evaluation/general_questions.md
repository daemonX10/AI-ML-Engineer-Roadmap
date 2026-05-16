# Model Evaluation Interview Questions - General Questions

## Question 1
- [ ] Done

**Define precision, recall, and F1-score.**

**Answer:**

### Definitions
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1-score** = 2 × (P × R) / (P + R)

### Interview Tip
Choose based on cost: FP cost → precision; FN cost → recall.

---

## Question 2
- [ ] Done

**What do you understand by the term “Confusion Matrix”?**

**Answer:**

### Structure
- True Positives (TP), False Positives (FP)
- False Negatives (FN), True Negatives (TN)

### Derived Metrics
Accuracy, Precision, Recall, Specificity

### Interview Tip
Can derive all classification metrics from confusion matrix.

---

## Question 3
- [ ] Done

**Why is accuracy not always the best metric for model evaluation?**

**Answer:**

### Problem with Accuracy
- Misleading for imbalanced datasets
- A model predicting all majority class gets high accuracy

### Better Alternatives
F1, AUC, precision-recall curves

### Interview Tip
Always check class distribution before using accuracy.

---

## Question 4
- [ ] Done

**How can learning curves help in model evaluation?**

**Answer:**

### What They Show
- Training score vs validation score over sample size
- Diagnose overfitting/underfitting

### Interpretation
- Gap between curves → overfitting
- Both curves low → underfitting

### Interview Tip
Use learning curves to decide if more data would help.

---

## Question 5
- [ ] Done

**How do you evaluate a regression model's performance?**

**Answer:**

### Common Metrics
- **MSE/RMSE**: Penalizes large errors
- **MAE**: Robust to outliers
- **R²**: Proportion of variance explained

### Interview Tip
RMSE for interpretable units, R² for explained variance.

---

## Question 6
- [ ] Done

**What metrics would you use to evaluate a classifier's performance?**

**Answer:**

### Metrics by Use Case
- Balanced: Accuracy, F1
- Imbalanced: Precision, Recall, F1, AUC
- Probability: Log-loss, Brier score

### Interview Tip
Match metric to business problem.

---

## Question 7
- [ ] Done

**How is the Area Under the Precision-Recall Curve (AUPRC) beneficial?**

**Answer:**

### Benefits
- Better than ROC for imbalanced datasets
- Focuses on positive class performance
- Ignores true negatives

### Interview Tip
Use PR-AUC when positives are rare (fraud, disease).

---

## Question 8
- [ ] Done

**How do you interpret a model's calibration curve?**

**Answer:**

### What It Shows
- Predicted probability vs actual frequency
- Diagonal = perfectly calibrated
- Above diagonal = overconfident

### Calibration Methods
Platt scaling, isotonic regression

### Interview Tip
Always calibrate before using probabilities for decisions.

---

## Question 9
- [ ] Done

**When is it appropriate to use the Matthews Correlation Coefficient (MCC)?**

**Answer:**

### When to Use
- Imbalanced datasets
- Binary classification
- Want single balanced metric

### Range
-1 (worst) to +1 (perfect), 0 = random

### Interview Tip
MCC is most balanced metric for binary classification.

---

## Question 10
- [ ] Done

**How do you assess statistical significance in differences of model performance?**

**Answer:**

### Methods
- Paired t-test on CV folds
- McNemar's test for classifiers
- Bootstrap confidence intervals

### Interview Tip
Statistical significance ≠ practical significance.

---

## Question 11
- [ ] Done

**What role do confidence intervals play in model evaluation?**

**Answer:**

### Purpose
- Quantify uncertainty in metrics
- Compare models' overlapping intervals
- Determine if differences are significant

### Interview Tip
Non-overlapping CIs suggest significant difference.

---

## Question 12
- [ ] Done

**How can Bayesian methods be used in model evaluation?**

**Answer:**

### Applications
- Credible intervals for metrics
- Bayesian comparison of classifiers
- Probabilistic model selection

### Advantages
- Natural uncertainty quantification
- Works with small samples

### Interview Tip
Bayesian methods give probability of one model being better.

---

## Question 13
- [ ] Done

**How do you compare multiple models with each other?**

**Answer:**

### Methods
- Cross-validation with same splits
- Statistical tests (Friedman, Nemenyi)
- Nested CV for hyperparameter tuning
- AIC/BIC for nested models

### Interview Tip
Use critical difference diagrams for multiple comparisons.

---

## Question 14
- [ ] Done

**When would you choose to use AIC over BIC?**

**Answer:**

### Use AIC When
- Focus on prediction
- Sample size is small
- Willing to accept more complex models

### Use BIC When
- Focus on finding true model
- Prefer simpler models

### Interview Tip
BIC has stronger penalty, more conservative.

---

## Question 15
- [ ] Done

**How is the Gini Coefficient used in evaluating classification models?**

**Answer:**

### Definition
Gini = 2 × AUC - 1

### Interpretation
- 0 = random
- 1 = perfect separation

### Use Case
Common in credit scoring, insurance.

### Interview Tip
Gini and AUC are directly related.

---

## Question 16
- [ ] Done

**What special considerations are there for evaluating models on imbalanced datasets?**

**Answer:**

### Considerations
- Don't use accuracy
- Use stratified sampling
- Focus on minority class metrics

### Recommended Metrics
- Precision, Recall, F1
- PR-AUC over ROC-AUC
- MCC

### Interview Tip
Always report confusion matrix for imbalanced problems.

---

