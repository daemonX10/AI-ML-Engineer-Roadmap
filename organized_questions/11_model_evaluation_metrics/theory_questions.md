# Accuracy/Precision/Recall/F1-Score - Theory Questions

## Question 1
- [ ] Done
**How do you choose the most appropriate primary metric between accuracy, precision, recall, and F1-score for different business contexts?**

**Answer:**

**Definition:**
Metric selection depends on the business cost of errors. Choose **Accuracy** for balanced classes with equal misclassification costs. Choose **Precision** when false positives are costly (spam filter). Choose **Recall** when false negatives are costly (disease detection). Choose **F1-score** when you need balance between precision and recall on imbalanced data.

**Core Concepts:**
- **Accuracy** = (TP + TN) / Total → Works only when classes are balanced
- **Precision** = TP / (TP + FP) → "Of predicted positives, how many are correct?"
- **Recall** = TP / (TP + FN) → "Of actual positives, how many did we catch?"
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall) → Harmonic mean

**Decision Framework:**

| Business Context | Primary Metric | Reason |
|-----------------|----------------|--------|
| Balanced classes, equal costs | Accuracy | Simple, interpretable |
| Imbalanced classes | F1-score or Recall | Accuracy misleading |
| High cost of false positives | Precision | Minimize wrong alerts |
| High cost of false negatives | Recall | Minimize missed cases |

**Practical Examples:**
- **Cancer detection** → Recall (missing cancer is deadly)
- **Email spam filter** → Precision (blocking good emails is bad UX)
- **Fraud detection** → Depends on investigation cost vs fraud loss
- **Recommendation system** → Precision (bad recommendations hurt trust)

**Interview Tip:**
Always ask: "What is more expensive - false positive or false negative?" The answer directly maps to precision vs recall priority.

---

## Question 2
- [ ] Done
**When would you prioritize precision over recall in a fraud detection system, and how do you quantify the business impact?**

**Answer:**

**Definition:**
Prioritize **Precision** in fraud detection when the cost of investigating false positives (wasted analyst time, customer friction) exceeds the cost of missing some fraud cases. This happens when fraud investigation resources are limited or when false alerts damage customer relationships.

**Scenario Analysis:**

| Scenario | Prioritize | Reason |
|----------|-----------|--------|
| Limited investigation team | Precision | Can't handle high false alerts |
| High customer friction from blocks | Precision | Bad UX hurts retention |
| High-value transactions | Recall | Missing fraud is very costly |
| Real-time blocking needed | Balance (F1) | Both errors are expensive |

**Business Impact Quantification:**

```
Cost of False Positive = (Investigation time × Analyst hourly rate) + Customer churn risk
Cost of False Negative = Average fraud amount × (1 - recovery rate)

Total Cost = (FP × Cost_FP) + (FN × Cost_FN)
```

**Example Calculation:**
- FP cost = $50 (analyst time) + $20 (customer friction) = $70
- FN cost = $500 average fraud loss
- If FP/FN ratio > 7:1, prioritize precision to reduce FP count

**Practical Approach:**
1. Calculate cost matrix from historical data
2. Set threshold to minimize total expected cost
3. Use precision when Cost_FP is significant relative to Cost_FN

**Interview Tip:**
Never give a generic answer. Always frame as: "It depends on the cost ratio. If investigation cost is X and fraud loss is Y, then..." — shows business thinking.

---

## Question 3
- [ ] Done
**How do you handle class imbalance when accuracy becomes misleading, and which alternative metrics provide better insights?**

**Answer:**

**Definition:**
When classes are imbalanced (e.g., 95% negative, 5% positive), a model predicting all negatives achieves 95% accuracy but is useless. Use metrics that focus on minority class performance: **Precision, Recall, F1-score, AUC-ROC, AUC-PR, or Balanced Accuracy**.

**Why Accuracy Fails:**
- Dataset: 1000 samples (950 negative, 50 positive)
- Model predicts all negative → Accuracy = 95%
- But Recall = 0% (misses all positives!)

**Alternative Metrics:**

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **F1-Score** | 2×(P×R)/(P+R) | Balance precision/recall |
| **Balanced Accuracy** | (TPR + TNR) / 2 | Equal weight to both classes |
| **AUC-ROC** | Area under ROC curve | Ranking ability across thresholds |
| **AUC-PR** | Area under Precision-Recall curve | Highly imbalanced data (better than ROC) |
| **MCC** | See Q10 | Balanced even with imbalance |

**Mathematical Formulation:**
$$\text{Balanced Accuracy} = \frac{1}{2}\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right)$$

**Python Code:**
```python
from sklearn.metrics import (f1_score, balanced_accuracy_score, 
                             roc_auc_score, average_precision_score)

y_true = [0]*950 + [1]*50  # Imbalanced
y_pred = [0]*980 + [1]*20  # Model predictions

# Misleading metric
accuracy = sum(t==p for t,p in zip(y_true, y_pred)) / len(y_true)

# Better metrics
f1 = f1_score(y_true, y_pred)
balanced_acc = balanced_accuracy_score(y_true, y_pred)
auc_pr = average_precision_score(y_true, y_pred)  # needs probabilities ideally

print(f"Accuracy: {accuracy:.2f}")  # High but misleading
print(f"F1-Score: {f1:.2f}")        # More honest
print(f"Balanced Accuracy: {balanced_acc:.2f}")
```

**Interview Tip:**
When interviewer mentions imbalanced data, immediately pivot away from accuracy. State: "Accuracy is misleading here. I would use F1-score or AUC-PR depending on whether I need a fixed threshold or ranking."

---

## Question 4
- [ ] Done
**What strategies help you optimize for F1-score when precision and recall have conflicting optimization directions?**

**Answer:**

**Definition:**
Precision and recall have an inherent trade-off: increasing one typically decreases the other. To optimize F1-score (their harmonic mean), use **threshold tuning, class weights, or direct F1 optimization** to find the sweet spot where their product is maximized.

**Why They Conflict:**
- Lower threshold → More positive predictions → Higher Recall, Lower Precision
- Higher threshold → Fewer positive predictions → Higher Precision, Lower Recall

**Strategies:**

**1. Threshold Optimization**
- Train model, get probability scores
- Sweep thresholds from 0 to 1
- Pick threshold that maximizes F1

**2. Class Weight Adjustment**
- Increase weight of minority class
- Forces model to pay more attention to positives
- Improves recall without destroying precision

**3. Resampling Techniques**
- SMOTE / Oversampling minority
- Undersampling majority
- Helps balance the optimization landscape

**4. Direct F1 Optimization**
- Use F1 as validation metric during hyperparameter tuning
- Some libraries support F1-based loss (approximately)

**Python Code:**
```python
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

# y_true: actual labels, y_prob: predicted probabilities
y_true = [0,0,0,1,1,1,0,1,0,1]
y_prob = [0.1,0.2,0.3,0.6,0.7,0.8,0.4,0.65,0.35,0.9]

# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

# Calculate F1 for each threshold
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[min(optimal_idx, len(thresholds)-1)]

print(f"Optimal Threshold: {optimal_threshold:.2f}")
print(f"Best F1: {f1_scores[optimal_idx]:.2f}")

# Apply optimal threshold
y_pred = [1 if p >= optimal_threshold else 0 for p in y_prob]
```

**Interview Tip:**
Mention that F1 uses harmonic mean (not arithmetic) because it penalizes extreme imbalance between precision and recall more heavily.

---

## Question 5
- [ ] Done
**How do you implement weighted F1-score calculations for multi-class problems with uneven class distributions?**

**Answer:**

**Definition:**
Weighted F1-score computes F1 for each class, then takes a weighted average where weights are proportional to class support (number of samples). This gives more importance to frequent classes and is suitable when class sizes reflect real-world importance.

**Types of Averaging:**

| Method | Formula | Use Case |
|--------|---------|----------|
| **Macro** | Simple mean of per-class F1 | Equal importance to all classes |
| **Micro** | Global TP, FP, FN → single F1 | Overall performance |
| **Weighted** | Support-weighted mean | Class frequency matters |

**Mathematical Formulation:**
$$\text{Weighted F1} = \sum_{c=1}^{C} \frac{n_c}{N} \times F1_c$$

Where:
- $n_c$ = number of samples in class $c$
- $N$ = total samples
- $F1_c$ = F1-score for class $c$

**Python Code:**
```python
from sklearn.metrics import f1_score, classification_report

# Multi-class predictions
y_true = [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2]  # Uneven: 4, 2, 6 samples
y_pred = [0, 0, 1, 0, 1, 0, 2, 2, 2, 1, 2, 2]

# Different averaging methods
macro_f1 = f1_score(y_true, y_pred, average='macro')      # Equal weight
weighted_f1 = f1_score(y_true, y_pred, average='weighted') # Support-weighted
micro_f1 = f1_score(y_true, y_pred, average='micro')      # Global

print(f"Macro F1: {macro_f1:.3f}")
print(f"Weighted F1: {weighted_f1:.3f}")
print(f"Micro F1: {micro_f1:.3f}")

# Detailed per-class breakdown
print(classification_report(y_true, y_pred))
```

**When to Use Weighted F1:**
- Class distribution in test set reflects real-world distribution
- Larger classes are genuinely more important
- You want overall system performance metric

**Interview Tip:**
If interviewer asks about multi-class metrics, clarify: "Are all classes equally important, or should I weight by frequency?" This shows you understand the nuance.

---

## Question 6
- [ ] Done
**When should you use macro-averaged versus micro-averaged F1-scores in multi-label classification scenarios?**

**Answer:**

**Definition:**
**Micro-averaging** aggregates all TP, FP, FN globally then computes F1 — favors frequent labels. **Macro-averaging** computes F1 per label then takes simple mean — treats all labels equally. Use micro when overall correctness matters; use macro when rare labels are equally important.

**Key Difference:**

| Aspect | Micro-F1 | Macro-F1 |
|--------|----------|----------|
| Calculation | Global TP/FP/FN → F1 | Per-label F1 → Average |
| Bias | Favors frequent labels | Equal weight to all labels |
| Use case | Overall system performance | Care about rare labels |

**Mathematical Formulation:**

$$\text{Micro-F1} = \frac{2 \times \sum TP}{\sum(2TP + FP + FN)}$$

$$\text{Macro-F1} = \frac{1}{L}\sum_{l=1}^{L} F1_l$$

**Example Scenario:**
- Multi-label: Article tagged with [Sports, Politics, Tech, ...]
- Sports appears 1000 times, Politics 50 times
- Micro-F1: Dominated by Sports performance
- Macro-F1: Politics counts equally

**When to Use:**

| Scenario | Choice | Reason |
|----------|--------|--------|
| Labels equally important | Macro | Rare labels matter |
| User-facing accuracy | Micro | Overall experience |
| Imbalanced but rare labels critical | Macro | Don't ignore minorities |
| Ranking/recommendation | Micro | Volume matters |

**Python Code:**
```python
from sklearn.metrics import f1_score

# Multi-label format: each sample has binary vector of labels
y_true = [[1,0,1], [0,1,0], [1,1,1], [0,0,1]]  # 3 labels
y_pred = [[1,0,0], [0,1,0], [1,0,1], [0,0,1]]

micro_f1 = f1_score(y_true, y_pred, average='micro')
macro_f1 = f1_score(y_true, y_pred, average='macro')

print(f"Micro-F1: {micro_f1:.3f}")  # Global performance
print(f"Macro-F1: {macro_f1:.3f}")  # Per-label average
```

**Interview Tip:**
For multi-label problems, always report both micro and macro. They tell different stories — micro is "how often are we right overall?" while macro is "how well do we do on each label?"

---

## Question 7
- [ ] Done
**How do you handle threshold selection to optimize different metrics in binary classification problems?**

**Answer:**

**Definition:**
Classification models output probabilities. The default threshold (0.5) is rarely optimal. Choose threshold by: (1) Maximizing target metric (F1, recall at precision constraint), (2) Using business cost matrix, or (3) Finding intersection point on precision-recall curve based on requirements.

**Threshold Impact:**
```
Threshold ↑ → Fewer positives → Precision ↑, Recall ↓
Threshold ↓ → More positives → Precision ↓, Recall ↑
```

**Threshold Selection Strategies:**

| Goal | Strategy |
|------|----------|
| Maximize F1 | Find threshold where F1 is highest |
| Minimum recall (e.g., 90%) | Lowest threshold that gives recall ≥ 90% |
| Minimum precision | Highest threshold that gives precision ≥ X |
| Minimize cost | Threshold minimizing: FP×cost_FP + FN×cost_FN |

**Algorithm Steps:**
1. Train model, get probability predictions on validation set
2. Generate thresholds: 0.01, 0.02, ..., 0.99
3. For each threshold, compute target metric
4. Select threshold with best metric value
5. Apply to test set (only once)

**Python Code:**
```python
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

y_true = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]
y_prob = [0.1, 0.3, 0.6, 0.8, 0.7, 0.2, 0.9, 0.4, 0.85, 0.55]

# Method 1: Maximize F1
thresholds = np.arange(0.1, 0.9, 0.05)
best_f1, best_thresh = 0, 0.5

for thresh in thresholds:
    y_pred = [1 if p >= thresh else 0 for p in y_prob]
    f1 = f1_score(y_true, y_pred)
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh

print(f"Optimal threshold: {best_thresh:.2f}, F1: {best_f1:.3f}")

# Method 2: Precision-Recall curve approach
precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)
# Find threshold for recall >= 0.8 with max precision
valid_idx = recall[:-1] >= 0.8
if valid_idx.any():
    best_idx = np.argmax(precision[:-1][valid_idx])
    recall_thresh = thresholds_pr[valid_idx][best_idx]
    print(f"Threshold for recall>=0.8: {recall_thresh:.2f}")
```

**Interview Tip:**
Never evaluate different thresholds on test set. Use validation set for threshold selection, then apply chosen threshold once on test set for final evaluation.

---

## Question 8
- [ ] Done
**What techniques help you visualize the trade-offs between precision and recall for stakeholder communication?**

**Answer:**

**Definition:**
Use **Precision-Recall Curve** to show trade-off across thresholds, **Confusion Matrix heatmap** for raw numbers, and **bar charts comparing metrics** for simplicity. For executives, translate to business terms: "catch rate vs false alarm rate."

**Visualization Techniques:**

| Visualization | Audience | Shows |
|--------------|----------|-------|
| PR Curve | Technical | Trade-off across thresholds |
| ROC Curve | Technical | TPR vs FPR trade-off |
| Confusion Matrix | Semi-technical | Raw TP, FP, FN, TN |
| Bar Chart | Non-technical | Simple metric comparison |
| Cost Curve | Business | Dollar impact at thresholds |

**Python Code:**
```python
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, confusion_matrix, 
                             ConfusionMatrixDisplay, PrecisionRecallDisplay)

y_true = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]
y_prob = [0.1, 0.3, 0.6, 0.8, 0.7, 0.2, 0.9, 0.4, 0.85, 0.55]
y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

# 1. Precision-Recall Curve
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
axes[0].plot(recall, precision, 'b-', linewidth=2)
axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Precision')
axes[0].set_title('Precision-Recall Trade-off')

# 2. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive']).plot(ax=axes[1])
axes[1].set_title('Confusion Matrix')

# 3. Simple Bar Chart for stakeholders
metrics = {'Precision': 0.83, 'Recall': 0.75, 'F1-Score': 0.79}
axes[2].bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange'])
axes[2].set_ylim(0, 1)
axes[2].set_title('Model Performance')

plt.tight_layout()
plt.show()
```

**Stakeholder Communication Tips:**
- Executives: "We catch 90% of fraud but have 20% false alarms"
- Use dollar amounts: "Each point of recall = $50K recovered"
- Show threshold slider impact: "Moving threshold from 0.5 to 0.3 increases catch rate from 80% to 95% but doubles false alarms"

**Interview Tip:**
Always translate metrics to business language for non-technical stakeholders. "Precision" means nothing to a VP; "false alarm rate" does.

---

## Question 9
- [ ] Done
**How do you implement confidence interval calculations for accuracy and F1-score in small dataset scenarios?**

**Answer:**

**Definition:**
For small datasets, point estimates are unreliable. Use **Bootstrap sampling** (resample with replacement, compute metric many times, take percentiles) or **Wilson score interval** for accuracy. Report metrics as ranges (e.g., "F1: 0.75 [0.68-0.82]") to convey uncertainty.

**Methods for Confidence Intervals:**

| Method | Use Case | Assumptions |
|--------|----------|-------------|
| Bootstrap | Any metric, small samples | Non-parametric |
| Wilson Score | Accuracy/proportions | Binomial distribution |
| Normal Approximation | Large samples only | CLT applies |

**Mathematical Formulation (Wilson Score for Accuracy):**
$$CI = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

Where $\hat{p}$ = accuracy, $n$ = sample size, $z$ = 1.96 for 95% CI

**Bootstrap Method (Preferred for F1):**
1. Resample (y_true, y_pred) with replacement, same size
2. Compute F1 on resampled data
3. Repeat 1000+ times
4. Take 2.5th and 97.5th percentiles

**Python Code:**
```python
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

y_true = [0, 1, 1, 0, 1, 1, 0, 1, 0, 0]  # Small dataset
y_pred = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0]

def bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=1000, ci=0.95):
    """Calculate confidence interval using bootstrap."""
    scores = []
    n = len(y_true)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.randint(0, n, size=n)
        y_true_boot = [y_true[i] for i in indices]
        y_pred_boot = [y_pred[i] for i in indices]
        
        # Calculate metric
        score = metric_func(y_true_boot, y_pred_boot)
        scores.append(score)
    
    # Calculate percentiles
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    mean = np.mean(scores)
    
    return mean, lower, upper

# Calculate CI for F1 and Accuracy
f1_mean, f1_low, f1_high = bootstrap_ci(y_true, y_pred, f1_score)
acc_mean, acc_low, acc_high = bootstrap_ci(y_true, y_pred, accuracy_score)

print(f"F1-Score: {f1_mean:.3f} [{f1_low:.3f} - {f1_high:.3f}]")
print(f"Accuracy: {acc_mean:.3f} [{acc_low:.3f} - {acc_high:.3f}]")
```

**Interview Tip:**
Always mention uncertainty when dataset is small. Saying "F1 is 0.85" on 50 samples is misleading; "F1 is 0.85 ± 0.12" is honest.

---

## Question 10
- [ ] Done
**When would you use Matthews Correlation Coefficient instead of F1-score for binary classification evaluation?**

**Answer:**

**Definition:**
Use **MCC** when you want a balanced metric that accounts for all four confusion matrix cells (TP, TN, FP, FN). Unlike F1-score which ignores TN, MCC gives a more complete picture, especially when both classes matter equally or when dealing with imbalanced datasets.

**Key Differences:**

| Aspect | F1-Score | MCC |
|--------|----------|-----|
| Uses TN | No | Yes |
| Range | [0, 1] | [-1, +1] |
| Balanced measure | Partial | Complete |
| Interpretability | Higher | Medium |
| Imbalance handling | Good | Better |

**Mathematical Formulation:**
$$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

**MCC Interpretation:**
- +1: Perfect prediction
- 0: Random prediction (no better than coin flip)
- -1: Complete disagreement

**When to Use MCC:**
- Both classes are equally important
- You care about correctly predicting negatives too
- Highly imbalanced data (MCC is more robust)
- Scientific/medical research requiring rigorous evaluation

**When to Use F1:**
- Only positive class matters
- Comparing with published benchmarks using F1
- Stakeholders understand F1 better

**Python Code:**
```python
from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix

y_true = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  # Imbalanced
y_pred = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

# Both metrics
f1 = f1_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

# Confusion matrix for context
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"F1-Score: {f1:.3f}")  # Ignores TN
print(f"MCC: {mcc:.3f}")      # Uses all four
```

**Interview Tip:**
MCC is preferred in academic ML papers and competitions (like Kaggle) for imbalanced problems. Mention it shows you know metrics beyond basics.

---

## Question 11
- [ ] Done
**How do you handle metric evaluation when dealing with hierarchical or nested class structures?**

**Answer:**

**Definition:**
In hierarchical classification (e.g., Animal → Mammal → Dog), standard flat metrics miss partial correctness. Use **hierarchical precision/recall** that gives partial credit for ancestors, or **level-wise metrics** that evaluate each hierarchy level separately.

**Problem with Flat Metrics:**
- True: Dog, Predicted: Cat → Flat F1 says completely wrong
- But both are Mammals and Animals → Should get partial credit

**Hierarchical Evaluation Strategies:**

| Strategy | Description |
|----------|-------------|
| **Hierarchical Precision** | Credit for correct ancestors |
| **Hierarchical Recall** | Penalize for missed descendants |
| **Level-wise Metrics** | Separate F1 for each level |
| **Tree-distance Loss** | Cost based on graph distance |

**Mathematical Formulation:**

For hierarchical precision/recall using ancestor sets:
$$h\text{-}Precision = \frac{|Ancestors(pred) \cap Ancestors(true)|}{|Ancestors(pred)|}$$

$$h\text{-}Recall = \frac{|Ancestors(pred) \cap Ancestors(true)|}{|Ancestors(true)|}$$

**Example:**
Hierarchy: Root → Animal → Mammal → Dog
- True: Dog → Ancestors: {Dog, Mammal, Animal, Root}
- Pred: Cat → Ancestors: {Cat, Mammal, Animal, Root}
- Overlap: {Mammal, Animal, Root} = 3 nodes
- h-Precision = 3/4 = 0.75 (partial credit!)

**Python Code:**
```python
# Simple level-wise evaluation approach
def hierarchical_metrics(y_true_hierarchy, y_pred_hierarchy):
    """
    y_true_hierarchy: list of lists [[level0, level1, level2], ...]
    y_pred_hierarchy: same format
    """
    from sklearn.metrics import f1_score
    
    n_levels = len(y_true_hierarchy[0])
    level_f1 = {}
    
    for level in range(n_levels):
        true_level = [y[level] for y in y_true_hierarchy]
        pred_level = [y[level] for y in y_pred_hierarchy]
        level_f1[f'Level_{level}'] = f1_score(true_level, pred_level, average='macro')
    
    return level_f1

# Example: Animal classification
# Format: [Kingdom, Class, Species]
y_true = [['Animal', 'Mammal', 'Dog'], 
          ['Animal', 'Bird', 'Eagle']]
y_pred = [['Animal', 'Mammal', 'Cat'],   # Wrong species, right class
          ['Animal', 'Mammal', 'Dog']]   # Wrong class and species

metrics = hierarchical_metrics(y_true, y_pred)
print(metrics)  # Shows F1 at each level
```

**Interview Tip:**
Mention that flat metrics underestimate performance in hierarchical problems. Proposing level-wise evaluation shows domain understanding.

---

## Question 12
- [ ] Done
**What are the best practices for reporting metric confidence when using cross-validation or bootstrap sampling?**

**Answer:**

**Definition:**
Report metrics with uncertainty: **mean ± standard deviation** from cross-validation folds, or **confidence intervals** from bootstrap. Always state the methodology (k-fold, stratified, bootstrap iterations) so results are reproducible.

**Best Practices:**

| Practice | Description |
|----------|-------------|
| Report mean ± std | Shows variability across folds |
| Use stratified CV | Maintains class balance in each fold |
| Report all fold scores | Reveals if one fold is anomalous |
| Use enough folds/iterations | k=5 or k=10; bootstrap ≥1000 |
| State random seed | For reproducibility |

**Reporting Format Examples:**
```
Good: "F1-Score: 0.847 ± 0.032 (5-fold stratified CV)"
Good: "Accuracy: 0.91 [0.88, 0.94] (95% CI, 1000 bootstrap)"
Bad:  "F1-Score: 0.85"  (no uncertainty!)
```

**Cross-Validation Reporting:**
$$\text{Mean} = \frac{1}{k}\sum_{i=1}^{k} m_i, \quad \text{Std} = \sqrt{\frac{1}{k}\sum_{i=1}^{k}(m_i - \text{Mean})^2}$$

**Python Code:**
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Sample data
X = np.random.randn(200, 10)
y = np.array([0]*100 + [1]*100)

model = RandomForestClassifier(random_state=42)

# Stratified K-Fold for classification
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Get scores for each fold
f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

# Report with uncertainty
print(f"F1-Score: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
print(f"Accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
print(f"Per-fold F1: {[f'{s:.3f}' for s in f1_scores]}")
```

**Red Flags in Reporting:**
- High variance across folds → unstable model or data issues
- One fold significantly different → possible data leakage or outlier fold
- Only reporting best fold → cherry-picking

**Interview Tip:**
If you report a single number without uncertainty, interviewers may question your statistical rigor. Always add "±" or confidence intervals.

---

## Question 13
- [ ] Done
**How do you implement custom F1-score variants for domain-specific evaluation requirements?**

**Answer:**

**Definition:**
Custom F1 variants modify the standard formula to fit domain needs: **F-beta score** adjusts precision-recall weighting, **class-weighted F1** prioritizes certain classes, and **sample-weighted F1** weights individual predictions. Implement by extending sklearn or writing custom functions.

**Common Customizations:**

| Variant | Use Case |
|---------|----------|
| **F-beta** | Adjust precision vs recall importance |
| **Class-weighted** | Some classes more important |
| **Threshold-specific** | Different thresholds per class |
| **Cost-sensitive** | Incorporate misclassification costs |

**F-beta Score Formula:**
$$F_\beta = (1 + \beta^2) \times \frac{Precision \times Recall}{\beta^2 \times Precision + Recall}$$

- β = 1 → Standard F1 (equal weight)
- β = 2 → F2 (Recall 2x more important)
- β = 0.5 → F0.5 (Precision 2x more important)

**Python Code:**
```python
from sklearn.metrics import fbeta_score, f1_score
import numpy as np

y_true = [0, 1, 1, 0, 1, 1, 0, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1, 1, 1, 1, 0, 1]

# 1. F-beta scores
f1 = fbeta_score(y_true, y_pred, beta=1)      # Standard
f2 = fbeta_score(y_true, y_pred, beta=2)      # Recall-focused
f05 = fbeta_score(y_true, y_pred, beta=0.5)   # Precision-focused

print(f"F1: {f1:.3f}, F2: {f2:.3f}, F0.5: {f05:.3f}")

# 2. Custom class-weighted F1 for multi-class
def custom_weighted_f1(y_true, y_pred, class_weights):
    """F1 with custom importance weights per class."""
    classes = np.unique(y_true)
    weighted_f1 = 0
    total_weight = sum(class_weights.values())
    
    for cls in classes:
        # Binary conversion for this class
        y_true_bin = [1 if y == cls else 0 for y in y_true]
        y_pred_bin = [1 if y == cls else 0 for y in y_pred]
        
        cls_f1 = f1_score(y_true_bin, y_pred_bin)
        weighted_f1 += class_weights.get(cls, 1) * cls_f1
    
    return weighted_f1 / total_weight

# Example: Class 2 is 3x more important
y_true_mc = [0, 1, 2, 0, 1, 2, 0, 1, 2, 2]
y_pred_mc = [0, 1, 1, 0, 0, 2, 0, 1, 2, 2]
weights = {0: 1, 1: 1, 2: 3}

custom_f1 = custom_weighted_f1(y_true_mc, y_pred_mc, weights)
print(f"Custom Weighted F1: {custom_f1:.3f}")
```

**Interview Tip:**
When asked about domain-specific metrics, first ask: "What errors are more costly?" Then propose F-beta with appropriate beta value.

---

## Question 14
- [ ] Done
**When should you use balanced accuracy versus regular accuracy in imbalanced classification problems?**

**Answer:**

**Definition:**
Use **Balanced Accuracy** when classes are imbalanced and you want equal importance to each class's performance. It averages recall across classes, preventing majority class from dominating. Regular accuracy should only be used when classes are balanced and misclassification costs are equal.

**Comparison:**

| Metric | Formula | Imbalanced Data |
|--------|---------|-----------------|
| Accuracy | (TP+TN)/Total | Misleading (majority dominates) |
| Balanced Accuracy | (TPR + TNR)/2 | Fair (equal class weight) |

**Mathematical Formulation:**
$$\text{Balanced Accuracy} = \frac{1}{2}\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right) = \frac{Recall + Specificity}{2}$$

For multi-class:
$$\text{Balanced Accuracy} = \frac{1}{C}\sum_{c=1}^{C} Recall_c$$

**Example:**
- Data: 950 negative, 50 positive
- Model predicts all negative
- Regular Accuracy: 950/1000 = 95% (looks great!)
- Balanced Accuracy: (100% + 0%) / 2 = 50% (reveals problem)

**When to Use:**

| Scenario | Use |
|----------|-----|
| Balanced classes | Regular Accuracy |
| Imbalanced classes | Balanced Accuracy |
| All classes equally important | Balanced Accuracy |
| Majority class more important | Regular Accuracy |

**Python Code:**
```python
from sklearn.metrics import balanced_accuracy_score, accuracy_score

# Imbalanced scenario
y_true = [0]*95 + [1]*5           # 95% class 0, 5% class 1
y_pred = [0]*100                   # Model predicts all 0

regular_acc = accuracy_score(y_true, y_pred)
balanced_acc = balanced_accuracy_score(y_true, y_pred)

print(f"Regular Accuracy: {regular_acc:.2f}")    # 0.95 - misleading!
print(f"Balanced Accuracy: {balanced_acc:.2f}") # 0.50 - honest!

# A better model
y_pred_better = [0]*92 + [1]*8  # Tries to predict some 1s
balanced_acc_better = balanced_accuracy_score(y_true, y_pred_better)
print(f"Better model Balanced Acc: {balanced_acc_better:.2f}")
```

**Interview Tip:**
If interviewer mentions imbalanced data and asks about accuracy, immediately point out the limitation and suggest balanced accuracy or F1-score as alternatives.

---

## Question 15
- [ ] Done
**How do you handle metric calculation for multi-output classification problems with different evaluation priorities?**

**Answer:**

**Definition:**
In multi-output classification (multiple target variables per sample), calculate metrics per output, then aggregate using **output-weighted average** based on business priority. Some outputs may be more critical and deserve higher weights in the final score.

**Problem Structure:**
- Input: X (features)
- Output: [y1, y2, y3] (multiple classification targets)
- Each output may have different importance

**Evaluation Strategies:**

| Strategy | Description |
|----------|-------------|
| Per-output metrics | Calculate F1 for each output separately |
| Uniform average | Simple mean across outputs |
| Weighted average | Weight by output importance |
| Sample-based | Average metrics across samples |

**Python Code:**
```python
from sklearn.metrics import f1_score
import numpy as np

# Multi-output: 3 outputs per sample
y_true = np.array([
    [0, 1, 1],  # Sample 1
    [1, 1, 0],  # Sample 2
    [0, 0, 1],  # Sample 3
    [1, 1, 1],  # Sample 4
])

y_pred = np.array([
    [0, 1, 0],  # Sample 1
    [1, 0, 0],  # Sample 2
    [0, 0, 1],  # Sample 3
    [1, 1, 1],  # Sample 4
])

# Output importance weights (business-defined)
output_weights = [0.5, 0.3, 0.2]  # Output 1 most important
output_names = ['Priority-High', 'Priority-Medium', 'Priority-Low']

# Calculate F1 per output
per_output_f1 = []
for i in range(y_true.shape[1]):
    f1 = f1_score(y_true[:, i], y_pred[:, i])
    per_output_f1.append(f1)
    print(f"{output_names[i]} F1: {f1:.3f}")

# Weighted aggregate
weighted_f1 = sum(w * f for w, f in zip(output_weights, per_output_f1))
uniform_f1 = np.mean(per_output_f1)

print(f"\nUniform Average F1: {uniform_f1:.3f}")
print(f"Weighted Average F1: {weighted_f1:.3f}")

# Sample-based approach (how many outputs correct per sample)
sample_scores = []
for i in range(len(y_true)):
    correct = sum(y_true[i] == y_pred[i])
    sample_scores.append(correct / len(y_true[i]))
print(f"Sample-based Accuracy: {np.mean(sample_scores):.3f}")
```

**Decision Framework:**
1. Define importance weights with stakeholders
2. Calculate per-output metrics
3. Report both individual and weighted aggregate
4. Track which outputs degrade over time

**Interview Tip:**
Always clarify output priorities before proposing aggregation. "Are all outputs equally important, or should we weight them?"

---

## Question 16
- [ ] Done
**What strategies help you communicate metric trade-offs to non-technical stakeholders effectively?**

**Answer:**

**Definition:**
Translate technical metrics into business language: **Precision** → "Of alerts raised, how many are real?", **Recall** → "Of real cases, how many do we catch?". Use concrete examples, dollar amounts, and visual comparisons. Avoid jargon.

**Translation Table:**

| Technical Term | Business Language |
|----------------|-------------------|
| Precision | "Accuracy of our alerts" / "Hit rate" |
| Recall | "Catch rate" / "Coverage" |
| False Positive | "False alarm" / "Wasted investigation" |
| False Negative | "Missed case" / "Slipped through" |
| F1-Score | "Overall effectiveness" |
| Threshold | "Sensitivity dial" |

**Communication Strategies:**

**1. Use Concrete Numbers:**
- Bad: "Recall is 0.85"
- Good: "We catch 85 out of every 100 fraud cases"

**2. Monetize the Trade-off:**
- "If we increase catch rate from 80% to 95%, we save $500K in fraud but add $200K in investigation costs"

**3. Use Analogies:**
- "It's like a spam filter - we can block more spam but risk blocking real emails"

**4. Interactive Threshold Demo:**
```
Threshold Setting:  Conservative ←——→ Aggressive
Fraud Caught:           70%              95%
False Alarms/day:       10               100
Investigation Cost:    $500            $5,000
```

**5. Visual Trade-off Chart:**
```python
import matplotlib.pyplot as plt

thresholds = ['Conservative', 'Balanced', 'Aggressive']
fraud_caught = [70, 85, 95]
false_alarms = [10, 40, 100]

fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.bar(thresholds, fraud_caught, color='green', alpha=0.7, label='Fraud Caught %')
ax1.set_ylabel('Fraud Caught (%)', color='green')

ax2 = ax1.twinx()
ax2.plot(thresholds, false_alarms, 'ro-', linewidth=2, label='False Alarms')
ax2.set_ylabel('False Alarms per Day', color='red')

plt.title('Trade-off: Catching More Fraud vs. More False Alarms')
plt.show()
```

**Stakeholder-Specific Framing:**
- **Executives**: Focus on revenue/cost impact
- **Operations**: Focus on workload implications
- **Legal/Compliance**: Focus on risk of missed cases

**Interview Tip:**
Demonstrating ability to translate technical concepts to business terms is a senior skill. Always offer to explain in "business impact" terms.

---

## Question 17
- [ ] Done
**How do you implement real-time metric monitoring for deployed models without ground truth labels?**

**Answer:**

**Definition:**
Without immediate ground truth, monitor **proxy metrics**: prediction distribution shifts, confidence scores, feature drift, and latency. Set up alerts for anomalies. When delayed labels arrive, compute actual accuracy retrospectively.

**Monitoring Strategies:**

| Metric Type | What to Monitor | Alert Condition |
|-------------|-----------------|-----------------|
| **Prediction Distribution** | Class ratio over time | Sudden shift (e.g., 20% → 50% positive) |
| **Confidence Scores** | Mean/median confidence | Drop in average confidence |
| **Feature Drift** | Input distribution | Statistical distance increase |
| **Latency** | Inference time | P99 latency spike |
| **Error Rates** | API errors, timeouts | Any increase |

**Delayed Ground Truth:**
- Many applications get labels later (e.g., fraud confirmed after investigation)
- Compute actual F1/accuracy when labels arrive
- Compare predicted vs actual distributions

**Python Code:**
```python
import numpy as np
from scipy import stats

class ModelMonitor:
    def __init__(self, baseline_pred_ratio, baseline_confidence):
        self.baseline_pred_ratio = baseline_pred_ratio  # e.g., 0.1 (10% positive)
        self.baseline_confidence = baseline_confidence  # e.g., 0.85
        self.alerts = []
    
    def check_prediction_drift(self, predictions, threshold=0.05):
        """Check if positive prediction ratio drifted."""
        current_ratio = np.mean(predictions)
        drift = abs(current_ratio - self.baseline_pred_ratio)
        
        if drift > threshold:
            self.alerts.append(f"Prediction drift: {self.baseline_pred_ratio:.2f} → {current_ratio:.2f}")
        return drift
    
    def check_confidence_drop(self, confidences, threshold=0.1):
        """Check if model confidence dropped."""
        current_conf = np.mean(confidences)
        drop = self.baseline_confidence - current_conf
        
        if drop > threshold:
            self.alerts.append(f"Confidence drop: {self.baseline_confidence:.2f} → {current_conf:.2f}")
        return drop
    
    def report(self):
        if self.alerts:
            print("ALERTS:", self.alerts)
        else:
            print("No anomalies detected")

# Usage
monitor = ModelMonitor(baseline_pred_ratio=0.1, baseline_confidence=0.85)

# Simulated production predictions
predictions = [0, 0, 1, 0, 1, 1, 0, 1, 0, 1]  # 50% positive (drift!)
confidences = [0.7, 0.6, 0.8, 0.65, 0.75, 0.7, 0.6, 0.8, 0.55, 0.72]

monitor.check_prediction_drift(predictions)
monitor.check_confidence_drop(confidences)
monitor.report()
```

**Key Monitoring Dashboards:**
1. Rolling prediction class distribution
2. Confidence score histogram
3. Feature distribution comparison (baseline vs current)
4. Delayed accuracy metrics (when labels available)

**Interview Tip:**
Mention that production ML monitoring is different from offline evaluation. Show awareness of drift detection and proxy metrics.

---

## Question 18
- [ ] Done
**When would you use precision@k or recall@k metrics instead of traditional precision and recall?**

**Answer:**

**Definition:**
Use **Precision@K** and **Recall@K** in ranking/recommendation systems where you only care about top K results. Precision@K measures "of top K recommendations, how many are relevant?" Recall@K measures "of all relevant items, how many appear in top K?"

**When to Use:**

| Scenario | Metric | Reason |
|----------|--------|--------|
| Search engine | Precision@10 | Users only see first page |
| Recommendation | Recall@K | Want to surface relevant items in top K |
| Information retrieval | Both | Balance relevance and coverage |
| Traditional classification | Standard P/R | No ranking involved |

**Mathematical Formulation:**
$$Precision@K = \frac{|\text{Relevant items in top K}|}{K}$$

$$Recall@K = \frac{|\text{Relevant items in top K}|}{|\text{Total relevant items}|}$$

**Example:**
- 100 items total, 10 are relevant
- Model returns top 5: [R, R, X, R, X] (R=relevant, X=not)
- Precision@5 = 3/5 = 0.6
- Recall@5 = 3/10 = 0.3

**Python Code:**
```python
def precision_at_k(y_true, y_scores, k):
    """
    y_true: list of binary labels (1 = relevant)
    y_scores: model scores (higher = more confident)
    k: number of top results to consider
    """
    # Sort by score descending
    sorted_indices = sorted(range(len(y_scores)), key=lambda i: y_scores[i], reverse=True)
    top_k_indices = sorted_indices[:k]
    
    relevant_in_k = sum(y_true[i] for i in top_k_indices)
    return relevant_in_k / k

def recall_at_k(y_true, y_scores, k):
    """Recall@K: What fraction of relevant items are in top K."""
    sorted_indices = sorted(range(len(y_scores)), key=lambda i: y_scores[i], reverse=True)
    top_k_indices = sorted_indices[:k]
    
    relevant_in_k = sum(y_true[i] for i in top_k_indices)
    total_relevant = sum(y_true)
    
    return relevant_in_k / total_relevant if total_relevant > 0 else 0

# Example
y_true = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]  # 5 relevant items
y_scores = [0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.6, 0.4, 0.5, 0.15]

for k in [3, 5, 10]:
    p_k = precision_at_k(y_true, y_scores, k)
    r_k = recall_at_k(y_true, y_scores, k)
    print(f"K={k}: Precision@K={p_k:.2f}, Recall@K={r_k:.2f}")
```

**Interview Tip:**
For any ranking problem (search, recommendations, retrieval), immediately mention Precision@K and Recall@K instead of standard metrics. Shows domain awareness.

---

## Question 19
- [ ] Done
**How do you handle metric evaluation for streaming data with concept drift and evolving class distributions?**

**Answer:**

**Definition:**
In streaming data, the relationship between features and labels can change over time (concept drift). Use **sliding window metrics** (evaluate on recent data), **decay-weighted metrics** (recent samples weighted more), and **drift detection** to trigger model retraining.

**Types of Drift:**

| Drift Type | Description | Impact on Metrics |
|------------|-------------|-------------------|
| **Concept Drift** | P(Y\|X) changes | Model predictions become wrong |
| **Data Drift** | P(X) changes | Features shift distribution |
| **Label Drift** | P(Y) changes | Class balance changes |

**Evaluation Strategies:**

**1. Sliding Window Evaluation:**
- Compute metrics on last N samples only
- Reflects current model performance

**2. Time-Decayed Metrics:**
- Recent errors weighted more than old ones
- Exponential decay: weight = exp(-λ × age)

**3. Drift Detection:**
- Monitor metric over time
- Alert when significant drop detected

**Python Code:**
```python
import numpy as np
from collections import deque

class StreamingMetricsMonitor:
    def __init__(self, window_size=100, decay_factor=0.99):
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.y_true_window = deque(maxlen=window_size)
        self.y_pred_window = deque(maxlen=window_size)
        self.historical_accuracy = []
    
    def update(self, y_true, y_pred):
        """Add new prediction to stream."""
        self.y_true_window.append(y_true)
        self.y_pred_window.append(y_pred)
    
    def sliding_accuracy(self):
        """Accuracy on sliding window."""
        if len(self.y_true_window) == 0:
            return 0
        correct = sum(t == p for t, p in zip(self.y_true_window, self.y_pred_window))
        return correct / len(self.y_true_window)
    
    def decayed_accuracy(self):
        """Accuracy with exponential decay (recent samples matter more)."""
        if len(self.y_true_window) == 0:
            return 0
        
        weighted_correct = 0
        total_weight = 0
        
        for i, (t, p) in enumerate(zip(self.y_true_window, self.y_pred_window)):
            weight = self.decay_factor ** (len(self.y_true_window) - 1 - i)
            weighted_correct += weight * (t == p)
            total_weight += weight
        
        return weighted_correct / total_weight
    
    def detect_drift(self, threshold=0.1):
        """Simple drift detection: compare recent vs older performance."""
        if len(self.y_true_window) < self.window_size:
            return False
        
        mid = len(self.y_true_window) // 2
        old_acc = sum(t == p for t, p in list(zip(self.y_true_window, self.y_pred_window))[:mid]) / mid
        new_acc = sum(t == p for t, p in list(zip(self.y_true_window, self.y_pred_window))[mid:]) / (len(self.y_true_window) - mid)
        
        return (old_acc - new_acc) > threshold

# Simulation
monitor = StreamingMetricsMonitor(window_size=50)

# Simulate stream with drift at t=100
for t in range(150):
    y_true = np.random.randint(0, 2)
    
    # Model degrades after t=100 (concept drift)
    if t < 100:
        y_pred = y_true if np.random.random() > 0.1 else 1 - y_true  # 90% accurate
    else:
        y_pred = y_true if np.random.random() > 0.4 else 1 - y_true  # 60% accurate
    
    monitor.update(y_true, y_pred)

print(f"Sliding Accuracy: {monitor.sliding_accuracy():.3f}")
print(f"Drift Detected: {monitor.detect_drift()}")
```

**Interview Tip:**
For streaming scenarios, mention that static train/test splits don't work. Propose sliding windows and drift detection.

---

## Question 20
- [ ] Done
**What techniques help you assess metric stability across different data splits and validation strategies?**

**Answer:**

**Definition:**
Metric stability refers to how consistent performance is across different data splits. Assess using **multiple cross-validation runs**, **variance analysis**, and **stability indices**. High variance indicates model is sensitive to data composition; low variance suggests robust performance.

**Stability Assessment Techniques:**

| Technique | What It Reveals |
|-----------|-----------------|
| **Repeated K-Fold CV** | Variance across different folds |
| **Multiple Random Seeds** | Sensitivity to initialization |
| **Bootstrap Variance** | Confidence interval width |
| **Leave-One-Out** | Per-sample influence |
| **Nested CV** | Hyperparameter stability |

**Stability Metrics:**
$$\text{Coefficient of Variation} = \frac{\text{Std Dev}}{\text{Mean}} \times 100\%$$

- CV < 5%: Very stable
- CV 5-15%: Moderately stable
- CV > 15%: Unstable, investigate

**Python Code:**
```python
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Sample data
np.random.seed(42)
X = np.random.randn(200, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

model = RandomForestClassifier(n_estimators=50, random_state=42)

# Method 1: Repeated K-Fold (different splits)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1')

print(f"F1 Scores across 50 folds: {scores.mean():.3f} ± {scores.std():.3f}")
print(f"Coefficient of Variation: {(scores.std()/scores.mean())*100:.1f}%")
print(f"Range: [{scores.min():.3f}, {scores.max():.3f}]")

# Method 2: Different random seeds
seed_scores = []
for seed in range(10):
    cv_seed = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
    mean_score = cross_val_score(model, X, y, cv=cv_seed, scoring='f1').mean()
    seed_scores.append(mean_score)

print(f"\nAcross seeds: {np.mean(seed_scores):.3f} ± {np.std(seed_scores):.3f}")

# Stability assessment
def assess_stability(scores, threshold=0.10):
    cv = scores.std() / scores.mean()
    if cv < 0.05:
        return "Very Stable"
    elif cv < threshold:
        return "Moderately Stable"
    else:
        return "Unstable - Investigate"

print(f"Stability: {assess_stability(scores)}")
```

**Red Flags:**
- One fold significantly different from others
- High variance compared to mean
- Results change drastically with random seed

**Interview Tip:**
If model performance varies widely across folds, investigate: data leakage, small dataset, or model overfitting to specific patterns.

---

## Question 21
- [ ] Done
**How do you implement cost-sensitive evaluation metrics that account for different misclassification costs?**

**Answer:**

**Definition:**
Cost-sensitive metrics weight errors by their business cost instead of treating all errors equally. Define a **cost matrix** (FP cost, FN cost), then compute **total cost** or **cost-weighted accuracy**. Optimize threshold to minimize expected cost.

**Cost Matrix:**
|  | Predicted Negative | Predicted Positive |
|--|--------------------|--------------------|
| **Actual Negative** | 0 (TN) | C_FP (False Alarm) |
| **Actual Positive** | C_FN (Missed) | 0 (TP) |

**Mathematical Formulation:**
$$\text{Total Cost} = FP \times C_{FP} + FN \times C_{FN}$$

$$\text{Normalized Cost} = \frac{FP \times C_{FP} + FN \times C_{FN}}{N \times \max(C_{FP}, C_{FN})}$$

**Example Costs:**
- Fraud detection: C_FN = $500 (missed fraud), C_FP = $50 (investigation)
- Medical: C_FN = $10,000 (missed disease), C_FP = $100 (extra test)

**Python Code:**
```python
from sklearn.metrics import confusion_matrix
import numpy as np

def cost_sensitive_evaluation(y_true, y_pred, cost_fp, cost_fn):
    """Calculate cost-based metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    total_cost = fp * cost_fp + fn * cost_fn
    cost_per_sample = total_cost / len(y_true)
    
    # Cost-weighted accuracy (considering costs)
    max_cost = len(y_true) * max(cost_fp, cost_fn)
    normalized_cost = total_cost / max_cost
    cost_accuracy = 1 - normalized_cost
    
    return {
        'total_cost': total_cost,
        'cost_per_sample': cost_per_sample,
        'cost_accuracy': cost_accuracy,
        'fp': fp, 'fn': fn
    }

def find_optimal_threshold(y_true, y_prob, cost_fp, cost_fn):
    """Find threshold that minimizes total cost."""
    best_threshold = 0.5
    min_cost = float('inf')
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = [1 if p >= threshold else 0 for p in y_prob]
        result = cost_sensitive_evaluation(y_true, y_pred, cost_fp, cost_fn)
        
        if result['total_cost'] < min_cost:
            min_cost = result['total_cost']
            best_threshold = threshold
    
    return best_threshold, min_cost

# Example
y_true = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]
y_prob = [0.1, 0.3, 0.6, 0.8, 0.7, 0.2, 0.9, 0.4, 0.85, 0.55]

# Fraud scenario: Missing fraud is 10x worse than false alarm
cost_fp, cost_fn = 50, 500

# Default threshold
y_pred_default = [1 if p >= 0.5 else 0 for p in y_prob]
result_default = cost_sensitive_evaluation(y_true, y_pred_default, cost_fp, cost_fn)
print(f"Default (0.5): Cost=${result_default['total_cost']}")

# Optimal threshold
opt_thresh, opt_cost = find_optimal_threshold(y_true, y_prob, cost_fp, cost_fn)
print(f"Optimal ({opt_thresh:.2f}): Cost=${opt_cost}")
```

**Interview Tip:**
Always ask about misclassification costs early. A model optimized for accuracy may be suboptimal for business value.

---

## Question 22
- [ ] Done
**When should you use per-class precision and recall versus aggregate metrics for multi-class evaluation?**

**Answer:**

**Definition:**
Use **per-class metrics** when you need to identify underperforming classes for targeted improvement. Use **aggregate metrics** (macro/micro/weighted) for overall model comparison. Report both in practice: aggregate for summary, per-class for diagnosis.

**Comparison:**

| Metric Type | Use Case | Shows |
|-------------|----------|-------|
| **Per-class** | Debugging, class-specific analysis | Which classes fail |
| **Macro-average** | Overall comparison, equal class importance | Mean performance |
| **Micro-average** | Overall accuracy-like measure | Global TP/FP/FN |
| **Weighted** | When class frequency matters | Support-weighted |

**When to Use Per-Class:**
- Debugging model failures
- Stakeholders care about specific classes
- Class imbalance (identify minority class performance)
- Iterative improvement cycles

**When to Use Aggregate:**
- Comparing different models
- Quick summary for stakeholders
- Hyperparameter tuning (single optimization target)

**Python Code:**
```python
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np

y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2]
y_pred = [0, 0, 1, 1, 1, 0, 2, 2, 2, 1, 2, 2]

class_names = ['Class_A', 'Class_B', 'Class_C']

# Method 1: Full classification report (best for analysis)
print("=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names))

# Method 2: Programmatic access to per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average=None
)

print("\n=== Per-Class Breakdown ===")
for i, name in enumerate(class_names):
    print(f"{name}: P={precision[i]:.2f}, R={recall[i]:.2f}, F1={f1[i]:.2f}, N={support[i]}")

# Method 3: Aggregate metrics
print("\n=== Aggregate Metrics ===")
for avg in ['macro', 'micro', 'weighted']:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg)
    print(f"{avg.capitalize():8s}: P={p:.2f}, R={r:.2f}, F1={f:.2f}")

# Identify worst-performing class
worst_class = class_names[np.argmin(f1)]
print(f"\n⚠ Lowest F1: {worst_class} (consider resampling or feature engineering)")
```

**Practical Workflow:**
1. Train model, get aggregate F1 for comparison
2. Examine per-class metrics to find weak spots
3. Address worst class (more data, resampling, etc.)
4. Re-evaluate and iterate

**Interview Tip:**
Never just report macro F1. Show awareness by saying: "I'd also check per-class metrics to ensure no class is being ignored."

---

## Question 23
- [ ] Done
**How do you handle metric evaluation for imbalanced time-series classification problems?**

**Answer:**

**Definition:**
Imbalanced time-series adds two challenges: class imbalance AND temporal ordering. Use **time-aware splits** (no future data leakage), **balanced metrics** (F1, AUC-PR), and **event-based evaluation** for rare events. Never use random shuffle splits.

**Key Challenges:**

| Challenge | Solution |
|-----------|----------|
| Random split causes leakage | Use time-based train/test split |
| Class imbalance | Use F1, AUC-PR instead of accuracy |
| Rare events clustered | Event-based rather than point-wise evaluation |
| Temporal patterns | Consider detection latency in metrics |

**Evaluation Strategies:**

**1. Time-Based Splitting:**
- Train: First 70% of timeline
- Validation: Next 15%
- Test: Last 15%
- NO random shuffling!

**2. Event-Based Metrics:**
- For rare events (faults, anomalies), evaluate event detection
- True positive = event detected within tolerance window
- Precision: How many alarms correspond to real events?
- Recall: How many real events were detected?

**Python Code:**
```python
import numpy as np
from sklearn.metrics import f1_score, average_precision_score

# Time-series data (chronological order)
timestamps = list(range(100))
y_true = [0]*80 + [1]*5 + [0]*10 + [1]*3 + [0]*2  # Rare events
y_prob = np.random.random(100)
y_prob[80:85] = y_prob[80:85] + 0.5  # Model detects some events
y_prob[95:98] = y_prob[95:98] + 0.3

# Time-based split (NOT random!)
train_end = int(0.7 * len(y_true))
val_end = int(0.85 * len(y_true))

y_train = y_true[:train_end]
y_val = y_true[train_end:val_end]
y_test = y_true[val_end:]
y_prob_test = y_prob[val_end:]

print(f"Train period: 0-{train_end}")
print(f"Test period: {val_end}-{len(y_true)}")
print(f"Test class distribution: {sum(y_test)}/{len(y_test)} positive")

# Use imbalance-aware metrics
y_pred_test = [1 if p > 0.5 else 0 for p in y_prob_test]
f1 = f1_score(y_test, y_pred_test)
auc_pr = average_precision_score(y_test, y_prob_test)

print(f"F1-Score: {f1:.3f}")
print(f"AUC-PR: {auc_pr:.3f}")

# Event-based evaluation with tolerance window
def event_based_recall(y_true, y_pred, tolerance=2):
    """Check if events are detected within tolerance window."""
    event_indices = [i for i, y in enumerate(y_true) if y == 1]
    detected = 0
    
    for event_idx in event_indices:
        # Check if any prediction within window
        start = max(0, event_idx - tolerance)
        end = min(len(y_pred), event_idx + tolerance + 1)
        if any(y_pred[start:end]):
            detected += 1
    
    return detected / len(event_indices) if event_indices else 0

event_recall = event_based_recall(y_test, y_pred_test, tolerance=1)
print(f"Event-based Recall (tolerance=1): {event_recall:.3f}")
```

**Interview Tip:**
For time-series, immediately mention "temporal validation" to avoid data leakage. Interviewers specifically look for this awareness.

---

## Question 24
- [ ] Done
**What are the best practices for comparing model performance across different metric combinations?**

**Answer:**

**Definition:**
No single metric tells the complete story. Compare models using a **metric dashboard** (multiple metrics), **radar charts** for visualization, and **decision matrices** weighted by business priorities. Rank models using a composite score when needed.

**Best Practices:**

| Practice | Description |
|----------|-------------|
| Report multiple metrics | Accuracy, F1, AUC-ROC, AUC-PR |
| Use primary + secondary | One for selection, others for context |
| Visualize trade-offs | Radar charts, parallel coordinates |
| Weight by importance | Create composite score |
| Statistical testing | Ensure differences are significant |

**Comparison Framework:**

1. **Define Metric Hierarchy:**
   - Primary: Business-critical (e.g., Recall for fraud)
   - Secondary: Supporting metrics (Precision, F1)
   - Tertiary: Efficiency (latency, model size)

2. **Create Comparison Table:**
   | Model | F1 | Recall | Precision | AUC | Latency |
   |-------|----|----|----|----|---|

3. **Composite Score (if needed):**
   $$Score = w_1 \cdot F1 + w_2 \cdot Recall + w_3 \cdot (1/Latency)$$

**Python Code:**
```python
import numpy as np

# Model comparison data
models = {
    'Logistic Regression': {'f1': 0.75, 'recall': 0.80, 'precision': 0.70, 'auc': 0.82, 'latency_ms': 5},
    'Random Forest':       {'f1': 0.82, 'recall': 0.78, 'precision': 0.86, 'auc': 0.88, 'latency_ms': 50},
    'XGBoost':            {'f1': 0.85, 'recall': 0.82, 'precision': 0.88, 'auc': 0.91, 'latency_ms': 30},
    'Neural Network':      {'f1': 0.83, 'recall': 0.85, 'precision': 0.81, 'auc': 0.89, 'latency_ms': 100},
}

# Business weights (sum to 1)
weights = {'f1': 0.3, 'recall': 0.4, 'precision': 0.2, 'auc': 0.1}

# Calculate composite scores
print("=== Model Comparison ===")
print(f"{'Model':<20} {'F1':>6} {'Recall':>8} {'Precision':>10} {'AUC':>6} {'Composite':>10}")
print("-" * 65)

scores = {}
for model, metrics in models.items():
    composite = sum(weights[k] * metrics[k] for k in weights)
    scores[model] = composite
    print(f"{model:<20} {metrics['f1']:>6.2f} {metrics['recall']:>8.2f} {metrics['precision']:>10.2f} {metrics['auc']:>6.2f} {composite:>10.3f}")

# Winner
winner = max(scores, key=scores.get)
print(f"\nRecommended: {winner} (highest composite score)")

# Caveat: Check if differences are meaningful
sorted_scores = sorted(scores.values(), reverse=True)
if sorted_scores[0] - sorted_scores[1] < 0.02:
    print("Note: Top models are very close. Consider other factors (interpretability, latency).")
```

**Decision Matrix Approach:**
- If recall is critical → NN wins
- If precision matters → XGBoost wins
- If latency matters → Logistic Regression wins
- Balanced → Composite score

**Interview Tip:**
Show that you consider multiple metrics, not just accuracy. Present trade-offs clearly: "Model A has better recall, but Model B has better precision."

---

## Question 25
- [ ] Done
**How do you implement statistical significance testing for metric differences between competing models?**

**Answer:**

**Definition:**
Metric differences may be due to random chance, not true superiority. Use statistical tests to verify significance: **McNemar's test** for paired binary predictions, **paired t-test** for CV scores, or **bootstrap test** for any metric. Report p-values alongside metrics.

**Common Tests:**

| Test | Use Case | Requirements |
|------|----------|--------------|
| **McNemar's Test** | Binary classification, same test set | Paired predictions |
| **Paired t-test** | Cross-validation scores | Normally distributed |
| **Wilcoxon Signed-Rank** | Non-normal CV scores | Non-parametric |
| **Bootstrap Test** | Any metric, any distribution | Resampling |

**McNemar's Test:**
Compares disagreement between two models on same test set.

|  | Model B Correct | Model B Wrong |
|--|-----------------|---------------|
| Model A Correct | a | b |
| Model A Wrong | c | d |

$$\chi^2 = \frac{(b-c)^2}{b+c}$$

Significant if p < 0.05

**Python Code:**
```python
from scipy import stats
import numpy as np

# Method 1: McNemar's Test (paired predictions)
def mcnemar_test(y_true, pred_a, pred_b):
    """Compare two models on same test set."""
    # Count disagreements
    b = sum((pred_a[i] == y_true[i]) and (pred_b[i] != y_true[i]) for i in range(len(y_true)))
    c = sum((pred_a[i] != y_true[i]) and (pred_b[i] == y_true[i]) for i in range(len(y_true)))
    
    if b + c == 0:
        return 1.0  # No disagreement
    
    # Chi-squared statistic
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # With continuity correction
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return p_value

y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0]
pred_a = [0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0]  # Model A
pred_b = [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0]  # Model B

p_value = mcnemar_test(y_true, pred_a, pred_b)
print(f"McNemar's p-value: {p_value:.4f}")
print(f"Significant difference: {p_value < 0.05}")

# Method 2: Paired t-test on CV scores
scores_a = [0.82, 0.85, 0.80, 0.83, 0.84]  # 5-fold CV for Model A
scores_b = [0.78, 0.81, 0.79, 0.80, 0.82]  # 5-fold CV for Model B

t_stat, p_value_t = stats.ttest_rel(scores_a, scores_b)
print(f"\nPaired t-test p-value: {p_value_t:.4f}")
print(f"Model A better: {np.mean(scores_a) > np.mean(scores_b) and p_value_t < 0.05}")

# Method 3: Bootstrap significance test
def bootstrap_significance(metric_a, metric_b, n_bootstrap=1000):
    """Test if difference is significant via bootstrap."""
    observed_diff = metric_a - metric_b
    
    # Combined and permute
    combined = [observed_diff] * n_bootstrap
    count_extreme = sum(abs(np.random.choice([-1, 1]) * observed_diff) >= abs(observed_diff) 
                       for _ in range(n_bootstrap))
    
    return count_extreme / n_bootstrap

# Typically use this with bootstrap resampling of the data
```

**Interpretation:**
- p < 0.05: Difference is statistically significant
- p >= 0.05: Cannot conclude one model is better
- Always report effect size alongside p-value

**Interview Tip:**
Don't claim "Model A is better" without significance testing. Say: "Model A has higher F1, and McNemar's test confirms this is statistically significant (p=0.02)."

---

## Question 26
- [ ] Done
**When would you use harmonic mean versus arithmetic mean for combining precision and recall?**

**Answer:**

**Definition:**
Use **Harmonic Mean** (F1-score) when you want to penalize extreme imbalances between precision and recall. Harmonic mean is lower than arithmetic mean and is dominated by the smaller value. Use arithmetic mean only when both metrics contribute equally regardless of their balance.

**Comparison:**

| Mean Type | Formula | Property |
|-----------|---------|----------|
| Arithmetic | (P + R) / 2 | Average value |
| Harmonic | 2PR / (P + R) | Penalizes imbalance |
| Geometric | √(P × R) | Middle ground |

**Mathematical Intuition:**
For P = 0.9, R = 0.1:
- Arithmetic Mean = (0.9 + 0.1) / 2 = 0.50 (seems okay)
- Harmonic Mean = 2 × 0.9 × 0.1 / (0.9 + 0.1) = 0.18 (reveals problem!)

**Why Harmonic Mean for F1:**
- Precision = 100%, Recall = 0% → F1 = 0 (correctly identifies useless model)
- Forces both metrics to be reasonable for good F1
- Cannot game by excelling at only one metric

**When to Use Each:**

| Scenario | Use | Reason |
|----------|-----|--------|
| Need both P and R to be good | Harmonic (F1) | Penalizes imbalance |
| P and R are independent targets | Arithmetic | Simple average |
| Multiplicative relationship | Geometric (G-mean) | See Q34 |
| Care more about one | F-beta | Adjustable weight |

**Python Code:**
```python
import numpy as np

def compare_means(precision, recall):
    """Compare different ways to combine P and R."""
    arithmetic = (precision + recall) / 2
    harmonic = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    geometric = np.sqrt(precision * recall)
    
    return {
        'arithmetic': arithmetic,
        'harmonic': harmonic,
        'geometric': geometric
    }

# Balanced case
balanced = compare_means(0.8, 0.8)
print("Balanced (P=0.8, R=0.8):")
print(f"  Arithmetic: {balanced['arithmetic']:.3f}")
print(f"  Harmonic (F1): {balanced['harmonic']:.3f}")
print(f"  Geometric: {balanced['geometric']:.3f}")

# Imbalanced case
imbalanced = compare_means(0.95, 0.10)
print("\nImbalanced (P=0.95, R=0.10):")
print(f"  Arithmetic: {imbalanced['arithmetic']:.3f}")  # 0.525 - misleading!
print(f"  Harmonic (F1): {imbalanced['harmonic']:.3f}")  # 0.18 - honest!
print(f"  Geometric: {imbalanced['geometric']:.3f}")
```

**Interview Tip:**
Explain that harmonic mean is used for F1 because we want BOTH precision and recall to be good. A model with perfect precision but zero recall is useless, and F1 correctly gives it a score of 0.

---

## Question 27
- [ ] Done
**How do you handle metric evaluation when ground truth labels have varying degrees of certainty?**

**Answer:**

**Definition:**
When labels are uncertain (annotator disagreement, probabilistic labels), standard metrics assuming binary truth become unreliable. Use **soft labels** (probability instead of 0/1), **annotator agreement metrics** (Fleiss' Kappa), or **expected metrics** weighted by label confidence.

**Scenarios with Uncertain Labels:**

| Scenario | Cause | Solution |
|----------|-------|----------|
| Annotator disagreement | Subjective task | Majority vote or soft labels |
| Noisy labels | Labeling errors | Label smoothing, confidence weighting |
| Probabilistic labels | Inherent uncertainty | Expected loss/metrics |

**Approaches:**

**1. Majority Voting:**
- Multiple annotators → Use majority label
- Report inter-annotator agreement (Kappa)

**2. Soft Label Evaluation:**
- Label = probability (e.g., 0.7 instead of 1)
- Use cross-entropy or Brier score

**3. Confidence-Weighted Metrics:**
- Weight each sample by label confidence
- High confidence samples contribute more

**Python Code:**
```python
import numpy as np
from sklearn.metrics import f1_score

# Scenario 1: Multiple annotators - use majority vote
def majority_vote_labels(annotations):
    """Convert multi-annotator labels to majority vote."""
    return [1 if sum(ann) > len(ann)/2 else 0 for ann in zip(*annotations)]

annotator_1 = [0, 1, 1, 0, 1, 1, 0, 0, 1, 1]
annotator_2 = [0, 1, 0, 0, 1, 1, 0, 1, 1, 1]
annotator_3 = [1, 1, 1, 0, 0, 1, 0, 0, 1, 0]

y_true = majority_vote_labels([annotator_1, annotator_2, annotator_3])
print(f"Majority labels: {y_true}")

# Scenario 2: Soft labels - use weighted metrics
soft_labels = [0.2, 0.9, 0.7, 0.1, 0.6, 0.95, 0.15, 0.3, 0.8, 0.85]
y_pred = [0, 1, 1, 0, 1, 1, 0, 0, 1, 1]
y_prob = [0.1, 0.85, 0.75, 0.2, 0.55, 0.9, 0.1, 0.25, 0.78, 0.8]

def brier_score(y_soft, y_prob):
    """Brier score for soft labels."""
    return np.mean((np.array(y_soft) - np.array(y_prob)) ** 2)

brier = brier_score(soft_labels, y_prob)
print(f"Brier Score: {brier:.4f}")  # Lower is better

# Scenario 3: Confidence-weighted F1
def confidence_weighted_accuracy(y_true, y_pred, confidence):
    """Weight samples by confidence in labels."""
    correct = [c if t == p else 0 for t, p, c in zip(y_true, y_pred, confidence)]
    return sum(correct) / sum(confidence)

confidence = [0.9, 0.95, 0.7, 0.85, 0.6, 0.99, 0.8, 0.5, 0.9, 0.85]
hard_labels = [0, 1, 1, 0, 1, 1, 0, 0, 1, 1]

weighted_acc = confidence_weighted_accuracy(hard_labels, y_pred, confidence)
print(f"Confidence-Weighted Accuracy: {weighted_acc:.3f}")
```

**Inter-Annotator Agreement:**
- Fleiss' Kappa > 0.8: Excellent agreement, trust labels
- Kappa 0.6-0.8: Moderate, consider soft labels
- Kappa < 0.6: Poor, question task definition

**Interview Tip:**
If interviewer mentions "noisy labels" or "annotator disagreement," propose soft label evaluation or confidence weighting instead of treating all labels as certain.

---

## Question 28
- [ ] Done
**What strategies help you optimize metrics during hyperparameter tuning without overfitting to validation data?**

**Answer:**

**Definition:**
Repeated tuning on validation set causes **validation overfitting** — model appears good on validation but fails on true test data. Mitigate using **nested cross-validation**, **holdout test set never touched during tuning**, and **early stopping on validation trends**.

**Problem:**
- Tune hyperparameters using validation F1
- After 100 experiments, validation F1 = 0.92
- Test F1 = 0.84 → Overfitted to validation!

**Prevention Strategies:**

| Strategy | Description |
|----------|-------------|
| **Nested CV** | Inner CV for tuning, outer CV for evaluation |
| **Holdout Test Set** | Never touch until final evaluation |
| **Limit Experiments** | Fewer tuning iterations = less overfitting |
| **Regularization** | Don't chase marginal gains |
| **Cross-Validation** | Average across folds reduces variance |

**Nested Cross-Validation:**
```
Outer Loop (5 folds): Evaluate model performance
  └── Inner Loop (5 folds): Tune hyperparameters
```

**Python Code:**
```python
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

np.random.seed(42)
X = np.random.randn(200, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# BAD: Single validation set, many experiments
# (validation score inflated after many trials)

# GOOD: Nested Cross-Validation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Hyperparameter search
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 10]
}

nested_scores = []
for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Inner CV for hyperparameter tuning
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=inner_cv,
        scoring='f1'
    )
    grid_search.fit(X_train, y_train)
    
    # Evaluate on outer test fold (unbiased)
    best_model = grid_search.best_estimator_
    from sklearn.metrics import f1_score
    y_pred = best_model.predict(X_test)
    score = f1_score(y_test, y_pred)
    nested_scores.append(score)

print(f"Nested CV F1: {np.mean(nested_scores):.3f} ± {np.std(nested_scores):.3f}")
print("This is an unbiased estimate of generalization performance")
```

**Rules of Thumb:**
- Keep true test set locked until final evaluation
- If validation score >> test score, you've overfitted
- Log all experiments to track how many times you've tuned

**Interview Tip:**
Mention nested CV to show you understand the difference between hyperparameter selection and performance estimation. This is a common source of ML mistakes.

---

## Question 29
- [ ] Done
**How do you implement metric evaluation for active learning scenarios with continuously updated training data?**

**Answer:**

**Definition:**
In active learning, the model iteratively selects informative samples to label, expanding training data over time. Evaluate using **learning curves** (metric vs labeled samples), **sample efficiency** (how fast metrics improve), and **holdout test set** fixed throughout iterations.

**Active Learning Evaluation Challenges:**

| Challenge | Solution |
|-----------|----------|
| Training set grows | Track metrics over iterations |
| Selection bias | Fixed holdout test set |
| Comparing strategies | Learning curve comparison |
| Stopping criteria | Metric plateau detection |

**Evaluation Framework:**

1. **Fixed Test Set:** Never changes, never used for selection
2. **Learning Curve:** Plot F1 vs number of labeled samples
3. **Area Under Learning Curve (AULC):** Higher = more sample-efficient
4. **Annotation Budget:** Performance at fixed budget

**Python Code:**
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Simulated pool-based active learning evaluation
np.random.seed(42)
X_pool = np.random.randn(500, 5)
y_pool = (X_pool[:, 0] + X_pool[:, 1] > 0).astype(int)

# Fixed test set (never touched during AL)
X_test = np.random.randn(100, 5)
y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

def active_learning_evaluation(X_pool, y_pool, X_test, y_test, 
                                budget=200, batch_size=20):
    """Evaluate active learning with learning curve."""
    n_pool = len(X_pool)
    labeled_idx = list(np.random.choice(n_pool, 10, replace=False))  # Initial seed
    unlabeled_idx = list(set(range(n_pool)) - set(labeled_idx))
    
    learning_curve = []
    
    while len(labeled_idx) < budget and unlabeled_idx:
        # Train on current labeled set
        X_train = X_pool[labeled_idx]
        y_train = y_pool[labeled_idx]
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Evaluate on fixed test set
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        learning_curve.append((len(labeled_idx), f1))
        
        # Select next batch (uncertainty sampling: closest to 0.5)
        if unlabeled_idx:
            X_unlabeled = X_pool[unlabeled_idx]
            probs = model.predict_proba(X_unlabeled)[:, 1]
            uncertainty = np.abs(probs - 0.5)
            
            # Select most uncertain samples
            n_select = min(batch_size, len(unlabeled_idx))
            selected_local = np.argsort(uncertainty)[:n_select]
            selected_global = [unlabeled_idx[i] for i in selected_local]
            
            labeled_idx.extend(selected_global)
            unlabeled_idx = [i for i in unlabeled_idx if i not in selected_global]
    
    return learning_curve

# Run and plot
curve = active_learning_evaluation(X_pool, y_pool, X_test, y_test)

print("Learning Curve (samples, F1):")
for samples, f1 in curve:
    print(f"  {samples:3d} samples: F1 = {f1:.3f}")

# Calculate Area Under Learning Curve (sample efficiency)
samples = [c[0] for c in curve]
f1s = [c[1] for c in curve]
aulc = np.trapz(f1s, samples) / max(samples)
print(f"\nArea Under Learning Curve: {aulc:.3f}")
```

**Comparison Metrics:**
- **AULC:** Higher = better sample efficiency
- **Samples to reach F1=0.8:** Fewer = better
- **Final F1 at budget:** Higher = better

**Interview Tip:**
Emphasize that the test set must be fixed and never used for sample selection. Active learning evaluation requires tracking metrics over time, not just final performance.

---

## Question 30
- [ ] Done
**When should you use top-k accuracy versus standard accuracy in multi-class classification problems?**

**Answer:**

**Definition:**
**Top-K Accuracy** considers a prediction correct if the true label is among the model's top K predictions. Use when: (1) classes are many and similar, (2) humans will review top suggestions, or (3) standard accuracy is too strict. Common in image classification and recommendation systems.

**Comparison:**

| Metric | Definition | Use Case |
|--------|------------|----------|
| **Accuracy** | Top-1 must match | Single definitive answer needed |
| **Top-5 Accuracy** | True label in top 5 | Many similar classes (ImageNet) |
| **Top-K Accuracy** | True label in top K | User selects from suggestions |

**When to Use Top-K:**
- Image classification with 1000+ classes
- Autocomplete/suggestion systems
- Medical diagnosis with differential diagnosis
- When classes are easily confused
- Search result ranking

**Mathematical Formulation:**
$$\text{Top-K Accuracy} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[y_i \in \text{TopK}(\hat{y}_i)]$$

**Python Code:**
```python
import numpy as np

def top_k_accuracy(y_true, y_prob, k=5):
    """
    Calculate top-k accuracy.
    y_true: true labels (class indices)
    y_prob: predicted probabilities (N x num_classes)
    k: number of top predictions to consider
    """
    correct = 0
    for i, true_label in enumerate(y_true):
        top_k_preds = np.argsort(y_prob[i])[-k:]  # Top k class indices
        if true_label in top_k_preds:
            correct += 1
    return correct / len(y_true)

# Example: 10-class classification
np.random.seed(42)
n_samples = 100
n_classes = 10

y_true = np.random.randint(0, n_classes, n_samples)
y_prob = np.random.random((n_samples, n_classes))

# Make predictions somewhat aligned with true labels
for i in range(n_samples):
    y_prob[i, y_true[i]] += 0.3  # Boost true class

# Normalize to probabilities
y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

# Calculate different accuracies
top_1 = top_k_accuracy(y_true, y_prob, k=1)
top_3 = top_k_accuracy(y_true, y_prob, k=3)
top_5 = top_k_accuracy(y_true, y_prob, k=5)

print(f"Top-1 Accuracy: {top_1:.3f}")
print(f"Top-3 Accuracy: {top_3:.3f}")
print(f"Top-5 Accuracy: {top_5:.3f}")

# Using sklearn (if available)
# from sklearn.metrics import top_k_accuracy_score
# top_5_sklearn = top_k_accuracy_score(y_true, y_prob, k=5)
```

**Typical Values (ImageNet):**
- Top-1: ~75-80%
- Top-5: ~93-95%

**Interview Tip:**
If asked about ImageNet or large-scale classification, mention Top-5 accuracy as the standard benchmark. It's more forgiving for fine-grained classes (e.g., dog breeds).

---

## Question 31
- [ ] Done
**How do you handle metric calculation for multi-label problems where partial matches should be credited?**

**Answer:**

**Definition:**
In multi-label classification, each sample can have multiple correct labels. Standard metrics treat partial matches as failures. Use **Hamming Score** (fraction of labels correct), **Jaccard Index** (intersection over union), or **subset accuracy relaxations** to give partial credit.

**Multi-Label Metrics:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Subset Accuracy** | Exact match | Strict (all or nothing) |
| **Hamming Score** | 1 - Hamming Loss | Per-label accuracy |
| **Jaccard Index** | \|P ∩ T\| / \|P ∪ T\| | Overlap ratio |
| **Example-Based F1** | Per-sample F1 averaged | Partial credit |

**Mathematical Formulation:**

For sample $i$ with true labels $T_i$ and predicted labels $P_i$:

$$\text{Hamming Score} = \frac{1}{n \cdot L}\sum_{i=1}^{n}\sum_{l=1}^{L} \mathbb{1}[y_{il} = \hat{y}_{il}]$$

$$\text{Jaccard} = \frac{1}{n}\sum_{i=1}^{n}\frac{|P_i \cap T_i|}{|P_i \cup T_i|}$$

**Example:**
- True: [1, 1, 0, 1] (labels 0, 1, 3 are positive)
- Pred: [1, 0, 0, 1] (labels 0, 3 are positive)
- Subset Accuracy: 0 (not exact match)
- Hamming Score: 3/4 = 0.75 (3 correct out of 4)
- Jaccard: 2/3 = 0.67 (intersection=2, union=3)

**Python Code:**
```python
import numpy as np
from sklearn.metrics import hamming_loss, jaccard_score

def multilabel_metrics(y_true, y_pred):
    """Calculate various multi-label metrics with partial credit."""
    n_samples = len(y_true)
    
    # Subset accuracy (exact match)
    subset_acc = sum(np.array_equal(t, p) for t, p in zip(y_true, y_pred)) / n_samples
    
    # Hamming score (1 - hamming loss)
    hamming_score = 1 - hamming_loss(y_true, y_pred)
    
    # Jaccard index (per-sample, then average)
    jaccard_scores = []
    for t, p in zip(y_true, y_pred):
        t, p = set(np.where(t)[0]), set(np.where(p)[0])
        if len(t | p) == 0:
            jaccard_scores.append(1.0)  # Both empty = perfect match
        else:
            jaccard_scores.append(len(t & p) / len(t | p))
    jaccard_avg = np.mean(jaccard_scores)
    
    # Example-based F1 (per-sample F1, then average)
    f1_scores = []
    for t, p in zip(y_true, y_pred):
        t_pos, p_pos = sum(t), sum(p)
        intersection = sum(a and b for a, b in zip(t, p))
        
        if t_pos + p_pos == 0:
            f1_scores.append(1.0)
        else:
            precision = intersection / p_pos if p_pos > 0 else 0
            recall = intersection / t_pos if t_pos > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
    
    return {
        'subset_accuracy': subset_acc,
        'hamming_score': hamming_score,
        'jaccard': jaccard_avg,
        'example_f1': np.mean(f1_scores)
    }

# Example
y_true = np.array([
    [1, 1, 0, 1],
    [0, 1, 1, 0],
    [1, 0, 0, 0]
])
y_pred = np.array([
    [1, 0, 0, 1],  # Partial match
    [0, 1, 1, 0],  # Exact match
    [1, 1, 0, 0]   # Partial match
])

metrics = multilabel_metrics(y_true, y_pred)
for name, value in metrics.items():
    print(f"{name}: {value:.3f}")
```

**Interview Tip:**
For multi-label problems, subset accuracy is too strict. Propose Hamming score or Jaccard for partial credit. Shows understanding of the task nuances.

---

## Question 32
- [ ] Done
**What techniques help you detect and handle metric gaming or exploitation in production systems?**

**Answer:**

**Definition:**
**Metric gaming** occurs when optimizing for a proxy metric causes undesired behavior (Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure"). Detect by monitoring multiple metrics, comparing proxy vs true outcomes, and watching for sudden metric improvements without real gains.

**Common Gaming Patterns:**

| Pattern | Example | Detection |
|---------|---------|-----------|
| Threshold gaming | Predicting just above threshold | Check score distribution |
| Easy case focus | Model avoids hard cases | Stratified performance analysis |
| Short-term optimization | High immediate metrics, poor long-term | Track delayed outcomes |
| Adversarial users | Submitting easy test cases | Monitor data distribution |

**Detection Strategies:**

1. **Multiple Metrics Dashboard:**
   - If one metric improves but others degrade, investigate

2. **Proxy vs True Outcome:**
   - Compare prediction metric with actual business outcome
   - High CTR prediction ≠ high actual CTR

3. **Distribution Monitoring:**
   - Alert on unusual prediction distributions
   - Watch for clustering around thresholds

4. **Stratified Analysis:**
   - Performance by user segment, data source, time
   - Gaming often affects specific strata

**Python Code:**
```python
import numpy as np
from collections import Counter

def detect_threshold_gaming(y_prob, threshold=0.5, window=0.05):
    """Detect if predictions cluster suspiciously around threshold."""
    near_threshold = sum(1 for p in y_prob if abs(p - threshold) < window)
    gaming_ratio = near_threshold / len(y_prob)
    
    # Normal: ~10% near threshold; Gaming: >30%
    is_suspicious = gaming_ratio > 0.3
    return {
        'near_threshold_ratio': gaming_ratio,
        'suspicious': is_suspicious
    }

def detect_metric_divergence(metric_a_values, metric_b_values, correlation_threshold=0.5):
    """Detect if two metrics that should correlate are diverging."""
    correlation = np.corrcoef(metric_a_values, metric_b_values)[0, 1]
    diverging = correlation < correlation_threshold
    return {
        'correlation': correlation,
        'diverging': diverging
    }

def detect_distribution_shift(baseline_preds, current_preds, threshold=0.1):
    """Detect if prediction distribution shifted unexpectedly."""
    baseline_mean = np.mean(baseline_preds)
    current_mean = np.mean(current_preds)
    shift = abs(current_mean - baseline_mean)
    
    return {
        'baseline_mean': baseline_mean,
        'current_mean': current_mean,
        'shift': shift,
        'alert': shift > threshold
    }

# Example: Normal predictions vs gaming
normal_probs = np.random.beta(2, 5, 1000)  # Natural distribution
gaming_probs = np.clip(np.random.normal(0.5, 0.03, 1000), 0.48, 0.52)  # Clustered!

print("Normal predictions:", detect_threshold_gaming(normal_probs))
print("Gaming predictions:", detect_threshold_gaming(gaming_probs))
```

**Mitigation Strategies:**
- Use multiple complementary metrics
- Track long-term business outcomes, not just proxies
- Randomize threshold or evaluation criteria
- Audit high-stakes predictions manually
- Regular A/B testing against control

**Interview Tip:**
Cite Goodhart's Law to show awareness of gaming risks. Propose multi-metric dashboards and delayed outcome tracking as solutions.

---

## Question 33
- [ ] Done
**How do you implement metric evaluation for zero-shot or few-shot classification scenarios?**

**Answer:**

**Definition:**
In zero-shot/few-shot learning, models classify classes never (zero) or rarely (few) seen during training. Standard metrics apply, but evaluation must account for: (1) per-class performance breakdown, (2) seen vs unseen class splits, and (3) generalization gap between training and novel classes.

**Evaluation Challenges:**

| Challenge | Solution |
|-----------|----------|
| Small sample size per class | Bootstrap confidence intervals |
| Class imbalance | Per-class and macro metrics |
| Novel vs seen classes | Report separately |
| High variance | Multiple random few-shot samples |

**Evaluation Protocol:**

**For Zero-Shot:**
- Evaluate only on unseen classes
- No training examples of test classes exist

**For Few-Shot (K-shot, N-way):**
- K examples per class, N classes
- Report: Accuracy/F1 averaged over multiple episodes
- Episode = random sample of K examples

**Key Metrics:**
- Mean accuracy across episodes (with std)
- Seen class accuracy vs unseen class accuracy
- Generalization gap = Seen performance - Unseen performance

**Python Code:**
```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def few_shot_evaluation(model, X_support, y_support, X_query, y_query, 
                        n_episodes=100, n_way=5, k_shot=5):
    """
    Evaluate few-shot model over multiple episodes.
    Each episode: randomly sample support set, evaluate on query set.
    """
    episode_accuracies = []
    episode_f1s = []
    
    available_classes = np.unique(y_support)
    
    for _ in range(n_episodes):
        # Sample N classes
        if len(available_classes) >= n_way:
            selected_classes = np.random.choice(available_classes, n_way, replace=False)
        else:
            selected_classes = available_classes
        
        # Sample K examples per class (support set)
        support_idx = []
        for cls in selected_classes:
            cls_idx = np.where(y_support == cls)[0]
            if len(cls_idx) >= k_shot:
                selected = np.random.choice(cls_idx, k_shot, replace=False)
            else:
                selected = cls_idx  # Use all available if < k
            support_idx.extend(selected)
        
        # Query set (remaining samples from selected classes)
        query_idx = [i for i, y in enumerate(y_query) if y in selected_classes]
        
        if len(query_idx) == 0:
            continue
        
        # Simulate model prediction (replace with actual model)
        # For demo: random predictions
        y_pred = np.random.choice(selected_classes, len(query_idx))
        y_true = y_query[query_idx]
        
        episode_accuracies.append(accuracy_score(y_true, y_pred))
        episode_f1s.append(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    return {
        'accuracy_mean': np.mean(episode_accuracies),
        'accuracy_std': np.std(episode_accuracies),
        'f1_mean': np.mean(episode_f1s),
        'f1_std': np.std(episode_f1s),
        'n_episodes': len(episode_accuracies)
    }

# Example usage
np.random.seed(42)
n_classes = 10
X_support = np.random.randn(100, 64)  # 100 support examples
y_support = np.repeat(np.arange(n_classes), 10)  # 10 per class
X_query = np.random.randn(50, 64)
y_query = np.random.randint(0, n_classes, 50)

results = few_shot_evaluation(None, X_support, y_support, X_query, y_query,
                               n_episodes=50, n_way=5, k_shot=3)

print(f"5-way 3-shot Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}")
print(f"5-way 3-shot Macro-F1: {results['f1_mean']:.3f} ± {results['f1_std']:.3f}")
```

**Reporting Best Practices:**
- Always report mean ± std over multiple episodes
- Specify: N-way, K-shot clearly
- Report seen vs unseen class performance separately

**Interview Tip:**
For few-shot learning, emphasize episodic evaluation with multiple random samplings. Single-run metrics are unreliable due to high variance.

---

## Question 34
- [ ] Done
**When would you use geometric mean versus F1-score for combining precision and recall in specific domains?**

**Answer:**

**Definition:**
Use **Geometric Mean (G-Mean)** when you care about balanced performance on both classes independently, especially with class imbalance. G-Mean = √(TPR × TNR). Unlike F1 which focuses on positive class, G-Mean equally weights positive and negative class performance.

**Comparison:**

| Metric | Formula | Focus |
|--------|---------|-------|
| **F1-Score** | 2PR / (P+R) | Positive class only |
| **G-Mean** | √(TPR × TNR) | Both classes equally |
| **Balanced Accuracy** | (TPR + TNR) / 2 | Both classes (arithmetic) |

**Mathematical Formulation:**
$$\text{G-Mean} = \sqrt{TPR \times TNR} = \sqrt{\frac{TP}{TP+FN} \times \frac{TN}{TN+FP}}$$

**Key Properties:**
- G-Mean = 0 if either TPR or TNR is 0
- Maximized when TPR = TNR (balanced performance)
- More sensitive to class imbalance than F1

**When to Use G-Mean:**
- Both classes equally important (e.g., balanced medical diagnosis)
- Imbalanced data where negative class performance matters
- Credit scoring (both accept and reject decisions matter)

**When to Use F1:**
- Only positive class matters (fraud detection)
- Negative class is trivial or less important
- Standard benchmark comparisons

**Python Code:**
```python
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

def g_mean_score(y_true, y_pred):
    """Calculate Geometric Mean (G-Mean) score."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/Sensitivity
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    
    g_mean = np.sqrt(tpr * tnr)
    return g_mean, tpr, tnr

# Example: Imbalanced classification
y_true = [0]*90 + [1]*10  # 90% negative, 10% positive

# Model A: Good on positive, ignores negative
y_pred_a = [1]*20 + [0]*80  # Predicts positive for 20 samples

# Model B: Balanced performance
y_pred_b = [0]*85 + [1]*5 + [0]*5 + [1]*5  # Balanced approach

print("Model A (positive-focused):")
f1_a = f1_score(y_true, y_pred_a)
gm_a, tpr_a, tnr_a = g_mean_score(y_true, y_pred_a)
print(f"  F1: {f1_a:.3f}, G-Mean: {gm_a:.3f} (TPR={tpr_a:.2f}, TNR={tnr_a:.2f})")

print("\nModel B (balanced):")
f1_b = f1_score(y_true, y_pred_b)
gm_b, tpr_b, tnr_b = g_mean_score(y_true, y_pred_b)
print(f"  F1: {f1_b:.3f}, G-Mean: {gm_b:.3f} (TPR={tpr_b:.2f}, TNR={tnr_b:.2f})")

# F1 might prefer A, but G-Mean prefers B
```

**Domain Examples:**
- **Medical screening:** G-Mean (need to catch disease AND avoid unnecessary tests)
- **Fraud detection:** F1 (focus on catching fraud, false alarms acceptable)
- **Credit approval:** G-Mean (bad approvals AND missed good customers both costly)

**Interview Tip:**
If interviewer mentions "both classes matter equally" or discusses true negative importance, propose G-Mean over F1.

---

## Question 35
- [ ] Done
**How do you handle metric evaluation for classification problems with missing or incomplete labels?**

**Answer:**

**Definition:**
When labels are partially missing (e.g., multi-label with unknown labels, or unlabeled test samples), standard metrics become undefined. Handle by: (1) evaluating only on labeled subset, (2) treating missing as separate class, or (3) using semi-supervised evaluation metrics.

**Scenarios:**

| Scenario | Handling |
|----------|----------|
| Some samples fully unlabeled | Evaluate on labeled subset only |
| Multi-label: some labels unknown | Mask unknown labels, evaluate known only |
| Partial supervision | Report on labeled portion |
| Missing at random | Simple exclusion |
| Missing not at random | Bias correction needed |

**Strategies:**

**1. Exclude Unknown Labels:**
- Only compute metrics on known label-prediction pairs
- Report coverage: "Evaluated on X% of data"

**2. Multi-Label with Unknown:**
- Missing label ≠ negative
- Mask and exclude from TP/FP/FN counts

**3. Semi-Supervised Metrics:**
- Cluster purity for unlabeled data
- Consistency metrics

**Python Code:**
```python
import numpy as np
from sklearn.metrics import f1_score

def metrics_with_missing(y_true, y_pred, missing_value=-1):
    """
    Calculate metrics ignoring missing labels.
    missing_value: Indicator for unknown labels (e.g., -1 or np.nan)
    """
    # Filter to known labels only
    mask = np.array(y_true) != missing_value
    y_true_known = np.array(y_true)[mask]
    y_pred_known = np.array(y_pred)[mask]
    
    coverage = mask.sum() / len(y_true)
    
    if len(y_true_known) == 0:
        return {'f1': None, 'coverage': 0}
    
    f1 = f1_score(y_true_known, y_pred_known, average='binary', zero_division=0)
    
    return {
        'f1': f1,
        'coverage': coverage,
        'n_evaluated': len(y_true_known),
        'n_missing': len(y_true) - len(y_true_known)
    }

# Example: Some labels are unknown (-1)
y_true = [0, 1, -1, 1, 0, -1, 1, 0, 1, -1]  # -1 = unknown
y_pred = [0, 1, 1, 0, 0, 1, 1, 0, 1, 0]

result = metrics_with_missing(y_true, y_pred, missing_value=-1)
print(f"F1 Score: {result['f1']:.3f}")
print(f"Coverage: {result['coverage']:.1%}")
print(f"Evaluated: {result['n_evaluated']}, Missing: {result['n_missing']}")

# Multi-label with missing (mask approach)
def multilabel_f1_with_missing(y_true, y_pred, missing=-1):
    """
    Multi-label F1 ignoring unknown labels per sample.
    y_true: list of lists, -1 indicates unknown
    """
    sample_f1s = []
    
    for true_labels, pred_labels in zip(y_true, y_pred):
        # Get indices where label is known
        known_idx = [i for i, t in enumerate(true_labels) if t != missing]
        
        if not known_idx:
            continue
        
        true_known = [true_labels[i] for i in known_idx]
        pred_known = [pred_labels[i] for i in known_idx]
        
        # Per-sample F1 on known labels
        tp = sum(t == 1 and p == 1 for t, p in zip(true_known, pred_known))
        pred_pos = sum(pred_known)
        true_pos = sum(true_known)
        
        precision = tp / pred_pos if pred_pos > 0 else 0
        recall = tp / true_pos if true_pos > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        sample_f1s.append(f1)
    
    return np.mean(sample_f1s) if sample_f1s else 0

# Example
y_true_ml = [[1, 0, -1], [0, 1, 1], [-1, -1, 1]]
y_pred_ml = [[1, 1, 0], [0, 1, 0], [1, 0, 1]]

ml_f1 = multilabel_f1_with_missing(y_true_ml, y_pred_ml)
print(f"Multi-label F1 (known only): {ml_f1:.3f}")
```

**Interview Tip:**
Always report coverage when dealing with missing labels. "F1 = 0.85 evaluated on 70% of test set" is more honest than just "F1 = 0.85".

---

## Question 36
- [ ] Done
**What are the considerations for implementing custom metrics that align with specific business objectives?**

**Answer:**

**Definition:**
Standard metrics (F1, accuracy) may not capture true business value. Custom metrics directly encode business logic: costs, revenues, user satisfaction. Key considerations: define the objective clearly, ensure metric is differentiable if used for training, validate correlation with actual business outcomes.

**Design Considerations:**

| Consideration | Description |
|--------------|-------------|
| **Business alignment** | Does improving metric improve business outcomes? |
| **Measurability** | Can we compute it reliably? |
| **Sensitivity** | Does it detect meaningful model changes? |
| **Gaming resistance** | Can it be exploited? |
| **Interpretability** | Can stakeholders understand it? |

**Custom Metric Examples:**

| Business Goal | Custom Metric |
|--------------|---------------|
| Minimize financial loss | Cost = FP×$50 + FN×$500 |
| User engagement | (1-Bounce Rate) × CTR |
| Customer satisfaction | Correct×1 + PartialCorrect×0.5 |
| Time-sensitive detection | Recall × (1/Detection_Latency) |

**Implementation Steps:**
1. Define business objective in measurable terms
2. Map predictions to business impact (cost matrix)
3. Implement metric function
4. Validate correlation with actual business KPIs
5. Monitor for gaming or unintended optimization

**Python Code:**
```python
import numpy as np
from sklearn.metrics import confusion_matrix

def create_cost_metric(cost_fp, cost_fn, cost_tp=0, cost_tn=0):
    """Factory function to create custom cost-based metric."""
    def cost_metric(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = tp*cost_tp + tn*cost_tn + fp*cost_fp + fn*cost_fn
        return total_cost
    return cost_metric

def create_revenue_metric(revenue_tp, cost_fp, cost_fn):
    """Metric that maximizes net revenue."""
    def revenue_metric(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        net_revenue = tp*revenue_tp - fp*cost_fp - fn*cost_fn
        return net_revenue
    return revenue_metric

# Example: Fraud detection
# - Catching fraud (TP) saves $1000
# - False alarm (FP) costs $50 investigation
# - Missed fraud (FN) costs $1000 loss
fraud_revenue = create_revenue_metric(revenue_tp=1000, cost_fp=50, cost_fn=1000)

y_true = [0, 0, 1, 1, 1, 0, 1, 0, 0, 1]
y_pred_conservative = [0, 0, 0, 1, 1, 0, 0, 0, 0, 1]  # High precision
y_pred_aggressive = [0, 1, 1, 1, 1, 1, 1, 0, 0, 1]    # High recall

rev_conservative = fraud_revenue(y_true, y_pred_conservative)
rev_aggressive = fraud_revenue(y_true, y_pred_aggressive)

print(f"Conservative model revenue: ${rev_conservative}")
print(f"Aggressive model revenue: ${rev_aggressive}")

# Custom metric for recommendation (partial credit)
def recommendation_score(y_true, y_pred, y_scores):
    """
    Custom recommendation metric:
    - Full credit for click on top recommendation
    - Partial credit for click in top 3
    """
    scores = []
    for true, pred, rank_scores in zip(y_true, y_pred, y_scores):
        top_3 = np.argsort(rank_scores)[-3:][::-1]
        if true == top_3[0]:
            scores.append(1.0)  # Top pick
        elif true in top_3:
            scores.append(0.5)  # In top 3
        else:
            scores.append(0.0)  # Miss
    return np.mean(scores)

# Usage: Optimize threshold/model to maximize custom business metric
```

**Validation:**
- Track custom metric alongside actual business KPI
- High correlation → metric is a good proxy
- Low correlation → revise metric definition

**Interview Tip:**
Propose custom metrics when standard metrics don't capture business value. Show you think beyond technical metrics to business impact.

---

## Question 37
- [ ] Done
**How do you optimize model performance when different metrics conflict with each other?**

**Answer:**

**Definition:**
Metrics often conflict (e.g., Precision vs Recall, Accuracy vs Latency). Handle by: (1) defining a **primary metric** for selection, (2) setting **constraints** on secondary metrics, (3) using **Pareto optimization** to find the best trade-off frontier, or (4) creating a **weighted composite score**.

**Common Conflicts:**

| Conflict | Trade-off |
|----------|-----------|
| Precision vs Recall | Threshold adjustment |
| Accuracy vs Fairness | Fairness constraints |
| Performance vs Latency | Model complexity |
| F1 vs Interpretability | Model selection |

**Resolution Strategies:**

**1. Primary + Constraints:**
- Maximize primary metric (e.g., Recall)
- Subject to constraints (e.g., Precision ≥ 0.7)

**2. Weighted Composite:**
$$Score = w_1 \cdot M_1 + w_2 \cdot M_2$$
- Weights reflect relative importance

**3. Pareto Frontier:**
- Find models where no metric can improve without degrading another
- Present options to stakeholders

**4. Satisficing:**
- Find model that meets minimum thresholds for all metrics
- No need for optimality, just acceptability

**Python Code:**
```python
import numpy as np

# Multiple models with different trade-offs
models = {
    'Model_A': {'precision': 0.90, 'recall': 0.65, 'latency_ms': 10},
    'Model_B': {'precision': 0.80, 'recall': 0.80, 'latency_ms': 25},
    'Model_C': {'precision': 0.70, 'recall': 0.90, 'latency_ms': 50},
    'Model_D': {'precision': 0.75, 'recall': 0.75, 'latency_ms': 15},
}

# Strategy 1: Primary metric with constraints
def select_with_constraints(models, primary='recall', constraints=None):
    """Select best model on primary metric meeting all constraints."""
    constraints = constraints or {}
    
    valid_models = []
    for name, metrics in models.items():
        meets_constraints = all(metrics[k] >= v for k, v in constraints.items() if k in metrics)
        if meets_constraints:
            valid_models.append((name, metrics[primary]))
    
    if not valid_models:
        return None, "No model meets constraints"
    
    best = max(valid_models, key=lambda x: x[1])
    return best[0], f"{primary}={best[1]:.2f}"

# Maximize recall with precision >= 0.75
selected, reason = select_with_constraints(
    models, 
    primary='recall', 
    constraints={'precision': 0.75}
)
print(f"Strategy 1 - Constrained optimization: {selected} ({reason})")

# Strategy 2: Weighted composite
def weighted_selection(models, weights):
    """Select model with highest weighted score."""
    scores = {}
    for name, metrics in models.items():
        score = sum(weights.get(k, 0) * v for k, v in metrics.items() if k in weights)
        scores[name] = score
    
    best = max(scores, key=scores.get)
    return best, scores[best]

weights = {'precision': 0.3, 'recall': 0.5, 'latency_ms': -0.005}  # Negative for latency
selected, score = weighted_selection(models, weights)
print(f"Strategy 2 - Weighted composite: {selected} (score={score:.3f})")

# Strategy 3: Find Pareto frontier
def pareto_frontier(models, metrics_to_max):
    """Find models on Pareto frontier."""
    pareto = []
    model_list = list(models.items())
    
    for i, (name_i, m_i) in enumerate(model_list):
        dominated = False
        for j, (name_j, m_j) in enumerate(model_list):
            if i == j:
                continue
            # Check if j dominates i
            better_or_equal = all(m_j[k] >= m_i[k] for k in metrics_to_max)
            strictly_better = any(m_j[k] > m_i[k] for k in metrics_to_max)
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(name_i)
    
    return pareto

pareto_models = pareto_frontier(models, ['precision', 'recall'])
print(f"Strategy 3 - Pareto frontier: {pareto_models}")
```

**Decision Framework:**
1. Ask stakeholders: What's the primary objective?
2. Define acceptable ranges for all metrics
3. Use constrained optimization or Pareto analysis
4. Present options if trade-offs are significant

**Interview Tip:**
Never optimize blindly for one metric. Show awareness of trade-offs and propose structured decision frameworks.

---

## Question 38
- [ ] Done
**When should you use micro-averaging versus macro-averaging for different types of classification problems?**

**Answer:**

**Definition:**
**Micro-averaging** aggregates all TP, FP, FN globally before computing metrics — gives more weight to frequent classes. **Macro-averaging** computes metrics per class then averages — treats all classes equally. Use micro for overall accuracy, macro when minority classes matter.

**Comparison:**

| Aspect | Micro-Average | Macro-Average |
|--------|--------------|---------------|
| Calculation | Global TP/FP/FN | Per-class then mean |
| Class weighting | By frequency | Equal weight |
| Affected by | Large classes | All classes equally |
| Use when | Overall performance | Minority classes important |

**Mathematical Formulation:**

$$\text{Micro-Precision} = \frac{\sum_c TP_c}{\sum_c (TP_c + FP_c)}$$

$$\text{Macro-Precision} = \frac{1}{C}\sum_c \frac{TP_c}{TP_c + FP_c}$$

**When to Use:**

| Scenario | Use | Reason |
|----------|-----|--------|
| Imbalanced, minority important | Macro | Equal weight to rare classes |
| Balanced classes | Either | Results similar |
| Overall system performance | Micro | Reflects actual accuracy |
| Benchmark comparison | Check dataset | Use what's standard |
| Multi-label | Both | Different perspectives |

**Example:**
Classes: A (900 samples), B (100 samples)
- Micro: Dominated by Class A performance
- Macro: Equal weight to A and B

**Python Code:**
```python
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Imbalanced multi-class
y_true = [0]*90 + [1]*10 + [2]*10  # Class 0 dominant
y_pred = [0]*85 + [1]*5 + [1]*8 + [2]*2 + [2]*8 + [0]*2  # Mixed predictions

# Calculate both averaging methods
print("=== Precision ===")
micro_p = precision_score(y_true, y_pred, average='micro')
macro_p = precision_score(y_true, y_pred, average='macro')
print(f"Micro: {micro_p:.3f} (dominated by Class 0)")
print(f"Macro: {macro_p:.3f} (equal class weight)")

print("\n=== Recall ===")
micro_r = recall_score(y_true, y_pred, average='micro')
macro_r = recall_score(y_true, y_pred, average='macro')
print(f"Micro: {micro_r:.3f}")
print(f"Macro: {macro_r:.3f}")

print("\n=== F1-Score ===")
micro_f1 = f1_score(y_true, y_pred, average='micro')
macro_f1 = f1_score(y_true, y_pred, average='macro')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')
print(f"Micro: {micro_f1:.3f}")
print(f"Macro: {macro_f1:.3f}")
print(f"Weighted: {weighted_f1:.3f}")

# Per-class breakdown
print("\n=== Per-Class F1 ===")
per_class = f1_score(y_true, y_pred, average=None)
for i, f1 in enumerate(per_class):
    count = y_true.count(i) if isinstance(y_true, list) else (y_true == i).sum()
    print(f"Class {i}: F1={f1:.3f} (n={count})")
```

**Quick Decision Guide:**
- Micro ≈ Accuracy for multi-class (single-label)
- Macro: Use when you want to surface poor minority class performance
- Weighted: Compromise (by support)

**Interview Tip:**
Always clarify: "Are all classes equally important?" If yes → Macro. If majority class dominates business value → Micro.

---

## Question 39
- [ ] Done
**How do you implement metric evaluation for multi-task learning scenarios with shared representations?**

**Answer:**

**Definition:**
In multi-task learning (MTL), one model learns multiple tasks simultaneously with shared layers. Evaluate each task independently, then aggregate using task-weighted average. Consider task correlations and whether improvement in one task harms another.

**MTL Evaluation Challenges:**

| Challenge | Solution |
|-----------|----------|
| Different task scales | Normalize metrics to [0,1] |
| Different task importance | Weighted aggregation |
| Negative transfer | Track per-task vs baseline |
| Task interference | Compare with single-task models |

**Evaluation Framework:**

1. **Per-Task Metrics:** Compute F1/accuracy for each task
2. **Aggregated Score:** Weighted combination
3. **Transfer Analysis:** MTL vs single-task comparison
4. **Correlation:** Track if tasks help/hurt each other

**Python Code:**
```python
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def mtl_evaluation(task_predictions, task_labels, task_weights=None, 
                   single_task_baselines=None):
    """
    Evaluate multi-task learning model.
    
    task_predictions: dict {'task_name': y_pred}
    task_labels: dict {'task_name': y_true}
    task_weights: dict {'task_name': weight}
    single_task_baselines: dict {'task_name': baseline_f1} for comparison
    """
    task_names = list(task_predictions.keys())
    n_tasks = len(task_names)
    
    if task_weights is None:
        task_weights = {t: 1.0/n_tasks for t in task_names}
    
    results = {}
    per_task_metrics = {}
    
    for task in task_names:
        y_true = task_labels[task]
        y_pred = task_predictions[task]
        
        f1 = f1_score(y_true, y_pred, average='weighted')
        acc = accuracy_score(y_true, y_pred)
        
        per_task_metrics[task] = {'f1': f1, 'accuracy': acc}
        
        # Compare to single-task baseline if available
        if single_task_baselines and task in single_task_baselines:
            baseline = single_task_baselines[task]
            transfer = f1 - baseline
            per_task_metrics[task]['transfer'] = transfer
            per_task_metrics[task]['positive_transfer'] = transfer > 0
    
    results['per_task'] = per_task_metrics
    
    # Weighted aggregate
    weighted_f1 = sum(task_weights[t] * per_task_metrics[t]['f1'] for t in task_names)
    results['aggregate_f1'] = weighted_f1
    
    # Check for negative transfer (any task hurt)
    if single_task_baselines:
        negative_transfers = [t for t in task_names 
                            if per_task_metrics[t].get('transfer', 0) < -0.02]
        results['negative_transfer_tasks'] = negative_transfers
    
    return results

# Example: 3 tasks
task_predictions = {
    'sentiment': [1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
    'topic': [0, 1, 2, 1, 0, 2, 1, 0, 2, 1],
    'intent': [1, 1, 0, 1, 0, 0, 1, 1, 0, 1]
}

task_labels = {
    'sentiment': [1, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    'topic': [0, 1, 2, 1, 0, 1, 1, 0, 2, 0],
    'intent': [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
}

# Task importance weights
task_weights = {'sentiment': 0.4, 'topic': 0.3, 'intent': 0.3}

# Single-task baselines for transfer analysis
single_task_baselines = {'sentiment': 0.75, 'topic': 0.70, 'intent': 0.80}

results = mtl_evaluation(task_predictions, task_labels, 
                         task_weights, single_task_baselines)

print("=== Per-Task Results ===")
for task, metrics in results['per_task'].items():
    transfer = metrics.get('transfer', 0)
    print(f"{task}: F1={metrics['f1']:.3f}, Transfer={transfer:+.3f}")

print(f"\n=== Aggregate ===")
print(f"Weighted F1: {results['aggregate_f1']:.3f}")

if results.get('negative_transfer_tasks'):
    print(f"Warning: Negative transfer on: {results['negative_transfer_tasks']}")
```

**Key Considerations:**
- Report both aggregate and per-task metrics
- Compare to single-task baselines to measure transfer
- Investigate tasks with negative transfer
- Consider task correlations in weight selection

**Interview Tip:**
For MTL, always mention checking for negative transfer. Sometimes MTL hurts minority tasks while boosting majority ones.

---

## Question 40
- [ ] Done
**What strategies help you maintain metric reliability when transitioning from development to production?**

**Answer:**

**Definition:**
Metrics computed during development often don't match production reality due to: data drift, different distributions, sampling bias, and missing labels in production. Maintain reliability by: establishing baselines, shadow testing, gradual rollout, and continuous monitoring.

**Dev vs Production Gaps:**

| Issue | Cause | Mitigation |
|-------|-------|------------|
| Data distribution shift | Production data differs | Monitor input distributions |
| Label delay | No immediate ground truth | Proxy metrics + delayed evaluation |
| Sampling bias | Test set not representative | Production A/B testing |
| Feedback loops | Model affects future data | Track long-term outcomes |
| Scale differences | Dev is small scale | Load test metrics |

**Strategies:**

**1. Shadow Mode Testing:**
- Run new model in parallel, log predictions
- Compare with existing model
- Evaluate when labels become available

**2. Gradual Rollout:**
- 1% → 10% → 50% → 100%
- Monitor metrics at each stage
- Rollback if degradation detected

**3. Proxy Metrics:**
- User engagement (clicks, dwell time)
- Prediction confidence distribution
- Error rate compared to baseline

**4. Baseline Establishment:**
- Document dev metrics with confidence intervals
- Set alert thresholds for production

**Python Code:**
```python
import numpy as np
from datetime import datetime

class ProductionMetricsMonitor:
    def __init__(self, dev_baseline, alert_threshold=0.05):
        """
        dev_baseline: {'metric_name': (mean, std)}
        alert_threshold: relative drop triggering alert
        """
        self.dev_baseline = dev_baseline
        self.alert_threshold = alert_threshold
        self.production_metrics = []
        self.alerts = []
    
    def log_batch(self, metrics, timestamp=None):
        """Log a batch of production metrics."""
        timestamp = timestamp or datetime.now()
        self.production_metrics.append({
            'timestamp': timestamp,
            'metrics': metrics
        })
        
        # Check for degradation
        self._check_alerts(metrics, timestamp)
    
    def _check_alerts(self, metrics, timestamp):
        for metric_name, value in metrics.items():
            if metric_name in self.dev_baseline:
                baseline_mean, baseline_std = self.dev_baseline[metric_name]
                relative_drop = (baseline_mean - value) / baseline_mean
                
                if relative_drop > self.alert_threshold:
                    self.alerts.append({
                        'timestamp': timestamp,
                        'metric': metric_name,
                        'value': value,
                        'baseline': baseline_mean,
                        'drop': relative_drop
                    })
    
    def get_summary(self):
        """Get production performance summary."""
        if not self.production_metrics:
            return None
        
        summary = {}
        metric_names = self.production_metrics[0]['metrics'].keys()
        
        for name in metric_names:
            values = [m['metrics'][name] for m in self.production_metrics]
            summary[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'n_batches': len(values)
            }
            
            if name in self.dev_baseline:
                summary[name]['dev_baseline'] = self.dev_baseline[name][0]
                summary[name]['gap'] = summary[name]['mean'] - self.dev_baseline[name][0]
        
        return summary

# Usage example
dev_baseline = {
    'f1': (0.85, 0.02),
    'precision': (0.88, 0.03),
    'recall': (0.82, 0.02)
}

monitor = ProductionMetricsMonitor(dev_baseline, alert_threshold=0.05)

# Simulate production batches
production_batches = [
    {'f1': 0.84, 'precision': 0.87, 'recall': 0.81},
    {'f1': 0.83, 'precision': 0.86, 'recall': 0.80},
    {'f1': 0.79, 'precision': 0.85, 'recall': 0.74},  # Degradation!
]

for batch in production_batches:
    monitor.log_batch(batch)

summary = monitor.get_summary()
print("=== Production Summary ===")
for metric, stats in summary.items():
    print(f"{metric}: {stats['mean']:.3f} (dev: {stats.get('dev_baseline', 'N/A')}, gap: {stats.get('gap', 'N/A'):.3f})")

print(f"\n=== Alerts ===")
for alert in monitor.alerts:
    print(f"⚠ {alert['metric']}: {alert['value']:.3f} (dropped {alert['drop']:.1%} from baseline)")
```

**Checklist for Production Transition:**
1. Document dev metrics with confidence intervals
2. Set up monitoring dashboards
3. Define alert thresholds
4. Establish rollback criteria
5. Plan for delayed ground truth

**Interview Tip:**
Mention that dev metrics are optimistic. Production has data drift, label delays, and sampling differences. Show awareness of MLOps concerns.

---

## Question 41
- [ ] Done
**How do you handle metric evaluation for federated learning scenarios with distributed data?**

**Answer:**

**Definition:**
In federated learning, data stays on local devices and cannot be centralized. Metrics must be computed locally and aggregated. Challenges include: non-IID data across clients, varying sample sizes, privacy constraints, and no central test set.

**Federated Evaluation Challenges:**

| Challenge | Solution |
|-----------|----------|
| Distributed data | Local evaluation + aggregation |
| Non-IID distribution | Per-client and global metrics |
| Privacy constraints | Secure aggregation protocols |
| Different sample sizes | Weighted aggregation |
| No central test set | Representative holdout per client |

**Aggregation Methods:**

**1. Weighted Average (by samples):**
$$\text{Global Metric} = \frac{\sum_k n_k \cdot m_k}{\sum_k n_k}$$

**2. Simple Average:**
$$\text{Global Metric} = \frac{1}{K}\sum_k m_k$$

**3. Median (robust to outliers):**
$$\text{Global Metric} = \text{median}(m_1, m_2, ..., m_K)$$

**Python Code:**
```python
import numpy as np
from sklearn.metrics import f1_score

def federated_evaluation(client_data, model_predict_fn, aggregation='weighted'):
    """
    Evaluate model across federated clients.
    
    client_data: list of (X, y) tuples for each client
    model_predict_fn: function to predict
    aggregation: 'weighted', 'mean', or 'median'
    """
    client_metrics = []
    client_sizes = []
    
    for client_id, (X, y) in enumerate(client_data):
        y_pred = model_predict_fn(X)
        f1 = f1_score(y, y_pred, average='binary', zero_division=0)
        
        client_metrics.append(f1)
        client_sizes.append(len(y))
        
        print(f"Client {client_id}: F1={f1:.3f}, n={len(y)}")
    
    # Aggregate
    if aggregation == 'weighted':
        global_metric = np.average(client_metrics, weights=client_sizes)
    elif aggregation == 'mean':
        global_metric = np.mean(client_metrics)
    elif aggregation == 'median':
        global_metric = np.median(client_metrics)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    # Fairness check: variance across clients
    metric_std = np.std(client_metrics)
    
    return {
        'global_metric': global_metric,
        'client_metrics': client_metrics,
        'client_variance': metric_std,
        'aggregation': aggregation,
        'fairness_flag': metric_std > 0.1  # High variance = unfair
    }

# Simulate federated clients (non-IID data)
np.random.seed(42)

def mock_predict(X):
    # Simulate model predictions
    return (X[:, 0] > 0).astype(int)

client_data = [
    # Client 0: Balanced
    (np.random.randn(100, 5), (np.random.randn(100) > 0).astype(int)),
    # Client 1: Imbalanced (mostly 0)
    (np.random.randn(50, 5), np.array([0]*45 + [1]*5)),
    # Client 2: Imbalanced (mostly 1)
    (np.random.randn(200, 5), np.array([0]*20 + [1]*180)),
]

results = federated_evaluation(client_data, mock_predict, aggregation='weighted')

print(f"\n=== Global Results ===")
print(f"Aggregated F1: {results['global_metric']:.3f}")
print(f"Client Variance: {results['client_variance']:.3f}")
if results['fairness_flag']:
    print("⚠ High variance across clients - check for fairness issues")
```

**Best Practices:**
- Report both global and per-client metrics
- Monitor variance across clients (fairness)
- Use weighted aggregation when sample sizes vary significantly
- Maintain local validation sets on each client

**Interview Tip:**
Emphasize that federated learning requires thinking about heterogeneous data. Non-IID clients can have wildly different metrics; global averages can be misleading.

---

## Question 42
- [ ] Done
**When would you implement time-weighted metrics for classification problems with temporal importance?**

**Answer:**

**Definition:**
Use time-weighted metrics when recent predictions matter more than old ones, or when outcomes have time-sensitive value. Weight samples by recency (exponential decay), time-to-event value, or business urgency. Common in fraud detection, churn prediction, and real-time systems.

**When to Use:**

| Scenario | Weighting Strategy |
|----------|-------------------|
| Recent performance matters | Exponential decay by age |
| Early detection valuable | Inverse of detection time |
| Seasonal patterns | Periodic weighting |
| Concept drift | Higher weight to recent |
| SLA-based systems | Time-to-resolution weight |

**Time-Weighting Approaches:**

**1. Recency Decay:**
$$w_i = e^{-\lambda (t_{now} - t_i)}$$

**2. Time-Value Weighting:**
$$w_i = \frac{1}{\text{time\_to\_detect}_i}$$

**3. Business Urgency:**
$$w_i = \text{urgency\_score}_i$$

**Python Code:**
```python
import numpy as np
from sklearn.metrics import f1_score
from datetime import datetime, timedelta

def time_weighted_f1(y_true, y_pred, timestamps, decay_rate=0.1, 
                     reference_time=None):
    """
    Calculate F1 with exponential time weighting.
    Recent samples weighted more heavily.
    
    decay_rate: Higher = faster decay of old samples
    """
    if reference_time is None:
        reference_time = max(timestamps)
    
    # Calculate age in days
    ages = [(reference_time - t).days for t in timestamps]
    
    # Exponential decay weights
    weights = np.exp(-decay_rate * np.array(ages))
    weights = weights / weights.sum()  # Normalize
    
    # Weighted confusion matrix
    tp = sum(w for w, t, p in zip(weights, y_true, y_pred) if t == 1 and p == 1)
    fp = sum(w for w, t, p in zip(weights, y_true, y_pred) if t == 0 and p == 1)
    fn = sum(w for w, t, p in zip(weights, y_true, y_pred) if t == 1 and p == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

def detection_time_weighted_recall(y_true, y_pred, detection_delays):
    """
    Weight recall by how quickly events were detected.
    Faster detection = higher weight.
    """
    true_positives = [(t == 1 and p == 1, delay) 
                      for t, p, delay in zip(y_true, y_pred, detection_delays)]
    
    total_positives = sum(y_true)
    if total_positives == 0:
        return 0
    
    # Weight by 1/delay (faster = better)
    weighted_recall = sum(1/delay if tp else 0 for tp, delay in true_positives if delay > 0)
    max_possible = sum(1/delay for t, delay in zip(y_true, detection_delays) if t == 1 and delay > 0)
    
    return weighted_recall / max_possible if max_possible > 0 else 0

# Example with timestamps
np.random.seed(42)
today = datetime.now()
timestamps = [today - timedelta(days=i) for i in range(30, 0, -1)]

y_true = [0, 1, 0, 1, 1, 0, 0, 1, 0, 0,
          1, 0, 1, 0, 0, 1, 1, 0, 0, 1,
          0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
y_pred = [0, 1, 0, 0, 1, 0, 1, 1, 0, 0,
          1, 0, 0, 0, 0, 1, 1, 0, 0, 1,
          0, 1, 1, 0, 0, 1, 0, 0, 1, 0]

# Regular F1
regular_f1 = f1_score(y_true, y_pred)

# Time-weighted F1 (recent matters more)
weighted_f1 = time_weighted_f1(y_true, y_pred, timestamps, decay_rate=0.05)

print(f"Regular F1: {regular_f1:.3f}")
print(f"Time-Weighted F1 (recent emphasized): {weighted_f1:.3f}")

# If recent predictions are worse, weighted F1 will be lower
```

**Use Cases:**
- **Fraud detection:** Catching fraud quickly is more valuable
- **Churn prediction:** Recent behavior more predictive
- **Concept drift:** Focus on recent model performance
- **Real-time systems:** SLA compliance timing

**Interview Tip:**
Propose time-weighted metrics when temporal context is mentioned. Shows awareness that not all predictions have equal value.

---

## Question 43
- [ ] Done
**How do you optimize threshold selection strategies when dealing with multiple competing metrics?**

**Answer:**

**Definition:**
When optimizing threshold, different metrics suggest different optimal values. Handle by: (1) setting a primary metric to optimize with constraints on others, (2) finding the Pareto-optimal threshold range, or (3) using business cost functions to weight trade-offs.

**Problem:**
- Threshold for max F1: 0.45
- Threshold for min cost: 0.35
- Threshold for 90% recall: 0.30
- Which to choose?

**Strategies:**

| Strategy | Description |
|----------|-------------|
| **Constrained optimization** | Maximize A subject to B ≥ threshold |
| **Pareto analysis** | Find non-dominated threshold range |
| **Cost function** | Single objective incorporating all metrics |
| **Stakeholder choice** | Present options with trade-offs |

**Python Code:**
```python
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix

def multi_metric_threshold_analysis(y_true, y_prob, cost_fp=1, cost_fn=1):
    """
    Analyze thresholds considering multiple metrics.
    Returns optimal threshold for different objectives.
    """
    thresholds = np.arange(0.1, 0.9, 0.02)
    results = []
    
    for thresh in thresholds:
        y_pred = [1 if p >= thresh else 0 for p in y_prob]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        cost = fp * cost_fp + fn * cost_fn
        
        results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cost': cost,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        })
    
    return results

def find_optimal_thresholds(results, constraints=None):
    """Find optimal thresholds for different objectives."""
    
    # 1. Max F1
    max_f1_result = max(results, key=lambda x: x['f1'])
    
    # 2. Min cost
    min_cost_result = min(results, key=lambda x: x['cost'])
    
    # 3. Constrained: Max F1 with recall >= 0.9
    constrained_results = [r for r in results if r['recall'] >= 0.9]
    max_f1_constrained = max(constrained_results, key=lambda x: x['f1']) if constrained_results else None
    
    # 4. Balanced: closest to precision = recall
    balanced_result = min(results, key=lambda x: abs(x['precision'] - x['recall']))
    
    return {
        'max_f1': max_f1_result,
        'min_cost': min_cost_result,
        'recall_constrained': max_f1_constrained,
        'balanced': balanced_result
    }

# Example
np.random.seed(42)
y_true = [0]*80 + [1]*20
y_prob = np.random.beta(2, 5, 80).tolist() + np.random.beta(5, 2, 20).tolist()

# Analyze
results = multi_metric_threshold_analysis(y_true, y_prob, cost_fp=10, cost_fn=100)
optima = find_optimal_thresholds(results)

print("=== Optimal Thresholds by Objective ===")
for objective, result in optima.items():
    if result:
        print(f"\n{objective.upper()}:")
        print(f"  Threshold: {result['threshold']:.2f}")
        print(f"  Precision: {result['precision']:.3f}, Recall: {result['recall']:.3f}")
        print(f"  F1: {result['f1']:.3f}, Cost: {result['cost']:.0f}")

# Present trade-off table for stakeholders
print("\n=== Threshold Options for Stakeholders ===")
print(f"{'Option':<20} {'Threshold':<10} {'Recall':<10} {'Precision':<10} {'Cost':<10}")
print("-" * 60)
for name, r in optima.items():
    if r:
        print(f"{name:<20} {r['threshold']:<10.2f} {r['recall']:<10.3f} {r['precision']:<10.3f} {r['cost']:<10.0f}")
```

**Decision Framework:**
1. Define business constraints first (e.g., "recall must be ≥ 90%")
2. Within constraints, optimize primary metric
3. If no constraints, present Pareto options to stakeholders
4. Document threshold choice rationale

**Interview Tip:**
Show that threshold selection is a business decision, not just a technical one. Ask about constraints before proposing a threshold.

---

## Question 44
- [ ] Done
**What techniques help you assess metric robustness against adversarial examples or data poisoning?**

**Answer:**

**Definition:**
Adversarial examples are inputs designed to fool the model; data poisoning corrupts training data. Assess robustness by: testing on adversarial samples, measuring metric stability under perturbations, and using certified robustness bounds. Report both clean and adversarial metrics.

**Attack Types:**

| Attack | Description | Impact on Metrics |
|--------|-------------|-------------------|
| **Adversarial examples** | Perturbed inputs at test time | Clean accuracy >> Adversarial accuracy |
| **Data poisoning** | Corrupted training data | Degraded test metrics |
| **Label flipping** | Wrong labels in training | Lower precision/recall |
| **Backdoor attacks** | Hidden triggers in data | Normal metrics, fails on triggers |

**Robustness Metrics:**

**1. Adversarial Accuracy:**
$$\text{Adv Accuracy} = \frac{\text{Correct on adversarial examples}}{\text{Total adversarial examples}}$$

**2. Robustness Gap:**
$$\text{Gap} = \text{Clean Accuracy} - \text{Adversarial Accuracy}$$

**3. Attack Success Rate:**
$$\text{ASR} = \frac{\text{Misclassified after attack}}{\text{Correct before attack}}$$

**Python Code:**
```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def add_adversarial_perturbation(X, epsilon=0.1):
    """Simple FGSM-like perturbation (for demonstration)."""
    noise = np.random.choice([-epsilon, epsilon], size=X.shape)
    return X + noise

def evaluate_robustness(model, X_test, y_test, epsilon_values=[0.01, 0.05, 0.1]):
    """Evaluate model robustness at different perturbation levels."""
    results = []
    
    # Clean accuracy
    y_pred_clean = model.predict(X_test)
    clean_acc = accuracy_score(y_test, y_pred_clean)
    clean_f1 = f1_score(y_test, y_pred_clean, average='weighted')
    
    results.append({
        'epsilon': 0,
        'accuracy': clean_acc,
        'f1': clean_f1,
        'gap': 0
    })
    
    for eps in epsilon_values:
        # Perturbed data
        X_adv = add_adversarial_perturbation(X_test, epsilon=eps)
        y_pred_adv = model.predict(X_adv)
        
        adv_acc = accuracy_score(y_test, y_pred_adv)
        adv_f1 = f1_score(y_test, y_pred_adv, average='weighted')
        
        results.append({
            'epsilon': eps,
            'accuracy': adv_acc,
            'f1': adv_f1,
            'gap': clean_acc - adv_acc
        })
    
    return results, clean_acc

def detect_data_poisoning(model, X_val, y_val, X_val_clean, y_val_clean):
    """Compare metrics on potentially poisoned vs clean validation."""
    y_pred = model.predict(X_val)
    y_pred_clean = model.predict(X_val_clean)
    
    acc_val = accuracy_score(y_val, y_pred)
    acc_clean = accuracy_score(y_val_clean, y_pred_clean)
    
    poisoning_indicator = acc_clean - acc_val
    
    return {
        'validation_accuracy': acc_val,
        'clean_accuracy': acc_clean,
        'gap': poisoning_indicator,
        'potential_poisoning': poisoning_indicator > 0.1
    }

# Example usage (simulated)
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
X_train = np.random.randn(200, 10)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
X_test = np.random.randn(50, 10)
y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Evaluate robustness
results, clean_acc = evaluate_robustness(model, X_test, y_test)

print("=== Robustness Analysis ===")
print(f"{'Epsilon':<10} {'Accuracy':<12} {'F1':<10} {'Gap':<10}")
print("-" * 42)
for r in results:
    print(f"{r['epsilon']:<10.2f} {r['accuracy']:<12.3f} {r['f1']:<10.3f} {r['gap']:<10.3f}")

# Robustness score
robustness_score = results[-1]['accuracy'] / clean_acc
print(f"\nRobustness Score (eps=0.1): {robustness_score:.3f}")
```

**Best Practices:**
- Report both clean and adversarial metrics
- Test multiple perturbation strengths
- Include attack success rate
- Consider certified defenses for critical applications

**Interview Tip:**
For security-sensitive applications (fraud, malware), mention adversarial robustness testing. Shows awareness of real-world threats to ML systems.

---

## Question 45
- [ ] Done
**How do you implement metric evaluation for continual learning scenarios with evolving task definitions?**

**Answer:**

**Definition:**
In continual learning, the model learns new tasks sequentially without forgetting old ones. Evaluate using: **backward transfer** (impact on old tasks), **forward transfer** (benefit to new tasks), **average accuracy**, and **forgetting measure**. Track metrics over time, not just final performance.

**Continual Learning Metrics:**

| Metric | Definition |
|--------|------------|
| **Average Accuracy** | Mean accuracy across all tasks after training |
| **Forgetting** | Accuracy drop on old tasks after learning new ones |
| **Backward Transfer** | How new learning affects old task performance |
| **Forward Transfer** | How past learning helps new tasks |

**Mathematical Formulation:**

**Average Accuracy** (after training on T tasks):
$$A = \frac{1}{T}\sum_{i=1}^{T} a_{T,i}$$

Where $a_{T,i}$ = accuracy on task $i$ after training on task $T$

**Forgetting** for task $i$:
$$F_i = \max_{t \in \{1,...,T-1\}} a_{t,i} - a_{T,i}$$

**Backward Transfer**:
$$BWT = \frac{1}{T-1}\sum_{i=1}^{T-1}(a_{T,i} - a_{i,i})$$

Negative BWT = catastrophic forgetting

**Python Code:**
```python
import numpy as np

class ContinualLearningEvaluator:
    def __init__(self, n_tasks):
        self.n_tasks = n_tasks
        # accuracy_matrix[t][i] = accuracy on task i after training on task t
        self.accuracy_matrix = np.zeros((n_tasks, n_tasks))
    
    def record_accuracy(self, trained_up_to_task, task_id, accuracy):
        """Record accuracy on task_id after training up to trained_up_to_task."""
        self.accuracy_matrix[trained_up_to_task][task_id] = accuracy
    
    def average_accuracy(self):
        """Average accuracy on all seen tasks after final training."""
        final_row = self.accuracy_matrix[-1]
        return np.mean(final_row[:len(final_row)])
    
    def forgetting(self):
        """Average forgetting across tasks."""
        forgetting_per_task = []
        
        for i in range(self.n_tasks - 1):
            # Max accuracy on task i across training stages
            max_acc = np.max(self.accuracy_matrix[:, i])
            # Final accuracy on task i
            final_acc = self.accuracy_matrix[-1, i]
            forgetting_per_task.append(max_acc - final_acc)
        
        return np.mean(forgetting_per_task) if forgetting_per_task else 0
    
    def backward_transfer(self):
        """Average backward transfer."""
        bwt_per_task = []
        
        for i in range(self.n_tasks - 1):
            # Accuracy right after training on task i
            acc_after_i = self.accuracy_matrix[i, i]
            # Final accuracy on task i
            final_acc = self.accuracy_matrix[-1, i]
            bwt_per_task.append(final_acc - acc_after_i)
        
        return np.mean(bwt_per_task) if bwt_per_task else 0
    
    def forward_transfer(self):
        """Average forward transfer (learning speed improvement)."""
        # Compare accuracy on new task before vs after prior training
        # Simplified: compare random baseline
        fwt_per_task = []
        random_baseline = 0.1  # Assuming 10% random accuracy
        
        for i in range(1, self.n_tasks):
            # Accuracy on task i right after learning it
            acc_on_i = self.accuracy_matrix[i, i]
            fwt_per_task.append(acc_on_i - random_baseline)
        
        return np.mean(fwt_per_task) if fwt_per_task else 0
    
    def report(self):
        print("=== Continual Learning Evaluation ===")
        print(f"Average Accuracy: {self.average_accuracy():.3f}")
        print(f"Forgetting: {self.forgetting():.3f}")
        print(f"Backward Transfer: {self.backward_transfer():+.3f}")
        print(f"Forward Transfer: {self.forward_transfer():+.3f}")
        print("\nAccuracy Matrix (rows=trained_up_to, cols=task):")
        print(self.accuracy_matrix)

# Example: 4 tasks
evaluator = ContinualLearningEvaluator(n_tasks=4)

# Simulate: After training on task 0
evaluator.record_accuracy(0, 0, 0.90)

# After training on task 1
evaluator.record_accuracy(1, 0, 0.85)  # Slight forgetting
evaluator.record_accuracy(1, 1, 0.88)

# After training on task 2
evaluator.record_accuracy(2, 0, 0.75)  # More forgetting
evaluator.record_accuracy(2, 1, 0.82)
evaluator.record_accuracy(2, 2, 0.91)

# After training on task 3
evaluator.record_accuracy(3, 0, 0.70)  # Catastrophic forgetting
evaluator.record_accuracy(3, 1, 0.78)
evaluator.record_accuracy(3, 2, 0.85)
evaluator.record_accuracy(3, 3, 0.92)

evaluator.report()
```

**Interview Tip:**
For continual learning, emphasize forgetting as the key metric. A model that achieves 95% on new tasks but forgets old ones is problematic.

---

## Question 46
- [ ] Done
**When should you use application-specific metrics versus standard classification metrics?**

**Answer:**

**Definition:**
Use **standard metrics** (F1, accuracy) for benchmarking and comparison across models. Use **application-specific metrics** when standard metrics don't capture business value, user experience, or domain-specific requirements. Often, you need both: standard for technical validation, custom for business validation.

**Comparison:**

| Aspect | Standard Metrics | Application-Specific |
|--------|-----------------|---------------------|
| Comparability | Easy across papers/models | Domain-specific |
| Interpretability | Technical | Business-aligned |
| Optimization target | General | Tailored |
| Examples | F1, AUC, Accuracy | Revenue, NPS, LTV |

**When to Use Each:**

| Scenario | Use | Example |
|----------|-----|---------|
| Academic benchmarking | Standard | ImageNet accuracy |
| Model comparison | Standard | F1-score |
| Production deployment | Application-specific | Revenue per prediction |
| A/B testing | Application-specific | Conversion rate |
| Stakeholder reporting | Application-specific | Customer satisfaction |
| Regulatory compliance | Often both | Fairness metrics + accuracy |

**Application-Specific Metric Examples:**

| Domain | Standard Metric | Application Metric |
|--------|----------------|-------------------|
| **Fraud detection** | Precision, Recall | Dollars saved, Investigation cost |
| **Medical diagnosis** | Sensitivity | Years of life saved |
| **Recommendation** | Click accuracy | Session engagement time |
| **Search** | Precision@K | Time to find answer |
| **Churn prediction** | AUC | Customer lifetime value saved |

**Python Code:**
```python
from sklearn.metrics import f1_score, precision_score, recall_score

def fraud_application_metrics(y_true, y_pred, avg_fraud_amount=500, 
                              investigation_cost=50):
    """Application-specific metrics for fraud detection."""
    
    # Standard metrics
    standard = {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    # Application metrics
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    
    application = {
        'fraud_caught_dollars': tp * avg_fraud_amount,
        'fraud_missed_dollars': fn * avg_fraud_amount,
        'investigation_cost': (tp + fp) * investigation_cost,
        'net_savings': tp * avg_fraud_amount - fn * avg_fraud_amount - (tp + fp) * investigation_cost,
        'roi': (tp * avg_fraud_amount) / ((tp + fp) * investigation_cost) if (tp + fp) > 0 else 0
    }
    
    return standard, application

# Example
y_true = [0]*90 + [1]*10
y_pred = [0]*85 + [1]*5 + [0]*2 + [1]*8  # Some false positives and false negatives

standard, application = fraud_application_metrics(y_true, y_pred)

print("=== Standard Metrics ===")
for k, v in standard.items():
    print(f"  {k}: {v:.3f}")

print("\n=== Application Metrics ===")
for k, v in application.items():
    print(f"  {k}: ${v:,.0f}" if 'dollar' in k or k in ['net_savings', 'investigation_cost'] else f"  {k}: {v:.2f}")
```

**Decision Framework:**
1. Use standard metrics for model development and comparison
2. Define application metrics with stakeholders
3. Ensure correlation between standard and application metrics
4. Report both in production dashboards

**Interview Tip:**
Always propose application-specific metrics for business contexts. Saying "I'd also track revenue impact, not just F1" shows business thinking.

---

## Question 47
- [ ] Done
**How do you handle metric reporting and visualization for complex multi-class, multi-label problems?**

**Answer:**

**Definition:**
Complex classification problems (many classes, multiple labels) require structured reporting: **confusion matrices** (heatmaps), **per-class breakdowns**, **hierarchical summaries**, and **interactive dashboards**. Avoid overwhelming stakeholders; use progressive disclosure (summary → detail).

**Visualization Strategies:**

| Problem Type | Visualizations |
|--------------|----------------|
| Multi-class (5-10) | Full confusion matrix heatmap |
| Multi-class (50+) | Hierarchical or grouped confusion matrix |
| Multi-label | Per-label bar charts, Hamming/Jaccard scores |
| Both | Dashboard with drill-down |

**Reporting Levels:**

1. **Executive Summary:** Single aggregate metric
2. **Overview:** Top-level class/label performance
3. **Detailed:** Per-class confusion, error analysis
4. **Deep Dive:** Sample-level predictions for debugging

**Python Code:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix

def multilabel_report(y_true, y_pred, label_names):
    """Comprehensive multi-label reporting."""
    n_labels = len(label_names)
    
    # Per-label metrics
    per_label = {}
    for i, name in enumerate(label_names):
        y_true_i = [y[i] for y in y_true]
        y_pred_i = [y[i] for y in y_pred]
        
        tp = sum(t == 1 and p == 1 for t, p in zip(y_true_i, y_pred_i))
        fp = sum(t == 0 and p == 1 for t, p in zip(y_true_i, y_pred_i))
        fn = sum(t == 1 and p == 0 for t, p in zip(y_true_i, y_pred_i))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_label[name] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    # Aggregate
    macro_f1 = np.mean([v['f1'] for v in per_label.values()])
    
    return per_label, macro_f1

def create_multiclass_visualization(y_true, y_pred, class_names, figsize=(12, 8)):
    """Create comprehensive multi-class visualization."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    im = axes[0, 0].imshow(cm, cmap='Blues')
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    plt.colorbar(im, ax=axes[0, 0])
    
    # 2. Per-class F1 Bar Chart
    from sklearn.metrics import f1_score
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    axes[0, 1].barh(class_names, per_class_f1, color='steelblue')
    axes[0, 1].set_xlabel('F1-Score')
    axes[0, 1].set_title('Per-Class F1')
    axes[0, 1].set_xlim(0, 1)
    
    # 3. Class Distribution (predicted vs actual)
    actual_counts = [list(y_true).count(i) for i in range(len(class_names))]
    pred_counts = [list(y_pred).count(i) for i in range(len(class_names))]
    
    x = np.arange(len(class_names))
    width = 0.35
    axes[1, 0].bar(x - width/2, actual_counts, width, label='Actual')
    axes[1, 0].bar(x + width/2, pred_counts, width, label='Predicted')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(class_names, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].set_title('Class Distribution')
    
    # 4. Summary Text
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    macro_f1 = np.mean(per_class_f1)
    axes[1, 1].axis('off')
    summary_text = f"""
    Summary Metrics
    ===============
    Accuracy: {accuracy:.3f}
    Macro F1: {macro_f1:.3f}
    
    Best Class: {class_names[np.argmax(per_class_f1)]} (F1={max(per_class_f1):.3f})
    Worst Class: {class_names[np.argmin(per_class_f1)]} (F1={min(per_class_f1):.3f})
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    return fig

# Example usage
np.random.seed(42)
class_names = ['Sports', 'Politics', 'Tech', 'Entertainment', 'Business']
y_true = np.random.randint(0, 5, 200)
y_pred = np.where(np.random.random(200) > 0.2, y_true, np.random.randint(0, 5, 200))

# Create visualization
fig = create_multiclass_visualization(y_true, y_pred, class_names)
# plt.show()

# Print text report
print(classification_report(y_true, y_pred, target_names=class_names))
```

**Dashboard Components:**
1. Executive: Single KPI card (Macro F1)
2. Overview: Class performance heatmap
3. Drill-down: Click class → see confusion details
4. Trends: Time-series of metrics

**Interview Tip:**
For complex problems, propose hierarchical reporting. Start with summary, allow drill-down. Shows awareness of stakeholder communication needs.

---

## Question 48
- [ ] Done
**What are the best practices for implementing metric-based early stopping during model training?**

**Answer:**

**Definition:**
Early stopping monitors a validation metric during training and stops when it stops improving, preventing overfitting. Best practices: use held-out validation set, choose appropriate patience (epochs to wait), save best model checkpoint, and select metric aligned with task goal.

**Key Parameters:**

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| **Metric** | What to monitor | val_loss, val_f1, val_accuracy |
| **Patience** | Epochs without improvement | 5-20 epochs |
| **Min delta** | Minimum improvement threshold | 0.001-0.01 |
| **Mode** | min (loss) or max (accuracy) | Task-dependent |
| **Restore best** | Load best weights at end | Usually True |

**Best Practices:**

1. **Use validation set, not training set**
2. **Choose metric aligned with task** (F1 for imbalanced, loss for general)
3. **Set reasonable patience** (too low = underfitting, too high = overfitting)
4. **Save checkpoints** for best model
5. **Consider multiple metrics** (stop on loss, report F1)

**Python Code:**
```python
import numpy as np

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.001, mode='max', 
                 restore_best=True):
        """
        patience: epochs to wait after improvement stops
        min_delta: minimum change to qualify as improvement
        mode: 'max' for metrics like accuracy, 'min' for loss
        restore_best: whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch = 0
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0
    
    def check(self, score, epoch, weights=None):
        """
        Check if training should stop.
        Returns: (should_stop, is_improvement)
        """
        if self.mode == 'max':
            is_improvement = score > self.best_score + self.min_delta
        else:
            is_improvement = score < self.best_score - self.min_delta
        
        if is_improvement:
            self.best_score = score
            self.best_epoch = epoch
            self.best_weights = weights
            self.counter = 0
            return False, True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                return True, False
            return False, False
    
    def get_best(self):
        return self.best_score, self.best_epoch, self.best_weights

# Simulation of training with early stopping
def train_with_early_stopping():
    early_stopping = EarlyStopping(patience=5, min_delta=0.005, mode='max')
    
    # Simulated validation F1 scores (improves then plateaus)
    val_f1_history = [0.5, 0.6, 0.68, 0.73, 0.77, 0.79, 0.805, 0.81, 
                      0.808, 0.806, 0.805, 0.803, 0.801, 0.799]
    
    print("=== Training with Early Stopping ===")
    print(f"{'Epoch':<8} {'Val F1':<10} {'Best':<10} {'Counter':<8} {'Action'}")
    print("-" * 50)
    
    for epoch, f1 in enumerate(val_f1_history):
        should_stop, is_improvement = early_stopping.check(f1, epoch)
        
        action = "New Best!" if is_improvement else f"Wait {early_stopping.counter}/{early_stopping.patience}"
        if should_stop:
            action = "STOP"
        
        print(f"{epoch:<8} {f1:<10.3f} {early_stopping.best_score:<10.3f} {early_stopping.counter:<8} {action}")
        
        if should_stop:
            break
    
    best_f1, best_epoch, _ = early_stopping.get_best()
    print(f"\nBest model: Epoch {best_epoch} with F1 = {best_f1:.3f}")
    return best_epoch, best_f1

train_with_early_stopping()

# Using Keras callback (conceptual)
# from keras.callbacks import EarlyStopping
# early_stop = EarlyStopping(
#     monitor='val_f1_score',
#     patience=10,
#     mode='max',
#     restore_best_weights=True
# )
# model.fit(X, y, callbacks=[early_stop])
```

**Common Mistakes:**
- Using training metric instead of validation
- Patience too low (stops before convergence)
- Not saving best checkpoint (loses best model)
- Wrong mode (min vs max confusion)

**Interview Tip:**
Always mention using validation metric, not training. Early stopping on training loss doesn't prevent overfitting!

---

## Question 49
- [ ] Done
**How do you optimize metric calculation efficiency for high-frequency evaluation in production systems?**

**Answer:**

**Definition:**
Production systems may need metric calculation thousands of times per second. Optimize by: using incremental/streaming metrics (avoid recomputing from scratch), batching updates, using efficient data structures (running counts), and sampling for approximate metrics.

**Optimization Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Incremental updates** | Maintain running TP/FP/FN counts | Real-time dashboards |
| **Batch aggregation** | Compute metrics on batches | High-throughput systems |
| **Sampling** | Approximate on random sample | Very high volume |
| **Caching** | Cache intermediate results | Repeated queries |
| **Approximate algorithms** | Probabilistic data structures | Memory-constrained |

**Incremental Metrics:**
Instead of recomputing from all predictions:
```
New F1 = f(old_TP + new_TP, old_FP + new_FP, old_FN + new_FN)
```

**Python Code:**
```python
import numpy as np
import time

class IncrementalMetrics:
    """Efficient incremental metric calculation for production."""
    
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.n_samples = 0
    
    def update(self, y_true, y_pred):
        """Update counts with new batch (efficient)."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        self.tp += np.sum((y_true == 1) & (y_pred == 1))
        self.fp += np.sum((y_true == 0) & (y_pred == 1))
        self.fn += np.sum((y_true == 1) & (y_pred == 0))
        self.tn += np.sum((y_true == 0) & (y_pred == 0))
        self.n_samples += len(y_true)
    
    def precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
    
    def recall(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
    
    def f1(self):
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    def accuracy(self):
        return (self.tp + self.tn) / self.n_samples if self.n_samples > 0 else 0
    
    def reset(self):
        """Reset for new evaluation window."""
        self.tp = self.fp = self.fn = self.tn = self.n_samples = 0

class SampledMetrics:
    """Approximate metrics using reservoir sampling."""
    
    def __init__(self, max_samples=10000):
        self.max_samples = max_samples
        self.samples = []
        self.n_seen = 0
    
    def update(self, y_true, y_pred):
        """Add samples using reservoir sampling."""
        for t, p in zip(y_true, y_pred):
            self.n_seen += 1
            if len(self.samples) < self.max_samples:
                self.samples.append((t, p))
            else:
                # Reservoir sampling
                idx = np.random.randint(0, self.n_seen)
                if idx < self.max_samples:
                    self.samples[idx] = (t, p)
    
    def f1(self):
        if not self.samples:
            return 0
        y_true = [s[0] for s in self.samples]
        y_pred = [s[1] for s in self.samples]
        
        tp = sum(t == 1 and p == 1 for t, p in self.samples)
        fp = sum(t == 0 and p == 1 for t, p in self.samples)
        fn = sum(t == 1 and p == 0 for t, p in self.samples)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Performance comparison
def benchmark():
    np.random.seed(42)
    
    # Simulate 100 batches of 1000 predictions each
    n_batches = 100
    batch_size = 1000
    
    incremental = IncrementalMetrics()
    all_true, all_pred = [], []
    
    start = time.time()
    for _ in range(n_batches):
        y_true = np.random.randint(0, 2, batch_size)
        y_pred = np.random.randint(0, 2, batch_size)
        incremental.update(y_true, y_pred)
    incremental_time = time.time() - start
    incremental_f1 = incremental.f1()
    
    # Compare with naive full recomputation
    from sklearn.metrics import f1_score
    start = time.time()
    all_true, all_pred = [], []
    for _ in range(n_batches):
        y_true = np.random.randint(0, 2, batch_size)
        y_pred = np.random.randint(0, 2, batch_size)
        all_true.extend(y_true)
        all_pred.extend(y_pred)
        naive_f1 = f1_score(all_true, all_pred)  # Recompute everything!
    naive_time = time.time() - start
    
    print("=== Efficiency Comparison ===")
    print(f"Incremental: {incremental_time*1000:.2f}ms, F1={incremental_f1:.4f}")
    print(f"Naive: {naive_time*1000:.2f}ms")
    print(f"Speedup: {naive_time/incremental_time:.1f}x")

benchmark()
```

**Production Recommendations:**
- Use incremental counters for exact metrics
- Use sampling for approximate real-time metrics
- Batch updates (e.g., every 1000 predictions)
- Consider approximate algorithms (HyperLogLog for cardinality)

**Interview Tip:**
For production ML, mention that O(n) recomputation doesn't scale. Incremental updates are O(batch) and constant memory.

---

## Question 50
- [ ] Done
**What strategies help you balance multiple competing metrics when making model selection decisions?**

**Answer:**

**Definition:**
When metrics conflict (e.g., high precision model vs high recall model), use structured decision frameworks: **primary metric with constraints**, **weighted composite scores**, **Pareto frontier analysis**, or **business value optimization**. The key is aligning technical metrics with business goals.

**Decision Strategies:**

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Primary + Constraints** | Maximize A, subject to B ≥ threshold | Clear business requirement |
| **Weighted Composite** | Score = w₁M₁ + w₂M₂ + ... | Quantified trade-offs |
| **Pareto Frontier** | Find non-dominated options | Present options to stakeholders |
| **Business Value** | Convert metrics to $ | ROI-focused decisions |
| **Satisficing** | Meet minimum thresholds for all | Risk-averse scenarios |

**Algorithm: Structured Model Selection**

```
1. Define must-have constraints (e.g., recall ≥ 90%)
2. Filter models that meet constraints
3. Among valid models, optimize primary metric
4. If tie, use secondary metrics
5. Document rationale for selection
```

**Python Code:**
```python
import numpy as np

def model_selection(models, constraints=None, weights=None, primary_metric='f1'):
    """
    Select best model using structured decision framework.
    
    models: dict {name: {metric: value}}
    constraints: dict {metric: min_value}
    weights: dict {metric: weight} for composite score
    primary_metric: metric to optimize if no weights
    """
    
    # Step 1: Filter by constraints
    if constraints:
        valid_models = {
            name: metrics for name, metrics in models.items()
            if all(metrics.get(m, 0) >= v for m, v in constraints.items())
        }
        
        if not valid_models:
            print("Warning: No models meet all constraints")
            valid_models = models  # Fall back to all
    else:
        valid_models = models
    
    # Step 2: Score models
    scores = {}
    
    if weights:
        # Weighted composite score
        for name, metrics in valid_models.items():
            score = sum(weights.get(m, 0) * metrics.get(m, 0) for m in weights)
            scores[name] = score
    else:
        # Primary metric
        for name, metrics in valid_models.items():
            scores[name] = metrics.get(primary_metric, 0)
    
    # Step 3: Rank and select
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_model = ranking[0][0]
    
    return {
        'selected': best_model,
        'score': scores[best_model],
        'ranking': ranking,
        'valid_models': list(valid_models.keys()),
        'filtered_out': list(set(models.keys()) - set(valid_models.keys()))
    }

def pareto_frontier(models, metrics_to_max):
    """Find models on Pareto frontier (non-dominated)."""
    pareto = []
    model_list = list(models.items())
    
    for i, (name_i, m_i) in enumerate(model_list):
        dominated = False
        for j, (name_j, m_j) in enumerate(model_list):
            if i == j:
                continue
            better_or_equal = all(m_j.get(k, 0) >= m_i.get(k, 0) for k in metrics_to_max)
            strictly_better = any(m_j.get(k, 0) > m_i.get(k, 0) for k in metrics_to_max)
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(name_i)
    
    return pareto

# Example: Model selection for fraud detection
models = {
    'Logistic Regression': {'f1': 0.72, 'precision': 0.80, 'recall': 0.65, 'latency_ms': 5},
    'Random Forest':       {'f1': 0.78, 'precision': 0.75, 'recall': 0.82, 'latency_ms': 50},
    'XGBoost':            {'f1': 0.82, 'precision': 0.78, 'recall': 0.86, 'latency_ms': 30},
    'Deep Learning':       {'f1': 0.80, 'precision': 0.70, 'recall': 0.92, 'latency_ms': 100},
}

print("=== Strategy 1: Constrained Optimization ===")
# Must have recall >= 80%, then maximize F1
result = model_selection(
    models, 
    constraints={'recall': 0.80},
    primary_metric='f1'
)
print(f"Selected: {result['selected']}")
print(f"Filtered out (recall < 80%): {result['filtered_out']}")

print("\n=== Strategy 2: Weighted Composite ===")
# Business weights: recall is 2x more important than precision
result = model_selection(
    models,
    weights={'f1': 0.4, 'recall': 0.4, 'precision': 0.2}
)
print(f"Selected: {result['selected']}")
print(f"Ranking: {result['ranking']}")

print("\n=== Strategy 3: Pareto Frontier ===")
pareto = pareto_frontier(models, ['f1', 'recall', 'precision'])
print(f"Pareto-optimal models: {pareto}")
print("Present these options to stakeholders for final decision")

print("\n=== Strategy 4: Business Value ===")
# Convert to dollars
for name, metrics in models.items():
    # Assume: TP saves $1000, FP costs $50, FN costs $1000
    # Estimated on 1000 predictions with 10% fraud rate
    tp_rate = metrics['recall'] * 0.10
    fp_rate = (1 - metrics['precision']) * metrics['recall'] * 0.10 / metrics['precision'] if metrics['precision'] > 0 else 0
    fn_rate = (1 - metrics['recall']) * 0.10
    
    value = 1000 * (tp_rate * 1000 - fp_rate * 50 - fn_rate * 1000)
    print(f"{name}: Estimated Value = ${value:,.0f}")
```

**Decision Documentation Template:**
```
Model Selection Decision
========================
Primary Metric: Recall (business requirement: catch fraud)
Constraints: Recall ≥ 80%, Latency ≤ 50ms
Selected Model: XGBoost

Rationale:
- Meets recall constraint (86%)
- Highest F1 among valid models
- Acceptable latency (30ms)

Trade-offs Accepted:
- Slightly lower precision than LR (0.78 vs 0.80)
- Higher latency than LR (30ms vs 5ms)
```

**Interview Tip:**
Show structured thinking: constraints first, then optimization. Never say "I'd pick the one with highest F1" without considering constraints and trade-offs.

---

---

# --- Accuracy, Precision, Recall & F1-Score Questions (from source .txt) ---

## Question 1
- [ ] Done

**How do you choose the most appropriate primary metric between accuracy, precision, recall, and F1-score for different business contexts?**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**When would you prioritize precision over recall in a fraud detection system, and how do you quantify the business impact?**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**How do you handle class imbalance when accuracy becomes misleading, and which alternative metrics provide better insights?**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**What strategies help you optimize for F1-score when precision and recall have conflicting optimization directions?**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**How do you implement weighted F1-score calculations for multi-class problems with uneven class distributions?**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**When should you use macro-averaged versus micro-averaged F1-scores in multi-label classification scenarios?**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**How do you handle threshold selection to optimize different metrics in binary classification problems?**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**What techniques help you visualize the trade-offs between precision and recall for stakeholder communication?**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**How do you implement confidence interval calculations for accuracy and F1-score in small dataset scenarios?**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**When would you use Matthews Correlation Coefficient instead of F1-score for binary classification evaluation?**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**How do you handle metric evaluation when dealing with hierarchical or nested class structures?**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**What are the best practices for reporting metric confidence when using cross-validation or bootstrap sampling?**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**How do you implement custom F1-score variants for domain-specific evaluation requirements?**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**When should you use balanced accuracy versus regular accuracy in imbalanced classification problems?**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**How do you handle metric calculation for multi-output classification problems with different evaluation priorities?**

**Answer:** _[To be filled]_

---

## Question 16
- [ ] Done

**What strategies help you communicate metric trade-offs to non-technical stakeholders effectively?**

**Answer:** _[To be filled]_

---

## Question 17
- [ ] Done

**How do you implement real-time metric monitoring for deployed models without ground truth labels?**

**Answer:** _[To be filled]_

---

## Question 18
- [ ] Done

**When would you use precision@k or recall@k metrics instead of traditional precision and recall?**

**Answer:** _[To be filled]_

---

## Question 19
- [ ] Done

**How do you handle metric evaluation for streaming data with concept drift and evolving class distributions?**

**Answer:** _[To be filled]_

---

## Question 20
- [ ] Done

**What techniques help you assess metric stability across different data splits and validation strategies?**

**Answer:** _[To be filled]_

---

## Question 21
- [ ] Done

**How do you implement cost-sensitive evaluation metrics that account for different misclassification costs?**

**Answer:** _[To be filled]_

---

## Question 22
- [ ] Done

**When should you use per-class precision and recall versus aggregate metrics for multi-class evaluation?**

**Answer:** _[To be filled]_

---

## Question 23
- [ ] Done

**How do you handle metric evaluation for imbalanced time-series classification problems?**

**Answer:** _[To be filled]_

---

## Question 24
- [ ] Done

**What are the best practices for comparing model performance across different metric combinations?**

**Answer:** _[To be filled]_

---

## Question 25
- [ ] Done

**How do you implement statistical significance testing for metric differences between competing models?**

**Answer:** _[To be filled]_

---

## Question 26
- [ ] Done

**When would you use harmonic mean versus arithmetic mean for combining precision and recall?**

**Answer:** _[To be filled]_

---

## Question 27
- [ ] Done

**How do you handle metric evaluation when ground truth labels have varying degrees of certainty?**

**Answer:** _[To be filled]_

---

## Question 28
- [ ] Done

**What strategies help you optimize metrics during hyperparameter tuning without overfitting to validation data?**

**Answer:** _[To be filled]_

---

## Question 29
- [ ] Done

**How do you implement metric evaluation for active learning scenarios with continuously updated training data?**

**Answer:** _[To be filled]_

---

## Question 30
- [ ] Done

**When should you use top-k accuracy versus standard accuracy in multi-class classification problems?**

**Answer:** _[To be filled]_

---

## Question 31
- [ ] Done

**How do you handle metric calculation for multi-label problems where partial matches should be credited?**

**Answer:** _[To be filled]_

---

## Question 32
- [ ] Done

**What techniques help you detect and handle metric gaming or exploitation in production systems?**

**Answer:** _[To be filled]_

---

## Question 33
- [ ] Done

**How do you implement metric evaluation for zero-shot or few-shot classification scenarios?**

**Answer:** _[To be filled]_

---

## Question 34
- [ ] Done

**When would you use geometric mean versus F1-score for combining precision and recall in specific domains?**

**Answer:** _[To be filled]_

---

## Question 35
- [ ] Done

**How do you handle metric evaluation for classification problems with missing or incomplete labels?**

**Answer:** _[To be filled]_

---

## Question 36
- [ ] Done

**What are the considerations for implementing custom metrics that align with specific business objectives?**

**Answer:** _[To be filled]_

---

## Question 37
- [ ] Done

**How do you optimize model performance when different metrics conflict with each other?**

**Answer:** _[To be filled]_

---

## Question 38
- [ ] Done

**When should you use micro-averaging versus macro-averaging for different types of classification problems?**

**Answer:** _[To be filled]_

---

## Question 39
- [ ] Done

**How do you implement metric evaluation for multi-task learning scenarios with shared representations?**

**Answer:** _[To be filled]_

---

## Question 40
- [ ] Done

**What strategies help you maintain metric reliability when transitioning from development to production?**

**Answer:** _[To be filled]_

---

## Question 41
- [ ] Done

**How do you handle metric evaluation for federated learning scenarios with distributed data?**

**Answer:** _[To be filled]_

---

## Question 42
- [ ] Done

**When would you implement time-weighted metrics for classification problems with temporal importance?**

**Answer:** _[To be filled]_

---

## Question 43
- [ ] Done

**How do you optimize threshold selection strategies when dealing with multiple competing metrics?**

**Answer:** _[To be filled]_

---

## Question 44
- [ ] Done

**What techniques help you assess metric robustness against adversarial examples or data poisoning?**

**Answer:** _[To be filled]_

---

## Question 45
- [ ] Done

**How do you implement metric evaluation for continual learning scenarios with evolving task definitions?**

**Answer:** _[To be filled]_

---

## Question 46
- [ ] Done

**When should you use application-specific metrics versus standard classification metrics?**

**Answer:** _[To be filled]_

---

## Question 47
- [ ] Done

**How do you handle metric reporting and visualization for complex multi-class, multi-label problems?**

**Answer:** _[To be filled]_

---

## Question 48
- [ ] Done

**What are the best practices for implementing metric-based early stopping during model training?**

**Answer:** _[To be filled]_

---

## Question 49
- [ ] Done

**How do you optimize metric calculation efficiency for high-frequency evaluation in production systems?**

**Answer:** _[To be filled]_

---

## Question 50
- [ ] Done

**What strategies help you balance multiple competing metrics when making model selection decisions?**

**Answer:** _[To be filled]_

---



---

# --- Confusion Matrix Questions (from source .txt) ---

## Question 1
- [ ] Done

**How do you interpret confusion matrices for multi-class problems with more than 10 classes effectively?**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**What visualization techniques help communicate confusion matrix insights to non-technical stakeholders?**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**How do you identify and address systematic misclassification patterns revealed by confusion matrix analysis?**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**When should you normalize confusion matrices by row, column, or total for different analytical purposes?**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**How do you handle confusion matrix analysis for imbalanced datasets where some classes have very few samples?**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**What techniques help you extract actionable insights from confusion matrices in production model monitoring?**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**How do you implement confidence intervals for confusion matrix elements in statistical evaluation scenarios?**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**When would you use cost-weighted confusion matrices to account for different misclassification penalties?**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**How do you handle confusion matrix analysis for hierarchical classification problems with nested categories?**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**What strategies help you identify which classes are most commonly confused and why?**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**How do you implement automated analysis of confusion matrix patterns for model debugging?**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**When should you use micro-averaged versus macro-averaged metrics derived from confusion matrices?**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**How do you handle confusion matrix visualization for streaming data with evolving class distributions?**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**What techniques help you compare confusion matrices across different models or time periods?**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**How do you implement confusion matrix analysis for multi-label classification problems?**

**Answer:** _[To be filled]_

---

## Question 16
- [ ] Done

**When would you use confusion matrices for feature selection and engineering insights?**

**Answer:** _[To be filled]_

---

## Question 17
- [ ] Done

**How do you handle confusion matrix interpretation when dealing with ordinal classification problems?**

**Answer:** _[To be filled]_

---

## Question 18
- [ ] Done

**What are the best practices for storing and versioning confusion matrices in MLOps pipelines?**

**Answer:** _[To be filled]_

---

## Question 19
- [ ] Done

**How do you implement real-time confusion matrix monitoring for deployed classification models?**

**Answer:** _[To be filled]_

---

## Question 20
- [ ] Done

**When should you focus on specific regions of confusion matrices versus overall patterns?**

**Answer:** _[To be filled]_

---

## Question 21
- [ ] Done

**How do you handle confusion matrix analysis for few-shot learning scenarios with limited examples?**

**Answer:** _[To be filled]_

---

## Question 22
- [ ] Done

**What techniques help you identify data quality issues through confusion matrix pattern analysis?**

**Answer:** _[To be filled]_

---

## Question 23
- [ ] Done

**How do you implement confusion matrix-based stopping criteria during iterative model training?**

**Answer:** _[To be filled]_

---

## Question 24
- [ ] Done

**When would you use multiple confusion matrices for different evaluation metrics simultaneously?**

**Answer:** _[To be filled]_

---

## Question 25
- [ ] Done

**How do you handle confusion matrix analysis for models with prediction confidence scores?**

**Answer:** _[To be filled]_

---

## Question 26
- [ ] Done

**What strategies help you optimize model architecture based on confusion matrix insights?**

**Answer:** _[To be filled]_

---

## Question 27
- [ ] Done

**How do you implement confusion matrix analysis for ensemble methods with multiple base models?**

**Answer:** _[To be filled]_

---

## Question 28
- [ ] Done

**When should you use confusion matrices for calibration assessment in probabilistic classifiers?**

**Answer:** _[To be filled]_

---

## Question 29
- [ ] Done

**How do you handle confusion matrix comparison across different data splits or validation strategies?**

**Answer:** _[To be filled]_

---

## Question 30
- [ ] Done

**What techniques help you detect concept drift using confusion matrix evolution over time?**

**Answer:** _[To be filled]_

---

## Question 31
- [ ] Done

**How do you implement confusion matrix analysis for active learning annotation strategies?**

**Answer:** _[To be filled]_

---

## Question 32
- [ ] Done

**When would you use confusion matrices to guide data augmentation strategies?**

**Answer:** _[To be filled]_

---

## Question 33
- [ ] Done

**How do you handle confusion matrix interpretation for models with reject options or abstention?**

**Answer:** _[To be filled]_

---

## Question 34
- [ ] Done

**What are the considerations for implementing confusion matrix analysis in federated learning?**

**Answer:** _[To be filled]_

---

## Question 35
- [ ] Done

**How do you optimize confusion matrix visualization for high-dimensional classification problems?**

**Answer:** _[To be filled]_

---

## Question 36
- [ ] Done

**When should you use confusion matrices for model interpretability and explainability analysis?**

**Answer:** _[To be filled]_

---

## Question 37
- [ ] Done

**How do you handle confusion matrix analysis for classification problems with missing labels?**

**Answer:** _[To be filled]_

---

## Question 38
- [ ] Done

**What techniques help you identify bias patterns through confusion matrix demographic analysis?**

**Answer:** _[To be filled]_

---

## Question 39
- [ ] Done

**How do you implement confusion matrix-based metrics for continuous model evaluation?**

**Answer:** _[To be filled]_

---

## Question 40
- [ ] Done

**When would you use confusion matrices for transfer learning evaluation across different domains?**

**Answer:** _[To be filled]_

---

## Question 41
- [ ] Done

**How do you handle confusion matrix analysis for time-series classification with temporal dependencies?**

**Answer:** _[To be filled]_

---

## Question 42
- [ ] Done

**What strategies help you optimize hyperparameters based on confusion matrix pattern analysis?**

**Answer:** _[To be filled]_

---

## Question 43
- [ ] Done

**How do you implement confusion matrix analysis for multi-task learning scenarios?**

**Answer:** _[To be filled]_

---

## Question 44
- [ ] Done

**When should you use confusion matrices for data preprocessing and cleaning decisionmaking?**

**Answer:** _[To be filled]_

---

## Question 45
- [ ] Done

**How do you handle confusion matrix interpretation for models with varying prediction granularity?**

**Answer:** _[To be filled]_

---

## Question 46
- [ ] Done

**What techniques help you assess model robustness using confusion matrix stability analysis?**

**Answer:** _[To be filled]_

---

## Question 47
- [ ] Done

**How do you implement automated alerts based on confusion matrix pattern changes?**

**Answer:** _[To be filled]_

---

## Question 48
- [ ] Done

**When would you use confusion matrices for curriculum learning and training data ordering?**

**Answer:** _[To be filled]_

---

## Question 49
- [ ] Done

**How do you handle confusion matrix analysis for zero-shot classification evaluation?**

**Answer:** _[To be filled]_

---

## Question 50
- [ ] Done

**What are the best practices for integrating confusion matrix insights into model improvement workflows?**

**Answer:** _[To be filled]_

---



---

# --- ROC Curve & AUC Questions (from source .txt) ---

## Question 1
- [ ] Done

**How do you interpret ROC curves when dealing with severely imbalanced datasets, and what alternatives should you consider?**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**What are the advantages of Precision-Recall curves over ROC curves for rare event detection scenarios?**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**How do you handle ROC curve analysis for multi-class problems using one-vs-all or one-vsone approaches?**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**When would you use micro-averaged versus macro-averaged AUC for multi-class evaluation?**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**How do you implement statistical significance testing for AUC differences between competing models?**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**What techniques help you optimize the operating point selection using ROC curve analysis?**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**How do you handle ROC curve interpretation when class distributions change between training and production?**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**When should you use partial AUC (pAUC) instead of full AUC for specific application requirements?**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**How do you implement confidence intervals and bootstrap sampling for ROC curve stability assessment?**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**What strategies help you communicate ROC curve insights to business stakeholders without statistical backgrounds?**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**How do you handle ROC curve analysis for cost-sensitive classification with asymmetric misclassification costs?**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**When would you use lift curves or KS statistics instead of ROC curves for model evaluation?**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**How do you implement ROC curve analysis for models producing prediction intervals rather than point estimates?**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**What techniques help you identify optimal threshold selection strategies using ROC curve analysis?**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**How do you handle ROC curve comparison across different validation strategies and data splits?**

**Answer:** _[To be filled]_

---

## Question 16
- [ ] Done

**When should you use time-dependent ROC analysis for survival analysis or time-to-event problems?**

**Answer:** _[To be filled]_

---

## Question 17
- [ ] Done

**How do you implement ROC curve analysis for streaming data with concept drift monitoring?**

**Answer:** _[To be filled]_

---

## Question 18
- [ ] Done

**What are the best practices for ROC curve visualization in multi-model comparison scenarios?**

**Answer:** _[To be filled]_

---

## Question 19
- [ ] Done

**How do you handle ROC curve interpretation for ensemble methods with multiple base classifiers?**

**Answer:** _[To be filled]_

---

## Question 20
- [ ] Done

**When would you use ROC convex hull analysis for optimal classifier combination strategies?**

**Answer:** _[To be filled]_

---

## Question 21
- [ ] Done

**How do you implement ROC curve analysis for active learning and annotation prioritization?**

**Answer:** _[To be filled]_

---

## Question 22
- [ ] Done

**What techniques help you detect overfitting using ROC curve analysis across train, validation, and test sets?**

**Answer:** _[To be filled]_

---

## Question 23
- [ ] Done

**How do you handle ROC curve analysis for federated learning scenarios with distributed evaluation?**

**Answer:** _[To be filled]_

---

## Question 24
- [ ] Done

**When should you use ROC curves versus other evaluation metrics for specific domain applications?**

**Answer:** _[To be filled]_

---

## Question 25
- [ ] Done

**How do you implement ROC curve analysis for hierarchical classification problems?**

**Answer:** _[To be filled]_

---

## Question 26
- [ ] Done

**What strategies help you optimize model calibration using ROC curve and reliability diagram analysis?**

**Answer:** _[To be filled]_

---

## Question 27
- [ ] Done

**How do you handle ROC curve analysis for ordinal classification problems with ranked outcomes?**

**Answer:** _[To be filled]_

---

## Question 28
- [ ] Done

**When would you use ROC curves for feature selection and engineering evaluation?**

**Answer:** _[To be filled]_

---

## Question 29
- [ ] Done

**How do you implement ROC curve analysis for multi-label classification with label dependencies?**

**Answer:** _[To be filled]_

---

## Question 30
- [ ] Done

**What techniques help you assess model robustness using ROC curve stability analysis?**

**Answer:** _[To be filled]_

---

## Question 31
- [ ] Done

**How do you handle ROC curve interpretation for models with prediction confidence or uncertainty estimates?**

**Answer:** _[To be filled]_

---

## Question 32
- [ ] Done

**When should you use ROC curves for anomaly detection and outlier identification evaluation?**

**Answer:** _[To be filled]_

---

## Question 33
- [ ] Done

**How do you implement ROC curve analysis for continual learning scenarios with evolving tasks?**

**Answer:** _[To be filled]_

---

## Question 34
- [ ] Done

**What are the considerations for ROC curve analysis in privacy-preserving machine learning?**

**Answer:** _[To be filled]_

---

## Question 35
- [ ] Done

**How do you optimize threshold selection when multiple ROC-derived metrics have conflicting optima?**

**Answer:** _[To be filled]_

---

## Question 36
- [ ] Done

**When would you use ROC curves for transfer learning evaluation across different domains?**

**Answer:** _[To be filled]_

---

## Question 37
- [ ] Done

**How do you handle ROC curve analysis for imbalanced time-series classification problems?**

**Answer:** _[To be filled]_

---

## Question 38
- [ ] Done

**What techniques help you identify bias and fairness issues using ROC curve demographic analysis?**

**Answer:** _[To be filled]_

---

## Question 39
- [ ] Done

**How do you implement ROC curve analysis for zero-shot and few-shot learning evaluation?**

**Answer:** _[To be filled]_

---

## Question 40
- [ ] Done

**When should you use ROC curves versus other metrics for real-time model monitoring?**

**Answer:** _[To be filled]_

---

## Question 41
- [ ] Done

**How do you handle ROC curve comparison for models with different output formats or scales?**

**Answer:** _[To be filled]_

---

## Question 42
- [ ] Done

**What strategies help you optimize hyperparameters using ROC curve-based objective functions?**

**Answer:** _[To be filled]_

---

## Question 43
- [ ] Done

**How do you implement ROC curve analysis for multi-task learning with shared representations?**

**Answer:** _[To be filled]_

---

## Question 44
- [ ] Done

**When would you use ROC curves for data quality assessment and preprocessing validation?**

**Answer:** _[To be filled]_

---

## Question 45
- [ ] Done

**How do you handle ROC curve interpretation for models with missing or incomplete predictions?**

**Answer:** _[To be filled]_

---

## Question 46
- [ ] Done

**What techniques help you assess generalization performance using ROC curve crossvalidation analysis?**

**Answer:** _[To be filled]_

---

## Question 47
- [ ] Done

**How do you implement automated model selection using ROC curve-based criteria?**

**Answer:** _[To be filled]_

---

## Question 48
- [ ] Done

**When should you use ROC curves for experimental design and sample size determination?**

**Answer:** _[To be filled]_

---

## Question 49
- [ ] Done

**How do you handle ROC curve analysis for classification problems with label noise or uncertainty?**

**Answer:** _[To be filled]_

---

## Question 50
- [ ] Done

**What are the best practices for integrating ROC curve insights into automated ML pipelines?**

**Answer:** _[To be filled]_

---



---

# --- Log Loss Questions (from source .txt) ---

## Question 1
- [ ] Done

**How do you interpret log loss values in the context of different classification problems and baseline models?**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**When would you choose log loss over accuracy or AUC as your primary optimization metric?**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**How do you handle log loss calculation when your model produces probability estimates very close to 0 or 1?**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**What techniques help you optimize log loss while maintaining good calibration in probabilistic classifiers?**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**How do you implement weighted log loss for imbalanced datasets with different class importance?**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**When should you use log loss versus other probabilistic metrics like Brier score for model evaluation?**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**How do you handle log loss interpretation for multi-class problems with varying numbers of classes?**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**What strategies help you communicate log loss improvements to stakeholders unfamiliar with the metric?**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**How do you implement log loss monitoring for deployed models without immediate ground truth feedback?**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**When would you use log loss for early stopping during training versus other convergence criteria?**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**How do you handle log loss calculation for multi-label classification problems?**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**What techniques help you identify overconfident predictions using log loss analysis?**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**How do you implement log loss-based hyperparameter optimization in automated ML pipelines?**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**When should you use log loss for model selection versus other evaluation metrics?**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**How do you handle log loss evaluation for streaming data with concept drift?**

**Answer:** _[To be filled]_

---

## Question 16
- [ ] Done

**What are the best practices for comparing log loss across different model architectures?**

**Answer:** _[To be filled]_

---

## Question 17
- [ ] Done

**How do you implement confidence intervals for log loss estimates in small dataset scenarios?**

**Answer:** _[To be filled]_

---

## Question 18
- [ ] Done

**When would you use log loss decomposition to understand prediction errors better?**

**Answer:** _[To be filled]_

---

## Question 19
- [ ] Done

**How do you handle log loss optimization in the presence of label noise or uncertainty?**

**Answer:** _[To be filled]_

---

## Question 20
- [ ] Done

**What strategies help you balance log loss minimization with other objectives like fairness or interpretability?**

**Answer:** _[To be filled]_

---

## Question 21
- [ ] Done

**How do you implement log loss calculation for hierarchical classification problems?**

**Answer:** _[To be filled]_

---

## Question 22
- [ ] Done

**When should you use log loss versus cross-entropy in different machine learning frameworks?**

**Answer:** _[To be filled]_

---

## Question 23
- [ ] Done

**How do you handle log loss evaluation for zero-shot and few-shot learning scenarios?**

**Answer:** _[To be filled]_

---

## Question 24
- [ ] Done

**What techniques help you detect calibration issues using log loss and reliability diagrams?**

**Answer:** _[To be filled]_

---

## Question 25
- [ ] Done

**How do you implement log loss monitoring for ensemble methods with multiple base models?**

**Answer:** _[To be filled]_

---

## Question 26
- [ ] Done

**When would you use log loss for active learning and annotation prioritization strategies?**

**Answer:** _[To be filled]_

---

## Question 27
- [ ] Done

**How do you handle log loss interpretation for models with reject options or abstention mechanisms?**

**Answer:** _[To be filled]_

---

## Question 28
- [ ] Done

**What are the considerations for log loss optimization in federated learning scenarios?**

**Answer:** _[To be filled]_

---

## Question 29
- [ ] Done

**How do you optimize log loss while maintaining computational efficiency in large-scale applications?**

**Answer:** _[To be filled]_

---

## Question 30
- [ ] Done

**When should you use log loss derivatives for gradient-based optimization versus other loss functions?**

**Answer:** _[To be filled]_

---

## Question 31
- [ ] Done

**How do you handle log loss calculation for ordinal classification problems?**

**Answer:** _[To be filled]_

---

## Question 32
- [ ] Done

**What techniques help you identify bias patterns using log loss demographic analysis?**

**Answer:** _[To be filled]_

---

## Question 33
- [ ] Done

**How do you implement log loss-based stopping criteria for iterative training algorithms?**

**Answer:** _[To be filled]_

---

## Question 34
- [ ] Done

**When would you use log loss for transfer learning evaluation across different domains?**

**Answer:** _[To be filled]_

---

## Question 35
- [ ] Done

**How do you handle log loss optimization for multi-task learning with shared representations?**

**Answer:** _[To be filled]_

---

## Question 36
- [ ] Done

**What strategies help you optimize log loss in the presence of missing labels or partial supervision?**

**Answer:** _[To be filled]_

---

## Question 37
- [ ] Done

**How do you implement log loss analysis for continual learning scenarios?**

**Answer:** _[To be filled]_

---

## Question 38
- [ ] Done

**When should you use log loss versus other metrics for real-time model performance monitoring?**

**Answer:** _[To be filled]_

---

## Question 39
- [ ] Done

**How do you handle log loss evaluation for models producing structured outputs?**

**Answer:** _[To be filled]_

---

## Question 40
- [ ] Done

**What techniques help you assess model robustness using log loss sensitivity analysis?**

**Answer:** _[To be filled]_

---

## Question 41
- [ ] Done

**How do you implement log loss optimization for cost-sensitive classification problems?**

**Answer:** _[To be filled]_

---

## Question 42
- [ ] Done

**When would you use log loss for curriculum learning and training data ordering?**

**Answer:** _[To be filled]_

---

## Question 43
- [ ] Done

**How do you handle log loss interpretation for models with varying prediction granularity?**

**Answer:** _[To be filled]_

---

## Question 44
- [ ] Done

**What are the best practices for log loss reporting in research and production environments?**

**Answer:** _[To be filled]_

---

## Question 45
- [ ] Done

**How do you optimize log loss while maintaining privacy constraints in sensitive applications?**

**Answer:** _[To be filled]_

---

## Question 46
- [ ] Done

**When should you use log loss for anomaly detection and outlier identification?**

**Answer:** _[To be filled]_

---

## Question 47
- [ ] Done

**How do you handle log loss calculation for time-series classification with temporal dependencies?**

**Answer:** _[To be filled]_

---

## Question 48
- [ ] Done

**What strategies help you optimize log loss in resource-constrained environments?**

**Answer:** _[To be filled]_

---

## Question 49
- [ ] Done

**How do you implement log loss-based model averaging and ensemble strategies?**

**Answer:** _[To be filled]_

---

## Question 50
- [ ] Done

**What techniques help you validate log loss improvements through statistical significance testing?**

**Answer:** _[To be filled]_

---



---

# --- MAE, MSE & RMSE Questions (from source .txt) ---

## Question 1
- [ ] Done

**How do you choose between MAE, MSE, and RMSE based on your specific regression problem's error tolerance requirements?**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**When would you use MAE over MSE to reduce the influence of outliers in your evaluation metrics?**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**How do you handle the scale dependency of MSE and RMSE when comparing models across different target variable ranges?**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**What techniques help you interpret RMSE values in the context of your domain-specific applications?**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**How do you implement weighted versions of MAE, MSE, and RMSE for heteroscedastic regression problems?**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**When should you use normalized or relative versions of these metrics for cross-dataset model comparison?**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**How do you handle MAE, MSE, and RMSE evaluation for multi-output regression problems?**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**What strategies help you communicate regression metric differences to non-technical stakeholders?**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**How do you implement confidence intervals for MAE, MSE, and RMSE estimates in small sample scenarios?**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**When would you use these metrics for hyperparameter optimization versus other regression evaluation criteria?**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**How do you handle metric evaluation for time-series forecasting with seasonal patterns and trends?**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**What techniques help you identify systematic bias patterns using these regression metrics?**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**How do you implement real-time monitoring of MAE, MSE, and RMSE for deployed regression models?**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**When should you use log-transformed versions of these metrics for skewed target distributions?**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**How do you handle metric evaluation for regression problems with censored or truncated data?**

**Answer:** _[To be filled]_

---

## Question 16
- [ ] Done

**What are the best practices for comparing MAE, MSE, and RMSE across different validation strategies?**

**Answer:** _[To be filled]_

---

## Question 17
- [ ] Done

**How do you implement these metrics for streaming regression with concept drift monitoring?**

**Answer:** _[To be filled]_

---

## Question 18
- [ ] Done

**When would you use quantile-based alternatives to these metrics for robust regression evaluation?**

**Answer:** _[To be filled]_

---

## Question 19
- [ ] Done

**How do you handle metric calculation for regression problems with missing target values?**

**Answer:** _[To be filled]_

---

## Question 20
- [ ] Done

**What strategies help you optimize model performance when different metrics suggest conflicting improvements?**

**Answer:** _[To be filled]_

---

## Question 21
- [ ] Done

**How do you implement these metrics for multi-task regression with shared representations?**

**Answer:** _[To be filled]_

---

## Question 22
- [ ] Done

**When should you use directional accuracy alongside MAE, MSE, and RMSE for forecasting evaluation?**

**Answer:** _[To be filled]_

---

## Question 23
- [ ] Done

**How do you handle metric evaluation for regression problems with heterogeneous data types?**

**Answer:** _[To be filled]_

---

## Question 24
- [ ] Done

**What techniques help you assess model calibration using these metrics combined with prediction intervals?**

**Answer:** _[To be filled]_

---

## Question 25
- [ ] Done

**How do you implement statistical significance testing for metric differences between competing models?**

**Answer:** _[To be filled]_

---

## Question 26
- [ ] Done

**When would you use these metrics for active learning and data collection prioritization?**

**Answer:** _[To be filled]_

---

## Question 27
- [ ] Done

**How do you handle metric interpretation for ordinal regression problems with ranked outcomes?**

**Answer:** _[To be filled]_

---

## Question 28
- [ ] Done

**What are the considerations for implementing these metrics in federated learning scenarios?**

**Answer:** _[To be filled]_

---

## Question 29
- [ ] Done

**How do you optimize these metrics while maintaining computational efficiency in large-scale applications?**

**Answer:** _[To be filled]_

---

## Question 30
- [ ] Done

**When should you use these metrics for transfer learning evaluation across different domains?**

**Answer:** _[To be filled]_

---

## Question 31
- [ ] Done

**How do you handle metric evaluation for zero-shot and few-shot regression scenarios?**

**Answer:** _[To be filled]_

---

## Question 32
- [ ] Done

**What techniques help you identify feature importance using these metrics in sensitivity analysis?**

**Answer:** _[To be filled]_

---

## Question 33
- [ ] Done

**How do you implement these metrics for continual learning with evolving regression tasks?**

**Answer:** _[To be filled]_

---

## Question 34
- [ ] Done

**When would you use these metrics for anomaly detection in regression model outputs?**

**Answer:** _[To be filled]_

---

## Question 35
- [ ] Done

**How do you handle metric calculation for regression problems with label noise or measurement errors?**

**Answer:** _[To be filled]_

---

## Question 36
- [ ] Done

**What strategies help you balance these metrics with other objectives like fairness or interpretability?**

**Answer:** _[To be filled]_

---

## Question 37
- [ ] Done

**How do you implement these metrics for ensemble regression methods with multiple base models?**

**Answer:** _[To be filled]_

---

## Question 38
- [ ] Done

**When should you use these metrics for early stopping during iterative training procedures?**

**Answer:** _[To be filled]_

---

## Question 39
- [ ] Done

**How do you handle metric evaluation for regression problems with non-stationary target distributions?**

**Answer:** _[To be filled]_

---

## Question 40
- [ ] Done

**What techniques help you assess model robustness using these metrics across different data conditions?**

**Answer:** _[To be filled]_

---

## Question 41
- [ ] Done

**How do you implement these metrics for cost-sensitive regression with asymmetric loss functions?**

**Answer:** _[To be filled]_

---

## Question 42
- [ ] Done

**When would you use these metrics for curriculum learning and training data sequencing?**

**Answer:** _[To be filled]_

---

## Question 43
- [ ] Done

**How do you handle metric interpretation for regression models with prediction uncertainty estimates?**

**Answer:** _[To be filled]_

---

## Question 44
- [ ] Done

**What are the best practices for reporting these metrics in research and production environments?**

**Answer:** _[To be filled]_

---

## Question 45
- [ ] Done

**How do you optimize these metrics in privacy-preserving regression scenarios?**

**Answer:** _[To be filled]_

---

## Question 46
- [ ] Done

**When should you use these metrics for experimental design and sample size determination?**

**Answer:** _[To be filled]_

---

## Question 47
- [ ] Done

**How do you handle metric evaluation for regression problems with structured or hierarchical outputs?**

**Answer:** _[To be filled]_

---

## Question 48
- [ ] Done

**What strategies help you validate metric improvements through cross-validation and bootstrap sampling?**

**Answer:** _[To be filled]_

---

## Question 49
- [ ] Done

**How do you implement these metrics for reinforcement learning with continuous action spaces?**

**Answer:** _[To be filled]_

---

## Question 50
- [ ] Done

**What techniques help you combine these metrics with domain-specific evaluation criteria for comprehensive assessment?**

**Answer:** _[To be filled]_

---



---

# --- BLEU & ROUGE (NLP) Questions (from source .txt) ---

## Question 1
- [ ] Done

**How do you choose between BLEU and ROUGE metrics based on your specific NLP evaluation requirements?**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**What are the limitations of BLEU scores for evaluating creative text generation versus factual content?**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**How do you handle ROUGE evaluation when reference summaries have varying lengths and styles?**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**When would you use BLEU-4 versus BLEU-1 or BLEU-2 for different text generation tasks?**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**How do you implement meaningful BLEU/ROUGE evaluation with limited reference texts?**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**What techniques help you interpret BLEU and ROUGE scores in the context of human evaluation?**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**How do you handle multi-reference evaluation scenarios to improve BLEU/ROUGE reliability?**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**When should you use sentence-level versus corpus-level BLEU/ROUGE calculations?**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**How do you implement BLEU and ROUGE evaluation for multilingual text generation systems?**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**What strategies help you address the known biases and limitations of BLEU/ROUGE metrics?**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**How do you handle BLEU/ROUGE evaluation for dialogue systems with conversational context?**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**When would you use ROUGE-L versus ROUGE-1/ROUGE-2 for different summarization tasks?**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**How do you implement statistical significance testing for BLEU/ROUGE score differences?**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**What techniques help you correlate BLEU/ROUGE scores with human judgments of quality?**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**How do you handle evaluation when generated text contains novel but correct information not in references?**

**Answer:** _[To be filled]_

---

## Question 16
- [ ] Done

**When should you complement BLEU/ROUGE with semantic similarity metrics like BERTScore?**

**Answer:** _[To be filled]_

---

## Question 17
- [ ] Done

**How do you implement BLEU/ROUGE evaluation for streaming or real-time text generation?**

**Answer:** _[To be filled]_

---

## Question 18
- [ ] Done

**What are the best practices for preprocessing text before BLEU/ROUGE calculation?**

**Answer:** _[To be filled]_

---

## Question 19
- [ ] Done

**How do you handle BLEU/ROUGE evaluation for domain-specific text with specialized terminology?**

**Answer:** _[To be filled]_

---

## Question 20
- [ ] Done

**When would you use variants like METEOR or CIDEr instead of traditional BLEU/ROUGE?**

**Answer:** _[To be filled]_

---

## Question 21
- [ ] Done

**How do you implement BLEU/ROUGE evaluation for code generation and programming tasks?**

**Answer:** _[To be filled]_

---

## Question 22
- [ ] Done

**What strategies help you optimize text generation models using BLEU/ROUGE as training objectives?**

**Answer:** _[To be filled]_

---

## Question 23
- [ ] Done

**How do you handle evaluation for abstractive summarization where paraphrasing is expected?**

**Answer:** _[To be filled]_

---

## Question 24
- [ ] Done

**When should you use BLEU/ROUGE for model selection versus other evaluation criteria?**

**Answer:** _[To be filled]_

---

## Question 25
- [ ] Done

**How do you implement BLEU/ROUGE evaluation for multi-document summarization tasks?**

**Answer:** _[To be filled]_

---

## Question 26
- [ ] Done

**What techniques help you assess the stability and reliability of BLEU/ROUGE measurements?**

**Answer:** _[To be filled]_

---

## Question 27
- [ ] Done

**How do you handle BLEU/ROUGE evaluation for personalized or style-specific text generation?**

**Answer:** _[To be filled]_

---

## Question 28
- [ ] Done

**When would you use weighted versions of BLEU/ROUGE for different evaluation priorities?**

**Answer:** _[To be filled]_

---

## Question 29
- [ ] Done

**How do you implement BLEU/ROUGE evaluation for zero-shot and few-shot text generation?**

**Answer:** _[To be filled]_

---

## Question 30
- [ ] Done

**What are the considerations for BLEU/ROUGE evaluation in low-resource language scenarios?**

**Answer:** _[To be filled]_

---

## Question 31
- [ ] Done

**How do you handle evaluation for text generation with controllable attributes or constraints?**

**Answer:** _[To be filled]_

---

## Question 32
- [ ] Done

**When should you use BLEU/ROUGE for curriculum learning and training data ordering?**

**Answer:** _[To be filled]_

---

## Question 33
- [ ] Done

**How do you implement BLEU/ROUGE evaluation for multi-turn conversation generation?**

**Answer:** _[To be filled]_

---

## Question 34
- [ ] Done

**What strategies help you balance BLEU/ROUGE optimization with other quality aspects like fluency?**

**Answer:** _[To be filled]_

---

## Question 35
- [ ] Done

**How do you handle BLEU/ROUGE evaluation for text generation with factual accuracy requirements?**

**Answer:** _[To be filled]_

---

## Question 36
- [ ] Done

**When would you use BLEU/ROUGE for active learning and data annotation prioritization?**

**Answer:** _[To be filled]_

---

## Question 37
- [ ] Done

**How do you implement BLEU/ROUGE evaluation for ensemble text generation methods?**

**Answer:** _[To be filled]_

---

## Question 38
- [ ] Done

**What techniques help you identify systematic biases in BLEU/ROUGE evaluation results?**

**Answer:** _[To be filled]_

---

## Question 39
- [ ] Done

**How do you handle evaluation for text generation tasks with multiple valid output formats?**

**Answer:** _[To be filled]_

---

## Question 40
- [ ] Done

**When should you use BLEU/ROUGE for transfer learning evaluation across different domains?**

**Answer:** _[To be filled]_

---

## Question 41
- [ ] Done

**How do you implement BLEU/ROUGE evaluation for continual learning in text generation?**

**Answer:** _[To be filled]_

---

## Question 42
- [ ] Done

**What are the best practices for reporting BLEU/ROUGE scores in research publications?**

**Answer:** _[To be filled]_

---

## Question 43
- [ ] Done

**How do you handle BLEU/ROUGE evaluation for text generation with structured outputs?**

**Answer:** _[To be filled]_

---

## Question 44
- [ ] Done

**When would you use BLEU/ROUGE derivatives for gradient-based optimization of generation models?**

**Answer:** _[To be filled]_

---

## Question 45
- [ ] Done

**How do you implement efficient BLEU/ROUGE calculation for large-scale evaluation scenarios?**

**Answer:** _[To be filled]_

---

## Question 46
- [ ] Done

**What strategies help you validate BLEU/ROUGE improvements through human evaluation studies?**

**Answer:** _[To be filled]_

---

## Question 47
- [ ] Done

**How do you handle BLEU/ROUGE evaluation for text generation with privacy constraints?**

**Answer:** _[To be filled]_

---

## Question 48
- [ ] Done

**When should you use BLEU/ROUGE for anomaly detection in text generation quality?**

**Answer:** _[To be filled]_

---

## Question 49
- [ ] Done

**How do you implement BLEU/ROUGE evaluation for federated learning scenarios?**

**Answer:** _[To be filled]_

---

## Question 50
- [ ] Done

**What techniques help you combine BLEU/ROUGE with domain-specific evaluation criteria for comprehensive assessment?**

**Answer:** _[To be filled]_

---



---

# --- PSNR & SSIM (Computer Vision) Questions (from source .txt) ---

## Question 1
- [ ] Done

**How do you choose between PSNR and SSIM based on your specific computer vision evaluation requirements?**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**What are the limitations of PSNR for evaluating perceptual image quality, and when should you prioritize SSIM?**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**How do you handle PSNR/SSIM evaluation for different image formats, bit depths, and color spaces?**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**When would you use MS-SSIM (Multi-Scale SSIM) versus traditional SSIM for multiresolution image analysis?**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**How do you implement meaningful PSNR/SSIM evaluation for images with different dynamic ranges?**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**What techniques help you interpret PSNR and SSIM values in the context of human visual perception?**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**How do you handle PSNR/SSIM evaluation for image restoration tasks with varying degradation types?**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**When should you use perceptual metrics like LPIPS alongside PSNR/SSIM for comprehensive evaluation?**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**How do you implement PSNR and SSIM calculation for video quality assessment with temporal considerations?**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**What strategies help you correlate PSNR/SSIM scores with subjective human quality ratings?**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**How do you handle PSNR/SSIM evaluation for super-resolution tasks with different upscaling factors?**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**When would you use local PSNR/SSIM analysis versus global metrics for spatially varying quality?**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**How do you implement statistical significance testing for PSNR/SSIM differences between methods?**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**What techniques help you optimize image processing algorithms using PSNR/SSIM as objective functions?**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**How do you handle evaluation when processed images contain improvements not captured by reference images?**

**Answer:** _[To be filled]_

---

## Question 16
- [ ] Done

**When should you use PSNR-HVS or other human visual system-weighted variants of PSNR?**

**Answer:** _[To be filled]_

---

## Question 17
- [ ] Done

**How do you implement PSNR/SSIM evaluation for streaming or real-time image processing applications?**

**Answer:** _[To be filled]_

---

## Question 18
- [ ] Done

**What are the best practices for image preprocessing before PSNR/SSIM calculation?**

**Answer:** _[To be filled]_

---

## Question 19
- [ ] Done

**How do you handle PSNR/SSIM evaluation for domain-specific images like medical or satellite imagery?**

**Answer:** _[To be filled]_

---

## Question 20
- [ ] Done

**When would you use variants like VIF, FSIM, or GSM instead of traditional PSNR/SSIM?**

**Answer:** _[To be filled]_

---

## Question 21
- [ ] Done

**How do you implement PSNR/SSIM evaluation for image enhancement tasks without ground truth references?**

**Answer:** _[To be filled]_

---

## Question 22
- [ ] Done

**What strategies help you balance PSNR/SSIM optimization with computational efficiency constraints?**

**Answer:** _[To be filled]_

---

## Question 23
- [ ] Done

**How do you handle evaluation for image generation tasks where multiple valid outputs exist?**

**Answer:** _[To be filled]_

---

## Question 24
- [ ] Done

**When should you use PSNR/SSIM for model selection versus other evaluation criteria?**

**Answer:** _[To be filled]_

---

## Question 25
- [ ] Done

**How do you implement PSNR/SSIM evaluation for multi-channel or hyperspectral image processing?**

**Answer:** _[To be filled]_

---

## Question 26
- [ ] Done

**What techniques help you assess the stability and reliability of PSNR/SSIM measurements?**

**Answer:** _[To be filled]_

---

## Question 27
- [ ] Done

**How do you handle PSNR/SSIM evaluation for image processing with controllable quality parameters?**

**Answer:** _[To be filled]_

---

## Question 28
- [ ] Done

**When would you use masked or region-of-interest specific PSNR/SSIM calculations?**

**Answer:** _[To be filled]_

---

## Question 29
- [ ] Done

**How do you implement PSNR/SSIM evaluation for zero-shot and few-shot image processing methods?**

**Answer:** _[To be filled]_

---

## Question 30
- [ ] Done

**What are the considerations for PSNR/SSIM evaluation in resource-constrained environments?**

**Answer:** _[To be filled]_

---

## Question 31
- [ ] Done

**How do you handle evaluation for image processing with artifacts that aren't captured by traditional metrics?**

**Answer:** _[To be filled]_

---

## Question 32
- [ ] Done

**When should you use PSNR/SSIM for curriculum learning and training data ordering?**

**Answer:** _[To be filled]_

---

## Question 33
- [ ] Done

**How do you implement PSNR/SSIM evaluation for multi-frame image processing like denoising or super-resolution?**

**Answer:** _[To be filled]_

---

## Question 34
- [ ] Done

**What strategies help you identify systematic biases in PSNR/SSIM evaluation results?**

**Answer:** _[To be filled]_

---

## Question 35
- [ ] Done

**How do you handle PSNR/SSIM evaluation for image processing tasks with temporal consistency requirements?**

**Answer:** _[To be filled]_

---

## Question 36
- [ ] Done

**When would you use PSNR/SSIM for active learning and data annotation prioritization?**

**Answer:** _[To be filled]_

---

## Question 37
- [ ] Done

**How do you implement PSNR/SSIM evaluation for ensemble image processing methods?**

**Answer:** _[To be filled]_

---

## Question 38
- [ ] Done

**What techniques help you validate PSNR/SSIM improvements through human evaluation studies?**

**Answer:** _[To be filled]_

---

## Question 39
- [ ] Done

**How do you handle evaluation for image processing tasks with multiple quality aspects beyond fidelity?**

**Answer:** _[To be filled]_

---

## Question 40
- [ ] Done

**When should you use PSNR/SSIM for transfer learning evaluation across different image domains?**

**Answer:** _[To be filled]_

---

## Question 41
- [ ] Done

**How do you implement PSNR/SSIM evaluation for continual learning in image processing?**

**Answer:** _[To be filled]_

---

## Question 42
- [ ] Done

**What are the best practices for reporting PSNR/SSIM scores in research publications?**

**Answer:** _[To be filled]_

---

## Question 43
- [ ] Done

**How do you handle PSNR/SSIM evaluation for image processing with privacy-preserving constraints?**

**Answer:** _[To be filled]_

---

## Question 44
- [ ] Done

**When would you use PSNR/SSIM derivatives for gradient-based optimization of processing algorithms?**

**Answer:** _[To be filled]_

---

## Question 45
- [ ] Done

**How do you implement efficient PSNR/SSIM calculation for large-scale evaluation scenarios?**

**Answer:** _[To be filled]_

---

## Question 46
- [ ] Done

**What strategies help you combine PSNR/SSIM with task-specific evaluation criteria?**

**Answer:** _[To be filled]_

---

## Question 47
- [ ] Done

**How do you handle PSNR/SSIM evaluation for image processing in federated learning scenarios?**

**Answer:** _[To be filled]_

---

## Question 48
- [ ] Done

**When should you use PSNR/SSIM for anomaly detection in image processing quality?**

**Answer:** _[To be filled]_

---

## Question 49
- [ ] Done

**How do you implement PSNR/SSIM evaluation for image processing with uncertainty quantification?**

**Answer:** _[To be filled]_

---

## Question 50
- [ ] Done

**What techniques help you address the limitations of PSNR/SSIM in modern deep learningbased image processing?**

**Answer:** _[To be filled]_

---



---

# --- MOTA, MOTP & mAP Questions (from source .txt) ---

## Question 1
- [ ] Done

**How do you choose between MOTA, MOTP, and mAP based on your specific object detection and tracking evaluation needs?**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**What are the key differences in how MOTA handles identity switches versus missed detections in multi-object tracking?**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**How do you implement mAP calculation for object detection with varying IoU thresholds and class imbalances?**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**When would you prioritize MOTP (tracking precision) over MOTA (tracking accuracy) in surveillance applications?**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**How do you handle mAP evaluation for small objects versus large objects in detection systems?**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**What techniques help you interpret MOTA scores in the context of real-world tracking performance?**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**How do you implement MOTA/MOTP evaluation for multi-camera tracking systems with overlapping fields of view?**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**When should you use AP@0.5 versus AP@0.75 or AP@0.50.95] for different detection applications?**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**How do you handle evaluation when ground truth annotations have temporal or spatial uncertainty?**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**What strategies help you balance detection accuracy (mAP) with tracking consistency (MOTA) in system design?**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**How do you implement MOTA evaluation for tracking systems with track fragmentation and merging?**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**When would you use class-specific mAP versus overall mAP for multi-class detection evaluation?**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**How do you handle MOTP calculation when tracking predictions have varying confidence levels?**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**What techniques help you assess the impact of false positives versus false negatives on MOTA scores?**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**How do you implement mAP evaluation for detection systems with hierarchical or finegrained class structures?**

**Answer:** _[To be filled]_

---

## Question 16
- [ ] Done

**When should you use CLEAR MOT metrics versus other tracking evaluation frameworks?**

**Answer:** _[To be filled]_

---

## Question 17
- [ ] Done

**How do you handle evaluation for detection and tracking in crowded scenes with heavy occlusion?**

**Answer:** _[To be filled]_

---

## Question 18
- [ ] Done

**What are the best practices for annotation quality control that affect MOTA/MOTP/mAP reliability?**

**Answer:** _[To be filled]_

---

## Question 19
- [ ] Done

**How do you implement real-time MOTA/MOTP monitoring for deployed tracking systems?**

**Answer:** _[To be filled]_

---

## Question 20
- [ ] Done

**When would you use track-based evaluation metrics versus frame-based detection metrics?**

**Answer:** _[To be filled]_

---

## Question 21
- [ ] Done

**How do you handle mAP calculation for detection systems with temporal consistency requirements?**

**Answer:** _[To be filled]_

---

## Question 22
- [ ] Done

**What strategies help you optimize tracking algorithms using MOTA as an objective function?**

**Answer:** _[To be filled]_

---

## Question 23
- [ ] Done

**How do you implement MOTA evaluation for online versus offline tracking algorithms?**

**Answer:** _[To be filled]_

---

## Question 24
- [ ] Done

**When should you use higher-order MOT metrics like MOSAL or track quality measures?**

**Answer:** _[To be filled]_

---

## Question 25
- [ ] Done

**How do you handle evaluation for detection systems with varying object sizes and aspect ratios?**

**Answer:** _[To be filled]_

---

## Question 26
- [ ] Done

**What techniques help you identify systematic biases in MOTA/MOTP/mAP evaluation results?**

**Answer:** _[To be filled]_

---

## Question 27
- [ ] Done

**How do you implement evaluation for tracking systems with object re-identification requirements?**

**Answer:** _[To be filled]_

---

## Question 28
- [ ] Done

**When would you use soft assignment versus hard assignment in MOTA calculation?**

**Answer:** _[To be filled]_

---

## Question 29
- [ ] Done

**How do you handle mAP evaluation for few-shot or zero-shot object detection scenarios?**

**Answer:** _[To be filled]_

---

## Question 30
- [ ] Done

**What are the considerations for MOTA/MOTP evaluation in privacy-sensitive surveillance applications?**

**Answer:** _[To be filled]_

---

## Question 31
- [ ] Done

**How do you implement evaluation for detection and tracking with partial or incomplete annotations?**

**Answer:** _[To be filled]_

---

## Question 32
- [ ] Done

**When should you use MOTA/MOTP for model selection versus other evaluation criteria?**

**Answer:** _[To be filled]_

---

## Question 33
- [ ] Done

**How do you handle evaluation for multi-modal tracking systems combining vision with other sensors?**

**Answer:** _[To be filled]_

---

## Question 34
- [ ] Done

**What strategies help you assess tracking performance across different object motion patterns?**

**Answer:** _[To be filled]_

---

## Question 35
- [ ] Done

**How do you implement MOTA evaluation for distributed tracking systems with communication delays?**

**Answer:** _[To be filled]_

---

## Question 36
- [ ] Done

**When would you use interpolation or extrapolation in MOTA calculation for missing detections?**

**Answer:** _[To be filled]_

---

## Question 37
- [ ] Done

**How do you handle mAP evaluation for detection systems with domain adaptation requirements?**

**Answer:** _[To be filled]_

---

## Question 38
- [ ] Done

**What techniques help you validate MOTA/MOTP improvements through human evaluation studies?**

**Answer:** _[To be filled]_

---

## Question 39
- [ ] Done

**How do you implement evaluation for tracking systems with object lifecycle management?**

**Answer:** _[To be filled]_

---

## Question 40
- [ ] Done

**When should you use MOTA/MOTP for active learning and annotation prioritization?**

**Answer:** _[To be filled]_

---

## Question 41
- [ ] Done

**How do you handle evaluation for detection and tracking in challenging environmental conditions?**

**Answer:** _[To be filled]_

---

## Question 42
- [ ] Done

**What are the best practices for reporting MOTA/MOTP/mAP scores in research publications?**

**Answer:** _[To be filled]_

---

## Question 43
- [ ] Done

**How do you implement evaluation for tracking systems with real-time processing constraints?**

**Answer:** _[To be filled]_

---

## Question 44
- [ ] Done

**When would you use trajectory-based metrics alongside frame-based MOTA/MOTP evaluation?**

**Answer:** _[To be filled]_

---

## Question 45
- [ ] Done

**How do you handle evaluation for detection systems with incremental learning capabilities?**

**Answer:** _[To be filled]_

---

## Question 46
- [ ] Done

**What strategies help you combine MOTA/MOTP/mAP with task-specific performance requirements?**

**Answer:** _[To be filled]_

---

## Question 47
- [ ] Done

**How do you implement evaluation for federated detection and tracking systems?**

**Answer:** _[To be filled]_

---

## Question 48
- [ ] Done

**When should you use MOTA/MOTP for curriculum learning in tracking algorithm development?**

**Answer:** _[To be filled]_

---

## Question 49
- [ ] Done

**How do you handle evaluation for detection and tracking with adversarial robustness requirements?**

**Answer:** _[To be filled]_

---

## Question 50
- [ ] Done

**What techniques help you address the limitations of traditional metrics in modern deep learning-based tracking systems? ⚙️ MLOps / LLMOps / Deployment**

**Answer:** _[To be filled]_

---

