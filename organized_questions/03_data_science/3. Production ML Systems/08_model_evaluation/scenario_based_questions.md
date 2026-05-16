# Model Evaluation Interview Questions - Scenario_Based Questions

## Question 1
- [ ] Done

**How would you evaluate a time-series forecasting model?**

**Answer:**

### Key Considerations
- Use time-based train/test split (no shuffling)
- Forward-chaining or expanding window CV
- Never leak future data into training

### Metrics
- MAE, RMSE, MAPE for point forecasts
- Prediction intervals for uncertainty
- Directional accuracy for trend

### Implementation
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

### Interview Tip
Standard CV is invalid for time series; use temporal splits.

---

## Question 2
- [ ] Done

**How would you validate a natural language processing model?**

**Answer:**

### Considerations
- Train/test split by document, not sentence
- Out-of-domain evaluation
- Robustness testing (adversarial examples)

### Metrics by Task
- Classification: F1, accuracy
- NER: Entity-level F1
- Translation: BLEU, ROUGE
- Generation: Perplexity, human evaluation

### Interview Tip
Always validate on data from different distribution.

---

## Question 3
- [ ] Done

**How would you approach the evaluation of a fraud detection algorithm with highly imbalanced classes?**

**Answer:**

### Strategy
1. Don't use accuracy (misleading)
2. Focus on precision-recall trade-off
3. Use PR-AUC over ROC-AUC
4. Consider business costs

### Metrics
- Precision at fixed recall
- F1, F2 (if recall more important)
- Cost-sensitive metrics

### Practical Approach
```python
from sklearn.metrics import precision_recall_curve, average_precision_score
```

### Interview Tip
Tune threshold based on business cost of FP vs FN.

---

## Question 4
- [ ] Done

**Discuss how you would evaluate a computer vision model used for self-driving cars.**

**Answer:**

### Multi-level Evaluation
1. **Object Detection**: mAP, IoU
2. **Segmentation**: Dice, pixel accuracy
3. **Safety**: Edge cases, adversarial robustness
4. **Real-world**: Closed-course testing

### Special Considerations
- Test on various conditions (weather, lighting)
- Failure mode analysis
- Human-in-the-loop evaluation
- Latency constraints

### Interview Tip
Safety-critical systems need extensive edge case testing.

---

## Question 5
- [ ] Done

**Propose a framework for continuous evaluation of an online learning system.**

**Answer:**

### Framework Components
1. **Real-time monitoring**: Prediction distributions, latency
2. **Drift detection**: Statistical tests, PSI
3. **Performance tracking**: Rolling window metrics
4. **Alerting**: Thresholds for intervention

### Implementation
```python
# Monitor prediction drift
from scipy.stats import ks_2samp
stat, p_value = ks_2samp(old_predictions, new_predictions)
```

### Interview Tip
Set up automated retraining triggers.

---

## Question 6
- [ ] Done

**How would you assess the business impact of precision and recall in a customer churn prediction model?**

**Answer:**

### Cost Analysis
- **False Positive (predict churn, not churn)**: Cost of unnecessary retention offer
- **False Negative (miss churn)**: Lost customer lifetime value

### Business Translation
- High precision → fewer wasted retention efforts
- High recall → catch more churners

### Decision Framework
```
Optimal threshold = argmax[Recall × CLV_saved - FP × Offer_cost]
```

### Interview Tip
Always translate metrics to business dollars.

---

## Question 7
- [ ] Done

**Discuss the role of model explainability in model evaluation.**

**Answer:**

### Importance
- Build trust with stakeholders
- Identify model biases
- Debug unexpected behavior
- Regulatory compliance

### Techniques
- SHAP values
- LIME
- Feature importance
- Partial dependence plots

### Evaluation Aspects
- Are explanations consistent?
- Do they align with domain knowledge?
- Are they actionable?

### Interview Tip
Explainability is required for high-stakes decisions.

---

