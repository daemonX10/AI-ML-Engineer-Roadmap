# Pca Interview Questions - Scenario_Based Questions

## Question 1
- [ ] Done

**Discuss how PCA can suffer from outlier sensitivity and ways to address it.**

### Answer

**Why PCA is Sensitive to Outliers:**

- PCA maximizes variance (based on squared distances)
- Outliers have extreme values → disproportionate variance
- Single outlier can dominate PC1 direction
- Covariance/mean affected by outliers

**Example:**
100 points clustered at (0,0), one outlier at (100, 100)
- PC1 will point toward outlier instead of cluster structure

**Solutions:**

| Approach | Method | When to Use |
|----------|--------|-------------|
| **Pre-removal** | IQR, Z-score filtering | Clear outliers, small % of data |
| **Robust scaling** | Median/MAD instead of mean/std | Moderate contamination |
| **Robust PCA** | L + S decomposition | Gross errors, structured outliers |
| **Winsorization** | Cap extreme values at percentiles | Preserve sample size |
| **Robust covariance** | Minimum Covariance Determinant | Methodologically rigorous |

**Implementation Strategy:**
```python
# Option 1: Remove outliers first
X_clean = remove_outliers_zscore(X, threshold=3)
pca.fit(X_clean)

# Option 2: Robust scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()  # Uses median and IQR
X_scaled = scaler.fit_transform(X)
pca.fit(X_scaled)
```

**Interview Tip:**
Always visualize data before PCA. If outliers exist, address them first—PCA results on contaminated data are unreliable.

---

## Question 2
- [ ] Done

**How would you use PCA for data compression in a real-time streaming application?**

### Answer

**Scenario Logic:**

**Challenge:**
- Data streams continuously (sensor data, network packets)
- Cannot store full data
- Need compressed representation in real-time

**Solution Architecture:**

**1. Initial Training Phase:**
```python
# Train on historical data batch
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=k)
ipca.fit(historical_data)
```

**2. Real-Time Compression:**
```python
# As new data arrives:
while streaming:
    batch = get_next_batch()
    
    # Compress (transform to PC space)
    compressed = ipca.transform(batch)
    
    # Store compressed (k dimensions instead of p)
    store(compressed)
```

**3. Periodic Model Update:**
```python
# Update PCA periodically with new data
ipca.partial_fit(new_batch)
```

**4. Reconstruction (if needed):**
```python
reconstructed = ipca.inverse_transform(compressed)
```

**Key Considerations:**

| Aspect | Approach |
|--------|----------|
| **Compression ratio** | p/k (e.g., 1000→50 = 20x) |
| **Quality trade-off** | More k = better quality, less compression |
| **Concept drift** | Periodic retraining or sliding window |
| **Latency** | Transform operation is fast (matrix multiplication) |

**Use Cases:**
- IoT sensor data compression
- Network traffic monitoring
- Video stream summarization
- Real-time anomaly detection

---

## Question 3
- [ ] Done

**How would you decide whether to use PCA or a classification algorithm for a given dataset?**

### Answer

**Key Logic:**
PCA and classification algorithms serve DIFFERENT purposes. This question tests understanding of supervised vs unsupervised techniques.

**Decision Framework:**

| Scenario | Use PCA | Use Classification |
|----------|---------|-------------------|
| **Goal** | Reduce dimensions, find structure | Predict categories |
| **Labels available?** | Not required | Required |
| **Output** | Transformed features | Class predictions |
| **When** | Preprocessing step | Final prediction task |

**They Often Work Together:**
```
Data → Standardize → PCA → Classification Algorithm → Prediction
```

**When to Use PCA (Preprocessing):**
- High-dimensional data (curse of dimensionality)
- Multicollinearity present
- Visualization needed
- Speed up training
- Reduce overfitting

**When to Skip PCA:**
- Low dimensions already
- All features independently important
- Tree-based models (handle high-dim well)
- Interpretability critical (PCA obscures features)

**Decision Process:**
1. Define goal: Prediction → Need classifier
2. Is data high-dimensional? → Consider PCA first
3. Try with and without PCA
4. Compare classifier performance
5. Choose based on accuracy AND interpretability needs

**Interview Answer:**
"It's not either/or. PCA is preprocessing; classification is the task. I'd use PCA before classification when dimensions are high, features correlated, or training is slow. I'd validate by comparing model performance with and without PCA."

---

## Question 4
- [ ] Done

**Discuss a case where PCA helped improve model performance by reducing overfitting.**

### Answer

**Scenario:**
Medical diagnosis with 500 genetic markers, only 200 patients.

**Problem:**
- Features (500) > Samples (200)
- High risk of overfitting
- Model memorizes training data
- Poor generalization to new patients

**Without PCA:**
```
Training accuracy: 98%
Test accuracy: 55%  ← Overfitting!
```

**With PCA:**
```python
# Reduce to components explaining 95% variance
pca = PCA(n_components=0.95)  # Results in ~30 components
X_train_pca = pca.fit_transform(X_train_scaled)

# Now: 30 features, 160 training samples
# Ratio improved from 500:200 to 30:160
```

**After PCA:**
```
Training accuracy: 85%
Test accuracy: 82%  ← Better generalization!
```

**Why It Worked:**

| Factor | Before PCA | After PCA |
|--------|------------|-----------|
| Features | 500 | 30 |
| Feature/Sample ratio | 2.5 | 0.19 |
| Model complexity | Very high | Moderate |
| Noise | Included | Filtered (low-variance removed) |

**Mechanism:**
1. PCA removes redundant information (correlated features)
2. Discards low-variance components (often noise)
3. Reduces effective model complexity
4. Simpler model → better generalization

**Key Insight:**
PCA acts as implicit regularization by constraining the model to a lower-dimensional subspace.

---

## Question 5
- [ ] Done

**Give an example of how PCA might be incorrectly applied to a dataset and propose a solution.**

### Answer

**Scenario: Incorrect Application**

**Dataset:** Customer purchase data
- `total_spend`: Range $10 - $100,000
- `num_purchases`: Range 1 - 500
- `avg_item_price`: Range $5 - $200
- `customer_complained`: Binary (0 or 1)

**Mistake Made:**
```python
# WRONG: Applied PCA without standardization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)  # Raw data!
```

**What Went Wrong:**
1. `total_spend` has variance ~ billions (large range)
2. `customer_complained` has variance ~ 0.05 (binary)
3. PC1 is 99.9% aligned with `total_spend`
4. Important churn indicator (`customer_complained`) is ignored

**Solution:**
```python
# CORRECT: Standardize first
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

**Other Common Mistakes & Solutions:**

| Mistake | Problem | Solution |
|---------|---------|----------|
| Not centering | PC1 points to data center | Always center data |
| Fitting on test data | Data leakage | fit on train, transform on test |
| Applying to categorical | Meaningless results | Use MCA for categorical |
| Ignoring missing values | Algorithm fails | Impute first |
| Using too few components | Lose important info | Check cumulative variance |

**Interview Tip:**
"PCA preprocessing is as important as PCA itself. Standardization ensures all features contribute fairly, not just those with large scales."

---

## Question 6
- [ ] Done

**Discuss how you would ensure the robustness of PCA results against variations in the dataset.**

### Answer

**Robustness Concerns:**
- Small data changes → Large PC changes?
- Are components stable across samples?
- Will results generalize?

**Strategies for Robust PCA:**

**1. Cross-Validation of Components:**
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
components_list = []

for train_idx, val_idx in kf.split(X):
    pca = PCA(n_components=k)
    pca.fit(X_scaled[train_idx])
    components_list.append(pca.components_)

# Check consistency across folds
# Similar components → Robust
```

**2. Bootstrap Confidence Intervals:**
```python
n_bootstrap = 100
loadings = []

for _ in range(n_bootstrap):
    idx = np.random.choice(len(X), len(X), replace=True)
    pca = PCA(n_components=k)
    pca.fit(X_scaled[idx])
    loadings.append(pca.components_)

# Compute confidence intervals for loadings
```

**3. Sensitivity Analysis:**
- Artificially perturb data slightly
- Re-run PCA
- Check if conclusions change

**4. Outlier Handling:**
- Use Robust PCA or pre-filter outliers
- Compare results with/without suspected outliers

**5. Sufficient Sample Size:**
- Rule of thumb: n > 5p (samples > 5× features)
- More samples → More stable estimates

**Validation Checklist:**

| Check | Method |
|-------|--------|
| Component stability | Cross-validation |
| Loading significance | Bootstrap CI |
| Outlier impact | Compare with/without |
| Explained variance consistency | Multiple splits |

**Interview Point:**
"Robust PCA results should be reproducible across different subsamples. I'd validate using cross-validation and bootstrap to ensure components aren't artifacts of specific data points."

---

## Question 7
- [ ] Done

**Explain how you would apply PCA in a stock market data analysis situation.**

### Answer

**Definition:**
PCA on stock return data identifies independent factors driving market movements. The first component typically captures overall market movement, while subsequent components reveal sector or style factors.

**Step-by-Step Application:**

1. **Data Preparation:**
   - Select stocks (e.g., S&P 500)
   - Convert prices to returns (daily/weekly)
   - Matrix: rows = days, columns = stocks
   - Standardize returns per stock

2. **Apply PCA:**
   - Compute principal components of returns matrix
   - Each PC = orthogonal source of market variation

3. **Interpretation:**
   - **PC1 (Market Factor)**: Positive loadings on all stocks → overall market movement
   - **PC2+**: Sector/style factors (e.g., Tech vs Utilities, Growth vs Value)

**Example Interpretation:**
| Component | Interpretation | Loadings Pattern |
|-----------|----------------|------------------|
| PC1 | Market movement | All stocks positive |
| PC2 | Growth vs Value | Tech +, Utilities - |
| PC3 | Small vs Large cap | Small cap +, Large cap - |

**Use Cases in Finance:**
- **Factor investing**: Build strategies around principal components
- **Risk management**: Hedge exposure to specific factors
- **Portfolio diversification**: Ensure exposure across different PCs
- **Anomaly detection**: Extreme PC scores = unusual market behavior

---

## Question 8
- [ ] Done

**Describe a scenario where using PCA might be detrimental to the performance of a machine learning model.**

### Answer

**Scenario: Low-Variance, High-Predictive-Power Features**

**Setup:**
Predicting customer churn with features:
- `total_spend`: Range $10-$10,000 (high variance)
- `monthly_usage`: Range 1-500 GB (high variance)  
- `complaints_filed`: Binary 0/1 (very low variance, <5% have complaints)

**The Problem:**
- Most customers never filed complaints → low variance
- BUT filing a complaint strongly predicts churn → high predictive power
- PCA focuses on high-variance features (spend, usage)
- `complaints_filed` contributes little to top components
- When reducing dimensions, complaint information is lost

**The Outcome:**
- Model trained on PCA-transformed data performs worse
- Critical predictor was discarded by unsupervised PCA
- PCA equated low variance with low importance (incorrectly!)

**Key Lesson:**
PCA is unsupervised—it has NO knowledge of the target variable. High variance ≠ High predictive power.

**When PCA Can Hurt:**
- Low-variance features are strong predictors
- Important information exists in small variations
- Classes differ in subtle ways (not variance directions)

**Alternative:**
Use supervised feature selection (feature importance from Random Forest, RFE) when target-feature relationship matters.

---
