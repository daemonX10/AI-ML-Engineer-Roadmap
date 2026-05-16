# Linear Regression Interview Questions - Scenario-Based Questions

## Question 1: Discuss how linear regression can be used for sales forecasting
- [ ] Done

### Answer

**Definition:**
Linear regression predicts future sales by modeling the relationship between sales (target) and predictors like time, price, marketing spend, and seasonality.

**Approach Framework:**

**Step 1: Feature Engineering**
- **Time features:** trend, month, day_of_week, is_holiday
- **Lag features:** sales_last_week, sales_last_month
- **External:** price, promotion_flag, competitor_price, weather

**Step 2: Model Structure**
$$Sales = \beta_0 + \beta_1 \cdot trend + \beta_2 \cdot price + \beta_3 \cdot promo + \beta_4 \cdot season + \epsilon$$

**Step 3: Handle Non-linearities**
- Log-transform sales (if skewed)
- Add polynomial terms for diminishing returns
- Use adstock for marketing (carryover effect)

**Key Considerations:**

| Challenge | Solution |
|-----------|----------|
| Seasonality | Add month/quarter dummies |
| Trend | Include time index |
| Autocorrelation | Add lag features |
| Diminishing returns | Log transform ad spend |

**Practical Application:**
1. Collect 2-3 years of weekly/daily sales data
2. Create features (time, external factors)
3. Split chronologically (no random split for time series)
4. Fit model, check residuals for patterns
5. Generate forecasts with confidence intervals

**Business Value:**
- Inventory planning
- Budget allocation
- Staffing decisions
- Goal setting

---

## Question 2: How would you approach building a linear regression model to predict customer churn?
- [ ] Done

### Answer

**Important Clarification:**
Customer churn is typically a **binary classification problem** (churn: yes/no), so **logistic regression** is more appropriate than linear regression. However, you can use linear regression for related continuous targets.

**Scenario A: If interviewer insists on Linear Regression**
- Predict **churn probability score** (0 to 1) or **days until churn**
- Caveat: predictions may fall outside [0,1] range

**Scenario B: Correct Approach (Logistic Regression)**

**Step 1: Define Target**
- Binary: churned (1) vs retained (0)
- Use logistic regression for probability output

**Step 2: Feature Engineering**
| Category | Features |
|----------|----------|
| Behavioral | login_frequency, feature_usage, support_tickets |
| Engagement | days_since_last_login, session_duration |
| Financial | contract_length, payment_delays, plan_type |
| Demographics | tenure, age, location |

**Step 3: Model Building**
```
Churn_Probability = σ(β₀ + β₁·recency + β₂·frequency + β₃·complaints + ...)
```
Where σ is sigmoid function

**Step 4: Interpretation**
- Positive coefficient → increases churn probability
- Use odds ratios for business interpretation

**Interview Tip:**
If asked about "linear regression for churn," politely clarify that logistic regression is more appropriate for binary outcomes, then explain why (bounded probabilities, proper classification metrics).

---

## Question 3: Propose a framework for using regression to evaluate promotional impact on sales
- [ ] Done

### Answer

**Definition:**
Marketing Mix Modeling (MMM) uses regression to quantify the ROI of different promotional channels and optimize budget allocation.

**Framework:**

**Step 1: Data Collection (Weekly/Daily)**
- Target: Sales volume or revenue
- Predictors:
  - TV_spend, Radio_spend, Digital_spend
  - Price, competitor_price
  - Seasonality (month dummies, holidays)
  - Distribution/availability

**Step 2: Feature Transformations**

| Transformation | Purpose |
|----------------|---------|
| **Log(Sales)** | Handle skewness |
| **Log(Ad_spend)** | Model diminishing returns |
| **Adstock** | Capture carryover effect |

**Adstock Formula:** $Adstock_t = Spend_t + \lambda \cdot Adstock_{t-1}$
- λ = decay rate (how quickly effect fades)

**Step 3: Model**
$$\log(Sales) = \beta_0 + \beta_1\log(TV_{adstock}) + \beta_2\log(Radio_{adstock}) + Controls + \epsilon$$

**Step 4: Interpretation**
- Coefficients = elasticities
- β₁ = 0.1 means 10% increase in TV → 1% increase in sales
- Calculate ROI: (Sales Lift × Margin) / Ad Spend

**Step 5: Optimization**
- Use coefficients to simulate different budget allocations
- Find mix that maximizes total sales given fixed budget

**Key Considerations:**
- Control for baseline sales (what happens without promotion)
- Account for seasonality to avoid false attribution
- Check for multicollinearity between channels

---

## Question 4: Discuss the role of linear regression in AI for personalized medicine
- [ ] Done

### Answer

**Definition:**
Linear regression helps model patient-specific treatment responses by relating outcomes to patient characteristics, enabling personalized dosing and treatment selection.

**Applications:**

**1. Dose-Response Modeling**
$$Response = \beta_0 + \beta_1 \cdot Dose + \beta_2 \cdot Age + \beta_3 \cdot Weight + \beta_4 \cdot (Dose \times Genetics)$$

- Interaction terms capture personalized effects
- Different patients may need different doses

**2. Treatment Effect Estimation**
- Estimate how treatment effect varies by patient subgroup
- Heterogeneous treatment effects (HTE)

**3. Biomarker Discovery**
- Use Lasso to identify which genes/proteins predict drug response
- High-dimensional feature selection

**4. Risk Prediction**
- Predict patient outcomes (survival time, readmission risk)
- Combine with clinical decision support

**Key Considerations:**

| Challenge | Approach |
|-----------|----------|
| High-dimensional genomics | Regularized regression (Lasso) |
| Interpretability for doctors | Keep model simple, use coefficients |
| Causal inference | Careful confounding control |
| Small samples | Regularization, Bayesian methods |

**Why Linear Regression (vs Black-box ML):**
- **Interpretability:** Doctors need to understand why
- **Regulatory:** FDA prefers explainable models
- **Trust:** Coefficients can be validated clinically

**Limitations:**
- May miss complex interactions (consider tree-based or neural networks for discovery, then validate with interpretable models)

---

## Question 5: How would you explain linear regression to a non-technical stakeholder?
- [ ] Done

### Answer

**Simple Explanation:**

"Linear regression finds the best straight line that describes the relationship between things we know (inputs) and something we want to predict (output)."

**Analogy:**
"Imagine plotting house prices against square footage on a graph. Linear regression draws the best-fit line through those points. Once we have that line, given any square footage, we can predict the price by reading off the line."

**Key Points to Communicate:**

**1. What it does:**
- Finds patterns in historical data
- Uses those patterns to make predictions

**2. The output:**
- A formula: *Price = Base + (Rate × Size)*
- Each factor has a "weight" showing its importance

**3. Business value:**
- "For every additional 100 sq ft, price increases by $X"
- Quantifies relationships for decision-making

**4. Limitations (honest communication):**
- Works best when relationships are roughly linear
- Past patterns may not hold in future
- Correlation ≠ causation

**Visual Aid:**
Draw a scatter plot with a line through it. Show:
- Points = actual data
- Line = model's prediction
- Gap between point and line = error

**Avoid:**
- Mathematical formulas
- Technical jargon (coefficients, residuals)
- Overcomplicating

**Key Message:**
"It's a tool that helps us understand how different factors influence an outcome, and lets us make data-driven predictions."

---

## Question 6: How would you use A/B testing to validate a linear regression model in production?
- [ ] Done

### Answer

**Definition:**
A/B testing validates that the regression model's recommendations actually improve real-world outcomes compared to the current approach.

**Framework:**

**Step 1: Define the Hypothesis**
- H₀: Model recommendations perform same as current method
- H₁: Model recommendations improve the metric

**Step 2: Design the Test**

| Group | Treatment |
|-------|-----------|
| Control (A) | Current approach (no model) |
| Treatment (B) | Use model predictions |

**Step 3: Sample Size Calculation**
- Determine minimum detectable effect
- Calculate required sample size for statistical power (typically 80%)
- Ensure random assignment

**Step 4: Implementation Example**
*Scenario: Model predicts optimal discount percentage*
- Control: Fixed 10% discount to everyone
- Treatment: Model-predicted personalized discount

**Step 5: Run the Test**
- Random 50/50 split of users
- Run for sufficient duration (account for seasonality)
- Track primary metric (e.g., conversion, revenue)

**Step 6: Analyze Results**
- Compare mean outcome between groups
- Calculate confidence interval
- Check statistical significance (p < 0.05)

**Key Considerations:**

| Issue | Solution |
|-------|----------|
| Novelty effect | Run test long enough |
| Seasonality | Run full business cycle |
| Simpson's paradox | Check subgroup results |
| Multiple testing | Adjust p-values |

**Success Criteria:**
- Statistically significant improvement
- Practically meaningful lift (business impact)
- No negative secondary effects

**After Validation:**
- Gradual rollout (10% → 50% → 100%)
- Monitor for drift over time

## Question 7: Describe a situation where linear regression could be applied in the finance sector.
- [ ] Done

### Answer

**Application: Capital Asset Pricing Model (CAPM)**

**Goal:** Estimate a stock's Beta (systematic risk) and Alpha (excess return).

**The Model:**
$$R_{stock} - R_f = \alpha + \beta (R_{market} - R_f) + \epsilon$$

| Term | Meaning |
|------|---------|
| $R_{stock} - R_f$ | Stock's excess return (target) |
| $R_{market} - R_f$ | Market's excess return (feature) |
| $\beta$ | Stock's volatility relative to market |
| $\alpha$ | Stock's risk-adjusted outperformance |

**Interpretation of Beta:**

| Beta Value | Meaning |
|------------|---------|
| β = 1 | Stock moves with market |
| β > 1 | More volatile than market |
| β < 1 | Less volatile than market |
| β < 0 | Moves opposite to market |

**Implementation:**
```python
import yfinance as yf
from sklearn.linear_model import LinearRegression

# Get data
stock = yf.download('AAPL', start='2020-01-01')
market = yf.download('^GSPC', start='2020-01-01')

# Calculate excess returns
stock_returns = stock['Adj Close'].pct_change()
market_returns = market['Adj Close'].pct_change()

# Fit CAPM model
model = LinearRegression()
model.fit(market_returns.dropna().values.reshape(-1,1), 
          stock_returns.dropna().values)

print(f"Beta: {model.coef_[0]:.4f}")
print(f"Alpha: {model.intercept_:.4f}")
```

---

## Question 8: Explain how you might use regression analysis to assess the effect of marketing campaigns.
- [ ] Done

### Answer

**Application: Marketing Mix Modeling**

**Goal:** Quantify ROI of different marketing channels.

**Model Structure:**
$$\log(Sales) = \beta_0 + \beta_1\log(TV) + \beta_2\log(Radio) + \beta_3\log(Digital) + Controls + \epsilon$$

**Key Feature Engineering:**

| Feature | Why |
|---------|-----|
| **Adstock** | Captures carryover effect (ads impact lingers) |
| **Log transform** | Models diminishing returns |
| **Seasonality** | Control for seasonal patterns |
| **Price** | Control for pricing effects |

**Interpretation:**
- Coefficients represent elasticity
- β₁ = 0.1 means 10% increase in TV spend → 1% sales increase

**Business Applications:**
1. Calculate ROI per channel
2. Optimize budget allocation
3. Simulate "what-if" scenarios

```python
# Marketing Mix Model
model = LinearRegression()
model.fit(X[['log_tv_adstock', 'log_radio_adstock', 
             'log_digital_adstock', 'price', 'season']], 
          y_log_sales)

# Interpret coefficients as elasticities
for name, coef in zip(features, model.coef_):
    print(f"{name}: {coef:.4f} elasticity")
```

---

## Question 9: Describe how linear regression models could be used in predicting real estate prices.
- [ ] Done

### Answer

**Application: Hedonic Pricing Model**

**Features to Include:**

| Category | Features |
|----------|----------|
| **Size** | SquareFootage, Bedrooms, Bathrooms |
| **Location** | Neighborhood (one-hot), Distance to amenities |
| **Quality** | OverallQuality, YearBuilt, YearRemodeled |
| **Amenities** | HasGarage, HasPool, HasFireplace |

**Common Transformations:**
- Log(Price) - handles skewness
- Log(SquareFootage) - linearizes relationship
- Polynomial features for non-linear effects

**Model Choice:**
```python
from sklearn.linear_model import LassoCV

# Lasso for automatic feature selection
model = LassoCV(cv=5)
model.fit(X_train_scaled, y_log_price)

# Interpret: value contribution of each feature
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)
```

**Business Value:**
- Automated valuation models (Zillow Zestimate)
- Feature value estimation (how much is extra bathroom worth?)
- Investment analysis

---

## Question 10: Describe how you might use linear regression to optimize inventory levels in a supply chain context.
- [ ] Done

### Answer

**Application: Demand Forecasting**

**Goal:** Predict demand to optimize stock levels.

**Features:**

| Category | Examples |
|----------|----------|
| **Time** | DayOfWeek, Month, Holiday flag |
| **Lag** | Sales_last_week, Sales_last_month |
| **Pricing** | Current price, Promotion flag |
| **External** | Weather, Economic indicators |

**Model:**
$$Sales_t = \beta_0 + \beta_1 \cdot trend + \beta_2 \cdot Sales_{t-1} + \beta_3 \cdot Price + \epsilon$$

**Using Forecast for Inventory:**

```python
# 1. Get demand forecast
forecast = model.predict(X_next_week)

# 2. Calculate forecast uncertainty from residuals
forecast_std = np.std(y_train - model.predict(X_train))

# 3. Set reorder point
service_level = 0.95  # 95% service level
z_score = 1.65
safety_stock = z_score * forecast_std * np.sqrt(lead_time)

reorder_point = forecast * lead_time + safety_stock
```

**Output:**
- Point forecast for demand
- Uncertainty estimate for safety stock calculation
- Optimal reorder point

---
