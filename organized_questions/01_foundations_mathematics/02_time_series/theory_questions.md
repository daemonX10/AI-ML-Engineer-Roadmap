# Time Series Interview Questions - Theory Questions

## Question 1
- [x] Done

**What is a time series?**

**Definition:**
A time series is a stochastic process $\{Y_t\}$ indexed by time (discrete or continuous) where ordering matters and observations are typically dependent. The sampling frequency defines what $t$ represents and the forecasting horizon.

**Core Concepts:**
- **Trend (T):** Long-term level movement (deterministic or stochastic drift)
- **Seasonality (S):** Fixed-period patterns tied to calendars
- **Cyclicity (C):** Irregular, longer-horizon oscillations not tied to a fixed period
- **Noise/Residual (ε):** Short-term, unpredictable variation
- **Regime Shifts:** Structural breaks or level changes

**Mathematical Formulation:**
- Additive Model: $Y_t = T_t + S_t + C_t + \epsilon_t$
- Multiplicative Model: $Y_t = T_t \times S_t \times C_t \times \epsilon_t$
- State-Space View: $Y_t = \ell_t + s_t + \eta_t$ with latent states for level/seasonality

**Practical Notes:**
- Use the ACF to see how far back the series "remembers"; flat ACF suggests little dependence.
- A frequency view (periodogram) helps reveal hidden cycles and multiple seasonalities.
- If behavior changes over time, use rolling statistics or split the series by regime.

**Engineering Considerations:**
- Validate timestamps (time zones, daylight savings, missing intervals) and treat gaps explicitly.
- Choose aggregation levels to balance signal and noise; avoid leakage by preventing future data in features.
- If seasonal amplitude grows with level, use a log transform or multiplicative seasonality.

**Python Example:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pipeline: Create time series -> Visualize -> Observe patterns

# Step 1: Create sample time series data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
values = [100 + 0.5*i + 10*np.sin(2*np.pi*i/30) + np.random.randn()*5 
          for i in range(365)]  # trend + seasonality + noise

ts = pd.Series(values, index=dates)

# Step 2: Plot the time series
plt.figure(figsize=(12, 4))
plt.plot(ts)
plt.title('Sample Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Output: A line plot showing upward trend with monthly cycles
```

**Interview Tips:**
- State sampling frequency, forecast horizon, and whether the time index is regular or irregular.
- Call out multiple seasonalities or regime shifts early; they drive model choice.
- Mention both time-domain and frequency-domain perspectives for senior-level discussions.

---

## Question 2
- [x] Done

**In the context of time series, what is stationarity, and why is it important?**

**Definition:**
A time series is stationary if its distribution does not change over time. In practice, weak (covariance) stationarity is assumed: constant mean, variance, and autocovariance. Distinguish trend-stationary processes from difference-stationary (unit root) processes.

**Core Concepts:**
- **Strict Stationarity:** Full joint distribution is invariant to time shifts (rare in practice)
- **Weak (Covariance) Stationarity:** Constant mean/variance/autocovariance (practical definition)
- **Trend-Stationary vs Unit Root:** Detrend vs difference; a unit root means one differencing step is needed
- **Structural Breaks:** Regime changes can break stationarity even if each regime is stable

**Conditions for Weak Stationarity:**
1. $E[Y_t] = \mu$ (constant mean - no trend)
2. $Var(Y_t) = \sigma^2$ (constant variance)
3. $Cov(Y_t, Y_{t-k}) = \gamma_k$ (depends only on lag $k$, not time $t$)

**Why It's Important:**
| Reason | Explanation |
|--------|-------------|
| Model Validity | ARMA/ARIMA inference relies on stable moments and autocovariance |
| Forecast Behavior | Error variance and prediction intervals depend on stationarity |
| Spurious Regression Risk | Non-stationarity can create misleading relationships |

**How to Check Stationarity:**
1. **Visual + Rolling Stats:** Look for drift, variance changes, and breaks
2. **ADF Test (Augmented Dickey-Fuller):**
    - $H_0$: Unit root (non-stationary)
3. **KPSS Test:**
    - $H_0$: Stationary (use alongside ADF for bracketing)
4. **Seasonal Checks:** ACF spikes at seasonal lags can signal seasonal non-stationarity
5. **Break Checks:** If you suspect a regime change, test or split the series

**Remedies:**
- Detrend or difference (including seasonal differencing) only when needed
- Variance-stabilizing transforms (log, Box-Cox)
- Model time-varying variance explicitly (ARCH/GARCH)

**Python Example:**
```python
from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np

# Pipeline: Generate data -> Apply ADF test -> Interpret result

# Step 1: Sample data
data = np.cumsum(np.random.randn(100))  # Random walk (non-stationary)

# Step 2: ADF and KPSS Tests
adf_result = adfuller(data)
kpss_stat, kpss_p, _, _ = kpss(data, regression='c', nlags='auto')
print(f'ADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}')
print(f'KPSS Statistic: {kpss_stat:.4f}, p-value: {kpss_p:.4f}')

# Step 3: Interpret
if adf_result[1] < 0.05 and kpss_p > 0.05:
    print("Stationary")
else:
    print("Non-stationary - differencing needed")

# Output: ADF p-value > 0.05 and KPSS p-value < 0.05 indicate non-stationarity
```

**Interview Tips:**
- Use ADF + KPSS together; conflicting results often indicate trend-stationarity or breaks.
- Structural breaks can look like unit roots; test for breaks before over-differencing.
- Stationarity does not mean no autocorrelation; it means stable moments.

---

## Question 3
- [x] Done

**What is seasonality in time series analysis, and how do you detect it?**

**Definition:**
Seasonality refers to predictable, periodic fluctuations that occur at fixed intervals (daily, weekly, monthly, yearly). It is tied to calendar/clock effects and is distinct from irregular business cycles.

**Core Concepts:**
- **Deterministic vs Stochastic Seasonality:** Fixed seasonal pattern vs seasonality that evolves over time
- **Multiple Seasonalities:** Weekly + yearly patterns in daily data are common
- **Amplitude Effects:** Additive vs multiplicative seasonality (amplitude grows with level)
- **Calendar Effects:** Holidays, trading days, and special events

**Detection Methods:**

| Method | What to Look For |
|--------|------------------|
| Time Series Plot | Regular repeating patterns at fixed intervals |
| Seasonal Subseries Plot | Group by season (e.g., all Januaries) - similar patterns confirm seasonality |
| Box Plots by Period | Different distributions across months/days indicate seasonality |
| ACF Plot | Significant spikes at seasonal lag and multiples (lag 12, 24, 36 for monthly data) |
| Periodogram / STL | Dominant seasonal frequencies; stable seasonal component |

**Handling Seasonality:**
1. **Seasonal Differencing:** $Y'_t = Y_t - Y_{t-s}$ (where $s$ = seasonal period)
2. **Decomposition:** STL to separate trend/seasonal/residual and model the remainder
3. **Seasonal Regressors:** Fourier terms or calendar dummies
4. **Seasonal Models:** SARIMA, TBATS, Holt-Winters

**Python Example:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Pipeline: Load data -> Plot ACF -> Look for seasonal spikes

# Step 1: Create monthly data with yearly seasonality
np.random.seed(42)
months = 36  # 3 years
seasonal_pattern = [10, 12, 15, 20, 25, 30, 32, 30, 25, 18, 12, 8]  # monthly pattern
data = [seasonal_pattern[i % 12] + np.random.randn()*2 for i in range(months)]

# Step 2: Plot ACF
fig, ax = plt.subplots(figsize=(10, 4))
plot_acf(data, lags=24, ax=ax)
plt.title('ACF - Look for spikes at lag 12, 24')
plt.show()

# Output: Spikes at lag 12 and 24 indicate annual seasonality
```

**Interview Tips:**
- State all seasonal periods explicitly (e.g., weekly and yearly for daily data).
- Check whether seasonality is stable over time; if not, prefer STL/State-Space approaches.
- Calendar effects often explain residual seasonality after standard seasonal differencing.

---

## Question 4
- [x] Done

**Explain the concept of trend in time series analysis.**

**Definition:**
A trend is the underlying long-term direction of a time series - a persistent increase, decrease, or stable movement in the data's level over time.

**Core Concepts:**
- **Deterministic Trend:** Modeled directly as a function of time (trend-stationary)
- **Stochastic Trend:** Unit root with drift (difference-stationary)
- **Structural Breaks:** Trend changes at specific times (regime shifts)
- **Nonlinear Trends:** Exponential, logistic, or piecewise trends

**Methods to Identify and Handle Trend:**

| Method | Description | Use Case |
|--------|-------------|----------|
| Visual Inspection | Plot the series | Initial exploration |
| Moving Average | Rolling mean smooths out noise | Visualization |
| Regression | Fit $Y_t = \beta_0 + \beta_1 t$ | Deterministic trend modeling |
| Unit Root Tests | ADF/KPSS to separate deterministic vs stochastic trend | Model choice |
| State-Space Trend | Local level/linear trend | Time-varying trends |
| Differencing | $Y'_t = Y_t - Y_{t-1}$ | Remove stochastic trend |

**Mathematical Formulation:**
- Linear trend: $Y_t = \beta_0 + \beta_1 t + \epsilon_t$
- Quadratic trend: $Y_t = \beta_0 + \beta_1 t + \beta_2 t^2 + \epsilon_t$
- Stochastic trend (random walk with drift): $Y_t = Y_{t-1} + \delta + \epsilon_t$
- First difference removes linear trend
- Second difference removes quadratic trend

**Python Example:**
```python
import numpy as np
import pandas as pd

# Pipeline: Create trending data -> Apply differencing -> Check stationarity

# Step 1: Create data with linear trend
t = np.arange(100)
trend_data = 50 + 2*t + np.random.randn(100)*5  # Linear trend + noise

# Step 2: Remove trend via differencing
differenced = np.diff(trend_data)  # Y_t - Y_{t-1}

# Step 3: Compare
print(f"Original mean: {trend_data.mean():.2f}")  # Changes over time
print(f"Differenced mean: {differenced.mean():.2f}")  # ~2 (the slope)

# Output: Differenced series has constant mean ≈ slope of original trend
```

**Interview Tips:**
- Decide whether the trend is deterministic or stochastic; it changes the model class.
- Check for structural breaks before differencing; breaks can mimic trends.
- Over-differencing inflates variance and can induce negative lag-1 autocorrelation.

---

## Question 5
- [x] Done

**Describe the difference between white noise and a random walk in time series.**

**Definition:**
- **White Noise:** i.i.d. random variables with zero mean and constant variance; no temporal dependence
- **Random Walk:** $Y_t = Y_{t-1} + \epsilon_t$ (often with drift); unit-root process with cumulative shocks

**Mathematical Formulation:**
- White Noise: $Y_t = \epsilon_t$ where $\epsilon_t \sim N(0, \sigma^2)$
- Random Walk: $Y_t = Y_{t-1} + \epsilon_t$

**Key Differences:**

| Property | White Noise | Random Walk |
|----------|-------------|-------------|
| Equation | $Y_t = \epsilon_t$ | $Y_t = Y_{t-1} + \epsilon_t$ |
| Stationarity | Stationary | Non-stationary |
| Autocorrelation | Zero (ACF = 0 for all lags) | High, slowly decaying ACF |
| Variance | Constant ($\sigma^2$) | Grows with time ($t \cdot \sigma^2$) |
| Best Forecast | Mean (0) | Last observed value |
| Integration Order | I(0) | I(1) |
| After Differencing | Introduces correlation | Becomes white noise |

**Advanced Notes:**
**Practical Notes:**
- A random walk has no predictable direction; the best forecast is the last value plus drift.
- Forecast uncertainty grows with horizon, unlike white noise.
- Unit-root tests (ADF/KPSS) help distinguish random walks from stationary noise.

**Python Example:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Pipeline: Generate both -> Plot -> Compare ACF patterns

np.random.seed(42)
n = 200

# White Noise
white_noise = np.random.randn(n)

# Random Walk (cumulative sum of white noise)
random_walk = np.cumsum(np.random.randn(n))

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(white_noise)
axes[0].set_title('White Noise (Stationary)')

axes[1].plot(random_walk)
axes[1].set_title('Random Walk (Non-stationary)')
plt.show()

# Key observation: Random walk "wanders" far from origin; white noise stays bounded
```

**Interview Tips:**
- Distinguish random walk with drift from deterministic trend; they imply different forecasts.
- Over-differencing white noise yields negative lag-1 autocorrelation.
- Use unit-root language (I(1) vs I(0)) for senior-level discussions.

---

## Question 6
- [x] Done

**What is meant by autocorrelation, and how is it quantified in time series?**

**Definition:**
Autocorrelation measures the correlation of a time series with a lagged version of itself. It quantifies how much the value at time $t$ is related to the value at time $t-k$.

**Core Concepts:**
- **Positive Autocorrelation:** High value at $t$ → likely high value at $t+k$
- **Negative Autocorrelation:** High value at $t$ → likely low value at $t+k$
- **No Autocorrelation:** Values are independent (white noise)

**Mathematical Formulation:**
$$ACF(k) = \frac{Cov(Y_t, Y_{t-k})}{Var(Y_t)} = \frac{\gamma_k}{\gamma_0}$$

where $\gamma_k$ is the autocovariance at lag $k$.

**Estimation and Inference:**
- Sample ACF is biased for small samples; significance bands are approximate.
- For white noise, a common 95% band is $\pm 1.96/\sqrt{N}$.
- Ljung-Box Q-test checks whether a set of autocorrelations is jointly zero.
- Non-stationary data can create spurious autocorrelation patterns.

**Interpreting ACF Plot:**

| Pattern | Indicates |
|---------|-----------|
| Slow decay | Trend present (non-stationary) |
| Spikes at regular intervals | Seasonality (lag 12, 24... for monthly) |
| Sharp cutoff after lag $q$ | MA(q) process |
| Gradual exponential decay | AR process |

**Python Example:**
```python
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
import matplotlib.pyplot as plt

# Pipeline: Create AR process -> Plot ACF -> Interpret decay pattern

np.random.seed(42)
n = 200

# Create AR(1) process: Y_t = 0.7*Y_{t-1} + noise
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.7 * y[t-1] + np.random.randn()

# Plot ACF
fig, ax = plt.subplots(figsize=(10, 4))
plot_acf(y, lags=20, ax=ax)
plt.title('ACF of AR(1) Process')
plt.show()

# Output: Exponentially decaying ACF (characteristic of AR process)
# Blue bands = 95% confidence interval; outside = significant
```

**Interview Tips:**
- Use ACF for MA order hints and PACF for AR order hints, but confirm with diagnostics.
- Report confidence bands and avoid over-interpreting single spikes.
- Always assess residual ACF after fitting a model.

---

## Question 7
- [x] Done

**Explain the purpose of differencing in time series analysis.**

**Definition:**
Differencing is a transformation that computes the difference between consecutive observations to make a non-stationary series stationary. It removes trends and seasonality.

**Core Concepts:**
- **Difference Operator:** $(1 - B)^d Y_t$ where $B$ is the backshift operator
- **First-Order Differencing:** $Y'_t = Y_t - Y_{t-1}$ (removes stochastic linear trend)
- **Second-Order Differencing:** $Y''_t = Y'_t - Y'_{t-1}$ (removes quadratic trend)
- **Seasonal Differencing:** $(1 - B^s)Y_t$ for seasonal unit roots
- **Fractional Differencing:** Preserves long memory while achieving stationarity (ARFIMA)

**Why Differencing is Needed:**
1. **Achieve Stationarity:** Required for ARMA/ARIMA models
2. **Stabilize Mean:** Removes trend so series fluctuates around constant level
3. **Enable Modeling:** The "I" in ARIMA(p,d,q) - parameter $d$ is the differencing order

**Advanced Notes:**
- Over-differencing inflates variance and can introduce negative lag-1 autocorrelation.
- Differencing can destroy long-run relationships; use VECM when cointegration exists.
- Seasonal + non-seasonal differencing can be combined but should be minimized.

**Algorithm to Determine d:**
```
1. Check stationarity with ADF test
2. If p-value > 0.05 (non-stationary):
   - Apply first difference (d=1)
   - Re-test with ADF
3. If still non-stationary:
   - Apply second difference (d=2)
   - Re-test
4. Rarely need d > 2
```

**Python Example:**
```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Pipeline: Create trending data -> Difference -> Verify stationarity

# Step 1: Non-stationary data (random walk)
np.random.seed(42)
random_walk = np.cumsum(np.random.randn(100))

# Step 2: Apply first difference
differenced = np.diff(random_walk)

# Step 3: Test stationarity
def adf_test(data, name):
    result = adfuller(data)
    print(f"{name}: ADF={result[0]:.2f}, p-value={result[1]:.4f}")

adf_test(random_walk, "Original")      # p > 0.05 (non-stationary)
adf_test(differenced, "Differenced")   # p < 0.05 (stationary)

# Output: Differencing converts non-stationary random walk to stationary white noise
```

**Interview Tips:**
- Prefer the smallest $d$ that achieves stationarity; confirm with ADF/KPSS.
- If residual ACF shows strong negative lag 1, back off differencing.
- Use seasonal differencing only when a seasonal unit root is present.

---

## Question 8
- [x] Done

**What is an AR model (Autoregressive Model) in time series?**

**Definition:**
An Autoregressive (AR) model predicts the current value as a linear combination of its own past values plus a random error term. "Autoregressive" means regression of a variable against itself.

**Mathematical Formulation:**
$$Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + ... + \phi_p Y_{t-p} + \epsilon_t$$

where:
- $p$ = order (number of lags)
- $\phi_1, ..., \phi_p$ = coefficients (strength of each lag's influence)
- $c$ = constant/intercept
- $\epsilon_t$ = white noise error

**Stationarity and Estimation:**
- AR is stationary iff all roots of $1 - \phi_1 z - ... - \phi_p z^p = 0$ lie outside the unit circle (shocks die out).
- Parameters are commonly estimated via Yule-Walker, OLS, or maximum likelihood.
- Order selection uses AIC/BIC plus residual diagnostics.

**Identifying Order p (Using PACF):**
- PACF (Partial ACF) measures **direct** correlation at lag $k$ after removing effects of shorter lags
- For AR(p): PACF **cuts off sharply** after lag $p$
- ACF: Tails off gradually (exponential/sinusoidal decay)

**Python Example:**
```python
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf
import numpy as np

# Pipeline: Create AR data -> Plot PACF to find p -> Fit model

np.random.seed(42)
n = 200

# Generate AR(2) process: Y_t = 0.6*Y_{t-1} - 0.3*Y_{t-2} + noise
y = np.zeros(n)
for t in range(2, n):
    y[t] = 0.6*y[t-1] - 0.3*y[t-2] + np.random.randn()

# Step 1: Use PACF to identify order (expect cutoff after lag 2)
plot_pacf(y, lags=15)

# Step 2: Fit AR(2) model
model = AutoReg(y, lags=2).fit()
print(f"Coefficients: {model.params}")

# Output: PACF shows significant spikes at lag 1 and 2, then cuts off
```

**Algorithm to Remember:**
```
AR Model Identification:
1. Make series stationary (if needed)
2. Plot PACF
3. Count significant spikes before cutoff → that's p
4. Fit AR(p) model
```

**Interview Tips:**
- Quote the unit-circle root condition for stationarity.
- PACF cutoff is a heuristic; validate with residual checks and information criteria.
- Use AIC/BIC or cross-validation to avoid overfitting high-order AR models.

---

## Question 9
- [x] Done

**Describe a MA model (Moving Average Model) and its use in time series.**

**Definition:**
A Moving Average (MA) model predicts the current value as a linear combination of current and past **error terms** (white noise shocks), not past values. Note: This is different from the "moving average" used for smoothing.

**Mathematical Formulation:**
$$Y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q}$$

where:
- $q$ = order (number of past error terms)
- $\theta_1, ..., \theta_q$ = coefficients
- $\mu$ = mean of the series
- $\epsilon_t$ = white noise shock at time $t$

**Invertibility and Estimation:**
- MA is invertible iff roots of $1 + \theta_1 z + ... + \theta_q z^q = 0$ lie outside the unit circle.
- Invertibility gives a unique parameterization; without it, different parameters fit the same data.
- Parameters are typically estimated by MLE or the innovations algorithm.

**Identifying Order q (Using ACF):**
- For MA(q): ACF **cuts off sharply** after lag $q$
- PACF: Tails off gradually
- Reason: A shock at time $t$ only affects the series for $q$ periods

**Key Difference from AR:**

| Aspect | AR(p) | MA(q) |
|--------|-------|-------|
| Depends on | Past values $Y_{t-1}, ...$ | Past errors $\epsilon_{t-1}, ...$ |
| ACF | Tails off | Cuts off after lag q |
| PACF | Cuts off after lag p | Tails off |

**Python Example:**
```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np

# Pipeline: Create MA data -> Plot ACF to find q -> Fit model

np.random.seed(42)
n = 200

# Generate MA(2) process: Y_t = noise + 0.7*noise_{t-1} + 0.3*noise_{t-2}
errors = np.random.randn(n)
y = np.zeros(n)
for t in range(2, n):
    y[t] = errors[t] + 0.7*errors[t-1] + 0.3*errors[t-2]

# Step 1: ACF should cut off after lag 2
plot_acf(y, lags=15)

# Step 2: Fit MA(2) model using ARIMA(0,0,2)
model = ARIMA(y, order=(0, 0, 2)).fit()
print(f"MA Coefficients: {model.params}")

# Output: ACF shows spikes at lag 1 and 2, then cuts off to zero
```

**Interview Tips:**
- ACF cutoff is a heuristic; confirm with residual diagnostics and information criteria.
- Mention invertibility when discussing identifiability of MA models.
- MA is stationary by construction but can be non-invertible without constraints.

---

## Question 10
- [x] Done

**Explain the ARMA (Autoregressive Moving Average) model.**

**Definition:**
ARMA combines both AR and MA components to model stationary time series that exhibit both momentum effects (from past values) and shock effects (from past errors).

**Mathematical Formulation:**
$$Y_t = c + \underbrace{\phi_1 Y_{t-1} + ... + \phi_p Y_{t-p}}_{\text{AR part}} + \underbrace{\epsilon_t + \theta_1 \epsilon_{t-1} + ... + \theta_q \epsilon_{t-q}}_{\text{MA part}}$$

ARMA(p, q) where:
- $p$ = AR order (past values)
- $q$ = MA order (past errors)

**Why Use ARMA?**
- Pure AR or MA may not capture all patterns
- Real series often have both momentum (AR) and shock memory (MA)
- More flexible with fewer parameters

**Stationarity and Invertibility:**
- ARMA requires AR roots outside the unit circle (stationarity) and MA roots outside the unit circle (invertibility).
- Non-invertible MA components can lead to non-unique parameterizations.

**Identifying Orders (p, q):**

| Model | ACF | PACF |
|-------|-----|------|
| AR(p) | Tails off | Cuts off at lag p |
| MA(q) | Cuts off at lag q | Tails off |
| ARMA(p,q) | Tails off | Tails off |

**When both ACF and PACF tail off, ARMA is a candidate (not definitive)**

Use **AIC/BIC** to select best (p, q):
- Fit multiple models: ARMA(1,1), ARMA(1,2), ARMA(2,1), etc.
- Choose lowest AIC or BIC (balances fit vs. complexity)

**Python Example:**
```python
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Pipeline: Create ARMA data -> Try different orders -> Select by AIC

np.random.seed(42)
n = 200

# Generate ARMA(1,1): Y_t = 0.7*Y_{t-1} + noise + 0.5*noise_{t-1}
errors = np.random.randn(n)
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.7*y[t-1] + errors[t] + 0.5*errors[t-1]

# Fit different models, compare AIC
for p in [1, 2]:
    for q in [1, 2]:
        model = ARIMA(y, order=(p, 0, q)).fit()
        print(f"ARMA({p},{q}) - AIC: {model.aic:.2f}")

# Output: ARMA(1,1) should have lowest AIC (true model)
```

**Interview Tips:**
- Use information criteria and residual whiteness tests (Ljung-Box) to confirm fit.
- ACF/PACF patterns are suggestive; identification is often ambiguous in finite samples.
- ARMA is most effective on stationary series with stable variance.

---

## Question 11
- [x] Done

**How does the ARIMA (Autoregressive Integrated Moving Average) model extend the ARMA model?**

**Definition:**
ARIMA extends ARMA by adding an Integration (I) component that handles **non-stationary** data through differencing. It's the most widely used classical forecasting model.

**Mathematical Formulation:**
ARIMA(p, d, q):
- $p$ = AR order
- $d$ = differencing order (makes series stationary)
- $q$ = MA order

**Process:**
1. **Difference** the series $d$ times to get stationary series $Y'_t$
2. **Fit ARMA(p, q)** to $Y'_t$
3. **Forecast** and integrate back (reverse differencing)

Equivalently, ARIMA applies $(1 - B)^d$ to the original series and models the stationary result.

**Key Extension:**
- ARMA requires stationary data
- ARIMA can handle **trends** via differencing
- ARIMA(p, 0, q) = ARMA(p, q)

**Common Models:**

| Model | Description |
|-------|-------------|
| ARIMA(0,1,0) | Random walk |
| ARIMA(0,1,1) | Simple exponential smoothing |
| ARIMA(1,1,0) | Differenced first-order AR |
| ARIMA(1,1,1) | Common general model |

**Advanced Notes:**
- Include a drift term when differencing removes a non-zero mean.
- Seasonal ARIMA extends ARIMA with seasonal AR/MA and seasonal differencing.
- Forecast intervals widen with horizon; for $d>0$ they grow faster than stationary models.

**Box-Jenkins Method (Algorithm):**
```
1. Identification:
   - Plot series, check for trend/seasonality
   - Apply ADF test; if non-stationary, difference (set d)
   - Examine ACF/PACF of differenced series for p, q hints

2. Estimation:
   - Fit candidate models
   - Compare using AIC/BIC

3. Diagnostic:
   - Check residuals are white noise (no autocorrelation)
   - If patterns remain, adjust model
```

**Python Example:**
```python
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Pipeline: Non-stationary data -> Fit ARIMA -> Forecast

np.random.seed(42)

# Create random walk with drift (non-stationary)
n = 100
y = np.cumsum(0.5 + np.random.randn(n))  # drift = 0.5

# Fit ARIMA(1,1,1) - d=1 handles the trend
model = ARIMA(y, order=(1, 1, 1)).fit()

# Forecast next 10 steps
forecast = model.forecast(steps=10)
print(f"Next 10 forecasts: {forecast[:5]}...")

# Output: Forecasts continue the trend learned from differenced data
```

**Interview Tips:**
- Choose $d$ with unit-root tests and residual diagnostics, not just ADF alone.
- Avoid over-differencing; it can mask structure and inflate variance.
- Mention seasonal ARIMA when strong seasonality remains after differencing.

---

## Question 12
- [x] Done

**What is the role of the ACF (autocorrelation function) and PACF (partial autocorrelation function) in time series analysis?**

**Definition:**
- **ACF:** Measures **total** correlation between $Y_t$ and $Y_{t-k}$ (includes indirect effects through intermediate lags)
- **PACF:** Measures **direct** correlation between $Y_t$ and $Y_{t-k}$ (removes effects of intermediate lags)

**Role in Model Identification:**

| Model | ACF Pattern | PACF Pattern |
|-------|-------------|--------------|
| AR(p) | Tails off (decays gradually) | **Cuts off after lag p** |
| MA(q) | **Cuts off after lag q** | Tails off (decays gradually) |
| ARMA(p,q) | Tails off | Tails off |
| Non-stationary | Very slow decay | First lag close to 1 |

**Interpretation Guide:**
- **Cuts off:** Drops to zero (within blue bands) abruptly after lag $k$
- **Tails off:** Decays gradually (exponential or sinusoidal)
- **Spikes at regular intervals:** Seasonality present

**Mathematical Insight:**
- ACF(k) = Total correlation (direct + all indirect paths)
- PACF(k) = Correlation after removing linear effects of lags $1, 2, ..., k-1$

**Advanced Notes:**
- PACF can be computed via successive regressions or the Durbin-Levinson recursion.
- ACF is the inverse Fourier transform of the spectral density.
- Use ACF/PACF for residual diagnostics; remaining structure implies model misspecification.

**Python Example:**
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import numpy as np

# Pipeline: Generate data -> Plot ACF & PACF -> Identify model type

np.random.seed(42)
n = 200

# AR(2) process
y_ar = np.zeros(n)
for t in range(2, n):
    y_ar[t] = 0.6*y_ar[t-1] + 0.2*y_ar[t-2] + np.random.randn()

# Plot both
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(y_ar, lags=20, ax=axes[0], title='ACF')
plot_pacf(y_ar, lags=20, ax=axes[1], title='PACF')
plt.tight_layout()
plt.show()

# Output: ACF tails off, PACF cuts off after lag 2 → AR(2)
```

**Algorithm to Remember:**
```
Model Selection from ACF/PACF:
1. If ACF tails off, PACF cuts off at p → AR(p)
2. If ACF cuts off at q, PACF tails off → MA(q)
3. If both tail off → ARMA(p,q), use AIC/BIC to find orders
4. If ACF decays very slowly → non-stationary, difference first
```

**Interview Tips:**
- Confidence bands are approximate; treat isolated spikes cautiously.
- Use AIC/BIC alongside ACF/PACF to avoid overfitting.
- Always check residual ACF after fitting ARIMA/SARIMA.

---

## Question 13
- [x] Done

**What is Exponential Smoothing, and when would you use it in time series forecasting?**

**Definition:**
Exponential Smoothing is a family of forecasting methods where predictions are weighted averages of past observations, with weights decaying exponentially (more recent observations get higher weights).

**Core Concepts:**
- Smoothing parameter $\alpha$ (0 < $\alpha$ < 1) controls decay rate
- Higher $\alpha$ → more weight to recent data → more reactive to changes
- Lower $\alpha$ → smoother forecasts → more stable

**Types of Exponential Smoothing:**

| Method | Handles | Use When |
|--------|---------|----------|
| Simple (SES) | Level only | No trend, no seasonality |
| Holt's (Double) | Level + Trend | Trend, no seasonality |
| Holt-Winters (Triple) | Level + Trend + Seasonality | Both trend and seasonality |

**Mathematical Formulation (SES):**
$$\hat{Y}_{t+1} = \alpha Y_t + (1-\alpha) \hat{Y}_t$$

Equivalently: New forecast = $\alpha \times$ (actual) + $(1-\alpha) \times$ (previous forecast)

**When to Use:**
- Strong baseline model (simple, fast, often surprisingly accurate)
- When recent data is more important than distant past
- Automated forecasting at scale (thousands of SKUs)
- When interpretability matters (clear level/trend/seasonal components)

**State-Space (ETS) View:**
- ETS models represent exponential smoothing as a probabilistic state-space system.
- Error, trend, and seasonality can be additive or multiplicative (ETS(A,A,A), ETS(M,A,M), etc.).
- Parameters are estimated by maximum likelihood, enabling principled intervals.

**Equivalence to ARIMA:**
- SES ≈ ARIMA(0,1,1)
- Holt's ≈ ARIMA(0,2,2)

**Python Example:**
```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
import numpy as np

# Pipeline: Create data -> Fit model -> Forecast

np.random.seed(42)
n = 50
data = 100 + np.cumsum(np.random.randn(n))  # trending data

# Simple Exponential Smoothing
ses_model = SimpleExpSmoothing(data).fit(smoothing_level=0.3)
ses_forecast = ses_model.forecast(5)

# Holt's method (with trend)
holt_model = ExponentialSmoothing(data, trend='add').fit()
holt_forecast = holt_model.forecast(5)

print(f"SES forecast: {ses_forecast[0]:.2f}")
print(f"Holt forecast: {holt_forecast[0]:.2f}")

# Output: Holt's captures trend better; SES stays flat
```

**Interview Tips:**
- Mention damped trends for long-horizon stability.
- ETS often outperforms ARIMA on short series with strong seasonality.
- Use likelihood-based model selection for ETS variants, not just visual fit.

---

## Question 14
- [x] Done

**Describe the steps involved in building a time series forecasting model.**

**Definition:**
Building a time series model is an iterative process that involves understanding data, selecting appropriate models, validating performance, and deploying for production use.

**Step-by-Step Algorithm:**

```
STEP 1: PROBLEM DEFINITION
├── What to predict? (target variable)
├── Forecast horizon? (1 day, 1 week, 1 month?)
└── Business decision to support?

STEP 2: DATA AUDIT & EDA
├── Validate timestamps, frequency, gaps, time zones
├── Plot data; check trend, seasonality, outliers, breaks
└── Inspect ACF/PACF and seasonal subseries patterns

STEP 3: PREPROCESSING & FEATURES
├── Handle missingness and outliers with documented rules
├── Variance stabilization (log/Box-Cox) if needed
├── Calendar and exogenous features (holidays, promos)
└── Check stationarity; apply differencing or trend models

STEP 4: BASELINES & CANDIDATES
├── Naive and seasonal-naive baselines
├── ETS/ARIMA/SARIMA for classical patterns
├── ML/GBM or global models for large-scale forecasting
└── Consider probabilistic models for uncertainty

STEP 5: BACKTESTING (TIME-BASED)
├── Rolling-origin evaluation (no shuffling)
├── Multi-horizon metrics (MAE, RMSE, MAPE, MASE)
└── Use the same feature pipeline as production

STEP 6: MODEL SELECTION & DIAGNOSTICS
├── Compare AIC/BIC or validation scores
├── Check residuals (ACF, Ljung-Box, distribution)
└── Stress-test with scenario or holiday effects

STEP 7: FORECASTING
├── Produce point forecasts and prediction intervals
├── Validate calibration (coverage vs nominal)
└── Reconcile hierarchical forecasts if needed

STEP 8: DEPLOYMENT & MONITORING
├── Automate retraining and drift checks
├── Track performance by segment and horizon
└── Trigger alerts when accuracy degrades
```

**Python Example:**
```python
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# Pipeline: Load → Split → Train → Evaluate

# Step 1-2: Sample data (monthly sales)
np.random.seed(42)
data = 100 + 2*np.arange(60) + 10*np.sin(np.arange(60)*np.pi/6) + np.random.randn(60)*5

# Step 3: Train-test split (last 12 months for test)
train, test = data[:-12], data[-12:]

# Step 4-5: Fit ARIMA
model = ARIMA(train, order=(1, 1, 1)).fit()

# Step 6: Forecast and evaluate
forecast = model.forecast(steps=12)
mae = mean_absolute_error(test, forecast)
print(f"MAE: {mae:.2f}")

# Output: MAE score to compare with baseline
```

**Interview Tips:**
- Emphasize data leakage prevention and time-aware validation.
- Baselines are mandatory; advanced models must beat them consistently.
- Residual diagnostics are as important as headline metrics.

---

## Question 15
- [x] Done

**Explain the concept of cross-validation in the context of time series analysis.**

**Definition:**
Time series cross-validation (backtesting) estimates model performance on unseen future data while respecting temporal order. It evaluates models across multiple forecast origins and horizons without training on the future.

**Why Standard K-Fold Fails:**
- Shuffles data randomly → model trains on future, tests on past
- Leads to overly optimistic (invalid) performance estimates
- Violates the fundamental rule: only past data to predict future

**Time Series CV Techniques:**

| Method | Description | Use Case |
|--------|-------------|----------|
| **Rolling Forecast Origin** | Train on [1...k], test on k+1; Train on [1...k+1], test on k+2; ... | Most realistic, computationally expensive |
| **Expanding Window** | Training window grows, test window slides forward | General purpose |
| **Sliding Window** | Fixed-size training window slides forward | When old data is less relevant |
| **Blocked CV with Gap** | Leave a buffer between train/test to avoid leakage | Feature pipelines with lagged aggregates |

**Rolling Forecast Origin (Walk-Forward) Algorithm:**
```
For i = 1 to n_folds:
    train_end = initial_size + i - 1
    test_index = train_end + 1
    
    Train model on data[1 : train_end]
    Predict data[test_index]
    Calculate error
    
Final score = Average of all fold errors
```

**Advanced Notes:**
- Evaluate multiple horizons (1-step, 7-step, 30-step) to capture horizon-dependent error.
- Use a gap/embargo when features use rolling windows to prevent leakage.
- For hyperparameter tuning, nested backtesting avoids optimistic bias.

**Visual Representation:**
```
Fold 1: [Train Train Train] [Test]
Fold 2: [Train Train Train Train] [Test]
Fold 3: [Train Train Train Train Train] [Test]
...
```

**Python Example:**
```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Pipeline: Create data -> Apply TimeSeriesSplit -> Train/Evaluate each fold

data = np.arange(100)  # Sample data

# TimeSeriesSplit with 5 folds
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
    print(f"Fold {fold+1}:")
    print(f"  Train: {train_idx[0]}-{train_idx[-1]}")
    print(f"  Test:  {test_idx[0]}-{test_idx[-1]}")

# Output shows expanding training window, test always comes after train
```

**Full Example with Model:**
```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

np.random.seed(42)
data = np.cumsum(np.random.randn(100))

tscv = TimeSeriesSplit(n_splits=5)
errors = []

for train_idx, test_idx in tscv.split(data):
    train, test = data[train_idx], data[test_idx]
    model = ARIMA(train, order=(1,1,0)).fit()
    pred = model.forecast(len(test))
    errors.append(np.mean(np.abs(test - pred)))

print(f"Average MAE across folds: {np.mean(errors):.3f}")
```

**Interview Tips:**
- Report how many origins and horizons were evaluated.
- Keep preprocessing (scaling, imputation) inside each fold.
- Choose window strategy based on stationarity and concept drift.

---

## Question 16
- [ ] Done

**How does the ARCH (Autoregressive Conditional Heteroskedasticity) model deal with time series volatility?**

**Definition:**
ARCH models time-varying volatility by making the variance conditional on past squared errors. It captures **volatility clustering** - periods of high volatility followed by high volatility, and vice versa.

**Core Concepts:**
- Standard models (ARIMA) assume **constant variance** (homoskedasticity)
- Financial data shows **changing variance** (heteroskedasticity)
- ARCH models the conditional variance as a function of past shocks

**Mathematical Formulation:**
ARCH(q) has two equations:

1. **Mean Equation:** $Y_t = \mu + \epsilon_t$ (or ARMA for mean)

2. **Variance Equation:** $\sigma_t^2 = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \alpha_2 \epsilon_{t-2}^2 + ... + \alpha_q \epsilon_{t-q}^2$

where:
- $\sigma_t^2$ = conditional variance at time $t$
- $\epsilon_{t-1}^2$ = squared error (shock) from previous period
- Large past shock → high current variance → volatility clustering

**Diagnostics and Constraints:**
- Test for ARCH effects using Engle's LM test on residuals.
- Positivity constraints: $\alpha_0 > 0$, $\alpha_i \ge 0$.
- Weak stationarity requires $\sum_{i=1}^q \alpha_i < 1$ (finite unconditional variance).

**How ARCH Captures Volatility Clustering:**
- Large $\epsilon_{t-1}^2$ → high $\sigma_t^2$ → expect large moves at time $t$
- Small $\epsilon_{t-1}^2$ → low $\sigma_t^2$ → expect calm period
- Past volatility predicts future volatility

**Python Example:**
```python
from arch import arch_model
import numpy as np

# Pipeline: Create volatile data -> Fit ARCH -> Forecast volatility

np.random.seed(42)

# Simulate returns with volatility clustering
n = 500
returns = np.random.randn(n) * 0.01

# Fit ARCH(1) model
model = arch_model(returns, vol='ARCH', p=1)
result = model.fit(disp='off')

print(result.summary().tables[1])  # Variance parameters

# Forecast volatility
forecast = result.forecast(horizon=5)
print(f"\nForecast variance: {forecast.variance.values[-1]}")
```

**Applications:**
- **Risk Management:** Value at Risk (VaR) calculation
- **Options Pricing:** Volatility input for Black-Scholes
- **Portfolio Optimization:** Understanding risk dynamics

**Interview Tips:**
- Model the mean first (ARMA/ARIMA), then apply ARCH to residuals.
- ARCH often requires large $q$; GARCH is typically preferred.
- Use heavy-tailed innovations (Student-t) when returns exhibit fat tails.

---

## Question 17
- [ ] Done

**Describe the GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model and its application.**

**Definition:**
GARCH extends ARCH by including lagged conditional variances in addition to lagged squared errors. It's the most widely used model for volatility forecasting.

**Mathematical Formulation:**
GARCH(p, q):
$$\sigma_t^2 = \alpha_0 + \underbrace{\sum_{i=1}^{q} \alpha_i \epsilon_{t-i}^2}_{\text{ARCH terms}} + \underbrace{\sum_{j=1}^{p} \beta_j \sigma_{t-j}^2}_{\text{GARCH terms}}$$

**GARCH(1,1) - Most Common:**
$$\sigma_t^2 = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$

- $\alpha_1$ = impact of past shock
- $\beta_1$ = persistence of past variance
- $\alpha_1 + \beta_1$ = overall persistence (close to 1 means long memory)

**Why GARCH > ARCH:**

| Aspect | ARCH | GARCH |
|--------|------|-------|
| Parameters | Many needed for long memory | Parsimonious (GARCH(1,1) often enough) |
| Persistence | Requires many lags | Built-in via $\sigma_{t-1}^2$ term |
| Practice | Rarely used alone | Industry standard |

**Advanced Notes:**
**Advanced Notes:**
- Stationarity (finite variance) typically requires $\alpha_1 + \beta_1 < 1$.
- Persistence near 1 means shocks decay very slowly.
- Leverage effects are modeled with EGARCH or GJR-GARCH.
- Heavy tails are handled with Student-t or GED innovations.

**Python Example:**
```python
from arch import arch_model
import numpy as np

# Pipeline: Get returns → Fit GARCH(1,1) → Forecast volatility

np.random.seed(42)
# Simulate financial returns
returns = np.random.randn(1000) * 0.02

# Fit GARCH(1,1)
model = arch_model(returns, vol='Garch', p=1, q=1)
result = model.fit(disp='off')

# Key parameters
print("α₀ (omega):", result.params['omega'])
print("α₁ (alpha):", result.params['alpha[1]'])  
print("β₁ (beta):", result.params['beta[1]'])
print("Persistence (α+β):", result.params['alpha[1]'] + result.params['beta[1]'])

# Forecast next 5 days volatility
forecast = result.forecast(horizon=5)
print("\nVolatility forecast:", np.sqrt(forecast.variance.values[-1]))
```

**Applications:**
- **Value at Risk (VaR):** Estimate potential losses
- **Options Pricing:** Volatility surface modeling
- **Risk Management:** Portfolio risk assessment
- **Trading Strategies:** Volatility-based signals

**Interview Tips:**
- Report persistence ($\alpha + \beta$) and unconditional variance.
- Choose EGARCH/GJR when negative shocks increase volatility more than positive shocks.
- Always check standardized residuals for remaining ARCH effects.

---

## Question 18
- [ ] Done

**Explain the concepts of cointegration and error correction models in time series.**

**Definition:**
- **Cointegration:** Two or more non-stationary series are cointegrated if a linear combination of them is stationary. They share a long-run equilibrium relationship.
- **Error Correction Model (ECM):** Models short-run dynamics while accounting for long-run equilibrium.

**Core Concepts:**

**Interpretation:**
Cointegration implies shared stochastic trends. Even though each series is I(1), a specific linear combination is I(0), indicating a long-run equilibrium.

**Mathematical Formulation:**
If $Y_t$ and $X_t$ are both I(1) (one difference makes them stationary), but $Y_t - \gamma X_t$ is I(0) (stationary), then $Y_t$ and $X_t$ are cointegrated.

**Error Correction Model:**
$$\Delta Y_t = \alpha(Y_{t-1} - \gamma X_{t-1}) + \beta \Delta X_t + \epsilon_t$$

- $(Y_{t-1} - \gamma X_{t-1})$ = error correction term (deviation from equilibrium)
- $\alpha$ = speed of adjustment (should be negative)
- If $Y$ was too high → it decreases to restore equilibrium

**Engle-Granger Two-Step Method:**
```
Step 1: Test for Cointegration
   - Regress Y on X: Y_t = γX_t + u_t
   - Get residuals: û_t = Y_t - γ̂X_t
   - Apply ADF test on residuals
   - If p-value < 0.05 → residuals are stationary → series are cointegrated

Step 2: Build ECM (if cointegrated)
   - Use lagged residuals as error correction term
   - ΔY_t = α·û_{t-1} + β·ΔX_t + ε_t
```

**Example:**
Stock prices on NYSE and LSE for same company - both non-stationary, but their difference (spread) should be stationary due to arbitrage.

**Advanced Notes:**
- Johansen tests estimate cointegration rank for multivariate systems.
- VECM generalizes ECM with multiple cointegration relationships.
- Structural breaks can invalidate cointegration tests; use break-robust methods when needed.

**Python Example:**
```python
from statsmodels.tsa.stattools import coint, adfuller
import numpy as np

# Pipeline: Generate cointegrated series → Test → Interpret

np.random.seed(42)
n = 200

# X is random walk
x = np.cumsum(np.random.randn(n))

# Y is cointegrated with X (Y = 2*X + stationary noise)
y = 2*x + np.random.randn(n)

# Test for cointegration
score, pvalue, _ = coint(y, x)
print(f"Cointegration p-value: {pvalue:.4f}")

if pvalue < 0.05:
    print("Series are cointegrated")
    # Calculate spread (error correction term)
    spread = y - 2*x
    print(f"Spread ADF p-value: {adfuller(spread)[1]:.4f}")  # Should be stationary
```

**Interview Tips:**
- Always verify each series is I(1) before cointegration testing.
- Use Johansen for multivariate systems; Engle-Granger is bivariate and order-dependent.
- Interpret $\alpha$ in ECM/VECM as the speed of adjustment back to equilibrium.

---

## Question 19
- [ ] Done

**What is meant by multivariate time series analysis, and how does it differ from univariate time series analysis?**

**Definition:**
- **Univariate:** Analyzing a single time series variable
- **Multivariate:** Analyzing two or more interacting time series simultaneously

**Key Differences:**

| Aspect | Univariate | Multivariate |
|--------|------------|--------------|
| Variables | One | Two or more |
| Question | "How does Y's past predict Y's future?" | "How do X and Y together predict both futures?" |
| Information | Own past values only | Past of all variables in the system |
| Models | ARIMA, Exponential Smoothing | VAR, VECM, Multivariate GARCH |
| Complexity | Simpler | More parameters, more complex |

**Multivariate Models:**

1. **VAR (Vector Autoregression):**
   - Each variable modeled as linear function of its own lags AND lags of all other variables
   - For 2 variables:
   $$Y_t = c_1 + \phi_{11}Y_{t-1} + \phi_{12}X_{t-1} + \epsilon_{1t}$$
   $$X_t = c_2 + \phi_{21}Y_{t-1} + \phi_{22}X_{t-1} + \epsilon_{2t}$$

2. **VECM (Vector Error Correction Model):**
   - VAR for cointegrated non-stationary series
   - Includes error correction term

**Advanced Considerations:**
- VAR stability requires all eigenvalues within the unit circle.
- Use impulse response functions (IRF) and forecast error variance decomposition (FEVD) to interpret dynamics.
- High-dimensional systems often need regularization (Lasso VAR) or dynamic factor models.
- VARX includes exogenous drivers; mixed-frequency data may require state-space models.

**When to Use Multivariate:**
- Variables are related (e.g., sales and advertising spend)
- Want to improve forecasts by leveraging related series
- Need to understand dynamic relationships between variables

**Python Example:**
```python
from statsmodels.tsa.api import VAR
import numpy as np
import pandas as pd

# Pipeline: Create related series → Fit VAR → Forecast jointly

np.random.seed(42)
n = 200

# Two related series (Y depends on lagged X)
x = np.cumsum(np.random.randn(n))
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.5*y[t-1] + 0.3*x[t-1] + np.random.randn()

# Prepare data
data = pd.DataFrame({'Y': y, 'X': x})

# Fit VAR model (difference first for stationarity)
data_diff = data.diff().dropna()
model = VAR(data_diff)
result = model.fit(maxlags=2, ic='aic')

# Forecast
forecast = result.forecast(data_diff.values[-2:], steps=5)
print("Joint forecast:\n", pd.DataFrame(forecast, columns=['Y', 'X']))
```

**Interview Tips:**
- Confirm stationarity (or cointegration) before fitting VAR/VECM.
- Report IRFs/FEVD for interpretability, not just point forecasts.
- Consider dimensionality and sample size; VAR parameters grow quickly with lags.

---

## Question 20
- [ ] Done

**Explain the concept of Granger causality in time series analysis.**

**Definition:**
Granger causality tests whether past values of one time series (X) help predict another series (Y) beyond what Y's own past provides. It's about **predictive causality**, not true cause-and-effect.

**Core Concept:**
"X Granger-causes Y" if knowing past X improves prediction of Y.

**Mathematical Formulation:**

**Restricted Model (Y predicts itself):**
$$Y_t = \alpha_0 + \alpha_1 Y_{t-1} + ... + \alpha_p Y_{t-p} + \epsilon_t$$

**Unrestricted Model (Y + X):**
$$Y_t = \beta_0 + \beta_1 Y_{t-1} + ... + \beta_p Y_{t-p} + \gamma_1 X_{t-1} + ... + \gamma_p X_{t-p} + \epsilon_t$$

**Test:**
- $H_0$: $\gamma_1 = \gamma_2 = ... = \gamma_p = 0$ (X does NOT Granger-cause Y)
- $H_1$: At least one $\gamma \neq 0$ (X Granger-causes Y)
- Use F-test; if p-value < 0.05 → X Granger-causes Y

**Important Caveats:**

| Caveat | Explanation |
|--------|-------------|
| Not true causation | Rooster crowing "Granger-causes" sunrise (prediction ≠ causation) |
| Omitted variable | Z might cause both X and Y, making X appear to cause Y |
| Stationarity required | Series should be stationary for valid test |
| Bidirectional | Test both directions (X→Y and Y→X) |

**Advanced Notes:**
- Select lag length using AIC/BIC to avoid under- or over-conditioning.
- If series are cointegrated, use VECM-based Granger tests.
- Conditional Granger causality controls for other variables in a multivariate system.
- Nonlinear Granger variants exist for regime-dependent effects.

**Python Example:**
```python
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import pandas as pd

# Pipeline: Create data → Test Granger causality → Interpret

np.random.seed(42)
n = 200

# X causes Y (with lag)
x = np.random.randn(n)
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.5*y[t-1] + 0.4*x[t-1] + np.random.randn()*0.5

# Prepare data
data = pd.DataFrame({'Y': y, 'X': x})

# Test: Does X Granger-cause Y?
print("Testing: X → Y")
result = grangercausalitytests(data[['Y', 'X']], maxlag=3, verbose=True)

# Look at p-values: if < 0.05, X Granger-causes Y
```

**Interpretation:**
- Low p-value → X helps predict Y beyond Y's own history
- Check both directions: X→Y and Y→X
- Possible outcomes: X→Y, Y→X, bidirectional, or no causality

**Interview Tips:**
- Report the lag order and test type (VAR vs VECM).
- Use conditional tests when multiple related series exist.
- Treat results as predictive, not causal, and validate with domain knowledge.

---

## Question 21
- [ ] Done

**Describe how time series analysis could be used for demand forecasting in retail.**

**Definition:**
Time series analysis enables data-driven demand forecasting by capturing historical patterns (trend, seasonality) and external factors to predict future product demand, optimizing inventory management.

**Business Problem:**
- **Overstocking:** Ties up capital, storage costs, spoilage risk
- **Understocking:** Lost sales, unhappy customers
- **Goal:** Predict demand accurately to balance both

**Solution Approach:**

**Step 1: Data Collection**
- Historical sales data (2-3 years)
- External factors: promotions, holidays, prices, competitor prices

**Step 2: Exploratory Analysis**
- Plot data to identify trend, seasonality, outliers
- Weekly/annual patterns common in retail

**Step 3: Model Selection - SARIMAX**
- **S:** Seasonal patterns (weekly peaks, holiday spikes)
- **ARI:** Handle trends via differencing
- **MA:** Capture short-term shock effects
- **X:** External variables (promotions, holidays)

**Step 4: Business Application**
- Point forecast → baseline order quantity
- Prediction intervals → safety stock decisions
- What-if analysis → "If we run a promotion, how much extra demand?"

**Advanced Considerations:**
- **Hierarchical Forecasting:** Reconcile SKU/store/category forecasts for coherence.
- **Intermittent Demand:** Use Croston/TSB or zero-inflated models for sparse items.
- **Global vs Local Models:** Global models share strength across SKUs to reduce cold-start issues.
- **Probabilistic Forecasts:** Quantiles or full distributions support safety stock decisions.

**Python Example:**
```python
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Pipeline: Prepare data → Fit SARIMAX → Forecast with exogenous variables

np.random.seed(42)

# Simulate 2 years of weekly sales with seasonality + promotion effect
weeks = 104
seasonal = 10 * np.sin(2*np.pi*np.arange(weeks)/52)  # yearly cycle
trend = 0.1 * np.arange(weeks)
promotion = np.random.choice([0, 1], weeks, p=[0.9, 0.1])  # 10% weeks have promo
sales = 100 + trend + seasonal + 20*promotion + np.random.randn(weeks)*5

# Split data
train_sales = sales[:-12]
train_promo = promotion[:-12].reshape(-1, 1)
test_promo = promotion[-12:].reshape(-1, 1)

# Fit SARIMAX(1,1,1)(1,1,1,52) with promotion as exogenous
model = SARIMAX(train_sales, 
                exog=train_promo,
                order=(1, 1, 1), 
                seasonal_order=(1, 1, 1, 52))
result = model.fit(disp=False)

# Forecast next 12 weeks
forecast = result.get_forecast(steps=12, exog=test_promo)
print(f"Forecast: {forecast.predicted_mean[:3]}")
print(f"95% CI: {forecast.conf_int().iloc[:3]}")

# Output: Point forecasts with confidence intervals for inventory planning
```

**Interview Tips:**
- Tie forecast evaluation to business costs (stockout vs overstock).
- Use WAPE/MASE for comparability across SKUs and scales.
- Emphasize reconciliation and probabilistic outputs for inventory planning.

---

## Question 22
- [ ] Done

**Describe how you would use time series data to optimize pricing strategies over time.**

**Definition:**
Using time series analysis to understand price elasticity of demand and find the revenue-maximizing price point. This is a causal inference + optimization problem.

**Approach:**

**Step 1: Build Causal Demand Model**

Goal: Understand how price change **causes** sales change

Log-Log Model (for elasticity interpretation):
$$\log(Sales_t) = \beta_0 + \beta_1 \log(Price_t) + \beta_2 \log(Competitor\_Price_t) + \beta_3 \log(Ad\_Spend_t) + ARMA\_errors$$

- $\beta_1$ = **price elasticity** (e.g., $\beta_1 = -1.5$ means 1% price increase → 1.5% sales decrease)

**Step 2: Data Requirements**
- Historical sales, prices, competitor prices, marketing spend
- **Critical:** Need historical price variation; if price never changed, can't estimate effect

**Step 3: Revenue Optimization**

Revenue = Price × Sales

$$Revenue(P) = P \times E[Sales | Price=P]$$

Use model to find $P^*$ that maximizes revenue:
- Simulate revenue for range of prices
- Or solve analytically using calculus

**Step 4: Implementation**
- A/B test the optimal price vs current price
- Monitor and retrain as market conditions change

**Advanced Considerations:**
- Use causal methods (IV, diff-in-diff, synthetic control) when price changes are endogenous.
- Model cross-elasticities and competitor response in multi-product settings.
- Apply guardrails for inventory, margin, and customer fairness constraints.
- Consider dynamic pricing or bandit approaches for continuous learning.

**Python Example:**
```python
import numpy as np
from scipy.optimize import minimize_scalar

# Pipeline: Estimate elasticity → Define revenue function → Optimize

# Simulated elasticity (from demand model): -1.5
elasticity = -1.5
base_price = 100
base_sales = 1000

# Demand function: Sales = base_sales * (Price/base_price)^elasticity
def demand(price):
    return base_sales * (price / base_price) ** elasticity

# Revenue function
def revenue(price):
    return price * demand(price)

# Find optimal price (negative because we minimize)
result = minimize_scalar(lambda p: -revenue(p), bounds=(50, 200), method='bounded')
optimal_price = result.x

print(f"Optimal Price: ${optimal_price:.2f}")
print(f"Expected Sales: {demand(optimal_price):.0f}")
print(f"Max Revenue: ${revenue(optimal_price):.2f}")

# Compare with current price
print(f"\nCurrent Revenue: ${revenue(base_price):.2f}")
print(f"Revenue Gain: ${revenue(optimal_price) - revenue(base_price):.2f}")
```

**Interview Tips:**
- Separate correlation from causation; price is usually endogenous.
- Report elasticity with confidence intervals and sensitivity analysis.
- Treat pricing as an optimization under constraints, not just a point estimate.

---

## Question 23
- [ ] Done

**What are some current research areas in time series analysis and forecasting?**

**Definition:**
Time series research is rapidly evolving, driven by big data, compute power, and new application domains.

**Key Research Areas:**

**1. Deep Learning for Time Series**
| Architecture | Application |
|--------------|-------------|
| **Transformers** | Self-attention captures long-range dependencies; Informer, Autoformer |
| **Graph Neural Networks** | Spatio-temporal data (traffic sensors, portfolio of assets) |
| **Hybrid Models** | Combine ARIMA + deep learning (DL learns residuals) |

**2. Probabilistic Forecasting**
- Move beyond point forecasts to **full probability distributions**
- "90% chance sales will be 80-120 units" vs "sales will be 100"
- Crucial for risk management and decision-making
- Tools: GluonTS, Mixture Density Networks

**3. Causal Inference for Time Series**
- Estimate **causal impact** of interventions
- "What was the effect of our marketing campaign?"
- Methods: Synthetic Control, CausalImpact (Google), Dynamic Treatment Effects

**4. AutoML for Time Series**
- Automatic pipeline: feature engineering → model selection → hyperparameter tuning
- Examples: auto_arima, AutoGluon-TimeSeries
- Goal: minimal human intervention for large-scale forecasting

**5. Hierarchical Time Series**
- Forecast thousands of related series simultaneously
- Ensure **coherence**: store-level forecasts sum to regional totals
- Methods: top-down, bottom-up, optimal reconciliation

**6. Foundation Models for Time Series**
- Pre-trained models on massive time series corpora
- Transfer learning to new domains with minimal fine-tuning
- Examples: TimesFM (Google), Chronos (Amazon)

**7. Uncertainty Calibration**
- Conformal prediction for distribution-free intervals
- Probabilistic calibration metrics (PIT, coverage)

**8. Change Point and Regime Detection**
- Online detection of structural breaks and regime shifts
- Adaptive models for non-stationary data

**Practical Relevance:**

| Area | Industry Application |
|------|---------------------|
| Deep Learning | Complex patterns, NLP-like sequence modeling |
| Probabilistic | Supply chain, finance risk |
| Causal Inference | Marketing attribution, policy evaluation |
| AutoML | Retail with 1000s of SKUs |
| Hierarchical | Multi-store retail, organizational budgets |

**Interview Tips:**
- Highlight uncertainty calibration and change-point detection for production systems.
- Foundation models are emerging; emphasize transfer learning and zero-shot forecasting.
- Pair deep learning advances with classical diagnostics for robustness.

---

## Question 24
- [ ] Done

**Describe the concept of wavelet analysis in the context of time series.**

**Definition:**
Wavelet analysis decomposes a time series into frequency components while preserving time localization. Unlike Fourier transform (frequency only), wavelets show **when** different frequencies occur.

**Core Concepts:**

**Fourier vs Wavelet:**

| Aspect | Fourier Transform | Wavelet Transform |
|--------|-------------------|-------------------|
| Output | Frequency content | Frequency + Time location |
| Basis | Sine/cosine waves (infinite) | Localized wavelets (finite) |
| Best for | Stationary signals | Non-stationary signals |
| Limitation | Loses time information | More complex interpretation |

**How Wavelets Work:**
- **Wavelet:** Small, wave-like oscillation localized in time
- **High-scale wavelets:** Stretched → capture low-frequency, long-term features
- **Low-scale wavelets:** Compressed → capture high-frequency, short-term events

**Advanced Notes:**
- **CWT vs DWT:** CWT provides a continuous time-frequency map; DWT enables fast multiresolution analysis via filter banks.
- **MODWT:** Handles non-dyadic lengths and is shift-invariant (useful for real data).
- **Boundary Effects:** Edge artifacts require padding or tapering; interpret edges cautiously.
- **Wavelet Choice:** Mother wavelet and decomposition level control time-frequency resolution.

**Use Cases in Time Series:**

| Application | Description |
|-------------|-------------|
| **Feature Extraction** | Wavelet coefficients as ML features |
| **Denoising** | Remove high-frequency noise, keep signal |
| **Anomaly Detection** | Detect sudden spikes/glitches localized in time |
| **Coherence Analysis** | How correlation between series changes over time and frequency |

**Python Example:**
```python
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Pipeline: Create signal with transient → Apply wavelet → Visualize time-frequency

np.random.seed(42)

# Create signal: low-freq base + high-freq burst in middle
t = np.linspace(0, 1, 1000)
signal = np.sin(2*np.pi*5*t)  # 5 Hz base
signal[400:600] += np.sin(2*np.pi*50*t[400:600])  # 50 Hz burst in middle
signal += np.random.randn(1000) * 0.1  # noise

# Continuous Wavelet Transform
scales = np.arange(1, 128)
coefficients, frequencies = pywt.cwt(signal, scales, 'morl')

# Plot scalogram (time-frequency representation)
plt.figure(figsize=(12, 4))
plt.imshow(np.abs(coefficients), aspect='auto', cmap='jet')
plt.title('Wavelet Scalogram - Shows 50Hz burst at t=0.4-0.6')
plt.ylabel('Scale (inverse frequency)')
plt.xlabel('Time')
plt.colorbar(label='Magnitude')
plt.show()

# Output: Scalogram shows high-frequency activity only in middle (localized!)
```

**Denoising Example:**
```python
# Wavelet denoising
coeffs = pywt.wavedec(signal, 'db4', level=4)
threshold = 0.5
coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
denoised = pywt.waverec(coeffs_thresholded, 'db4')

# denoised signal has reduced noise while preserving structure
```

**Interview Tips:**
- Mention multiresolution analysis and filter-bank implementation for efficiency.
- Discuss boundary handling and choice of mother wavelet as practical concerns.
- Use wavelets when frequency content is time-varying or transient.

---
