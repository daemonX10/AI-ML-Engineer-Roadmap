# Time Series Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Implement a Python function to perform simple exponential smoothing on a time series.**

**Simple Exponential Smoothing Formula:**
$$F_{t+1} = \alpha \cdot Y_t + (1-\alpha) \cdot F_t$$

where $\alpha$ is smoothing parameter (0 to 1)

**Code:**
```python
import numpy as np

def simple_exponential_smoothing(series, alpha, init=None):
    """
    Simple Exponential Smoothing implementation.

    Args:
        series: list or array-like values
        alpha: smoothing parameter (0 <= alpha <= 1)
        init: initial level (defaults to first value)

    Returns:
        list of smoothed values
    """
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")

    values = np.asarray(series, dtype=float)
    if values.size == 0:
        return []

    level = values[0] if init is None else float(init)
    result = np.empty_like(values)
    result[0] = level

    for t in range(1, len(values)):
        level = alpha * values[t - 1] + (1 - alpha) * level
        result[t] = level

    return result.tolist()

# Example usage
data = [20, 22, 25, 23, 26, 24, 28, 27, 29, 30]
smoothed = simple_exponential_smoothing(data, alpha=0.3)
print(f"Original: {data}")
print(f"Smoothed: {[round(x, 2) for x in smoothed]}")
```

**Output:** Smoothed series that follows original with less noise

**Notes:**
- Tune $\alpha$ by minimizing in-sample error if you need the best fit.
- Initialization affects short series; consider the mean of the first few points.
- Handle missing values before smoothing (forward fill or drop).

---

## Question 2
- [ ] Done

**Using pandas, write a script to detect seasonality in a time series dataset.**

**Approach:** Group by seasonal period and estimate seasonal strength

**Code:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Step 1: Create sample data with weekly seasonality
dates = pd.date_range('2022-01-01', periods=365, freq='D')
base_value = 100
seasonal = [15 if d.dayofweek >= 5 else 0 for d in dates]
noise = np.random.randn(365) * 5
sales = base_value + seasonal + noise

df = pd.DataFrame({'sales': sales}, index=dates)

# Step 2: Add day of week column
df['day_of_week'] = df.index.dayofweek

# Step 3: Method 1 - Group by day, calculate mean
daily_means = df.groupby('day_of_week')['sales'].mean()
print("Mean by day of week:")
print(daily_means)

# Step 4: Method 2 - Seasonal strength via STL
stl = STL(df['sales'], period=7, robust=True).fit()
seasonal_strength = 1 - np.var(stl.resid) / np.var(stl.resid + stl.seasonal)
print(f"Seasonal strength (0-1): {seasonal_strength:.2f}")

# Step 5: Visualize with boxplots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

daily_means.plot(kind='bar', ax=axes[0])
axes[0].set_title('Mean Sales by Day')
axes[0].set_xlabel('Day (0=Mon, 6=Sun)')

df.boxplot(column='sales', by='day_of_week', ax=axes[1])
axes[1].set_title('Sales Distribution by Day')
plt.tight_layout()
plt.show()
```

**Output:** Higher means and a strong seasonal strength score indicate weekly seasonality

**Notes:**
- Use the correct seasonal period (7 for daily, 12 for monthly).
- If seasonal strength is low, avoid seasonal differencing.

---

## Question 3
- [ ] Done

**Code an ARIMA model in Python on a given dataset and visualize the forecasts.**

**Pipeline:** Check stationarity -> Fit ARIMA -> Forecast -> Plot

**Code:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

# Step 1: Create sample data (random walk with drift)
np.random.seed(42)
n = 100
data = np.cumsum(0.5 + np.random.randn(n))

# Step 2: Check stationarity
result = adfuller(data)
print(f"ADF p-value: {result[1]:.4f}")
print("Non-stationary" if result[1] > 0.05 else "Stationary")

# Step 3: Train-test split
train_size = 80
train, test = data[:train_size], data[train_size:]

# Step 4: Fit ARIMA model
model = ARIMA(train, order=(1, 1, 1))
fitted = model.fit()
print(fitted.summary())

# Step 5: Forecast with intervals
forecast_res = fitted.get_forecast(steps=len(test))
forecast = forecast_res.predicted_mean
conf_int = forecast_res.conf_int()

# Step 6: Visualize
plt.figure(figsize=(12, 5))
plt.plot(range(train_size), train, label='Train', color='blue')
plt.plot(range(train_size, n), test, label='Test', color='green')
plt.plot(range(train_size, n), forecast, label='Forecast', color='red', linestyle='--')
plt.fill_between(range(train_size, n),
                 conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                 color='red', alpha=0.1, label='95% CI')
plt.axvline(x=train_size, color='black', linestyle=':')
plt.legend()
plt.title('ARIMA(1,1,1) Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# Step 7: Calculate error
mae = np.mean(np.abs(test - forecast))
print(f"MAE: {mae:.2f}")

# Step 8: Residual check
lb_pvalue = acorr_ljungbox(fitted.resid, lags=[10], return_df=True)['lb_pvalue'].iloc[0]
print(f"Ljung-Box p-value (lag 10): {lb_pvalue:.3f}")
```

---

## Question 4
- [ ] Done

**Fit a GARCH model to a financial time series dataset and interpret the results.**

**Pipeline:** Get returns -> Fit GARCH(1,1) -> Interpret persistence

**Code:**
```python
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
import matplotlib.pyplot as plt

# Step 1: Create sample returns (or use real data)
np.random.seed(42)
n = 1000
returns = np.random.randn(n) * 0.02
returns = returns * 100  # scale for stable optimization

# Optional: check ARCH effect
arch_test = het_arch(returns)
print(f"ARCH test p-value: {arch_test[1]:.4f}")

# Step 2: Fit GARCH(1,1) model
model = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant', dist='t')
result = model.fit(disp='off')

# Step 3: Print summary
print(result.summary())

# Step 4: Extract key parameters
omega = result.params['omega']
alpha = result.params['alpha[1]']
beta = result.params['beta[1]']

print("\n--- Interpretation ---")
print(f"omega (omega): {omega:.6f}")
print(f"alpha (alpha): {alpha:.4f} - Impact of past shock")
print(f"beta (beta): {beta:.4f} - Persistence of volatility")
print(f"alpha + beta: {alpha + beta:.4f} - Total persistence")

# Step 5: Forecast volatility
forecast = result.forecast(horizon=10)
vol_forecast = np.sqrt(forecast.variance.values[-1])
print(f"\nVolatility forecast (next 10 periods): {vol_forecast}")

# Step 6: Plot conditional volatility
fig, ax = plt.subplots(figsize=(12, 4))
result.conditional_volatility.plot(ax=ax)
ax.set_title('Conditional Volatility (GARCH)')
ax.set_ylabel('Volatility')
plt.show()
```

**Interpretation:** alpha + beta close to 1 means high persistence (shocks last long)

---

## Question 5
- [ ] Done

**Create a Python script that decomposes a time series into trend, seasonality, and residuals using statsmodels.**

**Pipeline:** Load data -> STL decompose -> Plot components

**Code:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Step 1: Create sample data with trend + seasonality + noise
np.random.seed(42)
n = 365 * 2  # 2 years of daily data

time = np.arange(n)
trend = 100 + 0.05 * time
seasonal = 10 * np.sin(2 * np.pi * time / 365)
noise = np.random.randn(n) * 3

data = trend + seasonal + noise

dates = pd.date_range('2022-01-01', periods=n, freq='D')
ts = pd.Series(data, index=dates)

# Step 2: STL Decomposition (robust to outliers)
stl = STL(ts, period=365, robust=True)
result = stl.fit()

# Step 3: Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(ts)
axes[0].set_title('Original')

axes[1].plot(result.trend)
axes[1].set_title('Trend')

axes[2].plot(result.seasonal)
axes[2].set_title('Seasonal')

axes[3].plot(result.resid)
axes[3].set_title('Residuals')

plt.tight_layout()
plt.show()

# Step 4: Verify decomposition
reconstructed = result.trend + result.seasonal + result.resid
print(f"Reconstruction error: {np.mean(np.abs(ts - reconstructed)):.6f}")
```

**Output:** Four plots showing original data and extracted components

**Notes:**
- Choose the seasonal period from the calendar (7, 12, 365) rather than guessing.
- If you have multiple seasonalities, consider TBATS or multiple seasonal STL.
- Handle missing timestamps before decomposition.

---

## Question 6
- [ ] Done

**Write a Python function to calculate and plot the ACF and PACF for a given time series.**

**Pipeline:** Input series -> Calculate ACF/PACF -> Plot with significance bands

**Code:**
```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_acf_pacf(series, lags=30, title=""):
    """
    Plot ACF and PACF for a time series.

    Args:
        series: array-like time series data
        lags: number of lags to plot
        title: plot title prefix

    Returns:
        fig: matplotlib figure
    """
    series = np.asarray(series, dtype=float)
    series = series[~np.isnan(series)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    plot_acf(series, lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title(f"{title} ACF")
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('Autocorrelation')

    plot_pacf(series, lags=lags, ax=axes[1], alpha=0.05, method='ywm')
    axes[1].set_title(f"{title} PACF")
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Partial Autocorrelation')

    plt.tight_layout()
    return fig

# Example: Create AR(2) process
np.random.seed(42)
n = 300
y = np.zeros(n)
for t in range(2, n):
    y[t] = 0.6 * y[t - 1] - 0.3 * y[t - 2] + np.random.randn()

fig = plot_acf_pacf(y, lags=20, title="AR(2) Process")
plt.show()

print("--- Interpretation ---")
print("If ACF tails off, PACF cuts off at p -> AR(p)")
print("If ACF cuts off at q, PACF tails off -> MA(q)")
print("If both tail off -> ARMA(p,q)")
print("For AR(2): Expect PACF to cut off after lag 2")
```

**Output:**
- ACF: Shows total correlation (should tail off for AR)
- PACF: Shows direct correlation (should cut off at lag 2 for AR(2))
- Blue bands = 95% confidence interval

**Notes:**
- Apply ACF/PACF on a stationary series to avoid misleading patterns.
- Use residual ACF to validate a fitted model.
- Keep lags reasonable; very long lag plots are noisy in small samples.
