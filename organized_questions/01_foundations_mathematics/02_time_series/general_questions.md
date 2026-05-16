# Time Series Interview Questions - General Questions

## Question 1
- [ ] Done

**How do time series differ from cross-sectional data?**

**Definition:**
- **Time Series:** Observations of a single entity over time; order matters and values are correlated
- **Cross-Sectional:** Observations of multiple entities at a single point in time; order does not matter

**Key Differences:**

| Aspect | Time Series | Cross-Sectional |
|--------|-------------|-----------------|
| Dimension | Over time | At one time point |
| Entities | One | Multiple |
| Order | Matters (temporal) | Doesn't matter (can shuffle) |
| Dependence | Adjacent observations dependent | Assumed independent |
| Goal | Forecast, trend analysis | Compare entities, relationships |
| Example | Apple stock price for 10 years | S&P 500 prices on Dec 31, 2023 |

**Panel Data (Hybrid):**
- Multiple entities tracked over multiple time periods
- Example: Stock prices of all S&P 500 companies for 10 years

**Practical Relevance:**
- Time series: "How will this stock perform tomorrow?"
- Cross-sectional: "Which companies have highest revenue today?"
- Panel: "How do factors affect performance across companies over time?"

**Practical Implications:**
- Time series needs time-based splits; random shuffling leaks future information.
- Cross-sectional methods assume independence; time series violates that.
- Panel data usually needs entity-level effects to avoid mixing apples and oranges.

---

## Question 2
- [ ] Done

**How is seasonality addressed in the SARIMA (Seasonal ARIMA) model?**

**Definition:**
SARIMA extends ARIMA with seasonal components: SARIMA(p,d,q)(P,D,Q)s

**Seasonal Parameters:**
- $s$ = seasonal period (12 for monthly, 7 for daily)
- $D$ = seasonal differencing order
- $P$ = seasonal AR order
- $Q$ = seasonal MA order

**Three Mechanisms:**

| Component | Formula | Purpose |
|-----------|---------|---------|
| Seasonal Differencing (D) | $Y'_t = Y_t - Y_{t-s}$ | Removes seasonal pattern |
| Seasonal AR (P) | Depends on $Y_{t-s}, Y_{t-2s}...$ | This January depends on last January |
| Seasonal MA (Q) | Depends on $\epsilon_{t-s}, \epsilon_{t-2s}...$ | Seasonal shock effects |

**Practical Guidance:**
- Set $s$ from the calendar (7 for daily -> weekly, 12 for monthly -> yearly).
- Use seasonal differencing only if the seasonal pattern is not stable.
- Check residual ACF at seasonal lags; spikes mean seasonality remains.

**Python Example:**
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Monthly data with yearly seasonality
model = SARIMAX(data, order=(1,1,1), seasonal_order=(1,1,1,12))
result = model.fit()
```

---

## Question 3
- [ ] Done

**What metrics are commonly used to evaluate the accuracy of time series models?**

**Scale-Dependent Metrics:**

| Metric | Formula | Use When |
|--------|---------|----------|
| **MAE** | $\frac{1}{n}\sum|Y_t - \hat{Y}_t|$ | Costs are symmetric, outlier robust |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(Y_t - \hat{Y}_t)^2}$ | Large errors are particularly bad |

**Percentage Metrics:**

| Metric | Formula | Use When |
|--------|---------|----------|
| **MAPE** | $\frac{100}{n}\sum|\frac{Y_t - \hat{Y}_t}{Y_t}|$ | Compare across series, no zeros |

**Scaled Metrics:**

| Metric | Interpretation | Use When |
|--------|---------------|----------|
| **MASE** | < 1 beats naive, > 1 worse than naive | Best general-purpose metric |

**Other Useful Metrics:**
- **sMAPE:** Handles zeros better than MAPE for intermittent demand.
- **WAPE:** Weighted absolute error; useful across many SKUs.
- **Pinball Loss:** For quantile forecasts (prediction intervals).

**Practical Notes:**
- Report metrics by horizon (1-day, 7-day, 30-day) because errors grow with horizon.
- Use a business-weighted metric if under-forecast and over-forecast costs differ.

**Python Example:**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(actual, forecast)
rmse = np.sqrt(mean_squared_error(actual, forecast))
mape = np.mean(np.abs((actual - forecast) / actual)) * 100
```

---

## Question 4
- [ ] Done

**How do you ensure that a time series forecasting model is not overfitting?**

**Strategies:**

**1. Proper Train-Test Split (Most Important)**
- Split by TIME, not randomly
- Training set: earlier period; Test set: later period
- Poor train score + great test score = underfitting
- Great train score + poor test score = overfitting

**2. Keep Model Simple (Parsimony)**
- Prefer lower p, q orders in ARIMA
- Don't add unnecessary exogenous variables
- Start with baseline, add complexity only if justified

**3. Use Information Criteria (AIC/BIC)**
- Balance fit vs. complexity
- Lower AIC/BIC = better (penalizes extra parameters)

**4. Time Series Cross-Validation**
- Rolling forecast origin validation
- Consistent performance across folds = good generalization

**5. Regularization (for ML models)**
- L1/L2 penalties on coefficients
- Dropout in neural networks

**Practical Checks:**
- Always compare to a naive baseline.
- Inspect residual ACF; remaining structure means the model is too simple or mis-specified.
- Tune hyperparameters inside backtesting, not on the test set.

---

## Question 5
- [ ] Done

**In what ways can machine learning models be applied to time series forecasting?**

**Supervised Learning Framework:**
1. Create lag features: $Y_{t-1}, Y_{t-2}, ...$
2. Create time features: day, month, hour
3. Create rolling features: rolling mean, std
4. Train standard ML model on (features → target)

**Popular Models:**

| Model | Strengths | Use Case |
|-------|-----------|----------|
| **LightGBM/XGBoost** | Non-linear, handles many features, robust | Most common choice |
| **Linear Regression** | Simple, interpretable | Baseline |
| **LSTM** | Long-term dependencies | Complex sequences |
| **Transformers** | Attention mechanism | State-of-the-art |

**Advantages over ARIMA:**
- No stationarity requirement (learns from features)
- Easily handles many exogenous variables
- Captures non-linear relationships

**Practical Notes:**
- Leakage risk is high: lag and rolling features must use only past data.
- Global models (one model for many series) often outperform per-series models at scale.
- Multi-horizon forecasting is better done with direct or sequence-to-sequence targets.

**Python Example:**
```python
import pandas as pd
from lightgbm import LGBMRegressor

# Create lag features
df['lag_1'] = df['sales'].shift(1)
df['lag_7'] = df['sales'].shift(7)
df['rolling_mean_7'] = df['sales'].rolling(7).mean()

X = df[['lag_1', 'lag_7', 'rolling_mean_7']].dropna()
y = df.loc[X.index, 'sales']

model = LGBMRegressor()
model.fit(X_train, y_train)
```

---

## Question 6
- [ ] Done

**What considerations should be taken into account when using time series analysis for climate change research?**

**Key Considerations:**

| Challenge | Approach |
|-----------|----------|
| **Non-stationarity** | Trend IS the signal; model and quantify it, don't remove |
| **Multiple time scales** | Daily, annual, decadal cycles - use multi-seasonal models |
| **Spatio-temporal data** | Temperature at one location affects neighbors; use spatial models |
| **Causal attribution** | "Is warming caused by humans?" - need causal inference methods |
| **Uncertainty quantification** | Probabilistic forecasts essential; use ensemble approaches |

**Special Techniques:**
- Climate models + statistical methods for attribution
- Graph Neural Networks for spatial dependencies
- Long-term trend modeling (not differencing)

**Practical Notes:**
- Separate long-term trend (signal) from short-term variability (noise).
- Use anomalies (de-meaned series) to compare across regions.
- Report uncertainty bands; point estimates alone are misleading.

---

## Question 7
- [ ] Done

**How can time series models improve the forecasting of inventory levels in supply chain management?**

**Benefits:**

| Without Models | With Models |
|----------------|-------------|
| "Just-in-case" overstocking | "Just-in-time" optimization |
| Manual guesswork | Data-driven decisions |
| Reactive | Proactive |

**Key Improvements:**
1. **Capture Seasonality:** Pre-order before peak season (winter coats before winter)
2. **Handle Promotions:** SARIMAX with promotion flag predicts demand spikes
3. **Optimize Safety Stock:** Prediction intervals → scientific safety stock levels
4. **Scale:** Automated forecasting for 1000s of SKUs

**Model Choice:** SARIMAX (seasonal + external variables like promotions)

**Practical Notes:**
- Forecast lead-time demand, not just daily demand.
- Use prediction intervals to set safety stock by service level.
- For intermittent demand, use Croston/TSB or zero-inflated models.

---

## Question 8
- [ ] Done

**Outline a time series analysis method to identify trends in social media engagement.**

**Method: STL Decomposition**

**Steps:**
1. **Data:** Daily engagement (likes + comments + shares) for 2+ years
2. **STL Decomposition:** $Y_t = T_t + S_t + R_t$ (Trend + Seasonal + Residual)
3. **Analyze Trend:** Plot $T_t$ alone - shows long-term growth/decline
4. **Analyze Seasonality:** Weekly patterns (best days to post)
5. **Analyze Residuals:** Large spikes = viral posts or campaign effects

**Practical Notes:**
- Normalize by impressions or followers to avoid growth bias.
- Tag campaigns and platform changes; they create step changes in trend.
- Use changepoint detection if the trend shifts abruptly.

**Python Example:**
```python
from statsmodels.tsa.seasonal import STL

stl = STL(engagement_series, period=7)  # weekly seasonality
result = stl.fit()
result.plot()

# Quantify trend: fit regression to trend component
```

---

## Question 9
- [ ] Done

**How are Fourier transforms used in analyzing time series data?**

**Definition:**
Fourier transform decomposes a signal from **time domain** to **frequency domain** - shows which frequencies (cycles) are present.

**Applications:**

| Use Case | Process |
|----------|---------|
| **Identify Seasonality** | Peaks in spectrum at frequency = 1/period |
| **Denoising** | Remove high-frequency noise, inverse transform |
| **Feature Engineering** | Top frequency amplitudes as ML features |

**Limitation:**
- Loses time information (knows "what frequency" but not "when")
- For non-stationary: use wavelets instead

**Practical Notes:**
- Use windowing (e.g., Hann) to reduce spectral leakage.
- Frequency resolution depends on series length and sampling rate.
- Best for stable seasonal patterns; avoid if the period changes over time.

**Python Example:**
```python
import numpy as np
from scipy.fft import fft

# FFT of signal
freq_spectrum = np.abs(fft(signal))
frequencies = np.fft.fftfreq(len(signal), d=1/sampling_rate)

# Peak at frequency f means cycle of period 1/f
```

---

## Question 10
- [ ] Done

**How can deep learning models, such as Long Short-Term Memory (LSTM) networks, be utilized for complex time series analysis tasks?**

**Why LSTMs for Time Series:**
- Handle **long-range dependencies** (events from months ago still matter)
- Overcome vanishing gradient problem via **gating mechanism**
- Automatic **feature learning** (no manual engineering)

**LSTM Cell Components:**
- **Forget Gate:** What to discard from memory
- **Input Gate:** What new info to store
- **Output Gate:** What to output

**Applications:**

| Task | LSTM Advantage |
|------|----------------|
| Long-term forecasting | Learns dependencies ARIMA can't capture |
| Multivariate forecasting | Naturally handles multiple input sequences |
| Anomaly detection | LSTM autoencoder - high reconstruction error = anomaly |
| Multi-step forecasting | Encoder-decoder architecture |

**Python Example:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(50, input_shape=(lookback, n_features)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50)
```

**Interview Tips:**
- LSTM for complex patterns; ARIMA for simpler, interpretable cases
- LSTMs need more data to train effectively
- Consider Transformers as state-of-the-art alternative

**Practical Tips:**
- Always start with a strong baseline and backtest across horizons.
- Scale inputs and use early stopping to avoid overfitting.
- Use sequence-to-sequence outputs for multi-step forecasts rather than rolling 1-step.

---