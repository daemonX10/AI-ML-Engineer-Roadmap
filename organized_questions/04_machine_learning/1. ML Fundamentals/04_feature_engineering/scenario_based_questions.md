# Feature Engineering Interview Questions - Scenario-Based Questions

## Question 1: Explain how you would perform feature engineering for a recommendation system.
- [ ] Done

### Answer

**Three Categories of Features:**

**1. User Features:**
```python
user_features = {
    'user_age': 28,
    'user_gender': 'M',
    'user_location': 'NYC',
    'avg_rating_given': 3.8,
    'num_ratings': 150,
    'favorite_genres': ['action', 'comedy'],
    'user_embedding': [0.1, 0.3, ...]  # From collaborative filtering
}
```

**2. Item Features:**
```python
item_features = {
    'item_category': 'electronics',
    'item_price': 299.99,
    'item_avg_rating': 4.2,
    'num_reviews': 1500,
    'description_embedding': [0.2, 0.4, ...],  # From NLP
    'item_popularity': 0.85
}
```

**3. Interaction Features (Most Powerful):**
```python
interaction_features = {
    'user_item_similarity': 0.72,
    'user_category_affinity': 0.8,
    'days_since_last_interaction': 5,
    'num_previous_purchases': 3,
    'context_time_of_day': 'evening',
    'context_device': 'mobile'
}
```

**Feature Engineering Pipeline:**
```python
# Collaborative filtering embeddings
from sklearn.decomposition import TruncatedSVD

# Create user-item matrix
user_item_matrix = create_user_item_matrix(interactions)

# Get embeddings
svd = TruncatedSVD(n_components=50)
user_embeddings = svd.fit_transform(user_item_matrix)
item_embeddings = svd.components_.T

# Interaction features
def create_interaction_features(user_id, item_id, df):
    features = {}
    features['user_item_sim'] = cosine_similarity(
        user_embeddings[user_id], item_embeddings[item_id]
    )
    features['user_avg_rating_for_category'] = df[
        (df['user_id'] == user_id) & 
        (df['category'] == item_category)
    ]['rating'].mean()
    return features
```

---

## Question 2: Describe the feature engineering process you would use for a customer churn prediction model.
- [ ] Done

### Answer

**Feature Categories:**

**1. Static/Demographic Features:**
```python
static_features = [
    'age', 'gender', 'location', 'acquisition_channel',
    'subscription_tier', 'payment_method', 'signup_date'
]
```

**2. Behavioral Features (Time-Windowed):**
```python
def create_behavioral_features(df, window_days):
    features = {
        'login_frequency_30d': count_logins(df, 30),
        'login_frequency_90d': count_logins(df, 90),
        'time_since_last_login': days_since_last_login(df),
        'feature_usage_counts': count_feature_usage(df),
        'session_duration_avg': avg_session_duration(df),
        'pages_viewed_per_session': avg_pages_per_session(df)
    }
    return features
```

**3. Trend Features (Critical for Churn):**
```python
def create_trend_features(df):
    return {
        'usage_trend_30d_vs_90d': (
            usage_30d - usage_90d_avg
        ) / usage_90d_avg,
        'activity_slope': calculate_activity_slope(df),
        'engagement_decay_rate': calculate_decay_rate(df)
    }
```

**4. Support/Billing Features:**
```python
support_features = {
    'num_support_tickets_90d': count_tickets(df, 90),
    'avg_satisfaction_score': avg_csat(df),
    'num_failed_payments': count_failed_payments(df),
    'days_since_last_complaint': days_since_complaint(df)
}
```

**Complete Pipeline:**
```python
def engineer_churn_features(df, snapshot_date):
    features = {}
    
    # Static
    features.update(get_static_features(df))
    
    # Behavioral (multiple windows)
    for window in [7, 30, 90]:
        features.update(create_behavioral_features(df, window))
    
    # Trends
    features.update(create_trend_features(df))
    
    # Support
    features.update(get_support_features(df))
    
    # Preprocessing
    features_df = pd.DataFrame([features])
    features_df = pd.get_dummies(features_df, columns=categorical_cols)
    features_df = StandardScaler().fit_transform(features_df)
    
    return features_df
```

---

## Question 3: You're building a predictive maintenance model. What types of features would you engineer from sensor data?
- [ ] Done

### Answer

**Predictive Maintenance Feature Engineering Strategy:**

**1. Time-Domain Statistical Features:**

```python
import numpy as np
import pandas as pd
from scipy import stats

def extract_statistical_features(signal, window_name=""):
    """
    Extract statistical features from a sensor signal.
    """
    features = {
        f'{window_name}mean': np.mean(signal),
        f'{window_name}std': np.std(signal),
        f'{window_name}var': np.var(signal),
        f'{window_name}min': np.min(signal),
        f'{window_name}max': np.max(signal),
        f'{window_name}range': np.max(signal) - np.min(signal),
        f'{window_name}median': np.median(signal),
        f'{window_name}skewness': stats.skew(signal),
        f'{window_name}kurtosis': stats.kurtosis(signal),
        f'{window_name}rms': np.sqrt(np.mean(signal**2)),
        f'{window_name}peak_to_peak': np.ptp(signal),
        f'{window_name}crest_factor': np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)),
        f'{window_name}percentile_25': np.percentile(signal, 25),
        f'{window_name}percentile_75': np.percentile(signal, 75),
        f'{window_name}iqr': np.percentile(signal, 75) - np.percentile(signal, 25),
    }
    return features
```

**2. Frequency-Domain Features:**

```python
from scipy.fft import fft
from scipy.signal import welch

def extract_frequency_features(signal, sampling_rate, window_name=""):
    """
    Extract frequency domain features using FFT.
    """
    # FFT
    n = len(signal)
    fft_vals = np.abs(fft(signal))[:n//2]
    freqs = np.fft.fftfreq(n, 1/sampling_rate)[:n//2]
    
    # Power spectral density
    f_psd, psd = welch(signal, fs=sampling_rate)
    
    features = {
        f'{window_name}dominant_freq': freqs[np.argmax(fft_vals)],
        f'{window_name}spectral_centroid': np.sum(freqs * fft_vals) / np.sum(fft_vals),
        f'{window_name}spectral_spread': np.sqrt(np.sum(((freqs - np.sum(freqs * fft_vals) / np.sum(fft_vals))**2) * fft_vals) / np.sum(fft_vals)),
        f'{window_name}spectral_entropy': stats.entropy(psd + 1e-10),
        f'{window_name}total_power': np.sum(psd),
        f'{window_name}peak_power': np.max(psd),
        f'{window_name}power_ratio_low': np.sum(psd[f_psd < 50]) / np.sum(psd),
        f'{window_name}power_ratio_high': np.sum(psd[f_psd >= 50]) / np.sum(psd),
    }
    return features
```

**3. Rolling Window Features (Trend Detection):**

```python
def extract_rolling_features(df, sensor_cols, windows=[60, 300, 3600]):
    """
    Extract rolling window features for trend detection.
    
    Args:
        df: DataFrame with sensor data
        sensor_cols: List of sensor column names
        windows: Window sizes in seconds/samples
    """
    for col in sensor_cols:
        for window in windows:
            # Rolling statistics
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
            df[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
            df[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()
            
            # Deviation from rolling mean (anomaly indicator)
            df[f'{col}_deviation_{window}'] = df[col] - df[f'{col}_rolling_mean_{window}']
            
            # Rate of change
            df[f'{col}_rate_of_change_{window}'] = df[col].diff(window) / window
            
            # Expanding features (cumulative)
            df[f'{col}_expanding_mean'] = df[col].expanding().mean()
            df[f'{col}_expanding_std'] = df[col].expanding().std()
    
    return df
```

**4. Health Indicators:**

```python
def create_health_indicators(df, sensor_cols, baseline_period=1000):
    """
    Create health indicators comparing current state to baseline.
    """
    # Establish baseline (healthy state)
    baseline_stats = {}
    for col in sensor_cols:
        baseline_stats[col] = {
            'mean': df[col].iloc[:baseline_period].mean(),
            'std': df[col].iloc[:baseline_period].std()
        }
    
    # Calculate health indicators
    for col in sensor_cols:
        # Mahalanobis-like distance from baseline
        df[f'{col}_health_score'] = np.abs(
            (df[col] - baseline_stats[col]['mean']) / 
            (baseline_stats[col]['std'] + 1e-10)
        )
        
        # Cumulative degradation
        df[f'{col}_cumulative_deviation'] = (
            np.abs(df[col] - baseline_stats[col]['mean']).cumsum()
        )
        
        # Threshold exceedance count
        threshold = baseline_stats[col]['mean'] + 3 * baseline_stats[col]['std']
        df[f'{col}_exceedance_count'] = (df[col] > threshold).cumsum()
    
    return df
```

**5. Cross-Sensor Features:**

```python
def create_cross_sensor_features(df, sensor_pairs):
    """
    Create features from relationships between sensors.
    
    Args:
        sensor_pairs: List of tuples [(sensor1, sensor2), ...]
    """
    for sensor1, sensor2 in sensor_pairs:
        # Ratio features
        df[f'{sensor1}_{sensor2}_ratio'] = df[sensor1] / (df[sensor2] + 1e-10)
        
        # Difference features
        df[f'{sensor1}_{sensor2}_diff'] = df[sensor1] - df[sensor2]
        
        # Correlation (rolling)
        df[f'{sensor1}_{sensor2}_corr_60'] = df[sensor1].rolling(60).corr(df[sensor2])
        
        # Product (interaction)
        df[f'{sensor1}_{sensor2}_product'] = df[sensor1] * df[sensor2]
    
    return df
```

**Complete Feature Engineering Pipeline:**

```python
def predictive_maintenance_features(df, sensor_cols, sampling_rate=100):
    """
    Complete feature engineering pipeline for predictive maintenance.
    """
    features_list = []
    
    # Process each time window
    window_size = 1000  # samples per window
    
    for i in range(0, len(df) - window_size, window_size // 2):  # 50% overlap
        window_features = {'window_start': i}
        
        for col in sensor_cols:
            signal = df[col].iloc[i:i+window_size].values
            
            # Statistical features
            window_features.update(extract_statistical_features(signal, f'{col}_'))
            
            # Frequency features
            window_features.update(extract_frequency_features(signal, sampling_rate, f'{col}_'))
        
        features_list.append(window_features)
    
    return pd.DataFrame(features_list)

# Example usage
# features_df = predictive_maintenance_features(sensor_df, ['vibration', 'temperature', 'pressure'])
```

**Summary of Feature Categories:**

| Category | Examples | Purpose |
|----------|----------|---------|
| **Statistical** | Mean, std, RMS, kurtosis | Capture signal distribution |
| **Frequency** | Dominant freq, spectral entropy | Detect vibration patterns |
| **Rolling** | Moving averages, trends | Capture degradation trends |
| **Health** | Deviation from baseline | Quantify degradation |
| **Cross-sensor** | Ratios, correlations | Capture system-wide patterns |

---

## Question 4: You're implementing a real-time anomaly detection system. What feature engineering strategies would you employ to detect anomalies in streaming data?
- [ ] Done

### Answer

**Real-Time Feature Engineering Challenges:**
- Limited memory (can't store all history)
- Low latency requirements
- Concept drift handling
- Efficient computation

**Strategy 1: Online Statistical Features**

```python
import numpy as np
from collections import deque

class OnlineStatistics:
    """
    Compute statistics incrementally for streaming data.
    Uses Welford's algorithm for numerical stability.
    """
    
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.n = 0
        self.mean = 0
        self.M2 = 0  # Sum of squared differences
        self.min_val = float('inf')
        self.max_val = float('-inf')
    
    def update(self, value):
        """Update statistics with new value."""
        self.window.append(value)
        self.n += 1
        
        # Welford's online algorithm for mean and variance
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2
        
        # Update min/max
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        
        # For windowed statistics
        if len(self.window) == self.window_size:
            old_value = self.window[0]
            # Adjust for removed value (approximate)
            self._adjust_for_removal(old_value)
    
    def _adjust_for_removal(self, old_value):
        """Approximate adjustment when removing old value from window."""
        if self.n > 1:
            self.n -= 1
            delta = old_value - self.mean
            self.mean -= delta / self.n
            delta2 = old_value - self.mean
            self.M2 -= delta * delta2
    
    def get_features(self):
        """Return current statistical features."""
        variance = self.M2 / self.n if self.n > 1 else 0
        window_list = list(self.window)
        
        return {
            'mean': self.mean,
            'std': np.sqrt(variance),
            'min': self.min_val,
            'max': self.max_val,
            'range': self.max_val - self.min_val,
            'window_mean': np.mean(window_list),
            'window_std': np.std(window_list),
            'z_score': (window_list[-1] - self.mean) / (np.sqrt(variance) + 1e-10)
        }


# Usage
stats = OnlineStatistics(window_size=100)
for value in streaming_data:
    stats.update(value)
    features = stats.get_features()
    # Use features for anomaly detection
```

**Strategy 2: Exponentially Weighted Moving Statistics**

```python
class ExponentialMovingStats:
    """
    Exponentially weighted moving average and variance.
    More recent values have higher weight.
    """
    
    def __init__(self, alpha=0.1):
        """
        Args:
            alpha: Smoothing factor (0-1). Higher = more weight to recent values.
        """
        self.alpha = alpha
        self.ewma = None
        self.ewmvar = None
        self.initialized = False
    
    def update(self, value):
        """Update with new value."""
        if not self.initialized:
            self.ewma = value
            self.ewmvar = 0
            self.initialized = True
        else:
            diff = value - self.ewma
            incr = self.alpha * diff
            self.ewma += incr
            self.ewmvar = (1 - self.alpha) * (self.ewmvar + self.alpha * diff * diff)
        
        return self.get_features(value)
    
    def get_features(self, current_value):
        """Get current features."""
        ewm_std = np.sqrt(self.ewmvar) if self.ewmvar > 0 else 1e-10
        
        return {
            'ewma': self.ewma,
            'ewm_std': ewm_std,
            'ewm_z_score': (current_value - self.ewma) / ewm_std,
            'deviation': current_value - self.ewma,
            'deviation_ratio': (current_value - self.ewma) / (self.ewma + 1e-10)
        }


# Usage for multiple sensors
class MultiSensorOnlineFeatures:
    def __init__(self, sensor_names, alpha=0.1, window_size=100):
        self.sensors = {name: {
            'ewm': ExponentialMovingStats(alpha),
            'online': OnlineStatistics(window_size)
        } for name in sensor_names}
    
    def update(self, sensor_data):
        """
        Update with new readings from all sensors.
        
        Args:
            sensor_data: dict {sensor_name: value}
        """
        features = {}
        for name, value in sensor_data.items():
            # EWM features
            ewm_features = self.sensors[name]['ewm'].update(value)
            features.update({f'{name}_{k}': v for k, v in ewm_features.items()})
            
            # Window features
            self.sensors[name]['online'].update(value)
            online_features = self.sensors[name]['online'].get_features()
            features.update({f'{name}_win_{k}': v for k, v in online_features.items()})
        
        return features
```

**Strategy 3: Change Point Detection Features**

```python
class ChangePointDetector:
    """
    Detect sudden changes in streaming data.
    """
    
    def __init__(self, short_window=10, long_window=100, threshold=3):
        self.short_window = deque(maxlen=short_window)
        self.long_window = deque(maxlen=long_window)
        self.threshold = threshold
    
    def update(self, value):
        """Update and compute change detection features."""
        self.short_window.append(value)
        self.long_window.append(value)
        
        if len(self.short_window) < 10 or len(self.long_window) < 50:
            return {'change_score': 0, 'is_change_point': False}
        
        short_mean = np.mean(self.short_window)
        long_mean = np.mean(self.long_window)
        long_std = np.std(self.long_window)
        
        # CUSUM-like score
        change_score = abs(short_mean - long_mean) / (long_std + 1e-10)
        
        return {
            'short_mean': short_mean,
            'long_mean': long_mean,
            'change_score': change_score,
            'is_change_point': change_score > self.threshold
        }
```

**Strategy 4: Lag Features for Temporal Patterns**

```python
class LagFeatureGenerator:
    """
    Generate lag features for time-dependent patterns.
    """
    
    def __init__(self, lags=[1, 5, 10, 30, 60]):
        self.lags = lags
        self.max_lag = max(lags)
        self.buffer = deque(maxlen=self.max_lag + 1)
    
    def update(self, value):
        """Update buffer and generate lag features."""
        self.buffer.append(value)
        
        features = {'current': value}
        
        if len(self.buffer) > max(self.lags):
            buffer_list = list(self.buffer)
            for lag in self.lags:
                lag_value = buffer_list[-lag - 1]
                features[f'lag_{lag}'] = lag_value
                features[f'diff_{lag}'] = value - lag_value
                features[f'pct_change_{lag}'] = (value - lag_value) / (abs(lag_value) + 1e-10)
        
        return features
```

**Complete Real-Time Anomaly Detection Pipeline:**

```python
class RealTimeAnomalyFeatures:
    """
    Complete feature engineering pipeline for real-time anomaly detection.
    """
    
    def __init__(self, sensor_names, config=None):
        self.config = config or {
            'ewm_alpha': 0.1,
            'window_size': 100,
            'lags': [1, 5, 10, 30],
            'change_threshold': 3
        }
        
        self.multi_sensor = MultiSensorOnlineFeatures(
            sensor_names, 
            alpha=self.config['ewm_alpha'],
            window_size=self.config['window_size']
        )
        
        self.change_detectors = {
            name: ChangePointDetector(threshold=self.config['change_threshold'])
            for name in sensor_names
        }
        
        self.lag_generators = {
            name: LagFeatureGenerator(lags=self.config['lags'])
            for name in sensor_names
        }
    
    def process(self, sensor_data, timestamp=None):
        """
        Process new sensor readings and generate features.
        
        Args:
            sensor_data: dict {sensor_name: value}
            timestamp: optional timestamp
            
        Returns:
            dict of features for anomaly detection
        """
        features = {}
        
        # Add timestamp features if provided
        if timestamp:
            features.update(self._timestamp_features(timestamp))
        
        # Statistical features
        features.update(self.multi_sensor.update(sensor_data))
        
        # Change detection features
        for name, value in sensor_data.items():
            change_features = self.change_detectors[name].update(value)
            features.update({f'{name}_{k}': v for k, v in change_features.items()})
            
            # Lag features
            lag_features = self.lag_generators[name].update(value)
            features.update({f'{name}_{k}': v for k, v in lag_features.items()})
        
        # Cross-sensor features
        features.update(self._cross_sensor_features(sensor_data))
        
        return features
    
    def _timestamp_features(self, timestamp):
        """Extract time-based features."""
        return {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
            'minute_of_day': timestamp.hour * 60 + timestamp.minute
        }
    
    def _cross_sensor_features(self, sensor_data):
        """Compute cross-sensor features."""
        features = {}
        sensors = list(sensor_data.keys())
        
        for i, s1 in enumerate(sensors):
            for s2 in sensors[i+1:]:
                features[f'{s1}_{s2}_ratio'] = sensor_data[s1] / (sensor_data[s2] + 1e-10)
                features[f'{s1}_{s2}_diff'] = sensor_data[s1] - sensor_data[s2]
        
        return features


# Usage example
feature_engine = RealTimeAnomalyFeatures(['temperature', 'pressure', 'vibration'])

for reading in data_stream:
    features = feature_engine.process(reading['sensors'], reading['timestamp'])
    # Feed features to anomaly detection model
```

---

## Question 5: You're working on a sentiment analysis project for social media. What features would you engineer from the text data to improve the model's performance?
- [ ] Done

### Answer

**Comprehensive NLP Feature Engineering for Sentiment Analysis:**

**1. Basic Text Features:**

```python
import pandas as pd
import numpy as np
import re
from collections import Counter

def extract_basic_text_features(text):
    """
    Extract basic statistical features from text.
    """
    # Clean text
    text_lower = text.lower()
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    features = {
        # Length features
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'avg_sentence_length': len(words) / max(len(sentences), 1),
        
        # Case features
        'uppercase_count': sum(1 for c in text if c.isupper()),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'all_caps_word_count': sum(1 for w in words if w.isupper() and len(w) > 1),
        
        # Punctuation features
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'period_count': text.count('.'),
        'punctuation_ratio': sum(1 for c in text if c in '!?.,;:') / max(len(text), 1),
        
        # Special characters
        'hashtag_count': len(re.findall(r'#\w+', text)),
        'mention_count': len(re.findall(r'@\w+', text)),
        'url_count': len(re.findall(r'http\S+|www\S+', text)),
        'emoji_count': len(re.findall(r'[\U0001F600-\U0001F64F]', text)),
        
        # Repeated characters
        'repeated_chars': len(re.findall(r'(.)\1{2,}', text)),  # e.g., "sooo"
    }
    
    return features
```

**2. Sentiment Lexicon Features:**

```python
# Positive and negative word lists (subset examples)
POSITIVE_WORDS = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                  'love', 'happy', 'joy', 'best', 'awesome', 'perfect', 'beautiful'}
NEGATIVE_WORDS = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 
                  'sad', 'angry', 'poor', 'disappointing', 'ugly', 'boring'}
INTENSIFIERS = {'very', 'really', 'extremely', 'absolutely', 'totally', 'completely'}
NEGATIONS = {'not', "n't", 'no', 'never', 'neither', 'nobody', 'nothing'}

def extract_lexicon_features(text):
    """
    Extract features based on sentiment lexicons.
    """
    words = text.lower().split()
    
    positive_count = sum(1 for w in words if w in POSITIVE_WORDS)
    negative_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    intensifier_count = sum(1 for w in words if w in INTENSIFIERS)
    negation_count = sum(1 for w in words if w in NEGATIONS)
    
    features = {
        'positive_word_count': positive_count,
        'negative_word_count': negative_count,
        'sentiment_word_ratio': (positive_count - negative_count) / max(len(words), 1),
        'positive_ratio': positive_count / max(len(words), 1),
        'negative_ratio': negative_count / max(len(words), 1),
        'intensifier_count': intensifier_count,
        'negation_count': negation_count,
        'has_negation': 1 if negation_count > 0 else 0,
    }
    
    return features


# Using VADER (Valence Aware Dictionary)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def extract_vader_features(text):
    """
    Extract VADER sentiment scores.
    """
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    
    return {
        'vader_positive': scores['pos'],
        'vader_negative': scores['neg'],
        'vader_neutral': scores['neu'],
        'vader_compound': scores['compound'],
    }
```

**3. N-gram and TF-IDF Features:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def create_ngram_features(texts, max_features=5000, ngram_range=(1, 2)):
    """
    Create TF-IDF features from n-grams.
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        min_df=5,
        max_df=0.95
    )
    
    tfidf_matrix = tfidf.fit_transform(texts)
    
    return tfidf_matrix, tfidf


# Character n-grams (robust to typos)
def create_char_ngram_features(texts, max_features=3000, ngram_range=(2, 5)):
    """
    Create character n-gram features.
    """
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        max_features=max_features,
        ngram_range=ngram_range
    )
    
    return char_vectorizer.fit_transform(texts), char_vectorizer
```

**4. Word Embedding Features:**

```python
import numpy as np

def get_word_embeddings(text, word2vec_model):
    """
    Get average word embedding for text.
    """
    words = text.lower().split()
    word_vectors = []
    
    for word in words:
        if word in word2vec_model:
            word_vectors.append(word2vec_model[word])
    
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)


def extract_embedding_features(texts, word2vec_model):
    """
    Create embedding-based features for all texts.
    """
    embeddings = np.array([get_word_embeddings(text, word2vec_model) for text in texts])
    return embeddings


# Using pre-trained embeddings (example with sentence-transformers)
from sentence_transformers import SentenceTransformer

def get_sentence_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """
    Get sentence embeddings using transformer models.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings
```

**5. Part-of-Speech and Syntactic Features:**

```python
import spacy
nlp = spacy.load('en_core_web_sm')

def extract_pos_features(text):
    """
    Extract part-of-speech features.
    """
    doc = nlp(text)
    
    # POS counts
    pos_counts = Counter([token.pos_ for token in doc])
    
    features = {
        'noun_count': pos_counts.get('NOUN', 0),
        'verb_count': pos_counts.get('VERB', 0),
        'adj_count': pos_counts.get('ADJ', 0),
        'adv_count': pos_counts.get('ADV', 0),
        'pron_count': pos_counts.get('PRON', 0),
        
        # Ratios
        'noun_ratio': pos_counts.get('NOUN', 0) / max(len(doc), 1),
        'adj_ratio': pos_counts.get('ADJ', 0) / max(len(doc), 1),
        
        # Entity counts
        'entity_count': len(doc.ents),
        'person_entity_count': sum(1 for ent in doc.ents if ent.label_ == 'PERSON'),
        'org_entity_count': sum(1 for ent in doc.ents if ent.label_ == 'ORG'),
    }
    
    return features
```

**6. Social Media Specific Features:**

```python
def extract_social_media_features(text):
    """
    Extract features specific to social media text.
    """
    # Emoji sentiment (simplified mapping)
    positive_emojis = ['😀', '😊', '😃', '❤️', '👍', '🎉', '😍', '🙂']
    negative_emojis = ['😢', '😡', '😠', '👎', '💔', '😞', '😤', '🙁']
    
    pos_emoji_count = sum(text.count(e) for e in positive_emojis)
    neg_emoji_count = sum(text.count(e) for e in negative_emojis)
    
    features = {
        'positive_emoji_count': pos_emoji_count,
        'negative_emoji_count': neg_emoji_count,
        'emoji_sentiment': (pos_emoji_count - neg_emoji_count) / max(pos_emoji_count + neg_emoji_count, 1),
        
        # Slang and abbreviations
        'lol_count': text.lower().count('lol'),
        'omg_count': text.lower().count('omg'),
        
        # Elongated words (emphasis)
        'elongated_word_count': len(re.findall(r'\b\w*(.)\1{2,}\w*\b', text)),
        
        # All caps (shouting)
        'caps_word_ratio': sum(1 for w in text.split() if w.isupper()) / max(len(text.split()), 1),
    }
    
    return features
```

**Complete Feature Engineering Pipeline:**

```python
class SentimentFeatureEngineering:
    """
    Complete feature engineering pipeline for sentiment analysis.
    """
    
    def __init__(self, use_embeddings=True):
        self.use_embeddings = use_embeddings
        self.tfidf = None
        self.vader = SentimentIntensityAnalyzer()
    
    def fit(self, texts):
        """Fit feature extractors on training data."""
        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
        self.tfidf.fit(texts)
        return self
    
    def transform(self, texts):
        """Extract all features from texts."""
        features_list = []
        
        for text in texts:
            features = {}
            
            # Basic features
            features.update(extract_basic_text_features(text))
            
            # Lexicon features
            features.update(extract_lexicon_features(text))
            
            # VADER features
            features.update(extract_vader_features(text))
            
            # Social media features
            features.update(extract_social_media_features(text))
            
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Add TF-IDF features
        if self.tfidf:
            tfidf_features = self.tfidf.transform(texts).toarray()
            tfidf_df = pd.DataFrame(
                tfidf_features, 
                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            )
            features_df = pd.concat([features_df, tfidf_df], axis=1)
        
        return features_df
    
    def fit_transform(self, texts):
        """Fit and transform."""
        return self.fit(texts).transform(texts)


# Usage
fe = SentimentFeatureEngineering()
X_train_features = fe.fit_transform(train_texts)
X_test_features = fe.transform(test_texts)
```

**Feature Summary:**

| Category | Features | Purpose |
|----------|----------|---------|
| **Basic** | Length, case, punctuation | Capture writing style |
| **Lexicon** | Positive/negative word counts | Direct sentiment indicators |
| **VADER** | Compound score | Pre-trained sentiment |
| **N-gram** | TF-IDF vectors | Capture phrases and context |
| **Embedding** | Word2Vec, BERT | Semantic meaning |
| **POS** | Adjective/adverb counts | Grammatical patterns |
| **Social** | Emojis, hashtags | Platform-specific signals |

---
