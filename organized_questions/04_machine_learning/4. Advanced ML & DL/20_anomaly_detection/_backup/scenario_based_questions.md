# Anomaly Detection Interview Questions - Scenario-Based Questions

---

## Question 1: How would you deal with class imbalance in a dataset for supervised anomaly detection?
- [ ] Done

### Answer

**Challenge**: Anomalies are rare (often <1%), causing classifiers to be biased toward the majority class.

**Impact of Imbalance**:

```
Imbalanced Dataset:
Normal:   █████████████████████████ 99%
Anomaly:  █                          1%

Model behavior:
- Predicts all as "normal" → 99% accuracy but 0% anomaly recall
- Decision boundary biased toward majority class
```

**Solution Strategies**:

| Strategy | Approach | When to Use |
|----------|----------|-------------|
| **Resampling** | SMOTE, undersampling | Moderate imbalance |
| **Cost-sensitive** | Higher misclassification cost for anomalies | Business-driven |
| **Ensemble** | Balanced bagging, EasyEnsemble | Large datasets |
| **Threshold adjustment** | Lower decision threshold | Post-training |
| **One-class methods** | Train only on normal | Extreme imbalance |

**Python Implementation**:

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

class ImbalancedAnomalyDetector:
    """Handle class imbalance for supervised anomaly detection."""
    
    def __init__(self, method='smote', class_weight=None):
        self.method = method
        self.class_weight = class_weight
        self.resampler = None
        self.classifier = None
    
    def fit(self, X, y):
        """Fit with imbalance handling."""
        
        if self.method == 'smote':
            # Oversample minority class
            self.resampler = SMOTE(random_state=42)
            X_resampled, y_resampled = self.resampler.fit_resample(X, y)
            self.classifier = RandomForestClassifier(random_state=42)
            self.classifier.fit(X_resampled, y_resampled)
        
        elif self.method == 'adasyn':
            # Adaptive synthetic sampling
            self.resampler = ADASYN(random_state=42)
            X_resampled, y_resampled = self.resampler.fit_resample(X, y)
            self.classifier = RandomForestClassifier(random_state=42)
            self.classifier.fit(X_resampled, y_resampled)
        
        elif self.method == 'class_weight':
            # Cost-sensitive learning
            weights = {0: 1, 1: (y == 0).sum() / (y == 1).sum()}
            self.classifier = RandomForestClassifier(class_weight=weights, random_state=42)
            self.classifier.fit(X, y)
        
        elif self.method == 'balanced_ensemble':
            # Balanced Random Forest
            self.classifier = BalancedRandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            self.classifier.fit(X, y)
        
        elif self.method == 'easy_ensemble':
            # EasyEnsemble: Multiple balanced subsets
            self.classifier = EasyEnsembleClassifier(
                n_estimators=10,
                random_state=42
            )
            self.classifier.fit(X, y)
        
        return self
    
    def predict_proba(self, X):
        """Get probability scores."""
        return self.classifier.predict_proba(X)
    
    def predict(self, X, threshold=0.5):
        """Predict with adjustable threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


def optimize_threshold_for_imbalanced(model, X_val, y_val, metric='f1'):
    """Find optimal threshold for imbalanced classification."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    probas = model.predict_proba(X_val)[:, 1]
    
    thresholds = np.linspace(0.1, 0.9, 50)
    best_threshold = 0.5
    best_score = 0
    
    for thresh in thresholds:
        preds = (probas >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_val, preds)
        elif metric == 'recall':
            score = recall_score(y_val, preds)
        elif metric == 'precision':
            score = precision_score(y_val, preds)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


# Cost-sensitive loss for neural networks
import tensorflow as tf

def create_cost_sensitive_model(input_dim, pos_weight=10):
    """Neural network with cost-sensitive loss."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Weighted binary crossentropy
    def weighted_bce(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weight = y_true * pos_weight + (1 - y_true)
        return bce * weight
    
    model.compile(optimizer='adam', loss=weighted_bce, metrics=['accuracy'])
    return model
```

**Strategy Selection Guide**:

| Imbalance Ratio | Recommended Approach |
|-----------------|---------------------|
| 1:10 | Class weights |
| 1:100 | SMOTE + class weights |
| 1:1000 | One-class methods or EasyEnsemble |
| 1:10000+ | One-class SVM or Isolation Forest |

**Interview Tip**: Always evaluate with appropriate metrics (PR-AUC, F1) not accuracy. Consider the business cost of false positives vs. false negatives.

---

## Question 2: How would you approach anomaly detection in a network security context?
- [ ] Done

### Answer

**Network Security Anomaly Detection Framework**:

```
┌─────────────────────────────────────────────────────────┐
│               NETWORK DATA SOURCES                       │
│  • NetFlow/IPFIX records                                │
│  • Packet captures (PCAP)                               │
│  • Firewall logs, IDS alerts                            │
│  • DNS queries, HTTP logs                               │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌───────────────────────────┴─────────────────────────────┐
│              FEATURE ENGINEERING                         │
│  • Traffic volume features (bytes, packets)             │
│  • Connection features (duration, ports)                │
│  • Behavioral features (patterns, sequences)            │
│  • Statistical aggregations                             │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌───────────────────────────┴─────────────────────────────┐
│              DETECTION MODELS                            │
│  • Signature-based (known attacks)                      │
│  • Anomaly-based (unknown attacks)                      │
│  • Hybrid approach                                       │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌───────────────────────────┴─────────────────────────────┐
│              ALERT & RESPONSE                            │
│  • Alert correlation and prioritization                 │
│  • Automated blocking/quarantine                        │
│  • Security analyst investigation                       │
└─────────────────────────────────────────────────────────┘
```

**Types of Network Anomalies**:

| Attack Type | Characteristics | Detection Features |
|-------------|-----------------|-------------------|
| **DDoS** | High volume, many sources | Packets/sec, unique IPs |
| **Port Scan** | Sequential port access | Unique ports per IP |
| **Brute Force** | Repeated failed logins | Failed auth count |
| **Data Exfiltration** | Large outbound transfers | Bytes out, destinations |
| **C&C Communication** | Periodic beaconing | Time regularity |

**Python Implementation**:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import time

class NetworkAnomalyDetector:
    """Network intrusion detection using anomaly detection."""
    
    def __init__(self, time_window=60):  # 60 second windows
        self.time_window = time_window
        self.detector = None
        self.scaler = StandardScaler()
        self.baseline_stats = {}
    
    def extract_features(self, netflow_records):
        """
        Extract features from network flow records.
        
        Expected fields: src_ip, dst_ip, src_port, dst_port, 
                        protocol, bytes, packets, duration, timestamp
        """
        df = pd.DataFrame(netflow_records)
        
        features = {}
        
        # Volume features
        features['total_bytes'] = df['bytes'].sum()
        features['total_packets'] = df['packets'].sum()
        features['avg_packet_size'] = features['total_bytes'] / max(features['total_packets'], 1)
        
        # Connection features
        features['unique_src_ips'] = df['src_ip'].nunique()
        features['unique_dst_ips'] = df['dst_ip'].nunique()
        features['unique_dst_ports'] = df['dst_port'].nunique()
        features['unique_connections'] = len(df[['src_ip', 'dst_ip', 'dst_port']].drop_duplicates())
        
        # Protocol distribution
        protocol_counts = df['protocol'].value_counts(normalize=True)
        features['tcp_ratio'] = protocol_counts.get('TCP', 0)
        features['udp_ratio'] = protocol_counts.get('UDP', 0)
        features['icmp_ratio'] = protocol_counts.get('ICMP', 0)
        
        # Port analysis
        features['privileged_port_ratio'] = (df['dst_port'] < 1024).mean()
        
        # Duration statistics
        features['avg_duration'] = df['duration'].mean()
        features['std_duration'] = df['duration'].std()
        
        # Ratio features
        features['bytes_per_connection'] = features['total_bytes'] / max(features['unique_connections'], 1)
        features['packets_per_connection'] = features['total_packets'] / max(features['unique_connections'], 1)
        
        return pd.Series(features)
    
    def detect_port_scan(self, df, threshold=100):
        """Detect port scanning behavior."""
        # Group by source IP, count unique destination ports
        port_counts = df.groupby('src_ip')['dst_port'].nunique()
        
        # Scanning if many ports in short time
        scanners = port_counts[port_counts > threshold].index.tolist()
        
        return scanners
    
    def detect_ddos(self, df, packet_threshold=10000, unique_src_threshold=100):
        """Detect DDoS attack patterns."""
        # Group by destination IP
        dst_stats = df.groupby('dst_ip').agg({
            'packets': 'sum',
            'src_ip': 'nunique'
        })
        
        # DDoS: High packets from many sources to single destination
        ddos_targets = dst_stats[
            (dst_stats['packets'] > packet_threshold) & 
            (dst_stats['src_ip'] > unique_src_threshold)
        ].index.tolist()
        
        return ddos_targets
    
    def detect_beaconing(self, df, regularity_threshold=0.9):
        """Detect C&C beaconing behavior."""
        # Group by src_ip, dst_ip pair
        pairs = df.groupby(['src_ip', 'dst_ip'])
        
        beaconing_pairs = []
        
        for (src, dst), group in pairs:
            if len(group) < 10:
                continue
            
            # Calculate time intervals
            timestamps = group['timestamp'].sort_values()
            intervals = timestamps.diff().dropna()
            
            if len(intervals) < 5:
                continue
            
            # Check regularity (coefficient of variation)
            cv = intervals.std() / intervals.mean() if intervals.mean() > 0 else float('inf')
            
            # Low CV = regular intervals = beaconing
            if cv < (1 - regularity_threshold):
                beaconing_pairs.append((src, dst, cv))
        
        return beaconing_pairs
    
    def train(self, historical_features):
        """Train anomaly detector on historical normal traffic."""
        X_scaled = self.scaler.fit_transform(historical_features)
        
        self.detector = IsolationForest(
            contamination=0.01,  # Expect 1% anomalies
            random_state=42
        )
        self.detector.fit(X_scaled)
        
        # Store baseline statistics
        for col in historical_features.columns:
            self.baseline_stats[col] = {
                'mean': historical_features[col].mean(),
                'std': historical_features[col].std()
            }
        
        return self
    
    def detect(self, current_features):
        """Detect anomalies in current traffic window."""
        X_scaled = self.scaler.transform(current_features.values.reshape(1, -1))
        
        prediction = self.detector.predict(X_scaled)[0]
        score = self.detector.decision_function(X_scaled)[0]
        
        # Identify which features are anomalous
        anomalous_features = []
        for col in current_features.index:
            z_score = (current_features[col] - self.baseline_stats[col]['mean']) / (
                self.baseline_stats[col]['std'] + 1e-10)
            if abs(z_score) > 3:
                anomalous_features.append((col, z_score))
        
        return {
            'is_anomaly': prediction == -1,
            'anomaly_score': score,
            'anomalous_features': anomalous_features
        }


# Real-time monitoring
class RealTimeNetworkMonitor:
    """Real-time network anomaly monitoring."""
    
    def __init__(self, detector, alert_callback=None):
        self.detector = detector
        self.alert_callback = alert_callback or print
        self.window_buffer = []
        self.alert_cooldown = {}  # Prevent alert flooding
    
    def process_flow(self, flow_record):
        """Process single flow record."""
        self.window_buffer.append(flow_record)
        
        # Process when window is full
        if len(self.window_buffer) >= 1000:
            self._process_window()
            self.window_buffer = []
    
    def _process_window(self):
        """Analyze current window for anomalies."""
        df = pd.DataFrame(self.window_buffer)
        
        # Extract features
        features = self.detector.extract_features(df)
        
        # Detect general anomalies
        result = self.detector.detect(features)
        
        if result['is_anomaly']:
            self._raise_alert('GENERAL_ANOMALY', result)
        
        # Detect specific attack patterns
        scanners = self.detector.detect_port_scan(df)
        if scanners:
            self._raise_alert('PORT_SCAN', {'scanners': scanners})
        
        ddos_targets = self.detector.detect_ddos(df)
        if ddos_targets:
            self._raise_alert('DDOS', {'targets': ddos_targets})
    
    def _raise_alert(self, alert_type, details):
        """Raise alert with cooldown to prevent flooding."""
        current_time = time.time()
        
        if alert_type in self.alert_cooldown:
            if current_time - self.alert_cooldown[alert_type] < 60:
                return  # Skip if alerted recently
        
        self.alert_cooldown[alert_type] = current_time
        self.alert_callback(f"ALERT [{alert_type}]: {details}")
```

**Interview Tip**: Emphasize the need for both signature-based (known attacks) and anomaly-based (zero-day) detection. Real systems use hybrid approaches.

---

## Question 3: Propose a method for detecting fraud in credit card transactions
- [ ] Done

### Answer

**Fraud Detection System Architecture**:

```
Transaction Flow:
                                    
Customer → Transaction → Real-time → Decision → Approve/Decline
              ↓          Scoring        ↑
           Feature                   Model
           Engine                   Ensemble
              ↓                        ↑
          Historical              Fraud Models
           Profile                    ↓
                               Rule Engine
```

**Multi-Layer Detection Approach**:

| Layer | Method | Purpose |
|-------|--------|---------|
| **Rules** | Hard limits, blacklists | Catch obvious fraud |
| **ML Models** | Supervised classifiers | Pattern-based detection |
| **Anomaly** | Unsupervised methods | Novel fraud patterns |
| **Network** | Graph analysis | Organized fraud rings |

**Python Implementation**:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionSystem:
    """Multi-layer credit card fraud detection system."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.rules_engine = RulesEngine()
        self.ml_ensemble = None
        self.anomaly_detector = None
        self.customer_profiles = {}
    
    def build_transaction_features(self, transaction, customer_history=None):
        """
        Engineer features for a transaction.
        """
        features = {}
        
        # Basic transaction features
        features['amount'] = transaction['amount']
        features['hour'] = transaction['timestamp'].hour
        features['day_of_week'] = transaction['timestamp'].dayofweek
        features['is_weekend'] = int(features['day_of_week'] >= 5)
        features['is_night'] = int(features['hour'] < 6 or features['hour'] > 22)
        
        # Merchant features
        features['merchant_category'] = transaction.get('mcc', 0)
        features['is_online'] = int(transaction.get('channel') == 'online')
        features['is_international'] = int(transaction.get('country') != 'US')
        
        # Customer profile features (if history available)
        if customer_history is not None and len(customer_history) > 0:
            # Spending patterns
            features['avg_amount'] = customer_history['amount'].mean()
            features['std_amount'] = customer_history['amount'].std()
            features['amount_zscore'] = (
                (features['amount'] - features['avg_amount']) / 
                (features['std_amount'] + 1)
            )
            
            # Velocity features
            recent = customer_history[
                customer_history['timestamp'] > 
                (transaction['timestamp'] - pd.Timedelta(hours=24))
            ]
            features['txn_count_24h'] = len(recent)
            features['amount_sum_24h'] = recent['amount'].sum()
            
            # Time since last transaction
            if len(customer_history) > 0:
                last_txn = customer_history['timestamp'].max()
                features['hours_since_last'] = (
                    transaction['timestamp'] - last_txn
                ).total_seconds() / 3600
            else:
                features['hours_since_last'] = 999
            
            # Merchant diversity
            features['unique_merchants_30d'] = (
                customer_history['merchant_id'].nunique()
            )
            
            # Is this a new merchant for customer?
            features['is_new_merchant'] = int(
                transaction['merchant_id'] not in 
                customer_history['merchant_id'].values
            )
        else:
            # No history - higher risk features
            features['avg_amount'] = features['amount']
            features['std_amount'] = 0
            features['amount_zscore'] = 0
            features['txn_count_24h'] = 1
            features['amount_sum_24h'] = features['amount']
            features['hours_since_last'] = 999
            features['unique_merchants_30d'] = 1
            features['is_new_merchant'] = 1
        
        return pd.Series(features)
    
    def train(self, X_train, y_train):
        """Train the fraud detection ensemble."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Build ensemble of diverse models
        self.ml_ensemble = {
            'rf': RandomForestClassifier(
                n_estimators=100, 
                class_weight='balanced',
                random_state=42
            ),
            'gbm': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            'nn': MLPClassifier(
                hidden_layer_sizes=(64, 32),
                random_state=42
            )
        }
        
        # Train each model
        for name, model in self.ml_ensemble.items():
            model.fit(X_scaled, y_train)
            print(f"Trained {name}")
        
        # Train anomaly detector on legitimate transactions
        from sklearn.ensemble import IsolationForest
        X_legit = X_scaled[y_train == 0]
        self.anomaly_detector = IsolationForest(contamination=0.01)
        self.anomaly_detector.fit(X_legit)
        
        return self
    
    def predict(self, transaction_features):
        """
        Predict fraud probability with multi-layer approach.
        """
        result = {
            'rule_flags': [],
            'ml_score': 0,
            'anomaly_score': 0,
            'final_decision': 'APPROVE',
            'risk_level': 'LOW'
        }
        
        # Layer 1: Rules
        rule_result = self.rules_engine.check(transaction_features)
        result['rule_flags'] = rule_result['flags']
        
        if rule_result['block']:
            result['final_decision'] = 'DECLINE'
            result['risk_level'] = 'HIGH'
            return result
        
        # Layer 2: ML Ensemble
        X = transaction_features.values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        ml_scores = []
        for name, model in self.ml_ensemble.items():
            proba = model.predict_proba(X_scaled)[0, 1]
            ml_scores.append(proba)
        
        # Ensemble: weighted average
        result['ml_score'] = np.mean(ml_scores)
        
        # Layer 3: Anomaly detection
        result['anomaly_score'] = -self.anomaly_detector.decision_function(X_scaled)[0]
        
        # Final decision logic
        if result['ml_score'] > 0.8 or result['anomaly_score'] > 0.7:
            result['final_decision'] = 'DECLINE'
            result['risk_level'] = 'HIGH'
        elif result['ml_score'] > 0.5 or result['anomaly_score'] > 0.5:
            result['final_decision'] = 'REVIEW'
            result['risk_level'] = 'MEDIUM'
        else:
            result['final_decision'] = 'APPROVE'
            result['risk_level'] = 'LOW'
        
        return result


class RulesEngine:
    """Rule-based fraud detection layer."""
    
    def __init__(self):
        self.rules = [
            ('HIGH_AMOUNT', lambda x: x.get('amount', 0) > 5000),
            ('VELOCITY', lambda x: x.get('txn_count_24h', 0) > 10),
            ('AMOUNT_SPIKE', lambda x: x.get('amount_zscore', 0) > 5),
            ('NIGHT_INTL', lambda x: x.get('is_night', 0) and x.get('is_international', 0)),
            ('NEW_CARD_HIGH', lambda x: x.get('hours_since_last', 0) > 168 and x.get('amount', 0) > 1000),
        ]
        
        self.block_rules = [
            ('BLACKLIST', lambda x: x.get('merchant_id') in self.blacklisted_merchants),
            ('EXTREME_AMOUNT', lambda x: x.get('amount', 0) > 10000),
        ]
        
        self.blacklisted_merchants = set()
    
    def check(self, features):
        """Check transaction against rules."""
        flags = []
        block = False
        
        # Check warning rules
        for rule_name, rule_func in self.rules:
            if rule_func(features):
                flags.append(rule_name)
        
        # Check blocking rules
        for rule_name, rule_func in self.block_rules:
            if rule_func(features):
                flags.append(rule_name)
                block = True
        
        return {'flags': flags, 'block': block}
```

**Key Fraud Indicators**:

| Feature | Fraud Signal |
|---------|--------------|
| Amount Z-score > 3 | Unusual spending |
| Velocity > 5 txn/hour | Card compromise |
| New merchant + high amount | Testing stolen card |
| International + night | Unusual location/time |
| Chip-less after chip | Card cloned |

**Interview Tip**: Real fraud systems need sub-100ms latency for real-time decisioning. Mention the trade-off between model complexity and inference speed.

---

## Question 4: Discuss how you would set up an anomaly detection system for monitoring industrial equipment
- [ ] Done

### Answer

**Predictive Maintenance Framework**:

```
Industrial IoT Anomaly Detection System

┌─────────────────────────────────────────────────────────┐
│                   SENSOR LAYER                           │
│  • Vibration sensors       • Temperature sensors         │
│  • Pressure sensors        • Current/voltage monitors    │
│  • Acoustic sensors        • Flow meters                 │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌───────────────────────────┴─────────────────────────────┐
│                  DATA PIPELINE                           │
│  • Real-time streaming (Kafka)                          │
│  • Time-series database (InfluxDB)                      │
│  • Feature computation (Spark Streaming)                │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌───────────────────────────┴─────────────────────────────┐
│                ANOMALY DETECTION                         │
│  • Statistical methods (control charts)                 │
│  • ML models (autoencoders, Isolation Forest)          │
│  • Physics-based models                                 │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌───────────────────────────┴─────────────────────────────┐
│                 ALERT & ACTION                           │
│  • Dashboard visualization                              │
│  • Alert prioritization                                 │
│  • Maintenance scheduling                               │
│  • Automated shutdowns (critical)                       │
└─────────────────────────────────────────────────────────┘
```

**Equipment Health Monitoring Approach**:

| Level | Method | Purpose |
|-------|--------|---------|
| **Threshold** | Simple limits | Critical safety bounds |
| **Statistical** | Control charts, SPC | Detect drift from normal |
| **ML-based** | Autoencoders, LSTM | Complex pattern detection |
| **Physics** | Degradation models | Remaining useful life |

**Python Implementation**:

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

class IndustrialAnomalyDetector:
    """Anomaly detection for industrial equipment monitoring."""
    
    def __init__(self, equipment_id, sensor_config):
        self.equipment_id = equipment_id
        self.sensor_config = sensor_config
        self.baseline = {}
        self.control_limits = {}
        self.ml_model = None
        self.scaler = StandardScaler()
    
    def compute_features(self, sensor_data):
        """
        Compute features from raw sensor data.
        
        sensor_data: DataFrame with columns for each sensor
        """
        features = {}
        
        for sensor in self.sensor_config['sensors']:
            col = sensor['name']
            data = sensor_data[col]
            
            # Statistical features
            features[f'{col}_mean'] = data.mean()
            features[f'{col}_std'] = data.std()
            features[f'{col}_max'] = data.max()
            features[f'{col}_min'] = data.min()
            features[f'{col}_range'] = data.max() - data.min()
            features[f'{col}_skew'] = stats.skew(data)
            features[f'{col}_kurtosis'] = stats.kurtosis(data)
            
            # Frequency domain features (for vibration)
            if sensor.get('type') == 'vibration':
                fft = np.fft.fft(data)
                freqs = np.fft.fftfreq(len(data), 1/sensor.get('sample_rate', 1000))
                
                # Dominant frequency
                dominant_idx = np.argmax(np.abs(fft[:len(fft)//2]))
                features[f'{col}_dominant_freq'] = abs(freqs[dominant_idx])
                
                # Energy in frequency bands
                for band_name, (low, high) in sensor.get('freq_bands', {}).items():
                    band_mask = (np.abs(freqs) >= low) & (np.abs(freqs) < high)
                    features[f'{col}_energy_{band_name}'] = np.sum(np.abs(fft[band_mask])**2)
        
        return pd.Series(features)
    
    def establish_baseline(self, historical_data, window_size='1H'):
        """
        Establish baseline from historical normal operation.
        """
        # Compute features for each time window
        features_list = []
        for start in pd.date_range(
            historical_data.index.min(), 
            historical_data.index.max(), 
            freq=window_size
        ):
            end = start + pd.Timedelta(window_size)
            window_data = historical_data[start:end]
            if len(window_data) > 100:
                features = self.compute_features(window_data)
                features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Store baseline statistics
        for col in features_df.columns:
            self.baseline[col] = {
                'mean': features_df[col].mean(),
                'std': features_df[col].std()
            }
        
        # Compute control limits (3-sigma)
        for col in features_df.columns:
            self.control_limits[col] = {
                'upper': self.baseline[col]['mean'] + 3 * self.baseline[col]['std'],
                'lower': self.baseline[col]['mean'] - 3 * self.baseline[col]['std'],
                'warning_upper': self.baseline[col]['mean'] + 2 * self.baseline[col]['std'],
                'warning_lower': self.baseline[col]['mean'] - 2 * self.baseline[col]['std']
            }
        
        # Train ML model
        X_scaled = self.scaler.fit_transform(features_df)
        self._train_autoencoder(X_scaled)
        
        return self
    
    def _train_autoencoder(self, X):
        """Train autoencoder for anomaly detection."""
        input_dim = X.shape[1]
        encoding_dim = max(input_dim // 4, 8)
        
        self.ml_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(encoding_dim, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='linear')
        ])
        
        self.ml_model.compile(optimizer='adam', loss='mse')
        self.ml_model.fit(X, X, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
        
        # Compute reconstruction error threshold
        reconstructions = self.ml_model.predict(X)
        self.reconstruction_errors = np.mean((X - reconstructions)**2, axis=1)
        self.error_threshold = np.percentile(self.reconstruction_errors, 99)
    
    def detect_anomalies(self, current_data):
        """
        Detect anomalies in current sensor readings.
        """
        features = self.compute_features(current_data)
        
        result = {
            'status': 'NORMAL',
            'alerts': [],
            'warnings': [],
            'ml_score': 0
        }
        
        # Check control limits
        for col, value in features.items():
            if col in self.control_limits:
                limits = self.control_limits[col]
                
                if value > limits['upper'] or value < limits['lower']:
                    result['alerts'].append({
                        'feature': col,
                        'value': value,
                        'limit': 'upper' if value > limits['upper'] else 'lower'
                    })
                elif value > limits['warning_upper'] or value < limits['warning_lower']:
                    result['warnings'].append({
                        'feature': col,
                        'value': value
                    })
        
        # ML-based detection
        X = self.scaler.transform(features.values.reshape(1, -1))
        reconstruction = self.ml_model.predict(X)
        reconstruction_error = np.mean((X - reconstruction)**2)
        result['ml_score'] = reconstruction_error / self.error_threshold
        
        # Determine overall status
        if result['alerts'] or result['ml_score'] > 1.5:
            result['status'] = 'CRITICAL'
        elif result['warnings'] or result['ml_score'] > 1.0:
            result['status'] = 'WARNING'
        
        return result


class VibrationAnalyzer:
    """Specialized analyzer for vibration-based fault detection."""
    
    def __init__(self, sample_rate=10000):
        self.sample_rate = sample_rate
        self.fault_signatures = {
            'imbalance': {'freq_ratio': 1.0, 'name': 'Rotor Imbalance'},
            'misalignment': {'freq_ratio': 2.0, 'name': 'Shaft Misalignment'},
            'bearing_outer': {'freq_ratio': 3.5, 'name': 'Bearing Outer Race'},
            'bearing_inner': {'freq_ratio': 5.5, 'name': 'Bearing Inner Race'},
        }
    
    def analyze(self, vibration_signal, rpm):
        """
        Analyze vibration signal for fault patterns.
        """
        # FFT analysis
        n = len(vibration_signal)
        fft_result = np.fft.fft(vibration_signal)
        freqs = np.fft.fftfreq(n, 1/self.sample_rate)
        
        # Get positive frequencies
        positive_mask = freqs >= 0
        freqs = freqs[positive_mask]
        magnitudes = np.abs(fft_result[positive_mask])
        
        # Running frequency
        running_freq = rpm / 60  # Hz
        
        # Check for fault signatures
        faults_detected = []
        
        for fault_type, signature in self.fault_signatures.items():
            expected_freq = running_freq * signature['freq_ratio']
            
            # Find peak near expected frequency
            freq_tolerance = running_freq * 0.1
            mask = (freqs >= expected_freq - freq_tolerance) & (freqs <= expected_freq + freq_tolerance)
            
            if mask.any():
                peak_magnitude = magnitudes[mask].max()
                baseline_magnitude = np.median(magnitudes)
                
                # Fault if peak is significantly above baseline
                if peak_magnitude > 5 * baseline_magnitude:
                    faults_detected.append({
                        'type': fault_type,
                        'name': signature['name'],
                        'frequency': expected_freq,
                        'magnitude': peak_magnitude
                    })
        
        return faults_detected
```

**Deployment Considerations**:

| Aspect | Recommendation |
|--------|----------------|
| Latency | Sub-second for safety-critical |
| Storage | Time-series DB (InfluxDB, TimescaleDB) |
| Visualization | Grafana dashboards |
| Alerting | Tiered: Warning → Alert → Emergency |
| Edge vs Cloud | Critical detection at edge |

**Interview Tip**: Emphasize domain expertise - understanding failure modes (bearing wear, imbalance, misalignment) is as important as ML algorithms.

---

## Question 5: Describe your approach to identifying bot behavior in web traffic data
- [ ] Done

### Answer

**Bot Detection Framework**:

```
Web Traffic Bot Detection System

┌─────────────────────────────────────────────────────────┐
│                  DATA SOURCES                            │
│  • Web server logs (Apache, Nginx)                      │
│  • CDN logs (Cloudflare, Akamai)                       │
│  • JavaScript events (mouse, keyboard)                  │
│  • API request logs                                      │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌───────────────────────────┴─────────────────────────────┐
│               FEATURE EXTRACTION                         │
│  • Request patterns (timing, sequence)                  │
│  • User agent analysis                                  │
│  • Session behavior                                     │
│  • Geographic/network features                          │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌───────────────────────────┴─────────────────────────────┐
│               DETECTION MODELS                           │
│  • Rule-based (known bot signatures)                    │
│  • Behavioral ML (human vs bot patterns)               │
│  • Anomaly detection (unusual traffic)                  │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌───────────────────────────┴─────────────────────────────┐
│               RESPONSE                                   │
│  • Block, Challenge (CAPTCHA), Allow                   │
│  • Rate limiting                                        │
│  • Honeypots                                            │
└─────────────────────────────────────────────────────────┘
```

**Bot Types and Characteristics**:

| Bot Type | Characteristics | Detection Strategy |
|----------|-----------------|-------------------|
| **Simple bots** | No JS, predictable timing | User agent, JS challenge |
| **Headless browsers** | No mouse events, fast | Mouse tracking, timing |
| **Advanced bots** | Mimic human behavior | Session analysis, ML |
| **Botnets** | Distributed, coordinated | IP clustering, timing |

**Python Implementation**:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import re

class BotDetector:
    """Web traffic bot detection system."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.ml_classifier = None
        self.anomaly_detector = None
        self.known_bot_patterns = self._load_bot_patterns()
    
    def _load_bot_patterns(self):
        """Known bot user agent patterns."""
        return [
            r'bot', r'crawler', r'spider', r'scraper',
            r'curl', r'wget', r'python-requests',
            r'headless', r'phantom', r'selenium'
        ]
    
    def extract_session_features(self, session_requests):
        """
        Extract features from a user session.
        
        session_requests: List of request records
        """
        features = {}
        
        if len(session_requests) == 0:
            return None
        
        df = pd.DataFrame(session_requests)
        
        # Request timing features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            inter_request_times = df['timestamp'].diff().dt.total_seconds().dropna()
            
            features['request_count'] = len(df)
            features['session_duration'] = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
            features['avg_inter_request_time'] = inter_request_times.mean() if len(inter_request_times) > 0 else 0
            features['std_inter_request_time'] = inter_request_times.std() if len(inter_request_times) > 1 else 0
            features['min_inter_request_time'] = inter_request_times.min() if len(inter_request_times) > 0 else 0
            
            # Timing regularity (low std = bot-like)
            if features['avg_inter_request_time'] > 0:
                features['timing_cv'] = features['std_inter_request_time'] / features['avg_inter_request_time']
            else:
                features['timing_cv'] = 0
        
        # Page sequence features
        if 'url' in df.columns:
            features['unique_pages'] = df['url'].nunique()
            features['page_diversity'] = features['unique_pages'] / len(df)
            
            # Common bot patterns: sequential page access
            features['accessed_robots_txt'] = int(df['url'].str.contains('robots.txt').any())
            features['accessed_sitemap'] = int(df['url'].str.contains('sitemap').any())
        
        # Response code analysis
        if 'status_code' in df.columns:
            features['error_rate'] = (df['status_code'] >= 400).mean()
            features['not_found_rate'] = (df['status_code'] == 404).mean()
        
        # User agent analysis
        if 'user_agent' in df.columns:
            ua = df['user_agent'].iloc[0]
            features['ua_length'] = len(ua)
            features['ua_has_browser'] = int(bool(re.search(r'chrome|firefox|safari|edge', ua.lower())))
            features['ua_is_known_bot'] = int(any(re.search(p, ua.lower()) for p in self.known_bot_patterns))
        
        # Mouse/keyboard events (if available)
        if 'mouse_events' in df.columns:
            total_mouse = df['mouse_events'].sum()
            features['mouse_events_per_page'] = total_mouse / len(df)
            features['has_mouse_activity'] = int(total_mouse > 0)
        
        return pd.Series(features)
    
    def extract_request_features(self, request):
        """Extract features from single request for real-time scoring."""
        features = {}
        
        # User agent
        ua = request.get('user_agent', '')
        features['ua_length'] = len(ua)
        features['ua_is_known_bot'] = int(any(re.search(p, ua.lower()) for p in self.known_bot_patterns))
        
        # Headers
        headers = request.get('headers', {})
        features['header_count'] = len(headers)
        features['has_accept_language'] = int('accept-language' in [h.lower() for h in headers])
        features['has_accept_encoding'] = int('accept-encoding' in [h.lower() for h in headers])
        features['has_referer'] = int('referer' in [h.lower() for h in headers])
        
        # Request characteristics
        features['request_size'] = request.get('content_length', 0)
        
        return pd.Series(features)
    
    def train(self, X_sessions, y_labels):
        """Train the bot detection model."""
        X_scaled = self.scaler.fit_transform(X_sessions)
        
        self.ml_classifier = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        self.ml_classifier.fit(X_scaled, y_labels)
        
        # Anomaly detector for unknown bot patterns
        X_human = X_scaled[y_labels == 0]
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.anomaly_detector.fit(X_human)
        
        return self
    
    def detect(self, session_features):
        """Detect if session is bot or human."""
        result = {
            'is_bot': False,
            'confidence': 0,
            'bot_type': None,
            'signals': []
        }
        
        # Rule-based checks
        if session_features.get('ua_is_known_bot', 0):
            result['signals'].append('KNOWN_BOT_UA')
        
        if session_features.get('timing_cv', 1) < 0.1:
            result['signals'].append('REGULAR_TIMING')
        
        if session_features.get('min_inter_request_time', 1) < 0.1:
            result['signals'].append('FAST_REQUESTS')
        
        if session_features.get('has_mouse_activity', 1) == 0:
            result['signals'].append('NO_MOUSE')
        
        # ML prediction
        X = session_features.values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        ml_proba = self.ml_classifier.predict_proba(X_scaled)[0, 1]
        anomaly_score = -self.anomaly_detector.decision_function(X_scaled)[0]
        
        # Combine signals
        combined_score = 0.6 * ml_proba + 0.3 * anomaly_score + 0.1 * len(result['signals']) / 5
        
        result['confidence'] = min(combined_score, 1.0)
        result['is_bot'] = result['confidence'] > 0.5
        
        # Classify bot type
        if result['is_bot']:
            if session_features.get('ua_is_known_bot', 0):
                result['bot_type'] = 'DECLARED_BOT'
            elif session_features.get('has_mouse_activity', 1) == 0:
                result['bot_type'] = 'HEADLESS_BROWSER'
            elif session_features.get('timing_cv', 1) < 0.1:
                result['bot_type'] = 'AUTOMATED_SCRIPT'
            else:
                result['bot_type'] = 'SOPHISTICATED_BOT'
        
        return result


class RealTimeBotScorer:
    """Real-time bot scoring for individual requests."""
    
    def __init__(self, detector):
        self.detector = detector
        self.session_cache = defaultdict(list)
        self.session_scores = {}
    
    def score_request(self, request, session_id):
        """Score individual request and update session."""
        # Add request to session
        self.session_cache[session_id].append(request)
        
        # Real-time features
        request_features = self.detector.extract_request_features(request)
        
        # Quick checks
        quick_score = 0
        
        if request_features.get('ua_is_known_bot', 0):
            quick_score += 0.8
        
        if request_features.get('header_count', 5) < 3:
            quick_score += 0.2
        
        if not request_features.get('has_referer', 1):
            quick_score += 0.1
        
        # Session-level analysis (every N requests)
        if len(self.session_cache[session_id]) >= 5:
            session_features = self.detector.extract_session_features(
                self.session_cache[session_id]
            )
            if session_features is not None:
                result = self.detector.detect(session_features)
                self.session_scores[session_id] = result['confidence']
                quick_score = max(quick_score, result['confidence'])
        
        return {
            'score': min(quick_score, 1.0),
            'action': 'BLOCK' if quick_score > 0.8 else 'CHALLENGE' if quick_score > 0.5 else 'ALLOW'
        }
```

**Defense Strategies**:

| Strategy | Implementation | Effectiveness |
|----------|---------------|---------------|
| CAPTCHA | Challenge suspicious sessions | High for simple bots |
| Rate limiting | Limit requests per IP/session | Medium |
| JavaScript challenges | Require JS execution | High for headless |
| Mouse tracking | Verify human movement | High for advanced |
| Honeypots | Hidden links trap bots | Medium |

**Interview Tip**: Mention that sophisticated bots evolve constantly - a good bot detection system needs continuous model updates and A/B testing of detection strategies.

---

## Question 6: How would you detect anomalies in a multi-tenant cloud system's resource utilization?
- [ ] Done

### Answer

**Multi-Tenant Cloud Anomaly Detection**:

```
Cloud Resource Monitoring Architecture

┌─────────────────────────────────────────────────────────┐
│                 DATA COLLECTION                          │
│  Per-tenant metrics: CPU, Memory, Network, Storage      │
│  System metrics: Node health, API latency               │
│  Events: Deployments, scaling, failures                 │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌───────────────────────────┴─────────────────────────────┐
│               TENANT PROFILING                           │
│  • Baseline per tenant                                  │
│  • Peer comparison (similar tenants)                    │
│  • Temporal patterns (daily, weekly)                    │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌───────────────────────────┴─────────────────────────────┐
│            ANOMALY DETECTION                             │
│  • Individual tenant anomalies                          │
│  • Cross-tenant anomalies (noisy neighbor)             │
│  • System-wide anomalies                               │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌───────────────────────────┴─────────────────────────────┐
│             ALERT & ACTION                               │
│  • Auto-scaling triggers                                │
│  • Noisy neighbor mitigation                           │
│  • Capacity planning alerts                            │
└─────────────────────────────────────────────────────────┘
```

**Anomaly Types in Multi-Tenant Systems**:

| Type | Description | Impact |
|------|-------------|--------|
| **Noisy neighbor** | One tenant affecting others | Performance degradation |
| **Resource abuse** | Excessive consumption | Cost overrun, DoS |
| **Compromised tenant** | Cryptomining, attacks | Security breach |
| **Capacity anomaly** | Unexpected demand | Service degradation |

**Python Implementation**:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

class CloudResourceAnomalyDetector:
    """Anomaly detection for multi-tenant cloud systems."""
    
    def __init__(self):
        self.tenant_baselines = {}
        self.tenant_detectors = {}
        self.global_detector = None
        self.peer_groups = {}
        self.scaler = StandardScaler()
    
    def build_tenant_profile(self, tenant_id, historical_data):
        """
        Build baseline profile for a tenant.
        
        historical_data: DataFrame with columns:
            timestamp, cpu_percent, memory_percent, network_in, network_out, 
            disk_io, request_count
        """
        profile = {}
        
        # Time-based patterns
        historical_data['hour'] = historical_data['timestamp'].dt.hour
        historical_data['dayofweek'] = historical_data['timestamp'].dt.dayofweek
        
        # Aggregate by hour of day
        hourly_stats = historical_data.groupby('hour').agg({
            'cpu_percent': ['mean', 'std'],
            'memory_percent': ['mean', 'std'],
            'network_in': ['mean', 'std'],
            'network_out': ['mean', 'std']
        })
        
        profile['hourly_baseline'] = hourly_stats
        
        # Overall statistics
        for metric in ['cpu_percent', 'memory_percent', 'network_in', 'network_out', 'disk_io']:
            profile[f'{metric}_mean'] = historical_data[metric].mean()
            profile[f'{metric}_std'] = historical_data[metric].std()
            profile[f'{metric}_p95'] = historical_data[metric].quantile(0.95)
        
        # Correlation patterns
        profile['cpu_memory_corr'] = historical_data['cpu_percent'].corr(historical_data['memory_percent'])
        
        self.tenant_baselines[tenant_id] = profile
        
        # Train tenant-specific anomaly detector
        features = historical_data[['cpu_percent', 'memory_percent', 'network_in', 'network_out', 'disk_io']]
        detector = IsolationForest(contamination=0.05, random_state=42)
        detector.fit(features)
        self.tenant_detectors[tenant_id] = detector
        
        return profile
    
    def cluster_tenants_into_peers(self, tenant_profiles):
        """Group similar tenants for peer comparison."""
        from sklearn.cluster import KMeans
        
        # Create feature matrix from profiles
        profile_features = []
        tenant_ids = []
        
        for tenant_id, profile in tenant_profiles.items():
            features = [
                profile.get('cpu_percent_mean', 0),
                profile.get('memory_percent_mean', 0),
                profile.get('network_in_mean', 0),
                profile.get('cpu_percent_std', 0)
            ]
            profile_features.append(features)
            tenant_ids.append(tenant_id)
        
        X = np.array(profile_features)
        X_scaled = StandardScaler().fit_transform(X)
        
        # Cluster into peer groups
        n_clusters = min(5, len(tenant_ids) // 10 + 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        
        # Build peer group mapping
        for tenant_id, label in zip(tenant_ids, labels):
            self.peer_groups[tenant_id] = label
        
        return self.peer_groups
    
    def detect_tenant_anomaly(self, tenant_id, current_metrics):
        """Detect anomalies for a specific tenant."""
        result = {
            'tenant_id': tenant_id,
            'is_anomaly': False,
            'anomaly_type': None,
            'details': [],
            'severity': 'LOW'
        }
        
        if tenant_id not in self.tenant_baselines:
            result['details'].append('No baseline for tenant')
            return result
        
        baseline = self.tenant_baselines[tenant_id]
        current_hour = current_metrics.get('timestamp').hour if 'timestamp' in current_metrics else 12
        
        # Check against baseline
        for metric in ['cpu_percent', 'memory_percent', 'network_in', 'network_out']:
            current_value = current_metrics.get(metric, 0)
            baseline_mean = baseline.get(f'{metric}_mean', 0)
            baseline_std = baseline.get(f'{metric}_std', 1)
            
            z_score = (current_value - baseline_mean) / (baseline_std + 1e-10)
            
            if abs(z_score) > 3:
                result['details'].append({
                    'metric': metric,
                    'current': current_value,
                    'baseline_mean': baseline_mean,
                    'z_score': z_score
                })
                result['is_anomaly'] = True
        
        # ML-based detection
        if tenant_id in self.tenant_detectors:
            features = np.array([[
                current_metrics.get('cpu_percent', 0),
                current_metrics.get('memory_percent', 0),
                current_metrics.get('network_in', 0),
                current_metrics.get('network_out', 0),
                current_metrics.get('disk_io', 0)
            ]])
            
            ml_prediction = self.tenant_detectors[tenant_id].predict(features)[0]
            if ml_prediction == -1:
                result['is_anomaly'] = True
                result['details'].append({'ml_anomaly': True})
        
        # Determine severity and type
        if result['is_anomaly']:
            high_cpu = current_metrics.get('cpu_percent', 0) > 90
            high_memory = current_metrics.get('memory_percent', 0) > 90
            high_network = current_metrics.get('network_in', 0) > baseline.get('network_in_p95', float('inf'))
            
            if high_cpu and high_memory:
                result['anomaly_type'] = 'RESOURCE_EXHAUSTION'
                result['severity'] = 'HIGH'
            elif high_network:
                result['anomaly_type'] = 'NETWORK_SPIKE'
                result['severity'] = 'MEDIUM'
            elif high_cpu:
                result['anomaly_type'] = 'CPU_SPIKE'
                result['severity'] = 'MEDIUM'
            else:
                result['anomaly_type'] = 'BEHAVIORAL_ANOMALY'
                result['severity'] = 'LOW'
        
        return result
    
    def detect_noisy_neighbor(self, all_tenant_metrics, node_metrics):
        """
        Detect noisy neighbor effect.
        
        One tenant's high usage correlating with others' performance degradation.
        """
        noisy_neighbors = []
        
        # Find tenants with high resource usage
        high_usage_tenants = []
        for tenant_id, metrics in all_tenant_metrics.items():
            if metrics.get('cpu_percent', 0) > 80 or metrics.get('memory_percent', 0) > 80:
                high_usage_tenants.append(tenant_id)
        
        # Check if node is under stress
        node_stressed = (
            node_metrics.get('cpu_percent', 0) > 85 or 
            node_metrics.get('memory_percent', 0) > 90
        )
        
        if node_stressed and high_usage_tenants:
            # Check other tenants for performance degradation
            affected_tenants = []
            for tenant_id, metrics in all_tenant_metrics.items():
                if tenant_id not in high_usage_tenants:
                    if tenant_id in self.tenant_baselines:
                        baseline = self.tenant_baselines[tenant_id]
                        # Check if response time is degraded
                        if 'response_time_ms' in metrics:
                            baseline_rt = baseline.get('response_time_ms_mean', 100)
                            if metrics['response_time_ms'] > 2 * baseline_rt:
                                affected_tenants.append(tenant_id)
            
            if affected_tenants:
                noisy_neighbors = [{
                    'noisy_tenant': nt,
                    'affected_tenants': affected_tenants,
                    'node_id': node_metrics.get('node_id')
                } for nt in high_usage_tenants]
        
        return noisy_neighbors
    
    def detect_cross_tenant_anomaly(self, all_tenant_metrics):
        """Detect anomalies by comparing across tenants."""
        anomalies = []
        
        # Build feature matrix for all tenants
        tenant_ids = list(all_tenant_metrics.keys())
        features = []
        
        for tenant_id in tenant_ids:
            metrics = all_tenant_metrics[tenant_id]
            features.append([
                metrics.get('cpu_percent', 0),
                metrics.get('memory_percent', 0),
                metrics.get('network_in', 0),
                metrics.get('network_out', 0)
            ])
        
        X = np.array(features)
        
        # Detect outliers
        if len(X) > 10:
            detector = IsolationForest(contamination=0.1)
            predictions = detector.fit_predict(X)
            
            for i, (tenant_id, pred) in enumerate(zip(tenant_ids, predictions)):
                if pred == -1:
                    anomalies.append({
                        'tenant_id': tenant_id,
                        'type': 'CROSS_TENANT_OUTLIER',
                        'metrics': all_tenant_metrics[tenant_id]
                    })
        
        return anomalies


# Real-time monitoring
class CloudAnomalyMonitor:
    """Real-time cloud resource monitoring."""
    
    def __init__(self, detector):
        self.detector = detector
        self.alert_handlers = []
        self.metrics_buffer = defaultdict(list)
    
    def process_metrics(self, metrics_batch):
        """Process batch of tenant metrics."""
        alerts = []
        
        # Group by tenant
        for metrics in metrics_batch:
            tenant_id = metrics['tenant_id']
            self.metrics_buffer[tenant_id].append(metrics)
        
        # Detect anomalies
        all_metrics = {tid: buf[-1] for tid, buf in self.metrics_buffer.items() if buf}
        
        # Per-tenant detection
        for tenant_id, metrics in all_metrics.items():
            result = self.detector.detect_tenant_anomaly(tenant_id, metrics)
            if result['is_anomaly']:
                alerts.append({
                    'type': 'TENANT_ANOMALY',
                    'data': result
                })
        
        # Cross-tenant detection
        cross_anomalies = self.detector.detect_cross_tenant_anomaly(all_metrics)
        for anomaly in cross_anomalies:
            alerts.append({
                'type': 'CROSS_TENANT_ANOMALY',
                'data': anomaly
            })
        
        return alerts
```

**Key Considerations**:

| Consideration | Implementation |
|---------------|----------------|
| Tenant isolation | Separate baselines per tenant |
| Fairness | Compare against peer groups |
| Scalability | Streaming processing |
| Privacy | No cross-tenant data leakage |
| Action | Auto-scaling, migration, throttling |

**Interview Tip**: Emphasize the multi-level approach - individual tenant, cross-tenant, and system-wide anomaly detection are all important for a robust cloud monitoring system.

---

## Question 7: Discuss recent advances in deep learning for anomaly detection
- [ ] Done

### Answer

**Recent Deep Learning Advances**:

| Approach | Method | Key Innovation |
|----------|--------|----------------|
| **Autoencoders** | VAE, AE-GAN | Probabilistic latent space |
| **Transformers** | Anomaly Transformer | Attention for temporal |
| **Self-supervised** | Contrastive learning | No labels needed |
| **Graph Neural Networks** | GNN-AD | Structural anomalies |

**1. Variational Autoencoders (VAE) for Anomaly Detection**:

```python
import tensorflow as tf
import numpy as np

class VAEAnomalyDetector:
    """Variational Autoencoder for anomaly detection."""
    
    def __init__(self, input_dim, latent_dim=32):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model = self._build_model()
    
    def _build_model(self):
        # Encoder
        encoder_inputs = tf.keras.layers.Input(shape=(self.input_dim,))
        x = tf.keras.layers.Dense(128, activation='relu')(encoder_inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        
        z_mean = tf.keras.layers.Dense(self.latent_dim)(x)
        z_log_var = tf.keras.layers.Dense(self.latent_dim)(x)
        
        # Sampling layer
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])
        
        # Decoder
        decoder_inputs = tf.keras.layers.Dense(64, activation='relu')(z)
        decoder_outputs = tf.keras.layers.Dense(128, activation='relu')(decoder_inputs)
        outputs = tf.keras.layers.Dense(self.input_dim)(decoder_outputs)
        
        # Full model
        model = tf.keras.Model(encoder_inputs, outputs)
        
        # VAE loss
        reconstruction_loss = tf.keras.losses.mse(encoder_inputs, outputs)
        reconstruction_loss *= self.input_dim
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        
        model.add_loss(vae_loss)
        model.compile(optimizer='adam')
        
        return model
    
    def fit(self, X_normal, epochs=100):
        """Train on normal data."""
        self.model.fit(X_normal, X_normal, epochs=epochs, batch_size=32, validation_split=0.1, verbose=0)
        
        # Compute threshold from training data
        reconstructions = self.model.predict(X_normal)
        self.reconstruction_errors = np.mean((X_normal - reconstructions)**2, axis=1)
        self.threshold = np.percentile(self.reconstruction_errors, 95)
    
    def detect(self, X):
        """Detect anomalies."""
        reconstructions = self.model.predict(X)
        errors = np.mean((X - reconstructions)**2, axis=1)
        return errors > self.threshold, errors
```

**2. Transformer-based Anomaly Detection**:

```python
import tensorflow as tf

class AnomalyTransformer:
    """Transformer for time-series anomaly detection."""
    
    def __init__(self, seq_len, n_features, d_model=64, n_heads=4, n_layers=2):
        self.seq_len = seq_len
        self.n_features = n_features
        self.model = self._build_model(d_model, n_heads, n_layers)
    
    def _build_model(self, d_model, n_heads, n_layers):
        inputs = tf.keras.layers.Input(shape=(self.seq_len, self.n_features))
        
        # Positional encoding
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        position_embedding = tf.keras.layers.Embedding(self.seq_len, d_model)(positions)
        
        # Project input
        x = tf.keras.layers.Dense(d_model)(inputs)
        x = x + position_embedding
        
        # Transformer blocks
        for _ in range(n_layers):
            # Multi-head attention
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=n_heads, key_dim=d_model // n_heads
            )(x, x)
            x = tf.keras.layers.LayerNormalization()(x + attn_output)
            
            # Feed-forward
            ff = tf.keras.layers.Dense(d_model * 4, activation='relu')(x)
            ff = tf.keras.layers.Dense(d_model)(ff)
            x = tf.keras.layers.LayerNormalization()(x + ff)
        
        # Output projection
        outputs = tf.keras.layers.Dense(self.n_features)(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def fit(self, X_sequences, epochs=50):
        """Train on normal sequences."""
        # Self-supervised: predict next step
        X_input = X_sequences[:, :-1, :]
        X_target = X_sequences[:, 1:, :]
        
        self.model.fit(X_input, X_target, epochs=epochs, batch_size=32, validation_split=0.1, verbose=0)
    
    def detect(self, X_sequence):
        """Detect anomalies in sequence."""
        X_input = X_sequence[:-1].reshape(1, -1, self.n_features)
        X_target = X_sequence[1:]
        
        prediction = self.model.predict(X_input)[0]
        errors = np.mean((X_target - prediction)**2, axis=1)
        
        return errors
```

**3. Self-Supervised Contrastive Learning**:

```python
class ContrastiveAnomalyDetector:
    """Contrastive learning for anomaly detection."""
    
    def __init__(self, input_dim, embedding_dim=64, temperature=0.5):
        self.temperature = temperature
        self.encoder = self._build_encoder(input_dim, embedding_dim)
    
    def _build_encoder(self, input_dim, embedding_dim):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(embedding_dim),
            tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        ])
    
    def _augment(self, x):
        """Create augmented views of data."""
        # Add noise
        noise = tf.random.normal(tf.shape(x), stddev=0.1)
        x_aug1 = x + noise
        
        # Random masking
        mask = tf.random.uniform(tf.shape(x)) > 0.1
        x_aug2 = x * tf.cast(mask, tf.float32)
        
        return x_aug1, x_aug2
    
    def contrastive_loss(self, z1, z2):
        """NT-Xent loss."""
        batch_size = tf.shape(z1)[0]
        
        # Similarity matrix
        z = tf.concat([z1, z2], axis=0)
        sim = tf.matmul(z, z, transpose_b=True) / self.temperature
        
        # Positive pairs
        pos_mask = tf.eye(batch_size * 2, dtype=tf.bool)
        pos_mask = tf.roll(pos_mask, batch_size, axis=1)
        
        # Negative pairs (all others)
        neg_mask = ~tf.eye(batch_size * 2, dtype=tf.bool)
        
        # Loss
        exp_sim = tf.exp(sim)
        pos_sim = tf.reduce_sum(exp_sim * tf.cast(pos_mask, tf.float32), axis=1)
        neg_sim = tf.reduce_sum(exp_sim * tf.cast(neg_mask, tf.float32), axis=1)
        
        loss = -tf.reduce_mean(tf.math.log(pos_sim / (pos_sim + neg_sim)))
        
        return loss
    
    def fit(self, X_normal, epochs=100, batch_size=256):
        """Train encoder with contrastive learning."""
        optimizer = tf.keras.optimizers.Adam(1e-3)
        
        for epoch in range(epochs):
            # Sample batch
            idx = np.random.choice(len(X_normal), batch_size)
            x_batch = X_normal[idx]
            
            with tf.GradientTape() as tape:
                x_aug1, x_aug2 = self._augment(x_batch)
                z1 = self.encoder(x_aug1)
                z2 = self.encoder(x_aug2)
                loss = self.contrastive_loss(z1, z2)
            
            gradients = tape.gradient(loss, self.encoder.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))
        
        # Store normal embeddings for detection
        self.normal_embeddings = self.encoder.predict(X_normal)
    
    def detect(self, X):
        """Detect anomalies based on embedding distance."""
        embeddings = self.encoder.predict(X)
        
        # Distance to nearest normal embedding
        distances = []
        for emb in embeddings:
            dist = np.min(np.linalg.norm(self.normal_embeddings - emb, axis=1))
            distances.append(dist)
        
        return np.array(distances)
```

**4. Graph Neural Networks for Anomaly Detection**:

```python
# Simplified GNN for structural anomalies
class GNNAnomalyDetector:
    """Graph-based anomaly detection."""
    
    def compute_structural_features(self, adjacency_matrix, node_features):
        """Compute graph structural features."""
        import networkx as nx
        
        G = nx.from_numpy_array(adjacency_matrix)
        
        features = []
        for node in G.nodes():
            node_feat = {
                'degree': G.degree(node),
                'clustering': nx.clustering(G, node),
                'betweenness': nx.betweenness_centrality(G)[node],
                'pagerank': nx.pagerank(G)[node]
            }
            features.append(node_feat)
        
        return pd.DataFrame(features)
```

**Comparison of Deep Learning Methods**:

| Method | Strengths | Weaknesses | Best For |
|--------|-----------|------------|----------|
| VAE | Probabilistic, interpretable | Blurry reconstructions | Continuous data |
| Transformer | Long-range dependencies | Computational cost | Sequential data |
| Contrastive | No reconstruction needed | Requires good augmentations | High-dimensional |
| GNN | Captures structure | Requires graph structure | Network data |

**Interview Tip**: Highlight that deep learning methods shine for high-dimensional, complex data but traditional methods often work well for simpler cases with less computational cost.

---
