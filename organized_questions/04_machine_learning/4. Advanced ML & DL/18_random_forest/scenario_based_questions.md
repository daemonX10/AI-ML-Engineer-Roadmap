# Random Forest Interview Questions - Scenario-Based Questions

## Question 1
- [ ] Done

**How would you use Random Forest for a real-time recommendation system?**

### Answer

**Definition:**
Random Forest can power recommendation systems by learning user-item preferences from features, though real-time constraints require careful optimization for prediction speed.

**Architecture:**

```
[User Features] + [Item Features] + [Context]
                    ↓
              [Random Forest]
                    ↓
              [Score/Ranking]
                    ↓
            [Top-N Recommendations]
```

**Implementation:**

**1. Feature Engineering:**
```python
def create_recommendation_features(user, item, context):
    return {
        # User features
        'user_age_group': user['age_group'],
        'user_purchase_history_count': user['total_purchases'],
        'user_avg_rating': user['avg_rating_given'],
        
        # Item features
        'item_category': item['category'],
        'item_price_bucket': item['price_bucket'],
        'item_popularity': item['view_count'],
        
        # User-Item interaction features
        'user_category_affinity': get_category_affinity(user, item),
        'similar_items_purchased': count_similar_purchases(user, item),
        
        # Context
        'time_of_day': context['hour'],
        'day_of_week': context['dayofweek'],
        'device': context['device_type']
    }
```

**2. Training:**
```python
from sklearn.ensemble import RandomForestClassifier

# Target: 1 if user engaged (clicked/purchased), 0 otherwise
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,  # Limit depth for speed
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
```

**3. Real-time Prediction Optimization:**
```python
# Reduce tree count for latency
rf_fast = RandomForestClassifier(n_estimators=50, max_depth=8)

# Pre-compute item embeddings
item_features_cache = {item_id: features for item_id, features in precompute_items()}

# Batch prediction for candidate items
def get_recommendations(user, candidate_items, top_n=10):
    user_features = get_user_features(user)
    
    X_predict = np.array([
        np.concatenate([user_features, item_features_cache[item_id]])
        for item_id in candidate_items
    ])
    
    scores = rf_fast.predict_proba(X_predict)[:, 1]
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return [candidate_items[i] for i in top_indices]
```

**4. Speed Optimizations:**
- Limit n_estimators and max_depth
- Pre-filter candidates (collaborative filtering for candidates, RF for ranking)
- Cache user/item features
- Use ONNX or compiled models

**Hybrid Approach:**
- Candidate generation: Collaborative filtering (fast)
- Ranking: Random Forest (accurate)

---

## Question 2
- [ ] Done

**Describe a scenario where Random Forest could be applied to detect credit card fraud.**

### Answer

**Definition:**
Random Forest can detect credit card fraud by learning patterns from historical transaction features (amount, time, location, etc.) to classify transactions as legitimate or fraudulent.

**Scenario Setup:**

**Features:**
- Transaction amount
- Time since last transaction
- Distance from last transaction location
- Merchant category
- Card-present vs card-not-present
- Velocity features (transactions per hour)
- Historical spending patterns

**Challenges & Solutions:**

| Challenge | RF Solution |
|-----------|-------------|
| **Class Imbalance** (0.1% fraud) | Use class_weight='balanced' or SMOTE |
| **Real-time Prediction** | Pre-trained model, fast inference |
| **Feature Engineering** | RF handles raw + engineered features |
| **Concept Drift** | Retrain periodically |

**Implementation:**

```python
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Handle imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train RF
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=5,
    class_weight='balanced',
    n_jobs=-1
)
rf.fit(X_resampled, y_resampled)

# Predict probabilities for threshold tuning
proba = rf.predict_proba(X_test)[:, 1]
# Use threshold that maximizes recall while maintaining precision
```

**Evaluation Metrics:**
- Precision-Recall curve (not accuracy)
- F1 score or F2 score (emphasize recall)
- Cost-based: Cost of false negative >> false positive

**Why RF Works Well:**
- Handles mixed feature types
- Captures non-linear fraud patterns
- Provides feature importance for explainability
- OOB error for quick validation

---

## Question 3
- [ ] Done

**Explain how Random Forest might be used for customer segmentation.**

### Answer

**Definition:**
Random Forest can be used for customer segmentation through its proximity matrix (unsupervised clustering) or by predicting customer value tiers/segments as a supervised classification task.

**Approach 1: Proximity-Based Clustering (Unsupervised)**

1. Train RF on a related supervised task (e.g., predict purchase behavior)
2. Extract proximity matrix
3. Use (1 - proximity) as distance matrix
4. Apply hierarchical clustering or MDS visualization

```python
from sklearn.ensemble import RandomForestClassifier
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.manifold import MDS

# Train RF (even dummy target works)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_customers, y_dummy)

# Get proximity matrix
leaf_indices = rf.apply(X_customers)
proximity = compute_proximity_matrix(leaf_indices)

# Cluster
distance = 1 - proximity
linkage_matrix = linkage(distance, method='ward')
segments = fcluster(linkage_matrix, t=5, criterion='maxclust')
```

**Approach 2: Supervised Segmentation**

If you have labeled segments (e.g., High-Value, Medium, Low):

```python
# Features: RFM, demographics, behavior
# Target: Customer segment label

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_segment_labels)

# Segment new customers
new_customer_segments = rf.predict(X_new_customers)

# Understand segments via feature importance
importance = rf.feature_importances_
```

**Features for Segmentation:**
- RFM (Recency, Frequency, Monetary)
- Demographic data
- Purchase categories
- Engagement metrics
- Channel preferences

**Why RF for Segmentation:**
- Handles mixed feature types
- Non-linear segment boundaries
- Feature importance reveals segment drivers
- Robust to outliers

---

## Question 4
- [ ] Done

**How would you apply Random Forest for predictive maintenance in manufacturing?**

### Answer

**Definition:**
Predictive maintenance uses Random Forest to predict equipment failures before they occur by learning patterns from sensor data, maintenance logs, and operational parameters.

**Problem Setup:**

| Component | Description |
|-----------|-------------|
| **Target** | Time to failure (regression) or Will fail soon? (classification) |
| **Features** | Sensor readings, operational metrics, maintenance history |
| **Goal** | Predict failures, schedule maintenance proactively |

**Implementation:**

**1. Feature Engineering:**
```python
def create_maintenance_features(machine_id, timestamp, window='7d'):
    sensor_data = get_sensor_data(machine_id, timestamp, window)
    
    features = {
        # Current readings
        'temperature': sensor_data['temperature'].iloc[-1],
        'vibration': sensor_data['vibration'].iloc[-1],
        'pressure': sensor_data['pressure'].iloc[-1],
        
        # Rolling statistics (trend detection)
        'temp_mean_7d': sensor_data['temperature'].mean(),
        'temp_std_7d': sensor_data['temperature'].std(),
        'temp_slope': calculate_slope(sensor_data['temperature']),
        
        'vibration_max_7d': sensor_data['vibration'].max(),
        'vibration_spike_count': count_spikes(sensor_data['vibration']),
        
        # Machine metadata
        'operating_hours': get_operating_hours(machine_id),
        'days_since_maintenance': days_since_last_maintenance(machine_id),
        'maintenance_count': get_maintenance_count(machine_id),
        
        # Environmental
        'ambient_temp': get_ambient_temp(timestamp),
        'production_load': get_production_load(machine_id, timestamp)
    }
    return features
```

**2. Model Training:**
```python
from sklearn.ensemble import RandomForestClassifier

# Classification: Will machine fail in next 7 days?
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=10,
    class_weight='balanced',  # Failures are rare
    random_state=42
)
rf.fit(X_train, y_train)

# Or Regression: Days until failure
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=200)
rf_reg.fit(X_train, y_rul)  # RUL = Remaining Useful Life
```

**3. Deployment:**
```python
def daily_prediction_job():
    for machine in machines:
        features = create_maintenance_features(machine.id, today)
        
        failure_prob = rf.predict_proba([features])[0, 1]
        
        if failure_prob > 0.7:
            alert_maintenance_team(machine, priority='HIGH')
        elif failure_prob > 0.4:
            schedule_inspection(machine)
```

**4. Feature Importance for Root Cause:**
```python
# Identify failure drivers
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Top features indicate failure causes
# e.g., "vibration_spike_count" → bearing wear
```

**Business Impact:**
- Reduce unplanned downtime (predict failures)
- Optimize maintenance scheduling
- Extend equipment life
- Lower maintenance costs

---

## Question 5
- [ ] Done

**Discuss the use of Random Forest in natural language processing (NLP) applications.**

### Answer

**Definition:**
Random Forest can be applied to NLP tasks like text classification, sentiment analysis, and named entity recognition when text is converted to numerical features (TF-IDF, embeddings). It's simpler than deep learning but effective for many tasks.

**NLP Applications:**

**1. Text Classification (e.g., Spam Detection):**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Create pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('rf', RandomForestClassifier(n_estimators=200, n_jobs=-1))
])

text_clf.fit(X_train_text, y_train)
predictions = text_clf.predict(X_test_text)
```

**2. Sentiment Analysis:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Features: TF-IDF + custom features
vectorizer = TfidfVectorizer(max_features=3000)
X_tfidf = vectorizer.fit_transform(texts)

# Add custom features
custom_features = pd.DataFrame({
    'text_length': [len(t) for t in texts],
    'exclamation_count': [t.count('!') for t in texts],
    'caps_ratio': [sum(c.isupper() for c in t)/len(t) for t in texts],
    'positive_words': [count_positive(t) for t in texts],
    'negative_words': [count_negative(t) for t in texts]
})

# Combine features
X_combined = hstack([X_tfidf, custom_features.values])

rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_combined, y_sentiment)
```

**3. Topic Classification:**
```python
# Multi-class classification
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(topics)  # ['sports', 'politics', 'tech', ...]

rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_tfidf, y_encoded)
```

**4. Using Pre-trained Embeddings:**
```python
from sentence_transformers import SentenceTransformer

# Get embeddings (more semantic than TF-IDF)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)

# Train RF on embeddings
rf = RandomForestClassifier(n_estimators=200)
rf.fit(embeddings, labels)
```

**RF vs Deep Learning for NLP:**

| Aspect | Random Forest | Deep Learning |
|--------|---------------|---------------|
| **Training Data Needed** | Works with smaller datasets | Needs large datasets |
| **Training Time** | Fast | Slow |
| **Interpretability** | Feature importance available | Black box |
| **Sequence Understanding** | Poor (bag of words) | Excellent (transformers) |
| **Performance** | Good for simple tasks | Superior for complex tasks |

**When to Use RF for NLP:**
- Small to medium datasets
- Simple classification tasks
- Need interpretability
- Quick baseline model
- Combined with embeddings for competitive results

---

