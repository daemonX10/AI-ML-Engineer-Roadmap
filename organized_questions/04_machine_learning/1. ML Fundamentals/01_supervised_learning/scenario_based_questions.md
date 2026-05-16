# Supervised Learning - Scenario-Based Questions

## Question 1: How would you design a supervised learning model for predicting customer churn?
- [ ] Done

### Problem Setup

- **Task**: Binary classification - will customer churn?
- **Target**: 1 = churned, 0 = retained
- **Challenge**: Imbalanced (churners are minority)

### Design Pipeline

#### Step 1: Define Churn

Be specific: "Customer who cancels subscription within 30 days"

#### Step 2: Feature Engineering

| Category | Features |
|----------|----------|
| **Demographics** | Age, location, tenure |
| **Usage Behavior** | Login frequency, features used, time spent |
| **Engagement Trends** | Usage this week vs last month (decline?) |
| **Support** | Tickets raised, satisfaction scores |
| **Billing** | Plan type, payment failures |

**Key**: Create time-windowed trend features
```python
df['usage_decline'] = df['usage_last_7d'] / df['usage_last_30d']
df['days_since_last_login'] = (today - df['last_login']).days
```

#### Step 3: Model Selection

| Model | Why |
|-------|-----|
| **Logistic Regression** | Interpretable - shows churn drivers |
| **XGBoost/LightGBM** | Best accuracy for tabular data |

#### Step 4: Handle Imbalance

```python
from imblearn.over_sampling import SMOTE
# OR
model = XGBClassifier(scale_pos_weight=10)  # If 10:1 ratio
```

#### Step 5: Evaluation

- **NOT Accuracy**
- **Precision**: Of flagged customers, how many actually churn?
- **Recall**: Of actual churners, how many did we catch?
- **AUPRC**: Best single metric for imbalanced problems
- **Lift Chart**: "Top 10% flagged captures 60% of churners"

#### Step 6: Deployment

- Run weekly on all customers
- Output: churn probability score
- Action: Target high-risk customers with retention campaigns

---

## Question 2: Discuss your strategy for developing a sentiment analysis model with supervised learning.
- [ ] Done

### Two-Phase Strategy

#### Phase 1: Strong Baseline (TF-IDF + Classical ML)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train and evaluate
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

**Why start here?**
- Fast to implement
- Often surprisingly effective
- Provides benchmark to beat

#### Phase 2: State-of-the-Art (Fine-tuned BERT)

```python
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    Trainer, 
    TrainingArguments
)

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=3  # positive, negative, neutral
)

# Tokenize data
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

# Fine-tune with low learning rate
training_args = TrainingArguments(
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16
)
```

### Why BERT Is Better

| Aspect | TF-IDF | BERT |
|--------|--------|------|
| "not good" vs "good" | Same features | Different meanings |
| Word order | Ignored | Understood |
| Context | None | Full sentence context |

### Evaluation Metrics

- **Accuracy** (if balanced classes)
- **F1-Score** (per class and macro)
- **Confusion Matrix** (identify which sentiments are confused)

---

## Question 3: Propose a supervised learning approach for fraud detection in transactions.
- [ ] Done

### Challenges

1. **Extreme Imbalance**: Fraud < 0.1% of transactions
2. **High Stakes**: FN (missed fraud) is very costly
3. **Evolving Patterns**: Fraudsters adapt quickly

### Approach

#### Step 1: Feature Engineering

| Feature Type | Examples |
|-------------|----------|
| **Transaction** | Amount, time, merchant category |
| **User Behavior** | Transaction count in 24h, avg amount |
| **Anomaly Signals** | Amount deviation from user average, new merchant, unusual location |

```python
# Example features
df['txn_count_24h'] = df.groupby('user_id')['amount'].transform(
    lambda x: x.rolling('24h').count()
)
df['amount_vs_avg'] = df['amount'] / df['user_avg_amount']
df['is_new_merchant'] = ~df['merchant_id'].isin(user_past_merchants)
```

#### Step 2: Model Selection

**Best Choice**: XGBoost or LightGBM
- Handles tabular data well
- Built-in class imbalance handling
- Robust to outliers

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    is_unbalance=True,  # Handle imbalance
    n_estimators=500,
    learning_rate=0.05
)
```

#### Step 3: Validation Strategy

**Critical**: Time-based split only!

```python
# Train on past, test on future
train = df[df['date'] < cutoff_date]
test = df[df['date'] >= cutoff_date]
```

#### Step 4: Evaluation

| Metric | Why |
|--------|-----|
| **AUPRC** | Best for imbalanced; focuses on fraud class |
| **Precision @ Recall** | "What precision if we catch 80% of fraud?" |
| **Cost-based** | Weight FN more heavily than FP |

```python
from sklearn.metrics import precision_recall_curve, auc

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
auprc = auc(recall, precision)
```

#### Step 5: Threshold Selection

Choose threshold based on business tradeoff:
- Lower threshold → More fraud caught, more false alarms
- Higher threshold → Fewer false alarms, some fraud missed

#### Step 6: Deployment

1. **Real-time scoring**: Score each transaction as it occurs
2. **Action**: Block, flag for review, or request 2FA based on score
3. **Monitoring**: Retrain regularly as fraud patterns evolve

### Summary Pipeline

```
Transaction → Feature Generation → Model Score → Threshold → Action
                                                    ↓
                                         Monitor & Retrain
```

---

## Question 4: Predictive Maintenance System
- [ ] Done

**Definition:**  
Predictive maintenance uses supervised learning to predict equipment failures before they occur. It's framed as classification (failure within N days?) or regression (remaining useful life) using sensor data features.

**Approach:**

| Step | Action |
|------|--------|
| **1. Problem Definition** | Classify: Fail in next N days? or Regress: RUL |
| **2. Data Collection** | Sensor time-series, maintenance logs, machine attributes |
| **3. Feature Engineering** | Rolling statistics, trend features, FFT features |
| **4. Labeling** | Mark N-day window before failure as positive |
| **5. Model Training** | XGBoost/LightGBM (handles imbalance well) |
| **6. Deployment** | Real-time scoring, alert when threshold exceeded |

**Feature Engineering Example:**
```python
# Rolling statistics from sensor data
df['temp_mean_24h'] = df['temperature'].rolling(24).mean()
df['temp_std_24h'] = df['temperature'].rolling(24).std()
df['vibration_max_24h'] = df['vibration'].rolling(24).max()
df['temp_trend'] = df['temperature'].diff(24)  # slope
```

**Handle Imbalance:** Weighted loss, SMOTE, or use Precision-Recall AUC for evaluation.

---

## Question 5: Automating Medical Image Diagnosis
- [ ] Done

**Definition:**  
Medical image diagnosis automation uses CNNs with transfer learning to classify or segment medical images. It requires expert-labeled data, rigorous validation, interpretability (Grad-CAM), and careful deployment as a clinical decision support tool.

**Approach:**

| Step | Action |
|------|--------|
| **1. Expert Collaboration** | Define problem, label data with radiologists |
| **2. Data Preparation** | Anonymize, normalize intensity, resize, augment |
| **3. Model Development** | Transfer learning from ImageNet (ResNet, EfficientNet) |
| **4. Validation** | Patient-level split, external validation, sensitivity/specificity |
| **5. Interpretability** | Grad-CAM heatmaps to show model focus |
| **6. Deployment** | As "second reader" supporting human experts |

**Transfer Learning Strategy:**
```python
import torch
import torchvision.models as models

# Load pretrained model
model = models.resnet50(pretrained=True)

# Replace final layer for our task (binary classification)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

# Training strategy:
# 1. Freeze backbone, train classifier head
# 2. Unfreeze all, fine-tune with low learning rate
```

**Critical Considerations:**
- Split by patient ID (not image) to avoid leakage
- External validation on different hospital data
- Regulatory approval (FDA clearance for clinical use)
- Interpretability for clinician trust

---
