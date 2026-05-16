# Ensemble Learning Interview Questions - Scenario-Based Questions

## Question 1: Discuss the principle behind the LightGBM algorithm
- [ ] Done

### Definition
LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework that uses histogram-based learning and leaf-wise tree growth for faster training and lower memory usage while maintaining high accuracy.

### Core Innovations

**1. Histogram-Based Algorithm**
- Buckets continuous features into discrete bins
- Reduces computation: O(n × features) → O(bins × features)
- Bins typically 255 (8-bit representation)

**2. Leaf-Wise (Best-First) Tree Growth**

| Traditional (Level-Wise) | LightGBM (Leaf-Wise) |
|-------------------------|---------------------|
| Grows all leaves at same level | Grows leaf with max loss reduction |
| Balanced trees | Potentially unbalanced |
| More splits total | Fewer, more effective splits |

```
Level-wise:      Leaf-wise:
    ○                ○
   / \              / \
  ○   ○            ○   ○
 /\   /\          /\
○ ○  ○ ○         ○ ○
```

**3. Gradient-based One-Side Sampling (GOSS)**
- Keep all instances with large gradients (hard examples)
- Randomly sample instances with small gradients
- Maintains accuracy while reducing data

**4. Exclusive Feature Bundling (EFB)**
- Bundle mutually exclusive features (rarely non-zero together)
- Reduces effective number of features
- Common in sparse datasets

### When to Use LightGBM

| Scenario | Why LightGBM |
|----------|--------------|
| Large datasets | Fast training with histograms |
| High-dimensional data | EFB handles sparsity |
| Categorical features | Native categorical support |
| Production systems | Low memory footprint |

### Key Hyperparameters
```python
params = {
    'num_leaves': 31,          # Max leaves per tree (main complexity control)
    'max_depth': -1,           # No limit by default
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'min_child_samples': 20,   # Min data in leaf
    'subsample': 0.8,          # Row sampling
    'colsample_bytree': 0.8,   # Column sampling
    'reg_alpha': 0.1,          # L1
    'reg_lambda': 0.1          # L2
}
```

### Caution
- Leaf-wise growth can overfit on small datasets
- Use `num_leaves` < 2^(`max_depth`) to control

---

## Question 2: How would you approach feature selection for ensemble models?
- [ ] Done

### Definition
Feature selection for ensembles can leverage built-in importance measures, use wrapper methods with the ensemble itself, or apply filter methods before ensemble training. Tree ensembles provide natural feature importance rankings.

### Approach 1: Built-in Feature Importance

**From Random Forest/Gradient Boosting:**
```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Train model
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Get importance
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Select top features
top_features = importance.head(20)['feature'].tolist()
```

**Types of Importance:**

| Type | Description |
|------|-------------|
| **Gini/Entropy** | Reduction in impurity from splits |
| **Permutation** | Performance drop when feature shuffled |
| **SHAP** | Average contribution to predictions |

### Approach 2: Permutation Importance (More Reliable)
```python
from sklearn.inspection import permutation_importance

# Calculate permutation importance
result = permutation_importance(rf, X_val, y_val, n_repeats=10)

# Features with positive importance
important_features = [f for f, imp in zip(feature_names, result.importances_mean) 
                     if imp > 0]
```

### Approach 3: Recursive Feature Elimination
```python
from sklearn.feature_selection import RFECV

# Wrap ensemble in RFE with cross-validation
selector = RFECV(
    estimator=RandomForestClassifier(),
    step=1,
    cv=5,
    scoring='accuracy'
)
selector.fit(X_train, y_train)

# Get selected features
selected = X_train.columns[selector.support_].tolist()
```

### Approach 4: Filter Before Ensemble
1. Remove low-variance features
2. Remove highly correlated features (keep one)
3. Use mutual information or chi-squared

### Best Practice Pipeline
```
1. Remove zero/low variance features
2. Remove highly correlated features (>0.95)
3. Train initial ensemble
4. Use permutation importance for ranking
5. Select top-k features or use threshold
6. Retrain ensemble on selected features
7. Validate improvement
```

---

## Question 3: Discuss how ensemble learning can be applied in a distributed computing environment
- [ ] Done

### Definition
Distributed ensemble learning partitions data or models across multiple machines to train large-scale ensembles efficiently. Key approaches include data parallelism (same model, different data) and model parallelism (different models).

### Distribution Strategies

**1. Data Parallelism (Bagging-Friendly)**
```
       Original Data
      /     |     \
   Node1  Node2  Node3
   [Data] [Data] [Data]
   [Tree] [Tree] [Tree]
      \     |     /
     Aggregate Predictions
```

Each node:
- Gets a partition of data (or bootstrap sample)
- Trains local model independently
- Sends predictions/model to master

**2. Gradient Boosting Distribution (XGBoost/LightGBM)**
```
       Master Node
       [Gradient Calculation]
            ↓
    Broadcast to Workers
   /         |        \
Worker1   Worker2   Worker3
[Build histograms on data partition]
   \         |        /
    Collect, Find Best Split
            ↓
       Master Node
```

### Frameworks for Distributed Ensembles

| Framework | Use Case |
|-----------|----------|
| **Spark MLlib** | Random Forest, GBT on Spark clusters |
| **Dask** | Distributed sklearn-compatible |
| **XGBoost distributed** | Native multi-node training |
| **Ray** | Flexible distributed ML |

### Spark Example
```python
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

# Distributed Random Forest
rf = RandomForestClassifier(
    numTrees=100,
    maxDepth=10,
    featureSubsetStrategy='sqrt'
)

# Train on distributed data
model = rf.fit(training_data_distributed)
```

### Dask Example
```python
import dask.dataframe as dd
from dask_ml.ensemble import ParallelPostFit
from sklearn.ensemble import RandomForestClassifier

# Load distributed data
ddf = dd.read_parquet('large_dataset/')

# Wrap sklearn model for distributed prediction
rf = ParallelPostFit(RandomForestClassifier())
rf.fit(X_sample, y_sample)  # Fit on sample

# Predict on distributed data
predictions = rf.predict(ddf)
```

### Challenges
- Communication overhead between nodes
- Data shuffling costs
- Synchronization points
- Fault tolerance

### Best Practices
- Use bagging for embarrassingly parallel workloads
- Minimize data movement
- Use histogram-based methods (LightGBM) for large data
- Consider model partitioning for huge ensembles

---

## Question 4: How would you configure an ensemble model for real-time prediction in a production environment?
- [ ] Done

### Definition
Real-time ensemble deployment requires optimizing for latency, throughput, and reliability. Key strategies include model compression, parallel inference, caching, and proper infrastructure design.

### Latency Optimization Strategies

**1. Reduce Ensemble Size**
```python
# Prune ensemble to fastest subset that maintains accuracy
# Keep only most impactful trees
def prune_forest(rf, X_val, y_val, target_trees=50):
    original_trees = rf.estimators_
    # Evaluate contribution of each tree
    # Keep top performers
    rf.estimators_ = best_trees[:target_trees]
    return rf
```

**2. Model Quantization**
```python
# Convert to ONNX for faster inference
import onnx
from skl2onnx import convert_sklearn

onnx_model = convert_sklearn(rf, initial_types=initial_type)
# Use ONNX Runtime for fast inference
```

**3. Parallel Prediction**
```python
import joblib

# Predict in parallel
predictions = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(tree.predict)(X) 
    for tree in rf.estimators_
)
```

### Architecture for Production

```
Load Balancer
      ↓
[Inference Server 1] [Inference Server 2] [Inference Server 3]
      ↓                    ↓                    ↓
   [Model Cache]      [Model Cache]       [Model Cache]
      ↓                    ↓                    ↓
        →  Feature Store  ←
                ↓
          [Logging/Monitoring]
```

### Configuration Considerations

| Aspect | Configuration |
|--------|---------------|
| **Model loading** | Load once at startup, keep in memory |
| **Batch vs single** | Batch requests for throughput |
| **Timeout** | Set reasonable inference timeout |
| **Fallback** | Simpler model if ensemble fails |
| **Monitoring** | Track latency percentiles (p50, p95, p99) |

### Sample Deployment Code
```python
from fastapi import FastAPI
import numpy as np
import joblib

app = FastAPI()

# Load model once at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = joblib.load('ensemble_model.pkl')

@app.post("/predict")
async def predict(features: list):
    X = np.array(features).reshape(1, -1)
    prediction = model.predict(X)
    probability = model.predict_proba(X)
    return {
        "prediction": int(prediction[0]),
        "confidence": float(probability.max())
    }
```

### Monitoring Checklist
- [ ] Latency (p50, p95, p99)
- [ ] Throughput (requests/second)
- [ ] Error rate
- [ ] Model staleness
- [ ] Feature drift

---

## Question 5: Discuss how ensemble learning can be used to improve recommendation systems
- [ ] Done

### Definition
Recommendation system ensembles combine collaborative filtering, content-based, and deep learning approaches to leverage their complementary strengths and provide more robust, diverse recommendations.

### Ensemble Architecture for Recommendations

```
User-Item Interaction Data
         ↓
[Collaborative Filtering] [Content-Based] [Deep Learning]
    (Matrix Factor)        (TF-IDF,        (Neural CF,
                           Embeddings)      Autoencoders)
         ↓                      ↓               ↓
      Scores1               Scores2         Scores3
         ↓                      ↓               ↓
              [Blending / Stacking]
                      ↓
            Final Recommendations
```

### Ensemble Methods for Recommendations

**1. Score Blending**
```python
def ensemble_recommend(user_id, n_items=10):
    # Get scores from each model
    cf_scores = collaborative_filter.predict(user_id)      # Finds similar users
    cb_scores = content_based.predict(user_id)             # Uses item features
    dl_scores = neural_model.predict(user_id)              # Deep patterns
    
    # Weighted combination
    final_scores = (0.4 * cf_scores + 
                   0.3 * cb_scores + 
                   0.3 * dl_scores)
    
    # Return top items
    return np.argsort(final_scores)[-n_items:][::-1]
```

**2. Stacking with Meta-Learner**
```python
# Level 0: Base recommenders generate scores
X_meta = np.column_stack([
    cf_scores,
    cb_scores,
    dl_scores,
    user_features,  # Add context
    item_features
])

# Level 1: Learn optimal combination
meta_model = XGBRegressor()
meta_model.fit(X_meta_train, y_train)  # y = actual ratings/clicks
```

**3. Switching Ensemble**
```python
def switching_recommend(user_id):
    user_history_length = len(get_user_history(user_id))
    
    if user_history_length < 5:  # Cold start
        return content_based.recommend(user_id)
    elif user_history_length < 50:  # Some history
        return (cf.recommend(user_id) + 
                content_based.recommend(user_id)) / 2
    else:  # Rich history
        return collaborative_filter.recommend(user_id)
```

### Benefits of Ensemble Recommendations

| Challenge | How Ensemble Helps |
|-----------|-------------------|
| **Cold start** | Content-based for new users/items |
| **Sparsity** | Multiple views of limited data |
| **Diversity** | Different models suggest different items |
| **Coverage** | Reach more items in catalog |

### Netflix Prize Example
- Winning solution was ensemble of 100+ models
- Combined memory-based, matrix factorization, RBM
- Blending improved RMSE significantly

---

## Question 6: If model interpretability is crucial, how would you ensure ensemble models are understandable?
- [ ] Done

### Definition
Interpretable ensembles require post-hoc explanation methods (SHAP, LIME), simpler ensemble designs, or distillation into interpretable models. The choice depends on whether global understanding or individual prediction explanations are needed.

### Interpretation Levels

| Level | Question | Method |
|-------|----------|--------|
| **Global** | What features matter overall? | Feature importance |
| **Local** | Why this specific prediction? | SHAP, LIME |
| **Model** | How does the model work? | Simplified rules, distillation |

### Method 1: SHAP Values
```python
import shap

# Create explainer
explainer = shap.TreeExplainer(xgb_model)

# Global interpretation: Summary plot
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)

# Local interpretation: Single prediction
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0]
)
```

### Method 2: Feature Importance Comparison
```python
import matplotlib.pyplot as plt

# Compare different importance measures
importance_df = pd.DataFrame({
    'feature': feature_names,
    'gini': rf.feature_importances_,
    'permutation': permutation_importance(rf, X_val, y_val).importances_mean,
    'shap': np.abs(shap_values).mean(axis=0)
})

# If measures agree → confident interpretation
# If they disagree → investigate further
```

### Method 3: Model Distillation
```python
from sklearn.tree import DecisionTreeClassifier, export_text

# Train complex ensemble
ensemble = RandomForestClassifier(n_estimators=100)
ensemble.fit(X_train, y_train)

# Create "soft labels" from ensemble
soft_labels = ensemble.predict_proba(X_train)[:, 1]

# Distill into simple, interpretable tree
simple_tree = DecisionTreeClassifier(max_depth=5)
simple_tree.fit(X_train, (soft_labels > 0.5).astype(int))

# Get interpretable rules
print(export_text(simple_tree, feature_names=feature_names))
```

### Method 4: Rule Extraction
```python
from skrules import SkopeRules

# Extract interpretable rules from ensemble
rules = SkopeRules(
    max_depth_duplication=2,
    n_estimators=30,
    precision_min=0.5,
    recall_min=0.01
)
rules.fit(X_train, y_train)

# Print rules
for rule in rules.rules_:
    print(rule)
```

### Interpretation Strategy by Stakeholder

| Stakeholder | What They Need | Approach |
|-------------|---------------|----------|
| **Data Scientist** | Feature importance, interactions | SHAP summary |
| **Business User** | Why this decision? | LIME/SHAP force plot |
| **Regulator** | Auditable logic | Distilled rules |
| **End User** | Simple explanation | Top 3 contributing factors |

---

## Question 7: How would you deploy an ensemble learning model for detecting fraudulent transactions in a banking system?
- [ ] Done

### Definition
Fraud detection ensembles require real-time scoring, handling extreme class imbalance, continuous model updates, and robust monitoring. The system must balance precision (avoiding false positives) with recall (catching fraud).

### System Architecture

```
Transaction Stream
       ↓
[Feature Engineering Service]
       ↓
[Real-time Scoring API]
   /         \
[Model A]  [Model B]  ← Ensemble
   \         /
[Score Aggregation]
       ↓
[Decision Engine]
   /    |    \
Block  Alert  Allow
       ↓
[Feedback Loop]
       ↓
[Model Retraining Pipeline]
```

### Ensemble Design for Fraud

**Multi-Model Ensemble:**
```python
class FraudEnsemble:
    def __init__(self):
        self.models = {
            'xgb': XGBClassifier(scale_pos_weight=100),  # Handles imbalance
            'rf': RandomForestClassifier(class_weight='balanced'),
            'isolation': IsolationForest(),  # Anomaly detection
            'autoencoder': fraud_autoencoder  # Reconstruction error
        }
    
    def predict_fraud_score(self, transaction):
        scores = {}
        scores['xgb'] = self.models['xgb'].predict_proba(transaction)[0, 1]
        scores['rf'] = self.models['rf'].predict_proba(transaction)[0, 1]
        scores['isolation'] = -self.models['isolation'].score_samples(transaction)[0]
        scores['autoencoder'] = self.models['autoencoder'].reconstruction_error(transaction)
        
        # Aggregate with rules
        final_score = self.aggregate_scores(scores)
        return final_score
```

### Handling Class Imbalance

| Technique | Implementation |
|-----------|----------------|
| **SMOTE** | Oversample minority class |
| **Class weights** | `class_weight='balanced'` |
| **Threshold tuning** | Lower threshold for positive class |
| **Cost-sensitive** | Higher penalty for false negatives |

### Feature Engineering for Fraud
```python
# Transaction features
features = {
    'amount': transaction.amount,
    'hour_of_day': transaction.timestamp.hour,
    'day_of_week': transaction.timestamp.dayofweek,
    'merchant_category': transaction.mcc,
    
    # Aggregated features
    'txn_count_1h': count_transactions(user, hours=1),
    'avg_amount_30d': avg_amount(user, days=30),
    'distance_from_last': geo_distance(transaction, last_transaction),
    'velocity': transaction.amount / time_since_last,
    
    # Deviation features
    'amount_zscore': (transaction.amount - user_avg) / user_std
}
```

### Monitoring and Feedback
```python
# Track key metrics
metrics = {
    'precision': true_positives / (true_positives + false_positives),
    'recall': true_positives / (true_positives + false_negatives),
    'false_positive_rate': false_positives / total_legitimate,
    'latency_p99': percentile(latencies, 99)
}

# Alert on drift
if metrics['precision'] < threshold:
    alert("Model performance degraded - review needed")
```

### Deployment Considerations
- **Latency**: Score in <100ms
- **Availability**: 99.99% uptime required
- **Scalability**: Handle transaction spikes
- **Explainability**: Log reasons for blocks
- **A/B Testing**: Champion/challenger model setup

---

## Question 8: Propose an ensemble learning strategy for a large-scale image classification problem
- [ ] Done

### Definition
Large-scale image classification ensembles typically combine multiple CNN architectures (transfer learning), use test-time augmentation, and may include vision transformers. The strategy balances accuracy gains against computational costs.

### Proposed Strategy

**Architecture:**
```
Input Image
     ↓
[Augmentation Pipeline]
     ↓
[ResNet-50] [EfficientNet-B4] [ViT-Base] [ConvNeXt]
     ↓            ↓              ↓           ↓
  logits1      logits2        logits3     logits4
     ↓            ↓              ↓           ↓
         [Weighted Average / Stacking]
                    ↓
            Final Prediction
```

### Component Selection

| Model | Why Include |
|-------|-------------|
| **ResNet** | Robust baseline, well-understood |
| **EfficientNet** | Excellent accuracy/efficiency |
| **Vision Transformer** | Different inductive bias (attention) |
| **ConvNeXt** | Modern CNN, competitive with ViT |

### Implementation
```python
import torch
import torch.nn as nn
from torchvision import models

class ImageEnsemble(nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()
        
        # Load pretrained models
        self.resnet = models.resnet50(pretrained=True)
        self.efficientnet = models.efficientnet_b4(pretrained=True)
        
        # Replace final layers
        self.resnet.fc = nn.Linear(2048, num_classes)
        self.efficientnet.classifier[1] = nn.Linear(1792, num_classes)
        
        # Ensemble weights
        self.weights = weights or [0.5, 0.5]
    
    def forward(self, x):
        # Get predictions from each model
        out1 = torch.softmax(self.resnet(x), dim=1)
        out2 = torch.softmax(self.efficientnet(x), dim=1)
        
        # Weighted average
        ensemble_out = (self.weights[0] * out1 + 
                       self.weights[1] * out2)
        return ensemble_out
```

### Test-Time Augmentation (TTA)
```python
def predict_with_tta(model, image, n_augments=5):
    predictions = []
    
    # Original
    predictions.append(model(image))
    
    # Horizontal flip
    predictions.append(model(torch.flip(image, [3])))
    
    # Multiple crops
    for crop in get_random_crops(image, n=n_augments-2):
        predictions.append(model(crop))
    
    # Average predictions
    return torch.stack(predictions).mean(dim=0)
```

### Training Strategy
1. **Fine-tune each model separately** on target dataset
2. **Freeze base models**, train ensemble weights
3. **Cross-validation** to find optimal weights
4. **Validation**: Monitor ensemble vs individual performance

### Scaling Considerations

| Challenge | Solution |
|-----------|----------|
| Training time | Parallel training on multiple GPUs |
| Inference speed | Model distillation or pruning |
| Memory | Mixed precision (FP16) |
| Large dataset | Distributed data loading |

---

## Question 9: Discuss the latest research trends around ensemble learning methods
- [ ] Done

### Definition
Current research focuses on neural ensemble techniques, automated ensemble construction (AutoML), interpretability, uncertainty quantification, and efficient ensemble inference.

### Key Research Trends

**1. Deep Ensembles for Uncertainty**
- Train multiple neural networks with different initializations
- Use disagreement to quantify uncertainty
- Critical for safety-critical applications

```python
# Deep ensemble uncertainty
predictions = [model(x) for model in ensemble]
mean_pred = np.mean(predictions, axis=0)
uncertainty = np.std(predictions, axis=0)  # Epistemic uncertainty
```

**2. Neural Architecture Search (NAS) for Ensembles**
- Automatically design ensemble architectures
- Search for complementary models
- Optimize diversity-accuracy trade-off

**3. Snapshot Ensembles**
- Single training run, multiple models
- Save checkpoints during cyclic learning rate
- Near-free ensemble

```python
# Cyclic learning rate for snapshot ensemble
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)
# Save model at each minimum
```

**4. Knowledge Distillation from Ensembles**
- Compress ensemble into single model
- Student learns from ensemble "soft targets"
- Maintains much of ensemble performance

**5. Ensemble of Vision Transformers + CNNs**
- Combine attention-based and convolution-based models
- Different inductive biases complement each other

**6. Self-Ensembling**
- Dropout as ensemble (MC Dropout)
- BatchEnsemble: Efficient weight sharing
- Stochastic Weight Averaging (SWA)

### Emerging Areas

| Area | Description |
|------|-------------|
| **Federated Ensemble Learning** | Ensemble across distributed/private data |
| **Continual Learning Ensembles** | Growing ensemble for new tasks |
| **Green AI Ensembles** | Energy-efficient ensemble methods |
| **Ensemble Calibration** | Better probability estimates |

### Papers to Know
- "Deep Ensembles: A Loss Landscape Perspective" (2019)
- "Snapshot Ensembles: Train 1, Get M for Free" (2017)
- "BatchEnsemble: An Alternative Approach to Efficient Ensemble" (2020)

---

## Question 10: Discuss dynamic ensembling and its potential for adaptive learning over time
- [ ] Done

### Definition
Dynamic ensembling adapts the ensemble composition or weights based on changing data distributions or incoming feedback. This enables continuous learning without full retraining and handles concept drift effectively.

### Types of Dynamic Ensembling

**1. Dynamic Weight Adjustment**
```python
class DynamicWeightedEnsemble:
    def __init__(self, models, window_size=1000):
        self.models = models
        self.weights = [1/len(models)] * len(models)
        self.window_size = window_size
        self.recent_performance = {m: [] for m in range(len(models))}
    
    def predict(self, X):
        predictions = [m.predict(X) for m in self.models]
        weighted_pred = sum(w * p for w, p in zip(self.weights, predictions))
        return weighted_pred
    
    def update(self, X, y_true):
        # Evaluate each model on recent data
        for i, model in enumerate(self.models):
            pred = model.predict(X)
            error = np.mean((pred - y_true) ** 2)
            self.recent_performance[i].append(error)
            
            # Keep only recent window
            if len(self.recent_performance[i]) > self.window_size:
                self.recent_performance[i].pop(0)
        
        # Update weights inversely proportional to recent error
        recent_errors = [np.mean(self.recent_performance[i]) 
                        for i in range(len(self.models))]
        inverse_errors = [1/(e + 0.001) for e in recent_errors]
        total = sum(inverse_errors)
        self.weights = [ie/total for ie in inverse_errors]
```

**2. Add/Remove Models**
```python
class AdaptiveEnsemble:
    def __init__(self, base_learner_class):
        self.models = []
        self.learner_class = base_learner_class
        self.performance_threshold = 0.7
    
    def process_data_batch(self, X_new, y_new):
        # Evaluate existing models
        for model in self.models:
            score = model.score(X_new, y_new)
            if score < self.performance_threshold:
                self.models.remove(model)  # Remove poor performers
        
        # Train new model on recent data
        new_model = self.learner_class()
        new_model.fit(X_new, y_new)
        self.models.append(new_model)
        
        # Limit ensemble size
        if len(self.models) > 10:
            self.models.pop(0)  # Remove oldest
```

**3. Instance-Based Selection**
- For each new sample, select which models to use
- Based on similarity to training regions
- Different models for different input regions

### Benefits of Dynamic Ensembling

| Benefit | Description |
|---------|-------------|
| **Handles concept drift** | Adapts to changing patterns |
| **No full retraining** | Incremental updates |
| **Always current** | Recent data weighted higher |
| **Graceful degradation** | Poor models downweighted |

### Challenges

| Challenge | Mitigation |
|-----------|------------|
| Catastrophic forgetting | Keep diverse age of models |
| Computational overhead | Limit ensemble size |
| Feedback delay | Use proxy metrics |
| Noise in updates | Smooth weight changes |

### Use Cases
- Fraud detection (fraud patterns evolve)
- Stock prediction (market regimes change)
- Recommendation systems (user preferences shift)
- IoT/sensor data (environmental changes)

### Implementation Considerations
```python
# Online learning ensemble pattern
while True:
    # Get new data batch
    X_batch, y_batch = get_new_data()
    
    # Predict with current ensemble
    predictions = ensemble.predict(X_batch)
    
    # Log predictions for later evaluation
    log_predictions(predictions)
    
    # When ground truth becomes available
    if ground_truth_available():
        y_true = get_ground_truth()
        ensemble.update_weights(X_batch, y_true)
        
        # Periodically retrain poor models
        if time_to_retrain():
            ensemble.retrain_worst_model(recent_data)
```

---
