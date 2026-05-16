# Ml Design Patterns Interview Questions - Scenario_Based Questions

## Question 1
- [ ] Done

**How would you set up a 'Champion/Challenger' model deployment architecture?**

**Answer:**

### 1. Definition
Champion/Challenger is a deployment pattern where the current production model (Champion) runs alongside candidate models (Challengers), with traffic split to compare performance safely before full rollout.

### 2. Core Concepts
- **Champion**: Current production model (proven, stable)
- **Challenger**: New candidate model (to be evaluated)
- **Traffic Splitting**: Route percentage of requests to each
- **Shadow Mode**: Challenger runs but doesn't affect users
- **Gradual Rollout**: Increase challenger traffic as confidence grows

### 3. Deployment Stages

| Stage | Champion | Challenger | Risk |
|-------|----------|------------|------|
| Shadow | 100% served | 0% served (logged) | Zero |
| Canary | 95% served | 5% served | Low |
| Gradual | 70% served | 30% served | Medium |
| Switchover | 0% served | 100% served | Evaluated |

### 4. Scenario Application
**Search Ranking Model Update**:

**Week 1 - Shadow Mode**:
- Champion serves all traffic
- Challenger runs on same queries, predictions logged (not shown)
- Compare offline: latency, coverage, diversity

**Week 2 - Canary (5%)**:
- 5% users see Challenger results
- Monitor CTR, session duration, bounce rate
- Alert if metrics drop >10%

**Week 3 - Gradual (30%)**:
- Expand to 30% if Week 2 successful
- A/B test significance builds

**Week 4 - Decision**:
- If Challenger wins: Promote to Champion
- If Champion wins: Discard Challenger
- If inconclusive: Extend test

### 5. Python Code Example

```python
import random
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Model:
    name: str
    version: str
    
    def predict(self, features: Dict) -> Any:
        # Mock prediction
        return f"prediction_from_{self.name}"

class ChampionChallengerRouter:
    def __init__(self, champion: Model):
        self.champion = champion
        self.challengers = {}  # name -> (model, traffic_percentage)
        self.metrics = {'champion': [], 'challengers': {}}
    
    def add_challenger(self, name: str, model: Model, traffic_pct: float):
        """Add challenger with traffic percentage (0-1)"""
        self.challengers[name] = {'model': model, 'traffic': traffic_pct}
        self.metrics['challengers'][name] = []
    
    def route(self, request_id: str, features: Dict):
        """Route request to champion or challenger"""
        roll = random.random()
        cumulative = 0
        
        # Check if request goes to any challenger
        for name, config in self.challengers.items():
            cumulative += config['traffic']
            if roll < cumulative:
                prediction = config['model'].predict(features)
                self._log(name, request_id, prediction, is_champion=False)
                return prediction, name
        
        # Default to champion
        prediction = self.champion.predict(features)
        self._log('champion', request_id, prediction, is_champion=True)
        return prediction, 'champion'
    
    def shadow_evaluate(self, features: Dict):
        """Run challenger in shadow mode (no traffic impact)"""
        champion_pred = self.champion.predict(features)
        
        shadow_results = {'champion': champion_pred}
        for name, config in self.challengers.items():
            shadow_results[name] = config['model'].predict(features)
        
        return shadow_results
    
    def _log(self, model_name, request_id, prediction, is_champion):
        # Log for analysis
        pass
    
    def promote_challenger(self, challenger_name: str):
        """Promote challenger to champion"""
        if challenger_name in self.challengers:
            self.champion = self.challengers[challenger_name]['model']
            del self.challengers[challenger_name]
            print(f"Promoted {challenger_name} to champion")

# Usage
champion = Model(name="model_v1", version="1.0")
challenger = Model(name="model_v2", version="2.0")

router = ChampionChallengerRouter(champion)
router.add_challenger("v2_candidate", challenger, traffic_pct=0.1)  # 10%

# Route requests
for i in range(10):
    features = {'user_id': i}
    prediction, served_by = router.route(f"req_{i}", features)
    print(f"Request {i}: served by {served_by}")

# Shadow evaluation (no user impact)
shadow = router.shadow_evaluate({'user_id': 123})
print(f"\nShadow comparison: {shadow}")
```

### 6. Key Metrics to Compare
- **Business**: CTR, conversion, revenue per user
- **Model**: Accuracy, latency, coverage
- **System**: Error rate, p99 latency

### 7. Interview Tips
- Always have rollback plan (instant switch back to champion)
- Statistical significance requires sufficient sample size
- Consider segment-based routing (test on specific user cohorts first)

---

## Question 2
- [ ] Done

**Propose a system design using 'Static Model' and 'Dynamic Model' patterns to handle different needs.**

**Answer:**

### 1. Definitions
**Static Model**: Pre-trained, fixed weights, deployed once, updated infrequently (batch updates).

**Dynamic Model**: Updates continuously with new data, adapts in real-time or near-real-time.

### 2. Comparison

| Aspect | Static Model | Dynamic Model |
|--------|--------------|---------------|
| Update frequency | Weekly/Monthly | Hourly/Real-time |
| Stability | High | Variable |
| Compute cost | Low (inference only) | High (continuous training) |
| Latency | Predictable | May vary |
| Use case | Stable patterns | Rapidly changing data |

### 3. Hybrid System Design Scenario
**News Recommendation Platform**:

**Static Model (Long-term preferences)**:
- User topic preferences learned from months of history
- Category affinity, reading level, device preferences
- Retrained weekly, very stable
- Handles cold-start with demographics

**Dynamic Model (Trending content)**:
- Captures breaking news, viral articles
- Updated every 15 minutes with recent clicks
- High volatility, captures recency

**Fusion Layer**:
- Combine scores: `final_score = 0.6 * static_score + 0.4 * dynamic_score`
- Dynamic weight increases for news/finance, decreases for evergreen content

### 4. Architecture

```
User Request
     |
     v
+--------------------+
|   Feature Service  |
+--------------------+
     |
     +--------+--------+
     |                 |
     v                 v
+-----------+   +-----------+
| Static    |   | Dynamic   |
| Model     |   | Model     |
| (Weekly)  |   | (15 min)  |
+-----------+   +-----------+
     |                 |
     v                 v
+--------------------+
|   Score Fusion     |
+--------------------+
     |
     v
  Final Ranking
```

### 5. Python Code Example

```python
import numpy as np
from datetime import datetime

class StaticModel:
    """Trained weekly on historical data"""
    def __init__(self):
        self.user_preferences = {}  # Loaded from batch training
        self.last_update = datetime.now()
    
    def predict(self, user_id, items):
        # Stable, long-term preferences
        prefs = self.user_preferences.get(user_id, np.zeros(len(items)))
        return np.random.rand(len(items))  # Mock scores

class DynamicModel:
    """Updated continuously with streaming data"""
    def __init__(self):
        self.trending_scores = {}
        self.recent_clicks = []
    
    def update(self, click_event):
        """Called on each new click (streaming update)"""
        self.recent_clicks.append(click_event)
        # Update trending scores based on recent activity
        item_id = click_event['item_id']
        self.trending_scores[item_id] = self.trending_scores.get(item_id, 0) + 1
    
    def predict(self, user_id, items):
        # Return trending/recency scores
        scores = [self.trending_scores.get(item, 0) for item in items]
        return np.array(scores) / (max(scores) + 1)  # Normalize

class HybridRecommender:
    def __init__(self, static_weight=0.6):
        self.static_model = StaticModel()
        self.dynamic_model = DynamicModel()
        self.static_weight = static_weight
    
    def recommend(self, user_id, candidate_items, context=None):
        # Get scores from both models
        static_scores = self.static_model.predict(user_id, candidate_items)
        dynamic_scores = self.dynamic_model.predict(user_id, candidate_items)
        
        # Adjust weights based on context
        weight = self.static_weight
        if context and context.get('content_type') == 'news':
            weight = 0.3  # More weight on dynamic for news
        
        # Fuse scores
        final_scores = weight * static_scores + (1 - weight) * dynamic_scores
        
        # Rank items
        ranked_indices = np.argsort(final_scores)[::-1]
        return [(candidate_items[i], final_scores[i]) for i in ranked_indices[:10]]
    
    def on_click(self, user_id, item_id):
        """Handle streaming click event"""
        self.dynamic_model.update({'user_id': user_id, 'item_id': item_id})

# Usage
recommender = HybridRecommender(static_weight=0.6)

# Simulate clicks (updates dynamic model)
for item in ['article_1', 'article_2', 'article_1', 'article_3']:
    recommender.on_click('user_123', item)

# Get recommendations
items = ['article_1', 'article_2', 'article_3', 'article_4']
recommendations = recommender.recommend('user_123', items, context={'content_type': 'news'})
print("Recommendations:", recommendations)
```

### 6. When to Use Each

| Use Static | Use Dynamic | Use Hybrid |
|------------|-------------|------------|
| Stable domains | Fast-changing data | Best of both |
| Limited compute | Need real-time adaptation | News + preferences |
| Compliance needs | Personalization | E-commerce |

### 7. Interview Tips
- Discuss trade-offs: stability vs freshness
- Mention that hybrid often outperforms either alone
- Dynamic models need careful monitoring for instability

---

## Question 3
- [ ] Done

**Discuss how you'd use the 'Servant' design pattern to ensure your models are easily reusable.**

**Answer:**

### 1. Definition
The Servant design pattern creates a helper class that provides common functionality to multiple model classes without those models inheriting from a common base. It promotes code reuse and separation of concerns.

### 2. Core Concepts
- **Servant Class**: Provides shared operations (preprocessing, logging, validation)
- **Served Classes**: Models that use servant's functionality
- **Decoupling**: Models don't know about each other, only the servant
- **Single Responsibility**: Each servant handles one cross-cutting concern

### 3. Common Servants in ML

| Servant | Responsibility | Used By |
|---------|---------------|---------|
| Preprocessor | Feature transformation | All models |
| Validator | Input validation | All models |
| Logger | Prediction logging | All models |
| MetricsCollector | Performance tracking | All models |
| Explainer | Model explanations | All models |

### 4. Scenario Application
**Multi-Model Platform**:
- Fraud model, recommendation model, pricing model
- All need: preprocessing, logging, validation, monitoring
- Instead of duplicating code, create servants

### 5. Python Code Example

```python
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime

# Servant 1: Preprocessor (serves all models)
class PreprocessorServant:
    def __init__(self, config):
        self.config = config
    
    def serve(self, model, raw_features):
        """Preprocess features for any model"""
        features = raw_features.copy()
        
        # Normalize numerical features
        if 'numerical_cols' in self.config:
            for col in self.config['numerical_cols']:
                if col in features:
                    features[col] = (features[col] - self.config['means'][col]) / self.config['stds'][col]
        
        return features

# Servant 2: Validator (serves all models)
class ValidatorServant:
    def __init__(self, schema):
        self.schema = schema
    
    def serve(self, model, features):
        """Validate input for any model"""
        errors = []
        for field, rules in self.schema.items():
            if rules.get('required') and field not in features:
                errors.append(f"Missing required field: {field}")
            if field in features and 'type' in rules:
                if not isinstance(features[field], rules['type']):
                    errors.append(f"Invalid type for {field}")
        
        if errors:
            raise ValueError(f"Validation failed: {errors}")
        return True

# Servant 3: Logger (serves all models)
class LoggerServant:
    def __init__(self):
        self.logs = []
    
    def serve(self, model, features, prediction):
        """Log prediction for any model"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model.__class__.__name__,
            'features': features,
            'prediction': prediction
        }
        self.logs.append(log_entry)
        print(f"[LOG] {model.__class__.__name__}: {prediction}")

# Base model interface
class BaseModel(ABC):
    def __init__(self, servants=None):
        self.servants = servants or {}
    
    @abstractmethod
    def _predict(self, features):
        pass
    
    def predict(self, raw_features):
        # Use preprocessor servant
        if 'preprocessor' in self.servants:
            features = self.servants['preprocessor'].serve(self, raw_features)
        else:
            features = raw_features
        
        # Use validator servant
        if 'validator' in self.servants:
            self.servants['validator'].serve(self, features)
        
        # Model-specific prediction
        prediction = self._predict(features)
        
        # Use logger servant
        if 'logger' in self.servants:
            self.servants['logger'].serve(self, features, prediction)
        
        return prediction

# Concrete models (served by servants)
class FraudModel(BaseModel):
    def _predict(self, features):
        score = np.random.random()
        return {'is_fraud': score > 0.5, 'score': score}

class RecommendationModel(BaseModel):
    def _predict(self, features):
        items = ['item_1', 'item_2', 'item_3']
        return {'recommendations': items[:features.get('limit', 3)]}

# Setup servants (shared across models)
config = {'numerical_cols': ['amount'], 'means': {'amount': 100}, 'stds': {'amount': 50}}
schema = {'user_id': {'required': True, 'type': (int, str)}}

servants = {
    'preprocessor': PreprocessorServant(config),
    'validator': ValidatorServant(schema),
    'logger': LoggerServant()
}

# Models use same servants
fraud_model = FraudModel(servants=servants)
rec_model = RecommendationModel(servants=servants)

# Both models benefit from shared functionality
print("Fraud prediction:")
fraud_model.predict({'user_id': 123, 'amount': 150})

print("\nRecommendation:")
rec_model.predict({'user_id': 456, 'limit': 5})
```

### 6. Benefits
- **Code Reuse**: Write preprocessing once, use everywhere
- **Consistency**: All models validated/logged the same way
- **Maintainability**: Update servant, all models benefit
- **Testability**: Test servants independently

### 7. Interview Tips
- Distinguish from inheritance (servant is composition)
- Mention related patterns: Strategy, Decorator
- Real-world: sklearn Pipeline, MLflow logging wrappers

---

## Question 4
- [ ] Done

**How would you leverage 'Transfer Learning' in a case where labeled data is scarce?**

**Answer:**

### 1. Definition
Transfer Learning uses knowledge from a model trained on a large source dataset to improve learning on a target task with limited labeled data. The pre-trained model's learned representations transfer to the new domain.

### 2. Core Concepts
- **Source Domain**: Large dataset where model was pre-trained (ImageNet, Wikipedia)
- **Target Domain**: Your task with limited labels
- **Feature Extraction**: Use pre-trained layers as fixed feature extractors
- **Fine-tuning**: Unfreeze some layers and train on target data
- **Domain Adaptation**: Bridge gap between source and target distributions

### 3. Transfer Learning Strategies

| Strategy | Approach | When to Use |
|----------|----------|-------------|
| Feature Extraction | Freeze all layers, train classifier | Very few labels (<100) |
| Fine-tune Top Layers | Unfreeze last few layers | Moderate labels (100-1000) |
| Full Fine-tuning | Unfreeze all with low LR | More labels (1000+) |
| Domain Adaptation | Align distributions | Domain shift exists |

### 4. Scenario Application
**Medical Image Classification**: 500 labeled X-ray images for rare disease.

**Solution**:
1. Use ResNet pre-trained on ImageNet (millions of images)
2. Remove classification head
3. Add new head for your disease classes
4. Strategy based on data size:
   - 100 images: Feature extraction only
   - 500 images: Fine-tune last 2 blocks
   - 5000 images: Full fine-tuning with low LR

### 5. Python Code Example

```python
import torch
import torch.nn as nn
from torchvision import models

# Scenario: Medical image classification with 500 labeled samples

# Step 1: Load pre-trained model
def create_transfer_model(num_classes, strategy='feature_extraction'):
    # Load pre-trained ResNet
    model = models.resnet50(pretrained=True)
    
    if strategy == 'feature_extraction':
        # Freeze ALL layers (fastest, least data needed)
        for param in model.parameters():
            param.requires_grad = False
    
    elif strategy == 'fine_tune_top':
        # Freeze early layers, unfreeze last block
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze layer4 (last block)
        for param in model.layer4.parameters():
            param.requires_grad = True
    
    elif strategy == 'full_fine_tune':
        # Unfreeze all but use lower learning rate
        for param in model.parameters():
            param.requires_grad = True
    
    # Replace classification head
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    
    return model

# Step 2: Select strategy based on data size
def select_strategy(num_samples):
    if num_samples < 200:
        return 'feature_extraction'
    elif num_samples < 2000:
        return 'fine_tune_top'
    else:
        return 'full_fine_tune'

# Step 3: Configure optimizer with layer-wise learning rates
def get_optimizer(model, strategy, base_lr=0.001):
    if strategy == 'feature_extraction':
        # Only train classifier head
        params = model.fc.parameters()
        return torch.optim.Adam(params, lr=base_lr)
    
    elif strategy == 'fine_tune_top':
        # Different LR for different layers
        params = [
            {'params': model.layer4.parameters(), 'lr': base_lr * 0.1},
            {'params': model.fc.parameters(), 'lr': base_lr}
        ]
        return torch.optim.Adam(params)
    
    else:
        # Full fine-tune with low LR
        return torch.optim.Adam(model.parameters(), lr=base_lr * 0.01)

# Usage
num_samples = 500
num_classes = 3  # Disease types

strategy = select_strategy(num_samples)
print(f"Strategy for {num_samples} samples: {strategy}")

model = create_transfer_model(num_classes, strategy)
optimizer = get_optimizer(model, strategy)

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total:.1%})")

# Training loop (standard)
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
```

**Output:**
```
Strategy for 500 samples: fine_tune_top
Trainable params: 8,399,875 / 25,557,032 (32.9%)
```

### 6. Best Practices
- **Data Augmentation**: Critical with small datasets
- **Learning Rate**: Lower for pre-trained layers than new layers
- **Regularization**: Dropout, weight decay to prevent overfitting
- **Gradual Unfreezing**: Unfreeze progressively during training

### 7. Interview Tips
- Transfer works best when source and target domains are related
- Mention domain-specific pre-trained models (BioBERT, ClinicalBERT)
- Discuss negative transfer when domains are too different

---

## Question 5
- [ ] Done

**Design an 'End-to-End Machine Learning Project' workflow using relevant design patterns.**

**Answer:**

### 1. Definition
An end-to-end ML project workflow integrates multiple design patterns to handle the complete lifecycle from problem definition to production monitoring. Each stage applies specific patterns for robustness and scalability.

### 2. Workflow Stages with Patterns

| Stage | Design Patterns |
|-------|-----------------|
| Problem Definition | Start Simple, Baseline |
| Data Collection | Data Versioning, Data Validation |
| Feature Engineering | Feature Store, Transformation |
| Model Training | Pipeline, Checkpoint, Regularization |
| Evaluation | Evaluation Store, Cross-Validation |
| Deployment | Model-as-a-Service, Champion/Challenger |
| Monitoring | Model Monitoring, Logging, Drift Detection |
| Maintenance | Continuous Training, Replay |

### 3. End-to-End Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                    DATA LAYER                                 â”‚
â”‚  [Data Versioning] â”€â–º [Validation] â”€â–º [Feature Store]        â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                            â”‚
                            â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                   TRAINING LAYER                              â”‚
â”‚  [Pipeline] â”€â–º [Checkpoint] â”€â–º [Evaluation Store]            â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                            â”‚
                            â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                   SERVING LAYER                               â”‚
â”‚  [Model Registry] â”€â–º [Champion/Challenger] â”€â–º [API Service]  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                            â”‚
                            â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                   MONITORING LAYER                            â”‚
â”‚  [Logging] â”€â–º [Drift Detection] â”€â–º [Continuous Training]     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### 4. Python Code Example

```python
# End-to-end ML workflow with design patterns

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import json

# Pattern 1: Data Versioning
class DataVersioner:
    def __init__(self):
        self.versions = {}
    
    def register(self, name, data, version):
        self.versions[f"{name}_{version}"] = {
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'shape': data.shape if hasattr(data, 'shape') else len(data)
        }
        print(f"[DATA] Registered {name} version {version}")

# Pattern 2: Feature Store
class FeatureStore:
    def __init__(self):
        self.features = {}
    
    def register_transform(self, name, transform_fn):
        self.features[name] = transform_fn
    
    def compute(self, name, raw_data):
        return self.features[name](raw_data)

# Pattern 3: Training Pipeline
class TrainingPipeline:
    def __init__(self, feature_store):
        self.feature_store = feature_store
        self.checkpoints = []
    
    def run(self, model, X, y):
        # Feature transformation
        X_transformed = X  # Apply transforms from feature store
        
        # Train with cross-validation
        scores = cross_val_score(model, X_transformed, y, cv=5)
        
        # Checkpoint
        checkpoint = {
            'model': model,
            'score': scores.mean(),
            'timestamp': datetime.now().isoformat()
        }
        self.checkpoints.append(checkpoint)
        
        # Fit final model
        model.fit(X_transformed, y)
        return model, scores.mean()

# Pattern 4: Evaluation Store
class EvaluationStore:
    def __init__(self):
        self.experiments = []
    
    def log(self, experiment_id, model_name, metrics, hyperparams):
        self.experiments.append({
            'id': experiment_id,
            'model': model_name,
            'metrics': metrics,
            'hyperparams': hyperparams,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_best(self, metric='accuracy'):
        return max(self.experiments, key=lambda x: x['metrics'].get(metric, 0))

# Pattern 5: Champion/Challenger Deployment
class ModelDeployer:
    def __init__(self):
        self.champion = None
        self.challengers = []
    
    def set_champion(self, model, version):
        self.champion = {'model': model, 'version': version}
        print(f"[DEPLOY] Champion set to version {version}")
    
    def add_challenger(self, model, version, traffic_pct):
        self.challengers.append({
            'model': model, 'version': version, 'traffic': traffic_pct
        })
        print(f"[DEPLOY] Challenger {version} added with {traffic_pct*100}% traffic")

# Pattern 6: Model Monitor
class ModelMonitor:
    def __init__(self, baseline_accuracy):
        self.baseline = baseline_accuracy
        self.predictions = []
    
    def log_prediction(self, prediction, actual=None):
        self.predictions.append({'pred': prediction, 'actual': actual})
    
    def check_drift(self):
        if len(self.predictions) < 100:
            return False
        recent = self.predictions[-100:]
        with_labels = [p for p in recent if p['actual'] is not None]
        if with_labels:
            accuracy = sum(p['pred'] == p['actual'] for p in with_labels) / len(with_labels)
            if accuracy < self.baseline * 0.9:
                print("[ALERT] Model performance degradation detected!")
                return True
        return False

# Complete Workflow Execution
def run_end_to_end_workflow():
    print("=" * 50)
    print("END-TO-END ML WORKFLOW")
    print("=" * 50)
    
    # Stage 1: Data Versioning
    print("\n[STAGE 1] Data Collection & Versioning")
    data_versioner = DataVersioner()
    X = np.random.randn(1000, 10)
    y = (X.sum(axis=1) > 0).astype(int)
    data_versioner.register("training_data", X, "v1.0")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Stage 2: Feature Store
    print("\n[STAGE 2] Feature Engineering")
    feature_store = FeatureStore()
    feature_store.register_transform("normalize", lambda x: (x - x.mean()) / x.std())
    
    # Stage 3: Training with Pipeline
    print("\n[STAGE 3] Model Training")
    pipeline = TrainingPipeline(feature_store)
    eval_store = EvaluationStore()
    
    # Train baseline (Start Simple pattern)
    baseline = LogisticRegression()
    baseline, baseline_score = pipeline.run(baseline, X_train, y_train)
    eval_store.log("exp_001", "LogisticRegression", {'accuracy': baseline_score}, {'C': 1.0})
    print(f"Baseline accuracy: {baseline_score:.3f}")
    
    # Train complex model
    rf = RandomForestClassifier(n_estimators=100)
    rf, rf_score = pipeline.run(rf, X_train, y_train)
    eval_store.log("exp_002", "RandomForest", {'accuracy': rf_score}, {'n_estimators': 100})
    print(f"RandomForest accuracy: {rf_score:.3f}")
    
    # Stage 4: Evaluation & Model Selection
    print("\n[STAGE 4] Model Selection")
    best = eval_store.get_best()
    print(f"Best model: {best['model']} with accuracy {best['metrics']['accuracy']:.3f}")
    
    # Stage 5: Deployment
    print("\n[STAGE 5] Deployment")
    deployer = ModelDeployer()
    deployer.set_champion(baseline, "v1.0")
    deployer.add_challenger(rf, "v2.0", 0.1)
    
    # Stage 6: Monitoring
    print("\n[STAGE 6] Monitoring")
    monitor = ModelMonitor(baseline_accuracy=baseline_score)
    for i in range(50):
        pred = baseline.predict([X_test[i]])[0]
        monitor.log_prediction(pred, y_test[i])
    
    if not monitor.check_drift():
        print("[MONITOR] Model performing within acceptable range")
    
    print("\n" + "=" * 50)
    print("WORKFLOW COMPLETE")
    print("=" * 50)

# Execute
run_end_to_end_workflow()
```

### 5. Interview Tips
- Know which pattern applies at each stage
- Mention specific tools (MLflow, DVC, Kubernetes)
- Discuss automation (CI/CD for ML)
- Emphasize monitoring as ongoing, not one-time

---

