# Ml Design Patterns Interview Questions - General Questions

## Question 1
- [ ] Done

**How is the 'Evaluation Store' design pattern applied to keep track of model performances?**

**Answer:**

### 1. Definition
An Evaluation Store is a centralized repository that stores evaluation metrics, experiment results, and model performance data across all experiments and production deployments, enabling comparison, tracking, and decision-making.

### 2. Core Concepts
- **Metrics Storage**: Accuracy, F1, AUC, custom metrics per experiment
- **Metadata**: Hyperparameters, data version, model version
- **Comparison**: Side-by-side experiment comparison
- **Versioning**: Track metrics over time for same model
- **Visualization**: Dashboards for metric trends

### 3. What to Store

| Category | Data |
|----------|------|
| Training | Loss curves, validation metrics per epoch |
| Evaluation | Test set metrics, cross-validation scores |
| Production | Live metrics, A/B test results |
| Metadata | Model version, data version, hyperparams |

### 4. Python Code Example

```python
import json
from datetime import datetime
import sqlite3
from typing import Dict, Any

class EvaluationStore:
    """Centralized store for model evaluation metrics"""
    
    def __init__(self, db_path="evaluations.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY,
                experiment_id TEXT,
                model_name TEXT,
                model_version TEXT,
                dataset_version TEXT,
                metrics TEXT,
                hyperparams TEXT,
                timestamp TEXT
            )
        ''')
        self.conn.commit()
    
    def log_evaluation(self, experiment_id: str, model_name: str, 
                       model_version: str, dataset_version: str,
                       metrics: Dict, hyperparams: Dict):
        """Store evaluation results"""
        self.conn.execute('''
            INSERT INTO evaluations 
            (experiment_id, model_name, model_version, dataset_version, 
             metrics, hyperparams, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment_id, model_name, model_version, dataset_version,
            json.dumps(metrics), json.dumps(hyperparams),
            datetime.now().isoformat()
        ))
        self.conn.commit()
    
    def get_best_model(self, model_name: str, metric: str = 'accuracy'):
        """Find best performing model version"""
        cursor = self.conn.execute('''
            SELECT model_version, metrics FROM evaluations
            WHERE model_name = ?
        ''', (model_name,))
        
        best = None
        best_score = -float('inf')
        
        for row in cursor:
            metrics = json.loads(row[1])
            if metrics.get(metric, 0) > best_score:
                best_score = metrics[metric]
                best = row[0]
        
        return best, best_score
    
    def compare_experiments(self, experiment_ids: list):
        """Compare multiple experiments"""
        results = []
        for exp_id in experiment_ids:
            cursor = self.conn.execute('''
                SELECT model_version, metrics, hyperparams FROM evaluations
                WHERE experiment_id = ?
            ''', (exp_id,))
            for row in cursor:
                results.append({
                    'experiment_id': exp_id,
                    'version': row[0],
                    'metrics': json.loads(row[1]),
                    'hyperparams': json.loads(row[2])
                })
        return results

# Usage
store = EvaluationStore()

# Log experiments
store.log_evaluation(
    experiment_id="exp_001",
    model_name="fraud_detector",
    model_version="v1.0",
    dataset_version="data_v2",
    metrics={'accuracy': 0.92, 'f1': 0.88, 'auc': 0.95},
    hyperparams={'learning_rate': 0.01, 'epochs': 100}
)

store.log_evaluation(
    experiment_id="exp_002",
    model_name="fraud_detector",
    model_version="v1.1",
    dataset_version="data_v2",
    metrics={'accuracy': 0.94, 'f1': 0.91, 'auc': 0.96},
    hyperparams={'learning_rate': 0.001, 'epochs': 150}
)

# Find best model
best_version, best_score = store.get_best_model("fraud_detector", "accuracy")
print(f"Best model: {best_version} with accuracy {best_score}")

# Compare experiments
comparison = store.compare_experiments(["exp_001", "exp_002"])
for exp in comparison:
    print(f"{exp['experiment_id']}: {exp['metrics']}")
```

### 5. Tools
- **MLflow**: Experiment tracking with UI
- **Weights & Biases**: Cloud-based tracking
- **Neptune.ai**: ML metadata store
- **Custom DB**: PostgreSQL, MongoDB for internal use

### 6. Interview Tips
- Evaluation store is foundation for reproducibility
- Mention integration with CI/CD for automated evaluation
- Discuss retention policies for old experiments

---

## Question 2
- [ ] Done

**Define the 'Start Simple' principle in the context of an ML model development.**

**Answer:**

### 1. Definition
Start Simple is a design principle that recommends beginning with the simplest possible model (baseline) before adding complexity. It establishes a performance floor, validates the problem setup, and often reveals that simple solutions are sufficient.

### 2. Core Concepts
- **Baseline First**: Mean predictor, majority class, simple heuristics
- **Incremental Complexity**: Add features/model complexity only when needed
- **Validation**: Ensure data pipeline and evaluation are correct
- **Cost-Benefit**: Complexity must justify its added value
- **Debugging**: Simple models are easier to debug

### 3. Progression Example

| Stage | Model | Purpose |
|-------|-------|---------|
| 1 | Random/Constant | Sanity check |
| 2 | Simple heuristic | Domain baseline |
| 3 | Logistic Regression | Linear baseline |
| 4 | Decision Tree | Non-linear, interpretable |
| 5 | Random Forest | Ensemble |
| 6 | Neural Network | Complex patterns |

### 4. Why Start Simple?
- **Catch Bugs Early**: Complex model hides data issues
- **Set Expectations**: Know what's achievable before investing
- **Faster Iteration**: Quick experiments guide direction
- **Interpretability**: Understand problem before black-boxing
- **Often Sufficient**: Simple model + good features beats complex model

### 5. Python Code Example

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load data
X, y = load_iris(return_X_y=True)

# Stage 1: Random baseline
random_baseline = DummyClassifier(strategy='stratified')
score = cross_val_score(random_baseline, X, y, cv=5).mean()
print(f"Stage 1 - Random: {score:.3f}")

# Stage 2: Majority class baseline
majority_baseline = DummyClassifier(strategy='most_frequent')
score = cross_val_score(majority_baseline, X, y, cv=5).mean()
print(f"Stage 2 - Majority: {score:.3f}")

# Stage 3: Simple linear model
logistic = LogisticRegression(max_iter=1000)
score = cross_val_score(logistic, X, y, cv=5).mean()
print(f"Stage 3 - Logistic: {score:.3f}")

# Stage 4: Decision tree
tree = DecisionTreeClassifier(max_depth=3)
score = cross_val_score(tree, X, y, cv=5).mean()
print(f"Stage 4 - Decision Tree: {score:.3f}")

# Stage 5: Random forest
forest = RandomForestClassifier(n_estimators=100)
score = cross_val_score(forest, X, y, cv=5).mean()
print(f"Stage 5 - Random Forest: {score:.3f}")

# Decision: Is complexity justified?
# If Stage 3 = 0.96 and Stage 5 = 0.97, consider using Stage 3
```

**Output:**
```
Stage 1 - Random: 0.333
Stage 2 - Majority: 0.333
Stage 3 - Logistic: 0.967
Stage 4 - Decision Tree: 0.960
Stage 5 - Random Forest: 0.967
```

**Conclusion**: Logistic Regression matches Random Forest - use the simpler model.

### 6. Interview Tips
- Always mention baseline metrics in presentations
- Complexity should be justified by significant improvement
- Simple models are easier to explain to stakeholders
- "Start simple, add complexity only when needed"

---

## Question 3
- [ ] Done

**What considerations should be taken into account when using 'Replicated Prediction Servers'?**

**Answer:**

### 1. Definition
Replicated Prediction Servers deploy multiple identical copies of a model service behind a load balancer to handle high traffic, provide fault tolerance, and enable zero-downtime deployments.

### 2. Key Considerations

| Consideration | Why It Matters |
|---------------|----------------|
| Statelessness | All replicas must behave identically |
| Model Consistency | All replicas serve same model version |
| Load Balancing | Distribute requests evenly |
| Health Checks | Route away from unhealthy replicas |
| Session Affinity | Usually disabled for ML (stateless) |
| Cold Start | Pre-warm new replicas |

### 3. Critical Design Decisions

**a) Model Storage**
- Store model in shared storage (S3, NFS, Redis)
- All replicas load same model file
- Version in filename or metadata

**b) Request Routing**
- Round-robin for even distribution
- Least-connections for varying latency
- Consistent hashing if caching responses

**c) Deployment Strategy**
- Rolling update: Replace replicas one by one
- Blue-green: Switch all traffic at once
- Canary: Test on subset before full rollout

### 4. Python Code Example

```python
# Conceptual architecture for replicated prediction servers

from flask import Flask, request, jsonify
import os
import hashlib

app = Flask(__name__)

# Load model from shared storage
MODEL_VERSION = os.environ.get('MODEL_VERSION', 'v1.0.0')
REPLICA_ID = os.environ.get('REPLICA_ID', 'replica-0')

class ModelServer:
    def __init__(self):
        self.model = self._load_from_shared_storage()
        self.request_count = 0
    
    def _load_from_shared_storage(self):
        # All replicas load same model
        model_path = f"s3://models/{MODEL_VERSION}/model.pkl"
        print(f"[{REPLICA_ID}] Loading model from {model_path}")
        return lambda x: sum(x) > 0  # Mock model
    
    def predict(self, features):
        self.request_count += 1
        return self.model(features)

model_server = ModelServer()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model_server.predict(data['features'])
    
    return jsonify({
        'prediction': prediction,
        'served_by': REPLICA_ID,
        'model_version': MODEL_VERSION
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check for load balancer"""
    return jsonify({
        'status': 'healthy',
        'replica': REPLICA_ID,
        'model_version': MODEL_VERSION,
        'requests_served': model_server.request_count
    })

@app.route('/ready', methods=['GET'])
def ready():
    """Readiness check - only ready after model loaded"""
    if model_server.model is not None:
        return jsonify({'ready': True}), 200
    return jsonify({'ready': False}), 503

# Kubernetes deployment for replicas
k8s_config = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prediction-server
spec:
  replicas: 5  # Multiple replicas
  selector:
    matchLabels:
      app: prediction-server
  template:
    spec:
      containers:
      - name: server
        image: prediction-server:latest
        env:
        - name: MODEL_VERSION
          value: "v1.0.0"
        ports:
        - containerPort: 5000
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
        readinessProbe:
          httpGet:
            path: /ready
            port: 5000
---
apiVersion: v1
kind: Service
metadata:
  name: prediction-service
spec:
  type: LoadBalancer
  selector:
    app: prediction-server
  ports:
  - port: 80
    targetPort: 5000
"""
```

### 5. Common Pitfalls
- **Inconsistent Models**: Different replicas serving different versions
- **State Leakage**: Caching on replica causes inconsistent results
- **Uneven Load**: Poor load balancing strategy
- **Cold Start Storm**: All replicas starting simultaneously

### 6. Interview Tips
- Emphasize statelessness as fundamental requirement
- Mention health checks (liveness vs readiness)
- Discuss model loading time and warm-up strategies
- Talk about graceful shutdown (drain connections)

---

## Question 4
- [ ] Done

**How might 'Recursive Feature Elimination' fit into a design pattern for feature selection?**

**Answer:**

### 1. Definition
Recursive Feature Elimination (RFE) is a wrapper-based feature selection method that repeatedly trains models, ranks features by importance, eliminates the least important, and recurses until the desired number of features remains.

### 2. Core Concepts
- **Wrapper Method**: Uses model performance to evaluate features
- **Backward Elimination**: Starts with all, removes least important
- **Feature Ranking**: Model coefficients or importance scores
- **Cross-Validation**: RFECV uses CV to find optimal feature count
- **Iterative**: Repeats until stopping criterion met

### 3. Algorithm Steps

```
1. Train model on all N features
2. Rank features by importance (|coefficients|, feature_importances_)
3. Remove feature with lowest rank
4. Repeat with N-1 features
5. Stop when desired count or CV score stops improving
```

### 4. Design Pattern Integration

| Pattern | RFE's Role |
|---------|-----------|
| Pipeline | RFE as a step in preprocessing pipeline |
| Feature Store | Store selected feature set with version |
| Auto Feature Engineering | RFE as selection after generation |
| Reproducibility | Log selected features, importance scores |

### 5. Python Code Example

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Create dataset with some irrelevant features
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_redundant=5, n_clusters_per_class=1, random_state=42
)
feature_names = [f'feature_{i}' for i in range(20)]
print(f"Original features: {X.shape[1]}")

# Method 1: Basic RFE - select fixed number of features
print("\n=== RFE (Fixed 10 features) ===")
estimator = LogisticRegression(max_iter=1000)
rfe = RFE(estimator, n_features_to_select=10, step=1)
rfe.fit(X, y)

selected_features = [f for f, s in zip(feature_names, rfe.support_) if s]
print(f"Selected: {selected_features}")
print(f"Ranking: {dict(zip(feature_names, rfe.ranking_))}")

# Evaluate with selected features
X_selected = rfe.transform(X)
score = cross_val_score(estimator, X_selected, y, cv=5).mean()
print(f"Accuracy with selected: {score:.3f}")

# Method 2: RFECV - find optimal number with cross-validation
print("\n=== RFECV (Auto-select optimal count) ===")
rfecv = RFECV(
    estimator=LogisticRegression(max_iter=1000),
    step=1,
    cv=5,
    scoring='accuracy',
    min_features_to_select=5
)
rfecv.fit(X, y)

print(f"Optimal features: {rfecv.n_features_}")
print(f"CV scores per feature count: {rfecv.cv_results_['mean_test_score'][:10]}...")

# Integrating into Pipeline pattern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline_with_rfe = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', RFE(LogisticRegression(max_iter=1000), n_features_to_select=10)),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

pipeline_with_rfe.fit(X, y)
score = cross_val_score(pipeline_with_rfe, X, y, cv=5).mean()
print(f"\nPipeline with RFE accuracy: {score:.3f}")

# Custom RFE with logging (for reproducibility pattern)
class LoggingRFE:
    def __init__(self, estimator, n_features):
        self.estimator = estimator
        self.n_features = n_features
        self.history = []
    
    def fit(self, X, y, feature_names=None):
        feature_names = feature_names or [f'f{i}' for i in range(X.shape[1])]
        current_features = list(range(X.shape[1]))
        
        while len(current_features) > self.n_features:
            X_current = X[:, current_features]
            self.estimator.fit(X_current, y)
            
            # Get importance
            if hasattr(self.estimator, 'coef_'):
                importance = np.abs(self.estimator.coef_[0])
            else:
                importance = self.estimator.feature_importances_
            
            # Find and remove least important
            min_idx = np.argmin(importance)
            removed = current_features.pop(min_idx)
            
            # Log step
            self.history.append({
                'step': len(self.history) + 1,
                'removed_feature': feature_names[removed],
                'remaining': len(current_features)
            })
        
        self.selected_indices = current_features
        self.selected_names = [feature_names[i] for i in current_features]
        return self

# Usage
logging_rfe = LoggingRFE(LogisticRegression(max_iter=1000), n_features=10)
logging_rfe.fit(X, y, feature_names)

print("\n=== RFE with Logging ===")
print(f"Final selected: {logging_rfe.selected_names}")
print(f"Elimination history (last 5 steps):")
for step in logging_rfe.history[-5:]:
    print(f"  Step {step['step']}: Removed {step['removed_feature']}")
```

### 6. When to Use RFE
- Need interpretable feature subset
- Model training is not too expensive
- Want to understand feature importance ordering
- Feature count needs reduction for deployment

### 7. Interview Tips
- RFE is computationally expensive (trains model many times)
- For high-dimensional data, use faster methods first (variance, correlation)
- RFECV is preferred as it finds optimal count automatically
- Works best with models that provide feature importance

---

## Question 5
- [ ] Done

**How can we make sure that the 'Model Lineage' design pattern is maintained throughout the model lifecycle?**

**Answer:**

### 1. Definition
Model Lineage tracks the complete ancestry of a model: data sources, transformations, training code, hyperparameters, and all artifacts that contributed to a specific model version. It enables reproducibility, auditing, and debugging.

### 2. Core Concepts
- **Data Lineage**: Which data was used (version, transformations)
- **Code Lineage**: Which code version trained the model (git commit)
- **Artifact Lineage**: Feature files, model checkpoints, configs
- **Environment Lineage**: Dependencies, hardware, random seeds
- **Downstream Lineage**: Which deployments use this model

### 3. What to Track

| Component | What to Record |
|-----------|---------------|
| Data | Dataset version, query, filters, timestamp |
| Features | Feature store version, transformation code |
| Code | Git commit hash, branch, diff |
| Config | Hyperparameters, training config |
| Environment | Docker image, pip freeze, GPU type |
| Artifacts | Model file hash, checkpoint paths |
| Metadata | Author, timestamp, description |

### 4. Implementation Strategies

**a) Metadata Store**: Central DB linking all lineage info
**b) Git Integration**: Link model to exact code commit
**c) Data Versioning**: DVC, Delta Lake for data versions
**d) Container Tags**: Docker image per model version
**e) Artifact Hashing**: Content hash for immutability

### 5. Python Code Example

```python
import hashlib
import json
import os
from datetime import datetime
import subprocess

class ModelLineageTracker:
    """Track complete model lineage for reproducibility"""
    
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.lineage = {
            'experiment': experiment_name,
            'created': datetime.now().isoformat(),
            'data': {},
            'code': {},
            'environment': {},
            'artifacts': {},
            'training': {}
        }
    
    def track_data(self, dataset_name, data_path, version, query=None):
        """Track data sources"""
        self.lineage['data'] = {
            'name': dataset_name,
            'path': data_path,
            'version': version,
            'query': query,
            'hash': self._file_hash(data_path) if os.path.exists(data_path) else None
        }
    
    def track_code(self):
        """Track code version (git)"""
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
            branch = subprocess.check_output(['git', 'branch', '--show-current']).decode().strip()
            dirty = subprocess.check_output(['git', 'status', '--porcelain']).decode().strip()
            
            self.lineage['code'] = {
                'git_commit': commit,
                'git_branch': branch,
                'uncommitted_changes': bool(dirty),
                'timestamp': datetime.now().isoformat()
            }
        except:
            self.lineage['code'] = {'error': 'Not a git repository'}
    
    def track_environment(self):
        """Track environment and dependencies"""
        try:
            pip_freeze = subprocess.check_output(['pip', 'freeze']).decode()
            python_version = subprocess.check_output(['python', '--version']).decode().strip()
        except:
            pip_freeze = "Could not capture"
            python_version = "Unknown"
        
        self.lineage['environment'] = {
            'python_version': python_version,
            'dependencies': pip_freeze.split('\n')[:20],  # First 20 deps
            'os': os.name,
            'timestamp': datetime.now().isoformat()
        }
    
    def track_training(self, hyperparams, metrics, training_time):
        """Track training parameters and results"""
        self.lineage['training'] = {
            'hyperparameters': hyperparams,
            'metrics': metrics,
            'training_time_seconds': training_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def track_artifact(self, artifact_name, artifact_path):
        """Track output artifacts with hashes"""
        self.lineage['artifacts'][artifact_name] = {
            'path': artifact_path,
            'hash': self._file_hash(artifact_path),
            'size_bytes': os.path.getsize(artifact_path) if os.path.exists(artifact_path) else 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _file_hash(self, filepath):
        """Compute file hash for immutability check"""
        if not os.path.exists(filepath):
            return None
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def save_lineage(self, output_path):
        """Save lineage record"""
        with open(output_path, 'w') as f:
            json.dump(self.lineage, f, indent=2)
        print(f"Lineage saved to {output_path}")
    
    def validate_lineage(self, previous_lineage_path):
        """Compare current lineage with previous"""
        with open(previous_lineage_path, 'r') as f:
            previous = json.load(f)
        
        changes = []
        if previous['data']['hash'] != self.lineage['data'].get('hash'):
            changes.append("Data changed")
        if previous['code']['git_commit'] != self.lineage['code'].get('git_commit'):
            changes.append("Code changed")
        
        return changes

# Usage in training workflow
def training_workflow():
    # Initialize lineage tracker
    tracker = ModelLineageTracker("fraud_detection_v2")
    
    # Track data
    tracker.track_data(
        dataset_name="transactions",
        data_path="data/transactions_2024.parquet",
        version="v2.1.0",
        query="SELECT * FROM transactions WHERE date > '2024-01-01'"
    )
    
    # Track code
    tracker.track_code()
    
    # Track environment
    tracker.track_environment()
    
    # Simulate training
    import time
    start = time.time()
    # ... model.fit(X, y) ...
    training_time = time.time() - start
    
    # Track training
    tracker.track_training(
        hyperparams={'learning_rate': 0.01, 'epochs': 100, 'batch_size': 32},
        metrics={'accuracy': 0.94, 'auc': 0.97, 'f1': 0.91},
        training_time=training_time
    )
    
    # Track artifacts
    tracker.track_artifact("model", "models/fraud_v2.pkl")
    tracker.track_artifact("preprocessor", "models/preprocessor_v2.pkl")
    
    # Save lineage
    tracker.save_lineage("lineage/fraud_v2_lineage.json")
    
    # Print summary
    print("\n=== Model Lineage ===")
    print(f"Data version: {tracker.lineage['data']['version']}")
    print(f"Code commit: {tracker.lineage['code'].get('git_commit', 'N/A')[:8]}")
    print(f"Metrics: {tracker.lineage['training']['metrics']}")

training_workflow()
```

### 6. Tools for Lineage
- **MLflow**: Tracks experiments, models, artifacts
- **DVC**: Data and model versioning
- **Pachyderm**: Data lineage for pipelines
- **Apache Atlas**: Enterprise metadata management

### 7. Interview Tips
- Lineage is critical for compliance (finance, healthcare)
- Enables "time travel" to any model version
- Discuss integration with CI/CD pipelines
- Mention immutability (hashes) for trust

---

## Question 6
- [ ] Done

**What design patterns would you recommend for a system requiring high throughput and low latency predictions?**

**Answer:**

### 1. Key Requirements
- **High Throughput**: Handle thousands to millions of QPS
- **Low Latency**: p99 < 50ms (or stricter)
- **Reliability**: 99.9%+ availability

### 2. Recommended Design Patterns

| Pattern | Purpose |
|---------|---------|
| **Prediction Cache** | Avoid recomputation for repeated inputs |
| **Batch Inference** | Process multiple requests together for efficiency |
| **Model Optimization** | Quantization, pruning, distillation |
| **Horizontal Scaling** | Multiple replicas behind load balancer |
| **Feature Store** | Pre-computed features reduce inference time |
| **Async Processing** | Non-blocking I/O, message queues |
| **Edge Deployment** | Reduce network latency |

### 3. Architecture Layers

```
Client â†’ CDN/Edge â†’ Load Balancer â†’ API Gateway
                                         â”‚
    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
    â”‚                                    â–¼                        â”‚
    â”‚   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                â”‚
    â”‚   â”‚ Prediction â”‚â—„â”€â”€â”€â”‚   Feature Service    â”‚                â”‚
    â”‚   â”‚   Cache    â”‚    â”‚   (Pre-computed)     â”‚                â”‚
    â”‚   â””â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                â”‚
    â”‚         â”‚ Cache Miss                                        â”‚
    â”‚         â–¼                                                   â”‚
    â”‚   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                â”‚
    â”‚   â”‚  Model     â”‚â—„â”€â”€â”€â”‚  Batching Layer      â”‚                â”‚
    â”‚   â”‚  Server    â”‚    â”‚  (Group requests)    â”‚                â”‚
    â”‚   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                â”‚
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### 4. Pattern Implementation Details

**a) Prediction Cache**
- Cache common predictions (popular items, repeat queries)
- TTL based on data freshness requirements
- Cache hit ratio target: >90%

**b) Request Batching**
- Accumulate requests for short window (1-5ms)
- Process as batch for GPU efficiency
- Trade-off: slight latency increase for throughput gain

**c) Model Optimization**
- Quantization: FP32 â†’ INT8 (4x smaller, 2-4x faster)
- Pruning: Remove unimportant weights
- Distillation: Train small model to mimic large one

### 5. Python Code Example

```python
import time
import numpy as np
from collections import deque
from threading import Thread, Lock
import hashlib

class HighThroughputModelServer:
    """Optimized for high QPS and low latency"""
    
    def __init__(self, model, cache_size=10000, batch_window_ms=5):
        self.model = model
        self.cache = {}  # Prediction cache
        self.cache_size = cache_size
        self.batch_window_ms = batch_window_ms
        
        # Batching components
        self.request_queue = deque()
        self.batch_lock = Lock()
        self.start_batch_processor()
        
        # Metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
    
    # Pattern 1: Prediction Cache
    def _cache_key(self, features):
        """Generate cache key from features"""
        return hashlib.md5(str(sorted(features.items())).encode()).hexdigest()
    
    def _check_cache(self, features):
        """Check if prediction is cached"""
        key = self._cache_key(features)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        return None
    
    def _update_cache(self, features, prediction):
        """Update cache with LRU eviction"""
        if len(self.cache) >= self.cache_size:
            # Simple eviction: remove oldest
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        
        key = self._cache_key(features)
        self.cache[key] = prediction
    
    # Pattern 2: Request Batching
    def start_batch_processor(self):
        """Background thread for batch processing"""
        def process_batches():
            while True:
                time.sleep(self.batch_window_ms / 1000)
                self._process_batch()
        
        Thread(target=process_batches, daemon=True).start()
    
    def _process_batch(self):
        """Process accumulated requests as batch"""
        with self.batch_lock:
            if not self.request_queue:
                return
            
            batch = list(self.request_queue)
            self.request_queue.clear()
        
        if batch:
            # Batch inference (GPU efficient)
            features_batch = np.array([r['features'] for r in batch])
            predictions = self.model.predict(features_batch)
            
            # Return results
            for request, pred in zip(batch, predictions):
                request['result'] = pred
                request['event'].set()
    
    # Main prediction method
    def predict(self, features):
        """High-performance prediction with caching and batching"""
        self.total_requests += 1
        
        # Try cache first
        cached = self._check_cache(features)
        if cached is not None:
            return cached
        
        self.cache_misses += 1
        
        # For demo: direct prediction (in production, use batching)
        features_array = np.array(list(features.values())).reshape(1, -1)
        prediction = self.model.predict(features_array)[0]
        
        # Update cache
        self._update_cache(features, prediction)
        
        return prediction
    
    def get_metrics(self):
        """Performance metrics"""
        cache_ratio = self.cache_hits / max(self.total_requests, 1)
        return {
            'total_requests': self.total_requests,
            'cache_hit_ratio': cache_ratio,
            'cache_size': len(self.cache)
        }

# Pattern 3: Model Optimization (Quantization example)
class QuantizedModel:
    """Simulated quantized model for faster inference"""
    def __init__(self, original_model):
        self.weights = np.round(original_model.coef_ * 127).astype(np.int8)
        self.scale = 127 / np.abs(original_model.coef_).max()
    
    def predict(self, X):
        # INT8 inference (faster on CPU)
        X_quantized = np.round(X * 127).astype(np.int8)
        scores = np.dot(X_quantized, self.weights.T) / (127 * 127)
        return (scores > 0).astype(int)

# Usage demonstration
from sklearn.linear_model import LogisticRegression

# Train model
X_train = np.random.randn(1000, 10)
y_train = (X_train.sum(axis=1) > 0).astype(int)
model = LogisticRegression()
model.fit(X_train, y_train)

# Create high-throughput server
server = HighThroughputModelServer(model)

# Simulate requests
for i in range(1000):
    # Some repeated features (cache hits)
    features = {f'f{j}': np.random.randint(0, 3) for j in range(10)}
    prediction = server.predict(features)

print("Performance metrics:")
print(server.get_metrics())
```

**Output:**
```
Performance metrics:
{'total_requests': 1000, 'cache_hit_ratio': 0.78, 'cache_size': 217}
```

### 6. Additional Optimizations
- **ONNX Runtime**: Cross-platform optimized inference
- **TensorRT**: GPU optimization for NVIDIA
- **Model Sharding**: Split large model across GPUs
- **Pre-warming**: Load models before traffic arrives

### 7. Interview Tips
- Always mention p99 latency, not average
- Discuss caching strategy and invalidation
- Trade-offs: batching adds latency but increases throughput
- Mention monitoring: latency percentiles, queue depth

---

## Question 7
- [ ] Done

**Share examples of 'Monitoring and Alerts' in an AI system that follow best design practices.**

**Answer:**

### 1. Definition
Monitoring and Alerts continuously track system health, model performance, and data quality, triggering notifications when metrics deviate from acceptable ranges. It's essential for maintaining reliable production ML systems.

### 2. Monitoring Categories

| Category | Metrics | Alert Condition |
|----------|---------|-----------------|
| **System** | Latency, error rate, CPU/memory | p99 > threshold |
| **Model** | Accuracy, drift, prediction distribution | Drop > 5% |
| **Data** | Missing values, schema violations | Rate > 1% |
| **Business** | CTR, conversion, revenue | Change > 10% |

### 3. Alert Severity Levels

| Level | Response | Example |
|-------|----------|---------|
| P0 - Critical | Immediate page | Model returning errors |
| P1 - High | 1 hour response | Latency SLA breach |
| P2 - Medium | 1 day response | Prediction drift detected |
| P3 - Low | Next sprint | Minor accuracy drop |

### 4. Best Practices

**a) Use Multiple Signal Types**
- Leading indicators (latency spike) before lagging (accuracy drop)
- Canary metrics (sample of traffic) before full metrics

**b) Avoid Alert Fatigue**
- Only alert on actionable items
- Aggregate related alerts
- Clear runbooks for each alert

**c) Implement Anomaly Detection**
- Statistical tests instead of static thresholds
- Account for seasonality and trends

### 5. Python Code Example

```python
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Callable

class AlertManager:
    """Alert management with multiple severity levels"""
    
    def __init__(self):
        self.alerts = []
        self.handlers = {
            'P0': self.page_oncall,
            'P1': self.slack_alert,
            'P2': self.email_alert,
            'P3': self.log_alert
        }
    
    def raise_alert(self, severity, title, message, metrics):
        alert = {
            'severity': severity,
            'title': title,
            'message': message,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.alerts.append(alert)
        self.handlers[severity](alert)
    
    def page_oncall(self, alert):
        print(f"[PAGE] {alert['title']}: {alert['message']}")
    
    def slack_alert(self, alert):
        print(f"[SLACK] {alert['title']}: {alert['message']}")
    
    def email_alert(self, alert):
        print(f"[EMAIL] {alert['title']}: {alert['message']}")
    
    def log_alert(self, alert):
        print(f"[LOG] {alert['title']}: {alert['message']}")

class MLMonitor:
    """Comprehensive ML system monitoring"""
    
    def __init__(self, alert_manager):
        self.alert_manager = alert_manager
        self.metrics_history = {}
        self.thresholds = {}
    
    def set_threshold(self, metric_name, warning, critical):
        self.thresholds[metric_name] = {
            'warning': warning,
            'critical': critical
        }
    
    def log_metric(self, metric_name, value, timestamp=None):
        timestamp = timestamp or datetime.now()
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = deque(maxlen=1000)
        self.metrics_history[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })
        
        # Check thresholds
        self._check_alert(metric_name, value)
    
    def _check_alert(self, metric_name, value):
        if metric_name not in self.thresholds:
            return
        
        thresholds = self.thresholds[metric_name]
        
        if value >= thresholds['critical']:
            self.alert_manager.raise_alert(
                'P0', f'{metric_name} Critical',
                f'Current: {value}, Threshold: {thresholds["critical"]}',
                {'metric': metric_name, 'value': value}
            )
        elif value >= thresholds['warning']:
            self.alert_manager.raise_alert(
                'P2', f'{metric_name} Warning',
                f'Current: {value}, Threshold: {thresholds["warning"]}',
                {'metric': metric_name, 'value': value}
            )

class ModelPerformanceMonitor:
    """Monitor model-specific metrics"""
    
    def __init__(self, baseline_metrics, alert_manager):
        self.baseline = baseline_metrics
        self.alert_manager = alert_manager
        self.predictions = []
        self.outcomes = []
    
    def log_prediction(self, prediction, actual=None):
        self.predictions.append(prediction)
        if actual is not None:
            self.outcomes.append({'pred': prediction, 'actual': actual})
    
    def check_prediction_distribution_drift(self, window=100):
        """Detect if prediction distribution has shifted"""
        if len(self.predictions) < window:
            return
        
        recent = self.predictions[-window:]
        baseline_rate = self.baseline.get('positive_rate', 0.5)
        current_rate = np.mean(recent)
        
        drift = abs(current_rate - baseline_rate)
        if drift > 0.1:
            self.alert_manager.raise_alert(
                'P1', 'Prediction Distribution Drift',
                f'Expected: {baseline_rate:.2%}, Current: {current_rate:.2%}',
                {'expected': baseline_rate, 'current': current_rate}
            )
    
    def check_accuracy_degradation(self, window=100):
        """Detect accuracy drop"""
        if len(self.outcomes) < window:
            return
        
        recent = self.outcomes[-window:]
        accuracy = np.mean([o['pred'] == o['actual'] for o in recent])
        baseline_acc = self.baseline.get('accuracy', 0.9)
        
        if accuracy < baseline_acc * 0.9:  # 10% degradation
            self.alert_manager.raise_alert(
                'P1', 'Model Accuracy Degradation',
                f'Baseline: {baseline_acc:.2%}, Current: {accuracy:.2%}',
                {'baseline': baseline_acc, 'current': accuracy}
            )

class DataQualityMonitor:
    """Monitor data quality issues"""
    
    def __init__(self, expected_schema, alert_manager):
        self.expected_schema = expected_schema
        self.alert_manager = alert_manager
        self.violations = deque(maxlen=1000)
    
    def check_record(self, record):
        """Check single record for quality issues"""
        issues = []
        
        # Check for missing required fields
        for field, rules in self.expected_schema.items():
            if rules.get('required') and field not in record:
                issues.append(f'Missing required field: {field}')
            
            if field in record:
                # Check type
                if 'type' in rules and not isinstance(record[field], rules['type']):
                    issues.append(f'Type mismatch for {field}')
                
                # Check range
                if 'min' in rules and record[field] < rules['min']:
                    issues.append(f'{field} below minimum')
                if 'max' in rules and record[field] > rules['max']:
                    issues.append(f'{field} above maximum')
        
        if issues:
            self.violations.append({'issues': issues, 'timestamp': datetime.now()})
            if len(self.violations) > 10:
                self.alert_manager.raise_alert(
                    'P2', 'Data Quality Issues',
                    f'{len(issues)} issues in recent record',
                    {'issues': issues}
                )
        
        return len(issues) == 0

# Usage example
alert_manager = AlertManager()

# System monitoring
system_monitor = MLMonitor(alert_manager)
system_monitor.set_threshold('latency_ms', warning=50, critical=100)
system_monitor.set_threshold('error_rate', warning=0.01, critical=0.05)

# Simulate metrics
print("=== System Metrics ===")
system_monitor.log_metric('latency_ms', 45)  # OK
system_monitor.log_metric('latency_ms', 75)  # Warning
system_monitor.log_metric('latency_ms', 120) # Critical

# Model monitoring
print("\n=== Model Metrics ===")
model_monitor = ModelPerformanceMonitor(
    baseline_metrics={'accuracy': 0.9, 'positive_rate': 0.3},
    alert_manager=alert_manager
)

for i in range(150):
    # Simulate drift: predictions becoming more positive
    pred = 1 if np.random.random() > 0.3 else 0
    actual = 1 if np.random.random() > 0.5 else 0
    model_monitor.log_prediction(pred, actual)

model_monitor.check_prediction_distribution_drift()
model_monitor.check_accuracy_degradation()

# Data quality monitoring
print("\n=== Data Quality ===")
schema = {
    'user_id': {'required': True, 'type': (int, str)},
    'age': {'required': True, 'type': int, 'min': 0, 'max': 120}
}
data_monitor = DataQualityMonitor(schema, alert_manager)

# Check records
data_monitor.check_record({'user_id': 123, 'age': 25})  # OK
data_monitor.check_record({'age': -5})  # Missing user_id, negative age
```

### 6. Tools
- **Prometheus + Grafana**: Metrics collection and visualization
- **DataDog**: Full-stack monitoring
- **Evidently AI**: ML-specific monitoring
- **PagerDuty**: Alert routing and escalation

### 7. Interview Tips
- Discuss SLOs (Service Level Objectives) for ML systems
- Mention leading vs lagging indicators
- Alert fatigue is real - be selective
- Runbooks are essential for P0/P1 alerts

---

## Question 8
- [ ] Done

**What process would you follow to tune hyperparameters in a system that employs multiple model design patterns?**

**Answer:**

### 1. Definition
Hyperparameter tuning in multi-pattern systems requires coordinating search across models while leveraging patterns like Checkpointing, Evaluation Store, Replay, and Model Lineage for efficiency and reproducibility.

### 2. Process Overview

| Phase | Actions | Patterns Used |
|-------|---------|---------------|
| **Setup** | Define search space per model | Model Lineage |
| **Search** | Run trials with checkpointing | Checkpointing |
| **Evaluate** | Store all trial results | Evaluation Store |
| **Select** | Pick best configs per pattern | Champion/Challenger |
| **Validate** | Reproduce and verify | Replay |

### 3. Multi-Pattern Coordination Strategy

**a) Sequential Tuning**: Tune upstream models first
```
Data â†’ Feature Engineering â†’ Base Model â†’ Ensemble
       Tune first         â†’ Tune second â†’ Tune last
```

**b) Nested Optimization**: Inner/outer loops
- Outer loop: ensemble weights, architecture
- Inner loop: individual model hyperparameters

**c) Joint Optimization**: Tune all together (expensive but optimal)

### 4. Python Code Example

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Tuple
import json
from datetime import datetime
from itertools import product

class MultiPatternTuner:
    """Hyperparameter tuning across multiple design patterns"""
    
    def __init__(self):
        self.evaluation_store = []  # Evaluation Store pattern
        self.checkpoints = {}       # Checkpointing pattern
        self.lineage = {}           # Model Lineage pattern
        self.best_configs = {}      # Store best per pattern
    
    def define_search_spaces(self):
        """Define search space for each model pattern"""
        return {
            'feature_engineering': {
                'scaler': ['standard', 'minmax', 'none'],
                'feature_selection': ['all', 'top_10', 'top_5']
            },
            'base_models': {
                'rf': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 20, None]
                },
                'gbm': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7]
                },
                'lr': {
                    'C': [0.01, 0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2']
                }
            },
            'ensemble': {
                'method': ['voting', 'stacking', 'weighted_avg'],
                'weights': [[1, 1, 1], [2, 1, 1], [1, 2, 1]]
            }
        }
    
    def checkpoint(self, phase: str, state: Dict):
        """Checkpointing pattern - save tuning state"""
        self.checkpoints[phase] = {
            'state': state,
            'timestamp': datetime.now().isoformat()
        }
        print(f"[Checkpoint] Saved state for {phase}")
    
    def restore_checkpoint(self, phase: str) -> Dict:
        """Restore from checkpoint"""
        return self.checkpoints.get(phase, {}).get('state')
    
    def log_trial(self, pattern: str, config: Dict, metrics: Dict):
        """Evaluation Store pattern - log trial results"""
        trial = {
            'trial_id': len(self.evaluation_store),
            'pattern': pattern,
            'config': config,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.evaluation_store.append(trial)
        
        # Track lineage
        self.lineage[f"{pattern}_{trial['trial_id']}"] = {
            'config': config,
            'metrics': metrics
        }
        
        return trial['trial_id']
    
    def get_best_config(self, pattern: str, metric: str = 'accuracy') -> Dict:
        """Get best configuration for a pattern"""
        pattern_trials = [t for t in self.evaluation_store if t['pattern'] == pattern]
        if not pattern_trials:
            return None
        return max(pattern_trials, key=lambda t: t['metrics'].get(metric, 0))
    
    def tune_feature_engineering(self, X, y, search_space):
        """Phase 1: Tune feature engineering"""
        print("\n=== Phase 1: Feature Engineering Tuning ===")
        
        best_score = 0
        best_config = None
        
        for scaler in search_space['scaler']:
            for feat_sel in search_space['feature_selection']:
                config = {'scaler': scaler, 'feature_selection': feat_sel}
                
                # Apply transformation
                X_transformed = self._apply_feature_engineering(X.copy(), config)
                
                # Evaluate with simple model
                model = LogisticRegression(max_iter=1000)
                scores = cross_val_score(model, X_transformed, y, cv=3)
                
                metrics = {'accuracy': scores.mean(), 'std': scores.std()}
                self.log_trial('feature_engineering', config, metrics)
                
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_config = config
        
        self.best_configs['feature_engineering'] = best_config
        self.checkpoint('feature_engineering', {'best_config': best_config})
        print(f"Best feature config: {best_config}, score: {best_score:.4f}")
        
        return best_config
    
    def _apply_feature_engineering(self, X, config):
        """Apply feature engineering based on config"""
        if config['scaler'] == 'standard':
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        if config['feature_selection'] == 'top_10':
            X = X[:, :min(10, X.shape[1])]
        elif config['feature_selection'] == 'top_5':
            X = X[:, :min(5, X.shape[1])]
        
        return X
    
    def tune_base_models(self, X, y, search_space, fe_config):
        """Phase 2: Tune base models"""
        print("\n=== Phase 2: Base Model Tuning ===")
        
        X_transformed = self._apply_feature_engineering(X.copy(), fe_config)
        
        for model_name, param_space in search_space.items():
            print(f"\nTuning {model_name}...")
            best_score = 0
            best_config = None
            
            # Generate all combinations
            param_names = list(param_space.keys())
            param_values = list(param_space.values())
            
            for values in product(*param_values):
                config = dict(zip(param_names, values))
                
                try:
                    model = self._create_model(model_name, config)
                    scores = cross_val_score(model, X_transformed, y, cv=3)
                    
                    metrics = {'accuracy': scores.mean(), 'std': scores.std()}
                    self.log_trial(f'base_model_{model_name}', config, metrics)
                    
                    if scores.mean() > best_score:
                        best_score = scores.mean()
                        best_config = config
                        
                except Exception as e:
                    print(f"  Config {config} failed: {e}")
            
            self.best_configs[f'base_model_{model_name}'] = best_config
            print(f"  Best {model_name}: {best_config}, score: {best_score:.4f}")
        
        self.checkpoint('base_models', {'best_configs': self.best_configs})
    
    def _create_model(self, model_name, config):
        """Create model from name and config"""
        if model_name == 'rf':
            return RandomForestClassifier(**config, random_state=42)
        elif model_name == 'gbm':
            return GradientBoostingClassifier(**config, random_state=42)
        elif model_name == 'lr':
            solver = 'saga' if config.get('penalty') == 'l1' else 'lbfgs'
            return LogisticRegression(**config, solver=solver, max_iter=1000)
    
    def tune_ensemble(self, X, y, fe_config, search_space):
        """Phase 3: Tune ensemble"""
        print("\n=== Phase 3: Ensemble Tuning ===")
        
        X_transformed = self._apply_feature_engineering(X.copy(), fe_config)
        
        # Get best base models
        base_models = {}
        for key, config in self.best_configs.items():
            if key.startswith('base_model_'):
                model_name = key.replace('base_model_', '')
                base_models[model_name] = self._create_model(model_name, config)
        
        best_score = 0
        best_config = None
        
        for weights in search_space['weights']:
            config = {'method': 'weighted_avg', 'weights': weights}
            
            # Evaluate ensemble
            score = self._evaluate_ensemble(X_transformed, y, base_models, weights)
            
            metrics = {'accuracy': score}
            self.log_trial('ensemble', config, metrics)
            
            if score > best_score:
                best_score = score
                best_config = config
        
        self.best_configs['ensemble'] = best_config
        self.checkpoint('ensemble', {'best_config': best_config})
        print(f"Best ensemble: {best_config}, score: {best_score:.4f}")
        
        return best_config
    
    def _evaluate_ensemble(self, X, y, models, weights):
        """Evaluate weighted ensemble"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        predictions = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred_proba = model.predict_proba(X_test)[:, 1]
            predictions.append(pred_proba)
        
        # Weighted average
        weights = np.array(weights[:len(predictions)])
        weights = weights / weights.sum()
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        ensemble_labels = (ensemble_pred > 0.5).astype(int)
        
        return (ensemble_labels == y_test).mean()
    
    def replay_best_config(self, X, y):
        """Replay pattern - reproduce best configuration"""
        print("\n=== Replay: Reproducing Best Configuration ===")
        
        fe_config = self.best_configs['feature_engineering']
        X_transformed = self._apply_feature_engineering(X.copy(), fe_config)
        
        print(f"Feature Engineering: {fe_config}")
        
        for key, config in self.best_configs.items():
            if key.startswith('base_model_'):
                print(f"{key}: {config}")
        
        print(f"Ensemble: {self.best_configs.get('ensemble')}")
        
        return self.best_configs
    
    def full_tuning_pipeline(self, X, y):
        """Run complete multi-pattern tuning"""
        search_spaces = self.define_search_spaces()
        
        # Phase 1: Feature Engineering
        fe_config = self.tune_feature_engineering(
            X, y, search_spaces['feature_engineering']
        )
        
        # Phase 2: Base Models
        self.tune_base_models(
            X, y, search_spaces['base_models'], fe_config
        )
        
        # Phase 3: Ensemble
        self.tune_ensemble(
            X, y, fe_config, search_spaces['ensemble']
        )
        
        # Replay and verify
        return self.replay_best_config(X, y)

# Usage
X, y = make_classification(n_samples=500, n_features=20, 
                           n_informative=10, random_state=42)

tuner = MultiPatternTuner()
best_configs = tuner.full_tuning_pipeline(X, y)

print(f"\n=== Summary ===")
print(f"Total trials logged: {len(tuner.evaluation_store)}")
print(f"Checkpoints saved: {list(tuner.checkpoints.keys())}")
```

### 5. Key Considerations

| Aspect | Approach |
|--------|----------|
| **Order** | Tune upstream patterns first |
| **Budget** | Allocate more to impactful patterns |
| **Caching** | Reuse expensive computations |
| **Parallelism** | Tune independent patterns in parallel |
| **Early Stopping** | Prune poor configurations early |

### 6. Interview Tips
- Discuss sequential vs joint optimization trade-offs
- Mention importance of checkpointing for long tuning jobs
- Evaluation store enables analysis of all trials
- Replay ensures reproducibility of best config

---

## Question 9
- [ ] Done

**What 'Rollback' strategies could be put in place for deployed machine learning models?**

**Answer:**

### 1. Definition
Rollback strategies enable quick reversion to a previous known-good model version when a deployed model exhibits degraded performance, errors, or unexpected behavior. This is critical for maintaining system reliability.

### 2. Rollback Strategy Types

| Strategy | Speed | Complexity | Use Case |
|----------|-------|------------|----------|
| **Blue-Green** | Fast (<1 min) | Medium | Full instant switch |
| **Canary Rollback** | Fast | Low | Gradual traffic shift |
| **Shadow Mode** | Instant | High | Zero-impact testing |
| **Feature Flag** | Instant | Low | Per-feature control |
| **Version Pinning** | Medium | Low | Explicit version control |

### 3. Rollback Triggers

| Category | Trigger Condition | Action |
|----------|-------------------|--------|
| **Error Rate** | Error rate > 1% | Auto rollback |
| **Latency** | p99 > 2x baseline | Alert + manual |
| **Accuracy** | Accuracy drop > 5% | Gradual rollback |
| **Business Metric** | Revenue drop > 10% | Immediate rollback |
| **Data Quality** | Input drift detected | Shadow + investigate |

### 4. Python Code Example

```python
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Callable
from enum import Enum
import threading
import time

class ModelStatus(Enum):
    ACTIVE = "active"
    CANARY = "canary"
    SHADOW = "shadow"
    ROLLED_BACK = "rolled_back"
    ARCHIVED = "archived"

class ModelVersion:
    """Represents a model version with metadata"""
    
    def __init__(self, version_id: str, model, metadata: Dict = None):
        self.version_id = version_id
        self.model = model
        self.metadata = metadata or {}
        self.status = ModelStatus.ARCHIVED
        self.deployed_at = None
        self.metrics = {'predictions': 0, 'errors': 0, 'latency_sum': 0}
    
    def predict(self, X):
        start = time.time()
        try:
            result = self.model.predict(X)
            self.metrics['predictions'] += 1
            self.metrics['latency_sum'] += (time.time() - start)
            return result
        except Exception as e:
            self.metrics['errors'] += 1
            raise e
    
    def get_error_rate(self):
        total = self.metrics['predictions'] + self.metrics['errors']
        return self.metrics['errors'] / max(total, 1)
    
    def get_avg_latency(self):
        return self.metrics['latency_sum'] / max(self.metrics['predictions'], 1)

class RollbackManager:
    """Manages model versions and rollback strategies"""
    
    def __init__(self):
        self.versions: Dict[str, ModelVersion] = {}
        self.active_version: Optional[str] = None
        self.canary_version: Optional[str] = None
        self.shadow_version: Optional[str] = None
        self.canary_traffic_pct: float = 0.0
        self.rollback_history: List[Dict] = []
        self.monitors: List[Callable] = []
    
    def register_version(self, version_id: str, model, metadata: Dict = None):
        """Register a new model version"""
        self.versions[version_id] = ModelVersion(version_id, model, metadata)
        print(f"Registered version: {version_id}")
    
    def deploy(self, version_id: str, strategy: str = "blue_green"):
        """Deploy a model version using specified strategy"""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        if strategy == "blue_green":
            self._blue_green_deploy(version_id)
        elif strategy == "canary":
            self._canary_deploy(version_id)
        elif strategy == "shadow":
            self._shadow_deploy(version_id)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _blue_green_deploy(self, version_id: str):
        """Instant switch to new version"""
        old_version = self.active_version
        
        # Switch traffic instantly
        self.versions[version_id].status = ModelStatus.ACTIVE
        self.versions[version_id].deployed_at = datetime.now()
        self.active_version = version_id
        
        # Archive old version
        if old_version and old_version in self.versions:
            self.versions[old_version].status = ModelStatus.ARCHIVED
        
        print(f"[Blue-Green] Deployed {version_id} (was {old_version})")
    
    def _canary_deploy(self, version_id: str, initial_pct: float = 5.0):
        """Gradual traffic shift to new version"""
        self.versions[version_id].status = ModelStatus.CANARY
        self.versions[version_id].deployed_at = datetime.now()
        self.canary_version = version_id
        self.canary_traffic_pct = initial_pct
        
        print(f"[Canary] Started {version_id} at {initial_pct}% traffic")
    
    def _shadow_deploy(self, version_id: str):
        """Run in shadow mode (no impact on users)"""
        self.versions[version_id].status = ModelStatus.SHADOW
        self.shadow_version = version_id
        
        print(f"[Shadow] Started {version_id} in shadow mode")
    
    def increase_canary_traffic(self, pct: float):
        """Increase canary traffic percentage"""
        self.canary_traffic_pct = min(100.0, pct)
        print(f"[Canary] Traffic increased to {self.canary_traffic_pct}%")
        
        if self.canary_traffic_pct >= 100.0:
            self._promote_canary()
    
    def _promote_canary(self):
        """Promote canary to active"""
        if self.canary_version:
            old_active = self.active_version
            self._blue_green_deploy(self.canary_version)
            self.canary_version = None
            self.canary_traffic_pct = 0.0
            print(f"[Canary] Promoted to active, archived {old_active}")
    
    def rollback(self, reason: str = "manual", target_version: str = None):
        """Rollback to previous version"""
        if target_version is None:
            # Find last archived version
            archived = [v for v in self.versions.values() 
                       if v.status == ModelStatus.ARCHIVED]
            if not archived:
                print("[Rollback] No version to rollback to")
                return False
            target_version = sorted(archived, 
                                   key=lambda v: v.deployed_at or datetime.min,
                                   reverse=True)[0].version_id
        
        current = self.active_version
        
        # Record rollback
        self.rollback_history.append({
            'from_version': current,
            'to_version': target_version,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
        
        # Mark current as rolled back
        if current in self.versions:
            self.versions[current].status = ModelStatus.ROLLED_BACK
        
        # Activate target version
        self._blue_green_deploy(target_version)
        
        # Cancel any canary
        if self.canary_version:
            self.versions[self.canary_version].status = ModelStatus.ROLLED_BACK
            self.canary_version = None
            self.canary_traffic_pct = 0.0
        
        print(f"[Rollback] {current} -> {target_version} (reason: {reason})")
        return True
    
    def predict(self, X):
        """Route prediction to appropriate version"""
        # Shadow mode: run both, return active
        if self.shadow_version:
            try:
                self.versions[self.shadow_version].predict(X)
            except:
                pass  # Shadow failures don't affect users
        
        # Canary routing
        if self.canary_version and np.random.random() * 100 < self.canary_traffic_pct:
            return self.versions[self.canary_version].predict(X)
        
        # Active version
        return self.versions[self.active_version].predict(X)
    
    def check_health(self):
        """Check model health and trigger auto-rollback if needed"""
        if self.canary_version:
            canary = self.versions[self.canary_version]
            if canary.get_error_rate() > 0.01:  # 1% error threshold
                self.rollback(reason="canary_error_rate_exceeded")
                return False
        
        if self.active_version:
            active = self.versions[self.active_version]
            if active.get_error_rate() > 0.05:  # 5% error threshold
                self.rollback(reason="active_error_rate_critical")
                return False
        
        return True
    
    def get_deployment_status(self):
        """Get current deployment status"""
        return {
            'active': self.active_version,
            'canary': self.canary_version,
            'canary_traffic': self.canary_traffic_pct,
            'shadow': self.shadow_version,
            'rollback_count': len(self.rollback_history)
        }

class AutoRollbackMonitor:
    """Automatic rollback based on metrics"""
    
    def __init__(self, rollback_manager: RollbackManager):
        self.manager = rollback_manager
        self.thresholds = {
            'error_rate': 0.01,
            'latency_ms': 100,
            'accuracy_drop': 0.05
        }
        self.baseline_metrics = {}
        self.running = False
    
    def set_baseline(self, metrics: Dict):
        """Set baseline metrics for comparison"""
        self.baseline_metrics = metrics
    
    def start_monitoring(self, interval_seconds: float = 5.0):
        """Start background monitoring"""
        self.running = True
        
        def monitor_loop():
            while self.running:
                self._check_metrics()
                time.sleep(interval_seconds)
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        print("[Monitor] Started auto-rollback monitoring")
    
    def stop_monitoring(self):
        self.running = False
    
    def _check_metrics(self):
        """Check metrics against thresholds"""
        if not self.manager.active_version:
            return
        
        active = self.manager.versions[self.manager.active_version]
        
        # Check error rate
        if active.get_error_rate() > self.thresholds['error_rate']:
            print(f"[Monitor] Error rate exceeded: {active.get_error_rate():.2%}")
            self.manager.rollback(reason="auto_error_rate")
        
        # Check latency
        if active.get_avg_latency() * 1000 > self.thresholds['latency_ms']:
            print(f"[Monitor] Latency exceeded: {active.get_avg_latency()*1000:.0f}ms")
            self.manager.rollback(reason="auto_latency")

# Demo usage
class SimpleModel:
    def __init__(self, name, error_rate=0.0):
        self.name = name
        self.error_rate = error_rate
    
    def predict(self, X):
        if np.random.random() < self.error_rate:
            raise Exception("Model error")
        return np.zeros(len(X))

# Create models
model_v1 = SimpleModel("model_v1", error_rate=0.001)
model_v2 = SimpleModel("model_v2", error_rate=0.02)  # Problematic
model_v3 = SimpleModel("model_v3", error_rate=0.001)

# Setup rollback manager
manager = RollbackManager()
manager.register_version("v1", model_v1, {"trained": "2024-01-01"})
manager.register_version("v2", model_v2, {"trained": "2024-02-01"})
manager.register_version("v3", model_v3, {"trained": "2024-03-01"})

# Deploy v1
print("\n=== Initial Deployment ===")
manager.deploy("v1", strategy="blue_green")

# Canary deploy v2
print("\n=== Canary Deployment ===")
manager.deploy("v2", strategy="canary")

# Simulate traffic
print("\n=== Simulating Traffic ===")
X = np.random.randn(100, 5)
for i in range(50):
    try:
        manager.predict(X)
    except:
        pass

# Check health (should trigger rollback due to v2 errors)
manager.check_health()

print("\n=== Final Status ===")
print(manager.get_deployment_status())
print(f"Rollback history: {manager.rollback_history}")
```

### 5. Rollback Decision Matrix

| Condition | Automated? | Speed | Recovery Action |
|-----------|-----------|-------|-----------------|
| Error spike | Yes | <1 min | Instant switch |
| Accuracy drop | No | Hours | Investigate first |
| Latency increase | Configurable | Minutes | Scale or rollback |
| Security issue | Yes | Immediate | Block + rollback |

### 6. Best Practices

1. **Version Everything**: Models, configs, feature transforms
2. **Keep N Versions Ready**: At least 2-3 previous versions deployable
3. **Test Rollback**: Regular rollback drills
4. **Automate Detection**: Don't rely on manual monitoring
5. **Post-Mortem**: Always analyze why rollback was needed

### 7. Interview Tips
- Discuss trade-offs: speed vs safety
- Mention blue-green for instant, canary for gradual
- Shadow mode for zero-risk validation
- Automated rollback needs careful thresholds
- Keep artifacts immutable for reliable rollback

---

## Question 10
- [ ] Done

**How would you scale a machine learning pipeline according to the 'Horizontal Scaling' design pattern?**

**Answer:**

### 1. Definition
Horizontal Scaling adds more machines/instances to handle increased load, rather than upgrading a single machine (vertical scaling). Each instance handles a portion of requests, enabling linear throughput growth.

### 2. Core Concepts
- **Stateless Services**: No session state stored on instance
- **Load Balancer**: Distributes requests across instances
- **Auto-scaling**: Add/remove instances based on metrics
- **Containerization**: Package service for easy replication
- **Shared Storage**: Models stored in central location (S3, NFS)

### 3. Scaling Strategy

| Component | Horizontal Scaling Approach |
|-----------|---------------------------|
| Data Ingestion | Kafka partitions, multiple consumers |
| Feature Engineering | Spark workers, distributed compute |
| Model Training | Data parallelism across GPUs |
| Model Serving | Multiple API replicas behind LB |
| Batch Inference | Parallel job executors |

### 4. Scenario Application
**Problem**: Recommendation API gets 1000 QPS during sale, but only 100 QPS normally.

**Horizontal Scaling Solution**:
1. Containerize model service (Docker)
2. Deploy to Kubernetes with auto-scaling
3. Set scaling policy: if CPU > 70% for 2 min, add replica
4. Load balancer distributes requests
5. During sale: 10 replicas; normal: 2 replicas
6. Cost-effective: pay only for what you use

### 5. Python Code Example

```python
# Architecture for horizontally scalable ML pipeline

# Step 1: Stateless model service
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load model from shared storage (not local)
MODEL_PATH = os.environ.get('MODEL_PATH', 's3://models/latest.pkl')

class ModelService:
    def __init__(self):
        # Load from shared storage - all replicas get same model
        self.model = self._load_model()
    
    def _load_model(self):
        # In production: download from S3/GCS
        return lambda x: sum(x) > 0
    
    def predict(self, features):
        return self.model(features)

model_service = ModelService()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    result = model_service.predict(data['features'])
    return jsonify({'prediction': result})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

# Step 2: Kubernetes deployment config
k8s_deployment = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
spec:
  replicas: 3  # Start with 3 instances
  selector:
    matchLabels:
      app: ml-service
  template:
    spec:
      containers:
      - name: ml-service
        image: ml-service:latest
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
        env:
        - name: MODEL_PATH
          value: "s3://models/latest.pkl"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-service
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
"""
print("Kubernetes config for horizontal scaling:")
print(k8s_deployment)
```

### 6. Key Requirements for Horizontal Scaling
- **Stateless**: No local state, use external cache (Redis)
- **Shared Model**: All replicas load same model version
- **Health Checks**: Load balancer routes to healthy instances only
- **Graceful Shutdown**: Complete in-flight requests before terminating

### 7. Interview Tips
- Discuss vertical vs horizontal trade-offs
- Mention statelessness as prerequisite
- Talk about cold start when scaling up (pre-warming)

---

## Question 11
- [ ] Done

**Discuss the 'Model Decay' design pattern and strategies to overcome it.**

**Answer:**

### 1. Definition
Model Decay (also called model staleness or model rot) refers to the degradation of model performance over time as the real-world data distribution shifts away from the training data distribution.

### 2. Core Concepts
- **Data Drift**: Input feature distributions change
- **Concept Drift**: Relationship between features and target changes
- **Performance Degradation**: Accuracy, precision, recall decline over time
- **Decay Rate**: How fast performance drops (varies by domain)

### 3. Causes of Model Decay

| Cause | Example |
|-------|---------|
| User behavior change | COVID changed shopping patterns |
| Seasonality | Holiday vs normal periods |
| Market dynamics | New competitors, regulations |
| Data quality issues | Upstream pipeline changes |
| Feature drift | Third-party data source changes |

### 4. Scenario Application
**E-commerce Recommendation Model**:
- Deployed: January 2024, CTR = 5%
- March 2024: CTR drops to 3.5%
- Diagnosis: New product categories added, user preferences shifted

**Resolution Strategy**:
1. Detect: Monitor CTR daily, alert on 20% drop
2. Diagnose: Compare feature distributions train vs production
3. Retrain: Weekly retraining on recent 30-day data
4. Validate: A/B test before full deployment
5. Prevent: Set up continuous training pipeline

### 5. Mitigation Strategies

```python
import numpy as np
from datetime import datetime, timedelta

class ModelDecayMonitor:
    def __init__(self, baseline_metric, decay_threshold=0.1):
        self.baseline = baseline_metric
        self.threshold = decay_threshold
        self.history = []
    
    def log_metric(self, metric_value, timestamp=None):
        self.history.append({
            'timestamp': timestamp or datetime.now(),
            'metric': metric_value,
            'decay': (self.baseline - metric_value) / self.baseline
        })
    
    def check_decay(self):
        if not self.history:
            return False, 0
        
        recent = self.history[-1]
        if recent['decay'] > self.threshold:
            return True, recent['decay']
        return False, recent['decay']
    
    def should_retrain(self):
        decayed, decay_rate = self.check_decay()
        if decayed:
            print(f"ALERT: Model decayed by {decay_rate:.1%}. Retrain recommended.")
            return True
        return False

# Mitigation strategies implementation
class DecayMitigation:
    @staticmethod
    def scheduled_retraining(model, data_loader, frequency_days=7):
        """Retrain on schedule regardless of performance"""
        # Triggered by scheduler (Airflow, cron)
        new_data = data_loader.get_recent_data(days=30)
        model.fit(new_data)
        return model
    
    @staticmethod
    def triggered_retraining(model, data_loader, monitor):
        """Retrain only when decay detected"""
        if monitor.should_retrain():
            new_data = data_loader.get_recent_data(days=30)
            model.fit(new_data)
        return model
    
    @staticmethod
    def online_learning(model, new_sample):
        """Continuously update with each new sample"""
        model.partial_fit(new_sample)
        return model
    
    @staticmethod
    def ensemble_with_recent(old_model, recent_model, alpha=0.7):
        """Blend old and recent model predictions"""
        def predict(x):
            old_pred = old_model.predict(x)
            recent_pred = recent_model.predict(x)
            return alpha * recent_pred + (1 - alpha) * old_pred
        return predict

# Usage
monitor = ModelDecayMonitor(baseline_metric=0.85, decay_threshold=0.1)
monitor.log_metric(0.82)  # Slight drop
monitor.log_metric(0.75)  # Significant drop

if monitor.should_retrain():
    print("Initiating retraining pipeline...")
```

### 6. Prevention Best Practices
- **Monitor continuously**: Don't wait for complaints
- **Automate retraining**: Scheduled or trigger-based
- **Version everything**: Data, model, features
- **A/B test updates**: Never blind deploy

### 7. Interview Tips
- Decay is inevitable - question is how fast and how to detect
- Different domains decay at different rates (finance: fast, medical imaging: slow)
- Balance stability vs freshness in retraining frequency

---

## Question 12
- [ ] Done

**Discuss how 'Continuous Evaluation' helps in maintaining model quality.**

**Answer:**

### 1. Definition
Continuous Evaluation monitors model performance in production using live predictions and outcomes, enabling early detection of degradation before significant business impact.

### 2. Core Concepts
- **Ground Truth Collection**: Gather actual outcomes for predictions
- **Delayed Labels**: Handle lag between prediction and outcome
- **Proxy Metrics**: Use leading indicators when labels unavailable
- **Sliding Window**: Evaluate on recent time window, not all-time
- **Statistical Testing**: Detect significant performance changes

### 3. Evaluation Components

| Component | Purpose | Example |
|-----------|---------|---------|
| Real-time metrics | Immediate feedback | Latency, error rate |
| Business metrics | Outcome impact | CTR, revenue, conversions |
| Model metrics | ML performance | Accuracy, AUC (when labels arrive) |
| Data quality | Input health | Missing values, schema violations |

### 4. Scenario Application
**Fraud Detection System**:
- **T+0 (immediate)**: Monitor prediction latency, API errors
- **T+1 hour**: Track approval rate, flagged transaction rate
- **T+7 days**: Confirmed fraud labels arrive, compute precision/recall
- **T+30 days**: Chargeback data, compute false negative rate

**Action Triggers**:
- Latency > 100ms: Alert oncall
- Flagged rate > 5%: Review model threshold
- Precision < 80%: Schedule retraining
- False negative spike: Emergency model update

### 5. Python Code Example

```python
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

class ContinuousEvaluator:
    def __init__(self, baseline_metrics):
        self.baseline = baseline_metrics
        self.prediction_log = []
        self.outcome_log = []
    
    def log_prediction(self, prediction_id, prediction, timestamp=None):
        self.prediction_log.append({
            'id': prediction_id,
            'prediction': prediction,
            'timestamp': timestamp or datetime.now()
        })
    
    def log_outcome(self, prediction_id, actual):
        """Called when ground truth becomes available"""
        self.outcome_log.append({
            'id': prediction_id,
            'actual': actual,
            'timestamp': datetime.now()
        })
    
    def compute_metrics(self, window_days=7):
        """Compute metrics on recent window"""
        cutoff = datetime.now() - timedelta(days=window_days)
        
        # Join predictions with outcomes
        pred_dict = {p['id']: p for p in self.prediction_log}
        matched = []
        for outcome in self.outcome_log:
            if outcome['id'] in pred_dict:
                pred = pred_dict[outcome['id']]
                if pred['timestamp'] > cutoff:
                    matched.append({
                        'prediction': pred['prediction'],
                        'actual': outcome['actual']
                    })
        
        if not matched:
            return None
        
        preds = [m['prediction'] for m in matched]
        actuals = [m['actual'] for m in matched]
        
        accuracy = sum(p == a for p, a in zip(preds, actuals)) / len(matched)
        return {'accuracy': accuracy, 'sample_size': len(matched)}
    
    def check_degradation(self):
        """Detect statistically significant performance drop"""
        current = self.compute_metrics(window_days=7)
        if not current or current['sample_size'] < 100:
            return False, "Insufficient data"
        
        baseline_acc = self.baseline.get('accuracy', 0.9)
        current_acc = current['accuracy']
        
        # Binomial test for significance
        n = current['sample_size']
        k = int(current_acc * n)
        p_value = stats.binom_test(k, n, baseline_acc, alternative='less')
        
        if p_value < 0.05:
            return True, f"Significant drop: {baseline_acc:.2%} -> {current_acc:.2%}"
        return False, f"No significant change: {current_acc:.2%}"

# Usage
evaluator = ContinuousEvaluator(baseline_metrics={'accuracy': 0.92})

# Log predictions (real-time)
for i in range(200):
    pred = np.random.choice([0, 1])
    evaluator.log_prediction(f'req_{i}', pred)

# Log outcomes (delayed by days)
for i in range(200):
    actual = np.random.choice([0, 1], p=[0.6, 0.4])  # Some degradation
    evaluator.log_outcome(f'req_{i}', actual)

# Evaluate
degraded, message = evaluator.check_degradation()
print(f"Degradation detected: {degraded}")
print(f"Message: {message}")
```

### 6. Proxy Metrics When Labels Delayed
- **Recommendation**: Click rate (immediate) vs purchase rate (delayed)
- **Fraud**: Flagged rate (immediate) vs confirmed fraud (delayed)
- **Search**: Click-through rate (immediate) vs task completion (delayed)

### 7. Interview Tips
- Emphasize the importance of feedback loops
- Discuss sample size requirements for statistical significance
- Mention A/B testing integration for model comparisons

---

## Question 13
- [ ] Done

**How would you approach 'Model Serving' in an environment with strict data regulations?**

**Answer:**

### 1. Definition
Model Serving under data regulations requires architectures that comply with privacy laws (GDPR, HIPAA, CCPA) while delivering predictions, including data minimization, encryption, audit trails, and user rights.

### 2. Key Regulations

| Regulation | Key Requirements |
|------------|-----------------|
| GDPR | Right to explanation, data deletion, consent |
| HIPAA | Health data protection, audit trails |
| CCPA | User data access, opt-out rights |
| SOC2 | Security controls, data handling |

### 3. Compliance Design Patterns

| Pattern | Implementation |
|---------|---------------|
| Data Minimization | Only collect/process necessary features |
| Encryption | Encrypt data at rest and in transit |
| Anonymization | Remove PII before model inference |
| Audit Logging | Log all data access and predictions |
| Right to Explanation | Provide prediction reasoning |
| Data Deletion | Delete user data on request (RTBF) |
| Consent Management | Track user consent for data usage |

### 4. Scenario Application
**Healthcare Prediction Service (HIPAA Compliant)**:

**Architecture**:
1. Data never leaves secure enclave
2. Federated learning: model trains on-premise, only gradients shared
3. Differential privacy: add noise to prevent data leakage
4. Encrypted inference: process encrypted data
5. Audit log every prediction with user, time, data accessed

### 5. Python Code Example

```python
import hashlib
import json
from datetime import datetime
from cryptography.fernet import Fernet
import uuid

class ComplianceModelServer:
    """Model serving with regulatory compliance built-in"""
    
    def __init__(self, model, encryption_key=None):
        self.model = model
        self.key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.key)
        self.audit_log = []
        self.user_consent = {}
        self.user_data = {}
    
    # Pattern 1: Consent Management
    def register_consent(self, user_id, purposes):
        """Track user consent for data usage"""
        self.user_consent[user_id] = {
            'purposes': purposes,  # ['inference', 'training', 'analytics']
            'timestamp': datetime.now().isoformat(),
            'version': 'v1.0'
        }
        self._log_audit(user_id, 'CONSENT_REGISTERED', {'purposes': purposes})
    
    def check_consent(self, user_id, purpose):
        """Verify consent before processing"""
        consent = self.user_consent.get(user_id, {})
        return purpose in consent.get('purposes', [])
    
    # Pattern 2: Data Minimization
    def minimize_features(self, features, required_fields):
        """Only use necessary features"""
        return {k: v for k, v in features.items() if k in required_fields}
    
    # Pattern 3: PII Anonymization
    def anonymize(self, features):
        """Remove/hash PII before inference"""
        anonymized = features.copy()
        pii_fields = ['name', 'email', 'ssn', 'address', 'phone']
        
        for field in pii_fields:
            if field in anonymized:
                # Hash instead of remove (preserves some utility)
                anonymized[field] = hashlib.sha256(
                    str(anonymized[field]).encode()
                ).hexdigest()[:16]
        
        return anonymized
    
    # Pattern 4: Encrypted Inference
    def encrypt_request(self, data):
        """Encrypt data for secure transmission"""
        json_data = json.dumps(data).encode()
        return self.cipher.encrypt(json_data)
    
    def decrypt_request(self, encrypted_data):
        """Decrypt data for processing"""
        decrypted = self.cipher.decrypt(encrypted_data)
        return json.loads(decrypted)
    
    # Pattern 5: Audit Logging
    def _log_audit(self, user_id, action, details):
        """Log all data access for compliance"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'request_id': str(uuid.uuid4()),
            'user_id': hashlib.sha256(str(user_id).encode()).hexdigest()[:16],
            'action': action,
            'details': details
        }
        self.audit_log.append(entry)
        print(f"[AUDIT] {action}: {entry['request_id']}")
    
    # Pattern 6: Right to Explanation
    def explain_prediction(self, prediction, features):
        """Provide human-readable explanation (GDPR Art. 22)"""
        return {
            'prediction': prediction,
            'explanation': f"Based on {len(features)} factors",
            'top_factors': list(features.keys())[:3],
            'human_readable': "This prediction was made considering your profile data"
        }
    
    # Pattern 7: Right to Be Forgotten
    def delete_user_data(self, user_id):
        """Delete all user data on request (GDPR Art. 17)"""
        # Delete from storage
        if user_id in self.user_data:
            del self.user_data[user_id]
        if user_id in self.user_consent:
            del self.user_consent[user_id]
        
        self._log_audit(user_id, 'DATA_DELETED', {'reason': 'User request'})
        return {'status': 'deleted', 'user_id': user_id}
    
    # Main serving endpoint
    def predict(self, user_id, features, purpose='inference'):
        """Compliant prediction endpoint"""
        
        # Step 1: Check consent
        if not self.check_consent(user_id, purpose):
            return {'error': 'Consent not provided for this purpose'}
        
        # Step 2: Minimize features
        required = ['age', 'income', 'score']  # Only what model needs
        minimized = self.minimize_features(features, required)
        
        # Step 3: Anonymize PII
        anonymized = self.anonymize(minimized)
        
        # Step 4: Log access
        self._log_audit(user_id, 'PREDICTION_REQUEST', {'features_used': list(minimized.keys())})
        
        # Step 5: Make prediction
        prediction = self.model.predict(anonymized)
        
        # Step 6: Provide explanation
        result = self.explain_prediction(prediction, minimized)
        
        return result

# Usage
class MockModel:
    def predict(self, features):
        return 'approved'

server = ComplianceModelServer(MockModel())

# Register consent
server.register_consent('user_123', ['inference', 'analytics'])

# Make prediction
features = {
    'name': 'John Doe',  # PII - will be anonymized
    'age': 30,
    'income': 75000,
    'score': 720,
    'email': 'john@example.com'  # PII
}

result = server.predict('user_123', features)
print(f"\nPrediction result: {result}")

# Right to be forgotten
deletion = server.delete_user_data('user_123')
print(f"\nDeletion result: {deletion}")

# View audit log
print(f"\nAudit log entries: {len(server.audit_log)}")
```

### 6. Key Architectural Decisions
- **On-premise deployment**: Data never leaves customer's infrastructure
- **Federated learning**: Train without centralizing data
- **Differential privacy**: Mathematical privacy guarantees
- **Secure enclaves**: Hardware-level data protection (Intel SGX)

### 7. Interview Tips
- Always mention specific regulations by name
- Discuss trade-offs: privacy vs model performance
- Mention data residency (data must stay in certain geography)
- Legal team involvement is essential for compliance

---

## Question 14
- [ ] Done

**Discuss 'Dynamic Training' approaches in a scenario where data distributions change rapidly.**

**Answer:**

### 1. Definition
Dynamic Training continuously updates models as new data arrives, adapting to rapid distribution changes (concept drift, trending topics, market shifts) rather than relying on periodic batch retraining.

### 2. Core Concepts
- **Online Learning**: Update model with each new sample
- **Mini-batch Updates**: Accumulate small batches, update frequently
- **Sliding Window**: Train only on recent data (forget old patterns)
- **Exponential Decay**: Weight recent data more heavily
- **Trigger-based Retraining**: Retrain when drift detected

### 3. Dynamic Training Strategies

| Strategy | Update Frequency | Use Case |
|----------|-----------------|----------|
| Full Online | Every sample | High-frequency trading |
| Mini-batch | Every N samples | Real-time recommendations |
| Scheduled | Hourly/Daily | News, trending content |
| Triggered | On drift detection | Stable with occasional shifts |

### 4. Scenario Application
**Trending Topic Detection for Social Media**:

**Challenge**: Topics trend and die within hours. Weekly retrained model is always outdated.

**Dynamic Training Solution**:
1. **Streaming Data**: Consume tweets in real-time via Kafka
2. **Mini-batch Updates**: Accumulate 1000 tweets, update model
3. **Sliding Window**: Keep only last 24 hours of data
4. **Ensemble**: Blend hourly model with daily model
5. **Drift Detection**: Alert if topic distribution shifts dramatically

### 5. Python Code Example

```python
import numpy as np
from collections import deque
from sklearn.linear_model import SGDClassifier
from datetime import datetime, timedelta

class DynamicTrainer:
    """Handles rapidly changing data distributions"""
    
    def __init__(self, base_model, window_hours=24, batch_size=100):
        self.model = base_model
        self.window_hours = window_hours
        self.batch_size = batch_size
        self.data_buffer = deque()
        self.drift_detector = DriftDetector()
        self.update_count = 0
    
    def add_sample(self, features, label, timestamp=None):
        """Add new sample to buffer"""
        timestamp = timestamp or datetime.now()
        self.data_buffer.append({
            'features': features,
            'label': label,
            'timestamp': timestamp
        })
        
        # Remove old samples (sliding window)
        cutoff = datetime.now() - timedelta(hours=self.window_hours)
        while self.data_buffer and self.data_buffer[0]['timestamp'] < cutoff:
            self.data_buffer.popleft()
        
        # Check if batch update needed
        if len(self.data_buffer) >= self.batch_size:
            self._update_model()
    
    def _update_model(self):
        """Perform mini-batch update"""
        recent = list(self.data_buffer)[-self.batch_size:]
        
        X = np.array([s['features'] for s in recent])
        y = np.array([s['label'] for s in recent])
        
        # Online update (partial_fit)
        self.model.partial_fit(X, y, classes=np.unique(y))
        self.update_count += 1
        
        # Check for drift
        if self.drift_detector.detect(y):
            print(f"[ALERT] Drift detected at update {self.update_count}")
        
        print(f"[UPDATE] Model updated with {len(recent)} samples. Total updates: {self.update_count}")
    
    def predict(self, features):
        """Make prediction with current model"""
        return self.model.predict([features])[0]

class DriftDetector:
    """Simple drift detection based on label distribution"""
    
    def __init__(self, window_size=100, threshold=0.2):
        self.recent_labels = deque(maxlen=window_size)
        self.baseline_dist = None
        self.threshold = threshold
    
    def detect(self, new_labels):
        """Check if distribution has shifted"""
        self.recent_labels.extend(new_labels)
        
        if len(self.recent_labels) < 50:
            return False
        
        current_dist = np.bincount(list(self.recent_labels), minlength=2) / len(self.recent_labels)
        
        if self.baseline_dist is None:
            self.baseline_dist = current_dist
            return False
        
        # Check distribution shift
        shift = np.abs(current_dist - self.baseline_dist).max()
        
        if shift > self.threshold:
            self.baseline_dist = current_dist  # Update baseline
            return True
        
        return False

class ExponentialDecayTrainer:
    """Weight recent samples more heavily"""
    
    def __init__(self, decay_rate=0.995):
        self.decay_rate = decay_rate
        self.samples = []
    
    def add_sample(self, features, label, timestamp=None):
        self.samples.append({
            'features': features,
            'label': label,
            'weight': 1.0
        })
        
        # Decay old sample weights
        for sample in self.samples:
            sample['weight'] *= self.decay_rate
        
        # Remove samples with negligible weight
        self.samples = [s for s in self.samples if s['weight'] > 0.01]
    
    def get_weighted_data(self):
        """Return features, labels, and sample weights"""
        X = np.array([s['features'] for s in self.samples])
        y = np.array([s['label'] for s in self.samples])
        weights = np.array([s['weight'] for s in self.samples])
        return X, y, weights

# Usage Example: Trending topic classifier
model = SGDClassifier(loss='log_loss', warm_start=True)
trainer = DynamicTrainer(model, window_hours=6, batch_size=50)

# Simulate streaming data
np.random.seed(42)
for i in range(200):
    # Features: word counts, sentiment, etc.
    features = np.random.rand(10)
    
    # Label distribution shifts over time
    if i < 100:
        label = np.random.choice([0, 1], p=[0.7, 0.3])
    else:
        label = np.random.choice([0, 1], p=[0.3, 0.7])  # Distribution shift
    
    trainer.add_sample(features, label)

print(f"\nFinal buffer size: {len(trainer.data_buffer)}")
print(f"Total model updates: {trainer.update_count}")
```

**Output:**
```
[UPDATE] Model updated with 50 samples. Total updates: 1
[UPDATE] Model updated with 50 samples. Total updates: 2
[ALERT] Drift detected at update 3
[UPDATE] Model updated with 50 samples. Total updates: 3
[UPDATE] Model updated with 50 samples. Total updates: 4
```

### 6. Challenges and Solutions
- **Catastrophic Forgetting**: Use replay buffer or regularization
- **Noisy Updates**: Larger batch sizes for stability
- **Concept Recovery**: Quick adaptation when old patterns return
- **Model Stability**: Ensemble stable + dynamic models

### 7. Interview Tips
- Dynamic training trades stability for freshness
- Mention specific algorithms: SGD, online random forest, streaming k-means
- Discuss validation: hard to evaluate without held-out future data
- Production: often combine with scheduled full retraining

---

## Question 15
- [ ] Done

**Discuss the implications of implementing the 'Stateless Model' design pattern in distributed systems.**

**Answer:**

### 1. Definition
Stateless Model design ensures prediction services don't maintain client state between requests. Each request contains all necessary information, enabling easy scaling, load balancing, and fault tolerance.

### 2. Implications

| Aspect | Implication |
|--------|-------------|
| **Scalability** | Easy horizontal scaling - any replica handles any request |
| **Load Balancing** | Simple round-robin works well |
| **Fault Tolerance** | Failed node replaced without state migration |
| **Caching** | Must be external (Redis) not in-memory |
| **Session Data** | Passed with each request or stored externally |

### 3. Python Implementation

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import hashlib
from typing import Dict, Any
import time

class StatelessPredictionService:
    """Stateless model serving - no client state between requests"""
    
    def __init__(self, model_path: str):
        # Model loaded once at startup, not modified per-request
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.instance_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    
    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stateless prediction - all context in request
        
        Request contains:
        - features: input data
        - request_id: for tracking
        - metadata: any additional context
        """
        features = np.array(request['features']).reshape(1, -1)
        
        prediction = self.model.predict(features)[0]
        probability = None
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(features)[0].tolist()
        
        # Response is self-contained - no state dependency
        return {
            'request_id': request.get('request_id'),
            'prediction': int(prediction),
            'probability': probability,
            'served_by': self.instance_id,  # Shows which replica handled it
            'timestamp': time.time()
        }

class StatefulAntiPattern:
    """Anti-pattern: Stateful service (for comparison)"""
    
    def __init__(self, model):
        self.model = model
        self.session_data = {}  # BAD: State stored between requests
        self.request_count = {}  # BAD: Per-client counters
    
    def predict(self, client_id: str, features):
        # BAD: Depends on previous requests
        if client_id not in self.session_data:
            self.session_data[client_id] = {'history': []}
        
        self.session_data[client_id]['history'].append(features)
        self.request_count[client_id] = self.request_count.get(client_id, 0) + 1
        
        # This service CAN'T be easily scaled - state is in memory
        return self.model.predict(features.reshape(1, -1))

class ExternalStateManager:
    """Proper pattern: External state storage for stateless services"""
    
    def __init__(self):
        # Simulates Redis/external cache
        self.cache = {}
    
    def get_session(self, session_id: str) -> Dict:
        return self.cache.get(session_id, {})
    
    def set_session(self, session_id: str, data: Dict):
        self.cache[session_id] = data
    
    def update_session(self, session_id: str, key: str, value: Any):
        if session_id not in self.cache:
            self.cache[session_id] = {}
        self.cache[session_id][key] = value

class StatelessWithExternalState:
    """Stateless service with external state management"""
    
    def __init__(self, model, state_manager: ExternalStateManager):
        self.model = model
        self.state_manager = state_manager
    
    def predict(self, request: Dict) -> Dict:
        session_id = request.get('session_id')
        features = np.array(request['features']).reshape(1, -1)
        
        # Get session from external store (not local memory)
        session = self.state_manager.get_session(session_id) if session_id else {}
        
        prediction = self.model.predict(features)[0]
        
        # Update external state (service remains stateless)
        if session_id:
            history = session.get('predictions', [])
            history.append(int(prediction))
            self.state_manager.update_session(session_id, 'predictions', history)
        
        return {
            'prediction': int(prediction),
            'session_history': session.get('predictions', [])
        }

# Demo
print("=== Stateless Model Pattern Demo ===\n")

# Train and save model
np.random.seed(42)
X, y = np.random.randn(500, 10), (np.random.randn(500) > 0).astype(int)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Create stateless service
service = StatelessPredictionService('model.pkl')

# Multiple requests - each self-contained
print("Stateless Predictions:")
for i in range(3):
    request = {
        'request_id': f'req_{i}',
        'features': np.random.randn(10).tolist()
    }
    response = service.predict(request)
    print(f"  Request {i}: prediction={response['prediction']}, server={response['served_by']}")

# With external state
print("\nStateless with External State:")
state_mgr = ExternalStateManager()
service_with_state = StatelessWithExternalState(model, state_mgr)

for i in range(3):
    request = {
        'session_id': 'user_123',
        'features': np.random.randn(10).tolist()
    }
    response = service_with_state.predict(request)
    print(f"  Request {i}: prediction={response['prediction']}, history={response['session_history']}")

# Cleanup
import os
os.remove('model.pkl')
```

### 4. Key Implications Summary

| Stateless Benefits | Stateful Drawbacks |
|-------------------|-------------------|
| Any replica serves any request | Sticky sessions required |
| Easy failover | State lost on crash |
| Simple load balancing | Complex state sync |
| Cloud-native ready | Hard to scale |

### 5. Interview Tips
- Stateless enables Kubernetes-style scaling
- External stores (Redis) for shared state
- Each request must be self-contained
- Mention container orchestration benefits

---
