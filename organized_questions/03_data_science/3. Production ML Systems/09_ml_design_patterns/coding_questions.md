# Ml Design Patterns Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Implement a basic â€˜Replayâ€™ pattern mechanism using Python to simulate model retraining with different datasets.**

**Answer:**

### 1. Definition
The Replay pattern stores all training inputs (data, configs, code versions) to enable exact reproduction of any model training run. This ensures reproducibility and enables debugging of past model versions.

### 2. Key Components

| Component | Purpose |
|-----------|---------|
| **Data Snapshot** | Versioned copy of training data |
| **Config Store** | Hyperparameters, settings |
| **Code Version** | Git commit hash |
| **Artifact Store** | Models, metrics, logs |
| **Replay Engine** | Re-execute training |

### 3. Python Implementation

```python
import numpy as np
import hashlib
import json
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any, Optional
import subprocess

class DataSnapshot:
    """Snapshot and version training data"""
    
    def __init__(self, storage_path: str = "data_snapshots"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def save(self, X: np.ndarray, y: np.ndarray, name: str) -> str:
        """Save data snapshot and return version hash"""
        # Create hash from data content
        data_bytes = X.tobytes() + y.tobytes()
        version = hashlib.sha256(data_bytes).hexdigest()[:12]
        
        # Save data
        snapshot = {'X': X, 'y': y, 'name': name}
        path = os.path.join(self.storage_path, f"{name}_{version}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(snapshot, f)
        
        return version
    
    def load(self, name: str, version: str):
        """Load data snapshot"""
        path = os.path.join(self.storage_path, f"{name}_{version}.pkl")
        with open(path, 'rb') as f:
            snapshot = pickle.load(f)
        return snapshot['X'], snapshot['y']
    
    def list_versions(self, name: str):
        """List all versions for a dataset"""
        versions = []
        for f in os.listdir(self.storage_path):
            if f.startswith(name) and f.endswith('.pkl'):
                version = f.replace(f"{name}_", "").replace(".pkl", "")
                versions.append(version)
        return versions

class ReplayStore:
    """Store training runs for replay"""
    
    def __init__(self, storage_path: str = "replay_store"):
        self.storage_path = storage_path
        self.runs = {}
        os.makedirs(storage_path, exist_ok=True)
        self._load_index()
    
    def _load_index(self):
        """Load run index"""
        index_path = os.path.join(self.storage_path, "index.json")
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                self.runs = json.load(f)
    
    def _save_index(self):
        """Save run index"""
        index_path = os.path.join(self.storage_path, "index.json")
        with open(index_path, 'w') as f:
            json.dump(self.runs, f, indent=2)
    
    def log_run(self, run_id: str, data_version: str, config: Dict, 
                metrics: Dict, model_path: str):
        """Log a training run"""
        self.runs[run_id] = {
            'run_id': run_id,
            'data_version': data_version,
            'config': config,
            'metrics': metrics,
            'model_path': model_path,
            'timestamp': datetime.now().isoformat(),
            'git_commit': self._get_git_commit()
        }
        self._save_index()
    
    def _get_git_commit(self) -> str:
        """Get current git commit"""
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', 'HEAD']
            ).decode().strip()[:8]
        except:
            return "unknown"
    
    def get_run(self, run_id: str) -> Optional[Dict]:
        """Get run details"""
        return self.runs.get(run_id)
    
    def list_runs(self):
        """List all runs"""
        return list(self.runs.values())

class ReplayEngine:
    """Engine to replay training runs"""
    
    def __init__(self):
        self.data_snapshots = DataSnapshot()
        self.replay_store = ReplayStore()
        self.model_classes = {
            'RandomForest': RandomForestClassifier,
            'GradientBoosting': GradientBoostingClassifier,
            'LogisticRegression': LogisticRegression
        }
    
    def train_and_log(self, X: np.ndarray, y: np.ndarray, 
                      config: Dict, run_name: str) -> str:
        """Train model and log for replay"""
        # Snapshot data
        data_version = self.data_snapshots.save(X, y, run_name)
        
        # Create run ID
        run_id = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Train model
        model_class = self.model_classes[config['model_type']]
        model_params = {k: v for k, v in config.items() if k != 'model_type'}
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.get('random_state', 42)
        )
        
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Save model
        model_path = f"models/{run_id}.pkl"
        os.makedirs("models", exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Log run
        self.replay_store.log_run(run_id, data_version, config, metrics, model_path)
        
        print(f"Run logged: {run_id}")
        print(f"  Data version: {data_version}")
        print(f"  Metrics: {metrics}")
        
        return run_id
    
    def replay(self, run_id: str) -> Dict:
        """Replay a previous training run"""
        run = self.replay_store.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")
        
        print(f"\n=== Replaying Run: {run_id} ===")
        print(f"Original timestamp: {run['timestamp']}")
        print(f"Original config: {run['config']}")
        
        # Load data snapshot
        data_name = run_id.rsplit('_', 2)[0]
        X, y = self.data_snapshots.load(data_name, run['data_version'])
        
        # Retrain with same config
        config = run['config']
        model_class = self.model_classes[config['model_type']]
        model_params = {k: v for k, v in config.items() if k != 'model_type'}
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.get('random_state', 42)
        )
        
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        replay_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Compare
        result = {
            'original_metrics': run['metrics'],
            'replay_metrics': replay_metrics,
            'metrics_match': np.isclose(
                run['metrics']['accuracy'], 
                replay_metrics['accuracy'],
                rtol=1e-5
            )
        }
        
        print(f"Original metrics: {run['metrics']}")
        print(f"Replay metrics: {replay_metrics}")
        print(f"Metrics match: {result['metrics_match']}")
        
        return result
    
    def compare_runs(self, run_ids: list) -> Dict:
        """Compare multiple runs"""
        comparison = []
        for run_id in run_ids:
            run = self.replay_store.get_run(run_id)
            if run:
                comparison.append({
                    'run_id': run_id,
                    'config': run['config'],
                    'accuracy': run['metrics']['accuracy'],
                    'f1': run['metrics']['f1']
                })
        
        # Sort by accuracy
        comparison.sort(key=lambda x: x['accuracy'], reverse=True)
        return comparison

# Demo
print("=== Replay Pattern Demo ===\n")

engine = ReplayEngine()

# Generate sample data
np.random.seed(42)
X = np.random.randn(500, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Run 1: Random Forest
run1 = engine.train_and_log(X, y, {
    'model_type': 'RandomForest',
    'n_estimators': 100,
    'max_depth': 5,
    'random_state': 42
}, 'experiment')

# Run 2: Gradient Boosting
run2 = engine.train_and_log(X, y, {
    'model_type': 'GradientBoosting',
    'n_estimators': 50,
    'learning_rate': 0.1,
    'random_state': 42
}, 'experiment')

# Run 3: Different hyperparameters
run3 = engine.train_and_log(X, y, {
    'model_type': 'RandomForest',
    'n_estimators': 200,
    'max_depth': 10,
    'random_state': 42
}, 'experiment')

# Replay first run
print("\n")
replay_result = engine.replay(run1)

# Compare all runs
print("\n=== Run Comparison ===")
comparison = engine.compare_runs([run1, run2, run3])
for run in comparison:
    print(f"  {run['run_id']}: accuracy={run['accuracy']:.4f}")
```

### 4. Interview Tips
- Emphasize deterministic training (fixed seeds)
- Data versioning is critical for true replay
- Include environment (dependencies) for full reproducibility
- Mention MLflow, DVC as production tools

---

## Question 2
- [ ] Done

**Develop a simple ensemble model using Python, following the â€˜Model Ensembleâ€™ design pattern.**

**Answer:**

### 1. Definition
Model Ensemble combines multiple models to produce better predictions than any single model. It reduces variance, bias, or improves overall accuracy through techniques like voting, averaging, stacking, or boosting.

### 2. Ensemble Types

| Type | Method | Use Case |
|------|--------|----------|
| **Voting** | Majority vote (classification) | Diverse models |
| **Averaging** | Mean predictions (regression) | Reduce variance |
| **Weighted** | Weighted combination | Different model quality |
| **Stacking** | Meta-learner on base predictions | Maximum accuracy |
| **Bagging** | Bootstrap aggregating | Reduce overfitting |
| **Boosting** | Sequential error correction | Bias reduction |

### 3. Python Implementation

```python
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class VotingEnsemble:
    """Hard and soft voting ensemble"""
    
    def __init__(self, models: List, voting: str = 'hard', weights: List = None):
        self.models = models
        self.voting = voting
        self.weights = weights or [1] * len(models)
    
    def fit(self, X, y):
        """Fit all base models"""
        for name, model in self.models:
            model.fit(X, y)
        self.classes_ = np.unique(y)
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.voting == 'hard':
            return self._hard_vote(X)
        else:
            return self._soft_vote(X)
    
    def _hard_vote(self, X):
        """Majority voting"""
        predictions = np.array([model.predict(X) for _, model in self.models])
        # Weighted voting
        result = []
        for i in range(X.shape[0]):
            votes = {}
            for j, pred in enumerate(predictions[:, i]):
                votes[pred] = votes.get(pred, 0) + self.weights[j]
            result.append(max(votes, key=votes.get))
        return np.array(result)
    
    def _soft_vote(self, X):
        """Probability-based voting"""
        probas = []
        for (name, model), weight in zip(self.models, self.weights):
            if hasattr(model, 'predict_proba'):
                probas.append(model.predict_proba(X) * weight)
            else:
                # Fall back to hard predictions
                pred = model.predict(X)
                one_hot = np.zeros((len(pred), len(self.classes_)))
                for i, p in enumerate(pred):
                    one_hot[i, np.where(self.classes_ == p)[0][0]] = weight
                probas.append(one_hot)
        
        avg_proba = np.mean(probas, axis=0)
        return self.classes_[np.argmax(avg_proba, axis=1)]
    
    def score(self, X, y):
        """Calculate accuracy"""
        return accuracy_score(y, self.predict(X))

class StackingEnsemble:
    """Stacking ensemble with meta-learner"""
    
    def __init__(self, base_models: List, meta_model, use_proba: bool = True):
        self.base_models = base_models
        self.meta_model = meta_model
        self.use_proba = use_proba
    
    def fit(self, X, y):
        """Fit base models and meta-learner"""
        # Fit base models
        for name, model in self.base_models:
            model.fit(X, y)
        
        # Create meta-features
        meta_features = self._get_meta_features(X)
        
        # Fit meta-model
        self.meta_model.fit(meta_features, y)
        self.classes_ = np.unique(y)
        return self
    
    def _get_meta_features(self, X):
        """Get predictions from base models as features"""
        meta_features = []
        for name, model in self.base_models:
            if self.use_proba and hasattr(model, 'predict_proba'):
                meta_features.append(model.predict_proba(X))
            else:
                meta_features.append(model.predict(X).reshape(-1, 1))
        return np.hstack(meta_features)
    
    def predict(self, X):
        """Make predictions using meta-learner"""
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)
    
    def score(self, X, y):
        """Calculate accuracy"""
        return accuracy_score(y, self.predict(X))

class WeightedAverageEnsemble:
    """Weighted average for regression"""
    
    def __init__(self, models: List, weights: List = None):
        self.models = models
        self.weights = weights
    
    def fit(self, X, y):
        """Fit all models and optionally learn weights"""
        for name, model in self.models:
            model.fit(X, y)
        
        if self.weights is None:
            # Learn weights using validation performance
            self.weights = self._learn_weights(X, y)
        
        # Normalize weights
        self.weights = np.array(self.weights) / sum(self.weights)
        return self
    
    def _learn_weights(self, X, y):
        """Learn optimal weights from CV scores"""
        weights = []
        for name, model in self.models:
            scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
            # Higher weight for better (less negative) MSE
            weights.append(1 / (abs(scores.mean()) + 1e-6))
        return weights
    
    def predict(self, X):
        """Weighted average prediction"""
        predictions = np.array([model.predict(X) for _, model in self.models])
        return np.average(predictions, axis=0, weights=self.weights)
    
    def score(self, X, y):
        """Calculate R^2"""
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))

class BaggingEnsemble:
    """Bootstrap aggregating ensemble"""
    
    def __init__(self, base_model_class, n_estimators: int = 10, 
                 sample_ratio: float = 0.8):
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.sample_ratio = sample_ratio
        self.models = []
    
    def fit(self, X, y):
        """Fit models on bootstrap samples"""
        n_samples = int(len(X) * self.sample_ratio)
        
        for i in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(len(X), n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            
            # Train model
            model = self.base_model_class()
            model.fit(X_sample, y_sample)
            self.models.append(model)
        
        self.classes_ = np.unique(y)
        return self
    
    def predict(self, X):
        """Majority vote from all models"""
        predictions = np.array([model.predict(X) for model in self.models])
        # Mode for each sample
        from scipy import stats
        mode_result = stats.mode(predictions, axis=0, keepdims=False)
        return mode_result.mode
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

# Demo
print("=== Model Ensemble Pattern Demo ===\n")

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# 1. Voting Ensemble
print("1. Voting Ensemble")
voting = VotingEnsemble(
    [('rf', RandomForestClassifier(n_estimators=50)),
     ('gb', GradientBoostingClassifier(n_estimators=50)),
     ('lr', LogisticRegression(max_iter=1000))],
    voting='soft'
)
voting.fit(X_train, y_train)
print(f"   Accuracy: {voting.score(X_test, y_test):.4f}")

# 2. Stacking Ensemble
print("\n2. Stacking Ensemble")
stacking = StackingEnsemble(
    base_models=[
        ('rf', RandomForestClassifier(n_estimators=50)),
        ('gb', GradientBoostingClassifier(n_estimators=50)),
        ('knn', KNeighborsClassifier())
    ],
    meta_model=LogisticRegression(max_iter=1000)
)
stacking.fit(X_train, y_train)
print(f"   Accuracy: {stacking.score(X_test, y_test):.4f}")

# 3. Bagging Ensemble
print("\n3. Bagging Ensemble")
bagging = BaggingEnsemble(DecisionTreeClassifier, n_estimators=20)
bagging.fit(X_train, y_train)
print(f"   Accuracy: {bagging.score(X_test, y_test):.4f}")

# 4. Compare with individual models
print("\n4. Individual Model Comparison")
for name, model_class in [('RandomForest', RandomForestClassifier),
                          ('GradientBoosting', GradientBoostingClassifier),
                          ('LogisticRegression', LogisticRegression)]:
    model = model_class() if name != 'LogisticRegression' else model_class(max_iter=1000)
    model.fit(X_train, y_train)
    print(f"   {name}: {model.score(X_test, y_test):.4f}")
```

### 4. Interview Tips
- Ensemble reduces variance (bagging) or bias (boosting)
- Stacking often gives best results but is complex
- Diverse models work better in ensembles
- Consider computational cost vs accuracy gain

---

## Question 3
- [ ] Done

**Write a script to perform batch serving of a machine learning model using dummy data.**

**Answer:**

### 1. Definition
Batch Serving processes large volumes of data in bulk for offline predictions, as opposed to real-time serving. It's used for scheduled predictions, report generation, and processing historical data.

### 2. Batch vs Real-time

| Aspect | Batch Serving | Real-time Serving |
|--------|---------------|-------------------|
| **Latency** | Minutes to hours | Milliseconds |
| **Throughput** | Very high | Medium |
| **Use Case** | Reports, ETL | User-facing apps |
| **Cost** | Lower | Higher |

### 3. Python Implementation

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import time
import os
from datetime import datetime
from typing import List, Dict, Generator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchDataLoader:
    """Load data in batches for memory efficiency"""
    
    def __init__(self, data_source: str, batch_size: int = 1000):
        self.data_source = data_source
        self.batch_size = batch_size
    
    def generate_dummy_data(self, total_samples: int) -> pd.DataFrame:
        """Generate dummy data for demonstration"""
        np.random.seed(42)
        data = {
            'feature_1': np.random.randn(total_samples),
            'feature_2': np.random.randn(total_samples),
            'feature_3': np.random.rand(total_samples) * 100,
            'feature_4': np.random.randint(0, 10, total_samples),
            'feature_5': np.random.randn(total_samples),
            'user_id': range(total_samples)
        }
        return pd.DataFrame(data)
    
    def load_batches(self, total_samples: int) -> Generator:
        """Yield data in batches"""
        df = self.generate_dummy_data(total_samples)
        
        for start in range(0, len(df), self.batch_size):
            end = min(start + self.batch_size, len(df))
            yield df.iloc[start:end].copy()

class BatchPredictor:
    """Batch prediction engine"""
    
    def __init__(self, model, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_columns = None
    
    def set_feature_columns(self, columns: List[str]):
        """Set columns to use for prediction"""
        self.feature_columns = columns
    
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess batch data"""
        X = df[self.feature_columns].values
        if self.preprocessor:
            X = self.preprocessor.transform(X)
        return X
    
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on a batch"""
        X = self.preprocess(df)
        predictions = self.model.predict(X)
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[:, 1]
            return predictions, probabilities
        
        return predictions, None

class BatchServingPipeline:
    """Complete batch serving pipeline"""
    
    def __init__(self, model, preprocessor, output_path: str = "predictions"):
        self.predictor = BatchPredictor(model, preprocessor)
        self.output_path = output_path
        self.metrics = {
            'batches_processed': 0,
            'samples_processed': 0,
            'total_time': 0,
            'errors': 0
        }
        os.makedirs(output_path, exist_ok=True)
    
    def run(self, data_loader: BatchDataLoader, total_samples: int,
            feature_columns: List[str], id_column: str = 'user_id'):
        """Run batch prediction pipeline"""
        
        self.predictor.set_feature_columns(feature_columns)
        
        start_time = time.time()
        all_results = []
        
        logger.info(f"Starting batch prediction for {total_samples} samples")
        
        for batch_num, batch_df in enumerate(data_loader.load_batches(total_samples)):
            batch_start = time.time()
            
            try:
                # Make predictions
                predictions, probabilities = self.predictor.predict_batch(batch_df)
                
                # Create results DataFrame
                results = pd.DataFrame({
                    id_column: batch_df[id_column].values,
                    'prediction': predictions,
                    'probability': probabilities if probabilities is not None else np.nan,
                    'processed_at': datetime.now().isoformat()
                })
                
                all_results.append(results)
                
                self.metrics['batches_processed'] += 1
                self.metrics['samples_processed'] += len(batch_df)
                
                batch_time = time.time() - batch_start
                logger.info(f"Batch {batch_num + 1}: {len(batch_df)} samples in {batch_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in batch {batch_num + 1}: {e}")
                self.metrics['errors'] += 1
        
        self.metrics['total_time'] = time.time() - start_time
        
        # Combine and save results
        final_results = pd.concat(all_results, ignore_index=True)
        output_file = os.path.join(
            self.output_path, 
            f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        final_results.to_csv(output_file, index=False)
        
        logger.info(f"Results saved to {output_file}")
        self._print_summary()
        
        return final_results
    
    def _print_summary(self):
        """Print processing summary"""
        throughput = self.metrics['samples_processed'] / max(self.metrics['total_time'], 0.01)
        print("\n" + "="*50)
        print("BATCH PROCESSING SUMMARY")
        print("="*50)
        print(f"Batches processed: {self.metrics['batches_processed']}")
        print(f"Samples processed: {self.metrics['samples_processed']}")
        print(f"Total time: {self.metrics['total_time']:.2f} seconds")
        print(f"Throughput: {throughput:.0f} samples/second")
        print(f"Errors: {self.metrics['errors']}")
        print("="*50)

class ParallelBatchProcessor:
    """Parallel batch processing for faster throughput"""
    
    def __init__(self, model, preprocessor, n_workers: int = 4):
        self.model = model
        self.preprocessor = preprocessor
        self.n_workers = n_workers
    
    def process_parallel(self, data_loader: BatchDataLoader, 
                        total_samples: int,
                        feature_columns: List[str]) -> pd.DataFrame:
        """Process batches in parallel"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def process_batch(batch_df):
            X = batch_df[feature_columns].values
            if self.preprocessor:
                X = self.preprocessor.transform(X)
            predictions = self.model.predict(X)
            return batch_df.index.tolist(), predictions
        
        all_predictions = {}
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(process_batch, batch): batch_num 
                for batch_num, batch in enumerate(data_loader.load_batches(total_samples))
            }
            
            for future in as_completed(futures):
                indices, predictions = future.result()
                for idx, pred in zip(indices, predictions):
                    all_predictions[idx] = pred
        
        return all_predictions

# Demo
print("=== Batch Serving Pattern Demo ===\n")

# Train a sample model
logger.info("Training model...")
np.random.seed(42)
X_train = np.random.randn(1000, 5)
y_train = (X_train.sum(axis=1) > 0).astype(int)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train_scaled, y_train)

# Create batch serving pipeline
pipeline = BatchServingPipeline(model, scaler)
data_loader = BatchDataLoader("dummy", batch_size=500)

# Run batch predictions
feature_cols = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
results = pipeline.run(
    data_loader, 
    total_samples=5000,
    feature_columns=feature_cols
)

# Show sample results
print("\nSample predictions:")
print(results.head(10))
print(f"\nPrediction distribution:")
print(results['prediction'].value_counts())
```

### 4. Production Considerations

| Aspect | Implementation |
|--------|----------------|
| **Storage** | Parquet, Delta Lake |
| **Scheduling** | Airflow, Cron |
| **Scaling** | Spark, Dask |
| **Monitoring** | Row counts, latency |
| **Idempotency** | Overwrite partitions |

### 5. Interview Tips
- Discuss memory efficiency with batching
- Mention checkpointing for long jobs
- Parallel processing for throughput
- Consider output partitioning strategies

---

## Question 4
- [ ] Done

**Code a 'Model Checkpointing' system during training using TensorFlow/Keras.**

**Answer:**

### 1. Definition
Model Checkpointing saves model state at regular intervals during training, enabling recovery from failures and selection of the best performing model.

### 2. Python Implementation

```python
import numpy as np
import os
import pickle

class CheckpointManager:
    """Checkpoint manager for any model"""
    
    def __init__(self, checkpoint_dir, max_to_keep=5):
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.checkpoints = []
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, model, epoch, metrics, is_best=False):
        path = os.path.join(self.checkpoint_dir, f"ckpt_{epoch:04d}.pkl")
        with open(path, 'wb') as f:
            pickle.dump({'model': model, 'epoch': epoch, 'metrics': metrics}, f)
        
        self.checkpoints.append(path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pkl")
            with open(best_path, 'wb') as f:
                pickle.dump(model, f)
        
        while len(self.checkpoints) > self.max_to_keep:
            old = self.checkpoints.pop(0)
            if os.path.exists(old): os.remove(old)
    
    def load_best(self):
        with open(os.path.join(self.checkpoint_dir, "best.pkl"), 'rb') as f:
            return pickle.load(f)

# Keras checkpointing example
def keras_checkpointing():
    try:
        from tensorflow import keras
        
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                'ckpt/model_{epoch:02d}.keras',
                save_best_only=True, monitor='val_loss'
            ),
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
        
        X, y = np.random.randn(500, 10), np.random.randn(500)
        model.fit(X, y, epochs=20, validation_split=0.2, callbacks=callbacks, verbose=0)
        print("Keras training complete")
    except ImportError:
        print("TensorFlow not available")

keras_checkpointing()
```

### 3. Interview Tips
- Save optimizer state for resume capability
- Keep N best checkpoints, not all
- Use validation metric to determine "best"

---

## Question 5
- [ ] Done

**Create a feature store simulation in Python to demonstrate sharing and reuse of feature transformation code.**

**Answer:**

### 1. Definition
A Feature Store is a centralized repository for storing, managing, and serving features for ML models. It enables feature reuse across teams and ensures consistency between training and serving.

### 2. Python Implementation

```python
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Callable, Any
import hashlib
import json

class FeatureTransform:
    """Reusable feature transformation"""
    
    def __init__(self, name: str, transform_fn: Callable, version: str = "1.0"):
        self.name = name
        self.transform_fn = transform_fn
        self.version = version
        self.created_at = datetime.now().isoformat()
    
    def apply(self, data):
        return self.transform_fn(data)
    
    def get_signature(self):
        return f"{self.name}_v{self.version}"

class FeatureGroup:
    """Collection of related features"""
    
    def __init__(self, name: str, entity_key: str):
        self.name = name
        self.entity_key = entity_key
        self.features = {}
        self.transforms = {}
    
    def add_feature(self, feature_name: str, transform: FeatureTransform):
        self.features[feature_name] = transform.get_signature()
        self.transforms[feature_name] = transform
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df[[self.entity_key]].copy()
        for name, transform in self.transforms.items():
            result[name] = transform.apply(df)
        return result

class FeatureStore:
    """Simulated Feature Store"""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.feature_groups = {}
        self.feature_data = {}
        self.metadata = {}
    
    def register_feature_group(self, group: FeatureGroup):
        self.feature_groups[group.name] = group
        print(f"Registered feature group: {group.name}")
    
    def materialize(self, group_name: str, df: pd.DataFrame):
        """Compute and store features"""
        group = self.feature_groups[group_name]
        features = group.compute(df)
        
        key = f"{group_name}_{datetime.now().strftime('%Y%m%d')}"
        self.feature_data[key] = features
        self.metadata[key] = {
            'group': group_name,
            'rows': len(features),
            'timestamp': datetime.now().isoformat()
        }
        print(f"Materialized {len(features)} rows for {group_name}")
        return features
    
    def get_features(self, group_name: str, entity_ids: List) -> pd.DataFrame:
        """Retrieve features for entities"""
        keys = [k for k in self.feature_data if k.startswith(group_name)]
        if not keys:
            raise ValueError(f"No data for {group_name}")
        
        latest = sorted(keys)[-1]
        df = self.feature_data[latest]
        
        entity_key = self.feature_groups[group_name].entity_key
        return df[df[entity_key].isin(entity_ids)]
    
    def get_training_data(self, feature_groups: List[str], 
                          entity_ids: List) -> pd.DataFrame:
        """Get combined features for training"""
        dfs = []
        for group_name in feature_groups:
            df = self.get_features(group_name, entity_ids)
            dfs.append(df)
        
        if len(dfs) == 1:
            return dfs[0]
        
        result = dfs[0]
        for df in dfs[1:]:
            entity_key = list(df.columns)[0]
            result = result.merge(df, on=entity_key, how='outer')
        return result

# Demo
print("=== Feature Store Demo ===\n")

# Create transforms
age_normalize = FeatureTransform(
    "age_normalize",
    lambda df: (df['age'] - df['age'].mean()) / df['age'].std()
)

income_bucket = FeatureTransform(
    "income_bucket",
    lambda df: pd.cut(df['income'], bins=[0, 30000, 70000, 150000, float('inf')],
                     labels=['low', 'medium', 'high', 'very_high'])
)

# Create feature group
user_features = FeatureGroup("user_features", "user_id")
user_features.add_feature("age_normalized", age_normalize)
user_features.add_feature("income_level", income_bucket)

# Initialize store
store = FeatureStore("ml_store")
store.register_feature_group(user_features)

# Sample data
df = pd.DataFrame({
    'user_id': range(100),
    'age': np.random.randint(18, 70, 100),
    'income': np.random.randint(20000, 200000, 100)
})

# Materialize features
features = store.materialize("user_features", df)
print(features.head())

# Retrieve for training
training_data = store.get_training_data(["user_features"], [1, 2, 3, 4, 5])
print(f"\nTraining data:\n{training_data}")
```

### 3. Interview Tips
- Feature stores ensure training-serving consistency
- Version features for reproducibility
- Mention Feast, Tecton as production tools

---

## Question 6
- [ ] Done

**Implement a 'Warm Start' to accelerate training for a new model using an established pretrained model.**

**Answer:**

### 1. Definition
Warm Start initializes a new model with weights from a previously trained model to accelerate training and improve convergence. Useful when training data changes or when fine-tuning for new tasks.

### 2. Python Implementation

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

class WarmStartTrainer:
    """Warm start training for iterative models"""
    
    def __init__(self, base_model=None):
        self.base_model = base_model
        self.history = []
    
    def train_cold(self, X, y, model_class, **params):
        """Train from scratch (cold start)"""
        start = time.time()
        model = model_class(**params)
        model.fit(X, y)
        train_time = time.time() - start
        
        self.history.append({
            'type': 'cold', 'time': train_time,
            'accuracy': model.score(X, y)
        })
        return model
    
    def train_warm(self, X, y, pretrained_model, **additional_params):
        """Warm start from pretrained model"""
        start = time.time()
        
        # For sklearn models with warm_start
        if hasattr(pretrained_model, 'warm_start'):
            pretrained_model.warm_start = True
            pretrained_model.n_estimators += additional_params.get('additional_estimators', 50)
            pretrained_model.fit(X, y)
            model = pretrained_model
        else:
            # Copy weights and continue training
            model = self._copy_and_train(pretrained_model, X, y)
        
        train_time = time.time() - start
        self.history.append({
            'type': 'warm', 'time': train_time,
            'accuracy': model.score(X, y)
        })
        return model
    
    def _copy_and_train(self, base_model, X, y):
        """Copy model and continue training"""
        import copy
        model = copy.deepcopy(base_model)
        if hasattr(model, 'partial_fit'):
            model.partial_fit(X, y)
        else:
            model.fit(X, y)
        return model

# Keras/TensorFlow warm start
def keras_warm_start():
    try:
        from tensorflow import keras
        
        # Create base model
        base = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(20,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        base.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train base model
        X = np.random.randn(1000, 20)
        y = (X.sum(axis=1) > 0).astype(int)
        base.fit(X, y, epochs=5, verbose=0)
        base_acc = base.evaluate(X, y, verbose=0)[1]
        print(f"Base model accuracy: {base_acc:.4f}")
        
        # Warm start: load weights into new model
        new_model = keras.models.clone_model(base)
        new_model.set_weights(base.get_weights())
        new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Fine-tune on new data
        X_new = np.random.randn(200, 20)
        y_new = (X_new.sum(axis=1) > 0).astype(int)
        new_model.fit(X_new, y_new, epochs=3, verbose=0)
        warm_acc = new_model.evaluate(X_new, y_new, verbose=0)[1]
        print(f"Warm start accuracy: {warm_acc:.4f}")
        
    except ImportError:
        print("TensorFlow not available")

# Demo
print("=== Warm Start Pattern Demo ===\n")

# Generate data
np.random.seed(42)
X, y = np.random.randn(1000, 10), (np.random.randn(1000) > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

trainer = WarmStartTrainer()

# Cold start
print("Cold Start Training:")
cold_model = trainer.train_cold(X_train, y_train, 
    RandomForestClassifier, n_estimators=50, random_state=42)
print(f"  Time: {trainer.history[-1]['time']:.3f}s")
print(f"  Accuracy: {cold_model.score(X_test, y_test):.4f}")

# Warm start (add more trees)
print("\nWarm Start Training:")
warm_model = trainer.train_warm(X_train, y_train, cold_model, additional_estimators=50)
print(f"  Time: {trainer.history[-1]['time']:.3f}s")
print(f"  Accuracy: {warm_model.score(X_test, y_test):.4f}")

print("\nKeras Example:")
keras_warm_start()
```

### 3. Interview Tips
- Warm start significantly reduces training time
- Essential for transfer learning scenarios
- Use for incremental learning with new data

---

## Question 7
- [ ] Done

**Simulate a horizontal scaling of a machine learning pipeline handling increasing workloads.**

**Answer:**

### 1. Definition
Horizontal Scaling adds more worker nodes to handle increased load, distributing predictions across multiple instances. Essential for high-throughput production systems.

### 2. Python Implementation

```python
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue
import threading
from typing import List, Callable
from sklearn.ensemble import RandomForestClassifier

class ModelWorker:
    """Individual prediction worker"""
    
    def __init__(self, worker_id: int, model):
        self.worker_id = worker_id
        self.model = model
        self.requests_processed = 0
    
    def predict(self, X):
        self.requests_processed += 1
        return self.model.predict(X)

class LoadBalancer:
    """Round-robin load balancer"""
    
    def __init__(self, workers: List[ModelWorker]):
        self.workers = workers
        self.current_idx = 0
        self.lock = threading.Lock()
    
    def get_worker(self) -> ModelWorker:
        with self.lock:
            worker = self.workers[self.current_idx]
            self.current_idx = (self.current_idx + 1) % len(self.workers)
            return worker

class HorizontalScaler:
    """Horizontally scaled prediction service"""
    
    def __init__(self, model, initial_workers: int = 2):
        self.base_model = model
        self.workers = []
        self.load_balancer = None
        self.executor = None
        self.metrics = {'total_requests': 0, 'total_time': 0}
        
        for i in range(initial_workers):
            self.add_worker()
    
    def add_worker(self):
        """Scale out: add a new worker"""
        worker_id = len(self.workers)
        worker = ModelWorker(worker_id, self.base_model)
        self.workers.append(worker)
        self.load_balancer = LoadBalancer(self.workers)
        print(f"Added worker {worker_id}. Total workers: {len(self.workers)}")
    
    def remove_worker(self):
        """Scale in: remove a worker"""
        if len(self.workers) > 1:
            removed = self.workers.pop()
            self.load_balancer = LoadBalancer(self.workers)
            print(f"Removed worker. Total workers: {len(self.workers)}")
    
    def predict(self, X):
        """Single prediction via load balancer"""
        worker = self.load_balancer.get_worker()
        return worker.predict(X)
    
    def predict_batch_parallel(self, X_batch: List[np.ndarray]) -> List:
        """Parallel batch prediction"""
        start = time.time()
        
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = [executor.submit(self.predict, X) for X in X_batch]
            results = [f.result() for f in futures]
        
        self.metrics['total_requests'] += len(X_batch)
        self.metrics['total_time'] += time.time() - start
        
        return results
    
    def get_metrics(self):
        throughput = self.metrics['total_requests'] / max(self.metrics['total_time'], 0.01)
        worker_stats = [(w.worker_id, w.requests_processed) for w in self.workers]
        return {
            'workers': len(self.workers),
            'total_requests': self.metrics['total_requests'],
            'throughput': throughput,
            'worker_distribution': worker_stats
        }
    
    def auto_scale(self, current_load: float, threshold_high: float = 0.8,
                   threshold_low: float = 0.3):
        """Auto-scale based on load"""
        if current_load > threshold_high:
            self.add_worker()
            return 'scaled_out'
        elif current_load < threshold_low and len(self.workers) > 1:
            self.remove_worker()
            return 'scaled_in'
        return 'no_change'

# Demo
print("=== Horizontal Scaling Demo ===\n")

# Train model
np.random.seed(42)
X_train = np.random.randn(500, 10)
y_train = (X_train.sum(axis=1) > 0).astype(int)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Create scaler with 2 workers
scaler = HorizontalScaler(model, initial_workers=2)

# Simulate increasing workload
print("Simulating workload...")
test_batches = [np.random.randn(1, 10) for _ in range(100)]

# Phase 1: Low load
print("\n--- Phase 1: Low Load ---")
_ = scaler.predict_batch_parallel(test_batches[:20])
print(scaler.get_metrics())

# Phase 2: Scale out
print("\n--- Phase 2: Scale Out ---")
scaler.auto_scale(current_load=0.9)  # High load
_ = scaler.predict_batch_parallel(test_batches[20:60])
print(scaler.get_metrics())

# Phase 3: More scaling
print("\n--- Phase 3: More Scaling ---")
scaler.add_worker()
_ = scaler.predict_batch_parallel(test_batches[60:])
print(scaler.get_metrics())
```

### 3. Interview Tips
- Load balancing ensures even distribution
- Auto-scaling based on metrics (CPU, latency, queue depth)
- Stateless models enable easy horizontal scaling

---

## Question 8
- [ ] Done

**Write a Python function that demonstrates the 'Handling Missing Data' pattern.**

**Answer:**

### 1. Definition
Handling Missing Data pattern provides strategies to deal with incomplete data during training and inference. Proper handling prevents model errors and ensures consistent behavior.

### 2. Strategies

| Strategy | When to Use | Impact |
|----------|-------------|--------|
| **Deletion** | <5% missing, MCAR | Loses data |
| **Mean/Median** | Numerical, low missing | Simple but biased |
| **Mode** | Categorical | May introduce bias |
| **Indicator** | Missingness is signal | Adds features |
| **Model-based** | Complex patterns | Best accuracy |
| **Forward Fill** | Time series | Preserves trends |

### 3. Python Implementation

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from typing import Dict, List, Optional

class MissingDataHandler:
    """Comprehensive missing data handling"""
    
    def __init__(self, strategy: str = 'auto'):
        self.strategy = strategy
        self.imputers = {}
        self.statistics = {}
        self.missing_indicators = []
    
    def fit(self, df: pd.DataFrame, numerical_cols: List[str], 
            categorical_cols: List[str]):
        """Fit imputers based on training data"""
        
        # Record missing patterns
        self.statistics['missing_before'] = df.isnull().sum().to_dict()
        
        # Numerical columns
        for col in numerical_cols:
            if df[col].isnull().any():
                if self.strategy == 'auto':
                    # Use median for skewed, mean for normal
                    skewness = abs(df[col].skew())
                    strategy = 'median' if skewness > 1 else 'mean'
                else:
                    strategy = self.strategy
                
                imputer = SimpleImputer(strategy=strategy)
                imputer.fit(df[[col]])
                self.imputers[col] = imputer
                self.statistics[col] = {
                    'strategy': strategy,
                    'value': imputer.statistics_[0]
                }
        
        # Categorical columns
        for col in categorical_cols:
            if df[col].isnull().any():
                imputer = SimpleImputer(strategy='most_frequent')
                imputer.fit(df[[col]].astype(str))
                self.imputers[col] = imputer
                self.statistics[col] = {
                    'strategy': 'most_frequent',
                    'value': imputer.statistics_[0]
                }
        
        return self
    
    def transform(self, df: pd.DataFrame, add_indicators: bool = True) -> pd.DataFrame:
        """Apply imputation and optionally add missing indicators"""
        result = df.copy()
        
        for col, imputer in self.imputers.items():
            if col in result.columns:
                # Add indicator before imputation
                if add_indicators and result[col].isnull().any():
                    indicator_col = f'{col}_was_missing'
                    result[indicator_col] = result[col].isnull().astype(int)
                    self.missing_indicators.append(indicator_col)
                
                # Impute
                if result[col].dtype == 'object':
                    result[col] = imputer.transform(result[[col]].astype(str))
                else:
                    result[col] = imputer.transform(result[[col]])
        
        return result
    
    def fit_transform(self, df: pd.DataFrame, numerical_cols: List[str],
                      categorical_cols: List[str], add_indicators: bool = True):
        """Fit and transform in one step"""
        self.fit(df, numerical_cols, categorical_cols)
        return self.transform(df, add_indicators)
    
    def get_report(self) -> Dict:
        """Get imputation report"""
        return {
            'missing_before': self.statistics.get('missing_before', {}),
            'imputation_values': {k: v for k, v in self.statistics.items() 
                                  if k != 'missing_before'},
            'indicators_added': self.missing_indicators
        }

class AdvancedImputer:
    """Advanced imputation methods"""
    
    @staticmethod
    def knn_impute(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
        """KNN-based imputation"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        return df
    
    @staticmethod
    def iterative_impute(df: pd.DataFrame, max_iter: int = 10) -> pd.DataFrame:
        """MICE-style iterative imputation"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        imputer = IterativeImputer(max_iter=max_iter, random_state=42)
        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        return df
    
    @staticmethod
    def forward_fill_timeseries(df: pd.DataFrame, 
                                 time_col: str,
                                 group_col: Optional[str] = None) -> pd.DataFrame:
        """Forward fill for time series"""
        df = df.sort_values(time_col)
        if group_col:
            df = df.groupby(group_col).ffill()
        else:
            df = df.ffill()
        return df

def validate_no_missing(df: pd.DataFrame) -> bool:
    """Validate that data has no missing values"""
    missing = df.isnull().sum().sum()
    if missing > 0:
        raise ValueError(f"Data still has {missing} missing values")
    return True

# Demo
print("=== Handling Missing Data Pattern Demo ===\n")

# Create data with missing values
np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.randint(20, 60, 100).astype(float),
    'income': np.random.randint(30000, 150000, 100).astype(float),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'score': np.random.randn(100)
})

# Introduce missing values
df.loc[np.random.choice(100, 15, replace=False), 'age'] = np.nan
df.loc[np.random.choice(100, 10, replace=False), 'income'] = np.nan
df.loc[np.random.choice(100, 8, replace=False), 'category'] = np.nan

print("Missing values before:")
print(df.isnull().sum())

# Apply handling
handler = MissingDataHandler(strategy='auto')
df_clean = handler.fit_transform(
    df,
    numerical_cols=['age', 'income', 'score'],
    categorical_cols=['category']
)

print("\nMissing values after:")
print(df_clean.isnull().sum())

print("\nImputation Report:")
report = handler.get_report()
for col, info in report['imputation_values'].items():
    if isinstance(info, dict):
        print(f"  {col}: {info['strategy']} = {info['value']:.2f}")

print(f"\nIndicator columns added: {report['indicators_added']}")

# Validate
validate_no_missing(df_clean)
print("\nValidation passed: No missing values")
```

### 4. Interview Tips
- Always analyze missing patterns (MCAR, MAR, MNAR)
- Missing indicators can be valuable features
- KNN/Iterative imputers for complex dependencies
- Consistent handling between training and serving

---

## Question 9
- [ ] Done

**Code a simplified 'Hyperparameter Database' to track experiments in machine learning.**

**Answer:**

### 1. Definition
A Hyperparameter Database stores experiment configurations, metrics, and artifacts to enable comparison, reproducibility, and optimization of ML experiments.

### 2. Python Implementation

```python
import numpy as np
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib

class HyperparameterDatabase:
    """SQLite-backed experiment tracking database"""
    
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT,
                model_type TEXT,
                hyperparameters TEXT,
                metrics TEXT,
                status TEXT,
                created_at TEXT,
                duration_seconds REAL,
                git_commit TEXT,
                notes TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                epoch INTEGER,
                metric_name TEXT,
                metric_value REAL,
                timestamp TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_experiment(self, name: str, model_type: str,
                          hyperparameters: Dict, notes: str = "") -> str:
        """Create new experiment"""
        exp_id = hashlib.md5(
            f"{name}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO experiments 
            (id, name, model_type, hyperparameters, status, created_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (exp_id, name, model_type, json.dumps(hyperparameters),
              'running', datetime.now().isoformat(), notes))
        
        conn.commit()
        conn.close()
        
        print(f"Created experiment: {exp_id}")
        return exp_id
    
    def log_metric(self, experiment_id: str, metric_name: str,
                   metric_value: float, epoch: int = 0):
        """Log a metric value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO metrics_history 
            (experiment_id, epoch, metric_name, metric_value, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (experiment_id, epoch, metric_name, metric_value,
              datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def complete_experiment(self, experiment_id: str, 
                            final_metrics: Dict, duration: float):
        """Mark experiment as completed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE experiments 
            SET status = ?, metrics = ?, duration_seconds = ?
            WHERE id = ?
        ''', ('completed', json.dumps(final_metrics), duration, experiment_id))
        
        conn.commit()
        conn.close()
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Retrieve experiment details"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM experiments WHERE id = ?', (experiment_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0], 'name': row[1], 'model_type': row[2],
                'hyperparameters': json.loads(row[3]),
                'metrics': json.loads(row[4]) if row[4] else {},
                'status': row[5], 'created_at': row[6],
                'duration': row[7], 'notes': row[9]
            }
        return None
    
    def get_best_experiment(self, metric_name: str = 'accuracy',
                            maximize: bool = True) -> Optional[Dict]:
        """Get best experiment by metric"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM experiments WHERE status = ?', ('completed',))
        rows = cursor.fetchall()
        conn.close()
        
        best = None
        best_value = float('-inf') if maximize else float('inf')
        
        for row in rows:
            metrics = json.loads(row[4]) if row[4] else {}
            value = metrics.get(metric_name)
            if value is not None:
                if (maximize and value > best_value) or \
                   (not maximize and value < best_value):
                    best_value = value
                    best = {
                        'id': row[0], 'name': row[1],
                        'hyperparameters': json.loads(row[3]),
                        'metrics': metrics
                    }
        
        return best
    
    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiments"""
        import pandas as pd
        
        experiments = []
        for exp_id in experiment_ids:
            exp = self.get_experiment(exp_id)
            if exp:
                flat = {'id': exp['id'], 'name': exp['name']}
                flat.update(exp['hyperparameters'])
                flat.update(exp['metrics'])
                experiments.append(flat)
        
        return pd.DataFrame(experiments)
    
    def search_experiments(self, **filters) -> List[Dict]:
        """Search experiments by hyperparameters"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM experiments')
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            params = json.loads(row[3])
            match = all(params.get(k) == v for k, v in filters.items())
            if match:
                results.append({
                    'id': row[0], 'name': row[1],
                    'hyperparameters': params,
                    'metrics': json.loads(row[4]) if row[4] else {}
                })
        
        return results

# Demo
import pandas as pd
print("=== Hyperparameter Database Demo ===\n")

db = HyperparameterDatabase("demo_experiments.db")

# Run experiments
experiments = []
for lr in [0.001, 0.01, 0.1]:
    for depth in [3, 5, 10]:
        exp_id = db.create_experiment(
            name="gradient_boosting_tuning",
            model_type="GradientBoosting",
            hyperparameters={'learning_rate': lr, 'max_depth': depth}
        )
        
        # Simulate training
        import time
        start = time.time()
        accuracy = 0.8 + np.random.random() * 0.15
        f1_score = 0.75 + np.random.random() * 0.2
        
        db.log_metric(exp_id, 'accuracy', accuracy, epoch=1)
        db.log_metric(exp_id, 'f1_score', f1_score, epoch=1)
        
        db.complete_experiment(exp_id, 
            {'accuracy': accuracy, 'f1_score': f1_score},
            time.time() - start)
        
        experiments.append(exp_id)

# Find best
print("\nBest Experiment:")
best = db.get_best_experiment('accuracy', maximize=True)
print(f"  ID: {best['id']}")
print(f"  Params: {best['hyperparameters']}")
print(f"  Accuracy: {best['metrics']['accuracy']:.4f}")

# Compare
print("\nExperiment Comparison:")
comparison = db.compare_experiments(experiments[:4])
print(comparison[['id', 'learning_rate', 'max_depth', 'accuracy']].to_string())

# Cleanup
import os
os.remove("demo_experiments.db")
```

### 3. Interview Tips
- Mention MLflow, Weights & Biases, Neptune as production tools
- Track not just hyperparameters but environment, data version
- Enable search and comparison across experiments
- Store artifacts (models, plots) alongside metadata

---

## Question 10
- [ ] Done

**Develop a basic implementation of 'Model Factories' using Python for creating and deploying models.**

**Answer:**

### 1. Definition
Model Factory pattern provides a standardized interface for creating, configuring, and deploying different model types. It abstracts model creation logic and enables consistent handling across various algorithms.

### 2. Python Implementation

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, Optional, Type
from abc import ABC, abstractmethod
import pickle
import os

class ModelWrapper(ABC):
    """Abstract base for model wrappers"""
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def get_params(self) -> Dict:
        pass

class SklearnModelWrapper(ModelWrapper):
    """Wrapper for sklearn models"""
    
    def __init__(self, model, name: str):
        self.model = model
        self.name = name
        self.is_trained = False
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
    
    def get_params(self) -> Dict:
        return self.model.get_params()
    
    def score(self, X, y):
        return self.model.score(X, y)

class ModelFactory:
    """Factory for creating models"""
    
    # Registry of available model types
    _registry: Dict[str, Type] = {
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'logistic_regression': LogisticRegression,
        'svm': SVC,
        'mlp': MLPClassifier
    }
    
    # Default configurations
    _defaults: Dict[str, Dict] = {
        'random_forest': {'n_estimators': 100, 'random_state': 42},
        'gradient_boosting': {'n_estimators': 100, 'random_state': 42},
        'logistic_regression': {'max_iter': 1000, 'random_state': 42},
        'svm': {'probability': True, 'random_state': 42},
        'mlp': {'hidden_layer_sizes': (64, 32), 'max_iter': 500, 'random_state': 42}
    }
    
    @classmethod
    def register_model(cls, name: str, model_class: Type, defaults: Dict = None):
        """Register a new model type"""
        cls._registry[name] = model_class
        if defaults:
            cls._defaults[name] = defaults
    
    @classmethod
    def create(cls, model_type: str, **kwargs) -> SklearnModelWrapper:
        """Create a model instance"""
        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(cls._registry.keys())}")
        
        # Merge defaults with provided kwargs
        config = cls._defaults.get(model_type, {}).copy()
        config.update(kwargs)
        
        model_class = cls._registry[model_type]
        model = model_class(**config)
        
        return SklearnModelWrapper(model, model_type)
    
    @classmethod
    def create_from_config(cls, config: Dict) -> SklearnModelWrapper:
        """Create model from configuration dict"""
        model_type = config.pop('model_type')
        return cls.create(model_type, **config)
    
    @classmethod
    def list_available(cls) -> list:
        """List available model types"""
        return list(cls._registry.keys())

class ModelDeployer:
    """Handles model deployment"""
    
    def __init__(self, models_dir: str = "deployed_models"):
        self.models_dir = models_dir
        self.deployed_models = {}
        os.makedirs(models_dir, exist_ok=True)
    
    def deploy(self, model: SklearnModelWrapper, version: str) -> str:
        """Deploy a model"""
        if not model.is_trained:
            raise ValueError("Cannot deploy untrained model")
        
        deployment_id = f"{model.name}_v{version}"
        model_path = os.path.join(self.models_dir, f"{deployment_id}.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        self.deployed_models[deployment_id] = {
            'path': model_path,
            'params': model.get_params(),
            'name': model.name
        }
        
        print(f"Deployed: {deployment_id}")
        return deployment_id
    
    def load(self, deployment_id: str) -> SklearnModelWrapper:
        """Load a deployed model"""
        if deployment_id not in self.deployed_models:
            raise ValueError(f"Model {deployment_id} not found")
        
        path = self.deployed_models[deployment_id]['path']
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def list_deployed(self) -> list:
        """List deployed models"""
        return list(self.deployed_models.keys())

class ModelPipeline:
    """Pipeline for training and deploying models"""
    
    def __init__(self, deployer: ModelDeployer):
        self.deployer = deployer
    
    def train_and_deploy(self, model_type: str, X, y, 
                         version: str, **model_kwargs) -> str:
        """Train and deploy in one step"""
        # Create model
        model = ModelFactory.create(model_type, **model_kwargs)
        
        # Evaluate
        scores = cross_val_score(model.model, X, y, cv=3)
        print(f"CV Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        # Train on full data
        model.fit(X, y)
        
        # Deploy
        deployment_id = self.deployer.deploy(model, version)
        
        return deployment_id
    
    def compare_models(self, X, y, model_types: list = None) -> Dict:
        """Compare multiple model types"""
        model_types = model_types or ModelFactory.list_available()
        results = {}
        
        for model_type in model_types:
            try:
                model = ModelFactory.create(model_type)
                scores = cross_val_score(model.model, X, y, cv=3)
                results[model_type] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std()
                }
            except Exception as e:
                results[model_type] = {'error': str(e)}
        
        return results

# Demo
print("=== Model Factory Pattern Demo ===\n")

# Generate data
np.random.seed(42)
X = np.random.randn(500, 10)
y = (X.sum(axis=1) > 0).astype(int)

# List available models
print("Available model types:")
for model_type in ModelFactory.list_available():
    print(f"  - {model_type}")

# Create and train models
print("\nCreating models:")
rf_model = ModelFactory.create('random_forest', n_estimators=50)
print(f"Created: {rf_model.name} with params: {rf_model.get_params()}")

gb_model = ModelFactory.create('gradient_boosting', learning_rate=0.1)
print(f"Created: {gb_model.name}")

# Train
rf_model.fit(X, y)
gb_model.fit(X, y)

# Deploy
deployer = ModelDeployer()
deployer.deploy(rf_model, "1.0")
deployer.deploy(gb_model, "1.0")

print(f"\nDeployed models: {deployer.list_deployed()}")

# Compare all models
pipeline = ModelPipeline(deployer)
print("\nModel Comparison:")
comparison = pipeline.compare_models(X, y)
for model, scores in comparison.items():
    if 'error' not in scores:
        print(f"  {model}: {scores['mean_score']:.4f}")

# Cleanup
import shutil
shutil.rmtree("deployed_models")
```

### 3. Interview Tips
- Factory pattern abstracts model creation complexity
- Registry pattern for extensibility
- Consistent interface across model types
- Combines well with Deployment and Versioning patterns

---

## Question 11
- [ ] Done

**Simulate a 'Data Validator' using Python to check for data skew or anomalies as new data arrives.**

**Answer:**

### 1. Definition
Data Validator pattern checks incoming data for quality issues, schema violations, distribution drift, and anomalies before it's used for training or inference. Prevents garbage-in-garbage-out problems.

### 2. Validation Types

| Type | What It Checks | Example |
|------|---------------|---------|
| **Schema** | Data types, required fields | Missing columns |
| **Range** | Value bounds | Age > 200 |
| **Distribution** | Statistical drift | Mean shift |
| **Anomaly** | Outliers | Z-score > 3 |
| **Referential** | Foreign key validity | Unknown category |

### 3. Python Implementation

```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ValidationLevel(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationResult:
    passed: bool
    level: ValidationLevel
    check_name: str
    message: str
    details: Dict = None

class SchemaValidator:
    """Validate data schema"""
    
    def __init__(self, expected_schema: Dict):
        self.schema = expected_schema
    
    def validate(self, df: pd.DataFrame) -> List[ValidationResult]:
        results = []
        
        # Check required columns
        for col, spec in self.schema.items():
            if spec.get('required', False) and col not in df.columns:
                results.append(ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    check_name="missing_column",
                    message=f"Required column '{col}' is missing"
                ))
            
            if col in df.columns:
                # Type check
                expected_type = spec.get('dtype')
                if expected_type and not np.issubdtype(df[col].dtype, expected_type):
                    results.append(ValidationResult(
                        passed=False,
                        level=ValidationLevel.ERROR,
                        check_name="type_mismatch",
                        message=f"Column '{col}' has wrong type"
                    ))
        
        # Check for unexpected columns
        expected_cols = set(self.schema.keys())
        actual_cols = set(df.columns)
        extra_cols = actual_cols - expected_cols
        if extra_cols:
            results.append(ValidationResult(
                passed=True,
                level=ValidationLevel.WARNING,
                check_name="extra_columns",
                message=f"Unexpected columns: {extra_cols}"
            ))
        
        return results

class RangeValidator:
    """Validate value ranges"""
    
    def __init__(self, ranges: Dict):
        self.ranges = ranges
    
    def validate(self, df: pd.DataFrame) -> List[ValidationResult]:
        results = []
        
        for col, bounds in self.ranges.items():
            if col not in df.columns:
                continue
            
            min_val = bounds.get('min')
            max_val = bounds.get('max')
            
            violations = 0
            if min_val is not None:
                violations += (df[col] < min_val).sum()
            if max_val is not None:
                violations += (df[col] > max_val).sum()
            
            if violations > 0:
                pct = violations / len(df) * 100
                results.append(ValidationResult(
                    passed=pct < 1,  # Allow < 1% violations
                    level=ValidationLevel.WARNING if pct < 5 else ValidationLevel.ERROR,
                    check_name="range_violation",
                    message=f"Column '{col}': {violations} values ({pct:.1f}%) out of range",
                    details={'column': col, 'violations': violations}
                ))
        
        return results

class DistributionValidator:
    """Detect distribution drift"""
    
    def __init__(self, baseline_stats: Dict):
        self.baseline = baseline_stats
    
    @staticmethod
    def compute_stats(df: pd.DataFrame, numerical_cols: List[str]) -> Dict:
        """Compute baseline statistics"""
        stats_dict = {}
        for col in numerical_cols:
            if col in df.columns:
                stats_dict[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'median': df[col].median(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q1': df[col].quantile(0.25),
                    'q3': df[col].quantile(0.75)
                }
        return stats_dict
    
    def validate(self, df: pd.DataFrame) -> List[ValidationResult]:
        results = []
        
        for col, baseline in self.baseline.items():
            if col not in df.columns:
                continue
            
            current_mean = df[col].mean()
            current_std = df[col].std()
            
            # Check mean drift
            baseline_mean = baseline['mean']
            baseline_std = baseline['std']
            
            if baseline_std > 0:
                z_score = abs(current_mean - baseline_mean) / baseline_std
                
                if z_score > 3:
                    results.append(ValidationResult(
                        passed=False,
                        level=ValidationLevel.ERROR,
                        check_name="distribution_drift",
                        message=f"Column '{col}': significant mean drift (z={z_score:.2f})",
                        details={'baseline_mean': baseline_mean, 'current_mean': current_mean}
                    ))
                elif z_score > 2:
                    results.append(ValidationResult(
                        passed=True,
                        level=ValidationLevel.WARNING,
                        check_name="distribution_drift",
                        message=f"Column '{col}': moderate mean drift (z={z_score:.2f})"
                    ))
        
        return results

class AnomalyDetector:
    """Detect anomalies in data"""
    
    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
    
    def validate(self, df: pd.DataFrame, columns: List[str]) -> List[ValidationResult]:
        results = []
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Z-score method
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            z_outliers = (z_scores > self.z_threshold).sum()
            
            # IQR method
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            iqr_outliers = ((df[col] < q1 - self.iqr_multiplier * iqr) | 
                           (df[col] > q3 + self.iqr_multiplier * iqr)).sum()
            
            if z_outliers > 0 or iqr_outliers > 0:
                results.append(ValidationResult(
                    passed=z_outliers < len(df) * 0.01,
                    level=ValidationLevel.WARNING,
                    check_name="anomaly_detected",
                    message=f"Column '{col}': {z_outliers} z-score outliers, {iqr_outliers} IQR outliers",
                    details={'z_outliers': int(z_outliers), 'iqr_outliers': int(iqr_outliers)}
                ))
        
        return results

class DataValidator:
    """Comprehensive data validation"""
    
    def __init__(self, schema: Dict, ranges: Dict, baseline_stats: Dict):
        self.schema_validator = SchemaValidator(schema)
        self.range_validator = RangeValidator(ranges)
        self.distribution_validator = DistributionValidator(baseline_stats)
        self.anomaly_detector = AnomalyDetector()
    
    def validate(self, df: pd.DataFrame, numerical_cols: List[str] = None) -> Dict:
        """Run all validations"""
        results = {
            'schema': self.schema_validator.validate(df),
            'range': self.range_validator.validate(df),
            'distribution': self.distribution_validator.validate(df),
            'anomaly': self.anomaly_detector.validate(df, numerical_cols or [])
        }
        
        # Summary
        all_results = [r for group in results.values() for r in group]
        errors = [r for r in all_results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in all_results if r.level == ValidationLevel.WARNING]
        
        return {
            'passed': len(errors) == 0,
            'error_count': len(errors),
            'warning_count': len(warnings),
            'results': results
        }
    
    def print_report(self, validation_result: Dict):
        """Print validation report"""
        print(f"\n{'='*50}")
        print(f"VALIDATION REPORT - {'PASSED' if validation_result['passed'] else 'FAILED'}")
        print(f"{'='*50}")
        print(f"Errors: {validation_result['error_count']}")
        print(f"Warnings: {validation_result['warning_count']}")
        
        for category, results in validation_result['results'].items():
            if results:
                print(f"\n{category.upper()}:")
                for r in results:
                    icon = '❌' if r.level == ValidationLevel.ERROR else '⚠️'
                    print(f"  {icon} {r.message}")

# Demo
print("=== Data Validator Pattern Demo ===\n")

# Define schema and constraints
schema = {
    'user_id': {'required': True, 'dtype': np.integer},
    'age': {'required': True, 'dtype': np.floating},
    'income': {'required': True, 'dtype': np.floating}
}

ranges = {
    'age': {'min': 0, 'max': 120},
    'income': {'min': 0, 'max': 10000000}
}

# Create baseline from "training" data
np.random.seed(42)
baseline_df = pd.DataFrame({
    'user_id': range(1000),
    'age': np.random.normal(35, 10, 1000),
    'income': np.random.normal(50000, 20000, 1000)
})
baseline_stats = DistributionValidator.compute_stats(baseline_df, ['age', 'income'])

# Create validator
validator = DataValidator(schema, ranges, baseline_stats)

# Test with good data
print("Testing with GOOD data:")
good_df = pd.DataFrame({
    'user_id': range(100),
    'age': np.random.normal(35, 10, 100),
    'income': np.random.normal(50000, 20000, 100)
})
result = validator.validate(good_df, ['age', 'income'])
validator.print_report(result)

# Test with problematic data
print("\n\nTesting with PROBLEMATIC data:")
bad_df = pd.DataFrame({
    'user_id': range(100),
    'age': np.random.normal(55, 10, 100),  # Mean shifted
    'income': np.random.normal(50000, 20000, 100)
})
bad_df.loc[0:5, 'age'] = -10  # Invalid ages
bad_df.loc[10, 'income'] = 999999999  # Outlier

result = validator.validate(bad_df, ['age', 'income'])
validator.print_report(result)
```

### 4. Interview Tips
- Data validation is first line of defense
- Combine multiple validation strategies
- Set appropriate thresholds per use case
- Log and alert on validation failures

---

## Question 12
- [ ] Done

**Create a 'Feature Monitoring' tool using Python and a visualization library to track changes over time.**

**Answer:**

### 1. Definition
Feature Monitoring tracks the statistical properties of features over time to detect drift, data quality issues, and anomalies that could affect model performance.

### 2. Python Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque
from scipy import stats

class FeatureStatistics:
    """Compute and store feature statistics"""
    
    def __init__(self, feature_name: str):
        self.feature_name = feature_name
        self.history = []
    
    def compute(self, data: np.ndarray, timestamp: datetime = None) -> Dict:
        """Compute statistics for a data batch"""
        timestamp = timestamp or datetime.now()
        
        stats_dict = {
            'timestamp': timestamp,
            'count': len(data),
            'missing_rate': np.isnan(data).sum() / len(data),
            'mean': np.nanmean(data),
            'std': np.nanstd(data),
            'min': np.nanmin(data),
            'max': np.nanmax(data),
            'median': np.nanmedian(data),
            'q1': np.nanpercentile(data, 25),
            'q3': np.nanpercentile(data, 75),
            'skewness': stats.skew(data[~np.isnan(data)]),
            'kurtosis': stats.kurtosis(data[~np.isnan(data)])
        }
        
        self.history.append(stats_dict)
        return stats_dict
    
    def get_history_df(self) -> pd.DataFrame:
        """Get history as DataFrame"""
        return pd.DataFrame(self.history)

class DriftDetector:
    """Detect feature drift"""
    
    def __init__(self, baseline_stats: Dict, threshold: float = 0.05):
        self.baseline = baseline_stats
        self.threshold = threshold
    
    def detect_drift(self, current_data: np.ndarray) -> Dict:
        """Detect drift using multiple methods"""
        results = {}
        
        # KS test
        ks_stat, ks_pvalue = stats.ks_2samp(
            np.random.choice(current_data, min(1000, len(current_data))),
            np.random.normal(self.baseline['mean'], self.baseline['std'], 1000)
        )
        results['ks_test'] = {
            'statistic': ks_stat,
            'p_value': ks_pvalue,
            'drift_detected': ks_pvalue < self.threshold
        }
        
        # Population Stability Index (PSI)
        psi = self._compute_psi(current_data)
        results['psi'] = {
            'value': psi,
            'drift_detected': psi > 0.2  # 0.1-0.2 moderate, >0.2 significant
        }
        
        # Mean shift
        mean_shift = abs(np.mean(current_data) - self.baseline['mean'])
        normalized_shift = mean_shift / max(self.baseline['std'], 0.001)
        results['mean_shift'] = {
            'value': normalized_shift,
            'drift_detected': normalized_shift > 2
        }
        
        return results
    
    def _compute_psi(self, current_data: np.ndarray, bins: int = 10) -> float:
        """Compute Population Stability Index"""
        # Create bins based on baseline
        bin_edges = np.linspace(
            self.baseline['mean'] - 3 * self.baseline['std'],
            self.baseline['mean'] + 3 * self.baseline['std'],
            bins + 1
        )
        
        # Expected distribution (baseline)
        expected = np.ones(bins) / bins
        
        # Actual distribution
        actual, _ = np.histogram(current_data, bins=bin_edges)
        actual = actual / actual.sum() + 0.0001
        expected = expected + 0.0001
        
        psi = np.sum((actual - expected) * np.log(actual / expected))
        return psi

class FeatureMonitor:
    """Complete feature monitoring system"""
    
    def __init__(self, features: List[str]):
        self.features = features
        self.stats = {f: FeatureStatistics(f) for f in features}
        self.baseline = {}
        self.drift_detectors = {}
        self.alerts = []
    
    def set_baseline(self, df: pd.DataFrame):
        """Set baseline from training data"""
        for feature in self.features:
            if feature in df.columns:
                data = df[feature].values
                self.baseline[feature] = {
                    'mean': np.nanmean(data),
                    'std': np.nanstd(data),
                    'min': np.nanmin(data),
                    'max': np.nanmax(data)
                }
                self.drift_detectors[feature] = DriftDetector(self.baseline[feature])
    
    def log_batch(self, df: pd.DataFrame, timestamp: datetime = None):
        """Log a batch of data"""
        timestamp = timestamp or datetime.now()
        
        for feature in self.features:
            if feature in df.columns:
                data = df[feature].values
                
                # Compute stats
                self.stats[feature].compute(data, timestamp)
                
                # Check drift
                if feature in self.drift_detectors:
                    drift = self.drift_detectors[feature].detect_drift(data)
                    
                    if any(d['drift_detected'] for d in drift.values()):
                        self.alerts.append({
                            'feature': feature,
                            'timestamp': timestamp,
                            'drift_results': drift
                        })
    
    def plot_feature_history(self, feature: str, metrics: List[str] = None,
                            figsize: tuple = (12, 6)):
        """Plot feature statistics over time"""
        metrics = metrics or ['mean', 'std', 'missing_rate']
        history = self.stats[feature].get_history_df()
        
        if history.empty:
            print(f"No history for feature: {feature}")
            return
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            ax.plot(history['timestamp'], history[metric], marker='o', label=metric)
            
            # Add baseline if available
            if feature in self.baseline and metric in ['mean']:
                ax.axhline(self.baseline[feature][metric], color='r', 
                          linestyle='--', label=f'baseline {metric}')
            
            ax.set_ylabel(metric)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time')
        fig.suptitle(f'Feature Monitoring: {feature}')
        plt.tight_layout()
        plt.savefig(f'feature_monitoring_{feature}.png', dpi=100)
        plt.close()
        print(f"Saved plot: feature_monitoring_{feature}.png")
    
    def plot_drift_summary(self, figsize: tuple = (10, 6)):
        """Plot drift summary across all features"""
        drift_data = []
        
        for feature in self.features:
            history = self.stats[feature].get_history_df()
            if not history.empty and feature in self.baseline:
                latest = history.iloc[-1]
                baseline = self.baseline[feature]
                drift = abs(latest['mean'] - baseline['mean']) / max(baseline['std'], 0.001)
                drift_data.append({'feature': feature, 'drift_score': drift})
        
        if drift_data:
            df = pd.DataFrame(drift_data)
            plt.figure(figsize=figsize)
            bars = plt.bar(df['feature'], df['drift_score'])
            
            # Color based on severity
            for bar, score in zip(bars, df['drift_score']):
                if score > 2:
                    bar.set_color('red')
                elif score > 1:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
            
            plt.axhline(2, color='red', linestyle='--', label='Alert threshold')
            plt.axhline(1, color='orange', linestyle='--', label='Warning threshold')
            plt.xlabel('Feature')
            plt.ylabel('Drift Score (normalized)')
            plt.title('Feature Drift Summary')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig('drift_summary.png', dpi=100)
            plt.close()
            print("Saved plot: drift_summary.png")
    
    def get_alerts(self) -> List[Dict]:
        """Get all drift alerts"""
        return self.alerts

# Demo
print("=== Feature Monitoring Demo ===\n")

np.random.seed(42)

# Define features to monitor
features = ['age', 'income', 'score']
monitor = FeatureMonitor(features)

# Set baseline from "training" data
baseline_df = pd.DataFrame({
    'age': np.random.normal(35, 10, 1000),
    'income': np.random.normal(50000, 15000, 1000),
    'score': np.random.normal(0, 1, 1000)
})
monitor.set_baseline(baseline_df)
print("Baseline set from training data")

# Simulate data over time with gradual drift
print("\nSimulating data over time...")
base_time = datetime.now() - timedelta(days=30)

for day in range(30):
    timestamp = base_time + timedelta(days=day)
    
    # Introduce gradual drift in 'age' and sudden drift in 'income'
    age_shift = day * 0.3  # Gradual drift
    income_shift = 20000 if day > 20 else 0  # Sudden shift
    
    batch = pd.DataFrame({
        'age': np.random.normal(35 + age_shift, 10, 100),
        'income': np.random.normal(50000 + income_shift, 15000, 100),
        'score': np.random.normal(0, 1, 100)
    })
    
    monitor.log_batch(batch, timestamp)

# Generate visualizations
print("\nGenerating plots...")
monitor.plot_feature_history('age', ['mean', 'std'])
monitor.plot_feature_history('income', ['mean', 'std'])
monitor.plot_drift_summary()

# Show alerts
alerts = monitor.get_alerts()
print(f"\nDrift Alerts: {len(alerts)}")
for alert in alerts[-3:]:  # Show last 3
    print(f"  Feature: {alert['feature']}, Time: {alert['timestamp']}")
```

### 3. Interview Tips
- Monitor both input features and predictions
- Use multiple drift detection methods
- Set appropriate alerting thresholds
- Mention tools like Evidently, WhyLabs, Great Expectations

---

## Question 13
- [ ] Done

**Implement a 'Retry' pattern in a Python script for a machine learning service that handles intermittent failures.**

**Answer:**

### 1. Definition
The Retry pattern automatically retries failed operations with configurable backoff strategies. Essential for ML services that depend on external resources (APIs, databases, model servers) that may have transient failures.

### 2. Retry Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Fixed** | Same delay between retries | Simple cases |
| **Exponential** | Delay doubles each retry | API rate limiting |
| **Exponential + Jitter** | Random component added | Multiple clients |
| **Linear** | Delay increases linearly | Gradual recovery |

### 3. Python Implementation

```python
import time
import random
import functools
from typing import Callable, List, Type, Optional
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackoffStrategy(Enum):
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"

class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER,
                 retryable_exceptions: List[Type[Exception]] = None):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.strategy = strategy
        self.retryable_exceptions = retryable_exceptions or [Exception]
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number"""
        if self.strategy == BackoffStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == BackoffStrategy.LINEAR:
            delay = self.base_delay * attempt
        elif self.strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.base_delay * (2 ** (attempt - 1))
        elif self.strategy == BackoffStrategy.EXPONENTIAL_JITTER:
            delay = self.base_delay * (2 ** (attempt - 1))
            delay = delay * (0.5 + random.random())  # Add jitter
        else:
            delay = self.base_delay
        
        return min(delay, self.max_delay)

class RetryError(Exception):
    """Raised when all retries are exhausted"""
    
    def __init__(self, message: str, last_exception: Exception, attempts: int):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts

def retry(config: RetryConfig = None):
    """Decorator for retry behavior"""
    config = config or RetryConfig()
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, config.max_retries + 2):
                try:
                    return func(*args, **kwargs)
                    
                except tuple(config.retryable_exceptions) as e:
                    last_exception = e
                    
                    if attempt > config.max_retries:
                        raise RetryError(
                            f"Failed after {attempt} attempts: {str(e)}",
                            last_exception,
                            attempt
                        )
                    
                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt} failed: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
            
            raise RetryError(
                f"Failed after {config.max_retries + 1} attempts",
                last_exception,
                config.max_retries + 1
            )
        
        return wrapper
    return decorator

class CircuitBreaker:
    """Circuit breaker to prevent cascading failures"""
    
    def __init__(self, failure_threshold: int = 5, 
                 recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "half-open"
                logger.info("Circuit breaker: half-open, testing...")
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
                logger.info("Circuit breaker: closed")
            
            return result
            
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.failures >= self.failure_threshold:
                self.state = "open"
                logger.warning(f"Circuit breaker: open after {self.failures} failures")
            
            raise

class MLServiceClient:
    """ML service client with retry and circuit breaker"""
    
    def __init__(self, service_url: str):
        self.service_url = service_url
        self.circuit_breaker = CircuitBreaker(failure_threshold=3)
        self.retry_config = RetryConfig(
            max_retries=3,
            base_delay=0.5,
            strategy=BackoffStrategy.EXPONENTIAL_JITTER,
            retryable_exceptions=[ConnectionError, TimeoutError]
        )
    
    @retry(RetryConfig(max_retries=3, base_delay=0.5))
    def predict(self, features: list) -> dict:
        """Make prediction with retry"""
        return self._call_service(features)
    
    def predict_with_circuit_breaker(self, features: list) -> dict:
        """Make prediction with circuit breaker"""
        return self.circuit_breaker.call(self._call_service, features)
    
    def _call_service(self, features: list) -> dict:
        """Simulate service call with random failures"""
        # Simulate 30% failure rate
        if random.random() < 0.3:
            raise ConnectionError("Service temporarily unavailable")
        
        # Simulate prediction
        return {
            'prediction': sum(features) > 0,
            'confidence': random.random()
        }

class BatchPredictorWithRetry:
    """Batch predictor with per-item retry"""
    
    def __init__(self, client: MLServiceClient):
        self.client = client
        self.retry_config = RetryConfig(max_retries=2, base_delay=0.1)
    
    def predict_batch(self, batch: List[list]) -> List[dict]:
        """Process batch with individual retries"""
        results = []
        failed_indices = []
        
        for i, features in enumerate(batch):
            try:
                result = self.client.predict(features)
                results.append({'index': i, 'result': result, 'status': 'success'})
            except RetryError as e:
                logger.error(f"Item {i} failed after retries: {e}")
                results.append({'index': i, 'result': None, 'status': 'failed'})
                failed_indices.append(i)
        
        if failed_indices:
            logger.warning(f"Batch completed with {len(failed_indices)} failures")
        
        return results

# Demo
print("=== Retry Pattern Demo ===\n")

# Simple retry example
@retry(RetryConfig(max_retries=3, strategy=BackoffStrategy.EXPONENTIAL_JITTER))
def unreliable_operation():
    """Simulates an unreliable operation"""
    if random.random() < 0.7:  # 70% failure rate
        raise ConnectionError("Random failure")
    return "Success!"

print("Testing simple retry:")
try:
    for i in range(3):
        print(f"\nAttempt set {i+1}:")
        result = unreliable_operation()
        print(f"Result: {result}")
except RetryError as e:
    print(f"Final failure: {e}")

# ML Service with retry
print("\n" + "="*50)
print("Testing ML Service Client:")
client = MLServiceClient("http://ml-service:8080")

# Make predictions with retry
print("\nMaking predictions with retry:")
for i in range(5):
    try:
        features = [random.random() for _ in range(5)]
        result = client.predict(features)
        print(f"Prediction {i+1}: {result['prediction']}")
    except RetryError as e:
        print(f"Prediction {i+1} failed: {e}")

# Batch processing
print("\n" + "="*50)
print("Testing Batch Predictor:")
batch_predictor = BatchPredictorWithRetry(client)
batch = [[random.random() for _ in range(5)] for _ in range(10)]
results = batch_predictor.predict_batch(batch)

success_count = sum(1 for r in results if r['status'] == 'success')
print(f"\nBatch Results: {success_count}/{len(batch)} successful")

# Circuit breaker demo
print("\n" + "="*50)
print("Testing Circuit Breaker:")
for i in range(10):
    try:
        features = [random.random() for _ in range(5)]
        result = client.predict_with_circuit_breaker(features)
        print(f"Request {i+1}: Success")
    except Exception as e:
        print(f"Request {i+1}: {type(e).__name__}")
    time.sleep(0.1)
```

### 4. Key Considerations

| Aspect | Best Practice |
|--------|---------------|
| **Idempotency** | Ensure retried operations are safe to repeat |
| **Timeout** | Set reasonable timeouts before retry |
| **Logging** | Log all retries for debugging |
| **Metrics** | Track retry rates and failure patterns |
| **Circuit Breaker** | Combine with retry to prevent overload |

### 5. Interview Tips
- Exponential backoff with jitter is industry standard
- Circuit breaker prevents cascading failures
- Not all operations are safe to retry (idempotency)
- Mention Tenacity, resilience4j as production libraries

---


