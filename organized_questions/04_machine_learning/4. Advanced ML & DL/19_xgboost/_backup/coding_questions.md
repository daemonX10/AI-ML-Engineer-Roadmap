# XGBoost Interview Questions - Coding Questions

## Question 1
- [ ] Done

**What is early stopping in XGBoost and how can it be implemented?**

### Answer

**Definition:**
Early stopping halts training when the validation metric stops improving for a specified number of rounds, preventing overfitting and reducing training time. It automatically finds the optimal number of boosting rounds.

**How It Works:**
```
Round 1-50: Validation score improving → Continue
Round 51-70: Validation score plateaus → Continue watching
Round 71-100: Still no improvement for 50 rounds → STOP at round 50
```

**Implementation:**

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Generate data
X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Method 1: Using XGBClassifier API
model = xgb.XGBClassifier(
    n_estimators=1000,            # Set high
    early_stopping_rounds=50,     # Stop if no improvement for 50 rounds
    learning_rate=0.1,
    max_depth=6,
    eval_metric='logloss'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, 'train'), (X_val, 'validation')],
    verbose=10  # Print every 10 rounds
)

print(f"Best iteration: {model.best_iteration}")
print(f"Best score: {model.best_score:.4f}")

# Predictions use best_iteration automatically
y_pred = model.predict(X_test)


# Method 2: Using native xgb.train API
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'eta': 0.1
}

evals = [(dtrain, 'train'), (dval, 'validation')]
evals_result = {}

model_native = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    evals_result=evals_result,
    verbose_eval=20
)

print(f"Best iteration: {model_native.best_iteration}")


# Method 3: Plot learning curves
def plot_learning_curve(evals_result):
    """Plot training and validation curves."""
    train_metric = evals_result['train']['logloss']
    val_metric = evals_result['validation']['logloss']
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_metric, label='Train')
    plt.plot(val_metric, label='Validation')
    plt.axvline(model_native.best_iteration, color='r', linestyle='--', label=f'Best iteration: {model_native.best_iteration}')
    plt.xlabel('Boosting Round')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.title('Learning Curves with Early Stopping')
    plt.show()

# plot_learning_curve(evals_result)


# Method 4: Custom early stopping callback
class EarlyStoppingCallback:
    """Custom early stopping with additional logic."""
    def __init__(self, stopping_rounds, min_delta=0.001):
        self.stopping_rounds = stopping_rounds
        self.min_delta = min_delta
        self.best_score = float('inf')
        self.best_iteration = 0
        self.counter = 0
    
    def __call__(self, env):
        current_score = env.evaluation_result_list[-1][1]
        
        if current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.best_iteration = env.iteration
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.stopping_rounds:
            print(f"Early stopping at round {env.iteration}")
            raise xgb.core.EarlyStopException(env.iteration)
```

**Key Parameters:**
- `n_estimators`: Set high (early stopping will find optimal)
- `early_stopping_rounds`: Patience (typically 20-100)
- `eval_set`: Required for early stopping
- `eval_metric`: Metric to monitor

**Best Practices:**
- Always use separate validation set (not test)
- Set `n_estimators` high enough
- Patience depends on learning rate (lower η → more patience)

---

## Question 2
- [ ] Done

**Write a Python code to load a dataset, create an XGBoost model, and fit it to the data.**

### Answer

**Complete Training Pipeline:**

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATA
# ==========================================
# Using sklearn dataset (replace with your data loading)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("="*50)
print("DATASET OVERVIEW")
print("="*50)
print(f"Shape: {X.shape}")
print(f"Features: {X.columns.tolist()[:5]}... ({len(X.columns)} total)")
print(f"Target distribution: {np.bincount(y)}")
print(f"Class balance: {y.mean():.2%} positive")


# ==========================================
# 2. PREPARE DATA
# ==========================================
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,  # Maintain class balance
    random_state=42
)

# Further split for validation (early stopping)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, 
    test_size=0.2, 
    random_state=42
)

print(f"\nTrain: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")


# ==========================================
# 3. CREATE AND CONFIGURE MODEL
# ==========================================
model = xgb.XGBClassifier(
    # Core parameters
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    
    # Sampling for regularization
    subsample=0.8,
    colsample_bytree=0.8,
    
    # Regularization
    reg_lambda=1,
    reg_alpha=0,
    
    # Early stopping
    early_stopping_rounds=50,
    
    # Training settings
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 4. FIT MODEL
# ==========================================
print("\n" + "="*50)
print("TRAINING MODEL")
print("="*50)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=20
)

print(f"\nBest iteration: {model.best_iteration}")
print(f"Best validation score: {model.best_score:.4f}")


# ==========================================
# 5. EVALUATE MODEL
# ==========================================
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Metrics
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))


# ==========================================
# 6. FEATURE IMPORTANCE
# ==========================================
print("\n" + "="*50)
print("TOP 10 IMPORTANT FEATURES")
print("="*50)

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head(10).to_string(index=False))


# ==========================================
# 7. SAVE MODEL
# ==========================================
# Save model for later use
model.save_model('xgboost_model.json')
print("\nModel saved to 'xgboost_model.json'")

# Load model
loaded_model = xgb.XGBClassifier()
loaded_model.load_model('xgboost_model.json')


# ==========================================
# 8. VISUALIZATION (Optional)
# ==========================================
def plot_feature_importance(model, feature_names, top_n=15):
    """Plot feature importance."""
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True).tail(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance['feature'], importance['importance'])
    plt.xlabel('Importance')
    plt.title('Top Feature Importance')
    plt.tight_layout()
    plt.show()

# Uncomment to visualize
# plot_feature_importance(model, X.columns)
```

**Output Example:**
```
==================================================
DATASET OVERVIEW
==================================================
Shape: (569, 30)
Target distribution: [212 357]
Class balance: 62.74% positive

Train: 364, Val: 91, Test: 114

==================================================
TRAINING MODEL
==================================================
[0]	validation_0-logloss:0.59238	validation_1-logloss:0.59847
[20]	validation_0-logloss:0.07152	validation_1-logloss:0.11234
[40]	validation_0-logloss:0.03121	validation_1-logloss:0.10856
Best iteration: 45

==================================================
MODEL EVALUATION
==================================================
Accuracy: 0.9649
AUC-ROC: 0.9912
```

---

## Question 3
- [ ] Done

**Implement a Python function that uses cross-validation to optimize the hyperparameters of an XGBoost model.**

### Answer

**Multiple Hyperparameter Optimization Approaches:**

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, 
                                      StratifiedKFold, cross_val_score)
from sklearn.datasets import make_classification
from scipy.stats import randint, uniform
import time

# Generate sample data
X, y = make_classification(n_samples=5000, n_features=20, 
                           n_informative=10, random_state=42)


# ==========================================
# METHOD 1: GRID SEARCH CV
# ==========================================
def grid_search_xgboost(X, y, cv=5):
    """Hyperparameter tuning using Grid Search."""
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    start = time.time()
    grid_search.fit(X, y)
    elapsed = time.time() - start
    
    print(f"\nGrid Search Results (took {elapsed:.1f}s):")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.cv_results_


# ==========================================
# METHOD 2: RANDOMIZED SEARCH CV
# ==========================================
def random_search_xgboost(X, y, n_iter=50, cv=5):
    """Hyperparameter tuning using Randomized Search."""
    
    param_distributions = {
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'n_estimators': randint(100, 500),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_lambda': uniform(0, 10),
        'reg_alpha': uniform(0, 1),
        'gamma': uniform(0, 5)
    }
    
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    random_search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    start = time.time()
    random_search.fit(X, y)
    elapsed = time.time() - start
    
    print(f"\nRandom Search Results (took {elapsed:.1f}s):")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.cv_results_


# ==========================================
# METHOD 3: BAYESIAN OPTIMIZATION (OPTUNA)
# ==========================================
def optuna_xgboost(X, y, n_trials=50, cv=5):
    """Hyperparameter tuning using Optuna."""
    import optuna
    from sklearn.model_selection import cross_val_score
    
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.01, 5.0),
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nOptuna Results:")
    print(f"Best parameters: {study.best_params}")
    print(f"Best score: {study.best_value:.4f}")
    
    # Train final model with best params
    best_model = xgb.XGBClassifier(**study.best_params, use_label_encoder=False, 
                                    eval_metric='logloss', random_state=42)
    best_model.fit(X, y)
    
    return best_model, study


# ==========================================
# METHOD 4: STAGED TUNING (PRACTICAL APPROACH)
# ==========================================
def staged_tuning_xgboost(X, y, cv=5):
    """Two-stage hyperparameter tuning."""
    
    # Stage 1: Tune tree structure
    print("Stage 1: Tuning tree structure...")
    param_grid_1 = {
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1, 3, 5]
    }
    
    base_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    grid1 = GridSearchCV(base_model, param_grid_1, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid1.fit(X, y)
    best_depth = grid1.best_params_['max_depth']
    best_mcw = grid1.best_params_['min_child_weight']
    print(f"  Best: max_depth={best_depth}, min_child_weight={best_mcw}")
    
    # Stage 2: Tune sampling parameters
    print("Stage 2: Tuning sampling parameters...")
    param_grid_2 = {
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    model2 = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=best_depth,
        min_child_weight=best_mcw,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    grid2 = GridSearchCV(model2, param_grid_2, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid2.fit(X, y)
    best_subsample = grid2.best_params_['subsample']
    best_colsample = grid2.best_params_['colsample_bytree']
    print(f"  Best: subsample={best_subsample}, colsample_bytree={best_colsample}")
    
    # Stage 3: Final model with lower learning rate
    print("Stage 3: Training final model...")
    final_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=best_depth,
        min_child_weight=best_mcw,
        subsample=best_subsample,
        colsample_bytree=best_colsample,
        early_stopping_rounds=50,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    # Use CV to estimate performance
    scores = cross_val_score(final_model, X, y, cv=cv, scoring='roc_auc')
    print(f"\nFinal CV Score: {scores.mean():.4f} ± {scores.std():.4f}")
    
    return final_model


# Run optimization
print("="*60)
print("HYPERPARAMETER OPTIMIZATION")
print("="*60)

# Choose one method:
best_model, results = random_search_xgboost(X, y, n_iter=30)

# Or use staged tuning
# best_model = staged_tuning_xgboost(X, y)
```

---

## Question 4
- [ ] Done

**Code a Python script that demonstrates how to use XGBoost's built-in feature importance to rank features.**

### Answer

**Multiple Feature Importance Methods:**

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)


# ==========================================
# METHOD 1: BUILT-IN FEATURE IMPORTANCE
# ==========================================
def get_builtin_importance(model, feature_names):
    """Get all types of built-in feature importance."""
    
    importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    results = {}
    
    for imp_type in importance_types:
        importance = model.get_booster().get_score(importance_type=imp_type)
        # Fill missing features with 0
        importance = {f: importance.get(f, 0) for f in feature_names}
        results[imp_type] = importance
    
    # Create DataFrame
    df = pd.DataFrame(results, index=feature_names)
    df.columns = [c.upper() for c in df.columns]
    
    return df

importance_df = get_builtin_importance(model, X.columns.tolist())
print("="*60)
print("BUILT-IN FEATURE IMPORTANCE")
print("="*60)
print("\nTop 10 by GAIN:")
print(importance_df.sort_values('GAIN', ascending=False).head(10)[['GAIN', 'WEIGHT', 'COVER']])


# ==========================================
# METHOD 2: USING feature_importances_ ATTRIBUTE
# ==========================================
def rank_features_simple(model, feature_names, top_n=10):
    """Simple ranking using feature_importances_."""
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance['rank'] = range(1, len(importance) + 1)
    importance['cumulative'] = importance['importance'].cumsum()
    
    print("\n" + "="*60)
    print(f"TOP {top_n} FEATURES (by default importance)")
    print("="*60)
    print(importance.head(top_n).to_string(index=False))
    
    return importance

importance_simple = rank_features_simple(model, X.columns.tolist())


# ==========================================
# METHOD 3: PLOT FEATURE IMPORTANCE
# ==========================================
def plot_importance_comparison(model, feature_names, top_n=15):
    """Plot different importance types side by side."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    importance_types = ['weight', 'gain', 'cover']
    titles = ['Weight (# of splits)', 'Gain (avg improvement)', 'Cover (# samples)']
    
    for ax, imp_type, title in zip(axes, importance_types, titles):
        xgb.plot_importance(
            model, 
            importance_type=imp_type, 
            max_num_features=top_n,
            ax=ax,
            title=title
        )
    
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png', dpi=150)
    plt.show()

# plot_importance_comparison(model, X.columns.tolist())


# ==========================================
# METHOD 4: FEATURE SELECTION BASED ON IMPORTANCE
# ==========================================
def select_features_by_importance(model, X, threshold='median'):
    """Select features based on importance threshold."""
    from sklearn.feature_selection import SelectFromModel
    
    selector = SelectFromModel(model, threshold=threshold, prefit=True)
    X_selected = selector.transform(X)
    
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    
    print(f"\n" + "="*60)
    print(f"FEATURE SELECTION (threshold={threshold})")
    print("="*60)
    print(f"Selected {len(selected_features)} of {X.shape[1]} features:")
    print(selected_features)
    
    return selected_features, X_selected

selected, X_selected = select_features_by_importance(model, X_train)


# ==========================================
# METHOD 5: CUMULATIVE IMPORTANCE ANALYSIS
# ==========================================
def analyze_cumulative_importance(model, feature_names, target_coverage=0.95):
    """Find minimum features needed for target importance coverage."""
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance['cumulative'] = importance['importance'].cumsum()
    importance['cumulative_pct'] = importance['cumulative'] / importance['importance'].sum()
    
    # Find features needed for target coverage
    n_features = (importance['cumulative_pct'] <= target_coverage).sum() + 1
    
    print(f"\n" + "="*60)
    print(f"CUMULATIVE IMPORTANCE ANALYSIS")
    print("="*60)
    print(f"Features needed for {target_coverage:.0%} importance: {n_features}")
    print(f"\nTop features covering {target_coverage:.0%}:")
    print(importance.head(n_features)[['feature', 'importance', 'cumulative_pct']].to_string(index=False))
    
    return importance

cumulative_analysis = analyze_cumulative_importance(model, X.columns.tolist())


# ==========================================
# METHOD 6: COMPARE WITH PERMUTATION IMPORTANCE
# ==========================================
def compare_importance_methods(model, X_test, y_test, feature_names, top_n=10):
    """Compare built-in vs permutation importance."""
    from sklearn.inspection import permutation_importance
    
    # Built-in (Gain)
    builtin = pd.DataFrame({
        'feature': feature_names,
        'builtin': model.feature_importances_
    })
    
    # Permutation
    perm_result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'permutation': perm_result.importances_mean,
        'perm_std': perm_result.importances_std
    })
    
    # Merge and compare
    comparison = builtin.merge(perm_df, on='feature')
    comparison['builtin_rank'] = comparison['builtin'].rank(ascending=False)
    comparison['perm_rank'] = comparison['permutation'].rank(ascending=False)
    comparison['rank_diff'] = abs(comparison['builtin_rank'] - comparison['perm_rank'])
    
    print(f"\n" + "="*60)
    print(f"IMPORTANCE METHOD COMPARISON (Top {top_n})")
    print("="*60)
    print(comparison.sort_values('builtin', ascending=False).head(top_n).to_string(index=False))
    
    return comparison

comparison = compare_importance_methods(model, X_test, y_test, X.columns.tolist())
```

---

## Question 5
- [ ] Done

**Implement an XGBoost model on a given dataset and use SHAP values to interpret the model's predictions.**

### Answer

**Complete SHAP Interpretation Pipeline:**

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Install: pip install shap
import shap

# ==========================================
# 1. LOAD DATA AND TRAIN MODEL
# ==========================================
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

print(f"Model Accuracy: {model.score(X_test, y_test):.4f}")


# ==========================================
# 2. CREATE SHAP EXPLAINER
# ==========================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print(f"\nSHAP values shape: {shap_values.shape}")
print(f"Expected value (base): {explainer.expected_value:.4f}")


# ==========================================
# 3. GLOBAL INTERPRETATION: Summary Plot
# ==========================================
def global_interpretation(shap_values, X_test, feature_names):
    """Show overall feature importance and effects."""
    
    print("\n" + "="*60)
    print("GLOBAL INTERPRETATION")
    print("="*60)
    
    # Summary plot (bar)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                      plot_type='bar', show=False)
    plt.title('Global Feature Importance (Mean |SHAP|)')
    plt.tight_layout()
    plt.savefig('shap_global_bar.png', dpi=150)
    plt.show()
    
    # Summary plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title('Feature Impact Distribution')
    plt.tight_layout()
    plt.savefig('shap_beeswarm.png', dpi=150)
    plt.show()

# global_interpretation(shap_values, X_test, X.columns.tolist())


# ==========================================
# 4. LOCAL INTERPRETATION: Single Prediction
# ==========================================
def explain_single_prediction(model, explainer, X_sample, feature_names, sample_idx=0):
    """Explain a single prediction in detail."""
    
    print("\n" + "="*60)
    print(f"LOCAL INTERPRETATION (Sample {sample_idx})")
    print("="*60)
    
    # Get SHAP values for this sample
    shap_vals = explainer.shap_values(X_sample.iloc[[sample_idx]])
    
    # Prediction
    pred_proba = model.predict_proba(X_sample.iloc[[sample_idx]])[0]
    pred_class = model.predict(X_sample.iloc[[sample_idx]])[0]
    
    print(f"Prediction: Class {pred_class}")
    print(f"Probability: {pred_proba[1]:.4f} (benign)")
    print(f"Base value: {explainer.expected_value:.4f}")
    
    # Top contributing features
    contributions = pd.DataFrame({
        'feature': feature_names,
        'value': X_sample.iloc[sample_idx].values,
        'shap_value': shap_vals[0]
    }).sort_values('shap_value', key=abs, ascending=False)
    
    print("\nTop 5 Contributing Features:")
    for _, row in contributions.head(5).iterrows():
        direction = "↑" if row['shap_value'] > 0 else "↓"
        print(f"  {direction} {row['feature']}: {row['value']:.3f} → SHAP: {row['shap_value']:.4f}")
    
    # Waterfall plot
    shap.waterfall_plot(shap.Explanation(
        values=shap_vals[0],
        base_values=explainer.expected_value,
        data=X_sample.iloc[sample_idx].values,
        feature_names=feature_names
    ), show=False)
    plt.title(f'Prediction Explanation (Sample {sample_idx})')
    plt.tight_layout()
    plt.savefig(f'shap_waterfall_{sample_idx}.png', dpi=150)
    plt.show()
    
    return contributions

# explain_single_prediction(model, explainer, X_test, X.columns.tolist(), sample_idx=0)


# ==========================================
# 5. FEATURE DEPENDENCE PLOTS
# ==========================================
def feature_dependence_analysis(shap_values, X_test, feature_names, feature_idx=0):
    """Show how a feature affects predictions."""
    
    feature = feature_names[feature_idx]
    
    print("\n" + "="*60)
    print(f"DEPENDENCE ANALYSIS: {feature}")
    print("="*60)
    
    # Dependence plot
    shap.dependence_plot(
        feature_idx, 
        shap_values, 
        X_test,
        feature_names=feature_names,
        show=False
    )
    plt.title(f'SHAP Dependence: {feature}')
    plt.tight_layout()
    plt.savefig(f'shap_dependence_{feature}.png', dpi=150)
    plt.show()

# feature_dependence_analysis(shap_values, X_test, X.columns.tolist(), feature_idx=0)


# ==========================================
# 6. FORCE PLOT (INTERACTIVE)
# ==========================================
def force_plot_explanation(explainer, shap_values, X_test, sample_idx=0):
    """Interactive force plot for single prediction."""
    
    # Initialize JavaScript visualization
    shap.initjs()
    
    # Single prediction force plot
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[sample_idx],
        X_test.iloc[sample_idx],
        feature_names=X_test.columns.tolist()
    )
    
    return force_plot

# force_plot = force_plot_explanation(explainer, shap_values, X_test, 0)


# ==========================================
# 7. GENERATE INTERPRETATION REPORT
# ==========================================
def generate_interpretation_report(model, X_test, y_test, explainer, shap_values):
    """Generate comprehensive interpretation report."""
    
    print("\n" + "="*60)
    print("SHAP INTERPRETATION REPORT")
    print("="*60)
    
    # Model performance
    accuracy = model.score(X_test, y_test)
    print(f"\n1. Model Accuracy: {accuracy:.4f}")
    
    # Global importance (mean |SHAP|)
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'mean_abs_shap': mean_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    print("\n2. Top 10 Features (Mean |SHAP|):")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"   {i}. {row['feature']}: {row['mean_abs_shap']:.4f}")
    
    # Feature direction summary
    print("\n3. Feature Effects Summary:")
    mean_shap_signed = shap_values.mean(axis=0)
    for i, (feat, shap_val) in enumerate(zip(X_test.columns, mean_shap_signed)):
        if abs(shap_val) > 0.05:  # Only significant features
            direction = "increases" if shap_val > 0 else "decreases"
            print(f"   • {feat}: {direction} prediction probability")
    
    # Base value interpretation
    base_prob = 1 / (1 + np.exp(-explainer.expected_value))
    print(f"\n4. Base probability (average prediction): {base_prob:.4f}")
    
    return importance_df

report = generate_interpretation_report(model, X_test, y_test, explainer, shap_values)


# ==========================================
# 8. PRACTICAL: EXPLAIN WHY PREDICTION DIFFERS
# ==========================================
def compare_predictions(model, explainer, X_test, y_test, idx1=0, idx2=1):
    """Compare SHAP explanations for two different predictions."""
    
    print("\n" + "="*60)
    print(f"COMPARING SAMPLES {idx1} vs {idx2}")
    print("="*60)
    
    for idx in [idx1, idx2]:
        sample = X_test.iloc[[idx]]
        shap_vals = explainer.shap_values(sample)[0]
        pred = model.predict(sample)[0]
        proba = model.predict_proba(sample)[0, 1]
        actual = y_test.iloc[idx]
        
        print(f"\nSample {idx}:")
        print(f"  Prediction: {pred}, Actual: {actual}")
        print(f"  Probability: {proba:.4f}")
        
        top_features = pd.Series(shap_vals, index=X_test.columns).abs().nlargest(3)
        print(f"  Top factors: {', '.join(top_features.index.tolist())}")

compare_predictions(model, explainer, X_test, y_test, 0, 5)
```

**Key SHAP Concepts:**
- **Base Value**: Average model prediction
- **SHAP Value**: Feature's contribution to moving prediction from base
- **Positive SHAP**: Pushes prediction toward positive class
- **Negative SHAP**: Pushes prediction toward negative class

**Interview Point:**
"SHAP provides mathematically grounded explanations. For any prediction, I can show exactly how much each feature contributed, making it suitable for regulated industries requiring model transparency."
