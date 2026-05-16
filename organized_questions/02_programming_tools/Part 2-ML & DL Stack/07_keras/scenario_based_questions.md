# Keras Interview Questions - Scenario-Based Questions

## Question 1
- [ ] Done

**Your model trains fine but predictions are poor in production. What could be wrong?**

### Answer

### Common Issues

| Issue | Solution |
|-------|----------|
| **Training mode in inference** | Use `model.predict()` not `model(x, training=True)` |
| **Different preprocessing** | Save and reuse preprocessing params |
| **Data distribution shift** | Monitor input distributions |
| **Missing normalization** | Ensure same scaling in production |

### Python Code Example
```python
from tensorflow import keras
import numpy as np

# Problem: Dropout/BatchNorm behavior
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1)
])

# WRONG: training=True in production
# bad_pred = model(X_test, training=True)  # Dropout active!

# CORRECT: Use predict() or training=False
good_pred = model.predict(X_test)  # training=False by default
# or
good_pred = model(X_test, training=False)

# Save preprocessing parameters
class ProductionPipeline:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return self
    
    def transform(self, X):
        return (X - self.mean) / (self.std + 1e-7)
    
    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)
    
    def load(self, path):
        data = np.load(path)
        self.mean = data['mean']
        self.std = data['std']

# Use same preprocessing
pipeline = ProductionPipeline()
pipeline.fit(X_train)
pipeline.save('preprocessing.npz')

# In production
prod_pipeline = ProductionPipeline()
prod_pipeline.load('preprocessing.npz')
X_prod_normalized = prod_pipeline.transform(X_production)
```

---

## Question 2
- [ ] Done

**Your model is too large for mobile deployment. How do you reduce its size?**

### Answer

### Size Reduction Techniques

| Technique | Size Reduction | Accuracy Impact |
|-----------|----------------|-----------------|
| **Quantization** | 2-4x | Minimal |
| **Pruning** | 2-10x | Minimal |
| **Knowledge Distillation** | Variable | Some loss |
| **Architecture** | Variable | Depends |

### Python Code Example
```python
from tensorflow import keras
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Original model
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 1. Post-training quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

original_size = len(model.to_json())
quantized_size = len(quantized_model)
print(f"Size reduction: {original_size/quantized_size:.1f}x")

# 2. Pruning during training
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
}

pruned_model = prune_low_magnitude(model, **pruning_params)
pruned_model.compile(optimizer='adam', 
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

# 3. Knowledge distillation
class Distiller(keras.Model):
    def __init__(self, student, teacher, temperature=3):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
    
    def train_step(self, data):
        x, y = data
        
        # Teacher predictions (soft labels)
        teacher_predictions = self.teacher(x, training=False)
        
        with tf.GradientTape() as tape:
            # Student predictions
            student_predictions = self.student(x, training=True)
            
            # Distillation loss
            soft_loss = keras.losses.KLDivergence()(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1)
            )
            
            # Hard label loss
            hard_loss = keras.losses.sparse_categorical_crossentropy(
                y, student_predictions
            )
            
            loss = soft_loss + hard_loss
        
        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.student.trainable_variables)
        )
        
        return {'loss': loss}
```

---

## Question 3
- [ ] Done

**You need to train on data that doesn't fit in memory. What's your approach?**

### Answer

### Solutions

| Method | Use Case |
|--------|----------|
| **tf.data.Dataset** | Most flexible |
| **ImageDataGenerator** | Image data (legacy) |
| **Generators** | Custom data loading |
| **TFRecord** | Large datasets |

### Python Code Example
```python
from tensorflow import keras
import tensorflow as tf
import numpy as np

# Method 1: tf.data.Dataset from generator
def data_generator(file_paths, batch_size):
    """Generator that yields batches"""
    while True:
        np.random.shuffle(file_paths)
        for path in file_paths:
            # Load data from file
            data = np.load(path)
            X, y = data['X'], data['y']
            
            for i in range(0, len(X), batch_size):
                yield X[i:i+batch_size], y[i:i+batch_size]

# Create dataset from generator
def create_dataset(file_paths, batch_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(file_paths, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Method 2: Memory mapping for large files
class MemoryMappedDataset(keras.utils.Sequence):
    def __init__(self, data_path, batch_size):
        self.data = np.load(data_path, mmap_mode='r')  # Memory mapped
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.data['X']) // self.batch_size
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        return self.data['X'][start:end], self.data['y'][start:end]

# Method 3: TFRecord for efficient storage
def create_tfrecord(X, y, filename):
    writer = tf.io.TFRecordWriter(filename)
    for i in range(len(X)):
        feature = {
            'X': tf.train.Feature(float_list=tf.train.FloatList(value=X[i])),
            'y': tf.train.Feature(int64_list=tf.train.Int64List(value=[y[i]]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()

def read_tfrecord(filename):
    def parse_fn(example):
        features = {
            'X': tf.io.FixedLenFeature([10], tf.float32),
            'y': tf.io.FixedLenFeature([1], tf.int64)
        }
        parsed = tf.io.parse_single_example(example, features)
        return parsed['X'], parsed['y'][0]
    
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_fn)
    return dataset

# Use with model
# model.fit(dataset, epochs=10, steps_per_epoch=1000)
```

---

## Question 4
- [ ] Done

**Your model accuracy plateaus early in training. How do you diagnose and fix this?**

### Answer

### Diagnostic Steps

| Check | What to Look For |
|-------|------------------|
| Learning rate | Too high/low |
| Loss function | Appropriate for task |
| Data quality | Label noise, imbalance |
| Architecture | Capacity issues |
| Gradients | Vanishing/exploding |

### Python Code Example
```python
from tensorflow import keras
import tensorflow as tf
import numpy as np

# 1. Check if model can overfit on small batch
def overfit_test(model, X, y, epochs=100):
    """Model should reach ~100% on small batch"""
    X_small, y_small = X[:32], y[:32]
    
    model_copy = keras.models.clone_model(model)
    model_copy.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
                       metrics=['accuracy'])
    
    history = model_copy.fit(X_small, y_small, epochs=epochs, verbose=0)
    
    if history.history['accuracy'][-1] < 0.95:
        print("❌ Cannot overfit - check model capacity or data")
    else:
        print("✅ Model can learn")

# 2. Learning rate finder
class LRFinder(keras.callbacks.Callback):
    def __init__(self, min_lr=1e-7, max_lr=10, steps=100):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.steps = steps
        self.losses = []
        self.lrs = []
    
    def on_train_begin(self, logs=None):
        self.step = 0
        self.factor = (self.max_lr / self.min_lr) ** (1 / self.steps)
        keras.backend.set_value(self.model.optimizer.lr, self.min_lr)
    
    def on_batch_end(self, batch, logs=None):
        self.step += 1
        lr = self.min_lr * (self.factor ** self.step)
        keras.backend.set_value(self.model.optimizer.lr, lr)
        
        self.losses.append(logs['loss'])
        self.lrs.append(lr)
        
        if self.step >= self.steps:
            self.model.stop_training = True
    
    def get_optimal_lr(self):
        min_loss_idx = np.argmin(self.losses)
        return self.lrs[min_loss_idx] / 10  # Use 1/10 of min

# 3. Gradient checking
@tf.function
def check_gradients(model, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = keras.losses.sparse_categorical_crossentropy(y, predictions)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    for var, grad in zip(model.trainable_variables, gradients):
        if grad is not None:
            grad_norm = tf.norm(grad).numpy()
            if grad_norm < 1e-7:
                print(f"⚠️ {var.name}: Vanishing gradient")
            elif grad_norm > 1000:
                print(f"⚠️ {var.name}: Exploding gradient")

# 4. Solutions
# Use learning rate warmup
class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        return self.initial_lr * tf.minimum(1.0, step / self.warmup_steps)

# Use gradient clipping
optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
```

---

## Question 5
- [ ] Done

**You need to explain your Keras model's predictions to stakeholders. How do you do it?**

### Answer

### Interpretability Techniques

| Technique | Use Case |
|-----------|----------|
| **Feature importance** | Tabular data |
| **Grad-CAM** | Image models |
| **SHAP** | Model-agnostic |
| **Attention visualization** | Sequence models |

### Python Code Example
```python
from tensorflow import keras
import tensorflow as tf
import numpy as np

# 1. Grad-CAM for CNNs
def make_gradcam_heatmap(model, img_array, last_conv_layer_name):
    # Create model that outputs conv layer and predictions
    grad_model = keras.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        top_class = tf.argmax(predictions[0])
        top_class_channel = predictions[:, top_class]
    
    # Gradients of top class w.r.t. conv output
    grads = tape.gradient(top_class_channel, conv_outputs)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight conv outputs by importance
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    
    # Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# 2. Feature importance for Dense networks
def get_feature_importance(model, X, y, feature_names):
    baseline_loss = model.evaluate(X, y, verbose=0)[0]
    importances = []
    
    for i in range(X.shape[1]):
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[:, i])
        
        permuted_loss = model.evaluate(X_permuted, y, verbose=0)[0]
        importance = permuted_loss - baseline_loss
        importances.append(importance)
    
    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx[:10]:
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")
    
    return importances

# 3. SHAP values (pip install shap)
import shap

def explain_with_shap(model, X_train, X_explain):
    # For Keras models
    explainer = shap.DeepExplainer(model, X_train[:100])
    shap_values = explainer.shap_values(X_explain)
    
    # Summary plot
    shap.summary_plot(shap_values, X_explain)
    
    # Force plot for single prediction
    shap.force_plot(
        explainer.expected_value[0],
        shap_values[0][0],
        X_explain[0]
    )

# 4. Attention visualization
def visualize_attention(model, text, tokenizer):
    """For models with attention layers"""
    # Get attention weights from model
    attention_layer = model.get_layer('attention')
    
    attention_model = keras.Model(
        inputs=model.input,
        outputs=[attention_layer.output, model.output]
    )
    
    # Tokenize input
    tokens = tokenizer.texts_to_sequences([text])
    attention_weights, predictions = attention_model.predict(tokens)
    
    # Display attention weights
    words = text.split()
    for word, weight in zip(words, attention_weights[0]):
        print(f"{word}: {'█' * int(weight * 20)}")
```

---

## Question 6
- [ ] Done

**Your model works well on validation but poorly on real-world data. What's happening?**

### Answer

### Possible Causes

| Issue | Description |
|-------|-------------|
| **Data leakage** | Information from future/target in features |
| **Distribution shift** | Real data differs from training |
| **Overfitting to validation** | Hyperparameter tuning on val set |
| **Sample bias** | Training data not representative |

### Python Code Example
```python
from tensorflow import keras
import numpy as np
from scipy.stats import ks_2samp

# 1. Check for data drift
def detect_data_drift(training_data, production_data, threshold=0.05):
    drifted_features = []
    
    for i in range(training_data.shape[1]):
        stat, p_value = ks_2samp(
            training_data[:, i],
            production_data[:, i]
        )
        if p_value < threshold:
            drifted_features.append(i)
    
    if drifted_features:
        print(f"⚠️ Data drift in features: {drifted_features}")
    return drifted_features

# 2. Check for data leakage
def check_leakage(X_train, X_val, y_train, y_val):
    """Check if validation examples appear in training"""
    # Check for duplicate rows
    combined = np.vstack([X_train, X_val])
    unique_rows = np.unique(combined, axis=0)
    
    if len(unique_rows) < len(combined):
        print("⚠️ Possible data leakage: duplicate rows found")
    
    # Check correlation of features with target
    for i in range(X_train.shape[1]):
        corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
        if abs(corr) > 0.95:
            print(f"⚠️ Feature {i} has high correlation with target: {corr:.3f}")

# 3. Proper train/val/test split
from sklearn.model_selection import train_test_split

def proper_split(X, y):
    """Use separate test set that's never touched"""
    # First split: separate test set (hold out completely)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    # Second split: train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.18, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# 4. Cross-validation for hyperparameter tuning
from sklearn.model_selection import KFold

def cross_validate_model(create_model_fn, X, y, n_folds=5):
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kfold.split(X):
        model = create_model_fn()
        model.fit(X[train_idx], y[train_idx], epochs=10, verbose=0)
        score = model.evaluate(X[val_idx], y[val_idx], verbose=0)[1]
        scores.append(score)
    
    print(f"CV Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores
```

---

## Question 7
- [ ] Done

**You need to serve a Keras model with low latency. What optimizations do you apply?**

### Answer

### Optimization Techniques

| Technique | Benefit |
|-----------|---------|
| **Batching** | Higher throughput |
| **Quantization** | Faster inference |
| **TF-TRT** | GPU optimization |
| **XLA compilation** | Graph optimization |
| **Model simplification** | Reduce complexity |

### Python Code Example
```python
from tensorflow import keras
import tensorflow as tf
import numpy as np
import time

# 1. Quantization for faster inference
def quantize_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Full integer quantization
    def representative_data():
        for _ in range(100):
            yield [np.random.randn(1, 10).astype(np.float32)]
    
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    return converter.convert()

# 2. XLA compilation
@tf.function(jit_compile=True)
def fast_inference(model, x):
    return model(x, training=False)

# 3. Batch inference
def batch_predict(model, data, batch_size=256):
    predictions = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        pred = model.predict(batch, verbose=0)
        predictions.append(pred)
    return np.vstack(predictions)

# 4. TensorFlow Serving optimized model
def export_for_serving(model, export_path):
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 10], dtype=tf.float32, name='input')
    ])
    def serve(x):
        return {'predictions': model(x)}
    
    tf.saved_model.save(
        model, 
        export_path,
        signatures={'serving_default': serve}
    )

# 5. Benchmark
def benchmark_inference(model, data, n_runs=100):
    # Warmup
    _ = model.predict(data[:1], verbose=0)
    
    # Single inference
    start = time.time()
    for _ in range(n_runs):
        _ = model.predict(data[:1], verbose=0)
    single_time = (time.time() - start) / n_runs
    
    # Batch inference
    start = time.time()
    _ = model.predict(data, verbose=0)
    batch_time = time.time() - start
    
    print(f"Single inference: {single_time*1000:.2f}ms")
    print(f"Batch inference ({len(data)}): {batch_time*1000:.2f}ms")
    print(f"Per-sample in batch: {batch_time/len(data)*1000:.2f}ms")
```

---

## Question 8
- [ ] Done

**You're asked to implement A/B testing for two model versions. How do you approach this?**

### Answer

### A/B Testing Framework

| Component | Purpose |
|-----------|---------|
| **Traffic split** | Random user assignment |
| **Metrics** | Define success criteria |
| **Statistical test** | Determine significance |
| **Monitoring** | Track real-time performance |

### Python Code Example
```python
from tensorflow import keras
import numpy as np
from scipy import stats
import hashlib

class ABTestingFramework:
    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio
        
        # Metrics storage
        self.metrics_a = {'predictions': [], 'outcomes': []}
        self.metrics_b = {'predictions': [], 'outcomes': []}
    
    def assign_variant(self, user_id):
        """Deterministic assignment based on user_id"""
        hash_value = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        return 'A' if (hash_value % 100) < (self.split_ratio * 100) else 'B'
    
    def predict(self, user_id, features):
        """Route prediction to appropriate model"""
        variant = self.assign_variant(user_id)
        
        if variant == 'A':
            prediction = self.model_a.predict(features, verbose=0)
            return prediction, 'A'
        else:
            prediction = self.model_b.predict(features, verbose=0)
            return prediction, 'B'
    
    def log_outcome(self, variant, prediction, actual_outcome):
        """Log outcomes for analysis"""
        if variant == 'A':
            self.metrics_a['predictions'].append(prediction)
            self.metrics_a['outcomes'].append(actual_outcome)
        else:
            self.metrics_b['predictions'].append(prediction)
            self.metrics_b['outcomes'].append(actual_outcome)
    
    def analyze_results(self):
        """Statistical comparison of A vs B"""
        outcomes_a = np.array(self.metrics_a['outcomes'])
        outcomes_b = np.array(self.metrics_b['outcomes'])
        
        # Conversion rate comparison
        rate_a = np.mean(outcomes_a)
        rate_b = np.mean(outcomes_b)
        
        # Statistical significance (chi-square test)
        contingency = [
            [np.sum(outcomes_a), len(outcomes_a) - np.sum(outcomes_a)],
            [np.sum(outcomes_b), len(outcomes_b) - np.sum(outcomes_b)]
        ]
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        
        print(f"Model A rate: {rate_a:.4f} (n={len(outcomes_a)})")
        print(f"Model B rate: {rate_b:.4f} (n={len(outcomes_b)})")
        print(f"Relative improvement: {(rate_b - rate_a) / rate_a * 100:.2f}%")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            winner = 'B' if rate_b > rate_a else 'A'
            print(f"✅ Statistically significant. Model {winner} wins.")
        else:
            print("❌ Not statistically significant yet.")
        
        return {'rate_a': rate_a, 'rate_b': rate_b, 'p_value': p_value}

# Usage
ab_test = ABTestingFramework(model_a, model_b, split_ratio=0.5)

# Simulate serving
for user_id in range(1000):
    features = np.random.randn(1, 10).astype(np.float32)
    prediction, variant = ab_test.predict(user_id, features)
    
    # Simulate outcome (real outcome would come later)
    outcome = 1 if np.random.random() < 0.1 + 0.02 * (variant == 'B') else 0
    ab_test.log_outcome(variant, prediction, outcome)

# Analyze
results = ab_test.analyze_results()
```

---

## Question 9
- [ ] Done

**Your training is unstable with loss spikes. How do you stabilize it?**

### Answer

### Stabilization Techniques

| Technique | Implementation |
|-----------|----------------|
| **Gradient clipping** | `clipnorm`, `clipvalue` |
| **Learning rate warmup** | Gradual LR increase |
| **Batch normalization** | Normalize activations |
| **Proper initialization** | He/Glorot init |
| **Mixed precision fix** | Loss scaling |

### Python Code Example
```python
from tensorflow import keras
import tensorflow as tf

# 1. Gradient clipping
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    clipnorm=1.0,  # Clip gradient norm
    # clipvalue=0.5  # Or clip by value
)

# 2. Learning rate warmup
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, total_steps):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # Warmup phase
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        
        # Cosine decay phase
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_lr = self.initial_lr * 0.5 * (1 + tf.cos(np.pi * progress))
        
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

# 3. Proper weight initialization
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer='he_normal',  # For ReLU
                       input_shape=(10,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(32, activation='relu',
                       kernel_initializer='he_normal'),
    keras.layers.Dense(1)
])

# 4. Mixed precision with loss scaling
policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)

# Loss scaling is automatic, but can be manual:
optimizer = keras.mixed_precision.LossScaleOptimizer(
    keras.optimizers.Adam(),
    dynamic=True  # Automatically adjust loss scale
)

# 5. Gradient monitoring callback
class GradientMonitor(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        # Check for NaN loss
        if logs and np.isnan(logs.get('loss', 0)):
            print(f"⚠️ NaN loss detected at batch {batch}")
            self.model.stop_training = True

# 6. Complete stable training setup
def create_stable_model():
    model = keras.Sequential([
        keras.layers.Dense(128, kernel_initializer='he_normal', input_shape=(10,)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(64, kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        
        keras.layers.Dense(1)
    ])
    
    optimizer = keras.optimizers.Adam(
        learning_rate=WarmupCosineDecay(0.001, 1000, 10000),
        clipnorm=1.0
    )
    
    model.compile(optimizer=optimizer, loss='mse')
    return model
```

---

## Question 10
- [ ] Done

**You need to implement continual learning where the model learns from new data without forgetting old patterns. How?**

### Answer

### Continual Learning Strategies

| Strategy | Description |
|----------|-------------|
| **Elastic Weight Consolidation** | Penalize changes to important weights |
| **Experience Replay** | Mix old and new data |
| **Progressive Networks** | Add capacity for new tasks |
| **Knowledge Distillation** | Preserve old knowledge |

### Python Code Example
```python
from tensorflow import keras
import tensorflow as tf
import numpy as np

# 1. Experience Replay
class ExperienceReplay:
    def __init__(self, buffer_size=10000):
        self.buffer_size = buffer_size
        self.X_buffer = []
        self.y_buffer = []
    
    def add(self, X, y):
        self.X_buffer.extend(X)
        self.y_buffer.extend(y)
        
        # Keep buffer manageable
        if len(self.X_buffer) > self.buffer_size:
            self.X_buffer = self.X_buffer[-self.buffer_size:]
            self.y_buffer = self.y_buffer[-self.buffer_size:]
    
    def sample(self, n):
        indices = np.random.choice(len(self.X_buffer), min(n, len(self.X_buffer)))
        return np.array([self.X_buffer[i] for i in indices]), \
               np.array([self.y_buffer[i] for i in indices])

# 2. Elastic Weight Consolidation
class EWC:
    def __init__(self, model, X_old, y_old, lambda_ewc=1000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        
        # Store old weights
        self.old_weights = [w.numpy() for w in model.trainable_weights]
        
        # Compute Fisher information (importance of each weight)
        self.fisher = self._compute_fisher(X_old, y_old)
    
    def _compute_fisher(self, X, y):
        fisher = []
        
        for _ in range(len(self.model.trainable_weights)):
            fisher.append(np.zeros_like(self.model.trainable_weights[_].numpy()))
        
        # Compute gradients for each sample
        for i in range(min(len(X), 100)):
            with tf.GradientTape() as tape:
                pred = self.model(X[i:i+1], training=True)
                loss = keras.losses.sparse_categorical_crossentropy(y[i:i+1], pred)
            
            grads = tape.gradient(loss, self.model.trainable_weights)
            
            for j, grad in enumerate(grads):
                if grad is not None:
                    fisher[j] += grad.numpy() ** 2
        
        # Normalize
        fisher = [f / len(X) for f in fisher]
        return fisher
    
    def ewc_loss(self):
        """Penalty for changing important weights"""
        ewc_loss = 0
        for i, (w, old_w, f) in enumerate(zip(
            self.model.trainable_weights, self.old_weights, self.fisher
        )):
            ewc_loss += tf.reduce_sum(f * (w - old_w) ** 2)
        return self.lambda_ewc * ewc_loss

# 3. Continual Learning Training
class ContinualLearner:
    def __init__(self, model):
        self.model = model
        self.replay = ExperienceReplay(buffer_size=5000)
        self.ewc = None
    
    def learn_task(self, X_new, y_new, epochs=10, replay_ratio=0.3):
        """Learn new task while remembering old"""
        
        # Add new data to replay buffer
        self.replay.add(list(X_new), list(y_new))
        
        for epoch in range(epochs):
            # Mix new data with replay
            X_replay, y_replay = self.replay.sample(int(len(X_new) * replay_ratio))
            
            X_combined = np.vstack([X_new, X_replay])
            y_combined = np.concatenate([y_new, y_replay])
            
            # Shuffle
            indices = np.random.permutation(len(X_combined))
            X_combined = X_combined[indices]
            y_combined = y_combined[indices]
            
            # Train with EWC if we have old task
            if self.ewc:
                # Custom training step with EWC loss
                self._train_with_ewc(X_combined, y_combined)
            else:
                self.model.fit(X_combined, y_combined, epochs=1, verbose=0)
        
        # Update EWC after learning task
        self.ewc = EWC(self.model, X_new, y_new)
    
    def _train_with_ewc(self, X, y):
        optimizer = self.model.optimizer
        
        with tf.GradientTape() as tape:
            predictions = self.model(X, training=True)
            task_loss = keras.losses.sparse_categorical_crossentropy(y, predictions)
            task_loss = tf.reduce_mean(task_loss)
            
            ewc_loss = self.ewc.ewc_loss()
            total_loss = task_loss + ewc_loss
        
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

# Usage
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(5, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

learner = ContinualLearner(model)

# Learn task 1
learner.learn_task(X_task1, y_task1)

# Learn task 2 (without forgetting task 1)
learner.learn_task(X_task2, y_task2)
```

## Question 11
- [ ] Done

**How would you architect a Keras model to handle a large-scale image recognition problem ?**

### Answer

### Architecture Design

| Component | Choice | Reason |
|-----------|--------|--------|
| **Base Model** | EfficientNet / ResNet pre-trained | Transfer learning for fast convergence |
| **Input Pipeline** | `tf.data` with prefetching | Handle large datasets efficiently |
| **Augmentation** | Keras preprocessing layers | On-GPU augmentation |
| **Training** | Mixed precision + distributed | Scale to millions of images |
| **Regularization** | Dropout, label smoothing, weight decay | Prevent overfitting |

```python
import tensorflow as tf
from tensorflow import keras

# 1. Efficient data pipeline for large datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    'train/', image_size=(224, 224), batch_size=64, label_mode='categorical'
)
train_ds = train_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)

# 2. On-GPU augmentation
augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2),
])

# 3. Transfer learning with EfficientNet
base = keras.applications.EfficientNetB3(weights='imagenet', include_top=False,
                                          input_shape=(224, 224, 3))
base.trainable = False  # Freeze initially

model = keras.Sequential([
    augmentation,
    base,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')
])

# 4. Mixed precision for 2-3x speedup
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 5. Multi-GPU training
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )

model.fit(train_ds, epochs=20, callbacks=[
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint('best.keras', save_best_only=True)
])
```

> **Interview Tip:** For large-scale image recognition, mention: **transfer learning** (don't train from scratch), **tf.data pipelines** with prefetching, **mixed-precision training**, and **distributed strategies** for multi-GPU scaling.

---

## Question 12
- [ ] Done

**How would you use Keras to build models for sequence-to-sequence tasks , such as machine translation ?**

### Answer

### Architecture

| Component | Choice | Purpose |
|-----------|--------|---------|
| **Encoder** | Bidirectional LSTM/GRU | Compress source sentence into context |
| **Decoder** | LSTM/GRU with attention | Generate target tokens one by one |
| **Attention** | Bahdanau or multi-head | Focus on relevant source positions |
| **Embedding** | Learned or pre-trained | Convert tokens to dense vectors |
| **Modern alternative** | Transformer | State-of-the-art for translation |

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Encoder
encoder_input = layers.Input(shape=(None,), name='encoder_input')
encoder_embed = layers.Embedding(src_vocab_size, 256, mask_zero=True)(encoder_input)
encoder_lstm = layers.Bidirectional(
    layers.LSTM(256, return_sequences=True, return_state=True)
)
encoder_out, fwd_h, fwd_c, bwd_h, bwd_c = encoder_lstm(encoder_embed)
state_h = layers.Concatenate()([fwd_h, bwd_h])
state_c = layers.Concatenate()([fwd_c, bwd_c])

# Attention mechanism
attention = layers.Attention()

# Decoder
decoder_input = layers.Input(shape=(None,), name='decoder_input')
decoder_embed = layers.Embedding(tgt_vocab_size, 256, mask_zero=True)(decoder_input)
decoder_lstm = layers.LSTM(512, return_sequences=True)
decoder_out = decoder_lstm(decoder_embed, initial_state=[state_h, state_c])

# Apply attention: decoder attends to encoder outputs
context = attention([decoder_out, encoder_out])
decoder_combined = layers.Concatenate()([decoder_out, context])
output = layers.TimeDistributed(layers.Dense(tgt_vocab_size, activation='softmax'))(decoder_combined)

model = keras.Model([encoder_input, decoder_input], output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# For production: use Transformer architecture (tf.keras.layers.MultiHeadAttention)
```

### Inference (Autoregressive Decoding)
1. Encode source sentence
2. Start with `<START>` token
3. Predict next token, append to output
4. Repeat until `<END>` token or max length

> **Interview Tip:** Mention that while LSTM seq2seq is foundational, modern translation uses **Transformers** (`MultiHeadAttention`). Key concepts: **teacher forcing** during training, **beam search** during inference, and **BLEU score** for evaluation.

---
## Question 13
- [ ] Done

**Discuss a strategy for implementing a real-time object detection system using Keras.**

**Answer:**

```python
import tensorflow as tf
import numpy as np
import cv2

# === Strategy 1: Using Pre-trained YOLOv5/YOLOv8 ===
# pip install ultralytics
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Nano model for speed

# Real-time webcam detection
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    annotated = results[0].plot()  # Draw bounding boxes
    cv2.imshow('Detection', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

# === Strategy 2: TFLite for Edge/Mobile ===
# Convert model
converter = tf.lite.TFLiteConverter.from_saved_model('ssd_mobilenet/')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Run inference
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

def detect_tflite(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_data = cv2.resize(image, (300, 300))
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32) / 255.0
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    
    return boxes, classes, scores

# === Strategy 3: SSD MobileNet in Keras ===
# Fine-tune for custom objects
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet'
)
base.trainable = False

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4 + num_classes)  # bbox (4) + class scores
])
```

| Model | FPS | Accuracy (mAP) | Best For |
|-------|-----|----------------|----------|
| YOLOv8n | ~60+ | 37.3 | Real-time, general |
| SSD MobileNet | ~30 | 22.0 | Mobile/edge |
| EfficientDet | ~15 | 51.0 | High accuracy |
| Faster R-CNN | ~5 | 42.0 | Accuracy over speed |

| Optimization | Effect |
|-------------|--------|
| Quantization (INT8) | 2-4x faster, 4x smaller |
| TensorRT | 2-5x faster on NVIDIA GPUs |
| Model pruning | Remove unused weights |
| Input resolution | Lower = faster (trade accuracy) |

> **Interview Tip:** For real-time detection, YOLO family models dominate. For edge deployment, convert to TFLite with quantization. The key trade-off is always speed vs. accuracy—choose based on latency requirements.

## Question 14
- [ ] Done

**Describe how you would use Keras to develop a recommendation system.**

### Answer

**Definition**: Recommendation systems predict user preferences for items. Keras can implement various recommendation approaches including collaborative filtering (learning user-item interactions), content-based filtering, and hybrid models using neural networks.

### Recommendation Approaches

| Approach | Method | When to Use |
|----------|--------|-------------|
| **Collaborative Filtering** | User-item interaction patterns | Sufficient user history |
| **Content-Based** | Item/user feature similarity | Cold-start items |
| **Neural Collaborative Filtering** | Deep learning embeddings | Large-scale, complex patterns |
| **Hybrid** | Combine multiple approaches | Production systems |

### Python Code Example — Neural Collaborative Filtering
```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Assume: user_ids, item_ids, ratings as training data
num_users = 10000
num_items = 5000
embedding_dim = 64

# METHOD 1: Matrix Factorization with Embeddings
user_input = keras.Input(shape=(1,), name='user_id')
item_input = keras.Input(shape=(1,), name='item_id')

# User embedding
user_embedding = layers.Embedding(num_users, embedding_dim,
                                   name='user_embedding')(user_input)
user_vec = layers.Flatten()(user_embedding)

# Item embedding
item_embedding = layers.Embedding(num_items, embedding_dim,
                                   name='item_embedding')(item_input)
item_vec = layers.Flatten()(item_embedding)

# Dot product → predicted rating
dot_product = layers.Dot(axes=1)([user_vec, item_vec])
output = layers.Dense(1, activation='sigmoid')(dot_product)  # Scale to [0, 1]

mf_model = keras.Model([user_input, item_input], output)
mf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
mf_model.fit([user_ids, item_ids], ratings, epochs=20, batch_size=256,
             validation_split=0.2)

# METHOD 2: Deep Neural Collaborative Filtering (NCF)
user_input = keras.Input(shape=(1,), name='user_id')
item_input = keras.Input(shape=(1,), name='item_id')

# GMF path (Generalized Matrix Factorization)
user_emb_gmf = layers.Embedding(num_users, 32)(user_input)
item_emb_gmf = layers.Embedding(num_items, 32)(item_input)
gmf = layers.Multiply()([layers.Flatten()(user_emb_gmf),
                          layers.Flatten()(item_emb_gmf)])

# MLP path (Multi-Layer Perceptron)
user_emb_mlp = layers.Embedding(num_users, 32)(user_input)
item_emb_mlp = layers.Embedding(num_items, 32)(item_input)
mlp = layers.Concatenate()([layers.Flatten()(user_emb_mlp),
                             layers.Flatten()(item_emb_mlp)])
mlp = layers.Dense(128, activation='relu')(mlp)
mlp = layers.Dropout(0.3)(mlp)
mlp = layers.Dense(64, activation='relu')(mlp)
mlp = layers.Dropout(0.3)(mlp)

# Combine GMF + MLP
combined = layers.Concatenate()([gmf, mlp])
output = layers.Dense(32, activation='relu')(combined)
output = layers.Dense(1, activation='sigmoid')(output)

ncf_model = keras.Model([user_input, item_input], output)
ncf_model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

# METHOD 3: Content-Based with Side Features
user_input = keras.Input(shape=(1,), name='user_id')
item_input = keras.Input(shape=(1,), name='item_id')
user_features = keras.Input(shape=(20,), name='user_features')  # Age, etc.
item_features = keras.Input(shape=(50,), name='item_features')  # Genre, etc.

user_emb = layers.Flatten()(layers.Embedding(num_users, 32)(user_input))
item_emb = layers.Flatten()(layers.Embedding(num_items, 32)(item_input))

# Combine embeddings with side features
combined = layers.Concatenate()([user_emb, item_emb,
                                  user_features, item_features])
x = layers.Dense(256, activation='relu')(combined)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
output = layers.Dense(1, activation='sigmoid')(x)

hybrid_model = keras.Model(
    [user_input, item_input, user_features, item_features], output
)
hybrid_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

### Making Recommendations
```python
# Predict ratings for all items for a user
user_id = 42
all_items = np.arange(num_items)
user_array = np.full_like(all_items, user_id)

predicted_ratings = mf_model.predict([user_array, all_items])

# Get top-10 recommendations
top_10_items = np.argsort(predicted_ratings.flatten())[-10:][::-1]
print(f"Top 10 recommended items for user {user_id}: {top_10_items}")
```

### Interview Tips
- Embedding layers are the foundation of neural recommendation — they learn latent representations of users and items
- Neural Collaborative Filtering (NCF) combines matrix factorization with deep learning for superior performance
- Cold-start problem (new users/items): use content-based features alongside embeddings
- For production, consider approximate nearest neighbor (ANN) search for fast retrieval from millions of items
- Mention evaluation metrics: NDCG, Hit Rate, MAP, and not just MSE/MAE

---

## Question 15
- [ ] Done

**Present a framework for anomaly detection using autoencoders in Keras.**

**Answer:**

Autoencoders learn to reconstruct normal data. High reconstruction error = anomaly.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# === 1. Build Autoencoder ===
def build_autoencoder(input_dim, encoding_dim=32):
    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(128, activation='relu')(inputs)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)  # Bottleneck
    
    # Decoder
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
    
    autoencoder = models.Model(inputs, decoded)
    encoder = models.Model(inputs, encoded)
    return autoencoder, encoder

autoencoder, encoder = build_autoencoder(input_dim=100)
autoencoder.compile(optimizer='adam', loss='mse')

# === 2. Train on Normal Data Only ===
# X_normal = ... (only non-anomalous samples)
autoencoder.fit(
    X_normal, X_normal,  # Input = Target (reconstruction)
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
)

# === 3. Detect Anomalies ===
def detect_anomalies(model, data, threshold=None, percentile=95):
    reconstructed = model.predict(data)
    mse = np.mean(np.power(data - reconstructed, 2), axis=1)
    
    if threshold is None:
        threshold = np.percentile(mse, percentile)
    
    anomalies = mse > threshold
    return anomalies, mse, threshold

anomalies, errors, threshold = detect_anomalies(autoencoder, X_test)
print(f"Threshold: {threshold:.4f}")
print(f"Anomalies found: {anomalies.sum()} / {len(X_test)}")

# === 4. Variational Autoencoder (Better) ===
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(input_dim, latent_dim=16):
    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = models.Model(inputs, [z_mean, z_log_var, z])
    
    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(64, activation='relu')(latent_inputs)
    outputs = layers.Dense(input_dim, activation='sigmoid')(x)
    decoder = models.Model(latent_inputs, outputs)
    
    return encoder, decoder
```

| Component | Purpose |
|-----------|--------|
| Encoder | Compress input to latent representation |
| Bottleneck | Force learning of essential features |
| Decoder | Reconstruct from compressed representation |
| Reconstruction Error | High error = input doesn't match learned "normal" |
| Threshold | Percentile-based or domain-specific cutoff |

> **Interview Tip:** VAEs are generally better than vanilla autoencoders for anomaly detection because they learn a smooth latent space. The reconstruction error + KL divergence together provide a more robust anomaly score.
