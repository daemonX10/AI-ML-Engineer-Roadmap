# TensorFlow Interview Questions - Scenario-Based Questions

## Question 1
- [ ] Done

**Discuss how TensorFlow can be used for reinforcement learning.**

### Answer

TensorFlow provides tools for building RL agents through custom training loops and specialized libraries.

### Core Components

| Component | TensorFlow Implementation |
|-----------|---------------------------|
| Environment | OpenAI Gym, custom env |
| Policy Network | `tf.keras.Model` |
| Value Network | `tf.keras.Model` |
| Experience Replay | `tf.data.Dataset` |
| Training | `tf.GradientTape` |

### Python Code Example
```python
import tensorflow as tf
import numpy as np

# Simple Q-Learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Build Q-Network
        self.model = self._build_model()
        self.target_model = self._build_model()
    
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', 
                                  input_shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss='mse')
        return model
    
    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        for i in batch:
            state, action, reward, next_state, done = self.memory[i]
            target = reward
            if not done:
                target += self.gamma * np.max(
                    self.target_model.predict(next_state[np.newaxis], verbose=0)
                )
            
            target_q = self.model.predict(state[np.newaxis], verbose=0)
            target_q[0][action] = target
            
            self.model.fit(state[np.newaxis], target_q, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

# Usage with gym environment
# import gym
# env = gym.make('CartPole-v1')
# agent = DQNAgent(state_size=4, action_size=2)
```

---

## Question 2
- [ ] Done

**Your model trains slowly on large datasets. How do you optimize the data pipeline?**

### Answer

### Optimization Strategies

| Strategy | Implementation | Benefit |
|----------|----------------|---------|
| **Prefetching** | `prefetch(AUTOTUNE)` | Overlap I/O and compute |
| **Parallel mapping** | `map(..., num_parallel_calls)` | Parallelize transformations |
| **Caching** | `cache()` | Avoid re-reading data |
| **Interleaving** | `interleave()` | Parallel file reads |
| **Batching** | `batch()` before expensive ops | Process multiple at once |

### Python Code Example
```python
import tensorflow as tf

# INEFFICIENT pipeline
def slow_pipeline(file_paths):
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(load_image)  # Sequential
    dataset = dataset.batch(32)
    return dataset

# OPTIMIZED pipeline
def fast_pipeline(file_paths):
    AUTOTUNE = tf.data.AUTOTUNE
    
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    
    # Parallel file reading
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=4,
        num_parallel_calls=AUTOTUNE
    )
    
    # Parallel preprocessing
    dataset = dataset.map(
        parse_and_augment,
        num_parallel_calls=AUTOTUNE
    )
    
    # Cache after expensive operations
    dataset = dataset.cache()
    
    # Shuffle with buffer
    dataset = dataset.shuffle(buffer_size=1000)
    
    # Batch
    dataset = dataset.batch(32)
    
    # Prefetch next batch while training on current
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset

# Benchmark
import time

def benchmark(dataset, num_epochs=2):
    start = time.time()
    for _ in range(num_epochs):
        for batch in dataset:
            pass  # Simulate training step
    return time.time() - start

# Example usage with TFRecords (most efficient format)
def create_tfrecord(images, labels, filename):
    writer = tf.io.TFRecordWriter(filename)
    for img, label in zip(images, labels):
        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()
```

---

## Question 3
- [ ] Done

**You need to deploy a TensorFlow model to production. What's your approach?**

### Answer

### Deployment Options

| Option | Use Case | Latency |
|--------|----------|---------|
| **TensorFlow Serving** | Server-side, high throughput | Low |
| **TensorFlow Lite** | Mobile, embedded | Very low |
| **TensorFlow.js** | Browser | Medium |
| **ONNX** | Cross-platform | Varies |

### Python Code Example
```python
import tensorflow as tf

# 1. Save model in SavedModel format
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train model...
# model.fit(X_train, y_train, epochs=10)

# Save for TensorFlow Serving
model.save('saved_model/my_model')

# 2. Convert to TFLite (for mobile)
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/my_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantization
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# 3. Define serving signature (for REST API)
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 10], dtype=tf.float32)])
def serve(x):
    return {'predictions': model(x)}

# Save with explicit signature
tf.saved_model.save(
    model, 
    'saved_model/my_model_serving',
    signatures={'serving_default': serve}
)

# 4. Docker deployment with TensorFlow Serving
# Dockerfile:
# FROM tensorflow/serving
# COPY saved_model/my_model /models/my_model
# ENV MODEL_NAME=my_model

# Run: docker run -p 8501:8501 my_model_image
# API: curl -X POST http://localhost:8501/v1/models/my_model:predict \
#      -d '{"instances": [[1,2,3,4,5,6,7,8,9,10]]}'
```

---

## Question 4
- [ ] Done

**Your model has memory issues when training on large images. How do you solve this?**

### Answer

### Memory Optimization Strategies

| Strategy | Description |
|----------|-------------|
| **Batch size reduction** | Smaller batches use less memory |
| **Mixed precision** | Float16 uses half the memory |
| **Gradient accumulation** | Simulate large batches |
| **Memory growth** | Allocate GPU memory as needed |
| **Model checkpointing** | Trade compute for memory |

### Python Code Example
```python
import tensorflow as tf

# 1. Enable memory growth (don't allocate all GPU memory)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 2. Mixed precision training (float16)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Model will use float16 for most operations
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    # Output layer in float32 for stability
    tf.keras.layers.Dense(10, activation='softmax', dtype='float32')
])

# 3. Gradient accumulation (simulate larger batch)
def train_with_accumulation(model, dataset, accumulation_steps=4):
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    @tf.function
    def train_step(x, y, accumulated_gradients):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_fn(y, predictions) / accumulation_steps
        
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Accumulate gradients
        for i, grad in enumerate(gradients):
            if grad is not None:
                accumulated_gradients[i].assign_add(grad)
        
        return loss
    
    for epoch in range(10):
        # Initialize accumulated gradients
        accumulated_grads = [tf.Variable(tf.zeros_like(v)) 
                           for v in model.trainable_variables]
        
        for step, (x, y) in enumerate(dataset):
            loss = train_step(x, y, accumulated_grads)
            
            # Apply gradients every accumulation_steps
            if (step + 1) % accumulation_steps == 0:
                optimizer.apply_gradients(
                    zip([g.value() for g in accumulated_grads], 
                        model.trainable_variables)
                )
                for g in accumulated_grads:
                    g.assign(tf.zeros_like(g))

# 4. Use generators for large datasets
def data_generator(file_paths, batch_size):
    while True:
        for path in file_paths:
            # Load one batch at a time
            images = load_images(path)
            for i in range(0, len(images), batch_size):
                yield images[i:i+batch_size]
```

---

## Question 5
- [ ] Done

**You need to handle a multi-GPU training setup. What's your approach?**

### Answer

### Multi-GPU Strategies

| Strategy | Use Case |
|----------|----------|
| `MirroredStrategy` | Single machine, multiple GPUs |
| `MultiWorkerMirroredStrategy` | Multiple machines |
| `TPUStrategy` | TPU pods |
| `ParameterServerStrategy` | Large scale async training |

### Python Code Example
```python
import tensorflow as tf

# Check available GPUs
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

# 1. MirroredStrategy (most common)
strategy = tf.distribute.MirroredStrategy()

print(f"Number of replicas: {strategy.num_replicas_in_sync}")

# Build model within strategy scope
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Adjust batch size for multi-GPU
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# Load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0

# Create distributed dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(60000).batch(GLOBAL_BATCH_SIZE)

# Train
model.fit(train_dataset, epochs=10)

# 2. Custom training loop with distribution
@tf.function
def distributed_train_step(dist_inputs):
    per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                          axis=None)

# 3. Multi-worker setup (cluster)
# Set TF_CONFIG environment variable for each worker
# os.environ['TF_CONFIG'] = json.dumps({
#     'cluster': {'worker': ['host1:port', 'host2:port']},
#     'task': {'type': 'worker', 'index': 0}
# })
# strategy = tf.distribute.MultiWorkerMirroredStrategy()
```

---

## Question 6
- [ ] Done

**Your model's accuracy is good but inference is too slow. How do you optimize?**

### Answer

### Optimization Techniques

| Technique | Speedup | Accuracy Impact |
|-----------|---------|-----------------|
| **Quantization** | 2-4x | Minimal |
| **Pruning** | 1.5-3x | Minimal |
| **Model distillation** | Variable | Some loss |
| **Graph optimization** | 1.2-2x | None |
| **Batching** | Linear | None |

### Python Code Example
```python
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Original model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 1. Post-training quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Full integer quantization
def representative_data_gen():
    for i in range(100):
        yield [X_train[i:i+1].astype('float32')]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

quantized_model = converter.convert()

# 2. Pruning (during training)
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
                     loss='sparse_categorical_crossentropy')

# 3. Use XLA compilation
@tf.function(jit_compile=True)
def fast_predict(x):
    return model(x)

# 4. Batch predictions
def batch_predict(model, data, batch_size=256):
    predictions = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        predictions.append(model.predict(batch, verbose=0))
    return np.concatenate(predictions)

# Benchmark
import time
test_data = tf.random.normal([1000, 28, 28, 1])

# Single inference
start = time.time()
for i in range(1000):
    model.predict(test_data[i:i+1], verbose=0)
print(f"Single: {time.time() - start:.2f}s")

# Batched inference
start = time.time()
model.predict(test_data, batch_size=256, verbose=0)
print(f"Batched: {time.time() - start:.2f}s")
```

---

## Question 7
- [ ] Done

**You need to debug a model that's not learning. What's your systematic approach?**

### Answer

### Debugging Checklist

| Step | What to Check |
|------|---------------|
| 1 | Data preprocessing |
| 2 | Learning rate |
| 3 | Loss function |
| 4 | Model architecture |
| 5 | Gradient flow |

### Python Code Example
```python
import tensorflow as tf
import numpy as np

# 1. Check data
def verify_data(X, y):
    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")
    print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"y unique values: {np.unique(y)}")
    print(f"NaN in X: {np.isnan(X).sum()}")
    print(f"Class distribution: {np.bincount(y.flatten().astype(int))}")

# 2. Check gradients
def check_gradients(model, x_sample, y_sample):
    with tf.GradientTape() as tape:
        predictions = model(x_sample, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_sample, predictions)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    for i, (var, grad) in enumerate(zip(model.trainable_variables, gradients)):
        if grad is None:
            print(f"⚠️ Layer {var.name}: No gradient!")
        else:
            grad_norm = tf.norm(grad).numpy()
            if grad_norm < 1e-7:
                print(f"⚠️ Layer {var.name}: Vanishing gradient ({grad_norm:.2e})")
            elif grad_norm > 1000:
                print(f"⚠️ Layer {var.name}: Exploding gradient ({grad_norm:.2e})")
            else:
                print(f"✅ Layer {var.name}: Gradient norm = {grad_norm:.4f}")

# 3. Overfit on small batch (sanity check)
def overfit_test(model, X, y, epochs=100):
    """Model should reach ~100% accuracy on small batch"""
    X_small, y_small = X[:32], y[:32]
    
    history = model.fit(X_small, y_small, epochs=epochs, verbose=0)
    
    final_loss = history.history['loss'][-1]
    final_acc = history.history.get('accuracy', [0])[-1]
    
    if final_loss > 0.1:
        print(f"⚠️ Cannot overfit: loss={final_loss:.4f}")
    else:
        print(f"✅ Successfully overfit: loss={final_loss:.4f}, acc={final_acc:.4f}")

# 4. Learning rate finder
def find_lr(model, X, y, min_lr=1e-7, max_lr=10, epochs=5):
    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: min_lr * (max_lr/min_lr)**(epoch/epochs)
    )
    
    history = model.fit(
        X, y, 
        epochs=epochs, 
        callbacks=[lr_callback],
        verbose=0
    )
    
    # Plot loss vs learning rate
    lrs = [min_lr * (max_lr/min_lr)**(i/epochs) for i in range(epochs)]
    losses = history.history['loss']
    
    # Find optimal LR (where loss starts increasing)
    min_loss_idx = np.argmin(losses)
    optimal_lr = lrs[min_loss_idx] / 10  # Use 1/10 of minimum
    
    print(f"Suggested learning rate: {optimal_lr:.6f}")
    return optimal_lr

# 5. Custom callback for debugging
class DebugCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Check for NaN loss
        if np.isnan(logs.get('loss', 0)):
            print("⚠️ NaN loss detected! Stopping training.")
            self.model.stop_training = True
        
        # Check for learning stagnation
        if epoch > 0:
            if hasattr(self, 'prev_loss'):
                if abs(logs['loss'] - self.prev_loss) < 1e-7:
                    print(f"⚠️ Loss not changing at epoch {epoch}")
        self.prev_loss = logs['loss']
```

---

## Question 8
- [ ] Done

**You're building a model for a streaming data application. How do you handle continuous learning?**

### Answer

### Strategies for Continuous Learning

| Approach | Description |
|----------|-------------|
| **Online learning** | Update model with each batch |
| **Incremental training** | Periodically retrain on new data |
| **Elastic weight consolidation** | Prevent forgetting |
| **Experience replay** | Mix old and new data |

### Python Code Example
```python
import tensorflow as tf
import numpy as np

# Online learning with streaming data
class OnlineLearningModel:
    def __init__(self, input_shape, num_classes):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', 
                                  input_shape=(input_shape,)),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        
        # Experience replay buffer
        self.buffer_size = 10000
        self.buffer_X = []
        self.buffer_y = []
    
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.loss_fn(y, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        return loss
    
    def update(self, new_x, new_y, replay_ratio=0.5):
        """Update model with new data + replay buffer"""
        # Add to buffer
        self.buffer_X.extend(new_x)
        self.buffer_y.extend(new_y)
        
        # Keep buffer size manageable
        if len(self.buffer_X) > self.buffer_size:
            self.buffer_X = self.buffer_X[-self.buffer_size:]
            self.buffer_y = self.buffer_y[-self.buffer_size:]
        
        # Mix new data with replay
        batch_size = len(new_x)
        replay_size = int(batch_size * replay_ratio)
        
        if len(self.buffer_X) > replay_size:
            indices = np.random.choice(len(self.buffer_X), replay_size, replace=False)
            replay_x = [self.buffer_X[i] for i in indices]
            replay_y = [self.buffer_y[i] for i in indices]
            
            combined_x = np.vstack([new_x, replay_x])
            combined_y = np.concatenate([new_y, replay_y])
        else:
            combined_x = new_x
            combined_y = new_y
        
        # Train step
        combined_x = tf.convert_to_tensor(combined_x, dtype=tf.float32)
        combined_y = tf.convert_to_tensor(combined_y, dtype=tf.int32)
        
        loss = self.train_step(combined_x, combined_y)
        return loss.numpy()
    
    def predict(self, x):
        return self.model.predict(x, verbose=0)

# Usage: Streaming data simulation
model = OnlineLearningModel(input_shape=10, num_classes=3)

# Simulate streaming batches
for batch_num in range(100):
    # New batch arrives
    new_X = np.random.randn(32, 10).astype(np.float32)
    new_y = np.random.randint(0, 3, 32)
    
    loss = model.update(new_X, new_y)
    
    if batch_num % 10 == 0:
        print(f"Batch {batch_num}: Loss = {loss:.4f}")
```

---

## Question 9
- [ ] Done

**Your model performs well on test data but poorly in production. What could be wrong?**

### Answer

### Common Issues

| Issue | Solution |
|-------|----------|
| **Data drift** | Monitor input distribution |
| **Feature skew** | Ensure same preprocessing |
| **Serving latency** | Optimize model |
| **Label leakage** | Review feature engineering |

### Python Code Example
```python
import tensorflow as tf
import numpy as np
from scipy.stats import ks_2samp

class ProductionMonitor:
    def __init__(self, reference_data):
        self.reference_data = reference_data
        self.reference_mean = np.mean(reference_data, axis=0)
        self.reference_std = np.std(reference_data, axis=0)
        self.prediction_history = []
    
    def check_data_drift(self, production_data, threshold=0.05):
        """Detect if production data distribution differs from training"""
        drift_detected = []
        
        for i in range(production_data.shape[1]):
            stat, p_value = ks_2samp(
                self.reference_data[:, i],
                production_data[:, i]
            )
            if p_value < threshold:
                drift_detected.append(i)
        
        if drift_detected:
            print(f"⚠️ Data drift detected in features: {drift_detected}")
        return drift_detected
    
    def check_prediction_drift(self, predictions, window=100):
        """Monitor prediction distribution over time"""
        self.prediction_history.extend(predictions.tolist())
        
        if len(self.prediction_history) < 2 * window:
            return False
        
        recent = self.prediction_history[-window:]
        historical = self.prediction_history[-2*window:-window]
        
        stat, p_value = ks_2samp(historical, recent)
        
        if p_value < 0.05:
            print(f"⚠️ Prediction drift detected (p={p_value:.4f})")
            return True
        return False
    
    def validate_input(self, x):
        """Check if input is within expected range"""
        z_scores = np.abs((x - self.reference_mean) / (self.reference_std + 1e-7))
        outliers = np.any(z_scores > 5, axis=1)  # > 5 std deviations
        
        if np.any(outliers):
            print(f"⚠️ {outliers.sum()} outlier samples detected")
        return ~outliers

# Consistent preprocessing
class ProductionPreprocessor:
    """Ensure same preprocessing in training and production"""
    
    def __init__(self):
        self.fitted = False
    
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-7
        self.fitted = True
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise ValueError("Preprocessor not fitted!")
        return (X - self.mean) / self.std
    
    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)
    
    def load(self, path):
        data = np.load(path)
        self.mean = data['mean']
        self.std = data['std']
        self.fitted = True
        return self

# Usage
preprocessor = ProductionPreprocessor()
preprocessor.fit(X_train)
preprocessor.save('preprocessor.npz')

# In production
prod_preprocessor = ProductionPreprocessor().load('preprocessor.npz')
X_prod_normalized = prod_preprocessor.transform(X_production)
```

---

## Question 10
- [ ] Done

**You need to build a TensorFlow model that can handle variable-length sequences. How do you approach this?**

### Answer

### Approaches for Variable-Length Sequences

| Method | Description |
|--------|-------------|
| **Padding** | Pad to max length |
| **Masking** | Ignore padded values |
| **Ragged tensors** | Native variable-length support |
| **Bucketing** | Group by similar lengths |

### Python Code Example
```python
import tensorflow as tf
import numpy as np

# Method 1: Padding + Masking
def pad_sequences_example():
    # Variable length sequences
    sequences = [
        [1, 2, 3],
        [4, 5, 6, 7, 8],
        [9, 10]
    ]
    
    # Pad to same length
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, 
        padding='post',  # Pad at end
        truncating='post',
        maxlen=10
    )
    print(f"Padded shape: {padded.shape}")
    
    # Model with masking
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(100, 32, mask_zero=True),  # mask_zero=True!
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1)
    ])
    
    return model

# Method 2: Ragged Tensors (native variable length)
def ragged_tensor_example():
    # Create ragged tensor
    sequences = tf.ragged.constant([
        [1, 2, 3],
        [4, 5, 6, 7, 8],
        [9, 10]
    ])
    
    # RNN layer can handle ragged tensors
    lstm = tf.keras.layers.LSTM(64)
    output = lstm(sequences)
    
    return output

# Method 3: Bucketing for efficient batching
def create_bucketed_dataset(sequences, labels, bucket_boundaries=[5, 10, 20]):
    """Group sequences by similar lengths for efficient batching"""
    
    def element_length_fn(x, y):
        return tf.shape(x)[0]
    
    def bucket_batch_sizes(bucket_id, batch_size=32):
        return batch_size
    
    dataset = tf.data.Dataset.from_generator(
        lambda: zip(sequences, labels),
        output_signature=(
            tf.TensorSpec(shape=[None], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.int32)
        )
    )
    
    # Bucket by sequence length
    dataset = dataset.bucket_by_sequence_length(
        element_length_func=element_length_fn,
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=[32] * (len(bucket_boundaries) + 1),
        padded_shapes=([None], [])
    )
    
    return dataset

# Method 4: Custom masking layer
class SequenceModel(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, lstm_units):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.lstm = tf.keras.layers.LSTM(lstm_units)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, mask=None, training=False):
        # Create mask for padding (0s)
        if mask is None:
            mask = tf.not_equal(inputs, 0)
        
        # Embed sequences
        x = self.embedding(inputs)
        
        # Apply mask to LSTM
        x = self.lstm(x, mask=mask)
        
        return self.dense(x)

# Usage
model = SequenceModel(vocab_size=10000, embed_dim=128, lstm_units=64)
model.compile(optimizer='adam', loss='binary_crossentropy')

# Sample data with variable lengths (padded)
X = tf.constant([
    [1, 2, 3, 0, 0],
    [4, 5, 6, 7, 8],
    [9, 10, 0, 0, 0]
])
y = tf.constant([0, 1, 0])

# Train
# model.fit(X, y, epochs=5)
```

## Question 11
- [ ] Done

**How would you apply TensorFlow to predict stock market trends ?**

### Answer

### Approach

| Component | Choice | Reason |
|-----------|--------|--------|
| **Data** | Historical OHLCV + technical indicators | Captures price patterns |
| **Model** | LSTM / Transformer | Handles sequential dependencies |
| **Features** | Moving averages, RSI, MACD, volume | Domain-specific signals |
| **Output** | Next-day direction (up/down) or price | Classification or regression |
| **Evaluation** | Sharpe ratio, directional accuracy | Finance-specific metrics |

```python
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Feature Engineering
def create_features(df):
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['Returns'] = df['Close'].pct_change()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df.dropna()

# 2. Prepare sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, :-1])
        y.append(data[i, -1])
    return np.array(X), np.array(y)

# 3. Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(60, n_features)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Walk-forward validation (crucial for time-series)
# Never shuffle time-series data; use expanding or rolling window
```

### Key Considerations

| Challenge | Solution |
|-----------|----------|
| Look-ahead bias | Strict train/test temporal split |
| Non-stationarity | Use returns, not raw prices |
| Overfitting | Dropout, early stopping, walk-forward validation |
| Feature leakage | Compute indicators before splitting |
| Market regime changes | Rolling window retraining |

> **Interview Tip:** Emphasize that stock prediction is extremely difficult. Mention **walk-forward validation** (never shuffle time-series), **feature engineering** from domain knowledge, and the importance of **risk-adjusted metrics** (Sharpe ratio) over raw accuracy.

---
## Question 12
- [ ] Done

**Explain what steps you would take to develop a chatbot using TensorFlow.**

### Answer

**Definition**: Building a chatbot with TensorFlow involves creating a **sequence-to-sequence (Seq2Seq)** or **Transformer-based** model that takes user input text and generates appropriate responses.

### Development Steps

| Step | Task | TensorFlow Tool |
|------|------|----------------|
| 1 | Collect conversation data | `tf.data.TextLineDataset` |
| 2 | Preprocess text | `TextVectorization`, tokenizers |
| 3 | Build model | Transformer / Seq2Seq with attention |
| 4 | Train on dialogue pairs | `model.fit()` |
| 5 | Implement inference | Greedy / beam search decoding |
| 6 | Deploy | TF Serving, TFLite |

### Python Code Example
```python
import tensorflow as tf
import numpy as np

# --- Step 1: Prepare dialogue data ---
questions = [
    "hi how are you", "what is your name",
    "what can you do", "tell me a joke",
    "goodbye", "thanks for the help"
]
answers = [
    "i am doing well thank you", "i am a chatbot",
    "i can answer your questions", "why did the chicken cross the road",
    "goodbye have a nice day", "you are welcome"
]

# Add start/end tokens for decoder
answers_in = ["<start> " + a for a in answers]
answers_out = [a + " <end>" for a in answers]

# --- Step 2: Tokenization ---
VOCAB_SIZE = 1000
MAX_LEN = 20

encoder_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LEN
)
decoder_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LEN
)

encoder_vectorizer.adapt(questions)
decoder_vectorizer.adapt(answers_in + answers_out)

# --- Step 3: Build Transformer Chatbot ---
def positional_encoding(length, depth):
    positions = np.arange(length)[:, np.newaxis]
    dims = np.arange(depth)[np.newaxis, :]
    angles = positions / np.power(10000, (2 * (dims // 2)) / depth)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)

EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256

# Encoder
encoder_input = tf.keras.Input(shape=(MAX_LEN,), name='encoder_input')
enc_emb = tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM)(encoder_input)
enc_emb = enc_emb + positional_encoding(MAX_LEN, EMBED_DIM)
enc_attn = tf.keras.layers.MultiHeadAttention(
    num_heads=NUM_HEADS, key_dim=EMBED_DIM // NUM_HEADS
)(enc_emb, enc_emb)
enc_norm = tf.keras.layers.LayerNormalization()(enc_emb + enc_attn)
enc_ff = tf.keras.layers.Dense(FF_DIM, activation='relu')(enc_norm)
enc_ff = tf.keras.layers.Dense(EMBED_DIM)(enc_ff)
encoder_output = tf.keras.layers.LayerNormalization()(enc_norm + enc_ff)

# Decoder
decoder_input = tf.keras.Input(shape=(MAX_LEN,), name='decoder_input')
dec_emb = tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM)(decoder_input)
dec_emb = dec_emb + positional_encoding(MAX_LEN, EMBED_DIM)
dec_self_attn = tf.keras.layers.MultiHeadAttention(
    num_heads=NUM_HEADS, key_dim=EMBED_DIM // NUM_HEADS
)(dec_emb, dec_emb)
dec_norm1 = tf.keras.layers.LayerNormalization()(dec_emb + dec_self_attn)
dec_cross_attn = tf.keras.layers.MultiHeadAttention(
    num_heads=NUM_HEADS, key_dim=EMBED_DIM // NUM_HEADS
)(dec_norm1, encoder_output)
dec_norm2 = tf.keras.layers.LayerNormalization()(dec_norm1 + dec_cross_attn)
dec_ff = tf.keras.layers.Dense(FF_DIM, activation='relu')(dec_norm2)
dec_ff = tf.keras.layers.Dense(EMBED_DIM)(dec_ff)
dec_output = tf.keras.layers.LayerNormalization()(dec_norm2 + dec_ff)
output = tf.keras.layers.Dense(VOCAB_SIZE, activation='softmax')(dec_output)

chatbot_model = tf.keras.Model(
    inputs=[encoder_input, decoder_input], outputs=output
)
chatbot_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
chatbot_model.summary()

# --- Step 4: Inference (Greedy Decoding) ---
def generate_response(input_text, max_len=MAX_LEN):
    enc_input = encoder_vectorizer([input_text])
    dec_input = decoder_vectorizer(["<start>"])
    
    for i in range(max_len):
        predictions = chatbot_model.predict([enc_input, dec_input], verbose=0)
        next_token = tf.argmax(predictions[0, i, :]).numpy()
        vocab = decoder_vectorizer.get_vocabulary()
        if vocab[next_token] == "<end>":
            break
        # Update decoder input with predicted token
    return "Generated response"
```

### Interview Tips
- Modern chatbots use **pre-trained Transformers** (BERT, GPT) fine-tuned on dialogue data
- For production, consider **TF Hub** models or **HuggingFace Transformers** with TF backend
- Key challenges: context tracking, handling out-of-vocabulary words, response diversity
- Retrieval-based chatbots (similarity search) are simpler and often more reliable than generative ones

---

## Question 13
- [ ] Done

**Describe how you would use TensorFlow in an autonomous vehicle perception system.**

### Answer

**Definition**: An autonomous vehicle perception system uses deep learning to **interpret sensor data** (cameras, LiDAR, radar) to detect objects, track motion, and understand the driving environment.

### Perception Pipeline Components

| Component | Task | TensorFlow Model |
|-----------|------|-----------------|
| **Object Detection** | Detect cars, pedestrians, signs | EfficientDet, SSD, Faster R-CNN |
| **Semantic Segmentation** | Pixel-level scene understanding | DeepLab, U-Net |
| **Depth Estimation** | Estimate distance from camera | Monocular depth networks |
| **Lane Detection** | Identify lane boundaries | Custom CNN/Transformer |
| **3D Object Detection** | Detect objects in LiDAR point clouds | PointPillars, VoxelNet |
| **Sensor Fusion** | Combine camera + LiDAR + radar | Multi-input Functional API |
| **Tracking** | Track objects across frames | SORT, DeepSORT |

### Python Code Example
```python
import tensorflow as tf

# --- 1. Object Detection with TF Hub ---
# import tensorflow_hub as hub
# detector = hub.load(
#     "https://tfhub.dev/tensorflow/efficientdet/d4/1")
# detections = detector(image_tensor)
# boxes, scores, classes = (detections['detection_boxes'],
#                           detections['detection_scores'],
#                           detections['detection_classes'])

# --- 2. Custom Multi-Task Perception Model ---
# Shared backbone (feature extractor)
backbone = tf.keras.applications.EfficientNetB3(
    include_top=False, weights='imagenet',
    input_shape=(512, 512, 3)
)
backbone.trainable = False  # Freeze for transfer learning

input_image = tf.keras.Input(shape=(512, 512, 3))
features = backbone(input_image)

# Task 1: Object Detection Head
det_x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(features)
det_x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(det_x)
detection_output = tf.keras.layers.Conv2D(
    5 * 9, 1, name='detection'  # 5 = (x, y, w, h, conf) * 9 anchors
)(det_x)

# Task 2: Semantic Segmentation Head
seg_x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same',
                                         activation='relu')(features)
seg_x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same',
                                         activation='relu')(seg_x)
segmentation_output = tf.keras.layers.Conv2D(
    20, 1, activation='softmax', name='segmentation'  # 20 classes
)(seg_x)

# Task 3: Depth Estimation Head
depth_x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same',
                                           activation='relu')(features)
depth_output = tf.keras.layers.Conv2D(
    1, 1, activation='sigmoid', name='depth'
)(depth_x)

# Multi-task model
perception_model = tf.keras.Model(
    inputs=input_image,
    outputs={
        'detection': detection_output,
        'segmentation': segmentation_output,
        'depth': depth_output
    }
)

# Compile with task-specific losses
perception_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={
        'detection': tf.keras.losses.Huber(),
        'segmentation': 'sparse_categorical_crossentropy',
        'depth': 'mse'
    },
    loss_weights={'detection': 1.0, 'segmentation': 0.5, 'depth': 0.3}
)

# --- 3. Sensor Fusion (Camera + LiDAR) ---
camera_input = tf.keras.Input(shape=(512, 512, 3), name='camera')
lidar_input = tf.keras.Input(shape=(100000, 4), name='lidar')  # x, y, z, intensity

# Camera branch
cam_features = tf.keras.applications.MobileNetV2(
    include_top=False, input_shape=(512, 512, 3)
)(camera_input)
cam_flat = tf.keras.layers.GlobalAveragePooling2D()(cam_features)

# LiDAR branch (PointNet-style)
lidar_features = tf.keras.layers.Conv1D(64, 1, activation='relu')(lidar_input)
lidar_features = tf.keras.layers.Conv1D(128, 1, activation='relu')(lidar_features)
lidar_flat = tf.keras.layers.GlobalMaxPooling1D()(lidar_features)

# Fusion
fused = tf.keras.layers.concatenate([cam_flat, lidar_flat])
fused = tf.keras.layers.Dense(256, activation='relu')(fused)
fusion_output = tf.keras.layers.Dense(100 * 7)(fused)  # 100 objects x 7 params

fusion_model = tf.keras.Model(
    inputs=[camera_input, lidar_input],
    outputs=fusion_output
)

# --- 4. TFLite for Edge Deployment ---
# Convert for in-car inference (e.g., NVIDIA Jetson)
converter = tf.lite.TFLiteConverter.from_keras_model(perception_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_model = converter.convert()
```

### Key Considerations

| Aspect | Requirement |
|--------|-------------|
| **Latency** | < 100ms inference for real-time (10+ FPS) |
| **Reliability** | Redundant models, fail-safe mechanisms |
| **Edge Deployment** | TFLite on NVIDIA Jetson, Google Coral |
| **Data** | Millions of annotated driving scenes |
| **Safety** | Extensive testing, adversarial robustness |

### Interview Tips
- **Multi-task learning** (shared backbone) is key for efficiency — one forward pass for detection + segmentation + depth
- **Sensor fusion** combining camera + LiDAR is standard in L4/L5 autonomy
- Models must be **quantized and optimized** for real-time edge inference
- Safety-critical systems require **formal verification**, extensive testing, and redundancy
- TensorFlow's ecosystem (TF Lite, TF Serving, TF Hub) covers the full AV pipeline from training to deployment

---

---

