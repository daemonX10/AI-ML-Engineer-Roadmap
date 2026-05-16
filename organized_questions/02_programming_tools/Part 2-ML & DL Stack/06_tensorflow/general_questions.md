# TensorFlow Interview Questions - General Questions

## Question 1
- [ ] Done

**What types of devices does TensorFlow support for computation?**

### Answer

### Supported Devices

| Device | Description | Use Case |
|--------|-------------|----------|
| **CPU** | Default, available everywhere | Small models, debugging |
| **GPU** | NVIDIA CUDA support | Training deep learning models |
| **TPU** | Google's custom hardware | Large-scale training |
| **Mobile** | TFLite on Android/iOS | On-device inference |
| **Browser** | TensorFlow.js | Web-based ML |
| **Microcontrollers** | TFLite Micro | Edge devices |

### Python Code Example
```python
import tensorflow as tf

# List available devices
devices = tf.config.list_physical_devices()
print("Available devices:", devices)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")

# Place operations on specific device
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)

# Enable memory growth (prevents TF from taking all GPU memory)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

---

## Question 2
- [ ] Done

**How do you build a neural network in TensorFlow ?**

### Answer

### Methods to Build Models

| Method | Use Case |
|--------|----------|
| **Sequential** | Linear stack of layers |
| **Functional** | Complex architectures (multi-input/output) |
| **Subclassing** | Custom behavior |

### Python Code Example
```python
import tensorflow as tf

# Method 1: Sequential API (simplest)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
# model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Summary
model.summary()
```

### Functional API (for complex models)
```python
# Method 2: Functional API
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

---

## Question 3
- [ ] Done

**What is tf.function and why is it important?**

### Answer

**Definition**: `tf.function` is a decorator that converts Python functions into TensorFlow graphs for better performance.

### Benefits

| Benefit | Description |
|---------|-------------|
| **Speed** | Graph optimization and compilation |
| **Portability** | Export graphs without Python |
| **Parallelism** | Operations can run in parallel |

### Python Code Example
```python
import tensorflow as tf
import time

# Without tf.function (eager mode)
def eager_function(x):
    return x ** 2 + 2 * x + 1

# With tf.function (graph mode)
@tf.function
def graph_function(x):
    return x ** 2 + 2 * x + 1

# Benchmark
x = tf.random.normal([1000, 1000])

# Eager execution
start = time.time()
for _ in range(100):
    eager_function(x)
eager_time = time.time() - start

# Graph execution
start = time.time()
for _ in range(100):
    graph_function(x)
graph_time = time.time() - start

print(f"Eager: {eager_time:.4f}s")
print(f"Graph: {graph_time:.4f}s")
print(f"Speedup: {eager_time/graph_time:.2f}x")
```

### When to Use
- Production inference
- Training loops that run many times
- Exporting models

---

## Question 4
- [ ] Done

**How do you save and load models in TensorFlow?**

### Answer

### Saving Methods

| Method | Format | Use Case |
|--------|--------|----------|
| `model.save()` | SavedModel or H5 | Complete model |
| `model.save_weights()` | Checkpoint | Weights only |
| `tf.saved_model.save()` | SavedModel | Production deployment |

### Python Code Example
```python
import tensorflow as tf

# Create a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Method 1: Save entire model (recommended)
model.save('my_model')  # SavedModel format (directory)
model.save('my_model.h5')  # HDF5 format (single file)

# Load entire model
loaded_model = tf.keras.models.load_model('my_model')

# Method 2: Save weights only
model.save_weights('weights/my_weights')
model.load_weights('weights/my_weights')

# Method 3: Checkpoints during training
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/epoch_{epoch:02d}',
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss'
)
# model.fit(..., callbacks=[checkpoint])
```

---

## Question 5
- [ ] Done

**What is TensorBoard and how do you use it?**

### Answer

**Definition**: TensorBoard is TensorFlow's visualization toolkit for monitoring training, viewing model graphs, and analyzing performance.

### Features

| Feature | Description |
|---------|-------------|
| **Scalars** | Track loss, accuracy over time |
| **Graphs** | Visualize model architecture |
| **Histograms** | Weight distributions |
| **Images** | View input/output images |
| **Profiler** | Performance analysis |

### Python Code Example
```python
import tensorflow as tf
import datetime

# Create log directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,  # Log weight histograms
    write_graph=True,  # Log model graph
    write_images=True  # Log weight images
)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train with TensorBoard
# model.fit(X_train, y_train, epochs=10, callbacks=[tensorboard_callback])

# Custom logging with tf.summary
writer = tf.summary.create_file_writer(log_dir)
with writer.as_default():
    tf.summary.scalar('custom_metric', 0.5, step=1)
```

### Launch TensorBoard
```bash
tensorboard --logdir logs/fit
# Open http://localhost:6006 in browser
```

---

## Question 6
- [ ] Done

**How do you handle datasets in TensorFlow?**

### Answer

**Definition**: `tf.data.Dataset` is TensorFlow's API for building efficient input pipelines.

### Key Methods

| Method | Description |
|--------|-------------|
| `from_tensor_slices()` | Create from NumPy/tensors |
| `batch()` | Group into batches |
| `shuffle()` | Randomize order |
| `prefetch()` | Load data while training |
| `map()` | Apply transformations |

### Python Code Example
```python
import tensorflow as tf
import numpy as np

# Create dataset from arrays
X = np.random.randn(1000, 10).astype(np.float32)
y = np.random.randint(0, 2, 1000).astype(np.float32)

dataset = tf.data.Dataset.from_tensor_slices((X, y))

# Build efficient pipeline
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Overlap data loading with training

# Use in training
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
# model.fit(dataset, epochs=5)

# Data augmentation with map
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    return image, label

# image_dataset = image_dataset.map(augment)
```

---

## Question 7
- [ ] Done

**What are callbacks in TensorFlow/Keras?**

### Answer

**Definition**: Callbacks are functions called at specific points during training to extend functionality.

### Common Callbacks

| Callback | Purpose |
|----------|---------|
| `ModelCheckpoint` | Save model during training |
| `EarlyStopping` | Stop when metric stops improving |
| `TensorBoard` | Logging for visualization |
| `ReduceLROnPlateau` | Reduce learning rate |
| `LearningRateScheduler` | Custom LR schedule |

### Python Code Example
```python
import tensorflow as tf

# Checkpoint: save best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Early stopping: prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Reduce learning rate when plateau
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# TensorBoard logging
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs')

# Use in training
callbacks = [checkpoint, early_stop, reduce_lr, tensorboard]
# model.fit(X_train, y_train, validation_split=0.2, 
#           epochs=100, callbacks=callbacks)
```

### Custom Callback
```python
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy, stopping!")
            self.model.stop_training = True
```

---

## Question 8
- [ ] Done

**How do you implement transfer learning in TensorFlow?**

### Answer

**Definition**: Transfer learning uses a pre-trained model as a starting point for a new task.

### Steps
1. Load pre-trained model (without top layers)
2. Freeze base layers
3. Add new classification layers
4. Train on new data

### Python Code Example
```python
import tensorflow as tf

# Load pre-trained model (without classification layers)
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,  # Remove classification head
    input_shape=(224, 224, 3)
)

# Freeze base model layers
base_model.trainable = False

# Build new model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 classes
])

# Compile with low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tuning (optional): unfreeze some layers
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Freeze all but last 20
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Question 9
- [ ] Done

**What is distributed training in TensorFlow?**

### Answer

**Definition**: Distributed training spreads model training across multiple GPUs or machines for faster training.

### Strategies

| Strategy | Description |
|----------|-------------|
| `MirroredStrategy` | Multiple GPUs, single machine |
| `MultiWorkerMirroredStrategy` | Multiple machines |
| `TPUStrategy` | Google TPUs |
| `ParameterServerStrategy` | Large-scale distributed |

### Python Code Example
```python
import tensorflow as tf

# Strategy: Single machine, multiple GPUs
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Build model within strategy scope
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer='adam',
        loss='mse'
    )

# Train as usual - distribution happens automatically
# model.fit(X_train, y_train, epochs=10, batch_size=64*strategy.num_replicas_in_sync)

# For TPU
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.TPUStrategy(resolver)
```

### Key Points
- Batch size scales with number of devices
- Data is automatically sharded
- Gradients are synchronized

---

## Question 10
- [ ] Done

**How do you handle overfitting in TensorFlow models?**

### Answer

### Techniques

| Technique | Implementation |
|-----------|----------------|
| **Dropout** | `tf.keras.layers.Dropout` |
| **L1/L2 Regularization** | `kernel_regularizer` |
| **Early Stopping** | Callback |
| **Data Augmentation** | `tf.image` transforms |
| **Batch Normalization** | `tf.keras.layers.BatchNormalization` |

### Python Code Example
```python
import tensorflow as tf

# Model with regularization techniques
model = tf.keras.Sequential([
    # L2 regularization on weights
    tf.keras.layers.Dense(128, activation='relu', 
                          kernel_regularizer=tf.keras.regularizers.l2(0.01),
                          input_shape=(10,)),
    
    # Batch normalization
    tf.keras.layers.BatchNormalization(),
    
    # Dropout
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(64, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation for images
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])
```

### Summary
- Use **dropout** (0.2-0.5) after dense layers
- Add **L2 regularization** for weight penalty
- Implement **early stopping** to prevent overtraining
- Use **data augmentation** to increase training variety


---

## Question 11
- [ ] Done

**Define a Variable in TensorFlow and its importance.**

**Answer:**

A **`tf.Variable`** is a mutable tensor whose value persists and can be updated during training. It represents **learnable parameters** (weights and biases).

```python
import tensorflow as tf

# Creating Variables
weights = tf.Variable(tf.random.normal([3, 2]), name='weights')  # Trainable
bias = tf.Variable(tf.zeros([2]), name='bias')

# Properties
print(weights.shape)       # (3, 2)
print(weights.dtype)       # float32
print(weights.trainable)   # True (included in gradient computation)

# Updating values
weights.assign(tf.ones([3, 2]))         # Replace entirely
weights.assign_add(tf.ones([3, 2]))     # Add to current value
weights.assign_sub(tf.ones([3, 2]))     # Subtract from current

# Non-trainable variable (e.g., step counter)
step = tf.Variable(0, trainable=False, name='global_step')
step.assign_add(1)

# Variable vs Constant
constant = tf.constant([1, 2, 3])       # Immutable
variable = tf.Variable([1, 2, 3])       # Mutable
# constant.assign([4, 5, 6])  # ERROR!
variable.assign([4, 5, 6])   # OK
```

| Feature | `tf.Variable` | `tf.constant` |
|---------|---------------|----------------|
| Mutable | Yes (`assign`) | No |
| Trainable | Yes (by default) | No |
| In gradients | Yes | No |
| Memory | Persists in memory | Can be optimized away |
| Use case | Weights, biases | Input data, hyperparameters |

> **Interview Tip:** Variables are the **trainable parameters** of a model. During backpropagation, `GradientTape` computes gradients with respect to Variables, and the optimizer updates them.

---

## Question 12
- [ ] Done

**How do you perform batch processing in TensorFlow?**

**Answer:**

```python
import tensorflow as tf
import numpy as np

# === Using tf.data.Dataset (recommended) ===

# From NumPy arrays
X = np.random.rand(10000, 28, 28)
y = np.random.randint(0, 10, 10000)

dataset = tf.data.Dataset.from_tensor_slices((X, y))

# Create batched pipeline
train_ds = (
    dataset
    .shuffle(buffer_size=1000)  # Randomize order
    .batch(32)                  # Create batches of 32
    .prefetch(tf.data.AUTOTUNE) # Prefetch next batch during training
)

# Training loop
model = tf.keras.Sequential([...])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(train_ds, epochs=10)  # Automatically iterates batches

# === Manual batch iteration ===
for batch_x, batch_y in train_ds:
    print(f"Batch shape: {batch_x.shape}")  # (32, 28, 28)
    # Custom training step...

# === Advanced batching ===
# Padded batching (variable-length sequences)
ds = tf.data.Dataset.from_generator(
    lambda: iter(sequences),
    output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32)
)
padded_ds = ds.padded_batch(32, padded_shapes=[100])  # Pad to length 100

# Drop remainder (for consistent batch size)
train_ds = dataset.batch(32, drop_remainder=True)

# Repeat for multiple epochs
train_ds = dataset.repeat(10).shuffle(1000).batch(32)
```

| Method | Description |
|--------|------------|
| `.batch(N)` | Group N samples into a batch |
| `.shuffle(buffer)` | Randomize with buffer |
| `.prefetch(AUTOTUNE)` | Overlap data loading & training |
| `.repeat(N)` | Repeat dataset N times |
| `.padded_batch()` | Batch with padding (NLP) |
| `.cache()` | Cache in memory/disk |

> **Interview Tip:** Always use `prefetch(tf.data.AUTOTUNE)` for performance — it loads the next batch while the GPU processes the current one. Use `shuffle` before `batch` to ensure random batches.

---

## Question 13
- [ ] Done

**How do you use TensorFlow Transformers for sequence modeling?**

**Answer:**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# === Multi-Head Self-Attention Layer ===
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)        # Self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)        # Residual + LayerNorm
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)           # Residual + LayerNorm

# === Positional Encoding ===
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.pos_embedding = layers.Embedding(max_len, embed_dim)
    
    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        return x + self.pos_embedding(positions)

# === Full Transformer Model (Text Classification) ===
def build_transformer(vocab_size=10000, max_len=200, embed_dim=128, 
                      num_heads=4, ff_dim=128, num_classes=2):
    inputs = layers.Input(shape=(max_len,))
    x = layers.Embedding(vocab_size, embed_dim)(inputs)
    x = PositionalEncoding(max_len, embed_dim)(x)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x, training=True)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)

model = build_transformer()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

> **Interview Tip:** The Transformer key components are: **multi-head self-attention**, **positional encoding**, **residual connections**, and **layer normalization**. For production NLP, use **HuggingFace Transformers** with TF backend.

---

## Question 14
- [ ] Done

**How do you approach optimizing TensorFlow model performance?**

**Answer:**

### Optimization Strategies

| Level | Technique | Impact |
|-------|-----------|--------|
| **Data pipeline** | `prefetch`, `cache`, `parallel map` | 2-5× throughput |
| **Model architecture** | Reduce parameters, efficient layers | Varies |
| **Training** | Mixed precision, XLA | 1.5-3× speed |
| **Hardware** | Multi-GPU, TPU | Linear scaling |

```python
import tensorflow as tf

# === 1. Data Pipeline Optimization ===
dataset = (
    tf.data.Dataset.from_tensor_slices((X, y))
    .cache()                              # Cache after first epoch
    .shuffle(10000)
    .batch(64)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)  # Parallel preprocessing
    .prefetch(tf.data.AUTOTUNE)           # Overlap CPU/GPU
)

# === 2. Mixed Precision Training ===
tf.keras.mixed_precision.set_global_policy('mixed_float16')
# Model uses float16 for compute, float32 for accumulation
# ~2× speedup on GPUs with Tensor Cores

# === 3. XLA Compilation ===
@tf.function(jit_compile=True)  # XLA optimization
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# === 4. Model Optimization ===
# Pruning
import tensorflow_model_optimization as tfmot
model = tfmot.sparsity.keras.prune_low_magnitude(model)

# Quantization (for deployment)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Dynamic quantization
tflite_model = converter.convert()

# === 5. Multi-GPU Training ===
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```

> **Interview Tip:** Optimization priority: 1) Fix data bottlenecks (`prefetch`), 2) Mixed precision, 3) XLA compilation, 4) Model pruning/quantization. Use **TensorBoard Profiler** to identify bottlenecks.

---

## Question 15
- [ ] Done

**What techniques are used in TensorFlow for graph optimizations?**

**Answer:**

TensorFlow's computational graph can be optimized at multiple levels.

### Key Graph Optimization Techniques

```python
import tensorflow as tf

# === 1. tf.function — Graph Tracing ===
@tf.function  # Converts eager code to optimized graph
def compute(x):
    return tf.reduce_sum(x ** 2 + 2 * x + 1)

# First call traces the function; subsequent calls reuse the graph
result = compute(tf.constant([1.0, 2.0, 3.0]))

# === 2. XLA (Accelerated Linear Algebra) ===
@tf.function(jit_compile=True)  # XLA compilation
def xla_compute(x, y):
    return tf.matmul(x, y) + tf.reduce_sum(x)

# === 3. Grappler Optimizations (automatic) ===
# TensorFlow's Grappler applies these automatically:
```

| Optimization | What It Does |
|--------------|-------------|
| **Constant folding** | Pre-computes constant expressions |
| **Common subexpression elimination** | Reuses identical computations |
| **Operator fusion** | Merges compatible ops (e.g., Conv+BN+ReLU) |
| **Dead code elimination** | Removes unused operations |
| **Layout optimization** | Converts NHWC ↔ NCHW for GPU |
| **Memory optimization** | Schedules ops to minimize peak memory |
| **Arithmetic simplification** | e.g., x * 1 → x |

```python
# === 4. Custom Graph Optimization Config ===
from tensorflow.core.protobuf import config_pb2

config = tf.compat.v1.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = (
    config_pb2.OptimizerOptions.ON_2  # Aggressive XLA
)

# === 5. SavedModel Optimization ===
# Convert to TFLite with optimizations
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
```

> **Interview Tip:** `tf.function` is the primary mechanism for graph optimization. Grappler handles most optimizations automatically. Use `jit_compile=True` for XLA when training on GPUs/TPUs.

---

## Question 16
- [ ] Done

**How do you perform memory optimization in TensorFlow?**

**Answer:**

```python
import tensorflow as tf

# === 1. GPU Memory Growth ===
# Prevent TF from allocating all GPU memory at once
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Or set a hard limit
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB
)

# === 2. Mixed Precision (halves memory for activations) ===
tf.keras.mixed_precision.set_global_policy('mixed_float16')
# float32 weights: 4 bytes/param  →  float16: 2 bytes/param

# === 3. Gradient Checkpointing (trade compute for memory) ===
# Recompute activations during backprop instead of storing them
model.compile(optimizer='adam', loss='mse')
# Use tf.recompute_grad for custom layers

# === 4. Efficient Data Pipeline ===
dataset = (
    tf.data.Dataset.from_generator(data_generator, ...)
    .batch(32)                    # Don't load all data at once
    .prefetch(tf.data.AUTOTUNE)
)

# === 5. Reduce Batch Size + Gradient Accumulation ===
accumulation_steps = 4
for step, (x, y) in enumerate(dataset):
    with tf.GradientTape() as tape:
        loss = model(x, training=True)
        scaled_loss = loss / accumulation_steps
    grads = tape.gradient(scaled_loss, model.trainable_variables)
    if (step + 1) % accumulation_steps == 0:
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

# === 6. Model Architecture ===
# Use depth-wise separable convolutions
layers.SeparableConv2D(64, 3)  # ~9× fewer params than Conv2D

# Use Global Average Pooling instead of Flatten
layers.GlobalAveragePooling2D()  # Much less memory than Flatten
```

| Technique | Memory Saved | Trade-off |
|-----------|-------------|----------|
| Memory growth | Prevents waste | None |
| Mixed precision | ~50% | Minimal accuracy impact |
| Gradient checkpointing | ~60-70% | ~20% slower |
| Smaller batch size | Linear | May need LR adjustment |
| Depthwise separable conv | ~9× fewer params | Slight accuracy drop |

> **Interview Tip:** GPU OOM is the most common TF error. First try: enable memory growth, reduce batch size, use mixed precision. For very large models: gradient checkpointing and model parallelism.

---

## Question 17
- [ ] Done

**How do you handle image data in TensorFlow?**

**Answer:**

```python
import tensorflow as tf
from tensorflow.keras import layers

# === 1. Loading Images ===

# From directory (most common)
train_ds = tf.keras.utils.image_dataset_from_directory(
    'data/train/',
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical',     # 'int', 'categorical', 'binary'
    validation_split=0.2,
    subset='training',
    seed=42
)

# Single image
img = tf.io.read_file('image.jpg')
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.resize(img, [224, 224])
img = img / 255.0  # Normalize

# === 2. Data Augmentation ===
augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.1, 0.1)
])

# === 3. Preprocessing ===
# For pretrained models
preprocess = tf.keras.applications.resnet50.preprocess_input  # Model-specific

# === 4. Complete Pipeline ===
def build_image_pipeline(directory, batch_size=32, img_size=(224, 224)):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory, image_size=img_size, batch_size=batch_size
    )
    ds = ds.map(lambda x, y: (augmentation(x, training=True), y))
    ds = ds.cache().prefetch(tf.data.AUTOTUNE)
    return ds

# === 5. Transfer Learning ===
base_model = tf.keras.applications.ResNet50(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
```

> **Interview Tip:** Use `image_dataset_from_directory` for quick loading, `tf.data.Dataset` for custom pipelines. Always use **data augmentation** to prevent overfitting and **transfer learning** for small datasets.

---

## Question 18
- [ ] Done

**How is TensorFlow deployed in mobile or edge devices?**

**Answer:**

### TensorFlow Lite (TFLite)

```python
import tensorflow as tf

# === 1. Convert Model to TFLite ===
model = tf.keras.models.load_model('my_model.h5')

# Basic conversion
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# === 2. Optimized Conversion ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Dynamic range quantization (~4× smaller, ~2-3× faster)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Full integer quantization (~4× smaller, ~3-4× faster)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

tflite_quantized = converter.convert()

# === 3. Inference on Device ===
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
```

### Deployment Options

| Platform | Tool | Language |
|----------|------|----------|
| Android | TFLite + Android SDK | Java/Kotlin |
| iOS | TFLite + Core ML | Swift |
| Raspberry Pi | TFLite runtime | Python/C++ |
| Microcontrollers | TFLite Micro | C/C++ |
| Web browser | TensorFlow.js | JavaScript |
| Edge TPU | Coral / Edge TPU | Python |

| Optimization | Size Reduction | Speed Improvement |
|-------------|---------------|-------------------|
| Dynamic quantization | ~4× | ~2-3× |
| Full int8 quantization | ~4× | ~3-4× |
| Float16 quantization | ~2× | ~1.5-2× |
| Pruning + quantization | ~10× | ~4-5× |

> **Interview Tip:** The typical pipeline: **Train (TF/Keras)** → **Convert (TFLite)** → **Optimize (quantize)** → **Deploy (mobile/edge)**. TFLite models are typically 4× smaller and 2-3× faster than full TF models.

---

## Question 19
- [ ] Done

**Can you give an example of how TensorFlow is used in healthcare?**

**Answer:**

### Healthcare Applications

| Application | Model Type | Example |
|------------|------------|---------|
| **Medical imaging** | CNN (Transfer Learning) | Detecting tumors in X-rays/CT scans |
| **Drug discovery** | GNN, RNN | Predicting molecular properties |
| **EHR analysis** | LSTM, Transformer | Patient outcome prediction |
| **Pathology** | CNN | Cancer cell classification |
| **Genomics** | 1D CNN, Attention | Gene expression analysis |
| **Wearables** | TFLite + LSTM | Real-time heart rate anomaly detection |

### Example: Medical Image Classification

```python
import tensorflow as tf
from tensorflow.keras import layers

# Transfer learning for chest X-ray classification
base_model = tf.keras.applications.DenseNet121(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')   # Binary: normal vs pneumonia
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Data pipeline with augmentation
train_ds = tf.keras.utils.image_dataset_from_directory(
    'chest_xray/train/', image_size=(224, 224), batch_size=32
)

# Class weights for imbalanced medical data
class_weights = {0: 1.0, 1: 3.0}  # Weight pneumonia cases higher
model.fit(train_ds, epochs=20, class_weight=class_weights)
```

### Real-World TF Healthcare Projects

| Project | Organization | Task |
|---------|-------------|------|
| DeepMind AlphaFold | Google | Protein structure prediction |
| CheXNet | Stanford | Chest X-ray diagnosis |
| Retinal screening | Google Health | Diabetic retinopathy detection |
| COVID-19 detection | Various | CT scan classification |

> **Interview Tip:** Healthcare ML requires: 1) **Explainability** (Grad-CAM for images), 2) **Class imbalance** handling (class weights, oversampling), 3) **Regulatory compliance** (FDA, HIPAA), 4) **Domain expert validation**.

---

## Question 20
- [ ] Done

**What steps would you take to investigate and fix a shape mismatch error in TensorFlow?**

**Answer:**

### Common Shape Mismatch Errors

```
ValueError: Shapes (32, 10) and (32, 1) are incompatible
InvalidArgumentError: Matrix size-incompatible: [64,128] vs [256,10]
```

### Debugging Steps

```python
import tensorflow as tf

# === Step 1: Print shapes at every layer ===
model.summary()  # Check all layer output shapes

# === Step 2: Add debug prints ===
@tf.function
def train_step(x, y):
    tf.print("Input shape:", tf.shape(x))     # Runtime shape
    tf.print("Label shape:", tf.shape(y))
    predictions = model(x)
    tf.print("Prediction shape:", tf.shape(predictions))
    loss = loss_fn(y, predictions)
    return loss

# === Step 3: Check data pipeline ===
for batch_x, batch_y in dataset.take(1):
    print(f"X: {batch_x.shape}, Y: {batch_y.shape}")
    print(f"X dtype: {batch_x.dtype}, Y dtype: {batch_y.dtype}")
```

### Common Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| `(32,10)` vs `(32,1)` | Loss expects different format | Use `sparse_categorical_crossentropy` or one-hot encode labels |
| `(32,28,28)` vs `(32,784)` | Missing reshape | Add `layers.Flatten()` or `layers.Reshape()` |
| Batch size mismatch | Last batch smaller | `drop_remainder=True` in `.batch()` |
| Dense input shape | 3D tensor into Dense | Add `layers.GlobalAveragePooling1D()` before Dense |
| Conv2D expects 4D | Missing channel dim | `tf.expand_dims(x, -1)` or `layers.Reshape((28,28,1))` |

```python
# Fix: One-hot vs sparse labels
# Option A: Change loss
model.compile(loss='sparse_categorical_crossentropy')  # Labels: [0, 3, 1]
# Option B: One-hot encode
y_onehot = tf.one_hot(y, depth=num_classes)  # Labels: [[1,0,0], [0,0,0,1], [0,1,0]]
model.compile(loss='categorical_crossentropy')

# Fix: Channel dimension
X = X[..., tf.newaxis]  # (28,28) → (28,28,1)
```

> **Interview Tip:** Shape mismatch is the #1 TF debugging issue. Always check: 1) `model.summary()`, 2) Data shapes with `.take(1)`, 3) Loss function matches label format.

---

## Question 21
- [ ] Done

**How is TensorFlow utilized in Natural Language Processing (NLP)?**

**Answer:**

```python
import tensorflow as tf
from tensorflow.keras import layers

# === 1. Text Preprocessing ===
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(train_texts)
sequences = tokenizer.texts_to_sequences(train_texts)
padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=200)

# Modern: TextVectorization layer
vectorizer = layers.TextVectorization(max_tokens=10000, output_sequence_length=200)
vectorizer.adapt(train_texts)

# === 2. Sentiment Analysis (LSTM) ===
model = tf.keras.Sequential([
    layers.Embedding(vocab_size=10000, output_dim=128, input_length=200),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# === 3. Text Classification (CNN) ===
model = tf.keras.Sequential([
    layers.Embedding(10000, 128),
    layers.Conv1D(128, 5, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# === 4. Seq2Seq (Machine Translation) ===
encoder = tf.keras.Sequential([
    layers.Embedding(src_vocab, 256),
    layers.LSTM(512, return_state=True)
])

# === 5. Pretrained Models (HuggingFace + TF) ===
from transformers import TFAutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = TFAutoModel.from_pretrained('bert-base-uncased')
```

| NLP Task | TF Approach | Model |
|----------|-------------|-------|
| Sentiment analysis | BiLSTM, CNN, BERT | Binary/multi-class |
| Text classification | Embedding + Dense | Multi-class |
| Named entity recognition | BiLSTM-CRF, BERT | Sequence labeling |
| Machine translation | Seq2Seq + Attention | Encoder-Decoder |
| Text generation | GPT-style Transformer | Autoregressive |
| Question answering | BERT fine-tuning | Extractive QA |

> **Interview Tip:** For modern NLP, use **pretrained Transformers** (BERT, GPT) via HuggingFace with TF backend. LSTMs/CNNs are still relevant for lightweight, on-device NLP.

---

## Question 22
- [ ] Done

**Present an approach for real-time object detection using TensorFlow.**

**Answer:**

### Architecture Options

| Model | Speed (FPS) | Accuracy (mAP) | Use Case |
|-------|-------------|----------------|----------|
| SSD MobileNet | 30-60 | Medium | Mobile/edge |
| EfficientDet | 15-30 | High | Balanced |
| YOLOv5 (via TF) | 30-60 | High | Real-time |
| Faster R-CNN | 5-10 | Highest | Accuracy-first |

```python
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# === 1. Load Pretrained Model ===
detector = hub.load(
    'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2'
)

# === 2. Real-time Detection Function ===
def detect_objects(image, threshold=0.5):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]  # Add batch dim
    
    detections = detector(input_tensor)
    
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)
    
    # Filter by confidence
    mask = scores >= threshold
    return boxes[mask], scores[mask], classes[mask]

# === 3. Real-time Video Pipeline ===
def real_time_detection():
    cap = cv2.VideoCapture(0)  # Webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detect_objects(rgb_frame)
        
        # Draw bounding boxes
        h, w, _ = frame.shape
        for box, score, cls in zip(boxes, scores, classes):
            y1, x1, y2, x2 = (box * [h, w, h, w]).astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Class {cls}: {score:.2f}',
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# === 4. Custom Object Detection (Fine-tuning) ===
# Use TF Object Detection API
# 1. Annotate images (LabelImg, CVAT)
# 2. Convert to TFRecord format
# 3. Fine-tune pretrained model
# 4. Export to TFLite for edge deployment
```

### Optimization for Real-time

| Technique | Impact |
|-----------|--------|
| TFLite conversion | 2-4× faster |
| INT8 quantization | 3-4× faster |
| Input resolution reduction | Linear speedup |
| Edge TPU (Coral) | 10-20× over CPU |
| Frame skipping | 2× throughput |

> **Interview Tip:** For real-time detection: use **SSD MobileNet** (fastest), quantize with TFLite, and deploy on edge devices. For accuracy-first: **EfficientDet** or **Faster R-CNN**.
