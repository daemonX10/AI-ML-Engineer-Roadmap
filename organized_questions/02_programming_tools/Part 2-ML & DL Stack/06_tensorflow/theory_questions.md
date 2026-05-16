# TensorFlow Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What is TensorFlow and who developed it?**

### Answer

**Definition**: TensorFlow is a free, open-source machine learning platform developed by the **Google Brain team**. It was open-sourced in November 2015.

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Computation Graph** | Mathematical operations represented as nodes, data (tensors) flow along edges |
| **Tensor** | Multi-dimensional arrays that flow through the graph |
| **Eager Execution** | Operations execute immediately (TF 2.x default) |
| **End-to-end Platform** | Covers training, deployment, and production ML |

### Python Code Example
```python
import tensorflow as tf

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Basic tensor operations
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
c = tf.add(a, b)  # Element-wise addition

print(f"a + b = {c.numpy()}")  # Output: [5 7 9]
```

---

## Question 2
- [ ] Done

**What are the main features of TensorFlow?**

### Answer

### Key Features

| Feature | Description |
|---------|-------------|
| **Eager Execution** | Operations run immediately, easy debugging |
| **Graph Mode** | `tf.function` converts code to optimized graphs |
| **Auto Differentiation** | `tf.GradientTape` computes gradients automatically |
| **Keras Integration** | High-level API for building models |
| **Multi-Platform** | CPU, GPU, TPU, mobile, browser support |
| **Distributed Training** | `tf.distribute.Strategy` for scaling |
| **TensorBoard** | Visualization tools for training |

### Python Code Example
```python
import tensorflow as tf

# Feature 1: Eager execution (default in TF 2.x)
x = tf.constant([[1, 2], [3, 4]])
print(x.numpy())  # Executes immediately

# Feature 2: Graph mode for optimization
@tf.function
def fast_multiply(a, b):
    return tf.matmul(a, b)

# Feature 3: Auto differentiation
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x ** 2
grad = tape.gradient(y, x)  # dy/dx = 2x = 6
print(f"Gradient: {grad.numpy()}")
```

---

## Question 3
- [ ] Done

**Can you explain the concept of a computation graph in TensorFlow?**

### Answer

**Definition**: A computation graph is a directed acyclic graph (DAG) where:
- **Nodes** = Operations (add, multiply, layers)
- **Edges** = Tensors (data flowing between operations)

### Components

| Component | Role | Example |
|-----------|------|---------|
| **Node** | Represents operation | `tf.add`, `tf.matmul` |
| **Edge** | Data flow (tensor) | Weights, activations |
| **Input** | Entry point | Training data |
| **Output** | Result | Predictions |

### Python Code Example
```python
import tensorflow as tf

# Graph mode with tf.function
@tf.function
def compute_graph(a, b):
    """This function gets converted to a computation graph"""
    c = tf.add(a, b)      # Node: addition
    d = tf.multiply(c, 2)  # Node: multiplication
    return d              # Edges: tensors flow between nodes

# Execute
x = tf.constant(3.0)
y = tf.constant(4.0)
result = compute_graph(x, y)
print(f"Result: {result.numpy()}")  # (3+4)*2 = 14

# View graph structure
print(compute_graph.get_concrete_function(x, y).graph.as_graph_def())
```

### Advantages
- **Optimization**: Graph analyzed before execution
- **Portability**: Export to different platforms
- **Parallelism**: Operations can run in parallel

---

## Question 4
- [ ] Done

**What are Tensors in TensorFlow?**

### Answer

**Definition**: A Tensor is a multi-dimensional array with a uniform data type.

### Tensor Ranks

| Rank | Name | Example | Shape |
|------|------|---------|-------|
| 0 | Scalar | `5` | `()` |
| 1 | Vector | `[1, 2, 3]` | `(3,)` |
| 2 | Matrix | `[[1,2], [3,4]]` | `(2, 2)` |
| 3 | 3D Tensor | Batch of images | `(batch, height, width)` |
| 4 | 4D Tensor | Color images | `(batch, H, W, channels)` |

### Python Code Example
```python
import tensorflow as tf

# Scalar (rank 0)
scalar = tf.constant(5)
print(f"Scalar shape: {scalar.shape}")  # ()

# Vector (rank 1)
vector = tf.constant([1, 2, 3])
print(f"Vector shape: {vector.shape}")  # (3,)

# Matrix (rank 2)
matrix = tf.constant([[1, 2], [3, 4]])
print(f"Matrix shape: {matrix.shape}")  # (2, 2)

# 3D Tensor
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"3D Tensor shape: {tensor_3d.shape}")  # (2, 2, 2)

# Key properties
print(f"dtype: {matrix.dtype}")  # int32
print(f"rank: {tf.rank(matrix).numpy()}")  # 2

# Variable (mutable tensor for model parameters)
var = tf.Variable([1.0, 2.0, 3.0])
var.assign([4.0, 5.0, 6.0])  # Can change values
```

---

## Question 5
- [ ] Done

**How does TensorFlow differ from other Machine Learning libraries?**

### Answer

### Comparison Table

| Feature | TensorFlow | PyTorch | Scikit-learn |
|---------|------------|---------|--------------|
| **Focus** | Production ML | Research | Classical ML |
| **Execution** | Eager + Graph | Eager | Imperative |
| **API** | Keras (high-level) | Explicit, Pythonic | fit/predict |
| **Deployment** | TF Serving, TFLite | TorchServe | Pickle |
| **Distributed** | `tf.distribute` | DDP | Limited |
| **Debugging** | Good (Eager mode) | Excellent | Simple |

### When to Use

```python
# TensorFlow: Production + Deep Learning
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Scikit-learn: Classical ML
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
```

### Summary
- **TensorFlow**: Best for production deployment, scalable systems
- **PyTorch**: Best for research, rapid prototyping
- **Scikit-learn**: Best for classical ML algorithms

---

## Question 6
- [ ] Done

**What is a Session in TensorFlow?**

### Answer

**Definition**: `tf.Session` was a TensorFlow 1.x concept for executing computation graphs. **Deprecated in TF 2.x**.

### TF 1.x vs TF 2.x

| Aspect | TF 1.x (Session) | TF 2.x (Eager) |
|--------|-----------------|----------------|
| **Execution** | Deferred (build graph first) | Immediate |
| **Syntax** | `sess.run()` | Direct function call |
| **Debugging** | Difficult | Easy |
| **Data input** | `feed_dict` | Direct arguments |

### Code Comparison
```python
# TensorFlow 1.x style (legacy)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.add(a, b)

with tf.Session() as sess:
    result = sess.run(c, feed_dict={a: 5.0, b: 3.0})
    print(result)  # 8.0

# TensorFlow 2.x style (modern)
import tensorflow as tf

a = tf.constant(5.0)
b = tf.constant(3.0)
c = tf.add(a, b)
print(c.numpy())  # 8.0 - executes immediately!
```

---

## Question 7
- [ ] Done

**What is the difference between TensorFlow 1.x and TensorFlow 2.x?**

### Answer

### Key Differences

| Feature | TensorFlow 1.x | TensorFlow 2.x |
|---------|---------------|----------------|
| **Default Mode** | Graph execution | Eager execution |
| **Sessions** | Required | Not needed |
| **Placeholders** | Required for input | Not needed |
| **API** | Multiple competing APIs | Keras is standard |
| **Control Flow** | `tf.cond`, `tf.while_loop` | Python if/for |
| **Performance** | Graph mode | `tf.function` decorator |

### Python Code Example
```python
import tensorflow as tf

# TF 2.x: Clean, Pythonic code
# No sessions, no placeholders!

# Simple computation
x = tf.constant([1.0, 2.0, 3.0])
y = x * 2 + 1
print(y.numpy())  # Immediate result

# Control flow with Python
def compute(x):
    if x > 0:  # Regular Python if
        return x * 2
    else:
        return x

# Convert to graph for performance
@tf.function
def fast_compute(x):
    return x * 2 + 1

result = fast_compute(tf.constant(5.0))
print(result.numpy())
```

---

## Question 8
- [ ] Done

**How does TensorFlow handle automatic differentiation?**

### Answer

**Definition**: TensorFlow uses **reverse-mode automatic differentiation** via `tf.GradientTape` to compute gradients.

### How It Works
1. **Record**: Operations inside `GradientTape` are recorded
2. **Forward pass**: Compute loss
3. **Backward pass**: `tape.gradient()` computes gradients
4. **Update**: Optimizer updates weights

### Python Code Example
```python
import tensorflow as tf

# Model parameters
w = tf.Variable(2.0)
b = tf.Variable(1.0)

# Data
x = tf.constant([1.0, 2.0, 3.0, 4.0])
y_true = tf.constant([3.0, 5.0, 7.0, 9.0])  # y = 2x + 1

# Optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Training step
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        # Forward pass
        y_pred = w * x + b
        loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Compute gradients
    gradients = tape.gradient(loss, [w, b])
    
    # Update weights
    optimizer.apply_gradients(zip(gradients, [w, b]))
    return loss

# Train
for epoch in range(100):
    loss = train_step()

print(f"w: {w.numpy():.4f}, b: {b.numpy():.4f}")  # ~2.0, ~1.0
```

---

## Question 9
- [ ] Done

**What is a Placeholder in TensorFlow?**

### Answer

**Definition**: `tf.placeholder` was a TF 1.x concept for declaring input variables. **Deprecated in TF 2.x**.

### Comparison

| TF 1.x (Placeholder) | TF 2.x (Modern) |
|---------------------|-----------------|
| Declare shape/dtype upfront | Pass data directly |
| Feed via `feed_dict` | Function arguments |
| Part of static graph | Eager execution |

### Code Evolution
```python
# TF 1.x: Placeholder (legacy)
# x = tf.placeholder(tf.float32, shape=[None, 784])
# result = sess.run(output, feed_dict={x: data})

# TF 2.x: Direct input (modern)
import tensorflow as tf

@tf.function
def model_forward(x):
    """x is just a function parameter, no placeholder needed"""
    return tf.nn.softmax(tf.matmul(x, weights) + bias)

# Call directly with data
data = tf.random.normal([32, 784])
output = model_forward(data)
```

### Summary
Placeholders are **obsolete** in TF 2.x. Use:
- Function parameters for inputs
- `tf.data.Dataset` for data pipelines

---

## Question 10
- [ ] Done

**Could you explain the concept of TensorFlow Lite and where it’s used?**

### Answer

**Definition**: TensorFlow Lite (TFLite) is a lightweight version for **mobile and embedded devices**.

### Key Features

| Feature | Description |
|---------|-------------|
| **Small size** | Optimized for mobile (~1MB) |
| **Quantization** | Float32 → Int8 (4x smaller) |
| **On-device** | No server needed |
| **Low latency** | Fast inference |

### Conversion Workflow
```python
import tensorflow as tf

# 1. Train a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10)

# 2. Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Apply quantization for smaller size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 3. Save the model
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Use Cases
- **Mobile apps**: Image classification, object detection
- **IoT devices**: Sensor data analysis
- **Microcontrollers**: Keyword spotting, gesture recognition

---

## Question 11
- [ ] Done

**What are the different data types supported by TensorFlow?**

### Answer

### Data Types Table

| Type | Bits | Use Case |
|------|------|----------|
| `tf.float32` | 32 | Default for ML (weights, inputs) |
| `tf.float64` | 64 | High precision scientific computing |
| `tf.float16` | 16 | Mixed precision training (faster) |
| `tf.bfloat16` | 16 | TPU optimization |
| `tf.int32` | 32 | Indices, labels |
| `tf.int64` | 64 | Large indices |
| `tf.bool` | 1 | Conditions, masks |
| `tf.string` | - | Text data |

### Python Code Example
```python
import tensorflow as tf

# Creating tensors with different dtypes
float_tensor = tf.constant([1.0, 2.0], dtype=tf.float32)
int_tensor = tf.constant([1, 2], dtype=tf.int32)
bool_tensor = tf.constant([True, False], dtype=tf.bool)

# Type casting
casted = tf.cast(int_tensor, tf.float32)

# Check dtype
print(f"dtype: {float_tensor.dtype}")  # <dtype: 'float32'>

# Mixed precision (for faster training on GPUs)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

### Best Practices
- Use `tf.float32` for most cases
- Use `tf.float16` for faster GPU training
- Use `tf.int32` for indices and labels

---

## Question 12
- [ ] Done

**Explain the process of compiling a model in TensorFlow.**

### Answer

**Definition**: Compiling a model in TensorFlow configures the model for training by specifying the **optimizer**, **loss function**, and **metrics**. It does not train the model — it prepares the computational machinery needed for the training loop.

### What `model.compile()` Configures

| Parameter | Purpose | Example |
|-----------|---------|---------|
| **optimizer** | Algorithm to update weights | `'adam'`, `'sgd'`, `tf.keras.optimizers.Adam(lr=0.001)` |
| **loss** | Function to minimize | `'sparse_categorical_crossentropy'`, `'mse'` |
| **metrics** | Values to monitor (not used for training) | `['accuracy']`, `[tf.keras.metrics.AUC()]` |
| **loss_weights** | Weight per output (multi-output models) | `[1.0, 0.5]` |
| **run_eagerly** | Force eager mode for debugging | `True` / `False` |

### Python Code Example
```python
import tensorflow as tf

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile — configure training
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Verify compilation
print(model.optimizer)   # <keras.optimizers.adam.Adam>
print(model.loss)        # sparse_categorical_crossentropy

# Custom compile for advanced use cases
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    ],
    run_eagerly=True  # Useful for debugging
)
```

### Interview Tips
- Compiling does **not** train — it only configures training parameters
- You can re-compile a model to change optimizer/loss without losing weights
- Use `run_eagerly=True` when debugging custom layers or losses
- For multi-output models, pass a dict of losses keyed by output layer name

---

## Question 13
- [ ] Done

**Describe how TensorFlow optimizers work and name a few common ones.**

### Answer

**Definition**: Optimizers are algorithms that **update model weights** to minimize the loss function. They use gradients computed via backpropagation to determine how to adjust each parameter.

### How Optimizers Work

1. **Forward Pass** → compute predictions
2. **Loss Computation** → measure error
3. **Backward Pass** → compute gradients via `tf.GradientTape`
4. **Weight Update** → optimizer adjusts weights using gradients

### Common Optimizers

| Optimizer | Key Idea | When to Use |
|-----------|----------|-------------|
| **SGD** | Basic gradient descent | Simple problems, fine-tuning |
| **SGD + Momentum** | Accumulates past gradients | Faster convergence |
| **Adam** | Adaptive learning rates + momentum | Default choice for most tasks |
| **AdamW** | Adam with decoupled weight decay | Transformers, NLP |
| **RMSprop** | Adaptive per-parameter rates | RNNs, non-stationary problems |
| **Adagrad** | Large updates for infrequent features | Sparse data (NLP, embeddings) |
| **Nadam** | Nesterov momentum + Adam | When Adam oscillates |

### Python Code Example
```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])

# SGD with momentum and learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=1000,
    decay_rate=0.96
)
sgd = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

# Adam — most popular default
adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

# Custom training loop with optimizer
optimizer = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.MeanSquaredError()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Gradient clipping to prevent exploding gradients
optimizer_clipped = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    clipnorm=1.0       # Clip gradients by norm
)
```

### Interview Tips
- **Adam** is the go-to default; it works well in most scenarios
- Learning rate is the most important hyperparameter to tune
- Use **learning rate schedules** for better convergence
- Gradient clipping (`clipnorm`, `clipvalue`) prevents exploding gradients in RNNs

---

## Question 14
- [ ] Done

**What is the role of loss functions in TensorFlow , and can you name some?**

### Answer

**Definition**: A loss function (or cost function) measures **how far the model's predictions are from the true values**. The optimizer minimizes this value during training.

### Common Loss Functions

| Loss Function | Task Type | Formula Intuition |
|---------------|-----------|-------------------|
| `MeanSquaredError` | Regression | Average of squared differences |
| `MeanAbsoluteError` | Regression (robust to outliers) | Average of absolute differences |
| `Huber` | Regression (balanced) | MSE for small errors, MAE for large |
| `BinaryCrossentropy` | Binary classification | Log loss for 2 classes |
| `CategoricalCrossentropy` | Multi-class (one-hot labels) | Log loss for N classes |
| `SparseCategoricalCrossentropy` | Multi-class (integer labels) | Same as above, integer labels |
| `CosineSimilarity` | Similarity / Ranking | Angle between prediction vectors |
| `KLDivergence` | Distribution matching (VAEs) | Divergence between distributions |

### Python Code Example
```python
import tensorflow as tf

# Regression losses
y_true = tf.constant([1.0, 2.0, 3.0])
y_pred = tf.constant([1.1, 2.2, 2.8])

mse = tf.keras.losses.MeanSquaredError()
print(f"MSE: {mse(y_true, y_pred).numpy():.4f}")  # 0.03

mae = tf.keras.losses.MeanAbsoluteError()
print(f"MAE: {mae(y_true, y_pred).numpy():.4f}")  # 0.1667

# Classification losses
y_true_cls = tf.constant([0, 1, 2])
y_pred_cls = tf.constant([[0.9, 0.05, 0.05],
                           [0.1, 0.8, 0.1],
                           [0.2, 0.3, 0.5]])

sce = tf.keras.losses.SparseCategoricalCrossentropy()
print(f"Sparse CE: {sce(y_true_cls, y_pred_cls).numpy():.4f}")

# Custom loss function
def custom_huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small = tf.abs(error) <= delta
    small_error = 0.5 * tf.square(error)
    large_error = delta * (tf.abs(error) - 0.5 * delta)
    return tf.reduce_mean(tf.where(is_small, small_error, large_error))

# Using custom loss
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(5,))])
model.compile(optimizer='adam', loss=custom_huber_loss)
```

### Interview Tips
- Use **MSE** for regression, **CrossEntropy** for classification
- Use `from_logits=True` when output layer has **no activation** (more numerically stable)
- `SparseCategoricalCrossentropy` takes integer labels; `CategoricalCrossentropy` takes one-hot
- Custom losses must accept `(y_true, y_pred)` and return a scalar tensor

---

## Question 15
- [ ] Done

**What are the differences between sequential and functional APIs in TensorFlow?**

### Answer

**Definition**: TensorFlow/Keras provides three model-building APIs: **Sequential**, **Functional**, and **Model Subclassing**. Sequential is the simplest; Functional supports complex architectures.

### Comparison

| Feature | Sequential API | Functional API |
|---------|---------------|----------------|
| **Topology** | Single linear stack | Any DAG (directed acyclic graph) |
| **Multiple inputs/outputs** | No | Yes |
| **Shared layers** | No | Yes |
| **Branching/merging** | No | Yes |
| **Residual connections** | No | Yes |
| **Complexity** | Very simple | Moderate |
| **Use case** | Simple feed-forward, CNNs | ResNets, multi-task, Inception |

### Python Code Example
```python
import tensorflow as tf

# --- Sequential API: Simple stack ---
seq_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# --- Functional API: Multi-input with residual ---
# Input layers
text_input = tf.keras.Input(shape=(100,), name='text')
meta_input = tf.keras.Input(shape=(5,), name='metadata')

# Text branch
x = tf.keras.layers.Dense(64, activation='relu')(text_input)
x = tf.keras.layers.Dense(32, activation='relu')(x)

# Metadata branch
y = tf.keras.layers.Dense(16, activation='relu')(meta_input)

# Merge branches
combined = tf.keras.layers.concatenate([x, y])
output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(combined)

func_model = tf.keras.Model(
    inputs=[text_input, meta_input],
    outputs=output
)

# Residual connection (skip connection)
inputs = tf.keras.Input(shape=(64,))
dense1 = tf.keras.layers.Dense(64, activation='relu')(inputs)
dense2 = tf.keras.layers.Dense(64)(dense1)
residual = tf.keras.layers.add([inputs, dense2])  # Skip connection
output = tf.keras.layers.Activation('relu')(residual)
res_model = tf.keras.Model(inputs=inputs, outputs=output)

print(func_model.summary())
```

### Interview Tips
- Start with **Sequential** for simple models; switch to **Functional** when architecture requires branching
- Functional API is required for multi-input/output models, skip connections, shared layers
- **Model subclassing** (inheriting `tf.keras.Model`) provides maximum flexibility but loses serialization benefits

---

## Question 16
- [ ] Done

**How does TensorFlow support regularization to prevent overfitting?**

### Answer

**Definition**: Regularization techniques constrain model complexity to reduce overfitting — when a model performs well on training data but poorly on unseen data.

### Regularization Techniques in TensorFlow

| Technique | How It Works | TensorFlow API |
|-----------|-------------|----------------|
| **L1 Regularization** | Penalizes absolute weight values (sparsity) | `tf.keras.regularizers.l1(0.01)` |
| **L2 Regularization** | Penalizes squared weight values (small weights) | `tf.keras.regularizers.l2(0.01)` |
| **L1+L2 (Elastic Net)** | Combines both penalties | `tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)` |
| **Dropout** | Randomly drops neurons during training | `tf.keras.layers.Dropout(0.5)` |
| **Batch Normalization** | Normalizes layer inputs | `tf.keras.layers.BatchNormalization()` |
| **Early Stopping** | Stops training when validation loss plateaus | `tf.keras.callbacks.EarlyStopping` |
| **Data Augmentation** | Generates varied training samples | `tf.keras.layers.RandomFlip`, etc. |

### Python Code Example
```python
import tensorflow as tf

model = tf.keras.Sequential([
    # L2 regularization on kernel weights
    tf.keras.layers.Dense(
        256, activation='relu', input_shape=(784,),
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(
        128, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Learning rate reduction on plateau
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3
)

# Data augmentation layer (for images)
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# model.fit(X_train, y_train, validation_split=0.2, epochs=100,
#           callbacks=[early_stop, reduce_lr])
```

### Interview Tips
- **Dropout** is the most widely used regularizer for deep networks
- **L2** keeps weights small; **L1** drives weights to zero (feature selection)
- **Early stopping** is simple and effective — always use it
- Combine multiple techniques: Dropout + L2 + BatchNorm + EarlyStopping

---

## Question 17
- [ ] Done

**Explain the concept of saving and restoring a model in TensorFlow.**

### Answer

**Definition**: TensorFlow provides multiple formats to **persist trained models** so they can be reloaded for inference, continued training, or deployment.

### Saving Formats

| Format | Saves | Use Case |
|--------|-------|----------|
| **SavedModel** (default) | Full model (architecture + weights + optimizer) | Production deployment, TF Serving |
| **HDF5 (.h5)** | Architecture + weights | Keras compatibility, sharing |
| **Checkpoints** | Weights only + optimizer state | Resuming training |
| **TFLite** | Optimized weights | Mobile/edge deployment |

### Python Code Example
```python
import tensorflow as tf

# Build and train a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
# model.fit(X_train, y_train, epochs=10)

# --- Method 1: SavedModel format (recommended) ---
model.save('saved_model/my_model')
loaded_model = tf.keras.models.load_model('saved_model/my_model')

# --- Method 2: HDF5 format ---
model.save('my_model.h5')
loaded_h5 = tf.keras.models.load_model('my_model.h5')

# --- Method 3: Weights only ---
model.save_weights('weights/my_weights')
new_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])
new_model.load_weights('weights/my_weights')

# --- Method 4: Checkpoints during training ---
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/epoch_{epoch:02d}_val_loss_{val_loss:.4f}',
    save_best_only=True,
    monitor='val_loss',
    save_weights_only=True
)

# model.fit(X_train, y_train, epochs=50,
#           validation_split=0.2, callbacks=[checkpoint_cb])

# Restore from checkpoint
# model.load_weights('checkpoints/epoch_42_val_loss_0.0123')

# --- Method 5: Export for TF Serving ---
tf.saved_model.save(model, 'export/1/')
# Serves via: tensorflow_model_server --model_base_path=export/
```

### Interview Tips
- **SavedModel** is the standard format for deployment and TF Serving
- **Checkpoints** are essential for long training runs to resume on failure
- Use `save_best_only=True` to keep only the best-performing model
- SavedModel includes the computation graph, making it language-agnostic

---

## Question 18
- [ ] Done

**How does TensorFlow integrate with Keras?**

### Answer

**Definition**: Since TensorFlow 2.0, **Keras is the official high-level API** of TensorFlow, fully integrated as `tf.keras`. It provides user-friendly model building, training, and evaluation interfaces on top of TensorFlow's engine.

### Integration History

| Version | Relationship |
|---------|-------------|
| **Pre-TF 2.0** | Keras was a separate library that could use TF as backend |
| **TF 2.0+** | Keras is bundled inside TensorFlow as `tf.keras` |
| **TF 2.12+** | Keras 3 (multi-backend) released separately; `tf.keras` remains |

### TensorFlow + Keras Architecture

| Layer | Component | Example |
|-------|-----------|---------|
| **High-level** | `tf.keras` (Model, Layers, Callbacks) | `model.fit()`, `model.predict()` |
| **Mid-level** | `tf.data`, `tf.image`, `tf.text` | Data pipelines, preprocessing |
| **Low-level** | `tf.GradientTape`, `tf.Variable`, `tf.function` | Custom training loops |

### Python Code Example
```python
import tensorflow as tf

# --- Using tf.keras high-level API ---
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model.fit(train_ds, epochs=5, validation_data=val_ds)

# --- Mixing Keras with low-level TF ---
class CustomModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        return self.dense2(x)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

custom_model = CustomModel()
custom_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
# custom_model.fit(X_train, y_train, epochs=5)
```

### Interview Tips
- Always use `tf.keras` (not standalone `keras`) for TensorFlow projects
- Keras provides three model APIs: Sequential, Functional, and Subclassing
- You can override `train_step()` to customize training while keeping `model.fit()` benefits
- Low-level TF operations work seamlessly inside Keras layers and models

---

## Question 19
- [ ] Done

**Describe the role of tf.data in TensorFlow.**

### Answer

**Definition**: `tf.data` is TensorFlow's API for building **efficient, scalable input pipelines**. It handles loading, transforming, batching, and prefetching data to feed models without becoming a training bottleneck.

### Key Capabilities

| Feature | Method | Purpose |
|---------|--------|---------|
| **Create** | `tf.data.Dataset.from_tensor_slices()` | From in-memory arrays |
| **Load files** | `tf.data.TFRecordDataset()` | From TFRecord files |
| **Transform** | `.map(fn)` | Apply preprocessing per element |
| **Batch** | `.batch(32)` | Group elements into batches |
| **Shuffle** | `.shuffle(1000)` | Randomize element order |
| **Prefetch** | `.prefetch(tf.data.AUTOTUNE)` | Overlap I/O and computation |
| **Cache** | `.cache()` | Keep dataset in memory after first epoch |
| **Repeat** | `.repeat()` | Loop dataset indefinitely |
| **Interleave** | `.interleave()` | Read from multiple files in parallel |

### Python Code Example
```python
import tensorflow as tf
import numpy as np

# --- Basic pipeline from arrays ---
X = np.random.randn(1000, 10).astype(np.float32)
y = np.random.randint(0, 2, size=(1000,)).astype(np.int32)

dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = (dataset
    .shuffle(buffer_size=1000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

# Iterate
for batch_x, batch_y in dataset.take(1):
    print(f"Batch shape: {batch_x.shape}")  # (32, 10)

# --- Image pipeline with parallel processing ---
def load_and_preprocess(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0  # Normalize
    return image, label

# file_paths = [...]  # list of image paths
# labels = [...]      # list of labels
# img_dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
# img_dataset = (img_dataset
#     .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
#     .cache()
#     .shuffle(1000)
#     .batch(32)
#     .prefetch(tf.data.AUTOTUNE)
# )

# --- TFRecord pipeline (production) ---
def parse_tfrecord(serialized):
    feature_spec = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed = tf.io.parse_single_example(serialized, feature_spec)
    image = tf.io.decode_raw(parsed['image'], tf.float32)
    return image, parsed['label']

# tfrecord_ds = tf.data.TFRecordDataset('data.tfrecord')
# tfrecord_ds = tfrecord_ds.map(parse_tfrecord).batch(32).prefetch(1)
```

### Interview Tips
- Always end pipelines with `.prefetch(tf.data.AUTOTUNE)` for best performance
- Use `.cache()` for small datasets that fit in memory
- `num_parallel_calls=tf.data.AUTOTUNE` in `.map()` parallelizes preprocessing
- TFRecord format is the recommended production format for large datasets

---

## Question 20
- [ ] Done

**What is TensorFlow Distribution Strategies and when would you use it?**

### Answer

**Definition**: `tf.distribute.Strategy` is TensorFlow's API for **distributing training across multiple GPUs, machines, or TPUs** with minimal code changes.

### Distribution Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `MirroredStrategy` | Synchronous training on multiple GPUs (single machine) | Most common multi-GPU setup |
| `MultiWorkerMirroredStrategy` | Synchronous across multiple machines | Multi-node clusters |
| `TPUStrategy` | Distributed on Google TPU pods | Cloud TPU training |
| `ParameterServerStrategy` | Async with parameter servers | Very large-scale training |
| `CentralStorageStrategy` | Variables on CPU, compute on GPUs | Models with large variables |
| `OneDeviceStrategy` | Run on single device (for testing) | Debugging distribution code |

### Python Code Example
```python
import tensorflow as tf

# --- MirroredStrategy: Multi-GPU on one machine ---
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

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

# Training automatically distributes across GPUs
# Effective batch size = batch_size * num_gpus
# model.fit(train_dataset, epochs=10)

# --- MultiWorkerMirroredStrategy (multi-machine) ---
# Requires TF_CONFIG environment variable
# import json, os
# os.environ['TF_CONFIG'] = json.dumps({
#     'cluster': {
#         'worker': ['worker0:port', 'worker1:port']
#     },
#     'task': {'type': 'worker', 'index': 0}
# })
# strategy = tf.distribute.MultiWorkerMirroredStrategy()

# --- TPUStrategy ---
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# tpu_strategy = tf.distribute.TPUStrategy(resolver)
```

### Interview Tips
- `MirroredStrategy` is the most common — wrap model building inside `strategy.scope()`
- Effective batch size scales with device count — adjust learning rate accordingly
- Data pipelines must use `tf.data` for distributed training to work properly
- Minimal code changes needed: just wrap in `strategy.scope()`

---

## Question 21
- [ ] Done

**Can you explain TensorFlow Extended (TFX) and its main components?**

### Answer

**Definition**: TensorFlow Extended (TFX) is an **end-to-end production ML platform** for deploying machine learning pipelines. It handles everything from data ingestion to model serving.

### TFX Pipeline Components

| Component | Purpose | Key Class |
|-----------|---------|-----------|
| **ExampleGen** | Ingest and split data | `CsvExampleGen`, `ImportExampleGen` |
| **StatisticsGen** | Compute dataset statistics | Uses TF Data Validation |
| **SchemaGen** | Infer data schema | Auto-generates feature schema |
| **ExampleValidator** | Detect data anomalies | Validates against schema |
| **Transform** | Feature engineering | `tf.Transform` preprocessing |
| **Trainer** | Train the model | Wraps `tf.keras` or Estimator |
| **Tuner** | Hyperparameter tuning | KerasTuner integration |
| **Evaluator** | Evaluate model quality | TF Model Analysis (TFMA) |
| **InfraValidator** | Test serving infrastructure | Validates deployability |
| **Pusher** | Deploy model | Push to TF Serving, TFLite, etc. |

### Python Code Example
```python
# TFX Pipeline Definition
import tfx
from tfx.components import (
    CsvExampleGen, StatisticsGen, SchemaGen,
    ExampleValidator, Transform, Trainer, Evaluator, Pusher
)
from tfx.orchestration.experimental.interactive.interactive_context import (
    InteractiveContext
)

# 1. Data Ingestion
example_gen = CsvExampleGen(input_base='data/')

# 2. Statistics
statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

# 3. Schema
schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])

# 4. Validation
example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema']
)

# 5. Feature Engineering (Transform)
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file='transform_module.py'
)

# 6. Training
trainer = Trainer(
    module_file='trainer_module.py',
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=tfx.proto.TrainArgs(num_steps=1000),
    eval_args=tfx.proto.EvalArgs(num_steps=200)
)

# 7. Push to serving
pusher = Pusher(
    model=trainer.outputs['model'],
    push_destination=tfx.proto.PushDestination(
        filesystem=tfx.proto.PushDestination.Filesystem(
            base_directory='serving_model/'
        )
    )
)
```

### Interview Tips
- TFX is for **production ML pipelines**, not experiments
- Each component produces **artifacts** tracked by ML Metadata (MLMD)
- Pipelines can be orchestrated with **Apache Beam, Airflow, or Kubeflow**
- TFX enforces best practices: data validation, model evaluation gates, reproducibility

---

## Question 22
- [ ] Done

**What is TensorFlow Serving and how does it facilitate model deployment?**

### Answer

**Definition**: TensorFlow Serving is a flexible, high-performance **serving system** for deploying ML models in production. It exposes trained models via **REST and gRPC APIs** for real-time inference.

### Key Features

| Feature | Description |
|---------|-------------|
| **Model Versioning** | Serve multiple model versions simultaneously |
| **Hot Swapping** | Update models without downtime |
| **Batching** | Automatically batch incoming requests |
| **gRPC + REST** | Both protocols supported |
| **GPU Support** | Hardware-accelerated inference |
| **Multi-Model** | Serve multiple models on one server |

### Deployment Workflow

```
Train Model -> Export SavedModel -> Deploy to TF Serving -> Client sends requests
```

### Python Code Example
```python
import tensorflow as tf
import json
import requests

# Step 1: Export model in SavedModel format
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
# model.fit(X_train, y_train)

# Export with version number
export_path = 'models/my_model/1/'  # version 1
tf.saved_model.save(model, export_path)

# Step 2: Start TF Serving (Docker)
# docker pull tensorflow/serving
# docker run -p 8501:8501 \
#   --mount type=bind,source=$(pwd)/models/my_model,target=/models/my_model \
#   -e MODEL_NAME=my_model \
#   tensorflow/serving

# Step 3: Send prediction request via REST API
data = json.dumps({
    "signature_name": "serving_default",
    "instances": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
})

# REST API call
response = requests.post(
    'http://localhost:8501/v1/models/my_model:predict',
    data=data,
    headers={"Content-Type": "application/json"}
)
predictions = response.json()['predictions']
print(f"Prediction: {predictions}")

# Check model status
status = requests.get('http://localhost:8501/v1/models/my_model')
print(status.json())
```

### Interview Tips
- Export models using `tf.saved_model.save()` with version directories (`1/`, `2/`, etc.)
- TF Serving auto-detects new versions and hot-swaps them
- Use **gRPC** for low-latency production; **REST** for simpler integration
- Docker is the easiest deployment method; Kubernetes for scaling

---

## Question 23
- [ ] Done

**How does one use TensorFlow’s Estimator API ?**

### Answer

**Definition**: The Estimator API is a **high-level TensorFlow API** (introduced in TF 1.x) that encapsulates training, evaluation, prediction, and export. It provides pre-made estimators and supports custom models. **Note**: Estimators are considered legacy in TF 2.x; `tf.keras` is preferred.

### Pre-made Estimators

| Estimator | Task |
|-----------|------|
| `tf.estimator.LinearClassifier` | Linear classification |
| `tf.estimator.LinearRegressor` | Linear regression |
| `tf.estimator.DNNClassifier` | Deep neural network classification |
| `tf.estimator.DNNRegressor` | Deep neural network regression |
| `tf.estimator.BoostedTreesClassifier` | Gradient boosted trees |

### Python Code Example
```python
import tensorflow as tf

# Define feature columns
feature_columns = [
    tf.feature_column.numeric_column('age'),
    tf.feature_column.numeric_column('hours_per_week'),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'education', ['Bachelors', 'Masters', 'Doctorate', 'HS-grad']
    ),
]

# Create a pre-made estimator
estimator = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[128, 64],
    n_classes=2,
    model_dir='estimator_model/'
)

# Input function — returns (features_dict, labels)
def input_fn(features, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.shuffle(1000).batch(batch_size)

# Train
# estimator.train(input_fn=lambda: input_fn(train_features, train_labels),
#                 steps=1000)

# Evaluate
# results = estimator.evaluate(
#     input_fn=lambda: input_fn(test_features, test_labels))

# Predict
# predictions = estimator.predict(
#     input_fn=lambda: input_fn(new_features, None))

# Convert Keras model to Estimator
keras_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
keras_model.compile(optimizer='adam', loss='binary_crossentropy')
keras_estimator = tf.keras.estimator.model_to_estimator(
    keras_model=keras_model, model_dir='keras_estimator/'
)
```

### Interview Tips
- **Estimators are legacy** — prefer `tf.keras` for new projects
- They are still found in production systems and older codebases
- `model_to_estimator()` bridges Keras models to Estimator infrastructure
- Estimators provide built-in distributed training, checkpointing, and TensorBoard integration

---

## Question 24
- [ ] Done

**Explain the concept of quantization in TensorFlow and when it might be used.**

### Answer

**Definition**: Quantization reduces model size and inference latency by converting **32-bit floating point** weights and activations to **lower-precision** formats (e.g., 8-bit integers), with minimal accuracy loss.

### Quantization Types

| Type | Method | Size Reduction | Accuracy Impact |
|------|--------|---------------|-----------------|
| **Post-Training Dynamic** | Quantize weights only at conversion | ~4x smaller | Minimal |
| **Post-Training Full (Static)** | Quantize weights + activations with calibration data | ~4x smaller | Very low |
| **Quantization-Aware Training (QAT)** | Simulate quantization during training | ~4x smaller | Lowest |
| **Float16 Quantization** | Convert to 16-bit float | ~2x smaller | Negligible |

### Python Code Example
```python
import tensorflow as tf
import numpy as np

# Train a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
# model.fit(X_train, y_train, epochs=5)

# --- Post-Training Dynamic Range Quantization ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
# Weights: float32 -> int8, activations: computed at runtime

# --- Post-Training Full Integer Quantization ---
def representative_dataset():
    """Calibration data for activation quantization"""
    for _ in range(100):
        yield [np.random.randn(1, 784).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
fully_quant_model = converter.convert()

# --- Quantization-Aware Training (QAT) ---
import tensorflow_model_optimization as tfmot

# Annotate model for quantization-aware training
qat_model = tfmot.quantization.keras.quantize_model(model)
qat_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
# qat_model.fit(X_train, y_train, epochs=3)  # Fine-tune

# Convert QAT model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
qat_tflite = converter.convert()

# Compare sizes
import os
with open('model_fp32.tflite', 'wb') as f:
    f.write(tf.lite.TFLiteConverter.from_keras_model(model).convert())
with open('model_int8.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print(f"FP32 size: {os.path.getsize('model_fp32.tflite') / 1024:.1f} KB")
print(f"INT8 size: {os.path.getsize('model_int8.tflite') / 1024:.1f} KB")
```

### Interview Tips
- Use **post-training quantization** for quick wins with minimal effort
- Use **QAT** when post-training quantization causes unacceptable accuracy loss
- Quantization is critical for **mobile/edge deployment** (TFLite)
- INT8 quantization gives ~4x size reduction and ~2-3x speed improvement

---

## Question 25
- [ ] Done

**How does TensorFlow support multi-GPU or distributed training?**

### Answer

**Definition**: TensorFlow supports scaling training across **multiple GPUs and machines** using `tf.distribute.Strategy`, which abstracts the complexity of data distribution, gradient synchronization, and variable placement.

### Multi-GPU / Distributed Approaches

| Approach | Parallelism Type | Description |
|----------|-----------------|-------------|
| **Data Parallelism** | Data | Same model on each GPU, different data batches |
| **Model Parallelism** | Model | Different parts of model on different GPUs |
| **Pipeline Parallelism** | Both | Layers split across GPUs in pipeline stages |

### Python Code Example
```python
import tensorflow as tf

# --- Check available GPUs ---
gpus = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {len(gpus)}")

# Enable memory growth (prevents TF from grabbing all GPU memory)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# --- Multi-GPU with MirroredStrategy ---
strategy = tf.distribute.MirroredStrategy()
# Uses NCCL for all-reduce by default

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Batch size scales with GPU count
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# Use tf.data for distributed input
# train_dataset = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
#     .shuffle(10000).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
# model.fit(train_dataset, epochs=10)

# --- Manual device placement ---
# Place specific operations on specific GPUs
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0]])
with tf.device('/GPU:1'):
    b = tf.constant([[3.0], [4.0]])
# c = tf.matmul(a, b)  # Result placed automatically

# --- Custom distributed training loop ---
@tf.function
def distributed_train_step(dataset_inputs):
    def replica_step(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(
                y, predictions)
            loss = tf.nn.compute_average_loss(per_example_loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))
        return loss

    per_replica_losses = strategy.run(replica_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM,
                           per_replica_losses, axis=None)
```

### Interview Tips
- **Data parallelism** (MirroredStrategy) is easiest and most common
- Scale learning rate linearly with number of GPUs
- Use `tf.data` pipelines — NumPy arrays don't distribute efficiently
- NCCL is the fastest all-reduce backend for NVIDIA GPUs

---

## Question 26
- [ ] Done

**What are some best practices for writing efficient TensorFlow code?**

### Answer

### Best Practices Summary

| Category | Best Practice | Why |
|----------|--------------|-----|
| **Data Pipeline** | Use `tf.data` with `.prefetch(AUTOTUNE)` | Prevents data loading bottleneck |
| **Graph Mode** | Decorate with `@tf.function` | ~3-10x speedup over eager |
| **Mixed Precision** | Use `mixed_float16` policy | 2-3x faster on modern GPUs |
| **Vectorization** | Use TF ops instead of Python loops | Leverages GPU parallelism |
| **Memory** | Enable GPU memory growth | Prevents OOM errors |
| **Profiling** | Use TensorBoard Profiler | Identify bottlenecks |
| **XLA Compilation** | Enable `jit_compile=True` | Further graph optimizations |

### Python Code Example
```python
import tensorflow as tf

# 1. Enable mixed precision for faster GPU training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 2. Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 3. Use @tf.function for graph execution
@tf.function(jit_compile=True)  # XLA compilation for extra speed
def train_step(model, x, y, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# 4. Efficient data pipeline
def create_dataset(X, y, batch_size=32):
    return (tf.data.Dataset.from_tensor_slices((X, y))
        .cache()                                   # Cache in memory
        .shuffle(buffer_size=len(X))               # Full shuffle
        .batch(batch_size)                         # Batch
        .prefetch(tf.data.AUTOTUNE))               # Overlap I/O & compute

# 5. Vectorized operations (DO)
x = tf.random.normal([1000, 100])
result = tf.reduce_sum(x ** 2, axis=1)  # Vectorized — fast

# Avoid Python loops (DON'T)
# result = [tf.reduce_sum(x[i] ** 2) for i in range(1000)]  # Slow!

# 6. Profile with TensorBoard
tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir='logs/', profile_batch='10,20'  # Profile batches 10-20
)
# model.fit(train_ds, epochs=5, callbacks=[tensorboard_cb])

# 7. Use tf.TensorSpec for input signatures
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32)])
def predict(x):
    return model(x)
```

### Interview Tips
- **`@tf.function`** is the single biggest performance improvement
- **Profile before optimizing** — use TensorBoard Profiler to find real bottlenecks
- Avoid Python-side operations inside `@tf.function` (use `tf.py_function` if needed)
- **XLA (`jit_compile=True`)** can give additional 10-30% speedup

---

## Question 27
- [ ] Done

**Describe the process of text preprocessing in TensorFlow.**

### Answer

**Definition**: Text preprocessing in TensorFlow converts raw text into **numerical representations** that neural networks can process. This involves tokenization, vocabulary building, padding, and optionally embedding.

### Text Preprocessing Pipeline

| Step | Purpose | TensorFlow API |
|------|---------|----------------|
| **Standardization** | Lowercase, strip HTML, remove punctuation | `TextVectorization(standardize=...)` |
| **Tokenization** | Split text into tokens (words/subwords) | `TextVectorization`, `tfds.deprecated.text` |
| **Vocabulary** | Map tokens to integer IDs | `TextVectorization(max_tokens=...)` |
| **Sequencing** | Convert text to integer sequences | `TextVectorization(output_mode='int')` |
| **Padding** | Make all sequences same length | `TextVectorization(output_sequence_length=...)` |
| **Embedding** | Map integers to dense vectors | `tf.keras.layers.Embedding` |

### Python Code Example
```python
import tensorflow as tf
import numpy as np

# --- TextVectorization Layer (recommended TF 2.x approach) ---
texts = [
    "TensorFlow is great for deep learning",
    "I love building neural networks",
    "Machine learning with Python is fun",
    "Deep learning models need lots of data"
]
labels = [1, 1, 1, 0]

# Create vectorization layer
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=1000,              # Vocabulary size
    output_mode='int',            # Output integer sequences
    output_sequence_length=20,    # Pad/truncate to length 20
    standardize='lower_and_strip_punctuation'
)

# Build vocabulary from data
vectorizer.adapt(texts)
print(f"Vocabulary: {vectorizer.get_vocabulary()[:10]}")

# Vectorize text
encoded = vectorizer(texts)
print(f"Encoded shape: {encoded.shape}")  # (4, 20)

# --- Build text classification model ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,), dtype=tf.string),
    vectorizer,
    tf.keras.layers.Embedding(
        input_dim=1000, output_dim=64, mask_zero=True
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
# model.fit(np.array(texts), np.array(labels), epochs=10)

# --- Subword tokenization with TF Text ---
# import tensorflow_text as text
# tokenizer = text.BertTokenizer('vocab.txt', lower_case=True)
# tokens = tokenizer.tokenize(["Hello world"])
```

### Interview Tips
- `TextVectorization` is the modern approach — it's a Keras layer that can be saved with the model
- Use `adapt()` to build vocabulary from training data
- Subword tokenization (BPE, WordPiece) is preferred for modern NLP (transformers)
- Include the vectorization layer inside the model for seamless deployment

---

## Question 28
- [ ] Done

**Explain how you would approach time-series forecasting with TensorFlow.**

### Answer

**Definition**: Time-series forecasting predicts future values based on historical temporal data. TensorFlow supports this via **RNNs, LSTMs, Transformers**, and specialized windowing utilities.

### Approach Overview

| Step | Action | Tool |
|------|--------|------|
| 1. **Data Preparation** | Window the series into (input, label) pairs | `tf.keras.utils.timeseries_dataset_from_array` |
| 2. **Normalization** | Scale features to [0,1] or standardize | `tf.keras.layers.Normalization` |
| 3. **Model Selection** | Choose architecture based on complexity | LSTM, GRU, Conv1D, Transformer |
| 4. **Training** | Train with appropriate loss (MAE, MSE) | `model.fit()` |
| 5. **Evaluation** | Compare with baselines (naive, moving avg) | MAE, RMSE, MAPE |

### Python Code Example
```python
import tensorflow as tf
import numpy as np

# --- Generate sample time series ---
np.random.seed(42)
time = np.arange(1000)
series = 10 + np.sin(time * 0.1) * 10 + np.random.randn(1000) * 2

# --- Windowing function ---
def create_dataset(series, window_size, forecast_horizon=1):
    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=series[:-forecast_horizon],
        targets=series[window_size:],
        sequence_length=window_size,
        batch_size=32,
        shuffle=True
    )
    return dataset

WINDOW_SIZE = 30
train_data = series[:800]
val_data = series[800:]

train_ds = create_dataset(train_data, WINDOW_SIZE)
val_ds = create_dataset(val_data, WINDOW_SIZE)

# --- Model 1: LSTM ---
lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True,
                         input_shape=(WINDOW_SIZE, 1)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

# --- Model 2: Conv1D + LSTM hybrid ---
hybrid_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, 3, activation='relu',
                           input_shape=(WINDOW_SIZE, 1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])

# --- Model 3: Transformer-based ---
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    ff = tf.keras.layers.Dense(ff_dim, activation='relu')(x)
    ff = tf.keras.layers.Dense(inputs.shape[-1])(ff)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)

# Compile and train
lstm_model.compile(optimizer='adam', loss='mae')
# lstm_model.fit(train_ds, validation_data=val_ds, epochs=50,
#                callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])
```

### Interview Tips
- Always establish a **baseline** (naive forecast, moving average) before using DL
- **LSTMs/GRUs** work well for most time-series problems
- Use **Conv1D** for capturing local patterns efficiently
- Normalize data before training; denormalize predictions for evaluation
- Multi-step forecasting: use autoregressive prediction or direct multi-output

---

## Question 29
- [ ] Done

**How does TensorFlow’s tf.debugging package assist in debugging?**

### Answer

**Definition**: `tf.debugging` provides a suite of **assertion and validation functions** that help catch numerical issues, shape mismatches, and data problems during development and training.

### Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `tf.debugging.assert_equal` | Check tensor equality | Verify shapes, labels |
| `tf.debugging.assert_near` | Check approximate equality | Floating point comparison |
| `tf.debugging.assert_positive` | Check all values > 0 | Validate probabilities |
| `tf.debugging.assert_shapes` | Validate tensor shapes | Dimension checking |
| `tf.debugging.check_numerics` | Detect NaN/Inf | Training stability |
| `tf.debugging.enable_check_numerics` | Global NaN/Inf detection | Catch issues anywhere |
| `tf.debugging.assert_rank` | Verify tensor rank | Input validation |
| `tf.debugging.assert_type` | Check dtype | Type safety |

### Python Code Example
```python
import tensorflow as tf

# --- Basic assertions ---
x = tf.constant([1.0, 2.0, 3.0])
y = tf.constant([1.0, 2.0, 3.0])

tf.debugging.assert_equal(x, y)                    # Passes
tf.debugging.assert_positive(x)                    # Passes
tf.debugging.assert_rank(x, 1)                     # Passes — x is rank 1

# Assert near (for floating point comparisons)
a = tf.constant(0.1 + 0.2)
tf.debugging.assert_near(a, tf.constant(0.3), atol=1e-6)

# --- Shape validation ---
batch = tf.random.normal([32, 784])
tf.debugging.assert_shapes([
    (batch, ('batch', 784)),
])

# --- Check for NaN/Inf ---
tensor = tf.constant([1.0, 2.0, float('nan')])
try:
    tf.debugging.check_numerics(tensor, "Found bad values!")
except tf.errors.InvalidArgumentError as e:
    print(f"Caught: {e}")

# --- Global NaN/Inf checking during training ---
tf.debugging.enable_check_numerics()  # Enable globally

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
# model.fit(X_train, y_train, epochs=5)
# Any NaN/Inf during training will raise an error with a traceback

tf.debugging.disable_check_numerics()  # Disable when done

# --- Custom debugging in training loop ---
@tf.function
def safe_train_step(model, x, y, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        tf.debugging.check_numerics(predictions, "NaN in predictions")
        loss = tf.reduce_mean(tf.square(y - predictions))
        tf.debugging.check_numerics(loss, "NaN in loss")
    gradients = tape.gradient(loss, model.trainable_variables)
    for g in gradients:
        tf.debugging.check_numerics(g, "NaN in gradients")
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

### Interview Tips
- Use `enable_check_numerics()` early in development to catch NaN/Inf issues
- `assert_shapes` is invaluable for debugging shape mismatches in complex models
- These assertions work inside `@tf.function` (graph mode)
- Disable debugging checks in production for performance (`disable_check_numerics()`)

---

## Question 30
- [ ] Done

**What are some of the latest features or additions to TensorFlow that are currently gaining traction?**

### Answer

### Recent TensorFlow Developments

| Feature | Description | Status |
|---------|-------------|--------|
| **Keras 3 (Multi-Backend)** | Keras works with TF, PyTorch, JAX | Released 2023, gaining adoption |
| **DTensor** | Distributed tensor API for model parallelism | Experimental to Stable |
| **tf.numpy** | NumPy-compatible API on accelerators | Stable in TF 2.x |
| **XLA (jit_compile)** | Ahead-of-time compilation for faster execution | Widely adopted |
| **Mixed Precision** | FP16/BF16 training for 2-3x speedup | Standard practice |
| **TF Lite for Microcontrollers** | ML on embedded devices (Arduino, ESP32) | Growing ecosystem |
| **SavedModel improvements** | Better signatures, fingerprinting | Ongoing |
| **JAX interop** | Shared XLA compiler backend | Growing convergence |
| **TensorFlow Decision Forests** | Gradient boosted trees, random forests in TF | Stable |
| **TF Recommenders (TFRS)** | Recommendation system library | Production use at Google |

### Python Code Example
```python
import tensorflow as tf

# --- Feature 1: XLA Compilation ---
@tf.function(jit_compile=True)
def xla_optimized(x, y):
    return tf.matmul(x, y) + tf.reduce_sum(x)

# --- Feature 2: Mixed Precision Training ---
tf.keras.mixed_precision.set_global_policy('mixed_float16')
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(1000,)),
    tf.keras.layers.Dense(10, dtype='float32')  # Output in float32
])

# --- Feature 3: DTensor for model parallelism ---
# from tensorflow.experimental import dtensor
# mesh = dtensor.create_mesh([("batch", 2)], devices=["GPU:0", "GPU:1"])
# layout = dtensor.Layout(["batch", dtensor.UNSHARDED], mesh)

# --- Feature 4: TensorFlow Decision Forests ---
# import tensorflow_decision_forests as tfdf
# forest_model = tfdf.keras.RandomForestModel()
# forest_model.fit(train_ds)

# --- Feature 5: tf.numpy for NumPy-style ops on GPU ---
x = tf.experimental.numpy.linspace(0, 10, 100)
y = tf.experimental.numpy.sin(x)

# --- Feature 6: Keras preprocessing layers (in-model) ---
preprocessing = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
])
```

### Interview Tips
- **Keras 3** multi-backend support is the biggest architectural shift — code runs on TF, PyTorch, or JAX
- **XLA compilation** (`jit_compile=True`) is increasingly standard for performance
- TensorFlow is converging with **JAX** on the compiler level (shared XLA backend)
- **TF Decision Forests** bridges the gap between classical ML and deep learning in one framework

---

## Question 31
- [ ] Done

**Explain the concept of eager execution in TensorFlow.**



**Eager execution** is TensorFlow's default execution mode (since TF 2.0) where operations are evaluated immediately as they are called, returning concrete values instead of building a computational graph.

| Aspect | Eager Execution | Graph Execution |
|--------|----------------|-----------------|
| Execution | Immediate | Deferred (build then run) |
| Debugging | Easy (use print, pdb) | Hard (requires tf.print, sessions) |
| Default in | TF 2.x | TF 1.x |
| Performance | Slightly slower | Optimized by compiler |
| Pythonic | Yes, natural Python flow | No, requires special TF ops |

```python
import tensorflow as tf

# Eager mode (default in TF 2.x) - operations execute immediately
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
c = tf.matmul(a, b)
print(c.numpy())  # Works immediately, no session needed

# You can use standard Python control flow
if tf.reduce_sum(c) > 50:
    print("Sum is large")

# For production performance, use @tf.function to get graph benefits
@tf.function
def fast_matmul(a, b):
    return tf.matmul(a, b)
```

> **Interview Tip:** Eager execution makes TF 2.x feel like NumPy. For production, wrap performance-critical code in `@tf.function` to get graph optimizations while keeping eager mode for debugging.

---

## Question 32
- [ ] Done

**Explain the concept of graph mode versus eager mode in TensorFlow.**



| Feature | Eager Mode | Graph Mode |
|---------|-----------|------------|
| **Execution** | Operations run immediately | Operations build a graph first, then execute |
| **Debugging** | Standard Python debugger works | Requires `tf.print`, `tfdbg` |
| **Performance** | Good for prototyping | Better for production (compiler optimizations) |
| **Portability** | Python-dependent | Graphs can be exported and run without Python |
| **Default** | TF 2.x default | TF 1.x default |
| **Usage** | Development, debugging | Deployment, mobile, serving |

```python
import tensorflow as tf

# Eager mode (default): runs immediately
x = tf.constant(3.0)
y = x ** 2 + 2 * x + 1
print(y.numpy())  # 16.0, available immediately

# Graph mode via @tf.function: traces Python into a graph
@tf.function
def compute(x):
    return x ** 2 + 2 * x + 1

result = compute(tf.constant(3.0))  # Runs as optimized graph

# Best practice: develop in eager, deploy with @tf.function
# @tf.function benefits:
# - Automatic graph optimization (constant folding, op fusion)
# - Exportable as SavedModel for serving
# - Runs on GPU/TPU efficiently
```

> **Interview Tip:** TF 2.x gives you the best of both worlds: debug in **eager mode**, deploy in **graph mode** using `@tf.function`. Mention that `@tf.function` traces your Python code into a portable, optimizable graph.

## Question 33
- [ ] Done

**What is a Session in TensorFlow ? Explain its role**

**Answer:**

A **Session** (`tf.Session`) was TensorFlow 1.x's mechanism to execute computational graphs. It allocated resources (CPU/GPU), ran operations, and evaluated tensors.

| Aspect | Detail |
|--------|--------|
| **Purpose** | Execute operations in a pre-built computational graph |
| **TF 1.x** | Required: `sess.run(tensor)` to get values |
| **TF 2.x** | Removed: eager execution replaces sessions |
| **Resource mgmt** | Sessions managed GPU memory and device placement |

```python
# TF 1.x (legacy) - Sessions required
import tensorflow.compat.v1 as tf1
tf1.disable_eager_execution()

a = tf1.placeholder(tf1.float32, shape=[2])
b = tf1.constant([3.0, 4.0])
c = a + b

with tf1.Session() as sess:
    result = sess.run(c, feed_dict={a: [1.0, 2.0]})
    print(result)  # [4.0, 6.0]

# TF 2.x (modern) - No sessions needed
import tensorflow as tf
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])
c = a + b
print(c.numpy())  # [4.0, 6.0] - immediate
```

> **Interview Tip:** Sessions are **TF 1.x only**. In TF 2.x, eager execution replaced them entirely. If you encounter legacy code, use `tf.compat.v1.Session()` for migration.

---

## Question 34
- [ ] Done

**What is a Placeholder in TensorFlow , and how is it used?**

**Answer:**

A **Placeholder** (`tf.placeholder`) was TF 1.x's way to define input nodes in a computational graph. Data was fed at runtime via `feed_dict` in a Session.

| Aspect | Detail |
|--------|--------|
| **Purpose** | Define input slots for data feeding |
| **TF 1.x** | `tf.placeholder(dtype, shape)` |
| **TF 2.x** | Replaced by `tf.function` arguments and `tf.data` |
| **Feed mechanism** | `sess.run(output, feed_dict={placeholder: data})` |

```python
# TF 1.x (legacy)
import tensorflow.compat.v1 as tf1
tf1.disable_eager_execution()

x = tf1.placeholder(tf1.float32, shape=[None, 784])  # Batch of images
y = tf1.placeholder(tf1.int32, shape=[None])           # Labels
logits = tf1.layers.dense(x, 10)

with tf1.Session() as sess:
    sess.run(tf1.global_variables_initializer())
    result = sess.run(logits, feed_dict={x: batch_images})

# TF 2.x equivalent - just use function arguments
@tf.function
def predict(x):
    return model(x)

# Or use tf.data.Dataset for efficient input pipelines
dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(32)
```

> **Interview Tip:** Placeholders are **obsolete in TF 2.x**. Modern TF uses `tf.data.Dataset` for input pipelines and Python function arguments with `@tf.function` for graph tracing.

---

## Question 35
- [ ] Done

**How do you use callbacks in TensorFlow ?**

**Answer:**

**Callbacks** are objects that perform actions at various stages of training (epoch start/end, batch start/end). They enable monitoring, early stopping, learning rate scheduling, and checkpointing.

| Callback | Purpose |
|----------|---------|
| `EarlyStopping` | Stop training when metric stops improving |
| `ModelCheckpoint` | Save best model during training |
| `ReduceLROnPlateau` | Reduce learning rate when metric plateaus |
| `TensorBoard` | Log metrics for visualization |
| `LearningRateScheduler` | Custom learning rate schedule |
| `CSVLogger` | Log metrics to CSV file |

```python
import tensorflow as tf

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras', monitor='val_loss', save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7
    ),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=callbacks)

# Custom callback
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate.numpy()
        print(f"Epoch {epoch}: LR = {lr:.6f}, Loss = {logs['loss']:.4f}")
```

> **Interview Tip:** Always use **EarlyStopping** + **ModelCheckpoint** together. Set `restore_best_weights=True` to automatically load the best model when training stops.

---

## Question 36
- [ ] Done

**What strategies does TensorFlow use to handle overfitting during training?**

**Answer:**

| Strategy | Implementation | When to Use |
|----------|---------------|-------------|
| **Dropout** | `layers.Dropout(0.5)` | Dense layers, large networks |
| **L2 Regularization** | `kernel_regularizer=tf.keras.regularizers.l2(0.01)` | Prevent large weights |
| **Early Stopping** | `EarlyStopping(patience=5)` | Always recommended |
| **Data Augmentation** | `RandomFlip`, `RandomRotation` layers | Image data |
| **Batch Normalization** | `layers.BatchNormalization()` | Stabilize training |
| **Reduce Model Size** | Fewer layers/units | Small datasets |
| **More Data** | Collect or augment | Best long-term solution |

```python
import tensorflow as tf
from tensorflow.keras import layers, regularizers

model = tf.keras.Sequential([
    layers.Dense(256, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)),  # L2 regularization
    layers.BatchNormalization(),                              # Batch norm
    layers.Dropout(0.5),                                      # Dropout
    layers.Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                      restore_best_weights=True)
]
model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=callbacks)
```

> **Interview Tip:** Combine multiple strategies: **Dropout + L2 + EarlyStopping + BatchNorm**. Monitor the gap between training and validation loss to detect overfitting early.

---

## Question 37
- [ ] Done

**Discuss how to use mixed-precision training in TensorFlow**

**Answer:**

**Mixed-precision training** uses both 16-bit (float16) and 32-bit (float32) floating-point types during training. Computations use float16 for speed, while critical operations (loss scaling, weight updates) use float32 for numerical stability.

| Aspect | float32 | float16 (mixed) |
|--------|---------|-----------------|
| Memory | Baseline | ~50% reduction |
| Speed | Baseline | 2-3x faster on modern GPUs |
| Precision | Full | Slightly reduced |
| GPU Support | All | Volta+ (V100, A100, RTX) |

```python
import tensorflow as tf

# Enable mixed precision globally (one line!)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Build model normally - layers auto-use float16
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    # Output layer should use float32 for numerical stability
    tf.keras.layers.Dense(10, dtype='float32')
])

# Optimizer with loss scaling to prevent underflow
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training proceeds normally
model.fit(X_train, y_train, epochs=10, batch_size=128)
```

> **Interview Tip:** Mixed precision provides **2-3x speedup** with minimal accuracy loss. The key requirement is a GPU with Tensor Cores (NVIDIA Volta or newer). Always keep the final output layer in float32.

---

## Question 38
- [ ] Done

**What support does TensorFlow offer for transfer learning ?**

**Answer:**

**Transfer learning** reuses a model pre-trained on a large dataset (e.g., ImageNet) and adapts it to a new task, significantly reducing training time and data requirements.

| Strategy | Method | When |
|----------|--------|------|
| **Feature Extraction** | Freeze base, train new head | Small dataset, similar domain |
| **Fine-tuning** | Unfreeze top layers, train with low LR | Medium dataset |
| **Full Fine-tuning** | Unfreeze all, train with very low LR | Large dataset |

```python
import tensorflow as tf

# 1. Load pre-trained model (without top classification layer)
base_model = tf.keras.applications.ResNet50(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3)
)

# 2. Feature extraction: freeze base model
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=10)  # Train only new layers

# 3. Fine-tuning: unfreeze top layers of base model
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Freeze all except last 20 layers
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Very low learning rate
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=10)  # Fine-tune
```

> **Interview Tip:** Always start with **feature extraction** (frozen base), then optionally **fine-tune** top layers with a **very low learning rate** (1e-5). This two-phase approach prevents catastrophic forgetting.

---

## Question 39
- [ ] Done

**Discuss how GradientTape works in TensorFlow**

**Answer:**

`tf.GradientTape` is TensorFlow's automatic differentiation engine. It records operations on tensors inside its context, then computes gradients via backpropagation.

| Feature | Description |
|---------|-------------|
| **Purpose** | Record operations for automatic differentiation |
| **Scope** | Only tracks operations inside `with tf.GradientTape()` block |
| **Default** | Watches only `tf.Variable` objects |
| **Persistent** | Set `persistent=True` to compute multiple gradients |

```python
import tensorflow as tf

# Basic gradient computation
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x ** 2 + 2 * x + 1  # y = x^2 + 2x + 1

dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())  # 8.0 (dy/dx = 2x + 2 = 8)

# Custom training with GradientTape
model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

for X_batch, y_batch in dataset:
    with tf.GradientTape() as tape:
        predictions = model(X_batch, training=True)
        loss = loss_fn(y_batch, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Higher-order gradients (persistent tape)
x = tf.Variable(3.0)
with tf.GradientTape(persistent=True) as tape:
    y = x ** 3
dy = tape.gradient(y, x)    # 27.0 (3x^2)
d2y = tape.gradient(dy, x)  # 18.0 (6x)
del tape  # Free resources
```

> **Interview Tip:** `GradientTape` is the foundation of custom training loops in TF 2.x. Use `persistent=True` only when you need multiple gradient computations from the same tape, and remember to `del tape` afterward.

---
## Question 40
- [ ] Done

**How would you go about debugging a TensorFlow model that isn’t learning?**

**Answer:**

### Systematic Debugging Checklist

| Check | What to Look For | Fix |
|-------|-----------------|-----|
| **Loss not decreasing** | Loss stays flat or NaN | Check learning rate, loss function |
| **Learning rate** | Too high (diverges) or too low (no progress) | Try 1e-3, then tune |
| **Data pipeline** | Wrong labels, no shuffling, no normalization | Verify X-y alignment, scale inputs |
| **Architecture** | Too small or too complex for the task | Start simple, then scale up |
| **Activation functions** | Dead ReLU, vanishing gradients | Try LeakyReLU, check init |
| **Loss function** | Wrong choice for the task | Classification: crossentropy; Regression: MSE |
| **Gradient flow** | Vanishing or exploding gradients | Gradient clipping, BatchNorm |

```python
import tensorflow as tf

# 1. Check data: verify labels match inputs
for X_batch, y_batch in train_ds.take(1):
    print(f"X shape: {X_batch.shape}, y shape: {y_batch.shape}")
    print(f"y unique values: {tf.unique(tf.reshape(y_batch, [-1]))[0].numpy()}")

# 2. Check gradients are flowing
model = build_model()
with tf.GradientTape() as tape:
    pred = model(X_batch, training=True)
    loss = loss_fn(y_batch, pred)
grads = tape.gradient(loss, model.trainable_variables)
for var, grad in zip(model.trainable_variables, grads):
    if grad is not None:
        print(f"{var.name}: grad mean={tf.reduce_mean(tf.abs(grad)):.6f}")
    else:
        print(f"{var.name}: NO GRADIENT!")

# 3. Overfit on a single batch first
small_ds = train_ds.take(1).repeat()
model.fit(small_ds, steps_per_epoch=100, epochs=5)
# If it can't overfit one batch, architecture or loss is wrong

# 4. Use TensorBoard for visualization
tensorboard_cb = tf.keras.callbacks.TensorBoard('./debug_logs', histogram_freq=1)
```

> **Interview Tip:** Always try to **overfit on a single batch first**. If the model cannot memorize one batch, the problem is in the architecture, loss function, or data pipeline rather than capacity.

---

## Question 41
- [ ] Done

**Discuss common errors encountered in TensorFlow and how to resolve them**

**Answer:**

| Error | Cause | Solution |
|-------|-------|----------|
| **Shape mismatch** | Input/output dimensions don't match | Check shapes with `model.summary()` |
| **OOM (Out of Memory)** | Model/batch too large for GPU | Reduce batch size, use mixed precision |
| **NaN loss** | Exploding gradients, bad data | Clip gradients, check for NaN in data |
| **Incompatible dtypes** | Mixing float32/float64/int | Cast tensors: `tf.cast(x, tf.float32)` |
| **Graph vs Eager issues** | Python ops inside `@tf.function` | Use `tf.py_function` or rewrite with TF ops |
| **Variable not tracked** | Creating variables outside `__init__` | Define all layers in `__init__` |
| **Slow training** | CPU instead of GPU | Check `tf.config.list_physical_devices('GPU')` |

```python
import tensorflow as tf

# 1. Shape mismatch - check before training
model.summary()  # Verify layer shapes
print(f"Input shape expected: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# 2. OOM fix - limit GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 3. NaN detection
if tf.reduce_any(tf.math.is_nan(loss)):
    print("NaN detected in loss!")

# 4. Gradient clipping to prevent exploding gradients
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

# 5. Check GPU usage
print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
```

> **Interview Tip:** Always call `model.summary()` before training to verify shapes. Use `set_memory_growth(True)` to prevent TF from allocating all GPU memory at once. Gradient clipping (`clipnorm=1.0`) solves most exploding gradient issues.

---

## Question 42
- [ ] Done

**How can the TensorBoard tool be used to debug TensorFlow programs ?**

**Answer:**

**TensorBoard** is TensorFlow's visualization toolkit for monitoring training metrics, model graphs, histograms, images, and more.

| Feature | Purpose |
|---------|---------|
| **Scalars** | Track loss, accuracy, learning rate over time |
| **Graphs** | Visualize model architecture |
| **Histograms** | Distribution of weights and gradients |
| **Images** | Display sample predictions |
| **Profiler** | Identify performance bottlenecks |
| **HParams** | Compare hyperparameter experiments |

```python
import tensorflow as tf
import datetime

# 1. Basic usage with callback
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,    # Log weight histograms every epoch
    write_graph=True,    # Visualize model graph
    write_images=True,
    profile_batch='10,20'  # Profile batches 10-20
)

model.fit(X_train, y_train, epochs=50, validation_split=0.2,
          callbacks=[tensorboard_cb])

# 2. Custom scalar logging
summary_writer = tf.summary.create_file_writer("logs/custom")
for epoch in range(100):
    with summary_writer.as_default():
        tf.summary.scalar('custom_metric', value, step=epoch)
        tf.summary.image('predictions', images, step=epoch)

# Launch TensorBoard:
# tensorboard --logdir=logs/
# Then open http://localhost:6006 in browser
```

> **Interview Tip:** Always log to **timestamped subdirectories** so you can compare runs. Use `histogram_freq=1` for weight distribution analysis and the **Profiler** tab to find GPU/CPU bottlenecks.

---

