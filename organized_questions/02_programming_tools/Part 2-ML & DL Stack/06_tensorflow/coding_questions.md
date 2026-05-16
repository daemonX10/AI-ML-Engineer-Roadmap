# TensorFlow Interview Questions - Coding Questions

## Question 1
- [ ] Done

**How do you implement a custom layer in TensorFlow?**

### Solution
```python
import tensorflow as tf

class CustomDenseLayer(tf.keras.layers.Layer):
    """Custom dense layer with optional L2 regularization"""
    
    def __init__(self, units, activation=None, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
    
    def build(self, input_shape):
        # Create trainable weights
        self.w = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        if self.use_bias:
            self.b = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True
            )
        super().build(input_shape)
    
    def call(self, inputs):
        output = tf.matmul(inputs, self.w)
        if self.use_bias:
            output = output + self.b
        if self.activation:
            output = self.activation(output)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias
        })
        return config

# Use custom layer
model = tf.keras.Sequential([
    CustomDenseLayer(64, activation='relu', input_shape=(10,)),
    CustomDenseLayer(32, activation='relu'),
    CustomDenseLayer(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()
```

---

## Question 2
- [ ] Done

**Implement a custom training loop with GradientTape.**

### Solution
```python
import tensorflow as tf
import numpy as np

# Sample data
X_train = np.random.randn(1000, 10).astype(np.float32)
y_train = np.random.randint(0, 2, (1000, 1)).astype(np.float32)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Loss and optimizer
loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Update metrics
    train_loss.update_state(loss)
    train_accuracy.update_state(y, predictions)
    
    return loss

# Dataset
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(1000).batch(32)

# Training loop
epochs = 10
for epoch in range(epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()
    
    for x_batch, y_batch in dataset:
        train_step(x_batch, y_batch)
    
    print(f"Epoch {epoch+1}: Loss={train_loss.result():.4f}, "
          f"Accuracy={train_accuracy.result():.4f}")
```

---

## Question 3
- [ ] Done

**Implement a CNN for image classification.**

### Solution
```python
import tensorflow as tf

# CNN Model
model = tf.keras.Sequential([
    # First conv block
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                           input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Second conv block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Third conv block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    
    # Flatten and dense layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Load MNIST data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Train
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

---

## Question 4
- [ ] Done

**Implement an RNN/LSTM for sequence prediction.**

### Solution
```python
import tensorflow as tf
import numpy as np

# Generate sequence data
def generate_sequences(n_samples=1000, seq_length=10):
    X = np.random.randn(n_samples, seq_length, 1).astype(np.float32)
    y = np.sum(X, axis=1)  # Target: sum of sequence
    return X, y

X_train, y_train = generate_sequences()

# LSTM Model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, 
                         input_shape=(10, 1)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Bidirectional LSTM
bi_lstm_model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True),
        input_shape=(10, 1)
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(1)
])

# GRU (faster alternative)
gru_model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, return_sequences=True, input_shape=(10, 1)),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(1)
])
```

---

## Question 5
- [ ] Done

**Implement a custom loss function.**

### Solution
```python
import tensorflow as tf

# Method 1: Simple function
def custom_mse(y_true, y_pred):
    """Custom Mean Squared Error"""
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Method 2: Class-based (for configurable losses)
class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for imbalanced classification"""
    
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Binary cross entropy
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Focal weight
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        return tf.reduce_mean(alpha_weight * focal_weight * bce)
    
    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config

# Method 3: Weighted loss
def weighted_bce(class_weights):
    """Weighted Binary Cross Entropy"""
    def loss(y_true, y_pred):
        weights = y_true * class_weights[1] + (1 - y_true) * class_weights[0]
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return tf.reduce_mean(weights * bce)
    return loss

# Use custom loss
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss=FocalLoss(gamma=2.0),  # or custom_mse
    metrics=['accuracy']
)
```

---

## Question 6
- [ ] Done

**Implement a custom metric.**

### Solution
```python
import tensorflow as tf

# Method 1: Simple function
def f1_score(y_true, y_pred):
    """F1 Score as a metric"""
    y_pred = tf.round(y_pred)
    
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))
    
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    
    return 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

# Method 2: Class-based (accumulates over batches)
class F1Score(tf.keras.metrics.Metric):
    """Stateful F1 Score metric"""
    
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.round(tf.cast(y_pred, tf.float32))
        
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))
    
    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        return 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    
    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)

# Use custom metric
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', F1Score()]
)
```

---

## Question 7
- [ ] Done

**Implement data augmentation for images.**

### Solution
```python
import tensorflow as tf

# Method 1: Using Keras layers (in model)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# Include in model
model = tf.keras.Sequential([
    # Augmentation layers (only active during training)
    data_augmentation,
    
    # Model layers
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Method 2: Using tf.data.Dataset
def augment_image(image, label):
    """Apply augmentations to image"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.image.random_saturation(image, 0.9, 1.1)
    return image, label

# Apply to dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Method 3: Using ImageDataGenerator (legacy but still common)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.1
)
```

---

## Question 8
- [ ] Done

**Implement an Autoencoder.**

### Solution
```python
import tensorflow as tf

# Simple Autoencoder
class Autoencoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(latent_dim, activation='relu')
        ])
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(784, activation='sigmoid'),
            tf.keras.layers.Reshape((28, 28))
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Create model
autoencoder = Autoencoder(latent_dim=32)
autoencoder.compile(optimizer='adam', loss='mse')

# Load MNIST
(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Train (input = output)
autoencoder.fit(X_train, X_train, 
                epochs=10, 
                batch_size=256,
                validation_data=(X_test, X_test))

# Use encoder for dimensionality reduction
encoded_imgs = autoencoder.encoder(X_test[:10])
decoded_imgs = autoencoder.decoder(encoded_imgs)
```

---

## Question 9
- [ ] Done

**Implement model with multiple inputs/outputs.**

### Solution
```python
import tensorflow as tf

# Multi-input model (e.g., image + metadata)
image_input = tf.keras.Input(shape=(224, 224, 3), name='image')
meta_input = tf.keras.Input(shape=(10,), name='metadata')

# Image branch
x1 = tf.keras.layers.Conv2D(32, 3, activation='relu')(image_input)
x1 = tf.keras.layers.MaxPooling2D()(x1)
x1 = tf.keras.layers.Conv2D(64, 3, activation='relu')(x1)
x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)

# Metadata branch
x2 = tf.keras.layers.Dense(32, activation='relu')(meta_input)

# Combine
combined = tf.keras.layers.Concatenate()([x1, x2])
x = tf.keras.layers.Dense(64, activation='relu')(combined)

# Multiple outputs
class_output = tf.keras.layers.Dense(10, activation='softmax', name='class')(x)
score_output = tf.keras.layers.Dense(1, activation='linear', name='score')(x)

# Create model
model = tf.keras.Model(
    inputs=[image_input, meta_input],
    outputs=[class_output, score_output]
)

# Compile with different losses
model.compile(
    optimizer='adam',
    loss={
        'class': 'sparse_categorical_crossentropy',
        'score': 'mse'
    },
    loss_weights={'class': 1.0, 'score': 0.5},
    metrics={
        'class': 'accuracy',
        'score': 'mae'
    }
)

# Train
# model.fit(
#     {'image': X_images, 'metadata': X_meta},
#     {'class': y_class, 'score': y_score},
#     epochs=10
# )
```

---

## Question 10
- [ ] Done

**Implement learning rate scheduling.**

### Solution
```python
import tensorflow as tf
import numpy as np

# Method 1: Built-in schedules
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.9
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Method 2: Callback-based
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Method 3: Custom schedule function
def lr_schedule_fn(epoch, lr):
    """Custom learning rate schedule"""
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule_fn)

# Method 4: Warmup + decay
class WarmupDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, decay_steps):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = self.initial_lr * step / self.warmup_steps
        decay = self.initial_lr * tf.math.exp(-step / self.decay_steps)
        return tf.minimum(warmup, decay)

# Use with model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='mse'
)

# Or use callbacks during fit
# model.fit(X, y, epochs=50, callbacks=[reduce_lr, lr_scheduler])
```


---

## Question 11
- [ ] Done

**Write a TensorFlow code to create two Tensors and perform element-wise multiplication.**

**Answer:**

```python
import tensorflow as tf

# Create tensors
a = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
b = tf.constant([10, 20, 30, 40, 50], dtype=tf.float32)

# Element-wise multiplication
result1 = a * b                        # Operator overload
result2 = tf.multiply(a, b)            # Explicit function
result3 = tf.math.multiply(a, b)       # Math module

print(result1)  # tf.Tensor([10. 40. 90. 160. 250.], shape=(5,), dtype=float32)

# 2D tensors
A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
B = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

elem_mul = A * B    # Element-wise: [[5, 12], [21, 32]]
mat_mul = A @ B     # Matrix multiplication: [[19, 22], [43, 50]]
mat_mul2 = tf.matmul(A, B)  # Same as @

# Broadcasting
scalar = tf.constant(3.0)
result = a * scalar  # [3, 6, 9, 12, 15]

# Different shapes with broadcasting
matrix = tf.constant([[1, 2, 3]], dtype=tf.float32)  # (1, 3)
vector = tf.constant([[10], [20]], dtype=tf.float32) # (2, 1)
broadcast_result = matrix * vector  # (2, 3)
print(broadcast_result)
# [[10, 20, 30],
#  [20, 40, 60]]
```

> **Interview Tip:** `*` is element-wise, `@` or `tf.matmul` is matrix multiplication. Broadcasting rules in TensorFlow follow NumPy conventions.

---

## Question 12
- [ ] Done

**Implement logistic regression using TensorFlow.**

**Answer:**

```python
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load & prepare data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# === Method 1: From Scratch with GradientTape ===
n_features = X_train.shape[1]
W = tf.Variable(tf.zeros([n_features, 1]))
b = tf.Variable(tf.zeros([1]))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

def logistic_regression(X):
    z = tf.matmul(X, W) + b
    return tf.sigmoid(z)

def compute_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1, 1])
    return -tf.reduce_mean(
        y_true * tf.math.log(y_pred + 1e-7) +
        (1 - y_true) * tf.math.log(1 - y_pred + 1e-7)
    )

# Training loop
for epoch in range(100):
    with tf.GradientTape() as tape:
        y_pred = logistic_regression(X_train)
        loss = compute_loss(y_train, y_pred)
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# Evaluate
y_pred_test = logistic_regression(X_test)
accuracy = tf.reduce_mean(
    tf.cast(tf.equal(tf.round(y_pred_test[:, 0]), y_test), tf.float32)
)
print(f"Test Accuracy: {accuracy:.4f}")

# === Method 2: Keras API (simpler) ===
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(n_features,))
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
print(f"Keras Accuracy: {model.evaluate(X_test, y_test, verbose=0)[1]:.4f}")
```

> **Interview Tip:** Method 1 shows understanding of the math; Method 2 shows practical skills. Logistic regression in TF is a single Dense layer with sigmoid activation.

---

## Question 13
- [ ] Done

**Write a TensorFlow script to normalize the features of a dataset.**

**Answer:**

```python
import tensorflow as tf
import numpy as np

# Sample data
data = np.array([
    [100, 0.5, 1000],
    [200, 0.8, 2000],
    [150, 0.3, 1500],
    [300, 0.9, 3000]
], dtype=np.float32)

# === Method 1: Manual Min-Max Normalization ===
def min_max_normalize(tensor):
    min_vals = tf.reduce_min(tensor, axis=0)
    max_vals = tf.reduce_max(tensor, axis=0)
    return (tensor - min_vals) / (max_vals - min_vals + 1e-8)

normalized = min_max_normalize(data)
print("Min-Max Normalized:\n", normalized.numpy())

# === Method 2: Manual Z-Score Standardization ===
def standardize(tensor):
    mean = tf.reduce_mean(tensor, axis=0)
    std = tf.math.reduce_std(tensor, axis=0)
    return (tensor - mean) / (std + 1e-8)

standardized = standardize(data)
print("Standardized:\n", standardized.numpy())

# === Method 3: Keras Normalization Layer (Recommended) ===
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(data)  # Computes mean and std from data
normalized_keras = normalizer(data)

# In a model pipeline
model = tf.keras.Sequential([
    tf.keras.layers.Normalization(axis=-1),  # Auto-normalize
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.layers[0].adapt(X_train)  # Fit on training data only

# === Method 4: tf.data pipeline normalization ===
def normalize_fn(features, labels):
    mean = tf.constant([175.0, 0.625, 1875.0])  # Precomputed
    std = tf.constant([70.7, 0.234, 707.1])
    features = (features - mean) / std
    return features, labels

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.map(normalize_fn).batch(32)
```

| Method | Use Case | Advantage |
|--------|----------|----------|
| Manual Min-Max | Custom range [0,1] | Full control |
| Manual Z-Score | Standard distribution | Mathematical transparency |
| `Normalization` layer | Keras models | Integrated in model, no leakage |
| `tf.data.map()` | Data pipeline | Efficient, on-the-fly |

> **Interview Tip:** Use the **Keras Normalization layer** in production — it stores mean/std in the model, preventing train/test leakage and simplifying deployment.

---

## Question 14
- [ ] Done

**Write a Python function using TensorFlow to compute the gradient of a given function.**

**Answer:**

```python
import tensorflow as tf

# === 1. Basic Gradient Computation ===
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2 + 2 * x + 1  # y = x² + 2x + 1

dy_dx = tape.gradient(y, x)  # dy/dx = 2x + 2 = 8.0
print(f"dy/dx at x=3: {dy_dx.numpy()}")  # 8.0

# === 2. Multiple Variables ===
w = tf.Variable(2.0)
b = tf.Variable(1.0)

with tf.GradientTape() as tape:
    y = w * 5.0 + b     # y = 5w + b

grads = tape.gradient(y, [w, b])
print(f"dy/dw: {grads[0].numpy()}")  # 5.0
print(f"dy/db: {grads[1].numpy()}")  # 1.0

# === 3. Higher-Order Gradients ===
x = tf.Variable(3.0)

with tf.GradientTape() as outer:
    with tf.GradientTape() as inner:
        y = x ** 3       # y = x³
    dy_dx = inner.gradient(y, x)   # dy/dx = 3x² = 27
d2y_dx2 = outer.gradient(dy_dx, x)  # d²y/dx² = 6x = 18
print(f"Second derivative: {d2y_dx2.numpy()}")  # 18.0

# === 4. Gradient Function (Reusable) ===
def compute_gradient(func, x_val):
    """Compute gradient of func at x_val."""
    x = tf.Variable(float(x_val))
    with tf.GradientTape() as tape:
        y = func(x)
    return tape.gradient(y, x).numpy()

# Usage
print(compute_gradient(lambda x: tf.sin(x), 0.0))         # 1.0 (cos(0))
print(compute_gradient(lambda x: tf.exp(x), 1.0))         # 2.718 (e¹)
print(compute_gradient(lambda x: x**3 - 2*x + 1, 2.0))    # 10.0

# === 5. Persistent Tape (use gradient multiple times) ===
x = tf.Variable(3.0)

with tf.GradientTape(persistent=True) as tape:
    y = x ** 2
    z = x ** 3

dy_dx = tape.gradient(y, x)  # 6.0
dz_dx = tape.gradient(z, x)  # 27.0
del tape  # Must delete persistent tape
```

| Feature | Description |
|---------|------------|
| `GradientTape()` | Records operations for auto-differentiation |
| `persistent=True` | Allow multiple `.gradient()` calls |
| `tape.watch(tensor)` | Track non-Variable tensors |
| Nested tapes | Higher-order derivatives |

> **Interview Tip:** `GradientTape` is TensorFlow's automatic differentiation engine. It's used in custom training loops. The tape is consumed after one `.gradient()` call unless `persistent=True`.

---

## Question 15
- [ ] Done

**Develop a code to save and load a trained TensorFlow model.**

**Answer:**

```python
import tensorflow as tf

# Build and train a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=10)

# === Method 1: SavedModel Format (Recommended) ===
model.save('saved_model/my_model')       # Directory
loaded = tf.keras.models.load_model('saved_model/my_model')

# === Method 2: HDF5 (.h5) Format ===
model.save('my_model.h5')
loaded = tf.keras.models.load_model('my_model.h5')

# === Method 3: Weights Only ===
model.save_weights('weights/my_weights')  # Save weights
model.load_weights('weights/my_weights')  # Load into same architecture

# === Method 4: Checkpoints During Training ===
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    'checkpoints/model_{epoch:02d}_{val_loss:.4f}.h5',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)
# model.fit(X_train, y_train, callbacks=[checkpoint_cb], validation_split=0.2)

# === Method 5: TFLite (for deployment) ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# === Method 6: Export with custom signature ===
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 10], dtype=tf.float32)])
def serve(x):
    return model(x, training=False)

tf.saved_model.save(model, 'export/', signatures={'serving_default': serve})
```

| Format | Saves | Size | Use Case |
|--------|-------|------|----------|
| SavedModel | Everything (graph+weights+optimizer) | Largest | Full reproducibility |
| HDF5 (.h5) | Architecture+weights+optimizer | Medium | Quick save/share |
| Weights only | Just weights | Smallest | Same architecture transfer |
| TFLite | Optimized inference graph | Smallest | Mobile/edge |
| Checkpoint | Weights at intervals | Medium | Training recovery |

> **Interview Tip:** Use **SavedModel** for production (includes computation graph). Use **checkpoints** during training to recover from crashes. Use **TFLite** for mobile deployment.

---

## Question 16
- [ ] Done

**Code a TensorFlow program that uses dataset shuffling , repetition , and batching**

**Answer:**

```python
import tensorflow as tf
import numpy as np

# Create sample data
X = np.arange(20).reshape(10, 2).astype(np.float32)
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# === Complete Pipeline ===
dataset = tf.data.Dataset.from_tensor_slices((X, y))

train_ds = (
    dataset
    .shuffle(buffer_size=10, seed=42)     # Randomize order
    .repeat(3)                            # Repeat dataset 3 times (3 epochs)
    .batch(4)                             # Group into batches of 4
    .prefetch(tf.data.AUTOTUNE)           # Overlap data prep with training
)

# Iterate and inspect
for i, (batch_x, batch_y) in enumerate(train_ds):
    print(f"Batch {i}: X shape={batch_x.shape}, y={batch_y.numpy()}")

# === Order Matters! ===
# Correct: shuffle -> repeat -> batch (different shuffle each epoch)
correct = dataset.shuffle(10).repeat(3).batch(4)

# Also valid: repeat -> shuffle -> batch (shuffles across epoch boundaries)
alternative = dataset.repeat(3).shuffle(10).batch(4)

# === Advanced Pipeline with Augmentation ===
def augment(features, label):
    features = features + tf.random.normal(tf.shape(features), stddev=0.01)
    return features, label

train_ds = (
    dataset
    .cache()                              # Cache in memory after first read
    .shuffle(buffer_size=len(X))          # Full shuffle
    .batch(4)
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)  # Parallel augmentation
    .prefetch(tf.data.AUTOTUNE)
)

# === Without repeat (use epochs in model.fit) ===
# Better approach: let fit() handle epochs
train_ds = dataset.shuffle(10).batch(4).prefetch(tf.data.AUTOTUNE)
# model.fit(train_ds, epochs=3)  # 3 epochs handled by fit()

# === Reading from files ===
file_ds = tf.data.Dataset.list_files('data/*.csv')
parsed_ds = file_ds.interleave(
    lambda f: tf.data.TextLineDataset(f).skip(1),
    num_parallel_calls=tf.data.AUTOTUNE
).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
```

| Operation | Purpose | Key Parameter |
|-----------|---------|---------------|
| `.shuffle(N)` | Randomize samples | `buffer_size` (larger = better shuffle) |
| `.repeat(N)` | Repeat dataset | `N` (None = infinite) |
| `.batch(N)` | Create mini-batches | Batch size |
| `.prefetch(N)` | Overlap CPU/GPU | `AUTOTUNE` (auto-optimal) |
| `.cache()` | Cache processed data | After decode, before augment |

> **Interview Tip:** The standard pipeline order is: `cache -> shuffle -> batch -> map (augmentation) -> prefetch`. Place `cache()` before random augmentations so cached data gets different augmentations each epoch. Use `shuffle -> repeat` (not `repeat -> shuffle`) for better randomness across epochs.

---

## Question 17
- [ ] Done

**How would you implement attention mechanisms in TensorFlow?**

**Answer:**

```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# === 1. Scaled Dot-Product Attention (Core) ===
def scaled_dot_product_attention(Q, K, V, mask=None):
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V"""
    d_k = tf.cast(tf.shape(K)[-1], tf.float32)
    scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)
    
    if mask is not None:
        scores += (mask * -1e9)  # Mask positions with large negative
    
    weights = tf.nn.softmax(scores, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights

# Test
Q = tf.random.normal((1, 4, 64))  # (batch, seq_len, d_k)
K = tf.random.normal((1, 4, 64))
V = tf.random.normal((1, 4, 64))
output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Output: {output.shape}, Weights: {weights.shape}")

# === 2. Multi-Head Attention (from scratch) ===
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, heads, seq, depth)
    
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)
        
        attention, weights = scaled_dot_product_attention(q, k, v, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concatenated = tf.reshape(attention, (batch_size, -1, self.d_model))
        return self.dense(concatenated)

# === 3. Using Built-in Keras Layer (Production) ===
attention_layer = layers.MultiHeadAttention(num_heads=8, key_dim=64)

inputs = layers.Input(shape=(100, 256))  # (seq_len, embed_dim)
attn_output = attention_layer(inputs, inputs)  # Self-attention (Q=K=V)
output = layers.LayerNormalization()(inputs + attn_output)  # Residual + norm

# === 4. Bahdanau (Additive) Attention ===
class BahdanauAttention(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)
    
    def call(self, query, values):
        # query: (batch, hidden), values: (batch, seq_len, hidden)
        query_expanded = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_expanded) + self.W2(values)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * values, axis=1)
        return context, weights

# === 5. Transformer Block ===
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model)
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)
    
    def call(self, x, training=False):
        attn_out = self.attn(x, x)                   # Self-attention
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.drop2(ffn_out, training=training))
```

| Attention Type | Formula | Use Case |
|---------------|---------|----------|
| Scaled Dot-Product | $\text{softmax}(QK^T / \sqrt{d_k}) \cdot V$ | Transformer core |
| Multi-Head | Parallel dot-product heads | Modern NLP/Vision |
| Bahdanau (Additive) | $V^T \tanh(W_1Q + W_2K)$ | Seq2Seq translation |
| Luong (Multiplicative) | $Q^T W K$ | Simpler Seq2Seq |

> **Interview Tip:** Multi-head attention lets the model attend to different representation subspaces simultaneously. Self-attention (Q=K=V from same input) captures intra-sequence relationships; cross-attention (Q from decoder, K/V from encoder) is used in translation tasks.

## Question 18
- [ ] Done

**Build a simple convolutional neural network in TensorFlow for image classification**

### Solution

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# Build CNN
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    # Classifier
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, batch_size=64,
                    validation_split=0.1,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

> **Interview Tip:** Always include **BatchNormalization** and **Dropout** to regularize CNNs. Use `padding='same'` to preserve spatial dimensions. EarlyStopping prevents overfitting.

---

## Question 19
- [ ] Done

**Create a recurrent neural network in TensorFlow to process sequential data**

### Solution

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

# Generate synthetic time-series data
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Sine wave prediction
t = np.linspace(0, 100, 1000)
data = np.sin(t)
seq_length = 50
X, y = create_sequences(data, seq_length)
X = X.reshape(-1, seq_length, 1)  # (samples, timesteps, features)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build RNN model with LSTM
model = models.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
    layers.Dropout(0.2),
    layers.LSTM(32, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_split=0.1,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,
                               restore_best_weights=True)])
print(f"Test MAE: {model.evaluate(X_test, y_test)[1]:.4f}")
```

> **Interview Tip:** Use **LSTM** or **GRU** over vanilla RNN to avoid vanishing gradients. Stack multiple LSTM layers with `return_sequences=True` on all but the last recurrent layer.

---

## Question 20
- [ ] Done

**Implement a custom training loop in TensorFlow for a basic neural network**

### Solution

```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Simple dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(64)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)

# Build model
model = tf.keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Metrics
train_loss = tf.keras.metrics.Mean()
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()

# Custom training loop
for epoch in range(10):
    train_loss.reset_state()
    train_acc.reset_state()

    for X_batch, y_batch in train_ds:
        with tf.GradientTape() as tape:
            logits = model(X_batch, training=True)
            loss = loss_fn(y_batch, logits)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss.update_state(loss)
        train_acc.update_state(y_batch, logits)

    print(f"Epoch {epoch+1}: Loss={train_loss.result():.4f}, Acc={train_acc.result():.4f}")

# Evaluate
test_acc = tf.keras.metrics.SparseCategoricalAccuracy()
for X_batch, y_batch in test_ds:
    logits = model(X_batch, training=False)
    test_acc.update_state(y_batch, logits)
print(f"Test Accuracy: {test_acc.result():.4f}")
```

> **Interview Tip:** Custom training loops use `tf.GradientTape()` for automatic differentiation. Use `training=True/False` to toggle dropout/batch norm behavior. This approach gives full control over gradient manipulation and mixed-precision training.

---

## Question 21
- [ ] Done

**Write a TensorFlow function for data augmentation on an image dataset**

### Solution

```python
import tensorflow as tf
from tensorflow.keras import layers

# Method 1: Using Keras preprocessing layers (recommended for TF 2.x)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.RandomTranslation(0.1, 0.1),
])

# Integrate into model
model = tf.keras.Sequential([
    data_augmentation,
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Method 2: Using ImageDataGenerator (legacy but widely used)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)

# Method 3: Using tf.image for custom augmentation in tf.data pipeline
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(1000).map(augment).batch(32).prefetch(tf.data.AUTOTUNE)
```

> **Interview Tip:** Prefer **Keras preprocessing layers** (Method 1) in TF 2.x as augmentation runs on GPU and integrates into the model. Augmentation only applies during training, not inference.

---
