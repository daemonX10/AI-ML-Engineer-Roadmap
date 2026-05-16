# Keras Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Implement a CNN for image classification.**

### Solution
```python
from tensorflow import keras
from tensorflow.keras import layers

# CNN for MNIST
model = keras.Sequential([
    # First conv block
    layers.Conv2D(32, (3, 3), activation='relu', 
                  input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Second conv block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Third conv block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    
    # Classifier
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Load and preprocess MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Train
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

---

## Question 2
- [ ] Done

**Implement an LSTM for sequence classification.**

### Solution
```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Sample: Sentiment classification
vocab_size = 10000
max_length = 200
embedding_dim = 128

model = keras.Sequential([
    # Embedding layer
    layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    
    # LSTM layers
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32),
    layers.Dropout(0.2),
    
    # Classifier
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Load IMDB dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(
    num_words=vocab_size
)

# Pad sequences
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_length)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_length)

# Train
model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Bidirectional LSTM (better for many tasks)
bi_lstm_model = keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(1, activation='sigmoid')
])
```

---

## Question 3
- [ ] Done

**Implement a model with multiple inputs using Functional API.**

### Solution
```python
from tensorflow import keras
from tensorflow.keras import layers

# Multi-input model: Image + Metadata

# Image input branch
image_input = keras.Input(shape=(64, 64, 3), name='image')
x1 = layers.Conv2D(32, 3, activation='relu')(image_input)
x1 = layers.MaxPooling2D()(x1)
x1 = layers.Conv2D(64, 3, activation='relu')(x1)
x1 = layers.GlobalAveragePooling2D()(x1)

# Metadata input branch
meta_input = keras.Input(shape=(10,), name='metadata')
x2 = layers.Dense(32, activation='relu')(meta_input)
x2 = layers.Dense(16, activation='relu')(x2)

# Merge branches
merged = layers.Concatenate()([x1, x2])
x = layers.Dense(64, activation='relu')(merged)
x = layers.Dropout(0.3)(x)

# Output
output = layers.Dense(5, activation='softmax', name='class')(x)

# Create model
model = keras.Model(
    inputs=[image_input, meta_input],
    outputs=output
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Training with multiple inputs
# model.fit(
#     {'image': X_images, 'metadata': X_meta},
#     y_labels,
#     epochs=10
# )
```

---

## Question 4
- [ ] Done

**Implement a custom training loop in Keras.**

### Solution
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Loss and optimizer
loss_fn = keras.losses.BinaryCrossentropy()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Metrics
train_loss = keras.metrics.Mean(name='train_loss')
train_acc = keras.metrics.BinaryAccuracy(name='train_accuracy')
val_loss = keras.metrics.Mean(name='val_loss')
val_acc = keras.metrics.BinaryAccuracy(name='val_accuracy')

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss.update_state(loss)
    train_acc.update_state(y, predictions)

@tf.function
def val_step(x, y):
    predictions = model(x, training=False)
    loss = loss_fn(y, predictions)
    
    val_loss.update_state(loss)
    val_acc.update_state(y, predictions)

# Training loop
epochs = 10
batch_size = 32

# Sample data
X_train = np.random.randn(1000, 10).astype(np.float32)
y_train = np.random.randint(0, 2, 1000).astype(np.float32)
X_val = np.random.randn(200, 10).astype(np.float32)
y_val = np.random.randint(0, 2, 200).astype(np.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(1000).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(batch_size)

for epoch in range(epochs):
    # Reset metrics
    train_loss.reset_states()
    train_acc.reset_states()
    val_loss.reset_states()
    val_acc.reset_states()
    
    # Training
    for x_batch, y_batch in train_dataset:
        train_step(x_batch, y_batch)
    
    # Validation
    for x_batch, y_batch in val_dataset:
        val_step(x_batch, y_batch)
    
    print(f"Epoch {epoch+1}: "
          f"Loss={train_loss.result():.4f}, Acc={train_acc.result():.4f}, "
          f"Val_Loss={val_loss.result():.4f}, Val_Acc={val_acc.result():.4f}")
```

---

## Question 5
- [ ] Done

**Implement an Autoencoder.**

### Solution
```python
from tensorflow import keras
from tensorflow.keras import layers

# Encoder
encoder_inputs = keras.Input(shape=(784,))
x = layers.Dense(256, activation='relu')(encoder_inputs)
x = layers.Dense(128, activation='relu')(x)
latent = layers.Dense(32, activation='relu')(x)  # Bottleneck

encoder = keras.Model(encoder_inputs, latent, name='encoder')

# Decoder
decoder_inputs = keras.Input(shape=(32,))
x = layers.Dense(128, activation='relu')(decoder_inputs)
x = layers.Dense(256, activation='relu')(x)
decoder_outputs = layers.Dense(784, activation='sigmoid')(x)

decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')

# Autoencoder
autoencoder_inputs = keras.Input(shape=(784,))
encoded = encoder(autoencoder_inputs)
decoded = decoder(encoded)

autoencoder = keras.Model(autoencoder_inputs, decoded, name='autoencoder')
autoencoder.compile(optimizer='adam', loss='mse')

# Train on MNIST
(X_train, _), (X_test, _) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

autoencoder.fit(
    X_train, X_train,  # Input = Output
    epochs=20,
    batch_size=256,
    validation_data=(X_test, X_test)
)

# Use encoder for dimensionality reduction
encoded_imgs = encoder.predict(X_test[:100])
print(f"Encoded shape: {encoded_imgs.shape}")  # (100, 32)

# Reconstruct images
reconstructed = autoencoder.predict(X_test[:10])
```

---

## Question 6
- [ ] Done

**Implement learning rate scheduling.**

### Solution
```python
from tensorflow import keras
import numpy as np

# Method 1: ReduceLROnPlateau callback
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,        # Reduce LR by half
    patience=3,        # Wait 3 epochs
    min_lr=1e-6,
    verbose=1
)

# Method 2: LearningRateScheduler callback
def schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * np.exp(-0.1)

lr_scheduler = keras.callbacks.LearningRateScheduler(schedule)

# Method 3: Built-in schedules
initial_lr = 0.01

# Exponential decay
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=1000,
    decay_rate=0.9
)

# Cosine decay
cosine_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_lr,
    decay_steps=10000
)

# Use schedule with optimizer
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# Method 4: Warmup schedule
class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, decay_steps):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
    
    def __call__(self, step):
        warmup = self.initial_lr * step / self.warmup_steps
        decay = self.initial_lr * (1 - step / self.decay_steps)
        return tf.minimum(warmup, decay)

# Use
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])
model.compile(optimizer=optimizer, loss='mse')
```

---

## Question 7
- [ ] Done

**Implement a Siamese network for similarity learning.**

### Solution
```python
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# Shared encoder
def create_encoder(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    return keras.Model(inputs, x, name='encoder')

# Create shared encoder
encoder = create_encoder((28, 28, 1))

# Siamese network
input_a = keras.Input(shape=(28, 28, 1), name='input_a')
input_b = keras.Input(shape=(28, 28, 1), name='input_b')

# Get embeddings (shared weights)
embedding_a = encoder(input_a)
embedding_b = encoder(input_b)

# Compute distance
distance = layers.Lambda(
    lambda x: tf.abs(x[0] - x[1])
)([embedding_a, embedding_b])

# Classifier
output = layers.Dense(1, activation='sigmoid')(distance)

siamese_model = keras.Model(
    inputs=[input_a, input_b],
    outputs=output
)

siamese_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Contrastive loss (alternative)
def contrastive_loss(y_true, y_pred, margin=1.0):
    """Contrastive loss for siamese networks"""
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# Use: pairs of similar (1) and dissimilar (0) samples
# siamese_model.fit([X1, X2], y_similar, epochs=10)
```

---

## Question 8
- [ ] Done

**Implement model ensembling in Keras.**

### Solution
```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Create multiple models
def create_model(seed):
    np.random.seed(seed)
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train ensemble
n_models = 5
models = []

for i in range(n_models):
    model = create_model(seed=i)
    # model.fit(X_train, y_train, epochs=10, verbose=0)
    models.append(model)

# Ensemble prediction (averaging)
def ensemble_predict(models, X):
    predictions = [model.predict(X) for model in models]
    avg_predictions = np.mean(predictions, axis=0)
    return np.argmax(avg_predictions, axis=1)

# Voting ensemble
def voting_predict(models, X):
    predictions = [np.argmax(model.predict(X), axis=1) for model in models]
    predictions = np.array(predictions)
    # Majority voting
    from scipy import stats
    ensemble_pred, _ = stats.mode(predictions, axis=0)
    return ensemble_pred.flatten()

# Keras ensemble model
def create_ensemble_model(models, input_shape):
    inputs = keras.Input(shape=input_shape)
    outputs = [model(inputs) for model in models]
    
    # Average predictions
    averaged = layers.Average()(outputs)
    
    ensemble = keras.Model(inputs=inputs, outputs=averaged)
    return ensemble

# Usage
# ensemble = create_ensemble_model(models, (10,))
# predictions = ensemble.predict(X_test)
```

---

## Question 9
- [ ] Done

**Implement attention mechanism for sequence models.**

### Solution
```python
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# Simple self-attention layer
class SelfAttention(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
    
    def call(self, x):
        # x shape: (batch, seq_len, features)
        score = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        context = tf.reduce_sum(x * tf.expand_dims(attention_weights, -1), axis=1)
        return context

# Model with attention
vocab_size = 10000
embedding_dim = 128
max_length = 100

model = keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    layers.LSTM(64, return_sequences=True),
    SelfAttention(64),  # Custom attention
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Using built-in MultiHeadAttention
inputs = keras.Input(shape=(max_length,))
x = layers.Embedding(vocab_size, embedding_dim)(inputs)
x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

attention_model = keras.Model(inputs, outputs)
attention_model.compile(optimizer='adam', loss='binary_crossentropy')
```

---

## Question 10
- [ ] Done

**Implement a GAN (Generative Adversarial Network).**

### Solution
```python
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

# Generator
def build_generator(latent_dim):
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(latent_dim,)),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(784, activation='sigmoid'),
        layers.Reshape((28, 28, 1))
    ])
    return model

# Discriminator
def build_discriminator():
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Build GAN
latent_dim = 100
generator = build_generator(latent_dim)
discriminator = build_discriminator()

discriminator.compile(
    optimizer=keras.optimizers.Adam(0.0002, 0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Combined model
discriminator.trainable = False
gan_input = keras.Input(shape=(latent_dim,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)

gan = keras.Model(gan_input, gan_output)
gan.compile(
    optimizer=keras.optimizers.Adam(0.0002, 0.5),
    loss='binary_crossentropy'
)

# Training
def train_gan(epochs, batch_size=128):
    (X_train, _), (_, _) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = generator.predict(noise, verbose=0)
        
        d_loss_real = discriminator.train_on_batch(real_imgs, real)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D_loss={d_loss[0]:.4f}, G_loss={g_loss:.4f}")

# train_gan(epochs=1000)
```

---

## Question 11
- [ ] Done

**Create a simple Keras model using the Sequential API for binary classification.**

**Answer:**

```python
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# === 1. Load & Prepare Data ===
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 2. Build Model ===
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

# === 3. Compile ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# === 4. Train ===
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
)

# === 5. Evaluate ===
loss, acc, auc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}, AUC: {auc:.4f}")

# === 6. Plot Training History ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['loss'], label='Train')
ax1.plot(history.history['val_loss'], label='Validation')
ax1.set_title('Loss'); ax1.legend()
ax2.plot(history.history['accuracy'], label='Train')
ax2.plot(history.history['val_accuracy'], label='Validation')
ax2.set_title('Accuracy'); ax2.legend()
plt.show()
```

> **Interview Tip:** For binary classification: output layer uses 1 neuron + sigmoid activation, loss is `binary_crossentropy`. For multi-class: N neurons + softmax, loss is `categorical_crossentropy`.

---

## Question 12
- [ ] Done

**Write a script to load and preprocess image data for a CNN in Keras.**

**Answer:**

```python
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# === Method 1: From Directory ===
train_ds = tf.keras.utils.image_dataset_from_directory(
    'data/train/',
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical',    # 'int', 'binary', 'categorical'
    validation_split=0.2,
    subset='training',
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'data/train/',
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical',
    validation_split=0.2,
    subset='validation',
    seed=42
)

# === Preprocessing Pipeline ===
# Normalization
normalize = layers.Rescaling(1./255)

# Data Augmentation
augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.1, 0.1)
])

# Apply preprocessing
train_ds = train_ds.map(lambda x, y: (augmentation(normalize(x), training=True), y))
val_ds = val_ds.map(lambda x, y: (normalize(x), y))

# Optimize performance
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

# === Method 2: From NumPy (e.g., CIFAR-10) ===
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# === Visualize ===
for images, labels in train_ds.take(1):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].numpy())
        ax.set_title(f"Label: {labels[i].numpy().argmax()}")
        ax.axis('off')
    plt.show()
```

| Step | Purpose | Method |
|------|---------|--------|
| Resize | Uniform input size | `image_size=(224, 224)` |
| Normalize | Scale to [0,1] | `Rescaling(1./255)` |
| Augment | Increase diversity | `RandomFlip`, `RandomRotation` |
| Cache | Speed up repeated access | `.cache()` |
| Prefetch | Overlap I/O with compute | `.prefetch(AUTOTUNE)` |

> **Interview Tip:** Apply augmentation only to training data, not validation/test. Use `cache()` before augmentation so different augmentations are applied each epoch.

---

## Question 13
- [ ] Done

**Code a Multi-Layer Perceptron (MLP) in Keras for a regression task.**

**Answer:**

```python
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# === 1. Load Data ===
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# === 2. Normalize Features ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 3. Build MLP ===
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Linear activation for regression (no activation)
])

# === 4. Compile ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
)

# === 5. Train ===
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    callbacks=[
        callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=7, factor=0.5, min_lr=1e-6)
    ]
)

# === 6. Evaluate ===
results = model.evaluate(X_test, y_test)
print(f"Test MSE: {results[0]:.4f}, MAE: {results[1]:.4f}, RMSE: {results[2]:.4f}")

# === 7. Predict ===
predictions = model.predict(X_test[:5])
for pred, actual in zip(predictions.flatten(), y_test[:5]):
    print(f"Predicted: {pred:.2f}, Actual: {actual:.2f}")
```

| Regression vs Classification | Difference |
|-------|--------|
| Output activation | None/linear (regression) vs sigmoid/softmax |
| Loss function | MSE/MAE (regression) vs crossentropy |
| Output neurons | 1 (single target) |
| Metrics | MAE, RMSE, R-squared |

> **Interview Tip:** For regression, the output layer should have **no activation function** (linear output). Using sigmoid/softmax on regression output is a common mistake that bounds predictions to [0,1].

---

## Question 14
- [ ] Done

**Develop a custom callback in Keras that logs the predictions of a model at the end of each epoch.**

**Answer:**

```python
import tensorflow as tf
from tensorflow.keras import callbacks
import numpy as np
import json

class PredictionLogger(callbacks.Callback):
    def __init__(self, validation_data, n_samples=5, log_file='predictions.json'):
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.n_samples = n_samples
        self.log_file = log_file
        self.logs = []
    
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions on sample
        predictions = self.model.predict(self.X_val[:self.n_samples], verbose=0)
        
        epoch_log = {
            'epoch': epoch + 1,
            'predictions': predictions.tolist(),
            'actuals': self.y_val[:self.n_samples].tolist(),
            'train_loss': float(logs.get('loss', 0)),
            'val_loss': float(logs.get('val_loss', 0))
        }
        self.logs.append(epoch_log)
        
        # Print summary
        print(f"\nEpoch {epoch+1} Sample Predictions:")
        for i in range(self.n_samples):
            pred = predictions[i].flatten()
            actual = self.y_val[i]
            print(f"  Sample {i}: Pred={pred}, Actual={actual}")
    
    def on_train_end(self, logs=None):
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f"\nPrediction logs saved to {self.log_file}")

# === Usage ===
logger = PredictionLogger(
    validation_data=(X_val, y_val),
    n_samples=5,
    log_file='pred_logs.json'
)

model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[logger]
)

# === Advanced: Log with Distribution Stats ===
class DistributionLogger(callbacks.Callback):
    def __init__(self, X_val):
        super().__init__()
        self.X_val = X_val
    
    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.X_val, verbose=0)
        print(f"\nPred stats - Mean: {preds.mean():.4f}, "
              f"Std: {preds.std():.4f}, "
              f"Min: {preds.min():.4f}, Max: {preds.max():.4f}")
```

| Callback Hook | When Called |
|--------------|-------------|
| `on_train_begin` | Start of training |
| `on_epoch_begin` | Start of each epoch |
| `on_batch_end` | End of each batch |
| `on_epoch_end` | End of each epoch |
| `on_train_end` | End of training |

> **Interview Tip:** Custom callbacks are powerful for monitoring training beyond standard metrics. Use `self.model` inside the callback to access the model being trained. Set `verbose=0` in `predict()` inside callbacks to avoid cluttering output.

---

## Question 15
- [ ] Done

**Implement a Keras data generator to handle large datasets that cannot fit into memory.**

**Answer:**

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import Sequence

# === Method 1: Custom Sequence Generator ===
class DataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size=32, dim=(224, 224, 3),
                 shuffle=True, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(file_paths))
        self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.file_paths) / self.batch_size))
    
    def __getitem__(self, index):
        """Get one batch"""
        batch_idx = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.array([self._load_image(self.file_paths[i]) for i in batch_idx])
        y = np.array([self.labels[i] for i in batch_idx])
        return X, y
    
    def on_epoch_end(self):
        """Shuffle after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _load_image(self, path):
        img = tf.keras.utils.load_img(path, target_size=self.dim[:2])
        img = tf.keras.utils.img_to_array(img) / 255.0
        if self.augment:
            img = tf.image.random_flip_left_right(img).numpy()
        return img

# Usage
train_gen = DataGenerator(train_paths, train_labels, batch_size=32, augment=True)
val_gen = DataGenerator(val_paths, val_labels, batch_size=32, augment=False)

model.fit(train_gen, validation_data=val_gen, epochs=50, workers=4, use_multiprocessing=True)

# === Method 2: tf.data.Dataset (Preferred) ===
def parse_image(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
train_ds = (
    dataset
    .shuffle(10000)
    .map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

# === Method 3: TFRecord for Maximum Performance ===
def parse_tfrecord(example):
    features = tf.io.parse_single_example(example, {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    })
    img = tf.io.decode_jpeg(features['image'])
    img = tf.image.resize(img, [224, 224]) / 255.0
    return img, features['label']

ds = tf.data.TFRecordDataset('data.tfrecord').map(parse_tfrecord).batch(32)
```

| Method | Speed | Memory | Multiprocessing |
|--------|-------|--------|----------------|
| `Sequence` generator | Medium | Low | `workers=N` |
| `tf.data.Dataset` | Fast | Low | `AUTOTUNE` |
| TFRecord | Fastest | Low | Built-in |

> **Interview Tip:** Prefer `tf.data.Dataset` over `Sequence` generators for new projects. TFRecords are the fastest option for very large datasets as they enable sequential disk reads.

---

## Question 16
- [ ] Done

**Write a Python function using Keras to calculate and display a confusion matrix for a classification model.**

**Answer:**

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(model, X_test, y_test, class_names=None):
    """
    Calculate and display confusion matrix for a Keras model.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: True labels (one-hot or integer)
        class_names: List of class names
    """
    # Get predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Convert one-hot to integer if needed
    if len(y_test.shape) > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalized version
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names)
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_ylabel('Actual'); ax1.set_xlabel('Predicted')
    
    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=ax2,
                xticklabels=class_names, yticklabels=class_names)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_ylabel('Actual'); ax2.set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return cm

# === Usage ===
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

class_names = [str(i) for i in range(10)]
cm = plot_confusion_matrix(model, X_test, y_test, class_names)
```

> **Interview Tip:** Always show both raw counts and normalized confusion matrices. Normalized matrices reveal per-class accuracy, which is crucial for imbalanced datasets.

---

## Question 17
- [ ] Done

**Create a script that fine-tunes a pre-trained convolutional neural network on a new dataset in Keras.**

**Answer:**

```python
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# === 1. Load Pre-trained Model ===
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,              # Remove classification head
    input_shape=(224, 224, 3)
)

# === 2. Freeze Base Model ===
base_model.trainable = False

# === 3. Add Custom Head ===
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 classes
])

# === 4. Phase 1: Train Top Layers Only ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_ds, validation_data=val_ds, epochs=10,
    callbacks=[callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

# === 5. Phase 2: Fine-tune Top Layers of Base ===
base_model.trainable = True

# Freeze all layers except last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

print(f"Trainable layers: {len([l for l in model.layers if l.trainable])}")

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # 100x lower!
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_ds, validation_data=val_ds, epochs=20,
    callbacks=[
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
    ]
)

# === 6. Data Preprocessing for Pre-trained Models ===
# Each model has its own preprocessing!
preprocess = tf.keras.applications.resnet50.preprocess_input
train_ds = train_ds.map(lambda x, y: (preprocess(x), y))

# === Available Pre-trained Models ===
# tf.keras.applications.VGG16
# tf.keras.applications.ResNet50
# tf.keras.applications.EfficientNetB0
# tf.keras.applications.MobileNetV2
```

| Phase | What Trains | Learning Rate | Epochs |
|-------|------------|---------------|--------|
| Phase 1 | Custom head only | 1e-3 (high) | 5-10 |
| Phase 2 | Head + top base layers | 1e-5 (very low) | 10-20 |

> **Interview Tip:** The two-phase approach is critical: first train the new head with a high LR (frozen base), then fine-tune upper base layers with a very low LR. Using a high LR during fine-tuning destroys pre-trained features.

---

## Question 18
- [ ] Done

**Implement custom training logic in Keras by overriding the training step function**

### Solution

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

class CustomModel(keras.Model):
    def __init__(self, units=64, num_classes=10):
        super().__init__()
        self.dense1 = keras.layers.Dense(units, activation='relu')
        self.bn = keras.layers.BatchNormalization()
        self.dropout = keras.layers.Dropout(0.3)
        self.dense2 = keras.layers.Dense(num_classes)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_metric = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)
        return self.dense2(x)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.loss_fn(y, y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.acc_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_metric.result()}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.loss_fn(y, y_pred)
        self.loss_tracker.update_state(loss)
        self.acc_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_metric.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_metric]

# Usage
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype("float32") / 255.0

model = CustomModel(units=128, num_classes=10)
model.compile(optimizer="adam")
model.fit(X_train, y_train, batch_size=64, epochs=5, validation_split=0.1)
```

> **Interview Tip:** Override `train_step()` when you need custom loss logic (e.g., GANs, multi-task learning). Override `test_step()` for custom evaluation. Always define tracked metrics in the `metrics` property.

---

## Question 19
- [ ] Done

**Use the Keras functional API to create a model with shared layers and multiple inputs/outputs**

### Solution

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Shared embedding layer (e.g., for Siamese networks)
shared_embedding = layers.Dense(64, activation='relu', name='shared_embed')

# Input A
input_a = layers.Input(shape=(100,), name='input_a')
encoded_a = shared_embedding(input_a)
encoded_a = layers.Dropout(0.3)(encoded_a)

# Input B (uses SAME shared layer - weights are tied)
input_b = layers.Input(shape=(100,), name='input_b')
encoded_b = shared_embedding(input_b)
encoded_b = layers.Dropout(0.3)(encoded_b)

# Merge branches
merged = layers.Concatenate()([encoded_a, encoded_b])
merged = layers.Dense(128, activation='relu')(merged)
merged = layers.Dropout(0.3)(merged)

# Multiple outputs
similarity_output = layers.Dense(1, activation='sigmoid', name='similarity')(merged)
category_output = layers.Dense(10, activation='softmax', name='category')(merged)

# Build multi-input, multi-output model
model = keras.Model(
    inputs=[input_a, input_b],
    outputs=[similarity_output, category_output]
)

model.compile(
    optimizer='adam',
    loss={'similarity': 'binary_crossentropy', 'category': 'categorical_crossentropy'},
    loss_weights={'similarity': 1.0, 'category': 0.5},
    metrics={'similarity': 'accuracy', 'category': 'accuracy'}
)

model.summary()
# model.fit({'input_a': X_a, 'input_b': X_b},
#           {'similarity': y_sim, 'category': y_cat}, epochs=10)
```

> **Interview Tip:** The **Functional API** is essential for shared layers (Siamese networks), multi-input/output models, and non-linear topologies. Use `loss_weights` to balance multiple objectives.

---

## Question 20
- [ ] Done

**Code an LSTM network in Keras to perform sentiment analysis on text data**

### Solution

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load IMDB dataset
vocab_size = 10000
max_length = 200

(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_length)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_length)

# Build LSTM model
model = keras.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_length),
    layers.SpatialDropout1D(0.3),       # Dropout for embeddings
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                   restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
]

history = model.fit(X_train, y_train, epochs=20, batch_size=64,
                    validation_split=0.2, callbacks=callbacks)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Predict sentiment
def predict_sentiment(text, word_index):
    tokens = [word_index.get(w, 0) for w in text.lower().split()]
    padded = keras.preprocessing.sequence.pad_sequences([tokens], maxlen=max_length)
    score = model.predict(padded)[0][0]
    return "Positive" if score > 0.5 else "Negative", score
```

> **Interview Tip:** Use **Bidirectional LSTM** to capture context from both directions. `SpatialDropout1D` is better than regular dropout for embedding layers. For production sentiment analysis, consider fine-tuning pre-trained transformers (BERT) instead.

---
