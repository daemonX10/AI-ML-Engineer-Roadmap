# Keras Interview Questions - General Questions

## Question 1
- [ ] Done

**How do you prevent overfitting in a Keras model ?**

### Answer

### Overfitting Solutions

| Technique | Implementation |
|-----------|----------------|
| **Dropout** | `layers.Dropout(0.5)` |
| **L1/L2 Regularization** | `kernel_regularizer=l2(0.01)` |
| **Early Stopping** | `EarlyStopping` callback |
| **Data Augmentation** | Random transforms |
| **Reduce Model Complexity** | Fewer layers/neurons |

### Python Code Example
```python
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Model with regularization
model = keras.Sequential([
    layers.Dense(128, activation='relu', 
                 kernel_regularizer=regularizers.l2(0.01),
                 input_shape=(784,)),
    layers.Dropout(0.5),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu', 
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# Early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X_train, y_train, validation_split=0.2,
          epochs=100, callbacks=[early_stop])
```

---

## Question 2
- [ ] Done

**How do you save and load a Keras model?**

### Answer

### Saving Options

| Method | Saves | Format |
|--------|-------|--------|
| `model.save()` | Full model | .h5 or SavedModel |
| `model.save_weights()` | Weights only | .h5 |
| `ModelCheckpoint` | During training | Configurable |

### Python Code Example
```python
from tensorflow import keras

# Build and train model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Save entire model
model.save('my_model.h5')  # HDF5 format
model.save('my_model')      # SavedModel format

# Load entire model
loaded_model = keras.models.load_model('my_model.h5')

# Save weights only
model.save_weights('weights.h5')

# Load weights (need same architecture)
new_model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])
new_model.load_weights('weights.h5')

# Checkpoint during training
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    save_best_only=True,
    monitor='val_loss'
)
```

---

## Question 3
- [ ] Done

**What is transfer learning and how do you implement it in Keras?**

### Answer

**Definition**: Transfer learning uses a pre-trained model as starting point for a new task.

### Steps
1. Load pre-trained model (without top layers)
2. Freeze base layers
3. Add new classification layers
4. Train on new data

### Python Code Example
```python
from tensorflow import keras
from tensorflow.keras import layers

# Load pre-trained model
base_model = keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,  # Remove classification head
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add new layers
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 classes
])

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train new layers
model.fit(X_train, y_train, epochs=10)

# Fine-tuning: unfreeze some layers
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Lower learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Question 4
- [ ] Done

**How do you handle imbalanced datasets in Keras?**

### Answer

### Solutions

| Method | Implementation |
|--------|----------------|
| **Class weights** | `class_weight` in fit() |
| **Sample weights** | `sample_weight` in fit() |
| **Oversampling** | SMOTE, data augmentation |
| **Custom loss** | Focal loss |

### Python Code Example
```python
from tensorflow import keras
import numpy as np
from collections import Counter

# Calculate class weights
counter = Counter(y_train)
total = len(y_train)
class_weight = {
    0: total / (2 * counter[0]),  # Majority class
    1: total / (2 * counter[1])   # Minority class
}

# Method 1: Class weights
model.fit(
    X_train, y_train,
    class_weight=class_weight,
    epochs=10
)

# Method 2: Sample weights
sample_weights = np.ones(len(y_train))
sample_weights[y_train == 1] = 5.0  # Increase minority weight

model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    epochs=10
)

# Method 3: Focal Loss (for severe imbalance)
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        bce = keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = (1 - p_t) ** self.gamma
        return self.alpha * focal_weight * bce

model.compile(optimizer='adam', loss=FocalLoss())
```

---

## Question 5
- [ ] Done

**What are the different types of layers available in Keras?**

### Answer

### Layer Categories

| Category | Layers | Use Case |
|----------|--------|----------|
| **Core** | Dense, Activation, Embedding | Basic building blocks |
| **Convolutional** | Conv2D, Conv1D, SeparableConv2D | Image processing |
| **Pooling** | MaxPooling2D, AveragePooling2D | Reduce dimensions |
| **Recurrent** | LSTM, GRU, SimpleRNN | Sequences |
| **Normalization** | BatchNormalization, LayerNormalization | Stabilize training |
| **Regularization** | Dropout, SpatialDropout2D | Prevent overfitting |
| **Attention** | MultiHeadAttention | Transformers |

### Python Code Example
```python
from tensorflow.keras import layers

# Dense layer
dense = layers.Dense(64, activation='relu')

# Convolutional layers
conv = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
pool = layers.MaxPooling2D(pool_size=(2, 2))

# Recurrent layers
lstm = layers.LSTM(64, return_sequences=True)
gru = layers.GRU(32)

# Normalization
batch_norm = layers.BatchNormalization()

# Dropout
dropout = layers.Dropout(0.5)

# Embedding (for text)
embedding = layers.Embedding(input_dim=10000, output_dim=128)

# Attention
attention = layers.MultiHeadAttention(num_heads=8, key_dim=64)

# Complete CNN example
cnn_model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

---

## Question 6
- [ ] Done

**How do you implement data augmentation in Keras?**

### Answer

### Augmentation Methods

| Method | Use Case |
|--------|----------|
| **Keras Layers** | In-model augmentation |
| **ImageDataGenerator** | Real-time augmentation |
| **tf.data** | Pipeline augmentation |

### Python Code Example
```python
from tensorflow import keras
from tensorflow.keras import layers

# Method 1: Augmentation as layers (TF 2.x)
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

model = keras.Sequential([
    data_augmentation,
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# Method 2: ImageDataGenerator (legacy but still common)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.1,
    rescale=1./255
)

# Flow from directory
train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Use in training
# model.fit(train_generator, epochs=10)
```

---

## Question 7
- [ ] Done

**How do you create a custom layer in Keras?**

### Answer

### Custom Layer Structure
- `__init__`: Define parameters
- `build`: Create weights
- `call`: Define forward pass

### Python Code Example
```python
from tensorflow import keras
import tensorflow as tf

class CustomDense(keras.layers.Layer):
    """Custom fully connected layer"""
    
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
    
    def build(self, input_shape):
        # Create trainable weights
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        super().build(input_shape)
    
    def call(self, inputs):
        output = tf.matmul(inputs, self.w) + self.b
        if self.activation:
            output = self.activation(output)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': keras.activations.serialize(self.activation)
        })
        return config

# Use custom layer
model = keras.Sequential([
    CustomDense(64, activation='relu', input_shape=(10,)),
    CustomDense(32, activation='relu'),
    CustomDense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
```

---

## Question 8
- [ ] Done

**What is batch normalization and why is it used?**

### Answer

**Definition**: Batch normalization normalizes layer inputs to have zero mean and unit variance, stabilizing training.

### Benefits

| Benefit | Description |
|---------|-------------|
| **Faster training** | Allows higher learning rates |
| **Stability** | Reduces internal covariate shift |
| **Regularization** | Acts as mild regularizer |
| **Reduce sensitivity** | Less sensitive to initialization |

### Python Code Example
```python
from tensorflow import keras
from tensorflow.keras import layers

# BatchNormalization placement
model = keras.Sequential([
    # Option 1: After activation
    layers.Dense(64),
    layers.Activation('relu'),
    layers.BatchNormalization(),
    
    # Option 2: Before activation (recommended)
    layers.Dense(32),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    
    layers.Dense(10, activation='softmax')
])

# In CNN
cnn_model = keras.Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])
```

### Key Points
- Different behavior in training vs inference
- Uses batch statistics during training
- Uses learned moving averages during inference

---

## Question 9
- [ ] Done

**How do you implement early stopping in Keras?**

### Answer

**Definition**: Early stopping stops training when validation metric stops improving, preventing overfitting.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `monitor` | Metric to track |
| `patience` | Epochs to wait |
| `restore_best_weights` | Return to best model |
| `min_delta` | Minimum improvement |

### Python Code Example
```python
from tensorflow import keras

# Early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',        # Metric to monitor
    patience=5,                 # Wait 5 epochs
    min_delta=0.001,           # Minimum improvement
    restore_best_weights=True,  # Return best model
    verbose=1
)

# Use with model checkpoint
callbacks = [
    early_stopping,
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_loss'
    )
]

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train with early stopping
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,  # Max epochs
    callbacks=callbacks
)

# Check when training stopped
print(f"Stopped at epoch: {len(history.history['loss'])}")
```

---

## Question 10
- [ ] Done

**How do you use pre-trained word embeddings in Keras?**

### Answer

### Steps
1. Load pre-trained embeddings (GloVe, Word2Vec)
2. Create embedding matrix
3. Initialize Embedding layer with weights

### Python Code Example
```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load GloVe embeddings
def load_glove(path, word_index, embedding_dim=100):
    embeddings_index = {}
    with open(path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    # Create embedding matrix
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))
    
    for word, i in word_index.items():
        if word in embeddings_index:
            embedding_matrix[i] = embeddings_index[word]
    
    return embedding_matrix

# word_index from tokenizer
# embedding_matrix = load_glove('glove.6B.100d.txt', word_index)

# Create embedding layer with pre-trained weights
vocab_size = 10000
embedding_dim = 100

embedding_layer = layers.Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    # weights=[embedding_matrix],  # Pre-trained weights
    trainable=False  # Freeze embeddings
)

# Model for text classification
model = keras.Sequential([
    embedding_layer,
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
```

---

## Question 11
- [ ] Done

**How do you configure a neural network in Keras?**

**Answer:**

Configuring a neural network in Keras involves three main steps: **defining architecture**, **compiling**, and **training**.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# === Step 1: Define Architecture ===
# Option A: Sequential API (linear stack)
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Option B: Functional API (complex architectures)
inputs = layers.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = models.Model(inputs=inputs, outputs=outputs)

# === Step 2: Compile ===
model.compile(
    optimizer='adam',              # Or tf.keras.optimizers.Adam(lr=0.001)
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Step 3: Train ===
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
)

# === Step 4: Inspect ===
model.summary()                   # Architecture overview
print(model.count_params())       # Total trainable parameters
```

| Configuration | Key Options |
|--------------|-------------|
| **Optimizer** | Adam, SGD, RMSprop, Adagrad |
| **Loss** | categorical_crossentropy, mse, binary_crossentropy |
| **Activation** | relu, sigmoid, softmax, tanh, leaky_relu |
| **Regularization** | Dropout, BatchNorm, L1/L2 |
| **Architecture** | Sequential, Functional, Subclassing |

> **Interview Tip:** The Functional API is preferred over Sequential for any model with shared layers, multiple inputs/outputs, or residual connections.

---

## Question 12
- [ ] Done

**How can you add regularization to a model in Keras?**

**Answer:**

Regularization prevents overfitting by penalizing model complexity. Keras provides multiple techniques:

```python
from tensorflow.keras import layers, regularizers

# === 1. L1/L2/ElasticNet Regularization ===
layers.Dense(64, activation='relu',
    kernel_regularizer=regularizers.l2(0.01),    # Weight decay
    bias_regularizer=regularizers.l1(0.001),     # Bias penalty
    activity_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01)  # Output penalty
)

# === 2. Dropout ===
layers.Dropout(0.5)           # Randomly zero 50% of neurons during training
layers.SpatialDropout1D(0.2)  # For sequential/1D data
layers.SpatialDropout2D(0.2)  # For images (drops entire feature maps)

# === 3. Batch Normalization ===
layers.BatchNormalization()   # Normalizes layer inputs, acts as regularizer

# === 4. Early Stopping ===
from tensorflow.keras.callbacks import EarlyStopping
callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# === 5. Data Augmentation ===
augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# === 6. Max-Norm Constraint ===
layers.Dense(64, kernel_constraint=tf.keras.constraints.MaxNorm(3))
```

| Technique | How It Works | When to Use |
|-----------|-------------|-------------|
| L2 (Ridge) | Penalizes large weights | Default choice |
| L1 (Lasso) | Drives weights to zero (sparsity) | Feature selection |
| Dropout | Randomly deactivates neurons | Dense layers |
| BatchNorm | Normalizes activations | Deep networks |
| Early Stopping | Stops when val_loss stalls | Always use |
| Data Augmentation | Increases training variety | Image tasks |

> **Interview Tip:** Combine multiple techniques: L2 + Dropout + BatchNorm + EarlyStopping is a common production stack. Start with Dropout(0.2-0.5) and tune based on validation performance.

---

## Question 13
- [ ] Done

**How do callbacks work in Keras and when would you use them?**

**Answer:**

Callbacks are objects that perform actions at various stages of training (epoch start/end, batch start/end). They allow you to monitor, control, and modify training behavior.

```python
from tensorflow.keras import callbacks

# === Built-in Callbacks ===
cb_list = [
    # Stop training when metric stops improving
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    
    # Save model at best checkpoint
    callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy'),
    
    # Reduce learning rate on plateau
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
    
    # TensorBoard logging
    callbacks.TensorBoard(log_dir='./logs', histogram_freq=1),
    
    # CSV logging
    callbacks.CSVLogger('training_log.csv'),
    
    # Learning rate scheduler
    callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 0.9 ** epoch)
]

model.fit(X_train, y_train, callbacks=cb_list, epochs=100)

# === Custom Callback ===
class CustomCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch}")
    
    def on_epoch_end(self, epoch, logs=None):
        if logs['val_loss'] < 0.01:
            self.model.stop_training = True
            print("Low enough loss, stopping!")
    
    def on_train_begin(self, logs=None):
        print("Training started")
    
    def on_batch_end(self, batch, logs=None):
        pass  # Called after each batch
```

| Callback | Purpose |
|----------|--------|
| `EarlyStopping` | Prevent overfitting |
| `ModelCheckpoint` | Save best model |
| `ReduceLROnPlateau` | Adaptive learning rate |
| `TensorBoard` | Visualization |
| `LearningRateScheduler` | Custom LR schedule |
| `TerminateOnNaN` | Stop on NaN loss |

> **Interview Tip:** Always use `EarlyStopping` + `ModelCheckpoint` together. EarlyStopping prevents overfitting; ModelCheckpoint ensures you keep the best weights, not the last ones.

---

## Question 14
- [ ] Done

**What methods does Keras provide for evaluating a model’s performance ?**

**Answer:**

```python
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# === 1. model.evaluate() — Loss & Metrics ===
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# === 2. model.predict() — Raw Predictions ===
y_pred_probs = model.predict(X_test)            # Probabilities
y_pred_classes = np.argmax(y_pred_probs, axis=1) # Class labels

# === 3. Training History ===
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.show()

# === 4. Built-in Metrics ===
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.AUC(),
        tf.keras.metrics.F1Score()
    ]
)

# === 5. Sklearn Integration ===
print(classification_report(y_true, y_pred_classes))
print(confusion_matrix(y_true, y_pred_classes))

# === 6. Cross-Validation (with sklearn wrapper) ===
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_model():
    m = tf.keras.Sequential([...])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

wrapper = KerasClassifier(model=build_model, epochs=10, batch_size=32)
scores = cross_val_score(wrapper, X, y, cv=5)
```

| Method | Returns | Use Case |
|--------|---------|----------|
| `evaluate()` | Loss + compiled metrics | Quick test set assessment |
| `predict()` | Raw outputs | Custom metric calculation |
| `history` | Per-epoch metrics dict | Learning curve analysis |
| Built-in metrics | Precision/Recall/AUC/F1 | Compiled into model |
| Sklearn wrappers | Cross-val scores | Robust evaluation |

> **Interview Tip:** Always plot training vs. validation curves from `history` to diagnose overfitting (diverging curves) or underfitting (both curves plateau high).

---

## Question 15
- [ ] Done

**How do you handle image data in Keras?**

**Answer:**

```python
import tensorflow as tf
from tensorflow.keras import layers

# === 1. ImageDataGenerator (Legacy but common) ===
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    'data/train/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# === 2. tf.keras.utils.image_dataset_from_directory (Modern) ===
train_ds = tf.keras.utils.image_dataset_from_directory(
    'data/train/',
    image_size=(224, 224),
    batch_size=32,
    validation_split=0.2,
    subset='training',
    seed=42
)

# Normalize
normalization = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization(x), y))

# === 3. Data Augmentation Layer ===
augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])

# === 4. CNN Model for Images ===
model = tf.keras.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    augmentation,
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# === 5. Transfer Learning ===
base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False
model = tf.keras.Sequential([base, layers.GlobalAveragePooling2D(), layers.Dense(10, activation='softmax')])
```

| Method | Best For | Notes |
|--------|----------|-------|
| `ImageDataGenerator` | Legacy projects | Augmentation on-the-fly |
| `image_dataset_from_directory` | Modern projects | Returns `tf.data.Dataset` |
| Augmentation layers | In-model augmentation | GPU-accelerated |
| Transfer learning | Small datasets | Use pretrained backbone |

> **Interview Tip:** Use `image_dataset_from_directory` + augmentation layers (not `ImageDataGenerator`) for new projects. Augmentation layers run on GPU and integrate into the model graph.

---

## Question 16
- [ ] Done

**What Keras functionality allows you to convert text to sequences or one-hot encoded vectors?**

**Answer:**

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts = ["I love machine learning", "Deep learning is amazing", "NLP is fun"]

# === 1. Tokenizer — Text to Sequences ===
tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)

word_index = tokenizer.word_index  # {'learning': 1, 'is': 2, 'i': 3, ...}
sequences = tokenizer.texts_to_sequences(texts)  # [[3, 4, 5, 1], [6, 1, 2, 7], [8, 2, 9]]

# === 2. Padding ===
padded = pad_sequences(sequences, maxlen=10, padding='post', truncating='post')
# [[3, 4, 5, 1, 0, 0, 0, 0, 0, 0], ...]

# === 3. One-Hot Encoding ===
one_hot = tokenizer.texts_to_matrix(texts, mode='binary')  # Shape: (3, 1000)
# Modes: 'binary', 'count', 'tfidf', 'freq'

# === 4. Modern TextVectorization Layer (Recommended) ===
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=1000,
    output_mode='int',           # 'int', 'multi_hot', 'count', 'tf_idf'
    output_sequence_length=50
)
vectorizer.adapt(texts)
encoded = vectorizer(texts)      # Tensor of integer sequences

# === 5. Using in Model ===
model = tf.keras.Sequential([
    tf.keras.layers.TextVectorization(max_tokens=5000, output_sequence_length=100),
    tf.keras.layers.Embedding(5000, 64),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

| Method | Output | Use Case |
|--------|--------|----------|
| `texts_to_sequences()` | Integer sequences | Input to Embedding layer |
| `texts_to_matrix()` | One-hot / TF-IDF matrix | Traditional ML |
| `pad_sequences()` | Fixed-length sequences | Batch processing |
| `TextVectorization` | In-model preprocessing | Production deployment |

> **Interview Tip:** Use `TextVectorization` layer for new projects—it integrates directly into the model, making deployment easier since preprocessing is part of the saved model.

---

## Question 17
- [ ] Done

**How do you troubleshoot a model that is not learning in Keras?**

**Answer:**

A systematic debugging checklist:

| Step | Check | Fix |
|------|-------|-----|
| 1. Data | Labels correct? Normalized? | Verify `y_train` matches `X_train`; scale features |
| 2. Loss | Correct for task? | Binary: `binary_crossentropy`, Multi-class: `categorical_crossentropy` |
| 3. Learning Rate | Too high/low? | Start with 1e-3; use `ReduceLROnPlateau` |
| 4. Architecture | Too simple/complex? | Start small, increase capacity |
| 5. Activation | Output activation matches loss? | Sigmoid + BCE; Softmax + CCE |
| 6. Gradient Flow | Vanishing/exploding? | Use BatchNorm, residual connections, gradient clipping |

```python
# === Diagnostic Steps ===

# 1. Verify data
print(f"X shape: {X_train.shape}, y shape: {y_train.shape}")
print(f"X range: [{X_train.min()}, {X_train.max()}]")
print(f"Class distribution: {np.bincount(y_train)}")

# 2. Overfit on small batch (should reach ~100% accuracy)
model.fit(X_train[:32], y_train[:32], epochs=100)  # Must overfit!

# 3. Check gradients
with tf.GradientTape() as tape:
    pred = model(X_train[:1])
    loss = tf.keras.losses.categorical_crossentropy(y_train[:1], pred)
grads = tape.gradient(loss, model.trainable_variables)
for g, v in zip(grads, model.trainable_variables):
    print(f"{v.name}: grad_mean={tf.reduce_mean(tf.abs(g)):.6f}")

# 4. Learning rate finder
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10 ** (epoch / 20)
)
history = model.fit(X_train, y_train, epochs=100, callbacks=[lr_schedule])
# Plot loss vs LR to find optimal

# 5. Monitor weights
for layer in model.layers:
    if layer.weights:
        w = layer.weights[0].numpy()
        print(f"{layer.name}: mean={w.mean():.4f}, std={w.std():.4f}")
```

> **Interview Tip:** The first thing to try is overfitting on a tiny batch. If the model can't memorize 32 samples, there's a bug in architecture, loss, or data pipeline—not a hyperparameter issue.

---

## Question 18
- [ ] Done

**How do you interpret NaN values in loss during training and what steps would you take to address this?**

**Answer:**

NaN loss means numerical instability during training. Common causes and fixes:

| Cause | How to Detect | Fix |
|-------|---------------|-----|
| Learning rate too high | Loss spikes then NaN | Reduce LR by 10x |
| Exploding gradients | Weights grow rapidly | Gradient clipping: `optimizer = Adam(clipnorm=1.0)` |
| Bad data (NaN/Inf in input) | `np.isnan(X).any()` | Clean data, impute missing values |
| Log of zero | Using `log` on predictions near 0 | Add epsilon: `log(x + 1e-7)` |
| Division by zero | Custom loss functions | Add small constant |
| Wrong loss/activation combo | Logits in wrong range | Match output activation to loss |

```python
import numpy as np
import tensorflow as tf

# === Step 1: Check Data ===
print(f"NaN in X: {np.isnan(X_train).sum()}")
print(f"Inf in X: {np.isinf(X_train).sum()}")
print(f"X range: [{X_train.min()}, {X_train.max()}]")

# === Step 2: Gradient Clipping ===
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
# Or clipvalue: optimizer = Adam(clipvalue=0.5)

# === Step 3: Numerically Stable Loss ===
# Instead of: loss = -tf.reduce_mean(y * tf.math.log(y_pred))
# Use: loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# === Step 4: Terminate on NaN ===
callbacks = [
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3)
]

# === Step 5: Use Mixed Precision Carefully ===
# If using float16, ensure loss scaling:
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

# === Step 6: Debug with NaN checking ===
tf.debugging.enable_check_numerics()  # Raises error on NaN/Inf
```

> **Interview Tip:** The most common cause is learning rate too high. First try reducing LR. Then check for NaN in data. Then add gradient clipping. Finally, ensure loss function uses numerically stable implementations (e.g., `from_logits=True`).

---

## Question 19
- [ ] Done

**How do you deal with overfitting after early epochs in a Keras model?**

**Answer:**

Overfitting after early epochs means the model learns training data too quickly. Here's a layered approach:

```python
import tensorflow as tf
from tensorflow.keras import layers, callbacks, regularizers

# === 1. Data-Level Solutions ===
# More data / Data Augmentation
augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
])

# === 2. Regularization Stack ===
model = tf.keras.Sequential([
    augmentation,
    layers.Conv2D(32, 3, kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),              # Stronger dropout
    layers.Conv2D(64, 3, kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.4),
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation='softmax')
])

# === 3. Training Controls ===
cb_list = [
    callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3
    )
]

# === 4. Reduce Model Capacity ===
# Use fewer layers, fewer units, smaller filters

# === 5. Label Smoothing ===
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
```

| Technique | Effect | Priority |
|-----------|--------|----------|
| More data / augmentation | Increases diversity | Highest |
| Early Stopping | Stops before overfitting | Always use |
| Dropout (0.2-0.5) | Ensemble effect | High |
| L2 Regularization | Penalizes large weights | Medium |
| Reduce model size | Less capacity to memorize | Medium |
| Label Smoothing | Softens targets | Low |
| Batch Normalization | Stabilizes training | Medium |

> **Interview Tip:** If overfitting happens in very early epochs (epoch 2-3), the model is likely too large for the dataset. Try reducing model capacity first, then add regularization.

---

## Question 20
- [ ] Done

**What factors do you consider when deploying a Keras model to production?**

**Answer:**

| Factor | Consideration | Solution |
|--------|--------------|----------|
| **Model Format** | SavedModel vs TFLite vs ONNX | SavedModel for servers; TFLite for mobile; ONNX for cross-framework |
| **Inference Speed** | Latency requirements | Quantization, pruning, TensorRT |
| **Model Size** | Memory/bandwidth constraints | Quantize (float32 -> int8), knowledge distillation |
| **Serving** | REST API vs gRPC vs edge | TF Serving, FastAPI, TFLite |
| **Preprocessing** | Must match training exactly | Embed in model or version pipeline |
| **Monitoring** | Drift detection, performance | Track predictions, latency, errors |
| **Versioning** | Rollback capability | Model registry (MLflow, W&B) |

```python
import tensorflow as tf

# === 1. Export as SavedModel ===
model.save('production_model')

# === 2. Quantization for Size/Speed ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Dynamic range quantization
tflite_model = converter.convert()  # ~4x smaller

# === 3. TF Serving (Docker) ===
# docker pull tensorflow/serving
# docker run -p 8501:8501 --mount type=bind,source=/models,target=/models \
#   -e MODEL_NAME=my_model tensorflow/serving

# === 4. FastAPI Wrapper ===
from fastapi import FastAPI
import numpy as np

app = FastAPI()
model = tf.keras.models.load_model('production_model')

@app.post('/predict')
async def predict(data: dict):
    x = np.array(data['features']).reshape(1, -1)
    pred = model.predict(x)
    return {'prediction': pred.tolist()}

# === 5. Include Preprocessing in Model ===
full_model = tf.keras.Sequential([
    tf.keras.layers.Normalization(),  # Preprocessing
    model                              # Trained model
])
full_model.save('full_pipeline_model')
```

> **Interview Tip:** The #1 deployment mistake is preprocessing mismatch—training and inference pipelines must apply identical transformations. Embed preprocessing into the model graph itself to avoid this.

---

## Question 21
- [ ] Done

**How can you monitor and maintain Keras models in a production environment?**

**Answer:**

| Aspect | What to Monitor | Tools |
|--------|----------------|-------|
| **Performance** | Accuracy, latency, throughput | Prometheus, Grafana |
| **Data Drift** | Feature distribution shifts | Evidently AI, Alibi Detect |
| **Model Drift** | Prediction distribution changes | Custom dashboards |
| **Infrastructure** | CPU/GPU usage, memory | Kubernetes, CloudWatch |
| **Errors** | NaN predictions, timeouts | Sentry, logging |

```python
import tensorflow as tf
import numpy as np
import logging

# === 1. Prediction Monitoring ===
class ModelMonitor:
    def __init__(self, model):
        self.model = model
        self.predictions_log = []
    
    def predict_and_log(self, x):
        pred = self.model.predict(x)
        self.predictions_log.append({
            'timestamp': pd.Timestamp.now(),
            'mean_pred': float(np.mean(pred)),
            'std_pred': float(np.std(pred)),
            'has_nan': bool(np.isnan(pred).any())
        })
        return pred
    
    def check_drift(self, reference_mean, threshold=0.1):
        recent = np.mean([p['mean_pred'] for p in self.predictions_log[-100:]])
        drift = abs(recent - reference_mean) / reference_mean
        if drift > threshold:
            logging.warning(f"Model drift detected: {drift:.2%}")

# === 2. TensorBoard for Continuous Monitoring ===
tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir='./logs', histogram_freq=1, update_freq='epoch'
)

# === 3. Model Versioning ===
import mlflow
mlflow.tensorflow.log_model(model, "model", registered_model_name="prod_model")

# === 4. A/B Testing ===
def ab_predict(x, model_a, model_b, traffic_split=0.1):
    if np.random.random() < traffic_split:
        return model_b.predict(x), 'B'
    return model_a.predict(x), 'A'

# === 5. Health Checks ===
def health_check(model):
    dummy = np.random.randn(1, *model.input_shape[1:])
    try:
        pred = model.predict(dummy, verbose=0)
        assert not np.isnan(pred).any()
        return True
    except Exception:
        return False
```

> **Interview Tip:** Production ML requires monitoring both model performance (accuracy drift) and system performance (latency, memory). Set up automated retraining triggers when drift exceeds a threshold.

---

## Question 22
- [ ] Done

**How is Keras being used in the context of Graph Neural Networks?**

**Answer:**

Graph Neural Networks (GNNs) handle non-Euclidean data (social networks, molecules, knowledge graphs). Keras supports GNNs through dedicated libraries.

```python
# === Using Spektral (Keras-based GNN library) ===
import spektral
from spektral.layers import GCNConv, GlobalSumPool
from tensorflow.keras import layers, models

# GCN (Graph Convolutional Network)
class GCN(models.Model):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = GCNConv(64, activation='relu')
        self.conv2 = GCNConv(32, activation='relu')
        self.pool = GlobalSumPool()
        self.dense = layers.Dense(n_classes, activation='softmax')
    
    def call(self, inputs):
        x, a = inputs            # x: node features, a: adjacency matrix
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.pool(x)
        return self.dense(x)

# === Using TF-GNN (TensorFlow's official GNN) ===
import tensorflow_gnn as tfgnn

# Define graph schema
graph_tensor = tfgnn.GraphTensor.from_pieces(
    node_sets={
        'atoms': tfgnn.NodeSet.from_fields(
            features={'element': tf.constant([[1,0],[0,1],[1,0]])},
            sizes=tf.constant([3])
        )
    },
    edge_sets={
        'bonds': tfgnn.EdgeSet.from_fields(
            features={'type': tf.constant([[1],[2]])},
            sizes=tf.constant([2]),
            adjacency=tfgnn.Adjacency.from_indices(
                source=('atoms', tf.constant([0, 1])),
                target=('atoms', tf.constant([1, 2]))
            )
        )
    }
)
```

| GNN Type | Mechanism | Use Case |
|----------|-----------|----------|
| GCN | Spectral convolution | Node classification |
| GAT | Attention on neighbors | Citation networks |
| GraphSAGE | Sampling + aggregation | Large-scale graphs |
| Message Passing | Edge message updates | Molecular property prediction |

| Library | Integration | Maturity |
|---------|------------|----------|
| Spektral | Native Keras layers | Production-ready |
| TF-GNN | TensorFlow official | Newer, scalable |
| DGL | Multi-backend | Most comprehensive |

> **Interview Tip:** GNNs are gaining importance in drug discovery (molecular graphs), fraud detection (transaction networks), and recommendation systems (user-item graphs). Spektral makes GNNs feel like standard Keras.

---

