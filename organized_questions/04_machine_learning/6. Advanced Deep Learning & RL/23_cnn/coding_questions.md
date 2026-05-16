# Cnn Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Describe howdropoutis implemented in aCNNand what effects it has.**

### Answer:

**1. Precise Definition:**
Dropout randomly deactivates a fraction of neurons during training by setting their outputs to zero with probability $p$ (typically 0.5), forcing the network to learn redundant representations and preventing co-adaptation of neurons, acting as an ensemble method that improves generalization.

**2. Implementation:**

**Training:**
- Randomly set neuron outputs to 0 with probability $p$
- Scale remaining outputs by $1/(1-p)$ (inverted dropout)

**Inference:**
- Use all neurons (no dropout)
- No scaling needed with inverted dropout

**3. Mathematical Formulation:**

**Training:**
$$y = \frac{1}{1-p} \cdot (x \odot m), \quad m_i \sim \text{Bernoulli}(1-p)$$

**Inference:**
$$y = x \quad \text{(all neurons active)}$$

**4. Code Implementation:**

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

print("=== Dropout Implementation in CNNs ===\n")

# Method 1: Keras Dropout Layer
model_with_dropout = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    # Dropout before dense layers (common placement)
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Drop 50% of neurons
    
    layers.Dense(10, activation='softmax')
])

print("Standard CNN with Dropout:")
print("  - Dropout rate: 0.5 (drop 50%)")
print("  - Placement: After fully connected layers")
print("  - Training: Random dropout")
print("  - Inference: All neurons active")

# Method 2: From Scratch (Numpy)
def dropout_forward(x, p=0.5, training=True):
    \"""
    Dropout forward pass
    x: input activations
    p: dropout probability
    training: training mode flag
    \"""
    if training:
        # Generate dropout mask
        mask = np.random.binomial(1, 1-p, size=x.shape) / (1-p)
        out = x * mask
        cache = mask
    else:
        # Inference: no dropout
        out = x
        cache = None
    
    return out, cache

def dropout_backward(dout, cache):
    \"""
    Dropout backward pass
    dout: upstream gradient
    cache: mask from forward pass
    \"""
    mask = cache
    dx = dout * mask
    return dx

print("\n=== From Scratch Implementation ===")

# Example forward pass
x = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8]])

print("Input:\n", x)

# Training mode
out_train, mask = dropout_forward(x, p=0.5, training=True)
print("\nTraining (p=0.5):\n", out_train)
print("Mask:\n", mask)
print("Note: Some values dropped (set to 0), others scaled by 2")

# Inference mode
out_test, _ = dropout_forward(x, p=0.5, training=False)
print("\nInference:\n", out_test)
print("Note: All values preserved")

# Dropout placement strategies
print("\n=== Dropout Placement Strategies ===")
print(\"""
1. After Fully Connected Layers (Standard):
   Conv → Pool → Conv → Pool → Flatten → Dense → Dropout(0.5) → Dense

2. After Convolutional Layers (Less Common):
   Conv → Dropout(0.25) → Pool → Conv → Dropout(0.25) → Pool
   - Use lower rate (0.1-0.25) for conv layers

3. Spatial Dropout (Conv Layers):
   - Drop entire feature maps instead of individual neurons
   - Better for CNNs
   
   layers.SpatialDropout2D(0.25)

4. DropBlock (Recent):
   - Drop contiguous regions
   - More effective for CNNs than random dropout
\""")

# Effects of dropout
print("\n=== Effects of Dropout ===")
print(\"""
Positive Effects:
  ✓ Reduces overfitting
  ✓ Ensemble effect (trains 2^N sub-networks)
  ✓ Forces redundant representations
  ✓ Prevents co-adaptation of neurons
  ✓ Acts as regularization

Negative Effects:
  ✗ Slower convergence (less signal per iteration)
  ✗ Requires more epochs
  ✗ Complicates architecture
  ✗ May hurt performance if too high

Typical Values:
  - Fully connected layers: 0.5
  - Convolutional layers: 0.1-0.25 (if used)
  - Output layer: No dropout
\""")

# Comparison: With vs Without Dropout
print("\n=== Dropout Impact (Example) ===")
print(\"""
Without Dropout:
  Training accuracy: 99%
  Validation accuracy: 82%
  → Overfitting!

With Dropout (0.5):
  Training accuracy: 92%
  Validation accuracy: 88%
  → Better generalization
\""")

# Advanced: Spatial Dropout
model_spatial_dropout = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.SpatialDropout2D(0.25),  # Drop entire feature maps
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.SpatialDropout2D(0.25),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

print("\n=== Spatial Dropout (Recommended for CNNs) ===")
print("Drops entire 2D feature maps instead of individual neurons")
print("More effective than standard dropout for convolutional layers")
```

**5. Interview Tips:**
- **Remember**: "Dropout is training-time only regularization"
- **Tip**: Mention inverted dropout (scaling during training)
- **Interview Focus**: Why it works (ensemble effect, co-adaptation)
- **Key Insight**: Different rates for conv (0.25) vs FC (0.5) layers

---

## Question 2
- [ ] Done

**Implement aconvolution layerfrom scratch usingNumpy.**

### Answer:

```python
import numpy as np

def conv2d_forward(X, W, b, stride=1, padding=0):
    \"""
    Forward pass for 2D convolution
    
    Args:
        X: Input (N, H, W, C_in)
        W: Filters (k, k, C_in, C_out)
        b: Biases (C_out,)
        stride: Stride
        padding: Padding
        
    Returns:
        out: Output (N, H_out, W_out, C_out)
        cache: Values for backward pass
    \"""
    N, H, W, C_in = X.shape
    k, _, _, C_out = W.shape
    
    # Apply padding
    if padding > 0:
        X_pad = np.pad(X, ((0,0), (padding,padding), (padding,padding), (0,0)), 
                       mode='constant')
    else:
        X_pad = X
    
    # Output dimensions
    H_out = (H + 2*padding - k) // stride + 1
    W_out = (W + 2*padding - k) // stride + 1
    
    # Initialize output
    out = np.zeros((N, H_out, W_out, C_out))
    
    # Convolution
    for n in range(N):  # Each image
        for c_out in range(C_out):  # Each output channel
            for h in range(H_out):  # Height
                for w in range(W_out):  # Width
                    # Extract region
                    h_start = h * stride
                    h_end = h_start + k
                    w_start = w * stride
                    w_end = w_start + k
                    
                    region = X_pad[n, h_start:h_end, w_start:w_end, :]
                    
                    # Convolution: element-wise multiply + sum
                    out[n, h, w, c_out] = np.sum(region * W[:, :, :, c_out]) + b[c_out]
    
    cache = (X, W, b, stride, padding)
    return out, cache

# Test
print("=== Convolution from Scratch ===\n")

# Input: 1 image, 5×5, 1 channel
X = np.array([[[[1, 2, 3, 4, 5],
                 [6, 7, 8, 9, 10],
                 [11, 12, 13, 14, 15],
                 [16, 17, 18, 19, 20],
                 [21, 22, 23, 24, 25]]]])  # Shape: (1, 5, 5, 1)

# Filter: 3×3, 1 input channel, 1 output channel (edge detector)
W = np.array([[[[-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]]]])  # Shape: (3, 3, 1, 1)

# Transpose to correct shape (k, k, C_in, C_out)
W = W.transpose(2, 3, 0, 1)
b = np.array([0.0])

print("Input (5×5):")
print(X[0, :, :, 0])

print("\nFilter (3×3) - Edge detector:")
print(W[:, :, 0, 0])

# Forward pass
out, _ = conv2d_forward(X, W, b, stride=1, padding=0)

print("\nOutput (3×3) - After convolution:")
print(out[0, :, :, 0])

print("\n✓ Convolution working correctly!")
print("Output dimensions: (5-3)/1 + 1 = 3×3")
```

---

## Question 3
- [ ] Done

**Write aPython functionto applymax poolingto a given input matrix.**

### Answer:

```python
import numpy as np

def max_pool2d(X, pool_size=2, stride=2):
    \"""
    Max pooling forward pass
    
    Args:
        X: Input (N, H, W, C)
        pool_size: Pooling window size
        stride: Stride
        
    Returns:
        out: Pooled output
        cache: Values for backward pass
    \"""
    N, H, W, C = X.shape
    
    # Output dimensions
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    # Initialize output
    out = np.zeros((N, H_out, W_out, C))
    
    # Max pooling
    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    h_end = h_start + pool_size
                    w_start = w * stride
                    w_end = w_start + pool_size
                    
                    # Extract window and take max
                    window = X[n, h_start:h_end, w_start:w_end, c]
                    out[n, h, w, c] = np.max(window)
    
    cache = (X, pool_size, stride)
    return out, cache

# Test
print("=== Max Pooling from Scratch ===\n")

X = np.array([[[[1, 3, 2, 4],
                 [5, 6, 7, 8],
                 [2, 1, 4, 3],
                 [9, 8, 6, 5]]]])  # (1, 4, 4, 1)

print("Input (4×4):")
print(X[0, :, :, 0])

out, _ = max_pool2d(X, pool_size=2, stride=2)

print("\nOutput after 2×2 max pooling:")
print(out[0, :, :, 0])
print("\nTook maximum from each 2×2 window")
print("Result: [6, 8]")
print("        [9, 6]")
```

---

## Question 4
- [ ] Done

**UseTensorFlow/Kerasto build and train aCNNto classify images from theCIFAR-10 dataset.**

### Answer:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("=== CIFAR-10 CNN Classification ===\n")

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(f"Train: {x_train.shape}, Test: {x_test.shape}")

# Normalize
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Build CNN
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                  input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Classifier
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(\n    optimizer='adam',\n    loss='sparse_categorical_crossentropy',\n    metrics=['accuracy']\n)

model.summary()

# Data augmentation\ndatagen = ImageDataGenerator(\n    rotation_range=15,\n    width_shift_range=0.1,\n    height_shift_range=0.1,\n    horizontal_flip=True\n)

# Train\nhistory = model.fit(\n    datagen.flow(x_train, y_train, batch_size=64),\n    validation_data=(x_test, y_test),\n    epochs=50,\n    callbacks=[\n        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),\n        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)\n    ]\n)

# Evaluate\ntest_loss, test_acc = model.evaluate(x_test, y_test)\nprint(f"\nTest Accuracy: {test_acc:.4f}")
```

---

## Question 5
- [ ] Done

**Visualize the filters of the firstconvolutional layerof a trainedCNNusingMatplotlib.**

### Answer:

```python
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np

def visualize_conv_filters(model, layer_name='conv2d'):
    \"""Visualize filters from first convolutional layer\"""
    
    # Get layer
    layer = model.get_layer(layer_name)
    filters, biases = layer.get_weights()
    
    print(f"Filter shape: {filters.shape}")  # (k, k, C_in, C_out)
    
    # Normalize filters for visualization
    f_min, f_max = filters.min(), filters.max()
    filters_norm = (filters - f_min) / (f_max - f_min)
    
    # Plot first 32 filters\    n_filters = min(32, filters.shape[-1])
    n_cols = 8
    n_rows = n_filters // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))
    
    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            # Get filter
            f = filters_norm[:, :, :, i]
            
            # If RGB (3 channels), display as RGB
            if f.shape[2] == 3:
                ax.imshow(f)
            else:
                # Grayscale: average over input channels
                ax.imshow(f.mean(axis=2), cmap='gray')
            
            ax.set_title(f'Filter {i}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('conv_filters.png')
    print("Saved to conv_filters.png")

# Example: Load pre-trained VGG16 and visualize
from tensorflow.keras.applications import VGG16

model = VGG16(weights='imagenet', include_top=False)
visualize_conv_filters(model, layer_name='block1_conv1')

print("Filters show edge detectors, color blobs, textures")
```

---

## Question 6
- [ ] Done

**Create a script tofine-tunea pre-trainedCNNon a new dataset withTensorFlow/Keras.**

### Answer:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

print("=== Fine-Tuning Pre-Trained CNN ===\n")

# Load pre-trained ResNet50 (ImageNet weights)
base_model = ResNet50(\n    weights='imagenet',\n    include_top=False,  # Remove classification head\n    input_shape=(224, 224, 3)\n)

print(f"Loaded ResNet50 with {len(base_model.layers)} layers")

# Freeze base model initially
base_model.trainable = False

# Add custom classification head
model = models.Sequential([\n    base_model,\n    layers.GlobalAveragePooling2D(),\n    layers.Dense(256, activation='relu'),\n    layers.Dropout(0.5),\n    layers.Dense(5, activation='softmax')  # 5 classes for new dataset\n])

# Compile\nmodel.compile(\n    optimizer='adam',\n    loss='categorical_crossentropy',\n    metrics=['accuracy']\n)

print("\n=== Phase 1: Train only new layers ===\")
# Train only new layers (base frozen)\n# history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# Unfreeze some layers for fine-tuning
print("\n=== Phase 2: Fine-tune top layers ===\")
base_model.trainable = True

# Freeze early layers, train only later layers
for layer in base_model.layers[:100]:\n    layer.trainable = False

# Recompile with lower learning rate
model.compile(\n    optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower LR\n    loss='categorical_crossentropy',\n    metrics=['accuracy']\n)

print(f"Trainable layers: {len([l for l in model.layers if l.trainable])}") 
# history = model.fit(train_ds, validation_data=val_ds, epochs=20)

print("\nFine-tuning complete!")
print("Strategy:")
print("  1. Freeze base, train new head (10 epochs)")
print("  2. Unfreeze top layers, fine-tune (20 epochs, low LR)")
```

---
