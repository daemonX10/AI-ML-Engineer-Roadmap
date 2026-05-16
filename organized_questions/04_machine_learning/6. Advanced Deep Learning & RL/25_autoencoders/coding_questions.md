# Autoencoders Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Implement a basic autoencoder in TensorFlow/Keras to compress and reconstruct images.**

**Pipeline:**
```
1. Load MNIST data → Normalize to [0,1]
2. Build encoder: Flatten → Dense → Dense → Latent
3. Build decoder: Dense → Dense → Reshape
4. Train: Minimize MSE between input and output
5. Visualize: Original vs Reconstructed
```

**Code:**

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

# Step 1: Load and preprocess data
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0  # Normalize to [0,1]
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 784)  # Flatten: 28x28 → 784
x_test = x_test.reshape(-1, 784)

# Step 2: Define dimensions
input_dim = 784
latent_dim = 32

# Step 3: Build Encoder
encoder_input = layers.Input(shape=(input_dim,))
x = layers.Dense(256, activation='relu')(encoder_input)
x = layers.Dense(128, activation='relu')(x)
latent = layers.Dense(latent_dim, activation='relu')(x)
encoder = Model(encoder_input, latent, name='encoder')

# Step 4: Build Decoder
decoder_input = layers.Input(shape=(latent_dim,))
x = layers.Dense(128, activation='relu')(decoder_input)
x = layers.Dense(256, activation='relu')(x)
decoder_output = layers.Dense(input_dim, activation='sigmoid')(x)  # sigmoid for [0,1]
decoder = Model(decoder_input, decoder_output, name='decoder')

# Step 5: Build Autoencoder
autoencoder_input = layers.Input(shape=(input_dim,))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = Model(autoencoder_input, decoded, name='autoencoder')

# Step 6: Compile and train
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(x_train, x_train, 
                epochs=20, 
                batch_size=256, 
                validation_data=(x_test, x_test))

# Step 7: Visualize results
reconstructed = autoencoder.predict(x_test[:10])
plt.figure(figsize=(20, 4))
for i in range(10):
    # Original
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    # Reconstructed
    plt.subplot(2, 10, i+11)
    plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle('Top: Original, Bottom: Reconstructed')
plt.show()
```

---

## Question 2
- [ ] Done

**Write a Python function that visualizes the latent space representation of data after going through an autoencoder.**

**Pipeline:**
```
1. Train autoencoder on labeled data
2. Encode all data points to latent space
3. Reduce to 2D if needed (PCA or direct if latent_dim=2)
4. Scatter plot colored by class label
```

**Code:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_latent_space(encoder, data, labels, latent_dim=32):
    """
    Visualize latent space representations.
    
    Args:
        encoder: Trained encoder model
        data: Input data (N, input_dim)
        labels: Class labels for coloring
        latent_dim: Dimension of latent space
    """
    # Step 1: Get latent representations
    latent_codes = encoder.predict(data)
    
    # Step 2: Reduce to 2D if needed
    if latent_dim > 2:
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_codes)
        print(f"Variance explained: {sum(pca.explained_variance_ratio_):.2%}")
    else:
        latent_2d = latent_codes
    
    # Step 3: Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Class')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Visualization')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return latent_2d

# Usage Example:
# Train autoencoder first (from Question 1)
# Then visualize:
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test_flat = x_test.reshape(-1, 784).astype('float32') / 255.0

# Visualize latent space
latent_2d = visualize_latent_space(encoder, x_test_flat[:5000], y_test[:5000])
```

---

## Question 3
- [ ] Done

**Create a denoising autoencoder using PyTorch that can clean noisy images.**

**Pipeline:**
```
1. Load clean images
2. Add noise to create noisy inputs
3. Build encoder-decoder network
4. Train: Input=noisy, Target=clean
5. Test: Feed noisy image → Get clean output
```

**Code:**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Step 1: Define Denoising Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()  # Output in [0,1]
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Step 2: Add noise function
def add_noise(images, noise_factor=0.3):
    noisy = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy, 0., 1.)  # Keep in [0,1]

# Step 3: Load data
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

# Step 4: Initialize model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Training loop
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (clean_images, _) in enumerate(train_loader):
        # Flatten and move to device
        clean_images = clean_images.view(-1, 784).to(device)
        
        # Add noise
        noisy_images = add_noise(clean_images)
        
        # Forward pass
        reconstructed = model(noisy_images)
        loss = criterion(reconstructed, clean_images)  # Compare with CLEAN
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')

# Step 6: Test and visualize
model.eval()
test_data = datasets.MNIST(root='./data', train=False, transform=transform)
test_images = test_data.data[:10].float().view(-1, 784) / 255.0
noisy_test = add_noise(test_images)

with torch.no_grad():
    denoised = model(noisy_test.to(device)).cpu()

# Plot results
fig, axes = plt.subplots(3, 10, figsize=(15, 5))
for i in range(10):
    axes[0, i].imshow(test_images[i].view(28, 28), cmap='gray')
    axes[1, i].imshow(noisy_test[i].view(28, 28), cmap='gray')
    axes[2, i].imshow(denoised[i].view(28, 28), cmap='gray')
    for ax in axes[:, i]: ax.axis('off')
axes[0, 0].set_ylabel('Original')
axes[1, 0].set_ylabel('Noisy')
axes[2, 0].set_ylabel('Denoised')
plt.tight_layout()
plt.show()
```

## Question 4
- [ ] Done

**Develop a variational autoencoder (VAE) using TensorFlow/Keras and demonstrate its generative capabilities.**

**Pipeline:**
```
1. Build encoder: Outputs mean (μ) and log-variance (log σ²)
2. Sampling layer: z = μ + σ × ε (reparameterization trick)
3. Build decoder: z → reconstructed image
4. Loss = Reconstruction + KL divergence
5. Generate: Sample z from N(0,1) → Decode
```

**Code:**

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
import matplotlib.pyplot as plt

# Step 1: Sampling layer (reparameterization trick)
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Step 2: Build VAE
latent_dim = 2  # 2D for easy visualization
input_dim = 784

# Encoder
encoder_inputs = layers.Input(shape=(input_dim,))
x = layers.Dense(256, activation='relu')(encoder_inputs)
x = layers.Dense(128, activation='relu')(x)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
z = Sampling()([z_mean, z_log_var])
encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

# Decoder
latent_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(128, activation='relu')(latent_inputs)
x = layers.Dense(256, activation='relu')(x)
decoder_outputs = layers.Dense(input_dim, activation='sigmoid')(x)
decoder = Model(latent_inputs, decoder_outputs, name='decoder')

# Step 3: VAE Model with custom training
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        
        # KL Divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        self.add_loss(kl_loss)
        return reconstructed

# Step 4: Compile and train
vae = VAE(encoder, decoder)
vae.compile(optimizer='adam', loss='mse')

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

vae.fit(x_train, x_train, epochs=30, batch_size=128, validation_data=(x_test, x_test))

# Step 5: Visualize latent space
z_mean, _, _ = encoder.predict(x_test)
plt.figure(figsize=(10, 8))
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test, cmap='tab10', alpha=0.5)
plt.colorbar()
plt.title('VAE Latent Space')
plt.show()

# Step 6: Generate new images
def generate_images(decoder, n=20, digit_size=28):
    """Generate images by sampling from latent space"""
    figure = np.zeros((digit_size * n, digit_size * n))
    
    # Sample from grid in latent space
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)
    
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
    
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.title('VAE Generated Digits')
    plt.axis('off')
    plt.show()

generate_images(decoder)
```

---

## Question 5
- [ ] Done

**Code a sparse autoencoder from scratch in Python to learn a representation of text data.**

**Pipeline:**
```
1. Preprocess text → TF-IDF vectors
2. Build autoencoder with L1 sparsity penalty
3. Train with reconstruction + sparsity loss
4. Extract sparse features from latent layer
```

**Code:**

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

# Step 1: Load and preprocess text data
categories = ['sci.space', 'rec.sport.baseball', 'comp.graphics']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(newsgroups.data).toarray().astype('float32')
print(f"Data shape: {X.shape}")  # (N, 1000)

# Step 2: Define Sparse Autoencoder
class SparseAutoencoder:
    def __init__(self, input_dim, hidden_dim, sparsity_weight=0.01, sparsity_target=0.05):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_weight = sparsity_weight
        self.sparsity_target = sparsity_target
        
        # Initialize weights (Xavier initialization)
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, input_dim) * scale
        self.b2 = np.zeros(input_dim)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Encoder
        self.z1 = X @ self.W1 + self.b1
        self.hidden = self.sigmoid(self.z1)
        
        # Decoder
        self.z2 = self.hidden @ self.W2 + self.b2
        self.output = self.sigmoid(self.z2)
        
        return self.output
    
    def kl_divergence(self, rho, rho_hat):
        """KL divergence for sparsity penalty"""
        return rho * np.log(rho / (rho_hat + 1e-10)) + \
               (1 - rho) * np.log((1 - rho) / (1 - rho_hat + 1e-10))
    
    def compute_loss(self, X, output):
        # Reconstruction loss (MSE)
        recon_loss = np.mean((X - output) ** 2)
        
        # Sparsity loss (KL divergence)
        rho_hat = np.mean(self.hidden, axis=0)  # Average activation per neuron
        sparsity_loss = np.sum(self.kl_divergence(self.sparsity_target, rho_hat))
        
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss
        return total_loss, recon_loss, sparsity_loss
    
    def backward(self, X, learning_rate=0.01):
        m = X.shape[0]
        
        # Sparsity gradient
        rho_hat = np.mean(self.hidden, axis=0)
        sparsity_grad = (-self.sparsity_target / (rho_hat + 1e-10) + 
                        (1 - self.sparsity_target) / (1 - rho_hat + 1e-10))
        
        # Output layer gradients
        d_output = (self.output - X) * self.sigmoid_derivative(self.output)
        dW2 = self.hidden.T @ d_output / m
        db2 = np.mean(d_output, axis=0)
        
        # Hidden layer gradients (with sparsity)
        d_hidden = (d_output @ self.W2.T + self.sparsity_weight * sparsity_grad / m) * \
                   self.sigmoid_derivative(self.hidden)
        dW1 = X.T @ d_hidden / m
        db1 = np.mean(d_hidden, axis=0)
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def fit(self, X, epochs=100, learning_rate=0.1, batch_size=32):
        for epoch in range(epochs):
            # Mini-batch training
            indices = np.random.permutation(len(X))
            for i in range(0, len(X), batch_size):
                batch = X[indices[i:i+batch_size]]
                self.forward(batch)
                self.backward(batch, learning_rate)
            
            # Compute loss on full data
            self.forward(X)
            total_loss, recon_loss, sparse_loss = self.compute_loss(X, self.output)
            
            if (epoch + 1) % 20 == 0:
                avg_activation = np.mean(self.hidden)
                print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, "
                      f"Recon={recon_loss:.4f}, Sparse={sparse_loss:.4f}, "
                      f"Avg Act={avg_activation:.4f}")
    
    def encode(self, X):
        """Get sparse latent representation"""
        z = X @ self.W1 + self.b1
        return self.sigmoid(z)

# Step 3: Train
sparse_ae = SparseAutoencoder(input_dim=1000, hidden_dim=100, 
                               sparsity_weight=0.1, sparsity_target=0.05)
sparse_ae.fit(X, epochs=100, learning_rate=0.5)

# Step 4: Get sparse features
sparse_features = sparse_ae.encode(X)
print(f"Sparse features shape: {sparse_features.shape}")
print(f"Average activation: {np.mean(sparse_features):.4f}")
print(f"Sparsity (% near zero): {np.mean(sparse_features < 0.1):.2%}")
```

---

## Question 6
- [ ] Done 

**Using scikit-learn, create a pipeline that includes feature extraction with an autoencoder followed by a classification model.**

**Pipeline:**
```
1. Create custom sklearn transformer wrapping autoencoder
2. Build pipeline: Autoencoder → Classifier
3. Train end-to-end
4. Evaluate
```

**Code:**

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, Model
import tensorflow as tf

# Step 1: Custom Autoencoder Transformer
class AutoencoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, latent_dim=32, epochs=50, batch_size=64):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.encoder = None
        self.autoencoder = None
    
    def _build_model(self, input_dim):
        # Encoder
        inputs = layers.Input(shape=(input_dim,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        latent = layers.Dense(self.latent_dim, activation='relu')(x)
        
        # Decoder
        x = layers.Dense(64, activation='relu')(latent)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(input_dim, activation='sigmoid')(x)
        
        # Models
        self.encoder = Model(inputs, latent)
        self.autoencoder = Model(inputs, outputs)
        self.autoencoder.compile(optimizer='adam', loss='mse')
    
    def fit(self, X, y=None):
        """Fit autoencoder on data"""
        self._build_model(X.shape[1])
        self.autoencoder.fit(X, X, 
                            epochs=self.epochs, 
                            batch_size=self.batch_size,
                            verbose=0)
        return self
    
    def transform(self, X):
        """Extract latent features"""
        return self.encoder.predict(X, verbose=0)
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

# Step 2: Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Use subset for faster demo
x_train, y_train = x_train[:10000], y_train[:10000]

# Step 3: Create Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('autoencoder', AutoencoderTransformer(latent_dim=32, epochs=30)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Step 4: Train pipeline
print("Training pipeline...")
pipeline.fit(x_train, y_train)

# Step 5: Evaluate
train_score = pipeline.score(x_train, y_train)
test_score = pipeline.score(x_test, y_test)
print(f"Train accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")

# Step 6: Compare with baseline (no autoencoder)
baseline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
baseline.fit(x_train, y_train)
baseline_score = baseline.score(x_test, y_test)
print(f"Baseline (no AE) accuracy: {baseline_score:.4f}")

# Cross-validation
cv_scores = cross_val_score(pipeline, x_train, y_train, cv=3)
print(f"CV scores: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

---

## Question 7
- [ ] Done

**Build a convolutional autoencoder for video frame prediction using TensorFlow/Keras.**

**Pipeline:**
```
1. Load video frames as sequence
2. Build Conv encoder-decoder
3. Train: Input = frame t, Target = frame t+1
4. Predict next frame given current frame
```

**Code:**

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

# Step 1: Generate synthetic video data (moving MNIST)
def generate_moving_mnist(n_sequences=1000, n_frames=10, img_size=64):
    """Generate moving MNIST sequences"""
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    
    sequences = []
    for _ in range(n_sequences):
        # Random digit
        digit_idx = np.random.randint(len(x_train))
        digit = x_train[digit_idx]
        
        # Random starting position and velocity
        x, y = np.random.randint(0, img_size-28, 2)
        vx, vy = np.random.randint(-2, 3, 2)
        
        frames = []
        for _ in range(n_frames):
            # Create frame
            frame = np.zeros((img_size, img_size))
            frame[y:y+28, x:x+28] = digit
            frames.append(frame)
            
            # Move digit (bounce off walls)
            x, y = x + vx, y + vy
            if x < 0 or x > img_size-28: vx = -vx; x += vx
            if y < 0 or y > img_size-28: vy = -vy; y += vy
        
        sequences.append(frames)
    
    return np.array(sequences) / 255.0

# Generate data
print("Generating data...")
data = generate_moving_mnist(n_sequences=500, n_frames=5)
print(f"Data shape: {data.shape}")  # (500, 5, 64, 64)

# Prepare input-output pairs: x=frame[t], y=frame[t+1]
X = data[:, :-1].reshape(-1, 64, 64, 1)  # All but last frame
Y = data[:, 1:].reshape(-1, 64, 64, 1)   # All but first frame
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# Step 2: Build Convolutional Autoencoder
def build_conv_autoencoder(input_shape=(64, 64, 1)):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)  # 32x32
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)  # 16x16
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)  # 8x8
    
    # Bottleneck
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    
    # Decoder
    x = layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(x)  # 16x16
    x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)   # 32x32
    x = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)   # 64x64
    outputs = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)
    
    model = Model(inputs, outputs)
    return model

# Step 3: Build and train
model = build_conv_autoencoder()
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train
history = model.fit(X, Y, epochs=20, batch_size=32, validation_split=0.1)

# Step 4: Predict and visualize
test_seq = data[0:1]  # Take first sequence
predictions = [test_seq[0, 0]]  # Start with first frame

# Auto-regressive prediction
current_frame = test_seq[0, 0:1].reshape(1, 64, 64, 1)
for _ in range(4):
    next_frame = model.predict(current_frame, verbose=0)
    predictions.append(next_frame[0, :, :, 0])
    current_frame = next_frame

# Plot
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(5):
    axes[0, i].imshow(test_seq[0, i], cmap='gray')
    axes[0, i].set_title(f'Ground Truth {i+1}')
    axes[0, i].axis('off')
    
    axes[1, i].imshow(predictions[i], cmap='gray')
    axes[1, i].set_title(f'Predicted {i+1}')
    axes[1, i].axis('off')

plt.suptitle('Video Frame Prediction')
plt.tight_layout()
plt.show()
```

---

## Question 8
- [ ] Done

**Implement a stacked autoencoder for multi-label classification and compare its performance with a basic neural network.**

**Pipeline:**
```
1. Greedy layer-wise pretraining
2. Stack encoders, add classification head
3. Fine-tune end-to-end
4. Compare with randomly initialized network
```

**Code:**

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Step 2: Greedy Layer-wise Pretraining
def train_single_autoencoder(X, input_dim, hidden_dim, epochs=20):
    """Train single autoencoder layer"""
    inputs = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(hidden_dim, activation='relu')(inputs)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    
    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X, X, epochs=epochs, batch_size=256, verbose=0)
    
    return encoder, autoencoder

print("Layer-wise pretraining...")

# Layer 1: 784 → 256
encoder1, _ = train_single_autoencoder(x_train, 784, 256)
h1 = encoder1.predict(x_train, verbose=0)
print("Layer 1 trained: 784 → 256")

# Layer 2: 256 → 128
encoder2, _ = train_single_autoencoder(h1, 256, 128)
h2 = encoder2.predict(h1, verbose=0)
print("Layer 2 trained: 256 → 128")

# Layer 3: 128 → 64
encoder3, _ = train_single_autoencoder(h2, 128, 64)
print("Layer 3 trained: 128 → 64")

# Step 3: Build Stacked Autoencoder Classifier
def build_stacked_classifier(encoder1, encoder2, encoder3, num_classes=10):
    """Build classifier using pretrained encoders"""
    inputs = layers.Input(shape=(784,))
    
    # Use pretrained weights
    x = layers.Dense(256, activation='relu',
                     weights=encoder1.layers[1].get_weights())(inputs)
    x = layers.Dense(128, activation='relu',
                     weights=encoder2.layers[1].get_weights())(x)
    x = layers.Dense(64, activation='relu',
                     weights=encoder3.layers[1].get_weights())(x)
    
    # Classification head
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

# Build pretrained model
stacked_model = build_stacked_classifier(encoder1, encoder2, encoder3)
stacked_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])

# Step 4: Build Baseline (random init) with same architecture
def build_baseline_classifier():
    inputs = layers.Input(shape=(784,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    return Model(inputs, outputs)

baseline_model = build_baseline_classifier()
baseline_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

# Step 5: Train both models
print("\nFine-tuning stacked autoencoder...")
history_stacked = stacked_model.fit(x_train, y_train, epochs=20, batch_size=128,
                                     validation_data=(x_test, y_test), verbose=0)

print("Training baseline...")
history_baseline = baseline_model.fit(x_train, y_train, epochs=20, batch_size=128,
                                       validation_data=(x_test, y_test), verbose=0)

# Step 6: Compare results
stacked_acc = stacked_model.evaluate(x_test, y_test, verbose=0)[1]
baseline_acc = baseline_model.evaluate(x_test, y_test, verbose=0)[1]

print(f"\n{'='*50}")
print(f"Stacked Autoencoder Accuracy: {stacked_acc:.4f}")
print(f"Baseline (Random Init) Accuracy: {baseline_acc:.4f}")
print(f"Improvement: {(stacked_acc - baseline_acc)*100:.2f}%")

# Plot learning curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_stacked.history['accuracy'], label='Stacked AE - Train')
plt.plot(history_stacked.history['val_accuracy'], label='Stacked AE - Val')
plt.plot(history_baseline.history['accuracy'], label='Baseline - Train', linestyle='--')
plt.plot(history_baseline.history['val_accuracy'], label='Baseline - Val', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training Progress')

plt.subplot(1, 2, 2)
plt.bar(['Stacked AE', 'Baseline'], [stacked_acc, baseline_acc])
plt.ylabel('Test Accuracy')
plt.title('Final Comparison')

plt.tight_layout()
plt.show()
```

---

## Question 9
- [ ] Done

**Train a Sparse Autoencoder on Residual-Stream Activations (full-batch GD).**

**Pipeline:**
```
1. Forward: pre -> ReLU -> reconstruction
2. Loss: MSE over all elements + L1 sparsity over samples
3. Backprop with ReLU mask and L1 term
4. Update W_enc, W_dec, b_enc for num_steps
5. Return final loss on the same batch
```

**Code:**

```python
import numpy as np

def train_sae(X, W_enc, W_dec, b_enc, lr, l1_coef, num_steps):
    X = np.array(X, dtype=float)
    W_enc = np.array(W_enc, dtype=float)
    W_dec = np.array(W_dec, dtype=float)
    b_enc = np.array(b_enc, dtype=float)
    N, D = X.shape

    for _ in range(num_steps):
        pre   = X @ W_enc + b_enc
        z     = np.maximum(0, pre)
        x_hat = z @ W_dec

        d_xhat  = 2 * (x_hat - X) / (N * D)
        d_W_dec = z.T @ d_xhat
        d_z     = d_xhat @ W_dec.T + l1_coef / N
        d_pre   = d_z * (pre > 0)
        d_W_enc = X.T @ d_pre
        d_b_enc = np.sum(d_pre, axis=0)

        W_enc -= lr * d_W_enc
        W_dec -= lr * d_W_dec
        b_enc -= lr * d_b_enc

    pre   = X @ W_enc + b_enc
    z     = np.maximum(0, pre)
    x_hat = z @ W_dec
    return float(np.mean((x_hat - X)**2) + l1_coef * np.mean(np.sum(z, axis=1)))
    
# Example usage
X = np.random.randn(8, 4)
W_enc = np.random.randn(4, 6)
W_dec = np.random.randn(6, 4)
b_enc = np.zeros(6)
loss = train_sae(X, W_enc, W_dec, b_enc, lr=0.1, l1_coef=0.05, num_steps=50)
print(loss)
```
