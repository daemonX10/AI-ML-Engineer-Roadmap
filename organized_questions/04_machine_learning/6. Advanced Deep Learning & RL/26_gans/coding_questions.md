# GANs Interview Questions - Coding Questions

---

## Question 1: Implement a simple GAN model in TensorFlow/Keras to generate new samples from a given dataset
- [ ] Done

### Pipeline
1. Define Generator: noise → fake image
2. Define Discriminator: image → real/fake probability
3. Compile GAN: freeze D when training G
4. Training loop: alternate D and G training
5. Generate samples from trained model

### Code
```python
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# Generator: maps noise to image
def build_generator(latent_dim=100):
    model = tf.keras.Sequential([
        layers.Dense(256, input_dim=latent_dim),
        layers.LeakyReLU(0.2),
        layers.BatchNormalization(),
        
        layers.Dense(512),
        layers.LeakyReLU(0.2),
        layers.BatchNormalization(),
        
        layers.Dense(1024),
        layers.LeakyReLU(0.2),
        layers.BatchNormalization(),
        
        layers.Dense(28 * 28 * 1, activation='tanh'),
        layers.Reshape((28, 28, 1))
    ])
    return model

# Discriminator: classifies real vs fake
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        
        layers.Dense(512),
        layers.LeakyReLU(0.2),
        
        layers.Dense(256),
        layers.LeakyReLU(0.2),
        
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Build and compile
latent_dim = 100
generator = build_generator(latent_dim)
discriminator = build_discriminator()

discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Combined model (for training generator)
discriminator.trainable = False
gan_input = layers.Input(shape=(latent_dim,))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)
gan = Model(gan_input, gan_output)
gan.compile(optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
            loss='binary_crossentropy')

# Training function
def train_gan(epochs, batch_size=128):
    # Load data (MNIST example)
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = X_train / 127.5 - 1.0  # Normalize to [-1, 1]
    X_train = X_train.reshape(-1, 28, 28, 1)
    
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise, verbose=0)
        
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
        
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real_labels)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D Loss={d_loss:.4f}, G Loss={g_loss:.4f}")

# Generate new samples
def generate_samples(n_samples=10):
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    generated = generator.predict(noise)
    return generated

# Train and generate
train_gan(epochs=1000)
samples = generate_samples(10)
```

---

## Question 2: Using PyTorch, code a discriminator network that can classify between real and generated images
- [ ] Done

### Pipeline
1. Define conv layers for downsampling
2. Add LeakyReLU activations
3. Flatten and output single probability
4. Include spectral normalization for stability

### Code
```python
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features_d=64):
        super().__init__()
        
        # Input: N x img_channels x 64 x 64
        self.model = nn.Sequential(
            # Layer 1: No batch norm in first layer
            spectral_norm(nn.Conv2d(img_channels, features_d, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            # Output: N x 64 x 32 x 32
            
            # Layer 2
            spectral_norm(nn.Conv2d(features_d, features_d * 2, 4, 2, 1)),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2),
            # Output: N x 128 x 16 x 16
            
            # Layer 3
            spectral_norm(nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1)),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2),
            # Output: N x 256 x 8 x 8
            
            # Layer 4
            spectral_norm(nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1)),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2),
            # Output: N x 512 x 4 x 4
            
            # Final layer: output single value
            spectral_norm(nn.Conv2d(features_d * 8, 1, 4, 1, 0)),
            nn.Sigmoid()
            # Output: N x 1 x 1 x 1
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 1)

# Usage example
discriminator = Discriminator(img_channels=3, features_d=64)

# Test with random input
real_images = torch.randn(16, 3, 64, 64)  # Batch of 16 images
fake_images = torch.randn(16, 3, 64, 64)

real_preds = discriminator(real_images)  # Should be close to 1
fake_preds = discriminator(fake_images)  # Should be close to 0

print(f"Real predictions shape: {real_preds.shape}")  # [16, 1]
print(f"Predictions range: [{real_preds.min():.3f}, {real_preds.max():.3f}]")

# Training step example
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

def train_discriminator_step(real_images, fake_images):
    optimizer.zero_grad()
    
    # Real images should be classified as 1
    real_labels = torch.ones(real_images.size(0), 1)
    real_output = discriminator(real_images)
    real_loss = criterion(real_output, real_labels)
    
    # Fake images should be classified as 0
    fake_labels = torch.zeros(fake_images.size(0), 1)
    fake_output = discriminator(fake_images.detach())
    fake_loss = criterion(fake_output, fake_labels)
    
    # Total loss
    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer.step()
    
    return d_loss.item()
```

---

## Question 3: Create a Python script using NumPy to visualize the loss of the generator and discriminator during training
- [ ] Done

### Pipeline
1. Store losses during training
2. Create smooth moving average
3. Plot both losses on same graph
4. Add annotations for key events

### Code
```python
import numpy as np
import matplotlib.pyplot as plt

class GANLossTracker:
    def __init__(self):
        self.g_losses = []
        self.d_losses = []
        self.d_real_acc = []
        self.d_fake_acc = []
        self.epochs = []
    
    def record(self, epoch, g_loss, d_loss, d_real_acc=None, d_fake_acc=None):
        self.epochs.append(epoch)
        self.g_losses.append(g_loss)
        self.d_losses.append(d_loss)
        if d_real_acc is not None:
            self.d_real_acc.append(d_real_acc)
            self.d_fake_acc.append(d_fake_acc)
    
    def moving_average(self, data, window=50):
        """Compute moving average for smoother visualization"""
        if len(data) < window:
            return data
        cumsum = np.cumsum(data)
        cumsum[window:] = cumsum[window:] - cumsum[:-window]
        return cumsum[window - 1:] / window
    
    def plot_losses(self, save_path=None):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Raw losses
        ax1 = axes[0]
        ax1.plot(self.epochs, self.g_losses, 'b-', alpha=0.3, label='G Loss (raw)')
        ax1.plot(self.epochs, self.d_losses, 'r-', alpha=0.3, label='D Loss (raw)')
        
        # Smoothed losses
        if len(self.g_losses) > 50:
            g_smooth = self.moving_average(self.g_losses)
            d_smooth = self.moving_average(self.d_losses)
            smooth_epochs = self.epochs[49:]
            ax1.plot(smooth_epochs, g_smooth, 'b-', linewidth=2, label='G Loss (smooth)')
            ax1.plot(smooth_epochs, d_smooth, 'r-', linewidth=2, label='D Loss (smooth)')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('GAN Training Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Discriminator accuracy (if available)
        if self.d_real_acc:
            ax2 = axes[1]
            ax2.plot(self.epochs, self.d_real_acc, 'g-', label='D Acc on Real')
            ax2.plot(self.epochs, self.d_fake_acc, 'm-', label='D Acc on Fake')
            ax2.axhline(y=0.5, color='k', linestyle='--', label='Random (0.5)')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Discriminator Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
    
    def detect_issues(self):
        """Detect common training issues"""
        issues = []
        
        recent_g = self.g_losses[-100:] if len(self.g_losses) > 100 else self.g_losses
        recent_d = self.d_losses[-100:] if len(self.d_losses) > 100 else self.d_losses
        
        # Mode collapse: G loss very low, D loss high
        if np.mean(recent_g) < 0.1 and np.mean(recent_d) > 1.0:
            issues.append("Possible mode collapse: G winning too easily")
        
        # D too strong: G loss very high
        if np.mean(recent_g) > 5.0:
            issues.append("D too strong: G cannot learn")
        
        # Training collapse: both losses exploding
        if np.mean(recent_g) > 10 and np.mean(recent_d) > 10:
            issues.append("Training collapse: losses diverging")
        
        # Oscillation
        g_std = np.std(recent_g)
        if g_std > 2.0:
            issues.append("High oscillation in G loss")
        
        return issues

# Example usage
tracker = GANLossTracker()

# Simulate training (replace with actual training loop)
np.random.seed(42)
for epoch in range(1000):
    # Simulated losses (replace with actual values)
    g_loss = 2.0 * np.exp(-epoch/300) + np.random.normal(0, 0.2) + 0.7
    d_loss = 0.5 + np.random.normal(0, 0.1)
    d_real_acc = 0.5 + 0.3 * (1 - np.exp(-epoch/200)) + np.random.normal(0, 0.05)
    d_fake_acc = 0.5 + 0.2 * (1 - np.exp(-epoch/200)) + np.random.normal(0, 0.05)
    
    tracker.record(epoch, g_loss, d_loss, d_real_acc, d_fake_acc)

# Visualize
tracker.plot_losses()

# Check for issues
issues = tracker.detect_issues()
for issue in issues:
    print(f"Warning: {issue}")
```

---

## Question 4: Code a DCGAN in TensorFlow/Keras and train it on a dataset of images to generate new ones
- [ ] Done

### Pipeline
1. Build Generator with transposed convolutions
2. Build Discriminator with strided convolutions
3. Follow DCGAN guidelines (BatchNorm, LeakyReLU, etc.)
4. Implement training loop
5. Generate and visualize samples

### Code
```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# DCGAN Generator
def build_dcgan_generator(latent_dim=100):
    model = tf.keras.Sequential([
        # Input: latent_dim -> 4x4x512
        layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(latent_dim,)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Reshape((4, 4, 512)),
        
        # 4x4 -> 8x8
        layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # 8x8 -> 16x16
        layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # 16x16 -> 32x32
        layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # 32x32 -> 64x64 (output)
        layers.Conv2DTranspose(3, 4, strides=2, padding='same', use_bias=False, 
                               activation='tanh')
    ])
    return model

# DCGAN Discriminator
def build_dcgan_discriminator():
    model = tf.keras.Sequential([
        # 64x64 -> 32x32
        layers.Conv2D(64, 4, strides=2, padding='same', input_shape=(64, 64, 3)),
        layers.LeakyReLU(0.2),
        
        # 32x32 -> 16x16
        layers.Conv2D(128, 4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        # 16x16 -> 8x8
        layers.Conv2D(256, 4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        # 8x8 -> 4x4
        layers.Conv2D(512, 4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        # 4x4 -> 1
        layers.Conv2D(1, 4, strides=1, padding='valid'),
        layers.Flatten(),
        layers.Sigmoid()
    ])
    return model

# Build models
latent_dim = 100
generator = build_dcgan_generator(latent_dim)
discriminator = build_dcgan_discriminator()

# Optimizers
g_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
loss_fn = tf.keras.losses.BinaryCrossentropy()

# Training step
@tf.function
def train_step(real_images):
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([batch_size, latent_dim])
    
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        # Generate fake images
        fake_images = generator(noise, training=True)
        
        # Discriminator outputs
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)
        
        # Discriminator loss
        real_loss = loss_fn(tf.ones_like(real_output), real_output)
        fake_loss = loss_fn(tf.zeros_like(fake_output), fake_output)
        d_loss = real_loss + fake_loss
        
        # Generator loss
        g_loss = loss_fn(tf.ones_like(fake_output), fake_output)
    
    # Apply gradients
    d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
    
    d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
    g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    
    return d_loss, g_loss

# Training loop
def train_dcgan(dataset, epochs):
    for epoch in range(epochs):
        for batch in dataset:
            d_loss, g_loss = train_step(batch)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: D Loss={d_loss:.4f}, G Loss={g_loss:.4f}")
            generate_and_save_images(generator, epoch)

# Generate sample images
def generate_and_save_images(generator, epoch, n=16):
    noise = tf.random.normal([n, latent_dim])
    images = generator(noise, training=False)
    images = (images + 1) / 2.0  # Rescale to [0, 1]
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.axis('off')
    plt.savefig(f'dcgan_epoch_{epoch}.png')
    plt.close()

# Prepare dataset (example with CIFAR-10)
(train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()
train_images = tf.image.resize(train_images, (64, 64))
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]
dataset = tf.data.Dataset.from_tensor_slices(train_images)
dataset = dataset.shuffle(10000).batch(64)

# Train
train_dcgan(dataset, epochs=100)
```

---

## Question 5: Implement a WGAN with gradient penalty in PyTorch and demonstrate its stability compared to standard GANs
- [ ] Done

### Pipeline
1. Build Generator and Critic (not Discriminator)
2. Implement gradient penalty function
3. Train Critic more steps than Generator
4. Compare loss curves for stability

### Code
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28 * img_channels),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

# Critic (not Discriminator - no sigmoid)
class Critic(nn.Module):
    def __init__(self, img_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28 * img_channels, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)  # No sigmoid - unbounded output
        )
    
    def forward(self, x):
        return self.model(x)

# Gradient Penalty
def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    
    # Random interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)
    
    # Critic output on interpolated
    critic_output = critic(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=critic_output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_output),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Flatten and compute norm
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    
    # Penalty: (||gradient|| - 1)^2
    gp = ((gradient_norm - 1) ** 2).mean()
    return gp

# Training
def train_wgan_gp(dataloader, epochs, n_critic=5, lambda_gp=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 100
    
    generator = Generator(latent_dim).to(device)
    critic = Critic().to(device)
    
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
    c_optimizer = optim.Adam(critic.parameters(), lr=0.0001, betas=(0.0, 0.9))
    
    g_losses, c_losses = [], []
    
    for epoch in range(epochs):
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            batch_size = real.size(0)
            
            # Train Critic (n_critic times)
            for _ in range(n_critic):
                z = torch.randn(batch_size, latent_dim, device=device)
                fake = generator(z)
                
                c_real = critic(real).mean()
                c_fake = critic(fake.detach()).mean()
                gp = gradient_penalty(critic, real, fake.detach(), device)
                
                c_loss = c_fake - c_real + lambda_gp * gp
                
                c_optimizer.zero_grad()
                c_loss.backward()
                c_optimizer.step()
            
            # Train Generator
            z = torch.randn(batch_size, latent_dim, device=device)
            fake = generator(z)
            g_loss = -critic(fake).mean()
            
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
            g_losses.append(g_loss.item())
            c_losses.append(c_loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: C Loss={c_loss.item():.4f}, G Loss={g_loss.item():.4f}")
    
    return generator, g_losses, c_losses

# Compare stability: plot loss curves
def plot_comparison(wgan_g_losses, wgan_c_losses, gan_g_losses, gan_d_losses):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Standard GAN
    axes[0].plot(gan_g_losses, 'b-', alpha=0.5, label='G Loss')
    axes[0].plot(gan_d_losses, 'r-', alpha=0.5, label='D Loss')
    axes[0].set_title('Standard GAN (may be unstable)')
    axes[0].legend()
    
    # WGAN-GP
    axes[1].plot(wgan_g_losses, 'b-', alpha=0.5, label='G Loss')
    axes[1].plot(wgan_c_losses, 'r-', alpha=0.5, label='C Loss')
    axes[1].set_title('WGAN-GP (more stable)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

# Usage
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

generator, g_losses, c_losses = train_wgan_gp(dataloader, epochs=50)
```

---

## Question 6: Build a Conditional GAN in TensorFlow/Keras to generate images conditioned on class labels
- [ ] Done

### Pipeline
1. Embed class labels
2. Concatenate embedding with noise for Generator
3. Concatenate embedding with image for Discriminator
4. Train with label conditioning
5. Generate specific class images

### Code
```python
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# Conditional Generator
def build_cgan_generator(latent_dim=100, n_classes=10):
    # Noise input
    noise_input = layers.Input(shape=(latent_dim,))
    
    # Label input
    label_input = layers.Input(shape=(1,), dtype='int32')
    label_embedding = layers.Embedding(n_classes, 50)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    
    # Concatenate noise and label
    combined = layers.Concatenate()([noise_input, label_embedding])
    
    # Generator network
    x = layers.Dense(256)(combined)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(28 * 28, activation='tanh')(x)
    output = layers.Reshape((28, 28, 1))(x)
    
    return Model([noise_input, label_input], output)

# Conditional Discriminator
def build_cgan_discriminator(n_classes=10):
    # Image input
    img_input = layers.Input(shape=(28, 28, 1))
    img_flat = layers.Flatten()(img_input)
    
    # Label input
    label_input = layers.Input(shape=(1,), dtype='int32')
    label_embedding = layers.Embedding(n_classes, 50)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    
    # Concatenate image and label
    combined = layers.Concatenate()([img_flat, label_embedding])
    
    # Discriminator network
    x = layers.Dense(512)(combined)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(0.2)(x)
    
    output = layers.Dense(1, activation='sigmoid')(x)
    
    return Model([img_input, label_input], output)

# Build models
latent_dim = 100
n_classes = 10

generator = build_cgan_generator(latent_dim, n_classes)
discriminator = build_cgan_discriminator(n_classes)

discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
    loss='binary_crossentropy'
)

# Combined model
discriminator.trainable = False
noise_input = layers.Input(shape=(latent_dim,))
label_input = layers.Input(shape=(1,), dtype='int32')
fake_img = generator([noise_input, label_input])
validity = discriminator([fake_img, label_input])
cgan = Model([noise_input, label_input], validity)
cgan.compile(optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
             loss='binary_crossentropy')

# Training
def train_cgan(epochs, batch_size=128):
    (X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = (X_train.astype('float32') - 127.5) / 127.5
    X_train = X_train.reshape(-1, 28, 28, 1)
    
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        # Random batch
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        img_labels = y_train[idx].reshape(-1, 1)
        
        # Generate fake images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = generator.predict([noise, img_labels], verbose=0)
        
        # Train Discriminator
        d_loss_real = discriminator.train_on_batch([real_imgs, img_labels], real_labels)
        d_loss_fake = discriminator.train_on_batch([fake_imgs, img_labels], fake_labels)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        sampled_labels = np.random.randint(0, n_classes, batch_size).reshape(-1, 1)
        g_loss = cgan.train_on_batch([noise, sampled_labels], real_labels)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D Loss={d_loss:.4f}, G Loss={g_loss:.4f}")

# Generate specific digits
def generate_digits(digit, n_samples=10):
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    labels = np.full((n_samples, 1), digit)
    generated = generator.predict([noise, labels])
    return generated

# Train
train_cgan(epochs=5000)

# Generate all digits 0-9
for digit in range(10):
    samples = generate_digits(digit, 5)
    print(f"Generated digit {digit}")
```

---

## Question 7: Write a script to monitor and report mode collapse during GAN training
- [ ] Done

### Pipeline
1. Track generated sample diversity metrics
2. Compute pairwise distances between generated samples
3. Compare to training data diversity
4. Alert when diversity drops significantly

### Code
```python
import numpy as np
import torch
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

class ModeCollapseMonitor:
    def __init__(self, generator, latent_dim, device='cpu'):
        self.generator = generator
        self.latent_dim = latent_dim
        self.device = device
        self.diversity_history = []
        self.unique_modes_history = []
    
    def compute_diversity(self, n_samples=500):
        """Compute diversity as average pairwise distance"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.generator(z)
            samples_flat = samples.view(n_samples, -1).cpu().numpy()
        
        # Pairwise distances
        distances = pdist(samples_flat, metric='euclidean')
        diversity = np.mean(distances)
        
        return diversity
    
    def estimate_modes(self, n_samples=500, n_bins=50):
        """Estimate number of modes using histogram binning"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.generator(z)
            samples_flat = samples.view(n_samples, -1).cpu().numpy()
        
        # PCA to reduce dimensionality
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        samples_2d = pca.fit_transform(samples_flat)
        
        # 2D histogram
        hist, _, _ = np.histogram2d(samples_2d[:, 0], samples_2d[:, 1], bins=n_bins)
        
        # Count non-empty bins as modes
        n_modes = np.sum(hist > 0)
        return n_modes
    
    def check_mode_collapse(self, epoch, threshold_ratio=0.5):
        """Check for mode collapse and return status"""
        diversity = self.compute_diversity()
        n_modes = self.estimate_modes()
        
        self.diversity_history.append(diversity)
        self.unique_modes_history.append(n_modes)
        
        status = {
            'epoch': epoch,
            'diversity': diversity,
            'n_modes': n_modes,
            'is_collapsed': False,
            'warning': None
        }
        
        # Check against history
        if len(self.diversity_history) > 10:
            initial_diversity = np.mean(self.diversity_history[:10])
            
            if diversity < initial_diversity * threshold_ratio:
                status['is_collapsed'] = True
                status['warning'] = f"Diversity dropped to {diversity:.2f} from {initial_diversity:.2f}"
            
            initial_modes = np.mean(self.unique_modes_history[:10])
            if n_modes < initial_modes * threshold_ratio:
                status['is_collapsed'] = True
                status['warning'] = f"Modes dropped to {n_modes} from {initial_modes:.0f}"
        
        return status
    
    def plot_history(self):
        """Visualize diversity and modes over training"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.diversity_history)
        axes[0].set_title('Sample Diversity Over Training')
        axes[0].set_xlabel('Check')
        axes[0].set_ylabel('Mean Pairwise Distance')
        
        axes[1].plot(self.unique_modes_history)
        axes[1].set_title('Estimated Modes Over Training')
        axes[1].set_xlabel('Check')
        axes[1].set_ylabel('Number of Modes')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """Generate final report"""
        report = f"""
Mode Collapse Report
====================
Total Checks: {len(self.diversity_history)}

Diversity:
  Initial: {np.mean(self.diversity_history[:10]):.4f}
  Final: {self.diversity_history[-1]:.4f}
  Min: {np.min(self.diversity_history):.4f}
  Max: {np.max(self.diversity_history):.4f}

Modes:
  Initial: {np.mean(self.unique_modes_history[:10]):.0f}
  Final: {self.unique_modes_history[-1]}
  Min: {np.min(self.unique_modes_history)}
  Max: {np.max(self.unique_modes_history)}

Status: {'COLLAPSED' if self.diversity_history[-1] < np.mean(self.diversity_history[:10]) * 0.5 else 'HEALTHY'}
"""
        return report

# Usage example
# monitor = ModeCollapseMonitor(generator, latent_dim=100, device='cuda')

# During training:
# for epoch in range(epochs):
#     train_step(...)
#     
#     if epoch % 100 == 0:
#         status = monitor.check_mode_collapse(epoch)
#         if status['is_collapsed']:
#             print(f"WARNING: {status['warning']}")
#             # Take action: reduce learning rate, add regularization, etc.

# After training:
# monitor.plot_history()
# print(monitor.generate_report())
```

---

## Question 8: Develop a CycleGAN in PyTorch for unpaired image-to-image translation
- [ ] Done

### Pipeline
1. Build two Generators (A→B, B→A)
2. Build two Discriminators (for A and B)
3. Implement cycle consistency loss
4. Implement identity loss (optional)
5. Train on unpaired images

### Code
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)

# Generator (ResNet-based)
class CycleGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual=9):
        super().__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling
        self.down = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residual = nn.Sequential(
            *[ResidualBlock(256) for _ in range(n_residual)]
        )
        
        # Upsampling
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(64, out_channels, 7, padding=3),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.initial(x)
        x = self.down(x)
        x = self.residual(x)
        x = self.up(x)
        return self.output(x)

# Discriminator (PatchGAN)
class CycleDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

# CycleGAN Model
class CycleGAN:
    def __init__(self, device='cuda'):
        self.device = device
        
        # Generators
        self.G_AB = CycleGenerator().to(device)  # A -> B
        self.G_BA = CycleGenerator().to(device)  # B -> A
        
        # Discriminators
        self.D_A = CycleDiscriminator().to(device)
        self.D_B = CycleDiscriminator().to(device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )
        self.d_optimizer = optim.Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )
        
        # Losses
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Loss weights
        self.lambda_cyc = 10.0
        self.lambda_id = 5.0
    
    def train_step(self, real_A, real_B):
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)
        
        # Generate fake images
        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)
        
        # Cycle reconstruction
        rec_A = self.G_BA(fake_B)
        rec_B = self.G_AB(fake_A)
        
        # Identity mapping (optional)
        id_A = self.G_BA(real_A)
        id_B = self.G_AB(real_B)
        
        # =========== Train Generators ===========
        self.g_optimizer.zero_grad()
        
        # Adversarial loss
        pred_fake_B = self.D_B(fake_B)
        pred_fake_A = self.D_A(fake_A)
        g_loss_AB = self.mse_loss(pred_fake_B, torch.ones_like(pred_fake_B))
        g_loss_BA = self.mse_loss(pred_fake_A, torch.ones_like(pred_fake_A))
        
        # Cycle consistency loss
        cyc_loss_A = self.l1_loss(rec_A, real_A)
        cyc_loss_B = self.l1_loss(rec_B, real_B)
        
        # Identity loss
        id_loss_A = self.l1_loss(id_A, real_A)
        id_loss_B = self.l1_loss(id_B, real_B)
        
        # Total generator loss
        g_loss = (g_loss_AB + g_loss_BA + 
                  self.lambda_cyc * (cyc_loss_A + cyc_loss_B) +
                  self.lambda_id * (id_loss_A + id_loss_B))
        
        g_loss.backward()
        self.g_optimizer.step()
        
        # =========== Train Discriminators ===========
        self.d_optimizer.zero_grad()
        
        # D_A
        pred_real_A = self.D_A(real_A)
        pred_fake_A = self.D_A(fake_A.detach())
        d_loss_A = 0.5 * (self.mse_loss(pred_real_A, torch.ones_like(pred_real_A)) +
                         self.mse_loss(pred_fake_A, torch.zeros_like(pred_fake_A)))
        
        # D_B
        pred_real_B = self.D_B(real_B)
        pred_fake_B = self.D_B(fake_B.detach())
        d_loss_B = 0.5 * (self.mse_loss(pred_real_B, torch.ones_like(pred_real_B)) +
                         self.mse_loss(pred_fake_B, torch.zeros_like(pred_fake_B)))
        
        d_loss = d_loss_A + d_loss_B
        d_loss.backward()
        self.d_optimizer.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'cyc_loss': (cyc_loss_A + cyc_loss_B).item()
        }
    
    def translate_A_to_B(self, image_A):
        self.G_AB.eval()
        with torch.no_grad():
            return self.G_AB(image_A.to(self.device))
    
    def translate_B_to_A(self, image_B):
        self.G_BA.eval()
        with torch.no_grad():
            return self.G_BA(image_B.to(self.device))

# Usage
# cyclegan = CycleGAN(device='cuda')
# for epoch in range(epochs):
#     for real_A, real_B in zip(dataloader_A, dataloader_B):
#         losses = cyclegan.train_step(real_A, real_B)
#         print(losses)
```

---

## Question 9: Implement a GAN using TensorFlow/Keras capable of generating high-resolution images of human faces (inspired by StyleGAN)
- [ ] Done

### Pipeline
1. Build Mapping Network (z → w)
2. Build Synthesis Network with style injection
3. Implement progressive training (optional, simplified here)
4. Generate high-resolution faces

### Code
```python
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# Mapping Network
def build_mapping_network(latent_dim=512, n_layers=8):
    z_input = layers.Input(shape=(latent_dim,))
    
    x = z_input
    for _ in range(n_layers):
        x = layers.Dense(512)(x)
        x = layers.LeakyReLU(0.2)(x)
    
    w = layers.Dense(512)(x)  # W latent code
    
    return Model(z_input, w, name='mapping_network')

# Style injection layer (Adaptive Instance Normalization)
class AdaIN(layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.style_scale = layers.Dense(channels)
        self.style_bias = layers.Dense(channels)
    
    def call(self, inputs):
        x, w = inputs
        
        # Instance normalization
        mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        std = tf.math.reduce_std(x, axis=[1, 2], keepdims=True) + 1e-8
        x_norm = (x - mean) / std
        
        # Style modulation
        scale = self.style_scale(w)
        bias = self.style_bias(w)
        
        scale = tf.reshape(scale, [-1, 1, 1, self.channels])
        bias = tf.reshape(bias, [-1, 1, 1, self.channels])
        
        return x_norm * (1 + scale) + bias

# Synthesis block
class SynthesisBlock(layers.Layer):
    def __init__(self, out_channels, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.out_channels = out_channels
        
        self.conv1 = layers.Conv2D(out_channels, 3, padding='same')
        self.conv2 = layers.Conv2D(out_channels, 3, padding='same')
        self.adain1 = AdaIN(out_channels)
        self.adain2 = AdaIN(out_channels)
        self.noise1 = layers.Dense(out_channels)
        self.noise2 = layers.Dense(out_channels)
    
    def call(self, inputs):
        x, w = inputs
        
        if self.upsample:
            x = tf.image.resize(x, [x.shape[1]*2, x.shape[2]*2], method='bilinear')
        
        # First convolution + style
        x = self.conv1(x)
        noise = tf.random.normal([tf.shape(x)[0], x.shape[1], x.shape[2], 1])
        x = x + self.noise1(tf.reshape(noise, [-1, x.shape[1]*x.shape[2]]))[:, tf.newaxis, tf.newaxis, :]
        x = self.adain1([x, w])
        x = layers.LeakyReLU(0.2)(x)
        
        # Second convolution + style
        x = self.conv2(x)
        noise = tf.random.normal([tf.shape(x)[0], x.shape[1], x.shape[2], 1])
        x = x + self.noise2(tf.reshape(noise, [-1, x.shape[1]*x.shape[2]]))[:, tf.newaxis, tf.newaxis, :]
        x = self.adain2([x, w])
        x = layers.LeakyReLU(0.2)(x)
        
        return x

# Simplified StyleGAN Generator (64x64 for demonstration)
def build_style_generator(latent_dim=512):
    w_input = layers.Input(shape=(512,))  # Style vector from mapping network
    
    # Constant input (learnable)
    const = tf.Variable(tf.random.normal([1, 4, 4, 512]), trainable=True)
    x = tf.tile(const, [tf.shape(w_input)[0], 1, 1, 1])
    
    # Synthesis blocks
    # 4x4 -> 8x8
    x = SynthesisBlock(512)([x, w_input])
    # 8x8 -> 16x16
    x = SynthesisBlock(256)([x, w_input])
    # 16x16 -> 32x32
    x = SynthesisBlock(128)([x, w_input])
    # 32x32 -> 64x64
    x = SynthesisBlock(64)([x, w_input])
    
    # To RGB
    output = layers.Conv2D(3, 1, activation='tanh')(x)
    
    return Model(w_input, output, name='synthesis_network')

# Full StyleGAN Generator
class StyleGAN(Model):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.mapping = build_mapping_network(latent_dim)
        self.synthesis = build_style_generator(latent_dim)
    
    def call(self, z):
        w = self.mapping(z)
        return self.synthesis(w)
    
    def generate(self, n_samples=1):
        z = tf.random.normal([n_samples, self.latent_dim])
        return self(z)

# Discriminator (similar to DCGAN)
def build_style_discriminator():
    return tf.keras.Sequential([
        layers.Conv2D(64, 4, strides=2, padding='same', input_shape=(64, 64, 3)),
        layers.LeakyReLU(0.2),
        
        layers.Conv2D(128, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        
        layers.Conv2D(256, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        
        layers.Conv2D(512, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        
        layers.Flatten(),
        layers.Dense(1)
    ])

# Training
stylegan = StyleGAN()
discriminator = build_style_discriminator()

g_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.0, beta_2=0.99)
d_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.0, beta_2=0.99)

@tf.function
def train_step(real_images):
    batch_size = tf.shape(real_images)[0]
    z = tf.random.normal([batch_size, 512])
    
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_images = stylegan(z)
        
        real_output = discriminator(real_images)
        fake_output = discriminator(fake_images)
        
        # WGAN-like loss
        d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        g_loss = -tf.reduce_mean(fake_output)
    
    g_grads = g_tape.gradient(g_loss, stylegan.trainable_variables)
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    
    g_optimizer.apply_gradients(zip(g_grads, stylegan.trainable_variables))
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    
    return d_loss, g_loss

# Generate faces
# faces = stylegan.generate(16)
```

---

## Question 10: Code an example of semi-supervised learning with GANs, using a limited number of labeled samples
- [ ] Done

### Pipeline
1. Modify Discriminator: K classes + 1 (fake)
2. Supervised loss on labeled data
3. Unsupervised GAN loss on unlabeled + fake
4. Combine losses for training

### Code
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np

# Generator (standard)
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

# Semi-supervised Discriminator
class SemiSupervisedDiscriminator(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.n_classes = n_classes
        
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # Output: K real classes + 1 fake class
        self.classifier = nn.Linear(256, n_classes + 1)
    
    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        return logits
    
    def get_class_probs(self, x):
        """Get probability for each real class (excluding fake)"""
        logits = self.forward(x)
        # Softmax over real classes only
        real_logits = logits[:, :self.n_classes]
        return F.softmax(real_logits, dim=1)
    
    def get_real_vs_fake(self, x):
        """Get probability that x is real (any class) vs fake"""
        logits = self.forward(x)
        # Real = sum of exp(real class logits)
        # Fake = exp(fake logit)
        real_logits = logits[:, :self.n_classes]
        fake_logit = logits[:, -1:]
        
        # Log-sum-exp for numerical stability
        real_lse = torch.logsumexp(real_logits, dim=1, keepdim=True)
        combined = torch.cat([real_lse, fake_logit], dim=1)
        
        probs = F.softmax(combined, dim=1)
        return probs[:, 0]  # Probability of being real

# Training
class SemiSupervisedGAN:
    def __init__(self, n_classes=10, latent_dim=100, device='cuda'):
        self.device = device
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        
        self.G = Generator(latent_dim).to(device)
        self.D = SemiSupervisedDiscriminator(n_classes).to(device)
        
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=0.0003, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=0.0003, betas=(0.5, 0.999))
    
    def train_step(self, x_labeled, y_labeled, x_unlabeled):
        batch_size = x_unlabeled.size(0)
        
        # Move to device
        x_labeled = x_labeled.to(self.device)
        y_labeled = y_labeled.to(self.device)
        x_unlabeled = x_unlabeled.to(self.device)
        
        # Generate fake images
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        x_fake = self.G(z)
        
        # ============ Train Discriminator ============
        self.d_optimizer.zero_grad()
        
        # 1. Supervised loss on labeled data
        logits_labeled = self.D(x_labeled)
        supervised_loss = F.cross_entropy(logits_labeled[:, :self.n_classes], y_labeled)
        
        # 2. Unsupervised loss on unlabeled (should be classified as real)
        logits_unlabeled = self.D(x_unlabeled)
        # Maximize log(sum(exp(real class logits))) - this means "any real class"
        unsupervised_loss_real = -torch.mean(
            torch.logsumexp(logits_unlabeled[:, :self.n_classes], dim=1)
        )
        
        # 3. Fake detection loss
        logits_fake = self.D(x_fake.detach())
        # Fake class is the last one
        fake_labels = torch.full((batch_size,), self.n_classes, dtype=torch.long, device=self.device)
        unsupervised_loss_fake = F.cross_entropy(logits_fake, fake_labels)
        
        d_loss = supervised_loss + unsupervised_loss_real + unsupervised_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        # ============ Train Generator ============
        self.g_optimizer.zero_grad()
        
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        x_fake = self.G(z)
        logits_fake = self.D(x_fake)
        
        # Generator wants fake to be classified as real (any class)
        g_loss = -torch.mean(torch.logsumexp(logits_fake[:, :self.n_classes], dim=1))
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'supervised_loss': supervised_loss.item()
        }
    
    def evaluate_classifier(self, test_loader):
        """Evaluate classification accuracy"""
        self.D.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                probs = self.D.get_class_probs(x)
                pred = probs.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        return correct / total

# Usage example
from torchvision import datasets, transforms

# Load MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
full_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# Use only 100 labeled samples (semi-supervised)
n_labeled = 100
labeled_indices = np.random.choice(len(full_dataset), n_labeled, replace=False)
unlabeled_indices = np.array([i for i in range(len(full_dataset)) if i not in labeled_indices])

labeled_dataset = Subset(full_dataset, labeled_indices)
unlabeled_dataset = Subset(full_dataset, unlabeled_indices)

labeled_loader = DataLoader(labeled_dataset, batch_size=50, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100)

# Train
sgan = SemiSupervisedGAN(device='cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(100):
    for (x_labeled, y_labeled), (x_unlabeled, _) in zip(labeled_loader, unlabeled_loader):
        losses = sgan.train_step(x_labeled, y_labeled, x_unlabeled)
    
    if epoch % 10 == 0:
        acc = sgan.evaluate_classifier(test_loader)
        print(f"Epoch {epoch}: Accuracy with {n_labeled} labels = {acc:.4f}")
```

---
