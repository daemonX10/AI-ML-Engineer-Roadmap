# Deep Learning Interview Questions - Coding Questions

## Question 1: Implement a simple neural network from scratch
- [ ] Done

### Code
```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.relu(z)
            self.activations.append(a)
        
        # Output layer with softmax
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = self.softmax(z)
        self.activations.append(output)
        
        return output
    
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        
        # Output layer gradient
        dz = self.activations[-1] - y
        
        for i in range(len(self.weights) - 1, -1, -1):
            dw = np.dot(self.activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
            
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
    
    def train(self, X, y, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)
            
            if epoch % 10 == 0:
                loss = -np.mean(np.sum(y * np.log(output + 1e-8), axis=1))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Usage
nn = NeuralNetwork([784, 128, 64, 10])
# nn.train(X_train, y_train_onehot, epochs=100)
```

---

## Question 2: CNN in TensorFlow for MNIST
- [ ] Done

### Code
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Build CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=5, validation_split=0.1, batch_size=64)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

---

## Question 3: Real-time data augmentation with Keras
- [ ] Done

### Code
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_augmentation_generator():
    """Create ImageDataGenerator with various augmentations"""
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=1./255
    )
    return datagen

# Usage with training
datagen = create_augmentation_generator()

# Flow from directory
train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Or flow from array
# datagen.fit(x_train)  # Compute statistics
# model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=10)

# Train with generator
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10
)

# Visualize augmentations
def visualize_augmentations(image, datagen, num_samples=6):
    import matplotlib.pyplot as plt
    
    image = image.reshape((1,) + image.shape)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    
    for ax, batch in zip(axes.flat, datagen.flow(image, batch_size=1)):
        ax.imshow(batch[0])
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
```

---

## Question 4: RNN with LSTM for text generation
- [ ] Done

### Code
```python
import torch
import torch.nn as nn
import numpy as np

class TextGeneratorLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# Prepare data
def prepare_data(text, seq_length):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    
    encoded = [char2idx[c] for c in text]
    
    X, y = [], []
    for i in range(len(encoded) - seq_length):
        X.append(encoded[i:i+seq_length])
        y.append(encoded[i+1:i+seq_length+1])
    
    return np.array(X), np.array(y), char2idx, idx2char

# Training
model = TextGeneratorLSTM(vocab_size=65, embed_dim=128, hidden_dim=256, num_layers=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        output, _ = model(batch_x)
        loss = criterion(output.view(-1, vocab_size), batch_y.view(-1))
        loss.backward()
        optimizer.step()

# Generate text
def generate_text(model, seed, char2idx, idx2char, length=200, temperature=0.8):
    model.eval()
    hidden = None
    input_seq = torch.tensor([[char2idx[c] for c in seed]])
    generated = seed
    
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            probs = torch.softmax(output[0, -1] / temperature, dim=0)
            next_idx = torch.multinomial(probs, 1).item()
            generated += idx2char[next_idx]
            input_seq = torch.tensor([[next_idx]])
    
    return generated
```

---

## Question 5: GAN in PyTorch
- [ ] Done

### Code
```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.model(img)

# Training
latent_dim = 100
G = Generator(latent_dim)
D = Discriminator()
criterion = nn.BCELoss()
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(100):
    for real_imgs, _ in dataloader:
        batch_size = real_imgs.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # Train Discriminator
        d_optimizer.zero_grad()
        real_loss = criterion(D(real_imgs), real_labels)
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = G(z)
        fake_loss = criterion(D(fake_imgs.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = G(z)
        g_loss = criterion(D(fake_imgs), real_labels)
        g_loss.backward()
        g_optimizer.step()
```

---

## Question 6: Autoencoder for dimensionality reduction
- [ ] Done

### Code
```python
import tensorflow as tf
from tensorflow.keras import layers, models

class Autoencoder(models.Model):
    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        
        # Encoder
        self.encoder = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(encoding_dim, activation='relu')  # Bottleneck
        ])
        
        # Decoder
        self.decoder = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(encoding_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid')
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

# Usage
autoencoder = Autoencoder(input_dim=784, encoding_dim=32)
autoencoder.compile(optimizer='adam', loss='mse')

# Flatten and normalize data
x_train_flat = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test_flat = x_test.reshape(-1, 784).astype('float32') / 255.0

# Train
autoencoder.fit(x_train_flat, x_train_flat, 
                epochs=50, batch_size=256, 
                validation_data=(x_test_flat, x_test_flat))

# Get low-dimensional representations
encoded_data = autoencoder.encode(x_test_flat)
print(f"Original shape: {x_test_flat.shape}")
print(f"Encoded shape: {encoded_data.shape}")
```

---

## Question 7: Seq2Seq chatbot
- [ ] Done

### Code
```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        prediction = self.fc(output)
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)
        hidden = self.encoder(src)
        
        decoder_input = trg[:, 0:1]  # <SOS> token
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t] = output.squeeze(1)
            
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            decoder_input = trg[:, t:t+1] if use_teacher else output.argmax(2)
        
        return outputs

# Usage
encoder = Encoder(vocab_size=10000, embed_dim=256, hidden_dim=512)
decoder = Decoder(vocab_size=10000, embed_dim=256, hidden_dim=512)
model = Seq2Seq(encoder, decoder, device='cuda')
```

---

## Question 8: ResNet with transfer learning in Keras
- [ ] Done

### Code
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

# Load pretrained ResNet50
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Build complete model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 classes
])

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    validation_split=0.2
)

# Train
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

model.fit(train_generator, epochs=10)

# Fine-tune (unfreeze some layers)
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_generator, epochs=5)
```

---

## Question 9: Transformer for language translation
- [ ] Done

### Code
```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerTranslator(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, nhead=8, 
                 num_layers=6, dim_feedforward=2048):
        super().__init__()
        
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(d_model, tgt_vocab)
        self.d_model = d_model
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        tgt = self.pos_encoder(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.fc_out(output)
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

# Usage
model = TransformerTranslator(src_vocab=10000, tgt_vocab=10000)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Padding idx
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
```

---

## Question 10: Anomaly detection with autoencoder
- [ ] Done

### Code
```python
import torch
import torch.nn as nn
import numpy as np

class AnomalyDetector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Train on normal data only
model = AnomalyDetector(input_dim=30)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for batch in normal_data_loader:
        optimizer.zero_grad()
        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()

# Detection function
def detect_anomalies(model, data, threshold=None):
    model.eval()
    with torch.no_grad():
        reconstructed = model(data)
        mse = torch.mean((data - reconstructed) ** 2, dim=1)
    
    if threshold is None:
        # Use mean + 2*std as threshold
        threshold = mse.mean() + 2 * mse.std()
    
    anomalies = mse > threshold
    return anomalies, mse

# Usage
anomalies, scores = detect_anomalies(model, test_data)
print(f"Detected {anomalies.sum()} anomalies")

# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(scores.numpy())
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.xlabel('Sample')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.show()
```

---
