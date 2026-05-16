# Neural Networks Interview Questions - Coding Questions

## Question 1: Implement a simple perceptron in Python
- [ ] Done

### Code
```python
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self._activation(linear_output)
                
                # Update rule
                update = self.lr * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update
    
    def _activation(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self._activation(x) for x in linear_output])

# Usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND gate

perceptron = Perceptron()
perceptron.fit(X, y)
print(perceptron.predict(X))  # [0, 0, 0, 1]
```

---

## Question 2: Create a MLP using PyTorch
- [ ] Done

### Code
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)

# Training
model = MLP(input_size=784, hidden_size=128, output_size=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
```

---

## Question 3: Visualize weights of a trained neural network
- [ ] Done

### Code
```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_weights(model, layer_name='conv1'):
    # Get weights from specific layer
    for name, param in model.named_parameters():
        if layer_name in name and 'weight' in name:
            weights = param.detach().cpu().numpy()
            break
    
    # For conv layers: shape is (out_channels, in_channels, H, W)
    n_filters = min(weights.shape[0], 16)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    
    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            # Show first input channel
            ax.imshow(weights[i, 0], cmap='viridis')
            ax.set_title(f'Filter {i}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('weights.png')
    plt.show()

# For FC layer (784 -> 256)
def visualize_fc_weights(model):
    weights = model.fc1.weight.detach().cpu().numpy()
    # Reshape to image if from image input
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        w = weights[i].reshape(28, 28)  # For MNIST
        ax.imshow(w, cmap='seismic')
        ax.axis('off')
    plt.show()
```

---

## Question 4: Implement RNN for text generation
- [ ] Done

### Code
```python
import torch
import torch.nn as nn
import numpy as np

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

# Generate text
def generate(model, seed_text, char2idx, idx2char, length=100):
    model.eval()
    hidden = None
    input_seq = torch.tensor([[char2idx[c] for c in seed_text]])
    generated = seed_text
    
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            probs = torch.softmax(output[0, -1], dim=0)
            next_idx = torch.multinomial(probs, 1).item()
            generated += idx2char[next_idx]
            input_seq = torch.tensor([[next_idx]])
    
    return generated

# Usage: generate(model, "The ", char2idx, idx2char, 100)
```

---

## Question 5: CNN for CIFAR-10 classification
- [ ] Done

### Code
```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                         download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Training
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

## Question 6: Regularization in training loop
- [ ] Done

### Code
```python
import torch
import torch.nn as nn

# L2 Regularization (Weight Decay)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# L1 Regularization (manual)
def l1_regularization(model, lambda_l1=1e-5):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

# Dropout in model
class RegularizedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)

# Training loop with L1
for batch_x, batch_y in dataloader:
    optimizer.zero_grad()
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y) + l1_regularization(model)
    loss.backward()
    optimizer.step()
```

---

## Question 7: Dynamic learning rate adjustment
- [ ] Done

### Code
```python
import torch
import torch.optim as optim

# Method 1: Step decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Method 2: Reduce on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5
)

# Method 3: Cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Method 4: Manual warmup + decay
def adjust_learning_rate(optimizer, epoch, warmup=5, initial_lr=0.001):
    if epoch < warmup:
        lr = initial_lr * (epoch + 1) / warmup
    else:
        lr = initial_lr * (0.1 ** ((epoch - warmup) // 30))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training loop
for epoch in range(100):
    train_one_epoch()
    val_loss = validate()
    scheduler.step(val_loss)  # For ReduceLROnPlateau
    # scheduler.step()  # For others
```

---

## Question 8: Simple GAN for synthetic data
- [ ] Done

### Code
```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Training
latent_dim = 100
G = Generator(latent_dim, 784)
D = Discriminator(784)
criterion = nn.BCELoss()
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)

for epoch in range(100):
    for real_data, _ in dataloader:
        batch_size = real_data.size(0)
        real_data = real_data.view(batch_size, -1)
        
        # Train Discriminator
        d_optimizer.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        real_loss = criterion(D(real_data), real_labels)
        z = torch.randn(batch_size, latent_dim)
        fake_loss = criterion(D(G(z).detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        z = torch.randn(batch_size, latent_dim)
        g_loss = criterion(D(G(z)), real_labels)
        g_loss.backward()
        g_optimizer.step()
```

---

## Question 9: Transfer learning for classification
- [ ] Done

### Code
```python
import torch
import torchvision.models as models
import torch.nn as nn

# Load pretrained model
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Only fc layer will be trained
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Data transforms (use ImageNet normalization)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Training
for epoch in range(10):
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

## Question 10: Autoencoder on MNIST with visualization
- [ ] Done

### Code
```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16)  # Bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Training
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

for epoch in range(10):
    for images, _ in trainloader:
        images = images.view(-1, 784)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

# Visualization
def visualize_reconstruction(model, testloader):
    model.eval()
    images, _ = next(iter(testloader))
    images = images.view(-1, 784)
    
    with torch.no_grad():
        reconstructed = model(images)
    
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        axes[0, i].imshow(images[i].view(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i].view(28, 28), cmap='gray')
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstructed')
    plt.savefig('autoencoder_results.png')
    plt.show()
```

---
