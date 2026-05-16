# PyTorch Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Implement a complete training loop in PyTorch.**

### Answer

### Python Code Example
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)

# Training function
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Acc={accuracy:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model

# Example usage
X_train = torch.randn(1000, 20)
y_train = torch.randint(0, 5, (1000,))
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = NeuralNetwork(20, 64, 5)
trained_model = train_model(model, train_loader, train_loader, epochs=5)
```

---

## Question 2
- [ ] Done

**Build a CNN for image classification.**

### Answer

### Python Code Example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Conv block 1: 32x32 -> 16x16
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2: 16x16 -> 8x8
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3: 8x8 -> 4x4
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Modern CNN with residual connections
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Skip connection
        return F.relu(x)

# Usage
model = CNN(num_classes=10)
x = torch.randn(32, 3, 32, 32)  # Batch of 32, 3 channels, 32x32
output = model(x)
print(f"Output shape: {output.shape}")  # [32, 10]
```

---

## Question 3
- [ ] Done

**Implement LSTM for sequence classification.**

### Answer

### Python Code Example
```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths=None):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))  # (batch, seq_len, embed_dim)
        
        if lengths is not None:
            # Pack for variable length
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        return self.fc(self.dropout(hidden))

# Usage
model = LSTMClassifier(
    vocab_size=10000,
    embedding_dim=128,
    hidden_dim=256,
    output_dim=2,  # Binary classification
    n_layers=2,
    bidirectional=True
)

# Example input
batch_size = 32
seq_len = 50
x = torch.randint(0, 10000, (batch_size, seq_len))
lengths = torch.randint(10, seq_len, (batch_size,))

output = model(x, lengths)
print(f"Output shape: {output.shape}")  # [32, 2]
```

---

## Question 4
- [ ] Done

**Build a Transformer encoder from scratch.**

### Answer

### Python Code Example
```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(context)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        # Global average pooling
        x = x.mean(dim=1)
        return self.classifier(x)

# Usage
model = TransformerEncoder(
    vocab_size=10000, d_model=256, n_heads=8, 
    d_ff=512, n_layers=4, num_classes=10
)
x = torch.randint(0, 10000, (32, 100))
output = model(x)
print(f"Output shape: {output.shape}")  # [32, 10]
```

---

## Question 5
- [ ] Done

**Implement a custom Dataset and DataLoader with data augmentation.**

### Answer

### Python Code Example
```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None, train=True):
        self.image_dir = image_dir
        self.transform = transform
        self.train = train
        
        # Load labels
        self.images = []
        self.labels = []
        with open(labels_file, 'r') as f:
            for line in f:
                img_name, label = line.strip().split(',')
                self.images.append(img_name)
                self.labels.append(int(label))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data augmentation transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Custom collate function
def custom_collate(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=custom_collate,
    drop_last=True
)

# For in-memory data
class TensorDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]
```

---

## Question 6
- [ ] Done

**Build a Variational Autoencoder (VAE).**

### Answer

### Python Code Example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# VAE Loss function
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss

# Training
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_vae(model, train_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x)
            loss = vae_loss(recon_x, batch_x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader.dataset):.4f}')

# Generate new samples
def generate(model, num_samples=10):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, 20)
        samples = model.decode(z)
    return samples.view(num_samples, 1, 28, 28)
```

---

## Question 7
- [ ] Done

**Implement a GAN (Generative Adversarial Network).**

### Answer

### Python Code Example
```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        return img.view(z.size(0), *self.img_shape)

class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
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
def train_gan(generator, discriminator, train_loader, epochs=100, latent_dim=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    discriminator.to(device)
    
    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        for real_imgs, _ in train_loader:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            # Labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(real_imgs), real_labels)
            
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z)
            fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)
            
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            g_loss = criterion(discriminator(fake_imgs), real_labels)
            g_loss.backward()
            optimizer_G.step()
        
        print(f'Epoch {epoch+1}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}')

# Usage
generator = Generator()
discriminator = Discriminator()
train_gan(generator, discriminator, train_loader)
```

---

## Question 8
- [ ] Done

**Implement multi-GPU training with DataParallel and DistributedDataParallel.**

### Answer

### Python Code Example
```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

# Method 1: DataParallel (simple, single machine)
def train_data_parallel():
    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    
    model = model.cuda()
    
    # Training loop
    x = torch.randn(64, 100).cuda()
    output = model(x)  # Automatically splits batch across GPUs
    return output

# Method 2: DistributedDataParallel (recommended for multiple GPUs)
def setup_distributed(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def train_distributed(rank, world_size):
    setup_distributed(rank, world_size)
    
    # Create model and move to GPU
    torch.cuda.set_device(rank)
    model = nn.Linear(100, 10).cuda(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    
    # Distributed sampler
    dataset = torch.utils.data.TensorDataset(
        torch.randn(1000, 100),
        torch.randint(0, 10, (1000,))
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=32)
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):
        sampler.set_epoch(epoch)  # Important for shuffling
        for x, y in loader:
            x, y = x.cuda(rank), y.cuda(rank)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

# Launch distributed training
# python -m torch.distributed.launch --nproc_per_node=4 train.py

# Simplified with torchrun (PyTorch 1.9+)
# torchrun --nproc_per_node=4 train.py
```

---

## Question 9
- [ ] Done

**Create a custom loss function with gradient computation.**

### Answer

### Python Code Example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Method 1: Simple function
def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

# Method 2: nn.Module class
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        # label: 1 if similar, 0 if dissimilar
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = label * torch.pow(euclidean_distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()

# Dice Loss for segmentation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

# Combined loss
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        return self.ce_weight * self.ce(pred, target) + \
               self.dice_weight * self.dice(pred.argmax(1), target)
```

---

## Question 10
- [ ] Done

**Implement model inference optimization with TorchScript and ONNX export.**

### Answer

### Python Code Example
```python
import torch
import torch.nn as nn
import time

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleModel()
model.eval()

# Method 1: TorchScript - Tracing
example_input = torch.randn(1, 100)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('traced_model.pt')

# Load traced model
loaded_traced = torch.jit.load('traced_model.pt')

# Method 2: TorchScript - Scripting (for control flow)
class ModelWithControlFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)
    
    def forward(self, x, use_relu=True):
        x = self.fc(x)
        if use_relu:
            x = torch.relu(x)
        return x

scripted_model = torch.jit.script(ModelWithControlFlow())
scripted_model.save('scripted_model.pt')

# Method 3: ONNX Export
torch.onnx.export(
    model,
    example_input,
    'model.onnx',
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# Load ONNX with ONNX Runtime
import onnxruntime as ort

ort_session = ort.InferenceSession('model.onnx')
outputs = ort_session.run(None, {'input': example_input.numpy()})

# Benchmark comparison
def benchmark(model, input_tensor, num_runs=100):
    # Warmup
    for _ in range(10):
        _ = model(input_tensor)
    
    start = time.time()
    for _ in range(num_runs):
        _ = model(input_tensor)
    end = time.time()
    return (end - start) / num_runs * 1000  # ms

x = torch.randn(32, 100)
print(f"PyTorch: {benchmark(model, x):.2f} ms")
print(f"TorchScript: {benchmark(traced_model, x):.2f} ms")

# Quantization for faster inference
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
print(f"Quantized: {benchmark(quantized_model, x):.2f} ms")
```

---

## Question 11
- [ ] Done

**Code a Python script that demonstrates tensor operations, such as slicing, indexing, concatenating , and transposing , using PyTorch**

**Answer:**

```python
import torch

# === Creating Tensors ===
a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = torch.randn(3, 3)
print(f"a shape: {a.shape}, dtype: {a.dtype}")

# === Indexing ===
print(a[0])         # First row: tensor([1, 2, 3])
print(a[0, 2])      # Element at (0,2): 3
print(a[:, 1])      # Second column: tensor([2, 5, 8])
print(a[-1])        # Last row: tensor([7, 8, 9])

# === Slicing ===
print(a[0:2, 1:3])  # Rows 0-1, Cols 1-2: tensor([[2,3],[5,6]])
print(a[::2])       # Every other row: tensor([[1,2,3],[7,8,9]])

# === Boolean Indexing ===
mask = a > 4
print(a[mask])      # tensor([5, 6, 7, 8, 9])

# === Fancy Indexing ===
idx = torch.tensor([0, 2])
print(a[idx])       # Rows 0 and 2

# === Concatenation ===
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])

cat_row = torch.cat([x, y], dim=0)   # Vertical stack: (4, 2)
cat_col = torch.cat([x, y], dim=1)   # Horizontal stack: (2, 4)
stacked = torch.stack([x, y], dim=0) # New dimension: (2, 2, 2)
print(f"cat_row: {cat_row.shape}, cat_col: {cat_col.shape}, stacked: {stacked.shape}")

# === Transposing ===
t = torch.randn(2, 3, 4)
print(t.T.shape)                     # Full transpose (reverses all dims)
print(t.transpose(0, 2).shape)       # Swap dim 0 and 2: (4, 3, 2)
print(t.permute(2, 0, 1).shape)      # Custom order: (4, 2, 3)

# === Reshaping ===
r = torch.arange(12)
print(r.view(3, 4))                  # Reshape to (3, 4)
print(r.view(3, 4).contiguous())     # Ensure contiguous memory
print(r.reshape(2, 6))               # Like view but works on non-contiguous

# === Squeeze/Unsqueeze ===
s = torch.randn(1, 3, 1, 4)
print(s.squeeze().shape)             # Remove all size-1 dims: (3, 4)
print(s.squeeze(0).shape)            # Remove dim 0 only: (3, 1, 4)
u = torch.randn(3, 4)
print(u.unsqueeze(0).shape)          # Add dim at 0: (1, 3, 4)

# === Element-wise Operations ===
print(a + 10)         # Add scalar
print(a * b)          # Element-wise multiply
print(torch.matmul(a.float(), b))  # Matrix multiply
```

| Operation | Method | Example |
|-----------|--------|---------|
| Concatenate | `torch.cat([a,b], dim)` | Stack along existing dim |
| Stack | `torch.stack([a,b], dim)` | Create new dim |
| Transpose | `.transpose(d1, d2)` | Swap two dims |
| Permute | `.permute(d0, d1, ...)` | Reorder all dims |
| Reshape | `.view()` / `.reshape()` | Change shape |
| Squeeze | `.squeeze()` | Remove size-1 dims |

> **Interview Tip:** `.view()` requires contiguous memory; `.reshape()` works on any tensor. Use `.permute()` for multi-dimensional transposes, `.transpose()` for swapping exactly two dimensions.

---

## Question 12
- [ ] Done

**Create a simple feedforward neural network in PyTorch that works on the MNIST dataset.**

**Answer:**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# === 1. Data ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# === 2. Model ===
class FeedForwardNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.network(x)

# === 3. Setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FeedForwardNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === 4. Training ===
for epoch in range(10):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Evaluate
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
    
    acc = correct / len(test_data)
    print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.4f}")
```

> **Interview Tip:** PyTorch requires manual `model.train()` / `model.eval()` toggling (affects Dropout/BatchNorm). Always use `torch.no_grad()` during evaluation to save memory and speed up inference.

---

## Question 13
- [ ] Done

**Write a PyTorch function to manually compute the gradients for a basic linear regression model.**

**Answer:**

```python
import torch
import numpy as np

# === Manual Gradient Computation (No Autograd) ===
np.random.seed(42)
X = np.random.randn(100, 1).astype(np.float32)
y = 3 * X + 2 + np.random.randn(100, 1).astype(np.float32) * 0.1

# Initialize parameters
w = np.random.randn(1, 1).astype(np.float32)
b = np.zeros(1).astype(np.float32)
lr = 0.01

def forward(X, w, b):
    return X @ w + b

def compute_gradients(X, y, y_pred):
    n = len(X)
    error = y_pred - y
    dw = (2 / n) * (X.T @ error)   # dL/dw = (2/n) * X^T * (y_pred - y)
    db = (2 / n) * error.sum()      # dL/db = (2/n) * sum(y_pred - y)
    return dw, db

# Training loop
for epoch in range(100):
    y_pred = forward(X, w, b)
    loss = np.mean((y_pred - y) ** 2)  # MSE
    dw, db = compute_gradients(X, y, y_pred)
    w -= lr * dw
    b -= lr * db
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss={loss:.4f}, w={w[0,0]:.4f}, b={b[0]:.4f}")

print(f"Learned: w={w[0,0]:.4f} (true=3), b={b[0]:.4f} (true=2)")

# === Same with PyTorch Autograd (Comparison) ===
X_t = torch.tensor(X)
y_t = torch.tensor(y)
w_t = torch.randn(1, 1, requires_grad=True)
b_t = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([w_t, b_t], lr=0.01)

for epoch in range(100):
    y_pred = X_t @ w_t + b_t
    loss = torch.mean((y_pred - y_t) ** 2)
    
    loss.backward()           # Autograd computes gradients
    optimizer.step()          # Update parameters
    optimizer.zero_grad()     # Reset gradients

print(f"Autograd: w={w_t.item():.4f}, b={b_t.item():.4f}")
```

| Gradient | Formula | Meaning |
|----------|---------|--------|
| $\frac{\partial L}{\partial w}$ | $\frac{2}{n} X^T (\hat{y} - y)$ | How loss changes with weight |
| $\frac{\partial L}{\partial b}$ | $\frac{2}{n} \sum(\hat{y} - y)$ | How loss changes with bias |

> **Interview Tip:** Understanding manual gradient computation shows deep knowledge. In practice, always use `autograd`. The key insight: `loss.backward()` computes all gradients via chain rule, which replaces manual computation.

---

## Question 14
- [ ] Done

**Write a Python script using PyTorch that saves and loads a trained model.**

**Answer:**

```python
import torch
import torch.nn as nn

# Sample model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === Method 1: Save/Load State Dict (Recommended) ===
torch.save(model.state_dict(), 'model_weights.pth')

# Load
loaded_model = SimpleModel()  # Must create same architecture
loaded_model.load_state_dict(torch.load('model_weights.pth'))
loaded_model.eval()

# === Method 2: Save Entire Model (Not Recommended) ===
torch.save(model, 'full_model.pth')
loaded = torch.load('full_model.pth')

# === Method 3: Save Checkpoint (Best Practice) ===
checkpoint = {
    'epoch': 50,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0.05,
    'best_accuracy': 0.95
}
torch.save(checkpoint, 'checkpoint.pth')

# Resume training from checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
print(f"Resumed from epoch {start_epoch}, loss: {checkpoint['loss']}")

# === Method 4: Save for Inference (TorchScript) ===
scripted = torch.jit.script(model)
scripted.save('model_scripted.pt')

loaded_scripted = torch.jit.load('model_scripted.pt')
output = loaded_scripted(torch.randn(1, 10))

# === Method 5: ONNX Export ===
dummy_input = torch.randn(1, 10)
torch.onnx.export(model, dummy_input, 'model.onnx',
                  input_names=['input'], output_names=['output'])
```

| Method | Saves | Use Case | Portable |
|--------|-------|----------|----------|
| `state_dict` | Weights only | Resume training / inference | Code needed |
| `torch.save(model)` | Everything (pickle) | Quick save | Fragile |
| Checkpoint | Weights + optimizer + metadata | Resume training | Code needed |
| TorchScript | Compiled model | Production deployment | Yes |
| ONNX | Cross-framework model | Multi-framework deployment | Yes |

> **Interview Tip:** Always use `state_dict` over saving the entire model. Full model saving uses pickle which is fragile and tied to the exact class structure. For production, export to TorchScript or ONNX.

---

## Question 15
- [ ] Done

**Implement a PyTorch DataLoader for a given CSV dataset**

### Solution

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class CSVDataset(Dataset):
    def __init__(self, csv_path, target_col='target', transform=None):
        self.df = pd.read_csv(csv_path)
        self.features = self.df.drop(columns=[target_col]).values.astype(np.float32)
        self.labels = self.df[target_col].values.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.labels[idx])
        if self.transform:
            x = self.transform(x)
        return x, y

# Create DataLoaders
train_dataset = CSVDataset('train.csv', target_col='target')
test_dataset = CSVDataset('test.csv', target_col='target')

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,           # Shuffle training data
    num_workers=4,          # Parallel data loading
    pin_memory=True,        # Faster GPU transfer
    drop_last=True          # Drop incomplete last batch
)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training loop
for epoch in range(10):
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to('cuda')
        batch_y = batch_y.to('cuda')
        # Forward pass, loss, backward pass...
```

> **Interview Tip:** Always implement `__len__` and `__getitem__`. Use `pin_memory=True` for GPU training and `num_workers > 0` for parallel data loading. `shuffle=True` only for training, not validation/test.

---

## Question 16
- [ ] Done

**Use PyTorch to implement a convolutional neural network (CNN) for image classification**

### Solution

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data pipeline
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_ds = datasets.CIFAR10('.', train=True, download=True, transform=transform_train)
test_ds = datasets.CIFAR10('.', train=False, transform=transform_test)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

# CNN Model
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    print(f"Epoch {epoch+1}: Acc={100.*correct/total:.2f}%")
```

> **Interview Tip:** Always call `model.train()` and `model.eval()` to toggle dropout/batch norm. Use `torch.no_grad()` during evaluation to save memory. `optimizer.zero_grad()` must be called before each backward pass.

---
