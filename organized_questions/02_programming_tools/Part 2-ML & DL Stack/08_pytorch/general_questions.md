# PyTorch Interview Questions - General Questions

## Question 1
- [ ] Done

**How do you save and load models in PyTorch?**

### Answer

### Two Approaches

| Method | Saves | Size | Flexibility |
|--------|-------|------|-------------|
| `state_dict` | Weights only | Smaller | ✅ Recommended |
| Full model | Weights + architecture | Larger | Less portable |

### Python Code Example
```python
import torch
import torch.nn as nn

# Define model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

model = MyModel()

# Method 1: Save state_dict (RECOMMENDED)
torch.save(model.state_dict(), 'model_weights.pth')

# Load state_dict
loaded_model = MyModel()  # Must define architecture first
loaded_model.load_state_dict(torch.load('model_weights.pth'))
loaded_model.eval()

# Method 2: Save entire model
torch.save(model, 'full_model.pth')
loaded_full = torch.load('full_model.pth')

# Save checkpoint (for resuming training)
checkpoint = {
    'epoch': 10,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0.5,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

---

## Question 2
- [ ] Done

**What is DataLoader and how do you use it?**

### Answer

**Definition**: DataLoader wraps a Dataset and provides batching, shuffling, and parallel data loading.

### Key Parameters

| Parameter | Purpose |
|-----------|---------|
| `batch_size` | Samples per batch |
| `shuffle` | Randomize order |
| `num_workers` | Parallel loading processes |
| `drop_last` | Drop incomplete last batch |

### Python Code Example
```python
from torch.utils.data import Dataset, DataLoader
import torch

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataset
import numpy as np
X = np.random.randn(1000, 10)
y = np.random.randn(1000, 1)
dataset = CustomDataset(X, y)

# Create DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Parallel loading
    pin_memory=True  # Faster GPU transfer
)

# Training loop
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

---

## Question 3
- [ ] Done

**How do you handle overfitting in PyTorch?**

### Answer

### Techniques

| Technique | Implementation |
|-----------|----------------|
| Dropout | `nn.Dropout(p)` |
| Weight decay | `optimizer(weight_decay=0.01)` |
| Data augmentation | `torchvision.transforms` |
| Early stopping | Manual implementation |
| Batch normalization | `nn.BatchNorm1d/2d` |

### Python Code Example
```python
import torch
import torch.nn as nn

# 1. Dropout
class ModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Only active in training
        return self.fc2(x)

# 2. Weight decay (L2 regularization)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# 3. Early stopping
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            self.counter += 1
        return self.counter >= self.patience

early_stopping = EarlyStopping(patience=5)

# 4. Data augmentation
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
])
```

---

## Question 4
- [ ] Done

**What is transfer learning and how to implement it in PyTorch?**

### Answer

**Definition**: Transfer learning uses a pre-trained model as a starting point, then fine-tunes it for a new task.

### Approaches

| Approach | Description |
|----------|-------------|
| Feature extraction | Freeze base, train new head |
| Fine-tuning | Train entire model with low LR |
| Gradual unfreezing | Progressively unfreeze layers |

### Python Code Example
```python
import torch
import torch.nn as nn
from torchvision import models

# Load pre-trained ResNet
model = models.resnet18(pretrained=True)

# Option 1: Feature extraction (freeze base)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Only new layer will be trained
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Option 2: Fine-tuning (lower learning rate)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Different learning rates
optimizer = torch.optim.Adam([
    {'params': model.conv1.parameters(), 'lr': 1e-5},
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-4},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-3},
    {'params': model.fc.parameters(), 'lr': 1e-2},
])

# Option 3: Gradual unfreezing
def unfreeze_layer(model, layer_name):
    for name, param in model.named_parameters():
        if layer_name in name:
            param.requires_grad = True
```

---

## Question 5
- [ ] Done

**Explain the concept of gradient clipping in PyTorch.**

### Answer

**Definition**: Gradient clipping limits gradient values to prevent exploding gradients, especially important in RNNs.

### Types

| Type | Function | Description |
|------|----------|-------------|
| Norm clipping | `clip_grad_norm_` | Scales if norm exceeds threshold |
| Value clipping | `clip_grad_value_` | Clips each value |

### Python Code Example
```python
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

model = nn.LSTM(input_size=10, hidden_size=64, num_layers=2)
optimizer = torch.optim.Adam(model.parameters())

# Training step with gradient clipping
def train_step(model, x, y, max_norm=1.0):
    optimizer.zero_grad()
    
    output, _ = model(x)
    loss = nn.functional.mse_loss(output, y)
    loss.backward()
    
    # Option 1: Clip by norm (RECOMMENDED)
    total_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
    print(f"Gradient norm: {total_norm}")
    
    # Option 2: Clip by value
    # clip_grad_value_(model.parameters(), clip_value=0.5)
    
    optimizer.step()
    return loss.item()

# Monitor gradient norms
def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5
```

---

## Question 6
- [ ] Done

**How do you implement custom layers in PyTorch?**

### Answer

### Steps
1. Inherit from `nn.Module`
2. Define `__init__` with parameters
3. Implement `forward` method

### Python Code Example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom Linear layer
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        output = x @ self.weight.T
        if self.bias is not None:
            output += self.bias
        return output

# Custom activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Attention layer
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, x):
        # x: (seq_len, batch, embed_dim)
        attn_output, attn_weights = self.attention(x, x, x)
        return attn_output

# Using custom layers
model = nn.Sequential(
    CustomLinear(10, 64),
    Swish(),
    CustomLinear(64, 1)
)
```

---

## Question 7
- [ ] Done

**What is torch.no_grad() and when do you use it?**

### Answer

**Definition**: `torch.no_grad()` disables gradient computation, reducing memory usage and speeding up computation.

### Use Cases

| Use Case | Reason |
|----------|--------|
| Inference | Don't need gradients |
| Evaluation | Save memory |
| Manual updates | Prevent tracking |
| Computing metrics | Performance |

### Python Code Example
```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
x = torch.randn(32, 10, requires_grad=True)

# With gradients (training)
output_train = model(x)
print(f"Requires grad: {output_train.requires_grad}")  # True

# Without gradients (inference)
with torch.no_grad():
    output_eval = model(x)
    print(f"Requires grad: {output_eval.requires_grad}")  # False

# Alternative: decorator
@torch.no_grad()
def predict(model, x):
    return model(x)

# Set inference mode (even faster than no_grad)
with torch.inference_mode():
    output = model(x)

# Manual parameter update without tracking
with torch.no_grad():
    for param in model.parameters():
        param -= 0.01 * param.grad  # Manual SGD step

# Validation loop
def validate(model, val_loader):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            output = model(x)
            loss = nn.functional.mse_loss(output, y)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)
```

---

## Question 8
- [ ] Done

**How do you debug PyTorch models?**

### Answer

### Debugging Techniques

| Technique | Purpose |
|-----------|---------|
| Print shapes | Check tensor dimensions |
| `torchinfo` | Model summary |
| Hooks | Inspect activations |
| Anomaly detection | Find NaN sources |

### Python Code Example
```python
import torch
import torch.nn as nn

# 1. Print shapes at each layer
class DebugModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        print(f"Input: {x.shape}")
        x = self.fc1(x)
        print(f"After fc1: {x.shape}")
        x = torch.relu(x)
        x = self.fc2(x)
        print(f"Output: {x.shape}")
        return x

# 2. Model summary
from torchinfo import summary
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
summary(model, input_size=(32, 10))

# 3. Register hooks to inspect activations
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model.fc1.register_forward_hook(get_activation('fc1'))

# 4. Anomaly detection
torch.autograd.set_detect_anomaly(True)
# Will show where NaN/Inf originated

# 5. Check for NaN/Inf
def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")

# 6. Gradient checking
x = torch.randn(5, 10, requires_grad=True)
output = model(x)
loss = output.sum()
loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean={param.grad.mean():.6f}")
```

---

## Question 9
- [ ] Done

**What are hooks in PyTorch?**

### Answer

**Definition**: Hooks are functions that get called during forward/backward pass, useful for debugging and feature extraction.

### Types of Hooks

| Hook Type | When Called |
|-----------|-------------|
| Forward hook | After forward pass |
| Backward hook | After backward pass |
| Pre-forward hook | Before forward pass |

### Python Code Example
```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# Store activations
activations = {}
gradients = {}

# Forward hook
def forward_hook(module, input, output):
    activations[module] = output

# Backward hook
def backward_hook(module, grad_input, grad_output):
    gradients[module] = grad_output[0]

# Register hooks
hooks = []
for layer in model:
    hooks.append(layer.register_forward_hook(forward_hook))
    hooks.append(layer.register_full_backward_hook(backward_hook))

# Forward and backward pass
x = torch.randn(32, 10)
output = model(x)
loss = output.sum()
loss.backward()

# Inspect activations
for layer, activation in activations.items():
    print(f"{layer}: {activation.shape}")

# Remove hooks when done
for hook in hooks:
    hook.remove()

# Feature extraction with hooks
features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

# Get intermediate features from pretrained model
from torchvision import models
resnet = models.resnet18(pretrained=True)
resnet.layer3.register_forward_hook(get_features('layer3'))
```

---

## Question 10
- [ ] Done

**How do you handle variable-length sequences in PyTorch?**

### Answer

### Techniques

| Technique | Description |
|-----------|-------------|
| Padding | Pad to max length |
| `pack_padded_sequence` | Efficient RNN processing |
| Attention masking | Ignore padding in attention |

### Python Code Example
```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# Variable length sequences
seq1 = torch.tensor([1, 2, 3])
seq2 = torch.tensor([4, 5])
seq3 = torch.tensor([6, 7, 8, 9])

# Pad sequences
padded = pad_sequence([seq1, seq2, seq3], batch_first=True, padding_value=0)
print(padded)
# tensor([[1, 2, 3, 0],
#         [4, 5, 0, 0],
#         [6, 7, 8, 9]])

lengths = torch.tensor([3, 2, 4])

# Pack for efficient RNN processing
packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)

# LSTM with packed sequences
lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
packed_output, (h_n, c_n) = lstm(packed.float().unsqueeze(-1))

# Unpack back to padded
output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)

# Create padding mask for attention
def create_padding_mask(lengths, max_len):
    batch_size = len(lengths)
    mask = torch.arange(max_len).expand(batch_size, max_len) >= lengths.unsqueeze(1)
    return mask  # True where padded

mask = create_padding_mask(lengths, max_len=4)

# Collate function for DataLoader
def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    padded = pad_sequence(sequences, batch_first=True)
    return padded, lengths, torch.tensor(labels)
```
