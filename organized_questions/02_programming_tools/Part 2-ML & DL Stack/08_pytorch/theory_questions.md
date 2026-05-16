# PyTorch Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What is PyTorch and what are its main features?**

### Answer

**Definition**: PyTorch is an open-source deep learning framework developed by **Meta AI (Facebook)**. It's known for its **dynamic computation graphs** and **Pythonic** design.

### Main Features

| Feature | Description |
|---------|-------------|
| **Dynamic Graphs** | Build graphs on-the-fly (eager execution) |
| **Autograd** | Automatic differentiation |
| **GPU Acceleration** | CUDA support |
| **TorchScript** | Production deployment |
| **Pythonic** | Feels like native Python |
| **Research-friendly** | Flexible for experimentation |

### Python Code Example
```python
import torch
import torch.nn as nn

# Check PyTorch version and GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Basic tensor operations
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
z = x + y
print(f"x + y = {z}")

# Simple neural network
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
```

---

## Question 2
- [ ] Done

**What is the difference between PyTorch and TensorFlow?**

### Answer

### Comparison Table

| Aspect | PyTorch | TensorFlow |
|--------|---------|------------|
| **Execution** | Eager (dynamic) | Graph-based (eager in TF2) |
| **Graph** | Define-by-run | Define-and-run |
| **Debugging** | Easy (standard Python) | More complex |
| **API Style** | Pythonic, explicit | Keras (high-level) |
| **Deployment** | TorchServe, ONNX | TF Serving, TFLite |
| **Community** | Research-focused | Production-focused |

### Python Code Example
```python
# PyTorch: Explicit and Pythonic
import torch
import torch.nn as nn

class PyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Standard Python debugging works
model = PyTorchModel()
x = torch.randn(32, 10)
output = model(x)  # Can set breakpoints here
```

### When to Use
- **PyTorch**: Research, prototyping, flexibility needed
- **TensorFlow**: Production, mobile deployment, enterprise

---

## Question 3
- [ ] Done

**Explain the concept of tensors in PyTorch.**

### Answer

**Definition**: A tensor is a multi-dimensional array, similar to NumPy arrays, but with GPU support and automatic differentiation.

### Tensor Properties

| Property | Description |
|----------|-------------|
| **shape** | Dimensions of tensor |
| **dtype** | Data type (float32, int64, etc.) |
| **device** | CPU or CUDA |
| **requires_grad** | Track gradients |

### Python Code Example
```python
import torch

# Creating tensors
scalar = torch.tensor(5)           # 0D
vector = torch.tensor([1, 2, 3])   # 1D
matrix = torch.tensor([[1, 2], [3, 4]])  # 2D

# Tensor properties
print(f"Shape: {matrix.shape}")      # torch.Size([2, 2])
print(f"dtype: {matrix.dtype}")      # torch.int64
print(f"device: {matrix.device}")    # cpu

# Create with specific properties
x = torch.zeros(3, 4, dtype=torch.float32)
y = torch.ones(3, 4, device='cuda' if torch.cuda.is_available() else 'cpu')

# Enable gradient tracking
z = torch.randn(3, 3, requires_grad=True)

# From NumPy
import numpy as np
numpy_array = np.array([1, 2, 3])
tensor_from_numpy = torch.from_numpy(numpy_array)

# To NumPy
back_to_numpy = tensor_from_numpy.numpy()
```

---

## Question 4
- [ ] Done

**What is Autograd in PyTorch?**

### Answer

**Definition**: Autograd is PyTorch's automatic differentiation engine that computes gradients for backpropagation.

### How It Works

| Step | Description |
|------|-------------|
| 1. Forward | Operations recorded in computation graph |
| 2. Backward | `loss.backward()` computes gradients |
| 3. Update | Optimizer updates parameters |

### Python Code Example
```python
import torch

# Simple gradient computation
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2  # y = x^2

y.backward()  # Compute dy/dx
print(f"dy/dx at x=3: {x.grad}")  # Output: 6.0

# In neural networks
model = torch.nn.Linear(10, 1)
x = torch.randn(32, 10)
y_true = torch.randn(32, 1)

# Forward pass
y_pred = model(x)
loss = torch.nn.functional.mse_loss(y_pred, y_true)

# Backward pass - compute gradients
loss.backward()

# Gradients stored in .grad
for name, param in model.named_parameters():
    print(f"{name}: grad shape = {param.grad.shape}")

# Gradient context managers
with torch.no_grad():  # Disable gradient tracking
    inference_output = model(x)

# Detach from graph
detached = y_pred.detach()
```

---

## Question 5
- [ ] Done

**What is the difference between torch.Tensor and torch.tensor?**

### Answer

### Comparison

| Aspect | `torch.Tensor` | `torch.tensor` |
|--------|----------------|----------------|
| **Type** | Class constructor | Factory function |
| **dtype** | Default float32 | Infers from data |
| **Copy** | May share memory | Always copies |
| **Recommended** | Legacy | ✅ Preferred |

### Python Code Example
```python
import torch

# torch.Tensor - Class constructor (legacy)
a = torch.Tensor([1, 2, 3])
print(f"torch.Tensor dtype: {a.dtype}")  # float32 (always)

# torch.tensor - Factory function (recommended)
b = torch.tensor([1, 2, 3])
print(f"torch.tensor dtype: {b.dtype}")  # int64 (inferred)

c = torch.tensor([1.0, 2.0, 3.0])
print(f"torch.tensor float dtype: {c.dtype}")  # float32

# Specify dtype explicitly
d = torch.tensor([1, 2, 3], dtype=torch.float32)

# Other factory functions
zeros = torch.zeros(3, 3)
ones = torch.ones(3, 3)
randn = torch.randn(3, 3)  # Normal distribution
rand = torch.rand(3, 3)    # Uniform [0, 1)
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
```

---

## Question 6
- [ ] Done

**Explain torch.nn.Module and how to create custom modules.**

### Answer

**Definition**: `nn.Module` is the base class for all neural network modules in PyTorch. All layers and models inherit from it.

### Key Methods

| Method | Purpose |
|--------|---------|
| `__init__` | Define layers |
| `forward` | Define forward pass |
| `parameters()` | Get trainable parameters |
| `to(device)` | Move to GPU/CPU |

### Python Code Example
```python
import torch
import torch.nn as nn

class CustomNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # Initialize parent class
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create model
model = CustomNetwork(input_size=10, hidden_size=64, output_size=2)

# View parameters
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Forward pass
x = torch.randn(32, 10).to(device)
output = model(x)
print(f"Output shape: {output.shape}")  # [32, 2]
```

---

## Question 7
- [ ] Done

**What are the different optimizers available in PyTorch?**

### Answer

### Common Optimizers

| Optimizer | Use Case |
|-----------|----------|
| **SGD** | Simple, good with momentum |
| **Adam** | Most common default |
| **AdamW** | Adam with weight decay |
| **RMSprop** | Good for RNNs |
| **Adagrad** | Sparse gradients |

### Python Code Example
```python
import torch
import torch.optim as optim

model = torch.nn.Linear(10, 1)

# SGD with momentum
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (most popular)
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)

# AdamW (Adam with decoupled weight decay)
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Training step
def train_step(model, optimizer, x, y_true):
    optimizer.zero_grad()  # Clear gradients
    
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y_true)
    
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights
    
    return loss.item()

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer_adam, step_size=10, gamma=0.1)

# In training loop
for epoch in range(100):
    # ... training code ...
    scheduler.step()  # Update learning rate
```

---

## Question 8
- [ ] Done

**Explain loss functions in PyTorch.**

### Answer

### Common Loss Functions

| Loss | Use Case |
|------|----------|
| `CrossEntropyLoss` | Multi-class classification |
| `BCELoss` | Binary classification |
| `MSELoss` | Regression |
| `L1Loss` | Robust regression |
| `NLLLoss` | With log_softmax output |

### Python Code Example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-class classification
criterion = nn.CrossEntropyLoss()
logits = torch.randn(32, 10)  # Batch of 32, 10 classes
labels = torch.randint(0, 10, (32,))  # Integer labels
loss = criterion(logits, labels)

# Binary classification
bce_loss = nn.BCELoss()
probs = torch.sigmoid(torch.randn(32, 1))
binary_labels = torch.randint(0, 2, (32, 1)).float()
loss = bce_loss(probs, binary_labels)

# BCEWithLogitsLoss (more stable)
bce_logits = nn.BCEWithLogitsLoss()
logits = torch.randn(32, 1)
loss = bce_logits(logits, binary_labels)

# Regression
mse_loss = nn.MSELoss()
predictions = torch.randn(32, 1)
targets = torch.randn(32, 1)
loss = mse_loss(predictions, targets)

# Custom loss function
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
```

---

## Question 9
- [ ] Done

**What is the difference between model.train() and model.eval()?**

### Answer

### Comparison

| Mode | Dropout | BatchNorm | Gradients |
|------|---------|-----------|-----------|
| `train()` | Active | Updates stats | Computed |
| `eval()` | Disabled | Uses running stats | Optional |

### Python Code Example
```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.BatchNorm1d(64),
    nn.Dropout(0.5),
    nn.Linear(64, 1)
)

# Training mode
model.train()
x_train = torch.randn(32, 10)
output_train = model(x_train)  # Dropout active, BatchNorm updates

# Evaluation mode
model.eval()
x_test = torch.randn(32, 10)

with torch.no_grad():  # Also disable gradient computation
    output_eval = model(x_test)  # Dropout inactive, BatchNorm uses learned stats

# Common pattern
def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(x)

# Don't forget to switch back for training!
model.train()
```

### Key Points
- Always call `model.eval()` before inference
- Use `torch.no_grad()` for faster inference
- Remember to call `model.train()` when resuming training

---

## Question 10
- [ ] Done

**How does PyTorch handle GPU computation?**

### Answer

### GPU Operations

| Operation | Code |
|-----------|------|
| Check availability | `torch.cuda.is_available()` |
| Move to GPU | `tensor.to('cuda')` or `tensor.cuda()` |
| Move to CPU | `tensor.to('cpu')` or `tensor.cpu()` |
| Device count | `torch.cuda.device_count()` |

### Python Code Example
```python
import torch
import torch.nn as nn

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Move tensor to GPU
x = torch.randn(1000, 1000)
x_gpu = x.to(device)
# or x_gpu = x.cuda()

# Move model to GPU
model = nn.Linear(1000, 100)
model = model.to(device)

# Ensure data and model are on same device
x_gpu = torch.randn(32, 1000, device=device)
output = model(x_gpu)

# Mixed precision training (faster)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():  # Use float16 where safe
    output = model(x_gpu)
    loss = output.sum()

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Clear GPU memory
torch.cuda.empty_cache()
```


---

# --- Missing Questions Restored from Source (Q11-Q21) ---

## Question 11
- [ ] Done

**What is the purpose of zero_grad() in PyTorch, and when is it used?**

**Answer:**

### Definition
`zero_grad()` resets (zeros out) the gradients of all model parameters before each backward pass. PyTorch **accumulates gradients by default**, so without calling `zero_grad()`, gradients from multiple backward passes stack up.

### Why It's Needed

| Without `zero_grad()` | With `zero_grad()` |
|----------------------|--------------------|
| Gradients accumulate across batches | Fresh gradients each batch |
| Incorrect parameter updates | Correct parameter updates |
| Training diverges | Training converges |

### Code Example
```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for batch_x, batch_y in dataloader:
    optimizer.zero_grad()          # Step 1: Zero gradients
    output = model(batch_x)         # Step 2: Forward pass
    loss = criterion(output, batch_y)  # Step 3: Compute loss
    loss.backward()                 # Step 4: Backward pass (compute gradients)
    optimizer.step()                # Step 5: Update parameters

# Alternative: zero_grad on model
model.zero_grad()  # Equivalent if optimizer covers all model params

# PyTorch 1.7+: set_to_none for better performance
optimizer.zero_grad(set_to_none=True)  # Sets grads to None instead of 0
```

### When Gradient Accumulation is Intentional
```python
# Simulate larger batch size with gradient accumulation
accum_steps = 4
for i, (x, y) in enumerate(dataloader):
    loss = criterion(model(x), y) / accum_steps  # Scale loss
    loss.backward()  # Accumulate gradients
    if (i + 1) % accum_steps == 0:
        optimizer.step()        # Update every N batches
        optimizer.zero_grad()   # Then zero
```

### Interview Tip
Always explain that PyTorch accumulates gradients by design (useful for gradient accumulation). `optimizer.zero_grad(set_to_none=True)` (PyTorch 1.7+) is more memory-efficient than the default because it deallocates gradient tensors instead of filling them with zeros.

---

## Question 12
- [ ] Done

**Describe the process of backpropagation in PyTorch**

**Answer:**

### Definition
Backpropagation is the algorithm that computes gradients of the loss with respect to model parameters using the **chain rule of calculus**. PyTorch's **Autograd engine** handles this automatically.

### The Process

| Step | Code | What Happens |
|------|------|--------------|
| 1. **Forward pass** | `output = model(x)` | Build computation graph |
| 2. **Compute loss** | `loss = criterion(output, y)` | Scalar output |
| 3. **Backward pass** | `loss.backward()` | Compute all gradients |
| 4. **Update weights** | `optimizer.step()` | Apply gradients |

### Code Example
```python
import torch
import torch.nn as nn

# Simple network
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop with backpropagation
for epoch in range(10):
    for x, y in train_loader:
        optimizer.zero_grad()       # Clear old gradients
        
        # Forward: input → hidden → output (graph built dynamically)
        output = model(x)           
        loss = criterion(output, y) 
        
        # Backward: compute ∂loss/∂w for every parameter
        loss.backward()             
        
        # Inspect gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad norm = {param.grad.norm():.4f}")
        
        optimizer.step()            # w = w - lr * ∂loss/∂w
```

### How Autograd Builds the Graph
```python
x = torch.randn(3, requires_grad=True)
y = x * 2         # y.grad_fn = MulBackward
z = y.sum()        # z.grad_fn = SumBackward
z.backward()       # Traverses graph backward: Sum → Mul → x
print(x.grad)      # tensor([2., 2., 2.])
```

### Interview Tip
PyTorch uses **dynamic computation graphs** (define-by-run), meaning the graph is rebuilt every forward pass. This enables conditional logic and variable-length inputs, unlike TensorFlow 1.x's static graphs. The graph is automatically freed after `backward()` unless `retain_graph=True`.

---

## Question 13
- [ ] Done

**Explain how gradient clipping works in PyTorch and why it may be necessary**

**Answer:**

### Definition
Gradient clipping limits the magnitude of gradients during training to prevent the **exploding gradient problem**, where gradients become extremely large and cause unstable training.

### Types of Gradient Clipping

| Type | Method | How It Works |
|------|--------|-------------|
| **Clip by norm** | `clip_grad_norm_()` | Scales gradient vector if norm exceeds threshold |
| **Clip by value** | `clip_grad_value_()` | Clamps each gradient element to [-value, value] |

### Code Example
```python
import torch
import torch.nn as nn

model = nn.LSTM(input_size=100, hidden_size=256, num_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for x, y in dataloader:
    optimizer.zero_grad()
    output, _ = model(x)
    loss = criterion(output, y)
    loss.backward()
    
    # Clip by norm (most common) - scales gradients if total norm > max_norm
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # OR Clip by value - clamps each gradient independently
    # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
    
    optimizer.step()

# Monitor gradient norms
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"Gradient norm: {total_norm:.4f}")  # Returns norm before clipping
```

### When Gradient Clipping Is Necessary
- **RNNs/LSTMs**: Long sequences cause exploding gradients
- **Transformers**: Deep attention layers
- **GANs**: Unstable training dynamics
- **Very deep networks**: Gradients compound across layers
- **Large learning rates**: Amplify gradient issues

### How Clip by Norm Works
```
if ||gradient|| > max_norm:
    gradient = gradient * (max_norm / ||gradient||)
# Direction preserved, magnitude capped
```

### Interview Tip
`clip_grad_norm_()` is preferred over `clip_grad_value_()` because it preserves gradient direction while only reducing magnitude. The returned value is the original norm, which is useful for monitoring — if it's consistently above your threshold, your learning rate may be too high.

---

## Question 14
- [ ] Done

**Explain batch normalization and its effects on training convergence**

**Answer:**

### Definition
Batch Normalization (BatchNorm) normalizes layer inputs by re-centering and re-scaling across the batch dimension, stabilizing and accelerating training.

### Formula
$$\hat{x} = \frac{x - \mu_{batch}}{\sqrt{\sigma^2_{batch} + \epsilon}} \cdot \gamma + \beta$$

Where $\gamma$ (scale) and $\beta$ (shift) are **learnable parameters**.

### Effects on Training

| Effect | Description |
|--------|-------------|
| **Faster convergence** | Allows higher learning rates |
| **Reduces internal covariate shift** | Stabilizes layer input distributions |
| **Regularization** | Mini-batch noise acts as regularizer |
| **Gradient flow** | Mitigates vanishing/exploding gradients |
| **Less sensitive to initialization** | More forgiving weight init |

### Code Example
```python
import torch.nn as nn

# BatchNorm in CNN
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),          # BatchNorm for 2D (after conv)
    nn.ReLU(),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(128, 256),
    nn.BatchNorm1d(256),         # BatchNorm for 1D (after linear)
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Train vs Eval mode matters!
model.train()   # Uses batch statistics (mean, var of current batch)
model.eval()    # Uses running statistics (accumulated during training)
```

### BatchNorm Variants

| Variant | Class | Use Case |
|---------|-------|----------|
| **BatchNorm1d** | `nn.BatchNorm1d` | After Linear layers |
| **BatchNorm2d** | `nn.BatchNorm2d` | After Conv2d layers |
| **LayerNorm** | `nn.LayerNorm` | Transformers, RNNs |
| **GroupNorm** | `nn.GroupNorm` | Small batch sizes |
| **InstanceNorm** | `nn.InstanceNorm2d` | Style transfer |

### Interview Tip
Critical distinction: **train mode** uses current batch statistics, **eval mode** uses running averages accumulated during training. Forgetting to call `model.eval()` before inference is a common bug that causes inconsistent predictions.

---

## Question 15
- [ ] Done

**How does PyTorch handle weight initialization for neural networks?**

**Answer:**

### Default Initialization
PyTorch initializes layers with specific default strategies:

| Layer | Default Init |
|-------|--------------|
| **Linear** | Kaiming Uniform |
| **Conv2d** | Kaiming Uniform |
| **BatchNorm** | weight=1, bias=0 |
| **LSTM/GRU** | Uniform(-1/√h, 1/√h) |
| **Embedding** | Normal(0, 1) |

### Common Initialization Methods
```python
import torch.nn as nn
import torch.nn.init as init

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)      # Good for sigmoid/tanh
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Best for ReLU
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        init.ones_(m.weight)
        init.zeros_(m.bias)

# Apply to model
model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 10)
)
model.apply(init_weights)  # Applies recursively to all submodules
```

### Initialization Methods Comparison

| Method | Best For | Formula |
|--------|----------|---------|
| **Xavier/Glorot Uniform** | Sigmoid, Tanh | $U(-\sqrt{6/(fan_{in}+fan_{out})}, \sqrt{6/(fan_{in}+fan_{out})})$ |
| **Xavier Normal** | Sigmoid, Tanh | $N(0, \sqrt{2/(fan_{in}+fan_{out})})$ |
| **Kaiming/He Uniform** | ReLU, LeakyReLU | $U(-\sqrt{6/fan_{in}}, \sqrt{6/fan_{in}})$ |
| **Kaiming Normal** | ReLU, LeakyReLU | $N(0, \sqrt{2/fan_{in}})$ |
| **Orthogonal** | RNNs | Orthogonal matrix |
| **Zeros** | Biases | All zeros |

### Interview Tip
The key insight: Xavier init assumes linear activations, Kaiming init accounts for ReLU's zero-killing. Using the wrong init can cause vanishing/exploding activations. With modern architectures + BatchNorm + Adam, initialization matters less, but it's still important for deep networks without normalization.

---

## Question 16
- [ ] Done

**What are some common issues you may encounter when training models in PyTorch, and how do you troubleshoot them?**

**Answer:**

### Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Exploding gradients** | Loss becomes NaN/Inf | Gradient clipping, lower LR |
| **Vanishing gradients** | No learning, loss plateaus | Better init, BatchNorm, skip connections |
| **Overfitting** | Train acc high, val acc low | Dropout, augmentation, more data |
| **Underfitting** | Both accuracies low | Larger model, longer training, higher LR |
| **CUDA OOM** | `RuntimeError: CUDA out of memory` | Smaller batch, mixed precision, gradient checkpointing |
| **Shape mismatch** | `RuntimeError: size mismatch` | Print shapes at each layer |
| **Not learning** | Loss doesn't decrease | Check LR, loss function, data pipeline |

### Debugging Techniques
```python
import torch

# 1. Check for NaN/Inf in loss
if torch.isnan(loss) or torch.isinf(loss):
    print("NaN/Inf detected in loss!")
    # Check inputs, gradients, learning rate

# 2. Monitor gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        if grad_norm > 100:
            print(f"Exploding: {name} grad={grad_norm:.2f}")
        elif grad_norm < 1e-7:
            print(f"Vanishing: {name} grad={grad_norm:.2e}")

# 3. Overfit on one batch (sanity check)
for x, y in train_loader:
    for i in range(100):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        if i % 10 == 0: print(f"Step {i}: loss={loss.item():.4f}")
    break  # Only one batch

# 4. Check data pipeline
for x, y in train_loader:
    print(f"Input: shape={x.shape}, dtype={x.dtype}, range=[{x.min():.2f}, {x.max():.2f}]")
    print(f"Labels: shape={y.shape}, unique={y.unique()}")
    break

# 5. GPU memory debugging
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
torch.cuda.empty_cache()
```

### Common Mistakes
1. **Forgetting `model.eval()`** before inference (BatchNorm/Dropout behave differently)
2. **Forgetting `torch.no_grad()`** during inference (wastes memory)
3. **Forgetting `zero_grad()`** (gradients accumulate)
4. **Wrong loss function** (BCE for multi-class instead of CE)
5. **Data not on same device** (CPU tensor vs CUDA tensor)

### Interview Tip
The best debugging technique is the **overfit-one-batch test**: if your model can't perfectly memorize a single batch, the bug is in your model, data loading, or loss function — not in hyperparameters.

---

## Question 17
- [ ] Done

**What is the use of transforms in PyTorch’s torchvision package?**

**Answer:**

### Definition
Transforms in `torchvision.transforms` are preprocessing and data augmentation operations applied to images before feeding them to a model.

### Common Transforms

| Transform | Purpose | Example |
|-----------|---------|--------|
| **Resize** | Standardize size | `Resize(224)` |
| **ToTensor** | PIL/ndarray → Tensor | `ToTensor()` |
| **Normalize** | Match pre-trained stats | `Normalize(mean, std)` |
| **RandomCrop** | Data augmentation | `RandomCrop(224)` |
| **RandomHorizontalFlip** | Data augmentation | `RandomHorizontalFlip(p=0.5)` |
| **ColorJitter** | Color augmentation | `ColorJitter(0.2, 0.2, 0.2)` |
| **Compose** | Chain transforms | `Compose([...])` |

### Code Example
```python
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

# Validation transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Apply to dataset
train_dataset = datasets.ImageFolder('data/train', transform=train_transform)
val_dataset = datasets.ImageFolder('data/val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
```

### torchvision v2 Transforms (Modern API)
```python
from torchvision.transforms import v2

# v2: Works on tensors, batches, bounding boxes, masks
transform_v2 = v2.Compose([
    v2.RandomResizedCrop(224),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Interview Tip
Always apply augmentation only to training data, not validation/test. The `Normalize` values `[0.485, 0.456, 0.406]` and `[0.229, 0.224, 0.225]` are ImageNet statistics — use them when using pre-trained models. For models trained from scratch, compute your dataset's statistics.

---

## Question 18
- [ ] Done

**What is PyTorch’s TorchScript , and how does it aid in deploying PyTorch models in production environments?**

**Answer:**

### Definition
TorchScript is a way to serialize and optimize PyTorch models for production deployment, enabling them to run **without a Python runtime** (e.g., in C++, mobile, or embedded systems).

### Two Approaches

| Method | How | Best For |
|--------|-----|----------|
| **Tracing** (`torch.jit.trace`) | Records operations on example input | Simple models without control flow |
| **Scripting** (`torch.jit.script`) | Analyzes Python code directly | Models with if/else, loops |

### Code Example
```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    def forward(self, x):
        return torch.relu(self.linear(x))

model = MyModel()
model.eval()

# Method 1: Tracing (for models without control flow)
example_input = torch.randn(1, 10)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('model_traced.pt')

# Method 2: Scripting (for models with control flow)
class ConditionalModel(nn.Module):
    def forward(self, x):
        if x.sum() > 0:      # Control flow!
            return x * 2
        return x * 3

scripted_model = torch.jit.script(ConditionalModel())
scripted_model.save('model_scripted.pt')

# Load and run without Python
loaded = torch.jit.load('model_traced.pt')
output = loaded(torch.randn(1, 10))

# C++ deployment
# #include <torch/script.h>
# auto model = torch::jit::load("model_traced.pt");
# auto output = model.forward({input_tensor});
```

### Benefits for Production

| Benefit | Description |
|---------|-------------|
| **No Python needed** | Run in C++, Java, mobile |
| **Optimizations** | Graph fusion, constant folding |
| **Portability** | Same model file across platforms |
| **Mobile deployment** | PyTorch Mobile (iOS/Android) |
| **Reproducibility** | Frozen, serialized computation |

### Interview Tip
Use `torch.jit.trace` for simple feed-forward models and `torch.jit.script` for models with control flow (if/else, loops). In practice, tracing is more reliable and widely used. For maximum performance, combine TorchScript with `torch.compile()` (PyTorch 2.0+).

---

## Question 19
- [ ] Done

**Explain the concept of “model quantization” in PyTorch and when it is useful.**

**Answer:**

### Definition
Quantization reduces model size and increases inference speed by converting weights and activations from 32-bit floating point (FP32) to lower-precision formats like 8-bit integers (INT8).

### Quantization Types

| Type | When Applied | Accuracy | Speed |
|------|-------------|----------|-------|
| **Dynamic** | At runtime | Good | 2-3x |
| **Static (PTQ)** | Post-training with calibration | Better | 3-4x |
| **Quantization-Aware Training (QAT)** | During training | Best | 3-4x |

### Code Example
```python
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, quantize, prepare, convert

model = MyModel()
model.eval()

# 1. Dynamic Quantization (easiest, great for NLP)
quant_model = quantize_dynamic(
    model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
)

# 2. Static Quantization (Post-Training)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # x86
prepared = prepare(model)  # Insert observers

# Calibrate with representative data
with torch.no_grad():
    for x, _ in calibration_loader:
        prepared(x)  # Collect statistics

quant_model_static = convert(prepared)  # Convert to quantized

# 3. Quantization-Aware Training (best accuracy)
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_qat = prepare_qat(model.train())  # Insert fake quantize modules

for epoch in range(5):
    for x, y in train_loader:
        output = model_qat(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

quant_model_qat = convert(model_qat.eval())
```

### Size and Speed Comparison

| Metric | FP32 | INT8 (Quantized) |
|--------|------|------------------|
| **Model size** | 100 MB | ~25 MB (4x smaller) |
| **Inference speed** | 1x | 2-4x faster |
| **Memory** | 100% | ~25% |
| **Accuracy loss** | Baseline | 0.5-2% typical |

### When to Quantize
- **Edge/mobile deployment** — limited memory and compute
- **CPU inference** — INT8 is much faster on CPU
- **Cost reduction** — smaller models need less compute
- **Latency-sensitive** — real-time applications

### Interview Tip
Dynamic quantization is a "free lunch" for inference — one line of code for 2-3x speedup with minimal accuracy loss. It's especially effective for NLP models (LSTM, Transformer) where Linear layers dominate compute.

---

## Question 20
- [ ] Done

**What is the role of PyTorch in reinforcement learning research, and can you provide an example?**

**Answer:**

### PyTorch in RL
PyTorch is the dominant framework for RL research due to its **dynamic computation graphs**, which naturally handle variable-length episodes, conditional actions, and stochastic policies.

### Why PyTorch for RL

| Feature | RL Benefit |
|---------|--------|
| **Dynamic graphs** | Variable-length episodes, conditional logic |
| **Autograd** | Easy policy gradient computation |
| **Distributions** | `torch.distributions` for stochastic policies |
| **GPU support** | Fast environment simulation |
| **Ecosystem** | Stable-Baselines3, RLlib, TorchRL |

### REINFORCE (Policy Gradient) Example
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym

# Policy network
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

# Training
env = gym.make('CartPole-v1')
policy = PolicyNet(4, 2)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

for episode in range(1000):
    state, _ = env.reset()
    log_probs, rewards = [], []
    
    # Collect trajectory
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state)
        probs = policy(state_tensor)
        dist = Categorical(probs)           # Stochastic policy
        action = dist.sample()              # Sample action
        log_probs.append(dist.log_prob(action))
        
        state, reward, done, _, _ = env.step(action.item())
        rewards.append(reward)
    
    # Compute discounted returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # Policy gradient update
    loss = -sum(lp * G for lp, G in zip(log_probs, returns))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Popular RL Libraries Built on PyTorch

| Library | Focus |
|---------|-------|
| **Stable-Baselines3** | Reliable RL algorithms (PPO, SAC, DQN) |
| **TorchRL** | Official PyTorch RL library |
| **RLlib (Ray)** | Distributed, scalable RL |
| **CleanRL** | Single-file RL implementations |

### Interview Tip
PyTorch dominates RL research because `torch.distributions` + autograd makes implementing policy gradients natural. The key line is `dist.log_prob(action)` — this gives you the log probability needed for the REINFORCE gradient: $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot G_t]$.

---

## Question 21
- [ ] Done

**Describe your experience contributing to PyTorch’s open-source community or using community-created tools.**

**Answer:**

### PyTorch Ecosystem Overview

| Category | Key Projects | Description |
|----------|-------------|-------------|
| **Vision** | torchvision, timm, Detectron2 | Image models, detection |
| **NLP** | HuggingFace Transformers, torchtext | Language models, tokenizers |
| **Audio** | torchaudio, SpeechBrain | Audio processing, ASR |
| **Graphs** | PyG (PyTorch Geometric), DGL | Graph neural networks |
| **RL** | Stable-Baselines3, TorchRL | Reinforcement learning |
| **Production** | TorchServe, ONNX | Model serving, export |
| **Research** | PyTorch Lightning, FastAI | Training abstractions |

### Ways to Contribute and Engage

```
Contribution Levels:
1. 📚 User: Use PyTorch + community tools in projects
2. 🐛 Bug Reporter: File issues with reproducible examples
3. 📖 Documentation: Improve tutorials, fix docs
4. 🔧 Code: Fix bugs, add features, review PRs
5. 🏠 Maintainer: Own a module or ecosystem project
```

### Community Tools Example
```python
# PyTorch Lightning - reduces boilerplate
import pytorch_lightning as pl

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self.model(x), y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

trainer = pl.Trainer(max_epochs=10, accelerator='gpu')
trainer.fit(LitModel(), train_loader)

# timm - huge model zoo
import timm
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=10)

# HuggingFace - state-of-the-art NLP
from transformers import AutoModel
model = AutoModel.from_pretrained('bert-base-uncased')
```

### Contributing Best Practices
1. **Start small**: Fix typos, improve docstrings
2. **Read contributing guidelines**: Each project has different standards
3. **Write tests**: All contributions should include tests
4. **Follow code style**: Use the project's linting/formatting rules
5. **Engage in discussions**: GitHub Issues, PyTorch Forums, Discord

### Interview Tip
Mention specific community tools you've used (Lightning for training, timm for models, HuggingFace for NLP) and how they improved your workflow. Even using community tools extensively counts as ecosystem engagement — you don't need to be a core contributor.

---

## Question 22
- [ ] Done

**What is PyTorch and how does it differ from other deep learning frameworks like TensorFlow ?**

**Answer:**

**PyTorch** is an open-source deep learning framework developed by Meta AI (formerly Facebook). It provides a dynamic computational graph (define-by-run) and a Python-first development experience.

| Feature | PyTorch | TensorFlow |
|---------|---------|------------|
| **Graph** | Dynamic (eager by default) | Static (graph mode) / Eager (TF 2.x) |
| **Debugging** | Standard Python debugger | Requires special tools |
| **Community** | Dominates research | Strong in production |
| **Deployment** | TorchServe, ONNX | TF Serving, TFLite |
| **Mobile** | PyTorch Mobile | TensorFlow Lite |
| **API Style** | Pythonic, NumPy-like | Keras high-level API |

```python
import torch

# Tensors (like NumPy arrays but GPU-accelerated)
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2 + 2 * x
y.sum().backward()     # Automatic differentiation
print(x.grad)          # Gradients: [4.0, 6.0, 8.0]

# GPU support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = torch.randn(3, 3).to(device)
```

> **Interview Tip:** PyTorch dominates **research** (80%+ of papers) due to its dynamic graph and Pythonic API. TensorFlow has stronger **production deployment** tools. Both frameworks are converging in capabilities.

---

## Question 23
- [ ] Done

**In PyTorch, what is the difference between a Tensor and a Variable ?**

**Answer:**

| Aspect | Tensor | Variable (deprecated) |
|--------|--------|----------------------|
| **Current status** | Primary data structure | **Deprecated since PyTorch 0.4** |
| **Gradient tracking** | `requires_grad=True` | Was the only way to track gradients |
| **Usage** | All operations | Legacy code only |
| **API** | `torch.tensor()` | `torch.autograd.Variable()` |

```python
import torch

# Modern PyTorch: Tensors handle everything
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
y.sum().backward()
print(x.grad)  # tensor([2., 4., 6.])

# Legacy (deprecated, do NOT use):
# from torch.autograd import Variable
# x = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)

# Key: Variable was merged into Tensor in PyTorch 0.4
# torch.Tensor now has all Variable functionality:
# - .requires_grad  (track gradients)
# - .grad           (stored gradients)
# - .backward()     (compute gradients)
# - .detach()       (stop gradient tracking)
```

> **Interview Tip:** `Variable` is **completely deprecated**. If you see it in code, it's legacy. Modern PyTorch uses `torch.Tensor` with `requires_grad=True` for all gradient operations.

---

## Question 24
- [ ] Done

**How can you convert a NumPy array to a PyTorch Tensor ?**

**Answer:**

```python
import torch
import numpy as np

# NumPy -> PyTorch Tensor
np_array = np.array([1.0, 2.0, 3.0])
tensor1 = torch.from_numpy(np_array)    # Shares memory (changes reflect both)
tensor2 = torch.tensor(np_array)         # Creates a copy (independent)

# PyTorch Tensor -> NumPy
tensor = torch.tensor([1.0, 2.0, 3.0])
np_array1 = tensor.numpy()              # Shares memory (CPU tensors only)
np_array2 = tensor.detach().cpu().numpy()  # Safe: detach from graph, move to CPU
```

| Method | Direction | Memory | Notes |
|--------|-----------|--------|-------|
| `torch.from_numpy(arr)` | NumPy -> Tensor | Shared | Changes in one affect the other |
| `torch.tensor(arr)` | NumPy -> Tensor | Copy | Independent, preferred for safety |
| `tensor.numpy()` | Tensor -> NumPy | Shared | CPU tensors only |
| `tensor.detach().cpu().numpy()` | Tensor -> NumPy | Copy | Safe for GPU tensors with gradients |

> **Interview Tip:** Use `torch.tensor()` (copy) for safety unless memory sharing is intentional. For GPU tensors, always call `.detach().cpu().numpy()` to avoid errors. Shared memory via `from_numpy()` is useful for zero-copy data loading.

---

## Question 25
- [ ] Done

**What is the purpose of the .grad attribute in PyTorch Tensors ?**

**Answer:**

The `.grad` attribute stores the gradient of a tensor computed during backpropagation. It is only populated for leaf tensors with `requires_grad=True` after calling `.backward()`.

```python
import torch

# .grad stores computed gradients
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1  # y = x^2 + 2x + 1
y.backward()             # Compute dy/dx
print(x.grad)            # tensor(8.) -> dy/dx = 2x + 2 = 8

# Important: gradients accumulate by default!
x.grad.zero_()           # Must manually zero before next backward
y = x ** 3
y.backward()
print(x.grad)            # tensor(27.) -> dy/dx = 3x^2 = 27

# In training loops, use optimizer.zero_grad()
optimizer = torch.optim.SGD([x], lr=0.01)
optimizer.zero_grad()    # Zeros all parameter gradients
loss.backward()          # Compute gradients
optimizer.step()         # Update parameters using gradients
```

| Property | Description |
|----------|-------------|
| `tensor.grad` | Stores accumulated gradients |
| `tensor.requires_grad` | Whether to track gradients |
| `tensor.grad_fn` | Function that created this tensor (for graph tracing) |
| `tensor.is_leaf` | True for user-created tensors |

> **Interview Tip:** Gradients **accumulate** in PyTorch (they are summed, not replaced). Always call `optimizer.zero_grad()` or `tensor.grad.zero_()` before each backward pass. This design enables gradient accumulation for large effective batch sizes.

---

## Question 26
- [ ] Done

**Explain what CUDA is and how it relates to PyTorch**

**Answer:**

**CUDA** (Compute Unified Device Architecture) is NVIDIA's parallel computing platform that allows PyTorch to run computations on GPUs, providing massive speedups for tensor operations and neural network training.

```python
import torch

# Check CUDA availability
print(torch.cuda.is_available())          # True if GPU available
print(torch.cuda.device_count())          # Number of GPUs
print(torch.cuda.get_device_name(0))      # GPU name
print(torch.cuda.current_device())        # Current GPU index

# Move tensors to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = torch.randn(1000, 1000).to(device)
# Or: tensor = torch.randn(1000, 1000, device='cuda')

# Move model to GPU
model = MyModel().to(device)

# Move data to GPU in training loop
for x, y in dataloader:
    x, y = x.to(device), y.to(device)
    output = model(x)

# Multi-GPU with DataParallel
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Memory management
torch.cuda.empty_cache()                  # Free unused GPU memory
print(torch.cuda.memory_allocated())      # Current GPU memory usage
```

| Operation | CPU | GPU (CUDA) |
|-----------|-----|------------|
| Matrix multiply (1000x1000) | ~100ms | ~1ms |
| Model training | Hours/days | Minutes/hours |
| Batch inference | Slow | Fast with batching |

> **Interview Tip:** Always use `.to(device)` pattern for device-agnostic code. Use `torch.cuda.empty_cache()` to free memory between experiments. For multi-GPU, prefer `DistributedDataParallel` over `DataParallel` for better performance.

---

## Question 27
- [ ] Done

**How does automatic differentiation work in PyTorch using Autograd ?**

**Answer:**

**Autograd** is PyTorch's automatic differentiation engine that records all operations on tensors in a dynamic computational graph and computes gradients via reverse-mode differentiation (backpropagation).

| Concept | Description |
|---------|-------------|
| **Dynamic graph** | Graph is built on-the-fly during forward pass |
| **Leaf tensors** | User-created tensors with `requires_grad=True` |
| **grad_fn** | Records the operation that created a tensor |
| **backward()** | Traverses graph in reverse to compute gradients |

```python
import torch

# Autograd tracks operations automatically
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = x ** 2 + 3 * y    # z = x^2 + 3y
z.backward()           # Compute gradients
print(x.grad)          # dz/dx = 2x = 4.0
print(y.grad)          # dz/dy = 3.0

# Control gradient tracking
with torch.no_grad():
    # Operations here won't be tracked (inference, evaluation)
    output = model(input_data)

# Detach from graph
detached = tensor.detach()  # Creates tensor sharing data but not tracking gradients

# Custom autograd function
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * (x > 0).float()
```

> **Interview Tip:** Autograd builds a new graph for **every forward pass** (dynamic), unlike TF 1.x's static graph. Use `torch.no_grad()` during inference to save memory. The `backward()` call traverses this graph to compute all gradients in one pass.

---

## Question 28
- [ ] Done

**Describe the steps for creating a neural network model in PyTorch**

**Answer:**

### Steps to Create a Model in PyTorch

| Step | Action |
|------|--------|
| 1 | Define model class inheriting `nn.Module` |
| 2 | Define layers in `__init__` |
| 3 | Implement `forward()` method |
| 4 | Initialize model and move to device |
| 5 | Define loss function and optimizer |
| 6 | Write training loop |

```python
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

# Initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(784, 256, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
```

> **Interview Tip:** Always inherit from `nn.Module` and define all layers in `__init__`. The `forward()` method defines the computation graph. Never call `forward()` directly; use `model(x)` which also runs hooks.

---

## Question 29
- [ ] Done

**What is a Sequential model in PyTorch, and how does it differ from using the Module class?**

**Answer:**

| Aspect | `nn.Sequential` | `nn.Module` |
|--------|-----------------|-------------|
| **Use case** | Simple linear stacks | Any architecture |
| **Flexibility** | Limited (layers in order) | Full (custom forward logic) |
| **Multiple inputs/outputs** | No | Yes |
| **Skip connections** | No | Yes |
| **Conditional logic** | No | Yes |

```python
import torch.nn as nn

# nn.Sequential: for simple feed-forward networks
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
output = model(x)  # Passes through layers sequentially

# nn.Module: for complex architectures
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x                          # Skip connection
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual                  # Add skip connection
        return self.relu(out)

# Pro tip: use Sequential inside Module for cleaner code
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 14 * 14, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

> **Interview Tip:** Use `nn.Sequential` for simple stacks within larger `nn.Module` classes. Any architecture with skip connections, multiple branches, or conditional logic requires `nn.Module` with a custom `forward()` method.

---

## Question 30
- [ ] Done

**What is the role of the forward method in a PyTorch Module ?**

**Answer:**

The `forward()` method defines how data flows through the model. It is called automatically when you invoke the model as a function: `output = model(input)`.

| Aspect | Detail |
|--------|--------|
| **Purpose** | Defines the computation graph (forward pass) |
| **Called by** | `model(x)` (NOT `model.forward(x)`) |
| **Dynamic** | Can include Python control flow (if/else, loops) |
| **Hooks** | `model(x)` triggers registered hooks; `model.forward(x)` does not |

```python
import torch
import torch.nn as nn

class DynamicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(100, 64)
        self.linear2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))

        # Dynamic behavior: different paths based on input
        if x.mean() > 0:
            x = self.relu(self.linear2(x))
        else:
            x = self.relu(self.linear2(x)) * 2

        return self.output_layer(x)

# CORRECT: triggers hooks and proper behavior
model = DynamicNet()
output = model(input_tensor)

# INCORRECT: bypasses hooks
# output = model.forward(input_tensor)  # Don't do this!
```

> **Interview Tip:** Always call `model(x)`, never `model.forward(x)`. The `__call__` method wraps `forward()` and runs pre/post hooks. PyTorch's dynamic graph means `forward()` can contain arbitrary Python control flow, executed fresh each call.

---

## Question 31
- [ ] Done

**In PyTorch, what are optimizers , and how do you use them?**

**Answer:**

**Optimizers** update model parameters based on computed gradients to minimize the loss function.

| Optimizer | Key Feature | Best For |
|-----------|------------|----------|
| `SGD` | Simple, effective with momentum | CNNs, large-scale training |
| `Adam` | Adaptive learning rate per parameter | Default choice, most tasks |
| `AdamW` | Adam with decoupled weight decay | Transformers, modern architectures |
| `RMSprop` | Adaptive, good for non-stationary | RNNs |
| `LBFGS` | Quasi-Newton method | Small-scale, full-batch |

```python
import torch.optim as optim

# Adam (default choice)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

# Discriminative learning rates (different LR per layer group)
optimizer = optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.head.parameters(), 'lr': 1e-3}
])

# Learning rate scheduling
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
# Or:
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Training loop with scheduler
for epoch in range(epochs):
    train_one_epoch(model, optimizer)
    val_loss = evaluate(model)
    scheduler.step(val_loss)  # For ReduceLROnPlateau
    # scheduler.step()        # For other schedulers
```

> **Interview Tip:** **Adam** is the safe default. **SGD with momentum** often achieves better final accuracy with proper LR scheduling. Always use a **learning rate scheduler** (cosine annealing is popular). Mention **discriminative learning rates** for transfer learning.

---

## Question 32
- [ ] Done

**How do you create a data loader in PyTorch for custom datasets ?**

**Answer:**

Creating a custom DataLoader involves implementing a `Dataset` class with `__len__` and `__getitem__`, then wrapping it with `DataLoader`.

```python
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, root_dir, labels_dict, transform=None):
        self.root_dir = root_dir
        self.image_files = list(labels_dict.keys())
        self.labels = list(labels_dict.values())
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# DataLoader with options
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,          # Parallel loading threads
    pin_memory=True,        # Faster CPU->GPU transfer
    persistent_workers=True, # Keep workers alive between epochs
    collate_fn=None,        # Custom batching logic (optional)
    drop_last=True          # Drop incomplete final batch
)

# Custom collate function (for variable-length sequences)
def custom_collate(batch):
    data = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    data_padded = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    return data_padded, labels

loader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate)
```

> **Interview Tip:** Use `pin_memory=True` + `num_workers > 0` for GPU training. Implement `collate_fn` for variable-length sequences. Use `persistent_workers=True` to avoid worker respawn overhead between epochs.

---

## Question 33
- [ ] Done

**How do you manage and preprocess time-series data in PyTorch for RNNs ?**

**Answer:**

| Step | Action | Details |
|------|--------|---------|
| 1 | **Normalize** | StandardScaler or MinMaxScaler |
| 2 | **Create sequences** | Sliding window of fixed length |
| 3 | **Shape data** | (batch, seq_len, features) |
| 4 | **Split temporally** | Never shuffle; use chronological split |
| 5 | **Build RNN** | LSTM/GRU for long dependencies |

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Prepare sliding windows
def create_sequences(data, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + forecast_horizon])
    return np.array(X), np.array(y)

# 2. Normalize and split (temporal order!)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

seq_len = 60
X, y = create_sequences(data_scaled, seq_len)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]  # No shuffle for time-series!

# 3. Convert to tensors: (batch, seq_len, features)
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)

# 4. LSTM model
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Use last time step
```

> **Interview Tip:** Never shuffle time-series data. Use **walk-forward validation** instead of random splits. Normalize using only training data statistics. Shape must be `(batch, seq_len, features)` for `batch_first=True`.

---

## Question 34
- [ ] Done

**Explain the concept of data augmentation and its implementation in PyTorch**

**Answer:**

**Data augmentation** artificially increases training set diversity by applying random transformations, improving model generalization without collecting more data.

| Transform | Use Case | API |
|-----------|----------|-----|
| `RandomHorizontalFlip` | Natural images | `torchvision.transforms` |
| `RandomRotation` | Rotation-invariant tasks | `torchvision.transforms` |
| `ColorJitter` | Lighting variation | `torchvision.transforms` |
| `RandomCrop` | Position invariance | `torchvision.transforms` |
| `MixUp, CutMix` | Advanced regularization | Custom implementation |

```python
from torchvision import transforms

# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3)  # Cutout-style
])

# Validation/test transforms (NO augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# AutoAugment (learned augmentation policy)
train_transform = transforms.Compose([
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

> **Interview Tip:** Only augment **training data**, never validation/test. For state-of-the-art, use **AutoAugment** or **RandAugment** (learned policies). **MixUp** and **CutMix** are strong regularizers that blend samples together.

---

## Question 35
- [ ] Done

**How do you use GPU accelerators for distributed training in PyTorch?**

**Answer:**

| Strategy | PyTorch API | Description |
|----------|------------|-------------|
| **DataParallel** | `nn.DataParallel` | Simple multi-GPU (single machine) |
| **DistributedDataParallel** | `nn.parallel.DistributedDataParallel` | Recommended, multi-GPU/multi-node |
| **FSDP** | `FullyShardedDataParallel` | Memory-efficient for very large models |
| **DeepSpeed** | Third-party | ZeRO optimization, large models |

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Setup distributed training
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # DistributedSampler ensures each GPU gets different data
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Important for proper shuffling
        for x, y in loader:
            x, y = x.to(rank), y.to(rank)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    cleanup()

# Launch: torchrun --nproc_per_node=4 train.py
```

> **Interview Tip:** Always prefer **DistributedDataParallel** over DataParallel (it's faster due to per-GPU gradient reduction). Use `DistributedSampler` with `set_epoch()` for proper shuffling. For models exceeding GPU memory, use **FSDP** or **DeepSpeed**.

---

## Question 36
- [ ] Done

**Compare recurrent neural networks (RNNs) , long short-term memory networks (LSTMs) , and gated recurrent units (GRUs) in the context of PyTorch**

**Answer:**

| Feature | RNN | LSTM | GRU |
|---------|-----|------|-----|
| **Gates** | None | 3 (input, forget, output) | 2 (reset, update) |
| **Memory** | Short-term only | Long-term + short-term | Combined memory |
| **Parameters** | Fewest | Most | Fewer than LSTM |
| **Vanishing gradient** | Severe | Solved | Solved |
| **Speed** | Fastest | Slowest | Faster than LSTM |
| **Performance** | Worst | Best for long sequences | Near-LSTM, more efficient |

```python
import torch.nn as nn

# SimpleRNN: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)
rnn = nn.RNN(input_size=64, hidden_size=128, num_layers=2, batch_first=True)

# LSTM: 3 gates control information flow
# Forget gate: what to discard from cell state
# Input gate: what new info to store
# Output gate: what to output from cell state
lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2,
               batch_first=True, dropout=0.2, bidirectional=True)

# GRU: 2 gates (simplified LSTM)
# Reset gate: controls how much past info to forget
# Update gate: controls how much new info to add
gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2,
             batch_first=True, dropout=0.2)

# Usage
output, (h_n, c_n) = lstm(input_tensor)  # LSTM returns cell state
output, h_n = gru(input_tensor)           # GRU has no cell state
```

> **Interview Tip:** **LSTM** is best for tasks requiring long-range dependencies. **GRU** is preferred when training speed matters and performance is comparable. For modern NLP, **Transformers** have largely replaced all three architectures.

---

## Question 37
- [ ] Done

**Discuss the latest research on neural architecture search (NAS) and its application within PyTorch**

**Answer:**

**Neural Architecture Search (NAS)** automates the design of neural network architectures by searching over a defined search space to find optimal configurations.

| NAS Approach | Description | Example |
|-------------|-------------|---------|
| **Reinforcement Learning** | Controller network generates architectures | NASNet |
| **Evolutionary** | Mutate and select best architectures | AmoebaNet |
| **Differentiable (DARTS)** | Relaxed search with gradient descent | DARTS, ProxylessNAS |
| **One-shot** | Train supernet, evaluate sub-networks | OFA, BigNAS |

```python
# Example using NNI (Neural Network Intelligence) with PyTorch
# pip install nni

import torch.nn as nn

# Define search space
class SearchableBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ops = nn.ModuleDict({
            'conv3x3': nn.Conv2d(in_channels, out_channels, 3, padding=1),
            'conv5x5': nn.Conv2d(in_channels, out_channels, 5, padding=2),
            'sep_conv': nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1)
            ),
            'skip': nn.Identity() if in_channels == out_channels else
                    nn.Conv2d(in_channels, out_channels, 1)
        })

    def forward(self, x, choice):
        return self.ops[choice](x)

# Using optuna for hyperparameter + architecture search
# import optuna
# def objective(trial):
#     n_layers = trial.suggest_int('n_layers', 2, 6)
#     hidden = trial.suggest_categorical('hidden', [64, 128, 256])
#     activation = trial.suggest_categorical('activation', ['relu', 'gelu'])
#     model = build_model(n_layers, hidden, activation)
#     return evaluate(model)
```

> **Interview Tip:** DARTS (Differentiable Architecture Search) is the most practical NAS approach as it uses gradient descent instead of expensive RL/evolutionary methods. In practice, most practitioners use well-known architectures (EfficientNet, ResNet) rather than running NAS from scratch. Mention **optuna** for practical architecture/hyperparameter search.

---

## Question 38
- [ ] Done

**How can generative adversarial networks (GANs) be implemented in PyTorch, and what are some of their challenges?**

**Answer:**

**GANs** consist of a **Generator** (creates fake data) and **Discriminator** (classifies real vs fake), trained in an adversarial minimax game.

| Challenge | Solution |
|-----------|----------|
| **Mode collapse** | Spectral norm, diversity loss, WGAN |
| **Training instability** | WGAN-GP, progressive training |
| **Evaluation** | FID score, Inception Score |
| **Convergence** | Two-timescale update, careful LR tuning |

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 784), nn.Tanh()
        )
    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# Training
G = Generator().to(device)
D = Discriminator().to(device)
g_opt = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

for epoch in range(epochs):
    for real_imgs, _ in dataloader:
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        # Train Discriminator
        z = torch.randn(batch_size, 100).to(device)
        fake_imgs = G(z).detach()
        d_real = D(real_imgs)
        d_fake = D(fake_imgs)
        d_loss = criterion(d_real, torch.ones_like(d_real)) + \
                 criterion(d_fake, torch.zeros_like(d_fake))
        d_opt.zero_grad(); d_loss.backward(); d_opt.step()

        # Train Generator
        z = torch.randn(batch_size, 100).to(device)
        fake_imgs = G(z)
        g_loss = criterion(D(fake_imgs), torch.ones_like(d_real))
        g_opt.zero_grad(); g_loss.backward(); g_opt.step()
```

> **Interview Tip:** Use **LeakyReLU** in discriminator, **BatchNorm** in generator. Use Adam with `betas=(0.5, 0.999)`. For stable training, consider **WGAN-GP** (Wasserstein loss + gradient penalty) or **StyleGAN** architecture.

---
## Question 39
- [ ] Done

**How do you implement custom layers in PyTorch?**

**Answer:**

Custom layers in PyTorch extend `nn.Module` with learnable parameters and a `forward()` method.

```python
import torch
import torch.nn as nn

# === Basic Custom Layer ===
class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # nn.Parameter makes tensors trainable
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        return x @ self.weight + self.bias

# === Custom Layer with Existing Layers ===
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        return self.relu(out + residual)  # Skip connection

# === Attention Layer ===
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5
    
    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

# === Usage ===
model = nn.Sequential(
    LinearLayer(784, 128),
    nn.ReLU(),
    ResidualBlock(128),
    nn.Linear(128, 10)
)

x = torch.randn(32, 784)
output = model(x)
print(output.shape)  # torch.Size([32, 10])
```

| Component | Purpose |
|-----------|--------|
| `nn.Module` | Base class for all layers/models |
| `nn.Parameter` | Registers tensor as trainable parameter |
| `forward()` | Defines computation |
| `__init__()` | Define sub-layers and parameters |

> **Interview Tip:** Every custom layer must call `super().__init__()`. Use `nn.Parameter` for learnable weights; plain tensors won't be updated by the optimizer.

---

## Question 40
- [ ] Done

**How can you implement learning rate scheduling in PyTorch?**

**Answer:**

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# === 1. StepLR: Decay every N epochs ===
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# LR: 0.01 -> 0.005 (epoch 10) -> 0.0025 (epoch 20)

# === 2. ExponentialLR ===
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# LR *= 0.95 each epoch

# === 3. CosineAnnealingLR ===
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
# Cosine decay to eta_min over T_max epochs

# === 4. ReduceLROnPlateau (most practical) ===
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
)

# === 5. OneCycleLR (best for training from scratch) ===
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, total_steps=num_epochs * len(train_loader)
)

# === 6. Warmup + Decay (Custom) ===
def warmup_lambda(epoch):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return epoch / warmup_epochs  # Linear warmup
    return 0.5 * (1 + torch.cos(torch.tensor((epoch - warmup_epochs) / 45 * 3.14159)))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

# === Training Loop ===
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch[0]), batch[1])
        loss.backward()
        optimizer.step()
        # For OneCycleLR: scheduler.step() HERE (per batch)
    
    # For most schedulers: step per epoch
    scheduler.step()  # StepLR, ExponentialLR, CosineAnnealing
    # scheduler.step(val_loss)  # For ReduceLROnPlateau only
    
    print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']:.6f}")
```

| Scheduler | Best For | Step When |
|-----------|----------|----------|
| StepLR | Simple decay | Per epoch |
| CosineAnnealing | Training from scratch | Per epoch |
| ReduceLROnPlateau | Any (adaptive) | Per epoch (pass val_loss) |
| OneCycleLR | Fast convergence | Per batch |
| Warmup + Decay | Large models, transformers | Per epoch |

> **Interview Tip:** `OneCycleLR` often gives the best results for training from scratch. For fine-tuning, use `CosineAnnealingLR` or `ReduceLROnPlateau`. Always call `scheduler.step()` **after** `optimizer.step()`.

---

## Question 41
- [ ] Done

**Explain transfer learning and its implementation in PyTorch.**

**Answer:**

Transfer learning uses a model pre-trained on a large dataset and adapts it to a new, smaller dataset.

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

# === 1. Load Pre-trained Model ===
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# === 2. Freeze All Layers ===
for param in model.parameters():
    param.requires_grad = False

# === 3. Replace Classifier Head ===
num_classes = 10
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)
# New layers are trainable by default

# === 4. Setup Data ===
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 5. Phase 1: Train Head Only ===
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# === 6. Phase 2: Fine-tune Last Layers ===
# Unfreeze last few layers
for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},  # Low LR for pre-trained
    {'params': model.fc.parameters(), 'lr': 1e-4}       # Higher LR for new layers
], weight_decay=1e-4)

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
```

| Strategy | Freeze | When to Use |
|----------|--------|-------------|
| Feature extraction | All base layers | Very small dataset, similar domain |
| Fine-tuning top layers | Most base layers | Medium dataset |
| Full fine-tuning | Nothing | Large dataset, different domain |

> **Interview Tip:** PyTorch allows per-parameter-group learning rates in the optimizer—use lower LR for pre-trained layers and higher LR for new layers. Always use ImageNet normalization when using ImageNet-pretrained models.

---

## Question 42
- [ ] Done

**What are Graph Neural Networks (GNNs) and how can they be implemented in PyTorch?**

**Answer:**

GNNs operate on graph-structured data (nodes + edges), learning representations by aggregating information from neighbors.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# === 1. Simple GCN from Scratch ===
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x, adj):
        # x: (N, in_features), adj: (N, N) adjacency matrix
        support = torch.mm(x, self.weight)    # Transform features
        output = torch.mm(adj, support)        # Aggregate neighbors
        return output + self.bias

class GCN(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes):
        super().__init__()
        self.conv1 = GraphConvolution(n_features, n_hidden)
        self.conv2 = GraphConvolution(n_hidden, n_classes)
    
    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, adj)

# === 2. Using PyTorch Geometric (Production) ===
# pip install torch-geometric
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data

class GCN_PyG(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Graph-level pooling
        return self.classifier(x)

# Create graph data
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
node_features = torch.randn(3, 16)  # 3 nodes, 16 features each
graph = Data(x=node_features, edge_index=edge_index)

# === 3. Graph Attention Network (GAT) ===
class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
```

| GNN Type | Mechanism | Best For |
|----------|-----------|----------|
| GCN | Spectral convolution | Node classification |
| GAT | Attention on neighbors | When neighbor importance varies |
| GraphSAGE | Sampling + aggregating | Large-scale graphs |
| GIN | Isomorphism-based | Graph-level classification |

| Application | Input Graph |
|------------|-------------|
| Drug discovery | Molecular structure |
| Social networks | User connections |
| Fraud detection | Transaction networks |
| Recommendation | User-item interactions |

> **Interview Tip:** GNNs follow a message-passing paradigm: each node aggregates features from its neighbors, applies a transformation, and updates its representation. PyTorch Geometric provides optimized implementations for production use.

## Question 43
- [ ] Done

**How do you check if your PyTorch model is utilizing the GPU ?**

**Answer:**

```python
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.current_device()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Check if model is on GPU
model = MyModel()
print(f"Model device: {next(model.parameters()).device}")  # cpu

model = model.to('cuda')
print(f"Model device: {next(model.parameters()).device}")  # cuda:0

# Check if tensor is on GPU
tensor = torch.randn(3, 3)
print(f"Tensor device: {tensor.device}")  # cpu
tensor = tensor.cuda()
print(f"Tensor device: {tensor.device}")  # cuda:0

# Monitor GPU memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")

# Best practice: device-agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

> **Interview Tip:** Always use `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` for device-agnostic code. Check `next(model.parameters()).device` to verify model placement. Use `torch.cuda.memory_allocated()` to debug OOM errors.

---

## Question 44
- [ ] Done

**What strategies can you use to monitor and decrease overfitting in a PyTorch model?**

**Answer:**

| Strategy | Implementation | Effect |
|----------|---------------|--------|
| **Dropout** | `nn.Dropout(p=0.5)` | Random neuron deactivation |
| **Weight decay** | `optimizer(weight_decay=1e-4)` | L2 regularization |
| **Early stopping** | Monitor val loss, stop when it increases | Prevent overtraining |
| **Data augmentation** | `torchvision.transforms` | Increase training diversity |
| **Batch normalization** | `nn.BatchNorm2d(channels)` | Stabilize + slight regularization |
| **Label smoothing** | `CrossEntropyLoss(label_smoothing=0.1)` | Prevent overconfidence |
| **Gradient clipping** | `torch.nn.utils.clip_grad_norm_` | Prevent exploding gradients |

```python
import torch
import torch.nn as nn

# 1. Dropout
model = nn.Sequential(
    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(128, 10)
)

# 2. Weight decay (L2 regularization)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 3. Early stopping
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

early_stop = EarlyStopping(patience=5)

# 4. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 5. Monitor train vs val gap
for epoch in range(epochs):
    train_loss = train(model)
    val_loss = evaluate(model)
    gap = train_loss - val_loss
    if gap > 0.1:
        print("Warning: potential overfitting!")
    if early_stop(val_loss):
        print("Early stopping triggered")
        break
```

> **Interview Tip:** Combine **Dropout + Weight Decay + Early Stopping** as a baseline. Monitor the gap between train and val loss. If the gap is large, the model is overfitting; if both are high, the model is underfitting.

---

## Question 45
- [ ] Done

**How would you create a PyTorch extension module with custom C++/CUDA operations ?**

**Answer:**

PyTorch allows custom C++ and CUDA extensions for operations that need maximum performance or aren't available in standard PyTorch.

| Method | Use Case | Difficulty |
|--------|----------|------------|
| **torch.utils.cpp_extension** | Custom C++ ops | Medium |
| **CUDA kernels** | GPU-optimized operations | Hard |
| **JIT compilation** | On-the-fly compilation | Easy |

```python
# Method 1: JIT compilation (easiest)
from torch.utils.cpp_extension import load

# my_extension.cpp file:
# #include <torch/extension.h>
# torch::Tensor my_add(torch::Tensor a, torch::Tensor b) {
#     return a + b;
# }
# PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#     m.def("my_add", &my_add, "Custom add");
# }

my_ext = load(name="my_extension", sources=["my_extension.cpp"])
result = my_ext.my_add(tensor_a, tensor_b)

# Method 2: Setup-based (for distribution)
# setup.py:
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

setup(
    name='my_cuda_ext',
    ext_modules=[
        CUDAExtension('my_cuda_ext', [
            'my_extension.cpp',
            'my_kernel.cu',     # CUDA kernel
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)

# Method 3: Custom autograd with C++
class MyCustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return my_ext.forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        return my_ext.backward(grad_output)
```

> **Interview Tip:** Custom C++/CUDA extensions are used for performance-critical operations not available in PyTorch (e.g., custom attention mechanisms, specialized loss functions). Start with the JIT compilation approach (`load()`) for prototyping, then package with `setup.py` for distribution.

---

## Question 46
- [ ] Done

**How do you ensure reproducibility of experiments when using PyTorch?**

**Answer:**

Ensuring reproducible results in PyTorch requires controlling all sources of randomness.

| Source | Fix |
|--------|-----|
| Python random | `random.seed(42)` |
| NumPy random | `np.random.seed(42)` |
| PyTorch CPU | `torch.manual_seed(42)` |
| PyTorch GPU | `torch.cuda.manual_seed_all(42)` |
| cuDNN | `torch.backends.cudnn.deterministic = True` |
| DataLoader | Set `worker_init_fn` and `generator` |

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)           # For multi-GPU
    torch.backends.cudnn.deterministic = True   # Deterministic cuDNN
    torch.backends.cudnn.benchmark = False      # Disable auto-tuner

set_seed(42)

# DataLoader reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g
)

# Save and load complete training state for resumability
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'rng_state': torch.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state_all()
}
torch.save(checkpoint, 'checkpoint.pt')
```

> **Interview Tip:** Full reproducibility requires seeding ALL random sources + `deterministic=True`. Note that `cudnn.deterministic=True` may slow training by 10-20%. For benchmarking speed (not reproducibility), set `cudnn.benchmark=True` instead.

---

## Question 47
- [ ] Done

**Portray how PyTorch Lightning can simplify the standard PyTorch workflow**

**Answer:**

**PyTorch Lightning** is a high-level framework that organizes PyTorch code into a structured format, eliminating boilerplate while keeping full flexibility.

| Feature | Vanilla PyTorch | PyTorch Lightning |
|---------|----------------|-------------------|
| Training loop | Manual | Automatic |
| Multi-GPU | Manual DDP setup | `Trainer(devices=4)` |
| Mixed precision | Manual AMP | `Trainer(precision='16-mixed')` |
| Logging | Manual | Built-in (TensorBoard, W&B) |
| Checkpointing | Manual | Automatic |
| Early stopping | Custom class | Built-in callback |

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class LitModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        acc = (self(x).argmax(1) == y).float().mean()
        self.log_dict({'val_loss': loss, 'val_acc': acc}, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]

# Train with all the bells and whistles
trainer = pl.Trainer(
    max_epochs=50,
    accelerator='gpu',
    devices=2,                         # Multi-GPU
    precision='16-mixed',              # Mixed precision
    callbacks=[
        pl.callbacks.EarlyStopping(monitor='val_loss', patience=5),
        pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=3)
    ],
    logger=pl.loggers.TensorBoardLogger('logs/')
)

model = LitModel(784, 256, 10)
trainer.fit(model, train_loader, val_loader)
```

> **Interview Tip:** PyTorch Lightning reduces boilerplate by 40-50% while adding multi-GPU, mixed precision, and logging for free. The `LightningModule` organizes code into `training_step`, `validation_step`, and `configure_optimizers`. Use it for production; use vanilla PyTorch for learning fundamentals.

---
