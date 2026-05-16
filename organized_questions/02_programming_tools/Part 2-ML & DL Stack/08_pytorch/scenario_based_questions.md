# PyTorch Interview Questions - Scenario Based Questions

## Question 1
- [ ] Done

**Your PyTorch model is running out of GPU memory. How do you diagnose and fix this?**

### Answer

### Diagnosis Steps

| Step | Command/Action |
|------|----------------|
| Check memory | `torch.cuda.memory_allocated()` |
| Memory summary | `torch.cuda.memory_summary()` |
| Find large tensors | Profile with hooks |

### Solutions

| Solution | Description |
|----------|-------------|
| Reduce batch size | Simplest fix |
| Gradient checkpointing | Trade compute for memory |
| Mixed precision | Use FP16 |
| Clear cache | `torch.cuda.empty_cache()` |
| Gradient accumulation | Simulate larger batches |

### Python Code Example
```python
import torch
import torch.nn as nn

# 1. Monitor memory usage
def print_memory():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# 2. Gradient checkpointing
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.layer3 = nn.Linear(1000, 100)
    
    def forward(self, x):
        # Use checkpointing for memory-heavy layers
        x = checkpoint(self.layer1, x, use_reentrant=False)
        x = checkpoint(self.layer2, x, use_reentrant=False)
        x = self.layer3(x)
        return x

# 3. Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch_x, batch_y in train_loader:
    optimizer.zero_grad()
    
    with autocast():  # FP16 where safe
        output = model(batch_x.cuda())
        loss = criterion(output, batch_y.cuda())
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 4. Gradient accumulation
accumulation_steps = 4
optimizer.zero_grad()

for i, (batch_x, batch_y) in enumerate(train_loader):
    output = model(batch_x.cuda())
    loss = criterion(output, batch_y.cuda()) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 5. Delete intermediate tensors
del output, loss
torch.cuda.empty_cache()
```

---

## Question 2
- [ ] Done

**Your model training is very slow. How do you identify bottlenecks and speed it up?**

### Answer

### Common Bottlenecks

| Bottleneck | Solution |
|------------|----------|
| Data loading | Increase `num_workers`, use `pin_memory` |
| Small batches | Use larger batch size |
| CPU-GPU transfer | Keep data on GPU |
| Model size | Use mixed precision |

### Python Code Example
```python
import torch
from torch.utils.data import DataLoader
import time

# 1. Profile data loading
class TimedDataLoader:
    def __init__(self, loader):
        self.loader = loader
        self.load_time = 0
    
    def __iter__(self):
        for batch in self.loader:
            start = time.time()
            yield batch
            self.load_time += time.time() - start

# 2. Optimize DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=64,         # Larger batch size
    num_workers=8,         # Parallel loading
    pin_memory=True,       # Faster GPU transfer
    prefetch_factor=2,     # Prefetch batches
    persistent_workers=True  # Keep workers alive
)

# 3. Profile with PyTorch Profiler
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    for batch_x, batch_y in train_loader:
        with record_function("forward"):
            output = model(batch_x.cuda())
        with record_function("loss"):
            loss = criterion(output, batch_y.cuda())
        with record_function("backward"):
            loss.backward()
        break

print(prof.key_averages().table(sort_by="cuda_time_total"))

# 4. Use torch.compile (PyTorch 2.0+)
model = torch.compile(model)

# 5. Disable gradient sync during accumulation (DDP)
with model.no_sync():
    loss.backward()  # No sync until we actually update

# 6. Use channels_last memory format for CNNs
model = model.to(memory_format=torch.channels_last)
input = input.to(memory_format=torch.channels_last)

# 7. Benchmark different configurations
import torch.utils.benchmark as benchmark

timer = benchmark.Timer(
    stmt='model(x)',
    globals={'model': model, 'x': torch.randn(32, 3, 224, 224).cuda()}
)
print(timer.timeit(100))
```

---

## Question 3
- [ ] Done

**Your model achieves good training accuracy but poor validation accuracy. How do you handle this?**

### Answer

### Diagnosis

| Check | Purpose |
|-------|---------|
| Learning curves | Visualize train vs val loss |
| Gap size | Measure overfitting severity |
| Data distribution | Verify train/val similarity |

### Solutions

| Technique | Implementation |
|-----------|----------------|
| Regularization | Dropout, weight decay |
| Data augmentation | More training variety |
| Early stopping | Stop before overfitting |
| Reduce model size | Simpler architecture |

### Python Code Example
```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Track and visualize metrics
class MetricTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.train_losses, label='Train')
        ax1.plot(self.val_losses, label='Val')
        ax1.set_title('Loss')
        ax1.legend()
        
        ax2.plot(self.train_accs, label='Train')
        ax2.plot(self.val_accs, label='Val')
        ax2.set_title('Accuracy')
        ax2.legend()
        plt.show()

# 2. Add regularization
class RegularizedModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)

# 3. Weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 4. Early stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# 5. Data augmentation
from torchvision import transforms

augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.RandomErasing(p=0.1),
])

# 6. Label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## Question 4
- [ ] Done

**You need to deploy a PyTorch model to production. What's your approach?**

### Answer

### Deployment Options

| Option | Use Case |
|--------|----------|
| TorchServe | Full-featured serving |
| TorchScript | Portable, no Python |
| ONNX | Cross-platform |
| FastAPI | Custom REST API |

### Python Code Example
```python
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
model.eval()

# 1. Export to TorchScript
example_input = torch.randn(1, 10)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('model_traced.pt')

# 2. Export to ONNX
torch.onnx.export(
    model, example_input, 'model.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# 3. FastAPI deployment
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Load model once at startup
loaded_model = torch.jit.load('model_traced.pt')
loaded_model.eval()

class PredictionInput(BaseModel):
    features: list

class PredictionOutput(BaseModel):
    prediction: float

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    tensor_input = torch.tensor(input_data.features, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        prediction = loaded_model(tensor_input)
    
    return PredictionOutput(prediction=prediction.item())

# 4. TorchServe handler (save as handler.py)
"""
from ts.torch_handler.base_handler import BaseHandler
import torch

class ModelHandler(BaseHandler):
    def preprocess(self, data):
        return torch.tensor(data[0]['body']['features']).float().unsqueeze(0)
    
    def inference(self, model_input):
        with torch.no_grad():
            return self.model(model_input)
    
    def postprocess(self, inference_output):
        return [{'prediction': inference_output.item()}]
"""

# 5. Quantize for faster CPU inference
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
torch.jit.save(torch.jit.script(quantized_model), 'quantized_model.pt')
```

---

## Question 5
- [ ] Done

**Your model works on your machine but gives different results in production. How do you debug?**

### Answer

### Potential Causes

| Cause | Solution |
|-------|----------|
| Random seeds | Set all seeds |
| Model mode | Ensure `eval()` mode |
| Data preprocessing | Match exact pipeline |
| Numerical precision | Check dtypes |
| Environment | Same PyTorch version |

### Python Code Example
```python
import torch
import torch.nn as nn
import numpy as np
import random
import os

# 1. Set all random seeds
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 2. Ensure model is in eval mode
model.eval()

# 3. Create reproducible inference function
def reproducible_inference(model, input_tensor, seed=42):
    set_seed(seed)
    model.eval()
    
    with torch.no_grad():
        output = model(input_tensor)
    
    return output

# 4. Compare outputs
def compare_outputs(local_output, prod_output, rtol=1e-5, atol=1e-8):
    local_np = local_output.numpy()
    prod_np = np.array(prod_output)
    
    if np.allclose(local_np, prod_np, rtol=rtol, atol=atol):
        print("Outputs match!")
    else:
        diff = np.abs(local_np - prod_np)
        print(f"Max difference: {diff.max()}")
        print(f"Mean difference: {diff.mean()}")

# 5. Save preprocessing pipeline
import json

preprocessing_config = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'input_size': [224, 224],
    'dtype': 'float32'
}

with open('preprocessing_config.json', 'w') as f:
    json.dump(preprocessing_config, f)

# 6. Version tracking
def log_environment():
    import platform
    info = {
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'python_version': platform.python_version(),
        'cudnn_version': torch.backends.cudnn.version()
    }
    return info

# 7. Input validation
def validate_input(input_tensor, expected_shape, expected_dtype):
    assert input_tensor.shape == expected_shape, f"Shape mismatch: {input_tensor.shape} vs {expected_shape}"
    assert input_tensor.dtype == expected_dtype, f"Dtype mismatch: {input_tensor.dtype} vs {expected_dtype}"
    assert not torch.isnan(input_tensor).any(), "Input contains NaN"
    assert not torch.isinf(input_tensor).any(), "Input contains Inf"
```

---

## Question 6
- [ ] Done

**You need to train on a dataset that doesn't fit in memory. How do you handle this?**

### Answer

### Strategies

| Strategy | Description |
|----------|-------------|
| Custom Dataset | Load data on-demand |
| Memory mapping | `numpy.memmap` |
| HDF5/Zarr | Chunked data formats |
| Streaming | Load batches from disk |

### Python Code Example
```python
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import h5py

# 1. Lazy loading Dataset
class LazyDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load single sample from disk
        data = np.load(self.file_paths[idx])
        x = torch.tensor(data['x'], dtype=torch.float32)
        y = torch.tensor(data['y'], dtype=torch.long)
        return x, y

# 2. Memory-mapped NumPy
class MemmapDataset(Dataset):
    def __init__(self, data_path, labels_path):
        # Load metadata only
        self.data = np.memmap(data_path, dtype='float32', mode='r').reshape(-1, 100)
        self.labels = np.load(labels_path)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx].copy(), dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# 3. HDF5 Dataset
class HDF5Dataset(Dataset):
    def __init__(self, h5_path, x_key='features', y_key='labels'):
        self.h5_path = h5_path
        self.x_key = x_key
        self.y_key = y_key
        
        with h5py.File(h5_path, 'r') as f:
            self.length = len(f[x_key])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            x = torch.tensor(f[self.x_key][idx], dtype=torch.float32)
            y = torch.tensor(f[self.y_key][idx], dtype=torch.long)
        return x, y

# 4. Streaming/Iterable Dataset
class StreamingDataset(IterableDataset):
    def __init__(self, file_pattern, chunk_size=1000):
        self.file_pattern = file_pattern
        self.chunk_size = chunk_size
    
    def __iter__(self):
        import glob
        for file_path in glob.glob(self.file_pattern):
            data = np.load(file_path)
            for i in range(0, len(data['x']), self.chunk_size):
                for j in range(min(self.chunk_size, len(data['x']) - i)):
                    yield (
                        torch.tensor(data['x'][i+j], dtype=torch.float32),
                        torch.tensor(data['y'][i+j], dtype=torch.long)
                    )

# 5. WebDataset for large-scale data
"""
import webdataset as wds

dataset = wds.WebDataset("data/train-{000..999}.tar")
    .shuffle(1000)
    .decode("pil")
    .to_tuple("input.png", "output.cls")
    .map_tuple(transform, lambda x: x)
    .batched(32)
"""

# Usage
loader = DataLoader(
    LazyDataset(file_paths),
    batch_size=32,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=True
)
```

---

## Question 7
- [ ] Done

**You notice your gradients are exploding or vanishing. How do you diagnose and fix this?**

### Answer

### Diagnosis

| Symptom | Issue |
|---------|-------|
| Loss = NaN | Exploding gradients |
| Loss stuck | Vanishing gradients |
| Large grad norms | Unstable training |

### Solutions

| Fix | When to Use |
|-----|-------------|
| Gradient clipping | Exploding gradients |
| Residual connections | Deep networks |
| Better initialization | All networks |
| Batch normalization | General stability |

### Python Code Example
```python
import torch
import torch.nn as nn

# 1. Monitor gradient norms
def get_gradient_norms(model):
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm().item()
    return norms

def train_step_with_monitoring(model, x, y, optimizer, criterion):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    
    # Check for NaN loss
    if torch.isnan(loss):
        print("NaN loss detected!")
        return None
    
    loss.backward()
    
    # Monitor gradients
    norms = get_gradient_norms(model)
    max_norm = max(norms.values())
    print(f"Max gradient norm: {max_norm:.4f}")
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    return loss.item()

# 2. Better weight initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

model.apply(init_weights)

# 3. Residual connections
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
    
    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return torch.relu(x + residual)

# 4. Gradient scaling for mixed precision
from torch.cuda.amp import GradScaler

scaler = GradScaler()

for x, y in train_loader:
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast():
        output = model(x)
        loss = criterion(output, y)
    
    # Scale gradients to prevent underflow in FP16
    scaler.scale(loss).backward()
    
    # Unscale before clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    scaler.step(optimizer)
    scaler.update()

# 5. Learning rate warmup
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * self.step_count / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
```

---

## Question 8
- [ ] Done

**You need to implement a custom attention mechanism for your model. How do you approach this?**

### Answer

### Python Code Example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1. Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch, heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

# 2. Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        output, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(output), attention_weights

# 3. Cross-Attention
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, context, mask=None):
        # x attends to context
        attended, _ = self.mha(x, context, context, mask)
        return self.norm(x + attended)

# 4. Additive (Bahdanau) Attention
class AdditiveAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.W_q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(key_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, query, keys, mask=None):
        # query: (batch, query_dim)
        # keys: (batch, seq_len, key_dim)
        
        query = query.unsqueeze(1)  # (batch, 1, query_dim)
        scores = self.v(torch.tanh(self.W_q(query) + self.W_k(keys)))  # (batch, seq_len, 1)
        scores = scores.squeeze(-1)  # (batch, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        
        return context, weights

# Usage
mha = MultiHeadAttention(d_model=256, n_heads=8)
x = torch.randn(32, 50, 256)  # (batch, seq_len, d_model)
output, weights = mha(x, x, x)
print(f"Output: {output.shape}, Weights: {weights.shape}")
```

---

## Question 9
- [ ] Done

**You need to fine-tune a large pre-trained model but have limited GPU memory. What techniques do you use?**

### Answer

### Memory-Efficient Techniques

| Technique | Memory Reduction |
|-----------|------------------|
| LoRA | Train only adapters |
| Gradient checkpointing | Trade compute for memory |
| Mixed precision | 50% reduction |
| Freeze most layers | Train only head |

### Python Code Example
```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# 1. Freeze most parameters
def freeze_base_model(model, unfreeze_last_n=2):
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last n layers
    for layer in model.encoder.layer[-unfreeze_last_n:]:
        for param in layer.parameters():
            param.requires_grad = True

# 2. LoRA (Low-Rank Adaptation)
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original = original_layer
        self.original.weight.requires_grad = False
        
        d_in = original_layer.in_features
        d_out = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) / rank)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.scaling = alpha / rank
    
    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_out + lora_out

def apply_lora(model, rank=8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'attention' in name:
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, child_name, LoRALayer(module, rank))

# 3. Gradient checkpointing
from torch.utils.checkpoint import checkpoint

class MemoryEfficientEncoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return x

# 4. Mixed precision + gradient accumulation
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
accumulation_steps = 8

for i, (batch_x, batch_y) in enumerate(train_loader):
    with autocast():
        output = model(batch_x.cuda())
        loss = criterion(output, batch_y.cuda()) / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

# 5. 8-bit optimizers (with bitsandbytes)
"""
import bitsandbytes as bnb

optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)
"""

# 6. Parameter-efficient fine-tuning summary
def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

count_trainable_params(model)
```

---

## Question 10
- [ ] Done

**You need to implement a real-time inference system with PyTorch. What's your approach?**

### Answer

### Optimization Strategies

| Strategy | Benefit |
|----------|---------|
| TorchScript | Remove Python overhead |
| Batching | Better GPU utilization |
| Quantization | 2-4x speedup |
| CUDA streams | Async processing |

### Python Code Example
```python
import torch
import torch.nn as nn
from collections import deque
import threading
import time

# 1. Optimized inference model
class OptimizedModel:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        # Load TorchScript model
        self.model = torch.jit.load(model_path).to(device)
        self.model.eval()
        
        # Warmup
        dummy_input = torch.randn(1, 100).to(device)
        for _ in range(10):
            self.model(dummy_input)
    
    @torch.no_grad()
    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        return self.model(x).cpu().numpy()

# 2. Dynamic batching for high throughput
class DynamicBatcher:
    def __init__(self, model, max_batch_size=32, max_wait_ms=10):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        
        self.queue = deque()
        self.lock = threading.Lock()
        self.batch_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.batch_thread.start()
    
    def add_request(self, input_data, callback):
        with self.lock:
            self.queue.append((input_data, callback, time.time()))
    
    def _process_batches(self):
        while True:
            batch_inputs = []
            callbacks = []
            
            with self.lock:
                while self.queue and len(batch_inputs) < self.max_batch_size:
                    input_data, callback, timestamp = self.queue.popleft()
                    batch_inputs.append(input_data)
                    callbacks.append(callback)
            
            if batch_inputs:
                # Process batch
                batch_tensor = torch.stack(batch_inputs)
                outputs = self.model.predict(batch_tensor)
                
                # Return results
                for i, callback in enumerate(callbacks):
                    callback(outputs[i])
            
            time.sleep(self.max_wait_ms / 1000)

# 3. CUDA streams for async processing
class AsyncInference:
    def __init__(self, model, num_streams=4):
        self.model = model
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.current_stream = 0
    
    @torch.no_grad()
    def predict_async(self, inputs):
        stream = self.streams[self.current_stream]
        self.current_stream = (self.current_stream + 1) % len(self.streams)
        
        with torch.cuda.stream(stream):
            outputs = self.model(inputs.cuda())
        
        return outputs, stream
    
    def wait_for_result(self, output, stream):
        stream.synchronize()
        return output.cpu()

# 4. Quantized model for CPU deployment
def create_quantized_model(model):
    model.eval()
    
    # Dynamic quantization (easiest)
    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    
    return quantized

# 5. Complete inference service
class InferenceService:
    def __init__(self, model_path):
        # Load and optimize model
        self.model = torch.jit.load(model_path)
        self.model.eval()
        
        # Move to GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        
        # Pre-allocate memory
        self.input_buffer = torch.empty(32, 100, device=self.device)
    
    @torch.inference_mode()  # Faster than no_grad
    def predict(self, x, batch_size=None):
        if batch_size:
            x = x.view(batch_size, -1)
        
        x = x.to(self.device, non_blocking=True)
        output = self.model(x)
        
        return output.cpu().numpy()
    
    def benchmark(self, input_shape, num_runs=1000):
        x = torch.randn(*input_shape, device=self.device)
        
        # Warmup
        for _ in range(100):
            _ = self.model(x)
        
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(num_runs):
            _ = self.model(x)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"Avg latency: {elapsed/num_runs*1000:.2f} ms")
        print(f"Throughput: {num_runs/elapsed:.0f} inferences/sec")
```

## Question 11
- [ ] Done

**How would you handle imbalanced classes when training a classification model in PyTorch?**

### Answer

| Strategy | Implementation | Best For |
|----------|---------------|----------|
| **Weighted loss** | `CrossEntropyLoss(weight=class_weights)` | Most cases, simplest |
| **Oversampling** | `WeightedRandomSampler` | Training data balancing |
| **Focal loss** | Custom loss function | Severe imbalance |
| **Class-balanced batches** | Custom sampler | Ensures all classes seen |

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# Method 1: Weighted CrossEntropyLoss
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
class_weights = torch.FloatTensor(class_weights / class_weights.sum()).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Method 2: WeightedRandomSampler (oversample minority in each batch)
sample_weights = [class_weights[label] for label in y_train]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                 replacement=True)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

# Method 3: Focal Loss (reduces easy example contribution)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
```

> **Interview Tip:** Start with **weighted loss** (simplest). Use `WeightedRandomSampler` for batch-level balancing. For extreme imbalance, combine weighted loss + oversampling. Always use **F1-score**, not accuracy, for evaluation.

---

## Question 12
- [ ] Done

**How can PyTorch be utilized for real-time inference , and what concerns would you have in such a setting?**

### Answer

| Concern | Solution |
|---------|----------|
| **Latency** | Model optimization, TorchScript, ONNX Runtime |
| **Throughput** | Batching, GPU inference, async processing |
| **Memory** | Quantization, pruning, model distillation |
| **Deployment** | TorchServe, ONNX Runtime, TensorRT |

```python
import torch

# 1. Optimize for inference
model.eval()

# 2. TorchScript (JIT compilation)
scripted_model = torch.jit.script(model)  # or torch.jit.trace(model, sample_input)
scripted_model.save("model_scripted.pt")

# 3. Quantization (reduce model size and latency)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 4. ONNX export for cross-platform inference
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx",
                   input_names=['input'], output_names=['output'],
                   dynamic_axes={'input': {0: 'batch_size'}})

# 5. Inference with no gradient computation
with torch.no_grad():
    torch.cuda.synchronize()  # Ensure accurate timing
    output = model(input_tensor.to(device))

# 6. Half precision (FP16) for faster GPU inference
model.half()
input_tensor = input_tensor.half()
```

> **Interview Tip:** For real-time inference, prioritize: (1) `torch.no_grad()` + `model.eval()`, (2) TorchScript or ONNX export, (3) quantization, (4) batching requests. Mention **TensorRT** for NVIDIA GPU optimization and **ONNX Runtime** for cross-platform deployment.

---

## Question 13
- [ ] Done

**Discuss a scenario where you would need to convert a PyTorch model to ONNX format**

### Answer

### When to Use ONNX

| Scenario | Reason |
|----------|--------|
| **Cross-framework** | Run PyTorch model in TensorFlow, CoreML, etc. |
| **Production inference** | ONNX Runtime is highly optimized |
| **Edge deployment** | Convert to TFLite, CoreML via ONNX |
| **Hardware acceleration** | TensorRT, OpenVINO support ONNX |

```python
import torch
import torch.onnx

# 1. Export PyTorch model to ONNX
model.eval()
dummy_input = torch.randn(1, 3, 224, 224).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# 2. Verify ONNX model
import onnx
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# 3. Run inference with ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("model.onnx", providers=['CUDAExecutionProvider'])
result = session.run(None, {'input': input_numpy})

# 4. Compare outputs
torch_out = model(dummy_input).detach().cpu().numpy()
onnx_out = session.run(None, {'input': dummy_input.cpu().numpy()})[0]
assert np.allclose(torch_out, onnx_out, atol=1e-5)
```

> **Interview Tip:** Always verify ONNX output matches PyTorch output with `np.allclose()`. Use `dynamic_axes` for variable batch sizes. ONNX Runtime typically provides **2-3x speedup** over native PyTorch inference.

---

## Question 14
- [ ] Done

**Propose a method for deploying a PyTorch model as a REST API service**

### Answer

| Approach | Tool | Best For |
|----------|------|----------|
| **FastAPI** | Python web framework | Custom APIs, flexibility |
| **TorchServe** | PyTorch's official serving | Production, scaling |
| **Flask** | Lightweight Python framework | Quick prototypes |
| **Docker + K8s** | Containerization | Production scaling |

```python
# FastAPI deployment (recommended)
from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()

# Load model once at startup
model = torch.jit.load("model_scripted.pt")
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_idx = torch.topk(probabilities, 5)

    return {
        "predictions": [
            {"class": idx.item(), "probability": prob.item()}
            for prob, idx in zip(top5_prob, top5_idx)
        ]
    }

# Run: uvicorn app:app --host 0.0.0.0 --port 8000
```

### Production Considerations

| Concern | Solution |
|---------|----------|
| Scalability | Load balancer + multiple workers |
| GPU sharing | Batching requests, NVIDIA Triton |
| Monitoring | Prometheus + Grafana |
| Model versioning | MLflow, DVC |

> **Interview Tip:** Use **FastAPI** over Flask for async support and auto-generated docs. Always use **TorchScript** (`torch.jit`) for deployment. For high-throughput, consider **TorchServe** or **NVIDIA Triton** for GPU request batching.

---

## Question 15
- [ ] Done

**Describe your approach to fine-tuning a pre-trained model in PyTorch for a new task**

### Answer

### Step-by-Step Approach

| Phase | Action | Learning Rate |
|-------|--------|---------------|
| 1. Feature extraction | Freeze base, train head | Normal (1e-3) |
| 2. Fine-tuning | Unfreeze top layers | Low (1e-5) |
| 3. Full fine-tuning | Unfreeze all (optional) | Very low (1e-6) |

```python
import torch
import torch.nn as nn
import torchvision.models as models

# 1. Load pre-trained model
model = models.resnet50(weights='IMAGENET1K_V2')

# 2. Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# 3. Replace classification head
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, num_classes)
)

# 4. Phase 1: Train head only
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
train(model, train_loader, optimizer, epochs=5)

# 5. Phase 2: Fine-tune top layers
for param in model.layer4.parameters():
    param.requires_grad = True

# Use different learning rates for different parts
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-4}
])
train(model, train_loader, optimizer, epochs=10)

# Tip: Use cosine annealing scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
```

> **Interview Tip:** Use **discriminative learning rates** (lower LR for earlier layers, higher for head). Always train the head first with frozen base, then gradually unfreeze. Monitor validation loss to avoid catastrophic forgetting.

---

## Question 16
- [ ] Done

**Discuss a project where PyTorch played a key role in developing a machine learning solution**

### Answer

### Example: Medical Image Classification System

| Phase | Action | Tools |
|-------|--------|-------|
| **Problem** | Detect pneumonia from chest X-rays | Medical domain knowledge |
| **Data** | NIH Chest X-ray dataset (112K images) | PyTorch DataLoader |
| **Model** | DenseNet-121 with transfer learning | torchvision.models |
| **Training** | Mixed precision, cosine annealing | PyTorch AMP |
| **Evaluation** | AUC-ROC, sensitivity, specificity | sklearn.metrics |
| **Deployment** | TorchServe + Docker on AWS | Production infrastructure |

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler

# 1. Custom dataset with medical image preprocessing
class ChestXrayDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(image), self.labels[idx]

    def __len__(self):
        return len(self.image_paths)

# 2. Transfer learning with DenseNet
model = models.densenet121(weights='IMAGENET1K_V1')
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier.in_features, 14)  # 14 pathology classes
)

# 3. Handle class imbalance
pos_weights = torch.FloatTensor(compute_class_weights(train_labels)).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

# 4. Mixed precision training
scaler = GradScaler()
for epoch in range(epochs):
    for images, labels in train_loader:
        with autocast():
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

# 5. Results: 0.92 AUC-ROC, deployed via TorchServe
```

### Lessons Learned
- **Data quality** matters more than model architecture
- **Transfer learning** is essential when domain data is limited
- **Class imbalance** handling is critical in medical datasets
- **Interpretability** (Grad-CAM) is required for clinical acceptance

> **Interview Tip:** Structure your case study as: **Problem -> Data -> Model -> Training -> Evaluation -> Deployment -> Lessons Learned**. Mention specific metrics and real challenges you encountered.

---

