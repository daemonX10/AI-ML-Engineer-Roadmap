# Neural Networks Interview Questions - Theory Questions

## Question 1: What is a neural network, and how does it resemble human brain functionality?
- [ ] Done

### Definition
A neural network is a computational model inspired by biological neurons, consisting of interconnected nodes (neurons) organized in layers that learn patterns from data through weighted connections and non-linear transformations.

### Core Concepts

| Component | Biological Analog | Function |
|-----------|------------------|----------|
| **Neuron (Node)** | Brain neuron | Processes input, produces output |
| **Weights** | Synaptic strength | Controls signal importance |
| **Activation Function** | Firing threshold | Determines if neuron activates |
| **Layers** | Neural pathways | Hierarchical processing |

### Mathematical Formulation
For a single neuron:
$$z = \sum_{i=1}^{n} w_i x_i + b$$
$$a = \sigma(z)$$

Where $w_i$ are weights, $b$ is bias, and $\sigma$ is activation function.

### Brain-Neural Network Comparison

| Brain | Neural Network |
|-------|----------------|
| ~86 billion neurons | Hundreds to millions of nodes |
| Electrical signals | Numerical values |
| Synapses strengthen with use | Weights updated via learning |
| Parallel processing | Can be parallelized on GPU |

### Practical Relevance
- Image recognition (mimics visual cortex)
- Speech processing (mimics auditory processing)
- Pattern recognition tasks

---

## Question 2: Describe the architecture of a multi-layer perceptron (MLP)
- [ ] Done

### Definition
An MLP is a fully-connected feedforward neural network with at least three layers: input layer, one or more hidden layers, and an output layer, where each neuron connects to all neurons in adjacent layers.

### Architecture Components

| Layer | Role | Neurons |
|-------|------|---------|
| **Input Layer** | Receives raw features | One per feature |
| **Hidden Layer(s)** | Learn representations | Hyperparameter |
| **Output Layer** | Produces predictions | Task-dependent |

### Mathematical Formulation
Layer-wise computation:
$$h^{(1)} = \sigma(W^{(1)}x + b^{(1)})$$
$$h^{(2)} = \sigma(W^{(2)}h^{(1)} + b^{(2)})$$
$$\hat{y} = W^{(L)}h^{(L-1)} + b^{(L)}$$

### Python Code Example
```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)
```

### Key Characteristics
- Universal approximator (can learn any continuous function)
- Fully connected: O(n²) parameters per layer pair
- Prone to overfitting without regularization

---

## Question 3: How does feedforward neural network differ from recurrent neural networks (RNNs)?
- [ ] Done

### Definition
**Feedforward networks** process information in one direction (input → output) with no cycles. **RNNs** have recurrent connections that create internal memory, allowing them to process sequential data by passing information across time steps.

### Core Differences

| Aspect | Feedforward | RNN |
|--------|-------------|-----|
| **Data Flow** | One direction only | Has feedback loops |
| **Memory** | No internal state | Maintains hidden state |
| **Input** | Fixed-size vectors | Variable-length sequences |
| **Use Case** | Tabular, images (CNN) | Text, time series, speech |

### Mathematical Formulation

**Feedforward:**
$$h = \sigma(Wx + b)$$

**RNN (at time t):**
$$h_t = \sigma(W_x x_t + W_h h_{t-1} + b)$$

### Visual Representation
```
Feedforward:   Input → Hidden → Output

RNN:           Input₁ → Hidden₁ → Output₁
                           ↓
               Input₂ → Hidden₂ → Output₂
                           ↓
               Input₃ → Hidden₃ → Output₃
```

### When to Use
- **Feedforward**: Classification, regression, fixed-size input
- **RNN**: Sequence modeling, language, time series

---

## Question 4: What is backpropagation, and why is it important in neural networks?
- [ ] Done

### Definition
Backpropagation is the algorithm used to compute gradients of the loss function with respect to all weights in a neural network by applying the chain rule, enabling gradient descent optimization.

### Core Concepts
- Uses **chain rule** to propagate error backwards
- Computes gradients efficiently in one backward pass
- Enables training of deep networks

### Algorithm Steps
1. **Forward Pass**: Compute predictions layer by layer
2. **Compute Loss**: Calculate error at output
3. **Backward Pass**: Propagate gradients from output to input
4. **Update Weights**: $w = w - \eta \cdot \nabla_w L$

### Mathematical Formulation (Chain Rule)
For weight $w$ in layer $l$:
$$\frac{\partial L}{\partial w^{(l)}} = \frac{\partial L}{\partial a^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial a^{(L-1)}} \cdots \frac{\partial a^{(l)}}{\partial w^{(l)}}$$

### Why It's Important
- **Enables deep learning**: Without it, couldn't train multi-layer networks
- **Computationally efficient**: One forward + one backward pass computes all gradients
- **Foundation of modern AI**: Powers all neural network training

---

## Question 5: Explain the role of an activation function. Give examples of common activation functions
- [ ] Done

### Definition
Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Without them, any deep network would collapse to a single linear transformation.

### Why Non-linearity is Essential
- Stack of linear functions = still linear
- Without activation: Network ≡ Linear model
- Non-linearity enables universal approximation

### Common Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| **ReLU** | $\max(0, z)$ | [0, ∞) | Hidden layers (default) |
| **Sigmoid** | $\frac{1}{1+e^{-z}}$ | (0, 1) | Binary output |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | (-1, 1) | Zero-centered |
| **Softmax** | $\frac{e^{z_i}}{\sum e^{z_j}}$ | (0, 1) | Multi-class output |
| **Leaky ReLU** | $\max(0.01z, z)$ | (-∞, ∞) | Avoid dying ReLU |

### Python Code Example
```python
import torch.nn.functional as F

# Common activations
x = torch.randn(10)
relu_out = F.relu(x)
sigmoid_out = torch.sigmoid(x)
softmax_out = F.softmax(x, dim=0)
```

### Selection Guide
- **Hidden layers**: ReLU (fast, works well)
- **Binary classification output**: Sigmoid
- **Multi-class output**: Softmax
- **RNN hidden**: Tanh

---

## Question 6: Describe the concept of deep learning in relation to neural networks
- [ ] Done

### Definition
Deep learning refers to neural networks with multiple hidden layers (typically >2) that can automatically learn hierarchical feature representations from raw data, eliminating the need for manual feature engineering.

### Core Concepts

| Aspect | Description |
|--------|-------------|
| **Depth** | Multiple hidden layers (2+) |
| **Representation Learning** | Learns features automatically |
| **Hierarchical Features** | Simple → Complex patterns |
| **End-to-End** | Raw input → Final output |

### Hierarchical Feature Learning
```
Layer 1: Edges, corners
Layer 2: Textures, patterns
Layer 3: Parts (eyes, wheels)
Layer 4: Objects (faces, cars)
```

### Deep vs Shallow

| Shallow (1 hidden layer) | Deep (2+ hidden layers) |
|--------------------------|-------------------------|
| Manual feature engineering | Automatic feature learning |
| Limited representation power | Hierarchical abstractions |
| Works for simple patterns | Works for complex data (images, text) |

### Why Depth Works
- Exponentially more efficient than width
- Captures compositionality in data
- Each layer builds on previous representations

---

## Question 7: What is a vanishing gradient problem? How does it affect training?
- [ ] Done

### Definition
The vanishing gradient problem occurs when gradients become extremely small as they propagate backward through many layers, causing early layers to learn very slowly or not at all.

### Core Concepts

| Cause | Effect |
|-------|--------|
| Sigmoid/Tanh derivatives < 1 | Gradients multiply and shrink |
| Many layers | Exponential decay |
| Early layers | Near-zero gradients, no learning |

### Mathematical Explanation
With sigmoid (max derivative = 0.25):
$$\frac{\partial L}{\partial w^{(1)}} = \underbrace{0.25 \times 0.25 \times ... \times 0.25}_{n \text{ layers}} \rightarrow 0$$

### Impact on Training
- Early layers don't update
- Network only learns in last few layers
- Training stalls, poor performance

### Solutions

| Solution | How It Helps |
|----------|--------------|
| **ReLU activation** | Gradient = 1 for positive inputs |
| **Skip connections (ResNet)** | Gradients flow directly |
| **Batch Normalization** | Keeps activations in good range |
| **Proper initialization** | Xavier/He initialization |
| **LSTM/GRU** | Gating mechanisms for RNNs |

---

## Question 8: How does the exploding gradient problem occur, and what are the potential solutions?
- [ ] Done

### Definition
The exploding gradient problem occurs when gradients grow exponentially large during backpropagation, causing unstable weight updates, numerical overflow, and divergent training.

### Core Concepts

| Cause | Effect |
|-------|--------|
| Weight matrices with large eigenvalues | Gradients multiply and explode |
| Deep networks or long sequences | Exponential growth |
| Large gradient values | NaN weights, loss → infinity |

### Mathematical Explanation
If weight matrix has eigenvalue > 1:
$$\frac{\partial L}{\partial w^{(1)}} = \underbrace{2 \times 2 \times ... \times 2}_{n \text{ layers}} \rightarrow \infty$$

### Solutions

| Solution | Implementation |
|----------|----------------|
| **Gradient Clipping** | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` |
| **Proper Initialization** | Xavier, He initialization |
| **Batch Normalization** | Normalizes layer inputs |
| **Lower Learning Rate** | Smaller update steps |
| **LSTM/GRU** | Gating controls gradient flow |

### Python Code Example
```python
# Gradient clipping
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## Question 9: Explain the trade-offs between bias and variance
- [ ] Done

### Definition
The bias-variance trade-off is the tension between a model's ability to fit training data (low bias) and its ability to generalize to new data (low variance). Reducing one typically increases the other.

### Core Concepts

| Component | Definition | Effect |
|-----------|------------|--------|
| **Bias** | Error from oversimplified assumptions | Underfitting |
| **Variance** | Sensitivity to training data | Overfitting |
| **Total Error** | Bias² + Variance + Noise | What we minimize |

### In Neural Networks

| Network Property | Bias | Variance |
|------------------|------|----------|
| **Few layers/neurons** | High | Low |
| **Many layers/neurons** | Low | High |
| **With regularization** | Slightly higher | Lower |
| **More training data** | Same | Lower |

### Mathematical Formulation
$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

### Managing the Trade-off
- Start with high-capacity model (low bias)
- Use regularization to control variance
- Add more data to reduce variance without increasing bias

---

## Question 10: What is regularization in neural networks, and why is it used?
- [ ] Done

### Definition
Regularization comprises techniques that constrain neural network complexity to prevent overfitting, improving generalization to unseen data by trading slight increase in training error for better test performance.

### Common Techniques

| Technique | How It Works | Effect |
|-----------|--------------|--------|
| **L2 (Weight Decay)** | Adds $\lambda\sum w^2$ to loss | Shrinks weights |
| **L1** | Adds $\lambda\sum|w|$ to loss | Sparse weights |
| **Dropout** | Randomly zeros neurons | Prevents co-adaptation |
| **Batch Norm** | Normalizes layer inputs | Slight regularization |
| **Early Stopping** | Stop when val loss increases | Prevents overtraining |
| **Data Augmentation** | Artificially increase data | More diverse training |

### Mathematical Formulation
$$L_{regularized} = L_{original} + \lambda \cdot R(w)$$

### Python Code Example
```python
# L2 regularization (weight decay)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)

# Dropout in model
self.dropout = nn.Dropout(p=0.5)
```

### Why It's Used
- Neural networks have high capacity → prone to overfitting
- Regularization constrains effective capacity
- Improves generalization without collecting more data

---

## Question 11: What are dropout layers, and how do they help in preventing overfitting?
- [ ] Done

### Definition
Dropout is a regularization technique that randomly sets a fraction of neurons to zero during each training iteration, preventing neurons from co-adapting and forcing the network to learn redundant representations.

### Core Concepts
- **Training**: Each neuron has probability $p$ of being dropped
- **Inference**: All neurons active, outputs scaled by $(1-p)$
- **Effect**: Like training ensemble of sub-networks

### How It Works
```
Training:          Inference:
[1, 0, 1, 0, 1]   [1, 1, 1, 1, 1] × (1-p)
     ↓                   ↓
 Some zeros        All active, scaled
```

### Mathematical Formulation
During training for each neuron:
$$\hat{a}_i = \begin{cases} 0 & \text{with probability } p \\ a_i & \text{with probability } 1-p \end{cases}$$

### Python Code Example
```python
class ModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout = nn.Dropout(p=0.5)  # 50% dropout
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Applied during training only
        return self.fc2(x)
```

### Why It Prevents Overfitting
- Neurons can't rely on specific other neurons
- Forces learning of robust features
- Acts like training an ensemble

---

## Question 12: What are skip connections and residual blocks in neural networks?
- [ ] Done

### Definition
Skip connections (residual connections) allow gradients and information to bypass one or more layers by adding the input directly to the output, enabling training of very deep networks by solving the vanishing gradient problem.

### Core Concepts

**Residual Block:**
$$y = F(x) + x$$

Where $F(x)$ is the learned residual function.

### Why They Work

| Problem | How Skip Connections Help |
|---------|---------------------------|
| Vanishing gradients | Gradients flow directly through skip |
| Degradation problem | Easy to learn identity mapping |
| Deep network training | Enables training 100+ layer networks |

### Architecture
```
      x
      │
      ▼
┌─────────────┐
│ Conv + ReLU │
└─────────────┘
      │
      ▼
┌─────────────┐
│    Conv     │
└─────────────┘
      │
      │◄──────── x (skip connection)
      ▼
    + Add
      │
      ▼
    ReLU
```

### Python Code Example
```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return F.relu(out)
```

---

## Question 13: Explain how to initialize neural network weights effectively
- [ ] Done

### Definition
Weight initialization sets the starting values for network parameters. Proper initialization ensures signals neither explode nor vanish during forward/backward passes, enabling stable and efficient training.

### Common Methods

| Method | Formula | Use Case |
|--------|---------|----------|
| **Xavier/Glorot** | $w \sim U(-\sqrt{6/(n_{in}+n_{out})}, \sqrt{6/(n_{in}+n_{out})})$ | Sigmoid, Tanh |
| **He (Kaiming)** | $w \sim N(0, \sqrt{2/n_{in}})$ | ReLU |
| **Zero** | $w = 0$ | Only for biases |

### Why It Matters
- **Too small**: Signals shrink, vanishing gradients
- **Too large**: Signals explode, unstable training
- **Just right**: Maintains variance across layers

### Python Code Example
```python
import torch.nn as nn

# Xavier initialization (for tanh/sigmoid)
nn.init.xavier_uniform_(layer.weight)

# He initialization (for ReLU)
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# PyTorch default for Linear layers is already good
model = nn.Linear(100, 50)  # Uses Kaiming uniform by default
```

### Rules of Thumb
- **ReLU/Leaky ReLU**: Use He initialization
- **Sigmoid/Tanh**: Use Xavier initialization
- **Biases**: Initialize to zero

---

## Question 14: Explain the difference between local minima and global minima in neural networks
- [ ] Done

### Definition
The **global minimum** is the point where the loss function achieves its lowest possible value. **Local minima** are points that are lower than all nearby points but not necessarily the lowest overall.

### Core Concepts

| Type | Definition | In Neural Networks |
|------|------------|-------------------|
| **Global Minimum** | Absolute lowest loss | Ideal but rare |
| **Local Minimum** | Lowest in neighborhood | Many exist |
| **Saddle Point** | Min in some directions, max in others | Very common in high-D |

### Visual Representation
```
Loss
  │
  │    local min          global min
  │        ↓                  ↓
  │    ┌──────┐         ┌────────┐
  │   /        \       /          \
  │──/          \─────/            \──
  └─────────────────────────────────→ Parameters
```

### Modern Understanding
- In high dimensions, true local minima are rare
- Most "stuck" points are saddle points
- Good local minima often have similar loss to global minimum
- SGD noise helps escape poor local minima

### Practical Implications
- Don't worry too much about global minimum
- Multiple random restarts can help
- SGD with momentum escapes saddle points
- Generalization matters more than training loss

---

## Question 15: Describe the role of learning rate and learning rate schedules in training
- [ ] Done

### Definition
The **learning rate** controls the step size of parameter updates. **Learning rate schedules** adjust this value during training to achieve faster convergence and better final performance.

### Impact of Learning Rate

| Learning Rate | Effect |
|---------------|--------|
| **Too High** | Oscillates, diverges |
| **Too Low** | Very slow convergence |
| **Optimal** | Fast, stable convergence |

### Common Schedules

| Schedule | Description | Formula |
|----------|-------------|---------|
| **Step Decay** | Reduce by factor every N epochs | $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/N \rfloor}$ |
| **Exponential** | Continuous decay | $\eta_t = \eta_0 \cdot e^{-kt}$ |
| **Cosine Annealing** | Smooth oscillation | $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t\pi}{T}))$ |
| **Warmup** | Start low, increase | Linear increase for first N steps |

### Python Code Example
```python
# Step LR
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine Annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Training loop
for epoch in range(epochs):
    train(...)
    scheduler.step()
```

### Best Practices
- Start with 0.001 for Adam, 0.01 for SGD
- Use warmup for large batch training
- Reduce LR when validation plateaus

---

## Question 16: What are GRUs and LSTMs? What problems do they solve?
- [ ] Done

### Definition
**LSTM** (Long Short-Term Memory) and **GRU** (Gated Recurrent Unit) are RNN architectures with gating mechanisms that control information flow, solving the vanishing gradient problem and enabling learning of long-range dependencies.

### The Problem They Solve
Standard RNNs forget long-term information due to vanishing gradients. LSTM/GRU use gates to selectively remember and forget.

### LSTM Architecture

| Gate | Formula | Purpose |
|------|---------|---------|
| **Forget** | $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ | What to forget |
| **Input** | $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ | What to add |
| **Output** | $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ | What to output |
| **Cell State** | $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ | Long-term memory |

### GRU Architecture (Simplified LSTM)

| Gate | Purpose |
|------|---------|
| **Reset** | How much past info to forget |
| **Update** | How much new info to add |

### Comparison

| Aspect | LSTM | GRU |
|--------|------|-----|
| **Parameters** | More (3 gates + cell) | Fewer (2 gates) |
| **Training Speed** | Slower | Faster |
| **Performance** | Often slightly better | Often comparable |
| **When to Use** | Complex sequences | Shorter sequences, less data |

### Python Code Example
```python
lstm = nn.LSTM(input_size=100, hidden_size=256, num_layers=2, batch_first=True)
gru = nn.GRU(input_size=100, hidden_size=256, num_layers=2, batch_first=True)

output, (h_n, c_n) = lstm(x)  # LSTM returns cell state
output, h_n = gru(x)          # GRU only returns hidden state
```

---

## Question 17: Define and explain the significance of Convolutional Neural Networks (CNNs)
- [ ] Done

### Definition
CNNs are neural networks that use convolutional layers to automatically learn spatial hierarchies of features from grid-like data (images), using local connectivity, weight sharing, and translation equivariance.

### Core Components

| Component | Purpose |
|-----------|---------|
| **Convolutional Layer** | Extract local features using learnable filters |
| **Pooling Layer** | Reduce spatial dimensions, add invariance |
| **Fully Connected** | Combine features for final prediction |

### Key Properties

| Property | Description | Benefit |
|----------|-------------|---------|
| **Local Connectivity** | Each neuron connects to small region | Captures local patterns |
| **Weight Sharing** | Same filter across entire image | Fewer parameters |
| **Translation Equivariance** | Detect features regardless of position | Generalization |

### Mathematical Formulation
Convolution operation:
$$(f * g)(i, j) = \sum_m \sum_n f(m, n) \cdot g(i-m, j-n)$$

### Python Code Example
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        return self.fc(x)
```

### Significance
- Revolutionized computer vision
- State-of-the-art for image tasks
- Foundation for object detection, segmentation

---

## Question 18: What are the common use cases for CNNs in comparison to RNNs?
- [ ] Done

### Definition
**CNNs** excel at spatial pattern recognition in grid-like data (images). **RNNs** excel at sequential pattern recognition where order matters (text, time series).

### Use Case Comparison

| CNNs | RNNs |
|------|------|
| Image classification | Language modeling |
| Object detection | Machine translation |
| Image segmentation | Speech recognition |
| Face recognition | Time series forecasting |
| Medical imaging | Sentiment analysis |
| Video (frame-wise) | Music generation |

### When to Choose

| Choose CNN | Choose RNN |
|------------|------------|
| Grid structure data | Sequential data |
| Local patterns important | Order matters |
| Translation invariance needed | Variable length inputs |
| Images, audio spectrograms | Text, time series |

### Hybrid Approaches
- **CNN + RNN**: Video captioning (CNN extracts features, RNN generates text)
- **1D CNN**: Text classification (faster than RNN)
- **Attention (Transformer)**: Replacing RNNs for many sequence tasks

---

## Question 19: Explain what deconvolutional layers are and their role in neural networks
- [ ] Done

### Definition
Deconvolutional layers (transposed convolutions) perform the inverse of convolution, upsampling feature maps to higher spatial resolutions. They're used in decoder networks for tasks like image segmentation and generation.

### Core Concepts

| Term | Description |
|------|-------------|
| **Transposed Convolution** | Upsamples by inserting zeros, then convolving |
| **Deconvolution** | Common (but technically incorrect) name |
| **Upsampling** | Increasing spatial dimensions |

### Use Cases

| Application | Role |
|-------------|------|
| **Image Segmentation** | Upsample to pixel-level predictions |
| **Image Generation (GANs)** | Generate high-resolution images from latent |
| **Autoencoders** | Reconstruct input in decoder |
| **Super-resolution** | Increase image resolution |

### Python Code Example
```python
# Transposed convolution (deconv)
deconv = nn.ConvTranspose2d(
    in_channels=64, 
    out_channels=32, 
    kernel_size=4, 
    stride=2, 
    padding=1
)
# Input: (batch, 64, 16, 16) → Output: (batch, 32, 32, 32)
```

### Comparison with Other Upsampling

| Method | Description |
|--------|-------------|
| **Transposed Conv** | Learnable upsampling |
| **Bilinear Interpolation** | Fixed, no learning |
| **Nearest Neighbor + Conv** | Often produces fewer artifacts |

---

## Question 20: What is attention mechanism in neural networks? Give an example of its application
- [ ] Done

### Definition
Attention mechanisms allow neural networks to dynamically focus on relevant parts of the input when producing output, computing weighted combinations where weights indicate importance for the current task.

### Core Concept
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| Component | Role |
|-----------|------|
| **Query (Q)** | What we're looking for |
| **Key (K)** | What we're comparing against |
| **Value (V)** | What we retrieve |
| **Softmax** | Converts scores to probabilities |

### Types of Attention

| Type | Description |
|------|-------------|
| **Self-Attention** | Q, K, V from same sequence |
| **Cross-Attention** | Q from one sequence, K/V from another |
| **Multi-Head** | Multiple attention in parallel |

### Application Example: Machine Translation
When translating "The cat sat on the mat":
- When generating "chat" (French for cat), attention focuses on "cat"
- When generating "assis" (sat), attention focuses on "sat"

### Python Code Example
```python
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, V)
```

---

## Question 21: What are the challenges in training deep neural networks?
- [ ] Done

### Definition
Training deep networks presents unique challenges including vanishing/exploding gradients, computational cost, overfitting, hyperparameter sensitivity, and difficulty in debugging.

### Key Challenges

| Challenge | Description | Solution |
|-----------|-------------|----------|
| **Vanishing Gradients** | Gradients shrink in deep networks | ReLU, skip connections, batch norm |
| **Exploding Gradients** | Gradients grow unstably | Gradient clipping, proper init |
| **Computational Cost** | Huge memory and time requirements | GPU/TPU, mixed precision |
| **Overfitting** | Too many parameters | Dropout, regularization, data augmentation |
| **Hyperparameter Sensitivity** | Many hyperparameters to tune | Grid search, Bayesian optimization |
| **Degradation Problem** | Accuracy saturates with depth | Residual connections |
| **Data Requirements** | Need large labeled datasets | Transfer learning, self-supervised |

### Practical Solutions
- Use pretrained models (transfer learning)
- Start simple, gradually add complexity
- Monitor training curves carefully
- Use batch normalization throughout

---

## Question 22: Explain the concept of semantic segmentation in the context of CNNs
- [ ] Done

### Definition
Semantic segmentation classifies each pixel in an image into a predefined category, producing a dense pixel-level mask. Unlike classification (one label per image), segmentation provides pixel-level understanding.

### Core Concepts

| Task | Output |
|------|--------|
| **Classification** | One label per image |
| **Object Detection** | Bounding boxes + labels |
| **Semantic Segmentation** | One label per pixel |
| **Instance Segmentation** | Separate objects of same class |

### Architecture: Encoder-Decoder

```
Image → Encoder (downsamples) → Bottleneck → Decoder (upsamples) → Pixel mask
```

### Popular Architectures

| Architecture | Key Innovation |
|--------------|----------------|
| **FCN** | Fully convolutional, no FC layers |
| **U-Net** | Skip connections between encoder/decoder |
| **DeepLab** | Atrous (dilated) convolutions |

### Python Code Example
```python
# Using pretrained DeepLab
model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet50', pretrained=True)
output = model(image)['out']  # Shape: (batch, num_classes, H, W)
prediction = output.argmax(dim=1)  # Pixel-wise class prediction
```

### Applications
- Autonomous driving (road, car, pedestrian segmentation)
- Medical imaging (tumor segmentation)
- Satellite imagery (land use classification)

---

## Question 23: What is the purpose of pooling layers in CNNs?
- [ ] Done

### Definition
Pooling layers reduce spatial dimensions of feature maps while retaining important information, providing translation invariance and reducing computational cost.

### Types of Pooling

| Type | Operation | Output |
|------|-----------|--------|
| **Max Pooling** | Take maximum in window | Preserves strongest activations |
| **Average Pooling** | Take average in window | Smooth feature representation |
| **Global Average** | Average entire feature map | Reduces to single value per channel |

### Benefits

| Benefit | Explanation |
|---------|-------------|
| **Dimensionality Reduction** | Reduces computation |
| **Translation Invariance** | Small shifts don't change output |
| **Noise Reduction** | Averages out noise |
| **Controls Overfitting** | Fewer parameters downstream |

### Python Code Example
```python
# Max pooling: 2x2 window, stride 2
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
# Input: (batch, channels, 32, 32) → Output: (batch, channels, 16, 16)

# Global average pooling (before FC layer)
gap = nn.AdaptiveAvgPool2d(1)
# Input: (batch, channels, H, W) → Output: (batch, channels, 1, 1)
```

### Modern Trends
- Strided convolutions sometimes replace pooling
- Global Average Pooling before final classification (reduces parameters)

---

## Question 24: Describe the differences between 1D, 2D, and 3D convolutions
- [ ] Done

### Definition
Convolution dimensionality refers to the number of spatial dimensions the filter slides across. 1D for sequences, 2D for images, 3D for videos or volumetric data.

### Comparison

| Type | Input Shape | Filter Moves | Use Case |
|------|-------------|--------------|----------|
| **1D Conv** | (batch, channels, length) | Along 1 axis | Time series, text, audio |
| **2D Conv** | (batch, channels, H, W) | Along 2 axes | Images |
| **3D Conv** | (batch, channels, D, H, W) | Along 3 axes | Video, medical scans |

### Visual Representation
```
1D: [---filter---] → slides along sequence
2D: [[filter]] → slides across height and width
3D: [[[filter]]] → slides across depth, height, width
```

### Python Code Example
```python
# 1D Conv for text/time series
conv1d = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
# Input: (batch, 128, seq_len) → Output: (batch, 64, seq_len-2)

# 2D Conv for images
conv2d = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
# Input: (batch, 3, H, W) → Output: (batch, 64, H-2, W-2)

# 3D Conv for video
conv3d = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=3)
# Input: (batch, 3, D, H, W) → Output: (batch, 64, D-2, H-2, W-2)
```

---

## Question 25: What is gradient clipping, and why might it be useful?
- [ ] Done

### Definition
Gradient clipping limits the magnitude of gradients during backpropagation to prevent exploding gradients, which cause unstable training with very large weight updates.

### Methods

| Method | Description |
|--------|-------------|
| **Clip by Value** | Clip each gradient element to [-max, max] |
| **Clip by Norm** | Scale gradients if total norm exceeds threshold |

### Mathematical Formulation
**Clip by Norm:**
$$g_{clipped} = \begin{cases} g & \text{if } ||g|| \leq \text{threshold} \\ \frac{g \cdot \text{threshold}}{||g||} & \text{otherwise} \end{cases}$$

### Python Code Example
```python
# Clip by norm (most common)
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

### When to Use
- Training RNNs/LSTMs (prone to exploding gradients)
- Deep networks with skip connections
- When loss suddenly explodes during training

---

## Question 26: Explain the concepts of momentum and Nesterov accelerated gradient
- [ ] Done

### Definition
**Momentum** accelerates SGD by accumulating past gradients, helping to navigate ravines and maintain direction. **Nesterov momentum** looks ahead before computing the gradient for smarter updates.

### Standard Momentum
$$v_t = \gamma v_{t-1} + \eta \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_t$$

### Nesterov Momentum
$$v_t = \gamma v_{t-1} + \eta \nabla L(\theta_t - \gamma v_{t-1})$$
$$\theta_{t+1} = \theta_t - v_t$$

Key difference: Nesterov computes gradient at "look-ahead" position.

### Benefits

| Benefit | Explanation |
|---------|-------------|
| **Faster Convergence** | Accumulates velocity in consistent directions |
| **Smoother Path** | Dampens oscillations |
| **Escapes Local Minima** | Momentum carries past small bumps |

### Python Code Example
```python
# Standard momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Nesterov momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
```

---

## Question 27: What is Adam optimization, and how does it differ from SGD?
- [ ] Done

### Definition
Adam (Adaptive Moment Estimation) combines momentum with adaptive per-parameter learning rates, maintaining running averages of both gradients and squared gradients.

### Adam Update Rule
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$ (momentum)
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$ (RMSprop)
$$\hat{m}_t = m_t / (1-\beta_1^t)$$ (bias correction)
$$\hat{v}_t = v_t / (1-\beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

### Comparison

| Aspect | SGD | Adam |
|--------|-----|------|
| **Learning Rate** | Same for all parameters | Adaptive per parameter |
| **Momentum** | Optional | Built-in |
| **Tuning** | More sensitive to LR | More robust |
| **Memory** | O(n) | O(3n) |
| **Generalization** | Often better | May generalize worse |

### Python Code Example
```python
# Adam (commonly used defaults)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW (with proper weight decay)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### When to Use
- **Adam**: Quick convergence, good default
- **SGD + Momentum**: Often better final performance with proper tuning

---

## Question 28: What are the main strategies for hyperparameter tuning in neural networks?
- [ ] Done

### Definition
Hyperparameter tuning searches for optimal values of parameters set before training (learning rate, batch size, architecture) to maximize model performance.

### Key Hyperparameters

| Category | Parameters |
|----------|------------|
| **Optimization** | Learning rate, batch size, optimizer |
| **Regularization** | Dropout rate, weight decay |
| **Architecture** | Layers, neurons, activation functions |
| **Training** | Epochs, early stopping patience |

### Tuning Strategies

| Strategy | Description | Pros/Cons |
|----------|-------------|-----------|
| **Grid Search** | Try all combinations | Thorough but expensive |
| **Random Search** | Sample randomly | Often more efficient |
| **Bayesian Optimization** | Model the objective function | Smart but complex |
| **Hyperband/ASHA** | Early stopping of bad trials | Very efficient |

### Python Code Example
```python
# Using Optuna for Bayesian optimization
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    
    model = create_model(dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    val_loss = train_and_evaluate(model, optimizer)
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

### Best Practices
1. Start with proven defaults
2. Tune learning rate first (most important)
3. Use random search over grid search
4. Use early stopping to save compute

---

## Question 29: What is the role of recurrent connections in RNNs?
- [ ] Done

### Definition
Recurrent connections pass the hidden state from one time step to the next, enabling the network to maintain memory of past inputs and process sequential data of variable length.

### How It Works
$$h_t = \sigma(W_x x_t + W_h h_{t-1} + b)$$

| Component | Role |
|-----------|------|
| $x_t$ | Current input |
| $h_{t-1}$ | Previous hidden state (memory) |
| $h_t$ | Current hidden state |
| $W_h$ | Recurrent weight matrix |

### What Recurrent Connections Enable

| Capability | Description |
|------------|-------------|
| **Memory** | Information persists across time steps |
| **Variable Length** | Can process sequences of any length |
| **Context** | Current output depends on past inputs |
| **Sequential Patterns** | Learn temporal dependencies |

### Limitations
- Vanishing gradients limit long-term memory
- Sequential processing (cannot parallelize)
- Addressed by LSTM/GRU and Transformers

---

## Question 30: Explain the theory behind Siamese networks and their use cases
- [ ] Done

### Definition
Siamese networks use two identical subnetworks with shared weights to learn a similarity function between pairs of inputs, useful for one-shot learning and verification tasks.

### Architecture
```
Input A → [Shared Network] → Embedding A
                                         ↘
                                          → Distance/Similarity → Same/Different
                                         ↗
Input B → [Shared Network] → Embedding B
```

### Training
- **Contrastive Loss**: Pull similar pairs together, push dissimilar apart
- **Triplet Loss**: Anchor closer to positive than negative

### Mathematical Formulation
**Contrastive Loss:**
$$L = (1-y) \cdot D^2 + y \cdot \max(0, m-D)^2$$

Where $D$ is distance, $y=0$ if same class, $m$ is margin.

### Use Cases

| Application | How Siamese Helps |
|-------------|-------------------|
| **Face Verification** | Compare two faces (same person?) |
| **Signature Verification** | Compare signatures for forgery |
| **One-Shot Learning** | Learn from single example per class |
| **Tracking** | Match object across video frames |

### Python Code Example
```python
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, x1, x2):
        emb1 = self.encoder(x1)
        emb2 = self.encoder(x2)
        distance = F.pairwise_distance(emb1, emb2)
        return distance
```

---

## Question 31: Describe how an autoencoder works and potential applications
- [ ] Done

### Definition
An autoencoder is an unsupervised neural network that learns to compress input into a lower-dimensional latent representation and then reconstruct the original input, learning efficient data encodings.

### Architecture
```
Input → Encoder → Latent Space (Bottleneck) → Decoder → Reconstructed Input
```

### Training
- **Loss**: Reconstruction error (MSE or BCE)
- **Goal**: Minimize $L = ||x - \hat{x}||^2$

### Types of Autoencoders

| Type | Description | Use Case |
|------|-------------|----------|
| **Vanilla** | Basic encoder-decoder | Compression |
| **Variational (VAE)** | Learns probability distribution | Generation |
| **Denoising** | Reconstructs from corrupted input | Denoising |
| **Sparse** | Encourages sparse representations | Feature learning |

### Applications

| Application | How Autoencoders Help |
|-------------|----------------------|
| **Dimensionality Reduction** | Nonlinear alternative to PCA |
| **Anomaly Detection** | High reconstruction error = anomaly |
| **Denoising** | Remove noise from images |
| **Generation (VAE)** | Sample from latent space |
| **Feature Learning** | Pretrain encoder for downstream tasks |

### Python Code Example
```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
```

---

## Question 32: How do LSTMs work, and what are their advantages over basic RNNs?
- [ ] Done

### Definition
LSTMs use a gated architecture with a separate cell state that acts as a "conveyor belt" for information, allowing selective reading, writing, and erasing of memory to capture long-range dependencies.

### LSTM Components

| Component | Formula | Purpose |
|-----------|---------|---------|
| **Forget Gate** | $f_t = \sigma(W_f \cdot [h_{t-1}, x_t])$ | What to erase from memory |
| **Input Gate** | $i_t = \sigma(W_i \cdot [h_{t-1}, x_t])$ | What new info to store |
| **Cell State** | $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ | Long-term memory |
| **Output Gate** | $o_t = \sigma(W_o \cdot [h_{t-1}, x_t])$ | What to output |
| **Hidden State** | $h_t = o_t \odot \tanh(C_t)$ | Short-term memory |

### Advantages Over Basic RNNs

| Advantage | How LSTM Achieves It |
|-----------|---------------------|
| **Long-term Memory** | Cell state carries information unchanged |
| **No Vanishing Gradients** | Additive operations in cell state |
| **Selective Memory** | Gates control what to remember/forget |
| **Better Gradient Flow** | Highway for gradients through cell state |

### Python Code Example
```python
# LSTM layer
lstm = nn.LSTM(
    input_size=100,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    dropout=0.2,
    bidirectional=True
)

# Forward pass
output, (h_n, c_n) = lstm(x)
# output: (batch, seq_len, 2*hidden_size) if bidirectional
# h_n: final hidden state
# c_n: final cell state
```

---

## Question 33: Describe the process to debug a model that is not learning
- [ ] Done

### Definition
Debugging a non-learning model involves systematically checking data pipeline, model architecture, training setup, and hyperparameters to identify and fix the issue.

### Debugging Checklist

| Step | Check | Fix |
|------|-------|-----|
| **1. Data** | Is data loaded correctly? | Print samples, visualize |
| **2. Labels** | Are labels correct and aligned? | Verify label mapping |
| **3. Preprocessing** | Is normalization applied? | Check mean/std |
| **4. Loss** | Is loss decreasing at all? | Check loss function choice |
| **5. Learning Rate** | Is LR appropriate? | Try 1e-4 to 1e-2 |
| **6. Gradients** | Are gradients flowing? | Check for NaN/zero grads |
| **7. Overfitting Test** | Can it overfit 1 batch? | Remove regularization |
| **8. Architecture** | Are layers connected properly? | Print model summary |

### Quick Diagnostic Tests

```python
# Test 1: Can model overfit a single batch?
for _ in range(1000):
    loss = train_one_batch(model, single_batch)
# If loss doesn't decrease → fundamental problem

# Test 2: Check for dead ReLUs
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean = {param.grad.mean():.6f}")

# Test 3: Verify data
print(f"Input range: [{X.min():.3f}, {X.max():.3f}]")
print(f"Target distribution: {np.bincount(y)}")
```

### Common Issues and Solutions

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Loss = NaN | Exploding gradients | Gradient clipping, lower LR |
| Loss constant | Zero gradients or wrong loss | Check gradients, loss function |
| Loss very high | LR too high | Reduce LR by 10x |
| Accuracy ~random | Label mismatch | Verify label encoding |

---

## Question 34: What are strategies to improve computational efficiency in neural network training?
- [ ] Done

### Definition
Computational efficiency involves reducing training time and memory usage while maintaining model performance through hardware, software, and algorithmic optimizations.

### Key Strategies

| Category | Strategy | Impact |
|----------|----------|--------|
| **Hardware** | GPU/TPU training | 10-100x speedup |
| **Precision** | Mixed precision (FP16) | 2x memory, 2x speed |
| **Data Loading** | Parallel data loading | Remove I/O bottleneck |
| **Batch Size** | Larger batches (if memory allows) | Better GPU utilization |
| **Architecture** | Efficient architectures (MobileNet) | Fewer FLOPs |
| **Training** | Gradient checkpointing | Trade compute for memory |

### Python Code Example
```python
# Mixed precision training
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Multi-GPU training
model = nn.DataParallel(model)  # Simple multi-GPU
# or use DistributedDataParallel for better scaling
```

### Efficient Data Loading
```python
train_loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,       # Parallel loading
    pin_memory=True,     # Faster GPU transfer
    prefetch_factor=2    # Preload batches
)
```

---

## Question 35: Explain the importance of checkpoints and early stopping
- [ ] Done

### Definition
**Checkpoints** save model state during training for recovery and selecting best model. **Early stopping** halts training when validation performance stops improving to prevent overfitting.

### Checkpointing

| Benefit | Description |
|---------|-------------|
| **Recovery** | Resume from crashes |
| **Best Model** | Save model with best validation score |
| **Experimentation** | Compare different epochs |

### Early Stopping

| Parameter | Description |
|-----------|-------------|
| **monitor** | Metric to track (val_loss) |
| **patience** | Epochs to wait before stopping |
| **restore_best_weights** | Use best model, not last |

### Python Code Example
```python
# Checkpointing
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pt')

# Loading checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Early stopping (PyTorch Lightning or manual)
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
```

---

## Question 36: Describe a real-world application where CNNs could be applied
- [ ] Done

### Application: Automated Quality Inspection in Manufacturing

### Problem
Manually inspecting products for defects is slow, inconsistent, and expensive. Need automated visual inspection system.

### CNN Solution

| Component | Implementation |
|-----------|----------------|
| **Input** | Camera images of products on assembly line |
| **Model** | CNN classifier (defect/no defect) |
| **Output** | Real-time accept/reject decision |

### Architecture Choice
- **Transfer Learning**: Start with pretrained ResNet/EfficientNet
- **Fine-tune**: On factory-specific defect images
- **Real-time**: Use optimized inference (TensorRT)

### Pipeline
```
Camera → Preprocessing → CNN Model → Defect? → Actuator (accept/reject)
                                    ↓
                              Alert system
```

### Practical Considerations
- Need diverse defect examples for training
- Handle class imbalance (most products are good)
- Ensure consistent lighting/positioning
- Deploy on edge device for low latency

---

## Question 37: Describe a strategy to use neural networks for sentiment analysis on social media posts
- [ ] Done

### Definition
Sentiment analysis classifies text into positive, negative, or neutral sentiment. For social media, must handle informal language, emojis, and short texts.

### Strategy

| Step | Implementation |
|------|----------------|
| **1. Data Collection** | Gather labeled tweets/posts |
| **2. Preprocessing** | Handle mentions, hashtags, emojis |
| **3. Model Selection** | Fine-tuned BERT or distilled version |
| **4. Training** | Fine-tune on sentiment dataset |
| **5. Deployment** | API for real-time classification |

### Preprocessing Pipeline
```python
def preprocess_tweet(text):
    text = text.lower()
    text = re.sub(r'@\w+', '<USER>', text)      # Replace mentions
    text = re.sub(r'http\S+', '<URL>', text)    # Replace URLs
    text = re.sub(r'#(\w+)', r'\1', text)       # Remove # from hashtags
    # Keep emojis - they carry sentiment!
    return text
```

### Model Approach
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Fine-tune on your social media dataset
```

### Considerations
- Emojis carry strong sentiment signal
- Sarcasm is difficult to detect
- Domain-specific fine-tuning helps significantly

---

## Question 38: What are zero-shot and few-shot learning in neural networks?
- [ ] Done

### Definition
**Zero-shot learning** performs tasks on classes never seen during training by leveraging semantic knowledge. **Few-shot learning** learns from only a handful (1-10) of examples per class.

### Comparison

| Aspect | Zero-Shot | Few-Shot |
|--------|-----------|----------|
| **Training Examples** | 0 per new class | 1-10 per new class |
| **Requirement** | Semantic descriptions | Small support set |
| **Key Technique** | Transfer semantic knowledge | Meta-learning, metric learning |

### Zero-Shot Example
- Train classifier on "dog", "cat", "bird"
- Test on "zebra" using description: "striped horse-like animal"
- Model maps "zebra" to learned attributes

### Few-Shot Example
- Given 5 examples each of "rare disease A" and "rare disease B"
- Model must classify new cases correctly

### Approaches

| Approach | Description |
|----------|-------------|
| **Metric Learning** | Learn to compare (Siamese, Prototypical) |
| **Meta-Learning** | Learn to learn (MAML) |
| **Large LMs** | GPT-3 style prompting |

### Python Example (Prototypical Networks)
```python
# Compute class prototypes from support set
prototypes = []
for class_samples in support_set:
    prototype = encoder(class_samples).mean(dim=0)
    prototypes.append(prototype)

# Classify query by nearest prototype
query_embedding = encoder(query)
distances = [F.pairwise_distance(query_embedding, p) for p in prototypes]
prediction = torch.argmin(torch.stack(distances))
```

---

## Question 39: Describe research in neural network interpretability
- [ ] Done

### Definition
Interpretability research aims to understand why neural networks make specific predictions, making them more trustworthy, debuggable, and compliant with regulations.

### Key Techniques

| Technique | Type | Description |
|-----------|------|-------------|
| **Saliency Maps** | Gradient-based | Highlight important input features |
| **Grad-CAM** | Gradient-based | Visualize CNN attention regions |
| **SHAP** | Game-theoretic | Feature importance scores |
| **LIME** | Perturbation-based | Local linear approximation |
| **Attention Visualization** | Architecture-based | Show attention weights |
| **Probing** | Analysis | Test what layers encode |

### Why It Matters

| Domain | Requirement |
|--------|-------------|
| **Healthcare** | Explain diagnosis to doctors |
| **Finance** | Regulatory compliance (GDPR, "right to explanation") |
| **Autonomous Vehicles** | Understand failure cases |
| **Research** | Improve models by understanding them |

### Python Example (Grad-CAM)
```python
# Grad-CAM for CNN
def grad_cam(model, image, target_class):
    model.eval()
    features = model.features(image)  # Get final conv features
    output = model(image)
    
    model.zero_grad()
    output[0, target_class].backward()
    
    gradients = model.get_gradients()
    weights = gradients.mean(dim=[2, 3])
    cam = torch.sum(weights * features, dim=1)
    return F.relu(cam)  # Heatmap
```

---

## Question 40: Explain quantum neural networks and their potential
- [ ] Done

### Definition
Quantum neural networks (QNNs) leverage quantum computing principles (superposition, entanglement) to potentially solve certain problems exponentially faster than classical neural networks.

### Core Concepts

| Concept | Classical | Quantum |
|---------|-----------|---------|
| **Data Unit** | Bit (0 or 1) | Qubit (superposition of 0 and 1) |
| **Operations** | Matrix multiplication | Quantum gates |
| **Parallelism** | Limited | Exponential via superposition |

### Potential Advantages

| Advantage | Description |
|-----------|-------------|
| **Exponential Speedup** | For certain optimization problems |
| **Better Feature Space** | Quantum kernels may capture complex patterns |
| **Quantum Data** | Natural for quantum chemistry, physics |

### Current Limitations
- Noisy qubits (NISQ era)
- Few qubits available (~100s vs millions of classical neurons)
- Decoherence limits computation time
- Unclear quantum advantage for ML

### Current State
- Research stage, not production-ready
- Hybrid quantum-classical approaches promising
- Companies exploring: Google, IBM, Amazon Braket

---

## Question 41: Describe how adversarial examples affect neural networks and defense methods
- [ ] Done

### Definition
Adversarial examples are inputs with imperceptible perturbations that cause neural networks to make incorrect predictions with high confidence, revealing vulnerabilities in neural network robustness.

### How They Work
$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L(f(x), y))$$

Small, targeted noise that maximizes loss while being imperceptible.

### Impact

| Concern | Description |
|---------|-------------|
| **Security** | Attacks on autonomous vehicles, biometric systems |
| **Reliability** | Models not truly robust |
| **Transferability** | Adversarial examples often transfer between models |

### Defense Methods

| Defense | Description |
|---------|-------------|
| **Adversarial Training** | Train on adversarial examples |
| **Input Preprocessing** | Denoise inputs (JPEG compression, smoothing) |
| **Certified Defense** | Provably robust within epsilon ball |
| **Ensemble Methods** | Harder to attack multiple models |

### Python Example (Adversarial Training)
```python
def generate_adversarial(model, x, y, epsilon=0.03):
    x.requires_grad = True
    output = model(x)
    loss = F.cross_entropy(output, y)
    loss.backward()
    x_adv = x + epsilon * x.grad.sign()
    return x_adv.clamp(0, 1)

# Train on clean and adversarial examples
for x, y in dataloader:
    x_adv = generate_adversarial(model, x, y)
    loss_clean = criterion(model(x), y)
    loss_adv = criterion(model(x_adv), y)
    loss = loss_clean + loss_adv
    loss.backward()
```

---

## Question 42: What is reinforcement learning, and how do deep neural networks play a role?
- [ ] Done

### Definition
Reinforcement learning (RL) trains agents to make sequential decisions by maximizing cumulative reward through trial and error. Deep RL uses neural networks to approximate value functions or policies for complex, high-dimensional problems.

### RL Components

| Component | Description |
|-----------|-------------|
| **Agent** | The learner/decision maker |
| **Environment** | World the agent interacts with |
| **State** | Current situation |
| **Action** | What agent can do |
| **Reward** | Feedback signal |
| **Policy** | Strategy for choosing actions |

### Deep RL Algorithms

| Algorithm | Neural Network Role |
|-----------|-------------------|
| **DQN** | Approximates Q-function (action values) |
| **Policy Gradient** | Directly parameterizes policy |
| **Actor-Critic** | Separate networks for policy and value |
| **PPO** | Stable policy optimization |

### Why Neural Networks Help
- Handle high-dimensional inputs (images)
- Learn complex policies
- Generalize across similar states

### Python Example (DQN concept)
```python
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Q-values for each action

# Select action with epsilon-greedy
q_values = dqn(state)
action = q_values.argmax() if random.random() > epsilon else random_action
```

### Applications
- Game playing (AlphaGo, Atari)
- Robotics
- Recommendation systems
- Autonomous vehicles

---

## Question 43: Explain neural networks' contribution to drug discovery
- [ ] Done

### Definition
Neural networks accelerate drug discovery by predicting molecular properties, generating novel compounds, and identifying drug-target interactions, reducing the time and cost of bringing new drugs to market.

### Applications

| Application | Neural Network Role |
|-------------|-------------------|
| **Property Prediction** | Predict toxicity, solubility, binding affinity |
| **Molecule Generation** | Generate novel drug candidates (VAE, GAN) |
| **Target Identification** | Predict protein-drug interactions |
| **Virtual Screening** | Filter millions of compounds quickly |
| **Retrosynthesis** | Plan synthesis routes |

### Key Architectures

| Architecture | Use Case |
|--------------|----------|
| **Graph Neural Networks** | Model molecular structures as graphs |
| **Transformers** | Process SMILES strings |
| **VAE** | Generate novel molecules |
| **CNN** | Analyze protein structures |

### Impact
- Reduce drug development time from 10+ years
- Identify novel drug candidates
- Repurpose existing drugs
- AlphaFold revolutionized protein structure prediction

### Python Example (Molecular Property Prediction)
```python
from rdkit import Chem
from torch_geometric.nn import GCNConv

class MoleculeGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(atom_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, mol_graph):
        x = F.relu(self.conv1(mol_graph.x, mol_graph.edge_index))
        x = self.conv2(x, mol_graph.edge_index)
        x = global_mean_pool(x, mol_graph.batch)
        return self.fc(x)  # Predict property
```

---

## Question 44: How can neural networks be used for credit scoring and fraud detection in finance?
- [ ] Done

### Definition
Neural networks analyze transaction patterns and customer data to assess creditworthiness and detect fraudulent activities with higher accuracy than traditional rule-based systems.

### Credit Scoring

| Component | Implementation |
|-----------|----------------|
| **Input Features** | Income, credit history, debt ratio |
| **Model** | Deep network or gradient boosting hybrid |
| **Output** | Probability of default |
| **Requirement** | Explainability (SHAP for feature importance) |

### Fraud Detection

| Approach | Description |
|----------|-------------|
| **Supervised** | Train on labeled fraud/non-fraud |
| **Unsupervised** | Autoencoders detect anomalies |
| **Sequence Models** | LSTM for transaction sequences |

### Architecture Considerations

| Challenge | Solution |
|-----------|----------|
| **Class Imbalance** | Oversampling, weighted loss, SMOTE |
| **Real-time** | Optimized inference, edge deployment |
| **Explainability** | SHAP, attention weights |
| **Concept Drift** | Continuous retraining |

### Python Example (Fraud Detection)
```python
class FraudDetector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

# Use weighted loss for imbalance
criterion = nn.BCELoss(weight=torch.tensor([1.0, 100.0]))  # 100x weight for fraud
```

---

## Question 45: Describe application of neural networks in medical image analysis
- [ ] Done

### Definition
Neural networks analyze medical images (X-rays, CT, MRI, histopathology) to assist in diagnosis, detecting diseases often with accuracy comparable to or exceeding human experts.

### Applications

| Application | Image Type | Task |
|-------------|------------|------|
| **Diabetic Retinopathy** | Fundus images | Severity classification |
| **Lung Cancer** | CT scans | Nodule detection |
| **Skin Cancer** | Dermoscopy | Melanoma classification |
| **COVID-19** | Chest X-ray/CT | Detection |
| **Pathology** | Histology slides | Cancer grading |

### Typical Pipeline

```
Medical Image → Preprocessing → CNN → Diagnosis/Segmentation
                                  ↓
                         Grad-CAM Visualization
                                  ↓
                        Clinician Review
```

### Key Considerations

| Consideration | Approach |
|---------------|----------|
| **Small Datasets** | Transfer learning from ImageNet |
| **Explainability** | Grad-CAM heatmaps |
| **Validation** | External hospital validation |
| **Regulation** | FDA/CE approval required |
| **Integration** | DICOM compatibility |

### Python Example
```python
# Transfer learning for medical imaging
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Fine-tune with low learning rate
optimizer = torch.optim.Adam([
    {'params': model.fc.parameters(), 'lr': 1e-3},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
], lr=1e-5)
```

---

## Question 46: Explain how virtual assistants like Siri or Alexa use neural networks
- [ ] Done

### Definition
Virtual assistants use a pipeline of neural networks for speech recognition (ASR), natural language understanding (NLU), dialog management, and speech synthesis (TTS) to understand and respond to voice commands.

### Pipeline Components

| Component | Neural Network | Function |
|-----------|----------------|----------|
| **Wake Word** | Small CNN/RNN | Detect "Hey Siri", "Alexa" |
| **ASR** | Transformer (Whisper) | Speech → Text |
| **NLU** | BERT-based | Extract intent and entities |
| **Dialog Manager** | RL/Transformer | Decide response |
| **NLG** | GPT-based | Generate response text |
| **TTS** | WaveNet/Tacotron | Text → Speech |

### Example Flow
```
"Hey Siri, what's the weather in Seattle?"
         ↓
Wake Word Detection: Activated
         ↓
ASR: "what's the weather in Seattle"
         ↓
NLU: Intent=get_weather, Entity={location: "Seattle"}
         ↓
Dialog: Query weather API
         ↓
NLG: "The weather in Seattle is 55°F and cloudy"
         ↓
TTS: Generate speech audio
```

### Key Technologies

| Technology | Purpose |
|------------|---------|
| **Transformer ASR** | Accurate speech recognition |
| **Slot Filling** | Extract entities (dates, locations) |
| **Knowledge Graph** | Answer factual questions |
| **Personalization** | Learn user preferences |

### Challenges
- Noise handling (far-field speech)
- Accents and dialects
- Context management across turns
- Privacy (on-device vs cloud processing)
