# Neural Networks Interview Questions - General Questions

## Question 1: Elaborate on the structure of a basic artificial neuron
- [ ] Done

### Structure Components
A basic artificial neuron receives multiple inputs, applies weights to each, sums them with a bias, and passes through an activation function.

| Component | Function |
|-----------|----------|
| **Inputs (x₁, x₂, ...)** | Feature values from data or previous layer |
| **Weights (w₁, w₂, ...)** | Learnable parameters indicating importance |
| **Bias (b)** | Allows shifting the activation threshold |
| **Summation (Σ)** | Weighted sum: $z = \sum w_i x_i + b$ |
| **Activation (σ)** | Non-linear transformation: $a = \sigma(z)$ |

### Mathematical Model
$$output = \sigma\left(\sum_{i=1}^{n} w_i x_i + b\right)$$

### Visual Representation
```
x₁ ──w₁──┐
x₂ ──w₂──┼──→ [Σ] ──→ [σ] ──→ output
x₃ ──w₃──┘    ↑
              b
```

---

## Question 2: Difference between fully connected and convolutional layers
- [ ] Done

### Comparison

| Aspect | Fully Connected | Convolutional |
|--------|-----------------|---------------|
| **Connections** | Every neuron to every input | Local connections only |
| **Weight Sharing** | No | Yes (same filter across image) |
| **Parameters** | Input × Output | Filter_size × Channels |
| **Spatial Info** | Lost (flattened) | Preserved |
| **Use Case** | Classification head | Feature extraction |

### Key Differences
- **FC Layer**: $y = Wx + b$ where W is large matrix
- **Conv Layer**: Sliding filter preserves spatial structure
- FC: Good for final decision making
- Conv: Good for detecting patterns regardless of location

---

## Question 3: How do batch normalization layers work?
- [ ] Done

### Mechanism
Batch normalization normalizes activations across the mini-batch to zero mean and unit variance, then applies learnable scale (γ) and shift (β).

### Formula
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

### Problems It Solves

| Problem | How BN Helps |
|---------|--------------|
| **Internal Covariate Shift** | Stabilizes layer input distributions |
| **Slow Training** | Allows higher learning rates |
| **Gradient Issues** | Keeps activations in good range |
| **Need for Careful Init** | Less sensitive to initialization |

---

## Question 4: How to determine number of layers and types
- [ ] Done

### Decision Framework

| Factor | Consideration |
|--------|---------------|
| **Data Type** | Images→CNN, Sequences→RNN/Transformer |
| **Task Complexity** | Simple→shallow, Complex→deeper |
| **Data Size** | Small data→fewer layers, Big data→deeper |
| **Computational Budget** | Limited→efficient architectures |

### Practical Approach
1. Start with proven architectures (ResNet, BERT)
2. Use transfer learning when possible
3. Start simple, add complexity if needed
4. Validate on held-out data
5. Consider computational constraints

---

## Question 5: Criteria for choosing an optimizer
- [ ] Done

### Selection Criteria

| Optimizer | When to Use |
|-----------|-------------|
| **SGD + Momentum** | Well-understood problem, fine control needed |
| **Adam** | Default choice, works well generally |
| **AdamW** | When weight decay regularization matters |
| **RMSprop** | RNNs, non-stationary objectives |

### Key Factors
- **Learning Rate Sensitivity**: Adam is more robust
- **Generalization**: SGD often generalizes better
- **Convergence Speed**: Adam typically faster
- **Memory**: SGD uses less memory

### Common Practice
Start with Adam (lr=0.001), switch to SGD with momentum for fine-tuning if needed.

---

## Question 6: When to choose RNN over feedforward network
- [ ] Done

### Use RNN When

| Scenario | Reason |
|----------|--------|
| **Sequential Data** | Time series, text, audio |
| **Variable Length Input** | Sentences of different lengths |
| **Order Matters** | "dog bites man" ≠ "man bites dog" |
| **Temporal Dependencies** | Current output depends on history |

### Use Feedforward When
- Fixed-size input (tabular data, images with CNN)
- No temporal dependencies
- Order doesn't matter

### Note
For many sequence tasks, Transformers now outperform RNNs but require more compute.

---

## Question 7: Transfer learning and when to use it
- [ ] Done

### Definition
Transfer learning uses a model pretrained on a large dataset (e.g., ImageNet) and adapts it to a new task, leveraging learned features instead of training from scratch.

### Approaches

| Approach | When to Use |
|----------|-------------|
| **Feature Extraction** | Small dataset, similar domain |
| **Fine-tuning (all layers)** | Large dataset, different domain |
| **Fine-tuning (last layers)** | Medium dataset, similar domain |

### When to Use Transfer Learning
- Limited labeled data
- Pretrained model exists for similar domain
- Computational constraints
- Faster iteration needed

### Python Example
```python
# Feature extraction
model = resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, num_classes)  # Only train this
```

---

## Question 8: Metrics for evaluating neural network performance
- [ ] Done

### Classification Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Accuracy** | (TP+TN)/(All) | Balanced classes |
| **Precision** | TP/(TP+FP) | Cost of false positives high |
| **Recall** | TP/(TP+FN) | Cost of false negatives high |
| **F1-Score** | 2×(P×R)/(P+R) | Imbalanced classes |
| **AUC-ROC** | Area under curve | Ranking ability |

### Regression Metrics
- **MSE**: Penalizes large errors
- **MAE**: Robust to outliers
- **R²**: Variance explained

### Other Considerations
- Training vs validation loss gap (overfitting)
- Inference time (deployment)
- Model size (memory constraints)

---

## Question 9: How CNNs achieve translation invariance
- [ ] Done

### Mechanisms

| Mechanism | How It Helps |
|-----------|--------------|
| **Weight Sharing** | Same filter detects pattern anywhere |
| **Pooling Layers** | Summarizes regions, tolerates small shifts |
| **Data Augmentation** | Training with translated images |
| **Hierarchical Features** | Higher layers combine local features |

### Explanation
- Same convolutional filter slides across entire image
- If a "cat ear" detector finds an ear at position (10,10), it will also detect it at (50,50)
- Max pooling takes strongest activation in a region regardless of exact position

---

## Question 10: Effect of data normalization on neural networks
- [ ] Done

### Benefits

| Benefit | Explanation |
|---------|-------------|
| **Faster Convergence** | All features on similar scale |
| **Stable Gradients** | Avoids extreme values |
| **Better Initialization** | Standard init assumes normalized data |
| **Equal Feature Importance** | Prevents large-scale features dominating |

### Common Approaches
- **Min-Max**: Scale to [0, 1]
- **Z-score**: Mean=0, Std=1
- **Per-channel** (images): ImageNet mean/std

### Important
Always fit normalization on training data only, then apply to test data.

---

## Question 11: Challenge of catastrophic forgetting
- [ ] Done

### Definition
Catastrophic forgetting occurs when a neural network trained on new task B forgets previously learned task A, as weights are updated to optimize for new data only.

### Solutions

| Solution | Approach |
|----------|----------|
| **Elastic Weight Consolidation** | Penalize changing important weights |
| **Progressive Networks** | Add new columns, freeze old |
| **Replay** | Mix old data with new |
| **Multi-task Learning** | Train on all tasks jointly |

### Why It Happens
- New gradients update all weights
- No mechanism to "protect" important weights
- Single shared representation for all tasks

---

## Question 12: Attention mechanisms in transformer models
- [ ] Done

### How It Works

| Step | Description |
|------|-------------|
| **1. Create Q, K, V** | Linear projections of input |
| **2. Compute Scores** | $QK^T / \sqrt{d_k}$ |
| **3. Softmax** | Normalize to get attention weights |
| **4. Weighted Sum** | Apply weights to Values |

### Formula
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Multi-Head Attention
- Run multiple attention operations in parallel
- Each head can focus on different aspects
- Concatenate and project results

### Benefits
- Direct connections between any positions
- Parallelizable (unlike RNNs)
- Interpretable (visualize attention weights)

---

## Question 13: Bidirectional RNN and when to use it
- [ ] Done

### Definition
Bidirectional RNN processes sequence in both forward and backward directions, capturing context from both past and future.

### Architecture
```
Forward:  x₁ → x₂ → x₃ → x₄
                          ↘
                           [Concat] → Output
                          ↗
Backward: x₁ ← x₂ ← x₃ ← x₄
```

### When to Use

| Use Case | Example |
|----------|---------|
| **Full Sequence Available** | Text classification |
| **Context from Both Sides** | Named Entity Recognition |
| **NOT for Generation** | Can't see future during generation |

### Example
"The bank by the river" - knowing "river" helps disambiguate "bank"

---

## Question 14: Handling variable-length sequences
- [ ] Done

### Techniques

| Technique | Description |
|-----------|-------------|
| **Padding** | Add zeros to match max length |
| **Masking** | Ignore padded positions in computation |
| **Packing** | Efficient batching of different lengths |
| **Attention** | Naturally handles variable lengths |

### Python Example
```python
# Padding and masking
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

padded = pad_sequence(sequences, batch_first=True)
packed = pack_padded_sequence(padded, lengths, batch_first=True)
output, _ = rnn(packed)
```

---

## Question 15: Ensuring neural network generalizes well
- [ ] Done

### Strategies

| Strategy | Implementation |
|----------|----------------|
| **Train/Val/Test Split** | Monitor validation performance |
| **Cross-Validation** | Multiple splits for robust estimate |
| **Regularization** | Dropout, weight decay, early stopping |
| **Data Augmentation** | Increase effective training data |
| **Simpler Model** | Reduce capacity if overfitting |

### Signs of Good Generalization
- Small gap between train and validation loss
- Performance stable across different data splits
- Works on truly held-out test set

---

## Question 16: RNNs for time-series forecasting
- [ ] Done

### Approach

| Step | Implementation |
|------|----------------|
| **Input** | Windowed historical values |
| **Model** | LSTM/GRU to capture patterns |
| **Output** | Next value(s) prediction |
| **Training** | Minimize MSE on validation |

### Architecture Choices
- **Many-to-One**: Predict single future value
- **Many-to-Many**: Predict multiple future values
- **Encoder-Decoder**: For longer horizons

### Python Example
```python
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])
```

---

## Question 17: Autoencoders for anomaly detection
- [ ] Done

### Approach
1. Train autoencoder on normal data only
2. Learn to reconstruct normal patterns
3. At inference: high reconstruction error = anomaly

### Why It Works
- Autoencoder learns to compress and reconstruct normal data
- Anomalies don't fit learned patterns
- High reconstruction error indicates something unusual

### Python Example
```python
# During inference
reconstruction = autoencoder(input_data)
error = torch.mean((input_data - reconstruction) ** 2)
is_anomaly = error > threshold
```

### Applications
- Fraud detection
- Equipment failure prediction
- Network intrusion detection

---

## Question 18: GANs for image generation and style transfer
- [ ] Done

### Image Generation
- Generator learns to produce realistic images from noise
- Discriminator provides training signal
- Applications: faces (StyleGAN), art, data augmentation

### Style Transfer
- **Neural Style Transfer**: Combine content and style loss
- **CycleGAN**: Unpaired image-to-image translation
- **Pix2Pix**: Paired image translation

### Applications

| Application | Model |
|-------------|-------|
| Photo-realistic faces | StyleGAN |
| Image super-resolution | SRGAN |
| Domain adaptation | CycleGAN |
| Sketch to photo | Pix2Pix |

---

## Question 19: Neural network approach for recommendation system
- [ ] Done

### Architecture Options

| Approach | Description |
|----------|-------------|
| **Two-Tower** | Separate user and item encoders |
| **Neural CF** | Learn user-item interaction function |
| **Sequential** | RNN/Transformer for user history |

### Pipeline
1. **User Embedding**: Encode user features/history
2. **Item Embedding**: Encode item features
3. **Interaction**: Dot product or MLP
4. **Ranking**: Score all items, return top-K

### Python Example
```python
class RecModel(nn.Module):
    def __init__(self, num_users, num_items, dim):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        
    def forward(self, user, item):
        return (self.user_emb(user) * self.item_emb(item)).sum(1)
```

---

## Question 20: Architecture for self-driving car perception
- [ ] Done

### Components

| Task | Architecture |
|------|--------------|
| **Object Detection** | YOLO, Faster R-CNN |
| **Segmentation** | U-Net, DeepLab |
| **Depth Estimation** | Monocular depth networks |
| **Sensor Fusion** | Multi-modal networks |

### Design Considerations
- **Real-time**: Need fast inference (>30 FPS)
- **Multi-task**: Detect objects, lanes, signs simultaneously
- **Redundancy**: Multiple sensors (camera, LiDAR, radar)
- **Uncertainty**: Output confidence scores

---

## Question 21: Unsupervised learning in neural networks
- [ ] Done

### Applications

| Method | Use Case |
|--------|----------|
| **Autoencoders** | Dimensionality reduction, anomaly detection |
| **VAEs** | Generative modeling, representation learning |
| **Contrastive Learning** | Self-supervised pretraining (SimCLR) |
| **Clustering** | Deep clustering methods |

### Self-Supervised Learning
- Create labels from data itself
- Example: Predict masked words (BERT)
- Example: Match augmented views (SimCLR)

### Benefits
- Leverage unlabeled data
- Learn general representations
- Reduce annotation costs

---

## Question 22: Neural networks for predictive maintenance
- [ ] Done

### Approach

| Step | Implementation |
|------|----------------|
| **Data** | Sensor readings, maintenance logs |
| **Features** | Time-series patterns, statistics |
| **Model** | LSTM/CNN for temporal patterns |
| **Output** | Time-to-failure or failure probability |

### Architecture
- Input: Multi-variate sensor time series
- Model: 1D CNN or LSTM
- Output: Remaining useful life or classification

### Benefits
- Reduce unplanned downtime
- Optimize maintenance schedules
- Prevent costly failures
- Data-driven decisions

---
