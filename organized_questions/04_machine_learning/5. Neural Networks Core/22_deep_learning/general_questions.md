# Deep Learning Interview Questions - General Questions

## Question 1: Define deep learning and how it differs from other ML approaches
- [ ] Done

### Definition
Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep architectures) to automatically learn hierarchical representations from raw data, eliminating the need for manual feature engineering.

### Key Differences

| Aspect | Traditional ML | Deep Learning |
|--------|---------------|---------------|
| **Feature Engineering** | Manual, domain expertise required | Automatic, learned from data |
| **Data Requirements** | Works with less data | Requires large datasets |
| **Compute** | CPU sufficient | GPU/TPU typically needed |
| **Interpretability** | Often interpretable | Often black-box |
| **Representation** | Single level | Hierarchical |

### When Deep Learning Excels
- Large amounts of data available
- Complex patterns (images, speech, text)
- End-to-end learning preferred
- Compute resources available

---

## Question 2: How do dropout layers help prevent overfitting?
- [ ] Done

### Mechanism
Dropout randomly sets a fraction of neurons to zero during each training step, forcing the network to learn redundant representations and preventing co-adaptation of neurons.

### How It Works
1. During training: Randomly drop neurons with probability p
2. During inference: Use all neurons, scale by (1-p)

### Why It Prevents Overfitting

| Effect | Explanation |
|--------|-------------|
| **Ensemble Effect** | Each forward pass uses different sub-network |
| **Redundancy** | Network can't rely on specific neurons |
| **Co-adaptation** | Breaks dependencies between neurons |

### Python Example
```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # 50% dropout
    nn.Linear(256, 10)
)
```

---

## Question 3: Difference between standard neural network and Autoencoder
- [ ] Done

### Comparison

| Aspect | Standard NN | Autoencoder |
|--------|-------------|-------------|
| **Task** | Supervised (input→label) | Unsupervised (input→input) |
| **Architecture** | Input→Hidden→Output | Encoder→Bottleneck→Decoder |
| **Loss** | Task-specific (CE, MSE) | Reconstruction loss |
| **Purpose** | Prediction | Representation learning |

### Autoencoder Structure
```
Input → [Encoder] → Compressed Code → [Decoder] → Reconstructed Input
```

### Use Cases
- Dimensionality reduction
- Anomaly detection
- Denoising
- Feature learning

---

## Question 4: How to use transfer learning in deep learning
- [ ] Done

### Approaches

| Strategy | When to Use | Process |
|----------|-------------|---------|
| **Feature Extraction** | Small data, similar domain | Freeze all, train classifier |
| **Fine-tuning** | Medium data | Unfreeze some/all layers |
| **Full Training** | Large data, different domain | Use pretrained init only |

### Steps
1. Choose pretrained model (ImageNet, BERT)
2. Remove task-specific head
3. Add new head for your task
4. Freeze/unfreeze layers as needed
5. Train on your data

### Python Example
```python
# Load pretrained, replace classifier
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(512, num_classes)

# Optionally freeze backbone
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
```

---

## Question 5: How GPUs are utilized in training deep neural networks
- [ ] Done

### GPU Advantages

| Aspect | CPU | GPU |
|--------|-----|-----|
| **Cores** | Few (4-32) | Many (1000s) |
| **Parallelism** | Limited | Massive |
| **Matrix Operations** | Sequential | Parallel |
| **Memory Bandwidth** | Lower | Higher |

### How GPUs Help
- Matrix multiplications done in parallel
- Batch processing across many cores
- Specialized tensor cores for DL operations

### Practical Usage
```python
# Move model and data to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

### Multi-GPU Training
- Data parallelism: Same model, different batches
- Model parallelism: Split model across GPUs

---

## Question 6: Data preprocessing for deep learning
- [ ] Done

### Key Steps

| Step | Purpose |
|------|---------|
| **Normalization** | Scale to [0,1] or standardize (mean=0, std=1) |
| **Resizing** | Uniform input dimensions |
| **Augmentation** | Increase data diversity |
| **Encoding** | Convert categories to numbers |
| **Cleaning** | Handle missing values, outliers |

### Image-Specific
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### Text-Specific
- Tokenization
- Padding/truncation
- Vocabulary building

---

## Question 7: Handling overfitting beyond dropout
- [ ] Done

### Regularization Techniques

| Technique | How It Works |
|-----------|--------------|
| **L2 Regularization** | Penalize large weights |
| **Early Stopping** | Stop when validation loss increases |
| **Data Augmentation** | Artificially increase training data |
| **Batch Normalization** | Adds noise, regularizes |
| **Label Smoothing** | Soften one-hot targets |
| **Model Simplification** | Reduce layers/neurons |

### Additional Strategies
- Collect more training data
- Use pretrained models (transfer learning)
- Ensemble methods
- Cross-validation for model selection

---

## Question 8: Strategies for imbalanced datasets
- [ ] Done

### Techniques

| Approach | Implementation |
|----------|----------------|
| **Weighted Loss** | Higher weight for minority class |
| **Oversampling** | SMOTE, random oversampling |
| **Undersampling** | Reduce majority class |
| **Data Augmentation** | Augment minority class more |
| **Focal Loss** | Down-weight easy examples |
| **Threshold Adjustment** | Lower classification threshold |

### Python Example
```python
# Weighted cross-entropy
class_weights = torch.tensor([1.0, 10.0])  # Minority 10x weight
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Or use weighted sampler
sampler = WeightedRandomSampler(weights, len(weights))
dataloader = DataLoader(dataset, sampler=sampler)
```

---

## Question 9: Monitoring and debugging deep learning models
- [ ] Done

### What to Monitor

| Metric | What It Tells You |
|--------|-------------------|
| **Training Loss** | Model learning |
| **Validation Loss** | Generalization |
| **Loss Gap** | Overfitting indicator |
| **Gradient Norms** | Training stability |
| **Learning Rate** | Optimization progress |

### Debugging Steps
1. Overfit on small batch first (verify model can learn)
2. Check data pipeline (visualize inputs)
3. Monitor gradients (exploding/vanishing)
4. Start with known-good architecture
5. Simplify, then add complexity

### Tools
- TensorBoard, Weights & Biases
- Gradient hooks for inspection
- Print intermediate outputs

---

## Question 10: Framework for voice command recognition
- [ ] Done

### Pipeline

| Stage | Implementation |
|-------|----------------|
| **Input** | Audio waveform |
| **Feature Extraction** | Mel spectrograms or MFCCs |
| **Model** | CNN on spectrograms or Wav2Vec |
| **Output** | Command classification |

### Architecture
```
Audio → Spectrogram → CNN/Transformer → Softmax → Command
```

### Considerations
- Handle noise (augment with background sounds)
- Real-time processing requirement
- Wake word detection (separate model)
- Edge deployment (model compression)

---

## Question 11: CNNs for satellite imagery classification
- [ ] Done

### Architecture

| Layer | Purpose |
|-------|---------|
| **Conv Layers** | Extract spatial features |
| **Pooling** | Reduce dimensions |
| **FC Layers** | Classification |

### Considerations
- Large images: Use patch-based approach
- Multi-spectral: Handle multiple channels
- Class imbalance: Some land covers rare
- Transfer learning: Use pretrained on ImageNet

### Pipeline
1. Tile large images into manageable patches
2. Normalize spectral bands
3. Augment (rotations, flips)
4. Train CNN (ResNet, EfficientNet)
5. Post-process: smooth predictions spatially

---

## Question 12: Deep learning for genome sequence prediction
- [ ] Done

### Approaches

| Application | Model |
|-------------|-------|
| **Sequence Classification** | 1D CNN, LSTM |
| **Variant Effect** | Attention models |
| **Protein Structure** | AlphaFold-style transformers |
| **Gene Expression** | CNN on promoter regions |

### Representation
- One-hot encode nucleotides (A, T, G, C)
- Or learn embeddings
- Consider reverse complement

### Architecture Example
```
DNA Sequence → Embedding → Conv1D → Pooling → Dense → Output
```

---

## Question 13: Evaluating deep learning model performance
- [ ] Done

### Metrics by Task

| Task | Metrics |
|------|---------|
| **Classification** | Accuracy, F1, AUC, Precision, Recall |
| **Regression** | MSE, MAE, R² |
| **Generation** | FID, Inception Score |
| **Detection** | mAP, IoU |

### Beyond Accuracy
- Calibration: Are probabilities meaningful?
- Inference time: Fast enough for deployment?
- Robustness: Performance on edge cases
- Fairness: Equal performance across groups

---

## Question 14: Techniques for visualizing deep neural networks
- [ ] Done

### Methods

| Technique | Purpose |
|-----------|---------|
| **Saliency Maps** | What input pixels matter |
| **Grad-CAM** | Class activation maps |
| **t-SNE/UMAP** | Visualize embeddings |
| **Feature Maps** | What filters detect |
| **Attention Weights** | What model attends to |

### Python Example (Grad-CAM concept)
```python
# Hook to get gradients
activations = []
def hook(module, input, output):
    activations.append(output)

layer.register_forward_hook(hook)
output = model(image)
output[0, target_class].backward()
# Use gradients and activations for visualization
```

---

## Question 15: How confusion matrices help evaluate classification
- [ ] Done

### Information Provided

| From Confusion Matrix | Insight |
|----------------------|---------|
| **Diagonal** | Correct predictions |
| **Off-diagonal** | Misclassifications |
| **Row sums** | Actual class distribution |
| **Column sums** | Predicted distribution |

### Derived Metrics
- **Precision** per class: TP / (TP + FP)
- **Recall** per class: TP / (TP + FN)
- **Which classes confuse**: Common misclassification patterns

### Python Example
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot()
```

---

## Question 16: Error analysis on deep learning predictions
- [ ] Done

### Process

| Step | Action |
|------|--------|
| **1. Categorize Errors** | Group by error type |
| **2. Sample & Inspect** | Look at actual examples |
| **3. Find Patterns** | Common failure modes |
| **4. Prioritize** | Focus on biggest impact |
| **5. Address** | Data, model, or feature fixes |

### Common Findings
- Data quality issues (label noise)
- Distribution shift (train vs test)
- Underrepresented cases
- Edge cases model never saw

### Tools
- Sort by confidence: Low confidence correct, High confidence wrong
- Cluster errors to find patterns
- Compare with baseline model

---

## Question 17: Interpretability vs performance trade-off
- [ ] Done

### Trade-off Spectrum

| Model Type | Interpretability | Performance |
|------------|------------------|-------------|
| Linear models | High | Lower |
| Decision trees | High | Medium |
| Ensemble (RF, XGB) | Medium | High |
| Deep learning | Low | Highest |

### Strategies

| Approach | Description |
|----------|-------------|
| **Post-hoc Explanation** | SHAP, LIME on black-box model |
| **Attention Visualization** | For transformers/RNNs |
| **Simpler Architecture** | Sacrifice some accuracy |
| **Hybrid** | Complex model + simple proxy |

### When Interpretability Matters
- Regulated industries (healthcare, finance)
- High-stakes decisions
- Debugging and improvement
- Building trust with stakeholders

---
