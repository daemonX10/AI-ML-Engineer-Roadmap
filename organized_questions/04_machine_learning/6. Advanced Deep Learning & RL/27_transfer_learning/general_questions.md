# Transfer Learning Interview Questions - General Questions

---

## Question 1: In which scenarios is transfer learning most effective?
- [ ] Done

### High-Effectiveness Scenarios

| Scenario | Why Effective |
|----------|--------------|
| **Limited labeled data** | Leverages knowledge from large source datasets |
| **Related source/target** | Features transfer well |
| **Expensive data collection** | Reduces labeling costs |
| **Time constraints** | Faster than training from scratch |
| **Similar domains** | Minimal domain shift |

### Detailed Scenarios

**1. Small Target Dataset**
```
Available: 1,000 labeled images
Required from scratch: 100,000+ images
Transfer learning: Achieve 90%+ accuracy with 1,000 images
```

**2. Medical Imaging**
```
Source: ImageNet (millions of natural images)
Target: X-ray classification (thousands of images)
Why: General visual features (edges, textures) transfer well
```

**3. NLP with Domain Shift**
```
Source: General BERT (Wikipedia, books)
Target: Legal document classification
Why: Language understanding transfers; fine-tune on domain
```

**4. Computer Vision Tasks**
```
Object detection, segmentation, pose estimation
All benefit from ImageNet pre-trained backbones
```

### When Transfer Learning Excels

```python
effectiveness_factors = {
    'source_target_similarity': high,      # Related domains
    'source_data_size': large,             # Rich representations
    'target_data_size': small,             # Limited resources
    'task_complexity': high,               # Need learned features
    'compute_budget': limited              # Can't train from scratch
}
```

### Less Effective Scenarios
- Completely unrelated domains
- Abundant target data (scratch may match)
- Very different input modalities
- Real-time retraining needs

---

## Question 2: What role do pre-trained models play in transfer learning?
- [ ] Done

### Core Roles

| Role | Description |
|------|-------------|
| **Feature extraction** | Provide learned representations |
| **Initialization** | Better starting point than random |
| **Regularization** | Prevent overfitting on small data |
| **Knowledge source** | Encode patterns from large datasets |

### Feature Extraction Role
```python
# Pre-trained model provides rich features
pretrained = ResNet50(weights='imagenet')
feature_extractor = nn.Sequential(*list(pretrained.children())[:-1])

# Features capture: edges → textures → parts → objects
features = feature_extractor(images)  # (batch, 2048)
```

### Initialization Role
```
Random init: High loss, slow convergence
Pre-trained init: Lower loss, fast convergence

Training curves:
Random:     Loss: 2.5 → 0.5 (100 epochs)
Pre-trained: Loss: 0.8 → 0.3 (20 epochs)
```

### How Pre-trained Models Help

**1. Learned Representations**
```
ImageNet pre-training teaches:
- Edge detection (conv1)
- Texture recognition (conv2-3)
- Part detection (conv4)
- Object-level features (conv5)
```

**2. Regularization Effect**
```python
# Pre-trained weights act as prior
# Fine-tuning with low LR stays close to pre-trained
optimizer = Adam(model.parameters(), lr=1e-5)  # vs 1e-3 from scratch
```

**3. Reduced Compute**
```
Training ResNet-50 from scratch: 
  - 1.2M images, 90 epochs, days of GPU time
  
Fine-tuning:
  - 1K-10K images, 10-20 epochs, hours
```

### Popular Pre-trained Model Ecosystem
```python
# Vision
torchvision.models.resnet50(pretrained=True)
tf.keras.applications.EfficientNetB0(weights='imagenet')

# NLP
transformers.BertModel.from_pretrained('bert-base-uncased')
transformers.GPT2Model.from_pretrained('gpt2')

# Audio
torchaudio.pipelines.WAV2VEC2_BASE
```

---

## Question 3: How can transfer learning be deployed in small data scenarios?
- [ ] Done

### Strategy Overview

```
Small data (<1000 samples) + Transfer Learning:
1. Use pre-trained feature extractor
2. Freeze most/all layers
3. Train only classifier head
4. Apply strong regularization
5. Heavy data augmentation
```

### Approach 1: Feature Extraction Only
```python
# Best for very small data (<500 samples)
pretrained = models.resnet50(pretrained=True)
pretrained.eval()  # Keep frozen

# Extract features once
features = []
for images, labels in dataloader:
    with torch.no_grad():
        feat = pretrained(images)
    features.append(feat)

# Train simple classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(features, labels)
```

### Approach 2: Minimal Fine-tuning
```python
# For slightly more data (500-2000 samples)
model = models.resnet50(pretrained=True)

# Freeze everything except last block
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
    
# Replace and train classifier
model.fc = nn.Linear(2048, num_classes)

# Low learning rate
optimizer = Adam(model.parameters(), lr=1e-5)
```

### Data Augmentation Critical
```python
# Aggressive augmentation for small data
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
    transforms.RandomAffine(0, translate=(0.2, 0.2)),
    transforms.GaussianBlur(kernel_size=5),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

### Regularization Techniques
```python
# Dropout
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

# Weight decay
optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

# Early stopping
early_stopper = EarlyStopping(patience=10)
```

### Expected Performance

| Data Size | Strategy | Expected Accuracy |
|-----------|----------|-------------------|
| 50-100 | Feature extraction + SVM | 70-80% |
| 100-500 | Freeze all, train head | 75-85% |
| 500-2000 | Fine-tune last layers | 85-92% |
| 2000+ | Full fine-tuning | 90%+ |

---

## Question 4: How do multi-task learning and transfer learning compare?
- [ ] Done

### Definitions

**Transfer Learning:**
```
Source Task → Train → Target Task (sequential)
Knowledge flows one direction
```

**Multi-Task Learning:**
```
Task A ↔ Task B ↔ Task C (simultaneous)
Tasks help each other during training
```

### Key Differences

| Aspect | Transfer Learning | Multi-Task Learning |
|--------|-------------------|---------------------|
| **Training** | Sequential | Simultaneous |
| **Tasks** | Source → Target | All tasks together |
| **Knowledge flow** | One direction | Bidirectional |
| **Goal** | Improve target task | Improve all tasks |
| **Source task** | Discarded after transfer | Retained |

### Transfer Learning Flow
```
Phase 1: Train on source
    Source Data → Model → Source Task Performance
    
Phase 2: Transfer to target
    Pre-trained Model → Fine-tune → Target Task Performance
```

### Multi-Task Learning Flow
```
Shared Encoder
    ↓
Task A Head → Loss_A
Task B Head → Loss_B
Task C Head → Loss_C

Total Loss = w_A * Loss_A + w_B * Loss_B + w_C * Loss_C
```

### Multi-Task Implementation
```python
class MultiTaskModel(nn.Module):
    def __init__(self, num_classes_A, num_classes_B):
        super().__init__()
        # Shared backbone
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Task-specific heads
        self.head_A = nn.Linear(2048, num_classes_A)
        self.head_B = nn.Linear(2048, num_classes_B)
    
    def forward(self, x, task):
        features = self.backbone(x)
        if task == 'A':
            return self.head_A(features)
        else:
            return self.head_B(features)

# Training
for batch_A, batch_B in zip(loader_A, loader_B):
    loss_A = criterion(model(batch_A, 'A'), labels_A)
    loss_B = criterion(model(batch_B, 'B'), labels_B)
    loss = loss_A + loss_B
    loss.backward()
```

### When to Use Which

| Scenario | Best Approach |
|----------|--------------|
| Sequential tasks, one target | Transfer Learning |
| Related tasks, all important | Multi-Task Learning |
| Limited data, one task matters | Transfer Learning |
| Shared structure, joint training possible | Multi-Task Learning |

### Combining Both
```
Pre-train (ImageNet) → Multi-task fine-tune (Detection + Segmentation)
```

---

## Question 5: How do you decide how much of a pre-trained network to freeze?
- [ ] Done

### Decision Factors

| Factor | More Freezing | Less Freezing |
|--------|--------------|---------------|
| **Target data size** | Small | Large |
| **Domain similarity** | Different | Similar |
| **Task similarity** | Different | Similar |
| **Compute budget** | Limited | Ample |
| **Overfitting risk** | High | Low |

### Guidelines by Data Size

```
Very small (<500):     Freeze all, train only classifier
Small (500-2000):      Freeze early layers, fine-tune last 1-2 blocks
Medium (2000-10000):   Freeze first half, fine-tune second half
Large (10000+):        Fine-tune everything (with low LR)
```

### Practical Decision Tree
```
START
│
├─ Target data < 1000?
│   └─ YES → Freeze all backbone, train head only
│
├─ Source/Target domains similar?
│   ├─ YES → Can fine-tune more layers
│   └─ NO → Freeze more, especially early layers
│
├─ Overfitting during training?
│   └─ YES → Freeze more layers, add regularization
│
└─ Validation performance plateaus?
    └─ Try unfreezing more layers gradually
```

### Implementation: Gradual Unfreezing
```python
def get_layer_groups(model):
    return [
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.fc
    ]

def unfreeze_layers(model, num_layers):
    layer_groups = get_layer_groups(model)
    # Freeze all first
    for group in layer_groups:
        for param in group.parameters():
            param.requires_grad = False
    # Unfreeze last num_layers
    for group in layer_groups[-num_layers:]:
        for param in group.parameters():
            param.requires_grad = True

# Training schedule
epochs_per_stage = 5
for stage in range(1, 6):
    unfreeze_layers(model, stage)  # Progressively unfreeze
    train(model, epochs_per_stage)
```

### Discriminative Learning Rates
```python
# Lower LR for earlier layers, higher for later
optimizer = optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-6},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])
```

### Monitor These Metrics
1. Training vs validation loss (overfitting?)
2. Validation accuracy trend
3. Gradient magnitudes per layer
4. Training time per epoch

---

## Question 6: How can you adapt a pre-trained model from one domain to a different but related domain?
- [ ] Done

### Domain Adaptation Approaches

| Approach | Description |
|----------|-------------|
| **Fine-tuning** | Adjust weights on target data |
| **Feature alignment** | Match source/target distributions |
| **Adversarial** | Fool domain discriminator |
| **Self-training** | Pseudo-labels on target |

### Approach 1: Standard Fine-tuning
```python
# Load pre-trained on source domain
model = load_pretrained(source_domain)

# Fine-tune on target domain
model.train()
for epoch in range(num_epochs):
    for x, y in target_dataloader:
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
```

### Approach 2: Domain-Adversarial Neural Network (DANN)
```python
class DANNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = pretrained_backbone()
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Source vs Target
        )
    
    def forward(self, x, alpha):
        features = self.feature_extractor(x)
        class_output = self.classifier(features)
        
        # Gradient reversal for domain adaptation
        reversed_features = GradientReversal.apply(features, alpha)
        domain_output = self.domain_discriminator(reversed_features)
        
        return class_output, domain_output

# Training
for (source_x, source_y), target_x in zip(source_loader, target_loader):
    # Classification loss (source only)
    class_out, domain_out_s = model(source_x, alpha)
    class_loss = F.cross_entropy(class_out, source_y)
    
    # Domain loss (both domains)
    _, domain_out_t = model(target_x, alpha)
    domain_loss = (F.cross_entropy(domain_out_s, source_labels) + 
                   F.cross_entropy(domain_out_t, target_labels))
    
    loss = class_loss + domain_loss
```

### Approach 3: Maximum Mean Discrepancy (MMD)
```python
def mmd_loss(source_features, target_features):
    """Minimize distribution difference"""
    source_mean = source_features.mean(0)
    target_mean = target_features.mean(0)
    return ((source_mean - target_mean) ** 2).sum()

# Add MMD to training
loss = classification_loss + lambda_mmd * mmd_loss(source_feat, target_feat)
```

### Approach 4: Self-Training with Pseudo-Labels
```python
# Step 1: Train on source
model.fit(source_x, source_y)

# Step 2: Generate pseudo-labels for target
pseudo_labels = model.predict(target_x)
confidence = model.predict_proba(target_x).max(axis=1)

# Step 3: Keep high-confidence predictions
mask = confidence > threshold
pseudo_x = target_x[mask]
pseudo_y = pseudo_labels[mask]

# Step 4: Retrain on source + pseudo-labeled target
model.fit(concat(source_x, pseudo_x), concat(source_y, pseudo_y))
```

---

## Question 7: How can you measure the similarity between the source and target domains?
- [ ] Done

### Similarity Metrics

| Metric | What It Measures |
|--------|------------------|
| **MMD** | Distribution difference |
| **A-distance** | Domain discriminability |
| **KL Divergence** | Distribution divergence |
| **Feature correlation** | Representation alignment |
| **Task similarity** | Label space overlap |

### Maximum Mean Discrepancy (MMD)
```python
def compute_mmd(source_features, target_features, kernel='rbf'):
    """
    MMD = ||μ_s - μ_t||² in RKHS
    Lower MMD = More similar domains
    """
    if kernel == 'linear':
        # Linear kernel MMD
        mean_s = source_features.mean(0)
        mean_t = target_features.mean(0)
        mmd = ((mean_s - mean_t) ** 2).sum()
    else:
        # RBF kernel MMD
        K_ss = rbf_kernel(source_features, source_features)
        K_tt = rbf_kernel(target_features, target_features)
        K_st = rbf_kernel(source_features, target_features)
        mmd = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
    
    return mmd.item()
```

### A-distance (Proxy)
```python
def compute_a_distance(source_features, target_features):
    """
    Train classifier to distinguish domains
    A-distance = 2 * (1 - 2 * error)
    
    A-distance ≈ 0: Domains indistinguishable (similar)
    A-distance ≈ 2: Domains perfectly separable (different)
    """
    # Create domain labels
    X = np.vstack([source_features, target_features])
    y = np.array([0] * len(source_features) + [1] * len(target_features))
    
    # Train domain classifier
    clf = LogisticRegression()
    clf.fit(X, y)
    
    # Compute error
    error = 1 - clf.score(X, y)
    a_distance = 2 * (1 - 2 * error)
    
    return a_distance
```

### Feature Visualization
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_domain_shift(source_features, target_features):
    combined = np.vstack([source_features, target_features])
    labels = ['source'] * len(source_features) + ['target'] * len(target_features)
    
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(combined)
    
    plt.scatter(embedded[:len(source_features), 0], 
                embedded[:len(source_features), 1], 
                label='Source', alpha=0.5)
    plt.scatter(embedded[len(source_features):, 0], 
                embedded[len(source_features):, 1], 
                label='Target', alpha=0.5)
    plt.legend()
    plt.title('Domain Similarity Visualization')
```

### Practical Pipeline
```python
def assess_transfer_potential(source_model, source_data, target_data):
    # Extract features using source model
    source_features = extract_features(source_model, source_data)
    target_features = extract_features(source_model, target_data)
    
    # Compute metrics
    mmd = compute_mmd(source_features, target_features)
    a_dist = compute_a_distance(source_features, target_features)
    
    print(f"MMD: {mmd:.4f} (lower = more similar)")
    print(f"A-distance: {a_dist:.4f} (closer to 0 = more similar)")
    
    # Recommendation
    if a_dist < 0.5:
        return "High transfer potential - use pre-trained features directly"
    elif a_dist < 1.0:
        return "Moderate transfer potential - fine-tune carefully"
    else:
        return "Low transfer potential - consider domain adaptation"
```

---

## Question 8: How do GANs contribute to transfer learning in unsupervised scenarios?
- [ ] Done

### GAN-based Domain Adaptation

**Core Idea:** Use GANs to:
1. Generate target-like images from source domain
2. Align feature distributions across domains
3. Create synthetic training data

### Approach 1: Image-to-Image Translation
```
Source Domain Images → GAN → Target Domain Style
Train classifier on translated images
```

```python
# CycleGAN for unpaired domain translation
# Source: Synthetic images, Target: Real images

# Generator: Synthetic → Real-looking
G_syn2real = CycleGAN_Generator()

# Translate source data
translated_source = G_syn2real(source_images)

# Train classifier on translated data
classifier.fit(translated_source, source_labels)

# Apply to real target domain
predictions = classifier.predict(real_images)
```

### Approach 2: Feature-level Adversarial Adaptation
```python
class FeatureAdaptiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = Encoder()
        self.classifier = Classifier()
        self.discriminator = Discriminator()  # Distinguishes source/target features
    
    def train_step(self, source_x, source_y, target_x):
        # Extract features
        source_feat = self.feature_extractor(source_x)
        target_feat = self.feature_extractor(target_x)
        
        # Classification loss (source)
        class_loss = F.cross_entropy(self.classifier(source_feat), source_y)
        
        # Adversarial loss: Fool discriminator
        # Discriminator tries to tell source from target
        real_labels = torch.ones(len(source_feat))
        fake_labels = torch.zeros(len(target_feat))
        
        d_source = self.discriminator(source_feat)
        d_target = self.discriminator(target_feat)
        
        # Feature extractor wants to confuse discriminator
        adv_loss = -torch.log(d_target).mean()  # Make target look like source
        
        total_loss = class_loss + lambda_adv * adv_loss
        return total_loss
```

### Approach 3: Data Augmentation via GANs
```python
# Generate synthetic labeled data
class ConditionalGAN:
    def generate(self, class_label, num_samples):
        z = torch.randn(num_samples, latent_dim)
        condition = one_hot(class_label)
        fake_images = self.generator(z, condition)
        return fake_images

# Augment training data
for class_id in range(num_classes):
    real_samples = dataset[class_id]
    synthetic_samples = gan.generate(class_id, augment_count)
    augmented_data[class_id] = concat(real_samples, synthetic_samples)
```

### Benefits of GAN-based Transfer

| Benefit | Description |
|---------|-------------|
| **Unsupervised** | No labels needed in target domain |
| **Visual adaptation** | Bridge appearance gap |
| **Data generation** | Create more training samples |
| **Feature alignment** | Domain-invariant representations |

### Example: Sim-to-Real Transfer
```
1. Train GAN: Simulated images ↔ Real images
2. Translate simulated (labeled) to real-looking
3. Train task model on translated images
4. Deploy on real target domain
```

---
