# Transfer Learning Interview Questions - Theory Questions

---

## Question 1: What is transfer learning and how does it differ from traditional machine learning?
- [ ] Done

### Definition
Transfer learning is a machine learning technique where a model trained on one task (source) is reused as the starting point for a model on a different task (target).

### Core Concept
```
Traditional ML: Train from scratch on target task data
Transfer Learning: Leverage knowledge from source task → Apply to target task
```

### Key Differences

| Aspect | Traditional ML | Transfer Learning |
|--------|---------------|-------------------|
| **Starting Point** | Random initialization | Pre-trained weights |
| **Data Required** | Large labeled dataset | Can work with small data |
| **Training Time** | Long (full training) | Short (fine-tuning) |
| **Compute Cost** | High | Lower |
| **Knowledge** | Task-specific only | Transfers general features |

### Why Transfer Learning Works
- Early layers learn general features (edges, textures)
- Later layers learn task-specific features
- General features are often reusable

### Mathematical View
Traditional: $\theta^* = \arg\min_\theta \mathcal{L}(\theta; D_{target})$ starting from random $\theta_0$

Transfer: $\theta^* = \arg\min_\theta \mathcal{L}(\theta; D_{target})$ starting from pre-trained $\theta_{source}$

### Example
```python
# Traditional: Train from scratch
model = ResNet50(weights=None)
model.fit(small_dataset)  # Poor performance

# Transfer Learning: Use pre-trained
model = ResNet50(weights='imagenet')
model.fit(small_dataset)  # Better performance
```

---

## Question 2: Explain the concept of domain and task in the context of transfer learning
- [ ] Done

### Domain Definition
A domain $\mathcal{D}$ consists of:
- **Feature space** $\mathcal{X}$: Set of all possible features
- **Marginal distribution** $P(X)$: How features are distributed

$$\mathcal{D} = \{\mathcal{X}, P(X)\}$$

### Task Definition
A task $\mathcal{T}$ consists of:
- **Label space** $\mathcal{Y}$: Set of all possible labels
- **Predictive function** $f(x)$: Mapping from features to labels (learned from data)

$$\mathcal{T} = \{\mathcal{Y}, f(x)\}$$

### Transfer Learning Scenarios

| Scenario | Domain | Task | Example |
|----------|--------|------|---------|
| **Same domain, same task** | $\mathcal{D}_S = \mathcal{D}_T$ | $\mathcal{T}_S = \mathcal{T}_T$ | Regular ML (no transfer) |
| **Same domain, different task** | $\mathcal{D}_S = \mathcal{D}_T$ | $\mathcal{T}_S \neq \mathcal{T}_T$ | Multi-task learning |
| **Different domain, same task** | $\mathcal{D}_S \neq \mathcal{D}_T$ | $\mathcal{T}_S = \mathcal{T}_T$ | Domain adaptation |
| **Different domain, different task** | $\mathcal{D}_S \neq \mathcal{D}_T$ | $\mathcal{T}_S \neq \mathcal{T}_T$ | Full transfer learning |

### Examples

**Domain Shift (Same Task):**
- Source: Daytime driving images → Car detection
- Target: Nighttime driving images → Car detection
- Same task, different $P(X)$

**Task Shift (Same Domain):**
- Source: ImageNet images → 1000-class classification
- Target: ImageNet images → Object detection
- Same domain, different $\mathcal{Y}$ and $f$

---

## Question 3: What are the benefits of using transfer learning techniques?
- [ ] Done

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Reduced Data Requirements** | Learn effectively with small datasets |
| **Faster Training** | Start from good initialization |
| **Better Performance** | Leverage learned representations |
| **Lower Compute Cost** | Less training from scratch |
| **Knowledge Reuse** | Don't reinvent the wheel |

### Reduced Data Requirements
```
Without Transfer:  Need 1M+ labeled images
With Transfer:     Need 1K-10K labeled images for similar performance
```

Pre-trained features capture general patterns useful across tasks.

### Faster Convergence
```python
# Training curves comparison
Epochs to converge:
- From scratch: 100+ epochs
- Transfer learning: 10-20 epochs
```

### Better Generalization
- Pre-trained models learned robust features
- Less prone to overfitting on small datasets
- Regularization effect from prior knowledge

### Practical Impact

| Metric | Without Transfer | With Transfer |
|--------|-----------------|---------------|
| Training time | Days/Weeks | Hours |
| GPU hours | 1000s | 10s-100s |
| Data needed | Millions | Thousands |
| Final accuracy | Lower | Higher |

### When Benefits Are Greatest
1. Target dataset is small
2. Source and target are related
3. Compute resources limited
4. Time constraints exist

---

## Question 4: Describe the difference between transductive and inductive transfer learning
- [ ] Done

### Inductive Transfer Learning
**Definition:** Source and target tasks are different, but some labeled data exists in target domain.

**Characteristics:**
- $\mathcal{T}_S \neq \mathcal{T}_T$ (different tasks)
- Target domain has labeled data
- Goal: Learn target task using source knowledge

**Examples:**
- ImageNet classification → Medical image classification
- English NER → English Sentiment Analysis

```
Source: (X_s, Y_s) → Task_s
Target: (X_t, Y_t) → Task_t  [Has labels]
```

### Transductive Transfer Learning
**Definition:** Source and target tasks are the same, but domains differ. No labeled data in target domain.

**Characteristics:**
- $\mathcal{T}_S = \mathcal{T}_T$ (same task)
- $\mathcal{D}_S \neq \mathcal{D}_T$ (different domains)
- No labels in target domain

**Examples:**
- Product reviews → Movie reviews (sentiment)
- Synthetic images → Real images (same classes)

```
Source: (X_s, Y_s) → Task
Target: (X_t, ?) → Task  [No labels]
```

### Comparison

| Aspect | Inductive | Transductive |
|--------|-----------|--------------|
| Tasks | Different | Same |
| Target labels | Available | Not available |
| Domain shift | May or may not | Yes |
| Approach | Fine-tuning | Domain adaptation |
| Methods | Feature extraction, fine-tuning | DANN, MMD, adversarial |

### Visual Representation
```
Inductive:
Source Domain → [Task A] 
                    ↓ transfer features
Target Domain → [Task B] with labels

Transductive:
Source Domain → [Task A] with labels
                    ↓ adapt domain
Target Domain → [Task A] without labels
```

---

## Question 5: Explain the concept of 'negative transfer'. When can it occur?
- [ ] Done

### Definition
Negative transfer occurs when transfer learning hurts performance on the target task compared to training from scratch.

$$\text{Accuracy}_{transfer} < \text{Accuracy}_{from\_scratch}$$

### Causes of Negative Transfer

| Cause | Description |
|-------|-------------|
| **Domain mismatch** | Source and target too different |
| **Task mismatch** | Tasks require conflicting features |
| **Feature interference** | Source features mislead target |
| **Overwriting** | Fine-tuning destroys useful features |

### When It Occurs

**1. Unrelated Domains:**
```
Source: Natural images (ImageNet)
Target: Medical CT scans
→ Features learned may not transfer well
```

**2. Conflicting Objectives:**
```
Source: Classify by texture
Target: Classify by shape
→ Transferred features may be counterproductive
```

**3. Over-fine-tuning:**
```
Too much fine-tuning on small target data
→ Catastrophic forgetting of useful source features
```

### Detecting Negative Transfer
```python
# Compare performance
acc_scratch = train_from_scratch(target_data)
acc_transfer = transfer_and_finetune(pretrained, target_data)

if acc_transfer < acc_scratch:
    print("Negative transfer detected!")
```

### Mitigation Strategies

| Strategy | Description |
|----------|-------------|
| **Measure domain similarity** | Only transfer if domains are related |
| **Gradual fine-tuning** | Lower learning rates, freeze early layers |
| **Selective transfer** | Transfer only relevant layers |
| **Multi-source transfer** | Combine multiple sources |
| **Regularization** | Prevent diverging too far from source |

### Example
```python
# Avoid negative transfer
if domain_similarity(source, target) > threshold:
    model = transfer_learning(source_model, target_data)
else:
    model = train_from_scratch(target_data)
```

---

## Question 6: What are feature extractors in the context of transfer learning?
- [ ] Done

### Definition
Feature extractors are the parts of a pre-trained model (typically early/middle layers) used to transform raw inputs into meaningful representations, without being modified during transfer.

### How It Works
```
Pre-trained Model:
[Input] → [Conv1] → [Conv2] → ... → [ConvN] → [FC] → [Output]
          |_________Feature Extractor_________|    |__Classifier__|

Transfer Learning:
[New Input] → [Frozen Feature Extractor] → [New Classifier] → [New Output]
```

### Implementation
```python
import torch
import torchvision.models as models

# Load pre-trained model
resnet = models.resnet50(pretrained=True)

# Use as feature extractor (remove classifier)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

# Freeze weights
for param in feature_extractor.parameters():
    param.requires_grad = False

# Extract features
def get_features(images):
    with torch.no_grad():
        features = feature_extractor(images)
    return features.flatten(1)

# Train new classifier on extracted features
classifier = torch.nn.Linear(2048, num_classes)
```

### Why Use Feature Extractors?

| Advantage | Explanation |
|-----------|-------------|
| **Fast** | No backprop through frozen layers |
| **Memory efficient** | Only store new classifier gradients |
| **Prevents overfitting** | Fewer trainable parameters |
| **Reusable** | Extract features once, train many classifiers |

### Feature Quality by Layer

| Layer Depth | Feature Type | Transferability |
|-------------|--------------|-----------------|
| Early (1-3) | Edges, colors | Highly transferable |
| Middle (4-7) | Textures, patterns | Moderately transferable |
| Late (8+) | Task-specific | Less transferable |

### When to Use
- Small target dataset
- Limited compute resources
- Source and target domains similar
- Quick prototyping needed

---

## Question 7: Describe the process of fine-tuning a pre-trained neural network
- [ ] Done

### Definition
Fine-tuning is the process of taking a pre-trained model and further training it (partially or fully) on a new target dataset.

### Step-by-Step Process

**Step 1: Load Pre-trained Model**
```python
from torchvision import models

model = models.resnet50(pretrained=True)
```

**Step 2: Modify Final Layers**
```python
# Replace classifier for new task
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

**Step 3: Decide Freezing Strategy**
```python
# Option A: Freeze all except final layer
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Option B: Freeze early layers only
for name, param in model.named_parameters():
    if 'layer1' in name or 'layer2' in name:
        param.requires_grad = False
```

**Step 4: Set Lower Learning Rate**
```python
# Use smaller LR for pre-trained layers
optimizer = optim.Adam([
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-3}  # Higher for new layer
])
```

**Step 5: Train on Target Data**
```python
for epoch in range(num_epochs):
    for images, labels in dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Fine-tuning Strategies

| Strategy | Layers Trained | When to Use |
|----------|---------------|-------------|
| **Feature extraction** | Only new head | Very small data |
| **Fine-tune top layers** | Last few + head | Small data |
| **Fine-tune all** | All layers | Moderate data |
| **Gradual unfreezing** | Progressively more | Best results |

### Gradual Unfreezing
```python
# Epoch 1-5: Only classifier
# Epoch 6-10: Unfreeze layer4
# Epoch 11+: Unfreeze layer3
```

### Best Practices
1. Start with frozen backbone
2. Use lower learning rates (10-100x smaller)
3. Use discriminative learning rates
4. Monitor for overfitting
5. Use data augmentation

---

## Question 8: What is one-shot learning and how does it relate to transfer learning?
- [ ] Done

### Definition
One-shot learning is the ability to learn a new class from just one (or very few) example(s).

### The Challenge
```
Traditional ML: Need thousands of examples per class
One-shot: Learn from single example
```

### How It Relates to Transfer Learning
One-shot learning requires strong transfer of prior knowledge:
- Pre-training provides general feature representations
- These features must generalize to recognize new classes
- Without transfer, one-shot is impossible

### Common Approaches

**1. Siamese Networks**
```python
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared feature extractor (pre-trained)
        self.encoder = resnet_encoder()
    
    def forward(self, x1, x2):
        # Extract features
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        
        # Compare (e.g., L1 distance)
        distance = torch.abs(f1 - f2)
        return distance

# Training: Learn to compare
# Inference: Compare query with single example
```

**2. Prototypical Networks**
```
1. Compute prototype (mean embedding) for each class
2. Classify query by distance to prototypes
3. Works with 1 example per class
```

**3. Matching Networks**
```
1. Encode support set (few examples)
2. Encode query
3. Weighted sum of support labels by attention
```

### Transfer Learning Connection

| Aspect | Role of Transfer |
|--------|------------------|
| Feature quality | Pre-trained features capture semantic similarity |
| Generalization | Learn "how to learn" from source tasks |
| Embedding space | Meaningful distances require good representations |

### Example Use Case
```
Face verification:
- Train Siamese network on millions of face pairs
- At test time: Verify new person with just 1 photo
- Transfer: General face features → Specific person recognition
```

---

## Question 9: Explain the differences between few-shot learning and zero-shot learning
- [ ] Done

### Few-Shot Learning
**Definition:** Learn to classify new classes with only a few labeled examples (typically 1-5).

**Setup:**
```
Support Set: K examples per class (K-shot)
Query Set: Examples to classify
N-way: N classes to distinguish
```

**Example:**
```
5-way 1-shot: 5 new classes, 1 example each
Task: Classify query image into one of 5 classes
```

### Zero-Shot Learning
**Definition:** Classify classes never seen during training, using auxiliary information (attributes, descriptions).

**Setup:**
```
Seen Classes: Classes with training examples
Unseen Classes: No examples, only semantic descriptions
```

**Requires:**
- Semantic embeddings (word vectors, attributes)
- Mapping from visual → semantic space

### Key Differences

| Aspect | Few-Shot | Zero-Shot |
|--------|----------|-----------|
| **Examples at test time** | Few (1-5) | None |
| **Auxiliary info** | Not required | Required |
| **Learning** | From examples | From descriptions |
| **Approach** | Metric learning | Semantic embedding |

### Few-Shot Example
```python
# Prototypical Network for 5-way 1-shot
support_images = [img1, img2, img3, img4, img5]  # 1 per class
support_labels = [0, 1, 2, 3, 4]

# Compute prototypes
prototypes = []
for class_id in range(5):
    class_embedding = encoder(support_images[class_id])
    prototypes.append(class_embedding)

# Classify query
query_embedding = encoder(query_image)
distances = [dist(query_embedding, p) for p in prototypes]
prediction = argmin(distances)
```

### Zero-Shot Example
```python
# Zero-shot with attribute embeddings
class_attributes = {
    'zebra': [has_stripes=1, four_legs=1, horse_like=1],
    'panda': [has_stripes=0, four_legs=1, black_white=1]
}

# Train: Map images to attribute space
visual_to_semantic = train_mapping(seen_images, seen_attributes)

# Test: Predict unseen class
image_attributes = visual_to_semantic(zebra_image)
prediction = nearest_class(image_attributes, class_attributes)
```

### Hybrid: Generalized Zero-Shot
```
Test classes include both seen and unseen
More realistic but harder
```

---

## Question 10: What are the common pre-trained models available for use in transfer learning?
- [ ] Done

### Computer Vision Models

| Model | Dataset | Parameters | Use Case |
|-------|---------|------------|----------|
| **ResNet-50/101/152** | ImageNet | 25-60M | General image classification |
| **VGG-16/19** | ImageNet | 138M | Feature extraction |
| **EfficientNet** | ImageNet | 5-66M | Efficient, scalable |
| **Vision Transformer (ViT)** | ImageNet | 86-632M | State-of-the-art |
| **CLIP** | 400M image-text pairs | 400M | Multi-modal, zero-shot |

### NLP Models

| Model | Training Data | Parameters | Use Case |
|-------|--------------|------------|----------|
| **BERT** | Wikipedia, Books | 110M-340M | Understanding tasks |
| **GPT-2/3/4** | Web text | 124M-175B | Text generation |
| **RoBERTa** | More data than BERT | 125M-355M | Improved BERT |
| **T5** | C4 dataset | 60M-11B | Text-to-text tasks |
| **LLaMA** | Various | 7B-70B | Open-source LLM |

### Speech Models

| Model | Use Case |
|-------|----------|
| **Wav2Vec 2.0** | Speech recognition |
| **HuBERT** | Self-supervised speech |
| **Whisper** | Speech-to-text |

### Loading Pre-trained Models

```python
# PyTorch - Vision
from torchvision import models
resnet = models.resnet50(pretrained=True)
efficientnet = models.efficientnet_b0(pretrained=True)

# Hugging Face - NLP
from transformers import BertModel, GPT2Model
bert = BertModel.from_pretrained('bert-base-uncased')
gpt2 = GPT2Model.from_pretrained('gpt2')

# TensorFlow/Keras
from tensorflow.keras.applications import ResNet50, VGG16
resnet = ResNet50(weights='imagenet')
vgg = VGG16(weights='imagenet')
```

### Choosing a Model

| Criterion | Recommendation |
|-----------|---------------|
| Limited compute | EfficientNet, MobileNet |
| Best accuracy | ViT, EfficientNet-L2 |
| General NLP | BERT, RoBERTa |
| Text generation | GPT models |
| Multi-modal | CLIP |

---

## Question 11: Describe how you would approach transfer learning with an imbalanced dataset
- [ ] Done

### The Challenge
- Transfer learning + class imbalance
- Pre-trained model biased toward majority classes
- Fine-tuning may worsen imbalance

### Strategy Overview
```
1. Select appropriate pre-trained model
2. Apply data-level techniques
3. Apply algorithm-level techniques
4. Fine-tune carefully
5. Evaluate properly
```

### Data-Level Approaches

**1. Oversampling Minority Classes**
```python
from imblearn.over_sampling import SMOTE, RandomOverSampler

# After feature extraction
features = extract_features(pretrained_model, images)
oversampler = SMOTE()
X_balanced, y_balanced = oversampler.fit_resample(features, labels)
```

**2. Data Augmentation for Minority**
```python
# More augmentation for minority classes
minority_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.RandomAffine(0, translate=(0.1, 0.1))
])
```

**3. Class-Balanced Sampling**
```python
from torch.utils.data import WeightedRandomSampler

class_counts = [count_class_0, count_class_1, ...]
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = weights[labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
dataloader = DataLoader(dataset, sampler=sampler)
```

### Algorithm-Level Approaches

**1. Weighted Loss Function**
```python
# Class weights inversely proportional to frequency
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

**2. Focal Loss**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

### Fine-tuning Strategy
```python
# Two-stage fine-tuning
# Stage 1: Balance using feature extractor
features = frozen_model(images)
classifier.fit(features, labels, class_weight='balanced')

# Stage 2: Fine-tune with balanced batches
for epoch in range(epochs):
    for batch in balanced_dataloader:
        loss = focal_loss(model(batch), labels)
        loss.backward()
```

### Evaluation
- Use balanced metrics: F1-score, balanced accuracy
- Report per-class metrics
- Use stratified cross-validation

---

## Question 12: What are some challenges when applying transfer learning to sequential data?
- [ ] Done

### Challenges

| Challenge | Description |
|-----------|-------------|
| **Temporal dependencies** | Sequences have time-based structure |
| **Variable length** | Different sequence lengths |
| **Domain shift** | Distribution changes over time |
| **Feature representation** | What features to transfer? |
| **Architecture mismatch** | Source/target may need different models |

### Time Series Challenges

**1. Non-stationarity**
```
Source: Stationary financial data from 2010-2015
Target: Non-stationary data from 2020
→ Distribution shift over time
```

**2. Different Sampling Rates**
```
Source: Sensor data at 100Hz
Target: Sensor data at 50Hz
→ Temporal feature mismatch
```

**3. Different Sequence Lengths**
```
Source: Fixed 24-hour windows
Target: Variable-length recordings
→ Model architecture incompatibility
```

### Text/NLP Challenges

**1. Domain Vocabulary**
```
Source: General English (Wikipedia)
Target: Medical text
→ Many unseen tokens, domain-specific terms
```

**2. Contextual Differences**
```
Same word, different meaning:
"Python" in programming vs "Python" in biology
```

### Solutions

**1. Domain Adaptation for Sequences**
```python
# Adversarial domain adaptation for time series
class DomainAdaptiveModel(nn.Module):
    def __init__(self):
        self.feature_extractor = LSTM(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.domain_discriminator = nn.Linear(hidden_dim, 2)
    
    def forward(self, x, alpha):
        features = self.feature_extractor(x)
        class_output = self.classifier(features)
        # Gradient reversal for domain adaptation
        reversed_features = GradientReversal(features, alpha)
        domain_output = self.domain_discriminator(reversed_features)
        return class_output, domain_output
```

**2. Pre-training on Related Tasks**
```
Time series: Pre-train on forecasting → Fine-tune for classification
Text: Pre-train on language modeling → Fine-tune for NER
```

**3. Handling Variable Lengths**
```python
# Use attention pooling
class SequenceClassifier(nn.Module):
    def __init__(self, pretrained_encoder):
        self.encoder = pretrained_encoder
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, lengths):
        encoded = self.encoder(x)  # (seq_len, batch, hidden)
        # Attention pooling handles variable lengths
        pooled, _ = self.attention(encoded, encoded, encoded)
        return self.classifier(pooled.mean(0))
```

---

## Question 13: Can you explain how knowledge distillation works in the context of transfer learning?
- [ ] Done

### Definition
Knowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model by training the student to mimic the teacher's outputs.

### Core Concept
```
Teacher (large, accurate) → Soft labels → Student (small, efficient)
```

### How It Works

**1. Soft Labels vs Hard Labels**
```
Hard label: [0, 1, 0]  (one-hot)
Soft label: [0.1, 0.7, 0.2]  (teacher's softmax output)

Soft labels contain more information:
- "Cat is somewhat similar to dog"
- "Cat is very different from car"
```

**2. Temperature Scaling**
```python
def softmax_with_temperature(logits, T):
    return F.softmax(logits / T, dim=-1)

# T=1: Sharp distribution
# T>1: Softer distribution (more information)
```

### Distillation Loss
$$\mathcal{L} = \alpha \cdot \mathcal{L}_{CE}(y, \sigma(z_s)) + (1-\alpha) \cdot T^2 \cdot \mathcal{L}_{KL}(\sigma(z_t/T), \sigma(z_s/T))$$

Where:
- $z_s$: Student logits
- $z_t$: Teacher logits
- $T$: Temperature
- $\alpha$: Balancing weight

### Implementation
```python
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4, alpha=0.5):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        
    def forward(self, student_logits, teacher_logits, labels):
        # Hard label loss
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Soft label loss (KL divergence)
        soft_student = F.log_softmax(student_logits / self.T, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.T, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        
        # Combined loss
        loss = self.alpha * hard_loss + (1 - self.alpha) * (self.T ** 2) * soft_loss
        return loss

# Training
teacher.eval()
for x, y in dataloader:
    with torch.no_grad():
        teacher_logits = teacher(x)
    student_logits = student(x)
    loss = distillation_loss(student_logits, teacher_logits, y)
    loss.backward()
```

### Benefits in Transfer Learning

| Benefit | Explanation |
|---------|-------------|
| **Model compression** | Deploy smaller models |
| **Faster inference** | Reduced computation |
| **Knowledge transfer** | Dark knowledge from teacher |
| **Regularization** | Soft labels prevent overfitting |

### Variants
- **Feature distillation**: Match intermediate representations
- **Attention distillation**: Match attention maps
- **Self-distillation**: Teacher and student same architecture

---

## Question 14: Explain the concept of meta-learning and how it applies to transfer learning
- [ ] Done

### Definition
Meta-learning ("learning to learn") trains models to adapt quickly to new tasks using experience from many related tasks.

### Core Idea
```
Traditional ML: Learn one task well
Meta-learning: Learn how to learn new tasks efficiently
```

### How It Relates to Transfer Learning

| Aspect | Traditional Transfer | Meta-Learning |
|--------|---------------------|---------------|
| **Source** | One task | Many tasks |
| **Transfer what** | Features/weights | Learning algorithm |
| **Adaptation** | Fine-tuning | Few gradient steps |
| **Goal** | Good on target | Fast adaptation to any new task |

### Meta-Learning Framework
```
Meta-train: Learn from distribution of tasks
    Task 1: (support set, query set)
    Task 2: (support set, query set)
    ...
    
Meta-test: Adapt to new task with few examples
    New Task: (few examples) → Good performance
```

### MAML (Model-Agnostic Meta-Learning)
```python
def maml_train(model, task_distribution, inner_lr, outer_lr, K=5):
    outer_optimizer = Adam(model.parameters(), lr=outer_lr)
    
    for iteration in range(num_iterations):
        # Sample batch of tasks
        tasks = task_distribution.sample(batch_size)
        
        meta_loss = 0
        for task in tasks:
            # Clone model for inner loop
            model_copy = clone(model)
            
            # Inner loop: Adapt to task with K steps
            support_x, support_y = task.support()
            for _ in range(K):
                loss = F.cross_entropy(model_copy(support_x), support_y)
                grads = torch.autograd.grad(loss, model_copy.parameters())
                for param, grad in zip(model_copy.parameters(), grads):
                    param.data -= inner_lr * grad
            
            # Evaluate adapted model on query set
            query_x, query_y = task.query()
            meta_loss += F.cross_entropy(model_copy(query_x), query_y)
        
        # Outer loop: Update original model
        outer_optimizer.zero_grad()
        (meta_loss / len(tasks)).backward()
        outer_optimizer.step()
```

### Why Meta-Learning Works
- Learns good initialization for fast adaptation
- Captures task structure across many tasks
- Parameters become "ready to learn"

### Applications
- Few-shot image classification
- Quick robot adaptation
- Personalization
- Drug discovery

---

## Question 15: What is the role of attention mechanisms in transferring knowledge between tasks?
- [ ] Done

### Attention in Transfer Learning
Attention mechanisms identify which parts of input or which features are most relevant for the target task.

### How Attention Helps Transfer

| Role | Description |
|------|-------------|
| **Feature selection** | Focus on transferable features |
| **Alignment** | Match source/target representations |
| **Interpretability** | See what transfers |
| **Flexibility** | Adapt to different tasks dynamically |

### Self-Attention for Transfer (Transformers)
```python
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, x):
        # Self-attention captures relationships
        # These relationships transfer across tasks
        return self.transformer(x)
```

### Cross-Attention for Domain Adaptation
```python
class CrossDomainAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
    
    def forward(self, target_features, source_features):
        # Target queries attend to source keys
        Q = self.query(target_features)
        K = self.key(source_features)
        V = self.value(source_features)
        
        attention = F.softmax(Q @ K.T / sqrt(d_model), dim=-1)
        transferred = attention @ V
        
        return transferred
```

### Task-Specific Attention
```python
class TaskAdaptiveAttention(nn.Module):
    """Learn what to attend to for each task"""
    def __init__(self, feature_dim, num_tasks):
        super().__init__()
        self.task_embeddings = nn.Embedding(num_tasks, feature_dim)
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=4)
    
    def forward(self, features, task_id):
        task_query = self.task_embeddings(task_id)
        # Task-specific attention over features
        attended, weights = self.attention(task_query, features, features)
        return attended, weights
```

### Benefits in Transfer Learning
1. **Selective transfer**: Only relevant knowledge flows
2. **Dynamic**: Different attention for different instances
3. **Pre-trained attention**: Models like BERT transfer attention patterns
4. **Multi-task**: Shared attention with task-specific heads

---

## Question 16: How does transfer learning relate to reinforcement learning?
- [ ] Done

### Transfer in RL: The Challenge
RL agents typically learn from scratch for each new environment, which is:
- Sample inefficient
- Time consuming
- Computationally expensive

### Types of Transfer in RL

| Type | What Transfers |
|------|----------------|
| **Policy transfer** | Learned policy π(a|s) |
| **Value transfer** | Value function V(s) or Q(s,a) |
| **Model transfer** | Dynamics model P(s'|s,a) |
| **Representation transfer** | State/feature representations |
| **Skill transfer** | Reusable sub-policies (options) |

### Approaches

**1. Pre-trained Representations**
```python
# Use pre-trained encoder for state representation
class RLAgent:
    def __init__(self):
        # Pre-trained on similar tasks
        self.encoder = pretrained_state_encoder()
        self.encoder.freeze()  # Don't update
        
        # Learn policy on frozen features
        self.policy = PolicyNetwork(feature_dim, action_dim)
    
    def act(self, state):
        features = self.encoder(state)
        return self.policy(features)
```

**2. Sim-to-Real Transfer**
```
1. Train in simulation (cheap, safe)
2. Transfer to real world (expensive, risky)
Challenges: Domain gap between sim and real
```

**3. Multi-task RL**
```python
# Share representations across tasks
class MultiTaskAgent:
    def __init__(self, num_tasks):
        self.shared_encoder = StateEncoder()
        self.task_heads = nn.ModuleList([
            PolicyHead() for _ in range(num_tasks)
        ])
    
    def act(self, state, task_id):
        features = self.shared_encoder(state)
        return self.task_heads[task_id](features)
```

**4. Skill/Option Transfer**
```
Source: Learn primitive skills (walk, grasp, reach)
Target: Compose skills for new tasks (pick and place)
```

### Challenges in RL Transfer

| Challenge | Issue |
|-----------|-------|
| **Non-stationarity** | Environment/task may differ |
| **Reward structure** | Different reward functions |
| **Action space** | Actions may differ |
| **Negative transfer** | Source policy may be harmful |

### Example: Game Transfer
```
Source: Pong (paddle, ball dynamics)
Target: Breakout (similar dynamics, different goal)
Transfer: Visual features, physics intuition
```

---

## Question 17: Describe a scenario where transfer learning could significantly reduce the need for labeled data in a mobile app
- [ ] Done

### Scenario: Pet Breed Classification App

**Problem:**
- Mobile app to identify dog/cat breeds from photos
- 200+ breeds to classify
- Collecting labeled data is expensive
- Users want accurate predictions quickly

### Solution with Transfer Learning

**Step 1: Use Pre-trained Model**
```python
# MobileNetV2: Efficient for mobile, trained on ImageNet
from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```

**Step 2: Add Custom Classifier**
```python
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_breeds, activation='softmax')
])

# Freeze base model initially
base_model.trainable = False
```

**Step 3: Train on Small Dataset**
```
Without transfer: Need ~1000 images per breed = 200,000 images
With transfer: Need ~50-100 images per breed = 10,000-20,000 images

Data reduction: 10-20x fewer labeled images needed
```

### Data Collection Strategy
```python
# Minimal data collection
data_sources = {
    'manual_labeling': 20_per_class,    # Core accuracy
    'user_corrections': ongoing,          # Improves over time
    'web_scraping': 30_per_class,        # Augment training
}
```

### Mobile Optimization
```python
# Quantize for mobile deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Model size: ~10MB (vs 100MB+ from scratch)
# Inference: <100ms on mobile
```

### Impact Summary

| Metric | Without Transfer | With Transfer |
|--------|-----------------|---------------|
| Labeled images needed | 200,000+ | 10,000-20,000 |
| Training time | Weeks | Hours |
| Model accuracy | ~60% | ~90% |
| Development cost | High | Low |

---

## Question 18: What are the potential risks of bias when using transfer learning, particularly with pre-trained models?
- [ ] Done

### Sources of Bias

| Source | Description |
|--------|-------------|
| **Training data bias** | Pre-trained on biased datasets |
| **Representation bias** | Certain groups underrepresented |
| **Label bias** | Biased annotations in source |
| **Algorithmic bias** | Model architecture preferences |

### ImageNet Bias Examples
```
- Geographic bias: Mostly Western objects/scenes
- Demographic bias: Certain ethnicities underrepresented
- Object bias: Common objects overrepresented
```

### NLP Pre-trained Model Bias
```
BERT/GPT trained on web text may encode:
- Gender stereotypes ("nurse" → female)
- Racial biases
- Cultural biases toward English-speaking countries
```

### How Bias Transfers

```
Biased Pre-trained Model
        ↓
    Fine-tuning
        ↓
Target Model (inherits bias)
        ↓
Biased Predictions on Target Task
```

### Real-World Risks

| Domain | Potential Harm |
|--------|----------------|
| **Healthcare** | Misdiagnosis for underrepresented groups |
| **Hiring** | Discrimination in resume screening |
| **Criminal justice** | Biased risk assessment |
| **Finance** | Unfair loan decisions |

### Mitigation Strategies

**1. Audit Pre-trained Models**
```python
def audit_bias(model, protected_attributes):
    results = {}
    for attribute in protected_attributes:
        # Test performance across groups
        for group in attribute.groups:
            accuracy = evaluate(model, group.data)
            results[f'{attribute}_{group}'] = accuracy
    
    # Check for disparities
    report_disparities(results)
```

**2. Debiasing During Fine-tuning**
```python
# Adversarial debiasing
class FairModel(nn.Module):
    def __init__(self, pretrained):
        self.encoder = pretrained
        self.classifier = TaskClassifier()
        self.adversary = BiasPredictor()
    
    def forward(self, x):
        features = self.encoder(x)
        task_output = self.classifier(features)
        # Train adversary to predict protected attribute
        # Use gradient reversal to remove bias from features
        bias_output = self.adversary(GradientReversal(features))
        return task_output, bias_output
```

**3. Balanced Fine-tuning Data**
```python
# Ensure target data represents all groups
def balanced_sample(data, protected_attr):
    groups = data.groupby(protected_attr)
    min_size = groups.size().min()
    balanced = groups.apply(lambda x: x.sample(min_size))
    return balanced
```

**4. Fairness Constraints**
```python
# Add fairness regularization
loss = task_loss + lambda * fairness_penalty

# fairness_penalty: Demographic parity, equalized odds, etc.
```

### Best Practices
1. Document source data demographics
2. Test on diverse evaluation sets
3. Monitor for disparate impact
4. Use multiple fairness metrics
5. Involve diverse stakeholders in testing

---
