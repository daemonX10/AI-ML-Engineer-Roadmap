# Autoencoders Interview Questions - General Questions

## Question 1
- [ ] Done

**How do autoencoders perform dimensionality reduction?**

**Mechanism:**
Autoencoders reduce dimensionality by forcing input through a bottleneck layer with fewer neurons than the input dimension.

**Process:**
```
Input (n dimensions) → Encoder → Bottleneck (k dimensions, k < n) → Decoder → Output (n dimensions)
```

**How It Works:**
1. Encoder compresses n-dimensional input to k-dimensional latent code
2. Decoder reconstructs n-dimensional output from k-dimensional code
3. Network learns to preserve only essential information in bottleneck
4. Latent code = reduced dimensionality representation

**Comparison with PCA:**

| Aspect | PCA | Autoencoder |
|--------|-----|-------------|
| Mapping | Linear | Nonlinear |
| Features | Orthogonal components | Learned features |
| Optimization | Eigendecomposition | Gradient descent |
| Flexibility | Limited | High |

**Key Insight:**
Linear autoencoder = PCA. Nonlinear autoencoder can capture complex manifolds that PCA cannot.

**Applications:**
- Visualization (reduce to 2D/3D)
- Feature extraction for ML models
- Data compression

---

## Question 2
- [ ] Done

**How can autoencoders be used for unsupervised learning?**

**Autoencoders are inherently unsupervised:**
- No labels required
- Learn from data structure alone
- Self-supervised objective (reconstruct input)

**Unsupervised Tasks:**

| Task | How Autoencoder Helps |
|------|----------------------|
| **Clustering** | Cluster in latent space |
| **Anomaly Detection** | High reconstruction error = anomaly |
| **Feature Learning** | Latent codes as features |
| **Density Estimation** | VAE models p(x) |
| **Data Generation** | VAE/AAE generate new samples |
| **Denoising** | Learn to remove noise |

**Process:**
1. Train autoencoder on unlabeled data
2. Use learned representations for downstream tasks
3. No manual feature engineering needed

**Example - Clustering:**
```
Data → Autoencoder → Latent codes → K-Means on latent space → Clusters
```

**Why Unsupervised:**
The reconstruction objective requires no labels - just minimize $||x - \hat{x}||^2$.

---

## Question 3
- [ ] Done

**How do recurrent autoencoders differ from feedforward autoencoders, and when might they be useful?**

**Key Differences:**

| Aspect | Feedforward AE | Recurrent AE |
|--------|---------------|--------------|
| **Input** | Fixed-size vector | Variable-length sequence |
| **Architecture** | Dense/Conv layers | LSTM/GRU layers |
| **Memory** | None | Hidden state carries history |
| **Temporal patterns** | Cannot capture | Captures sequential dependencies |

**Recurrent Autoencoder Architecture:**
```
Sequence [x₁, x₂, ..., xₜ] → LSTM Encoder → Hidden state (z) → LSTM Decoder → [x̂₁, x̂₂, ..., x̂ₜ]
```

**When to Use Recurrent AE:**
- Time series data (stock prices, sensor readings)
- Text/NLP (sentences, documents)
- Audio/speech
- Video sequences
- Any sequential data with temporal dependencies

**When Feedforward is Better:**
- Images (use CNN instead)
- Tabular data
- Fixed-length independent features

**Sequence-to-Sequence:**
- Encoder processes input sequence, final hidden = latent
- Decoder generates output sequence from latent

---

## Question 4
- [ ] Done

**What loss functions are typically used when training autoencoders?**

**Common Loss Functions:**

| Loss | Formula | Use Case |
|------|---------|----------|
| **MSE** | $\frac{1}{n}\sum(x_i - \hat{x}_i)^2$ | Continuous data, images |
| **BCE** | $-\sum[x\log\hat{x} + (1-x)\log(1-\hat{x})]$ | Binary data, images [0,1] |
| **MAE** | $\frac{1}{n}\sum|x_i - \hat{x}_i|$ | Robust to outliers |
| **Cross-Entropy** | $-\sum x \log \hat{x}$ | Categorical reconstruction |

**VAE Loss:**
$$\mathcal{L} = \mathcal{L}_{recon} + D_{KL}(q(z|x)||p(z))$$

**Specialized Losses:**

| Application | Loss |
|-------------|------|
| Images | MSE + Perceptual loss (VGG features) |
| Sparse AE | MSE + L1 sparsity penalty |
| Contractive AE | MSE + Jacobian penalty |
| Generation | MSE + Adversarial loss |

**Choosing Loss:**
- Data in [0,1]: BCE
- Continuous unbounded: MSE
- Need sharp images: Add perceptual/adversarial loss
- VAE: Add KL divergence

---

## Question 5
- [ ] Done

**How do you prevent overfitting in an autoencoder?**

**Overfitting Signs:**
- Perfect training reconstruction, poor validation
- Memorizes training data instead of learning patterns
- Learns identity function (no useful compression)

**Prevention Methods:**

| Method | How It Helps |
|--------|--------------|
| **Bottleneck constraint** | Limits capacity, forces compression |
| **Dropout** | Regularization during training |
| **Early stopping** | Stop when validation loss increases |
| **Data augmentation** | More diverse training data |
| **Weight decay (L2)** | Penalizes large weights |
| **Noise injection** | Denoising AE approach |
| **Sparsity constraint** | Limits active neurons |

**Architecture Choices:**
- Use appropriate bottleneck size (not too large)
- Avoid overcomplete without regularization
- Tie encoder/decoder weights

**Monitoring:**
```
Plot training vs validation reconstruction loss
If validation loss ↑ while training loss ↓ → overfitting
```

**Best Practice:**
Combine multiple methods: bottleneck + dropout + early stopping.

---

## Question 6
- [ ] Done

**What factors influence the capacity and size of the latent space in an autoencoder?**

**Factors:**

| Factor | Impact on Latent Size |
|--------|----------------------|
| **Data complexity** | Complex data → larger latent |
| **Intrinsic dimensionality** | Match true data dimension |
| **Task** | Generation needs larger; anomaly detection smaller |
| **Reconstruction quality** | Higher quality → larger latent |
| **Regularization** | Sparsity/VAE allows larger dimensions |
| **Dataset size** | Small data → smaller latent (prevent overfitting) |

**Trade-offs:**

| Small Latent | Large Latent |
|--------------|--------------|
| More compression | Less compression |
| May lose information | May overfit |
| Blurrier reconstruction | Sharp reconstruction |
| Captures global patterns | Captures fine details |

**Guidelines:**
- Start with latent_dim = 10-20% of input_dim
- For MNIST (784 input): 32-64 latent
- For ImageNet: 128-512 latent
- Tune based on validation reconstruction

**Determining Intrinsic Dimensionality:**
- PCA scree plot
- Reconstruction error vs latent size curve
- Cross-validation

---

## Question 7
- [ ] Done

**How do you determine the number of layers and neurons in an autoencoder?**

**General Principles:**
- Symmetric encoder-decoder (usually)
- Gradual reduction in encoder, gradual increase in decoder
- Bottleneck is the smallest layer

**Typical Architecture:**
```
Input(784) → 512 → 256 → 64(latent) → 256 → 512 → Output(784)
```

**Guidelines:**

| Aspect | Guideline |
|--------|-----------|
| **Depth** | 2-5 layers each side typically |
| **Width** | Decrease by factor of 2 each layer |
| **Bottleneck** | 10-20% of input size |
| **Symmetry** | Often mirror encoder in decoder |

**Tuning Approach:**
1. Start simple (1-2 layers)
2. Increase depth if underfitting
3. Decrease if overfitting or slow training
4. Use validation loss to compare architectures

**Hyperparameter Search:**
- Grid search over architectures
- Use validation reconstruction loss
- Consider computational budget

**Modern Practices:**
- Use BatchNorm between layers
- ReLU activation for hidden layers
- Skip connections for deep networks

---

## Question 8
- [ ] Done

**How can autoencoders be applied for feature learning?**

**Process:**
```
Raw data → Trained Autoencoder → Latent representation (features) → Downstream task
```

**How Autoencoders Learn Features:**
1. Train autoencoder on unlabeled data
2. Bottleneck forces learning of compressed representation
3. Latent code captures essential data characteristics
4. Use latent codes as features for other models

**Feature Extraction Pipeline:**
```python
# Train autoencoder
autoencoder.fit(X_unlabeled)

# Extract features
encoder = autoencoder.encoder
features = encoder.predict(X)

# Use for downstream task
classifier.fit(features, y)
```

**Why Better Than Raw Features:**
- Dimensionality reduction
- Noise removal
- Captures underlying structure
- Nonlinear feature combinations

**Applications:**
- Image features for classification
- Text embeddings for NLP
- Pretraining for supervised learning
- Transfer learning

---

## Question 9
- [ ] Done

**How are autoencoders utilized in recommendation systems?**

**Approach:**
Use autoencoder to learn user/item representations and predict missing ratings.

**Architecture:**
```
User ratings [r₁, ?, r₃, ?, r₅, ...] → Encoder → Latent → Decoder → [r̂₁, r̂₂, r̂₃, r̂₄, r̂₅, ...]
```

**Types:**

**1. User-based:**
- Input: User's rating vector (all items)
- Latent: User preference embedding
- Output: Predicted ratings for all items

**2. Item-based:**
- Input: Item's rating profile (all users)
- Latent: Item characteristic embedding
- Output: Predicted ratings from all users

**Training (Masked Loss):**
$$\mathcal{L} = \sum_{(u,i) \in \text{observed}} (r_{ui} - \hat{r}_{ui})^2$$

Only compute loss on known ratings.

**Recommendation:**
1. Encode user profile
2. Decode to get all predicted ratings
3. Recommend items with highest predicted ratings not yet seen

**Advantages:**
- Handles sparse data
- Captures nonlinear patterns
- Can incorporate side information

---

## Question 10
- [ ] Done

**In what ways can autoencoders contribute to anomaly detection?**

**Principle:**
Train autoencoder on normal data only. Anomalies have high reconstruction error.

**Algorithm:**
```
1. Train: AE on normal samples
2. Threshold: τ = mean(error) + k×std(error) on validation
3. Detect: If ||x - AE(x)||² > τ → Anomaly
```

**Methods:**

| Approach | Description |
|----------|-------------|
| **Reconstruction Error** | High error = anomaly |
| **Latent Space Distance** | Anomaly latent far from normal cluster |
| **Ensemble** | Multiple AEs, combine scores |
| **VAE ELBO** | Low ELBO = anomaly |

**Why It Works:**
- AE learns to compress and reconstruct normal patterns
- Novel/anomalous patterns weren't seen during training
- Poor reconstruction = doesn't fit learned distribution

**Applications:**
- Fraud detection
- Network intrusion
- Manufacturing defects
- Medical diagnosis
- Equipment failure prediction

**Enhancements:**
- Use variational autoencoder
- Ensemble of autoencoders
- Combine reconstruction + latent space analysis

---

## Question 11
- [ ] Done

**How can generative adversarial networks (GANs) and autoencoders be used together?**

**Hybrid Architectures:**

**1. VAE-GAN:**
```
x → Encoder → z → Decoder/Generator → x̂
                           ↓
                    Discriminator
```
- VAE reconstruction + adversarial loss
- Sharper outputs than VAE alone

**2. Adversarial Autoencoder (AAE):**
```
x → Encoder → z → Decoder → x̂
        ↓
   Discriminator(z) matches z to prior
```
- Replace KL divergence with adversarial training
- Can use any prior distribution

**3. BiGAN/ALI:**
```
Real: (x, Enc(x)) → Discriminator
Fake: (Gen(z), z) → Discriminator
```
- Jointly trains encoder and generator
- Best of both worlds

**Benefits of Combination:**
- Sharper images (GAN)
- Structured latent space (AE)
- Both encode and generate

**Loss:**
$$\mathcal{L} = \mathcal{L}_{recon} + \lambda_1 \mathcal{L}_{adv} + \lambda_2 \mathcal{L}_{feature}$$

---

## Question 12
- [ ] Done

**How do autoencoders contribute to the understanding and visualization of high-dimensional data?**

**Visualization Pipeline:**
```
High-dim data → Autoencoder → Latent (low-dim) → Plot / Analyze
```

**Methods:**

**1. Direct Visualization (if latent_dim ≤ 3):**
- 2D latent: scatter plot
- 3D latent: 3D scatter plot
- Color by class/attribute

**2. Latent + t-SNE/UMAP:**
```
Data → AE → Latent (e.g., 64-dim) → t-SNE → 2D plot
```
- AE reduces to manageable dimension
- t-SNE/UMAP visualizes latent structure

**Understanding Data:**
- Cluster structure: Similar points cluster in latent space
- Manifold shape: Latent reveals underlying geometry
- Outliers: Points far from clusters
- Transitions: Interpolate between latent points

**Interactive Exploration:**
- Modify latent dimensions, observe output changes
- Identify what each dimension encodes
- Find meaningful directions in latent space

**Applications:**
- Visualize image datasets
- Explore molecular structures
- Analyze genomic data
- Customer segmentation visualization

---

## Question 13
- [ ] Done

**Provide an example of how autoencoders could be used for genomic data compression and feature extraction.**

**Scenario:**
Gene expression data with ~20,000 genes per sample, thousands of samples.

**Architecture:**
```
Gene expression (20,000 dim) → Encoder → Latent (100 dim) → Decoder → Reconstruction
```

**Compression:**
- Input: 20,000 genes × 4 bytes = 80KB per sample
- Latent: 100 × 4 bytes = 0.4KB
- Compression ratio: 200×

**Feature Extraction:**
```python
# Train autoencoder
ae.fit(gene_expression_data)

# Extract features
latent_features = ae.encoder.predict(gene_expression_data)

# Use for downstream tasks
# - Disease classification
# - Patient clustering
# - Drug response prediction
```

**Specialized Architectures:**
- **scVI:** VAE for single-cell RNA-seq with count distribution modeling
- **DCA:** Handles zero-inflation in scRNA-seq

**Applications:**
- Cell type identification
- Disease subtyping
- Biomarker discovery
- Drug target identification

**Benefits:**
- Captures nonlinear relationships between genes
- Handles noise and dropout in sequencing
- Enables visualization of high-dimensional data
