# GANs Interview Questions - General Questions

---

## Question 1: How do GANs handle the generation of new, unseen data?
- [ ] Done

### Definition
GANs generate new data by learning the underlying probability distribution of training data. The Generator maps random noise vectors from a simple prior distribution to the complex data distribution, creating novel samples that resemble training data but are not copies.

### Generation Process

| Step | Action |
|------|--------|
| 1 | Sample random noise z from prior (e.g., Gaussian) |
| 2 | Pass z through trained Generator network |
| 3 | Generator transforms z to data space |
| 4 | Output is new, unseen sample |

### Why Generated Data is Novel

| Aspect | Explanation |
|--------|-------------|
| **Continuous Latent Space** | Infinite points between training examples |
| **Interpolation** | Can generate samples between known examples |
| **Distribution Learning** | Learns patterns, not memorization |
| **Stochastic Sampling** | Each z gives different output |

### Mathematical Perspective
Generator learns mapping: $G: \mathcal{Z} \rightarrow \mathcal{X}$

- $\mathcal{Z}$: Low-dimensional latent space (e.g., 100D Gaussian)
- $\mathcal{X}$: High-dimensional data space (e.g., images)
- Ideally: $p_G(x) \approx p_{data}(x)$

### Practical Example
```python
# Generate new faces
z = torch.randn(batch_size, latent_dim)  # Random noise
new_faces = generator(z)  # Novel, unseen faces
```

---

## Question 2: What loss functions are commonly used in GANs and why?
- [ ] Done

### Definition
GAN loss functions measure how well the Generator fools the Discriminator and how well the Discriminator distinguishes real from fake. Different losses address training stability and convergence issues.

### Common Loss Functions

| Loss | Formula (D) | Formula (G) | Characteristics |
|------|-------------|-------------|-----------------|
| **Original GAN** | $-\log D(x) - \log(1-D(G(z)))$ | $\log(1-D(G(z)))$ | Can suffer vanishing gradients |
| **Non-saturating** | Same | $-\log D(G(z))$ | Better gradients early |
| **WGAN** | $D(x) - D(G(z))$ | $-D(G(z))$ | More stable, requires weight clipping |
| **WGAN-GP** | WGAN + gradient penalty | $-D(G(z))$ | Stable, no clipping needed |
| **Hinge Loss** | $\min(0, -1+D(x)) + \min(0, -1-D(G(z)))$ | $-D(G(z))$ | Used in BigGAN, spectral norm |
| **Least Squares** | $(D(x)-1)^2 + D(G(z))^2$ | $(D(G(z))-1)^2$ | Smoother gradients |

### Why Different Losses Exist

**Original GAN Problem:**
- When D is strong, $D(G(z)) \approx 0$
- Gradient $\nabla_G \log(1 - D(G(z))) \approx 0$
- Generator cannot learn

**Non-saturating Fix:**
- Use $-\log D(G(z))$ instead
- Stronger gradient when D is confident

**Wasserstein Loss Benefits:**
- Provides meaningful distance metric
- Gradients don't vanish
- Correlates with sample quality

### Python Example
```python
# Original GAN Loss
d_loss_real = F.binary_cross_entropy(D(real), ones)
d_loss_fake = F.binary_cross_entropy(D(G(z)), zeros)
g_loss = F.binary_cross_entropy(D(G(z)), ones)  # Non-saturating

# WGAN Loss
d_loss = -torch.mean(D(real)) + torch.mean(D(G(z)))
g_loss = -torch.mean(D(G(z)))
```

---

## Question 3: How is the training process different for the Generator and Discriminator?
- [ ] Done

### Definition
Generator and Discriminator have opposing objectives and are trained in alternating steps. Discriminator is trained to maximize classification accuracy while Generator is trained to minimize it.

### Training Differences

| Aspect | Discriminator | Generator |
|--------|---------------|-----------|
| **Objective** | Maximize $D(real)$, minimize $D(fake)$ | Maximize $D(fake)$ |
| **Input** | Real and fake samples | Only noise z |
| **Gradients From** | Classification loss | D's output on fake |
| **Training Steps** | Often multiple per G step | One per D step(s) |

### Alternating Training Process
```
For each iteration:
    1. Train Discriminator (k steps):
       - Sample real batch from data
       - Generate fake batch: fake = G(z)
       - Compute D loss on real (should output 1)
       - Compute D loss on fake (should output 0)
       - Update D weights
    
    2. Train Generator (1 step):
       - Generate fake batch: fake = G(z)
       - Compute G loss: D(fake) should be high
       - Backprop through D (frozen) to G
       - Update G weights
```

### Key Differences in Practice

| Practice | D Training | G Training |
|----------|------------|------------|
| **Batch Composition** | Half real, half fake | Only fake |
| **Label** | Real=1, Fake=0 | Fake should be classified as 1 |
| **D Weights** | Updated | Frozen (only provides gradients) |

### Python Code
```python
# Train Discriminator
optimizer_D.zero_grad()
real_loss = criterion(D(real_images), real_labels)
fake_images = G(z).detach()  # Detach to not update G
fake_loss = criterion(D(fake_images), fake_labels)
d_loss = real_loss + fake_loss
d_loss.backward()
optimizer_D.step()

# Train Generator
optimizer_G.zero_grad()
fake_images = G(z)  # No detach - need gradients
g_loss = criterion(D(fake_images), real_labels)  # Fool D
g_loss.backward()
optimizer_G.step()
```

---

## Question 4: How can we evaluate the performance and quality of GANs?
- [ ] Done

### Definition
GAN evaluation is challenging because there's no single metric that captures all aspects of generation quality. Multiple metrics assess different properties like realism, diversity, and distribution matching.

### Common Evaluation Metrics

| Metric | What It Measures | Calculation |
|--------|------------------|-------------|
| **Inception Score (IS)** | Quality + diversity | $\exp(\mathbb{E}[KL(p(y|x) || p(y))])$ |
| **FID** | Distribution similarity | Distance between real and fake feature distributions |
| **Precision** | Quality (realism) | Fraction of fakes that look real |
| **Recall** | Diversity (coverage) | Fraction of real modes covered |
| **LPIPS** | Perceptual diversity | Learned perceptual distance |

### Frechet Inception Distance (FID)
Most widely used metric:
$$FID = ||\mu_r - \mu_g||^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

Where $(\mu_r, \Sigma_r)$ and $(\mu_g, \Sigma_g)$ are mean and covariance of real and generated features from Inception network.

**Lower FID = Better quality**

### Qualitative Evaluation

| Method | Purpose |
|--------|---------|
| **Visual Inspection** | Human judgment of realism |
| **Interpolation** | Check smooth transitions |
| **Nearest Neighbor** | Ensure not memorizing |
| **User Studies** | Real vs fake classification |

### Python Code
```python
from pytorch_fid import fid_score

# Calculate FID
fid = fid_score.calculate_fid_given_paths(
    [real_images_path, generated_images_path],
    batch_size=50,
    device='cuda',
    dims=2048
)
print(f"FID: {fid}")
```

### Evaluation Best Practices
- Use multiple metrics (FID + Precision + Recall)
- Report with sufficient samples (50k+)
- Compare on same dataset/resolution
- Include qualitative examples

---

## Question 5: In what ways do GANs contribute to semi-supervised learning?
- [ ] Done

### Definition
GANs enhance semi-supervised learning by using the Discriminator not just for real/fake classification but also for actual class prediction. The GAN framework provides additional training signal from unlabeled data.

### How It Works

**Standard Discriminator**: 2 outputs (real/fake)
**Semi-supervised Discriminator**: K+1 outputs (K classes + fake)

### Architecture Modification

| Component | Change |
|-----------|--------|
| **Discriminator Output** | K+1 classes instead of 1 |
| **Real Data** | Predict actual class (if labeled) |
| **Fake Data** | Predict "fake" class |
| **Unlabeled Data** | Predict any of K real classes |

### Training Process
```
For labeled data:
    - D predicts correct class (1 to K)
    
For unlabeled data:
    - D predicts any real class (1 to K)
    - Forces D to learn meaningful features
    
For generated data:
    - D predicts fake class (K+1)
```

### Benefits

| Benefit | Explanation |
|---------|-------------|
| **Better Features** | GAN training improves D's representations |
| **Data Augmentation** | Generated samples add training variety |
| **Regularization** | Adversarial training prevents overfitting |
| **Label Efficiency** | Good accuracy with few labels |

### Loss Function
$$\mathcal{L} = \mathcal{L}_{supervised} + \mathcal{L}_{unsupervised}$$

Where:
- $\mathcal{L}_{supervised}$: Cross-entropy on labeled data
- $\mathcal{L}_{unsupervised}$: GAN loss (real vs fake)

### Practical Results
- Achieves near-supervised performance with 100-1000 labels
- Works especially well on image classification

---

## Question 6: How do generative models like GANs handle feature matching?
- [ ] Done

### Definition
Feature matching is a technique where the Generator is trained to match the statistics of intermediate Discriminator features rather than just fooling the final output. This provides more stable gradients and reduces mode collapse.

### Core Concept

**Standard GAN**: Generator maximizes $D(G(z))$
**Feature Matching**: Generator minimizes $||f(x_{real}) - f(G(z))||^2$

Where $f(x)$ is activation at intermediate layer of D.

### Why It Helps

| Problem | How Feature Matching Helps |
|---------|---------------------------|
| **Mode Collapse** | Matching statistics encourages diversity |
| **Unstable Gradients** | Feature distance provides stable signal |
| **Training Oscillation** | Smoother optimization landscape |

### Mathematical Formulation
$$\mathcal{L}_{FM} = ||\mathbb{E}_{x \sim p_{data}}[f(x)] - \mathbb{E}_{z \sim p_z}[f(G(z))]||_2^2$$

Match mean activations of real and fake distributions.

### Implementation
```python
# Feature matching loss
def feature_matching_loss(real_images, fake_images, discriminator):
    # Get intermediate features
    real_features = discriminator.get_features(real_images)
    fake_features = discriminator.get_features(fake_images)
    
    # Match mean statistics
    loss = torch.mean((real_features.mean(0) - fake_features.mean(0))**2)
    return loss
```

### Variations
- **Multi-scale Feature Matching**: Match at multiple layers
- **Perceptual Loss**: Use pre-trained VGG features
- **Moment Matching**: Match higher-order statistics

---

## Question 7: What techniques can be applied to stabilize the training of GANs?
- [ ] Done

### Definition
GAN training stabilization techniques address common issues like mode collapse, vanishing gradients, and oscillation through architectural choices, regularization, and training procedures.

### Key Stabilization Techniques

| Category | Technique | How It Helps |
|----------|-----------|--------------|
| **Architecture** | Spectral Normalization | Controls Lipschitz constant |
| **Architecture** | Progressive Growing | Gradual resolution increase |
| **Loss** | WGAN-GP | Stable gradients, no mode collapse |
| **Loss** | Hinge Loss | Bounded gradients |
| **Training** | Two-timescale Updates | Balance D and G learning |
| **Training** | Label Smoothing | Prevent overconfident D |
| **Regularization** | Gradient Penalty | Enforce Lipschitz constraint |
| **Regularization** | R1 Regularization | Penalize gradient on real data |

### Architectural Guidelines (DCGAN)
- Use strided convolutions instead of pooling
- Batch normalization in both networks
- ReLU in Generator, LeakyReLU in Discriminator
- Avoid fully connected layers

### Training Tips

| Tip | Implementation |
|-----|----------------|
| **Balance D/G** | Train D more steps if too weak |
| **Learning Rates** | Often D: 0.0004, G: 0.0001 |
| **Adam β₁** | Use 0.0 or 0.5 instead of 0.9 |
| **Batch Size** | Larger batches more stable |
| **Input Noise** | Add noise to D inputs early on |

### Python Example
```python
# Spectral Normalization
from torch.nn.utils import spectral_norm
conv = spectral_norm(nn.Conv2d(in_ch, out_ch, 3))

# Gradient Penalty (WGAN-GP)
def gradient_penalty(D, real, fake):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)
    d_out = D(interpolated)
    gradients = torch.autograd.grad(d_out, interpolated, 
                                     grad_outputs=torch.ones_like(d_out),
                                     create_graph=True)[0]
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp
```

---

## Question 8: How are GANs used for data augmentation?
- [ ] Done

### Definition
GANs generate synthetic training samples to augment limited datasets, increasing effective training set size and diversity, which improves model generalization and performance.

### Data Augmentation Process

```
1. Train GAN on original dataset
2. Generate synthetic samples
3. Combine real + synthetic for training downstream model
4. Train classifier/model on augmented dataset
```

### Benefits

| Benefit | Explanation |
|---------|-------------|
| **More Data** | Increase training set size |
| **Novel Variations** | Generate unseen combinations |
| **Class Balancing** | Generate more minority class samples |
| **Domain-Specific** | Realistic for specific domains |

### Comparison to Traditional Augmentation

| Traditional | GAN-based |
|-------------|-----------|
| Rotations, flips, crops | Novel samples |
| Simple transformations | Semantic variations |
| Limited diversity | High diversity |
| Fast | Requires GAN training |

### Applications

| Domain | Use Case |
|--------|----------|
| **Medical Imaging** | Generate rare disease images |
| **Fraud Detection** | Augment rare fraud cases |
| **Autonomous Driving** | Generate rare scenarios |
| **Defect Detection** | Create defect samples |

### Python Example
```python
# Generate augmentation samples
def augment_dataset(generator, num_samples, latent_dim):
    augmented = []
    for _ in range(num_samples):
        z = torch.randn(1, latent_dim)
        fake_sample = generator(z)
        augmented.append(fake_sample)
    return torch.cat(augmented)

# Conditional GAN for class-balanced augmentation
def balance_classes(cgan, minority_class, num_needed):
    z = torch.randn(num_needed, latent_dim)
    labels = torch.full((num_needed,), minority_class)
    synthetic_samples = cgan(z, labels)
    return synthetic_samples
```

### Best Practices
- Validate synthetic quality before use
- Don't over-augment (keep real:synthetic ratio reasonable)
- Use conditional GANs for class-specific generation
- Monitor downstream model performance

---

## Question 9: How can GANs be used for unsupervised representation learning?
- [ ] Done

### Definition
GANs learn meaningful representations without labels by training the Discriminator to distinguish real from fake. The learned features capture semantic information useful for downstream tasks.

### How Representations Are Learned

| Source | What It Learns |
|--------|----------------|
| **Generator** | Latent space encodes data factors |
| **Discriminator** | Features distinguish real patterns |
| **BiGAN/ALI** | Encoder maps data to latent space |

### BiGAN (Bidirectional GAN)
Adds an Encoder that maps data to latent space:

```
Standard GAN:
    z → G → x_fake

BiGAN:
    z → G → x_fake
    x_real → E → z_inferred
    
Discriminator: Distinguishes (x_real, E(x_real)) from (G(z), z)
```

### Feature Extraction Approaches

| Approach | Method |
|----------|--------|
| **D Features** | Use intermediate D layers as features |
| **Encoder** | Train encoder to invert generator |
| **Latent Space** | Use z as representation |

### Evaluation
- Train linear classifier on frozen features
- Compare to supervised pre-training
- Transfer to downstream tasks

### Python Example
```python
# Extract features from discriminator
class FeatureExtractor:
    def __init__(self, discriminator):
        self.D = discriminator
        
    def extract(self, x):
        # Get intermediate layer activations
        features = []
        for layer in self.D.layers[:-1]:  # Exclude final layer
            x = layer(x)
            features.append(x)
        return torch.cat([f.flatten() for f in features])

# Use for downstream task
features = extractor.extract(images)
classifier = LinearSVM()
classifier.fit(features, labels)
```

### Practical Relevance
- Pre-training when labels are expensive
- Feature learning for clustering
- Initialization for supervised fine-tuning

---

## Question 10: What metrics are suitable for assessing the diversity of generated samples in GANs?
- [ ] Done

### Definition
Diversity metrics measure whether GANs generate varied samples covering different modes of the data distribution, detecting mode collapse where generators produce limited variations.

### Key Diversity Metrics

| Metric | What It Measures | Formula/Method |
|--------|------------------|----------------|
| **Recall** | Mode coverage | Fraction of real data modes covered |
| **MS-SSIM** | Pairwise similarity | Mean structural similarity between generated pairs |
| **LPIPS** | Perceptual diversity | Learned perceptual distance |
| **NDB** | Mode coverage | Number of statistically different bins |
| **Coverage** | Manifold coverage | Fraction of real samples with nearby fakes |

### Precision vs Recall

| Metric | Measures | Detects |
|--------|----------|---------|
| **Precision** | Quality/Fidelity | Are fakes realistic? |
| **Recall** | Diversity/Coverage | Are all modes covered? |

Mode collapse: High precision, low recall

### MS-SSIM Diversity
$$Diversity = 1 - \frac{1}{N^2}\sum_{i,j} SSIM(G(z_i), G(z_j))$$

Lower average similarity = higher diversity.

### LPIPS Diversity
Use pre-trained network to measure perceptual distance:
$$Diversity = \mathbb{E}_{z_1, z_2}[LPIPS(G(z_1), G(z_2))]$$

### Python Example
```python
import lpips

# LPIPS diversity
loss_fn = lpips.LPIPS(net='alex')

def compute_diversity(generator, num_pairs=1000, latent_dim=100):
    distances = []
    for _ in range(num_pairs):
        z1, z2 = torch.randn(2, latent_dim)
        img1, img2 = generator(z1), generator(z2)
        dist = loss_fn(img1, img2)
        distances.append(dist.item())
    return np.mean(distances)

# Higher = more diverse
print(f"Average LPIPS diversity: {compute_diversity(generator)}")
```

### Best Practices
- Use multiple diversity metrics
- Compare to real data diversity baseline
- Monitor during training for mode collapse detection
- Report both quality (precision/FID) and diversity (recall)

---

## Question 11: Present a use case for GANs in financial modeling for generating synthetic time-series data
- [ ] Done

### Scenario
A financial institution needs realistic synthetic stock price data to test trading algorithms, perform stress testing, and train ML models without using sensitive real market data.

### Requirements

| Requirement | Why Important |
|-------------|---------------|
| **Realistic Patterns** | Must exhibit real market properties |
| **Statistical Properties** | Preserve volatility, correlations |
| **Privacy** | Cannot leak real trading patterns |
| **Scenario Generation** | Generate extreme/rare events |

### GAN Architecture for Time Series

**TimeGAN Architecture:**
```
Real sequence → Embedding Network → Latent space
Random noise → Generator → Synthetic latent → Recovery Network → Synthetic sequence

Discriminator: Operates in both latent and feature space
```

### Key Properties to Capture

| Property | Description |
|----------|-------------|
| **Volatility Clustering** | High volatility periods cluster |
| **Fat Tails** | Extreme events more common than Gaussian |
| **Autocorrelation** | Returns have temporal structure |
| **Cross-correlations** | Asset relationships preserved |

### Implementation Approach
```python
# TimeGAN-style training
class TimeSeriesGAN:
    def __init__(self, seq_len, n_features):
        self.embedder = EmbeddingNetwork(n_features, hidden_dim)
        self.recovery = RecoveryNetwork(hidden_dim, n_features)
        self.generator = SequenceGenerator(noise_dim, hidden_dim)
        self.discriminator = SequenceDiscriminator(hidden_dim)
        self.supervisor = SupervisorNetwork(hidden_dim)
    
    def generate(self, num_sequences):
        z = torch.randn(num_sequences, seq_len, noise_dim)
        synthetic_latent = self.generator(z)
        synthetic_data = self.recovery(synthetic_latent)
        return synthetic_data
```

### Validation Steps
1. Compare statistical moments (mean, variance, skewness)
2. Check autocorrelation structure
3. Validate extreme event distribution
4. Test discriminative score (can classifier distinguish?)
5. Predictive score (models trained on synthetic perform on real)

### Applications
- Algorithmic trading backtesting
- Risk model stress testing
- Training fraud detection models
- Regulatory compliance testing

---

## Question 12: How can GANs be defended against adversarial attacks, or used for adversarial training?
- [ ] Done

### Definition
GANs relate to adversarial robustness in two ways: (1) GANs themselves can be attacked, and (2) GANs can generate adversarial examples to train more robust models.

### Defending GANs Against Attacks

| Attack Type | Description | Defense |
|-------------|-------------|---------|
| **Input Perturbation** | Perturb input to fool D | Input preprocessing |
| **Membership Inference** | Detect if sample was in training | Differential privacy |
| **Model Extraction** | Steal generator | Rate limiting, watermarking |

### Using GANs for Adversarial Training

**Goal**: Train classifier robust to adversarial perturbations

**Approach**: Generate adversarial examples with GAN, include in training

```
Training Loop:
1. Train generator to create adversarial perturbations
2. Add perturbations to clean images
3. Train classifier on clean + adversarial
4. Classifier becomes robust to attacks
```

### Adversarial Example Generation with GANs

| Method | How It Works |
|--------|--------------|
| **AdvGAN** | Generator creates perturbations that fool classifier |
| **Natural GAN** | Generate realistic adversarial examples |
| **Defense-GAN** | Project inputs onto GAN manifold before classification |

### Defense-GAN Approach
```python
def defense_gan(classifier, generator, x_adv, iterations=1000):
    """Project adversarial input onto GAN manifold"""
    z = torch.randn(latent_dim, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=0.01)
    
    for _ in range(iterations):
        optimizer.zero_grad()
        x_reconstructed = generator(z)
        loss = F.mse_loss(x_reconstructed, x_adv)
        loss.backward()
        optimizer.step()
    
    # Classify the cleaned input
    x_clean = generator(z)
    return classifier(x_clean)
```

### Adversarial Training with GAN
```python
# Train robust classifier
def adversarial_training_step(classifier, adv_generator, x, y):
    # Clean loss
    clean_pred = classifier(x)
    clean_loss = criterion(clean_pred, y)
    
    # Adversarial loss
    perturbation = adv_generator(x)
    x_adv = x + perturbation
    adv_pred = classifier(x_adv)
    adv_loss = criterion(adv_pred, y)
    
    # Combined training
    total_loss = clean_loss + lambda_adv * adv_loss
    return total_loss
```

### Practical Considerations
- Trade-off between clean accuracy and robustness
- Adversarial training is computationally expensive
- Multiple attack types require diverse adversarial examples

---

## Question 13: What role do GANs play in the field of reinforcement learning?
- [ ] Done

### Definition
GANs enhance reinforcement learning by generating synthetic environments, augmenting experience data, learning reward functions from demonstrations, and improving sample efficiency.

### Key Applications

| Application | How GAN Helps |
|-------------|---------------|
| **World Models** | Generate predicted future states |
| **Data Augmentation** | Create additional training experiences |
| **Inverse RL** | Learn reward function from expert demos |
| **Sim-to-Real** | Adapt simulated images to real domain |
| **Goal Generation** | Create diverse training goals |

### World Models with GANs
```
Current State + Action → GAN → Predicted Next State
```
- Train policy in imagined trajectories
- Reduces need for real environment interaction
- Enables planning without simulation

### Generative Adversarial Imitation Learning (GAIL)
Learn policy by imitating expert without knowing reward:

```
Expert Demonstrations → Discriminator distinguishes expert vs policy
                              ↓
                    Policy improves to fool Discriminator
```

**GAIL Loss:**
- D: Classify expert trajectories as real, policy trajectories as fake
- Policy: Maximize D score (appear like expert)

### Sim-to-Real Transfer
```
Simulated Images → CycleGAN → Realistic Images
                              ↓
                    Policy trained on realistic sim
                              ↓
                    Better real-world transfer
```

### Python Example (Simplified GAIL)
```python
class GAIL:
    def __init__(self, policy, discriminator):
        self.policy = policy
        self.D = discriminator
    
    def train_step(self, expert_states, expert_actions):
        # Get policy trajectories
        policy_states, policy_actions = self.policy.rollout()
        
        # Train discriminator
        expert_pairs = torch.cat([expert_states, expert_actions], dim=1)
        policy_pairs = torch.cat([policy_states, policy_actions], dim=1)
        
        d_loss = -torch.mean(torch.log(self.D(expert_pairs))) \
                 - torch.mean(torch.log(1 - self.D(policy_pairs)))
        
        # Train policy with discriminator as reward
        rewards = -torch.log(1 - self.D(policy_pairs))
        policy_loss = self.policy.update(rewards)
```

### Benefits
- Learn complex behaviors from demonstrations
- Reduce sample complexity
- Enable training in rich, generated environments

---
