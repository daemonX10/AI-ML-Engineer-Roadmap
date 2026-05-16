# Autoencoders Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What is an autoencoder?**

**Definition:**
An autoencoder is an unsupervised neural network that learns to compress input data into a lower-dimensional latent representation (encoding) and then reconstruct the original input from this representation (decoding). The goal is to minimize reconstruction error.

**Core Concepts:**
- Consists of two parts: Encoder (compresses) and Decoder (reconstructs)
- Learns identity function under constraints (bottleneck)
- Trained to minimize difference between input and output
- Captures important features in latent space

**Mathematical Formulation:**
$$\text{Encoder: } z = f_\theta(x)$$
$$\text{Decoder: } \hat{x} = g_\phi(z)$$
$$\text{Loss: } \mathcal{L} = ||x - \hat{x}||^2$$

Where $x$ is input, $z$ is latent code, $\hat{x}$ is reconstruction.

**Intuition:**
Think of it as forcing data through a narrow pipe (bottleneck). Only essential information passes through. Like summarizing a book into key points, then recreating the book from those points.

**Practical Relevance:**
- Dimensionality reduction
- Feature learning
- Denoising data
- Anomaly detection
- Data compression

---

## Question 2
- [ ] Done

**Explain the architecture of a basic autoencoder.**

**Definition:**
A basic autoencoder has a symmetric architecture with an encoder network, a bottleneck (latent layer), and a decoder network. Input flows through encoder → bottleneck → decoder to produce reconstruction.

**Architecture Components:**

| Component | Function | Structure |
|-----------|----------|-----------|
| Input Layer | Receives raw data | n neurons (input dimension) |
| Encoder | Compresses data | Decreasing layer sizes |
| Bottleneck | Latent representation | m neurons (m << n) |
| Decoder | Reconstructs data | Increasing layer sizes |
| Output Layer | Reconstructed data | n neurons (same as input) |

**Mathematical Formulation:**
```
Input (x) → [Encoder layers] → Latent (z) → [Decoder layers] → Output (x̂)

Encoder: h₁ = σ(W₁x + b₁), z = σ(W₂h₁ + b₂)
Decoder: h₂ = σ(W₃z + b₃), x̂ = σ(W₄h₂ + b₄)
```

**Intuition:**
Like an hourglass - wide at input, narrow in middle (bottleneck), wide again at output. The narrow middle forces the network to learn only the most important features.

**Key Design Choices:**
- Number of hidden layers
- Size of bottleneck (latent dimension)
- Activation functions (ReLU for hidden, sigmoid/linear for output)
- Symmetric vs asymmetric architecture

---

## Question 3
- [ ] Done

**What is the difference between an encoder and a decoder?**

**Definition:**

| Aspect | Encoder | Decoder |
|--------|---------|---------|
| Purpose | Compress input to latent space | Reconstruct input from latent space |
| Direction | High-dim → Low-dim | Low-dim → High-dim |
| Function | $z = f(x)$ | $\hat{x} = g(z)$ |
| Layer sizes | Decreasing | Increasing |
| Output | Latent representation | Reconstructed data |

**Core Concepts:**
- **Encoder:** Maps input $x \in \mathbb{R}^n$ to latent code $z \in \mathbb{R}^m$ where $m < n$
- **Decoder:** Maps latent code $z$ back to reconstructed input $\hat{x} \in \mathbb{R}^n$

**Intuition:**
- Encoder = Compression (like zipping a file)
- Decoder = Decompression (like unzipping)

**Practical Usage:**
- Encoder alone: Used for feature extraction, dimensionality reduction
- Decoder alone: Used for generation (given a latent code, generate data)
- Together: Used for reconstruction, denoising, anomaly detection

---

## Question 4
- [ ] Done

**What are some key applications of autoencoders?**

**Applications:**

| Application | How Autoencoder Helps |
|-------------|----------------------|
| **Dimensionality Reduction** | Learns nonlinear compression (better than PCA for complex data) |
| **Anomaly Detection** | High reconstruction error indicates anomaly |
| **Denoising** | Learn to reconstruct clean data from noisy input |
| **Image Compression** | Compress images to smaller latent codes |
| **Feature Learning** | Latent space serves as learned features |
| **Data Generation** | VAEs generate new samples from latent space |
| **Recommendation Systems** | Learn user/item embeddings |
| **Image Inpainting** | Fill missing parts of images |
| **Drug Discovery** | Generate molecular structures |
| **Pretraining** | Initialize deep networks with useful weights |

**Real-World Examples:**
- **Fraud Detection:** Normal transactions reconstruct well; fraudulent ones have high error
- **Medical Imaging:** Compress/denoise X-rays, MRIs
- **NLP:** Learn sentence embeddings, text compression
- **Manufacturing:** Detect defective products via reconstruction error

---

## Question 5
- [ ] Done

**Describe the difference between a traditional autoencoder and a variational autoencoder (VAE).**

**Definition:**

| Aspect | Traditional Autoencoder | Variational Autoencoder (VAE) |
|--------|------------------------|-------------------------------|
| Latent Space | Deterministic point | Probabilistic distribution |
| Encoder Output | Single vector $z$ | Mean $\mu$ and variance $\sigma^2$ |
| Sampling | No sampling | Sample $z$ from $\mathcal{N}(\mu, \sigma^2)$ |
| Loss Function | Reconstruction loss only | Reconstruction + KL divergence |
| Generation | Poor (no structured latent space) | Good (smooth, continuous latent space) |
| Regularization | None inherent | KL divergence regularizes latent space |

**Mathematical Formulation:**

**Traditional AE:**
$$\mathcal{L} = ||x - \hat{x}||^2$$

**VAE:**
$$\mathcal{L} = \underbrace{||x - \hat{x}||^2}_{\text{Reconstruction}} + \underbrace{D_{KL}(q(z|x) || p(z))}_{\text{KL Divergence}}$$

**Reparameterization Trick (VAE):**
$$z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

**Intuition:**
- Traditional AE: Maps each input to a fixed point in latent space
- VAE: Maps each input to a "cloud" (distribution) in latent space, enabling smooth interpolation and generation

**When to Use:**
- Traditional AE: Feature extraction, compression, denoising
- VAE: Data generation, smooth latent space interpolation

---

## Question 6
- [ ] Done

**What is meant by the latent space in the context of autoencoders?**

**Definition:**
The latent space is the compressed, lower-dimensional representation space where the encoder maps input data. It captures the essential features/patterns of the data in fewer dimensions.

**Core Concepts:**
- **Bottleneck layer** output = latent representation
- Dimension of latent space << dimension of input
- Each point in latent space represents a compressed version of input
- Similar inputs should map to nearby points in latent space

**Mathematical Representation:**
$$z = f_{encoder}(x) \in \mathbb{R}^d$$

Where $d$ is latent dimension, typically $d << n$ (input dimension).

**Properties of Good Latent Space:**
- **Compact:** Captures essential information in few dimensions
- **Smooth:** Similar inputs map to nearby points
- **Disentangled:** Different latent dimensions capture different features
- **Continuous:** Small changes in latent space = small changes in output (especially VAE)

**Intuition:**
Think of latent space as a "summary room" where all the key information about an image/data point is stored. Instead of storing the full 1000-pixel image, you store just 10 numbers that capture its essence.

**Practical Uses of Latent Space:**
- Interpolation between data points
- Clustering similar data
- Visualization (using t-SNE on latent codes)
- Generation (sample from latent space, decode)

---

## Question 7
- [ ] Done

**Explain the concept of a sparse autoencoder.**

**Definition:**
A sparse autoencoder is an autoencoder with a sparsity constraint on the latent layer, forcing most neurons to be inactive (close to zero) for any given input. This encourages the network to learn more meaningful, distributed features.

**Core Concepts:**
- Adds sparsity penalty to loss function
- Only a few neurons activate per input
- Prevents trivial identity mapping
- Can have overcomplete latent layer (latent dim > input dim)

**Mathematical Formulation:**
$$\mathcal{L} = ||x - \hat{x}||^2 + \lambda \cdot \Omega_{sparsity}(h)$$

**Sparsity Penalty (KL Divergence approach):**
$$\Omega = \sum_j KL(\rho || \hat{\rho}_j) = \sum_j \left[ \rho \log\frac{\rho}{\hat{\rho}_j} + (1-\rho)\log\frac{1-\rho}{1-\hat{\rho}_j} \right]$$

Where:
- $\rho$ = target sparsity (e.g., 0.05)
- $\hat{\rho}_j$ = average activation of neuron $j$ across all inputs

**Alternative: L1 Penalty:**
$$\Omega = ||h||_1 = \sum_j |h_j|$$

**Intuition:**
Like forcing a team where only 2-3 players can be active at a time. Each player must specialize in specific scenarios, leading to diverse, specialized skills.

**Benefits:**
- Learns more interpretable features
- Prevents overfitting even with large latent space
- Each neuron learns unique, specialized feature

---

## Question 8
- [ ] Done

**What is a denoising autoencoder and how does it work?**

**Definition:**
A denoising autoencoder (DAE) is trained to reconstruct the original clean input from a corrupted/noisy version. It learns robust features that capture the underlying data structure rather than superficial patterns.

**How It Works:**
1. Take clean input $x$
2. Corrupt it: $\tilde{x} = x + \text{noise}$
3. Feed corrupted input to autoencoder
4. Train to reconstruct original clean $x$ (not $\tilde{x}$)

**Mathematical Formulation:**
$$\tilde{x} = \text{corrupt}(x)$$
$$\mathcal{L} = ||x - g(f(\tilde{x}))||^2$$

**Types of Corruption:**
- **Gaussian noise:** $\tilde{x} = x + \mathcal{N}(0, \sigma^2)$
- **Masking noise:** Randomly set some inputs to 0
- **Salt-and-pepper:** Random pixels set to 0 or 1
- **Dropout:** Randomly drop input features

**Intuition:**
Like learning to read handwriting by practicing with smudged text. You learn the essential patterns, not the smudges. The network must understand the "true" structure to remove noise.

**Benefits:**
- Learns more robust features
- Better generalization
- Implicit regularization
- More meaningful latent representations

**Applications:**
- Image denoising
- Audio noise removal
- Data preprocessing
- Robust feature learning

---

## Question 9
- [ ] Done

**Describe how a contractive autoencoder operates and its benefits.**

**Definition:**
A contractive autoencoder (CAE) adds a penalty term that penalizes the sensitivity of the encoder's output to small changes in input. This encourages the model to learn representations that are robust to small input variations.

**Core Concept:**
- Penalizes the Frobenius norm of the Jacobian matrix of encoder
- Jacobian measures how much latent code changes when input changes
- Small Jacobian = latent representation is "contracted" (insensitive to input perturbations)

**Mathematical Formulation:**
$$\mathcal{L} = ||x - \hat{x}||^2 + \lambda ||\frac{\partial h}{\partial x}||_F^2$$

**Jacobian Matrix:**
$$J = \frac{\partial h}{\partial x} \in \mathbb{R}^{d_{latent} \times d_{input}}$$

**Frobenius Norm:**
$$||J||_F^2 = \sum_{i,j} J_{ij}^2 = \sum_j \sum_i \left(\frac{\partial h_j}{\partial x_i}\right)^2$$

**Intuition:**
Imagine the encoder as a mapping on a rubber sheet. Contractive penalty makes the rubber sheet "stiff" - small movements on input side don't cause large movements in latent space. Points near each other stay near.

**Benefits:**
- Learns locally invariant features
- More robust to noise and perturbations
- Better generalization
- Captures essential data manifold structure

**Comparison with Denoising AE:**
- Denoising: Robust to explicit noise types
- Contractive: Robust to any small perturbation (more general)

---

## Question 10
- [ ] Done

**What are convolutional autoencoders and in what cases are they preferred?**

**Definition:**
Convolutional autoencoders (CAE) use convolutional layers in the encoder and transposed convolution (deconvolution) layers in the decoder. They exploit spatial structure in data like images.

**Architecture:**
```
Encoder: Conv → Pool → Conv → Pool → Flatten → Latent
Decoder: Dense → Reshape → ConvTranspose → Upsample → ConvTranspose → Output
```

**Key Differences from Feedforward AE:**

| Aspect | Feedforward AE | Convolutional AE |
|--------|---------------|------------------|
| Input handling | Flattened vector | Preserves spatial structure |
| Parameters | Many (dense connections) | Few (shared filters) |
| Features | Global patterns | Local + hierarchical patterns |
| Best for | Tabular, 1D data | Images, video, spatial data |

**Operations:**
- **Encoder:** Convolutions + pooling (downsampling)
- **Decoder:** Transposed convolutions + upsampling

**When to Use Convolutional AE:**
- Image data (compression, denoising, super-resolution)
- Video processing (frame prediction)
- Any data with spatial/local correlations
- When translation invariance is important

**Benefits:**
- Parameter efficient (weight sharing)
- Captures spatial hierarchies
- Translation invariant features
- Better reconstruction for images

---

## Question 11
- [ ] Done

**Explain the idea behind stacked autoencoders.**

**Definition:**
A stacked autoencoder is a deep autoencoder built by stacking multiple autoencoder layers. Each layer is typically pretrained individually, then fine-tuned together. This enables learning hierarchical representations.

**Architecture:**
```
Input → AE1(Encoder) → AE2(Encoder) → ... → Latent → ... → AE2(Decoder) → AE1(Decoder) → Output
```

**Training Approach (Greedy Layer-wise Pretraining):**

**Step 1:** Train first autoencoder on raw input
```
x → [Encoder1] → h1 → [Decoder1] → x̂
```

**Step 2:** Train second autoencoder on h1
```
h1 → [Encoder2] → h2 → [Decoder2] → ĥ1
```

**Step 3:** Stack and fine-tune entire network end-to-end

**Benefits:**
- Learns hierarchical features (low-level → high-level)
- Easier to train deep networks (avoids vanishing gradients)
- Better initialization than random weights
- Each layer captures progressively abstract features

**Intuition:**
Like building understanding in layers:
- Layer 1: Edges, basic patterns
- Layer 2: Shapes, textures
- Layer 3: Object parts
- Layer 4: Whole objects

**Applications:**
- Deep feature learning
- Pretraining deep networks
- Transfer learning

---

## Question 12
- [ ] Done

**Describe an application of autoencoders in natural language processing (NLP).**

**Application: Sentence/Document Embeddings**

**Definition:**
Autoencoders can learn compact vector representations (embeddings) of text by encoding sentences/documents into a fixed-size latent vector and reconstructing the original text.

**Architecture (Sequence-to-Sequence Autoencoder):**
```
Input sentence → [LSTM Encoder] → Latent vector → [LSTM Decoder] → Reconstructed sentence
```

**How It Works:**
1. Tokenize input text into word embeddings
2. Encoder (LSTM/Transformer) processes sequence → final hidden state = latent
3. Decoder generates text word-by-word from latent
4. Train to reconstruct original sentence

**NLP Applications:**

| Application | How Autoencoder Helps |
|-------------|----------------------|
| **Text Compression** | Encode documents to small vectors |
| **Paraphrase Detection** | Similar sentences → similar latent codes |
| **Sentence Similarity** | Compare latent vectors with cosine similarity |
| **Text Generation** | Sample/interpolate in latent space |
| **Spelling Correction** | Denoising autoencoder on character sequences |
| **Machine Translation** | Encoder-decoder with different language pairs |

**Example: Variational Sentence Autoencoder**
- Encode sentence to distribution (μ, σ)
- Sample latent vector
- Decode to generate coherent sentences
- Enables smooth interpolation between sentences

---

## Question 13
- [ ] Done

**How does backpropagation work in training an autoencoder?**

**Definition:**
Backpropagation in autoencoders works the same as in any neural network - computing gradients of the loss with respect to all weights by applying the chain rule backwards from output to input.

**Training Process:**

**Step 1: Forward Pass**
```
x → Encoder → z → Decoder → x̂
```

**Step 2: Compute Loss**
$$\mathcal{L} = ||x - \hat{x}||^2 \quad \text{(MSE loss)}$$

**Step 3: Backward Pass (Backpropagation)**
- Compute $\frac{\partial \mathcal{L}}{\partial \hat{x}}$ (gradient at output)
- Backpropagate through decoder: $\frac{\partial \mathcal{L}}{\partial W_{decoder}}$
- Backpropagate through bottleneck: $\frac{\partial \mathcal{L}}{\partial z}$
- Backpropagate through encoder: $\frac{\partial \mathcal{L}}{\partial W_{encoder}}$

**Step 4: Update Weights**
$$W \leftarrow W - \eta \frac{\partial \mathcal{L}}{\partial W}$$

**Key Points:**
- Loss compares input $x$ with reconstruction $\hat{x}$
- Gradients flow from output layer back through decoder and encoder
- Both encoder and decoder weights updated simultaneously
- Bottleneck creates information compression constraint

**Chain Rule Application:**
$$\frac{\partial \mathcal{L}}{\partial W_{enc}} = \frac{\partial \mathcal{L}}{\partial \hat{x}} \cdot \frac{\partial \hat{x}}{\partial z} \cdot \frac{\partial z}{\partial W_{enc}}$$

**Special Case - VAE:**
Reparameterization trick enables backprop through stochastic sampling layer.

---

## Question 14
- [ ] Done

**Describe how autoencoders can be integrated into a semi-supervised learning framework.**

**Definition:**
In semi-supervised learning, autoencoders leverage large amounts of unlabeled data to learn useful representations, which are then used with small amounts of labeled data for classification.

**Integration Approaches:**

**Approach 1: Pretrain + Fine-tune**
```
Step 1: Train autoencoder on ALL data (labeled + unlabeled) - unsupervised
Step 2: Remove decoder, keep encoder
Step 3: Add classification head on top of encoder
Step 4: Fine-tune with labeled data only
```

**Approach 2: Joint Training**
$$\mathcal{L} = \underbrace{\mathcal{L}_{reconstruction}}_{\text{all data}} + \lambda \cdot \underbrace{\mathcal{L}_{classification}}_{\text{labeled data only}}$$

**Approach 3: Ladder Networks**
- Add lateral connections between encoder and decoder
- Denoising cost at each layer
- Classification loss on top

**Why It Works:**
- Autoencoder learns data manifold from unlabeled data
- Latent features capture underlying structure
- Classifier benefits from pretrained representations
- Less labeled data needed

**Intuition:**
Like learning to recognize animals:
1. First, observe thousands of animal images (unlabeled) → learn shapes, textures, patterns
2. Then, with just a few labeled examples, quickly learn to distinguish cat vs dog

**Benefits:**
- Reduces need for labeled data
- Better generalization
- Utilizes abundant unlabeled data

---

## Question 15
- [ ] Done

**Explain how autoencoders can be used for domain adaptation.**

**Definition:**
Domain adaptation addresses the problem of training on source domain data and deploying on target domain data (with different distributions). Autoencoders learn domain-invariant representations that transfer across domains.

**Approaches:**

**Approach 1: Shared Encoder**
```
Source data → Shared Encoder → Domain-invariant latent → Source Decoder
Target data → Shared Encoder → Domain-invariant latent → Target Decoder
```
- Single encoder learns features useful for both domains
- Separate decoders handle domain-specific reconstruction

**Approach 2: Domain Adversarial Autoencoder**
- Encoder learns latent representation
- Domain classifier tries to predict source vs target from latent
- Encoder trained to fool domain classifier (adversarial)
- Result: Domain-invariant features

**Loss Function:**
$$\mathcal{L} = \mathcal{L}_{recon} - \lambda \cdot \mathcal{L}_{domain\_classifier}$$

**Approach 3: Cycle-Consistent Autoencoder**
- Learn mappings: Source ↔ Target
- Cycle consistency: Source → Target → Source ≈ Source

**Intuition:**
Like translating between languages - find the underlying meaning (latent) that's independent of the specific language (domain).

**Example Use Cases:**
- Adapting model from synthetic to real images
- Transferring from one medical imaging device to another
- Day-to-night image adaptation

---

## Question 16
- [ ] Done

**What are the challenges and potential solutions in training deep autoencoders?**

**Challenges and Solutions:**

| Challenge | Description | Solutions |
|-----------|-------------|-----------|
| **Vanishing Gradients** | Gradients become tiny in deep networks | ReLU activation, batch normalization, residual connections |
| **Exploding Gradients** | Gradients become very large | Gradient clipping, proper initialization |
| **Poor Initialization** | Bad starting weights slow/prevent convergence | Xavier/He initialization, pretrained weights |
| **Overfitting** | Learns identity function, memorizes data | Regularization, dropout, bottleneck constraint |
| **Mode Collapse (VAE)** | Decoder ignores latent, generates average | KL annealing, free bits |
| **Posterior Collapse (VAE)** | Encoder outputs prior, latent unused | Cyclical annealing, skip connections |
| **Training Instability** | Loss oscillates, doesn't converge | Learning rate scheduling, Adam optimizer |
| **Reconstruction vs Regularization Trade-off** | Balancing both objectives | Tune hyperparameters (β in β-VAE) |

**Practical Solutions:**

**1. Greedy Layer-wise Pretraining:**
- Train one layer at a time
- Stack pretrained layers
- Fine-tune end-to-end

**2. Batch Normalization:**
- Normalize activations in each layer
- Stabilizes training

**3. Skip/Residual Connections:**
- Allow gradients to flow directly
- Help train very deep networks

**4. Learning Rate Scheduling:**
- Start high, decay over time
- Or use warm-up then decay

---

## Question 17
- [ ] Done

**Describe how autoencoders can be used to create embeddings for graph data.**

**Definition:**
Graph autoencoders learn low-dimensional node/graph embeddings by encoding graph structure into latent vectors and reconstructing the adjacency matrix or node features.

**Architecture (Graph Autoencoder - GAE):**
```
Input: Adjacency matrix A, Node features X
Encoder: GCN layers → Node embeddings Z
Decoder: Reconstruct adjacency matrix  = σ(Z · Z^T)
```

**Mathematical Formulation:**

**Encoder (Graph Convolutional Network):**
$$Z = GCN(X, A) = \tilde{A} \cdot ReLU(\tilde{A} \cdot X \cdot W_0) \cdot W_1$$

Where $\tilde{A} = D^{-1/2}AD^{-1/2}$ (normalized adjacency)

**Decoder (Inner Product):**
$$\hat{A}_{ij} = \sigma(z_i^T \cdot z_j)$$

**Loss Function:**
$$\mathcal{L} = -\sum_{i,j} \left[ A_{ij} \log(\hat{A}_{ij}) + (1-A_{ij})\log(1-\hat{A}_{ij}) \right]$$

**Types:**
- **GAE:** Deterministic encoder
- **VGAE (Variational):** Encoder outputs μ, σ for each node

**Applications:**
- Node classification
- Link prediction
- Community detection
- Molecular graph generation
- Social network analysis

**Intuition:**
Similar to image autoencoders, but instead of pixels, we compress graph connectivity patterns into node vectors. Connected nodes should have similar embeddings.

---

## Question 18
- [ ] Done

**What are the current limitations of autoencoders in unsupervised learning applications?**

**Limitations:**

| Limitation | Explanation |
|------------|-------------|
| **Blurry Reconstructions** | MSE loss leads to averaged, blurry outputs (especially images) |
| **No Explicit Density** | Cannot compute $p(x)$ directly (unlike normalizing flows) |
| **Mode Collapse** | May learn only common patterns, miss rare modes |
| **Posterior Collapse (VAE)** | Encoder learns to ignore input, outputs prior |
| **Poor Generation Quality** | Generally worse than GANs/Diffusion for image generation |
| **Latent Space Holes** | Regions in latent space may decode to unrealistic outputs |
| **Hyperparameter Sensitivity** | Latent dimension, β values significantly affect results |
| **Identity Function Risk** | Overcomplete AE may just copy input without learning |
| **Limited Interpretability** | Latent dimensions often not human-interpretable |
| **Training Instability** | VAEs can be unstable, require careful tuning |

**Comparison with Alternatives:**

| Aspect | Autoencoders | GANs | Diffusion Models |
|--------|-------------|------|------------------|
| Generation Quality | Medium | High | Very High |
| Training Stability | Medium | Low | High |
| Latent Space | Structured | Not explicit | Not explicit |
| Density Estimation | VAE only | No | Yes |

**When Autoencoders Still Excel:**
- Anomaly detection
- Feature learning
- Dimensionality reduction
- Denoising
- When structured latent space is needed

---

## Question 19
- [ ] Done

**Explain the potential role of reinforcement learning in enhancing the capabilities of autoencoders.**

**Definition:**
Reinforcement learning (RL) can enhance autoencoders by learning better representations that are useful for specific downstream tasks, using reward signals to guide latent space learning.

**Integration Approaches:**

**1. World Models (RL + VAE)**
```
Environment → VAE Encoder → Latent state z
                    ↓
         RL Agent learns policy π(a|z)
                    ↓
         Actions affect environment
```
- VAE compresses high-dimensional observations (images) to latent state
- RL agent operates in compact latent space
- More efficient learning

**2. Representation Learning for RL**
- Autoencoder learns state representations
- RL uses these as input instead of raw pixels
- Auxiliary reconstruction loss improves representation

**3. Goal-Conditioned RL with VAE**
- Encode goals into latent space
- Agent learns to reach latent goals
- Enables generalization across similar goals

**4. Curiosity-Driven Learning**
- Autoencoder prediction error = intrinsic reward
- High error = novel state = worth exploring
- Encourages exploration

**Benefits:**
- Sample efficiency (compact state representation)
- Better generalization
- Disentangled control factors
- Imagination/planning in latent space

**Example: Dreamer Algorithm**
- VAE learns world model in latent space
- Agent "dreams" trajectories in latent space
- Learns policy from imagined experience

---

## Question 20
- [ ] Done

**Describe a scenario where autoencoders can be used to enhance collaborative filtering in a recommendation system.**

**Scenario: Movie Recommendation System**

**Problem:**
- User-item interaction matrix is sparse (most users rate few movies)
- Need to predict missing ratings

**Autoencoder Solution:**

**Architecture:**
```
Input: User's rating vector (all movies, 0 for unrated)
       ↓
    Encoder → Latent user representation
       ↓
    Decoder → Reconstructed ratings for ALL movies
       ↓
Output: Predicted ratings (including unrated movies)
```

**How It Works:**
1. Input: Sparse user rating vector $r_u \in \mathbb{R}^{|items|}$
2. Encoder learns user preferences in latent space
3. Decoder predicts ratings for all items
4. Recommend items with highest predicted ratings

**Loss Function (Masked):**
$$\mathcal{L} = \sum_{i \in \text{rated}} (r_{ui} - \hat{r}_{ui})^2$$
Only compute loss on observed ratings.

**Enhancements:**
- **Denoising:** Add noise to input → more robust
- **Variational:** VAE for better generalization
- **Side Information:** Include user/item features

**Benefits over Matrix Factorization:**
- Captures nonlinear patterns
- Handles implicit feedback
- Can incorporate additional features
- Better cold-start handling with side information

**Example Flow:**
```
User A: [5, ?, 3, ?, 4, ?] → Encoder → z → Decoder → [5.1, 4.2, 2.9, 1.5, 3.9, 4.8]
Recommend: Items 2 and 6 (highest predicted ratings)
```

---

## Question 21
- [ ] Done

**Define an autoencoder and its reconstruction objective.**

**Definition:**
An autoencoder is an unsupervised neural network that learns to encode input data into a compressed representation and reconstruct the original input. The reconstruction objective is to minimize the difference between input and output.

**Reconstruction Objective:**
$$\min_{\theta, \phi} \mathbb{E}_{x \sim p_{data}} \left[ \mathcal{L}(x, g_\phi(f_\theta(x))) \right]$$

Where:
- $f_\theta$: Encoder with parameters $\theta$
- $g_\phi$: Decoder with parameters $\phi$
- $\mathcal{L}$: Reconstruction loss

**Common Loss Functions:**

| Data Type | Loss Function | Formula |
|-----------|--------------|---------|
| Continuous | MSE | $||x - \hat{x}||_2^2$ |
| Binary | Binary Cross-Entropy | $-\sum[x\log\hat{x} + (1-x)\log(1-\hat{x})]$ |
| Categorical | Cross-Entropy | $-\sum x \log \hat{x}$ |

**Core Idea:**
The network learns the identity function $\hat{x} \approx x$, but the bottleneck constraint forces it to learn meaningful compression rather than trivial copying.

---

## Question 22
- [ ] Done

**Explain encoder-decoder architecture and bottleneck.**

**Encoder-Decoder Architecture:**

```
Input (n dim) → [Encoder] → Bottleneck (k dim) → [Decoder] → Output (n dim)
                  ↓                                    ↑
            Compression                          Reconstruction
```

**Components:**

| Component | Role | Typical Structure |
|-----------|------|-------------------|
| **Encoder** | Maps input to latent | Dense/Conv layers, decreasing size |
| **Bottleneck** | Compressed representation | Smallest layer, k << n |
| **Decoder** | Reconstructs from latent | Dense/Conv layers, increasing size |

**Bottleneck Importance:**
- Forces information compression
- k < n prevents trivial identity mapping
- Determines compression ratio
- Acts as information "funnel"

**Mathematical View:**
$$z = f_{enc}(x) \in \mathbb{R}^k, \quad \hat{x} = f_{dec}(z) \in \mathbb{R}^n$$

**Bottleneck Size Trade-off:**
- **Too small:** Poor reconstruction, loss of information
- **Too large:** May learn identity, no useful compression
- **Optimal:** Captures essential features, good reconstruction

**Design Principle:**
Bottleneck width should match the intrinsic dimensionality of data.

---

## Question 23
- [ ] Done

**Describe feed-forward vs. convolutional autoencoders.**

| Aspect | Feed-Forward AE | Convolutional AE |
|--------|----------------|------------------|
| **Architecture** | Fully connected layers | Conv + pooling layers |
| **Input** | Flattened vector | Preserves spatial structure |
| **Parameters** | Many (all-to-all connections) | Few (shared kernels) |
| **Best For** | Tabular data, 1D signals | Images, spatial data |
| **Features Learned** | Global patterns | Local → hierarchical patterns |
| **Translation Invariance** | No | Yes |

**Feed-Forward AE:**
```
Input (784) → Dense(256) → Dense(64) → Dense(256) → Output(784)
```

**Convolutional AE:**
```
Input (28×28×1) → Conv → Pool → Conv → Pool → Latent
              → ConvTranspose → Upsample → Output (28×28×1)
```

**When to Use:**
- **Feed-Forward:** Tabular data, small datasets, non-spatial data
- **Convolutional:** Images, video, any data with spatial correlations

**Key Difference:**
Convolutional AE exploits locality - nearby pixels are more related. Feed-forward treats all inputs independently.

---

## Question 24
- [ ] Done

**Explain denoising autoencoders and corruption process.**

**Definition:**
Denoising autoencoders learn to reconstruct clean data from corrupted input. The corruption process adds noise, and the network learns to remove it.

**Corruption Process Types:**

| Type | Process | Formula |
|------|---------|---------|
| **Gaussian Noise** | Add random noise | $\tilde{x} = x + \epsilon, \epsilon \sim \mathcal{N}(0, \sigma^2)$ |
| **Masking Noise** | Zero out random inputs | $\tilde{x}_i = 0$ with prob $p$ |
| **Salt & Pepper** | Random 0s and 1s | $\tilde{x}_i \in \{0, 1\}$ with prob $p$ |
| **Dropout** | Drop features | Similar to masking |

**Training:**
$$\mathcal{L} = ||x - g(f(\tilde{x}))||^2$$

Train to predict clean $x$ from corrupted $\tilde{x}$.

**Why It Works:**
- Cannot simply copy input (input is corrupted)
- Must learn underlying structure to "fill in" corrupted parts
- Learns robust, generalizable features

**Algorithm:**
1. Sample clean data $x$
2. Corrupt: $\tilde{x} = \text{corrupt}(x)$
3. Forward pass: $\hat{x} = \text{decoder}(\text{encoder}(\tilde{x}))$
4. Loss: Compare $\hat{x}$ with original $x$
5. Backpropagate and update

---

## Question 25
- [ ] Done

**Discuss sparsity penalty and sparse autoencoders.**

**Definition:**
Sparse autoencoders add a penalty that encourages most neurons in the hidden layer to be inactive (near zero) for any given input.

**Sparsity Constraint:**
$$\mathcal{L} = \mathcal{L}_{recon} + \lambda \cdot \mathcal{L}_{sparsity}$$

**Sparsity Penalty Options:**

**1. KL Divergence Penalty:**
$$\mathcal{L}_{sparsity} = \sum_j KL(\rho || \hat{\rho}_j)$$

Where:
- $\rho$ = target sparsity (e.g., 0.05)
- $\hat{\rho}_j = \frac{1}{m}\sum_i h_j(x^{(i)})$ = average activation of neuron $j$

**2. L1 Penalty:**
$$\mathcal{L}_{sparsity} = ||h||_1 = \sum_j |h_j|$$

**Why Sparsity Helps:**
- Prevents trivial identity mapping
- Each neuron specializes in specific features
- Allows overcomplete representations (latent dim > input dim)
- More interpretable features

**Intuition:**
Like a team where only few experts are active per task. Each expert becomes highly specialized.

**Hyperparameters:**
- $\rho$: Target sparsity level (typically 0.01-0.1)
- $\lambda$: Weight of sparsity penalty

---

## Question 26
- [ ] Done

**Explain contractive autoencoder and Frobenius norm penalty.**

**Definition:**
Contractive autoencoder penalizes the Frobenius norm of the Jacobian of the encoder, making the learned representation insensitive to small input changes.

**Loss Function:**
$$\mathcal{L} = ||x - \hat{x}||^2 + \lambda ||J_f(x)||_F^2$$

**Jacobian Matrix:**
$$J_f(x) = \frac{\partial h}{\partial x} \in \mathbb{R}^{d_h \times d_x}$$

**Frobenius Norm:**
$$||J_f(x)||_F^2 = \sum_{ij} \left(\frac{\partial h_j}{\partial x_i}\right)^2$$

**For Sigmoid Activation:**
$$\frac{\partial h_j}{\partial x_i} = h_j(1 - h_j) \cdot W_{ji}$$

$$||J_f||_F^2 = \sum_j h_j^2(1-h_j)^2 \sum_i W_{ji}^2$$

**Why It Works:**
- Small Jacobian → small change in latent for small change in input
- Representations "contract" nearby points together
- Learns locally invariant features
- Captures data manifold structure

**Comparison:**
- **Denoising AE:** Robust to specific noise types
- **Contractive AE:** Robust to ANY small perturbation (more general)

---

## Question 27
- [ ] Done

**Describe stacked autoencoders and greedy layer-wise pretraining.**

**Stacked Autoencoder:**
Multiple autoencoder layers stacked to form a deep network. Each layer learns increasingly abstract representations.

**Greedy Layer-wise Pretraining Algorithm:**

**Step 1:** Train first autoencoder on raw input
```
x → [W₁] → h₁ → [W₁'] → x̂
Minimize ||x - x̂||²
```

**Step 2:** Freeze W₁, train second AE on h₁
```
h₁ → [W₂] → h₂ → [W₂'] → ĥ₁
Minimize ||h₁ - ĥ₁||²
```

**Step 3:** Continue for more layers...

**Step 4:** Stack all encoders and fine-tune end-to-end
```
x → [W₁] → [W₂] → [W₃] → z → [W₃'] → [W₂'] → [W₁'] → x̂
```

**Why Greedy Pretraining:**
- Avoids vanishing gradients in deep networks
- Each layer gets good initialization
- Historically important (before BatchNorm, ReLU)

**Benefits:**
- Hierarchical feature learning
- Better weight initialization
- Easier to train deep networks

**Modern Alternative:**
With BatchNorm, ReLU, residual connections - can often train deep autoencoders end-to-end without greedy pretraining.

---

## Question 28
- [ ] Done

**Compare autoencoders with PCA in linear case.**

**Key Insight:**
A linear autoencoder with MSE loss and k-dimensional bottleneck learns the same subspace as PCA with k components.

**Comparison:**

| Aspect | PCA | Linear Autoencoder |
|--------|-----|-------------------|
| Method | Eigendecomposition | Gradient descent |
| Solution | Closed-form | Iterative |
| Subspace | Principal components | Same subspace (not same basis) |
| Computation | O(n³) or O(nmk) | Depends on iterations |
| Scalability | Memory intensive | Batch-friendly |
| Extension | None | Can add nonlinearity |

**Mathematical Equivalence:**
For linear encoder $z = W_{enc}x$ and decoder $\hat{x} = W_{dec}z$:

$$\min ||x - W_{dec}W_{enc}x||^2$$

Optimal solution: $W_{enc}W_{dec}$ spans same subspace as top-k PCA components.

**Important Difference:**
- PCA gives orthogonal principal components
- Linear AE gives a basis for the same subspace, but not necessarily orthogonal

**When Autoencoder Wins:**
- Large datasets (mini-batch training)
- When you want to add nonlinearity
- Online/streaming learning scenarios

---

## Question 29
- [ ] Done

**Explain variational autoencoder (VAE) and reparameterization trick.**

**VAE Definition:**
A variational autoencoder is a generative model that learns a probabilistic mapping from data to a latent distribution, enabling both reconstruction and generation.

**Key Difference from Standard AE:**
- Encoder outputs distribution parameters (μ, σ), not a single point
- Sample latent z from this distribution
- KL divergence regularizes latent to be close to prior N(0,1)

**VAE Loss (ELBO):**
$$\mathcal{L} = \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{Reconstruction}} - \underbrace{D_{KL}(q(z|x) || p(z))}_{\text{KL Regularization}}$$

**Problem with Sampling:**
Cannot backpropagate through random sampling $z \sim \mathcal{N}(\mu, \sigma^2)$

**Reparameterization Trick:**
Instead of: $z \sim \mathcal{N}(\mu, \sigma^2)$

Use: $z = \mu + \sigma \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$

**Why It Works:**
- Randomness moved to $\epsilon$ (not dependent on parameters)
- $z$ is now a deterministic function of $\mu$, $\sigma$, $\epsilon$
- Gradients can flow through $\mu$ and $\sigma$

**Diagram:**
```
x → Encoder → [μ, log σ²]
                  ↓
        z = μ + σ ⊙ ε,  ε ~ N(0,I)   ← Reparameterization
                  ↓
              Decoder → x̂
```

---

## Question 30
- [ ] Done

**Discuss beta-VAE and disentanglement.**

**Definition:**
β-VAE modifies VAE by adding a hyperparameter β > 1 to weight the KL divergence term, encouraging more disentangled latent representations.

**β-VAE Loss:**
$$\mathcal{L} = \mathbb{E}[\log p(x|z)] - \beta \cdot D_{KL}(q(z|x) || p(z))$$

**Disentanglement:**
A representation is disentangled when each latent dimension captures one independent factor of variation.

**Example (Faces):**
- $z_1$: controls smile
- $z_2$: controls hair color
- $z_3$: controls pose
- Each dimension = one independent factor

**Why Higher β Helps:**
- Stronger pressure to match prior N(0,1)
- Forces latent dimensions to be independent
- Prevents entangled representations
- Trade-off: Higher β → worse reconstruction

**β Selection:**

| β Value | Effect |
|---------|--------|
| β = 1 | Standard VAE |
| β > 1 | More disentangled, worse reconstruction |
| β >> 1 | Very disentangled, poor reconstruction |

**Measuring Disentanglement:**
- Factor VAE metric
- DCI disentanglement score
- MIG (Mutual Information Gap)

**Applications:**
- Interpretable representations
- Controlled generation (modify one factor at a time)
- Transfer learning

---

## Question 31
- [ ] Done

**Explain adversarial autoencoders vs. VAEs.**

**Definition:**
Adversarial Autoencoder (AAE) uses adversarial training (discriminator) to match the latent distribution to a prior, instead of KL divergence used in VAE.

**Comparison:**

| Aspect | VAE | Adversarial AE |
|--------|-----|----------------|
| **Regularization** | KL divergence | Adversarial discriminator |
| **Prior matching** | Analytical (closed-form) | Learned (discriminator) |
| **Flexibility** | Limited to simple priors | Any prior distribution |
| **Training** | Single network | Generator + Discriminator |
| **Stability** | More stable | Can be unstable (GAN issues) |

**AAE Architecture:**
```
x → Encoder → z → Decoder → x̂
        ↓
    Discriminator(z) tries to distinguish:
    - Real samples from prior p(z)
    - Fake samples from encoder q(z|x)
```

**AAE Loss:**
$$\mathcal{L} = \mathcal{L}_{recon} + \mathcal{L}_{adversarial}$$

**Advantages of AAE:**
- Can use any prior (mixture of Gaussians, categorical, etc.)
- No need for reparameterization trick
- Often sharper generations than VAE

**Advantages of VAE:**
- Principled probabilistic framework
- More stable training
- Analytical KL computation

---

## Question 32
- [ ] Done

**Describe sequence autoencoders with RNNs.**

**Definition:**
Sequence autoencoders use RNNs (LSTM/GRU) to encode variable-length sequences into fixed-size latent vectors and decode back to sequences.

**Architecture:**
```
Input sequence: [x₁, x₂, ..., xₜ]
        ↓
    RNN Encoder (processes sequentially)
        ↓
    Final hidden state = Latent z
        ↓
    RNN Decoder (generates sequentially)
        ↓
Output sequence: [x̂₁, x̂₂, ..., x̂ₜ]
```

**Encoder:**
- Processes input sequence token by token
- Final hidden state captures entire sequence information
- $z = h_T$ (final hidden state)

**Decoder:**
- Initialized with latent z
- Generates output sequence autoregressively
- At each step: $\hat{x}_t = f(h_{t-1}, \hat{x}_{t-1})$

**Applications:**
- Text compression/reconstruction
- Sentence embeddings
- Video summarization
- Time series compression

**Variations:**
- Bidirectional encoder
- Attention mechanism (Transformer)
- Variational sequence autoencoder

---

## Question 33
- [ ] Done

**Discuss role of latent space dimensionality.**

**Impact of Latent Dimension:**

| Dimension | Effect |
|-----------|--------|
| **Too Small** | Information loss, poor reconstruction |
| **Too Large** | Overfitting, may learn identity |
| **Optimal** | Good compression, meaningful features |

**Factors for Choosing Dimension:**

1. **Data Complexity:** Complex data needs larger latent space
2. **Intrinsic Dimensionality:** Match true data manifold dimension
3. **Task:** Generation needs larger than classification
4. **Regularization:** VAE/sparse AE can handle larger dimensions

**Trade-offs:**
- Small: Forces compression, may lose detail
- Large: Better reconstruction, risk of overfitting

**Practical Guidelines:**
- Start with dimension = 10-20% of input dimension
- Use validation loss to tune
- For images: 32-256 typically works
- For tabular: 2-10 dimensions

**Visualization:**
- 2D/3D latent: Can visualize directly
- Higher: Use t-SNE/UMAP for visualization

---

## Question 34
- [ ] Done

**Explain KL divergence term in VAE loss.**

**Definition:**
The KL divergence term in VAE measures how much the learned posterior $q(z|x)$ differs from the prior $p(z) = \mathcal{N}(0, I)$.

**VAE Loss:**
$$\mathcal{L} = -\mathbb{E}_{q(z|x)}[\log p(x|z)] + D_{KL}(q(z|x) || p(z))$$

**KL Divergence for Gaussians:**
$$D_{KL}(q||p) = \frac{1}{2}\sum_{j=1}^{d} \left[ \mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1 \right]$$

**Role of KL Term:**

| Purpose | Explanation |
|---------|-------------|
| **Regularization** | Prevents encoder from creating arbitrary distributions |
| **Structured Latent** | Forces latent to be close to N(0,1) |
| **Continuity** | Similar inputs → similar latent codes |
| **Generation** | Can sample from prior and decode meaningful outputs |

**Without KL:**
- Latent space becomes irregular
- "Holes" in latent space decode to garbage
- Cannot generate by sampling from prior

**With KL:**
- Smooth, continuous latent space
- Any point in latent space decodes to reasonable output
- Enables interpolation and generation

---

## Question 35
- [ ] Done

**Describe autoencoders for image super-resolution.**

**Definition:**
Super-resolution autoencoders learn to map low-resolution images to high-resolution versions by learning the inverse of the downsampling process.

**Architecture:**
```
Low-res image (32×32) → Encoder → Latent → Decoder/Upsampler → High-res image (128×128)
```

**Key Components:**
- **Encoder:** Extracts features from low-res input
- **Decoder:** Upsamples using transposed convolutions or sub-pixel convolution
- **Skip connections:** Preserve fine details

**Loss Functions:**
- **Pixel-wise MSE:** $||I_{HR} - I_{SR}||^2$ (can cause blurriness)
- **Perceptual Loss:** Compare features from pretrained VGG
- **Adversarial Loss:** Add discriminator for sharper outputs

**Techniques:**
- Sub-pixel convolution (pixel shuffle)
- Residual learning (learn residual, not full image)
- Progressive upsampling

**Intuition:**
The network learns patterns: edges → textures → object parts. When upsampling, it "hallucinates" plausible high-frequency details based on learned patterns.

---

## Question 36
- [ ] Done

**Explain anomaly detection via reconstruction error.**

**Principle:**
Train autoencoder on normal data only. Anomalies will have high reconstruction error because the model hasn't learned to reconstruct them.

**Algorithm:**
1. Train AE on normal data: minimize $||x - \hat{x}||^2$
2. Set threshold $\tau$ based on training reconstruction errors
3. For new data: if $||x_{new} - \hat{x}_{new}||^2 > \tau$ → anomaly

**Why It Works:**
- AE learns to compress and reconstruct normal patterns
- Anomalies don't fit learned patterns → poor reconstruction
- High error = doesn't match learned data manifold

**Threshold Selection:**
- Use validation set with known anomalies
- Percentile-based: 95th or 99th percentile of training errors
- Statistical: mean + k×std of training errors

**Applications:**
- Fraud detection
- Manufacturing defect detection
- Network intrusion detection
- Medical anomaly detection

**Variations:**
- VAE: Use reconstruction probability or ELBO
- Use ensemble of autoencoders
- Consider both reconstruction error and latent space distance

---

## Question 37
- [ ] Done

**Discuss limitations: overfitting and identity function risk.**

**Overfitting:**
- AE memorizes training data instead of learning generalizable features
- Perfect training reconstruction, poor on new data

**Identity Function Risk:**
- With too much capacity, AE learns trivial identity: $\hat{x} = x$
- No useful compression or feature learning
- Happens with overcomplete AE (latent dim ≥ input dim) without regularization

**Causes:**
- Latent dimension too large
- Too many parameters
- Insufficient regularization
- Not enough training data

**Solutions:**

| Problem | Solution |
|---------|----------|
| Overfitting | Dropout, early stopping, data augmentation |
| Identity mapping | Bottleneck constraint, sparsity, noise |
| Both | Denoising, contractive penalty, VAE regularization |

**Detection:**
- Compare train vs validation reconstruction error
- Visualize reconstructions - too perfect = suspicious
- Check if latent codes are spread out or clustered

---

## Question 38
- [ ] Done

**Explain tied weights and weight sharing.**

**Definition:**
Tied weights means decoder weights are the transpose of encoder weights: $W_{dec} = W_{enc}^T$

**Standard vs Tied:**

| Type | Encoder | Decoder | Parameters |
|------|---------|---------|------------|
| Untied | $W_1$ | $W_2$ | 2× weights |
| Tied | $W$ | $W^T$ | 1× weights |

**Mathematical Formulation:**
```
Encoder: h = σ(Wx + b₁)
Decoder: x̂ = σ(W^T h + b₂)
```

**Benefits of Tied Weights:**
- Fewer parameters (reduced overfitting)
- Implicit regularization
- Forces symmetric learning
- Faster training

**Drawbacks:**
- Less flexible
- May limit model capacity
- Not always optimal

**When to Use:**
- Limited training data
- Want regularization
- Simpler model preferred

**Implementation:**
```python
# In PyTorch
decoder_weight = encoder.weight.t()  # Transpose of encoder weight
```

---

## Question 39
- [ ] Done

**Describe importance of activation choice (ReLU, sigmoid).**

**Activation Functions in Autoencoders:**

| Layer | Common Activations | Reason |
|-------|-------------------|--------|
| **Hidden layers** | ReLU, LeakyReLU | Non-linearity, avoids vanishing gradient |
| **Latent layer** | Linear or ReLU | Depends on desired latent properties |
| **Output layer** | Sigmoid (0-1), Tanh (-1,1), Linear | Match data range |

**Activation Choice by Data Type:**

| Data Type | Output Activation | Loss |
|-----------|------------------|------|
| Images [0,1] | Sigmoid | BCE |
| Images [-1,1] | Tanh | MSE |
| Real-valued | Linear | MSE |
| Binary | Sigmoid | BCE |

**ReLU Benefits:**
- Mitigates vanishing gradients
- Sparse activations
- Computationally efficient

**ReLU Issues:**
- Dead neurons (LeakyReLU fixes this)
- Unbounded (can use batch normalization)

**Key Principle:**
Output activation must match data range. Mismatched activation = training issues.

---

## Question 40
- [ ] Done

**Discuss training with dropout inside autoencoders.**

**Definition:**
Dropout randomly sets neuron outputs to zero during training, acting as regularization.

**Dropout in AE:**
```
x → [Encoder with Dropout] → z → [Decoder with Dropout] → x̂
```

**Effects:**
- Regularization: Prevents overfitting
- Robustness: Model learns redundant representations
- Similar to ensemble: Averages many sub-networks

**Where to Apply Dropout:**
- **Encoder hidden layers:** Most common
- **Latent layer:** Creates denoising effect
- **Decoder hidden layers:** Additional regularization
- **Input layer:** Essentially becomes denoising AE

**Dropout Rate Guidelines:**
- Hidden layers: 0.2-0.5
- Input/latent: 0.1-0.3 (lower)

**Dropout at Input = Denoising AE:**
Applying dropout to input is equivalent to masking noise corruption.

**Important:**
- Turn off dropout during inference
- Adjust for proper probability scaling

---

## Question 41
- [ ] Done

**Explain contractive vs. Jacobian regularization.**

**Both penalize the Jacobian but with different goals:**

**Contractive Autoencoder:**
$$\mathcal{L} = \mathcal{L}_{recon} + \lambda ||J||_F^2$$

Penalizes Frobenius norm of Jacobian $J = \frac{\partial h}{\partial x}$

**Goal:** Make encoder insensitive to input perturbations.

**Jacobian Regularization (general):**
Various penalties on Jacobian for different purposes:
- Spectral norm: Control largest singular value
- Orthogonality: $||J^TJ - I||^2$

**Comparison:**

| Aspect | Contractive AE | General Jacobian Reg |
|--------|---------------|---------------------|
| What's penalized | All derivatives equally | Can target specific properties |
| Effect | Contracts neighborhood | Depends on penalty |
| Computation | Requires derivative computation | Same |

**Contractive Effect:**
Small input perturbations → very small latent changes. Representations are "contracted" to be robust.

---

## Question 42
- [ ] Done

**Describe InfoVAE and MMD regularization.**

**Definition:**
InfoVAE replaces or augments KL divergence with Maximum Mean Discrepancy (MMD), a kernel-based distance between distributions.

**Standard VAE:**
$$\mathcal{L} = \mathcal{L}_{recon} + D_{KL}(q(z|x)||p(z))$$

**InfoVAE:**
$$\mathcal{L} = \mathcal{L}_{recon} + \alpha \cdot D_{KL} + \lambda \cdot MMD(q(z)||p(z))$$

**MMD (Maximum Mean Discrepancy):**
$$MMD^2 = \mathbb{E}[k(z,z')] + \mathbb{E}[k(\tilde{z},\tilde{z}')] - 2\mathbb{E}[k(z,\tilde{z})]$$

Where $k$ is a kernel (e.g., RBF), $z \sim q(z)$, $\tilde{z} \sim p(z)$

**Why MMD:**
- Matches aggregate posterior $q(z)$ to prior (not per-sample)
- No closed-form requirement
- Can use any prior distribution
- Avoids posterior collapse issues

**Benefits over KL:**
- Better latent space utilization
- Works with complex priors
- More stable training

---

## Question 43
- [ ] Done

**Explain autoencoder-based collaborative filtering.**

**Definition:**
Using autoencoders to learn user/item representations from interaction data for recommendations.

**Architecture:**
```
User rating vector [r₁, r₂, ..., rₙ] (with missing values)
        ↓
    Encoder → User embedding
        ↓
    Decoder → Predicted ratings [r̂₁, r̂₂, ..., r̂ₙ]
```

**Training:**
- Input: User's known ratings (0 for unknown)
- Output: Predict ALL ratings
- Loss: Only on known ratings (masked loss)

$$\mathcal{L} = \sum_{(u,i) \in \text{known}} (r_{ui} - \hat{r}_{ui})^2$$

**Variations:**
- **Item-based:** Input = item's rating profile
- **Variational:** VAE for better generalization
- **Deep:** Multiple layers for complex patterns

**Benefits over Matrix Factorization:**
- Captures nonlinear patterns
- Easy to add side information
- Handles sparse data well

**Recommendation:**
For user u, recommend items with highest predicted $\hat{r}_{ui}$.

---

## Question 44
- [ ] Done

**Describe graph autoencoders for network embeddings.**

**Definition:**
Graph autoencoders encode graph structure (nodes, edges) into low-dimensional node embeddings and reconstruct the adjacency matrix.

**Architecture (GAE):**
```
Input: A (adjacency), X (node features)
        ↓
    GCN Encoder: Z = GCN(A, X)
        ↓
    Inner Product Decoder: Â = σ(Z·Z^T)
        ↓
Output: Reconstructed adjacency matrix
```

**Encoder (GCN):**
$$Z = \text{ReLU}(\tilde{A} \cdot \text{ReLU}(\tilde{A} X W_0) \cdot W_1)$$

**Decoder:**
$$\hat{A}_{ij} = \sigma(z_i^T z_j)$$

**Loss (Link Prediction):**
$$\mathcal{L} = -\sum_{i,j} [A_{ij}\log\hat{A}_{ij} + (1-A_{ij})\log(1-\hat{A}_{ij})]$$

**VGAE (Variational):**
- Encoder outputs $\mu$, $\sigma$ for each node
- Sample node embeddings from Gaussian
- Add KL divergence term

**Applications:**
- Link prediction
- Node classification
- Community detection
- Knowledge graph completion

---

## Question 45
- [ ] Done

**Explain vector quantized VAE for discrete latents.**

**Definition:**
VQ-VAE uses discrete latent codes from a learned codebook instead of continuous latents. Each latent vector is quantized to its nearest codebook entry.

**Architecture:**
```
x → Encoder → z_e → Quantize → z_q → Decoder → x̂
                      ↓
              Codebook: {e₁, e₂, ..., eₖ}
```

**Quantization:**
$$z_q = e_k \quad \text{where} \quad k = \arg\min_j ||z_e - e_j||^2$$

**Loss Function:**
$$\mathcal{L} = ||x - \hat{x}||^2 + ||sg[z_e] - e||^2 + \beta||z_e - sg[e]||^2$$

Where sg = stop gradient

**Components:**
- Reconstruction loss
- Codebook loss (move codebook towards encoder output)
- Commitment loss (encoder commits to codebook entries)

**Benefits:**
- Discrete representation (useful for language, audio)
- Avoids posterior collapse
- High-quality generation
- Used in DALL-E, audio synthesis

---

## Question 46
- [ ] Done

**Discuss autoencoders for multimodal fusion.**

**Definition:**
Multimodal autoencoders learn joint representations from multiple data modalities (text, image, audio, etc.).

**Architectures:**

**1. Shared Latent Space:**
```
Image → Image Encoder ↘
                       → Shared Latent → Decoders
Text → Text Encoder   ↗
```

**2. Cross-Modal Generation:**
```
Image → Encoder → z → Text Decoder → Caption
```

**3. Concatenation:**
```
Image features ⊕ Text features → Joint Encoder → z → Decoders
```

**Training Strategies:**
- Minimize reconstruction loss for each modality
- Align latent spaces across modalities
- Cross-modal reconstruction (image→text→image)

**Applications:**
- Image captioning
- Visual question answering
- Audio-visual learning
- Cross-modal retrieval

**Benefits:**
- Learn correlations between modalities
- Handle missing modalities
- Transfer learning across modalities

---

## Question 47
- [ ] Done

**Explain Wasserstein autoencoders.**

**Definition:**
Wasserstein Autoencoder (WAE) uses Wasserstein distance (optimal transport) to match the aggregate posterior to the prior, instead of KL divergence.

**WAE Objective:**
$$\mathcal{L} = \mathbb{E}[c(x, \hat{x})] + \lambda \cdot D_Z(q(z), p(z))$$

Where $D_Z$ is a divergence measure on latent distributions.

**Two Variants:**

**WAE-MMD:**
$$D_Z = MMD(q(z), p(z))$$
Uses Maximum Mean Discrepancy (kernel-based)

**WAE-GAN:**
$$D_Z = \text{Adversarial loss}$$
Uses discriminator to match distributions

**Advantages over VAE:**
- More flexible prior distributions
- No need for analytical KL computation
- Often produces sharper reconstructions
- Based on optimal transport theory

**Key Insight:**
Instead of matching $q(z|x)$ to prior for each x (VAE), match aggregate $q(z) = \int q(z|x)p(x)dx$ to prior.

---

## Question 48
- [ ] Done

**Describe beta-TCVAE penalizing total correlation.**

**Definition:**
β-TCVAE decomposes the KL term in VAE to specifically penalize total correlation, leading to better disentanglement.

**KL Decomposition:**
$$D_{KL}(q(z|x)||p(z)) = \underbrace{I(x;z)}_{\text{MI}} + \underbrace{KL(q(z)||p(z))}_{\text{Marginal KL}}$$

Further decompose marginal KL:
$$KL(q(z)||p(z)) = \underbrace{KL(q(z)||\prod_j q(z_j))}_{\text{Total Correlation}} + \underbrace{\sum_j KL(q(z_j)||p(z_j))}_{\text{Dimension-wise KL}}$$

**β-TCVAE Loss:**
$$\mathcal{L} = \mathcal{L}_{recon} + \alpha \cdot MI + \beta \cdot TC + \gamma \cdot \text{Dim-wise KL}$$

**Total Correlation:**
Measures dependence between latent dimensions. Low TC → independent (disentangled) latents.

**Benefits:**
- More principled disentanglement than β-VAE
- Can tune β specifically for total correlation
- Better trade-off between reconstruction and disentanglement

---

## Question 49
- [ ] Done

**Discuss InfoGAN vs. autoencoder generative approaches.**

**Comparison:**

| Aspect | InfoGAN | VAE/Autoencoders |
|--------|---------|------------------|
| **Type** | GAN-based | Reconstruction-based |
| **Latent** | Disentangled codes c + noise z | Learned encoding |
| **Training** | Adversarial | Reconstruction + regularization |
| **Objective** | Maximize mutual info I(c;G(z,c)) | Maximize ELBO |
| **Generation** | High quality | Often blurry |
| **Latent Control** | Structured by design | Learned structure |

**InfoGAN:**
- Adds structured latent codes c to random noise z
- Maximizes mutual information between c and generated output
- Learns disentangled, interpretable latent codes
- No encoder (can't encode new samples)

**VAE/AAE:**
- Has encoder (can encode new samples)
- Learned latent structure
- Reconstruction objective ensures faithfulness
- Can do both encoding and generation

**When to Use:**
- InfoGAN: Pure generation with interpretable control
- VAE: When need both encoding and generation
- AAE: GAN quality with encoding capability

---

## Question 50
- [ ] Done

**Explain training stability issues with VAEs.**

**Common Stability Issues:**

| Issue | Description | Solution |
|-------|-------------|----------|
| **Posterior Collapse** | Encoder ignores input, outputs prior | KL annealing, free bits |
| **KL Vanishing** | KL term becomes zero too quickly | Warm-up schedule |
| **Mode Collapse** | Generates only few variations | More expressive decoder |
| **Reconstruction-KL Trade-off** | Hard to balance both losses | β-VAE tuning |
| **Blurry Outputs** | MSE loss causes averaging | Perceptual loss, adversarial loss |

**Posterior Collapse:**
- Decoder becomes too powerful, ignores latent z
- KL term goes to zero (q matches prior exactly)
- Latent carries no information

**Solutions:**

**1. KL Annealing/Warm-up:**
Start with β=0, gradually increase to 1 over training.

**2. Free Bits:**
$$\mathcal{L}_{KL} = \sum_j \max(\lambda, KL_j)$$
Ensure minimum information per dimension.

**3. Cyclical Annealing:**
Cycle β from 0 to 1 multiple times during training.

**4. δ-VAE:**
Cap the KL term below a threshold.

**5. Stronger Prior:**
Use more expressive prior (mixture of Gaussians).

---

## Question 51
- [ ] Done

**Provide pseudo-code for training a basic autoencoder.**

```
Algorithm: Train Basic Autoencoder
─────────────────────────────────────
Input: Dataset X, learning_rate, epochs, batch_size
Output: Trained encoder, decoder

1. Initialize encoder and decoder weights randomly
2. For epoch = 1 to epochs:
   3. Shuffle dataset X
   4. For each mini-batch B in X:
      5. Forward Pass:
         z = encoder(B)           # Encode to latent
         x_hat = decoder(z)       # Decode to reconstruction
      
      6. Compute Loss:
         loss = MSE(B, x_hat)     # Reconstruction loss
      
      7. Backward Pass:
         gradients = backprop(loss)
      
      8. Update Weights:
         encoder.weights -= learning_rate * gradients_encoder
         decoder.weights -= learning_rate * gradients_decoder
   
   9. Print epoch loss
10. Return encoder, decoder
```

**Key Steps:**
1. Forward: x → encoder → z → decoder → x̂
2. Loss: Compare x with x̂
3. Backward: Compute gradients
4. Update: Adjust weights

---

## Question 52
- [ ] Done

**Describe visualization of latent space via t-SNE.**

**Definition:**
t-SNE (t-distributed Stochastic Neighbor Embedding) is used to visualize high-dimensional latent representations in 2D/3D.

**Process:**
```
Data X → Autoencoder → Latent Z (high-dim) → t-SNE → 2D visualization
```

**Algorithm Steps:**
1. Encode all data points to latent space
2. Compute pairwise similarities in latent space
3. Initialize random 2D positions
4. Iteratively adjust 2D positions to preserve similarities
5. Plot 2D points, color by class/label

**What Good Latent Space Looks Like:**
- Same-class points cluster together
- Different classes are separated
- Smooth transitions between similar points

**Interpretation:**

| Observation | Meaning |
|-------------|---------|
| Tight clusters | Good class separation |
| Overlapping clusters | Poor discrimination |
| Linear structure | Could use PCA instead |
| Manifold structure | AE learned meaningful representation |

**Alternatives:**
- UMAP (faster, preserves global structure better)
- PCA (linear, fast)

**Caution:**
t-SNE perplexity affects results. Try multiple values (5-50).

---

## Question 53
- [ ] Done

**Explain conditional VAEs for label-controlled generation.**

**Definition:**
Conditional VAE (CVAE) conditions both encoder and decoder on additional information (labels, attributes) to enable controlled generation.

**Architecture:**
```
Input x, label y
        ↓
Encoder(x, y) → μ, σ
        ↓
z = μ + σε
        ↓
Decoder(z, y) → x̂
```

**Loss Function:**
$$\mathcal{L} = -\mathbb{E}_{q(z|x,y)}[\log p(x|z,y)] + D_{KL}(q(z|x,y)||p(z|y))$$

**Training:**
- Encoder takes (x, y) as input
- Decoder takes (z, y) as input
- Learn p(x|y) conditioned on labels

**Controlled Generation:**
```
1. Choose desired label y (e.g., "digit 7")
2. Sample z from prior
3. Decode(z, y) → Generate sample of class y
```

**Applications:**
- Generate specific digit (MNIST)
- Generate face with specific attributes
- Text generation with style control
- Drug molecule generation with desired properties

**Conditioning Methods:**
- Concatenate y to input
- Use y as separate input branch
- Feature-wise transformation (FiLM)

---

## Question 54
- [ ] Done

**Discuss ladder network and denoising cost.**

**Definition:**
Ladder Network combines supervised and unsupervised learning by adding denoising costs at each layer, enabling semi-supervised learning.

**Architecture:**
```
Clean path:    x → h₁ → h₂ → ... → y (classification)
Noisy path:    x̃ → h̃₁ → h̃₂ → ... → ỹ
                ↓      ↓       ↓
Decoder:       ẑ₁ ← ẑ₂ ← ... (reconstruction)
```

**Denoising Cost:**
At each layer l:
$$C_l = ||\hat{z}_l - z_l||^2$$

Total unsupervised cost: $C_d = \sum_l \lambda_l C_l$

**Total Loss:**
$$\mathcal{L} = C_{supervised} + C_d$$

Where $C_{supervised}$ uses labeled data only, $C_d$ uses all data.

**Key Ideas:**
- Lateral connections between noisy and clean paths
- Denoising at multiple levels of abstraction
- Works with very few labels

**Benefits:**
- State-of-art semi-supervised results
- Uses unlabeled data effectively
- Learns hierarchical features

---

## Question 55
- [ ] Done

**Explain using autoencoders for feature compression on edge devices.**

**Use Case:**
Deploy ML models on resource-constrained devices (phones, IoT, embedded systems) by compressing features with autoencoders.

**Architecture:**
```
Edge Device:    Sensor → [Light Encoder] → Compressed features
                              ↓ (transmit small data)
Cloud:          Compressed features → [Full Model] → Prediction
```

**Approaches:**

**1. Split Computing:**
- Encoder on edge device
- Decoder + classifier on cloud
- Transmit latent codes (smaller than raw data)

**2. Model Compression:**
- Train autoencoder on intermediate features
- Replace layers with compressed version
- Smaller model footprint

**Benefits:**
- Reduced bandwidth (transmit latent codes, not raw data)
- Lower compute on edge (encoder is small)
- Privacy (raw data stays on device)

**Optimization for Edge:**
- Quantization (8-bit instead of 32-bit)
- Pruning (remove unnecessary weights)
- Knowledge distillation

**Example:**
Image (224×224×3) = 150KB → Encoder → Latent (256 floats) = 1KB
100× compression, transmit over slow network.

---

## Question 56
- [ ] Done

**Describe use in dimensionality reduction for scRNA-seq.**

**Definition:**
Single-cell RNA sequencing (scRNA-seq) produces high-dimensional gene expression data. Autoencoders reduce dimensionality for visualization and analysis.

**Challenge:**
- ~20,000 genes per cell
- Millions of cells
- Sparse, noisy data
- Need to find cell types, trajectories

**Autoencoder Solution:**
```
Gene expression (20,000 dim) → Encoder → Latent (32 dim) → Decoder → Reconstruction
```

**Popular Methods:**
- **scVI:** VAE for scRNA-seq with count distributions
- **DCA:** Deep Count Autoencoder (handles zero-inflation)
- **SAUCIE:** Clustering + batch correction

**Why AE over PCA:**
- Captures nonlinear relationships
- Handles sparse count data
- Can model noise distributions
- Better clustering results

**Applications:**
- Cell type clustering
- Trajectory inference
- Batch effect correction
- Data denoising

**Output:**
Latent space can be visualized with t-SNE/UMAP to see cell populations.

---

## Question 57
- [ ] Done

**Explain out-of-distribution detection with VAEs.**

**Definition:**
Use VAE to detect samples that are different from training distribution (out-of-distribution or OOD).

**Methods:**

**1. Reconstruction Error:**
$$\text{OOD score} = ||x - \hat{x}||^2$$
High error → likely OOD

**2. ELBO/Likelihood:**
$$\text{OOD score} = -\text{ELBO}(x)$$
Low ELBO → doesn't fit model → OOD

**3. Latent Space Analysis:**
- OOD samples may have unusual latent codes
- Far from training data distribution in latent space

**Challenge:**
VAEs can assign high likelihood to OOD samples (known issue).

**Solutions:**
- Use likelihood ratio instead of absolute likelihood
- Combine with input complexity measure
- Ensemble of VAEs
- Use reconstruction + KL together

**Algorithm:**
1. Train VAE on in-distribution data
2. For new sample x:
   - Compute ELBO or reconstruction error
   - If score > threshold → flag as OOD
3. Threshold determined on validation set

---

## Question 58
- [ ] Done

**Discuss variational dropout in autoencoders.**

**Definition:**
Variational dropout treats dropout rates as learnable parameters, allowing the network to learn optimal sparsity patterns.

**Standard Dropout:**
$$h_i = h_i \cdot \epsilon_i, \quad \epsilon_i \sim \text{Bernoulli}(1-p)$$
Fixed dropout rate p.

**Variational Dropout:**
$$h_i = h_i \cdot (1 + \epsilon_i \sigma_i), \quad \epsilon_i \sim \mathcal{N}(0,1)$$
Learn $\sigma_i$ for each connection/neuron.

**In Autoencoders:**
- Learn which latent dimensions are important
- Unused dimensions get high dropout (pruned)
- Automatic dimensionality selection

**Benefits:**
- Adaptive regularization
- Automatic model compression
- No need to tune dropout rate
- Sparse, interpretable latent codes

**Connection to Sparsity:**
High learned dropout rate → connection is unimportant → can be pruned.

---

## Question 59
- [ ] Done

**Explain energy-based autoencoders.**

**Definition:**
Energy-based models (EBMs) assign low energy to data points and high energy to non-data. Energy-based autoencoders use reconstruction error as energy.

**Concept:**
$$E(x) = ||x - \text{decode}(\text{encode}(x))||^2$$

Low energy = good reconstruction = on data manifold
High energy = poor reconstruction = off data manifold

**Training:**
- Minimize energy on real data
- Maximize energy on "negative" samples (contrastive)

**Contrastive Learning:**
$$\mathcal{L} = E(x_{real}) - E(x_{fake})$$

Push real data down, fake data up in energy landscape.

**Score Matching:**
Alternative: match gradient of energy to data score.

**Applications:**
- Anomaly detection (high energy = anomaly)
- Generative modeling
- Representation learning

**Connection:**
Denoising autoencoders implicitly learn energy function - denoising direction points toward low-energy region.

---

## Question 60
- [ ] Done

**Describe hierarchical VAEs with multiple stochastic layers.**

**Definition:**
Hierarchical VAEs have multiple levels of latent variables, each capturing different scales of variation.

**Architecture:**
```
x → Encoder → z₁ → z₂ → ... → zₗ (top level)
                ↓     ↓         ↓
              Prior hierarchy
```

**Generative Model:**
$$p(x, z_1, ..., z_L) = p(z_L) \prod_{l=1}^{L-1} p(z_l|z_{l+1}) \cdot p(x|z_1)$$

**Examples:**

**Ladder VAE:**
- Bottom-up encoder + top-down decoder
- Each level learns different abstraction

**NVAE:**
- Deep hierarchical VAE for images
- State-of-art VAE generation quality

**Benefits:**
- More expressive latent space
- Captures hierarchical structure
- Better generation quality
- Each level = different scale features

**Intuition:**
- Top levels: Global structure (pose, identity)
- Bottom levels: Local details (textures, edges)

---

## Question 61
- [ ] Done

**Explain temporal convolutional autoencoders for anomaly detection in ECG.**

**Application:**
Detect cardiac anomalies by training autoencoder on normal ECG patterns.

**Architecture:**
```
ECG signal (time series) → Temporal Conv Encoder → Latent → Temporal Conv Decoder → Reconstruction
```

**Why Temporal Convolutions:**
- Capture local patterns (heartbeats)
- Handle variable-length sequences
- Efficient (parallelizable)
- Dilated convolutions for long-range dependencies

**Training:**
1. Train on normal ECG signals only
2. Normal patterns: low reconstruction error
3. Anomalies: high reconstruction error

**Detection:**
$$\text{Anomaly score}_t = |x_t - \hat{x}_t|^2$$

Threshold or use sliding window for anomaly detection.

**Types of ECG Anomalies:**
- Arrhythmias
- ST elevation/depression
- Abnormal morphology

**Advantages:**
- Unsupervised (no labeled anomalies needed)
- Real-time detection possible
- Captures temporal dependencies

---

## Question 62
- [ ] Done

**Discuss invertible autoencoders vs. normalizing flows.**

**Comparison:**

| Aspect | Standard Autoencoder | Invertible/Normalizing Flow |
|--------|---------------------|---------------------------|
| **Encoder-Decoder** | Separate networks | Same network, inverted |
| **Information** | Lossy compression | Lossless (bijective) |
| **Likelihood** | Not tractable | Exact computation |
| **Latent dim** | Usually smaller | Must equal input dim |
| **Architecture** | Flexible | Must be invertible |

**Normalizing Flows:**
- Series of invertible transformations
- $x = f_K \circ f_{K-1} \circ ... \circ f_1(z)$
- Exact likelihood via change of variables:
$$\log p(x) = \log p(z) - \sum_k \log|\det J_{f_k}|$$

**Invertible Autoencoders:**
- Use invertible architectures (RevNets)
- Can compute exact reconstruction
- No information loss

**When to Use:**
- **Standard AE:** Feature learning, compression, when lossy is OK
- **Flows:** Density estimation, need exact likelihood

---

## Question 63
- [ ] Done

**Explain integration with generative adversarial networks (BiGAN).**

**Definition:**
BiGAN (Bidirectional GAN) adds an encoder to GAN, learning both generation and inference simultaneously.

**Architecture:**
```
Encoder E: x → z
Generator G: z → x
Discriminator D: distinguishes (x, E(x)) from (G(z), z)
```

**Training:**
$$\min_{G,E} \max_D V(D, G, E)$$
$$V = \mathbb{E}_x[\log D(x, E(x))] + \mathbb{E}_z[\log(1 - D(G(z), z))]$$

**Key Insight:**
Discriminator sees (data, encoding) pairs. Must learn to:
- Match real data with its encoding
- Match generated data with its latent code

**Benefits:**
- GAN quality generation + encoding capability
- Learn useful representations
- Better than standard AE features for downstream tasks

**Comparison:**

| Model | Encode | Generate | Training |
|-------|--------|----------|----------|
| AE | Yes | Poor | Reconstruction |
| GAN | No | Good | Adversarial |
| BiGAN | Yes | Good | Adversarial |

---

## Question 64
- [ ] Done

**Describe Transformer autoencoders for language pretraining.**

**Definition:**
Use Transformer architecture with autoencoding objective for pretraining language models.

**Examples:**

**BERT (Masked Language Model):**
```
Input:  "The [MASK] sat on the mat"
Output: "The cat sat on the mat"
```
- Mask random tokens
- Predict masked tokens from context
- Bidirectional attention

**BART (Denoising Autoencoder):**
```
Corrupted input → Encoder → Decoder → Original text
```
Corruptions: masking, deletion, permutation, infilling

**Architecture:**
```
Text → Tokenize → Transformer Encoder → Latent → Transformer Decoder → Text
```

**Pretraining Objectives:**
- MLM: Mask and predict tokens
- NSP: Next sentence prediction
- Denoising: Reconstruct from corrupted input

**Benefits:**
- Learn contextual representations
- Transfer to downstream tasks
- Capture language structure

**Fine-tuning:**
After pretraining, fine-tune on specific tasks with small labeled data.

---

## Question 65
- [ ] Done

**Discuss defense against adversarial attacks via reconstruction.**

**Problem:**
Adversarial examples: small perturbations that fool classifiers.
$$x_{adv} = x + \epsilon, \quad f(x_{adv}) \neq f(x)$$

**Defense Using Autoencoders:**

**1. Input Purification:**
```
x_adv → Autoencoder → x_clean → Classifier → Prediction
```
AE trained on clean data removes adversarial perturbations.

**2. Denoising Defense:**
- Adversarial perturbations = noise
- Denoising AE removes the perturbation
- Pass cleaned input to classifier

**3. Reconstruction Detection:**
$$||x_{adv} - \text{AE}(x_{adv})||^2 > \tau$$
High reconstruction error → adversarial sample detected

**Why It Works:**
- AE learns manifold of clean data
- Adversarial examples are off-manifold
- Reconstruction projects back to manifold

**Limitations:**
- Adaptive attacks can fool defense
- May not work against all attack types
- Adds computational overhead

---

## Question 66
- [ ] Done

**Explain overcomplete autoencoders and regularization needs.**

**Definition:**
Overcomplete autoencoder has latent dimension ≥ input dimension. Without regularization, it can learn trivial identity mapping.

**Problem:**
```
Input (100 dim) → Latent (200 dim) → Output (100 dim)
```
Trivial solution: Just copy input, no compression learned.

**Why Use Overcomplete:**
- Can learn richer representations
- More neurons = more features
- Useful when combined with regularization

**Required Regularization:**

| Method | How It Prevents Identity |
|--------|-------------------------|
| **Sparsity** | Only few neurons active per input |
| **Denoising** | Must learn structure to denoise |
| **Contractive** | Penalizes sensitivity to input |
| **Dropout** | Random neurons disabled |
| **VAE** | KL divergence regularizes latent |

**Best Practices:**
- Always use regularization with overcomplete AE
- Sparsity constraint most common
- Tune regularization strength

**Trade-off:**
More capacity + strong regularization = rich, useful features
More capacity + no regularization = identity function (useless)

---

## Question 67
- [ ] Done

**Provide an industry use case: predictive maintenance with autoencoders.**

**Scenario:**
Predict equipment failure before it happens using sensor data.

**Setup:**
```
Sensors (vibration, temperature, pressure, etc.) → Time series data
```

**Approach:**

**Training Phase:**
1. Collect sensor data from normal operation
2. Train autoencoder to reconstruct normal patterns
3. Establish baseline reconstruction error

**Deployment:**
1. Continuously feed sensor data to AE
2. Compute reconstruction error in real-time
3. Alert when error exceeds threshold

**Architecture:**
```
Sensor data window → LSTM/Conv Encoder → Latent → Decoder → Reconstruction
                                                         ↓
                                                  Compare with input
                                                         ↓
                                                 Error > threshold? → Alert
```

**Why Autoencoders:**
- Unsupervised: Don't need labeled failure data (rare, expensive)
- Anomalies = failures
- Can detect unknown failure modes

**Real Applications:**
- Manufacturing machinery
- Wind turbines
- Aircraft engines
- Power plants

**Benefits:**
- Reduce unplanned downtime
- Optimize maintenance schedules
- Lower costs

---

## Question 68
- [ ] Done

**Predict future research in self-supervised contrastive autoencoders.**

**Current Trends:**

**1. Combining Contrastive + Reconstruction:**
$$\mathcal{L} = \mathcal{L}_{recon} + \lambda \cdot \mathcal{L}_{contrastive}$$
- Reconstruction: pixel-level fidelity
- Contrastive: semantic-level similarity

**2. Masked Autoencoders (MAE):**
- Mask large portions of input (images, text)
- Predict masked regions
- Very effective for pretraining

**3. Multi-View Learning:**
- Different augmentations of same sample
- Learn representations invariant to augmentation

**Predicted Directions:**

| Area | Future Research |
|------|-----------------|
| **Architecture** | Transformers + autoencoders |
| **Objectives** | Hybrid contrastive + generative |
| **Efficiency** | Compute-efficient large-scale training |
| **Multimodal** | Cross-modal contrastive learning |
| **Theory** | Understanding when contrastive > reconstructive |

**Potential Breakthroughs:**
- Unified SSL framework combining all approaches
- Better understanding of representation properties
- Domain-specific SSL methods (science, medicine)

---

## Question 69
- [ ] Done

**Explain metrics to evaluate autoencoder quality beyond MSE.**

**Evaluation Metrics:**

| Metric | What It Measures | When to Use |
|--------|------------------|-------------|
| **MSE** | Pixel-level error | Basic reconstruction |
| **SSIM** | Structural similarity | Image quality |
| **PSNR** | Peak signal-to-noise | Image quality |
| **FID** | Feature distance (Inception) | Generation quality |
| **LPIPS** | Perceptual similarity | Human perception |
| **KL divergence** | Latent distribution match | VAE regularization |

**Perceptual Metrics:**

**SSIM (Structural Similarity):**
$$SSIM = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

**FID (Frechet Inception Distance):**
Lower = better generation quality (compares feature statistics)

**LPIPS:**
Perceptual loss using pretrained network features.

**Task-Specific Metrics:**
- Downstream task performance (classification accuracy)
- Clustering quality (NMI, ARI) for latent space
- Anomaly detection: AUC-ROC

**Best Practice:**
Use multiple metrics - MSE alone doesn't capture perceptual quality.

---

## Question 70
- [ ] Done

**Summarize pros/cons relative to GANs and diffusion models.**

| Aspect | Autoencoders/VAE | GANs | Diffusion Models |
|--------|------------------|------|------------------|
| **Generation Quality** | Medium (blurry) | High (sharp) | Very High |
| **Training Stability** | Stable | Unstable | Stable |
| **Mode Coverage** | Good | Mode collapse risk | Excellent |
| **Latent Space** | Structured | No encoder | No explicit latent |
| **Likelihood** | VAE: ELBO | No | Yes |
| **Encoding** | Yes | No (BiGAN: Yes) | Indirect |
| **Speed (inference)** | Fast | Fast | Slow |
| **Speed (training)** | Fast | Medium | Slow |

**When to Use Each:**

**Autoencoders:**
- Anomaly detection
- Feature learning
- Dimensionality reduction
- When need encoding capability

**GANs:**
- High-quality image generation
- Style transfer
- Image-to-image translation

**Diffusion Models:**
- State-of-art image/video generation
- When quality > speed
- Diverse samples needed

**Hybrid Approaches:**
- VAE-GAN: VAE + adversarial loss
- Latent Diffusion: Diffusion in AE latent space (Stable Diffusion)
