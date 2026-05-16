# LLM Architectures & Models - Theory Questions

## Core Transformer Architecture

### Question 1
- [ ] Done
**Explain the core innovation of the Transformer architecture and why it replaced RNNs/LSTMs.**

**Answer:**

**Definition:**
The Transformer architecture (Vaswani et al., 2017) replaced recurrent processing with **self-attention mechanism**, enabling parallel computation of all positions simultaneously. Unlike RNNs/LSTMs that process sequences step-by-step, Transformers capture dependencies between any two positions directly, eliminating the sequential bottleneck and vanishing gradient problem over long sequences.

**Core Concepts:**
- **Self-Attention:** Computes relationships between all pairs of tokens in O(1) sequential operations
- **Parallelization:** All positions computed simultaneously (vs O(n) sequential steps in RNNs)
- **Long-range dependencies:** Direct path between distant tokens (no information bottleneck)
- **No recurrence:** Removes hidden state passing, enabling massive parallelism on GPUs/TPUs

**Why RNNs/LSTMs Were Replaced:**
| Problem with RNN/LSTM | Transformer Solution |
|----------------------|---------------------|
| Sequential processing (slow) | Fully parallel computation |
| Vanishing/exploding gradients | Direct attention paths |
| Long-range dependency struggle | Attention spans entire sequence |
| Hidden state bottleneck | All tokens attend to all tokens |

**Intuition:**
Think of RNN as reading a book word-by-word, remembering through a single "memory cell." Transformer reads the entire page at once, connecting any word to any other word directly.

**Practical Relevance:**
- Enabled training of billion-parameter models (GPT, BERT, T5)
- 10-100x faster training than equivalent RNN models
- Foundation of all modern LLMs, machine translation, text generation

**Interview Tips:**
- Mention the paper: "Attention Is All You Need" (2017)
- Key trade-off: O(n²) memory for sequence length n (solved by sparse attention variants)
- Transformers still use positional encodings since attention is permutation-invariant

---

### Question 2
- [ ] Done
**What is the self-attention mechanism and how does it compute attention weights?**

**Answer:**

**Definition:**
Self-attention is a mechanism where each token in a sequence computes a weighted representation of **all tokens in the same sequence** (including itself). It determines "how much focus" each token should place on every other token, creating context-aware representations that capture relationships regardless of distance.

**Core Concepts:**
- Each token generates three vectors: **Query (Q), Key (K), Value (V)**
- Attention score = how relevant token j is to token i
- Output = weighted sum of Value vectors using attention weights
- "Self" = queries and keys come from the same sequence

**Mathematical Formulation:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Step-by-step computation:**
1. Compute Q, K, V: $Q = XW_Q$, $K = XW_K$, $V = XW_V$
2. Compute attention scores: $S = QK^T$ (dot product similarity)
3. Scale: $S_{scaled} = \frac{S}{\sqrt{d_k}}$
4. Normalize: $\alpha = \text{softmax}(S_{scaled})$ → attention weights
5. Output: $Z = \alpha V$ (weighted combination of values)

**Intuition:**
- **Query:** "What am I looking for?"
- **Key:** "What do I contain?"
- **Value:** "What information do I provide?"
- Attention weight = compatibility between query and key

**Example:** In "The cat sat on the mat because **it** was tired"
- "it" attends strongly to "cat" (high attention weight)
- Learns this relationship from data, not rules

**Python Code Example:**
```python
import numpy as np

def self_attention(X, W_q, W_k, W_v):
    """
    X: input embeddings (seq_len, d_model)
    W_q, W_k, W_v: weight matrices (d_model, d_k)
    """
    # Step 1: Compute Q, K, V
    Q = X @ W_q  # (seq_len, d_k)
    K = X @ W_k
    V = X @ W_v
    
    # Step 2: Compute attention scores
    d_k = K.shape[-1]
    scores = Q @ K.T  # (seq_len, seq_len)
    
    # Step 3: Scale
    scores_scaled = scores / np.sqrt(d_k)
    
    # Step 4: Softmax (row-wise)
    attention_weights = np.exp(scores_scaled) / np.exp(scores_scaled).sum(axis=-1, keepdims=True)
    
    # Step 5: Weighted sum of values
    output = attention_weights @ V  # (seq_len, d_k)
    
    return output, attention_weights

# Example
seq_len, d_model, d_k = 4, 8, 4
X = np.random.randn(seq_len, d_model)
W_q = np.random.randn(d_model, d_k)
W_k = np.random.randn(d_model, d_k)
W_v = np.random.randn(d_model, d_k)

output, weights = self_attention(X, W_q, W_k, W_v)
print("Attention weights shape:", weights.shape)  # (4, 4) - each token attends to all tokens
```

**Interview Tips:**
- Attention weights sum to 1 for each query (due to softmax)
- Self-attention is **permutation equivariant** (needs positional encoding)
- Complexity: O(n²) for sequence length n

---

### Question 3
- [ ] Done
**Describe the multi-head attention mechanism and why multiple heads are beneficial.**

**Answer:**

**Definition:**
Multi-head attention runs **multiple self-attention operations in parallel**, each with different learned projections (W_Q, W_K, W_V). The outputs are concatenated and linearly transformed. This allows the model to jointly attend to information from **different representation subspaces** at different positions.

**Core Concepts:**
- Each "head" learns different attention patterns (syntax, semantics, coreference, etc.)
- Heads operate on smaller dimensions (d_k = d_model / h)
- Concatenation preserves all learned relationships
- Final linear projection combines information from all heads

**Mathematical Formulation:**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where each head:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Parameters:**
- h = number of heads (typically 8 or 12)
- d_k = d_v = d_model / h (e.g., 512/8 = 64)
- Total parameters ≈ same as single-head with full dimension

**Intuition:**
Think of multiple heads as **different experts** analyzing the same text:
- Head 1: Tracks subject-verb relationships
- Head 2: Handles coreference ("it" → "cat")
- Head 3: Captures positional patterns
- Head 4: Focuses on semantic similarity

**Why Multiple Heads Are Beneficial:**
| Benefit | Explanation |
|---------|-------------|
| Diverse attention patterns | Each head specializes in different relationships |
| Richer representations | Captures multiple aspects simultaneously |
| Ensemble effect | Reduces risk of missing important dependencies |
| Computational efficiency | Parallel computation, same total cost as single large head |

**Python Code Example:**
```python
import numpy as np

def multi_head_attention(X, W_qs, W_ks, W_vs, W_o, num_heads):
    """
    X: input (seq_len, d_model)
    W_qs, W_ks, W_vs: list of weight matrices per head
    W_o: output projection (h*d_v, d_model)
    """
    heads = []
    
    for i in range(num_heads):
        # Each head computes attention independently
        Q = X @ W_qs[i]
        K = X @ W_ks[i]
        V = X @ W_vs[i]
        
        d_k = K.shape[-1]
        scores = Q @ K.T / np.sqrt(d_k)
        attn_weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
        head_output = attn_weights @ V
        heads.append(head_output)
    
    # Concatenate all heads
    concat = np.concatenate(heads, axis=-1)  # (seq_len, h * d_v)
    
    # Final linear projection
    output = concat @ W_o  # (seq_len, d_model)
    
    return output

# Example: 4 heads, d_model=32, d_k=d_v=8
seq_len, d_model, num_heads = 5, 32, 4
d_k = d_model // num_heads

X = np.random.randn(seq_len, d_model)
W_qs = [np.random.randn(d_model, d_k) for _ in range(num_heads)]
W_ks = [np.random.randn(d_model, d_k) for _ in range(num_heads)]
W_vs = [np.random.randn(d_model, d_k) for _ in range(num_heads)]
W_o = np.random.randn(num_heads * d_k, d_model)

output = multi_head_attention(X, W_qs, W_ks, W_vs, W_o, num_heads)
print("Output shape:", output.shape)  # (5, 32)
```

**Interview Tips:**
- Standard: 8 heads in BERT-base, 12 in BERT-large, 96 in GPT-3
- Research shows different heads learn interpretable patterns (can be visualized)
- Pruning studies show not all heads are equally important

---

### Question 4
- [ ] Done
**How are Query, Key, and Value matrices computed in attention and what role does each play?**

**Answer:**

**Definition:**
Query (Q), Key (K), and Value (V) are three linear projections of the input embeddings, computed by multiplying input X with learned weight matrices. Q represents "what I'm looking for," K represents "what I contain," and V represents "the information to retrieve." Attention scores come from Q-K compatibility, which then weights the V vectors.

**Core Concepts:**
- **Input:** X ∈ ℝ^(n × d_model) — sequence of n token embeddings
- **Projections:** Three learnable weight matrices W_Q, W_K, W_V ∈ ℝ^(d_model × d_k)
- Same input X is projected differently for each role

**Mathematical Formulation:**

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

Then attention is computed as:
$$\text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Role of Each Component:**

| Component | Role | Analogy |
|-----------|------|---------|
| **Query (Q)** | What am I looking for? | Search query you type |
| **Key (K)** | What do I contain/represent? | Index/title of documents |
| **Value (V)** | What information do I provide? | Content of documents |
| **QK^T** | Compatibility/relevance score | Search relevance ranking |
| **softmax(QK^T)** | Attention weights | Probability of selecting each doc |
| **Attention × V** | Retrieved information | Weighted blend of document contents |

**Intuition — Database Analogy:**
Imagine a key-value store:
1. You have a **query** (your question)
2. Compare query against all **keys** to find matches
3. Retrieve **values** corresponding to best-matching keys
4. Return weighted combination based on match strength

**Why Separate Projections?**
- Allows model to learn **different representations** for matching vs. retrieving
- A token's "searchability" (K) can differ from its "content" (V)
- Provides flexibility — the same word can attend differently based on context

**Python Code Example:**
```python
import numpy as np

def compute_qkv(X, d_k):
    """
    X: input embeddings (seq_len, d_model)
    d_k: dimension of Q, K, V
    Returns: Q, K, V matrices
    """
    d_model = X.shape[-1]
    
    # Initialize weight matrices (in practice, these are learned)
    W_Q = np.random.randn(d_model, d_k) * 0.1
    W_K = np.random.randn(d_model, d_k) * 0.1
    W_V = np.random.randn(d_model, d_k) * 0.1
    
    # Project input to Q, K, V
    Q = X @ W_Q  # (seq_len, d_k) - "what am I looking for"
    K = X @ W_K  # (seq_len, d_k) - "what do I contain"
    V = X @ W_V  # (seq_len, d_k) - "what info do I have"
    
    return Q, K, V

# Example
X = np.array([
    [1.0, 0.5, 0.3, 0.8],  # Token 1: "The"
    [0.2, 0.9, 0.1, 0.4],  # Token 2: "cat"
    [0.5, 0.2, 0.8, 0.1],  # Token 3: "sat"
])
Q, K, V = compute_qkv(X, d_k=3)

# Attention scores: how much does each query match each key
scores = Q @ K.T / np.sqrt(3)
print("Q shape:", Q.shape)  # (3, 3)
print("Attention scores:\n", scores)  # (3, 3) - token i to token j
```

**Interview Tips:**
- Q, K, V are **learned** — the model decides what makes a good query/key/value
- In cross-attention: Q from decoder, K and V from encoder
- In self-attention: Q, K, V all come from same input

---

### Question 5
- [ ] Done
**Explain the scaled dot-product attention formula and why scaling by √d_k is necessary.**

**Answer:**

**Definition:**
Scaled dot-product attention computes attention weights by taking the dot product of queries and keys, scaling by √d_k, applying softmax, then multiplying by values. The scaling factor **prevents dot products from becoming too large** in high dimensions, which would push softmax into regions with extremely small gradients.

**Mathematical Formulation:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- Q ∈ ℝ^(n × d_k): Query matrix
- K ∈ ℝ^(n × d_k): Key matrix  
- V ∈ ℝ^(n × d_v): Value matrix
- d_k: Dimension of key vectors

**Why Scaling by √d_k is Necessary:**

**Problem without scaling:**
- Dot product of two d_k-dimensional vectors: $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$
- If q, k have mean 0 and variance 1, then $q \cdot k$ has variance = d_k
- For large d_k (e.g., 64), dot products become very large (magnitude ~ √d_k)
- Large values → softmax saturates → outputs become one-hot (nearly 0 or 1)
- Saturated softmax → **vanishing gradients** (softmax derivatives ≈ 0)

**Solution:**
- Divide by √d_k to normalize variance back to 1
- Keeps dot products in reasonable range regardless of dimension
- Maintains healthy gradient flow through softmax

**Intuition:**
| Without Scaling | With Scaling |
|-----------------|--------------|
| Scores: [50, 2, 1] | Scores: [6.25, 0.25, 0.125] |
| Softmax: [1.0, 0.0, 0.0] | Softmax: [0.99, 0.008, 0.002] |
| Gradient: ≈ 0 | Gradient: reasonable |

**Mathematical Proof:**
```
Assume q_i, k_i ~ N(0, 1) i.i.d.

E[q·k] = E[Σ q_i k_i] = Σ E[q_i]E[k_i] = 0

Var[q·k] = Var[Σ q_i k_i] = Σ Var[q_i k_i] = Σ 1 = d_k

After scaling: Var[q·k / √d_k] = d_k / d_k = 1 ✓
```

**Python Code Example:**
```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Q: (seq_len, d_k)
    K: (seq_len, d_k)
    V: (seq_len, d_v)
    """
    d_k = K.shape[-1]
    
    # Step 1: Dot product of Q and K^T
    scores = Q @ K.T  # (seq_len, seq_len)
    
    # Step 2: Scale by sqrt(d_k) to prevent large values
    scores_scaled = scores / np.sqrt(d_k)
    
    # Step 3: Softmax to get attention weights
    exp_scores = np.exp(scores_scaled - scores_scaled.max(axis=-1, keepdims=True))
    attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    
    # Step 4: Multiply by values
    output = attention_weights @ V
    
    return output, attention_weights

# Demonstrate the effect of scaling
d_k = 64  # Typical dimension
Q = np.random.randn(3, d_k)
K = np.random.randn(3, d_k)
V = np.random.randn(3, d_k)

# Without scaling - observe large values
raw_scores = Q @ K.T
print("Raw scores (no scaling):", raw_scores.max())  # Large values ~8-15

# With scaling - reasonable range
scaled_scores = raw_scores / np.sqrt(d_k)
print("Scaled scores:", scaled_scores.max())  # ~1-2
```

**Interview Tips:**
- This is why it's called "scaled" dot-product attention
- Alternative: additive attention doesn't need scaling (uses tanh activation)
- Some implementations use √d_k for both Q and K separately (equivalent)

---

### Question 6
- [ ] Done
**What is positional encoding and why is it necessary in Transformers?**

**Answer:**

**Definition:**
Positional encoding injects information about token positions into the input embeddings since self-attention is **permutation-invariant** (treats input as a set, not sequence). Without positional encoding, "The cat ate the fish" and "The fish ate the cat" would produce identical representations.

**Why Necessary:**
- Self-attention computes pairwise relationships without considering order
- Attention(Q, K, V) gives same output regardless of token ordering
- Language heavily depends on word order for meaning
- Need to explicitly encode position information

**Mathematical Formulation (Sinusoidal Encoding):**

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where:
- pos = position in sequence (0, 1, 2, ...)
- i = dimension index (0 to d_model/2)
- d_model = embedding dimension

**Final Input:**
$$\text{Input} = \text{TokenEmbedding} + \text{PositionalEncoding}$$

**Types of Positional Encoding:**

| Type | Description | Used In |
|------|-------------|---------|
| **Sinusoidal (fixed)** | Deterministic sin/cos functions | Original Transformer |
| **Learned absolute** | Trainable embedding per position | BERT, GPT-2 |
| **Relative (RPE)** | Encodes distance between tokens | Transformer-XL, T5 |
| **RoPE** | Rotary position embeddings | LLaMA, GPT-NeoX |
| **ALiBi** | Adds bias to attention scores | BLOOM |

**Intuition — Why Sin/Cos:**
- Different frequencies capture different position scales
- Low frequencies: distinguish distant positions
- High frequencies: distinguish nearby positions
- Can potentially extrapolate to longer sequences (relative distances preserved)

**Python Code Example:**
```python
import numpy as np
import matplotlib.pyplot as plt

def sinusoidal_positional_encoding(seq_len, d_model):
    """
    Generate sinusoidal positional encodings
    """
    PE = np.zeros((seq_len, d_model))
    
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            denominator = 10000 ** (i / d_model)
            PE[pos, i] = np.sin(pos / denominator)
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / denominator)
    
    return PE

# Example
seq_len, d_model = 50, 64
PE = sinusoidal_positional_encoding(seq_len, d_model)

print("PE shape:", PE.shape)  # (50, 64)
print("Position 0 encoding (first 8 dims):", PE[0, :8])
print("Position 1 encoding (first 8 dims):", PE[1, :8])

# Adding to token embeddings
token_embeddings = np.random.randn(seq_len, d_model)
input_with_position = token_embeddings + PE
```

**Interview Tips:**
- BERT uses **learned** positional embeddings (up to 512 positions)
- Sinusoidal allows generalization beyond training length (theoretically)
- RoPE (Rotary) is now standard in modern LLMs — encodes relative positions via rotation
- Position info is added once at input, then propagates through layers

---

### Question 7
- [ ] Done
**Describe the encoder-decoder structure and how cross-attention connects them.**

**Answer:**

**Definition:**
The encoder-decoder architecture consists of two stacks: an **encoder** that processes the input sequence into contextualized representations, and a **decoder** that generates output tokens autoregressively. **Cross-attention** connects them by allowing decoder tokens to attend to all encoder outputs, enabling the decoder to "look at" the source sequence while generating.

**Core Components:**

| Component | Role | Attention Type |
|-----------|------|----------------|
| **Encoder** | Encodes source sequence | Self-attention (bidirectional) |
| **Decoder** | Generates target sequence | Masked self-attention + Cross-attention |
| **Cross-attention** | Connects encoder to decoder | Q from decoder, K/V from encoder |

**Architecture Flow:**
```
Input: "Hello world" → Encoder → [h1, h2] (encoder hidden states)
                                    ↓
                            Cross-Attention
                                    ↓
Decoder: "<start>" → Self-Attn → Cross-Attn → "Bonjour"
         "Bonjour" → Self-Attn → Cross-Attn → "monde"
```

**Cross-Attention Mechanism:**

$$\text{CrossAttention} = \text{softmax}\left(\frac{Q_{decoder}K_{encoder}^T}{\sqrt{d_k}}\right)V_{encoder}$$

- **Query (Q):** From decoder's current hidden state
- **Key (K):** From encoder output
- **Value (V):** From encoder output

**Intuition:**
- Decoder asks: "Which parts of the input should I focus on for generating this word?"
- Cross-attention provides the answer by weighting encoder representations
- Translation: When generating "Bonjour," attend strongly to "Hello"

**Encoder-Decoder Models:**
- **T5:** Text-to-text transfer (all tasks as seq2seq)
- **BART:** Denoising autoencoder
- **mBART:** Multilingual BART
- **Original Transformer:** Machine translation

**Python Code Example:**
```python
import numpy as np

def cross_attention(decoder_hidden, encoder_output, W_q, W_k, W_v):
    """
    decoder_hidden: current decoder state (1, d_model) or (dec_len, d_model)
    encoder_output: encoder hidden states (enc_len, d_model)
    """
    # Q from decoder, K and V from encoder
    Q = decoder_hidden @ W_q  # (dec_len, d_k)
    K = encoder_output @ W_k  # (enc_len, d_k)
    V = encoder_output @ W_v  # (enc_len, d_v)
    
    d_k = K.shape[-1]
    
    # Attention scores: decoder attends to encoder
    scores = Q @ K.T / np.sqrt(d_k)  # (dec_len, enc_len)
    attention_weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    
    # Weighted sum of encoder values
    context = attention_weights @ V  # (dec_len, d_v)
    
    return context, attention_weights

# Example: Translation
# Encoder processed "Hello world" → 2 hidden states
encoder_output = np.random.randn(2, 64)  # (enc_len=2, d_model=64)

# Decoder is generating token 3
decoder_hidden = np.random.randn(1, 64)  # Current decoder state

# Weight matrices
d_k = 32
W_q = np.random.randn(64, d_k)
W_k = np.random.randn(64, d_k)
W_v = np.random.randn(64, d_k)

context, weights = cross_attention(decoder_hidden, encoder_output, W_q, W_k, W_v)
print("Context shape:", context.shape)  # (1, 32)
print("Attention to encoder tokens:", weights)  # Shows which source words decoder focuses on
```

**Interview Tips:**
- Cross-attention is bidirectional (decoder can see all encoder positions)
- Decoder self-attention is masked (causal/unidirectional)
- GPT (decoder-only) doesn't have cross-attention
- Cross-attention enables conditioning on arbitrary input lengths

---

### Question 8
- [ ] Done
**How do residual connections and layer normalization work in Transformer blocks?**

**Answer:**

**Definition:**
**Residual connections** (skip connections) add the input of a sublayer to its output, preventing vanishing gradients and enabling training of deep networks. **Layer normalization** normalizes activations across the feature dimension (not batch), stabilizing training. Together they form: `LayerNorm(x + Sublayer(x))`.

**Core Concepts:**

**Residual Connection:**
$$\text{Output} = x + \text{Sublayer}(x)$$
- Creates a "shortcut" for gradients to flow
- If sublayer learns nothing useful, output ≈ input (identity)
- Enables training of 100+ layer networks

**Layer Normalization:**
$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

Where (computed per token, across features):
- μ = mean of x across feature dimension
- σ = standard deviation across feature dimension
- γ, β = learned scale and shift parameters
- ε = small constant for numerical stability

**Transformer Block Structure:**

```
Input x
    ↓
┌───────────────────────────────────┐
│  Multi-Head Attention             │
└───────────────────────────────────┘
    ↓
    + x  ← Residual connection
    ↓
  LayerNorm
    ↓
┌───────────────────────────────────┐
│  Feed-Forward Network             │
└───────────────────────────────────┘
    ↓
    + (previous output)  ← Residual connection
    ↓
  LayerNorm
    ↓
Output
```

**Pre-LN vs Post-LN:**
| Variant | Formula | Used In |
|---------|---------|---------|
| **Post-LN** | LayerNorm(x + Sublayer(x)) | Original Transformer, BERT |
| **Pre-LN** | x + Sublayer(LayerNorm(x)) | GPT-2, GPT-3, most modern LLMs |

Pre-LN is more stable for training very deep models.

**Why Layer Norm (not Batch Norm):**
- Batch Norm depends on batch statistics → problematic for variable sequence lengths
- Layer Norm normalizes each sample independently
- Works with batch size 1 (important for inference)
- No dependency on other samples in batch

**Python Code Example:**
```python
import numpy as np

def layer_norm(x, gamma, beta, eps=1e-6):
    """
    x: input (seq_len, d_model)
    gamma, beta: learnable parameters (d_model,)
    """
    # Compute mean and variance across features (last dim)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    
    # Normalize
    x_norm = (x - mean) / np.sqrt(var + eps)
    
    # Scale and shift
    return gamma * x_norm + beta

def transformer_sublayer(x, sublayer_fn, gamma, beta):
    """
    Applies sublayer with residual connection and layer norm (Post-LN)
    """
    # Apply sublayer (attention or FFN)
    sublayer_output = sublayer_fn(x)
    
    # Residual connection
    residual_output = x + sublayer_output
    
    # Layer normalization
    normalized = layer_norm(residual_output, gamma, beta)
    
    return normalized

# Example
d_model = 64
x = np.random.randn(5, d_model)  # 5 tokens
gamma = np.ones(d_model)
beta = np.zeros(d_model)

# Simple sublayer (identity for demo)
identity = lambda x: x * 0.1  # Small transformation

output = transformer_sublayer(x, identity, gamma, beta)
print("Input mean:", x.mean())
print("Output mean:", output.mean())  # Normalized
print("Output std:", output.std())    # Close to 1
```

**Interview Tips:**
- Residual connections: "Identity shortcut enables gradient flow"
- Layer Norm: "Normalizes across features, independent of batch"
- Pre-LN is preferred for LLMs (more stable at scale)
- Without residuals, gradients vanish in deep Transformers (6+ layers)

---

### Question 9
- [ ] Done
**What is the purpose of the feed-forward network (FFN) in each Transformer layer?**

**Answer:**

**Definition:**
The Feed-Forward Network (FFN) is a **position-wise** fully connected network applied independently to each token after attention. It consists of two linear transformations with a non-linear activation in between. FFN adds **non-linearity** and increases the model's capacity to learn complex transformations that attention alone cannot capture.

**Mathematical Formulation:**

$$\text{FFN}(x) = \text{Linear}_2(\text{Activation}(\text{Linear}_1(x)))$$

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$$

Or with GELU (modern LLMs):
$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$$

**Dimensions:**
- Input: d_model (e.g., 768)
- Hidden: d_ff = 4 × d_model (e.g., 3072)
- Output: d_model (e.g., 768)

**Why FFN is Necessary:**

| Aspect | Attention Alone | With FFN |
|--------|-----------------|----------|
| Non-linearity | Softmax only (limited) | ReLU/GELU adds expressiveness |
| Token-level processing | Mixes information across tokens | Processes each token deeply |
| Capacity | Limited transformation | Larger hidden dim = more parameters |
| Role | "What to attend to" | "What to do with attended info" |

**Core Concepts:**
- **Position-wise:** Same FFN applied to each position independently (no cross-position interaction)
- **Expansion-contraction:** Expands to 4× dimension, then compresses back
- **Non-linearity:** ReLU/GELU allows learning complex functions
- **Most parameters:** FFN contains ~2/3 of layer parameters

**Intuition:**
- Attention = "gather relevant information from other tokens"
- FFN = "process/transform the gathered information"
- Think of FFN as **memory storage** — research shows FFN stores factual knowledge

**Modern Variants:**
| Variant | Formula | Used In |
|---------|---------|---------|
| Standard | W₂·ReLU(W₁x) | Original Transformer |
| GLU | W₂·(W₁x ⊙ σ(W₃x)) | LLaMA, PaLM |
| SwiGLU | W₂·(Swish(W₁x) ⊙ W₃x) | LLaMA 2, Mistral |

**Python Code Example:**
```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def feed_forward_network(x, W1, b1, W2, b2, activation='relu'):
    """
    x: input (seq_len, d_model)
    W1: (d_model, d_ff) - expansion
    W2: (d_ff, d_model) - contraction
    """
    # First linear: expand to d_ff
    hidden = x @ W1 + b1  # (seq_len, d_ff)
    
    # Activation
    if activation == 'relu':
        hidden = relu(hidden)
    else:
        hidden = gelu(hidden)
    
    # Second linear: contract back to d_model
    output = hidden @ W2 + b2  # (seq_len, d_model)
    
    return output

# Example
d_model, d_ff = 64, 256  # 4x expansion
seq_len = 5

x = np.random.randn(seq_len, d_model)
W1 = np.random.randn(d_model, d_ff) * 0.02
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model) * 0.02
b2 = np.zeros(d_model)

output = feed_forward_network(x, W1, b1, W2, b2)
print("Input shape:", x.shape)   # (5, 64)
print("Output shape:", output.shape)  # (5, 64)
print("FFN parameters:", d_model*d_ff + d_ff + d_ff*d_model + d_model)
```

**Interview Tips:**
- FFN is applied **independently** to each token (no mixing)
- ~2/3 of Transformer parameters are in FFN layers
- Research shows FFN acts as key-value memory storing facts
- Modern LLMs use SwiGLU activation (better than ReLU)

---

### Question 10
- [ ] Done
**How does masked attention work in decoder layers for autoregressive generation?**

**Answer:**

**Definition:**
Masked attention (causal attention) prevents decoder tokens from attending to **future positions** during training and inference. A mask sets attention scores to -∞ for future tokens before softmax, ensuring each position can only depend on previous positions. This maintains the autoregressive property where token t is predicted using only tokens 0 to t-1.

**Why Masking is Necessary:**
- Training uses teacher forcing (all tokens fed simultaneously)
- Without masking, model could "cheat" by looking at future tokens
- Generation is sequential: can only use past context
- Masking during training mimics inference behavior

**Mathematical Formulation:**

$$\text{MaskedAttention} = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V$$

Where mask M:
$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \text{ (can attend)} \\ -\infty & \text{if } i < j \text{ (cannot attend)} \end{cases}$$

After softmax: $e^{-\infty} = 0$ → future positions get zero attention weight.

**Causal Mask Matrix:**
```
For sequence length 4:
     pos 0  pos 1  pos 2  pos 3
pos 0 [  0    -∞     -∞     -∞  ]   Token 0 sees only itself
pos 1 [  0     0     -∞     -∞  ]   Token 1 sees 0, 1
pos 2 [  0     0      0     -∞  ]   Token 2 sees 0, 1, 2
pos 3 [  0     0      0      0  ]   Token 3 sees all
```

**Resulting Attention Weights:**
```
After softmax (example):
     pos 0  pos 1  pos 2  pos 3
pos 0 [1.0    0      0      0  ]
pos 1 [0.3   0.7     0      0  ]
pos 2 [0.2   0.3    0.5     0  ]
pos 3 [0.1   0.2    0.3    0.4 ]
```

**Python Code Example:**
```python
import numpy as np

def create_causal_mask(seq_len):
    """
    Creates lower-triangular mask for causal attention
    """
    # Upper triangle = -inf (future positions)
    mask = np.triu(np.ones((seq_len, seq_len)) * (-np.inf), k=1)
    return mask

def masked_self_attention(Q, K, V):
    """
    Causal self-attention for decoder
    """
    seq_len = Q.shape[0]
    d_k = K.shape[-1]
    
    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Apply causal mask (prevent attending to future)
    mask = create_causal_mask(seq_len)
    scores_masked = scores + mask
    
    # Softmax (masked positions become 0)
    exp_scores = np.exp(scores_masked - scores_masked.max(axis=-1, keepdims=True))
    attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    
    # Weighted sum
    output = attention_weights @ V
    
    return output, attention_weights

# Example
seq_len, d_k = 4, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

output, weights = masked_self_attention(Q, K, V)

print("Attention weights (rows sum to 1):")
print(np.round(weights, 2))
# Notice: upper triangle is 0 (no future attention)
```

**Comparison:**

| Aspect | Encoder (BERT) | Decoder (GPT) |
|--------|----------------|---------------|
| Attention | Bidirectional | Causal (masked) |
| Token i sees | All tokens | Tokens 0 to i |
| Mask | None | Upper triangular -∞ |
| Use case | Understanding | Generation |

**Interview Tips:**
- GPT models use ONLY causal masked attention
- BERT uses bidirectional (no mask) — not for generation
- Mask is added before softmax, not after
- KV-cache optimization possible because past tokens don't change

---

## Attention Mechanism Deep Dive

### Question 11
- [ ] Done
**What is the difference between additive (Bahdanau) and multiplicative (Luong) attention?**

**Answer:**

**Definition:**
Additive (Bahdanau) attention computes alignment scores using a feed-forward network with learned weights, while multiplicative (Luong) attention uses a simple dot product or bilinear form. Multiplicative is computationally faster; additive is more expressive but slower. Modern Transformers use scaled multiplicative (dot-product) attention.

**Additive Attention (Bahdanau, 2015):**

$$e_{ij} = v^T \tanh(W_q q_i + W_k k_j)$$

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$$

- Uses a learned feed-forward layer
- Concatenates query and key, applies tanh
- More parameters, more expressive

**Multiplicative Attention (Luong, 2015):**

**Dot-product:**
$$e_{ij} = q_i^T k_j$$

**General (bilinear):**
$$e_{ij} = q_i^T W k_j$$

**Scaled dot-product (Transformer):**
$$e_{ij} = \frac{q_i^T k_j}{\sqrt{d_k}}$$

**Comparison:**

| Aspect | Additive (Bahdanau) | Multiplicative (Luong) |
|--------|---------------------|------------------------|
| **Formula** | v^T tanh(W_q q + W_k k) | q^T k or q^T W k |
| **Parameters** | W_q, W_k, v | None (dot) or W (general) |
| **Computation** | Slower (MLP) | Faster (matrix multiply) |
| **Expressiveness** | Higher | Lower (but often sufficient) |
| **GPU efficiency** | Lower | Higher (optimized matmul) |
| **Used in** | Original seq2seq | Transformers |

**When Each Works Better:**
- **Additive:** When d_k is small, additive can be more effective
- **Multiplicative:** When d_k is large, scales better with hardware
- **Scaled dot-product:** Best of both — fast and stable at high dimensions

**Intuition:**
- **Additive:** "Learn a neural network to compute similarity"
- **Multiplicative:** "Use geometric similarity (dot product) directly"

**Python Code Example:**
```python
import numpy as np

def additive_attention(query, keys, W_q, W_k, v):
    """
    Bahdanau-style additive attention
    query: (d_q,) or (1, d_q)
    keys: (seq_len, d_k)
    """
    # Project query and keys
    query_proj = query @ W_q  # (hidden,)
    keys_proj = keys @ W_k    # (seq_len, hidden)
    
    # Add and apply tanh
    combined = np.tanh(query_proj + keys_proj)  # Broadcasting
    
    # Compute scores with v
    scores = combined @ v  # (seq_len,)
    
    # Softmax
    weights = np.exp(scores) / np.exp(scores).sum()
    return weights

def multiplicative_attention(query, keys, mode='dot'):
    """
    Luong-style multiplicative attention
    """
    if mode == 'dot':
        scores = keys @ query  # (seq_len,)
    elif mode == 'scaled':
        d_k = query.shape[-1]
        scores = (keys @ query) / np.sqrt(d_k)
    
    weights = np.exp(scores) / np.exp(scores).sum()
    return weights

# Example
d_model, hidden = 32, 16
query = np.random.randn(d_model)
keys = np.random.randn(5, d_model)

# Additive
W_q = np.random.randn(d_model, hidden)
W_k = np.random.randn(d_model, hidden)
v = np.random.randn(hidden)
add_weights = additive_attention(query, keys, W_q, W_k, v)

# Multiplicative
mult_weights = multiplicative_attention(query, keys, mode='scaled')

print("Additive weights:", np.round(add_weights, 3))
print("Multiplicative weights:", np.round(mult_weights, 3))
```

**Interview Tips:**
- Transformers use **scaled dot-product** (multiplicative variant)
- Additive was popular in RNN-based seq2seq (pre-Transformer era)
- Scaling by √d_k makes dot-product work well at high dimensions
- Multiplicative is preferred for its computational efficiency

---

### Question 12
- [ ] Done
**Explain global vs local attention mechanisms and their trade-offs.**

**Answer:**

**Definition:**
**Global attention** allows each token to attend to all other tokens in the sequence (full context). **Local attention** restricts each token to attend only to a fixed-size window around it. Global captures long-range dependencies but is O(n²); local is O(n) but may miss distant relationships.

**Global Attention:**
- Every token attends to every other token
- Full context awareness
- Complexity: O(n²) in time and memory
- Used in: Standard Transformer, BERT, GPT

**Local Attention:**
- Each token attends to a fixed window (e.g., ±k positions)
- Limited context but efficient
- Complexity: O(n × w) where w = window size
- Used in: Longformer (local), Image Transformer

**Comparison:**

| Aspect | Global Attention | Local Attention |
|--------|------------------|-----------------|
| **Context** | Full sequence | Window only |
| **Complexity** | O(n²) | O(n × w) |
| **Memory** | Grows quadratically | Grows linearly |
| **Long-range** | Excellent | Poor (unless combined) |
| **Max length** | ~2K-8K tokens | 16K-100K+ tokens |
| **Use case** | Short sequences | Long documents |

**Visualization:**
```
Global Attention (each token → all tokens):
Token:   A   B   C   D   E
A:      [✓] [✓] [✓] [✓] [✓]
B:      [✓] [✓] [✓] [✓] [✓]
...

Local Attention (window=2, each token → ±1):
Token:   A   B   C   D   E
A:      [✓] [✓] [ ] [ ] [ ]
B:      [✓] [✓] [✓] [ ] [ ]
C:      [ ] [✓] [✓] [✓] [ ]
...
```

**Hybrid Approaches:**

| Model | Strategy |
|-------|----------|
| **Longformer** | Local + global attention on special tokens ([CLS]) |
| **BigBird** | Local + random + global attention |
| **Sparse Transformer** | Strided attention patterns |

**Trade-offs:**

| Trade-off | Global | Local |
|-----------|--------|-------|
| Accuracy on long deps | ✓ Better | ✗ Worse |
| Efficiency | ✗ Slower | ✓ Faster |
| Scalability | ✗ Limited | ✓ Scales well |
| Implementation | Simpler | More complex |

**Intuition:**
- **Global:** Reading entire book at once, remembering everything
- **Local:** Reading with a sliding window, focusing on nearby text
- **Hybrid:** Skim entire book (global on summaries) + detailed reading (local)

**Python Code Example:**
```python
import numpy as np

def global_attention(Q, K, V):
    """Standard full attention"""
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    return weights @ V, weights

def local_attention(Q, K, V, window_size=2):
    """
    Local attention with fixed window
    Each position attends to ±window_size positions
    """
    seq_len, d_k = Q.shape
    
    # Create local attention mask
    mask = np.full((seq_len, seq_len), -np.inf)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = 0
    
    scores = Q @ K.T / np.sqrt(d_k)
    scores_masked = scores + mask
    
    exp_scores = np.exp(scores_masked - scores_masked.max(axis=-1, keepdims=True))
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    
    return weights @ V, weights

# Example
seq_len, d_k = 6, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

_, global_w = global_attention(Q, K, V)
_, local_w = local_attention(Q, K, V, window_size=1)

print("Global attention pattern (all attend to all):")
print((global_w > 0.01).astype(int))  # Dense

print("\nLocal attention pattern (window=1):")
print((local_w > 0.01).astype(int))  # Sparse, banded
```

**Interview Tips:**
- Standard BERT/GPT use global attention (limits context length)
- For documents >4K tokens, use local + global hybrids
- Memory is often the bottleneck, not compute
- Modern trend: sliding window + few global tokens (Longformer-style)

---

### Question 13
- [ ] Done
**What is the computational complexity O(n²) of self-attention and how does it limit context length?**

**Answer:**

**Definition:**
Self-attention has **O(n²)** time and memory complexity because every token computes attention scores with every other token, creating an n×n attention matrix. For sequence length n=4096, this means 16 million attention scores per layer. This quadratic scaling makes long sequences prohibitively expensive in both compute and memory.

**Where O(n²) Comes From:**

$$\text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| Operation | Shape | Complexity |
|-----------|-------|------------|
| Q = X·W_Q | (n, d) × (d, d_k) | O(n·d·d_k) |
| K = X·W_K | (n, d) × (d, d_k) | O(n·d·d_k) |
| **QK^T** | **(n, d_k) × (d_k, n)** | **O(n²·d_k)** ← Bottleneck |
| softmax | (n, n) | O(n²) |
| **Attention × V** | **(n, n) × (n, d_v)** | **O(n²·d_v)** |

**Total: O(n²·d)** where d is hidden dimension.

**Memory Complexity:**
- Must store full n×n attention matrix
- For n=4096, d=768: ~64MB per attention layer
- 12 layers × 12 heads = ~9GB just for attention matrices

**Practical Limits:**

| Sequence Length | Attention Matrix Size | Feasibility |
|-----------------|----------------------|-------------|
| 512 (BERT) | 262K entries | Easy |
| 2048 (GPT-2) | 4M entries | Manageable |
| 4096 | 16M entries | Challenging |
| 32K | 1B entries | Requires optimizations |
| 100K+ | 10B+ entries | Impossible without sparse attention |

**Why This Limits Context:**
1. **Memory:** GPU memory exhausted storing attention matrices
2. **Compute:** Training/inference time grows quadratically
3. **Batch size:** Longer sequences → smaller batches → less parallelism

**Solutions to O(n²) Problem:**

| Approach | Complexity | Example |
|----------|------------|---------|
| Sparse attention | O(n·√n) or O(n·log n) | Longformer, BigBird |
| Linear attention | O(n) | Performer, Linear Transformer |
| Flash Attention | O(n²) but memory-efficient | GPU kernel optimization |
| Chunked/sliding | O(n·w) | Sliding window |
| State-space models | O(n) | Mamba, RWKV |

**Python Code Example:**
```python
import numpy as np
import time

def measure_attention_complexity(seq_lengths):
    """Demonstrate quadratic scaling"""
    results = []
    
    for n in seq_lengths:
        d_k = 64
        Q = np.random.randn(n, d_k)
        K = np.random.randn(n, d_k)
        
        start = time.time()
        scores = Q @ K.T  # O(n² × d_k)
        elapsed = time.time() - start
        
        memory_bytes = n * n * 8  # float64
        results.append({
            'seq_len': n,
            'time_ms': elapsed * 1000,
            'memory_mb': memory_bytes / 1e6,
            'matrix_size': n * n
        })
    
    return results

# Demonstrate scaling
lengths = [128, 256, 512, 1024, 2048]
results = measure_attention_complexity(lengths)

for r in results:
    print(f"n={r['seq_len']:4d}: matrix_size={r['matrix_size']:>8,}, "
          f"memory={r['memory_mb']:.1f}MB")

# Output shows quadratic growth:
# n=128:  matrix_size=16,384, memory=0.1MB
# n=256:  matrix_size=65,536, memory=0.5MB (4x)
# n=512:  matrix_size=262,144, memory=2.1MB (4x)
# ...
```

**Interview Tips:**
- "Attention is quadratic" = the fundamental scaling limitation of Transformers
- Flash Attention doesn't reduce O(n²) compute, but reduces memory via tiling
- For >8K tokens, sparse attention or linear attention is necessary
- Modern LLMs use sliding window (Mistral) or hybrid approaches

---

### Question 14
- [ ] Done
**Describe sparse attention patterns (Longformer, BigBird) and their benefits for long sequences.**

**Answer:**

**Definition:**
Sparse attention reduces the O(n²) complexity by having each token attend to only a **subset** of other tokens instead of all tokens. Patterns include local windows, global tokens, and random connections. This enables processing sequences of 4K-100K+ tokens with linear or near-linear complexity while maintaining reasonable quality.

**Types of Sparse Attention Patterns:**

| Pattern | Description | Tokens Attended |
|---------|-------------|-----------------|
| **Local/Sliding Window** | Attend to ±k neighbors | O(window_size) |
| **Global** | Special tokens attend to all | O(n) for few tokens |
| **Random** | Random sparse connections | O(random_count) |
| **Strided** | Every k-th token | O(n/k) |
| **Dilated** | Exponentially spaced | O(log n) |

**Longformer (2020):**

```
Pattern: Local + Global

Global tokens: [CLS], task-specific tokens
Local window: ±w positions (e.g., w=256)

        [CLS]  tok1   tok2   tok3   tok4   ...   tokN
[CLS]   [G]    [G]    [G]    [G]    [G]    ...   [G]    ← Global attends to all
tok1    [G]    [L]    [L]    [L]    [ ]    ...   [ ]    ← Local window
tok2    [G]    [L]    [L]    [L]    [L]    ...   [ ]
...

Complexity: O(n × w) where w = window size
```

**BigBird (2020):**

```
Pattern: Local + Global + Random

- Local: sliding window of size w
- Global: g special tokens attend to all
- Random: r random connections per token

Complexity: O(n × (w + g + r)) = O(n)
```

**Comparison:**

| Model | Patterns | Max Length | Complexity |
|-------|----------|------------|------------|
| BERT | Full | 512 | O(n²) |
| Longformer | Local + Global | 4,096 | O(n) |
| BigBird | Local + Global + Random | 4,096 | O(n) |
| Sparse Transformer | Strided + Local | 16K+ | O(n√n) |

**Benefits for Long Sequences:**

| Benefit | Explanation |
|---------|-------------|
| Memory efficiency | Don't store full n×n matrix |
| Longer context | Process 16K-100K tokens |
| Faster training | Linear scaling with length |
| Maintains quality | Local patterns capture most info |

**When Global Tokens Matter:**
- [CLS] for classification (needs whole-document view)
- Question tokens in QA (need to find answer anywhere)
- Special separator tokens

**Python Code Example:**
```python
import numpy as np

def longformer_attention_mask(seq_len, window_size, global_indices):
    """
    Create Longformer-style sparse attention mask
    """
    # Initialize with -inf (no attention)
    mask = np.full((seq_len, seq_len), -np.inf)
    
    # Local attention: sliding window
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = 0
    
    # Global attention: global tokens attend to all and are attended by all
    for g in global_indices:
        mask[g, :] = 0  # Global token attends to all
        mask[:, g] = 0  # All tokens attend to global
    
    return mask

def sparse_attention(Q, K, V, mask):
    """Apply attention with sparse mask"""
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    scores_masked = scores + mask
    
    exp_scores = np.exp(scores_masked - scores_masked.max(axis=-1, keepdims=True))
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    
    return weights @ V, weights

# Example: 8 tokens, window=2, token 0 is global
seq_len = 8
mask = longformer_attention_mask(seq_len, window_size=2, global_indices=[0])

print("Sparse attention pattern (1=can attend):")
print((mask == 0).astype(int))

# Count connections
full_connections = seq_len * seq_len  # 64
sparse_connections = (mask == 0).sum()  # Much fewer
print(f"\nFull attention: {full_connections} connections")
print(f"Sparse attention: {sparse_connections} connections")
print(f"Reduction: {100*(1-sparse_connections/full_connections):.1f}%")
```

**Interview Tips:**
- Sparse attention trades accuracy for efficiency
- Key insight: most attention is local anyway (~80% within ±128 tokens)
- Global tokens are critical for tasks needing full context (classification)
- Modern approach: sliding window (Mistral) + KV cache for long context

---

### Question 15
- [ ] Done
**Explain Flash Attention and why it's critical for memory-efficient training and inference.**

**Answer:**

**Definition:**
Flash Attention is an **IO-aware** exact attention algorithm that computes attention by tiling and recomputation, avoiding materializing the full n×n attention matrix in GPU high-bandwidth memory (HBM). It achieves 2-4x speedup and significant memory reduction while computing **mathematically identical** results to standard attention.

**The Memory Bottleneck Problem:**
- Standard attention: Q×K^T creates n×n matrix (stored in slow HBM)
- GPU has fast SRAM (limited) vs slow HBM (large)
- Moving data between SRAM and HBM is the bottleneck
- Standard attention is **memory-bound**, not compute-bound

**How Flash Attention Works:**

1. **Tiling:** Split Q, K, V into blocks that fit in SRAM
2. **Block-wise computation:** Compute attention for each block in SRAM
3. **Online softmax:** Use numerically stable incremental softmax
4. **Recomputation:** Recompute attention during backward pass (saves memory)
5. **Fused kernel:** Single GPU kernel, no intermediate HBM writes

```
Standard Attention:
Q, K, V (HBM) → Compute QK^T → Store in HBM → Softmax → Store → ×V → HBM
                  ↑ Slow memory transfers

Flash Attention:
Load Q_block, K_block, V_block into SRAM
→ Compute everything in SRAM
→ Write only final output to HBM
```

**Key Innovations:**

| Technique | Benefit |
|-----------|---------|
| **Tiling** | Fits in fast SRAM |
| **Kernel fusion** | Single kernel, no intermediate writes |
| **Recomputation** | Don't store attention matrix for backward |
| **Online softmax** | Compute softmax incrementally across blocks |

**Memory Comparison:**

| Method | Memory for Attention | Sequence Length Support |
|--------|---------------------|------------------------|
| Standard | O(n²) | ~2K-4K tokens |
| Flash Attention | O(n) | 16K-64K+ tokens |

**Performance Gains:**
- 2-4x faster training than PyTorch standard attention
- 5-20x less memory for attention computation
- Enables training with longer sequences without OOM
- Used by: LLaMA 2, GPT-4, Claude, Mistral

**Python Usage Example:**
```python
# Flash Attention is a CUDA kernel - conceptual demo of the idea

import numpy as np

def online_softmax_update(m_prev, l_prev, m_new, l_new, o_prev, o_new):
    """
    Numerically stable online softmax update
    Allows computing softmax incrementally across blocks
    """
    m = np.maximum(m_prev, m_new)
    l = np.exp(m_prev - m) * l_prev + np.exp(m_new - m) * l_new
    o = (np.exp(m_prev - m) * l_prev * o_prev + 
         np.exp(m_new - m) * l_new * o_new) / l
    return m, l, o

def flash_attention_conceptual(Q, K, V, block_size=2):
    """
    Conceptual Flash Attention (simplified)
    Real implementation is in CUDA
    """
    seq_len, d = Q.shape
    num_blocks = (seq_len + block_size - 1) // block_size
    
    output = np.zeros_like(Q)
    
    for i in range(num_blocks):
        q_start = i * block_size
        q_end = min(q_start + block_size, seq_len)
        Q_block = Q[q_start:q_end]
        
        # Initialize running statistics
        m_i = np.full(q_end - q_start, -np.inf)
        l_i = np.zeros(q_end - q_start)
        o_i = np.zeros((q_end - q_start, d))
        
        for j in range(num_blocks):
            k_start = j * block_size
            k_end = min(k_start + block_size, seq_len)
            K_block = K[k_start:k_end]
            V_block = V[k_start:k_end]
            
            # Compute block attention scores
            scores = Q_block @ K_block.T / np.sqrt(d)
            
            # Online softmax update
            m_block = scores.max(axis=-1)
            p_block = np.exp(scores - m_block[:, None])
            l_block = p_block.sum(axis=-1)
            o_block = p_block @ V_block
            
            # Update running statistics
            m_new = np.maximum(m_i, m_block)
            l_i = np.exp(m_i - m_new) * l_i + np.exp(m_block - m_new) * l_block
            o_i = (np.exp(m_i - m_new)[:, None] * o_i + 
                   np.exp(m_block - m_new)[:, None] * o_block)
            m_i = m_new
        
        output[q_start:q_end] = o_i / l_i[:, None]
    
    return output

# Usage
seq_len, d = 8, 4
Q = np.random.randn(seq_len, d)
K = np.random.randn(seq_len, d)
V = np.random.randn(seq_len, d)

output = flash_attention_conceptual(Q, K, V, block_size=2)
print("Output shape:", output.shape)
```

**Interview Tips:**
- Flash Attention computes **exact** attention (not approximate)
- Key insight: Memory IO, not compute, is the bottleneck
- Flash Attention 2 added parallelism across sequence length
- Essential for training modern LLMs with long context
- Available in: `torch.nn.functional.scaled_dot_product_attention` (PyTorch 2.0+)

---

### Question 16
- [ ] Done
**What is linear attention and how do variants like Performer approximate full attention?**

**Answer:**

**Definition:**
Linear attention reduces the O(n²) complexity of standard attention to **O(n)** by decomposing the attention computation using kernel approximations. Instead of computing the full n×n attention matrix, it computes attention as (K^T V) first, then multiplies by Q, changing the order of operations to avoid quadratic scaling.

**The Key Insight:**

Standard attention:
$$\text{Attention} = \text{softmax}(QK^T)V$$
Complexity: O(n² × d)

Linear attention (kernel trick):
$$\text{Attention} = \phi(Q)(\phi(K)^T V)$$
Complexity: O(n × d²)

**Where the savings come from:**
- Standard: (n×d) × (d×n) = n×n matrix, then (n×n) × (n×d)
- Linear: (d×n) × (n×d) = d×d matrix first, then (n×d) × (d×d)

When d << n (dimension << sequence length), this is much faster.

**Performer (Google, 2020):**

Approximates softmax using **Random Fourier Features (FAVOR+)**:

$$\text{softmax}(QK^T) \approx \phi(Q)\phi(K)^T$$

Where:
$$\phi(x) = \frac{1}{\sqrt{m}} \begin{bmatrix} \sin(\omega_1^T x) \\ \cos(\omega_1^T x) \\ \vdots \\ \sin(\omega_m^T x) \\ \cos(\omega_m^T x) \end{bmatrix}$$

- ω_i are random samples from a specific distribution
- m = number of random features (controls approximation quality)

**Comparison of Linear Attention Variants:**

| Model | Kernel/Method | Complexity | Quality |
|-------|---------------|------------|---------|
| Performer | FAVOR+ (random features) | O(n) | Good approximation |
| Linear Transformer | elu(x) + 1 | O(n) | Simpler, less accurate |
| Linformer | Low-rank projection | O(n) | Projects K, V to k dims |
| Nyströmformer | Nyström approximation | O(n) | Landmark-based |

**Trade-offs:**

| Aspect | Standard Attention | Linear Attention |
|--------|-------------------|------------------|
| Complexity | O(n²) | O(n) |
| Exact | Yes | No (approximation) |
| Quality | Best | Slightly lower |
| Long sequences | Limited | Excellent |
| Training stability | Stable | Can be tricky |

**Intuition:**
- Standard: Compute all pairwise similarities (expensive)
- Linear: Summarize keys into a "context vector" (d×d), then query it
- Think of it as: instead of comparing with each key, compare with an average

**Python Code Example:**
```python
import numpy as np

def standard_attention(Q, K, V):
    """O(n²) complexity"""
    n, d = Q.shape
    scores = Q @ K.T / np.sqrt(d)  # n×n matrix
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    return weights @ V

def linear_attention_simple(Q, K, V, eps=1e-6):
    """
    O(n) complexity using simple kernel (elu + 1)
    """
    # Feature map: elu(x) + 1 (ensures positivity)
    def feature_map(x):
        return np.maximum(x, 0) + 1  # Simplified; real uses elu
    
    Q_prime = feature_map(Q)  # (n, d)
    K_prime = feature_map(K)  # (n, d)
    
    # Key insight: compute K^T V first (d×d), then Q @ result
    KV = K_prime.T @ V        # (d, d) - O(n×d²)
    Z = K_prime.sum(axis=0)   # (d,) - normalization factor
    
    # Final computation: O(n×d²)
    numerator = Q_prime @ KV           # (n, d)
    denominator = Q_prime @ Z + eps    # (n,)
    
    return numerator / denominator[:, None]

def performer_attention(Q, K, V, num_features=64):
    """
    Performer-style with random Fourier features
    """
    d = Q.shape[-1]
    
    # Random projection matrix
    omega = np.random.randn(d, num_features) / np.sqrt(d)
    
    def random_feature_map(x):
        proj = x @ omega  # (n, num_features)
        return np.concatenate([np.sin(proj), np.cos(proj)], axis=-1) / np.sqrt(num_features)
    
    Q_prime = random_feature_map(Q)
    K_prime = random_feature_map(K)
    
    # Linear attention computation
    KV = K_prime.T @ V
    Z = K_prime.sum(axis=0)
    
    numerator = Q_prime @ KV
    denominator = Q_prime @ Z + 1e-6
    
    return numerator / denominator[:, None]

# Compare
n, d = 100, 32
Q = np.random.randn(n, d)
K = np.random.randn(n, d)
V = np.random.randn(n, d)

out_std = standard_attention(Q, K, V)
out_linear = linear_attention_simple(Q, K, V)

print("Standard output shape:", out_std.shape)
print("Linear output shape:", out_linear.shape)
# Note: outputs are similar but not identical (approximation)
```

**Interview Tips:**
- Linear attention is an **approximation** — trades accuracy for speed
- Works well for very long sequences (16K+ tokens)
- Performer showed linear attention can match Transformer on some tasks
- Modern trend: Prefer Flash Attention (exact) over linear attention for quality
- State-space models (Mamba) are alternative O(n) approach

---

### Question 17
- [ ] Done
**How do you visualize and interpret attention weights for model explainability?**

**Answer:**

**Definition:**
Attention weights can be extracted and visualized as heatmaps showing which tokens each position attends to. This provides interpretability by revealing learned patterns (syntax, coreference, semantic relationships). However, attention weights alone may not fully explain model predictions — they show "where" the model looks, not "why" it makes decisions.

**Methods for Visualizing Attention:**

| Method | Description | Use Case |
|--------|-------------|----------|
| **Heatmaps** | Token-to-token attention matrix | Single sentence analysis |
| **Attention flow** | Trace attention across layers | Understanding composition |
| **Head-level analysis** | Compare patterns across heads | Find specialized heads |
| **BertViz** | Interactive visualization tool | Detailed exploration |
| **Rollout/Flow** | Aggregate attention across layers | Layer interaction |

**What Attention Patterns Reveal:**

| Pattern | Example | Interpretation |
|---------|---------|----------------|
| **Diagonal** | Each token attends to itself | Self-referential |
| **Previous token** | Strong attention to token t-1 | Local dependencies |
| **Delimiter** | Attention to [SEP], [CLS] | Structural tokens |
| **Coreference** | "it" → "cat" | Pronoun resolution |
| **Syntactic** | Verb → Subject | Grammatical relations |

**Important Caveats:**

1. **Attention ≠ Importance:** High attention doesn't mean high contribution
2. **Multiple paths:** Information flows through many layers
3. **Residual connections:** Skip connections bypass attention
4. **Head redundancy:** Multiple heads may capture similar patterns

**Attention Rollout vs Raw Attention:**
- **Raw attention:** Attention weights from single layer
- **Rollout:** Multiply attention matrices across layers to trace full path
- **Attention flow:** Uses matrix inversion for better attribution

**Python Code Example:**
```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_attention(attention_weights, tokens, layer=0, head=0):
    """
    Visualize attention as a heatmap
    """
    # Get attention for specific layer and head
    attn = attention_weights[layer][head]  # (seq_len, seq_len)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(attn, cmap='Blues')
    
    # Set labels
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens)
    
    ax.set_xlabel('Key (attended to)')
    ax.set_ylabel('Query (attending from)')
    ax.set_title(f'Attention Weights - Layer {layer}, Head {head}')
    
    plt.colorbar(im)
    plt.tight_layout()
    return fig

def attention_rollout(attention_weights):
    """
    Compute attention rollout across layers
    Traces how attention flows from input to final layer
    """
    num_layers = len(attention_weights)
    seq_len = attention_weights[0].shape[-1]
    
    # Average across heads
    attn_matrices = [a.mean(axis=0) for a in attention_weights]
    
    # Add residual connection (identity matrix)
    rollout = np.eye(seq_len)
    
    for attn in attn_matrices:
        # Add residual
        attn_with_residual = 0.5 * attn + 0.5 * np.eye(seq_len)
        rollout = attn_with_residual @ rollout
    
    return rollout

# Example with mock data
tokens = ['[CLS]', 'The', 'cat', 'sat', 'on', 'the', 'mat', '[SEP]']
seq_len = len(tokens)

# Simulate attention weights (2 layers, 4 heads each)
attention_weights = [
    np.random.dirichlet(np.ones(seq_len), (4, seq_len)),  # Layer 0: 4 heads
    np.random.dirichlet(np.ones(seq_len), (4, seq_len)),  # Layer 1: 4 heads
]

# Visualize single head
# fig = visualize_attention(attention_weights, tokens, layer=0, head=0)
# plt.show()

# Compute rollout
rollout = attention_rollout(attention_weights)
print("Rollout shape:", rollout.shape)
print("Attention from [CLS] to each token:", np.round(rollout[0], 3))
```

**Using BertViz Library:**
```python
from transformers import BertTokenizer, BertModel
from bertviz import head_view

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

# Get attention
text = "The cat sat on the mat."
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)

# outputs.attentions: tuple of (batch, heads, seq, seq) per layer
attention = outputs.attentions
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# Visualize (in Jupyter notebook)
# head_view(attention, tokens)
```

**Interview Tips:**
- Attention visualization is useful for debugging, not ground-truth explanation
- Use **attention rollout** for multi-layer analysis
- Different heads learn different patterns (some are interpretable, some not)
- For rigorous explainability, prefer gradient-based methods (Integrated Gradients)

---

## BERT & Encoder Models

### Question 18
- [ ] Done
**How do you choose between BERT, RoBERTa, and DistilBERT for production deployment?**

**Answer:**

**Definition:**
Choose based on trade-off between **accuracy** and **efficiency**. RoBERTa offers best accuracy (improved training), BERT is the baseline with good ecosystem support, and DistilBERT provides 60% size reduction with ~97% performance — ideal for latency-sensitive or resource-constrained deployments.

**Quick Decision Framework:**

| Requirement | Best Choice | Why |
|-------------|-------------|-----|
| **Maximum accuracy** | RoBERTa | Better training, no NSP, larger batches |
| **Balanced performance** | BERT | Well-tested, broad compatibility |
| **Low latency/cost** | DistilBERT | 40% smaller, 60% faster, 97% quality |
| **Edge/mobile deployment** | DistilBERT | Smallest footprint |
| **Fine-tuning with limited data** | BERT or RoBERTa | More capacity helps generalization |

**Model Comparison:**

| Aspect | BERT-base | RoBERTa-base | DistilBERT |
|--------|-----------|--------------|------------|
| **Parameters** | 110M | 125M | 66M |
| **Layers** | 12 | 12 | 6 |
| **Hidden size** | 768 | 768 | 768 |
| **Training data** | 16GB | 160GB | Distilled from BERT |
| **Relative accuracy** | Baseline | +2-3% | -3% |
| **Inference speed** | 1x | 1x | 1.6x |
| **Memory** | 1x | 1x | 0.6x |

**Key Differences:**

| Feature | BERT | RoBERTa | DistilBERT |
|---------|------|---------|------------|
| **NSP task** | Yes | Removed | No |
| **Dynamic masking** | No | Yes | No |
| **Training data** | 16GB | 160GB | - |
| **Batch size** | 256 | 8000 | - |
| **Architecture** | Original | Same | Half layers |

**Scenario-Based Selection:**

**Scenario 1: Real-time API with strict latency (<50ms)**
→ **DistilBERT** + quantization + ONNX optimization

**Scenario 2: Offline batch classification (accuracy critical)**
→ **RoBERTa-large** for maximum quality

**Scenario 3: General production NLU with moderate resources**
→ **BERT-base** or **RoBERTa-base** with caching

**Scenario 4: Sentiment analysis on mobile app**
→ **DistilBERT** with TensorFlow Lite or CoreML

**Python Code Example:**
```python
from transformers import AutoTokenizer, AutoModel
import time

def benchmark_model(model_name, text, num_runs=10):
    """Compare inference speed"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    # Warmup
    _ = model(**inputs)
    
    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        _ = model(**inputs)
    elapsed = (time.time() - start) / num_runs * 1000
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    return {
        'model': model_name,
        'params_M': num_params / 1e6,
        'latency_ms': elapsed
    }

# Compare models
text = "This is a sample sentence for benchmarking transformer models."
models = ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased']

# results = [benchmark_model(m, text) for m in models]
# for r in results:
#     print(f"{r['model']}: {r['params_M']:.1f}M params, {r['latency_ms']:.1f}ms")

# Production deployment choice
def select_model(latency_budget_ms, accuracy_priority):
    """
    Simple decision logic
    """
    if latency_budget_ms < 30:
        return 'distilbert-base-uncased'
    elif accuracy_priority == 'high':
        return 'roberta-base'
    else:
        return 'bert-base-uncased'

print("Latency <30ms:", select_model(25, 'medium'))   # distilbert
print("High accuracy:", select_model(100, 'high'))    # roberta
print("Balanced:", select_model(50, 'medium'))        # bert
```

**Interview Tips:**
- DistilBERT maintains 97% of BERT performance with 40% fewer params
- RoBERTa's gains come from training choices, not architecture changes
- For multilingual: use mBERT or XLM-RoBERTa
- Consider ONNX export + INT8 quantization for further speedup

---

### Question 19
- [ ] Done
**What are the key architectural differences between BERT's masked language modeling and RoBERTa's training approach?**

**Answer:**

**Definition:**
BERT and RoBERTa share the **same architecture** (Transformer encoder) but differ in **training methodology**. RoBERTa removes Next Sentence Prediction (NSP), uses dynamic masking, trains on 10x more data with larger batches, and trains longer. These changes collectively improve performance by 2-3% on downstream tasks.

**Key Differences:**

| Aspect | BERT | RoBERTa |
|--------|------|---------|
| **NSP (Next Sentence Prediction)** | Yes | Removed |
| **Masking strategy** | Static (fixed per epoch) | Dynamic (new masks each epoch) |
| **Training data** | 16GB (Wikipedia + Books) | 160GB (+ CC-News, Stories, Web) |
| **Batch size** | 256 | 8,000 |
| **Training steps** | 1M | 500K (but with larger batches) |
| **Input format** | Sentence pairs | Full sentences up to 512 |
| **Byte-pair encoding** | WordPiece | BPE (50K vocab) |

**Why Each Change Helps:**

**1. Removing NSP:**
- BERT's NSP task: Predict if sentence B follows sentence A
- Finding: NSP hurts performance (too easy, wrong signal)
- RoBERTa: Use full sentences from single document instead

**2. Dynamic Masking:**
- BERT: Same tokens masked across all epochs (static)
- RoBERTa: Generate new mask each time sequence is fed
- Benefit: Model sees different masked versions → better generalization

**3. Larger Batches + More Data:**
- Larger batches → more stable gradients → higher learning rates
- 10x more data → better language understanding
- Longer training → fuller utilization of data

**4. Full-length Sequences:**
- BERT: Often short sentence pairs
- RoBERTa: Pack multiple sentences up to 512 tokens
- Benefit: More efficient, learns longer-range patterns

**Performance Impact:**

| Benchmark | BERT-base | RoBERTa-base | Improvement |
|-----------|-----------|--------------|-------------|
| MNLI | 84.4 | 87.6 | +3.2 |
| QNLI | 90.5 | 92.8 | +2.3 |
| SST-2 | 93.0 | 94.8 | +1.8 |
| SQuAD v1.1 | 88.5 | 91.5 | +3.0 |

**MLM (Masked Language Modeling) - Same in Both:**

$$\mathcal{L}_{MLM} = -\sum_{i \in \text{masked}} \log P(x_i | x_{\text{context}})$$

- Randomly mask 15% of tokens
- Predict masked tokens from context
- 80% [MASK], 10% random, 10% unchanged

**Python Code Example:**
```python
from transformers import BertTokenizer, RobertaTokenizer
from transformers import BertForMaskedLM, RobertaForMaskedLM

# BERT example
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# RoBERTa example  
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForMaskedLM.from_pretrained('roberta-base')

# Key differences in usage:
text = "The quick brown [MASK] jumps over the lazy dog."

# BERT
bert_inputs = bert_tokenizer(text, return_tensors='pt')
print("BERT vocab size:", bert_tokenizer.vocab_size)  # 30522

# RoBERTa uses <mask> token
text_roberta = "The quick brown <mask> jumps over the lazy dog."
roberta_inputs = roberta_tokenizer(text_roberta, return_tensors='pt')
print("RoBERTa vocab size:", roberta_tokenizer.vocab_size)  # 50265

# Training differences summary
training_comparison = {
    'BERT': {
        'data': '16GB',
        'batch_size': 256,
        'nsp': True,
        'masking': 'static',
        'tokenizer': 'WordPiece'
    },
    'RoBERTa': {
        'data': '160GB',
        'batch_size': 8000,
        'nsp': False,
        'masking': 'dynamic',
        'tokenizer': 'BPE'
    }
}

for model, config in training_comparison.items():
    print(f"\n{model}:")
    for k, v in config.items():
        print(f"  {k}: {v}")
```

**Interview Tips:**
- RoBERTa = "BERT done right" — same architecture, better training
- NSP removal is the most impactful single change
- These insights influenced later models (ALBERT, ELECTRA, DeBERTa)
- When citing performance, note that RoBERTa trained on 10x more data

---

### Question 20
- [ ] Done
**When should you use BERT's [CLS] token versus pooling strategies for document-level representations?**

**Answer:**

**Definition:**
Use **[CLS] token** when fine-tuning BERT end-to-end for classification (it learns to aggregate sequence information). Use **mean pooling** for sentence embeddings without fine-tuning or when [CLS] underperforms. For semantic similarity tasks, mean pooling of token embeddings often outperforms [CLS] out-of-the-box.

**Pooling Strategies:**

| Strategy | Formula | When to Use |
|----------|---------|-------------|
| **[CLS] token** | Take hidden state of [CLS] | Fine-tuned classification |
| **Mean pooling** | Average all token embeddings | Semantic similarity, no fine-tuning |
| **Max pooling** | Max over each dimension | Capture salient features |
| **Weighted mean** | Attention-weighted average | When certain tokens matter more |
| **Last 4 layers concat** | Concatenate multiple layers | Richer representations |

**When to Use Each:**

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| **Fine-tuned classification** | [CLS] | Learns to aggregate for task |
| **Sentence embeddings (frozen)** | Mean pooling | [CLS] not trained for this |
| **Semantic similarity** | Mean pooling | Better sentence representation |
| **Named Entity Recognition** | Token embeddings | Need per-token predictions |
| **Question Answering** | Token embeddings | Span extraction |

**Why Mean Pooling Works Better for Embeddings:**
- BERT's [CLS] was trained for NSP, not sentence similarity
- Without fine-tuning, [CLS] doesn't represent sentence meaning well
- Mean pooling captures information from all tokens
- Sentence-BERT specifically trains for similarity using mean pooling

**Empirical Evidence:**

| Method | STS Benchmark (without fine-tuning) |
|--------|-------------------------------------|
| BERT [CLS] | 29.2 (poor) |
| BERT Mean Pool | 54.8 (better) |
| SBERT Mean Pool | 76.5 (trained for similarity) |

**Python Code Example:**
```python
import torch
from transformers import BertTokenizer, BertModel

def get_cls_embedding(model, tokenizer, text):
    """Extract [CLS] token embedding"""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # [CLS] is the first token
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
    return cls_embedding

def get_mean_pooling(model, tokenizer, text):
    """Mean pooling over all tokens (excluding padding)"""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get attention mask to ignore padding
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden)
    
    # Expand mask for broadcasting
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # Sum embeddings, divide by number of real tokens
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    
    return mean_embeddings

def get_max_pooling(model, tokenizer, text):
    """Max pooling over all tokens"""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    
    # Set padding tokens to large negative value before max
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    token_embeddings[mask_expanded == 0] = -1e9
    
    max_embeddings = torch.max(token_embeddings, dim=1)[0]
    return max_embeddings

# Usage
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

text = "This is a sample sentence for embedding extraction."

cls_emb = get_cls_embedding(model, tokenizer, text)
mean_emb = get_mean_pooling(model, tokenizer, text)
max_emb = get_max_pooling(model, tokenizer, text)

print("CLS embedding shape:", cls_emb.shape)   # (1, 768)
print("Mean embedding shape:", mean_emb.shape) # (1, 768)
print("Max embedding shape:", max_emb.shape)   # (1, 768)
```

**Interview Tips:**
- For classification with fine-tuning: [CLS] is standard and effective
- For sentence embeddings without fine-tuning: use mean pooling
- Sentence-BERT (SBERT) is purpose-built for sentence similarity
- Can also try: [CLS] from last layer + [CLS] from second-to-last layer

---

### Question 21
- [ ] Done
**What are the best practices for handling long documents that exceed BERT's 512-token limit?**

**Answer:**

**Definition:**
When documents exceed 512 tokens, use strategies like **chunking with overlap** (split into overlapping segments), **hierarchical models** (encode chunks then aggregate), **Longformer/BigBird** (sparse attention for 4K+ tokens), or **extract most relevant sections** using heuristics or retrieval. The best approach depends on task requirements and available resources.

**Strategies for Long Documents:**

| Strategy | Description | Complexity | Best For |
|----------|-------------|------------|----------|
| **Truncation** | Keep first/last 512 tokens | Low | When start/end has key info |
| **Chunking + Pooling** | Split, encode, aggregate | Medium | Classification |
| **Sliding window** | Overlapping chunks | Medium | QA, NER |
| **Hierarchical** | Chunk → BERT → Document BERT | High | Document classification |
| **Longformer/BigBird** | Sparse attention models | Low (if available) | Native long support |
| **Retrieval-based** | Find relevant chunks first | Medium | QA, specific tasks |

**Strategy Details:**

**1. Chunking with Overlap:**
```
Document: [tokens 1-600]
Chunk 1: [1-512]
Chunk 2: [400-512] (overlap) + [513-600]

- Overlap prevents losing context at boundaries
- Typical overlap: 50-128 tokens
```

**2. Aggregation Methods:**
- **Mean pooling:** Average [CLS] embeddings from all chunks
- **Max pooling:** Take max across chunks per dimension
- **Attention:** Learn weights for each chunk
- **First + Last:** Concatenate first and last chunk embeddings

**3. Hierarchical Approach:**
```
Document → Split into paragraphs
           ↓
Each paragraph → BERT → paragraph embedding
           ↓
Paragraph embeddings → Transformer/LSTM → document embedding
           ↓
Classification head → prediction
```

**Python Code Example:**
```python
import torch
from transformers import BertTokenizer, BertModel

def chunk_document(text, tokenizer, max_length=512, overlap=128):
    """
    Split document into overlapping chunks
    """
    tokens = tokenizer.tokenize(text)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = min(start + max_length - 2, len(tokens))  # -2 for [CLS], [SEP]
        chunk_tokens = tokens[start:end]
        chunks.append(chunk_tokens)
        
        if end >= len(tokens):
            break
        start += max_length - overlap - 2  # Move by (max_length - overlap)
    
    return chunks

def encode_long_document(text, tokenizer, model, max_length=512, 
                         overlap=128, aggregation='mean'):
    """
    Encode document longer than 512 tokens
    """
    chunks = chunk_document(text, tokenizer, max_length, overlap)
    
    chunk_embeddings = []
    for chunk_tokens in chunks:
        # Add special tokens
        input_tokens = ['[CLS]'] + chunk_tokens + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        
        # Pad if needed
        attention_mask = [1] * len(input_ids)
        padding_length = max_length - len(input_ids)
        input_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        
        inputs = {
            'input_ids': torch.tensor([input_ids]),
            'attention_mask': torch.tensor([attention_mask])
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            chunk_embeddings.append(cls_embedding)
    
    # Stack all chunk embeddings
    all_embeddings = torch.cat(chunk_embeddings, dim=0)  # (num_chunks, hidden)
    
    # Aggregate
    if aggregation == 'mean':
        document_embedding = all_embeddings.mean(dim=0, keepdim=True)
    elif aggregation == 'max':
        document_embedding = all_embeddings.max(dim=0, keepdim=True)[0]
    elif aggregation == 'first_last':
        document_embedding = torch.cat([all_embeddings[0:1], all_embeddings[-1:]], dim=1)
    
    return document_embedding, all_embeddings

# Example usage
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Long document (simulated)
long_text = "This is a very long document. " * 200  # ~800 tokens

doc_emb, chunk_embs = encode_long_document(long_text, tokenizer, model)
print(f"Number of chunks: {chunk_embs.shape[0]}")
print(f"Document embedding shape: {doc_emb.shape}")
```

**Longformer/BigBird Approach:**
```python
from transformers import LongformerTokenizer, LongformerModel

tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = LongformerModel.from_pretrained('allenai/longformer-base-4096')

# Can handle up to 4096 tokens natively
text = "Very long document..." * 500
inputs = tokenizer(text, return_tensors='pt', max_length=4096, truncation=True)
outputs = model(**inputs)
```

**Interview Tips:**
- Simple truncation works surprisingly well for some tasks (first 512 often sufficient)
- For QA, use sliding window and take best answer from any chunk
- Longformer is drop-in replacement for BERT with 4K context
- Consider task: classification may only need summary, QA needs full document

---

### Question 22
- [ ] Done
**How do you implement domain adaptation when fine-tuning BERT for specialized text classification?**

**Answer:**

**Definition:**
Domain adaptation bridges the gap between BERT's general pretraining corpus and specialized domain text (medical, legal, financial). Implement through: **continued pretraining** on domain text (MLM on unlabeled domain data), followed by **task-specific fine-tuning**. For best results, combine with domain vocabulary expansion and gradual unfreezing.

**Domain Adaptation Pipeline:**

```
General BERT (Wikipedia, Books)
        ↓
Step 1: Continued Pretraining (MLM on domain corpus - unlabeled)
        ↓
Domain-Adapted BERT
        ↓
Step 2: Task Fine-tuning (classification on labeled data)
        ↓
Final Domain-Specific Classifier
```

**Key Strategies:**

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Continued pretraining** | MLM on domain text | Large unlabeled domain corpus |
| **Domain vocabulary** | Add domain-specific tokens | Many OOV terms (biomedical) |
| **Gradual unfreezing** | Unfreeze layers progressively | Limited labeled data |
| **Lower learning rate** | 2e-5 or less for adaptation | Preserve general knowledge |
| **Use domain BERT** | BioBERT, FinBERT, LegalBERT | If available for your domain |

**Step-by-Step Implementation:**

**1. Continued Pretraining (Domain Adaptation):**
```python
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load base model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Load domain corpus (e.g., medical abstracts)
# dataset = load_dataset('your_domain_corpus')

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512)

# tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Data collator for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Training arguments for continued pretraining
training_args = TrainingArguments(
    output_dir='./domain_bert',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=5e-5,  # Lower than initial pretraining
    warmup_steps=1000,
    save_steps=5000,
)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=tokenized_dataset,
# )
# trainer.train()
```

**2. Task-Specific Fine-tuning:**
```python
from transformers import BertForSequenceClassification

# Load domain-adapted model
model = BertForSequenceClassification.from_pretrained(
    './domain_bert',
    num_labels=num_classes
)

# Fine-tune on labeled task data with lower learning rate
training_args = TrainingArguments(
    output_dir='./domain_classifier',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    learning_rate=2e-5,  # Lower for fine-tuning
    weight_decay=0.01,
    warmup_ratio=0.1,
)
```

**3. Gradual Unfreezing (Alternative):**
```python
def freeze_layers(model, num_layers_to_freeze):
    """Freeze bottom N transformer layers"""
    # Freeze embeddings
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    
    # Freeze specified layers
    for i in range(num_layers_to_freeze):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False

# Training schedule:
# Epoch 1-2: Freeze all but top 2 layers
freeze_layers(model, num_layers_to_freeze=10)
# Train...

# Epoch 3-4: Unfreeze more layers
freeze_layers(model, num_layers_to_freeze=6)
# Train...

# Epoch 5+: Unfreeze all
for param in model.parameters():
    param.requires_grad = True
```

**Domain Vocabulary Expansion:**
```python
# Add domain-specific tokens
domain_tokens = ['[DRUG]', '[DISEASE]', '[PROTEIN]', 'hydroxychloroquine']
tokenizer.add_tokens(domain_tokens)
model.resize_token_embeddings(len(tokenizer))
```

**Interview Tips:**
- Always do continued pretraining before task fine-tuning for new domains
- Use domain-specific BERT if available (saves compute)
- Lower learning rates prevent "catastrophic forgetting" of general knowledge
- More domain data → more epochs of continued pretraining

---

### Question 23
- [ ] Done
**What techniques help reduce overfitting when fine-tuning BERT on small specialized datasets?**

**Answer:**

**Definition:**
With small datasets (<5K samples), BERT easily overfits. Key techniques: **lower learning rates** (2e-5 to 5e-5), **fewer epochs** (2-4), **dropout regularization**, **weight decay**, **early stopping**, **data augmentation**, and **gradual unfreezing**. For very small datasets (<500 samples), consider **few-shot learning** or **SetFit** approaches.

**Regularization Techniques:**

| Technique | Implementation | Effect |
|-----------|----------------|--------|
| **Lower learning rate** | 2e-5 instead of higher | Smaller updates, preserve pretrained weights |
| **Fewer epochs** | 2-4 epochs max | Stop before memorization |
| **Weight decay** | L2 regularization (0.01) | Penalize large weights |
| **Dropout** | Increase to 0.2-0.3 | Random neuron dropping |
| **Early stopping** | Monitor val loss | Stop when val loss increases |
| **Gradient clipping** | max_grad_norm=1.0 | Prevent unstable updates |

**Data-Side Techniques:**

| Technique | Description |
|-----------|-------------|
| **Data augmentation** | Synonym replacement, back-translation |
| **Cross-validation** | K-fold for robust evaluation |
| **Pseudo-labeling** | Use model to label unlabeled data |
| **Few-shot prompting** | Use LLM to generate training examples |

**Recommended Hyperparameters for Small Data:**

| Dataset Size | Learning Rate | Epochs | Batch Size | Weight Decay |
|--------------|---------------|--------|------------|--------------|
| <500 | 1e-5 | 2-3 | 8-16 | 0.01 |
| 500-2K | 2e-5 | 3-4 | 16 | 0.01 |
| 2K-10K | 3e-5 | 3-5 | 16-32 | 0.01 |
| >10K | 5e-5 | 3-5 | 32 | 0.01 |

**Python Code Example:**
```python
from transformers import BertForSequenceClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import torch

# Load model with increased dropout
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    hidden_dropout_prob=0.2,  # Increase from default 0.1
    attention_probs_dropout_prob=0.2
)

# Training arguments for small dataset
training_args = TrainingArguments(
    output_dir='./results',
    
    # Regularization settings
    num_train_epochs=3,              # Few epochs
    learning_rate=2e-5,              # Low learning rate
    weight_decay=0.01,               # L2 regularization
    warmup_ratio=0.1,                # Gradual warmup
    max_grad_norm=1.0,               # Gradient clipping
    
    # Batch size
    per_device_train_batch_size=16,
    
    # Evaluation
    eval_strategy='steps',
    eval_steps=50,
    load_best_model_at_end=True,     # For early stopping
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    
    # Logging
    logging_steps=10,
)

# Early stopping callback
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,        # Stop if no improvement for 3 evals
    early_stopping_threshold=0.01
)

# Trainer with early stopping
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     callbacks=[early_stopping]
# )

# Data augmentation example
def augment_text(text, method='synonym'):
    """Simple text augmentation"""
    import random
    
    if method == 'synonym':
        # Replace words with synonyms (simplified)
        synonyms = {
            'good': ['great', 'excellent', 'nice'],
            'bad': ['poor', 'terrible', 'awful'],
        }
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in synonyms and random.random() < 0.1:
                words[i] = random.choice(synonyms[word.lower()])
        return ' '.join(words)
    
    elif method == 'deletion':
        # Random word deletion
        words = text.split()
        if len(words) > 5:
            idx = random.randint(0, len(words) - 1)
            words.pop(idx)
        return ' '.join(words)
    
    return text

# Gradual unfreezing
def gradual_unfreeze_training(model, train_dataset, val_dataset):
    """Train with gradual unfreezing"""
    
    # Phase 1: Only classifier head
    for param in model.bert.parameters():
        param.requires_grad = False
    # Train for 1 epoch...
    
    # Phase 2: Unfreeze top layers
    for param in model.bert.encoder.layer[-4:].parameters():
        param.requires_grad = True
    # Train for 1-2 epochs...
    
    # Phase 3: Unfreeze all
    for param in model.parameters():
        param.requires_grad = True
    # Train for 1-2 epochs...
```

**Interview Tips:**
- With <500 samples, consider SetFit or few-shot approaches instead of fine-tuning
- Monitor validation loss — if it increases while train loss decreases → overfitting
- Cross-validation essential for small datasets (single split is unreliable)
- Consider using LoRA for parameter-efficient fine-tuning (fewer trainable params)

---

### Question 24
- [ ] Done
**When should you use specialized BERT variants (BioBERT, FinBERT, LegalBERT) over general models?**

**Answer:**

**Definition:**
Use specialized BERT variants when working with **domain-specific text** that contains vocabulary, patterns, or semantics significantly different from general text. These models are pretrained on domain corpora, providing better understanding of specialized terminology and concepts. Use them when: domain text is abundant, task is domain-specific, and general BERT underperforms.

**When to Use Domain-Specific BERT:**

| Use Domain BERT | Use General BERT |
|-----------------|------------------|
| Domain-specific vocabulary (medical terms, legal jargon) | General text classification |
| Specialized document types (clinical notes, contracts) | Multi-domain applications |
| Semantic nuances matter (drug interactions, legal precedents) | Limited domain data |
| Domain BERT exists and is well-maintained | Cross-domain transfer needed |

**Popular Domain BERT Variants:**

| Model | Domain | Pretrained On | Best For |
|-------|--------|---------------|----------|
| **BioBERT** | Biomedical | PubMed, PMC | Medical NER, QA, relation extraction |
| **ClinicalBERT** | Clinical | MIMIC-III notes | Clinical notes, patient records |
| **SciBERT** | Scientific | Semantic Scholar papers | Scientific text, citations |
| **FinBERT** | Financial | Financial news, reports | Sentiment, entity extraction |
| **LegalBERT** | Legal | Legal documents, cases | Contract analysis, case prediction |
| **CodeBERT** | Programming | Code + comments | Code search, bug detection |
| **PatentBERT** | Patents | Patent documents | Patent classification, search |

**Performance Comparison Example (Biomedical):**

| Task | BERT-base | BioBERT | Improvement |
|------|-----------|---------|-------------|
| NER (BC5CDR) | 85.0 F1 | 88.0 F1 | +3.0 |
| QA (BioASQ) | 40.0 | 47.0 | +7.0 |
| RE (ChemProt) | 68.0 | 74.0 | +6.0 |

**Decision Framework:**

```
Is your text domain-specific?
├── No → Use BERT/RoBERTa
└── Yes
    ├── Does domain BERT exist?
    │   ├── Yes → Use domain BERT
    │   └── No → Do continued pretraining on BERT
    │
    ├── How much domain data?
    │   ├── Large corpus → Continued pretraining helps
    │   └── Small corpus → Domain BERT + fine-tune
    │
    └── Task type?
        ├── Domain-specific entities → Domain BERT critical
        └── General semantics → General BERT may suffice
```

**Python Code Example:**
```python
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Load domain-specific models
def load_domain_model(domain):
    """Load appropriate domain-specific model"""
    
    model_map = {
        'biomedical': 'dmis-lab/biobert-base-cased-v1.1',
        'clinical': 'emilyalsentzer/Bio_ClinicalBERT',
        'scientific': 'allenai/scibert_scivocab_uncased',
        'financial': 'ProsusAI/finbert',
        'legal': 'nlpaueb/legal-bert-base-uncased',
        'code': 'microsoft/codebert-base',
        'general': 'bert-base-uncased'
    }
    
    model_name = model_map.get(domain, model_map['general'])
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    return tokenizer, model

# Example: Compare tokenization
general_tok = AutoTokenizer.from_pretrained('bert-base-uncased')
bio_tok = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

medical_text = "The patient was prescribed hydroxychloroquine for SLE."

general_tokens = general_tok.tokenize(medical_text)
bio_tokens = bio_tok.tokenize(medical_text)

print("General BERT tokens:", general_tokens)
# ['the', 'patient', 'was', 'prescribed', 'hydro', '##xy', '##chlor', '##oqui', '##ne', ...]

print("BioBERT tokens:", bio_tokens)
# ['The', 'patient', 'was', 'prescribed', 'hydroxychloroquine', ...]  # Fewer subwords

# Fewer subwords = better representation of domain terms

# Fine-tuning domain model
def fine_tune_domain_model(domain, train_data, num_labels):
    model_name = {
        'biomedical': 'dmis-lab/biobert-base-cased-v1.1',
        'financial': 'ProsusAI/finbert',
    }.get(domain)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    
    # Fine-tune as usual...
    return model
```

**Interview Tips:**
- Domain BERT biggest advantage: better tokenization of domain vocabulary
- "hydroxychloroquine" = 1 token in BioBERT vs 5+ tokens in general BERT
- If no domain BERT exists, do continued pretraining on your domain corpus
- Domain models may lack general knowledge — evaluate on both domain and general benchmarks

---

### Question 25
- [ ] Done
**How do you handle catastrophic forgetting when continually fine-tuning BERT on new domains?**

**Answer:**

**Definition:**
Catastrophic forgetting occurs when fine-tuning on new data causes the model to lose performance on previously learned tasks. Handle it through: **Elastic Weight Consolidation (EWC)** to protect important weights, **replay/rehearsal** of old data, **progressive networks** that add capacity, **adapter layers** for task-specific parameters, or **multi-task learning** with balanced sampling.

**Strategies to Prevent Catastrophic Forgetting:**

| Strategy | Mechanism | Complexity |
|----------|-----------|------------|
| **Data replay** | Mix old data with new during training | Low |
| **EWC** | Penalize changes to important weights | Medium |
| **Adapter modules** | Add small task-specific layers | Low |
| **LoRA** | Low-rank adaptation, freeze base | Low |
| **Progressive networks** | Add new columns, freeze old | High |
| **Multi-task learning** | Train on all tasks jointly | Medium |
| **Knowledge distillation** | Regularize toward old model's outputs | Medium |

**How Each Works:**

**1. Elastic Weight Consolidation (EWC):**
$$\mathcal{L}_{total} = \mathcal{L}_{new} + \lambda \sum_i F_i (θ_i - θ_i^*)^2$$

- F_i = Fisher information (importance of weight i)
- θ* = weights from old task
- Penalizes changing important weights

**2. Data Replay:**
- Keep subset of old task data
- Mix with new task data during training
- Maintains exposure to old distribution

**3. Adapter Modules:**
- Freeze BERT, add small trainable adapters
- Each task has own adapters
- Base model unchanged → no forgetting

**Python Code Example:**
```python
import torch
import torch.nn as nn

# Method 1: Elastic Weight Consolidation (EWC)
class EWCLoss:
    def __init__(self, model, fisher_dict, old_params, lambda_ewc=0.4):
        """
        fisher_dict: importance of each parameter
        old_params: parameter values from previous task
        """
        self.fisher = fisher_dict
        self.old_params = old_params
        self.lambda_ewc = lambda_ewc
    
    def compute_penalty(self, model):
        """Compute EWC penalty"""
        penalty = 0
        for name, param in model.named_parameters():
            if name in self.fisher:
                fisher = self.fisher[name]
                old_param = self.old_params[name]
                penalty += (fisher * (param - old_param).pow(2)).sum()
        return self.lambda_ewc * penalty

def compute_fisher_information(model, dataloader, num_samples=1000):
    """Estimate Fisher information for each parameter"""
    fisher = {name: torch.zeros_like(param) 
              for name, param in model.named_parameters()}
    
    model.eval()
    for batch in dataloader:
        if num_samples <= 0:
            break
        num_samples -= len(batch['input_ids'])
        
        outputs = model(**batch)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        
        # Sample from model's distribution
        labels = torch.argmax(log_probs, dim=-1)
        loss = nn.CrossEntropyLoss()(outputs.logits, labels)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.pow(2)
    
    # Normalize
    for name in fisher:
        fisher[name] /= num_samples
    
    return fisher


# Method 2: Adapter Modules (Parameter Efficient)
class AdapterLayer(nn.Module):
    """Task-specific adapter layer"""
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # Bottleneck: down -> activation -> up
        down = self.down_proj(x)
        activated = self.activation(down)
        up = self.up_proj(activated)
        return x + up  # Residual connection

class BERTWithAdapters(nn.Module):
    """BERT with task-specific adapters"""
    def __init__(self, bert_model, num_tasks, adapter_size=64):
        super().__init__()
        self.bert = bert_model
        hidden_size = bert_model.config.hidden_size
        
        # Freeze BERT
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Create adapters for each task
        self.adapters = nn.ModuleDict({
            f'task_{i}': AdapterLayer(hidden_size, adapter_size)
            for i in range(num_tasks)
        })
    
    def forward(self, task_id, **inputs):
        outputs = self.bert(**inputs)
        hidden = outputs.last_hidden_state
        
        # Apply task-specific adapter
        adapter = self.adapters[f'task_{task_id}']
        adapted = adapter(hidden)
        
        return adapted


# Method 3: Data Replay (Simple but Effective)
def create_replay_dataloader(old_data, new_data, replay_ratio=0.2):
    """Mix old and new data"""
    num_replay = int(len(new_data) * replay_ratio)
    
    # Sample from old data
    replay_indices = torch.randperm(len(old_data))[:num_replay]
    replay_samples = [old_data[i] for i in replay_indices]
    
    # Combine
    combined_data = list(new_data) + replay_samples
    
    return combined_data  # Create DataLoader from this
```

**Interview Tips:**
- Simplest approach: keep 10-20% of old data and mix during training
- Adapter modules = clean solution, ~2% extra parameters per task
- EWC is elegant but computationally expensive for large models
- LoRA is modern approach — parameter-efficient, minimal forgetting

---

## GPT & Decoder Models

### Question 26
- [ ] Done
**What are the cost-efficiency considerations when choosing between GPT-3.5-turbo and GPT-4?**

**Answer:**

**Definition:**
GPT-4 offers superior reasoning, accuracy, and instruction-following but costs 10-30x more than GPT-3.5-turbo. Choose GPT-3.5 for high-volume, simpler tasks (summarization, basic chat); choose GPT-4 for complex reasoning, code generation, and accuracy-critical applications. Optimize costs through caching, batching, and prompt optimization.

**Cost Comparison (as of 2024):**

| Model | Input Cost | Output Cost | Context |
|-------|------------|-------------|---------|
| **GPT-3.5-turbo** | $0.0005/1K | $0.0015/1K | 16K |
| **GPT-4-turbo** | $0.01/1K | $0.03/1K | 128K |
| **GPT-4** | $0.03/1K | $0.06/1K | 8K |

**Cost Ratio:** GPT-4-turbo is ~20x more expensive than GPT-3.5-turbo

**Decision Framework:**

| Scenario | Recommended | Rationale |
|----------|-------------|-----------|
| Simple chatbot | GPT-3.5-turbo | Good enough, low cost |
| Summarization | GPT-3.5-turbo | Simple task, high volume |
| Code generation | GPT-4 | Better accuracy saves debug time |
| Complex reasoning | GPT-4 | Multi-step logic required |
| Data extraction | GPT-3.5-turbo | Structured tasks work well |
| Legal/medical analysis | GPT-4 | Accuracy critical |
| Rapid prototyping | GPT-4 | Quality speeds iteration |
| Production at scale | GPT-3.5 + spot GPT-4 | Hybrid approach |

**Cost Optimization Strategies:**

| Strategy | Savings | Implementation |
|----------|---------|----------------|
| **Prompt caching** | 30-50% | Cache repeated prompts/responses |
| **Shorter prompts** | 10-30% | Minimize system prompt, be concise |
| **Tiered routing** | 40-60% | Use GPT-3.5 first, escalate to GPT-4 |
| **Batch processing** | 20-40% | Combine requests where possible |
| **Fine-tuning GPT-3.5** | 50%+ | Fine-tuned 3.5 can match base GPT-4 |
| **Max tokens limit** | Variable | Set reasonable output limits |

**Tiered Routing Architecture:**
```
User Query
    ↓
Complexity Classifier (cheap model or rules)
    ↓
├── Simple → GPT-3.5-turbo → Response
│
└── Complex → GPT-4 → Response
```

**Python Code Example:**
```python
import openai
from functools import lru_cache
import hashlib

class CostOptimizedLLM:
    def __init__(self):
        self.costs = {
            'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03}
        }
        self.total_cost = 0
    
    def estimate_cost(self, model, input_tokens, output_tokens):
        """Estimate cost for a request"""
        rates = self.costs[model]
        cost = (input_tokens * rates['input'] + 
                output_tokens * rates['output']) / 1000
        return cost
    
    def classify_complexity(self, query):
        """Simple heuristic for complexity classification"""
        complex_keywords = ['analyze', 'compare', 'reason', 'explain why',
                           'code', 'debug', 'multi-step', 'complex']
        
        query_lower = query.lower()
        complexity_score = sum(1 for kw in complex_keywords if kw in query_lower)
        
        return 'complex' if complexity_score >= 2 else 'simple'
    
    def get_cache_key(self, messages, model):
        """Create cache key for request"""
        content = str(messages) + model
        return hashlib.md5(content.encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def cached_completion(self, cache_key, model, messages_tuple):
        """Cached API call"""
        messages = list(messages_tuple)  # Convert back from tuple
        response = openai.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content
    
    def smart_completion(self, messages, force_gpt4=False):
        """Route to appropriate model based on complexity"""
        query = messages[-1]['content'] if messages else ''
        
        # Determine model
        if force_gpt4:
            model = 'gpt-4-turbo'
        elif self.classify_complexity(query) == 'complex':
            model = 'gpt-4-turbo'
        else:
            model = 'gpt-3.5-turbo'
        
        # Check cache
        cache_key = self.get_cache_key(messages, model)
        messages_tuple = tuple(str(m) for m in messages)
        
        response = self.cached_completion(cache_key, model, messages_tuple)
        
        # Track costs (simplified)
        est_cost = self.estimate_cost(model, 500, 200)  # Approximate
        self.total_cost += est_cost
        
        return response, model, est_cost

# Usage example
llm = CostOptimizedLLM()

# Simple query → routed to GPT-3.5
simple_messages = [{"role": "user", "content": "Summarize this text briefly."}]
# response, model, cost = llm.smart_completion(simple_messages)
# print(f"Routed to: {model}, Cost: ${cost:.6f}")

# Complex query → routed to GPT-4
complex_messages = [{"role": "user", "content": "Analyze and compare these two algorithms step by step, then explain why one is better."}]
# response, model, cost = llm.smart_completion(complex_messages)
# print(f"Routed to: {model}, Cost: ${cost:.6f}")
```

**Interview Tips:**
- Fine-tuned GPT-3.5 can match or exceed base GPT-4 for specific tasks
- Caching is crucial — many queries are repeated or similar
- Consider latency: GPT-4 is also slower (~2-3x)
- Monitor usage: set up alerts for unexpected cost spikes

---

### Question 27
- [ ] Done
**When would you choose fine-tuning GPT-2 locally versus using GPT-4's in-context learning?**

**Answer:**

**Definition:**
Choose **local GPT-2 fine-tuning** for: data privacy requirements, high-volume low-cost inference, offline operation, and specialized domains with labeled data. Choose **GPT-4 in-context learning** for: rapid prototyping, general tasks, complex reasoning, and when you lack fine-tuning data. Fine-tuning requires upfront effort but has zero marginal API cost; in-context learning is flexible but pays per token.

**Decision Matrix:**

| Factor | Fine-tune GPT-2 | GPT-4 In-Context |
|--------|-----------------|------------------|
| **Data privacy** | ✓ Data stays local | ✗ Data sent to API |
| **Setup time** | Hours-days | Minutes |
| **Inference cost** | Fixed (compute) | Per token ($$$) |
| **Task complexity** | Simple-medium | Any complexity |
| **Quality ceiling** | Limited (1.5B params) | State-of-the-art |
| **Labeled data needed** | Yes (100s-1000s) | No (few-shot) |
| **Offline operation** | ✓ Yes | ✗ No |
| **Customization** | Deep | Surface (prompts only) |

**When to Fine-Tune GPT-2 Locally:**
- Strict data privacy (healthcare, finance, legal)
- High volume (millions of inferences/day)
- Specialized narrow task (sentiment, classification)
- Edge deployment (no internet)
- Need deterministic outputs
- Cost constraints at scale

**When to Use GPT-4 In-Context Learning:**
- Rapid prototyping and experimentation
- Complex reasoning, code generation
- General-purpose assistant
- Limited or no training data
- Changing requirements
- Quality is paramount

**Cost Comparison (1M inferences/month):**

| Approach | Setup Cost | Monthly Cost | Total Year 1 |
|----------|------------|--------------|--------------|
| GPT-2 fine-tuned (local) | ~$50 (GPU hours) | ~$100 (hosting) | ~$1,250 |
| GPT-3.5-turbo API | $0 | ~$500-1000 | ~$6,000-12,000 |
| GPT-4 API | $0 | ~$10,000+ | ~$120,000+ |

**Python Code Example:**
```python
# Approach 1: Fine-tune GPT-2 locally
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def finetune_gpt2(train_data_path, output_dir):
    """Fine-tune GPT-2 on custom data"""
    
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and tokenize data
    dataset = load_dataset('text', data_files={'train': train_data_path})
    
    def tokenize(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            max_length=512,
            padding='max_length'
        )
    
    tokenized = dataset.map(tokenize, batched=True)
    
    # Training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=100,
    )
    
    # Train...
    return model

# Inference with fine-tuned model (zero marginal cost)
def local_inference(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0])


# Approach 2: GPT-4 in-context learning
import openai

def gpt4_inference(prompt, examples=None):
    """Few-shot in-context learning with GPT-4"""
    
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    
    # Add few-shot examples if provided
    if examples:
        for ex in examples:
            messages.append({"role": "user", "content": ex['input']})
            messages.append({"role": "assistant", "content": ex['output']})
    
    messages.append({"role": "user", "content": prompt})
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    
    return response.choices[0].message.content


# Decision helper
def choose_approach(requirements):
    """Help decide between fine-tuning and API"""
    
    score = 0
    
    # Factors favoring fine-tuning
    if requirements.get('data_privacy'):
        score += 2
    if requirements.get('volume_per_month', 0) > 100000:
        score += 2
    if requirements.get('offline_needed'):
        score += 3
    if requirements.get('labeled_data_available'):
        score += 1
    if requirements.get('specialized_domain'):
        score += 1
    
    # Factors favoring API
    if requirements.get('complex_reasoning'):
        score -= 2
    if requirements.get('rapid_prototyping'):
        score -= 2
    if requirements.get('quality_critical'):
        score -= 2
    if not requirements.get('labeled_data_available'):
        score -= 1
    
    if score >= 2:
        return "Fine-tune GPT-2 locally"
    elif score <= -2:
        return "Use GPT-4 API"
    else:
        return "Consider hybrid: GPT-3.5 API or fine-tuned larger model"

# Example usage
reqs = {
    'data_privacy': True,
    'volume_per_month': 500000,
    'labeled_data_available': True,
    'specialized_domain': True
}
print(choose_approach(reqs))  # → "Fine-tune GPT-2 locally"
```

**Interview Tips:**
- GPT-2 max quality is limited (1.5B params vs GPT-4's ~1.8T)
- Consider middle ground: fine-tune LLaMA-7B or Mistral-7B locally
- In-context learning doesn't truly "learn" — each call is independent
- Fine-tuning amortizes cost over many inferences

---

### Question 28
- [ ] Done
**What techniques help reduce hallucination in GPT models for factual content generation?**

**Answer:**

**Definition:**
Hallucinations are confident but incorrect statements generated by LLMs. Reduce them through: **Retrieval-Augmented Generation (RAG)** to ground responses in source documents, **explicit citations**, **temperature reduction**, **chain-of-thought prompting**, **self-consistency checks**, and **fine-tuning on factual data**. No technique eliminates hallucinations completely.

**Anti-Hallucination Strategies:**

| Strategy | Effectiveness | Implementation Complexity |
|----------|--------------|--------------------------|
| **RAG (retrieval)** | High | Medium |
| **Lower temperature** | Medium | Low |
| **Citation requirements** | Medium | Low |
| **Chain-of-thought** | Medium | Low |
| **Self-consistency** | High | Medium |
| **Fine-tuning on facts** | High | High |
| **Fact verification layer** | High | High |
| **Constrained decoding** | Medium | Medium |

**How Each Strategy Works:**

**1. RAG (Retrieval-Augmented Generation):**
```
Query → Retrieve relevant documents → Include in context → Generate with sources
```
- Grounds responses in actual documents
- Model can cite sources
- Reduces but doesn't eliminate hallucination

**2. Temperature Reduction:**
- Lower temperature (0.0-0.3) → more deterministic
- Higher temperature → more creative/risky
- For factual tasks, use temperature=0

**3. Chain-of-Thought Prompting:**
```
"Think step by step. First, identify the key facts. 
Then, verify each claim. Finally, provide your answer with reasoning."
```

**4. Self-Consistency:**
- Generate multiple responses
- Check for agreement
- Flag inconsistencies for review

**Python Code Example:**
```python
import openai
from typing import List, Dict

class AntiHallucinationLLM:
    def __init__(self):
        self.client = openai
    
    def basic_query(self, prompt, temperature=0.0):
        """Low temperature for factual queries"""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature  # Low for facts
        )
        return response.choices[0].message.content
    
    def rag_query(self, query, retrieved_docs: List[str]):
        """Retrieval-Augmented Generation"""
        context = "\n\n".join([f"Document {i+1}:\n{doc}" 
                               for i, doc in enumerate(retrieved_docs)])
        
        prompt = f"""Answer the question based ONLY on the provided documents.
If the documents don't contain the answer, say "I don't have information about this."
Always cite which document(s) you used.

Documents:
{context}

Question: {query}

Answer with citations:"""
        
        return self.basic_query(prompt, temperature=0)
    
    def chain_of_thought(self, query):
        """Force step-by-step reasoning"""
        prompt = f"""Question: {query}

Let's approach this step-by-step:
1. First, identify what facts I know with certainty
2. Then, identify what I'm uncertain about
3. Finally, provide my answer, clearly marking any uncertainties

Step-by-step analysis:"""
        
        return self.basic_query(prompt, temperature=0)
    
    def self_consistency_check(self, query, n_samples=3):
        """Generate multiple responses and check consistency"""
        responses = []
        
        for _ in range(n_samples):
            response = self.basic_query(query, temperature=0.7)
            responses.append(response)
        
        # Check for consistency
        consistency_prompt = f"""Given these {n_samples} responses to the same question,
identify any contradictions or inconsistencies:

Question: {query}

Responses:
{chr(10).join([f'{i+1}. {r}' for i, r in enumerate(responses)])}

Are there contradictions? If so, what are they?
Which response seems most reliable and why?"""
        
        analysis = self.basic_query(consistency_prompt, temperature=0)
        
        return {
            'responses': responses,
            'consistency_analysis': analysis
        }
    
    def with_citations(self, query, sources: List[Dict]):
        """Require citations for every claim"""
        source_list = "\n".join([f"[{s['id']}] {s['title']}: {s['content']}" 
                                  for s in sources])
        
        prompt = f"""Answer the question using ONLY the provided sources.
Every factual claim must have a citation in [brackets].
If you cannot cite a source, do not make the claim.

Sources:
{source_list}

Question: {query}

Answer (with citations for every fact):"""
        
        return self.basic_query(prompt, temperature=0)


# Example usage
llm = AntiHallucinationLLM()

# RAG example
docs = [
    "The Eiffel Tower was completed in 1889 and stands at 330 meters.",
    "The Eiffel Tower is located in Paris, France on the Champ de Mars."
]
# response = llm.rag_query("When was the Eiffel Tower built?", docs)

# Self-consistency check
# result = llm.self_consistency_check("What is the population of Tokyo?")
# print("Consistent responses:", result['consistency_analysis'])


# Simple prompt engineering for anti-hallucination
def anti_hallucination_prompt(query):
    """Prompt template that reduces hallucination"""
    return f"""You are a factual assistant. Follow these rules strictly:
1. Only state facts you are highly confident about
2. If uncertain, say "I'm not certain, but..."
3. If you don't know, say "I don't have reliable information about this"
4. Never make up statistics, dates, or names
5. Distinguish between well-known facts and your inferences

Question: {query}

Factual answer:"""
```

**Interview Tips:**
- RAG is the most effective production technique
- Temperature=0 doesn't guarantee correctness, just consistency
- "I don't know" is better than confident hallucination
- Always verify critical facts with external sources
- Fine-tuning on factual QA datasets (with refusals) helps

---

### Question 29
- [ ] Done
**How do you handle rate limiting, API quotas, and error recovery in GPT production systems?**

**Answer:**

**Definition:**
Production LLM systems must handle API constraints gracefully. Implement **exponential backoff** for rate limits (429 errors), **request queuing** to smooth traffic, **caching** to reduce redundant calls, **circuit breakers** to fail gracefully, and **fallback models** when primary API is unavailable. Monitor usage against quotas and set up alerts.

**Key Error Types and Handling:**

| Error Code | Meaning | Handling Strategy |
|------------|---------|-------------------|
| 429 | Rate limit exceeded | Exponential backoff, queue requests |
| 500 | Server error | Retry with backoff |
| 503 | Service unavailable | Circuit breaker, fallback |
| 400 | Bad request | Log and fix prompt |
| 401 | Auth error | Check API key |
| Timeout | Request too slow | Retry, consider shorter prompts |

**Production Architecture:**

```
Requests → Rate Limiter → Request Queue → API Client → Response
                ↓              ↓              ↓
            Reject if      Backpressure   Retry logic
            over limit                    + Circuit breaker
                                               ↓
                                          Fallback model
                                               ↓
                                            Cache
```

**Python Code Example:**
```python
import time
import random
from functools import wraps
from collections import deque
import threading

class RateLimiter:
    """Token bucket rate limiter"""
    def __init__(self, requests_per_minute=60):
        self.rpm = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def acquire(self):
        with self.lock:
            now = time.time()
            # Refill tokens
            elapsed = now - self.last_update
            self.tokens = min(self.rpm, self.tokens + elapsed * (self.rpm / 60))
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False
    
    def wait_for_token(self):
        while not self.acquire():
            time.sleep(0.1)


class CircuitBreaker:
    """Circuit breaker pattern"""
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
    
    def record_success(self):
        self.failure_count = 0
        self.state = 'closed'
    
    def can_proceed(self):
        if self.state == 'closed':
            return True
        
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'half-open'
                return True
            return False
        
        return True  # half-open: allow one request


def exponential_backoff(max_retries=5, base_delay=1, max_delay=60):
    """Decorator for exponential backoff retry"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_code = getattr(e, 'status_code', None)
                    
                    # Don't retry on client errors (except 429)
                    if error_code and 400 <= error_code < 500 and error_code != 429:
                        raise
                    
                    retries += 1
                    if retries >= max_retries:
                        raise
                    
                    # Calculate delay with jitter
                    delay = min(base_delay * (2 ** retries), max_delay)
                    jitter = random.uniform(0, delay * 0.1)
                    
                    print(f"Retry {retries}/{max_retries} after {delay:.1f}s")
                    time.sleep(delay + jitter)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


class RobustLLMClient:
    """Production-ready LLM client"""
    
    def __init__(self, primary_model="gpt-4", fallback_model="gpt-3.5-turbo"):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.rate_limiter = RateLimiter(requests_per_minute=60)
        self.circuit_breaker = CircuitBreaker()
        self.cache = {}
    
    def _cache_key(self, messages, model):
        import hashlib
        content = str(messages) + model
        return hashlib.md5(content.encode()).hexdigest()
    
    @exponential_backoff(max_retries=5)
    def _call_api(self, model, messages):
        """Actual API call with retry logic"""
        import openai
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            timeout=30
        )
        return response.choices[0].message.content
    
    def complete(self, messages, use_cache=True):
        """Main completion method with all protections"""
        
        # Check cache
        cache_key = self._cache_key(messages, self.primary_model)
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Rate limiting
        self.rate_limiter.wait_for_token()
        
        # Circuit breaker
        if not self.circuit_breaker.can_proceed():
            print("Circuit open, using fallback")
            return self._use_fallback(messages)
        
        try:
            response = self._call_api(self.primary_model, messages)
            self.circuit_breaker.record_success()
            
            # Cache successful response
            if use_cache:
                self.cache[cache_key] = response
            
            return response
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            print(f"Primary failed: {e}, trying fallback")
            return self._use_fallback(messages)
    
    def _use_fallback(self, messages):
        """Fallback to secondary model"""
        try:
            return self._call_api(self.fallback_model, messages)
        except Exception as e:
            return f"Service temporarily unavailable: {str(e)}"


# Usage
client = RobustLLMClient()

# This handles rate limits, retries, circuit breaking, caching
# response = client.complete([{"role": "user", "content": "Hello!"}])
```

**Interview Tips:**
- Always implement exponential backoff with jitter (prevents thundering herd)
- Cache at prompt level, not just response level
- Circuit breaker prevents cascade failures
- Monitor: track error rates, latencies, token usage, costs
- Set up alerts for quota approaching limits

---

### Question 30
- [ ] Done
**What strategies help control the creativity vs consistency trade-off using temperature and top-p?**

**Answer:**

**Definition:**
**Temperature** controls randomness by scaling logits before softmax (higher = more random). **Top-p (nucleus sampling)** samples from the smallest set of tokens whose cumulative probability exceeds p. Use low temperature/top-p for factual, consistent outputs; higher values for creative, diverse content. They can be used together or independently.

**How They Work:**

**Temperature:**
$$P(token_i) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| 0 | Deterministic (argmax) | Factual QA, code |
| 0.3-0.5 | Low randomness | Summarization, translation |
| 0.7-0.9 | Balanced | General chat |
| 1.0+ | High randomness | Creative writing, brainstorming |

**Top-p (Nucleus Sampling):**
- Sort tokens by probability
- Keep smallest set where sum ≥ p
- Sample only from this set

| Top-p | Effect | Use Case |
|-------|--------|----------|
| 0.1-0.3 | Very focused | Factual, specific |
| 0.5-0.7 | Moderately focused | Balanced |
| 0.9-1.0 | Diverse | Creative, varied |

**Combining Temperature and Top-p:**

| Setting | Temperature | Top-p | Result |
|---------|-------------|-------|--------|
| Maximum consistency | 0 | - | Deterministic |
| Factual | 0.2 | 0.3 | Very consistent |
| Balanced | 0.7 | 0.9 | Natural, varied |
| Creative | 1.0 | 0.95 | Diverse, surprising |
| Maximum creativity | 1.5 | 1.0 | Very random |

**Intuition:**
- **Temperature** = how "sharp" the probability distribution is
- **Top-p** = how many tokens are considered

```
Tokens:    [A]    [B]    [C]    [D]    [E]
Original:  0.40   0.30   0.15   0.10   0.05

T=0.5:     0.55   0.30   0.10   0.04   0.01  (sharper)
T=1.5:     0.28   0.25   0.20   0.15   0.12  (flatter)

Top-p=0.7: Only [A, B] considered (0.40+0.30=0.70)
Top-p=0.95: [A, B, C, D] considered
```

**Python Code Example:**
```python
import numpy as np

def apply_temperature(logits, temperature):
    """Apply temperature scaling to logits"""
    if temperature == 0:
        # Argmax (deterministic)
        result = np.zeros_like(logits)
        result[np.argmax(logits)] = 1.0
        return result
    
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))  # Numerical stability
    return exp_scaled / exp_scaled.sum()

def top_p_sampling(probs, p):
    """Top-p (nucleus) sampling"""
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    cumulative = np.cumsum(sorted_probs)
    cutoff_idx = np.searchsorted(cumulative, p) + 1
    
    # Zero out tokens beyond cutoff
    result = np.zeros_like(probs)
    result[sorted_indices[:cutoff_idx]] = probs[sorted_indices[:cutoff_idx]]
    
    # Renormalize
    return result / result.sum()

def sample_with_params(logits, temperature=1.0, top_p=1.0):
    """Complete sampling with temperature and top-p"""
    # Apply temperature
    probs = apply_temperature(logits, temperature)
    
    # Apply top-p
    if top_p < 1.0:
        probs = top_p_sampling(probs, top_p)
    
    # Sample
    token_idx = np.random.choice(len(probs), p=probs)
    return token_idx

# Demonstration
np.random.seed(42)
logits = np.array([2.0, 1.5, 1.0, 0.5, 0.1])
tokens = ['excellent', 'good', 'okay', 'fine', 'meh']

print("Original logits:", logits)

# Different settings
settings = [
    {'temperature': 0, 'top_p': 1.0, 'desc': 'Deterministic'},
    {'temperature': 0.3, 'top_p': 0.5, 'desc': 'Factual'},
    {'temperature': 0.7, 'top_p': 0.9, 'desc': 'Balanced'},
    {'temperature': 1.2, 'top_p': 0.95, 'desc': 'Creative'},
]

for s in settings:
    probs = apply_temperature(logits, s['temperature'])
    if s['top_p'] < 1.0:
        probs = top_p_sampling(probs, s['top_p'])
    print(f"\n{s['desc']} (T={s['temperature']}, top_p={s['top_p']}):")
    for t, p in zip(tokens, probs):
        if p > 0.01:
            print(f"  {t}: {p:.3f}")


# Practical usage with OpenAI
def get_completion(prompt, creativity_level='balanced'):
    """Get completion with appropriate settings"""
    import openai
    
    settings = {
        'deterministic': {'temperature': 0, 'top_p': 1},
        'factual': {'temperature': 0.2, 'top_p': 0.3},
        'balanced': {'temperature': 0.7, 'top_p': 0.9},
        'creative': {'temperature': 1.0, 'top_p': 0.95},
    }
    
    params = settings.get(creativity_level, settings['balanced'])
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=params['temperature'],
        top_p=params['top_p']
    )
    
    return response.choices[0].message.content

# Task-specific recommendations
task_settings = {
    'code_generation': {'temperature': 0, 'top_p': 1.0},
    'summarization': {'temperature': 0.3, 'top_p': 0.5},
    'translation': {'temperature': 0.3, 'top_p': 0.5},
    'chatbot': {'temperature': 0.7, 'top_p': 0.9},
    'creative_writing': {'temperature': 1.0, 'top_p': 0.95},
    'brainstorming': {'temperature': 1.2, 'top_p': 0.95},
}
```

**Interview Tips:**
- Use temperature=0 for deterministic outputs (code, facts)
- Top-p is often preferred over top-k (adapts to distribution shape)
- Don't use both top-p<1 AND top-k together (redundant)
- OpenAI recommends changing either temperature OR top-p, not both
- Test different settings for your specific use case

---

### Question 31
- [ ] Done
**How do you implement effective context management for multi-turn conversations with GPT?**

**Answer:**

**Definition:**
Context management involves maintaining conversation history within the token limit while preserving important information. Strategies include: **sliding window** (keep recent N turns), **summarization** (compress old context), **hierarchical memory** (short-term + long-term), and **relevance-based retrieval** (fetch relevant past turns). Balance context length against cost and coherence.

**Context Management Challenges:**

| Challenge | Impact | Solution |
|-----------|--------|----------|
| Token limit | Can't fit all history | Truncation/summarization |
| Cost | Longer context = higher cost | Efficient compression |
| Relevance | Old context may be irrelevant | Selective retrieval |
| Coherence | Losing context breaks flow | Smart summarization |

**Strategies:**

**1. Sliding Window:**
- Keep last N turns
- Simple, predictable
- May lose important early context

**2. Summarization:**
- Periodically summarize old turns
- Preserve key information
- Adds latency for summary generation

**3. Hierarchical Memory:**
- Recent: Full messages
- Mid-term: Summaries
- Long-term: Key facts only

**4. Retrieval-Based:**
- Store all turns in vector DB
- Retrieve relevant past turns based on current query
- Best for long conversations with topic shifts

**Python Code Example:**
```python
from typing import List, Dict
import tiktoken

class ConversationManager:
    def __init__(self, model="gpt-4", max_tokens=8000, reserved_for_response=1000):
        self.model = model
        self.max_context_tokens = max_tokens - reserved_for_response
        self.encoder = tiktoken.encoding_for_model(model)
        
        self.messages = []
        self.summary = ""
        self.system_prompt = "You are a helpful assistant."
    
    def count_tokens(self, text):
        return len(self.encoder.encode(text))
    
    def count_messages_tokens(self, messages):
        total = 0
        for msg in messages:
            total += self.count_tokens(msg['content']) + 4  # role tokens
        return total
    
    def add_message(self, role, content):
        """Add a message to conversation"""
        self.messages.append({'role': role, 'content': content})
        self._manage_context()
    
    def _manage_context(self):
        """Ensure we stay within token limits"""
        current_tokens = self.count_messages_tokens(self.messages)
        
        if current_tokens > self.max_context_tokens:
            self._compress_context()
    
    def _compress_context(self):
        """Compress old context when approaching limit"""
        # Strategy: Keep system + summary + last N messages
        
        # First, summarize old messages
        if len(self.messages) > 6:
            old_messages = self.messages[:-4]
            self._update_summary(old_messages)
            self.messages = self.messages[-4:]  # Keep last 4
    
    def _update_summary(self, old_messages):
        """Summarize old messages"""
        old_text = "\n".join([f"{m['role']}: {m['content']}" 
                              for m in old_messages])
        
        # In practice, call LLM to summarize
        summary_prompt = f"""Summarize the key points from this conversation:
{old_text}

Summary (keep it brief, focus on key facts and decisions):"""
        
        # Simulated summary (in practice, call API)
        self.summary = f"Previous discussion covered: {old_text[:200]}..."
    
    def get_context(self):
        """Build context for API call"""
        context = [{'role': 'system', 'content': self.system_prompt}]
        
        if self.summary:
            context.append({
                'role': 'system',
                'content': f"Summary of earlier conversation: {self.summary}"
            })
        
        context.extend(self.messages)
        return context


class RetrievalConversationManager:
    """Retrieval-based context management for long conversations"""
    
    def __init__(self, max_context_turns=10):
        self.all_messages = []  # Full history
        self.max_context_turns = max_context_turns
    
    def add_message(self, role, content, embedding=None):
        """Add message with optional embedding for retrieval"""
        self.all_messages.append({
            'role': role,
            'content': content,
            'embedding': embedding,
            'turn': len(self.all_messages)
        })
    
    def get_relevant_context(self, current_query, top_k=5):
        """Retrieve relevant past messages"""
        # In practice, use vector similarity
        # Here, simple keyword matching as demo
        
        query_words = set(current_query.lower().split())
        
        scored_messages = []
        for msg in self.all_messages[:-self.max_context_turns]:
            msg_words = set(msg['content'].lower().split())
            overlap = len(query_words & msg_words)
            scored_messages.append((overlap, msg))
        
        # Sort by relevance
        scored_messages.sort(reverse=True, key=lambda x: x[0])
        
        # Get top-k relevant
        relevant = [msg for score, msg in scored_messages[:top_k] if score > 0]
        
        return relevant
    
    def get_context(self, current_query):
        """Build context with recent + relevant messages"""
        # Recent messages
        recent = self.all_messages[-self.max_context_turns:]
        
        # Relevant older messages
        relevant = self.get_relevant_context(current_query)
        
        # Combine, avoiding duplicates
        recent_turns = {m['turn'] for m in recent}
        relevant_unique = [m for m in relevant if m['turn'] not in recent_turns]
        
        context = []
        
        if relevant_unique:
            context.append({
                'role': 'system',
                'content': 'Relevant earlier context:\n' + 
                          '\n'.join([f"{m['role']}: {m['content']}" for m in relevant_unique])
            })
        
        context.extend([{'role': m['role'], 'content': m['content']} for m in recent])
        
        return context


# Usage example
conv = ConversationManager(max_tokens=4000)

# Simulate conversation
conv.add_message('user', "Hi, I want to plan a trip to Japan.")
conv.add_message('assistant', "Great! When are you planning to visit?")
conv.add_message('user', "Next April, during cherry blossom season.")
conv.add_message('assistant', "Perfect timing! Sakura season is beautiful...")
# ... more turns

context = conv.get_context()
print(f"Context has {len(context)} messages")
```

**Interview Tips:**
- Token counting is essential — use tiktoken for accurate counts
- Summarization adds latency but preserves more information
- For customer support, keep the initial issue description always in context
- Consider topic detection to decide when context can be safely cleared

---

### Question 32
- [ ] Done
**When should you use GPT-4's function calling capabilities versus traditional API integrations?**

**Answer:**

**Definition:**
Use **function calling** when the LLM needs to decide which action to take based on natural language, handle complex parameter extraction, or chain multiple tools dynamically. Use **traditional API integration** when actions are predetermined, parameters are explicit, or you need maximum reliability. Function calling excels at natural language → structured action translation.

**Comparison:**

| Aspect | Function Calling | Traditional API |
|--------|-----------------|-----------------|
| **Decision maker** | LLM decides action | Code decides action |
| **Flexibility** | High (any user phrasing) | Low (predefined flows) |
| **Reliability** | Good but not 100% | Deterministic |
| **Complexity** | LLM handles | Developer handles |
| **Latency** | Higher (LLM inference) | Lower (direct calls) |
| **Cost** | Higher (tokens for function schema) | Lower |

**When to Use Function Calling:**

| Scenario | Why Function Calling |
|----------|---------------------|
| Chatbot with tools | User intent varies, LLM picks tool |
| Complex parameter extraction | "Book a flight from NYC to LA next Tuesday" |
| Multi-step reasoning | LLM chains function calls |
| Ambiguous requests | LLM can ask clarifying questions |
| Natural language interface | Any phrasing works |

**When to Use Traditional Integration:**

| Scenario | Why Traditional |
|----------|-----------------|
| Single known action | No decision needed |
| Form-based input | Parameters already structured |
| High reliability required | Can't risk wrong function call |
| Cost/latency sensitive | Avoid LLM overhead |
| Simple workflows | Overkill for simple cases |

**Python Code Example:**
```python
import openai
import json

# Define functions for GPT
functions = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g., 'New York, NY'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "fahrenheit"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "book_flight",
        "description": "Book a flight between two cities",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Departure city"},
                "destination": {"type": "string", "description": "Arrival city"},
                "date": {"type": "string", "description": "Flight date (YYYY-MM-DD)"},
                "passengers": {"type": "integer", "default": 1}
            },
            "required": ["origin", "destination", "date"]
        }
    },
    {
        "name": "search_hotels",
        "description": "Search for hotels in a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "check_in": {"type": "string"},
                "check_out": {"type": "string"},
                "guests": {"type": "integer", "default": 1}
            },
            "required": ["city", "check_in", "check_out"]
        }
    }
]

def call_with_functions(user_message):
    """Use GPT to determine and call appropriate function"""
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a travel assistant."},
            {"role": "user", "content": user_message}
        ],
        functions=functions,
        function_call="auto"  # Let model decide
    )
    
    message = response.choices[0].message
    
    if message.function_call:
        # Model wants to call a function
        func_name = message.function_call.name
        func_args = json.loads(message.function_call.arguments)
        
        return {
            'action': 'function_call',
            'function': func_name,
            'arguments': func_args
        }
    else:
        # Model responded directly
        return {
            'action': 'response',
            'content': message.content
        }

# Example usage
queries = [
    "What's the weather in Tokyo?",
    "I need to fly from LA to Miami on January 15th",
    "Find me a hotel in Paris for next weekend",
    "Hello, how are you?"  # No function call needed
]

# for q in queries:
#     result = call_with_functions(q)
#     print(f"Query: {q}")
#     print(f"Result: {result}\n")


# Traditional approach for comparison
def traditional_api_call(action, params):
    """Traditional API - predetermined action"""
    
    if action == "weather":
        # Direct API call, no LLM needed
        return get_weather_api(params['location'])
    
    elif action == "book_flight":
        return book_flight_api(
            params['origin'],
            params['destination'],
            params['date']
        )
    
    # etc.


# Decision helper: When to use each approach
def should_use_function_calling(requirements):
    """Decide between function calling and traditional"""
    
    # Favor function calling
    if requirements.get('natural_language_input'):
        return True
    if requirements.get('uncertain_user_intent'):
        return True
    if requirements.get('complex_parameter_extraction'):
        return True
    
    # Favor traditional
    if requirements.get('structured_input'):
        return False
    if requirements.get('high_reliability_required'):
        return False
    if requirements.get('cost_sensitive'):
        return False
    
    return True  # Default to function calling for flexibility

# Example
print(should_use_function_calling({
    'natural_language_input': True,
    'uncertain_user_intent': True
}))  # True - use function calling

print(should_use_function_calling({
    'structured_input': True,
    'high_reliability_required': True
}))  # False - use traditional
```

**Interview Tips:**
- Function calling = structured output generation by GPT
- Great for building agents that need to take actions
- Always validate function arguments before executing
- Consider fallback if LLM picks wrong function
- Newer: GPT-4 Turbo supports parallel function calls

---

### Question 33
- [ ] Done
**How do you optimize GPT's context window utilization for long-document summarization?**

**Answer:**

**Definition:**
For documents exceeding context limits, use **hierarchical summarization** (summarize chunks, then summarize summaries), **map-reduce** pattern, or **iterative refinement**. Within context limits, optimize by placing key content strategically, using concise prompts, and leveraging the model's attention patterns (beginning and end get more focus).

**Strategies for Long Documents:**

| Strategy | When to Use | Quality | Latency |
|----------|-------------|---------|---------|
| **Truncation** | Document fits after cutting | Low | Fast |
| **Map-Reduce** | Parallel chunk processing | Medium-High | Medium |
| **Hierarchical** | Very long documents | High | Slow |
| **Iterative refinement** | Quality critical | Highest | Slowest |
| **Stuff** | Document fits in context | Highest | Fast |

**Method Comparison:**

**1. Stuff (Simple):**
```
[Full document] → LLM → Summary
Best for: Documents within context limit
```

**2. Map-Reduce:**
```
Document → [Chunk 1] → Summary 1 ─┐
          [Chunk 2] → Summary 2 ──┼→ Combine → Final Summary
          [Chunk 3] → Summary 3 ─┘
Parallel processing, good for long docs
```

**3. Hierarchical:**
```
Level 0: [Chunks] → [Summaries]
Level 1: [Summaries] → [Meta-summaries]
Level 2: [Meta-summaries] → Final Summary
Good for very long documents (books, legal docs)
```

**4. Iterative Refinement:**
```
[Chunk 1] → Summary v1
[Summary v1 + Chunk 2] → Summary v2
[Summary v2 + Chunk 3] → Summary v3 (Final)
Best quality, maintains coherence
```

**Python Code Example:**
```python
import tiktoken
from typing import List

class DocumentSummarizer:
    def __init__(self, model="gpt-4", max_context=8000):
        self.model = model
        self.max_context = max_context
        self.encoder = tiktoken.encoding_for_model(model)
        self.prompt_reserve = 500  # Reserve for prompt
        self.output_reserve = 1000  # Reserve for output
    
    def count_tokens(self, text):
        return len(self.encoder.encode(text))
    
    def chunk_document(self, text, chunk_size=None):
        """Split document into chunks"""
        if chunk_size is None:
            chunk_size = self.max_context - self.prompt_reserve - self.output_reserve
        
        tokens = self.encoder.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def summarize_chunk(self, chunk, context=""):
        """Summarize a single chunk"""
        prompt = f"""Summarize the following text concisely, capturing key points.
{f"Context from previous sections: {context}" if context else ""}

Text:
{chunk}

Summary:"""
        
        # API call here
        # response = openai.chat.completions.create(...)
        return f"[Summary of chunk]"  # Placeholder
    
    def stuff_summarize(self, document):
        """Simple: put whole document in context"""
        if self.count_tokens(document) > self.max_context - self.prompt_reserve:
            raise ValueError("Document too long for stuff method")
        
        return self.summarize_chunk(document)
    
    def map_reduce_summarize(self, document, chunk_size=3000):
        """Map-reduce: summarize chunks, then combine"""
        chunks = self.chunk_document(document, chunk_size)
        
        # Map: Summarize each chunk
        summaries = []
        for chunk in chunks:
            summary = self.summarize_chunk(chunk)
            summaries.append(summary)
        
        # Reduce: Combine summaries
        combined = "\n\n".join(summaries)
        
        if self.count_tokens(combined) < self.max_context - self.prompt_reserve:
            # Fits in one final call
            final_summary = self.summarize_chunk(
                combined,
                context="These are summaries of different sections."
            )
        else:
            # Recursive map-reduce
            final_summary = self.map_reduce_summarize(combined)
        
        return final_summary
    
    def refine_summarize(self, document, chunk_size=3000):
        """Iterative refinement: refine summary with each chunk"""
        chunks = self.chunk_document(document, chunk_size)
        
        if not chunks:
            return ""
        
        # Initial summary from first chunk
        current_summary = self.summarize_chunk(chunks[0])
        
        # Refine with subsequent chunks
        for chunk in chunks[1:]:
            refine_prompt = f"""Given the existing summary and new content, 
create an updated comprehensive summary.

Existing Summary:
{current_summary}

New Content:
{chunk}

Updated Summary:"""
            
            # current_summary = call_api(refine_prompt)
            current_summary = f"[Refined summary]"  # Placeholder
        
        return current_summary
    
    def hierarchical_summarize(self, document, levels=2):
        """Hierarchical: multiple levels of summarization"""
        chunks = self.chunk_document(document)
        
        current_level = chunks
        
        for level in range(levels):
            next_level = []
            
            # Group into batches
            batch_size = 4
            for i in range(0, len(current_level), batch_size):
                batch = current_level[i:i + batch_size]
                combined = "\n\n---\n\n".join(batch)
                summary = self.summarize_chunk(combined)
                next_level.append(summary)
            
            current_level = next_level
            
            # If small enough, stop
            if len(current_level) <= 2:
                break
        
        # Final summary
        final_text = "\n\n".join(current_level)
        return self.summarize_chunk(final_text)
    
    def summarize(self, document, method='auto'):
        """Main summarization method"""
        doc_tokens = self.count_tokens(document)
        available = self.max_context - self.prompt_reserve - self.output_reserve
        
        if method == 'auto':
            if doc_tokens <= available:
                method = 'stuff'
            elif doc_tokens <= available * 5:
                method = 'map_reduce'
            else:
                method = 'hierarchical'
        
        methods = {
            'stuff': self.stuff_summarize,
            'map_reduce': self.map_reduce_summarize,
            'refine': self.refine_summarize,
            'hierarchical': self.hierarchical_summarize
        }
        
        return methods[method](document)


# Usage
summarizer = DocumentSummarizer()

long_document = "..." * 10000  # Very long document
# summary = summarizer.summarize(long_document, method='auto')
```

**Interview Tips:**
- Map-reduce is most common for batch processing
- Refine produces most coherent summaries but is sequential
- For very long docs (100K+ tokens), hierarchical is necessary
- Consider importance-based chunking (chapters, sections) over token-based

---

## Open Source LLMs (LLaMA, Falcon, Mistral)

### Question 34
- [ ] Done
**How do you choose between LLaMA, Falcon, and Mistral based on deployment constraints and licensing?**

**Answer:**

**Definition:**
Choose based on: **license requirements** (commercial use), **hardware constraints** (memory, GPU), **task requirements** (quality vs speed), and **ecosystem support**. LLaMA 2 is permissive for most commercial uses. Falcon is fully open (Apache 2.0). Mistral offers best performance-per-parameter with sliding window attention for long contexts.

**Model Comparison:**

| Aspect | LLaMA 2 | Falcon | Mistral |
|--------|---------|--------|---------|
| **License** | Meta Community (commercial OK if <700M users) | Apache 2.0 | Apache 2.0 |
| **Sizes** | 7B, 13B, 70B | 7B, 40B, 180B | 7B, 8x7B (MoE) |
| **Quality** | Excellent | Good | Best at 7B |
| **Context** | 4K | 2K | 32K (sliding window) |
| **Training data** | 2T tokens | 1.5T tokens | Undisclosed |
| **Fine-tuning** | Easy, good ecosystem | Good | Excellent |

**Decision Matrix:**

| Requirement | Best Choice | Reason |
|-------------|-------------|--------|
| **Maximum quality (7B)** | Mistral 7B | Outperforms LLaMA 7B |
| **Maximum quality (large)** | LLaMA 70B | Best open-source large model |
| **Fully open license** | Falcon or Mistral | Apache 2.0, no restrictions |
| **Long context needed** | Mistral | 32K with sliding window |
| **Limited memory (<16GB)** | Mistral 7B or LLaMA 7B | Smallest high-quality options |
| **Enterprise deployment** | LLaMA 2 or Mistral | Best support, documentation |
| **MoE efficiency** | Mixtral 8x7B | 47B params, 13B active |

**Hardware Requirements:**

| Model | FP16 Memory | INT8 Memory | INT4 Memory |
|-------|-------------|-------------|-------------|
| 7B | ~14GB | ~7GB | ~4GB |
| 13B | ~26GB | ~13GB | ~7GB |
| 70B | ~140GB | ~70GB | ~35GB |

**Licensing Deep Dive:**

| License | LLaMA 2 | Falcon | Mistral |
|---------|---------|--------|---------|
| **Commercial use** | Yes (with limits) | Yes | Yes |
| **Derivative works** | Yes | Yes | Yes |
| **>700M users** | Requires Meta license | Allowed | Allowed |
| **Attribution** | Required | Required | Required |
| **Liability** | Limited | None | None |

**Python Code Example:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_for_constraints(
    max_memory_gb,
    need_long_context=False,
    commercial_use=True,
    prefer_moe=False
):
    """Select and load appropriate model based on constraints"""
    
    # Model selection logic
    if need_long_context:
        model_choice = "mistralai/Mistral-7B-v0.1"  # 32K context
    elif prefer_moe and max_memory_gb >= 90:
        model_choice = "mistralai/Mixtral-8x7B-v0.1"  # MoE
    elif max_memory_gb >= 35:
        model_choice = "meta-llama/Llama-2-70b-hf"  # INT4
    elif max_memory_gb >= 14:
        model_choice = "meta-llama/Llama-2-13b-hf"  # FP16
    else:
        # Best 7B model
        model_choice = "mistralai/Mistral-7B-v0.1"
    
    # Determine quantization
    if max_memory_gb < 8:
        load_in_4bit = True
        load_in_8bit = False
    elif max_memory_gb < 16:
        load_in_4bit = False
        load_in_8bit = True
    else:
        load_in_4bit = False
        load_in_8bit = False
    
    print(f"Selected: {model_choice}")
    print(f"Quantization: {'4-bit' if load_in_4bit else '8-bit' if load_in_8bit else 'FP16'}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_choice)
    
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_choice,
            quantization_config=bnb_config,
            device_map="auto"
        )
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_choice,
            load_in_8bit=True,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_choice,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    return model, tokenizer

# Decision helper
def recommend_model(requirements):
    """Recommend model based on requirements"""
    
    recommendations = []
    
    # Check constraints
    if requirements.get('license') == 'apache_only':
        recommendations.append(("Mistral", "Apache 2.0, no restrictions"))
        recommendations.append(("Falcon", "Apache 2.0, fully open"))
    
    if requirements.get('context_length', 4096) > 4096:
        recommendations.insert(0, ("Mistral", "32K context with sliding window"))
    
    if requirements.get('max_gpu_memory_gb', 80) < 16:
        recommendations.append(("Mistral 7B + 4-bit", "~4GB memory"))
    
    if requirements.get('best_quality'):
        recommendations.insert(0, ("LLaMA 70B", "Best overall quality"))
    
    return recommendations

# Example usage
reqs = {
    'license': 'apache_only',
    'context_length': 16000,
    'max_gpu_memory_gb': 24,
    'commercial_use': True
}
print(recommend_model(reqs))
# [("Mistral", "32K context..."), ("Mistral", "Apache 2.0..."), ...]
```

**Interview Tips:**
- Mistral 7B outperforms LLaMA 2 13B on most benchmarks
- Mixtral 8x7B = MoE with 47B total params but 13B active per inference
- LLaMA 2 license excludes use in products with >700M monthly users
- For production: consider vLLM or TensorRT-LLM for serving

---

### Question 35
- [ ] Done
**What are the memory optimization techniques for deploying LLaMA models on consumer hardware?**

**Answer:**

**Definition:**
Deploy large LLMs on consumer GPUs (8-24GB) through: **quantization** (INT8/INT4 reduces memory 2-4x), **model sharding** (split across GPUs/CPU), **KV-cache optimization**, **gradient checkpointing** (for fine-tuning), and **efficient inference engines** (llama.cpp, GGUF format). A 7B model in INT4 fits on 6GB VRAM.

**Memory Requirements:**

| Model Size | FP32 | FP16 | INT8 | INT4 |
|------------|------|------|------|------|
| 7B | 28GB | 14GB | 7GB | 4GB |
| 13B | 52GB | 26GB | 13GB | 7GB |
| 70B | 280GB | 140GB | 70GB | 35GB |

**Optimization Techniques:**

| Technique | Memory Reduction | Quality Impact | Speed Impact |
|-----------|-----------------|----------------|--------------|
| FP16 | 2x | None | Faster |
| INT8 | 4x | Minimal | Similar |
| INT4 (GPTQ/AWQ) | 8x | Small | Similar-faster |
| CPU offload | Infinite | None | Much slower |
| Flash Attention | 10-20x for attention | None | Faster |
| KV-cache quantization | 2-4x | Minimal | Similar |

**Quantization Methods:**

| Method | Description | Best For |
|--------|-------------|----------|
| **bitsandbytes** | Dynamic INT8/INT4 | Quick setup |
| **GPTQ** | Calibrated INT4 | Best quality |
| **AWQ** | Activation-aware INT4 | Fast inference |
| **GGUF** | CPU-optimized format | llama.cpp |
| **ExLlama** | Optimized GPTQ | Maximum speed |

**Python Code Example:**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Method 1: bitsandbytes INT4 quantization
def load_model_4bit(model_name):
    """Load model with 4-bit quantization using bitsandbytes"""
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,  # Nested quantization
        bnb_4bit_quant_type="nf4",        # NormalFloat4
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    return model

# Method 2: CPU offloading
def load_with_cpu_offload(model_name, max_gpu_memory="6GB"):
    """Offload layers to CPU when GPU memory insufficient"""
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={
            0: max_gpu_memory,   # GPU 0
            "cpu": "32GB"        # CPU RAM
        }
    )
    
    return model

# Method 3: GPTQ quantized model
def load_gptq_model(model_name):
    """Load pre-quantized GPTQ model"""
    from auto_gptq import AutoGPTQForCausalLM
    
    model = AutoGPTQForCausalLM.from_quantized(
        model_name,  # e.g., "TheBloke/Llama-2-7B-GPTQ"
        device_map="auto",
        use_safetensors=True,
        trust_remote_code=True
    )
    
    return model

# Method 4: llama.cpp for CPU inference
def setup_llama_cpp():
    """Use llama.cpp for efficient CPU inference"""
    from llama_cpp import Llama
    
    # Download GGUF model
    llm = Llama(
        model_path="./models/llama-2-7b.Q4_K_M.gguf",
        n_ctx=4096,      # Context length
        n_threads=8,     # CPU threads
        n_gpu_layers=0   # 0 = full CPU, >0 = partial GPU offload
    )
    
    output = llm("What is AI?", max_tokens=100)
    return output

# Memory usage helper
def estimate_memory(model_size_b, precision):
    """Estimate memory in GB"""
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'int8': 1,
        'int4': 0.5
    }
    
    params = model_size_b * 1e9
    memory_bytes = params * bytes_per_param.get(precision, 2)
    memory_gb = memory_bytes / 1e9
    
    # Add overhead (KV cache, activations)
    overhead_factor = 1.2
    
    return memory_gb * overhead_factor

# Consumer hardware recommendations
consumer_setups = {
    '8GB GPU': {
        'max_model': '7B INT4',
        'config': {'load_in_4bit': True},
        'expected_speed': '~15-30 tokens/sec'
    },
    '12GB GPU': {
        'max_model': '13B INT4 or 7B INT8',
        'config': {'load_in_4bit': True},
        'expected_speed': '~20-40 tokens/sec'
    },
    '24GB GPU': {
        'max_model': '13B FP16 or 70B INT4 (partial)',
        'config': {'torch_dtype': torch.float16},
        'expected_speed': '~30-60 tokens/sec'
    },
    'CPU only (32GB RAM)': {
        'max_model': '13B GGUF Q4',
        'tool': 'llama.cpp',
        'expected_speed': '~2-5 tokens/sec'
    }
}

for setup, config in consumer_setups.items():
    print(f"{setup}: {config['max_model']}")
```

**Interview Tips:**
- INT4 quantization typically loses <1% performance
- GPTQ requires calibration data; bitsandbytes is zero-shot
- llama.cpp is best for CPU-only deployment
- Flash Attention 2 saves memory AND speeds up inference
- For 70B on consumer hardware: need multi-GPU or heavy CPU offload

---

### Question 36
- [ ] Done
**How do you optimize Mistral's sliding window attention for long-sequence processing?**

**Answer:**

**Definition:**
Mistral uses **Sliding Window Attention (SWA)** where each token attends to only the previous W tokens (window size = 4096) instead of all previous tokens. This reduces complexity from O(n²) to O(n×W). For long sequences, information propagates through layers — after L layers, a token can access context of L×W tokens through attention stacking.

**How Sliding Window Attention Works:**

**Standard Causal Attention:**
```
Token 5 attends to: [1, 2, 3, 4, 5] — all previous tokens
Complexity: O(n²)
```

**Sliding Window Attention (W=3):**
```
Token 5 attends to: [3, 4, 5] — only last W tokens
Complexity: O(n × W)
```

**Information Propagation Across Layers:**
```
Layer 1: Token sees positions [pos-W, pos]
Layer 2: Token sees positions [pos-2W, pos] (via layer 1)
Layer L: Token sees positions [pos-L×W, pos]

For Mistral with W=4096, L=32:
Effective context = 32 × 4096 = 131K tokens theoretically
```

**Mathematical Formulation:**

Standard attention mask:
$$M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

Sliding window mask:
$$M_{ij} = \begin{cases} 0 & \text{if } i - W < j \leq i \\ -\infty & \text{otherwise} \end{cases}$$

**Optimization Strategies:**

| Strategy | Benefit | Implementation |
|----------|---------|----------------|
| **Chunked processing** | Reduced memory peaks | Process in W-sized chunks |
| **Rolling KV-cache** | O(W) cache size | Discard old KV pairs |
| **Flash Attention + SWA** | Memory efficient | Native FA2 support |
| **Strided attention** | Layer specialization | Alternate window sizes |

**Python Code Example:**
```python
import torch
import torch.nn.functional as F

def create_sliding_window_mask(seq_len, window_size):
    """Create sliding window attention mask"""
    # Start with causal mask (lower triangular)
    mask = torch.triu(
        torch.ones(seq_len, seq_len) * float('-inf'),
        diagonal=1
    )
    
    # Add sliding window constraint
    for i in range(seq_len):
        # Mask positions outside window
        if i >= window_size:
            mask[i, :i-window_size+1] = float('-inf')
    
    return mask

def sliding_window_attention(Q, K, V, window_size):
    """Sliding window attention implementation"""
    seq_len, d_k = Q.shape
    
    # Create mask
    mask = create_sliding_window_mask(seq_len, window_size)
    
    # Compute attention
    scores = Q @ K.T / (d_k ** 0.5)
    scores = scores + mask
    
    attn_weights = F.softmax(scores, dim=-1)
    output = attn_weights @ V
    
    return output, attn_weights

# Efficient chunked processing for very long sequences
def chunked_generation(model, input_ids, max_new_tokens, window_size=4096):
    """Generate with efficient chunked processing"""
    
    # Initialize KV cache (fixed size = window_size)
    kv_cache = None
    cache_size = 0
    
    generated = input_ids.clone()
    
    for _ in range(max_new_tokens):
        # Only process tokens within window
        if len(generated) > window_size:
            input_chunk = generated[:, -window_size:]
        else:
            input_chunk = generated
        
        # Forward pass with KV cache
        outputs = model(
            input_chunk,
            past_key_values=kv_cache,
            use_cache=True
        )
        
        # Get next token
        next_token_logits = outputs.logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        
        # Update generated sequence
        generated = torch.cat([generated, next_token], dim=-1)
        
        # Update KV cache (rolling window)
        kv_cache = outputs.past_key_values
        
        # Trim cache if exceeds window
        if kv_cache is not None:
            kv_cache = trim_kv_cache(kv_cache, window_size)
    
    return generated

def trim_kv_cache(kv_cache, max_length):
    """Trim KV cache to maintain sliding window"""
    trimmed = []
    for layer_cache in kv_cache:
        k, v = layer_cache
        if k.shape[2] > max_length:
            k = k[:, :, -max_length:, :]
            v = v[:, :, -max_length:, :]
        trimmed.append((k, v))
    return tuple(trimmed)


# Memory comparison
def compare_memory_usage(seq_len, window_size, d_model):
    """Compare memory usage of different attention types"""
    
    # Standard attention: O(n²)
    standard_memory = seq_len * seq_len * 4  # float32
    
    # Sliding window: O(n × W)
    sliding_memory = seq_len * window_size * 4
    
    # KV cache comparison
    standard_kv = seq_len * d_model * 2 * 4  # K and V
    sliding_kv = window_size * d_model * 2 * 4
    
    print(f"Sequence length: {seq_len}")
    print(f"Window size: {window_size}")
    print(f"\nAttention matrix memory:")
    print(f"  Standard: {standard_memory / 1e6:.1f} MB")
    print(f"  Sliding:  {sliding_memory / 1e6:.1f} MB ({sliding_memory/standard_memory*100:.1f}%)")
    print(f"\nKV cache memory:")
    print(f"  Standard: {standard_kv / 1e6:.1f} MB")
    print(f"  Sliding:  {sliding_kv / 1e6:.1f} MB ({sliding_kv/standard_kv*100:.1f}%)")

compare_memory_usage(seq_len=32000, window_size=4096, d_model=4096)
```

**Output:**
```
Sequence length: 32000
Window size: 4096
Attention matrix memory:
  Standard: 4096.0 MB
  Sliding:  524.3 MB (12.8%)
KV cache memory:
  Standard: 1048.6 MB
  Sliding:  134.2 MB (12.8%)
```

**Interview Tips:**
- Mistral's SWA enables 32K context with 7B parameters efficiently
- Information propagates across layers — effective context > window size
- Rolling KV-cache = constant memory regardless of sequence length
- Flash Attention 2 has native support for sliding window masks

---

### Question 37
- [ ] Done
**When should you use Mistral's mixture-of-experts (MoE) architecture versus dense models?**

**Answer:**

**Definition:**
**Mixture-of-Experts (MoE)** models like Mixtral 8x7B have multiple expert FFN layers, but only activate a subset (e.g., 2 of 8) per token. This provides 47B total parameters but only 13B active per inference. Use MoE for **higher quality at same inference cost** or **faster inference at same quality**. Use dense models when memory is constrained or you need simpler deployment.

**MoE vs Dense Comparison:**

| Aspect | Mixtral 8x7B (MoE) | LLaMA 2 70B (Dense) |
|--------|-------------------|---------------------|
| **Total parameters** | 47B | 70B |
| **Active parameters** | 13B | 70B |
| **Memory (FP16)** | ~90GB | ~140GB |
| **Inference speed** | Faster | Slower |
| **Quality** | ~LLaMA 2 70B | Baseline |
| **Training cost** | Lower | Higher |

**How MoE Works:**

```
Input Token
    ↓
┌─────────────────────┐
│     Router          │ ← Selects top-k experts
│  (softmax over 8)   │
└─────────────────────┘
    ↓
┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
│E1 │ │E2 │ │E3 │ │E4 │ │E5 │ │E6 │ │E7 │ │E8 │
└───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘
  ↓     ↓
  ▼     ▼   ← Only E1 and E2 activated for this token
  Weighted Sum
    ↓
Output
```

**Mathematical Formulation:**

$$y = \sum_{i \in \text{TopK}} g_i \cdot E_i(x)$$

Where:
- $g_i$ = router weight for expert i (softmax output)
- $E_i(x)$ = output of expert i
- TopK = indices of top-k experts by router weight

**When to Use Each:**

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| **Limited inference budget** | MoE | Same quality, less compute |
| **Maximum quality needed** | Dense 70B | Slightly more consistent |
| **Multi-GPU available** | Either | Both parallelize well |
| **Single GPU constraint** | Dense 7B | MoE needs all experts in memory |
| **Diverse tasks** | MoE | Experts specialize in different tasks |
| **Latency sensitive** | MoE | Faster inference |
| **Simple deployment** | Dense | No routing overhead |

**Memory Considerations:**

```
Dense 70B:  Load all 70B parameters → 140GB FP16
Mixtral 8x7B: Load all 47B parameters → 90GB FP16
            (Can't load only active experts - routing is dynamic)
```

**Python Code Example:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    """Simplified Mixture-of-Experts layer"""
    
    def __init__(self, hidden_size, num_experts=8, top_k=2, expert_size=None):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        expert_size = expert_size or hidden_size * 4
        
        # Router (gate)
        self.router = nn.Linear(hidden_size, num_experts)
        
        # Expert networks (each is an FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, expert_size),
                nn.GELU(),
                nn.Linear(expert_size, hidden_size)
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        """
        x: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)  # (batch*seq, hidden)
        
        # Compute router logits
        router_logits = self.router(x_flat)  # (batch*seq, num_experts)
        
        # Select top-k experts
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize selected expert weights
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs (simplified - not optimized)
        output = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == i).any(dim=-1)
            
            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = expert(expert_input)
                
                # Weight by router probability
                for k in range(self.top_k):
                    mask_k = top_k_indices[:, k] == i
                    combined_mask = expert_mask & mask_k
                    if combined_mask.any():
                        weights = top_k_probs[combined_mask, k].unsqueeze(-1)
                        output[combined_mask] += weights * expert_output[:combined_mask.sum()]
        
        return output.view(batch_size, seq_len, hidden_size)

# Usage comparison
def compare_moe_vs_dense():
    """Compare compute for MoE vs Dense"""
    
    hidden_size = 4096
    num_experts = 8
    top_k = 2
    
    # Dense FFN: all parameters used
    dense_ffn_params = hidden_size * (hidden_size * 4) * 2  # up + down
    
    # MoE: only top-k experts used
    moe_total_params = num_experts * dense_ffn_params
    moe_active_params = top_k * dense_ffn_params
    
    print(f"Dense FFN parameters: {dense_ffn_params / 1e6:.1f}M")
    print(f"MoE total parameters: {moe_total_params / 1e6:.1f}M")
    print(f"MoE active parameters: {moe_active_params / 1e6:.1f}M")
    print(f"Compute ratio: {moe_active_params / dense_ffn_params:.2f}x of single FFN")
    print(f"Quality boost: ~{(num_experts / top_k):.1f}x effective capacity")

compare_moe_vs_dense()
```

**Interview Tips:**
- Mixtral 8x7B ≈ quality of LLaMA 2 70B at ~2x faster inference
- All expert weights must be in memory (can't dynamically load)
- Router can be a bottleneck — load balancing is important
- MoE excels when tasks are diverse (different experts specialize)
- Serving complexity: need to handle expert routing efficiently

---

### Question 38
- [ ] Done
**What quantization strategies (INT8, INT4, GPTQ, AWQ) work best for each model family?**

**Answer:**

**Definition:**
Quantization reduces model precision from FP16 to INT8/INT4, cutting memory 2-4x with minimal quality loss. **GPTQ** uses calibration data for optimal INT4 weights. **AWQ** (Activation-aware) preserves important weights based on activation patterns. **bitsandbytes** offers easy dynamic quantization. Choose based on quality requirements, available calibration data, and inference speed needs.

**Quantization Methods Comparison:**

| Method | Precision | Calibration | Quality | Speed | Best For |
|--------|-----------|-------------|---------|-------|----------|
| **FP16** | 16-bit | None | Baseline | Baseline | When memory allows |
| **INT8 (bitsandbytes)** | 8-bit | None | ~99% | Similar | Quick deployment |
| **GPTQ** | 4-bit | Required | ~97-99% | Fast | Pre-quantized models |
| **AWQ** | 4-bit | Required | ~98-99% | Fastest | Production inference |
| **GGUF** | 2-8 bit | None | Variable | CPU-optimized | llama.cpp |
| **ExLlama** | 4-bit | GPTQ-based | ~98% | Very fast | GPU inference |

**Model Family Recommendations:**

| Model Family | Best INT8 | Best INT4 | Notes |
|--------------|-----------|-----------|-------|
| **LLaMA 2** | bitsandbytes | GPTQ/AWQ | Best GPTQ support |
| **Mistral** | bitsandbytes | AWQ | AWQ works especially well |
| **Mixtral (MoE)** | INT8 preferred | AWQ | Experts benefit from higher precision |
| **Falcon** | bitsandbytes | GPTQ | Good GPTQ support |
| **GPT-NeoX** | bitsandbytes | GPTQ | Standard support |

**Quality vs Memory Trade-off:**

| Config | Memory | Quality Loss | Use Case |
|--------|--------|--------------|----------|
| FP16 | 100% | 0% | Training, best inference |
| INT8 | 50% | <1% | Production, memory-constrained |
| INT4 (GPTQ) | 25% | 1-3% | Consumer GPUs |
| INT4 (AWQ) | 25% | 0.5-2% | Best INT4 quality |
| INT3/INT2 | 15-20% | 5-10% | Extreme memory limits |

**Python Code Example:**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Method 1: bitsandbytes INT8 (simplest)
def load_int8_bitsandbytes(model_name):
    """Dynamic INT8 quantization"""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto"
    )
    return model

# Method 2: bitsandbytes INT4 (NF4)
def load_int4_bitsandbytes(model_name):
    """4-bit quantization with NormalFloat4"""
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4
        bnb_4bit_use_double_quant=True,      # Nested quantization
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    return model

# Method 3: GPTQ (pre-quantized model)
def load_gptq_model(model_name):
    """Load pre-quantized GPTQ model from HuggingFace"""
    # Example: "TheBloke/Llama-2-7B-GPTQ"
    from transformers import GPTQConfig
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return model

# Method 4: AWQ (pre-quantized model)
def load_awq_model(model_name):
    """Load pre-quantized AWQ model"""
    # Example: "TheBloke/Llama-2-7B-AWQ"
    from awq import AutoAWQForCausalLM
    
    model = AutoAWQForCausalLM.from_quantized(
        model_name,
        fuse_layers=True,   # Fuse for speed
        device_map="auto"
    )
    return model

# Method 5: Create GPTQ quantized model
def quantize_with_gptq(model_name, calibration_data, output_dir):
    """Quantize model using GPTQ"""
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
        damp_percent=0.1
    )
    
    model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config
    )
    
    # Prepare calibration data
    calibration_dataset = [
        tokenizer(text, return_tensors="pt")
        for text in calibration_data
    ]
    
    # Quantize
    model.quantize(calibration_dataset)
    
    # Save
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

# Benchmark function
def benchmark_quantization(model_name, methods):
    """Compare different quantization methods"""
    import time
    
    results = []
    prompt = "The quick brown fox jumps over"
    
    for method in methods:
        print(f"Testing {method}...")
        
        # Load model
        if method == 'fp16':
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )
        elif method == 'int8':
            model = load_int8_bitsandbytes(model_name)
        elif method == 'int4':
            model = load_int4_bitsandbytes(model_name)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Measure memory
        memory_gb = torch.cuda.max_memory_allocated() / 1e9
        
        # Measure speed
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start = time.time()
        for _ in range(10):
            _ = model.generate(**inputs, max_new_tokens=50)
        elapsed = (time.time() - start) / 10
        
        results.append({
            'method': method,
            'memory_gb': memory_gb,
            'latency_s': elapsed
        })
        
        del model
        torch.cuda.empty_cache()
    
    return results
```

**Decision Guide:**

| Situation | Recommendation |
|-----------|---------------|
| Quick testing | bitsandbytes INT4 |
| Production, quality critical | AWQ |
| Pre-quantized model available | Use GPTQ/AWQ from HuggingFace |
| CPU inference | GGUF with llama.cpp |
| Maximum speed | ExLlama2 with GPTQ |
| Fine-tuning after quant | QLoRA with bitsandbytes |

**Interview Tips:**
- AWQ generally preserves quality better than GPTQ
- GPTQ is more widely available (many pre-quantized models)
- bitsandbytes = easy but slightly slower than optimized GPTQ
- Always benchmark on your specific task — quality loss varies

---

### Question 39
- [ ] Done
**When should you use LoRA (Low-Rank Adaptation) versus full fine-tuning for open-source LLMs?**

**Answer:**

**Definition:**
**LoRA** freezes the base model and trains small low-rank matrices (A and B) that are added to attention weights, reducing trainable parameters by 10-1000x. Use LoRA when: memory is limited, need multiple task-specific adapters, or base model is large. Use full fine-tuning when: maximum quality is needed, small model, or sufficient compute available.

**How LoRA Works:**

$$W_{new} = W_{frozen} + \Delta W = W + BA$$

Where:
- W: Original weight matrix (d × d)
- B: Low-rank matrix (d × r)
- A: Low-rank matrix (r × d)
- r: Rank (typically 8-64, much smaller than d)

**Parameter Reduction:**
```
Original: d × d = d² parameters
LoRA: d × r + r × d = 2dr parameters
Reduction: d²/(2dr) = d/(2r)

Example: d=4096, r=16
Original: 16.7M parameters
LoRA: 131K parameters (127x reduction)
```

**Comparison:**

| Aspect | LoRA | Full Fine-tuning |
|--------|------|------------------|
| **Trainable params** | ~0.1-1% | 100% |
| **Memory** | ~10-20% of full | 100% |
| **Training speed** | Faster | Slower |
| **Quality ceiling** | Slightly lower | Highest |
| **Multi-task** | Easy (swap adapters) | Separate models |
| **Overfitting risk** | Lower | Higher |
| **Storage** | ~10-50MB per adapter | Full model per task |

**When to Use Each:**

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| **Limited GPU memory** | LoRA | 10-20% memory |
| **Large model (>30B)** | LoRA | Full FT infeasible |
| **Multiple tasks/domains** | LoRA | Swap adapters easily |
| **Small model (<3B)** | Full fine-tune | Manageable memory |
| **Maximum quality needed** | Full fine-tune | Higher ceiling |
| **Quick iteration** | LoRA | Faster training |
| **Production with variants** | LoRA | Single base + adapters |

**LoRA Hyperparameters:**

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| **r (rank)** | 8-64 | Higher = more capacity, more params |
| **alpha** | 16-32 | Scaling factor (alpha/r applied) |
| **target_modules** | q, v, k, o, mlp | Which layers to adapt |
| **dropout** | 0.05-0.1 | Regularization |

**Python Code Example:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch

def setup_lora_model(model_name, task_type="causal_lm"):
    """Setup model with LoRA adapters"""
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,                        # Rank
        lora_alpha=32,               # Scaling factor
        target_modules=[             # Which modules to adapt
            "q_proj", "v_proj",      # Attention
            "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"  # MLP
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Create PEFT model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    # Output: trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.06%
    
    return model

def train_lora(model, train_dataset, output_dir):
    """Train LoRA model"""
    from transformers import TrainingArguments, Trainer
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,         # Higher LR for LoRA
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    
    # Save only the adapter weights
    model.save_pretrained(output_dir)  # Saves only ~10-50MB

def load_and_merge_lora(base_model_name, adapter_path):
    """Load base model and merge LoRA adapter"""
    from peft import PeftModel
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load and merge adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()  # Merge into base model
    
    return model

def swap_lora_adapters(base_model, adapter_paths):
    """Demonstrate swapping adapters for multi-task"""
    from peft import PeftModel
    
    # Load base with first adapter
    model = PeftModel.from_pretrained(base_model, adapter_paths[0])
    
    # Task 1 inference
    # ... inference code ...
    
    # Swap to second adapter
    model.load_adapter(adapter_paths[1], adapter_name="task2")
    model.set_adapter("task2")
    
    # Task 2 inference
    # ... inference code ...
    
    return model

# QLoRA: LoRA + Quantization
def setup_qlora(model_name):
    """LoRA on quantized model (QLoRA)"""
    from transformers import BitsAndBytesConfig
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Prepare for training
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    return model

# Comparison helper
print("Memory comparison for 7B model:")
print("Full fine-tuning: ~28GB (FP16) + gradients + optimizer states = ~60GB+")
print("LoRA (r=16): ~14GB (frozen) + ~100MB trainable = ~16GB")
print("QLoRA (4-bit): ~4GB (frozen) + ~100MB trainable = ~6GB")
```

**Interview Tips:**
- QLoRA = LoRA + 4-bit quantization, enables fine-tuning 65B on single GPU
- Merge adapters into base model for deployment (no overhead)
- Higher rank (r) = more capacity but more parameters
- Target attention layers (q, v) at minimum; include MLP for better results

---

### Question 40
- [ ] Done
**How do you implement efficient serving infrastructure using vLLM, TensorRT-LLM, or text-generation-inference?**

**Answer:**

**Definition:**
Production LLM serving requires optimizations beyond basic inference. **vLLM** uses PagedAttention for efficient KV-cache management. **TensorRT-LLM** (NVIDIA) provides GPU kernel optimization. **Text-Generation-Inference (TGI)** offers production-ready server with batching. Choose based on: hardware (NVIDIA vs others), throughput needs, and operational complexity.

**Framework Comparison:**

| Feature | vLLM | TensorRT-LLM | TGI |
|---------|------|--------------|-----|
| **Developer** | UC Berkeley | NVIDIA | HuggingFace |
| **Key Innovation** | PagedAttention | Kernel fusion | Production ready |
| **Throughput** | Very high | Highest | High |
| **Ease of use** | Easy | Complex | Easy |
| **Hardware** | Any GPU | NVIDIA only | Any GPU |
| **Continuous batching** | Yes | Yes | Yes |
| **Quantization** | GPTQ, AWQ | INT8, FP8, INT4 | GPTQ, bitsandbytes |

**Key Optimizations:**

| Optimization | Description | Impact |
|--------------|-------------|--------|
| **Continuous batching** | Add requests to running batch | 2-5x throughput |
| **PagedAttention** | Paged KV-cache like OS memory | 10-30x memory efficiency |
| **KV-cache reuse** | Share cache for common prefixes | 2-10x for similar prompts |
| **Kernel fusion** | Combine operations | 20-50% speedup |
| **Speculative decoding** | Use small model to draft | 2-3x speedup |

**Python Code Example:**
```python
# Method 1: vLLM - Simple and efficient
from vllm import LLM, SamplingParams

def setup_vllm_server():
    """Setup vLLM for efficient inference"""
    
    # Initialize model
    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",
        tensor_parallel_size=1,      # Number of GPUs
        gpu_memory_utilization=0.9,  # Use 90% of GPU memory
        max_model_len=4096,          # Maximum sequence length
    )
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256
    )
    
    # Single inference
    prompts = ["Explain quantum computing in simple terms."]
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        print(f"Generated: {output.outputs[0].text}")
    
    return llm

def vllm_batch_inference(llm, prompts, sampling_params):
    """Efficient batch inference with vLLM"""
    # vLLM automatically handles batching
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]


# Method 2: TGI - Docker-based deployment
def setup_tgi_docker():
    """TGI deployment command"""
    docker_command = """
    docker run --gpus all --shm-size 1g -p 8080:80 \\
        -v /data/models:/data \\
        ghcr.io/huggingface/text-generation-inference:latest \\
        --model-id meta-llama/Llama-2-7b-hf \\
        --num-shard 1 \\
        --quantize bitsandbytes-nf4 \\
        --max-input-length 2048 \\
        --max-total-tokens 4096
    """
    return docker_command

def call_tgi_api(prompt, max_tokens=256):
    """Call TGI REST API"""
    import requests
    
    response = requests.post(
        "http://localhost:8080/generate",
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }
    )
    
    return response.json()["generated_text"]


# Method 3: vLLM OpenAI-compatible server
def setup_vllm_openai_server():
    """vLLM with OpenAI-compatible API"""
    # Start server via command line:
    # python -m vllm.entrypoints.openai.api_server \
    #     --model meta-llama/Llama-2-7b-hf \
    #     --host 0.0.0.0 --port 8000
    
    # Then use OpenAI client
    from openai import OpenAI
    
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy"  # Not needed for local
    )
    
    response = client.chat.completions.create(
        model="meta-llama/Llama-2-7b-hf",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    return response.choices[0].message.content


# Method 4: TensorRT-LLM (most complex, highest performance)
def setup_tensorrt_llm():
    """TensorRT-LLM setup (NVIDIA only)"""
    setup_steps = """
    # 1. Convert model to TensorRT format
    python convert_checkpoint.py \\
        --model_dir ./llama-7b \\
        --output_dir ./llama-7b-trt \\
        --dtype float16
    
    # 2. Build engine
    trtllm-build \\
        --checkpoint_dir ./llama-7b-trt \\
        --output_dir ./llama-7b-engine \\
        --gemm_plugin float16 \\
        --gpt_attention_plugin float16 \\
        --max_batch_size 8 \\
        --max_input_len 2048 \\
        --max_output_len 512
    
    # 3. Run inference
    python run.py \\
        --engine_dir ./llama-7b-engine \\
        --tokenizer_dir ./llama-7b \\
        --input_text "Hello, how are you?"
    """
    return setup_steps


# Performance tuning helper
class ServingConfig:
    def __init__(self, model_size, gpu_memory_gb, expected_qps):
        self.model_size = model_size
        self.gpu_memory = gpu_memory_gb
        self.qps = expected_qps
    
    def recommend_setup(self):
        """Recommend serving configuration"""
        
        recommendations = {
            'framework': None,
            'batch_size': None,
            'num_gpus': None,
            'quantization': None
        }
        
        # Framework selection
        if self.qps > 100:
            recommendations['framework'] = 'TensorRT-LLM'
        elif self.qps > 20:
            recommendations['framework'] = 'vLLM'
        else:
            recommendations['framework'] = 'TGI'
        
        # GPU and quantization
        model_memory = {'7b': 14, '13b': 26, '70b': 140}
        required = model_memory.get(self.model_size, 14)
        
        if self.gpu_memory < required:
            recommendations['quantization'] = 'INT4'
            required = required / 4
        
        recommendations['num_gpus'] = max(1, int(required / self.gpu_memory) + 1)
        
        # Batch size
        recommendations['batch_size'] = min(32, int(self.qps / 2))
        
        return recommendations

# Usage
config = ServingConfig(model_size='7b', gpu_memory_gb=24, expected_qps=50)
print(config.recommend_setup())
```

**Interview Tips:**
- vLLM's PagedAttention is key innovation — treats KV-cache like virtual memory
- Continuous batching > static batching (2-5x throughput)
- TensorRT-LLM requires NVIDIA but offers best performance
- For OpenAI drop-in replacement, use vLLM's OpenAI server

---

### Question 41
- [ ] Done
**When would you choose Code Llama versus general-purpose models for programming tasks?**

**Answer:**

**Definition:**
Choose **Code Llama** for: code generation, completion, debugging, and technical documentation. It's trained on 500B tokens of code with fill-in-the-middle capability. Choose **general-purpose models** (GPT-4, Claude) for: mixed code/natural language tasks, complex reasoning about code, or when code is just part of a broader task. Code Llama excels at pure code tasks; general models offer better reasoning.

**Comparison:**

| Aspect | Code Llama | General LLMs (GPT-4, Claude) |
|--------|------------|------------------------------|
| **Code generation** | Excellent | Very good |
| **Fill-in-the-middle** | Yes (trained for it) | Limited |
| **Long code context** | 100K tokens | 8K-128K |
| **Reasoning about code** | Good | Excellent |
| **Natural language + code** | Limited | Excellent |
| **Deployment** | Self-hosted, free | API, paid |
| **Cost at scale** | Fixed (compute) | Per token |

**Code Llama Variants:**

| Variant | Size | Specialty |
|---------|------|-----------|
| **Code Llama** | 7B, 13B, 34B, 70B | General code |
| **Code Llama - Python** | 7B, 13B, 34B | Python specialized |
| **Code Llama - Instruct** | 7B, 13B, 34B, 70B | Instruction-following |

**When to Use Each:**

| Task | Best Choice | Reason |
|------|-------------|--------|
| **IDE autocomplete** | Code Llama | Fast, fill-in-middle |
| **Code review** | General LLM | Needs reasoning |
| **Bug explanation** | General LLM | Natural language |
| **Docstring generation** | Code Llama | Code-focused |
| **Algorithm design** | General LLM | Complex reasoning |
| **Language translation (code)** | Either | Both capable |
| **Code search/understanding** | Code Llama | Better embeddings |
| **High-volume code completion** | Code Llama | Cost effective |

**Python Code Example:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Code Llama
def load_code_llama(variant="codellama/CodeLlama-7b-Instruct-hf"):
    """Load Code Llama for code generation"""
    tokenizer = AutoTokenizer.from_pretrained(variant)
    model = AutoModelForCausalLM.from_pretrained(
        variant,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def code_completion(model, tokenizer, code_prefix, max_new_tokens=100):
    """Complete code given a prefix"""
    inputs = tokenizer(code_prefix, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.2,  # Low for code
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Fill-in-the-middle (special Code Llama capability)
def fill_in_the_middle(model, tokenizer, prefix, suffix):
    """Code Llama's fill-in-the-middle capability"""
    # Special tokens for infilling
    INFILL_PREFIX = "<PRE>"
    INFILL_SUFFIX = "<SUF>"
    INFILL_MIDDLE = "<MID>"
    
    prompt = f"{INFILL_PREFIX}{prefix}{INFILL_SUFFIX}{suffix}{INFILL_MIDDLE}"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.2,
        do_sample=True
    )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the infilled part
    if INFILL_MIDDLE in generated:
        infill = generated.split(INFILL_MIDDLE)[1]
        return infill
    return generated

# Example usage
prefix = """def calculate_fibonacci(n):
    '''Calculate the nth Fibonacci number'''
    if n <= 1:
        return n
"""
suffix = """
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
"""

# infill = fill_in_the_middle(model, tokenizer, prefix, suffix)
# Result: "    else:\n"


# Decision helper: Code Llama vs General LLM
def choose_model_for_task(task_type, requirements):
    """Recommend Code Llama or General LLM"""
    
    code_llama_tasks = {
        'completion', 'autocomplete', 'infill', 'fill_in_middle',
        'docstring', 'type_hints', 'code_translation'
    }
    
    general_llm_tasks = {
        'code_review', 'bug_explanation', 'architecture_design',
        'documentation', 'code_plus_prose', 'complex_debugging'
    }
    
    if task_type in code_llama_tasks:
        if requirements.get('high_volume', False):
            return "Code Llama (self-hosted) - cost effective at scale"
        return "Code Llama or Codex API"
    
    if task_type in general_llm_tasks:
        return "GPT-4 / Claude - better reasoning"
    
    # Hybrid
    if requirements.get('needs_reasoning', False):
        return "General LLM (GPT-4/Claude)"
    
    return "Code Llama for code-heavy, General LLM for reasoning-heavy"


# Benchmark comparison
tasks = [
    {'task': 'autocomplete', 'description': 'IDE code completion'},
    {'task': 'code_review', 'description': 'Review PR for issues'},
    {'task': 'bug_explanation', 'description': 'Explain why code fails'},
    {'task': 'docstring', 'description': 'Generate function docs'},
]

for t in tasks:
    recommendation = choose_model_for_task(t['task'], {})
    print(f"{t['description']}: {recommendation}")
```

**Interview Tips:**
- Code Llama's 100K context allows analyzing entire codebases
- Fill-in-the-middle is unique to code models — essential for autocomplete
- For production IDEs: use smaller Code Llama (7B) for speed
- General LLMs better at explaining bugs in natural language
- Hybrid approach: Code Llama for generation, GPT-4 for review

---

## Encoder-Decoder Models (T5, BART)

### Question 42
- [ ] Done
**How do you choose between T5's text-to-text approach and BART's denoising pre-training?**

**Answer:**

**Definition:**
**T5** frames all NLP tasks as text-to-text (input text → output text), enabling a unified approach to diverse tasks. **BART** uses denoising autoencoding with document corruption/reconstruction. Choose T5 for multi-task learning and unified APIs. Choose BART for tasks requiring strong encoder (summarization) or when task-specific fine-tuning yields better results.

**Comparison:**

| Aspect | T5 | BART |
|--------|-----|------|
| **Pre-training** | Span corruption (mask spans) | Multiple noise functions |
| **Task format** | Text-to-text (prefix defines task) | Task-specific heads |
| **Architecture** | Encoder-decoder | Encoder-decoder |
| **Flexibility** | Very high (any task as text) | Medium (needs adaptation) |
| **Summarization** | Good | Excellent |
| **Translation** | Excellent | Good |
| **Zero-shot** | Better (unified format) | Limited |

**Pre-training Differences:**

**T5 (Span Corruption):**
```
Input:  "The <extra_id_0> brown fox <extra_id_1> the lazy dog"
Target: "<extra_id_0> quick <extra_id_1> jumps over"
```

**BART (Multiple Noise Functions):**
- Token masking
- Token deletion
- Sentence permutation
- Document rotation
- Text infilling

**When to Use Each:**

| Task | Better Choice | Reason |
|------|---------------|--------|
| **Multi-task learning** | T5 | Single model, task prefixes |
| **Abstractive summarization** | BART | Denoising helps generation |
| **Translation** | T5 | Strong encoder-decoder |
| **Question answering** | Either | Both work well |
| **Classification** | T5 | Easy text-to-text format |
| **Dialogue** | BART | Better at understanding |
| **Unified API** | T5 | Everything is text-to-text |

**Python Code Example:**
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration

# T5: Text-to-Text approach
def t5_multi_task():
    """T5 handles multiple tasks with prefixes"""
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    
    tasks = [
        # Translation
        ("translate English to German: Hello, how are you?", "translation"),
        # Summarization
        ("summarize: The quick brown fox jumps over the lazy dog. " * 10, "summarization"),
        # Classification (as generation)
        ("sst2 sentence: This movie was fantastic!", "sentiment"),
        # Question answering
        ("question: What is AI? context: AI is artificial intelligence.", "qa"),
    ]
    
    for text, task_type in tasks:
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(**inputs, max_length=50)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"{task_type}: {result}")
    
    return model

# BART: Specialized for summarization/generation
def bart_summarization():
    """BART excels at summarization"""
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    
    article = """
    Scientists have discovered a new species of deep-sea fish in the Pacific Ocean. 
    The fish, which has bioluminescent properties, was found at depths of over 3,000 meters.
    Researchers believe this discovery could lead to new insights about marine biology
    and the adaptation of life in extreme environments.
    """
    
    inputs = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        **inputs,
        max_length=60,
        min_length=20,
        num_beams=4,
        length_penalty=2.0
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(f"Summary: {summary}")
    
    return model

# Decision helper
def choose_model(tasks, requirements):
    """Recommend T5 or BART based on requirements"""
    
    if len(tasks) > 1:
        return "T5 - unified text-to-text for multiple tasks"
    
    if 'summarization' in tasks and requirements.get('quality_priority'):
        return "BART - best for abstractive summarization"
    
    if requirements.get('zero_shot'):
        return "T5 - better zero-shot via text format"
    
    if 'generation' in tasks:
        return "BART - strong decoder for generation"
    
    return "Either - both are capable"

# Example
print(choose_model(['summarization', 'translation'], {}))
# Output: "T5 - unified text-to-text for multiple tasks"
```

**Interview Tips:**
- T5's key insight: every NLP task can be text-to-text
- BART = BERT encoder + GPT decoder
- T5-large and BART-large have similar parameter counts (~400M)
- For production multi-task: T5 saves having multiple models
- Flan-T5 (instruction-tuned) is often better than base T5

---

### Question 43
- [ ] Done
**What are the advantages of T5's unified framework for multi-task learning systems?**

**Answer:**

**Definition:**
T5's text-to-text framework provides: **single model for all tasks** (no task-specific heads), **consistent API** (input string → output string), **easy multi-task training** (mix datasets with prefixes), and **zero-shot generalization** (new tasks work without retraining). This simplifies deployment, reduces maintenance, and enables knowledge transfer across tasks.

**Core Advantages:**

| Advantage | Description | Benefit |
|-----------|-------------|---------|
| **Unified architecture** | No task-specific layers | One model serves all |
| **Simple API** | text in → text out | Easy integration |
| **Multi-task training** | Mix datasets easily | Knowledge transfer |
| **New task adaptation** | Just add prefix | No architecture changes |
| **Zero-shot capability** | Describe task in text | Generalizes to unseen tasks |
| **Reduced complexity** | No task-specific heads | Simpler codebase |

**How Task Prefixes Work:**
```
Classification:  "classify: This movie is great"     → "positive"
Summarization:   "summarize: [long document]"        → "summary text"
Translation:     "translate to French: Hello"        → "Bonjour"
QA:              "question: What is X? context: ..." → "answer"
```

**Python Code Example:**
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

class UnifiedT5System:
    """Multi-task system using T5's text-to-text format"""
    
    def __init__(self, model_name="t5-base"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    def process(self, task, text, **kwargs):
        """Unified interface for all tasks"""
        
        task_prefixes = {
            'summarize': 'summarize:',
            'translate_en_de': 'translate English to German:',
            'translate_en_fr': 'translate English to French:',
            'sentiment': 'sst2 sentence:',
            'qa': 'question: {question} context: {context}',
            'grammar': 'grammar:',
            'paraphrase': 'paraphrase:',
        }
        
        prefix = task_prefixes.get(task, task + ':')
        
        if task == 'qa':
            input_text = prefix.format(**kwargs)
        else:
            input_text = f"{prefix} {text}"
        
        inputs = self.tokenizer(input_text, return_tensors="pt", 
                                max_length=512, truncation=True)
        outputs = self.model.generate(**inputs, max_length=100)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage - single model handles everything
system = UnifiedT5System()

# All tasks use the same interface
# summary = system.process('summarize', long_text)
# translation = system.process('translate_en_de', "Hello world")
# sentiment = system.process('sentiment', "Great product!")
# answer = system.process('qa', '', question="What is AI?", context="AI is...")
```

**Interview Tips:**
- T5's insight: "Every NLP task can be cast as text-to-text"
- Flan-T5 improves zero-shot through instruction tuning
- In production: one T5 model replaces multiple specialized models
- Trade-off: may not achieve absolute best on each individual task

---

### Question 44
- [ ] Done
**What are the best practices for fine-tuning BART on abstractive summarization tasks?**

**Answer:**

**Definition:**
Fine-tune BART for summarization by: using **BART-large-CNN** as starting point, implementing **label smoothing** (0.1), using **length penalties** in generation, setting appropriate **min/max length constraints**, and employing **beam search** (4-6 beams). Monitor ROUGE scores on validation set and use early stopping.

**Best Practices:**

| Practice | Recommendation | Reason |
|----------|---------------|--------|
| **Base model** | bart-large-cnn | Pre-finetuned for news |
| **Learning rate** | 3e-5 to 5e-5 | Standard for fine-tuning |
| **Batch size** | 8-16 (with gradient accumulation) | Memory constraints |
| **Label smoothing** | 0.1 | Prevents overconfidence |
| **Max source length** | 1024 | BART's limit |
| **Max target length** | 128-256 | Task dependent |
| **Beam size** | 4-6 | Quality vs speed trade-off |
| **Length penalty** | 1.5-2.0 | Encourages longer outputs |

**Python Code Example:**
```python
from transformers import (
    BartForConditionalGeneration, BartTokenizer,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
)
import evaluate

def fine_tune_bart_summarization(train_dataset, val_dataset):
    """Fine-tune BART for summarization"""
    
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Preprocessing
    max_source_length = 1024
    max_target_length = 128
    
    def preprocess(examples):
        inputs = tokenizer(
            examples['article'],
            max_length=max_source_length,
            truncation=True,
            padding='max_length'
        )
        
        targets = tokenizer(
            examples['summary'],
            max_length=max_target_length,
            truncation=True,
            padding='max_length'
        )
        
        inputs['labels'] = targets['input_ids']
        return inputs
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./bart-summarization",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        label_smoothing_factor=0.1,  # Key for summarization
        predict_with_generate=True,
        generation_max_length=max_target_length,
        generation_num_beams=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rouge2",
    )
    
    # ROUGE metrics
    rouge = evaluate.load("rouge")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = [[l for l in label if l != -100] for label in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return {k: round(v, 4) for k, v in result.items()}
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    return model

# Generation with best practices
def generate_summary(model, tokenizer, article):
    """Generate summary with optimized parameters"""
    inputs = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True)
    
    summary_ids = model.generate(
        **inputs,
        num_beams=4,           # Beam search
        length_penalty=2.0,    # Encourage longer summaries
        min_length=30,         # Minimum output length
        max_length=150,        # Maximum output length
        no_repeat_ngram_size=3,  # Avoid repetition
        early_stopping=True
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

**Interview Tips:**
- Start with bart-large-cnn for news; bart-large for other domains
- ROUGE-2 and ROUGE-L are standard metrics
- Length penalty > 1 encourages longer outputs
- no_repeat_ngram_size prevents repetitive generations

---

### Question 45
- [ ] Done
**When should you use T5's encoder-decoder architecture versus decoder-only models?**

**Answer:**

**Definition:**
Use **encoder-decoder (T5, BART)** for tasks where input and output are distinct (translation, summarization, structured extraction). Use **decoder-only (GPT)** for open-ended generation, continuation, and conversational tasks. Encoder-decoder excels at conditional generation with explicit source-target separation; decoder-only is simpler and better at free-form generation.

**Comparison:**

| Aspect | Encoder-Decoder (T5) | Decoder-Only (GPT) |
|--------|---------------------|-------------------|
| **Input processing** | Bidirectional (full context) | Left-to-right only |
| **Source-target separation** | Explicit | Implicit (in prompt) |
| **Summarization** | Excellent | Good |
| **Translation** | Excellent | Good |
| **Open generation** | Limited | Excellent |
| **Conversation** | Needs adaptation | Natural |
| **Context efficiency** | Better (encoder compresses) | Uses context for both |

**When to Use Each:**

| Task | Encoder-Decoder | Decoder-Only |
|------|-----------------|--------------|
| **Translation** | ✓ Best | Good |
| **Summarization** | ✓ Best | Good |
| **Question Answering** | ✓ Good | ✓ Good |
| **Open-ended chat** | Limited | ✓ Best |
| **Creative writing** | Limited | ✓ Best |
| **Data extraction** | ✓ Best | Good |
| **Code completion** | Limited | ✓ Best |

**Intuition:**
- **Encoder-Decoder:** "Here's document X, produce output Y based on it"
- **Decoder-Only:** "Continue this text naturally"

**Python Code Example:**
```python
# Encoder-Decoder: T5 for translation
from transformers import T5ForConditionalGeneration, T5Tokenizer

def encoder_decoder_translation():
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    
    text = "translate English to German: The house is wonderful."
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
    # "Das Haus ist wunderbar."

# Decoder-Only: GPT for continuation
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def decoder_only_generation():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    prompt = "Once upon a time, in a land far away,"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    
    return tokenizer.decode(outputs[0])
    # Continues the story naturally

# Decision helper
def choose_architecture(task):
    encoder_decoder_tasks = {'translation', 'summarization', 'data_extraction', 
                             'question_answering', 'text_transformation'}
    decoder_only_tasks = {'chat', 'creative_writing', 'code_completion',
                          'continuation', 'instruction_following'}
    
    if task in encoder_decoder_tasks:
        return "Encoder-Decoder (T5/BART)"
    elif task in decoder_only_tasks:
        return "Decoder-Only (GPT/LLaMA)"
    else:
        return "Either - depends on specific requirements"
```

**Interview Tips:**
- Encoder provides bidirectional context understanding of source
- Decoder-only is simpler, scales better (most modern LLMs)
- Cross-attention in enc-dec allows explicit source-target attention
- For new projects, decoder-only is often sufficient with good prompting

---

### Question 46
- [ ] Done
**How do you implement effective beam search and decoding strategies for optimal generation quality?**

**Answer:**

**Definition:**
**Beam search** maintains top-k sequences at each step, exploring multiple paths to find higher-probability outputs. Alternatives include **greedy** (fastest, lowest quality), **sampling** (creative but less coherent), **nucleus (top-p)** sampling (balanced), and **contrastive search** (recent, high quality). Choose based on task: beam search for translation/summarization, sampling for creative tasks.

**Decoding Strategies Comparison:**

| Strategy | Quality | Diversity | Speed | Best For |
|----------|---------|-----------|-------|----------|
| **Greedy** | Low | None | Fastest | Quick drafts |
| **Beam search** | High | Low | Medium | Translation, summarization |
| **Top-k sampling** | Medium | High | Fast | Creative writing |
| **Top-p (nucleus)** | Medium-High | Balanced | Fast | General use |
| **Contrastive** | High | High | Slow | High-quality generation |

**Mathematical Formulation:**

**Beam Search:**
At each step, keep top B sequences by:
$$\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i | y_{<i})$$

With length normalization:
$$\text{score}_{norm} = \frac{1}{t^\alpha} \sum_{i=1}^{t} \log P(y_i | y_{<i})$$

**Python Code Example:**
```python
import torch
import torch.nn.functional as F

def greedy_decode(model, input_ids, max_length):
    """Greedy decoding - always pick highest probability token"""
    generated = input_ids.clone()
    
    for _ in range(max_length):
        outputs = model(generated)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
    
    return generated

def beam_search(model, input_ids, beam_width=4, max_length=50, length_penalty=1.0):
    """Beam search decoding"""
    batch_size = input_ids.shape[0]
    
    # Initialize beams: (sequence, score)
    beams = [(input_ids, 0.0)]
    
    for step in range(max_length):
        all_candidates = []
        
        for seq, score in beams:
            outputs = model(seq)
            logits = outputs.logits[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Get top-k next tokens
            top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)
            
            for i in range(beam_width):
                next_token = top_indices[:, i:i+1]
                token_score = top_log_probs[:, i].item()
                
                new_seq = torch.cat([seq, next_token], dim=-1)
                new_score = score + token_score
                
                all_candidates.append((new_seq, new_score))
        
        # Keep top beam_width candidates
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_width]
    
    # Apply length normalization and return best
    best_seq, best_score = max(beams, 
        key=lambda x: x[1] / (len(x[0][0]) ** length_penalty))
    
    return best_seq

def nucleus_sampling(model, input_ids, max_length, top_p=0.9, temperature=1.0):
    """Top-p (nucleus) sampling"""
    generated = input_ids.clone()
    
    for _ in range(max_length):
        outputs = model(generated)
        logits = outputs.logits[:, -1, :] / temperature
        
        # Sort by probability
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        
        # Remove tokens with cumulative prob above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        
        # Sample from filtered distribution
        probs = F.softmax(sorted_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        next_token = sorted_indices.gather(-1, next_token)
        
        generated = torch.cat([generated, next_token], dim=-1)
    
    return generated

# Using HuggingFace generate
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")

# Different strategies
strategies = {
    'greedy': {'do_sample': False},
    'beam': {'do_sample': False, 'num_beams': 4},
    'sampling': {'do_sample': True, 'temperature': 0.8},
    'nucleus': {'do_sample': True, 'top_p': 0.9},
    'top_k': {'do_sample': True, 'top_k': 50},
}

# for name, params in strategies.items():
#     output = model.generate(**inputs, max_length=50, **params)
#     print(f"{name}: {tokenizer.decode(output[0])}")
```

**Interview Tips:**
- Beam search for deterministic, high-quality outputs
- Nucleus sampling (top_p) is default for chat/creative
- length_penalty > 1 encourages longer outputs
- no_repeat_ngram_size prevents repetition in beam search

---

## Model Training & Optimization

### Question 47
- [ ] Done
**Explain gradient accumulation and why it's essential for training large models on limited GPU memory.**

**Answer:**

**Definition:**
Gradient accumulation simulates larger batch sizes by accumulating gradients over multiple forward-backward passes before updating weights. Instead of updating after each batch, gradients are summed over N steps, then divided by N and applied. This enables effective batch size = N × micro_batch_size, training large models on GPUs with limited memory.

**Why It's Essential:**

| Problem | Solution via Gradient Accumulation |
|---------|-----------------------------------|
| Large batch doesn't fit in memory | Use small batches, accumulate |
| Training instability with small batches | Simulates large batch effect |
| GPU memory limited | Maintains quality with memory constraint |
| Need consistent training dynamics | Matches large-batch behavior |

**Mathematical Formulation:**

**Standard training:**
$$\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}(\text{batch})$$

**With gradient accumulation (N steps):**
$$g_{accumulated} = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \mathcal{L}(\text{micro\_batch}_i)$$
$$\theta \leftarrow \theta - \eta \cdot g_{accumulated}$$

**Effective batch size:** `micro_batch_size × accumulation_steps × num_gpus`

**Python Code Example:**
```python
import torch
from transformers import TrainingArguments

# Method 1: Manual implementation
def train_with_accumulation(model, dataloader, optimizer, 
                            accumulation_steps=4, num_epochs=3):
    """Manual gradient accumulation training loop"""
    
    model.train()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        accumulated_loss = 0
        
        for step, batch in enumerate(dataloader):
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Normalize loss by accumulation steps
            loss = loss / accumulation_steps
            accumulated_loss += loss.item()
            
            # Backward pass (accumulates gradients)
            loss.backward()
            
            # Update weights every accumulation_steps
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                print(f"Step {step+1}, Loss: {accumulated_loss:.4f}")
                accumulated_loss = 0
        
        # Handle remaining gradients
        if (step + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

# Method 2: HuggingFace Trainer (automatic)
training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=4,      # Fits in memory
    gradient_accumulation_steps=8,       # Effective batch = 4 × 8 = 32
    learning_rate=5e-5,
    num_train_epochs=3,
)

# Method 3: PyTorch with mixed precision
def train_with_amp_accumulation(model, dataloader, optimizer, 
                                 accumulation_steps=4):
    """Gradient accumulation with automatic mixed precision"""
    
    scaler = torch.cuda.amp.GradScaler()
    
    for step, batch in enumerate(dataloader):
        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps
        
        # Scaled backward
        scaler.scale(loss).backward()
        
        if (step + 1) % accumulation_steps == 0:
            # Unscale and clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

# Memory comparison
def calculate_effective_batch_size():
    """Show relationship between memory and batch size"""
    
    configs = [
        {'batch': 32, 'accum': 1, 'memory': 'High'},
        {'batch': 8, 'accum': 4, 'memory': 'Medium'},
        {'batch': 2, 'accum': 16, 'memory': 'Low'},
        {'batch': 1, 'accum': 32, 'memory': 'Very Low'},
    ]
    
    for c in configs:
        effective = c['batch'] * c['accum']
        print(f"Batch={c['batch']}, Accum={c['accum']}, "
              f"Effective={effective}, Memory={c['memory']}")

calculate_effective_batch_size()
```

**Interview Tips:**
- Gradient accumulation is mathematically equivalent to large batch
- Learning rate may need scaling with effective batch size
- Accumulation adds training time but not memory
- Essential for fine-tuning LLMs on consumer GPUs

---

### Question 48
- [ ] Done
**How does warmup and learning rate scheduling (linear decay, cosine) affect training convergence?**

**Answer:**

**Definition:**
**Learning rate warmup** gradually increases LR from near-zero to target value during early training, preventing large unstable updates when model weights are random. **LR scheduling** (linear decay, cosine annealing) reduces LR over time, allowing fine-grained convergence. Together they stabilize training and improve final model quality.

**Why Warmup is Necessary:**
- Initial gradients are noisy (random weights)
- Large LR + noisy gradients = unstable updates
- Warmup allows gradients to stabilize before using full LR
- Critical for Transformers with LayerNorm

**Common Schedules:**

| Schedule | Formula | Behavior |
|----------|---------|----------|
| **Linear warmup** | LR increases linearly | 0 → target_lr |
| **Linear decay** | LR decreases linearly | target_lr → 0 |
| **Cosine annealing** | LR follows cosine curve | Smooth decrease |
| **Warmup + linear decay** | Ramp up, then linear down | Most common |
| **Warmup + cosine** | Ramp up, then cosine | Used in LLMs |

**Mathematical Formulation:**

**Linear Warmup (steps 0 to T_warmup):**
$$lr(t) = \frac{t}{T_{warmup}} \cdot lr_{max}$$

**Linear Decay (steps T_warmup to T_total):**
$$lr(t) = lr_{max} \cdot \left(1 - \frac{t - T_{warmup}}{T_{total} - T_{warmup}}\right)$$

**Cosine Annealing:**
$$lr(t) = lr_{min} + \frac{1}{2}(lr_{max} - lr_{min})\left(1 + \cos\left(\frac{t \cdot \pi}{T_{total}}\right)\right)$$

**Python Code Example:**
```python
import math
import matplotlib.pyplot as plt
from transformers import get_scheduler

# Manual implementation
def get_linear_warmup_schedule(optimizer, warmup_steps, total_steps):
    """Linear warmup followed by linear decay"""
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Warmup phase
            return current_step / warmup_steps
        else:
            # Linear decay
            return max(0, (total_steps - current_step) / (total_steps - warmup_steps))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_cosine_warmup_schedule(optimizer, warmup_steps, total_steps):
    """Linear warmup followed by cosine decay"""
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Using HuggingFace transformers
import torch

model_params = [torch.nn.Parameter(torch.randn(10))]
optimizer = torch.optim.AdamW(model_params, lr=5e-5)

scheduler = get_scheduler(
    name="linear",  # or "cosine", "constant_with_warmup"
    optimizer=optimizer,
    num_warmup_steps=1000,
    num_training_steps=10000
)

# Visualize schedules
def plot_schedules():
    """Visualize different learning rate schedules"""
    total_steps = 10000
    warmup_steps = 1000
    
    schedules = {
        'linear': [],
        'cosine': [],
        'constant_warmup': [],
    }
    
    for step in range(total_steps):
        # Linear warmup + decay
        if step < warmup_steps:
            lr_linear = step / warmup_steps
        else:
            lr_linear = max(0, (total_steps - step) / (total_steps - warmup_steps))
        
        # Cosine warmup + decay
        if step < warmup_steps:
            lr_cosine = step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr_cosine = 0.5 * (1 + math.cos(math.pi * progress))
        
        # Constant with warmup
        lr_constant = min(1.0, step / warmup_steps)
        
        schedules['linear'].append(lr_linear)
        schedules['cosine'].append(lr_cosine)
        schedules['constant_warmup'].append(lr_constant)
    
    # Plot
    for name, lrs in schedules.items():
        plt.plot(lrs, label=name)
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate (relative)')
    plt.legend()
    plt.title('Learning Rate Schedules')
    plt.show()

# Practical settings for LLMs
recommended_settings = {
    'BERT fine-tuning': {
        'schedule': 'linear',
        'warmup_ratio': 0.1,
        'max_lr': 2e-5
    },
    'LLM pretraining': {
        'schedule': 'cosine',
        'warmup_steps': 2000,
        'max_lr': 3e-4
    },
    'LoRA fine-tuning': {
        'schedule': 'cosine',
        'warmup_ratio': 0.03,
        'max_lr': 2e-4
    }
}
```

**Interview Tips:**
- Warmup typically 5-10% of total training steps
- Cosine schedule often slightly better than linear for LLMs
- Higher warmup for larger batches (more stable)
- Monitor training loss — spikes often mean LR too high

---

### Question 49
- [ ] Done
**What is gradient checkpointing and how does it trade compute for memory savings?**

**Answer:**

**Definition:**
Gradient checkpointing (activation checkpointing) reduces memory by **not storing all intermediate activations** during forward pass. Instead, it saves checkpoints at selected layers and recomputes activations during backward pass. This trades ~30-50% extra compute time for 4-10x memory reduction, enabling training of larger models or bigger batches.

**How It Works:**

**Standard Training:**
```
Forward: Store all activations A1, A2, A3, ... AN → High memory
Backward: Use stored activations → Fast
```

**With Checkpointing:**
```
Forward: Store only checkpoints (A1, A5, A10) → Low memory
Backward: Recompute A2, A3, A4 from A1; A6, A7, A8, A9 from A5 → Slower
```

**Trade-off:**

| Aspect | Standard | Checkpointing |
|--------|----------|---------------|
| **Memory** | O(n) activations | O(√n) checkpoints |
| **Compute** | 1× forward | ~1.3× forward |
| **Training time** | Baseline | +20-50% |
| **Use case** | Memory sufficient | Memory limited |

**Python Code Example:**
```python
import torch
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

# Method 1: Manual checkpointing
class TransformerBlockWithCheckpoint(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(hidden_size, 8)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * 4),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm1 = torch.nn.LayerNorm(hidden_size)
        self.norm2 = torch.nn.LayerNorm(hidden_size)
    
    def forward(self, x, use_checkpoint=False):
        if use_checkpoint:
            # Recompute attention during backward
            x = x + checkpoint(self._attention_block, x)
            x = x + checkpoint(self._ffn_block, x)
        else:
            x = x + self._attention_block(x)
            x = x + self._ffn_block(x)
        return x
    
    def _attention_block(self, x):
        return self.attention(x, x, x)[0]
    
    def _ffn_block(self, x):
        return self.ffn(self.norm2(x))

# Method 2: HuggingFace automatic checkpointing
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Method 3: Sequential checkpointing
def create_checkpointed_model(layers, segments=4):
    """Checkpoint a sequential model"""
    
    def forward(x):
        return checkpoint_sequential(layers, segments, x)
    
    return forward

# Memory comparison
def compare_memory_usage():
    """Demonstrate memory savings"""
    
    hidden_size = 4096
    num_layers = 32
    batch_size = 4
    seq_len = 2048
    
    # Activation memory per layer (approximate)
    bytes_per_element = 2  # FP16
    activation_size = batch_size * seq_len * hidden_size * bytes_per_element
    
    # Standard: store all layers
    standard_memory = num_layers * activation_size / 1e9
    
    # Checkpointing: store sqrt(n) checkpoints
    import math
    checkpoints = int(math.sqrt(num_layers))
    checkpoint_memory = checkpoints * activation_size / 1e9
    
    print(f"Standard activation memory: {standard_memory:.2f} GB")
    print(f"Checkpointed memory: {checkpoint_memory:.2f} GB")
    print(f"Reduction: {standard_memory / checkpoint_memory:.1f}x")

compare_memory_usage()
```

**Interview Tips:**
- Essential for training large models on limited hardware
- Compute overhead typically 20-40%
- Works with mixed precision training (FP16/BF16)
- HuggingFace: `model.gradient_checkpointing_enable()`

---

### Question 50
- [ ] Done
**How do you implement model parallelism (tensor, pipeline) for training very large models?**

**Answer:**

**Definition:**
**Model parallelism** splits a model across multiple GPUs when it's too large for single GPU. **Tensor parallelism** splits individual layers (matrix operations) across GPUs. **Pipeline parallelism** places different layers on different GPUs, processing micro-batches in pipeline fashion. Use tensor for memory, pipeline for throughput.

**Types of Parallelism:**

| Type | What's Split | Communication | Best For |
|------|--------------|---------------|----------|
| **Data parallel** | Batch | Gradient sync | Fits on 1 GPU |
| **Tensor parallel** | Layer weights | Every operation | Single very large layer |
| **Pipeline parallel** | Layers | Between stages | Very deep models |
| **3D parallel** | All combined | Complex | Largest models (GPT-3+) |

**Tensor Parallelism:**
```
GPU 0: First half of weight matrix
GPU 1: Second half of weight matrix

Y = XW → Y = [X·W₁, X·W₂] → AllReduce
```

**Pipeline Parallelism:**
```
GPU 0: Layers 1-8    → Micro-batch 1 → GPU 1
GPU 1: Layers 9-16   → Micro-batch 1 → GPU 2
GPU 2: Layers 17-24  → Micro-batch 1 → ...

While GPU 1 processes MB1, GPU 0 starts MB2
```

**Python Code Example:**
```python
import torch
import torch.nn as nn
from torch.distributed import init_process_group

# Method 1: Simple Pipeline Parallelism with PyTorch
class PipelineParallelModel(nn.Module):
    """Split model across 2 GPUs"""
    
    def __init__(self, num_layers=24):
        super().__init__()
        
        # First half on GPU 0
        self.layers_gpu0 = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=1024, nhead=8)
            for _ in range(num_layers // 2)
        ]).to('cuda:0')
        
        # Second half on GPU 1
        self.layers_gpu1 = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=1024, nhead=8)
            for _ in range(num_layers // 2)
        ]).to('cuda:1')
    
    def forward(self, x):
        x = x.to('cuda:0')
        x = self.layers_gpu0(x)
        x = x.to('cuda:1')  # Transfer between GPUs
        x = self.layers_gpu1(x)
        return x

# Method 2: Using DeepSpeed for advanced parallelism
def setup_deepspeed_3d_parallelism():
    """DeepSpeed configuration for 3D parallelism"""
    
    deepspeed_config = {
        "train_batch_size": 256,
        "train_micro_batch_size_per_gpu": 4,
        
        # Zero Redundancy Optimizer (ZeRO) Stage 3
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu"},
            "offload_param": {"device": "cpu"}
        },
        
        # Tensor parallelism
        "tensor_parallel": {
            "enabled": True,
            "tp_size": 4
        },
        
        # Pipeline parallelism
        "pipeline": {
            "enabled": True,
            "pp_size": 4,
            "num_micro_batches": 8
        }
    }
    
    return deepspeed_config

# Method 3: Using Megatron-LM style tensor parallelism
class ColumnParallelLinear(nn.Module):
    """Column-wise tensor parallel linear layer"""
    
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        
        # Each GPU gets a column slice
        self.out_features_per_gpu = out_features // world_size
        self.weight = nn.Parameter(
            torch.randn(in_features, self.out_features_per_gpu)
        )
        self.rank = rank
        self.world_size = world_size
    
    def forward(self, x):
        # Local computation
        local_output = x @ self.weight
        
        # AllGather to combine outputs
        outputs = [torch.zeros_like(local_output) for _ in range(self.world_size)]
        torch.distributed.all_gather(outputs, local_output)
        
        return torch.cat(outputs, dim=-1)

# Using HuggingFace Accelerate for simple multi-GPU
from accelerate import Accelerator

def train_with_accelerate():
    accelerator = Accelerator()
    
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    optimizer = torch.optim.AdamW(model.parameters())
    
    # Automatically handles device placement
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    
    for batch in dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
```

**Interview Tips:**
- Tensor parallel: high communication, good for single-node multi-GPU
- Pipeline parallel: lower communication, good for multi-node
- ZeRO (DeepSpeed) partitions optimizer states, gradients, params
- 3D parallelism (data + tensor + pipeline) for largest models

---

### Question 51
- [ ] Done
**Describe knowledge distillation and how DistilBERT achieves 40% size reduction with minimal quality loss.**

**Answer:**

**Definition:**
**Knowledge distillation** transfers knowledge from a large "teacher" model to a smaller "student" model by training the student to match the teacher's soft probability outputs (not just hard labels). DistilBERT uses this plus architectural reduction (6 layers vs 12) to achieve 97% of BERT's performance with 40% fewer parameters and 60% faster inference.

**How Knowledge Distillation Works:**

**Standard Training:**
$$\mathcal{L} = \text{CrossEntropy}(y_{true}, y_{pred})$$

**Distillation Training:**
$$\mathcal{L} = \alpha \cdot \text{CE}(y_{true}, y_{student}) + (1-\alpha) \cdot \text{KL}(p_{teacher}, p_{student})$$

Where soft probabilities use temperature T:
$$p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

**DistilBERT Techniques:**

| Technique | Description | Impact |
|-----------|-------------|--------|
| **Layer reduction** | 6 layers instead of 12 | 40% size reduction |
| **Soft label distillation** | Match teacher softmax | Preserves knowledge |
| **Cosine embedding loss** | Match hidden representations | Better feature learning |
| **MLM loss** | Standard masked LM | Language understanding |
| **Teacher initialization** | Initialize from every 2nd layer | Better starting point |

**Distillation Loss Components:**
```
L_total = L_mlm + L_distil + L_cosine

L_mlm:    Standard masked language modeling loss
L_distil: KL divergence between teacher/student soft labels
L_cosine: Cosine similarity between hidden states
```

**Python Code Example:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """Knowledge distillation loss"""
    
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        # Hard label loss
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Soft label loss (distillation)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

def train_distillation(teacher_model, student_model, dataloader, epochs=3):
    """Train student model using knowledge distillation"""
    
    teacher_model.eval()
    student_model.train()
    
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
    distill_loss = DistillationLoss(temperature=4.0, alpha=0.5)
    cosine_loss = nn.CosineEmbeddingLoss()
    
    for epoch in range(epochs):
        for batch in dataloader:
            # Get teacher outputs (no gradient)
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch, output_hidden_states=True)
            
            # Get student outputs
            student_outputs = student_model(**batch, output_hidden_states=True)
            
            # Distillation loss on logits
            loss_distil = distill_loss(
                student_outputs.logits,
                teacher_outputs.logits,
                batch['labels']
            )
            
            # Cosine loss on hidden states
            target = torch.ones(batch['input_ids'].size(0)).to(batch['input_ids'].device)
            loss_cosine = cosine_loss(
                student_outputs.hidden_states[-1].mean(dim=1),
                teacher_outputs.hidden_states[-1].mean(dim=1),
                target
            )
            
            # Total loss
            loss = loss_distil + 0.1 * loss_cosine
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

# DistilBERT comparison
comparison = {
    'BERT-base': {'layers': 12, 'params': '110M', 'relative_perf': 1.00},
    'DistilBERT': {'layers': 6, 'params': '66M', 'relative_perf': 0.97},
    'TinyBERT': {'layers': 4, 'params': '14M', 'relative_perf': 0.92},
}
```

**Interview Tips:**
- Temperature controls softness of probability distribution
- Higher T = softer probabilities = more knowledge transfer
- Alpha balances hard labels vs distillation
- DistilBERT was trained on same data as BERT, not task-specific

---

### Question 52
- [ ] Done
**What are RLHF (Reinforcement Learning from Human Feedback) and DPO for aligning LLMs?**

**Answer:**

**Definition:**
**RLHF** trains LLMs to follow human preferences by: (1) collecting human comparisons of model outputs, (2) training a reward model on these preferences, (3) using PPO to optimize the LLM against the reward model. **DPO (Direct Preference Optimization)** simplifies this by directly optimizing the preference objective without a separate reward model, making alignment faster and more stable.

**RLHF Pipeline:**
```
Step 1: Supervised Fine-Tuning (SFT)
        Base LLM → Fine-tune on demonstrations → SFT Model

Step 2: Reward Model Training
        Collect (prompt, response_A, response_B, preference)
        Train classifier: which response is better?

Step 3: RL Optimization (PPO)
        Generate response → Get reward → Update policy
        With KL penalty to stay close to SFT model
```

**DPO vs RLHF:**

| Aspect | RLHF | DPO |
|--------|------|-----|
| **Reward model** | Required | Not needed |
| **RL training** | PPO (complex) | Direct optimization |
| **Stability** | Can be unstable | More stable |
| **Compute** | Higher | Lower |
| **Implementation** | Complex | Simpler |
| **Quality** | Excellent | Comparable |

**Mathematical Formulation:**

**RLHF Objective:**
$$\max_\pi \mathbb{E}_{x \sim D, y \sim \pi}[r(x, y)] - \beta \cdot \text{KL}(\pi || \pi_{ref})$$

**DPO Loss:**
$$\mathcal{L}_{DPO} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

Where:
- $y_w$ = preferred response (winner)
- $y_l$ = dispreferred response (loser)
- $\pi_{ref}$ = reference (SFT) model
- $\beta$ = temperature parameter

**Python Code Example:**
```python
import torch
import torch.nn.functional as F

class DPOTrainer:
    """Direct Preference Optimization trainer"""
    
    def __init__(self, model, ref_model, beta=0.1):
        self.model = model
        self.ref_model = ref_model  # Frozen reference
        self.beta = beta
    
    def compute_log_probs(self, model, input_ids, labels):
        """Compute log probabilities of sequence"""
        outputs = model(input_ids, labels=labels)
        
        # Get per-token log probs
        logits = outputs.logits[:, :-1, :]
        labels = labels[:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = torch.gather(
            log_probs, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding and sum
        mask = (labels != -100).float()
        return (selected_log_probs * mask).sum(dim=-1)
    
    def dpo_loss(self, batch):
        """Compute DPO loss"""
        # Get log probs from policy and reference for both responses
        
        # Preferred (winner) response
        policy_logp_w = self.compute_log_probs(
            self.model, batch['chosen_ids'], batch['chosen_labels']
        )
        with torch.no_grad():
            ref_logp_w = self.compute_log_probs(
                self.ref_model, batch['chosen_ids'], batch['chosen_labels']
            )
        
        # Dispreferred (loser) response
        policy_logp_l = self.compute_log_probs(
            self.model, batch['rejected_ids'], batch['rejected_labels']
        )
        with torch.no_grad():
            ref_logp_l = self.compute_log_probs(
                self.ref_model, batch['rejected_ids'], batch['rejected_labels']
            )
        
        # DPO loss
        pi_logratios = policy_logp_w - policy_logp_l
        ref_logratios = ref_logp_w - ref_logp_l
        
        logits = self.beta * (pi_logratios - ref_logratios)
        loss = -F.logsigmoid(logits).mean()
        
        return loss

# Using TRL library (recommended)
from trl import DPOTrainer, DPOConfig

def train_with_trl_dpo(model, ref_model, train_dataset):
    """Train using TRL's DPO implementation"""
    
    dpo_config = DPOConfig(
        beta=0.1,
        learning_rate=5e-7,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
    )
    
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    return model

# Comparison
alignment_methods = {
    'RLHF (PPO)': {
        'complexity': 'High',
        'stability': 'Medium',
        'used_by': 'ChatGPT, Claude'
    },
    'DPO': {
        'complexity': 'Low', 
        'stability': 'High',
        'used_by': 'Zephyr, Neural-Chat'
    },
    'ORPO': {
        'complexity': 'Low',
        'stability': 'High',
        'used_by': 'Recent models'
    }
}
```

**Interview Tips:**
- RLHF was used for ChatGPT, requires more infrastructure
- DPO is increasingly popular due to simplicity and stability
- Both need preference data: (prompt, chosen, rejected)
- KL penalty prevents model from drifting too far from SFT

---

## Tokenization & Input Processing

### Question 53
- [ ] Done
**What is subword tokenization (BPE, WordPiece, SentencePiece) and why is it used over word-level?**

**Answer:**

**Definition:**
Subword tokenization splits text into units between characters and words (e.g., "playing" → "play" + "##ing"). It handles **open vocabulary** problem — rare and unseen words are split into known subwords. This provides fixed vocabulary size, handles morphology, and eliminates unknown tokens while maintaining meaning.

**Why Not Word-Level:**

| Problem with Word-Level | Subword Solution |
|------------------------|------------------|
| Infinite vocabulary | Fixed vocab (30K-50K) |
| OOV (unknown) tokens | Decompose into known parts |
| Memory explosion | Manageable embedding table |
| Morphology ignored | Captures prefixes/suffixes |
| Typos = unknown | Graceful degradation |

**Subword Algorithms:**

| Algorithm | How It Works | Used In |
|-----------|-------------|---------|
| **BPE** | Iteratively merge frequent pairs | GPT-2, LLaMA |
| **WordPiece** | Maximize likelihood | BERT |
| **Unigram** | Remove tokens that hurt likelihood | T5 |
| **SentencePiece** | Language-agnostic BPE/Unigram | T5, LLaMA |

**BPE Algorithm Steps:**
```
1. Start with character vocabulary: {a, b, c, ..., z}
2. Count pair frequencies: "th"=100, "he"=80, ...
3. Merge most frequent: vocabulary += "th"
4. Repeat until vocab size reached

Example: "lower" → "l", "o", "w", "e", "r"
         "lowest" → "low", "est" (after training)
```

**Python Code Example:**
```python
from transformers import AutoTokenizer

# Compare tokenizers
def compare_tokenizers():
    """Show differences between tokenization methods"""
    
    text = "unbelievably"
    
    # BERT (WordPiece)
    bert_tok = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_tokens = bert_tok.tokenize(text)
    print(f"BERT (WordPiece): {bert_tokens}")
    # ['un', '##bel', '##ie', '##va', '##bly']
    
    # GPT-2 (BPE)
    gpt2_tok = AutoTokenizer.from_pretrained('gpt2')
    gpt2_tokens = gpt2_tok.tokenize(text)
    print(f"GPT-2 (BPE): {gpt2_tokens}")
    # ['un', 'believ', 'ably']
    
    # T5 (SentencePiece)
    t5_tok = AutoTokenizer.from_pretrained('t5-base')
    t5_tokens = t5_tok.tokenize(text)
    print(f"T5 (SentencePiece): {t5_tokens}")
    # ['▁un', 'believ', 'ably']

# Simple BPE implementation
def train_bpe(corpus, vocab_size):
    """Train BPE tokenizer from scratch"""
    
    # Initialize with characters
    vocab = set()
    for word in corpus:
        for char in word:
            vocab.add(char)
    
    # Count word frequencies
    word_freqs = {}
    for word in corpus:
        word_freqs[' '.join(list(word)) + ' </w>'] = word_freqs.get(word, 0) + 1
    
    while len(vocab) < vocab_size:
        # Count pairs
        pairs = {}
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                pairs[pair] = pairs.get(pair, 0) + freq
        
        if not pairs:
            break
        
        # Find most frequent pair
        best_pair = max(pairs, key=pairs.get)
        
        # Merge pair in vocabulary
        new_token = best_pair[0] + best_pair[1]
        vocab.add(new_token)
        
        # Merge pair in words
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            new_word = word.replace(
                f"{best_pair[0]} {best_pair[1]}", new_token
            )
            new_word_freqs[new_word] = freq
        word_freqs = new_word_freqs
    
    return vocab

# Handling OOV
def tokenize_oov_example():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Made-up word
    word = "antiestablishmentarianism"
    tokens = tokenizer.tokenize(word)
    print(f"'{word}' → {tokens}")
    # Splits into known subwords, no [UNK]

compare_tokenizers()
```

**Interview Tips:**
- BPE: bottom-up merging of frequent pairs
- WordPiece: similar but uses likelihood instead of frequency
- SentencePiece: handles raw text (including spaces) — good for multilingual
- Vocab size trade-off: larger = fewer tokens but bigger embedding table

---

### Question 54
- [ ] Done
**How do you handle out-of-vocabulary words and tokenization mismatches across different models?**

**Answer:**

**Definition:**
OOV issues arise when tokens aren't in the model's vocabulary or when transferring data between models with different tokenizers. Handle by: using **subword tokenization** (inherently handles OOV), **adding special tokens**, **vocabulary expansion**, or **re-tokenizing** for cross-model transfer. Never assume tokenizers are compatible between models.

**OOV Handling Strategies:**

| Strategy | When to Use | Implementation |
|----------|-------------|----------------|
| **Subword fallback** | Unknown words | Automatic with BPE/WordPiece |
| **Add tokens** | Domain-specific terms | `tokenizer.add_tokens()` |
| **Resize embeddings** | After adding tokens | `model.resize_token_embeddings()` |
| **Normalize text** | Encoding issues | Unicode normalization |
| **Re-tokenize** | Cross-model transfer | Always use target tokenizer |

**Common Mismatch Issues:**

| Issue | Cause | Solution |
|-------|-------|----------|
| Different vocab sizes | Different training | Use model's own tokenizer |
| Special tokens differ | [CLS] vs <s> | Map correctly |
| Casing | BERT-uncased vs cased | Preprocess consistently |
| Unicode handling | Different normalization | NFC/NFD normalization |
| Padding side | Left vs right | Set explicitly |

**Python Code Example:**
```python
from transformers import AutoTokenizer, AutoModel
import torch

# 1. Adding domain-specific tokens
def add_custom_tokens(model_name, new_tokens):
    """Add tokens to vocabulary and resize embeddings"""
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Check current handling of unknown word
    test_word = "hydroxychloroquine"
    before = tokenizer.tokenize(test_word)
    print(f"Before: {before}")  # Many subwords
    
    # Add custom tokens
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added} tokens")
    
    # Resize model embeddings to match
    model.resize_token_embeddings(len(tokenizer))
    
    # Now it's a single token
    after = tokenizer.tokenize(test_word)
    print(f"After: {after}")  # Single token if added
    
    return tokenizer, model

# 2. Handling cross-model transfer
def transfer_between_models(text, source_model, target_model):
    """Never transfer token IDs between different models"""
    
    # WRONG: Using source tokenizer IDs with target model
    # source_tokenizer = AutoTokenizer.from_pretrained(source_model)
    # target_model = AutoModel.from_pretrained(target_model)
    # ids = source_tokenizer(text)['input_ids']  # WRONG IDs for target!
    
    # CORRECT: Always use target's tokenizer
    target_tokenizer = AutoTokenizer.from_pretrained(target_model)
    target_model = AutoModel.from_pretrained(target_model)
    
    inputs = target_tokenizer(text, return_tensors="pt")
    outputs = target_model(**inputs)
    
    return outputs

# 3. Special token mapping
def map_special_tokens():
    """Different models use different special tokens"""
    
    token_mapping = {
        'bert': {'cls': '[CLS]', 'sep': '[SEP]', 'mask': '[MASK]', 'pad': '[PAD]'},
        'roberta': {'cls': '<s>', 'sep': '</s>', 'mask': '<mask>', 'pad': '<pad>'},
        'gpt2': {'cls': None, 'sep': None, 'mask': None, 'pad': '<|endoftext|>'},
        't5': {'cls': None, 'sep': '</s>', 'mask': '<extra_id_0>', 'pad': '<pad>'},
    }
    
    return token_mapping

# 4. Unicode normalization for OOV
import unicodedata

def normalize_text(text, form='NFC'):
    """Normalize unicode to prevent OOV from encoding issues"""
    
    # Common forms: NFC, NFD, NFKC, NFKD
    normalized = unicodedata.normalize(form, text)
    
    # Also handle special characters
    normalized = normalized.replace('\xa0', ' ')  # Non-breaking space
    normalized = normalized.replace('\u200b', '')  # Zero-width space
    
    return normalized

# 5. Handling subword alignment for NER/tagging
def align_labels_with_tokens(text, labels, tokenizer):
    """Align word-level labels with subword tokens"""
    
    words = text.split()
    encoding = tokenizer(text, return_offsets_mapping=True)
    
    aligned_labels = []
    word_idx = 0
    
    for offset in encoding['offset_mapping']:
        if offset == (0, 0):
            # Special token
            aligned_labels.append(-100)  # Ignore in loss
        elif offset[0] == 0 or text[offset[0]-1] == ' ':
            # Start of new word
            aligned_labels.append(labels[word_idx])
            word_idx += 1
        else:
            # Continuation of word
            aligned_labels.append(-100)  # Or copy previous label
    
    return aligned_labels

# Example usage
text = "New York is a city"
word_labels = [1, 1, 0, 0, 0]  # B-LOC, I-LOC, O, O, O
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
token_labels = align_labels_with_tokens(text, word_labels, tokenizer)
```

**Interview Tips:**
- Never mix tokenizers and models from different sources
- Subword tokenization eliminates most OOV issues
- Adding tokens requires resizing embeddings (random init for new)
- For NER, handle subword-to-word alignment carefully

---

### Question 55
- [ ] Done
**When should you implement custom tokenization versus using the model's default tokenizer?**

**Answer:**

**Definition:**
Use **default tokenizer** in most cases — it's trained with the model and ensures compatibility. Implement **custom tokenization** only when: adding domain-specific vocabulary, handling special formats (code, chemistry), building multi-lingual systems, or training from scratch. Custom tokenization risks breaking model's learned representations.

**Decision Framework:**

| Scenario | Recommendation |
|----------|---------------|
| Standard fine-tuning | Use default tokenizer |
| Domain with special terms | Add tokens to default |
| Training from scratch | Train new tokenizer |
| Special format (SMILES, code) | Consider custom |
| Multi-modal (text + special) | Extend default |

**When to Use Default:**
- Fine-tuning pre-trained models
- Standard NLP tasks
- Embeddings are aligned with tokenization
- No special domain vocabulary

**When to Customize:**
- Training model from scratch
- Domain has many OOV terms
- Special notation (math, chemistry, code)
- Need different granularity (character-level)
- Multi-lingual with unsupported languages

**Python Code Example:**
```python
from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Option 1: Extend existing tokenizer (recommended)
def extend_tokenizer(model_name, domain_terms):
    """Add domain-specific tokens to existing tokenizer"""
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special domain tokens
    num_added = tokenizer.add_tokens(domain_terms)
    
    # Add special tokens (like [ENTITY], [FORMULA])
    special_tokens = {'additional_special_tokens': ['[ENTITY]', '[FORMULA]']}
    tokenizer.add_special_tokens(special_tokens)
    
    print(f"Added {num_added} tokens + {len(special_tokens['additional_special_tokens'])} special")
    
    return tokenizer

# Option 2: Train new tokenizer (for from-scratch training)
def train_custom_tokenizer(corpus_files, vocab_size=30000):
    """Train BPE tokenizer from scratch"""
    
    # Initialize tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    # Pre-tokenizer (how to split before BPE)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=2
    )
    
    # Train on corpus
    tokenizer.train(corpus_files, trainer)
    
    # Save
    tokenizer.save("custom_tokenizer.json")
    
    return tokenizer

# Option 3: Wrap custom preprocessing
class CustomPreprocessingTokenizer:
    """Wrapper for custom preprocessing + standard tokenizer"""
    
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def preprocess(self, text):
        """Custom preprocessing before tokenization"""
        
        # Example: Handle chemical formulas
        import re
        text = re.sub(r'H2O', '[WATER]', text)
        text = re.sub(r'CO2', '[CARBON_DIOXIDE]', text)
        
        # Example: Normalize numbers
        text = re.sub(r'\d+\.\d+', '[FLOAT]', text)
        
        return text
    
    def __call__(self, text, **kwargs):
        processed = self.preprocess(text)
        return self.tokenizer(processed, **kwargs)

# Decision helper
def should_customize_tokenizer(requirements):
    """Decide whether to customize tokenization"""
    
    # Favor default
    if requirements.get('using_pretrained', True):
        return "Use default - pretrained alignment"
    
    # Favor custom
    if requirements.get('training_from_scratch'):
        return "Train new tokenizer on your corpus"
    
    if requirements.get('domain_oov_rate', 0) > 0.1:
        return "Extend default with domain terms"
    
    if requirements.get('special_format'):
        return "Consider custom preprocessing + default tokenizer"
    
    return "Use default tokenizer"

# Example
print(should_customize_tokenizer({
    'using_pretrained': True,
    'domain_oov_rate': 0.05
}))
# Output: "Use default - pretrained alignment"
```

**Interview Tips:**
- Default tokenizer is almost always the right choice
- Adding tokens is safer than replacing tokenizer
- Custom tokenizer breaks pre-trained embeddings
- If you must customize, extend rather than replace

---

## Model Deployment & Production

### Question 56
- [ ] Done
**What techniques help maintain LLM performance when quantizing models for production deployment?**

**Answer:**

**Definition:**
Maintain performance during quantization by: using **calibration data** to set optimal quantization ranges, applying **mixed-precision** (sensitive layers in higher precision), implementing **quantization-aware training (QAT)**, and choosing appropriate methods (GPTQ/AWQ over naive quantization). Post-training quantization works well with careful calibration.

**Quality Preservation Techniques:**

| Technique | Description | Impact |
|-----------|-------------|--------|
| **Calibration data** | Use representative samples to set ranges | Critical for PTQ |
| **Mixed precision** | FP16 for sensitive layers, INT4/8 for others | Preserves quality |
| **AWQ** | Protects important weights | ~0.5% quality loss |
| **GPTQ** | Layer-wise optimal quantization | <1% quality loss |
| **QAT** | Train with quantization | Best quality |
| **Per-channel quant** | Different scale per channel | Better than per-tensor |

**Quantization Quality Preservation Pipeline:**
```
1. Collect calibration data (100-1000 samples)
2. Run forward passes to collect activation statistics
3. Determine optimal quantization ranges
4. Apply quantization with calibration
5. Evaluate on validation set
6. Iterate if quality degraded
```

**Python Code Example:**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def quantize_with_calibration(model_name, calibration_texts):
    """Quantize with proper calibration for quality preservation"""
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    
    # Collect activation statistics
    def calibrate_model(model, calibration_data):
        """Run calibration pass"""
        model.eval()
        for text in calibration_data:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                _ = model(**inputs)
    
    calibrate_model(model, calibration_texts)
    return model

# GPTQ quantization with calibration
def gptq_quantization(model_name, calibration_dataset):
    """GPTQ: high-quality post-training quantization"""
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,          # Group-wise for better quality
        desc_act=True,           # Activation order (better quality)
        damp_percent=0.1,        # Dampening for stability
        sym=False,               # Asymmetric quantization
    )
    
    model = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config)
    
    # Prepare calibration data
    calibration_data = [
        tokenizer(text, return_tensors="pt", truncation=True)
        for text in calibration_dataset
    ]
    
    # Quantize with calibration
    model.quantize(calibration_data)
    
    return model

# AWQ quantization (activation-aware)
def awq_quantization(model_name):
    """AWQ: protects salient weights based on activation"""
    from awq import AutoAWQForCausalLM
    
    model = AutoAWQForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # AWQ identifies important weights via activation analysis
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
    }
    
    # Quantize (includes internal calibration)
    model.quantize(tokenizer, quant_config=quant_config)
    
    return model

# Quality evaluation
def evaluate_quantization_quality(original_model, quantized_model, test_data):
    """Compare quality before and after quantization"""
    
    results = {'original': [], 'quantized': []}
    
    for sample in test_data:
        # Get perplexity or other metrics
        orig_output = original_model.generate(**sample)
        quant_output = quantized_model.generate(**sample)
        
        # Compare outputs
        # ...
    
    return results
```

**Interview Tips:**
- Calibration data should represent deployment distribution
- AWQ typically better quality than GPTQ at same bit-width
- Always evaluate on downstream task, not just perplexity
- Some layers (first, last, attention) more sensitive to quantization

---

### Question 57
- [ ] Done
**How do you implement efficient batch processing for LLM inference in high-throughput systems?**

**Answer:**

**Definition:**
Efficient batch processing for LLM inference involves: **dynamic batching** (group requests of similar lengths), **continuous batching** (add/remove requests during generation), **request scheduling** (prioritize by deadline/length), and **memory optimization** (KV-cache management). Goal: maximize GPU utilization while minimizing latency.

**Key Techniques:**

| Technique | Description | Benefit |
|-----------|-------------|---------|
| **Dynamic batching** | Batch by similar sequence lengths | Reduce padding waste |
| **Continuous batching** | Iteration-level scheduling | 2-3x throughput |
| **Request bucketing** | Group by length ranges | Better utilization |
| **Prefill/decode split** | Separate compute phases | Optimize each phase |
| **PagedAttention** | Virtual memory for KV-cache | More concurrent requests |
| **Speculative decoding** | Draft model + verification | Faster generation |

**Batch Processing Architecture:**
```
Incoming Requests → Request Queue → Scheduler → Batch Former → GPU Inference → Results
                         ↑                          ↓
                    Priority Logic           Continuous Updates
```

**Python Code Example:**
```python
import asyncio
import torch
from queue import PriorityQueue
from dataclasses import dataclass
from typing import List
import time

@dataclass
class InferenceRequest:
    id: str
    prompt: str
    max_tokens: int
    priority: int
    timestamp: float
    
    def __lt__(self, other):
        return self.priority < other.priority

class EfficientBatchProcessor:
    def __init__(self, model, tokenizer, max_batch_size=32, max_wait_time=0.1):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.request_queue = PriorityQueue()
        self.results = {}
    
    async def add_request(self, prompt: str, max_tokens: int, priority: int = 0):
        """Add request to queue"""
        request = InferenceRequest(
            id=f"req_{time.time()}",
            prompt=prompt,
            max_tokens=max_tokens,
            priority=priority,
            timestamp=time.time()
        )
        self.request_queue.put(request)
        return request.id
    
    def _bucket_by_length(self, requests: List[InferenceRequest]):
        """Group requests by similar lengths"""
        buckets = {}
        for req in requests:
            length = len(self.tokenizer.encode(req.prompt))
            bucket_key = (length // 256) * 256
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(req)
        return buckets
    
    def _form_batch(self, requests: List[InferenceRequest]):
        """Form batch with minimal padding"""
        prompts = [r.prompt for r in requests]
        encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        return encodings
    
    async def process_batch(self, requests: List[InferenceRequest]):
        """Process a batch of requests"""
        if not requests:
            return {}
        
        batch = self._form_batch(requests)
        batch = {k: v.cuda() for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **batch,
                max_new_tokens=max(r.max_tokens for r in requests),
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        results = {}
        for i, req in enumerate(requests):
            decoded = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            results[req.id] = decoded
        
        return results

# Continuous batching (iteration-level scheduling)
class ContinuousBatcher:
    """Add/remove requests during generation"""
    
    def __init__(self, model, tokenizer, max_batch_size=64):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.active_sequences = {}
    
    def add_sequence(self, seq_id, prompt):
        """Add new sequence to active batch"""
        tokens = self.tokenizer.encode(prompt)
        self.active_sequences[seq_id] = {
            'tokens': tokens,
            'kv_cache': None,
            'generated': 0,
            'max_tokens': 100
        }
    
    def remove_sequence(self, seq_id):
        """Remove completed sequence"""
        if seq_id in self.active_sequences:
            del self.active_sequences[seq_id]
    
    async def generation_step(self):
        """One step for all active sequences"""
        if not self.active_sequences:
            return
        
        # Batch all active sequences
        batch_tokens = []
        batch_ids = []
        
        for seq_id, state in self.active_sequences.items():
            batch_tokens.append(state['tokens'][-1:])
            batch_ids.append(seq_id)
        
        # Single forward pass for all sequences
        next_tokens = self._model_forward(batch_tokens)
        
        # Update and check completion
        completed = []
        for i, seq_id in enumerate(batch_ids):
            state = self.active_sequences[seq_id]
            state['tokens'].append(next_tokens[i])
            state['generated'] += 1
            
            if (next_tokens[i] == self.tokenizer.eos_token_id or 
                state['generated'] >= state['max_tokens']):
                completed.append(seq_id)
        
        for seq_id in completed:
            self.remove_sequence(seq_id)
```

**Interview Tips:**
- Continuous batching can double throughput vs static batching
- PagedAttention (vLLM) enables efficient memory management
- Prefill is compute-bound, decode is memory-bound - optimize separately
- Monitor GPU utilization and queue depth for capacity planning

---

### Question 58
- [ ] Done
**What are the strategies for A/B testing different LLM variants in production environments?**

**Answer:**

**Definition:**
A/B testing LLMs compares model variants by **traffic splitting**, measuring **quality metrics** (relevance, accuracy), **latency**, **cost**, and **user satisfaction**. Key challenges include: accounting for **prompt sensitivity**, handling **non-deterministic outputs**, and ensuring **statistical significance** with high variance responses.

**A/B Testing Framework:**

| Component | Consideration |
|-----------|---------------|
| **Traffic allocation** | Random user assignment, consistent per-user |
| **Metrics** | Quality, latency, throughput, cost, user engagement |
| **Sample size** | Higher variance → need more samples |
| **Duration** | Account for temporal patterns |
| **Rollback** | Quick switch mechanism if issues arise |

**LLM-Specific Challenges:**

| Challenge | Solution |
|-----------|----------|
| **Non-determinism** | Multiple evaluations per prompt, use temperature=0 |
| **Subjective quality** | Human evaluation + automated metrics |
| **Long outputs** | Aggregate metrics (BLEU, ROUGE, semantic similarity) |
| **Prompt sensitivity** | Test across diverse prompt distribution |
| **Cost differences** | Include cost in decision criteria |

**Python Code Example:**
```python
import random
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy import stats

@dataclass
class ExperimentVariant:
    name: str
    model_id: str
    traffic_percentage: float
    config: Dict

@dataclass
class ExperimentResult:
    variant: str
    latency_ms: float
    quality_score: float
    cost: float
    user_feedback: Optional[int] = None

class LLMABTestFramework:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.variants: Dict[str, ExperimentVariant] = {}
        self.results: List[ExperimentResult] = []
    
    def add_variant(self, variant: ExperimentVariant):
        """Add a model variant to the experiment"""
        self.variants[variant.name] = variant
    
    def get_variant_for_user(self, user_id: str) -> str:
        """Deterministic variant assignment (consistent per user)"""
        # Hash user_id for consistent assignment
        hash_val = int(hashlib.md5(
            f"{self.experiment_name}:{user_id}".encode()
        ).hexdigest(), 16)
        
        normalized = (hash_val % 10000) / 10000
        
        cumulative = 0.0
        for name, variant in self.variants.items():
            cumulative += variant.traffic_percentage
            if normalized < cumulative:
                return name
        
        return list(self.variants.keys())[-1]
    
    def log_result(self, result: ExperimentResult):
        """Log experiment result"""
        self.results.append(result)
    
    def analyze_results(self) -> Dict:
        """Statistical analysis of results"""
        variant_results = {}
        for variant_name in self.variants:
            variant_data = [r for r in self.results if r.variant == variant_name]
            if variant_data:
                variant_results[variant_name] = {
                    'n': len(variant_data),
                    'latency': {
                        'mean': np.mean([r.latency_ms for r in variant_data]),
                        'p50': np.percentile([r.latency_ms for r in variant_data], 50),
                        'p99': np.percentile([r.latency_ms for r in variant_data], 99)
                    },
                    'quality': {
                        'mean': np.mean([r.quality_score for r in variant_data]),
                        'std': np.std([r.quality_score for r in variant_data])
                    },
                    'cost': {
                        'mean': np.mean([r.cost for r in variant_data]),
                        'total': sum(r.cost for r in variant_data)
                    }
                }
        
        return variant_results
    
    def statistical_significance(self, metric: str = 'quality_score', 
                                  alpha: float = 0.05) -> Dict:
        """Calculate statistical significance between variants"""
        variant_names = list(self.variants.keys())
        if len(variant_names) < 2:
            return {}
        
        control = [r for r in self.results if r.variant == variant_names[0]]
        treatment = [r for r in self.results if r.variant == variant_names[1]]
        
        control_values = [getattr(r, metric) for r in control]
        treatment_values = [getattr(r, metric) for r in treatment]
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(control_values) + np.var(treatment_values)) / 2
        )
        cohens_d = (np.mean(treatment_values) - np.mean(control_values)) / pooled_std
        
        return {
            'control': variant_names[0],
            'treatment': variant_names[1],
            'p_value': p_value,
            'significant': p_value < alpha,
            'effect_size': cohens_d,
            'control_mean': np.mean(control_values),
            'treatment_mean': np.mean(treatment_values),
            'lift': (np.mean(treatment_values) - np.mean(control_values)) / np.mean(control_values)
        }

# Usage example
experiment = LLMABTestFramework("model_comparison_v1")

# Add variants
experiment.add_variant(ExperimentVariant(
    name="control",
    model_id="gpt-3.5-turbo",
    traffic_percentage=0.5,
    config={"temperature": 0.7}
))
experiment.add_variant(ExperimentVariant(
    name="treatment",
    model_id="gpt-4",
    traffic_percentage=0.5,
    config={"temperature": 0.7}
))

# Route users
user_variant = experiment.get_variant_for_user("user123")
print(f"User assigned to: {user_variant}")
```

**Interview Tips:**
- Always use consistent user assignment (hash-based) for valid comparison
- Need larger sample sizes than traditional A/B tests due to output variance
- Combine automated metrics with human evaluation
- Consider multi-armed bandit for faster convergence

---

### Question 59
- [ ] Done
**How do you handle version control and model lifecycle management for LLM deployments?**

**Answer:**

**Definition:**
LLM lifecycle management tracks **model artifacts** (weights, configs, tokenizers), **training metadata** (data versions, hyperparameters), **prompt templates**, and **evaluation results** across development to production. Key tools: **MLflow**, **DVC**, **Weights & Biases**, and **model registries** for versioning and lineage tracking.

**Lifecycle Stages:**

```
Development → Training → Evaluation → Staging → Production → Monitoring → Retirement
     ↓            ↓           ↓            ↓           ↓            ↓
  Git/DVC    W&B/MLflow   Benchmarks   Canary     Blue-Green   Drift Detection
```

**Version Control Components:**

| Component | Version Strategy | Tool |
|-----------|-----------------|------|
| **Code** | Git branches/tags | Git |
| **Data** | Hash-based versions | DVC, Delta Lake |
| **Model weights** | Model registry | MLflow, HuggingFace Hub |
| **Configs** | Config files in Git | Hydra, YAML |
| **Prompts** | Template versioning | Git, prompt registry |
| **Dependencies** | Lock files | Poetry, pip freeze |

**Model Registry Structure:**

| Field | Example |
|-------|---------|
| **Model name** | `llm-chatbot-v2` |
| **Version** | `2.1.0` |
| **Stage** | `production` |
| **Training data** | `dataset-v3.2` |
| **Base model** | `llama-2-7b` |
| **Metrics** | `{"perplexity": 3.2, "accuracy": 0.87}` |
| **Artifact URI** | `s3://models/llm-chatbot-v2/` |

**Python Code Example:**
```python
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import json
import hashlib
from pathlib import Path

class LLMVersionManager:
    def __init__(self, tracking_uri: str, model_name: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.model_name = model_name
    
    def register_model(self, model_path: str, metadata: dict, 
                       metrics: dict, tags: dict = None):
        """Register a new model version"""
        
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params({
                'base_model': metadata.get('base_model'),
                'training_data_version': metadata.get('data_version'),
                'num_epochs': metadata.get('epochs'),
                'learning_rate': metadata.get('lr')
            })
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model artifacts
            mlflow.log_artifacts(model_path, "model")
            
            # Log config
            mlflow.log_dict(metadata, "config.json")
            
            # Register model
            model_uri = f"runs:/{run.info.run_id}/model"
            result = mlflow.register_model(model_uri, self.model_name)
            
            return result.version
    
    def promote_to_stage(self, version: str, stage: str):
        """Promote model to staging/production"""
        # Valid stages: Staging, Production, Archived
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=stage
        )
    
    def get_production_model(self):
        """Get current production model"""
        versions = self.client.get_latest_versions(
            self.model_name, stages=["Production"]
        )
        if versions:
            return versions[0]
        return None
    
    def compare_versions(self, v1: str, v2: str) -> dict:
        """Compare two model versions"""
        mv1 = self.client.get_model_version(self.model_name, v1)
        mv2 = self.client.get_model_version(self.model_name, v2)
        
        run1 = self.client.get_run(mv1.run_id)
        run2 = self.client.get_run(mv2.run_id)
        
        return {
            'v1_metrics': run1.data.metrics,
            'v2_metrics': run2.data.metrics,
            'v1_params': run1.data.params,
            'v2_params': run2.data.params
        }

# Prompt template versioning
class PromptVersionManager:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def save_prompt(self, name: str, template: str, metadata: dict = None):
        """Save versioned prompt template"""
        # Create hash for content
        content_hash = hashlib.sha256(template.encode()).hexdigest()[:8]
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        prompt_data = {
            'name': name,
            'version': version,
            'hash': content_hash,
            'template': template,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat()
        }
        
        filename = f"{name}_{version}_{content_hash}.json"
        filepath = self.storage_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(prompt_data, f, indent=2)
        
        return version
    
    def get_latest_prompt(self, name: str) -> dict:
        """Get latest version of a prompt"""
        prompts = list(self.storage_path.glob(f"{name}_*.json"))
        if not prompts:
            return None
        
        # Sort by timestamp in filename
        latest = sorted(prompts)[-1]
        with open(latest) as f:
            return json.load(f)

# Deployment configuration
class DeploymentConfig:
    """Track deployment configurations"""
    
    def __init__(self, model_version: str, prompt_version: str):
        self.config = {
            'model_version': model_version,
            'prompt_version': prompt_version,
            'deployed_at': datetime.now().isoformat(),
            'environment': {}
        }
    
    def add_runtime_config(self, **kwargs):
        self.config['environment'].update(kwargs)
    
    def to_dict(self):
        return self.config
```

**Interview Tips:**
- Always version prompts alongside models — they're coupled
- Use semantic versioning (major.minor.patch) for models
- Maintain rollback capability with previous production version
- Track data lineage for reproducibility and compliance

---

### Question 60
- [ ] Done
**What monitoring strategies help detect performance degradation in deployed LLMs?**

**Answer:**

**Definition:**
Monitor LLMs via: **latency/throughput metrics**, **quality indicators** (output length, refusal rates), **semantic drift** (embedding distribution shifts), **user feedback signals**, and **cost tracking**. Key challenge: no ground truth labels in production, so rely on proxy metrics and anomaly detection.

**Monitoring Dimensions:**

| Category | Metrics | Alert Threshold |
|----------|---------|-----------------|
| **Latency** | P50, P99, TTFT, TPS | >2x baseline |
| **Throughput** | Requests/sec, tokens/sec | <80% capacity |
| **Quality** | Output length, empty responses | >5% empty |
| **Errors** | Rate limits, timeouts, failures | >1% error rate |
| **Cost** | Tokens/request, $/request | >20% budget |
| **Semantic** | Embedding drift, topic shift | Statistically significant |

**Quality Proxy Metrics:**

| Metric | What It Detects |
|--------|-----------------|
| **Response length distribution** | Model producing shorter/longer outputs |
| **Refusal rate** | Model declining to answer more often |
| **Repetition score** | Degenerate outputs |
| **Toxicity scores** | Safety degradation |
| **User regeneration rate** | Dissatisfaction signal |
| **Embedding variance** | Output diversity changes |

**Python Code Example:**
```python
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

@dataclass
class LLMMetric:
    timestamp: datetime
    latency_ms: float
    input_tokens: int
    output_tokens: int
    success: bool
    response_length: int
    embedding: Optional[np.ndarray] = None

class LLMMonitor:
    def __init__(self, window_size: int = 1000, alert_threshold: float = 2.0):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.metrics = deque(maxlen=window_size)
        self.baseline = None
        self.logger = logging.getLogger(__name__)
    
    def log_request(self, metric: LLMMetric):
        """Log a single request metric"""
        self.metrics.append(metric)
        self._check_alerts(metric)
    
    def set_baseline(self, metrics: List[LLMMetric]):
        """Set baseline from historical data"""
        self.baseline = {
            'latency_mean': np.mean([m.latency_ms for m in metrics]),
            'latency_std': np.std([m.latency_ms for m in metrics]),
            'output_length_mean': np.mean([m.response_length for m in metrics]),
            'output_length_std': np.std([m.response_length for m in metrics]),
            'success_rate': np.mean([m.success for m in metrics]),
            'tokens_per_request': np.mean([m.output_tokens for m in metrics])
        }
        
        # Baseline embedding distribution
        embeddings = [m.embedding for m in metrics if m.embedding is not None]
        if embeddings:
            self.baseline['embedding_mean'] = np.mean(embeddings, axis=0)
            self.baseline['embedding_std'] = np.std(embeddings, axis=0)
    
    def _check_alerts(self, metric: LLMMetric):
        """Check for anomalies"""
        if not self.baseline:
            return
        
        alerts = []
        
        # Latency alert
        z_score = (metric.latency_ms - self.baseline['latency_mean']) / self.baseline['latency_std']
        if abs(z_score) > self.alert_threshold:
            alerts.append(f"Latency anomaly: {metric.latency_ms}ms (z={z_score:.2f})")
        
        # Output length alert
        if self.baseline['output_length_std'] > 0:
            z_score = (metric.response_length - self.baseline['output_length_mean']) / self.baseline['output_length_std']
            if abs(z_score) > self.alert_threshold:
                alerts.append(f"Output length anomaly: {metric.response_length} chars")
        
        for alert in alerts:
            self.logger.warning(alert)
    
    def get_current_stats(self) -> Dict:
        """Get current window statistics"""
        recent = list(self.metrics)
        if not recent:
            return {}
        
        return {
            'latency_p50': np.percentile([m.latency_ms for m in recent], 50),
            'latency_p99': np.percentile([m.latency_ms for m in recent], 99),
            'success_rate': np.mean([m.success for m in recent]),
            'avg_output_tokens': np.mean([m.output_tokens for m in recent]),
            'requests_count': len(recent),
            'error_count': sum(1 for m in recent if not m.success)
        }
    
    def detect_drift(self, recent_embeddings: List[np.ndarray]) -> Dict:
        """Detect semantic drift via embedding distribution"""
        if not self.baseline or 'embedding_mean' not in self.baseline:
            return {'drift_detected': False, 'reason': 'No baseline'}
        
        recent_mean = np.mean(recent_embeddings, axis=0)
        
        # Cosine similarity between baseline and recent
        cos_sim = np.dot(self.baseline['embedding_mean'], recent_mean) / (
            np.linalg.norm(self.baseline['embedding_mean']) * np.linalg.norm(recent_mean)
        )
        
        # KL divergence approximation
        kl_div = np.sum(
            (recent_mean - self.baseline['embedding_mean'])**2 / 
            (2 * self.baseline['embedding_std']**2 + 1e-8)
        )
        
        drift_detected = cos_sim < 0.95 or kl_div > 10
        
        return {
            'drift_detected': drift_detected,
            'cosine_similarity': float(cos_sim),
            'kl_divergence': float(kl_div)
        }

class QualityMonitor:
    """Monitor output quality proxies"""
    
    def __init__(self):
        self.refusal_patterns = [
            "I cannot", "I'm not able to", "I don't have access",
            "As an AI", "I apologize, but"
        ]
    
    def analyze_response(self, response: str) -> Dict:
        """Analyze response quality signals"""
        return {
            'length': len(response),
            'word_count': len(response.split()),
            'is_refusal': any(p.lower() in response.lower() for p in self.refusal_patterns),
            'is_empty': len(response.strip()) == 0,
            'repetition_score': self._calculate_repetition(response)
        }
    
    def _calculate_repetition(self, text: str, n: int = 4) -> float:
        """Detect repetitive n-grams"""
        words = text.lower().split()
        if len(words) < n:
            return 0.0
        
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        unique_ngrams = set(ngrams)
        
        if len(ngrams) == 0:
            return 0.0
        
        return 1 - (len(unique_ngrams) / len(ngrams))

# Dashboard metrics aggregation
def generate_dashboard_metrics(monitor: LLMMonitor) -> Dict:
    """Generate metrics for monitoring dashboard"""
    stats = monitor.get_current_stats()
    
    return {
        'performance': {
            'latency_p50_ms': stats.get('latency_p50', 0),
            'latency_p99_ms': stats.get('latency_p99', 0),
            'throughput_rps': stats.get('requests_count', 0) / 60
        },
        'reliability': {
            'success_rate': stats.get('success_rate', 0),
            'error_rate': 1 - stats.get('success_rate', 1)
        },
        'cost': {
            'avg_tokens_per_request': stats.get('avg_output_tokens', 0)
        }
    }
```

**Interview Tips:**
- No ground truth in production → rely on proxy metrics
- User feedback (regeneration, thumbs down) is strongest signal
- Monitor embedding distributions for semantic drift
- Set up automated alerting with appropriate thresholds

---

## Advanced Topics

### Question 61
- [ ] Done
**What is the Vision Transformer (ViT) and how does it apply self-attention to images?**

**Answer:**

**Definition:**
Vision Transformer (ViT) applies the Transformer architecture to images by: **splitting images into fixed-size patches** (16×16), **linearly embedding each patch** into a vector, adding **positional embeddings**, and processing through standard Transformer encoder layers. It demonstrates that pure attention (no convolutions) can achieve SOTA on image classification with sufficient data.

**ViT Architecture:**

```
Image (224×224×3) → Split into patches (14×14 patches of 16×16)
                          ↓
              Linear projection (patch → D dimensions)
                          ↓
              Add positional embeddings + [CLS] token
                          ↓
              Transformer Encoder (L layers)
                          ↓
              [CLS] token → Classification head
```

**Key Components:**

| Component | Description | Dimensions |
|-----------|-------------|------------|
| **Patch size** | Image split into P×P patches | Typically 16×16 |
| **Patch embedding** | Linear projection | (P²·C) → D |
| **Positional embedding** | Learnable 1D positions | N+1 positions |
| **[CLS] token** | Classification token | D dimensions |
| **Encoder** | Standard Transformer | L layers |

**Mathematical Formulation:**

For image $x \in \mathbb{R}^{H \times W \times C}$:

1. **Patch extraction**: Split into $N = \frac{HW}{P^2}$ patches

2. **Linear embedding**: $z_0 = [x_{class}; x_1^pE; x_2^pE; ...; x_N^pE] + E_{pos}$
   - $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ is patch embedding matrix
   - $E_{pos} \in \mathbb{R}^{(N+1) \times D}$ is positional embedding

3. **Transformer layers**: $z_l = \text{TransformerBlock}(z_{l-1})$

4. **Classification**: $y = \text{MLP}(z_L^0)$ using [CLS] token

**Python Code Example:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """Split image into patches and embed"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.projection(x)
        # Flatten: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings (learnable)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification from [CLS] token
        x = self.norm(x)
        cls_output = x[:, 0]  # (B, embed_dim)
        
        return self.head(cls_output)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x

# Example usage
vit = VisionTransformer(
    img_size=224, patch_size=16, num_classes=1000,
    embed_dim=768, depth=12, num_heads=12
)
images = torch.randn(4, 3, 224, 224)
outputs = vit(images)
print(f"Output shape: {outputs.shape}")  # (4, 1000)
```

**Interview Tips:**
- ViT requires large datasets (ImageNet-21k) to outperform CNNs
- DeiT adds distillation for training on smaller datasets
- Hybrid approaches (CNN stem + Transformer) often work well
- Attention in ViT captures global context from first layer (unlike CNNs)

---

### Question 62
- [ ] Done
**How do Transformers handle multilingual and cross-lingual transfer learning?**

**Answer:**

**Definition:**
Multilingual Transformers learn **shared representations across languages** by training on multilingual corpora. Cross-lingual transfer enables: training on high-resource languages (English) and applying to low-resource languages. Key mechanisms: **shared subword vocabulary**, **aligned embedding spaces**, and **attention patterns that generalize across languages**.

**Multilingual Models:**

| Model | Languages | Approach | Use Case |
|-------|-----------|----------|----------|
| **mBERT** | 104 | Shared vocabulary, joint training | Classification, NER |
| **XLM-RoBERTa** | 100 | Larger data, improved training | Cross-lingual tasks |
| **mT5** | 101 | Encoder-decoder, multilingual | Generation, translation |
| **BLOOM** | 46 | Decoder-only, multilingual | Text generation |

**Cross-Lingual Transfer Mechanisms:**

| Mechanism | Description | Importance |
|-----------|-------------|------------|
| **Shared vocabulary** | Subwords shared across languages | Critical for transfer |
| **Shared parameters** | Same model weights for all languages | Enables knowledge sharing |
| **Anchor words** | Cognates, numbers, named entities | Align embedding spaces |
| **Parallel data** | Translation pairs for alignment | Optional but helpful |

**Transfer Learning Approaches:**

```
Zero-shot:  Train on English → Apply directly to German
Few-shot:   Train on English + few German examples → Apply to German  
Translate-train: Translate training data → Train on target language
Translate-test:  Translate test data → Use English model
```

**Python Code Example:**
```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
import torch

class CrossLingualClassifier:
    def __init__(self, model_name="xlm-roberta-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.model_name = model_name
    
    def train_on_source_language(self, train_texts, train_labels, num_labels=2):
        """Train on high-resource language (e.g., English)"""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        
        # Tokenize
        encodings = self.tokenizer(
            train_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(train_labels)
        )
        
        # Training
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            learning_rate=2e-5
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )
        trainer.train()
    
    def zero_shot_predict(self, target_texts):
        """Apply directly to target language (zero-shot transfer)"""
        self.model.eval()
        
        encodings = self.tokenizer(
            target_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        return predictions.tolist()
    
    def few_shot_adapt(self, target_texts, target_labels, epochs=1):
        """Fine-tune with few target language examples"""
        encodings = self.tokenizer(
            target_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(target_labels)
        )
        
        # Light fine-tuning
        training_args = TrainingArguments(
            output_dir="./results_adapted",
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            learning_rate=1e-5  # Lower LR for adaptation
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )
        trainer.train()

# Language-specific adapter approach
from transformers import AutoModel

class LanguageAdapters:
    """Use adapters for efficient multilingual adaptation"""
    
    def __init__(self, base_model="xlm-roberta-base"):
        self.base_model = AutoModel.from_pretrained(base_model)
        self.language_adapters = {}
    
    def add_language_adapter(self, language: str, adapter_dim: int = 64):
        """Add language-specific adapter"""
        # Adapter: down-project → nonlinearity → up-project
        hidden_size = self.base_model.config.hidden_size
        
        adapter = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, adapter_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(adapter_dim, hidden_size)
        )
        
        self.language_adapters[language] = adapter
    
    def forward_with_adapter(self, inputs, language: str):
        """Forward pass with language-specific adapter"""
        # Base model forward
        outputs = self.base_model(**inputs)
        hidden_states = outputs.last_hidden_state
        
        # Apply language adapter
        if language in self.language_adapters:
            adapter = self.language_adapters[language]
            adapted = adapter(hidden_states)
            hidden_states = hidden_states + adapted  # Residual
        
        return hidden_states

# Evaluation across languages
def evaluate_cross_lingual(model, tokenizer, test_data_by_language):
    """Evaluate model performance across languages"""
    results = {}
    
    for language, (texts, labels) in test_data_by_language.items():
        predictions = model.predict(texts, tokenizer)
        accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)
        results[language] = accuracy
    
    # Calculate transfer gap
    if 'en' in results:
        for lang in results:
            if lang != 'en':
                results[f'{lang}_gap'] = results['en'] - results[lang]
    
    return results
```

**Interview Tips:**
- XLM-RoBERTa outperforms mBERT due to more data and better training
- Transfer works best for related languages (same script, similar structure)
- Subword overlap is crucial — languages with unique scripts transfer less
- For production, consider language-specific adapters for efficiency

---

### Question 63
- [ ] Done
**Explain the concept of emergent abilities in large language models at scale.**

**Answer:**

**Definition:**
Emergent abilities are capabilities that appear **abruptly at certain model scales** (parameters, compute, data) that weren't present or predictable from smaller models. Examples include: **chain-of-thought reasoning**, **in-context learning**, **multi-step arithmetic**, and **code generation**. These abilities don't improve gradually — they emerge suddenly past a threshold.

**Key Characteristics:**

| Characteristic | Description |
|----------------|-------------|
| **Non-linear appearance** | Performance jumps at scale threshold |
| **Unpredictable** | Can't be extrapolated from smaller models |
| **Task-specific** | Different abilities emerge at different scales |
| **Compositionality** | Combining learned patterns in novel ways |

**Examples of Emergent Abilities:**

| Ability | Approximate Threshold | Model Size |
|---------|----------------------|------------|
| **In-context learning** | ~1B parameters | GPT-3 (175B) |
| **Chain-of-thought** | ~100B parameters | PaLM, GPT-4 |
| **Multi-digit arithmetic** | ~10B parameters | Various |
| **Word unscrambling** | ~100B parameters | PaLM |
| **Code generation** | ~1B parameters | Codex, StarCoder |

**Scaling Laws vs Emergence:**

```
Scaling Laws (Predictable):
Loss ∝ N^(-α) + D^(-β) + C^(-γ)
- N: parameters, D: data, C: compute
- Smooth, predictable improvement

Emergent Abilities (Unpredictable):
- Near-zero performance → sudden jump
- Phase transition at critical scale
- Not captured by loss curves
```

**Theories of Emergence:**

| Theory | Explanation |
|--------|-------------|
| **Phase transition** | Sufficient capacity enables new computations |
| **Skill composition** | Combining sub-skills learned separately |
| **Metric artifact** | Discrete metrics mask gradual improvement |
| **Distribution coverage** | Scale enables rare pattern coverage |

**Python Code Example:**
```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_emergence(model_sizes, task_threshold, noise=0.05):
    """Simulate emergent ability behavior"""
    performances = []
    
    for size in model_sizes:
        if size < task_threshold:
            # Below threshold: near-random performance
            perf = 0.1 + np.random.normal(0, noise)
        else:
            # Above threshold: rapid improvement then saturation
            scale_factor = (size - task_threshold) / task_threshold
            perf = 0.9 * (1 - np.exp(-scale_factor)) + np.random.normal(0, noise)
        
        performances.append(max(0, min(1, perf)))
    
    return performances

# Simulate different emergent abilities
model_sizes = np.logspace(8, 12, 50)  # 100M to 1T parameters

abilities = {
    'In-context learning': 1e9,
    'Chain-of-thought': 1e11,
    'Multi-digit arithmetic': 1e10,
    'Code generation': 5e9
}

# Analyze emergence patterns
class EmergenceAnalyzer:
    def __init__(self):
        self.results = {}
    
    def detect_emergence_point(self, sizes, performances, threshold=0.5):
        """Detect where ability emerges"""
        for i, (size, perf) in enumerate(zip(sizes, performances)):
            if perf > threshold:
                return size
        return None
    
    def measure_emergence_sharpness(self, sizes, performances):
        """How sharp is the transition?"""
        # Find 10% to 90% performance range
        p10_idx = next((i for i, p in enumerate(performances) if p > 0.1), None)
        p90_idx = next((i for i, p in enumerate(performances) if p > 0.9), None)
        
        if p10_idx is None or p90_idx is None:
            return None
        
        # Sharpness = ratio of scale range
        return sizes[p90_idx] / sizes[p10_idx]
    
    def analyze_ability(self, name, sizes, performances):
        """Full emergence analysis"""
        emergence_point = self.detect_emergence_point(sizes, performances)
        sharpness = self.measure_emergence_sharpness(sizes, performances)
        
        self.results[name] = {
            'emergence_point': emergence_point,
            'sharpness': sharpness,
            'max_performance': max(performances)
        }
        
        return self.results[name]

# Research implications
def implications_for_scaling():
    """Key implications of emergence"""
    return {
        'capability_planning': 
            "Can't predict new abilities from smaller model eval",
        'safety_concerns': 
            "Dangerous capabilities might emerge unexpectedly",
        'eval_design': 
            "Need comprehensive evals at each scale",
        'resource_allocation': 
            "Justify large compute for potential emergent gains",
        'metric_choice': 
            "Use continuous metrics (perplexity) alongside discrete"
    }

# Distinguish from smooth scaling
def scaling_law_vs_emergence(task, model_sizes, performances):
    """
    Scaling law: log-linear improvement
    Emergence: step function behavior
    """
    log_sizes = np.log10(model_sizes)
    
    # Fit linear model (scaling law hypothesis)
    coeffs = np.polyfit(log_sizes, performances, 1)
    predicted_linear = np.polyval(coeffs, log_sizes)
    
    # Calculate residuals
    residuals = performances - predicted_linear
    
    # High residual variance suggests emergence (non-linear)
    residual_variance = np.var(residuals)
    
    # Detect sharp jump
    gradients = np.diff(performances)
    max_gradient = np.max(gradients)
    mean_gradient = np.mean(np.abs(gradients))
    
    is_emergent = max_gradient > 5 * mean_gradient
    
    return {
        'is_emergent': is_emergent,
        'residual_variance': residual_variance,
        'max_gradient_ratio': max_gradient / (mean_gradient + 1e-8)
    }
```

**Interview Tips:**
- Debate exists on whether emergence is "real" or measurement artifact
- Some argue continuous metrics (loss) don't show emergence
- Important for capability forecasting and AI safety
- In-context learning is clearest example of emergence

---

### Question 64
- [ ] Done
**What are the limitations and failure modes of Transformer-based models?**

**Answer:**

**Definition:**
Transformer limitations include: **quadratic attention complexity** O(n²), **position extrapolation** (fails on longer sequences than trained), **hallucination** (confident but incorrect outputs), **lack of true reasoning** (pattern matching, not logic), and **context length constraints**. Understanding failures is critical for production deployment.

**Categories of Limitations:**

| Category | Limitation | Impact |
|----------|------------|--------|
| **Computational** | O(n²) attention, memory | Limits sequence length |
| **Positional** | Poor length generalization | Fails on longer inputs |
| **Reasoning** | No true logic, arithmetic | Unreliable calculations |
| **Factual** | Hallucinations, outdated facts | Requires verification |
| **Robustness** | Prompt sensitivity, adversarial | Inconsistent outputs |

**Failure Modes in Detail:**

| Failure Mode | Description | Example |
|--------------|-------------|---------|
| **Hallucination** | Generates plausible but false info | Fabricated citations |
| **Sycophancy** | Agrees with incorrect user statements | Confirms wrong facts |
| **Position bias** | Ignores middle of long contexts | "Lost in the middle" |
| **Arithmetic errors** | Fails multi-digit computation | 347 × 892 wrong |
| **Logical fallacies** | Invalid reasoning chains | Affirming the consequent |
| **Prompt injection** | Follows malicious instructions | Ignores system prompt |

**Quadratic Complexity Problem:**

```
Self-Attention: O(n² · d)
- n: sequence length
- d: hidden dimension

Memory: O(n²) for attention matrix

Sequence Length | Attention Matrix Size
256            | 65K elements
1K             | 1M elements
4K             | 16M elements
32K            | 1B elements
```

**Python Code Example:**
```python
import numpy as np
from typing import List, Dict

class TransformerLimitationAnalyzer:
    """Analyze and detect Transformer failure modes"""
    
    def __init__(self):
        self.known_issues = []
    
    def check_length_extrapolation(self, model, train_length: int, 
                                    test_lengths: List[int]) -> Dict:
        """Test model on lengths beyond training"""
        results = {}
        
        for length in test_lengths:
            # Performance typically degrades beyond train length
            if length <= train_length:
                degradation = 0
            else:
                # Models often fail 1.5-2x beyond training length
                ratio = length / train_length
                degradation = max(0, (ratio - 1) * 0.5)  # Simplified model
            
            results[length] = {
                'relative_length': length / train_length,
                'expected_degradation': degradation,
                'likely_to_fail': length > 2 * train_length
            }
        
        return results
    
    def detect_position_bias(self, attention_weights: np.ndarray) -> Dict:
        """Detect 'lost in the middle' effect"""
        seq_len = attention_weights.shape[-1]
        
        # Average attention to each position
        position_attention = np.mean(attention_weights, axis=(0, 1))
        
        # Compare beginning, middle, end
        segment_size = seq_len // 3
        beginning = np.mean(position_attention[:segment_size])
        middle = np.mean(position_attention[segment_size:2*segment_size])
        end = np.mean(position_attention[2*segment_size:])
        
        return {
            'beginning_attention': float(beginning),
            'middle_attention': float(middle),
            'end_attention': float(end),
            'middle_neglect_ratio': float(middle / ((beginning + end) / 2)),
            'has_position_bias': middle < 0.7 * min(beginning, end)
        }
    
    def hallucination_indicators(self, logits: np.ndarray, 
                                  generated_tokens: List[int]) -> Dict:
        """Indicators that output might be hallucinated"""
        
        # Entropy of predictions
        probs = self._softmax(logits)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
        
        # Low entropy = high confidence (can still be wrong)
        avg_entropy = float(np.mean(entropy))
        
        # Repetition detection
        unique_ratio = len(set(generated_tokens)) / len(generated_tokens)
        
        return {
            'avg_entropy': avg_entropy,
            'confidence': 1 - avg_entropy / np.log(probs.shape[-1]),
            'repetition_ratio': 1 - unique_ratio,
            'potential_hallucination': avg_entropy < 0.5 and unique_ratio < 0.3
        }
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Reasoning failure detection
class ReasoningFailureDetector:
    """Detect common reasoning failures"""
    
    def __init__(self):
        self.failure_patterns = {
            'arithmetic': self._check_arithmetic,
            'logical_consistency': self._check_consistency,
            'factual_accuracy': self._check_facts
        }
    
    def _check_arithmetic(self, response: str, expected: float) -> Dict:
        """Check if arithmetic in response is correct"""
        import re
        
        # Extract numbers from response
        numbers = re.findall(r'-?\d+\.?\d*', response)
        
        if not numbers:
            return {'has_arithmetic': False}
        
        final_answer = float(numbers[-1])
        is_correct = abs(final_answer - expected) < 0.01
        
        return {
            'has_arithmetic': True,
            'extracted_answer': final_answer,
            'expected': expected,
            'is_correct': is_correct
        }
    
    def _check_consistency(self, statements: List[str]) -> Dict:
        """Check for logical contradictions"""
        # Simplified consistency check
        contradictions = []
        
        for i, s1 in enumerate(statements):
            for j, s2 in enumerate(statements[i+1:], i+1):
                # Would use NLI model in practice
                if self._potentially_contradicts(s1, s2):
                    contradictions.append((i, j))
        
        return {
            'num_statements': len(statements),
            'potential_contradictions': len(contradictions),
            'is_consistent': len(contradictions) == 0
        }
    
    def _potentially_contradicts(self, s1: str, s2: str) -> bool:
        # Placeholder - would use NLI model
        return False
    
    def _check_facts(self, claims: List[str], knowledge_base: Dict) -> Dict:
        """Check factual accuracy against knowledge base"""
        verified = 0
        unverified = 0
        incorrect = 0
        
        for claim in claims:
            # Would use fact-checking system in practice
            pass
        
        return {
            'total_claims': len(claims),
            'verified': verified,
            'unverified': unverified,
            'incorrect': incorrect
        }

# Mitigation strategies
def mitigation_strategies():
    """Known mitigations for Transformer limitations"""
    return {
        'quadratic_complexity': [
            'Sparse attention (BigBird, Longformer)',
            'Linear attention (Performer, Linear Transformer)',
            'Flash Attention (efficient implementation)',
            'Sliding window attention'
        ],
        'length_extrapolation': [
            'RoPE (Rotary Position Embeddings)',
            'ALiBi (Attention with Linear Biases)',
            'Position interpolation',
            'Train on longer sequences'
        ],
        'hallucination': [
            'Retrieval augmentation (RAG)',
            'Fact-checking post-processing',
            'Uncertainty quantification',
            'Human-in-the-loop verification'
        ],
        'reasoning': [
            'Chain-of-thought prompting',
            'External tools (calculators, search)',
            'Multi-step verification',
            'Fine-tuning on reasoning datasets'
        ]
    }
```

**Interview Tips:**
- O(n²) is theoretical; Flash Attention makes it practical for longer sequences
- "Lost in the middle" is well-documented research finding
- Hallucination is fundamental — can't be fully eliminated, only mitigated
- Tool use (calculators, search) addresses reasoning and factual limitations

---

### Question 65
- [ ] Done
**What are recent advances in efficient architectures (Mamba, RWKV, state space models)?**

**Answer:**

**Definition:**
State Space Models (SSMs) like **Mamba** and **RWKV** offer alternatives to Transformers with **linear complexity** O(n) vs O(n²). They use **recurrent-style computation** with **selective state updates**, enabling longer sequences with less memory. Key advantage: efficient inference (constant memory per token) while maintaining competitive quality.

**Architecture Comparison:**

| Architecture | Complexity | Memory | Parallelizable | Quality |
|--------------|------------|--------|----------------|---------|
| **Transformer** | O(n²) | O(n²) | ✓ Training | SOTA |
| **Mamba (SSM)** | O(n) | O(1)* | ✓ Training | Near-Transformer |
| **RWKV** | O(n) | O(1)* | ✓ Training | Competitive |
| **Linear Attention** | O(n) | O(n) | ✓ | Lower quality |

*Per-token memory during inference

**State Space Model (SSM) Fundamentals:**

```
Continuous SSM:
h'(t) = Ah(t) + Bx(t)    # State evolution
y(t) = Ch(t) + Dx(t)     # Output

Discretized (for sequences):
h_k = Āh_{k-1} + B̄x_k
y_k = Ch_k + Dx_k

Key insight: Can be computed as convolution for parallel training
```

**Mamba Architecture:**

| Component | Description |
|-----------|-------------|
| **Selective SSM** | Input-dependent state transitions |
| **Hardware-aware** | Optimized for GPU memory hierarchy |
| **Gated MLP** | Similar to Transformer FFN |
| **No attention** | Pure SSM + MLP blocks |

**RWKV Architecture:**

| Component | Description |
|-----------|-------------|
| **Time mixing** | Channel-wise RNN-like recurrence |
| **Channel mixing** | Token mixing across channels |
| **WKV mechanism** | Weighted key-value attention alternative |
| **Linear complexity** | RNN inference, parallel training |

**Python Code Example:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelectiveSSM(nn.Module):
    """Simplified Mamba-style Selective State Space Model"""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        
        # Input projection
        self.in_proj = nn.Linear(d_model, d_model * 2)
        
        # Convolution for local context
        self.conv1d = nn.Conv1d(
            d_model, d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model
        )
        
        # SSM parameters (made input-dependent for selectivity)
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1)  # Delta, B, C
        
        # Learnable A (discretized)
        A = torch.arange(1, d_state + 1).float()
        self.A_log = nn.Parameter(torch.log(A))
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        """Forward pass with selective scan"""
        batch, seq_len, d = x.shape
        
        # Input projection and split
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # Convolution
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        x = F.silu(x)
        
        # SSM (simplified)
        y = self.ssm(x)
        
        # Gating
        y = y * F.silu(z)
        
        return self.out_proj(y)
    
    def ssm(self, x):
        """Selective state space computation"""
        batch, seq_len, d = x.shape
        
        # Input-dependent parameters (selectivity)
        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split([1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(delta)
        
        # Discretize A
        A = -torch.exp(self.A_log)
        
        # Selective scan (simplified - real implementation is more efficient)
        h = torch.zeros(batch, self.d_state, d, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            # Discretized state update
            h = h * torch.exp(delta[:, t:t+1] * A.view(1, -1, 1)) + \
                B[:, t:t+1].unsqueeze(-1) * x[:, t:t+1].unsqueeze(1)
            
            # Output
            y = (C[:, t:t+1].unsqueeze(-1) * h).sum(dim=1)
            outputs.append(y)
        
        return torch.cat(outputs, dim=1)

class RWKVTimeMixing(nn.Module):
    """RWKV Time Mixing block (WKV attention alternative)"""
    
    def __init__(self, d_model: int, n_layer: int, layer_id: int):
        super().__init__()
        self.d_model = d_model
        
        # Time-decay factors (learnable)
        ratio = layer_id / (n_layer - 1) if n_layer > 1 else 0.5
        
        self.time_decay = nn.Parameter(torch.zeros(d_model))
        self.time_first = nn.Parameter(torch.zeros(d_model))
        
        # Mixing weights (interpolation with previous timestep)
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        
        # Projections
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, state=None):
        batch, seq_len, _ = x.shape
        
        # Shift for mixing with previous timestep
        if state is None:
            x_prev = torch.zeros_like(x[:, :1])
        else:
            x_prev = state
        
        x_shifted = torch.cat([x_prev, x[:, :-1]], dim=1)
        
        # Mix current and previous
        xk = x * self.time_mix_k + x_shifted * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_shifted * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + x_shifted * (1 - self.time_mix_r)
        
        # Projections
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        r = torch.sigmoid(r)
        
        # WKV computation (simplified - real uses custom CUDA kernel)
        wkv = self._wkv(k, v)
        
        return self.output(r * wkv), x[:, -1:]
    
    def _wkv(self, k, v):
        """Weighted Key-Value computation"""
        # Simplified version - real implementation more efficient
        batch, seq_len, d = k.shape
        
        w = -torch.exp(self.time_decay)
        u = self.time_first
        
        outputs = []
        a = torch.zeros(batch, d, device=k.device)
        b = torch.zeros(batch, d, device=k.device)
        
        for t in range(seq_len):
            kt = k[:, t]
            vt = v[:, t]
            
            wkv = (a + torch.exp(u + kt) * vt) / (b + torch.exp(u + kt))
            outputs.append(wkv.unsqueeze(1))
            
            a = torch.exp(w) * a + torch.exp(kt) * vt
            b = torch.exp(w) * b + torch.exp(kt)
        
        return torch.cat(outputs, dim=1)

# Comparison helper
def compare_architectures():
    """Compare Transformer vs SSM approaches"""
    return {
        'transformers': {
            'pros': [
                'Highest quality on most benchmarks',
                'Parallel training',
                'Well-understood',
                'Strong in-context learning'
            ],
            'cons': [
                'O(n²) attention complexity',
                'KV cache grows with context',
                'Expensive long-context inference'
            ]
        },
        'ssm_mamba': {
            'pros': [
                'O(n) complexity',
                'Constant memory inference',
                'Hardware-efficient',
                'Competitive quality'
            ],
            'cons': [
                'Less mature ecosystem',
                'May struggle with in-context learning',
                'Less interpretable'
            ]
        },
        'rwkv': {
            'pros': [
                'Linear complexity',
                'RNN inference, parallel training',
                'Active community',
                'Good for edge deployment'
            ],
            'cons': [
                'Quality gap on complex tasks',
                'Limited tooling',
                'Less research backing'
            ]
        }
    }
```

**Interview Tips:**
- Mamba achieves Transformer-level quality at 1-7B scale
- SSMs excel at long sequences but may lag on in-context learning
- Hybrid architectures (Mamba + Attention) show promise
- Watch this space — SSMs are actively evolving

---


---

# --- BERT/RoBERTa/DistilBERT Questions (from 09_llm_genai_tools/01_bert_roberta_distilbert) ---

# BERT/RoBERTa/DistilBERT - Theory Questions

## Question 1
- [ ] Done
**How do you choose between BERT, RoBERTa, and DistilBERT for a production sentiment analysis system with latency constraints?**
**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**What are the key architectural differences between BERT's masked language modeling and RoBERTa's training approach?**
**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How do you implement domain adaptation when fine-tuning BERT for specialized medical text classification?**
**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**In what scenarios would you choose DistilBERT's 40% size reduction over BERT's full performance capabilities?**
**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**How do you handle the computational trade-offs when deploying RoBERTa-large versus BERT-base in real-time applications?**
**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What preprocessing considerations are unique to BERT's WordPiece tokenization for multilingual text processing?**
**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How do you implement incremental learning with BERT models when new labeled data arrives continuously?**
**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What are the best practices for handling long documents that exceed BERT's 512-token limit?**
**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How do you optimize DistilBERT's knowledge distillation process for domain-specific tasks?**
**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**When would you use RoBERTa's dynamic masking strategy over BERT's static masking approach?**
**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How do you implement efficient batch processing for BERT inference in high-throughput production systems?**
**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**What techniques help reduce overfitting when fine-tuning BERT on small specialized datasets?**
**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do you handle class imbalance when fine-tuning BERT for multi-class classification tasks?**
**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What are the memory optimization strategies for deploying BERT models on edge devices?**
**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How do you implement effective warm-up strategies during BERT fine-tuning for optimal convergence?**
**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**When should you use BERT's [CLS] token versus pooling strategies for document-level representations?**
**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do you handle multilingual requirements when choosing between mBERT and language-specific BERT variants?**
**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What are the considerations for using DistilBERT in privacy-sensitive applications where model size matters?**
**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How do you implement gradient accumulation for BERT training when GPU memory is limited?**
**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What evaluation metrics best capture the performance differences between BERT variants on your specific task?**
**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you handle tokenization mismatches when transferring between different BERT model variants?**
**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**What are the best practices for implementing BERT model ensembles to improve robustness?**
**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How do you optimize the learning rate schedule for different BERT architectures during fine-tuning?**
**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**When would you implement custom attention mechanisms on top of pre-trained BERT representations?**
**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How do you handle catastrophic forgetting when continually fine-tuning BERT on new domains?**
**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What techniques help maintain BERT's performance when quantizing models for mobile deployment?**
**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you implement effective data augmentation strategies specifically for BERT-based models?**
**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**What are the considerations for using BERT in few-shot learning scenarios with limited examples per class?**
**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do you handle the computational cost of BERT's attention mechanism in very long sequence processing?**
**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**When should you use specialized BERT variants like BioBERT, FinBERT, or LegalBERT over general models?**
**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How do you implement model compression techniques beyond distillation for BERT deployment?**
**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**What are the best practices for handling noisy or adversarial inputs in BERT-based production systems?**
**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**How do you optimize BERT's performance for cross-lingual transfer learning tasks?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What monitoring strategies help detect performance degradation in deployed BERT models?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you implement effective prompt engineering techniques for BERT-style masked language models?**
**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**When would you choose task-specific fine-tuning versus feature extraction approaches with BERT?**
**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you handle the interpretability requirements when deploying BERT in regulated industries?**
**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What are the strategies for maintaining BERT model performance under dataset shift conditions?**
**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How do you implement cost-effective inference scaling for BERT models in cloud environments?**
**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**What techniques help reduce the environmental impact of training and deploying large BERT models?**
**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How do you handle version control and model lifecycle management for BERT variant deployments?**
**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What are the considerations for implementing BERT in real-time recommendation systems?**
**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you optimize BERT's throughput while maintaining quality in high-volume text processing pipelines?**
**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**When should you implement custom pre-training for BERT on domain-specific corpora?**
**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you handle the trade-offs between model accuracy and inference speed in BERT deployment?**
**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What are the best practices for A/B testing different BERT variants in production environments?**
**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you implement effective error handling and fallback mechanisms for BERT-based applications?**
**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**What techniques help ensure consistent BERT model performance across different hardware configurations?**
**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**How do you optimize BERT's performance for streaming text processing applications?**
**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**What are the key considerations when migrating from older transformer models to modern BERT variants?**
**Answer:** _To be filled_

---


---

# --- GPT-2/GPT-3/GPT-4 Questions (from 09_llm_genai_tools/02_gpt2_gpt3_gpt4) ---

# GPT-2/GPT-3/GPT-4 - Theory Questions

## Question 1
- [ ] Done
**How do you optimize GPT-4's context window utilization for long-document summarization tasks?**
**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**What are the cost-efficiency considerations when choosing between GPT-3.5-turbo and GPT-4 for production applications?**
**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How do you implement effective prompt engineering strategies that work consistently across GPT-2, GPT-3, and GPT-4?**
**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**When would you choose fine-tuning GPT-2 locally versus using GPT-4's in-context learning capabilities?**
**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**How do you handle rate limiting and API quotas when building production systems around GPT-3/GPT-4?**
**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What techniques help reduce hallucination in GPT models for factual content generation?**
**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How do you implement retrieval-augmented generation (RAG) to enhance GPT model accuracy?**
**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What are the best practices for handling sensitive data when using cloud-based GPT APIs?**
**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How do you measure and optimize the coherence of long-form text generated by GPT models?**
**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**When should you use GPT-4's multimodal capabilities versus separate vision and language models?**
**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How do you implement effective content filtering and safety measures for GPT-generated text?**
**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**What strategies help maintain consistent writing style and tone across GPT model generations?**
**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do you handle the computational requirements for self-hosting GPT-2 versus using API-based solutions?**
**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What are the techniques for implementing few-shot learning effectively with different GPT variants?**
**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How do you optimize prompt design for complex reasoning tasks using GPT-4's advanced capabilities?**
**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**When would you implement custom stopping criteria for GPT text generation in production systems?**
**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do you handle context management for multi-turn conversations using GPT models?**
**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What are the best practices for implementing GPT model caching to reduce API costs?**
**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How do you measure and improve the factual accuracy of GPT-generated content?**
**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What techniques help control the creativity versus consistency trade-off in GPT text generation?**
**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you implement effective batch processing for high-volume GPT API requests?**
**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**When should you use GPT-4's function calling capabilities versus traditional API integrations?**
**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How do you handle version migration when OpenAI updates GPT model versions?**
**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**What are the strategies for implementing GPT models in multilingual content generation systems?**
**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How do you optimize GPT model performance for domain-specific technical writing tasks?**
**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What techniques help ensure GPT-generated content meets specific compliance and regulatory requirements?**
**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you implement effective quality assessment for GPT-generated content at scale?**
**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**When would you combine multiple GPT model variants in an ensemble approach?**
**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do you handle the latency requirements for real-time applications using GPT APIs?**
**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What are the best practices for implementing GPT models in customer-facing chatbot applications?**
**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How do you optimize GPT prompt engineering for code generation and programming tasks?**
**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**What strategies help maintain GPT model performance consistency across different user contexts?**
**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**How do you implement effective monitoring and alerting for GPT-based production systems?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**When should you use GPT-4's advanced reasoning capabilities versus simpler, faster alternatives?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you handle the scalability challenges of GPT model deployment in enterprise environments?**
**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What techniques help reduce bias in GPT-generated content for diverse user bases?**
**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you implement cost optimization strategies for large-scale GPT API usage?**
**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What are the considerations for implementing GPT models in privacy-sensitive applications?**
**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How do you handle error recovery and fallback mechanisms in GPT-based systems?**
**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**When would you implement custom fine-tuning versus using GPT's built-in adaptability features?**
**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How do you optimize GPT model selection based on specific use case requirements and constraints?**
**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What strategies help ensure GPT-generated content maintains brand voice and messaging consistency?**
**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you implement effective testing and validation procedures for GPT-based applications?**
**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What techniques help manage the unpredictability of GPT model outputs in production systems?**
**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you handle data retention and privacy concerns when using GPT APIs with user data?**
**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What are the best practices for implementing GPT models in content moderation workflows?**
**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you optimize GPT performance for specific industries like healthcare, finance, or legal services?**
**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**When should you implement custom preprocessing versus relying on GPT's built-in text processing?**
**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**How do you handle the integration challenges when combining GPT models with existing enterprise systems?**
**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**What are the key performance indicators for measuring GPT model success in production applications?**
**Answer:** _To be filled_

---


---

# --- LLaMA/Falcon/Mistral Questions (from 09_llm_genai_tools/03_llama_falcon_mistral) ---

# LLaMA/Falcon/Mistral - Theory Questions

## Question 1
- [ ] Done
**How do you choose between LLaMA, Falcon, and Mistral based on specific deployment constraints and performance requirements?**
**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**What are the licensing implications when deploying LLaMA versus Falcon or Mistral in commercial applications?**
**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How do you optimize Mistral 7B's sliding window attention for long-sequence processing tasks?**
**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**When would you choose Falcon's 40B model over other alternatives for enterprise-scale applications?**
**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**How do you implement efficient fine-tuning strategies for LLaMA models on domain-specific datasets?**
**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What are the memory optimization techniques specific to deploying LLaMA models on consumer hardware?**
**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How do you handle the computational trade-offs between model size and inference speed across these three model families?**
**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What quantization strategies work best for each model family while maintaining performance quality?**
**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How do you implement effective instruction tuning for LLaMA, Falcon, and Mistral models?**
**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**When should you use Mistral's mixture-of-experts architecture versus traditional dense models?**
**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How do you optimize batch processing efficiency for different model architectures in production environments?**
**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**What are the best practices for implementing RLHF (Reinforcement Learning from Human Feedback) with these models?**
**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do you handle multilingual capabilities comparison between LLaMA, Falcon, and Mistral variants?**
**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What techniques help reduce inference latency when deploying these models in real-time applications?**
**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How do you implement effective model parallelism strategies for large LLaMA or Falcon deployments?**
**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**When would you choose specialized variants like Code Llama versus general-purpose models?**
**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do you optimize memory usage during training and inference for resource-constrained environments?**
**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What are the strategies for implementing continual learning with LLaMA, Falcon, or Mistral models?**
**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How do you handle the evaluation challenges when comparing performance across different model architectures?**
**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What techniques help maintain model performance when adapting to new domains or tasks?**
**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you implement efficient serving infrastructure for high-throughput applications using these models?**
**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**When should you use LoRA (Low-Rank Adaptation) versus full fine-tuning for each model family?**
**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How do you optimize the balance between model capability and deployment cost across different options?**
**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**What are the best practices for implementing safety measures and content filtering with these models?**
**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How do you handle version control and model lifecycle management for open-source model deployments?**
**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What strategies help ensure consistent performance across different hardware configurations?**
**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you implement effective prompt engineering techniques optimized for each model's characteristics?**
**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**When would you combine multiple models in an ensemble versus using a single larger model?**
**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do you optimize storage and loading efficiency for frequent model switching in production?**
**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What techniques help reduce the environmental impact of training and deploying these large models?**
**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How do you implement effective monitoring and performance tracking for self-hosted model deployments?**
**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**What are the considerations for implementing these models in edge computing scenarios?**
**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**How do you handle the interpretability and explainability requirements for regulated industry deployments?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**When should you implement custom tokenization versus using the model's default tokenizer?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you optimize these models for specific use cases like code generation, creative writing, or analysis?**
**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What strategies help maintain model security and prevent unauthorized access in self-hosted deployments?**
**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you implement effective load balancing and scaling strategies for high-demand applications?**
**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What techniques help optimize the training efficiency when customizing these models for specific domains?**
**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How do you handle the integration challenges when incorporating these models into existing ML pipelines?**
**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**When would you choose between different quantization formats (INT8, INT4, FP16) for each model family?**
**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How do you implement effective caching strategies to improve response times and reduce computational costs?**
**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What are the best practices for handling model updates and migrations in production environments?**
**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you optimize these models for specific hardware accelerators like GPUs, TPUs, or specialized AI chips?**
**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What strategies help ensure reliable performance under varying load conditions?**
**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you implement effective A/B testing frameworks for comparing different model variants?**
**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What techniques help optimize the model selection process based on real-time performance metrics?**
**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you handle the complexity of managing multiple model versions and configurations simultaneously?**
**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**When should you implement custom inference optimizations versus using existing frameworks like vLLM or TensorRT?**
**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**How do you optimize these models for specific deployment patterns like serverless, containerized, or bare-metal?**
**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**What are the key considerations for long-term maintenance and support of self-hosted model deployments?**
**Answer:** _To be filled_

---


---

# --- T5/BART Questions (from 09_llm_genai_tools/04_t5_bart) ---

# T5/BART - Theory Questions

## Question 1
- [ ] Done
**How do you choose between T5's text-to-text approach and BART's encoder-decoder architecture for specific NLP tasks?**
**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**What are the advantages of T5's unified framework when building multi-task learning systems?**
**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How do you optimize BART's denoising pre-training approach for domain-specific text generation tasks?**
**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**When would you use T5's prefix-based task specification versus BART's task-specific fine-tuning?**
**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**How do you handle the computational requirements differences between T5 and BART in production environments?**
**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What techniques help optimize T5's performance for extremely long input sequences?**
**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How do you implement effective prompt design strategies for T5's text-to-text paradigm?**
**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What are the best practices for fine-tuning BART on abstractive summarization tasks?**
**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How do you handle the memory optimization challenges when deploying large T5 or BART models?**
**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**When should you use T5's multi-task capabilities versus training separate specialized models?**
**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How do you optimize BART's attention mechanisms for document-level understanding tasks?**
**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**What strategies help improve T5's performance on few-shot learning scenarios?**
**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do you implement effective evaluation metrics for comparing T5 and BART on generation tasks?**
**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What techniques help reduce hallucination in both T5 and BART text generation outputs?**
**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How do you handle the tokenization differences between T5 and BART when processing diverse text types?**
**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**When would you implement custom pre-training objectives for T5 or BART on domain-specific corpora?**
**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do you optimize batch processing efficiency for T5 and BART in high-throughput applications?**
**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What are the best practices for implementing controllable text generation with T5 and BART?**
**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How do you handle the scalability challenges when deploying T5 or BART for real-time applications?**
**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What techniques help maintain consistency in generated outputs across different input contexts?**
**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you implement effective transfer learning strategies from general T5/BART to specialized domains?**
**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**When should you use T5's encoder-decoder architecture versus decoder-only models for specific tasks?**
**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How do you optimize BART's performance for multilingual text processing and generation?**
**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**What strategies help improve the factual accuracy of T5 and BART generated content?**
**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How do you handle the evaluation challenges when comparing T5 and BART across different task types?**
**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What techniques help optimize the training efficiency for large T5 and BART model variants?**
**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you implement effective quality control measures for T5 and BART generated outputs?**
**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**When would you combine T5 and BART in ensemble approaches for improved performance?**
**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do you optimize memory usage during inference for T5 and BART in resource-constrained environments?**
**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What are the best practices for implementing T5 and BART in conversational AI systems?**
**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How do you handle the context management challenges in multi-turn applications using T5 or BART?**
**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**What strategies help ensure T5 and BART model robustness against adversarial inputs?**
**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**How do you implement effective beam search and decoding strategies for optimal generation quality?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**When should you use T5's span corruption objective versus BART's document corruption approach?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you optimize T5 and BART for specific text generation tasks like dialogue, summaries, or creative writing?**
**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What techniques help maintain T5 and BART performance consistency across different deployment environments?**
**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you handle the versioning and model lifecycle management for T5 and BART deployments?**
**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What are the considerations for implementing T5 and BART in privacy-sensitive applications?**
**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How do you optimize the prompt engineering approaches specific to T5's text-to-text framework?**
**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**When would you implement custom loss functions versus standard objectives for T5 and BART training?**
**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How do you handle the integration challenges when incorporating T5 or BART into existing NLP pipelines?**
**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What strategies help optimize T5 and BART performance for domain-specific vocabularies and terminology?**
**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you implement effective monitoring and performance tracking for T5 and BART production systems?**
**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What techniques help reduce the computational overhead of T5 and BART attention mechanisms?**
**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you optimize T5 and BART for specific hardware configurations and acceleration platforms?**
**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**When should you use progressive training strategies versus standard fine-tuning for T5 and BART?**
**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you handle the data preprocessing requirements specific to T5's text-to-text format?**
**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**What are the best practices for implementing T5 and BART in content creation workflows?**
**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**How do you optimize the trade-offs between generation quality and inference speed for production deployment?**
**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**What strategies help ensure T5 and BART generated content meets specific style and format requirements?**
**Answer:** _To be filled_

---


---

# --- LLMs (from 06_ai_nlp/03_llms) - theory questions ---

# Llms Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What areLarge Language Models (LLMs)and how do they work?**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**Describe the architecture of atransformer modelthat is commonly used in LLMs.**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**What are the main differences betweenLLMsand traditionalstatistical language models?**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**Can you explain the concept ofattention mechanismsin transformer models?**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**What arepositional encodingsin the context of LLMs?**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**What is the role oftransformersin achieving parallelization in LLMs?**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**What are some prominentapplicationsof LLMs today?**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**Describe how LLMs can be used in thegeneration of synthetic text.**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**Explain how LLMs can improveinformation retrievalanddocument summarization.**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**Describe theBERT (Bidirectional Encoder Representations from Transformers)model and its significance.**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**Explain the core idea behind theT5 (Text-to-Text Transfer Transformer)model.**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**What is theRoBERTa modeland how does it differ from standard BERT?**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**How does theGPT (Generative Pre-trained Transformer)series of models work?**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**What are some of thelimitationsof the Transformer architecture in LLMs?**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**What is the importance ofbatch sizeandsequence lengthin LLM training?**

**Answer:** _[To be filled]_

---

## Question 16
- [ ] Done

**Explain the concept ofgradient checkpointingin the context of training efficiency.**

**Answer:** _[To be filled]_

---

## Question 17
- [ ] Done

**Can you describe techniques tomonitor and maintainLLMs in production?**

**Answer:** _[To be filled]_

---

## Question 18
- [ ] Done

**Explain the factors to consider when selectinghardwarefor training LLMs.**

**Answer:** _[To be filled]_

---

## Question 19
- [ ] Done

**Explainmodel versioningstrategies when updating LLMs in production.**

**Answer:** _[To be filled]_

---

## Question 20
- [ ] Done

**Describe a method to efficiently roll back to a previous LLM model state in case of failures.**

**Answer:** _[To be filled]_

---

## Question 21
- [ ] Done

**What are the potential futureapplicationsof LLMs that are currently being researched?**

**Answer:** _[To be filled]_

---

## Question 22
- [ ] Done

**Explain howcapsule networksmight be integrated with LLMs.**

**Answer:** _[To be filled]_

---

## Question 23
- [ ] Done

**What arezero-shotandfew-shot learningcapabilities in LLMs?**

**Answer:** _[To be filled]_

---


---

# --- LLMs (from 06_ai_nlp/03_llms) - coding questions ---

# Llms Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Write a Python function usingPyTorchorTensorFlowto tokenize input text forGPT-2.**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**Implement a simpletransformer blockusingPyTorchorTensorFlow.**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**Train aminiature transformer modelon a small text corpus.**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**Create a function that performsgreedy decodingfor text generation using a pre-trained transformer model.**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**Write code tovisualize attention weightsfrom a pre-trained transformer model.**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**Modify a pre-trainedBERT modelfor a classification task usingtransfer learning.**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**Implement abeam search algorithmfor better text generation in language models.**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**Develop a customloss functionfor a transformer model that accounts for both forward and backward prediction.**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**Fine-tune aGPT-2 modelfor a specific text style or author usingPyTorchorTensorFlow.**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**Code a routine to performabstractive text summarizationusing a pre-trainedT5 model.**

**Answer:** _[To be filled]_

---


---

# --- LLMs (from 06_ai_nlp/03_llms) - general questions ---

# Llms Interview Questions - General Questions

## Question 1
- [ ] Done

**How do LLMs handlecontextandlong-term dependenciesin text?**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**How isGPT-4different from its predecessors likeGPT-3in terms of capabilities and applications?**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**Can you mention anydomain-specificadaptations of LLMs?**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**How do LLMs contribute to the field ofsentiment analysis?**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**In what ways can LLMs be utilized forlanguage translation?**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**How dohyperparametersaffect the performance of LLMs?**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**How can one useknowledge distillationin the context of LLMs?**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**Whatpreprocessing stepsare crucial when dealing with input data for LLMs?**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**How istokenizationperformed in the context of LLMs, and why is it important?**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**What considerations should be taken into account for handling differentlanguagesin LLMs?**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**How do you address the challenge ofoverfittingin LLMs?**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**What approach would you take to build achatbotusing LLMs?**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**Design a system using LLMs to generatecode snippetsfrom natural language descriptions.**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**Whatmetricswould you use to evaluate the performance of a fine-tuned LLM?**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**How canreinforcement learningbe applied to further train or fine-tune LLMs?**

**Answer:** _[To be filled]_

---


---

# --- LLMs (from 06_ai_nlp/03_llms) - scenario based questions ---

# Llms Interview Questions - Scenario_Based Questions

## Question 1
- [ ] Done

**Discuss the significance ofpre-trainingandfine-tuningin the context of LLMs.**

**Answer:** _[To be filled]_

---

## Question 2
- [ ] Done

**Discuss theapplicationof LLMs inconversation AIandchatbots.**

**Answer:** _[To be filled]_

---

## Question 3
- [ ] Done

**Discuss the technique of ‘masking’ in transformer models like BERT.**

**Answer:** _[To be filled]_

---

## Question 4
- [ ] Done

**Discuss the role oflearning rate schedulesin training LLMs.**

**Answer:** _[To be filled]_

---

## Question 5
- [ ] Done

**Discuss techniques for reducing thememory footprintof LLMs during training.**

**Answer:** _[To be filled]_

---

## Question 6
- [ ] Done

**Discuss the process ofvocabulary creationand management in LLMs.**

**Answer:** _[To be filled]_

---

## Question 7
- [ ] Done

**Discuss strategies forefficient deploymentof LLMs in production environments.**

**Answer:** _[To be filled]_

---

## Question 8
- [ ] Done

**Discuss the role ofmulti-GPUanddistributed trainingin LLMs.**

**Answer:** _[To be filled]_

---

## Question 9
- [ ] Done

**How would you set up a LLM to create anews article summarizer?**

**Answer:** _[To be filled]_

---

## Question 10
- [ ] Done

**Discuss techniques to adapt a LLM for alegal document reviewapplication.**

**Answer:** _[To be filled]_

---

## Question 11
- [ ] Done

**Propose a framework to use LLMs in creatingpersonalized content recommendations.**

**Answer:** _[To be filled]_

---

## Question 12
- [ ] Done

**How would you conductA/B testingfor a new version of an LLM-based application?**

**Answer:** _[To be filled]_

---

## Question 13
- [ ] Done

**Discussgenerative adversarial networks (GANs)in the context of text generation with LLMs.**

**Answer:** _[To be filled]_

---

## Question 14
- [ ] Done

**Discuss the concept ofcatastrophic forgettingin LLMs and potential solutions.**

**Answer:** _[To be filled]_

---

## Question 15
- [ ] Done

**Discuss the implications ofattention flowin multi-head attention mechanisms.**

**Answer:** _[To be filled]_

---
