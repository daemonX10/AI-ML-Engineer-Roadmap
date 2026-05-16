# Nlp Interview Questions - General Questions

## Question 1
- [ ] Done

**What do you understand by the terms ‘corpus’, ‘tokenization’, and ‘stopwords’ in NLP?**

**Answer:**

**Definition:** These are three foundational building blocks of any NLP pipeline — corpus provides the raw data, tokenization breaks it into processable units, and stopwords filtering removes low-information words to improve signal-to-noise ratio.

| Term | Definition | Purpose | Example |
|------|-----------|---------|--------|
| **Corpus** | A large, structured collection of text documents used for training and evaluating NLP models | Provides the raw linguistic data for statistical learning | Wikipedia dump, Brown Corpus, Reuters news dataset |
| **Tokenization** | The process of splitting text into smaller units called tokens (words, subwords, or characters) | Converts raw text into discrete units that models can process | "I can't wait" → ["I", "ca", "n't", "wait"] |
| **Stopwords** | Common, high-frequency words that carry little semantic meaning on their own | Removing them reduces dimensionality and focuses on content-bearing words | "the", "is", "at", "and", "a", "in" |

**Types of Tokenization:**

| Type | Method | Use Case |
|------|--------|----------|
| Word-level | Split on whitespace/punctuation | Traditional NLP, bag-of-words |
| Subword (BPE) | Merge frequent character pairs iteratively | Transformer models (GPT, BERT) |
| Character-level | Each character is a token | Morphologically rich languages |
| Sentence-level | Split into sentences | Summarization, translation |

**Corpus Types:**
- **Monolingual** — single language (e.g., Brown Corpus for English)
- **Parallel** — aligned text in two or more languages (e.g., Europarl for machine translation)
- **Annotated** — text with labels like POS tags, NER tags (e.g., CoNLL-2003)
- **Domain-specific** — text from a particular field (e.g., PubMed for biomedical NLP)

**Python Code Example:**

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Pipeline: Raw Text → Sentence Tokenize → Word Tokenize → Stopword Removal → Clean Tokens

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# 1. Define a small corpus (collection of documents)
corpus = [
    "Natural language processing enables computers to understand human language.",
    "Tokenization is the first step in most NLP pipelines.",
    "Stopwords like 'the' and 'is' are removed to reduce noise."
]

stop_words = set(stopwords.words('english'))

for doc in corpus:
    # 2. Tokenize into words
    tokens = word_tokenize(doc.lower())
    print(f"Original tokens:  {tokens}")

    # 3. Remove stopwords
    filtered = [t for t in tokens if t.isalnum() and t not in stop_words]
    print(f"After filtering:  {filtered}")
    print(f"Removed {len(tokens) - len(filtered)} stopwords/punctuation\n")
```

**Interview Tips:**
- Mention that stopword lists are **task-dependent** — for sentiment analysis, words like "not" and "but" should be kept
- Highlight that modern transformer models (BERT, GPT) do **not** remove stopwords because attention mechanisms learn to handle them
- Emphasize that tokenization choice affects vocabulary size and OOV (out-of-vocabulary) handling
- Note that subword tokenization (BPE, WordPiece, SentencePiece) has largely replaced word-level tokenization in modern NLP
- Mention that corpus quality and representativeness directly impact model performance

---

## Question 2
- [ ] Done

**Distinguish between morphology and syntax in the context of NLP.**

**Answer:**

**Definition:** Morphology studies the internal structure and formation of words (how morphemes combine), while syntax studies how words combine to form grammatical sentences (phrase and sentence structure).

| Aspect | Morphology | Syntax |
|--------|-----------|--------|
| **Unit of analysis** | Morphemes (smallest meaningful units) | Words, phrases, clauses, sentences |
| **Focus** | Word formation and inflection | Sentence structure and word order |
| **Example** | "unhappiness" → un + happy + ness | "The cat sat on the mat" → NP + VP + PP |
| **NLP tasks** | Stemming, lemmatization, morphological analysis | Parsing, dependency analysis, constituency parsing |
| **Key tools** | Porter Stemmer, WordNet Lemmatizer | Dependency parsers (spaCy), CFG parsers (NLTK) |
| **Challenges** | Irregular forms, agglutinative languages | Ambiguity, long-range dependencies |

**Morphology Subtypes:**

| Type | Description | Example |
|------|-------------|--------|
| **Inflectional** | Changes grammatical properties without creating new words | run → runs, running, ran |
| **Derivational** | Creates new words by adding affixes | happy → unhappy, happiness |
| **Compounding** | Combines two or more root words | sun + flower → sunflower |

**Syntax Representations:**

| Type | Description | Use Case |
|------|-------------|----------|
| **Constituency parsing** | Breaks sentence into nested sub-phrases (NP, VP, PP) | Grammar checking, information extraction |
| **Dependency parsing** | Shows head-modifier relationships between words | Relation extraction, semantic role labeling |

**Python Code Example:**

```python
import spacy
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

# Pipeline: Text → Morphological Analysis (stemming/lemmatization) → Syntactic Parsing (dependency tree)

nlp = spacy.load("en_core_web_sm")
nltk.download('wordnet', quiet=True)

# --- Morphology: Analyzing word structure ---
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "happier", "unhappiness", "geese", "better"]
print("=== Morphological Analysis ===")
for word in words:
    stem = stemmer.stem(word)
    lemma = lemmatizer.lemmatize(word)
    print(f"  {word:15s} → stem: {stem:12s} | lemma: {lemma}")

# --- Syntax: Analyzing sentence structure ---
print("\n=== Syntactic Analysis (Dependency Parsing) ===")
doc = nlp("The quick brown fox jumps over the lazy dog.")
for token in doc:
    print(f"  {token.text:10s} → dep: {token.dep_:12s} | head: {token.head.text:10s} | POS: {token.pos_}")

# Visualize noun chunks (constituency-like)
print("\nNoun Chunks:")
for chunk in doc.noun_chunks:
    print(f"  '{chunk.text}' (root: {chunk.root.text}, dep: {chunk.root.dep_})")
```

**Interview Tips:**
- Morphology matters more for **morphologically rich languages** (Turkish, Finnish, Arabic) where a single word can encode an entire sentence
- Syntax is critical for tasks requiring **structural understanding** — question answering, relation extraction, machine translation
- Modern transformers implicitly learn both morphological and syntactic patterns through subword tokenization and attention
- Mention the **Chomsky hierarchy** for formal grammar classification if discussing theoretical foundations
- Dependency parsing is preferred over constituency parsing in modern NLP because it directly captures semantic relationships

---

## Question 3
- [ ] Done

**How are Hidden Markov Models (HMMs) applied in NLP tasks?**

**Answer:**

**Definition:** A Hidden Markov Model (HMM) is a probabilistic generative model where the system transitions between hidden states (e.g., POS tags) that emit observable outputs (e.g., words), using transition and emission probabilities to model sequential data.

**Core Components of an HMM:**

| Component | Symbol | Description | Example (POS Tagging) |
|-----------|--------|-------------|----------------------|
| Hidden states | S | Unobservable states the model transitions between | POS tags: {NN, VB, DT, JJ, ...} |
| Observations | O | Visible outputs emitted from hidden states | Words: {"the", "cat", "sat"} |
| Transition probs | A(i→j) | P(state_j \| state_i) — probability of moving between states | P(VB \| NN) = 0.3 |
| Emission probs | B(s→o) | P(observation \| state) — probability of emitting an observation | P("cat" \| NN) = 0.01 |
| Initial probs | π | Probability of starting in each state | P(DT as first tag) = 0.4 |

**Three Fundamental HMM Problems:**

| Problem | Algorithm | Description | NLP Application |
|---------|-----------|-------------|----------------|
| **Evaluation** | Forward Algorithm | P(observation sequence \| model) | Language model scoring |
| **Decoding** | Viterbi Algorithm | Find most likely hidden state sequence | POS tagging, NER |
| **Learning** | Baum-Welch (EM) | Estimate model parameters from data | Training from unlabeled data |

**NLP Applications of HMMs:**

| Application | Hidden States | Observations |
|-------------|--------------|-------------|
| POS Tagging | Part-of-speech tags | Words in sentence |
| Named Entity Recognition | BIO entity tags | Tokens |
| Speech Recognition | Phonemes | Acoustic features |
| Text Segmentation | Topic/segment labels | Sentences |
| Optical Character Recognition | Characters | Image features |

**Python Code Example:**

```python
import numpy as np

# Pipeline: Define HMM Parameters → Viterbi Decoding → Predict Most Likely POS Tag Sequence

# Simple POS tagging HMM
states = ['DT', 'NN', 'VB']  # Determiner, Noun, Verb
observations = ['the', 'cat', 'sat']

# Transition probabilities: P(next_state | current_state)
transition = {
    'START': {'DT': 0.6, 'NN': 0.3, 'VB': 0.1},
    'DT':    {'DT': 0.1, 'NN': 0.7, 'VB': 0.2},
    'NN':    {'DT': 0.1, 'NN': 0.2, 'VB': 0.7},
    'VB':    {'DT': 0.4, 'NN': 0.4, 'VB': 0.2},
}

# Emission probabilities: P(word | state)
emission = {
    'DT': {'the': 0.7, 'cat': 0.01, 'sat': 0.01},
    'NN': {'the': 0.01, 'cat': 0.5, 'sat': 0.05},
    'VB': {'the': 0.01, 'cat': 0.02, 'sat': 0.6},
}

def viterbi(obs, states, trans, emis):
    """Viterbi algorithm for finding most likely state sequence."""
    V = [{}]  # trellis
    path = {}

    # Initialize with start probabilities
    for s in states:
        V[0][s] = trans['START'][s] * emis[s].get(obs[0], 1e-6)
        path[s] = [s]

    # Fill trellis for subsequent observations
    for t in range(1, len(obs)):
        V.append({})
        new_path = {}
        for s in states:
            prob, prev = max(
                (V[t-1][s0] * trans[s0][s] * emis[s].get(obs[t], 1e-6), s0)
                for s0 in states
            )
            V[t][s] = prob
            new_path[s] = path[prev] + [s]
        path = new_path

    # Find best final state
    best_prob, best_state = max((V[-1][s], s) for s in states)
    return path[best_state], best_prob

tags, prob = viterbi(observations, states, transition, emission)
for word, tag in zip(observations, tags):
    print(f"  {word:8s} → {tag}")
print(f"  Sequence probability: {prob:.6f}")
```

**Interview Tips:**
- HMMs assume the **Markov property** (next state depends only on current state) — this is a limitation for long-range dependencies
- HMMs have been largely **replaced by CRFs** (Conditional Random Fields) for sequence labeling because CRFs don't require independence assumptions on observations
- Deep learning models (BiLSTM-CRF, Transformers) now dominate, but HMMs remain important for understanding **probabilistic sequence modeling foundations**
- Mention that HMMs are **generative models** (model P(X,Y)) while CRFs are **discriminative** (model P(Y|X) directly)
- The Viterbi algorithm uses **dynamic programming** and runs in O(T × S²) time where T = sequence length, S = number of states

---

## Question 4
- [ ] Done

**How do Long Short-Term Memory (LSTM) networks work, and when would you use them?**

**Answer:**

**Definition:** LSTMs are a specialized type of Recurrent Neural Network (RNN) designed to learn long-range dependencies in sequential data by using a gating mechanism (forget, input, output gates) and a cell state that acts as a memory highway, solving the vanishing gradient problem of vanilla RNNs.

**LSTM Gate Equations:**

| Gate | Equation | Purpose |
|------|----------|--------|
| **Forget gate** (f_t) | σ(W_f · [h_{t-1}, x_t] + b_f) | Decides what information to **discard** from cell state |
| **Input gate** (i_t) | σ(W_i · [h_{t-1}, x_t] + b_i) | Decides what new information to **store** in cell state |
| **Candidate values** (C̃_t) | tanh(W_C · [h_{t-1}, x_t] + b_C) | Creates candidate values to potentially add to cell state |
| **Cell state update** (C_t) | f_t ⊙ C_{t-1} + i_t ⊙ C̃_t | Combines old memory (gated) with new candidate memory |
| **Output gate** (o_t) | σ(W_o · [h_{t-1}, x_t] + b_o) | Decides what parts of cell state to **output** |
| **Hidden state** (h_t) | o_t ⊙ tanh(C_t) | Filtered cell state becomes the output/hidden state |

**LSTM vs Vanilla RNN vs GRU:**

| Feature | Vanilla RNN | LSTM | GRU |
|---------|------------|------|-----|
| Gates | None | 3 (forget, input, output) | 2 (reset, update) |
| Cell state | No | Yes (separate memory highway) | No (merged with hidden state) |
| Parameters | Fewest | Most | Moderate |
| Long-range memory | Poor (vanishing gradient) | Excellent | Good |
| Training speed | Fast | Slowest | Faster than LSTM |
| Use when | Short sequences | Long dependencies, need fine control | Balance of speed and performance |

**When to Use LSTMs:**

| Use Case | Why LSTM Works Well |
|----------|--------------------|
| Machine Translation | Captures long-range dependencies between source/target words |
| Sentiment Analysis | Understands negation and context across long sentences |
| Named Entity Recognition | Sequential context helps identify entity boundaries |
| Speech Recognition | Temporal patterns in audio features |
| Time Series Forecasting | Learns long-term trends and seasonal patterns |
| Text Generation | Maintains coherence over generated sequences |

**Python Code Example:**

```python
import torch
import torch.nn as nn

# Pipeline: Input Embedding → Bidirectional LSTM → Dropout → Fully Connected → Softmax → Classification

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim,
                 n_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))       # (batch, seq_len, embed_dim)
        output, (hidden, cell) = self.lstm(embedded)        # hidden: (n_layers*dirs, batch, hidden)
        # Concatenate final forward and backward hidden states
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1) # (batch, hidden*2)
        return self.fc(self.dropout(hidden))                # (batch, output_dim)

# Instantiate model
model = SentimentLSTM(vocab_size=10000, embed_dim=128, hidden_dim=256, output_dim=2)
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# Example forward pass
sample_input = torch.randint(0, 10000, (4, 50))  # batch=4, seq_len=50
logits = model(sample_input)
print(f"Output shape: {logits.shape}")  # (4, 2)
```

**Interview Tips:**
- The **cell state** is the key innovation — it acts as a conveyor belt allowing gradients to flow unchanged across many timesteps
- BiLSTMs (bidirectional) are almost always preferred over unidirectional for NLP because they capture both left and right context
- LSTMs have been largely **superseded by Transformers** for most NLP tasks due to parallelization advantages, but remain relevant for streaming/real-time applications
- Mention the **vanishing gradient problem** that motivates LSTMs — in vanilla RNNs, gradients shrink exponentially during backpropagation through time
- Common variants: **Peephole LSTM** (gates can see cell state), **Coupled forget-input gate** (simplification where i_t = 1 - f_t)

---

## Question 5
- [ ] Done

**How do word2vec and GloVe differ as word embedding techniques?**

**Answer:**

**Definition:** word2vec and GloVe are both methods to learn dense vector representations of words, but they differ fundamentally — word2vec is a **predictive** model that learns embeddings by predicting context words (or center words) using a sliding window, while GloVe is a **count-based** model that learns embeddings by factorizing the global word-word co-occurrence matrix.

**Detailed Comparison:**

| Aspect | word2vec | GloVe |
|--------|---------|-------|
| **Full name** | Word to Vector | Global Vectors for Word Representation |
| **Developed by** | Mikolov et al. (Google, 2013) | Pennington et al. (Stanford, 2014) |
| **Approach** | Predictive (neural network) | Count-based (matrix factorization) |
| **Training objective** | Predict context/center words | Minimize difference between dot product and log co-occurrence |
| **Context** | Local (sliding window) | Global (entire corpus co-occurrence statistics) |
| **Architectures** | Skip-gram and CBOW | Weighted least squares on co-occurrence matrix |
| **Training data** | Processes text sequentially | Builds full co-occurrence matrix first |
| **Handles rare words** | Better (sees each occurrence) | Worse (sparse co-occurrence entries) |
| **Memory** | Lower (streaming) | Higher (stores full co-occurrence matrix) |
| **Parallelization** | Harder (sequential SGD) | Easier (matrix operations) |
| **Performance** | Strong on syntactic analogies | Strong on semantic similarity |

**word2vec Architectures:**

| Architecture | Input → Output | Objective | Speed |
|-------------|---------------|-----------|-------|
| **CBOW** (Continuous Bag of Words) | Context words → Center word | P(w_t \| w_{t-c}, ..., w_{t+c}) | Faster, better for frequent words |
| **Skip-gram** | Center word → Context words | P(w_{t-c}, ..., w_{t+c} \| w_t) | Slower, better for rare words |

**Key Training Tricks:**

| Technique | Purpose | Used In |
|-----------|---------|--------|
| Negative sampling | Approximate softmax by sampling negative examples | word2vec |
| Hierarchical softmax | Use Huffman tree for efficient probability computation | word2vec |
| Subsampling frequent words | Reduce dominant common words during training | word2vec |
| Bias terms | Capture word frequency effects | GloVe |
| Weighting function | Limit influence of very frequent co-occurrences | GloVe |

**Python Code Example:**

```python
from gensim.models import Word2Vec, KeyedVectors
import numpy as np

# Pipeline: Corpus → Tokenize → Train word2vec → Query Embeddings → Compute Similarity/Analogies

# Sample corpus (list of tokenized sentences)
corpus = [
    ["king", "queen", "royal", "palace", "throne", "crown"],
    ["man", "woman", "child", "boy", "girl", "people"],
    ["king", "man", "rule", "kingdom", "power", "throne"],
    ["queen", "woman", "rule", "kingdom", "grace", "crown"],
    ["dog", "cat", "pet", "animal", "puppy", "kitten"],
    ["computer", "program", "code", "software", "algorithm", "data"],
]

# Train word2vec (Skip-gram)
w2v_model = Word2Vec(
    sentences=corpus,
    vector_size=50,      # embedding dimension
    window=3,            # context window size
    min_count=1,         # include all words
    sg=1,                # 1=Skip-gram, 0=CBOW
    negative=5,          # negative sampling count
    epochs=100,
    seed=42
)

# Query word vectors
print("=== word2vec Embeddings ===")
print(f"Vector for 'king' (first 10 dims): {w2v_model.wv['king'][:10].round(3)}")
print(f"Similarity('king', 'queen'):  {w2v_model.wv.similarity('king', 'queen'):.4f}")
print(f"Similarity('king', 'dog'):    {w2v_model.wv.similarity('king', 'dog'):.4f}")

# For GloVe: Load pre-trained vectors (already in word2vec format)
# glove_vectors = KeyedVectors.load_word2vec_format('glove.6B.100d.w2v.txt')
# similarity = glove_vectors.similarity('king', 'queen')

print("\n=== Comparison Summary ===")
print("word2vec: Local context window, predictive, streaming training")
print("GloVe:   Global co-occurrence stats, count-based, matrix factorization")
```

**Interview Tips:**
- Both produce similar quality embeddings in practice — the choice often depends on the **corpus size and compute constraints**
- GloVe's loss function: $J = \sum_{i,j} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$ where $X_{ij}$ is co-occurrence count
- Modern alternatives like **FastText** (extends word2vec with subword information) and **contextual embeddings** (BERT, ELMo) have largely superseded both
- Key limitation of both: they produce **static embeddings** — one vector per word regardless of context ("bank" has same vector in "river bank" and "bank account")
- The famous analogy: king - man + woman ≈ queen works with both methods

---

## Question 6
- [ ] Done

**What challenges does one face when using vector space models for semantic analysis?**

**Answer:**

**Definition:** Vector space models (VSMs) represent text as vectors in a high-dimensional space where similarity between documents/words is computed via geometric measures (cosine similarity, Euclidean distance). Key challenges arise around sparsity, semantics, scalability, and context sensitivity.

**Major Challenges:**

| Challenge | Description | Impact | Mitigation |
|-----------|-----------|--------|------------|
| **High dimensionality** | Vocabulary can be 100K+ words, creating very large sparse vectors | Curse of dimensionality, slow computation | Dimensionality reduction (SVD, PCA), dense embeddings |
| **Sparsity** | Most entries in term-document matrices are zero | Unreliable similarity estimates, memory waste | TF-IDF weighting, smoothing, latent representations |
| **Synonymy** | Different words with same meaning ("car" vs "automobile") | Fails to recognize semantic equivalence | LSA, word embeddings, topic models |
| **Polysemy** | Same word with multiple meanings ("bank", "crane") | One vector conflates multiple senses | Contextual embeddings (BERT), word sense disambiguation |
| **Bag-of-words assumption** | Ignores word order and syntactic structure | "Dog bites man" = "Man bites dog" | N-grams, sequence models (RNNs, Transformers) |
| **Out-of-vocabulary (OOV)** | New words not in the training vocabulary | Cannot represent unseen words | Subword tokenization (BPE), character-level models |
| **Scalability** | Co-occurrence matrices grow as O(V²) | Memory and compute bottleneck for large vocabularies | Sparse storage, approximate nearest neighbors |
| **Domain shift** | Vectors trained on one domain perform poorly on another | "Java" means coffee vs programming language | Domain-adaptive fine-tuning, domain-specific corpora |

**Comparison of Vector Space Approaches:**

| Method | Type | Handles Synonymy | Handles Polysemy | Dimensionality |
|--------|------|:-:|:-:|:-:|
| Bag-of-Words / TF-IDF | Sparse | No | No | High (V) |
| LSA / LSI | Dense (reduced) | Partially | No | Low (k) |
| word2vec / GloVe | Dense (learned) | Yes | No | Low (d) |
| ELMo / BERT | Contextual | Yes | Yes | Medium (768-1024) |

**Python Code Example:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import numpy as np

# Pipeline: Documents → TF-IDF Vectorization → Demonstrate Challenges → Apply SVD for Dimensionality Reduction

documents = [
    "The car was fast and furious on the highway",      # doc 0
    "The automobile was speedy on the road",             # doc 1 (synonym: car/automobile)
    "The bank approved the loan application",            # doc 2 (bank = financial)
    "The river bank was covered with wildflowers",       # doc 3 (bank = river)
    "Fast cars race on highways and roads",              # doc 4 (related to 0,1)
]

# 1. TF-IDF vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print("=== Challenge: Sparsity ===")
density = tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])
print(f"Matrix shape: {tfidf_matrix.shape} (docs × vocab)")
print(f"Density: {density:.2%} (rest are zeros)\n")

print("=== Challenge: Synonymy ===")
sim_01 = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
sim_04 = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[4:5])[0][0]
print(f"Sim(car doc, automobile doc): {sim_01:.4f}  ← Should be HIGH but isn't")
print(f"Sim(car doc, fast cars doc):  {sim_04:.4f}  ← Higher due to word overlap\n")

print("=== Challenge: Polysemy ===")
sim_23 = cosine_similarity(tfidf_matrix[2:3], tfidf_matrix[3:4])[0][0]
print(f"Sim(financial bank, river bank): {sim_23:.4f}  ← Inflated by shared word 'bank'\n")

# 3. Dimensionality reduction with LSA/SVD
svd = TruncatedSVD(n_components=3)
reduced = svd.fit_transform(tfidf_matrix)
print("=== After SVD (3 components) ===")
print(f"Reduced shape: {reduced.shape}")
print(f"Variance explained: {svd.explained_variance_ratio_.sum():.2%}")
```

**Interview Tips:**
- The **distributional hypothesis** ("words that occur in similar contexts have similar meanings") is the theoretical foundation but also the limitation of VSMs
- Mention the **curse of dimensionality** — in very high dimensions, all points become nearly equidistant, making similarity measures unreliable
- TF-IDF helps with sparsity by weighting important terms, but doesn't solve synonymy or polysemy
- **LSA/LSI** was the first major approach to address synonymy via truncated SVD, capturing latent semantic dimensions
- Modern contextual embeddings (BERT) solve polysemy by producing different vectors for the same word in different contexts

---

## Question 7
- [ ] Done

**How do you use spaCy for text processing tasks?**

**Answer:**

**Definition:** spaCy is an industrial-strength, open-source NLP library in Python designed for production use. It provides pre-trained statistical models and a streamlined pipeline architecture for tokenization, POS tagging, dependency parsing, NER, lemmatization, and more — all accessible through a single `nlp()` call.

**spaCy Pipeline Architecture:**

| Component | Function | Output Attribute |
|-----------|----------|------------------|
| **Tokenizer** | Splits text into tokens | `doc[i].text` |
| **Tagger** | Assigns POS tags | `token.pos_`, `token.tag_` |
| **Parser** | Dependency parsing | `token.dep_`, `token.head` |
| **NER** | Named Entity Recognition | `doc.ents`, `ent.label_` |
| **Lemmatizer** | Reduces words to base form | `token.lemma_` |
| **Attribute Ruler** | Rule-based token attributes | Custom attributes |
| **TextCategorizer** | Document classification | `doc.cats` |

**spaCy vs NLTK:**

| Feature | spaCy | NLTK |
|---------|-------|------|
| **Design** | Production-ready, opinionated | Educational, flexible |
| **Speed** | Fast (Cython) | Slower (pure Python) |
| **Models** | Pre-trained statistical/transformer models | Rule-based + some statistical |
| **Pipeline** | Integrated (one `nlp()` call) | Manual (chain individual tools) |
| **NER** | Built-in with pre-trained models | Requires external setup |
| **Customization** | Custom components, `spacy train` CLI | More modular pick-and-choose |
| **Best for** | Production NLP applications | Teaching, research, prototyping |

**Available Model Sizes:**

| Model | Size | Accuracy | Speed | Includes |
|-------|------|----------|-------|----------|
| `en_core_web_sm` | ~12 MB | Good | Fastest | No word vectors |
| `en_core_web_md` | ~40 MB | Better | Fast | 20K word vectors |
| `en_core_web_lg` | ~560 MB | Better | Moderate | 685K word vectors |
| `en_core_web_trf` | ~440 MB | Best | Slowest | Transformer-based (RoBERTa) |

**Python Code Example:**

```python
import spacy
from spacy.tokens import Doc
from collections import Counter

# Pipeline: Load Model → Process Text → Extract Tokens/POS/Entities/Dependencies → Custom Analysis

nlp = spacy.load("en_core_web_sm")

text = """Apple Inc. is planning to open a new headquarters in Austin, Texas by 2025.
Tim Cook announced the $1 billion investment during the earnings call on Tuesday."""

# Process text through the full pipeline
doc = nlp(text)

# 1. Tokenization + POS tagging
print("=== Tokens & POS Tags ===")
for token in doc[:10]:
    print(f"  {token.text:15s} POS: {token.pos_:8s} Tag: {token.tag_:6s} Lemma: {token.lemma_}")

# 2. Named Entity Recognition
print("\n=== Named Entities ===")
for ent in doc.ents:
    print(f"  {ent.text:25s} → {ent.label_:10s} ({spacy.explain(ent.label_)})")

# 3. Dependency Parsing
print("\n=== Dependency Parse (first sentence) ===")
for token in list(doc.sents)[0]:
    print(f"  {token.text:15s} --{token.dep_:12s}--> {token.head.text}")

# 4. Noun chunks
print("\n=== Noun Chunks ===")
for chunk in doc.noun_chunks:
    print(f"  '{chunk.text}' (root: {chunk.root.text})")

# 5. Sentence segmentation
print("\n=== Sentences ===")
for i, sent in enumerate(doc.sents):
    print(f"  Sentence {i}: {sent.text.strip()}")

# 6. Similarity (requires medium/large model with vectors)
# doc1 = nlp("I love programming")
# doc2 = nlp("I enjoy coding")
# print(f"Similarity: {doc1.similarity(doc2):.4f}")
```

**Interview Tips:**
- spaCy processes text through a **pipeline** — calling `nlp(text)` runs all components sequentially and returns a `Doc` object
- You can **disable components** for speed: `nlp(text, disable=["ner", "parser"])` when you only need tokenization
- spaCy supports **custom pipeline components** via `@Language.component` decorator for adding business logic
- The `Matcher` and `PhraseMatcher` classes enable **rule-based matching** combined with statistical models
- For training custom NER or text classification, use **spacy train** CLI with a config file — this is the modern recommended approach
- spaCy 3.x introduced **transformer integration** allowing BERT/RoBERTa as the backbone via `spacy-transformers`

---

## Question 8
- [ ] Done

**How do you handle multilingual text processing in modern NLP libraries?**

**Answer:**

**Definition:** Multilingual text processing involves building NLP systems that can understand, analyze, and generate text across multiple languages. Modern approaches leverage multilingual pre-trained models, shared subword vocabularies, and cross-lingual transfer learning to handle diverse languages within a single framework.

**Key Challenges in Multilingual NLP:**

| Challenge | Description | Example |
|-----------|-------------|--------|
| **Script diversity** | Different writing systems with different directions | Latin, Arabic (RTL), Chinese (logographic), Devanagari |
| **Tokenization** | No universal word boundary rules | Chinese/Japanese have no spaces; German has compound words |
| **Morphological complexity** | Languages vary in inflection richness | Turkish is agglutinative; Finnish has 15+ cases |
| **Resource imbalance** | Most NLP data is in English | 7000+ languages, but robust models for <100 |
| **Code-switching** | Mixing languages in one text | "Let's meet at the café, ça sera super" |
| **Transliteration** | Same language in different scripts | Hindi in Devanagari vs Romanized Hindi |

**Multilingual Models Comparison:**

| Model | Languages | Architecture | Key Feature |
|-------|-----------|-------------|-------------|
| **mBERT** | 104 languages | BERT (Transformer encoder) | Shared WordPiece vocabulary (110K tokens) |
| **XLM-RoBERTa** | 100 languages | RoBERTa (improved BERT) | Trained on 2.5TB CommonCrawl data |
| **mT5** | 101 languages | T5 (encoder-decoder) | Text-to-text framework for all tasks |
| **BLOOM** | 46 languages | GPT-style (decoder) | Open-source multilingual LLM |
| **NLLB** | 200 languages | Encoder-decoder | Focus on low-resource translation |

**Multilingual Tooling:**

| Library | Multilingual Support | Languages |
|---------|---------------------|----------|
| **spaCy** | Pre-trained models for 25+ languages | `xx_ent_wiki_sm` for multi-language NER |
| **Hugging Face** | All multilingual transformers | 100+ via model hub |
| **Stanza (Stanford)** | Neural pipeline for 66 languages | Tokenization, POS, NER, parsing |
| **polyglot** | NER, sentiment for 130+ languages | Lightweight, embedding-based |
| **fastText** | Language detection, embeddings | 157 languages |

**Python Code Example:**

```python
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

# Pipeline: Detect Language → Select/Use Multilingual Model → Process Text → Cross-lingual Analysis

# 1. Language Detection with fastText-style approach
texts = {
    "en": "Machine learning is transforming technology.",
    "fr": "L'apprentissage automatique transforme la technologie.",
    "de": "Maschinelles Lernen verändert die Technologie.",
    "es": "El aprendizaje automático está transformando la tecnología.",
    "ja": "機械学習はテクノロジーを変革しています。"
}

# 2. Multilingual NER using XLM-RoBERTa
ner_pipeline = pipeline(
    "ner",
    model="Davlan/xlm-roberta-base-ner-hrl",
    aggregation_strategy="simple"
)

test_texts = [
    "Barack Obama visited Berlin and met Angela Merkel.",        # English
    "Emmanuel Macron a rencontré Joe Biden à Paris.",            # French
    "安倍晋三は東京でバイデン大統領と会談した。",                           # Japanese
]

print("=== Multilingual NER ===")
for text in test_texts:
    entities = ner_pipeline(text)
    print(f"\nText: {text}")
    for ent in entities:
        print(f"  {ent['word']:20s} → {ent['entity_group']:10s} (score: {ent['score']:.3f})")

# 3. Multilingual sentence embeddings for cross-lingual similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

sentences = [
    "The cat is on the table",       # English
    "Le chat est sur la table",      # French (same meaning)
    "The weather is beautiful today"  # English (different meaning)
]

embeddings = model.encode(sentences)
from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity(embeddings)
print("\n=== Cross-lingual Similarity ===")
print(f"EN-FR (same meaning): {sims[0][1]:.4f}")
print(f"EN-EN (diff meaning): {sims[0][2]:.4f}")
```

**Interview Tips:**
- **Zero-shot cross-lingual transfer** is a powerful technique — fine-tune on English data, apply to other languages via multilingual models
- The key to multilingual models is a **shared subword vocabulary** (e.g., BPE/SentencePiece) that allows parameter sharing across languages
- For production systems, mention the **language detection step** as a prerequisite using fastText's `lid.176.bin` or langdetect
- **Romanization/transliteration** may be needed for languages where text appears in non-standard scripts (e.g., Romanized Hindi)
- Trade-off: multilingual models sacrifice per-language performance for breadth — a **language-specific model** will usually outperform a multilingual one on that language
- Mention **XTREME** and **XGLUE** benchmarks for evaluating multilingual model performance

---

## Question 9
- [ ] Done

**How can topic modeling be used in analyzing large collections of documents?**

**Answer:**

**Definition:** Topic modeling is an unsupervised machine learning technique that discovers abstract "topics" (clusters of co-occurring words) in a large collection of documents. Each document is represented as a mixture of topics, and each topic is a probability distribution over words, enabling automated thematic analysis without labeled data.

**Popular Topic Modeling Algorithms:**

| Algorithm | Type | Key Idea | Strengths | Weaknesses |
|-----------|------|----------|-----------|------------|
| **LDA** (Latent Dirichlet Allocation) | Probabilistic generative | Documents are mixtures of topics; topics are distributions over words | Interpretable, well-studied | Needs pre-set K, slow on large data |
| **NMF** (Non-negative Matrix Factorization) | Matrix factorization | Factorize term-document matrix into non-negative topic matrices | Faster, sparser topics | Less principled probabilistic interpretation |
| **LSA/LSI** | Matrix factorization (SVD) | Reduce term-document matrix via truncated SVD | Captures latent semantics, fast | Topics can have negative values, less interpretable |
| **BERTopic** | Neural | Cluster document embeddings, then extract topic words via c-TF-IDF | Leverages transformers, dynamic topics | Computationally heavier |
| **Top2Vec** | Neural | Joint document/word embeddings, clusters in embedding space | No need to specify K, finds topics automatically | Less customizable |

**LDA Generative Process:**

| Step | Action | Notation |
|------|--------|----------|
| 1 | For each document, draw a topic distribution | θ_d ~ Dirichlet(α) |
| 2 | For each word position, choose a topic | z_{dn} ~ Multinomial(θ_d) |
| 3 | Choose a word from that topic's word distribution | w_{dn} ~ Multinomial(φ_{z_{dn}}) |
| 4 | Each topic's word distribution is drawn from | φ_k ~ Dirichlet(β) |

**Applications of Topic Modeling:**

| Application | Use Case |
|------------|----------|
| **Document organization** | Automatically categorize/tag documents in large archives |
| **Content recommendation** | Recommend articles with similar topic distributions |
| **Trend analysis** | Track how topics evolve over time (Dynamic Topic Models) |
| **Customer feedback** | Discover themes in reviews, support tickets, surveys |
| **Academic research** | Analyze themes across scientific papers |
| **Legal/compliance** | Categorize legal documents, detect relevant regulations |

**Python Code Example:**

```python
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

# Pipeline: Documents → Vectorize (BoW/TF-IDF) → Fit Topic Model → Extract Top Words per Topic → Analyze

documents = [
    "Machine learning algorithms improve prediction accuracy in healthcare applications",
    "Deep neural networks achieve state of the art results in image recognition",
    "Natural language processing helps computers understand human language text",
    "Stock market prediction using machine learning and financial data analysis",
    "Computer vision and deep learning for autonomous driving vehicles",
    "Sentiment analysis of customer reviews using NLP text classification",
    "Reinforcement learning agents play games and optimize robot control",
    "Healthcare diagnosis systems powered by artificial intelligence technology",
    "Word embeddings capture semantic relationships between words in text",
    "Financial trading algorithms use predictive models for market analysis",
]

# 1. Vectorize documents
count_vec = CountVectorizer(max_df=0.9, min_df=1, stop_words='english')
bow_matrix = count_vec.fit_transform(documents)
feature_names = count_vec.get_feature_names_out()

# 2. Fit LDA model
n_topics = 3
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    max_iter=20
)
lda.fit(bow_matrix)

# 3. Display topics
def display_topics(model, features, n_words=5):
    for idx, topic in enumerate(model.components_):
        top_words = [features[i] for i in topic.argsort()[-n_words:][::-1]]
        print(f"  Topic {idx}: {', '.join(top_words)}")

print("=== LDA Topics ===")
display_topics(lda, feature_names)

# 4. Get document-topic distribution
doc_topics = lda.transform(bow_matrix)
print("\n=== Document-Topic Distribution ===")
for i, doc in enumerate(documents[:3]):
    top_topic = np.argmax(doc_topics[i])
    print(f"  Doc {i} (topic {top_topic}): {doc_topics[i].round(3)}")
    print(f"    '{doc[:60]}...'")

# 5. Model evaluation
perplexity = lda.perplexity(bow_matrix)
print(f"\nPerplexity: {perplexity:.2f} (lower is better)")
```

**Interview Tips:**
- LDA's hyperparameters: **α** (document-topic density — higher means more topics per document), **β** (topic-word density — higher means more words per topic), **K** (number of topics)
- Use **coherence score** (not just perplexity) to evaluate topic quality — it measures how semantically similar the top words in a topic are
- For choosing K, use **elbow method** on coherence scores or domain knowledge
- **BERTopic** is the modern go-to: it clusters document embeddings (UMAP + HDBSCAN) then extracts topics via c-TF-IDF, producing more coherent topics than LDA
- LDA assumes a **bag-of-words** representation, so word order is lost — mention this as a limitation
- For streaming/online scenarios, use **Online LDA** which processes documents in mini-batches

---

## Question 10
- [ ] Done

**How do you handle noisy text data from sources like social media for NLP tasks?**

**Answer:**

**Definition:** Noisy text refers to text that deviates from standard language conventions — including misspellings, slang, abbreviations, hashtags, emojis, irregular grammar, and mixed languages. Handling it requires specialized preprocessing, normalization techniques, and models robust to such variation.

**Types of Noise in Social Media Text:**

| Noise Type | Example | Challenge |
|-----------|---------|----------|
| **Misspellings/Typos** | "teh" → "the", "recieve" → "receive" | Vocabulary explosion, OOV tokens |
| **Abbreviations/Acronyms** | "u" → "you", "brb" → "be right back" | Non-standard vocabulary |
| **Slang/Informal** | "lit", "slay", "no cap" | Evolving vocabulary, domain shift |
| **Hashtags** | #MachineLearningIsFun | Compound words without spaces |
| **Emojis/Emoticons** | 😂 🔥 :) | Carry sentiment but are non-textual |
| **@mentions/URLs** | @user123, https://... | Noise for most NLP tasks |
| **Repetition** | "sooooo gooood" | Non-standard spelling for emphasis |
| **Code-switching** | "That's so كويس" (mixing English/Arabic) | Multiple languages in one text |
| **Missing punctuation** | "i cant believe this no way" | Hard to detect sentence boundaries |

**Preprocessing Pipeline for Noisy Text:**

| Step | Technique | Purpose |
|------|-----------|--------|
| 1. URL/mention removal | Regex `r'https?://\S+'`, `r'@\w+'` | Remove non-content tokens |
| 2. HTML entity decoding | `html.unescape()` | Convert &amp; → & |
| 3. Emoji handling | Convert to text or extract as features | Preserve sentiment signals |
| 4. Hashtag segmentation | CamelCase splitting or WordNinja | #MachineLearning → Machine Learning |
| 5. Spell correction | SymSpell, TextBlob, Hunspell | Normalize misspellings |
| 6. Abbreviation expansion | Custom dictionary lookup | "u" → "you", "brb" → "be right back" |
| 7. Character normalization | Collapse repeated chars | "soooo" → "so" |
| 8. Lowercasing | `.lower()` (with care) | Normalize case |
| 9. Unicode normalization | `unicodedata.normalize('NFKC', text)` | Standardize character encodings |

**Python Code Example:**

```python
import re
import html
import unicodedata

# Pipeline: Raw Social Media Text → URL/Mention Removal → Emoji Handling → Spell Fix → Normalize → Clean Text

# Abbreviation dictionary
ABBREVIATIONS = {
    "u": "you", "ur": "your", "r": "are", "b4": "before",
    "brb": "be right back", "imo": "in my opinion", "tbh": "to be honest",
    "lol": "laughing out loud", "smh": "shaking my head",
    "ngl": "not going to lie", "idk": "I don't know",
}

def clean_social_media_text(text):
    """Comprehensive noisy text cleaning pipeline."""
    # Step 1: Decode HTML entities
    text = html.unescape(text)

    # Step 2: Unicode normalization
    text = unicodedata.normalize('NFKC', text)

    # Step 3: Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)

    # Step 4: Handle @mentions
    text = re.sub(r'@\w+', '<USER>', text)

    # Step 5: Segment hashtags (CamelCase split)
    def split_hashtag(match):
        tag = match.group(1)
        words = re.sub(r'([A-Z])', r' \1', tag).strip()
        return words
    text = re.sub(r'#(\w+)', split_hashtag, text)

    # Step 6: Normalize repeated characters (3+ → 2)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Step 7: Expand abbreviations
    words = text.split()
    words = [ABBREVIATIONS.get(w.lower(), w) for w in words]
    text = ' '.join(words)

    # Step 8: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Test with noisy social media posts
noisy_texts = [
    "OMG @john_doe this is sooooo amazing!!! 😂😂 #MachineLearningIsFun https://t.co/abc123",
    "u r not going to believe this brb... tbh idk what happened lol",
    "The product is TERRRRIBLE &amp; i want my $$ back!!! @support_team #WorstExperience",
    "ngl this new feature is 🔥🔥🔥 #DeepLearning #AI",
]

print("=== Social Media Text Cleaning ===")
for text in noisy_texts:
    cleaned = clean_social_media_text(text)
    print(f"  Original: {text}")
    print(f"  Cleaned:  {cleaned}\n")
```

**Interview Tips:**
- Don't blindly remove all "noise" — **emojis carry sentiment** (😂 = positive), **capitalization signals emphasis**, repetition indicates intensity
- For production systems, consider using **models trained on noisy data** (e.g., BERTweet, TweetEval) instead of heavy preprocessing
- Mention the **trade-off**: heavy normalization loses information but helps traditional models; modern transformers are more robust to noise
- **Spell correction** can introduce errors — "ur" in context might mean "your" or be a valid abbreviation; context matters
- For multilingual social media, **language detection per segment** helps handle code-switching
- Mention that social media language **evolves rapidly** — static abbreviation dictionaries become outdated; consider data-driven approaches

---

## Question 11
- [ ] Done

**How do you address the issue of data scarcity when working with less-resourced languages?**

**Answer:**

**Definition:** Low-resource languages are languages for which there is insufficient annotated training data, pre-trained models, or NLP tools. Addressing data scarcity requires a combination of transfer learning, data augmentation, active learning, and community-driven data collection strategies.

**Strategies for Low-Resource Languages:**

| Strategy | Description | Effectiveness | Cost |
|----------|-----------|---------------|------|
| **Cross-lingual transfer** | Fine-tune multilingual model on high-resource language, apply to low-resource | High | Low |
| **Zero-shot transfer** | Use multilingual model (mBERT, XLM-R) directly without any target language data | Moderate | Very low |
| **Few-shot learning** | Train with very few labeled examples in target language | Moderate-High | Low |
| **Data augmentation** | Create synthetic training data via back-translation, paraphrasing | Moderate | Low |
| **Active learning** | Strategically select most informative examples for annotation | High | Moderate |
| **Self-training** | Train on labeled data, predict on unlabeled, retrain on confident predictions | Moderate | Low |
| **Distant supervision** | Use existing knowledge bases to automatically create labels | Moderate | Low |
| **Community annotation** | Crowdsource labels from native speakers | High | Moderate-High |

**Data Augmentation Techniques:**

| Technique | How It Works | Best For |
|-----------|-------------|----------|
| **Back-translation** | Translate to pivot language and back | Machine translation, classification |
| **Synonym replacement** | Replace words with dictionary/embedding synonyms | Text classification |
| **Random insertion/deletion/swap** | EDA (Easy Data Augmentation) operations | General text tasks |
| **Cross-lingual word replacement** | Replace words with aligned words from related language | Related language families |
| **Paraphrasing** | Use LLM to rephrase sentences | All tasks |
| **Code-switching augmentation** | Mix target language with related high-resource language | Multilingual tasks |

**Transfer Learning Approaches:**

| Approach | Method | Data Required |
|----------|--------|---------------|
| **Feature-based** | Use multilingual embeddings as features | Zero target labels possible |
| **Fine-tuning** | Fine-tune multilingual model on small target data | Few hundred examples |
| **Translate-train** | Translate high-resource training data into target language | Parallel corpus or MT system |
| **Translate-test** | Translate target text to high-resource language for inference | MT system only |
| **Multi-task learning** | Train on multiple languages/tasks jointly | Mix of resources |

**Python Code Example:**

```python
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import random

# Pipeline: Cross-lingual Transfer → Data Augmentation → Few-shot Fine-tuning → Evaluate on Target Language

# 1. Zero-shot cross-lingual classification using XLM-RoBERTa
classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli"
)

# Test on low-resource language (Swahili) with English labels
swahili_text = "Serikali imetangaza mpango mpya wa elimu"  # "Government announced new education plan"
result = classifier(
    swahili_text,
    candidate_labels=["politics", "education", "sports", "health"],
)
print("=== Zero-shot Cross-lingual Classification ===")
print(f"Text (Swahili): {swahili_text}")
for label, score in zip(result['labels'], result['scores']):
    print(f"  {label:15s}: {score:.4f}")

# 2. Easy Data Augmentation (EDA) for low-resource text
def eda_augment(text, n_aug=3):
    """Simple EDA: random swap, delete, insert operations."""
    words = text.split()
    augmented = []
    for _ in range(n_aug):
        new_words = words.copy()
        operation = random.choice(['swap', 'delete', 'duplicate'])
        if len(new_words) < 3:
            augmented.append(' '.join(new_words))
            continue
        idx = random.randint(0, len(new_words) - 2)
        if operation == 'swap':
            new_words[idx], new_words[idx+1] = new_words[idx+1], new_words[idx]
        elif operation == 'delete':
            new_words.pop(idx)
        elif operation == 'duplicate':
            new_words.insert(idx, new_words[idx])
        augmented.append(' '.join(new_words))
    return augmented

print("\n=== Data Augmentation (EDA) ===")
original = "The government announced a new education policy"
print(f"Original:  {original}")
for i, aug in enumerate(eda_augment(original)):
    print(f"Augmented {i+1}: {aug}")

# 3. Back-translation augmentation (conceptual)
print("\n=== Back-Translation Strategy ===")
print("1. Original (Swahili): 'Serikali imetangaza mpango mpya wa elimu'")
print("2. Translate to English: 'Government announced new education plan'")
print("3. Translate back to Swahili: 'Serikali ilitangaza programu mpya ya elimu'")
print("4. Now we have 2 Swahili training examples from 1 original")
```

**Interview Tips:**
- **Multilingual pre-trained models** (XLM-R, mBERT) are the most practical starting point — they provide surprisingly good zero-shot transfer
- Back-translation is the **single most effective augmentation technique** for NLP — it preserves semantics while varying surface form
- Mention the **language family advantage**: closely related languages transfer better (e.g., Spanish → Portuguese > English → Japanese)
- **Active learning** is critical when annotation budget is limited — select examples the model is most uncertain about
- For truly low-resource languages (no digital text), consider **speech-first approaches** since many languages have oral traditions but limited written resources
- The **Masakhane** and **AmericasNLP** communities are important references for African and Indigenous American language NLP

---

## Question 12
- [ ] Done

**What measures can be taken to reduce bias in NLP models?**

**Answer:**

**Definition:** Bias in NLP refers to systematic prejudices in model behavior that reflect and amplify societal biases present in training data — including stereotypes related to gender, race, age, religion, and other protected attributes. Reducing bias requires interventions at every stage: data collection, model training, evaluation, and deployment.

**Types of Bias in NLP:**

| Bias Type | Description | Example |
|-----------|-------------|--------|
| **Selection bias** | Training data not representative of target population | Models trained mostly on English Wikipedia underrepresent informal registers |
| **Stereotypical bias** | Associations that reflect societal stereotypes | "He is a doctor" vs "She is a nurse" — gendered occupation associations |
| **Representation bias** | Underrepresentation of certain groups | Sentiment models perform worse on African American Vernacular English |
| **Measurement bias** | Biased labels from annotators | Toxicity classifiers flagging minority dialect as "toxic" |
| **Aggregation bias** | One model for all subgroups ignoring differences | Same model for all demographics despite differing language patterns |
| **Temporal bias** | Training data doesn't reflect current norms | Historical text containing outdated or offensive terminology |

**Debiasing Strategies by Stage:**

| Stage | Strategy | Description |
|-------|----------|-------------|
| **Pre-processing (Data)** | Balanced dataset curation | Ensure diverse representation across demographics |
| | Data augmentation | Counterfactual augmentation (swap gender/race terms) |
| | Filtering | Remove toxic or heavily biased training examples |
| **In-processing (Training)** | Adversarial debiasing | Train adversary to predict protected attribute; penalize model if detectable |
| | Constraint-based training | Add fairness constraints to loss function |
| | Embedding debiasing | Project out bias subspace from word embeddings |
| **Post-processing (Output)** | Calibration | Adjust output probabilities to be fair across groups |
| | Filtering / guardrails | Block or modify biased outputs at inference time |
| | Human-in-the-loop | Human review of high-stakes decisions |
| **Evaluation** | Disaggregated metrics | Report performance per demographic group |
| | Bias benchmarks | Test on WinoBias, BBQ, BOLD, CrowS-Pairs |
| | Red teaming | Adversarial testing for bias |

**Embedding Debiasing Techniques:**

| Technique | Method | Limitation |
|-----------|--------|------------|
| **Hard debiasing** (Bolukbasi et al.) | Identify bias direction, project it out | Only removes linear bias |
| **Counterfactual augmentation** | "He is a nurse" + "She is a nurse" → equal representation | Doesn't capture intersectional bias |
| **INLP** (Iterative Null-space Projection) | Iteratively remove linear predictability of protected attribute | May remove useful information |
| **Sent-Debias** | Extend debiasing to sentence-level embeddings | Computationally expensive |

**Python Code Example:**

```python
import numpy as np
from sklearn.decomposition import PCA

# Pipeline: Load Embeddings → Identify Bias Direction → Measure Bias → Apply Debiasing → Verify Reduction

# Simulated word embeddings (in practice, use pre-trained word2vec/GloVe)
np.random.seed(42)
dim = 50

def make_embedding(base, bias_shift=0.0):
    vec = np.random.randn(dim) * 0.1
    vec[0] = bias_shift  # gender dimension
    return vec / np.linalg.norm(vec)

# Create biased embeddings
embeddings = {
    'he':        make_embedding(None, bias_shift=1.0),
    'she':       make_embedding(None, bias_shift=-1.0),
    'man':       make_embedding(None, bias_shift=0.8),
    'woman':     make_embedding(None, bias_shift=-0.8),
    'doctor':    make_embedding(None, bias_shift=0.5),   # biased male
    'nurse':     make_embedding(None, bias_shift=-0.5),  # biased female
    'engineer':  make_embedding(None, bias_shift=0.6),   # biased male
    'teacher':   make_embedding(None, bias_shift=-0.3),  # biased female
    'programmer': make_embedding(None, bias_shift=0.4),
}

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 1. Identify gender direction
gender_direction = embeddings['he'] - embeddings['she']
gender_direction = gender_direction / np.linalg.norm(gender_direction)

# 2. Measure bias before debiasing
print("=== Bias BEFORE Debiasing ===")
for word in ['doctor', 'nurse', 'engineer', 'teacher', 'programmer']:
    bias_score = np.dot(embeddings[word], gender_direction)
    direction = "male" if bias_score > 0 else "female"
    print(f"  {word:12s}: bias = {bias_score:+.4f} ({direction}-leaning)")

# 3. Hard debiasing: remove gender component from occupation words
def debias_word(embedding, bias_direction):
    projection = np.dot(embedding, bias_direction) * bias_direction
    return embedding - projection

neutral_words = ['doctor', 'nurse', 'engineer', 'teacher', 'programmer']
debiased = {}
for word in neutral_words:
    debiased[word] = debias_word(embeddings[word], gender_direction)

# 4. Measure bias after debiasing
print("\n=== Bias AFTER Debiasing ===")
for word in neutral_words:
    bias_score = np.dot(debiased[word], gender_direction)
    print(f"  {word:12s}: bias = {bias_score:+.4f} (≈ neutral)")
```

**Interview Tips:**
- Bias is a **sociotechnical problem** — purely technical solutions are insufficient; diverse teams, clear guidelines, and stakeholder input are essential
- The **WEAT** (Word Embedding Association Test) is a standard metric for measuring bias in embeddings, inspired by the Implicit Association Test
- Mention the **fairness-accuracy trade-off** — debiasing can sometimes reduce model performance; the goal is equitable performance across groups
- **Counterfactual fairness**: a prediction is fair if it would be the same in a counterfactual world where the individual's protected attribute was different
- For LLMs, **RLHF** (Reinforcement Learning from Human Feedback) and **Constitutional AI** are key techniques for reducing harmful outputs
- Reference important benchmarks: **WinoBias** (gender), **BBQ** (social bias), **CrowS-Pairs** (stereotypes), **BOLD** (open-ended generation)

---

## Question 13
- [ ] Done

**Outline your approach to develop a recommendation system based on textual content analysis.**

**Answer:**

**Definition:** A content-based recommendation system analyzes the textual attributes of items (descriptions, reviews, metadata) and user preferences to recommend items whose content is most similar to what a user has liked in the past. Unlike collaborative filtering, it doesn't require other users' behavior data.

**System Architecture Overview:**

| Stage | Component | Purpose |
|-------|-----------|--------|
| 1. **Data Collection** | Gather item descriptions, reviews, metadata | Build the content knowledge base |
| 2. **Text Preprocessing** | Tokenize, clean, normalize text | Prepare text for feature extraction |
| 3. **Feature Extraction** | Convert text to numerical representations | Create item and user profiles |
| 4. **Profile Building** | Build user preference profiles from interaction history | Capture user taste/interests |
| 5. **Similarity Computation** | Calculate item-item or user-item similarity | Identify relevant recommendations |
| 6. **Ranking & Filtering** | Score and rank candidates, apply business rules | Deliver top-N recommendations |
| 7. **Evaluation** | Measure quality of recommendations | Ensure system effectiveness |

**Text Representation Methods (from basic to advanced):**

| Method | Approach | Quality | Speed |
|--------|---------|---------|-------|
| **TF-IDF** | Sparse weighted bag-of-words | Baseline | Very fast |
| **Doc2Vec** | Learned document embeddings | Good | Fast |
| **Word2Vec + Averaging** | Average word embeddings per document | Good | Fast |
| **Sentence-BERT** | Dense sentence/document embeddings from transformer | Very good | Moderate |
| **LLM Embeddings** | embeddings from models like OpenAI, Cohere | Best | Slower, API cost |

**Similarity Metrics:**

| Metric | Formula Concept | Best For |
|--------|----------------|----------|
| **Cosine similarity** | Angle between vectors | TF-IDF and embedding vectors |
| **Euclidean distance** | L2 norm of difference | Dense embeddings |
| **Jaccard similarity** | Set overlap of tokens | Keyword/tag-based matching |
| **BM25** | Probabilistic term relevance | Search-like retrieval |

**Evaluation Metrics:**

| Metric | What It Measures |
|--------|------------------|
| **Precision@K** | Fraction of top-K recommendations that are relevant |
| **Recall@K** | Fraction of relevant items found in top-K |
| **NDCG** | Quality of ranking order (position-sensitive) |
| **MAP** | Mean average precision across queries |
| **Coverage** | Fraction of catalog items that get recommended |
| **Diversity** | How varied the recommendations are |

**Python Code Example:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Pipeline: Item Descriptions → TF-IDF Vectorize → Build User Profile → Compute Similarity → Rank → Recommend Top-N

# 1. Item catalog with textual descriptions
items = {
    0: {"title": "Intro to Machine Learning",
        "desc": "A comprehensive guide to machine learning algorithms, supervised and unsupervised learning, neural networks, and model evaluation."},
    1: {"title": "Deep Learning with Python",
        "desc": "Hands-on deep learning using Keras and TensorFlow, covering CNNs, RNNs, GANs, and transfer learning for image and text."},
    2: {"title": "NLP Fundamentals",
        "desc": "Natural language processing with transformers, tokenization, named entity recognition, sentiment analysis, and text classification."},
    3: {"title": "Data Visualization Cookbook",
        "desc": "Creating beautiful charts and dashboards with matplotlib, seaborn, plotly for exploratory data analysis and storytelling."},
    4: {"title": "Statistics for Data Science",
        "desc": "Probability, hypothesis testing, regression analysis, Bayesian statistics, and statistical inference for data science."},
    5: {"title": "Advanced NLP with Transformers",
        "desc": "BERT, GPT, T5 architectures, fine-tuning transformers for question answering, summarization, and language generation."},
    6: {"title": "Computer Vision Projects",
        "desc": "Image classification, object detection, segmentation using deep learning, CNNs, YOLO, and vision transformers."},
}

# 2. Vectorize all item descriptions
descriptions = [items[i]["desc"] for i in sorted(items.keys())]
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
item_vectors = vectorizer.fit_transform(descriptions)

# 3. Build user profile from interaction history (items the user liked)
def build_user_profile(liked_item_ids, item_matrix):
    """Average the vectors of items the user liked."""
    liked_vectors = item_matrix[liked_item_ids]
    return liked_vectors.mean(axis=0)

# User liked "NLP Fundamentals" (id=2) and "Advanced NLP" (id=5)
user_liked = [2, 5]
user_profile = build_user_profile(user_liked, item_vectors)

# 4. Compute similarity between user profile and all items
similarities = cosine_similarity(user_profile, item_vectors).flatten()

# 5. Rank and recommend (exclude already-liked items)
def recommend(similarities, liked_ids, items_dict, top_n=3):
    scored = []
    for idx, score in enumerate(similarities):
        if idx not in liked_ids:
            scored.append((idx, score, items_dict[idx]["title"]))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

print("=== User liked: ===")
for i in user_liked:
    print(f"  • {items[i]['title']}")

print("\n=== Top Recommendations: ===")
for rank, (idx, score, title) in enumerate(recommend(similarities, user_liked, items), 1):
    print(f"  {rank}. {title} (similarity: {score:.4f})")

print("\n=== All Similarity Scores: ===")
for idx in np.argsort(-similarities):
    marker = " ← liked" if idx in user_liked else ""
    print(f"  {items[idx]['title']:35s} : {similarities[idx]:.4f}{marker}")
```

**Interview Tips:**
- Content-based systems solve the **cold-start problem for new items** (unlike collaborative filtering) because they only need item descriptions, not user interactions
- The main weakness is the **filter bubble** — they only recommend items similar to what users already like; hybrid systems combining content and collaborative filtering mitigate this
- For scalability, use **approximate nearest neighbor** (ANN) search libraries like FAISS, Annoy, or Milvus instead of brute-force cosine similarity
- Modern systems use **two-stage architecture**: fast retrieval (ANN search) → precise re-ranking (cross-encoder or LLM)
- Mention **embedding-based retrieval** as the modern standard — encode items and queries into the same dense vector space using sentence transformers
- For handling new users (user cold-start), use **onboarding preferences**, popular items, or demographic-based recommendations

---

