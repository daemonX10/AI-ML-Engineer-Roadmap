# Embeddings & Vector Systems - Theory Questions

## Embedding Fundamentals

### Question 1
- [ ] Done
**How do you choose between Word2Vec, GloVe, FastText, and BERT embeddings for your NLP task?**

**Answer:**

**Definition:**
Choose based on: **task requirements** (static vs contextual), **computational budget**, **vocabulary coverage**, and **accuracy needs**. BERT provides contextual embeddings for highest accuracy; FastText handles OOV well; Word2Vec/GloVe are efficient for simpler tasks or constrained environments.

**Decision Framework:**

| Requirement | Best Choice | Reason |
|-------------|-------------|--------|
| **Highest accuracy** | BERT/Sentence-BERT | Contextualized representations |
| **Real-time inference** | Word2Vec/GloVe | Fast lookup |
| **OOV handling** | FastText | Subword embeddings |
| **Limited compute** | Word2Vec/GloVe | No model inference |
| **Morphological languages** | FastText | Captures word structure |
| **Domain-specific** | Fine-tuned BERT | Transfer learning |

**Embedding Comparison:**

| Aspect | Word2Vec | GloVe | FastText | BERT |
|--------|----------|-------|----------|------|
| **Type** | Static | Static | Static | Contextual |
| **Training** | Predictive | Co-occurrence | Predictive+subword | Masked LM |
| **OOV** | ❌ | ❌ | ✓ (subwords) | ✓ (tokenizer) |
| **Speed** | Fast | Fast | Fast | Slow |
| **Memory** | Low | Low | Medium | High |
| **Quality** | Good | Good | Better | Best |
| **Dimension** | 100-300 | 100-300 | 100-300 | 768-1024 |

**Python Code Example:**
```python
import numpy as np
from typing import List, Dict

class EmbeddingSelector:
    """Help select appropriate embedding for task"""
    
    def __init__(self):
        self.requirements = {}
    
    def analyze_requirements(self, 
                            need_contextual: bool,
                            latency_ms: float,
                            memory_mb: float,
                            oov_percentage: float,
                            accuracy_priority: str) -> str:
        """Recommend embedding based on requirements"""
        
        recommendations = []
        
        # Contextual requirement
        if need_contextual:
            recommendations.append(('BERT', 5))
        else:
            recommendations.append(('Word2Vec', 3))
            recommendations.append(('GloVe', 3))
        
        # Latency constraint
        if latency_ms < 10:
            recommendations.append(('Word2Vec', 4))
            recommendations.append(('GloVe', 4))
            recommendations.append(('FastText', 3))
        elif latency_ms < 100:
            recommendations.append(('FastText', 3))
            recommendations.append(('DistilBERT', 2))
        else:
            recommendations.append(('BERT', 3))
        
        # OOV handling
        if oov_percentage > 10:
            recommendations.append(('FastText', 5))
            recommendations.append(('BERT', 3))
        
        # Aggregate scores
        scores = {}
        for model, score in recommendations:
            scores[model] = scores.get(model, 0) + score
        
        return max(scores.keys(), key=lambda x: scores[x])

# Loading different embeddings
def load_word2vec(path: str):
    from gensim.models import KeyedVectors
    return KeyedVectors.load_word2vec_format(path, binary=True)

def load_glove(path: str, dim: int = 300):
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def load_fasttext(path: str):
    import fasttext
    return fasttext.load_model(path)

def load_bert(model_name: str = 'bert-base-uncased'):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

# Unified embedding interface
class UnifiedEmbedder:
    def __init__(self, embedding_type: str, model_path: str = None):
        self.embedding_type = embedding_type
        self._load_model(model_path)
    
    def _load_model(self, path):
        if self.embedding_type == 'bert':
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model = AutoModel.from_pretrained('bert-base-uncased')
        elif self.embedding_type == 'word2vec':
            from gensim.models import KeyedVectors
            self.model = KeyedVectors.load_word2vec_format(path, binary=True)
        # ... other loaders
    
    def embed(self, text: str) -> np.ndarray:
        if self.embedding_type == 'bert':
            return self._bert_embed(text)
        elif self.embedding_type == 'word2vec':
            return self._static_embed(text)
    
    def _bert_embed(self, text: str) -> np.ndarray:
        import torch
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling
        return outputs.last_hidden_state.mean(dim=1).numpy()[0]
    
    def _static_embed(self, text: str) -> np.ndarray:
        words = text.lower().split()
        vectors = [self.model[w] for w in words if w in self.model]
        return np.mean(vectors, axis=0) if vectors else np.zeros(300)
```

**Interview Tips:**
- BERT for accuracy-critical tasks; static embeddings for speed
- FastText is best for morphologically rich languages (German, Turkish)
- Consider Sentence-BERT for sentence-level similarity tasks
- Production often uses static embeddings + BERT fallback for important queries

---

### Question 2
- [ ] Done
**What are the computational trade-offs between static embeddings (Word2Vec) versus contextualized embeddings (BERT)?**

**Answer:**

**Definition:**
Static embeddings (Word2Vec, GloVe) use **precomputed lookup tables** (O(1) per word), while contextualized embeddings (BERT) require **model inference** for each input (O(n·d²) per sentence). Trade-off: static is 100-1000x faster but loses context; BERT captures polysemy and context at high compute cost.

**Computational Comparison:**

| Aspect | Static (Word2Vec) | Contextualized (BERT) |
|--------|-------------------|----------------------|
| **Lookup time** | O(1) | O(n·L·d²) |
| **Memory (vocab 100K)** | ~120MB | ~440MB (model) |
| **Latency per sentence** | <1ms | 10-100ms |
| **Batch throughput** | ~100K/sec | ~100-1000/sec |
| **GPU required** | No | Recommended |
| **Handles polysemy** | No | Yes |

**Cost Analysis:**

| Metric | Word2Vec | BERT-base | BERT-large |
|--------|----------|-----------|------------|
| **Parameters** | 0 (lookup) | 110M | 340M |
| **FLOPS/token** | ~0 | ~22B | ~84B |
| **Memory/batch** | O(batch·dim) | O(batch·seq·dim) | O(batch·seq·dim) |
| **GPU memory** | 0 | 1-2GB | 4-8GB |

**Python Code Example:**
```python
import numpy as np
import time
from typing import List

class EmbeddingBenchmark:
    """Compare embedding computational costs"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_static(self, embeddings: dict, texts: List[str], 
                         num_runs: int = 100) -> dict:
        """Benchmark static embedding lookup"""
        
        def embed_text(text):
            words = text.lower().split()
            vectors = [embeddings[w] for w in words if w in embeddings]
            return np.mean(vectors, axis=0) if vectors else np.zeros(300)
        
        # Warmup
        for text in texts[:10]:
            embed_text(text)
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(num_runs):
            for text in texts:
                embed_text(text)
        elapsed = time.perf_counter() - start
        
        return {
            'total_time': elapsed,
            'per_text_ms': (elapsed / (num_runs * len(texts))) * 1000,
            'throughput': (num_runs * len(texts)) / elapsed
        }
    
    def benchmark_bert(self, model, tokenizer, texts: List[str],
                       num_runs: int = 10) -> dict:
        """Benchmark BERT embedding extraction"""
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Warmup
        inputs = tokenizer(texts[:5], return_tensors='pt', 
                          padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_runs):
            inputs = tokenizer(texts, return_tensors='pt',
                              padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        return {
            'total_time': elapsed,
            'per_text_ms': (elapsed / (num_runs * len(texts))) * 1000,
            'throughput': (num_runs * len(texts)) / elapsed,
            'device': str(device)
        }
    
    def compare(self, static_results: dict, bert_results: dict) -> dict:
        """Compare static vs BERT performance"""
        speedup = bert_results['per_text_ms'] / static_results['per_text_ms']
        
        return {
            'static_latency_ms': static_results['per_text_ms'],
            'bert_latency_ms': bert_results['per_text_ms'],
            'speedup_factor': speedup,
            'recommendation': 'static' if speedup > 10 else 'bert'
        }

# Memory estimation
def estimate_memory(embedding_type: str, vocab_size: int = 100000,
                    dim: int = 300, batch_size: int = 32,
                    seq_len: int = 128) -> dict:
    """Estimate memory requirements"""
    
    if embedding_type == 'static':
        # Just the embedding matrix
        embedding_memory = vocab_size * dim * 4  # float32
        inference_memory = batch_size * dim * 4
        
        return {
            'embedding_mb': embedding_memory / 1e6,
            'inference_mb': inference_memory / 1e6,
            'total_mb': (embedding_memory + inference_memory) / 1e6
        }
    
    elif embedding_type == 'bert-base':
        # Model parameters + activations
        params = 110_000_000
        model_memory = params * 4  # float32
        
        # Activations for inference
        hidden_size = 768
        num_layers = 12
        activations = batch_size * seq_len * hidden_size * num_layers * 4
        
        # Attention matrices
        attention = batch_size * 12 * seq_len * seq_len * num_layers * 4
        
        return {
            'model_mb': model_memory / 1e6,
            'activations_mb': activations / 1e6,
            'attention_mb': attention / 1e6,
            'total_mb': (model_memory + activations + attention) / 1e6
        }
```

**Interview Tips:**
- Static embeddings: best for high-throughput, low-latency systems
- BERT: needed when context matters (polysemy, sentiment)
- Hybrid: pre-filter with static, re-rank with BERT
- DistilBERT offers 60% speedup with minimal quality loss

---

### Question 3
- [ ] Done
**How do you handle out-of-vocabulary (OOV) words with different embedding approaches?**

**Answer:**

**Definition:**
OOV handling strategies vary by embedding type: **FastText** uses subword n-grams (built-in), **BERT** uses WordPiece/BPE tokenization (splits unknown words), static embeddings require fallbacks like **random initialization**, **zero vectors**, or **character-level models**. Goal: maintain semantic meaning for unseen words.

**OOV Handling by Embedding Type:**

| Embedding | OOV Strategy | Quality |
|-----------|--------------|---------|
| **Word2Vec** | None (crashes or zero) | Poor |
| **GloVe** | None (crashes or zero) | Poor |
| **FastText** | Subword n-grams | Good |
| **BERT** | WordPiece tokenization | Excellent |
| **Custom** | Character CNN/RNN | Variable |

**Strategies for Static Embeddings:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Zero vector** | Return zeros | Simple fallback |
| **Random init** | Random vector | May hurt similarity |
| **Similar word** | Edit distance match | Typos |
| **Morphological** | Prefix/suffix matching | Related words |
| **Character model** | CharCNN/CharRNN | Any OOV |
| **Subword hashing** | Hash subwords to buckets | Scalable |

**Python Code Example:**
```python
import numpy as np
from collections import defaultdict
import re
from typing import Optional

class OOVHandler:
    """Handle out-of-vocabulary words for static embeddings"""
    
    def __init__(self, embeddings: dict, dim: int = 300):
        self.embeddings = embeddings
        self.dim = dim
        self.vocab = set(embeddings.keys())
        
        # Precompute prefix/suffix indices
        self._build_affix_index()
    
    def _build_affix_index(self, prefix_len: int = 3, suffix_len: int = 3):
        """Build index for morphological matching"""
        self.prefix_index = defaultdict(list)
        self.suffix_index = defaultdict(list)
        
        for word in self.vocab:
            if len(word) >= prefix_len:
                self.prefix_index[word[:prefix_len]].append(word)
            if len(word) >= suffix_len:
                self.suffix_index[word[-suffix_len:]].append(word)
    
    def get_embedding(self, word: str, strategy: str = 'subword') -> np.ndarray:
        """Get embedding with OOV handling"""
        
        word_lower = word.lower()
        
        # Known word
        if word_lower in self.embeddings:
            return self.embeddings[word_lower]
        
        # Apply OOV strategy
        if strategy == 'zero':
            return np.zeros(self.dim)
        
        elif strategy == 'random':
            np.random.seed(hash(word) % (2**32))
            return np.random.randn(self.dim) * 0.1
        
        elif strategy == 'similar':
            return self._find_similar_word(word_lower)
        
        elif strategy == 'morphological':
            return self._morphological_embedding(word_lower)
        
        elif strategy == 'subword':
            return self._subword_embedding(word_lower)
        
        return np.zeros(self.dim)
    
    def _find_similar_word(self, word: str) -> np.ndarray:
        """Find embedding of similar word (edit distance)"""
        
        # Simple approach: check words with same prefix
        if len(word) >= 3:
            prefix = word[:3]
            candidates = self.prefix_index.get(prefix, [])
            
            if candidates:
                # Return embedding of shortest candidate
                best = min(candidates, key=len)
                return self.embeddings[best]
        
        return np.zeros(self.dim)
    
    def _morphological_embedding(self, word: str) -> np.ndarray:
        """Combine prefix and suffix word embeddings"""
        
        vectors = []
        
        # Check prefix matches
        if len(word) >= 3:
            prefix = word[:3]
            prefix_words = self.prefix_index.get(prefix, [])
            if prefix_words:
                vectors.append(self.embeddings[prefix_words[0]])
        
        # Check suffix matches
        if len(word) >= 3:
            suffix = word[-3:]
            suffix_words = self.suffix_index.get(suffix, [])
            if suffix_words:
                vectors.append(self.embeddings[suffix_words[0]])
        
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.dim)
    
    def _subword_embedding(self, word: str, n: int = 3) -> np.ndarray:
        """Average embeddings of character n-grams (FastText-style)"""
        
        # Generate n-grams
        padded = f"<{word}>"
        ngrams = [padded[i:i+n] for i in range(len(padded) - n + 1)]
        
        # Hash n-grams to vocabulary words (simple approach)
        vectors = []
        for ngram in ngrams:
            # Check if ngram appears in any vocab word
            for vocab_word in list(self.vocab)[:1000]:  # Limit for speed
                if ngram in vocab_word:
                    vectors.append(self.embeddings[vocab_word])
                    break
        
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.dim)

# FastText (built-in OOV handling)
def fasttext_oov_demo():
    """FastText handles OOV via subword n-grams"""
    import fasttext
    
    # FastText automatically handles OOV
    model = fasttext.load_model('wiki.en.bin')
    
    # Works for any word, even made-up ones
    known = model.get_word_vector('computer')
    unknown = model.get_word_vector('computering')  # Made-up word
    misspelled = model.get_word_vector('computr')   # Typo
    
    return known, unknown, misspelled

# BERT (tokenizer handles OOV)
def bert_oov_demo():
    """BERT uses WordPiece for OOV"""
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Known word
    known = tokenizer.tokenize('computer')  # ['computer']
    
    # Unknown word gets split
    unknown = tokenizer.tokenize('electronegativity')  
    # ['electron', '##ega', '##tivity']
    
    # Complete gibberish
    gibberish = tokenizer.tokenize('xyzqwerty')
    # ['xyz', '##q', '##wer', '##ty']
    
    return known, unknown, gibberish
```

**Interview Tips:**
- FastText is best for production OOV handling in static embeddings
- BERT's subword tokenization rarely has true OOV issues
- For custom solutions, character-level models are most robust
- Track OOV rate in production — high rates indicate vocabulary drift

---

### Question 4
- [ ] Done
**When would you use subword embeddings (FastText) versus word-level embeddings for morphologically rich languages?**

**Answer:**

**Definition:**
Use **subword embeddings (FastText)** for morphologically rich languages (German, Turkish, Finnish, Arabic) where words have many inflected forms. Subword captures shared morphemes, handles OOV from affixes, and reduces vocabulary size. Word-level embeddings fail when vocabulary explodes due to morphological variation.

**When to Use Subword:**

| Scenario | Subword (FastText) | Word-level |
|----------|-------------------|------------|
| **Agglutinative languages** | ✓ Best | ✗ Poor |
| **Compound words** | ✓ (German) | ✗ |
| **Rich inflection** | ✓ | ✗ |
| **Limited training data** | ✓ Better generalization | ✗ |
| **Typo tolerance** | ✓ Partial matching | ✗ |
| **Fixed vocabulary** | Either | ✓ Simpler |

**Morphological Complexity by Language:**

| Language | Type | Unique Forms | Subword Benefit |
|----------|------|--------------|-----------------|
| **Turkish** | Agglutinative | Very High | Critical |
| **Finnish** | Agglutinative | Very High | Critical |
| **German** | Fusional + Compounds | High | Strong |
| **Arabic** | Root-based | High | Strong |
| **Russian** | Fusional | Medium | Moderate |
| **English** | Analytic | Low | Mild |
| **Chinese** | Isolating | Low | Word-level OK |

**Python Code Example:**
```python
import numpy as np
from collections import Counter

class MorphologicalAnalyzer:
    """Analyze vocabulary for morphological complexity"""
    
    def __init__(self, corpus_words: list):
        self.words = corpus_words
        self.word_freq = Counter(corpus_words)
    
    def morphological_richness(self) -> dict:
        """Calculate metrics indicating morphological complexity"""
        
        unique_words = len(set(self.words))
        total_words = len(self.words)
        
        # Type-token ratio (higher = more unique forms)
        ttr = unique_words / total_words
        
        # Average word length (longer = likely agglutinative)
        avg_length = np.mean([len(w) for w in self.words])
        
        # Hapax legomena ratio (words appearing once)
        hapax = sum(1 for w, c in self.word_freq.items() if c == 1)
        hapax_ratio = hapax / unique_words
        
        return {
            'type_token_ratio': ttr,
            'avg_word_length': avg_length,
            'hapax_ratio': hapax_ratio,
            'unique_words': unique_words,
            'recommend_subword': ttr > 0.3 or avg_length > 8
        }
    
    def estimate_vocabulary_explosion(self, sample_sizes: list) -> dict:
        """Estimate how vocabulary grows with corpus size"""
        
        results = {}
        for size in sample_sizes:
            sample = self.words[:size]
            unique = len(set(sample))
            results[size] = {
                'unique_words': unique,
                'growth_rate': unique / size
            }
        
        # If vocabulary keeps growing linearly, need subword
        return results

# FastText for morphological languages
def train_fasttext_for_morphological(corpus_file: str, 
                                      language: str) -> dict:
    """Configure FastText for morphologically rich languages"""
    import fasttext
    
    # Language-specific configurations
    configs = {
        'turkish': {
            'minn': 2,       # Min n-gram
            'maxn': 6,       # Max n-gram (longer for agglutinative)
            'dim': 300,
            'ws': 5          # Context window
        },
        'german': {
            'minn': 3,
            'maxn': 6,
            'dim': 300,
            'ws': 5
        },
        'english': {
            'minn': 3,
            'maxn': 5,
            'dim': 300,
            'ws': 5
        }
    }
    
    config = configs.get(language, configs['english'])
    
    model = fasttext.train_unsupervised(
        corpus_file,
        model='skipgram',
        **config
    )
    
    return model

# Demonstrating subword advantage
def subword_advantage_demo():
    """Show how subword handles morphological variants"""
    
    # German compound words
    german_examples = [
        'Handschuh',           # glove (hand + shoe)
        'Handschuhe',          # gloves
        'Lederhandschuh',      # leather glove
        'Lederhandschuhe',     # leather gloves
    ]
    
    # Turkish agglutination
    turkish_examples = [
        'ev',                   # house
        'evler',                # houses
        'evlerimiz',            # our houses
        'evlerimizde',          # in our houses
        'evlerimizdekiler',     # those in our houses
    ]
    
    # With FastText, all variants share subword components
    # With word-level, each is a separate embedding (or OOV)
    
    return {
        'german_unique_forms': len(set(german_examples)),
        'turkish_unique_forms': len(set(turkish_examples)),
        'subword_coverage': 'All forms represented via shared n-grams',
        'word_level_coverage': 'Each form needs separate training data'
    }

# Comparing approaches
def compare_approaches(vocab_size: int, corpus_size: int) -> dict:
    """Compare storage and coverage"""
    
    word_level = {
        'storage': vocab_size * 300 * 4,  # bytes
        'oov_rate': max(0, (vocab_size - 100000) / vocab_size),
        'training_data_needed': vocab_size * 5  # rough estimate
    }
    
    subword = {
        'storage': 50000 * 300 * 4 + 2_000_000 * 4,  # base + n-gram hashes
        'oov_rate': 0.01,  # Near-zero
        'training_data_needed': corpus_size  # Works with existing data
    }
    
    return {'word_level': word_level, 'subword': subword}
```

**Interview Tips:**
- Always use FastText/subword for Turkish, Finnish, Hungarian, Arabic
- German benefits greatly from subword due to compound words
- English is borderline — subword helps with typos and rare words
- BPE (used in BERT) is another subword approach optimized for compression

---

### Question 5
- [ ] Done
**How do you implement domain adaptation when pre-trained embeddings don't match your specific use case?**

**Answer:**

**Definition:**
Domain adaptation adjusts pre-trained embeddings to a target domain via: **continued pre-training** on domain corpus, **fine-tuning** on domain task, **retrofitting** with domain knowledge graphs, or **projection** to align domain-specific vectors. Goal: maintain general knowledge while capturing domain-specific semantics.

**Adaptation Strategies:**

| Strategy | Approach | Best For |
|----------|----------|----------|
| **Continued pre-training** | Train on domain corpus (MLM) | Large domain data |
| **Fine-tuning** | Train on labeled domain task | Supervised domain data |
| **Retrofitting** | Inject domain synonyms/relations | Domain ontologies |
| **Linear projection** | Learn A·x + b mapping | Small domain data |
| **Adapter modules** | Add domain-specific layers | Multiple domains |

**When to Use Each:**

| Scenario | Recommended Approach |
|----------|---------------------|
| **Medical NLP** | BioBERT (continued pre-training) |
| **Legal documents** | Fine-tune on legal classification |
| **Company jargon** | Retrofit with internal glossary |
| **Limited labeled data** | Linear projection + few examples |
| **Multi-domain system** | Adapter modules per domain |

**Python Code Example:**
```python
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class EmbeddingDomainAdapter:
    """Adapt pre-trained embeddings to target domain"""
    
    def __init__(self, pretrained_embeddings: Dict[str, np.ndarray], dim: int = 300):
        self.pretrained = pretrained_embeddings
        self.dim = dim
        self.adapted = pretrained_embeddings.copy()
    
    def continued_pretraining(self, domain_corpus: List[str], epochs: int = 5):
        """Continue training on domain corpus"""
        from gensim.models import Word2Vec
        
        # Tokenize corpus
        tokenized = [text.lower().split() for text in domain_corpus]
        
        # Initialize with pre-trained vectors
        model = Word2Vec(vector_size=self.dim, min_count=1)
        model.build_vocab(tokenized)
        
        # Load pre-trained weights where available
        for word in model.wv.index_to_key:
            if word in self.pretrained:
                model.wv[word] = self.pretrained[word]
        
        # Continue training
        model.train(tokenized, total_examples=len(tokenized), epochs=epochs)
        
        # Update adapted embeddings
        self.adapted = {word: model.wv[word] for word in model.wv.index_to_key}
        return self.adapted
    
    def retrofit(self, synonym_pairs: List[Tuple[str, str]], 
                 iterations: int = 10, alpha: float = 1.0):
        """Retrofit embeddings using domain relationships"""
        
        # Build adjacency from synonyms
        adjacency = {}
        for w1, w2 in synonym_pairs:
            if w1 not in adjacency:
                adjacency[w1] = []
            if w2 not in adjacency:
                adjacency[w2] = []
            adjacency[w1].append(w2)
            adjacency[w2].append(w1)
        
        # Retrofitting iterations
        new_embeddings = {w: v.copy() for w, v in self.pretrained.items()}
        
        for _ in range(iterations):
            for word in adjacency:
                if word not in self.pretrained:
                    continue
                
                neighbors = [n for n in adjacency[word] if n in new_embeddings]
                if not neighbors:
                    continue
                
                # Update: balance original embedding and neighbor average
                original = self.pretrained[word]
                neighbor_avg = np.mean([new_embeddings[n] for n in neighbors], axis=0)
                
                new_embeddings[word] = (original + alpha * neighbor_avg) / (1 + alpha)
        
        self.adapted = new_embeddings
        return self.adapted
    
    def linear_projection(self, source_words: List[str], 
                         target_vectors: np.ndarray):
        """Learn linear mapping from source to target domain"""
        
        # Get source vectors
        source_vectors = np.array([self.pretrained[w] for w in source_words])
        
        # Learn projection: W * source ≈ target
        W, _, _, _ = np.linalg.lstsq(source_vectors, target_vectors, rcond=None)
        
        # Apply to all embeddings
        self.adapted = {
            word: vec @ W for word, vec in self.pretrained.items()
        }
        return self.adapted

# BERT domain adaptation
class BERTDomainAdapter:
    """Adapt BERT to domain via continued pretraining"""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        from transformers import (BertForMaskedLM, BertTokenizer,
                                   DataCollatorForLanguageModeling,
                                   TrainingArguments, Trainer)
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
    
    def continued_pretraining(self, domain_texts: List[str], 
                               output_dir: str, epochs: int = 3):
        """Continue MLM training on domain corpus"""
        from transformers import (DataCollatorForLanguageModeling,
                                   TrainingArguments, Trainer)
        from datasets import Dataset
        
        # Prepare dataset
        dataset = Dataset.from_dict({'text': domain_texts})
        
        def tokenize(examples):
            return self.tokenizer(examples['text'], 
                                 truncation=True, 
                                 max_length=512)
        
        tokenized = dataset.map(tokenize, batched=True)
        
        # MLM data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=True, 
            mlm_probability=0.15
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            save_steps=1000,
            learning_rate=2e-5
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator
        )
        
        trainer.train()
        return self.model

# Adapter-based domain adaptation
class DomainAdapter(nn.Module):
    """Lightweight adapter for domain-specific tuning"""
    
    def __init__(self, input_dim: int, adapter_dim: int = 64):
        super().__init__()
        self.down_project = nn.Linear(input_dim, adapter_dim)
        self.up_project = nn.Linear(adapter_dim, input_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Residual adapter
        adapter_out = self.up_project(self.activation(self.down_project(x)))
        return x + adapter_out
```

**Interview Tips:**
- Continued pre-training is most effective but requires compute
- Retrofitting works well with domain ontologies (medical, legal)
- Adapters are efficient for multi-domain systems
- Always evaluate on domain-specific benchmarks, not just general tasks

---

### Question 6
- [ ] Done
**When should you train custom embeddings versus using pre-trained models?**

**Answer:**

**Definition:**
Train custom embeddings when: **domain vocabulary differs significantly** (medical, legal, code), **data is proprietary** (internal documents), or **language is low-resource**. Use pre-trained when domain overlaps with training corpus, data is limited, or compute is constrained. Generally, pre-trained + fine-tuning outperforms training from scratch.

**Decision Matrix:**

| Factor | Train Custom | Use Pre-trained |
|--------|--------------|-----------------|
| **Domain match** | Poor overlap | Good overlap |
| **Data volume** | >10M tokens | <10M tokens |
| **Vocabulary** | Specialized | General |
| **Compute budget** | Available | Limited |
| **Update frequency** | Static domain | Evolving |
| **Quality bar** | Maximum | Good enough |

**Use Cases Comparison:**

| Scenario | Recommendation | Reason |
|----------|----------------|--------|
| **General chatbot** | Pre-trained | Standard vocabulary |
| **Medical NER** | Custom or BioBERT | Medical terms |
| **Code search** | Custom (CodeBERT) | Programming syntax |
| **Internal docs** | Custom | Proprietary jargon |
| **Low-resource lang** | Custom + transfer | Limited pre-trained |
| **Quick prototype** | Pre-trained | Fast iteration |

**Python Code Example:**
```python
import numpy as np
from typing import List, Dict

class EmbeddingDecisionHelper:
    """Help decide between custom vs pre-trained embeddings"""
    
    def __init__(self, pretrained_vocab: set):
        self.pretrained_vocab = pretrained_vocab
    
    def analyze_domain_corpus(self, domain_texts: List[str]) -> Dict:
        """Analyze domain corpus for custom embedding decision"""
        
        # Extract domain vocabulary
        domain_words = set()
        for text in domain_texts:
            domain_words.update(text.lower().split())
        
        # Calculate overlap
        overlap = domain_words.intersection(self.pretrained_vocab)
        coverage = len(overlap) / len(domain_words)
        
        # Unique domain terms
        domain_specific = domain_words - self.pretrained_vocab
        
        return {
            'domain_vocab_size': len(domain_words),
            'pretrained_coverage': coverage,
            'oov_words': len(domain_specific),
            'oov_ratio': len(domain_specific) / len(domain_words),
            'sample_oov': list(domain_specific)[:20],
            'recommendation': 'custom' if coverage < 0.7 else 'pretrained'
        }
    
    def estimate_training_requirements(self, corpus_size: int) -> Dict:
        """Estimate resources for custom training"""
        
        # Rule of thumb: need ~100x occurrences per word for good embedding
        estimated_vocab = int(corpus_size ** 0.5)  # Heaps' law approximation
        min_corpus_size = estimated_vocab * 100
        
        # Training time estimation (very rough)
        training_hours = corpus_size / 10_000_000  # ~1 hour per 10M tokens
        
        return {
            'estimated_vocab': estimated_vocab,
            'min_recommended_corpus': min_corpus_size,
            'sufficient_data': corpus_size >= min_corpus_size,
            'estimated_training_hours': training_hours,
            'recommended_epochs': 5 if corpus_size > 10_000_000 else 10
        }

# Training custom Word2Vec
def train_custom_word2vec(corpus: List[List[str]], 
                          dim: int = 300,
                          window: int = 5,
                          min_count: int = 5) -> dict:
    """Train custom Word2Vec embeddings"""
    from gensim.models import Word2Vec
    
    model = Word2Vec(
        sentences=corpus,
        vector_size=dim,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=10
    )
    
    return {word: model.wv[word] for word in model.wv.index_to_key}

# Custom BERT pre-training
def train_custom_bert(corpus_file: str, output_dir: str):
    """Train BERT from scratch (requires significant compute)"""
    from transformers import (
        BertConfig, BertForMaskedLM, BertTokenizer,
        LineByLineTextDataset, DataCollatorForLanguageModeling,
        TrainingArguments, Trainer
    )
    
    # Train custom tokenizer first
    from tokenizers import BertWordPieceTokenizer
    
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train(files=[corpus_file], vocab_size=30000)
    tokenizer.save_model(output_dir)
    
    # Initialize model
    config = BertConfig(
        vocab_size=30000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072
    )
    model = BertForMaskedLM(config)
    
    # This requires significant compute (days on multiple GPUs)
    # Usually better to start from pre-trained and continue training
    
    return model

# Hybrid approach: pre-trained + domain-specific
class HybridEmbedding:
    """Combine pre-trained with custom domain embeddings"""
    
    def __init__(self, pretrained: Dict, custom: Dict, alpha: float = 0.5):
        self.pretrained = pretrained
        self.custom = custom
        self.alpha = alpha
    
    def get_embedding(self, word: str) -> np.ndarray:
        """Get blended embedding"""
        word = word.lower()
        
        has_pretrained = word in self.pretrained
        has_custom = word in self.custom
        
        if has_pretrained and has_custom:
            # Blend both
            return (self.alpha * self.pretrained[word] + 
                    (1 - self.alpha) * self.custom[word])
        elif has_custom:
            return self.custom[word]
        elif has_pretrained:
            return self.pretrained[word]
        else:
            return np.zeros(300)
```

**Interview Tips:**
- Pre-trained + fine-tuning almost always beats training from scratch
- Need >10M tokens for reasonable custom embeddings
- Domain-specific pre-trained models (BioBERT, CodeBERT) are best of both worlds
- For internal jargon, consider retrofitting pre-trained with company vocabulary

---

### Question 7
- [ ] Done
**What techniques help optimize BERT embedding extraction for real-time inference requirements?**

**Answer:**

**Definition:**
Optimize BERT inference via: **model distillation** (DistilBERT: 60% faster), **quantization** (INT8: 2-4x speedup), **ONNX Runtime** (optimized inference), **batch processing**, **layer pruning** (use fewer layers), and **caching** (store frequent embeddings). Target: <50ms latency for production real-time systems.

**Optimization Techniques:**

| Technique | Speedup | Quality Loss | Effort |
|-----------|---------|--------------|--------|
| **DistilBERT** | 1.6x | ~3% | Low |
| **TinyBERT** | 3-4x | ~5% | Low |
| **INT8 quantization** | 2-4x | <1% | Medium |
| **ONNX Runtime** | 1.5-2x | 0% | Low |
| **Layer pruning (6L)** | 2x | ~2% | Low |
| **Caching** | 10-100x | 0% | Medium |
| **GPU batching** | 5-10x | 0% | Low |

**Optimization Pipeline:**
```
Original BERT (440MB, 100ms)
    → Distillation (DistilBERT: 265MB, 60ms)
    → Quantization (INT8: 100MB, 30ms)
    → ONNX conversion (optimized ops: 25ms)
    → Caching layer (cache hits: <1ms)
```

**Python Code Example:**
```python
import torch
import numpy as np
from typing import List, Dict
import time

class OptimizedBERTEmbedder:
    """Optimized BERT for real-time embedding extraction"""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased',
                 use_onnx: bool = True,
                 use_cache: bool = True,
                 cache_size: int = 10000):
        
        self.use_onnx = use_onnx
        self.use_cache = use_cache
        
        if use_onnx:
            self._setup_onnx(model_name)
        else:
            self._setup_pytorch(model_name)
        
        if use_cache:
            self.cache = {}
            self.cache_size = cache_size
    
    def _setup_pytorch(self, model_name: str):
        from transformers import AutoModel, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
    
    def _setup_onnx(self, model_name: str):
        """Setup ONNX Runtime for faster inference"""
        from transformers import AutoTokenizer
        import onnxruntime as ort
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Export or load ONNX model
        onnx_path = f"{model_name.replace('/', '_')}.onnx"
        
        try:
            self.session = ort.InferenceSession(onnx_path)
        except:
            # Export model to ONNX
            self._export_to_onnx(model_name, onnx_path)
            self.session = ort.InferenceSession(onnx_path)
    
    def _export_to_onnx(self, model_name: str, output_path: str):
        """Export PyTorch model to ONNX"""
        from transformers import AutoModel
        
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        dummy_input = self.tokenizer(
            "dummy text",
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True
        )
        
        torch.onnx.export(
            model,
            (dummy_input['input_ids'], dummy_input['attention_mask']),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'sequence'},
                'attention_mask': {0: 'batch', 1: 'sequence'}
            },
            opset_version=12
        )
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Get embeddings with optimization"""
        
        # Check cache
        if self.use_cache:
            cached, uncached_texts, uncached_indices = self._check_cache(texts)
            if not uncached_texts:
                return cached
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Compute embeddings
        if self.use_onnx:
            embeddings = self._embed_onnx(uncached_texts)
        else:
            embeddings = self._embed_pytorch(uncached_texts)
        
        # Update cache
        if self.use_cache:
            self._update_cache(uncached_texts, embeddings)
            return self._merge_cached(cached, embeddings, uncached_indices, len(texts))
        
        return embeddings
    
    def _embed_pytorch(self, texts: List[str]) -> np.ndarray:
        """PyTorch inference"""
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()
    
    def _embed_onnx(self, texts: List[str]) -> np.ndarray:
        """ONNX Runtime inference"""
        inputs = self.tokenizer(
            texts,
            return_tensors='np',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        outputs = self.session.run(
            None,
            {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            }
        )
        
        # Mean pooling
        return outputs[0].mean(axis=1)
    
    def _check_cache(self, texts):
        """Check which texts are cached"""
        cached = {}
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if text in self.cache:
                cached[i] = self.cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        return cached, uncached_texts, uncached_indices
    
    def _update_cache(self, texts, embeddings):
        """Update cache with LRU eviction"""
        for text, emb in zip(texts, embeddings):
            if len(self.cache) >= self.cache_size:
                # Evict oldest
                oldest = next(iter(self.cache))
                del self.cache[oldest]
            self.cache[text] = emb

# Quantization for CPU inference
def quantize_for_cpu(model_name: str):
    """Dynamic quantization for CPU deployment"""
    from transformers import AutoModel
    
    model = AutoModel.from_pretrained(model_name)
    
    # Dynamic quantization (weights only)
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    return quantized

# Benchmark utility
def benchmark_embedder(embedder, texts: List[str], num_runs: int = 100):
    """Benchmark embedding extraction"""
    # Warmup
    _ = embedder.embed(texts[:5])
    
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = embedder.embed(texts)
    elapsed = time.perf_counter() - start
    
    return {
        'total_time': elapsed,
        'per_batch_ms': (elapsed / num_runs) * 1000,
        'per_text_ms': (elapsed / (num_runs * len(texts))) * 1000,
        'throughput': (num_runs * len(texts)) / elapsed
    }
```

**Interview Tips:**
- DistilBERT + ONNX + caching can achieve <10ms latency
- Quantization works best on CPU; use FP16 for GPU
- Cache hit rates of 60-80% common for production search
- Consider async batching to improve GPU utilization

---

### Question 8
- [ ] Done
**How do you implement effective dimensionality reduction (PCA, UMAP) for high-dimensional embeddings without losing semantic information?**

**Answer:**

**Definition:**
Reduce embedding dimensions via **PCA** (linear, fast, preserves variance), **UMAP** (non-linear, preserves local structure), or **learned projections** (task-optimized). Balance: reduce from 768→256 dims with <5% quality loss for storage/speed gains. Critical: evaluate on downstream task, not just reconstruction error.

**Method Comparison:**

| Method | Type | Speed | Quality | Use Case |
|--------|------|-------|---------|----------|
| **PCA** | Linear | Fast | Good | General compression |
| **UMAP** | Non-linear | Slow | Excellent | Visualization |
| **t-SNE** | Non-linear | Very slow | Good | Visualization only |
| **Autoencoder** | Learned | Medium | Best | Task-specific |
| **Random projection** | Linear | Very fast | Moderate | Approximate |

**Dimension Reduction Guidelines:**

| Original Dim | Target Dim | Expected Quality Loss |
|--------------|------------|----------------------|
| 768 (BERT) | 512 | <1% |
| 768 (BERT) | 256 | 2-3% |
| 768 (BERT) | 128 | 5-8% |
| 768 (BERT) | 64 | 10-15% |
| 300 (Word2Vec) | 100 | 3-5% |

**Python Code Example:**
```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from typing import Tuple

class EmbeddingReducer:
    """Dimensionality reduction for embeddings"""
    
    def __init__(self, target_dim: int, method: str = 'pca'):
        self.target_dim = target_dim
        self.method = method
        self.reducer = None
        self.scaler = None
    
    def fit(self, embeddings: np.ndarray):
        """Fit reducer on embeddings"""
        
        # Standardize for better reduction
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(embeddings)
        
        if self.method == 'pca':
            self.reducer = PCA(n_components=self.target_dim)
            self.reducer.fit(scaled)
            
            # Report variance explained
            var_explained = sum(self.reducer.explained_variance_ratio_)
            print(f"PCA: {var_explained:.2%} variance explained with {self.target_dim} dims")
        
        elif self.method == 'umap':
            self.reducer = umap.UMAP(
                n_components=self.target_dim,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine'
            )
            self.reducer.fit(scaled)
        
        return self
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to lower dimension"""
        scaled = self.scaler.transform(embeddings)
        return self.reducer.transform(scaled)
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        self.fit(embeddings)
        return self.transform(embeddings)

class WhiteningPCA:
    """PCA with whitening for better cosine similarity preservation"""
    
    def __init__(self, target_dim: int):
        self.target_dim = target_dim
        self.mean = None
        self.components = None
        self.singular_values = None
    
    def fit(self, embeddings: np.ndarray):
        """Fit whitening PCA"""
        self.mean = embeddings.mean(axis=0)
        centered = embeddings - self.mean
        
        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        self.components = Vt[:self.target_dim]
        self.singular_values = S[:self.target_dim]
        
        return self
    
    def transform(self, embeddings: np.ndarray, whiten: bool = True) -> np.ndarray:
        """Transform with optional whitening"""
        centered = embeddings - self.mean
        projected = centered @ self.components.T
        
        if whiten:
            # Scale by inverse singular values
            projected = projected / (self.singular_values + 1e-8)
        
        return projected

# Learned dimensionality reduction (autoencoder)
import torch
import torch.nn as nn

class EmbeddingAutoencoder(nn.Module):
    """Autoencoder for learned dimensionality reduction"""
    
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

def train_autoencoder(embeddings: np.ndarray, latent_dim: int,
                      epochs: int = 100) -> EmbeddingAutoencoder:
    """Train autoencoder for dimensionality reduction"""
    
    input_dim = embeddings.shape[1]
    model = EmbeddingAutoencoder(input_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Convert to tensor
    X = torch.FloatTensor(embeddings)
    dataset = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            x = batch[0]
            
            optimizer.zero_grad()
            x_recon, z = model(x)
            
            # Reconstruction loss + regularization
            loss = nn.MSELoss()(x_recon, x)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    model.eval()
    return model

# Evaluation of reduction quality
def evaluate_reduction(original: np.ndarray, reduced: np.ndarray,
                       pairs: list) -> dict:
    """Evaluate quality of dimensionality reduction"""
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Compute similarities before and after
    original_sims = []
    reduced_sims = []
    
    for i, j in pairs:
        orig_sim = cosine_similarity([original[i]], [original[j]])[0][0]
        red_sim = cosine_similarity([reduced[i]], [reduced[j]])[0][0]
        original_sims.append(orig_sim)
        reduced_sims.append(red_sim)
    
    correlation = np.corrcoef(original_sims, reduced_sims)[0, 1]
    
    return {
        'correlation': correlation,
        'mean_absolute_error': np.mean(np.abs(np.array(original_sims) - np.array(reduced_sims))),
        'quality_preserved': correlation > 0.95
    }
```

**Interview Tips:**
- PCA is usually sufficient for production; UMAP for visualization
- Always evaluate on downstream task (search accuracy), not reconstruction
- Whitening helps preserve cosine similarity after reduction
- 256 dimensions is often a sweet spot for BERT embeddings

---

## Embedding Operations & Similarity

### Question 9
- [ ] Done
**Explain the different similarity metrics (cosine, euclidean, dot product) and when to use each.**

**Answer:**

**Definition:**
**Cosine similarity** measures angle between vectors (normalized, ignores magnitude). **Euclidean distance** measures geometric distance (sensitive to magnitude). **Dot product** combines angle and magnitude. Choice depends on: whether magnitude is meaningful, whether embeddings are normalized, and computational constraints.

**Metric Comparison:**

| Metric | Formula | Range | Normalized? | Use Case |
|--------|---------|-------|-------------|----------|
| **Cosine** | $\frac{a \cdot b}{\|a\| \|b\|}$ | [-1, 1] | Yes | Text similarity |
| **Euclidean** | $\sqrt{\sum(a_i - b_i)^2}$ | [0, ∞) | No | Clustering |
| **Dot Product** | $\sum a_i \cdot b_i$ | (-∞, ∞) | No | Learned embeddings |
| **Manhattan** | $\sum|a_i - b_i|$ | [0, ∞) | No | High-dim sparse |

**When to Use Each:**

| Scenario | Best Metric | Reason |
|----------|-------------|--------|
| **Semantic search** | Cosine | Direction matters, not magnitude |
| **Sentence-BERT** | Cosine | Training objective uses cosine |
| **Recommendation** | Dot product | Magnitude encodes preference strength |
| **Two-tower models** | Dot product | Efficiently computed |
| **K-means clustering** | Euclidean | Cluster centroids meaningful |
| **Normalized embeddings** | Any equivalent | Cosine = dot product = Euclidean ordering |

**Mathematical Relationships:**
- If vectors are L2-normalized: $\cos(a,b) = a \cdot b$
- $\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2(a \cdot b)$
- For normalized vectors: $\|a-b\|^2 = 2(1 - \cos(a,b))$

**Python Code Example:**
```python
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityMetrics:
    """Comprehensive similarity metric implementations"""
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity: angle between vectors"""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot / (norm_a * norm_b + 1e-8)
    
    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance: geometric distance"""
        return np.sqrt(np.sum((a - b) ** 2))
    
    @staticmethod
    def dot_product(a: np.ndarray, b: np.ndarray) -> float:
        """Dot product: combines direction and magnitude"""
        return np.dot(a, b)
    
    @staticmethod
    def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Manhattan (L1) distance"""
        return np.sum(np.abs(a - b))
    
    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """L2 normalize vector"""
        norm = np.linalg.norm(v)
        return v / (norm + 1e-8)

# Metric selection helper
def select_metric(embedding_type: str, task: str, normalized: bool) -> str:
    """Recommend similarity metric"""
    
    recommendations = {
        ('bert', 'search', True): 'dot_product',  # Equiv to cosine when normalized
        ('bert', 'search', False): 'cosine',
        ('sentence-bert', 'search', True): 'cosine',
        ('word2vec', 'similarity', False): 'cosine',
        ('recommendation', 'ranking', False): 'dot_product',
        ('clustering', 'kmeans', False): 'euclidean'
    }
    
    key = (embedding_type, task, normalized)
    return recommendations.get(key, 'cosine')

# Batch similarity computation
def batch_similarity(queries: np.ndarray, candidates: np.ndarray, 
                     metric: str = 'cosine') -> np.ndarray:
    """Efficient batch similarity computation"""
    
    if metric == 'cosine':
        # Normalize and dot product
        queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        candidates_norm = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-8)
        return queries_norm @ candidates_norm.T
    
    elif metric == 'dot_product':
        return queries @ candidates.T
    
    elif metric == 'euclidean':
        # Efficient L2 distance using (a-b)^2 = a^2 + b^2 - 2ab
        q_sq = np.sum(queries ** 2, axis=1, keepdims=True)
        c_sq = np.sum(candidates ** 2, axis=1, keepdims=True).T
        cross = 2 * queries @ candidates.T
        distances = np.sqrt(np.maximum(q_sq + c_sq - cross, 0))
        return -distances  # Negate for similarity ordering

# Demonstration of metric equivalence
def demonstrate_equivalence():
    """Show when metrics are equivalent"""
    
    # Random vectors
    a = np.random.randn(768)
    b = np.random.randn(768)
    
    # Original metrics
    cos_orig = SimilarityMetrics.cosine_similarity(a, b)
    dot_orig = SimilarityMetrics.dot_product(a, b)
    euc_orig = SimilarityMetrics.euclidean_distance(a, b)
    
    # Normalize
    a_norm = SimilarityMetrics.normalize(a)
    b_norm = SimilarityMetrics.normalize(b)
    
    # Normalized metrics
    cos_norm = SimilarityMetrics.cosine_similarity(a_norm, b_norm)
    dot_norm = SimilarityMetrics.dot_product(a_norm, b_norm)
    euc_norm = SimilarityMetrics.euclidean_distance(a_norm, b_norm)
    
    print(f"Original - Cosine: {cos_orig:.4f}, Dot: {dot_orig:.4f}, Euclidean: {euc_orig:.4f}")
    print(f"Normalized - Cosine: {cos_norm:.4f}, Dot: {dot_norm:.4f}, Euclidean: {euc_norm:.4f}")
    print(f"Relationship: Cosine = Dot when normalized: {np.isclose(cos_norm, dot_norm)}")
    print(f"Euclidean = sqrt(2(1-cos)): {np.isclose(euc_norm, np.sqrt(2*(1-cos_norm)))}")
```

**Interview Tips:**
- Cosine is the default for text embeddings (magnitude is often noise)
- Always normalize embeddings before storing for consistent similarity
- Dot product is faster (no normalization) when embeddings are pre-normalized
- Inner product (dot) is preferred in FAISS for efficiency

---

### Question 10
- [ ] Done
**When should you use averaged word embeddings versus more sophisticated aggregation (attention pooling) for document representation?**

**Answer:**

**Definition:**
**Average pooling** sums word embeddings and divides by count — simple, fast, but loses word importance. **Attention pooling** learns to weight important words higher. Use averaging for quick baselines and efficiency; use attention pooling when accuracy is critical and importance varies across words.

**Aggregation Methods:**

| Method | Complexity | Quality | Use Case |
|--------|------------|---------|----------|
| **Mean pooling** | O(n) | Baseline | Fast similarity |
| **Max pooling** | O(n) | Similar | Feature detection |
| **TF-IDF weighted** | O(n) | Better | Important words |
| **Attention pooling** | O(n²) | Best | Quality-critical |
| **CLS token** | O(1) | Good | BERT classification |

**When to Use Each:**

| Scenario | Recommended | Why |
|----------|-------------|-----|
| **Real-time search** | Mean pooling | Speed priority |
| **Document classification** | Attention pooling | Accuracy matters |
| **Semantic similarity** | Mean or CLS | Sentence-BERT uses mean |
| **Short texts (<20 words)** | Mean pooling | All words matter |
| **Long documents** | Attention pooling | Need focus |
| **Topic detection** | TF-IDF weighted | Rare words important |

**Python Code Example:**
```python
import numpy as np
import torch
import torch.nn as nn
from typing import List

class DocumentEmbedder:
    """Multiple aggregation strategies for document embeddings"""
    
    def __init__(self, word_embeddings: dict, dim: int = 300):
        self.embeddings = word_embeddings
        self.dim = dim
    
    def mean_pooling(self, words: List[str]) -> np.ndarray:
        """Simple mean of word embeddings"""
        vectors = [self.embeddings[w] for w in words if w in self.embeddings]
        if not vectors:
            return np.zeros(self.dim)
        return np.mean(vectors, axis=0)
    
    def max_pooling(self, words: List[str]) -> np.ndarray:
        """Element-wise max pooling"""
        vectors = [self.embeddings[w] for w in words if w in self.embeddings]
        if not vectors:
            return np.zeros(self.dim)
        return np.max(vectors, axis=0)
    
    def tfidf_weighted(self, words: List[str], tfidf_weights: dict) -> np.ndarray:
        """TF-IDF weighted average"""
        weighted_sum = np.zeros(self.dim)
        weight_total = 0.0
        
        for word in words:
            if word in self.embeddings:
                weight = tfidf_weights.get(word, 1.0)
                weighted_sum += weight * self.embeddings[word]
                weight_total += weight
        
        if weight_total == 0:
            return np.zeros(self.dim)
        return weighted_sum / weight_total
    
    def sif_embedding(self, words: List[str], word_freq: dict, 
                      a: float = 1e-3) -> np.ndarray:
        """Smooth Inverse Frequency (SIF) embedding"""
        # SIF: weight = a / (a + p(w)) where p(w) is word probability
        total_words = sum(word_freq.values())
        
        weighted_sum = np.zeros(self.dim)
        weight_total = 0.0
        
        for word in words:
            if word in self.embeddings:
                p_w = word_freq.get(word, 1) / total_words
                weight = a / (a + p_w)
                weighted_sum += weight * self.embeddings[word]
                weight_total += weight
        
        if weight_total == 0:
            return np.zeros(self.dim)
        return weighted_sum / weight_total

# Attention pooling with learnable weights
class AttentionPooling(nn.Module):
    """Learnable attention-based pooling"""
    
    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            embeddings: (batch, seq_len, embed_dim)
            mask: (batch, seq_len) - 1 for valid, 0 for padding
        Returns:
            pooled: (batch, embed_dim)
        """
        # Compute attention scores
        scores = self.attention(embeddings).squeeze(-1)  # (batch, seq_len)
        
        # Mask padding tokens
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax attention weights
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # Weighted sum
        pooled = torch.bmm(weights.unsqueeze(1), embeddings).squeeze(1)
        
        return pooled, weights

# Self-attention pooling (more sophisticated)
class SelfAttentionPooling(nn.Module):
    """Self-attention based document representation"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Query vector for pooling
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            embeddings: (batch, seq_len, embed_dim)
        Returns:
            pooled: (batch, embed_dim)
        """
        batch_size = embeddings.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        
        # Attend to all word embeddings
        pooled, weights = self.attention(
            query, embeddings, embeddings,
            key_padding_mask=(mask == 0) if mask is not None else None
        )
        
        return pooled.squeeze(1), weights

# BERT-style pooling options
class BERTPooler:
    """Different pooling strategies for BERT outputs"""
    
    @staticmethod
    def cls_pooling(outputs, attention_mask):
        """Use [CLS] token representation"""
        return outputs.last_hidden_state[:, 0, :]
    
    @staticmethod
    def mean_pooling(outputs, attention_mask):
        """Mean of all token embeddings (Sentence-BERT style)"""
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    @staticmethod
    def max_pooling(outputs, attention_mask):
        """Max pooling over sequence"""
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, dim=1)[0]

# Comparison benchmark
def compare_pooling_methods(texts: List[str], embeddings: dict, labels: List[int]):
    """Compare pooling methods on classification task"""
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    embedder = DocumentEmbedder(embeddings)
    
    results = {}
    for method_name, method in [
        ('mean', embedder.mean_pooling),
        ('max', embedder.max_pooling)
    ]:
        X = np.array([method(text.split()) for text in texts])
        scores = cross_val_score(LogisticRegression(), X, labels, cv=5)
        results[method_name] = scores.mean()
    
    return results
```

**Interview Tips:**
- Sentence-BERT uses mean pooling — proven effective for similarity
- CLS token works well for classification but not similarity
- SIF (Smooth Inverse Frequency) is a strong baseline for sentence embeddings
- Attention pooling requires training data but gives best results

---

### Question 11
- [ ] Done
**How do you handle the context length limitations when extracting BERT embeddings from long documents?**

**Answer:**

**Definition:**
Handle BERT's 512-token limit via: **chunking** (split into overlapping windows), **hierarchical encoding** (encode chunks → aggregate), **Longformer/BigBird** (sparse attention for 4K+ tokens), or **sentence-level pooling** (embed sentences → aggregate). Choose based on document structure and downstream task requirements.

**Strategies for Long Documents:**

| Strategy | Max Length | Quality | Complexity |
|----------|------------|---------|------------|
| **Truncation** | 512 | Poor | Low |
| **Chunking + Mean** | Unlimited | Moderate | Low |
| **Chunking + Attention** | Unlimited | Good | Medium |
| **Hierarchical** | Unlimited | Good | Medium |
| **Longformer** | 4096+ | Best | Low |
| **Sentence pooling** | Unlimited | Good | Medium |

**Chunking Approaches:**

| Type | Description | Use Case |
|------|-------------|----------|
| **Non-overlapping** | Split every 512 tokens | Speed priority |
| **Overlapping** | 50% overlap | Context preservation |
| **Sentence-aware** | Split at sentence boundaries | Coherent chunks |
| **Paragraph-aware** | Split at paragraphs | Document structure |

**Python Code Example:**
```python
import torch
import numpy as np
from typing import List, Tuple
from transformers import AutoModel, AutoTokenizer

class LongDocumentEmbedder:
    """Handle long documents exceeding BERT's token limit"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', 
                 max_length: int = 512,
                 stride: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.stride = stride
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.max_length - 2, len(tokens))  # -2 for [CLS], [SEP]
            chunk_tokens = [self.tokenizer.cls_token_id] + tokens[start:end] + [self.tokenizer.sep_token_id]
            chunks.append(chunk_tokens)
            
            if end >= len(tokens):
                break
            start += self.stride
        
        return chunks
    
    def embed_chunks(self, chunk_tokens: List[List[int]]) -> List[np.ndarray]:
        """Embed each chunk"""
        embeddings = []
        
        for tokens in chunk_tokens:
            # Pad to max_length
            attention_mask = [1] * len(tokens)
            padding_length = self.max_length - len(tokens)
            tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            
            inputs = {
                'input_ids': torch.tensor([tokens]).to(self.device),
                'attention_mask': torch.tensor([attention_mask]).to(self.device)
            }
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean pooling over non-padding tokens
            hidden = outputs.last_hidden_state[0]
            mask = inputs['attention_mask'][0].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(0) / mask.sum()
            
            embeddings.append(pooled.cpu().numpy())
        
        return embeddings
    
    def embed_document(self, text: str, aggregation: str = 'mean') -> np.ndarray:
        """Get single embedding for long document"""
        chunks = self.chunk_text(text)
        chunk_embeddings = self.embed_chunks(chunks)
        
        if aggregation == 'mean':
            return np.mean(chunk_embeddings, axis=0)
        elif aggregation == 'max':
            return np.max(chunk_embeddings, axis=0)
        elif aggregation == 'first':
            return chunk_embeddings[0]
        elif aggregation == 'weighted':
            # Weight by position (earlier chunks often more important)
            weights = np.exp(-np.arange(len(chunk_embeddings)) * 0.1)
            weights /= weights.sum()
            return np.average(chunk_embeddings, axis=0, weights=weights)

# Hierarchical encoding
class HierarchicalEncoder:
    """Two-level encoding: sentences → document"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def split_sentences(self, text: str) -> List[str]:
        """Split document into sentences"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def encode_hierarchical(self, text: str, 
                           aggregation: str = 'mean') -> np.ndarray:
        """Encode via sentence-level pooling"""
        sentences = self.split_sentences(text)
        
        if not sentences:
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        # Encode all sentences
        sentence_embeddings = self.model.encode(sentences)
        
        if aggregation == 'mean':
            return np.mean(sentence_embeddings, axis=0)
        elif aggregation == 'max':
            return np.max(sentence_embeddings, axis=0)
        elif aggregation == 'weighted_position':
            # Later sentences often contain conclusions
            weights = np.linspace(0.5, 1.5, len(sentences))
            weights /= weights.sum()
            return np.average(sentence_embeddings, axis=0, weights=weights)

# Longformer for truly long documents
class LongformerEmbedder:
    """Use Longformer for documents up to 4096 tokens"""
    
    def __init__(self, model_name: str = 'allenai/longformer-base-4096'):
        from transformers import LongformerModel, LongformerTokenizer
        
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.model = LongformerModel.from_pretrained(model_name)
        self.model.eval()
    
    def embed(self, text: str) -> np.ndarray:
        """Embed long document directly"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=4096,
            padding=True
        )
        
        # Set global attention on [CLS] token
        global_attention_mask = torch.zeros_like(inputs['input_ids'])
        global_attention_mask[:, 0] = 1
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                global_attention_mask=global_attention_mask
            )
        
        # Use [CLS] token embedding
        return outputs.last_hidden_state[:, 0, :].numpy()[0]
```

**Interview Tips:**
- Overlapping chunks preserve context better but increase compute
- Sentence-level hierarchical is often best for diverse documents
- Longformer/BigBird are preferred when available and document < 4K tokens
- Weight chunks by importance (first/last often most relevant)

---

### Question 12
- [ ] Done
**What strategies help reduce bias in word embeddings for fair and inclusive NLP applications?**

**Answer:**

**Definition:**
Reduce embedding bias via: **debiasing projections** (remove gender/race subspace), **counterfactual data augmentation** (balance training data), **adversarial training** (learn bias-invariant representations), or **post-hoc calibration**. Goal: maintain semantic utility while reducing stereotypical associations.

**Bias Types in Embeddings:**

| Bias Type | Example | Impact |
|-----------|---------|--------|
| **Gender** | "programmer" → male | Hiring bias |
| **Racial** | Name associations | Discrimination |
| **Religious** | Stereotype association | Unfair treatment |
| **Age** | Negative stereotypes | Employment bias |

**Debiasing Approaches:**

| Method | Approach | When to Use |
|--------|----------|-------------|
| **Hard debiasing** | Project out bias direction | Post-training fix |
| **Soft debiasing** | Partial projection | Balance utility/fairness |
| **Counterfactual** | Augment training data | Pre-training |
| **Adversarial** | Train to hide protected attr | Training time |
| **Re-training** | Curated balanced corpus | Maximum control |

**Python Code Example:**
```python
import numpy as np
from typing import List, Tuple, Dict
from sklearn.decomposition import PCA

class EmbeddingDebiaser:
    """Methods to reduce bias in word embeddings"""
    
    def __init__(self, embeddings: Dict[str, np.ndarray]):
        self.embeddings = embeddings
        self.bias_direction = None
    
    def identify_bias_direction(self, 
                                 word_pairs: List[Tuple[str, str]]) -> np.ndarray:
        """
        Identify bias direction from definitional pairs
        E.g., [("man", "woman"), ("he", "she"), ("king", "queen")]
        """
        differences = []
        
        for w1, w2 in word_pairs:
            if w1 in self.embeddings and w2 in self.embeddings:
                diff = self.embeddings[w1] - self.embeddings[w2]
                differences.append(diff)
        
        if not differences:
            raise ValueError("No valid word pairs found")
        
        # PCA to find principal direction
        pca = PCA(n_components=1)
        pca.fit(differences)
        self.bias_direction = pca.components_[0]
        
        return self.bias_direction
    
    def hard_debias(self, words_to_debias: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Hard debiasing: project out bias direction completely
        (Bolukbasi et al., 2016)
        """
        if self.bias_direction is None:
            raise ValueError("Call identify_bias_direction first")
        
        debiased = {}
        words = words_to_debias or self.embeddings.keys()
        
        for word in words:
            if word not in self.embeddings:
                continue
            
            vec = self.embeddings[word]
            # Project out bias direction
            projection = np.dot(vec, self.bias_direction) * self.bias_direction
            debiased[word] = vec - projection
        
        return debiased
    
    def soft_debias(self, words_to_debias: List[str], 
                    lambda_param: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Soft debiasing: partial projection (retain some structure)
        """
        if self.bias_direction is None:
            raise ValueError("Call identify_bias_direction first")
        
        debiased = {}
        
        for word in words_to_debias:
            if word not in self.embeddings:
                continue
            
            vec = self.embeddings[word]
            projection = np.dot(vec, self.bias_direction) * self.bias_direction
            # Partial removal
            debiased[word] = vec - lambda_param * projection
        
        return debiased
    
    def measure_bias(self, word: str, 
                     attribute_words_a: List[str],
                     attribute_words_b: List[str]) -> float:
        """
        Measure bias of a word toward two attribute sets
        WEAT-style metric
        """
        if word not in self.embeddings:
            return 0.0
        
        word_vec = self.embeddings[word]
        
        # Average similarity to each attribute set
        sim_a = np.mean([
            self._cosine(word_vec, self.embeddings.get(w, np.zeros_like(word_vec)))
            for w in attribute_words_a if w in self.embeddings
        ])
        
        sim_b = np.mean([
            self._cosine(word_vec, self.embeddings.get(w, np.zeros_like(word_vec)))
            for w in attribute_words_b if w in self.embeddings
        ])
        
        return sim_a - sim_b  # Positive = biased toward A
    
    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# Counterfactual Data Augmentation
class CounterfactualAugmenter:
    """Generate counterfactual training data to reduce bias"""
    
    def __init__(self):
        self.swap_pairs = {
            'he': 'she', 'she': 'he',
            'him': 'her', 'her': 'him',
            'his': 'her', 'man': 'woman',
            'woman': 'man', 'male': 'female',
            'female': 'male', 'boy': 'girl',
            'girl': 'boy'
        }
    
    def augment(self, text: str) -> str:
        """Create counterfactual version of text"""
        words = text.split()
        augmented = []
        
        for word in words:
            lower = word.lower()
            if lower in self.swap_pairs:
                # Preserve capitalization
                swap = self.swap_pairs[lower]
                if word[0].isupper():
                    swap = swap.capitalize()
                augmented.append(swap)
            else:
                augmented.append(word)
        
        return ' '.join(augmented)
    
    def create_balanced_dataset(self, texts: List[str]) -> List[str]:
        """Create balanced dataset with counterfactuals"""
        balanced = []
        for text in texts:
            balanced.append(text)
            counterfactual = self.augment(text)
            if counterfactual != text:
                balanced.append(counterfactual)
        return balanced

# WEAT (Word Embedding Association Test)
def weat_score(embeddings: Dict[str, np.ndarray],
               target_x: List[str], target_y: List[str],
               attribute_a: List[str], attribute_b: List[str]) -> float:
    """
    WEAT: Measures differential association
    Caliskan et al., 2017
    """
    def mean_sim(word, attributes):
        if word not in embeddings:
            return 0
        sims = []
        for attr in attributes:
            if attr in embeddings:
                sim = np.dot(embeddings[word], embeddings[attr]) / (
                    np.linalg.norm(embeddings[word]) * np.linalg.norm(embeddings[attr]) + 1e-8
                )
                sims.append(sim)
        return np.mean(sims) if sims else 0
    
    def association(word):
        return mean_sim(word, attribute_a) - mean_sim(word, attribute_b)
    
    # Differential association
    x_assoc = [association(w) for w in target_x if w in embeddings]
    y_assoc = [association(w) for w in target_y if w in embeddings]
    
    effect_size = (np.mean(x_assoc) - np.mean(y_assoc)) / np.std(x_assoc + y_assoc)
    
    return effect_size
```

**Interview Tips:**
- Hard debiasing can hurt some downstream tasks — always evaluate
- Bias is often relearned during fine-tuning — debias at multiple stages
- WEAT is standard for measuring bias; include in evaluation pipeline
- Counterfactual augmentation is often more effective than post-hoc debiasing

---

### Question 13
- [ ] Done
**How do you implement effective cross-lingual embedding alignment techniques?**

**Answer:**

**Definition:**
Cross-lingual alignment maps embeddings from different languages into a **shared vector space** where similar concepts are close. Methods: **supervised** (parallel dictionary), **unsupervised** (adversarial + refinement), or **joint training** (multilingual models like mBERT). Enables zero-shot cross-lingual transfer.

**Alignment Approaches:**

| Method | Supervision | Quality | Use Case |
|--------|-------------|---------|----------|
| **Procrustes** | Dictionary | Good | Seed dictionary available |
| **Adversarial** | None | Moderate | No parallel data |
| **MUSE** | Optional | Good | General purpose |
| **mBERT/XLM-R** | None (joint) | Best | Full solution |
| **VecMap** | Dictionary | Good | Easy to use |

**Procrustes Alignment:**
Given source embeddings $X$ and target embeddings $Z$ with dictionary pairs:
$$W^* = \arg\min_W \|WX - Z\|_F^2 \text{ s.t. } W^TW = I$$

Solution: $W = VU^T$ from SVD of $Z^TX = U\Sigma V^T$

**Python Code Example:**
```python
import numpy as np
from typing import Dict, List, Tuple
from scipy.linalg import orthogonal_procrustes

class CrossLingualAligner:
    """Align embeddings across languages"""
    
    def __init__(self, source_embeddings: Dict[str, np.ndarray],
                 target_embeddings: Dict[str, np.ndarray]):
        self.source = source_embeddings
        self.target = target_embeddings
        self.W = None  # Alignment matrix
    
    def procrustes_alignment(self, 
                             dictionary: List[Tuple[str, str]]) -> np.ndarray:
        """
        Supervised alignment using bilingual dictionary
        (Conneau et al., 2017)
        """
        # Build matrices from dictionary pairs
        X = []  # Source vectors
        Z = []  # Target vectors
        
        for src_word, tgt_word in dictionary:
            if src_word in self.source and tgt_word in self.target:
                X.append(self.source[src_word])
                Z.append(self.target[tgt_word])
        
        X = np.array(X)
        Z = np.array(Z)
        
        # Normalize
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        
        # Orthogonal Procrustes
        self.W, _ = orthogonal_procrustes(X, Z)
        
        return self.W
    
    def align_embedding(self, source_vec: np.ndarray) -> np.ndarray:
        """Transform source embedding to target space"""
        if self.W is None:
            raise ValueError("Call procrustes_alignment first")
        return source_vec @ self.W
    
    def translate_word(self, source_word: str, k: int = 5) -> List[str]:
        """Find nearest neighbors in target language"""
        if source_word not in self.source:
            return []
        
        # Align source word
        aligned = self.align_embedding(self.source[source_word])
        
        # Find nearest in target
        similarities = {}
        for word, vec in self.target.items():
            sim = np.dot(aligned, vec) / (
                np.linalg.norm(aligned) * np.linalg.norm(vec) + 1e-8
            )
            similarities[word] = sim
        
        # Top-k
        return sorted(similarities.keys(), 
                     key=lambda x: similarities[x], 
                     reverse=True)[:k]

# Unsupervised alignment (adversarial)
import torch
import torch.nn as nn

class AdversarialAligner:
    """Unsupervised cross-lingual alignment via adversarial training"""
    
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        self.W = nn.Linear(embed_dim, embed_dim, bias=False)
        self.discriminator = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def train(self, source_embs: np.ndarray, target_embs: np.ndarray,
              epochs: int = 5, batch_size: int = 32):
        """
        Train alignment via adversarial game:
        - W tries to make aligned source look like target
        - Discriminator tries to distinguish source from target
        """
        X = torch.FloatTensor(source_embs)
        Z = torch.FloatTensor(target_embs)
        
        opt_W = torch.optim.Adam(self.W.parameters(), lr=1e-4)
        opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        
        for epoch in range(epochs):
            # Train discriminator
            for _ in range(5):
                idx_x = np.random.choice(len(X), batch_size)
                idx_z = np.random.choice(len(Z), batch_size)
                
                aligned = self.W(X[idx_x])
                target = Z[idx_z]
                
                # D should predict 0 for aligned source, 1 for target
                pred_aligned = self.discriminator(aligned.detach())
                pred_target = self.discriminator(target)
                
                loss_D = -torch.log(1 - pred_aligned + 1e-8).mean() - \
                         torch.log(pred_target + 1e-8).mean()
                
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()
            
            # Train W to fool discriminator
            idx_x = np.random.choice(len(X), batch_size)
            aligned = self.W(X[idx_x])
            pred = self.discriminator(aligned)
            
            loss_W = -torch.log(pred + 1e-8).mean()  # Fool D
            
            opt_W.zero_grad()
            loss_W.backward()
            opt_W.step()
            
            # Orthogonalize W (Procrustes refinement)
            with torch.no_grad():
                U, _, V = torch.svd(self.W.weight.data)
                self.W.weight.data = U @ V.T
        
        return self.W.weight.data.numpy()

# CSLS (Cross-domain Similarity Local Scaling) for better retrieval
def csls_similarity(source_vec: np.ndarray, 
                    target_embeddings: Dict[str, np.ndarray],
                    k: int = 10) -> Dict[str, float]:
    """
    CSLS: Reduces hubness problem in cross-lingual retrieval
    """
    # Compute all similarities
    sims = {}
    for word, vec in target_embeddings.items():
        sims[word] = np.dot(source_vec, vec) / (
            np.linalg.norm(source_vec) * np.linalg.norm(vec) + 1e-8
        )
    
    # Mean similarity of target words to their k nearest source neighbors
    # (In practice, precompute this for all target words)
    
    # CSLS score: 2*cos(s,t) - r_T(t) - r_S(s)
    # Simplified version here
    return sims

# Evaluation
def evaluate_alignment(aligned_source: Dict[str, np.ndarray],
                       target: Dict[str, np.ndarray],
                       test_dict: List[Tuple[str, str]]) -> Dict:
    """Evaluate word translation accuracy"""
    
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    for src, tgt in test_dict:
        if src not in aligned_source or tgt not in target:
            continue
        
        total += 1
        src_aligned = aligned_source[src]
        
        # Find nearest neighbors
        sims = [(w, np.dot(src_aligned, v) / (np.linalg.norm(src_aligned) * np.linalg.norm(v) + 1e-8))
                for w, v in target.items()]
        sims.sort(key=lambda x: x[1], reverse=True)
        
        top_5 = [w for w, _ in sims[:5]]
        
        if top_5[0] == tgt:
            correct_1 += 1
        if tgt in top_5:
            correct_5 += 1
    
    return {
        'P@1': correct_1 / total if total > 0 else 0,
        'P@5': correct_5 / total if total > 0 else 0,
        'total_evaluated': total
    }
```

**Interview Tips:**
- Procrustes is simple and effective with ~5K dictionary pairs
- mBERT/XLM-RoBERTa provide aligned spaces out-of-the-box
- CSLS retrieval metric handles hubness (some words are NN of many)
- Distant language pairs (EN-ZH, EN-AR) are harder to align

---

## Embedding Storage & Retrieval

### Question 14
- [ ] Done
**How do you implement efficient embedding storage and retrieval systems for large-scale applications?**

**Answer:**

**Definition:**
Efficient embedding storage uses: **vector databases** (FAISS, Milvus, Pinecone) with **ANN indices** (HNSW, IVF), **quantization** for compression, **sharding** for scale, and **caching** for frequent queries. Balance: recall@k vs latency vs storage cost. Target: sub-100ms retrieval at billion-scale.

**Storage Architecture:**

| Component | Purpose | Options |
|-----------|---------|---------|
| **Index** | Fast similarity search | HNSW, IVF, LSH |
| **Storage** | Persist vectors | Disk, memory-mapped |
| **Compression** | Reduce size | PQ, SQ, binary |
| **Sharding** | Horizontal scale | Hash, range |
| **Cache** | Reduce latency | Redis, in-memory |

**Scaling Considerations:**

| Scale | Vectors | Approach | Latency |
|-------|---------|----------|---------|
| **Small** | <100K | FAISS in-memory | <5ms |
| **Medium** | 100K-10M | Single node + IVF | <20ms |
| **Large** | 10M-100M | Sharded + PQ | <50ms |
| **Massive** | >100M | Distributed cluster | <100ms |

**Python Code Example:**
```python
import numpy as np
import faiss
from typing import List, Tuple, Dict
import pickle
import os

class EmbeddingStore:
    """Production-ready embedding storage and retrieval"""
    
    def __init__(self, dim: int, index_type: str = 'hnsw',
                 use_gpu: bool = False):
        self.dim = dim
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.index = None
        self.id_map = {}  # External ID -> Internal ID
        self.reverse_map = {}  # Internal ID -> External ID
        self.metadata = {}  # External ID -> metadata
    
    def build_index(self, embeddings: np.ndarray, ids: List[str] = None):
        """Build index from embeddings"""
        n = len(embeddings)
        
        if ids is None:
            ids = [str(i) for i in range(n)]
        
        # Create ID mappings
        for i, ext_id in enumerate(ids):
            self.id_map[ext_id] = i
            self.reverse_map[i] = ext_id
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build index based on type
        if self.index_type == 'flat':
            self.index = faiss.IndexFlatIP(self.dim)
        
        elif self.index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(self.dim, 32)  # 32 neighbors
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 64
        
        elif self.index_type == 'ivf':
            nlist = min(int(np.sqrt(n)), 4096)
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
            self.index.train(embeddings)
            self.index.nprobe = min(nlist // 4, 64)
        
        elif self.index_type == 'ivf_pq':
            nlist = min(int(np.sqrt(n)), 4096)
            m = 32  # Number of subquantizers
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, 8)
            self.index.train(embeddings)
        
        # Add vectors
        self.index.add(embeddings)
        
        # Move to GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, self.index
            )
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors"""
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Normalize query
        faiss.normalize_L2(query)
        
        # Search
        distances, indices = self.index.search(query, k)
        
        # Map back to external IDs
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:  # -1 indicates not found
                ext_id = self.reverse_map.get(idx, str(idx))
                results.append((ext_id, float(dist)))
        
        return results
    
    def batch_search(self, queries: np.ndarray, k: int = 10) -> List[List[Tuple[str, float]]]:
        """Batch search for efficiency"""
        faiss.normalize_L2(queries)
        distances, indices = self.index.search(queries, k)
        
        all_results = []
        for idx_row, dist_row in zip(indices, distances):
            results = []
            for idx, dist in zip(idx_row, dist_row):
                if idx >= 0:
                    ext_id = self.reverse_map.get(idx, str(idx))
                    results.append((ext_id, float(dist)))
            all_results.append(results)
        
        return all_results
    
    def add(self, embedding: np.ndarray, ext_id: str, metadata: Dict = None):
        """Add single embedding"""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        faiss.normalize_L2(embedding)
        
        internal_id = self.index.ntotal
        self.id_map[ext_id] = internal_id
        self.reverse_map[internal_id] = ext_id
        
        if metadata:
            self.metadata[ext_id] = metadata
        
        self.index.add(embedding)
    
    def save(self, path: str):
        """Save index and mappings"""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, os.path.join(path, 'index.faiss'))
        else:
            faiss.write_index(self.index, os.path.join(path, 'index.faiss'))
        
        # Save mappings
        with open(os.path.join(path, 'mappings.pkl'), 'wb') as f:
            pickle.dump({
                'id_map': self.id_map,
                'reverse_map': self.reverse_map,
                'metadata': self.metadata,
                'dim': self.dim,
                'index_type': self.index_type
            }, f)
    
    def load(self, path: str):
        """Load index and mappings"""
        self.index = faiss.read_index(os.path.join(path, 'index.faiss'))
        
        with open(os.path.join(path, 'mappings.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.id_map = data['id_map']
            self.reverse_map = data['reverse_map']
            self.metadata = data['metadata']
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, self.index
            )

# Sharded store for massive scale
class ShardedEmbeddingStore:
    """Distribute embeddings across multiple shards"""
    
    def __init__(self, num_shards: int, dim: int):
        self.num_shards = num_shards
        self.shards = [EmbeddingStore(dim) for _ in range(num_shards)]
    
    def _get_shard(self, ext_id: str) -> int:
        """Hash-based shard assignment"""
        return hash(ext_id) % self.num_shards
    
    def add(self, embedding: np.ndarray, ext_id: str, metadata: Dict = None):
        """Add to appropriate shard"""
        shard_id = self._get_shard(ext_id)
        self.shards[shard_id].add(embedding, ext_id, metadata)
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search all shards and merge results"""
        all_results = []
        
        for shard in self.shards:
            results = shard.search(query, k)
            all_results.extend(results)
        
        # Sort by score and take top-k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:k]
```

**Interview Tips:**
- HNSW is best for high recall; IVF+PQ for memory efficiency
- Always normalize embeddings for cosine similarity with inner product index
- Consider GPU for query-heavy workloads; CPU for memory-heavy
- Sharding strategy affects load balancing — hash by ID or cluster by embedding

---

### Question 15
- [ ] Done
**What techniques help optimize memory usage when working with large embedding matrices?**

**Answer:**

**Definition:**
Optimize embedding memory via: **quantization** (FP16/INT8 reduce 2-4x), **dimensionality reduction** (PCA), **sparse representations**, **memory mapping** (load on demand), **product quantization** (32x compression), and **pruning** (remove infrequent embeddings). Goal: fit billion-scale embeddings in available memory.

**Memory Reduction Techniques:**

| Technique | Compression | Quality Loss | Complexity |
|-----------|-------------|--------------|------------|
| **FP32 → FP16** | 2x | ~0% | Low |
| **FP32 → INT8** | 4x | 1-2% | Medium |
| **PQ (m=32)** | 32x | 5-10% | Medium |
| **Binary** | 32x | 10-20% | Low |
| **Dim reduction** | 2-4x | 2-5% | Low |
| **Pruning** | Variable | 0% (on kept) | Low |

**Memory Calculation:**
```
Full: N vectors × D dimensions × 4 bytes = 4ND bytes

1B vectors, 768-dim FP32: 1B × 768 × 4 = 3TB
1B vectors, 768-dim FP16: 1B × 768 × 2 = 1.5TB  
1B vectors, 768-dim INT8: 1B × 768 × 1 = 768GB
1B vectors, 768-dim PQ32: 1B × 32 × 1 = 32GB
```

**Python Code Example:**
```python
import numpy as np
import faiss
from typing import Optional
import mmap
import os

class MemoryEfficientEmbeddings:
    """Memory-optimized embedding storage"""
    
    def __init__(self, dim: int, dtype: str = 'float32'):
        self.dim = dim
        self.dtype = dtype
        self.dtype_map = {
            'float32': np.float32,
            'float16': np.float16,
            'int8': np.int8
        }
    
    def quantize_to_float16(self, embeddings: np.ndarray) -> np.ndarray:
        """Convert to FP16 (50% memory reduction)"""
        return embeddings.astype(np.float16)
    
    def quantize_to_int8(self, embeddings: np.ndarray) -> tuple:
        """
        Symmetric INT8 quantization (75% memory reduction)
        Returns quantized values and scale factors
        """
        # Compute scale per vector
        max_vals = np.abs(embeddings).max(axis=1, keepdims=True)
        scales = max_vals / 127.0
        
        # Quantize
        quantized = np.round(embeddings / (scales + 1e-8)).astype(np.int8)
        
        return quantized, scales.astype(np.float32)
    
    def dequantize_int8(self, quantized: np.ndarray, 
                        scales: np.ndarray) -> np.ndarray:
        """Dequantize INT8 back to float"""
        return quantized.astype(np.float32) * scales
    
    def product_quantization(self, embeddings: np.ndarray, 
                             m: int = 32, nbits: int = 8) -> tuple:
        """
        Product Quantization for extreme compression
        Splits vector into m subvectors, quantizes each to nbits
        """
        n, d = embeddings.shape
        
        # Train PQ
        pq = faiss.ProductQuantizer(d, m, nbits)
        pq.train(embeddings.astype(np.float32))
        
        # Encode
        codes = pq.compute_codes(embeddings.astype(np.float32))
        
        return codes, pq
    
    def binary_quantization(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Binary quantization (32x compression)
        Sign of each dimension → 1 bit
        """
        # Convert to binary: positive → 1, negative → 0
        binary = (embeddings > 0).astype(np.uint8)
        
        # Pack bits
        packed = np.packbits(binary, axis=1)
        
        return packed
    
    def hamming_distance(self, query_packed: np.ndarray, 
                         db_packed: np.ndarray) -> np.ndarray:
        """Compute Hamming distance for binary vectors"""
        # XOR and popcount
        xor = np.bitwise_xor(query_packed, db_packed)
        return np.unpackbits(xor, axis=1).sum(axis=1)

class MemoryMappedEmbeddings:
    """Memory-mapped embeddings for out-of-core processing"""
    
    def __init__(self, filepath: str, dim: int, dtype: np.dtype = np.float32):
        self.filepath = filepath
        self.dim = dim
        self.dtype = dtype
        self.mmap = None
    
    def create(self, embeddings: np.ndarray):
        """Create memory-mapped file"""
        fp = np.memmap(
            self.filepath,
            dtype=self.dtype,
            mode='w+',
            shape=embeddings.shape
        )
        fp[:] = embeddings[:]
        fp.flush()
        del fp
    
    def load(self, n_vectors: Optional[int] = None):
        """Load memory-mapped file"""
        # Get file size to determine number of vectors
        file_size = os.path.getsize(self.filepath)
        bytes_per_vector = self.dim * np.dtype(self.dtype).itemsize
        total_vectors = file_size // bytes_per_vector
        
        n = n_vectors or total_vectors
        
        self.mmap = np.memmap(
            self.filepath,
            dtype=self.dtype,
            mode='r',
            shape=(n, self.dim)
        )
        return self.mmap
    
    def get(self, indices: np.ndarray) -> np.ndarray:
        """Get specific embeddings by index"""
        return np.array(self.mmap[indices])
    
    def search_batch(self, query: np.ndarray, batch_size: int = 10000,
                     k: int = 10) -> list:
        """Search large mmap in batches"""
        n_total = len(self.mmap)
        
        # Normalize query
        query = query / (np.linalg.norm(query) + 1e-8)
        
        all_scores = []
        all_indices = []
        
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch = np.array(self.mmap[start:end])
            
            # Normalize batch
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            batch_norm = batch / (norms + 1e-8)
            
            # Compute similarities
            scores = batch_norm @ query
            
            # Get top-k for this batch
            top_k = min(k, len(scores))
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            
            for idx in top_indices:
                all_scores.append(scores[idx])
                all_indices.append(start + idx)
        
        # Global top-k
        combined = list(zip(all_indices, all_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        
        return combined[:k]

# Sparse embeddings for efficiency
class SparseEmbeddings:
    """Use sparse representations when applicable"""
    
    def __init__(self, dim: int, sparsity: float = 0.9):
        self.dim = dim
        self.sparsity = sparsity
    
    def sparsify(self, embeddings: np.ndarray, top_k: int = None) -> list:
        """Keep only top-k dimensions per vector"""
        if top_k is None:
            top_k = int(self.dim * (1 - self.sparsity))
        
        sparse_vectors = []
        for vec in embeddings:
            # Get top-k indices
            top_indices = np.argpartition(np.abs(vec), -top_k)[-top_k:]
            
            # Create sparse representation
            sparse = [(int(idx), float(vec[idx])) for idx in top_indices]
            sparse_vectors.append(sparse)
        
        return sparse_vectors
    
    def sparse_dot_product(self, sparse1: list, sparse2: list) -> float:
        """Compute dot product of two sparse vectors"""
        dict1 = dict(sparse1)
        
        result = 0.0
        for idx, val in sparse2:
            if idx in dict1:
                result += dict1[idx] * val
        
        return result
```

**Interview Tips:**
- FP16 is almost always safe; INT8 needs calibration
- PQ is best for read-heavy workloads (slower to add)
- Memory mapping enables larger-than-RAM datasets
- Combine techniques: PQ on disk, cache hot vectors in memory

---

### Question 16
- [ ] Done
**How do you implement embedding compression techniques for mobile and edge deployments?**

**Answer:**

**Definition:**
Edge deployment compression uses: **aggressive quantization** (INT4/binary), **knowledge distillation** (smaller models), **pruning** (remove dimensions), **vocabulary reduction**, and **on-device caching**. Target: <50MB model size, <100ms latency on mobile CPU, minimal quality degradation.

**Mobile Constraints:**

| Constraint | Typical Limit | Impact |
|------------|---------------|--------|
| **Model size** | 10-100MB | Limits vocabulary |
| **RAM** | 2-4GB (shared) | Can't load full embeddings |
| **Compute** | 2-4 TOPS | Limits model complexity |
| **Latency** | <100ms | Requires optimization |
| **Battery** | Limited | Favor smaller models |

**Compression Strategies for Mobile:**

| Technique | Compression | Quality Loss | Mobile-Friendly |
|-----------|-------------|--------------|-----------------|
| **INT8** | 4x | ~1% | ✓ Very |
| **INT4** | 8x | 3-5% | ✓ Good |
| **Binary** | 32x | 10-20% | ✓ Best perf |
| **Distillation** | 4-10x | 2-5% | ✓ Good |
| **Vocab pruning** | Variable | 0% (kept) | ✓ Very |

**Python Code Example:**
```python
import numpy as np
from typing import Dict, List, Tuple
import struct

class MobileEmbeddingCompressor:
    """Compress embeddings for mobile/edge deployment"""
    
    def __init__(self, embeddings: Dict[str, np.ndarray], dim: int = 300):
        self.embeddings = embeddings
        self.dim = dim
    
    def int8_quantization(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """INT8 quantization with single global scale"""
        # Find global scale
        all_values = np.concatenate([v for v in self.embeddings.values()])
        global_scale = np.abs(all_values).max() / 127.0
        
        quantized = {}
        for word, vec in self.embeddings.items():
            quantized[word] = np.round(vec / global_scale).astype(np.int8)
        
        return quantized, global_scale
    
    def int4_quantization(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """INT4 quantization (pack two values per byte)"""
        all_values = np.concatenate([v for v in self.embeddings.values()])
        global_scale = np.abs(all_values).max() / 7.0  # 4-bit signed: -8 to 7
        
        quantized = {}
        for word, vec in self.embeddings.items():
            q = np.clip(np.round(vec / global_scale), -8, 7).astype(np.int8)
            # Pack two values per byte
            packed = []
            for i in range(0, len(q), 2):
                high = (q[i] + 8) << 4
                low = (q[i+1] + 8) if i+1 < len(q) else 0
                packed.append(high | low)
            quantized[word] = np.array(packed, dtype=np.uint8)
        
        return quantized, global_scale
    
    def binary_quantization(self) -> Dict[str, np.ndarray]:
        """Binary quantization (1 bit per dimension)"""
        binary = {}
        for word, vec in self.embeddings.items():
            # Sign → binary
            bits = (vec > 0).astype(np.uint8)
            packed = np.packbits(bits)
            binary[word] = packed
        
        return binary
    
    def vocabulary_pruning(self, word_freq: Dict[str, int], 
                           top_k: int = 50000) -> Dict[str, np.ndarray]:
        """Keep only most frequent words"""
        sorted_words = sorted(word_freq.keys(), 
                             key=lambda w: word_freq[w], 
                             reverse=True)[:top_k]
        
        pruned = {w: self.embeddings[w] for w in sorted_words 
                  if w in self.embeddings}
        
        return pruned
    
    def dimension_reduction(self, target_dim: int = 128) -> Dict[str, np.ndarray]:
        """PCA-based dimension reduction"""
        from sklearn.decomposition import PCA
        
        words = list(self.embeddings.keys())
        vectors = np.array([self.embeddings[w] for w in words])
        
        pca = PCA(n_components=target_dim)
        reduced = pca.fit_transform(vectors)
        
        return {w: reduced[i] for i, w in enumerate(words)}, pca

class MobileEmbeddingExporter:
    """Export embeddings for mobile platforms"""
    
    def __init__(self, embeddings: Dict[str, np.ndarray], 
                 quantized: bool = True):
        self.embeddings = embeddings
        self.quantized = quantized
    
    def export_flatbuffer(self, output_path: str):
        """Export to FlatBuffer format (efficient for mobile)"""
        # Simplified - actual implementation uses FlatBuffers library
        words = list(self.embeddings.keys())
        vectors = np.array([self.embeddings[w] for w in words])
        
        with open(output_path, 'wb') as f:
            # Write header
            f.write(struct.pack('II', len(words), vectors.shape[1]))
            
            # Write vocabulary
            for word in words:
                encoded = word.encode('utf-8')
                f.write(struct.pack('H', len(encoded)))
                f.write(encoded)
            
            # Write vectors
            if self.quantized:
                f.write(vectors.astype(np.int8).tobytes())
            else:
                f.write(vectors.astype(np.float32).tobytes())
    
    def export_tflite_compatible(self, output_path: str):
        """Export format compatible with TFLite"""
        # Convert to TFLite lookup table
        words = list(self.embeddings.keys())
        vectors = np.array([self.embeddings[w] for w in words])
        
        # Save as numpy for conversion
        np.savez(output_path, 
                 vocabulary=np.array(words),
                 embeddings=vectors.astype(np.float16))

class MobileInference:
    """Lightweight inference for mobile"""
    
    def __init__(self, embeddings_path: str, quantized: bool = True):
        self.embeddings, self.scale = self._load(embeddings_path, quantized)
    
    def _load(self, path: str, quantized: bool):
        """Load compressed embeddings"""
        data = np.load(path, allow_pickle=True)
        embeddings = dict(zip(data['vocabulary'], data['embeddings']))
        scale = data.get('scale', 1.0)
        return embeddings, scale
    
    def get_embedding(self, word: str) -> np.ndarray:
        """Get embedding with dequantization"""
        if word not in self.embeddings:
            return None
        
        vec = self.embeddings[word]
        return vec.astype(np.float32) * self.scale
    
    def similarity(self, word1: str, word2: str) -> float:
        """Compute similarity (optimized for quantized)"""
        v1 = self.embeddings.get(word1)
        v2 = self.embeddings.get(word2)
        
        if v1 is None or v2 is None:
            return 0.0
        
        # Integer dot product (faster on mobile)
        if v1.dtype == np.int8:
            dot = np.dot(v1.astype(np.int32), v2.astype(np.int32))
            norm1 = np.sqrt(np.dot(v1.astype(np.int32), v1.astype(np.int32)))
            norm2 = np.sqrt(np.dot(v2.astype(np.int32), v2.astype(np.int32)))
            return dot / (norm1 * norm2 + 1e-8)
        else:
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
```

**Interview Tips:**
- INT8 is safe for most tasks; INT4 needs evaluation
- Vocabulary pruning to 50K words often sufficient
- Binary embeddings work well for candidate retrieval, re-rank with full
- Consider on-device model for privacy-sensitive applications

---

### Question 17
- [ ] Done
**How do you optimize embedding-based retrieval systems for sub-second response times?**

**Answer:**

**Definition:**
Achieve sub-second retrieval via: **ANN indices** (HNSW for speed, IVF for memory), **caching** (query and result caching), **pre-filtering** (metadata filters), **batching** (group concurrent queries), **hardware optimization** (GPU, SIMD), and **index partitioning**. Target: P99 < 100ms at scale.

**Latency Budget Breakdown:**

| Component | Target Latency | Optimization |
|-----------|----------------|--------------|
| **Query embedding** | 10-50ms | Distilled model, ONNX |
| **Vector search** | 5-20ms | HNSW, GPU index |
| **Post-processing** | 5-10ms | Efficient filtering |
| **Network** | 10-30ms | CDN, edge deployment |
| **Total** | <100ms | End-to-end |

**Optimization Strategies:**

| Strategy | Latency Impact | Implementation |
|----------|----------------|----------------|
| **HNSW index** | 5-10ms search | ef_search tuning |
| **Query caching** | <1ms (hit) | Redis/Memcached |
| **Result caching** | <1ms (hit) | LRU cache |
| **Pre-filtering** | 2-5x speedup | Metadata index |
| **GPU acceleration** | 2-5x speedup | FAISS GPU |
| **Batching** | Higher throughput | Async collection |

**Python Code Example:**
```python
import numpy as np
import faiss
import time
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
import hashlib
import asyncio
from collections import defaultdict

class OptimizedRetrievalSystem:
    """Sub-second retrieval with multiple optimizations"""
    
    def __init__(self, dim: int, use_gpu: bool = True):
        self.dim = dim
        self.use_gpu = use_gpu
        self.index = None
        self.query_cache = {}
        self.result_cache = {}
        self.metadata_index = {}
        
    def build_optimized_index(self, embeddings: np.ndarray, 
                               metadata: List[Dict] = None):
        """Build HNSW index optimized for speed"""
        n = len(embeddings)
        
        # Normalize for inner product (cosine)
        faiss.normalize_L2(embeddings)
        
        # HNSW with tuned parameters
        self.index = faiss.IndexHNSWFlat(self.dim, 32)
        self.index.hnsw.efConstruction = 200  # Build quality
        self.index.hnsw.efSearch = 64  # Search quality/speed tradeoff
        
        self.index.add(embeddings)
        
        # Move to GPU
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Build metadata index for pre-filtering
        if metadata:
            for i, meta in enumerate(metadata):
                for key, value in meta.items():
                    if key not in self.metadata_index:
                        self.metadata_index[key] = defaultdict(list)
                    self.metadata_index[key][value].append(i)
    
    def _get_query_hash(self, query: np.ndarray) -> str:
        """Hash query for caching"""
        return hashlib.md5(query.tobytes()).hexdigest()
    
    def search_with_cache(self, query: np.ndarray, k: int = 10,
                          filters: Dict = None) -> List[Tuple[int, float]]:
        """Search with query and result caching"""
        
        # Check cache
        cache_key = (self._get_query_hash(query), k, str(filters))
        if cache_key in self.result_cache:
            return self.result_cache[cache_key]
        
        # Pre-filter by metadata
        candidate_ids = None
        if filters:
            candidate_ids = self._apply_filters(filters)
            if len(candidate_ids) == 0:
                return []
        
        # Search
        query = query.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        
        if candidate_ids is not None:
            # Search only within candidates
            results = self._search_subset(query, candidate_ids, k)
        else:
            distances, indices = self.index.search(query, k)
            results = list(zip(indices[0], distances[0]))
        
        # Cache result
        if len(self.result_cache) < 10000:  # Limit cache size
            self.result_cache[cache_key] = results
        
        return results
    
    def _apply_filters(self, filters: Dict) -> set:
        """Apply metadata filters to get candidate IDs"""
        candidate_sets = []
        
        for key, value in filters.items():
            if key in self.metadata_index and value in self.metadata_index[key]:
                candidate_sets.append(set(self.metadata_index[key][value]))
        
        if not candidate_sets:
            return set()
        
        # Intersection of all filters
        return set.intersection(*candidate_sets)
    
    def _search_subset(self, query: np.ndarray, 
                       candidate_ids: set, k: int) -> List[Tuple[int, float]]:
        """Search within subset of vectors"""
        # For small subsets, brute force
        if len(candidate_ids) < 1000:
            # This is a simplified version
            # Real implementation would use IDSelector
            pass
        
        # For larger subsets, use IDSelectorArray
        ids = np.array(list(candidate_ids), dtype=np.int64)
        selector = faiss.IDSelectorArray(len(ids), faiss.swig_ptr(ids))
        
        # Create filtered index
        params = faiss.SearchParametersHNSW()
        params.sel = selector
        
        distances, indices = self.index.search(query, k, params=params)
        return list(zip(indices[0], distances[0]))

class BatchedRetrieval:
    """Batch queries for higher throughput"""
    
    def __init__(self, retrieval_system, batch_size: int = 32,
                 max_wait_ms: float = 10.0):
        self.system = retrieval_system
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_queries = []
        self.results = {}
    
    async def search_async(self, query: np.ndarray, k: int = 10) -> List:
        """Async search with batching"""
        query_id = id(query)
        future = asyncio.Future()
        
        self.pending_queries.append({
            'id': query_id,
            'query': query,
            'k': k,
            'future': future
        })
        
        # Trigger batch if full
        if len(self.pending_queries) >= self.batch_size:
            await self._process_batch()
        else:
            # Wait for batch or timeout
            await asyncio.sleep(self.max_wait_ms / 1000)
            if self.pending_queries:
                await self._process_batch()
        
        return await future
    
    async def _process_batch(self):
        """Process accumulated queries as batch"""
        if not self.pending_queries:
            return
        
        # Collect batch
        batch = self.pending_queries[:self.batch_size]
        self.pending_queries = self.pending_queries[self.batch_size:]
        
        # Stack queries
        queries = np.vstack([q['query'] for q in batch])
        max_k = max(q['k'] for q in batch)
        
        # Batch search
        faiss.normalize_L2(queries)
        distances, indices = self.system.index.search(queries, max_k)
        
        # Distribute results
        for i, q in enumerate(batch):
            k = q['k']
            results = list(zip(indices[i][:k], distances[i][:k]))
            q['future'].set_result(results)

# Performance monitoring
class RetrievalMonitor:
    """Monitor retrieval latency"""
    
    def __init__(self):
        self.latencies = []
    
    def timed_search(self, system, query: np.ndarray, k: int = 10):
        """Search with timing"""
        start = time.perf_counter()
        results = system.search_with_cache(query, k)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        
        self.latencies.append(elapsed)
        return results, elapsed
    
    def get_stats(self) -> Dict:
        """Get latency statistics"""
        if not self.latencies:
            return {}
        
        return {
            'p50': np.percentile(self.latencies, 50),
            'p95': np.percentile(self.latencies, 95),
            'p99': np.percentile(self.latencies, 99),
            'mean': np.mean(self.latencies),
            'count': len(self.latencies)
        }
```

**Interview Tips:**
- HNSW ef_search of 64-128 balances speed and recall
- Cache hit rates of 60-80% are common for production search
- Pre-filtering before vector search is critical for large collections
- Batch queries when throughput matters more than latency

---

### Question 18
- [ ] Done
**What are the best practices for handling embedding drift in production systems over time?**

**Answer:**

**Definition:**
Embedding drift occurs when **embedding model changes** (updates, fine-tuning) or **data distribution shifts** (new vocabulary, changing topics). Handle via: **versioning** (track embedding versions), **incremental re-indexing**, **drift detection** (monitor similarity distributions), and **backward-compatible updates**.

**Types of Drift:**

| Drift Type | Cause | Detection | Mitigation |
|------------|-------|-----------|------------|
| **Model drift** | Model update/retrain | Version tracking | Re-embed all data |
| **Vocabulary drift** | New terms, jargon | OOV rate increase | Retrain tokenizer |
| **Semantic drift** | Meaning changes | Similarity shift | Periodic retraining |
| **Data drift** | Distribution shift | Embedding statistics | Monitor + retrain |

**Drift Management Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Full re-index** | Re-embed everything | Major model change |
| **Incremental** | Re-embed new/changed | Minor updates |
| **Parallel index** | Run old + new side-by-side | Safe rollout |
| **Canary testing** | Test new on subset | Validate before switch |

**Python Code Example:**
```python
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from scipy import stats
import json

class EmbeddingDriftDetector:
    """Detect and handle embedding drift"""
    
    def __init__(self, baseline_stats: Dict = None):
        self.baseline_stats = baseline_stats or {}
        self.current_stats = {}
        self.drift_history = []
    
    def compute_statistics(self, embeddings: np.ndarray) -> Dict:
        """Compute embedding distribution statistics"""
        return {
            'mean': embeddings.mean(axis=0),
            'std': embeddings.std(axis=0),
            'global_mean': float(embeddings.mean()),
            'global_std': float(embeddings.std()),
            'norms_mean': float(np.linalg.norm(embeddings, axis=1).mean()),
            'norms_std': float(np.linalg.norm(embeddings, axis=1).std()),
            'computed_at': datetime.now().isoformat()
        }
    
    def set_baseline(self, embeddings: np.ndarray):
        """Set baseline from initial embeddings"""
        self.baseline_stats = self.compute_statistics(embeddings)
    
    def detect_drift(self, new_embeddings: np.ndarray, 
                     threshold: float = 0.1) -> Dict:
        """Detect if significant drift has occurred"""
        
        current = self.compute_statistics(new_embeddings)
        
        if not self.baseline_stats:
            self.baseline_stats = current
            return {'drift_detected': False, 'reason': 'Set as baseline'}
        
        # Compare statistics
        mean_shift = np.abs(current['mean'] - self.baseline_stats['mean']).mean()
        std_shift = np.abs(current['std'] - self.baseline_stats['std']).mean()
        norm_shift = abs(current['norms_mean'] - self.baseline_stats['norms_mean'])
        
        # Normalized shifts
        mean_shift_norm = mean_shift / (self.baseline_stats['global_std'] + 1e-8)
        
        # KL divergence approximation (assuming Gaussian)
        kl_div = 0.5 * (
            (current['global_std'] / (self.baseline_stats['global_std'] + 1e-8)) ** 2 +
            (self.baseline_stats['global_mean'] - current['global_mean']) ** 2 / 
            (self.baseline_stats['global_std'] ** 2 + 1e-8) - 1 +
            2 * np.log(self.baseline_stats['global_std'] / (current['global_std'] + 1e-8))
        )
        
        drift_detected = mean_shift_norm > threshold or kl_div > threshold
        
        result = {
            'drift_detected': drift_detected,
            'mean_shift': float(mean_shift_norm),
            'kl_divergence': float(kl_div),
            'norm_shift': float(norm_shift),
            'timestamp': datetime.now().isoformat()
        }
        
        self.drift_history.append(result)
        
        return result

class EmbeddingVersionManager:
    """Manage embedding versions for safe updates"""
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.versions = {}
        self.current_version = None
    
    def register_version(self, version: str, model_id: str, 
                         metadata: Dict = None):
        """Register a new embedding version"""
        self.versions[version] = {
            'model_id': model_id,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {},
            'status': 'pending'
        }
    
    def set_active(self, version: str):
        """Set version as active"""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        # Deactivate current
        if self.current_version:
            self.versions[self.current_version]['status'] = 'archived'
        
        self.versions[version]['status'] = 'active'
        self.current_version = version
    
    def get_version_info(self, version: str = None) -> Dict:
        """Get version information"""
        v = version or self.current_version
        return self.versions.get(v, {})

class IncrementalReindexer:
    """Handle incremental re-indexing during updates"""
    
    def __init__(self, old_embedder, new_embedder, index):
        self.old_embedder = old_embedder
        self.new_embedder = new_embedder
        self.index = index
        self.migration_progress = 0
    
    def compare_embeddings(self, texts: List[str]) -> Dict:
        """Compare old vs new embeddings for same texts"""
        old_embs = np.array([self.old_embedder.embed(t) for t in texts])
        new_embs = np.array([self.new_embedder.embed(t) for t in texts])
        
        # Compute similarities between old and new
        similarities = []
        for o, n in zip(old_embs, new_embs):
            sim = np.dot(o, n) / (np.linalg.norm(o) * np.linalg.norm(n) + 1e-8)
            similarities.append(sim)
        
        return {
            'mean_similarity': float(np.mean(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'high_drift_count': sum(1 for s in similarities if s < 0.9)
        }
    
    def migrate_batch(self, doc_ids: List[str], texts: List[str],
                      batch_size: int = 100):
        """Migrate documents in batches"""
        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            
            # Generate new embeddings
            new_embs = np.array([self.new_embedder.embed(t) for t in batch_texts])
            
            # Update index (implementation depends on index type)
            for doc_id, emb in zip(batch_ids, new_embs):
                self.index.update(doc_id, emb)
            
            self.migration_progress = (i + batch_size) / len(doc_ids)
            yield self.migration_progress

class ParallelIndexStrategy:
    """Run old and new index in parallel for safe migration"""
    
    def __init__(self, old_index, new_index):
        self.old_index = old_index
        self.new_index = new_index
        self.traffic_ratio = 0.0  # 0 = all old, 1 = all new
    
    def set_traffic_ratio(self, ratio: float):
        """Set traffic split between indices"""
        self.traffic_ratio = max(0.0, min(1.0, ratio))
    
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[List, str]:
        """Search with traffic splitting"""
        import random
        
        if random.random() < self.traffic_ratio:
            results = self.new_index.search(query, k)
            source = 'new'
        else:
            results = self.old_index.search(query, k)
            source = 'old'
        
        return results, source
    
    def compare_results(self, query: np.ndarray, k: int = 10) -> Dict:
        """Compare results from both indices"""
        old_results = set([r[0] for r in self.old_index.search(query, k)])
        new_results = set([r[0] for r in self.new_index.search(query, k)])
        
        overlap = len(old_results.intersection(new_results))
        
        return {
            'overlap_ratio': overlap / k,
            'only_old': len(old_results - new_results),
            'only_new': len(new_results - old_results)
        }
```

**Interview Tips:**
- Always version embeddings alongside the model that created them
- Use parallel indices for zero-downtime migrations
- Monitor embedding statistics continuously (not just accuracy)
- Re-indexing is expensive — design for incremental updates when possible

---

## Vector Databases - Selection & Architecture

### Question 19
- [ ] Done
**How do you choose between FAISS, ChromaDB, Pinecone, Weaviate, and Milvus based on your requirements?**

**Answer:**

**Definition:**
Choose based on: **scale** (FAISS/Milvus for billions, Pinecone for managed), **ops overhead** (Pinecone lowest, FAISS highest), **features** (Weaviate for hybrid search), **cost** (FAISS free, Pinecone managed), and **integration needs**. Trade-off: control vs convenience.

**Vector Database Comparison:**

| Feature | FAISS | ChromaDB | Pinecone | Weaviate | Milvus |
|---------|-------|----------|----------|----------|--------|
| **Type** | Library | Embedded | Managed | Self-host/Cloud | Self-host |
| **Scale** | Billions | Millions | Billions | Billions | Billions |
| **Ops effort** | High | Low | None | Medium | Medium |
| **Hybrid search** | No | Limited | Yes | Yes | Yes |
| **Filtering** | Basic | Yes | Yes | Yes | Yes |
| **Cost** | Free | Free | $$$ | $ (self-host) | Free |
| **Production-ready** | Yes | Developing | Yes | Yes | Yes |

**Decision Framework:**

| Requirement | Best Choice |
|-------------|-------------|
| **Quick prototype** | ChromaDB |
| **Maximum control** | FAISS |
| **No ops team** | Pinecone |
| **Hybrid search** | Weaviate |
| **Massive scale** | Milvus or FAISS |
| **Cloud-native** | Pinecone or Weaviate Cloud |
| **On-premise** | Milvus or FAISS |

**Python Code Example:**
```python
import numpy as np
from typing import List, Dict, Tuple

# FAISS - Maximum control
def faiss_example(embeddings: np.ndarray, queries: np.ndarray, k: int = 10):
    import faiss
    
    dim = embeddings.shape[1]
    
    # Build index
    index = faiss.IndexHNSWFlat(dim, 32)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Search
    faiss.normalize_L2(queries)
    distances, indices = index.search(queries, k)
    
    return indices, distances

# ChromaDB - Simple embedded database
def chromadb_example(documents: List[str], queries: List[str]):
    import chromadb
    
    # Create client and collection
    client = chromadb.Client()
    collection = client.create_collection("my_collection")
    
    # Add documents (auto-embeds)
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    
    # Query
    results = collection.query(
        query_texts=queries,
        n_results=10
    )
    
    return results

# Pinecone - Managed service
def pinecone_example(embeddings: np.ndarray, metadata: List[Dict]):
    from pinecone import Pinecone
    
    # Initialize
    pc = Pinecone(api_key="your-api-key")
    
    # Create index
    index = pc.create_index(
        name="my-index",
        dimension=embeddings.shape[1],
        metric="cosine"
    )
    
    # Upsert vectors
    vectors = [
        {"id": f"vec_{i}", "values": emb.tolist(), "metadata": meta}
        for i, (emb, meta) in enumerate(zip(embeddings, metadata))
    ]
    index.upsert(vectors=vectors)
    
    # Query with filter
    results = index.query(
        vector=embeddings[0].tolist(),
        top_k=10,
        filter={"category": "tech"}
    )
    
    return results

# Weaviate - Hybrid search
def weaviate_example(documents: List[Dict]):
    import weaviate
    
    client = weaviate.Client("http://localhost:8080")
    
    # Create schema
    schema = {
        "class": "Document",
        "vectorizer": "text2vec-transformers",
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "category", "dataType": ["string"]}
        ]
    }
    client.schema.create_class(schema)
    
    # Add data
    for doc in documents:
        client.data_object.create(doc, "Document")
    
    # Hybrid search (vector + BM25)
    results = client.query.get("Document", ["content"]) \
        .with_hybrid(query="machine learning", alpha=0.5) \
        .with_limit(10) \
        .do()
    
    return results

# Milvus - Open-source at scale
def milvus_example(embeddings: np.ndarray):
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
    
    # Connect
    connections.connect("default", host="localhost", port="19530")
    
    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embeddings.shape[1])
    ]
    schema = CollectionSchema(fields, "Document embeddings")
    
    # Create collection
    collection = Collection("documents", schema)
    
    # Insert
    collection.insert([embeddings.tolist()])
    
    # Create index
    index_params = {
        "metric_type": "IP",
        "index_type": "HNSW",
        "params": {"M": 32, "efConstruction": 200}
    }
    collection.create_index("embedding", index_params)
    
    # Load and search
    collection.load()
    results = collection.search(
        data=[embeddings[0].tolist()],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"ef": 64}},
        limit=10
    )
    
    return results

# Selection helper
class VectorDBSelector:
    """Help select appropriate vector database"""
    
    @staticmethod
    def recommend(requirements: Dict) -> str:
        """Recommend based on requirements"""
        
        # Score each option
        scores = {
            'faiss': 0,
            'chromadb': 0,
            'pinecone': 0,
            'weaviate': 0,
            'milvus': 0
        }
        
        # Scale requirements
        if requirements.get('scale', 0) > 100_000_000:
            scores['faiss'] += 3
            scores['milvus'] += 3
            scores['pinecone'] += 2
        elif requirements.get('scale', 0) > 1_000_000:
            scores['milvus'] += 2
            scores['pinecone'] += 2
            scores['weaviate'] += 2
        else:
            scores['chromadb'] += 3
        
        # Ops overhead
        if requirements.get('minimal_ops', False):
            scores['pinecone'] += 4
            scores['chromadb'] += 3
        
        # Hybrid search
        if requirements.get('hybrid_search', False):
            scores['weaviate'] += 4
            scores['pinecone'] += 2
        
        # Cost sensitivity
        if requirements.get('cost_sensitive', False):
            scores['faiss'] += 3
            scores['chromadb'] += 3
            scores['milvus'] += 2
        
        # On-premise requirement
        if requirements.get('on_premise', False):
            scores['faiss'] += 3
            scores['milvus'] += 3
            scores['weaviate'] += 2
            scores['pinecone'] -= 5
        
        return max(scores.keys(), key=lambda x: scores[x])
```

**Interview Tips:**
- Start with ChromaDB for prototypes, migrate later
- Pinecone for teams without dedicated ML infrastructure
- FAISS when you need maximum control and have engineering resources
- Weaviate for semantic search applications requiring hybrid retrieval

---

### Question 20
- [ ] Done
**What are the trade-offs between self-hosted (FAISS/ChromaDB) versus managed solutions (Pinecone)?**

**Answer:**

**Definition:**
**Self-hosted** offers control, customization, and cost efficiency at scale, but requires ops expertise. **Managed solutions** provide simplicity, scalability, and reliability guarantees, but with vendor lock-in and higher per-query costs. Choice depends on team capabilities, scale, and operational requirements.

**Trade-off Comparison:**

| Dimension | Self-Hosted | Managed (Pinecone) |
|-----------|-------------|-------------------|
| **Initial cost** | Low (free software) | Medium (usage-based) |
| **Scale cost** | Low (infrastructure) | High (per-query) |
| **Ops burden** | High | None |
| **Customization** | Full | Limited |
| **Reliability** | Your responsibility | SLA-backed |
| **Time to deploy** | Days/weeks | Hours |
| **Vendor lock-in** | None | Significant |

**Cost Analysis at Scale:**

| Scale | Self-Hosted (Monthly) | Pinecone (Monthly) |
|-------|----------------------|-------------------|
| **1M vectors** | ~$100 (server) | ~$70 (starter) |
| **10M vectors** | ~$300 (larger server) | ~$700 |
| **100M vectors** | ~$1,000 (cluster) | ~$3,000+ |
| **1B vectors** | ~$5,000 (cluster) | Custom pricing |

**When to Choose Each:**

| Scenario | Recommendation | Reason |
|----------|----------------|--------|
| **Startup MVP** | Managed | Speed to market |
| **Enterprise at scale** | Self-hosted | Cost control |
| **Small team** | Managed | No ops overhead |
| **Data sovereignty** | Self-hosted | Full control |
| **Variable load** | Managed | Auto-scaling |
| **Custom algorithms** | Self-hosted | Full flexibility |

**Python Code Example:**
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import numpy as np

class VectorStore(ABC):
    """Abstract interface for vector stores"""
    
    @abstractmethod
    def add(self, embeddings: np.ndarray, ids: List[str], 
            metadata: List[Dict] = None) -> None:
        pass
    
    @abstractmethod
    def search(self, query: np.ndarray, k: int = 10,
               filters: Dict = None) -> List[Tuple[str, float]]:
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        pass

# Self-hosted FAISS implementation
class FAISSVectorStore(VectorStore):
    """Self-hosted FAISS-based vector store"""
    
    def __init__(self, dim: int):
        import faiss
        
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product for cosine
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.metadata = {}
        
    def add(self, embeddings: np.ndarray, ids: List[str],
            metadata: List[Dict] = None):
        import faiss
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        start_idx = self.index.ntotal
        for i, ext_id in enumerate(ids):
            idx = start_idx + i
            self.id_to_idx[ext_id] = idx
            self.idx_to_id[idx] = ext_id
            if metadata and metadata[i]:
                self.metadata[ext_id] = metadata[i]
        
        self.index.add(embeddings.astype(np.float32))
    
    def search(self, query: np.ndarray, k: int = 10,
               filters: Dict = None) -> List[Tuple[str, float]]:
        import faiss
        
        query = query.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        
        # Over-fetch if filtering
        search_k = k * 10 if filters else k
        
        distances, indices = self.index.search(query, search_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            
            ext_id = self.idx_to_id.get(idx)
            if ext_id is None:
                continue
            
            # Apply filters
            if filters:
                doc_meta = self.metadata.get(ext_id, {})
                if not all(doc_meta.get(k) == v for k, v in filters.items()):
                    continue
            
            results.append((ext_id, float(dist)))
            if len(results) >= k:
                break
        
        return results
    
    def delete(self, ids: List[str]) -> None:
        # FAISS doesn't support direct deletion
        # Would need to rebuild index or use IDMap
        pass
    
    # Self-hosted responsibilities
    def backup(self, path: str):
        """Backup index to disk"""
        import faiss
        faiss.write_index(self.index, f"{path}/index.faiss")
        # Also save metadata, mappings...
    
    def monitor_health(self) -> Dict:
        """Health check - your responsibility"""
        return {
            'total_vectors': self.index.ntotal,
            'memory_mb': self.index.ntotal * self.dim * 4 / 1e6,
            'status': 'healthy'
        }

# Managed Pinecone implementation
class PineconeVectorStore(VectorStore):
    """Managed Pinecone vector store"""
    
    def __init__(self, api_key: str, index_name: str, dim: int):
        from pinecone import Pinecone
        
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.dim = dim
    
    def add(self, embeddings: np.ndarray, ids: List[str],
            metadata: List[Dict] = None):
        
        vectors = []
        for i, (emb, ext_id) in enumerate(zip(embeddings, ids)):
            vec = {
                "id": ext_id,
                "values": emb.tolist()
            }
            if metadata and metadata[i]:
                vec["metadata"] = metadata[i]
            vectors.append(vec)
        
        # Batch upsert
        self.index.upsert(vectors=vectors, batch_size=100)
    
    def search(self, query: np.ndarray, k: int = 10,
               filters: Dict = None) -> List[Tuple[str, float]]:
        
        results = self.index.query(
            vector=query.tolist(),
            top_k=k,
            filter=filters,
            include_metadata=True
        )
        
        return [(m.id, m.score) for m in results.matches]
    
    def delete(self, ids: List[str]) -> None:
        self.index.delete(ids=ids)

# Cost estimator
class CostEstimator:
    """Estimate costs for different deployment options"""
    
    @staticmethod
    def estimate_self_hosted(num_vectors: int, dim: int, 
                             queries_per_month: int) -> Dict:
        """Estimate self-hosted costs"""
        
        # Memory calculation
        memory_gb = (num_vectors * dim * 4) / 1e9
        
        # Server sizing (rough)
        if memory_gb < 16:
            server_cost = 100  # Small instance
        elif memory_gb < 64:
            server_cost = 300
        elif memory_gb < 256:
            server_cost = 800
        else:
            server_cost = 2000 + (memory_gb - 256) * 5
        
        # Ops overhead (engineering time)
        ops_cost = 500  # Hours per month * rate
        
        return {
            'server_monthly': server_cost,
            'ops_monthly': ops_cost,
            'total_monthly': server_cost + ops_cost,
            'cost_per_query': (server_cost + ops_cost) / queries_per_month
        }
    
    @staticmethod
    def estimate_pinecone(num_vectors: int, dim: int,
                          queries_per_month: int) -> Dict:
        """Estimate Pinecone costs (approximate)"""
        
        # Pod sizing
        vectors_per_pod = 1_000_000  # Approximate
        pods_needed = max(1, num_vectors // vectors_per_pod)
        
        # Pod cost (approximate)
        pod_cost = 70 * pods_needed  # Starter tier
        
        # Query costs at scale
        query_cost = queries_per_month * 0.0001  # Approximate
        
        return {
            'pods_monthly': pod_cost,
            'queries_monthly': query_cost,
            'total_monthly': pod_cost + query_cost,
            'cost_per_query': (pod_cost + query_cost) / queries_per_month
        }
```

**Interview Tips:**
- Self-hosted makes sense at >100M vectors due to cost savings
- Managed is worth the premium for small teams without ops expertise
- Always build abstraction layer to enable migration between options
- Evaluate total cost of ownership including engineering time

---

### Question 21
- [ ] Done
**How do you optimize index selection (IVF, HNSW, LSH, PQ) in FAISS for different query patterns?**

**Answer:**

**Definition:**
Index selection depends on: **recall requirements** (HNSW highest), **memory constraints** (PQ most efficient), **update frequency** (Flat/IVF for frequent adds), and **query latency** (HNSW fastest). Each index trades off search speed, memory, and accuracy differently.

**Index Comparison:**

| Index | Search Speed | Memory | Build Time | Recall | Updates |
|-------|--------------|--------|------------|--------|---------|
| **Flat** | O(n) | High | O(n) | 100% | Easy |
| **IVF** | O(n/k) | Medium | O(n) | 95-99% | Easy |
| **HNSW** | O(log n) | High | O(n log n) | 99%+ | Hard |
| **PQ** | O(n) | Very Low | O(n) | 80-95% | Medium |
| **IVF+PQ** | O(n/k) | Low | O(n) | 85-95% | Medium |
| **LSH** | O(1)* | Medium | O(n) | 80-90% | Easy |

**Selection Guide:**

| Scenario | Recommended Index | Configuration |
|----------|-------------------|---------------|
| **<100K vectors** | Flat | IndexFlatIP |
| **Speed priority** | HNSW | M=32, efSearch=64 |
| **Memory constrained** | IVF+PQ | nlist=√n, m=32 |
| **Frequent updates** | IVF | nlist=√n |
| **Maximum recall** | HNSW | efSearch=128+ |
| **Billion scale** | IVF+PQ + sharding | Custom |

**Python Code Example:**
```python
import numpy as np
import faiss
from typing import Tuple, Dict
import time

class FAISSIndexFactory:
    """Create optimized FAISS indices for different use cases"""
    
    def __init__(self, dim: int, n_vectors: int):
        self.dim = dim
        self.n_vectors = n_vectors
    
    def create_flat(self, metric: str = 'ip') -> faiss.Index:
        """Exact search - best for small datasets"""
        if metric == 'ip':
            return faiss.IndexFlatIP(self.dim)
        else:
            return faiss.IndexFlatL2(self.dim)
    
    def create_ivf(self, nlist: int = None, 
                   nprobe: int = None) -> faiss.Index:
        """IVF - good balance of speed and recall"""
        
        nlist = nlist or int(np.sqrt(self.n_vectors))
        nprobe = nprobe or max(1, nlist // 16)
        
        quantizer = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
        index.nprobe = nprobe
        
        return index
    
    def create_hnsw(self, M: int = 32, 
                    efConstruction: int = 200,
                    efSearch: int = 64) -> faiss.Index:
        """HNSW - fastest search, highest memory"""
        
        index = faiss.IndexHNSWFlat(self.dim, M)
        index.hnsw.efConstruction = efConstruction
        index.hnsw.efSearch = efSearch
        
        return index
    
    def create_ivf_pq(self, nlist: int = None,
                      m: int = 32, nbits: int = 8) -> faiss.Index:
        """IVF+PQ - memory efficient at scale"""
        
        nlist = nlist or min(int(np.sqrt(self.n_vectors)), 4096)
        
        quantizer = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, nbits)
        
        return index
    
    def create_opq_ivf_pq(self, nlist: int = None,
                          m: int = 32) -> faiss.Index:
        """OPQ + IVF + PQ - best compression with rotation"""
        
        nlist = nlist or min(int(np.sqrt(self.n_vectors)), 4096)
        
        # Build index string
        index_string = f"OPQ{m},IVF{nlist},PQ{m}"
        index = faiss.index_factory(self.dim, index_string)
        
        return index

class IndexBenchmark:
    """Benchmark different indices"""
    
    def __init__(self, embeddings: np.ndarray, queries: np.ndarray):
        self.embeddings = embeddings.astype(np.float32)
        self.queries = queries.astype(np.float32)
        faiss.normalize_L2(self.embeddings)
        faiss.normalize_L2(self.queries)
        
        # Ground truth with flat index
        flat_index = faiss.IndexFlatIP(embeddings.shape[1])
        flat_index.add(self.embeddings)
        self.gt_distances, self.gt_indices = flat_index.search(self.queries, 100)
    
    def benchmark_index(self, index: faiss.Index, k: int = 10) -> Dict:
        """Benchmark single index"""
        
        # Training if needed
        if not index.is_trained:
            index.train(self.embeddings)
        
        # Build time
        start = time.perf_counter()
        index.add(self.embeddings)
        build_time = time.perf_counter() - start
        
        # Search time
        start = time.perf_counter()
        distances, indices = index.search(self.queries, k)
        search_time = time.perf_counter() - start
        
        # Calculate recall
        recall = self._calculate_recall(indices, k)
        
        # Memory estimate
        if hasattr(index, 'code_size'):
            memory_per_vector = index.code_size
        else:
            memory_per_vector = self.embeddings.shape[1] * 4
        
        return {
            'build_time_s': build_time,
            'search_time_ms': search_time * 1000 / len(self.queries),
            'recall@k': recall,
            'memory_mb': (len(self.embeddings) * memory_per_vector) / 1e6,
            'qps': len(self.queries) / search_time
        }
    
    def _calculate_recall(self, indices: np.ndarray, k: int) -> float:
        """Calculate recall@k"""
        correct = 0
        for i, pred in enumerate(indices):
            gt = set(self.gt_indices[i][:k])
            found = set(pred)
            correct += len(gt.intersection(found))
        return correct / (len(indices) * k)

# Index tuning
class IndexTuner:
    """Tune index parameters for target performance"""
    
    def __init__(self, index: faiss.Index, benchmark: IndexBenchmark):
        self.index = index
        self.benchmark = benchmark
    
    def tune_ivf(self, target_recall: float = 0.95,
                 target_latency_ms: float = 10.0) -> Dict:
        """Tune IVF nprobe for recall/latency target"""
        
        results = []
        
        for nprobe in [1, 2, 4, 8, 16, 32, 64, 128]:
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = nprobe
            
            metrics = self.benchmark.benchmark_index(self.index, k=10)
            metrics['nprobe'] = nprobe
            results.append(metrics)
            
            # Check if targets met
            if (metrics['recall@k'] >= target_recall and 
                metrics['search_time_ms'] <= target_latency_ms):
                return {
                    'optimal_nprobe': nprobe,
                    'recall': metrics['recall@k'],
                    'latency_ms': metrics['search_time_ms']
                }
        
        # Return best effort
        best = max(results, key=lambda x: x['recall@k'])
        return {
            'optimal_nprobe': best['nprobe'],
            'recall': best['recall@k'],
            'latency_ms': best['search_time_ms'],
            'note': 'Target not achieved'
        }
    
    def tune_hnsw(self, target_recall: float = 0.99) -> Dict:
        """Tune HNSW efSearch"""
        
        for ef in [16, 32, 64, 128, 256, 512]:
            self.index.hnsw.efSearch = ef
            metrics = self.benchmark.benchmark_index(self.index, k=10)
            
            if metrics['recall@k'] >= target_recall:
                return {
                    'optimal_efSearch': ef,
                    'recall': metrics['recall@k'],
                    'latency_ms': metrics['search_time_ms']
                }
        
        return {'note': 'Target not achieved'}
```

**Interview Tips:**
- HNSW is best default for most cases (fast, high recall)
- IVF+PQ for memory-constrained billion-scale
- Always tune nprobe/efSearch on representative queries
- PQ m parameter should divide dimension evenly

---

### Question 22
- [ ] Done
**Explain HNSW (Hierarchical Navigable Small World) graphs and when they outperform other indices.**

**Answer:**

**Definition:**
HNSW builds a **multi-layer graph** where each layer is a "small world" network. Upper layers have sparse connections for fast global navigation; lower layers are denser for precise local search. Search starts at top layer, greedily navigates to nearest neighbor, then descends. Achieves O(log n) search with high recall.

**HNSW Structure:**

```
Layer 3: ● -------- ● (sparse, long-range connections)
         |          |
Layer 2: ● --- ● -- ● --- ● (medium density)
         |    |    |     |
Layer 1: ●-●--●-●--●-●--●-● (dense connections)
         | |  | |  | |  | |
Layer 0: ●●●●●●●●●●●●●●●●●● (all vectors, densest)
```

**Key Parameters:**

| Parameter | Description | Effect |
|-----------|-------------|--------|
| **M** | Connections per node | Higher = better recall, more memory |
| **efConstruction** | Build-time beam width | Higher = better graph, slower build |
| **efSearch** | Search-time beam width | Higher = better recall, slower search |

**When HNSW Outperforms:**

| Scenario | HNSW Advantage |
|----------|----------------|
| **High recall needed (>99%)** | Best recall at any speed |
| **Sub-10ms latency** | Fastest search |
| **Read-heavy workload** | Search optimized |
| **Medium-scale (1M-100M)** | Sweet spot |

**When to Avoid HNSW:**

| Scenario | Better Alternative |
|----------|-------------------|
| **Memory constrained** | IVF+PQ |
| **Frequent inserts** | IVF (HNSW rebuild costly) |
| **Billion+ scale** | Sharded IVF+PQ |
| **100% recall required** | Flat index |

**Python Code Example:**
```python
import numpy as np
import faiss
from typing import List, Tuple

class HNSWIndex:
    """HNSW index with tuning utilities"""
    
    def __init__(self, dim: int, M: int = 32, 
                 efConstruction: int = 200):
        self.dim = dim
        self.M = M
        self.efConstruction = efConstruction
        
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = efConstruction
        self.index.hnsw.efSearch = 64  # Default
    
    def add(self, embeddings: np.ndarray):
        """Add vectors to index"""
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
    
    def search(self, query: np.ndarray, k: int = 10,
               efSearch: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Search with optional efSearch override"""
        if efSearch:
            self.index.hnsw.efSearch = efSearch
        
        query = query.reshape(-1, self.dim).astype(np.float32)
        faiss.normalize_L2(query)
        
        return self.index.search(query, k)
    
    def get_graph_stats(self) -> dict:
        """Get HNSW graph statistics"""
        hnsw = self.index.hnsw
        
        return {
            'num_vectors': self.index.ntotal,
            'max_level': hnsw.max_level,
            'M': self.M,
            'efConstruction': self.efConstruction,
            'efSearch': hnsw.efSearch,
            'memory_mb': self._estimate_memory() / 1e6
        }
    
    def _estimate_memory(self) -> int:
        """Estimate memory usage in bytes"""
        n = self.index.ntotal
        
        # Vector storage
        vector_bytes = n * self.dim * 4
        
        # Graph storage (approximate)
        # Each level has fewer vectors: n, n/M, n/M², ...
        graph_bytes = 0
        level_n = n
        level = 0
        while level_n > 1:
            # Each node stores ~2M neighbors
            graph_bytes += level_n * 2 * self.M * 4
            level_n = level_n // self.M
            level += 1
        
        return vector_bytes + graph_bytes

class HNSWSearchTuner:
    """Tune HNSW for recall/latency trade-off"""
    
    def __init__(self, index: faiss.Index, 
                 queries: np.ndarray,
                 ground_truth: np.ndarray):
        self.index = index
        self.queries = queries.astype(np.float32)
        self.ground_truth = ground_truth
        faiss.normalize_L2(self.queries)
    
    def find_optimal_efSearch(self, target_recall: float = 0.95,
                               k: int = 10) -> dict:
        """Find minimum efSearch for target recall"""
        
        results = []
        
        for ef in [16, 24, 32, 48, 64, 96, 128, 192, 256]:
            self.index.hnsw.efSearch = ef
            
            # Measure
            import time
            start = time.perf_counter()
            D, I = self.index.search(self.queries, k)
            elapsed = time.perf_counter() - start
            
            # Calculate recall
            recall = self._calculate_recall(I, k)
            
            results.append({
                'efSearch': ef,
                'recall': recall,
                'latency_ms': elapsed * 1000 / len(self.queries),
                'qps': len(self.queries) / elapsed
            })
            
            if recall >= target_recall:
                return {
                    'optimal_efSearch': ef,
                    **results[-1],
                    'all_results': results
                }
        
        return {
            'note': f'Target recall {target_recall} not achieved',
            'best_result': max(results, key=lambda x: x['recall']),
            'all_results': results
        }
    
    def _calculate_recall(self, indices: np.ndarray, k: int) -> float:
        """Calculate recall@k"""
        correct = 0
        for i, pred in enumerate(indices):
            gt = set(self.ground_truth[i][:k])
            found = set(pred)
            correct += len(gt.intersection(found))
        return correct / (len(indices) * k)

# HNSW vs IVF comparison
def compare_hnsw_ivf(embeddings: np.ndarray, 
                     queries: np.ndarray,
                     k: int = 10) -> dict:
    """Compare HNSW and IVF performance"""
    
    dim = embeddings.shape[1]
    n = len(embeddings)
    
    # Normalize
    embeddings = embeddings.astype(np.float32)
    queries = queries.astype(np.float32)
    faiss.normalize_L2(embeddings)
    faiss.normalize_L2(queries)
    
    # Ground truth
    flat = faiss.IndexFlatIP(dim)
    flat.add(embeddings)
    gt_D, gt_I = flat.search(queries, k)
    
    results = {}
    
    # HNSW
    import time
    hnsw = faiss.IndexHNSWFlat(dim, 32)
    hnsw.hnsw.efConstruction = 200
    hnsw.add(embeddings)
    
    hnsw.hnsw.efSearch = 64
    start = time.perf_counter()
    D, I = hnsw.search(queries, k)
    hnsw_time = time.perf_counter() - start
    
    hnsw_recall = sum(
        len(set(I[i]).intersection(set(gt_I[i]))) 
        for i in range(len(I))
    ) / (len(I) * k)
    
    results['hnsw'] = {
        'latency_ms': hnsw_time * 1000 / len(queries),
        'recall': hnsw_recall,
        'memory_mb': (n * dim * 4 + n * 32 * 4 * 2) / 1e6
    }
    
    # IVF
    nlist = int(np.sqrt(n))
    ivf = faiss.IndexIVFFlat(flat, dim, nlist)
    ivf.train(embeddings)
    ivf.add(embeddings)
    ivf.nprobe = 16
    
    start = time.perf_counter()
    D, I = ivf.search(queries, k)
    ivf_time = time.perf_counter() - start
    
    ivf_recall = sum(
        len(set(I[i]).intersection(set(gt_I[i]))) 
        for i in range(len(I))
    ) / (len(I) * k)
    
    results['ivf'] = {
        'latency_ms': ivf_time * 1000 / len(queries),
        'recall': ivf_recall,
        'memory_mb': (n * dim * 4) / 1e6
    }
    
    return results
```

**Interview Tips:**
- HNSW is best for latency-critical applications with sufficient memory
- M=32 and efConstruction=200 are good defaults
- efSearch is the main tuning knob at query time
- Graph structure cannot be updated efficiently — rebuilds needed for major changes

---

### Question 23
- [ ] Done
**When should you use approximate nearest neighbor (ANN) search versus exact search?**

**Answer:**

**Definition:**
Use **exact search** when: dataset is small (<100K vectors), 100% recall is required, or latency isn't critical. Use **ANN** when: dataset is large (>100K), 95-99% recall is acceptable, and low latency matters. ANN trades small accuracy loss for 10-1000x speedup.

**Decision Framework:**

| Dataset Size | Latency Requirement | Recommendation |
|--------------|---------------------|----------------|
| <10K | Any | Exact (fast enough) |
| 10K-100K | <100ms | Exact possible |
| 10K-100K | <10ms | ANN |
| 100K-10M | Any | ANN |
| >10M | Any | ANN (exact impractical) |

**Trade-off Analysis:**

| Aspect | Exact Search | ANN Search |
|--------|--------------|------------|
| **Recall** | 100% | 90-99.9% |
| **Latency** | O(n) | O(log n) to O(1) |
| **Memory** | O(n·d) | O(n·d) or less (with PQ) |
| **Build time** | O(n) | O(n) to O(n log n) |
| **Use case** | Ground truth, small data | Production, large scale |

**When Exact Search is Worth It:**

| Scenario | Reason |
|----------|--------|
| **Legal/compliance** | Cannot miss any match |
| **Deduplication** | Must find all duplicates |
| **Evaluation** | Need ground truth |
| **Small dataset** | No speed benefit from ANN |
| **One-time batch** | Latency not critical |

**Python Code Example:**
```python
import numpy as np
import faiss
import time
from typing import Dict, List

class SearchModeSelector:
    """Select between exact and approximate search"""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.exact_index = None
        self.ann_index = None
    
    def build_indices(self, embeddings: np.ndarray):
        """Build both indices for comparison"""
        n = len(embeddings)
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Exact index
        self.exact_index = faiss.IndexFlatIP(self.dim)
        self.exact_index.add(embeddings)
        
        # ANN index (HNSW)
        self.ann_index = faiss.IndexHNSWFlat(self.dim, 32)
        self.ann_index.hnsw.efConstruction = 200
        self.ann_index.add(embeddings)
    
    def recommend(self, n_vectors: int, 
                  target_latency_ms: float,
                  min_recall: float = 0.95) -> Dict:
        """Recommend search mode based on requirements"""
        
        # Estimate exact search latency (rough)
        # ~1μs per vector for brute force on modern CPU
        exact_latency_ms = n_vectors * 0.001
        
        # Estimate ANN latency
        # HNSW: ~0.1-1ms regardless of size (log complexity)
        ann_latency_ms = 1.0
        
        if exact_latency_ms <= target_latency_ms:
            return {
                'recommendation': 'exact',
                'reason': 'Latency target achievable with exact search',
                'estimated_latency_ms': exact_latency_ms,
                'recall': 1.0
            }
        elif min_recall > 0.999:
            return {
                'recommendation': 'exact',
                'reason': 'Recall requirement too high for ANN',
                'estimated_latency_ms': exact_latency_ms,
                'recall': 1.0,
                'warning': 'Will exceed latency target'
            }
        else:
            return {
                'recommendation': 'ann',
                'reason': 'ANN provides acceptable recall with lower latency',
                'estimated_latency_ms': ann_latency_ms,
                'estimated_recall': 0.95
            }
    
    def compare_methods(self, query: np.ndarray, k: int = 10) -> Dict:
        """Compare exact vs ANN on actual query"""
        query = query.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        
        # Exact search
        start = time.perf_counter()
        exact_D, exact_I = self.exact_index.search(query, k)
        exact_time = (time.perf_counter() - start) * 1000
        
        # ANN search
        self.ann_index.hnsw.efSearch = 64
        start = time.perf_counter()
        ann_D, ann_I = self.ann_index.search(query, k)
        ann_time = (time.perf_counter() - start) * 1000
        
        # Calculate recall
        exact_set = set(exact_I[0])
        ann_set = set(ann_I[0])
        recall = len(exact_set.intersection(ann_set)) / k
        
        return {
            'exact_latency_ms': exact_time,
            'ann_latency_ms': ann_time,
            'speedup': exact_time / ann_time,
            'recall': recall,
            'exact_results': exact_I[0].tolist(),
            'ann_results': ann_I[0].tolist()
        }

# Hybrid approach: ANN then exact verification
class HybridSearch:
    """Use ANN for candidates, exact for final ranking"""
    
    def __init__(self, embeddings: np.ndarray, dim: int):
        self.embeddings = embeddings.astype(np.float32)
        self.dim = dim
        
        faiss.normalize_L2(self.embeddings)
        
        # ANN for fast candidate retrieval
        self.ann_index = faiss.IndexHNSWFlat(dim, 32)
        self.ann_index.add(self.embeddings)
    
    def search(self, query: np.ndarray, k: int = 10,
               candidate_multiplier: int = 5) -> List[int]:
        """Two-stage search: ANN candidates → exact reranking"""
        
        query = query.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        
        # Stage 1: Get more candidates via ANN
        n_candidates = k * candidate_multiplier
        _, candidate_ids = self.ann_index.search(query, n_candidates)
        candidate_ids = candidate_ids[0]
        
        # Stage 2: Exact scoring of candidates
        candidates = self.embeddings[candidate_ids]
        exact_scores = np.dot(candidates, query.T).flatten()
        
        # Rerank and return top-k
        sorted_indices = np.argsort(-exact_scores)[:k]
        return candidate_ids[sorted_indices].tolist()

# Decision helper for production
def production_search_decision(
    n_vectors: int,
    queries_per_second: float,
    latency_p99_target_ms: float,
    recall_requirement: float,
    available_memory_gb: float
) -> Dict:
    """Production decision framework"""
    
    decision = {
        'n_vectors': n_vectors,
        'target_qps': queries_per_second,
        'target_latency_ms': latency_p99_target_ms,
        'target_recall': recall_requirement
    }
    
    # Memory check for exact search
    dim = 768  # Assume BERT-sized
    exact_memory_gb = (n_vectors * dim * 4) / 1e9
    
    if exact_memory_gb > available_memory_gb:
        decision['exact_feasible'] = False
        decision['reason'] = f"Exact needs {exact_memory_gb:.1f}GB, only {available_memory_gb}GB available"
        decision['recommendation'] = 'ann_with_pq'
    elif n_vectors < 100_000 and latency_p99_target_ms > 50:
        decision['exact_feasible'] = True
        decision['recommendation'] = 'exact'
        decision['reason'] = 'Small enough for exact search'
    elif recall_requirement > 0.999:
        decision['exact_feasible'] = True
        decision['recommendation'] = 'exact_with_warning'
        decision['reason'] = 'Recall requirement too high for ANN'
    else:
        decision['recommendation'] = 'ann_hnsw'
        decision['reason'] = 'Best speed/recall trade-off'
    
    return decision
```

**Interview Tips:**
- Rule of thumb: ANN at >100K vectors, exact below
- Always measure recall on your actual queries, not synthetic
- Hybrid (ANN candidates + exact rerank) gives best of both
- For critical applications, consider ANN + verification step

---

### Question 24
- [ ] Done
**How do you implement hybrid search combining vector similarity with traditional keyword (BM25) search?**

**Answer:**

**Definition:**
Hybrid search combines **dense vectors** (semantic similarity) with **sparse/keyword search** (BM25/TF-IDF) to capture both semantic meaning and exact term matching. Approaches: **score fusion** (combine normalized scores), **reciprocal rank fusion (RRF)**, or **two-stage** (retrieve then rerank). Improves recall for queries with important keywords.

**Hybrid Search Benefits:**

| Query Type | Vector Only | Keyword Only | Hybrid |
|------------|-------------|--------------|--------|
| **Semantic** | ✓ Good | ✗ Poor | ✓ Good |
| **Exact match** | ✗ May miss | ✓ Good | ✓ Good |
| **Rare terms** | ✗ Poor | ✓ Good | ✓ Good |
| **Typos** | ✓ Robust | ✗ Fails | ✓ Robust |
| **Named entities** | Variable | ✓ Good | ✓ Good |

**Fusion Methods:**

| Method | Formula | Pros | Cons |
|--------|---------|------|------|
| **Linear** | α·vec + (1-α)·bm25 | Simple | Requires tuning α |
| **RRF** | Σ 1/(k + rank) | No tuning | Ignores scores |
| **Convex** | softmax(scores) | Normalized | Complex |
| **Cascade** | BM25 → rerank with vec | Fast | Two stages |

**Python Code Example:**
```python
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import math

class BM25:
    """BM25 keyword search implementation"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = defaultdict(int)
        self.term_freqs = []  # List of dicts
        self.n_docs = 0
    
    def fit(self, documents: List[str]):
        """Index documents for BM25"""
        self.n_docs = len(documents)
        
        for doc in documents:
            terms = doc.lower().split()
            self.doc_lengths.append(len(terms))
            
            # Term frequencies in this doc
            tf = defaultdict(int)
            for term in terms:
                tf[term] += 1
            self.term_freqs.append(dict(tf))
            
            # Document frequencies
            for term in set(terms):
                self.doc_freqs[term] += 1
        
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
    
    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Search with BM25 scoring"""
        query_terms = query.lower().split()
        scores = []
        
        for doc_id in range(self.n_docs):
            score = 0.0
            doc_len = self.doc_lengths[doc_id]
            
            for term in query_terms:
                if term not in self.term_freqs[doc_id]:
                    continue
                
                tf = self.term_freqs[doc_id][term]
                df = self.doc_freqs.get(term, 0)
                
                # IDF
                idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
                
                # TF with length normalization
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                )
                
                score += idf * tf_norm
            
            scores.append((doc_id, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

class HybridSearch:
    """Combine vector and keyword search"""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.bm25 = BM25()
        self.vector_index = None
        self.documents = []
    
    def index(self, documents: List[str], embeddings: np.ndarray):
        """Index documents for hybrid search"""
        import faiss
        
        self.documents = documents
        
        # BM25 index
        self.bm25.fit(documents)
        
        # Vector index
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        self.embeddings = embeddings
        
        self.vector_index = faiss.IndexFlatIP(self.dim)
        self.vector_index.add(embeddings)
    
    def search_hybrid(self, query: str, query_embedding: np.ndarray,
                      k: int = 10, alpha: float = 0.5,
                      method: str = 'linear') -> List[Tuple[int, float]]:
        """Hybrid search with score fusion"""
        import faiss
        
        # Vector search
        query_emb = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_emb)
        vec_scores, vec_ids = self.vector_index.search(query_emb, k * 2)
        
        # BM25 search
        bm25_results = self.bm25.search(query, k * 2)
        
        if method == 'linear':
            return self._linear_fusion(vec_ids[0], vec_scores[0],
                                       bm25_results, alpha, k)
        elif method == 'rrf':
            return self._rrf_fusion(vec_ids[0], bm25_results, k)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _linear_fusion(self, vec_ids: np.ndarray, vec_scores: np.ndarray,
                       bm25_results: List[Tuple], alpha: float,
                       k: int) -> List[Tuple[int, float]]:
        """Linear score combination"""
        
        # Normalize scores to [0, 1]
        vec_score_dict = {}
        if len(vec_scores) > 0:
            vec_min, vec_max = vec_scores.min(), vec_scores.max()
            vec_range = vec_max - vec_min + 1e-8
            for doc_id, score in zip(vec_ids, vec_scores):
                vec_score_dict[doc_id] = (score - vec_min) / vec_range
        
        bm25_score_dict = {}
        if bm25_results:
            bm25_scores = [s for _, s in bm25_results]
            bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)
            bm25_range = bm25_max - bm25_min + 1e-8
            for doc_id, score in bm25_results:
                bm25_score_dict[doc_id] = (score - bm25_min) / bm25_range
        
        # Combine scores
        all_docs = set(vec_score_dict.keys()) | set(bm25_score_dict.keys())
        combined = []
        
        for doc_id in all_docs:
            vec_s = vec_score_dict.get(doc_id, 0)
            bm25_s = bm25_score_dict.get(doc_id, 0)
            combined_score = alpha * vec_s + (1 - alpha) * bm25_s
            combined.append((doc_id, combined_score))
        
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:k]
    
    def _rrf_fusion(self, vec_ids: np.ndarray,
                    bm25_results: List[Tuple],
                    k: int, rrf_k: int = 60) -> List[Tuple[int, float]]:
        """Reciprocal Rank Fusion"""
        
        scores = defaultdict(float)
        
        # Vector rankings
        for rank, doc_id in enumerate(vec_ids):
            scores[doc_id] += 1.0 / (rrf_k + rank + 1)
        
        # BM25 rankings
        for rank, (doc_id, _) in enumerate(bm25_results):
            scores[doc_id] += 1.0 / (rrf_k + rank + 1)
        
        # Sort by combined RRF score
        results = [(doc_id, score) for doc_id, score in scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]

# Weaviate-style hybrid
class WeaviateStyleHybrid:
    """Weaviate-style hybrid with single alpha parameter"""
    
    def search(self, query: str, query_embedding: np.ndarray,
               alpha: float = 0.5) -> List[Dict]:
        """
        alpha = 1: pure vector
        alpha = 0: pure BM25
        alpha = 0.5: equal weight
        """
        # In Weaviate, this is built-in:
        # client.query.get("Document", ["content"])
        #     .with_hybrid(query=query, alpha=alpha)
        #     .do()
        pass

# Usage example
def hybrid_search_example():
    """Example of hybrid search setup"""
    
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with many layers",
        "Natural language processing enables computers to understand text",
        "Computer vision allows machines to interpret images"
    ]
    
    # Assume we have embeddings
    embeddings = np.random.randn(len(documents), 768).astype(np.float32)
    
    hybrid = HybridSearch(dim=768)
    hybrid.index(documents, embeddings)
    
    # Search
    query = "neural network deep learning"
    query_embedding = np.random.randn(768)
    
    # Try different alpha values
    for alpha in [0.0, 0.5, 1.0]:
        results = hybrid.search_hybrid(
            query, query_embedding, 
            k=3, alpha=alpha, method='linear'
        )
        print(f"Alpha={alpha}: {results}")
```

**Interview Tips:**
- α=0.5-0.7 is a good starting point for most applications
- RRF is more robust than linear (no score normalization needed)
- Keyword search is critical for rare terms and exact matches
- Production systems (Elasticsearch, Weaviate) have built-in hybrid

---

## Vector Database Operations

### Question 25
- [ ] Done
**How do you handle incremental updates and real-time indexing in production vector databases?**

**Answer:**

**Definition:**
Handle real-time updates via: **write-ahead log** (durability), **in-memory buffer** (fast writes), **background merge** (combine with main index), **versioned indices** (consistency), and **delta indices** (incremental). Challenge: ANN indices (HNSW) are expensive to update; need strategies to minimize rebuilds.

**Update Strategies:**

| Strategy | Latency | Consistency | Complexity |
|----------|---------|-------------|------------|
| **Buffer + merge** | Low | Eventual | Medium |
| **Delta index** | Low | Near real-time | Low |
| **Full rebuild** | High | Strong | Low |
| **Versioned** | Medium | Snapshot | High |
| **Streaming** | Very low | Eventual | High |

**Architecture Pattern:**

```
Writes → Write Buffer → Delta Index ←→ Search (queries both)
              ↓
         Background Merge
              ↓
         Main Index
```

**Python Code Example:**
```python
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
import threading
from collections import deque
import time
from datetime import datetime

class IncrementalVectorIndex:
    """Production-ready incremental vector indexing"""
    
    def __init__(self, dim: int, merge_threshold: int = 10000):
        self.dim = dim
        self.merge_threshold = merge_threshold
        
        # Main index (HNSW for fast search)
        self.main_index = faiss.IndexHNSWFlat(dim, 32)
        self.main_index.hnsw.efConstruction = 200
        self.main_count = 0
        
        # Delta index (Flat for fast inserts)
        self.delta_index = faiss.IndexFlatIP(dim)
        self.delta_ids = []
        
        # ID mappings
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.deleted_ids = set()
        
        # Write buffer
        self.write_buffer = deque()
        self.buffer_lock = threading.Lock()
        
        # Background merger
        self.merge_lock = threading.Lock()
        self.merge_in_progress = False
    
    def add(self, embedding: np.ndarray, doc_id: str, 
            metadata: Dict = None) -> bool:
        """Add vector to index (real-time)"""
        
        embedding = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(embedding)
        
        with self.buffer_lock:
            self.write_buffer.append({
                'embedding': embedding,
                'doc_id': doc_id,
                'metadata': metadata,
                'timestamp': datetime.now()
            })
        
        # Flush buffer if needed
        if len(self.write_buffer) >= 100:
            self._flush_buffer()
        
        # Trigger merge if delta too large
        if len(self.delta_ids) >= self.merge_threshold:
            self._background_merge()
        
        return True
    
    def _flush_buffer(self):
        """Flush write buffer to delta index"""
        with self.buffer_lock:
            if not self.write_buffer:
                return
            
            embeddings = []
            ids = []
            
            while self.write_buffer:
                item = self.write_buffer.popleft()
                embeddings.append(item['embedding'])
                ids.append(item['doc_id'])
            
            if embeddings:
                stacked = np.vstack(embeddings)
                self.delta_index.add(stacked)
                
                for doc_id in ids:
                    idx = len(self.delta_ids)
                    self.delta_ids.append(doc_id)
                    self.id_to_idx[doc_id] = ('delta', idx)
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search both main and delta indices"""
        
        # Flush any pending writes
        self._flush_buffer()
        
        query = query.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        
        results = []
        
        # Search main index
        if self.main_count > 0:
            distances, indices = self.main_index.search(query, k)
            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0:
                    doc_id = self.idx_to_id.get(('main', idx))
                    if doc_id and doc_id not in self.deleted_ids:
                        results.append((doc_id, float(dist)))
        
        # Search delta index
        if self.delta_index.ntotal > 0:
            distances, indices = self.delta_index.search(query, k)
            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0 and idx < len(self.delta_ids):
                    doc_id = self.delta_ids[idx]
                    if doc_id not in self.deleted_ids:
                        results.append((doc_id, float(dist)))
        
        # Merge and sort results
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Deduplicate
        seen = set()
        unique_results = []
        for doc_id, score in results:
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append((doc_id, score))
        
        return unique_results[:k]
    
    def delete(self, doc_id: str) -> bool:
        """Mark document as deleted"""
        if doc_id in self.id_to_idx:
            self.deleted_ids.add(doc_id)
            return True
        return False
    
    def update(self, doc_id: str, embedding: np.ndarray) -> bool:
        """Update = delete + add"""
        self.delete(doc_id)
        return self.add(embedding, doc_id)
    
    def _background_merge(self):
        """Merge delta into main index in background"""
        if self.merge_in_progress:
            return
        
        def merge():
            self.merge_in_progress = True
            
            with self.merge_lock:
                # Get all vectors (main + delta)
                all_embeddings = []
                all_ids = []
                
                # From main (excluding deleted)
                for idx in range(self.main_count):
                    doc_id = self.idx_to_id.get(('main', idx))
                    if doc_id and doc_id not in self.deleted_ids:
                        # Would need to store embeddings or reconstruct
                        pass
                
                # From delta (excluding deleted)
                for idx, doc_id in enumerate(self.delta_ids):
                    if doc_id not in self.deleted_ids:
                        # Reconstruct from delta index
                        pass
                
                # Rebuild main index
                # In practice, use FAISS's merge functions
                
            self.merge_in_progress = False
        
        thread = threading.Thread(target=merge)
        thread.start()

class StreamingIndexer:
    """Handle streaming updates efficiently"""
    
    def __init__(self, dim: int, batch_size: int = 100,
                 flush_interval_s: float = 1.0):
        self.dim = dim
        self.batch_size = batch_size
        self.flush_interval = flush_interval_s
        
        self.buffer = []
        self.last_flush = time.time()
        self.index = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(dim), dim, 100
        )
        self.trained = False
    
    def process_event(self, event: Dict):
        """Process streaming event"""
        self.buffer.append(event)
        
        should_flush = (
            len(self.buffer) >= self.batch_size or
            time.time() - self.last_flush > self.flush_interval
        )
        
        if should_flush:
            self._flush()
    
    def _flush(self):
        """Flush buffer to index"""
        if not self.buffer:
            return
        
        embeddings = np.vstack([e['embedding'] for e in self.buffer])
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Train if needed
        if not self.trained and len(embeddings) >= 1000:
            self.index.train(embeddings)
            self.trained = True
        
        # Add to index
        if self.trained:
            self.index.add(embeddings)
        
        self.buffer = []
        self.last_flush = time.time()

# Version-controlled index
class VersionedIndex:
    """Support point-in-time queries"""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.versions = {}  # version_id -> index
        self.current_version = 0
    
    def create_snapshot(self) -> int:
        """Create new version snapshot"""
        self.current_version += 1
        # Clone current index
        # In practice, use copy-on-write
        return self.current_version
    
    def search_at_version(self, query: np.ndarray, version: int,
                          k: int = 10) -> List[Tuple[str, float]]:
        """Search at specific version"""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        return self.versions[version].search(query, k)
```

**Interview Tips:**
- Delta index pattern is common in production systems
- HNSW is hard to update — use IVF for write-heavy workloads
- Background merging prevents write blocking
- Consider write-ahead log for durability guarantees

---

### Question 26
- [ ] Done
**What strategies help optimize memory usage and disk I/O for large vector datasets?**

**Answer:**

**Definition:**
Optimize memory/disk via: **memory-mapped files** (load on demand), **Product Quantization** (32x compression), **tiered storage** (hot vectors in RAM, cold on disk), **lazy loading** (load indices progressively), and **index sharding** (distribute across nodes).

**Memory Optimization Techniques:**

| Technique | Memory Reduction | I/O Impact | Quality |
|-----------|------------------|------------|---------|
| **Memory mapping** | Load on demand | Higher I/O | 100% |
| **PQ (m=32)** | 32x | Lower | 85-95% |
| **SQ (8-bit)** | 4x | Lower | 98%+ |
| **On-disk index** | ~0 RAM | High I/O | 100% |
| **Tiered storage** | Variable | Balanced | 100% |

**Python Code Example:**
```python
import numpy as np
import faiss
import os
import mmap

class MemoryOptimizedIndex:
    """Memory-efficient vector storage"""
    
    def __init__(self, dim: int, storage_path: str):
        self.dim = dim
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def create_pq_index(self, embeddings: np.ndarray, 
                        m: int = 32, nbits: int = 8):
        """Create memory-efficient PQ index"""
        n = len(embeddings)
        nlist = int(np.sqrt(n))
        
        # IVF + PQ for maximum compression
        quantizer = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIVFPQ(
            quantizer, self.dim, nlist, m, nbits
        )
        
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        self.index.train(embeddings)
        self.index.add(embeddings)
        
        # Save to disk
        faiss.write_index(self.index, 
                         f"{self.storage_path}/index.faiss")
    
    def create_ondisk_index(self, embeddings: np.ndarray):
        """Create on-disk index with minimal RAM"""
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Write raw vectors to disk
        embeddings.tofile(f"{self.storage_path}/vectors.bin")
        
        # Create memory-mapped index
        # FAISS supports OnDiskInvertedLists for IVF
    
    def load_mmap(self):
        """Load index with memory mapping"""
        self.index = faiss.read_index(
            f"{self.storage_path}/index.faiss",
            faiss.IO_FLAG_MMAP
        )

class TieredStorage:
    """Hot/warm/cold tiered storage"""
    
    def __init__(self, dim: int, hot_size: int = 100000):
        self.dim = dim
        self.hot_size = hot_size
        
        # Hot tier: frequently accessed, in RAM
        self.hot_index = faiss.IndexFlatIP(dim)
        self.hot_ids = []
        
        # Cold tier: disk-based
        self.cold_index = None
    
    def promote_to_hot(self, doc_ids: list, embeddings: np.ndarray):
        """Promote vectors to hot tier"""
        if self.hot_index.ntotal >= self.hot_size:
            self._evict_cold()
        
        self.hot_index.add(embeddings)
        self.hot_ids.extend(doc_ids)
    
    def _evict_cold(self):
        """Evict least-used to cold tier"""
        # Based on access patterns
        pass
```

**Interview Tips:**
- Memory-mapped files ideal for large static indices
- PQ essential for billion-scale deployments
- Tiered storage matches access patterns
- On-disk indices trade latency for capacity

---

### Question 27
- [ ] Done
**How do you implement vector compression and Product Quantization (PQ) to balance storage and accuracy?**

**Answer:**

**Definition:**
Product Quantization divides vectors into **m subvectors**, learns **codebooks** for each subspace, and stores **code indices** instead of floats. Compression: 768-dim float32 (3KB) → PQ32x8 (32 bytes) = **96x compression**. Accuracy loss: 5-15% recall depending on configuration.

**PQ Configuration:**

| Parameter | Description | Trade-off |
|-----------|-------------|-----------|
| **m** | Number of subquantizers | Higher = better quality, more codes |
| **nbits** | Bits per subquantizer | 8 = 256 centroids, 4 = 16 centroids |
| **nlist** (IVF) | Coarse quantizer clusters | More = faster search, more training |

**Python Code Example:**
```python
import numpy as np
import faiss

class ProductQuantizationManager:
    """Manage PQ compression for vectors"""
    
    def __init__(self, dim: int, m: int = 32, nbits: int = 8):
        self.dim = dim
        self.m = m
        self.nbits = nbits
        self.pq = faiss.ProductQuantizer(dim, m, nbits)
    
    def train(self, embeddings: np.ndarray):
        """Train PQ codebooks"""
        embeddings = embeddings.astype(np.float32)
        self.pq.train(embeddings)
    
    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """Compress vectors to codes"""
        embeddings = embeddings.astype(np.float32)
        return self.pq.compute_codes(embeddings)
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decompress codes to vectors"""
        return self.pq.decode(codes)
    
    def asymmetric_distance(self, query: np.ndarray, 
                            codes: np.ndarray) -> np.ndarray:
        """Fast distance computation using ADC"""
        # Compute distance tables
        distances = self.pq.compute_distance_tables(query.reshape(1, -1))
        
        # Look up distances from tables
        n = len(codes)
        result = np.zeros(n, dtype=np.float32)
        
        for i in range(n):
            for j in range(self.m):
                result[i] += distances[0, j, codes[i, j]]
        
        return result
    
    def get_compression_stats(self) -> dict:
        """Report compression statistics"""
        original_bytes = self.dim * 4  # float32
        compressed_bytes = self.m * (self.nbits // 8)
        
        return {
            'original_bytes_per_vector': original_bytes,
            'compressed_bytes_per_vector': compressed_bytes,
            'compression_ratio': original_bytes / compressed_bytes,
            'num_subquantizers': self.m,
            'centroids_per_subquantizer': 2 ** self.nbits
        }

# OPQ: Optimized Product Quantization
def create_opq_index(dim: int, n_vectors: int, m: int = 32):
    """OPQ rotates space for better quantization"""
    nlist = min(int(np.sqrt(n_vectors)), 4096)
    
    # OPQ_M,IVF_nlist,PQ_m
    index_string = f"OPQ{m},IVF{nlist},PQ{m}x8"
    index = faiss.index_factory(dim, index_string)
    
    return index
```

**Interview Tips:**
- m=32-64 is typical for 768-dim embeddings
- OPQ (rotation) improves PQ quality by 5-10%
- Always evaluate recall impact on your queries
- PQ is essential for billion-scale deployments

---

### Question 28
- [ ] Done
**How do you handle data partitioning and sharding strategies for large-scale vector collections?**

**Answer:**

**Definition:**
Shard vectors via: **hash partitioning** (uniform distribution), **semantic clustering** (similar vectors together), **range partitioning** (by ID/time), or **geographic** (data locality). Goal: even load distribution while minimizing cross-shard queries.

**Sharding Strategies:**

| Strategy | Distribution | Query Pattern | Use Case |
|----------|--------------|---------------|----------|
| **Hash** | Even | Query all shards | General purpose |
| **Semantic** | By cluster | Query subset | Topic-focused |
| **Range** | By ID/time | Range queries | Time-series |
| **Geographic** | By location | Local queries | Multi-region |

**Python Code Example:**
```python
import numpy as np
from typing import List, Dict
import hashlib

class ShardedVectorIndex:
    def __init__(self, num_shards: int, dim: int):
        self.num_shards = num_shards
        self.shards = [{} for _ in range(num_shards)]
    
    def _get_shard_hash(self, doc_id: str) -> int:
        """Hash-based shard assignment"""
        return int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % self.num_shards
    
    def add(self, doc_id: str, embedding: np.ndarray):
        shard_id = self._get_shard_hash(doc_id)
        self.shards[shard_id][doc_id] = embedding
    
    def search(self, query: np.ndarray, k: int = 10) -> List:
        # Query all shards, merge results
        all_results = []
        for shard in self.shards:
            # Search each shard
            results = self._search_shard(query, shard, k)
            all_results.extend(results)
        
        # Merge and return top-k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:k]
```

**Interview Tips:**
- Hash sharding simplest for uniform distribution
- Semantic sharding reduces cross-shard queries for topic searches
- Consider replication factor for fault tolerance
- Load balancing critical for hot shards

---

### Question 29
- [ ] Done
**How do you handle multi-tenancy and data isolation requirements in vector database deployments?**

**Answer:**

**Definition:**
Multi-tenancy isolates tenant data via: **namespace separation** (logical partition), **collection per tenant** (physical isolation), **metadata filtering** (soft isolation), or **dedicated clusters** (complete isolation). Trade-off: resource efficiency vs isolation strength.

**Isolation Levels:**

| Level | Method | Isolation | Efficiency |
|-------|--------|-----------|------------|
| **Soft** | Metadata filter | Low | High |
| **Namespace** | Logical partition | Medium | Medium |
| **Collection** | Separate index | High | Low |
| **Cluster** | Dedicated infra | Complete | Very Low |

**Python Code Example:**
```python
class MultiTenantVectorDB:
    def __init__(self):
        self.tenant_collections = {}
    
    def add(self, tenant_id: str, doc_id: str, embedding, metadata=None):
        if tenant_id not in self.tenant_collections:
            self.tenant_collections[tenant_id] = []
        
        # Include tenant_id in metadata for filtering
        if metadata is None:
            metadata = {}
        metadata['tenant_id'] = tenant_id
        
        self.tenant_collections[tenant_id].append({
            'id': doc_id, 'embedding': embedding, 'metadata': metadata
        })
    
    def search(self, tenant_id: str, query, k: int = 10):
        # Only search tenant's collection
        if tenant_id not in self.tenant_collections:
            return []
        
        collection = self.tenant_collections[tenant_id]
        # Perform search within tenant's data only
        return self._search_collection(query, collection, k)
```

**Interview Tips:**
- Metadata filtering is fastest but leakiest
- Collection per tenant for compliance requirements
- Consider noisy neighbor effects on shared resources
- Monitor per-tenant usage for capacity planning

---

### Question 30
- [ ] Done
**What techniques help ensure vector database performance consistency under varying load?**

**Answer:**

**Definition:**
Ensure performance consistency via: **load balancing** (distribute queries), **auto-scaling** (add/remove nodes), **rate limiting** (prevent overload), **query queuing** (smooth bursts), **resource isolation** (prevent noisy neighbors), and **caching** (reduce computation).

**Performance Techniques:**

| Technique | Purpose | Implementation |
|-----------|---------|----------------|
| **Load balancing** | Even distribution | Round-robin, least-connections |
| **Auto-scaling** | Handle load spikes | Horizontal pod autoscaler |
| **Rate limiting** | Prevent overload | Token bucket, leaky bucket |
| **Query caching** | Reduce repeat work | LRU cache, Redis |
| **Connection pooling** | Reduce overhead | Pool management |

**Python Code Example:**
```python
import time
from collections import deque
import threading

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = deque()
        self.lock = threading.Lock()
    
    def allow(self) -> bool:
        with self.lock:
            now = time.time()
            # Remove old requests
            while self.requests and self.requests[0] < now - self.window:
                self.requests.popleft()
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False

class LoadBalancer:
    def __init__(self, endpoints: list):
        self.endpoints = endpoints
        self.current = 0
    
    def get_endpoint(self) -> str:
        endpoint = self.endpoints[self.current]
        self.current = (self.current + 1) % len(self.endpoints)
        return endpoint
```

**Interview Tips:**
- P99 latency matters more than average
- Cache frequently queried embeddings
- Horizontal scaling for read-heavy workloads
- Monitor and alert on latency percentiles

---

## Production & Scaling

### Question 31
- [ ] Done
**How do you implement effective backup and disaster recovery for vector databases?**

**Answer:**

**Definition:**
Backup strategies include: **full snapshots** (periodic complete backup), **incremental backups** (changes only), **replication** (real-time copies), **WAL archiving** (transaction logs). DR requires: **RTO/RPO targets**, **cross-region replication**, and **tested recovery procedures**.

**Backup Strategies:**

| Strategy | Recovery Time | Data Loss | Storage Cost |
|----------|---------------|-----------|-------------|
| **Full snapshot** | Medium | Up to interval | High |
| **Incremental** | Long | Low | Medium |
| **Sync replication** | Instant | Zero | 2x |
| **Async replication** | Fast | Minimal | 2x |

**Python Code Example:**
```python
import faiss
import shutil
import os
from datetime import datetime

class VectorDBBackupManager:
    def __init__(self, db_path: str, backup_path: str):
        self.db_path = db_path
        self.backup_path = backup_path
        os.makedirs(backup_path, exist_ok=True)
    
    def create_snapshot(self, index: faiss.Index) -> str:
        """Create full snapshot"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        snapshot_path = f"{self.backup_path}/snapshot_{timestamp}"
        os.makedirs(snapshot_path)
        
        # Save FAISS index
        faiss.write_index(index, f"{snapshot_path}/index.faiss")
        
        # Save metadata, mappings
        return snapshot_path
    
    def restore_from_snapshot(self, snapshot_path: str) -> faiss.Index:
        """Restore from snapshot"""
        return faiss.read_index(f"{snapshot_path}/index.faiss")
```

**Interview Tips:**
- Define RTO (recovery time) and RPO (data loss tolerance)
- Sync replication for zero data loss, async for performance
- Test recovery procedures regularly
- Consider geo-redundancy for critical systems

---

### Question 32
- [ ] Done
**How do you implement load balancing and horizontal scaling for vector database clusters?**

**Answer:**

**Definition:**
Scale via: **sharding** (partition data), **replication** (read replicas), **query routing** (smart load distribution). Load balancing strategies: **round-robin**, **least-connections**, **latency-based**, or **consistent hashing** (for cache affinity).

**Scaling Strategies:**

| Strategy | Benefit | Use Case |
|----------|---------|----------|
| **Read replicas** | Read throughput | Read-heavy |
| **Sharding** | Data capacity | Large datasets |
| **Query routing** | Even distribution | Mixed workload |
| **Auto-scaling** | Handle spikes | Variable load |

**Python Code Example:**
```python
from typing import List
import random
import time

class VectorDBCluster:
    def __init__(self, num_shards: int, replicas_per_shard: int):
        self.num_shards = num_shards
        self.shards = [
            [f"shard{s}_replica{r}" for r in range(replicas_per_shard)]
            for s in range(num_shards)
        ]
        self.replica_latencies = {}  # Track latencies
    
    def route_query(self, query_embedding, shard_id: int) -> str:
        """Select best replica based on latency"""
        replicas = self.shards[shard_id]
        
        # Latency-based selection
        best = min(replicas, 
                   key=lambda r: self.replica_latencies.get(r, 0))
        return best
    
    def search(self, query, k: int = 10) -> List:
        """Search across all shards"""
        all_results = []
        
        for shard_id in range(self.num_shards):
            replica = self.route_query(query, shard_id)
            results = self._query_replica(replica, query, k)
            all_results.extend(results)
        
        # Merge and return top-k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:k]
```

**Interview Tips:**
- Read replicas are easiest way to scale reads
- Consistent hashing for stateful routing
- Monitor per-node latency for load balancing
- Auto-scaling based on queue depth, not just CPU

---

### Question 33
- [ ] Done
**What are the best practices for monitoring and alerting in vector database production environments?**

**Answer:**

**Definition:**
Monitor: **latency percentiles** (P50, P95, P99), **throughput** (QPS), **recall accuracy**, **resource usage** (CPU, memory, disk), **index health**, and **error rates**. Alert on SLO violations and capacity thresholds.

**Key Metrics:**

| Category | Metrics | Alert Threshold |
|----------|---------|----------------|
| **Latency** | P50, P95, P99 | >2x baseline |
| **Throughput** | QPS, success rate | <80% capacity |
| **Resources** | CPU, memory, disk | >80% usage |
| **Quality** | Recall (sampled) | <95% target |
| **Errors** | Timeouts, failures | >1% rate |

**Python Code Example:**
```python
import time
from dataclasses import dataclass
from collections import deque
import numpy as np

@dataclass
class QueryMetrics:
    timestamp: float
    latency_ms: float
    success: bool
    result_count: int

class VectorDBMonitor:
    def __init__(self, window_size: int = 1000):
        self.metrics = deque(maxlen=window_size)
    
    def record_query(self, latency_ms: float, success: bool, count: int):
        self.metrics.append(QueryMetrics(
            time.time(), latency_ms, success, count
        ))
    
    def get_stats(self) -> dict:
        if not self.metrics:
            return {}
        
        latencies = [m.latency_ms for m in self.metrics]
        return {
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'success_rate': sum(m.success for m in self.metrics) / len(self.metrics),
            'qps': len(self.metrics) / (self.metrics[-1].timestamp - self.metrics[0].timestamp + 0.001)
        }
    
    def check_alerts(self, thresholds: dict) -> list:
        stats = self.get_stats()
        alerts = []
        
        if stats.get('p99_ms', 0) > thresholds.get('p99_ms', 100):
            alerts.append(f"P99 latency {stats['p99_ms']:.1f}ms exceeds threshold")
        if stats.get('success_rate', 1) < thresholds.get('min_success_rate', 0.99):
            alerts.append(f"Success rate {stats['success_rate']:.2%} below threshold")
        
        return alerts
```

**Interview Tips:**
- P99 latency more important than average
- Sample recall periodically against ground truth
- Set up dashboards for real-time visibility
- Alert on trends, not just thresholds

---

### Question 34
- [ ] Done
**How do you implement effective caching strategies to improve vector search response times?**

**Answer:**

**Definition:**
Cache: **query embeddings** (avoid recomputation), **search results** (repeat queries), **popular vectors** (hot data in RAM). Strategies: **LRU** (recency), **LFU** (frequency), **TTL-based** (freshness). Cache hit rates of 60-80% typical for production.

**Caching Layers:**

| Layer | What to Cache | Benefit |
|-------|---------------|--------|
| **Query embedding** | Embedding of query text | Skip embedding computation |
| **Result cache** | Top-k results | Skip search entirely |
| **Vector cache** | Frequently accessed vectors | Faster retrieval |
| **Index cache** | Hot index segments | Lower I/O |

**Python Code Example:**
```python
import hashlib
from collections import OrderedDict
import time

class LRUCache:
    def __init__(self, capacity: int, ttl_seconds: float = 3600):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
    
    def get(self, key: str):
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl:
            del self.cache[key]
            del self.timestamps[key]
            return None
        
        # Move to end (most recent)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                oldest = next(iter(self.cache))
                del self.cache[oldest]
                del self.timestamps[oldest]
        
        self.cache[key] = value
        self.timestamps[key] = time.time()

class CachedVectorSearch:
    def __init__(self, index, embedder, cache_size: int = 10000):
        self.index = index
        self.embedder = embedder
        self.result_cache = LRUCache(cache_size)
    
    def search(self, query_text: str, k: int = 10):
        cache_key = hashlib.md5(f"{query_text}:{k}".encode()).hexdigest()
        
        # Check cache
        cached = self.result_cache.get(cache_key)
        if cached:
            return cached
        
        # Compute and search
        embedding = self.embedder.embed(query_text)
        results = self.index.search(embedding, k)
        
        # Cache results
        self.result_cache.put(cache_key, results)
        return results
```

**Interview Tips:**
- Query caching highest ROI for repeat queries
- Consider cache invalidation on index updates
- Use Redis for distributed caching
- Monitor cache hit rate as key metric

---

### Question 35
- [ ] Done
**How do you optimize batch operations versus real-time queries in vector databases?**

**Answer:**

**Definition:**
**Batch**: optimize for throughput (large inserts, bulk queries). **Real-time**: optimize for latency (single queries, immediate indexing). Use different configurations: batch uses larger buffers and parallelism; real-time uses smaller batches and streaming.

**Optimization Comparison:**

| Aspect | Batch | Real-time |
|--------|-------|-----------|
| **Priority** | Throughput | Latency |
| **Batch size** | Large (10K+) | Small (1-100) |
| **Indexing** | Offline rebuild | Delta index |
| **Resources** | Maximize usage | Reserve headroom |
| **Parallelism** | High | Controlled |

**Python Code Example:**
```python
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor

class OptimizedVectorOps:
    def __init__(self, index, dim: int):
        self.index = index
        self.dim = dim
    
    def batch_add(self, embeddings: np.ndarray, batch_size: int = 10000):
        """Optimized batch insertion"""
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Add in chunks for memory efficiency
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i+batch_size]
            self.index.add(batch)
    
    def batch_search(self, queries: np.ndarray, k: int = 10,
                     num_threads: int = 4) -> tuple:
        """Parallel batch search"""
        queries = queries.astype(np.float32)
        faiss.normalize_L2(queries)
        
        # FAISS natively parallelizes, but can split manually
        faiss.omp_set_num_threads(num_threads)
        return self.index.search(queries, k)
    
    def realtime_add(self, embedding: np.ndarray):
        """Low-latency single insertion"""
        embedding = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(embedding)
        self.index.add(embedding)
    
    def realtime_search(self, query: np.ndarray, k: int = 10) -> tuple:
        """Low-latency single search"""
        query = query.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        return self.index.search(query, k)
```

**Interview Tips:**
- Batch operations during off-peak hours
- Real-time queries need reserved capacity
- Use different thread pools for batch vs real-time
- Consider queue-based processing for mixed workloads

---

### Question 36
- [ ] Done
**What techniques help handle schema evolution and vector format changes in production?**

**Answer:**

**Definition:**
Handle evolution via: **versioned schemas** (track changes), **dimension padding** (extend vectors), **projection layers** (transform old to new), **dual-write** (parallel indices during migration), and **gradual migration** (convert incrementally).

**Evolution Scenarios:**

| Change | Strategy | Downtime |
|--------|----------|----------|
| **New dimension** | Pad zeros, project | None |
| **New embedding model** | Dual-write + migrate | None |
| **Metadata change** | Schema versioning | None |
| **Index type change** | Blue-green deploy | None |

**Python Code Example:**
```python
import numpy as np

class SchemaEvolution:
    def __init__(self, old_dim: int, new_dim: int):
        self.old_dim = old_dim
        self.new_dim = new_dim
        self.projection = None
    
    def pad_dimension(self, old_vectors: np.ndarray) -> np.ndarray:
        """Extend vectors with zero padding"""
        n = len(old_vectors)
        padding = np.zeros((n, self.new_dim - self.old_dim))
        return np.hstack([old_vectors, padding])
    
    def train_projection(self, old_vectors: np.ndarray, new_vectors: np.ndarray):
        """Learn linear transformation from old to new space"""
        # Solve: W * old = new
        self.projection, _, _, _ = np.linalg.lstsq(
            old_vectors, new_vectors, rcond=None
        )
    
    def project(self, old_vectors: np.ndarray) -> np.ndarray:
        """Transform old vectors to new space"""
        if self.projection is None:
            raise ValueError("Train projection first")
        return old_vectors @ self.projection

class DualWriteIndex:
    """Write to both old and new indices during migration"""
    
    def __init__(self, old_index, new_index, new_embedder):
        self.old_index = old_index
        self.new_index = new_index
        self.new_embedder = new_embedder
    
    def add(self, doc_id: str, text: str, old_embedding: np.ndarray):
        # Write to old
        self.old_index.add(doc_id, old_embedding)
        
        # Generate and write to new
        new_embedding = self.new_embedder.embed(text)
        self.new_index.add(doc_id, new_embedding)
    
    def search(self, query, use_new: bool = False, k: int = 10):
        if use_new:
            return self.new_index.search(query, k)
        return self.old_index.search(query, k)
```

**Interview Tips:**
- Always version your embedding models and schemas
- Dual-write during transitions for zero downtime
- Linear projection preserves similarity structure reasonably
- Plan migration strategy before deploying new models

---

## Evaluation & Benchmarking

### Question 37
- [ ] Done
**How do you evaluate embedding quality using both intrinsic (analogy tasks) and extrinsic (downstream tasks) methods?**

**Answer:**

**Definition:**
**Intrinsic**: evaluate embeddings directly via word similarity, analogy tasks, clustering. **Extrinsic**: evaluate on downstream tasks (classification, search, NER). Extrinsic is more reliable for production decisions; intrinsic is faster for development iteration.

**Evaluation Methods:**

| Type | Method | Measures | Speed |
|------|--------|----------|-------|
| **Intrinsic** | Word similarity | Correlation with human judgments | Fast |
| **Intrinsic** | Analogy | king - man + woman = queen | Fast |
| **Intrinsic** | Clustering | Cluster purity | Fast |
| **Extrinsic** | Classification | Accuracy on task | Slow |
| **Extrinsic** | Retrieval | Recall@k, MRR | Medium |
| **Extrinsic** | NER/NLU | F1 on task | Slow |

**Python Code Example:**
```python
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

class EmbeddingEvaluator:
    def __init__(self, embeddings: dict):
        self.embeddings = embeddings
    
    # Intrinsic: Word similarity
    def word_similarity(self, word_pairs: list) -> float:
        """Evaluate on word similarity dataset (e.g., SimLex-999)"""
        human_scores = []
        model_scores = []
        
        for w1, w2, human_sim in word_pairs:
            if w1 in self.embeddings and w2 in self.embeddings:
                v1, v2 = self.embeddings[w1], self.embeddings[w2]
                cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                human_scores.append(human_sim)
                model_scores.append(cos_sim)
        
        correlation, _ = spearmanr(human_scores, model_scores)
        return correlation
    
    # Intrinsic: Analogy
    def analogy(self, a: str, b: str, c: str, expected: str) -> bool:
        """Test: a is to b as c is to ?"""
        if not all(w in self.embeddings for w in [a, b, c]):
            return False
        
        # d = b - a + c
        target = self.embeddings[b] - self.embeddings[a] + self.embeddings[c]
        
        # Find nearest (excluding a, b, c)
        best_word, best_sim = None, -1
        for word, vec in self.embeddings.items():
            if word in [a, b, c]:
                continue
            sim = np.dot(target, vec) / (np.linalg.norm(target) * np.linalg.norm(vec))
            if sim > best_sim:
                best_sim = sim
                best_word = word
        
        return best_word == expected
    
    # Extrinsic: Classification
    def classification(self, texts: list, labels: list, 
                       text_to_embedding) -> float:
        """Evaluate on downstream classification"""
        X = np.array([text_to_embedding(t) for t in texts])
        y = np.array(labels)
        
        clf = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, X, y, cv=5)
        return scores.mean()
    
    # Extrinsic: Retrieval
    def retrieval_at_k(self, queries: list, relevant_docs: dict,
                       index, k: int = 10) -> dict:
        """Evaluate retrieval quality"""
        recalls = []
        mrrs = []
        
        for query, query_emb in queries:
            results = index.search(query_emb, k)
            retrieved = [r[0] for r in results]
            relevant = set(relevant_docs.get(query, []))
            
            # Recall@k
            recall = len(set(retrieved) & relevant) / len(relevant) if relevant else 0
            recalls.append(recall)
            
            # MRR
            for i, doc_id in enumerate(retrieved):
                if doc_id in relevant:
                    mrrs.append(1.0 / (i + 1))
                    break
            else:
                mrrs.append(0)
        
        return {'recall@k': np.mean(recalls), 'mrr': np.mean(mrrs)}
```

**Interview Tips:**
- Always prioritize extrinsic evaluation for production decisions
- Intrinsic useful for quick iteration during development
- Use task-specific benchmarks (MTEB for retrieval)
- Correlation between intrinsic and extrinsic is imperfect

---

### Question 38
- [ ] Done
**How do you handle the evaluation and benchmarking challenges when comparing vector database performance?**

**Answer:**

**Definition:**
Vector DB benchmarking faces: **approximate results** (trade accuracy for speed), **recall vs latency tradeoffs**, **index build time**, **query distribution variance**, **dimensionality sensitivity**, and **dataset-dependent performance**. Unlike exact-match DBs, "correct" is a spectrum.

**Key Challenges:**

| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| **Approximate results** | ANN returns approximate, not exact | Report recall along with latency |
| **Parameter sensitivity** | Results vary with ef, nprobe, etc. | Benchmark across parameter range |
| **Dataset dependency** | Performance varies by data distribution | Use diverse, representative datasets |
| **Build vs query** | Index build can take hours | Report both build and query times |
| **Recall measurement** | Need ground truth (expensive) | Sample-based evaluation |
| **Workload patterns** | Read/write ratios vary | Test realistic workloads |

**Benchmarking Framework:**
```python
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class BenchmarkResult:
    recall_at_k: float
    qps: float
    p50_latency_ms: float
    p99_latency_ms: float
    build_time_s: float
    memory_mb: float

class VectorDBBenchmark:
    def __init__(self, index, ground_truth_index):
        self.index = index
        self.gt_index = ground_truth_index  # Exact search for ground truth
    
    def compute_ground_truth(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Expensive exact search for ground truth"""
        return self.gt_index.search(queries, k)[1]
    
    def benchmark(self, queries: np.ndarray, k: int = 10,
                  num_runs: int = 3) -> BenchmarkResult:
        # Ground truth
        gt = self.compute_ground_truth(queries, k)
        
        latencies = []
        all_results = None
        
        for _ in range(num_runs):
            for i, query in enumerate(queries):
                start = time.perf_counter()
                _, ids = self.index.search(query.reshape(1, -1), k)
                latencies.append((time.perf_counter() - start) * 1000)
                
                if all_results is None:
                    all_results = []
                if _ == 0:  # Only store first run
                    all_results.append(ids[0])
        
        # Compute recall
        recalls = []
        for i, (result, truth) in enumerate(zip(all_results, gt)):
            recall = len(set(result) & set(truth)) / k
            recalls.append(recall)
        
        return BenchmarkResult(
            recall_at_k=np.mean(recalls),
            qps=len(queries) * num_runs / (sum(latencies) / 1000),
            p50_latency_ms=np.percentile(latencies, 50),
            p99_latency_ms=np.percentile(latencies, 99),
            build_time_s=0,  # Set during build
            memory_mb=0  # Measure separately
        )
```

**Interview Tips:**
- Always report recall alongside latency
- Use ann-benchmarks.com for standardized comparisons
- Different datasets can completely change rankings
- Index parameters dramatically affect results

---

### Question 39
- [ ] Done
**What metrics (recall@k, QPS, latency percentiles) are most important for vector search evaluation?**

**Answer:**

**Definition:**
- **Recall@k**: fraction of true top-k found (accuracy of approximation)
- **QPS**: throughput capacity (queries per second)
- **P50/P95/P99 latency**: user experience (tail latency matters)

These capture the fundamental **accuracy-speed tradeoff** unique to approximate search.

**Metric Details:**

| Metric | Formula | Why Important |
|--------|---------|---------------|
| **Recall@k** | \|retrieved ∩ true_top_k\| / k | Measures result quality |
| **QPS** | queries / second | Capacity planning |
| **P50 latency** | Median response time | Typical user experience |
| **P95/P99 latency** | Tail latency | Worst-case experience |
| **MRR** | 1/rank of first relevant | Ranking quality |
| **Build time** | Index construction time | Operational cost |

**Python Code Example:**
```python
import numpy as np
import time

class VectorMetrics:
    @staticmethod
    def recall_at_k(retrieved: list, relevant: list, k: int) -> float:
        """Fraction of true top-k retrieved"""
        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant[:k])
        return len(retrieved_set & relevant_set) / k
    
    @staticmethod
    def mrr(retrieved: list, relevant: set) -> float:
        """Mean Reciprocal Rank"""
        for i, item in enumerate(retrieved):
            if item in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved: list, relevance_scores: dict, k: int) -> float:
        """Normalized Discounted Cumulative Gain"""
        def dcg(items, scores, k):
            return sum(
                scores.get(item, 0) / np.log2(i + 2)
                for i, item in enumerate(items[:k])
            )
        
        actual_dcg = dcg(retrieved, relevance_scores, k)
        ideal = sorted(relevance_scores.values(), reverse=True)[:k]
        ideal_dcg = sum(s / np.log2(i + 2) for i, s in enumerate(ideal))
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0
    
    @staticmethod
    def measure_latency_percentiles(index, queries: np.ndarray, 
                                      k: int) -> dict:
        """Measure latency distribution"""
        latencies = []
        
        for query in queries:
            start = time.perf_counter()
            index.search(query.reshape(1, -1), k)
            latencies.append((time.perf_counter() - start) * 1000)
        
        return {
            'p50_ms': np.percentile(latencies, 50),
            'p90_ms': np.percentile(latencies, 90),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'qps': 1000 / np.mean(latencies)
        }
```

**Tradeoff Curves:**
```
Recall@10
   ^
1.0|          * exact
   |        *
   |      *       <- sweet spot
   |   *
0.8| *
   +-------------------> QPS
```

**Interview Tips:**
- P99 more important than average for user experience
- Recall@10 of 95%+ typically acceptable for production
- Always benchmark with production-like query distribution
- Plot recall vs QPS curves to find operating points

---

## Integration & Applications

### Question 40
- [ ] Done
**How do you optimize vector database integration with RAG (Retrieval-Augmented Generation) systems?**

**Answer:**

**Definition:**
RAG integration: **query → embed → retrieve → rerank → augment prompt → generate**. Optimization focuses on: chunk sizing, embedding model choice, k value tuning, reranking strategy, and prompt engineering. Vector DB stores document chunks with metadata.

**RAG Pipeline:**
```
Query → Embed → Vector Search → Rerank → Context Window → LLM → Response
                     ↓
              [Chunk 1, Chunk 2, ...]
```

**Optimization Strategies:**

| Component | Optimization | Impact |
|-----------|-------------|--------|
| **Chunk size** | 256-512 tokens optimal | Precision vs context |
| **Overlap** | 10-20% overlap | Continuity vs redundancy |
| **Top-k** | Retrieve 2-3x, rerank | Quality improvement |
| **Reranking** | Cross-encoder reranker | +10-15% relevance |
| **Metadata filters** | Pre-filter by source/date | Faster, focused search |
| **Hybrid search** | Vector + BM25 fusion | Best of both |

**Python Code Example:**
```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict
    embedding: np.ndarray = None

class OptimizedRAG:
    def __init__(self, vector_db, embedder, reranker, llm):
        self.vector_db = vector_db
        self.embedder = embedder
        self.reranker = reranker
        self.llm = llm
    
    def ingest(self, documents: list, chunk_size: int = 512, 
               overlap: int = 50):
        """Chunk and index documents"""
        for doc in documents:
            chunks = self._chunk_text(doc['text'], chunk_size, overlap)
            
            for i, chunk_text in enumerate(chunks):
                chunk = Chunk(
                    id=f"{doc['id']}_{i}",
                    text=chunk_text,
                    metadata={'source': doc['source'], 'doc_id': doc['id']}
                )
                chunk.embedding = self.embedder.embed(chunk_text)
                self.vector_db.add(chunk)
    
    def _chunk_text(self, text: str, size: int, overlap: int) -> list:
        words = text.split()
        chunks = []
        for i in range(0, len(words), size - overlap):
            chunk = ' '.join(words[i:i + size])
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def query(self, question: str, k: int = 5, 
              rerank_top: int = 3, filters: dict = None) -> str:
        # Embed query
        query_embedding = self.embedder.embed(question)
        
        # Retrieve more candidates for reranking
        candidates = self.vector_db.search(
            query_embedding, k=k * 2, filters=filters
        )
        
        # Rerank for quality
        if self.reranker:
            scored = self.reranker.score(question, [c.text for c in candidates])
            candidates = [c for _, c in sorted(
                zip(scored, candidates), reverse=True
            )][:rerank_top]
        else:
            candidates = candidates[:rerank_top]
        
        # Build context
        context = "\n\n".join([c.text for c in candidates])
        
        # Generate with context
        prompt = f"""Answer based on the context.

Context:
{context}

Question: {question}

Answer:"""
        
        return self.llm.generate(prompt)
```

**Interview Tips:**
- Reranking with cross-encoder significantly improves quality
- Metadata filtering reduces search space and improves relevance
- Chunk overlap prevents losing context at boundaries
- Monitor retrieval quality separately from generation quality

---

### Question 41
- [ ] Done
**What are the considerations for implementing embeddings in recommendation systems at scale?**

**Answer:**

**Definition:**
At-scale considerations: **embedding dimension** (balance quality vs memory), **serving latency** (ANN over brute-force), **cold start** (fallback strategies), **update frequency** (real-time vs batch), **two-tower architecture** (separate user/item encoding).

**Scale Challenges:**

| Challenge | Solution | Trade-off |
|-----------|----------|----------|
| **Billions of items** | Two-tower + ANN | Train time vs serving speed |
| **Cold start** | Content embeddings fallback | Accuracy for new items |
| **Real-time updates** | Incremental index updates | Freshness vs stability |
| **User diversity** | Multi-interest embeddings | Complexity vs coverage |
| **Latency** | Quantization, caching | Speed vs accuracy |

**Python Code Example:**
```python
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class User:
    id: str
    embedding: np.ndarray
    history: list

@dataclass  
class Item:
    id: str
    embedding: np.ndarray
    content_embedding: Optional[np.ndarray] = None

class ScalableRecommender:
    def __init__(self, item_index, user_index, 
                 content_embedder, dim: int = 128):
        self.item_index = item_index  # ANN index for items
        self.user_index = user_index  # Optional: for user similarity
        self.content_embedder = content_embedder
        self.dim = dim
        self.items = {}  # id -> Item
        self.users = {}  # id -> User
    
    def add_item(self, item_id: str, collab_embedding: np.ndarray,
                 content_text: str = None):
        """Add item with optional content fallback"""
        content_emb = None
        if content_text:
            content_emb = self.content_embedder.embed(content_text)
        
        item = Item(
            id=item_id,
            embedding=collab_embedding,
            content_embedding=content_emb
        )
        self.items[item_id] = item
        self.item_index.add(item_id, collab_embedding)
    
    def recommend_for_user(self, user_id: str, k: int = 10,
                           use_content_fallback: bool = True) -> list:
        """Get recommendations with cold-start handling"""
        
        if user_id not in self.users:
            # Cold start: return popular or diverse items
            return self._popular_items(k)
        
        user = self.users[user_id]
        
        # ANN search for similar items to user embedding
        candidates = self.item_index.search(user.embedding, k * 3)
        
        # Filter already seen
        seen = set(user.history)
        results = [c for c in candidates if c['id'] not in seen][:k]
        
        return results
    
    def _popular_items(self, k: int) -> list:
        """Fallback for cold start users"""
        # In production: precompute popularity scores
        return list(self.items.values())[:k]
    
    def update_user_embedding(self, user_id: str, 
                              interactions: list):
        """Real-time user embedding update"""
        if not interactions:
            return
        
        # Simple: average of interacted item embeddings
        item_embeddings = [
            self.items[item_id].embedding 
            for item_id in interactions 
            if item_id in self.items
        ]
        
        if item_embeddings:
            new_embedding = np.mean(item_embeddings, axis=0)
            
            if user_id in self.users:
                # Exponential moving average
                old = self.users[user_id].embedding
                self.users[user_id].embedding = 0.8 * old + 0.2 * new_embedding
            else:
                self.users[user_id] = User(
                    id=user_id,
                    embedding=new_embedding,
                    history=interactions
                )
```

**Interview Tips:**
- Two-tower architecture scales to billions of candidates
- Quantization critical for memory at scale (4-bit = 8x savings)
- Mix collaborative and content embeddings for robustness
- A/B test recommendation quality, not just model metrics

---

### Question 42
- [ ] Done
**How do you optimize vector database performance for semantic search applications?**

**Answer:**

**Definition:**
Optimize semantic search via: **hybrid search** (vector + keyword), **query expansion**, **metadata pre-filtering**, **tiered indexes** (hot/cold), **caching**, **reranking**, and **index tuning** (ef, nprobe parameters). Semantic search finds results by meaning, not exact keywords.

**Optimization Strategies:**

| Strategy | Implementation | Benefit |
|----------|---------------|--------|
| **Hybrid search** | Combine BM25 + vector | Best of both worlds |
| **Pre-filtering** | Metadata filters before search | Smaller search space |
| **Caching** | Cache embeddings + results | Reduce computation |
| **Reranking** | Cross-encoder second stage | Higher precision |
| **Index tuning** | Adjust ef_search, nprobe | Latency/recall balance |
| **Query expansion** | Synonyms, paraphrases | Better recall |

**Python Code Example:**
```python
import numpy as np
from rank_bm25 import BM25Okapi
import hashlib

class OptimizedSemanticSearch:
    def __init__(self, embedder, vector_db, reranker=None):
        self.embedder = embedder
        self.vector_db = vector_db
        self.reranker = reranker
        self.documents = []
        self.bm25 = None
        self.cache = {}  # Simple query cache
    
    def index(self, documents: list):
        """Build hybrid index"""
        self.documents = documents
        
        # BM25 index
        tokenized = [doc['text'].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        
        # Vector index
        for doc in documents:
            embedding = self.embedder.embed(doc['text'])
            self.vector_db.add(doc['id'], embedding, doc.get('metadata', {}))
    
    def search(self, query: str, k: int = 10, 
               filters: dict = None, use_hybrid: bool = True,
               use_rerank: bool = True) -> list:
        
        # Check cache
        cache_key = hashlib.md5(f"{query}:{k}:{filters}".encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Embed query
        query_embedding = self.embedder.embed(query)
        
        # Vector search (with optional pre-filtering)
        vector_results = self.vector_db.search(
            query_embedding, k=k*2, filters=filters
        )
        
        if use_hybrid and self.bm25:
            # BM25 search
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            # Combine scores (RRF - Reciprocal Rank Fusion)
            results = self._reciprocal_rank_fusion(
                vector_results, 
                self._bm25_to_results(bm25_scores, k*2)
            )
        else:
            results = vector_results
        
        # Rerank top candidates
        if use_rerank and self.reranker:
            texts = [self._get_doc_text(r['id']) for r in results[:k*2]]
            scores = self.reranker.score(query, texts)
            ranked = sorted(zip(results[:k*2], scores), 
                          key=lambda x: x[1], reverse=True)
            results = [r for r, _ in ranked]
        
        results = results[:k]
        self.cache[cache_key] = results
        return results
    
    def _reciprocal_rank_fusion(self, list1: list, list2: list, 
                                 k: int = 60) -> list:
        """Combine rankings using RRF"""
        scores = {}
        
        for rank, item in enumerate(list1):
            scores[item['id']] = scores.get(item['id'], 0) + 1 / (k + rank + 1)
        
        for rank, item in enumerate(list2):
            scores[item['id']] = scores.get(item['id'], 0) + 1 / (k + rank + 1)
        
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [{'id': id, 'score': scores[id]} for id in sorted_ids]
    
    def _bm25_to_results(self, scores: np.ndarray, k: int) -> list:
        """Convert BM25 scores to result format"""
        top_indices = np.argsort(scores)[-k:][::-1]
        return [
            {'id': self.documents[i]['id'], 'score': scores[i]}
            for i in top_indices
        ]
    
    def _get_doc_text(self, doc_id: str) -> str:
        for doc in self.documents:
            if doc['id'] == doc_id:
                return doc['text']
        return ""
```

**Interview Tips:**
- Hybrid search typically outperforms either alone
- RRF is simple but effective fusion method
- Cross-encoder reranking improves precision significantly
- Pre-filtering by metadata is crucial for large corpora

---

### Question 43
- [ ] Done
**How do you handle the interpretability challenges when using dense embeddings in explainable AI?**

**Answer:**

**Definition:**
Dense embeddings are **opaque** (dimensions have no inherent meaning). Interpretability approaches: **probing classifiers** (what's encoded), **nearest neighbors** (similar items), **dimension analysis** (PCA/clustering), **attention visualization** (for transformers), **concept vectors** (learned directions).

**Interpretability Techniques:**

| Technique | What It Reveals | Complexity |
|-----------|-----------------|------------|
| **Nearest neighbors** | Semantic relationships | Low |
| **Probing classifiers** | Encoded properties | Medium |
| **Visualization (t-SNE/UMAP)** | Cluster structure | Low |
| **Concept activation vectors** | Concept presence | High |
| **Dimension clustering** | Feature groups | Medium |
| **Counterfactual analysis** | Decision boundaries | High |

**Python Code Example:**
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class EmbeddingInterpreter:
    def __init__(self, embeddings: dict):
        self.embeddings = embeddings
        self.words = list(embeddings.keys())
        self.vectors = np.array([embeddings[w] for w in self.words])
    
    def nearest_neighbors(self, word: str, k: int = 10) -> list:
        """Explain by showing similar words"""
        if word not in self.embeddings:
            return []
        
        target = self.embeddings[word]
        similarities = []
        
        for w, vec in self.embeddings.items():
            if w != word:
                sim = np.dot(target, vec) / (
                    np.linalg.norm(target) * np.linalg.norm(vec) + 1e-9
                )
                similarities.append((w, sim))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
    
    def probing_classifier(self, labeled_data: list, property_name: str) -> dict:
        """Check if property is encoded in embeddings"""
        X = []
        y = []
        
        for word, label in labeled_data:
            if word in self.embeddings:
                X.append(self.embeddings[word])
                y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        accuracy = clf.score(X, y)
        
        return {
            'property': property_name,
            'accuracy': accuracy,
            'interpretation': 'strongly encoded' if accuracy > 0.8 else 'weakly encoded'
        }
    
    def find_concept_direction(self, positive_words: list, 
                               negative_words: list) -> np.ndarray:
        """Find direction representing a concept (e.g., gender, sentiment)"""
        pos_vecs = [self.embeddings[w] for w in positive_words 
                    if w in self.embeddings]
        neg_vecs = [self.embeddings[w] for w in negative_words 
                    if w in self.embeddings]
        
        if not pos_vecs or not neg_vecs:
            return None
        
        # Concept direction = difference of centroids
        pos_centroid = np.mean(pos_vecs, axis=0)
        neg_centroid = np.mean(neg_vecs, axis=0)
        
        direction = pos_centroid - neg_centroid
        return direction / np.linalg.norm(direction)
    
    def project_onto_concept(self, word: str, concept_direction: np.ndarray) -> float:
        """How much does word align with concept?"""
        if word not in self.embeddings:
            return 0
        
        return np.dot(self.embeddings[word], concept_direction)
    
    def visualize_clusters(self, words: list = None, method: str = 'tsne'):
        """Visualize embedding space"""
        if words:
            vecs = np.array([self.embeddings[w] for w in words 
                            if w in self.embeddings])
            labels = [w for w in words if w in self.embeddings]
        else:
            vecs = self.vectors
            labels = self.words
        
        if method == 'pca':
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(n_components=2, perplexity=30)
        
        coords = reducer.fit_transform(vecs)
        return list(zip(labels, coords.tolist()))
```

**Interview Tips:**
- Nearest neighbors is simplest explainability technique
- Probing reveals what information is captured
- Concept directions can detect bias (gender, race)
- For production, provide example-based explanations

---

### Question 44
- [ ] Done
**When would you implement multi-vector storage versus single embedding per document?**

**Answer:**

**Definition:**
**Single embedding**: one vector per document (simple, fast). **Multi-vector**: multiple vectors per document (passages, sentences, or aspects). Use multi-vector when documents are long, multi-topic, or require fine-grained retrieval.

**When to Use Each:**

| Scenario | Approach | Reason |
|----------|----------|--------|
| **Short documents** | Single | Sufficient coverage |
| **Long documents** | Multi (chunks) | Better granularity |
| **Multi-topic docs** | Multi (sections) | Topic isolation |
| **Q&A retrieval** | Multi (passages) | Precise answers |
| **Recommendations** | Single | Item-level matching |
| **Legal/Medical** | Multi + metadata | Section-level search |

**Multi-Vector Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Chunking** | Fixed-size overlapping chunks | General documents |
| **Sentence** | One vector per sentence | FAQ, short answers |
| **Paragraph** | One vector per paragraph | Structured docs |
| **ColBERT-style** | One vector per token | Maximum granularity |
| **Hierarchical** | Doc + section + chunk | Multi-level search |

**Python Code Example:**
```python
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class MultiVectorDoc:
    doc_id: str
    doc_embedding: np.ndarray  # Document-level
    chunk_embeddings: List[np.ndarray]  # Chunk-level
    chunk_texts: List[str]

class MultiVectorStore:
    def __init__(self, embedder, doc_index, chunk_index):
        self.embedder = embedder
        self.doc_index = doc_index  # Document-level ANN
        self.chunk_index = chunk_index  # Chunk-level ANN
        self.docs = {}  # doc_id -> MultiVectorDoc
    
    def add_document(self, doc_id: str, text: str, 
                     chunk_size: int = 512, overlap: int = 50):
        # Chunk the document
        chunks = self._chunk_text(text, chunk_size, overlap)
        
        # Embed chunks
        chunk_embeddings = [self.embedder.embed(c) for c in chunks]
        
        # Document embedding: mean of chunks or separate embedding
        doc_embedding = np.mean(chunk_embeddings, axis=0)
        
        doc = MultiVectorDoc(
            doc_id=doc_id,
            doc_embedding=doc_embedding,
            chunk_embeddings=chunk_embeddings,
            chunk_texts=chunks
        )
        self.docs[doc_id] = doc
        
        # Add to indices
        self.doc_index.add(doc_id, doc_embedding)
        for i, (chunk_emb, chunk_text) in enumerate(zip(chunk_embeddings, chunks)):
            chunk_id = f"{doc_id}__chunk_{i}"
            self.chunk_index.add(chunk_id, chunk_emb, 
                               {'doc_id': doc_id, 'text': chunk_text})
    
    def _chunk_text(self, text: str, size: int, overlap: int) -> list:
        words = text.split()
        chunks = []
        for i in range(0, len(words), size - overlap):
            chunk = ' '.join(words[i:i + size])
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def search_documents(self, query: str, k: int = 10) -> list:
        """Coarse-grained document search"""
        query_emb = self.embedder.embed(query)
        return self.doc_index.search(query_emb, k)
    
    def search_chunks(self, query: str, k: int = 10) -> list:
        """Fine-grained chunk search"""
        query_emb = self.embedder.embed(query)
        return self.chunk_index.search(query_emb, k)
    
    def hierarchical_search(self, query: str, 
                            doc_k: int = 5, chunk_k: int = 3) -> list:
        """Two-stage: find docs, then best chunks within"""
        query_emb = self.embedder.embed(query)
        
        # Stage 1: Find relevant documents
        top_docs = self.doc_index.search(query_emb, doc_k)
        
        # Stage 2: Find best chunks within those docs
        results = []
        for doc_result in top_docs:
            doc_id = doc_result['id']
            doc = self.docs[doc_id]
            
            # Score chunks
            chunk_scores = [
                (i, np.dot(query_emb, chunk_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb) + 1e-9
                ))
                for i, chunk_emb in enumerate(doc.chunk_embeddings)
            ]
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
            for i, score in chunk_scores[:chunk_k]:
                results.append({
                    'doc_id': doc_id,
                    'chunk_idx': i,
                    'text': doc.chunk_texts[i],
                    'score': score
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
```

**Interview Tips:**
- Multi-vector better for RAG (find specific passages)
- Single embedding faster and uses less storage
- Hierarchical search balances precision and efficiency
- ColBERT (token-level) best quality but highest cost

---

### Question 45
- [ ] Done
**How do you implement embedding visualization techniques (t-SNE, UMAP) for model debugging?**

**Answer:**

**Definition:**
**t-SNE** and **UMAP** reduce high-dimensional embeddings to 2D/3D for visualization. Use for: **debugging** (clusters form correctly?), **bias detection** (groups separate?), **model comparison** (different structures?). UMAP faster and preserves global structure better.

**Comparison:**

| Aspect | t-SNE | UMAP |
|--------|-------|------|
| **Speed** | Slow (O(n²)) | Fast |
| **Global structure** | Weak | Strong |
| **Hyperparameters** | perplexity | n_neighbors, min_dist |
| **Determinism** | Varies by run | More stable |
| **Use case** | Local clusters | Global + local |

**Python Code Example:**
```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Try importing UMAP
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

class EmbeddingVisualizer:
    def __init__(self, embeddings: dict):
        self.embeddings = embeddings
        self.words = list(embeddings.keys())
        self.vectors = np.array([embeddings[w] for w in self.words])
    
    def reduce_tsne(self, perplexity: int = 30, 
                    n_iter: int = 1000) -> np.ndarray:
        """t-SNE reduction to 2D"""
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=42
        )
        return tsne.fit_transform(self.vectors)
    
    def reduce_umap(self, n_neighbors: int = 15, 
                    min_dist: float = 0.1) -> np.ndarray:
        """UMAP reduction to 2D"""
        if not HAS_UMAP:
            raise ImportError("umap-learn not installed")
        
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        return reducer.fit_transform(self.vectors)
    
    def visualize(self, method: str = 'umap', 
                  labels: dict = None,
                  highlight_words: list = None):
        """Create visualization with optional labels/highlights"""
        
        # Reduce dimensions
        if method == 'umap' and HAS_UMAP:
            coords = self.reduce_umap()
        else:
            coords = self.reduce_tsne()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if labels:
            # Color by category
            unique_labels = set(labels.values())
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            label_to_color = dict(zip(unique_labels, colors))
            
            for i, word in enumerate(self.words):
                color = label_to_color.get(labels.get(word), 'gray')
                ax.scatter(coords[i, 0], coords[i, 1], c=[color], s=20)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=20, alpha=0.5)
        
        # Highlight specific words
        if highlight_words:
            for word in highlight_words:
                if word in self.words:
                    idx = self.words.index(word)
                    ax.annotate(
                        word, 
                        (coords[idx, 0], coords[idx, 1]),
                        fontsize=8,
                        alpha=0.8
                    )
        
        ax.set_title(f'Embedding Visualization ({method.upper()})')
        return fig
    
    def debug_clusters(self, expected_clusters: dict) -> dict:
        """Check if expected clusters are coherent"""
        from sklearn.metrics import silhouette_score
        
        # Get labels for words in clusters
        words_in_clusters = []
        labels = []
        
        for cluster_name, cluster_words in expected_clusters.items():
            for word in cluster_words:
                if word in self.embeddings:
                    words_in_clusters.append(word)
                    labels.append(cluster_name)
        
        if len(set(labels)) < 2:
            return {'error': 'Need at least 2 clusters'}
        
        # Get vectors
        vecs = np.array([self.embeddings[w] for w in words_in_clusters])
        
        # Compute silhouette score
        label_ids = [list(expected_clusters.keys()).index(l) for l in labels]
        score = silhouette_score(vecs, label_ids)
        
        return {
            'silhouette_score': score,
            'interpretation': 'good clustering' if score > 0.3 else 'weak clustering'
        }
```

**Debugging Use Cases:**

| Issue | What to Look For |
|-------|------------------|
| **Poor clustering** | Categories mixed together |
| **Bias** | Demographic groups separated |
| **Outliers** | Isolated points (data issues?) |
| **Model comparison** | Different structures = different behavior |
| **Domain shift** | Test data in different region than train |

**Interview Tips:**
- UMAP preferred for production (faster, better global structure)
- Perplexity in t-SNE: larger = more global structure
- Don't over-interpret: these are approximations
- Use consistent random seed for reproducibility

---

## Security & Compliance

### Question 46
- [ ] Done
**How do you implement effective security measures and access control for vector database deployments?**

**Answer:**

**Definition:**
Secure vector DBs via: **authentication** (API keys, OAuth), **authorization** (RBAC, row-level), **encryption** (at-rest, in-transit), **network isolation** (VPC, firewalls), **audit logging** (access tracking), and **input validation** (embedding injection prevention).

**Security Layers:**

| Layer | Measures | Tools |
|-------|----------|-------|
| **Network** | VPC, firewall, private endpoints | AWS VPC, Azure VNET |
| **Authentication** | API keys, JWT, OAuth2 | Auth0, Okta |
| **Authorization** | RBAC, collection-level ACL | DB-native or proxy |
| **Data** | Encryption at rest/transit | TLS, AES-256 |
| **Audit** | Access logs, query logging | SIEM integration |

**Python Code Example:**
```python
import hashlib
import hmac
import time
import jwt
from functools import wraps
from enum import Enum
from dataclasses import dataclass
from typing import Set

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

@dataclass
class User:
    user_id: str
    roles: Set[str]
    collections: Set[str]  # Allowed collections

class VectorDBSecurity:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.role_permissions = {
            'reader': {Permission.READ},
            'writer': {Permission.READ, Permission.WRITE},
            'admin': {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN}
        }
        self.users = {}  # user_id -> User
        self.audit_log = []
    
    def generate_api_key(self, user_id: str, expires_hours: int = 24) -> str:
        """Generate JWT API key"""
        payload = {
            'user_id': user_id,
            'exp': time.time() + expires_hours * 3600,
            'iat': time.time()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def validate_api_key(self, token: str) -> dict:
        """Validate and decode API key"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            if payload['exp'] < time.time():
                raise ValueError("Token expired")
            return payload
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
    
    def check_permission(self, user_id: str, collection: str, 
                        permission: Permission) -> bool:
        """Check if user has permission on collection"""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        
        # Check collection access
        if collection not in user.collections and '*' not in user.collections:
            return False
        
        # Check role permissions
        for role in user.roles:
            if permission in self.role_permissions.get(role, set()):
                return True
        
        return False
    
    def audit(self, user_id: str, action: str, 
              collection: str, details: dict = None):
        """Log access for audit"""
        self.audit_log.append({
            'timestamp': time.time(),
            'user_id': user_id,
            'action': action,
            'collection': collection,
            'details': details or {}
        })
    
    def requires_permission(self, permission: Permission):
        """Decorator for permission checking"""
        def decorator(func):
            @wraps(func)
            def wrapper(self, user_id: str, collection: str, *args, **kwargs):
                if not self.check_permission(user_id, collection, permission):
                    raise PermissionError(
                        f"User {user_id} lacks {permission.value} on {collection}"
                    )
                self.audit(user_id, func.__name__, collection)
                return func(self, user_id, collection, *args, **kwargs)
            return wrapper
        return decorator

# Usage with secure vector operations
class SecureVectorDB:
    def __init__(self, db, security: VectorDBSecurity):
        self.db = db
        self.security = security
    
    def search(self, user_id: str, collection: str, 
               query_vector, k: int = 10):
        if not self.security.check_permission(
            user_id, collection, Permission.READ
        ):
            raise PermissionError("Access denied")
        
        self.security.audit(user_id, 'search', collection, {'k': k})
        return self.db.search(collection, query_vector, k)
```

**Interview Tips:**
- Embeddings can leak sensitive info - treat as sensitive data
- Row-level security for multi-tenant deployments
- Encrypt at rest AND in transit
- Audit logging essential for compliance (SOC2, HIPAA)

---

### Question 47
- [ ] Done
**What are the considerations for using embeddings in privacy-preserving applications?**

**Answer:**

**Definition:**
Embeddings can **leak private information** (reconstruct text, identify individuals). Privacy techniques: **differential privacy** (add noise during training), **federated learning** (train locally), **homomorphic encryption** (compute on encrypted), **embedding anonymization** (remove identifiers).

**Privacy Risks:**

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Inversion attacks** | Reconstruct text from embeddings | Add noise, truncate precision |
| **Membership inference** | Detect if data was in training | Differential privacy |
| **Attribute inference** | Infer sensitive attributes | Remove or obfuscate |
| **Linkage attacks** | Link embeddings across datasets | Anonymization |

**Python Code Example:**
```python
import numpy as np
import hashlib

class PrivacyPreservingEmbeddings:
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon  # Differential privacy parameter
    
    def add_laplace_noise(self, embedding: np.ndarray, 
                          sensitivity: float = 1.0) -> np.ndarray:
        """Add differential privacy noise"""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, embedding.shape)
        return embedding + noise
    
    def add_gaussian_noise(self, embedding: np.ndarray,
                           delta: float = 1e-5,
                           sensitivity: float = 1.0) -> np.ndarray:
        """Gaussian mechanism for (epsilon, delta)-DP"""
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / self.epsilon
        noise = np.random.normal(0, sigma, embedding.shape)
        return embedding + noise
    
    def truncate_precision(self, embedding: np.ndarray, 
                           precision: int = 2) -> np.ndarray:
        """Reduce precision to limit information"""
        return np.round(embedding, precision)
    
    def normalize_and_clip(self, embedding: np.ndarray, 
                           max_norm: float = 1.0) -> np.ndarray:
        """Clip to bounded sensitivity"""
        norm = np.linalg.norm(embedding)
        if norm > max_norm:
            embedding = embedding * max_norm / norm
        return embedding
    
    def anonymize_embedding(self, embedding: np.ndarray,
                            method: str = 'noise') -> np.ndarray:
        """Full anonymization pipeline"""
        # Step 1: Normalize
        emb = self.normalize_and_clip(embedding)
        
        # Step 2: Add noise
        if method == 'noise':
            emb = self.add_gaussian_noise(emb)
        elif method == 'truncate':
            emb = self.truncate_precision(emb)
        
        # Step 3: Re-normalize
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        
        return emb

class SecureEmbeddingSearch:
    """Privacy-preserving similarity search"""
    
    def __init__(self, privacy: PrivacyPreservingEmbeddings):
        self.privacy = privacy
        self.embeddings = {}  # id -> (noisy_embedding, metadata)
    
    def add(self, doc_id: str, embedding: np.ndarray, 
            metadata: dict = None):
        """Store anonymized embedding"""
        noisy_emb = self.privacy.anonymize_embedding(embedding)
        self.embeddings[doc_id] = (noisy_emb, metadata or {})
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> list:
        """Search with noisy query"""
        # Also anonymize query for symmetry
        noisy_query = self.privacy.anonymize_embedding(query_embedding)
        
        scores = []
        for doc_id, (emb, meta) in self.embeddings.items():
            sim = np.dot(noisy_query, emb)
            scores.append((doc_id, sim, meta))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

# Federated embedding aggregation
class FederatedEmbeddings:
    """Aggregate embeddings without sharing raw data"""
    
    def aggregate_embeddings(self, local_embeddings: list,
                             weights: list = None) -> np.ndarray:
        """Weighted average of local embeddings"""
        if weights is None:
            weights = [1.0] * len(local_embeddings)
        
        weights = np.array(weights) / sum(weights)
        
        aggregated = np.zeros_like(local_embeddings[0])
        for emb, w in zip(local_embeddings, weights):
            aggregated += w * emb
        
        return aggregated
```

**Interview Tips:**
- Embeddings are NOT anonymous - treat as PII
- Differential privacy with ε < 1 provides strong protection
- Trade-off: more privacy = less utility (lower recall)
- GDPR/CCPA may apply to embedding storage

---

### Question 48
- [ ] Done
**How do you implement data governance and compliance measures for vector database systems?**

**Answer:**

**Definition:**
Governance for vector DBs: **data lineage** (track embedding sources), **retention policies** (TTL, deletion), **access controls** (who sees what), **audit trails** (compliance proof), **data quality** (embedding validation), and **regulatory compliance** (GDPR, HIPAA, SOC2).

**Governance Framework:**

| Aspect | Implementation | Compliance |
|--------|---------------|------------|
| **Lineage** | Track source doc → embedding | Auditability |
| **Retention** | TTL, scheduled deletion | GDPR right to erasure |
| **Access** | RBAC, collection-level | Data minimization |
| **Audit** | Query logs, change tracking | SOC2, HIPAA |
| **Quality** | Validation, monitoring | Data integrity |
| **Encryption** | At-rest, in-transit | Security requirements |

**Python Code Example:**
```python
import time
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Optional, List
from enum import Enum

class RetentionPolicy(Enum):
    DAYS_30 = 30
    DAYS_90 = 90
    DAYS_365 = 365
    INDEFINITE = -1

@dataclass
class EmbeddingRecord:
    id: str
    source_doc_id: str
    source_hash: str  # Hash of source for integrity
    created_at: float
    created_by: str
    retention_policy: RetentionPolicy
    tags: List[str]
    pii_flag: bool = False

class VectorDBGovernance:
    def __init__(self, db, audit_logger):
        self.db = db
        self.audit = audit_logger
        self.records = {}  # id -> EmbeddingRecord
    
    def add_with_lineage(self, embedding_id: str, embedding,
                        source_doc: str, user_id: str,
                        retention: RetentionPolicy = RetentionPolicy.DAYS_90,
                        tags: List[str] = None,
                        pii: bool = False):
        """Add embedding with full lineage tracking"""
        
        # Create governance record
        record = EmbeddingRecord(
            id=embedding_id,
            source_doc_id=source_doc,
            source_hash=hashlib.sha256(source_doc.encode()).hexdigest()[:16],
            created_at=time.time(),
            created_by=user_id,
            retention_policy=retention,
            tags=tags or [],
            pii_flag=pii
        )
        
        # Store embedding
        self.db.add(embedding_id, embedding)
        self.records[embedding_id] = record
        
        # Audit log
        self.audit.log('embedding_created', {
            'embedding_id': embedding_id,
            'user': user_id,
            'pii': pii,
            'retention': retention.name
        })
    
    def delete_for_gdpr(self, source_doc_id: str, user_id: str) -> int:
        """Delete all embeddings from a source (right to erasure)"""
        deleted = 0
        to_delete = [
            record.id for record in self.records.values()
            if record.source_doc_id == source_doc_id
        ]
        
        for embedding_id in to_delete:
            self.db.delete(embedding_id)
            del self.records[embedding_id]
            deleted += 1
        
        self.audit.log('gdpr_deletion', {
            'source_doc_id': source_doc_id,
            'user': user_id,
            'embeddings_deleted': deleted
        })
        
        return deleted
    
    def enforce_retention(self) -> int:
        """Delete expired embeddings"""
        now = time.time()
        expired = []
        
        for record in self.records.values():
            if record.retention_policy == RetentionPolicy.INDEFINITE:
                continue
            
            age_days = (now - record.created_at) / 86400
            if age_days > record.retention_policy.value:
                expired.append(record.id)
        
        for embedding_id in expired:
            self.db.delete(embedding_id)
            del self.records[embedding_id]
        
        self.audit.log('retention_enforcement', {
            'deleted_count': len(expired)
        })
        
        return len(expired)
    
    def get_lineage(self, embedding_id: str) -> dict:
        """Get full lineage for an embedding"""
        if embedding_id not in self.records:
            return None
        
        record = self.records[embedding_id]
        return {
            **asdict(record),
            'retention_policy': record.retention_policy.name
        }
    
    def export_audit_report(self, start_time: float, 
                           end_time: float) -> dict:
        """Generate compliance audit report"""
        return {
            'period': {'start': start_time, 'end': end_time},
            'total_embeddings': len(self.records),
            'pii_embeddings': sum(1 for r in self.records.values() if r.pii_flag),
            'by_retention': {
                policy.name: sum(1 for r in self.records.values() 
                               if r.retention_policy == policy)
                for policy in RetentionPolicy
            }
        }
```

**Interview Tips:**
- Embeddings from PII data are still PII under GDPR
- Implement deletion cascades for right to erasure
- Retention policies prevent data accumulation
- Lineage tracking essential for auditability

---

## Advanced Topics

### Question 49
- [ ] Done
**How do you implement version control and reproducibility for embedding-based systems?**

**Answer:**

**Definition:**
Version control for embeddings: **model versioning** (which model created embeddings), **data versioning** (source data snapshots), **index versioning** (index parameters), **config tracking** (hyperparameters). Ensures reproducibility and rollback capability.

**Versioning Components:**

| Component | What to Version | Tool/Method |
|-----------|-----------------|-------------|
| **Embedding model** | Weights, architecture | MLflow, DVC |
| **Source data** | Documents, updates | DVC, Git LFS |
| **Index config** | Parameters (ef, nprobe) | Config files |
| **Embeddings** | Vector snapshots | Object storage |
| **Metadata** | Schema versions | DB migrations |

**Python Code Example:**
```python
import json
import hashlib
import time
import os
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class EmbeddingVersion:
    version_id: str
    model_name: str
    model_version: str
    model_hash: str  # Hash of model weights
    data_version: str
    index_config: dict
    created_at: float
    num_vectors: int
    metrics: dict  # Recall, latency benchmarks

class EmbeddingVersionControl:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.versions = {}  # version_id -> EmbeddingVersion
        self.active_version = None
        self._load_history()
    
    def _load_history(self):
        history_file = os.path.join(self.storage_path, 'versions.json')
        if os.path.exists(history_file):
            with open(history_file) as f:
                data = json.load(f)
                self.versions = {
                    k: EmbeddingVersion(**v) for k, v in data['versions'].items()
                }
                self.active_version = data.get('active')
    
    def _save_history(self):
        history_file = os.path.join(self.storage_path, 'versions.json')
        data = {
            'versions': {k: asdict(v) for k, v in self.versions.items()},
            'active': self.active_version
        }
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_version(self, model_name: str, model_version: str,
                       model_weights_path: str, data_version: str,
                       index_config: dict, num_vectors: int,
                       metrics: dict = None) -> str:
        """Create new embedding version"""
        
        # Hash model weights for integrity
        with open(model_weights_path, 'rb') as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        
        # Generate version ID
        version_id = f"v_{int(time.time())}_{model_hash[:8]}"
        
        version = EmbeddingVersion(
            version_id=version_id,
            model_name=model_name,
            model_version=model_version,
            model_hash=model_hash,
            data_version=data_version,
            index_config=index_config,
            created_at=time.time(),
            num_vectors=num_vectors,
            metrics=metrics or {}
        )
        
        self.versions[version_id] = version
        self._save_history()
        
        return version_id
    
    def set_active(self, version_id: str):
        """Set active version for serving"""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        self.active_version = version_id
        self._save_history()
    
    def rollback(self, version_id: str):
        """Rollback to previous version"""
        self.set_active(version_id)
        # In production: also reload index from snapshot
    
    def compare_versions(self, v1: str, v2: str) -> dict:
        """Compare two versions"""
        ver1 = self.versions[v1]
        ver2 = self.versions[v2]
        
        return {
            'model_changed': ver1.model_hash != ver2.model_hash,
            'data_changed': ver1.data_version != ver2.data_version,
            'config_changed': ver1.index_config != ver2.index_config,
            'metrics_diff': {
                k: ver2.metrics.get(k, 0) - ver1.metrics.get(k, 0)
                for k in set(ver1.metrics) | set(ver2.metrics)
            }
        }
    
    def get_reproducibility_spec(self, version_id: str) -> dict:
        """Get everything needed to reproduce this version"""
        ver = self.versions[version_id]
        return {
            'model': {
                'name': ver.model_name,
                'version': ver.model_version,
                'hash': ver.model_hash
            },
            'data_version': ver.data_version,
            'index_config': ver.index_config,
            'expected_vectors': ver.num_vectors
        }
```

**Interview Tips:**
- Version embedding model AND data together
- Store index snapshots for fast rollback
- Track metrics per version for comparison
- Use content-addressable storage for deduplication

---

### Question 50
- [ ] Done
**When would you implement federated search across multiple vector databases versus centralized storage?**

**Answer:**

**Definition:**
**Federated**: query multiple independent indices, merge results. **Centralized**: single index with all data. Use federated for: **data residency** (geo/legal), **organizational silos**, **scale beyond single node**, **incremental migration**. Centralized simpler but has limits.

**Trade-offs:**

| Aspect | Federated | Centralized |
|--------|-----------|-------------|
| **Latency** | Higher (network) | Lower |
| **Consistency** | Eventually | Strong |
| **Data locality** | Yes | No |
| **Complexity** | Higher | Lower |
| **Scale** | Unlimited | Single node limits |
| **Data residency** | Per-region | Single location |

**When to Use Federated:**
- Data must stay in specific regions (GDPR)
- Different teams own different indices
- Sharding beyond single cluster capacity
- Gradual migration between systems

**Python Code Example:**
```python
import asyncio
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

@dataclass
class SearchResult:
    id: str
    score: float
    source: str  # Which index
    metadata: dict

class FederatedVectorSearch:
    def __init__(self, indices: dict):
        """indices: {name: index_client}"""
        self.indices = indices
        self.executor = ThreadPoolExecutor(max_workers=len(indices))
    
    def search_single(self, index_name: str, index,
                      query_vector: np.ndarray, k: int) -> List[SearchResult]:
        """Search single index"""
        try:
            results = index.search(query_vector, k)
            return [
                SearchResult(
                    id=r['id'],
                    score=r['score'],
                    source=index_name,
                    metadata=r.get('metadata', {})
                )
                for r in results
            ]
        except Exception as e:
            print(f"Error searching {index_name}: {e}")
            return []
    
    def federated_search(self, query_vector: np.ndarray, k: int = 10,
                         indices: List[str] = None,
                         timeout_s: float = 5.0) -> List[SearchResult]:
        """Search across all (or specified) indices"""
        
        target_indices = indices or list(self.indices.keys())
        
        # Parallel search
        futures = [
            self.executor.submit(
                self.search_single, name, self.indices[name], query_vector, k
            )
            for name in target_indices
        ]
        
        # Collect results with timeout
        all_results = []
        for future in futures:
            try:
                results = future.result(timeout=timeout_s)
                all_results.extend(results)
            except Exception as e:
                print(f"Search timeout or error: {e}")
        
        # Merge and rank
        return self._merge_results(all_results, k)
    
    def _merge_results(self, results: List[SearchResult], 
                       k: int) -> List[SearchResult]:
        """Merge results from multiple indices"""
        # Simple: sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Deduplicate by ID (keep highest score)
        seen = set()
        merged = []
        for r in results:
            if r.id not in seen:
                seen.add(r.id)
                merged.append(r)
                if len(merged) >= k:
                    break
        
        return merged
    
    def search_with_routing(self, query_vector: np.ndarray,
                            routing_fn, k: int = 10) -> List[SearchResult]:
        """Route query to specific indices based on query"""
        target_indices = routing_fn(query_vector)
        return self.federated_search(query_vector, k, target_indices)

# Example routing function
def geo_routing(query_metadata: dict) -> List[str]:
    """Route based on user's region"""
    region = query_metadata.get('region', 'us')
    region_map = {
        'us': ['us-east', 'us-west'],
        'eu': ['eu-central'],
        'asia': ['asia-pacific']
    }
    return region_map.get(region, list(region_map.values())[0])
```

**Interview Tips:**
- Federated adds latency (network hops)
- Score normalization important when merging
- Consider partial failures gracefully
- Routing can reduce fan-out and latency

---

### Question 51
- [ ] Done
**How do you optimize embeddings for multimodal use cases (image-text, code similarity)?**

**Answer:**

**Definition:**
Multimodal embeddings map **different modalities** (text, image, code, audio) to **shared vector space** where similar concepts are close regardless of modality. Key: **contrastive learning** (CLIP), **shared encoders**, **cross-modal alignment**.

**Multimodal Approaches:**

| Approach | Description | Use Case |
|----------|-------------|----------|
| **CLIP-style** | Contrastive image-text | Image search by text |
| **CodeBERT** | Code-NL alignment | Code search, documentation |
| **CLAP** | Audio-text | Audio search |
| **ImageBind** | 6 modalities aligned | Universal search |
| **Late fusion** | Separate embeddings, combined | Simpler, modular |

**Python Code Example:**
```python
import numpy as np
from dataclasses import dataclass
from typing import Union, List
from abc import ABC, abstractmethod

class ModalityEncoder(ABC):
    @abstractmethod
    def encode(self, input_data) -> np.ndarray:
        pass

class MultimodalEmbedder:
    def __init__(self, encoders: dict, projection_dim: int = 512):
        """
        encoders: {'text': TextEncoder, 'image': ImageEncoder, ...}
        """
        self.encoders = encoders
        self.projection_dim = projection_dim
        self.projections = {}  # modality -> projection matrix
    
    def encode(self, data, modality: str) -> np.ndarray:
        """Encode to shared space"""
        if modality not in self.encoders:
            raise ValueError(f"Unknown modality: {modality}")
        
        # Get modality-specific embedding
        raw_embedding = self.encoders[modality].encode(data)
        
        # Project to shared space (if projection exists)
        if modality in self.projections:
            embedding = raw_embedding @ self.projections[modality]
        else:
            embedding = raw_embedding
        
        # L2 normalize for cosine similarity
        return embedding / (np.linalg.norm(embedding) + 1e-9)
    
    def train_alignment(self, pairs: List[tuple], 
                        modality1: str, modality2: str,
                        epochs: int = 100, lr: float = 0.01):
        """Learn projection to align two modalities"""
        # Get embeddings for all pairs
        emb1 = np.array([
            self.encoders[modality1].encode(p[0]) for p in pairs
        ])
        emb2 = np.array([
            self.encoders[modality2].encode(p[1]) for p in pairs
        ])
        
        # Learn linear projection: W such that emb1 @ W ≈ emb2
        W, _, _, _ = np.linalg.lstsq(emb1, emb2, rcond=None)
        self.projections[modality1] = W
    
    def cross_modal_search(self, query, query_modality: str,
                           index, target_modality: str,
                           k: int = 10) -> list:
        """Search across modalities"""
        query_embedding = self.encode(query, query_modality)
        return index.search(query_embedding, k)

class CodeTextEmbedder:
    """Specialized for code-text similarity"""
    
    def __init__(self, code_encoder, text_encoder, projection_dim: int = 256):
        self.code_encoder = code_encoder
        self.text_encoder = text_encoder
        self.dim = projection_dim
    
    def encode_code(self, code: str) -> np.ndarray:
        """Encode code snippet"""
        # Preprocess: remove comments, normalize whitespace
        processed = self._preprocess_code(code)
        embedding = self.code_encoder.encode(processed)
        return embedding / (np.linalg.norm(embedding) + 1e-9)
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode natural language query"""
        embedding = self.text_encoder.encode(query)
        return embedding / (np.linalg.norm(embedding) + 1e-9)
    
    def _preprocess_code(self, code: str) -> str:
        """Normalize code for embedding"""
        # Remove single-line comments
        lines = []
        for line in code.split('\n'):
            if not line.strip().startswith('#'):
                lines.append(line)
        return '\n'.join(lines)
    
    def search_code_by_description(self, description: str,
                                   code_index, k: int = 10) -> list:
        """Find code snippets matching description"""
        query_embedding = self.encode_query(description)
        return code_index.search(query_embedding, k)
```

**Optimization Tips:**

| Optimization | Benefit |
|-------------|--------|
| **Hard negatives** | Better discrimination |
| **Temperature scaling** | Control similarity distribution |
| **Projection layers** | Align modality-specific dims |
| **Data augmentation** | More robust embeddings |

**Interview Tips:**
- CLIP: contrastive loss on (image, text) pairs
- Shared space enables cross-modal retrieval
- Normalization ensures comparable scores across modalities
- Fine-tune on domain-specific pairs for best results

---

### Question 52
- [ ] Done
**What strategies help future-proof vector database architectures for evolving AI requirements?**

**Answer:**

**Definition:**
Future-proof via: **abstraction layers** (swap implementations), **dimension flexibility** (handle varying sizes), **schema evolution** (versioned metadata), **hybrid capabilities** (vector + keyword + graph), **modular design** (pluggable components).

**Future-Proofing Strategies:**

| Strategy | Implementation | Protects Against |
|----------|---------------|------------------|
| **Abstraction layer** | Generic vector DB interface | Vendor lock-in |
| **Dimension handling** | Padding/projection | New embedding models |
| **Schema versioning** | Migrate metadata | Requirements change |
| **Hybrid search** | Vector + sparse + graph | New search paradigms |
| **Modular embedders** | Pluggable encoders | Model evolution |
| **API versioning** | v1/v2 endpoints | Breaking changes |

**Python Code Example:**
```python
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Any
from dataclasses import dataclass

@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict

class VectorDBInterface(ABC):
    """Abstract interface for any vector database"""
    
    @abstractmethod
    def add(self, id: str, vector: np.ndarray, metadata: dict = None): pass
    
    @abstractmethod
    def search(self, query: np.ndarray, k: int, 
               filters: dict = None) -> List[SearchResult]: pass
    
    @abstractmethod
    def delete(self, id: str): pass
    
    @abstractmethod
    def update_metadata(self, id: str, metadata: dict): pass

class DimensionAdapter:
    """Handle different embedding dimensions"""
    
    def __init__(self, target_dim: int):
        self.target_dim = target_dim
        self.projections = {}  # source_dim -> projection matrix
    
    def adapt(self, vector: np.ndarray) -> np.ndarray:
        """Adapt vector to target dimension"""
        source_dim = len(vector)
        
        if source_dim == self.target_dim:
            return vector
        
        if source_dim < self.target_dim:
            # Pad with zeros
            padded = np.zeros(self.target_dim)
            padded[:source_dim] = vector
            return padded
        
        # Project down (use PCA or learned projection)
        if source_dim not in self.projections:
            # Random projection as fallback
            self.projections[source_dim] = np.random.randn(
                source_dim, self.target_dim
            ) / np.sqrt(source_dim)
        
        return vector @ self.projections[source_dim]

class FutureProofVectorDB:
    """Wrapper with future-proofing features"""
    
    def __init__(self, backend: VectorDBInterface,
                 embedder_registry: dict,
                 target_dim: int = 1024):
        self.backend = backend
        self.embedders = embedder_registry  # name -> embedder
        self.dimension_adapter = DimensionAdapter(target_dim)
        self.active_embedder = list(embedder_registry.keys())[0]
        self.schema_version = 1
    
    def set_embedder(self, name: str):
        """Switch embedding model"""
        if name not in self.embedders:
            raise ValueError(f"Unknown embedder: {name}")
        self.active_embedder = name
    
    def add(self, id: str, content: Any, metadata: dict = None):
        """Add with current embedder"""
        # Generate embedding
        raw_embedding = self.embedders[self.active_embedder].embed(content)
        
        # Adapt dimension
        embedding = self.dimension_adapter.adapt(raw_embedding)
        
        # Add metadata tracking
        full_metadata = {
            'original_dim': len(raw_embedding),
            'embedder': self.active_embedder,
            'schema_version': self.schema_version,
            **(metadata or {})
        }
        
        self.backend.add(id, embedding, full_metadata)
    
    def search(self, query: Any, k: int = 10,
               filters: dict = None) -> List[SearchResult]:
        """Search with current embedder"""
        raw_query = self.embedders[self.active_embedder].embed(query)
        query_embedding = self.dimension_adapter.adapt(raw_query)
        return self.backend.search(query_embedding, k, filters)
    
    def migrate_embeddings(self, new_embedder: str,
                          batch_size: int = 1000):
        """Migrate to new embedding model"""
        old_embedder = self.active_embedder
        self.active_embedder = new_embedder
        self.schema_version += 1
        
        # In production: batch process all documents
        # Re-embed and update
        print(f"Migration from {old_embedder} to {new_embedder}")
        print(f"New schema version: {self.schema_version}")
    
    def hybrid_search(self, query: Any, k: int = 10,
                      vector_weight: float = 0.7,
                      keyword_results: List[SearchResult] = None):
        """Combine vector with other search results"""
        vector_results = self.search(query, k * 2)
        
        if keyword_results:
            # Merge using reciprocal rank fusion
            return self._rrf_merge(vector_results, keyword_results, 
                                  vector_weight, k)
        return vector_results[:k]
    
    def _rrf_merge(self, list1, list2, weight1, k):
        """Reciprocal rank fusion"""
        scores = {}
        for rank, r in enumerate(list1):
            scores[r.id] = weight1 / (60 + rank + 1)
        for rank, r in enumerate(list2):
            scores[r.id] = scores.get(r.id, 0) + (1 - weight1) / (60 + rank + 1)
        
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_ids[:k]
```

**Interview Tips:**
- Abstract interfaces prevent vendor lock-in
- Store original dimension for reversibility
- Track embedder version in metadata
- Plan for migration paths before they're needed

---

### Question 53
- [ ] Done
**How do you handle temporal aspects of embeddings when working with evolving vocabularies?**

**Answer:**

**Definition:**
Temporal challenges: **vocabulary drift** (new words, changing meanings), **concept drift** (relationships change), **model staleness** (old embeddings outdated). Solutions: **incremental updates**, **temporal versioning**, **continuous learning**, **alignment across time**.

**Temporal Challenges:**

| Challenge | Example | Solution |
|-----------|---------|----------|
| **New vocabulary** | "COVID-19" (2020) | Incremental training |
| **Meaning shift** | "viral" (disease → social media) | Temporal embeddings |
| **Concept drift** | Industry terminology | Periodic retraining |
| **Historical data** | Old documents still relevant | Multi-version index |

**Python Code Example:**
```python
import numpy as np
from collections import defaultdict
import time

class TemporalEmbeddings:
    def __init__(self, base_embedder, update_interval_days: int = 30):
        self.base_embedder = base_embedder
        self.update_interval = update_interval_days * 86400
        
        self.vocab_by_time = defaultdict(set)  # timestamp -> new words
        self.embeddings_by_time = {}  # (word, timestamp) -> embedding
        self.current_vocab = set()
        self.last_update = time.time()
    
    def add_new_term(self, term: str, context_examples: list,
                     timestamp: float = None):
        """Add new vocabulary term with examples"""
        timestamp = timestamp or time.time()
        
        if term in self.current_vocab:
            # Track as meaning update
            pass
        else:
            self.vocab_by_time[timestamp].add(term)
            self.current_vocab.add(term)
        
        # Generate embedding from context
        if context_examples:
            embeddings = [self.base_embedder.embed(ex) for ex in context_examples]
            self.embeddings_by_time[(term, timestamp)] = np.mean(embeddings, axis=0)
    
    def get_embedding(self, term: str, 
                      as_of: float = None) -> np.ndarray:
        """Get embedding at specific point in time"""
        as_of = as_of or time.time()
        
        # Find most recent embedding before as_of
        best_time = None
        for (word, ts), emb in self.embeddings_by_time.items():
            if word == term and ts <= as_of:
                if best_time is None or ts > best_time:
                    best_time = ts
        
        if best_time:
            return self.embeddings_by_time[(term, best_time)]
        
        # Fall back to base embedder
        return self.base_embedder.embed(term)
    
    def detect_drift(self, term: str, 
                     time_windows: list) -> dict:
        """Detect if term's meaning has drifted over time"""
        embeddings = []
        for window_start, window_end in time_windows:
            emb = self.get_embedding(term, as_of=window_end)
            embeddings.append((window_start, emb))
        
        if len(embeddings) < 2:
            return {'drift_detected': False}
        
        # Compute similarity between consecutive windows
        similarities = []
        for i in range(1, len(embeddings)):
            sim = np.dot(embeddings[i][1], embeddings[i-1][1]) / (
                np.linalg.norm(embeddings[i][1]) * 
                np.linalg.norm(embeddings[i-1][1]) + 1e-9
            )
            similarities.append(sim)
        
        avg_sim = np.mean(similarities)
        return {
            'drift_detected': avg_sim < 0.9,
            'average_similarity': avg_sim,
            'min_similarity': min(similarities)
        }

class IncrementalVocabularyUpdate:
    """Update embeddings incrementally as new terms appear"""
    
    def __init__(self, base_dim: int = 300):
        self.dim = base_dim
        self.embeddings = {}
        self.oov_embeddings = {}  # Temporary embeddings for new terms
    
    def handle_oov(self, term: str, context: str) -> np.ndarray:
        """Generate embedding for out-of-vocabulary term"""
        if term in self.oov_embeddings:
            return self.oov_embeddings[term]
        
        # Simple: average of context words
        context_words = context.lower().split()
        known_embeddings = [
            self.embeddings[w] for w in context_words 
            if w in self.embeddings
        ]
        
        if known_embeddings:
            oov_emb = np.mean(known_embeddings, axis=0)
        else:
            # Random initialization based on term hash
            np.random.seed(hash(term) % (2**32))
            oov_emb = np.random.randn(self.dim) * 0.1
        
        self.oov_embeddings[term] = oov_emb
        return oov_emb
    
    def promote_oov(self, terms: list):
        """Promote OOV terms to permanent vocabulary after retraining"""
        for term in terms:
            if term in self.oov_embeddings:
                self.embeddings[term] = self.oov_embeddings.pop(term)
```

**Interview Tips:**
- Track when embeddings were created (temporal metadata)
- Periodic retraining catches vocabulary evolution
- Align embeddings across time for fair comparison
- OOV handling is first line of defense for new terms

---

### Question 54
- [ ] Done
**What techniques help optimize embedding training for domain-specific corpora with limited data?**

**Answer:**

**Definition:**
Limited domain data techniques: **transfer learning** (start from pretrained), **fine-tuning** (adapt to domain), **data augmentation** (expand training set), **contrastive learning** (leverage pairs), **multi-task learning** (share representations). Key: leverage general knowledge + domain signal.

**Techniques for Limited Data:**

| Technique | Description | Data Requirement |
|-----------|-------------|------------------|
| **Fine-tune pretrained** | Start from BERT, adapt | 1K-10K examples |
| **Adapter layers** | Freeze base, train adapters | 100-1K examples |
| **Contrastive fine-tune** | Positive/negative pairs | 1K+ pairs |
| **Data augmentation** | Paraphrase, back-translate | Multiplies data |
| **Few-shot prompting** | Use LLM for embeddings | 10-100 examples |

**Python Code Example:**
```python
import numpy as np
from typing import List, Tuple
import random

class DomainEmbeddingTrainer:
    def __init__(self, base_model, embedding_dim: int = 768):
        self.base_model = base_model
        self.dim = embedding_dim
        self.adapter = None  # Lightweight adapter layer
    
    def init_adapter(self, hidden_dim: int = 64):
        """Initialize lightweight adapter for domain adaptation"""
        # Down-project, nonlinearity, up-project
        self.adapter = {
            'down': np.random.randn(self.dim, hidden_dim) * 0.01,
            'up': np.random.randn(hidden_dim, self.dim) * 0.01
        }
    
    def embed_with_adapter(self, text: str) -> np.ndarray:
        """Apply adapter to base embeddings"""
        base_emb = self.base_model.embed(text)
        
        if self.adapter:
            # Adapter: residual connection
            hidden = np.tanh(base_emb @ self.adapter['down'])
            adapted = base_emb + hidden @ self.adapter['up']
            return adapted / (np.linalg.norm(adapted) + 1e-9)
        
        return base_emb
    
    def fine_tune_contrastive(self, pairs: List[Tuple[str, str, bool]],
                               epochs: int = 10, lr: float = 0.001,
                               margin: float = 0.5):
        """
        Fine-tune with contrastive loss
        pairs: [(text1, text2, is_similar), ...]
        """
        if not self.adapter:
            self.init_adapter()
        
        for epoch in range(epochs):
            random.shuffle(pairs)
            total_loss = 0
            
            for text1, text2, is_similar in pairs:
                emb1 = self.embed_with_adapter(text1)
                emb2 = self.embed_with_adapter(text2)
                
                # Cosine similarity
                sim = np.dot(emb1, emb2)
                
                # Contrastive loss
                if is_similar:
                    loss = max(0, 1 - sim)  # Should be close to 1
                else:
                    loss = max(0, sim - margin)  # Should be < margin
                
                total_loss += loss
                
                # Gradient update (simplified)
                if loss > 0:
                    grad_scale = lr * (1 if is_similar else -1)
                    # Update adapter weights
                    self.adapter['up'] += grad_scale * np.outer(
                        np.tanh(emb1 @ self.adapter['down']), emb1 - emb2
                    )
            
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(pairs):.4f}")

class DataAugmenter:
    """Augment limited domain data"""
    
    def __init__(self, synonym_dict: dict = None):
        self.synonyms = synonym_dict or {}
    
    def synonym_replacement(self, text: str, p: float = 0.1) -> str:
        """Replace words with synonyms"""
        words = text.split()
        augmented = []
        
        for word in words:
            if word.lower() in self.synonyms and random.random() < p:
                augmented.append(random.choice(self.synonyms[word.lower()]))
            else:
                augmented.append(word)
        
        return ' '.join(augmented)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words"""
        words = text.split()
        if len(words) <= 1:
            return text
        
        return ' '.join([w for w in words if random.random() > p])
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """Randomly swap adjacent words"""
        words = text.split()
        for _ in range(n):
            if len(words) < 2:
                break
            i = random.randint(0, len(words) - 2)
            words[i], words[i+1] = words[i+1], words[i]
        return ' '.join(words)
    
    def augment(self, text: str, n_augments: int = 4) -> List[str]:
        """Generate multiple augmentations"""
        augmented = [text]  # Include original
        
        for _ in range(n_augments):
            aug_type = random.choice(['synonym', 'delete', 'swap'])
            if aug_type == 'synonym':
                augmented.append(self.synonym_replacement(text))
            elif aug_type == 'delete':
                augmented.append(self.random_deletion(text))
            else:
                augmented.append(self.random_swap(text))
        
        return augmented

def create_training_pairs(documents: List[str], 
                          labels: List[str]) -> List[Tuple[str, str, bool]]:
    """Create contrastive pairs from labeled documents"""
    pairs = []
    
    # Group by label
    by_label = {}
    for doc, label in zip(documents, labels):
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(doc)
    
    # Create positive and negative pairs
    for label, docs in by_label.items():
        # Positive pairs (same label)
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                pairs.append((docs[i], docs[j], True))
        
        # Negative pairs (different labels)
        for other_label, other_docs in by_label.items():
            if other_label != label:
                for doc1 in docs[:5]:  # Limit pairs
                    for doc2 in other_docs[:5]:
                        pairs.append((doc1, doc2, False))
    
    return pairs
```

**Interview Tips:**
- Start from best pretrained model for your domain
- Adapter layers train 10x fewer parameters
- Data augmentation can 2-4x your effective dataset
- Contrastive learning works well with limited pairs

---

### Question 55
- [ ] Done
**How do you implement effective embedding caching strategies to reduce computational overhead?**

**Answer:**

**Definition:**
Cache embeddings to avoid recomputation: **input caching** (text → embedding), **result caching** (query → results), **batch caching** (prefetch popular), **tiered caching** (memory → disk → compute). Key metrics: hit rate, latency savings, memory usage.

**Caching Layers:**

| Layer | What to Cache | Latency Reduction |
|-------|---------------|-------------------|
| **L1 (Memory)** | Hot embeddings | 100-1000x |
| **L2 (Redis)** | Warm embeddings | 10-100x |
| **L3 (Disk)** | Cold embeddings | 2-10x |
| **Compute** | Recompute on miss | Baseline |

**Python Code Example:**
```python
import hashlib
import time
import pickle
from collections import OrderedDict
from typing import Optional
import numpy as np

class LRUEmbeddingCache:
    """In-memory LRU cache for embeddings"""
    
    def __init__(self, capacity: int = 10000, ttl_seconds: float = 3600):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.stats = {'hits': 0, 'misses': 0}
    
    def _make_key(self, text: str, model: str = 'default') -> str:
        """Create cache key from text"""
        return hashlib.sha256(f"{model}:{text}".encode()).hexdigest()
    
    def get(self, text: str, model: str = 'default') -> Optional[np.ndarray]:
        key = self._make_key(text, model)
        
        if key not in self.cache:
            self.stats['misses'] += 1
            return None
        
        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl:
            del self.cache[key]
            del self.timestamps[key]
            self.stats['misses'] += 1
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.stats['hits'] += 1
        return self.cache[key]
    
    def put(self, text: str, embedding: np.ndarray, 
            model: str = 'default'):
        key = self._make_key(text, model)
        
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                oldest = next(iter(self.cache))
                del self.cache[oldest]
                del self.timestamps[oldest]
        
        self.cache[key] = embedding
        self.timestamps[key] = time.time()
    
    def hit_rate(self) -> float:
        total = self.stats['hits'] + self.stats['misses']
        return self.stats['hits'] / total if total > 0 else 0

class TieredEmbeddingCache:
    """Multi-tier caching with memory, Redis, and compute fallback"""
    
    def __init__(self, embedder, memory_capacity: int = 10000,
                 redis_client=None):
        self.embedder = embedder
        self.l1_cache = LRUEmbeddingCache(memory_capacity)
        self.redis = redis_client
        self.stats = {'l1_hits': 0, 'l2_hits': 0, 'computes': 0}
    
    def get_embedding(self, text: str) -> np.ndarray:
        # L1: Memory cache
        embedding = self.l1_cache.get(text)
        if embedding is not None:
            self.stats['l1_hits'] += 1
            return embedding
        
        # L2: Redis cache
        if self.redis:
            key = f"emb:{hashlib.sha256(text.encode()).hexdigest()}"
            cached = self.redis.get(key)
            if cached:
                embedding = pickle.loads(cached)
                self.l1_cache.put(text, embedding)  # Promote to L1
                self.stats['l2_hits'] += 1
                return embedding
        
        # Compute embedding
        embedding = self.embedder.embed(text)
        self.stats['computes'] += 1
        
        # Store in caches
        self.l1_cache.put(text, embedding)
        if self.redis:
            key = f"emb:{hashlib.sha256(text.encode()).hexdigest()}"
            self.redis.setex(key, 3600, pickle.dumps(embedding))
        
        return embedding
    
    def batch_embed(self, texts: list) -> list:
        """Batch embedding with caching"""
        results = [None] * len(texts)
        to_compute = []  # (index, text)
        
        # Check cache first
        for i, text in enumerate(texts):
            embedding = self.l1_cache.get(text)
            if embedding is not None:
                results[i] = embedding
                self.stats['l1_hits'] += 1
            else:
                to_compute.append((i, text))
        
        # Batch compute missing
        if to_compute:
            texts_to_compute = [t for _, t in to_compute]
            computed = self.embedder.batch_embed(texts_to_compute)
            
            for (i, text), embedding in zip(to_compute, computed):
                results[i] = embedding
                self.l1_cache.put(text, embedding)
                self.stats['computes'] += 1
        
        return results
    
    def prefetch(self, texts: list):
        """Prefetch popular embeddings into cache"""
        for text in texts:
            if self.l1_cache.get(text) is None:
                embedding = self.embedder.embed(text)
                self.l1_cache.put(text, embedding)
    
    def get_stats(self) -> dict:
        total = sum(self.stats.values())
        return {
            **self.stats,
            'total': total,
            'l1_hit_rate': self.stats['l1_hits'] / total if total else 0,
            'compute_rate': self.stats['computes'] / total if total else 0
        }

class SearchResultCache:
    """Cache complete search results"""
    
    def __init__(self, capacity: int = 5000, ttl_seconds: float = 300):
        self.cache = LRUEmbeddingCache(capacity, ttl_seconds)
    
    def get_results(self, query: str, k: int, 
                    filters: dict = None) -> Optional[list]:
        key = f"{query}:{k}:{hash(frozenset(filters.items()) if filters else '')}"
        cached = self.cache.get(key)
        return cached if cached is not None else None
    
    def store_results(self, query: str, k: int, 
                      results: list, filters: dict = None):
        key = f"{query}:{k}:{hash(frozenset(filters.items()) if filters else '')}"
        self.cache.put(key, results)
```

**Interview Tips:**
- Embedding computation often 10-100ms (caching valuable)
- L1 hit rate >80% is good for typical workloads
- Invalidate cache when embedder model changes
- Result caching has shorter TTL (data may change)

---


---

# --- Embeddings Questions (from 09_llm_genai_tools/05_embeddings) ---

# Embeddings (Word2Vec, GloVe, BERT Embeddings) - Theory Questions

## Question 1
- [ ] Done
**How do you choose between Word2Vec, GloVe, and BERT embeddings based on your specific NLP task requirements?**
**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**What are the computational trade-offs when using static embeddings versus contextualized BERT embeddings in production?**
**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How do you handle out-of-vocabulary words when using pre-trained Word2Vec or GloVe embeddings?**
**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**When would you combine multiple embedding types (Word2Vec + BERT) for improved downstream task performance?**
**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**How do you implement efficient embedding storage and retrieval systems for large-scale applications?**
**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What techniques help optimize BERT embedding extraction for real-time inference requirements?**
**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How do you handle domain adaptation when pre-trained embeddings don't match your specific use case?**
**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What are the best practices for fine-tuning BERT embeddings while preserving general language understanding?**
**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How do you implement effective dimensionality reduction for high-dimensional BERT embeddings without losing semantic information?**
**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**When should you train custom Word2Vec or GloVe embeddings versus using pre-trained models?**
**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How do you handle multilingual requirements when choosing between different embedding approaches?**
**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**What strategies help maintain embedding quality when dealing with noisy or informal text data?**
**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do you implement effective similarity metrics for different types of embeddings in search applications?**
**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What techniques help optimize memory usage when working with large embedding matrices?**
**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How do you handle the temporal aspects of embeddings when working with evolving vocabularies?**
**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**When would you use subword embeddings (FastText) versus word-level embeddings for morphologically rich languages?**
**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do you implement effective embedding alignment techniques for cross-lingual applications?**
**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What are the best practices for evaluating embedding quality using both intrinsic and extrinsic methods?**
**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How do you handle the scalability challenges when computing embeddings for large document collections?**
**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What strategies help reduce bias in word embeddings for fair and inclusive NLP applications?**
**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you implement effective embedding compression techniques for mobile and edge deployments?**
**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**When should you use averaged embeddings versus more sophisticated aggregation methods for document representation?**
**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How do you optimize embedding-based retrieval systems for sub-second response times?**
**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**What techniques help maintain embedding consistency across different model versions and updates?**
**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How do you handle the context length limitations when extracting BERT embeddings from long documents?**
**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What are the considerations for implementing embedding-based recommendation systems at scale?**
**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you optimize embedding training procedures for domain-specific corpora with limited data?**
**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**When would you implement custom embedding architectures versus using established approaches?**
**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do you handle the interpretability challenges when using dense embeddings in explainable AI systems?**
**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What strategies help optimize embedding-based clustering and classification performance?**
**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How do you implement effective embedding visualization techniques for model debugging and analysis?**
**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**What are the best practices for handling embedding drift in production systems over time?**
**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**How do you optimize batch processing for embedding generation in high-throughput applications?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**When should you use sentence-level embeddings versus aggregated word embeddings for document analysis?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you handle the evaluation challenges when comparing different embedding approaches on downstream tasks?**
**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What techniques help optimize embedding-based semantic search systems for diverse query types?**
**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you implement effective embedding caching strategies to reduce computational overhead?**
**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What are the considerations for using embeddings in privacy-preserving machine learning applications?**
**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How do you optimize embedding models for specific hardware configurations and memory constraints?**
**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**When would you implement ensemble approaches using multiple embedding types for robust representations?**
**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How do you handle the version control and reproducibility challenges with embedding-based systems?**
**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What strategies help ensure embedding quality when dealing with adversarial or malicious inputs?**
**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you optimize embedding-based transfer learning for low-resource languages or domains?**
**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What techniques help reduce the computational cost of real-time embedding generation and matching?**
**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you implement effective embedding model monitoring and performance tracking in production?**
**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**When should you use specialized embeddings (medical, legal, scientific) versus general-purpose embeddings?**
**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you handle the integration challenges when incorporating embeddings into existing ML pipelines?**
**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**What are the best practices for implementing embedding-based content moderation and safety systems?**
**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**How do you optimize embedding approaches for specific use cases like code similarity, image-text matching, or audio processing?**
**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**What strategies help ensure embedding-based systems maintain performance under changing data distributions?**
**Answer:** _To be filled_

---


---

# --- Vector Databases Questions (from 09_llm_genai_tools/06_vector_databases) ---

# Vector Databases (FAISS, ChromaDB, Pinecone) - Theory Questions

## Question 1
- [ ] Done
**How do you choose between FAISS, ChromaDB, and Pinecone based on scalability, cost, and performance requirements?**
**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**What are the trade-offs between self-hosted FAISS/ChromaDB versus managed solutions like Pinecone?**
**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How do you optimize index selection (IVF, HNSW, LSH) in FAISS for different query patterns and dataset sizes?**
**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**When would you use ChromaDB's local deployment versus distributed setups for production applications?**
**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**How do you handle Pinecone's pricing and quota limitations in high-volume vector search applications?**
**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What techniques help optimize vector indexing speed without compromising search accuracy across different databases?**
**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How do you implement effective data partitioning strategies for large-scale vector collections?**
**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What are the best practices for handling vector dimensionality and normalization across different database systems?**
**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How do you optimize query performance for different similarity metrics (cosine, euclidean, dot product)?**
**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**When should you implement hybrid search combining vector similarity with traditional keyword search?**
**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How do you handle incremental updates and real-time indexing in production vector database systems?**
**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**What strategies help optimize memory usage and disk I/O for large vector datasets?**
**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do you implement effective backup and disaster recovery procedures for vector databases?**
**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What techniques help ensure vector database performance consistency under varying load conditions?**
**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How do you handle multi-tenancy and data isolation requirements in vector database deployments?**
**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**When would you implement custom distance metrics versus using built-in similarity functions?**
**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do you optimize vector compression and quantization strategies to balance storage and accuracy?**
**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What are the best practices for monitoring and alerting in vector database production environments?**
**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How do you handle the integration challenges when connecting vector databases to ML pipelines?**
**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What strategies help optimize batch operations versus real-time queries in vector databases?**
**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you implement effective security measures and access control for vector database deployments?**
**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**When should you use approximate nearest neighbor search versus exact search based on application requirements?**
**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How do you optimize vector database performance for different embedding types and dimensions?**
**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**What techniques help handle schema evolution and vector format changes in production systems?**
**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How do you implement effective load balancing and horizontal scaling for vector database clusters?**
**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What are the considerations for implementing vector databases in edge computing environments?**
**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you optimize vector ingestion pipelines for high-throughput data processing scenarios?**
**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**When would you implement federated search across multiple vector databases versus single centralized storage?**
**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do you handle the evaluation and benchmarking challenges when comparing vector database performance?**
**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What strategies help optimize vector database deployments for specific hardware configurations?**
**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How do you implement effective caching strategies to improve vector search response times?**
**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**What techniques help optimize vector database performance for recommendation system applications?**
**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**How do you handle version control and model lifecycle management for vector embeddings?**
**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**When should you implement custom indexing strategies versus using default configurations?**
**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you optimize vector database performance for semantic search applications with diverse query types?**
**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What are the best practices for handling vector database migrations and upgrades in production?**
**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you implement effective cost optimization strategies for cloud-based vector database deployments?**
**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What techniques help ensure vector database reliability and fault tolerance in distributed environments?**
**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How do you handle the complexity of tuning index parameters for optimal search performance?**
**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**When would you implement multi-vector storage versus single embedding per document approaches?**
**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How do you optimize vector database integration with RAG (Retrieval-Augmented Generation) systems?**
**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What strategies help maintain vector database performance during peak traffic periods?**
**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you implement effective data governance and compliance measures for vector database systems?**
**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What techniques help optimize vector database performance for real-time recommendation engines?**
**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you handle the challenges of vector database testing and validation in development environments?**
**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**When should you implement custom similarity algorithms versus using database-provided functions?**
**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you optimize vector database deployments for specific use cases like image search, document retrieval, or product recommendations?**
**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**What are the considerations for implementing vector databases in privacy-sensitive applications?**
**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**How do you handle the operational challenges of managing large-scale vector database infrastructure?**
**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**What strategies help future-proof vector database architectures for evolving AI and ML requirements?**
**Answer:** _To be filled_

---
