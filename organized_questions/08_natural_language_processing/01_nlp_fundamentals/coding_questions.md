# Nlp Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Write a Python function for tokenizing text using NLTK.**

**Answer:**

**Definition:**
Tokenization splits text into meaningful units (words, sentences). NLTK provides `word_tokenize`, `sent_tokenize`, and `RegexpTokenizer` for different needs.

**Python Code Example:**
```python
# Pipeline: Raw text -> Sentence tokenization -> Word tokenization -> Custom tokenization

import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer

def tokenize_text(text, level='word'):
    """
    Tokenize text at word or sentence level using NLTK.
    
    Args:
        text: Input text string
        level: 'word', 'sentence', or 'both'
    Returns:
        List of tokens or dict with both levels
    """
    if level == 'word':
        return word_tokenize(text)
    elif level == 'sentence':
        return sent_tokenize(text)
    elif level == 'both':
        sentences = sent_tokenize(text)
        words_per_sent = [word_tokenize(s) for s in sentences]
        return {'sentences': sentences, 'words_per_sentence': words_per_sent}

# Basic usage
text = "Natural Language Processing is fascinating! It powers chatbots, translators, and more."

word_tokens = tokenize_text(text, level='word')
print(f"Word tokens: {word_tokens}")
# ['Natural', 'Language', 'Processing', 'is', 'fascinating', '!', 'It', 'powers', ...]

sent_tokens = tokenize_text(text, level='sentence')
print(f"Sentences: {sent_tokens}")
# ['Natural Language Processing is fascinating!', 'It powers chatbots, translators, and more.']

# Custom tokenizer: only alphabetic words (no punctuation)
alpha_tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
alpha_tokens = alpha_tokenizer.tokenize(text)
print(f"Alpha only: {alpha_tokens}")
# ['Natural', 'Language', 'Processing', 'is', 'fascinating', 'It', 'powers', ...]

# Advanced: tokenize with stopword removal
from nltk.corpus import stopwords
nltk.download('stopwords')

def tokenize_and_clean(text):
    """Tokenize, lowercase, and remove stopwords"""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t.isalpha() and t not in stop_words]

clean_tokens = tokenize_and_clean(text)
print(f"Cleaned: {clean_tokens}")
# ['natural', 'language', 'processing', 'fascinating', 'powers', 'chatbots', 'translators']
```

**Key Points:**
- `word_tokenize` handles punctuation and contractions (e.g., "don't" → ["do", "n't"])
- `sent_tokenize` uses Punkt tokenizer trained on English
- `RegexpTokenizer` gives full control over token patterns
- Always lowercase and remove stopwords for downstream tasks like classification

**Interview Tips:**
- NLTK's tokenizer is rule-based; for production, consider spaCy or HuggingFace tokenizers
- `word_tokenize` requires the `punkt_tab` data package
- Mention that subword tokenizers (BPE, WordPiece) are preferred for transformer models

---

## Question 2
- [ ] Done

**Implement an n-gram language model in Python from scratch.**

**Answer:**

**Definition:**
An n-gram language model estimates the probability of a word given the previous (n-1) words using maximum likelihood estimation: $P(w_n | w_1, ..., w_{n-1}) \approx \frac{C(w_1, ..., w_n)}{C(w_1, ..., w_{n-1})}$

**Python Code Example:**
```python
# Pipeline: Corpus -> Tokenize -> Count n-grams -> Estimate probabilities -> Generate/Evaluate

from collections import defaultdict, Counter
import random
import math

class NGramLanguageModel:
    def __init__(self, n=3, smoothing='laplace'):
        """
        Args:
            n: n-gram order (2=bigram, 3=trigram)
            smoothing: 'none' or 'laplace' for add-1 smoothing
        """
        self.n = n
        self.smoothing = smoothing
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = defaultdict(int)
        self.vocab = set()
    
    def _get_ngrams(self, tokens):
        """Extract n-grams from token list"""
        # Add start/end tokens
        padded = ['<s>'] * (self.n - 1) + tokens + ['</s>']
        ngrams = []
        for i in range(len(padded) - self.n + 1):
            context = tuple(padded[i:i + self.n - 1])
            word = padded[i + self.n - 1]
            ngrams.append((context, word))
        return ngrams
    
    def train(self, corpus):
        """
        Train on a list of sentences (each a list of tokens).
        """
        for sentence in corpus:
            self.vocab.update(sentence)
            for context, word in self._get_ngrams(sentence):
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1
        
        self.vocab.add('</s>')
        print(f"Trained on {len(corpus)} sentences, vocab size: {len(self.vocab)}")
    
    def probability(self, word, context):
        """P(word | context) with optional smoothing"""
        context = tuple(context[-(self.n - 1):])  # Keep only last n-1 words
        
        count_ngram = self.ngram_counts[context][word]
        count_context = self.context_counts[context]
        
        if self.smoothing == 'laplace':
            V = len(self.vocab)
            return (count_ngram + 1) / (count_context + V)
        else:
            if count_context == 0:
                return 1 / len(self.vocab)
            return count_ngram / count_context
    
    def perplexity(self, test_sentences):
        """Evaluate model using perplexity (lower = better)"""
        total_log_prob = 0
        total_words = 0
        
        for sentence in test_sentences:
            for context, word in self._get_ngrams(sentence):
                prob = self.probability(word, list(context))
                if prob > 0:
                    total_log_prob += math.log2(prob)
                total_words += 1
        
        avg_log_prob = total_log_prob / total_words
        return 2 ** (-avg_log_prob)
    
    def generate(self, max_length=20, seed=None):
        """Generate text from the model"""
        if seed:
            random.seed(seed)
        
        context = list(['<s>'] * (self.n - 1))
        generated = []
        
        for _ in range(max_length):
            ctx = tuple(context[-(self.n - 1):])
            
            if ctx not in self.ngram_counts:
                break
            
            # Weighted random choice
            candidates = self.ngram_counts[ctx]
            words = list(candidates.keys())
            weights = list(candidates.values())
            
            chosen = random.choices(words, weights=weights, k=1)[0]
            if chosen == '</s>':
                break
            
            generated.append(chosen)
            context.append(chosen)
        
        return ' '.join(generated)


# Usage
corpus = [
    "the cat sat on the mat".split(),
    "the cat ate the fish".split(),
    "the dog sat on the log".split(),
    "the dog chased the cat".split(),
    "a cat is a small animal".split(),
]

# Train trigram model
model = NGramLanguageModel(n=3, smoothing='laplace')
model.train(corpus)

# Probabilities
print(f"P('mat' | 'the', 'mat'): {model.probability('mat', ['on', 'the']):.4f}")
print(f"P('cat' | 'the',): {model.probability('cat', ['the']):.4f}")

# Perplexity
ppl = model.perplexity([["the", "cat", "sat"]])
print(f"Perplexity: {ppl:.2f}")

# Generate
print(f"Generated: {model.generate(max_length=10)}")
```

**Key Formulas:**
- Bigram: $P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$
- Laplace smoothing: $P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + V}$
- Perplexity: $PP = 2^{-\frac{1}{N}\sum \log_2 P(w_i | context)}$

**Interview Tips:**
- Always mention smoothing (Laplace, Kneser-Ney) to handle zero counts
- Perplexity is the standard evaluation metric for LMs
- N-grams suffer from data sparsity for large n; neural LMs solve this
- Start/end tokens (`<s>`, `</s>`) are essential for proper probability estimation

---

## Question 3
- [ ] Done

**Code a regular expression in Python for extracting email addresses from text.**

**Answer:**

**Definition:**
Email extraction uses regex patterns matching the RFC 5322 standard: `local-part@domain`. The local part allows letters, digits, dots, hyphens, and underscores; the domain requires valid DNS format.

**Python Code Example:**
```python
# Pipeline: Raw text -> Regex pattern matching -> Extract & validate emails

import re

def extract_emails(text):
    """
    Extract valid email addresses from text using regex.
    
    Pattern breakdown:
    [a-zA-Z0-9._%+-]+   : local part (letters, digits, special chars)
    @                     : literal @ symbol
    [a-zA-Z0-9.-]+       : domain name
    \.[a-zA-Z]{2,}       : top-level domain (2+ letters)
    """
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(pattern, text)

# Basic usage
text = """
Contact us at support@company.com or sales@company.co.uk.
John's email is john.doe+work@example.org. 
Invalid: @missing.com, noat.com, user@.com
Visit us at http://www.example.com (not an email).
"""

emails = extract_emails(text)
print(f"Found emails: {emails}")
# ['support@company.com', 'sales@company.co.uk', 'john.doe+work@example.org']

# Advanced: extract with context (who, email)
def extract_emails_with_context(text):
    """Extract emails along with surrounding context"""
    pattern = r'(\b\w+(?:\s\w+)?\s)?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    matches = re.finditer(pattern, text)
    
    results = []
    for match in matches:
        context = match.group(1).strip() if match.group(1) else 'Unknown'
        email = match.group(2)
        results.append({'context': context, 'email': email})
    
    return results

# Validation function
def is_valid_email(email):
    """Strict email validation"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

# Test validation
test_emails = ['user@example.com', 'bad@.com', 'no-at-sign', 'a@b.c', 'valid@domain.co.uk']
for e in test_emails:
    print(f"{e}: {'✓' if is_valid_email(e) else '✗'}")
```

**Pattern Explanation:**

| Part | Regex | Matches |
|------|-------|---------|
| Local part | `[a-zA-Z0-9._%+-]+` | user.name+tag |
| Separator | `@` | @ symbol |
| Domain | `[a-zA-Z0-9.-]+` | mail.example |
| TLD | `\.[a-zA-Z]{2,}` | .com, .co.uk |

**Interview Tips:**
- This pattern covers 95%+ of real-world emails but isn't fully RFC 5322 compliant
- For production, use libraries like `email-validator` for strict validation
- `\b` word boundaries help avoid partial matches
- `re.findall` returns all matches; `re.finditer` gives match objects with positions

---

## Question 4
- [ ] Done

**Design a Python function that calculates cosine similarity between two text documents.**

**Answer:**

**Definition:**
Cosine similarity measures the angle between two document vectors in a vector space. It ranges from 0 (orthogonal/unrelated) to 1 (identical direction):

$$\text{cosine\_sim}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|} = \frac{\sum a_i b_i}{\sqrt{\sum a_i^2} \times \sqrt{\sum b_i^2}}$$

**Python Code Example:**
```python
# Pipeline: Documents -> Vectorize (TF-IDF or BoW) -> Compute cosine similarity

import math
from collections import Counter

# Method 1: From scratch
def cosine_similarity(doc1, doc2):
    """
    Compute cosine similarity between two text documents.
    Uses term frequency (bag-of-words) vectors.
    """
    # Tokenize and count words
    words1 = doc1.lower().split()
    words2 = doc2.lower().split()
    
    freq1 = Counter(words1)
    freq2 = Counter(words2)
    
    # Get all unique words
    all_words = set(freq1.keys()) | set(freq2.keys())
    
    # Dot product
    dot_product = sum(freq1.get(w, 0) * freq2.get(w, 0) for w in all_words)
    
    # Magnitudes
    mag1 = math.sqrt(sum(v ** 2 for v in freq1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in freq2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)


# Method 2: Using scikit-learn with TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

def tfidf_cosine_similarity(doc1, doc2):
    """Compute cosine similarity using TF-IDF vectors"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    similarity = sklearn_cosine(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]


# Usage
doc1 = "Natural language processing is a field of artificial intelligence"
doc2 = "NLP is an area of AI that deals with language understanding"
doc3 = "The weather is sunny today and perfect for a walk"

print(f"doc1 vs doc2 (BoW):   {cosine_similarity(doc1, doc2):.4f}")
print(f"doc1 vs doc3 (BoW):   {cosine_similarity(doc1, doc3):.4f}")
print(f"doc1 vs doc2 (TF-IDF): {tfidf_cosine_similarity(doc1, doc2):.4f}")
print(f"doc1 vs doc3 (TF-IDF): {tfidf_cosine_similarity(doc1, doc3):.4f}")

# Method 3: Using sentence embeddings (semantic similarity)
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# emb1 = model.encode(doc1)
# emb2 = model.encode(doc2)
# sim = numpy.dot(emb1, emb2) / (numpy.linalg.norm(emb1) * numpy.linalg.norm(emb2))
```

**Comparison of Approaches:**

| Method | Captures Semantics | Speed | Quality |
|--------|-------------------|-------|---------|
| BoW cosine | No (exact word match) | Fast | Basic |
| TF-IDF cosine | No (weighted terms) | Fast | Better |
| Embedding cosine | Yes (semantic meaning) | Slower | Best |

**Interview Tips:**
- TF-IDF is better than raw BoW because it down-weights common words
- For semantic similarity, use sentence embeddings (SBERT, OpenAI embeddings)
- Cosine similarity is scale-invariant (document length doesn't matter)
- Range: [0, 1] for non-negative vectors; [-1, 1] for general vectors

---

## Question 5
- [ ] Done

**Implement a simple sentiment analysis classifier using a bag-of-words model and Scikit-learn.**

**Answer:**

**Definition:**
A BoW sentiment classifier converts text into word-count vectors, then trains a classifier (Naive Bayes, Logistic Regression) to predict sentiment labels (positive/negative).

**Python Code Example:**
```python
# Pipeline: Raw text -> BoW vectorization -> Train classifier -> Predict sentiment

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Sample dataset
texts = [
    "I love this movie, it was amazing and wonderful",
    "Great product, highly recommend to everyone",
    "The food was delicious and the service excellent",
    "Best experience ever, absolutely fantastic",
    "This is the worst thing I have ever bought",
    "Terrible quality, complete waste of money",
    "Horrible experience, never going back again",
    "Very disappointing and poorly made product",
    "The movie was okay, nothing special",
    "Absolutely brilliant performance and storytelling",
    "Awful customer service, very rude staff",
    "I really enjoyed the atmosphere and food",
]
labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1]  # 1=positive, 0=negative

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42
)

# Method 1: Step-by-step
vectorizer = CountVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 2)  # Unigrams + bigrams
)
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train Naive Bayes classifier
nb_clf = MultinomialNB()
nb_clf.fit(X_train_bow, y_train)
y_pred = nb_clf.predict(X_test_bow)

print("Naive Bayes Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Method 2: Using Pipeline (cleaner, avoids data leakage)
sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000
    )),
    ('classifier', LogisticRegression(max_iter=1000))
])

sentiment_pipeline.fit(X_train, y_train)
y_pred_lr = sentiment_pipeline.predict(X_test)

print("\nLogistic Regression Pipeline Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.2f}")

# Predict new text
new_texts = ["This product is amazing!", "Worst purchase ever, total garbage"]
predictions = sentiment_pipeline.predict(new_texts)
for text, pred in zip(new_texts, predictions):
    label = "Positive" if pred == 1 else "Negative"
    print(f"'{text}' -> {label}")

# Feature importance (top words per class)
if hasattr(sentiment_pipeline.named_steps['classifier'], 'coef_'):
    feature_names = sentiment_pipeline.named_steps['tfidf'].get_feature_names_out()
    coefs = sentiment_pipeline.named_steps['classifier'].coef_[0]
    top_positive = sorted(zip(coefs, feature_names), reverse=True)[:5]
    top_negative = sorted(zip(coefs, feature_names))[:5]
    print(f"\nTop positive words: {[w for _, w in top_positive]}")
    print(f"Top negative words: {[w for _, w in top_negative]}")
```

**Model Comparison:**

| Model | Pros | Cons |
|-------|------|------|
| Naive Bayes | Fast, works well with small data | Assumes feature independence |
| Logistic Regression | Interpretable, strong baseline | Linear decision boundary |
| SVM | Good for high-dimensional text | Slower training |

**Interview Tips:**
- Always use Pipeline to prevent data leakage (vectorizer fit on train only)
- TF-IDF typically outperforms raw CountVectorizer
- Bigrams (`ngram_range=(1,2)`) capture phrases like "not good"
- For real applications, use pre-trained models (BERT, DistilBERT) for much better accuracy
- Mention class imbalance handling: `class_weight='balanced'`

---

