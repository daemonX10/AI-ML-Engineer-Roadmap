# Naive Bayes Interview Questions - General Questions

## Question 1
- [ ] Done

**Why is the Naive Bayes classifier a good choice for text classification tasks?**

**Answer:**

Naive Bayes excels at text classification because it handles high-dimensional sparse data efficiently, works well with limited training data, and its independence assumption is "good enough" for word-based features. Training is fast (just counting), and it scales linearly with vocabulary size. It's the go-to baseline for spam detection, sentiment analysis, and document categorization.

**Reasons NB Works Well for Text:**

| Property | Why It Helps |
|----------|--------------|
| High-dimensionality | Vocabulary = thousands of features; NB handles this naturally |
| Sparsity | Most words absent in each doc; NB works with sparse matrices |
| Limited data | NB needs fewer samples than discriminative models |
| Fast training | Just counting - processes millions of docs quickly |
| Independence assumption | Words being treated independently often works well enough |

**Practical Evidence:**
- First successful spam filters used NB (late 1990s)
- Often competitive with more complex models
- Strong baseline that's hard to beat without deep learning

---

## Question 2
- [ ] Done

**Why do we often use log probabilities instead of probabilities in Naive Bayes computation?**

**Answer:**

We use log probabilities to avoid numerical underflow. Multiplying many small probabilities (e.g., 0.001 × 0.002 × ...) quickly approaches zero, causing floating-point underflow. Taking logs converts multiplication to addition: log(a×b) = log(a) + log(b). This keeps values in a manageable range and is more numerically stable.

**The Problem:**

```python
# Multiplying 100 small probabilities
prob = 0.01 ** 100  # = 1e-200, underflows to 0.0 in floating point
```

**The Solution:**

$$\log P(Y|X) = \log P(Y) + \sum_{i=1}^{n} \log P(X_i|Y)$$

```python
# In log space - addition instead of multiplication
log_prob = 100 * np.log(0.01)  # = -460.5, perfectly representable
```

**Key Points:**
- Log is monotonic: if P(A) > P(B), then log P(A) > log P(B)
- Classification only needs to compare, not compute exact probability
- If exact probability needed: P = exp(log_prob), but often not necessary

---

## Question 3
- [ ] Done

**What role does Laplace smoothing (additive smoothing) play in Naive Bayes?**

**Answer:**

Laplace smoothing solves the zero-frequency problem - when a feature-class combination never appears in training, its probability is zero, making the entire posterior zero. By adding a small count (typically 1) to every feature, we ensure no probability is zero. This allows the model to handle unseen feature combinations gracefully.

**The Formula (Laplace Smoothing):**

$$P(word|class) = \frac{count(word, class) + \alpha}{count(class) + \alpha \times |V|}$$

Where:
- α = smoothing parameter (typically 1)
- |V| = vocabulary size

**Choosing α:**
- α = 1: Laplace smoothing (add-one)
- α < 1: Lidstone smoothing (less aggressive)
- α > 1: Heavy smoothing (for very large vocabularies)

---

## Question 4
- [ ] Done

**Can Naive Bayes be used for regression tasks? Why or why not?**

**Answer:**

Standard Naive Bayes is not designed for regression (predicting continuous values) - it's fundamentally a classifier that outputs discrete class probabilities. However, the Bayesian principle can be extended to regression through Gaussian Processes or Bayesian Linear Regression, but these are different algorithms, not "Naive Bayes regression."

**Why NB is Classification-Only:**

1. **Output is Probability Over Classes:**
   - NB computes P(Y=c|X) for discrete classes
   - There's no concept of P(Y=3.7|X) for continuous Y

2. **Bayes Theorem for Classes:**
   - Posterior P(Y|X) is a probability distribution over classes
   - For regression, we need to predict a value, not a class

**Bottom Line:**
If you need regression, use Linear Regression, Random Forest Regressor, or other regression algorithms - not Naive Bayes.

---

## Question 5
- [ ] Done

**In what kind of situations might the 'naivety' assumption of Naive Bayes lead to poor performance?**

**Answer:**

The independence assumption fails badly when features are highly correlated - the model "double counts" evidence. Examples: pixel values in images (nearby pixels correlated), time-series features (autocorrelation), engineered features (feature and its square), or any domain where features naturally interact.

**Situations Where NB Struggles:**

| Scenario | Why Independence Fails |
|----------|----------------------|
| Image classification | Neighboring pixels highly correlated |
| Sequential data | Time-series features autocorrelated |
| Feature interactions | XOR-like patterns require interactions |
| Redundant features | Same info in multiple features |

**What to Do:**
- Remove redundant/highly correlated features
- Use PCA for decorrelation
- Consider discriminative models (Logistic Regression, SVM)

---

## Question 6
- [ ] Done

**What preprocessing steps would you take for text data before applying Naive Bayes?**

**Answer:**

Standard text preprocessing pipeline: (1) Lowercase conversion, (2) Remove punctuation/special characters, (3) Tokenization, (4) Remove stopwords, (5) Stemming/Lemmatization, (6) Create feature vectors (TF-IDF or BoW). Optional: handle negations, remove rare/common words, n-gram extraction.

**Preprocessing Pipeline:**

```python
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove special characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Tokenize
    tokens = text.split()
    
    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # 5. Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    
    return ' '.join(tokens)

# 6. Create TF-IDF features
vectorizer = TfidfVectorizer(preprocessor=preprocess_text, ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)
```

---

## Question 7
- [ ] Done

**What metrics would you use to evaluate the performance of a Naive Bayes classification model?**

**Answer:**

Use accuracy for balanced classes, but prefer precision/recall/F1 for imbalanced data. Since NB outputs probabilities, also evaluate with log-loss (probability quality) and ROC-AUC (ranking ability). Confusion matrix shows error patterns.

**Metrics by Use Case:**

| Metric | When to Use |
|--------|-------------|
| Accuracy | Balanced classes |
| Precision | FP costly (spam marking ham) |
| Recall | FN costly (missing fraud) |
| F1-Score | Balance precision/recall |
| ROC-AUC | Ranking quality |
| Log-Loss | Probability accuracy |

**Implementation:**

```python
from sklearn.metrics import classification_report, roc_auc_score, log_loss

y_pred = nb.predict(X_test)
y_proba = nb.predict_proba(X_test)

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba[:, 1]):.4f}")
print(f"Log-Loss: {log_loss(y_test, y_proba):.4f}")
```

---

## Question 8
- [ ] Done

**Compare and contrast Naive Bayes with logistic regression.**

**Answer:**

Both are linear classifiers but differ fundamentally: NB is generative (models P(X|Y)), LR is discriminative (models P(Y|X) directly). NB assumes feature independence; LR doesn't. NB trains by counting (fast), LR by optimization (slower). With small data, NB often wins; with large data, LR typically performs better.

**Comparison Table:**

| Aspect | Naive Bayes | Logistic Regression |
|--------|-------------|---------------------|
| Type | Generative | Discriminative |
| Assumption | Feature independence | None explicit |
| Training | Counting (fast) | Optimization (slower) |
| Small data | Often better | May overfit |
| Large data | LR usually wins | Usually better |
| Correlated features | Performance degrades | Handles better |
| Calibration | Poor (extreme probs) | Better calibrated |

**When to Choose:**
- **NB**: Quick baseline, text data, limited samples, real-time needs
- **LR**: Need accurate probabilities, correlated features, more data available

---

## Question 9
- [ ] Done

**How can you use the Naive Bayes classifier in a semi-supervised learning scenario?**

**Answer:**

Semi-supervised NB leverages both labeled and unlabeled data through Expectation-Maximization (EM). Start by training NB on labeled data, then iterate: (E-step) predict soft labels for unlabeled data, (M-step) retrain using both labeled and probabilistically-weighted unlabeled data.

**EM Algorithm for Semi-Supervised NB:**

```
1. Initialize: Train NB on labeled data only
2. Repeat until convergence:
   - E-step: Compute P(Y|X) for each unlabeled sample
   - M-step: Re-estimate parameters using both labeled and unlabeled data
```

**Simplified Implementation (Self-Training):**

```python
def semi_supervised_nb(X_labeled, y_labeled, X_unlabeled, confidence_threshold=0.9):
    nb = MultinomialNB()
    nb.fit(X_labeled, y_labeled)
    
    for _ in range(10):  # iterations
        probs = nb.predict_proba(X_unlabeled)
        confident_mask = probs.max(axis=1) >= confidence_threshold
        
        if not confident_mask.any():
            break
            
        X_labeled = np.vstack([X_labeled, X_unlabeled[confident_mask]])
        y_labeled = np.concatenate([y_labeled, nb.predict(X_unlabeled[confident_mask])])
        X_unlabeled = X_unlabeled[~confident_mask]
        
        nb.fit(X_labeled, y_labeled)
    
    return nb
```

---

## Question 10
- [ ] Done

**How would a Naive Bayes classifier identify fake news articles?**

**Answer:**

Fake news detection with NB learns linguistic and stylistic patterns that distinguish fake from real news. Features include: word patterns (sensational language, clickbait), source credibility markers, writing style (excessive punctuation, all caps), claim verifiability indicators.

**Implementation:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

fake_news_detector = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
    ('nb', MultinomialNB(alpha=0.1))
])

fake_news_detector.fit(articles, labels)  # 1=fake, 0=real
```

**Fake News Indicators NB Can Learn:**
- Sensational words: "shocking", "unbelievable", "secret"
- Emotional appeals over facts
- Clickbait patterns in headlines

---

## Question 11
- [ ] Done

**Explore the challenges and solutions for Naive Bayes classification in the context of multi-label classification tasks.**

**Answer:**

Multi-label classification assigns multiple labels to each sample. Challenge: standard NB predicts one class. Solutions: (1) Binary Relevance - train separate NB per label, (2) Label Powerset - treat each label combination as unique class, (3) Classifier Chains - chain NB classifiers to model label dependencies.

**Approaches:**

**1. Binary Relevance (Most Common):**
```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

multi_label_nb = OneVsRestClassifier(MultinomialNB())
multi_label_nb.fit(X_train, Y_train)  # Y is multi-label indicator matrix
```

**2. Classifier Chains:**
```python
from sklearn.multioutput import ClassifierChain

chain = ClassifierChain(MultinomialNB(), order='random')
chain.fit(X_train, Y_train)
```

| Method | Pros | Cons |
|--------|------|------|
| Binary Relevance | Simple, scalable | Ignores label correlations |
| Label Powerset | Captures correlations | Exponential combinations |
| Classifier Chains | Models dependencies | Order-dependent |

---

## Question 12
- [ ] Done

**How can active learning algorithms benefit from the Naive Bayes classifier in data-scarce scenarios?**

**Answer:**

Active learning selects the most informative samples for labeling, maximizing learning from minimal labels. NB benefits active learning by: (1) providing probability scores for uncertainty sampling, (2) fast retraining after each new label, (3) working well with small initial labeled sets.

**Active Learning Loop with NB:**

```python
def active_learning_nb(X_pool, X_initial, y_initial, n_queries=100):
    X_labeled = X_initial.copy()
    y_labeled = y_initial.copy()
    X_unlabeled = X_pool.copy()
    
    nb = MultinomialNB()
    
    for _ in range(n_queries):
        nb.fit(X_labeled, y_labeled)
        probs = nb.predict_proba(X_unlabeled)
        
        # Uncertainty sampling: select most uncertain
        uncertainty = 1 - np.max(probs, axis=1)
        query_idx = np.argmax(uncertainty)
        
        # Get label from oracle (human)
        y_query = oracle_label(X_unlabeled[query_idx])
        
        # Update sets
        X_labeled = np.vstack([X_labeled, X_unlabeled[query_idx]])
        y_labeled = np.append(y_labeled, y_query)
        X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0)
    
    return nb
```

**Why NB is Good for Active Learning:**
- Fast retraining (essential for iterative process)
- Probability scores enable uncertainty sampling
- Works with very small initial labeled sets

---
