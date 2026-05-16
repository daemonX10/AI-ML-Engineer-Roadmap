# Classification Algorithms Interview Questions - General Questions

## Question 1
- [ ] Done

**Can you differentiate between binary and multiclass classification?**

### Answer

**Definition:**
Binary classification predicts between exactly 2 classes (yes/no, spam/not spam). Multiclass classification predicts among 3+ mutually exclusive classes (cat/dog/bird, digits 0-9).

**Key Differences:**

| Aspect | Binary | Multiclass |
|--------|--------|------------|
| Classes | 2 | 3 or more |
| Output | Single probability | Probability per class |
| Activation | Sigmoid | Softmax |
| Loss | Binary Cross-Entropy | Categorical Cross-Entropy |
| Metrics | Precision, Recall, F1 | Macro/Micro F1 |

**Examples:**

| Type | Example | Classes |
|------|---------|---------|
| Binary | Spam detection | Spam, Not Spam |
| Binary | Disease diagnosis | Positive, Negative |
| Multiclass | Digit recognition | 0, 1, 2, ..., 9 |
| Multiclass | Sentiment | Positive, Neutral, Negative |

**Multiclass Strategies:**

| Strategy | Description |
|----------|-------------|
| One-vs-Rest (OvR) | Train N binary classifiers, each class vs all others |
| One-vs-One (OvO) | Train N(N-1)/2 classifiers, every pair of classes |
| Native | Algorithm handles multiclass directly (softmax, trees) |

**Interview Tips:**
- Sigmoid outputs 1 probability; Softmax outputs N probabilities summing to 1
- Multi-label ≠ Multiclass (multi-label: multiple labels per sample)
- OvR is common in sklearn for binary classifiers
- Trees naturally handle multiclass

---

## Question 2
- [ ] Done

**How do you deal with unbalanced datasets in classification?**

### Answer

**Definition:**
Imbalanced datasets have unequal class distributions (e.g., 95% negative, 5% positive). This causes models to favor the majority class, poor minority class predictions, and misleading accuracy.

**Techniques to Handle Imbalance:**

| Category | Technique | Description |
|----------|-----------|-------------|
| **Data-level** | Oversampling | Duplicate/generate minority samples |
| | SMOTE | Synthetic samples via interpolation |
| | Undersampling | Remove majority samples |
| | Combined | SMOTE + Tomek links |
| **Algorithm-level** | Class weights | Penalize majority class errors less |
| | Threshold tuning | Adjust decision threshold |
| | Ensemble | Balanced Random Forest |
| **Metric-level** | Use appropriate metrics | Precision, Recall, F1, PR-AUC |

**Python Example:**
```python
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Method 1: Class weights
weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
model = RandomForestClassifier(class_weight='balanced')

# Method 2: SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Method 3: Threshold tuning
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.3).astype(int)  # Lower threshold for minority
```

**When to Use What:**

| Scenario | Recommendation |
|----------|----------------|
| Slight imbalance (70:30) | Class weights |
| Moderate (90:10) | SMOTE or class weights |
| Severe (99:1) | Combine SMOTE + undersampling |
| Very few minority samples | Collect more data |

**Interview Tips:**
- Never use accuracy alone on imbalanced data
- SMOTE on training data only, not test
- Class weights are simplest and often sufficient
- Mention PR-AUC over ROC-AUC for severe imbalance

---

## Question 3
- [ ] Done

**What techniques can be used to prevent overfitting in classification models?**

### Answer

**Definition:**
Overfitting occurs when a model learns training data noise, performing well on training but poorly on unseen data. Prevention techniques constrain model complexity or increase effective training data.

**Techniques by Category:**

| Category | Technique | How It Helps |
|----------|-----------|--------------|
| **Regularization** | L1/L2 | Penalize large weights |
| | Dropout | Randomly drop neurons |
| | Early stopping | Stop when validation worsens |
| **Data** | More data | Reduces overfitting naturally |
| | Data augmentation | Generate variations |
| | Cross-validation | Better generalization estimate |
| **Model** | Simpler model | Fewer parameters |
| | Pruning (trees) | Remove unnecessary branches |
| | Ensemble | Combine multiple models |

**Algorithm-Specific:**

| Algorithm | Technique |
|-----------|-----------|
| Decision Tree | max_depth, min_samples_split, pruning |
| Neural Network | Dropout, L2, early stopping, batch norm |
| Logistic Regression | L1/L2 regularization (C parameter) |
| SVM | C parameter (smaller = more regularization) |
| Random Forest | n_estimators, max_features |

**Python Example:**
```python
# Regularization in Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=0.1, penalty='l2')  # Smaller C = stronger regularization

# Tree constraints
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)

# Early stopping in Neural Network
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```

**Interview Tips:**
- Mention train-test split and cross-validation first
- Regularization is most common answer
- Early stopping is easy win for neural networks
- More data is best solution when available

---

## Question 4
- [ ] Done

**Compare and contrast shallow and deep learning classifiers.**

### Answer

**Definition:**
Shallow learning uses traditional ML with manual feature engineering (SVM, Random Forest). Deep learning uses neural networks with multiple layers that automatically learn hierarchical features.

**Comparison:**

| Aspect | Shallow Learning | Deep Learning |
|--------|------------------|---------------|
| Feature Engineering | Manual, expert-driven | Automatic, learned |
| Data Requirements | Small-medium | Large datasets |
| Interpretability | Higher | Lower (black box) |
| Training Time | Fast | Slow (GPU needed) |
| Compute | CPU sufficient | GPU/TPU required |
| Overfitting Risk | Lower | Higher (more params) |

**When to Use:**

| Use Shallow When | Use Deep When |
|------------------|---------------|
| Small dataset (<10K) | Large dataset (>100K) |
| Tabular data | Images, text, audio |
| Need interpretability | Performance is priority |
| Limited compute | GPU available |
| Quick prototyping | State-of-the-art needed |

**Performance by Domain:**

| Domain | Shallow | Deep |
|--------|---------|------|
| Tabular | Often wins (XGBoost) | May overfit |
| Images | HOG+SVM (weak) | CNN (strong) |
| Text | TF-IDF+NB (baseline) | Transformers (SOTA) |
| Time Series | Features+RF | LSTM/Transformer |

**Examples:**

| Shallow | Deep |
|---------|------|
| Logistic Regression | Neural Network |
| SVM | CNN, RNN |
| Random Forest | ResNet, BERT |
| XGBoost | Transformer |

**Interview Tips:**
- Deep learning doesn't always win (tabular: XGBoost)
- Feature engineering skill still valuable
- Transfer learning bridges the data gap
- Mention computational cost as real constraint
- Ensemble of both can be powerful

---

## Question 5
- [ ] Done

**How do decision tree splitting criteria like Gini impurity and entropy affect the model?**

### Answer

**Definition:**
Splitting criteria measure node impurity to determine optimal splits. Gini impurity and Entropy are most common. They affect how the tree structure develops and computational efficiency.

**Mathematical Formulas:**

| Criterion | Formula | Range |
|-----------|---------|-------|
| Gini | $1 - \sum p_i^2$ | 0 to 0.5 (binary) |
| Entropy | $-\sum p_i \log_2(p_i)$ | 0 to 1 (binary) |

**Comparison:**

| Aspect | Gini | Entropy |
|--------|------|---------|
| Computation | Faster (no log) | Slower |
| Range | [0, 0.5] | [0, 1] |
| Sensitivity | Prefers larger partitions | More balanced splits |
| Default | sklearn default | Information theory based |

**Effect on Model:**

| Factor | Impact |
|--------|--------|
| Tree Structure | Slightly different splits |
| Final Performance | Usually similar |
| Training Speed | Gini faster |
| Interpretability | Both equally interpretable |

**Visual Comparison:**
```
At p = 0.5 (max impurity):
- Gini = 0.5
- Entropy = 1.0

At p = 0 or 1 (pure):
- Both = 0
```

**Python Example:**
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# Gini (default)
tree_gini = DecisionTreeClassifier(criterion='gini')
tree_gini.fit(X, y)

# Entropy
tree_entropy = DecisionTreeClassifier(criterion='entropy')
tree_entropy.fit(X, y)

# Usually similar accuracy
print(f"Gini depth: {tree_gini.get_depth()}")
print(f"Entropy depth: {tree_entropy.get_depth()}")
```

**Interview Tips:**
- In practice, both give similar results
- Gini is computationally faster (no logarithm)
- Entropy has information-theoretic interpretation
- sklearn uses Gini by default
- Focus on max_depth, min_samples for tuning

---

## Question 6
- [ ] Done

**How do Convolutional Neural Networks (CNNs) differ from regular Neural Networks in classification tasks related to images?**

### Answer

**Definition:**
CNNs use convolutional layers with local receptive fields and weight sharing, designed specifically for spatial data. Regular NNs (MLPs) use fully connected layers treating all inputs independently, losing spatial structure.

**Key Differences:**

| Aspect | Regular NN (MLP) | CNN |
|--------|------------------|-----|
| Input handling | Flatten to 1D | Preserve 2D/3D structure |
| Connections | Fully connected | Local (kernel-based) |
| Parameters | Many (width × height × depth) | Few (kernel size) |
| Spatial info | Lost | Preserved |
| Translation invariance | No | Yes |

**CNN Architecture Components:**

| Layer | Purpose | Output |
|-------|---------|--------|
| Convolutional | Extract local features | Feature maps |
| Pooling | Reduce dimensions | Smaller feature maps |
| Activation (ReLU) | Non-linearity | Same shape |
| Fully Connected | Final classification | Class scores |

**Why CNN for Images:**

| Advantage | Explanation |
|-----------|-------------|
| Local patterns | Detects edges, textures locally |
| Weight sharing | Same filter across image |
| Hierarchical | Low → high level features |
| Translation invariant | Cat detected anywhere |
| Fewer parameters | Kernel vs full connection |

**Parameter Comparison (100×100 image):**

| Architecture | Parameters (first layer) |
|--------------|-------------------------|
| MLP (100 hidden) | 100×100×3×100 = 3M |
| CNN (32 3×3 filters) | 3×3×3×32 = 864 |

**Python Example:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# MLP for images (not recommended)
mlp = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

# CNN for images (recommended)
cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

print(f"MLP params: {mlp.count_params():,}")
print(f"CNN params: {cnn.count_params():,}")
```

**Interview Tips:**
- CNNs preserve spatial relationships
- Weight sharing dramatically reduces parameters
- Pooling provides translation invariance
- MLPs would overfit on images
- Mention transfer learning (pretrained CNNs)

---

## Question 7
- [ ] Done

**How has the field of Natural Language Processing evolved with advancements in classification models?**

### Answer

**Definition:**
NLP text classification evolved from rule-based systems → statistical ML (Naive Bayes, SVM) → neural networks (RNN, LSTM) → transformers (BERT, GPT). Each era brought improved accuracy and reduced manual feature engineering.

**Evolution Timeline:**

| Era | Approach | Characteristics |
|-----|----------|-----------------|
| 1990s | Rule-based | Hand-crafted rules |
| 2000s | Statistical ML | TF-IDF + Naive Bayes/SVM |
| 2010s | Neural (Word2Vec) | Dense embeddings + RNN |
| 2017+ | Transformers | Attention, transfer learning |
| 2020+ | Large Language Models | GPT, few-shot learning |

**Key Milestones:**

| Year | Innovation | Impact |
|------|------------|--------|
| 2003 | TF-IDF + SVM | Strong baseline |
| 2013 | Word2Vec | Semantic embeddings |
| 2017 | Transformer | Parallel training, attention |
| 2018 | BERT | Bidirectional context |
| 2020 | GPT-3 | Few-shot learning |

**Paradigm Shifts:**

| Aspect | Traditional | Modern (Transformers) |
|--------|-------------|----------------------|
| Features | Manual (TF-IDF, n-grams) | Learned embeddings |
| Context | Limited (bag of words) | Full sequence context |
| Training | Task-specific | Pre-train + fine-tune |
| Data needs | Small per task | Large pre-training, small fine-tuning |

**Current State:**

| Task | Best Approach |
|------|---------------|
| Sentiment | Fine-tuned BERT |
| Topic classification | BERT or GPT embeddings |
| Named Entity | BERT + CRF |
| Question Answering | Large Language Models |

**Python Example (Evolution):**
```python
# Era 1: TF-IDF + Naive Bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(texts)
model = MultinomialNB().fit(X, labels)

# Era 2: Word2Vec + Neural Network
# from gensim.models import Word2Vec
# word_model = Word2Vec(sentences, vector_size=100)

# Era 3: Transformers (BERT)
from transformers import BertTokenizer, BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
```

**Interview Tips:**
- Transformers replaced RNNs due to parallelization and attention
- Transfer learning is the key innovation (pre-train once, fine-tune many)
- BERT is bidirectional, GPT is unidirectional
- Mention trade-off: performance vs computational cost
- LLMs enable few-shot and zero-shot classification

---
