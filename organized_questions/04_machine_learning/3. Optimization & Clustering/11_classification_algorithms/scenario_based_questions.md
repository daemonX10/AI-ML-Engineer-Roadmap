# Classification Algorithms Interview Questions - Scenario_Based Questions

## Question 1
- [ ] Done

**How would you handle categorical features in a classification problem?**

### Answer

**Definition:**
Categorical features are non-numeric variables (like color, city, gender) that need encoding before ML algorithms can process them. The encoding choice depends on cardinality, ordinality, and the algorithm being used.

**Core Concepts:**
- Nominal: No order (red, blue, green)
- Ordinal: Has order (low, medium, high)
- Cardinality: Number of unique values
- Tree-based models handle categories differently than linear models

**Encoding Methods:**

| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| One-Hot | Low cardinality, nominal | No ordinal assumption | High dimensionality |
| Label Encoding | Ordinal, tree-based | Simple, compact | Implies false order |
| Target Encoding | High cardinality | Compact, captures relationship | Data leakage risk |
| Frequency Encoding | High cardinality | Simple, no leakage | Loses uniqueness |

**Decision Process:**
```
Is feature ordinal?
├── Yes → Label Encoding (map to 0, 1, 2...)
└── No → Check cardinality
    ├── Low (<10) → One-Hot Encoding
    └── High (>10) → Target Encoding or Frequency Encoding
```

**Python Code Example:**
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder

df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red'],
    'size': ['S', 'M', 'L', 'XL'],
    'city': ['NYC', 'LA', 'Chicago', 'NYC'],
    'target': [1, 0, 1, 0]
})

# 1. One-Hot Encoding (low cardinality, nominal)
df_onehot = pd.get_dummies(df, columns=['color'], drop_first=True)

# 2. Label Encoding (ordinal)
size_map = {'S': 0, 'M': 1, 'L': 2, 'XL': 3}
df['size_encoded'] = df['size'].map(size_map)

# 3. Target Encoding (high cardinality)
encoder = TargetEncoder(cols=['city'])
df['city_encoded'] = encoder.fit_transform(df['city'], df['target'])

# 4. Frequency Encoding
freq = df['city'].value_counts(normalize=True)
df['city_freq'] = df['city'].map(freq)
```

**Interview Tips:**
- Always mention data leakage risk with target encoding (use CV folds)
- Tree-based models (RF, XGBoost) can handle label encoding even for nominal
- One-hot creates N-1 columns (drop_first=True) to avoid multicollinearity
- For very high cardinality (>100), consider embedding layers

---

## Question 2
- [ ] Done

**How would you select the appropriate metrics for evaluating a classification model?**

### Answer

**Definition:**
Metric selection depends on the problem type (binary/multi-class), class distribution (balanced/imbalanced), and business requirements (cost of FP vs FN). No single metric fits all scenarios.

**Decision Framework:**

| Scenario | Recommended Metric | Reason |
|----------|-------------------|--------|
| Balanced classes | Accuracy, F1 | All classes equally important |
| Imbalanced classes | Precision, Recall, F1, ROC-AUC | Accuracy is misleading |
| FP costly (spam filter) | Precision | Minimize false positives |
| FN costly (cancer detection) | Recall | Minimize false negatives |
| Probability ranking | ROC-AUC, PR-AUC | Threshold-independent |
| Multi-class | Macro/Micro F1 | Aggregated performance |

**Key Metrics:**

| Metric | Formula | Use When |
|--------|---------|----------|
| Precision | TP / (TP + FP) | FP is costly |
| Recall | TP / (TP + FN) | FN is costly |
| F1-Score | 2 × (P × R) / (P + R) | Balance P and R |
| ROC-AUC | Area under ROC | Compare models, ranking |
| PR-AUC | Area under PR curve | Highly imbalanced |

**Selection Process:**
```
Step 1: Identify problem type (binary/multi-class)
Step 2: Check class balance
Step 3: Determine business cost (FP vs FN)
Step 4: Choose primary metric
Step 5: Select secondary metrics for monitoring
```

**Python Code Example:**
```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report)

y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0]
y_prob = [0.9, 0.2, 0.4, 0.8, 0.3, 0.7, 0.6, 0.1]

# Balanced data → Accuracy, F1
print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
print(f"F1: {f1_score(y_true, y_pred):.3f}")

# Imbalanced data → Focus on Precision/Recall
print(f"Precision: {precision_score(y_true, y_pred):.3f}")
print(f"Recall: {recall_score(y_true, y_pred):.3f}")

# Ranking performance → ROC-AUC
print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.3f}")

# Full report
print(classification_report(y_true, y_pred))
```

**Interview Tips:**
- Always ask "What's the cost of FP vs FN?"
- For imbalanced data, never use accuracy alone
- ROC-AUC can be misleading for highly imbalanced data; use PR-AUC
- Macro F1 treats all classes equally; Micro F1 weights by support

---

## Question 3
- [ ] Done

**Discuss the process of feature engineering and its importance in classification.**

### Answer

**Definition:**
Feature engineering transforms raw data into informative features that improve classification model performance. It bridges domain knowledge with ML algorithms, often having more impact than algorithm choice.

**Why Important:**
- Good features > Complex algorithms
- Reduces need for deep models
- Improves interpretability
- Captures domain knowledge

**Feature Engineering Process:**

| Step | Activity | Example |
|------|----------|---------|
| 1. Understand Data | EDA, domain research | Analyze distributions |
| 2. Handle Missing | Impute, flag | Create "is_missing" indicator |
| 3. Encode Categories | One-hot, target encode | Convert city to numeric |
| 4. Create Features | Combine, transform | age_group from age |
| 5. Scale Features | Normalize, standardize | For distance-based models |
| 6. Select Features | Filter, wrapper | Remove low-importance |

**Common Feature Engineering Techniques:**

| Technique | When to Use | Example |
|-----------|-------------|---------|
| Binning | Continuous → Categorical | age → age_group |
| Interaction | Non-linear relationships | price × quantity |
| Polynomial | Capture curves | x, x², x³ |
| Aggregation | Group-level info | avg_spend_per_category |
| Date extraction | Time features | day_of_week, month |
| Text features | NLP | word_count, TF-IDF |

**Python Code Example:**
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({
    'age': [25, 45, 35, None, 55],
    'income': [50000, 80000, 60000, 70000, 90000],
    'purchase_date': pd.to_datetime(['2024-01-15', '2024-02-20', '2024-01-10', '2024-03-05', '2024-02-28']),
    'category': ['A', 'B', 'A', 'C', 'B']
})

# 1. Handle missing with indicator
df['age_missing'] = df['age'].isna().astype(int)
df['age'] = df['age'].fillna(df['age'].median())

# 2. Create bins
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['young', 'mid', 'senior'])

# 3. Interaction features
df['income_per_age'] = df['income'] / df['age']

# 4. Date features
df['day_of_week'] = df['purchase_date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# 5. Aggregation (group statistics)
df['mean_income_by_category'] = df.groupby('category')['income'].transform('mean')

# 6. Scale numeric features
scaler = StandardScaler()
df[['income_scaled', 'age_scaled']] = scaler.fit_transform(df[['income', 'age']])

print(df.head())
```

**Interview Tips:**
- Feature engineering is iterative, not one-time
- Always validate new features with cross-validation
- Domain knowledge is crucial for good features
- Tree models need less feature engineering than linear models
- Automate feature engineering with libraries like Featuretools

---

## Question 4
- [ ] Done

**How would you approach a text classification task?**

### Answer

**Definition:**
Text classification assigns predefined labels to text documents using NLP techniques. It combines text preprocessing, feature extraction (vectorization), and ML/DL classifiers to categorize documents like spam detection, sentiment analysis, or topic classification.

**Text Classification Pipeline:**

| Step | Purpose | Techniques |
|------|---------|------------|
| 1. Text Cleaning | Remove noise | Lowercase, remove punctuation |
| 2. Tokenization | Split into words | word_tokenize, spaCy |
| 3. Normalization | Standardize | Stemming, Lemmatization |
| 4. Stop Word Removal | Remove common words | "the", "is", "at" |
| 5. Vectorization | Text → Numbers | TF-IDF, Word2Vec |
| 6. Model Training | Learn patterns | Naive Bayes, BERT |

**Vectorization Methods:**

| Method | Type | When to Use |
|--------|------|-------------|
| Bag of Words | Count-based | Simple, baseline |
| TF-IDF | Weighted count | Reduce common word impact |
| Word2Vec | Dense embedding | Capture semantics |
| BERT | Contextual | State-of-the-art, more data |

**Model Selection for Text:**

| Model | Pros | Cons |
|-------|------|------|
| Naive Bayes | Fast, good baseline | Assumes independence |
| Logistic Regression | Interpretable | Limited to linear |
| SVM | Works well with TF-IDF | Slow on large data |
| LSTM/GRU | Captures sequence | Needs more data |
| BERT/Transformers | Best accuracy | Computationally heavy |

**Python Code Example:**
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import re

# Sample data
texts = ["Great product, love it!", "Terrible quality, waste", 
         "Amazing service, recommend", "Very disappointed", "Good value"]
labels = [1, 0, 1, 0, 1]  # 1=positive, 0=negative

# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

texts_clean = [preprocess(t) for t in texts]

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf.fit_transform(texts_clean)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict new text
new_text = ["This is wonderful!"]
new_vec = tfidf.transform([preprocess(new_text[0])])
print(f"Prediction: {model.predict(new_vec)[0]}")
```

**Interview Tips:**
- Start simple: TF-IDF + Naive Bayes as baseline
- Preprocessing quality > model complexity
- Handle class imbalance with oversampling/class weights
- Use pre-trained embeddings for small datasets
- Consider computational constraints for model choice

---

## Question 5
- [ ] Done

**Discuss the use of classification algorithms in image recognition.**

### Answer

**Definition:**
Image classification assigns labels to images based on visual content. Traditional approaches used handcrafted features (HOG, SIFT), while modern methods use Convolutional Neural Networks (CNNs) that automatically learn hierarchical features from pixels.

**Evolution of Image Classification:**

| Era | Approach | Example |
|-----|----------|---------|
| Traditional | Handcrafted features + ML | HOG + SVM |
| Early DL | Basic CNNs | LeNet, AlexNet |
| Modern DL | Deep CNNs | VGG, ResNet, EfficientNet |
| Current | Transfer Learning | Pre-trained + Fine-tune |

**CNN Architecture for Classification:**

| Layer Type | Purpose | Example |
|------------|---------|---------|
| Convolutional | Extract features | Edges, textures, shapes |
| Pooling | Reduce dimensions | Max pooling 2×2 |
| Fully Connected | Classification | Final decision layer |
| Softmax | Output probabilities | Multi-class output |

**Key Concepts:**

| Concept | Description |
|---------|-------------|
| Feature Hierarchy | Low-level → High-level (edges → objects) |
| Transfer Learning | Use pre-trained weights, fine-tune |
| Data Augmentation | Rotation, flip, crop for more data |
| Batch Normalization | Stabilize training |

**When to Use What:**

| Scenario | Approach |
|----------|----------|
| Small dataset (<1000) | Transfer learning, heavy augmentation |
| Medium dataset | Fine-tune pre-trained CNN |
| Large dataset | Train from scratch or fine-tune |
| Limited compute | MobileNet, EfficientNet-B0 |

**Python Code Example:**
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Method 1: Simple CNN from scratch
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Method 2: Transfer Learning with VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base

transfer_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)
```

**Interview Tips:**
- Transfer learning is almost always better for small datasets
- Data augmentation is crucial for generalization
- Deeper isn't always better; consider ResNet skip connections
- For production: balance accuracy vs inference speed
- Mention ImageNet pre-training as standard practice

---

## Question 6
- [ ] Done

**How would you handle time-series data for a classification task?**

### Answer

**Definition:**
Time-series classification assigns labels to sequences of data points ordered in time. Unlike standard classification, it must preserve temporal order and capture patterns like trends, seasonality, and temporal dependencies (e.g., activity recognition, anomaly detection).

**Key Challenges:**

| Challenge | Description | Solution |
|-----------|-------------|----------|
| Temporal Order | Sequence matters | Use sequence models |
| Variable Length | Different sequence lengths | Padding, truncation |
| Temporal Dependencies | Past affects future | RNN, LSTM, sliding window |
| Feature Extraction | Raw signals need processing | Statistical features, FFT |

**Approaches to Time-Series Classification:**

| Approach | Method | When to Use |
|----------|--------|-------------|
| Feature-based | Extract statistics + ML | Small data, interpretable |
| Distance-based | DTW + KNN | Similar shape matching |
| Deep Learning | LSTM, 1D-CNN | Large data, complex patterns |
| Hybrid | Features + DL | Best of both worlds |

**Common Feature Extraction:**

| Feature Type | Examples |
|--------------|----------|
| Statistical | Mean, std, min, max, skew |
| Temporal | Autocorrelation, trend |
| Frequency | FFT, spectral power |
| Domain-specific | Peak count, zero crossings |

**Python Code Example:**
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Sample time-series data (e.g., sensor readings)
# Each row is one time-series, columns are time steps
n_samples, n_timesteps = 100, 50
X_raw = np.random.randn(n_samples, n_timesteps)
y = np.random.randint(0, 3, n_samples)  # 3 classes

# Method 1: Feature extraction approach
def extract_features(series):
    return {
        'mean': np.mean(series),
        'std': np.std(series),
        'min': np.min(series),
        'max': np.max(series),
        'trend': np.polyfit(range(len(series)), series, 1)[0],
        'zero_crossings': np.sum(np.diff(np.sign(series)) != 0)
    }

# Extract features for all samples
features = pd.DataFrame([extract_features(X_raw[i]) for i in range(n_samples)])

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.3f}")

# Method 2: Sliding window for sequential input
def create_windows(data, window_size=10, step=5):
    windows = []
    for i in range(0, len(data) - window_size, step):
        windows.append(data[i:i+window_size])
    return np.array(windows)
```

**Deep Learning Approach (LSTM):**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Reshape for LSTM: (samples, timesteps, features)
X_lstm = X_raw.reshape(n_samples, n_timesteps, 1)

model = Sequential([
    LSTM(64, input_shape=(n_timesteps, 1)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

**Interview Tips:**
- Never shuffle time-series randomly; use time-based splits
- Feature extraction is often sufficient for small datasets
- LSTM/GRU capture long-term dependencies better than 1D-CNN
- Consider DTW (Dynamic Time Warping) for shape-based matching
- Mention stationarity and detrending if relevant

---

## Question 7
- [ ] Done

**Discuss the concept of multi-label classification and how it differs from multiclass classification.**

### Answer

**Definition:**
**Multiclass classification** assigns exactly ONE label from multiple classes (e.g., cat OR dog OR bird). **Multi-label classification** assigns MULTIPLE labels simultaneously to each instance (e.g., a movie can be Action AND Comedy AND Romance).

**Key Differences:**

| Aspect | Multiclass | Multi-label |
|--------|------------|-------------|
| Labels per sample | Exactly 1 | 0 to N |
| Output | Single class | Set of classes |
| Example | Animal type | Movie genres |
| Output activation | Softmax | Sigmoid (per label) |
| Loss function | Cross-entropy | Binary cross-entropy |
| Probability sum | = 1 | Independent per label |

**Examples:**

| Task | Type | Labels |
|------|------|--------|
| Digit recognition | Multiclass | 0, 1, 2...9 (one) |
| Movie genre | Multi-label | Action, Comedy, Drama (multiple) |
| News category | Multiclass | Sports, Politics, Tech (one) |
| Image tagging | Multi-label | Dog, Outdoor, Sunny (multiple) |

**Multi-label Approaches:**

| Approach | Description | Pros/Cons |
|----------|-------------|-----------|
| Binary Relevance | Train N independent binary classifiers | Simple, ignores label correlation |
| Classifier Chain | Sequential classifiers, use previous predictions | Captures correlation, order matters |
| Label Powerset | Treat each label combination as class | Captures correlation, combinatorial explosion |
| Neural Network | Single model with sigmoid outputs | End-to-end, handles correlation |

**Evaluation Metrics:**

| Metric | Multiclass | Multi-label |
|--------|------------|-------------|
| Accuracy | Standard | Exact match (strict) or Subset |
| Precision/Recall | Per-class or macro/micro | Sample-avg, micro, macro |
| Hamming Loss | N/A | Fraction of wrong labels |

**Python Code Example:**
```python
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, accuracy_score
import numpy as np

# Create multi-label dataset
X, y = make_multilabel_classification(n_samples=1000, n_features=20, 
                                       n_classes=5, n_labels=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Method 1: Binary Relevance (OneVsRest)
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Multi-label metrics
print(f"Hamming Loss: {hamming_loss(y_test, y_pred):.3f}")
print(f"Exact Match Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Method 2: Neural Network with Sigmoid
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dense(32, activation='relu'),
    Dense(5, activation='sigmoid')  # Sigmoid for multi-label
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, verbose=0)

# Predict with threshold
y_prob = model.predict(X_test)
y_pred_nn = (y_prob > 0.5).astype(int)
```

**Interview Tips:**
- Softmax forces one class; Sigmoid allows multiple
- Hamming loss is key metric for multi-label
- Classifier chains capture label dependencies
- In deep learning, use sigmoid + binary cross-entropy
- Multi-label is more complex but real-world common

---

## Question 8
- [ ] Done

**Discuss the impact of deep learning on traditional classification algorithms.**

### Answer

**Definition:**
Deep learning has revolutionized classification by automatically learning hierarchical features from raw data, eliminating manual feature engineering. While traditional ML requires domain expertise for feature extraction, deep learning models learn optimal representations end-to-end.

**Paradigm Shift:**

| Aspect | Traditional ML | Deep Learning |
|--------|---------------|---------------|
| Feature Engineering | Manual, expert-driven | Automatic, learned |
| Data Requirements | Small-medium | Large datasets |
| Interpretability | High | Low (black box) |
| Compute Requirements | Low | High (GPU needed) |
| Training Time | Fast | Slow |

**Where Deep Learning Excels:**

| Domain | Traditional | Deep Learning |
|--------|-------------|---------------|
| Image | HOG + SVM | CNN (ResNet, VGG) |
| Text | TF-IDF + Naive Bayes | Transformers (BERT) |
| Audio | MFCC + HMM | RNN, WaveNet |
| Tabular | Still competitive | May overfit |

**When to Use What:**

| Scenario | Recommendation |
|----------|----------------|
| Small dataset (<10K) | Traditional ML |
| Tabular data | XGBoost, RF often win |
| Image/Text/Audio | Deep Learning |
| Need interpretability | Traditional ML |
| Abundant data & compute | Deep Learning |
| Fast inference needed | Traditional ML |

**Key Deep Learning Architectures:**

| Architecture | Best For | Key Innovation |
|--------------|----------|----------------|
| CNN | Images | Local pattern detection |
| RNN/LSTM | Sequences | Temporal dependencies |
| Transformer | Text, General | Attention mechanism |
| GNN | Graphs | Node relationships |

**Python Code Comparison:**
```python
# Traditional ML: Feature engineering + classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Manual feature extraction
tfidf = TfidfVectorizer(max_features=1000)
X_features = tfidf.fit_transform(texts)

# Step 2: Train classifier
rf = RandomForestClassifier()
rf.fit(X_features, labels)

# Deep Learning: End-to-end learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Single model learns features + classification
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_len),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_sequences, labels, epochs=10)
```

**Impact Summary:**

| Area | Impact |
|------|--------|
| Accuracy | SOTA in vision, NLP, speech |
| Feature Engineering | Reduced/eliminated |
| Transfer Learning | Pre-trained models available |
| Hardware | Drove GPU development |
| Democratization | Easier high performance |

**Interview Tips:**
- Deep learning doesn't always win (tabular data: XGBoost)
- Transfer learning makes DL accessible with small data
- Traditional ML is still valuable for interpretability
- Ensemble of DL + traditional can be powerful
- Mention computational cost as a real constraint

---

## Question 9
- [ ] Done

**Discuss the role of attention mechanisms in classification tasks.**

### Answer

**Definition:**
Attention mechanisms allow models to focus on relevant parts of input when making predictions, rather than treating all input equally. They compute weighted importance scores, enabling the model to "attend" to specific features, words, or regions that matter most for classification.

**Core Concept:**

| Component | Description |
|-----------|-------------|
| Query (Q) | What we're looking for |
| Key (K) | What we match against |
| Value (V) | What we retrieve |
| Attention Score | Relevance weight |

**Mathematical Formulation:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$, $K$, $V$ = Query, Key, Value matrices
- $d_k$ = dimension of keys (scaling factor)

**Types of Attention:**

| Type | Description | Use Case |
|------|-------------|----------|
| Self-Attention | Input attends to itself | Transformers, BERT |
| Cross-Attention | Query from one source, K/V from another | Seq2Seq |
| Multi-Head | Multiple attention in parallel | Capture different patterns |
| Local | Attend to nearby positions | Efficiency |

**Why Attention Helps Classification:**

| Benefit | Explanation |
|---------|-------------|
| Interpretability | Shows which parts influenced decision |
| Long-range dependencies | Captures distant relationships |
| Parallel computation | Unlike RNN, can parallelize |
| Dynamic weighting | Context-dependent importance |

**Applications in Classification:**

| Domain | Application |
|--------|-------------|
| Text | Which words matter for sentiment |
| Images | Which regions indicate class |
| Time-series | Which time steps are important |
| Tabular | Feature importance per sample |

**Python Code Example:**
```python
import tensorflow as tf
from tensorflow.keras.layers import (Layer, Dense, Embedding, 
                                      GlobalAveragePooling1D, Input)
from tensorflow.keras.models import Model
import numpy as np

# Simple Self-Attention Layer
class SelfAttention(Layer):
    def __init__(self, units):
        super().__init__()
        self.W_q = Dense(units)
        self.W_k = Dense(units)
        self.W_v = Dense(units)
        
    def call(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Attention scores
        scores = tf.matmul(Q, K, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(K.shape[-1], tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        
        # Weighted sum
        output = tf.matmul(weights, V)
        return output, weights  # Return weights for interpretability

# Text classification with attention
vocab_size, max_len, embed_dim = 10000, 100, 64

inputs = Input(shape=(max_len,))
x = Embedding(vocab_size, embed_dim)(inputs)
x, attn_weights = SelfAttention(64)(x)
x = GlobalAveragePooling1D()(x)
x = Dense(32, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy')

# Using pre-built: MultiHeadAttention
from tensorflow.keras.layers import MultiHeadAttention

mha = MultiHeadAttention(num_heads=4, key_dim=64)
attended = mha(x, x)  # Self-attention
```

**Attention in Popular Models:**

| Model | Attention Type | Task |
|-------|----------------|------|
| BERT | Multi-head self-attention | Text classification |
| Vision Transformer (ViT) | Patch self-attention | Image classification |
| Transformer | Encoder self-attention | General |
| TabNet | Sparse attention | Tabular classification |

**Interview Tips:**
- Attention = "soft" selection vs hard feature selection
- Transformers replaced RNNs due to attention + parallelization
- Attention weights provide interpretability (which words/pixels matter)
- Multi-head attention captures different relationship types
- Self-attention is $O(n^2)$ complexity; mention efficiency variants

---

## Question 10
- [ ] Done

**What considerations would you take into account when building a credit scoring model?**

### Answer

**Definition:**
Credit scoring predicts loan default probability. It requires special considerations for regulatory compliance, interpretability, fairness, and business impact of errors.

**Key Considerations:**

| Area | Consideration | Why It Matters |
|------|---------------|----------------|
| **Regulatory** | Explainability | Laws require explanation of denial |
| | Fair lending | Cannot discriminate on protected attributes |
| | Documentation | Audit trail required |
| **Model Choice** | Interpretable models | Logistic Regression, Scorecard |
| | Feature importance | Must explain each factor |
| **Features** | Avoid proxies | ZIP code may proxy race |
| | Economic cycles | Features may behave differently |
| **Metrics** | Cost-sensitive | FN (default) more costly than FP |
| | Calibration | Predicted probabilities must be accurate |
| **Monitoring** | Concept drift | Economic changes affect patterns |
| | Population stability | Monitor feature distributions |

**Recommended Approach:**

| Step | Action |
|------|--------|
| 1 | Use interpretable models (Logistic Regression, Scorecard) |
| 2 | Feature engineering with domain knowledge |
| 3 | Check for disparate impact on protected groups |
| 4 | Optimize for business metrics (expected loss) |
| 5 | Calibrate probabilities |
| 6 | Document everything |
| 7 | Monitor in production |

**Important Metrics:**

| Metric | Purpose |
|--------|---------|
| Gini/KS | Discriminatory power |
| PSI | Population stability |
| Expected Loss | Business impact |
| Adverse Action Reasons | Compliance |

**Interview Tips:**
- Emphasize interpretability and compliance
- Mention fair lending and protected classes
- Discuss cost of FP vs FN (business perspective)
- Know that black-box models face regulatory challenges
- Mention model monitoring and drift detection

---

## Question 11
- [ ] Done

**Describe how you would build a spam detection classifier.**

### Answer

**Pipeline:**
```
Raw Text -> Preprocessing -> Feature Extraction -> Model Training -> Evaluation
```

**Steps:**

1. **Data Collection**: Labeled emails (spam/ham)

2. **Preprocessing**:
   - Lowercase
   - Remove HTML, special chars
   - Tokenization
   - Remove stopwords
   - Stemming

3. **Feature Extraction**:
   - TF-IDF
   - N-grams
   - Additional: caps ratio, links count

4. **Model Selection**:
   - Naive Bayes (fast, good for text)
   - Logistic Regression (interpretable)

**Python Code Example:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re

emails = [
    "Congratulations! You won $1000!",
    "Meeting tomorrow at 3pm",
    "FREE gift card!",
    "Please review the document"
]
labels = [1, 0, 1, 0]  # 1=spam

def preprocess(text):
    return re.sub(r'[^a-zA-Z\s]', '', text.lower())

emails_clean = [preprocess(e) for e in emails]
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(emails_clean)

model = MultinomialNB().fit(X, labels)
```

**Interview Tips:**
- Emphasize precision (false positives costly)
- Need continuous retraining (spam evolves)
- Handle imbalanced data

---

## Question 12
- [ ] Done

**Describe a real-world application where precision is more important than recall, and vice versa.**

### Answer

**Definitions:**
- **Precision** = TP / (TP + FP) - "Of predicted positive, how many correct?"
- **Recall** = TP / (TP + FN) - "Of actual positive, how many caught?"

**Precision More Important:**

**Example: Spam Detection**
- FP: Important email marked spam (missed meeting!)
- FN: Spam reaches inbox (minor annoyance)
- **Better to let spam through than lose important emails**

**Recall More Important:**

**Example: Cancer Screening**
- FP: Healthy person flagged (extra tests)
- FN: Cancer missed (life-threatening)
- **Better to over-test than miss cancer**

**Summary:**
| Scenario | Prioritize | Reason |
|----------|-----------|--------|
| Spam filter | Precision | Don't lose emails |
| Cancer screening | Recall | Don't miss cases |
| Fraud detection | Recall | Catch maximum fraud |

**Interview Tips:**
- Relate to business impact
- Threshold tuning: lower threshold -> higher recall
- F1-score balances both

---

## Question 13
- [ ] Done

**Explain how you could use classification models to predict customer churn.**

### Answer

**Definition:**
Binary classification identifying customers likely to stop using a service. Model learns from historical behavior to flag at-risk customers.

**Pipeline:**
```
Data Collection -> Feature Engineering -> Model Training -> Prediction -> Intervention
```

**Feature Engineering:**
| Category | Examples |
|----------|---------|
| Behavioral | Days since last login |
| Financial | Monthly spend |
| Support | Number of complaints |
| Trend | Declining usage |

**Challenges:**
- Class imbalance (5-15% churn rate)
- Use SMOTE, class weights

**Python Code Example:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

# Simulated data
np.random.seed(42)
n = 1000
X = np.random.randn(n, 5)
y = (np.random.random(n) < 0.15).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(class_weight='balanced', n_estimators=100)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
```

**Interview Tips:**
- Emphasize business impact
- Handle imbalanced data properly
- Feature engineering is crucial

---
