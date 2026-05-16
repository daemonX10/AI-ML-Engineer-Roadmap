# Svm Interview Questions - Scenario_Based Questions

## Question 1
- [ ] Done

**Discuss the use of SVM in bioinformatics and computational biology.**

### Answer

**Why SVM is Popular in Bioinformatics:**
1. Works well with high-dimensional data (genes >> samples)
2. Handles sparse data effectively
3. Custom kernels for biological sequences
4. Strong theoretical guarantees

**Key Applications:**

| Application | Description | Kernel Used |
|-------------|-------------|-------------|
| **Gene Expression Classification** | Cancer subtype classification from microarray data | Linear/RBF |
| **Protein Function Prediction** | Predict protein function from sequence | String kernels |
| **Drug Discovery** | Predict molecular activity | Chemical fingerprint kernels |
| **Gene Selection** | Identify disease biomarkers | Linear (for interpretability) |
| **Splice Site Prediction** | Identify exon-intron boundaries | Sequence kernels |

**Advantages for Bioinformatics:**
- Handles p >> n (many genes, few samples)
- Effective with noisy biological data
- Support vectors identify critical samples
- Kernels for sequences, graphs, structures

**Challenges:**
- Interpretability with complex kernels
- Large-scale genomics data (scalability)
- Class imbalance (disease vs healthy)

**Practical Approach:**
1. Feature selection/dimensionality reduction first
2. Use linear SVM for interpretability (gene importance)
3. RBF for complex non-linear patterns
4. Handle class imbalance with `class_weight='balanced'`

---

## Question 2
- [ ] Done

**How would you apply SVM for image classification tasks?**

### Answer

**Approach:**
Extract features from images first, then train SVM classifier on features. SVMs work well with engineered features but are largely replaced by CNNs for raw image classification.

**Pipeline:**

1. **Feature Extraction:**
   - HOG (Histogram of Oriented Gradients)
   - SIFT/SURF (Scale-Invariant Feature Transform)
   - Color histograms
   - CNN features (transfer learning)

2. **Preprocessing:**
   - Resize images to uniform size
   - Normalize pixel values
   - Flatten or use extracted features

3. **SVM Training:**
   - RBF kernel for non-linear patterns
   - Tune C and gamma

**When SVM is Still Useful:**
- Small datasets (insufficient for deep learning)
- Interpretability needed
- Limited computational resources
- As final classifier on CNN features

**Example Pipeline:**
```python
from sklearn.svm import SVC
from skimage.feature import hog

# Extract HOG features
features = [hog(img, orientations=9, pixels_per_cell=(8,8)) for img in images]
X = np.array(features)

# Train SVM
svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_train, y_train)
```

**Modern Hybrid Approach:**
1. Use pre-trained CNN (ResNet, VGG) as feature extractor
2. Take features from last fully-connected layer
3. Train SVM on these high-level features
4. Often outperforms fine-tuning CNN on small datasets

---

## Question 3
- [ ] Done

**Discuss the application of SVMs in text categorization.**

### Answer

**Why SVM Excels at Text Classification:**
1. High-dimensional sparse data (bag-of-words, TF-IDF)
2. Linear SVM works exceptionally well
3. Fast training with LinearSVC
4. Feature weights provide interpretability

**Pipeline:**
1. Text preprocessing (tokenize, lowercase, remove stopwords)
2. Feature extraction (TF-IDF, word embeddings)
3. Train Linear SVM
4. Evaluate and tune

**Applications:**
| Task | Description |
|------|-------------|
| Spam Detection | Email/SMS spam filtering |
| Sentiment Analysis | Positive/negative opinion |
| Topic Classification | News categorization |
| Language Detection | Identify text language |
| Intent Classification | Chatbot intent recognition |

**Why Linear Kernel for Text:**
- TF-IDF creates high-dimensional sparse vectors
- Data is often linearly separable in high dimensions
- Linear is faster and scales better
- Non-linear kernels rarely improve accuracy

**Practical Implementation:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=10000)),
    ('svm', LinearSVC(C=1.0, class_weight='balanced'))
])

pipeline.fit(texts_train, labels_train)
predictions = pipeline.predict(texts_test)
```

**Tips:**
- Use `class_weight='balanced'` for imbalanced categories
- Try different n-gram ranges (1,1), (1,2), (1,3)
- Feature selection can improve speed without hurting accuracy

---

## Question 4
- [ ] Done

**How would you leverage SVM for intrusion detection in cybersecurity?**

### Answer

**Scenario:**
Detect malicious network traffic or system behavior by classifying network packets/events as normal or attack.

**Approach:**

1. **Binary Classification**: Normal vs Attack
2. **Multi-class**: Normal vs specific attack types (DoS, Probe, R2L, U2R)
3. **Anomaly Detection**: One-Class SVM on normal traffic only

**Feature Engineering for Network Data:**
- Packet-level: Protocol, port, flags, payload size
- Flow-level: Duration, byte count, packet count
- Statistical: Mean, variance of timing/size
- Behavioral: Connection patterns, unusual hours

**Implementation Strategy:**

| Approach | When to Use | SVM Type |
|----------|-------------|----------|
| **Supervised** | Labeled attack data available | SVC or LinearSVC |
| **Anomaly Detection** | Only normal data available | OneClassSVM |
| **Real-time** | Streaming network traffic | SGDClassifier (hinge loss) |

**Challenges and Solutions:**

| Challenge | Solution |
|-----------|----------|
| Class imbalance (few attacks) | `class_weight='balanced'`, SMOTE |
| High volume data | LinearSVC, streaming with SGD |
| New attack types (zero-day) | One-Class SVM, periodic retraining |
| Feature engineering | Domain expertise, automated feature extraction |

**Example Pipeline:**
```python
from sklearn.svm import SVC, OneClassSVM

# Supervised approach
svm = SVC(kernel='rbf', class_weight='balanced')
svm.fit(X_train, y_train)

# Anomaly detection (train on normal only)
normal_data = X_train[y_train == 0]
oc_svm = OneClassSVM(nu=0.01, kernel='rbf')
oc_svm.fit(normal_data)
# -1 = anomaly/attack, +1 = normal
```

---

## Question 5
- [ ] Done

**Propose an application of SVM in the healthcare industry for disease diagnosis.**

### Answer

**Scenario:**
Build a diagnostic support system to classify patients as having/not having a specific disease based on clinical features, lab results, or medical imaging features.

**Example: Cancer Diagnosis from Gene Expression**

**Problem:** Classify tissue samples as cancerous vs healthy based on gene expression levels.

**Why SVM is Suitable:**
- High-dimensional data (thousands of genes)
- Small sample sizes (expensive to collect)
- Need for interpretable feature weights (biomarker discovery)
- Strong theoretical foundation (important for medical validation)

**Pipeline:**

1. **Data Collection:** Gene expression microarray or RNA-seq data
2. **Preprocessing:** Normalization, missing value handling
3. **Feature Selection:** Reduce genes to most relevant
4. **Model Training:** SVM with cross-validation
5. **Interpretation:** Identify key genes (biomarkers)

**Implementation:**
```python
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_select', SelectKBest(f_classif, k=100)),  # Top 100 genes
    ('svm', SVC(kernel='linear', probability=True))     # Linear for interpretability
])

pipeline.fit(X_train, y_train)
proba = pipeline.predict_proba(X_test)  # Probability for clinical use
```

**Key Considerations:**
- Use `probability=True` for clinical decision support
- Linear kernel for biomarker identification (gene importance)
- Cross-validation essential for small medical datasets
- Class imbalance handling (usually more healthy than diseased)
- Regulatory compliance (explainability, validation)

**Other Healthcare Applications:**
- Diabetic retinopathy from retinal images
- Heart disease risk from ECG features
- Mental health diagnosis from behavioral data

---

## Question 6
- [ ] Done

**Discuss the role of SVMs in the development of self-driving cars.**

### Answer

**Context:**
SVMs played a historical role in early autonomous vehicle research but are now largely supplemented by deep learning. However, SVMs still have niche applications in specific subsystems.

**Historical Role:**
- Early pedestrian detection
- Road sign classification
- Lane detection
- Object recognition (pre-deep learning era)

**Current Limited Use Cases:**

| Application | Why SVM | Modern Alternative |
|-------------|---------|-------------------|
| **Sensor Fusion** | Combine features from multiple sensors | Often neural networks now |
| **Simple Classification** | Low-latency decisions | Still viable |
| **Anomaly Detection** | Detect sensor failures | One-Class SVM |
| **Edge Cases** | Small dataset scenarios | When insufficient deep learning data |

**Why Deep Learning Replaced SVM for Autonomous Vehicles:**
1. End-to-end learning from raw sensor data
2. Automatic feature learning (no manual feature engineering)
3. Better performance on complex perception tasks
4. Hardware acceleration (GPUs, TPUs)

**Where SVM Might Still Fit:**
- Resource-constrained embedded systems
- Interpretable safety-critical decisions
- Ensemble member with other models
- Quick prototyping before deep learning

**Modern Self-Driving Stack (For Context):**
- Perception: CNNs, Transformers
- Prediction: RNNs, Graph Neural Networks
- Planning: Optimization, Reinforcement Learning
- Control: Model Predictive Control

**Key Takeaway:**
SVM was foundational in autonomous vehicle research but has been largely superseded by deep learning for main perception tasks. It may still find use in specific, constrained scenarios.
