# Dimensionality Reduction Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Implement PCA on a given dataset using scikit-learn and plot the explained variance ratio.**

### Answer

**Pipeline:**
1. Load dataset
2. Standardize features (essential for PCA)
3. Apply PCA
4. Extract explained variance ratios
5. Plot cumulative variance

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Step 1: Load data
data = load_iris()
X = data.data
y = data.target

# Step 2: Standardize (important - PCA is sensitive to scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA (fit on all components first to see variance)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Step 4: Get explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Explained Variance Ratio:", explained_variance)
print("Cumulative Variance:", cumulative_variance)

# Step 5: Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar plot - individual variance
axes[0].bar(range(1, len(explained_variance) + 1), explained_variance)
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Variance per Component')

# Line plot - cumulative variance
axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative Variance Explained')
axes[1].legend()

plt.tight_layout()
plt.show()

# Apply PCA keeping 95% variance
pca_95 = PCA(n_components=0.95)
X_reduced = pca_95.fit_transform(X_scaled)
print(f"Components for 95% variance: {pca_95.n_components_}")
```

**Output Interpretation:**
- First component explains most variance
- Usually 2-3 components capture 90%+ for simple datasets
- Use cumulative plot to decide number of components

---

## Question 2
- [ ] Done

**Write a Python function that performs feature selection using Recursive Feature Elimination (RFE).**

### Answer

**Pipeline:**
1. Create base estimator (with `coef_` or `feature_importances_`)
2. Initialize RFE with desired number of features
3. Fit on data
4. Get selected features
5. Return reduced dataset

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd

def perform_rfe(X, y, n_features=10, estimator=None):
    """
    Perform Recursive Feature Elimination.
    
    Parameters:
    - X: feature matrix
    - y: target
    - n_features: number of features to select
    - estimator: model with coef_ or feature_importances_
    
    Returns:
    - X_selected: reduced feature matrix
    - selected_features: indices of selected features
    - ranking: ranking of all features
    """
    # Default estimator
    if estimator is None:
        estimator = LogisticRegression(max_iter=1000)
    
    # Initialize RFE
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    
    # Fit RFE
    rfe.fit(X, y)
    
    # Get results
    X_selected = rfe.transform(X)
    selected_features = np.where(rfe.support_)[0]
    ranking = rfe.ranking_
    
    return X_selected, selected_features, ranking


def perform_rfecv(X, y, estimator=None, cv=5):
    """
    RFE with Cross-Validation to find optimal number of features.
    """
    if estimator is None:
        estimator = LogisticRegression(max_iter=1000)
    
    # RFECV automatically finds optimal number
    rfecv = RFECV(estimator=estimator, cv=cv, scoring='accuracy')
    rfecv.fit(X, y)
    
    print(f"Optimal number of features: {rfecv.n_features_}")
    
    return rfecv.transform(X), rfecv.support_, rfecv.n_features_


# Example usage
if __name__ == "__main__":
    # Load data
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    # Method 1: Fixed number of features
    X_selected, selected_idx, ranking = perform_rfe(X, y, n_features=10)
    
    print("Selected Features:")
    for idx in selected_idx:
        print(f"  {feature_names[idx]}")
    
    print(f"\nOriginal shape: {X.shape}")
    print(f"Reduced shape: {X_selected.shape}")
    
    # Method 2: With cross-validation (optimal features)
    X_optimal, mask, n_optimal = perform_rfecv(X, y, cv=5)
    print(f"\nOptimal features selected: {n_optimal}")
```

**Key Points:**
- RFE recursively removes least important features
- Estimator must have `coef_` or `feature_importances_`
- RFECV finds optimal number via cross-validation

---

## Question 3
- [ ] Done

**Code a small example to demonstrate the use of LDA for classification.**

### Answer

**Pipeline:**
1. Load classification dataset
2. Split into train/test
3. Optionally use LDA for dimensionality reduction
4. Use LDA as classifier
5. Evaluate and visualize

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load data
data = load_iris()
X = data.data
y = data.target
class_names = data.target_names

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ========== LDA as Classifier ==========
lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit(X_train, y_train)

# Predictions
y_pred = lda_clf.predict(X_test)

# Evaluation
print("=== LDA as Classifier ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ========== LDA for Dimensionality Reduction ==========
# Max components = n_classes - 1 = 2
lda_reduce = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda_reduce.fit_transform(X_train, y_train)
X_test_lda = lda_reduce.transform(X_test)

print(f"\nOriginal dimensions: {X_train.shape[1]}")
print(f"Reduced dimensions: {X_train_lda.shape[1]}")

# ========== Visualization ==========
plt.figure(figsize=(10, 4))

# Plot training data
plt.subplot(1, 2, 1)
for i, class_name in enumerate(class_names):
    mask = y_train == i
    plt.scatter(X_train_lda[mask, 0], X_train_lda[mask, 1], 
                label=class_name, alpha=0.7)
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.title('LDA: Training Data')
plt.legend()

# Plot test data
plt.subplot(1, 2, 2)
for i, class_name in enumerate(class_names):
    mask = y_test == i
    plt.scatter(X_test_lda[mask, 0], X_test_lda[mask, 1], 
                label=class_name, alpha=0.7)
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.title('LDA: Test Data')
plt.legend()

plt.tight_layout()
plt.show()
```

**Key Points:**
- LDA can be both classifier and dimensionality reducer
- Max components = number of classes - 1
- Requires labels (supervised method)
- Works well when classes are separable

---

## Question 4
- [ ] Done

**Implement a basic version of an autoencoder for dimensionality reduction using TensorFlow/Keras.**

### Answer

**Pipeline:**
1. Define encoder (input → latent)
2. Define decoder (latent → output)
3. Combine into autoencoder
4. Train to minimize reconstruction error
5. Use encoder for dimension reduction

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Load and preprocess data
digits = load_digits()
X = digits.data
y = digits.target

# Scale to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Step 2: Define Autoencoder
input_dim = X_train.shape[1]  # 64 for digits
latent_dim = 10  # Reduce to 10 dimensions

# Encoder
encoder_input = layers.Input(shape=(input_dim,), name='encoder_input')
x = layers.Dense(32, activation='relu')(encoder_input)
x = layers.Dense(16, activation='relu')(x)
latent = layers.Dense(latent_dim, activation='linear', name='latent')(x)

encoder = Model(encoder_input, latent, name='encoder')

# Decoder
decoder_input = layers.Input(shape=(latent_dim,), name='decoder_input')
x = layers.Dense(16, activation='relu')(decoder_input)
x = layers.Dense(32, activation='relu')(x)
decoder_output = layers.Dense(input_dim, activation='sigmoid')(x)

decoder = Model(decoder_input, decoder_output, name='decoder')

# Autoencoder (Encoder + Decoder)
autoencoder_input = layers.Input(shape=(input_dim,))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = Model(autoencoder_input, decoded, name='autoencoder')

# Step 3: Compile and train
autoencoder.compile(optimizer='adam', loss='mse')

print(autoencoder.summary())

history = autoencoder.fit(
    X_train, X_train,  # Input = Target (reconstruction)
    epochs=50,
    batch_size=32,
    validation_data=(X_test, X_test),
    verbose=1
)

# Step 4: Use encoder for dimensionality reduction
X_encoded = encoder.predict(X_scaled)

print(f"\nOriginal shape: {X_scaled.shape}")
print(f"Encoded shape: {X_encoded.shape}")

# Step 5: Visualize latent space
plt.figure(figsize=(10, 4))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')

# 2D projection of latent space
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_encoded[:, 0], X_encoded[:, 1], c=y, cmap='tab10', s=5)
plt.colorbar(scatter)
plt.xlabel('Latent Dim 1')
plt.ylabel('Latent Dim 2')
plt.title('Latent Space (first 2 dims)')

plt.tight_layout()
plt.show()
```

**Key Points:**
- Encoder compresses, decoder reconstructs
- Loss = reconstruction error (MSE)
- Latent space = reduced representation
- Can capture nonlinear relationships

---

## Question 5
- [ ] Done

**Modify a given t-SNE implementation to work more efficiently on a large-scale dataset.**

### Answer

**Pipeline:**
1. Reduce dimensions with PCA first (critical for speed)
2. Sample data if very large
3. Use optimized t-SNE parameters
4. Consider UMAP as faster alternative

```python
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time

def efficient_tsne(X, n_components=2, pca_components=50, 
                   sample_size=None, random_state=42):
    """
    Efficient t-SNE for large datasets.
    
    Key optimizations:
    1. PCA preprocessing to reduce dimensions
    2. Sampling for very large datasets
    3. Optimized parameters for speed
    
    Parameters:
    - X: input data
    - n_components: final dimensions (usually 2)
    - pca_components: intermediate PCA dimensions
    - sample_size: if set, sample this many points
    - random_state: for reproducibility
    
    Returns:
    - X_tsne: transformed data
    - sample_idx: indices of sampled points (if sampled)
    """
    
    n_samples, n_features = X.shape
    print(f"Input shape: {X.shape}")
    sample_idx = None
    
    # Step 1: Sample if dataset is very large
    if sample_size and n_samples > sample_size:
        print(f"Sampling {sample_size} points from {n_samples}")
        np.random.seed(random_state)
        sample_idx = np.random.choice(n_samples, sample_size, replace=False)
        X = X[sample_idx]
    
    # Step 2: PCA preprocessing (critical for speed!)
    if n_features > pca_components:
        print(f"Applying PCA: {n_features} -> {pca_components} dimensions")
        start = time.time()
        pca = PCA(n_components=pca_components, random_state=random_state)
        X = pca.fit_transform(X)
        print(f"PCA time: {time.time() - start:.2f}s")
        print(f"Variance retained: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Step 3: t-SNE with optimized parameters
    print("Running t-SNE...")
    start = time.time()
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=30,           # Good default
        learning_rate='auto',    # Auto-set based on n_samples
        n_iter=1000,             # Enough for convergence
        init='pca',              # PCA init is faster
        method='barnes_hut',     # O(n log n) approximation
        random_state=random_state,
        n_jobs=-1                # Parallel computation
    )
    
    X_tsne = tsne.fit_transform(X)
    print(f"t-SNE time: {time.time() - start:.2f}s")
    
    return X_tsne, sample_idx


# Example: Compare standard vs efficient t-SNE
if __name__ == "__main__":
    # Create large dataset
    np.random.seed(42)
    n_samples = 10000
    n_features = 500
    X_large = np.random.randn(n_samples, n_features)
    
    # Efficient method
    print("=== Efficient t-SNE ===")
    X_tsne_efficient, _ = efficient_tsne(
        X_large, 
        pca_components=50,
        sample_size=5000
    )
    
    print(f"\nOutput shape: {X_tsne_efficient.shape}")


# Alternative: Use UMAP for very large datasets
def umap_alternative(X, n_components=2, n_neighbors=15):
    """
    UMAP is much faster than t-SNE for large datasets.
    """
    import umap
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric='euclidean'
    )
    
    return reducer.fit_transform(X)
```

**Optimization Summary:**

| Optimization | Impact |
|-------------|--------|
| PCA first (50-100 dims) | 10-100x faster |
| `method='barnes_hut'` | O(n log n) vs O(n²) |
| `init='pca'` | Faster convergence |
| Sampling | Linear reduction |
| UMAP instead | 10x faster, similar quality |

---

## Question 6
- [ ] Done

**Develop a Python script to compare the performance of PCA and LDA on a sample dataset.**

### Answer

**Pipeline:**
1. Load dataset with labels
2. Apply PCA and LDA
3. Train classifier on both
4. Compare accuracy and visualization
5. Analyze variance explained vs class separation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

# Step 1: Load data
data = load_wine()
X = data.data
y = data.target
class_names = data.target_names

print(f"Original shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# Step 2: Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA (keep same number of components for fair comparison)
n_components = 2  # For visualization and comparison
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Step 4: Apply LDA (max = n_classes - 1)
lda = LinearDiscriminantAnalysis(n_components=n_components)
X_lda = lda.fit_transform(X_scaled, y)  # LDA needs labels

# Step 5: Compare with classifier
X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42, stratify=y
)
X_train_lda, X_test_lda, _, _ = train_test_split(
    X_lda, y, test_size=0.3, random_state=42, stratify=y
)

clf = LogisticRegression(random_state=42)

# Cross-validation scores
pca_scores = cross_val_score(clf, X_pca, y, cv=5)
lda_scores = cross_val_score(clf, X_lda, y, cv=5)

print("\n=== Performance Comparison ===")
print(f"PCA ({n_components} components):")
print(f"  CV Accuracy: {pca_scores.mean():.4f} (+/- {pca_scores.std():.4f})")
print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

print(f"\nLDA ({n_components} components):")
print(f"  CV Accuracy: {lda_scores.mean():.4f} (+/- {lda_scores.std():.4f})")
print(f"  Explained variance ratio: {lda.explained_variance_ratio_.sum():.2%}")

# Step 6: Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# PCA plot
for i, class_name in enumerate(class_names):
    mask = y == i
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], label=class_name, alpha=0.7)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[0].set_title(f'PCA (Acc: {pca_scores.mean():.2%})')
axes[0].legend()

# LDA plot
for i, class_name in enumerate(class_names):
    mask = y == i
    axes[1].scatter(X_lda[mask, 0], X_lda[mask, 1], label=class_name, alpha=0.7)
axes[1].set_xlabel('LD1')
axes[1].set_ylabel('LD2')
axes[1].set_title(f'LDA (Acc: {lda_scores.mean():.2%})')
axes[1].legend()

plt.tight_layout()
plt.show()

# Summary table
print("\n=== Summary ===")
print(f"{'Metric':<25} {'PCA':>10} {'LDA':>10}")
print("-" * 45)
print(f"{'Accuracy':<25} {pca_scores.mean():>10.2%} {lda_scores.mean():>10.2%}")
print(f"{'Supervised':<25} {'No':>10} {'Yes':>10}")
print(f"{'Max Components':<25} {X.shape[1]:>10} {len(np.unique(y))-1:>10}")
```

**Key Observations:**
- LDA usually achieves higher classification accuracy (uses labels)
- PCA explains more total variance (unsupervised objective)
- LDA limited to n_classes - 1 components
- Use PCA for exploration, LDA for classification preprocessing

---

## Question 7
- [ ] Done

**Create a Python function that uses Factor Analysis for dimensionality reduction on multivariate data.**

### Answer

**Pipeline:**
1. Load multivariate data
2. Standardize features
3. Apply Factor Analysis
4. Analyze loadings (factor structure)
5. Use factors as reduced features

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def factor_analysis_reduction(X, n_factors=2, feature_names=None):
    """
    Perform Factor Analysis for dimensionality reduction.
    
    Parameters:
    - X: input data matrix
    - n_factors: number of latent factors to extract
    - feature_names: names of features (for interpretation)
    
    Returns:
    - X_factors: transformed data (factor scores)
    - loadings: factor loadings matrix
    - fa: fitted FactorAnalysis model
    """
    
    # Step 1: Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 2: Fit Factor Analysis
    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    X_factors = fa.fit_transform(X_scaled)
    
    # Step 3: Get loadings
    loadings = fa.components_.T  # Shape: (n_features, n_factors)
    
    # Step 4: Create loadings DataFrame for interpretation
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    factor_names = [f'Factor_{i+1}' for i in range(n_factors)]
    loadings_df = pd.DataFrame(loadings, 
                                index=feature_names, 
                                columns=factor_names)
    
    return X_factors, loadings_df, fa


def interpret_factors(loadings_df, threshold=0.5):
    """
    Interpret factors based on loadings.
    High absolute loadings indicate strong relationship.
    """
    print("\n=== Factor Interpretation ===")
    for factor in loadings_df.columns:
        print(f"\n{factor}:")
        # Get features with high loadings
        high_loadings = loadings_df[factor][abs(loadings_df[factor]) > threshold]
        high_loadings_sorted = high_loadings.sort_values(key=abs, ascending=False)
        for feature, loading in high_loadings_sorted.items():
            direction = "+" if loading > 0 else "-"
            print(f"  {direction} {feature}: {loading:.3f}")


# Example usage
if __name__ == "__main__":
    # Load data
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    # Apply Factor Analysis
    X_factors, loadings_df, fa = factor_analysis_reduction(
        X, n_factors=2, feature_names=feature_names
    )
    
    print("=== Factor Analysis Results ===")
    print(f"\nOriginal dimensions: {X.shape[1]}")
    print(f"Reduced dimensions: {X_factors.shape[1]}")
    
    print("\n=== Factor Loadings ===")
    print(loadings_df.round(3))
    
    # Interpret factors
    interpret_factors(loadings_df, threshold=0.4)
    
    # Visualize
    plt.figure(figsize=(12, 4))
    
    # Loadings heatmap
    plt.subplot(1, 2, 1)
    plt.imshow(loadings_df.values, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Loading')
    plt.xticks(range(len(loadings_df.columns)), loadings_df.columns)
    plt.yticks(range(len(loadings_df.index)), loadings_df.index)
    plt.title('Factor Loadings')
    
    # Scatter plot of factors
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_factors[:, 0], X_factors[:, 1], c=y, cmap='viridis')
    plt.colorbar(scatter, label='Class')
    plt.xlabel('Factor 1')
    plt.ylabel('Factor 2')
    plt.title('Factor Scores')
    
    plt.tight_layout()
    plt.show()
```

**Factor Analysis vs PCA:**

| Aspect | Factor Analysis | PCA |
|--------|-----------------|-----|
| Assumption | Latent factors + noise | Linear combinations |
| Noise | Models explicitly | Not modeled |
| Loadings | More interpretable | Rotational freedom |
| Use case | Psychology, social sciences | General DR |

---

## Question 8
- [ ] Done

**Write a code snippet to perform feature extraction using Non-negative Matrix Factorization (NMF).**

### Answer

**Pipeline:**
1. Ensure data is non-negative
2. Apply NMF to decompose X ≈ W × H
3. W = transformed data (samples × components)
4. H = component features (components × features)
5. Analyze components for interpretation

```python
import numpy as np
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

def nmf_feature_extraction(X, n_components=10, feature_names=None):
    """
    Perform Non-negative Matrix Factorization.
    
    X ≈ W × H
    - W: (n_samples, n_components) - transformed data
    - H: (n_components, n_features) - component basis
    
    Parameters:
    - X: non-negative input matrix
    - n_components: number of components to extract
    - feature_names: for interpreting components
    
    Returns:
    - W: transformed data
    - H: component matrix
    - nmf: fitted model
    """
    
    # Ensure non-negative
    if (X < 0).any():
        raise ValueError("NMF requires non-negative data!")
    
    # Fit NMF
    nmf = NMF(n_components=n_components, random_state=42, max_iter=500)
    W = nmf.fit_transform(X)  # Reduced representation
    H = nmf.components_       # Component features
    
    # Reconstruction error
    reconstruction_error = nmf.reconstruction_err_
    print(f"Reconstruction error: {reconstruction_error:.4f}")
    
    return W, H, nmf


def display_top_features(H, feature_names, n_top=10):
    """Display top features for each component."""
    for i, component in enumerate(H):
        top_idx = component.argsort()[-n_top:][::-1]
        top_features = [feature_names[j] for j in top_idx]
        print(f"\nComponent {i+1}:")
        print(", ".join(top_features))


# Example 1: NMF on text data (Topic Modeling)
print("=== NMF for Topic Modeling ===")

# Load text data
categories = ['sci.space', 'rec.sport.baseball', 'comp.graphics']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, 
                                 remove=('headers', 'footers', 'quotes'))

# TF-IDF (produces non-negative matrix)
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = tfidf.fit_transform(newsgroups.data)

print(f"TF-IDF matrix shape: {X_tfidf.shape}")

# Apply NMF
n_topics = 3
W, H, nmf_model = nmf_feature_extraction(X_tfidf, n_components=n_topics)

print(f"\nTransformed shape: {W.shape}")
print(f"Components shape: {H.shape}")

# Display topics
feature_names = tfidf.get_feature_names_out()
display_top_features(H, feature_names, n_top=8)


# Example 2: NMF on numerical data
print("\n\n=== NMF on Numerical Data ===")

# Create non-negative data
np.random.seed(42)
X_numeric = np.abs(np.random.randn(100, 20))  # Ensure non-negative

W_num, H_num, _ = nmf_feature_extraction(X_numeric, n_components=5)

print(f"Original shape: {X_numeric.shape}")
print(f"Reduced shape: {W_num.shape}")

# Visualize components
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(H_num, aspect='auto', cmap='viridis')
plt.xlabel('Original Features')
plt.ylabel('Components')
plt.title('NMF Components (H matrix)')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.scatter(W_num[:, 0], W_num[:, 1], alpha=0.5)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Transformed Data (W matrix)')

plt.tight_layout()
plt.show()
```

**NMF Key Properties:**
- Requires non-negative input
- Parts-based representation (additive)
- Good for: topic modeling, image decomposition, recommendation
- Interpretable components (all positive contributions)

---

## Question 9
- [ ] Done

**Use the feature importance provided by a trained ensemble model to reduce the dimensionality of a dataset in Python.**

### Answer

**Pipeline:**
1. Train ensemble model (Random Forest / Gradient Boosting)
2. Extract feature importances
3. Select top-k features or use threshold
4. Create reduced dataset
5. Validate with model performance

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def ensemble_feature_selection(X, y, feature_names=None, 
                                n_features=None, threshold='median'):
    """
    Use ensemble model feature importance for dimensionality reduction.
    
    Parameters:
    - X: feature matrix
    - y: target
    - feature_names: names of features
    - n_features: number of features to select (if None, use threshold)
    - threshold: 'mean', 'median', or float value
    
    Returns:
    - X_reduced: reduced feature matrix
    - selected_features: names/indices of selected features
    - importances: all feature importances
    """
    
    # Step 1: Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Step 2: Get feature importances
    importances = rf.feature_importances_
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Step 3: Select features
    if n_features is not None:
        # Select top n features
        top_features = importance_df.head(n_features)['feature'].values
        mask = np.isin(feature_names, top_features)
    else:
        # Use SelectFromModel with threshold
        selector = SelectFromModel(rf, threshold=threshold, prefit=True)
        mask = selector.get_support()
    
    X_reduced = X[:, mask]
    selected_features = np.array(feature_names)[mask]
    
    return X_reduced, selected_features, importance_df


# Example usage
if __name__ == "__main__":
    # Load data
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    print(f"Original shape: {X.shape}")
    
    # Method 1: Select top k features
    X_reduced_k, selected_k, importances = ensemble_feature_selection(
        X, y, feature_names, n_features=10
    )
    print(f"\nTop 10 features shape: {X_reduced_k.shape}")
    
    # Method 2: Use threshold
    X_reduced_th, selected_th, _ = ensemble_feature_selection(
        X, y, feature_names, threshold='median'
    )
    print(f"Threshold (median) shape: {X_reduced_th.shape}")
    
    # Display top features
    print("\n=== Top 15 Feature Importances ===")
    print(importances.head(15).to_string(index=False))
    
    # Compare performance
    print("\n=== Performance Comparison ===")
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    
    score_all = cross_val_score(clf, X, y, cv=5).mean()
    score_top10 = cross_val_score(clf, X_reduced_k, y, cv=5).mean()
    score_threshold = cross_val_score(clf, X_reduced_th, y, cv=5).mean()
    
    print(f"All features ({X.shape[1]}):        Accuracy = {score_all:.4f}")
    print(f"Top 10 features:           Accuracy = {score_top10:.4f}")
    print(f"Threshold features ({X_reduced_th.shape[1]}):  Accuracy = {score_threshold:.4f}")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    # Top 15 feature importances
    plt.subplot(1, 2, 1)
    top15 = importances.head(15)
    plt.barh(range(15), top15['importance'].values)
    plt.yticks(range(15), top15['feature'].values)
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importances')
    plt.gca().invert_yaxis()
    
    # Cumulative importance
    plt.subplot(1, 2, 2)
    cumsum = importances['importance'].cumsum()
    plt.plot(range(1, len(cumsum)+1), cumsum.values)
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% importance')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.title('Cumulative Feature Importance')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
```

**Key Points:**
- Random Forest provides MDI (Mean Decrease in Impurity)
- Can also use permutation importance (more reliable but slower)
- SelectFromModel automates threshold-based selection
- Often achieves similar accuracy with fewer features

---
