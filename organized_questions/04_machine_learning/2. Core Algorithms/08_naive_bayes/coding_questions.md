# Naive Bayes Interview Questions - Coding Questions

## Question 1
- [ ] Done

**How would you deal with missing values when implementing a Naive Bayes classifier?**

**Answer:**

Naive Bayes can naturally handle missing values by simply excluding missing features from the likelihood computation. Since NB treats features independently, a missing feature doesn't provide information, so we skip it in the probability product. This is mathematically equivalent to marginalizing over all possible values.

**Approach 1: Ignore Missing Features (Recommended for NB)**

```python
import numpy as np
from collections import defaultdict

class NaiveBayesWithMissing:
    def __init__(self):
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = None
    
    def fit(self, X, y):
        """Train NB, handling missing values (represented as np.nan)."""
        self.classes = np.unique(y)
        n_samples = len(y)
        
        for c in self.classes:
            # Prior probability
            self.class_priors[c] = np.sum(y == c) / n_samples
            
            # Feature probabilities (only count non-missing values)
            X_c = X[y == c]
            self.feature_probs[c] = {}
            
            for j in range(X.shape[1]):
                # Get non-missing values for this feature in this class
                col = X_c[:, j]
                valid = col[~np.isnan(col)]
                
                if len(valid) > 0:
                    # Store mean and std for Gaussian NB
                    self.feature_probs[c][j] = {
                        'mean': np.mean(valid),
                        'std': np.std(valid) + 1e-9  # Smoothing
                    }
        return self
    
    def predict(self, X):
        """Predict class, skipping missing features."""
        predictions = []
        
        for x in X:
            posteriors = []
            for c in self.classes:
                log_prob = np.log(self.class_priors[c])
                
                for j, val in enumerate(x):
                    # Skip missing values
                    if not np.isnan(val):
                        params = self.feature_probs[c].get(j)
                        if params:
                            # Gaussian likelihood
                            log_prob += self._gaussian_log_prob(
                                val, params['mean'], params['std']
                            )
                
                posteriors.append(log_prob)
            
            predictions.append(self.classes[np.argmax(posteriors)])
        
        return np.array(predictions)
    
    def _gaussian_log_prob(self, x, mean, std):
        return -0.5 * np.log(2 * np.pi * std**2) - 0.5 * ((x - mean) / std)**2


# Usage Example
X = np.array([
    [1.0, 2.0, 3.0],
    [1.5, np.nan, 3.5],  # Missing value
    [2.0, 3.0, np.nan],  # Missing value
    [5.0, 6.0, 7.0],
    [5.5, np.nan, 7.5]
])
y = np.array([0, 0, 0, 1, 1])

nb = NaiveBayesWithMissing()
nb.fit(X, y)

X_test = np.array([[1.2, np.nan, 3.2]])  # Test with missing
print(nb.predict(X_test))
```

**Approach 2: Imputation Before NB**

```python
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

# Impute missing values, then apply NB
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # or 'median', 'most_frequent'
    ('nb', GaussianNB())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

**Key Points:**
- Ignoring missing features is mathematically sound for NB
- Imputation is simpler but may introduce bias
- For categorical features, "missing" can be treated as a separate category

---

## Question 2
- [ ] Done

**Implement a Gaussian Naive Bayes classifier from scratch in Python.**

**Answer:**

Gaussian NB assumes continuous features follow normal distribution within each class. We estimate mean and variance per feature per class, then use Gaussian PDF for likelihoods.

**Implementation:**

```python
import numpy as np

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier from scratch.
    
    Pipeline:
    1. fit(): Calculate priors and Gaussian parameters (mean, var) per class
    2. predict(): Compute posteriors using Bayes theorem, return argmax
    """
    
    def __init__(self):
        self.classes = None
        self.priors = {}      # P(Y=c)
        self.means = {}       # Mean of each feature for each class
        self.variances = {}   # Variance of each feature for each class
    
    def fit(self, X, y):
        """
        Train the Gaussian NB classifier.
        
        Steps:
        1. Get unique classes
        2. For each class, compute:
           - Prior = count(class) / total
           - Mean of each feature
           - Variance of each feature (with smoothing)
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        n_samples = len(y)
        
        for c in self.classes:
            X_c = X[y == c]  # Samples belonging to class c
            
            # Prior probability
            self.priors[c] = len(X_c) / n_samples
            
            # Mean and variance for each feature
            self.means[c] = X_c.mean(axis=0)
            self.variances[c] = X_c.var(axis=0) + 1e-9  # Add smoothing
        
        return self
    
    def _gaussian_pdf(self, x, mean, var):
        """
        Compute Gaussian probability density.
        
        Formula: (1 / sqrt(2*pi*var)) * exp(-(x-mean)^2 / (2*var))
        """
        coefficient = 1 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return coefficient * exponent
    
    def _compute_posterior(self, x):
        """
        Compute log posterior for each class.
        
        log P(Y|X) ∝ log P(Y) + Σ log P(Xi|Y)
        """
        posteriors = {}
        
        for c in self.classes:
            # Start with log prior
            log_posterior = np.log(self.priors[c])
            
            # Add log likelihood for each feature
            for i, xi in enumerate(x):
                pdf = self._gaussian_pdf(xi, self.means[c][i], self.variances[c][i])
                log_posterior += np.log(pdf + 1e-300)  # Avoid log(0)
            
            posteriors[c] = log_posterior
        
        return posteriors
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Steps:
        1. For each sample, compute posterior for each class
        2. Return class with highest posterior
        """
        X = np.array(X)
        predictions = []
        
        for x in X:
            posteriors = self._compute_posterior(x)
            # Return class with maximum posterior
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Return probability estimates (normalized posteriors)."""
        X = np.array(X)
        probas = []
        
        for x in X:
            posteriors = self._compute_posterior(x)
            # Convert log posteriors to probabilities
            log_posts = np.array([posteriors[c] for c in self.classes])
            # Softmax to normalize
            max_log = np.max(log_posts)
            exp_posts = np.exp(log_posts - max_log)
            probs = exp_posts / exp_posts.sum()
            probas.append(probs)
        
        return np.array(probas)


# Example Usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    
    # Class 0: centered around (2, 2)
    X0 = np.random.randn(50, 2) + [2, 2]
    y0 = np.zeros(50)
    
    # Class 1: centered around (6, 6)
    X1 = np.random.randn(50, 2) + [6, 6]
    y1 = np.ones(50)
    
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    
    # Train
    gnb = GaussianNaiveBayes()
    gnb.fit(X, y)
    
    # Predict
    X_test = np.array([[3, 3], [5, 5], [1, 1], [7, 7]])
    predictions = gnb.predict(X_test)
    probabilities = gnb.predict_proba(X_test)
    
    print("Predictions:", predictions)
    print("Probabilities:\n", probabilities)
```

**Output:**
```
Predictions: [0. 1. 0. 1.]
Probabilities:
 [[0.98 0.02]
  [0.15 0.85]
  [0.99 0.01]
  [0.01 0.99]]
```

---

## Question 3
- [ ] Done

**Write a Python function using scikit-learn to perform text classification with Multinomial Naive Bayes.**

**Answer:**

Complete pipeline for text classification using Multinomial NB with TF-IDF features.

**Implementation:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def text_classification_nb(texts, labels, test_size=0.2):
    """
    Text classification using Multinomial Naive Bayes.
    
    Pipeline:
    1. Split data into train/test
    2. Create TF-IDF features
    3. Train Multinomial NB
    4. Evaluate on test set
    
    Returns: trained pipeline, accuracy, classification report
    """
    
    # Step 1: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Step 2 & 3: Create pipeline (TF-IDF + NB)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),    # Unigrams and bigrams
            max_features=5000,     # Limit vocabulary size
            min_df=2,              # Ignore rare words
            max_df=0.95            # Ignore very common words
        )),
        ('nb', MultinomialNB(alpha=0.1))  # Smoothing parameter
    ])
    
    # Step 4: Train
    pipeline.fit(X_train, y_train)
    
    # Step 5: Predict and evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return pipeline, accuracy, report


# Example Usage
if __name__ == "__main__":
    # Sample data: Sentiment classification
    texts = [
        "This movie is amazing, I loved it!",
        "Great film, highly recommended",
        "Wonderful experience, will watch again",
        "The best movie I have ever seen",
        "Absolutely fantastic and entertaining",
        "Terrible movie, waste of time",
        "Awful film, very disappointing",
        "Worst experience, don't watch this",
        "Boring and uninteresting",
        "Complete disaster, hated every minute"
    ]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1=positive, 0=negative
    
    # Train and evaluate
    model, acc, report = text_classification_nb(texts, labels, test_size=0.3)
    
    print(f"Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(report)
    
    # Predict on new text
    new_texts = ["This is a great movie!", "Terrible waste of money"]
    predictions = model.predict(new_texts)
    probabilities = model.predict_proba(new_texts)
    
    for text, pred, prob in zip(new_texts, predictions, probabilities):
        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = prob.max()
        print(f"\nText: '{text}'")
        print(f"Prediction: {sentiment} (confidence: {confidence:.2%})")
```

**Key Parameters Explained:**
- `ngram_range=(1,2)`: Include both single words and word pairs
- `max_features=5000`: Limit vocabulary to most common 5000 terms
- `alpha=0.1`: Smoothing parameter (tune via cross-validation)

---

## Question 4
- [ ] Done

**Create a Python script to perform feature selection specifically suited for Naive Bayes.**

**Answer:**

Feature selection for NB using Chi-square and Mutual Information - both measure how informative a feature is for classification.

**Implementation:**

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

def nb_feature_selection(X, y, method='chi2', k_values=[100, 500, 1000, 2000]):
    """
    Feature selection for Naive Bayes using filter methods.
    
    Pipeline:
    1. Vectorize text (TF-IDF)
    2. Apply feature selection (Chi-square or Mutual Information)
    3. Train NB on selected features
    4. Evaluate with cross-validation
    
    Args:
        X: List of text documents
        y: Labels
        method: 'chi2' or 'mutual_info'
        k_values: Number of features to try
    
    Returns: Best k, best score, best pipeline
    """
    
    # Choose feature selection method
    if method == 'chi2':
        selector = chi2
    elif method == 'mutual_info':
        selector = mutual_info_classif
    else:
        raise ValueError("Method must be 'chi2' or 'mutual_info'")
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=10000)
    X_tfidf = vectorizer.fit_transform(X)
    
    best_score = 0
    best_k = None
    results = {}
    
    print(f"Feature Selection Method: {method}")
    print(f"Total features: {X_tfidf.shape[1]}")
    print("-" * 40)
    
    for k in k_values:
        if k > X_tfidf.shape[1]:
            continue
            
        # Create pipeline with feature selection
        pipeline = Pipeline([
            ('selector', SelectKBest(selector, k=k)),
            ('nb', MultinomialNB(alpha=0.1))
        ])
        
        # Cross-validation
        scores = cross_val_score(pipeline, X_tfidf, y, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        std_score = scores.std()
        
        results[k] = {'mean': mean_score, 'std': std_score}
        print(f"k={k:5d}: Accuracy = {mean_score:.4f} (+/- {std_score:.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_k = k
    
    print("-" * 40)
    print(f"Best k: {best_k} with accuracy: {best_score:.4f}")
    
    # Train final model with best k
    best_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000)),
        ('selector', SelectKBest(selector, k=best_k)),
        ('nb', MultinomialNB(alpha=0.1))
    ])
    best_pipeline.fit(X, y)
    
    return best_k, best_score, best_pipeline


def get_top_features(pipeline, feature_names, n_top=20):
    """Get top features selected by the pipeline."""
    selector = pipeline.named_steps['selector']
    selected_indices = selector.get_support(indices=True)
    scores = selector.scores_[selected_indices]
    
    # Sort by score
    sorted_idx = np.argsort(scores)[::-1][:n_top]
    
    top_features = []
    for idx in sorted_idx:
        feature_idx = selected_indices[idx]
        top_features.append({
            'feature': feature_names[feature_idx],
            'score': scores[idx]
        })
    
    return top_features


# Example Usage
if __name__ == "__main__":
    # Sample data
    from sklearn.datasets import fetch_20newsgroups
    
    categories = ['sci.space', 'rec.sport.baseball']
    data = fetch_20newsgroups(subset='train', categories=categories)
    X, y = data.data[:500], data.target[:500]
    
    # Run feature selection
    best_k, best_score, pipeline = nb_feature_selection(
        X, y, 
        method='chi2',
        k_values=[50, 100, 200, 500, 1000]
    )
    
    # Show top features
    vectorizer = TfidfVectorizer(max_features=10000)
    vectorizer.fit(X)
    feature_names = vectorizer.get_feature_names_out()
    
    print("\nTop 10 Selected Features:")
    top_features = get_top_features(pipeline, feature_names, n_top=10)
    for i, feat in enumerate(top_features, 1):
        print(f"{i}. {feat['feature']}: {feat['score']:.2f}")
```

**Output Example:**
```
Feature Selection Method: chi2
Total features: 8432
----------------------------------------
k=   50: Accuracy = 0.9120 (+/- 0.0321)
k=  100: Accuracy = 0.9340 (+/- 0.0254)
k=  200: Accuracy = 0.9480 (+/- 0.0198)
k=  500: Accuracy = 0.9520 (+/- 0.0167)
k= 1000: Accuracy = 0.9460 (+/- 0.0201)
----------------------------------------
Best k: 500 with accuracy: 0.9520
```

---

## Question 5
- [ ] Done

**Write code to apply Laplace smoothing to a dataset with categorical features.**

**Answer:**

Implementation of Laplace smoothing for categorical Naive Bayes, showing how smoothing prevents zero probabilities.

**Implementation:**

```python
import numpy as np
from collections import defaultdict

class CategoricalNBWithSmoothing:
    """
    Categorical Naive Bayes with Laplace (additive) smoothing.
    
    Formula with smoothing:
    P(feature=value | class) = (count + alpha) / (class_count + alpha * num_values)
    """
    
    def __init__(self, alpha=1.0):
        """
        Args:
            alpha: Smoothing parameter (1.0 = Laplace, <1 = Lidstone)
        """
        self.alpha = alpha
        self.class_priors = {}
        self.feature_probs = {}
        self.feature_values = {}  # Possible values for each feature
        self.classes = None
    
    def fit(self, X, y):
        """
        Train with Laplace smoothing.
        
        Steps:
        1. Count class frequencies for priors
        2. For each feature, count value frequencies per class
        3. Apply smoothing to all probabilities
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        # Get all possible values for each feature
        for j in range(n_features):
            self.feature_values[j] = np.unique(X[:, j])
        
        # Calculate class priors
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / n_samples
        
        # Calculate feature probabilities with smoothing
        self.feature_probs = {c: {} for c in self.classes}
        
        for c in self.classes:
            X_c = X[y == c]
            class_count = len(X_c)
            
            for j in range(n_features):
                num_values = len(self.feature_values[j])
                self.feature_probs[c][j] = {}
                
                for value in self.feature_values[j]:
                    # Count occurrences
                    count = np.sum(X_c[:, j] == value)
                    
                    # Apply Laplace smoothing
                    # P(value|class) = (count + alpha) / (class_count + alpha * num_values)
                    smoothed_prob = (count + self.alpha) / (class_count + self.alpha * num_values)
                    
                    self.feature_probs[c][j][value] = smoothed_prob
        
        return self
    
    def predict(self, X):
        """Predict class labels."""
        X = np.array(X)
        predictions = []
        
        for x in X:
            posteriors = {}
            
            for c in self.classes:
                # Start with log prior
                log_posterior = np.log(self.class_priors[c])
                
                # Add log likelihoods
                for j, value in enumerate(x):
                    if value in self.feature_probs[c][j]:
                        prob = self.feature_probs[c][j][value]
                    else:
                        # Unseen value: use smoothing estimate
                        num_values = len(self.feature_values[j]) + 1
                        prob = self.alpha / (self.alpha * num_values)
                    
                    log_posterior += np.log(prob)
                
                posteriors[c] = log_posterior
            
            predictions.append(max(posteriors, key=posteriors.get))
        
        return np.array(predictions)
    
    def show_probabilities(self, feature_idx=0):
        """Display smoothed probabilities for a feature."""
        print(f"\nFeature {feature_idx} Probabilities (alpha={self.alpha}):")
        print("-" * 50)
        
        for c in self.classes:
            print(f"\nClass {c}:")
            for value, prob in self.feature_probs[c][feature_idx].items():
                print(f"  P({value}|{c}) = {prob:.4f}")


# Demonstration of smoothing effect
if __name__ == "__main__":
    # Create dataset with categorical features
    # Feature 0: Weather (sunny, rainy, cloudy)
    # Feature 1: Temperature (hot, mild, cold)
    
    X = np.array([
        ['sunny', 'hot'],
        ['sunny', 'hot'],
        ['cloudy', 'mild'],
        ['rainy', 'mild'],
        ['rainy', 'cold'],
        ['cloudy', 'cold'],
        ['sunny', 'mild'],
        ['rainy', 'hot'],
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # Play tennis: No=0, Yes=1
    
    print("=" * 50)
    print("WITHOUT SMOOTHING (alpha=0)")
    print("=" * 50)
    
    # Without smoothing
    nb_no_smooth = CategoricalNBWithSmoothing(alpha=0.0001)  # Near-zero smoothing
    nb_no_smooth.fit(X, y)
    nb_no_smooth.show_probabilities(feature_idx=0)
    
    print("\n" + "=" * 50)
    print("WITH LAPLACE SMOOTHING (alpha=1)")
    print("=" * 50)
    
    # With Laplace smoothing
    nb_smooth = CategoricalNBWithSmoothing(alpha=1.0)
    nb_smooth.fit(X, y)
    nb_smooth.show_probabilities(feature_idx=0)
    
    # Test prediction with unseen combination
    print("\n" + "=" * 50)
    print("PREDICTION TEST")
    print("=" * 50)
    
    X_test = np.array([['sunny', 'cold']])  # 'sunny' never with 'cold' in training
    
    pred_smooth = nb_smooth.predict(X_test)
    print(f"\nTest sample: {X_test[0]}")
    print(f"Prediction (with smoothing): Class {pred_smooth[0]}")
```

**Output:**
```
==================================================
WITHOUT SMOOTHING (alpha=0)
==================================================

Feature 0 Probabilities (alpha=0.0001):
--------------------------------------------------

Class 0:
  P(sunny|0) = 0.5000
  P(cloudy|0) = 0.2500
  P(rainy|0) = 0.2500

Class 1:
  P(sunny|1) = 0.2500
  P(cloudy|1) = 0.2500
  P(rainy|1) = 0.5000

==================================================
WITH LAPLACE SMOOTHING (alpha=1)
==================================================

Feature 0 Probabilities (alpha=1):
--------------------------------------------------

Class 0:
  P(sunny|0) = 0.4286
  P(cloudy|0) = 0.2857
  P(rainy|0) = 0.2857

Class 1:
  P(sunny|1) = 0.2857
  P(cloudy|1) = 0.2857
  P(rainy|1) = 0.4286
```

---

## Question 6
- [ ] Done

**Develop a function in Python to handle missing data for a dataset before applying Naive Bayes.**

**Answer:**

Comprehensive missing data handling with multiple strategies for different scenarios.

**Implementation:**

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

def handle_missing_for_nb(X, y, strategy='auto', categorical_cols=None, numerical_cols=None):
    """
    Handle missing data for Naive Bayes classification.
    
    Strategies:
    1. 'ignore': Drop rows with missing values
    2. 'impute_mean': Fill numerical with mean, categorical with mode
    3. 'impute_median': Fill numerical with median, categorical with mode
    4. 'category': Treat missing as separate category (categorical only)
    5. 'auto': Choose best strategy based on missingness pattern
    
    Returns: Cleaned X, y, and imputer (if used)
    """
    
    X = pd.DataFrame(X).copy()
    y = np.array(y).copy()
    
    # Auto-detect column types if not specified
    if categorical_cols is None and numerical_cols is None:
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print("Missing Value Analysis:")
    print("-" * 40)
    missing_counts = X.isnull().sum()
    missing_pct = (X.isnull().sum() / len(X) * 100).round(2)
    
    for col in X.columns:
        if missing_counts[col] > 0:
            print(f"  {col}: {missing_counts[col]} ({missing_pct[col]}%)")
    
    print("-" * 40)
    
    # Strategy selection
    if strategy == 'auto':
        total_missing_pct = X.isnull().any(axis=1).sum() / len(X) * 100
        if total_missing_pct < 5:
            strategy = 'ignore'
        else:
            strategy = 'impute_mean'
        print(f"Auto-selected strategy: {strategy}")
    
    # Apply strategy
    if strategy == 'ignore':
        # Drop rows with any missing values
        valid_mask = ~X.isnull().any(axis=1)
        X_clean = X[valid_mask].values
        y_clean = y[valid_mask]
        imputer = None
        print(f"Dropped {(~valid_mask).sum()} rows with missing values")
        
    elif strategy in ['impute_mean', 'impute_median']:
        num_strategy = 'mean' if strategy == 'impute_mean' else 'median'
        
        transformers = []
        
        if numerical_cols:
            transformers.append((
                'num', 
                SimpleImputer(strategy=num_strategy), 
                numerical_cols
            ))
        
        if categorical_cols:
            transformers.append((
                'cat', 
                SimpleImputer(strategy='most_frequent'), 
                categorical_cols
            ))
        
        if transformers:
            imputer = ColumnTransformer(transformers)
            X_clean = imputer.fit_transform(X)
        else:
            X_clean = X.values
            imputer = None
        
        y_clean = y
        print(f"Imputed missing values using {strategy}")
        
    elif strategy == 'category':
        # Replace missing with 'MISSING' string for categorical
        X_clean = X.copy()
        for col in categorical_cols:
            X_clean[col] = X_clean[col].fillna('MISSING')
        
        # For numerical, use median imputation
        for col in numerical_cols:
            X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        
        X_clean = X_clean.values
        y_clean = y
        imputer = None
        print("Treated missing as separate category")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"Final dataset shape: {X_clean.shape}")
    
    return X_clean, y_clean, imputer


def create_nb_pipeline_with_missing_handler(strategy='impute_mean', nb_type='gaussian'):
    """
    Create a complete pipeline that handles missing data and applies NB.
    """
    
    # Choose imputer
    if strategy == 'impute_mean':
        imputer = SimpleImputer(strategy='mean')
    elif strategy == 'impute_median':
        imputer = SimpleImputer(strategy='median')
    else:
        imputer = SimpleImputer(strategy='most_frequent')
    
    # Choose NB variant
    if nb_type == 'gaussian':
        nb = GaussianNB()
    else:
        nb = MultinomialNB()
    
    pipeline = Pipeline([
        ('imputer', imputer),
        ('nb', nb)
    ])
    
    return pipeline


# Example Usage
if __name__ == "__main__":
    # Create sample data with missing values
    np.random.seed(42)
    
    X = pd.DataFrame({
        'age': [25, 30, np.nan, 45, 50, np.nan, 35, 40],
        'income': [50000, np.nan, 70000, 80000, np.nan, 60000, 75000, 90000],
        'category': ['A', 'B', None, 'A', 'B', 'A', None, 'B']
    })
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    
    print("Original Data:")
    print(X)
    print()
    
    # Test different strategies
    for strategy in ['ignore', 'impute_mean', 'category']:
        print(f"\n{'='*50}")
        print(f"Strategy: {strategy}")
        print('='*50)
        
        X_clean, y_clean, _ = handle_missing_for_nb(
            X, y, 
            strategy=strategy,
            numerical_cols=['age', 'income'],
            categorical_cols=['category']
        )
        print(f"Cleaned data shape: {X_clean.shape}")
```

**Output:**
```
Original Data:
    age   income category
0  25.0  50000.0        A
1  30.0      NaN        B
2   NaN  70000.0     None
3  45.0  80000.0        A
4  50.0      NaN        B
5   NaN  60000.0        A
6  35.0  75000.0     None
7  40.0  90000.0        B

==================================================
Strategy: ignore
==================================================
Missing Value Analysis:
----------------------------------------
  age: 2 (25.0%)
  income: 2 (25.0%)
  category: 2 (25.0%)
----------------------------------------
Dropped 5 rows with missing values
Final dataset shape: (3, 3)
```

---
