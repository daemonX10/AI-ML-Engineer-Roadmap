# Naive Bayes Interview Questions - Scenario-Based Questions

## Question 1
- [ ] Done

**Discuss how the Naive Bayes classifier can be applied to recommendation systems.**

**Answer:**

Naive Bayes can power content-based and collaborative filtering recommendations by predicting whether a user will like an item based on features (content-based) or similar users' preferences (collaborative). It's fast, handles sparse data well, and provides probability scores for ranking recommendations.

**Scenario: Movie Recommendation System**

**Approach 1: Content-Based Filtering**
```
Goal: Predict if User A will like Movie X based on movie features

Features of Movie X: [Action=1, Comedy=0, Year=2020, Rating=8.5]
User A's preference: Learned P(Like | feature) from their history

P(Like | Movie X) = P(Like) × ∏ P(feature | Like)
```

**Implementation:**
```python
import numpy as np
from sklearn.naive_bayes import MultinomialNB

# Movie features (one-hot encoded genres)
# Columns: Action, Comedy, Drama, Sci-Fi, Romance
movie_features = np.array([
    [1, 0, 0, 1, 0],  # Movie 1: Action, Sci-Fi
    [0, 1, 1, 0, 0],  # Movie 2: Comedy, Drama
    [1, 0, 0, 0, 0],  # Movie 3: Action
    [0, 0, 1, 0, 1],  # Movie 4: Drama, Romance
])

# User ratings (1=liked, 0=not liked)
user_ratings = np.array([1, 0, 1, 0])

# Train NB on user's preferences
nb = MultinomialNB()
nb.fit(movie_features, user_ratings)

# Recommend: predict for new movies
new_movies = np.array([
    [1, 0, 0, 1, 0],  # Action, Sci-Fi - should recommend
    [0, 1, 0, 0, 1],  # Comedy, Romance - probably not
])

# Get like probabilities
probs = nb.predict_proba(new_movies)[:, 1]  # P(Like)
print("Recommendation scores:", probs)
```

**Approach 2: Collaborative Filtering (User-Based)**
```python
def collaborative_nb(user_item_matrix, target_user, item):
    """
    Predict if target_user will like item based on similar users.
    
    Features: Other items the user rated
    Target: Rating for the specific item
    """
    # Users who rated this item
    users_rated = user_item_matrix[:, item] != 0
    
    if users_rated.sum() < 5:
        return 0.5  # Not enough data
    
    # Features: ratings on other items
    X = user_item_matrix[users_rated][:, np.arange(user_item_matrix.shape[1]) != item]
    y = (user_item_matrix[users_rated, item] > 3).astype(int)  # Liked if rating > 3
    
    nb = MultinomialNB()
    nb.fit(X, y)
    
    # Predict for target user
    target_features = user_item_matrix[target_user:target_user+1, np.arange(user_item_matrix.shape[1]) != item]
    prob_like = nb.predict_proba(target_features)[0, 1]
    
    return prob_like
```

**Advantages of NB for Recommendations:**
- Fast training and prediction (real-time recommendations)
- Handles sparse user-item matrices
- Provides probability scores for ranking
- Works with limited user history

**Limitations:**
- Independence assumption may not capture complex preferences
- Doesn't model user-item interactions well
- Modern systems prefer deep learning or matrix factorization

---

## Question 2
- [ ] Done

**How would you use Naive Bayes to build an email categorization system (e.g., important, social, promotions)?**

**Answer:**

Build a multi-class text classifier that categorizes emails into folders like Gmail's Primary, Social, Promotions, Updates. Use Multinomial NB with TF-IDF features extracted from email subject, body, and metadata. Train on user-labeled emails to learn category-specific word patterns.

**System Design:**

```
Email → Preprocessing → Feature Extraction → NB Classifier → Category
                              ↓
              [Subject, Body, Sender, Headers]
                              ↓
              [TF-IDF, Metadata Features]
```

**Implementation:**

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

class EmailFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from email text and metadata."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, emails):
        # Combine subject and body
        texts = [f"{email['subject']} {email['body']}" for email in emails]
        return texts

class MetadataFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract metadata features (sender domain, etc.)."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, emails):
        features = []
        for email in emails:
            sender_domain = email['sender'].split('@')[-1] if '@' in email['sender'] else 'unknown'
            features.append(f"domain_{sender_domain}")
        return features


def build_email_classifier():
    """
    Build email categorization pipeline.
    
    Categories:
    - 0: Primary (important)
    - 1: Social (social networks)
    - 2: Promotions (marketing)
    - 3: Updates (notifications, receipts)
    """
    
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text', Pipeline([
                ('extract', EmailFeatureExtractor()),
                ('tfidf', TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=5000,
                    stop_words='english'
                ))
            ])),
            ('metadata', Pipeline([
                ('extract', MetadataFeatureExtractor()),
                ('tfidf', TfidfVectorizer())  # Vectorize domain features
            ]))
        ])),
        ('classifier', MultinomialNB(alpha=0.1))
    ])
    
    return pipeline


# Example usage
if __name__ == "__main__":
    # Sample training data
    emails = [
        {'subject': 'Meeting tomorrow', 'body': 'Please join the project meeting', 'sender': 'boss@company.com'},
        {'subject': 'John liked your post', 'body': 'See what John commented', 'sender': 'notify@facebook.com'},
        {'subject': '50% OFF Sale!', 'body': 'Limited time offer on all items', 'sender': 'deals@shop.com'},
        {'subject': 'Your order shipped', 'body': 'Track your package here', 'sender': 'orders@amazon.com'},
        {'subject': 'Quarterly report', 'body': 'Attached is the Q3 report', 'sender': 'cfo@company.com'},
        {'subject': 'New follower', 'body': 'Jane started following you', 'sender': 'notify@twitter.com'},
    ]
    labels = [0, 1, 2, 3, 0, 1]  # Primary, Social, Promotion, Update
    
    # Build and train
    classifier = build_email_classifier()
    classifier.fit(emails, labels)
    
    # Classify new email
    new_email = [{'subject': 'Flash Sale Today!', 'body': 'Buy now and save big', 'sender': 'promo@store.com'}]
    prediction = classifier.predict(new_email)
    
    categories = ['Primary', 'Social', 'Promotions', 'Updates']
    print(f"Category: {categories[prediction[0]]}")
```

**Key Features to Include:**
- Subject line keywords
- Body text patterns
- Sender domain (facebook.com → Social)
- Presence of unsubscribe links (Promotions)
- Keywords: "sale", "offer", "liked", "commented"

**Continuous Improvement:**
- Collect user corrections (moved emails)
- Retrain periodically with new labels
- A/B test against rule-based system

---

## Question 3
- [ ] Done

**Propose a strategy for using Naive Bayes in a real-time bidding system for online advertising.**

**Answer:**

In Real-Time Bidding (RTB), the ad system must decide within milliseconds whether to bid on an ad impression and how much to bid. Naive Bayes predicts click probability P(click | user, context) extremely fast, making it ideal for this latency-critical application.

**Scenario: Ad Click Prediction**

```
User visits webpage → Ad Request → NB Predicts P(click) → Bid = P(click) × Value
                                          ↓
              Features: [User demographics, Page context, Ad features, Time]
```

**System Architecture:**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Ad Request │ →  │ Feature     │ →  │ NB Model    │
│  (10-100ms) │    │ Extraction  │    │ Inference   │
└─────────────┘    └─────────────┘    └─────────────┘
                                             ↓
                                      P(click) × CPC
                                             ↓
                                      ┌─────────────┐
                                      │ Bid/No-Bid  │
                                      │ Decision    │
                                      └─────────────┘
```

**Implementation Strategy:**

```python
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import time

class RTBClickPredictor:
    """
    Real-time bidding click prediction using Naive Bayes.
    
    Requirements:
    - Prediction latency < 5ms
    - Handle 100K+ requests/second
    - Update model periodically
    """
    
    def __init__(self):
        self.model = MultinomialNB(alpha=0.01)
        self.feature_map = {}  # Feature name to index
        self.n_features = 0
    
    def prepare_features(self, request):
        """
        Extract features from bid request.
        Must be fast - avoid complex processing.
        """
        features = np.zeros(self.n_features)
        
        # User features (pre-computed, stored in cookie/user_id lookup)
        user_segment = request.get('user_segment', 'unknown')
        if f'user_{user_segment}' in self.feature_map:
            features[self.feature_map[f'user_{user_segment}']] = 1
        
        # Context features
        device = request.get('device', 'desktop')
        features[self.feature_map.get(f'device_{device}', 0)] = 1
        
        hour = request.get('hour', 12)
        features[self.feature_map.get(f'hour_{hour}', 0)] = 1
        
        # Page category
        page_cat = request.get('page_category', 'other')
        features[self.feature_map.get(f'page_{page_cat}', 0)] = 1
        
        return features.reshape(1, -1)
    
    def predict_click_probability(self, request):
        """
        Predict P(click) for given ad request.
        Must complete in < 5ms.
        """
        start = time.time()
        
        features = self.prepare_features(request)
        prob_click = self.model.predict_proba(features)[0, 1]
        
        latency = (time.time() - start) * 1000
        if latency > 5:
            print(f"Warning: Latency exceeded {latency:.2f}ms")
        
        return prob_click
    
    def calculate_bid(self, request, cpc_value=1.0):
        """
        Calculate bid amount.
        
        Bid = P(click) × Expected value per click
        """
        p_click = self.predict_click_probability(request)
        
        # Apply bid floor and ceiling
        bid = p_click * cpc_value
        bid = max(0.01, min(bid, cpc_value))  # Floor: $0.01, Ceiling: CPC
        
        return bid, p_click
    
    def batch_train(self, impressions, clicks):
        """
        Periodic model update (offline).
        Train on collected impression data.
        """
        self.model.fit(impressions, clicks)


# Example bid request
request = {
    'user_segment': 'tech_enthusiast',
    'device': 'mobile',
    'hour': 14,
    'page_category': 'news',
    'ad_id': 'ad_12345'
}

# Latency test
predictor = RTBClickPredictor()
# ... (after training)
# bid, prob = predictor.calculate_bid(request, cpc_value=0.50)
```

**Why NB is Ideal for RTB:**

| Requirement | NB Advantage |
|-------------|--------------|
| < 5ms latency | O(d) prediction, no matrix ops |
| High throughput | Simple multiplication, vectorized |
| Sparse features | Works natively with sparse matrices |
| Online learning | partial_fit for incremental updates |
| Interpretability | Feature contributions visible |

**Production Considerations:**
1. Pre-compute feature vectors where possible
2. Use sparse matrices for memory efficiency
3. Periodic batch retraining (hourly/daily)
4. A/B test against more complex models (XGBoost, Deep Learning)
5. Calibrate probabilities for accurate bidding

---
