# Recommendation Systems Interview Questions - Coding Questions

## Question 1
- [ ] Done

**How would you implement a recommendation system using the k-NN algorithm?**

**Answer:**

**Concept:** Find k most similar users, recommend items they liked.

```python
import numpy as np
from collections import defaultdict

# Sample user-item ratings (0 = not rated)
ratings = np.array([
    [5, 3, 0, 1],  # User 0
    [4, 0, 0, 1],  # User 1
    [1, 1, 0, 5],  # User 2
    [0, 0, 5, 4],  # User 3
    [0, 1, 4, 4],  # User 4
])

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    # Only consider items both users rated
    mask = (a > 0) & (b > 0)
    if mask.sum() == 0:
        return 0
    a_masked, b_masked = a[mask], b[mask]
    dot = np.dot(a_masked, b_masked)
    norm = np.linalg.norm(a_masked) * np.linalg.norm(b_masked)
    return dot / norm if norm > 0 else 0

def knn_recommend(user_id, ratings, k=2, n_recommendations=2):
    """Recommend items using k-NN collaborative filtering."""
    n_users, n_items = ratings.shape
    user_ratings = ratings[user_id]
    
    # Calculate similarity to all other users
    similarities = []
    for other_id in range(n_users):
        if other_id != user_id:
            sim = cosine_similarity(user_ratings, ratings[other_id])
            similarities.append((other_id, sim))
    
    # Get k nearest neighbors
    neighbors = sorted(similarities, key=lambda x: -x[1])[:k]
    
    # Predict scores for unrated items
    scores = defaultdict(float)
    for item_id in range(n_items):
        if user_ratings[item_id] == 0:  # Not rated
            for neighbor_id, sim in neighbors:
                if ratings[neighbor_id][item_id] > 0:
                    scores[item_id] += sim * ratings[neighbor_id][item_id]
    
    # Return top recommendations
    recommendations = sorted(scores.items(), key=lambda x: -x[1])
    return recommendations[:n_recommendations]

# Example
recs = knn_recommend(user_id=0, ratings=ratings, k=2)
print(f"Recommendations for User 0: {recs}")
# Output: [(2, score), ...] - Item 2 recommended
```

**Key Steps:**
1. Compute similarity between target user and all others
2. Select k most similar users
3. Aggregate their ratings for unrated items
4. Return highest scored items

---

## Question 2
- [ ] Done

**Implement a simple content-based recommendation algorithm in Python.**

**Answer:**

**Concept:** Recommend items similar to what user liked, based on item features.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample items with descriptions
items = {
    0: "action adventure superhero movie",
    1: "romantic comedy love story",
    2: "action thriller spy mission",
    3: "romantic drama love tragedy",
    4: "superhero action comics movie"
}

# User's liked items (item_ids)
user_liked = [0, 2]  # User likes action movies

def content_based_recommend(items, user_liked, n_recommendations=2):
    """Recommend items based on content similarity."""
    item_ids = list(items.keys())
    descriptions = list(items.values())
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    
    # Calculate similarity between all items
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Build user profile: average of liked item vectors
    liked_indices = [item_ids.index(i) for i in user_liked]
    user_profile = tfidf_matrix[liked_indices].mean(axis=0)
    user_profile = np.asarray(user_profile).flatten()
    
    # Score all items against user profile
    scores = []
    for idx, item_id in enumerate(item_ids):
        if item_id not in user_liked:
            item_vector = tfidf_matrix[idx].toarray().flatten()
            score = np.dot(user_profile, item_vector)
            scores.append((item_id, score))
    
    # Return top recommendations
    scores.sort(key=lambda x: -x[1])
    return scores[:n_recommendations]

# Example
recs = content_based_recommend(items, user_liked)
print(f"Recommendations: {recs}")
# Output: [(4, score), ...] - Item 4 (superhero action) recommended
```

**Key Steps:**
1. Convert item descriptions to TF-IDF vectors
2. Create user profile from liked items
3. Score unrated items by similarity to profile
4. Return highest scoring items

---

## Question 3
- [ ] Done

**Write a collaborative filtering recommendation engine using Python's Surprise library.**

**Answer:**

**Concept:** Use Surprise library for easy CF implementation with SVD.

```python
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

# Sample ratings data
ratings_data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'item_id': [1, 2, 3, 1, 4, 2, 3, 4, 1, 3],
    'rating':  [5, 3, 4, 4, 5, 2, 4, 3, 3, 5]
}
df = pd.DataFrame(ratings_data)

# Load data into Surprise format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Split data
trainset, testset = train_test_split(data, test_size=0.2)

# Train SVD model
model = SVD(n_factors=20, n_epochs=20, random_state=42)
model.fit(trainset)

# Evaluate
predictions = model.test(testset)
print(f"RMSE: {accuracy.rmse(predictions):.4f}")

# Predict rating for user 1 on item 4
pred = model.predict(uid=1, iid=4)
print(f"Predicted rating for User 1, Item 4: {pred.est:.2f}")

# Get top-N recommendations for a user
def get_recommendations(model, user_id, all_items, rated_items, n=3):
    """Get top N recommendations for a user."""
    predictions = []
    for item_id in all_items:
        if item_id not in rated_items:
            pred = model.predict(user_id, item_id)
            predictions.append((item_id, pred.est))
    predictions.sort(key=lambda x: -x[1])
    return predictions[:n]

# Example
all_items = df['item_id'].unique()
rated_by_user1 = df[df['user_id'] == 1]['item_id'].tolist()
recs = get_recommendations(model, 1, all_items, rated_by_user1)
print(f"Recommendations for User 1: {recs}")
```

**Installation:** `pip install scikit-surprise`

---

## Question 4
- [ ] Done

**Demonstrate matrix factorization using the NMF (Non-negative Matrix Factorization) algorithm on a sample dataset.**

**Answer:**

**Concept:** Decompose ratings matrix into non-negative user and item factors.

```python
import numpy as np
from sklearn.decomposition import NMF

# Sample user-item ratings matrix (0 = missing)
ratings = np.array([
    [5, 3, 0, 1, 0],
    [4, 0, 0, 1, 2],
    [1, 1, 0, 5, 0],
    [0, 0, 5, 4, 3],
    [0, 1, 4, 0, 4],
])

def nmf_recommend(ratings, n_factors=2, n_recommendations=2):
    """Matrix factorization using NMF for recommendations."""
    # Replace 0s with small value for NMF (handles sparsity)
    ratings_filled = ratings.copy().astype(float)
    ratings_filled[ratings_filled == 0] = np.nan
    # Fill NaN with row mean for initial factorization
    row_means = np.nanmean(ratings_filled, axis=1, keepdims=True)
    ratings_filled = np.where(np.isnan(ratings_filled), row_means, ratings_filled)
    
    # Apply NMF
    nmf = NMF(n_components=n_factors, init='random', random_state=42)
    user_factors = nmf.fit_transform(ratings_filled)  # W: users x factors
    item_factors = nmf.components_                     # H: factors x items
    
    # Reconstruct ratings
    predicted_ratings = np.dot(user_factors, item_factors)
    
    # Get recommendations for each user
    recommendations = {}
    for user_id in range(ratings.shape[0]):
        scores = []
        for item_id in range(ratings.shape[1]):
            if ratings[user_id, item_id] == 0:  # Not rated
                scores.append((item_id, predicted_ratings[user_id, item_id]))
        scores.sort(key=lambda x: -x[1])
        recommendations[user_id] = scores[:n_recommendations]
    
    return recommendations, predicted_ratings

# Example
recs, predicted = nmf_recommend(ratings)
print("Recommendations per user:")
for user_id, items in recs.items():
    print(f"  User {user_id}: {items}")

print(f"\nOriginal ratings:\n{ratings}")
print(f"\nPredicted ratings:\n{np.round(predicted, 1)}")
```

**Key Points:**
- NMF ensures non-negative factors (interpretable)
- Factors can represent latent features (genres, quality)
- Handle missing values before factorization

---

## Question 5
- [ ] Done

**Code a recommender that uses cosine similarity to recommend similar items.**

**Answer:**

**Concept:** Find items most similar to a given item using cosine similarity.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Item-feature matrix (rows=items, cols=features)
# Features could be: genre_action, genre_comedy, rating, popularity, etc.
item_features = np.array([
    [1, 0, 0, 5, 0.8],  # Item 0: Action
    [0, 1, 0, 4, 0.6],  # Item 1: Comedy
    [1, 0, 1, 5, 0.9],  # Item 2: Action-Thriller
    [0, 1, 0, 3, 0.4],  # Item 3: Comedy
    [1, 0, 1, 4, 0.7],  # Item 4: Action-Thriller
])

item_names = ['Die Hard', 'Superbad', 'Mission Impossible', 'Hangover', 'James Bond']

def find_similar_items(item_id, item_features, item_names, n=2):
    """Find n most similar items using cosine similarity."""
    # Compute cosine similarity between all items
    similarity_matrix = cosine_similarity(item_features)
    
    # Get similarities for target item
    similarities = similarity_matrix[item_id]
    
    # Get indices sorted by similarity (excluding self)
    similar_indices = np.argsort(similarities)[::-1]
    
    # Filter out the item itself and return top n
    results = []
    for idx in similar_indices:
        if idx != item_id:
            results.append((item_names[idx], similarities[idx]))
        if len(results) >= n:
            break
    
    return results

# Example: Find items similar to "Die Hard" (item 0)
target_item = 0
similar = find_similar_items(target_item, item_features, item_names, n=2)

print(f"Items similar to '{item_names[target_item]}':")
for name, score in similar:
    print(f"  {name}: {score:.3f}")

# Output:
# Items similar to 'Die Hard':
#   Mission Impossible: 0.97
#   James Bond: 0.96
```

**Use Case:** "Customers who liked X also liked..."

---

## Question 6
- [ ] Done

**Build a user-based collaborative filtering system in Python from scratch.**

**Answer:**

**Concept:** Predict ratings based on similar users' preferences.

```python
import numpy as np

# User-item ratings matrix (0 = not rated)
ratings = np.array([
    [5, 3, 0, 1, 4],  # User 0
    [4, 0, 4, 1, 0],  # User 1
    [1, 1, 0, 5, 4],  # User 2
    [5, 4, 4, 0, 3],  # User 3
    [0, 2, 5, 4, 4],  # User 4
])

def pearson_similarity(user1, user2):
    """Calculate Pearson correlation between two users."""
    # Find items both users rated
    mask = (user1 > 0) & (user2 > 0)
    if mask.sum() < 2:  # Need at least 2 common items
        return 0
    
    u1, u2 = user1[mask], user2[mask]
    mean1, mean2 = u1.mean(), u2.mean()
    
    numerator = np.sum((u1 - mean1) * (u2 - mean2))
    denominator = np.sqrt(np.sum((u1 - mean1)**2) * np.sum((u2 - mean2)**2))
    
    return numerator / denominator if denominator > 0 else 0

def predict_rating(user_id, item_id, ratings, k=3):
    """Predict rating for user on item using k similar users."""
    if ratings[user_id, item_id] > 0:
        return ratings[user_id, item_id]  # Already rated
    
    n_users = ratings.shape[0]
    user_ratings = ratings[user_id]
    
    # Calculate similarity to all other users
    similarities = []
    for other_id in range(n_users):
        if other_id != user_id and ratings[other_id, item_id] > 0:
            sim = pearson_similarity(user_ratings, ratings[other_id])
            similarities.append((other_id, sim))
    
    if not similarities:
        return 0  # No similar users rated this item
    
    # Get top k similar users
    top_k = sorted(similarities, key=lambda x: -abs(x[1]))[:k]
    
    # Weighted average prediction
    numerator = sum(sim * ratings[other, item_id] for other, sim in top_k)
    denominator = sum(abs(sim) for _, sim in top_k)
    
    return numerator / denominator if denominator > 0 else 0

def recommend(user_id, ratings, n=2):
    """Get top n recommendations for a user."""
    predictions = []
    for item_id in range(ratings.shape[1]):
        if ratings[user_id, item_id] == 0:
            pred = predict_rating(user_id, item_id, ratings)
            predictions.append((item_id, pred))
    
    predictions.sort(key=lambda x: -x[1])
    return predictions[:n]

# Example
recs = recommend(user_id=0, ratings=ratings)
print(f"Recommendations for User 0: {recs}")
```

---

## Question 7
- [ ] Done

**Use TensorFlow/Keras to develop a deep learning-based recommendation model.**

**Answer:**

**Concept:** Neural Collaborative Filtering with embeddings.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate

# Sample data
n_users, n_items = 100, 50
embedding_dim = 16

# Generate sample training data
np.random.seed(42)
user_ids = np.random.randint(0, n_users, 1000)
item_ids = np.random.randint(0, n_items, 1000)
ratings = np.random.randint(1, 6, 1000).astype(float)

# Build NCF Model
def build_ncf_model(n_users, n_items, embedding_dim=16):
    """Build Neural Collaborative Filtering model."""
    # Input layers
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')
    
    # Embedding layers
    user_embed = Embedding(n_users, embedding_dim, name='user_embed')(user_input)
    item_embed = Embedding(n_items, embedding_dim, name='item_embed')(item_input)
    
    # Flatten embeddings
    user_flat = Flatten()(user_embed)
    item_flat = Flatten()(item_embed)
    
    # Concatenate and pass through MLP
    concat = Concatenate()([user_flat, item_flat])
    dense1 = Dense(64, activation='relu')(concat)
    dense2 = Dense(32, activation='relu')(dense1)
    output = Dense(1)(dense2)  # Rating prediction
    
    model = Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train model
model = build_ncf_model(n_users, n_items, embedding_dim)
model.fit(
    [user_ids, item_ids], ratings,
    epochs=10, batch_size=32, verbose=1
)

# Predict
test_user, test_item = np.array([5]), np.array([10])
predicted_rating = model.predict([test_user, test_item])
print(f"Predicted rating for User 5, Item 10: {predicted_rating[0][0]:.2f}")

# Get recommendations for a user
def get_recommendations(model, user_id, n_items, n_recs=5):
    user_ids = np.full(n_items, user_id)
    item_ids = np.arange(n_items)
    predictions = model.predict([user_ids, item_ids], verbose=0)
    top_items = np.argsort(predictions.flatten())[::-1][:n_recs]
    return top_items

top_items = get_recommendations(model, user_id=5, n_items=n_items)
print(f"Top items for User 5: {top_items}")
```

---

## Question 8
- [ ] Done

**Create a Python script that recommends items to users based on item-item similarity.**

**Answer:**

**Concept:** Find similar items to what user liked, recommend those.

```python
import numpy as np

# User-item ratings matrix
ratings = np.array([
    [5, 3, 0, 1, 4],
    [4, 0, 4, 1, 0],
    [1, 1, 0, 5, 4],
    [5, 4, 4, 0, 3],
    [0, 2, 5, 4, 4],
])

def compute_item_similarity(ratings):
    """Compute item-item similarity matrix using cosine similarity."""
    n_items = ratings.shape[1]
    similarity = np.zeros((n_items, n_items))
    
    for i in range(n_items):
        for j in range(i, n_items):
            # Get users who rated both items
            mask = (ratings[:, i] > 0) & (ratings[:, j] > 0)
            if mask.sum() > 0:
                vec_i = ratings[mask, i]
                vec_j = ratings[mask, j]
                # Cosine similarity
                dot = np.dot(vec_i, vec_j)
                norm = np.linalg.norm(vec_i) * np.linalg.norm(vec_j)
                sim = dot / norm if norm > 0 else 0
            else:
                sim = 0
            similarity[i, j] = similarity[j, i] = sim
    
    return similarity

def item_based_recommend(user_id, ratings, item_sim, n_recs=2):
    """Recommend items based on item-item similarity."""
    user_ratings = ratings[user_id]
    n_items = ratings.shape[1]
    
    scores = []
    for item_id in range(n_items):
        if user_ratings[item_id] == 0:  # Not rated
            # Weighted sum of ratings for similar items
            numerator = 0
            denominator = 0
            for rated_item in range(n_items):
                if user_ratings[rated_item] > 0:
                    sim = item_sim[item_id, rated_item]
                    numerator += sim * user_ratings[rated_item]
                    denominator += abs(sim)
            
            score = numerator / denominator if denominator > 0 else 0
            scores.append((item_id, score))
    
    scores.sort(key=lambda x: -x[1])
    return scores[:n_recs]

# Compute similarity matrix
item_sim = compute_item_similarity(ratings)
print(f"Item similarity matrix:\n{np.round(item_sim, 2)}")

# Get recommendations
recs = item_based_recommend(user_id=0, ratings=ratings, item_sim=item_sim)
print(f"\nRecommendations for User 0: {recs}")
```

**Advantage:** Item similarity is more stable than user similarity (fewer items than users).

---

## Question 9
- [ ] Done

**Implement a recommendation engine that leverages user ratings and item metadata for suggestions.**

**Answer:**

**Concept:** Hybrid approach combining collaborative filtering and content-based.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# User-item ratings
ratings = np.array([
    [5, 0, 4, 0, 0],  # User 0
    [0, 4, 0, 5, 0],  # User 1
    [4, 0, 5, 0, 3],  # User 2
])

# Item metadata (descriptions)
item_metadata = [
    "action adventure thriller",
    "romantic comedy love",
    "action thriller spy",
    "romantic drama love",
    "action adventure hero"
]

def content_similarity(item_metadata):
    """Compute item-item similarity from content."""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(item_metadata)
    return cosine_similarity(tfidf)

def collaborative_score(user_id, item_id, ratings):
    """Simple CF score based on similar users."""
    user_ratings = ratings[user_id]
    scores = []
    for other_id in range(ratings.shape[0]):
        if other_id != user_id and ratings[other_id, item_id] > 0:
            # Simple similarity: count of common rated items
            common = (user_ratings > 0) & (ratings[other_id] > 0)
            if common.sum() > 0:
                scores.append(ratings[other_id, item_id])
    return np.mean(scores) if scores else 0

def hybrid_recommend(user_id, ratings, item_metadata, alpha=0.5, n_recs=2):
    """Hybrid recommendation: alpha*CF + (1-alpha)*content."""
    content_sim = content_similarity(item_metadata)
    user_ratings = ratings[user_id]
    n_items = ratings.shape[1]
    
    # Get user's liked items for content-based
    liked_items = np.where(user_ratings >= 4)[0]
    
    recommendations = []
    for item_id in range(n_items):
        if user_ratings[item_id] == 0:  # Not rated
            # Collaborative score
            cf_score = collaborative_score(user_id, item_id, ratings)
            
            # Content-based score (similarity to liked items)
            cb_score = 0
            if len(liked_items) > 0:
                cb_score = np.mean([content_sim[item_id, liked] for liked in liked_items])
            
            # Hybrid score
            hybrid_score = alpha * cf_score + (1 - alpha) * cb_score * 5  # Scale CB
            recommendations.append((item_id, hybrid_score))
    
    recommendations.sort(key=lambda x: -x[1])
    return recommendations[:n_recs]

# Example
recs = hybrid_recommend(user_id=0, ratings=ratings, item_metadata=item_metadata)
print(f"Hybrid recommendations for User 0: {recs}")
# User 0 likes action (items 0, 2), so item 4 (action adventure) should rank high
```

---

## Question 10
- [ ] Done

**Write an algorithm to suggest items using the Pearson Correlation Coefficient in a user-item ratings matrix.**

**Answer:**

**Concept:** Use Pearson correlation to find similar users, then predict ratings.

```python
import numpy as np

# User-item ratings matrix
ratings = np.array([
    [5, 3, 0, 1, 4],
    [4, 0, 4, 1, 0],
    [1, 1, 0, 5, 4],
    [5, 4, 4, 0, 3],
    [0, 2, 5, 4, 4],
])

def pearson_correlation(user1, user2):
    """Calculate Pearson correlation between two users."""
    # Find items both users rated
    mask = (user1 > 0) & (user2 > 0)
    if mask.sum() < 2:
        return 0
    
    x, y = user1[mask], user2[mask]
    n = len(x)
    
    # Pearson formula
    sum_x, sum_y = x.sum(), y.sum()
    sum_xy = (x * y).sum()
    sum_x2, sum_y2 = (x**2).sum(), (y**2).sum()
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
    
    return numerator / denominator if denominator > 0 else 0

def predict_with_pearson(user_id, item_id, ratings, k=3):
    """Predict rating using Pearson-weighted average."""
    if ratings[user_id, item_id] > 0:
        return ratings[user_id, item_id]
    
    user_mean = ratings[user_id][ratings[user_id] > 0].mean()
    
    # Find similar users who rated this item
    similarities = []
    for other_id in range(ratings.shape[0]):
        if other_id != user_id and ratings[other_id, item_id] > 0:
            sim = pearson_correlation(ratings[user_id], ratings[other_id])
            if sim > 0:  # Only consider positively correlated users
                similarities.append((other_id, sim))
    
    if not similarities:
        return user_mean  # Default to user's average
    
    # Top k similar users
    top_k = sorted(similarities, key=lambda x: -x[1])[:k]
    
    # Weighted prediction
    numerator = 0
    denominator = 0
    for other_id, sim in top_k:
        other_mean = ratings[other_id][ratings[other_id] > 0].mean()
        numerator += sim * (ratings[other_id, item_id] - other_mean)
        denominator += abs(sim)
    
    prediction = user_mean + (numerator / denominator if denominator > 0 else 0)
    return np.clip(prediction, 1, 5)  # Clip to valid range

def recommend_with_pearson(user_id, ratings, n_recs=2):
    """Get top recommendations using Pearson-based CF."""
    predictions = []
    for item_id in range(ratings.shape[1]):
        if ratings[user_id, item_id] == 0:
            pred = predict_with_pearson(user_id, item_id, ratings)
            predictions.append((item_id, pred))
    
    predictions.sort(key=lambda x: -x[1])
    return predictions[:n_recs]

# Example
recs = recommend_with_pearson(user_id=0, ratings=ratings)
print(f"Recommendations for User 0: {recs}")

# Show Pearson correlations for User 0
print("\nUser 0's Pearson correlations:")
for other_id in range(1, ratings.shape[0]):
    corr = pearson_correlation(ratings[0], ratings[other_id])
    print(f"  with User {other_id}: {corr:.3f}")
```

**Key Points:**
- Pearson handles rating scale differences (centers data)
- More accurate than cosine for ratings
- Negative correlation can indicate opposite preferences

---

