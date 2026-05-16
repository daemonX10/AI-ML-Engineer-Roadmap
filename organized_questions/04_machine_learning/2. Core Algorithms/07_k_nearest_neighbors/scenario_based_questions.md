# K Nearest Neighbors Interview Questions - Scenario_Based Questions

## Question 1
- [ ] Done

**How would you apply the K-NN algorithm in a recommendation system?**

### Answer

**Approach: Memory-Based Collaborative Filtering**

K-NN is the foundation of collaborative filtering recommendation systems. Two approaches: User-based (find similar users, recommend what they liked) and Item-based (find similar items to what user liked).

**User-Based Collaborative Filtering:**

**Logic:**
1. Represent each user as vector of item ratings
2. Find K most similar users to target user
3. Recommend items those users rated highly

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# User-Item rating matrix (rows=users, cols=items)
# 0 means not rated
ratings = np.array([
    [5, 3, 0, 1, 4],  # User 0
    [4, 0, 0, 1, 5],  # User 1
    [1, 1, 0, 5, 2],  # User 2
    [0, 0, 5, 4, 0],  # User 3 (target - wants recommendation)
])

def recommend_for_user(user_id, ratings, k=2):
    # Step 1: Calculate user similarity (cosine)
    user_sim = cosine_similarity(ratings)
    
    # Step 2: Find K most similar users (exclude self)
    similar_users = np.argsort(user_sim[user_id])[-k-1:-1][::-1]
    
    # Step 3: Find items user hasn't rated
    unrated_items = np.where(ratings[user_id] == 0)[0]
    
    # Step 4: Predict rating for unrated items
    predictions = {}
    for item in unrated_items:
        # Weighted average of similar users' ratings
        weights = user_sim[user_id, similar_users]
        item_ratings = ratings[similar_users, item]
        
        # Only consider users who rated this item
        mask = item_ratings > 0
        if mask.sum() > 0:
            pred = np.dot(weights[mask], item_ratings[mask]) / weights[mask].sum()
            predictions[item] = pred
    
    # Return top recommendations
    return sorted(predictions.items(), key=lambda x: x[1], reverse=True)

recommendations = recommend_for_user(3, ratings, k=2)
print(f"Recommended items: {recommendations}")
```

**Item-Based Collaborative Filtering:**
- Find items similar to what user already liked
- More stable (item similarity changes less than user similarity)
- Used by Amazon

**Advantages of K-NN for Recommendations:**
- **Interpretable**: "Recommended because users like you also liked this"
- **No model training**: Easy to add new users/items
- **Simple**: Easy to understand and implement

**Challenges:**
- **Sparsity**: Many users rate few items
- **Cold start**: New users have no ratings
- **Scalability**: O(N) for finding neighbors

**Modern Alternative:**
For production, use Matrix Factorization or Deep Learning, but K-NN remains useful for explainability.

---
