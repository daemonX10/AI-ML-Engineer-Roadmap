# Recommendation Systems Interview Questions - Scenario_Based Questions

## Question 1
- [ ] Done

**Discuss the importance of serendipity, novelty, and diversity in recommendation systems.**

**Answer:**

**Beyond Accuracy:**
Optimizing only for accuracy leads to filter bubbles and repetitive recommendations. These metrics ensure better user experience.

---

**Definitions:**

| Metric | Definition | Example |
|--------|------------|--------|
| **Novelty** | Items user hasn't seen before | Recommending new releases |
| **Diversity** | Variety within recommendation list | Different genres, not all action |
| **Serendipity** | Unexpected but relevant items | Jazz to a rock fan who'd love it |

---

**Why They Matter:**

| Benefit | Description |
|---------|-------------|
| Avoid filter bubbles | Don't trap users in narrow interests |
| Discovery | Help users find new favorites |
| Engagement | Surprise keeps users interested |
| Long-term value | Short-term clicks ≠ long-term satisfaction |

---

**How to Measure:**

```python
# Novelty: Average inverse popularity
novelty = np.mean([1/popularity[item] for item in recommended])

# Diversity: Average pairwise distance
diversity = np.mean([distance(i, j) for i, j in combinations(recommended, 2)])

# Serendipity: Unexpected + Relevant
serendipity = unexpected_items ∩ liked_items
```

---

**Strategies to Improve:**

| Strategy | How |
|----------|-----|
| Diversification | Re-rank top-N to maximize diversity |
| Exploration | Use bandits to try new items |
| Long-tail boosting | Include less popular items |
| MMR | Maximal Marginal Relevance |

**Trade-off:** Balance with accuracy—too much novelty = irrelevant items

---

## Question 2
- [ ] Done

**How would you handle scalability and sparsity issues in recommendation systems?**

**Answer:**

**The Challenges:**

| Issue | Description |
|-------|-------------|
| **Sparsity** | <1% of user-item matrix filled |
| **Scalability** | Millions of users × millions of items |

---

**Handling Sparsity:**

**1. Matrix Factorization**
- Reduce to low-rank representation
- Only learn from observed entries

**2. Implicit Feedback**
- Use views, clicks, time spent (more data than ratings)

**3. Side Information**
- Content features fill gaps
- User demographics

**4. Graph-Based**
- Propagate signals through connections

---

**Handling Scalability:**

**1. Approximate Nearest Neighbors (ANN)**
```python
# Instead of exact similarity search
from annoy import AnnoyIndex
index = AnnoyIndex(embedding_dim, 'angular')
for i, vec in enumerate(item_embeddings):
    index.add_item(i, vec)
index.build(n_trees=10)
# Fast approximate search
similar = index.get_nns_by_vector(query, k=10)
```

**2. Distributed Computing**
- Spark MLlib for large-scale ALS
- Parameter servers for distributed training

**3. Sampling Strategies**
- Negative sampling (don't use all negatives)
- Mini-batch training

**4. Two-Stage Architecture**
```
Candidate Generation → Ranking
(Fast, approximate)    (Accurate, small set)
```

**5. Model Compression**
- Quantization, pruning
- Knowledge distillation

**Best Practice:** Separate candidate retrieval (fast) from ranking (accurate)

---

## Question 3
- [ ] Done

**Discuss the application of the Gradient Boosting Machines (GBM) in recommendation engines.**

**Answer:**

**Where GBM Fits in Recommendations:**

GBM is commonly used in the **ranking stage** of two-stage recommender systems.

```
Candidate Generation → GBM Ranker → Top-K Results
```

---

**Use Cases:**

| Stage | GBM Application |
|-------|----------------|
| **Ranking** | Score and rank candidates |
| **CTR prediction** | Predict click probability |
| **Feature engineering** | Learn feature interactions |

---

**Why GBM for Ranking?**

| Advantage | Description |
|-----------|-------------|
| Handles mixed features | Numerical + categorical |
| Feature interactions | Learns automatically |
| Robust | Handles missing values |
| Fast inference | Tree traversal is efficient |
| Interpretable | Feature importance |

---

**Typical Features for GBM Ranker:**

```python
features = [
    # User features
    'user_age', 'user_activity_level', 'user_avg_rating',
    # Item features  
    'item_popularity', 'item_age_days', 'item_avg_rating',
    # Interaction features
    'user_item_category_affinity', 'collaborative_score',
    # Context
    'hour_of_day', 'device_type'
]
```

---

**Training Setup:**

| Aspect | Approach |
|--------|----------|
| **Objective** | Binary (click/no-click), regression (rating) |
| **Labels** | Implicit feedback or explicit ratings |
| **Negatives** | Sample from non-interacted items |
| **Ranking loss** | LambdaMART for listwise learning |

**Popular Libraries:** XGBoost, LightGBM, CatBoost

**Limitation:** Doesn't learn embeddings—needs engineered features or embeddings from other models

---

## Question 4
- [ ] Done

**How would you approach designing a music recommendation engine?**

**Answer:**

**Music-Specific Challenges:**
- Short content (3 min songs vs 2 hr movies)
- Repeat consumption (same song many times)
- Context-dependent (workout, sleep, mood)
- Audio features matter

---

**Design Approach:**

**1. Data Signals**

| Signal | Weight |
|--------|--------|
| Full listens | High positive |
| Skips (<30s) | Negative |
| Saves/playlists | Strong positive |
| Repeats | Very positive |
| Time of day | Context |

**2. Feature Types**

| Type | Features |
|------|----------|
| **Audio** | Tempo, key, energy, acousticness |
| **Metadata** | Artist, genre, release year |
| **Collaborative** | Similar user preferences |
| **Sequential** | Listening history patterns |

**3. System Architecture**
```
┌─────────────────────────────────────────┐
│ Candidate Generation                    │
│ - Similar artists/genres                │
│ - User history-based                    │
│ - Audio similarity (embeddings)         │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ Context-Aware Ranking                   │
│ - Time of day, activity                 │
│ - Recent listening mood                 │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ Playlist Generation                     │
│ - Smooth transitions                    │
│ - Diversity within coherence            │
└─────────────────────────────────────────┘
```

**4. Context-Aware Recommendations**
```python
def recommend(user, context):
    base_recs = get_candidates(user)
    if context.activity == 'workout':
        return filter_by_energy(base_recs, min_energy=0.7)
    elif context.time.hour > 22:
        return filter_by_energy(base_recs, max_energy=0.4)
    return base_recs
```

**Key:** Sequence modeling (RNN/Transformer) for session continuity

---

## Question 5
- [ ] Done

**Discuss a personalized approach for recommendations in a video streaming service.**

**Answer:**

**Video Streaming Specifics:**
- Long-form content (consume once)
- Strong sequential dependencies (series)
- Household vs individual accounts
- Multiple content types (movies, series, docs)

---

**Personalization Approach:**

**1. User Profiling**

| Profile Aspect | Data Used |
|----------------|----------|
| Genre preferences | Watch history distribution |
| Quality preferences | Ratings, completion rates |
| Viewing patterns | Time, device, binge behavior |
| Household members | Profile switching patterns |

**2. Signals to Use**

| Signal | Interpretation |
|--------|---------------|
| Completion >90% | Strong positive |
| Dropped <10% | Negative |
| Rewatched | Very positive |
| Added to list | Interest |
| Searched | Intent |

**3. Multi-Model Architecture**
```
┌─────────────────────────────────────────┐
│ Homepage: Personalized rows             │
│ - Continue Watching                     │
│ - Because you watched X                 │
│ - Trending in your genres               │
│ - New releases you might like           │
└─────────────────────────────────────────┘
```

**4. Row Generation**
```python
def generate_rows(user):
    rows = []
    rows.append(('Continue Watching', get_in_progress(user)))
    for watched_item in user.recently_watched[:3]:
        similar = get_similar(watched_item)
        rows.append((f'Because you watched {watched_item.title}', similar))
    rows.append(('Trending', personalized_trending(user)))
    return rows
```

**5. Contextual Adaptations**
- Weekend: Longer content
- Kids profile: Age-appropriate only
- Mobile: Shorter content

**Evaluation:**
- Watch time, completion rate, returning users
- Not just clicks (clickbait problem)

---

## Question 6
- [ ] Done

**How would you develop a recommendation system for a social network to suggest new connections?**

**Answer:**

**Goal:** Suggest people the user is likely to know or want to connect with.

---

**Graph-Based Approach:**

**1. Data Signals**

| Signal | Weight |
|--------|--------|
| Mutual friends | High |
| Same school/company | High |
| Interacted with posts | Medium |
| Similar interests | Medium |
| Geographic proximity | Low-Medium |

**2. Friend-of-Friend (FoF)**
```python
def fof_recommendations(user):
    friends = user.friends
    candidates = {}
    for friend in friends:
        for fof in friend.friends:
            if fof != user and fof not in friends:
                candidates[fof] = candidates.get(fof, 0) + 1
    # Score by mutual friend count
    return sorted(candidates, key=candidates.get, reverse=True)
```

**3. Graph Neural Network Approach**
- Learn node embeddings from graph structure
- Predict link probability between user pairs
- Node2Vec, GraphSAGE for embeddings

**4. Feature Engineering**

| Feature | Description |
|---------|-------------|
| Common neighbors | # mutual friends |
| Jaccard similarity | Mutual / Union of friends |
| Adamic-Adar | Weighted by friend popularity |
| Profile similarity | Interests, demographics |

**5. Scoring Model**
```python
features = [
    num_mutual_friends,
    jaccard_similarity,
    same_location,
    same_school,
    profile_embedding_similarity
]
connection_score = gbm_model.predict(features)
```

**Privacy Considerations:**
- Respect user privacy settings
- Don't reveal lurking behavior
- Allow opting out

**Metrics:** Connection acceptance rate, engagement after connection

---

## Question 7
- [ ] Done

**Discuss the state-of-the-art models used in recommendation systems, such as Neural Collaborative Filtering.**

**Answer:**

**Evolution of Recommendation Models:**

```
Matrix Factorization → Neural CF → Deep Learning → Transformers
```

---

**Neural Collaborative Filtering (NCF):**

**Architecture:**
```
User ID → Embedding → \
                        MLP Layers → Prediction
Item ID → Embedding → /
```

**Key Innovation:**
- Replace dot product with neural network
- Learn non-linear user-item interactions

```python
class NCF(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, hidden_dims):
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
    
    def forward(self, user, item):
        u = self.user_embed(user)
        i = self.item_embed(item)
        x = torch.cat([u, i], dim=1)
        return self.mlp(x)
```

---

**Other State-of-the-Art Models:**

| Model | Key Idea |
|-------|----------|
| **Wide & Deep** | Combines memorization (wide) + generalization (deep) |
| **DeepFM** | FM layer + deep network |
| **AutoInt** | Self-attention for feature interactions |
| **BERT4Rec** | Transformer for sequential recommendation |
| **SASRec** | Self-attention for sequences |
| **LightGCN** | Simplified GNN for collaborative filtering |
| **Two-Tower** | Separate user/item encoders for scalability |

---

**Transformer-Based (Current SOTA):**

| Model | Application |
|-------|-------------|
| **BERT4Rec** | Masked item prediction |
| **SASRec** | Sequential next-item |
| **BST** | Behavior sequence transformer |

**Trade-off:** More complex models need more data and compute; simpler models often competitive with good features

---

