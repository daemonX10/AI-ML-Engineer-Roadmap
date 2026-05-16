# Recommendation Systems Interview Questions - General Questions

## Question 1
- [ ] Done

**How do cold start problems impact recommendation systems and how can they be mitigated?**

**Answer:**

**Types of Cold Start:**

| Type | Description | Challenge |
|------|-------------|-----------|
| **New User** | No interaction history | Cannot personalize |
| **New Item** | No ratings/views yet | Cannot recommend |
| **New System** | No data at all | No patterns to learn |

---

**Mitigation Strategies:**

**For New Users:**

| Strategy | Description |
|----------|-------------|
| Onboarding survey | Ask preferences explicitly |
| Popular items | Recommend globally popular |
| Demographic-based | Use similar demographic group |
| Social | Import preferences from social networks |
| Exploration | Use bandits to learn quickly |

**For New Items:**

| Strategy | Description |
|----------|-------------|
| Content-based | Use item attributes |
| Boost exposure | Show to random subset |
| Editorial | Human curation initially |
| Similar items | Match to existing items by content |

**For New System:**

| Strategy | Description |
|----------|-------------|
| Import data | Transfer from similar domain |
| Synthetic data | Generate initial interactions |
| Rule-based | Start with business rules |

---

**Hybrid Approach (Best Practice):**
```python
def recommend(user):
    if user.has_history():
        return collaborative_filter(user)
    elif user.has_demographics():
        return demographic_based(user)
    else:
        return popular_items()
```

---

## Question 2
- [ ] Done

**How do matrix factorization techniques work in recommendation engines?**

**Answer:**

**Core Idea:**
Decompose the sparse user-item rating matrix into low-rank matrices representing latent factors for users and items.

$$R \approx P \times Q^T$$

Where:
- R = m×n rating matrix (m users, n items)
- P = m×k user factor matrix
- Q = n×k item factor matrix
- k = number of latent factors

---

**How It Works:**

**1. Learn Latent Factors:**
Each user/item represented by k-dimensional vector capturing hidden features (e.g., genre preference, quality).

**2. Predict Ratings:**
$$\hat{r}_{ui} = p_u \cdot q_i = \sum_{f=1}^{k} p_{uf} \cdot q_{if}$$

**3. Optimize:**
Minimize reconstruction error + regularization:
$$\min_{P,Q} \sum_{(u,i) \in \text{observed}} (r_{ui} - p_u \cdot q_i)^2 + \lambda(||P||^2 + ||Q||^2)$$

---

**Optimization Methods:**

| Method | Description |
|--------|-------------|
| **SGD** | Update factors for each observed rating |
| **ALS** | Alternate fixing P/Q, solve convex problem |
| **SVD** | Singular value decomposition (for dense) |

---

**Extensions:**

| Technique | Enhancement |
|-----------|-------------|
| Bias terms | Add user/item biases |
| SVD++ | Include implicit feedback |
| NMF | Non-negative constraints |
| Temporal | Time-varying factors |

**Advantages:** Handles sparsity, captures latent patterns, scalable

---

## Question 3
- [ ] Done

**What kind of data preprocessing is typically required when building a recommendation system?**

**Answer:**

**Data Preprocessing Steps:**

**1. Data Cleaning**

| Task | Action |
|------|--------|
| Duplicates | Remove duplicate interactions |
| Invalid entries | Filter invalid user/item IDs |
| Outliers | Remove extreme ratings (e.g., bots) |
| Missing values | Handle or impute strategically |

**2. User/Item Filtering**
```python
# Remove users with too few interactions
users_to_keep = df.groupby('user_id').size() >= min_interactions
# Remove items with too few interactions  
items_to_keep = df.groupby('item_id').size() >= min_interactions
```

**3. Interaction Normalization**
- Center ratings (subtract user mean)
- Scale to [0, 1] range
- Log-transform counts for implicit feedback

**4. Temporal Processing**
- Sort by timestamp
- Create train/test splits by time
- Add time-based features

**5. Encoding**
```python
# Map IDs to continuous integers
user_encoder = LabelEncoder().fit(df['user_id'])
item_encoder = LabelEncoder().fit(df['item_id'])
```

**6. Feature Engineering**
- Item features: TF-IDF on descriptions
- User features: Aggregate behavior stats
- Embeddings: Pre-trained or learned

---

**For Content-Based:**
- Text: Tokenization, TF-IDF, embeddings
- Images: CNN features
- Categories: One-hot or embeddings

**Best Practice:** Keep raw data, preprocessing pipeline reproducible.

---

## Question 4
- [ ] Done

**What strategies can be used to evaluate the performance of a recommendation system?**

**Answer:**

**Offline Evaluation Metrics:**

| Metric | Use Case | Formula/Description |
|--------|----------|--------------------|
| **RMSE** | Rating prediction | √(Σ(actual - predicted)²/n) |
| **MAE** | Rating prediction | Σ|actual - predicted|/n |
| **Precision@K** | Top-K accuracy | Relevant in top-K / K |
| **Recall@K** | Coverage | Relevant in top-K / Total relevant |
| **NDCG@K** | Ranking quality | Considers position of relevant items |
| **MAP** | Ranking | Mean Average Precision |
| **AUC** | Binary relevance | Area under ROC curve |

---

**Beyond Accuracy:**

| Metric | What It Measures |
|--------|------------------|
| **Coverage** | % of items recommended |
| **Diversity** | Variety in recommendations |
| **Novelty** | How surprising/new items are |
| **Serendipity** | Unexpected but liked items |

---

**Online Evaluation:**

| Method | Description |
|--------|-------------|
| **A/B Testing** | Compare algorithms on real users |
| **Click-through rate** | User engagement |
| **Conversion rate** | Purchases/actions taken |
| **Dwell time** | Time spent on recommendations |

---

**Evaluation Strategies:**

1. **Train/Test Split** - Random or temporal
2. **Cross-Validation** - K-fold for robust estimates
3. **Leave-One-Out** - Hide one interaction per user

**Best Practice:** Combine offline metrics with online A/B tests

---

## Question 5
- [ ] Done

**How do recommendation systems handle changing user preferences over time?**

**Answer:**

**Challenges:**
- User interests evolve (short-term vs long-term)
- Concept drift in preferences
- Seasonality effects

---

**Strategies to Handle Temporal Dynamics:**

**1. Time Decay Weighting**
```python
# Recent interactions weighted more
weight = np.exp(-decay_rate * (current_time - interaction_time))
weighted_rating = rating * weight
```

**2. Sliding Window**
- Only use last N interactions or last T days
- Discards outdated preferences

**3. Temporal Models**

| Approach | Description |
|----------|-------------|
| **TimeSVD++** | Time-varying latent factors |
| **RNN/LSTM** | Sequence of user actions |
| **Attention** | Focus on relevant past items |
| **Session-based** | Model short-term intent |

**4. Multi-Interest Models**
- Separate short-term (session) and long-term (history) preferences
- Combine with weighted sum

**5. Periodic Retraining**
- Retrain model on schedule
- Incremental/online learning

---

**Best Practices:**
- Separate seasonal preferences (winter clothes in winter)
- Detect preference shifts using change point detection
- A/B test recency window size

---

## Question 6
- [ ] Done

**How can content-based recommendation systems utilize natural language processing (NLP)?**

**Answer:**

**NLP Applications in Content-Based Recommendations:**

**1. Text Representation**

| Technique | Description |
|-----------|-------------|
| **TF-IDF** | Term frequency-inverse doc frequency |
| **Word2Vec** | Dense word embeddings |
| **Doc2Vec** | Document-level embeddings |
| **BERT** | Contextual embeddings |
| **Sentence-BERT** | Sentence-level embeddings |

**2. Feature Extraction from Text**
```python
# TF-IDF for item descriptions
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000)
item_vectors = vectorizer.fit_transform(descriptions)
```

**3. Semantic Similarity**
```python
# Cosine similarity between item embeddings
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(user_profile, item_vectors)
```

---

**Use Cases:**

| Application | NLP Usage |
|-------------|----------|
| **News/Articles** | Topic modeling, entity extraction |
| **E-commerce** | Product description similarity |
| **Books** | Genre classification, theme extraction |
| **Reviews** | Sentiment analysis for filtering |

---

**Advanced Techniques:**
- Named Entity Recognition (extract actors, brands)
- Topic Modeling (LDA for item topics)
- Sentiment Analysis (positive/negative aspects)
- Aspect-Based Analysis (specific feature preferences)

**Advantage:** Captures semantic meaning, not just keyword overlap

---

## Question 7
- [ ] Done

**How are Bayesian networks used in recommendation systems?**

**Answer:**

**What are Bayesian Networks?**
Directed acyclic graphs (DAGs) representing probabilistic dependencies between variables.

---

**Applications in Recommendations:**

**1. Modeling User Preferences**
```
[Demographics] → [Latent Preferences] → [Item Ratings]
           ↑
    [Context]
```

**2. Probabilistic Inference**
- Compute P(like item | observed evidence)
- Handle uncertainty naturally
- Update beliefs as new data arrives

**3. Explainable Recommendations**
- Clear causal structure
- Can trace why an item was recommended

---

**Key Advantages:**

| Benefit | Description |
|---------|-------------|
| **Uncertainty** | Quantifies confidence in predictions |
| **Prior knowledge** | Incorporate domain expertise |
| **Missing data** | Handles partial observations |
| **Interpretability** | Graphical structure is intuitive |

---

**Example Structure:**
```
User Demographics → Genre Preference
        ↓                  ↓
   Time Context    →    Rating
        ↓                  ↓
   Device Type   →   Engagement
```

**Algorithms:**
- Variable Elimination, Belief Propagation for inference
- EM algorithm for learning parameters

**Limitation:** Scales poorly with many variables; often combined with other methods in hybrid systems.

---

## Question 8
- [ ] Done

**Outline a strategy to improve movie recommendations on a platform with diverse user demographics.**

**Answer:**

**Strategy Framework:**

**1. Multi-Signal Data Collection**

| Signal | Description |
|--------|-------------|
| Explicit | Ratings, reviews |
| Implicit | Watch time, completion rate, replays |
| Demographics | Age, region, language |
| Context | Time, device, mood |

**2. Hybrid Recommendation System**
```python
def hybrid_recommend(user):
    cf_scores = collaborative_filtering(user)      # Similar users
    cb_scores = content_based(user)                # Genre, actors
    demo_scores = demographic_model(user)          # Age group patterns
    
    # Weighted combination
    return alpha * cf_scores + beta * cb_scores + gamma * demo_scores
```

**3. Demographic-Aware Modeling**
- Segment users by demographics
- Learn separate embedding spaces or biases
- Personalize within and across segments

**4. Cultural/Regional Adaptation**
- Regional popular items
- Language-specific content
- Cultural context in descriptions

**5. Diversity in Recommendations**
- Ensure genre variety
- Include niche and mainstream
- Serendipity: occasionally surface unexpected items

**6. Evaluation by Segment**
- Monitor metrics per demographic group
- Ensure fair coverage for all groups

**Key Success Factors:**
- Continuous A/B testing
- Personalized onboarding for new users
- Balance popularity bias with personalization

---

## Question 9
- [ ] Done

**Present an approach for a recommendation system in the educational technology sector.**

**Answer:**

**Unique EdTech Challenges:**
- Learning goals matter, not just engagement
- Prerequisite knowledge dependencies
- Spaced repetition for retention
- Different skill levels

---

**Approach:**

**1. Knowledge Graph Structure**
```
Concept A (prereq) → Concept B → Concept C (advanced)
      ↓
  Related Topics
```

**2. Learner Modeling**

| Component | Description |
|-----------|-------------|
| Skill assessment | Current mastery level |
| Learning style | Visual, reading, practice |
| Progress tracking | Completed topics |
| Performance | Quiz scores, time spent |

**3. Recommendation Types**

| Type | Purpose |
|------|--------|
| **Next concept** | Follow prerequisite chain |
| **Remedial** | Review if struggling |
| **Challenge** | Stretch when mastering |
| **Practice** | Reinforce weak areas |

**4. Adaptive Learning Path**
```python
def recommend_next(learner):
    mastered = learner.mastered_concepts
    ready_to_learn = get_concepts_with_prereqs_met(mastered)
    weakest = sort_by_mastery(ready_to_learn)[:k]
    return blend(weakest, learner.goals)
```

**5. Spaced Repetition**
- Schedule reviews based on forgetting curve
- Prioritize concepts about to be forgotten

**Metrics:**
- Learning outcomes (test scores improvement)
- Completion rates
- Long-term retention
- Student engagement

**Key:** Optimize for learning, not just clicks.

---

## Question 10
- [ ] Done

**What roles do multi-armed bandit algorithms play in recommendation systems?**

**Answer:**

**What is Multi-Armed Bandit?**
Framework for balancing exploration (try new items) vs exploitation (use known good items) in sequential decisions.

---

**Why Use Bandits in Recommendations?**

| Challenge | How Bandits Help |
|-----------|------------------|
| Cold start | Explore new items/users |
| Changing preferences | Continuously adapt |
| Real-time learning | Update with each interaction |
| Uncertainty | Quantify confidence in predictions |

---

**Common Algorithms:**

**1. ε-Greedy**
```python
if random.random() < epsilon:
    return random_item()       # Explore
else:
    return best_known_item()   # Exploit
```

**2. Upper Confidence Bound (UCB)**
$$a = \arg\max_a \left[ \hat{\mu}_a + c\sqrt{\frac{\ln t}{n_a}} \right]$$

Select item with highest (estimated value + uncertainty bonus)

**3. Thompson Sampling**
- Maintain probability distribution per item
- Sample from distribution, pick highest
- Naturally balances exploration/exploitation

---

**Contextual Bandits:**
Incorporate user/item features:
$$\text{Reward} = f(\text{context}) + \text{noise}$$

Examples: LinUCB, Neural Bandits

---

**Use Cases:**
- Homepage item ranking
- New item exposure
- Personalized notifications
- A/B testing automation

**Advantage:** Online learning without full retraining

---

## Question 11
- [ ] Done

**Explore the use of graph-based recommendation systems and the potential advantages they offer.**

**Answer:**

**Graph Representation:**
```
Users ←→ Items ←→ Attributes
  ↓         ↓
Users     Items (via co-purchase)
```

**Types of Graphs:**

| Graph Type | Nodes | Edges |
|------------|-------|-------|
| Bipartite | Users, Items | Interactions |
| Heterogeneous | Users, Items, Attributes | Multiple relations |
| Knowledge Graph | Entities | Semantic relations |

---

**Graph-Based Methods:**

**1. Random Walk**
```python
# Personalized PageRank from user node
for _ in range(num_steps):
    if random.random() < alpha:
        current = start_user    # Restart
    else:
        current = random_neighbor(current)
    visit_counts[current] += 1
# Recommend most visited items
```

**2. Graph Neural Networks (GNN)**
- Learn node embeddings by aggregating neighbor info
- Examples: GraphSAGE, GAT, PinSage

**3. Graph Convolutions**
- Message passing between connected nodes
- LightGCN: Simplified GNN for recommendations

---

**Advantages:**

| Benefit | Description |
|---------|-------------|
| **Higher-order relations** | Capture user-item-user paths |
| **Rich side information** | Incorporate attributes naturally |
| **Explainability** | Path-based explanations |
| **Cold start** | Propagate through content edges |
| **Scalability** | Efficient with sampling (e.g., PinSage) |

---

**Real-World Examples:**
- Pinterest: PinSage for visual recommendations
- Alibaba: Knowledge graph for products
- Social networks: Friend-of-friend suggestions

**Limitation:** Graph construction and storage overhead

---

