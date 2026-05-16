# Light Gbm Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What is LightGBM and how does it differ from other gradient boosting frameworks?**

**Answer:**

**Definition:**
LightGBM (Light Gradient Boosting Machine) is a fast, distributed, high-performance gradient boosting framework developed by Microsoft. It uses tree-based learning algorithms designed for speed and efficiency.

---

**Key Differences from Other Frameworks:**

| Aspect | LightGBM | XGBoost | CatBoost |
|--------|----------|---------|----------|
| Tree growth | Leaf-wise | Depth-wise | Symmetric |
| Speed | Fastest | Fast | Moderate |
| Memory | Low | High | Moderate |
| Categorical features | Native support | Encoding needed | Best native support |
| Histogram binning | Yes | Yes | Yes |

---

**LightGBM Unique Features:**

**1. Leaf-wise Tree Growth:**
- Grows tree by splitting leaf with max gain
- Faster convergence, may overfit

**2. GOSS (Gradient-based One-Side Sampling):**
- Keeps instances with large gradients
- Randomly samples from small gradients
- Reduces data while preserving accuracy

**3. EFB (Exclusive Feature Bundling):**
- Bundles mutually exclusive sparse features
- Reduces effective feature count

**4. Histogram-based Algorithm:**
- Bins continuous values into discrete buckets
- Faster split finding, lower memory

---

**When to Use LightGBM:**
- Large datasets (millions of rows)
- Need fast training/inference
- High-dimensional data
- Tabular data with mixed feature types

---

## Question 2
- [ ] Done

**How does LightGBM handle categorical features differently from other tree-based algorithms?**

**Answer:**

**Traditional Approach (XGBoost, Random Forest):**
- Requires preprocessing: one-hot encoding or label encoding
- One-hot creates sparse, high-dimensional data
- Label encoding implies false ordinal relationship

**LightGBM Native Categorical Handling:**
- Uses optimal split algorithm for categorical features
- No preprocessing required
- Finds best split by partitioning categories into two groups

---

**How It Works:**

```python
# Specify categorical features
params = {'categorical_feature': [0, 2, 5]}  # indices
# OR
params = {'categorical_feature': 'name:feature1,feature2'}
```

**Split Finding Algorithm:**
1. Sort categories by gradient statistics
2. Find optimal split point in sorted order
3. Time complexity: O(k log k) where k = number of categories

---

**Advantages:**

| Aspect | One-Hot Encoding | LightGBM Native |
|--------|-----------------|-----------------|
| Memory | High (sparse) | Low |
| Speed | Slow | Fast |
| Feature count | Explodes | Unchanged |
| High cardinality | Problematic | Handles well |

---

**Best Practices:**
- Convert categorical columns to `category` dtype in pandas
- Let LightGBM handle splits internally
- Works best when categories have meaningful groupings

```python
df['category_col'] = df['category_col'].astype('category')
```

---

## Question 3
- [ ] Done

**Can you explain the concept of Gradient Boosting and how LightGBM utilizes it?**

**Answer:**

**Gradient Boosting Concept:**
Gradient boosting is an ensemble technique that builds models sequentially, where each new model corrects errors made by previous models by fitting to the negative gradient (residuals) of the loss function.

**Mathematical Formulation:**
$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

Where:
- F_m = model after m iterations
- h_m = new tree fitted to negative gradients
- η = learning rate

---

**How LightGBM Implements Gradient Boosting:**

**1. Compute Gradients:**
For each sample, calculate gradient g_i and Hessian h_i of loss function

**2. Build Histogram:**
Bin continuous features into discrete buckets

**3. Find Best Split (Leaf-wise):**
- Select leaf with maximum potential gain
- Use histogram to find optimal split point
- Split criterion: maximize gain

**4. Update Predictions:**
$$\text{prediction} += \text{learning\_rate} \times \text{leaf\_value}$$

---

**LightGBM Optimization:**

| Standard GB | LightGBM Enhancement |
|-------------|---------------------|
| Full data scan | GOSS (sample data) |
| All features | EFB (bundle features) |
| Exact split | Histogram approximation |
| Depth-wise | Leaf-wise growth |

**Result:** Same gradient boosting algorithm, but highly optimized for speed and memory.

---

## Question 4
- [ ] Done

**What are some of the advantages of LightGBM over XGBoost or CatBoost?**

**Answer:**

**LightGBM Advantages:**

| Advantage | Compared to XGBoost | Compared to CatBoost |
|-----------|--------------------|--------------------|
| Training speed | 10-20x faster | 5-10x faster |
| Memory usage | Much lower | Lower |
| Large datasets | Better scalability | Better |
| Categorical features | Native support | Similar |

---

**Specific Advantages:**

**1. Speed**
- Histogram-based algorithm reduces split finding complexity
- GOSS samples data, EFB reduces features
- Leaf-wise growth converges faster

**2. Memory Efficiency**
- Histogram binning: O(bins) vs O(data_size)
- Sparse feature handling via EFB
- No need for one-hot encoding

**3. Accuracy on Large Data**
- More trees in same training time
- Leaf-wise often reaches better accuracy
- Better utilization of compute resources

**4. Distributed Training**
- Built-in parallel training
- Data and feature parallelism
- GPU support

**5. Flexibility**
- Supports classification, regression, ranking, multi-class
- Custom loss functions
- Multiple boosting types (gbdt, dart, goss)

---

**When Others May Be Better:**

| Scenario | Better Choice |
|----------|---------------|
| Small data (risk of overfit) | XGBoost (depth-wise) |
| Many high-cardinality categoricals | CatBoost |
| Need symmetric trees | CatBoost |

---

## Question 5
- [ ] Done

**How does LightGBM achieve faster training and lower memory usage?**

**Answer:**

**Key Optimizations:**

**1. Histogram-Based Algorithm**
- Bins continuous values into k discrete buckets (default 255)
- Split finding: O(k) instead of O(n)
- Memory: O(k) instead of O(n)

```
Before: Sort all n values for each feature
After: Scan k histogram bins
```

**2. GOSS (Gradient-based One-Side Sampling)**
- Keep samples with large gradients (high learning value)
- Randomly sample from small gradients
- Reduces data size while preserving distribution

```
Large gradients: Keep top a% (e.g., 20%)
Small gradients: Random sample b% (e.g., 10%)
Total: ~30% of data with similar accuracy
```

**3. EFB (Exclusive Feature Bundling)**
- Bundles sparse features that rarely take non-zero values together
- Reduces feature count significantly for sparse data
- Especially useful for one-hot encoded features

**4. Leaf-wise Tree Growth**
- Grows deepest leaf first (highest gain)
- Fewer splits needed for same reduction in loss
- Converges faster than depth-wise

---

**Memory Comparison:**

| Component | XGBoost | LightGBM |
|-----------|---------|----------|
| Feature values | O(n × features) | O(bins × features) |
| Gradient storage | Full data | Sampled (GOSS) |
| Sparse features | Individual | Bundled (EFB) |

**Result:** 10x+ speedup, 5x+ memory reduction on large datasets.

---

## Question 6
- [ ] Done

**Explain the histogram-based approach used by LightGBM.**

**Answer:**

**Definition:**
Histogram-based approach discretizes continuous feature values into fixed-size bins (buckets), then uses these histograms for faster split finding.

---

**How It Works:**

**Step 1: Binning**
```
Continuous values: [0.1, 0.3, 0.5, 0.8, 1.2, 1.5, 2.0, 2.3]
Bins (4): [0-0.6] [0.6-1.2] [1.2-1.8] [1.8-2.4]
Histogram: [3, 2, 1, 2]  # count per bin
```

**Step 2: Gradient Accumulation**
- Each bin stores sum of gradients and Hessians
- Histogram[bin_id] = Σ(g_i, h_i) for samples in bin

**Step 3: Split Finding**
- Scan histogram bins instead of all values
- Find split with maximum gain
- Complexity: O(bins) instead of O(n)

---

**Histogram Subtraction Trick:**
```
Histogram(sibling) = Histogram(parent) - Histogram(child)
```
Only compute histogram for smaller child, get other by subtraction.

---

**Benefits:**

| Aspect | Traditional | Histogram |
|--------|-------------|-----------|
| Memory per feature | O(n) | O(bins) |
| Split finding time | O(n log n) | O(bins) |
| Cache efficiency | Poor | Good (continuous memory) |

**Key Parameter:**
- `max_bin` (default=255): Number of bins
- More bins = more precision, slower training
- Fewer bins = faster, slight accuracy loss

---

## Question 7
- [ ] Done

**What is meant by "leaf-wise" tree growth in LightGBM, and how is it different from "depth-wise" growth?**

**Answer:**

**Leaf-wise Growth (LightGBM):**
- Splits the leaf with highest potential gain
- Tree grows asymmetrically
- Can create very deep trees on one side

**Depth-wise Growth (XGBoost, Traditional):**
- Expands all nodes at current depth before going deeper
- Tree grows level by level
- Produces balanced trees

---

**Visual Comparison:**

```
Depth-wise (level-by-level):
       [Root]
      /      \
   [L1]      [L1]
   /  \      /  \
 [L2] [L2] [L2] [L2]

Leaf-wise (best leaf first):
       [Root]
      /      \
   [L1]      [Deep]
             /    \
          [L2]    [L3]
                    \
                   [L4]
```

---

**Comparison:**

| Aspect | Leaf-wise | Depth-wise |
|--------|-----------|------------|
| Speed | Faster convergence | Slower |
| Accuracy | Often higher | Stable |
| Overfitting risk | Higher (deep trees) | Lower |
| Memory | Variable | Predictable |

---

**When to Use:**

| Scenario | Recommendation |
|----------|----------------|
| Large dataset | Leaf-wise (LightGBM) |
| Small dataset | Depth-wise (set `max_depth`) |
| High overfitting | Limit `num_leaves`, `max_depth` |

**Control Overfitting in Leaf-wise:**
```python
params = {
    'num_leaves': 31,    # Limit leaves
    'max_depth': 6,      # Limit depth
    'min_data_in_leaf': 20  # Minimum samples per leaf
}
```

---

## Question 8
- [ ] Done

**Explain how LightGBM deals with overfitting.**

**Answer:**

**Overfitting in LightGBM:**
Leaf-wise growth can create very deep trees that memorize training data. LightGBM provides multiple regularization mechanisms.

---

**Regularization Parameters:**

**1. Tree Structure Controls**

| Parameter | Effect | Default |
|-----------|--------|---------|
| `num_leaves` | Max leaves per tree | 31 |
| `max_depth` | Max tree depth | -1 (unlimited) |
| `min_data_in_leaf` | Min samples per leaf | 20 |
| `min_sum_hessian_in_leaf` | Min sum of Hessian | 1e-3 |

**2. Regularization Terms**

| Parameter | Effect |
|-----------|--------|
| `lambda_l1` | L1 regularization on leaf weights |
| `lambda_l2` | L2 regularization on leaf weights |
| `min_gain_to_split` | Minimum gain required to make split |

**3. Sampling**

| Parameter | Effect |
|-----------|--------|
| `bagging_fraction` | Row subsampling (< 1.0) |
| `feature_fraction` | Column subsampling (< 1.0) |
| `bagging_freq` | How often to apply bagging |

**4. Early Stopping**
```python
model = lgb.train(params, train_data,
                  valid_sets=[valid_data],
                  early_stopping_rounds=50)
```

---

**Recommended Settings for Overfitting:**
```python
params = {
    'num_leaves': 31,         # Lower = less overfit
    'max_depth': 6,           # Limit depth
    'min_data_in_leaf': 50,   # Increase
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'bagging_fraction': 0.8,
    'feature_fraction': 0.8
}
```

---

## Question 9
- [ ] Done

**What is Feature Parallelism and Data Parallelism in the context of LightGBM?**

**Answer:**

**Data Parallelism:**
Split data across machines, each machine processes subset of rows.

```
Machine 1: Rows 1-1M
Machine 2: Rows 1M-2M
Machine 3: Rows 2M-3M
→ Aggregate histograms → Find global best split
```

**How it works:**
1. Each worker builds local histograms
2. Workers communicate histograms
3. Global histogram = sum of local histograms
4. Best split determined from global histogram

---

**Feature Parallelism:**
Split features across machines, each processes subset of features.

```
Machine 1: Features 1-100
Machine 2: Features 101-200
Machine 3: Features 201-300
→ Find local best → Compare → Global best split
```

**How it works:**
1. Each worker finds best split for its features
2. Workers share their best splits
3. Global best chosen from all candidates

---

**Comparison:**

| Aspect | Data Parallel | Feature Parallel |
|--------|---------------|------------------|
| Best for | Many rows | Many features |
| Communication | Histogram aggregation | Split point sharing |
| Scalability | Better (less comm) | Limited by feature count |
| LightGBM default | Preferred | Alternative |

**LightGBM Voting Parallel (Hybrid):**
- Each worker finds local best splits
- Vote on best features
- Only aggregate histograms for top features
- Reduces communication overhead

**Usage:**
```python
params = {
    'tree_learner': 'data'  # or 'feature', 'voting'
}
```

---

## Question 10
- [ ] Done

**Explain the role of the learning rate in the LightGBM algorithm.**

**Answer:**

**Definition:**
Learning rate (η) controls how much each tree contributes to the final prediction. It shrinks the step size during gradient descent, preventing overshooting.

**Update Formula:**
$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

Where η = learning_rate

---

**Effect of Learning Rate:**

| Learning Rate | Effect |
|---------------|--------|
| **High (0.3-1.0)** | Fast learning, may overshoot optimal, fewer trees needed |
| **Low (0.01-0.1)** | Slow learning, better generalization, more trees needed |
| **Very low (<0.01)** | Very slow, may need thousands of trees |

---

**Trade-off:**

```
High η + Few trees = Fast but may overfit
Low η + Many trees = Slow but better generalization
```

**Best Practice:**
- Start with low learning rate (0.01-0.1)
- Increase `num_iterations` (more trees)
- Use early stopping to find optimal number

---

**Common Values:**

| Scenario | Learning Rate | Trees |
|----------|---------------|-------|
| Quick testing | 0.1-0.3 | 100-500 |
| Final model | 0.01-0.05 | 1000-5000 |
| Competition | 0.005-0.02 | 5000+ |

**Parameter Name:**
```python
params = {
    'learning_rate': 0.05,  # or 'eta'
    'num_iterations': 1000
}
```

---

## Question 11
- [ ] Done

**What is the significance of the min_data_in_leaf parameter in LightGBM?**

**Answer:**

**Definition:**
`min_data_in_leaf` specifies the minimum number of samples required in a leaf node. A split is only allowed if both resulting leaves have at least this many samples.

---

**Role and Significance:**

**1. Regularization**
- Prevents leaves with too few samples
- Reduces overfitting by avoiding memorization of noise

**2. Statistical Reliability**
- Ensures predictions are based on sufficient data
- More robust leaf value estimates

**3. Tree Complexity Control**
- Higher value = simpler trees
- Lower value = more complex trees

---

**Effect of Different Values:**

| Value | Effect |
|-------|--------|
| **Low (1-10)** | Very detailed trees, may overfit |
| **Medium (20-50)** | Balanced, good default |
| **High (100+)** | Simple trees, may underfit |

---

**Guidelines:**

| Dataset Size | Recommended min_data_in_leaf |
|--------------|------------------------------|
| Small (<10K) | 50-100 |
| Medium (10K-100K) | 20-50 |
| Large (>100K) | 10-20 |

**Relationship with Other Parameters:**
- Related to `num_leaves`: More leaves need higher min_data_in_leaf
- Rule of thumb: Total samples / num_leaves > min_data_in_leaf

```python
params = {
    'min_data_in_leaf': 20,  # Default
    'num_leaves': 31
}
```

---

## Question 12
- [ ] Done

**How does LightGBM handle missing values?**

**Answer:**

**Native Missing Value Support:**
LightGBM handles missing values automatically without imputation. Missing values are assigned to the side of the split that minimizes loss.

---

**How It Works:**

**During Training:**
1. For each split, try sending missing values to left OR right
2. Choose direction that maximizes gain
3. Store optimal direction in tree structure

**During Prediction:**
- Follow stored direction for missing values
- No preprocessing needed

---

**Key Parameters:**

| Parameter | Description |
|-----------|-------------|
| `use_missing` | Enable missing value handling (default: True) |
| `zero_as_missing` | Treat zeros as missing (default: False) |

```python
params = {
    'use_missing': True,
    'zero_as_missing': False
}
```

---

**Comparison with Manual Imputation:**

| Aspect | LightGBM Native | Manual Imputation |
|--------|-----------------|-------------------|
| Preprocessing | None needed | Required |
| Optimal direction | Learned per split | Fixed value |
| Flexibility | Different handling per feature | Same for all |
| Speed | Faster | Slower |

---

**Best Practices:**
- Let LightGBM handle missing values natively
- No need to fill NaN/None values
- Ensure missing values are represented as NaN (not placeholder like -999)

**Note:** For categorical features, missing is treated as a separate category.

---

## Question 13
- [ ] Done

**What are the potential pitfalls when using LightGBM on small datasets?**

**Answer:**

**Main Issue: Overfitting**
LightGBM's leaf-wise growth is optimized for large datasets. On small data, it can easily memorize training samples.

---

**Specific Pitfalls:**

**1. Deep Trees Overfit**
- Leaf-wise creates asymmetric, deep trees
- Few samples per leaf = noisy predictions

**2. Insufficient Regularization**
- Default parameters tuned for large data
- May need aggressive regularization

**3. GOSS May Be Harmful**
- Gradient-based sampling reduces data further
- Not beneficial when data is already small

**4. High Variance**
- Small validation set = unreliable metrics
- Cross-validation becomes essential

---

**Recommended Settings for Small Data:**

```python
params = {
    'boosting_type': 'gbdt',  # Not GOSS
    'num_leaves': 15,         # Fewer leaves (default 31)
    'max_depth': 5,           # Limit depth
    'min_data_in_leaf': 50,   # Higher minimum
    'min_sum_hessian_in_leaf': 10,
    'lambda_l1': 1.0,         # Strong L1 regularization
    'lambda_l2': 1.0,         # Strong L2 regularization
    'bagging_fraction': 0.7,
    'feature_fraction': 0.7,
    'learning_rate': 0.01,    # Lower learning rate
}
```

---

**Alternatives to Consider:**
- XGBoost (depth-wise may be safer)
- Random Forest (more stable on small data)
- Simpler models (Logistic Regression, SVM)
- Always use cross-validation for evaluation

---

## Question 14
- [ ] Done

**Explain the importance of early stopping in training LightGBM models.**

**Answer:**

**Definition:**
Early stopping halts training when validation performance stops improving, preventing overfitting and saving computation time.

---

**How It Works:**
1. Train model iteratively (add trees)
2. Evaluate on validation set after each iteration
3. Track best validation score
4. Stop if no improvement for N rounds
5. Return model with best validation score

```python
model = lgb.train(
    params,
    train_data,
    num_boost_round=10000,       # Max iterations
    valid_sets=[valid_data],      # Validation data
    early_stopping_rounds=100     # Patience
)
```

---

**Benefits:**

| Benefit | Explanation |
|---------|-------------|
| **Prevents overfitting** | Stops before model memorizes training data |
| **Saves time** | No need to train full num_iterations |
| **Automatic tuning** | Finds optimal number of trees |
| **Simpler hyperparameter tuning** | Set high max iterations, let early stopping decide |

---

**Key Parameters:**

| Parameter | Description |
|-----------|-------------|
| `early_stopping_rounds` | Number of rounds without improvement before stopping |
| `valid_sets` | Validation dataset(s) for monitoring |
| `metric` | Which metric to monitor |

**Best Practices:**
- Use 10-20% of data for validation
- Set `early_stopping_rounds` = 50-200
- Set `num_boost_round` very high (10000+)
- Model automatically uses best iteration for prediction

---

## Question 15
- [ ] Done

**Describe a strategy for updating a LightGBM model as new data becomes available.**

**Answer:**

**Strategies for Model Updates:**

**1. Incremental Training (Continue Training)**
```python
# Initial training
model = lgb.train(params, initial_data, num_boost_round=100)

# Continue training with new data
model = lgb.train(
    params,
    new_data,
    num_boost_round=50,
    init_model=model  # Start from existing model
)
```
- Adds more trees to existing model
- Fast, preserves learned patterns
- Risk: may overfit to new data distribution

**2. Full Retraining**
```python
# Combine old and new data
all_data = lgb.Dataset(np.vstack([old_X, new_X]), 
                        label=np.hstack([old_y, new_y]))
model = lgb.train(params, all_data)
```
- Most reliable
- Computationally expensive
- Best when distribution shifts

**3. Sliding Window**
- Keep only recent N months of data
- Retrain periodically on window
- Good for time-series with drift

---

**Recommended Approach:**

| Scenario | Strategy |
|----------|----------|
| Stable distribution | Incremental training |
| Data drift suspected | Full retraining |
| Time-series | Sliding window retraining |
| Real-time updates | Periodic batch retraining |

---

**Monitoring for Retraining:**
- Track prediction performance over time
- Detect distribution drift (PSI, KL divergence)
- Automated triggers when performance degrades

```python
if validation_metric < threshold:
    model = retrain_model(all_data)
```

---

## Question 16
- [ ] Done

**Explain how LightGBM models can be interpreted and what tools can assist in model interpretation.**

**Answer:**

**Built-in Interpretation Methods:**

**1. Feature Importance**
```python
# Gain-based importance
importance = model.feature_importance(importance_type='gain')

# Split-based importance
importance = model.feature_importance(importance_type='split')
```

| Type | Meaning |
|------|---------|
| `gain` | Total gain from splits on this feature |
| `split` | Number of times feature is used for splitting |

**2. Plot Feature Importance**
```python
lgb.plot_importance(model, importance_type='gain', max_num_features=20)
```

---

**External Interpretation Tools:**

**1. SHAP (SHapley Additive exPlanations)**
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
```
- Shows feature contribution per prediction
- Global and local interpretability

**2. LIME (Local Interpretable Model-agnostic Explanations)**
- Explains individual predictions
- Creates local linear approximation

**3. Partial Dependence Plots**
```python
from sklearn.inspection import PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(model, X, features=[0, 1])
```
- Shows marginal effect of feature on prediction

---

**Summary:**

| Tool | Use Case |
|------|----------|
| Feature importance | Global feature ranking |
| SHAP | Individual prediction explanation |
| PDP | Feature-target relationship |
| Tree visualization | Understand decision rules |

```python
lgb.plot_tree(model, tree_index=0)  # Visualize first tree
```

---

## Question 17
- [ ] Done

**What is a decision tree's "gain" and "split" in the context of LightGBM, and how are they important?**

**Answer:**

**Definitions:**

**Gain:**
The reduction in loss achieved by making a split. Higher gain = better split.

$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma$$

Where:
- G_L, G_R = sum of gradients in left/right child
- H_L, H_R = sum of Hessians in left/right child
- λ = L2 regularization
- γ = minimum gain to split

**Split:**
The decision point where a node divides data based on feature value.
- Feature: which feature to use
- Threshold: value to split on

---

**Importance:**

**1. Feature Importance Calculation**

| importance_type | What it measures |
|-----------------|------------------|
| `gain` | Total improvement from feature's splits |
| `split` | How often feature is used |

**2. Tree Construction**
- LightGBM chooses split with maximum gain
- Splits continue until gain < `min_gain_to_split`

**3. Regularization**
- `min_gain_to_split`: Minimum gain required (default: 0)
- Higher value = fewer splits = simpler trees

---

**Practical Usage:**
```python
# Get feature importance by gain
gain_importance = model.feature_importance(importance_type='gain')

# Get feature importance by split count
split_importance = model.feature_importance(importance_type='split')

# Prevent low-gain splits
params = {'min_gain_to_split': 0.1}
```

---

## Question 18
- [ ] Done

**Describe how you would train a LightGBM model to recommend products based on user behavior data.**

**Answer:**

**Approach: Learning-to-Rank with LightGBM**

LightGBM supports ranking objectives that can be used for recommendation.

---

**Data Preparation:**

| Feature Type | Examples |
|--------------|----------|
| User features | Age, location, purchase history |
| Product features | Category, price, ratings |
| Interaction features | Views, time spent, add-to-cart |
| Context | Time of day, device, season |

**Label:** Relevance score (clicks, purchases, ratings)

---

**Data Format for Ranking:**
```python
# Each query (user session) has multiple items with relevance scores
train_data = lgb.Dataset(
    X_train,
    label=y_train,  # Relevance scores
    group=group_sizes  # Number of items per query
)
```

---

**Training:**
```python
params = {
    'objective': 'lambdarank',  # or 'rank_xendcg'
    'metric': 'ndcg',
    'ndcg_eval_at': [5, 10],    # NDCG@5, NDCG@10
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=1000,
    early_stopping_rounds=50
)
```

---

**Prediction:**
```python
# Get scores for candidate products
scores = model.predict(X_candidates)
# Rank by score, recommend top-N
top_n = np.argsort(scores)[::-1][:10]
```

**Ranking Objectives Available:**
- `lambdarank`: Optimizes NDCG
- `rank_xendcg`: Cross-entropy loss for ranking

---

## Question 19
- [ ] Done

**What are the challenges and strategies associated with distributed training of LightGBM models?**

**Answer:**

**Challenges:**

| Challenge | Description |
|-----------|-------------|
| **Communication overhead** | Workers must share histograms/splits |
| **Data skew** | Uneven distribution across workers |
| **Synchronization** | Waiting for slowest worker |
| **Fault tolerance** | Handling worker failures |
| **Memory management** | Coordinating across nodes |

---

**Distributed Training Strategies:**

**1. Data Parallel (Default)**
```python
params = {
    'tree_learner': 'data',
    'num_machines': 4
}
```
- Split data rows across machines
- Aggregate histograms
- Best for: many rows

**2. Feature Parallel**
```python
params = {'tree_learner': 'feature'}
```
- Split features across machines
- Share split decisions
- Best for: many features

**3. Voting Parallel**
```python
params = {'tree_learner': 'voting'}
```
- Hybrid approach
- Workers vote on best features
- Reduces communication

---

**Setup with Dask/Spark:**

```python
# Dask example
import dask.dataframe as dd
from dask_lightgbm import train

ddf = dd.from_pandas(df, npartitions=4)
model = train(params, ddf, label='target')
```

**Best Practices:**
- Use data parallelism when rows >> features
- Ensure balanced data partitioning
- Use high-bandwidth network
- Consider GPU training for large data (faster than distributed CPU)

**Alternative:** Train on sampled data locally, then validate on full dataset.

---

