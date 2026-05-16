# Gradient Descent Interview Questions - Scenario Based Questions

## Question 1
- [ ] Done

**What could cause gradient descent to converge very slowly, and how would you counteract it?**

**Answer:**

**Common Causes and Solutions:**

| Cause | How to Identify | Solution |
|-------|-----------------|----------|
| Learning rate too small | Loss decreases very slowly | Increase LR, use LR finder |
| Poor feature scaling | Features have different ranges | Standardize/normalize features |
| Ill-conditioned Hessian | Narrow ravines in loss surface | Use Adam (adaptive LR) or momentum |
| Saddle points | Loss plateaus, gradient near zero | Add momentum, use SGD noise |
| Vanishing gradients | Early layers don't learn | Use ReLU, batch norm, skip connections |

**Step-by-Step Diagnosis:**

1. **Check learning rate first:**
   - Plot loss curve
   - If loss barely moves, try 10x higher LR
   - Use learning rate finder

2. **Check feature scaling:**
   - Are features on similar scales?
   - Apply StandardScaler or MinMaxScaler

3. **Try different optimizer:**
   - Switch from vanilla SGD to Adam
   - Adam handles ill-conditioned problems better

4. **Add momentum:**
   - Helps push through flat regions
   - Try momentum = 0.9

5. **Check for vanishing gradients:**
   - Monitor gradient norms per layer
   - Switch to ReLU, add batch normalization

**Quick Fix Checklist:**
- [ ] Increase learning rate
- [ ] Scale features
- [ ] Use Adam optimizer
- [ ] Add momentum
- [ ] Check gradient magnitudes

---

## Question 2
- [ ] Done

**Discuss the importance of convergence criteria in gradient descent.**

**Answer:**

Convergence criteria determine when to stop training. Without proper criteria, you may train too long (overfitting, wasted compute) or stop too early (underfitting). Multiple criteria should be used together for robust stopping.

**Common Convergence Criteria:**

| Criterion | How It Works | When to Use |
|-----------|--------------|-------------|
| Max epochs | Stop after N epochs | Always set as safety limit |
| Gradient norm | Stop when \|\|gradient\|\| < threshold | Theoretical convergence |
| Loss plateau | Stop when loss change < threshold | Practical convergence |
| Validation loss | Stop when val_loss stops improving | Prevent overfitting (early stopping) |
| Parameter change | Stop when \|\|theta_new - theta_old\|\| < threshold | Stability check |

**Early Stopping (Most Important in Practice):**
```
patience = 10
best_val_loss = infinity
counter = 0

for epoch in epochs:
    train()
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        save_best_model()
    else:
        counter += 1
        if counter >= patience:
            stop_training()
```

**Practical Guidelines:**
1. Always set max_epochs as upper bound
2. Use early stopping based on validation loss
3. Patience of 5-20 epochs is typical
4. Save best model, not last model

---

## Question 3
- [ ] Done

**How would you adapt gradient descent to handle a large amount of data that does not fit into memory?**

**Answer:**

When data exceeds memory, use mini-batch SGD with data streaming/loading. Key techniques: batch processing, data generators, gradient accumulation for effective larger batches.

**Approach 1: Mini-Batch Gradient Descent**
```python
# Load and process data in batches
for epoch in range(epochs):
    for batch in data_loader:  # loads one batch at a time
        X_batch, y_batch = batch
        gradient = compute_gradient(X_batch, y_batch)
        theta -= lr * gradient
```

**Approach 2: Data Generators (Python)**
```python
def data_generator(file_path, batch_size):
    while True:
        # Read batch_size rows from disk
        chunk = pd.read_csv(file_path, chunksize=batch_size)
        for batch in chunk:
            yield batch.values

# Use generator in training
for X_batch, y_batch in data_generator(path, 64):
    train_step(X_batch, y_batch)
```

**Approach 3: Gradient Accumulation**
```python
# Simulate larger batch with limited memory
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(batches):
    loss = model(batch) / accumulation_steps
    loss.backward()  # accumulate gradients
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # update only after accumulating
        optimizer.zero_grad()
```

**Tools for Large Data:**
- PyTorch DataLoader with num_workers > 0
- TensorFlow tf.data pipeline
- Dask for out-of-core computation
- Memory-mapped files (np.memmap)

---

## Question 4
- [ ] Done

**Discuss how you might use feature engineering to improve the performance of gradient descent in a model.**

**Answer:**

Feature engineering improves gradient descent by creating features that are: properly scaled, informative, and have better-behaved gradients. Good features make the loss surface smoother and easier to optimize.

**Key Feature Engineering for GD:**

| Technique | Why It Helps GD |
|-----------|-----------------|
| Scaling/Normalization | Equal treatment of all features, faster convergence |
| Log transform (skewed data) | Reduces outlier impact, smoother gradients |
| Polynomial features | Can fit complex patterns without deep networks |
| Binning (discretization) | Reduces noise, can help with outliers |
| One-hot encoding | Proper representation for categorical data |

**1. Feature Scaling (Critical for GD):**
```python
# Standardization
X_scaled = (X - X.mean()) / X.std()

# Min-Max
X_scaled = (X - X.min()) / (X.max() - X.min())
```

**2. Handling Skewed Features:**
```python
# Log transform for right-skewed data
X_log = np.log1p(X)  # log(1+x) handles zeros
```

**3. Creating Interaction Features:**
```python
# Polynomial features help linear models
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

**Impact on Gradient Descent:**
- Scaled features → circular contours → direct convergence path
- Unscaled features → elongated contours → zig-zag slow convergence

**Remember:** Tree-based models (Random Forest, XGBoost) don't need feature scaling, but gradient-based models always benefit from it.

---

## Question 5
- [ ] Done

**Present a strategy to choose the right optimizer for a given ML problem.**

**Answer:**

Start with AdamW as robust default. For CNN SOTA, use SGD + momentum. For Transformers, use AdamW + warmup. For RNNs, use Adam + gradient clipping.

| Architecture | Optimizer | Extra |
|--------------|-----------|-------|
| CNN | SGD + Momentum or AdamW | LR schedule |
| RNN/LSTM | Adam | Gradient Clipping |
| Transformer | AdamW | Warmup + Decay |

---

## Question 6
- [ ] Done

**Describe a scenario where gradient descent might fail to find the optimal solution and what alternatives could mitigate this.**

**Answer:**

Gradient descent fails on: (1) Non-convex functions - gets stuck in poor local minima, (2) Saddle points - gradient near 0 but not minimum. Mitigations: momentum, random restarts, learning rate schedules, or gradient-free methods for non-differentiable problems.

**Scenarios and Solutions:**

| Failure | Mitigation |
|---------|------------|
| Poor local minimum | Momentum, random restarts |
| Saddle point | SGD noise, momentum |
| Non-differentiable | Subgradient methods |

---

## Question 7
- [ ] Done

**Explain how you would use gradient descent to optimize hyperparameters in a machine learning model.**

**Answer:**

Gradient-based hyperparameter optimization computes hypergradient by differentiating validation loss w.r.t. hyperparameters through the training process. Complex to implement; works for continuous hyperparameters only (learning rate, regularization), not discrete (number of layers).

---
