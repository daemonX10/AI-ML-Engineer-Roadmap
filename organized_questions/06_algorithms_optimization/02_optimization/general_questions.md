# Optimization Interview Questions - General Questions

## Question 1
- [ ] Done

**Why is convexity important in optimization problems?**

**Answer:**

Convexity guarantees that **any local minimum is the global minimum**, ensuring gradient descent will find the optimal solution regardless of initialization.

**Key Benefits of Convex Functions:**
- Single minimum (no local minima traps)
- Gradient = 0 implies optimality
- Convergence guarantees for gradient descent
- Efficient algorithms exist

**Mathematical Definition:**
Function f is convex if:
$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

for all $\lambda \in [0,1]$

**Examples:**
| Convex | Non-Convex |
|--------|------------|
| MSE for linear regression | Neural network loss |
| Logistic regression loss | Any multi-layer model |
| SVM hinge loss | GANs |

**Practical Implication:**
- Convex: Use any gradient method, guaranteed optimal
- Non-convex: Need careful initialization, may need multiple runs

---

## Question 2
- [ ] Done

**Distinguish between local minima and global minima.**

**Answer:**

**Global minimum** is the absolute lowest value of the function. **Local minimum** is the lowest value in a neighborhood but may not be the global lowest.

**Definitions:**
- **Global**: $f(x^*) \leq f(x)$ for all x in domain
- **Local**: $f(x^*) \leq f(x)$ for x near $x^*$

**Visual Representation:**
```
f(x)
  |   /\      /\
  |  /  \    /  \
  | /    \  /    \_____  <- global minimum
  |       \/
  |       Local min
  +-------------------------> x
```

**Impact on Optimization:**
| Aspect | Local Minima | Global Minima |
|--------|-------------|---------------|
| Gradient | Zero | Zero |
| Optimality | Not guaranteed | Optimal |
| Stuck | GD can get stuck | Where we want to be |

**In Deep Learning:**
- Many local minima exist
- Research suggests: Good local minima are often "good enough"
- Saddle points are more problematic than local minima
- Modern optimizers (momentum, Adam) help escape shallow local minima

---

## Question 3
- [ ] Done

**When would you choose to use a conjugate gradient method?**

**Answer:**

Use conjugate gradient for **large-scale optimization where storing the full Hessian is impractical** but you want faster convergence than basic gradient descent.

**When to Use:**
- Large sparse linear systems (Ax = b)
- Second-order optimization without full Hessian
- Quadratic objective functions
- Memory-constrained environments

**How It Works:**
- Search directions are conjugate (orthogonal w.r.t. A)
- No repeated search in same direction
- Converges in n steps for n-dimensional quadratic

**Algorithm:**
1. Choose conjugate direction (orthogonal to previous)
2. Line search along that direction
3. Repeat with new conjugate direction

**Comparison:**
| Method | Storage | Convergence | Per-iteration |
|--------|---------|-------------|---------------|
| GD | O(n) | Slow | Cheap |
| Newton | O(n²) | Fast | Expensive |
| **CG** | O(n) | Medium | Medium |

**Practical Applications:**
- Solving large linear systems in physics simulations
- Optimizing convex quadratic functions
- Second-order approximations (Hessian-free optimization)

**Key Point:** CG provides second-order-like convergence with first-order memory requirements.

---

## Question 4
- [ ] Done

**What factors influence the convergence rate of an optimization algorithm?**

**Answer:**

Convergence rate depends on **function properties, algorithm choice, hyperparameters, and initialization**.

**Key Factors:**

| Factor | Impact on Convergence |
|--------|----------------------|
| **Condition number** | Higher = slower (elongated contours) |
| **Learning rate** | Too high = diverge, too low = slow |
| **Convexity** | Convex = guaranteed, non-convex = may get stuck |
| **Momentum** | Accelerates in consistent directions |
| **Batch size** | Affects gradient noise and update frequency |

**Condition Number:**
$$\kappa = \frac{\lambda_{max}}{\lambda_{min}}$$
- High κ → elliptical contours → slow convergence
- Preconditioning or adaptive methods help

**Algorithm Comparison:**
| Algorithm | Convergence Rate (Convex) |
|-----------|--------------------------|
| GD | $O(1/k)$ - sublinear |
| GD + Momentum | $O(1/k^2)$ - accelerated |
| Newton | Quadratic near optimum |
| Adam | Adaptive, robust |

**Practical Considerations:**
- Feature scaling improves condition number
- Learning rate scheduling helps convergence
- Early stopping balances convergence vs overfitting

---

## Question 5
- [ ] Done

**How do you approach selecting an appropriate optimization algorithm for a given problem?**

**Answer:**

Selection depends on **problem size, function properties, available compute, and desired trade-offs**.

**Decision Framework:**

| If you have... | Use |
|---------------|-----|
| Small dataset, convex | L-BFGS, exact methods |
| Large dataset, DL | Adam (default) |
| Very large model | AdamW + LR scheduler |
| Need best generalization | SGD + momentum + tuning |
| Sparse features | AdaGrad |
| RNNs | RMSprop or Adam |
| Computer vision SOTA | SGD + momentum + scheduler |

**Selection Process:**
1. **Start with Adam** - robust default for deep learning
2. **If overfitting** → try SGD + momentum
3. **If slow convergence** → tune learning rate, try scheduler
4. **If memory issues** → gradient accumulation, 8-bit Adam
5. **For SOTA** → match what works in literature

**Problem Characteristics:**

| Characteristic | Recommendation |
|---------------|----------------|
| Convex, smooth | Gradient descent variants |
| Non-convex, DL | Adam, SGD+momentum |
| Discrete/combinatorial | Genetic algorithms, SA |
| Constrained | Projected gradient, barrier methods |

**Practical Tips:**
- Check what works in papers for similar problems
- Adam is safe default, SGD often gives better final performance with tuning
- Use learning rate finder before training

---

## Question 6
- [ ] Done

**What methods can be used to tune hyperparameters effectively?**

**Answer:**

Hyperparameter tuning methods range from **manual search to sophisticated Bayesian optimization**, trading off simplicity against efficiency.

**Methods Comparison:**

| Method | How It Works | Pros | Cons |
|--------|-------------|------|------|
| **Grid Search** | Try all combinations | Exhaustive | Exponential cost |
| **Random Search** | Random samples | Often better than grid | No learning |
| **Bayesian Opt** | Model the objective | Sample efficient | Complex setup |
| **Hyperband** | Early stopping of bad configs | Fast | Needs many epochs |
| **Population-based** | Evolve hyperparams | Adapts during training | Resource intensive |

**Practical Recommendations:**

1. **Start Simple**: Random search beats grid search
2. **Use Ranges Wisely**: Log scale for LR (0.001, 0.01, 0.1)
3. **Early Stopping**: Don't fully train bad configurations
4. **Most Important HPs**: Learning rate > batch size > architecture

**Key Hyperparameters by Priority:**
1. Learning rate (most critical)
2. Number of layers/units
3. Regularization strength
4. Batch size
5. Optimizer-specific (β1, β2 for Adam)

**Tools:**
- Optuna, Ray Tune: Bayesian optimization
- Weights & Biases: Experiment tracking
- Keras Tuner: Easy integration

**Key Insight:** Random search with log-uniform LR distribution often matches more complex methods.

---

## Question 7
- [ ] Done

**Outline a strategy for optimizing models in a distributed computing environment.**

**Answer:**

Distributed optimization requires **synchronization strategy, communication efficiency, and proper scaling of hyperparameters**.

**Strategy Framework:**

**Step 1: Choose Parallelism Type**
| Type | Use Case | How |
|------|----------|-----|
| Data Parallel | Large dataset | Same model, different data |
| Model Parallel | Model too big | Split model across GPUs |
| Pipeline Parallel | Very deep models | Sequential layer groups |

**Step 2: Synchronization Strategy**
- **Synchronous**: All workers wait, average gradients (stable)
- **Asynchronous**: Workers update independently (faster, less stable)

**Step 3: Scale Hyperparameters**
- **Linear scaling**: batch × N → LR × N
- **Warmup**: Start small LR, ramp up
- **Gradient accumulation**: Simulate larger batch

**Step 4: Communication Efficiency**
- All-reduce for gradient averaging
- Gradient compression (top-k, quantization)
- Overlap communication with computation

**Implementation:**
```python
# PyTorch DistributedDataParallel
model = DistributedDataParallel(model)
# Automatically handles gradient sync

# Scale learning rate
effective_batch = batch_size * num_gpus
lr = base_lr * (effective_batch / base_batch)
```

**Challenges & Solutions:**
| Challenge | Solution |
|-----------|----------|
| Communication overhead | Gradient compression |
| Stale gradients (async) | Use sync SGD for stability |
| Large batch generalization | Warmup + LAMB optimizer |

---

