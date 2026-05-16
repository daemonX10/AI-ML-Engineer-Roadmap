# Cost Function Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Implement a Python function that calculates the Mean Squared Error between predicted and actual values.**

**Answer:**

**Pipeline:**
1. Input: y_true (actual), y_pred (predicted)
2. Calculate: difference → square → mean
3. Output: single MSE value

**Mathematical Formula:**
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

```python
import numpy as np

def mse(y_true, y_pred):
    """
    Calculate Mean Squared Error
    
    Steps:
    1. Find difference between actual and predicted
    2. Square each difference
    3. Take mean of all squared differences
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    differences = y_true - y_pred      # Step 1: differences
    squared = differences ** 2          # Step 2: square
    mean_squared = np.mean(squared)     # Step 3: mean
    
    return mean_squared

# Example usage
y_true = np.array([3, 5, 2, 7])
y_pred = np.array([2.5, 5.5, 2, 8])

print(f"MSE: {mse(y_true, y_pred)}")  # Output: 0.375

# Verify with sklearn
from sklearn.metrics import mean_squared_error
print(f"Sklearn MSE: {mean_squared_error(y_true, y_pred)}")  # Same: 0.375
```

**Key Points:**
- MSE penalizes larger errors more (due to squaring)
- Always non-negative
- Units are squared (use RMSE for original units)

---

## Question 2
- [ ] Done

**Write a Python code snippet to compute the Cross-Entropy loss given predicted probabilities and actual labels.**

**Answer:**

**Pipeline:**
1. Input: y_true (0/1 labels), y_pred (probabilities)
2. Clip predictions to avoid log(0)
3. Apply cross-entropy formula
4. Output: average loss

**Mathematical Formula:**
$$BCE = -\frac{1}{n}\sum[y \cdot \log(p) + (1-y) \cdot \log(1-p)]$$

```python
import numpy as np

def binary_cross_entropy(y_true, y_pred):
    """
    Calculate Binary Cross-Entropy Loss
    
    Steps:
    1. Clip predictions to avoid log(0)
    2. Calculate: y*log(p) + (1-y)*log(1-p)
    3. Take negative mean
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Step 1: Clip to prevent log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Step 2 & 3: Cross-entropy formula
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return np.mean(loss)

# Example usage
y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.1, 0.8, 0.7])

print(f"BCE: {binary_cross_entropy(y_true, y_pred):.4f}")

# For multi-class (categorical cross-entropy)
def categorical_cross_entropy(y_true, y_pred):
    """
    y_true: class indices (e.g., [0, 1, 2])
    y_pred: probability matrix (n_samples x n_classes)
    """
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    n_samples = len(y_true)
    
    # Get probability of true class for each sample
    correct_probs = y_pred[np.arange(n_samples), y_true]
    
    return -np.mean(np.log(correct_probs))

# Example
y_true_multi = np.array([0, 1, 2])
y_pred_multi = np.array([
    [0.8, 0.1, 0.1],  # Class 0
    [0.2, 0.7, 0.1],  # Class 1
    [0.1, 0.2, 0.7]   # Class 2
])
print(f"CCE: {categorical_cross_entropy(y_true_multi, y_pred_multi):.4f}")
```

**Key Points:**
- Always clip predictions to avoid log(0) = -infinity
- Penalizes confident wrong predictions heavily
- Use for classification problems

---

## Question 3
- [ ] Done

**Implement a gradient descent algorithm in Python to minimize a simple quadratic cost function.**

**Answer:**

**Concept:** Iteratively move in the direction of steepest descent (negative gradient).

**Mathematical Formula:**
$$x_{t+1} = x_t - \alpha \cdot \nabla f(x_t)$$

```python
import numpy as np

def quadratic_cost(x):
    """Cost function: f(x) = x^2 + 4x + 4 (minimum at x = -2)"""
    return x**2 + 4*x + 4

def gradient(x):
    """Gradient: df/dx = 2x + 4"""
    return 2*x + 4

def gradient_descent(start, lr=0.1, iterations=50):
    """
    Gradient Descent Algorithm
    
    Steps:
    1. Start at initial point
    2. Compute gradient at current point
    3. Update: x = x - learning_rate * gradient
    4. Repeat until convergence
    """
    x = start
    history = [x]
    
    for i in range(iterations):
        grad = gradient(x)
        x = x - lr * grad  # Update step
        history.append(x)
        
        # Optional: early stopping
        if abs(grad) < 1e-6:
            break
    
    return x, history

# Example
start_point = 5.0
optimal_x, history = gradient_descent(start_point, lr=0.1, iterations=50)

print(f"Starting point: {start_point}")
print(f"Optimal x: {optimal_x:.4f}")  # Should be close to -2
print(f"Minimum cost: {quadratic_cost(optimal_x):.4f}")  # Should be close to 0
```

**For Multi-Dimensional:**
```python
def gradient_descent_nd(cost_fn, grad_fn, start, lr=0.01, iterations=100):
    """N-dimensional gradient descent."""
    x = np.array(start, dtype=float)
    
    for _ in range(iterations):
        grad = grad_fn(x)
        x = x - lr * grad
    
    return x
```

**Key Points:**
- Learning rate too high → diverges
- Learning rate too low → slow convergence
- Converges to local minimum (global for convex functions)

---

## Question 4
- [ ] Done

**Create a Python simulation that compares the convergence speed of batch and stochastic gradient descent.**

**Answer:**

**Concept:**
- **Batch GD:** Uses all data for each update (slow but stable)
- **SGD:** Uses one sample per update (fast but noisy)
- **Mini-batch:** Compromise between both

```python
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)  # 100 samples, 2 features
y = 3*X[:, 0] + 2*X[:, 1] + np.random.randn(100)*0.1

def compute_gradient(X, y, w):
    """Compute MSE gradient."""
    predictions = X @ w
    error = predictions - y
    gradient = (2/len(y)) * X.T @ error
    return gradient

def compute_loss(X, y, w):
    """Compute MSE loss."""
    predictions = X @ w
    return np.mean((predictions - y)**2)

def batch_gradient_descent(X, y, lr=0.01, iterations=100):
    """Batch GD: Use ALL data for each update."""
    w = np.zeros(X.shape[1])
    losses = []
    
    for _ in range(iterations):
        gradient = compute_gradient(X, y, w)  # Full batch
        w = w - lr * gradient
        losses.append(compute_loss(X, y, w))
    
    return w, losses

def stochastic_gradient_descent(X, y, lr=0.01, epochs=10):
    """SGD: Use ONE sample for each update."""
    w = np.zeros(X.shape[1])
    losses = []
    n = len(y)
    
    for epoch in range(epochs):
        indices = np.random.permutation(n)  # Shuffle each epoch
        for i in indices:
            xi = X[i:i+1]  # Single sample
            yi = y[i:i+1]
            gradient = compute_gradient(xi, yi, w)
            w = w - lr * gradient
        losses.append(compute_loss(X, y, w))
    
    return w, losses

def mini_batch_gradient_descent(X, y, lr=0.01, epochs=10, batch_size=16):
    """Mini-batch GD: Use small batches."""
    w = np.zeros(X.shape[1])
    losses = []
    n = len(y)
    
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            gradient = compute_gradient(X_batch, y_batch, w)
            w = w - lr * gradient
        losses.append(compute_loss(X, y, w))
    
    return w, losses

# Compare methods
w_batch, losses_batch = batch_gradient_descent(X, y, lr=0.1, iterations=100)
w_sgd, losses_sgd = stochastic_gradient_descent(X, y, lr=0.01, epochs=10)
w_mini, losses_mini = mini_batch_gradient_descent(X, y, lr=0.05, epochs=10)

print("Final Losses:")
print(f"  Batch GD:      {losses_batch[-1]:.6f}")
print(f"  SGD:           {losses_sgd[-1]:.6f}")
print(f"  Mini-batch:    {losses_mini[-1]:.6f}")
```

**Comparison:**

| Method | Per-Update Cost | Convergence | Noise |
|--------|-----------------|-------------|-------|
| Batch GD | O(n) | Smooth | None |
| SGD | O(1) | Noisy | High |
| Mini-batch | O(batch_size) | Balanced | Medium |

**Key Points:**
- SGD is faster per iteration but noisier
- Batch GD is smoother but slower per iteration
- Mini-batch is most commonly used in practice

---

## Question 5
- [ ] Done

**Build a Python class that implements an adaptive learning rate algorithm, like Adam or AdaGrad, from scratch.**

**Answer:**

**Concept:** Adapt learning rate per parameter based on gradient history.

**Adam Algorithm Steps:**
1. Compute gradient
2. Update first moment (momentum): m = β₁m + (1-β₁)g
3. Update second moment (adaptive LR): v = β₂v + (1-β₂)g²
4. Bias correction
5. Update: θ = θ - α * m̂ / (√v̂ + ε)

```python
import numpy as np

class Adam:
    """Adam optimizer from scratch."""
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Timestep
    
    def update(self, params, grads):
        """Update parameters using Adam."""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        
        # Update moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params

class AdaGrad:
    """AdaGrad optimizer from scratch."""
    
    def __init__(self, lr=0.01, epsilon=1e-8):
        self.lr = lr
        self.epsilon = epsilon
        self.G = None  # Accumulated squared gradients
    
    def update(self, params, grads):
        """Update parameters using AdaGrad."""
        if self.G is None:
            self.G = np.zeros_like(params)
        
        # Accumulate squared gradients
        self.G += grads ** 2
        
        # Update with adaptive learning rate
        params = params - self.lr * grads / (np.sqrt(self.G) + self.epsilon)
        return params

class RMSprop:
    """RMSprop optimizer from scratch."""
    
    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.v = None
    
    def update(self, params, grads):
        """Update parameters using RMSprop."""
        if self.v is None:
            self.v = np.zeros_like(params)
        
        # Exponential moving average of squared gradients
        self.v = self.beta * self.v + (1 - self.beta) * (grads ** 2)
        
        # Update
        params = params - self.lr * grads / (np.sqrt(self.v) + self.epsilon)
        return params

# Example usage
def demo_adam():
    # Simple optimization: minimize f(x) = x^2
    x = np.array([5.0])
    optimizer = Adam(lr=0.1)
    
    for i in range(50):
        grad = 2 * x  # Gradient of x^2
        x = optimizer.update(x, grad)
        if i % 10 == 0:
            print(f"Step {i}: x = {x[0]:.4f}, f(x) = {x[0]**2:.6f}")

demo_adam()
```

**Key Points:**
- Adam: Combines momentum + adaptive LR + bias correction
- AdaGrad: Good for sparse features, but LR decays too fast
- RMSprop: Fixes AdaGrad's aggressive LR decay

---

## Question 6
- [ ] Done

**Write a Python function that minimizes a cost function using simulated annealing.**

**Answer:**

**Concept:** Probabilistic optimization that can escape local minima by occasionally accepting worse solutions.

**Algorithm:**
1. Start with random solution
2. Generate neighbor solution
3. Accept if better, or with probability exp(-ΔE/T) if worse
4. Decrease temperature gradually

```python
import numpy as np

def simulated_annealing(cost_fn, x0, temp=100, cooling=0.95, 
                        min_temp=1e-6, max_iter=1000):
    """
    Simulated Annealing Optimization
    
    Args:
        cost_fn: Function to minimize
        x0: Initial solution
        temp: Initial temperature
        cooling: Cooling rate (0 < cooling < 1)
        min_temp: Minimum temperature (stopping criterion)
        max_iter: Maximum iterations per temperature
    
    Returns:
        Best solution found
    """
    current = np.array(x0, dtype=float)
    current_cost = cost_fn(current)
    
    best = current.copy()
    best_cost = current_cost
    
    while temp > min_temp:
        for _ in range(max_iter):
            # Generate neighbor (random perturbation)
            neighbor = current + np.random.normal(0, temp * 0.1, len(current))
            neighbor_cost = cost_fn(neighbor)
            
            # Acceptance probability
            delta = neighbor_cost - current_cost
            if delta < 0:  # Better solution
                accept = True
            else:  # Worse solution - accept with probability
                accept = np.random.random() < np.exp(-delta / temp)
            
            if accept:
                current = neighbor
                current_cost = neighbor_cost
                
                # Update best
                if current_cost < best_cost:
                    best = current.copy()
                    best_cost = current_cost
        
        # Cool down
        temp *= cooling
    
    return best, best_cost

# Example: Minimize Rastrigin function (many local minima)
def rastrigin(x):
    """Rastrigin function: many local minima, global min at origin."""
    return 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

# Run optimization
x0 = [3.0, 4.0]  # Start away from optimum
best_x, best_cost = simulated_annealing(rastrigin, x0, temp=100, cooling=0.95)

print(f"Initial: x = {x0}, cost = {rastrigin(x0):.4f}")
print(f"Found: x = {best_x}, cost = {best_cost:.4f}")
print(f"Global optimum is at [0, 0] with cost = 0")
```

**Key Points:**
- Temperature controls exploration vs exploitation
- High temp → accepts worse solutions (exploration)
- Low temp → only accepts better solutions (exploitation)
- Good for non-convex problems with many local minima

---

## Question 7
- [ ] Done

**Implement a basic version of the RMSprop optimization algorithm in Python.**

**Answer:**

**Concept:** Adapts learning rate using exponential moving average of squared gradients.

**Formula:**
$$v_t = \beta \cdot v_{t-1} + (1-\beta) \cdot g_t^2$$
$$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{v_t} + \epsilon} \cdot g_t$$

```python
import numpy as np

def rmsprop(gradient_fn, x0, lr=0.01, beta=0.9, epsilon=1e-8, iterations=100):
    """
    RMSprop Optimizer
    
    Args:
        gradient_fn: Function that returns gradient at x
        x0: Initial parameters
        lr: Learning rate
        beta: Decay rate for moving average
        epsilon: Small constant for numerical stability
        iterations: Number of iterations
    
    Returns:
        Optimized parameters
    """
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)  # Moving average of squared gradients
    history = [x.copy()]
    
    for _ in range(iterations):
        grad = gradient_fn(x)
        
        # Update moving average
        v = beta * v + (1 - beta) * (grad ** 2)
        
        # Adaptive learning rate update
        x = x - lr * grad / (np.sqrt(v) + epsilon)
        
        history.append(x.copy())
    
    return x, history

# Example: Minimize f(x,y) = x^2 + 10*y^2 (ill-conditioned)
def cost_fn(x):
    return x[0]**2 + 10 * x[1]**2

def gradient_fn(x):
    return np.array([2*x[0], 20*x[1]])

# Optimize
x0 = [5.0, 5.0]
optimal_x, history = rmsprop(gradient_fn, x0, lr=0.1, beta=0.9, iterations=100)

print(f"Initial: {x0}, cost = {cost_fn(x0):.4f}")
print(f"Final: {optimal_x}, cost = {cost_fn(optimal_x):.6f}")
print(f"Optimal should be [0, 0]")
```

**RMSprop Class Version:**
```python
class RMSprop:
    def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v = None
    
    def step(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        
        self.v = self.beta * self.v + (1 - self.beta) * grads**2
        params -= self.lr * grads / (np.sqrt(self.v) + self.eps)
        return params
```

**Key Points:**
- Fixes AdaGrad's diminishing learning rate problem
- Uses exponential decay instead of accumulation
- Works well for RNNs and non-stationary problems
- Default: β=0.9, lr=0.001

---
