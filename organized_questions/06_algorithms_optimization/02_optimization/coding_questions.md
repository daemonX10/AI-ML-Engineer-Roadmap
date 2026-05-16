# Optimization Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Implement basic Gradient Descent to minimize a simple quadratic function.**

**Answer:**

```python
import numpy as np

def gradient_descent(f, grad_f, x_init, lr=0.1, n_iters=100):
    """
    Basic gradient descent optimizer
    f: function to minimize
    grad_f: gradient of f
    """
    x = x_init
    history = [x]
    
    for _ in range(n_iters):
        gradient = grad_f(x)
        x = x - lr * gradient
        history.append(x)
    
    return x, history

# Example: Minimize f(x) = x^2
f = lambda x: x ** 2
grad_f = lambda x: 2 * x

x_min, history = gradient_descent(f, grad_f, x_init=5.0, lr=0.1)
print(f"Minimum at x = {x_min:.4f}")  # Should approach 0
print(f"f(x_min) = {f(x_min):.6f}")
```

**Key Points:**
- Update rule: $x = x - \alpha \cdot \nabla f(x)$
- Learning rate controls step size
- Converges to local minimum for convex functions

---

## Question 2
- [ ] Done

**Write a Python function to perform SGD on a sample dataset.**

**Answer:**

```python
import numpy as np

def sgd_linear_regression(X, y, lr=0.01, epochs=100):
    """
    Stochastic Gradient Descent for linear regression
    Updates weights using one sample at a time
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    
    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(n_samples)
        
        for i in indices:
            # Predict for single sample
            y_pred = np.dot(X[i], weights) + bias
            error = y_pred - y[i]
            
            # Update using single sample gradient
            weights -= lr * error * X[i]
            bias -= lr * error
    
    return weights, bias

# Example usage
np.random.seed(42)
X = np.random.randn(100, 3)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1

weights, bias = sgd_linear_regression(X, y, lr=0.01, epochs=50)
print(f"Learned weights: {weights}")  # Should be close to [2, 3, -1]
```

**Key Difference from Batch GD:**
- Updates per sample (noisy but fast)
- Shuffling prevents learning order bias

---

## Question 3
- [ ] Done

**Code a simulation in Python demonstrating the effects of different learning rates on convergence.**

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt

def gd_with_history(grad_f, x_init, lr, n_iters=50):
    """Track optimization path for different learning rates"""
    x = x_init
    history = [x]
    for _ in range(n_iters):
        x = x - lr * grad_f(x)
        history.append(x)
    return history

# Function: f(x) = x^2, gradient = 2x
grad_f = lambda x: 2 * x

# Test different learning rates
learning_rates = [0.01, 0.1, 0.5, 0.9, 1.1]
x_init = 5.0

plt.figure(figsize=(10, 6))
for lr in learning_rates:
    history = gd_with_history(grad_f, x_init, lr)
    plt.plot(history, label=f'lr={lr}')

plt.xlabel('Iteration')
plt.ylabel('x value')
plt.title('Effect of Learning Rate on Convergence')
plt.legend()
plt.axhline(y=0, color='r', linestyle='--', label='Optimum')
plt.show()
```

**Observations:**
| Learning Rate | Behavior |
|--------------|----------|
| 0.01 | Slow convergence |
| 0.1 | Good convergence |
| 0.5 | Fast convergence |
| 0.9 | Oscillates but converges |
| 1.1 | Diverges! |

---

## Question 4
- [ ] Done

**Implement the Momentum technique in a Gradient Descent optimizer.**

**Answer:**

```python
import numpy as np

def gd_with_momentum(grad_f, x_init, lr=0.1, momentum=0.9, n_iters=100):
    """
    Gradient Descent with Momentum
    velocity accumulates past gradients
    """
    x = x_init
    velocity = 0
    history = [x]
    
    for _ in range(n_iters):
        gradient = grad_f(x)
        
        # Update velocity (accumulate momentum)
        velocity = momentum * velocity - lr * gradient
        
        # Update parameters
        x = x + velocity
        history.append(x)
    
    return x, history

# Example: Minimize f(x) = x^2
grad_f = lambda x: 2 * x

# Compare with and without momentum
x_no_mom, hist_no_mom = gd_with_momentum(grad_f, 5.0, lr=0.1, momentum=0.0)
x_with_mom, hist_with_mom = gd_with_momentum(grad_f, 5.0, lr=0.1, momentum=0.9)

print(f"Without momentum: converged in ~{len([h for h in hist_no_mom if abs(h) > 0.01])} iters")
print(f"With momentum: converged in ~{len([h for h in hist_with_mom if abs(h) > 0.01])} iters")
```

**Key Insight:**
- Momentum accelerates through flat regions
- Dampens oscillations in steep directions
- Update: $v_t = \beta v_{t-1} - \alpha \nabla f$, then $x = x + v_t$

---

## Question 5
- [ ] Done

**Create a regularization function in Python that penalizes large weights in a linear regression model.**

**Answer:**

```python
import numpy as np

def compute_loss_with_regularization(X, y, weights, lambda_reg=0.1, reg_type='l2'):
    """
    Compute MSE loss with L1 or L2 regularization penalty
    """
    # Predictions and MSE loss
    y_pred = X @ weights
    mse_loss = np.mean((y - y_pred) ** 2)
    
    # Regularization penalty
    if reg_type == 'l2':
        penalty = lambda_reg * np.sum(weights ** 2)
    elif reg_type == 'l1':
        penalty = lambda_reg * np.sum(np.abs(weights))
    else:
        penalty = 0
    
    total_loss = mse_loss + penalty
    return total_loss, mse_loss, penalty

# Example
np.random.seed(42)
X = np.random.randn(50, 5)
y = X @ np.array([1, 2, 0, 0, 3]) + np.random.randn(50) * 0.1

# Compare different weight sizes
small_weights = np.array([1, 2, 0, 0, 3])
large_weights = np.array([10, 20, 5, 5, 30])

for w, name in [(small_weights, "Small"), (large_weights, "Large")]:
    total, mse, penalty = compute_loss_with_regularization(X, y, w, lambda_reg=0.1)
    print(f"{name} weights: MSE={mse:.2f}, Penalty={penalty:.2f}, Total={total:.2f}")
```

**Output shows:** Large weights get penalized more, encouraging smaller weights.

---

## Question 6
- [ ] Done

**Develop a Python script that uses the Adam optimizer from a library like TensorFlow or PyTorch.**

**Answer:**

```python
import torch
import torch.nn as nn

# Simple linear regression with Adam
X = torch.randn(100, 3)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + torch.randn(100) * 0.1

# Model: single linear layer
model = nn.Linear(3, 1)

# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(100):
    # Forward pass
    y_pred = model(X).squeeze()
    loss = loss_fn(y_pred, y)
    
    # Backward pass
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights using Adam
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Check learned weights (should be close to [2, 3, -1])
print(f"Learned weights: {model.weight.data.squeeze().tolist()}")
```

**Adam Key Parameters:**
```python
torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8)
# betas[0] = momentum decay
# betas[1] = RMSprop decay
```

---

## Question 7
- [ ] Done

**Write a function that showcases the difference between L1 and L2 regularization on a small dataset.**

**Answer:**

```python
from sklearn.linear_model import Lasso, Ridge
from sklearn.datasets import make_regression
import numpy as np

# Create dataset with some irrelevant features
X, y = make_regression(n_samples=100, n_features=10, n_informative=3, 
                       noise=10, random_state=42)

# L1 Regularization (Lasso) - produces sparse weights
lasso = Lasso(alpha=1.0)
lasso.fit(X, y)

# L2 Regularization (Ridge) - shrinks all weights
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

# Compare coefficients
print("Feature Coefficients Comparison:")
print("-" * 40)
print(f"{'Feature':<10} {'Lasso (L1)':<15} {'Ridge (L2)':<15}")
print("-" * 40)

for i in range(10):
    print(f"  {i:<8} {lasso.coef_[i]:>12.2f} {ridge.coef_[i]:>12.2f}")

# Count zero coefficients
print(f"\nZero coefficients - Lasso: {np.sum(np.abs(lasso.coef_) < 0.01)}")
print(f"Zero coefficients - Ridge: {np.sum(np.abs(ridge.coef_) < 0.01)}")
```

**Key Observations:**
| L1 (Lasso) | L2 (Ridge) |
|------------|------------|
| Sets some weights to exactly 0 | Shrinks all weights, none exactly 0 |
| Feature selection | No feature selection |
| Sparse solution | Dense solution |

---

