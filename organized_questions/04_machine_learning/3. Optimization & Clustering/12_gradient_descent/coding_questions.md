# Gradient Descent Interview Questions - Coding Questions

## Question 1
- [ ] Done

**How would you implement early stopping in a gradient descent algorithm?**

**Answer:**

Early stopping monitors validation loss and stops training when it stops improving for a certain number of epochs (patience). This prevents overfitting.

**Pipeline:**
1. Track best validation loss
2. Count epochs without improvement
3. Stop if patience exceeded
4. Save best model

```python
import numpy as np

def train_with_early_stopping(X_train, y_train, X_val, y_val, 
                               lr=0.01, max_epochs=1000, patience=10):
    """
    Train with early stopping based on validation loss.
    
    Output: best_weights, training_history
    """
    n_features = X_train.shape[1]
    weights = np.random.randn(n_features) * 0.01
    
    best_val_loss = float('inf')
    best_weights = weights.copy()
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(max_epochs):
        # Compute gradient (MSE for linear regression)
        predictions = X_train @ weights
        errors = predictions - y_train
        gradient = (2 / len(y_train)) * (X_train.T @ errors)
        
        # Update weights
        weights = weights - lr * gradient
        
        # Compute losses
        train_loss = np.mean((X_train @ weights - y_train) ** 2)
        val_loss = np.mean((X_val @ weights - y_val) ** 2)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = weights.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return best_weights, history

# Usage
# weights, history = train_with_early_stopping(X_train, y_train, X_val, y_val)
```

---

## Question 2
- [ ] Done

**Write a Python implementation of basic gradient descent to find the minimum of a quadratic function.**

**Answer:**

Minimize f(x) = x^2 (minimum at x=0).

**Pipeline:**
1. Initialize x randomly
2. Compute gradient: df/dx = 2x
3. Update: x = x - lr * gradient
4. Repeat until convergence

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent_quadratic(start_x, lr=0.1, n_iterations=50):
    """
    Find minimum of f(x) = x^2 using gradient descent.
    
    Output: final_x, history of x values
    """
    x = start_x
    history = [x]
    
    for _ in range(n_iterations):
        # Gradient of x^2 is 2x
        gradient = 2 * x
        
        # Update
        x = x - lr * gradient
        history.append(x)
        
        # Check convergence
        if abs(gradient) < 1e-6:
            break
    
    return x, history

# Run
x_final, history = gradient_descent_quadratic(start_x=5.0, lr=0.1)
print(f"Minimum found at x = {x_final:.6f}")

# Plot convergence
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history)
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.title('x converging to minimum')

plt.subplot(1, 2, 2)
x_range = np.linspace(-6, 6, 100)
plt.plot(x_range, x_range**2, label='f(x) = x^2')
plt.scatter(history, [x**2 for x in history], c='red', s=20)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Path on cost surface')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## Question 3
- [ ] Done

**Implement batch gradient descent for linear regression from scratch using Python.**

**Answer:**

**Pipeline:**
1. Initialize weights to zeros/random
2. For each epoch: compute predictions, compute gradient over ALL data, update weights
3. Track loss history

```python
import numpy as np

def batch_gradient_descent_linear_regression(X, y, lr=0.01, n_epochs=1000):
    """
    Batch GD for linear regression: minimize MSE.
    
    Input: X (n_samples, n_features), y (n_samples,)
    Output: weights, loss_history
    """
    n_samples, n_features = X.shape
    
    # Add bias column
    X_bias = np.c_[np.ones(n_samples), X]  # shape: (n_samples, n_features+1)
    
    # Initialize weights
    weights = np.zeros(n_features + 1)
    
    loss_history = []
    
    for epoch in range(n_epochs):
        # Predictions (all samples at once - batch)
        predictions = X_bias @ weights
        
        # Errors
        errors = predictions - y
        
        # MSE loss
        loss = np.mean(errors ** 2)
        loss_history.append(loss)
        
        # Gradient (average over all samples)
        gradient = (2 / n_samples) * (X_bias.T @ errors)
        
        # Update weights
        weights = weights - lr * gradient
    
    return weights, loss_history

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.flatten() + np.random.randn(100) * 0.5

# Train
weights, losses = batch_gradient_descent_linear_regression(X, y, lr=0.1, n_epochs=100)
print(f"Learned: intercept={weights[0]:.3f}, slope={weights[1]:.3f}")
print(f"True: intercept=4, slope=3")
```

---

## Question 4
- [ ] Done

**Create a stochastic gradient descent algorithm in Python for optimizing a logistic regression model.**

**Answer:**

**Pipeline:**
1. Shuffle data each epoch
2. For each sample: compute gradient for that sample only, update weights
3. Use sigmoid and binary cross-entropy loss

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sgd_logistic_regression(X, y, lr=0.01, n_epochs=100):
    """
    SGD for logistic regression.
    
    Input: X (n_samples, n_features), y (n_samples,) binary labels
    Output: weights, loss_history
    """
    n_samples, n_features = X.shape
    
    # Add bias
    X_bias = np.c_[np.ones(n_samples), X]
    
    # Initialize weights
    weights = np.zeros(n_features + 1)
    
    loss_history = []
    
    for epoch in range(n_epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(n_samples)
        
        epoch_loss = 0
        
        for i in indices:
            # Single sample
            xi = X_bias[i]
            yi = y[i]
            
            # Prediction
            z = np.dot(xi, weights)
            pred = sigmoid(z)
            
            # Gradient for single sample
            gradient = (pred - yi) * xi
            
            # Update
            weights = weights - lr * gradient
            
            # Accumulate loss (for monitoring)
            eps = 1e-15
            epoch_loss += -yi * np.log(pred + eps) - (1 - yi) * np.log(1 - pred + eps)
        
        loss_history.append(epoch_loss / n_samples)
    
    return weights, loss_history

# Generate binary classification data
np.random.seed(42)
X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Train
weights, losses = sgd_logistic_regression(X, y, lr=0.1, n_epochs=50)
print(f"Final loss: {losses[-1]:.4f}")
```

---

## Question 5
- [ ] Done

**Simulate annealing of the learning rate in gradient descent and plot the convergence over time.**

**Answer:**

Learning rate annealing decreases LR over time for better fine-tuning.

```python
import numpy as np
import matplotlib.pyplot as plt

def gd_with_lr_annealing(X, y, initial_lr=0.5, decay_rate=0.01, n_epochs=200):
    """
    GD with exponential LR decay: lr(t) = initial_lr * exp(-decay_rate * t)
    
    Output: weights, loss_history, lr_history
    """
    n_samples, n_features = X.shape
    X_bias = np.c_[np.ones(n_samples), X]
    weights = np.zeros(n_features + 1)
    
    loss_history = []
    lr_history = []
    
    for epoch in range(n_epochs):
        # Annealed learning rate
        lr = initial_lr * np.exp(-decay_rate * epoch)
        lr_history.append(lr)
        
        # Forward pass
        predictions = X_bias @ weights
        errors = predictions - y
        loss = np.mean(errors ** 2)
        loss_history.append(loss)
        
        # Gradient and update
        gradient = (2 / n_samples) * (X_bias.T @ errors)
        weights = weights - lr * gradient
    
    return weights, loss_history, lr_history

# Generate data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.flatten() + np.random.randn(100) * 0.5

# Train with LR annealing
weights, losses, lrs = gd_with_lr_annealing(X, y, initial_lr=0.5, decay_rate=0.02)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(losses)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Convergence')

axes[1].plot(lrs)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Learning Rate')
axes[1].set_title('LR Annealing (Exponential Decay)')

plt.tight_layout()
plt.show()
```

---

## Question 6
- [ ] Done

**Design a Python function to compare the convergence speed of gradient descent with and without momentum.**

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt

def gd_without_momentum(X, y, lr=0.01, n_epochs=200):
    """Standard GD without momentum."""
    n_samples = X.shape[0]
    X_bias = np.c_[np.ones(n_samples), X]
    weights = np.zeros(X_bias.shape[1])
    losses = []
    
    for _ in range(n_epochs):
        pred = X_bias @ weights
        loss = np.mean((pred - y) ** 2)
        losses.append(loss)
        gradient = (2 / n_samples) * (X_bias.T @ (pred - y))
        weights = weights - lr * gradient
    
    return weights, losses

def gd_with_momentum(X, y, lr=0.01, momentum=0.9, n_epochs=200):
    """GD with momentum."""
    n_samples = X.shape[0]
    X_bias = np.c_[np.ones(n_samples), X]
    weights = np.zeros(X_bias.shape[1])
    velocity = np.zeros_like(weights)
    losses = []
    
    for _ in range(n_epochs):
        pred = X_bias @ weights
        loss = np.mean((pred - y) ** 2)
        losses.append(loss)
        gradient = (2 / n_samples) * (X_bias.T @ (pred - y))
        
        # Momentum update
        velocity = momentum * velocity + lr * gradient
        weights = weights - velocity
    
    return weights, losses

# Generate data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.flatten() + np.random.randn(100) * 0.5

# Compare
_, losses_no_mom = gd_without_momentum(X, y, lr=0.1, n_epochs=100)
_, losses_with_mom = gd_with_momentum(X, y, lr=0.1, momentum=0.9, n_epochs=100)

# Plot comparison
plt.figure(figsize=(10, 5))
plt.plot(losses_no_mom, label='Without Momentum')
plt.plot(losses_with_mom, label='With Momentum (0.9)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Convergence: With vs Without Momentum')
plt.legend()
plt.grid(True)
plt.show()

print(f"Final loss without momentum: {losses_no_mom[-1]:.6f}")
print(f"Final loss with momentum: {losses_with_mom[-1]:.6f}")
```

---

## Question 7
- [ ] Done

**Implement gradient descent with early stopping using Python.**

**Answer:**

(Same as Question 1 - see above for complete implementation)

Early stopping key points:
- Monitor validation loss
- Save best model
- Stop when val_loss doesn't improve for `patience` epochs

---

## Question 8
- [ ] Done

**Code a mini-batch gradient descent optimizer and test it on a small dataset.**

**Answer:**

```python
import numpy as np

def mini_batch_gd(X, y, batch_size=32, lr=0.01, n_epochs=100):
    """
    Mini-batch gradient descent for linear regression.
    
    Output: weights, loss_history
    """
    n_samples, n_features = X.shape
    X_bias = np.c_[np.ones(n_samples), X]
    weights = np.zeros(n_features + 1)
    losses = []
    
    for epoch in range(n_epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X_bias[indices]
        y_shuffled = y[indices]
        
        # Process mini-batches
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # Gradient on mini-batch
            pred = X_batch @ weights
            gradient = (2 / len(y_batch)) * (X_batch.T @ (pred - y_batch))
            weights = weights - lr * gradient
        
        # Track loss on full dataset
        full_loss = np.mean((X_bias @ weights - y) ** 2)
        losses.append(full_loss)
    
    return weights, losses

# Test
np.random.seed(42)
X = 2 * np.random.rand(200, 2)
y = 4 + 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(200) * 0.5

weights, losses = mini_batch_gd(X, y, batch_size=32, lr=0.1, n_epochs=50)
print(f"Learned weights: {weights}")
print(f"Final loss: {losses[-1]:.4f}")
```

---

## Question 9
- [ ] Done

**Write a Python function to check the gradients computed by a gradient descent algorithm.**

**Answer:**

Gradient checking compares analytical gradient with numerical approximation.

```python
import numpy as np

def numerical_gradient(f, x, epsilon=1e-7):
    """Compute numerical gradient using central difference."""
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)
    
    return grad

def gradient_check(analytical_grad_fn, loss_fn, x, epsilon=1e-7):
    """
    Compare analytical and numerical gradients.
    
    Input:
        analytical_grad_fn: function that computes gradient analytically
        loss_fn: loss function
        x: point to check gradient at
    
    Output: relative error (should be < 1e-5 for correct implementation)
    """
    analytical = analytical_grad_fn(x)
    numerical = numerical_gradient(loss_fn, x, epsilon)
    
    # Relative error
    diff = np.linalg.norm(analytical - numerical)
    norm_sum = np.linalg.norm(analytical) + np.linalg.norm(numerical)
    
    if norm_sum == 0:
        return 0.0
    
    relative_error = diff / norm_sum
    return relative_error, analytical, numerical

# Example: Check gradient for f(x) = x^2 (gradient should be 2x)
def loss_fn(x):
    return np.sum(x ** 2)

def analytical_grad(x):
    return 2 * x

# Test
x_test = np.array([3.0, 4.0, 5.0])
error, analytical, numerical = gradient_check(analytical_grad, loss_fn, x_test)

print(f"Analytical gradient: {analytical}")
print(f"Numerical gradient: {numerical}")
print(f"Relative error: {error:.2e}")
print(f"Gradient check: {'PASSED' if error < 1e-5 else 'FAILED'}")
```

---

## Question 10
- [ ] Done

**Experiment with different weight initializations and observe their impact on gradient descent optimization.**

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt

def train_with_init(X, y, init_method, lr=0.01, n_epochs=100):
    """Train and return loss history with specified initialization."""
    n_samples, n_features = X.shape
    X_bias = np.c_[np.ones(n_samples), X]
    
    # Different initialization methods
    if init_method == 'zeros':
        weights = np.zeros(n_features + 1)
    elif init_method == 'small_random':
        weights = np.random.randn(n_features + 1) * 0.01
    elif init_method == 'large_random':
        weights = np.random.randn(n_features + 1) * 10
    elif init_method == 'xavier':
        weights = np.random.randn(n_features + 1) * np.sqrt(2.0 / (n_features + 1))
    elif init_method == 'he':
        weights = np.random.randn(n_features + 1) * np.sqrt(2.0 / n_features)
    
    losses = []
    for _ in range(n_epochs):
        pred = X_bias @ weights
        loss = np.mean((pred - y) ** 2)
        losses.append(loss)
        gradient = (2 / n_samples) * (X_bias.T @ (pred - y))
        weights = weights - lr * gradient
    
    return losses

# Generate data
np.random.seed(42)
X = np.random.randn(100, 5)
y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(100) * 0.5

# Compare initializations
init_methods = ['zeros', 'small_random', 'large_random', 'xavier', 'he']

plt.figure(figsize=(10, 6))
for method in init_methods:
    np.random.seed(42)  # Same random seed for fair comparison
    losses = train_with_init(X, y, method, lr=0.01, n_epochs=200)
    plt.plot(losses, label=method)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Impact of Weight Initialization on Convergence')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()
```

---

## Question 11
- [ ] Done

**Implement and visualize the optimization path of the Adam optimizer vs. vanilla gradient descent.**

**Answer:**

```python
import numpy as np
import matplotlib.pyplot as plt

def vanilla_gd(grad_fn, x_init, lr=0.1, n_steps=50):
    """Vanilla gradient descent."""
    x = x_init.copy()
    path = [x.copy()]
    
    for _ in range(n_steps):
        grad = grad_fn(x)
        x = x - lr * grad
        path.append(x.copy())
    
    return np.array(path)

def adam_optimizer(grad_fn, x_init, lr=0.1, beta1=0.9, beta2=0.999, 
                   epsilon=1e-8, n_steps=50):
    """Adam optimizer."""
    x = x_init.copy()
    m = np.zeros_like(x)  # First moment
    v = np.zeros_like(x)  # Second moment
    path = [x.copy()]
    
    for t in range(1, n_steps + 1):
        grad = grad_fn(x)
        
        # Update moments
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update
        x = x - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        path.append(x.copy())
    
    return np.array(path)

# Rosenbrock function (challenging optimization surface)
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    dx0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dx1 = 200 * (x[1] - x[0]**2)
    return np.array([dx0, dx1])

# Run both optimizers
x_init = np.array([-1.0, 1.0])
path_gd = vanilla_gd(rosenbrock_grad, x_init, lr=0.001, n_steps=200)
path_adam = adam_optimizer(rosenbrock_grad, x_init, lr=0.1, n_steps=200)

# Visualize
x_range = np.linspace(-2, 2, 100)
y_range = np.linspace(-1, 3, 100)
X_grid, Y_grid = np.meshgrid(x_range, y_range)
Z = (1 - X_grid)**2 + 100 * (Y_grid - X_grid**2)**2

plt.figure(figsize=(10, 8))
plt.contour(X_grid, Y_grid, Z, levels=np.logspace(0, 3, 20), cmap='viridis')
plt.plot(path_gd[:, 0], path_gd[:, 1], 'r.-', label='Vanilla GD', markersize=3)
plt.plot(path_adam[:, 0], path_adam[:, 1], 'b.-', label='Adam', markersize=3)
plt.scatter([1], [1], c='green', s=100, marker='*', label='Minimum (1,1)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimization Path: Adam vs Vanilla GD on Rosenbrock Function')
plt.legend()
plt.colorbar(label='Loss')
plt.show()

print(f"GD final point: {path_gd[-1]}")
print(f"Adam final point: {path_adam[-1]}")
```

---
