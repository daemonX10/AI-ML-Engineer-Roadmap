# Gradient Descent Interview Questions - General Questions

## Question 1
- [ ] Done

**What challenges arise when using gradient descent on non-convex functions?**

**Answer:**

Non-convex functions have multiple local minima, saddle points, and plateaus. GD can get stuck in poor local minima (depending on initialization), slow down at saddle points (gradient near 0), and there is no guarantee of finding the global minimum.

**Key Challenges:**

| Challenge | Description | Impact |
|-----------|-------------|--------|
| Local minima | Multiple valleys | May converge to suboptimal solution |
| Saddle points | Flat regions | Training stalls |
| Plateaus | Large flat areas | Very slow progress |

**Mitigation Strategies:**
- Use momentum to carry through flat regions
- SGD noise helps escape shallow minima
- Multiple random restarts

---

## Question 2
- [ ] Done

**How can gradient clipping help in training deep learning models?**

**Answer:**

Gradient clipping caps the magnitude of gradients before the parameter update to prevent exploding gradients. When gradient norm exceeds a threshold, it scales down the gradient while preserving direction. Essential for RNNs/LSTMs.

**Two Types:**

| Type | Method |
|------|--------|
| Clip by Value | Clip each element to [-max, max] |
| Clip by Norm | Scale if norm exceeds threshold |

**When to Use:** RNNs/LSTMs, very deep networks, when loss becomes NaN

**Typical threshold values:** 1.0, 5.0, or 10.0

---

## Question 3
- [ ] Done

**How do you choose an appropriate learning rate?**

**Answer:**

Start with defaults (0.001 for Adam, 0.01-0.1 for SGD), use learning rate finder, monitor training and adjust based on loss behavior.

**Quick Guide:**

| Observation | Action |
|-------------|--------|
| Loss exploding/NaN | Reduce LR by 10x |
| Loss oscillates | Reduce LR by 2-5x |
| Loss decreases slowly | Increase LR by 2-5x |
| Loss plateaus | Use LR decay schedule |

---

## Question 4
- [ ] Done

**How do you avoid overfitting when using gradient descent for training models?**

**Answer:**

Use: (1) Early stopping - stop when val_loss increases, (2) L1/L2 Regularization, (3) Dropout, (4) Data augmentation, (5) Reduce model complexity.

**Key Insight:** Monitor gap between train and val loss. If train << val, overfitting.

---

## Question 5
- [ ] Done

**How do learning rate schedules improve gradient descent optimization?**

**Answer:**

LR schedules start high for fast initial progress, then decrease for fine-tuning near minimum. Common schedules: Step Decay, Cosine Annealing, Warmup + Decay, ReduceLROnPlateau.

---

## Question 6
- [ ] Done

**What metrics or visualizations can be used to monitor gradient descent progress?**

**Answer:**

Key metrics: training loss, validation loss, gradient norm, learning rate.

**Diagnosis Table:**

| Pattern | Diagnosis |
|---------|-----------|
| Train and val both high | Underfitting |
| Train low, val high | Overfitting |
| Loss oscillating | LR too high |
| Loss flat from start | LR too low |
| Loss NaN | LR too high |

---
