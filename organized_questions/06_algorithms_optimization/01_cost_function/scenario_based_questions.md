# Cost Function Interview Questions - Scenario-Based Questions

## Question 1
- [ ] Done

**Discuss the role of the Huber loss and where it is preferable over MSE.**

**Answer:**

**Scenario:** You're building a house price prediction model and notice some luxury mansions ($10M+) in your dataset that are causing poor predictions for regular homes.

**Problem with MSE:**
- MSE squares errors: $(10M - 500K)^2 = 90$ trillion
- Single outlier dominates the entire loss
- Model fits outliers instead of majority

**Huber Loss Solution:**

$$L_\delta = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

| Error Size | Huber Behavior | MSE Behavior |
|------------|---------------|--------------|
| Small (≤δ) | Quadratic (like MSE) | Quadratic |
| Large (>δ) | Linear (like MAE) | Quadratic (explosive) |

**When to Use Huber over MSE:**
- Dataset has outliers but you can't remove them
- Sensor data with occasional measurement errors
- Financial data with extreme values
- Any fat-tailed error distribution

**How to Choose δ:**
- Look at error distribution
- Set δ at ~95th percentile of expected errors
- Cross-validate different values

**Logic:** Huber says "small mistakes deserve quadratic penalty for smooth optimization, but large mistakes just get linear penalty to avoid outlier domination."

---

## Question 2
- [ ] Done

**Discuss the trade-off between bias and variance in cost function optimization.**

**Answer:**

**Scenario:** Your model has low training error but high test error. How does cost function choice affect this?

**The Trade-off:**
- **Bias**: Error from wrong assumptions (underfitting)
- **Variance**: Error from sensitivity to training data (overfitting)
- **Total Error = Bias² + Variance + Irreducible Noise**

**How Cost Function Affects Bias-Variance:**

| Cost Function Choice | Effect |
|---------------------|--------|
| MSE alone | Low bias, high variance (can overfit) |
| MSE + L2 regularization | Slightly higher bias, lower variance |
| MSE + L1 regularization | Feature selection, sparse solution |
| Strong regularization | Higher bias, low variance (underfit) |

**Regularized Cost Function:**
$$J(\theta) = \underbrace{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}_{\text{Fit training data}} + \underbrace{\lambda||\theta||^2}_{\text{Control complexity}}$$

**The λ Trade-off:**
- λ = 0: Pure MSE → fits training data exactly → high variance
- λ → ∞: Ignores data → all weights → 0 → high bias
- Optimal λ: Minimizes total test error

**Practical Strategy:**
1. Start with unregularized model
2. If train error low, test error high → add regularization
3. Cross-validate to find optimal λ
4. Plot train/val curves to diagnose

**Key Insight:** The cost function implicitly defines what "good" means. Adding regularization changes this definition from "fit training data perfectly" to "fit reasonably while keeping model simple."

---

## Question 3
- [ ] Done

**How would you select an appropriate cost function for a stock price prediction model?**

**Answer:**

**Scenario Analysis:**
Stock prices have specific characteristics that influence cost function choice:
- High volatility and outliers (market crashes, spikes)
- Directional accuracy matters (up/down)
- Magnitude matters differently (missing 10% crash vs 1% move)

**Decision Framework:**

| Requirement | Recommended Loss | Reason |
|-------------|-----------------|--------|
| General prediction | Huber Loss | Robust to outliers |
| Direction matters more | Custom directional loss | Penalize wrong direction more |
| Percentage error | MAPE (Mean Absolute Percentage Error) | Scale-independent |
| Trading decisions | Profit-based loss | Directly optimize returns |

**Recommended Approach:**

**Option 1: Huber Loss** (for robustness)
- Stock data has fat tails (crashes, spikes)
- MSE would be dominated by extreme days
- Huber smoothly transitions to linear for outliers

**Option 2: MAPE** (for percentage accuracy)
$$MAPE = \frac{1}{n}\sum\left|\frac{y - \hat{y}}{y}\right| \times 100$$
- Missing a $100 stock by $10 = 10% error
- Missing a $10 stock by $10 = 100% error

**Option 3: Asymmetric Loss** (if underpredicting is worse)
$$L = \begin{cases} \alpha(y - \hat{y})^2 & \text{if } y > \hat{y} \\ (y - \hat{y})^2 & \text{otherwise} \end{cases}$$

**Practical Recommendation:**
1. Start with Huber Loss (δ based on typical daily volatility)
2. Evaluate on directional accuracy, not just MSE
3. Consider ensemble of models with different losses
4. Backtest trading strategy based on predictions

**Key Point:** The "best" loss depends on how predictions will be used - pure accuracy vs trading decisions vs risk management.

---

## Question 4
- [ ] Done

**Propose a strategy for choosing a cost function when dealing with imbalanced classification tasks.**

**Answer:**

**Scenario:** Fraud detection with 99% legitimate, 1% fraud. Standard cross-entropy treats all samples equally.

**Problem:**
- Model predicts "all legitimate" → 99% accuracy!
- But catches 0% of fraud
- Loss is dominated by majority class

**Strategy Framework:**

**Step 1: Weighted Cross-Entropy**
$$J = -\frac{1}{n}\sum[w_1 \cdot y\log(p) + w_0 \cdot (1-y)\log(1-p)]$$

| Class | Original Weight | New Weight (Inverse Freq) |
|-------|-----------------|---------------------------|
| Legitimate (99%) | 1.0 | 1.0 |
| Fraud (1%) | 1.0 | 99.0 |

**Step 2: Consider Focal Loss** (for extreme imbalance)
$$FL = -\alpha(1-p_t)^\gamma \log(p_t)$$
- $(1-p_t)^\gamma$ down-weights easy examples
- Forces model to focus on hard cases (the rare class)
- γ = 2 is common choice

**Step 3: Match to Business Cost**
$$Cost = w_{FN} \cdot FN + w_{FP} \cdot FP$$

| Error Type | Business Impact | Weight |
|------------|-----------------|--------|
| False Negative (miss fraud) | Lose $10,000 | High |
| False Positive (flag legitimate) | Inconvenience | Low |

**Decision Matrix:**

| Imbalance Level | Recommended Approach |
|-----------------|---------------------|
| Mild (70-30) | Weighted cross-entropy |
| Moderate (90-10) | Weighted + class weights |
| Severe (99-1) | Focal loss or SMOTE + weighted |
| Extreme (99.9-0.1) | Anomaly detection approach |

**Implementation:**
```python
# Sklearn
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(class_weight='balanced')  # Auto-weights

# PyTorch
weights = torch.tensor([1.0, 99.0])  # For 1% positive class
criterion = nn.CrossEntropyLoss(weight=weights)
```

**Key Logic:** Make each fraud case "count" as much as 99 legitimate cases in the loss.

---

## Question 5
- [ ] Done

**Discuss how you would modify the cost function in a situation where false negatives are more costly than false positives.**

**Answer:**

**Scenario:** Medical cancer screening - missing cancer (FN) is far worse than unnecessary follow-up (FP).

**Cost Matrix:**

| | Predicted: No Cancer | Predicted: Cancer |
|---|---------------------|-------------------|
| **Actual: No Cancer** | TN (correct) | FP (cost = $500 test) |
| **Actual: Cancer** | **FN (cost = life)** | TP (correct) |

**Solution 1: Asymmetric Weighted Cross-Entropy**

$$J = -\frac{1}{n}\sum[w_{FN} \cdot y\log(p) + w_{FP} \cdot (1-y)\log(1-p)]$$

Where $w_{FN} >> w_{FP}$

```python
# If FN is 10x worse than FP
def asymmetric_bce(y_true, y_pred, fn_weight=10, fp_weight=1):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    
    # FN: when y=1 but predict low (miss positive)
    fn_loss = fn_weight * y_true * np.log(y_pred)
    
    # FP: when y=0 but predict high (false alarm)
    fp_loss = fp_weight * (1 - y_true) * np.log(1 - y_pred)
    
    return -np.mean(fn_loss + fp_loss)
```

**Solution 2: Threshold Adjustment**
- Train with standard loss
- Lower decision threshold from 0.5 to 0.3
- More positives predicted → fewer FN

**Solution 3: Cost-Sensitive Learning**
Directly incorporate business costs:
$$Total Cost = C_{FN} \times \text{FN rate} + C_{FP} \times \text{FP rate}$$

**Practical Steps:**
1. Quantify business cost of FN vs FP
2. Set weight ratio = $C_{FN} / C_{FP}$
3. Apply weighted loss during training
4. Adjust threshold post-training if needed
5. Validate on business metric, not just accuracy

**Trade-off Visualization:**
```
Threshold: 0.5 → Balanced FP/FN
Threshold: 0.3 → More FP, fewer FN (conservative)
Threshold: 0.7 → Fewer FP, more FN (risky for cancer)
```

**Key Insight:** The cost function should reflect real-world asymmetric costs, not treat all errors equally.

---

## Question 6
- [ ] Done

**What are some recently proposed cost functions in academic literature for specialized machine learning tasks?**

**Answer:**

**1. Focal Loss (Lin et al., 2017 - RetinaNet)**
- **Task:** Object detection with extreme class imbalance
- **Formula:** $FL = -\alpha(1-p_t)^\gamma \log(p_t)$
- **Innovation:** Down-weights easy examples, focuses on hard cases
- **Impact:** Enabled single-stage detectors to match two-stage accuracy

**2. Dice Loss (Medical Imaging)**
- **Task:** Image segmentation with class imbalance
- **Formula:** $Dice = 1 - \frac{2|A \cap B|}{|A| + |B|}$
- **Innovation:** Directly optimizes overlap metric
- **Use:** Tumor segmentation, organ detection

**3. Contrastive Loss (SimCLR, MoCo - 2020)**
- **Task:** Self-supervised learning
- **Formula:** $L = -\log\frac{e^{sim(z_i, z_j)/\tau}}{\sum_k e^{sim(z_i, z_k)/\tau}}$
- **Innovation:** Learn representations without labels
- **Impact:** Foundation for modern vision models

**4. InfoNCE Loss (CLIP, 2021)**
- **Task:** Multi-modal learning (image-text)
- **Innovation:** Align different modalities in same embedding space
- **Impact:** Enabled zero-shot image classification

**5. Label Smoothing (Szegedy et al.)**
- **Task:** Prevent overconfident predictions
- **Formula:** $y_{smooth} = (1-\epsilon)y_{hard} + \epsilon/K$
- **Innovation:** Soft targets improve generalization

**6. Triplet Loss (FaceNet)**
- **Task:** Face recognition, similarity learning
- **Formula:** $L = \max(d(a,p) - d(a,n) + margin, 0)$
- **Innovation:** Learn embeddings where similar items are close

**7. WGAN Loss (Wasserstein GAN)**
- **Task:** Generative modeling
- **Innovation:** More stable GAN training via Earth Mover distance

**Key Trend:** Modern loss functions are designed for specific tasks rather than using generic MSE/cross-entropy.

---

## Question 7
- [ ] Done

**Discuss the application of cost functions in reinforcement learning, particularly in reward shaping strategies.**

**Answer:**

**Key Difference from Supervised Learning:**
- Supervised: Loss = prediction error (immediate feedback)
- RL: Reward signal may be sparse and delayed

**RL Objective:**
Maximize expected cumulative reward:
$$J(\pi) = \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

**Reward Shaping Strategies:**

**1. Sparse Reward Problem**
- Robot reaching goal: reward only at end
- Problem: Hard to learn from rare signal
- Solution: Add intermediate rewards

**2. Potential-Based Reward Shaping**
$$r'(s, a, s') = r(s, a, s') + \gamma \Phi(s') - \Phi(s)$$

Where $\Phi(s)$ = potential function (e.g., negative distance to goal)
- Preserves optimal policy
- Provides gradient toward goal

**3. Curriculum Learning**
- Start with easy tasks (more rewards)
- Gradually increase difficulty
- Analogous to learning rate scheduling

**Example: Robot Navigation**

| Reward Type | Formula | Effect |
|-------------|---------|--------|
| Sparse | +1 at goal, 0 elsewhere | Hard to learn |
| Dense | -distance_to_goal | Faster learning |
| Shaped | -Δdistance (improvement) | Even faster |

**Cost Function View in RL:**

**Policy Gradient:**
$$\nabla J(\theta) = \mathbb{E}[\nabla \log \pi(a|s) \cdot Q(s,a)]$$

**Value Function Loss (TD Error):**
$$L = (r + \gamma V(s') - V(s))^2$$

**Actor-Critic:**
- Actor: Policy gradient (maximize reward)
- Critic: MSE on value prediction

**Practical Tips:**
1. Design reward to align with actual goal
2. Avoid reward hacking (agent finds loopholes)
3. Use intrinsic motivation for exploration
4. Consider inverse RL to learn reward from demonstrations

**Key Insight:** In RL, you design the cost/reward function, which directly shapes agent behavior. A poorly designed reward leads to unintended behavior.

---

