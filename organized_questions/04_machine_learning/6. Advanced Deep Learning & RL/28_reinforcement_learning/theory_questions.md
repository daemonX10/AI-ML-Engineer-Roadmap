# Reinforcement Learning Interview Questions - Theory Questions

---

## Question 1: What is reinforcement learning, and how does it differ from supervised and unsupervised learning?
- [ ] Done

### Definition
Reinforcement Learning (RL) is a paradigm where an agent learns to make decisions by interacting with an environment, receiving rewards or penalties for actions, and optimizing behavior to maximize cumulative reward over time.

### Key Differences

| Aspect | Supervised | Unsupervised | Reinforcement |
|--------|------------|--------------|---------------|
| **Feedback** | Labeled data (correct answer) | No labels | Delayed reward signal |
| **Goal** | Learn input→output mapping | Find patterns/structure | Maximize cumulative reward |
| **Data** | Static dataset | Static dataset | Generated through interaction |
| **Learning** | Direct error correction | Self-organization | Trial and error |

### Core RL Components
- **Agent**: The learner/decision maker
- **Environment**: What agent interacts with
- **State (s)**: Current situation
- **Action (a)**: Choice made by agent
- **Reward (r)**: Feedback signal
- **Policy (π)**: Strategy for choosing actions

### Mathematical Framework
Agent learns policy π(a|s) to maximize expected cumulative reward:
$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

Where γ is discount factor (0 to 1).

### Intuition
- **Supervised**: Learning from a teacher with correct answers
- **Unsupervised**: Finding patterns without guidance
- **RL**: Learning from experience through trial and error (like learning to ride a bicycle)

---

## Question 2: Explain the concept of the Markov Decision Process (MDP) in reinforcement learning
- [ ] Done

### Definition
MDP is the mathematical framework for modeling decision-making where outcomes are partly random and partly under the control of a decision maker. It formalizes the environment in RL with the Markov property (future depends only on current state, not history).

### MDP Components (5-tuple)

| Component | Symbol | Description |
|-----------|--------|-------------|
| **States** | S | Set of all possible states |
| **Actions** | A | Set of all possible actions |
| **Transition** | P(s'|s,a) | Probability of next state given current state and action |
| **Reward** | R(s,a,s') | Immediate reward for transition |
| **Discount** | γ | Factor for future rewards (0 ≤ γ ≤ 1) |

### Markov Property
$$P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1}|s_t, a_t)$$

Future depends only on current state, not on how we got there.

### Dynamics
```
Agent in state s → Takes action a → Environment transitions to s' → Agent receives reward r
                                    (with probability P(s'|s,a))
```

### Example: Grid World
- States: Grid positions
- Actions: Up, Down, Left, Right
- Transitions: Move in direction (might slip)
- Rewards: +1 at goal, -1 at trap, 0 elsewhere

---

## Question 3: What is the role of a policy in reinforcement learning?
- [ ] Done

### Definition
A policy defines the agent's behavior - it's a mapping from states to actions (or probability distributions over actions) that specifies what action to take in each state.

### Types of Policies

| Type | Definition | Example |
|------|------------|---------|
| **Deterministic** | π(s) = a | Always go left in state 5 |
| **Stochastic** | π(a|s) = P(a|s) | 70% left, 30% right in state 5 |

### Policy Representation

**Tabular**: Lookup table (small state spaces)
```
State 1 → Action A
State 2 → Action B
```

**Parametric**: Neural network π_θ(a|s) (large state spaces)

### Optimal Policy
The policy that maximizes expected cumulative reward:
$$\pi^* = \arg\max_\pi \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$

### Role in RL
1. **Defines behavior**: What agent does in each state
2. **Optimization target**: RL algorithms improve policy
3. **Evaluation metric**: Compare policies by expected return
4. **Exploration**: Stochastic policies enable exploration

---

## Question 4: What are value functions and how do they relate to reinforcement learning policies?
- [ ] Done

### Definition
Value functions estimate "how good" it is for an agent to be in a state (or take an action in a state) under a given policy. They quantify expected cumulative reward from that point forward.

### Types of Value Functions

**State Value Function V(s):**
$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} | s_t = s\right]$$
Expected return starting from state s, following policy π.

**Action Value Function Q(s,a):**
$$Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} | s_t = s, a_t = a\right]$$
Expected return starting from state s, taking action a, then following π.

### Relationship
$$V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s,a)$$

State value = weighted average of action values.

### Optimal Value Functions
$$V^*(s) = \max_\pi V^\pi(s)$$
$$Q^*(s,a) = \max_\pi Q^\pi(s,a)$$

### Role in RL

| Use | How Value Functions Help |
|-----|-------------------------|
| **Policy Evaluation** | Assess how good current policy is |
| **Policy Improvement** | Choose actions with highest Q values |
| **Temporal Difference** | Bootstrap from estimated values |

### Deriving Optimal Policy from Q*
$$\pi^*(s) = \arg\max_a Q^*(s,a)$$

---

## Question 5: Describe the difference between on-policy and off-policy learning
- [ ] Done

### Definition
**On-policy** methods learn about the policy currently being used to make decisions. **Off-policy** methods learn about a different (target) policy than the one generating experience (behavior policy).

### Key Differences

| Aspect | On-Policy | Off-Policy |
|--------|-----------|------------|
| **Learning** | From own actions | From any policy's actions |
| **Target Policy** | Same as behavior | Different from behavior |
| **Example** | SARSA | Q-Learning |
| **Data Efficiency** | Lower | Higher (can reuse data) |
| **Stability** | More stable | Can be unstable |

### On-Policy (SARSA)
```
Current policy π generates: (s, a, r, s', a')
Update: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
                                    ↑ action from same policy
```

### Off-Policy (Q-Learning)
```
Behavior policy generates: (s, a, r, s')
Update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
                                    ↑ max over all actions (not from policy)
```

### When to Use

| Scenario | Preference |
|----------|------------|
| Experience replay | Off-policy |
| Learning from demonstrations | Off-policy |
| Safe exploration | On-policy |
| Simple implementation | On-policy |

### Importance Sampling
Off-policy methods may use importance sampling to correct for distribution mismatch:
$$\rho = \frac{\pi(a|s)}{b(a|s)}$$

---

## Question 6: What is the exploration vs. exploitation trade-off in reinforcement learning?
- [ ] Done

### Definition
The exploration-exploitation dilemma is the fundamental tension between trying new actions to discover potentially better rewards (exploration) and using current knowledge to maximize immediate reward (exploitation).

### Core Concepts

| Strategy | Definition | Risk |
|----------|------------|------|
| **Exploration** | Try unfamiliar actions | May waste time on bad actions |
| **Exploitation** | Use best known action | May miss better alternatives |

### Why It Matters
- Too much exploration: Never converges to good policy
- Too much exploitation: Gets stuck in local optima

### Common Strategies

**1. ε-Greedy:**
```python
if random() < epsilon:
    action = random_action()  # Explore
else:
    action = argmax(Q[state])  # Exploit
```

**2. Decaying ε:**
- Start with high ε (more exploration)
- Decrease over time (more exploitation)
- ε_t = ε_0 * decay^t

**3. Softmax/Boltzmann:**
$$P(a|s) = \frac{e^{Q(s,a)/\tau}}{\sum_{a'} e^{Q(s,a')/\tau}}$$
- Temperature τ controls exploration
- High τ: more uniform (exploration)
- Low τ: more greedy (exploitation)

**4. Upper Confidence Bound (UCB):**
$$a = \arg\max_a \left[Q(a) + c\sqrt{\frac{\ln t}{N(a)}}\right]$$
- Bonus for less-tried actions
- Balances exploration automatically

### Practical Guidance
- Early training: More exploration
- Later training: More exploitation
- Complex environments: More sophisticated methods (UCB, Thompson Sampling)

---

## Question 7: What are the Bellman equations, and how are they used in reinforcement learning?
- [ ] Done

### Definition
Bellman equations express the relationship between the value of a state (or state-action pair) and the values of successor states. They form the foundation for computing and learning value functions in RL.

### Bellman Expectation Equations

**For V:**
$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

**For Q:**
$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$$

### Bellman Optimality Equations

**For V*:**
$$V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]$$

**For Q*:**
$$Q^*(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]$$

### Intuition
Current value = Immediate reward + Discounted future value

```
V(s) = E[r + γV(s')]
       ↑   ↑
    now  later
```

### Use in RL Algorithms

| Algorithm | How Bellman Equations Used |
|-----------|---------------------------|
| **Value Iteration** | Iteratively apply optimality equation |
| **Policy Iteration** | Solve expectation equation, then improve |
| **Q-Learning** | Sample-based optimality update |
| **TD Learning** | Bootstrap with current estimates |

### TD Update (Sample-based Bellman)
$$V(s) \leftarrow V(s) + \alpha[r + \gamma V(s') - V(s)]$$

The term [r + γV(s') - V(s)] is the TD error.

---

## Question 8: Explain the difference between model-based and model-free reinforcement learning
- [ ] Done

### Definition
**Model-based RL** learns or uses a model of the environment (transition and reward functions) to plan actions. **Model-free RL** learns policies or value functions directly from experience without learning environment dynamics.

### Key Differences

| Aspect | Model-Based | Model-Free |
|--------|-------------|------------|
| **What's Learned** | Environment model | Policy/Value directly |
| **Planning** | Can simulate and plan | No planning |
| **Sample Efficiency** | Higher | Lower |
| **Computation** | Planning is expensive | Simpler |
| **Model Errors** | Can compound | N/A |

### Model-Based Approach
```
1. Learn model: P(s'|s,a), R(s,a)
2. Plan using model (e.g., tree search)
3. Execute best action
4. Update model with new experience
```

### Model-Free Approach
```
1. Take action, observe (s, a, r, s')
2. Update Q(s,a) or π directly
3. Repeat
```

### Examples

| Model-Based | Model-Free |
|-------------|------------|
| Dyna-Q | Q-Learning |
| AlphaGo (MCTS) | DQN |
| World Models | Policy Gradient |
| MuZero | PPO, A3C |

### Hybrid (Dyna)
```
Real experience → Update model and value function
Simulated experience → Use model to generate more updates
```

### When to Use

| Scenario | Preference |
|----------|------------|
| Accurate model available | Model-based |
| Complex environment | Model-free |
| Sample-limited | Model-based |
| Real-time decisions | Model-free |

---

## Question 9: What are the advantages and disadvantages of model-based reinforcement learning?
- [ ] Done

### Advantages

| Advantage | Explanation |
|-----------|-------------|
| **Sample Efficiency** | Can learn from fewer real interactions |
| **Planning** | Look ahead before acting |
| **Transfer** | Model transfers to new reward functions |
| **Safety** | Test actions in simulation first |
| **Explanation** | Model provides interpretability |

### Disadvantages

| Disadvantage | Explanation |
|--------------|-------------|
| **Model Errors** | Errors compound during planning |
| **Complexity** | Need to learn accurate model |
| **Computation** | Planning can be expensive |
| **State Space** | Hard to model high-dimensional states |
| **Partial Observability** | Model harder to learn |

### Model Error Problem
```
Small model error → Plan many steps → Large cumulative error
```

Solution: Short planning horizons, model uncertainty

### When Model-Based Works Well
- Environment has clear structure
- Good simulator available
- Sample collection is expensive
- Safety is critical (test in simulation)

### When Model-Based Struggles
- High-dimensional observations (images)
- Chaotic dynamics
- Model is harder to learn than policy

---

## Question 10: How does Q-learning work, and why is it considered a model-free method?
- [ ] Done

### Definition
Q-learning is an off-policy, model-free RL algorithm that learns the optimal action-value function Q* directly from experience, without requiring a model of the environment's dynamics.

### Algorithm

**Update Rule:**
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

Where:
- α: Learning rate
- γ: Discount factor
- r + γ max Q(s',a'): TD target
- r + γ max Q(s',a') - Q(s,a): TD error

### Algorithm Steps
```
1. Initialize Q(s,a) arbitrarily
2. For each episode:
   a. Initialize state s
   b. For each step:
      - Choose action a using policy (e.g., ε-greedy)
      - Take action, observe r, s'
      - Update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',·) - Q(s,a)]
      - s ← s'
   c. Until terminal state
```

### Why Model-Free

| Model-Free Property | Q-Learning Characteristic |
|--------------------|---------------------------|
| **No transition model** | Doesn't learn P(s'|s,a) |
| **No reward model** | Doesn't learn R(s,a) |
| **Direct Q learning** | Learns Q* from samples |
| **Experience only** | Just needs (s,a,r,s') tuples |

### Off-Policy Nature
- Behavior: ε-greedy (for exploration)
- Target: Greedy max Q (optimal policy)
- Can learn from any experience, including other agents

### Convergence
Q-learning converges to Q* if:
- All state-action pairs visited infinitely often
- Learning rate decays appropriately
- MDP is finite

### Python Implementation
```python
def q_learning_update(Q, s, a, r, s_next, alpha=0.1, gamma=0.99):
    td_target = r + gamma * np.max(Q[s_next])
    td_error = td_target - Q[s, a]
    Q[s, a] += alpha * td_error
    return Q
```

---

## Question 11: Describe the Monte Carlo method in the context of reinforcement learning
- [ ] Done

### Definition
Monte Carlo (MC) methods learn value functions by averaging complete episode returns. They estimate V(s) or Q(s,a) by sampling full trajectories and computing actual returns, rather than bootstrapping from estimates.

### Core Idea
$$V(s) \approx \frac{1}{N}\sum_{i=1}^{N} G_i(s)$$

Average of returns observed after visiting state s.

### Types of MC Methods

| Type | When to Update |
|------|----------------|
| **First-Visit MC** | Update only on first visit to state in episode |
| **Every-Visit MC** | Update on every visit to state |

### Algorithm (First-Visit MC)
```
Initialize V(s), Returns(s) for all s

For each episode:
    Generate episode following π
    G ← 0
    For each step t (backwards):
        G ← γG + r_{t+1}
        If s_t not in earlier steps:
            Append G to Returns(s_t)
            V(s_t) ← average(Returns(s_t))
```

### MC vs TD Comparison

| Aspect | Monte Carlo | TD Learning |
|--------|-------------|-------------|
| **Bootstrap** | No (uses actual returns) | Yes (uses estimates) |
| **Bias** | Unbiased | Biased |
| **Variance** | High | Lower |
| **Requires** | Complete episodes | Can be online |
| **Convergence** | Slower | Faster |

### Advantages
- Simple to understand
- Works for non-Markov environments
- Unbiased estimates
- Good for episodic tasks

### Disadvantages
- High variance
- Requires complete episodes
- Slow convergence
- Not suitable for continuing tasks

---

## Question 12: What is Deep Q-Network (DQN), and how does it combine reinforcement learning with deep neural networks?
- [ ] Done

### Definition
DQN is an algorithm that combines Q-learning with deep neural networks to handle high-dimensional state spaces. It uses a neural network to approximate the Q-function and includes techniques like experience replay and target networks for stable training.

### Key Components

| Component | Purpose |
|-----------|---------|
| **Deep Q-Network** | Neural network approximates Q(s,a;θ) |
| **Experience Replay** | Store and sample random experiences |
| **Target Network** | Stable Q targets for updates |

### Architecture
```
State (e.g., image) → CNN → FC layers → Q-values for each action
```

### DQN Loss Function
$$L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2]$$

Where θ⁻ are target network parameters (updated periodically).

### Algorithm
```
Initialize:
    Q-network with weights θ
    Target network with weights θ⁻ = θ
    Replay buffer D

For each episode:
    For each step:
        1. Select action (ε-greedy from Q)
        2. Execute action, observe r, s'
        3. Store (s, a, r, s', done) in D
        4. Sample mini-batch from D
        5. Compute targets: y = r + γ max Q(s'; θ⁻)
        6. Update θ by gradient descent on (y - Q(s,a; θ))²
        7. Periodically: θ⁻ ← θ
```

### Why Deep Learning + RL Works

| Challenge | Solution |
|-----------|----------|
| High-dim states (images) | CNNs extract features |
| Large state spaces | Function approximation |
| Correlation in data | Experience replay breaks correlation |
| Moving targets | Target network stabilizes |

### Practical Relevance
- Atari games (first superhuman performance)
- Foundation for modern deep RL
- Robotics, games, recommendations

---

## Question 13: Describe the concept of experience replay in DQN and why it's important
- [ ] Done

### Definition
Experience replay stores agent's experiences (s, a, r, s', done) in a replay buffer and samples random mini-batches for training, breaking temporal correlations and enabling efficient data reuse.

### Mechanism
```
1. Collect experience: (s, a, r, s', done)
2. Store in buffer D (size N, e.g., 1M)
3. When training:
   - Sample random mini-batch from D
   - Update Q-network on mini-batch
```

### Why It's Important

| Benefit | Explanation |
|---------|-------------|
| **Breaks Correlation** | Consecutive experiences are correlated; random sampling decorrelates |
| **Data Efficiency** | Each experience used multiple times |
| **Stability** | Smooths out training distribution |
| **Off-Policy** | Enables learning from old experiences |

### Without Experience Replay
- Consecutive samples highly correlated
- Network overfits to recent experiences
- Training unstable, poor convergence

### Implementation
```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones
```

### Prioritized Experience Replay (PER)
Sample important experiences more often:
$$P(i) \propto |\delta_i|^\alpha + \epsilon$$

Where δ_i is TD error - larger errors = more to learn.

---

## Question 14: What are the main elements of the Proximal Policy Optimization (PPO) algorithm?
- [ ] Done

### Definition
PPO is a policy gradient algorithm that improves training stability by limiting policy updates through a clipped objective function, preventing destructively large policy changes.

### Key Elements

| Element | Purpose |
|---------|---------|
| **Clipped Objective** | Limits policy change magnitude |
| **Advantage Estimation** | Measures action quality relative to baseline |
| **Multiple Epochs** | Reuse data for multiple updates |
| **Value Function** | Baseline for variance reduction |

### Clipped Objective Function
$$L^{CLIP}(\theta) = \mathbb{E}\left[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)\right]$$

Where:
- r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) (probability ratio)
- A_t: Advantage estimate
- ε: Clip range (typically 0.2)

### How Clipping Works
```
If advantage > 0 (good action):
    Clip prevents ratio from going above 1+ε
    
If advantage < 0 (bad action):
    Clip prevents ratio from going below 1-ε
```

### Algorithm Overview
```
For each iteration:
    1. Collect trajectories using current policy π_old
    2. Compute advantages using GAE
    3. For K epochs:
        - Sample mini-batches
        - Update policy using clipped objective
        - Update value function
    4. π_old ← π
```

### Advantages of PPO
- Simple to implement
- Stable training
- Good sample efficiency
- Works across many tasks
- Minimal hyperparameter tuning

### PPO vs TRPO
| PPO | TRPO |
|-----|------|
| Clipped objective | KL constraint |
| First-order optimization | Second-order |
| Simpler | More complex |
| Similar performance | |

---

## Question 15: Explain how Actor-Critic methods work in reinforcement learning
- [ ] Done

### Definition
Actor-Critic methods combine policy-based (Actor) and value-based (Critic) approaches. The Actor learns the policy, while the Critic evaluates actions by learning the value function, providing feedback to improve the Actor.

### Two Components

| Component | What It Learns | Role |
|-----------|---------------|------|
| **Actor** | Policy π(a|s; θ) | Selects actions |
| **Critic** | Value V(s; w) or Q(s,a; w) | Evaluates actions |

### How They Work Together
```
1. Actor selects action a based on current policy
2. Agent takes action, receives reward r
3. Critic evaluates: "How good was that action?"
4. Actor updates using Critic's feedback
5. Critic updates to better estimate values
```

### Mathematical Framework

**Actor Update (Policy Gradient):**
$$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)]$$

**Critic Update (TD Learning):**
$$\delta = r + \gamma V(s') - V(s)$$
$$w \leftarrow w + \alpha_w \delta \nabla_w V(s)$$

### Advantage Function
$$A(s,a) = Q(s,a) - V(s) \approx r + \gamma V(s') - V(s) = \delta_{TD}$$

Using TD error as advantage estimate.

### Benefits

| Benefit | Explanation |
|---------|-------------|
| **Lower Variance** | Critic provides baseline, reducing variance |
| **Online Learning** | Can update every step (unlike MC) |
| **Continuous Actions** | Actor naturally handles continuous actions |
| **Sample Efficiency** | Better than pure policy gradient |

### Variants
- **A2C**: Synchronous advantage actor-critic
- **A3C**: Asynchronous version
- **SAC**: Soft Actor-Critic with entropy bonus
- **TD3**: Twin Delayed DDPG

---

## Question 16: How does the Asynchronous Advantage Actor-Critic (A3C) algorithm work?
- [ ] Done

### Definition
A3C uses multiple parallel actor-learners running on different CPU threads, each interacting with its own copy of the environment. They asynchronously update a shared global network, providing diverse experience and stable training without replay buffer.

### Architecture
```
                    Global Network (θ, w)
                    /    |    |    \
    Worker 1    Worker 2    Worker 3    Worker N
    (env copy)  (env copy)  (env copy)  (env copy)
```

### Algorithm

**Each Worker:**
```
1. Sync with global: θ' ← θ, w' ← w
2. Collect n-step trajectory using local policy
3. Compute gradients locally
4. Apply gradients to global network asynchronously
5. Repeat
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Multiple Workers** | Parallel exploration |
| **Async Updates** | No locking, faster training |
| **n-step Returns** | Balance bias-variance |
| **Shared Network** | Actor and Critic share layers |

### n-step Return
$$R_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n})$$

### Gradient Computation

**Policy Loss:**
∇_θ log π(a_t|s_t; θ')(R_t - V(s_t; w'))

**Value Loss:**
(R_t - V(s_t; w'))²

**Entropy Bonus** (for exploration):
$$H(\pi) = -\sum_a \pi(a|s) \log \pi(a|s)$$

### Why Async Works
- Diverse experiences from different workers
- Natural exploration through parameter lag
- No replay buffer needed
- Faster wall-clock training time

### A2C vs A3C
| A3C | A2C |
|-----|-----|
| Asynchronous | Synchronous |
| No batching | Batched updates |
| Stale gradients | Fresh gradients |
| Faster | More stable |

---

## Question 17: What is reward shaping, and how can it affect the performance of a reinforcement learning agent?
- [ ] Done

### Definition
Reward shaping adds supplementary rewards to guide agent learning, making sparse reward problems tractable. It provides intermediate feedback beyond the environment's natural reward signal.

### Why Needed
- Sparse rewards: Only reward at goal
- Long horizons: Reward far in future
- Hard exploration: Random actions unlikely to find reward

### Types of Reward Shaping

| Type | Description |
|------|-------------|
| **Potential-Based** | Theory-grounded, preserves optimal policy |
| **Heuristic** | Domain knowledge, may change optimal policy |
| **Curiosity-Based** | Reward for visiting novel states |

### Potential-Based Shaping (PBRS)
$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$

Where Φ(s) is potential function. Provably preserves optimal policy.

### Examples

**Maze Navigation:**
- Natural reward: +1 at goal only
- Shaped reward: -distance_to_goal (provides gradient)

**Robot Manipulation:**
- Natural: +1 when object placed correctly
- Shaped: + for approaching object, + for grasping, etc.

### Dangers of Reward Shaping

| Risk | Example |
|------|---------|
| **Reward Hacking** | Agent exploits shaped reward, ignores true goal |
| **Changed Optimal** | Shaped reward leads to different optimal policy |
| **Local Optima** | Agent gets stuck maximizing intermediate rewards |

### Best Practices
1. Use potential-based shaping when possible
2. Keep shaping rewards smaller than true reward
3. Validate that shaped policy achieves true goal
4. Consider curriculum learning instead

---

## Question 18: Explain the concept of policy gradients and how they are used to learn policies
- [ ] Done

### Definition
Policy gradient methods directly optimize the policy parameters by computing gradients of expected return with respect to policy parameters, then updating in the direction that increases expected reward.

### Core Idea
Instead of learning Q-values, directly parameterize and optimize policy:
$$\pi_\theta(a|s)$$

### Policy Gradient Theorem
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)\right]$$

### REINFORCE Algorithm
Using Monte Carlo returns:
$$\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t^i|s_t^i) \cdot G_t^i$$

### With Baseline (Variance Reduction)
$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot (Q(s,a) - b(s))\right]$$

Common baseline: V(s), giving advantage A(s,a) = Q(s,a) - V(s)

### Algorithm
```
For each episode:
    1. Collect trajectory τ = (s₀, a₀, r₀, s₁, ...)
    2. Compute returns G_t for each step
    3. Compute policy gradient
    4. Update: θ ← θ + α∇θJ(θ)
```

### Advantages

| Advantage | Explanation |
|-----------|-------------|
| **Continuous Actions** | Natural handling of continuous action spaces |
| **Stochastic Policies** | Can learn optimal stochastic policies |
| **Convergence** | Guaranteed to local optimum |

### Disadvantages
- High variance
- Sample inefficient
- Sensitive to hyperparameters

---

## Question 19: What are some common challenges with reward functions in reinforcement learning?
- [ ] Done

### Common Challenges

| Challenge | Description |
|-----------|-------------|
| **Sparse Rewards** | Reward only at goal; hard to learn |
| **Reward Hacking** | Agent exploits reward loopholes |
| **Reward Shaping Pitfalls** | Shaped rewards change optimal behavior |
| **Delayed Rewards** | Long delay between action and reward |
| **Multi-objective** | Conflicting reward signals |
| **Specification** | Hard to encode true objective |

### Sparse Rewards Problem
```
Most steps: r = 0
Goal reached: r = +1
```
Random exploration unlikely to reach goal.

**Solutions:**
- Reward shaping
- Curriculum learning
- Intrinsic motivation
- Hindsight Experience Replay (HER)

### Reward Hacking Examples
- Game: Agent finds bug that gives infinite score
- Robot: Shakes to maximize "progress" reward
- Safety: Disables reward sensor

### Delayed Rewards
Action now → Effect later
- Credit assignment problem
- Solution: TD methods, eligibility traces

### Multi-objective Rewards
$$r = w_1 \cdot r_{speed} + w_2 \cdot r_{safety} + w_3 \cdot r_{efficiency}$$

Challenges:
- Setting weights
- Conflicting objectives
- Pareto optimality

### Reward Design Best Practices
1. Start simple, iterate
2. Test for unintended behaviors
3. Use potential-based shaping
4. Consider human feedback
5. Validate on diverse scenarios

---

## Question 20: Describe Trust Region Policy Optimization (TRPO) and how it differs from other policy gradient methods
- [ ] Done

### Definition
TRPO is a policy gradient algorithm that constrains policy updates to a "trust region" using KL divergence, ensuring monotonic improvement and stable training without destructively large updates.

### Key Idea
Maximize expected improvement while constraining how much policy can change:
$$\max_\theta \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A(s,a)\right]$$
$$\text{subject to: } \mathbb{E}[KL(\pi_{\theta_{old}} || \pi_\theta)] \leq \delta$$

### How TRPO Differs

| Aspect | Vanilla PG | TRPO |
|--------|-----------|------|
| **Update Size** | Fixed learning rate | Constrained by KL |
| **Stability** | Can diverge | Monotonic improvement |
| **Optimization** | First-order | Second-order (natural gradient) |
| **Complexity** | Simple | Complex |

### Algorithm Outline
```
1. Collect trajectories with current policy
2. Compute advantages
3. Solve constrained optimization:
   - Compute policy gradient g
   - Compute Fisher Information Matrix F
   - Compute natural gradient: F⁻¹g
   - Line search to satisfy KL constraint
4. Update policy
```

### Natural Gradient
$$\theta_{new} = \theta_{old} + \sqrt{\frac{2\delta}{g^T F^{-1} g}} F^{-1} g$$

Where F is Fisher Information Matrix.

### Conjugate Gradient
Computing F⁻¹g directly is expensive. Use conjugate gradient to approximate.

### Advantages
- Theoretical guarantees
- Stable training
- Works for complex policies

### Disadvantages
- Computationally expensive
- Complex implementation
- PPO often preferred in practice

---

## Question 21: How does one scale reinforcement learning to handle high-dimensional state spaces?
- [ ] Done

### Challenges with High Dimensions
- Curse of dimensionality
- Sample complexity explodes
- Tabular methods infeasible

### Scaling Solutions

| Technique | How It Helps |
|-----------|--------------|
| **Function Approximation** | Neural networks generalize across states |
| **Convolutional Networks** | Handle image observations |
| **State Representation** | Learn compact representations |
| **Hierarchical RL** | Decompose into sub-problems |
| **Transfer Learning** | Leverage prior knowledge |

### Function Approximation
Instead of table: Q[s,a]
Use neural network: Q(s,a; θ)

### Deep RL Techniques

**Experience Replay:**
- Reuse samples efficiently
- Break correlations

**Target Networks:**
- Stabilize training
- Prevent oscillation

**Prioritized Replay:**
- Focus on important experiences

### State Representation Learning
```
High-dim observation → Encoder → Low-dim representation → Policy
```

Methods:
- Autoencoders
- Contrastive learning
- World models

### Hierarchical RL
```
High-level policy: Choose subgoal
Low-level policy: Achieve subgoal
```

Reduces effective horizon and complexity.

### Distributed Training
- Multiple parallel environments
- Distributed gradient computation
- Large-scale training (A3C, IMPALA)

---

## Question 22: Describe some strategies for transferring knowledge in reinforcement learning across different tasks
- [ ] Done

### Transfer Learning in RL

| Strategy | Description |
|----------|-------------|
| **Direct Transfer** | Use policy from source task on target |
| **Fine-tuning** | Pre-train on source, fine-tune on target |
| **Feature Transfer** | Reuse learned representations |
| **Policy Distillation** | Train student from teacher |
| **Reward Shaping** | Use source Q-values to shape target rewards |

### Domain Randomization
Train on varied simulated environments:
```
Simulation (many variations) → Real world transfer
```

Variations in:
- Physics parameters
- Visual appearance
- Dynamics

### Sim-to-Real Transfer
```
Train in simulation → Deploy in real world
```

Challenges:
- Reality gap
- Model mismatch

Solutions:
- Domain randomization
- System identification
- Adaptive policies

### Multi-task Learning
Train single policy on multiple tasks:
- Shared representations
- Task-specific heads
- Meta-learning

### Meta-Learning (Learning to Learn)
Learn to quickly adapt to new tasks:
- MAML: Gradient-based meta-learning
- RL²: RNN-based meta-learning

### Progressive Networks
```
Task 1 → Fixed columns
           ↓ lateral connections
Task 2 → New columns (with access to Task 1 features)
```

### Practical Considerations
- Task similarity affects transfer success
- Negative transfer possible if tasks too different
- Start with related tasks

---

## Question 23: What are the potential issues with overfitting in reinforcement learning and how can they be mitigated?
- [ ] Done

### Overfitting in RL

| Type | Description |
|------|-------------|
| **Environment Overfitting** | Policy only works in training environment |
| **Policy Overfitting** | Overfits to specific trajectories |
| **Value Function Overfitting** | Value estimates don't generalize |

### Causes
- Limited environment diversity
- Deterministic environments
- Small replay buffers
- Overtraining on same experiences

### Detection
- Performance gap: train env vs test env
- Policy fails in slightly different conditions
- Value estimates inaccurate on new states

### Mitigation Strategies

| Strategy | How It Helps |
|----------|--------------|
| **Environment Randomization** | Train on diverse conditions |
| **Procedural Generation** | Generate diverse training levels |
| **Regularization** | L2, dropout in networks |
| **Ensemble Methods** | Multiple value functions |
| **Data Augmentation** | Transform observations |
| **Early Stopping** | Stop before overfitting |

### Environment Randomization
```python
# Vary environment parameters
gravity = uniform(8, 12)
friction = uniform(0.5, 1.5)
wind = uniform(-1, 1)
```

### Data Augmentation for RL
- Image observations: crop, flip, color jitter
- DrQ: Data regularized Q-learning
- RAD: Reinforcement learning with augmented data

### Regularization Techniques
```python
# Dropout in policy network
policy = nn.Sequential(
    nn.Linear(state_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, action_dim)
)
```

### Best Practices
1. Test on held-out environments
2. Use diverse training conditions
3. Monitor generalization metrics
4. Apply appropriate regularization

---

## Question 24: In what way does the REINFORCE algorithm update policies, and how does it handle variance in updates?
- [ ] Done

### REINFORCE Update
REINFORCE uses Monte Carlo returns to estimate policy gradients:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

### Algorithm
```
For each episode:
    1. Generate trajectory: τ = (s₀, a₀, r₀, ..., s_T)
    2. For each timestep t:
        Compute return: G_t = Σᵢ γⁱ r_{t+i}
        Compute gradient: ∇log π(aₜ|sₜ) · Gₜ
    3. Update: θ ← θ + α · gradient
```

### Variance Problem
Monte Carlo returns have high variance:
- Long episodes → more randomness
- Single sample estimates
- Reward scale affects variance

### Variance Reduction Techniques

**1. Baseline Subtraction:**
$$\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi(a|s) \cdot (G_t - b(s))\right]$$

Common baselines:
- Running average of returns
- Learned value function V(s)

**2. Advantage Function:**
$$A(s,a) = G_t - V(s)$$

Only updates for actions better than average.

**3. Reward Normalization:**
```python
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

**4. Reward-to-Go:**
Use future returns only (not past rewards):
$$G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$$

### Implementation with Baseline
```python
# Compute returns
returns = compute_returns(rewards, gamma)

# Compute advantages (using baseline)
advantages = returns - baseline_values

# Normalize advantages
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# Policy gradient loss
log_probs = policy.log_prob(actions)
loss = -(log_probs * advantages).mean()
```

---

## Question 25: Explain the concept of inverse reinforcement learning
- [ ] Done

### Definition
Inverse Reinforcement Learning (IRL) recovers the reward function from observed expert behavior. Instead of learning policy from reward, it learns reward from policy (demonstrations).

### Problem Setup
```
Standard RL:  Reward → Learn → Policy
Inverse RL:   Policy (demonstrations) → Learn → Reward
```

### Why IRL?
- Rewards are hard to specify manually
- Experts demonstrate desired behavior easily
- Learn transferable reward function

### Core Assumption
Expert behavior is (near) optimal for some reward function.

### Challenges

| Challenge | Description |
|-----------|-------------|
| **Ill-posed** | Many rewards explain same behavior |
| **Ambiguity** | Constant reward, scaled reward work |
| **Sample Complexity** | Need many demonstrations |

### Methods

**Maximum Entropy IRL:**
$$\max_R \mathbb{E}_\pi[\sum r(s,a)] + H(\pi)$$

Choose reward that makes demonstrations most likely while maximizing entropy.

**Apprenticeship Learning:**
Match feature expectations:
$$\mathbb{E}_\pi[\phi(s)] \approx \mathbb{E}_{\pi_{expert}}[\phi(s)]$$

**Generative Adversarial Imitation Learning (GAIL):**
Use GAN framework:
- Generator: Policy
- Discriminator: Distinguishes expert from policy

### GAIL
```
Expert demos → Discriminator ← Policy trajectories
                    ↓
              Reward signal → Policy update
```

### Applications
- Autonomous driving (learn from human drivers)
- Robotics (learn from demonstrations)
- Game AI (learn from expert players)

---

## Question 26: What is partial observability in reinforcement learning, and how can it be addressed?
- [ ] Done

### Definition
Partial observability occurs when the agent cannot fully observe the true state of the environment, receiving only incomplete or noisy observations. This is modeled as a Partially Observable MDP (POMDP).

### POMDP vs MDP

| MDP | POMDP |
|-----|-------|
| Agent sees true state s | Agent sees observation o |
| Policy: π(a|s) | Policy: π(a|h) where h is history |
| Markov property holds | Markov property on observations doesn't hold |

### Examples
- Poker: Can't see opponent's cards
- Robot: Limited sensor range
- Healthcare: Can't observe all patient state

### Challenges
- Current observation insufficient for optimal action
- Must reason about hidden state
- Maintain belief over possible states

### Solutions

**1. History/Memory-Based:**
```
Policy conditioned on observation history:
π(a | o₁, o₂, ..., oₜ)
```

**2. Recurrent Neural Networks:**
```
LSTM/GRU maintains hidden state:
h_t = RNN(o_t, h_{t-1})
a_t = policy(h_t)
```

**3. Belief State:**
Maintain probability distribution over states:
$$b(s) = P(s_t = s | o_1, ..., o_t, a_1, ..., a_{t-1})$$

**4. Frame Stacking:**
Stack recent observations:
```
input = [o_{t-3}, o_{t-2}, o_{t-1}, o_t]
```

### Recurrent Policy Network
```python
class RecurrentPolicy(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        self.lstm = nn.LSTM(obs_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, obs, hidden):
        lstm_out, hidden = self.lstm(obs, hidden)
        action_probs = F.softmax(self.policy(lstm_out), dim=-1)
        return action_probs, hidden
```

---

## Question 27: Describe ways in which reinforcement learning can be used in healthcare
- [ ] Done

### Applications

| Application | Description |
|-------------|-------------|
| **Treatment Optimization** | Sequential treatment decisions |
| **Drug Dosing** | Personalized medication dosing |
| **Clinical Trials** | Adaptive trial design |
| **Resource Allocation** | Hospital resource management |
| **Diagnosis Support** | Sequential diagnostic testing |

### Treatment Recommendation
```
Patient state → RL Agent → Treatment action → Outcome → Reward
```

**Sepsis Treatment:**
- State: Vital signs, lab values
- Actions: Treatments (fluids, vasopressors)
- Reward: Patient survival, recovery

### Challenges in Healthcare RL

| Challenge | Consideration |
|-----------|---------------|
| **Safety** | Cannot explore freely on patients |
| **Offline Learning** | Learn from historical data only |
| **Interpretability** | Clinicians need to understand decisions |
| **Data Quality** | Missing data, noise, confounders |
| **Delayed Outcomes** | Long-term health effects |

### Offline/Batch RL
Learn from historical patient records without real interaction:
```
Historical data → Learn policy → Evaluate → Deploy (carefully)
```

### Evaluation Methods
- Off-policy evaluation (OPE)
- Importance sampling
- Simulation validation

### Example: Insulin Dosing
```
State: Blood glucose, meal info, activity
Action: Insulin dose
Reward: Time in target glucose range
```

### Ethical Considerations
- Patient consent
- Fairness across populations
- Human oversight
- Failure modes

---

## Question 28: Given a specific game, describe how you would design an agent to learn optimal strategies using reinforcement learning
- [ ] Done

### Scenario: Design RL agent for Pac-Man

### Step 1: Environment Definition

| Component | Definition |
|-----------|------------|
| **State** | Grid position, ghost positions, pellet locations, power-up status |
| **Actions** | Up, Down, Left, Right |
| **Rewards** | +10 pellet, +200 ghost (powered), -500 caught, +500 level complete |
| **Termination** | Caught by ghost OR level complete |

### Step 2: State Representation
```python
# Option 1: Raw grid (image-like)
state = grid_array  # (height, width, channels)

# Option 2: Feature vector
state = [
    pacman_x, pacman_y,
    ghost1_x, ghost1_y, ghost1_scared,
    # ... more ghosts
    nearest_pellet_direction,
    is_powered_up,
]
```

### Step 3: Algorithm Selection

| Algorithm | Why Consider |
|-----------|--------------|
| **DQN** | Image input, discrete actions |
| **PPO** | Stable, good baseline |
| **A2C** | Fast training |

### Step 4: Network Architecture
```python
class PacmanDQN(nn.Module):
    def __init__(self):
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 4)  # 4 actions
        )
```

### Step 5: Training Setup
- Experience replay buffer (100k+)
- Target network (update every 1000 steps)
- ε-greedy exploration (1.0 → 0.1)
- Reward clipping or normalization

### Step 6: Evaluation
- Track average score per episode
- Test on unseen levels
- Compare to human/baseline performance

### Step 7: Improvements
- Prioritized experience replay
- Dueling DQN architecture
- Reward shaping for intermediate goals

---

## Question 29: What are the latest advancements in multi-agent reinforcement learning?
- [ ] Done

### Key Advancements

| Area | Advancement |
|------|-------------|
| **Communication** | Learned communication protocols |
| **Cooperation** | Emergent collaborative behavior |
| **Competition** | Game-theoretic approaches |
| **Scalability** | Large-scale multi-agent training |
| **Generalization** | Zero-shot coordination |

### Centralized Training, Decentralized Execution (CTDE)
```
Training: Central critic sees all agents' observations
Execution: Each agent acts on local observation only
```

**QMIX, VDN:** Decompose joint Q-function
**MAPPO:** Multi-agent PPO with shared critic

### Emergent Communication
Agents learn to communicate:
```
Agent 1 → Message → Agent 2
   ↓                   ↓
Action              Action
```

Learning when and what to communicate.

### Self-Play
Train against copies of self:
- AlphaStar (StarCraft)
- OpenAI Five (Dota 2)

### Population-Based Training
Maintain diverse population of agents:
- Prevents exploitation
- Robust strategies

### Zero-Shot Coordination
Train agent to cooperate with unseen partners:
- Other-play
- Symmetry breaking
- Convention learning

### Applications
- Autonomous vehicles (coordination)
- Game AI (team games)
- Robotics (swarm)
- Economics (market simulation)

---

## Question 30: How does curriculum learning work in the context of reinforcement learning?
- [ ] Done

### Definition
Curriculum learning trains agents on progressively harder tasks, starting with simple versions and gradually increasing difficulty. This guides exploration and accelerates learning.

### Core Idea
```
Easy tasks → Medium tasks → Hard tasks
   ↓            ↓            ↓
Basic skills  Build on skills  Master full task
```

### Types of Curricula

| Type | How Difficulty Increases |
|------|-------------------------|
| **Task Curriculum** | Progressively harder tasks |
| **Environment Curriculum** | Simpler to complex environments |
| **Reward Curriculum** | Dense to sparse rewards |
| **Goal Curriculum** | Closer to farther goals |

### Automatic Curriculum Learning
Let agent or algorithm decide progression:

**Self-Paced Learning:**
Focus on tasks of appropriate difficulty.

**Goal Generation:**
```
Generate goals slightly beyond current capability
```

### PAIRED (Protagonist Antagonist Induced Regret)
- Teacher generates environments
- Student learns on generated environments
- Teacher maximizes student's regret

### Implementation Example
```python
def get_curriculum_task(agent_performance, curriculum_stages):
    """Progress through curriculum based on performance"""
    for i, (threshold, task) in enumerate(curriculum_stages):
        if agent_performance < threshold:
            return task
    return curriculum_stages[-1][1]  # Hardest task

curriculum_stages = [
    (0.5, easy_task),
    (0.7, medium_task),
    (0.85, hard_task)
]
```

### Benefits
- Faster convergence
- Better final performance
- Handles sparse rewards
- More stable training

### Challenges
- Designing curriculum
- Knowing when to progress
- Avoiding forgetting earlier skills

---

## Question 31: Explain the concept of meta-reinforcement learning
- [ ] Done

### Definition
Meta-RL learns how to learn, training agents to quickly adapt to new tasks with minimal experience. The agent learns a learning algorithm or adaptation strategy that generalizes across tasks.

### Core Idea
```
Standard RL: Learn one task well
Meta-RL: Learn to learn tasks quickly
```

### Mathematical Framework
Maximize expected return across task distribution:
$$\max_\theta \mathbb{E}_{\tau \sim p(\tau)}\left[\mathbb{E}_\pi\left[\sum_t r_t | \text{task } \tau\right]\right]$$

### Approaches

**1. Recurrent Meta-RL (RL²):**
```
Task 1 experience → RNN → Adapted policy
Task 2 experience → Same RNN → Different adapted policy
```

Hidden state encodes task-specific adaptation.

**2. Gradient-Based (MAML):**
```
θ → Few gradient steps on task → θ'_task
```

Learn initialization θ that adapts quickly with few gradients.

### MAML (Model-Agnostic Meta-Learning)
```
1. Sample batch of tasks
2. For each task:
   - Take k gradient steps on task
   - Evaluate adapted policy
3. Meta-update: Optimize for post-adaptation performance
```

### RL² Architecture
```python
class RL2Agent(nn.Module):
    def __init__(self):
        self.gru = nn.GRU(obs_dim + action_dim + reward_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, obs, prev_action, prev_reward, hidden):
        x = torch.cat([obs, prev_action, prev_reward], dim=-1)
        out, hidden = self.gru(x, hidden)
        return self.policy(out), hidden
```

### Applications
- Robotics (adapt to new objects)
- Game playing (adapt to opponents)
- Personalization (adapt to users)

---

## Question 32: What is the significance of interpretability in reinforcement learning, and how can it be achieved?
- [ ] Done

### Why Interpretability Matters

| Reason | Explanation |
|--------|-------------|
| **Trust** | Humans need to trust agent decisions |
| **Safety** | Understand failure modes |
| **Debugging** | Diagnose learning problems |
| **Regulation** | Legal requirements for explainability |
| **Improvement** | Identify areas for enhancement |

### Types of Interpretability

**1. Policy Interpretability:**
- Why did agent take this action?
- What factors influenced decision?

**2. Value Interpretability:**
- What does agent expect to happen?
- How valuable are different states?

### Achieving Interpretability

| Method | Description |
|--------|-------------|
| **Attention Visualization** | Show what agent focuses on |
| **Saliency Maps** | Highlight important input features |
| **Decision Trees** | Extract simple policy approximation |
| **Reward Decomposition** | Break down reward into components |
| **Language Explanations** | Generate natural language reasons |

### Attention Visualization
```python
# Visualize attention weights over state
attention_weights = agent.attention_layer(state)
plot_attention_heatmap(observation, attention_weights)
```

### Saliency Maps
Compute gradient of action w.r.t. input:
$$\text{saliency} = \left|\frac{\partial Q(s,a)}{\partial s}\right|$$

### Policy Distillation to Interpretable Model
```
Complex neural network policy → Distill → Decision tree
```

### Contrastive Explanations
"Why action A instead of action B?"
- Compare Q-values
- Highlight distinguishing features

### Challenges
- Accuracy vs interpretability trade-off
- Post-hoc explanations may not reflect true reasoning
- Different stakeholders need different explanations

---

## Question 33: Describe any emerging trends in reinforcement learning within financial technology
- [ ] Done

### Applications in FinTech

| Application | Description |
|-------------|-------------|
| **Algorithmic Trading** | Learn trading strategies |
| **Portfolio Optimization** | Dynamic asset allocation |
| **Market Making** | Optimal bid-ask spreads |
| **Risk Management** | Adaptive hedging strategies |
| **Fraud Detection** | Sequential detection |

### Algorithmic Trading
```
State: Market data, position, portfolio
Action: Buy, sell, hold (with quantities)
Reward: Returns, risk-adjusted returns
```

### Challenges in Finance RL

| Challenge | Description |
|-----------|-------------|
| **Non-stationarity** | Markets change over time |
| **Low Signal-to-Noise** | Returns are noisy |
| **Transaction Costs** | Actions have costs |
| **Partial Observability** | Cannot observe all market info |
| **Data Limitations** | Limited historical data |

### Recent Trends

**1. Multi-Agent Market Simulation:**
Model market as multi-agent system:
- Agent represents traders
- Emergent market dynamics

**2. Safe RL for Trading:**
- Constrained optimization
- Risk-aware policies
- Position limits

**3. Offline RL:**
Learn from historical trading data without online interaction.

**4. Hierarchical RL:**
```
High-level: Strategic allocation decisions
Low-level: Execution timing
```

### Risk-Adjusted Rewards
$$r = \text{return} - \lambda \cdot \text{risk}$$

Or use Sharpe ratio, CVaR constraints.

### Regulatory Considerations
- Model explainability required
- Fair trading practices
- Risk management oversight

---

## Question 34: What are some common pitfalls when scaling reinforcement learning applications?
- [ ] Done

### Common Pitfalls

| Pitfall | Description |
|---------|-------------|
| **Sample Inefficiency** | Need massive amounts of data |
| **Hyperparameter Sensitivity** | Small changes break training |
| **Reward Hacking** | Agent exploits reward loopholes |
| **Distribution Shift** | Training ≠ deployment conditions |
| **Credit Assignment** | Long delays between actions and rewards |
| **Exploration Challenges** | Random exploration fails at scale |

### Sample Inefficiency
```
Simple task: Millions of steps
Complex task: Billions of steps
Real-world: Often infeasible
```

**Solutions:**
- Better algorithms (PPO, SAC)
- Model-based RL
- Transfer learning
- Simulation

### Hyperparameter Sensitivity
```
Small learning rate change → Completely different results
```

**Solutions:**
- Population-based training
- Automated hyperparameter search
- Robust algorithm design

### Sim-to-Real Gap
```
Works in simulation → Fails in real world
```

**Solutions:**
- Domain randomization
- System identification
- Adaptive policies

### Reward Specification
```
Intended: Robot walks forward
Actual: Robot exploits physics bug
```

**Solutions:**
- Careful reward design
- Reward learning from humans
- Constrained RL

### Debugging Difficulty
- Stochastic outcomes
- Delayed feedback
- Complex interactions

### Best Practices
1. Start simple, iterate
2. Extensive logging and visualization
3. Test in diverse environments
4. Use established baselines
5. Gradual scaling

---

## Question 35: How does one monitor and manage the ongoing performance of a deployed reinforcement learning system?
- [ ] Done

### Monitoring Framework

| Metric | What to Track |
|--------|---------------|
| **Reward** | Average, variance, trends |
| **Actions** | Distribution, unusual patterns |
| **States** | Coverage, out-of-distribution |
| **Model** | Prediction accuracy, confidence |
| **System** | Latency, errors, resources |

### Key Monitoring Components

**1. Performance Dashboards:**
```
- Cumulative reward over time
- Action frequency distribution
- Episode length statistics
- Success/failure rates
```

**2. Anomaly Detection:**
- Sudden performance drops
- Unusual action patterns
- Out-of-distribution states

**3. A/B Testing:**
- Compare new policy vs baseline
- Gradual rollout

### Continuous Evaluation
```python
def evaluate_deployed_policy(policy, env, n_episodes=100):
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = policy(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward)
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards)
    }
```

### Safety Monitoring

| Safety Measure | Implementation |
|----------------|----------------|
| **Action Constraints** | Hard limits on actions |
| **Fallback Policy** | Switch to safe policy if anomaly |
| **Human Override** | Allow manual intervention |
| **Confidence Thresholds** | Flag low-confidence decisions |

### Model Updates
- Periodic retraining on new data
- Online learning with safeguards
- Canary deployments

### Logging for Debugging
```python
log = {
    'timestamp': time.time(),
    'state': state,
    'action': action,
    'reward': reward,
    'q_values': q_values,
    'policy_entropy': entropy
}
```

---

## Question 36: Explain any new technique presented in a recent conference like NeurIPS or ICML that pertains to reinforcement learning
- [ ] Done

### Decision Transformer (NeurIPS 2021)
Frames RL as sequence modeling problem using Transformers.

### Core Idea
Instead of learning value functions or policy gradients, treat RL as sequence prediction:
```
Input: (R₁, s₁, a₁, R₂, s₂, a₂, ..., Rₜ, sₜ)
Output: aₜ
```

Where Rₜ is return-to-go (desired future reward).

### Architecture
```
Return-to-go → Embedding
State → Embedding      → Transformer → Action prediction
Action → Embedding
```

### Key Innovation
- Condition on desired return
- At test time: Specify high return → Get optimal actions
- No temporal difference learning
- No separate value function

### Advantages
- Stable training (supervised learning)
- Long-horizon credit assignment via attention
- Works well with offline data
- Simple implementation

### Training
```python
# Sequence of (return, state, action) tuples
# Supervised learning: predict action given return and state
loss = cross_entropy(predicted_action, actual_action)
```

### Results
- Competitive with state-of-the-art offline RL
- Better on long-horizon tasks
- Simpler than many RL algorithms

### Implications
- Bridge between language models and RL
- New paradigm for decision-making
- Sequence modeling for control

---

## Question 37: Describe an end-to-end pipeline you would set up for training, validating, and deploying a reinforcement learning model in a commercial project
- [ ] Done

### Pipeline Overview
```
1. Problem Definition → 2. Environment → 3. Training → 4. Evaluation → 5. Deployment → 6. Monitoring
```

### Step 1: Problem Definition

| Item | Details |
|------|---------|
| **Objective** | Clear business goal |
| **State Space** | What agent observes |
| **Action Space** | What agent can do |
| **Reward** | How success is measured |
| **Constraints** | Safety, resource limits |

### Step 2: Environment Setup
```python
class ProductionEnv:
    def __init__(self, config):
        self.config = config
    
    def reset(self):
        return initial_state
    
    def step(self, action):
        next_state, reward, done, info = self.simulate(action)
        return next_state, reward, done, info
```

### Step 3: Training Infrastructure
```
- Distributed training (multiple workers)
- Hyperparameter search (Ray Tune)
- Experiment tracking (MLflow, W&B)
- Checkpointing
```

### Step 4: Evaluation Protocol
```python
evaluation_suite = {
    'in_distribution': test_env_same,
    'out_of_distribution': test_env_different,
    'edge_cases': test_env_edge,
    'safety_tests': test_env_safety
}

for name, env in evaluation_suite.items():
    metrics = evaluate(policy, env)
    log_metrics(name, metrics)
```

### Step 5: Deployment

**A/B Testing:**
```
10% traffic → New policy (monitoring)
90% traffic → Baseline policy
```

**Gradual Rollout:**
```
Week 1: 10% → Week 2: 25% → Week 3: 50% → Week 4: 100%
```

### Step 6: Production Monitoring
```python
class ProductionMonitor:
    def __init__(self):
        self.metrics = []
    
    def log_decision(self, state, action, reward, confidence):
        self.metrics.append({
            'timestamp': time.time(),
            'state_hash': hash(state),
            'action': action,
            'reward': reward,
            'confidence': confidence
        })
        
        # Anomaly detection
        if confidence < THRESHOLD:
            alert("Low confidence decision")
        
        if self.detect_drift():
            alert("Distribution drift detected")
```

### Safety Layers
```
1. Action bounds checking
2. Confidence thresholds
3. Human approval for high-impact decisions
4. Fallback to safe policy
5. Kill switch
```

### Documentation
- Model cards
- Training reports
- Failure mode analysis
- Maintenance procedures

---


---

# --- SARSA Questions (from 05_reinforcement_learning/01_sarsa) ---

# SARSA - Theory Questions

## Question 1
- [ ] Done
**What does the acronym SARSA stand for?**

**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**Explain the "on-policy" nature of SARSA.**

**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How is the SARSA update rule formulated mathematically?**

**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**Compare SARSA and Q-Learning: what are the key differences?**

**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**Describe the process of learning a policy using SARSA on a gridworld.**

**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**When would SARSA be preferred over Q-Learning?**

**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**What are the main steps in the SARSA algorithm?**

**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**How do you select the next action in SARSA?**

**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**What role does the learning rate (α) play in SARSA?**

**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**What is the impact of a high discount factor (γ) in SARSA?**

**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How does SARSA handle the exploration-exploitation tradeoff?**

**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**In which scenarios can SARSA perform worse than Q-Learning?**

**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How does on-policy learning affect convergence in SARSA?**

**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**Explain the term "temporal difference" in SARSA.**

**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**What is Eligibility Traces in the context of SARSA(λ)?**

**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**How are rewards propagated in episodes using SARSA?**

**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How can SARSA be extended for continuous state and action spaces?**

**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What is the role of ε-greedy policy in SARSA?**

**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**Provide an example of a real-world task suitable for SARSA.**

**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**How does SARSA handle stochastic environments?**

**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**What is SARSA(λ) and why is λ introduced?**

**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**How do you initialize Q-values in SARSA?**

**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**What is the effect of the initial Q-values on learning speed?**

**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**How does Expected SARSA differ from regular SARSA?**

**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**What are the benefits of Expected SARSA over SARSA?**

**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What is the computational complexity of SARSA per iteration?**

**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How would you implement SARSA for the Taxi-v3 environment?**

**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**Explain why SARSA is considered an on-policy algorithm.**

**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How does batch updating work in SARSA?**

**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What is the policy improvement step in SARSA?**

**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**Describe a scenario of policy evaluation in SARSA.**

**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**How would you visualize the learning progress in SARSA?**

**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**What challenges do you face in hyperparameter tuning for SARSA?**

**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**How can function approximation be integrated with SARSA?**

**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**What are the limitations of tabular SARSA?**

**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What makes SARSA robust to policy changes?**

**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do terminal states affect updates in SARSA?**

**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**Give an example of using decaying epsilon for exploration in SARSA.**

**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**What are the theoretical convergence guarantees for SARSA?**

**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**How does stochasticity in the environment dynamics affect SARSA learning?**

**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**What is the practical impact of action stochasticity on SARSA's updates?**

**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**Explain reward shaping with respect to SARSA.**

**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you modify SARSA for partially observable environments?**

**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What is the difference between Monte Carlo and SARSA methods?**

**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How can SARSA be used in multi-agent settings?**

**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**Illustrate SARSA's performance in sparse reward environments.**

**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**What role does delayed reward play in SARSA's learning curve?**

**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**How do you adapt SARSA for continuous reward spaces?**

**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**Describe applications where SARSA may underperform.**

**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**What recent advancements or variants exist for the SARSA algorithm?**

**Answer:** _To be filled_

---


---

# --- DQN (Deep Q-Network) Questions (from 05_reinforcement_learning/02_dqn) ---

# Deep Q-Network (DQN) - Theory Questions

## Question 1
- [ ] Done
**What is a Deep Q-Network (DQN)?**

**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**How does DQN differ from traditional Q-learning?**

**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**Describe the architecture of a typical DQN.**

**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**What role does the replay buffer play in DQN?**

**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**Why is a target network used in DQN?**

**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**Describe the concept of "experience replay" in DQN.**

**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How are Q-values represented in a DQN?**

**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What types of problems are well suited for DQN?**

**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**Explain how the Bellman equation is applied in DQN.**

**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**What are some common challenges when training DQNs?**

**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How is stability achieved in DQN training?**

**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**Why are mini-batches used in DQN updates?**

**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How does the target network improve convergence?**

**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**Explain the concept of reward clipping in DQN.**

**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How does DQN handle high-dimensional inputs like images?**

**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**What is Double DQN and why is it needed?**

**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**What are dueling architectures in DQN?**

**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**Explain prioritized experience replay.**

**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How is the loss function defined in DQN?**

**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What metrics can be used to evaluate a DQN?**

**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**What are common pitfalls in hyperparameter tuning for DQN?**

**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**How do you prevent overestimation of Q-values in DQN?**

**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**What is the role of the discount factor in DQN?**

**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**How is exploration implemented in DQN agents?**

**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How can DQN be extended to continuous action spaces?**

**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What are common activation functions used in DQN networks?**

**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How does DQN perform in non-stationary environments?**

**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**When should you update the target network in DQN?**

**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**What are the main differences between DQN and Policy Gradient methods?**

**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**How does DQN manage memory constraints with large replay buffers?**

**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**What is the effect of batch size in DQN training?**

**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**How do you monitor and debug a DQN agent's performance?**

**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**What is "catastrophic forgetting" and how does it manifest in DQN?**

**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What are possible improvements for vanilla DQN?**

**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How would you visualize the learned Q-function in DQN?**

**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What are real-world applications of DQN?**

**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How robust is DQN to different reward structures?**

**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What's the impact of reward delay on DQN?**

**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How can transfer learning be applied to DQN?**

**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**What regularization techniques are effective for DQN?**

**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How does DQN scale to multi-agent systems?**

**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What is the impact of exploration vs. exploitation in DQN?**

**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you choose when to end training for a DQN agent?**

**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What metrics diagnose overfitting in DQN?**

**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you use DQN for policy distillation?**

**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What are the limitations of DQN for real-time control tasks?**

**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**In what scenarios would DQN fail?**

**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**How can DQN be combined with other RL algorithms?**

**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**How does network size affect DQN performance?**

**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**What are the main research trends in improving DQN?**

**Answer:** _To be filled_

---


---

# --- Policy Gradient Questions (from 05_reinforcement_learning/03_policy_gradient) ---

# Policy Gradient Methods - Theory Questions

## Question 1
- [ ] Done
**What are Policy Gradient methods in RL?**

**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**How do Policy Gradient methods differ from value-based methods?**

**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**Explain the general objective function optimized in Policy Gradient.**

**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**Why are Policy Gradient methods suitable for continuous action spaces?**

**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**Describe the steps of the REINFORCE algorithm.**

**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What is the variance problem with Policy Gradient methods?**

**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**How is the likelihood ratio trick used?**

**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What does "stochastic policy" mean in Policy Gradients?**

**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How does baseline function help reduce variance?**

**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**What's the difference between actor-only and actor-critic architectures?**

**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**What is entropy regularization and why is it used?**

**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**Explain episodic vs. step-wise policy gradient methods.**

**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do Policy Gradient methods handle large state spaces?**

**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What is reward-to-go and how is it used?**

**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How do you use advantage estimates in Policy Gradients?**

**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**Why is reward normalization important?**

**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do discount factors affect Policy Gradient performance?**

**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What is the gradient estimator in the classic Policy Gradient method?**

**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**Describe the use of trust regions in TRPO.**

**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**How does PPO improve stability over vanilla Policy Gradients?**

**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**What is exploration noise in Policy Gradients?**

**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**How do you implement constraint policies in Policy Gradient methods?**

**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How does off-policy policy gradient differ from on-policy?**

**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**What does "credit assignment" mean in policy gradients?**

**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How is policy improvement measured in practice?**

**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What optimization techniques work well for Policy Gradients?**

**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**What does it mean to "clip" objective functions in PPO?**

**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**How is advantage estimated in PPO?**

**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**Describe sample efficiency in Policy Gradient methods.**

**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What is GAE (Generalized Advantage Estimator)?**

**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How does batch size affect Policy Gradient performance?**

**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**When do Policy Gradient methods diverge?**

**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**What are deterministic Policy Gradient methods?**

**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**Can deterministic Policy Gradients be used with discrete actions?**

**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do Policy Gradient methods apply to multi-agent settings?**

**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**When would you not use Policy Gradients?**

**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do Policy Gradient methods scale to large environments?**

**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What is reward shaping in Policy Gradient approaches?**

**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**Describe a scenario where Policy Gradients outperform value-based methods.**

**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**How is the policy parameterized in deep Policy Gradient methods?**

**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**What is the relationship between actor-critic and Policy Gradients?**

**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What are common pitfalls in hyperparameter selection for Policy Gradients?**

**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you debug a Policy Gradient agent?**

**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What are popular libraries implementing Policy Gradient methods?**

**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you evaluate a trained Policy Gradient agent?**

**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What are the main limitations of Policy Gradient methods?**

**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How would you extend Policy Gradient methods to partial observability?**

**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**Discuss recent advancements in Policy Gradient research.**

**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**How is robustness analyzed in Policy Gradient methods?**

**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**What are practical ways to visualize policy evolution over training?**

**Answer:** _To be filled_

---


---

# --- Actor-Critic / A3C / PPO / TRPO Questions (from 05_reinforcement_learning/04_actor_critic_a3c_ppo_trpo) ---

# Actor-Critic / A3C / PPO / TRPO - Theory Questions

## Question 1
- [ ] Done
**What is an Actor-Critic architecture?**

**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**How does the actor-critic framework differ from pure policy gradients?**

**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**What is the main advantage of A3C over traditional RL algorithms?**

**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**How do multiple agents in A3C speed up learning?**

**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**What is asynchronous updating in A3C?**

**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What does PPO stand for? How does it work?**

**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**What problem does PPO solve in policy gradient methods?**

**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**How does TRPO enforce trust regions?**

**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**What are surrogate objective functions?**

**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**What is the advantage function, and how is it estimated?**

**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**What is the purpose of the critic in actor-critic methods?**

**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**How can the bias-variance tradeoff be managed in actor-critic models?**

**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**Compare PPO and TRPO: when would you use each?**

**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**What role does entropy bonus play in actor-critic methods?**

**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**How is experience replay handled in actor-critic approaches?**

**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**How do actor-critic methods ensure stability?**

**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**Describe synchronous vs. asynchronous actor-critic training.**

**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What is GAE(λ) and how is it used?**

**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**What are the limitations of on-policy actor-critic methods?**

**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**How is off-policy learning used in actor-critic variants?**

**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you implement clipping in PPO?**

**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**What is the main benefit of parallel environments in RL?**

**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How are recurrent neural networks used in A3C?**

**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**Why is PPO considered less sensitive to changes in hyperparameters?**

**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**Describe a typical use case for PPO.**

**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**How are policy updates constrained in TRPO?**

**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**What are the main challenges in scaling actor-critic methods?**

**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**How do you handle exploration in actor-critic models?**

**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**What is shared parameterization in actor-critic networks?**

**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**Describe sample efficiency in actor-critic approaches.**

**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**What is the role of the advantage estimator in PPO?**

**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**How do you prevent reward hacking in actor-critic RL?**

**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**Describe how actor-critic methods are used in robotics.**

**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What is the computational cost of TRPO per update?**

**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How is the KL divergence used in PPO and TRPO?**

**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What are common pitfalls when implementing A3C from scratch?**

**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**Explain transfer learning in the context of actor-critic RL.**

**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**How is TRPO's constraint implemented mathematically?**

**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**What are the main differences between A3C and DDPG?**

**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**How are distributed systems applied in actor-critic training?**

**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**What is stochasticity in policy outputs and why does it matter?**

**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**How does actor-critic handle delayed rewards?**

**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**In what ways can credit assignment be improved in actor-critic?**

**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**How do you monitor convergence in actor-critic agents?**

**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**Why might actor-critic methods be unstable?**

**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**How can you regularize actor-critic models?**

**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How do you ensure reproducibility in actor-critic experiments?**

**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**What are recent trends in actor-critic research?**

**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**What are real-world applications of PPO and TRPO?**

**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**How can actor-critic architectures be applied to multi-task RL?**

**Answer:** _To be filled_

---


---

# --- Multi-Agent RL Questions (from 05_reinforcement_learning/05_multi_agent_rl) ---

# Multi-Agent RL - Theory Questions

## Question 1
- [ ] Done
**What is Multi-Agent Reinforcement Learning (MARL)?**

**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**How does MARL differ from single-agent RL?**

**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**What are the primary challenges in MARL?**

**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**How do agents communicate in MARL?**

**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**What is non-stationarity in MARL environments?**

**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**How is cooperation handled in multi-agent settings?**

**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**Describe the credit assignment problem in MARL.**

**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**What are Markov Games (Stochastic Games) in MARL?**

**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**Give examples of real-world MARL applications.**

**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**When is competition preferred over cooperation in MARL?**

**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**What are decentralized policies in MARL?**

**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**How do centralized critics improve learning in MARL?**

**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**What methods exist for credit assignment in MARL?**

**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**How does reward sharing affect MARL performance?**

**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**What is the role of partial observability in MARL?**

**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**How can communication protocols be learned among agents?**

**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**What is independent Q-learning in the context of MARL?**

**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**How does scalability affect MARL design?**

**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**Explain the role of agent identity and role assignment.**

**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**How do you evaluate MARL agents?**

**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**What challenges exist for exploration in MARL?**

**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**How does reward shaping work in multi-agent scenarios?**

**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**What are typical failures in MARL systems?**

**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**How are policies coordinated among agents?**

**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**What is the role of competition and adversarial learning?**

**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What approaches exist for cooperative MARL?**

**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you prevent collusion among agents?**

**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**What is transfer learning in MARL?**

**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do agents learn to negotiate in MARL?**

**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**Describe policy gradients applied in multi-agent contexts.**

**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How is robustness ensured in MARL?**

**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**When might centralized training with decentralized execution be needed?**

**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**What are emergent behaviors in MARL?**

**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**Give an example of hierarchical learning in MARL.**

**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**How do you implement curriculum learning in MARL?**

**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**How is communication overhead managed in MARL?**

**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do policies adapt to changing agent populations?**

**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What role does game theory play in MARL?**

**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How can exploration-exploitation tradeoff be balanced in MARL?**

**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**What are benchmarks for MARL?**

**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**What is the biggest challenge in scaling MARL systems?**

**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**How are rewards split in cooperative tasks?**

**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How is stability measured in MARL?**

**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What are meta-learning approaches in MARL?**

**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**How do you visualize interactions among agents?**

**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What optimization techniques are used in MARL training?**

**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How is experience replay managed for multiple agents?**

**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**When do MARL systems fail to generalize?**

**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**What recent advancements exist in MARL research?**

**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**How can explainability be incorporated in MARL systems?**

**Answer:** _To be filled_

---


---

# --- Reward Shaping Questions (from 05_reinforcement_learning/06_reward_shaping) ---

# Reward Shaping - Theory Questions

## Question 1
- [ ] Done
**What is reward shaping in RL?**

**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**Why is reward shaping used?**

**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**How does reward shaping accelerate learning?**

**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**Give examples of potential-based reward shaping.**

**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**What are the risks of incorrect reward shaping?**

**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**How can reward shaping cause reward hacking?**

**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**What are the criteria for a "safe" reward shaping function?**

**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**Explain the concept of "potential" in potential-based reward shaping.**

**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How does reward shaping affect policy convergence?**

**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**What is the Bellman equation with reward shaping?**

**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**How do you evaluate the effectiveness of reward shaping?**

**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**What challenges arise in multi-objective reward shaping?**

**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**How do you design reward functions for complex tasks?**

**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**How is domain knowledge incorporated into reward shaping?**

**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**What is sparse vs. dense reward shaping?**

**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**How does reward shaping affect exploration?**

**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**What is the impact of reward shaping on credit assignment?**

**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**Can reward shaping change the optimal policy? Why or why not?**

**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How is reward decomposition related to reward shaping?**

**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**What are intrinsic vs. extrinsic rewards?**

**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**How do you balance multiple reward shaping signals?**

**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**How is reward shaping handled in multi-agent settings?**

**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**What is the effect of noisy rewards on reward shaping?**

**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**How is reward shaping validated empirically?**

**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**Describe applications where reward shaping is essential.**

**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What is the impact of delayed reward on reward shaping?**

**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**How do you prevent negative side effects of reward shaping?**

**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**How can reward shaping be used for transfer learning?**

**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**How do you combine reward shaping with policy gradients?**

**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**What is the role of auxiliary tasks in reward shaping?**

**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**How do you manage conflicting objectives in reward shaping?**

**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**When does reward shaping fail to help?**

**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**How is interpretability ensured in reward shaping?**

**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**What are examples of unintended consequences in reward shaping?**

**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**What is curriculum learning and how does it relate to reward shaping?**

**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What role does reward scaling play?**

**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you debug a poorly shaped reward function?**

**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**How does reward shaping affect robustness?**

**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**What tools exist to visualize reward shaping effects?**

**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**How is reward shaping applied in robotics?**

**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**What are the theoretical guarantees of reward shaping?**

**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**How do you empirically test if reward shaping improves performance?**

**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**How do you choose hyperparameters for shaped rewards?**

**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**What is the interaction of reward shaping with exploration?**

**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**When might you not use reward shaping?**

**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**How do humans inform reward function design?**

**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**What is the link between reward shaping and behavioral cloning?**

**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**How do you adapt reward shaping for dynamic environments?**

**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**How is fairness integrated with reward shaping in social applications?**

**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**Give examples of recent trends and innovations in reward shaping.**

**Answer:** _To be filled_

---


---

# --- Exploration vs Exploitation Questions (from 05_reinforcement_learning/07_exploration_exploitation) ---

# Exploration-Exploitation Tradeoff - Theory Questions

## Question 1
- [ ] Done
**What is the exploration-exploitation tradeoff?**

**Answer:** _To be filled_

---

## Question 2
- [ ] Done
**Why is it fundamental in reinforcement learning?**

**Answer:** _To be filled_

---

## Question 3
- [ ] Done
**Give examples of exploration strategies in RL.**

**Answer:** _To be filled_

---

## Question 4
- [ ] Done
**What is ε-greedy exploration?**

**Answer:** _To be filled_

---

## Question 5
- [ ] Done
**How does softmax exploration differ from ε-greedy?**

**Answer:** _To be filled_

---

## Question 6
- [ ] Done
**What are annealing schedules for exploration parameters?**

**Answer:** _To be filled_

---

## Question 7
- [ ] Done
**What is optimism in the face of uncertainty?**

**Answer:** _To be filled_

---

## Question 8
- [ ] Done
**Explain Upper Confidence Bound (UCB) exploration.**

**Answer:** _To be filled_

---

## Question 9
- [ ] Done
**How does Thompson Sampling handle the tradeoff?**

**Answer:** _To be filled_

---

## Question 10
- [ ] Done
**Why is balancing short-term and long-term rewards challenging?**

**Answer:** _To be filled_

---

## Question 11
- [ ] Done
**What is the regret in RL, and how is it measured?**

**Answer:** _To be filled_

---

## Question 12
- [ ] Done
**How do deep RL algorithms deal with the tradeoff?**

**Answer:** _To be filled_

---

## Question 13
- [ ] Done
**What is the effect of over-exploration?**

**Answer:** _To be filled_

---

## Question 14
- [ ] Done
**How can under-exploration hinder learning?**

**Answer:** _To be filled_

---

## Question 15
- [ ] Done
**What is intrinsic motivation and how is it used for exploration?**

**Answer:** _To be filled_

---

## Question 16
- [ ] Done
**How does count-based exploration work?**

**Answer:** _To be filled_

---

## Question 17
- [ ] Done
**How do curiosity-driven approaches affect RL training?**

**Answer:** _To be filled_

---

## Question 18
- [ ] Done
**What are the risks of exploration in real-world systems?**

**Answer:** _To be filled_

---

## Question 19
- [ ] Done
**How does the multi-armed bandit problem illustrate this tradeoff?**

**Answer:** _To be filled_

---

## Question 20
- [ ] Done
**How is the tradeoff quantified mathematically?**

**Answer:** _To be filled_

---

## Question 21
- [ ] Done
**Why is exploration more challenging in sparse reward environments?**

**Answer:** _To be filled_

---

## Question 22
- [ ] Done
**What is the impact of exploration on convergence time?**

**Answer:** _To be filled_

---

## Question 23
- [ ] Done
**How do you visualize exploration in state space?**

**Answer:** _To be filled_

---

## Question 24
- [ ] Done
**What are hybrid strategies for exploration?**

**Answer:** _To be filled_

---

## Question 25
- [ ] Done
**How does Bayesian RL address exploration-exploitation?**

**Answer:** _To be filled_

---

## Question 26
- [ ] Done
**What is parameter noise and how is it used?**

**Answer:** _To be filled_

---

## Question 27
- [ ] Done
**What is bootstrapped DQN, and how does it affect exploration?**

**Answer:** _To be filled_

---

## Question 28
- [ ] Done
**How do meta-learning approaches inform exploration-exploitation?**

**Answer:** _To be filled_

---

## Question 29
- [ ] Done
**What is exploration bonus in RL?**

**Answer:** _To be filled_

---

## Question 30
- [ ] Done
**How do biological systems solve the exploration-exploitation tradeoff?**

**Answer:** _To be filled_

---

## Question 31
- [ ] Done
**What practical problems are caused by misbalancing exploration and exploitation?**

**Answer:** _To be filled_

---

## Question 32
- [ ] Done
**How is hierarchical reinforcement learning related to exploration?**

**Answer:** _To be filled_

---

## Question 33
- [ ] Done
**What role does exploration play in transfer learning?**

**Answer:** _To be filled_

---

## Question 34
- [ ] Done
**How do ensembles help in exploration strategies?**

**Answer:** _To be filled_

---

## Question 35
- [ ] Done
**When is pure exploitation preferred?**

**Answer:** _To be filled_

---

## Question 36
- [ ] Done
**What is the impact of reward shaping on exploration?**

**Answer:** _To be filled_

---

## Question 37
- [ ] Done
**How do you empirically measure exploration?**

**Answer:** _To be filled_

---

## Question 38
- [ ] Done
**What tools exist for monitoring exploration vs. exploitation?**

**Answer:** _To be filled_

---

## Question 39
- [ ] Done
**How do dynamic environments affect the tradeoff?**

**Answer:** _To be filled_

---

## Question 40
- [ ] Done
**What is "deep exploration" and how is it achieved?**

**Answer:** _To be filled_

---

## Question 41
- [ ] Done
**How do you set exploration schedules for curriculum learning?**

**Answer:** _To be filled_

---

## Question 42
- [ ] Done
**What are theoretical bounds for exploration algorithms?**

**Answer:** _To be filled_

---

## Question 43
- [ ] Done
**What is the exploration-exploitation dilemma in MARL?**

**Answer:** _To be filled_

---

## Question 44
- [ ] Done
**How do you tune exploration parameters in practice?**

**Answer:** _To be filled_

---

## Question 45
- [ ] Done
**Why does exploration sometimes lead to catastrophic forgetting?**

**Answer:** _To be filled_

---

## Question 46
- [ ] Done
**What is the connection between generalization and exploration?**

**Answer:** _To be filled_

---

## Question 47
- [ ] Done
**How is exploration managed in partially observable environments?**

**Answer:** _To be filled_

---

## Question 48
- [ ] Done
**What is the computational cost of different exploration techniques?**

**Answer:** _To be filled_

---

## Question 49
- [ ] Done
**What are possible future directions in exploration-exploitation research?**

**Answer:** _To be filled_

---

## Question 50
- [ ] Done
**How does the exploration-exploitation tradeoff influence agent behavior long-term?**

**Answer:** _To be filled_

---
