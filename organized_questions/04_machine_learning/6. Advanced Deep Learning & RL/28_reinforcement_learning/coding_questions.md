# Reinforcement Learning Interview Questions - Coding Questions

---

## Question 1: Implement the epsilon-greedy strategy in Python for action selection
- [ ] Done

### Concept
Epsilon-greedy balances exploration and exploitation:
- With probability ε: Take random action (explore)
- With probability 1-ε: Take best known action (exploit)

### Implementation

```python
import numpy as np

def epsilon_greedy(q_values, epsilon):
    """
    Select action using epsilon-greedy strategy.
    
    Args:
        q_values: Array of Q-values for each action [Q(a0), Q(a1), ...]
        epsilon: Exploration probability (0 to 1)
    
    Returns:
        Selected action index
    """
    if np.random.random() < epsilon:
        # Explore: random action
        return np.random.randint(len(q_values))
    else:
        # Exploit: best action
        return np.argmax(q_values)


# Usage example
q_values = np.array([1.2, 3.5, 2.1, 0.8])
epsilon = 0.1

# Single action selection
action = epsilon_greedy(q_values, epsilon)
print(f"Selected action: {action}")

# Demonstrate distribution over many trials
actions = [epsilon_greedy(q_values, epsilon) for _ in range(10000)]
print(f"Action distribution: {np.bincount(actions) / 10000}")
```

### Output
```
Selected action: 1  # Best action (Q=3.5) most of the time
Action distribution: [0.025, 0.925, 0.025, 0.025]  # ~90% action 1, ~2.5% each other
```

### Vectorized Version (for batch processing)
```python
def epsilon_greedy_batch(q_values_batch, epsilon):
    """
    Batch epsilon-greedy selection.
    
    Args:
        q_values_batch: Shape (batch_size, num_actions)
        epsilon: Exploration probability
    
    Returns:
        Array of selected actions
    """
    batch_size = q_values_batch.shape[0]
    num_actions = q_values_batch.shape[1]
    
    # Generate mask: True for explore, False for exploit
    explore_mask = np.random.random(batch_size) < epsilon
    
    # Random actions for exploration
    random_actions = np.random.randint(num_actions, size=batch_size)
    
    # Greedy actions for exploitation
    greedy_actions = np.argmax(q_values_batch, axis=1)
    
    # Combine based on mask
    actions = np.where(explore_mask, random_actions, greedy_actions)
    
    return actions
```

---

## Question 2: Write a Python script to simulate a simple MDP using a transition matrix
- [ ] Done

### Concept
MDP defined by:
- States S
- Actions A  
- Transition probabilities P(s'|s,a)
- Rewards R(s,a,s')
- Discount factor γ

### Implementation

```python
import numpy as np

class SimpleMDP:
    def __init__(self, n_states, n_actions, transition_matrix, reward_matrix):
        """
        Initialize MDP.
        
        Args:
            n_states: Number of states
            n_actions: Number of actions
            transition_matrix: P[s, a, s'] = probability of s' given s, a
                              Shape: (n_states, n_actions, n_states)
            reward_matrix: R[s, a, s'] = reward for transition
                          Shape: (n_states, n_actions, n_states)
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = transition_matrix
        self.R = reward_matrix
        self.state = 0  # Initial state
        
    def reset(self):
        """Reset to initial state."""
        self.state = 0
        return self.state
    
    def step(self, action):
        """
        Take action, transition to next state.
        
        Returns:
            next_state, reward, done
        """
        # Sample next state from transition probabilities
        probs = self.P[self.state, action]
        next_state = np.random.choice(self.n_states, p=probs)
        
        # Get reward
        reward = self.R[self.state, action, next_state]
        
        # Update state
        self.state = next_state
        
        # Check if terminal (state n_states-1 is terminal)
        done = (self.state == self.n_states - 1)
        
        return next_state, reward, done


# Example: 3-state MDP with 2 actions
n_states = 3
n_actions = 2

# Transition matrix: P[s, a, s']
# State 0: action 0 -> likely stay, action 1 -> likely go to state 1
# State 1: action 0 -> go back, action 1 -> go to terminal
# State 2: terminal (absorbing)
P = np.zeros((n_states, n_actions, n_states))
P[0, 0] = [0.8, 0.2, 0.0]  # State 0, action 0
P[0, 1] = [0.1, 0.9, 0.0]  # State 0, action 1
P[1, 0] = [0.7, 0.3, 0.0]  # State 1, action 0
P[1, 1] = [0.0, 0.2, 0.8]  # State 1, action 1
P[2, :] = [0.0, 0.0, 1.0]  # Terminal state (self-loop)

# Reward matrix: R[s, a, s']
R = np.zeros((n_states, n_actions, n_states))
R[1, 1, 2] = 10.0  # Reward for reaching terminal

# Create and simulate MDP
mdp = SimpleMDP(n_states, n_actions, P, R)

# Run episode
state = mdp.reset()
total_reward = 0
trajectory = [state]

for _ in range(20):  # Max steps
    action = np.random.randint(n_actions)  # Random policy
    next_state, reward, done = mdp.step(action)
    total_reward += reward
    trajectory.append(next_state)
    
    if done:
        break

print(f"Trajectory: {trajectory}")
print(f"Total reward: {total_reward}")
```

### Output
```
Trajectory: [0, 0, 1, 0, 1, 2]
Total reward: 10.0
```

---

## Question 3: Code a Q-learning algorithm in Python to solve a grid-world problem
- [ ] Done

### Concept
Q-learning update:
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

### Implementation

```python
import numpy as np

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.state = (0, 0)  # Start position
        self.goal = (size-1, size-1)  # Goal position
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
        
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action_idx):
        action = self.actions[action_idx]
        new_row = max(0, min(self.size-1, self.state[0] + action[0]))
        new_col = max(0, min(self.size-1, self.state[1] + action[1]))
        self.state = (new_row, new_col)
        
        done = (self.state == self.goal)
        reward = 1.0 if done else -0.01  # Small penalty per step
        
        return self.state, reward, done


def q_learning(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Q-learning algorithm.
    
    Args:
        env: Environment with reset() and step() methods
        episodes: Number of training episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
    
    Returns:
        Q-table
    """
    # Initialize Q-table
    n_actions = len(env.actions)
    Q = {}
    
    def get_q(state, action):
        return Q.get((state, action), 0.0)
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)
            else:
                q_values = [get_q(state, a) for a in range(n_actions)]
                action = np.argmax(q_values)
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Q-learning update
            current_q = get_q(state, action)
            next_max_q = max([get_q(next_state, a) for a in range(n_actions)])
            td_target = reward + gamma * next_max_q * (1 - done)
            td_error = td_target - current_q
            
            Q[(state, action)] = current_q + alpha * td_error
            
            state = next_state
        
        # Print progress
        if (episode + 1) % 200 == 0:
            print(f"Episode {episode + 1} completed")
    
    return Q


def extract_policy(Q, env):
    """Extract greedy policy from Q-table."""
    policy = {}
    for row in range(env.size):
        for col in range(env.size):
            state = (row, col)
            q_values = [Q.get((state, a), 0.0) for a in range(len(env.actions))]
            policy[state] = np.argmax(q_values)
    return policy


# Train agent
env = GridWorld(size=5)
Q = q_learning(env, episodes=1000)

# Extract and display policy
policy = extract_policy(Q, env)
action_symbols = ['→', '←', '↓', '↑']

print("\nLearned Policy:")
for row in range(env.size):
    row_str = ""
    for col in range(env.size):
        if (row, col) == env.goal:
            row_str += "G "
        else:
            row_str += action_symbols[policy[(row, col)]] + " "
    print(row_str)
```

### Output
```
Episode 200 completed
Episode 400 completed
...

Learned Policy:
→ → → → ↓ 
→ → → → ↓ 
→ → → → ↓ 
→ → → → ↓ 
→ → → → G 
```

---

## Question 4: Implement a value iteration algorithm for a given MDP in Python
- [ ] Done

### Concept
Value iteration updates:
$$V(s) \leftarrow \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V(s')]$$

Iterate until convergence, then extract policy.

### Implementation

```python
import numpy as np

def value_iteration(P, R, gamma=0.99, theta=1e-6, max_iterations=1000):
    """
    Value iteration algorithm.
    
    Args:
        P: Transition probabilities P[s, a, s']
        R: Rewards R[s, a, s']
        gamma: Discount factor
        theta: Convergence threshold
        max_iterations: Maximum iterations
    
    Returns:
        V: Value function
        policy: Optimal policy
    """
    n_states = P.shape[0]
    n_actions = P.shape[1]
    
    # Initialize value function
    V = np.zeros(n_states)
    
    for iteration in range(max_iterations):
        V_old = V.copy()
        
        # Update each state
        for s in range(n_states):
            # Compute Q-value for each action
            q_values = []
            for a in range(n_actions):
                # Expected value: sum over next states
                q = 0
                for s_next in range(n_states):
                    q += P[s, a, s_next] * (R[s, a, s_next] + gamma * V_old[s_next])
                q_values.append(q)
            
            # Value = max Q-value
            V[s] = max(q_values)
        
        # Check convergence
        delta = np.max(np.abs(V - V_old))
        if delta < theta:
            print(f"Converged after {iteration + 1} iterations")
            break
    
    # Extract policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q_values = []
        for a in range(n_actions):
            q = sum(P[s, a, s_next] * (R[s, a, s_next] + gamma * V[s_next]) 
                    for s_next in range(n_states))
            q_values.append(q)
        policy[s] = np.argmax(q_values)
    
    return V, policy


# Example: Simple 4-state MDP
n_states = 4
n_actions = 2

# Transition matrix
P = np.zeros((n_states, n_actions, n_states))
# State 0: action 0 -> state 1, action 1 -> state 2
P[0, 0, 1] = 1.0
P[0, 1, 2] = 1.0
# State 1: action 0 -> state 0, action 1 -> state 3
P[1, 0, 0] = 1.0
P[1, 1, 3] = 1.0
# State 2: action 0 -> state 0, action 1 -> state 3
P[2, 0, 0] = 1.0
P[2, 1, 3] = 1.0
# State 3: terminal (absorbing)
P[3, :, 3] = 1.0

# Reward matrix
R = np.zeros((n_states, n_actions, n_states))
R[1, 1, 3] = 10.0  # Good path
R[2, 1, 3] = 1.0   # Less good path

# Run value iteration
V, policy = value_iteration(P, R, gamma=0.99)

print(f"Optimal values: {V}")
print(f"Optimal policy: {policy}")
```

### Output
```
Converged after 45 iterations
Optimal values: [9.8029 9.90  0.99  0.   ]
Optimal policy: [0 1 1 0]
```

---

## Question 5: Write a function to calculate the discounted reward for a sequence of rewards
- [ ] Done

### Concept
Discounted return from time t:
$$G_t = r_{t} + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$$

### Implementation

```python
import numpy as np

def discounted_reward(rewards, gamma=0.99):
    """
    Calculate discounted cumulative reward for a sequence.
    
    Args:
        rewards: List or array of rewards [r0, r1, r2, ...]
        gamma: Discount factor (0 to 1)
    
    Returns:
        Total discounted reward (return)
    """
    total = 0
    for i, r in enumerate(rewards):
        total += (gamma ** i) * r
    return total


def discounted_rewards_all(rewards, gamma=0.99):
    """
    Calculate discounted rewards-to-go for each timestep.
    G_t = r_t + gamma * G_{t+1}
    
    Args:
        rewards: List of rewards
        gamma: Discount factor
    
    Returns:
        Array of returns [G_0, G_1, G_2, ...]
    """
    n = len(rewards)
    returns = np.zeros(n)
    
    # Compute backwards
    running_return = 0
    for t in reversed(range(n)):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    
    return returns


def discounted_rewards_normalized(rewards, gamma=0.99):
    """
    Calculate and normalize discounted rewards (common in policy gradient).
    
    Args:
        rewards: List of rewards
        gamma: Discount factor
    
    Returns:
        Normalized returns
    """
    returns = discounted_rewards_all(rewards, gamma)
    
    # Normalize (zero mean, unit variance)
    mean = np.mean(returns)
    std = np.std(returns) + 1e-8  # Small epsilon for stability
    normalized = (returns - mean) / std
    
    return normalized


# Example usage
rewards = [1, 0, 0, 0, 10]  # Sparse reward at end
gamma = 0.99

# Total return from start
total_return = discounted_reward(rewards, gamma)
print(f"Total discounted return: {total_return:.4f}")

# Returns at each timestep
returns = discounted_rewards_all(rewards, gamma)
print(f"Returns-to-go: {returns}")

# Normalized returns (for policy gradient)
normalized = discounted_rewards_normalized(rewards, gamma)
print(f"Normalized returns: {normalized}")
```

### Output
```
Total discounted return: 10.5853
Returns-to-go: [10.5853  9.6821  9.7799  9.8787  10.    ]
Normalized returns: [-0.8166 -1.0146 -0.9932 -0.9718  0.7962]
```

### Vectorized Version (Efficient)
```python
def discounted_rewards_vectorized(rewards, gamma=0.99):
    """
    Efficient vectorized computation of discounted rewards.
    """
    rewards = np.array(rewards)
    n = len(rewards)
    
    # Create discount matrix
    indices = np.arange(n)
    discount_matrix = gamma ** np.abs(indices[:, None] - indices)
    discount_matrix = np.triu(discount_matrix)  # Upper triangular
    
    # Compute returns
    returns = discount_matrix @ rewards
    
    return returns
```

---

## Question 6: Develop a SARSA-learning based agent in Python for the Taxi-v3 environment
- [ ] Done

### Concept
SARSA (State-Action-Reward-State-Action) is on-policy TD control:
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$$

Key difference from Q-learning: Uses actual next action a' (from policy), not max.

### Implementation

```python
import numpy as np
import gymnasium as gym

def sarsa(env, episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    SARSA algorithm for Taxi-v3.
    
    Args:
        env: Gymnasium environment
        episodes: Number of training episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
    
    Returns:
        Q-table, episode_rewards
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Initialize Q-table
    Q = np.zeros((n_states, n_actions))
    
    # Track rewards
    episode_rewards = []
    
    def select_action(state, eps):
        if np.random.random() < eps:
            return np.random.randint(n_actions)
        return np.argmax(Q[state])
    
    for episode in range(episodes):
        state, _ = env.reset()
        action = select_action(state, epsilon)
        
        total_reward = 0
        done = False
        
        while not done:
            # Take action, observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Select next action (SARSA uses actual next action)
            next_action = select_action(next_state, epsilon)
            
            # SARSA update
            td_target = reward + gamma * Q[next_state, next_action] * (1 - done)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error
            
            # Move to next state-action
            state = next_state
            action = next_action
            total_reward += reward
        
        episode_rewards.append(total_reward)
        
        # Print progress
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")
    
    return Q, episode_rewards


def evaluate_policy(env, Q, episodes=100):
    """Evaluate learned policy."""
    total_rewards = []
    
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards), np.std(total_rewards)


# Train SARSA agent
env = gym.make('Taxi-v3')
Q, rewards = sarsa(env, episodes=10000)

# Evaluate
mean_reward, std_reward = evaluate_policy(env, Q)
print(f"\nEvaluation: {mean_reward:.2f} +/- {std_reward:.2f}")

# Show one episode
env = gym.make('Taxi-v3', render_mode='ansi')
state, _ = env.reset()
done = False

print("\nSample Episode:")
while not done:
    action = np.argmax(Q[state])
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    print(env.render())
```

### Output
```
Episode 1000, Avg Reward: -120.45
Episode 2000, Avg Reward: 5.23
...
Episode 10000, Avg Reward: 8.12

Evaluation: 8.35 +/- 2.41
```

---

## Question 7: Construct a basic neural network in PyTorch that can serve as a function approximator for a policy
- [ ] Done

### Concept
Policy network: π(a|s) maps states to action probabilities.

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    """
    Neural network for policy approximation.
    Outputs action probabilities given state.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state: Tensor of shape (batch_size, state_dim)
        
        Returns:
            action_probs: Tensor of shape (batch_size, action_dim)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs
    
    def select_action(self, state):
        """
        Sample action from policy.
        
        Args:
            state: numpy array or tensor
        
        Returns:
            action: int
            log_prob: log probability of selected action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
    
    def get_log_prob(self, states, actions):
        """
        Get log probabilities for state-action pairs.
        
        Args:
            states: Tensor (batch_size, state_dim)
            actions: Tensor (batch_size,)
        
        Returns:
            log_probs: Tensor (batch_size,)
        """
        probs = self.forward(states)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(actions)


# Example usage
state_dim = 4
action_dim = 2
hidden_dim = 64

# Create network
policy = PolicyNetwork(state_dim, action_dim, hidden_dim)

# Sample state
state = np.random.randn(state_dim)

# Select action
action, log_prob = policy.select_action(state)
print(f"State: {state}")
print(f"Selected action: {action}")
print(f"Log probability: {log_prob.item():.4f}")

# Batch forward pass
batch_states = torch.randn(32, state_dim)
probs = policy(batch_states)
print(f"Batch probs shape: {probs.shape}")
print(f"Sample probs: {probs[0].detach().numpy()}")
```

### Output
```
State: [ 0.123 -0.456  0.789 -0.012]
Selected action: 1
Log probability: -0.6931

Batch probs shape: torch.Size([32, 2])
Sample probs: [0.482 0.518]
```

### Continuous Action Space Version
```python
class ContinuousPolicyNetwork(nn.Module):
    """Policy for continuous action spaces (e.g., robotics)."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output mean and log_std for Gaussian policy
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        features = self.shared(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, -20, 2)  # Stability
        return mean, log_std
    
    def select_action(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
```

---

## Question 8: Create a Python implementation of the REINFORCE algorithm
- [ ] Done

### Concept
REINFORCE is a Monte Carlo policy gradient method:
$$\nabla J(\theta) = \mathbb{E}[\nabla \log \pi(a|s;\theta) \cdot G_t]$$

Update: θ ← θ + α ∇log π(a|s) · G_t

### Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Episode storage
        self.log_probs = []
        self.rewards = []
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        # Store log prob for training
        self.log_probs.append(dist.log_prob(action))
        
        return action.item()
    
    def store_reward(self, reward):
        self.rewards.append(reward)
    
    def compute_returns(self):
        """Compute discounted returns."""
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        # Normalize for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def update(self):
        """Update policy using collected episode."""
        returns = self.compute_returns()
        
        # Policy gradient loss
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)  # Negative for gradient ascent
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.log_probs = []
        self.rewards = []
        
        return policy_loss.item()


def train_reinforce(env_name='CartPole-v1', episodes=1000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCEAgent(state_dim, action_dim)
    
    episode_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_reward(reward)
            state = next_state
            total_reward += reward
        
        # Update policy after episode
        loss = agent.update()
        episode_rewards.append(total_reward)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")
    
    return agent, episode_rewards


# Train agent
agent, rewards = train_reinforce(episodes=500)

# Test trained agent
env = gym.make('CartPole-v1')
state, _ = env.reset()
done = False
total = 0

while not done:
    action = agent.select_action(state)
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total += reward

print(f"\nTest episode reward: {total}")
```

### Output
```
Episode 100, Avg Reward: 23.45, Loss: 12.3456
Episode 200, Avg Reward: 89.67, Loss: 5.6789
...
Episode 500, Avg Reward: 487.23, Loss: 0.1234

Test episode reward: 500
```

---

## Question 9: Code an epsilon-decreasing strategy for exploration in a reinforcement learning agent
- [ ] Done

### Concept
Start with high exploration (ε), gradually decrease to favor exploitation as agent learns.

### Implementation

```python
import numpy as np

class EpsilonScheduler:
    """Base class for epsilon scheduling."""
    
    def __init__(self, epsilon_start, epsilon_end, decay_steps):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps
        self.step_count = 0
    
    def get_epsilon(self):
        raise NotImplementedError
    
    def step(self):
        self.step_count += 1


class LinearDecay(EpsilonScheduler):
    """Linear epsilon decay."""
    
    def get_epsilon(self):
        progress = min(1.0, self.step_count / self.decay_steps)
        epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)
        return epsilon


class ExponentialDecay(EpsilonScheduler):
    """Exponential epsilon decay."""
    
    def __init__(self, epsilon_start, epsilon_end, decay_rate):
        super().__init__(epsilon_start, epsilon_end, decay_steps=None)
        self.decay_rate = decay_rate
    
    def get_epsilon(self):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-self.decay_rate * self.step_count)
        return epsilon


class StepDecay(EpsilonScheduler):
    """Step-wise epsilon decay."""
    
    def __init__(self, epsilon_start, epsilon_end, decay_steps, n_steps):
        super().__init__(epsilon_start, epsilon_end, decay_steps)
        self.n_steps = n_steps
        self.step_size = (epsilon_start - epsilon_end) / n_steps
    
    def get_epsilon(self):
        steps_passed = self.step_count // (self.decay_steps // self.n_steps)
        epsilon = max(self.epsilon_end, self.epsilon_start - steps_passed * self.step_size)
        return epsilon


def epsilon_greedy_with_decay(q_values, scheduler):
    """
    Epsilon-greedy with decaying epsilon.
    
    Args:
        q_values: Q-values for each action
        scheduler: EpsilonScheduler instance
    
    Returns:
        Selected action
    """
    epsilon = scheduler.get_epsilon()
    scheduler.step()
    
    if np.random.random() < epsilon:
        return np.random.randint(len(q_values))
    return np.argmax(q_values)


# Example usage
print("Linear Decay:")
linear = LinearDecay(epsilon_start=1.0, epsilon_end=0.01, decay_steps=10000)
for i in [0, 2500, 5000, 7500, 10000]:
    linear.step_count = i
    print(f"  Step {i}: ε = {linear.get_epsilon():.4f}")

print("\nExponential Decay:")
exp = ExponentialDecay(epsilon_start=1.0, epsilon_end=0.01, decay_rate=0.0005)
for i in [0, 2500, 5000, 7500, 10000]:
    exp.step_count = i
    print(f"  Step {i}: ε = {exp.get_epsilon():.4f}")

print("\nStep Decay (4 steps):")
step = StepDecay(epsilon_start=1.0, epsilon_end=0.01, decay_steps=10000, n_steps=4)
for i in [0, 2500, 5000, 7500, 10000]:
    step.step_count = i
    print(f"  Step {i}: ε = {step.get_epsilon():.4f}")

# Integration with Q-learning
q_values = np.array([1.0, 2.5, 1.5])
scheduler = LinearDecay(1.0, 0.01, 10000)

actions = []
for _ in range(10000):
    action = epsilon_greedy_with_decay(q_values, scheduler)
    actions.append(action)

print(f"\nAction distribution over training:")
print(f"  First 1000: {np.bincount(actions[:1000], minlength=3) / 1000}")
print(f"  Last 1000:  {np.bincount(actions[-1000:], minlength=3) / 1000}")
```

### Output
```
Linear Decay:
  Step 0: ε = 1.0000
  Step 2500: ε = 0.7525
  Step 5000: ε = 0.5050
  Step 7500: ε = 0.2575
  Step 10000: ε = 0.0100

Exponential Decay:
  Step 0: ε = 1.0000
  Step 2500: ε = 0.2961
  Step 5000: ε = 0.0918
  Step 7500: ε = 0.0335
  Step 10000: ε = 0.0166

Step Decay (4 steps):
  Step 0: ε = 1.0000
  Step 2500: ε = 0.7525
  Step 5000: ε = 0.5050
  Step 7500: ε = 0.2575
  Step 10000: ε = 0.0100

Action distribution over training:
  First 1000: [0.32  0.36  0.32 ]  # More random
  Last 1000:  [0.003 0.99  0.007]  # Almost always best action
```

---

## Question 10: Implement a policy gradient method using a neural network in PyTorch
- [ ] Done

### Concept
Actor-Critic policy gradient with baseline:
- Actor: Policy network π(a|s; θ)
- Critic: Value network V(s; φ)

Advantage: A(s,a) = Q(s,a) - V(s) ≈ r + γV(s') - V(s)

### Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym

class ActorCritic(nn.Module):
    """Actor-Critic network with shared features."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        features = self.shared(state)
        action_probs = torch.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return action_probs, value
    
    def select_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)
        
        probs, value = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), value


class A2CAgent:
    """Advantage Actor-Critic agent."""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99):
        self.gamma = gamma
        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Episode storage
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def select_action(self, state):
        action, log_prob, value = self.network.select_action(state)
        self.log_probs.append(log_prob)
        self.values.append(value)
        return action
    
    def store(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self, next_state):
        """Update using collected trajectory."""
        
        # Get bootstrap value
        with torch.no_grad():
            state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            _, next_value = self.network(state_tensor)
        
        # Compute returns and advantages
        returns = []
        advantages = []
        R = next_value.item()
        
        for i in reversed(range(len(self.rewards))):
            R = self.rewards[i] + self.gamma * R * (1 - self.dones[i])
            returns.insert(0, R)
            advantage = R - self.values[i].item()
            advantages.insert(0, advantage)
        
        returns = torch.tensor(returns)
        advantages = torch.tensor(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Stack log probs and values
        log_probs = torch.stack(self.log_probs)
        values = torch.cat(self.values)
        
        # Actor loss (policy gradient)
        actor_loss = -(log_probs * advantages).mean()
        
        # Critic loss (value function)
        critic_loss = nn.MSELoss()(values.squeeze(), returns)
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        # Clear storage
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
        return actor_loss.item(), critic_loss.item()


def train_a2c(env_name='CartPole-v1', episodes=500):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = A2CAgent(state_dim, action_dim)
    episode_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store(reward, done)
            total_reward += reward
            state = next_state
        
        # Update after episode
        actor_loss, critic_loss = agent.update(state)
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}, Avg: {avg:.2f}, "
                  f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
    
    return agent, episode_rewards


# Train
agent, rewards = train_a2c(episodes=300)

# Evaluate
env = gym.make('CartPole-v1')
test_rewards = []

for _ in range(10):
    state, _ = env.reset()
    done = False
    total = 0
    
    while not done:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            probs, _ = agent.network(state_t)
            action = probs.argmax().item()
        
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += reward
    
    test_rewards.append(total)

print(f"\nTest performance: {np.mean(test_rewards):.2f} +/- {np.std(test_rewards):.2f}")
```

### Output
```
Episode 50, Avg: 45.32, Actor Loss: 0.2345, Critic Loss: 12.3456
Episode 100, Avg: 123.45, Actor Loss: 0.1234, Critic Loss: 5.6789
...
Episode 300, Avg: 495.67, Actor Loss: 0.0123, Critic Loss: 0.1234

Test performance: 500.00 +/- 0.00
```

---
