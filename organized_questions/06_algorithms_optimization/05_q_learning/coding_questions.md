# Q Learning Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Implement a basic Q-learning agent that learns to navigate a simple gridworld environment.**

**Answer:**

```python
import numpy as np
import random

# Simple 4x4 GridWorld
# S = Start (0,0), G = Goal (3,3)
# Agent must reach goal

GRID_SIZE = 4
ACTIONS = ['up', 'down', 'left', 'right']
NUM_ACTIONS = 4

def get_next_state(state, action):
    row, col = state // GRID_SIZE, state % GRID_SIZE
    if action == 0: row = max(0, row - 1)       # up
    elif action == 1: row = min(GRID_SIZE-1, row + 1)  # down
    elif action == 2: col = max(0, col - 1)     # left
    elif action == 3: col = min(GRID_SIZE-1, col + 1)  # right
    return row * GRID_SIZE + col

def get_reward(state):
    if state == GRID_SIZE * GRID_SIZE - 1:  # Goal state
        return 10
    return -1  # Small penalty per step

# Initialize Q-table
num_states = GRID_SIZE * GRID_SIZE
Q = np.zeros((num_states, NUM_ACTIONS))

# Hyperparameters
alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 0.1    # Exploration rate
episodes = 500

# Training loop
for ep in range(episodes):
    state = 0  # Start state
    done = False
    
    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, NUM_ACTIONS - 1)
        else:
            action = np.argmax(Q[state])
        
        # Take action
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)
        
        # Q-learning update
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        
        state = next_state
        if state == num_states - 1:  # Goal reached
            done = True

# Test the learned policy
print("Learned Q-table:")
print(Q.round(2))
print("\nOptimal policy:")
for s in range(num_states):
    print(f"State {s}: {ACTIONS[np.argmax(Q[s])]}")
```

---

## Question 2
- [ ] Done

**Write a function that updates the Q-table given a state, action, reward, and next state.**

**Answer:**

```python
import numpy as np

def q_learning_update(Q, state, action, reward, next_state, alpha=0.1, gamma=0.9, done=False):
    """
    Update Q-table using Q-learning update rule
    
    Parameters:
    - Q: Q-table (2D array: states x actions)
    - state: current state index
    - action: action taken
    - reward: reward received
    - next_state: resulting state index
    - alpha: learning rate
    - gamma: discount factor
    - done: whether episode ended
    
    Returns:
    - Updated Q-table
    - TD error (for monitoring)
    """
    # Target value
    if done:
        target = reward  # No future rewards if episode ended
    else:
        target = reward + gamma * np.max(Q[next_state])
    
    # TD error
    td_error = target - Q[state, action]
    
    # Update Q-value
    Q[state, action] += alpha * td_error
    
    return Q, td_error


# Example usage
Q = np.zeros((10, 4))  # 10 states, 4 actions

# Simulated transition
state = 3
action = 1
reward = -1
next_state = 4
done = False

Q, td_error = q_learning_update(Q, state, action, reward, next_state)
print(f"Updated Q[{state}, {action}] = {Q[state, action]:.4f}")
print(f"TD Error: {td_error:.4f}")

# Terminal state example
Q, td_error = q_learning_update(Q, state=9, action=2, reward=10, next_state=9, done=True)
print(f"Terminal update Q[9, 2] = {Q[9, 2]:.4f}")
```

**Key Points:**
- TD target = r + γ × max Q(s', a') for non-terminal
- TD target = r for terminal states
- TD error measures surprise/learning signal

---

## Question 3
- [ ] Done

**Create a simulation of a Q-learning agent in a stochastic environment and show how the agent improves over time.**

**Answer:**

```python
import numpy as np
import random

# Stochastic GridWorld: actions succeed 80% of time
# 20% chance of random perpendicular movement

GRID_SIZE = 4
NUM_STATES = GRID_SIZE * GRID_SIZE
NUM_ACTIONS = 4  # up, down, left, right
SUCCESS_PROB = 0.8

def get_next_state(state, action):
    row, col = state // GRID_SIZE, state % GRID_SIZE
    
    # Stochastic: 20% chance of slipping perpendicular
    if random.random() > SUCCESS_PROB:
        action = random.choice([a for a in range(4) if a != action])
    
    if action == 0: row = max(0, row - 1)
    elif action == 1: row = min(GRID_SIZE-1, row + 1)
    elif action == 2: col = max(0, col - 1)
    elif action == 3: col = min(GRID_SIZE-1, col + 1)
    return row * GRID_SIZE + col

def get_reward(state):
    if state == NUM_STATES - 1: return 10  # Goal
    return -0.1  # Step cost

# Training
Q = np.zeros((NUM_STATES, NUM_ACTIONS))
alpha, gamma, epsilon = 0.1, 0.95, 1.0
epsilon_decay = 0.995
min_epsilon = 0.01

episode_rewards = []

for ep in range(1000):
    state = 0
    total_reward = 0
    steps = 0
    
    while state != NUM_STATES - 1 and steps < 100:
        # Epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state])
        
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)
        
        # Q-learning update
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        
        state = next_state
        total_reward += reward
        steps += 1
    
    episode_rewards.append(total_reward)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    if (ep + 1) % 200 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {ep+1}: Avg Reward (last 100) = {avg_reward:.2f}, Epsilon = {epsilon:.3f}")

# Show improvement
print(f"\nFirst 100 episodes avg: {np.mean(episode_rewards[:100]):.2f}")
print(f"Last 100 episodes avg: {np.mean(episode_rewards[-100:]):.2f}")
```

---

## Question 4
- [ ] Done

**Code a solution that demonstrates epsilon-greedy action selection in Q-learning.**

**Answer:**

```python
import numpy as np
import random

class EpsilonGreedyAgent:
    def __init__(self, num_states, num_actions, epsilon=1.0, 
                 epsilon_decay=0.995, min_epsilon=0.01):
        self.Q = np.zeros((num_states, num_actions))
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.action_counts = {'explore': 0, 'exploit': 0}
    
    def select_action(self, state):
        """
        Epsilon-greedy action selection:
        - With probability epsilon: random action (explore)
        - With probability 1-epsilon: best action (exploit)
        """
        if random.random() < self.epsilon:
            self.action_counts['explore'] += 1
            return random.randint(0, self.num_actions - 1)
        else:
            self.action_counts['exploit'] += 1
            return np.argmax(self.Q[state])
    
    def decay_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def get_stats(self):
        total = sum(self.action_counts.values())
        if total == 0:
            return 0, 0
        explore_pct = self.action_counts['explore'] / total * 100
        exploit_pct = self.action_counts['exploit'] / total * 100
        return explore_pct, exploit_pct


# Demonstration
agent = EpsilonGreedyAgent(num_states=10, num_actions=4)

# Set some Q-values
agent.Q[0] = [0.1, 0.5, 0.3, 0.2]  # Action 1 is best

print("Epsilon-Greedy Demo:")
print(f"Initial epsilon: {agent.epsilon}")
print(f"Q[0] = {agent.Q[0]}")

# Simulate action selection over episodes
for ep in range(5):
    for _ in range(100):  # 100 steps per episode
        action = agent.select_action(state=0)
    agent.decay_epsilon()
    explore_pct, exploit_pct = agent.get_stats()
    print(f"Episode {ep+1}: epsilon={agent.epsilon:.3f}, explore={explore_pct:.1f}%, exploit={exploit_pct:.1f}%")
```

**Key Concept:**
Epsilon starts high (more exploration) and decays over time (more exploitation as agent learns).

---

## Question 5
- [ ] Done

**Develop a Python script that visualizes the convergence of Q-values over episodes.**

**Answer:**

```python
import numpy as np
import random
import matplotlib.pyplot as plt

# Simple environment
NUM_STATES = 5
NUM_ACTIONS = 2
GOAL_STATE = 4

def step(state, action):
    if action == 0:  # Move right
        next_state = min(state + 1, GOAL_STATE)
    else:  # Stay
        next_state = state
    reward = 10 if next_state == GOAL_STATE else -1
    done = next_state == GOAL_STATE
    return next_state, reward, done

# Track Q-values over training
Q = np.zeros((NUM_STATES, NUM_ACTIONS))
alpha, gamma, epsilon = 0.1, 0.9, 0.3

q_history = []  # Store Q-value snapshots
episodes = 500

for ep in range(episodes):
    state = 0
    done = False
    
    while not done:
        # Epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, done = step(state, action)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state
    
    # Record Q-values at each state for action 0 (move right)
    q_history.append(Q[:, 0].copy())

# Convert to array for plotting
q_history = np.array(q_history)

# Plot convergence
plt.figure(figsize=(10, 6))
for s in range(NUM_STATES):
    plt.plot(q_history[:, s], label=f'State {s}')

plt.xlabel('Episode')
plt.ylabel('Q-value (action=right)')
plt.title('Q-Value Convergence Over Episodes')
plt.legend()
plt.grid(True)
plt.savefig('q_convergence.png')
plt.show()

print("Final Q-table:")
print(Q.round(2))
```

**What It Shows:**
- Q-values start at 0 and gradually converge to optimal values
- States closer to goal converge faster
- Stable Q-values indicate learning convergence

---

## Question 6
- [ ] Done

**Write a Python function that evaluates a Q-learning agent's policy after training.**

**Answer:**

```python
import numpy as np

def evaluate_policy(Q, env_step_fn, start_state=0, num_episodes=100, max_steps=100):
    """
    Evaluate learned policy without exploration (greedy)
    
    Parameters:
    - Q: Trained Q-table
    - env_step_fn: Function(state, action) -> (next_state, reward, done)
    - start_state: Starting state for each episode
    - num_episodes: Number of evaluation episodes
    - max_steps: Max steps per episode
    
    Returns:
    - avg_reward: Average total reward per episode
    - avg_steps: Average steps to completion
    - success_rate: Percentage of successful episodes
    """
    total_rewards = []
    steps_list = []
    successes = 0
    
    for _ in range(num_episodes):
        state = start_state
        episode_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            # Greedy action selection (no exploration)
            action = np.argmax(Q[state])
            next_state, reward, done = env_step_fn(state, action)
            episode_reward += reward
            state = next_state
            steps += 1
        
        total_rewards.append(episode_reward)
        steps_list.append(steps)
        if done and steps < max_steps:
            successes += 1
    
    return {
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_steps': np.mean(steps_list),
        'success_rate': successes / num_episodes * 100
    }


# Example usage with simple environment
def simple_env(state, action):
    next_state = min(state + 1, 4) if action == 0 else state
    reward = 10 if next_state == 4 else -1
    done = next_state == 4
    return next_state, reward, done

# Trained Q-table (example)
Q = np.array([
    [5.0, 0.0],
    [6.0, 0.0],
    [7.0, 0.0],
    [8.0, 0.0],
    [10.0, 0.0]
])

results = evaluate_policy(Q, simple_env)
print("Evaluation Results:")
print(f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
print(f"Average Steps: {results['avg_steps']:.2f}")
print(f"Success Rate: {results['success_rate']:.1f}%")
```

---

## Question 7
- [ ] Done

**Create a Q-learning agent that can solve the Taxi-v3 environment from OpenAI Gym.**

**Answer:**

```python
import numpy as np
import random
import gymnasium as gym  # or 'import gym' for older versions

# Create environment
env = gym.make('Taxi-v3')

# Initialize Q-table
num_states = env.observation_space.n  # 500 states
num_actions = env.action_space.n       # 6 actions
Q = np.zeros((num_states, num_actions))

# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
episodes = 2000

# Training
rewards_per_episode = []

for ep in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Epsilon-greedy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Q-learning update
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) * (not done) - Q[state, action]
        )
        
        state = next_state
        total_reward += reward
    
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards_per_episode.append(total_reward)
    
    if (ep + 1) % 500 == 0:
        avg = np.mean(rewards_per_episode[-100:])
        print(f"Episode {ep+1}: Avg Reward = {avg:.2f}")

# Evaluate
print("\n--- Evaluation (10 episodes) ---")
for _ in range(10):
    state, _ = env.reset()
    done = False
    total = 0
    while not done:
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += reward
    print(f"Episode reward: {total}")

env.close()
```

**Taxi-v3 Environment:**
- 500 states (taxi position, passenger location, destination)
- 6 actions (N, S, E, W, pickup, dropoff)
- Goal: Pick up passenger and drop at destination
- Optimal: ~8 reward per episode

---

## Question 8
- [ ] Done

**Implement a Q-learning solution where the agent must learn context-specific rules, such as traffic signal control with variable vehicle flow.**

**Answer:**

```python
import numpy as np
import random

class TrafficLightEnv:
    """Traffic signal control with variable vehicle flow"""
    
    def __init__(self):
        # State: (queue_north, queue_east, current_light, time_in_phase)
        self.max_queue = 10
        self.max_phase_time = 5
        self.reset()
    
    def reset(self):
        self.queue_n = random.randint(0, 5)  # North queue
        self.queue_e = random.randint(0, 5)  # East queue
        self.light = 0  # 0=North green, 1=East green
        self.phase_time = 0
        return self._get_state()
    
    def _get_state(self):
        # Discretize to state index
        return (min(self.queue_n, self.max_queue) * 11 * 2 * 6 +
                min(self.queue_e, self.max_queue) * 2 * 6 +
                self.light * 6 + min(self.phase_time, 5))
    
    def step(self, action):
        # Action: 0=keep current, 1=switch light
        if action == 1:
            self.light = 1 - self.light
            self.phase_time = 0
        else:
            self.phase_time = min(self.phase_time + 1, 5)
        
        # Process vehicles (green direction clears, others accumulate)
        if self.light == 0:  # North green
            self.queue_n = max(0, self.queue_n - 2)
            self.queue_e += random.randint(0, 2)  # Arrivals
        else:  # East green
            self.queue_e = max(0, self.queue_e - 2)
            self.queue_n += random.randint(0, 2)
        
        # Reward: negative total wait
        reward = -(self.queue_n + self.queue_e)
        
        done = False
        return self._get_state(), reward, done

# Training
env = TrafficLightEnv()
num_states = 11 * 11 * 2 * 6  # queue_n * queue_e * light * phase_time
Q = np.zeros((num_states, 2))

alpha, gamma, epsilon = 0.1, 0.9, 0.3
episodes = 2000

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    
    for _ in range(50):  # 50 time steps per episode
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, _ = env.step(action)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state
        total_reward += reward
    
    if (ep + 1) % 500 == 0:
        print(f"Episode {ep+1}: Total reward = {total_reward}")

print("\nLearned: Agent switches lights based on queue lengths")
```

---

## Question 9
- [ ] Done

**Code a Q-learning agent to solve a simple maze with dynamic obstacles, demonstrating how you manage changing environments.**

**Answer:**

```python
import numpy as np
import random

class DynamicMaze:
    """5x5 maze with moving obstacle"""
    
    def __init__(self):
        self.size = 5
        self.goal = (4, 4)
        self.reset()
    
    def reset(self):
        self.agent = [0, 0]
        self.obstacle = [2, 2]
        return self._get_state()
    
    def _get_state(self):
        # State = agent_pos + obstacle_pos
        return (self.agent[0] * 5 + self.agent[1]) * 25 + \
               (self.obstacle[0] * 5 + self.obstacle[1])
    
    def step(self, action):
        # Move agent
        new_pos = self.agent.copy()
        if action == 0: new_pos[0] = max(0, new_pos[0] - 1)  # up
        elif action == 1: new_pos[0] = min(4, new_pos[0] + 1)  # down
        elif action == 2: new_pos[1] = max(0, new_pos[1] - 1)  # left
        elif action == 3: new_pos[1] = min(4, new_pos[1] + 1)  # right
        
        # Check collision with obstacle
        if new_pos != self.obstacle:
            self.agent = new_pos
        
        # Move obstacle randomly (dynamic environment)
        moves = [[0,1], [0,-1], [1,0], [-1,0]]
        move = random.choice(moves)
        new_obs = [self.obstacle[0] + move[0], self.obstacle[1] + move[1]]
        if 0 <= new_obs[0] < 5 and 0 <= new_obs[1] < 5:
            if new_obs != self.agent and tuple(new_obs) != self.goal:
                self.obstacle = new_obs
        
        # Reward
        if tuple(self.agent) == self.goal:
            return self._get_state(), 10, True
        return self._get_state(), -0.1, False

# Q-learning with higher learning rate for non-stationary
env = DynamicMaze()
num_states = 25 * 25  # agent_pos * obstacle_pos
Q = np.zeros((num_states, 4))

alpha = 0.2  # Higher for dynamic environment
gamma = 0.9
epsilon = 0.3

for ep in range(3000):
    state = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 50:
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, done = env.step(action)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state
        steps += 1
    
    if (ep + 1) % 1000 == 0:
        print(f"Episode {ep+1}: Reached goal = {done}, Steps = {steps}")

print("\nAgent learns to navigate while avoiding moving obstacle")
```

**Key Adaptation for Dynamic Environment:**
- Higher learning rate (α=0.2) to quickly adapt to changes
- State includes obstacle position
- Agent learns general obstacle avoidance, not fixed paths

---

