# Reinforcement Learning Interview Questions - Scenario-Based Questions

---

## Question 1: Discuss the improvements of Double DQN over the standard DQN
- [ ] Done

### The Problem with Standard DQN
Standard DQN uses the same network to both select and evaluate actions in the target:

$$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

This leads to **overestimation bias** because:
- max operation is applied over noisy estimates
- Noise causes consistent upward bias
- Errors compound during bootstrapping

### Double DQN Solution
Decouple action selection from action evaluation:

**Action Selection:** Use online network (θ)
**Action Evaluation:** Use target network (θ⁻)

$$y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$$

### Step-by-Step Comparison

| Step | DQN | Double DQN |
|------|-----|------------|
| 1 | Find best action: a* = argmax Q(s', a'; θ⁻) | Find best action: a* = argmax Q(s', a'; θ) |
| 2 | Get value: Q(s', a*; θ⁻) | Get value: Q(s', a*; θ⁻) |
| 3 | Both use same network | Different networks |

### Implementation
```python
# Standard DQN target
def dqn_target(next_states, rewards, dones, gamma, target_net):
    with torch.no_grad():
        max_q = target_net(next_states).max(dim=1)[0]
        targets = rewards + gamma * max_q * (1 - dones)
    return targets

# Double DQN target
def double_dqn_target(next_states, rewards, dones, gamma, q_net, target_net):
    with torch.no_grad():
        # Action selection with online network
        best_actions = q_net(next_states).argmax(dim=1, keepdim=True)
        # Action evaluation with target network
        q_values = target_net(next_states).gather(1, best_actions).squeeze()
        targets = rewards + gamma * q_values * (1 - dones)
    return targets
```

### Why It Works
- Online network is updated frequently → may have some noise
- Target network is stable → provides grounded evaluation
- Even if online network picks wrong action due to noise, target network gives realistic value
- Reduces overestimation while maintaining learning signal

### Empirical Improvements
| Metric | DQN | Double DQN |
|--------|-----|------------|
| Q-value estimates | Overestimated | More accurate |
| Performance | Good | Often better |
| Stability | Can be unstable | More stable |

---

## Question 2: Discuss the use of hierarchical reinforcement learning for complex tasks
- [ ] Done

### Why Hierarchical RL?
Complex tasks require:
- Long-horizon planning
- Temporal abstraction
- Transfer of skills
- Structured exploration

Flat RL struggles with sparse rewards and long credit assignment chains.

### The Options Framework
**Option** = (I, π, β)
- I: Initiation set (where option can start)
- π: Intra-option policy (how to execute)
- β: Termination condition (when to stop)

```
High-level policy selects options
Each option executes until termination
Results in temporal abstraction
```

### Feudal Networks / Hierarchical Architecture
```
Manager (high-level):
- Sets goals for workers
- Operates at slower timescale
- Plans over abstract space

Worker (low-level):
- Executes primitive actions
- Tries to achieve manager's goal
- Operates at every timestep
```

### Implementation Example
```python
class HierarchicalAgent:
    def __init__(self):
        self.manager = ManagerNetwork()  # Goal setter
        self.worker = WorkerNetwork()    # Action taker
    
    def act(self, state, timestep):
        # Manager sets goal every k steps
        if timestep % self.manager_freq == 0:
            self.current_goal = self.manager(state)
        
        # Worker takes action conditioned on goal
        action = self.worker(state, self.current_goal)
        return action
    
    def train(self, trajectory):
        # Train worker on intrinsic reward (reaching goals)
        worker_reward = goal_reaching_reward(trajectory, self.current_goal)
        self.worker.update(trajectory, worker_reward)
        
        # Train manager on extrinsic reward
        self.manager.update(trajectory, extrinsic_reward)
```

### Benefits

| Benefit | Explanation |
|---------|-------------|
| **Temporal Abstraction** | Think in terms of subgoals, not steps |
| **Transfer Learning** | Reuse low-level skills |
| **Exploration** | Structured exploration via subgoals |
| **Sample Efficiency** | Learn reusable components |

### Challenges
- How to discover/learn options?
- Non-stationarity between levels
- Credit assignment across hierarchy
- Balancing abstraction levels

### Applications
- Robotics (navigate → grasp → assemble)
- Game playing (resource gathering → attack → defend)
- Dialogue systems (topic selection → response generation)

---

## Question 3: How would you use reinforcement learning to optimize traffic signal control in a simulated city environment?
- [ ] Done

### Problem Formulation

**State Representation:**
```python
state = {
    'queue_lengths': [vehicles_waiting_per_lane],
    'phase_duration': current_green_time,
    'current_phase': one_hot_phase,
    'time_of_day': normalized_time,
    'neighboring_signals': [neighbor_states],
    'vehicle_speeds': [avg_speed_per_lane]
}
```

**Action Space:**
- Option 1: Choose next phase (discrete)
- Option 2: Extend/switch current phase (binary)
- Option 3: Duration of each phase (continuous)

**Reward Design:**
```python
def compute_reward(intersection_state):
    # Primary: Minimize waiting time
    wait_penalty = -sum(queue_lengths * wait_times)
    
    # Secondary: Throughput
    throughput_bonus = vehicles_passed * throughput_weight
    
    # Safety: Penalize very short phases
    safety_penalty = -1 if phase_duration < min_duration else 0
    
    return wait_penalty + throughput_bonus + safety_penalty
```

### Multi-Agent Coordination
Each intersection is an agent. Coordination approaches:

| Approach | Description |
|----------|-------------|
| **Independent** | Each intersection learns independently |
| **Centralized** | One policy controls all signals |
| **Networked** | Share information with neighbors |
| **MARL** | Cooperative multi-agent RL |

### Architecture
```python
class TrafficSignalAgent:
    def __init__(self, intersection_id, neighbors):
        self.id = intersection_id
        self.neighbors = neighbors
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_phases)
        )
```

### Training Pipeline
```
1. Build simulation (SUMO, CityFlow)
2. Define realistic traffic patterns
3. Train agents with varying demand
4. Evaluate on held-out scenarios
5. Compare to baseline (fixed timing, actuated)
```

### Simulation Setup
```python
# Using SUMO simulator
def create_traffic_env():
    env = SumoEnvironment(
        net_file='city_network.net.xml',
        route_file='traffic_demand.rou.xml',
        delta_time=5,  # Decision interval
        min_green=10,
        max_green=50
    )
    return env
```

### Evaluation Metrics
- Average waiting time
- Total throughput
- Queue length
- Travel time
- CO2 emissions (environmental)

### Challenges
- Partial observability
- Non-stationarity (traffic patterns change)
- Multi-agent credit assignment
- Safety constraints (minimum green times)
- Sim-to-real transfer

---

## Question 4: Discuss the application of reinforcement learning in personalization and recommendation systems
- [ ] Done

### Problem Setup

**State:**
```python
state = {
    'user_history': [clicked_items, watched_time, purchases],
    'user_features': [demographics, preferences],
    'context': [time, device, location],
    'candidate_items': [item_embeddings]
}
```

**Action:** Select item(s) to recommend

**Reward:**
```python
# Immediate rewards
click_reward = 1 if clicked else 0
watch_reward = watch_time / video_length
purchase_reward = purchase_amount

# Long-term considerations
engagement_reward = session_length
retention_reward = user_returned_next_day
```

### Why RL Over Supervised Learning?

| Aspect | Supervised | RL |
|--------|------------|-----|
| **Objective** | Predict clicks | Maximize long-term engagement |
| **Feedback** | Static labels | Sequential interactions |
| **Exploration** | Random/heuristic | Principled exploration |
| **Long-term** | Ignores future | Optimizes cumulative reward |

### Exploration vs Exploitation
```python
class RecommenderAgent:
    def recommend(self, user_state, epsilon=0.1):
        if random.random() < epsilon:
            # Explore: Try new items
            return self.explore(user_state)
        else:
            # Exploit: Best known recommendation
            return self.policy(user_state).argmax()
    
    def explore(self, user_state):
        # Thompson sampling / UCB / ε-greedy
        uncertainty = self.get_uncertainty(user_state)
        return (self.policy(user_state) + uncertainty * exploration_bonus).argmax()
```

### Handling Large Action Spaces
Millions of items → Can't enumerate all actions

**Solutions:**
```python
# Two-stage: Retrieve then rank
class TwoStageRecommender:
    def recommend(self, state):
        # Stage 1: Candidate generation (fast)
        candidates = self.retriever.get_candidates(state, k=1000)
        
        # Stage 2: RL ranking (accurate)
        scores = self.policy(state, candidates)
        return candidates[scores.argmax()]
```

### Slate Recommendation
Recommend multiple items as a slate:

```python
# Sequential selection
def select_slate(state, slate_size):
    slate = []
    for i in range(slate_size):
        item = policy(state, slate)  # Conditioned on current slate
        slate.append(item)
    return slate
```

### Challenges

| Challenge | Solution |
|-----------|----------|
| **Delayed reward** | Credit assignment, TD learning |
| **Sparse feedback** | Auxiliary rewards |
| **Cold start** | Exploration bonuses |
| **Position bias** | Inverse propensity weighting |
| **Filter bubbles** | Diversity constraints |

### Offline RL for Recommendations
Train on logged data:
```python
# Counterfactual policy evaluation
def evaluate_new_policy(logged_data, new_policy):
    total_reward = 0
    for (state, action, reward, propensity) in logged_data:
        new_action = new_policy(state)
        if new_action == action:
            # Importance sampling
            total_reward += reward / propensity
    return total_reward / len(logged_data)
```

---

## Question 5: How would you approach the problem of tuning hyperparameters of a reinforcement learning model?
- [ ] Done

### Key Hyperparameters in RL

| Category | Hyperparameters |
|----------|-----------------|
| **Learning** | Learning rate, batch size, optimizer |
| **RL-specific** | Discount γ, GAE λ, entropy coefficient |
| **Exploration** | ε schedule, temperature |
| **Architecture** | Hidden layers, units, activation |
| **Algorithm** | PPO clip, target update frequency |

### Tuning Approaches

**1. Grid/Random Search:**
```python
# Random search often more effective than grid
param_distributions = {
    'learning_rate': loguniform(1e-5, 1e-2),
    'gamma': uniform(0.9, 0.999),
    'hidden_size': choice([64, 128, 256]),
    'batch_size': choice([32, 64, 128, 256])
}

for trial in range(n_trials):
    params = sample_from(param_distributions)
    performance = train_and_evaluate(params)
    log_result(params, performance)
```

**2. Bayesian Optimization:**
```python
# Using Optuna
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
    hidden = trial.suggest_categorical('hidden', [64, 128, 256])
    
    agent = create_agent(lr=lr, gamma=gamma, hidden_size=hidden)
    return evaluate_agent(agent)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

**3. Population-Based Training (PBT):**
```
1. Initialize population of agents with different hyperparams
2. Train all in parallel
3. Periodically:
   - Evaluate all agents
   - Poor agents copy weights from good agents
   - Mutate hyperparameters of poor agents
4. Hyperparameters evolve during training
```

### Practical Guidelines

| Hyperparameter | Starting Point | Tuning Range |
|----------------|----------------|--------------|
| Learning rate | 3e-4 | 1e-5 to 1e-2 |
| Discount γ | 0.99 | 0.9 to 0.999 |
| Batch size | 64-256 | 32 to 2048 |
| Hidden units | 256 | 64 to 512 |
| Entropy coef | 0.01 | 0.001 to 0.1 |

### Evaluation Challenges

```python
def robust_evaluation(agent, env, n_episodes=100):
    # Multiple seeds for reliability
    returns = []
    for seed in range(n_episodes):
        env.seed(seed)
        episode_return = run_episode(agent, env)
        returns.append(episode_return)
    
    return {
        'mean': np.mean(returns),
        'std': np.std(returns),
        'median': np.median(returns)
    }
```

### Tips
1. Start with published hyperparameters
2. Tune learning rate first
3. Use multiple random seeds
4. Log everything for analysis
5. Consider computational budget
6. Use early stopping for bad configs

---

## Question 6: Propose a reinforcement learning framework for an energy management system in smart grids
- [ ] Done

### Problem Formulation

**State:**
```python
state = {
    'current_demand': total_load_kw,
    'renewable_generation': [solar_kw, wind_kw],
    'storage_level': battery_soc,  # State of charge
    'grid_price': current_price,
    'forecast': {
        'demand_next_24h': [...],
        'solar_next_24h': [...],
        'price_next_24h': [...]
    },
    'time_features': [hour, day_of_week, season]
}
```

**Actions:**
```python
actions = {
    'battery': [-1, 1],  # Discharge/charge rate
    'grid_import': [0, max_import],  # Buy from grid
    'grid_export': [0, max_export],  # Sell to grid
    'load_shifting': [delay_loads, advance_loads]
}
```

**Reward:**
```python
def compute_reward(state, action):
    # Cost minimization
    import_cost = grid_import * price
    export_revenue = grid_export * sell_price
    
    # Reliability penalty
    unmet_demand_penalty = -100 * unmet_demand
    
    # Battery degradation
    degradation_cost = battery_cycles * degradation_rate
    
    # Environmental bonus
    green_bonus = renewable_usage * green_weight
    
    return -import_cost + export_revenue + unmet_demand_penalty 
           - degradation_cost + green_bonus
```

### System Architecture
```
┌─────────────────────────────────────────────┐
│           Smart Grid RL Framework           │
├─────────────────────────────────────────────┤
│  Forecasting Module                         │
│  - Demand prediction (LSTM/Transformer)     │
│  - Renewable generation forecast            │
│  - Price prediction                         │
├─────────────────────────────────────────────┤
│  RL Agent                                   │
│  - State: Current + Forecast features       │
│  - Policy: Actor-Critic (SAC/PPO)           │
│  - Handles continuous action space          │
├─────────────────────────────────────────────┤
│  Safety Layer                               │
│  - Enforce physical constraints             │
│  - Ensure reliability                       │
│  - Emergency protocols                      │
└─────────────────────────────────────────────┘
```

### Training Approach
```python
class SmartGridAgent:
    def __init__(self):
        self.actor = ContinuousPolicy(state_dim, action_dim)
        self.critic = ValueNetwork(state_dim)
        
    def train(self, historical_data):
        # Create environment from historical data
        env = SmartGridEnv(historical_data)
        
        for episode in range(n_episodes):
            state = env.reset()
            while not done:
                action = self.actor(state)
                action = self.safety_filter(action, state)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state
```

### Safety Constraints
```python
def safety_filter(self, action, state):
    # Battery limits
    action['battery'] = np.clip(
        action['battery'],
        -max_discharge_rate,
        max_charge_rate
    )
    
    # SOC limits
    if state['storage_level'] < 0.1:
        action['battery'] = max(action['battery'], 0)  # Don't discharge
    
    # Grid limits
    action['grid_import'] = min(action['grid_import'], max_import)
    
    # Ensure demand is met
    total_supply = (state['renewable'] + action['grid_import'] 
                   + action['battery_discharge'])
    if total_supply < state['demand']:
        action['grid_import'] += (state['demand'] - total_supply)
    
    return action
```

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Non-stationarity | Online adaptation, meta-learning |
| Forecast uncertainty | Robust optimization, ensemble |
| Physical constraints | Constrained RL, safety layer |
| Multi-objective | Weighted rewards, Pareto RL |

---

## Question 7: Discuss how to set up a reinforcement learning environment for teaching an AI to play chess
- [ ] Done

### Environment Design

**State Representation:**
```python
def encode_board(board):
    # Option 1: 8x8x12 tensor (one-hot per piece type)
    planes = np.zeros((12, 8, 8))
    for piece_type in range(6):  # Pawn to King
        for color in range(2):    # White, Black
            # Set 1 where piece exists
            planes[piece_type * 2 + color] = get_piece_positions(board, piece_type, color)
    
    # Additional planes for:
    # - Castling rights (4 planes)
    # - En passant square (1 plane)
    # - Repetition count
    # - Move count (50-move rule)
    
    return planes  # Shape: (18, 8, 8) typically
```

**Action Space:**
```python
# All possible moves ~4672 (from-square × to-square × promotion)
# Or use move encoding
action_space = {
    'from_square': 64,  # 8x8
    'to_square': 64,
    'promotion': 5  # None, Q, R, B, N
}

# Alternative: Policy outputs probability over all legal moves only
```

**Reward:**
```python
def get_reward(result, move_made):
    if result == 'win':
        return +1
    elif result == 'loss':
        return -1
    elif result == 'draw':
        return 0
    else:
        # Intermediate reward (optional, often not used)
        return 0  # Sparse reward works better for games
```

### Architecture (AlphaZero-style)
```python
class ChessNet(nn.Module):
    def __init__(self):
        self.conv_blocks = nn.Sequential(
            ConvBlock(18, 256),  # Input channels = state planes
            *[ResidualBlock(256) for _ in range(19)]  # 19 residual blocks
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, 1),
            nn.Flatten(),
            nn.Linear(128, 4672)  # All possible moves
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, state):
        features = self.conv_blocks(state)
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value
```

### Training Pipeline (AlphaZero Approach)
```
1. Self-play: Generate games using MCTS + neural network
2. Store games in replay buffer
3. Train network on stored games
4. Repeat
```

```python
def self_play_game(network):
    game = Chess()
    examples = []
    
    while not game.is_over():
        # MCTS for move selection
        pi, v = mcts(game, network, simulations=800)
        
        # Store training example
        examples.append((game.state, pi, None))  # Value filled later
        
        # Make move
        action = sample_from(pi)
        game.make_move(action)
    
    # Assign game result to all positions
    result = game.result()
    return [(s, p, result) for s, p, _ in examples]
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **MCTS** | Search for best moves |
| **Neural Network** | Guide search (policy + value) |
| **Self-play** | Generate training data |
| **Curriculum** | Start simple, increase difficulty |

### Training Considerations
- Massive compute required (TPUs/GPUs)
- Self-play parallelization
- Efficient move generation
- Opening book vs learning from scratch

---

## Question 8: Discuss the challenges of safe reinforcement learning when deploying models in sensitive areas
- [ ] Done

### Healthcare Challenges

| Challenge | Description |
|-----------|-------------|
| **Patient Safety** | Wrong dosage could be fatal |
| **Regulatory** | FDA approval, clinical trials |
| **Interpretability** | Must explain decisions to doctors |
| **Data Limitations** | Rare conditions, limited data |
| **Irreversibility** | Some actions cannot be undone |

**Example - Sepsis Treatment:**
```python
class SafeSepsisAgent:
    def recommend_treatment(self, patient_state):
        action = self.policy(patient_state)
        
        # Safety checks
        if dosage_too_high(action, patient_state):
            action = reduce_dosage(action)
        
        if contraindicated(action, patient_history):
            action = alternative_treatment(action)
        
        # Uncertainty quantification
        confidence = self.get_confidence(patient_state)
        if confidence < THRESHOLD:
            return DEFER_TO_PHYSICIAN
        
        return action
```

### Autonomous Driving Challenges

| Challenge | Solution Approach |
|-----------|-------------------|
| **Edge cases** | Simulation + real-world testing |
| **Reaction time** | Real-time constraints |
| **Sensor failure** | Redundancy, graceful degradation |
| **Liability** | Clear responsibility chain |

**Safety Framework:**
```python
class AutonomousVehicleController:
    def __init__(self):
        self.learned_policy = NeuralPolicy()
        self.safety_controller = RuleBasedSafety()
        
    def act(self, observation):
        # Get learned action
        action = self.learned_policy(observation)
        
        # Safety override
        if self.safety_controller.unsafe(observation, action):
            action = self.safety_controller.safe_action(observation)
        
        # Emergency protocols
        if self.detect_imminent_collision(observation):
            action = self.emergency_brake()
        
        return action
```

### General Safe RL Approaches

**1. Constrained RL:**
$$\max_\pi J(\pi) \text{ subject to } C_i(\pi) \leq d_i$$

**2. Conservative Policy Optimization:**
```python
# Stay close to known-safe policy
loss = policy_loss + lambda * kl_divergence(new_policy, safe_baseline)
```

**3. Formal Verification:**
- Prove safety properties mathematically
- Certify behavior in bounded regions

**4. Human Oversight:**
```python
def get_action_with_oversight(state, policy, human):
    action = policy(state)
    confidence = policy.confidence(state)
    
    if confidence < threshold or is_high_stakes(state):
        action = human.review_and_approve(state, action)
    
    return action
```

### Deployment Best Practices
1. Extensive simulation testing
2. Gradual rollout
3. Continuous monitoring
4. Clear escalation procedures
5. Regular audits
6. Fail-safe defaults

---

## Question 9: Discuss the importance of fairness and bias considerations in reinforcement learning
- [ ] Done

### Sources of Bias in RL

| Source | Example |
|--------|---------|
| **Training Data** | Historical data reflects past discrimination |
| **Reward Design** | Reward function encodes biased objectives |
| **State Features** | Using protected attributes (directly/proxy) |
| **Environment** | Simulations may not represent all groups |

### Examples of Unfair RL

**Healthcare:**
- Treatment recommendations biased toward majority populations
- Less effective for underrepresented groups

**Hiring/Loans:**
- RL-based systems perpetuate historical bias
- Feedback loops amplify discrimination

**Content Recommendation:**
- Filter bubbles disproportionately affect certain groups
- Engagement optimization exploits vulnerabilities

### Fairness Definitions

**Group Fairness:**
$$P(\text{positive outcome} | \text{group A}) = P(\text{positive outcome} | \text{group B})$$

**Individual Fairness:**
Similar individuals should be treated similarly.

**Counterfactual Fairness:**
Outcome should be the same if individual's protected attribute were different.

### Fair RL Approaches

**1. Constrained Optimization:**
```python
# Add fairness constraint to RL objective
def fair_reward(state, action, group):
    base_reward = compute_reward(state, action)
    fairness_penalty = compute_disparity(action, group)
    return base_reward - lambda * fairness_penalty
```

**2. Balanced Replay:**
```python
# Ensure training data represents all groups
def sample_batch(replay_buffer, groups):
    batch = []
    for group in groups:
        group_samples = replay_buffer.sample_from_group(group, n=batch_size//len(groups))
        batch.extend(group_samples)
    return batch
```

**3. Adversarial Debiasing:**
```python
class FairPolicy(nn.Module):
    def __init__(self):
        self.encoder = StateEncoder()
        self.policy_head = PolicyNetwork()
        self.adversary = GroupPredictor()  # Tries to predict group from representation
    
    def forward(self, state):
        representation = self.encoder(state)
        action = self.policy_head(representation)
        
        # Adversarial loss: Make representation group-invariant
        group_pred = self.adversary(representation)
        return action, group_pred
```

### Evaluation Framework
```python
def evaluate_fairness(policy, test_data, groups):
    metrics = {}
    for group in groups:
        group_data = test_data[test_data.group == group]
        metrics[group] = {
            'reward': average_reward(policy, group_data),
            'success_rate': success_rate(policy, group_data)
        }
    
    # Compute disparity metrics
    metrics['demographic_parity'] = max_group_difference(metrics, 'success_rate')
    metrics['equalized_odds'] = compute_equalized_odds(metrics)
    
    return metrics
```

### Best Practices
1. Define fairness metrics early
2. Audit training data for bias
3. Test on disaggregated groups
4. Monitor deployed system for emergent bias
5. Include diverse stakeholders in design

---

## Question 10: Discuss a recent research paper on reinforcement learning that caught your attention and its implications
- [ ] Done

### Example: Decision Transformer (Chen et al., 2021)

**Key Idea:**
Frame RL as sequence modeling problem. Use Transformer architecture to predict actions conditioned on desired return.

### How It Works
```
Input sequence: (R̂₁, s₁, a₁, R̂₂, s₂, a₂, ..., R̂ₜ, sₜ)
Output: aₜ

R̂ₜ = desired return-to-go (total future reward wanted)
```

At test time, condition on high return to get good actions.

### Architecture
```python
class DecisionTransformer(nn.Module):
    def __init__(self):
        self.embed_return = nn.Linear(1, embed_dim)
        self.embed_state = StateEncoder(state_dim, embed_dim)
        self.embed_action = nn.Linear(action_dim, embed_dim)
        
        self.transformer = TransformerDecoder(
            embed_dim=embed_dim,
            n_heads=8,
            n_layers=6
        )
        
        self.predict_action = nn.Linear(embed_dim, action_dim)
    
    def forward(self, returns, states, actions, timesteps):
        # Embed each modality
        return_embeddings = self.embed_return(returns)
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        
        # Interleave: R, s, a, R, s, a, ...
        sequence = interleave(return_embeddings, state_embeddings, action_embeddings)
        
        # Transformer forward
        output = self.transformer(sequence)
        
        # Predict actions
        action_preds = self.predict_action(output[:, 1::3, :])  # State positions
        return action_preds
```

### Implications

| Implication | Description |
|-------------|-------------|
| **Offline RL simplified** | No complex value estimation |
| **Generalization** | Leverage LLM advances |
| **Multi-task** | Same architecture across tasks |
| **Controllability** | Specify desired performance |

### Why It Matters

**1. Unifies RL and Sequence Modeling:**
- Transfer pretrained Transformers to RL
- Benefit from scaling laws

**2. Offline RL Without Bellman:**
- Avoids bootstrapping issues
- More stable training

**3. Goal-Conditioned:**
```python
# At test time
desired_return = high_return  # Want good performance
for t in range(episode_length):
    action = model(desired_return, state_history, action_history)
    state, reward = env.step(action)
    desired_return -= reward  # Update remaining return needed
```

### Limitations
- Needs diverse offline data with varying returns
- May not extrapolate beyond data
- Computational cost of Transformers

### Follow-up Work
- Gato (DeepMind): Generalist agent using similar ideas
- Online Decision Transformer
- Trajectory Transformer

---
