# Genetic Algorithms Interview Questions - Scenario_Based Questions

## Question 1
- [ ] Done

**Discuss the concept of 'dominance' and 'epistasis' in genetic algorithms.**

**Answer:**

**Dominance:**

When one allele masks the effect of another at the same gene position (from diploid GAs).

```
Gene position: Color
Allele A (dominant): Blue
Allele a (recessive): Red

Genotype AA → Blue (expressed)
Genotype Aa → Blue (A dominates)
Genotype aa → Red (only when both recessive)
```

**Use in GAs:**
- Stores hidden variation
- Useful for dynamic environments
- Hidden alleles can express when needed

**Epistasis:**

When genes at different positions interact - one gene affects expression of another.

```
Without epistasis: Fitness = f(gene1) + f(gene2)
With epistasis: Fitness = f(gene1, gene2)  # Interaction
```

**Impact on GAs:**

| Low Epistasis | High Epistasis |
|---------------|----------------|
| Easy to optimize | Hard to optimize |
| Genes independent | Genes interact |
| Building blocks work | Blocks may not combine |

**NK Landscapes:**
- K = number of interacting genes
- K=0: No epistasis, easy
- K=high: Complex, rugged landscape

**Implications:**
- High epistasis → harder for crossover to work
- May need specialized operators
- Problem difficulty indicator

---

## Question 2
- [ ] Done

**How would you apply GAs to solve a traveling salesman problem (TSP)?**

**Answer:**

**Problem:** Visit N cities exactly once, minimize total distance.

**Encoding:**
```
Permutation: [3, 1, 4, 2, 5]
Meaning: Visit city 3 → 1 → 4 → 2 → 5 → back to 3
```

**Fitness Function:**
```python
def fitness(tour):
    total_distance = 0
    for i in range(len(tour)):
        city_a = tour[i]
        city_b = tour[(i + 1) % len(tour)]
        total_distance += distance_matrix[city_a][city_b]
    return -total_distance  # Negative because minimizing
```

**Specialized Operators:**

| Operator | Type | Description |
|----------|------|-------------|
| **PMX** | Crossover | Partially mapped, preserves order |
| **OX** | Crossover | Order crossover |
| **Swap** | Mutation | Exchange two cities |
| **Inversion** | Mutation | Reverse a segment |

**PMX Example:**
```
Parent 1: [1,2,3|4,5,6|7,8,9]
Parent 2: [9,3,7|1,2,8|6,5,4]
Swap middle → Child with valid permutation
```

**Best Practices:**
1. Use permutation encoding (not binary)
2. Apply local search (2-opt) after genetic ops
3. Use nearest neighbor heuristic for initialization
4. Large population for big instances

**Performance:** GA finds good (not always optimal) solutions for large TSP efficiently.

---

## Question 3
- [ ] Done

**Discuss the use of GAs in evolving neural network architectures.**

**Answer:**

**Neural Architecture Search (NAS) with GAs:**

**What to Evolve:**

| Aspect | Encoding |
|--------|----------|
| Layer types | [conv, pool, dense, ...] |
| Layer sizes | [64, 128, 256, ...] |
| Connections | Adjacency matrix |
| Activations | [relu, tanh, sigmoid] |
| Hyperparameters | [lr, dropout, ...] |

**Encoding Example:**
```python
chromosome = [
    {'type': 'conv', 'filters': 32, 'kernel': 3},
    {'type': 'pool', 'size': 2},
    {'type': 'conv', 'filters': 64, 'kernel': 3},
    {'type': 'dense', 'units': 128},
    {'type': 'output', 'units': 10}
]
```

**NEAT (NeuroEvolution of Augmenting Topologies):**
- Evolves topology AND weights
- Starts minimal, grows complexity
- Uses speciation to protect innovation
- Historical markings for crossover

**Fitness:**
```python
def fitness(architecture):
    model = build_network(architecture)
    model.fit(X_train, y_train, epochs=5)
    return model.evaluate(X_val, y_val)
```

**Challenges:**
- Expensive fitness evaluation
- Variable-length chromosomes
- Large search space

**Solutions:**
- Weight sharing across architectures
- Proxy tasks (fewer epochs)
- Surrogate models

---

## Question 4
- [ ] Done

**Discuss how to choose an appropriate selection method for a specific GA application.**

**Answer:**

**Selection Methods Comparison:**

| Method | Pressure | Best For |
|--------|----------|----------|
| **Roulette** | Variable | Smooth fitness landscapes |
| **Tournament** | Adjustable (k) | Most applications |
| **Rank** | Moderate | Avoiding fitness scaling issues |
| **Truncation** | High | Fast convergence needed |

**Factors to Consider:**

**1. Selection Pressure Needed:**
- Early exploration → Low pressure (small tournament, k=2)
- Final exploitation → High pressure (k=5+)

**2. Fitness Distribution:**
```
Roulette problems:
- Negative fitness (can't handle)
- Super-fit individual dominates
→ Use rank-based instead
```

**3. Population Diversity:**
- High pressure → Fast convergence, risk premature
- Low pressure → Maintain diversity, slower

**Decision Guide:**

| Situation | Recommended |
|-----------|-------------|
| General purpose | Tournament (k=2 or 3) |
| Need diversity | Tournament k=2, or rank |
| Fast convergence | Truncation or tournament k=5+ |
| Avoid super-dominance | Rank-based |
| Simple implementation | Tournament |

**Adaptive Approach:**
```python
# Start with low pressure, increase over time
k = 2 + int(3 * generation / max_gen)
selected = tournament_select(population, k)
```

---

## Question 5
- [ ] Done

**How would you design a GA for optimizing hyperparameters of an SVM classifier?**

**Answer:**

**SVM Hyperparameters:**
- C (regularization)
- kernel type (rbf, linear, poly)
- gamma (rbf kernel width)
- degree (polynomial kernel)

**Chromosome Encoding:**
```python
chromosome = {
    'C': 1.0,           # Real: [0.001, 1000]
    'kernel': 'rbf',    # Categorical: ['linear', 'rbf', 'poly']
    'gamma': 0.1,       # Real: [0.0001, 10]
    'degree': 3         # Integer: [2, 5] (only for poly)
}
```

**Fitness Function:**
```python
def fitness(chromosome):
    svm = SVC(
        C=chromosome['C'],
        kernel=chromosome['kernel'],
        gamma=chromosome['gamma']
    )
    score = cross_val_score(svm, X, y, cv=5).mean()
    return score
```

**Operators:**

| Parameter | Mutation |
|-----------|----------|
| C, gamma | Multiply by random in [0.5, 2] |
| kernel | Random choice from options |
| degree | Add/subtract 1 |

**Crossover:**
```python
# Uniform crossover for each parameter
child['C'] = parent1['C'] if random() > 0.5 else parent2['C']
```

**Best Practices:**
1. Log-scale for C and gamma (wide range)
2. Use stratified CV for imbalanced data
3. Early stopping to speed up poor configurations
4. Seed with default SVM parameters

---

## Question 6
- [ ] Done

**Propose a GA approach to create a timetable for a university course schedule.**

**Answer:**

**Problem:** Assign courses to rooms and time slots without conflicts.

**Constraints:**
- Hard: No teacher in two places, room capacity
- Soft: Teacher preferences, minimize gaps

**Encoding:**
```
Chromosome: [(course, room, time_slot), ...]
Example: [(CS101, R1, Mon9am), (CS102, R2, Mon10am), ...]
```

**Alternative Encoding:**
```
Matrix: courses × time_slots → room assignment
```

**Fitness Function:**
```python
def fitness(schedule):
    hard_violations = count_conflicts(schedule)
    soft_violations = count_preference_violations(schedule)
    
    if hard_violations > 0:
        return -1000 * hard_violations  # Heavily penalize
    return 100 - soft_violations  # Optimize preferences
```

**Operators:**

| Operator | Description |
|----------|-------------|
| Crossover | Exchange course blocks between parents |
| Mutation | Move course to different slot/room |
| Repair | Fix obvious conflicts after operators |

**Multi-Objective Alternative:**
- Objective 1: Minimize hard constraint violations
- Objective 2: Minimize soft constraint violations
- Use NSGA-II

**Practical Tips:**
1. Start with feasible (greedy) initialization
2. Use repair operators to maintain validity
3. Neighborhood-based mutation
4. Hybrid with local search (hill climbing)

---

## Question 7
- [ ] Done

**Discuss a scenario where a GA could be used to optimize the layout of a wind farm.**

**Answer:**

**Problem:** Position N wind turbines to maximize energy output while minimizing wake effects.

**Encoding:**
```
Chromosome: [(x1,y1), (x2,y2), ..., (xN,yN)]
Each pair = turbine location coordinates
```

**Fitness Function:**
```python
def fitness(positions):
    total_power = 0
    for turbine in positions:
        # Calculate wake effects from upstream turbines
        wind_speed = calculate_wake_reduced_speed(turbine, positions)
        power = turbine_power_curve(wind_speed)
        total_power += power
    
    # Penalty for minimum distance violations
    penalty = check_spacing_constraints(positions)
    return total_power - penalty
```

**Constraints:**
- Minimum spacing between turbines
- Boundary of available land
- Exclusion zones (roads, buildings)

**Multi-Objective:**
- Maximize: Annual energy production
- Minimize: Number of turbines, cost, visual impact

**Operators:**
- **Crossover**: Exchange turbine groups between layouts
- **Mutation**: Move turbine to nearby location

**Why GA Works Well:**
- Complex wake interactions (no gradient)
- Large search space
- Multiple optima exist
- Can handle irregular boundaries

**Real Applications:**
- Offshore wind farms
- Solar panel array layouts
- Cellular tower placement

---

## Question 8
- [ ] Done

**How would you use a genetic algorithm to handle the problem of vehicle routing with time windows?**

**Answer:**

**Problem (VRPTW):** Deliver to customers using vehicles, each customer has time window [earliest, latest].

**Constraints:**
- Vehicle capacity
- Time windows for each customer
- Depot start/end times

**Encoding:**
```
Chromosome: [3, 1, 4 | 2, 5, 7 | 6, 8]
            Route 1  | Route 2 | Route 3
```

Or priority-based:
```
[0.8, 0.2, 0.9, 0.1, ...] → Decode to routes by priority
```

**Fitness Function:**
```python
def fitness(routes):
    total_distance = sum(route_distance(r) for r in routes)
    time_violations = sum(time_window_violations(r) for r in routes)
    capacity_violations = sum(capacity_violations(r) for r in routes)
    
    penalty = 1000 * time_violations + 500 * capacity_violations
    return -(total_distance + penalty)
```

**Operators:**

| Operator | Description |
|----------|-------------|
| Crossover | Exchange route segments |
| Mutation | Move customer, swap, 2-opt |
| Repair | Fix time window violations |

**Best Practices:**
1. Use route-based encoding (giant tour + split)
2. Local search (2-opt, Or-opt) for improvement
3. Penalty method for soft constraints
4. Seed with greedy insertion heuristic

**Extensions:** 
- Pickup and delivery, multiple depots, heterogeneous fleet

---

## Question 9
- [ ] Done

**Discuss the role of ‘speciation’ in genetic algorithms and its potential benefits.**

**Answer:**

**Definition:**

**Speciation** groups similar individuals into species that primarily mate within their group, protecting different niches from competitive exclusion.

**How It Works:**
```
Population divided by similarity:
Species 1: ●●●● (similar to each other)
Species 2: ▲▲▲ (different from Species 1)
Species 3: ■■■■■ (different from both)

Crossover mainly within species
```

**NEAT Algorithm Speciation:**
- Measure genome distance (topology + weights)
- Group into species by threshold
- Share fitness within species
- Protect innovative structures

**Benefits:**

| Benefit | Description |
|---------|-------------|
| **Protects innovation** | New structures have time to optimize |
| **Maintains diversity** | Different solutions coexist |
| **Multi-modal** | Finds multiple optima |
| **Reduces competition** | Species compete, not all individuals |

**Fitness Sharing:**
```python
shared_fitness = raw_fitness / species_size
# Small species get fitness boost
# Prevents large species from dominating
```

**When to Use:**
- Evolving complex structures (neural networks)
- Multi-modal optimization
- Want diverse solution set
- Problem has multiple valid approaches

---

## Question 10
- [ ] Done

**Discuss the considerations in balancing exploration and exploitation in GAs.**

**Answer:**

**Definitions:**
- **Exploration**: Searching new regions of solution space
- **Exploitation**: Refining solutions in current good regions

**The Trade-off:**
```
Too much exploration → Never converges, random search
Too much exploitation → Premature convergence, local optima
```

**Factors Affecting Balance:**

| Factor | More Exploration | More Exploitation |
|--------|------------------|-------------------|
| **Mutation rate** | High | Low |
| **Selection pressure** | Low (k=2) | High (k=5+) |
| **Population size** | Large | Small |
| **Crossover** | Disruptive | Preserving |

**Adaptive Strategies:**

**1. Time-Based:**
```python
# Explore early, exploit late
mutation_rate = initial_rate * (1 - generation/max_gen)
```

**2. Fitness-Based:**
```python
# Explore when stuck
if no_improvement(n_gens):
    mutation_rate *= 2  # Increase exploration
```

**3. Diversity-Based:**
```python
if diversity < threshold:
    add_random_immigrants()  # Restore exploration
```

**Best Practices:**
1. Start with exploration (high mutation, low pressure)
2. Monitor diversity metrics
3. Gradually shift to exploitation
4. Use island model for natural balance
5. Allow for periodic restarts

**Interview Tip:**
The exploration-exploitation trade-off is fundamental to all optimization. GAs provide multiple levers to control it.

---

