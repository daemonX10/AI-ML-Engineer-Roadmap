# Genetic Algorithms Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Explain how ‘coevolution’ is implemented in GAs.**

**Answer:**

**Definition:** Multiple populations evolve together, where fitness depends on interactions between populations.

**Types:**

| Type | Description |
|------|-------------|
| **Cooperative** | Populations work together (combined fitness) |
| **Competitive** | Populations compete (game-theoretic) |

**Implementation:**

```python
# Cooperative Coevolution Example
population_A = [...]  # Subcomponent 1
population_B = [...]  # Subcomponent 2

for generation in range(max_gens):
    # Evaluate A with best of B
    for individual_a in population_A:
        combined = combine(individual_a, best_of_B)
        individual_a.fitness = evaluate(combined)
    
    # Evaluate B with best of A  
    for individual_b in population_B:
        combined = combine(best_of_A, individual_b)
        individual_b.fitness = evaluate(combined)
    
    # Evolve each population
    population_A = evolve(population_A)
    population_B = evolve(population_B)
```

**Competitive Example (Predator-Prey):**
```python
# Predator fitness = catching prey
# Prey fitness = escaping predators
# Both improve through arms race
```

**Applications:**
- Evolving game strategies (player vs opponent)
- Multi-component system optimization
- Test case generation (tests vs programs)

---

## Question 2
- [ ] Done

**Implement a basic genetic algorithm in Python to solve the problem of maximizing the function f(x) = x^2.**

**Answer:**

```python
import random

def fitness(x):
    return x ** 2  # Maximize x^2

def create_individual():
    return random.uniform(-10, 10)

def crossover(p1, p2):
    return (p1 + p2) / 2  # Arithmetic crossover

def mutate(x, rate=0.1):
    if random.random() < rate:
        return x + random.gauss(0, 1)
    return x

def tournament_select(population, k=3):
    selected = random.sample(population, k)
    return max(selected, key=fitness)

# GA Parameters
pop_size = 50
generations = 100

# Initialize population
population = [create_individual() for _ in range(pop_size)]

# Evolution loop
for gen in range(generations):
    # Create next generation
    new_pop = []
    for _ in range(pop_size):
        p1 = tournament_select(population)
        p2 = tournament_select(population)
        child = crossover(p1, p2)
        child = mutate(child)
        new_pop.append(child)
    
    population = new_pop
    best = max(population, key=fitness)
    
    if gen % 20 == 0:
        print(f"Gen {gen}: Best x = {best:.2f}, f(x) = {fitness(best):.2f}")

# Result (should find x near ±10)
print(f"Final: x = {best:.2f}, f(x) = {fitness(best):.2f}")
```

---

## Question 3
- [ ] Done

**Write a genetic algorithm to evolve a simple string of characters toward a target string.**

**Answer:**

```python
import random
import string

TARGET = "HELLO WORLD"
CHARS = string.ascii_uppercase + " "

def fitness(individual):
    # Count matching characters
    return sum(a == b for a, b in zip(individual, TARGET))

def create_individual():
    return ''.join(random.choice(CHARS) for _ in range(len(TARGET)))

def crossover(p1, p2):
    # Single-point crossover
    point = random.randint(1, len(TARGET) - 1)
    return p1[:point] + p2[point:]

def mutate(individual, rate=0.1):
    result = list(individual)
    for i in range(len(result)):
        if random.random() < rate:
            result[i] = random.choice(CHARS)
    return ''.join(result)

# Initialize
pop_size = 100
population = [create_individual() for _ in range(pop_size)]

generation = 0
while True:
    # Sort by fitness
    population.sort(key=fitness, reverse=True)
    best = population[0]
    
    if best == TARGET:
        print(f"Found '{best}' in generation {generation}")
        break
    
    if generation % 50 == 0:
        print(f"Gen {generation}: '{best}' (fitness: {fitness(best)})")
    
    # Create next generation
    next_gen = population[:10]  # Elitism: keep top 10
    while len(next_gen) < pop_size:
        p1, p2 = random.sample(population[:50], 2)
        child = mutate(crossover(p1, p2))
        next_gen.append(child)
    
    population = next_gen
    generation += 1
```

---

## Question 4
- [ ] Done

**Create a mutation function for a GA that operates on real-valued vectors.**

**Answer:**

```python
import numpy as np

def gaussian_mutation(individual, mutation_rate=0.1, sigma=0.5):
    """
    Gaussian mutation for real-valued chromosomes
    Each gene mutated with probability mutation_rate
    """
    mutated = individual.copy()
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            mutated[i] += np.random.normal(0, sigma)
    return mutated

def uniform_mutation(individual, mutation_rate=0.1, bounds=(-10, 10)):
    """
    Replace gene with random value from bounds
    """
    mutated = individual.copy()
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            mutated[i] = np.random.uniform(bounds[0], bounds[1])
    return mutated

def polynomial_mutation(individual, mutation_rate=0.1, eta=20):
    """
    Polynomial mutation (used in NSGA-II)
    eta controls distribution: higher = smaller changes
    """
    mutated = individual.copy()
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            u = np.random.random()
            if u < 0.5:
                delta = (2*u)**(1/(eta+1)) - 1
            else:
                delta = 1 - (2*(1-u))**(1/(eta+1))
            mutated[i] += delta
    return mutated

# Example usage
individual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Original: {individual}")
print(f"Gaussian: {gaussian_mutation(individual)}")
print(f"Uniform:  {uniform_mutation(individual)}")
```

---

## Question 5
- [ ] Done

**Code a crossover function that combines two parent solutions to produce offspring for a bit-string representation.**

**Answer:**

```python
import random

def single_point_crossover(parent1, parent2):
    """Single point crossover for bit strings"""
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def two_point_crossover(parent1, parent2):
    """Two point crossover"""
    p1 = random.randint(1, len(parent1) - 2)
    p2 = random.randint(p1 + 1, len(parent1) - 1)
    child1 = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
    child2 = parent2[:p1] + parent1[p1:p2] + parent2[p2:]
    return child1, child2

def uniform_crossover(parent1, parent2, prob=0.5):
    """Each bit randomly from one parent"""
    child1, child2 = [], []
    for i in range(len(parent1)):
        if random.random() < prob:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    return child1, child2

# Example
parent1 = [1, 1, 1, 1, 1, 1, 1, 1]
parent2 = [0, 0, 0, 0, 0, 0, 0, 0]

print(f"Parents: {parent1}, {parent2}")
print(f"Single-point: {single_point_crossover(parent1, parent2)}")
print(f"Two-point: {two_point_crossover(parent1, parent2)}")
print(f"Uniform: {uniform_crossover(parent1, parent2)}")
```

---

## Question 6
- [ ] Done

**Develop a selection mechanism in Python to select fittest individuals based on the roulette wheel method.**

**Answer:**

```python
import random

def roulette_wheel_selection(population, fitness_scores, num_select):
    """
    Roulette wheel (fitness proportionate) selection
    Higher fitness = higher probability of selection
    """
    # Handle negative fitness by shifting
    min_fit = min(fitness_scores)
    if min_fit < 0:
        fitness_scores = [f - min_fit + 1 for f in fitness_scores]
    
    total_fitness = sum(fitness_scores)
    
    # Calculate selection probabilities
    probabilities = [f / total_fitness for f in fitness_scores]
    
    # Calculate cumulative probabilities
    cumulative = []
    cum_sum = 0
    for p in probabilities:
        cum_sum += p
        cumulative.append(cum_sum)
    
    # Select individuals
    selected = []
    for _ in range(num_select):
        r = random.random()
        for i, cum_prob in enumerate(cumulative):
            if r <= cum_prob:
                selected.append(population[i])
                break
    
    return selected

# Example
population = ['A', 'B', 'C', 'D', 'E']
fitness_scores = [10, 20, 30, 25, 15]  # Higher is better

print("Population with fitness:", list(zip(population, fitness_scores)))
selected = roulette_wheel_selection(population, fitness_scores, 3)
print("Selected:", selected)
```

**Key Points:**
- Selection probability proportional to fitness
- Handles negative fitness by shifting
- Higher fitness individuals more likely to be selected

---

## Question 7
- [ ] Done

**Write a Python function that implements a rank-based selection method for a GA.**

**Answer:**

```python
import random

def rank_based_selection(population, fitness_scores, num_select):
    """
    Rank-based selection - probability based on rank, not raw fitness
    Reduces selection pressure compared to roulette wheel
    """
    n = len(population)
    
    # Create (individual, fitness) pairs and sort by fitness
    paired = list(zip(population, fitness_scores))
    paired.sort(key=lambda x: x[1])  # Ascending order
    
    # Assign ranks (1 = worst, n = best)
    ranks = list(range(1, n + 1))
    
    # Selection probability proportional to rank
    total_rank = sum(ranks)
    probabilities = [r / total_rank for r in ranks]
    
    # Calculate cumulative probabilities
    cumulative = []
    cum_sum = 0
    for p in probabilities:
        cum_sum += p
        cumulative.append(cum_sum)
    
    # Select individuals
    selected = []
    for _ in range(num_select):
        r = random.random()
        for i, cum_prob in enumerate(cumulative):
            if r <= cum_prob:
                selected.append(paired[i][0])
                break
    
    return selected

# Example
population = ['A', 'B', 'C', 'D', 'E']
fitness_scores = [100, 500, 150, 200, 180]  # Varied fitness

print("Population with fitness:", list(zip(population, fitness_scores)))
selected = rank_based_selection(population, fitness_scores, 3)
print("Selected:", selected)
```

**Advantages of Rank-Based Selection:**
- Prevents dominance by very fit individuals
- Maintains selection pressure even with similar fitness values
- More robust to fitness scaling issues

---

## Question 8
- [ ] Done

**Implement a GA in Python to solve the problem of finding the minimal-cost path in a graph.**

**Answer:**

```python
import random

# Graph: adjacency matrix (infinity = no edge)
INF = float('inf')
graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
NUM_CITIES = len(graph)

def path_cost(path):
    """Calculate total cost of a path"""
    cost = 0
    for i in range(len(path) - 1):
        cost += graph[path[i]][path[i+1]]
    cost += graph[path[-1]][path[0]]  # Return to start
    return cost

def fitness(path):
    """Lower cost = higher fitness"""
    return 1 / (1 + path_cost(path))

def create_individual():
    """Random permutation of cities"""
    path = list(range(NUM_CITIES))
    random.shuffle(path)
    return path

def pmx_crossover(parent1, parent2):
    """Partially Mapped Crossover for permutations"""
    size = len(parent1)
    p1, p2 = sorted(random.sample(range(size), 2))
    
    child = [-1] * size
    child[p1:p2] = parent1[p1:p2]
    
    for i in range(p1, p2):
        if parent2[i] not in child:
            val = parent2[i]
            pos = i
            while p1 <= pos < p2:
                pos = parent2.index(parent1[pos])
            child[pos] = val
    
    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]
    
    return child

def swap_mutation(path, rate=0.1):
    """Swap two cities"""
    if random.random() < rate:
        i, j = random.sample(range(len(path)), 2)
        path[i], path[j] = path[j], path[i]
    return path

# GA Main Loop
pop_size = 50
generations = 100
population = [create_individual() for _ in range(pop_size)]

for gen in range(generations):
    population.sort(key=fitness, reverse=True)
    best = population[0]
    
    if gen % 20 == 0:
        print(f"Gen {gen}: Best path {best}, Cost: {path_cost(best)}")
    
    # Create next generation
    next_gen = population[:5]  # Elitism
    while len(next_gen) < pop_size:
        p1, p2 = random.sample(population[:20], 2)
        child = pmx_crossover(p1, p2)
        child = swap_mutation(child)
        next_gen.append(child)
    
    population = next_gen

print(f"\nBest path: {population[0]}, Cost: {path_cost(population[0])}")
```

---

## Question 9
- [ ] Done

**Create a simple GA in Python for optimizing the weights of a small neural network.**

**Answer:**

```python
import numpy as np

# Simple XOR problem
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def decode_weights(chromosome, input_size=2, hidden_size=4, output_size=1):
    """Convert flat chromosome to weight matrices"""
    idx = 0
    w1_size = input_size * hidden_size
    w1 = chromosome[idx:idx+w1_size].reshape(input_size, hidden_size)
    idx += w1_size
    
    b1 = chromosome[idx:idx+hidden_size]
    idx += hidden_size
    
    w2_size = hidden_size * output_size
    w2 = chromosome[idx:idx+w2_size].reshape(hidden_size, output_size)
    idx += w2_size
    
    b2 = chromosome[idx:idx+output_size]
    return w1, b1, w2, b2

def forward(X, chromosome):
    """Forward pass through network"""
    w1, b1, w2, b2 = decode_weights(chromosome)
    hidden = sigmoid(X @ w1 + b1)
    output = sigmoid(hidden @ w2 + b2)
    return output

def fitness(chromosome):
    """MSE loss (negative for maximization)"""
    pred = forward(X, chromosome)
    mse = np.mean((pred - y) ** 2)
    return 1 / (1 + mse)  # Higher fitness = lower error

def mutate(chrom, rate=0.2, sigma=0.5):
    """Gaussian mutation"""
    mutated = chrom.copy()
    for i in range(len(mutated)):
        if np.random.random() < rate:
            mutated[i] += np.random.normal(0, sigma)
    return mutated

def crossover(p1, p2):
    """Single point crossover"""
    point = np.random.randint(1, len(p1))
    return np.concatenate([p1[:point], p2[point:]])

# Total weights: 2*4 + 4 + 4*1 + 1 = 17
chrom_size = 17
pop_size = 50

# Initialize population
population = [np.random.randn(chrom_size) for _ in range(pop_size)]

# Evolution
for gen in range(200):
    population.sort(key=fitness, reverse=True)
    best = population[0]
    
    if gen % 50 == 0:
        pred = forward(X, best)
        print(f"Gen {gen}: Fitness={fitness(best):.4f}")
        print(f"  Predictions: {pred.flatten().round(2)}")
    
    # Next generation
    next_gen = population[:5]  # Elitism
    while len(next_gen) < pop_size:
        p1, p2 = [population[i] for i in np.random.choice(15, 2)]
        child = mutate(crossover(p1, p2))
        next_gen.append(child)
    
    population = next_gen

print("\nFinal predictions:", forward(X, population[0]).flatten().round(2))
print("Expected:         ", y.flatten())
```

---

## Question 10
- [ ] Done

**Share an example where a GA has been successfully implemented in an industrial setting.**

**Answer:**

**Example: NASA Antenna Design Using Genetic Algorithms**

NASA used genetic algorithms to design an antenna for the Space Technology 5 (ST5) mission. The evolved antenna outperformed human-designed antennas and had an unusual, non-intuitive shape that engineers would never have conceived.

**Industrial Applications Overview:**

| Industry | Application | GA Role |
|----------|-------------|---------|
| Aerospace | Antenna design, wing optimization | Shape optimization |
| Automotive | Engine tuning, crash safety | Multi-objective optimization |
| Manufacturing | Job scheduling, supply chain | Combinatorial optimization |
| Finance | Portfolio optimization, trading | Risk-return balancing |
| Telecom | Network routing, frequency allocation | Resource optimization |

**Simple Industrial Scheduling Example:**

```python
import random

# Job scheduling: minimize total completion time
jobs = [
    {'id': 'A', 'duration': 3, 'deadline': 10},
    {'id': 'B', 'duration': 5, 'deadline': 8},
    {'id': 'C', 'duration': 2, 'deadline': 6},
    {'id': 'D', 'duration': 4, 'deadline': 12},
    {'id': 'E', 'duration': 1, 'deadline': 4}
]

def fitness(schedule):
    """Minimize tardiness (late jobs penalty)"""
    time = 0
    tardiness = 0
    for job_idx in schedule:
        job = jobs[job_idx]
        time += job['duration']
        if time > job['deadline']:
            tardiness += (time - job['deadline'])
    return 1 / (1 + tardiness)

def create_schedule():
    schedule = list(range(len(jobs)))
    random.shuffle(schedule)
    return schedule

def mutate(schedule):
    s = schedule.copy()
    i, j = random.sample(range(len(s)), 2)
    s[i], s[j] = s[j], s[i]
    return s

# Simple GA
pop_size = 30
population = [create_schedule() for _ in range(pop_size)]

for gen in range(100):
    population.sort(key=fitness, reverse=True)
    best = population[0]
    
    next_gen = population[:3]
    while len(next_gen) < pop_size:
        parent = random.choice(population[:10])
        next_gen.append(mutate(parent))
    population = next_gen

# Result
best_schedule = population[0]
print("Best schedule:", [jobs[i]['id'] for i in best_schedule])
print("Fitness:", fitness(best_schedule))
```

**Key Industrial Success Factors:**
1. **Well-defined fitness function**: Clear optimization objective
2. **Appropriate representation**: Problem-specific chromosome encoding
3. **Hybrid approaches**: GA combined with local search (memetic algorithms)
4. **Parallelization**: Distributed evaluation for large populations
5. **Domain expertise**: Problem-specific operators improve convergence

---

