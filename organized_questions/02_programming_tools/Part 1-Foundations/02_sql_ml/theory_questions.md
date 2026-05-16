# SQL ML Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What are the different types of JOIN operations in SQL?**

### Definition
JOINs combine rows from two or more tables based on related columns. The JOIN type determines which rows appear in the result set.

### Types of JOINs

| JOIN Type | Description | Returns |
|-----------|-------------|---------|
| INNER JOIN | Match in both tables | Only matching rows |
| LEFT JOIN | All from left + matches from right | All left, NULLs for non-matches |
| RIGHT JOIN | All from right + matches from left | All right, NULLs for non-matches |
| FULL OUTER JOIN | All from both tables | All rows, NULLs where no match |
| CROSS JOIN | Cartesian product | Every combination |
| SELF JOIN | Table joined with itself | Hierarchical/comparative queries |

### Code Examples
```sql
-- INNER JOIN: Customers with orders only
SELECT c.name, o.order_date
FROM customers c
INNER JOIN orders o ON c.id = o.customer_id;

-- LEFT JOIN: All customers, including those without orders
SELECT c.name, o.order_date
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id;

-- Find customers with NO orders
SELECT c.name
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.order_id IS NULL;
```

### ML Use Cases
- **INNER**: Create feature matrix with complete data only
- **LEFT**: Find inactive users for churn prediction
- **CROSS**: Generate all user-product pairs for recommendations

---

## Question 2
- [ ] Done

**Explain the difference between WHERE and HAVING clauses.**

### Definition
- **WHERE**: Filters individual rows BEFORE grouping
- **HAVING**: Filters groups AFTER aggregation

### SQL Execution Order
```
1. FROM / JOIN
2. WHERE        ← Row filtering (no aggregates)
3. GROUP BY
4. HAVING       ← Group filtering (aggregates allowed)
5. SELECT
6. ORDER BY
7. LIMIT
```

### Code Example
```sql
-- Find customers with >5 orders where each order was >$10
SELECT 
    customer_id,
    COUNT(*) AS order_count,
    SUM(amount) AS total_spent
FROM orders
WHERE amount > 10              -- Filter rows first
  AND order_date >= '2024-01-01'
GROUP BY customer_id
HAVING COUNT(*) > 5;           -- Filter groups after
```

### Best Practice
**Filter early with WHERE** - reduces data before aggregation, improving performance.

---

## Question 3
- [ ] Done

**Describe a subquery and its typical use case.**

### Definition
A subquery (inner query) is a SELECT statement nested inside another query.

### Types

| Type | Runs | Description |
|------|------|-------------|
| Non-correlated | Once | Independent of outer query |
| Correlated | Per row | References outer query |

### Code Examples
```sql
-- Non-correlated: Find employees in Engineering dept
SELECT name FROM employees
WHERE dept_id IN (
    SELECT id FROM departments WHERE name = 'Engineering'
);

-- Correlated: Each employee vs their dept average
SELECT name, salary,
    (SELECT AVG(salary) FROM employees e2 
     WHERE e2.dept_id = e1.dept_id) AS dept_avg
FROM employees e1;
```

### Performance Tip
Correlated subqueries run per row - consider rewriting as JOINs for better performance.

---

## Question 4
- [ ] Done

**What are SQL Window Functions and how can they be used for Machine Learning feature engineering ?**

### Definition
Window functions perform calculations across a set of rows related to the current row, without collapsing rows like GROUP BY.

### Syntax
```sql
function() OVER (
    PARTITION BY column    -- Groups (keeps rows)
    ORDER BY column        -- Order within partition
)
```

### Key Window Functions

| Function | Purpose | ML Use Case |
|----------|---------|-------------|
| LAG(col, n) | Previous row value | Previous purchase amount |
| LEAD(col, n) | Next row value | Target variable creation |
| ROW_NUMBER() | Sequential rank | Deduplication |
| RANK() | Rank with gaps | Rank within category |
| SUM() OVER | Running total | Cumulative spending |
| AVG() OVER | Moving average | Smoothed features |

### Code Example
```sql
SELECT
    customer_id,
    transaction_date,
    amount,
    
    -- Previous transaction
    LAG(amount, 1) OVER (
        PARTITION BY customer_id ORDER BY transaction_date
    ) AS prev_amount,
    
    -- Running total
    SUM(amount) OVER (
        PARTITION BY customer_id ORDER BY transaction_date
    ) AS cumulative_spent,
    
    -- Rank by amount
    RANK() OVER (
        PARTITION BY customer_id ORDER BY amount DESC
    ) AS amount_rank

FROM transactions;
```

---

## Question 5
- [ ] Done

**Explain how to discretize a continuous variable in SQL.**

### Definition
Discretization (binning) converts continuous variables into categorical by grouping values into intervals.

### Two Approaches

```sql
-- Equal-width: Custom bins with CASE
SELECT 
    user_id,
    age,
    CASE 
        WHEN age < 25 THEN '18-24'
        WHEN age < 35 THEN '25-34'
        WHEN age < 50 THEN '35-49'
        ELSE '50+'
    END AS age_group
FROM users;

-- Equal-frequency: Quartiles with NTILE
SELECT 
    user_id,
    spending,
    NTILE(4) OVER (ORDER BY spending) AS spending_quartile
FROM user_stats;
-- Returns 1 (lowest 25%), 2, 3, 4 (highest 25%)
```

### When to Use Each
- **CASE**: Domain-specific bins (age groups, income brackets)
- **NTILE**: Statistical bins (percentiles, deciles)

---

## Question 6
- [ ] Done

**Describe SQL techniques to perform data sampling.**

### Sampling Methods

```sql
-- Method 1: Random sample (slow for large tables)
SELECT * FROM users
ORDER BY RAND()
LIMIT 1000;

-- Method 2: TABLESAMPLE (fast, database-specific)
SELECT * FROM users TABLESAMPLE BERNOULLI(10);  -- 10% sample

-- Method 3: Stratified sampling
WITH Ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY customer_segment ORDER BY RAND()
        ) AS rn
    FROM customers
)
SELECT * FROM Ranked WHERE rn <= 100;  -- 100 from each segment
```

### Interview Tip
For ML, stratified sampling is critical for imbalanced datasets.

---

## Question 7
- [ ] Done

**Explain how to perform binning of categorical variables in SQL for use in a Machine Learning model.**

### Definition
Binning (or grouping) of categorical variables consolidates low-frequency or high-cardinality categories into broader groups, reducing dimensionality and improving model generalization.

### Why Bin Categorical Variables?
| Problem | Solution via Binning |
|---------|---------------------|
| Too many unique categories | Group rare values into "Other" |
| Sparse one-hot encoding | Fewer columns after encoding |
| Overfitting on rare categories | Merged groups have more samples |
| Noisy labels | Semantically similar categories combined |

### Code Examples
```sql
-- Method 1: Group rare categories into "Other"
WITH category_counts AS (
    SELECT 
        product_category,
        COUNT(*) AS cnt,
        COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () AS pct
    FROM products
    GROUP BY product_category
)
SELECT 
    p.product_id,
    CASE 
        WHEN cc.pct < 0.02 THEN 'Other'
        ELSE p.product_category
    END AS binned_category
FROM products p
JOIN category_counts cc ON p.product_category = cc.product_category;

-- Method 2: Domain-driven grouping with CASE
SELECT 
    employee_id,
    job_title,
    CASE 
        WHEN job_title IN ('CEO', 'CFO', 'CTO', 'VP') THEN 'Executive'
        WHEN job_title IN ('Manager', 'Director', 'Lead') THEN 'Management'
        WHEN job_title IN ('Engineer', 'Developer', 'Analyst') THEN 'Technical'
        ELSE 'Support'
    END AS role_group
FROM employees;

-- Method 3: Frequency-based binning using a lookup table
CREATE TABLE category_mapping AS
SELECT 
    product_category AS original,
    CASE 
        WHEN RANK() OVER (ORDER BY COUNT(*) DESC) <= 10 THEN product_category
        ELSE 'Other'
    END AS mapped_category
FROM products
GROUP BY product_category;
```

### Interview Tip
Always explain *why* you bin — mention the trade-off between information loss and dimensionality reduction, and how it prevents overfitting on rare categories.

---

## Question 8
- [ ] Done

**How does SQL play a role in ML model deployment?**

### Definition
SQL plays a critical role in ML deployment by serving as the interface between production databases and ML models — handling data pipelines, serving predictions, storing results, and monitoring model performance in real time.

### Key Roles of SQL in ML Deployment

| Role | Description |
|------|-------------|
| Feature serving | Real-time feature extraction for inference |
| Prediction storage | Storing model outputs alongside business data |
| Batch scoring | Running predictions on large datasets via SQL |
| Monitoring | Tracking data drift and model performance |
| A/B testing | Splitting traffic and comparing model versions |

### Code Examples
```sql
-- 1. Real-time feature extraction for model inference
CREATE VIEW customer_features AS
SELECT 
    customer_id,
    COUNT(*) AS total_orders,
    AVG(amount) AS avg_order_value,
    DATEDIFF(DAY, MAX(order_date), GETDATE()) AS days_since_last_order,
    SUM(CASE WHEN returned = 1 THEN 1 ELSE 0 END) AS return_count
FROM orders
GROUP BY customer_id;

-- 2. Store model predictions back to database
CREATE TABLE model_predictions (
    prediction_id INT IDENTITY PRIMARY KEY,
    customer_id INT,
    model_version VARCHAR(20),
    prediction_score FLOAT,
    predicted_label VARCHAR(50),
    prediction_timestamp DATETIME DEFAULT GETDATE()
);

INSERT INTO model_predictions (customer_id, model_version, prediction_score, predicted_label)
VALUES (1001, 'v2.3', 0.87, 'churn');

-- 3. Batch scoring via SQL/ML integration (SQL Server example)
SELECT 
    customer_id,
    prediction_score
FROM PREDICT(MODEL = 'churn_model', DATA = customer_features);

-- 4. Monitor prediction distribution over time
SELECT 
    CAST(prediction_timestamp AS DATE) AS pred_date,
    AVG(prediction_score) AS avg_score,
    STDEV(prediction_score) AS score_stddev,
    COUNT(*) AS prediction_count
FROM model_predictions
WHERE model_version = 'v2.3'
GROUP BY CAST(prediction_timestamp AS DATE)
ORDER BY pred_date;
```

### Interview Tip
Mention that modern databases (SQL Server, BigQuery, Redshift) support in-database ML scoring via `PREDICT()` or ML-specific extensions, eliminating the need to move data out for inference.

---

## Question 9
- [ ] Done

**What is the significance of in-database analytics for Machine Learning?**

### Definition
In-database analytics refers to performing data processing, statistical analysis, and even model training/scoring directly within the database engine, rather than extracting data to external tools.

### Why It Matters

| Benefit | Explanation |
|---------|-------------|
| Eliminates data movement | No ETL overhead — data stays where it lives |
| Leverages database optimization | Query optimizer, indexing, parallel execution |
| Scalability | Processes billions of rows without memory limits |
| Security | Data never leaves the governed database environment |
| Lower latency | Real-time feature computation and scoring |

### Platforms Supporting In-Database ML

| Platform | ML Capability |
|----------|--------------|
| SQL Server | `sp_execute_external_script`, PREDICT() |
| BigQuery | `CREATE MODEL` (BigQuery ML) |
| Redshift | `CREATE MODEL` (Redshift ML) |
| PostgreSQL | MADlib extension |
| Oracle | Oracle Machine Learning (OML) |

### Code Example
```sql
-- BigQuery ML: Train a logistic regression model entirely in SQL
CREATE OR REPLACE MODEL `project.dataset.churn_model`
OPTIONS(
    model_type = 'LOGISTIC_REG',
    input_label_cols = ['churned']
) AS
SELECT 
    total_orders,
    avg_order_value,
    days_since_last_order,
    churned
FROM `project.dataset.customer_features`;

-- Score new customers using the trained model
SELECT 
    customer_id,
    predicted_churned,
    predicted_churned_probs
FROM ML.PREDICT(
    MODEL `project.dataset.churn_model`,
    (SELECT * FROM `project.dataset.new_customers`)
);
```

### Interview Tip
Emphasize that in-database analytics is ideal for organizations with strict data governance requirements, as sensitive data never leaves the database perimeter.

---

## Question 10
- [ ] Done

**Explain recursive SQL queries and how they can be used to prepare data for hierarchical Machine Learning algorithms.**

### Definition
A recursive SQL query (recursive CTE) references itself to iteratively traverse hierarchical or graph-like data structures such as org charts, bill-of-materials, or category trees.

### Syntax Structure
```sql
WITH RECURSIVE cte_name AS (
    -- Anchor member: starting rows
    SELECT ... FROM table WHERE condition
    
    UNION ALL
    
    -- Recursive member: joins back to itself
    SELECT ... FROM table
    JOIN cte_name ON parent-child relationship
)
SELECT * FROM cte_name;
```

### Code Examples
```sql
-- 1. Traverse an organizational hierarchy
WITH RECURSIVE org_tree AS (
    -- Anchor: top-level managers (no manager above)
    SELECT 
        employee_id, name, manager_id, 
        1 AS depth,
        CAST(name AS VARCHAR(500)) AS path
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive: employees reporting to someone in the tree
    SELECT 
        e.employee_id, e.name, e.manager_id,
        ot.depth + 1,
        CAST(ot.path || ' > ' || e.name AS VARCHAR(500))
    FROM employees e
    JOIN org_tree ot ON e.manager_id = ot.employee_id
)
SELECT * FROM org_tree ORDER BY path;

-- 2. Generate hierarchy-based features for ML
WITH RECURSIVE category_tree AS (
    SELECT category_id, parent_id, name, 0 AS level
    FROM categories WHERE parent_id IS NULL
    
    UNION ALL
    
    SELECT c.category_id, c.parent_id, c.name, ct.level + 1
    FROM categories c
    JOIN category_tree ct ON c.parent_id = ct.category_id
)
SELECT 
    p.product_id,
    ct.level AS category_depth,
    ct.name AS root_category,
    COUNT(*) OVER (PARTITION BY ct.category_id) AS sibling_count
FROM products p
JOIN category_tree ct ON p.category_id = ct.category_id;

-- 3. Compute subtree size as a feature
WITH RECURSIVE subtree AS (
    SELECT employee_id, employee_id AS root_id
    FROM employees
    
    UNION ALL
    
    SELECT e.employee_id, s.root_id
    FROM employees e
    JOIN subtree s ON e.manager_id = s.employee_id
)
SELECT 
    root_id AS employee_id,
    COUNT(*) - 1 AS team_size  -- exclude self
FROM subtree
GROUP BY root_id;
```

### ML Use Cases
- **Hierarchy depth**: Feature indicating how deep a node sits
- **Subtree size**: Number of descendants (e.g., team size for manager performance models)
- **Path encoding**: Convert tree paths to embeddings for hierarchical classification

### Interview Tip
Always mention the termination condition — recursive CTEs need a base case and a condition that eventually stops recursion. In practice, add `WHERE depth < 50` as a safety limit.

---

## Question 11
- [ ] Done

**Describe how graph-based features can be generated from SQL data.**

### Definition
Graph-based features capture relationships and connectivity patterns between entities stored in relational tables. Nodes represent entities (users, products) and edges represent relationships (purchases, follows, messages).

### Common Graph Features

| Feature | Description | ML Use Case |
|---------|-------------|-------------|
| Degree | Number of connections | Social influence score |
| In-degree / Out-degree | Directional connections | Authority vs hub detection |
| Mutual connections | Shared neighbors | Link prediction |
| Shortest path | Minimum hops between nodes | Fraud ring detection |
| Triangle count | Closed triads involving a node | Community strength |
| PageRank (approx.) | Iterative importance score | Recommendation ranking |

### Code Examples
```sql
-- 1. Node degree: count of connections per user
SELECT 
    user_id,
    COUNT(DISTINCT friend_id) AS degree
FROM friendships
GROUP BY user_id;

-- 2. In-degree and out-degree for directed graphs
SELECT 
    u.user_id,
    COALESCE(out_d.out_degree, 0) AS out_degree,
    COALESCE(in_d.in_degree, 0) AS in_degree
FROM users u
LEFT JOIN (
    SELECT sender_id AS user_id, COUNT(*) AS out_degree
    FROM messages GROUP BY sender_id
) out_d ON u.user_id = out_d.user_id
LEFT JOIN (
    SELECT receiver_id AS user_id, COUNT(*) AS in_degree
    FROM messages GROUP BY receiver_id
) in_d ON u.user_id = in_d.user_id;

-- 3. Mutual connections between two users
SELECT COUNT(*) AS mutual_friends
FROM friendships f1
JOIN friendships f2 
    ON f1.friend_id = f2.friend_id
WHERE f1.user_id = 101 AND f2.user_id = 202;

-- 4. Triangle count per node (closed triads)
SELECT 
    f1.user_id,
    COUNT(*) AS triangle_count
FROM friendships f1
JOIN friendships f2 ON f1.friend_id = f2.user_id
JOIN friendships f3 ON f2.friend_id = f3.user_id
    AND f3.friend_id = f1.user_id
WHERE f1.user_id < f1.friend_id  -- avoid double counting
GROUP BY f1.user_id;

-- 5. Two-hop neighbors (feature for link prediction)
SELECT 
    f1.user_id AS source,
    f2.friend_id AS two_hop_neighbor,
    COUNT(DISTINCT f1.friend_id) AS paths_between
FROM friendships f1
JOIN friendships f2 ON f1.friend_id = f2.user_id
WHERE f2.friend_id != f1.user_id
GROUP BY f1.user_id, f2.friend_id;
```

### Interview Tip
Graph features are expensive to compute in SQL for large datasets. Mention that for production-scale graph analytics, tools like Apache Spark GraphX or Neo4j are more efficient, but SQL works well for prototyping and small-to-medium graphs.

---

## Question 12
- [ ] Done

**What are SQL Common Table Expressions (CTEs) and how can they be used for feature generation?**

### Definition
A Common Table Expression (CTE) is a temporary named result set defined within a `WITH` clause. It exists only for the duration of the query and makes complex feature engineering pipelines readable, modular, and maintainable.

### Syntax
```sql
WITH cte_name AS (
    SELECT ...
)
SELECT * FROM cte_name;
```

### CTE vs Subquery vs Temp Table

| Feature | CTE | Subquery | Temp Table |
|---------|-----|----------|------------|
| Readability | High | Low (nested) | Medium |
| Reusability in query | Yes (reference multiple times) | No | Yes |
| Persistence | Query scope only | Inline | Session scope |
| Recursive support | Yes | No | No |

### Code Examples
```sql
-- Multi-step feature engineering pipeline using chained CTEs
WITH base_metrics AS (
    -- Step 1: Aggregate order-level features per customer
    SELECT 
        customer_id,
        COUNT(*) AS total_orders,
        AVG(amount) AS avg_order_value,
        MAX(order_date) AS last_order_date,
        MIN(order_date) AS first_order_date
    FROM orders
    GROUP BY customer_id
),
recency_features AS (
    -- Step 2: Calculate recency features
    SELECT 
        customer_id,
        total_orders,
        avg_order_value,
        DATEDIFF(DAY, last_order_date, GETDATE()) AS days_since_last_order,
        DATEDIFF(DAY, first_order_date, last_order_date) AS customer_lifespan_days
    FROM base_metrics
),
behavioral_features AS (
    -- Step 3: Add behavioral features
    SELECT 
        r.*,
        CASE 
            WHEN r.days_since_last_order > 90 THEN 1 ELSE 0 
        END AS is_dormant,
        r.total_orders * 1.0 / NULLIF(r.customer_lifespan_days, 0) AS order_frequency
    FROM recency_features r
)
-- Final: Join all features into ML-ready dataset
SELECT 
    b.*,
    c.segment,
    c.signup_channel
FROM behavioral_features b
JOIN customers c ON b.customer_id = c.customer_id;
```

### Interview Tip
CTEs shine in interviews because they show you can decompose complex problems into clear, logical steps. Always name CTEs descriptively (e.g., `recency_features` not `t1`).

---

## Question 13
- [ ] Done

**Explain the role of partitioning in large-scale SQL databases.**

### Definition
Partitioning divides a large table into smaller, more manageable segments (partitions) based on column values, while the table remains logically unified. Queries automatically target only relevant partitions (partition pruning), dramatically improving performance.

### Types of Partitioning

| Type | Strategy | Best For |
|------|----------|----------|
| Range | Value ranges (dates, IDs) | Time-series data |
| List | Explicit value lists | Region, category columns |
| Hash | Hash function on column | Even distribution |
| Composite | Combination of above | Multi-dimensional queries |

### Code Examples
```sql
-- 1. Range partitioning by date (PostgreSQL)
CREATE TABLE transactions (
    txn_id SERIAL,
    customer_id INT,
    amount DECIMAL(10,2),
    txn_date DATE
) PARTITION BY RANGE (txn_date);

CREATE TABLE transactions_2024 PARTITION OF transactions
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
CREATE TABLE transactions_2025 PARTITION OF transactions
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- 2. Query benefits from partition pruning automatically
SELECT customer_id, SUM(amount)
FROM transactions
WHERE txn_date BETWEEN '2025-01-01' AND '2025-06-30'
GROUP BY customer_id;
-- Only scans transactions_2025, skips 2024 entirely

-- 3. List partitioning by region
CREATE TABLE sales (
    sale_id INT,
    region VARCHAR(20),
    amount DECIMAL(10,2)
) PARTITION BY LIST (region);

CREATE TABLE sales_na PARTITION OF sales FOR VALUES IN ('US', 'CA', 'MX');
CREATE TABLE sales_eu PARTITION OF sales FOR VALUES IN ('UK', 'DE', 'FR');

-- 4. Hash partitioning for even distribution
CREATE TABLE user_events (
    event_id BIGINT,
    user_id INT,
    event_type VARCHAR(50)
) PARTITION BY HASH (user_id);

CREATE TABLE user_events_p0 PARTITION OF user_events FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE user_events_p1 PARTITION OF user_events FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE user_events_p2 PARTITION OF user_events FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE user_events_p3 PARTITION OF user_events FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

### ML Relevance
- **Faster feature engineering**: Window functions and aggregations run on smaller partitions
- **Efficient training data extraction**: Pull specific time ranges without full table scans
- **Data lifecycle management**: Archive old partitions, keep recent data hot

### Interview Tip
Always mention **partition pruning** — the query optimizer skips irrelevant partitions entirely, which is the primary performance benefit. Also note that partitioning helps with maintenance tasks like `VACUUM`, index rebuilds, and data archival.

---

## Question 14
- [ ] Done

**Describe how you could use SQL to report the performance metrics of a Machine Learning model.**

### Definition
After deploying an ML model, its predictions are stored in a database alongside actual outcomes. SQL can compute standard classification and regression metrics directly from these tables, enabling real-time dashboards and monitoring.

### Code Examples
```sql
-- Assuming a predictions table with columns:
-- actual_label (0/1), predicted_label (0/1), predicted_score (float)

-- 1. Confusion matrix components
SELECT
    SUM(CASE WHEN actual_label = 1 AND predicted_label = 1 THEN 1 ELSE 0 END) AS true_positive,
    SUM(CASE WHEN actual_label = 0 AND predicted_label = 1 THEN 1 ELSE 0 END) AS false_positive,
    SUM(CASE WHEN actual_label = 1 AND predicted_label = 0 THEN 1 ELSE 0 END) AS false_negative,
    SUM(CASE WHEN actual_label = 0 AND predicted_label = 0 THEN 1 ELSE 0 END) AS true_negative
FROM model_predictions
WHERE model_version = 'v2.3';

-- 2. Classification metrics: Accuracy, Precision, Recall, F1
WITH confusion AS (
    SELECT
        SUM(CASE WHEN actual_label = 1 AND predicted_label = 1 THEN 1 ELSE 0 END) AS tp,
        SUM(CASE WHEN actual_label = 0 AND predicted_label = 1 THEN 1 ELSE 0 END) AS fp,
        SUM(CASE WHEN actual_label = 1 AND predicted_label = 0 THEN 1 ELSE 0 END) AS fn,
        SUM(CASE WHEN actual_label = 0 AND predicted_label = 0 THEN 1 ELSE 0 END) AS tn
    FROM model_predictions
    WHERE model_version = 'v2.3'
)
SELECT 
    (tp + tn) * 1.0 / (tp + tn + fp + fn) AS accuracy,
    tp * 1.0 / NULLIF(tp + fp, 0) AS precision,
    tp * 1.0 / NULLIF(tp + fn, 0) AS recall,
    2.0 * tp / NULLIF(2 * tp + fp + fn, 0) AS f1_score
FROM confusion;

-- 3. Regression metrics: MAE, RMSE, R²
WITH errors AS (
    SELECT 
        actual_value,
        predicted_value,
        actual_value - predicted_value AS error
    FROM regression_predictions
)
SELECT
    AVG(ABS(error)) AS mae,
    SQRT(AVG(error * error)) AS rmse,
    1.0 - SUM(error * error) / 
        NULLIF(SUM(POWER(actual_value - (SELECT AVG(actual_value) FROM errors), 2)), 0) AS r_squared
FROM errors;

-- 4. Track metrics over time for model monitoring
SELECT 
    DATE_TRUNC('week', prediction_timestamp) AS week,
    COUNT(*) AS predictions,
    AVG(CASE WHEN actual_label = predicted_label THEN 1.0 ELSE 0.0 END) AS weekly_accuracy,
    AVG(predicted_score) AS avg_confidence
FROM model_predictions
WHERE model_version = 'v2.3'
GROUP BY DATE_TRUNC('week', prediction_timestamp)
ORDER BY week;
```

### Interview Tip
Highlight that SQL-based metric reporting enables **automated monitoring** — you can schedule these queries to detect model degradation (e.g., accuracy dropping below a threshold) and trigger retraining alerts.

---

## Question 15
- [ ] Done

**Describe how you would version control the datasets used for building Machine Learning models in SQL.**

### Definition
Dataset version control in SQL ensures reproducibility of ML experiments by tracking which data was used to train, validate, and test each model version. It maintains an audit trail linking every model to its exact training data.

### Strategies

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| Snapshot tables | Copy entire dataset per version | Simple, full reproducibility | Storage-heavy |
| Temporal tables | System-versioned with time travel | Built-in, no duplication | DB-specific support |
| Hash-based tracking | Store data fingerprints | Lightweight | Cannot reconstruct data |
| Delta/changelog | Track row-level inserts/updates/deletes | Storage-efficient | Complex to reconstruct |

### Code Examples
```sql
-- 1. Snapshot approach: Save versioned training sets
CREATE TABLE training_snapshots (
    snapshot_id INT,
    version VARCHAR(20),
    created_at DATETIME DEFAULT GETDATE(),
    description VARCHAR(500)
);

CREATE TABLE training_data_v2 AS
SELECT * FROM feature_store
WHERE snapshot_date = '2025-06-01';

INSERT INTO training_snapshots (snapshot_id, version, description)
VALUES (2, 'v2.0', 'Added recency features, filtered inactive users');

-- 2. Temporal tables (SQL Server system-versioned)
CREATE TABLE customer_features (
    customer_id INT PRIMARY KEY,
    total_orders INT,
    avg_order_value DECIMAL(10,2),
    valid_from DATETIME2 GENERATED ALWAYS AS ROW START,
    valid_to DATETIME2 GENERATED ALWAYS AS ROW END,
    PERIOD FOR SYSTEM_TIME (valid_from, valid_to)
) WITH (SYSTEM_VERSIONING = ON);

-- Query data as it was on a specific training date
SELECT * FROM customer_features
FOR SYSTEM_TIME AS OF '2025-01-15T00:00:00';

-- 3. Hash-based integrity check
SELECT 
    'v2.0' AS dataset_version,
    COUNT(*) AS row_count,
    CHECKSUM_AGG(CHECKSUM(*)) AS data_fingerprint
FROM training_data_v2;

-- 4. Model-to-data lineage tracking
CREATE TABLE model_registry (
    model_id INT PRIMARY KEY,
    model_version VARCHAR(20),
    training_table VARCHAR(100),
    training_row_count INT,
    data_hash BIGINT,
    trained_at DATETIME,
    hyperparameters VARCHAR(1000)
);

INSERT INTO model_registry VALUES (
    1, 'v2.3', 'training_data_v2', 50000, 
    -1283749182, GETDATE(), 
    '{"lr": 0.01, "epochs": 100, "batch_size": 32}'
);
```

### Interview Tip
Mention that temporal tables (SQL:2011 standard) are the most elegant solution since the database engine automatically tracks all historical states — you can always reproduce the exact dataset used for any past training run.

---

## Question 16
- [ ] Done

**What is Data Lineage , and how can you track it using SQL?**

### Definition
Data lineage is the end-to-end tracking of data's origin, movement, and transformations throughout its lifecycle — from raw source to final ML features. It answers: *Where did this data come from? What happened to it? Who changed it?*

### Why It Matters for ML

| Concern | How Lineage Helps |
|---------|-------------------|
| Debugging bad predictions | Trace which raw data produced which features |
| Regulatory compliance (GDPR) | Prove data handling and consent tracking |
| Reproducibility | Reconstruct exact pipeline for any model version |
| Impact analysis | Know which models break if a source table changes |

### Code Examples
```sql
-- 1. Transformation audit log table
CREATE TABLE data_lineage_log (
    lineage_id INT IDENTITY PRIMARY KEY,
    source_table VARCHAR(100),
    target_table VARCHAR(100),
    transformation VARCHAR(500),
    row_count_before INT,
    row_count_after INT,
    executed_by VARCHAR(100),
    executed_at DATETIME DEFAULT GETDATE()
);

-- Log each transformation step
INSERT INTO data_lineage_log 
    (source_table, target_table, transformation, row_count_before, row_count_after, executed_by)
VALUES 
    ('raw_orders', 'clean_orders', 'Removed NULLs in amount, filtered test accounts', 1000000, 985432, 'etl_pipeline_v3');

-- 2. Column-level lineage tracking
CREATE TABLE column_lineage (
    target_table VARCHAR(100),
    target_column VARCHAR(100),
    source_table VARCHAR(100),
    source_column VARCHAR(100),
    transformation_logic VARCHAR(500)
);

INSERT INTO column_lineage VALUES
    ('customer_features', 'avg_order_value', 'orders', 'amount', 'AVG(amount) GROUP BY customer_id'),
    ('customer_features', 'days_since_last_order', 'orders', 'order_date', 'DATEDIFF(DAY, MAX(order_date), GETDATE())'),
    ('customer_features', 'is_dormant', 'customer_features', 'days_since_last_order', 'CASE WHEN > 90 THEN 1 ELSE 0');

-- 3. Query full lineage for a feature
SELECT 
    cl.target_table,
    cl.target_column,
    cl.source_table,
    cl.source_column,
    cl.transformation_logic
FROM column_lineage cl
WHERE cl.target_table = 'customer_features'
ORDER BY cl.target_column;

-- 4. Track row-level provenance with audit triggers
CREATE TRIGGER trg_orders_audit
ON orders
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
    INSERT INTO orders_audit (order_id, action, changed_by, changed_at, old_values, new_values)
    SELECT 
        COALESCE(i.order_id, d.order_id),
        CASE 
            WHEN d.order_id IS NULL THEN 'INSERT'
            WHEN i.order_id IS NULL THEN 'DELETE'
            ELSE 'UPDATE'
        END,
        SYSTEM_USER,
        GETDATE(),
        (SELECT d.* FOR JSON PATH),
        (SELECT i.* FOR JSON PATH)
    FROM inserted i
    FULL OUTER JOIN deleted d ON i.order_id = d.order_id;
END;

-- 5. Impact analysis: Which models are affected by a source change?
SELECT 
    mr.model_version,
    mr.training_table,
    cl.source_table AS upstream_dependency
FROM model_registry mr
JOIN column_lineage cl ON mr.training_table = cl.target_table
WHERE cl.source_table = 'raw_orders'  -- If this changes, which models are affected?
GROUP BY mr.model_version, mr.training_table, cl.source_table;
```

### Interview Tip
Emphasize that data lineage is not just a compliance requirement — it is essential for debugging ML models in production. When a model's accuracy drops, lineage lets you trace back to the exact upstream data change that caused it.

---

---

## Question 17
- [ ] Done

**Can you explain the use of indexes in databases and how they relate to Machine Learning?**

**Answer:**

### Database Indexes

An **index** is a data structure (typically a **B-tree** or **hash table**) that speeds up data retrieval by providing quick lookup paths, similar to a book's index.

### Types of Indexes

| Index Type | Description | Best For |
|-----------|-------------|----------|
| **B-tree** | Balanced tree, default type | Range queries, sorting, equality |
| **Hash** | Direct key-value lookup | Exact equality matches |
| **Bitmap** | Bit arrays per value | Low-cardinality columns |
| **GIN/GiST** | Generalized inverted/search | Full-text search, JSON, arrays |
| **Covering** | Includes all query columns | Avoiding table lookups |

### Index Creation

```sql
-- Single column index
CREATE INDEX idx_customer_email ON customers(email);

-- Composite index (order matters!)
CREATE INDEX idx_orders_user_date ON orders(user_id, order_date);

-- Unique index (also enforces constraint)
CREATE UNIQUE INDEX idx_unique_email ON customers(email);

-- Partial index (PostgreSQL)
CREATE INDEX idx_active_users ON users(last_login) WHERE is_active = true;
```

### Relevance to Machine Learning

1. **Feature extraction speed**: Indexed columns allow fast aggregation queries for feature engineering
2. **Training data retrieval**: Indexes on timestamp/partition columns enable efficient batch extraction
3. **Real-time serving**: Index on user_id enables fast feature lookup during inference
4. **Sampling**: Indexed primary keys allow efficient random sampling for training sets

```sql
-- Fast feature extraction with proper indexes
CREATE INDEX idx_transactions_user_date ON transactions(user_id, transaction_date);

-- This query is now fast:
SELECT user_id,
    COUNT(*) AS txn_count,
    AVG(amount) AS avg_amount
FROM transactions
WHERE transaction_date >= '2025-01-01'
GROUP BY user_id;
```

### Trade-offs
- **Pros**: Faster reads, efficient WHERE/JOIN/ORDER BY
- **Cons**: Slower writes (INSERT/UPDATE), extra storage, maintenance overhead

> **Interview Tip:** For ML pipelines, index the columns used in WHERE clauses and JOINs of your feature extraction queries. Over-indexing slows down writes; use `EXPLAIN ANALYZE` to verify indexes are actually being used.

---

## Question 18
- [ ] Done

**Explain the importance of data normalization in SQL and how it affects Machine Learning models.**

**Answer:**

### Database Normalization

Database normalization organizes tables to **reduce redundancy** and **improve data integrity** through a series of normal forms.

### Normal Forms

| Normal Form | Rule | Example Violation |
|------------|------|--------------------|
| **1NF** | Atomic values, no repeating groups | Storing `"red,blue"` in one cell |
| **2NF** | 1NF + no partial dependencies | Non-key column depends on part of composite key |
| **3NF** | 2NF + no transitive dependencies | `zip_code → city` stored in orders table |
| **BCNF** | Every determinant is a candidate key | Stronger version of 3NF |

### Normalization Example

```sql
-- DENORMALIZED (bad: redundancy)
-- | order_id | customer_name | customer_email | product | price |
-- Repeats customer info for every order

-- NORMALIZED (3NF)
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100) UNIQUE
);

CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    product VARCHAR(100),
    price DECIMAL(10, 2)
);
```

### Impact on Machine Learning

| Aspect | Normalized DB | Denormalized/Flat Table |
|--------|--------------|-------------------------|
| **Data quality** | Higher (no redundancy) | Risk of inconsistencies |
| **Feature extraction** | Requires JOINs | Direct queries |
| **Query performance** | Slower (joins) | Faster (pre-joined) |
| **ML best practice** | Source of truth | Feature store / training table |

```sql
-- ML workflow: normalize for storage, denormalize for training
CREATE TABLE ml_training_data AS
SELECT 
    o.order_id,
    c.name, c.email,
    o.product, o.price,
    COUNT(*) OVER (PARTITION BY c.customer_id) AS total_orders,
    AVG(o.price) OVER (PARTITION BY c.customer_id) AS avg_order_value
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id;
```

> **Interview Tip:** Distinguish **database normalization** (reducing redundancy, normal forms) from **data normalization** in ML (scaling features to [0,1] or z-scores). For ML, you typically denormalize data into flat feature tables for training — but maintain normalized source tables for data integrity.

---

## Question 19
- [ ] Done

**How would you optimize a SQL query that seems to be running slowly?**

### SQL Query Optimization Checklist

### Step 1: Analyze the Query Plan

```sql
-- PostgreSQL
EXPLAIN ANALYZE SELECT * FROM orders WHERE customer_id = 42;

-- MySQL
EXPLAIN SELECT * FROM orders WHERE customer_id = 42;

-- Look for: Seq Scan (full table scan), high row estimates, nested loops on large tables
```

### Step 2: Common Optimizations

| Problem | Solution | Example |
|---------|----------|--------|
| **No index** on WHERE column | Add index | `CREATE INDEX idx_cust ON orders(customer_id)` |
| **SELECT *** | Select only needed columns | `SELECT id, name FROM ...` |
| **N+1 queries** | Use JOINs instead | Replace loop queries with single JOIN |
| **Functions on indexed columns** | Avoid wrapping in functions | `WHERE date_col >= '2025-01-01'` not `WHERE YEAR(date_col) = 2025` |
| **Large IN lists** | Use JOINs or temp tables | Join against a values table |
| **Missing statistics** | Update table statistics | `ANALYZE table_name;` |
| **Correlated subqueries** | Rewrite as JOINs | Replace `WHERE x IN (SELECT ...)` with JOIN |

### Step 3: Advanced Techniques

```sql
-- Partition large tables
CREATE TABLE events (
    event_id BIGINT,
    event_date DATE,
    data JSONB
) PARTITION BY RANGE (event_date);

-- Use materialized views for complex aggregations
CREATE MATERIALIZED VIEW daily_stats AS
SELECT DATE(created_at) AS day, COUNT(*), AVG(amount)
FROM transactions GROUP BY DATE(created_at);

-- Batch processing instead of one huge query
SELECT * FROM large_table WHERE id BETWEEN 1 AND 100000;
SELECT * FROM large_table WHERE id BETWEEN 100001 AND 200000;
```

> **Interview Tip:** Always start with `EXPLAIN ANALYZE`. The #1 optimization is usually **adding the right index**. For ML feature extraction, pre-compute expensive aggregations in materialized views and schedule refreshes.

---

## Question 20
- [ ] Done

**How do you handle missing values in a SQL dataset?**

### Handling Missing Values (NULLs) in SQL

### Detecting Missing Values

```sql
-- Count NULLs per column
SELECT 
    COUNT(*) AS total_rows,
    COUNT(age) AS non_null_age,
    COUNT(*) - COUNT(age) AS null_age,
    ROUND(100.0 * (COUNT(*) - COUNT(age)) / COUNT(*), 2) AS null_pct_age
FROM patients;

-- Find rows with any NULL in critical columns
SELECT * FROM patients
WHERE age IS NULL OR weight IS NULL OR diagnosis IS NULL;
```

### Strategies for Handling NULLs

| Strategy | SQL Implementation | When to Use |
|----------|-------------------|-------------|
| **Drop rows** | `WHERE col IS NOT NULL` | Few NULLs, large dataset |
| **Default value** | `COALESCE(col, 0)` | Known safe default |
| **Mean/Median imputation** | Subquery with AVG | Numerical columns, MCAR |
| **Mode imputation** | Subquery with MODE | Categorical columns |
| **Forward fill** | Window function LAG | Time series data |
| **Flag + impute** | Add indicator column | Preserve missingness signal |

```sql
-- Method 1: Replace NULLs with default
SELECT COALESCE(age, 0) AS age,
       COALESCE(income, 'Unknown') AS income
FROM customers;

-- Method 2: Mean imputation
SELECT 
    COALESCE(age, (SELECT AVG(age) FROM patients)) AS age_imputed
FROM patients;

-- Method 3: Forward fill (last known value) for time series
SELECT 
    date, sensor_id,
    COALESCE(reading, 
        LAG(reading IGNORE NULLS) OVER (PARTITION BY sensor_id ORDER BY date)
    ) AS reading_filled
FROM sensor_data;

-- Method 4: Flag missing + impute (best for ML)
SELECT *,
    CASE WHEN age IS NULL THEN 1 ELSE 0 END AS age_is_missing,
    COALESCE(age, (SELECT AVG(age) FROM patients)) AS age_imputed
FROM patients;
```

> **Interview Tip:** For ML, creating a **missing indicator** column alongside imputation often improves model performance — missingness itself can be informative. Never drop columns with >50% NULLs without checking if the missingness pattern is predictive.

---

## Question 21
- [ ] Done

**How would you merge multiple result sets in SQL without duplicates?**

### Merging Result Sets Without Duplicates

### UNION vs. UNION ALL

| Operator | Duplicates | Performance |
|----------|-----------|-------------|
| **UNION** | Removes duplicates | Slower (requires sort/hash) |
| **UNION ALL** | Keeps duplicates | Faster (no dedup step) |

```sql
-- UNION: combines and removes duplicates
SELECT name, email FROM customers_us
UNION
SELECT name, email FROM customers_eu;

-- UNION ALL: keeps all rows (faster, use when duplicates impossible)
SELECT name, email FROM customers_us
UNION ALL
SELECT name, email FROM customers_eu;
```

### Advanced Merging Patterns

```sql
-- Merge 3+ tables with source tracking
SELECT name, email, 'US' AS source FROM customers_us
UNION
SELECT name, email, 'EU' AS source FROM customers_eu
UNION
SELECT name, email, 'APAC' AS source FROM customers_apac;

-- INTERSECT: only rows in BOTH sets
SELECT user_id FROM premium_users
INTERSECT
SELECT user_id FROM active_users;  -- premium AND active

-- EXCEPT / MINUS: rows in first set but NOT second
SELECT user_id FROM all_users
EXCEPT
SELECT user_id FROM churned_users;  -- retained users
```

### ML Use Case: Combining Training Data

```sql
-- Combine positive and negative samples for training
SELECT features.*, 1 AS label FROM fraud_cases
JOIN features USING (transaction_id)
UNION ALL
SELECT features.*, 0 AS label FROM legitimate_cases
JOIN features USING (transaction_id);
```

### Rules for UNION
1. All SELECT statements must have the **same number of columns**
2. Corresponding columns must have **compatible data types**
3. Column names come from the **first SELECT**

> **Interview Tip:** Use `UNION ALL` by default for performance; only switch to `UNION` when deduplication is needed. For ML data pipelines, `UNION ALL` with an explicit `DISTINCT` at the end gives you more control over which columns define uniqueness.

---

## Question 22
- [ ] Done

**How would you handle very large datasets in SQL for Machine Learning purposes?**

### Handling Large Datasets in SQL for ML

### Strategies Overview

| Strategy | Technique | Scale |
|----------|-----------|-------|
| **Partitioning** | Split tables by date/key | Millions to billions of rows |
| **Sampling** | Extract representative subset | Training data selection |
| **Incremental processing** | Process in batches | Continuous pipelines |
| **Materialized views** | Pre-compute aggregations | Expensive feature queries |
| **Columnar storage** | Use columnar formats | Analytical queries |

### Partitioning

```sql
-- Partition by date range
CREATE TABLE events (
    event_id BIGINT,
    event_date DATE,
    user_id INT,
    data JSONB
) PARTITION BY RANGE (event_date);

CREATE TABLE events_2025_q1 PARTITION OF events
    FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');

-- Queries automatically scan only relevant partitions
SELECT * FROM events WHERE event_date >= '2025-03-01';  -- scans only Q1 partition
```

### Efficient Sampling

```sql
-- Random sampling (PostgreSQL)
SELECT * FROM large_table TABLESAMPLE BERNOULLI(1);  -- ~1% sample

-- Stratified sampling for balanced classes
WITH ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY label ORDER BY RANDOM()) AS rn
    FROM training_data
)
SELECT * FROM ranked WHERE rn <= 10000;  -- 10k per class
```

### Batch Processing

```sql
-- Process in chunks using cursor-style pagination
SELECT * FROM features
WHERE id > :last_processed_id
ORDER BY id
LIMIT 50000;  -- process 50k at a time
```

### Feature Computation at Scale

```sql
-- Materialized view for expensive features
CREATE MATERIALIZED VIEW user_features AS
SELECT 
    user_id,
    COUNT(*) AS total_orders,
    AVG(amount) AS avg_order_value,
    MAX(order_date) AS last_order_date
FROM orders
GROUP BY user_id;

-- Refresh periodically
REFRESH MATERIALIZED VIEW CONCURRENTLY user_features;
```

> **Interview Tip:** For very large ML datasets, the typical pattern is: **partition tables** → **sample for development** → **train on full data using batch extraction** → **materialize features**. Mention tools like Apache Spark SQL or BigQuery for datasets that exceed single-node capacity.

---

## Question 23
- [ ] Done

**Discuss how you would design a system to regularly feed a Machine Learning model with SQL data**

### ML Data Pipeline Design with SQL

### Architecture Overview

```
Source DB → ETL/ELT → Feature Store → Training Pipeline → Model
   ↑                        ↓
Scheduler (Airflow)    Prediction Service
```

### Key Components

| Component | Tool Options | Purpose |
|-----------|-------------|--------|
| **Scheduler** | Airflow, Prefect, dbt | Orchestrate pipeline runs |
| **ETL** | dbt, Spark, SQL scripts | Transform raw data to features |
| **Feature Store** | Feast, Tecton, SQL views | Consistent feature access |
| **Data Validation** | Great Expectations, dbt tests | Ensure data quality |
| **Model Registry** | MLflow, W&B | Track model versions |

### Implementation

```sql
-- Step 1: Scheduled feature extraction (run daily via Airflow)
CREATE TABLE daily_features AS
SELECT 
    user_id,
    DATE(NOW()) AS feature_date,
    COUNT(*) FILTER (WHERE txn_date >= NOW() - INTERVAL '30 days') AS txn_count_30d,
    AVG(amount) FILTER (WHERE txn_date >= NOW() - INTERVAL '30 days') AS avg_amount_30d,
    MAX(txn_date) AS last_active
FROM transactions
GROUP BY user_id;

-- Step 2: Data quality checks
SELECT 
    CASE WHEN COUNT(*) = 0 THEN 'FAIL: No rows' 
         WHEN MIN(txn_count_30d) < 0 THEN 'FAIL: Negative counts'
         ELSE 'PASS' END AS quality_check
FROM daily_features;

-- Step 3: Incremental updates (only process new data)
INSERT INTO feature_history
SELECT * FROM daily_features
WHERE feature_date = CURRENT_DATE
ON CONFLICT (user_id, feature_date) DO UPDATE 
SET txn_count_30d = EXCLUDED.txn_count_30d;
```

### Python Integration

```python
# Airflow DAG example
from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator

dag = DAG('ml_feature_pipeline', schedule_interval='@daily')

extract_features = PostgresOperator(
    task_id='extract_features',
    sql='sql/daily_features.sql',
    dag=dag
)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_and_evaluate,
    dag=dag
)

extract_features >> train_model
```

> **Interview Tip:** Emphasize **idempotency** (running the pipeline twice produces the same result), **data versioning** (snapshots of training data), and **monitoring** (alerts on data drift or quality failures). Mention the shift from batch to real-time with tools like Kafka + Flink.

---

## Question 24
- [ ] Done

**How would you extract and prepare a confusion matrix for a classification problem using SQL?**

### Building a Confusion Matrix in SQL

A confusion matrix compares **predicted labels** vs. **actual labels** for a classification model.

### Step 1: Predictions Table Structure

```sql
-- Assume we have a predictions table
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    actual_label VARCHAR(50),
    predicted_label VARCHAR(50),
    prediction_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Step 2: Generate Confusion Matrix

```sql
-- Binary classification confusion matrix
SELECT 
    actual_label,
    SUM(CASE WHEN predicted_label = 'positive' THEN 1 ELSE 0 END) AS predicted_positive,
    SUM(CASE WHEN predicted_label = 'negative' THEN 1 ELSE 0 END) AS predicted_negative,
    COUNT(*) AS total
FROM predictions
GROUP BY actual_label
ORDER BY actual_label;

-- Result:
-- | actual_label | predicted_positive | predicted_negative | total |
-- |-------------|-------------------|-------------------|-------|
-- | negative    | 5 (FP)            | 90 (TN)           | 95    |
-- | positive    | 80 (TP)           | 15 (FN)           | 95    |
```

### Step 3: Calculate Metrics from Confusion Matrix

```sql
WITH cm AS (
    SELECT
        SUM(CASE WHEN actual_label = 'positive' AND predicted_label = 'positive' THEN 1 ELSE 0 END) AS tp,
        SUM(CASE WHEN actual_label = 'negative' AND predicted_label = 'positive' THEN 1 ELSE 0 END) AS fp,
        SUM(CASE WHEN actual_label = 'positive' AND predicted_label = 'negative' THEN 1 ELSE 0 END) AS fn,
        SUM(CASE WHEN actual_label = 'negative' AND predicted_label = 'negative' THEN 1 ELSE 0 END) AS tn
    FROM predictions
)
SELECT
    tp, fp, fn, tn,
    ROUND(1.0 * (tp + tn) / (tp + fp + fn + tn), 4) AS accuracy,
    ROUND(1.0 * tp / NULLIF(tp + fp, 0), 4) AS precision,
    ROUND(1.0 * tp / NULLIF(tp + fn, 0), 4) AS recall,
    ROUND(2.0 * tp / NULLIF(2 * tp + fp + fn, 0), 4) AS f1_score
FROM cm;
```

### Multi-Class Confusion Matrix

```sql
-- Dynamic pivot for any number of classes
SELECT 
    actual_label,
    predicted_label,
    COUNT(*) AS count
FROM predictions
GROUP BY actual_label, predicted_label
ORDER BY actual_label, predicted_label;
```

> **Interview Tip:** Computing metrics in SQL is useful for **monitoring dashboards** (e.g., Grafana querying a predictions table). For complex analysis, extract to Python and use `sklearn.metrics.confusion_matrix()`. The SQL approach enables real-time metric tracking without Python.

---

## Question 25
- [ ] Done

**How would you log and track predictions made by a Machine Learning model within a SQL environment?**

### ML Prediction Logging in SQL

### Prediction Log Schema

```sql
CREATE TABLE prediction_log (
    prediction_id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    input_features JSONB,                    -- store raw input
    predicted_label VARCHAR(100),
    prediction_score FLOAT,
    prediction_probabilities JSONB,          -- class probabilities
    actual_label VARCHAR(100),               -- filled later (ground truth)
    latency_ms INT,                          -- inference time
    created_at TIMESTAMP DEFAULT NOW(),
    request_id UUID DEFAULT gen_random_uuid()
);

-- Indexes for common queries
CREATE INDEX idx_pred_model ON prediction_log(model_name, model_version);
CREATE INDEX idx_pred_time ON prediction_log(created_at);
CREATE INDEX idx_pred_label ON prediction_log(predicted_label);
```

### Logging from Python

```python
import psycopg2
import json
from datetime import datetime

def log_prediction(conn, model_name, version, features, prediction, score, latency):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO prediction_log 
            (model_name, model_version, input_features, predicted_label, 
             prediction_score, latency_ms)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (model_name, version, json.dumps(features), prediction, score, latency))
    conn.commit()
```

### Monitoring Queries

```sql
-- Daily prediction volume and accuracy
SELECT 
    DATE(created_at) AS day,
    model_version,
    COUNT(*) AS total_predictions,
    AVG(CASE WHEN predicted_label = actual_label THEN 1.0 ELSE 0.0 END) AS accuracy,
    AVG(latency_ms) AS avg_latency_ms
FROM prediction_log
WHERE actual_label IS NOT NULL
GROUP BY DATE(created_at), model_version
ORDER BY day DESC;

-- Detect prediction distribution drift
SELECT 
    predicted_label,
    COUNT(*) AS count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct
FROM prediction_log
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY predicted_label;

-- Compare model versions (A/B testing)
SELECT 
    model_version,
    COUNT(*) AS predictions,
    AVG(prediction_score) AS avg_confidence,
    AVG(latency_ms) AS avg_latency
FROM prediction_log
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY model_version;
```

> **Interview Tip:** Prediction logging enables **model monitoring**, **A/B testing**, **debugging**, and **retraining** (using logged predictions as ground truth after labeling). Mention GDPR considerations — input features may contain PII and need encryption or anonymization.

---

## Question 26
- [ ] Done

**Discuss how to manage the entire lifecycle of a Machine Learning model using SQL tools**

### ML Lifecycle Management with SQL

### Lifecycle Stages

```
Data Collection → Feature Engineering → Training → Evaluation → Deployment → Monitoring → Retraining
      ↑                                                                              ↓
      └────────────────────────────────────────────────────────────────────────┘
```

### SQL Schema for Lifecycle Management

```sql
-- 1. Data versioning
CREATE TABLE dataset_versions (
    dataset_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    version INT,
    row_count BIGINT,
    schema_hash VARCHAR(64),
    created_at TIMESTAMP DEFAULT NOW(),
    query_used TEXT                     -- SQL that produced this dataset
);

-- 2. Experiment tracking
CREATE TABLE experiments (
    experiment_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    dataset_id INT REFERENCES dataset_versions(dataset_id),
    hyperparameters JSONB,
    metrics JSONB,                      -- {"accuracy": 0.95, "f1": 0.92}
    artifact_path VARCHAR(500),         -- path to saved model
    status VARCHAR(20) DEFAULT 'running',
    created_at TIMESTAMP DEFAULT NOW()
);

-- 3. Model registry
CREATE TABLE model_registry (
    registry_id SERIAL PRIMARY KEY,
    experiment_id INT REFERENCES experiments(experiment_id),
    stage VARCHAR(20) DEFAULT 'staging', -- staging / production / archived
    approved_by VARCHAR(100),
    deployed_at TIMESTAMP,
    notes TEXT
);

-- 4. Prediction monitoring (see Q25)
-- 5. Retraining triggers
CREATE TABLE retraining_triggers (
    trigger_id SERIAL PRIMARY KEY,
    trigger_type VARCHAR(50),            -- 'accuracy_drop', 'data_drift', 'scheduled'
    threshold FLOAT,
    current_value FLOAT,
    triggered_at TIMESTAMP DEFAULT NOW(),
    action_taken VARCHAR(100)
);
```

### Automated Monitoring

```sql
-- Check if model needs retraining
INSERT INTO retraining_triggers (trigger_type, threshold, current_value, action_taken)
SELECT 
    'accuracy_drop', 0.90, daily_accuracy,
    CASE WHEN daily_accuracy < 0.90 THEN 'retrain_initiated' ELSE 'no_action' END
FROM (
    SELECT AVG(CASE WHEN predicted_label = actual_label THEN 1.0 ELSE 0.0 END) AS daily_accuracy
    FROM prediction_log
    WHERE created_at >= NOW() - INTERVAL '24 hours'
    AND actual_label IS NOT NULL
) daily;

-- Model comparison for promotion decisions
SELECT 
    e.model_name, e.model_version,
    e.metrics->>'accuracy' AS accuracy,
    e.metrics->>'f1' AS f1_score,
    mr.stage
FROM experiments e
JOIN model_registry mr ON e.experiment_id = mr.experiment_id
WHERE e.model_name = 'fraud_detector'
ORDER BY (e.metrics->>'f1')::FLOAT DESC;
```

### Tools Integration

| Stage | SQL Role | Supporting Tool |
|-------|----------|----------------|
| Data | Source of truth, feature extraction | dbt, Airflow |
| Training | Dataset versioning, experiment logging | MLflow, W&B |
| Deployment | Prediction logging, A/B routing | Seldon, BentoML |
| Monitoring | Drift detection, metric tracking | Evidently, Grafana |
| Retraining | Trigger detection, data preparation | Airflow, Prefect |

> **Interview Tip:** SQL serves as the **backbone** for ML lifecycle management — it stores data, features, predictions, and monitoring metrics. Dedicated MLOps tools (MLflow, Kubeflow) handle model artifacts and deployment, but SQL remains the central data layer. Emphasize the importance of **reproducibility** (tracking which data version trained which model).

---
