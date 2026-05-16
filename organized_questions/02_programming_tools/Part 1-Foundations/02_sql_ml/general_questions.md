# SQL ML Interview Questions - General Questions

## Question 1
- [ ] Done

**What does GROUP BY do in a SQL query?**

### Definition
GROUP BY partitions rows into groups based on column values. Aggregate functions then operate on each group separately.

### Syntax
```sql
SELECT column, AGG_FUNCTION(column)
FROM table
GROUP BY column
HAVING AGG_CONDITION;
```

### Code Example
```sql
-- Basic aggregation
SELECT 
    department,
    COUNT(*) AS employee_count,
    AVG(salary) AS avg_salary,
    MAX(salary) AS max_salary
FROM employees
GROUP BY department
HAVING COUNT(*) >= 5;

-- Multiple column grouping
SELECT 
    department, job_level,
    COUNT(*) AS count,
    AVG(salary) AS avg_salary
FROM employees
GROUP BY department, job_level
ORDER BY department, job_level;
```

### ML Feature Engineering Example
```sql
-- Create customer features
SELECT 
    customer_id,
    COUNT(*) AS total_purchases,
    SUM(amount) AS total_spent,
    AVG(amount) AS avg_purchase,
    MAX(purchase_date) AS last_purchase,
    COUNT(DISTINCT product_category) AS categories_bought
FROM purchases
GROUP BY customer_id;
```

### Rules
- SELECT can only include GROUP BY columns or aggregates
- Use HAVING for filtering on aggregates (not WHERE)
- NULL values form their own group

---

## Question 2
- [ ] Done

**Explain indexes and their importance for ML pipelines.**

### Definition
An index is a B-Tree data structure that enables O(log n) lookups instead of O(n) full table scans.

### Creating Indexes
```sql
-- Single column
CREATE INDEX idx_customer ON orders(customer_id);

-- Composite (order matters!)
CREATE INDEX idx_cust_date ON orders(customer_id, order_date);
```

### ML Impact

| Operation | Without Index | With Index |
|-----------|---------------|------------|
| Feature lookup | Full scan | Direct access |
| JOIN operations | Nested loops | Index scan |
| Real-time prediction | Seconds | Milliseconds |

### Best Practices
- Index columns used in WHERE, JOIN, ORDER BY
- Avoid over-indexing (slows writes)
- Don't index low-cardinality columns

---

## Question 3
- [ ] Done

**How do you handle NULL values in SQL?**

### Definition
NULL represents missing/unknown data. It's not equal to anything, including itself.

### Checking for NULL
```sql
-- Correct
WHERE column IS NULL
WHERE column IS NOT NULL

-- Wrong (never matches!)
WHERE column = NULL
```

### Handling NULLs
```sql
-- Replace NULL with default
SELECT COALESCE(column, 'default_value') AS column;

-- CASE statement
SELECT CASE WHEN column IS NULL THEN 'Unknown' ELSE column END;

-- NULL-safe comparison (MySQL)
WHERE column <=> other_column;
```

### ML Considerations
- Decide imputation strategy before querying
- Track NULL percentages for data quality
- Some models can't handle NULLs - handle in SQL or Python

---

## Question 4
- [ ] Done

**What is a Common Table Expression (CTE)?**

### Definition
A CTE is a named temporary result set defined within a query using WITH clause. Makes complex queries readable.

### Syntax
```sql
WITH cte_name AS (
    SELECT ...
)
SELECT * FROM cte_name;
```

### Code Example
```sql
-- Multiple CTEs for feature engineering
WITH customer_orders AS (
    SELECT 
        customer_id,
        COUNT(*) AS order_count,
        SUM(amount) AS total_spent
    FROM orders
    GROUP BY customer_id
),
customer_returns AS (
    SELECT 
        customer_id,
        COUNT(*) AS return_count
    FROM returns
    GROUP BY customer_id
)
SELECT 
    co.customer_id,
    co.order_count,
    co.total_spent,
    COALESCE(cr.return_count, 0) AS return_count,
    COALESCE(cr.return_count, 0) * 1.0 / co.order_count AS return_rate
FROM customer_orders co
LEFT JOIN customer_returns cr ON co.customer_id = cr.customer_id;
```

### Benefits
- Improved readability over nested subqueries
- Can reference CTE multiple times in query
- Recursive CTEs for hierarchical data

---

## Question 5
- [ ] Done

**What is the difference between UNION and UNION ALL?**

### Definition
Both combine results from multiple SELECT statements.

| Operator | Duplicates | Performance |
|----------|------------|-------------|
| UNION | Removed | Slower (sorts) |
| UNION ALL | Kept | Faster |

### Code Example
```sql
-- UNION: Remove duplicates
SELECT city FROM customers
UNION
SELECT city FROM suppliers;

-- UNION ALL: Keep all rows
SELECT 'train' AS dataset, * FROM train_data
UNION ALL
SELECT 'test' AS dataset, * FROM test_data;
```

### Best Practice
Use UNION ALL unless you specifically need deduplication - it's more efficient.

---

## Question 6
- [ ] Done

**How do you calculate running totals and moving averages in SQL?**

### Code Examples
```sql
-- Running total
SELECT 
    date,
    sales,
    SUM(sales) OVER (ORDER BY date) AS running_total
FROM daily_sales;

-- 7-day moving average
SELECT 
    date,
    sales,
    AVG(sales) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d
FROM daily_sales;

-- Running total per customer
SELECT 
    customer_id,
    transaction_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date
    ) AS cumulative_spent
FROM transactions;
```

### ML Applications
- Cumulative features (lifetime value)
- Smoothed features (reduce noise)
- Trend detection


---

## Question 7
- [ ] Done

**How can you aggregate data in SQL (e.g., COUNT , AVG , SUM , MAX , MIN )?**

**Answer:**

SQL aggregate functions compute a single result from a set of input values and are essential for feature engineering in ML.

### Core Aggregate Functions

| Function | Description | Example |
|----------|-------------|--------|
| `COUNT(*)` | Number of rows | Total transactions |
| `COUNT(DISTINCT col)` | Unique values | Unique customers |
| `SUM(col)` | Total sum | Total revenue |
| `AVG(col)` | Mean value | Average order value |
| `MAX(col)` | Maximum value | Highest purchase |
| `MIN(col)` | Minimum value | Earliest date |
| `STDEV(col)` | Standard deviation | Variability |
| `VARIANCE(col)` | Variance | Data spread |

### Examples for ML Feature Engineering

```sql
-- Customer-level aggregation for churn prediction
SELECT
    customer_id,
    COUNT(*) AS total_orders,
    SUM(amount) AS total_spent,
    AVG(amount) AS avg_order_value,
    MAX(amount) AS max_order,
    MIN(amount) AS min_order,
    STDEV(amount) AS order_variability,
    COUNT(DISTINCT product_id) AS unique_products,
    DATEDIFF(DAY, MIN(order_date), MAX(order_date)) AS customer_lifetime_days,
    DATEDIFF(DAY, MAX(order_date), GETDATE()) AS days_since_last_order
FROM orders
GROUP BY customer_id;

-- Conditional aggregation
SELECT
    customer_id,
    SUM(CASE WHEN status = 'returned' THEN 1 ELSE 0 END) AS return_count,
    AVG(CASE WHEN category = 'electronics' THEN amount END) AS avg_electronics_spend
FROM orders
GROUP BY customer_id
HAVING COUNT(*) >= 5;  -- filter after aggregation
```

> **Interview Tip:** `WHERE` filters **before** aggregation; `HAVING` filters **after** aggregation. `COUNT(*)` counts all rows including NULLs; `COUNT(col)` excludes NULLs.

---

## Question 8
- [ ] Done

**How can you extract time-based features from a SQL datetime field for use in a Machine Learning model?**

**Answer:**

Time-based features capture temporal patterns (seasonality, trends) critical for ML models.

### Common Time Features

```sql
SELECT
    order_id,
    order_date,
    
    -- Date components
    YEAR(order_date) AS order_year,
    MONTH(order_date) AS order_month,
    DAY(order_date) AS order_day,
    DATEPART(WEEKDAY, order_date) AS day_of_week,     -- 1=Sun, 7=Sat
    DATEPART(HOUR, order_date) AS order_hour,
    DATEPART(QUARTER, order_date) AS quarter,
    DATEPART(WEEK, order_date) AS week_of_year,
    DATEPART(DAYOFYEAR, order_date) AS day_of_year,
    
    -- Binary features
    CASE WHEN DATEPART(WEEKDAY, order_date) IN (1, 7) THEN 1 ELSE 0 END AS is_weekend,
    CASE WHEN MONTH(order_date) IN (11, 12) THEN 1 ELSE 0 END AS is_holiday_season,
    CASE WHEN DATEPART(HOUR, order_date) BETWEEN 9 AND 17 THEN 1 ELSE 0 END AS is_business_hours,
    
    -- Time differences
    DATEDIFF(DAY, order_date, GETDATE()) AS days_ago,
    DATEDIFF(DAY, LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date),
             order_date) AS days_between_orders
    
FROM orders;
```

### Cyclical Encoding (for models that need numeric features)

```sql
-- Encode hour/month as cyclical features using sin/cos
SELECT
    SIN(2 * PI() * DATEPART(HOUR, order_date) / 24.0) AS hour_sin,
    COS(2 * PI() * DATEPART(HOUR, order_date) / 24.0) AS hour_cos,
    SIN(2 * PI() * MONTH(order_date) / 12.0) AS month_sin,
    COS(2 * PI() * MONTH(order_date) / 12.0) AS month_cos
FROM orders;
```

> **Interview Tip:** Cyclical encoding (sin/cos) ensures that December (12) and January (1) are close in feature space. Without it, linear models treat them as far apart.

---

## Question 9
- [ ] Done

**How do you join transactional data to a dimension table in such a way that features for Machine Learning can be extracted?**

**Answer:**

### Star Schema Pattern

```
        dim_customer
             |
 fact_orders -+- dim_product
             |
        dim_store
```

### Feature Extraction via Joins

```sql
-- Join transaction facts with dimension tables for rich ML features
SELECT
    -- Customer features (from dimension)
    c.customer_id,
    c.age,
    c.gender,
    c.membership_tier,
    DATEDIFF(DAY, c.signup_date, GETDATE()) AS account_age_days,
    
    -- Product features (from dimension)
    p.category,
    p.brand,
    p.price_tier,
    
    -- Aggregated transaction features
    COUNT(*) AS total_purchases,
    SUM(o.amount) AS total_spent,
    AVG(o.amount) AS avg_order_value,
    COUNT(DISTINCT p.category) AS categories_bought,
    
    -- Store features (from dimension)
    s.region,
    s.store_size
    
FROM fact_orders o
INNER JOIN dim_customer c ON o.customer_id = c.customer_id
INNER JOIN dim_product p ON o.product_id = p.product_id
LEFT JOIN dim_store s ON o.store_id = s.store_id
GROUP BY c.customer_id, c.age, c.gender, c.membership_tier,
         c.signup_date, p.category, p.brand, p.price_tier,
         s.region, s.store_size;
```

### Join Types for ML

| Join | When to Use |
|------|------------|
| **INNER JOIN** | Only matching records (drop unmatched) |
| **LEFT JOIN** | Keep all primary records, NULL for missing |
| **CROSS JOIN** | Generate all feature combinations |

> **Interview Tip:** Use **LEFT JOIN** when missing dimension data is informative (e.g., customers without store visits). The NULL values become a feature themselves. Always check for **fan-out** (1-to-many joins inflating row count).

---

## Question 10
- [ ] Done

**How can you deal with outliers in a SQL database before passing data to Machine Learning algorithms?**

**Answer:**

### Outlier Detection Methods in SQL

```sql
-- Method 1: IQR (Interquartile Range)
WITH stats AS (
    SELECT
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) AS Q1,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) AS Q3
    FROM measurements
)
SELECT m.*,
    CASE WHEN m.value < s.Q1 - 1.5*(s.Q3-s.Q1)
              OR m.value > s.Q3 + 1.5*(s.Q3-s.Q1)
         THEN 1 ELSE 0 END AS is_outlier
FROM measurements m
CROSS JOIN stats s;

-- Method 2: Z-Score
WITH stats AS (
    SELECT AVG(value) AS mean_val, STDEV(value) AS std_val
    FROM measurements
)
SELECT m.*,
    ABS(m.value - s.mean_val) / NULLIF(s.std_val, 0) AS z_score
FROM measurements m
CROSS JOIN stats s
WHERE ABS(m.value - s.mean_val) / NULLIF(s.std_val, 0) <= 3;  -- keep |z| <= 3

-- Method 3: Percentile Capping (Winsorization)
WITH bounds AS (
    SELECT
        PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY value) AS lower,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY value) AS upper
    FROM measurements
)
SELECT
    CASE
        WHEN value < b.lower THEN b.lower
        WHEN value > b.upper THEN b.upper
        ELSE value
    END AS capped_value
FROM measurements
CROSS JOIN bounds b;
```

### Outlier Handling Strategies

| Strategy | SQL Approach | When |
|----------|-------------|------|
| **Remove** | `WHERE` filter with IQR/Z-score | Clearly erroneous data |
| **Cap** | Winsorize to percentile bounds | Extreme but valid values |
| **Flag** | Add `is_outlier` column as feature | Outlier status is informative |
| **Transform** | `LOG(value)`, `SQRT(value)` | Reduce skewness |
| **Separate model** | Filter to outlier/non-outlier groups | Different behavior patterns |

> **Interview Tip:** Always investigate outliers before removing them — they may be valid data points (e.g., high-value customers). Log transformation is often better than removal for skewed distributions.

---

## Question 11
- [ ] Done

**How can you execute a Machine Learning model stored in a database (such as a SQL Server with R or Python integration)?**

**Answer:**

### SQL Server — In-Database ML Execution

SQL Server supports running Python and R scripts directly within the database engine using `sp_execute_external_script`.

```sql
-- Execute Python model inside SQL Server
EXEC sp_execute_external_script
    @language = N'Python',
    @script = N'
import pandas as pd
import joblib

# Load pre-trained model
model = joblib.load("C:/models/churn_model.pkl")

# InputDataSet is automatically provided by SQL Server
predictions = model.predict(InputDataSet[["age", "tenure", "monthly_charges"]])
OutputDataSet = InputDataSet.copy()
OutputDataSet["prediction"] = predictions
',
    @input_data_1 = N'SELECT customer_id, age, tenure, monthly_charges FROM customers',
    @output_data_1_name = N'OutputDataSet'
WITH RESULT SETS ((
    customer_id INT, age INT, tenure INT,
    monthly_charges FLOAT, prediction INT
));
```

### Other Database ML Options

| Database | ML Capability |
|----------|---------------|
| **SQL Server** | `sp_execute_external_script` (Python/R) |
| **PostgreSQL** | MADlib extension, PL/Python |
| **BigQuery** | `CREATE MODEL` with BigQuery ML |
| **Amazon Redshift** | `CREATE MODEL` with SageMaker integration |
| **Oracle** | Oracle Machine Learning (OML4Py) |

### BigQuery ML Example

```sql
-- Train a model directly in SQL
CREATE OR REPLACE MODEL `project.dataset.churn_model`
OPTIONS(model_type='logistic_reg', input_label_cols=['churned'])
AS
SELECT age, tenure, monthly_charges, churned
FROM `project.dataset.customers`;

-- Predict using the model
SELECT * FROM ML.PREDICT(MODEL `project.dataset.churn_model`,
    (SELECT age, tenure, monthly_charges FROM `project.dataset.new_customers`));
```

> **Interview Tip:** In-database ML eliminates data movement overhead (no ETL to Python). Mention **BigQuery ML** and **SQL Server ML Services** as the most common production approaches.

---

## Question 12
- [ ] Done

**Can you update a Machine Learning model directly from SQL? If so, how might you do it?**

**Answer:**

### Yes — Several Approaches

| Approach | How | Database |
|----------|-----|----------|
| **BigQuery ML** | `CREATE OR REPLACE MODEL` | BigQuery |
| **SQL Server ML Services** | `sp_execute_external_script` with Python | SQL Server |
| **Stored Procedures** | Trigger Python/R retraining script | Any with UDF support |
| **Amazon Redshift ML** | `CREATE MODEL` with SageMaker | Redshift |

### BigQuery ML — Retrain Model

```sql
-- Retrain with latest data (replaces existing model)
CREATE OR REPLACE MODEL `project.dataset.fraud_model`
OPTIONS(
    model_type='BOOSTED_TREE_CLASSIFIER',
    input_label_cols=['is_fraud'],
    max_iterations=50,
    learn_rate=0.1
) AS
SELECT * FROM `project.dataset.transactions`
WHERE transaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY);
```

### SQL Server — Retrain via Stored Procedure

```sql
CREATE PROCEDURE retrain_model
AS
EXEC sp_execute_external_script
    @language = N'Python',
    @script = N'
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

model = RandomForestClassifier(n_estimators=200)
model.fit(InputDataSet.drop("target", axis=1), InputDataSet["target"])
joblib.dump(model, "C:/models/model_latest.pkl")
',
    @input_data_1 = N'SELECT * FROM training_data';
GO

-- Schedule periodic retraining
EXEC msdb.dbo.sp_add_job @job_name = 'Retrain_ML_Model';
```

### Automated Retraining Pipeline

```
New Data → SQL Trigger/Schedule → Retrain Model → Evaluate → Deploy if better
```

> **Interview Tip:** Always **evaluate** the new model before replacing the old one. Implement A/B testing or shadow deployment to ensure the retrained model performs better.

---

## Question 13
- [ ] Done

**What strategies can be used to efficiently update a large SQL-based Machine Learning model?**

**Answer:**

### Strategies for Efficient Model Updates

| Strategy | Description | When to Use |
|----------|-------------|------------|
| **Incremental training** | Train on new data only, update weights | Online learning models (SGD) |
| **Sliding window** | Retrain on most recent N days of data | Time-sensitive models |
| **Feature store caching** | Pre-compute features, cache in materialized views | Expensive feature queries |
| **Partitioned retraining** | Update only affected model partitions | Regional/segment-specific models |
| **Warm start** | Initialize with previous model weights | Reduce training time |
| **Change data capture (CDC)** | Only process changed records | Large slowly-changing datasets |

### SQL Implementation

```sql
-- 1. Sliding Window — Keep only recent training data
CREATE VIEW v_training_data AS
SELECT * FROM transactions
WHERE transaction_date >= DATEADD(DAY, -90, GETDATE());

-- 2. Materialized View for Feature Caching
CREATE MATERIALIZED VIEW mv_customer_features AS
SELECT
    customer_id,
    COUNT(*) AS total_orders,
    AVG(amount) AS avg_amount,
    MAX(order_date) AS last_order_date
FROM orders
GROUP BY customer_id;
-- Refresh: REFRESH MATERIALIZED VIEW mv_customer_features;

-- 3. CDC — Track changed records for incremental update
SELECT * FROM CHANGETABLE(CHANGES orders, @last_sync_version) AS ct
INNER JOIN orders o ON o.order_id = ct.order_id;

-- 4. Partitioned approach — retrain per segment
SELECT region, *
FROM training_data
WHERE region = @target_region;  -- only retrain one region's model
```

> **Interview Tip:** Incremental learning avoids reprocessing the entire dataset. Use **materialized views** to cache expensive feature computations and **CDC** for tracking data changes since last training.

---

## Question 14
- [ ] Done

**How do you ensure the consistency and reliability of SQL data used for Machine Learning?**

**Answer:**

### Data Quality Dimensions

| Dimension | Check | SQL Implementation |
|-----------|-------|--------------------|
| **Completeness** | No missing critical values | `WHERE col IS NOT NULL` |
| **Uniqueness** | No duplicate records | `GROUP BY ... HAVING COUNT(*) > 1` |
| **Validity** | Values within expected ranges | `CHECK` constraints |
| **Consistency** | Cross-table agreement | Foreign key constraints, joins |
| **Timeliness** | Data freshness | Timestamp checks |
| **Accuracy** | Correct values | Referential integrity |

### SQL Data Quality Checks

```sql
-- 1. Completeness — check for NULLs
SELECT
    COUNT(*) AS total_rows,
    SUM(CASE WHEN age IS NULL THEN 1 ELSE 0 END) AS null_age,
    SUM(CASE WHEN income IS NULL THEN 1 ELSE 0 END) AS null_income,
    CAST(SUM(CASE WHEN age IS NULL THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100 AS pct_null_age
FROM customers;

-- 2. Uniqueness — find duplicates
SELECT customer_id, COUNT(*)
FROM customers
GROUP BY customer_id
HAVING COUNT(*) > 1;

-- 3. Validity — range checks
SELECT * FROM customers
WHERE age < 0 OR age > 150
   OR income < 0
   OR signup_date > GETDATE();

-- 4. Consistency — referential integrity
SELECT o.* FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE c.customer_id IS NULL;  -- orphaned orders

-- 5. Freshness — data staleness
SELECT
    MAX(updated_at) AS last_update,
    DATEDIFF(HOUR, MAX(updated_at), GETDATE()) AS hours_since_update
FROM data_table;

-- 6. Schema enforcement
ALTER TABLE customers ADD CONSTRAINT chk_age CHECK (age BETWEEN 0 AND 150);
ALTER TABLE customers ADD CONSTRAINT chk_email CHECK (email LIKE '%@%.%');
```

> **Interview Tip:** Implement data quality checks as **automated tests** that run before model training. Use **Great Expectations** (Python) or SQL assertions to fail the pipeline early if data quality drops below thresholds.

---

## Question 15
- [ ] Done

**What SQL features are there for report generation that might be useful for analyzing Machine Learning model performance?**

**Answer:**

### SQL Reporting Features for ML Analysis

```sql
-- 1. PIVOT — Create confusion matrix
SELECT *
FROM (
    SELECT actual_label, predicted_label, COUNT(*) AS cnt
    FROM predictions
    GROUP BY actual_label, predicted_label
) src
PIVOT (
    SUM(cnt) FOR predicted_label IN ([0], [1])
) AS pivot_table;

-- 2. Window Functions — Performance over time
SELECT
    prediction_date,
    COUNT(*) AS total_predictions,
    AVG(CASE WHEN actual = predicted THEN 1.0 ELSE 0.0 END) AS daily_accuracy,
    AVG(AVG(CASE WHEN actual = predicted THEN 1.0 ELSE 0.0 END))
        OVER (ORDER BY prediction_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
        AS rolling_7day_accuracy
FROM predictions
GROUP BY prediction_date;

-- 3. GROUPING SETS — Multi-level aggregation
SELECT
    COALESCE(model_version, 'ALL') AS model,
    COALESCE(segment, 'ALL') AS segment,
    COUNT(*) AS predictions,
    AVG(CASE WHEN correct = 1 THEN 1.0 ELSE 0.0 END) AS accuracy
FROM prediction_log
GROUP BY GROUPING SETS (
    (model_version, segment),
    (model_version),
    ()
);

-- 4. NTILE — Decile analysis for probability calibration
SELECT
    decile,
    AVG(predicted_probability) AS avg_predicted,
    AVG(CAST(actual_label AS FLOAT)) AS avg_actual,
    COUNT(*) AS count
FROM (
    SELECT *, NTILE(10) OVER (ORDER BY predicted_probability) AS decile
    FROM predictions
) t
GROUP BY decile
ORDER BY decile;
```

### Key SQL Features for ML Reporting

| Feature | Use Case |
|---------|----------|
| **PIVOT/UNPIVOT** | Confusion matrices, cross-tabulations |
| **Window functions** | Rolling accuracy, cumulative metrics |
| **GROUPING SETS** | Report by model version, segment, overall |
| **NTILE** | Probability calibration, decile analysis |
| **CTE** | Step-by-step metric computation |
| **CASE WHEN** | Custom metrics (precision, recall) |

> **Interview Tip:** Use **decile analysis** (NTILE) to check model calibration — predicted probabilities should match actual event rates within each decile.

---

## Question 16
- [ ] Done

**How can you use SQL to visualize the distribution of data points before feeding them into an ML algorithm?**

**Answer:**

While SQL doesn’t create visual charts directly, it can compute the data needed for distribution analysis.

### Distribution Analysis in SQL

```sql
-- 1. Histogram — Binned frequency distribution
SELECT
    FLOOR(value / 10) * 10 AS bin_start,
    FLOOR(value / 10) * 10 + 10 AS bin_end,
    COUNT(*) AS frequency,
    REPLICATE('*', COUNT(*) / 10) AS visual_bar  -- ASCII histogram
FROM measurements
GROUP BY FLOOR(value / 10)
ORDER BY bin_start;

-- 2. Five-number summary (Box plot data)
SELECT
    MIN(value) AS minimum,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) AS Q1,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY value) AS median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) AS Q3,
    MAX(value) AS maximum,
    AVG(value) AS mean,
    STDEV(value) AS std_dev
FROM measurements;

-- 3. Value distribution for categorical features
SELECT
    category,
    COUNT(*) AS count,
    CAST(COUNT(*) AS FLOAT) / SUM(COUNT(*)) OVER () * 100 AS percentage
FROM products
GROUP BY category
ORDER BY count DESC;

-- 4. Class balance check (critical for ML)
SELECT
    target_label,
    COUNT(*) AS count,
    ROUND(CAST(COUNT(*) AS FLOAT) / SUM(COUNT(*)) OVER () * 100, 2) AS pct
FROM training_data
GROUP BY target_label;
```

### Using SQL Results with Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# Query distribution data from SQL
df = pd.read_sql("SELECT value FROM measurements", connection)
df['value'].hist(bins=30)
plt.title('Distribution of Values')
plt.show()
```

> **Interview Tip:** SQL is for computing distributions; Python/BI tools (Tableau, Power BI) are for visualizing them. Always check **class balance** and **feature distributions** before training.

---

## Question 17
- [ ] Done

**Can SQL be used to visualize false positives and false negatives in classification models? If so, how?**

**Answer:**

### Confusion Matrix Analysis in SQL

```sql
-- 1. Compute confusion matrix components
SELECT
    SUM(CASE WHEN actual=1 AND predicted=1 THEN 1 ELSE 0 END) AS true_positives,
    SUM(CASE WHEN actual=0 AND predicted=1 THEN 1 ELSE 0 END) AS false_positives,
    SUM(CASE WHEN actual=1 AND predicted=0 THEN 1 ELSE 0 END) AS false_negatives,
    SUM(CASE WHEN actual=0 AND predicted=0 THEN 1 ELSE 0 END) AS true_negatives
FROM prediction_results;

-- 2. Calculate precision, recall, F1
WITH cm AS (
    SELECT
        SUM(CASE WHEN actual=1 AND predicted=1 THEN 1 ELSE 0 END) AS tp,
        SUM(CASE WHEN actual=0 AND predicted=1 THEN 1 ELSE 0 END) AS fp,
        SUM(CASE WHEN actual=1 AND predicted=0 THEN 1 ELSE 0 END) AS fn,
        SUM(CASE WHEN actual=0 AND predicted=0 THEN 1 ELSE 0 END) AS tn
    FROM prediction_results
)
SELECT
    ROUND(CAST(tp AS FLOAT) / NULLIF(tp + fp, 0), 4) AS precision,
    ROUND(CAST(tp AS FLOAT) / NULLIF(tp + fn, 0), 4) AS recall,
    ROUND(CAST(tp + tn AS FLOAT) / NULLIF(tp + fp + fn + tn, 0), 4) AS accuracy
FROM cm;

-- 3. Examine false positives (what did we incorrectly predict as positive?)
SELECT *
FROM prediction_results
WHERE actual = 0 AND predicted = 1
ORDER BY confidence_score DESC;  -- most confident false positives

-- 4. Examine false negatives (what did we miss?)
SELECT *
FROM prediction_results
WHERE actual = 1 AND predicted = 0
ORDER BY confidence_score ASC;  -- least confident misses

-- 5. Error analysis by segment
SELECT
    segment,
    SUM(CASE WHEN actual != predicted THEN 1 ELSE 0 END) AS errors,
    COUNT(*) AS total,
    ROUND(SUM(CASE WHEN actual != predicted THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS error_rate_pct
FROM prediction_results
GROUP BY segment
ORDER BY error_rate_pct DESC;
```

> **Interview Tip:** Analyzing **false positives and false negatives by segment** reveals where the model struggles. This drives targeted improvements — more training data for weak segments or segment-specific models.

---

## Question 18
- [ ] Done

**What strategies might you use to automate the retraining and evaluation of Machine Learning models from within SQL?**

**Answer:**

### Automation Strategies

| Strategy | Tool | Description |
|----------|------|-------------|
| **Scheduled stored procedures** | SQL Agent / cron | Run retraining on schedule |
| **Trigger-based** | SQL Triggers | Retrain when data threshold met |
| **Event-driven** | Change Data Capture | Retrain when data changes |
| **Pipeline orchestration** | Airflow / ADF | End-to-end ML pipeline |
| **In-database ML** | BigQuery ML | `CREATE OR REPLACE MODEL` on schedule |

### SQL Server Automated Pipeline

```sql
-- Step 1: Create retraining procedure
CREATE PROCEDURE sp_retrain_and_evaluate
AS
BEGIN
    -- Check if enough new data exists
    DECLARE @new_records INT;
    SELECT @new_records = COUNT(*)
    FROM training_data
    WHERE created_at >= DATEADD(DAY, -7, GETDATE());
    
    IF @new_records < 100
    BEGIN
        PRINT 'Not enough new data for retraining';
        RETURN;
    END
    
    -- Retrain model
    EXEC sp_execute_external_script
        @language = N'Python',
        @script = N'
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

X = InputDataSet.drop("target", axis=1)
y = InputDataSet["target"]

new_model = RandomForestClassifier(n_estimators=200)
scores = cross_val_score(new_model, X, y, cv=5)
new_score = np.mean(scores)

# Load current model score
try:
    old_model = joblib.load("C:/models/current_model.pkl")
    old_score = np.mean(cross_val_score(old_model, X, y, cv=5))
except:
    old_score = 0

# Only deploy if better
if new_score > old_score:
    new_model.fit(X, y)
    joblib.dump(new_model, "C:/models/current_model.pkl")
    print(f"Model updated: {old_score:.4f} -> {new_score:.4f}")
else:
    print(f"Keeping current model ({old_score:.4f} >= {new_score:.4f})")
',
        @input_data_1 = N'SELECT * FROM training_data';
    
    -- Log retraining event
    INSERT INTO model_log (retrain_date, status)
    VALUES (GETDATE(), 'completed');
END;

-- Step 2: Schedule weekly execution
EXEC msdb.dbo.sp_add_schedule @schedule_name = 'Weekly_Retrain',
    @freq_type = 8, @freq_interval = 2;  -- Every Monday
```

### Evaluation Logging Table

```sql
CREATE TABLE model_log (
    log_id INT IDENTITY PRIMARY KEY,
    retrain_date DATETIME,
    model_version VARCHAR(50),
    accuracy FLOAT,
    f1_score FLOAT,
    data_rows INT,
    status VARCHAR(20)
);
```

> **Interview Tip:** Always implement a **champion-challenger** pattern — only deploy the new model if it outperforms the current one. Log all retraining events for audit trails.
