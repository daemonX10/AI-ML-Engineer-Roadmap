# SQL ML Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Write a SQL query that joins two tables and retrieves only the rows with matching keys**

### Code
```sql
-- Basic INNER JOIN
SELECT 
    c.customer_id,
    c.name,
    o.order_id,
    o.order_date,
    o.amount
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id;

-- Multiple JOINs with aggregation
SELECT 
    c.customer_id,
    c.name,
    COUNT(o.order_id) AS total_orders,
    SUM(o.amount) AS total_spent
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name
HAVING COUNT(o.order_id) > 1;
```

---

## Question 2
- [ ] Done

**Write a SQL query to pivot rows into columns.**

### Code
```sql
-- Manual PIVOT using CASE (works in all databases)
SELECT 
    customer_id,
    SUM(CASE WHEN category = 'Electronics' THEN amount ELSE 0 END) AS electronics,
    SUM(CASE WHEN category = 'Clothing' THEN amount ELSE 0 END) AS clothing,
    SUM(CASE WHEN category = 'Books' THEN amount ELSE 0 END) AS books,
    SUM(amount) AS total
FROM sales
GROUP BY customer_id;

-- Time series pivot (monthly sales)
SELECT 
    product_id,
    SUM(CASE WHEN MONTH(sale_date) = 1 THEN amount ELSE 0 END) AS jan_sales,
    SUM(CASE WHEN MONTH(sale_date) = 2 THEN amount ELSE 0 END) AS feb_sales,
    SUM(CASE WHEN MONTH(sale_date) = 3 THEN amount ELSE 0 END) AS mar_sales
FROM sales
WHERE YEAR(sale_date) = 2024
GROUP BY product_id;
```

---

## Question 3
- [ ] Done

**Write a SQL query to find duplicate records.**

### Code
```sql
-- Find duplicates
SELECT 
    email,
    COUNT(*) AS count
FROM users
GROUP BY email
HAVING COUNT(*) > 1;

-- See all duplicate rows
WITH Duplicates AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY email ORDER BY created_at) AS rn
    FROM users
)
SELECT * FROM Duplicates WHERE rn > 1;

-- Delete duplicates (keep first)
DELETE FROM users
WHERE id NOT IN (
    SELECT MIN(id) 
    FROM users 
    GROUP BY email
);
```

---

## Question 4
- [ ] Done

**Write a SQL query to calculate cumulative sum.**

### Code
```sql
SELECT 
    customer_id,
    transaction_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY customer_id 
        ORDER BY transaction_date
        ROWS UNBOUNDED PRECEDING
    ) AS cumulative_amount
FROM transactions;
```

---

## Question 5
- [ ] Done

**Write a SQL query to find the second highest salary.**

### Code
```sql
-- Method 1: LIMIT OFFSET
SELECT salary FROM employees
ORDER BY salary DESC
LIMIT 1 OFFSET 1;

-- Method 2: Subquery
SELECT MAX(salary) AS second_highest
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

-- Method 3: DENSE_RANK (handles ties)
SELECT salary FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rank
    FROM employees
) ranked
WHERE rank = 2;
```

---

## Question 6
- [ ] Done

**Write a SQL query to calculate year-over-year growth.**

### Code
```sql
WITH YearlySales AS (
    SELECT 
        YEAR(sale_date) AS year,
        SUM(amount) AS total_sales
    FROM sales
    GROUP BY YEAR(sale_date)
)
SELECT 
    year,
    total_sales,
    LAG(total_sales) OVER (ORDER BY year) AS prev_year_sales,
    ROUND(
        (total_sales - LAG(total_sales) OVER (ORDER BY year)) * 100.0 
        / LAG(total_sales) OVER (ORDER BY year), 
        2
    ) AS yoy_growth_pct
FROM YearlySales;
```

---

## Question 7
- [ ] Done

**Write a SQL query to create ML features from transaction data.**

### Code
```sql
-- Comprehensive feature engineering
SELECT 
    customer_id,
    
    -- Recency
    DATEDIFF(CURRENT_DATE, MAX(transaction_date)) AS days_since_last_purchase,
    
    -- Frequency
    COUNT(*) AS total_transactions,
    COUNT(DISTINCT DATE(transaction_date)) AS unique_purchase_days,
    
    -- Monetary
    SUM(amount) AS total_spent,
    AVG(amount) AS avg_transaction,
    MAX(amount) AS max_transaction,
    MIN(amount) AS min_transaction,
    STDDEV(amount) AS transaction_stddev,
    
    -- Time patterns
    COUNT(CASE WHEN DAYOFWEEK(transaction_date) IN (1,7) THEN 1 END) AS weekend_purchases,
    COUNT(CASE WHEN HOUR(transaction_date) < 12 THEN 1 END) AS morning_purchases,
    
    -- Category diversity
    COUNT(DISTINCT product_category) AS categories_bought,
    
    -- Tenure
    DATEDIFF(MAX(transaction_date), MIN(transaction_date)) AS customer_tenure_days

FROM transactions
WHERE transaction_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR)
GROUP BY customer_id;
```

---

## Question 8
- [ ] Done

**Write a SQL query to detect outliers using IQR method.**

### Code
```sql
WITH Stats AS (
    SELECT 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount) AS q1,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) AS q3
    FROM transactions
),
Bounds AS (
    SELECT 
        q1,
        q3,
        q3 - q1 AS iqr,
        q1 - 1.5 * (q3 - q1) AS lower_bound,
        q3 + 1.5 * (q3 - q1) AS upper_bound
    FROM Stats
)
SELECT t.*
FROM transactions t, Bounds b
WHERE t.amount < b.lower_bound OR t.amount > b.upper_bound;
```

---

## Question 9
- [ ] Done

**Write a SQL query to create a train/test split.**

### Code
```sql
-- Random 80/20 split
SELECT 
    *,
    CASE 
        WHEN RAND() < 0.8 THEN 'train'
        ELSE 'test'
    END AS dataset
FROM features;

-- Reproducible split using modulo
SELECT 
    *,
    CASE 
        WHEN MOD(customer_id, 5) = 0 THEN 'test'  -- 20%
        ELSE 'train'                               -- 80%
    END AS dataset
FROM features;

-- Stratified split
WITH Ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY target_class ORDER BY RAND()) AS rn,
        COUNT(*) OVER (PARTITION BY target_class) AS class_count
    FROM features
)
SELECT *,
    CASE 
        WHEN rn <= class_count * 0.8 THEN 'train'
        ELSE 'test'
    END AS dataset
FROM Ranked;
```

---

## Question 10
- [ ] Done

**Write a SQL query to one-hot encode a categorical column.**

### Code
```sql
-- One-hot encoding for category column
SELECT 
    id,
    CASE WHEN category = 'A' THEN 1 ELSE 0 END AS category_A,
    CASE WHEN category = 'B' THEN 1 ELSE 0 END AS category_B,
    CASE WHEN category = 'C' THEN 1 ELSE 0 END AS category_C,
    CASE WHEN category NOT IN ('A', 'B', 'C') THEN 1 ELSE 0 END AS category_other
FROM products;
```


---

## Question 11
- [ ] Done

**Write a SQL query to calculate moving averages.**

**Answer:**

Moving averages smooth out short-term fluctuations and highlight long-term trends — essential for time-series features.

```sql
-- Simple Moving Average (SMA) — Last 7 days
SELECT
    order_date,
    daily_sales,
    AVG(daily_sales) OVER (
        ORDER BY order_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS sma_7day,
    
    -- 30-day moving average
    AVG(daily_sales) OVER (
        ORDER BY order_date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS sma_30day
FROM daily_sales_summary
ORDER BY order_date;

-- Moving Average per Customer
SELECT
    customer_id,
    order_date,
    amount,
    AVG(amount) OVER (
        PARTITION BY customer_id
        ORDER BY order_date
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS customer_ma_5
FROM orders;

-- Exponential Moving Average (approximation)
SELECT
    order_date,
    daily_sales,
    daily_sales * 0.3 + LAG(daily_sales, 1) OVER (ORDER BY order_date) * 0.7 AS ema_approx
FROM daily_sales_summary;
```

### Window Frame Options

| Frame | Meaning |
|-------|---------|
| `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` | Fixed 7-row window |
| `RANGE BETWEEN INTERVAL '7' DAY PRECEDING AND CURRENT ROW` | Calendar-based |
| `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` | Cumulative average |

> **Interview Tip:** `ROWS` counts fixed number of rows; `RANGE` is value-based (handles gaps in dates). For time-series ML features, always specify the window frame explicitly.

---

## Question 12
- [ ] Done

**How can you create lagged features in SQL?**

**Answer:**

Lagged features capture previous values of a variable — critical for time-series and sequence models.

```sql
-- LAG and LEAD functions
SELECT
    customer_id,
    order_date,
    amount,
    
    -- Previous values (lagged)
    LAG(amount, 1) OVER (PARTITION BY customer_id ORDER BY order_date) AS prev_1_amount,
    LAG(amount, 2) OVER (PARTITION BY customer_id ORDER BY order_date) AS prev_2_amount,
    LAG(amount, 7) OVER (PARTITION BY customer_id ORDER BY order_date) AS prev_7_amount,
    
    -- Future values (lead) — for target creation
    LEAD(amount, 1) OVER (PARTITION BY customer_id ORDER BY order_date) AS next_amount,
    
    -- Time-based lags
    DATEDIFF(DAY,
        LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date),
        order_date
    ) AS days_since_last_order,
    
    -- Change from previous
    amount - LAG(amount, 1) OVER (PARTITION BY customer_id ORDER BY order_date) AS amount_change,
    
    -- Percentage change
    (amount - LAG(amount, 1) OVER (PARTITION BY customer_id ORDER BY order_date))
    / NULLIF(LAG(amount, 1) OVER (PARTITION BY customer_id ORDER BY order_date), 0) * 100
    AS pct_change
    
FROM orders
ORDER BY customer_id, order_date;
```

### Multiple Lags for ML Features

```sql
-- Create feature table with multiple lags
SELECT
    date,
    value AS value_t,
    LAG(value, 1) OVER (ORDER BY date) AS value_t1,
    LAG(value, 2) OVER (ORDER BY date) AS value_t2,
    LAG(value, 3) OVER (ORDER BY date) AS value_t3,
    LAG(value, 7) OVER (ORDER BY date) AS value_t7,
    LEAD(value, 1) OVER (ORDER BY date) AS target  -- next value to predict
FROM time_series_data;
```

> **Interview Tip:** LAG looks backward (previous rows), LEAD looks forward (future rows). For ML, use LAG for features and LEAD for target variable creation. Always handle NULLs in the first rows with `COALESCE`.

---

## Question 13
- [ ] Done

**Describe how to compute a ratio feature within groups using SQL.**

**Answer:**

Ratio features normalize values relative to their group, capturing proportional relationships.

```sql
-- 1. Ratio within group using window functions
SELECT
    employee_id,
    department,
    salary,
    
    -- Salary as ratio of department average
    salary / AVG(salary) OVER (PARTITION BY department) AS salary_to_dept_avg,
    
    -- Percentage of department total
    salary / SUM(salary) OVER (PARTITION BY department) * 100 AS pct_of_dept_total,
    
    -- Rank percentile within department
    PERCENT_RANK() OVER (PARTITION BY department ORDER BY salary) AS salary_percentile,
    
    -- Ratio to maximum in group
    salary / MAX(salary) OVER (PARTITION BY department) AS ratio_to_max
    
FROM employees;

-- 2. Product category ratio for customer profiling
SELECT
    customer_id,
    category,
    SUM(amount) AS category_spend,
    SUM(amount) / SUM(SUM(amount)) OVER (PARTITION BY customer_id) AS spend_ratio
FROM orders
GROUP BY customer_id, category;

-- 3. Conversion ratio by channel
SELECT
    channel,
    COUNT(*) AS total_visits,
    SUM(CASE WHEN converted = 1 THEN 1 ELSE 0 END) AS conversions,
    CAST(SUM(CASE WHEN converted = 1 THEN 1 ELSE 0 END) AS FLOAT)
        / NULLIF(COUNT(*), 0) AS conversion_rate
FROM website_visits
GROUP BY channel;
```

> **Interview Tip:** Ratio features are powerful because they capture **relative importance** rather than absolute values, making them more robust across different scales. Always use `NULLIF(denominator, 0)` to prevent division by zero.

---

## Question 14
- [ ] Done

**In SQL, how would you format strings or concatenate columns for text-based Machine Learning features?**

**Answer:**

```sql
-- 1. Basic concatenation
SELECT
    -- CONCAT function (handles NULLs)
    CONCAT(first_name, ' ', last_name) AS full_name,
    
    -- || operator (ANSI SQL)
    first_name || ' ' || last_name AS full_name_alt,
    
    -- CONCAT_WS (with separator)
    CONCAT_WS(', ', city, state, country) AS location
FROM customers;

-- 2. Text cleaning for NLP features
SELECT
    -- Lowercase normalization
    LOWER(description) AS clean_text,
    
    -- Trim whitespace
    TRIM(BOTH FROM description) AS trimmed,
    
    -- Remove special characters (PostgreSQL)
    REGEXP_REPLACE(description, '[^a-zA-Z0-9\s]', '', 'g') AS alpha_only,
    
    -- Extract patterns
    SUBSTRING(email FROM '@(.+)$') AS email_domain,
    
    -- String length as feature
    LENGTH(description) AS text_length,
    LENGTH(description) - LENGTH(REPLACE(description, ' ', '')) + 1 AS word_count
FROM products;

-- 3. Create combined text feature for TF-IDF
SELECT
    product_id,
    CONCAT(
        COALESCE(title, ''), ' ',
        COALESCE(description, ''), ' ',
        COALESCE(category, ''), ' ',
        COALESCE(brand, '')
    ) AS combined_text
FROM products;

-- 4. String aggregation (combine multiple rows)
-- PostgreSQL
SELECT
    customer_id,
    STRING_AGG(product_name, ', ' ORDER BY order_date) AS purchased_products
FROM orders
GROUP BY customer_id;

-- SQL Server
SELECT
    customer_id,
    STRING_AGG(product_name, ', ') WITHIN GROUP (ORDER BY order_date) AS purchased_products
FROM orders
GROUP BY customer_id;
```

> **Interview Tip:** Use `CONCAT_WS` instead of `CONCAT` with manual separators — it’s cleaner and handles NULLs better. `STRING_AGG` is essential for creating document-level text from row-level data.

---

## Question 15
- [ ] Done

**Write a SQL stored procedure that calls a Machine Learning scoring function.**

**Answer:**

```sql
-- SQL Server stored procedure for ML scoring
CREATE PROCEDURE sp_score_customers
    @model_path NVARCHAR(500) = 'C:/models/churn_model.pkl'
AS
BEGIN
    SET NOCOUNT ON;
    
    -- Create temp table for results
    CREATE TABLE #predictions (
        customer_id INT,
        churn_probability FLOAT,
        churn_prediction INT,
        scored_at DATETIME DEFAULT GETDATE()
    );
    
    -- Call Python ML model
    INSERT INTO #predictions (customer_id, churn_probability, churn_prediction)
    EXEC sp_execute_external_script
        @language = N'Python',
        @script = N'
import pandas as pd
import joblib
import numpy as np

model = joblib.load(model_path)

features = InputDataSet[["age", "tenure", "monthly_charges", "total_charges"]]
probabilities = model.predict_proba(features)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

OutputDataSet = pd.DataFrame({
    "customer_id": InputDataSet["customer_id"],
    "churn_probability": probabilities,
    "churn_prediction": predictions
})
',
        @input_data_1 = N'SELECT customer_id, age, tenure, monthly_charges, total_charges FROM customers',
        @output_data_1_name = N'OutputDataSet',
        @params = N'@model_path NVARCHAR(500)',
        @model_path = @model_path
    WITH RESULT SETS ((
        customer_id INT,
        churn_probability FLOAT,
        churn_prediction INT
    ));
    
    -- Save predictions to permanent table
    INSERT INTO prediction_history (customer_id, churn_probability, churn_prediction, scored_at)
    SELECT customer_id, churn_probability, churn_prediction, GETDATE()
    FROM #predictions;
    
    -- Return high-risk customers
    SELECT * FROM #predictions
    WHERE churn_probability > 0.7
    ORDER BY churn_probability DESC;
    
    DROP TABLE #predictions;
END;
GO

-- Execute
EXEC sp_score_customers;
```

> **Interview Tip:** Stored procedures encapsulate the scoring logic, making it callable by applications, scheduled jobs, or other procedures. Always log predictions to a history table for monitoring model performance over time.

---

## Question 16
- [ ] Done

**How would you construct a complex SQL query to extract time series features for a Machine Learning model?**

**Answer:**

```sql
-- Comprehensive time-series feature extraction
WITH daily_stats AS (
    SELECT
        customer_id,
        CAST(order_date AS DATE) AS order_day,
        SUM(amount) AS daily_total,
        COUNT(*) AS daily_orders
    FROM orders
    GROUP BY customer_id, CAST(order_date AS DATE)
),
lagged_features AS (
    SELECT
        *,
        -- Lag features
        LAG(daily_total, 1) OVER (PARTITION BY customer_id ORDER BY order_day) AS lag_1,
        LAG(daily_total, 7) OVER (PARTITION BY customer_id ORDER BY order_day) AS lag_7,
        LAG(daily_total, 30) OVER (PARTITION BY customer_id ORDER BY order_day) AS lag_30,
        
        -- Rolling statistics
        AVG(daily_total) OVER (PARTITION BY customer_id ORDER BY order_day
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_avg_7,
        STDEV(daily_total) OVER (PARTITION BY customer_id ORDER BY order_day
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_std_7,
        MAX(daily_total) OVER (PARTITION BY customer_id ORDER BY order_day
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS rolling_max_30,
        
        -- Cumulative features
        SUM(daily_total) OVER (PARTITION BY customer_id ORDER BY order_day
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_total,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_day) AS order_sequence,
        
        -- Trend (difference from moving average)
        daily_total - AVG(daily_total) OVER (PARTITION BY customer_id ORDER BY order_day
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS deviation_from_ma,
        
        -- Seasonality features
        DATEPART(WEEKDAY, order_day) AS day_of_week,
        DATEPART(MONTH, order_day) AS month,
        
        -- Days between events
        DATEDIFF(DAY,
            LAG(order_day) OVER (PARTITION BY customer_id ORDER BY order_day),
            order_day) AS inter_order_days
    FROM daily_stats
)
SELECT
    *,
    -- Target: next day's total (for prediction)
    LEAD(daily_total, 1) OVER (PARTITION BY customer_id ORDER BY order_day) AS target
FROM lagged_features
WHERE lag_30 IS NOT NULL;  -- ensure enough history
```

> **Interview Tip:** Always create features from the **past** (LAG) and the target from the **future** (LEAD) to prevent data leakage. Use CTEs to build features layer by layer for readability.

---

## Question 17
- [ ] Done

**Discuss ways to implement regular expressions in SQL for natural language processing tasks.**

**Answer:**

Regex support varies by database but is essential for text pattern extraction in NLP feature engineering.

### PostgreSQL (Best Regex Support)

```sql
-- Pattern matching
SELECT * FROM documents WHERE content ~ '\bmachine\s+learning\b';

-- Case-insensitive match
SELECT * FROM documents WHERE content ~* 'neural\s+network';

-- Extract patterns
SELECT
    REGEXP_MATCHES(text, '\b[A-Z][a-z]+\b', 'g') AS capitalized_words,
    (REGEXP_MATCHES(email, '@([\w.]+)'))[1] AS email_domain,
    REGEXP_REPLACE(text, '\s+', ' ', 'g') AS normalized_whitespace,
    REGEXP_REPLACE(text, '[^a-zA-Z\s]', '', 'g') AS letters_only;

-- Extract all email addresses
SELECT REGEXP_MATCHES(content, '[\w.]+@[\w.]+\.[a-z]{2,}', 'g') AS emails
FROM documents;

-- Count pattern occurrences (word frequency)
SELECT
    doc_id,
    (LENGTH(content) - LENGTH(REGEXP_REPLACE(content, 'machine learning', '', 'gi')))
    / LENGTH('machine learning') AS ml_mentions
FROM documents;
```

### MySQL

```sql
-- MySQL 8.0+ REGEXP_LIKE
SELECT * FROM documents WHERE REGEXP_LIKE(content, 'data\\s+science', 'i');
SELECT REGEXP_SUBSTR(text, '[0-9]+') AS first_number FROM data;
SELECT REGEXP_REPLACE(text, '[[:punct:]]', '') AS no_punctuation FROM data;
```

### SQL Server

```sql
-- SQL Server uses LIKE with limited patterns (no full regex)
SELECT * FROM documents WHERE content LIKE '%machine%learning%';
-- For full regex, use CLR functions or Python integration
```

### NLP Feature Extraction

```sql
-- Tokenization approximation
SELECT
    doc_id,
    LENGTH(content) AS char_count,
    LENGTH(content) - LENGTH(REPLACE(content, ' ', '')) + 1 AS approx_word_count,
    LENGTH(content) - LENGTH(REPLACE(content, '.', '')) AS sentence_count,
    CASE WHEN content ~* '\b(excellent|great|amazing)\b' THEN 'positive'
         WHEN content ~* '\b(terrible|awful|bad)\b' THEN 'negative'
         ELSE 'neutral' END AS simple_sentiment
FROM documents;
```

> **Interview Tip:** PostgreSQL has the best regex support. For complex NLP, extract text via SQL then process in Python. SQL regex is best for **simple pattern extraction** (emails, URLs, hashtags) and **text cleaning**.

---

## Question 18
- [ ] Done

**Write a SQL script to identify and replace missing values with the column mean.**

**Answer:**

```sql
-- Method 1: UPDATE with subquery (modifies data in place)
UPDATE measurements
SET value = (
    SELECT AVG(value) FROM measurements WHERE value IS NOT NULL
)
WHERE value IS NULL;

-- Method 2: SELECT with COALESCE (non-destructive)
SELECT
    id,
    COALESCE(value, (SELECT AVG(value) FROM measurements)) AS imputed_value,
    CASE WHEN value IS NULL THEN 1 ELSE 0 END AS was_imputed
FROM measurements;

-- Method 3: Window function — Fill with group mean
SELECT
    id,
    category,
    COALESCE(value, AVG(value) OVER (PARTITION BY category)) AS imputed_by_group
FROM measurements;

-- Method 4: CTE approach (multiple columns)
WITH col_means AS (
    SELECT
        AVG(age) AS mean_age,
        AVG(income) AS mean_income,
        AVG(score) AS mean_score
    FROM customers
)
SELECT
    customer_id,
    COALESCE(age, cm.mean_age) AS age,
    COALESCE(income, cm.mean_income) AS income,
    COALESCE(score, cm.mean_score) AS score,
    -- Track which columns were imputed
    CASE WHEN c.age IS NULL THEN 1 ELSE 0 END AS age_imputed,
    CASE WHEN c.income IS NULL THEN 1 ELSE 0 END AS income_imputed
FROM customers c
CROSS JOIN col_means cm;

-- Method 5: Create imputed view for ML
CREATE VIEW v_imputed_data AS
WITH stats AS (
    SELECT
        AVG(col1) AS mean_col1,
        AVG(col2) AS mean_col2,
        -- For categorical: use mode
        MODE() WITHIN GROUP (ORDER BY category) AS mode_category
    FROM raw_data
)
SELECT
    COALESCE(col1, s.mean_col1) AS col1,
    COALESCE(col2, s.mean_col2) AS col2,
    COALESCE(category, s.mode_category) AS category
FROM raw_data r
CROSS JOIN stats s;
```

> **Interview Tip:** Use **COALESCE** for non-destructive imputation (keeps original data intact). For ML, always add an `_imputed` flag column so the model can learn that missingness itself is informative.

---

## Question 19
- [ ] Done

**Create a SQL query that normalizes a column (scales between 0 and 1).**

**Answer:**

Min-Max Normalization: $x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$

```sql
-- Method 1: Subquery approach
SELECT
    id,
    value,
    (value - min_val) / NULLIF(max_val - min_val, 0) AS normalized_value
FROM measurements
CROSS JOIN (
    SELECT MIN(value) AS min_val, MAX(value) AS max_val
    FROM measurements
) stats;

-- Method 2: Window function approach
SELECT
    id,
    value,
    (value - MIN(value) OVER ()) / NULLIF(MAX(value) OVER () - MIN(value) OVER (), 0)
        AS normalized_value
FROM measurements;

-- Method 3: Normalize multiple columns
WITH stats AS (
    SELECT
        MIN(age) AS min_age, MAX(age) AS max_age,
        MIN(income) AS min_income, MAX(income) AS max_income,
        MIN(score) AS min_score, MAX(score) AS max_score
    FROM customers
)
SELECT
    customer_id,
    (age - s.min_age) / NULLIF(s.max_age - s.min_age, 0) AS age_norm,
    (income - s.min_income) / NULLIF(s.max_income - s.min_income, 0) AS income_norm,
    (score - s.min_score) / NULLIF(s.max_score - s.min_score, 0) AS score_norm
FROM customers c
CROSS JOIN stats s;

-- Method 4: Z-Score standardization
SELECT
    id,
    (value - AVG(value) OVER ()) / NULLIF(STDEV(value) OVER (), 0) AS z_score
FROM measurements;

-- Method 5: Group-level normalization
SELECT
    id,
    category,
    value,
    (value - MIN(value) OVER (PARTITION BY category))
    / NULLIF(MAX(value) OVER (PARTITION BY category) - MIN(value) OVER (PARTITION BY category), 0)
        AS normalized_within_group
FROM measurements;
```

> **Interview Tip:** Always use `NULLIF(denominator, 0)` to prevent division by zero when all values are the same. For ML, compute normalization stats on **training data** and apply the same min/max to test data.

---

## Question 20
- [ ] Done

**Generate a feature that is a count over a rolling time window using SQL.**

**Answer:**

```sql
-- Method 1: Window function with RANGE (date-based)
SELECT
    customer_id,
    order_date,
    -- Count of orders in last 30 days
    COUNT(*) OVER (
        PARTITION BY customer_id
        ORDER BY order_date
        RANGE BETWEEN INTERVAL '30' DAY PRECEDING AND CURRENT ROW
    ) AS orders_last_30d,
    
    -- Sum of amount in last 7 days
    SUM(amount) OVER (
        PARTITION BY customer_id
        ORDER BY order_date
        RANGE BETWEEN INTERVAL '7' DAY PRECEDING AND CURRENT ROW
    ) AS spend_last_7d
FROM orders;

-- Method 2: Self-join approach (works in all SQL dialects)
SELECT
    o1.customer_id,
    o1.order_date,
    o1.amount,
    COUNT(o2.order_id) AS orders_last_30d,
    COALESCE(SUM(o2.amount), 0) AS spend_last_30d
FROM orders o1
LEFT JOIN orders o2
    ON o1.customer_id = o2.customer_id
    AND o2.order_date BETWEEN DATEADD(DAY, -30, o1.order_date) AND o1.order_date
GROUP BY o1.customer_id, o1.order_date, o1.amount;

-- Method 3: Multiple rolling windows for ML features
SELECT
    customer_id,
    order_date,
    COUNT(*) OVER (w7) AS orders_7d,
    COUNT(*) OVER (w30) AS orders_30d,
    COUNT(*) OVER (w90) AS orders_90d,
    SUM(amount) OVER (w7) AS spend_7d,
    SUM(amount) OVER (w30) AS spend_30d,
    AVG(amount) OVER (w30) AS avg_spend_30d
FROM orders
WINDOW
    w7 AS (PARTITION BY customer_id ORDER BY order_date RANGE BETWEEN INTERVAL '7' DAY PRECEDING AND CURRENT ROW),
    w30 AS (PARTITION BY customer_id ORDER BY order_date RANGE BETWEEN INTERVAL '30' DAY PRECEDING AND CURRENT ROW),
    w90 AS (PARTITION BY customer_id ORDER BY order_date RANGE BETWEEN INTERVAL '90' DAY PRECEDING AND CURRENT ROW);
```

> **Interview Tip:** Use the `WINDOW` clause to define reusable window frames. Rolling count features at multiple time scales (7, 30, 90 days) capture both short-term and long-term behavioral patterns.

---

## Question 21
- [ ] Done

**Code an SQL function that categorizes continuous variables into bins.**

**Answer:**

```sql
-- Method 1: CASE WHEN — Fixed-width bins
SELECT
    id,
    age,
    CASE
        WHEN age < 18 THEN 'Under 18'
        WHEN age BETWEEN 18 AND 25 THEN '18-25'
        WHEN age BETWEEN 26 AND 35 THEN '26-35'
        WHEN age BETWEEN 36 AND 50 THEN '36-50'
        WHEN age BETWEEN 51 AND 65 THEN '51-65'
        ELSE '65+'
    END AS age_bin
FROM customers;

-- Method 2: WIDTH_BUCKET — Equal-width bins (PostgreSQL, Oracle)
SELECT
    id,
    income,
    WIDTH_BUCKET(income, 0, 200000, 10) AS income_decile  -- 10 equal bins from 0 to 200k
FROM customers;

-- Method 3: NTILE — Equal-frequency bins (quantiles)
SELECT
    id,
    income,
    NTILE(4) OVER (ORDER BY income) AS income_quartile,
    NTILE(10) OVER (ORDER BY income) AS income_decile
FROM customers;

-- Method 4: Custom SQL Function for reusable binning
CREATE FUNCTION fn_bin_value(
    @value FLOAT,
    @min_val FLOAT,
    @max_val FLOAT,
    @n_bins INT
)
RETURNS INT
AS
BEGIN
    DECLARE @bin_width FLOAT = (@max_val - @min_val) / @n_bins;
    DECLARE @bin INT = FLOOR((@value - @min_val) / @bin_width) + 1;
    IF @bin > @n_bins SET @bin = @n_bins;
    IF @bin < 1 SET @bin = 1;
    RETURN @bin;
END;

-- Method 5: Logarithmic binning (for skewed data)
SELECT
    id,
    amount,
    CASE
        WHEN amount <= 0 THEN 'zero_or_negative'
        WHEN LOG10(amount) < 1 THEN 'under_10'
        WHEN LOG10(amount) < 2 THEN '10_to_100'
        WHEN LOG10(amount) < 3 THEN '100_to_1000'
        ELSE 'over_1000'
    END AS amount_log_bin
FROM transactions;
```

> **Interview Tip:** Use `NTILE` for **equal-frequency** bins (each bin has the same number of records) and `WIDTH_BUCKET` for **equal-width** bins (each bin covers the same range). Equal-frequency is generally better for ML features.

---

## Question 22
- [ ] Done

**Implement a SQL solution to compute the TF-IDF score for text data.**

**Answer:**

**TF-IDF** (Term Frequency – Inverse Document Frequency) measures word importance in a document relative to a corpus.

$$TF\text{-}IDF = TF \times IDF = \frac{\text{word count in doc}}{\text{total words in doc}} \times \log\frac{\text{total docs}}{\text{docs containing word}}$$

```sql
-- Step 1: Tokenize (assuming a words table or using string splitting)
-- Create a table of words per document
WITH words AS (
    SELECT
        doc_id,
        LOWER(TRIM(word)) AS word
    FROM documents,
    LATERAL UNNEST(STRING_TO_ARRAY(content, ' ')) AS word
    WHERE TRIM(word) != ''
),

-- Step 2: Compute Term Frequency (TF)
tf AS (
    SELECT
        doc_id,
        word,
        COUNT(*) AS word_count,
        COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY doc_id)::FLOAT AS tf
    FROM words
    GROUP BY doc_id, word
),

-- Step 3: Compute Document Frequency (DF)
df AS (
    SELECT
        word,
        COUNT(DISTINCT doc_id) AS doc_count
    FROM words
    GROUP BY word
),

-- Step 4: Total document count
total_docs AS (
    SELECT COUNT(DISTINCT doc_id) AS n_docs FROM documents
),

-- Step 5: Compute IDF
idf AS (
    SELECT
        word,
        LN(td.n_docs::FLOAT / df.doc_count) AS idf
    FROM df
    CROSS JOIN total_docs td
)

-- Step 6: Compute TF-IDF
SELECT
    tf.doc_id,
    tf.word,
    tf.tf,
    idf.idf,
    ROUND((tf.tf * idf.idf)::NUMERIC, 6) AS tf_idf
FROM tf
JOIN idf ON tf.word = idf.word
ORDER BY tf.doc_id, tf_idf DESC;
```

### Simplified Version for Quick Features

```sql
-- Top keywords per document by TF-IDF
WITH ranked AS (
    SELECT
        doc_id, word, tf_idf,
        ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY tf_idf DESC) AS rank
    FROM tf_idf_scores
)
SELECT * FROM ranked WHERE rank <= 10;  -- top 10 keywords per document
```

> **Interview Tip:** Computing TF-IDF in pure SQL is complex. In practice, use SQL for text extraction and Python’s `TfidfVectorizer` for the actual computation. The SQL approach is useful for understanding the math and for in-database NLP features.

## Question 23
- [ ] Done

**Create a SQL query to pivot a table transforming rows into columns**

### Pivoting Data in SQL

Pivoting transforms **row values** into **column headers**, summarizing data in a cross-tabular format.

### Sample Data
```sql
-- Original table: sales
-- | product | quarter | revenue |
-- |---------|---------|--------|
-- | Widget  | Q1      | 1000   |
-- | Widget  | Q2      | 1500   |
-- | Gadget  | Q1      | 2000   |
-- | Gadget  | Q2      | 2500   |
```

### Method 1: CASE + GROUP BY (Standard SQL — works everywhere)

```sql
SELECT 
    product,
    SUM(CASE WHEN quarter = 'Q1' THEN revenue ELSE 0 END) AS Q1_revenue,
    SUM(CASE WHEN quarter = 'Q2' THEN revenue ELSE 0 END) AS Q2_revenue,
    SUM(CASE WHEN quarter = 'Q3' THEN revenue ELSE 0 END) AS Q3_revenue,
    SUM(CASE WHEN quarter = 'Q4' THEN revenue ELSE 0 END) AS Q4_revenue
FROM sales
GROUP BY product;

-- Result:
-- | product | Q1_revenue | Q2_revenue | Q3_revenue | Q4_revenue |
-- |---------|------------|------------|------------|------------|
-- | Widget  | 1000       | 1500       | 0          | 0          |
-- | Gadget  | 2000       | 2500       | 0          | 0          |
```

### Method 2: PIVOT (SQL Server / Oracle)

```sql
-- SQL Server specific syntax
SELECT product, [Q1], [Q2], [Q3], [Q4]
FROM (
    SELECT product, quarter, revenue
    FROM sales
) AS src
PIVOT (
    SUM(revenue)
    FOR quarter IN ([Q1], [Q2], [Q3], [Q4])
) AS pvt;
```

### Method 3: crosstab (PostgreSQL)

```sql
-- PostgreSQL using tablefunc extension
CREATE EXTENSION IF NOT EXISTS tablefunc;

SELECT * FROM crosstab(
    'SELECT product, quarter, revenue FROM sales ORDER BY 1, 2',
    'SELECT DISTINCT quarter FROM sales ORDER BY 1'
) AS ct(product TEXT, Q1 NUMERIC, Q2 NUMERIC, Q3 NUMERIC, Q4 NUMERIC);
```

### Reverse: UNPIVOT (columns to rows)

```sql
-- SQL Server
SELECT product, quarter, revenue
FROM pivoted_sales
UNPIVOT (revenue FOR quarter IN (Q1, Q2, Q3, Q4)) AS unpvt;

-- Standard SQL equivalent
SELECT product, 'Q1' AS quarter, Q1_revenue AS revenue FROM pivoted_sales
UNION ALL
SELECT product, 'Q2', Q2_revenue FROM pivoted_sales;
```

> **Interview Tip:** The CASE/GROUP BY approach is the most portable. PIVOT syntax varies by database. For ML, pivoting is often used to create **feature matrices** where each column represents a category (one-hot style).

---

## Question 24
- [ ] Done

**Write a SQL query that identifies and removes duplicate records from a dataset**

### Identifying and Removing Duplicates

### Step 1: Identify Duplicates

```sql
-- Find duplicate rows based on specific columns
SELECT 
    name, email, phone,
    COUNT(*) AS duplicate_count
FROM customers
GROUP BY name, email, phone
HAVING COUNT(*) > 1;
```

### Step 2: See All Duplicate Records with Row Numbers

```sql
-- Use ROW_NUMBER to mark duplicates
WITH ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY name, email, phone  -- columns defining uniqueness
            ORDER BY id                       -- keep the first occurrence
        ) AS rn
    FROM customers
)
SELECT * FROM ranked WHERE rn > 1;  -- shows all duplicate rows
```

### Step 3: Remove Duplicates (Keep First Occurrence)

```sql
-- Method 1: DELETE with CTE (SQL Server, PostgreSQL)
WITH ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY name, email, phone
            ORDER BY id
        ) AS rn
    FROM customers
)
DELETE FROM ranked WHERE rn > 1;

-- Method 2: DELETE with subquery (MySQL)
DELETE c1 FROM customers c1
INNER JOIN customers c2
    ON c1.name = c2.name
    AND c1.email = c2.email
    AND c1.phone = c2.phone
    AND c1.id > c2.id;           -- keep lower id

-- Method 3: Create clean table (safe approach)
CREATE TABLE customers_clean AS
SELECT DISTINCT ON (name, email, phone) *  -- PostgreSQL
FROM customers
ORDER BY name, email, phone, id;

-- Or using GROUP BY
CREATE TABLE customers_clean AS
SELECT MIN(id) AS id, name, email, phone
FROM customers
GROUP BY name, email, phone;
```

### ML Context: Deduplication in Data Pipelines

```sql
-- For ML features: deduplicate events per user per day
WITH deduped_events AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY user_id, event_type, DATE(event_time)
            ORDER BY event_time
        ) AS rn
    FROM user_events
)
SELECT * FROM deduped_events WHERE rn = 1;
```

> **Interview Tip:** Always back up data before deleting. Use the CTE + `ROW_NUMBER()` approach — it's the most flexible and readable. For ML datasets, duplicates can bias training; identify whether duplicates are exact copies vs. near-duplicates (fuzzy matching).

---
