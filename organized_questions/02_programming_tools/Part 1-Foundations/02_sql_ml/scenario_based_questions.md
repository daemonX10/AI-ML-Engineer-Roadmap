# SQL ML Interview Questions - Scenario-Based Questions

## Question 1
- [ ] Done

**How would you write a SQL query to select distinct values from a column?**

### Code
```sql
-- Basic DISTINCT
SELECT DISTINCT city FROM customers;

-- Count distinct values
SELECT COUNT(DISTINCT city) AS unique_cities FROM customers;

-- Distinct on multiple columns
SELECT DISTINCT city, state FROM customers;
```

---

## Question 2
- [ ] Done

**Scenario: You have a customer table with duplicate emails. How would you identify and handle them?**

### Solution
```sql
-- Step 1: Identify duplicates
SELECT email, COUNT(*) AS count
FROM customers
GROUP BY email
HAVING COUNT(*) > 1;

-- Step 2: See all duplicate records with details
WITH Ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY email 
            ORDER BY created_date DESC
        ) AS rn
    FROM customers
)
SELECT * FROM Ranked WHERE email IN (
    SELECT email FROM customers GROUP BY email HAVING COUNT(*) > 1
);

-- Step 3: Keep only the most recent record per email
DELETE FROM customers
WHERE id NOT IN (
    SELECT id FROM (
        SELECT id, ROW_NUMBER() OVER (
            PARTITION BY email ORDER BY created_date DESC
        ) AS rn
        FROM customers
    ) t WHERE rn = 1
);
```

---

## Question 3
- [ ] Done

**Scenario: Build a churn prediction feature set for customers who haven't purchased in 30 days.**

### Solution
```sql
WITH CustomerActivity AS (
    SELECT 
        c.customer_id,
        c.signup_date,
        MAX(o.order_date) AS last_order_date,
        COUNT(o.order_id) AS total_orders,
        SUM(o.amount) AS total_spent,
        AVG(o.amount) AS avg_order_value,
        DATEDIFF(CURRENT_DATE, MAX(o.order_date)) AS days_since_last_order
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.signup_date
)
SELECT 
    customer_id,
    total_orders,
    total_spent,
    avg_order_value,
    days_since_last_order,
    DATEDIFF(CURRENT_DATE, signup_date) AS customer_age_days,
    
    -- Churn label (target variable)
    CASE 
        WHEN days_since_last_order > 30 OR days_since_last_order IS NULL 
        THEN 1 ELSE 0 
    END AS is_churned

FROM CustomerActivity;
```

---

## Question 4
- [ ] Done

**Scenario: Calculate customer RFM (Recency, Frequency, Monetary) scores.**

### Solution
```sql
WITH RFM_Metrics AS (
    SELECT 
        customer_id,
        DATEDIFF(CURRENT_DATE, MAX(purchase_date)) AS recency,
        COUNT(*) AS frequency,
        SUM(amount) AS monetary
    FROM purchases
    WHERE purchase_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR)
    GROUP BY customer_id
),
RFM_Scores AS (
    SELECT 
        customer_id,
        recency,
        frequency,
        monetary,
        NTILE(5) OVER (ORDER BY recency DESC) AS R_score,      -- Lower recency = higher score
        NTILE(5) OVER (ORDER BY frequency) AS F_score,          -- Higher frequency = higher score
        NTILE(5) OVER (ORDER BY monetary) AS M_score            -- Higher monetary = higher score
    FROM RFM_Metrics
)
SELECT 
    customer_id,
    recency,
    frequency,
    monetary,
    R_score,
    F_score,
    M_score,
    CONCAT(R_score, F_score, M_score) AS RFM_segment,
    R_score + F_score + M_score AS total_score
FROM RFM_Scores
ORDER BY total_score DESC;
```

---

## Question 5
- [ ] Done

**Scenario: Find products frequently bought together for recommendation system.**

### Solution
```sql
-- Find product pairs bought in same order
WITH OrderProducts AS (
    SELECT 
        o.order_id,
        oi.product_id,
        p.product_name
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
)
SELECT 
    a.product_name AS product_1,
    b.product_name AS product_2,
    COUNT(*) AS times_bought_together
FROM OrderProducts a
JOIN OrderProducts b ON a.order_id = b.order_id 
    AND a.product_id < b.product_id  -- Avoid duplicates
GROUP BY a.product_name, b.product_name
HAVING COUNT(*) >= 10
ORDER BY times_bought_together DESC
LIMIT 20;
```

---

## Question 6
- [ ] Done

**Scenario: Detect anomalous transactions (potential fraud).**

### Solution
```sql
WITH CustomerStats AS (
    SELECT 
        customer_id,
        AVG(amount) AS avg_amount,
        STDDEV(amount) AS std_amount
    FROM transactions
    WHERE transaction_date >= DATE_SUB(CURRENT_DATE, INTERVAL 90 DAY)
    GROUP BY customer_id
)
SELECT 
    t.transaction_id,
    t.customer_id,
    t.amount,
    t.transaction_date,
    cs.avg_amount,
    cs.std_amount,
    (t.amount - cs.avg_amount) / NULLIF(cs.std_amount, 0) AS z_score,
    
    -- Flag anomalies
    CASE 
        WHEN ABS((t.amount - cs.avg_amount) / NULLIF(cs.std_amount, 0)) > 3 
        THEN 'ANOMALY'
        ELSE 'NORMAL'
    END AS flag
    
FROM transactions t
JOIN CustomerStats cs ON t.customer_id = cs.customer_id
WHERE t.transaction_date >= DATE_SUB(CURRENT_DATE, INTERVAL 7 DAY)
ORDER BY z_score DESC;
```

---

## Question 7
- [ ] Done

**Scenario: Create time-based features for time series forecasting.**

### Solution
```sql
SELECT 
    sale_date,
    amount,
    
    -- Lag features
    LAG(amount, 1) OVER (ORDER BY sale_date) AS sales_lag_1,
    LAG(amount, 7) OVER (ORDER BY sale_date) AS sales_lag_7,
    LAG(amount, 30) OVER (ORDER BY sale_date) AS sales_lag_30,
    
    -- Moving averages
    AVG(amount) OVER (
        ORDER BY sale_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d,
    
    -- Date features
    DAYOFWEEK(sale_date) AS day_of_week,
    MONTH(sale_date) AS month,
    CASE WHEN DAYOFWEEK(sale_date) IN (1, 7) THEN 1 ELSE 0 END AS is_weekend,
    
    -- Year-over-year comparison
    LAG(amount, 365) OVER (ORDER BY sale_date) AS sales_same_day_last_year

FROM daily_sales
ORDER BY sale_date;
```

---

## Question 8
- [ ] Done

**Scenario: Segment customers into tiers based on spending.**

### Solution
```sql
WITH CustomerSpending AS (
    SELECT 
        customer_id,
        SUM(amount) AS total_spent,
        COUNT(*) AS order_count
    FROM orders
    WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR)
    GROUP BY customer_id
)
SELECT 
    customer_id,
    total_spent,
    order_count,
    NTILE(4) OVER (ORDER BY total_spent) AS spending_quartile,
    CASE 
        WHEN total_spent >= (SELECT PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY total_spent) FROM CustomerSpending) 
            THEN 'Platinum'
        WHEN total_spent >= (SELECT PERCENTILE_CONT(0.7) WITHIN GROUP (ORDER BY total_spent) FROM CustomerSpending) 
            THEN 'Gold'
        WHEN total_spent >= (SELECT PERCENTILE_CONT(0.4) WITHIN GROUP (ORDER BY total_spent) FROM CustomerSpending) 
            THEN 'Silver'
        ELSE 'Bronze'
    END AS customer_tier
FROM CustomerSpending
ORDER BY total_spent DESC;
```

