# Pandas Interview Questions - General Questions

## Question 1
- [ ] Done

**How can you read and write data from and to a CSV file in Pandas ?**

### Reading CSV
```python
import pandas as pd

# Basic read
df = pd.read_csv('data.csv')

# With options
df = pd.read_csv('data.csv',
                 sep=',',              # Delimiter
                 header=0,             # Row number for header
                 index_col='id',       # Column to use as index
                 usecols=['A', 'B'],   # Only read specific columns
                 dtype={'A': 'int32'}, # Specify dtypes
                 na_values=['NA', ''], # Treat as NaN
                 parse_dates=['date'], # Parse date columns
                 nrows=1000)           # Read only first 1000 rows
```

### Writing CSV
```python
df.to_csv('output.csv',
          index=False,           # Don't write index
          columns=['A', 'B'],    # Only write specific columns
          encoding='utf-8')
```

---

## Question 2
- [ ] Done

**How do you handle missing data in a DataFrame?**

### Detection
```python
import pandas as pd

# Check for missing values
df.isnull()              # Boolean mask
df.isnull().sum()        # Count per column
df.isnull().sum().sum()  # Total count
```

### Handling Methods
```python
# 1. Drop rows with NaN
df.dropna()                    # Drop any row with NaN
df.dropna(subset=['col1'])     # Only check specific column
df.dropna(thresh=3)            # Keep rows with at least 3 non-NaN

# 2. Fill with value
df.fillna(0)                   # Fill all with 0
df['col'].fillna(df['col'].mean())  # Fill with mean
df.fillna(method='ffill')      # Forward fill
df.fillna(method='bfill')      # Backward fill

# 3. Interpolate
df.interpolate()               # Linear interpolation
```

---

## Question 3
- [ ] Done

**How do you apply a function to all elements in a DataFrame column?**

### Methods
```python
import pandas as pd

df = pd.DataFrame({'text': ['hello', 'world', 'pandas']})

# 1. map() - for element-wise operations on Series
df['upper'] = df['text'].map(str.upper)
df['upper'] = df['text'].map(lambda x: x.upper())

# 2. apply() - more flexible
df['length'] = df['text'].apply(len)

# 3. Vectorized string methods (preferred for strings)
df['upper'] = df['text'].str.upper()
df['first_char'] = df['text'].str[0]
```

### Performance Tip
```python
# Slow (apply)
df['result'] = df['num'].apply(lambda x: x * 2)

# Fast (vectorized)
df['result'] = df['num'] * 2
```

---

## Question 4
- [ ] Done

**Demonstrate how to handle duplicate rows in a DataFrame**

### Detection and Removal
```python
import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 2, 3],
    'B': ['x', 'y', 'y', 'z']
})

# Detect duplicates
df.duplicated()                    # Boolean mask
df.duplicated(subset=['A'])        # Check only column A
df.duplicated(keep='last')         # Mark first occurrence as duplicate

# Remove duplicates
df.drop_duplicates()               # Keep first by default
df.drop_duplicates(keep='last')    # Keep last
df.drop_duplicates(subset=['A'])   # Based on column A only

# Count duplicates
df.duplicated().sum()
```

---

## Question 5
- [ ] Done

**How can you pivot data in a DataFrame?**

### pivot_table()
```python
import pandas as pd

df = pd.DataFrame({
    'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
    'product': ['A', 'B', 'A', 'B'],
    'sales': [100, 150, 120, 180]
})

# Create pivot table
pivot = pd.pivot_table(df,
                       values='sales',      # Values to aggregate
                       index='date',        # Rows
                       columns='product',   # Columns
                       aggfunc='sum')       # Aggregation function

#          product    A    B
# date                      
# 2023-01          100  150
# 2023-02          120  180
```

### melt() (unpivot)
```python
# Wide to long format
df_long = pd.melt(pivot.reset_index(),
                  id_vars=['date'],
                  var_name='product',
                  value_name='sales')
```

---

## Question 6
- [ ] Done

**How do you apply conditional logic using where()?**

### where() Method
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})

# Keep values where condition is True, else replace
df['A'].where(df['A'] > 2, other=0)  # [0, 0, 3, 4, 5]

# np.where for if-else logic
df['category'] = np.where(df['A'] > 2, 'high', 'low')
# ['low', 'low', 'high', 'high', 'high']

# Multiple conditions with np.select
conditions = [
    df['A'] <= 2,
    df['A'] <= 4,
    df['A'] > 4
]
choices = ['low', 'medium', 'high']
df['category'] = np.select(conditions, choices)
```

---

## Question 7
- [ ] Done

**How do you reshape a DataFrame using stack and unstack methods?**

### stack() and unstack()
```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'A': [1, 2],
    'B': [3, 4]
}, index=['row1', 'row2'])

# stack(): columns → rows (wide to long)
stacked = df.stack()
# row1  A    1
#       B    3
# row2  A    2
#       B    4

# unstack(): rows → columns (long to wide)
unstacked = stacked.unstack()
# Back to original shape
```

### Use Case
```python
# MultiIndex DataFrame
df = pd.DataFrame({
    ('Sales', 'Q1'): [100, 200],
    ('Sales', 'Q2'): [150, 250]
}, index=['Product A', 'Product B'])

df.stack(level=1)  # Move quarter level to rows
```

---

## Question 8
- [ ] Done

**How can you perform statistical aggregation on DataFrame groups?**

### groupby() Aggregation
```python
import pandas as pd

df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B'],
    'value': [10, 20, 30, 40],
    'count': [1, 2, 3, 4]
})

# Single aggregation
df.groupby('category')['value'].sum()

# Multiple aggregations
df.groupby('category').agg({
    'value': ['sum', 'mean', 'max'],
    'count': 'sum'
})

# Named aggregations (cleaner)
df.groupby('category').agg(
    total_value=('value', 'sum'),
    avg_value=('value', 'mean'),
    total_count=('count', 'sum')
)
```

---

## Question 9
- [ ] Done

**How do you use window functions in Pandas for running calculations?**

### Rolling Window
```python
import pandas as pd

df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

# Rolling average (window of 3)
df['rolling_avg'] = df['value'].rolling(window=3).mean()

# Expanding window (cumulative)
df['cumulative_avg'] = df['value'].expanding().mean()

# Exponential weighted moving average
df['ewm'] = df['value'].ewm(span=3).mean()

# Rolling with min_periods
df['rolling_avg'] = df['value'].rolling(window=3, min_periods=1).mean()
```

---

## Question 10
- [ ] Done

**How do you normalize data within a DataFrame column?**

### Normalization Methods
```python
import pandas as pd

df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})

# 1. Z-score normalization (standardization)
df['z_score'] = (df['value'] - df['value'].mean()) / df['value'].std()

# 2. Min-Max normalization (scale to 0-1)
df['min_max'] = (df['value'] - df['value'].min()) / (df['value'].max() - df['value'].min())

# 3. Max normalization (scale to 0-max)
df['max_norm'] = df['value'] / df['value'].max()

# 4. Group-wise normalization
df['group_norm'] = df.groupby('category')['value'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

---

## Question 11
- [ ] Done

**How do you create simple plots from a DataFrame?**

### Pandas Visualization
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'month': ['Jan', 'Feb', 'Mar', 'Apr'],
    'sales': [100, 150, 120, 180],
    'profit': [20, 35, 25, 45]
})

# Line plot
df.plot(x='month', y='sales', kind='line')

# Bar plot
df.plot(x='month', y='sales', kind='bar')

# Multiple columns
df.plot(x='month', y=['sales', 'profit'], kind='bar')

# Histogram
df['sales'].hist(bins=10)

# Box plot
df[['sales', 'profit']].boxplot()

# Scatter plot
df.plot.scatter(x='sales', y='profit')

plt.show()
```

### Quick Statistics Visualization
```python
# Distribution of all numeric columns
df.hist(figsize=(10, 6))

# Correlation heatmap
import seaborn as sns
sns.heatmap(df.corr(), annot=True)
```


---

## Question 12
- [ ] Done

**What techniques can you use to improve the performance of Pandas operations?**

**Answer:**

### Key Optimization Techniques

| Technique | Speedup | When to Use |
|-----------|---------|-------------|
| Vectorization | 10-100× | Always — replace loops with built-in ops |
| `apply()` → vectorized | 5-50× | Replace `df.apply()` with column math |
| Categorical dtype | 2-10× | Low-cardinality string columns |
| `eval()` / `query()` | 2-5× | Complex expressions on large DataFrames |
| Chunking | N/A | Data larger than RAM |

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'a': range(1_000_000), 'b': range(1_000_000)})

# BAD: Python loop (~10s)
# for i in range(len(df)): df.loc[i, 'c'] = df.loc[i, 'a'] + df.loc[i, 'b']

# GOOD: Vectorized (~5ms)
df['c'] = df['a'] + df['b']

# BAD: apply (~1s)
# df['c'] = df.apply(lambda row: row['a'] + row['b'], axis=1)

# GOOD: eval (fast for large DataFrames)
df['c'] = df.eval('a + b')

# GOOD: query instead of boolean indexing
result = df.query('a > 500 and b < 1000')  # vs df[(df['a']>500) & (df['b']<1000)]

# Use appropriate dtypes
df['category_col'] = df['category_col'].astype('category')  # 90% less memory
df['int_col'] = pd.to_numeric(df['int_col'], downcast='integer')  # int64 → int8

# Use numpy under the hood
df['c'] = np.where(df['a'] > 500, df['a'], df['b'])  # Faster than .apply()

# Parallel with swifter or modin
# import modin.pandas as pd  # Drop-in replacement, auto-parallel
```

> **Interview Tip:** The #1 rule is **avoid Python loops**. Use vectorized operations, and if you must use `apply()`, consider `np.where()` or `np.vectorize()` first.

---

## Question 13
- [ ] Done

**Compare and contrast the memory usage in Pandas for categories vs. objects.**

**Answer:**

### Memory Comparison

```python
import pandas as pd
import numpy as np

# Create sample data
n = 1_000_000
df = pd.DataFrame({
    'color_object': np.random.choice(['red', 'green', 'blue', 'yellow'], n),
})
df['color_category'] = df['color_object'].astype('category')

# Memory usage
print(df['color_object'].memory_usage(deep=True))    # ~62 MB (object dtype)
print(df['color_category'].memory_usage(deep=True))  # ~1 MB  (category dtype)
# ~62× less memory!
```

### How Categories Work

| Aspect | Object (string) | Category |
|--------|-----------------|----------|
| Storage | Full string per row | Integer codes + lookup table |
| Memory (1M rows, 4 unique) | ~62 MB | ~1 MB |
| Groupby speed | Slower | 2-5× faster |
| Sort speed | String comparison | Integer comparison |
| Merge speed | String matching | Integer matching |
| New values | Flexible | Must add to categories first |

```python
# Internal representation
cat = df['color_category']
print(cat.cat.codes[:5])         # [2, 0, 1, 3, 0] (integer codes)
print(cat.cat.categories)        # ['blue', 'green', 'red', 'yellow']

# Ordered categories (for ordinal data)
size_cat = pd.Categorical(['S', 'M', 'L', 'XL'],
                           categories=['S', 'M', 'L', 'XL'],
                           ordered=True)
print(size_cat > 'M')  # [False, False, True, True]

# When NOT to use categories
# - High cardinality (unique values > 50% of rows) — no benefit
# - Columns that need frequent new value insertion
```

> **Interview Tip:** Convert to **category** when cardinality is low (e.g., <50 unique values). It saves memory AND speeds up groupby, merge, and sort operations.

---

## Question 14
- [ ] Done

**How do you manage memory usage when working with large DataFrames?**

**Answer:**

### Strategy 1: Optimize dtypes

```python
import pandas as pd
import numpy as np

# Check current memory
df.info(memory_usage='deep')
print(df.memory_usage(deep=True).sum() / 1e6, 'MB')

# Downcast numeric types
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')  # float64 → float32
        elif col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')  # int64 → int8/16/32
        elif col_type == 'object':
            if df[col].nunique() / len(df) < 0.5:  # Low cardinality
                df[col] = df[col].astype('category')
    return df

df = reduce_mem_usage(df)  # Often reduces by 50-80%
```

### Strategy 2: Load only what you need

```python
# Select columns at read time
df = pd.read_csv('data.csv', usecols=['col1', 'col2', 'col3'])

# Specify dtypes at read time
df = pd.read_csv('data.csv', dtype={'id': 'int32', 'name': 'category', 'value': 'float32'})

# Read in chunks
chunks = pd.read_csv('big.csv', chunksize=100_000)
result = pd.concat([chunk.query('value > 0') for chunk in chunks])
```

### Strategy 3: Delete and collect

```python
import gc

# Drop unneeded columns
df.drop(['unnecessary_col'], axis=1, inplace=True)

# Delete intermediate DataFrames
del temp_df
gc.collect()  # Force garbage collection

# Use inplace operations
df.fillna(0, inplace=True)  # Avoids creating a copy
```

### Strategy 4: Alternative backends

```python
# Parquet (columnar, compressed)
df.to_parquet('data.parquet')  # 5-10× smaller than CSV
df = pd.read_parquet('data.parquet', columns=['col1', 'col2'])

# PyArrow backend (Pandas 2.0+)
df = pd.read_csv('data.csv', dtype_backend='pyarrow')  # ~50% less memory
```

| Strategy | Savings |
|----------|--------|
| float64 → float32 | 50% |
| int64 → int8 | 87.5% |
| object → category | 60-95% |
| Parquet over CSV | 70-90% |

> **Interview Tip:** The biggest wins come from **dtype optimization** (especially category for strings) and **reading only needed columns**. For truly massive data, switch to **Dask** or **Polars**.

---

## Question 15
- [ ] Done

**How can you use chunking to process large CSV files with Pandas?**

**Answer:**

```python
import pandas as pd
import numpy as np

# === Basic Chunked Reading ===
chunks = pd.read_csv('large_file.csv', chunksize=100_000)

for chunk in chunks:
    print(f"Processing {len(chunk)} rows")
    # Process each chunk
    process(chunk)

# === Aggregation across chunks ===
def chunked_statistics(filename, chunksize=100_000):
    total_sum = 0
    total_count = 0
    value_counts = pd.Series(dtype='int64')
    
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        total_sum += chunk['value'].sum()
        total_count += len(chunk)
        value_counts = value_counts.add(chunk['category'].value_counts(), fill_value=0)
    
    mean = total_sum / total_count
    return mean, value_counts

# === Filter and concatenate ===
filtered_chunks = []
for chunk in pd.read_csv('large.csv', chunksize=50_000):
    filtered = chunk[chunk['amount'] > 100]  # Filter rows
    filtered_chunks.append(filtered)

result = pd.concat(filtered_chunks, ignore_index=True)

# === Write results incrementally ===
header_written = False
for chunk in pd.read_csv('input.csv', chunksize=100_000):
    processed = chunk.copy()
    processed['new_col'] = processed['value'] * 2
    
    processed.to_csv('output.csv',
                     mode='a',
                     header=not header_written,
                     index=False)
    header_written = True

# === Memory-efficient groupby ===
grouped_results = {}
for chunk in pd.read_csv('sales.csv', chunksize=100_000):
    chunk_grouped = chunk.groupby('region')['revenue'].sum()
    for region, revenue in chunk_grouped.items():
        grouped_results[region] = grouped_results.get(region, 0) + revenue

final = pd.Series(grouped_results)

# === Using context manager ===
with pd.read_csv('large.csv', chunksize=50_000) as reader:
    for chunk in reader:
        process(chunk)
```

| Parameter | Description |
|-----------|------------|
| `chunksize=N` | Rows per chunk |
| `nrows=N` | Read only first N rows |
| `usecols=[...]` | Read only specific columns |
| `dtype={...}` | Specify dtypes to save memory |

> **Interview Tip:** Chunking keeps memory constant regardless of file size. Combine with `usecols` and `dtype` for maximum efficiency. For complex aggregations, consider **Dask** which handles chunking automatically.
