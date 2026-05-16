# Pandas Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What is Pandas in Python and why is it used for data analysis?**

### Definition
Pandas is an open-source Python library for data manipulation and analysis. Built on NumPy, it provides high-performance, easy-to-use data structures (DataFrame and Series) for working with structured data.

### Why It's Essential for Data Analysis

| Feature | Description |
|---------|-------------|
| **DataFrame** | 2D table structure like Excel/SQL |
| **Data Alignment** | Automatic alignment based on labels |
| **Missing Data** | Built-in handling of NaN values |
| **Reshaping** | Pivot, melt, stack operations |
| **Grouping** | SQL-like GROUP BY operations |
| **Time Series** | First-class datetime support |
| **Integration** | Works with NumPy, Matplotlib, Scikit-learn |

### Code Example
```python
import pandas as pd

# Read data
df = pd.read_csv('sales.csv')

# Quick analysis
print(df.head())           # First 5 rows
print(df.describe())       # Statistics
print(df.info())           # Data types and memory

# Grouping and aggregation
sales_by_region = df.groupby('region')['amount'].sum()
```

---

## Question 2
- [ ] Done

**Explain the difference between a Series and a DataFrame in Pandas**

### Comparison

| Feature | Series | DataFrame |
|---------|--------|-----------|
| Dimensions | 1D (column) | 2D (table) |
| Analogy | One column | Entire spreadsheet |
| Data Types | Homogeneous | Heterogeneous (per column) |
| Components | Values + Index | Values + Index + Columns |

### Code Example
```python
import pandas as pd

# Series: 1D labeled array
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'], name='values')
print(s)
# a    10
# b    20
# c    30

# DataFrame: 2D table
df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 30],
    'salary': [50000, 60000]
})

# Accessing a column returns a Series
print(type(df['name']))  # <class 'pandas.core.series.Series'>
```

---

## Question 3
- [ ] Done

**What are Pandas indexes , and how are they used?**

### Definition
An index is an immutable array of labels that identifies rows (and columns). It enables fast lookups, data alignment, and relational operations.

### Key Uses

| Use | Description |
|-----|-------------|
| Label-based Selection | `df.loc['row_label']` |
| Data Alignment | Auto-aligns during operations |
| Fast Lookups | O(1) for hash-based indexes |
| Joins/Merges | Acts as keys for combining data |
| MultiIndex | Hierarchical indexing for complex data |

### Code Example
```python
import pandas as pd

df = pd.DataFrame({
    'product_id': ['P101', 'P102', 'P103'],
    'price': [50, 75, 120]
})

# Set meaningful index
df.set_index('product_id', inplace=True)

# Fast label-based lookup
print(df.loc['P102', 'price'])  # 75

# Data alignment works on index
prices_new = pd.Series([60, 80], index=['P101', 'P102'])
df['new_price'] = prices_new  # Auto-aligns by index
```

---

## Question 4
- [ ] Done

**Explain the concept of data alignment and broadcasting in Pandas**

### Data Alignment
Pandas automatically aligns data based on index labels before operations. Mismatched labels result in NaN.

```python
import pandas as pd

s1 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
s2 = pd.Series([100, 200, 300], index=['c', 'b', 'd'])

result = s1 + s2
# a      NaN   (only in s1)
# b    220.0   (20 + 200)
# c    130.0   (30 + 100)
# d      NaN   (only in s2)
```

### Broadcasting
Operations between DataFrame and Series broadcast across rows/columns.

```python
df = pd.DataFrame([[1, 2, 3],
                   [4, 5, 6]], columns=['A', 'B', 'C'])

row = df.iloc[0]  # Series: A=1, B=2, C=3

# Subtract first row from all rows
result = df - row
#    A  B  C
# 0  0  0  0
# 1  3  3  3
```

---

## Question 5
- [ ] Done

**What is data slicing vs filtering in Pandas?**

### Comparison

| Aspect | Slicing | Filtering |
|--------|---------|-----------|
| Purpose | Select by position/range | Select by condition |
| Syntax | `df.iloc[1:4]`, `df.loc['a':'c']` | `df[df['col'] > 5]` |
| Mechanism | Index/position based | Boolean mask based |

### Code Example
```python
import pandas as pd

df = pd.DataFrame({'A': [10, 20, 30, 40, 50],
                   'B': ['x', 'y', 'x', 'y', 'z']})

# --- Slicing ---
# Positional: rows 1-3
print(df.iloc[1:4])

# Label-based (if string index)
df.index = list('abcde')
print(df.loc['b':'d'])

# --- Filtering ---
# Rows where A > 25
print(df[df['A'] > 25])

# Multiple conditions
print(df[(df['A'] > 20) & (df['B'] == 'y')])
```

---

## Question 6
- [ ] Done

**Describe how joining and merging data works in Pandas**

### Join Types

| Type | Returns |
|------|---------|
| `inner` | Only matching keys |
| `left` | All from left + matched from right |
| `right` | All from right + matched from left |
| `outer` | All from both |

### Code Example
```python
import pandas as pd

customers = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103],
    'customer_id': [1, 2, 4],
    'amount': [50, 75, 120]
})

# Left join: keep all customers
result = pd.merge(customers, orders, on='customer_id', how='left')
#    customer_id     name  order_id  amount
# 0            1    Alice     101.0    50.0
# 1            2      Bob     102.0    75.0
# 2            3  Charlie       NaN     NaN
```

### Methods
- `pd.merge()`: Database-style joins on columns
- `df.join()`: Join on index
- `pd.concat()`: Stack DataFrames (not a true join)

---

## Question 7
- [ ] Done

**How do you convert categorical data to numerical?**

### Three Methods

| Method | Use Case | Result |
|--------|----------|--------|
| One-Hot Encoding | Nominal (no order) | Binary columns |
| Label Encoding | Ordinal (has order) | Integer codes |
| Manual Mapping | Custom order | Defined integers |

### Code Examples
```python
import pandas as pd

df = pd.DataFrame({'color': ['Red', 'Green', 'Blue', 'Green'],
                   'size': ['Small', 'Large', 'Medium', 'Small']})

# 1. One-Hot Encoding (nominal)
one_hot = pd.get_dummies(df['color'], prefix='color')
#    color_Blue  color_Green  color_Red
# 0           0            0          1
# 1           0            1          0

# 2. Label Encoding (ordinal)
size_order = pd.Categorical(df['size'], 
                            categories=['Small', 'Medium', 'Large'],
                            ordered=True)
df['size_encoded'] = size_order.codes  # 0, 2, 1, 0

# 3. Manual Mapping
risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['risk_code'] = df['risk'].map(risk_map)
```

---

## Question 8
- [ ] Done

**What is the purpose of the apply() function in Pandas ?**

### Definition
`apply()` applies a function along an axis of a DataFrame (row-wise or column-wise). It's flexible but slower than vectorized operations.

### Axis Parameter
- `axis=0`: Apply to each column
- `axis=1`: Apply to each row

### Code Example
```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [10, 20, 30]})

# Row-wise custom function
def custom_calc(row):
    if row['A'] > 1:
        return row['A'] * row['B']
    return row['B']

df['C'] = df.apply(custom_calc, axis=1)
#    A   B    C
# 0  1  10   10
# 1  2  20   40
# 2  3  30   90
```

### Performance Note
Prefer vectorized operations when possible:
```python
# Slow (apply)
df['result'] = df.apply(lambda row: row['A'] + row['B'], axis=1)

# Fast (vectorized)
df['result'] = df['A'] + df['B']
```

---

## Question 9
- [ ] Done

**Explain astype, to_numeric, and pd.to_datetime.**

### Comparison

| Function | Purpose | Error Handling |
|----------|---------|----------------|
| `astype()` | General type conversion | Raises error on failure |
| `pd.to_numeric()` | Convert to number | `errors='coerce'` → NaN |
| `pd.to_datetime()` | Convert to datetime | `errors='coerce'` → NaT |

### Code Examples
```python
import pandas as pd

# astype: simple conversion
df['age'] = df['age'].astype('int32')
df['category'] = df['category'].astype('category')

# to_numeric: handles errors gracefully
s = pd.Series(['1', '2', 'bad', '4'])
pd.to_numeric(s, errors='coerce')  # [1, 2, NaN, 4]

# to_datetime: parse dates
s = pd.Series(['2023-01-01', '2023-02-15', 'invalid'])
pd.to_datetime(s, errors='coerce')  # [2023-01-01, 2023-02-15, NaT]
```

---

## Question 10
- [ ] Done

**Explain data ranking in Pandas.**

### Ranking Methods

| Method | Description |
|--------|-------------|
| `average` | Mean of tied ranks (default) |
| `min` | Lowest rank for ties |
| `max` | Highest rank for ties |
| `first` | Ranks in order of appearance |
| `dense` | Like min, but no gaps |

### Code Example
```python
import pandas as pd

s = pd.Series([3, 1, 4, 1, 5])

print(s.rank(method='average'))  # [3.0, 1.5, 4.0, 1.5, 5.0]
print(s.rank(method='min'))      # [3.0, 1.0, 4.0, 1.0, 5.0]
print(s.rank(method='dense'))    # [2.0, 1.0, 3.0, 1.0, 4.0]
```

---

## Question 11
- [ ] Done

**What is a crosstab in Pandas?**

### Definition
`pd.crosstab()` computes a cross-tabulation (frequency table) of two or more categorical variables. It's useful for exploring relationships between categories.

### Code Example
```python
import pandas as pd

df = pd.DataFrame({
    'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
    'product': ['A', 'A', 'B', 'A', 'B', 'B'],
    'purchased': [1, 1, 0, 1, 1, 0]
})

# Simple frequency table
print(pd.crosstab(df['gender'], df['product']))
# product  A  B
# gender       
# F        2  1
# M        1  2

# With aggregation
print(pd.crosstab(df['gender'], df['product'], 
                  values=df['purchased'], aggfunc='sum'))
```

---

## Question 12
- [ ] Done

**How do you perform a MultiIndex query?**

### Definition
MultiIndex (hierarchical index) allows multiple levels of indexing. Query using `.loc` with tuples or `xs()` for cross-section selection.

### Code Example
```python
import pandas as pd

# Create MultiIndex DataFrame
arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
index = pd.MultiIndex.from_arrays(arrays, names=['letter', 'number'])
df = pd.DataFrame({'value': [10, 20, 30, 40]}, index=index)

#                 value
# letter number       
# A      1           10
#        2           20
# B      1           30
#        2           40

# Query specific level
print(df.loc['A'])           # All rows where letter='A'
print(df.loc[('A', 1)])      # Specific combination
print(df.xs(1, level='number'))  # All rows where number=1
```

---

## Question 13
- [ ] Done

**How do you export a DataFrame to different file formats?**

### Export Methods

| Format | Method | Common Parameters |
|--------|--------|-------------------|
| CSV | `df.to_csv()` | `index=False`, `encoding='utf-8'` |
| Excel | `df.to_excel()` | `sheet_name`, `index=False` |
| JSON | `df.to_json()` | `orient='records'` |
| Parquet | `df.to_parquet()` | `compression='snappy'` |
| SQL | `df.to_sql()` | `if_exists='replace'` |

### Code Example
```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})

# CSV
df.to_csv('output.csv', index=False)

# Excel
df.to_excel('output.xlsx', sheet_name='Data', index=False)

# JSON
df.to_json('output.json', orient='records')

# Parquet (efficient for large data)
df.to_parquet('output.parquet')
```

---

## Question 14
- [ ] Done

**How to handle larger-than-memory data with Dask or Modin?**

### Dask
Dask provides parallel computing and out-of-core processing. It uses lazy evaluation.

```python
import dask.dataframe as dd

# Read large CSV in chunks
df = dd.read_csv('large_file.csv')

# Operations are lazy (not executed yet)
result = df.groupby('category').sum()

# Execute and get pandas DataFrame
result_pd = result.compute()
```

### Modin
Modin is a drop-in replacement that parallelizes Pandas operations.

```python
import modin.pandas as pd  # Just change the import!

df = pd.read_csv('large_file.csv')  # Automatically parallelized
df.groupby('category').mean()  # Uses all CPU cores
```

---

## Question 15
- [ ] Done

**How to use Pandas to preprocess data for ML?**

### Common Preprocessing Steps

```python
import pandas as pd
import numpy as np

# 1. Load data
df = pd.read_csv('data.csv')

# 2. Handle missing values
df['age'].fillna(df['age'].median(), inplace=True)
df.dropna(subset=['target'], inplace=True)

# 3. Encode categorical variables
df = pd.get_dummies(df, columns=['category'])

# 4. Feature scaling
df['age_scaled'] = (df['age'] - df['age'].mean()) / df['age'].std()

# 5. Create features
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100],
                         labels=['child', 'young', 'middle', 'senior'])

# 6. Split features and target
X = df.drop('target', axis=1)
y = df['target']
```

---

## Question 16
- [ ] Done

**What are some strategies for optimizing Pandas code performance ?**

### Optimization Strategies

| Strategy | Impact |
|----------|--------|
| Use `category` dtype | 90%+ memory reduction for low cardinality |
| Downcast numeric types | int64→int32, float64→float32 |
| Use vectorized operations | 10-100x faster than apply() |
| Use `query()` for filtering | More readable, sometimes faster |
| Avoid chained indexing | Use `.loc[]` instead |

### Code Example
```python
import pandas as pd

# 1. Use category dtype
df['country'] = df['country'].astype('category')

# 2. Downcast numerics
df['id'] = pd.to_numeric(df['id'], downcast='integer')

# 3. Vectorized vs apply
# Slow
df['result'] = df.apply(lambda x: x['a'] + x['b'], axis=1)
# Fast
df['result'] = df['a'] + df['b']

# 4. Use query() for readability
df.query('age > 30 and salary > 50000')
```

---

## Question 17
- [ ] Done

**Explain the importance of categorical data types.**

### Benefits

| Benefit | Description |
|---------|-------------|
| Memory Efficiency | Stores as integers, not strings |
| Performance | Faster groupby, sorting |
| Ordinality | Can define order for ordinal data |
| Type Safety | Prevents invalid categories |

### Code Example
```python
import pandas as pd

# Before: object dtype (inefficient)
df = pd.DataFrame({'country': ['USA', 'UK', 'USA', 'UK'] * 100000})
print(f"Object memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

# After: category dtype (efficient)
df['country'] = df['country'].astype('category')
print(f"Category memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
# Typically 90%+ reduction!

# Ordinal categories
size_type = pd.CategoricalDtype(categories=['S', 'M', 'L', 'XL'], ordered=True)
df['size'] = df['size'].astype(size_type)

# Now comparisons work
df[df['size'] > 'M']  # Returns L and XL
```

## Question 18
- [ ] Done

**Discuss the use of groupby in Pandas and provide an example**

### GroupBy in Pandas

The `groupby()` operation follows the **split-apply-combine** pattern:
1. **Split**: Divide data into groups based on key(s)
2. **Apply**: Perform a function on each group
3. **Combine**: Merge results back together

### Basic Usage

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'department': ['Sales', 'Sales', 'Engineering', 'Engineering', 'HR', 'HR'],
    'employee': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    'salary': [70000, 80000, 95000, 105000, 65000, 60000],
    'experience_years': [3, 5, 7, 10, 2, 4]
})

# Basic aggregation
result = df.groupby('department')['salary'].mean()
print(result)
# department
# Engineering    100000.0
# HR              62500.0
# Sales           75000.0

# Multiple aggregations
result = df.groupby('department').agg(
    avg_salary=('salary', 'mean'),
    max_salary=('salary', 'max'),
    headcount=('employee', 'count'),
    avg_experience=('experience_years', 'mean')
)
```

### Advanced GroupBy Operations

```python
# Transform: returns same-shaped result (useful for broadcast)
df['salary_zscore'] = df.groupby('department')['salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Filter: keep only groups meeting a condition
high_salary_depts = df.groupby('department').filter(
    lambda x: x['salary'].mean() > 70000
)

# Apply: custom function per group
def top_earner(group):
    return group.nlargest(1, 'salary')

df.groupby('department').apply(top_earner)

# Multiple groupby keys
df.groupby(['department', 'experience_years'])['salary'].mean()

# Named aggregations with different functions per column
df.groupby('department').agg(
    salary_mean=('salary', 'mean'),
    salary_std=('salary', 'std'),
    exp_median=('experience_years', 'median')
)
```

### GroupBy for ML Feature Engineering
```python
# Create aggregated features per group
df['dept_avg_salary'] = df.groupby('department')['salary'].transform('mean')
df['salary_vs_dept'] = df['salary'] / df['dept_avg_salary']
df['dept_rank'] = df.groupby('department')['salary'].rank(ascending=False)
```

> **Interview Tip:** `groupby().transform()` is the most useful for ML — it broadcasts group-level statistics back to each row (e.g., "salary relative to department average"). Avoid `apply()` when possible as it's slower; prefer built-in aggregations like `mean()`, `sum()`, `count()`.

---

## Question 19
- [ ] Done

**What is data slicing in Pandas , and how does it differ from filtering ?**

### Data Slicing vs. Filtering

| Aspect | Slicing | Filtering |
|--------|---------|----------|
| **What it does** | Selects by position or label range | Selects rows meeting a condition |
| **Methods** | `iloc[]`, `loc[]`, `:` | Boolean indexing, `query()` |
| **Returns** | Contiguous subset | Potentially non-contiguous rows |
| **Use case** | "Give me rows 10-20" | "Give me rows where age > 30" |

### Slicing

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age': [25, 30, 35, 40, 45],
    'salary': [50000, 60000, 70000, 80000, 90000]
})

# Position-based slicing (iloc)
df.iloc[1:3]        # rows 1-2 (exclusive end)
df.iloc[:, 0:2]     # first 2 columns
df.iloc[0:3, 1:]    # rows 0-2, columns from index 1

# Label-based slicing (loc) — inclusive end!
df.loc[1:3]          # rows with labels 1, 2, 3 (inclusive!)
df.loc[:, 'name':'age']  # columns from 'name' to 'age'

# Shorthand slicing
df[:3]              # first 3 rows
df['name']          # single column
df[['name', 'age']] # multiple columns
```

### Filtering

```python
# Boolean indexing
df[df['age'] > 30]                        # age over 30
df[(df['age'] > 25) & (df['salary'] > 60000)]  # multiple conditions
df[df['name'].isin(['Alice', 'Bob'])]     # specific values
df[df['name'].str.contains('li')]         # string matching

# query() method (cleaner syntax)
df.query('age > 30 and salary > 60000')
df.query('name in ["Alice", "Bob"]')

# Filtering with functions
df[df['age'].between(25, 35)]             # range filter
df[df['salary'].apply(lambda x: x > 65000)]  # custom function
```

### Key Differences
```python
# Slicing: always contiguous
df.iloc[2:5]  # rows 2, 3, 4 — always sequential

# Filtering: may skip rows
df[df['age'] > 30]  # rows 2, 3, 4 — could be any pattern

# Performance: slicing is faster (no condition evaluation)
# Slicing:   O(1) for view creation
# Filtering: O(n) to evaluate condition per row
```

> **Interview Tip:** Use `iloc` for positional access, `loc` for label-based access. Remember `loc` has inclusive endpoints while `iloc` has exclusive endpoints (like Python slicing). For ML, filtering is more common (selecting training subsets by condition).

---

## Question 20
- [ ] Done

**Describe how you would convert categorical data into numeric format**

### Encoding Categorical Data

| Method | Type | When to Use | Output |
|--------|------|------------|--------|
| **One-Hot Encoding** | Nominal | Low cardinality (<20 categories) | Binary columns |
| **Label Encoding** | Ordinal | Ordered categories | Single integer column |
| **Ordinal Encoding** | Ordinal | Custom ordering | Single integer column |
| **Target Encoding** | Any | High cardinality | Float column |
| **Frequency Encoding** | Any | Simple baseline | Float column |
| **Binary Encoding** | Nominal | Medium cardinality | Multiple binary columns |

### Implementation

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['S', 'M', 'L', 'XL', 'M'],
    'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA'],
    'target': [1, 0, 1, 0, 1]
})

# 1. One-Hot Encoding (pd.get_dummies)
df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True)
# Adds: color_green, color_red (drops color_blue as reference)

# 2. Label Encoding (sklearn)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['color_label'] = le.fit_transform(df['color'])  # blue=0, green=1, red=2

# 3. Ordinal Encoding (manual mapping)
size_order = {'S': 1, 'M': 2, 'L': 3, 'XL': 4}
df['size_ordinal'] = df['size'].map(size_order)

# 4. Ordinal Encoding (sklearn)
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[['S', 'M', 'L', 'XL']])
df['size_sklearn'] = oe.fit_transform(df[['size']])

# 5. Target Encoding (mean of target per category)
target_means = df.groupby('city')['target'].mean()
df['city_target_enc'] = df['city'].map(target_means)

# 6. Frequency Encoding
freq = df['city'].value_counts(normalize=True)
df['city_freq'] = df['city'].map(freq)

# 7. Binary Encoding (category_encoders library)
# pip install category_encoders
# import category_encoders as ce
# encoder = ce.BinaryEncoder(cols=['city'])
# df_binary = encoder.fit_transform(df)
```

### Choosing the Right Encoding
```
Nominal + Low cardinality  →  One-Hot (get_dummies)
Nominal + High cardinality →  Target Encoding or Frequency Encoding
Ordinal                    →  Ordinal Encoding (with explicit order)
Tree-based models          →  Label Encoding works fine
Linear models              →  One-Hot Encoding required
```

> **Interview Tip:** Never use Label Encoding for **nominal** (unordered) data with linear models — it implies false ordering (blue=0 < green=1 < red=2). Tree-based models (XGBoost, Random Forest) handle label encoding fine. Use `drop_first=True` in one-hot encoding to avoid multicollinearity.

---

## Question 21
- [ ] Done

**Show how to apply conditional logic to columns using the where() method**

### Conditional Logic with `where()` and Alternatives

### `DataFrame.where()`

`where()` keeps values where the condition is `True` and replaces with another value where `False`.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'score': [85, 42, 73, 91, 58, 67],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank']
})

# where(): keep values where condition is True, replace others
df['score_pass'] = df['score'].where(df['score'] >= 60, other=0)
# [85, 0, 73, 91, 0, 67]  — scores below 60 become 0

# mask(): opposite of where() — replace where condition IS True
df['score_masked'] = df['score'].mask(df['score'] >= 60, other=-1)
# [-1, 42, -1, -1, 58, -1]  — scores >= 60 become -1
```

### Alternatives for Conditional Logic

```python
# 1. np.where() — if-else (most common)
df['grade'] = np.where(df['score'] >= 60, 'Pass', 'Fail')

# 2. np.select() — multiple conditions
conditions = [
    df['score'] >= 90,
    df['score'] >= 80,
    df['score'] >= 70,
    df['score'] >= 60,
]
choices = ['A', 'B', 'C', 'D']
df['letter_grade'] = np.select(conditions, choices, default='F')
# [B, F, C, A, F, D]

# 3. pd.cut() — binning by value ranges
df['category'] = pd.cut(df['score'], bins=[0, 60, 70, 80, 90, 100],
                         labels=['F', 'D', 'C', 'B', 'A'])

# 4. apply() with custom function (flexible but slower)
def classify(score):
    if score >= 90: return 'Excellent'
    elif score >= 70: return 'Good'
    else: return 'Needs Improvement'

df['classification'] = df['score'].apply(classify)

# 5. map() with dictionary
df['pass_fail'] = df['letter_grade'].map({'A': 'Pass', 'B': 'Pass', 
                                           'C': 'Pass', 'D': 'Pass', 'F': 'Fail'})
```

### Performance Comparison
| Method | Speed | Flexibility |
|--------|-------|------------|
| `np.where()` | Fastest | Binary condition |
| `np.select()` | Fast | Multiple conditions |
| `pd.cut()` | Fast | Numeric binning |
| `.where()` | Fast | Keep/replace pattern |
| `.apply()` | Slow | Any logic |

> **Interview Tip:** Prefer `np.where()` for simple if-else and `np.select()` for multiple conditions — both are vectorized and fast. Avoid `apply()` for simple conditions as it's 10-100x slower than vectorized alternatives.

---

## Question 22
- [ ] Done

**Explain the usage and differences between astype , to_numeric , and pd.to_datetime**

### Type Conversion Methods in Pandas

| Method | Purpose | Error Handling | Best For |
|--------|---------|---------------|----------|
| `astype()` | General type casting | Raises error on failure | Clean data, known types |
| `pd.to_numeric()` | Convert to numeric | `errors='coerce'` → NaN | Dirty data with mixed types |
| `pd.to_datetime()` | Convert to datetime | `errors='coerce'` → NaT | Date strings in various formats |

### `astype()` — General Type Casting

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'a': ['1', '2', '3'], 'b': [1.5, 2.5, 3.5]})

# Basic conversions
df['a'] = df['a'].astype(int)           # str → int
df['b'] = df['b'].astype(np.float32)    # float64 → float32
df['a'] = df['a'].astype('category')    # to categorical
df['a'] = df['a'].astype(str)           # to string

# Multiple columns at once
df = df.astype({'a': 'int32', 'b': 'float32'})

# FAILS on invalid data:
# pd.Series(['1', '2', 'abc']).astype(int)  # ValueError!
```

### `pd.to_numeric()` — Safe Numeric Conversion

```python
s = pd.Series(['1', '2.5', 'abc', '4', None])

# errors='coerce': invalid → NaN (default would raise)
result = pd.to_numeric(s, errors='coerce')
# [1.0, 2.5, NaN, 4.0, NaN]

# errors='ignore': return original on failure
result = pd.to_numeric(s, errors='ignore')
# ['1', '2.5', 'abc', '4', None]  — unchanged

# downcast: reduce memory
result = pd.to_numeric(s, errors='coerce', downcast='integer')
# Uses int8 instead of int64 when possible

# Apply to multiple columns
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
```

### `pd.to_datetime()` — DateTime Conversion

```python
dates = pd.Series(['2025-01-15', 'Jan 20, 2025', '15/01/2025', 'invalid'])

# Auto-detect format
pd.to_datetime(dates, errors='coerce')
# [2025-01-15, 2025-01-20, 2025-01-15, NaT]

# Specify format (faster for large datasets)
pd.to_datetime(dates, format='%Y-%m-%d', errors='coerce')

# Unix timestamps
pd.to_datetime([1706000000], unit='s')  # seconds since epoch

# From multiple columns
df = pd.DataFrame({'year': [2025], 'month': [1], 'day': [15]})
pd.to_datetime(df)  # combines into Timestamp

# infer_datetime_format for speed
pd.to_datetime(dates, infer_datetime_format=True)  # auto-detect, cached
```

### When to Use What
```
Clean data, known type    →  astype() (fastest)
Dirty data with errors    →  pd.to_numeric(errors='coerce')
Date parsing              →  pd.to_datetime(errors='coerce')
Memory optimization       →  astype() with specific dtypes (int32, float32)
```

> **Interview Tip:** `errors='coerce'` is essential for real-world data — it converts invalid values to NaN/NaT instead of crashing. After coercion, always check `df.isna().sum()` to see how many values failed conversion. For ML pipelines, use `pd.to_numeric` in preprocessing to handle dirty CSV data gracefully.

---

## Question 23
- [ ] Done

**Discuss how to deal with time series data in Pandas**

### Time Series in Pandas

Pandas has first-class support for time series data with specialized types: `Timestamp`, `DatetimeIndex`, `Period`, and `Timedelta`.

### Creating Time Series

```python
import pandas as pd
import numpy as np

# Create datetime index
dates = pd.date_range(start='2025-01-01', periods=365, freq='D')
ts = pd.Series(np.random.randn(365).cumsum(), index=dates, name='price')

# From CSV with datetime parsing
df = pd.read_csv('sales.csv', parse_dates=['date'], index_col='date')
```

### Key Operations

```python
# 1. Resampling (change frequency)
monthly = ts.resample('M').mean()      # daily → monthly averages
weekly = ts.resample('W').sum()        # daily → weekly sums
quarterly = ts.resample('Q').last()    # last value per quarter

# 2. Rolling windows
ts.rolling(window=7).mean()            # 7-day moving average
ts.rolling(window=30).std()            # 30-day rolling std
ts.ewm(span=20).mean()                 # exponential moving average

# 3. Shifting and lagging
ts.shift(1)                            # lag by 1 period
ts.shift(-1)                           # lead by 1 period
ts.diff()                              # first difference (returns - previous)
ts.pct_change()                        # percentage change

# 4. Date/time components  
df['year'] = df.index.year
df['month'] = df.index.month
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = df.index.dayofweek >= 5
df['quarter'] = df.index.quarter

# 5. Slicing by date
ts['2025-03']                          # all of March 2025
ts['2025-01-01':'2025-06-30']          # date range slice
ts.last('3M')                          # last 3 months
ts.first('1W')                         # first week

# 6. Timezone handling
ts_utc = ts.tz_localize('UTC')
ts_est = ts_utc.tz_convert('US/Eastern')
```

### Time Series Features for ML

```python
def create_ts_features(df, target_col='value'):
    """Create time series features for ML."""
    df['lag_1'] = df[target_col].shift(1)
    df['lag_7'] = df[target_col].shift(7)
    df['rolling_mean_7'] = df[target_col].rolling(7).mean()
    df['rolling_std_7'] = df[target_col].rolling(7).std()
    df['ewm_mean_7'] = df[target_col].ewm(span=7).mean()
    df['diff_1'] = df[target_col].diff()
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['is_month_start'] = df.index.is_month_start.astype(int)
    return df.dropna()
```

> **Interview Tip:** For ML on time series, always use **chronological splits** (not random). Key features: lags, rolling statistics, calendar features. Mention `pandas.tseries.offsets` for business day logic and `pd.infer_freq()` to detect frequency automatically.

---

## Question 24
- [ ] Done

**Explain the different types of data ranking available in Pandas**

### Data Ranking in Pandas

The `rank()` method assigns ranks to values. The `method` parameter controls how ties are handled.

### Ranking Methods

| Method | Description | Example: [10, 20, 20, 30] |
|--------|-------------|---------------------------|
| `'average'` | Mean rank for ties (default) | [1.0, 2.5, 2.5, 4.0] |
| `'min'` | Lowest rank for ties | [1, 2, 2, 4] |
| `'max'` | Highest rank for ties | [1, 3, 3, 4] |
| `'first'` | Rank by order of appearance | [1, 2, 3, 4] |
| `'dense'` | Like 'min' but no gaps | [1, 2, 2, 3] |

### Examples

```python
import pandas as pd

df = pd.DataFrame({
    'student': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'score': [85, 92, 92, 78, 95]
})

# Different ranking methods
df['rank_avg'] = df['score'].rank(method='average', ascending=False)
df['rank_min'] = df['score'].rank(method='min', ascending=False)
df['rank_dense'] = df['score'].rank(method='dense', ascending=False)
df['rank_first'] = df['score'].rank(method='first', ascending=False)

print(df)
#   student  score  rank_avg  rank_min  rank_dense  rank_first
# 0   Alice     85       4.0         4           3           4
# 1     Bob     92       2.5         2           2           2
# 2 Charlie     92       2.5         2           2           3
# 3   Diana     78       5.0         5           4           5
# 4     Eve     95       1.0         1           1           1

# Percentile rank (0 to 1)
df['percentile'] = df['score'].rank(pct=True)

# Rank within groups
df['dept'] = ['Math', 'Math', 'Science', 'Science', 'Math']
df['dept_rank'] = df.groupby('dept')['score'].rank(ascending=False)
```

### ML Use Cases
```python
# Rank-based features for recommendation systems
df['popularity_rank'] = df['views'].rank(ascending=False, method='dense')
df['popularity_percentile'] = df['views'].rank(pct=True)

# Rank transformation (robustness to outliers)
from sklearn.preprocessing import QuantileTransformer
# Similar to rank(pct=True) but maps to normal/uniform distribution
```

> **Interview Tip:** `dense` ranking is most common for leaderboards (no gaps: 1, 2, 2, 3). `first` is useful when you need unique ranks. Rank-based features are robust to outliers and useful in **learning-to-rank** problems (search engines, recommendation systems).

---

## Question 25
- [ ] Done

**What is a crosstab in Pandas , and when would you use it?**

### Crosstab in Pandas

`pd.crosstab()` computes a **cross-tabulation** (frequency table) of two or more factors, showing how variables relate to each other.

### Basic Usage

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
    'department': ['Sales', 'Sales', 'Eng', 'Eng', 'Sales', 'Eng', 'HR', 'HR'],
    'satisfied': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No']
})

# Simple frequency table
ct = pd.crosstab(df['gender'], df['department'])
print(ct)
# department  Eng  HR  Sales
# gender
# F            2   1     1
# M            1   1     2

# With margins (totals)
ct = pd.crosstab(df['gender'], df['department'], margins=True, margins_name='Total')

# Proportions instead of counts
ct = pd.crosstab(df['gender'], df['department'], normalize='all')    # of total
ct = pd.crosstab(df['gender'], df['department'], normalize='index')  # row percentages
ct = pd.crosstab(df['gender'], df['department'], normalize='columns')  # column percentages
```

### Advanced Crosstab

```python
# Multiple indices
ct = pd.crosstab(
    [df['gender'], df['satisfied']],  # rows
    df['department']                   # columns
)

# Custom aggregation
df['salary'] = [50, 60, 70, 80, 55, 75, 65, 45]
ct = pd.crosstab(
    df['gender'], df['department'],
    values=df['salary'], aggfunc='mean'  # average salary by gender x department
)

# Heatmap visualization
import matplotlib.pyplot as plt
import seaborn as sns
ct = pd.crosstab(df['gender'], df['department'])
sns.heatmap(ct, annot=True, cmap='Blues', fmt='d')
plt.title('Employee Distribution')
plt.show()
```

### Crosstab vs. Pivot Table
| Feature | `pd.crosstab()` | `pd.pivot_table()` |
|---------|-----------------|--------------------|
| Input | Series/arrays | DataFrame |
| Default | Counts | Mean |
| Margins | `margins=True` | `margins=True` |
| Best for | Frequency analysis | Aggregation |

### ML Use Cases
- **Confusion matrices**: `pd.crosstab(y_true, y_pred)`
- **Feature analysis**: Check class distribution across categories
- **Chi-squared test**: Crosstab feeds `scipy.stats.chi2_contingency()`

> **Interview Tip:** `pd.crosstab(y_true, y_pred)` is a quick way to create a confusion matrix. Use `normalize='index'` to see **recall per class** and `normalize='columns'` for **precision per class**.

---

## Question 26
- [ ] Done

**Describe how to perform a multi-index query on a DataFrame**

### Multi-Index (Hierarchical Index) in Pandas

A MultiIndex allows multiple levels of indexing, enabling efficient storage and querying of higher-dimensional data in 2D DataFrames.

### Creating MultiIndex DataFrames

```python
import pandas as pd
import numpy as np

# Method 1: set_index with multiple columns
df = pd.DataFrame({
    'department': ['Sales', 'Sales', 'Eng', 'Eng', 'HR', 'HR'],
    'year': [2024, 2025, 2024, 2025, 2024, 2025],
    'revenue': [100, 120, 200, 250, 50, 55],
    'headcount': [10, 12, 20, 22, 5, 6]
})
df_mi = df.set_index(['department', 'year'])

# Method 2: From arrays
arrays = [['Sales', 'Sales', 'Eng', 'Eng'], [2024, 2025, 2024, 2025]]
index = pd.MultiIndex.from_arrays(arrays, names=['dept', 'year'])
```

### Querying MultiIndex DataFrames

```python
# 1. Select by top-level index
df_mi.loc['Sales']                     # all Sales rows
df_mi.loc[['Sales', 'Eng']]           # multiple departments

# 2. Select by both levels
df_mi.loc[('Sales', 2025)]             # specific row
df_mi.loc[('Sales', 2025), 'revenue']  # specific cell

# 3. Cross-section with xs()
df_mi.xs(2025, level='year')            # all departments for 2025
df_mi.xs('Sales', level='department')   # Sales for all years

# 4. Slice within levels (must sort index first)
df_mi = df_mi.sort_index()
df_mi.loc[('Eng', 2024):('Sales', 2025)]   # range slice

# 5. IndexSlice for complex slicing
idx = pd.IndexSlice
df_mi.loc[idx[:, 2025], :]              # all departments, year 2025
df_mi.loc[idx['Sales':'Sales', :], :]   # Sales, all years

# 6. Query with .query() (reset index first or use level names)
df_mi.query('department == "Sales" and year == 2025')

# 7. Boolean filtering on index
df_mi[df_mi.index.get_level_values('year') == 2025]
df_mi[df_mi.index.get_level_values('department').isin(['Sales', 'HR'])]
```

### MultiIndex Operations

```python
# Group operations across levels
df_mi.groupby(level='department').sum()
df_mi.groupby(level='year').mean()

# Unstacking (pivot multi-index to columns)
df_mi['revenue'].unstack('year')
# department  2024  2025
# Eng          200   250
# HR            50    55
# Sales        100   120

# Reset back to flat DataFrame
df_flat = df_mi.reset_index()

# Swap levels
df_mi.swaplevel().sort_index()
```

> **Interview Tip:** MultiIndex is useful for **panel data** (e.g., multiple stocks over time) and **pivot tables**. Use `xs()` for clean cross-section queries and `IndexSlice` for complex slicing. For most ML workflows, a flat DataFrame (`reset_index()`) is preferred since sklearn expects 2D arrays.

---

## Question 27
- [ ] Done

**Provide an example of how to normalize data within a DataFrame column**

### Data Normalization in Pandas

Normalization scales features to a standard range, which is critical for models sensitive to feature magnitudes (e.g., linear regression, KNN, SVM, neural networks).

### Common Methods

| Method | Formula | Range | Best For |
|--------|---------|-------|----------|
| **Min-Max** | $(x - min) / (max - min)$ | [0, 1] | Bounded features, neural networks |
| **Z-Score** | $(x - \mu) / \sigma$ | Unbounded | Gaussian-like data |
| **Max Abs** | $x / |max|$ | [-1, 1] | Sparse data |
| **Robust** | $(x - median) / IQR$ | Unbounded | Data with outliers |
| **Log** | $\log(1 + x)$ | Unbounded | Right-skewed data |

### Implementation

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 100],     # has outlier
    'income': [30000, 50000, 70000, 90000, 120000],
    'score': [0.5, 0.8, 0.3, 0.9, 0.1]
})

# 1. Min-Max Normalization (scale to [0, 1])
df['age_minmax'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())

# 2. Z-Score Standardization
df['income_zscore'] = (df['income'] - df['income'].mean()) / df['income'].std()

# 3. Robust Scaling (outlier-resistant)
median = df['age'].median()
q1, q3 = df['age'].quantile([0.25, 0.75])
df['age_robust'] = (df['age'] - median) / (q3 - q1)

# 4. Log Transformation
df['income_log'] = np.log1p(df['income'])  # log(1 + x)

# 5. Using sklearn (recommended for ML)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

scaler = MinMaxScaler()
df[['age_scaled', 'income_scaled']] = scaler.fit_transform(df[['age', 'income']])

# IMPORTANT: fit on train, transform on test
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # use SAME scaler
```

### When to Normalize
```
Tree-based models (RF, XGBoost)  →  NOT needed (scale-invariant)
Linear models, SVM, KNN          →  REQUIRED
Neural networks                  →  REQUIRED (usually Min-Max or Z-Score)
Clustering (K-Means)             →  REQUIRED
```

> **Interview Tip:** Always **fit scaler on training data only** and use the same parameters to transform test data — otherwise you have data leakage. Use `RobustScaler` when outliers are present. `StandardScaler` is the most common default choice.

---

## Question 28
- [ ] Done

**Show how to create simple plots from a DataFrame using Pandas ’ visualization tools**

### Plotting with Pandas

Pandas provides a convenient `.plot()` method that wraps Matplotlib for quick data visualization.

### Basic Plot Types

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.DataFrame({
    'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'sales': [200, 250, 300, 280, 350, 400],
    'profit': [50, 60, 80, 70, 100, 120]
})

# Line plot (default)
df.plot(x='month', y=['sales', 'profit'], title='Monthly Performance')
plt.ylabel('Amount')
plt.show()

# Bar chart
df.plot(x='month', y='sales', kind='bar', color='steelblue')

# Horizontal bar
df.plot(x='month', y='sales', kind='barh')

# Scatter plot
df.plot(kind='scatter', x='sales', y='profit', title='Sales vs Profit')

# Histogram
df['sales'].plot(kind='hist', bins=5, edgecolor='black')

# Box plot
df[['sales', 'profit']].plot(kind='box')

# Pie chart
df.set_index('month')['sales'].plot(kind='pie', autopct='%1.1f%%')

# Area chart
df.plot(x='month', y=['sales', 'profit'], kind='area', alpha=0.5)
```

### Customization

```python
# Pandas plot returns a Matplotlib Axes object
ax = df.plot(x='month', y='sales', kind='bar', figsize=(10, 6),
             color='coral', edgecolor='black', alpha=0.8)
ax.set_title('Monthly Sales', fontsize=16)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Sales ($)', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('sales_chart.png', dpi=150)
plt.show()

# Subplots
df.plot(subplots=True, layout=(2, 1), figsize=(10, 8), sharex=True)
```

### Key Parameters
| Parameter | Description |
|-----------|-------------|
| `kind` | Plot type: line, bar, barh, scatter, hist, box, pie, area |
| `figsize` | Figure size as (width, height) |
| `title` | Plot title |
| `color` | Bar/line color |
| `alpha` | Transparency (0-1) |
| `subplots` | Create separate subplot per column |
| `grid` | Show grid lines |

> **Interview Tip:** Pandas plotting is ideal for **quick EDA** during data exploration. For production-quality plots, switch to Matplotlib (full control) or Seaborn (statistical plots). The key advantage is that `df.plot()` understands DataFrame structure (column names become labels automatically).

---

## Question 29
- [ ] Done

**Discuss how Pandas integrates with Matplotlib and Seaborn for data visualization**

### Pandas + Matplotlib + Seaborn Integration

Pandas has built-in plotting that wraps Matplotlib, and Seaborn is designed to work directly with Pandas DataFrames.

### Pandas Built-in Plotting (wraps Matplotlib)

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.DataFrame({
    'month': pd.date_range('2025-01', periods=12, freq='M'),
    'sales': np.random.randint(100, 500, 12),
    'expenses': np.random.randint(50, 300, 12),
    'category': np.random.choice(['A', 'B', 'C'], 12)
})

# Pandas .plot() uses Matplotlib under the hood
df.plot(x='month', y=['sales', 'expenses'], kind='line', figsize=(10, 6))
plt.title('Monthly Sales vs Expenses')
plt.ylabel('Amount ($)')
plt.show()

# All Pandas plot types
df['sales'].plot(kind='hist', bins=10)        # histogram
df.plot(kind='scatter', x='sales', y='expenses')  # scatter
df['category'].value_counts().plot(kind='bar')      # bar chart
df.plot(kind='box')                                  # boxplot
df.plot(kind='area', stacked=True)                   # area chart
```

### Matplotlib Direct Integration

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

df['sales'].plot(ax=axes[0, 0], kind='hist', title='Sales Distribution')
df.plot(ax=axes[0, 1], x='sales', y='expenses', kind='scatter')
df.groupby('category')['sales'].mean().plot(ax=axes[1, 0], kind='bar')
df.set_index('month')['sales'].plot(ax=axes[1, 1], title='Sales Trend')

plt.tight_layout()
plt.show()
```

### Seaborn Integration (designed for DataFrames)

```python
import seaborn as sns

# Seaborn accepts DataFrames directly with column names
sns.scatterplot(data=df, x='sales', y='expenses', hue='category', size='sales')
sns.boxplot(data=df, x='category', y='sales')
sns.heatmap(df[['sales', 'expenses']].corr(), annot=True, cmap='coolwarm')
sns.pairplot(df[['sales', 'expenses', 'category']], hue='category')
sns.histplot(data=df, x='sales', hue='category', kde=True)

# Statistical plots
sns.violinplot(data=df, x='category', y='sales')
sns.jointplot(data=df, x='sales', y='expenses', kind='hex')
```

### When to Use Which
| Tool | Best For | Complexity |
|------|----------|------------|
| `df.plot()` | Quick exploration | Low |
| Matplotlib | Full customization | High |
| Seaborn | Statistical visualization | Medium |

> **Interview Tip:** Use `df.plot()` for quick EDA, Seaborn for presentation-quality statistical plots, and Matplotlib for custom/complex layouts. All three interoperate — Seaborn plots can be customized with Matplotlib commands (`plt.title()`, `plt.xlabel()`).

---

## Question 30
- [ ] Done

**Explain how you would export a DataFrame to different file formats for reporting purposes**

### Exporting DataFrames to Various Formats

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'score': [95, 87, 92],
    'grade': ['A', 'B+', 'A-']
})

# ---- CSV ----
df.to_csv('report.csv', index=False)                       # standard CSV
df.to_csv('report.tsv', sep='\t', index=False)             # tab-separated
df.to_csv('report.csv', columns=['name', 'grade'])         # select columns
df.to_csv('report.csv.gz', compression='gzip')             # compressed

# ---- Excel ----
df.to_excel('report.xlsx', sheet_name='Results', index=False)

# Multiple sheets
with pd.ExcelWriter('report_multi.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Summary', index=False)
    df.describe().to_excel(writer, sheet_name='Statistics')

# ---- JSON ----
df.to_json('report.json', orient='records', indent=2)      # array of objects
df.to_json('report.json', orient='columns')                 # column-oriented

# ---- HTML ----
html = df.to_html(index=False, classes='table table-striped')
with open('report.html', 'w') as f:
    f.write(f'<html><body>{html}</body></html>')

# ---- Parquet (efficient for large data) ----
df.to_parquet('report.parquet', engine='pyarrow')           # columnar format

# ---- SQL Database ----
from sqlalchemy import create_engine
engine = create_engine('sqlite:///reports.db')
df.to_sql('results', engine, if_exists='replace', index=False)

# ---- LaTeX (academic papers) ----
latex = df.to_latex(index=False, caption='Student Results')

# ---- Markdown ----
md = df.to_markdown(index=False)

# ---- Clipboard (paste into Excel/Google Sheets) ----
df.to_clipboard(index=False)

# ---- Pickle (Python-specific, preserves dtypes) ----
df.to_pickle('report.pkl')
```

### Format Comparison
| Format | Size | Speed | Human Readable | Schema |
|--------|------|-------|---------------|--------|
| CSV | Large | Slow | Yes | No |
| Parquet | Small | Fast | No | Yes |
| Excel | Medium | Slow | Yes | Partial |
| JSON | Large | Medium | Yes | No |
| Pickle | Medium | Fast | No | Yes |
| SQL | Medium | Medium | Via queries | Yes |

> **Interview Tip:** Use **CSV/Excel** for business stakeholders, **Parquet** for data engineering pipelines (columnar, compressed, schema-aware), **SQL** for shared database access, and **Pickle** for quick Python-to-Python serialization. Parquet preserves dtypes and is 5-10x smaller than CSV.

---

## Question 31
- [ ] Done

**How does one use Dask or Modin to handle larger-than-memory data in Pandas ?**

### Dask and Modin: Scaling Pandas

Both Dask and Modin extend Pandas to handle datasets that exceed available RAM.

### Dask: Distributed Computing

```python
import dask.dataframe as dd

# Load large file (not loaded into memory at once)
ddf = dd.read_csv('huge_file_*.csv')  # supports glob patterns
ddf = dd.read_parquet('huge_data/')   # Parquet directory

# Dask API mirrors Pandas
result = ddf.groupby('category')['amount'].mean()  # lazy computation
result = result.compute()                           # triggers execution

# Key Dask features
ddf.head()                          # peek at first rows
ddf.npartitions                     # number of chunks
ddf.describe().compute()            # statistics

# Filter and aggregate
filtered = ddf[ddf['amount'] > 1000]
agg = filtered.groupby('region').agg({'amount': ['sum', 'mean', 'count']})
result = agg.compute()  # only now does it read and process

# Save results
agg.to_parquet('output/')           # save as Parquet
agg.to_csv('output_*.csv')          # save as multiple CSVs

# Dask with distributed scheduler (multi-machine)
from dask.distributed import Client
client = Client('scheduler-address:8786')  # connect to cluster
```

### Modin: Drop-in Replacement

```python
# Simply change the import!
import modin.pandas as pd  # instead of: import pandas as pd

# Everything works the same, but uses all CPU cores
df = pd.read_csv('large_file.csv')   # parallelized I/O
result = df.groupby('col').mean()     # parallelized computation
df_filtered = df[df['x'] > 100]      # parallelized filtering
```

### Comparison

| Feature | Pandas | Modin | Dask |
|---------|--------|-------|------|
| **API changes** | N/A | None (drop-in) | Minor differences |
| **Max data size** | RAM | RAM | Larger than RAM |
| **Parallelism** | Single core | All cores | Cluster-scale |
| **Lazy evaluation** | No | No | Yes |
| **Best for** | Small-medium data | Medium data, quick speedup | Large data, complex pipelines |

### When to Use Which
```
Data fits in RAM (< 10 GB)     →  Pandas
Need quick speedup, same API   →  Modin (just change import)
Data > RAM, complex pipelines  →  Dask
Data > 100 GB, distributed     →  Spark (PySpark)
```

> **Interview Tip:** Modin is the easiest win — change one import line for 2-4x speedup on multi-core machines. Dask is more powerful (handles larger-than-memory data) but requires understanding lazy evaluation and `.compute()`. For production ML pipelines at scale, PySpark is the industry standard.

---

## Question 32
- [ ] Done

**Discuss the advantages of vectorized operations in Pandas over iteration**

### Vectorized Operations vs. Iteration

Vectorized operations apply a function to an entire array/Series at once, using optimized C/NumPy code instead of Python loops.

### Performance Comparison

```python
import pandas as pd
import numpy as np
import time

n = 1_000_000
df = pd.DataFrame({
    'a': np.random.randn(n),
    'b': np.random.randn(n)
})

# Method 1: Python loop (SLOW)
start = time.time()
result = []
for i in range(len(df)):
    result.append(df.iloc[i]['a'] + df.iloc[i]['b'])
df['c_loop'] = result
print(f"Loop: {time.time()-start:.2f}s")

# Method 2: iterrows (SLOW)
start = time.time()
for idx, row in df.iterrows():
    df.at[idx, 'c_iterrows'] = row['a'] + row['b']
print(f"iterrows: {time.time()-start:.2f}s")

# Method 3: apply (MEDIUM)
start = time.time()
df['c_apply'] = df.apply(lambda row: row['a'] + row['b'], axis=1)
print(f"apply: {time.time()-start:.2f}s")

# Method 4: Vectorized (FAST) ← USE THIS
start = time.time()
df['c_vec'] = df['a'] + df['b']
print(f"Vectorized: {time.time()-start:.2f}s")

# Typical speedups for 1M rows:
# Loop:       ~30s
# iterrows:   ~15s
# apply:      ~3s
# Vectorized: ~0.005s  (1000x faster!)
```

### Vectorized Alternatives for Common Operations

| Instead of | Use |
|-----------|-----|
| `for` loop with iloc | Direct column arithmetic: `df['a'] + df['b']` |
| `apply(lambda x: ...)` | `np.where()`, `np.select()` |
| `iterrows()` | `.str` accessor, `.dt` accessor |
| Conditional loop | `df.loc[condition, 'col'] = value` |
| Custom aggregation loop | `groupby().agg()` |

```python
# Vectorized string operations
df['name_upper'] = df['name'].str.upper()       # not: apply(str.upper)
df['name_len'] = df['name'].str.len()            # not: apply(len)

# Vectorized datetime operations
df['year'] = df['date'].dt.year                  # not: apply(lambda x: x.year)
df['month'] = df['date'].dt.month

# Vectorized conditional
df['category'] = np.where(df['value'] > 0, 'positive', 'negative')
```

### Why Vectorization Is Faster
1. **No Python loop overhead**: Operations run in C, not interpreted Python
2. **SIMD instructions**: CPU processes multiple values simultaneously
3. **Memory locality**: Contiguous NumPy arrays are cache-friendly
4. **No type checking**: Homogeneous dtypes skip per-element checks

> **Interview Tip:** The rule of thumb: if you're using `iterrows()` or `apply()`, there's probably a vectorized alternative that's 100-1000x faster. The only exception is when you need complex row-wise logic that can't be expressed with NumPy — even then, consider Cython or `numba.jit` before `apply()`.

---

## Question 33
- [ ] Done

**Explain the importance of using categorical data types , especially when working with a large number of unique values**

### Categorical Data Type in Pandas

The `category` dtype stores repeating string values as **integer codes** with a lookup table, dramatically reducing memory and improving performance.

### Memory Savings

```python
import pandas as pd
import numpy as np

# Create DataFrame with repeating string values
n = 1_000_000
df = pd.DataFrame({
    'country': np.random.choice(['USA', 'UK', 'Germany', 'France', 'Japan'], n),
    'status': np.random.choice(['active', 'inactive', 'pending'], n)
})

# Before: object dtype (strings)
print(f"Object memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
# ~64 MB

# After: category dtype
df['country'] = df['country'].astype('category')
df['status'] = df['status'].astype('category')
print(f"Category memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
# ~2 MB  (32x reduction!)
```

### How It Works Internally
```
Object dtype:   ['USA', 'UK', 'USA', 'France', 'USA', 'UK', ...]
                 Each string stored separately = lots of memory

Category dtype: codes:      [0,    1,   0,      2,      0,   1, ...]
                categories: ['France', 'UK', 'USA']
                Just integers + small lookup table!
```

### Performance Benefits

```python
import time

# GroupBy is much faster with categories
start = time.time()
df.groupby('country')['status'].count()  # category: ~0.01s vs object: ~0.1s
print(f"GroupBy: {time.time()-start:.3f}s")

# Sorting is faster
df.sort_values('country')  # sorts integer codes, not strings

# Merge/Join is faster
df.merge(other_df, on='country')  # integer comparison vs string comparison
```

### Ordered Categories (Ordinal Data)

```python
# Define ordered categories
size_type = pd.CategoricalDtype(
    categories=['XS', 'S', 'M', 'L', 'XL'], 
    ordered=True
)
df['size'] = df['size'].astype(size_type)

# Now comparisons work correctly
df[df['size'] > 'M']       # returns L and XL
df['size'].min()            # returns 'XS'
df.sort_values('size')      # sorts by defined order, not alphabetical
```

### When to Use Categories

| Scenario | Use Category? | Reason |
|----------|--------------|--------|
| Few unique values, many rows | **Yes** | Huge memory savings |
| High cardinality (>50% unique) | **No** | Overhead exceeds savings |
| Ordinal data | **Yes** | Enables correct ordering |
| Frequently grouped/sorted column | **Yes** | Performance improvement |
| Columns used in ML encoding | **Yes** | Tracks valid categories |

```python
# Automatic conversion for all low-cardinality columns
for col in df.select_dtypes(include='object').columns:
    if df[col].nunique() / len(df) < 0.5:  # less than 50% unique
        df[col] = df[col].astype('category')
```

> **Interview Tip:** Categories give **10-50x memory reduction** for columns with repeating strings. They also prevent invalid values — assigning a value not in the category list raises an error. For ML, `pd.CategoricalDtype` ensures train and test data have the same categories, avoiding encoding mismatches.

---
