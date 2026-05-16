# Pandas Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Write a Pandas script to filter rows in a DataFrame based on a column’s value being higher than a specified percentile**

### Solution
```python
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
df = pd.DataFrame({
    'id': range(1, 101),
    'value': np.random.exponential(scale=100, size=100)
})

def filter_above_percentile(df, column, percentile):
    """Filter rows where column value is above the specified percentile."""
    threshold = df[column].quantile(percentile / 100)
    return df[df[column] > threshold]

# Filter values above 90th percentile
result = filter_above_percentile(df, 'value', 90)

print(f"Original rows: {len(df)}")
print(f"Rows above 90th percentile: {len(result)}")
print(f"Threshold value: {df['value'].quantile(0.90):.2f}")
```

---

## Question 2
- [ ] Done

**Code a function that concatenates two DataFrames and handles overlapping indices correctly.**

### Solution
```python
import pandas as pd

def concatenate_dataframes(df1, df2, reset_index=True):
    """
    Concatenate two DataFrames, handling overlapping indices.
    
    Parameters:
    - reset_index: If True, create new sequential index
                   If False, keep original indices (may have duplicates)
    """
    if reset_index:
        # Reset indices to avoid duplicates
        result = pd.concat([df1, df2], ignore_index=True)
    else:
        # Keep original indices, verify_integrity will raise if duplicates
        try:
            result = pd.concat([df1, df2], verify_integrity=True)
        except ValueError:
            # If duplicates exist, add suffix to distinguish
            df2_copy = df2.copy()
            df2_copy.index = [f"{idx}_2" for idx in df2.index]
            result = pd.concat([df1, df2_copy])
    
    return result

# Example
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=[0, 1])
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]}, index=[0, 1])  # Overlapping indices

result = concatenate_dataframes(df1, df2, reset_index=True)
print(result)
```

---

## Question 3
- [ ] Done

**Implement a data cleaning function that drops columns with more than 50% missing values and fills the remaining ones with column mean.**

### Solution
```python
import pandas as pd
import numpy as np

def clean_dataframe(df, missing_threshold=0.5, fill_strategy='mean'):
    """
    Clean DataFrame by:
    1. Dropping columns with > threshold missing values
    2. Filling remaining missing values with column statistic
    """
    df_clean = df.copy()
    
    # Calculate missing percentage per column
    missing_pct = df_clean.isnull().sum() / len(df_clean)
    
    # Drop columns exceeding threshold
    cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
    df_clean = df_clean.drop(columns=cols_to_drop)
    
    print(f"Dropped columns ({len(cols_to_drop)}): {cols_to_drop}")
    
    # Fill remaining missing values
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            if df_clean[col].dtype in ['float64', 'int64']:
                if fill_strategy == 'mean':
                    fill_value = df_clean[col].mean()
                elif fill_strategy == 'median':
                    fill_value = df_clean[col].median()
                df_clean[col].fillna(fill_value, inplace=True)
            else:
                # For non-numeric, fill with mode
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    return df_clean

# Example
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, np.nan, np.nan, np.nan, 5],  # >50% missing
    'C': [1, np.nan, 3, 4, 5],
    'D': ['x', 'y', np.nan, 'x', 'y']
})

cleaned = clean_dataframe(df, missing_threshold=0.5)
print(cleaned)
```

---

## Question 4
- [ ] Done

**Create a Pandas pipeline that ingests, processes, and summarizes time-series data from a CSV file.**

### Solution
```python
import pandas as pd
import numpy as np

def process_timeseries_pipeline(filepath):
    """
    Pipeline for time-series data processing:
    1. Read data
    2. Parse dates and set index
    3. Handle missing values
    4. Resample and aggregate
    5. Calculate rolling statistics
    """
    # 1. Read data
    df = pd.read_csv(filepath, parse_dates=['date'])
    
    # 2. Set datetime index
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    # 3. Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # 4. Resample to monthly and aggregate
    monthly = df.resample('M').agg({
        'value': ['sum', 'mean', 'std', 'count']
    })
    monthly.columns = ['total', 'average', 'std', 'count']
    
    # 5. Calculate rolling statistics
    monthly['rolling_avg_3m'] = monthly['average'].rolling(window=3).mean()
    monthly['pct_change'] = monthly['total'].pct_change()
    
    # Summary statistics
    summary = {
        'start_date': df.index.min(),
        'end_date': df.index.max(),
        'total_records': len(df),
        'total_value': df['value'].sum(),
        'average_value': df['value'].mean(),
        'max_month': monthly['total'].idxmax(),
        'min_month': monthly['total'].idxmin()
    }
    
    return monthly, summary

# Example usage (with synthetic data)
# Create sample data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
df = pd.DataFrame({
    'date': dates,
    'value': np.random.randint(100, 1000, 365)
})
df.to_csv('timeseries.csv', index=False)

monthly, summary = process_timeseries_pipeline('timeseries.csv')
print(monthly.head())
print(summary)
```

---

## Question 5
- [ ] Done

**Write a Python function that takes a DataFrame and computes the correlation matrix , then visualizes it using Seaborn’s heatmap**

### Solution
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_matrix(df, method='pearson', figsize=(10, 8)):
    """
    Compute correlation matrix and visualize as heatmap.
    
    Parameters:
    - method: 'pearson', 'spearman', or 'kendall'
    - figsize: tuple for figure size
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr(method=method)
    
    # Create mask for upper triangle (optional)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5)
    
    plt.title(f'Correlation Matrix ({method.capitalize()})')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

# Example
df = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'feature3': np.random.randn(100) * 2,
    'target': np.random.randn(100)
})
df['feature4'] = df['feature1'] * 0.8 + np.random.randn(100) * 0.2  # Correlated

corr = plot_correlation_matrix(df)
```

---

## Question 6
- [ ] Done

**Given a DataFrame with multiple datetime columns, create a new column with the earliest datetime.**

### Solution
```python
import pandas as pd
import numpy as np

def get_earliest_datetime(df, datetime_columns):
    """
    Create a new column with the earliest datetime from multiple columns.
    Handles NaT (missing) values.
    """
    # Convert columns to datetime if not already
    for col in datetime_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Method 1: Using min() across columns
    df['earliest'] = df[datetime_columns].min(axis=1)
    
    return df

# Example
df = pd.DataFrame({
    'created_date': ['2023-01-15', '2023-02-01', '2023-03-10'],
    'modified_date': ['2023-01-20', '2023-01-25', np.nan],
    'review_date': ['2023-01-10', '2023-02-15', '2023-03-05']
})

result = get_earliest_datetime(df, ['created_date', 'modified_date', 'review_date'])
print(result)
```

### Alternative: Row-wise Apply
```python
def earliest_date_apply(row, cols):
    dates = row[cols].dropna()
    return dates.min() if len(dates) > 0 else pd.NaT

df['earliest'] = df.apply(lambda row: earliest_date_apply(row, datetime_columns), axis=1)
```

---

## Question 7
- [ ] Done

**Develop a routine in Pandas to detect and flag rows that deviate by more than three standard deviations from the mean of specific columns**

### Solution
```python
import pandas as pd
import numpy as np

def detect_outliers(df, columns, n_std=3):
    """
    Detect outliers using z-score method.
    Flags rows where any specified column deviates > n_std from mean.
    
    Returns:
    - df with 'is_outlier' column
    - outlier details
    """
    df_result = df.copy()
    
    # Calculate z-scores for each column
    outlier_flags = pd.DataFrame(index=df.index)
    
    for col in columns:
        if df[col].dtype in ['float64', 'int64']:
            mean = df[col].mean()
            std = df[col].std()
            z_scores = (df[col] - mean) / std
            outlier_flags[f'{col}_zscore'] = z_scores
            outlier_flags[f'{col}_outlier'] = np.abs(z_scores) > n_std
    
    # Flag row as outlier if ANY column is an outlier
    outlier_cols = [col for col in outlier_flags.columns if col.endswith('_outlier')]
    df_result['is_outlier'] = outlier_flags[outlier_cols].any(axis=1)
    
    # Summary
    outlier_count = df_result['is_outlier'].sum()
    print(f"Detected {outlier_count} outliers ({outlier_count/len(df)*100:.2f}%)")
    
    return df_result, outlier_flags

# Example
np.random.seed(42)
df = pd.DataFrame({
    'value1': np.random.randn(100),
    'value2': np.random.randn(100)
})
# Add some outliers
df.loc[0, 'value1'] = 10  # Outlier
df.loc[1, 'value2'] = -8  # Outlier

result, details = detect_outliers(df, ['value1', 'value2'], n_std=3)
print(result[result['is_outlier']])
```

---

## Question 8
- [ ] Done

**Outline how to merge multiple time series datasets effectively in Pandas , ensuring correct alignment and handling missing values**

### Solution
```python
import pandas as pd
import numpy as np

def merge_timeseries(dataframes, date_column='date', how='outer', fill_method='ffill'):
    """
    Merge multiple time series DataFrames on datetime column.
    
    Parameters:
    - dataframes: dict of {name: df} or list of DataFrames
    - date_column: name of datetime column
    - how: merge type ('inner', 'outer', 'left', 'right')
    - fill_method: 'ffill', 'bfill', 'interpolate', or None
    """
    if isinstance(dataframes, dict):
        dfs = list(dataframes.items())
    else:
        dfs = [(f'df_{i}', df) for i, df in enumerate(dataframes)]
    
    # Prepare first DataFrame
    name, df = dfs[0]
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column)
    
    # Rename columns to avoid conflicts
    df.columns = [f'{name}_{col}' for col in df.columns]
    merged = df
    
    # Merge remaining DataFrames
    for name, df in dfs[1:]:
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
        df.columns = [f'{name}_{col}' for col in df.columns]
        
        merged = merged.merge(df, left_index=True, right_index=True, how=how)
    
    # Sort by index
    merged = merged.sort_index()
    
    # Handle missing values
    if fill_method == 'ffill':
        merged = merged.fillna(method='ffill')
    elif fill_method == 'bfill':
        merged = merged.fillna(method='bfill')
    elif fill_method == 'interpolate':
        merged = merged.interpolate(method='time')
    
    return merged

# Example
dates1 = pd.date_range('2023-01-01', periods=5, freq='D')
dates2 = pd.date_range('2023-01-03', periods=5, freq='D')

df1 = pd.DataFrame({'date': dates1, 'value': [1, 2, 3, 4, 5]})
df2 = pd.DataFrame({'date': dates2, 'price': [10, 20, 30, 40, 50]})

merged = merge_timeseries({'stock': df1, 'price': df2}, fill_method='ffill')
print(merged)
```

### Key Considerations
1. **Alignment**: Use datetime index for automatic alignment
2. **Missing Values**: Forward fill for time series (last known value)
3. **Frequency**: Resample if datasets have different frequencies
4. **Duplicates**: Handle duplicate timestamps before merging

## Question 9
- [ ] Done

**If you have a DataFrame with multiple datetime columns , detail how you would create a new column combining them into the earliest datetime**

### Finding the Earliest Datetime Across Multiple Columns

```python
import pandas as pd
import numpy as np

# Sample DataFrame with multiple datetime columns
df = pd.DataFrame({
    'order_date': ['2025-01-15', '2025-02-20', '2025-03-10'],
    'ship_date': ['2025-01-18', '2025-02-15', '2025-03-12'],
    'delivery_date': ['2025-01-25', '2025-02-22', '2025-03-08']
})

# Convert to datetime
datetime_cols = ['order_date', 'ship_date', 'delivery_date']
for col in datetime_cols:
    df[col] = pd.to_datetime(df[col])

# Method 1: Using min() across columns (recommended)
df['earliest_date'] = df[datetime_cols].min(axis=1)
print(df)
#   order_date  ship_date delivery_date earliest_date
# 0 2025-01-15 2025-01-18    2025-01-25    2025-01-15
# 1 2025-02-20 2025-02-15    2025-02-22    2025-02-15
# 2 2025-03-10 2025-03-12    2025-03-08    2025-03-08

# Method 2: Using np.minimum for two columns
df['earliest_two'] = np.minimum(df['order_date'], df['ship_date'])

# Method 3: Using reduce for multiple columns
from functools import reduce
df['earliest_reduce'] = reduce(np.minimum, [df[col] for col in datetime_cols])

# Method 4: Stack and groupby (handles NaT gracefully)
df['earliest_stack'] = df[datetime_cols].stack().groupby(level=0).min()
```

### Handling NaT (Missing Datetimes)

```python
# With missing values
df_with_nat = pd.DataFrame({
    'date1': pd.to_datetime(['2025-01-15', pd.NaT, '2025-03-10']),
    'date2': pd.to_datetime(['2025-01-18', '2025-02-15', pd.NaT]),
})

# min() skips NaT by default
df_with_nat['earliest'] = df_with_nat[['date1', 'date2']].min(axis=1)
# Row 1: 2025-02-15 (skips NaT in date1)
# Row 2: 2025-03-10 (skips NaT in date2)

# To get NaT if ANY column is NaT:
df_with_nat['earliest_strict'] = df_with_nat[['date1', 'date2']].min(axis=1, skipna=False)
```

> **Interview Tip:** `df[cols].min(axis=1)` is the cleanest approach and handles NaT gracefully. For maximum datetime, use `.max(axis=1)`. This pattern is common in ML for computing **time-to-event features** (e.g., first interaction, earliest symptom).

---
