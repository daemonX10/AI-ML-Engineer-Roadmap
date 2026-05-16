# Apache Spark Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Write a Scala/Python Spark code snippet that reads a CSV file and calculates the average of a column**

### Answer

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col

"""
Pipeline:
1. Create Spark session
2. Read CSV file
3. Calculate average
4. Output result
"""

spark = SparkSession.builder.appName("AverageCalculation").getOrCreate()

# Read CSV
df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("data/sales.csv")

# Method 1: Using agg
avg_price = df.agg(avg("price")).collect()[0][0]
print(f"Average price: {avg_price:.2f}")

# Method 2: Using select
df.select(avg(col("price")).alias("avg_price")).show()

# Method 3: Group by category
df.groupBy("category").agg(avg("price").alias("avg_price")).show()
```

---

## Question 2
- [ ] Done

**Write Spark code to perform word count.**

### Answer

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, lower, col

"""
Pipeline:
1. Read text file
2. Split into words
3. Count occurrences
4. Sort by count
"""

spark = SparkSession.builder.appName("WordCount").getOrCreate()

# RDD approach
text_rdd = spark.sparkContext.textFile("data/text.txt")
word_counts = text_rdd \
    .flatMap(lambda line: line.lower().split()) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .sortBy(lambda x: x[1], ascending=False)

print(word_counts.take(10))

# DataFrame approach
df = spark.read.text("data/text.txt")
word_counts_df = df \
    .select(explode(split(lower(col("value")), "\\s+")).alias("word")) \
    .filter(col("word") != "") \
    .groupBy("word") \
    .count() \
    .orderBy(col("count").desc())

word_counts_df.show(10)
```

---

## Question 3
- [ ] Done

**Write Spark code to join two DataFrames.**

### Answer

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast

"""
Pipeline:
1. Create DataFrames
2. Perform join
3. Output result
"""

spark = SparkSession.builder.appName("JoinExample").getOrCreate()

orders = spark.createDataFrame([
    (1, 101, 100.0),
    (2, 102, 200.0),
    (3, 101, 150.0)
], ["order_id", "customer_id", "amount"])

customers = spark.createDataFrame([
    (101, "Alice", "NYC"),
    (102, "Bob", "LA")
], ["customer_id", "name", "city"])

# Inner join
inner_result = orders.join(customers, "customer_id")
inner_result.show()

# Left outer join
left_result = orders.join(customers, "customer_id", "left")

# Broadcast join for small table
broadcast_result = orders.join(broadcast(customers), "customer_id")
```

---

## Question 4
- [ ] Done

**Write Spark code for window functions.**

### Answer

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

"""
Pipeline:
1. Create DataFrame
2. Define window spec
3. Apply window functions
"""

spark = SparkSession.builder.appName("WindowFunctions").getOrCreate()

sales = spark.createDataFrame([
    ("2024-01-01", "Electronics", 1000),
    ("2024-01-02", "Electronics", 1500),
    ("2024-01-01", "Clothing", 500)
], ["date", "category", "sales"])

window_spec = Window.partitionBy("category").orderBy("date")

result = sales.withColumn(
    "running_total", sum("sales").over(window_spec)
).withColumn(
    "row_number", row_number().over(window_spec)
).withColumn(
    "lag_sales", lag("sales", 1).over(window_spec)
)

result.show()
```

---

## Question 5
- [ ] Done

**Write Spark code to handle null values.**

### Answer

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, coalesce, lit

"""
Pipeline:
1. Identify nulls
2. Apply handling strategy
3. Validate results
"""

spark = SparkSession.builder.appName("NullHandling").getOrCreate()

data = [(1, "Alice", 30, 50000.0),
        (2, "Bob", None, 60000.0),
        (3, None, 25, None)]

df = spark.createDataFrame(data, ["id", "name", "age", "salary"])

# Count nulls
null_counts = df.select([
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df.columns
])
null_counts.show()

# Drop rows with nulls
df_dropped = df.dropna()

# Fill nulls
df_filled = df.fillna({
    "name": "Unknown",
    "age": 0,
    "salary": 50000.0
})
df_filled.show()
```

---

## Question 6
- [ ] Done

**Write Spark code for reading and writing different file formats.**

### Answer

```python
from pyspark.sql import SparkSession

"""
Pipeline:
1. Read from format
2. Transform
3. Write to target format
"""

spark = SparkSession.builder.appName("FileFormats").getOrCreate()

df = spark.range(1000)

# CSV
df.write.option("header", "true").csv("output/csv")
csv_df = spark.read.option("header", "true").csv("output/csv")

# JSON
df.write.json("output/json")
json_df = spark.read.json("output/json")

# Parquet (recommended)
df.write.parquet("output/parquet")
parquet_df = spark.read.parquet("output/parquet")

# Parquet with partitioning
df.write.partitionBy("id").parquet("output/partitioned")
```

---

## Question 7
- [ ] Done

**Write Spark code for data aggregation and grouping.**

### Answer

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

"""
Pipeline:
1. Load data
2. Group by columns
3. Apply aggregations
"""

spark = SparkSession.builder.appName("Aggregation").getOrCreate()

sales = spark.createDataFrame([
    ("2024-01", "Electronics", "Laptop", 1200, 5),
    ("2024-01", "Electronics", "Phone", 800, 10),
    ("2024-01", "Clothing", "Shirt", 50, 100)
], ["month", "category", "product", "price", "quantity"])

# Basic aggregation
sales.groupBy("category").agg(
    sum("quantity").alias("total_qty"),
    avg("price").alias("avg_price"),
    count("*").alias("count")
).show()

# Pivot table
pivot_df = sales.groupBy("category").pivot("month").sum("quantity")
pivot_df.show()

# Rollup
sales.rollup("category", "product").agg(
    sum("quantity").alias("total_qty")
).show()
```

---

## Question 8
- [ ] Done

**Write Spark Streaming code to process real-time data.**

### Answer

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

"""
Pipeline:
1. Create streaming source
2. Transform data
3. Write to sink
"""

spark = SparkSession.builder.appName("StreamingExample").getOrCreate()

schema = StructType([
    StructField("user_id", StringType()),
    StructField("action", StringType()),
    StructField("timestamp", TimestampType()),
    StructField("value", DoubleType())
])

# Read from Kafka
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "events") \
    .load()

events = kafka_df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# Windowed aggregation
windowed = events \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(window("timestamp", "5 minutes"), "action") \
    .count()

# Write to console
query = windowed.writeStream \
    .format("console") \
    .outputMode("update") \
    .start()
```

---

## Question 9
- [ ] Done

**Write Spark code for data validation.**

### Answer

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

"""
Pipeline:
1. Load data
2. Define rules
3. Apply checks
"""

spark = SparkSession.builder.appName("Validation").getOrCreate()

data = [(1, "Alice", 30, "alice@email.com"),
        (2, "Bob", -5, "invalid"),
        (3, None, 25, "c@email.com")]

df = spark.createDataFrame(data, ["id", "name", "age", "email"])

# Validation rules
def validate(df):
    results = {}
    results['null_name'] = df.filter(col("name").isNull()).count()
    results['invalid_age'] = df.filter((col("age") < 0) | (col("age") > 120)).count()
    results['invalid_email'] = df.filter(
        ~col("email").rlike("^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+$")
    ).count()
    return results

print(validate(df))

# Flag invalid records
flagged = df.withColumn(
    "is_valid",
    col("name").isNotNull() & 
    (col("age") >= 0) & 
    col("email").rlike("^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+$")
)
flagged.show()
```

---

## Question 10
- [ ] Done

**Write Spark ML code for a simple machine learning pipeline.**

### Answer

```python
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

"""
Pipeline:
1. Prepare features
2. Build pipeline
3. Train and evaluate
"""

spark = SparkSession.builder.appName("MLPipeline").getOrCreate()

data = spark.createDataFrame([
    (0, "male", 25, 50000, 0),
    (1, "female", 30, 60000, 1),
    (2, "male", 35, 70000, 1)
], ["id", "gender", "age", "salary", "label"])

train, test = data.randomSplit([0.8, 0.2], seed=42)

# Build pipeline
gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_idx")
assembler = VectorAssembler(
    inputCols=["gender_idx", "age", "salary"],
    outputCol="features_raw"
)
scaler = StandardScaler(inputCol="features_raw", outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=[gender_indexer, assembler, scaler, lr])

# Train
model = pipeline.fit(train)

# Predict
predictions = model.transform(test)
predictions.select("id", "label", "prediction").show()

# Evaluate
evaluator = BinaryClassificationEvaluator(labelCol="label")
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc:.4f}")
```

## Question 11
- [ ] Done

**Implement a Spark streaming application that counts words in text data received from a socket**

*Answer to be added.*

---

## Question 12
- [ ] Done

**Code a Spark SQL query that finds the top 3 most occurring words in a DataFrame**

*Answer to be added.*

---

## Question 13
- [ ] Done

**Create a Spark job that joins two datasets on a key and summarizes data**

*Answer to be added.*

---

## Question 14
- [ ] Done

**Write a Python function using PySpark to filter out records with null values in a specific column**

*Answer to be added.*

---

## Question 15
- [ ] Done

**Implement a logistic regression model in Spark MLlib**

*Answer to be added.*

---

## Question 16
- [ ] Done

**Create a Spark Streaming application to aggregate event data by time windows**

*Answer to be added.*

---

## Question 17
- [ ] Done

**Write Spark code to calculate the PageRank of a simple website link graph**

*Answer to be added.*

---
