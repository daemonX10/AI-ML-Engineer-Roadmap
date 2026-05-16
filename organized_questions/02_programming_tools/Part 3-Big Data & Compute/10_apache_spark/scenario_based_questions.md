# Apache Spark Interview Questions - Scenario Based Questions

## Question 1
- [ ] Done

**Discuss the role of accumulators and broadcast variables in Spark.**

### Answer

### Accumulators

| Feature | Description |
|---------|-------------|
| Purpose | Aggregate values across executors |
| Direction | Write-only from executors |
| Use cases | Counters, debugging |

### Broadcast Variables

| Feature | Description |
|---------|-------------|
| Purpose | Share read-only data with executors |
| Use cases | Lookup tables, model parameters |
| Benefit | Avoids shipping data with each task |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, broadcast
from pyspark.sql.types import StringType

spark = SparkSession.builder.appName("SharedVars").getOrCreate()
sc = spark.sparkContext

# Accumulator
error_count = sc.accumulator(0)

def process(value):
    global error_count
    if value < 0:
        error_count += 1
        return None
    return value * 2

rdd = sc.parallelize([-1, 2, 3, -4, 5])
result = rdd.map(process).filter(lambda x: x is not None).collect()
print(f"Errors: {error_count.value}")

# Broadcast variable
country_codes = {"US": "United States", "UK": "United Kingdom"}
broadcast_codes = sc.broadcast(country_codes)

@udf(StringType())
def get_country(code):
    return broadcast_codes.value.get(code, "Unknown")

df = spark.createDataFrame([("US",), ("UK",)], ["code"])
df.withColumn("name", get_country(col("code"))).show()
```

---

## Question 2
- [ ] Done

**Your Spark job is running out of memory. How do you diagnose and fix it?**

### Answer

### Diagnosis Steps

| Step | Action |
|------|--------|
| 1 | Check Spark UI for failed tasks |
| 2 | Review executor memory settings |
| 3 | Look for data skew |
| 4 | Check GC overhead |

### Solutions

| Cause | Solution |
|-------|----------|
| Large shuffle | Increase partitions |
| Data skew | Salt keys, AQE |
| Collect too much | Use take(), limit() |
| Cache bloat | Unpersist when done |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast

# Memory-optimized config
spark = SparkSession.builder \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.memoryOverhead", "2g") \
    .config("spark.sql.shuffle.partitions", "400") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# BAD: Collecting large data
# all_data = df.collect()  # OOM

# GOOD: Write instead
df.write.parquet("output/")

# BAD: Both large tables join
# result = large1.join(large2, "key")

# GOOD: Broadcast small table
result = large_df.join(broadcast(small_df), "key")

# Release cache
df.cache()
df.count()
df.unpersist()  # Free memory
```

---

## Question 3
- [ ] Done

**You need to process 100TB of data daily. How would you design the Spark job?**

### Answer

### Architecture

| Component | Configuration |
|-----------|---------------|
| Cluster size | 100+ nodes |
| Executor memory | 16-32GB |
| Partitions | 10,000+ |
| Storage | Parquet on HDFS/S3 |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Production config
spark = SparkSession.builder \
    .config("spark.executor.instances", "200") \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.memory", "16g") \
    .config("spark.sql.shuffle.partitions", "2000") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.parquet.compression.codec", "snappy") \
    .getOrCreate()

def process_daily_data(date):
    # Partition pruning
    raw = spark.read.parquet(f"s3://data/raw/date={date}/")
    
    # Transform
    processed = raw \
        .repartition(2000, "user_id") \
        .groupBy("user_id") \
        .agg({"value": "sum", "*": "count"})
    
    # Write partitioned
    processed.write \
        .partitionBy("region") \
        .mode("overwrite") \
        .parquet(f"s3://data/processed/date={date}/")
```

---

## Question 4
- [ ] Done

**How would you handle late-arriving data in Spark Streaming?**

### Answer

### Watermarking

| Concept | Description |
|---------|-------------|
| Event time | Timestamp in data |
| Watermark | Late data threshold |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark = SparkSession.builder.appName("LateData").getOrCreate()

schema = StructType([
    StructField("user_id", StringType()),
    StructField("event_time", TimestampType()),
    StructField("value", DoubleType())
])

events = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "events") \
    .load() \
    .select(from_json(col("value").cast("string"), schema).alias("data")) \
    .select("data.*")

# Accept data up to 1 hour late
windowed = events \
    .withWatermark("event_time", "1 hour") \
    .groupBy(window("event_time", "10 minutes"), "user_id") \
    .sum("value")

query = windowed.writeStream \
    .format("parquet") \
    .option("path", "/output/events") \
    .option("checkpointLocation", "/checkpoint") \
    .outputMode("append") \
    .start()
```

---

## Question 5
- [ ] Done

**Your Spark job has some very slow tasks. How do you identify and fix?**

### Answer

### Diagnosis

| Indicator | Cause |
|-----------|-------|
| One slow task | Data skew |
| All slow | Under-provisioned |
| Spill to disk | Memory issue |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .config("spark.speculation", "true") \
    .appName("SlowTaskFix").getOrCreate()

# Analyze skew
def analyze_skew(df, key_col):
    counts = df.groupBy(key_col).count()
    stats = counts.select(
        mean("count"), stddev("count"), max("count")
    ).collect()[0]
    print(f"Mean: {stats[0]:.0f}, Std: {stats[1]:.0f}, Max: {stats[2]}")
    
    # Find skewed keys
    threshold = stats[0] + 3 * stats[1]
    skewed = counts.filter(col("count") > threshold)
    skewed.show()

# Enable speculation for stragglers
spark.conf.set("spark.speculation", "true")
spark.conf.set("spark.speculation.multiplier", "1.5")
```

---

## Question 6
- [ ] Done

**How would you migrate a Hadoop MapReduce job to Spark?**

### Answer

### Migration Steps

| Step | Action |
|------|--------|
| 1 | Analyze current job |
| 2 | Convert Mapper to map/flatMap |
| 3 | Convert Reducer to reduceByKey |
| 4 | Test with same data |

### Python Code Example
```python
# Original MapReduce concept:
# Mapper: emit (word, 1)
# Reducer: sum counts

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("Migration").getOrCreate()

# RDD approach (closest to MapReduce)
text_rdd = spark.sparkContext.textFile("hdfs:///input/")
word_counts = text_rdd \
    .flatMap(lambda line: line.split()) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda a, b: a + b)
word_counts.saveAsTextFile("hdfs:///output/")

# DataFrame approach (recommended)
df = spark.read.text("hdfs:///input/")
result = df.select(explode(split(col("value"), " ")).alias("word")) \
           .groupBy("word").count()
result.write.parquet("hdfs:///output/")
```

### Benefits
- 10-100x faster
- Simpler code
- Better debugging

---

## Question 7
- [ ] Done

**How do you handle schema evolution in Spark with Parquet files?**

### Answer

### Options

| Option | Description |
|--------|-------------|
| `mergeSchema` | Merge schemas from all files |
| `schema` | Enforce specific schema |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import lit

spark = SparkSession.builder \
    .config("spark.sql.parquet.mergeSchema", "true") \
    .appName("SchemaEvolution").getOrCreate()

# Merge schema automatically
df = spark.read.option("mergeSchema", "true").parquet("data/events/")

# Enforce schema
schema = StructType([
    StructField("id", LongType()),
    StructField("name", StringType()),
    StructField("new_field", StringType())
])
df = spark.read.schema(schema).parquet("data/events/")

# Handle missing columns
def safe_read(path, required_cols):
    df = spark.read.parquet(path)
    for col_name, (dtype, default) in required_cols.items():
        if col_name not in df.columns:
            df = df.withColumn(col_name, lit(default).cast(dtype))
    return df

required = {"new_col": ("string", "unknown")}
df = safe_read("data/events/", required)

# Union with different schemas
df1.unionByName(df2, allowMissingColumns=True)
```

### Best Practices
- Add new columns as nullable
- Never remove columns
- Version your schemas

## Question 8
- [ ] Done

**How would you build a recommender system in Spark?**

*Answer to be added.*

---

## Question 9
- [ ] Done

**Discuss a strategy to perform real-time sentiment analysis using Spark Streaming**

*Answer to be added.*

---

## Question 10
- [ ] Done

**Propose an ETL pipeline design using Spark to process large datasets**

*Answer to be added.*

---

## Question 11
- [ ] Done

**How would you use Spark to detect fraudulent behavior in financial transactions?**

*Answer to be added.*

---

## Question 12
- [ ] Done

**Illustrate how you would use Spark to optimize marketing strategies based on customer behavior data**

*Answer to be added.*

---
