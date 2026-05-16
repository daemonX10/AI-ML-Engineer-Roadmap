# Apache Spark Interview Questions - General Questions

## Question 1
- [ ] Done

**How do you handle partitioning in Spark to optimize performance?**

### Answer

### Partitioning Strategies

| Strategy | Use Case |
|----------|----------|
| **Hash partitioning** | Even distribution by key |
| **Range partitioning** | Ordered data |
| **Custom partitioning** | Domain-specific logic |

### Key Considerations

| Factor | Recommendation |
|--------|----------------|
| Number of partitions | 2-4x number of cores |
| Partition size | 128MB - 1GB ideal |
| Shuffle partitions | `spark.sql.shuffle.partitions` |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .config("spark.sql.shuffle.partitions", "200") \
    .appName("Partitioning").getOrCreate()

df = spark.range(1000000)
print(f"Partitions: {df.rdd.getNumPartitions()}")

# Repartition (causes shuffle)
df_repartitioned = df.repartition(100)

# Coalesce (no shuffle, reduces only)
df_coalesced = df.coalesce(10)

# Partition when writing
df.write.partitionBy("date", "country").parquet("/data/output")
```

---

## Question 2
- [ ] Done

**How do you handle data skew in Spark?**

### Answer

### Solutions

| Solution | Description |
|----------|-------------|
| **Salting** | Add random prefix to keys |
| **Broadcast join** | Avoid shuffle for small tables |
| **Adaptive Query Execution** | Dynamic optimization |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .appName("DataSkew").getOrCreate()

# Salting technique
def salt_join(df_large, df_small, key_col, num_salts=10):
    df_large_salted = df_large.withColumn(
        "salt", (rand() * num_salts).cast("int")
    ).withColumn(
        "salted_key", concat(col(key_col), lit("_"), col("salt"))
    )
    
    salts = spark.range(num_salts).withColumnRenamed("id", "salt")
    df_small_replicated = df_small.crossJoin(salts).withColumn(
        "salted_key", concat(col(key_col), lit("_"), col("salt"))
    )
    
    return df_large_salted.join(df_small_replicated, "salted_key")
```

---

## Question 3
- [ ] Done

**What is the difference between cache() and persist()?**

### Answer

### Comparison

| Method | Storage Level |
|--------|---------------|
| `cache()` | MEMORY_ONLY (default) |
| `persist()` | Configurable |

### Storage Levels

| Level | Memory | Disk | Serialized |
|-------|--------|------|------------|
| MEMORY_ONLY | Yes | No | No |
| MEMORY_AND_DISK | Yes | Yes | No |
| MEMORY_ONLY_SER | Yes | No | Yes |
| DISK_ONLY | No | Yes | N/A |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark import StorageLevel

spark = SparkSession.builder.appName("Caching").getOrCreate()

df = spark.range(1000000)

df.cache()  # Same as persist(MEMORY_ONLY)
df.persist(StorageLevel.MEMORY_AND_DISK)

# Check if cached
print(f"Is cached: {df.is_cached}")

# Clear cache
df.unpersist()
```

---

## Question 4
- [ ] Done

**How do you optimize Spark joins?**

### Answer

### Join Types and Performance

| Join Type | Description |
|-----------|-------------|
| Broadcast Hash Join | Small table broadcast (fastest) |
| Sort Merge Join | Both tables sorted |
| Shuffle Hash Join | Hash partitioning |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast

spark = SparkSession.builder \
    .config("spark.sql.autoBroadcastJoinThreshold", "10485760") \
    .appName("JoinOptimization").getOrCreate()

large_df = spark.range(10000000).withColumnRenamed("id", "key")
small_df = spark.createDataFrame([(i, f"v{i}") for i in range(1000)], ["key", "value"])

# Broadcast join (explicit)
result = large_df.join(broadcast(small_df), "key")
result.explain()

# Bucket tables for frequent joins
large_df.write.bucketBy(100, "key").sortBy("key").saveAsTable("bucketed_large")
```

---

## Question 5
- [ ] Done

**How do you debug and monitor Spark applications?**

### Answer

### Monitoring Tools

| Tool | Purpose |
|------|---------|
| Spark UI | Jobs, stages, tasks, storage |
| History Server | Completed applications |
| Event logs | Detailed execution info |

### Python Code Example
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs:///spark-logs") \
    .appName("Debugging").getOrCreate()

sc = spark.sparkContext

# Check active jobs
print(f"Active jobs: {sc.statusTracker().getActiveJobIds()}")

# Log execution plan
df = spark.range(1000000).filter("id > 500000")
df.explain(mode="extended")

# Show partition info
print(f"Partitions: {df.rdd.getNumPartitions()}")
```

---

## Question 6
- [ ] Done

**What configuration parameters are important for Spark performance?**

### Answer

### Key Configurations

| Parameter | Description | Default |
|-----------|-------------|---------|
| `spark.executor.memory` | Executor heap size | 1g |
| `spark.executor.cores` | Cores per executor | 1 |
| `spark.sql.shuffle.partitions` | Shuffle partitions | 200 |

### Python Code Example
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("OptimizedSpark") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.instances", "10") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .getOrCreate()

# Check configuration
for item in spark.sparkContext.getConf().getAll():
    print(f"{item[0]}: {item[1]}")
```

### Rules of Thumb
- Executor memory: 4-8GB
- Executor cores: 4-5 per executor
- Shuffle partitions: 100-200 per core

---

## Question 7
- [ ] Done

**How can you minimize data shuffling in Spark?**

*Answer to be added.*

---

## Question 8
- [ ] Done

**Discuss the importance and methods of caching/persistence in Spark**

*Answer to be added.*

---