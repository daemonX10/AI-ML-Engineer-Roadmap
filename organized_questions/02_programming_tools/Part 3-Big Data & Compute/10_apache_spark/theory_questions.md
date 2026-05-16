# Apache Spark Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What is Apache Spark and what are its main components?**

### Answer

**Definition**: Apache Spark is an open-source, distributed computing engine designed for fast, large-scale data processing with in-memory computing capabilities.

### Core Components

| Component | Purpose |
|-----------|---------|
| **Spark Core** | Task scheduling, memory management, RDD API |
| **Spark SQL** | Structured data processing with DataFrame/Dataset |
| **Spark Streaming** | Real-time stream processing |
| **MLlib** | Machine learning library |
| **GraphX** | Graph processing |

### Key Features

| Feature | Description |
|---------|-------------|
| In-memory computing | 100x faster than MapReduce |
| Lazy evaluation | Optimizes execution plan |
| DAG execution | Efficient task scheduling |
| Fault tolerance | RDD lineage for recovery |

### Python Code Example
```python
from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("SparkDemo") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Check Spark version
print(f"Spark version: {spark.version}")

# Create DataFrame
data = [("Alice", 30), ("Bob", 25), ("Charlie", 35)]
df = spark.createDataFrame(data, ["name", "age"])
df.show()

# Stop session
spark.stop()
```

---

## Question 2
- [ ] Done

**Explain RDD, DataFrame, and Dataset in Spark.**

### Answer

### Comparison

| Feature | RDD | DataFrame | Dataset |
|---------|-----|-----------|---------|
| **Type safety** | Compile-time | Runtime | Compile-time |
| **Schema** | No | Yes | Yes |
| **Optimization** | None | Catalyst | Catalyst |
| **API** | Low-level | High-level | High-level |
| **Language** | All | All | Scala/Java |

### When to Use

| Use Case | Best Choice |
|----------|-------------|
| Unstructured data | RDD |
| Structured/semi-structured | DataFrame |
| Type safety needed | Dataset |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("RDDvsDF").getOrCreate()

# RDD: Low-level, functional transformations
rdd = spark.sparkContext.parallelize([1, 2, 3, 4, 5])
result_rdd = rdd.filter(lambda x: x > 2).map(lambda x: x * 2)
print(f"RDD result: {result_rdd.collect()}")

# DataFrame: High-level, optimized
data = [(1, "a"), (2, "b"), (3, "c")]
df = spark.createDataFrame(data, ["id", "value"])
result_df = df.filter(col("id") > 1).select("value")
result_df.show()
```

---

## Question 3
- [ ] Done

**What is lazy evaluation in Spark?**

### Answer

**Definition**: Spark delays execution of transformations until an action is called, allowing it to optimize the entire computation plan.

### Transformation vs Action

| Transformation | Action |
|----------------|--------|
| `map`, `filter`, `select` | `collect`, `count`, `show` |
| `groupBy`, `join` | `write`, `take` |
| Returns new RDD/DF | Returns value/writes data |
| Lazy | Triggers execution |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("LazyEval").getOrCreate()

df = spark.range(1000000)

# Transformations (LAZY - nothing executes yet)
transformed = df \
    .filter(col("id") > 500000) \
    .select((col("id") * 2).alias("doubled"))

print("Transformations defined - no execution yet")

# Action (TRIGGERS execution)
count = transformed.count()
print(f"Count: {count}")

# View execution plan
transformed.explain(True)
```

---

## Question 4
- [ ] Done

**Explain Spark's DAG and execution model.**

### Answer

**Definition**: DAG (Directed Acyclic Graph) represents the logical execution plan showing dependencies between RDDs/operations.

### Execution Flow

| Step | Description |
|------|-------------|
| 1. DAG creation | Transformations create DAG |
| 2. Stage division | Split at shuffle boundaries |
| 3. Task creation | Tasks per partition per stage |
| 4. Execution | Tasks run on executors |

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Job** | Action triggers a job |
| **Stage** | Set of tasks with no shuffle |
| **Task** | Unit of work on one partition |
| **Shuffle** | Data redistribution across nodes |

### Python Code Example
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DAGExample").getOrCreate()
sc = spark.sparkContext

rdd = sc.parallelize(range(1000), 4)

# Transformations create DAG
mapped = rdd.map(lambda x: (x % 10, x))      # Stage 1
grouped = mapped.groupByKey()                 # Shuffle boundary
summed = grouped.mapValues(sum)               # Stage 2

# View DAG
print(summed.toDebugString())

# Action triggers execution
result = summed.collect()
```

---

## Question 5
- [ ] Done

**What are narrow and wide transformations?**

### Answer

### Comparison

| Narrow Transformation | Wide Transformation |
|-----------------------|---------------------|
| Data stays in partition | Data shuffles across partitions |
| No network transfer | Network transfer required |
| `map`, `filter`, `flatMap` | `groupBy`, `reduceByKey`, `join` |

### Python Code Example
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Transformations").getOrCreate()
sc = spark.sparkContext

rdd = sc.parallelize([(1, 'a'), (2, 'b'), (3, 'a'), (4, 'b')], 2)

# Narrow transformations (no shuffle)
narrow_1 = rdd.map(lambda x: (x[0] * 2, x[1]))
narrow_2 = rdd.filter(lambda x: x[0] > 1)

# Wide transformations (shuffle required)
wide_1 = rdd.groupByKey()
wide_2 = rdd.reduceByKey(lambda a, b: a + b)

print(narrow_1.toDebugString())
print(wide_1.toDebugString())
```

### Interview Tip
Prefer `reduceByKey` over `groupByKey` as it performs local aggregation before shuffle.

---

## Question 6
- [ ] Done

**How does Spark achieve fault tolerance ?**

### Answer

### Fault Tolerance Mechanisms

| Mechanism | Description |
|-----------|-------------|
| **RDD Lineage** | Recompute lost partitions from source |
| **Checkpointing** | Save RDD to reliable storage |
| **Task retry** | Re-execute failed tasks |

### Python Code Example
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("FaultTolerance").getOrCreate()
sc = spark.sparkContext

# Set checkpoint directory
sc.setCheckpointDir("hdfs:///checkpoints")

rdd = sc.parallelize(range(1000))
rdd1 = rdd.map(lambda x: x * 2)
rdd2 = rdd1.filter(lambda x: x > 100)

# View lineage
print(rdd2.toDebugString())

# Checkpoint breaks lineage
rdd2.checkpoint()
rdd2.count()

print("After checkpoint:")
print(rdd2.toDebugString())
```

---

## Question 7
- [ ] Done

**What is the Catalyst optimizer?**

### Answer

**Definition**: Catalyst is Spark SQL's query optimizer that transforms logical plans into optimized physical execution plans.

### Optimization Phases

| Phase | Description |
|-------|-------------|
| **Analysis** | Resolve references, validate |
| **Logical Optimization** | Predicate pushdown, constant folding |
| **Physical Planning** | Generate execution strategies |
| **Code Generation** | Generate optimized bytecode |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("Catalyst").getOrCreate()

df = spark.range(1000000).withColumn("value", col("id") * 2)

result = df \
    .filter(col("id") > 500000) \
    .filter(col("value") < 1500000) \
    .select("id")

# View optimization plans
result.explain(mode="extended")
```

---

## Question 8
- [ ] Done

**Explain Spark memory management.**

### Answer

### Memory Regions

| Region | Purpose | Default |
|--------|---------|---------|
| **Execution** | Shuffles, joins, sorts | 50% |
| **Storage** | Cache, broadcast | 50% |
| **Reserved** | System overhead | 300MB |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark import StorageLevel

spark = SparkSession.builder \
    .config("spark.executor.memory", "4g") \
    .config("spark.memory.fraction", "0.6") \
    .config("spark.memory.storageFraction", "0.5") \
    .appName("MemoryManagement").getOrCreate()

df = spark.range(1000000)

# Different storage levels
df.cache()  # MEMORY_ONLY
df.persist(StorageLevel.MEMORY_AND_DISK)
df.persist(StorageLevel.MEMORY_ONLY_SER)

# Unpersist when done
df.unpersist()
```

---

## Question 9
- [ ] Done

**What is data shuffling in Spark?**

### Answer

**Definition**: Shuffling is the process of redistributing data across partitions, typically required for wide transformations.

### Shuffle Operations

| Operation | Requires Shuffle |
|-----------|------------------|
| `groupByKey` | Yes |
| `reduceByKey` | Yes (with combiner) |
| `join` | Yes (unless co-partitioned) |
| `map`, `filter` | No |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast

spark = SparkSession.builder \
    .config("spark.sql.shuffle.partitions", "200") \
    .appName("ShuffleExample").getOrCreate()

# BAD: groupByKey shuffles all data
rdd = spark.sparkContext.parallelize([(i % 100, i) for i in range(10000)])
grouped = rdd.groupByKey().mapValues(sum)

# BETTER: reduceByKey does local combine first
reduced = rdd.reduceByKey(lambda a, b: a + b)

# Broadcast join (avoid shuffle)
large_df = spark.range(1000000).withColumnRenamed("id", "key")
small_df = spark.createDataFrame([(i, f"v{i}") for i in range(100)], ["key", "value"])
broadcast_join = large_df.join(broadcast(small_df), "key")
```

---

## Question 10
- [ ] Done

**Explain Spark Streaming vs Structured Streaming.**

### Answer

### Comparison

| Feature | DStream (old) | Structured Streaming |
|---------|---------------|----------------------|
| **API** | RDD-based | DataFrame-based |
| **Exactly-once** | Difficult | Built-in |
| **Event time** | Manual | Native support |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("Streaming").getOrCreate()

# Read from Kafka
stream_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "events") \
    .load()

# Windowed aggregation with watermark
schema = "user_id STRING, event_time TIMESTAMP, action STRING"
events = stream_df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

result = events \
    .withWatermark("event_time", "10 minutes") \
    .groupBy(window("event_time", "5 minutes"), "user_id") \
    .count()

# Write to console
query = result.writeStream \
    .format("console") \
    .outputMode("update") \
    .start()
```



---

# --- Missing Questions Restored from Source (Q11-Q33) ---

## Question 11
- [ ] Done

**Describe the various ways to run Spark applications (cluster, client, local modes)**

**Answer:**

### Spark Deploy Modes

| Mode | Driver Location | Use Case |
|------|----------------|----------|
| **Local** | Single JVM on local machine | Development, testing |
| **Client** | Driver on submitting machine | Interactive (spark-shell, notebooks) |
| **Cluster** | Driver on a cluster worker | Production, long-running jobs |

### Local Mode
```bash
# Run with all available cores
spark-submit --master local[*] my_app.py

# Run with 4 cores
spark-submit --master local[4] my_app.py
```
```python
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("LocalApp").getOrCreate()
```

### Client Mode (default)
```bash
# Driver runs on the submitting machine
spark-submit --master yarn --deploy-mode client my_app.py
```
```
Client Machine          Cluster
┌──────────────┐     ┌──────────────┐
│  Driver     │───▶│  Executor 1  │
│  (here)     │     │  Executor 2  │
└──────────────┘     └──────────────┘
```

### Cluster Mode
```bash
# Driver runs on a cluster worker node
spark-submit --master yarn --deploy-mode cluster my_app.py
```
```
Cluster
┌────────────────────────┐
│  Driver (on worker)  │
│  Executor 1          │
│  Executor 2          │
└────────────────────────┘
```

### Cluster Managers

| Manager | Command | Notes |
|---------|---------|-------|
| **YARN** | `--master yarn` | Most common in Hadoop ecosystems |
| **Kubernetes** | `--master k8s://...` | Cloud-native deployments |
| **Standalone** | `--master spark://host:7077` | Spark's built-in manager |
| **Mesos** | `--master mesos://host:5050` | Deprecated in Spark 3.2+ |

### Interview Tip
Use **client mode** for interactive work (spark-shell, Jupyter) so you see output directly. Use **cluster mode** for production because the driver runs inside the cluster — if the submitting machine disconnects, the job continues. Local mode is only for development/testing.

---

## Question 12
- [ ] Done

**What are Spark’s data source APIs and how do you use them?**

**Answer:**

### Definition
Spark's Data Source API provides a unified interface for reading/writing data from various formats and storage systems through `DataFrameReader` and `DataFrameWriter`.

### Built-in Data Sources

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| **Parquet** | `spark.read.parquet()` | `df.write.parquet()` | Default format, columnar |
| **JSON** | `spark.read.json()` | `df.write.json()` | Schema inference |
| **CSV** | `spark.read.csv()` | `df.write.csv()` | Header, delimiter options |
| **ORC** | `spark.read.orc()` | `df.write.orc()` | Hive-optimized |
| **JDBC** | `spark.read.jdbc()` | `df.write.jdbc()` | Databases |
| **Avro** | `spark.read.format("avro")` | `df.write.format("avro")` | Requires spark-avro package |

### Code Examples
```python
# Reading with options
df = spark.read \
    .format("csv") \
    .option("header", True) \
    .option("inferSchema", True) \
    .option("delimiter", ",") \
    .load("/data/sales.csv")

# Reading with explicit schema
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
schema = StructType([
    StructField("name", StringType()),
    StructField("age", IntegerType())
])
df = spark.read.schema(schema).json("/data/people.json")

# Writing with partitioning
df.write \
    .mode("overwrite") \
    .partitionBy("year", "month") \
    .parquet("/output/partitioned_data")

# JDBC (database)
df = spark.read.jdbc(
    url="jdbc:postgresql://host:5432/db",
    table="users",
    properties={"user": "admin", "password": "pass"}
)
```

### Write Modes

| Mode | Behavior |
|------|----------|
| `append` | Add to existing data |
| `overwrite` | Replace existing data |
| `ignore` | Skip if data exists |
| `error` (default) | Throw error if exists |

### Interview Tip
Always specify a **schema** instead of relying on `inferSchema` in production — schema inference reads the entire file first (slow) and may infer wrong types. Use **Parquet** as the default format (columnar, compressed, schema-embedded). Partition output by frequently-filtered columns for faster queries.

---

## Question 14
- [ ] Done

**How does Tungsten contribute to Spark’s performance ?**

**Answer:**

### Definition
Project Tungsten is Spark's initiative for **CPU and memory efficiency**, moving from JVM-managed objects to manual memory management and code generation.

### Key Contributions

| Feature | What It Does | Impact |
|---------|-------------|--------|
| **Off-heap memory** | Manages memory outside JVM heap | Avoids GC overhead |
| **Binary format** | Stores data in compact binary (UnsafeRow) | 5-10x less memory |
| **Whole-stage codegen** | Generates optimized Java bytecode at runtime | 2-10x faster execution |
| **Cache-aware computation** | Algorithms designed for CPU cache hierarchy | Better L1/L2 cache usage |

### How UnsafeRow Works
```
JVM Object (traditional):             UnsafeRow (Tungsten):
┌────────────────────┐         ┌────────────────────┐
│ Object header (16B) │         │ null bitmap (8B)     │
│ String ref (8B)     │         │ field1 offset (8B)   │
│ Int value (4B)      │         │ field2 value (8B)    │
│ padding (4B)        │         │ var-length data      │
└────────────────────┘         └────────────────────┘
~32+ bytes                         ~24 bytes (compact)
```

### Whole-Stage Code Generation
```python
# Before Tungsten: interpreted, virtual function calls per row
# After Tungsten: generates optimized loop for entire pipeline

df = spark.range(10000000)
result = df.filter("id > 5000000").selectExpr("id * 2 as doubled")

# View generated code
result.explain("codegen")
```

### Configuration
```python
spark.conf.set("spark.sql.tungsten.enabled", "true")  # Default: true
spark.conf.set("spark.sql.codegen.wholeStage", "true")  # Default: true
```

### Interview Tip
Tungsten is why DataFrame/Dataset operations are **faster than RDD** — RDDs use JVM objects with GC overhead, while DataFrames use Tungsten's binary format. The key insight: Spark moved from an **interpreted volcano model** to a **compiled push-based model**, generating tight loops that maximize CPU cache utilization.

---

## Question 15
- [ ] Done

**Briefly describe the Spark Core API and its features**

**Answer:**

### Definition
Spark Core is the **foundation** of Apache Spark, providing distributed task dispatching, scheduling, I/O, and the RDD abstraction.

### Core Components

| Component | Purpose |
|-----------|--------|
| **RDD** | Resilient Distributed Dataset — immutable distributed collection |
| **SparkContext** | Entry point, connects to cluster manager |
| **DAG Scheduler** | Translates RDD lineage into stages |
| **Task Scheduler** | Assigns tasks to executors |
| **Block Manager** | Manages data blocks across nodes |
| **Broadcast/Accumulators** | Shared variables across tasks |

### RDD Operations
```python
from pyspark import SparkContext
sc = SparkContext("local[*]", "CoreAPI")

# Create RDD
rdd = sc.parallelize([1, 2, 3, 4, 5], numSlices=4)

# Transformations (lazy)
filtered = rdd.filter(lambda x: x > 2)
mapped = filtered.map(lambda x: x * 10)

# Actions (trigger execution)
result = mapped.collect()        # [30, 40, 50]
total = mapped.reduce(lambda a, b: a + b)  # 120
count = mapped.count()           # 3
```

### Shared Variables
```python
# Broadcast: read-only shared data (avoid shipping with every task)
lookup = sc.broadcast({"US": "United States", "UK": "United Kingdom"})
rdd.map(lambda x: lookup.value.get(x, "Unknown"))

# Accumulator: write-only aggregation (e.g., counters)
error_count = sc.accumulator(0)
def process(record):
    if record is None:
        error_count.add(1)
    return record
rdd.foreach(process)
print(f"Errors: {error_count.value}")
```

### Key Features
- **In-memory computing**: 100x faster than disk-based MapReduce
- **Lazy evaluation**: Transformations build DAG, actions trigger execution
- **Fault tolerance**: RDD lineage enables recomputation on failure
- **Partitioning**: Data distributed across cluster nodes

### Interview Tip
Spark Core's RDD API is the **lowest level** API. For most work, prefer **DataFrames/Datasets** (higher level, Catalyst-optimized). Use RDD API only when you need fine-grained control over partitioning, custom data types, or unstructured data processing.

---

## Question 16
- [ ] Done

**What are the transformations and actions in Spark RDDs ?**

**Answer:**

### Definition
- **Transformations**: Create a new RDD from an existing one (lazy — not executed immediately)
- **Actions**: Trigger computation and return results to the driver or write to storage

### Common Transformations

| Transformation | Description | Example |
|---------------|-------------|--------|
| `map(f)` | Apply function to each element | `rdd.map(lambda x: x*2)` |
| `filter(f)` | Keep elements where f is True | `rdd.filter(lambda x: x>5)` |
| `flatMap(f)` | Map then flatten | `rdd.flatMap(lambda x: x.split())` |
| `union(rdd2)` | Combine two RDDs | `rdd1.union(rdd2)` |
| `distinct()` | Remove duplicates | `rdd.distinct()` |
| `groupByKey()` | Group values by key | `pairs.groupByKey()` |
| `reduceByKey(f)` | Aggregate values by key | `pairs.reduceByKey(lambda a,b: a+b)` |
| `sortByKey()` | Sort by key | `pairs.sortByKey()` |
| `join(rdd2)` | Inner join on keys | `rdd1.join(rdd2)` |

### Common Actions

| Action | Description | Example |
|--------|-------------|--------|
| `collect()` | Return all elements | `rdd.collect()` |
| `count()` | Number of elements | `rdd.count()` |
| `first()` | First element | `rdd.first()` |
| `take(n)` | First n elements | `rdd.take(5)` |
| `reduce(f)` | Aggregate all elements | `rdd.reduce(lambda a,b: a+b)` |
| `foreach(f)` | Apply function (no return) | `rdd.foreach(print)` |
| `saveAsTextFile()` | Write to file | `rdd.saveAsTextFile("/out")` |

### Code Example
```python
rdd = sc.parallelize(["hello world", "hello spark", "world of data"])

# Transformation chain (lazy — builds DAG)
words = rdd.flatMap(lambda line: line.split(" "))  # Split into words
pairs = words.map(lambda word: (word, 1))           # Create pairs
counts = pairs.reduceByKey(lambda a, b: a + b)       # Count by key

# Action (triggers execution)
result = counts.collect()  # [('hello',2), ('world',2), ('spark',1), ('of',1), ('data',1)]
```

### Narrow vs Wide Transformations

| Type | Shuffle? | Examples |
|------|----------|----------|
| **Narrow** | No | map, filter, flatMap, union |
| **Wide** | Yes | groupByKey, reduceByKey, join, distinct |

### Interview Tip
Always prefer `reduceByKey` over `groupByKey` — `reduceByKey` combines values locally before shuffling (like a combiner in MapReduce), reducing network transfer significantly. `groupByKey` shuffles ALL data, which can cause OOM errors on large datasets.

---

## Question 17
- [ ] Done

**What are Key-Value pair RDDs , and when would you use them?**

**Answer:**

### Definition
Key-Value pair RDDs (PairRDDs) are RDDs where each element is a `(key, value)` tuple, enabling aggregation, grouping, and join operations by key.

### Creating Pair RDDs
```python
# From regular RDD
lines = sc.textFile("/data/logs.txt")
pairs = lines.map(lambda line: (line.split(" ")[0], 1))  # (word, 1)

# From list
rdd = sc.parallelize([("Alice", 85), ("Bob", 92), ("Alice", 78)])
```

### Key Operations

| Operation | Description | Example |
|-----------|-------------|--------|
| `reduceByKey(f)` | Aggregate values per key | `rdd.reduceByKey(lambda a,b: a+b)` |
| `groupByKey()` | Group values per key | `rdd.groupByKey()` |
| `sortByKey()` | Sort by key | `rdd.sortByKey()` |
| `mapValues(f)` | Transform values only | `rdd.mapValues(lambda v: v*2)` |
| `keys()` / `values()` | Extract keys/values | `rdd.keys().collect()` |
| `join(rdd2)` | Inner join | `rdd1.join(rdd2)` |
| `cogroup(rdd2)` | Group both RDDs by key | `rdd1.cogroup(rdd2)` |
| `countByKey()` | Count per key | `rdd.countByKey()` |

### Code Example
```python
sales = sc.parallelize([
    ("electronics", 500), ("clothing", 200),
    ("electronics", 300), ("clothing", 150),
    ("electronics", 700)
])

# Total sales per category
totals = sales.reduceByKey(lambda a, b: a + b)
# [('electronics', 1500), ('clothing', 350)]

# Average per category
sum_count = sales.mapValues(lambda v: (v, 1)) \
    .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1]))
avg = sum_count.mapValues(lambda v: v[0]/v[1])
# [('electronics', 500.0), ('clothing', 175.0)]
```

### When to Use
- **Aggregations**: Sum, count, average by key
- **Joins**: Combining two datasets by key
- **Grouping**: All records for a given key
- **Log analysis**: Count errors by type

### Interview Tip
Pair RDDs are the **RDD-level equivalent** of DataFrame `groupBy` operations. In modern Spark, prefer DataFrames for most tasks (Catalyst optimization). Use Pair RDDs only when you need low-level control or work with unstructured data.

---

## Question 18
- [ ] Done

**Explain how to perform a join operation in Spark**

**Answer:**

### Join Types in Spark

| Join Type | Description | Keeps |
|-----------|-------------|-------|
| **Inner** | Matching keys only | Common keys |
| **Left outer** | All left + matching right | All left rows |
| **Right outer** | All right + matching left | All right rows |
| **Full outer** | All rows from both sides | Everything |
| **Left semi** | Left rows with match (no right columns) | Filter-like |
| **Left anti** | Left rows without match | Exclusion |
| **Cross** | Cartesian product | All combinations |

### DataFrame Joins
```python
employees = spark.createDataFrame([
    (1, "Alice", 101), (2, "Bob", 102), (3, "Charlie", 103)
], ["id", "name", "dept_id"])

departments = spark.createDataFrame([
    (101, "Engineering"), (102, "Marketing"), (104, "HR")
], ["dept_id", "dept_name"])

# Inner join
result = employees.join(departments, "dept_id")  # Shared column name

# Left outer join with different column names
result = employees.join(departments,
    employees.dept_id == departments.dept_id, "left")

# Multiple conditions
result = df1.join(df2,
    (df1.key1 == df2.key1) & (df1.key2 == df2.key2), "inner")
```

### RDD Joins
```python
emp_rdd = sc.parallelize([(101, "Alice"), (102, "Bob")])
dept_rdd = sc.parallelize([(101, "Eng"), (103, "HR")])

# Inner join
emp_rdd.join(dept_rdd).collect()
# [(101, ('Alice', 'Eng'))]

# Left outer join
emp_rdd.leftOuterJoin(dept_rdd).collect()
# [(101, ('Alice', 'Eng')), (102, ('Bob', None))]
```

### Join Strategies

| Strategy | When Used | Performance |
|----------|-----------|------------|
| **Broadcast hash join** | One side < 10 MB | Fastest (no shuffle) |
| **Sort-merge join** | Both sides large | Default for large tables |
| **Shuffle hash join** | Medium-sized tables | Alternative to sort-merge |

```python
from pyspark.sql.functions import broadcast
# Force broadcast join for small table
result = large_df.join(broadcast(small_df), "key")
```

### Interview Tip
The **broadcast join** is the biggest join optimization — it sends the small table to all executors, eliminating shuffle. Default threshold is 10 MB (`spark.sql.autoBroadcastJoinThreshold`). For large-large joins, ensure both sides are **pre-partitioned** by the join key (`repartition`) to minimize shuffle.

---

## Question 19
- [ ] Done

**Describe the concept of discretized streams (DStreams) in Spark**

**Answer:**

### Definition
DStreams (Discretized Streams) are Spark Streaming's abstraction representing a **continuous stream of data** as a sequence of RDDs, each containing data from a small time interval (micro-batch).

### How DStreams Work
```
Continuous data stream:
  ...data...data...data...data...data...
       |         |         |         |
       v         v         v         v
  [RDD @ t=0] [RDD @ t=1] [RDD @ t=2] [RDD @ t=3]
  (batch 1)   (batch 2)   (batch 3)   (batch 4)
  
  Each batch interval (e.g., 1 second) creates one RDD
```

### Code Example
```python
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

sc = SparkContext("local[2]", "DStreamExample")
ssc = StreamingContext(sc, batchDuration=1)  # 1-second batches

# Create DStream from socket
lines = ssc.socketTextStream("localhost", 9999)

# Transformations on DStream
words = lines.flatMap(lambda line: line.split(" "))
pairs = words.map(lambda word: (word, 1))
counts = pairs.reduceByKey(lambda a, b: a + b)

counts.pprint()  # Print to console

ssc.start()      # Start streaming
ssc.awaitTermination()
```

### DStream Operations

| Operation | Description |
|-----------|------------|
| `map(f)` | Transform each element |
| `flatMap(f)` | Map then flatten |
| `filter(f)` | Keep matching elements |
| `reduceByKey(f)` | Aggregate by key per batch |
| `window(duration, slide)` | Windowed computation |
| `updateStateByKey(f)` | Maintain running state |
| `transform(f)` | Apply arbitrary RDD operation |

### Window Operations
```python
# Count words in last 30 seconds, updated every 10 seconds
windowed_counts = pairs.reduceByKeyAndWindow(
    lambda a, b: a + b,    # Reduce function
    windowDuration=30,      # Window length
    slideDuration=10        # Slide interval
)
```

### Interview Tip
DStreams are the **legacy streaming API** in Spark (Spark Streaming). Modern Spark uses **Structured Streaming** (DataFrame-based), which provides exactly-once semantics, event-time processing, and Catalyst optimization. DStreams are still tested in interviews but are not recommended for new projects.

---

## Question 21
- [ ] Done

**What are the fault-tolerance mechanisms in Spark Streaming ?**

**Answer:**

### Fault Tolerance Layers

| Layer | Mechanism | What It Protects |
|-------|-----------|------------------|
| **RDD lineage** | Recompute lost partitions from DAG | Data loss from executor failure |
| **Checkpointing** | Save state to reliable storage | Driver failure recovery |
| **WAL** | Write-Ahead Log | Zero data loss for receivers |
| **Receiver replication** | Replicate received data | Receiver failure |
| **Kafka direct** | Read offsets from Kafka directly | Exactly-once with Kafka |

### RDD Lineage Recovery
```
Input DStream → map() → filter() → reduceByKey()

If executor fails mid-computation:
  - Spark knows the DAG (lineage)
  - Re-reads input data
  - Recomputes lost RDD partitions
```

### Checkpointing
```python
ssc = StreamingContext(sc, 1)
ssc.checkpoint("/checkpoint/dir")  # HDFS or S3 path

# Required for stateful operations
def update_func(new_values, running_count):
    return sum(new_values) + (running_count or 0)

running_counts = pairs.updateStateByKey(update_func)
```

### Write-Ahead Log (WAL)
```python
# Enable WAL for zero data loss
spark.conf.set("spark.streaming.receiver.writeAheadLog.enable", "true")

# WAL flow:
# 1. Receiver gets data from source
# 2. Data written to WAL (HDFS) before processing
# 3. If failure, replay from WAL
```

### Structured Streaming Guarantees
```python
# Structured Streaming provides exactly-once by default
query = df.writeStream \
    .format("parquet") \
    .option("checkpointLocation", "/checkpoint") \
    .outputMode("append") \
    .start()

# Checkpoint stores:
# - Offset ranges (what was read)
# - State data (for aggregations)
# - Committed output (what was written)
```

### Interview Tip
Structured Streaming achieves **exactly-once** end-to-end through checkpointing offsets + idempotent sinks. DStream-based streaming only guarantees **at-least-once** by default (WAL + checkpointing). For **exactly-once** with DStreams, you need the Kafka Direct approach (no receiver, read offsets directly).

---

## Question 22
- [ ] Done

**Explain watermarks and windowing operations in Structured Streaming**

**Answer:**

### Watermarks
A watermark tells Spark **how late data can arrive** before it's dropped. It allows Spark to clean up old state while handling late-arriving events.

```python
from pyspark.sql.functions import window, col

# Allow data up to 10 minutes late
events = spark.readStream.format("kafka").load()

result = events \
    .withWatermark("event_time", "10 minutes") \
    .groupBy(window("event_time", "5 minutes")) \
    .count()
```

### How Watermarks Work
```
Time axis:  10:00  10:05  10:10  10:15  10:20
                                  |
               Watermark = max_event_time - threshold
               = 10:20 - 10 min = 10:10
               
               ✅ Events with time >= 10:10 are processed
               ❌ Events with time < 10:10 are dropped (too late)

Benefit: Spark can discard state for windows before 10:10
         → prevents unbounded state growth
```

### Windowing Operations

| Window Type | Description | Use Case |
|-------------|-------------|----------|
| **Tumbling** | Fixed, non-overlapping windows | Hourly aggregates |
| **Sliding** | Overlapping windows | Moving averages |
| **Session** | Gap-based, variable size | User sessions |

### Code Examples
```python
# Tumbling window: 5-minute non-overlapping
result = events.groupBy(
    window("event_time", "5 minutes")
).agg(count("*").alias("events"))

# Sliding window: 10-minute window, sliding every 5 minutes
result = events.groupBy(
    window("event_time", "10 minutes", "5 minutes")
).agg(avg("value").alias("avg_value"))

# Session window (Spark 3.2+)
from pyspark.sql.functions import session_window
result = events.groupBy(
    session_window("event_time", "5 minutes"),
    "user_id"
).count()
```

### Tumbling vs Sliding
```
Tumbling (5 min):
|--W1--|--W2--|--W3--|--W4--|
0     5     10     15     20

Sliding (10 min window, 5 min slide):
|----W1----|
      |----W2----|
            |----W3----|
0     5     10     15     20
```

### Interview Tip
Watermarks solve two problems: **late data handling** and **state cleanup**. Without watermarks, Spark keeps ALL window state forever (memory grows unbounded). The watermark threshold is a tradeoff: too short = drop valid late data, too long = excessive memory usage. Session windows are the newest addition (Spark 3.2+) for user behavior analysis.

---

## Question 23
- [ ] Done

**What are some common Spark performance issues and how do you resolve them?**

**Answer:**

### Common Performance Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Data skew** | One task takes much longer | Salting keys, repartition |
| **Shuffle spill** | Slow stages, disk I/O | Increase memory, reduce partition count |
| **Small files** | Slow reads, many tasks | Coalesce, repartition |
| **Too many partitions** | Task scheduling overhead | `coalesce()` |
| **Too few partitions** | Underutilized cluster | `repartition()` |
| **Large broadcast** | OOM on driver | Increase `autoBroadcastJoinThreshold` or disable |
| **GC pressure** | Long GC pauses | Use `MEMORY_ONLY_SER`, off-heap |
| **Cartesian join** | Exploding data | Use proper join keys |

### Data Skew Solution (Salting)
```python
import pyspark.sql.functions as F

# Problem: one key has 90% of data
# Solution: add random salt to distribute
skewed_df = skewed_df.withColumn("salt", (F.rand() * 10).cast("int"))
skewed_df = skewed_df.withColumn("salted_key", F.concat("key", F.lit("_"), "salt"))

# Join on salted key, then aggregate
result = skewed_df.join(other_df_exploded, "salted_key") \
    .groupBy("original_key").agg(F.sum("value"))
```

### Partition Tuning
```python
# Check current partitions
df.rdd.getNumPartitions()

# Rule of thumb: 2-4 partitions per CPU core
df = df.repartition(200)          # Increase (with shuffle)
df = df.coalesce(50)              # Decrease (no shuffle)

# Set default parallelism
spark.conf.set("spark.sql.shuffle.partitions", 200)  # Default: 200
```

### Memory Configuration
```python
spark.conf.set("spark.executor.memory", "8g")
spark.conf.set("spark.executor.memoryOverhead", "2g")
spark.conf.set("spark.memory.fraction", "0.6")
spark.conf.set("spark.memory.storageFraction", "0.5")
```

### Interview Tip
The #1 performance issue is **data skew** — always check with `df.groupBy("key").count().orderBy(F.desc("count"))`. Three main solutions: 1) **Salting** (add random prefix), 2) **Broadcast join** (for skewed joins), 3) **Adaptive Query Execution** (AQE in Spark 3.0+ handles skew automatically with `spark.sql.adaptive.enabled=true`).

---

## Question 24
- [ ] Done

**Explain how you would monitor and log a Spark application**

**Answer:**

### Monitoring Tools

| Tool | Purpose | Access |
|------|---------|--------|
| **Spark UI** | Job/stage/task details, DAG visualization | `http://driver:4040` |
| **History Server** | View completed application UIs | `http://host:18080` |
| **YARN UI** | Cluster resource usage | `http://rm:8088` |
| **Metrics System** | JMX, Graphite, Prometheus metrics | Configuration-based |
| **Event Logs** | JSON logs of all Spark events | HDFS/S3 |

### Spark UI Tabs
```
Spark UI (port 4040):
├─ Jobs:       Overall job progress, failed stages
├─ Stages:     Task distribution, shuffle statistics
├─ Storage:    Cached RDDs/DataFrames, memory usage
├─ Environment: Configuration, classpath
├─ Executors:  Per-executor memory, GC time, tasks
├─ SQL:        Query plans, physical plans
└─ Streaming:  Batch processing times, scheduling delay
```

### Logging Configuration
```python
# Set log level
spark.sparkContext.setLogLevel("WARN")  # DEBUG, INFO, WARN, ERROR

# Custom logging in application
import logging
logger = logging.getLogger("my_spark_app")
logger.setLevel(logging.INFO)
logger.info("Processing batch: %s", batch_id)
```

### Metrics Configuration
```properties
# metrics.properties
*.sink.graphite.class=org.apache.spark.metrics.sink.GraphiteSink
*.sink.graphite.host=graphite-host
*.sink.graphite.port=2003
*.sink.graphite.period=10
*.sink.graphite.unit=seconds

# Prometheus sink (Spark 3.0+)
*.sink.prometheusServlet.class=org.apache.spark.metrics.sink.PrometheusServlet
*.sink.prometheusServlet.path=/metrics/prometheus
```

### Event Log for History Server
```python
spark = SparkSession.builder \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs:///spark-logs") \
    .config("spark.history.fs.logDirectory", "hdfs:///spark-logs") \
    .getOrCreate()
```

### Interview Tip
Key metrics to monitor: **shuffle read/write size** (indicates data movement), **GC time** (>10% means memory pressure), **task duration distribution** (skew detection), and **scheduling delay** (in streaming). Use **Spark UI's SQL tab** to check if Catalyst is using broadcast joins vs sort-merge joins.

---

## Question 25
- [ ] Done

**What is the role of partitioner objects in Spark and how do they affect performance?**

**Answer:**

### Definition
Partitioners determine **how data is distributed across partitions** in an RDD. They control data placement, which directly impacts shuffle performance and data locality.

### Built-in Partitioners

| Partitioner | Algorithm | Use Case |
|-------------|-----------|----------|
| **HashPartitioner** | `hash(key) % numPartitions` | Default, even distribution |
| **RangePartitioner** | Sampling + range boundaries | Sorted output, range queries |

### How Partitioners Affect Performance
```
Without proper partitioning (2 joins = 2 shuffles):
  rdd1.join(rdd2)   → shuffle both
  result.join(rdd3) → shuffle again

With co-partitioning (2 joins = 0 shuffles):
  rdd1 = rdd1.partitionBy(HashPartitioner(100))
  rdd2 = rdd2.partitionBy(HashPartitioner(100))
  rdd3 = rdd3.partitionBy(HashPartitioner(100))
  rdd1.join(rdd2).join(rdd3)  → no shuffle (already aligned)
```

### Code Example
```python
from pyspark import SparkContext

sc = SparkContext("local[*]", "Partitioner")

# Create pair RDD and partition
rdd = sc.parallelize([("a", 1), ("b", 2), ("c", 3), ("a", 4)])
partitioned = rdd.partitionBy(4)  # HashPartitioner with 4 partitions

# Check partitioner
print(partitioned.partitioner)  # HashPartitioner

# mapValues preserves partitioning
# map() does NOT preserve partitioning
still_partitioned = partitioned.mapValues(lambda v: v * 2)  # ✅
not_partitioned = partitioned.map(lambda x: (x[0], x[1] * 2))  # ❌
```

### DataFrame Partitioning
```python
# Repartition by column (uses HashPartitioner internally)
df = df.repartition(200, "user_id")

# Repartition by range (sorted partitions)
df = df.repartitionByRange(200, "date")

# Coalesce (reduce partitions without shuffle)
df = df.coalesce(50)
```

### Operations That Preserve/Break Partitioning

| Preserves | Breaks |
|-----------|--------|
| `mapValues` | `map` |
| `flatMapValues` | `flatMap` |
| `filter` | `repartition` |
| `join` (same partitioner) | `groupBy` (new keys) |

### Interview Tip
The biggest performance win from partitioning is **avoiding redundant shuffles** in iterative algorithms (e.g., PageRank). If two RDDs have the same `HashPartitioner`, joins are **partition-local** (no network transfer). Always use `mapValues` instead of `map` on pair RDDs to preserve the partitioner.

---

## Question 26
- [ ] Done

**What are the main features of Spark MLlib ?**

**Answer:**

### Definition
Spark MLlib is Spark's **distributed machine learning library**, providing scalable implementations of common ML algorithms and utilities.

### Key Features

| Feature | Description |
|---------|-------------|
| **ML Algorithms** | Classification, regression, clustering, collaborative filtering |
| **Pipelines** | End-to-end ML workflow (Transformer, Estimator, Pipeline) |
| **Feature Engineering** | Tokenizer, VectorAssembler, StandardScaler, PCA |
| **Model Selection** | CrossValidator, TrainValidationSplit, ParamGridBuilder |
| **Persistence** | Save/Load models to HDFS/S3 |
| **Distributed** | Scales to billions of rows across cluster |

### Supported Algorithms

| Category | Algorithms |
|----------|------------|
| **Classification** | LogisticRegression, RandomForest, GBT, NaiveBayes, SVM |
| **Regression** | LinearRegression, RandomForest, GBT, DecisionTree |
| **Clustering** | KMeans, BisectingKMeans, GMM, LDA |
| **Recommendation** | ALS (Alternating Least Squares) |
| **Dimensionality** | PCA, SVD |
| **NLP** | Tokenizer, Word2Vec, TF-IDF, CountVectorizer |

### Code Example
```python
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Prepare features
assembler = VectorAssembler(inputCols=["age", "income", "score"], outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Train model
lr = LogisticRegression(featuresCol="scaled_features", labelCol="label", maxIter=100)
model = lr.fit(train_df)

# Evaluate
predictions = model.transform(test_df)
evaluator = BinaryClassificationEvaluator(labelCol="label")
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc:.4f}")
```

### Two APIs

| API | Package | Status |
|-----|---------|--------|
| **DataFrame-based** | `pyspark.ml` | Current (recommended) |
| **RDD-based** | `pyspark.mllib` | Legacy (maintenance only) |

### Interview Tip
MLlib's key advantage over scikit-learn is **horizontal scaling** — it can train on datasets that don't fit in memory of a single machine. The tradeoff: fewer algorithms and less flexibility than scikit-learn. For small-medium data, use scikit-learn; for big data, use MLlib.

---

## Question 27
- [ ] Done

**How does Spark MLlib handle machine learning pipelines ?**

**Answer:**

### Pipeline Components

| Component | Role | Example |
|-----------|------|--------|
| **Transformer** | Transforms DataFrame (has `.transform()`) | VectorAssembler, StandardScaler (fitted) |
| **Estimator** | Learns from data (has `.fit()`) → produces Transformer | LogisticRegression, StandardScaler |
| **Pipeline** | Chain of Transformers and Estimators | End-to-end ML workflow |
| **PipelineModel** | Fitted pipeline (all stages are Transformers) | Ready for prediction |
| **ParamGrid** | Hyperparameter search space | Grid of parameter combinations |

### Pipeline Architecture
```
Training:
  DataFrame → [Tokenizer] → [HashingTF] → [LogisticRegression.fit()]
                Transformer   Transformer    Estimator
                                              ↓
                                          PipelineModel

Prediction:
  New Data → [Tokenizer] → [HashingTF] → [LogisticRegressionModel.transform()]
              Transformer   Transformer    Transformer (fitted model)
```

### Code Example
```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Define pipeline stages
indexer = StringIndexer(inputCol="category", outputCol="label")
assembler = VectorAssembler(inputCols=["f1", "f2", "f3"], outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="label")

# Build pipeline
pipeline = Pipeline(stages=[indexer, assembler, scaler, rf])

# Hyperparameter tuning
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100, 200]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .build()

crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=MulticlassClassificationEvaluator(),
    numFolds=5
)

# Train and select best model
cv_model = crossval.fit(train_df)
best_model = cv_model.bestModel

# Save/Load
best_model.save("/models/best_rf_pipeline")
loaded = PipelineModel.load("/models/best_rf_pipeline")
```

### Interview Tip
Pipelines ensure **reproducibility** and prevent data leakage — feature scaling is fitted only on training data and applied consistently to test data. The `CrossValidator` wraps the entire pipeline, so preprocessing is re-fitted on each fold (correct behavior). This is equivalent to scikit-learn's `Pipeline` but distributed.

---

## Question 28
- [ ] Done

**Describe a use case for MLlib’s collaborative filtering algorithms**

**Answer:**

### Use Case: Movie Recommendation System

| Component | Details |
|-----------|--------|
| **Data** | User ratings (userId, movieId, rating) |
| **Algorithm** | ALS (Alternating Least Squares) |
| **Input** | Sparse user-movie rating matrix |
| **Output** | Predicted ratings for unseen movies |

### How ALS Works
```
Rating Matrix (sparse):
       Movie1  Movie2  Movie3
User1  [ 5      ?       3   ]
User2  [ ?      4       ?   ]
User3  [ 3      ?       5   ]

ALS decomposes into User Factors x Item Factors (rank k)
Alternates: fix users -> solve items, fix items -> solve users
```

### Code Example
```python
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

ratings = spark.read.csv("/data/ratings.csv", header=True, inferSchema=True)
train, test = ratings.randomSplit([0.8, 0.2], seed=42)

als = ALS(
    userCol="userId", itemCol="movieId", ratingCol="rating",
    rank=10, maxIter=15, regParam=0.1,
    coldStartStrategy="drop"
)
model = als.fit(train)

predictions = model.transform(test)
rmse = RegressionEvaluator(metricName="rmse", labelCol="rating").evaluate(predictions)
print(f"RMSE: {rmse:.4f}")

user_recs = model.recommendForAllUsers(10)
```

### Real-World Applications
- **Netflix/Spotify**: Content recommendations
- **E-commerce**: Product suggestions
- **News/Social**: Personalized feed ranking

### Interview Tip
ALS handles the **cold start problem** poorly. Use `coldStartStrategy="drop"` for evaluation. In production, combine CF with content-based features (hybrid approach). ALS scales to **billions of interactions** on Spark.

---

## Question 29
- [ ] Done

**Explain the difference between Spark MLlib and external machine learning libraries**

**Answer:**

### Comparison

| Feature | Spark MLlib | Scikit-learn | TensorFlow/PyTorch |
|---------|------------|-------------|-------------------|
| **Scale** | Distributed (TB+) | Single machine (GB) | GPU-based (GB-TB) |
| **Algorithms** | ~30 algorithms | ~100+ algorithms | Deep learning focus |
| **Speed (small data)** | Slower (overhead) | Fastest | Medium |
| **Speed (big data)** | Fastest (distributed) | OOM errors | Needs distributed setup |
| **Ease of use** | Medium | Easiest | Complex |
| **Deep learning** | Limited | None | Full support |
| **Deployment** | Spark cluster | Any server | Serving frameworks |

### When to Use Which

| Scenario | Best Choice | Reason |
|----------|------------|--------|
| Data fits in memory (<10 GB) | Scikit-learn | Faster, more algorithms |
| Big data (TB+) | Spark MLlib | Distributed processing |
| Deep learning (NLP, CV) | TensorFlow/PyTorch | GPU support, architectures |
| Real-time inference | Scikit-learn or ONNX | Low latency |
| Streaming ML | Spark MLlib | Native Spark integration |
| AutoML | H2O, AutoGluon | Automated model selection |

### Hybrid Approach
```python
# Common pattern: preprocess with Spark, train with scikit-learn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# Step 1: Preprocess big data with Spark
df = spark.read.parquet("/data/large_dataset")
processed = df.filter(...).groupBy(...).agg(...)  # Distributed

# Step 2: Collect to Pandas (if result fits in memory)
pandas_df = processed.toPandas()

# Step 3: Train with scikit-learn
model = GradientBoostingClassifier()
model.fit(pandas_df[features], pandas_df["label"])

# Alternative: Use pandas_udf for distributed scikit-learn
from pyspark.sql.functions import pandas_udf
@pandas_udf("double")
def predict_udf(features: pd.Series) -> pd.Series:
    return pd.Series(model.predict(features))
```

### Interview Tip
The most common real-world pattern is **Spark for ETL + scikit-learn/XGBoost for modeling**. MLlib is ideal when data is too large for a single machine. For deep learning at scale, use **Spark + Horovod** (distributed training) or **Spark + TensorFlow** via `spark-tensorflow-connector`.

---

## Question 30
- [ ] Done

**Explain a big data analytics project where Spark would be a better choice than other big data technologies and why**

**Answer:**

### Project: Real-Time Fraud Detection for a Financial Institution

| Aspect | Details |
|--------|--------|
| **Data volume** | 500 million transactions/day (50 TB/month) |
| **Latency** | Sub-second fraud alerts |
| **ML** | Real-time model scoring + batch retraining |
| **Sources** | Kafka streams, HDFS historical data, JDBC databases |

### Why Spark Over Alternatives

| Technology | Limitation | Why Spark Wins |
|-----------|-----------|----------------|
| **Hadoop MapReduce** | Batch only, high latency | Spark: in-memory + streaming |
| **Flink** | Smaller ecosystem, fewer ML libraries | Spark: MLlib + unified batch/stream |
| **Hive** | SQL-only, slow (disk-based) | Spark SQL: 100x faster, richer API |
| **Storm** | No batch, limited ML | Spark: unified batch + stream + ML |
| **Pandas** | Single machine, no streaming | Spark: distributed, handles 50 TB |

### Architecture
```
Kafka (transactions) → Structured Streaming → ML Model Scoring → Alert Service
                                 ↓
                        Feature Store (Delta Lake)
                                 ↓
                    Batch Retraining (MLlib) → Updated Model
```

### Implementation
```python
# 1. Stream transactions from Kafka
transactions = spark.readStream \
    .format("kafka") \
    .option("subscribe", "transactions") \
    .load()

# 2. Feature engineering
features = transactions \
    .withColumn("hour", F.hour("timestamp")) \
    .withColumn("amount_zscore", ...) \
    .withColumn("location_risk", ...)

# 3. Score with pre-trained model
model = PipelineModel.load("/models/fraud_detector")
scored = model.transform(features)

# 4. Alert on high-risk transactions
alerts = scored.filter("prediction = 1 AND probability > 0.8")
alerts.writeStream \
    .format("kafka") \
    .option("topic", "fraud_alerts") \
    .start()

# 5. Batch retraining (nightly)
historical = spark.read.parquet("/data/labeled_transactions")
new_model = pipeline.fit(historical)
new_model.save("/models/fraud_detector_v2")
```

### Interview Tip
Spark's **unified engine** is the key selling point — one framework for batch ETL, streaming, ML, and SQL. No need to maintain separate systems (Hadoop for batch + Storm for streaming + separate ML platform). This reduces operational complexity and enables **feature consistency** between training (batch) and inference (streaming).

---

## Question 31
- [ ] Done

**Explain how Dynamic Resource Allocation works in Spark**

**Answer:**

### Definition
Dynamic Resource Allocation (DRA) allows Spark to **automatically scale executors** up and down based on workload, instead of using a fixed number throughout the application's lifetime.

### How It Works
```
Time →
Executors:
  Stage 1 (heavy): [E1][E2][E3][E4][E5]   ← 5 executors (scaled up)
  Idle period:      [E1][E2]                ← 3 released
  Stage 2 (light):  [E1][E2][E3]            ← 1 added back
  Idle period:      [E1]                    ← 2 released
```

### Configuration
```python
spark = SparkSession.builder \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.dynamicAllocation.minExecutors", 2) \
    .config("spark.dynamicAllocation.maxExecutors", 100) \
    .config("spark.dynamicAllocation.initialExecutors", 5) \
    .config("spark.dynamicAllocation.executorIdleTimeout", "60s") \
    .config("spark.dynamicAllocation.schedulerBacklogTimeout", "1s") \
    .config("spark.shuffle.service.enabled", "true") \
    .getOrCreate()
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `minExecutors` | 0 | Minimum executors to keep |
| `maxExecutors` | ∞ | Maximum executors allowed |
| `initialExecutors` | minExecutors | Starting count |
| `executorIdleTimeout` | 60s | Remove idle executor after this |
| `schedulerBacklogTimeout` | 1s | Add executors when tasks are pending |
| `cachedExecutorIdleTimeout` | ∞ | Idle timeout for executors with cached data |

### External Shuffle Service
```
Without shuffle service:
  Executor removed → shuffle data lost → must recompute

With shuffle service (required for DRA):
  Executor removed → shuffle data preserved by NodeManager → no recompute
```

### Benefits
- **Cost savings**: Release unused resources (especially on cloud)
- **Multi-tenancy**: Fair sharing of cluster resources
- **Burst handling**: Scale up for peak workloads
- **Idle efficiency**: Don't hold resources during idle periods

### Interview Tip
DRA requires the **External Shuffle Service** (`spark.shuffle.service.enabled=true`) because when an executor is removed, its shuffle files must survive. Without it, removing an executor means losing shuffle data and recomputing. On **Kubernetes**, use shuffle tracking instead (`spark.dynamicAllocation.shuffleTracking.enabled=true`).

---

## Question 32
- [ ] Done

**What are the current research areas or challenges in the Apache Spark ecosystem?**

**Answer:**

### Current Challenges and Research Areas

| Area | Challenge | Current Efforts |
|------|-----------|----------------|
| **GPU acceleration** | CPUs limit ML/DL workloads | RAPIDS (GPU DataFrames), Spark 3.x GPU scheduling |
| **Cloud-native** | Kubernetes integration maturity | Spark on K8s (operator, volcano scheduler) |
| **Streaming** | True real-time (not micro-batch) | Continuous processing mode (experimental) |
| **Data lakehouse** | Unified batch/stream storage | Delta Lake, Apache Iceberg, Hudi |
| **Adaptive execution** | Runtime optimization | AQE (Adaptive Query Execution) in Spark 3.0+ |
| **Cost optimization** | Cloud resource efficiency | Spot instances, serverless Spark |
| **Security** | Fine-grained access control | Column-level encryption, Apache Ranger |

### Emerging Trends

1. **Data Lakehouse Architecture**
```python
# Delta Lake: ACID transactions on data lake
df.write.format("delta").mode("append").save("/data/delta_table")

# Time travel
spark.read.format("delta").option("versionAsOf", 5).load("/data/delta_table")
```

2. **Spark Connect (3.4+)**: Thin client protocol for remote Spark access
3. **Photon Engine**: Databricks' C++ vectorized engine (10x faster)
4. **Project Zen**: Python-first Spark experience
5. **Serverless Spark**: Auto-scaling without cluster management

### Performance Challenges

| Problem | Description |
|---------|-------------|
| **Python serialization** | PySpark UDFs serialize data between JVM and Python |
| **Shuffle bottleneck** | Network I/O during wide transformations |
| **Small file problem** | Many small Parquet files from streaming |
| **Skew handling** | Uneven data distribution across partitions |
| **Memory management** | OOM errors with large joins/aggregations |

### Interview Tip
The biggest current trend is the **data lakehouse** (Delta Lake, Iceberg) — combining data lake flexibility with data warehouse reliability (ACID transactions, schema enforcement, time travel). Also mention **Spark Connect** (Spark 3.4+) as the future of client-server Spark architecture, enabling thin clients in any language.

---

## Question 33
- [ ] Done

**How does Spark support deep learning workloads and integration with popular deep learning frameworks ?**

**Answer:**

### Integration Approaches

| Approach | Framework | Description |
|----------|-----------|-------------|
| **Spark + Horovod** | TensorFlow/PyTorch | Distributed training via MPI |
| **Spark + TensorFlow** | TensorFlow | spark-tensorflow-connector |
| **Spark + PyTorch** | PyTorch | TorchDistributor (Spark 3.4+) |
| **Spark DL Pipelines** | TensorFlow/Keras | Deep learning in MLlib pipelines |
| **RAPIDS** | GPU | GPU-accelerated Spark |
| **Pandas UDFs** | Any | Apply DL models in PySpark |

### TorchDistributor (Spark 3.4+)
```python
from pyspark.ml.torch.distributor import TorchDistributor
import torch
import torch.nn as nn

def train_fn():
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    # Training loop here...
    return model

# Distribute training across Spark executors
distributor = TorchDistributor(
    num_processes=4,
    local_mode=False,
    use_gpu=True
)
model = distributor.run(train_fn)
```

### Horovod Integration
```python
import horovod.spark.torch as hvd_spark

def train_fn(epoch):
    import horovod.torch as hvd
    hvd.init()
    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters())
    optimizer = hvd.DistributedOptimizer(optimizer)
    # Training loop...

# Run on Spark cluster
hvd_spark.run(train_fn, args=(10,), num_proc=4)
```

### Inference at Scale with Pandas UDFs
```python
import pandas as pd
from pyspark.sql.functions import pandas_udf
import torch

# Load model once per executor (broadcast)
model_bc = spark.sparkContext.broadcast(torch.load("model.pt"))

@pandas_udf("float")
def predict(features: pd.Series) -> pd.Series:
    model = model_bc.value
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(features.tolist())
        predictions = model(inputs).squeeze().numpy()
    return pd.Series(predictions)

# Apply to DataFrame
df = df.withColumn("prediction", predict("features"))
```

### Common Patterns

| Pattern | When to Use |
|---------|-----------|
| **ETL with Spark + Train with DL framework** | Most common; Spark preprocesses, GPU trains |
| **Distributed training** | Model too large or data too big for one GPU |
| **Batch inference** | Score millions of records with trained model |
| **Transfer learning** | Fine-tune pre-trained models on Spark data |

### Interview Tip
Spark is typically used for **data preparation and batch inference**, not training. Training happens on dedicated GPU clusters (TensorFlow/PyTorch). The exception is **Horovod/TorchDistributor** for distributed training across Spark executors with GPUs. The most practical pattern is: **Spark ETL → Feature Store → GPU Training → Spark Batch Inference**.

---

## Question 34
- [ ] Done

**Explain how Apache Spark differs from Hadoop MapReduce**

*Answer to be added.*

---

## Question 35
- [ ] Done

**Describe the concept of RDDs (Resilient Distributed Datasets) in Spark**

*Answer to be added.*

---

## Question 36
- [ ] Done

**What are DataFrames in Spark and how do they compare to RDDs ?**

*Answer to be added.*

---

## Question 37
- [ ] Done

**What is lazy evaluation and how does it benefit Spark computations ?**

*Answer to be added.*

---

## Question 38
- [ ] Done

**What is the role of Spark Driver and Executors ?**

*Answer to be added.*

---

## Question 39
- [ ] Done

**How does Spark’s DAG (Directed Acyclic Graph) Scheduler work?**

*Answer to be added.*

---

## Question 40
- [ ] Done

**Explain the concept of a Spark Session and its purpose**

*Answer to be added.*

---

## Question 41
- [ ] Done

**How does Spark integrate with Hadoop components like HDFS and YARN ?**

*Answer to be added.*

---

## Question 42
- [ ] Done

**What is the significance of the Catalyst optimizer in Spark SQL ?**

*Answer to be added.*

---

## Question 43
- [ ] Done

**Illustrate the differences between map and flatMap functions in Spark**

*Answer to be added.*

---

## Question 44
- [ ] Done

**Detail how window functions work in Spark SQL**

*Answer to be added.*

---

## Question 45
- [ ] Done

**How does Structured Streaming differ from DStream-based streaming ?**

*Answer to be added.*

---

## Question 46
- [ ] Done

**How do you handle late data and stateful processing in Spark Streaming?**

*Answer to be added.*

---

## Question 47
- [ ] Done

**Discuss the advancements in Spark 3.x and their impact on big data processing**

*Answer to be added.*

---

## Question 48
- [ ] Done

**How do you implement custom aggregations in Spark?**

*Answer to be added.*

---
