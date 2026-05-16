# Data Engineer Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What is data modeling and why is it important?**

**Answer:**

### Definition
Data modeling is the process of creating a visual representation of data structures, relationships, and rules that govern data within an organization. It serves as a blueprint for database design.

### Core Concepts
- **Entities**: Objects about which data is stored (Customer, Order)
- **Attributes**: Properties of entities (name, date, amount)
- **Relationships**: How entities connect (Customer places Order)
- **Constraints**: Rules enforcing data integrity

### Why Important
| Reason | Benefit |
|--------|---------|
| Clarity | Shared understanding of data |
| Consistency | Standardized data definitions |
| Efficiency | Optimized queries and storage |
| Scalability | Supports growth |
| Data Quality | Reduces redundancy and errors |

### Interview Tip
Good data modeling prevents costly changes later. "Measure twice, cut once."

---

## Question 2
- [ ] Done

**Explain the difference between conceptual, logical, and physical data models.**

**Answer:**

### Overview

| Level | Purpose | Audience | Detail |
|-------|---------|----------|--------|
| Conceptual | Business view | Business users | High-level entities |
| Logical | Technical specification | Analysts/Architects | Detailed attributes, keys |
| Physical | Implementation | DBAs/Developers | Tables, columns, types |

### Conceptual Model
- What data the business needs
- Entity names and relationships only
- No technical details
- Example: "Customer places Order"

### Logical Model
- How data should be organized
- Attributes, primary/foreign keys
- Normalized structure
- Database-agnostic

### Physical Model
- How data is stored
- Table/column names, data types
- Indexes, partitions
- Specific to database platform

### Interview Tip
Start conceptual (business), then logical (design), then physical (implementation).

---

## Question 3
- [ ] Done

**What are the key steps in the data modeling process?**

**Answer:**

### Steps

1. **Gather Requirements**
   - Understand business needs
   - Identify data sources
   - Define use cases

2. **Identify Entities**
   - Core objects (Customer, Product, Transaction)
   - Define attributes for each

3. **Define Relationships**
   - One-to-one, one-to-many, many-to-many
   - Cardinality constraints

4. **Create Conceptual Model**
   - High-level ER diagram
   - Business stakeholder review

5. **Develop Logical Model**
   - Add all attributes
   - Define primary/foreign keys
   - Apply normalization

6. **Create Physical Model**
   - Map to database platform
   - Define data types, indexes
   - Optimize for performance

7. **Review and Iterate**
   - Validate with stakeholders
   - Test with sample data

### Interview Tip
Emphasize iterative nature and stakeholder collaboration throughout.

---

## Question 4
- [ ] Done

**Describe the different types of relationships in a relational database.**

**Answer:**

### Relationship Types

| Type | Description | Example |
|------|-------------|---------|
| One-to-One (1:1) | One record relates to exactly one | Person ↔ Passport |
| One-to-Many (1:N) | One record relates to many | Customer → Orders |
| Many-to-Many (M:N) | Many relate to many | Students ↔ Courses |

### Implementation

**One-to-One:**
- Foreign key in either table
- Often merged into one table

**One-to-Many:**
- Foreign key in "many" side
- Most common relationship

**Many-to-Many:**
- Requires junction/bridge table
- Junction table has FKs to both tables

### Example (M:N)
```
Students         Enrollments        Courses
---------        -----------        --------
student_id  ←─── student_id
                 course_id    ───→  course_id
```

### Interview Tip
Many-to-many always needs a junction table in relational databases.

---

## Question 5
- [ ] Done

**What is normalization and why is it used in database design?**

**Answer:**

### Definition
Normalization is the process of organizing data to reduce redundancy and dependency by dividing tables into smaller, related tables following normal forms.

### Normal Forms

| Form | Rule | Goal |
|------|------|------|
| 1NF | Atomic values, no repeating groups | Eliminate repeating groups |
| 2NF | 1NF + No partial dependencies | Remove partial dependencies |
| 3NF | 2NF + No transitive dependencies | Remove transitive dependencies |
| BCNF | 3NF + Every determinant is a key | Stricter 3NF |

### Benefits
- Reduces data redundancy
- Prevents update anomalies
- Ensures data integrity
- Saves storage space

### Trade-offs
| Pros | Cons |
|------|------|
| Less redundancy | More joins needed |
| Better data integrity | May slow queries |
| Easier updates | More complex queries |

### When to Denormalize
- Read-heavy analytics workloads
- Performance-critical applications
- Data warehousing (star schema)

### Interview Tip
Know the first 3 normal forms. Also understand when denormalization is appropriate.

---

## Question 6
- [ ] Done

**Explain the difference between OLTP and OLAP systems.**

**Answer:**

### Comparison

| Aspect | OLTP | OLAP |
|--------|------|------|
| Purpose | Transaction processing | Analytics/Reporting |
| Operations | INSERT, UPDATE, DELETE | SELECT (aggregations) |
| Data | Current, real-time | Historical, aggregated |
| Users | Clerks, customers | Analysts, managers |
| Queries | Simple, short | Complex, long-running |
| Schema | Normalized (3NF) | Denormalized (star/snowflake) |
| Example | Banking transactions | Sales analytics dashboard |

### OLTP Characteristics
- High volume of small transactions
- ACID compliance critical
- Millisecond response time
- Row-level operations

### OLAP Characteristics
- Complex aggregations
- Batch updates (nightly/weekly)
- Query response: seconds to minutes
- Column-level operations (often columnar storage)

### Interview Tip
"OLTP for operations, OLAP for analysis." Know examples of each.

---

## Question 7
- [ ] Done

**What is a star schema and when would you use it?**

**Answer:**

### Definition
A star schema is a data warehouse design with a central fact table connected to multiple dimension tables, forming a star-like structure.

### Structure
```
        [Dim_Date]
             |
[Dim_Product]──[Fact_Sales]──[Dim_Store]
             |
        [Dim_Customer]
```

### Components

| Component | Description | Example |
|-----------|-------------|---------|
| Fact Table | Metrics/measures, FKs | sales_amount, quantity |
| Dimension Table | Descriptive attributes | product_name, date, location |

### When to Use
- OLAP/Data warehouse environments
- Business intelligence reporting
- Dashboard and analytics queries
- When query performance matters

### Pros and Cons
| Pros | Cons |
|------|------|
| Simple queries | Data redundancy |
| Fast aggregations | Storage overhead |
| Easy to understand | Update complexity |

### Interview Tip
Star schema prioritizes query performance over normalization. It's the standard for data warehousing.

---

## Question 8
- [ ] Done

**Describe the concept of slowly changing dimensions (SCDs) in data warehousing.**

**Answer:**

### Definition
SCDs are dimension tables where attribute values change over time, requiring special handling to track historical data while maintaining data integrity.

### SCD Types

| Type | Method | History | Example |
|------|--------|---------|---------|
| Type 0 | No change | None | Retain original value |
| Type 1 | Overwrite | Lost | Update address, lose old |
| Type 2 | New row | Kept | New row with dates |
| Type 3 | New column | Limited | current_addr, previous_addr |
| Type 4 | History table | Separate | Current table + history table |
| Type 6 | Hybrid | Full | Combines 1, 2, 3 |

### Type 2 Example
```
customer_id | name    | address | start_date | end_date   | current
1001        | John    | NYC     | 2020-01-01 | 2022-06-30 | N
1001        | John    | LA      | 2022-07-01 | 9999-12-31 | Y
```

### When to Use Each
- **Type 1**: History not needed
- **Type 2**: Full history required (most common)
- **Type 3**: Limited history (only previous)

### Interview Tip
Type 2 is most common for analytics. Know how to implement with start/end dates and current flag.

---

## Question 9
- [ ] Done

**What is a fact table and how does it differ from a dimension table?**

**Answer:**

### Comparison

| Aspect | Fact Table | Dimension Table |
|--------|------------|-----------------|
| Contains | Measures/metrics | Descriptive attributes |
| Data | Numeric, additive | Text, categorical |
| Size | Very large (millions/billions) | Smaller (thousands) |
| Keys | Surrogate + FKs to dimensions | Surrogate + natural key |
| Example | sales_amount, quantity | product_name, date, store_name |
| Changes | Append-only (usually) | Updated (SCD) |

### Fact Table Types
| Type | Description | Example |
|------|-------------|---------|
| Transaction | Individual events | Each sale |
| Periodic Snapshot | State at intervals | Daily balance |
| Accumulating Snapshot | Lifecycle tracking | Order lifecycle |

### Fact Table Measures
- **Additive**: Can sum across all dimensions (revenue)
- **Semi-additive**: Some dimensions only (account balance)
- **Non-additive**: Cannot sum (ratio, percentage)

### Interview Tip
Facts answer "how much/many" (metrics), dimensions answer "who/what/where/when" (context).

---

## Question 10
- [ ] Done

**Explain the purpose of surrogate keys in data modeling.**

**Answer:**

### Definition
A surrogate key is a system-generated unique identifier (usually auto-increment integer) used as the primary key instead of natural business keys.

### Why Use Surrogate Keys

| Reason | Explanation |
|--------|-------------|
| Stability | Natural keys can change (email, SSN) |
| Performance | Integer keys are faster to join |
| Simplicity | Single column vs composite keys |
| Integration | Unified keys across sources |
| History | SCD Type 2 requires unique row IDs |

### Surrogate vs Natural Keys

| Surrogate | Natural |
|-----------|---------|
| System-generated | Business meaning |
| Integer (small) | Can be large (email, SSN) |
| Never changes | May change |
| No meaning | Has meaning |

### Implementation
```sql
customer_id INT IDENTITY(1,1) PRIMARY KEY,  -- Surrogate
customer_email VARCHAR(255) UNIQUE           -- Natural key kept for lookup
```

### Interview Tip
Always use surrogate keys in data warehouses. Keep natural keys as attributes for business lookups.

---

## Question 11
- [ ] Done

**What is a data warehouse and its key characteristics?**

**Answer:**

### Definition
A data warehouse is a centralized repository of integrated data from multiple sources, optimized for analytics, reporting, and business intelligence.

### Key Characteristics (Bill Inmon's Definition)
- **Subject-oriented**: Organized by business subjects (sales, customers)
- **Integrated**: Consistent format from multiple sources
- **Non-volatile**: Data is stable, not frequently updated
- **Time-variant**: Historical data maintained

### Architecture
```
Sources → ETL → Staging → Data Warehouse → Data Marts → BI Tools
```

### Components
| Component | Purpose |
|-----------|---------|
| Staging Area | Raw data landing zone |
| Data Warehouse | Integrated, enterprise-wide |
| Data Marts | Subject-specific subsets |
| ETL | Data movement and transformation |

### Modern Approaches
- Cloud data warehouses (Snowflake, BigQuery, Redshift)
- Data lakehouse (combines lake + warehouse)
- Real-time streaming ingestion

### Interview Tip
Understand the difference between Kimball (dimensional) and Inmon (normalized) approaches.

---

## Question 12
- [ ] Done

**Explain the ETL (Extract, Transform, Load) process and its stages.**

**Answer:**

### Definition
ETL is the process of extracting data from source systems, transforming it to meet business requirements, and loading it into a target data warehouse.

### Stages

**1. Extract**
- Connect to source systems
- Pull data (full or incremental)
- Sources: databases, APIs, files, streams
- Handle different formats

**2. Transform**
- Clean data (handle nulls, duplicates)
- Standardize formats (dates, names)
- Apply business rules
- Aggregate/calculate derived values
- Join data from multiple sources

**3. Load**
- Insert into target tables
- Update existing records
- Handle conflicts and errors
- Full load vs incremental

### ETL vs ELT
| ETL | ELT |
|-----|-----|
| Transform before loading | Transform in target system |
| Traditional approach | Modern cloud approach |
| Limited by ETL server | Leverages warehouse compute |

### Common Tools
- Apache Spark, Airflow
- dbt (transform)
- Informatica, Talend
- Cloud-native: Glue, Data Factory

### Interview Tip
ELT is increasingly popular with cloud warehouses due to their scalable compute.

---

## Question 13
- [ ] Done

**What are the common challenges faced during ETL processes?**

**Answer:**

### Key Challenges

| Challenge | Description |
|-----------|-------------|
| Data Quality | Missing, invalid, inconsistent data |
| Schema Changes | Source system modifications |
| Volume/Performance | Processing large datasets |
| Latency | Meeting SLA requirements |
| Error Handling | Graceful failure recovery |
| Data Integration | Merging disparate sources |

### Data Quality Issues
- Missing values
- Duplicates
- Format inconsistencies
- Invalid values
- Encoding problems

### Solutions
| Problem | Solution |
|---------|----------|
| Schema changes | Schema evolution, contracts |
| Performance | Parallel processing, partitioning |
| Data quality | Validation rules, monitoring |
| Errors | Retry logic, dead-letter queues |
| Complexity | Modular design, testing |

### Best Practices
- Implement data quality checks
- Log extensively
- Build idempotent processes
- Use version control
- Monitor and alert

### Interview Tip
Always mention data quality as the #1 challenge. Know specific examples and solutions.

---

## Question 14
- [ ] Done

**Describe the difference between full load and incremental load in ETL.**

**Answer:**

### Comparison

| Aspect | Full Load | Incremental Load |
|--------|-----------|------------------|
| Data Volume | All data | Only changed data |
| Frequency | Less frequent | More frequent |
| Complexity | Simple | More complex |
| Performance | Slow for large data | Faster |
| Use Case | Initial load, small tables | Large tables, regular updates |

### Incremental Load Methods

| Method | How It Works |
|--------|--------------|
| Timestamp-based | Use modified_date column |
| Sequence-based | Track last processed ID |
| CDC (Change Data Capture) | Log-based detection |
| Hash comparison | Compare row hashes |

### CDC Approaches
- Log-based: Read database transaction logs
- Trigger-based: Database triggers capture changes
- Timestamp-based: Query by modification time

### Choosing Approach
| Scenario | Approach |
|----------|----------|
| Small reference tables | Full load |
| Large transaction tables | Incremental |
| Real-time needs | CDC/Streaming |
| Initial migration | Full load |

### Interview Tip
Incremental loads are preferred for production. Know how to track "what changed" (timestamp, CDC, etc.).

---

## Question 15
- [ ] Done

**What is data staging and why is it important in ETL?**

**Answer:**

### Definition
A staging area is an intermediate storage location where raw data is temporarily stored before being transformed and loaded into the data warehouse.

### Purpose

| Purpose | Benefit |
|---------|---------|
| Landing zone | Raw data preservation |
| Decoupling | Separate extract from transform |
| Performance | Reduce source system load |
| Recovery | Reprocess without re-extraction |
| Auditing | Track original data |

### Staging Area Characteristics
- Temporary storage
- Minimal transformation (1:1 copy or near)
- Source system structure preserved
- Cleared after successful load

### ETL Flow
```
Source → Extract → Staging → Transform → Target (DW)
```

### Best Practices
- Keep staging data for recovery period
- Log extraction metadata
- Use consistent naming conventions
- Partition by load date

### Interview Tip
Staging enables restart/recovery without hitting sources again. Critical for production reliability.

---

## Question 16
- [ ] Done

**Explain the concept of data lineage and its significance in data warehousing.**

**Answer:**

### Definition
Data lineage tracks the complete journey of data from source to destination, documenting transformations, movements, and dependencies at each stage.

### What It Shows
- Where data originated
- What transformations were applied
- Where data is consumed
- Dependencies between datasets

### Importance

| Benefit | Description |
|---------|-------------|
| Debugging | Trace errors to source |
| Impact Analysis | Understand downstream effects |
| Compliance | Audit requirements (GDPR, SOX) |
| Trust | Confidence in data quality |
| Documentation | Understand the system |

### Levels of Lineage
| Level | Granularity |
|-------|-------------|
| Table/Dataset | Which tables feed which |
| Column | Column-level mapping |
| Row | Individual record tracking |

### Tools
- Apache Atlas
- DataHub
- Collibra
- dbt (model lineage)

### Interview Tip
Data lineage is critical for compliance and debugging. Modern data catalogs provide automated lineage tracking.

---

## Question 17
- [ ] Done

**What are the benefits of using a data warehouse?**

**Answer:**

### Key Benefits

| Benefit | Description |
|---------|-------------|
| Centralized Data | Single source of truth |
| Historical Analysis | Time-series data preserved |
| Improved Performance | Optimized for analytics |
| Data Quality | Cleaned and standardized |
| Decision Support | Enable BI and reporting |
| Reduced Load | Offload from operational systems |

### Business Benefits
- Faster insights and decision-making
- Consistent metrics across organization
- Self-service analytics
- Trend analysis and forecasting

### Technical Benefits
- Query performance optimization
- Separation of concerns (OLTP vs OLAP)
- Data governance and security
- Scalable architecture

### ROI Factors
- Reduced ad-hoc query time
- Better business decisions
- Compliance and auditing
- Reduced IT support burden

### Interview Tip
Frame benefits in terms of business value, not just technical features.

---

## Question 18
- [ ] Done

**Describe the role of data quality in ETL processes.**

**Answer:**

### Definition
Data quality in ETL ensures that data is accurate, complete, consistent, timely, and valid before loading into the data warehouse.

### Data Quality Dimensions

| Dimension | Description |
|-----------|-------------|
| Accuracy | Data matches reality |
| Completeness | No missing values |
| Consistency | Same format everywhere |
| Timeliness | Data is current |
| Validity | Values within expected range |
| Uniqueness | No duplicates |

### Quality Checks in ETL

| Stage | Checks |
|-------|--------|
| Extract | Row counts, source availability |
| Transform | Null handling, format validation |
| Load | Constraint violations, referential integrity |
| Post-load | Aggregate reconciliation |

### Common Data Quality Rules
```sql
-- Null check
WHERE column IS NOT NULL

-- Range check
WHERE age BETWEEN 0 AND 120

-- Referential integrity
WHERE customer_id IN (SELECT id FROM customers)

-- Uniqueness
GROUP BY key HAVING COUNT(*) = 1
```

### Interview Tip
"Garbage in, garbage out." Data quality checks are non-negotiable in production ETL.

---

## Question 19
- [ ] Done

**What is a slowly changing dimension (SCD) and how is it handled in ETL?**

**Answer:**

### Definition
SCDs are dimensions where values change over time. ETL must handle these changes while maintaining historical accuracy for reporting.

### Handling in ETL

**Type 1 (Overwrite):**
```sql
UPDATE dim_customer
SET address = 'New Address'
WHERE customer_id = 123;
```

**Type 2 (History):**
```sql
-- Close old record
UPDATE dim_customer
SET end_date = CURRENT_DATE - 1, is_current = 0
WHERE customer_id = 123 AND is_current = 1;

-- Insert new record
INSERT INTO dim_customer (customer_id, address, start_date, end_date, is_current)
VALUES (123, 'New Address', CURRENT_DATE, '9999-12-31', 1);
```

### ETL Logic for Type 2
1. Lookup incoming record in dimension
2. If exists and changed:
   - Close existing record (set end_date)
   - Insert new record (new surrogate key)
3. If new: Insert with start_date = today

### Best Practices
- Use surrogate keys (required for Type 2)
- Include is_current flag for easy current-state queries
- Store business key for matching

### Interview Tip
Be able to write the SQL for Type 2 SCD. It's a common interview question.

---

## Question 20
- [ ] Done

**Explain the difference between a data warehouse and a data mart.**

**Answer:**

### Definition
A data warehouse is an enterprise-wide repository of integrated data. A data mart is a subset of the warehouse focused on a specific business area or department.

### Comparison

| Aspect | Data Warehouse | Data Mart |
|--------|----------------|-----------|
| Scope | Enterprise-wide | Departmental/Subject |
| Size | Terabytes to petabytes | Gigabytes to terabytes |
| Users | All business users | Specific team/department |
| Data Sources | All enterprise sources | Limited sources |
| Complexity | High | Lower |
| Build Time | Longer | Shorter |

### Types of Data Marts
- **Dependent**: Built from data warehouse
- **Independent**: Built directly from sources
- **Hybrid**: Combination of both

### Architecture
```
Sources → DW (Enterprise) → Data Marts (Sales, Marketing, Finance)
```

### Interview Tip
Data marts are often the first step toward a data warehouse, or specialized views for different teams.

---

## Question 21
- [ ] Done

**What is Hadoop and its core components?**

**Answer:**

### Definition
Hadoop is an open-source distributed computing framework designed to store and process large datasets across clusters of commodity hardware.

### Core Components

| Component | Purpose |
|-----------|---------|
| HDFS | Distributed file storage |
| YARN | Resource management |
| MapReduce | Processing paradigm |
| Common | Shared utilities and libraries |

### HDFS Architecture
- **NameNode**: Metadata management (master)
- **DataNode**: Actual data storage (workers)
- **Block Size**: 128MB (default)
- **Replication**: 3 copies by default

### YARN Components
- **ResourceManager**: Cluster resource allocation
- **NodeManager**: Node-level resource management
- **ApplicationMaster**: Per-application management

### Ecosystem
- Hive (SQL), Pig (scripting)
- HBase (NoSQL), Spark (processing)
- Oozie (workflow), Sqoop (data transfer)

### Interview Tip
Hadoop is less popular now due to Spark and cloud services, but understanding its architecture is still valuable.

---

## Question 22
- [ ] Done

**Explain the difference between Hadoop and Spark.**

**Answer:**

### Comparison

| Aspect | Hadoop MapReduce | Spark |
|--------|------------------|-------|
| Speed | Slower (disk-based) | 10-100x faster (in-memory) |
| Processing | Batch only | Batch + Streaming |
| Ease of Use | Complex API | Higher-level APIs |
| Memory Usage | Low | High |
| Fault Tolerance | High (disk) | High (RDD lineage) |
| Cost | Lower memory needs | Higher memory needs |

### Key Differences
- **Disk vs Memory**: MapReduce writes intermediate results to disk; Spark keeps in memory
- **DAG vs Map-Reduce**: Spark uses DAG for optimization
- **Languages**: Spark supports Python, Scala, Java, R, SQL

### When to Use
| Hadoop | Spark |
|--------|-------|
| Very large batch jobs | Interactive analytics |
| Limited memory | Memory-intensive ops |
| Legacy systems | ML, streaming |
| Cost-sensitive | Performance-critical |

### Modern Reality
- Spark largely replaced MapReduce for processing
- HDFS still used for storage
- Cloud services (EMR, Databricks) host both

### Interview Tip
Know that Spark is generally preferred, but understand the trade-offs (memory cost, complexity).

---

## Question 23
- [ ] Done

**What is MapReduce and how does it work?**

**Answer:**

### Definition
MapReduce is a programming model for processing large datasets in parallel across a distributed cluster using two key operations: Map and Reduce.

### How It Works

**1. Map Phase:**
- Input split into chunks
- Map function processes each chunk
- Outputs key-value pairs

**2. Shuffle and Sort:**
- Groups all values by key
- Sends to reducers

**3. Reduce Phase:**
- Aggregates values for each key
- Produces final output

### Example: Word Count
```
Input: "hello world hello"

Map Output:
(hello, 1), (world, 1), (hello, 1)

Shuffle:
hello → [1, 1]
world → [1]

Reduce Output:
(hello, 2), (world, 1)
```

### Characteristics
- Fault-tolerant (reruns failed tasks)
- Data locality (process where data is)
- Scalable (add more nodes)
- Batch processing only

### Interview Tip
Be able to explain word count example. Understand shuffle is the expensive operation.

---

## Question 24
- [ ] Done

**Describe the role of HDFS in Hadoop.**

**Answer:**

### Definition
HDFS (Hadoop Distributed File System) is the primary storage system of Hadoop, designed to store very large files across multiple machines reliably.

### Key Features

| Feature | Description |
|---------|-------------|
| Distributed | Data spread across cluster |
| Fault-tolerant | Replication (default 3) |
| Scalable | Add nodes easily |
| High Throughput | Optimized for batch reads |
| Cost-effective | Commodity hardware |

### Architecture
- **NameNode**: Master, stores metadata
- **DataNode**: Workers, store actual data
- **Block Size**: 128MB (large for streaming)
- **Write Once**: Append-only model

### Data Flow
```
Client → NameNode (get block locations) → DataNodes (read/write)
```

### Limitations
- Not good for small files (NameNode memory)
- High latency (not for random access)
- Single NameNode was SPOF (now HA available)

### Interview Tip
HDFS is designed for "write once, read many" workloads. Understand the block replication concept.

---

## Question 25
- [ ] Done

**What is Hive and how is it used in big data processing?**

**Answer:**

### Definition
Hive is a data warehouse infrastructure built on Hadoop that provides SQL-like query interface (HiveQL) for processing large datasets stored in HDFS.

### Key Features
- **SQL Interface**: Familiar syntax for analysts
- **Schema on Read**: Schema applied at query time
- **Scalability**: Leverages Hadoop cluster
- **Extensibility**: UDFs, custom SerDes

### Architecture
```
HiveQL → Parser → Optimizer → Execution Engine (MapReduce/Tez/Spark) → HDFS
```

### Use Cases
- Ad-hoc querying of big data
- ETL on Hadoop
- Data summarization
- Log analysis

### Hive vs Traditional DB

| Hive | RDBMS |
|------|-------|
| Schema on read | Schema on write |
| Batch processing | Real-time |
| Append/overwrite | Full CRUD |
| Petabyte scale | Terabyte scale |

### Modern Context
- Often replaced by Spark SQL or Presto
- Still used in legacy Hadoop environments

### Interview Tip
Hive translates SQL to MapReduce/Spark jobs. Great for SQL users on Hadoop.

---

## Question 26
- [ ] Done

**Explain the concept of data partitioning in Hadoop.**

**Answer:**

### Definition
Partitioning divides large datasets into smaller, manageable chunks based on column values, improving query performance by scanning only relevant partitions.

### How It Works
```
/data/sales/year=2023/month=01/data.parquet
/data/sales/year=2023/month=02/data.parquet
/data/sales/year=2024/month=01/data.parquet
```

Query for year=2023 scans only those partitions.

### Benefits

| Benefit | Description |
|---------|-------------|
| Performance | Skip irrelevant data |
| Parallelism | Process partitions independently |
| Manageability | Drop/archive by partition |
| Cost | Less data scanned = less compute |

### Partitioning Strategies
- **Range**: Date ranges, numeric ranges
- **Hash**: Distribute evenly
- **List**: Specific values (country, category)

### Best Practices
- Choose columns with good cardinality
- Avoid too many partitions (small files)
- Partition by frequently filtered columns
- Consider partition pruning in queries

### Interview Tip
Partition by columns used in WHERE clauses. Date is the most common partition key.

---

## Question 27
- [ ] Done

**What is Kafka and its use cases in data engineering?**

**Answer:**

### Definition
Apache Kafka is a distributed event streaming platform for high-throughput, fault-tolerant, real-time data pipelines and streaming applications.

### Core Concepts

| Concept | Description |
|---------|-------------|
| Topic | Named feed of messages |
| Partition | Topic divided for parallelism |
| Producer | Sends messages to topics |
| Consumer | Reads messages from topics |
| Broker | Server in Kafka cluster |
| Consumer Group | Consumers sharing load |

### Use Cases
- **Log Aggregation**: Collect logs from services
- **Event Sourcing**: Track state changes
- **Real-time ETL**: Stream data to warehouse
- **Metrics**: Collect application metrics
- **Message Queue**: Decouple services

### Key Features
- High throughput (millions/sec)
- Horizontal scalability
- Durability (replicated)
- Exactly-once semantics (with config)

### Architecture
```
Producers → Kafka Brokers (Topics/Partitions) → Consumers
```

### Interview Tip
Kafka is the backbone of modern data platforms. Know producers, consumers, topics, partitions.

---

## Question 28
- [ ] Done

**Describe the difference between batch processing and stream processing.**

**Answer:**

### Comparison

| Aspect | Batch | Stream |
|--------|-------|--------|
| Data | Bounded (finite) | Unbounded (infinite) |
| Latency | Minutes to hours | Seconds to milliseconds |
| Processing | All at once | Event by event |
| Example | Daily reports | Real-time dashboards |
| Complexity | Lower | Higher |
| Tools | Spark Batch, Hadoop | Kafka, Flink, Spark Streaming |

### When to Use

| Batch | Stream |
|-------|--------|
| Historical analysis | Real-time alerts |
| Large transformations | Fraud detection |
| Nightly ETL | Live dashboards |
| Cost optimization | Time-sensitive decisions |

### Lambda Architecture
- Batch layer: Accurate, slow
- Speed layer: Approximate, fast
- Serving layer: Merge results

### Kappa Architecture
- Stream-only, reprocess by replaying

### Interview Tip
Modern trend is toward unified batch+stream (Spark, Flink). Understand trade-offs of each.

---

## Question 29
- [ ] Done

**What is Cassandra and its key features?**

**Answer:**

### Definition
Apache Cassandra is a distributed NoSQL database designed for high availability and linear scalability, using a wide-column store model.

### Key Features

| Feature | Description |
|---------|-------------|
| Distributed | No single point of failure |
| Scalable | Linear scaling with nodes |
| High Availability | Replication across data centers |
| Tunable Consistency | Trade-off between consistency and availability |
| Write-optimized | High write throughput |

### Data Model
- Keyspace (like database)
- Tables with rows and columns
- Partition key (distribution)
- Clustering key (sorting within partition)

### When to Use
- High write volume (IoT, time-series)
- Need high availability
- Geographically distributed
- Simple query patterns

### CAP Theorem
- Cassandra chooses AP (Availability, Partition tolerance)
- Eventual consistency by default

### Interview Tip
Know that Cassandra sacrifices consistency for availability. Good for writes, limited queries.

---

## Question 30
- [ ] Done

**Explain the concept of data replication in Hadoop.**

**Answer:**

### Definition
HDFS replicates each data block across multiple DataNodes to ensure fault tolerance and data availability even when nodes fail.

### How It Works
- Default replication factor: 3
- Blocks stored on different nodes/racks
- NameNode tracks block locations
- Automatic re-replication on failure

### Replica Placement Policy
1. First replica: Same node as writer (or random)
2. Second replica: Different rack
3. Third replica: Same rack as second, different node

### Benefits

| Benefit | Description |
|---------|-------------|
| Fault Tolerance | Node failure doesn't lose data |
| Availability | Multiple copies to read from |
| Performance | Read from nearest replica |

### Trade-offs
- Storage overhead (3x by default)
- Write amplification
- Network bandwidth for replication

### Configuring Replication
```xml
<property>
  <name>dfs.replication</name>
  <value>3</value>
</property>
```

### Interview Tip
Understand rack awareness for replica placement. It protects against rack-level failures.

---

## Question 31
- [ ] Done

**What is data processing and its stages?**

**Answer:**

### Definition
Data processing is the collection, transformation, and organization of raw data into meaningful information for analysis and decision-making.

### Stages

| Stage | Description |
|-------|-------------|
| Collection | Gather data from sources |
| Preparation | Clean and validate data |
| Input | Load into processing system |
| Processing | Transform and compute |
| Output | Store results |
| Distribution | Deliver to consumers |

### Processing Types
- **Batch**: Process accumulated data periodically
- **Real-time**: Process as data arrives
- **Interactive**: On-demand queries

### Common Operations
- Filtering and selection
- Aggregation and grouping
- Joining datasets
- Sorting and ranking
- Transformation and enrichment

### Tools by Stage
| Stage | Tools |
|-------|-------|
| Collection | Kafka, Flume, APIs |
| Processing | Spark, Flink |
| Storage | HDFS, S3, databases |
| Analysis | SQL, Python, BI tools |

### Interview Tip
Frame data processing as a pipeline with clear stages and checkpoints.

---

## Question 32
- [ ] Done

**Explain the difference between batch processing and real-time processing.**

**Answer:**

### Comparison

| Aspect | Batch | Real-time |
|--------|-------|-----------|
| Timing | Scheduled intervals | Continuous |
| Data | Historical, accumulated | Current, streaming |
| Latency | Minutes to hours | Milliseconds to seconds |
| Volume | Large volumes at once | Event by event |
| Accuracy | High (complete data) | May be approximate |
| Complexity | Lower | Higher |

### Use Cases

| Batch | Real-time |
|-------|-----------|
| Monthly reports | Fraud detection |
| Data warehouse loads | Live dashboards |
| Model training | Recommendation engines |
| Historical analysis | Alerting systems |

### Tools

| Batch | Real-time |
|-------|-----------|
| Hadoop MapReduce | Apache Kafka |
| Spark (batch mode) | Apache Flink |
| dbt | Spark Streaming |
| Airflow | Storm, Kinesis |

### Choosing Between
- Latency requirements
- Data volume patterns
- Cost considerations
- Accuracy needs

### Interview Tip
Real-time adds complexity and cost. Use batch unless latency requirements demand streaming.

---

## Question 33
- [ ] Done

**What are the common data transformation techniques?**

**Answer:**

### Transformation Types

| Technique | Description | Example |
|-----------|-------------|---------|
| Filtering | Select subset of rows | WHERE amount > 100 |
| Mapping | Transform column values | UPPER(name) |
| Aggregation | Summarize data | SUM, COUNT, AVG |
| Joining | Combine datasets | Join on customer_id |
| Pivoting | Rows to columns | Reshape for reporting |
| Unpivoting | Columns to rows | Normalize wide tables |
| Deduplication | Remove duplicates | DISTINCT |
| Type Conversion | Change data types | CAST to integer |

### Common Operations
```sql
-- Filtering
SELECT * FROM sales WHERE year = 2024

-- Mapping
SELECT LOWER(email), amount * 1.1 AS adjusted_amount FROM orders

-- Aggregation
SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id

-- Join
SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id
```

### ETL Transformations
- Data cleansing (nulls, formats)
- Business rule application
- Derived column creation
- Surrogate key assignment

### Interview Tip
Know SQL transformations well. Most data engineering is SQL-based transformation.

---

## Question 34
- [ ] Done

**Describe the role of data cleansing in data processing.**

**Answer:**

### Definition
Data cleansing identifies and corrects errors, inconsistencies, and quality issues in data to ensure accuracy and reliability for downstream processing.

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Missing values | Impute, default, or flag |
| Duplicates | Deduplicate by key |
| Invalid formats | Parse and standardize |
| Outliers | Investigate, cap, or remove |
| Inconsistent values | Map to standard values |
| Wrong data types | Convert and validate |

### Cleansing Steps
1. Profile data (understand issues)
2. Define quality rules
3. Apply corrections
4. Validate results
5. Document changes

### Tools and Techniques
```python
# Missing values
df.fillna(0)
df.dropna()

# Duplicates
df.drop_duplicates()

# Standardization
df['date'] = pd.to_datetime(df['date'])
df['name'] = df['name'].str.strip().str.upper()
```

### Impact of Poor Data Quality
- Wrong analytics and decisions
- Customer dissatisfaction
- Compliance violations
- Operational inefficiency

### Interview Tip
Data quality is everyone's responsibility but data engineers are the first line of defense.

---

## Question 35
- [ ] Done

**What is data enrichment and why is it important?**

**Answer:**

### Definition
Data enrichment enhances existing data by adding relevant information from external or internal sources to make it more valuable for analysis.

### Types of Enrichment

| Type | Example |
|------|---------|
| Geographic | Add city, state from zip code |
| Demographic | Add age group from birth date |
| Firmographic | Add company size from company name |
| Behavioral | Add customer segment from history |
| Reference | Add product category from SKU |

### Benefits

| Benefit | Description |
|---------|-------------|
| Better Analytics | More dimensions for analysis |
| Personalization | Target customers better |
| Completeness | Fill in missing information |
| Context | Understand data better |

### Common Enrichment Sources
- Third-party data providers
- Public datasets (census, geo)
- Internal reference tables
- APIs (weather, exchange rates)

### Implementation
```sql
-- Enrich orders with customer segment
SELECT o.*, c.segment, c.lifetime_value
FROM orders o
JOIN customers c ON o.customer_id = c.id
```

### Interview Tip
Enrichment adds value but consider data freshness and source reliability.

---

## Question 36
- [ ] Done

**Explain the concept of data aggregation and its use cases.**

**Answer:**

### Definition
Data aggregation summarizes detailed data into higher-level summaries using functions like SUM, COUNT, AVG, MIN, MAX, typically grouped by dimensions.

### Common Aggregations

| Function | Purpose |
|----------|---------|
| SUM | Total value |
| COUNT | Number of records |
| AVG | Average value |
| MIN/MAX | Range bounds |
| MEDIAN | Middle value |
| STDDEV | Variability |

### Use Cases
- Daily/monthly sales totals
- Average order value by segment
- Customer counts by region
- Website visits by hour
- Inventory levels by location

### Aggregation Levels
```
Transactions (millions/day)
    ↓ Aggregate
Daily Summaries (thousands)
    ↓ Aggregate
Monthly Reports (hundreds)
    ↓ Aggregate
Annual KPIs (few)
```

### SQL Example
```sql
SELECT 
    DATE(order_date) as date,
    region,
    COUNT(*) as order_count,
    SUM(amount) as total_sales,
    AVG(amount) as avg_order_value
FROM orders
GROUP BY DATE(order_date), region
```

### Interview Tip
Pre-aggregation in data warehouses speeds up reporting queries significantly.

---

## Question 37
- [ ] Done

**What is data deduplication and how is it achieved?**

**Answer:**

### Definition
Data deduplication identifies and removes duplicate records to ensure each entity appears only once, maintaining data integrity and accurate analytics.

### Types of Duplicates

| Type | Description |
|------|-------------|
| Exact | Identical rows |
| Fuzzy | Similar but not identical (typos) |
| Semantic | Same entity, different representation |

### Detection Methods

| Method | Use Case |
|--------|----------|
| Exact match | Simple duplicates |
| Key-based | Match on business key |
| Fuzzy matching | Spelling variations |
| Probabilistic | Complex matching rules |

### Deduplication Strategies
```sql
-- Keep first occurrence
SELECT DISTINCT * FROM table

-- Keep specific version (latest)
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY customer_id 
        ORDER BY updated_at DESC
    ) as rn
    FROM customers
) WHERE rn = 1
```

### Best Practices
- Define matching criteria clearly
- Handle edge cases (null keys)
- Log duplicates found
- Validate after deduplication

### Interview Tip
Know the ROW_NUMBER() pattern for deduplication. Very common in data engineering.

---

## Question 38
- [ ] Done

**Describe the difference between data filtering and data sorting.**

**Answer:**

### Comparison

| Aspect | Filtering | Sorting |
|--------|-----------|---------|
| Purpose | Select subset | Order results |
| Result Size | Reduced | Same |
| Clause | WHERE/HAVING | ORDER BY |
| Output | Matching rows only | All rows, ordered |

### Filtering
- Removes rows not meeting criteria
- Reduces data volume
- Applied early in query execution

```sql
-- Filter to high-value orders
SELECT * FROM orders WHERE amount > 1000
```

### Sorting
- Arranges rows in order
- Doesn't change row count
- Applied late in query execution

```sql
-- Sort by date descending
SELECT * FROM orders ORDER BY order_date DESC
```

### Combined Example
```sql
SELECT customer_id, amount
FROM orders
WHERE amount > 100           -- Filter first
ORDER BY amount DESC         -- Then sort
LIMIT 10                     -- Finally limit
```

### Performance Considerations
- Filtering early reduces data to sort
- Indexes help both operations
- Sorting large datasets is expensive

### Interview Tip
Filter early, sort late. Reduces the amount of data to process.

---

## Question 39
- [ ] Done

**What is data normalization and its techniques?**

**Answer:**

### Definition
Data normalization organizes data to reduce redundancy and improve integrity by dividing tables into smaller related tables and defining relationships.

### Normal Forms

| Form | Rule |
|------|------|
| 1NF | Atomic values, no repeating groups |
| 2NF | 1NF + No partial dependencies |
| 3NF | 2NF + No transitive dependencies |
| BCNF | Every determinant is a candidate key |

### Example: 1NF Violation
```
Before: Order(id, items: "A,B,C")  -- Not atomic
After:  Order(id), OrderItem(order_id, item)  -- Atomic
```

### Normalization vs Denormalization

| Normalized | Denormalized |
|------------|--------------|
| Less redundancy | More redundancy |
| More joins | Fewer joins |
| OLTP systems | OLAP systems |
| Data integrity | Query performance |

### When to Denormalize
- Read-heavy workloads
- Reporting and analytics
- When join performance is critical

### Interview Tip
Know first 3 normal forms. Understand when denormalization is appropriate (analytics).

---

## Question 40
- [ ] Done

**Explain the purpose of data validation in data processing.**

**Answer:**

### Definition
Data validation ensures that data meets defined quality standards, business rules, and constraints before being accepted into the system.

### Validation Types

| Type | Description | Example |
|------|-------------|---------|
| Type | Correct data type | Is it a number? |
| Range | Within bounds | Age between 0-120 |
| Format | Matches pattern | Email format |
| Referential | FK exists | Customer ID in customers table |
| Business | Business rules | Order amount > 0 |
| Completeness | Required fields present | Name is not null |

### Validation Levels

| Level | When |
|-------|------|
| Source | At data entry |
| Ingestion | During ETL extract |
| Processing | During transformation |
| Load | Before warehouse insert |

### Implementation
```python
def validate_order(order):
    errors = []
    
    if order['amount'] <= 0:
        errors.append("Amount must be positive")
    
    if order['customer_id'] not in valid_customers:
        errors.append("Invalid customer")
    
    if not order.get('date'):
        errors.append("Date is required")
    
    return errors
```

### Interview Tip
Validation should be at multiple stages. Fail fast and provide clear error messages.

---

## Question 41
- [ ] Done

**What is data integration and its challenges?**

**Answer:**

### Definition
Data integration combines data from different sources into a unified view, enabling comprehensive analysis across the organization.

### Approaches

| Approach | Description |
|----------|-------------|
| ETL | Extract, Transform, Load to warehouse |
| ELT | Load raw, transform in warehouse |
| Data Virtualization | Query across sources without moving |
| Data Federation | Unified interface, data stays in place |

### Key Challenges

| Challenge | Description |
|-----------|-------------|
| Schema Heterogeneity | Different structures across sources |
| Semantic Differences | Same field, different meanings |
| Data Quality | Varying quality levels |
| Timing | Different update frequencies |
| Volume | Large amounts of data |
| Security | Access control across sources |

### Solutions

| Challenge | Solution |
|-----------|----------|
| Schema differences | Canonical data model |
| Semantics | Master data management |
| Quality | Validation and cleansing |
| Timing | CDC, incremental loads |
| Security | Data governance |

### Interview Tip
Integration is often the hardest part of data engineering. Emphasize data governance and standards.

---

## Question 42
- [ ] Done

**Explain the difference between ETL and ELT approaches.**

**Answer:**

### Comparison

| Aspect | ETL | ELT |
|--------|-----|-----|
| Transform Location | ETL server | Target warehouse |
| Processing | Before load | After load |
| Best For | Traditional DW | Cloud DW |
| Examples | Informatica | dbt, Snowflake |

### Interview Tip
ELT is the modern standard with cloud warehouses. Know both approaches.

---

## Question 43
- [ ] Done

**What are the common data integration patterns?**

**Answer:**

### Common Patterns
- Migration (one-time transfer)
- Broadcast (one-to-many)
- Aggregation (many-to-one)
- Bidirectional sync
- Hub-and-spoke architecture

### Interview Tip
Modern trend is toward data mesh (decentralized) with data contracts.

---

## Question 44
- [ ] Done

**Describe the role of data pipelines in data engineering.**

**Answer:**

### Definition
Data pipelines automate moving and transforming data from sources to destinations reliably.

### Key Characteristics
- Automated, scheduled/triggered
- Idempotent (repeatable)
- Observable (logging, monitoring)

### Interview Tip
Emphasize reliability and observability in pipeline design.

---

## Question 45
- [ ] Done

**What is a data lake and how does it differ from a data warehouse?**

**Answer:**

### Comparison

| Aspect | Data Lake | Data Warehouse |
|--------|-----------|----------------|
| Data Format | Raw, any format | Structured |
| Schema | On read | On write |
| Users | Data scientists | Analysts |
| Cost | Lower storage | Higher |

### Interview Tip
Know medallion architecture (bronze/silver/gold).

---

## Question 46
- [ ] Done

**Explain the concept of data ingestion and its methods.**

**Answer:**

### Methods
- Batch: Scheduled bulk loads
- Streaming: Continuous real-time
- Micro-batch: Small frequent batches

### Interview Tip
Know pull vs push ingestion and backpressure in streaming.

---

## Question 47
- [ ] Done

**What is change data capture (CDC) and its use cases?**

**Answer:**

### Definition
CDC captures database changes (inserts, updates, deletes) for near real-time propagation.

### Methods
- Log-based (preferred): Read transaction logs
- Trigger-based: DB triggers
- Timestamp-based: Query by modified_date

### Interview Tip
Debezium is the popular open-source solution.

---

## Question 48
- [ ] Done

**Describe the difference between batch and streaming data integration.**

**Answer:**

### Comparison

| Batch | Streaming |
|-------|-----------|
| Bounded data | Unbounded |
| Higher latency | Low latency |
| Simpler | More complex |

### Interview Tip
Start with batch unless latency requirements demand streaming.

---

## Question 49
- [ ] Done

**What is data replication and its techniques?**

**Answer:**

### Types
- Synchronous: Wait for all copies
- Asynchronous: Background copy
- Log shipping: Replicate transaction logs

### Interview Tip
Understand CAP theorem trade-offs.

---

## Question 50
- [ ] Done

**Explain the purpose of data orchestration in data pipelines.**

**Answer:**

### Definition
Orchestration coordinates pipeline tasks, handling dependencies, scheduling, and error handling.

### Key Tools
- Apache Airflow (most popular)
- Prefect, Dagster

### Interview Tip
Airflow is industry standard. Know DAGs and operators.

---

## Question 51
- [ ] Done

**What is a database management system (DBMS) and its types?**

**Answer:**

### Types
- Relational: PostgreSQL, MySQL
- Document: MongoDB
- Key-Value: Redis
- Column-family: Cassandra
- Graph: Neo4j

### Interview Tip
Choose based on use case: RDBMS for transactions, NoSQL for scale.

---

## Question 52
- [ ] Done

**Explain the difference between SQL and NoSQL databases.**

**Answer:**

### Comparison

| SQL | NoSQL |
|-----|-------|
| Fixed schema | Flexible |
| ACID | BASE |
| Vertical scaling | Horizontal |
| Complex queries | Simple queries |

### Interview Tip
It's not either/or. Many systems use both.

---

## Question 53
- [ ] Done

**What is data partitioning and its strategies?**

**Answer:**

### Strategies
- Range: By value ranges (dates)
- List: By discrete values
- Hash: Even distribution

### Interview Tip
Date is the most common partition key.

---

## Question 54
- [ ] Done

**Describe the concept of data indexing and its benefits.**

**Answer:**

### Benefits
- Faster queries
- Reduced I/O
- Efficient sorting

### Trade-offs
- Slows writes
- Storage overhead

### Interview Tip
Index columns in WHERE clauses and foreign keys.

---

## Question 55
- [ ] Done

**What is data sharding and when is it used?**

**Answer:**

### Definition
Sharding partitions data across multiple database instances for horizontal scaling.

### When to Use
- Data exceeds single server
- High write volume
- Geographic distribution

### Interview Tip
Sharding adds complexity. Delay until truly needed.

---

## Question 56
- [ ] Done

**Explain the difference between vertical and horizontal scaling in databases.**

**Answer:**

### Comparison

| Vertical | Horizontal |
|----------|------------|
| Bigger machine | More machines |
| Simpler | Complex |
| Limited | Unlimited |

### Interview Tip
Start vertical, go horizontal when needed.

---

## Question 57
- [ ] Done

**What is data replication and its types?**

**Answer:**

### Replication Types
- Synchronous: Strong consistency
- Asynchronous: Better performance
- Semi-synchronous: Balanced

### Interview Tip
Know CAP theorem trade-offs.

---

## Question 58
- [ ] Done

**Describe the role of caching in data retrieval.**

**Answer:**

### Benefits
- Reduced latency
- Lower database load
- Improved user experience

### Strategies
- Cache-aside (app manages)
- Write-through
- TTL-based invalidation

### Interview Tip
Cache invalidation is hard. Know the strategies.

---

## Question 59
- [ ] Done

**What is a data lake and its architecture?**

**Answer:**

### Architecture
```
Bronze (Raw) → Silver (Cleaned) → Gold (Curated)
```

### Technologies
- Storage: S3, ADLS
- Formats: Parquet, Delta Lake
- Processing: Spark

### Interview Tip
Know medallion architecture.

---

## Question 60
- [ ] Done

**Explain the concept of data archiving and its importance.**

**Answer:**

### Purpose
- Cost optimization
- Compliance
- Performance improvement

### Storage Tiers
Hot → Warm → Cold → Glacier

### Interview Tip
Know retention policies and compliance requirements.

---

## Question 61
- [ ] Done

**What is data governance and its key components?**

**Answer:**

### Components
- Data quality management
- Data security and privacy
- Metadata management
- Data lineage
- Master data management

### Interview Tip
Governance is about policies, not just tools.

---

## Question 62
- [ ] Done

**Explain the difference between data governance and data management.**

**Answer:**

### Comparison

| Governance | Management |
|------------|------------|
| Policies and rules | Implementation |
| Strategic | Operational |
| What and why | How |

### Interview Tip
Governance sets rules, management executes them.

---

## Question 63
- [ ] Done

**What are the common data governance frameworks?**

**Answer:**

### Frameworks
- DAMA-DMBOK
- COBIT
- ISO 8000
- DCAM (EDM Council)

### Interview Tip
Know DAMA-DMBOK as the most referenced framework.

---

## Question 64
- [ ] Done

**Describe the role of data lineage in data governance.**

**Answer:**

### Purpose
- Track data origin and transformations
- Impact analysis
- Compliance and auditing
- Debugging

### Interview Tip
Lineage is critical for compliance (GDPR, SOX).

---

## Question 65
- [ ] Done

**What is data quality and its dimensions?**

**Answer:**

### Dimensions
- Accuracy, Completeness
- Consistency, Timeliness
- Validity, Uniqueness

### Interview Tip
Know all six dimensions and how to measure each.

---

## Question 66
- [ ] Done

**Explain the concept of data stewardship and its responsibilities.**

**Answer:**

### Responsibilities
- Define data standards
- Ensure quality
- Manage metadata
- Resolve data issues

### Interview Tip
Data stewards are business-side, not IT.

---

## Question 67
- [ ] Done

**What is data security and its best practices?**

**Answer:**

### Best Practices
- Encryption (at rest, in transit)
- Access control (RBAC)
- Auditing and monitoring
- Data masking

### Interview Tip
Defense in depth—multiple layers of security.

---

## Question 68
- [ ] Done

**Describe the difference between authentication and authorization in data security.**

**Answer:**

### Comparison

| Authentication | Authorization |
|----------------|---------------|
| Who are you? | What can you do? |
| Identity verification | Permission checking |
| Login | Access control |

### Interview Tip
Know the difference clearly—common interview question.

---

## Question 69
- [ ] Done

**What is data encryption and its types?**

**Answer:**

### Types
- At rest: Stored data encrypted
- In transit: Data encrypted during transfer
- Application-level: Encrypted before storage

### Algorithms
- AES-256 (symmetric)
- RSA (asymmetric)

### Interview Tip
Use TLS for transit, AES for storage.

---

## Question 70
- [ ] Done

**Explain the purpose of data auditing and its techniques.**

**Answer:**

### Purpose
- Compliance verification
- Security monitoring
- Change tracking

### Techniques
- Audit logs
- Database triggers
- CDC for change tracking

### Interview Tip
Audit logs should be immutable and tamper-proof.

---

## Question 71
- [ ] Done

**What is data monitoring and its importance?**

**Answer:**

### Importance
- Early issue detection
- Performance tracking
- SLA compliance
- Capacity planning

### Interview Tip
Monitor data quality, not just pipeline health.

---

## Question 72
- [ ] Done

**Explain the difference between real-time and batch monitoring.**

**Answer:**

### Comparison

| Real-time | Batch |
|-----------|-------|
| Immediate alerts | Periodic reports |
| Streaming metrics | Aggregated analysis |
| Higher cost | Lower cost |

### Interview Tip
Use real-time for critical alerts, batch for trends.

---

## Question 73
- [ ] Done

**What are the common data monitoring tools and techniques?**

**Answer:**

### Tools
- Datadog, New Relic
- Great Expectations
- Monte Carlo, Bigeye
- dbt tests

### Interview Tip
Data observability is a growing field. Know the tools.

---

## Question 74
- [ ] Done

**Describe the role of data profiling in data monitoring.**

**Answer:**

### Purpose
- Understand data characteristics
- Detect anomalies
- Validate assumptions

### Metrics
- Null counts, distributions
- Min/max, cardinality
- Patterns and formats

### Interview Tip
Profile data before building pipelines.

---

## Question 75
- [ ] Done

**What is data optimization and its strategies?**

**Answer:**

### Strategies
- Partitioning
- Indexing
- Caching
- Query optimization
- Compression

### Interview Tip
Start with partitioning and indexing—biggest wins.

---

## Question 76
- [ ] Done

**Explain the concept of data partitioning and its benefits in optimization.**

**Answer:**

### Benefits
- Query performance (partition pruning)
- Parallel processing
- Easier maintenance

### Interview Tip
Partition by frequently filtered columns.

---

## Question 77
- [ ] Done

**What is query optimization and its techniques?**

**Answer:**

### Techniques
- Use indexes
- Avoid SELECT *
- Filter early
- Use EXPLAIN plans
- Optimize joins

### Interview Tip
Always analyze query plans for slow queries.

---

## Question 78
- [ ] Done

**Describe the difference between data compression and data deduplication.**

**Answer:**

### Comparison

| Compression | Deduplication |
|-------------|---------------|
| Reduce size | Remove duplicates |
| All data | Identical blocks |
| Algorithm-based | Hash-based |

### Interview Tip
Use both for optimal storage efficiency.

---

## Question 79
- [ ] Done

**What is data caching and its use cases in optimization?**

**Answer:**

### Use Cases
- Frequently accessed data
- Expensive computations
- API responses
- Session data

### Interview Tip
Cache invalidation is the hard part.

---

## Question 80
- [ ] Done

**Explain the purpose of data archiving in data optimization.**

**Answer:**

### Purpose
- Free up primary storage
- Improve query performance
- Reduce costs

### Interview Tip
Archive based on access patterns and retention policies.

---

## Question 81
- [ ] Done

**What is Apache Spark and its key features?**

**Answer:**

### Key Features
- In-memory processing (10-100x faster)
- Unified engine (batch, streaming, ML)
- Multiple languages (Python, Scala, SQL)
- Distributed computing

### Interview Tip
Know RDDs, DataFrames, and Spark SQL.

---

## Question 82
- [ ] Done

**Explain the difference between Spark RDDs and DataFrames.**

**Answer:**

### Comparison

| RDD | DataFrame |
|-----|-----------|
| Low-level | High-level |
| Unstructured | Structured |
| No optimization | Catalyst optimizer |
| Any type | Row-based |

### Interview Tip
Use DataFrames by default—better performance.

---

## Question 83
- [ ] Done

**What is Apache Airflow and its use cases?**

**Answer:**

### Use Cases
- ETL orchestration
- ML pipelines
- Data quality checks
- Report generation

### Key Concepts
- DAGs, Operators, Sensors
- Scheduling, dependencies

### Interview Tip
Airflow is the industry standard orchestrator.

---

## Question 84
- [ ] Done

**Describe the role of Apache Kafka in data streaming.**

**Answer:**

### Role
- Message broker
- Event streaming platform
- Real-time data pipelines

### Key Concepts
- Topics, partitions
- Producers, consumers
- Consumer groups

### Interview Tip
Kafka is the backbone of modern data architectures.

---

## Question 85
- [ ] Done

**What is Talend and its key components?**

**Answer:**

### Components
- Talend Open Studio (ETL)
- Talend Data Quality
- Talend MDM
- Talend Big Data

### Interview Tip
Talend is visual ETL—good for non-programmers.

---

## Question 86
- [ ] Done

**Explain the concept of data pipelines in Apache NiFi.**

**Answer:**

### Features
- Visual dataflow design
- Data provenance
- Back-pressure handling
- Extensible processors

### Interview Tip
NiFi excels at data routing and transformation.

---

## Question 87
- [ ] Done

**What is Informatica PowerCenter and its features?**

**Answer:**

### Features
- Enterprise ETL
- Data quality
- Master data management
- Cloud integration

### Interview Tip
Informatica is enterprise-grade, expensive.

---

## Question 88
- [ ] Done

**Describe the difference between Hadoop and Apache Flink.**

**Answer:**

### Comparison

| Hadoop | Flink |
|--------|-------|
| Batch | Stream-first |
| Disk-based | Memory-based |
| MapReduce | DataStream API |
| Higher latency | Low latency |

### Interview Tip
Flink is better for streaming; Spark handles both.

---

## Question 89
- [ ] Done

**What is dbt (Data Build Tool) and its benefits?**

**Answer:**

### Benefits
- SQL-based transformations
- Version control
- Documentation
- Testing
- Lineage

### Interview Tip
dbt is the standard for transformation in modern stacks.

---

## Question 90
- [ ] Done

**Explain the purpose of Presto in data querying.**

**Answer:**

### Purpose
- Distributed SQL query engine
- Query data where it lives
- Federated queries

### Interview Tip
Presto/Trino is for ad-hoc queries across sources.

---

## Question 91
- [ ] Done

**What is cloud data engineering and its advantages?**

**Answer:**

### Advantages
- Scalability
- Pay-as-you-go
- Managed services
- Global availability
- Faster deployment

### Interview Tip
Cloud is the default for new projects.

---

## Question 92
- [ ] Done

**Explain the difference between AWS, Azure, and Google Cloud Platform for data engineering.**

**Answer:**

### Comparison

| AWS | Azure | GCP |
|-----|-------|-----|
| Redshift | Synapse | BigQuery |
| Glue | Data Factory | Dataflow |
| S3 | Blob Storage | GCS |
| Kinesis | Event Hubs | Pub/Sub |

### Interview Tip
BigQuery is serverless; Redshift/Synapse need sizing.

---

## Question 93
- [ ] Done

**What is Amazon S3 and its use cases in data storage?**

**Answer:**

### Use Cases
- Data lake storage
- Backup and archiving
- Static assets
- Analytics staging

### Interview Tip
S3 is the foundation of AWS data architecture.

---

## Question 94
- [ ] Done

**Describe the role of Azure Data Factory in data integration.**

**Answer:**

### Features
- Cloud ETL/ELT
- Data pipelines
- Integration with Azure services
- Mapping data flows

### Interview Tip
ADF is Azure's answer to AWS Glue.

---

## Question 95
- [ ] Done

**What is Google BigQuery and its key features?**

**Answer:**

### Key Features
- Serverless (no sizing)
- Columnar storage
- Separation of compute/storage
- ML integration (BQML)

### Interview Tip
BigQuery is pay-per-query, great for analytics.

---

## Question 96
- [ ] Done

**Explain the concept of serverless data processing in AWS Lambda.**

**Answer:**

### Use Cases
- Event-driven processing
- Light transformations
- API backends
- Glue job triggers

### Limitations
- 15 min timeout
- Memory limits
- Cold starts

### Interview Tip
Lambda is for light processing, not heavy ETL.

---

## Question 97
- [ ] Done

**What is Azure Databricks and its benefits?**

**Answer:**

### Benefits
- Managed Spark
- Collaborative notebooks
- Delta Lake integration
- ML workflows

### Interview Tip
Databricks is premium Spark with collaboration features.

---

## Question 98
- [ ] Done

**Describe the difference between Amazon Redshift and Google BigQuery.**

**Answer:**

### Comparison

| Redshift | BigQuery |
|----------|----------|
| Provisioned clusters | Serverless |
| Hourly pricing | Per-query pricing |
| Manual scaling | Auto-scaling |
| Materialized views | BI Engine |

### Interview Tip
BigQuery for variable workloads, Redshift for steady.

---

## Question 99
- [ ] Done

**What is AWS Glue and its use cases in data integration?**

**Answer:**

### Use Cases
- ETL jobs (Spark)
- Data catalog
- Schema discovery
- Crawlers for metadata

### Interview Tip
Glue is serverless ETL + data catalog.

---

## Question 100
- [ ] Done

**Explain the purpose of Google Cloud Dataflow in data processing.**

**Answer:**

### Purpose
- Unified batch and streaming
- Apache Beam-based
- Fully managed
- Auto-scaling

### Interview Tip
Dataflow = managed Beam. Great for streaming.

---

