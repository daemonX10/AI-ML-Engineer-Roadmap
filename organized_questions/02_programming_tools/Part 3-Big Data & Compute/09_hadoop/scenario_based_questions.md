# Hadoop Interview Questions - Scenario Based Questions

## Question 1
- [ ] Done

**Your Hadoop cluster is running slow. How do you diagnose and fix performance issues?**

### Answer

### Diagnosis Steps

| Step | Tool/Action |
|------|-------------|
| 1. Check YARN metrics | ResourceManager UI or REST API |
| 2. Review job counters | MapReduce history server |
| 3. Analyze logs | Container logs on NodeManagers |
| 4. Check resource utilization | Ganglia, Ambari |
| 5. Identify bottlenecks | GC logs, I/O wait |

### Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Memory pressure | GC overhead, OOM | Increase container memory |
| Data skew | Some reducers slow | Salting keys, custom partitioner |
| Small files | Many short-running mappers | Use HAR, CombineInputFormat |
| Network bottleneck | High shuffle time | Enable compression |
| Disk I/O | High spill counts | Increase sort buffer |

### Python Code Example
```python
import requests

class HadoopDiagnostics:
    def __init__(self, rm_url, history_url):
        self.rm_url = rm_url
        self.history_url = history_url
    
    def get_slow_jobs(self, threshold_minutes=30):
        """Find jobs running longer than threshold"""
        response = requests.get(f"{self.rm_url}/ws/v1/cluster/apps?state=RUNNING")
        apps = response.json()['apps']['app'] if response.json()['apps'] else []
        
        slow_jobs = []
        for app in apps:
            elapsed = (app['finishedTime'] or 0) - app['startedTime']
            if elapsed > threshold_minutes * 60 * 1000:
                slow_jobs.append({
                    'id': app['id'],
                    'name': app['name'],
                    'elapsed_min': elapsed / 60000
                })
        return slow_jobs
    
    def get_job_counters(self, job_id):
        """Get detailed job counters"""
        response = requests.get(
            f"{self.history_url}/ws/v1/history/mapreduce/jobs/{job_id}/counters"
        )
        return response.json()
    
    def diagnose_data_skew(self, job_id):
        """Check for data skew in reducers"""
        response = requests.get(
            f"{self.history_url}/ws/v1/history/mapreduce/jobs/{job_id}/tasks?type=REDUCE"
        )
        tasks = response.json()['tasks']['task']
        
        times = [t['elapsedTime'] for t in tasks]
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        if max_time > 2 * avg_time:
            return {
                'skew_detected': True,
                'avg_time': avg_time,
                'max_time': max_time,
                'ratio': max_time / avg_time
            }
        return {'skew_detected': False}

# Solution for data skew: Salted keys
from mrjob.job import MRJob
import random

class SaltedWordCount(MRJob):
    """Use salting to distribute skewed keys"""
    
    NUM_SALTS = 10
    
    def mapper(self, _, line):
        for word in line.split():
            # Add salt to popular words
            salt = random.randint(0, self.NUM_SALTS - 1)
            yield f"{word}_{salt}", 1
    
    def combiner(self, salted_key, counts):
        yield salted_key, sum(counts)
    
    def reducer(self, salted_key, counts):
        # Remove salt for final output
        word = salted_key.rsplit('_', 1)[0]
        yield word, sum(counts)
```

---

## Question 2
- [ ] Done

**You need to migrate data from an RDBMS to Hadoop. How would you approach this?**

### Answer

### Migration Strategy

| Step | Action |
|------|--------|
| 1. Analyze source | Schema, volume, data types |
| 2. Choose tool | Sqoop, Spark, custom ETL |
| 3. Plan partitioning | Split column selection |
| 4. Handle incremental | Timestamp or ID-based |
| 5. Validate | Row counts, checksums |

### Python Code Example
```python
import subprocess
import os

class DataMigration:
    def __init__(self, jdbc_url, username, password, hdfs_base):
        self.jdbc_url = jdbc_url
        self.username = username
        self.password = password
        self.hdfs_base = hdfs_base
    
    def full_import(self, table, split_column=None, mappers=4):
        """Import entire table to HDFS"""
        cmd = [
            'sqoop', 'import',
            '--connect', self.jdbc_url,
            '--username', self.username,
            '--password', self.password,
            '--table', table,
            '--target-dir', f'{self.hdfs_base}/{table}',
            '--num-mappers', str(mappers),
            '--as-parquetfile'  # Store as Parquet
        ]
        
        if split_column:
            cmd.extend(['--split-by', split_column])
        
        subprocess.run(cmd, check=True)
    
    def incremental_import(self, table, check_column, last_value):
        """Incremental import based on column value"""
        cmd = [
            'sqoop', 'import',
            '--connect', self.jdbc_url,
            '--username', self.username,
            '--password', self.password,
            '--table', table,
            '--target-dir', f'{self.hdfs_base}/{table}_incremental',
            '--incremental', 'lastmodified',
            '--check-column', check_column,
            '--last-value', str(last_value),
            '--append'
        ]
        subprocess.run(cmd, check=True)
    
    def validate_import(self, table, hdfs_client):
        """Validate row counts match"""
        # Get RDBMS count (using subprocess for simplicity)
        result = subprocess.run(
            ['sqoop', 'eval', '--connect', self.jdbc_url,
             '--username', self.username, '--password', self.password,
             '--query', f'SELECT COUNT(*) FROM {table}'],
            capture_output=True, text=True
        )
        rdbms_count = int(result.stdout.strip().split('\n')[-1])
        
        # Get HDFS count
        hdfs_path = f'{self.hdfs_base}/{table}'
        hdfs_count = sum(1 for _ in hdfs_client.read(hdfs_path))
        
        return {
            'rdbms_count': rdbms_count,
            'hdfs_count': hdfs_count,
            'match': rdbms_count == hdfs_count
        }

# Usage
migration = DataMigration(
    jdbc_url='jdbc:mysql://db-server/mydb',
    username='user',
    password='pass',
    hdfs_base='/user/data'
)
migration.full_import('customers', split_column='customer_id')
```

---

## Question 3
- [ ] Done

**Your NameNode has failed. How do you recover?**

### Answer

### Recovery Steps

| Scenario | Action |
|----------|--------|
| **HA enabled** | Automatic failover to standby |
| **No HA, backup exists** | Restore from checkpoint |
| **No HA, no backup** | Possible data loss |

### Recovery Procedure (Non-HA)

```bash
# 1. Stop all services
# 2. Copy backup metadata to NameNode directory
# 3. Run recovery commands
```

### Python Code Example
```python
import subprocess
import shutil
from datetime import datetime

class NameNodeRecovery:
    def __init__(self, hadoop_home, namenode_dir, backup_dir):
        self.hadoop_home = hadoop_home
        self.namenode_dir = namenode_dir
        self.backup_dir = backup_dir
    
    def check_namenode_health(self, namenode_url):
        """Check if NameNode is healthy"""
        import requests
        try:
            response = requests.get(f"{namenode_url}/jmx?qry=Hadoop:service=NameNode,name=NameNodeStatus")
            return response.json()['beans'][0]['State']
        except:
            return 'UNREACHABLE'
    
    def backup_metadata(self):
        """Backup current metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{self.backup_dir}/namenode_backup_{timestamp}"
        shutil.copytree(self.namenode_dir, backup_path)
        return backup_path
    
    def restore_from_backup(self, backup_path):
        """Restore from backup"""
        # Stop NameNode first
        subprocess.run([f'{self.hadoop_home}/sbin/hadoop-daemon.sh', 'stop', 'namenode'])
        
        # Restore metadata
        shutil.rmtree(self.namenode_dir, ignore_errors=True)
        shutil.copytree(backup_path, self.namenode_dir)
        
        # Start NameNode
        subprocess.run([f'{self.hadoop_home}/sbin/hadoop-daemon.sh', 'start', 'namenode'])
    
    def recover_from_secondary(self, secondary_dir):
        """Recover using Secondary NameNode checkpoint"""
        # Stop NameNode
        subprocess.run([f'{self.hadoop_home}/sbin/hadoop-daemon.sh', 'stop', 'namenode'])
        
        # Copy from Secondary
        for item in ['fsimage', 'edits']:
            src = f"{secondary_dir}/current/{item}*"
            subprocess.run(f'cp {src} {self.namenode_dir}/current/', shell=True)
        
        # Start with recovery flag
        subprocess.run([
            f'{self.hadoop_home}/bin/hdfs', 'namenode', '-recover', '-force'
        ])

# HA Failover
def trigger_ha_failover(active_nn, standby_nn):
    """Manual HA failover"""
    cmd = ['hdfs', 'haadmin', '-failover', active_nn, standby_nn]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0
```

---

## Question 4
- [ ] Done

**You need to process real-time data in Hadoop. How would you architect this?**

### Answer

### Architecture Options

| Approach | Latency | Complexity |
|----------|---------|------------|
| Kafka + Spark Streaming | Seconds | Medium |
| Kafka + Flink | Milliseconds | High |
| Kafka + Storm | Milliseconds | Medium |
| Flume + Spark | Minutes | Low |

### Lambda Architecture

| Layer | Purpose |
|-------|---------|
| **Batch** | Historical accuracy (HDFS + MapReduce) |
| **Speed** | Real-time updates (Spark/Flink) |
| **Serving** | Query results (HBase/Druid) |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

def create_streaming_pipeline():
    """
    Pipeline:
    1. Read from Kafka
    2. Process in micro-batches
    3. Write to HDFS and HBase
    """
    
    spark = SparkSession.builder \
        .appName("RealTimeProcessing") \
        .getOrCreate()
    
    # Define schema
    schema = StructType([
        StructField("event_id", StringType()),
        StructField("timestamp", TimestampType()),
        StructField("user_id", StringType()),
        StructField("action", StringType()),
        StructField("value", DoubleType())
    ])
    
    # Read from Kafka
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:9092") \
        .option("subscribe", "events") \
        .load()
    
    # Parse JSON
    events = df.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*")
    
    # Process - aggregate by user per minute
    aggregated = events \
        .withWatermark("timestamp", "1 minute") \
        .groupBy(
            window("timestamp", "1 minute"),
            "user_id"
        ).agg(
            count("*").alias("event_count"),
            sum("value").alias("total_value")
        )
    
    # Write to HDFS (batch layer)
    hdfs_query = aggregated.writeStream \
        .format("parquet") \
        .option("path", "hdfs:///data/events/") \
        .option("checkpointLocation", "hdfs:///checkpoints/events/") \
        .outputMode("append") \
        .trigger(processingTime="1 minute") \
        .start()
    
    # Write to console (for debugging)
    console_query = aggregated.writeStream \
        .format("console") \
        .outputMode("update") \
        .start()
    
    spark.streams.awaitAnyTermination()

if __name__ == '__main__':
    create_streaming_pipeline()
```

---

## Question 5
- [ ] Done

**Your MapReduce job is failing with OutOfMemoryError. How do you fix it?**

### Answer

### Diagnosis

| Check | Command/Location |
|-------|------------------|
| Task logs | YARN container logs |
| Heap usage | GC logs |
| Container limits | YARN configuration |
| Data per mapper | Input split sizes |

### Solutions

| Solution | When to Use |
|----------|-------------|
| Increase memory | GC overhead high |
| Reduce split size | Large records |
| Use combiner | High mapper output |
| Stream processing | Can't fit in memory |
| Spill settings | Sort buffer too small |

### Python Code Example
```python
from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol

class MemoryEfficientJob(MRJob):
    """Memory-optimized MapReduce job"""
    
    # Increase memory settings
    JOBCONF = {
        # Container memory
        'mapreduce.map.memory.mb': '4096',
        'mapreduce.reduce.memory.mb': '8192',
        
        # JVM heap (should be ~80% of container memory)
        'mapreduce.map.java.opts': '-Xmx3276m',
        'mapreduce.reduce.java.opts': '-Xmx6553m',
        
        # Sort buffer (increase for fewer spills)
        'mapreduce.task.io.sort.mb': '512',
        'mapreduce.task.io.sort.factor': '100',
        
        # Reduce memory-intensive operations
        'mapreduce.reduce.shuffle.input.buffer.percent': '0.5',
        
        # Enable compression
        'mapreduce.map.output.compress': 'true',
    }
    
    def mapper(self, _, line):
        """Process line by line without storing"""
        # Don't accumulate data in memory
        parts = line.strip().split(',')
        if len(parts) >= 2:
            yield parts[0], float(parts[1])
    
    def combiner(self, key, values):
        """Reduce data volume before shuffle"""
        total = 0
        count = 0
        for v in values:
            total += v
            count += 1
        yield key, (total, count)
    
    def reducer(self, key, values):
        """Stream processing without collecting all"""
        total = 0
        count = 0
        for v in values:
            if isinstance(v, tuple):
                total += v[0]
                count += v[1]
            else:
                total += v
                count += 1
        yield key, total / count if count > 0 else 0

# Alternative: Process large files in chunks
class ChunkedProcessor(MRJob):
    """Process large records in chunks"""
    
    INTERNAL_PROTOCOL = RawValueProtocol
    
    def mapper_init(self):
        self.buffer = []
        self.buffer_size = 1000
    
    def mapper(self, _, line):
        self.buffer.append(line)
        if len(self.buffer) >= self.buffer_size:
            yield from self.process_buffer()
            self.buffer = []
    
    def mapper_final(self):
        if self.buffer:
            yield from self.process_buffer()
    
    def process_buffer(self):
        """Process accumulated records"""
        for line in self.buffer:
            yield 'key', 1

if __name__ == '__main__':
    MemoryEfficientJob.run()
```

---

## Question 6
- [ ] Done

**How would you handle schema evolution in a Hadoop data lake?**

### Answer

### Strategies

| Strategy | Pros | Cons |
|----------|------|------|
| **Avro** | Schema in file, backward compatible | Learning curve |
| **Parquet** | Column pruning, schema evolution | Complex updates |
| **Hive metastore** | Central schema registry | Extra component |
| **Delta Lake** | ACID, time travel | Spark dependency |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.types import *

class SchemaEvolution:
    def __init__(self, spark):
        self.spark = spark
    
    def write_with_schema(self, df, path, format='parquet'):
        """Write data with schema"""
        df.write \
            .format(format) \
            .mode('append') \
            .option('mergeSchema', 'true') \
            .save(path)
    
    def read_with_schema_merge(self, path):
        """Read with merged schema"""
        return self.spark.read \
            .option('mergeSchema', 'true') \
            .parquet(path)
    
    def add_column_safely(self, df, col_name, col_type, default_value):
        """Add new column with default"""
        from pyspark.sql.functions import lit
        if col_name not in df.columns:
            df = df.withColumn(col_name, lit(default_value).cast(col_type))
        return df
    
    def migrate_schema(self, old_path, new_path, transformations):
        """Migrate data with schema changes"""
        df = self.spark.read.parquet(old_path)
        
        for transform in transformations:
            if transform['type'] == 'rename':
                df = df.withColumnRenamed(transform['old'], transform['new'])
            elif transform['type'] == 'add':
                df = self.add_column_safely(
                    df, transform['name'], 
                    transform['dtype'], transform['default']
                )
            elif transform['type'] == 'drop':
                df = df.drop(transform['name'])
        
        df.write.mode('overwrite').parquet(new_path)

# Usage
spark = SparkSession.builder.appName("SchemaEvolution").getOrCreate()
schema_mgr = SchemaEvolution(spark)

# Example: Add new column
transformations = [
    {'type': 'add', 'name': 'new_field', 'dtype': 'string', 'default': 'unknown'},
    {'type': 'rename', 'old': 'old_name', 'new': 'new_name'}
]
schema_mgr.migrate_schema('/data/v1', '/data/v2', transformations)
```

---

## Question 7
- [ ] Done

**How do you secure a Hadoop cluster?**

### Answer

### Security Components

| Component | Purpose |
|-----------|---------|
| **Kerberos** | Authentication |
| **Ranger/Sentry** | Authorization |
| **Knox** | Gateway/perimeter security |
| **TLS/SSL** | Encryption in transit |
| **HDFS encryption** | Encryption at rest |

### Security Checklist

| Layer | Implementation |
|-------|----------------|
| Authentication | Kerberos enabled |
| Authorization | ACLs + Ranger policies |
| Encryption | TLS for RPC, HDFS encryption zones |
| Auditing | Ranger audit, HDFS audit logs |
| Network | Firewall rules, Knox gateway |

### Python Code Example
```python
import subprocess
from hdfs import InsecureClient
from hdfs.ext.kerberos import KerberosClient

class SecureHDFSClient:
    def __init__(self, namenode_url, principal=None, keytab=None):
        self.namenode_url = namenode_url
        self.principal = principal
        self.keytab = keytab
    
    def kinit(self):
        """Authenticate with Kerberos"""
        if self.principal and self.keytab:
            cmd = ['kinit', '-kt', self.keytab, self.principal]
            subprocess.run(cmd, check=True)
    
    def get_client(self):
        """Get authenticated HDFS client"""
        if self.principal:
            self.kinit()
            return KerberosClient(self.namenode_url)
        return InsecureClient(self.namenode_url)
    
    def setup_encryption_zone(self, path, key_name):
        """Create encryption zone"""
        # First create encryption key
        subprocess.run([
            'hadoop', 'key', 'create', key_name
        ], check=True)
        
        # Create encryption zone
        subprocess.run([
            'hdfs', 'crypto', '-createZone', 
            '-keyName', key_name, 
            '-path', path
        ], check=True)
    
    def set_acl(self, path, user, permissions):
        """Set HDFS ACL"""
        acl_spec = f"user:{user}:{permissions}"
        subprocess.run([
            'hdfs', 'dfs', '-setfacl', '-m', acl_spec, path
        ], check=True)

# Usage
secure_client = SecureHDFSClient(
    namenode_url='http://namenode:50070',
    principal='hdfs@REALM.COM',
    keytab='/etc/security/keytabs/hdfs.keytab'
)

client = secure_client.get_client()
# Now perform secure operations
```

### Best Practices
- Enable Kerberos for all services
- Use service principals for automation
- Implement least-privilege access
- Encrypt sensitive data at rest
- Enable comprehensive audit logging
- Regular security assessments

## Question 8
- [ ] Done

**How would you recover a Hadoop cluster from a Namenode failure ?**

*Answer to be added.*

---

## Question 9
- [ ] Done

**What considerations should be made for Hadoop cluster backup and disaster recovery ?**

*Answer to be added.*

---

## Question 10
- [ ] Done

**How would you monitor the health of a Hadoop cluster , and what tools would you use?**

*Answer to be added.*

---

## Question 11
- [ ] Done

**Discuss a strategy for Hadoop cluster capacity planning and scaling**

*Answer to be added.*

---

## Question 12
- [ ] Done

**A company wants to process clickstream data in real-time . How would you integrate Hadoop and Spark to meet this requirement?**

*Answer to be added.*

---

## Question 13
- [ ] Done

**Propose a data pipeline using Hadoop components to manage and analyze sensor data from IoT devices**

*Answer to be added.*

---
