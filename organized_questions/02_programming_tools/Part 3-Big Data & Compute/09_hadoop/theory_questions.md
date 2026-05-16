# Hadoop Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What is Hadoop and what are its core components?**

### Answer

**Definition**: Hadoop is an open-source framework for distributed storage and processing of large datasets across clusters of computers using simple programming models.

### Core Components

| Component | Description |
|-----------|-------------|
| **HDFS** | Hadoop Distributed File System - storage layer |
| **MapReduce** | Processing engine for batch processing |
| **YARN** | Yet Another Resource Negotiator - resource management |
| **Hadoop Common** | Utilities and libraries |

### HDFS Architecture

| Component | Role |
|-----------|------|
| NameNode | Master - manages metadata |
| DataNode | Slave - stores actual data |
| Secondary NameNode | Checkpoint for NameNode |

### Python Code Example (using hdfs library)
```python
from hdfs import InsecureClient

# Connect to HDFS
client = InsecureClient('http://namenode:50070', user='hadoop')

# List directory contents
files = client.list('/')
print(f"Root files: {files}")

# Upload file
client.upload('/user/data/', 'local_file.txt')

# Download file
client.download('/user/data/file.txt', 'local_copy.txt')

# Read file content
with client.read('/user/data/file.txt') as reader:
    content = reader.read()
```

---

## Question 2
- [ ] Done

**Explain the HDFS architecture in detail.**

### Answer

### HDFS Components

| Component | Function |
|-----------|----------|
| **NameNode** | Stores metadata (file names, block locations) |
| **DataNode** | Stores actual data blocks |
| **Block** | Default 128MB chunks |
| **Replication** | Default 3 copies for fault tolerance |

### Data Flow

| Operation | Flow |
|-----------|------|
| Write | Client → NameNode → DataNodes (pipeline) |
| Read | Client → NameNode → DataNode |

### Python Code Example
```python
from hdfs import InsecureClient

client = InsecureClient('http://namenode:50070', user='hadoop')

# Check file status
status = client.status('/user/data/large_file.csv')
print(f"File size: {status['length']} bytes")
print(f"Replication: {status['replication']}")
print(f"Block size: {status['blockSize']}")

# Create directory
client.makedirs('/user/new_directory')

# Delete file
client.delete('/user/old_file.txt')

# Rename/Move
client.rename('/user/old_path', '/user/new_path')

# Set replication factor
client.set_replication('/user/important_file.txt', replication=5)
```

### HDFS Design Principles
- **Write-once, read-many**: Optimized for batch processing
- **Large files**: Designed for GB/TB size files
- **Streaming access**: Sequential reads preferred
- **Commodity hardware**: Fault tolerance through replication

---

## Question 3
- [ ] Done

**What is MapReduce and how does it work?**

### Answer

**Definition**: MapReduce is a programming model for processing large datasets in parallel across a Hadoop cluster.

### Phases

| Phase | Description |
|-------|-------------|
| **Map** | Process input, emit key-value pairs |
| **Shuffle** | Group by key, sort |
| **Reduce** | Aggregate values for each key |

### Python Code Example (using mrjob)
```python
from mrjob.job import MRJob
from mrjob.step import MRStep

# Word Count Example
class WordCount(MRJob):
    
    def mapper(self, _, line):
        """Map phase: emit (word, 1) for each word"""
        for word in line.strip().split():
            yield word.lower(), 1
    
    def reducer(self, word, counts):
        """Reduce phase: sum counts for each word"""
        yield word, sum(counts)

# Run: python wordcount.py input.txt

# Advanced: Multi-step MapReduce
class TopWords(MRJob):
    
    def steps(self):
        return [
            MRStep(mapper=self.mapper_count,
                   reducer=self.reducer_count),
            MRStep(reducer=self.reducer_top)
        ]
    
    def mapper_count(self, _, line):
        for word in line.strip().split():
            yield word.lower(), 1
    
    def reducer_count(self, word, counts):
        yield None, (sum(counts), word)
    
    def reducer_top(self, _, word_counts):
        # Get top 10 words
        top_10 = sorted(word_counts, reverse=True)[:10]
        for count, word in top_10:
            yield word, count

if __name__ == '__main__':
    WordCount.run()
```

---

## Question 4
- [ ] Done

**What is YARN and what are its components?**

### Answer

**Definition**: YARN (Yet Another Resource Negotiator) is Hadoop's resource management layer that separates resource management from processing.

### Components

| Component | Description |
|-----------|-------------|
| **ResourceManager** | Master - allocates resources |
| **NodeManager** | Slave - manages containers on each node |
| **ApplicationMaster** | Per-app coordinator |
| **Container** | Resource allocation unit |

### YARN vs MapReduce 1.0

| Aspect | MapReduce 1.0 | YARN |
|--------|---------------|------|
| Resource Management | JobTracker | ResourceManager |
| Processing | MapReduce only | Multiple frameworks |
| Scalability | ~4000 nodes | ~10000 nodes |

### Python Code Example
```python
# Using YARN REST API
import requests
import json

YARN_RM_URL = "http://resourcemanager:8088"

# Get cluster info
def get_cluster_info():
    response = requests.get(f"{YARN_RM_URL}/ws/v1/cluster/info")
    return response.json()

# Get cluster metrics
def get_cluster_metrics():
    response = requests.get(f"{YARN_RM_URL}/ws/v1/cluster/metrics")
    metrics = response.json()['clusterMetrics']
    return {
        'activeNodes': metrics['activeNodes'],
        'totalMemory': metrics['totalMB'],
        'availableMemory': metrics['availableMB'],
        'appsRunning': metrics['appsRunning']
    }

# List applications
def list_applications(state='RUNNING'):
    response = requests.get(
        f"{YARN_RM_URL}/ws/v1/cluster/apps",
        params={'state': state}
    )
    return response.json()

# Kill application
def kill_application(app_id):
    response = requests.put(
        f"{YARN_RM_URL}/ws/v1/cluster/apps/{app_id}/state",
        json={'state': 'KILLED'}
    )
    return response.status_code == 200

print(get_cluster_metrics())
```

---

## Question 5
- [ ] Done

**What is the difference between HDFS and traditional file systems?**

### Answer

### Comparison

| Aspect | HDFS | Traditional FS |
|--------|------|----------------|
| **File Size** | GB to TB | KB to GB |
| **Access Pattern** | Write-once, read-many | Random read/write |
| **Block Size** | 128MB default | 4KB typical |
| **Replication** | Built-in (3x) | RAID or manual |
| **Fault Tolerance** | Automatic | Hardware dependent |
| **Scalability** | Thousands of nodes | Single machine |

### When to Use HDFS

| Use Case | HDFS Suitable |
|----------|---------------|
| Log analysis | ✅ Yes |
| Random access | ❌ No |
| Large batch processing | ✅ Yes |
| Low-latency queries | ❌ No |
| Data archival | ✅ Yes |

### Python Code Example
```python
from hdfs import InsecureClient
import os

# HDFS operations vs local file system
class FileSystemComparison:
    def __init__(self, hdfs_url, hdfs_user):
        self.hdfs = InsecureClient(hdfs_url, user=hdfs_user)
    
    def upload_to_hdfs(self, local_path, hdfs_path):
        """Upload local file to HDFS"""
        self.hdfs.upload(hdfs_path, local_path)
        
        # Verify
        status = self.hdfs.status(hdfs_path)
        print(f"Uploaded: {status['length']} bytes")
        print(f"Block size: {status['blockSize']} bytes")
        print(f"Replication: {status['replication']}")
    
    def compare_read_performance(self, hdfs_path, local_path):
        """Compare read performance"""
        import time
        
        # HDFS read
        start = time.time()
        with self.hdfs.read(hdfs_path) as reader:
            hdfs_data = reader.read()
        hdfs_time = time.time() - start
        
        # Local read
        start = time.time()
        with open(local_path, 'rb') as f:
            local_data = f.read()
        local_time = time.time() - start
        
        print(f"HDFS read: {hdfs_time:.2f}s")
        print(f"Local read: {local_time:.2f}s")
```

---

## Question 6
- [ ] Done

**Explain data replication in HDFS.**

### Answer

### Replication Strategy

| Factor | Default | Description |
|--------|---------|-------------|
| Replication Factor | 3 | Number of copies |
| Rack Awareness | Enabled | Distributes across racks |
| Block Placement | Optimized | 1 local, 2 remote |

### Block Placement Policy
1. First replica: Same node as writer (or random if external)
2. Second replica: Different rack
3. Third replica: Same rack as second, different node

### Python Code Example
```python
from hdfs import InsecureClient

client = InsecureClient('http://namenode:50070', user='hadoop')

# Set replication for specific file
def set_replication(path, factor):
    """Change replication factor"""
    client.set_replication(path, replication=factor)
    status = client.status(path)
    print(f"New replication: {status['replication']}")

# Check replication status
def check_replication(path):
    """Check if file is under-replicated"""
    status = client.status(path)
    current = status['replication']
    
    # Using WebHDFS to get block info
    import requests
    response = requests.get(
        f'http://namenode:50070/webhdfs/v1{path}?op=GETFILEBLOCKLOCATIONS'
    )
    blocks = response.json()
    
    for block in blocks.get('BlockLocations', {}).get('BlockLocation', []):
        actual_replicas = len(block.get('hosts', []))
        if actual_replicas < current:
            print(f"Block under-replicated: {actual_replicas}/{current}")

# Different replication for different data types
def upload_with_replication(local_path, hdfs_path, data_type):
    """Upload with appropriate replication"""
    replication_config = {
        'critical': 5,
        'normal': 3,
        'temporary': 1
    }
    
    client.upload(hdfs_path, local_path)
    client.set_replication(hdfs_path, replication_config.get(data_type, 3))
```

---

## Question 7
- [ ] Done

**What is NameNode federation and High Availability?**

### Answer

### NameNode High Availability (HA)

| Component | Role |
|-----------|------|
| Active NameNode | Handles all client operations |
| Standby NameNode | Ready to take over |
| JournalNodes | Store edit logs (quorum) |
| ZooKeeper | Automatic failover |

### Federation

| Feature | Description |
|---------|-------------|
| Multiple NameNodes | Each manages namespace portion |
| Block Pools | Each NameNode has its own |
| Scalability | Horizontal namespace scaling |

### Python Code Example
```python
import requests
from kazoo.client import KazooClient

class HDFSHAClient:
    def __init__(self, namenodes, zk_hosts):
        self.namenodes = namenodes  # List of NameNode URLs
        self.zk = KazooClient(hosts=zk_hosts)
        self.zk.start()
    
    def get_active_namenode(self):
        """Find active NameNode"""
        for nn in self.namenodes:
            try:
                response = requests.get(f"{nn}/jmx?qry=Hadoop:service=NameNode,name=NameNodeStatus")
                status = response.json()['beans'][0]['State']
                if status == 'active':
                    return nn
            except:
                continue
        return None
    
    def check_health(self):
        """Check cluster health"""
        active = self.get_active_namenode()
        if not active:
            return {'status': 'CRITICAL', 'message': 'No active NameNode'}
        
        response = requests.get(f"{active}/jmx?qry=Hadoop:service=NameNode,name=FSNamesystem")
        metrics = response.json()['beans'][0]
        
        return {
            'status': 'OK',
            'activeNN': active,
            'totalBlocks': metrics['BlocksTotal'],
            'missingBlocks': metrics['MissingBlocks'],
            'underReplicatedBlocks': metrics['UnderReplicatedBlocks']
        }
    
    def trigger_failover(self):
        """Manual failover (use with caution)"""
        # This would typically use hdfs haadmin command
        import subprocess
        result = subprocess.run(
            ['hdfs', 'haadmin', '-failover', 'nn1', 'nn2'],
            capture_output=True, text=True
        )
        return result.returncode == 0
```

---

## Question 8
- [ ] Done

**What are the different input formats in Hadoop?**

### Answer

### Common Input Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| TextInputFormat | Line-by-line text | Log files |
| KeyValueTextInputFormat | Tab-separated K-V | Structured text |
| SequenceFileInputFormat | Binary K-V | Intermediate data |
| NLineInputFormat | N lines per split | Fixed-size splits |
| CombineFileInputFormat | Combine small files | Many small files |

### Python Code Example (mrjob)
```python
from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol, RawValueProtocol
import json
import csv

# Processing different input formats

# 1. JSON Lines format
class ProcessJSON(MRJob):
    INPUT_PROTOCOL = RawValueProtocol
    
    def mapper(self, _, line):
        record = json.loads(line)
        yield record['category'], record['amount']
    
    def reducer(self, category, amounts):
        yield category, sum(amounts)

# 2. CSV format
class ProcessCSV(MRJob):
    
    def mapper(self, _, line):
        row = next(csv.reader([line]))
        if len(row) >= 3:
            yield row[0], float(row[2])  # category, amount
    
    def reducer(self, key, values):
        yield key, sum(values)

# 3. Custom delimiter
class ProcessCustomDelimiter(MRJob):
    
    def mapper(self, _, line):
        fields = line.split('|')  # Pipe delimiter
        yield fields[0], 1
    
    def reducer(self, key, counts):
        yield key, sum(counts)

# 4. Multi-line records (using combiner)
class ProcessMultiLine(MRJob):
    
    def mapper_init(self):
        self.buffer = []
    
    def mapper(self, _, line):
        if line.startswith('---'):  # Record separator
            if self.buffer:
                record = '\n'.join(self.buffer)
                yield 'record', record
                self.buffer = []
        else:
            self.buffer.append(line)
    
    def mapper_final(self):
        if self.buffer:
            yield 'record', '\n'.join(self.buffer)

if __name__ == '__main__':
    ProcessJSON.run()
```

---

## Question 9
- [ ] Done

**Explain the Hadoop ecosystem components.**

### Answer

### Ecosystem Overview

| Component | Category | Purpose |
|-----------|----------|---------|
| **Hive** | SQL | SQL-like queries |
| **Pig** | Scripting | Data flow language |
| **HBase** | NoSQL | Real-time random access |
| **Spark** | Processing | In-memory processing |
| **Sqoop** | Data Transfer | RDBMS ↔ HDFS |
| **Flume** | Ingestion | Log collection |
| **Kafka** | Streaming | Message queue |
| **Oozie** | Workflow | Job scheduling |
| **ZooKeeper** | Coordination | Distributed coordination |

### Python Code Example
```python
# Working with Hadoop ecosystem

# 1. Hive with PyHive
from pyhive import hive

conn = hive.Connection(host='hiveserver', port=10000, database='default')
cursor = conn.cursor()

cursor.execute('SELECT * FROM sales LIMIT 10')
for row in cursor.fetchall():
    print(row)

# 2. HBase with happybase
import happybase

connection = happybase.Connection('hbase-master')
table = connection.table('users')

# Put data
table.put(b'user1', {b'info:name': b'John', b'info:age': b'30'})

# Get data
row = table.row(b'user1')
print(row)

# Scan
for key, data in table.scan(row_prefix=b'user'):
    print(key, data)

# 3. Sqoop-like data transfer
import subprocess

def sqoop_import(jdbc_url, table, target_dir, username, password):
    """Import from RDBMS to HDFS"""
    cmd = [
        'sqoop', 'import',
        '--connect', jdbc_url,
        '--username', username,
        '--password', password,
        '--table', table,
        '--target-dir', target_dir,
        '--num-mappers', '4'
    ]
    subprocess.run(cmd, check=True)

# 4. Oozie workflow submission
def submit_oozie_workflow(oozie_url, properties):
    """Submit Oozie workflow"""
    import requests
    
    response = requests.post(
        f"{oozie_url}/v1/jobs",
        params={'action': 'start'},
        data=properties
    )
    return response.json()['id']
```

---

## Question 10
- [ ] Done

**What is speculative execution in Hadoop?**

### Answer

**Definition**: Speculative execution launches backup copies of slow-running tasks to reduce job completion time.

### How It Works

| Step | Description |
|------|-------------|
| Monitor | Track task progress |
| Detect | Identify slow tasks |
| Launch | Start backup task |
| Kill | Terminate slower one |

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mapreduce.map.speculative` | true | Enable for mappers |
| `mapreduce.reduce.speculative` | true | Enable for reducers |

### Python Code Example
```python
from mrjob.job import MRJob
from mrjob.step import MRStep
import time
import random

class JobWithSpeculative(MRJob):
    """Example showing when speculative execution helps"""
    
    JOBCONF = {
        'mapreduce.map.speculative': 'true',
        'mapreduce.reduce.speculative': 'true',
        'mapreduce.job.speculative.slowtaskthreshold': '1.0',  # 100% slower than average
    }
    
    def mapper(self, _, line):
        # Simulate variable processing time
        # In real scenarios, this could be due to:
        # - Bad hardware
        # - Network issues
        # - Data skew
        
        if random.random() < 0.01:  # 1% chance of being slow
            time.sleep(10)  # Slow task
        
        for word in line.split():
            yield word, 1
    
    def reducer(self, word, counts):
        yield word, sum(counts)

# Monitoring speculative execution
class SpeculativeMonitor:
    def __init__(self, yarn_url):
        self.yarn_url = yarn_url
    
    def get_speculative_tasks(self, app_id):
        """Get info about speculative task attempts"""
        import requests
        
        response = requests.get(
            f"{self.yarn_url}/ws/v1/history/mapreduce/jobs/{app_id}/tasks"
        )
        tasks = response.json()['tasks']['task']
        
        speculative = []
        for task in tasks:
            if task.get('successfulAttempt') != task.get('id') + '_0':
                # Original attempt wasn't successful
                speculative.append({
                    'taskId': task['id'],
                    'type': task['type'],
                    'elapsedTime': task['elapsedTime']
                })
        
        return speculative

# When to disable speculative execution
"""
Disable when:
1. Tasks have side effects (writing to external systems)
2. Tasks are expensive to restart
3. Cluster is heavily loaded
4. Tasks are non-idempotent
"""

if __name__ == '__main__':
    JobWithSpeculative.run()
```

### Best Practices
- Enable for CPU-bound tasks
- Disable for tasks with external side effects
- Monitor cluster resources
- Consider data locality impact


---

# --- Missing Questions Restored from Source (Q11-Q30) ---

## Question 11
- [ ] Done

**Explain how Apache Flume helps with log and event data collection for Hadoop**

**Answer:**

### Definition
Apache Flume is a distributed, reliable service for collecting, aggregating, and moving **large amounts of log/event data** into HDFS or other centralized data stores.

### Architecture

| Component | Role |
|-----------|------|
| **Source** | Ingests data (log files, HTTP, Kafka, syslog) |
| **Channel** | Buffers between source and sink (memory or file) |
| **Sink** | Writes to destination (HDFS, HBase, Kafka) |
| **Agent** | JVM process containing source → channel → sink |

### Data Flow
```
Web Servers ──┐
               ├──▶ Flume Agent (Source → Channel → Sink) ──▶ HDFS
App Logs ─────┘
```

### Configuration Example
```properties
# flume-conf.properties
agent.sources = weblog
agent.channels = memchannel
agent.sinks = hdfssink

# Source: tail log files
agent.sources.weblog.type = exec
agent.sources.weblog.command = tail -F /var/log/apache/access.log
agent.sources.weblog.channels = memchannel

# Channel: in-memory buffer
agent.channels.memchannel.type = memory
agent.channels.memchannel.capacity = 10000

# Sink: write to HDFS
agent.sinks.hdfssink.type = hdfs
agent.sinks.hdfssink.hdfs.path = /user/logs/%Y/%m/%d
agent.sinks.hdfssink.hdfs.fileType = DataStream
agent.sinks.hdfssink.channel = memchannel
```

### Interview Tip
Flume provides **at-least-once delivery** with file channels (durable) or **best-effort** with memory channels (faster). In modern stacks, Apache Kafka has largely replaced Flume for event streaming, but Flume is still used for direct-to-HDFS ingestion.

---

## Question 12
- [ ] Done

**What is Apache Sqoop and how does it interact with Hadoop ?**

**Answer:**

### Definition
Apache Sqoop (SQL-to-Hadoop) is a tool for efficiently transferring **bulk data between relational databases (RDBMS) and Hadoop** (HDFS, Hive, HBase).

### Operations

| Operation | Direction | Command |
|-----------|-----------|--------|
| **Import** | RDBMS → HDFS | `sqoop import` |
| **Export** | HDFS → RDBMS | `sqoop export` |
| **List databases** | Check RDBMS | `sqoop list-databases` |
| **List tables** | Check RDBMS | `sqoop list-tables` |

### Code Examples
```bash
# Import from MySQL to HDFS
sqoop import \
  --connect jdbc:mysql://dbserver/mydb \
  --username root --password secret \
  --table employees \
  --target-dir /user/hadoop/employees \
  --num-mappers 4 \
  --split-by emp_id

# Import into Hive table
sqoop import \
  --connect jdbc:mysql://dbserver/mydb \
  --table sales \
  --hive-import \
  --hive-table sales_data \
  --incremental append \
  --check-column id \
  --last-value 1000

# Export from HDFS to MySQL
sqoop export \
  --connect jdbc:mysql://dbserver/mydb \
  --table results \
  --export-dir /user/hadoop/output \
  --input-fields-terminated-by ','
```

### How It Works
- Uses **MapReduce** for parallel data transfer
- Each mapper handles a portion of the data
- `--split-by` determines how to partition the import across mappers
- Supports incremental imports (`--incremental append/lastmodified`)

### Interview Tip
Sqoop is being deprecated in favor of **Apache Spark JDBC** connectors and tools like **Apache NiFi**. However, it's still widely used in legacy Hadoop clusters. Key advantage: it uses MapReduce for parallelism, so a 4-mapper import is ~4x faster than a single JDBC connection.

---

## Question 13
- [ ] Done

**How does Apache Oozie help in workflow scheduling in Hadoop ?**

**Answer:**

### Definition
Apache Oozie is a **workflow scheduler** for managing and orchestrating Hadoop jobs (MapReduce, Pig, Hive, Spark, etc.) as directed acyclic graphs (DAGs).

### Workflow Types

| Type | Purpose | Trigger |
|------|---------|--------|
| **Workflow** | Sequential/parallel job execution | Manual or programmatic |
| **Coordinator** | Time/data-triggered recurring workflows | Cron-like schedule |
| **Bundle** | Group of coordinators | Manage related pipelines |

### Workflow XML Example
```xml
<workflow-app name="etl-pipeline" xmlns="uri:oozie:workflow:0.5">
    <start to="extract"/>
    
    <action name="extract">
        <sqoop xmlns="uri:oozie:sqoop-action:0.2">
            <command>import --connect jdbc:mysql://db/sales --table orders --target-dir /data/raw</command>
        </sqoop>
        <ok to="transform"/>
        <error to="fail"/>
    </action>
    
    <action name="transform">
        <hive xmlns="uri:oozie:hive-action:0.2">
            <script>transform.hql</script>
        </hive>
        <ok to="load"/>
        <error to="fail"/>
    </action>
    
    <action name="load">
        <spark xmlns="uri:oozie:spark-action:0.1">
            <master>yarn</master>
            <jar>analytics.jar</jar>
        </spark>
        <ok to="end"/>
        <error to="fail"/>
    </action>
    
    <kill name="fail"><message>Pipeline failed</message></kill>
    <end name="end"/>
</workflow-app>
```

### Coordinator (Scheduled)
```xml
<coordinator-app name="daily-etl" frequency="${coord:days(1)}">
    <action>
        <workflow>
            <app-path>/user/oozie/etl-pipeline</app-path>
        </workflow>
    </action>
</coordinator-app>
```

### Interview Tip
Oozie is Hadoop-native but verbose (XML-based). Modern alternatives include **Apache Airflow** (Python DAGs, more flexible), **Luigi**, and **Prefect**. However, Oozie integrates tightly with Hadoop security (Kerberos) and YARN, making it still relevant in enterprise Hadoop clusters.

---

## Question 14
- [ ] Done

**What is Apache ZooKeeper and why is it important for Hadoop ?**

**Answer:**

### Definition
Apache ZooKeeper is a centralized service for **distributed coordination** — it provides configuration management, naming, synchronization, and group services for distributed systems.

### Role in Hadoop

| Function | Description |
|----------|-------------|
| **Leader election** | NameNode HA (active/standby failover) |
| **Configuration management** | Centralized config for cluster services |
| **Distributed locking** | Prevent concurrent modifications |
| **Service discovery** | Track which services are alive |
| **Barrier synchronization** | Coordinate distributed processes |

### Architecture
```
Client 1 ──┐                    ┌─ ZK Node 1 (Leader)
Client 2 ──├─▶ ZooKeeper Ensemble ├─ ZK Node 2 (Follower)
Client 3 ──┘                    └─ ZK Node 3 (Follower)
```

- **Ensemble**: Cluster of ZooKeeper nodes (odd number: 3, 5, 7)
- **Quorum**: Majority must agree (3/5 nodes = tolerates 2 failures)
- **ZNodes**: Hierarchical data nodes (like a file system)

### Hadoop HA with ZooKeeper
```xml
<!-- hdfs-site.xml: NameNode HA configuration -->
<property>
    <name>dfs.ha.automatic-failover.enabled</name>
    <value>true</value>
</property>
<property>
    <name>ha.zookeeper.quorum</name>
    <value>zk1:2181,zk2:2181,zk3:2181</value>
</property>
```

### Services That Use ZooKeeper
- **HDFS HA**: NameNode failover
- **YARN HA**: ResourceManager failover
- **HBase**: Region server coordination
- **Kafka**: Broker management (legacy, now KRaft)
- **Hive**: Lock management

### Interview Tip
ZooKeeper solves the fundamental problem of **distributed consensus** — how multiple nodes agree on state. In Hadoop HA, it detects NameNode failure and triggers automatic failover to the standby. Key fact: ZooKeeper requires an **odd number** of nodes to form a quorum.

---

## Question 15
- [ ] Done

**How does Hadoop handle the failure of a datanode ?**

**Answer:**

### Detection Mechanism
The NameNode detects DataNode failures through **heartbeat signals**.

| Step | What Happens |
|------|--------------|
| 1. **Heartbeat timeout** | DataNode stops sending heartbeats (default: every 3 seconds) |
| 2. **Marked dead** | NameNode marks DataNode as dead after ~10 minutes (configurable) |
| 3. **Under-replicated blocks** | NameNode identifies blocks that lost a replica |
| 4. **Re-replication** | NameNode instructs other DataNodes to copy under-replicated blocks |
| 5. **Rack awareness** | New replicas placed according to rack-aware policy |

### Process Flow
```
DataNode X dies
    ↓
NameNode detects (no heartbeat for dfs.namenode.heartbeat.recheck-interval)
    ↓
NameNode scans block map for blocks stored on DataNode X
    ↓
Blocks with replicas < dfs.replication (default 3) are marked under-replicated
    ↓
NameNode schedules re-replication on healthy DataNodes
    ↓
Replication factor restored (transparent to clients)
```

### Configuration
```xml
<!-- hdfs-site.xml -->
<property>
    <name>dfs.heartbeat.interval</name>
    <value>3</value>  <!-- Heartbeat every 3 seconds -->
</property>
<property>
    <name>dfs.namenode.heartbeat.recheck-interval</name>
    <value>300000</value>  <!-- 5 minutes recheck -->
</property>
<property>
    <name>dfs.replication</name>
    <value>3</value>  <!-- Default replication factor -->
</property>
```

### Key Points
- **No data loss** as long as at least 1 replica survives
- **Rack awareness** ensures replicas are on different racks
- **Automatic recovery** — no manual intervention needed
- **Decommissioning** allows graceful removal of nodes

### Interview Tip
The NameNode doesn't immediately declare a DataNode dead after missing one heartbeat — it waits for `2 * heartbeat.recheck-interval + 10 * heartbeat.interval` (default ~10.5 minutes). This avoids false positives from network blips. Mention that this is why Hadoop favors **high replication** over single-copy storage.

---

## Question 16
- [ ] Done

**Explain the process of data replication in HDFS**

**Answer:**

### Definition
HDFS replicates each data block across multiple DataNodes to ensure **fault tolerance** and **data availability**.

### Replication Process

| Step | Action |
|------|--------|
| 1. **Client writes** | Client sends block to first DataNode |
| 2. **Pipeline replication** | First DN forwards to second, second to third |
| 3. **Acknowledgment** | ACKs flow back through pipeline |
| 4. **NameNode metadata** | NameNode records block locations |

### Pipeline Architecture
```
Client → DataNode 1 → DataNode 2 → DataNode 3
              ← ACK  ←  ACK   ← ACK
```

### Rack-Aware Replica Placement
```
Default policy (replication factor = 3):
- Replica 1: Same node as writer (or random node)
- Replica 2: Different rack (fault tolerance across racks)
- Replica 3: Same rack as Replica 2, different node (bandwidth optimization)
```

| Replica | Location | Reason |
|---------|----------|--------|
| **1st** | Local node/rack | Low latency write |
| **2nd** | Remote rack | Rack-level fault tolerance |
| **3rd** | Same rack as 2nd | Balance between safety and bandwidth |

### Configuration
```xml
<!-- hdfs-site.xml -->
<property>
    <name>dfs.replication</name>
    <value>3</value>  <!-- Default replication factor -->
</property>
<property>
    <name>dfs.replication.max</name>
    <value>512</value>  <!-- Maximum allowed -->
</property>
```

```bash
# Change replication for specific file
hdfs dfs -setrep -w 5 /path/to/important_file

# Check replication status
hdfs fsck /path/to/file -files -blocks -locations
```

### Interview Tip
The key design choice is the **pipeline replication** model — the client only sends data once, and DataNodes forward to each other. This minimizes client bandwidth usage. Also, the rack-aware placement policy balances **fault tolerance** (cross-rack) with **write performance** (intra-rack for later replicas).

---

## Question 17
- [ ] Done

**What is speculative execution in Hadoop , and why is it used ?**

**Answer:**

### Definition
Speculative execution is a Hadoop optimization where the framework launches **duplicate copies of slow-running tasks** on other nodes, using the output of whichever finishes first.

### How It Works

| Step | Action |
|------|---------|
| 1. **Monitor** | YARN tracks task progress across all mappers/reducers |
| 2. **Detect straggler** | Task running significantly slower than average |
| 3. **Launch backup** | Start duplicate task on a different node |
| 4. **First wins** | Use output from whichever copy finishes first |
| 5. **Kill duplicate** | Terminate the slower copy |

```
Node A (slow):  [=====...........]  ← Straggler detected
Node B (backup): [============]     ← Backup launched, finishes first ✅
Node A:          [killed]           ← Original killed
```

### Why It's Needed
- **Hardware heterogeneity**: Some nodes are slower (disk issues, old hardware)
- **Resource contention**: Competing workloads slow down tasks
- **Data skew**: Some tasks process more data
- **Network issues**: Slow rack switch or congestion

### Configuration
```xml
<!-- mapred-site.xml -->
<property>
    <name>mapreduce.map.speculative</name>
    <value>true</value>  <!-- Default: true -->
</property>
<property>
    <name>mapreduce.reduce.speculative</name>
    <value>true</value>  <!-- Default: true -->
</property>
```

### When to Disable
- **Non-idempotent tasks**: Writing to external databases (duplicates!)
- **Resource-constrained clusters**: Backup tasks waste resources
- **Tasks with side effects**: Email sending, API calls

### Interview Tip
Speculative execution trades **extra resources** for **lower latency**. It works because Hadoop clusters often have idle capacity. Disable it for tasks with side effects or when cluster utilization is already high (>80%).

---

## Question 18
- [ ] Done

**What is the significance of the input split in MapReduce jobs ?**

**Answer:**

### Definition
An **input split** is a logical division of input data that defines the chunk of data processed by a single mapper. It determines parallelism and data locality.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Block** | Physical storage unit in HDFS (default 128 MB) |
| **Input Split** | Logical division for MapReduce (usually = 1 block) |
| **Mapper** | One mapper per input split |
| **Data locality** | Split assigned to node storing the data |

### Split vs Block
```
File (512 MB) stored in HDFS:
  Block 1 (128 MB) → Split 1 → Mapper 1
  Block 2 (128 MB) → Split 2 → Mapper 2
  Block 3 (128 MB) → Split 3 → Mapper 3
  Block 4 (128 MB) → Split 4 → Mapper 4
```

### How Splits Are Created
```java
// InputFormat.getSplits() determines split strategy
// Default: FileInputFormat creates one split per block

// Custom split size
// mapreduce.input.fileinputformat.split.minsize = 256MB (fewer mappers)
// mapreduce.input.fileinputformat.split.maxsize = 64MB  (more mappers)

// Split size formula:
// splitSize = max(minSize, min(maxSize, blockSize))
```

### Configuration
```xml
<property>
    <name>mapreduce.input.fileinputformat.split.minsize</name>
    <value>0</value>  <!-- Default: 0 (use block size) -->
</property>
<property>
    <name>mapreduce.input.fileinputformat.split.maxsize</name>
    <value>268435456</value>  <!-- 256 MB -->
</property>
```

### Impact on Performance
- **Too many splits** → Too many mappers → Overhead from task startup
- **Too few splits** → Low parallelism → Underutilized cluster
- **Optimal**: Split size ≈ HDFS block size (default behavior)

### Interview Tip
The key insight is that splits are **logical** (defined by InputFormat) while blocks are **physical** (stored in HDFS). By default, 1 split = 1 block, which maximizes **data locality** — the mapper runs on the node storing the data, avoiding network transfer.

---

## Question 19
- [ ] Done

**How does partitioning work in Hadoop , and when is it used ?**

**Answer:**

### Definition
Partitioning determines **which reducer receives which key** during the shuffle phase. The default `HashPartitioner` distributes keys evenly across reducers.

### Partitioning Process
```
Mapper outputs (key, value) pairs
    ↓
Partitioner: partition = hash(key) % numReducers
    ↓
Each reducer gets all values for its assigned keys
```

### Default Partitioner
```java
public class HashPartitioner<K, V> extends Partitioner<K, V> {
    public int getPartition(K key, V value, int numReduceTasks) {
        return (key.hashCode() & Integer.MAX_VALUE) % numReduceTasks;
    }
}
```

### Custom Partitioner
```java
// Partition by country for geographic analysis
public class CountryPartitioner extends Partitioner<Text, IntWritable> {
    @Override
    public int getPartition(Text key, IntWritable value, int numReduceTasks) {
        String country = key.toString().split("_")[0];
        if (country.equals("US")) return 0;
        if (country.equals("EU")) return 1;
        return 2;  // Rest of world
    }
}

// Set in driver
job.setPartitionerClass(CountryPartitioner.class);
job.setNumReduceTasks(3);  // Must match partitioner logic
```

### When to Use Custom Partitioning

| Scenario | Reason |
|----------|--------|
| **Data skew** | Default hash creates uneven distribution |
| **Secondary sort** | Composite keys need custom partitioning |
| **Data locality** | Group related keys to same reducer |
| **Output organization** | Separate output files by category |
| **Total order sort** | Range-based partitioning |

### Interview Tip
Data skew is the biggest partitioning problem — one reducer gets most of the data while others sit idle. The solution is a **custom partitioner** that distributes hot keys across multiple reducers, or using a **combiner** to pre-aggregate before shuffling.

---

## Question 20
- [ ] Done

**Explain how reducers work in MapReduce and their interaction with shufflers**

**Answer:**

### Shuffle and Reduce Process

| Phase | What Happens |
|-------|--------------|
| 1. **Map output** | Mappers produce (key, value) pairs |
| 2. **Partition** | Partitioner assigns keys to reducers |
| 3. **Sort** | Map outputs sorted by key (on mapper side) |
| 4. **Shuffle** | Transfer sorted data from mappers to reducers over network |
| 5. **Merge sort** | Reducer merges sorted data from all mappers |
| 6. **Reduce** | Reducer processes all values for each key |

### Shuffle and Sort in Detail
```
Mapper 1: (A,1) (B,2) (A,3)     Mapper 2: (B,4) (A,5) (C,6)
    ↓ Sort                          ↓ Sort
    (A,1)(A,3)(B,2)                (A,5)(B,4)(C,6)
    ↓ Shuffle (network transfer)    ↓

Reducer 0 (keys A):     (A, [1, 3, 5])  ─→ reduce() → (A, 9)
Reducer 1 (keys B,C):   (B, [2, 4])     ─→ reduce() → (B, 6)
                         (C, [6])       ─→ reduce() → (C, 6)
```

### Reducer Mechanics
```java
public class SumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
    @Override
    public void reduce(Text key, Iterable<IntWritable> values, Context context) {
        int sum = 0;
        for (IntWritable val : values) {
            sum += val.get();  // Iterate through all values for this key
        }
        context.write(key, new IntWritable(sum));
    }
}
```

### Shuffle Optimization

| Optimization | Description |
|-------------|-------------|
| **Combiner** | Mini-reducer on mapper side (reduces shuffle data) |
| **Compression** | Compress map output before shuffle |
| **Sort buffer** | `mapreduce.task.io.sort.mb` (default 100 MB) |
| **Spill threshold** | `mapreduce.map.sort.spill.percent` (default 0.80) |

### Interview Tip
The shuffle phase is typically the **most expensive** part of MapReduce — it involves disk I/O (spilling), network transfer, and merge sorting. Using a **combiner** can reduce shuffle data by 10-100x. The combiner must be commutative and associative (e.g., sum, max, but not average).

---

## Question 21
- [ ] Done

**What are SequenceFiles in Hadoop ?**

**Answer:**

### Definition
SequenceFiles are Hadoop's **binary file format** that stores data as serialized key-value pairs. They are designed for efficient storage and processing within the Hadoop ecosystem.

### Compression Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Uncompressed** | No compression | Debugging, small files |
| **Record compressed** | Each value compressed independently | General purpose |
| **Block compressed** | Group of records compressed together | Best compression ratio |

### Structure
```
SequenceFile Format:
┌───────────────────────────┐
│ Header (version, key/val classes) │
├───────────────────────────┤
│ Record 1: (key1, value1)          │
│ Record 2: (key2, value2)          │
│ Sync Marker (every ~2000 records) │  ← Enables splitting
│ Record N: (keyN, valueN)          │
└───────────────────────────┘
```

### Code Example
```java
import org.apache.hadoop.io.*;
import org.apache.hadoop.conf.Configuration;

// Write SequenceFile
Configuration conf = new Configuration();
SequenceFile.Writer writer = SequenceFile.createWriter(conf,
    SequenceFile.Writer.file(new Path("/output/data.seq")),
    SequenceFile.Writer.keyClass(Text.class),
    SequenceFile.Writer.valueClass(IntWritable.class),
    SequenceFile.Writer.compression(CompressionType.BLOCK));

writer.append(new Text("key1"), new IntWritable(100));
writer.append(new Text("key2"), new IntWritable(200));
writer.close();

// Read SequenceFile
SequenceFile.Reader reader = new SequenceFile.Reader(conf,
    SequenceFile.Reader.file(new Path("/output/data.seq")));
Text key = new Text();
IntWritable val = new IntWritable();
while (reader.next(key, val)) {
    System.out.println(key + ": " + val);
}
```

### Advantages
- **Splittable** (sync markers enable MapReduce parallelism)
- **Binary format** (efficient serialization)
- **Compressible** (built-in compression support)
- **Small file solution** (merge many small files into one SequenceFile)

### Interview Tip
SequenceFiles solve the **small files problem** in HDFS — instead of storing millions of small files (each consuming a NameNode metadata entry), you merge them into SequenceFiles. This dramatically reduces NameNode memory pressure.

---

## Question 22
- [ ] Done

**Describe the ways to optimize a MapReduce job**

**Answer:**

### Optimization Categories

| Category | Techniques |
|----------|------------|
| **Input** | Proper input format, split size tuning |
| **Map phase** | Combiner, in-mapper combining, compression |
| **Shuffle** | Compression, buffer tuning, partitioning |
| **Reduce** | Fewer reducers, secondary sort |
| **Output** | Compression, proper format |
| **Cluster** | JVM reuse, speculative execution |

### Key Optimizations
```xml
<!-- 1. Use Combiner (reduces shuffle data by 10-100x) -->
job.setCombinerClass(SumReducer.class);

<!-- 2. Compress map output (reduce shuffle traffic) -->
<property>
    <name>mapreduce.map.output.compress</name>
    <value>true</value>
</property>
<property>
    <name>mapreduce.map.output.compress.codec</name>
    <value>org.apache.hadoop.io.compress.SnappyCodec</value>
</property>

<!-- 3. Tune sort buffer (reduce disk spills) -->
<property>
    <name>mapreduce.task.io.sort.mb</name>
    <value>256</value>  <!-- Default: 100 MB -->
</property>
<property>
    <name>mapreduce.map.sort.spill.percent</name>
    <value>0.90</value>  <!-- Default: 0.80 -->
</property>

<!-- 4. JVM reuse (avoid JVM startup overhead) -->
<property>
    <name>mapreduce.job.jvm.numtasks</name>
    <value>-1</value>  <!-- Reuse JVM for all tasks -->
</property>

<!-- 5. Optimal number of reducers -->
job.setNumReduceTasks(cluster_nodes * reducers_per_node * 0.95);
```

### Advanced Optimizations
1. **Use efficient file formats**: ORC, Parquet (columnar, compressed)
2. **Avoid small files**: Merge with CombineFileInputFormat
3. **Data locality**: Ensure splits align with HDFS blocks
4. **Proper data types**: Use `Writable` types, not Java serialization
5. **Pre-sort data**: If doing joins, pre-sort by join key

### Interview Tip
The three highest-impact optimizations are: 1) **Combiner** (reduces network I/O), 2) **Map output compression** (Snappy for speed, Gzip for ratio), 3) **Proper number of reducers** (rule of thumb: 0.95 * total reduce slots). Always profile before optimizing — use Hadoop counters to find bottlenecks.

---

## Question 23
- [ ] Done

**What is the significance of combiner in the Hadoop MapReduce framework ?**

**Answer:**

### Definition
A **combiner** (also called a mini-reducer) is an optional optimization that runs on the **mapper node** after the map phase, performing local aggregation before data is shuffled to reducers.

### How It Works
```
Without Combiner:
  Mapper 1: (cat,1)(dog,1)(cat,1)(cat,1)  ──shuffle──▶ Reducer: (cat,[1,1,1,1,1]) → (cat,5)
  Mapper 2: (cat,1)(dog,1)                ──shuffle──▶  6 records transferred

With Combiner:
  Mapper 1: (cat,1)(dog,1)(cat,1)(cat,1) → Combiner → (cat,3)(dog,1) ─▶ Reducer
  Mapper 2: (cat,1)(dog,1)               → Combiner → (cat,1)(dog,1) ─▶  4 records transferred
```

### Code Example
```java
// The combiner class is often the SAME as the reducer
job.setMapperClass(WordCountMapper.class);
job.setCombinerClass(WordCountReducer.class);  // <-- Same as reducer
job.setReducerClass(WordCountReducer.class);
```

### Rules for Combiners

| Rule | Explanation |
|------|-------------|
| **Commutative** | Order doesn't matter: a + b = b + a |
| **Associative** | Grouping doesn't matter: (a+b)+c = a+(b+c) |
| **Same input/output types** | Combiner output = Reducer input |
| **No guarantee** | Hadoop may run it 0, 1, or many times |

### When NOT to Use
- **Average**: `avg(1,3) ≠ avg(avg(1), avg(3))` — not associative
- **Median**: Not decomposable
- **Standard deviation**: Requires all data points

### Interview Tip
The combiner can reduce shuffle data by **10-100x** for aggregation operations. Always mention that it's **not guaranteed to run** — the program must produce correct results without it. Think of it as a performance hint, not a correctness requirement.

---

## Question 24
- [ ] Done

**Explain what you can do to optimize the performance of HDFS**

**Answer:**

### HDFS Performance Optimization Strategies

| Category | Strategy | Impact |
|----------|----------|--------|
| **Block size** | Increase to 256/512 MB for large files | Reduces NameNode memory, fewer seeks |
| **Replication** | Tune factor based on access patterns | Balance reliability vs storage |
| **Compression** | Snappy (speed) or Gzip (ratio) | 2-5x storage reduction |
| **Short-circuit reads** | Read local blocks without DataNode | 10-30% read improvement |
| **Caching** | Centralized cache for hot data | Eliminates disk I/O |
| **SSD tiering** | Store hot data on SSD, cold on HDD | 5-10x faster reads |

### Key Configurations
```xml
<!-- Increase block size for large files -->
<property>
    <name>dfs.blocksize</name>
    <value>268435456</value>  <!-- 256 MB -->
</property>

<!-- Enable short-circuit local reads -->
<property>
    <name>dfs.client.read.shortcircuit</name>
    <value>true</value>
</property>

<!-- Increase handler threads -->
<property>
    <name>dfs.namenode.handler.count</name>
    <value>64</value>  <!-- Default: 10 -->
</property>
```

### Application-Level Optimizations
- **Avoid small files**: Merge into SequenceFiles or HAR archives
- **Use columnar formats**: ORC/Parquet for analytical workloads
- **Batch writes**: Buffer data and write in large chunks
- **Rack-aware placement**: Configure topology for optimal replication

### Interview Tip
The single biggest HDFS optimization is avoiding the **small files problem** — each file/block consumes ~150 bytes of NameNode memory. A million small files = 300 MB of heap. Solutions: merge files, use CombineFileInputFormat, or switch to HBase for small random reads.

---

## Question 25
- [ ] Done

**What are the best practices for managing memory and CPU resources in a Hadoop cluster ?**

**Answer:**

### Resource Management with YARN

YARN manages **memory** and **CPU (vcores)** as the two fundamental cluster resources.

### Key Configurations
```xml
<!-- yarn-site.xml: Per-node resource limits -->
<property>
    <name>yarn.nodemanager.resource.memory-mb</name>
    <value>65536</value>  <!-- 64 GB available for YARN -->
</property>
<property>
    <name>yarn.nodemanager.resource.cpu-vcores</name>
    <value>16</value>
</property>

<!-- Container size bounds -->
<property>
    <name>yarn.scheduler.minimum-allocation-mb</name>
    <value>1024</value>  <!-- Min 1 GB per container -->
</property>
<property>
    <name>yarn.scheduler.maximum-allocation-mb</name>
    <value>16384</value>  <!-- Max 16 GB per container -->
</property>

<!-- MapReduce task resources -->
<property>
    <name>mapreduce.map.memory.mb</name>
    <value>4096</value>
</property>
<property>
    <name>mapreduce.reduce.memory.mb</name>
    <value>8192</value>
</property>
```

### Scheduling Strategies

| Scheduler | Description | Best For |
|-----------|-------------|----------|
| **FIFO** | First come, first served | Single-user clusters |
| **Capacity** | Guaranteed queue capacities | Multi-tenant organizations |
| **Fair** | Equal resource sharing | Mixed workload clusters |

### Memory Sizing Best Practices
```
Reserved for OS:         ~20% of total RAM
YARN NodeManager memory: Total RAM - OS reserved - other services
Mapper memory:           1-4 GB (depending on data)
Reducer memory:          1.5x - 2x of mapper memory
JVM heap (-Xmx):         0.8 × container memory
```

### Interview Tip
A common mistake is setting container memory equal to JVM heap. The container memory must be **larger** than JVM heap to account for off-heap usage. Rule: set `-Xmx` to **80%** of `mapreduce.map.memory.mb`. Also, leave 15-20% of node resources for OS and DataNode/NodeManager daemons.

---

## Question 26
- [ ] Done

**What is the concept of erasure coding in HDFS , and how does it differ from replication ?**

**Answer:**

### Definition
Erasure Coding (EC) is a data protection method that provides **fault tolerance with less storage overhead** than replication by encoding data into fragments and computing parity blocks.

### Replication vs Erasure Coding

| Feature | Replication (3x) | Erasure Coding (RS-6-3) |
|---------|------------------|-------------------------|
| **Storage overhead** | 200% (3 copies) | 50% (6 data + 3 parity) |
| **Fault tolerance** | Survives 2 failures | Survives 3 failures |
| **100 TB raw data** | 300 TB total | 150 TB total |
| **Read performance** | Fast (data-local) | Slower (reconstruction) |
| **Write performance** | Fast (pipeline) | Slower (encoding) |
| **Best for** | Hot data, low latency | Cold/warm data, archival |

### How It Works
```
Reed-Solomon (6,3) example:

Original data: [D1][D2][D3][D4][D5][D6]  (6 data blocks)
Parity blocks: [P1][P2][P3]               (3 parity blocks)
Total: 9 blocks for 6 data = 50% overhead

If D2 and D5 fail → reconstruct from remaining blocks
Any 6 of 9 blocks can reconstruct the original data
```

### Configuration (Hadoop 3.x+)
```bash
# List available EC policies
hdfs ec -listPolicies

# Enable and set policy
hdfs ec -enablePolicy -policy RS-6-3-1024k
hdfs ec -setPolicy -path /data/cold -policy RS-6-3-1024k
```

### Available Policies

| Policy | Data:Parity | Overhead | Min Nodes |
|--------|-------------|----------|-----------|
| RS-3-2-1024k | 3:2 | 67% | 5 |
| RS-6-3-1024k | 6:3 | 50% | 9 |
| RS-10-4-1024k | 10:4 | 40% | 14 |

### Interview Tip
Erasure coding (Hadoop 3.0+) saves **~50% storage** vs 3x replication while providing equal or better fault tolerance. The tradeoff is higher CPU for encoding/decoding and slower reconstruction. Use it for **cold data** (infrequent access) and keep replication for **hot data** (frequent access).

---

## Question 27
- [ ] Done

**Explain how Hadoop uses data locality to improve performance**

**Answer:**

### Definition
Data locality means **moving computation to where the data resides** rather than transferring data to the compute node. It's a core Hadoop optimization principle.

### Locality Levels

| Level | Description | Network Cost | Speed |
|-------|-------------|-------------|-------|
| **Data-local** | Task runs on node storing the block | None | Fastest |
| **Rack-local** | Same rack, different node | Intra-rack | Medium |
| **Off-rack** | Different rack entirely | Cross-rack | Slowest |

### How It Works
```
1. File stored in HDFS → blocks distributed across nodes
   B1 → Node1, B2 → Node2

2. MapReduce job submitted
   - Mapper for B1 → assigned to Node1 (data-local ✅)
   - If Node1 busy → Node in same rack (rack-local ⚠️)
   - Last resort → any node (off-rack ❌)
```

### Configuration
```xml
<!-- Rack awareness -->
<property>
    <name>net.topology.script.file.name</name>
    <value>/etc/hadoop/topology.sh</value>
</property>

<!-- Delay scheduling (wait for data-local slot) -->
<property>
    <name>yarn.scheduler.capacity.node-locality-delay</name>
    <value>40</value>
</property>
```

### Performance Impact
```
Data-local:  ~100-200 MB/s (disk speed)
Rack-local:  Limited by rack switch (~1-10 Gbps shared)
Off-rack:    Limited by core switch (~10-40 Gbps shared)

10 TB job: Data-local ~15 min vs Off-rack ~45 min
```

### Interview Tip
Data locality is why Hadoop scales linearly — it avoids the **network bottleneck** by processing data in-place. YARN uses **delay scheduling** to wait for a data-local slot before falling back. Proper **rack awareness configuration** is essential for intelligent task placement.

---

## Question 28
- [ ] Done

**How does Hadoop support different file formats , and what are some of them?**

**Answer:**

### Hadoop File Formats Comparison

| Format | Type | Splittable | Schema | Compression | Best For |
|--------|------|-----------|--------|-------------|----------|
| **Text/CSV** | Row | Yes (uncompressed) | No | External | Simple data, interop |
| **SequenceFile** | Row (binary) | Yes | No | Built-in | MapReduce intermediate |
| **Avro** | Row (binary) | Yes | Embedded | Built-in | Schema evolution, streaming |
| **Parquet** | Columnar | Yes | Embedded | Built-in | Analytics, column queries |
| **ORC** | Columnar | Yes | Embedded | Built-in | Hive, heavy analytics |

### Row vs Columnar Storage
```
Row-based (Avro, SequenceFile):
  [id=1, name="Alice", age=30] [id=2, name="Bob", age=25]
  ✅ Fast full-row reads   ❌ Slow column queries

Columnar (Parquet, ORC):
  Column 'id':   [1, 2, 3, ...]
  Column 'name': ["Alice", "Bob", ...]
  ✅ Fast column queries   ✅ Better compression
```

### When to Use Which

| Scenario | Recommended Format |
|----------|-------------------|
| ETL intermediate | SequenceFile or Avro |
| Data warehouse | Parquet or ORC |
| Hive workloads | ORC |
| Spark workloads | Parquet |
| Schema evolution | Avro |
| Streaming ingestion | Avro |

### Code Example
```python
# Spark: Reading different formats
df_parquet = spark.read.parquet("/data/file.parquet")
df_orc = spark.read.orc("/data/file.orc")
df_avro = spark.read.format("avro").load("/data/file.avro")

# Save as Parquet
df.write.parquet("/output/data.parquet", compression="snappy")
```

### Interview Tip
For analytics: use **Parquet** (Spark) or **ORC** (Hive) — columnar formats achieve **10-100x** speedup through predicate pushdown and column pruning. For streaming/ETL: use **Avro** for schema evolution support (add/remove fields without breaking consumers).

---

## Question 29
- [ ] Done

**What is Hadoop federation , and how can it scale a Hadoop cluster ?**

**Answer:**

### Definition
Hadoop Federation allows **multiple independent NameNodes** to share a pool of DataNodes, each managing its own namespace (directory tree). This overcomes the single-NameNode scalability bottleneck.

### Architecture
```
Traditional (Single NameNode):
  NameNode (single point of bottleneck)
    └─ manages ALL files, blocks, metadata
    └─ limited by single JVM heap

Federated (Multiple NameNodes):
  NameNode 1 (/user)    ──┐
  NameNode 2 (/data)    ──┤──▶ Shared DataNode Pool
  NameNode 3 (/logs)    ──┘
  Each manages its own namespace independently
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Namespace** | Directory tree managed by one NameNode |
| **Block Pool** | Set of blocks belonging to a namespace |
| **Cluster ID** | Identifies the federated cluster |
| **ViewFS** | Client-side mount table mapping paths to NameNodes |

### Configuration
```xml
<!-- hdfs-site.xml -->
<property>
    <name>dfs.nameservices</name>
    <value>ns1,ns2,ns3</value>
</property>
<property>
    <name>dfs.namenode.rpc-address.ns1</name>
    <value>nn1:8020</value>
</property>
<property>
    <name>dfs.namenode.rpc-address.ns2</name>
    <value>nn2:8020</value>
</property>
```

```xml
<!-- ViewFS client mount table (core-site.xml) -->
<property>
    <name>fs.viewfs.mounttable.default.link./user</name>
    <value>hdfs://ns1/user</value>
</property>
<property>
    <name>fs.viewfs.mounttable.default.link./data</name>
    <value>hdfs://ns2/data</value>
</property>
```

### Benefits
- **Horizontal scalability**: Each NameNode manages a subset of metadata
- **Isolation**: Workloads in different namespaces don't affect each other
- **Performance**: Parallel metadata operations across NameNodes
- **No single bottleneck**: Distributes NameNode memory pressure

### Federation vs HA

| Feature | Federation | HA |
|---------|-----------|----|
| **Purpose** | Scalability | Fault tolerance |
| **NameNodes** | Multiple active | 1 active + 1 standby |
| **Namespace** | Separate per NN | Same shared namespace |
| **Can combine** | Yes — Federation + HA = each NN has a standby |

### Interview Tip
Federation solves the **NameNode memory bottleneck** — a single NameNode stores ~150 bytes per block in memory. With billions of files, one NameNode runs out of heap. Federation distributes metadata across NameNodes while DataNodes remain shared. It's complementary to HA (you can have federated NameNodes, each with an HA standby).

---

## Question 30
- [ ] Done

**What are the implications of small files on HDFS performance and how can this be mitigated ?**

**Answer:**

### The Small Files Problem

HDFS is designed for large files. Storing many small files creates significant performance problems.

### Why Small Files Are Problematic

| Problem | Explanation |
|---------|-------------|
| **NameNode memory** | Each file/block uses ~150 bytes of heap |
| **1 million files** | ~300 MB NameNode memory consumed |
| **1 billion files** | ~300 GB — exceeds practical NameNode heap |
| **Mapper overhead** | 1 mapper per file → millions of short-lived mappers |
| **Seek time** | Time to find file > time to read it |

### Impact Example
```
1 million × 1 KB files = 1 GB data, but:
  - NameNode memory: ~300 MB (for metadata)
  - MapReduce: 1 million mappers (massive overhead)
  - Seek time: dominates read time

1 file × 1 GB = same data, but:
  - NameNode memory: ~1.2 KB (8 blocks × 150 bytes)
  - MapReduce: 8 mappers (efficient)
  - Sequential read: optimal throughput
```

### Mitigation Strategies

| Solution | How It Works | Best For |
|----------|-------------|----------|
| **HAR (Hadoop Archive)** | Pack small files into archive; NameNode sees 1 file | Archival, cold data |
| **SequenceFile** | Merge files as key-value pairs in binary format | MapReduce processing |
| **CombineFileInputFormat** | Multiple files per mapper input split | MapReduce jobs |
| **HBase** | Store small records in columnar database | Random access patterns |
| **Compaction** | Periodic merge of small files | Streaming/ingestion |

### Code Examples
```bash
# Create Hadoop Archive
hadoop archive -archiveName data.har -p /input/small_files /output/

# Access archived files
hdfs dfs -ls har:///output/data.har/
```

```java
// Use CombineFileInputFormat in MapReduce
job.setInputFormatClass(CombineTextInputFormat.class);
// Set max split size (combine files up to 256 MB per mapper)
CombineTextInputFormat.setMaxInputSplitSize(job, 268435456);
```

### Prevention Strategies
- **Buffer and batch**: Collect data before writing to HDFS
- **Use append**: Append to existing files instead of creating new ones
- **Streaming frameworks**: Kafka + Spark Streaming to batch micro-files
- **Compaction jobs**: Periodic jobs to merge small files

### Interview Tip
The small files problem is one of the **most common HDFS issues** in production. The root cause is that NameNode metadata is stored **in memory**. The best solution depends on the use case: **SequenceFile** for processing, **HAR** for archival, **HBase** for random access. Always mention that prevention (batching writes) is better than mitigation.

---

## Question 31
- [ ] Done

**Explain the concept of a Hadoop Distributed File System (HDFS) and its architecture**

*Answer to be added.*

---

## Question 32
- [ ] Done

**How does MapReduce programming model work in Hadoop ?**

*Answer to be added.*

---

## Question 33
- [ ] Done

**What is YARN , and how does it improve Hadoop’s resource management ?**

*Answer to be added.*

---

## Question 34
- [ ] Done

**Explain the role of the Namenode and Datanode in HDFS**

*Answer to be added.*

---

## Question 35
- [ ] Done

**What is a Rack Awareness algorithm in HDFS , and why is it important ?**

*Answer to be added.*

---

## Question 36
- [ ] Done

**What are some of the characteristics that differentiate Hadoop from traditional RDBMS ?**

*Answer to be added.*

---

## Question 37
- [ ] Done

**How can you secure a Hadoop cluster ? Name some of the security mechanisms available**

*Answer to be added.*

---

## Question 38
- [ ] Done

**Describe the role of HBase in Hadoop ecosystem**

*Answer to be added.*

---

## Question 39
- [ ] Done

**What is Apache Hive and what types of problems does it solve ?**

*Answer to be added.*

---

## Question 40
- [ ] Done

**How does Apache Pig fit into the Hadoop ecosystem ?**

*Answer to be added.*

---

## Question 41
- [ ] Done

**Discuss the role of Apache Spark in the Hadoop ecosystem**

*Answer to be added.*

---

## Question 42
- [ ] Done

**How are large datasets processed in Hadoop ?**

*Answer to be added.*

---

## Question 43
- [ ] Done

**Discuss the concept and benefits of a journal node in HDFS HA configuration**

*Answer to be added.*

---
