# Hadoop Interview Questions - General Questions

## Question 1
- [ ] Done

**How do you monitor a Hadoop cluster?**

### Answer

### Monitoring Tools

| Tool | Purpose |
|------|---------|
| **Ambari** | Complete cluster management UI |
| **Cloudera Manager** | Enterprise management |
| **Ganglia** | Metrics collection |
| **Nagios** | Alerting |
| **YARN Web UI** | Application monitoring |

### Key Metrics to Monitor

| Metric | Description |
|--------|-------------|
| NameNode heap usage | Memory health |
| DataNode disk usage | Storage capacity |
| Block replication | Data safety |
| Running applications | Workload |
| Failed tasks | Job health |

### Python Code Example
```python
import requests

class HadoopMonitor:
    def __init__(self, namenode_url, yarn_url):
        self.namenode = namenode_url
        self.yarn = yarn_url
    
    def get_hdfs_metrics(self):
        """Get HDFS cluster metrics"""
        response = requests.get(
            f"{self.namenode}/jmx?qry=Hadoop:service=NameNode,name=FSNamesystem"
        )
        data = response.json()['beans'][0]
        return {
            'total_capacity': data['CapacityTotal'],
            'used_capacity': data['CapacityUsed'],
            'remaining': data['CapacityRemaining'],
            'total_blocks': data['BlocksTotal'],
            'missing_blocks': data['MissingBlocks'],
            'live_datanodes': data['NumLiveDataNodes']
        }
    
    def get_yarn_metrics(self):
        """Get YARN cluster metrics"""
        response = requests.get(f"{self.yarn}/ws/v1/cluster/metrics")
        data = response.json()['clusterMetrics']
        return {
            'active_nodes': data['activeNodes'],
            'apps_running': data['appsRunning'],
            'apps_pending': data['appsPending'],
            'memory_total': data['totalMB'],
            'memory_available': data['availableMB']
        }
    
    def check_health(self):
        """Overall health check"""
        hdfs = self.get_hdfs_metrics()
        yarn = self.get_yarn_metrics()
        
        issues = []
        if hdfs['missing_blocks'] > 0:
            issues.append(f"Missing blocks: {hdfs['missing_blocks']}")
        if yarn['active_nodes'] == 0:
            issues.append("No active NodeManagers")
        
        return {'healthy': len(issues) == 0, 'issues': issues}

# Usage
monitor = HadoopMonitor('http://namenode:50070', 'http://yarn:8088')
print(monitor.check_health())
```

---

## Question 2
- [ ] Done

**What is rack awareness in Hadoop?**

### Answer

**Definition**: Rack awareness is HDFS's ability to understand network topology and place data replicas across different racks for fault tolerance and performance.

### Benefits

| Benefit | Description |
|---------|-------------|
| **Fault tolerance** | Survives rack failures |
| **Network efficiency** | Reduces cross-rack traffic |
| **Data locality** | Improves read performance |

### Replica Placement Strategy
1. First replica: Local node
2. Second replica: Different rack
3. Third replica: Same rack as second, different node

### Python Code Example
```python
# Rack awareness configuration
# In core-site.xml, configure topology script

# topology.py - Sample rack topology script
#!/usr/bin/env python3
import sys

# Map hostnames to racks
RACK_MAP = {
    'node1.cluster.local': '/rack1',
    'node2.cluster.local': '/rack1',
    'node3.cluster.local': '/rack2',
    'node4.cluster.local': '/rack2',
    '192.168.1.1': '/rack1',
    '192.168.1.2': '/rack2',
}

def get_rack(host):
    return RACK_MAP.get(host, '/default-rack')

if __name__ == '__main__':
    for host in sys.argv[1:]:
        print(get_rack(host))

# Check rack info via API
def get_rack_info(namenode_url):
    import requests
    response = requests.get(
        f"{namenode_url}/jmx?qry=Hadoop:service=NameNode,name=NameNodeInfo"
    )
    data = response.json()['beans'][0]
    return data.get('LiveNodes')  # Contains rack info per node
```

---

## Question 3
- [ ] Done

**How do you handle small files problem in Hadoop?**

### Answer

### Problem
- Each file = 1 block = 1 metadata entry
- Millions of small files = NameNode memory pressure
- Inefficient MapReduce (1 mapper per file)

### Solutions

| Solution | Description |
|----------|-------------|
| **HAR files** | Hadoop Archive - combine files |
| **Sequence Files** | Binary container format |
| **CombineFileInputFormat** | Process multiple files per mapper |
| **HBase** | Store small files as cells |

### Python Code Example
```python
from mrjob.job import MRJob
import subprocess

# Solution 1: Create HAR archive
def create_har_archive(src_dir, dest_dir, archive_name):
    """Create Hadoop Archive"""
    cmd = [
        'hadoop', 'archive',
        '-archiveName', f'{archive_name}.har',
        '-p', src_dir,
        dest_dir
    ]
    subprocess.run(cmd, check=True)

# Solution 2: Use CombineFileInputFormat
class ProcessSmallFiles(MRJob):
    """Process multiple small files efficiently"""
    
    HADOOP_INPUT_FORMAT = 'org.apache.hadoop.mapreduce.lib.input.CombineTextInputFormat'
    
    JOBCONF = {
        'mapreduce.input.fileinputformat.split.maxsize': '134217728',  # 128MB
        'mapreduce.input.fileinputformat.split.minsize': '1048576',    # 1MB
    }
    
    def mapper(self, _, line):
        yield 'lines', 1
    
    def reducer(self, key, counts):
        yield key, sum(counts)

# Solution 3: Merge small files
def merge_small_files(input_dir, output_file, hdfs_client):
    """Merge multiple small files into one"""
    files = hdfs_client.list(input_dir)
    
    with hdfs_client.write(output_file) as writer:
        for f in files:
            with hdfs_client.read(f"{input_dir}/{f}") as reader:
                writer.write(reader.read())
```

---

## Question 4
- [ ] Done

**What is data locality in Hadoop?**

### Answer

**Definition**: Data locality means processing data on the node where it's stored, minimizing network transfer.

### Locality Levels

| Level | Description | Performance |
|-------|-------------|-------------|
| **Data-local** | Task on same node as data | Best |
| **Rack-local** | Task in same rack | Good |
| **Off-rack** | Task in different rack | Slowest |

### Python Code Example
```python
import requests

def check_data_locality(yarn_url, app_id):
    """Check data locality for completed job"""
    response = requests.get(
        f"{yarn_url}/ws/v1/history/mapreduce/jobs/{app_id}/counters"
    )
    counters = response.json()['jobCounters']['counterGroup']
    
    for group in counters:
        if group['counterGroupName'] == 'org.apache.hadoop.mapreduce.FileSystemCounter':
            continue
        if 'DATA_LOCAL' in str(group):
            return {
                'data_local': group.get('DATA_LOCAL_MAPS', 0),
                'rack_local': group.get('RACK_LOCAL_MAPS', 0),
                'other_local': group.get('OTHER_LOCAL_MAPS', 0)
            }

# Optimize for locality
"""
Best practices:
1. Increase block size for large files
2. Use short-circuit reads (reads bypass DataNode)
3. Configure locality wait time
4. Co-locate compute with storage nodes
"""
```

---

## Question 5
- [ ] Done

**How do you tune Hadoop performance?**

### Answer

### Key Tuning Areas

| Area | Parameters |
|------|------------|
| **Memory** | Mapper/Reducer heap, YARN container sizes |
| **I/O** | Buffer sizes, compression |
| **Parallelism** | Number of mappers/reducers |
| **Network** | Sort buffer, shuffle settings |

### Important Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `mapreduce.map.memory.mb` | Mapper memory | 1024 |
| `mapreduce.reduce.memory.mb` | Reducer memory | 1024 |
| `mapreduce.task.io.sort.mb` | Sort buffer | 100 |
| `mapreduce.map.sort.spill.percent` | Spill threshold | 0.8 |

### Python Code Example (mrjob tuning)
```python
from mrjob.job import MRJob

class TunedJob(MRJob):
    """Example with performance tuning"""
    
    JOBCONF = {
        # Memory settings
        'mapreduce.map.memory.mb': '2048',
        'mapreduce.reduce.memory.mb': '4096',
        'mapreduce.map.java.opts': '-Xmx1638m',
        'mapreduce.reduce.java.opts': '-Xmx3276m',
        
        # I/O settings
        'mapreduce.task.io.sort.mb': '256',
        'mapreduce.task.io.sort.factor': '100',
        
        # Compression
        'mapreduce.map.output.compress': 'true',
        'mapreduce.map.output.compress.codec': 'org.apache.hadoop.io.compress.SnappyCodec',
        
        # Parallelism
        'mapreduce.job.reduces': '10',
    }
    
    def mapper(self, _, line):
        for word in line.split():
            yield word, 1
    
    def combiner(self, word, counts):
        """Local aggregation to reduce shuffle"""
        yield word, sum(counts)
    
    def reducer(self, word, counts):
        yield word, sum(counts)

if __name__ == '__main__':
    TunedJob.run()
```

### Performance Checklist
- Use combiners for aggregation
- Enable compression for shuffle
- Right-size containers
- Monitor GC overhead
- Check data locality metrics

---

## Question 6
- [ ] Done

**How can you diagnose and troubleshoot Hadoop performance issues ?**

*Answer to be added.*

---

## Question 7
- [ ] Done

**How can job scheduling be optimized in Hadoop ?**

*Answer to be added.*

---