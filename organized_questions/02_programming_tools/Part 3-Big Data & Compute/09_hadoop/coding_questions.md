# Hadoop Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Write a MapReduce job to count word frequencies.**

### Answer

### Python Code Example
```python
from mrjob.job import MRJob
from mrjob.step import MRStep
import re

class WordCount(MRJob):
    """
    Pipeline:
    Input: Text file with lines of text
    Map: Split line into words, emit (word, 1)
    Reduce: Sum counts for each word
    Output: (word, total_count)
    """
    
    WORD_RE = re.compile(r"[\w']+")
    
    def mapper(self, _, line):
        """Emit (word, 1) for each word"""
        for word in self.WORD_RE.findall(line):
            yield word.lower(), 1
    
    def combiner(self, word, counts):
        """Local aggregation (optional but improves performance)"""
        yield word, sum(counts)
    
    def reducer(self, word, counts):
        """Sum all counts for each word"""
        yield word, sum(counts)

if __name__ == '__main__':
    WordCount.run()

# Run: python word_count.py input.txt > output.txt
# Or: python word_count.py -r hadoop hdfs:///input > output.txt
```

---

## Question 2
- [ ] Done

**Write a MapReduce job to find top N most frequent words.**

### Answer

### Python Code Example
```python
from mrjob.job import MRJob
from mrjob.step import MRStep
import heapq

class TopNWords(MRJob):
    """
    Pipeline:
    Step 1: Count words (Map + Reduce)
    Step 2: Find top N (Map sends all to one reducer)
    Output: Top N words with counts
    """
    
    def configure_args(self):
        super().configure_args()
        self.add_passthru_arg('--top-n', type=int, default=10)
    
    def steps(self):
        return [
            MRStep(mapper=self.mapper_count,
                   combiner=self.combiner_count,
                   reducer=self.reducer_count),
            MRStep(reducer=self.reducer_top_n)
        ]
    
    def mapper_count(self, _, line):
        """Step 1 Map: emit (word, 1)"""
        for word in line.strip().lower().split():
            yield word, 1
    
    def combiner_count(self, word, counts):
        """Step 1 Combine: local sum"""
        yield word, sum(counts)
    
    def reducer_count(self, word, counts):
        """Step 1 Reduce: emit (None, (count, word)) for sorting"""
        yield None, (sum(counts), word)
    
    def reducer_top_n(self, _, count_word_pairs):
        """Step 2 Reduce: find top N"""
        top_n = heapq.nlargest(self.options.top_n, count_word_pairs)
        for count, word in top_n:
            yield word, count

if __name__ == '__main__':
    TopNWords.run()

# Run: python top_n_words.py --top-n 20 input.txt
```

---

## Question 3
- [ ] Done

**Write a MapReduce job to calculate average.**

### Answer

### Python Code Example
```python
from mrjob.job import MRJob

class AverageCalculator(MRJob):
    """
    Pipeline:
    Input: CSV with category, value
    Map: Parse line, emit (category, value)
    Reduce: Calculate average per category
    Output: (category, average)
    """
    
    def mapper(self, _, line):
        """Parse CSV and emit (category, value)"""
        try:
            parts = line.strip().split(',')
            category = parts[0]
            value = float(parts[1])
            yield category, (value, 1)  # (sum, count)
        except (ValueError, IndexError):
            pass  # Skip malformed lines
    
    def combiner(self, category, values):
        """Partial aggregation"""
        total_sum = 0
        total_count = 0
        for val, count in values:
            total_sum += val
            total_count += count
        yield category, (total_sum, total_count)
    
    def reducer(self, category, values):
        """Calculate final average"""
        total_sum = 0
        total_count = 0
        for val, count in values:
            total_sum += val
            total_count += count
        
        average = total_sum / total_count if total_count > 0 else 0
        yield category, round(average, 2)

if __name__ == '__main__':
    AverageCalculator.run()

# Input example:
# electronics,150.00
# electronics,200.00
# clothing,50.00
# Output: electronics 175.0, clothing 50.0
```

---

## Question 4
- [ ] Done

**Write a MapReduce job to perform a join operation.**

### Answer

### Python Code Example
```python
from mrjob.job import MRJob
from mrjob.step import MRStep

class ReduceSideJoin(MRJob):
    """
    Pipeline:
    Input: Two files (orders.csv, customers.csv)
    Map: Tag records by source
    Reduce: Join on common key (customer_id)
    Output: Joined records
    
    orders.csv: order_id,customer_id,amount
    customers.csv: customer_id,name,city
    """
    
    def mapper(self, _, line):
        """Tag records by type"""
        parts = line.strip().split(',')
        
        # Detect file type by number of columns or prefix
        if len(parts) == 3 and parts[0].startswith('O'):
            # Order record: order_id, customer_id, amount
            customer_id = parts[1]
            yield customer_id, ('order', parts[0], parts[2])
        elif len(parts) == 3:
            # Customer record: customer_id, name, city
            customer_id = parts[0]
            yield customer_id, ('customer', parts[1], parts[2])
    
    def reducer(self, customer_id, records):
        """Join records"""
        orders = []
        customer_info = None
        
        for record in records:
            if record[0] == 'customer':
                customer_info = (record[1], record[2])  # name, city
            else:
                orders.append((record[1], record[2]))  # order_id, amount
        
        # Emit joined records
        if customer_info:
            for order_id, amount in orders:
                yield customer_id, {
                    'customer_name': customer_info[0],
                    'city': customer_info[1],
                    'order_id': order_id,
                    'amount': amount
                }

if __name__ == '__main__':
    ReduceSideJoin.run()
```

---

## Question 5
- [ ] Done

**Write a MapReduce job to find duplicate records.**

### Answer

### Python Code Example
```python
from mrjob.job import MRJob

class FindDuplicates(MRJob):
    """
    Pipeline:
    Input: Records (one per line)
    Map: Emit (record, 1)
    Reduce: Count occurrences, emit if > 1
    Output: Duplicate records with counts
    """
    
    def mapper(self, _, line):
        """Emit entire record as key"""
        record = line.strip()
        if record:
            yield record, 1
    
    def combiner(self, record, counts):
        """Local count"""
        yield record, sum(counts)
    
    def reducer(self, record, counts):
        """Emit only duplicates"""
        total = sum(counts)
        if total > 1:
            yield record, total

if __name__ == '__main__':
    FindDuplicates.run()
```

---

## Question 6
- [ ] Done

**Write a MapReduce job to compute inverted index.**

### Answer

### Python Code Example
```python
from mrjob.job import MRJob
from mrjob.step import MRStep
import os

class InvertedIndex(MRJob):
    """
    Pipeline:
    Input: Multiple text documents
    Map: For each word, emit (word, (doc_id, position))
    Reduce: Aggregate all documents containing each word
    Output: word -> list of (doc_id, positions)
    """
    
    def mapper(self, _, line):
        """Emit (word, doc_info)"""
        # Get filename from environment or input path
        try:
            input_file = os.environ.get('mapreduce_map_input_file', 'unknown')
            doc_id = os.path.basename(input_file)
        except:
            doc_id = 'unknown'
        
        words = line.strip().lower().split()
        for position, word in enumerate(words):
            yield word, (doc_id, position)
    
    def reducer(self, word, doc_positions):
        """Build posting list"""
        postings = {}
        for doc_id, position in doc_positions:
            if doc_id not in postings:
                postings[doc_id] = []
            postings[doc_id].append(position)
        
        yield word, dict(postings)

if __name__ == '__main__':
    InvertedIndex.run()

# Output example:
# "hadoop" -> {"doc1.txt": [5, 20], "doc2.txt": [3]}
```

---

## Question 7
- [ ] Done

**Write code to upload and process files in HDFS.**

### Answer

### Python Code Example
```python
from hdfs import InsecureClient
import os

class HDFSProcessor:
    """
    Pipeline:
    1. Connect to HDFS
    2. Upload local files
    3. List and verify
    4. Read and process
    """
    
    def __init__(self, namenode_url, user='hadoop'):
        self.client = InsecureClient(namenode_url, user=user)
    
    def upload_file(self, local_path, hdfs_path):
        """Upload single file"""
        self.client.upload(hdfs_path, local_path, overwrite=True)
        status = self.client.status(hdfs_path)
        print(f"Uploaded: {hdfs_path}, Size: {status['length']} bytes")
        return status
    
    def upload_directory(self, local_dir, hdfs_dir):
        """Upload entire directory"""
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                hdfs_path = f"{hdfs_dir}/{relative_path}"
                self.upload_file(local_path, hdfs_path)
    
    def list_files(self, hdfs_path, recursive=False):
        """List files in directory"""
        if recursive:
            return list(self.client.walk(hdfs_path))
        return self.client.list(hdfs_path, status=True)
    
    def read_file(self, hdfs_path):
        """Read file content"""
        with self.client.read(hdfs_path, encoding='utf-8') as reader:
            return reader.read()
    
    def process_files(self, hdfs_dir, processor_func):
        """Apply function to all files"""
        results = {}
        for filename in self.client.list(hdfs_dir):
            path = f"{hdfs_dir}/{filename}"
            content = self.read_file(path)
            results[filename] = processor_func(content)
        return results

# Usage
if __name__ == '__main__':
    hdfs = HDFSProcessor('http://namenode:50070')
    
    # Upload
    hdfs.upload_file('data.csv', '/user/data/data.csv')
    
    # List
    files = hdfs.list_files('/user/data')
    print(f"Files: {files}")
    
    # Process
    def count_lines(content):
        return len(content.split('\n'))
    
    line_counts = hdfs.process_files('/user/data', count_lines)
    print(f"Line counts: {line_counts}")
```

---

## Question 8
- [ ] Done

**Write a MapReduce job to compute percentiles.**

### Answer

### Python Code Example
```python
from mrjob.job import MRJob
from mrjob.step import MRStep
import numpy as np

class PercentileCalculator(MRJob):
    """
    Pipeline:
    Step 1: Group values by key
    Step 2: Compute percentiles
    Output: Key with percentile values
    """
    
    def configure_args(self):
        super().configure_args()
        self.add_passthru_arg('--percentiles', default='25,50,75,90,95')
    
    def steps(self):
        return [
            MRStep(mapper=self.mapper,
                   reducer=self.reducer_collect),
            MRStep(reducer=self.reducer_percentiles)
        ]
    
    def mapper(self, _, line):
        """Parse and emit (category, value)"""
        try:
            parts = line.strip().split(',')
            category = parts[0]
            value = float(parts[1])
            yield category, value
        except:
            pass
    
    def reducer_collect(self, category, values):
        """Collect all values (send to single reducer)"""
        yield category, list(values)
    
    def reducer_percentiles(self, category, value_lists):
        """Calculate percentiles"""
        all_values = []
        for vlist in value_lists:
            all_values.extend(vlist)
        
        percentiles = [int(p) for p in self.options.percentiles.split(',')]
        
        results = {}
        for p in percentiles:
            results[f'p{p}'] = round(np.percentile(all_values, p), 2)
        
        results['count'] = len(all_values)
        results['min'] = min(all_values)
        results['max'] = max(all_values)
        
        yield category, results

if __name__ == '__main__':
    PercentileCalculator.run()
```

## Question 9
- [ ] Done

**Write a MapReduce job in Java that counts the number of words in a text file**

*Answer to be added.*

---

## Question 10
- [ ] Done

**Create an HDFS command script that copies files from a local file system to HDFS**

*Answer to be added.*

---

## Question 11
- [ ] Done

**Implement a simple Hive query to summarize data from a Hive table**

*Answer to be added.*

---

## Question 12
- [ ] Done

**Code a Pig Latin script to process and transform a dataset into a desired format**

*Answer to be added.*

---

## Question 13
- [ ] Done

**Automate a process to import data from a MySQL database into HDFS using Sqoop**

*Answer to be added.*

---

## Question 14
- [ ] Done

**Implement a Spark job in Scala or Python to perform a join operation on two datasets**

*Answer to be added.*

---

## Question 15
- [ ] Done

**Develop an Oozie workflow that schedules and runs a set of MapReduce and Hive jobs**

*Answer to be added.*

---

## Question 16
- [ ] Done

**Write a Java program to implement custom input and output format classes in Hadoop**

*Answer to be added.*

---
