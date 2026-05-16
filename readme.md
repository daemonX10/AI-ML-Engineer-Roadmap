# Artificial Intelligence and Machine Learning Workspace (2023-2027)

A refactored, end-to-end learning workspace that tracks my AI/ML/GenAI journey from foundations to production systems. The repository is organized as a staged learning path and contains notebooks, code, datasets, PDFs, and reference notes.

| Item | Value |
| --- | --- |
| Last updated | 2026-05-16 |
| Status | Active (refactored) |
| Primary formats | Jupyter notebooks, Markdown, PDFs |
| Focus | AI, ML, GenAI, MLOps |

## Table of contents
- [How to use this repo](#how-to-use-this-repo)
- [Learning path and repository map](#learning-path-and-repository-map)
- [Question banks and interview prep](#question-banks-and-interview-prep)
- [Key resources](#key-resources)
- [Roadmap and next steps](#roadmap-and-next-steps)
- [Repository notes](#repository-notes)

## How to use this repo
1. Follow the numbered folders in order for a guided path (1.0 -> 7.1).
2. Use day-based folders for lecture-style sequences and practice.
3. Track progress in [organized_questions/PROGRESS.md](organized_questions/PROGRESS.md) and [organized_questions/track_progress.py](organized_questions/track_progress.py).
4. Use [Todo.md](Todo.md) as the long-term roadmap and milestone checklist.

## Learning path and repository map

### 0. Root navigation
- [HELLOWORLD.py](HELLOWORLD.py) is a simple sanity-check script.
- [Todo.md](Todo.md) is the master roadmap and long-horizon plan.

### 1. Python foundations
- 1.0 Python covers core programming and tooling.
  - 0. live lecture: live session notebooks.
  - 1.1 Tuple,dictionaries,set: data structures.
  - 1.2 string and list: string and list operations.
  - 1.3 Function: functions, lambdas, generators.
  - 1.4 Files: file I/O, logging, exceptions.
  - 1.5 Module and packages: module structure and imports.
  - 1.6 multi_processing: multiprocessing basics.
  - 1.7 Multi_Threading: threading practice.
  - 1.8 Oops: OOP concepts and class patterns.
  - Flask + Deployment: web app basics and deployment demos.
  - MongoDB: database practice.
  - Web scrapping: scraping notebooks.
- Python-practice is a practice sandbox.
  - core: foundational notebooks and file handling.
  - python Numerical: numeric practice notebooks.

### 2. Data analysis and visualization
- 2.0 N P V is the NumPy, Pandas, Visualization track.
  - Numpy: array programming and vectorization.
  - Pandas: data wrangling, CSV and Excel workflows.
  - Visualization: Matplotlib, Seaborn, Plotly, Bokeh demos.

### 3. Programming patterns and performance
- 2.5 Programming Patterns focuses on optimization and SE practice.
  - 1.0 Profiling and Performance Analysis: profiling notes and demos.
  - 2.0 Cython: Cython setup and usage.
  - 3.0 Numba: JIT optimization notes.
  - 4.0 Parallel Processing: concurrency patterns and notes.
  - 5.0 Memory Mangagement: memory notes.
  - 6.0 SE Best Practices: software engineering practices.
  - 7.0 Implementation Exercises: applied exercises.
  - [2.5 Programming Patterns/ADP.md](2.5%20Programming%20Patterns/ADP.md) summarizes advanced design patterns.

### 4. Statistics and mathematics
- 3.0 Statistics is the day-by-day statistics curriculum.
  - 06 days to 14 days: lecture-driven notebooks and PDFs.
  - Notes: supplemental notes and PDFs.
  - [3.0 Statistics/Probability_ML_Master_Notes.md](3.0%20Statistics/Probability_ML_Master_Notes.md) is the probability master notes.
  - [3.0 Statistics/prompt.md](3.0%20Statistics/prompt.md) contains study prompts.
  - [3.0 Statistics/statistics.pdf](3.0%20Statistics/statistics.pdf) is the statistics reference.
- 3.5 Advance Statistics expands into advanced math topics.
  - 01. Eigen In ML
  - 02. Matrix Decomposition
  - 03. Advance Transformation and Space
  - 04. Calculas Extension
  - 05. Lagrangian Multipliers and Constrained Optimization
  - 06. Gradient Descent Variants and Convergence Properties
  - 07. Probability Theory Mastery
  - 08. Advanced Sampling Methods
  - 09. Information Theory
  - [3.5 Advance Statistics/statistics.md](3.5%20Advance%20Statistics/statistics.md) is the section summary.

### 5. Feature engineering and EDA
- 4.0 Feature Engineering + EDA is the applied data prep track.
  - 1 Data handling: missing values, outliers, imbalance.
  - 2 feature scaling + extracting: scaling and extraction workflows.
  - 3 Data encoding: categorical encoding techniques.
  - 4 Covariance_Correlation: correlation analysis.
  - 5 Exploratory Data Analysis: EDA case studies.
  - [4.0 Feature Engineering + EDA/readme.md](4.0%20Feature%20Engineering%20+%20EDA/readme.md) enumerates notebooks and files.

### 6. Machine learning core
- 5.0 Machine learning holds core ML algorithms and projects.
  - 1 SUPERVISED: linear regression, ridge/lasso, logistic regression, decision trees, SVM, Naive Bayes, K-NN, and a full Algerian forest project.
  - 2 Unsupervised: K-Means, hierarchical clustering, DBSCAN notebooks and PDFs.
  - 3 Ensemble Technique: bagging, boosting, stacking.
  - 4 Dimension Reduction: PCA and dimensionality reduction notebooks.
  - 5 Time Series: EDA and forecasting notebooks with PDFs.
  - [5.0 Machine learning/Amazing Machine Learning book .pdf](5.0%20Machine%20learning/Amazing%20Machine%20Learning%20book%20.pdf) is a core reference book.
  - [5.0 Machine learning/DOC-20240509-WA0001..pdf](5.0%20Machine%20learning/DOC-20240509-WA0001..pdf) is an additional reference.

### 7. Acceleration and frameworks
- 5.1 CUDA contains GPU basics and advanced notes.
  - 1.0 Basic
  - 2.0 Advance
  - [5.1 CUDA/notes.md](5.1%20CUDA/notes.md)
- 5.2 PYTORCH is the framework track.
  - 1.0 Fundamentals
  - 2.0 Workflow
  - One-short

### 8. Advanced ML techniques
- 5.5 Advanced ML Techniques extends the core ML stack.
  - 1.0 Advanced Supervised Learning
  - 2.0 Unsupervised Learning Extensions
  - 3.0 Time Series Advanced Techniques
  - 4.0 Implementation Exercises

### 9. Deep learning
- 6.0 Deep Learning is the main DL curriculum.
  - CampusX: ANN, CNN, RNN tracks with notebooks and datasets.
  - Main Resource: ANN & Performance, Pytorch and Tensorflow, CNN, Object Detection, GAN, RNN.
  - [6.0 Deep Learning/DL.pdf](6.0%20Deep%20Learning/DL.pdf) is the DL reference PDF.

### 10. NLP and LLM foundations
- 6.0 NLP And Basic LLM covers NLP fundamentals and early LLM work.
  - 1.0 Notes: slide decks and PDFs.
  - 1.0 Text Processing Technique: preprocessing, embeddings, tokenization.
  - Day_01 to Day_13: day-wise NLP curriculum, including RNN/LSTM/GRU and Transformers.

### 11. Deep learning expertise
- 6.5 Deep Learning Expertise is the advanced DL track.
  - 1.0 Neural Network Architecture Mastery
  - 2.0 Deep Learning Optimization
  - 3.0 Generative Models GANs
  - Implementation Exercises

### 12. NLP-LLM-GenAI stack
- 6.5 NLP-LLM-GENAI is the advanced NLP and GenAI track.
  - 1.0 Advanced NLP
  - 2.0 Large Language Models
  - 3.0 GenAI Applications
  - 4.0 LANGCHAIN: Intro, Components, Models, Prompts, Parser, Chains, Runnables, Memory, Document Loaders, Text Splitters, Vector Stores, Retriever, RAG, Tools, Agentic AI.
  - 5.0 LANGGRAPH
  - 6.0 LangFamily

### 13. Advanced LLMs and agents
- 7.0 Advance LLM targets engineering and systems.
  - 1.0 LLM Engineering
  - 2.0 RAG (Retrieval-Augmented Generation)
  - 2.5 MCP
  - 3.0 AI Agents & Tools
  - Scratch: from-scratch GPT and tokenization work.

### 14. MLOps and production systems
- 7.1 MLOPS focuses on production ML.
  - 1.0 Model Deployment & Serving Systems
  - 2.0 Distributed Training & Large-Scale ML
  - 3.0 ML System Design & Optimization
  - Implementation Exercises

## Question banks and interview prep
- organized_questions is the structured question repository.
  - [organized_questions/README.md](organized_questions/README.md) explains the full learning path and question types.
  - 01_foundations_mathematics: linear algebra and time series.
  - 02_programming_tools: Python, SQL, NumPy, Pandas, scikit-learn, TensorFlow, Keras, PyTorch, Hadoop, Spark, MATLAB.
  - 03_data_science: probability, statistics, data processing, roles, model evaluation, ML design patterns, MLOps, LLMOps.
  - 04_machine_learning: ML fundamentals through advanced DL and RL topics.
  - 06_algorithms_optimization: cost functions, optimization, genetic algorithms, Q-learning, LightGBM, recommendations.
  - 07_computer_vision: core CV questions, architectures, segmentation, generative models.
  - 08_natural_language_processing: fundamentals, understanding, generation.
  - 09_large_language_models_genai: LLM architectures, embeddings, applications.
  - 10_explainable_ai: theory and code.
  - 11_model_evaluation_metrics: theory and code.
  - AI Questions only: topic-based AI/ML notes and roadmaps.
  - DSA: a full DSA interview bank with counts in [organized_questions/DSA/README.md](organized_questions/DSA/README.md).
  - system design: software architecture and system design bank in [organized_questions/system design/README.md](organized_questions/system%20design/README.md).
  - Sheets: curated notebook series in [organized_questions/Sheets/Grind75ML/README.md](organized_questions/Sheets/Grind75ML/README.md) and [organized_questions/Sheets/GrindLLM50/README.md](organized_questions/Sheets/GrindLLM50/README.md).
  - Additional: lecture notes, projects, research papers, and guides.
  - CS: CS fundamentals PDFs.
  - ALL Questions: aggregated question folders.
  - website: a web view of question content.
  - [organized_questions/PROGRESS.md](organized_questions/PROGRESS.md) and [organized_questions/track_progress.py](organized_questions/track_progress.py) track study progress.

## Key resources
- [2.0 N P V/A Quick Reference Handbook for Data Enthusiasts.pdf](2.0%20N%20P%20V/A%20Quick%20Reference%20Handbook%20for%20Data%20Enthusiasts.pdf)
- [3.0 Statistics/statistics.pdf](3.0%20Statistics/statistics.pdf)
- [5.0 Machine learning/Amazing Machine Learning book .pdf](5.0%20Machine%20learning/Amazing%20Machine%20Learning%20book%20.pdf)
- [5.0 Machine learning/DOC-20240509-WA0001..pdf](5.0%20Machine%20learning/DOC-20240509-WA0001..pdf)
- [6.0 Deep Learning/DL.pdf](6.0%20Deep%20Learning/DL.pdf)

## Roadmap and next steps
The full roadmap is in [Todo.md](Todo.md). The current long-horizon phases are:
1. Foundation reinforcement (math, optimization, performance engineering).
2. Advanced supervised and unsupervised ML.
3. Deep learning architectures and optimization.
4. NLP and LLM systems.
5. Advanced LLM engineering, RAG, and agents.
6. MLOps and production systems.
7. Responsible AI and security.
8. Specialization and applied projects.

## Repository notes
- This repo is intentionally recursive with nested topic folders and day-based sequences.
- Expect large data and PDF files alongside notebooks.
- Some folders contain scratch or experiment artifacts; they are kept to preserve learning context.
