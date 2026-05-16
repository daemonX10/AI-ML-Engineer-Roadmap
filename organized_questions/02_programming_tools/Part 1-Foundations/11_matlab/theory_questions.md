# MATLAB Interview Questions - Theory Questions

## Question 1
- [ ] Done

**What are the main features of MATLAB that make it suitable for machine learning?**

### Answer

**Definition**: MATLAB (Matrix Laboratory) is a high-level programming environment designed for numerical computing, visualization, and algorithm development.

### Key Features for ML

| Feature | Description |
|---------|-------------|
| **Matrix operations** | Native matrix/vector support |
| **Toolboxes** | Statistics, ML, Deep Learning toolboxes |
| **Visualization** | Built-in plotting functions |
| **Prototyping** | Rapid algorithm development |
| **Integration** | C/C++, Python, hardware integration |

### MATLAB Code Example
```matlab
% Load sample data
load fisheriris
X = meas;
y = species;

% Train classifier
mdl = fitcsvm(X, y);

% Cross-validation
cvmdl = crossval(mdl);
loss = kfoldLoss(cvmdl);
fprintf('CV Error: %.2f%%\n', loss*100);

% Visualize
gscatter(X(:,1), X(:,2), y);
xlabel('Sepal Length');
ylabel('Sepal Width');
title('Iris Dataset');
```

---

## Question 2
- [ ] Done

**Explain MATLAB's matrix operations and their importance.**

### Answer

### Core Matrix Operations

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Addition | `A + B` | Element-wise addition |
| Multiplication | `A * B` | Matrix multiplication |
| Element-wise mult | `A .* B` | Element-wise multiplication |
| Transpose | `A'` | Matrix transpose |
| Inverse | `inv(A)` | Matrix inverse |
| Determinant | `det(A)` | Matrix determinant |

### MATLAB Code Example
```matlab
% Create matrices
A = [1 2; 3 4];
B = [5 6; 7 8];

% Matrix operations
C = A * B;           % Matrix multiplication
D = A .* B;          % Element-wise multiplication
E = A';              % Transpose
F = inv(A);          % Inverse

% Solve linear system Ax = b
b = [1; 2];
x = A \ b;           % More efficient than inv(A)*b

% Eigenvalues and eigenvectors
[V, D] = eig(A);

% SVD decomposition
[U, S, V] = svd(A);

fprintf('Matrix product:\n');
disp(C);
fprintf('Solution to Ax=b:\n');
disp(x);
```

### Interview Tip
Always use `\` (backslash) for solving linear systems instead of `inv()` - it's more numerically stable and efficient.

---

## Question 3
- [ ] Done

**What are MATLAB toolboxes and which ones are relevant for ML?**

### Answer

### ML-Related Toolboxes

| Toolbox | Purpose |
|---------|---------|
| **Statistics & ML** | Classical ML algorithms |
| **Deep Learning** | Neural networks, CNNs, RNNs |
| **Computer Vision** | Image processing, object detection |
| **Signal Processing** | Time series, filtering |
| **Optimization** | Convex optimization |
| **Parallel Computing** | GPU, distributed computing |

### MATLAB Code Example
```matlab
% Statistics and Machine Learning Toolbox
load carsmall
X = [Weight, Horsepower];
y = MPG;

% Remove NaN
idx = ~any(isnan([X y]), 2);
X = X(idx,:);
y = y(idx);

% Fit linear regression
mdl = fitlm(X, y);
disp(mdl);

% Random Forest
rf_mdl = TreeBagger(100, X, y, 'Method', 'regression');
y_pred = predict(rf_mdl, X);

% Deep Learning Toolbox
layers = [
    featureInputLayer(2)
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];

options = trainingOptions('adam', 'MaxEpochs', 100);
% net = trainNetwork(X, y, layers, options);
```

---

## Question 4
- [ ] Done

**Explain the difference between scripts and functions in MATLAB.**

### Answer

### Comparison

| Feature | Script | Function |
|---------|--------|----------|
| **Scope** | Workspace variables | Local variables |
| **Input/Output** | None | Arguments/returns |
| **Reusability** | Limited | High |
| **File** | `.m` file | `.m` file with function keyword |

### MATLAB Code Example
```matlab
% SCRIPT: my_script.m
% Operates on workspace variables
data = randn(100, 1);
mean_val = mean(data);
fprintf('Mean: %.2f\n', mean_val);

% FUNCTION: my_function.m
function [mean_val, std_val] = compute_stats(data)
    % Computes statistics for input data
    % Input: data - numeric array
    % Output: mean_val, std_val - statistics
    
    mean_val = mean(data);
    std_val = std(data);
end

% Anonymous function
square = @(x) x.^2;
result = square(5);  % Returns 25

% Function with validation
function y = safe_divide(a, b)
    arguments
        a double
        b double {mustBeNonzero}
    end
    y = a / b;
end
```

---

## Question 5
- [ ] Done

**What is vectorization in MATLAB and why is it important?**

### Answer

**Definition**: Vectorization is writing code that operates on entire arrays at once, avoiding explicit loops.

### Benefits

| Benefit | Description |
|---------|-------------|
| **Speed** | 10-100x faster than loops |
| **Readability** | Cleaner code |
| **Memory** | Better memory management |

### MATLAB Code Example
```matlab
n = 1000000;
x = randn(n, 1);

% SLOW: Loop-based
tic
y_loop = zeros(n, 1);
for i = 1:n
    y_loop(i) = x(i)^2 + 2*x(i) + 1;
end
loop_time = toc;

% FAST: Vectorized
tic
y_vec = x.^2 + 2*x + 1;
vec_time = toc;

fprintf('Loop time: %.4f s\n', loop_time);
fprintf('Vectorized time: %.4f s\n', vec_time);
fprintf('Speedup: %.1fx\n', loop_time/vec_time);

% More vectorization examples
A = randn(1000, 1000);
B = randn(1000, 1000);

% Matrix operations are inherently vectorized
C = A * B;

% Logical indexing (vectorized)
positive_vals = A(A > 0);

% Apply function to each element
result = arrayfun(@(x) x^2, A);  % Slower
result = A.^2;  % Faster, vectorized
```

---

## Question 6
- [ ] Done

**Explain memory management in MATLAB.**

### Answer

### Memory Concepts

| Concept | Description |
|---------|-------------|
| **Copy-on-write** | Data copied only when modified |
| **In-place operations** | Modify without copying |
| **Preallocation** | Allocate before loops |
| **Clear** | Release memory |

### MATLAB Code Example
```matlab
% BAD: Growing array in loop
tic
data_bad = [];
for i = 1:10000
    data_bad = [data_bad; i];  % Reallocates each iteration
end
bad_time = toc;

% GOOD: Preallocate
tic
data_good = zeros(10000, 1);
for i = 1:10000
    data_good(i) = i;
end
good_time = toc;

fprintf('Without preallocation: %.4f s\n', bad_time);
fprintf('With preallocation: %.4f s\n', good_time);

% Check memory usage
whos

% Clear variables
clear data_bad data_good

% Monitor memory
memory  % Windows only

% Use sparse matrices for large sparse data
I = speye(10000);  % Sparse identity matrix
whos I
```

---

## Question 7
- [ ] Done

**What are cell arrays and structures in MATLAB?**

### Answer

### Comparison

| Type | Use Case | Access |
|------|----------|--------|
| **Cell array** | Mixed data types | `{}` |
| **Structure** | Named fields | `.fieldname` |
| **Table** | Tabular data | `.VariableName` or `()` |

### MATLAB Code Example
```matlab
% Cell array - mixed types
cell_data = {1, 'text', [1 2 3], @sin};
disp(cell_data{1});      % Access element
disp(cell_data{2});      % 'text'

% Structure - named fields
person.name = 'Alice';
person.age = 30;
person.scores = [85, 90, 95];

fprintf('%s is %d years old\n', person.name, person.age);

% Structure array
people(1).name = 'Alice';
people(1).age = 30;
people(2).name = 'Bob';
people(2).age = 25;

% Access all names
names = {people.name};

% Table - best for ML data
T = table({'Alice'; 'Bob'}, [30; 25], [85; 90], ...
    'VariableNames', {'Name', 'Age', 'Score'});

disp(T);
disp(T.Age);  % Access column
disp(T(1,:)); % Access row
```

---

## Question 8
- [ ] Done

**Explain parallel computing in MATLAB.**

### Answer

### Parallel Options

| Feature | Use Case |
|---------|----------|
| **parfor** | Parallel for loops |
| **parfeval** | Async function execution |
| **GPU arrays** | GPU computing |
| **Distributed** | Cluster computing |

### MATLAB Code Example
```matlab
% Start parallel pool
pool = parpool('local', 4);

% Parallel for loop
n = 100;
results = zeros(n, 1);

parfor i = 1:n
    results(i) = expensive_computation(i);
end

% GPU computing
if gpuDeviceCount > 0
    A_gpu = gpuArray(randn(1000));
    B_gpu = gpuArray(randn(1000));
    C_gpu = A_gpu * B_gpu;  % Runs on GPU
    C = gather(C_gpu);  % Transfer back to CPU
end

% Async execution
f = parfeval(@expensive_computation, 1, 100);
% Do other work...
result = fetchOutputs(f);

% Delete pool when done
delete(pool);

function y = expensive_computation(x)
    pause(0.1);  % Simulate work
    y = x^2;
end
```

### Interview Tip
Not all loops can be parallelized with `parfor` - iterations must be independent.



---

# --- Missing Questions Restored from Source (Q9-Q25) ---

## Question 9
- [ ] Done

**What is the purpose of the ‘ eig ’ function, and how is it used?**

**Answer:**

### Definition
The `eig` function computes **eigenvalues and eigenvectors** of a square matrix, which are fundamental in linear algebra, PCA, stability analysis, and many ML algorithms.

### Mathematical Background
```
For matrix A:
  A * v = λ * v

where:
  λ = eigenvalue (scalar)
  v = eigenvector (direction unchanged by transformation)
```

### Usage
```matlab
A = [4 1; 2 3];

% Eigenvalues only
lambda = eig(A);          % Returns vector of eigenvalues [5; 2]

% Eigenvalues and eigenvectors
[V, D] = eig(A);
% V = matrix of eigenvectors (columns)
% D = diagonal matrix of eigenvalues

% Verify: A * V = V * D
disp(A * V);       % Should equal V * D
disp(V * D);

% Generalized eigenvalue problem: A*v = λ*B*v
B = [2 0; 0 1];
[V, D] = eig(A, B);
```

### Applications in ML

| Application | How `eig` Is Used |
|-------------|------------------|
| **PCA** | Eigendecomposition of covariance matrix |
| **Spectral clustering** | Eigenvectors of Laplacian matrix |
| **Stability analysis** | Eigenvalues of system dynamics matrix |
| **Google PageRank** | Dominant eigenvector of link matrix |

### PCA Example
```matlab
data = randn(100, 5);  % 100 samples, 5 features
C = cov(data);         % Covariance matrix
[V, D] = eig(C);      % Eigenvectors and eigenvalues

% Sort by eigenvalue (descending)
[eigenvalues, idx] = sort(diag(D), 'descend');
V_sorted = V(:, idx);

% Project to top 2 components
reduced = data * V_sorted(:, 1:2);
```

### Interview Tip
`eig` returns eigenvectors as columns of V. For symmetric matrices (like covariance matrices), eigenvectors are **orthogonal**. In PCA, eigenvectors with the **largest eigenvalues** capture the most variance. Use `eigs` (sparse) for large matrices where you only need the top-k eigenvalues.

---

## Question 10
- [ ] Done

**Explain how to customize plots in MATLAB (e.g. adding labels, titles, legends)**

**Answer:**

### Basic Plot Customization
```matlab
x = 0:0.1:2*pi;
y1 = sin(x);
y2 = cos(x);

% Create plot
figure;
plot(x, y1, 'r-', 'LineWidth', 2);  % Red solid line
hold on;
plot(x, y2, 'b--', 'LineWidth', 2); % Blue dashed line
hold off;

% Labels and title
xlabel('Time (s)', 'FontSize', 14);
ylabel('Amplitude', 'FontSize', 14);
title('Trigonometric Functions', 'FontSize', 16, 'FontWeight', 'bold');
legend('sin(x)', 'cos(x)', 'Location', 'best');

% Grid and axis
grid on;
axis([0 2*pi -1.5 1.5]);
```

### Customization Options

| Property | Options | Example |
|----------|---------|--------|
| **Line style** | `-`, `--`, `:`, `-.` | `plot(x, y, '--')` |
| **Color** | `r`, `g`, `b`, `k`, hex | `plot(x, y, 'Color', '#FF6600')` |
| **Marker** | `o`, `*`, `x`, `s`, `d` | `plot(x, y, 'o-')` |
| **LineWidth** | Numeric | `'LineWidth', 2` |
| **MarkerSize** | Numeric | `'MarkerSize', 8` |
| **FontSize** | Numeric | `'FontSize', 14` |

### Advanced Customization
```matlab
% Subplots
figure;
subplot(2, 1, 1); plot(x, y1); title('Sine');
subplot(2, 1, 2); plot(x, y2); title('Cosine');

% Annotations
text(pi, 0, '\leftarrow \pi', 'FontSize', 12);
annotation('arrow', [0.5 0.3], [0.7 0.5]);

% Save figure
saveas(gcf, 'plot.png');
exportgraphics(gcf, 'plot.pdf', 'Resolution', 300);
```

### Interview Tip
Use `set(gca, ...)` for axis properties and `set(gcf, ...)` for figure properties. For publication-quality plots, use `exportgraphics` (R2020a+) instead of `saveas` for better resolution control.

---

## Question 11
- [ ] Done

**What are the functions used to plot multiple data series in MATLAB ?**

**Answer:**

### Multiple Plot Functions

| Function | Description | Use Case |
|----------|-------------|----------|
| `plot` + `hold on` | Overlay lines on same axes | Comparing trends |
| `subplot` | Multiple plots in grid layout | Side-by-side comparison |
| `yyaxis` | Dual y-axes on same plot | Different scales |
| `plotyy` | Legacy dual y-axis | Deprecated, use `yyaxis` |
| `tiledlayout` | Modern grid layout (R2019b+) | Professional multi-panel |
| `stackedplot` | Stacked time-series | Multi-variable time data |

### Code Examples
```matlab
% Method 1: hold on/off
figure;
plot(x, sin(x), 'r-', 'LineWidth', 2);
hold on;
plot(x, cos(x), 'b--', 'LineWidth', 2);
plot(x, tan(x), 'g:', 'LineWidth', 2);
hold off;
legend('sin', 'cos', 'tan');

% Method 2: Matrix columns (each column = one series)
Y = [sin(x)', cos(x)', 0.5*sin(2*x)'];
plot(x, Y);  % Automatically plots 3 series

% Method 3: subplot
figure;
subplot(2, 2, 1); plot(x, sin(x)); title('sin');
subplot(2, 2, 2); plot(x, cos(x)); title('cos');
subplot(2, 2, 3); bar(1:5, rand(1,5)); title('bar');
subplot(2, 2, 4); pie([30 20 50]); title('pie');

% Method 4: tiledlayout (modern, preferred)
figure;
tiledlayout(2, 2);
nexttile; plot(x, sin(x)); title('sin');
nexttile; plot(x, cos(x)); title('cos');
nexttile; stem(1:10, rand(1,10)); title('stem');
nexttile; area(x, abs(sin(x))); title('area');

% Method 5: yyaxis (dual y-axes)
figure;
yyaxis left; plot(x, sin(x)); ylabel('Sin');
yyaxis right; plot(x, 100*cos(x)); ylabel('Cos (scaled)');
```

### Interview Tip
Prefer `tiledlayout`/`nexttile` over `subplot` for modern MATLAB (R2019b+) — it offers better spacing control and shared axes. For comparing series with different scales, `yyaxis` is cleaner than `plotyy`.

---

## Question 12
- [ ] Done

**Explain the various methods for data normalization and standardization in MATLAB**

**Answer:**

### Normalization vs Standardization

| Method | Formula | Range | When to Use |
|--------|---------|-------|------------|
| **Min-Max** | $(x - min) / (max - min)$ | [0, 1] | Neural networks, bounded algorithms |
| **Z-score** | $(x - \mu) / \sigma$ | Unbounded | Gaussian-distributed features |
| **Max-Abs** | $x / max(|x|)$ | [-1, 1] | Sparse data |
| **Robust** | $(x - median) / IQR$ | Unbounded | Data with outliers |

### MATLAB Implementation
```matlab
data = [10 200 3000; 15 180 2800; 20 220 3200; 12 190 2900];

% Method 1: normalize() function (R2018a+)
norm_minmax = normalize(data, 'range');           % Min-Max [0,1]
norm_zscore = normalize(data, 'zscore');           % Z-score
norm_center = normalize(data, 'center');           % Center (subtract mean)
norm_scale  = normalize(data, 'scale');            % Scale by std
norm_range  = normalize(data, 'range', [-1, 1]);   % Custom range

% Method 2: Manual implementation
% Min-Max normalization
data_min = min(data);
data_max = max(data);
norm_manual = (data - data_min) ./ (data_max - data_min);

% Z-score standardization
mu = mean(data);
sigma = std(data);
std_manual = (data - mu) ./ sigma;

% Method 3: mapminmax (Neural Network Toolbox)
[normalized, settings] = mapminmax(data', 0, 1);
% Apply same normalization to new data
new_normalized = mapminmax('apply', new_data', settings);

% Method 4: fitcecoc / ML functions (auto-normalize)
Mdl = fitcecoc(data, labels, 'Standardize', true);
```

### For ML Pipelines
```matlab
% Store normalization parameters from training
mu_train = mean(X_train);
sigma_train = std(X_train);

% Apply to both train and test
X_train_norm = (X_train - mu_train) ./ sigma_train;
X_test_norm  = (X_test - mu_train) ./ sigma_train;  % Use TRAIN stats!
```

### Interview Tip
Always normalize using **training set statistics** and apply those same parameters to test data — never compute stats from test data (data leakage). Use `normalize()` for quick operations. For ML models, many MATLAB functions support `'Standardize', true` as an option.

---

## Question 13
- [ ] Done

**Explain the concept of recursion in MATLAB**

**Answer:**

### Definition
Recursion is when a function **calls itself** to solve a problem by breaking it into smaller subproblems. Each call must have a **base case** to stop the recursion.

### Structure
```matlab
function result = recursive_func(input)
    if base_condition(input)     % Base case (stops recursion)
        result = base_value;
    else
        result = combine(recursive_func(smaller_input));  % Recursive call
    end
end
```

### Examples
```matlab
% Factorial: n! = n * (n-1)!
function result = factorial_recursive(n)
    if n <= 1
        result = 1;              % Base case
    else
        result = n * factorial_recursive(n - 1);  % Recursive call
    end
end

% Fibonacci: F(n) = F(n-1) + F(n-2)
function result = fibonacci(n)
    if n <= 1
        result = n;              % Base case: F(0)=0, F(1)=1
    else
        result = fibonacci(n-1) + fibonacci(n-2);
    end
end

% Binary search (recursive)
function idx = binary_search(arr, target, low, high)
    if low > high
        idx = -1;                % Not found
        return;
    end
    mid = floor((low + high) / 2);
    if arr(mid) == target
        idx = mid;
    elseif arr(mid) < target
        idx = binary_search(arr, target, mid+1, high);
    else
        idx = binary_search(arr, target, low, mid-1);
    end
end
```

### Recursion vs Iteration

| Aspect | Recursion | Iteration |
|--------|-----------|----------|
| **Readability** | Cleaner for tree/graph problems | Simpler for loops |
| **Memory** | Stack frames (can overflow) | Constant memory |
| **Performance** | Slower (function call overhead) | Faster |
| **MATLAB limit** | Default ~500 recursion depth | No limit |

### Interview Tip
MATLAB has a **recursion limit** (default ~500). Check with `get(0, 'RecursionLimit')` and set with `set(0, 'RecursionLimit', 1000)`. For performance-critical code, convert recursion to **iteration** or use **memoization** (cache results in a persistent variable or containers.Map).

---

## Question 14
- [ ] Done

**Explain how to export MATLAB data to an Excel file**

**Answer:**

### Methods for Exporting to Excel

| Function | Description | Best For |
|----------|-------------|----------|
| `writematrix` | Write matrix to file | Numeric data |
| `writetable` | Write table to file | Mixed data types |
| `writecell` | Write cell array to file | Heterogeneous data |
| `xlswrite` | Legacy Excel write | Older MATLAB versions |

### Code Examples
```matlab
% Method 1: writetable (recommended)
data = table([25; 30; 35], {'Alice'; 'Bob'; 'Charlie'}, [85.5; 92.3; 78.1], ...
    'VariableNames', {'Age', 'Name', 'Score'});
writetable(data, 'output.xlsx');
writetable(data, 'output.xlsx', 'Sheet', 'Results', 'Range', 'B2');

% Method 2: writematrix (numeric only)
matrix_data = rand(10, 5);
writematrix(matrix_data, 'output.xlsx', 'Sheet', 'Data');

% Method 3: writecell (mixed types)
cell_data = {'Name', 'Score', 'Grade'; 'Alice', 95, 'A'; 'Bob', 82, 'B'};
writecell(cell_data, 'output.xlsx');

% Method 4: Multiple sheets
writetable(train_data, 'ml_results.xlsx', 'Sheet', 'Training');
writetable(test_data, 'ml_results.xlsx', 'Sheet', 'Testing');
writetable(metrics, 'ml_results.xlsx', 'Sheet', 'Metrics');

% Method 5: Append to existing file
writetable(new_data, 'output.xlsx', 'WriteMode', 'append');
```

### Advanced Options
```matlab
% Custom formatting with ActiveX (Windows only)
excel = actxserver('Excel.Application');
workbook = excel.Workbooks.Open(fullfile(pwd, 'output.xlsx'));
sheet = workbook.Sheets.Item(1);
sheet.Range('A1').Font.Bold = true;
sheet.Range('A1:C1').Interior.Color = hex2dec('4472C4');
workbook.Save();
workbook.Close();
excel.Quit();
```

### Interview Tip
Use `writetable` for most cases — it handles headers, mixed types, and multiple sheets. `writematrix` is for pure numeric data. Note: `xlswrite` is legacy and doesn't work on Mac/Linux without Excel installed. `writetable`/`writematrix` work cross-platform.

---

## Question 15
- [ ] Done

**What are the different data formats that MATLAB supports for import and export ?**

**Answer:**

### Supported Data Formats

| Category | Formats | Import Function | Export Function |
|----------|---------|----------------|----------------|
| **Spreadsheet** | .xlsx, .xls, .csv | `readtable`, `readmatrix` | `writetable`, `writematrix` |
| **Text** | .txt, .csv, .tsv | `readtable`, `textscan` | `writetable`, `fprintf` |
| **MATLAB** | .mat | `load` | `save` |
| **Image** | .png, .jpg, .tiff, .bmp | `imread` | `imwrite` |
| **Audio** | .wav, .mp3, .flac | `audioread` | `audiowrite` |
| **Video** | .mp4, .avi | `VideoReader` | `VideoWriter` |
| **HDF5** | .h5, .hdf5 | `h5read` | `h5write` |
| **JSON** | .json | `jsondecode` | `jsonencode` |
| **XML** | .xml | `xmlread` | `xmlwrite` |
| **Database** | SQL databases | `database`, `fetch` | `sqlwrite` |
| **Web** | URLs, REST APIs | `webread` | `webwrite` |

### Code Examples
```matlab
% CSV/Text
data = readtable('data.csv');
writetable(data, 'output.csv');

% MAT file (MATLAB native, fastest)
save('workspace.mat', 'var1', 'var2');  % Save specific variables
load('workspace.mat');                   % Load all variables
save('data.mat', 'large_array', '-v7.3');  % HDF5 format for >2GB

% JSON
json_data = webread('https://api.example.com/data');
jsonStr = jsonencode(struct('name', 'Alice', 'age', 30));

% HDF5 (large scientific datasets)
h5create('data.h5', '/dataset1', [100 50]);
h5write('data.h5', '/dataset1', rand(100, 50));
result = h5read('data.h5', '/dataset1');

% Images
img = imread('photo.png');
imwrite(img, 'output.jpg', 'Quality', 95);
```

### Interview Tip
Use `.mat` format (v7.3) for large MATLAB data — it's HDF5-based and supports partial loading. For interoperability with Python/R, use **CSV**, **HDF5**, or **Parquet** (R2019a+). `readtable` is the most versatile import function, auto-detecting formats.

---

## Question 17
- [ ] Done

**Explain the process of embedding MATLAB code in a Java application**

**Answer:**

### Overview
MATLAB code can be compiled into **Java packages** using MATLAB Compiler SDK, allowing Java applications to call MATLAB functions without requiring a MATLAB installation.

### Process Steps

| Step | Action | Tool |
|------|--------|------|
| 1 | Write MATLAB function | MATLAB Editor |
| 2 | Compile to Java package | `compiler.build.javaPackage` or `deploytool` |
| 3 | Include JAR + MCR in Java project | Build system |
| 4 | Call MATLAB functions from Java | Generated API |

### Step-by-Step Implementation
```matlab
% Step 1: Create MATLAB function (myAnalysis.m)
function result = myAnalysis(data, threshold)
    result = data(data > threshold);
    result = mean(result);
end
```

```matlab
% Step 2: Compile to Java package
compiler.build.javaPackage('myAnalysis.m', ...
    'PackageName', 'com.myapp.analysis', ...
    'ClassName', 'MatlabAnalyzer');

% Or use deploytool GUI
deploytool
```

```java
// Step 3-4: Java code calling MATLAB function
import com.myapp.analysis.MatlabAnalyzer;
import com.mathworks.toolbox.javabuilder.*;

public class App {
    public static void main(String[] args) throws Exception {
        MatlabAnalyzer analyzer = new MatlabAnalyzer();
        try {
            // Call MATLAB function
            double[] data = {1.5, 2.3, 4.7, 5.1, 3.2};
            Object[] result = analyzer.myAnalysis(1, data, 3.0);
            MWArray output = (MWArray) result[0];
            System.out.println("Result: " + output);
        } finally {
            analyzer.dispose();
        }
    }
}
```

### Requirements
- **MATLAB Compiler SDK** (for building)
- **MATLAB Runtime (MCR)** on target machine (free, no MATLAB license needed)
- **javabuilder.jar** from MATLAB installation

### Alternative Integration Methods

| Method | Description |
|--------|-------------|
| **MATLAB Engine for Java** | Direct MATLAB session from Java (requires MATLAB) |
| **MATLAB Production Server** | REST API for MATLAB functions |
| **MEX** | C/C++ code callable from MATLAB |
| **Python integration** | `py.` prefix or MATLAB Engine for Python |

### Interview Tip
The key distinction: **MATLAB Compiler SDK** creates standalone JARs (no MATLAB license on deployment), while **MATLAB Engine for Java** requires a MATLAB installation. For production deployment, use Compiler SDK + MATLAB Runtime (MCR). MCR is free to distribute.

---

## Question 18
- [ ] Done

**Describe MATLAB’s capabilities for hypothesis testing**

**Answer:**

### Common Hypothesis Tests in MATLAB

| Test | Function | Purpose |
|------|----------|--------|
| **t-test (one sample)** | `ttest` | Mean vs known value |
| **t-test (two sample)** | `ttest2` | Compare two group means |
| **Paired t-test** | `ttest(x-y)` | Before/after comparison |
| **ANOVA (one-way)** | `anova1` | Compare 3+ group means |
| **Chi-square** | `chi2gof` | Goodness of fit |
| **KS test** | `kstest` | Distribution test |
| **Wilcoxon** | `signrank`, `ranksum` | Non-parametric alternative |
| **F-test** | `vartest2` | Compare variances |

### Code Examples
```matlab
% One-sample t-test: Is mean = 50?
data = [48 52 55 47 51 49 53 50 46 54];
[h, p, ci, stats] = ttest(data, 50);
fprintf('H0 rejected: %d, p-value: %.4f\n', h, p);
fprintf('95%% CI: [%.2f, %.2f]\n', ci(1), ci(2));

% Two-sample t-test: Do groups differ?
group_a = [85 90 88 92 87 91];
group_b = [78 82 80 85 79 81];
[h, p] = ttest2(group_a, group_b);
fprintf('Groups differ: %d (p=%.4f)\n', h, p);

% One-way ANOVA: Compare 3+ groups
group1 = [23 25 28 22 27];
group2 = [30 33 31 35 29];
group3 = [19 21 20 18 22];
[p, tbl, stats] = anova1([group1; group2; group3]');
multcompare(stats);  % Post-hoc pairwise comparisons

% Chi-square goodness of fit
observed = [50 30 20];
expected = [40 35 25];
[h, p] = chi2gof(1:3, 'Frequency', observed, 'Expected', expected);

% Kolmogorov-Smirnov test (normality)
[h, p] = kstest(data);  % Test against standard normal
[h, p] = lillietest(data);  % Lilliefors test (better for normality)
```

### Workflow
```
1. State hypotheses (H0, H1)
2. Choose significance level (α = 0.05)
3. Select appropriate test
4. Check assumptions (normality, equal variance)
5. Run test, get p-value
6. Reject H0 if p < α
```

### Interview Tip
Always check **assumptions** before running parametric tests: normality (`lillietest`, `kstest`), equal variance (`vartest2`). If assumptions violated, use **non-parametric alternatives** (Wilcoxon instead of t-test, Kruskal-Wallis instead of ANOVA). The `h` output is 1 (reject H0) or 0 (fail to reject).

---

## Question 19
- [ ] Done

**Explain how to use MATLAB for principal component analysis (PCA)**

**Answer:**

### PCA in MATLAB

| Function | Description |
|----------|------------|
| `pca()` | Built-in PCA (Statistics Toolbox) |
| `eig()` | Manual PCA via eigendecomposition |
| `svd()` | Manual PCA via SVD |

### Using Built-in `pca()`
```matlab
% Generate sample data
rng(42);
data = randn(200, 5);  % 200 samples, 5 features
data(:,2) = data(:,1) + 0.1*randn(200,1);  % Correlated feature

% Run PCA
[coeff, score, latent, tsquared, explained, mu] = pca(data);

% coeff    = Principal component coefficients (loadings)
% score    = Projected data (principal components)
% latent   = Eigenvalues (variance of each component)
% explained = Percentage of variance explained
% mu       = Mean of each variable

% Display variance explained
fprintf('Variance explained:\n');
for i = 1:length(explained)
    fprintf('  PC%d: %.2f%%\n', i, explained(i));
end
```

### Dimensionality Reduction
```matlab
% Keep components explaining 95% variance
cumulative = cumsum(explained);
num_components = find(cumulative >= 95, 1);
fprintf('Components for 95%% variance: %d\n', num_components);

% Reduce dimensions
data_reduced = score(:, 1:num_components);

% Scree plot
figure;
pareto(explained);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('Scree Plot');
```

### PCA for ML Pipeline
```matlab
% Standardize first (important!)
data_std = normalize(data, 'zscore');
[coeff, score, ~, ~, explained] = pca(data_std);

% Apply to new data
new_data_std = (new_data - mu) ./ std(data);
new_score = new_data_std * coeff(:, 1:num_components);

% Visualize 2D projection
figure;
gscatter(score(:,1), score(:,2), labels);
xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
title('PCA Projection');
```

### Manual PCA (using eig)
```matlab
% Center data
data_centered = data - mean(data);

% Covariance matrix
C = cov(data_centered);

% Eigendecomposition
[V, D] = eig(C);
[eigenvalues, idx] = sort(diag(D), 'descend');
V = V(:, idx);

% Project
scores = data_centered * V;
```

### Interview Tip
Always **standardize** data before PCA if features have different scales. `pca()` centers data automatically but doesn't scale. The number of components is chosen by the **elbow method** (scree plot) or a **variance threshold** (e.g., 95%). PCA is also useful for **multicollinearity** detection — near-zero eigenvalues indicate collinear features.

---

## Question 20
- [ ] Done

**What is the Deep Learning Toolbox in MATLAB , and what can it be used for?**

**Answer:**

### Definition
The Deep Learning Toolbox provides functions and apps for designing, training, and deploying **deep neural networks** in MATLAB, with support for CNNs, RNNs, LSTMs, GANs, and transfer learning.

### Key Features

| Feature | Description |
|---------|-------------|
| **Network Designer** | Visual drag-and-drop network builder |
| **Pre-trained models** | AlexNet, VGG, ResNet, GoogLeNet, etc. |
| **Transfer learning** | Fine-tune pre-trained networks |
| **GPU training** | CUDA/cuDNN acceleration |
| **Training monitor** | Real-time loss/accuracy visualization |
| **ONNX support** | Import/export ONNX models |
| **Code generation** | Deploy to embedded systems (GPU Coder) |

### Supported Architectures

| Architecture | Use Case | MATLAB Layer |
|-------------|----------|-------------|
| **CNN** | Image classification | `convolution2dLayer` |
| **LSTM** | Sequence/time-series | `lstmLayer` |
| **GRU** | Sequence (faster) | `gruLayer` |
| **Autoencoder** | Anomaly detection | Custom architecture |
| **GAN** | Image generation | Custom training loop |
| **Transformer** | NLP (R2023a+) | `transformerLayer` |

### Code Example: Image Classification
```matlab
% Transfer learning with ResNet-18
net = resnet18;
layers = net.Layers;

% Modify for custom classes
numClasses = 5;
layers(end-2) = fullyConnectedLayer(numClasses);
layers(end) = classificationLayer;

% Prepare data
imds = imageDatastore('images/', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[trainData, testData] = splitEachLabel(imds, 0.8);

% Augmentation
augTrainData = augmentedImageDatastore([224 224], trainData, ...
    'DataAugmentation', imageDataAugmenter('RandRotation', [-20 20]));

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', testData, ...
    'Plots', 'training-progress');

% Train
trainedNet = trainNetwork(augTrainData, layers, options);

% Evaluate
predictions = classify(trainedNet, testData);
accuracy = mean(predictions == testData.Labels);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
```

### Interview Tip
MATLAB's Deep Learning Toolbox excels in **rapid prototyping** and **embedded deployment** (via GPU Coder/MATLAB Coder). It can import models from **TensorFlow and PyTorch** via ONNX. The tradeoff vs Python: fewer community models and less flexibility, but better integration with Simulink for control systems and signal processing.

---

## Question 21
- [ ] Done

**Describe how MATLAB could be utilized for signal processing and analysis**

**Answer:**

### Signal Processing Toolbox Capabilities

| Category | Functions | Use Case |
|----------|----------|----------|
| **Filtering** | `filter`, `designfilt`, `lowpass` | Remove noise, isolate frequencies |
| **FFT** | `fft`, `ifft`, `fftshift` | Frequency domain analysis |
| **Spectral** | `pwelch`, `spectrogram`, `periodogram` | Power spectral density |
| **Wavelets** | `cwt`, `dwt`, `wavedec` | Time-frequency analysis |
| **Resampling** | `resample`, `decimate`, `interp` | Change sampling rate |
| **Window** | `hamming`, `hanning`, `blackman` | Spectral leakage reduction |

### Code Examples
```matlab
% Generate noisy signal
fs = 1000;  % Sampling frequency
t = 0:1/fs:1;
signal = sin(2*pi*50*t) + sin(2*pi*120*t);  % 50 Hz + 120 Hz
noisy = signal + 0.5*randn(size(t));

% FFT analysis
N = length(noisy);
Y = fft(noisy);
f = fs*(0:N/2)/N;
P = abs(Y(1:N/2+1))/N;
figure; plot(f, P); xlabel('Frequency (Hz)'); title('FFT');

% Low-pass filter (remove 120 Hz)
lpFilt = designfilt('lowpassfir', 'PassbandFrequency', 80, ...
    'StopbandFrequency', 100, 'SampleRate', fs);
filtered = filter(lpFilt, noisy);

% Spectrogram (time-frequency representation)
figure; spectrogram(noisy, 256, 200, 256, fs, 'yaxis');
title('Spectrogram');

% Power spectral density
[pxx, f] = pwelch(noisy, [], [], [], fs);
figure; plot(f, 10*log10(pxx)); xlabel('Hz'); ylabel('dB/Hz');
```

### ML Applications
```matlab
% Feature extraction from signals for ML
features = [];
for i = 1:num_signals
    sig = signals{i};
    features(i,:) = [
        mean(sig), std(sig), ...
        rms(sig), kurtosis(sig), ...
        bandpower(sig, fs, [0 50]), ...
        bandpower(sig, fs, [50 200])
    ];
end

% Train classifier on signal features
Mdl = fitcsvm(features, labels);
```

### Interview Tip
MATLAB is the **industry standard** for signal processing, especially in communications, radar, and biomedical engineering. Its DSP System Toolbox enables **real-time streaming** processing. For ML on signals, extract time-domain (mean, RMS, kurtosis) and frequency-domain (band power, spectral entropy) features before classification.

---

## Question 22
- [ ] Done

**Explain how MATLAB can be applied to design and train a predictive model for financial time-series data**

**Answer:**

### End-to-End Financial Time-Series Pipeline

| Step | MATLAB Tool | Action |
|------|-------------|--------|
| 1. **Data import** | `readtable`, `fetch` | Load stock/financial data |
| 2. **Preprocessing** | `fillmissing`, `normalize` | Handle gaps, scale |
| 3. **Feature engineering** | Custom functions | Technical indicators |
| 4. **Model training** | `fitrensemble`, `lstmLayer` | Regression/DL models |
| 5. **Evaluation** | `rmse`, `mape` | Backtest performance |
| 6. **Deployment** | `compiler.build` | Production deployment |

### Code Example
```matlab
% Step 1: Load and prepare data
data = readtable('stock_prices.csv');
prices = data.Close;
dates = data.Date;

% Step 2: Feature engineering
% Moving averages
sma_20 = movmean(prices, 20);
sma_50 = movmean(prices, 50);

% Returns and volatility
returns = diff(log(prices));
volatility = movstd(returns, 20);

% RSI (Relative Strength Index)
gains = max(diff(prices), 0);
losses = abs(min(diff(prices), 0));
avg_gain = movmean(gains, 14);
avg_loss = movmean(losses, 14);
rsi = 100 - (100 ./ (1 + avg_gain ./ avg_loss));

% Step 3: Create feature matrix
features = [sma_20(51:end), sma_50(51:end), ...
            volatility(50:end), rsi(50:end)];
target = prices(52:end);  % Next-day price

% Step 4: Train-test split (time-based, no shuffle!)
split = floor(0.8 * length(target));
X_train = features(1:split, :);
y_train = target(1:split);
X_test = features(split+1:end, :);
y_test = target(split+1:end);

% Step 5: Train model (ensemble regression)
Mdl = fitrensemble(X_train, y_train, 'Method', 'Bag', ...
    'NumLearningCycles', 100);

% Step 6: Evaluate
predictions = predict(Mdl, X_test);
rmse_val = sqrt(mean((predictions - y_test).^2));
mape_val = mean(abs((predictions - y_test) ./ y_test)) * 100;
fprintf('RMSE: %.2f, MAPE: %.2f%%\n', rmse_val, mape_val);

% Visualize
figure;
plot(y_test, 'b-', 'LineWidth', 1.5); hold on;
plot(predictions, 'r--', 'LineWidth', 1.5);
legend('Actual', 'Predicted');
title('Stock Price Prediction');
```

### LSTM Approach
```matlab
% Prepare sequences for LSTM
numFeatures = size(X_train, 2);
layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(128, 'OutputMode', 'last')
    dropoutLayer(0.3)
    fullyConnectedLayer(1)
    regressionLayer
];
options = trainingOptions('adam', 'MaxEpochs', 50, 'MiniBatchSize', 32);
net = trainNetwork(X_train', y_train', layers, options);
```

### Interview Tip
For financial time-series: **never shuffle** data (use chronological train/test split), use **walk-forward validation** instead of k-fold, and be cautious of **lookahead bias** in feature engineering. MATLAB's Econometrics Toolbox provides GARCH, ARIMA, and VAR models specifically designed for financial data.

---

## Question 23
- [ ] Done

**How does MATLAB support the deployment of machine learning models ?**

**Answer:**

### Deployment Options

| Method | Target | Requires MATLAB? | Tool |
|--------|--------|-------------------|------|
| **MATLAB Compiler** | Standalone app (.exe) | No (MCR needed) | `mcc` |
| **MATLAB Compiler SDK** | Java/Python/.NET library | No (MCR needed) | `compiler.build` |
| **MATLAB Production Server** | REST API/microservice | Server license | MPS |
| **MATLAB Coder** | C/C++ code | No | `codegen` |
| **GPU Coder** | CUDA code | No | `gpucoder` |
| **Simulink** | Embedded systems | No (code gen) | Embedded Coder |
| **ONNX export** | Cross-platform | No | `exportONNXNetwork` |

### Code Examples
```matlab
% 1. Export to ONNX (use in Python/C++)
net = trainNetwork(...);
exportONNXNetwork(net, 'model.onnx');

% 2. Generate C/C++ code
function prediction = predictModel(features)
    persistent mdl;
    if isempty(mdl)
        mdl = loadLearnerForCoder('trained_model.mat');
    end
    prediction = predict(mdl, features);
end
% Then: codegen predictModel -args {zeros(1,10)}

% 3. Standalone application
mcc -m myApp.m -o MyMLApp

% 4. Python package
compiler.build.pythonPackage('predict.m', ...
    'PackageName', 'ml_predictor');

% 5. Docker container (MATLAB Production Server)
compiler.build.productionServerArchive('predict.m');
```

### saveLearnerForCoder Workflow
```matlab
% Training phase
Mdl = fitcsvm(X_train, y_train);
saveLearnerForCoder(Mdl, 'trained_svm');

% Deployment phase (generates C code)
function label = predict_svm(X)
    persistent mdl;
    if isempty(mdl)
        mdl = loadLearnerForCoder('trained_svm');
    end
    label = predict(mdl, X);
end
```

### Interview Tip
MATLAB's deployment strength is **embedded systems** — MATLAB Coder generates optimized C/C++ code for microcontrollers (automotive, medical devices). This is where MATLAB beats Python significantly. For web/cloud deployment, MATLAB Production Server provides REST APIs but is less common than Python-based serving (Flask/FastAPI).

---

## Question 24
- [ ] Done

**What are the benefits and limitations of using MATLAB for machine learning in comparison to other programming languages like Python ?**

**Answer:**

### MATLAB vs Python for ML

| Aspect | MATLAB | Python |
|--------|--------|--------|
| **Ease of use** | Easier for math/engineering | Easier for general programming |
| **Toolboxes** | Curated, tested, documented | Open-source, vast but variable quality |
| **Visualization** | Built-in, interactive | Matplotlib, Plotly (more setup) |
| **Matrix ops** | Native (designed for matrices) | NumPy (excellent but add-on) |
| **Deep learning** | Deep Learning Toolbox | TensorFlow, PyTorch (more models) |
| **Community** | Academic, engineering | Largest ML community |
| **Cost** | Expensive licenses | Free |
| **Deployment** | Embedded (Coder), apps | Web, cloud, everywhere |
| **GPU support** | Good (Parallel Computing Toolbox) | Excellent (CUDA, cuDNN) |
| **Big data** | Limited (single machine) | Spark, Dask, distributed |

### Benefits of MATLAB

| Benefit | Details |
|---------|--------|
| **All-in-one environment** | IDE, debugger, profiler, visualization integrated |
| **Verified toolboxes** | Professionally maintained, documented, tested |
| **Simulink integration** | Unique for control systems and simulation |
| **Embedded deployment** | C/C++ code generation for microcontrollers |
| **Signal/Image processing** | Industry standard in these domains |
| **Interactive apps** | App Designer for GUI-based ML tools |
| **Technical support** | MathWorks provides professional support |

### Limitations of MATLAB

| Limitation | Details |
|------------|--------|
| **Cost** | $2,350+ base + $1,000+ per toolbox |
| **Community** | Smaller community, fewer tutorials |
| **Deep learning** | Fewer pre-trained models than PyTorch/TF |
| **Big data** | No native distributed computing (like Spark) |
| **Web/cloud** | Limited deployment options |
| **Package ecosystem** | No equivalent of pip/conda |
| **Open source** | Proprietary, vendor lock-in |

### When to Choose MATLAB
- Control systems and Simulink workflows
- Signal processing and communications
- Embedded systems deployment (automotive, medical)
- Academic research in engineering
- Rapid prototyping with guaranteed numerical accuracy

### When to Choose Python
- Production ML/DL systems
- Big data and distributed computing
- Web deployment and APIs
- Cost-sensitive projects
- NLP and transformer models

### Interview Tip
The honest answer: **Python dominates ML** in industry due to cost, community, and ecosystem. MATLAB's niche is **engineering applications** (control systems, signal processing, embedded deployment) where Simulink integration and code generation are critical. Many organizations use **both**: MATLAB for prototyping/simulation, Python for production ML.

---

## Question 25
- [ ] Done

**Describe how MATLAB can be used for image and video processing tasks in the context of machine learning**

**Answer:**

### Image Processing Toolbox + Computer Vision Toolbox

| Task | Functions | ML Application |
|------|----------|----------------|
| **Reading/Writing** | `imread`, `imwrite`, `VideoReader` | Data loading |
| **Preprocessing** | `imresize`, `rgb2gray`, `imadjust` | Data preparation |
| **Augmentation** | `imageDataAugmenter` | Training data expansion |
| **Feature extraction** | `extractHOGFeatures`, `extractLBPFeatures` | Traditional ML features |
| **Segmentation** | `imbinarize`, `watershed`, `semanticseg` | Region detection |
| **Object detection** | `trainYOLOv4Detector`, `trainFasterRCNNObjectDetector` | Localization |
| **Classification** | `trainNetwork` + CNN | Image categorization |
| **OCR** | `ocr` | Text recognition |

### Image Classification Pipeline
```matlab
% Load and prepare data
imds = imageDatastore('images/', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
[trainData, testData] = splitEachLabel(imds, 0.8, 'randomized');

% Data augmentation
augementer = imageDataAugmenter( ...
    'RandRotation', [-30 30], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10], ...
    'RandXReflection', true);
augData = augmentedImageDatastore([224 224], trainData, ...
    'DataAugmentation', augementer);

% Transfer learning with pre-trained network
net = resnet50;
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'});
numClasses = numel(categories(trainData.Labels));
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc_new')
    softmaxLayer('Name', 'softmax_new')
    classificationLayer('Name', 'output')];
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'avg_pool', 'fc_new');

% Train
options = trainingOptions('adam', 'MaxEpochs', 15, ...
    'MiniBatchSize', 32, 'InitialLearnRate', 1e-4, ...
    'ValidationData', testData, 'Plots', 'training-progress');
trainedNet = trainNetwork(augData, lgraph, options);
```

### Video Processing
```matlab
% Process video frames
v = VideoReader('video.mp4');
while hasFrame(v)
    frame = readFrame(v);
    % Classification per frame
    label = classify(trainedNet, imresize(frame, [224 224]));
    % Object detection
    [bboxes, scores] = detect(detector, frame);
    % Annotate
    annotated = insertObjectAnnotation(frame, 'rectangle', bboxes, labels);
end
```

### Object Detection
```matlab
% Train YOLO v4 detector
detector = trainYOLOv4ObjectDetector(trainingData, net, ...
    anchorBoxes, options);
[bboxes, scores, labels] = detect(detector, testImage);
```

### Interview Tip
MATLAB's Computer Vision Toolbox provides **end-to-end workflows** from data labeling (Image Labeler app) to deployment (GPU Coder). Its strength over Python/OpenCV is the **integrated GUI tools** for labeling, augmentation preview, and training monitoring. For production CV systems, Python (PyTorch + OpenCV) is more common, but MATLAB excels in **medical imaging** and **autonomous driving** prototyping (Simulink integration).

---

## Question 26
- [ ] Done

**Explain the MATLAB environment and its primary components**

### MATLAB Environment Overview

MATLAB (Matrix Laboratory) is an integrated numerical computing environment with several key components:

### Primary Components

| Component | Purpose |
|-----------|--------|
| **Command Window** | Interactive console for executing commands and expressions |
| **Workspace** | Displays all variables currently in memory with their values, types, and sizes |
| **Editor** | Script/function editor with syntax highlighting, debugging, and code folding |
| **Current Folder** | File browser for navigating directories and managing scripts |
| **Command History** | Log of previously executed commands (searchable) |
| **Figure Window** | Displays plots, images, and GUI elements |
| **Variable Editor** | Spreadsheet-like view for inspecting/editing matrices and tables |

### Additional Tools

```matlab
% App Designer       - build GUIs with drag-and-drop
% Simulink           - model-based design for dynamic systems
% Live Editor        - notebook-style (.mlx) combining code, output, and text
% Profiler           - identify performance bottlenecks
% MATLAB Drive       - cloud storage for scripts and data
```

### Toolboxes (Domain-Specific Add-ons)
```
Statistics & ML Toolbox     - classification, regression, clustering
Deep Learning Toolbox       - neural networks, CNNs, LSTMs
Image Processing Toolbox    - filtering, segmentation, feature extraction
Signal Processing Toolbox   - FFT, filtering, spectral analysis
Optimization Toolbox        - linear/nonlinear optimization
Control System Toolbox      - transfer functions, PID controllers
```

### Key Workflow Features
```matlab
% Help system
help mean            % quick help in command window
doc mean             % full documentation browser
lookfor 'correlation' % search for functions by keyword

% Path management
addpath('my_functions/')  % add directory to search path
savepath                  % save path for future sessions
```

> **Interview Tip:** MATLAB's strength is its **all-in-one environment** — editor, debugger, profiler, and visualizer are tightly integrated. Unlike Python (which needs separate packages: NumPy, Matplotlib, Jupyter), MATLAB provides everything out of the box. The tradeoff is cost (commercial license) and less open-source ecosystem support.

---

## Question 27
- [ ] Done

**What is the difference between MATLAB and Octave ?**

### MATLAB vs. GNU Octave

Octave is a free, open-source alternative designed to be largely compatible with MATLAB.

| Feature | MATLAB | Octave |
|---------|--------|--------|
| **License** | Commercial ($$$) | Free (GPL) |
| **IDE** | Full-featured (Editor, Debugger, Profiler, App Designer) | Basic GUI or command line |
| **Toolboxes** | 90+ official toolboxes | Community packages (Octave Forge) |
| **Performance** | Highly optimized (JIT, multi-threaded) | Slower for large computations |
| **Simulink** | Yes | No equivalent |
| **GPU Computing** | Built-in `gpuArray` | Limited |
| **Deep Learning** | Deep Learning Toolbox | Not available |
| **Syntax** | Reference standard | 99% compatible |
| **Support** | MathWorks support | Community only |
| **Deployment** | MATLAB Compiler, Coder | Limited |

### Key Syntax Differences

```matlab
% ---- String Handling ----
% MATLAB: both single and double quotes
str1 = 'hello';      % character array (both)
str2 = "hello";      % string object (MATLAB only, R2017a+)

% Octave: single quotes only (double quotes also work but differ)

% ---- End of Block ----
% MATLAB: end / endfor / endwhile / endif are all just 'end'
for i = 1:5
    disp(i);
end

% Octave: supports both 'end' and specific keywords
for i = 1:5
    disp(i);
endfor  % Octave-specific (also accepts 'end')

% ---- Line Continuation ----
% MATLAB: uses ...
result = 1 + 2 + ...
         3 + 4;

% Octave: uses ... or \
result = 1 + 2 + \
         3 + 4;

% ---- Increment Operators ----
% MATLAB: no ++ or -- operators
x = x + 1;

% Octave: supports ++, --
x++;
```

### When to Use Each
```
MATLAB  →  Industry / academia with license, Simulink needed,
           toolbox-specific work, deployment, GPU computing
Octave  →  Learning MATLAB syntax for free, academic projects,
           budget constraints, Linux servers
```

> **Interview Tip:** Octave is ideal for **learning and prototyping** when you can't access MATLAB. Most interview-level MATLAB code runs identically on both. However, for production ML/DL, Python (free) has largely replaced both in industry. MATLAB remains dominant in **control systems**, **signal processing**, and **automotive** (Simulink).

---

## Question 28
- [ ] Done

**Explain the use of the MATLAB workspace and how it helps in managing variables**

### MATLAB Workspace

The workspace is MATLAB's variable storage area — it holds all variables created during the session and is visible in the Workspace panel.

### Core Operations

```matlab
% ---- Create variables ----
x = 42;
A = [1 2 3; 4 5 6];
name = 'Alice';

% ---- Inspect workspace ----
who        % list variable names
whos       % detailed list: name, size, bytes, class

%   Name    Size    Bytes   Class
%   A       2x3     48      double
%   name    1x5     10      char
%   x       1x1     8       double

% ---- Memory management ----
clear x         % remove specific variable
clear A name    % remove multiple
clear all       % remove ALL variables (use with caution)
clearvars -except x  % keep only x, clear everything else

% ---- Check existence ----
exist('x', 'var')   % returns 1 if variable exists, 0 otherwise
isempty(A)          % check if variable is empty

% ---- Save and load workspace ----
save('mydata.mat')             % save entire workspace to .mat file
save('subset.mat', 'A', 'x')   % save specific variables
load('mydata.mat')             % restore all variables
load('subset.mat', 'x')       % load specific variable

% ---- Save as text (v7.3 for large files) ----
save('bigdata.mat', '-v7.3')   % HDF5 format, supports >2GB
```

### Workspace vs. Function Scope

```matlab
% Base workspace: variables from Command Window and scripts
x = 10;
my_script;   % script shares the base workspace

% Function workspace: private to each function
function result = my_func(a)
    b = a * 2;   % 'b' exists only inside my_func
    result = b;
end
% 'b' is NOT visible in base workspace

% Sharing between scopes
global G;      % accessible from any function that declares it global
persistent P;  % retains value between function calls
assignin('base', 'var', value);  % inject variable into base workspace
evalin('base', 'expression');    % evaluate in base workspace
```

### Variable Editor
```matlab
openvar('A')   % open matrix in spreadsheet-like editor
               % allows visual inspection and editing of large matrices
```

> **Interview Tip:** The workspace is MATLAB's equivalent of Python's `globals()` dictionary but with a visual GUI. Key distinction: **scripts** share the base workspace (can accidentally overwrite variables) while **functions** have isolated scopes. Always prefer functions over scripts to avoid namespace pollution. Use `save`/`load` with `.mat` files for session persistence.

---

## Question 29
- [ ] Done

**What are MATLAB’s built-in functions for statistical analysis ?**

### MATLAB Statistics Functions

### Core Statistical Functions (Built-in)

```matlab
data = [4 8 15 16 23 42 8 15 16];

% ---- Central Tendency ----
mean(data)        % arithmetic mean = 16.33
median(data)      % median = 15
mode(data)        % most frequent value = 8 (or 15, 16 - returns first)

% ---- Dispersion ----
std(data)         % standard deviation (sample, N-1)
var(data)         % variance (sample, N-1)
std(data, 1)      % population std (N)
range(data)       % max - min = 38
iqr(data)         % interquartile range (Q3 - Q1)

% ---- Extremes & Percentiles ----
max(data)         % 42
min(data)         % 4
prctile(data, [25 50 75])  % 25th, 50th, 75th percentiles
quantile(data, 0.95)       % 95th quantile

% ---- Correlation & Covariance ----
X = randn(100, 3);
corrcoef(X)       % correlation matrix
cov(X)            % covariance matrix
corr(X(:,1), X(:,2))  % pairwise correlation

% ---- Distribution Fitting ----
histogram(data, 'Normalization', 'pdf');  % probability density
normfit(data)     % fit normal distribution (returns mu, sigma)
```

### Statistics Toolbox Functions

```matlab
% ---- Hypothesis Testing ----
[h, p] = ttest(data, 15)       % one-sample t-test (H0: mean = 15)
[h, p] = ttest2(group1, group2) % two-sample t-test
[p, tbl] = anova1(data, groups) % one-way ANOVA
[h, p] = chi2gof(data)         % chi-squared goodness of fit
[h, p] = kstest(data)          % Kolmogorov-Smirnov normality test

% ---- Regression ----
mdl = fitlm(X, y);             % linear regression
mdl = fitglm(X, y, 'Distribution', 'binomial');  % logistic regression
[b, stats] = robustfit(X, y);  % robust regression

% ---- Descriptive Summary ----
grpstats(data, groups)          % group-wise statistics
tabulate(categories)            % frequency table
bootstrap(1000, @mean, data)    % bootstrap confidence interval

% ---- Probability Distributions ----
normpdf(x, mu, sigma)   % normal PDF
normcdf(x, mu, sigma)   % normal CDF
norminv(p, mu, sigma)   % inverse CDF (quantile function)
normrnd(mu, sigma, [m,n])  % random samples
% Replace 'norm' with: t, chi2, f, bino, poiss, exp, gamma, beta
```

| Category | Functions |
|----------|----------|
| Central tendency | `mean`, `median`, `mode`, `trimmean` |
| Dispersion | `std`, `var`, `range`, `iqr`, `mad` |
| Correlation | `corrcoef`, `cov`, `corr` |
| Testing | `ttest`, `ttest2`, `anova1`, `chi2gof` |
| Distributions | `normpdf/cdf/inv/rnd`, `fitdist` |
| Regression | `fitlm`, `fitglm`, `polyfit` |

> **Interview Tip:** MATLAB distinguishes **sample** statistics (default, divides by N-1) from **population** statistics (use flag 1, divides by N). The Statistics Toolbox adds hypothesis testing, distribution fitting, and advanced regression - essential for data science workflows.

---

## Question 30
- [ ] Done

**Explain how matrix operations are performed in MATLAB**

### Matrix Operations in MATLAB

MATLAB was designed for matrix computation — matrices are first-class citizens.

### Creating Matrices

```matlab
% Manual creation
A = [1 2 3; 4 5 6; 7 8 9];

% Special matrices
I = eye(3);           % 3x3 identity
Z = zeros(3, 4);      % 3x4 zeros
O = ones(2, 3);       % 2x3 ones
R = rand(3);          % 3x3 random [0,1]
D = diag([1 2 3]);    % diagonal matrix
```

### Arithmetic Operations

```matlab
A = [1 2; 3 4];
B = [5 6; 7 8];

% Matrix operations
C = A + B;       % element-wise addition
C = A - B;       % element-wise subtraction
C = A * B;       % matrix multiplication (dot product)
C = A ^ 2;       % matrix power (A * A)

% Element-wise operations (use dot prefix)
C = A .* B;      % element-wise multiplication
C = A ./ B;      % element-wise division
C = A .^ 2;      % element-wise power
```

### Matrix Algebra

```matlab
% Transpose
A'                % conjugate transpose
A.'               % non-conjugate transpose (matters for complex)

% Inverse and determinant
inv(A)            % matrix inverse (avoid in practice)
det(A)            % determinant
pinv(A)           % pseudo-inverse (for non-square/singular)

% Solving linear systems  Ax = b
x = A \ b;        % left division (preferred over inv(A)*b)
x = b' / A';      % right division

% Decompositions
[L, U, P] = lu(A);          % LU decomposition
[Q, R] = qr(A);             % QR decomposition
[U, S, V] = svd(A);         % singular value decomposition
[V, D] = eig(A);            % eigenvalues and eigenvectors
R = chol(A);                % Cholesky (symmetric positive definite)
```

### Matrix Properties

```matlab
size(A)           % dimensions [rows, cols]
rank(A)           % matrix rank
trace(A)          % sum of diagonal elements
norm(A)           % matrix norm (default: 2-norm)
cond(A)           % condition number
null(A)           % null space
orth(A)           % column space (orthonormal basis)
```

### Indexing and Manipulation

```matlab
A(2, 3)           % element at row 2, col 3
A(1, :)           % entire first row
A(:, 2)           % entire second column
A(1:2, 2:3)       % submatrix

% Concatenation
C = [A B];         % horizontal concatenation
C = [A; B];        % vertical concatenation

% Reshaping
reshape(A, 1, [])  % flatten to row vector
A(:)               % flatten to column vector
```

| Operation | Syntax | Equivalent Math |
|-----------|--------|-----------------|
| Multiply | `A * B` | $AB$ |
| Element-wise | `A .* B` | $a_{ij} \cdot b_{ij}$ |
| Solve $Ax=b$ | `A \ b` | $x = A^{-1}b$ |
| Transpose | `A'` | $A^T$ |
| Eigenvalues | `eig(A)` | $Av = \lambda v$ |

> **Interview Tip:** The `*` vs `.*` distinction is critical in MATLAB. Matrix multiply (`*`) follows linear algebra rules ($O(n^3)$), while element-wise (`.*`) is a Hadamard product ($O(n^2)$). Always use `A \ b` instead of `inv(A) * b` for solving linear systems — it's faster and more numerically stable.

---

## Question 31
- [ ] Done

**What are element-wise operations , and how do you perform them in MATLAB ?**

### Element-Wise Operations in MATLAB

Element-wise operations apply a function independently to each corresponding element of arrays, rather than following matrix algebra rules.

### Syntax: The Dot (`.`) Prefix

```matlab
A = [1 2; 3 4];
B = [5 6; 7 8];

% ---- Matrix vs Element-wise ----
A * B        % Matrix multiplication: [19 22; 43 50]
A .* B       % Element-wise:          [5 12; 21 32]

A ^ 2        % Matrix power: A * A = [7 10; 15 22]
A .^ 2       % Element-wise:         [1 4; 9 16]

% Division
A / B        % Matrix right division: A * inv(B)
A ./ B       % Element-wise: [1/5  2/6;  3/7  4/8]

% Left division
A \ B        % Matrix left division: inv(A) * B
A .\ B       % Element-wise: [5/1  6/2;  7/3  8/4]
```

### Common Element-Wise Operations

```matlab
% Arithmetic
A + B       % addition (always element-wise)
A - B       % subtraction (always element-wise)
A .* B      % element-wise multiplication
A ./ B      % element-wise division
A .^ n      % element-wise power

% Mathematical functions (inherently element-wise)
sqrt(A)     % [1.00 1.41; 1.73 2.00]
sin(A)      % sine of each element
exp(A)      % e^x for each element
log(A)      % natural log of each element
abs(A)      % absolute value of each element

% Comparison (return logical arrays)
A > 2       % [0 0; 1 1]
A == B      % [0 0; 0 0]
A >= 3      % [0 0; 1 1]

% Logical
A > 1 & B < 7   % element-wise AND
A > 3 | B > 7   % element-wise OR
~(A > 2)        % element-wise NOT
```

### Practical ML Example

```matlab
% Sigmoid function (element-wise by nature)
sigmoid = @(z) 1 ./ (1 + exp(-z));

z = [-2 -1 0 1 2];
result = sigmoid(z);
% [0.119  0.269  0.500  0.731  0.881]

% Mean Squared Error (element-wise then aggregate)
y_true = [1 0 1 1];
y_pred = [0.9 0.2 0.8 0.7];
mse = mean((y_true - y_pred) .^ 2);  % 0.0350

% Feature normalization
X = randn(100, 5);
X_norm = (X - mean(X)) ./ std(X);   % z-score, element-wise
```

| Operator | Matrix | Element-wise |
|----------|--------|--------------|
| `*` / `.*` | Matrix product | Hadamard product |
| `^` / `.^` | Matrix power | Per-element power |
| `/` / `./` | Right division ($AB^{-1}$) | Per-element division |
| `\` / `.\` | Left division ($A^{-1}B$) | Per-element division |

> **Interview Tip:** All MATLAB math functions (`sin`, `exp`, `sqrt`) are already element-wise. The dot prefix is only needed for `*`, `^`, `/`, and `\` to distinguish from their matrix counterparts. Forgetting the dot (e.g., `A * B` instead of `A .* B`) is one of the most common MATLAB bugs.

---

## Question 32
- [ ] Done

**Explain the concept of broadcasting in MATLAB**

### Broadcasting in MATLAB

Broadcasting (called **implicit expansion** in MATLAB, introduced in R2016b) automatically expands arrays with compatible sizes for element-wise operations without explicit `repmat`.

### How It Works

```matlab
% Before R2016b: needed repmat
A = [1; 2; 3];           % 3x1 column vector
B = [10 20 30];           % 1x3 row vector

% Old way
C = repmat(A, 1, 3) + repmat(B, 3, 1);

% New way: implicit expansion (broadcasting)
C = A + B;
% C = [11 21 31;
%      12 22 32;
%      13 23 33]
```

### Broadcasting Rules

```
Dimension compatibility:
  - Same size           → operate element-wise
  - One is size 1       → expand (broadcast) to match
  - Different (not 1)   → ERROR

Examples:
  (3x1) + (1x3)  →  (3x3)   ✔  both expand
  (3x4) + (1x4)  →  (3x4)   ✔  row broadcasts
  (3x4) + (3x1)  →  (3x4)   ✔  column broadcasts
  (3x4) + (2x4)  →  ERROR   ✘  3 vs 2, neither is 1
```

### Practical Examples

```matlab
% ---- Center data (subtract column means) ----
X = rand(100, 5);         % 100 samples, 5 features
X_centered = X - mean(X); % mean(X) is 1x5, broadcasts over 100 rows

% ---- Z-score normalization ----
X_norm = (X - mean(X)) ./ std(X);   % both 1x5, broadcast to 100x5

% ---- Distance from each point to each center ----
points = rand(100, 2);    % 100 points (100x2)
centers = rand(5, 2);     % 5 centers (5x2)

% Compute pairwise distances using broadcasting
% Reshape points to (100x1x2) and centers to (1x5x2)
diff = reshape(points, [], 1, 2) - reshape(centers, 1, [], 2);
dist = sqrt(sum(diff.^2, 3));  % result: 100x5 distance matrix

% ---- Outer product ----
a = [1; 2; 3];   % 3x1
b = [4 5 6];     % 1x3
outer = a .* b;  % 3x3 outer product via broadcasting

% ---- Apply threshold per column ----
data = rand(50, 4);
thresholds = [0.3 0.5 0.7 0.2];   % 1x4
mask = data > thresholds;           % broadcasts 1x4 across 50 rows
```

### Broadcasting vs. `bsxfun`

```matlab
% bsxfun was the pre-R2016b way to broadcast
C = bsxfun(@plus, A, B);    % old way
C = A + B;                   % new way (implicit expansion)

% bsxfun still works but is no longer necessary
```

| Approach | MATLAB Version | Readability | Performance |
|----------|---------------|-------------|-------------|
| `repmat` | All | Low | Slow (copies data) |
| `bsxfun` | R2007a+ | Medium | Fast |
| Implicit expansion | R2016b+ | High | Fast |

> **Interview Tip:** MATLAB's broadcasting is equivalent to NumPy's broadcasting. The key rule: dimensions are compatible when they're equal or one of them is 1. Unlike NumPy, MATLAB added this feature relatively late (R2016b), so older code uses `bsxfun` or `repmat`. Always prefer implicit expansion for readability.

---

## Question 33
- [ ] Done

**How do you create a basic plot in MATLAB ?**

### Basic Plotting in MATLAB

```matlab
% ---- Line Plot ----
x = 0:0.1:2*pi;
y = sin(x);

figure;
plot(x, y);
title('Sine Wave');
xlabel('x (radians)');
ylabel('sin(x)');
grid on;

% ---- Customized Line Plot ----
figure;
plot(x, sin(x), 'r-', 'LineWidth', 2);    % red solid line
hold on;
plot(x, cos(x), 'b--', 'LineWidth', 2);   % blue dashed line
hold off;
title('Trigonometric Functions');
xlabel('x');
ylabel('y');
legend('sin(x)', 'cos(x)');
grid on;
```

### Line Style Specifiers

```matlab
% Format: 'ColorMarkerLineStyle'
plot(x, y, 'r-')    % red solid
plot(x, y, 'b--')   % blue dashed
plot(x, y, 'g:')    % green dotted
plot(x, y, 'ko')    % black circles
plot(x, y, 'ms-')   % magenta squares with solid line

% Colors: r g b c m y k w
% Markers: o + * . x s d ^ v > < p h
% Lines:  - -- : -.
```

### Common Plot Types

```matlab
% ---- Scatter Plot ----
scatter(x, y, 50, 'filled');  % 50 = marker size

% ---- Bar Chart ----
categories = {'A', 'B', 'C', 'D'};
values = [25 40 35 30];
bar(values);
set(gca, 'XTickLabel', categories);

% ---- Histogram ----
data = randn(1000, 1);
histogram(data, 30);

% ---- Subplot (multiple plots) ----
figure;
subplot(2, 2, 1); plot(x, sin(x));  title('sin');
subplot(2, 2, 2); plot(x, cos(x));  title('cos');
subplot(2, 2, 3); plot(x, tan(x));  title('tan');
subplot(2, 2, 4); plot(x, x.^2);    title('x^2');

% ---- 3D Plots ----
[X, Y] = meshgrid(-3:0.1:3);
Z = X.^2 + Y.^2;
figure;
surf(X, Y, Z);         % surface plot
colorbar;
title('Paraboloid');

% ---- Pie Chart ----
pie([30 25 20 15 10], {'A','B','C','D','E'});

% ---- Heatmap ----
heatmap(rand(5, 5));
```

### Plot Customization

```matlab
figure;
plot(x, y);

% Axes
xlim([0 2*pi]);      % x-axis range
ylim([-1.5 1.5]);    % y-axis range
axis equal;          % equal aspect ratio

% Text and annotations
text(pi, 0, '\leftarrow \pi', 'FontSize', 14);
annotation('arrow', [0.5 0.6], [0.5 0.7]);

% Save figure
saveas(gcf, 'plot.png');        % save as PNG
exportgraphics(gcf, 'plot.pdf');% high-quality PDF (R2020a+)
```

> **Interview Tip:** `hold on` / `hold off` controls whether subsequent plots overlay or replace. `subplot(rows, cols, index)` creates multi-panel figures. For publication-quality figures, use `exportgraphics` (R2020a+) or `print('-dpdf', 'file.pdf')`. MATLAB's plotting is more concise than Matplotlib but less flexible for complex layouts.

---

## Question 34
- [ ] Done

**How can you improve the performance of your MATLAB code ?**

### MATLAB Performance Optimization

### 1. Preallocate Arrays

```matlab
% BAD: array grows each iteration (O(n^2) due to reallocation)
result = [];
for i = 1:100000
    result = [result, i^2];  % copies entire array each time!
end

% GOOD: preallocate (O(n))
result = zeros(1, 100000);
for i = 1:100000
    result(i) = i^2;
end
% Speedup: 100-1000x for large arrays
```

### 2. Vectorize Operations

```matlab
% BAD: loop
for i = 1:length(x)
    y(i) = sin(x(i))^2 + cos(x(i))^2;
end

% GOOD: vectorized
y = sin(x).^2 + cos(x).^2;
% Speedup: 10-100x
```

### 3. Use Built-in Functions

```matlab
% BAD: manual sum
total = 0;
for i = 1:length(data)
    total = total + data(i);
end

% GOOD: built-in (C-optimized)
total = sum(data);
% Also: mean, max, min, cumsum, diff, sort
```

### 4. Logical Indexing

```matlab
% BAD: loop with if
for i = 1:length(data)
    if data(i) > threshold
        data(i) = threshold;
    end
end

% GOOD: logical indexing
data(data > threshold) = threshold;
% Or: data = min(data, threshold);
```

### 5. Avoid Unnecessary Copies

```matlab
% BAD: function modifies and returns large array
function A = process(A)
    A(1,1) = 0;  % in-place if only output = input
end

% Use sparse matrices for large sparse data
A = sparse(rows, cols, values, m, n);  % much less memory
```

### 6. Profile Your Code

```matlab
profile on;
my_function(data);
profile viewer;   % interactive GUI showing time per line

% Timing specific sections
tic;
result = expensive_operation(data);
toc;              % prints elapsed time

% More accurate
timeit(@() my_function(data));  % averages multiple runs
```

### 7. Parallel Computing

```matlab
% parfor: parallel for loop
parpool(4);  % start 4 workers
parfor i = 1:1000
    results(i) = heavy_computation(i);
end

% GPU computing
gA = gpuArray(A);
gB = gpuArray(B);
gC = gA * gB;        % runs on GPU
C = gather(gC);      % bring back to CPU
```

### 8. Data Types

```matlab
% Use appropriate data types
data_single = single(data);   % 4 bytes vs 8 bytes (double)
data_int = int32(data);       % integer operations
data_logical = logical(mask); % 1 byte per element
```

### Performance Checklist

| Technique | Typical Speedup | Effort |
|-----------|----------------|--------|
| Preallocation | 100-1000x | Low |
| Vectorization | 10-100x | Medium |
| Built-in functions | 5-50x | Low |
| Logical indexing | 5-20x | Low |
| Parallel computing | 2-8x (per core) | Medium |
| GPU computing | 10-100x | High |
| MEX (C/C++) | 2-10x | High |

> **Interview Tip:** The **Profiler** (`profile viewer`) is your first step — find the bottleneck before optimizing. The biggest wins come from **preallocation** and **vectorization**. Only use `parfor` or GPU when vectorization isn't possible, as the overhead of parallelism can actually slow down small tasks.

---

## Question 35
- [ ] Done

**Explain the use of vectorization for optimizing computations in MATLAB**

### Vectorization in MATLAB

Vectorization replaces explicit loops with array operations that execute in optimized, precompiled C/Fortran code.

### Why Vectorization Matters

```matlab
n = 1000000;
x = rand(n, 1);
y = rand(n, 1);

% ---- Loop version (SLOW) ----
tic;
z = zeros(n, 1);
for i = 1:n
    z(i) = x(i)^2 + 2*x(i)*y(i) + y(i)^2;
end
toc;  % ~0.5 seconds

% ---- Vectorized version (FAST) ----
tic;
z = x.^2 + 2.*x.*y + y.^2;
toc;  % ~0.005 seconds  (100x faster)
```

### Common Vectorization Patterns

```matlab
% ---- Replace if-else loops with logical indexing ----
% Loop:
for i = 1:n
    if x(i) > 0
        result(i) = sqrt(x(i));
    else
        result(i) = 0;
    end
end

% Vectorized:
result = zeros(size(x));
mask = x > 0;
result(mask) = sqrt(x(mask));

% ---- Replace accumulation loops with cumsum/cumprod ----
% Loop:
running_sum = zeros(n, 1);
running_sum(1) = x(1);
for i = 2:n
    running_sum(i) = running_sum(i-1) + x(i);
end

% Vectorized:
running_sum = cumsum(x);

% ---- Replace nested loops with matrix operations ----
% Loop (matrix multiply):
C = zeros(m, p);
for i = 1:m
    for j = 1:p
        for k = 1:n
            C(i,j) = C(i,j) + A(i,k) * B(k,j);
        end
    end
end

% Vectorized:
C = A * B;

% ---- Replace distance loops with broadcasting ----
% Loop:
for i = 1:n
    for j = 1:m
        dist(i,j) = sqrt(sum((X(i,:) - Y(j,:)).^2));
    end
end

% Vectorized:
dist = pdist2(X, Y);  % or use broadcasting
```

### ML-Specific Vectorization

```matlab
% ---- Gradient Descent (vectorized) ----
% Loop version:
for j = 1:num_features
    gradient(j) = 0;
    for i = 1:num_samples
        gradient(j) = gradient(j) + (h(i) - y(i)) * X(i,j);
    end
    gradient(j) = gradient(j) / num_samples;
end

% Vectorized:
gradient = (1/num_samples) * X' * (h - y);

% ---- Cost Function ----
% Loop:
cost = 0;
for i = 1:m
    cost = cost + (h(i) - y(i))^2;
end
cost = cost / (2*m);

% Vectorized:
cost = (1/(2*m)) * sum((h - y).^2);
% Or: cost = (1/(2*m)) * (h - y)' * (h - y);
```

### Vectorization Decision Guide

| Pattern | Loop | Vectorized Alternative |
|---------|------|------------------------|
| Element-wise math | `for` + indexing | Array operators `.* ./ .^` |
| Conditional assignment | `if` inside `for` | Logical indexing |
| Running totals | Accumulator loop | `cumsum`, `cumprod` |
| Matrix multiply | Triple nested loop | `*` operator |
| Pairwise distances | Double loop | `pdist2` or broadcasting |
| Aggregation | Sum in loop | `sum`, `mean`, `max` |

> **Interview Tip:** Vectorization is MATLAB's #1 performance technique. It works because: (1) operations run in compiled C, not interpreted MATLAB, (2) BLAS/LAPACK libraries use SIMD and multi-threading, (3) contiguous memory layout enables CPU cache efficiency. If you can express it as a matrix operation, do so.

---

## Question 36
- [ ] Done

**Discuss the implementation of logistic regression in MATLAB**

### Logistic Regression in MATLAB

Logistic regression models the probability of a binary outcome using the sigmoid function.

### Mathematical Foundation

$$h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))]$$

### From-Scratch Implementation

```matlab
function [theta, cost_history] = logistic_regression(X, y, alpha, num_iters)
% LOGISTIC_REGRESSION  Train logistic regression via gradient descent.
%   X: m x n feature matrix (add intercept column before calling)
%   y: m x 1 binary labels (0 or 1)
%   alpha: learning rate
%   num_iters: number of gradient descent iterations

    [m, n] = size(X);
    theta = zeros(n, 1);       % initialize parameters
    cost_history = zeros(num_iters, 1);
    
    for iter = 1:num_iters
        % Forward pass: compute predictions
        z = X * theta;
        h = sigmoid(z);         % predicted probabilities
        
        % Compute cost (cross-entropy loss)
        cost_history(iter) = -(1/m) * (y' * log(h) + (1-y)' * log(1-h));
        
        % Compute gradient
        gradient = (1/m) * X' * (h - y);
        
        % Update parameters
        theta = theta - alpha * gradient;
    end
end

function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end

function p = predict(X, theta)
    p = sigmoid(X * theta) >= 0.5;
end
```

### Complete Example

```matlab
% Generate sample data
rng(42);
X_raw = randn(200, 2);       % 200 samples, 2 features
y = double(X_raw(:,1) + X_raw(:,2) > 0);  % binary labels

% Add intercept column
X = [ones(size(X_raw, 1), 1), X_raw];

% Split train/test
train_idx = 1:160;
test_idx = 161:200;
X_train = X(train_idx, :);  y_train = y(train_idx);
X_test = X(test_idx, :);    y_test = y(test_idx);

% Train
alpha = 0.1;
num_iters = 1000;
[theta, cost_history] = logistic_regression(X_train, y_train, alpha, num_iters);

% Plot cost convergence
figure;
plot(cost_history);
title('Training Cost'); xlabel('Iteration'); ylabel('Cost');

% Evaluate
y_pred = predict(X_test, theta);
accuracy = mean(y_pred == y_test) * 100;
fprintf('Accuracy: %.1f%%\n', accuracy);

% Decision boundary
figure;
gscatter(X_raw(:,1), X_raw(:,2), y);
hold on;
x1_range = linspace(min(X_raw(:,1)), max(X_raw(:,1)), 100);
x2_boundary = -(theta(1) + theta(2)*x1_range) / theta(3);
plot(x1_range, x2_boundary, 'k-', 'LineWidth', 2);
title('Logistic Regression Decision Boundary');
legend('Class 0', 'Class 1', 'Boundary');
hold off;
```

### Using MATLAB Built-in Functions

```matlab
% Method 1: fitglm (Statistics Toolbox)
mdl = fitglm(X_raw, y, 'Distribution', 'binomial');
y_pred = predict(mdl, X_raw) >= 0.5;

% Method 2: mnrfit (multinomial logistic regression)
[B, dev, stats] = mnrfit(X_raw, y + 1);  % labels must be 1,2

% Method 3: fitclinear (for large datasets)
mdl = fitclinear(X_raw, y, 'Learner', 'logistic');

% Method 4: Using fminunc (optimization)
options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
    'GradObj', 'on', 'MaxIter', 400);
costFn = @(t) cost_function(t, X, y);
theta = fminunc(costFn, zeros(size(X,2),1), options);
```

### Regularized Logistic Regression

```matlab
function [J, grad] = cost_function_reg(theta, X, y, lambda)
    m = length(y);
    h = sigmoid(X * theta);
    
    % Cost with L2 regularization (don't regularize theta(1))
    reg_term = (lambda/(2*m)) * sum(theta(2:end).^2);
    J = -(1/m) * (y'*log(h) + (1-y)'*log(1-h)) + reg_term;
    
    % Gradient with regularization
    grad = (1/m) * X' * (h - y);
    grad(2:end) = grad(2:end) + (lambda/m) * theta(2:end);
end
```

| Approach | Pros | Cons |
|----------|------|------|
| From scratch | Full understanding, customizable | Slower, more code |
| `fitglm` | Statistical output (p-values, CI) | Requires Statistics Toolbox |
| `fminunc` | Faster convergence (quasi-Newton) | Need to define cost function |
| `fitclinear` | Scales to large data | Less statistical detail |

> **Interview Tip:** The from-scratch implementation demonstrates understanding of **gradient descent**, **cross-entropy loss**, and **vectorization**. In practice, use `fitglm` for statistical analysis (gives p-values and confidence intervals) or `fminunc` for faster convergence. Always add regularization (`lambda > 0`) to prevent overfitting.

---
