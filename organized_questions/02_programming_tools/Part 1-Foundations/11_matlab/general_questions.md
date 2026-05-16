# MATLAB Interview Questions - General Questions

## Question 1
- [ ] Done

**How do you read and write data in MATLAB?**

### Answer

### File I/O Functions

| Function | File Type |
|----------|-----------|
| `readtable` | CSV, Excel, text |
| `load` | MAT files, text |
| `imread` | Images |
| `audioread` | Audio |
| `readmatrix` | Numeric data |

### MATLAB Code Example
```matlab
% CSV file
data = readtable('data.csv');
writetable(data, 'output.csv');

% Excel file
data = readtable('data.xlsx', 'Sheet', 'Sheet1');
writetable(data, 'output.xlsx');

% MAT file (MATLAB native)
save('data.mat', 'data');
loaded = load('data.mat');

% Text file
matrix = readmatrix('data.txt');
writematrix(matrix, 'output.txt');

% Image
img = imread('image.png');
imwrite(img, 'output.png');

% Custom text file
fid = fopen('custom.txt', 'w');
fprintf(fid, 'Value: %d\n', 42);
fclose(fid);

% Read line by line
fid = fopen('data.txt', 'r');
while ~feof(fid)
    line = fgetl(fid);
    disp(line);
end
fclose(fid);
```

---

## Question 2
- [ ] Done

**How do you handle missing data in MATLAB?**

### Answer

### Missing Data Functions

| Function | Purpose |
|----------|---------|
| `ismissing` | Detect missing values |
| `rmmissing` | Remove missing values |
| `fillmissing` | Fill missing values |
| `standardizeMissing` | Define missing indicators |

### MATLAB Code Example
```matlab
% Create data with missing values
data = [1 2 NaN 4 5 NaN 7];

% Detect missing
missing_idx = isnan(data);
fprintf('Missing count: %d\n', sum(missing_idx));

% Remove missing
clean_data = rmmissing(data);

% Fill missing values
filled_mean = fillmissing(data, 'constant', mean(data, 'omitnan'));
filled_interp = fillmissing(data, 'linear');
filled_prev = fillmissing(data, 'previous');

disp('Original:'); disp(data);
disp('Fill with mean:'); disp(filled_mean);
disp('Linear interpolation:'); disp(filled_interp);

% Table with missing values
T = table([1;2;NaN;4], {'A';'B';'';'D'}, ...
    'VariableNames', {'Num', 'Cat'});

% Define empty string as missing
T = standardizeMissing(T, '');

% Fill different columns differently
T.Num = fillmissing(T.Num, 'constant', 0);
T.Cat = fillmissing(T.Cat, 'constant', 'Unknown');
```

---

## Question 3
- [ ] Done

**How do you visualize data in MATLAB?**

### Answer

### Common Plot Functions

| Function | Use Case |
|----------|----------|
| `plot` | Line plots |
| `scatter` | Scatter plots |
| `histogram` | Distributions |
| `bar` | Bar charts |
| `heatmap` | Correlation matrices |
| `subplot` | Multiple plots |

### MATLAB Code Example
```matlab
% Generate data
x = linspace(0, 2*pi, 100);
y = sin(x);

% Line plot
figure;
plot(x, y, 'b-', 'LineWidth', 2);
xlabel('x');
ylabel('sin(x)');
title('Sine Wave');
grid on;

% Multiple plots
figure;
subplot(2, 2, 1);
plot(x, sin(x));
title('Sine');

subplot(2, 2, 2);
plot(x, cos(x));
title('Cosine');

subplot(2, 2, 3);
histogram(randn(1000, 1), 30);
title('Normal Distribution');

subplot(2, 2, 4);
scatter(randn(100, 1), randn(100, 1));
title('Scatter Plot');

% Heatmap for correlation
data = randn(100, 5);
corr_matrix = corrcoef(data);
figure;
heatmap(corr_matrix);
title('Correlation Heatmap');

% Save figure
saveas(gcf, 'figure.png');
```

---

## Question 4
- [ ] Done

**How do you debug MATLAB code?**

### Answer

### Debugging Tools

| Tool | Purpose |
|------|---------|
| `dbstop` | Set breakpoint |
| `dbcont` | Continue execution |
| `dbstep` | Step through code |
| `dbstack` | View call stack |
| `disp/fprintf` | Print values |

### MATLAB Code Example
```matlab
% Set breakpoint on error
dbstop if error

% Set breakpoint at line
dbstop in my_function at 10

% Conditional breakpoint
dbstop in my_function at 15 if x > 10

% Debug function
function result = buggy_function(data)
    % Add assertions for validation
    assert(~isempty(data), 'Data cannot be empty');
    assert(isnumeric(data), 'Data must be numeric');
    
    % Debug output
    fprintf('Input size: %s\n', mat2str(size(data)));
    
    % Try-catch for error handling
    try
        result = process_data(data);
    catch ME
        fprintf('Error: %s\n', ME.message);
        fprintf('Stack trace:\n');
        disp(ME.stack);
        result = [];
    end
end

% Use keyboard to pause execution
function result = interactive_debug(x)
    intermediate = x * 2;
    keyboard;  % Pauses here for inspection
    result = intermediate + 1;
end

% Clear breakpoints
dbclear all
```

---

## Question 5
- [ ] Done

**How do you optimize MATLAB code performance?**

### Answer

### Optimization Strategies

| Strategy | Impact |
|----------|--------|
| Vectorization | High |
| Preallocation | High |
| Built-in functions | Medium |
| Profiling | Diagnostic |
| Data types | Medium |

### MATLAB Code Example
```matlab
% Profile code
profile on
slow_function();
profile viewer

% Timing
tic
result = my_function(data);
elapsed = toc;
fprintf('Time: %.4f s\n', elapsed);

% Compare implementations
function benchmark()
    n = 1000000;
    x = randn(n, 1);
    
    % Method 1: Loop
    tic
    y1 = zeros(n, 1);
    for i = 1:n
        y1(i) = sqrt(abs(x(i)));
    end
    t1 = toc;
    
    % Method 2: Vectorized
    tic
    y2 = sqrt(abs(x));
    t2 = toc;
    
    % Method 3: arrayfun
    tic
    y3 = arrayfun(@(v) sqrt(abs(v)), x);
    t3 = toc;
    
    fprintf('Loop: %.4f s\n', t1);
    fprintf('Vectorized: %.4f s\n', t2);
    fprintf('arrayfun: %.4f s\n', t3);
end

% Use appropriate data types
x_single = single(randn(1000));  % Less memory than double
x_int = int32(1:1000);  % Integer type

% Sparse matrices for sparse data
A_sparse = sparse(eye(10000));  % Much less memory
```


---

## Question 6
- [ ] Done

**How do MATLAB scripts differ from functions?**

**Answer:**

| Feature | Script (.m) | Function (.m) |
|---------|------------|---------------|
| **Workspace** | Shares base workspace | Has own workspace |
| **Input/Output** | No arguments | Accepts input/output args |
| **Variables** | All variables persist | Local variables only |
| **Reusability** | Run as-is | Callable with parameters |
| **File naming** | Any name | Filename must match function name |

```matlab
% === Script: my_script.m ===
x = 1:10;
y = x.^2;
plot(x, y);
title('Square Function');
% Variables x, y remain in workspace after execution

% === Function: compute_stats.m ===
function [avg, sd] = compute_stats(data)
    % Function with inputs and outputs
    avg = mean(data);
    sd = std(data);
    temp = data * 2;  % temp is LOCAL, not visible outside
end

% Call the function:
[m, s] = compute_stats([1 2 3 4 5]);

% === Local Functions (within a script, R2016b+) ===
x = 1:10;
y = double_values(x);

function result = double_values(input)
    result = input * 2;
end
```

> **Interview Tip:** Functions are preferred for reusable, modular code. Scripts are useful for quick prototyping and interactive analysis. Since R2016b, scripts can contain local functions at the end of the file.

---

## Question 7
- [ ] Done

**How do you create 3D plots in MATLAB?**

**Answer:**

```matlab
% === 1. 3D Line Plot ===
t = 0:0.01:10*pi;
x = sin(t); y = cos(t); z = t;
plot3(x, y, z);
xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D Helix'); grid on;

% === 2. Surface Plot ===
[X, Y] = meshgrid(-5:0.25:5);
Z = sin(sqrt(X.^2 + Y.^2));
figure;
surf(X, Y, Z);          % Surface with color
colormap(jet);
colorbar;
title('Surface Plot');

% === 3. Mesh Plot ===
figure;
mesh(X, Y, Z);           % Wireframe
title('Mesh Plot');

% === 4. Contour Plot (2D projection of 3D) ===
figure;
contour(X, Y, Z, 20);   % 20 contour levels
contourf(X, Y, Z, 20);  % Filled contours

% === 5. Scatter 3D ===
figure;
x = randn(100, 1);
y = randn(100, 1);
z = randn(100, 1);
scatter3(x, y, z, 50, z, 'filled');  % Size 50, color by z
colorbar;

% === 6. Interactive Rotation ===
rotate3d on;             % Enable mouse rotation

% === 7. Multiple Subplots ===
figure;
subplot(1,2,1); surf(X,Y,Z); title('Surface');
subplot(1,2,2); mesh(X,Y,Z); title('Mesh');
```

| Function | Type | Use Case |
|----------|------|----------|
| `plot3` | Line | 3D trajectories |
| `surf` | Surface | Continuous 3D data |
| `mesh` | Wireframe | See-through surface |
| `contour` | 2D projection | Topographic maps |
| `scatter3` | Points | 3D data points |
| `bar3` | Bars | 3D bar charts |

> **Interview Tip:** Use `meshgrid` to create coordinate grids for surface plots. `surf` fills faces with color while `mesh` shows only wireframe. Add `colorbar` and `rotate3d on` for better interactivity.

---

## Question 8
- [ ] Done

**How do you deal with time series data in MATLAB?**

**Answer:**

```matlab
% === 1. Creating Time Series ===
dates = datetime(2020,1,1):days(1):datetime(2020,12,31);
values = randn(length(dates), 1) + sin((1:length(dates))'/30);

% Timetable (recommended data structure)
tt = timetable(dates', values, 'VariableNames', {'Price'});

% === 2. Resampling ===
weekly = retime(tt, 'weekly', 'mean');     % Weekly average
monthly = retime(tt, 'monthly', 'lastvalue'); % Monthly last value

% === 3. Moving Statistics ===
tt.MA7 = movmean(tt.Price, 7);             % 7-day moving average
tt.MA30 = movmean(tt.Price, 30);           % 30-day moving average
tt.Volatility = movstd(tt.Price, 20);      % 20-day rolling std

% === 4. Decomposition ===
[trend, seasonal, residual] = trenddecomp(tt.Price);

% === 5. Visualization ===
figure;
plot(tt.dates, tt.Price, 'b-', 'LineWidth', 0.5);
hold on;
plot(tt.dates, tt.MA30, 'r-', 'LineWidth', 2);
legend('Daily', '30-Day MA');
title('Time Series with Moving Average');

% === 6. Forecasting with ARIMA ===
model = arima(2, 1, 1);                   % ARIMA(2,1,1)
estModel = estimate(model, tt.Price);
[forecast, ~] = forecast(estModel, 30, 'Y0', tt.Price);

% === 7. Stationarity Test ===
[h, pValue] = adftest(tt.Price);           % Augmented Dickey-Fuller
fprintf('ADF test p-value: %.4f (stationary: %d)\n', pValue, h);
```

| Function | Purpose |
|----------|--------|
| `timetable` | Time-indexed data structure |
| `retime` | Resample at different frequency |
| `movmean/movstd` | Rolling statistics |
| `trenddecomp` | Trend-seasonal decomposition |
| `arima` | ARIMA modeling |
| `adftest` | Stationarity testing |

> **Interview Tip:** MATLAB's `timetable` is the preferred structure for time series. It handles irregular sampling, missing data, and time zone conversions natively.

---

## Question 9
- [ ] Done

**How do loops work in MATLAB , and when would you use them?**

**Answer:**

```matlab
% === 1. for Loop ===
for i = 1:10
    fprintf('Iteration %d\n', i);
end

% Loop over array elements
fruits = {'apple', 'banana', 'cherry'};
for i = 1:length(fruits)
    fprintf('Fruit: %s\n', fruits{i});
end

% Loop over matrix columns
A = [1 2 3; 4 5 6; 7 8 9];
for col = A
    disp(col);  % Each iteration gives a column vector
end

% === 2. while Loop ===
count = 1;
while count <= 5
    fprintf('Count: %d\n', count);
    count = count + 1;
end

% Convergence check
tol = 1e-6;
error = 1;
while error > tol
    % iterative computation
    error = error * 0.5;
end

% === 3. Loop Control ===
for i = 1:100
    if i == 5
        continue;   % Skip this iteration
    end
    if i > 10
        break;      % Exit loop entirely
    end
end

% === 4. Vectorization (Preferred Over Loops!) ===
% Slow (loop):
result = zeros(1, 1000);
for i = 1:1000
    result(i) = sin(i) * cos(i);
end

% Fast (vectorized):
x = 1:1000;
result = sin(x) .* cos(x);  % ~10-100x faster
```

| Loop Type | When to Use |
|-----------|------------|
| `for` | Known number of iterations |
| `while` | Unknown iterations, convergence checks |
| **Vectorization** | Always prefer when possible |

> **Interview Tip:** In MATLAB, loops are slower than vectorized operations due to interpreter overhead. Always try to vectorize first. Pre-allocate arrays with `zeros()` if you must use loops to avoid dynamic memory allocation.

---

## Question 10
- [ ] Done

**Demonstrate how to use conditional statements in MATLAB.**

**Answer:**

```matlab
% === 1. if-elseif-else ===
score = 85;
if score >= 90
    grade = 'A';
elseif score >= 80
    grade = 'B';
elseif score >= 70
    grade = 'C';
else
    grade = 'F';
end
fprintf('Grade: %s\n', grade);

% === 2. switch-case ===
day = 'Monday';
switch day
    case {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'}
        type = 'Weekday';
    case {'Saturday', 'Sunday'}
        type = 'Weekend';
    otherwise
        type = 'Unknown';
end

% === 3. Logical Operators ===
x = 15;
if x > 10 && x < 20      % AND
    disp('Between 10 and 20');
end

if x < 5 || x > 10       % OR
    disp('Outside 5-10 range');
end

if ~(x == 0)              % NOT
    disp('Not zero');
end

% === 4. Vectorized Conditional ===
data = [1 -2 3 -4 5];

% Using logical indexing (no loop needed!)
positive = data(data > 0);          % [1, 3, 5]
data(data < 0) = 0;                 % Replace negatives: [1, 0, 3, 0, 5]

% === 5. Ternary-like (no ternary operator in MATLAB) ===
result = (x > 10) * 100 + (x <= 10) * 0;  % Workaround
```

| Construct | Use Case |
|-----------|----------|
| `if-elseif-else` | General branching |
| `switch-case` | Multiple discrete values |
| Logical indexing | Vectorized conditions |
| `&&` / `||` | Short-circuit (scalars) |
| `&` / `|` | Element-wise (arrays) |

> **Interview Tip:** Use `&&`/`||` for scalar conditions (short-circuit) and `&`/`|` for element-wise array operations. MATLAB's `switch-case` doesn't need `break` statements—it only executes the matching case.

---

## Question 11
- [ ] Done

**How do you create and use MATLAB cell arrays?**

**Answer:**

Cell arrays store data of different types and sizes in a single container.

```matlab
% === Creating Cell Arrays ===
C = {1, 'hello', [1 2 3], true};          % Mixed types
C2 = cell(3, 2);                           % Pre-allocate 3x2 empty cell array

% === Accessing Elements ===
% Curly braces {} access CONTENTS; parentheses () access CELLS
C{1}             % Returns: 1 (the number)
C(1)             % Returns: {1} (a 1x1 cell)
C{3}(2)          % Returns: 2 (second element of the array in cell 3)

% === Assigning Values ===
C{5} = struct('name', 'John', 'age', 30);
C{2} = 'world';                            % Overwrite

% === Cell Array of Strings ===
names = {'Alice', 'Bob', 'Charlie'};
for i = 1:length(names)
    fprintf('Name: %s\n', names{i});
end

% === Nested Cell Arrays ===
nested = {{1, 2}, {'a', 'b', 'c'}, {true}};
nested{2}{3}     % Returns: 'c'

% === Useful Operations ===
cellfun(@length, names)       % Apply function: [5, 3, 7]
cellfun(@ischar, C)           % Check types: [0, 1, 0, 0, 0]
cell2mat({1 2; 3 4})          % Convert to matrix: [1 2; 3 4]
cell2table(C)                 % Convert to table

% === Cell Arrays vs Arrays ===
% Array: all same type, same size
% Cell array: any type, any size per cell
% String array (modern): use string() instead of cell for text
```

| Operation | Syntax |
|-----------|--------|
| Create | `{val1, val2, ...}` or `cell(m,n)` |
| Access content | `C{i}` |
| Access cell | `C(i)` |
| Apply function | `cellfun(@func, C)` |
| To matrix | `cell2mat(C)` |

> **Interview Tip:** Use `{}` to access cell contents and `()` to access the cell itself. Modern MATLAB prefers `string` arrays over cell arrays of char vectors for text processing.

---

## Question 12
- [ ] Done

**How to import data from a CSV file into MATLAB?**

**Answer:**

```matlab
% === 1. readtable (Recommended) ===
T = readtable('data.csv');
% Returns a table with column headers as variable names
head(T)                    % First 8 rows
T.Properties.VariableNames % Column names
T.Age                      % Access column by name

% With options
opts = detectImportOptions('data.csv');
opts.VariableTypes{3} = 'categorical';  % Set column 3 as categorical
opts = setvaropts(opts, 'Date', 'InputFormat', 'yyyy-MM-dd');
T = readtable('data.csv', opts);

% === 2. readmatrix (Numeric data only) ===
M = readmatrix('numbers.csv');          % Returns numeric matrix

% === 3. csvread (Legacy) ===
data = csvread('data.csv', 1, 0);       % Skip 1 header row

% === 4. readcell (Mixed types) ===
C = readcell('mixed_data.csv');

% === 5. Import Tool (Interactive) ===
% uiimport('data.csv')  % Opens GUI

% === 6. Large Files ===
% Read in chunks using datastore
ds = tabularTextDatastore('large_data.csv');
ds.ReadSize = 10000;                    % Read 10k rows at a time
while hasdata(ds)
    chunk = read(ds);
    % Process chunk
end

% === 7. Writing CSV ===
writetable(T, 'output.csv');
writematrix(M, 'output.csv');
```

| Function | Input Type | Returns | Best For |
|----------|-----------|---------|----------|
| `readtable` | Any CSV | Table | General use (recommended) |
| `readmatrix` | Numeric CSV | Matrix | Pure numeric data |
| `readcell` | Mixed CSV | Cell array | Mixed types |
| `datastore` | Large CSV | Iterator | Files too large for memory |

> **Interview Tip:** `readtable` is the modern standard—it auto-detects headers, data types, and delimiters. Use `detectImportOptions` to customize parsing before reading.

---

## Question 13
- [ ] Done

**What toolbox does MATLAB offer for machine learning , and what features does it include?**

**Answer:**

MATLAB offers **Statistics and Machine Learning Toolbox** and **Deep Learning Toolbox** as primary ML tools.

| Toolbox | Key Features |
|---------|-------------|
| **Statistics & ML Toolbox** | Classification, regression, clustering, dimensionality reduction, feature selection |
| **Deep Learning Toolbox** | CNNs, RNNs, LSTMs, GANs, transfer learning, training from scratch |
| **Computer Vision Toolbox** | Object detection, image segmentation, feature extraction |
| **Text Analytics Toolbox** | NLP, sentiment analysis, topic modeling |
| **Reinforcement Learning Toolbox** | RL agents, environments, training |

```matlab
% === Statistics & Machine Learning Toolbox ===
% Classification
model = fitcsvm(X_train, y_train);          % SVM
model = fitctree(X_train, y_train);         % Decision Tree
model = fitcensemble(X_train, y_train);     % Ensemble (Random Forest)
model = fitcknn(X_train, y_train, 'NumNeighbors', 5);  % KNN

% Regression
model = fitlm(X, y);                        % Linear Regression
model = fitrensemble(X, y);                  % Ensemble Regression

% Clustering
idx = kmeans(X, 3);                          % K-Means
idx = dbscan(X, epsilon, minPts);            % DBSCAN

% Dimensionality Reduction
[coeff, score] = pca(X);                     % PCA
Y = tsne(X);                                 % t-SNE

% Cross-Validation
cv = cvpartition(y, 'KFold', 5);
cvModel = crossval(model, 'CVPartition', cv);
cvLoss = kfoldLoss(cvModel);

% === Deep Learning Toolbox ===
layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];
options = trainingOptions('adam', 'MaxEpochs', 10);
net = trainNetwork(X_train, y_train, layers, options);

% === Classification Learner App (GUI) ===
% classificationLearner  % Interactive ML training
```

> **Interview Tip:** MATLAB's ML toolboxes are strong for prototyping with built-in apps (Classification Learner, Deep Network Designer). For production, models can be exported to C/C++ code or deployed via MATLAB Production Server.

---

## Question 14
- [ ] Done

**How do neural networks work in MATLAB?**

**Answer:**

MATLAB's Deep Learning Toolbox provides tools for building, training, and deploying neural networks.

```matlab
% === 1. Define Network Architecture ===
layers = [
    imageInputLayer([28 28 1], 'Name', 'input')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    
    fullyConnectedLayer(128, 'Name', 'fc1')
    dropoutLayer(0.5, 'Name', 'drop')
    fullyConnectedLayer(10, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% === 2. Training Options ===
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 5, ...
    'LearnRateDropFactor', 0.5, ...
    'ValidationData', {X_val, y_val}, ...
    'ValidationFrequency', 30, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

% === 3. Train ===
net = trainNetwork(X_train, y_train, layers, options);

% === 4. Predict & Evaluate ===
y_pred = classify(net, X_test);
accuracy = mean(y_pred == y_test);
confusionchart(y_test, y_pred);

% === 5. Transfer Learning ===
net = resnet50;                             % Load pre-trained
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
lgraph = addLayers(lgraph, [
    fullyConnectedLayer(10, 'Name', 'new_fc')
    softmaxLayer('Name', 'new_softmax')
    classificationLayer('Name', 'new_output')
]);
lgraph = connectLayers(lgraph, 'pool5', 'new_fc');

% === 6. Deep Network Designer (GUI) ===
% deepNetworkDesigner  % Visual drag-and-drop
```

| Component | MATLAB Function |
|-----------|----------------|
| CNN | `convolution2dLayer`, `maxPooling2dLayer` |
| RNN/LSTM | `lstmLayer`, `gruLayer` |
| Training | `trainNetwork`, `trainingOptions` |
| Prediction | `classify`, `predict` |
| Transfer Learning | `resnet50`, `vgg16`, `googlenet` |

> **Interview Tip:** MATLAB provides real-time training visualization plots by default. Its Deep Network Designer app allows visual network design—useful for prototyping before coding.

---

## Question 15
- [ ] Done

**What functions does MATLAB provide for cross-validation?**

**Answer:**

```matlab
% === 1. cvpartition: Create Cross-Validation Partitions ===
cv = cvpartition(y, 'KFold', 5);           % 5-fold CV
cv = cvpartition(y, 'HoldOut', 0.2);       % 80/20 split
cv = cvpartition(y, 'LeaveOut');           % Leave-one-out
cv = cvpartition(y, 'Resubstitution');     % Train = Test (not recommended)

% Access folds
for i = 1:cv.NumTestSets
    trainIdx = training(cv, i);
    testIdx = test(cv, i);
    X_train = X(trainIdx, :);
    X_test = X(testIdx, :);
end

% === 2. crossval: General Cross-Validation ===
model = fitcsvm(X, y);
cvModel = crossval(model, 'KFold', 10);
cvLoss = kfoldLoss(cvModel);               % Classification error
cvAcc = 1 - cvLoss;                        % Accuracy

% Predictions from CV
y_pred = kfoldPredict(cvModel);

% === 3. Cross-Validate with Custom Metric ===
fun = @(X_train, y_train, X_test, y_test) ...
    sum(predict(fitctree(X_train, y_train), X_test) ~= y_test);
errors = crossval(fun, X, y, 'KFold', 5);
meanError = mean(errors ./ cv.TestSize');

% === 4. Hyperparameter Tuning with CV ===
model = fitcsvm(X, y, ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', ...
    struct('KFold', 5, 'MaxObjectiveEvaluations', 30));

% === 5. Stratified CV (default for classification) ===
cv = cvpartition(y, 'KFold', 5);  % Automatically stratified for categorical y
```

| Function | Purpose |
|----------|--------|
| `cvpartition` | Create CV partition |
| `crossval` | Cross-validate a model |
| `kfoldLoss` | CV loss (error rate) |
| `kfoldPredict` | CV predictions |
| `kfoldMargin` | CV margin (SVM) |

> **Interview Tip:** MATLAB's `cvpartition` with categorical labels automatically creates stratified folds. Use `'OptimizeHyperparameters'` with `'auto'` for Bayesian hyperparameter optimization with built-in CV.

---

## Question 16
- [ ] Done

**How is parallel computing supported in MATLAB?**

**Answer:**

MATLAB's **Parallel Computing Toolbox** enables multi-core, GPU, and cluster computing.

```matlab
% === 1. parfor: Parallel for Loop ===
parpool(4);                    % Start pool with 4 workers
results = zeros(1, 1000);
parfor i = 1:1000
    results(i) = heavy_computation(i);  % Runs on 4 cores
end
delete(gcp);                   % Close pool

% === 2. parfeval: Async Execution ===
pool = parpool(4);
f1 = parfeval(pool, @process_data, 1, data1);  % Non-blocking
f2 = parfeval(pool, @process_data, 1, data2);
result1 = fetchOutputs(f1);                     % Get result when ready
result2 = fetchOutputs(f2);

% === 3. GPU Computing ===
A = gpuArray(rand(1000));      % Move to GPU
B = gpuArray(rand(1000));
C = A * B;                     % Computed on GPU
result = gather(C);            % Move back to CPU

% === 4. spmd: Single Program Multiple Data ===
spmd
    localData = codistributed(bigMatrix);  % Distribute across workers
    localResult = eig(localData);          % Each worker processes its portion
end

% === 5. Parallel-Aware Functions ===
% Many built-in functions auto-parallelize:
options = trainingOptions('adam', 'ExecutionEnvironment', 'multi-gpu');
net = trainNetwork(data, layers, options);  % Multi-GPU training

% === 6. MapReduce for Big Data ===
ds = datastore('big_data.csv');
result = mapreduce(ds, @mapFun, @reduceFun);
```

| Method | Use Case | Scaling |
|--------|----------|--------|
| `parfor` | Independent iterations | Multi-core |
| `parfeval` | Async tasks | Multi-core |
| `gpuArray` | Matrix/DL computations | GPU |
| `spmd` | Distributed data | Cluster |
| `mapreduce` | Big data processing | Cluster |

> **Interview Tip:** `parfor` cannot have loop-carried dependencies—each iteration must be independent. GPU computing gives the biggest speedup for large matrix operations and deep learning.

---

## Question 17
- [ ] Done

**How do you call a C/C++ library function from MATLAB?**

**Answer:**

```matlab
% === Method 1: MEX Files (Most Common) ===
% Create a C file: myfunction.c
% #include "mex.h"
% void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
%     double *input = mxGetPr(prhs[0]);
%     int n = mxGetNumberOfElements(prhs[0]);
%     plhs[0] = mxCreateDoubleMatrix(1, n, mxREAL);
%     double *output = mxGetPr(plhs[0]);
%     for (int i = 0; i < n; i++) output[i] = input[i] * 2;
% }

% Compile and call
mex myfunction.c                    % Compile to MEX
result = myfunction([1 2 3 4]);     % Call like MATLAB function

% === Method 2: loadlibrary (Shared Libraries) ===
loadlibrary('mylib.dll', 'mylib.h');        % Load DLL
result = calllib('mylib', 'add', 5, 3);     % Call function
unloadlibrary('mylib');

% === Method 3: coder.ceval (within MATLAB Coder) ===
function y = my_wrapper(x)
    y = 0;
    coder.ceval('c_function', x, coder.wref(y));
end

% === Method 4: System Call ===
[status, output] = system('./my_c_program arg1 arg2');

% === Setup Compiler ===
mex -setup c          % Select C compiler
mex -setup c++        % Select C++ compiler
```

| Method | Complexity | Speed | Use Case |
|--------|-----------|-------|----------|
| MEX files | Medium | Fastest | Performance-critical functions |
| `loadlibrary` | Low | Fast | Existing shared libraries |
| `coder.ceval` | High | Fastest | Code generation workflows |
| `system()` | Low | Slow (process overhead) | External executables |

> **Interview Tip:** MEX files are the standard way to integrate C/C++ with MATLAB. They run in-process (no overhead) and access MATLAB data directly. Use `mex -setup` to configure the compiler first.

---

## Question 18
- [ ] Done

**How can you run Python scripts within MATLAB?**

**Answer:**

```matlab
% === 1. Direct Python Function Calls ===
% Set Python version (one-time)
pyenv('Version', 'C:\Python39\python.exe');

% Call Python built-in functions
result = py.math.sqrt(144);           % Returns 12.0
pi_val = py.math.pi;                   % 3.14159...

% === 2. Call Python Libraries ===
np = py.importlib.import_module('numpy');
arr = np.array({1, 2, 3, 4, 5});
mean_val = double(np.mean(arr));

% Pandas
pd = py.importlib.import_module('pandas');
df = pd.read_csv('data.csv');
column = double(df{'column_name'}.values);  % Convert to MATLAB array

% Scikit-learn
sklearn = py.importlib.import_module('sklearn.linear_model');
model = sklearn.LinearRegression();

% === 3. Run Python Scripts ===
pyrunfile('my_script.py');             % Run entire script
result = pyrunfile('compute.py', 'output_var', x=42);

% === 4. Type Conversion ===
% MATLAB -> Python
py_list = py.list({1, 2, 3});
py_dict = py.dict(pyargs('key1', 1, 'key2', 2));
py_str = py.str('hello');

% Python -> MATLAB
mat_array = double(py_array);          % Numeric
mat_str = string(py_string);           % String
mat_cell = cell(py_list);              % List -> Cell

% === 5. Custom Python Module ===
% Add to path:
insert(py.sys.path, int32(0), 'C:\my_python_modules');
my_mod = py.importlib.import_module('my_module');
result = my_mod.my_function(42);
```

| Data Type | MATLAB to Python | Python to MATLAB |
|-----------|-----------------|------------------|
| Number | Auto | `double()` |
| String | `py.str()` | `string()` / `char()` |
| Array | `py.numpy.array()` | `double()` |
| List | `py.list()` | `cell()` |
| Dict | `py.dict()` | `struct()` |

> **Interview Tip:** MATLAB's Python integration is ideal for leveraging Python ML libraries (scikit-learn, TensorFlow) while using MATLAB for visualization and signal processing. Data conversion between types is the main challenge.

---

## Question 19
- [ ] Done

**How do you perform time-series analysis in MATLAB?**

**Answer:**

```matlab
% === 1. Create & Visualize ===
dates = datetime(2020,1,1):days(1):datetime(2023,12,31);
values = cumsum(randn(length(dates),1)) + 100;
tt = timetable(dates', values, 'VariableNames', {'StockPrice'});
plot(tt.Time, tt.StockPrice); title('Stock Price');

% === 2. Preprocessing ===
tt = rmmissing(tt);                        % Remove NaN
tt = fillmissing(tt, 'linear');            % Interpolate missing
tt_smooth = smoothdata(tt, 'movmean', 30); % 30-day smoothing

% === 3. Decomposition ===
[trend, seasonal, residual] = trenddecomp(tt.StockPrice);
figure;
subplot(3,1,1); plot(trend); title('Trend');
subplot(3,1,2); plot(seasonal); title('Seasonal');
subplot(3,1,3); plot(residual); title('Residual');

% === 4. Stationarity Tests ===
[h, pValue] = adftest(tt.StockPrice);      % ADF test
[h, pValue] = kpsstest(tt.StockPrice);     % KPSS test

% === 5. ARIMA Modeling ===
model = arima(2, 1, 1);                    % ARIMA(p,d,q)
estModel = estimate(model, tt.StockPrice);
[yForecast, yMSE] = forecast(estModel, 30, 'Y0', tt.StockPrice);

% Plot forecast with confidence
upper = yForecast + 1.96*sqrt(yMSE);
lower = yForecast - 1.96*sqrt(yMSE);
figure;
hold on;
plot(tt.StockPrice); plot(length(tt.StockPrice)+(1:30), yForecast, 'r');
fill([1:30, 30:-1:1]'+length(tt.StockPrice), [upper; flipud(lower)], 'r', 'FaceAlpha', 0.2);

% === 6. Autocorrelation ===
figure;
subplot(2,1,1); autocorr(tt.StockPrice, 50);   title('ACF');
subplot(2,1,2); parcorr(tt.StockPrice, 50);     title('PACF');

% === 7. Spectral Analysis ===
[pxx, f] = periodogram(tt.StockPrice);
plot(f, 10*log10(pxx)); title('Power Spectrum');
```

| Analysis | Function |
|----------|----------|
| Decomposition | `trenddecomp` |
| Stationarity | `adftest`, `kpsstest` |
| ARIMA | `arima`, `estimate`, `forecast` |
| Autocorrelation | `autocorr`, `parcorr` |
| Spectral | `periodogram`, `pwelch` |

> **Interview Tip:** For time-series analysis in MATLAB, start with visualization, check stationarity (ADF test), examine ACF/PACF to determine ARIMA orders, then model and forecast. MATLAB's Econometrics Toolbox adds GARCH, VAR, and cointegration tests.

---

## Question 20
- [ ] Done

**How do you train a Long Short-Term Memory (LSTM) network in MATLAB?**

**Answer:**

```matlab
% === 1. Prepare Sequence Data ===
% Each sample is a sequence of variable length
% X: cell array of sequences, Y: categorical labels
X_train = {randn(5, 100), randn(5, 80), randn(5, 120)};  % 5 features
y_train = categorical({'A', 'B', 'A'});

% === 2. Define LSTM Architecture ===
numFeatures = 5;
numClasses = 2;
numHiddenUnits = 100;

layers = [
    sequenceInputLayer(numFeatures, 'Name', 'input')
    lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'lstm1')
    dropoutLayer(0.3, 'Name', 'drop')
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% === 3. Training Options ===
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'GradientThreshold', 1, ...         % Gradient clipping
    'SequenceLength', 'longest', ...    % Padding strategy
    'Shuffle', 'every-epoch', ...
    'ValidationData', {X_val, y_val}, ...
    'Plots', 'training-progress');

% === 4. Train ===
net = trainNetwork(X_train, y_train, layers, options);

% === 5. Predict ===
y_pred = classify(net, X_test);
accuracy = mean(y_pred == y_test);

% === 6. Sequence-to-Sequence (e.g., time series regression) ===
layers_s2s = [
    sequenceInputLayer(1)
    lstmLayer(128, 'OutputMode', 'sequence')  % Output at each timestep
    fullyConnectedLayer(1)
    regressionLayer
];

% === 7. Bidirectional LSTM ===
layers_bi = [
    sequenceInputLayer(numFeatures)
    bilstmLayer(100, 'OutputMode', 'last')    % Bidirectional
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];
```

| Parameter | Options |
|-----------|--------|
| `OutputMode` | `'last'` (classification), `'sequence'` (seq-to-seq) |
| `SequenceLength` | `'longest'`, `'shortest'`, fixed number |
| `GradientThreshold` | Clips gradients (prevents exploding) |
| Layer types | `lstmLayer`, `bilstmLayer`, `gruLayer` |

> **Interview Tip:** Use `'OutputMode', 'last'` for classification (one label per sequence) and `'OutputMode', 'sequence'` for regression/forecasting (prediction at each timestep). Always use gradient clipping (`GradientThreshold`) with LSTMs.

---

## Question 21
- [ ] Done

**Present a strategy for using MATLAB to analyze genomic data.**

**Answer:**

MATLAB's **Bioinformatics Toolbox** provides comprehensive tools for genomic analysis.

```matlab
% === 1. Read Genomic Data ===
fastaData = fastaread('genome.fasta');          % FASTA format
fastqData = fastqread('reads.fastq');           % FASTQ (with quality)
seq = fastaData.Sequence;

% === 2. Sequence Analysis ===
baseCount = basecount(seq);                     % A, T, G, C counts
gcContent = (baseCount.G + baseCount.C) / length(seq);
compSeq = seqcomplement(seq);                   % Complement
revComp = seqrcomplement(seq);                  % Reverse complement
aa = nt2aa(seq);                                % Translate to protein

% === 3. Sequence Alignment ===
[score, alignment] = nwalign(seq1, seq2);       % Global alignment
[score, alignment] = swalign(seq1, seq2);       % Local alignment

% === 4. Gene Expression (Microarray/RNA-Seq) ===
% Read expression data
data = readtable('expression.csv');
exprMatrix = table2array(data(:, 2:end));
geneNames = data.GeneName;

% Normalize
normData = quantilenorm(exprMatrix);            % Quantile normalization

% Differential expression
[h, p] = ttest2(group1, group2);               % t-test per gene
adj_p = mafdr(p, 'BHFDR', true);               % FDR correction
significant = geneNames(adj_p < 0.05);

% === 5. Clustering & Visualization ===
cg = clustergram(normData, 'RowLabels', geneNames, ...
    'ColumnLabels', sampleNames, 'Colormap', redbluecmap);

% PCA
[coeff, score] = pca(normData');
scatter(score(:,1), score(:,2)); title('PCA of Samples');

% === 6. Machine Learning on Genomic Data ===
model = fitcensemble(exprMatrix, labels, 'Method', 'Bag');
importance = model.predictorImportance();
[~, idx] = sort(importance, 'descend');
topGenes = geneNames(idx(1:20));               % Top 20 important genes
```

| Step | MATLAB Tool |
|------|------------|
| Read sequences | `fastaread`, `fastqread` |
| Alignment | `nwalign`, `swalign`, `multialign` |
| Expression analysis | `quantilenorm`, `mafdr` |
| Visualization | `clustergram`, `heatmap` |
| ML classification | `fitcensemble`, `fitcsvm` |
| Pathway analysis | `goannotread`, `getgeodata` |

> **Interview Tip:** Genomic data analysis in MATLAB follows: data import -> quality control -> normalization -> differential expression -> pathway enrichment -> ML classification. The Bioinformatics Toolbox handles most standard bioinformatics file formats.

---

## Question 22
- [ ] Done

**How can you utilize MATLAB’s App Designer to create interactive applications featuring machine learning models ?**

**Answer:**

App Designer is MATLAB's visual environment for creating professional desktop apps with interactive UI components.

```matlab
% === App Designer Structure ===
% File: MLApp.mlapp (created via appdesigner command)

classdef MLApp < matlab.apps.AppBase
    properties (Access = public)
        UIFigure       matlab.ui.Figure
        LoadButton     matlab.ui.control.Button
        PredictButton  matlab.ui.control.Button
        ResultLabel    matlab.ui.control.Label
        DataTable      matlab.ui.control.Table
        PlotAxes       matlab.ui.control.UIAxes
        TrainedModel               % Stores ML model
    end
    
    methods (Access = private)
        % Load Data
        function LoadButtonPushed(app, ~)
            [file, path] = uigetfile('*.csv');
            if file
                data = readtable(fullfile(path, file));
                app.DataTable.Data = data;
                scatter(app.PlotAxes, data{:,1}, data{:,2});
                title(app.PlotAxes, 'Data Preview');
            end
        end
        
        % Train & Predict
        function PredictButtonPushed(app, ~)
            data = app.DataTable.Data;
            X = data{:, 1:end-1};
            y = data{:, end};
            
            % Train model
            app.TrainedModel = fitctree(X, y);
            
            % Cross-validate
            cvModel = crossval(app.TrainedModel);
            accuracy = 1 - kfoldLoss(cvModel);
            
            app.ResultLabel.Text = sprintf('Accuracy: %.2f%%', accuracy*100);
            
            % Plot confusion matrix
            y_pred = kfoldPredict(cvModel);
            confusionchart(app.PlotAxes, y, y_pred);
        end
    end
end

% === Launch App Designer ===
% >> appdesigner   % Opens visual editor
% Drag-and-drop: Buttons, Axes, Tables, Sliders, Dropdowns
% Wire callbacks to functions

% === Deploy as Standalone ===
% Use MATLAB Compiler:
% >> mcc -m MLApp.mlapp   % Creates standalone .exe
```

| Component | Use Case |
|-----------|----------|
| `UIAxes` | Plots, confusion matrices |
| `Button` | Trigger training/prediction |
| `Table` | Display data, results |
| `Slider` | Adjust hyperparameters |
| `DropDown` | Select model type |
| `Gauge/Lamp` | Show accuracy/status |

> **Interview Tip:** App Designer apps can be compiled into standalone executables (no MATLAB license needed) using MATLAB Compiler. This enables deploying ML models to end users who don't have MATLAB.
