# MATLAB Interview Questions - Coding Questions

## Question 1
- [ ] Done

**Explain how to implement linear regression in MATLAB.**

### Answer

### MATLAB Code Example
```matlab
%% Linear Regression Implementation

% Generate sample data
n = 100;
X = randn(n, 2);  % 2 features
true_weights = [3; -2];
y = X * true_weights + 0.5 * randn(n, 1);  % Add noise

% Method 1: Using fitlm (Statistics Toolbox)
mdl = fitlm(X, y);
disp(mdl);
fprintf('R-squared: %.4f\n', mdl.Rsquared.Ordinary);

% Method 2: Normal equation (manual)
X_augmented = [ones(n, 1), X];  % Add bias term
weights_manual = (X_augmented' * X_augmented) \ (X_augmented' * y);
fprintf('Manual weights: %.4f, %.4f, %.4f\n', weights_manual);

% Method 3: Using regress
[b, bint, r, rint, stats] = regress(y, X_augmented);
fprintf('R-squared: %.4f\n', stats(1));

% Predictions
y_pred = predict(mdl, X);

% Visualization
figure;
subplot(1, 2, 1);
scatter(y, y_pred);
hold on;
plot([min(y), max(y)], [min(y), max(y)], 'r--');
xlabel('Actual');
ylabel('Predicted');
title('Actual vs Predicted');

subplot(1, 2, 2);
residuals = y - y_pred;
histogram(residuals, 20);
xlabel('Residual');
ylabel('Frequency');
title('Residual Distribution');
```

---

## Question 2
- [ ] Done

**Implement K-means clustering from scratch in MATLAB**

### Answer

### MATLAB Code Example
```matlab
%% K-Means Clustering

% Generate sample data (3 clusters)
rng(42);
n_per_cluster = 100;
X1 = randn(n_per_cluster, 2) + [0, 0];
X2 = randn(n_per_cluster, 2) + [5, 5];
X3 = randn(n_per_cluster, 2) + [5, 0];
X = [X1; X2; X3];

% Method 1: Built-in kmeans
k = 3;
[idx, centroids] = kmeans(X, k, 'Replicates', 10);

% Method 2: Manual implementation
function [labels, centers] = my_kmeans(X, k, max_iter)
    [n, d] = size(X);
    
    % Random initialization
    rand_idx = randperm(n, k);
    centers = X(rand_idx, :);
    
    for iter = 1:max_iter
        % Assign points to nearest centroid
        distances = pdist2(X, centers);
        [~, labels] = min(distances, [], 2);
        
        % Update centroids
        new_centers = zeros(k, d);
        for j = 1:k
            new_centers(j, :) = mean(X(labels == j, :), 1);
        end
        
        % Check convergence
        if max(vecnorm(new_centers - centers, 2, 2)) < 1e-6
            break;
        end
        centers = new_centers;
    end
end

[my_idx, my_centers] = my_kmeans(X, 3, 100);

% Visualization
figure;
gscatter(X(:,1), X(:,2), idx);
hold on;
plot(centroids(:,1), centroids(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
title('K-Means Clustering');
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Centroids');

% Elbow method for optimal k
inertia = zeros(10, 1);
for k = 1:10
    [~, ~, sumd] = kmeans(X, k, 'Replicates', 5);
    inertia(k) = sum(sumd);
end

figure;
plot(1:10, inertia, 'bo-');
xlabel('Number of Clusters');
ylabel('Inertia');
title('Elbow Method');
```

---

## Question 3
- [ ] Done

**Implement a neural network in MATLAB.**

### Answer

### MATLAB Code Example
```matlab
%% Neural Network for Classification

% Load data
load fisheriris
X = meas;
y = categorical(species);

% Split data
cv = cvpartition(y, 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv));
X_test = X(test(cv), :);
y_test = y(test(cv));

% Method 1: Using Deep Learning Toolbox
layers = [
    featureInputLayer(4, 'Name', 'input')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(32, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(3, 'Name', 'fc3')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {X_test, y_test}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

net = trainNetwork(X_train, y_train, layers, options);

% Evaluate
y_pred = classify(net, X_test);
accuracy = sum(y_pred == y_test) / length(y_test);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

% Confusion matrix
figure;
confusionchart(y_test, y_pred);
title('Confusion Matrix');

% Method 2: Using patternnet (Neural Network Toolbox)
net2 = patternnet([20, 10]);
net2 = train(net2, X_train', dummyvar(y_train)');
```

---

## Question 4
- [ ] Done

**Implement PCA for dimensionality reduction.**

### Answer

### MATLAB Code Example
```matlab
%% Principal Component Analysis

% Load data
load fisheriris
X = meas;

% Standardize data
X_std = (X - mean(X)) ./ std(X);

% Method 1: Built-in pca
[coeff, score, latent, ~, explained] = pca(X_std);

fprintf('Variance explained:\n');
for i = 1:length(explained)
    fprintf('PC%d: %.2f%% (cumulative: %.2f%%)\n', ...
        i, explained(i), sum(explained(1:i)));
end

% Method 2: Manual implementation
function [V, D, scores] = my_pca(X)
    % Center data
    X_centered = X - mean(X);
    
    % Covariance matrix
    C = cov(X_centered);
    
    % Eigendecomposition
    [V, D] = eig(C);
    
    % Sort by eigenvalue (descending)
    [D, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    
    % Project data
    scores = X_centered * V;
end

[V_manual, D_manual, scores_manual] = my_pca(X_std);

% Visualization
figure;
subplot(1, 2, 1);
gscatter(score(:,1), score(:,2), species);
xlabel('PC1');
ylabel('PC2');
title('PCA: First 2 Components');

subplot(1, 2, 2);
bar(explained);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('Scree Plot');

% Biplot
figure;
biplot(coeff(:,1:2), 'Scores', score(:,1:2), 'VarLabels', ...
    {'Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'});
title('PCA Biplot');
```

---

## Question 5
- [ ] Done

**Implement cross-validation for model evaluation.**

### Answer

### MATLAB Code Example
```matlab
%% Cross-Validation Implementation

% Load data
load fisheriris
X = meas;
y = categorical(species);

% Method 1: Built-in cross-validation
mdl = fitcsvm(X, y);
cv_mdl = crossval(mdl, 'KFold', 5);
cv_loss = kfoldLoss(cv_mdl);
fprintf('5-Fold CV Error: %.2f%%\n', cv_loss * 100);

% Method 2: Manual k-fold CV
function [mean_acc, std_acc] = manual_kfold_cv(X, y, k)
    n = size(X, 1);
    indices = crossvalind('Kfold', n, k);
    accuracies = zeros(k, 1);
    
    for i = 1:k
        % Split data
        test_idx = (indices == i);
        train_idx = ~test_idx;
        
        X_train = X(train_idx, :);
        y_train = y(train_idx);
        X_test = X(test_idx, :);
        y_test = y(test_idx);
        
        % Train and evaluate
        mdl = fitcsvm(X_train, y_train);
        y_pred = predict(mdl, X_test);
        accuracies(i) = sum(y_pred == y_test) / length(y_test);
    end
    
    mean_acc = mean(accuracies);
    std_acc = std(accuracies);
end

[mean_acc, std_acc] = manual_kfold_cv(X, y, 5);
fprintf('Manual 5-Fold CV: %.2f%% (+/- %.2f%%)\n', ...
    mean_acc * 100, std_acc * 100);

% Leave-one-out CV
cv_loo = crossval(mdl, 'Leaveout', 'on');
loo_loss = kfoldLoss(cv_loo);
fprintf('LOO CV Error: %.2f%%\n', loo_loss * 100);

% Stratified k-fold
cv_stratified = cvpartition(y, 'KFold', 5);
```

---

## Question 6
- [ ] Done

**Implement gradient descent optimization.**

### Answer

### MATLAB Code Example
```matlab
%% Gradient Descent Implementation

% Generate linear regression data
n = 100;
X = randn(n, 1);
y = 3 * X + 2 + 0.5 * randn(n, 1);

% Add bias term
X_aug = [ones(n, 1), X];

% Gradient descent for linear regression
function [weights, history] = gradient_descent(X, y, lr, epochs)
    [n, d] = size(X);
    weights = randn(d, 1);
    history = zeros(epochs, 1);
    
    for epoch = 1:epochs
        % Predictions
        y_pred = X * weights;
        
        % Compute loss (MSE)
        loss = mean((y - y_pred).^2);
        history(epoch) = loss;
        
        % Compute gradient
        gradient = (-2/n) * X' * (y - y_pred);
        
        % Update weights
        weights = weights - lr * gradient;
    end
end

% Train
learning_rate = 0.1;
epochs = 100;
[weights, loss_history] = gradient_descent(X_aug, y, learning_rate, epochs);

fprintf('Learned weights: bias=%.4f, slope=%.4f\n', weights(1), weights(2));

% Visualization
figure;
subplot(1, 2, 1);
scatter(X, y, 'b.');
hold on;
x_line = linspace(min(X), max(X), 100)';
y_line = [ones(100, 1), x_line] * weights;
plot(x_line, y_line, 'r-', 'LineWidth', 2);
xlabel('X');
ylabel('y');
title('Linear Regression Fit');

subplot(1, 2, 2);
plot(1:epochs, loss_history, 'b-');
xlabel('Epoch');
ylabel('MSE Loss');
title('Training Loss');
```

---

## Question 7
- [ ] Done

**Implement a decision tree classifier.**

### Answer

### MATLAB Code Example
```matlab
%% Decision Tree Classification

% Load data
load fisheriris
X = meas;
y = categorical(species);

% Split data
cv = cvpartition(y, 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv));
X_test = X(test(cv), :);
y_test = y(test(cv));

% Train decision tree
tree = fitctree(X_train, y_train);

% View tree
view(tree, 'Mode', 'graph');

% Predictions
y_pred = predict(tree, X_test);
accuracy = sum(y_pred == y_test) / length(y_test);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

% Cross-validation
cv_tree = crossval(tree, 'KFold', 5);
cv_loss = kfoldLoss(cv_tree);
fprintf('CV Error: %.2f%%\n', cv_loss * 100);

% Hyperparameter tuning
min_leaf_sizes = [1, 5, 10, 20, 50];
cv_errors = zeros(length(min_leaf_sizes), 1);

for i = 1:length(min_leaf_sizes)
    tree_tuned = fitctree(X_train, y_train, ...
        'MinLeafSize', min_leaf_sizes(i));
    cv_tree_tuned = crossval(tree_tuned, 'KFold', 5);
    cv_errors(i) = kfoldLoss(cv_tree_tuned);
end

figure;
plot(min_leaf_sizes, cv_errors, 'bo-');
xlabel('Min Leaf Size');
ylabel('CV Error');
title('Hyperparameter Tuning');

% Feature importance
imp = predictorImportance(tree);
figure;
bar(imp);
xticklabels({'Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'});
ylabel('Importance');
title('Feature Importance');
```

---

## Question 8
- [ ] Done

**Implement image classification with CNN.**

### Answer

### MATLAB Code Example
```matlab
%% CNN for Image Classification

% Load sample data
digitDatasetPath = fullfile(matlabroot, 'toolbox', 'nnet', ...
    'nndemos', 'nndatasets', 'DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split data
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

% Define CNN architecture
layers = [
    imageInputLayer([28 28 1], 'Name', 'input')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu3')
    dropoutLayer(0.5, 'Name', 'dropout')
    
    fullyConnectedLayer(10, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'ValidationData', imdsTest, ...
    'ValidationFrequency', 30, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Train network
net = trainNetwork(imdsTrain, layers, options);

% Evaluate
y_pred = classify(net, imdsTest);
y_test = imdsTest.Labels;
accuracy = sum(y_pred == y_test) / numel(y_test);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Confusion matrix
figure;
confusionchart(y_test, y_pred);
title('Confusion Matrix');
```

## Question 9
- [ ] Done

**Write a MATLAB function that performs matrix multiplication without using the built-in ‘*’ operator**

### Matrix Multiplication from Scratch

```matlab
function C = mat_mult(A, B)
% MAT_MULT  Multiply two matrices without using * operator.
%   C = mat_mult(A, B) returns A*B computed via nested loops.

    [m, n] = size(A);
    [n2, p] = size(B);
    
    % Dimension check
    if n ~= n2
        error('Inner dimensions must agree: A is %dx%d, B is %dx%d', m, n, n2, p);
    end
    
    C = zeros(m, p);  % preallocate result
    
    for i = 1:m
        for j = 1:p
            total = 0;
            for k = 1:n
                total = total + A(i,k) * B(k,j);  % scalar multiply only
            end
            C(i,j) = total;
        end
    end
end
```

### Usage and Verification

```matlab
A = [1 2 3; 4 5 6];
B = [7 8; 9 10; 11 12];

C_custom = mat_mult(A, B);
C_builtin = A * B;

disp(C_custom);
%     58    64
%    139   154

assert(isequal(C_custom, C_builtin), 'Results should match');
```

### Optimized Version (Row-Column Dot Product)

```matlab
function C = mat_mult_vec(A, B)
    [m, n] = size(A);
    [~, p] = size(B);
    C = zeros(m, p);
    
    for i = 1:m
        for j = 1:p
            C(i,j) = sum(A(i,:) .* B(:,j)');  % vectorized inner loop
        end
    end
end
```

> **Interview Tip:** The naive triple-loop is $O(n^3)$. MATLAB's built-in `*` uses optimized BLAS libraries (LAPACK) that exploit CPU cache, SIMD, and multi-threading — making it 100-1000x faster. This question tests understanding of the underlying algorithm, not practical usage.

---

## Question 10
- [ ] Done

**Implement a function to normalize a vector between 0 and 1**

### Min-Max Normalization

```matlab
function v_norm = normalize_minmax(v)
% NORMALIZE_MINMAX  Scale vector to [0, 1] range.
%   v_norm = normalize_minmax(v)
%   Uses formula: (v - min) / (max - min)

    v_min = min(v);
    v_max = max(v);
    
    if v_max == v_min
        v_norm = zeros(size(v));  % avoid division by zero
        warning('All elements are identical; returning zeros.');
        return;
    end
    
    v_norm = (v - v_min) / (v_max - v_min);
end
```

### Usage

```matlab
v = [10 20 30 40 50];
result = normalize_minmax(v);
disp(result);
%  0   0.25   0.50   0.75   1.00

% Works with negative values too
v2 = [-5 0 5 10];
result2 = normalize_minmax(v2);
disp(result2);
%  0   0.3333   0.6667   1.0000

% Matrix normalization (column-wise)
M = [1 2; 3 4; 5 6];
M_norm = (M - min(M)) ./ (max(M) - min(M));
disp(M_norm);
%  0   0
%  0.5 0.5
%  1   1
```

### Alternative: Z-Score Normalization

```matlab
function v_z = normalize_zscore(v)
% Z-score: (v - mean) / std
    v_z = (v - mean(v)) / std(v);
end

% MATLAB built-in (R2018a+)
v_builtin = normalize(v, 'range');    % min-max to [0,1]
v_zscore  = normalize(v, 'zscore');   % z-score
```

| Method | Formula | Range | Use Case |
|--------|---------|-------|----------|
| Min-Max | $(x - min)/(max - min)$ | [0, 1] | Neural networks, image pixels |
| Z-Score | $(x - \mu)/\sigma$ | Unbounded | Features with outliers |
| L2 Norm | $x / \|x\|_2$ | Unit sphere | Text/embedding vectors |

> **Interview Tip:** Always handle the edge case where `max == min` (constant vector). In ML, normalize **on training data** and apply the same `min`/`max` to test data to avoid data leakage.

---

## Question 11
- [ ] Done

**Write a script to import a text file and count the frequency of each unique word**

### Word Frequency Counter

```matlab
function freq = word_frequency(filename)
% WORD_FREQUENCY  Count occurrences of each unique word in a text file.
%   freq = word_frequency('file.txt') returns a containers.Map object.

    % Read entire file
    text = fileread(filename);
    
    % Preprocess: lowercase and remove punctuation
    text = lower(text);
    text = regexprep(text, '[^a-z\s]', '');  % keep only letters and spaces
    
    % Split into words
    words = strsplit(strtrim(text));
    
    % Count frequencies using containers.Map
    freq = containers.Map();
    for i = 1:length(words)
        w = words{i};
        if isempty(w)
            continue;
        end
        if freq.isKey(w)
            freq(w) = freq(w) + 1;
        else
            freq(w) = 1;
        end
    end
    
    % Display sorted results
    all_words = keys(freq);
    all_counts = cell2mat(values(freq));
    [sorted_counts, idx] = sort(all_counts, 'descend');
    
    fprintf('\n%-20s %s\n', 'Word', 'Count');
    fprintf('%s\n', repmat('-', 1, 30));
    for i = 1:min(20, length(idx))  % show top 20
        fprintf('%-20s %d\n', all_words{idx(i)}, sorted_counts(i));
    end
end
```

### Usage

```matlab
% Create sample file
fid = fopen('sample.txt', 'w');
fprintf(fid, 'The cat sat on the mat. The cat is happy.');
fclose(fid);

% Count words
freq = word_frequency('sample.txt');

% Output:
%   Word                 Count
%   ------------------------------
%   the                  3
%   cat                  2
%   sat                  1
%   on                   1
%   mat                  1
%   is                   1
%   happy                1

% Access specific word count
disp(freq('the'));  % 3
```

### Modern Approach (R2020b+)

```matlab
text = fileread('sample.txt');
doc = tokenizedDocument(lower(text));
bag = bagOfWords(doc);
tbl = topkwords(bag, 20);  % top 20 words as table
disp(tbl);
```

> **Interview Tip:** Use `containers.Map` for hash-map-like behavior in MATLAB. For NLP tasks, the Text Analytics Toolbox provides `tokenizedDocument` and `bagOfWords` for production-quality text processing with stopword removal and stemming.

---

## Question 12
- [ ] Done

**Create a MATLAB function that solves a system of linear equations**

### Solving $Ax = b$

```matlab
function x = solve_linear(A, b)
% SOLVE_LINEAR  Solve the system Ax = b.
%   x = solve_linear(A, b) solves using Gaussian elimination (backslash).

    [m, n] = size(A);
    
    % Validate inputs
    if m ~= n
        error('Coefficient matrix must be square (got %dx%d)', m, n);
    end
    if length(b) ~= m
        error('b must have %d elements', m);
    end
    
    b = b(:);  % ensure column vector
    
    % Check if system has a unique solution
    if rank(A) < n
        warning('Matrix is rank-deficient; solution may not be unique.');
    end
    
    % Method 1: Backslash operator (recommended)
    x = A \ b;
end
```

### Usage Examples

```matlab
% System:  2x + y = 5
%          x + 3y = 10
A = [2 1; 1 3];
b = [5; 10];

x = solve_linear(A, b);
fprintf('x = %.4f, y = %.4f\n', x(1), x(2));
% x = 1.0000, y = 3.0000

% Verify
disp(A * x);  % Should equal b: [5; 10]
```

### Multiple Solution Methods

```matlab
% Method 1: Backslash (uses LU, QR, or Cholesky internally)
x1 = A \ b;

% Method 2: Explicit inverse (less numerically stable)
x2 = inv(A) * b;

% Method 3: LU decomposition
[L, U, P] = lu(A);
y = L \ (P * b);
x3 = U \ y;

% Method 4: QR decomposition (better for ill-conditioned)
[Q, R] = qr(A);
x4 = R \ (Q' * b);

% Method 5: Cramer's Rule (educational only)
det_A = det(A);
x5 = zeros(size(b));
for i = 1:length(b)
    Ai = A;
    Ai(:,i) = b;
    x5(i) = det(Ai) / det_A;
end
```

| Method | Speed | Stability | When to Use |
|--------|-------|-----------|-------------|
| `A \ b` | Fast | Best | Default choice |
| `inv(A) * b` | Slow | Poor | Avoid in practice |
| LU | Fast | Good | Multiple right-hand sides |
| QR | Medium | Excellent | Ill-conditioned systems |
| Cramer | Slow | Poor | Educational only |

> **Interview Tip:** **Never use `inv(A) * b`** in production — the backslash operator `A \ b` is faster and more numerically stable. It automatically selects the best algorithm (LU for dense, Cholesky for symmetric positive definite). Check `cond(A)` to detect ill-conditioning.

---

## Question 13
- [ ] Done

**Code a MATLAB function that computes the Fibonacci sequence using recursion**

### Recursive Fibonacci

```matlab
function f = fibonacci(n)
% FIBONACCI  Compute the n-th Fibonacci number using recursion.
%   f = fibonacci(n) returns F(n) where F(0)=0, F(1)=1.

    if n < 0
        error('Input must be a non-negative integer');
    end
    
    % Base cases
    if n == 0
        f = 0;
        return;
    elseif n == 1
        f = 1;
        return;
    end
    
    % Recursive case: F(n) = F(n-1) + F(n-2)
    f = fibonacci(n-1) + fibonacci(n-2);
end
```

### Usage

```matlab
% Single value
disp(fibonacci(10));  % 55

% Generate sequence
for i = 0:10
    fprintf('F(%d) = %d\n', i, fibonacci(i));
end
% F(0)=0, F(1)=1, F(2)=1, F(3)=2, F(4)=3, F(5)=5, ..., F(10)=55
```

### Memoized Version (Efficient)

```matlab
function f = fibonacci_memo(n)
% Recursive Fibonacci with memoization using persistent variable.
    persistent cache;
    if isempty(cache)
        cache = containers.Map('KeyType', 'int32', 'ValueType', 'double');
    end
    
    if n <= 1
        f = n;
        return;
    end
    
    if cache.isKey(n)
        f = cache(n);
        return;
    end
    
    f = fibonacci_memo(n-1) + fibonacci_memo(n-2);
    cache(n) = f;
end
```

### Iterative Version (Best Performance)

```matlab
function f = fibonacci_iter(n)
% Iterative Fibonacci - O(n) time, O(1) space.
    if n <= 1
        f = n;
        return;
    end
    
    prev2 = 0; prev1 = 1;
    for i = 2:n
        f = prev1 + prev2;
        prev2 = prev1;
        prev1 = f;
    end
end
```

| Version | Time Complexity | Space | Practical Limit |
|---------|----------------|-------|-----------------|
| Naive recursive | $O(2^n)$ | $O(n)$ stack | n < 25 |
| Memoized | $O(n)$ | $O(n)$ cache | n < 1000 |
| Iterative | $O(n)$ | $O(1)$ | n < 1476 (double overflow) |

> **Interview Tip:** The naive recursive approach has exponential time complexity because it recomputes the same subproblems (e.g., `F(5)` computes `F(3)` twice). Memoization converts it to $O(n)$ — a classic **dynamic programming** example. Always mention this tradeoff in interviews.

---

## Question 14
- [ ] Done

**Develop a MATLAB script to plot a histogram of random numbers following a normal distribution**

### Normal Distribution Histogram

```matlab
% Generate random numbers from normal distribution
mu = 0;       % mean
sigma = 1;    % standard deviation
n = 10000;    % number of samples

data = mu + sigma * randn(n, 1);   % N(mu, sigma)
% Or equivalently: data = normrnd(mu, sigma, [n, 1]);

% ---- Basic Histogram ----
figure;
histogram(data, 50);  % 50 bins
title('Histogram of Normal Distribution');
xlabel('Value');
ylabel('Frequency');

% ---- Normalized Histogram with PDF Overlay ----
figure;
histogram(data, 50, 'Normalization', 'pdf');  % probability density
hold on;

% Overlay theoretical PDF
x = linspace(min(data), max(data), 200);
y = normpdf(x, mu, sigma);
plot(x, y, 'r-', 'LineWidth', 2);

title(sprintf('Normal Distribution (\\mu=%.1f, \\sigma=%.1f, n=%d)', mu, sigma, n));
xlabel('Value');
ylabel('Probability Density');
legend('Histogram', 'Theoretical PDF');
grid on;
hold off;

% ---- Customized Appearance ----
figure;
h = histogram(data, 50);
h.FaceColor = [0.2 0.6 0.8];     % custom color
h.EdgeColor = 'white';            % white edges
h.FaceAlpha = 0.7;                % transparency

% Add statistics annotation
text_str = sprintf('Mean: %.3f\nStd: %.3f\nN: %d', mean(data), std(data), n);
annotation('textbox', [0.7 0.7 0.2 0.2], 'String', text_str, ...
    'FontSize', 10, 'BackgroundColor', 'white');

title('Customized Normal Distribution Histogram');
xlabel('Value');
ylabel('Count');
```

### Multiple Distributions

```matlab
figure;
data1 = normrnd(0, 1, [5000, 1]);
data2 = normrnd(3, 0.5, [5000, 1]);

histogram(data1, 40, 'Normalization', 'pdf', 'FaceAlpha', 0.5);
hold on;
histogram(data2, 40, 'Normalization', 'pdf', 'FaceAlpha', 0.5);
legend('N(0,1)', 'N(3, 0.5)');
title('Comparing Two Normal Distributions');
hold off;
```

> **Interview Tip:** Use `'Normalization', 'pdf'` to compare distributions with different sample sizes. `randn` generates standard normal $N(0,1)$; transform with `mu + sigma*randn(...)` for arbitrary $N(\mu, \sigma)$. The `normrnd` function from Statistics Toolbox is a convenient alternative.

---

## Question 15
- [ ] Done

**Write a MATLAB program that detects edges in an image using the Sobel operator**

### Sobel Edge Detection

```matlab
function edges = sobel_edge_detect(img_path)
% SOBEL_EDGE_DETECT  Detect edges using the Sobel operator.
%   edges = sobel_edge_detect('image.jpg') returns binary edge map.

    % Read and convert to grayscale
    img = imread(img_path);
    if size(img, 3) == 3
        gray = rgb2gray(img);
    else
        gray = img;
    end
    gray = double(gray);  % convert to double for computation
    
    % Define Sobel kernels
    Gx = [-1  0  1;
          -2  0  2;
          -1  0  1];  % horizontal edges
      
    Gy = [-1 -2 -1;
           0  0  0;
           1  2  1];  % vertical edges
    
    % Apply convolution
    grad_x = conv2(gray, Gx, 'same');
    grad_y = conv2(gray, Gy, 'same');
    
    % Compute gradient magnitude
    magnitude = sqrt(grad_x.^2 + grad_y.^2);
    
    % Normalize to [0, 255]
    magnitude = magnitude / max(magnitude(:)) * 255;
    
    % Apply threshold
    threshold = 0.3 * max(magnitude(:));  % 30% of max
    edges = magnitude > threshold;
    
    % Display results
    figure;
    subplot(2, 2, 1); imshow(uint8(gray));    title('Original (Grayscale)');
    subplot(2, 2, 2); imshow(uint8(abs(grad_x)));  title('Horizontal Edges (Gx)');
    subplot(2, 2, 3); imshow(uint8(abs(grad_y)));  title('Vertical Edges (Gy)');
    subplot(2, 2, 4); imshow(edges);          title('Edge Detection Result');
end
```

### Usage

```matlab
% Run on an image
edges = sobel_edge_detect('cameraman.tif');  % built-in MATLAB image

% Using MATLAB built-in (for comparison)
img = imread('cameraman.tif');
gray = im2double(rgb2gray(img));
edge_builtin = edge(gray, 'Sobel');  % built-in Sobel
edge_canny = edge(gray, 'Canny');    % Canny (more advanced)

% Compare methods
figure;
subplot(1,3,1); imshow(gray);         title('Original');
subplot(1,3,2); imshow(edge_builtin); title('Sobel (built-in)');
subplot(1,3,3); imshow(edge_canny);   title('Canny');
```

### How the Sobel Operator Works
```
  Gx (horizontal):        Gy (vertical):
  [-1  0 +1]              [-1 -2 -1]
  [-2  0 +2]              [ 0  0  0]
  [-1  0 +1]              [+1 +2 +1]

  Gradient magnitude = sqrt(Gx^2 + Gy^2)
  Edge direction     = atan2(Gy, Gx)
```

| Edge Detector | Smoothing | Noise Sensitivity | Localization |
|--------------|-----------|-------------------|--------------|
| Sobel | 3x3 Gaussian | Moderate | Good |
| Prewitt | None | High | Good |
| Canny | Multi-scale | Low | Excellent |
| Laplacian | None | Very High | Poor |

> **Interview Tip:** Sobel combines **smoothing** (Gaussian-weighted) with **differentiation** in one 3x3 kernel, making it more noise-resistant than Prewitt. In practice, **Canny** is preferred because it adds non-maximum suppression and hysteresis thresholding. For deep learning-based edge detection, look into HED (Holistically-Nested Edge Detection).

---
