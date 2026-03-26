# Statistics — Statistical Computing and Analysis

## Overview

The `stats` module provides comprehensive statistical functionality, from descriptive statistics to hypothesis testing and regression analysis. It's built on top of NDArray and designed for scientific data analysis.

## Module Structure

```zig
const stats = zuda.stats;

// Descriptive statistics
const desc = stats.descriptive;

// Probability distributions
const dist = stats.distributions;

// Hypothesis testing
const test_stat = stats.hypothesis;

// Correlation and regression
const corr = stats.correlation;
```

## Descriptive Statistics

### Basic Measures

**Mean (Average)**
```zig
const data = [_]f64{1.0, 2.0, 3.0, 4.0, 5.0};
var arr = try NDArray(f64, 1).fromSlice(allocator, &data, .row_major);
defer arr.deinit();

const mean_val = try desc.mean(f64, arr.data, 0);  // 3.0
```

**Median**
```zig
const median_val = try desc.median(f64, data, allocator);  // 3.0

// For even-length arrays, returns average of two middle values
const even_data = [_]f64{1, 2, 3, 4};
const median_even = try desc.median(f64, even_data, allocator);  // 2.5
```

**Mode**
```zig
const mode_data = [_]f64{1, 2, 2, 3, 3, 3, 4};
const mode_val = try desc.mode(f64, mode_data, allocator);  // 3.0
```

### Variance and Standard Deviation

**Population Statistics (ddof=0)**
```zig
const variance = try desc.variance(f64, data, 0);  // Population variance
const std_dev = try desc.stdDev(f64, data, 0);     // sqrt(variance)
```

**Sample Statistics (ddof=1)**
```zig
const sample_var = try desc.variance(f64, data, 1);  // Sample variance (Bessel's correction)
const sample_std = try desc.stdDev(f64, data, 1);    // Sample standard deviation
```

**Delta Degrees of Freedom (ddof)**:
- `ddof=0`: Population statistics (divide by N)
- `ddof=1`: Sample statistics (divide by N-1, unbiased estimator)

### Quantiles and Percentiles

```zig
const q25 = try desc.quantile(f64, data, 0.25, allocator);  // 25th percentile (Q1)
const q50 = try desc.quantile(f64, data, 0.50, allocator);  // Median (Q2)
const q75 = try desc.quantile(f64, data, 0.75, allocator);  // 75th percentile (Q3)

// Interquartile range
const iqr = q75 - q25;
```

### Shape Measures

**Skewness** (asymmetry)
```zig
const skew = try desc.skewness(f64, data, allocator);

// Interpretation:
// skew ≈ 0: Symmetric
// skew > 0: Right-skewed (tail to right)
// skew < 0: Left-skewed (tail to left)
```

**Kurtosis** (tail heaviness)
```zig
const kurt = try desc.kurtosis(f64, data, allocator);

// Interpretation (excess kurtosis):
// kurt ≈ 0: Normal distribution
// kurt > 0: Heavy tails (leptokurtic)
// kurt < 0: Light tails (platykurtic)
```

### Summary Statistics

```zig
var summary = try desc.describe(f64, data, allocator);
defer summary.deinit(allocator);

std.debug.print("Count: {}\n", .{summary.count});
std.debug.print("Mean: {d:.4}\n", .{summary.mean});
std.debug.print("Std: {d:.4}\n", .{summary.std});
std.debug.print("Min: {d:.4}\n", .{summary.min});
std.debug.print("25%: {d:.4}\n", .{summary.q25});
std.debug.print("50%: {d:.4}\n", .{summary.median});
std.debug.print("75%: {d:.4}\n", .{summary.q75});
std.debug.print("Max: {d:.4}\n", .{summary.max});
```

## Probability Distributions

### Normal (Gaussian) Distribution

```zig
const normal = dist.Normal(f64);

// Create N(μ=0, σ²=1) — standard normal
var rng = std.rand.DefaultPrng.init(42);
var std_normal = normal.init(0.0, 1.0);

// Sample random values
const sample = std_normal.sample(&rng.random());

// Probability density function
const pdf_val = std_normal.pdf(1.5);

// Cumulative distribution function
const cdf_val = std_normal.cdf(1.96);  // ≈ 0.975

// Quantile function (inverse CDF)
const quantile = try std_normal.quantile(0.975);  // ≈ 1.96

// Statistics
const mean_val = std_normal.mean();      // 0.0
const var_val = std_normal.variance();   // 1.0
```

**Use cases**: Heights, IQ scores, measurement errors, central limit theorem

### Uniform Distribution

```zig
const uniform = dist.Uniform(f64);

// U[a, b] — uniform on [a, b]
var unif = uniform.init(0.0, 10.0);

const sample = unif.sample(&rng.random());  // Random value in [0, 10]
const pdf_val = unif.pdf(5.0);              // 1/(b-a) = 0.1
const cdf_val = unif.cdf(7.5);              // (x-a)/(b-a) = 0.75
```

**Use cases**: Random sampling, Monte Carlo simulation, initial guesses

### Exponential Distribution

```zig
const exponential = dist.Exponential(f64);

// Exp(λ) — rate parameter λ
var exp_dist = exponential.init(0.5);  // Mean = 1/λ = 2.0

const sample = exp_dist.sample(&rng.random());
const pdf_val = exp_dist.pdf(2.0);     // λ * exp(-λx)
const cdf_val = exp_dist.cdf(2.0);     // 1 - exp(-λx)
```

**Use cases**: Time between events, survival analysis, queueing theory

### Gamma Distribution

```zig
const gamma = dist.Gamma(f64);

// Gamma(α, β) — shape α, rate β
var gamma_dist = gamma.init(2.0, 0.5);  // Mean = α/β = 4.0

const sample = gamma_dist.sample(&rng.random());
const pdf_val = gamma_dist.pdf(3.0);
const mean_val = gamma_dist.mean();      // α/β
const var_val = gamma_dist.variance();   // α/β²
```

**Use cases**: Waiting times, rainfall models, Bayesian priors

### Beta Distribution

```zig
const beta = dist.Beta(f64);

// Beta(α, β) — support on [0, 1]
var beta_dist = beta.init(2.0, 5.0);

const sample = beta_dist.sample(&rng.random());  // In [0, 1]
const pdf_val = beta_dist.pdf(0.3);
const mean_val = beta_dist.mean();      // α/(α+β)
```

**Use cases**: Proportions, probabilities, Bayesian conjugate priors

### Binomial Distribution

```zig
const binomial = dist.Binomial(f64);

// Binomial(n, p) — n trials, success probability p
var binom = binomial.init(10, 0.5);

const sample = binom.sample(&rng.random());  // Count of successes
const pmf_val = binom.pmf(5);                // P(X = 5)
const cdf_val = binom.cdf(7);                // P(X ≤ 7)
const mean_val = binom.mean();               // n*p = 5.0
```

**Use cases**: Coin flips, A/B testing, quality control

### Poisson Distribution

```zig
const poisson = dist.Poisson(f64);

// Poisson(λ) — rate parameter λ
var pois = poisson.init(3.5);

const sample = pois.sample(&rng.random());  // Count of events
const pmf_val = pois.pmf(4);                // P(X = 4)
const mean_val = pois.mean();               // λ = 3.5
```

**Use cases**: Event counts, arrivals, rare events

### Chi-Squared Distribution

```zig
const chi2 = dist.ChiSquared(f64);

// χ²(k) — k degrees of freedom
var chi2_dist = chi2.init(5);

const sample = chi2_dist.sample(&rng.random());
const pdf_val = chi2_dist.pdf(7.0);
const cdf_val = chi2_dist.cdf(11.07);  // For significance testing
```

**Use cases**: Goodness-of-fit tests, variance testing, independence tests

### Student's t-Distribution

```zig
const student_t = dist.StudentT(f64);

// t(ν) — ν degrees of freedom
var t_dist = student_t.init(10);

const sample = t_dist.sample(&rng.random());
const pdf_val = t_dist.pdf(2.0);
const cdf_val = t_dist.cdf(2.228);  // For t-test critical values
```

**Use cases**: t-tests, small sample inference, confidence intervals

### F-Distribution

```zig
const f_dist_type = dist.FDistribution(f64);

// F(d₁, d₂) — numerator and denominator degrees of freedom
var f_dist = f_dist_type.init(5, 20);

const sample = f_dist.sample(&rng.random());
const pdf_val = f_dist.pdf(2.5);
const cdf_val = f_dist.cdf(2.71);  // For ANOVA F-tests
```

**Use cases**: ANOVA, variance ratio tests, regression F-tests

## Hypothesis Testing

### t-Test (One-Sample)

Test if sample mean differs from population mean.

```zig
const sample = [_]f64{2.1, 2.5, 2.3, 2.8, 2.4};
const pop_mean = 2.0;

const result = try test_stat.ttest_1samp(f64, &sample, pop_mean, allocator);

std.debug.print("t-statistic: {d:.4}\n", .{result.statistic});
std.debug.print("p-value: {d:.4}\n", .{result.p_value});
std.debug.print("Reject H0: {}\n", .{result.p_value < 0.05});

// H0: μ = pop_mean
// H1: μ ≠ pop_mean
```

### t-Test (Two-Sample Independent)

Test if two independent samples have different means.

```zig
const sample1 = [_]f64{2.1, 2.5, 2.3, 2.8, 2.4};
const sample2 = [_]f64{2.8, 3.1, 3.0, 3.3, 2.9};

const result = try test_stat.ttest_ind(f64, &sample1, &sample2, true, allocator);
// true = assume equal variances (pooled t-test)

std.debug.print("t-statistic: {d:.4}\n", .{result.statistic});
std.debug.print("p-value: {d:.4}\n", .{result.p_value});

// H0: μ₁ = μ₂
// H1: μ₁ ≠ μ₂
```

### t-Test (Paired)

Test if paired samples have different means.

```zig
const before = [_]f64{120, 130, 125, 135, 128};
const after = [_]f64{115, 125, 122, 130, 124};

const result = try test_stat.ttest_rel(f64, &before, &after, allocator);

// Equivalent to one-sample t-test on differences
```

**Use cases**: Before-after studies, matched pairs

### Chi-Squared Test (Goodness-of-Fit)

Test if observed frequencies match expected distribution.

```zig
const observed = [_]f64{18, 25, 20, 17};
const expected = [_]f64{20, 20, 20, 20};  // Uniform distribution

const result = try test_stat.chisquare(f64, &observed, &expected);

std.debug.print("χ² statistic: {d:.4}\n", .{result.statistic});
std.debug.print("p-value: {d:.4}\n", .{result.p_value});

// H0: Data follows expected distribution
```

### Chi-Squared Test (Independence)

Test if two categorical variables are independent.

```zig
// Contingency table: rows = treatment, cols = outcome
const observed = [_][_]f64{
    .{10, 20, 30},  // Treatment A
    .{15, 25, 10},  // Treatment B
};

const result = try test_stat.chi2_contingency(f64, &observed, allocator);

// H0: Variables are independent
```

### ANOVA (One-Way)

Test if multiple groups have the same mean.

```zig
const group1 = [_]f64{2.1, 2.5, 2.3};
const group2 = [_]f64{2.8, 3.1, 3.0};
const group3 = [_]f64{3.5, 3.8, 3.6};

const groups = [_][]const f64{ &group1, &group2, &group3 };
const result = try test_stat.f_oneway(f64, &groups, allocator);

std.debug.print("F-statistic: {d:.4}\n", .{result.statistic});
std.debug.print("p-value: {d:.4}\n", .{result.p_value});

// H0: μ₁ = μ₂ = μ₃
// H1: At least one mean differs
```

### Kolmogorov-Smirnov Test

Test if sample comes from specified distribution.

```zig
const sample = [_]f64{0.1, 0.5, 0.8, 1.2, 1.5};
const normal_dist = dist.Normal(f64).init(1.0, 0.5);

const result = try test_stat.kstest(f64, &sample, normal_dist, allocator);

std.debug.print("KS statistic: {d:.4}\n", .{result.statistic});
std.debug.print("p-value: {d:.4}\n", .{result.p_value});

// H0: Sample follows specified distribution
```

## Correlation and Regression

### Pearson Correlation

Measures linear relationship between variables.

```zig
const x = [_]f64{1, 2, 3, 4, 5};
const y = [_]f64{2, 4, 5, 4, 5};

const r = try corr.pearsonr(f64, &x, &y);

std.debug.print("Correlation: {d:.4}\n", .{r.correlation});
std.debug.print("p-value: {d:.4}\n", .{r.p_value});

// r ∈ [-1, 1]
// r = 1: Perfect positive correlation
// r = 0: No linear correlation
// r = -1: Perfect negative correlation
```

### Spearman Correlation

Measures monotonic relationship (rank-based).

```zig
const r_s = try corr.spearmanr(f64, &x, &y, allocator);

// More robust to outliers than Pearson
// Works for non-linear monotonic relationships
```

### Linear Regression

Fit y = β₀ + β₁x + ε

```zig
const x = [_]f64{1, 2, 3, 4, 5};
const y = [_]f64{2.1, 3.9, 6.2, 7.8, 10.1};

var result = try corr.linregress(f64, &x, &y, allocator);
defer result.deinit(allocator);

std.debug.print("Slope: {d:.4}\n", .{result.slope});
std.debug.print("Intercept: {d:.4}\n", .{result.intercept});
std.debug.print("R²: {d:.4}\n", .{result.r_squared});
std.debug.print("p-value: {d:.4}\n", .{result.p_value});

// Predict new value
const x_new = 6.0;
const y_pred = result.intercept + result.slope * x_new;
```

### Multiple Linear Regression

Fit y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

```zig
// X is m×n design matrix (m samples, n features)
var X = try NDArray(f64, 2).fromSlice(allocator, &[_]f64{
    1, 2,  // Sample 1: [x₁, x₂]
    2, 3,  // Sample 2
    3, 5,  // Sample 3
    4, 6,  // Sample 4
}, .row_major);
X.shape = .{4, 2};
defer X.deinit();

var y = try NDArray(f64, 1).fromSlice(allocator, &[_]f64{3, 5, 7, 9}, .row_major);
defer y.deinit();

// Use least squares solver from linalg
const linalg = zuda.linalg;
var beta = try linalg.solve.lstsq(f64, X, y, allocator);
defer beta.deinit();

// beta[0] = β₁, beta[1] = β₂
```

## Common Patterns

### Standardization (Z-Score)

```zig
const mean_val = try desc.mean(f64, data, 0);
const std_val = try desc.stdDev(f64, data, 0);

var standardized = std.ArrayList(f64).init(allocator);
defer standardized.deinit();

for (data) |x| {
    const z = (x - mean_val) / std_val;
    try standardized.append(z);
}

// Standardized data has mean=0, std=1
```

### Outlier Detection (IQR Method)

```zig
const q25 = try desc.quantile(f64, data, 0.25, allocator);
const q75 = try desc.quantile(f64, data, 0.75, allocator);
const iqr = q75 - q25;

const lower_bound = q25 - 1.5 * iqr;
const upper_bound = q75 + 1.5 * iqr;

var outliers = std.ArrayList(f64).init(allocator);
defer outliers.deinit();

for (data) |x| {
    if (x < lower_bound or x > upper_bound) {
        try outliers.append(x);
    }
}
```

### Bootstrapping

```zig
const n_bootstrap = 1000;
var bootstrap_means = try std.ArrayList(f64).initCapacity(allocator, n_bootstrap);
defer bootstrap_means.deinit();

var rng = std.rand.DefaultPrng.init(42);

for (0..n_bootstrap) |_| {
    // Resample with replacement
    var resample = try std.ArrayList(f64).initCapacity(allocator, data.len);
    defer resample.deinit();

    for (0..data.len) |_| {
        const idx = rng.random().intRangeAtMost(usize, 0, data.len - 1);
        try resample.append(data[idx]);
    }

    const boot_mean = try desc.mean(f64, resample.items, 0);
    try bootstrap_means.append(boot_mean);
}

// Compute 95% confidence interval
const ci_lower = try desc.quantile(f64, bootstrap_means.items, 0.025, allocator);
const ci_upper = try desc.quantile(f64, bootstrap_means.items, 0.975, allocator);
```

## Performance Tips

1. **Use appropriate ddof**: `ddof=0` for populations, `ddof=1` for samples
2. **Preallocate for resampling**: Use `ArrayList.initCapacity` for bootstrap/permutation tests
3. **Cache statistics**: Store mean/std if used multiple times
4. **Use NDArray for large datasets**: More efficient than slices for multi-dimensional data
5. **Choose right test**: Parametric tests (t-test) assume normality; use non-parametric (rank-based) for skewed data

## Error Handling

```zig
const mean_val = desc.mean(f64, data, 0) catch |err| switch (err) {
    error.EmptyArray => {
        std.debug.print("Cannot compute mean of empty array\n", .{});
        return;
    },
    else => return err,
};

const corr_result = corr.pearsonr(f64, x, y) catch |err| switch (err) {
    error.DimensionMismatch => {
        std.debug.print("Arrays must have same length\n", .{});
        return;
    },
    error.InsufficientData => {
        std.debug.print("Need at least 2 data points\n", .{});
        return;
    },
    else => return err,
};
```

## See Also

- [NDArray Guide](ndarray.md) — Array operations for statistical data
- [Linear Algebra Guide](linalg.md) — Matrix operations for multivariate statistics
- [NumPy Compatibility](../NUMPY_COMPATIBILITY.md) — NumPy stats → zuda stats mapping
