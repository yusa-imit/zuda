# Statistics Module API Reference

Comprehensive statistical analysis library for the zuda v2.0 scientific computing platform. Provides descriptive statistics, probability distributions, hypothesis testing, correlation analysis, and regression modeling.

**Module Path**: `zuda.stats`

**Import Examples**:
```zig
const stats = @import("zuda").stats;

// Specific imports
const descriptive = stats.descriptive;
const Normal = stats.distributions.Normal;
const hypothesis = stats.hypothesis;
const correlation = stats.correlation;
```

---

## Table of Contents

1. [Descriptive Statistics](#descriptive-statistics)
2. [Probability Distributions](#probability-distributions)
3. [Hypothesis Testing](#hypothesis-testing)
4. [Correlation and Covariance](#correlation-and-covariance)
5. [Regression and Fitting](#regression-and-fitting)
6. [Common Patterns and Examples](#common-patterns-and-examples)

---

## Descriptive Statistics

Module: `zuda.stats.descriptive`

Univariate summary statistics for analyzing data distributions. All functions accept `NDArray(T, 1)` where `T` is a numeric type (f32, f64, i32, i64, etc.).

### Mean

Compute arithmetic mean (average) of a 1D array.

**Signature**:
```zig
pub fn mean(comptime T: type, data: NDArray(T, 1)) T
```

**Description**:
Calculates the arithmetic mean as `(sum of all elements) / count`. For integer types, the result is truncated (not rounded).

**Parameters**:
- `T`: Numeric type (f32, f64, i32, i64, etc.)
- `data`: 1D NDArray of type T

**Returns**: Mean value of type T

**Complexity**:
- **Time**: O(n) where n = data.size()
- **Space**: O(1)

**Example**:
```zig
const allocator = std.heap.page_allocator;
const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
var data = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
defer data.deinit();

const m = descriptive.mean(f64, data);  // 3.0
```

**Notes**:
- For f64 calculations use floating-point mean for precise results
- Empty arrays return 0; should validate separately if needed

---

### Median

Compute median (middle value) of a 1D array.

**Signature**:
```zig
pub fn median(
    comptime T: type,
    data: NDArray(T, 1),
    allocator: Allocator,
) (NDArray(T, 1).Error || Allocator.Error)!T
```

**Description**:
- **Odd length**: Returns the middle value after sorting
- **Even length**: For float types, returns average of two middle values; for integers, returns lower middle

**Parameters**:
- `T`: Numeric type
- `data`: 1D NDArray
- `allocator`: Used for creating sorted copy (original unchanged)

**Returns**: Median value

**Errors**:
- `error.EmptyArray` if data is empty
- `Allocator.Error` if allocation fails

**Complexity**:
- **Time**: O(n log n) due to sorting
- **Space**: O(n) for sorted copy

**Example**:
```zig
const data_slice = [_]f64{ 5.0, 1.0, 3.0, 2.0, 4.0 };
var data = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
defer data.deinit();

const med = try descriptive.median(f64, data, allocator);  // 3.0
```

---

### Mode

Compute mode (most frequently occurring value) of a 1D array.

**Signature**:
```zig
pub fn mode(
    comptime T: type,
    data: NDArray(T, 1),
    allocator: Allocator,
) (NDArray(T, 1).Error || Allocator.Error)!T
```

**Description**:
Finds the most frequently occurring value. If multiple values have the same highest frequency (multimodal), returns the one encountered first.

**Parameters**:
- `T`: Integer type only (not float — floats cannot be hashed in Zig 0.15.x)
- `data`: 1D NDArray
- `allocator`: For hash map construction

**Returns**: Most frequent value

**Errors**:
- `error.EmptyArray` if data is empty
- `Allocator.Error` if allocation fails

**Complexity**:
- **Time**: O(n) average, O(n²) worst case (hash collisions)
- **Space**: O(k) where k = number of unique values

**Example**:
```zig
const data_slice = [_]i32{ 1, 2, 2, 3 };
var data = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{4}, &data_slice, .row_major);
defer data.deinit();

const m = try descriptive.mode(i32, data, allocator);  // 2
```

**Notes**:
- Integer types only (float hashing not available in Zig 0.15.x)

---

### Variance

Compute variance of a 1D array.

**Signature**:
```zig
pub fn variance(
    comptime T: type,
    data: NDArray(T, 1),
    ddof: usize,
) NDArray(T, 1).Error!T
```

**Description**:
Calculates variance using two-pass algorithm for numerical stability:
- **Population variance** (ddof=0): `sum((x_i - mean)²) / n`
- **Sample variance** (ddof=1): `sum((x_i - mean)²) / (n - 1)`

**Parameters**:
- `T`: Numeric type
- `data`: 1D NDArray
- `ddof`: Degrees of freedom adjustment (0 for population, 1 for sample)

**Returns**: Variance value (non-negative)

**Errors**:
- `error.EmptyArray` if data is empty
- `error.CapacityExceeded` if ddof >= data.size()

**Complexity**:
- **Time**: O(n) — two passes over data
- **Space**: O(1)

**Example**:
```zig
const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
var data = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
defer data.deinit();

const pop_var = try descriptive.variance(f64, data, 0);    // Population variance
const samp_var = try descriptive.variance(f64, data, 1);   // Sample variance
```

**Notes**:
- Two-pass algorithm prevents loss of precision for large values
- Use ddof=1 for samples, ddof=0 for entire population

---

### Standard Deviation

Compute standard deviation (square root of variance).

**Signature**:
```zig
pub fn stdDev(
    comptime T: type,
    data: NDArray(T, 1),
    ddof: usize,
) NDArray(T, 1).Error!T
```

**Description**:
Standard deviation = sqrt(variance). Measures spread of data around mean.

**Parameters**:
- `T`: Numeric type
- `data`: 1D NDArray
- `ddof`: Degrees of freedom adjustment (0 for population, 1 for sample)

**Returns**: Standard deviation

**Errors**: Same as variance()

**Complexity**:
- **Time**: O(n) — calls variance()
- **Space**: O(1)

**Example**:
```zig
const std_pop = try descriptive.stdDev(f64, data, 0);
const std_samp = try descriptive.stdDev(f64, data, 1);
```

---

### Quantile

Compute q-th quantile (0-quantile to 1-quantile).

**Signature**:
```zig
pub fn quantile(
    comptime T: type,
    data: NDArray(T, 1),
    q: T,
    allocator: Allocator,
) (NDArray(T, 1).Error || Allocator.Error)!T
```

**Description**:
Quantiles divide sorted data into equal-probability segments. Uses linear interpolation (NumPy default).

**Common Quantiles**:
- q=0.0: minimum
- q=0.25: first quartile (Q1)
- q=0.5: median
- q=0.75: third quartile (Q3)
- q=1.0: maximum

**Parameters**:
- `T`: Numeric type (float recommended)
- `data`: 1D NDArray
- `q`: Quantile parameter in [0.0, 1.0]
- `allocator`: For sorted copy

**Returns**: q-th quantile value

**Errors**:
- `error.EmptyArray` if data is empty
- `error.CapacityExceeded` if q not in [0, 1]
- `Allocator.Error` if allocation fails

**Complexity**:
- **Time**: O(n log n) due to sorting
- **Space**: O(n) for sorted copy

**Example**:
```zig
const q25 = try descriptive.quantile(f64, data, 0.25, allocator);  // Q1
const q50 = try descriptive.quantile(f64, data, 0.50, allocator);  // Median
const q95 = try descriptive.quantile(f64, data, 0.95, allocator);  // 95th percentile
```

---

### Percentile

Compute p-th percentile (0-percentile to 100-percentile).

**Signature**:
```zig
pub fn percentile(
    comptime T: type,
    data: NDArray(T, 1),
    p: T,
    allocator: Allocator,
) (NDArray(T, 1).Error || Allocator.Error)!T
```

**Description**:
Percentile is quantile scaled to [0, 100] range: `percentile(p) = quantile(p / 100)`

**Parameters**:
- `T`: Numeric type
- `data`: 1D NDArray
- `p`: Percentile in [0.0, 100.0]
- `allocator`: For sorted copy

**Returns**: p-th percentile value

**Errors**: Same as quantile()

**Complexity**:
- **Time**: O(n log n)
- **Space**: O(n)

**Example**:
```zig
const p95 = try descriptive.percentile(f64, data, 95.0, allocator);
```

---

### Skewness

Compute Fisher's skewness coefficient (standardized third moment).

**Signature**:
```zig
pub fn skewness(
    comptime T: type,
    data: NDArray(T, 1),
) NDArray(T, 1).Error!T
```

**Description**:
Skewness = E[(X - μ)³] / σ³ measures asymmetry of distribution.

**Interpretation**:
- skew < -0.5: left-skewed (longer tail on left)
- -0.5 ≤ skew ≤ 0.5: fairly symmetric
- skew > 0.5: right-skewed (longer tail on right)
- |skew| > 1.0: highly skewed

**Parameters**:
- `T`: Numeric type
- `data`: 1D NDArray

**Returns**: Skewness coefficient (dimensionless, typically [-3, 3])

**Errors**:
- `error.EmptyArray` if data is empty

**Complexity**:
- **Time**: O(n) — three passes (mean, std, third moment)
- **Space**: O(1)

**Example**:
```zig
const skew = try descriptive.skewness(f64, data);
if (skew > 0.5) {
    std.debug.print("Right-skewed distribution\n", .{});
}
```

**Notes**:
- Returns 0 for constant data (zero variance)

---

### Kurtosis

Compute excess kurtosis (standardized fourth moment).

**Signature**:
```zig
pub fn kurtosis(
    comptime T: type,
    data: NDArray(T, 1),
) NDArray(T, 1).Error!T
```

**Description**:
Excess kurtosis = E[(X - μ)⁴] / σ⁴ - 3 measures tail heaviness.

**Interpretation**:
- kurt ≈ 0: mesokurtic (normal-like)
- kurt < -2: platykurtic (light tails, flat)
- kurt > 3: leptokurtic (heavy tails, sharp peak)
- |kurt| < 2: similar to normal distribution

**Parameters**:
- `T`: Numeric type
- `data`: 1D NDArray

**Returns**: Excess kurtosis (typically [-2, ∞])

**Errors**:
- `error.EmptyArray` if data is empty

**Complexity**:
- **Time**: O(n) — four passes
- **Space**: O(1)

**Example**:
```zig
const kurt = try descriptive.kurtosis(f64, data);
if (kurt > 3.0) {
    std.debug.print("Heavy-tailed distribution\n", .{});
}
```

---

## Probability Distributions

Module: `zuda.stats.distributions`

Parameterized probability distributions for modeling random variables. Each distribution provides PDF, CDF, quantile, log-PDF, and sampling functions.

### Common Interface

All distributions follow a consistent API pattern:

```zig
// Initialize
var dist = try Distribution(f64).init(param1, param2);

// Evaluate functions
const pdf_val = dist.pdf(x);
const cdf_val = dist.cdf(x);
const q_val = try dist.quantile(p);  // p ∈ [0, 1]
const logpdf_val = dist.logpdf(x);

// Sample
var rng = std.Random.DefaultPrng.init(seed);
const sample = dist.sample(rng.random());
```

---

### Normal Distribution

Standard normal distribution (Gaussian) with mean μ and standard deviation σ.

**Type**: `Normal(T)` where T ∈ {f32, f64}

**Initialization**:
```zig
pub fn init(mu: T, sigma: T) !Normal(T)
```

**Parameters**:
- `mu`: Mean (location, any real value)
- `sigma`: Standard deviation (scale, must be > 0)

**Errors**:
- `error.InvalidStdDev` if sigma ≤ 0

**Methods**:

#### pdf — Probability Density Function

**Signature**: `pub fn pdf(self: Self, x: T) T`

Formula: f(x; μ, σ) = (1/(σ√(2π))) × exp(-(x-μ)²/(2σ²))

**Example**:
```zig
var normal = try Normal(f64).init(0.0, 1.0);
const density = normal.pdf(0.0);  // Peak at mean
```

#### cdf — Cumulative Distribution Function

**Signature**: `pub fn cdf(self: Self, x: T) T`

Returns: P(X ≤ x) ∈ [0, 1]

Formula: F(x; μ, σ) = 0.5 × [1 + erf((x-μ)/(σ√2))]

**Example**:
```zig
const prob = normal.cdf(1.96);  // Standard normal: ≈ 0.975
```

#### quantile — Inverse CDF

**Signature**: `pub fn quantile(self: Self, p: T) !T`

Returns: x such that P(X ≤ x) = p

Uses Acklam's rational approximation (error < 1.15e-9 for f64)

**Errors**:
- `error.InvalidProbability` if p < 0 or p > 1

**Example**:
```zig
const q95 = try normal.quantile(0.95);  // 95th percentile
```

#### logpdf — Log Probability Density

**Signature**: `pub fn logpdf(self: Self, x: T) T`

More numerically stable than log(pdf(x)) for extreme values.

Formula: log(f(x)) = -0.5×log(2π) - log(σ) - (x-μ)²/(2σ²)

#### sample — Random Sampling

**Signature**: `pub fn sample(self: Self, rng: std.Random) T`

Uses Box-Muller transform to generate random samples.

**Example**:
```zig
var prng = std.Random.DefaultPrng.init(12345);
var dist = try Normal(f64).init(100.0, 15.0);
const sample = dist.sample(prng.random());  // IQ-like distribution
```

**Complexity**:
- **Time**: pdf, cdf, quantile, logpdf: O(1); sample: O(1)
- **Space**: O(1)

---

### Uniform Distribution

Continuous uniform distribution over [a, b].

**Type**: `Uniform(T)` where T ∈ {f32, f64}

**Initialization**:
```zig
pub fn init(a: T, b: T) !Uniform(T)
```

**Parameters**:
- `a`: Lower bound (must be < b)
- `b`: Upper bound (must be > a)

**Errors**:
- `error.InvalidBounds` if a ≥ b

**Properties**:
- Mean: (a + b) / 2
- Variance: (b - a)² / 12

**Example**:
```zig
var uniform = try Uniform(f64).init(0.0, 1.0);
const density = uniform.pdf(0.5);  // 1.0 (constant)
const prob = uniform.cdf(0.75);    // 0.75
const q = try uniform.quantile(0.5);  // 0.5 (median)
```

---

### Exponential Distribution

Continuous exponential distribution modeling time between events (rate parameter λ).

**Type**: `Exponential(T)` where T ∈ {f32, f64}

**Initialization**:
```zig
pub fn init(lambda: T) !Exponential(T)
```

**Parameters**:
- `lambda`: Rate parameter (must be > 0); higher λ = events occur more frequently

**Errors**:
- `error.InvalidParameter` if lambda ≤ 0

**Properties**:
- Mean: 1/λ
- Variance: 1/λ²
- Memoryless: P(X > s+t | X > s) = P(X > t)

**Example**:
```zig
var exp = try Exponential(f64).init(0.5);  // λ=0.5, mean=2.0
const mle = exp.pdf(1.0);
const prob = exp.cdf(3.0);  // P(X ≤ 3)
```

**Use Cases**:
- Waiting times between arrivals (queuing theory)
- Equipment failure times
- Radioactive decay

---

### Gamma Distribution

Continuous gamma distribution with shape α and rate β parameters.

**Type**: `Gamma(T)` where T ∈ {f32, f64}

**Initialization**:
```zig
pub fn init(alpha: T, beta: T) !Gamma(T)
```

**Parameters**:
- `alpha`: Shape parameter (must be > 0); controls skewness
- `beta`: Rate parameter (must be > 0); scales the distribution

**Errors**:
- `error.InvalidShape` if alpha ≤ 0
- `error.InvalidRate` if beta ≤ 0

**Properties**:
- Mean: α/β
- Variance: α/β²
- Special cases: Gamma(1, β) = Exponential(β); Gamma(ν/2, 1/2) = ChiSquared(ν)

**Example**:
```zig
var gamma = try Gamma(f64).init(2.0, 1.0);
const pdf_val = gamma.pdf(1.5);
```

---

### Beta Distribution

Continuous beta distribution over [0, 1] with shape parameters α and β.

**Type**: `Beta(T)` where T ∈ {f32, f64}

**Initialization**:
```zig
pub fn init(alpha: T, beta: T) !Beta(T)
```

**Parameters**:
- `alpha`: First shape parameter (must be > 0)
- `beta`: Second shape parameter (must be > 0)

**Properties**:
- Range: [0, 1]
- Mean: α/(α+β)
- Variance: αβ/[(α+β)²(α+β+1)]
- Symmetric if α = β
- Uniform(0,1) when α = β = 1

**Example**:
```zig
var beta = try Beta(f64).init(2.0, 5.0);  // Skewed left
const prob = beta.cdf(0.5);
```

---

### Chi-Squared Distribution

Continuous chi-squared distribution with ν degrees of freedom (special case of Gamma).

**Type**: `ChiSquared(T)` where T ∈ {f32, f64}

**Initialization**:
```zig
pub fn init(nu: T) !ChiSquared(T)
```

**Parameters**:
- `nu`: Degrees of freedom (must be > 0)

**Properties**:
- Mean: ν
- Variance: 2ν
- Range: [0, ∞)
- Right-skewed; approaches normal as ν → ∞

**Use Cases**:
- Chi-squared goodness-of-fit tests
- Independence testing (contingency tables)
- Confidence intervals for variance

**Example**:
```zig
var chi2 = try ChiSquared(f64).init(10.0);
const critical = try chi2.quantile(0.95);  // α=0.05 critical value
```

---

### Student's t-Distribution

Continuous t-distribution with ν degrees of freedom (heavy-tailed alternative to normal).

**Type**: `StudentT(T)` where T ∈ {f32, f64}

**Initialization**:
```zig
pub fn init(nu: T) !StudentT(T)
```

**Parameters**:
- `nu`: Degrees of freedom (must be > 0)

**Properties**:
- Mean: 0 (for ν > 1)
- Variance: ν/(ν-2) (for ν > 2)
- Symmetric around 0
- Heavier tails than normal
- t(1) = Cauchy distribution; t(ν) → Normal(0,1) as ν → ∞

**Use Cases**:
- Student's t-tests (hypothesis testing)
- Confidence intervals for means with unknown variance
- Regression analysis

**Example**:
```zig
var t = try StudentT(f64).init(20.0);  // df=20
const critical = try t.quantile(0.975);  // Two-tailed α=0.05
const p_value = 2.0 * (1.0 - t.cdf(@abs(t_stat)));  // Two-tailed p
```

---

### F-Distribution

Continuous F-distribution with ν₁ and ν₂ degrees of freedom (ratio of chi-squared).

**Type**: `FDistribution(T)` where T ∈ {f32, f64}

**Initialization**:
```zig
pub fn init(nu1: T, nu2: T) !FDistribution(T)
```

**Parameters**:
- `nu1`: Numerator degrees of freedom (must be > 0)
- `nu2`: Denominator degrees of freedom (must be > 0)

**Properties**:
- Range: [0, ∞)
- Mean: ν₂/(ν₂-2) (for ν₂ > 2)
- Right-skewed

**Use Cases**:
- ANOVA (Analysis of Variance)
- Testing equality of variances (Levene's test)
- Model comparison (F-test for regression)

---

### Binomial Distribution

Discrete binomial distribution: number of successes in n independent Bernoulli trials.

**Type**: `Binomial(T)` where T ∈ {f32, f64}

**Initialization**:
```zig
pub fn init(n: i64, p: T) !Binomial(T)
```

**Parameters**:
- `n`: Number of trials (must be > 0)
- `p`: Probability of success per trial ∈ [0, 1]

**Properties**:
- Mean: n×p
- Variance: n×p×(1-p)
- Support: {0, 1, ..., n}
- Binomial(n, 0.5) is symmetric

**Example**:
```zig
var binom = try Binomial(f64).init(20, 0.6);
const pmf = binom.pmf(12);  // P(X = 12)
const cdf = binom.cdf(12);  // P(X ≤ 12)
```

---

### Poisson Distribution

Discrete Poisson distribution: count of events in fixed interval with rate λ.

**Type**: `Poisson(T)` where T ∈ {f32, f64}

**Initialization**:
```zig
pub fn init(lambda: T) !Poisson(T)
```

**Parameters**:
- `lambda`: Rate parameter (must be > 0); expected count

**Properties**:
- Mean: λ
- Variance: λ
- Support: {0, 1, 2, ...}
- Approaches normal as λ → ∞

**Use Cases**:
- Modeling rare events (accidents, defects)
- Queuing systems
- Spatial/temporal point processes

**Example**:
```zig
var poisson = try Poisson(f64).init(5.0);
const prob_3 = poisson.pmf(3);  // P(X = 3)
const prob_le_5 = poisson.cdf(5);  // P(X ≤ 5)
```

---

### Geometric Distribution

Discrete geometric distribution: number of trials until first success.

**Type**: `Geometric(T)` where T ∈ {f32, f64}

**Initialization**:
```zig
pub fn init(p: T) !Geometric(T)
```

**Parameters**:
- `p`: Probability of success per trial ∈ (0, 1]

**Properties**:
- Mean: 1/p
- Variance: (1-p)/p²
- Support: {1, 2, 3, ...}
- Memoryless: P(X > m+n | X > m) = P(X > n)

**Example**:
```zig
var geom = try Geometric(f64).init(0.3);
const prob_first_in_5 = geom.cdf(5);
```

---

### Bernoulli Distribution

Discrete Bernoulli distribution: single binary trial with success probability p.

**Type**: `Bernoulli(T)` where T ∈ {f32, f64}

**Initialization**:
```zig
pub fn init(p: T) !Bernoulli(T)
```

**Parameters**:
- `p`: Probability of success ∈ [0, 1]

**Properties**:
- Mean: p
- Variance: p(1-p)
- Support: {0, 1}
- Binomial(1, p) = Bernoulli(p)

---

## Hypothesis Testing

Module: `zuda.stats.hypothesis`

Classical statistical tests for comparing means and proportions. All tests return `TestResult(T)` containing statistic, p-value, degrees of freedom, and rejection decision.

### TestResult Type

Generic result container for hypothesis tests.

**Definition**:
```zig
pub fn TestResult(comptime T: type) type {
    return struct {
        statistic: T,      // Test statistic (t-value, chi-square, etc.)
        p_value: T,        // p-value ∈ [0, 1]
        df: T,             // Degrees of freedom
        reject: bool,      // true if p_value < alpha
    };
}
```

**Creation**:
```zig
const result = TestResult(f64).init(statistic, p_value, df, alpha);
```

---

### One-Sample t-Test

Test whether sample mean differs from hypothesized population mean.

**Signature**:
```zig
pub fn ttest_1samp(
    comptime T: type,
    data: NDArray(T, 1),
    population_mean: T,
    alpha: T,
) !TestResult(T)
```

**Hypothesis**:
- H₀: Sample mean = population_mean
- H₁: Sample mean ≠ population_mean (two-tailed)

**Formula**:
t = (x̄ - μ) / (s / √n)

where x̄ = sample mean, μ = hypothesized mean, s = sample std dev, n = sample size
df = n - 1

**Parameters**:
- `T`: f32 or f64
- `data`: 1D NDArray of observations
- `population_mean`: Hypothesized mean under H₀
- `alpha`: Significance level (typical 0.05)

**Returns**: TestResult with t-statistic, two-tailed p-value, df, and rejection decision

**Errors**:
- `error.EmptyArray` if data is empty
- `error.InvalidParameter` if alpha ∉ (0, 1)

**Complexity**:
- **Time**: O(n) — two passes (mean, variance)
- **Space**: O(1)

**Example**:
```zig
const data_slice = [_]f64{ 18.5, 19.2, 17.8, 20.1, 19.5 };
var data = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
defer data.deinit();

const result = try hypothesis.ttest_1samp(f64, data, 20.0, 0.05);
if (result.reject) {
    std.debug.print("Reject H0: sample mean ≠ 20.0 (p = {d})\n", .{result.p_value});
}
```

---

### Independent Samples t-Test

Test whether two independent groups have different means.

**Signature**:
```zig
pub fn ttest_ind(
    comptime T: type,
    sample1: NDArray(T, 1),
    sample2: NDArray(T, 1),
    alpha: T,
    equal_var: bool,
) !TestResult(T)
```

**Hypothesis**:
- H₀: μ₁ = μ₂
- H₁: μ₁ ≠ μ₂ (two-tailed)

**Variants**:

1. **Welch's t-test** (equal_var=false, recommended):
   - Does not assume equal variances
   - t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)
   - df via Welch-Satterthwaite approximation

2. **Pooled variance t-test** (equal_var=true):
   - Assumes equal variances
   - Uses pooled standard error
   - df = n₁ + n₂ - 2

**Parameters**:
- `sample1`, `sample2`: 1D NDArray of observations
- `alpha`: Significance level
- `equal_var`: If true, assume equal variances; else use Welch's test

**Returns**: TestResult with t-statistic, two-tailed p-value, df, and rejection

**Errors**:
- `error.EmptyArray` if either sample is empty
- `error.InvalidParameter` if alpha ∉ (0, 1)

**Complexity**:
- **Time**: O(n₁ + n₂)
- **Space**: O(1)

**Example**:
```zig
// Compare two treatment groups
const ctrl = [_]f64{ 10.2, 10.5, 10.1, 10.3 };
const treat = [_]f64{ 11.5, 11.8, 11.2, 11.7 };

var ctrl_data = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &ctrl, .row_major);
var treat_data = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &treat, .row_major);
defer ctrl_data.deinit();
defer treat_data.deinit();

const result = try hypothesis.ttest_ind(f64, ctrl_data, treat_data, 0.05, false);
std.debug.print("t = {d}, p = {d}\n", .{result.statistic, result.p_value});
```

---

### Paired Samples t-Test

Test whether paired observations have different means.

**Signature**:
```zig
pub fn ttest_rel(
    comptime T: type,
    before: NDArray(T, 1),
    after: NDArray(T, 1),
    alpha: T,
) !TestResult(T)
```

**Description**:
Tests paired data (before-after, matched controls). Equivalent to one-sample t-test on differences.

**Formula**:
- d_i = before_i - after_i
- t = d̄ / (s_d / √n)
- df = n - 1

**Parameters**:
- `before`, `after`: 1D NDArray of equal length
- `alpha`: Significance level

**Returns**: TestResult with t-statistic, p-value, df

**Errors**:
- `error.EmptyArray` if arrays are empty
- `error.UnequalLengths` if arrays have different sizes
- `error.InvalidParameter` if alpha ∉ (0, 1)

**Complexity**:
- **Time**: O(n)
- **Space**: O(n) for differences array

**Example**:
```zig
const before = [_]f64{ 120.0, 125.0, 122.0, 128.0 };
const after = [_]f64{ 115.0, 120.0, 118.0, 123.0 };

var before_data = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &before, .row_major);
var after_data = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &after, .row_major);
defer before_data.deinit();
defer after_data.deinit();

const result = try hypothesis.ttest_rel(f64, before_data, after_data, 0.05);
if (result.reject) {
    std.debug.print("Significant change (p = {d})\n", .{result.p_value});
}
```

---

### Chi-Squared Goodness-of-Fit Test

Test whether observed frequencies match expected distribution.

**Signature**:
```zig
pub fn chi2_test(
    observed: []const u64,
    expected: []const f64,
    alpha: f64,
    allocator: Allocator,
) !TestResult(f64)
```

**Hypothesis**:
- H₀: Observed frequencies match expected distribution
- H₁: Observed frequencies differ from expected

**Formula**:
χ² = Σ(O_i - E_i)² / E_i

where O_i = observed count, E_i = expected count
df = k - 1 (k = number of categories)

**Parameters**:
- `observed`: Array of observed frequencies (non-negative integers)
- `expected`: Array of expected frequencies (must match observed length)
- `alpha`: Significance level
- `allocator`: For internal computations

**Returns**: TestResult with χ² statistic, p-value, df

**Errors**:
- `error.LengthMismatch` if observed.len ≠ expected.len
- `error.InvalidParameter` if any expected < 0 or sum = 0

**Complexity**:
- **Time**: O(k) where k = number of categories
- **Space**: O(1)

**Example**:
```zig
const observed = [_]u64{ 10, 15, 8, 12 };
const expected = [_]f64{ 10.0, 10.0, 10.0, 15.0 };

const result = try hypothesis.chi2_test(&observed, &expected, 0.05, allocator);
std.debug.print("χ² = {d}, p = {d}\n", .{result.statistic, result.p_value});
```

---

### One-Way ANOVA

Test whether multiple groups have equal means.

**Signature**:
```zig
pub fn anova_oneway(
    comptime T: type,
    groups: []const NDArray(T, 1),
    alpha: T,
    allocator: Allocator,
) !TestResult(T)
```

**Hypothesis**:
- H₀: μ₁ = μ₂ = ... = μₖ (all group means equal)
- H₁: At least one group mean differs

**Formula**:
F = MSB / MSW

where:
- MSB = Between-group mean square
- MSW = Within-group mean square (error)
- df_between = k - 1
- df_within = n - k

**Parameters**:
- `T`: f32 or f64
- `groups`: Slice of NDArray(T, 1), one per group
- `alpha`: Significance level
- `allocator`: For internal arrays

**Returns**: TestResult with F-statistic, p-value, df_between, rejection decision

**Errors**:
- `error.InsufficientGroups` if fewer than 2 groups
- `error.EmptyGroup` if any group is empty
- `error.InvalidParameter` if alpha ∉ (0, 1)

**Complexity**:
- **Time**: O(N) where N = total observations
- **Space**: O(k) for group statistics

**Example**:
```zig
const ctrl = [_]f64{ 10.0, 10.5, 10.2 };
const drug_a = [_]f64{ 12.0, 12.5, 11.8 };
const drug_b = [_]f64{ 11.0, 11.3, 11.1 };

var g1 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &ctrl, .row_major);
var g2 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &drug_a, .row_major);
var g3 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &drug_b, .row_major);
defer g1.deinit();
defer g2.deinit();
defer g3.deinit();

var groups = [_]NDArray(f64, 1){ g1, g2, g3 };
const result = try hypothesis.anova_oneway(f64, &groups, 0.05, allocator);
std.debug.print("F = {d}, p = {d}\n", .{result.statistic, result.p_value});
```

---

## Correlation and Covariance

Module: `zuda.stats.correlation`

Functions for measuring linear relationships between variables and fitting regression models.

### Pearson Correlation Coefficient

Parametric correlation measure for linear relationships.

**Signature**:
```zig
pub fn pearson(
    x: NDArray(f64, 1),
    y: NDArray(f64, 1),
    allocator: Allocator,
) !f64
```

**Description**:
Pearson r measures linear relationship: r = cov(x,y) / (σ_x × σ_y)

**Properties**:
- **Range**: [-1, 1]
- **Interpretation**:
  - r = 1: perfect positive linear
  - r = 0: no linear correlation
  - r = -1: perfect negative linear
  - |r| < 0.3: weak; 0.3-0.7: moderate; >0.7: strong
- **Assumptions**: Bivariate normality for inference
- **Robustness**: Sensitive to outliers

**Parameters**:
- `x`, `y`: 1D NDArray(f64, 1) of equal length
- `allocator`: (unused, for API consistency)

**Returns**: Correlation coefficient ∈ [-1, 1]

**Errors**:
- `error.EmptyArray` if either is empty
- `error.DimensionMismatch` if lengths differ
- `error.ZeroStdDev` if either has zero std dev

**Complexity**:
- **Time**: O(n) — two passes
- **Space**: O(1)

**Example**:
```zig
const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
const y_data = [_]f64{ 2.0, 4.0, 6.0, 8.0, 10.0 };

var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &x_data, .row_major);
var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &y_data, .row_major);
defer x.deinit();
defer y.deinit();

const r = try correlation.pearson(x, y, allocator);  // 1.0 (perfect)
```

---

### Spearman Rank Correlation

Non-parametric correlation using ranks instead of values.

**Signature**:
```zig
pub fn spearman(
    x: NDArray(f64, 1),
    y: NDArray(f64, 1),
    allocator: Allocator,
) !f64
```

**Description**:
Non-parametric alternative to Pearson. Detects monotonic (not just linear) relationships.

**Algorithm**:
1. Convert x and y to ranks (handles ties by averaging)
2. Compute Pearson r on ranks

**Properties**:
- **Range**: [-1, 1]
- **Assumptions**: None (non-parametric)
- **Robustness**: More robust to outliers than Pearson
- **Detects**: Monotonic relationships

**Parameters**:
- `x`, `y`: 1D NDArray(f64, 1) of equal length
- `allocator`: For temporary rank arrays

**Returns**: Spearman ρ ∈ [-1, 1]

**Errors**:
- `error.EmptyArray`
- `error.DimensionMismatch`
- `Allocator.Error`

**Complexity**:
- **Time**: O(n log n) due to sorting/ranking
- **Space**: O(n) for temporary rank arrays

**Example**:
```zig
// Non-linear but monotonic: y = x²
const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
const y_data = [_]f64{ 1.0, 4.0, 9.0, 16.0, 25.0 };

var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &x_data, .row_major);
var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &y_data, .row_major);
defer x.deinit();
defer y.deinit();

const rho = try correlation.spearman(x, y, allocator);  // 1.0 (perfect monotonic)
```

---

### Simple Linear Regression

Fit ordinary least squares (OLS) regression model: ŷ = intercept + slope × x

**Signature**:
```zig
pub fn linregress(
    x: NDArray(f64, 1),
    y: NDArray(f64, 1),
    allocator: Allocator,
) !RegressionResult
```

**Description**:
Fits linear model minimizing Σ(y_i - ŷ_i)².

**Formulas**:
- slope = cov(x,y) / var(x)
- intercept = mean(y) - slope × mean(x)
- R² = 1 - (RSS / TSS) — proportion of variance explained
- std_err = √(MSE / Σ(x_i - x̄)²)
- t = slope / std_err → p-value via t-distribution

**where**:
- RSS = Σ(y_i - ŷ_i)² — residual sum of squares
- TSS = Σ(y_i - ȳ)² — total sum of squares
- MSE = RSS / (n - 2) — mean squared error

**Return Type**: `RegressionResult`
```zig
pub const RegressionResult = struct {
    slope: f64,
    intercept: f64,
    r_squared: f64,       // R²: [0, 1]
    p_value: f64,         // p-value for slope significance
    std_err: f64,         // Standard error of slope
};
```

**Parameters**:
- `x`, `y`: 1D NDArray(f64, 1) of equal length, ≥ 3 observations
- `allocator`: (unused, for API consistency)

**Returns**: RegressionResult

**Errors**:
- `error.EmptyArray` if either is empty
- `error.DimensionMismatch` if lengths differ
- `error.ConstantX` if var(x) = 0 (undefined slope)
- `error.InsufficientSamples` if n < 3

**Complexity**:
- **Time**: O(n) — single pass with accumulation
- **Space**: O(1)

**Example**:
```zig
const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
const y_data = [_]f64{ 3.0, 5.0, 7.0, 9.0, 11.0 };  // y = 2x + 1

var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &x_data, .row_major);
var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &y_data, .row_major);
defer x.deinit();
defer y.deinit();

const result = try correlation.linregress(x, y, allocator);
std.debug.print("ŷ = {d} + {d}x\n", .{result.intercept, result.slope});
std.debug.print("R² = {d}, p = {d}\n", .{result.r_squared, result.p_value});
```

**Interpretation**:
- **slope**: Change in y for unit increase in x
- **intercept**: Predicted y when x=0
- **R²**: Proportion of y variance explained by x (0.8+ is good)
- **p_value**: Significance of slope (p<0.05 = significant)
- **std_err**: Uncertainty in slope estimate

---

## Common Patterns and Examples

### Example 1: Exploratory Data Analysis

```zig
const allocator = std.heap.page_allocator;

// Sample dataset
const data_raw = [_]f64{
    72.0, 68.0, 75.0, 81.0, 69.0,
    78.0, 85.0, 71.0, 79.0, 82.0,
};

var data = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{10}, &data_raw, .row_major);
defer data.deinit();

// Descriptive statistics
const mean_val = descriptive.mean(f64, data);
const median_val = try descriptive.median(f64, data, allocator);
const std_val = try descriptive.stdDev(f64, data, 1);
const skew = try descriptive.skewness(f64, data);
const kurt = try descriptive.kurtosis(f64, data);

std.debug.print("Mean: {d}\n", .{mean_val});
std.debug.print("Median: {d}\n", .{median_val});
std.debug.print("Std Dev: {d}\n", .{std_val});
std.debug.print("Skewness: {d}\n", .{skew});
std.debug.print("Kurtosis: {d}\n", .{kurt});

// Quantiles
const q1 = try descriptive.quantile(f64, data, 0.25, allocator);
const q3 = try descriptive.quantile(f64, data, 0.75, allocator);
const iqr = q3 - q1;

std.debug.print("IQR: {d}\n", .{iqr});
```

---

### Example 2: Hypothesis Testing Workflow

```zig
// Test: Did drug treatment significantly increase performance?
const control = [_]f64{ 10.2, 10.5, 10.1, 10.3, 10.4 };
const treated = [_]f64{ 12.1, 12.5, 11.8, 12.3, 12.0 };

var ctrl_data = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &control, .row_major);
var treat_data = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &treated, .row_major);
defer ctrl_data.deinit();
defer treat_data.deinit();

const alpha = 0.05;
const result = try hypothesis.ttest_ind(f64, ctrl_data, treat_data, alpha, false);

std.debug.print("t-statistic: {d}\n", .{result.statistic});
std.debug.print("p-value: {d}\n", .{result.p_value});
std.debug.print("df: {d}\n", .{result.df});

if (result.reject) {
    std.debug.print("CONCLUSION: Reject H0 (drug is effective at α={d})\n", .{alpha});
} else {
    std.debug.print("CONCLUSION: Fail to reject H0 (insufficient evidence)\n", .{});
}
```

---

### Example 3: Regression and Prediction

```zig
// Study: Relationship between study hours and exam score
const hours = [_]f64{ 1.0, 2.0, 2.5, 3.0, 4.0, 5.0 };
const scores = [_]f64{ 55.0, 65.0, 70.0, 75.0, 85.0, 95.0 };

var x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{6}, &hours, .row_major);
var y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{6}, &scores, .row_major);
defer x.deinit();
defer y.deinit();

const result = try correlation.linregress(x, y, allocator);

std.debug.print("Model: score = {d} + {d} * hours\n", .{result.intercept, result.slope});
std.debug.print("R² = {d} ({d}% variance explained)\n", .{result.r_squared, result.r_squared * 100});
std.debug.print("p-value = {d} (slope is ", .{result.p_value});

if (result.p_value < 0.05) {
    std.debug.print("significant)\n", .{});
} else {
    std.debug.print("not significant)\n", .{});
}

// Prediction: What score for 3.5 hours?
const x_new = 3.5;
const pred = result.intercept + result.slope * x_new;
std.debug.print("Predicted score for {d} hours: {d}\n", .{x_new, pred});
```

---

### Example 4: Probability Distribution Sampling

```zig
var prng = std.Random.DefaultPrng.init(12345);
var rng = prng.random();

// Normal distribution sampling
var normal = try Normal(f64).init(100.0, 15.0);
std.debug.print("IQ-like samples:\n", .{});
for (0..10) |_| {
    const sample = normal.sample(rng);
    std.debug.print("{d} ", .{sample});
}
std.debug.print("\n", .{});

// Binomial distribution
var binom = try Binomial(f64).init(20, 0.6);
const pmf_12 = binom.pmf(12);
const cdf_15 = binom.cdf(15);
std.debug.print("P(X=12) = {d}\n", .{pmf_12});
std.debug.print("P(X≤15) = {d}\n", .{cdf_15});

// Quantile: 95th percentile of normal
const normal_std = try Normal(f64).init(0.0, 1.0);
const p95 = try normal_std.quantile(0.95);
std.debug.print("95th percentile of N(0,1): {d}\n", .{p95});
```

---

### Example 5: Correlation Matrix Computation

```zig
// Multiple variables: Height, Weight, Age
const heights = [_]f64{ 170, 165, 180, 175, 168 };
const weights = [_]f64{ 72, 65, 82, 78, 70 };
const ages = [_]f64{ 25, 22, 30, 28, 24 };

var h = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &heights, .row_major);
var w = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &weights, .row_major);
var a = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &ages, .row_major);
defer h.deinit();
defer w.deinit();
defer a.deinit();

const r_hw = try correlation.pearson(h, w, allocator);
const r_ha = try correlation.pearson(h, a, allocator);
const r_wa = try correlation.pearson(w, a, allocator);

std.debug.print("Correlation Matrix:\n", .{});
std.debug.print("Height-Weight: {d}\n", .{r_hw});
std.debug.print("Height-Age: {d}\n", .{r_ha});
std.debug.print("Weight-Age: {d}\n", .{r_wa});
```

---

## Performance Considerations

### Memory Allocation

- **Quantile/Median/Spearman**: Allocate O(n) for sorted/ranked copies
- Use consistent allocator throughout for predictable performance
- Consider page allocator for large datasets

### Numerical Stability

- **Variance**: Two-pass algorithm prevents precision loss
- **Correlation**: Covariance computed with mean-centering
- **Quantile**: Linear interpolation handles boundary cases

### Time Complexity Summary

| Function | Time | Space |
|----------|------|-------|
| mean, variance, stdDev, skewness, kurtosis | O(n) | O(1) |
| median, quantile, percentile | O(n log n) | O(n) |
| mode | O(n) | O(k) |
| pearson, linregress | O(n) | O(1) |
| spearman | O(n log n) | O(n) |
| ttest_1samp, ttest_ind, ttest_rel | O(n) | O(n) or O(1) |
| chi2_test | O(k) | O(1) |
| anova_oneway | O(N) | O(k) |
| Distribution pdf/cdf/quantile | O(1) | O(1) |

---

## Error Handling

All functions return error unions. Common errors:

- **error.EmptyArray**: Input data is empty
- **error.InvalidParameter**: Parameter out of valid range (e.g., alpha, p)
- **error.InvalidStdDev**: Standard deviation ≤ 0
- **error.ConstantX**: Regression with zero variance in x
- **error.DimensionMismatch**: Arrays of unequal length
- **error.UnequalLengths**: Paired test arrays differ
- **error.ZeroStdDev**: Zero variance (can't compute correlation)
- **Allocator.Error**: Memory allocation failed

Example error handling:

```zig
const result = median(f64, data, allocator) catch |err| {
    switch (err) {
        error.EmptyArray => std.debug.print("Error: data is empty\n", .{}),
        else => std.debug.print("Allocation failed\n", .{}),
    }
    return;
};
```

---

## Type Support

- **Floating-point**: f32, f64 (preferred for most operations)
- **Integer**: i32, i64, u32, u64
- **Limitations**:
  - Mode: integer types only
  - Quantile: float types recommended for interpolation accuracy
  - Distributions: f32, f64 only
  - Hypothesis tests: f32, f64 only

---

## References and Further Reading

- **Descriptive Statistics**: Press et al. "Numerical Recipes" (3rd ed.)
- **Distributions**: "The Handbook of Mathematical Functions" (Abramowitz & Stegun)
- **Hypothesis Testing**: Rice, "Mathematical Statistics and Data Analysis" (3rd ed.)
- **Correlation**: "Introduction to Statistical Learning" (James et al.)
- **Regression**: "The Elements of Statistical Learning" (Hastie, Tibshirani, Friedman)

---

**Module Status**: Complete for v2.0.0 release ✅

Last Updated: 2026-03-27
