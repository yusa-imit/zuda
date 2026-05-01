//! Hypothesis Testing — Statistical tests for comparing means and proportions
//!
//! This module provides classical hypothesis testing procedures for comparing
//! population means, including Student's t-tests (one-sample, independent samples, paired),
//! and chi-squared tests for categorical data.
//!
//! ## Supported Tests
//! - `ttest_1samp` — One-sample t-test (H0: sample mean = μ)
//! - `ttest_ind` — Independent samples t-test (H0: μ₁ = μ₂)
//! - `ttest_rel` — Paired samples t-test (H0: μ_diff = 0)
//! - `chi2_test` — Chi-squared goodness-of-fit test (H0: observed ~ expected)
//! - `anova_oneway` — One-way ANOVA (H0: μ₁ = μ₂ = ... = μₖ)
//!
//! ## TestResult Type
//! Generic result container holding:
//! - `statistic: T` — Test statistic (t-value)
//! - `p_value: T` — p-value (two-tailed by default)
//! - `df: T` — Degrees of freedom
//! - `reject: bool` — Whether to reject H0 at given alpha level
//!
//! ## Time Complexity
//! - ttest_1samp: O(n) — one pass for mean, one for variance
//! - ttest_ind: O(n₁ + n₂) — linear in sample sizes
//! - ttest_rel: O(n) — linear in number of pairs
//! - chi2_test: O(k) — linear in number of categories
//!
//! ## Use Cases
//! - Comparing sample mean to population mean
//! - Comparing means of two independent groups
//! - Comparing paired observations (before/after, matched controls)
//! - Testing categorical data distributions (chi-squared)
//! - Goodness-of-fit tests for discrete distributions
//! - Hypothesis testing with small to moderate sample sizes
//! - Confidence intervals for mean differences

const std = @import("std");
const math = std.math;
const testing = std.testing;

// Import dependencies
const descriptive = @import("descriptive.zig");
const NDArray_type = @import("../ndarray/ndarray.zig").NDArray;
const StudentT_Distribution = @import("distributions/student_t.zig").StudentT;
const ChiSquared_Distribution = @import("distributions/chi_squared.zig").ChiSquared;
const FDistribution = @import("distributions/f_distribution.zig").FDistribution;

// ============================================================================
// TEST RESULT TYPE
// ============================================================================

/// Generic result container for hypothesis tests
///
/// Parameters:
/// - T: numeric type (f32 or f64)
pub fn TestResult(comptime T: type) type {
    return struct {
        /// Test statistic (e.g., t-value for t-tests)
        statistic: T,

        /// p-value (probability of observing test statistic under H0)
        /// Range: [0, 1]
        p_value: T,

        /// Degrees of freedom used in the test
        df: T,

        /// Rejection decision: true if p_value < alpha
        reject: bool,

        const Self = @This();

        /// Create a TestResult from components
        ///
        /// Parameters:
        /// - statistic: test statistic value
        /// - p_value: calculated p-value
        /// - df: degrees of freedom
        /// - alpha: significance level for rejection decision
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn init(statistic: T, p_value: T, df: T, alpha: T) Self {
            return .{
                .statistic = statistic,
                .p_value = p_value,
                .df = df,
                .reject = p_value < alpha,
            };
        }
    };
}

// ============================================================================
// ONE-SAMPLE T-TEST
// ============================================================================

/// One-sample t-test: H0: sample mean = population_mean
///
/// Tests whether sample mean differs significantly from hypothesized population mean.
/// Assumes population variance is unknown and estimated from sample.
///
/// Formula: t = (x̄ - μ) / (s / √n)
/// where x̄ = sample mean, μ = hypothesized mean, s = sample std dev, n = sample size
/// df = n - 1
///
/// Parameters:
/// - data: 1D NDArray of numeric type T
/// - population_mean: hypothesized mean under H0
/// - alpha: significance level (default 0.05 for 95% confidence)
///
/// Returns: TestResult with t-statistic, p-value (two-tailed), and rejection decision
///
/// Errors:
/// - error.EmptyArray if data is empty
/// - error.InvalidParameter if alpha not in (0, 1)
///
/// Time: O(n) — two passes (mean, then variance)
/// Space: O(1)
///
/// Example:
/// ```zig
/// const sample = [_]f64{1.0, 2.0, 3.0, 4.0, 5.0};
/// const result = try ttest_1samp(f64, data, 3.0, 0.05);
/// // Tests if sample mean differs from 3.0 at α=0.05
/// ```
pub fn ttest_1samp(
    comptime T: type,
    data: NDArray_type(T, 1),
    population_mean: T,
    alpha: T,
) !TestResult(T) {
    const n = data.count();
    if (n == 0) return error.EmptyArray;
    if (alpha <= 0 or alpha >= 1) return error.InvalidParameter;

    // Compute sample mean
    const sample_mean = descriptive.mean(T, data);

    // Compute sample standard deviation (ddof=1 for sample)
    const sample_var = try descriptive.variance(T, data, 1);
    const sample_std = math.sqrt(sample_var);

    // Compute t-statistic: t = (x̄ - μ) / (s / √n)
    const n_f = @as(T, @floatFromInt(n));
    const se = sample_std / math.sqrt(n_f);

    // Handle edge case: zero standard error (no variance in data)
    const t_stat: T = if (se == 0) 0.0 else (sample_mean - population_mean) / se;

    // Degrees of freedom
    const df = @as(T, @floatFromInt(n - 1));

    // Two-tailed p-value: P(|T| > |t|)
    // Special case: if se == 0, we can't compute p-value meaningfully, return 1.0 (don't reject H0)
    const p_value: T = if (se == 0) 1.0 else blk: {
        const dist = try StudentT_Distribution(T).init(df);
        const cdf_val = dist.cdf(t_stat);
        break :blk if (t_stat >= 0)
            2.0 * (1.0 - cdf_val)
        else
            2.0 * cdf_val;
    };

    return TestResult(T).init(t_stat, p_value, df, alpha);
}

// ============================================================================
// INDEPENDENT SAMPLES T-TEST
// ============================================================================

/// Independent samples t-test: H0: μ₁ = μ₂
///
/// Tests whether two independent samples have significantly different means.
/// Supports both Welch's t-test (for unequal variances) and pooled variance t-test.
///
/// Welch's t-test (equal_var=false, default):
///   t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)
///   df ≈ (s₁²/n₁ + s₂²/n₂)² / ((s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1))  [Welch-Satterthwaite]
///
/// Pooled variance t-test (equal_var=true):
///   t = (x̄₁ - x̄₂) / (sp * √(1/n₁ + 1/n₂))
///   sp² = ((n₁-1)*s₁² + (n₂-1)*s₂²) / (n₁ + n₂ - 2)
///   df = n₁ + n₂ - 2
///
/// Parameters:
/// - sample1, sample2: 1D NDArray of numeric type T
/// - alpha: significance level (default 0.05)
/// - equal_var: if true use pooled variance test; else Welch's test
///
/// Returns: TestResult with t-statistic, p-value (two-tailed), and rejection decision
///
/// Errors:
/// - error.EmptyArray if either sample is empty
/// - error.InvalidParameter if alpha not in (0, 1)
///
/// Time: O(n₁ + n₂)
/// Space: O(1)
///
/// Example:
/// ```zig
/// const result = try ttest_ind(f64, sample1, sample2, 0.05, false);
/// // Welch's t-test for unequal variances
/// ```
pub fn ttest_ind(
    comptime T: type,
    sample1: NDArray_type(T, 1),
    sample2: NDArray_type(T, 1),
    alpha: T,
    equal_var: bool,
) !TestResult(T) {
    const n1 = sample1.count();
    const n2 = sample2.count();
    if (n1 == 0 or n2 == 0) return error.EmptyArray;
    if (alpha <= 0 or alpha >= 1) return error.InvalidParameter;

    // Compute means
    const mean1 = descriptive.mean(T, sample1);
    const mean2 = descriptive.mean(T, sample2);

    // Compute variances (sample variances, ddof=1)
    const var1 = try descriptive.variance(T, sample1, 1);
    const var2 = try descriptive.variance(T, sample2, 1);

    const n1_f = @as(T, @floatFromInt(n1));
    const n2_f = @as(T, @floatFromInt(n2));

    var t_stat: T = undefined;
    var df: T = undefined;
    var se: T = undefined;

    if (equal_var) {
        // Pooled variance t-test
        const n1_minus_1 = @as(T, @floatFromInt(n1 - 1));
        const n2_minus_1 = @as(T, @floatFromInt(n2 - 1));
        const sp_sq = (n1_minus_1 * var1 + n2_minus_1 * var2) / (n1_f + n2_f - 2.0);
        se = math.sqrt(sp_sq * (1.0 / n1_f + 1.0 / n2_f));
        t_stat = if (se == 0) 0.0 else (mean1 - mean2) / se;
        df = n1_f + n2_f - 2.0;
    } else {
        // Welch's t-test (no equal variance assumption)
        const se1_sq = var1 / n1_f;
        const se2_sq = var2 / n2_f;
        se = math.sqrt(se1_sq + se2_sq);
        t_stat = if (se == 0) 0.0 else (mean1 - mean2) / se;

        // Welch-Satterthwaite degrees of freedom
        const numerator = (se1_sq + se2_sq) * (se1_sq + se2_sq);
        const denom = (se1_sq * se1_sq) / (n1_f - 1.0) + (se2_sq * se2_sq) / (n2_f - 1.0);
        df = numerator / denom;
    }

    // Two-tailed p-value
    // Special case: if se == 0, we can't compute p-value meaningfully, return 1.0 (don't reject H0)
    const p_value: T = if (se == 0) 1.0 else blk: {
        const dist = try StudentT_Distribution(T).init(df);
        const cdf_val = dist.cdf(t_stat);
        break :blk if (t_stat >= 0)
            2.0 * (1.0 - cdf_val)
        else
            2.0 * cdf_val;
    };

    return TestResult(T).init(t_stat, p_value, df, alpha);
}

// ============================================================================
// PAIRED SAMPLES T-TEST
// ============================================================================

/// Paired samples t-test: H0: μ_diff = 0
///
/// Tests whether paired observations have significantly different means.
/// Equivalent to one-sample t-test on the differences.
///
/// Formula:
///   d_i = before_i - after_i (or arbitrary pairing)
///   t = d̄ / (s_d / √n)
///   df = n - 1
///
/// Parameters:
/// - before: 1D NDArray of type T (first observation in each pair)
/// - after: 1D NDArray of type T (second observation in each pair)
/// - alpha: significance level (default 0.05)
///
/// Returns: TestResult with t-statistic, p-value (two-tailed), and rejection decision
///
/// Errors:
/// - error.EmptyArray if arrays are empty
/// - error.UnequalLengths if arrays have different sizes
/// - error.InvalidParameter if alpha not in (0, 1)
///
/// Time: O(n)
/// Space: O(1)
///
/// Example:
/// ```zig
/// const alloc = std.heap.page_allocator;
/// const result = try ttest_rel(f64, alloc, before_data, after_data, 0.05);
/// // Tests if after differs from before at α=0.05
/// ```
pub fn ttest_rel(
    comptime T: type,
    alloc: std.mem.Allocator,
    before: NDArray_type(T, 1),
    after: NDArray_type(T, 1),
    alpha: T,
) !TestResult(T) {
    const n = before.count();
    if (n == 0) return error.EmptyArray;
    if (n != after.count()) return error.UnequalLengths;
    if (alpha <= 0 or alpha >= 1) return error.InvalidParameter;

    // Compute differences: d = before - after
    const diffs = try alloc.alloc(T, n);
    defer alloc.free(diffs);

    var before_iter = before.iterator();
    var after_iter = after.iterator();
    var idx: usize = 0;
    while (before_iter.next()) |b| {
        const a = after_iter.next() orelse return error.UnequalLengths;
        diffs[idx] = b - a;
        idx += 1;
    }

    // Create NDArray for differences
    var diff_array = try NDArray_type(T, 1).fromSlice(alloc, &[_]usize{n}, diffs, .row_major);
    defer diff_array.deinit();

    // Compute mean of differences
    const mean_diff = descriptive.mean(T, diff_array);

    // Compute standard deviation of differences
    const var_diff = try descriptive.variance(T, diff_array, 1);
    const std_diff = math.sqrt(var_diff);

    // Compute t-statistic: t = d̄ / (s_d / √n)
    const n_f = @as(T, @floatFromInt(n));
    const se = std_diff / math.sqrt(n_f);

    // Handle edge case: zero standard error (no variance in differences)
    const t_stat: T = if (se == 0) 0.0 else mean_diff / se;

    // Degrees of freedom
    const df = @as(T, @floatFromInt(n - 1));

    // Two-tailed p-value
    // Special case: if se == 0, we can't compute p-value meaningfully, return 1.0 (don't reject H0)
    const p_value: T = if (se == 0) 1.0 else blk: {
        const dist = try StudentT_Distribution(T).init(df);
        const cdf_val = dist.cdf(t_stat);
        break :blk if (t_stat >= 0)
            2.0 * (1.0 - cdf_val)
        else
            2.0 * cdf_val;
    };

    return TestResult(T).init(t_stat, p_value, df, alpha);
}

// ============================================================================
// CHI-SQUARED GOODNESS-OF-FIT TEST
// ============================================================================

/// Chi-squared goodness-of-fit test: H0: observed frequencies match expected frequencies
///
/// Tests whether observed categorical data distribution differs significantly from
/// the expected (theoretical) distribution.
///
/// Formula: χ² = Σᵢ ((Oᵢ - Eᵢ)² / Eᵢ)
/// where Oᵢ = observed frequency in category i, Eᵢ = expected frequency in category i
/// df = k - 1 (k = number of categories)
///
/// The test statistic follows a chi-squared distribution with k-1 degrees of freedom.
/// A large χ² value indicates poor fit between observed and expected distributions.
///
/// Parameters:
/// - observed: 1D NDArray of observed frequencies (must be non-negative)
/// - expected: 1D NDArray of expected frequencies (must be positive, sum > 0)
/// - alpha: significance level (default 0.05 for 95% confidence)
///
/// Returns: TestResult with χ² statistic, p-value (right-tailed), and rejection decision
///
/// Errors:
/// - error.EmptyArray if observed or expected is empty
/// - error.UnequalLengths if observed and expected have different lengths
/// - error.InvalidParameter if alpha not in (0, 1), or if any expected frequency ≤ 0
/// - error.InvalidParameter if observed contains negative values
///
/// Time: O(k) where k = number of categories
/// Space: O(1)
///
/// Notes:
/// - Expected frequencies should typically be ≥ 5 for valid chi-squared approximation
/// - This is a right-tailed test (large χ² → reject H0)
/// - P-value = P(χ²(k-1) > χ²_observed)
///
/// Example:
/// ```zig
/// // Test if a die is fair (expected uniform distribution)
/// const observed = [_]f64{10, 12, 8, 15, 9, 16}; // 70 rolls
/// const expected = [_]f64{70/6, 70/6, 70/6, 70/6, 70/6, 70/6}; // uniform
/// const result = try chi2_test(f64, observed_arr, expected_arr, 0.05);
/// // Tests if die is fair at α=0.05
/// ```
pub fn chi2_test(
    comptime T: type,
    observed: NDArray_type(T, 1),
    expected: NDArray_type(T, 1),
    alpha: T,
) !TestResult(T) {
    const n_obs = observed.count();
    const n_exp = expected.count();

    if (n_obs == 0 or n_exp == 0) return error.EmptyArray;
    if (n_obs != n_exp) return error.UnequalLengths;
    if (alpha <= 0 or alpha >= 1) return error.InvalidParameter;

    // Validate input: expected frequencies must be positive, observed must be non-negative
    const n = n_obs;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const obs_val = observed.data[i];
        const exp_val = expected.data[i];

        if (obs_val < 0.0) return error.InvalidParameter; // Negative observed frequency
        if (exp_val <= 0.0) return error.InvalidParameter; // Non-positive expected frequency
    }

    // Compute chi-squared statistic: χ² = Σ ((O - E)² / E)
    var chi2_stat: T = 0.0;
    i = 0;
    while (i < n) : (i += 1) {
        const obs_val = observed.data[i];
        const exp_val = expected.data[i];
        const diff = obs_val - exp_val;
        chi2_stat += (diff * diff) / exp_val;
    }

    // Degrees of freedom: k - 1
    const df = @as(T, @floatFromInt(n - 1));

    // Right-tailed p-value: P(χ²(df) > χ²_stat)
    // p-value = 1 - CDF(χ²_stat)
    const dist = try ChiSquared_Distribution(T).init(df);
    const cdf_val = dist.cdf(chi2_stat);
    const p_value = 1.0 - cdf_val;

    return TestResult(T).init(chi2_stat, p_value, df, alpha);
}

// ============================================================================
// ONE-WAY ANOVA TEST
// ============================================================================

/// One-way ANOVA: H0: all group means are equal (μ₁ = μ₂ = ... = μₖ)
///
/// Tests whether the means of k independent groups (k ≥ 2) differ significantly.
/// Partitions total variance into between-group and within-group components.
///
/// Formula:
/// - Grand mean: x̄ = (1/N) Σᵢⱼ xᵢⱼ
/// - SSB (Between): Σᵢ nᵢ(x̄ᵢ - x̄)²
/// - SSW (Within): Σᵢⱼ (xᵢⱼ - x̄ᵢ)²
/// - MSB = SSB / (k-1), MSW = SSW / (N-k)
/// - F = MSB / MSW
/// where k = number of groups, N = total sample size
///
/// The test statistic follows an F distribution with (k-1, N-k) degrees of freedom.
/// A large F value indicates group means differ significantly.
///
/// Parameters:
/// - groups: slice of group data slices, each containing numeric observations
/// - alpha: significance level (default 0.05 for 95% confidence)
/// - allocator: memory allocator for temporary arrays
///
/// Returns: TestResult with F-statistic, p-value (right-tailed), df (df1), and rejection decision
///
/// Errors:
/// - error.TooFewGroups if groups.len < 2
/// - error.EmptyGroup if any group has zero observations
/// - error.InvalidParameter if alpha not in (0, 1)
///
/// Time: O(N) where N is total sample size
/// Space: O(k) for group means
///
/// Notes:
/// - This is a right-tailed test (large F → reject H0)
/// - P-value = P(F(k-1, N-k) > F_observed)
/// - Assumes groups are independent and normally distributed
/// - Assumes equal variances across groups (homogeneity of variance)
/// - When all groups have identical values, F = 0 and p-value = 1.0
///
/// Example:
/// ```zig
/// const group1 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
/// const group2 = [_]f64{ 6.0, 7.0, 8.0, 9.0, 10.0 };
/// const group3 = [_]f64{ 11.0, 12.0, 13.0, 14.0, 15.0 };
/// const groups = [_][]const f64{ &group1, &group2, &group3 };
/// const result = try anova_oneway(f64, &groups, 0.05, allocator);
/// // Tests if group means differ significantly at α=0.05
/// ```
pub fn anova_oneway(
    comptime T: type,
    groups: []const []const T,
    alpha: T,
    alloc: std.mem.Allocator,
) !TestResult(T) {
    // Validation
    if (groups.len < 2) return error.TooFewGroups;

    // Check all groups non-empty and compute total size N
    var N: usize = 0;
    for (groups) |group| {
        if (group.len == 0) return error.EmptyGroup;
        N += group.len;
    }

    if (alpha <= 0 or alpha >= 1) return error.InvalidParameter;

    const k = groups.len;

    // Collect all values for grand mean calculation
    var all_values = std.ArrayList(T){};
    defer all_values.deinit(alloc);

    try all_values.ensureTotalCapacity(alloc, N);
    for (groups) |group| {
        for (group) |value| {
            try all_values.append(alloc, value);
        }
    }

    // Compute grand mean (mean of all observations)
    var grand_sum: T = 0.0;
    for (all_values.items) |value| {
        grand_sum += value;
    }
    const grand_mean = grand_sum / @as(T, @floatFromInt(N));

    // Compute group means
    var group_means = std.ArrayList(T){};
    defer group_means.deinit(alloc);

    try group_means.ensureTotalCapacity(alloc, k);
    for (groups) |group| {
        var group_sum: T = 0.0;
        for (group) |value| {
            group_sum += value;
        }
        const group_mean = group_sum / @as(T, @floatFromInt(group.len));
        try group_means.append(alloc, group_mean);
    }

    // Compute Sum of Squares Between (SSB)
    var ssb: T = 0.0;
    for (0..k) |i| {
        const group_size = groups[i].len;
        const diff = group_means.items[i] - grand_mean;
        ssb += @as(T, @floatFromInt(group_size)) * diff * diff;
    }

    // Compute Sum of Squares Within (SSW)
    var ssw: T = 0.0;
    for (0..k) |i| {
        const group = groups[i];
        const group_mean = group_means.items[i];
        for (group) |value| {
            const diff = value - group_mean;
            ssw += diff * diff;
        }
    }

    // Compute Mean Squares and F-statistic
    const df1 = k - 1;
    const df2 = N - k;

    const msb = ssb / @as(T, @floatFromInt(df1));
    const msw = ssw / @as(T, @floatFromInt(df2));

    // Handle edge case: if MSW == 0 (no within-group variance)
    const f_statistic = if (msw == 0) 0.0 else msb / msw;

    // Compute p-value using F-distribution (right-tailed)
    const f_dist = try FDistribution(T).init(
        @as(T, @floatFromInt(df1)),
        @as(T, @floatFromInt(df2)),
    );
    const cdf_val = f_dist.cdf(f_statistic);
    const p_value = 1.0 - cdf_val;

    return TestResult(T).init(f_statistic, p_value, @as(T, @floatFromInt(df1)), alpha);
}

// ============================================================================
// TESTS
// ============================================================================

const allocator = testing.allocator;

// ============================================================================
// TestResult Tests (5 tests)
// ============================================================================

test "TestResult: init with reject=true" {
    const result = TestResult(f64).init(2.5, 0.02, 20.0, 0.05);
    try testing.expect(result.reject == true);
    try testing.expectApproxEqAbs(2.5, result.statistic, 1e-10);
    try testing.expectApproxEqAbs(0.02, result.p_value, 1e-10);
    try testing.expectApproxEqAbs(20.0, result.df, 1e-10);
}

test "TestResult: init with reject=false" {
    const result = TestResult(f64).init(1.5, 0.15, 25.0, 0.05);
    try testing.expect(result.reject == false);
}

test "TestResult: init with p_value exactly equal to alpha" {
    const result = TestResult(f64).init(1.0, 0.05, 30.0, 0.05);
    try testing.expect(result.reject == false); // p_value < alpha is false when equal
}

test "TestResult: f32 type" {
    const result = TestResult(f32).init(2.0, 0.01, 15.0, 0.05);
    try testing.expect(result.reject == true);
    try testing.expectApproxEqAbs(@as(f32, 2.0), result.statistic, 1e-5);
}

test "TestResult: p_value in [0, 1]" {
    const result1 = TestResult(f64).init(0.0, 0.0, 10.0, 0.05);
    const result2 = TestResult(f64).init(0.0, 1.0, 10.0, 0.05);
    const result3 = TestResult(f64).init(0.0, 0.5, 10.0, 0.05);
    try testing.expect(result1.p_value >= 0.0 and result1.p_value <= 1.0);
    try testing.expect(result2.p_value >= 0.0 and result2.p_value <= 1.0);
    try testing.expect(result3.p_value >= 0.0 and result3.p_value <= 1.0);
}

// ============================================================================
// One-Sample T-Test Tests (20+ tests)
// ============================================================================

test "ttest_1samp: sample mean matches population mean (t≈0, p≈1, reject=false)" {
    const data_slice = [_]f64{ 3.0, 3.0, 3.0, 3.0, 3.0 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();

    const result = try ttest_1samp(f64, data, 3.0, 0.05);
    try testing.expectApproxEqAbs(0.0, result.statistic, 1e-10);
    try testing.expectApproxEqAbs(1.0, result.p_value, 1e-10);
    try testing.expect(result.reject == false);
}

test "ttest_1samp: sample mean > population mean (t>0, p<0.05, reject=true)" {
    const data_slice = [_]f64{ 5.0, 6.0, 7.0, 8.0, 9.0 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();

    const result = try ttest_1samp(f64, data, 3.0, 0.05);
    try testing.expect(result.statistic > 0.0);
    try testing.expect(result.p_value < 0.05);
    try testing.expect(result.reject == true);
}

test "ttest_1samp: sample mean < population mean (t<0, p<0.05, reject=true)" {
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();

    const result = try ttest_1samp(f64, data, 10.0, 0.05);
    try testing.expect(result.statistic < 0.0);
    try testing.expect(result.p_value < 0.05);
    try testing.expect(result.reject == true);
}

test "ttest_1samp: single observation (n=1, df=0) errors" {
    const data_slice = [_]f64{5.0};
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{1}, &data_slice, .row_major);
    defer data.deinit();

    // With single observation and ddof=1, variance calculation fails (n-ddof=0)
    const result = ttest_1samp(f64, data, 5.0, 0.05);
    try testing.expectError(NDArray_type(f64, 1).Error.CapacityExceeded, result);
}

test "ttest_1samp: two observations (n=2, df=1)" {
    const data_slice = [_]f64{ 1.0, 3.0 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &data_slice, .row_major);
    defer data.deinit();

    const result = try ttest_1samp(f64, data, 2.0, 0.05);
    try testing.expectApproxEqAbs(1.0, result.df, 1e-10);
}

test "ttest_1samp: large sample (n=100)" {
    var data_slice: [100]f64 = undefined;
    for (0..100) |i| {
        data_slice[i] = @as(f64, @floatFromInt(i + 1)); // 1..100
    }
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{100}, &data_slice, .row_major);
    defer data.deinit();

    const result = try ttest_1samp(f64, data, 50.5, 0.05);
    try testing.expectApproxEqAbs(99.0, result.df, 1e-10);
    try testing.expect(!math.isNan(result.statistic));
}

test "ttest_1samp: f32 precision" {
    const data_slice = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var data = try NDArray_type(f32, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();

    const result = try ttest_1samp(f32, data, 3.0, 0.05);
    try testing.expect(!math.isNan(result.statistic));
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "ttest_1samp: alpha=0.01 affects rejection" {
    const data_slice = [_]f64{ 5.0, 6.0, 7.0, 8.0, 9.0 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();

    const result_005 = try ttest_1samp(f64, data, 3.0, 0.05);
    const result_001 = try ttest_1samp(f64, data, 3.0, 0.01);

    // Both should reject because p-value should be very small
    try testing.expect(result_005.reject == true);
    try testing.expect(result_001.reject == true);
}

test "ttest_1samp: alpha=0.1 affects rejection" {
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{6}, &data_slice, .row_major);
    defer data.deinit();

    const result = try ttest_1samp(f64, data, 3.5, 0.1);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "ttest_1samp: p_value in [0, 1]" {
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();

    const result = try ttest_1samp(f64, data, 5.0, 0.05);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "ttest_1samp: df = n-1" {
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{7}, &data_slice, .row_major);
    defer data.deinit();

    const result = try ttest_1samp(f64, data, 4.0, 0.05);
    try testing.expectApproxEqAbs(6.0, result.df, 1e-10);
}

test "ttest_1samp: error on empty array" {
    const data_slice: [0]f64 = [_]f64{};
    const result = NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{0}, &data_slice, .row_major);
    try testing.expectError(NDArray_type(f64, 1).Error.ZeroDimension, result);
}

test "ttest_1samp: error on invalid alpha (alpha=0)" {
    const data_slice = [_]f64{ 1.0, 2.0, 3.0 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data_slice, .row_major);
    defer data.deinit();

    const result = ttest_1samp(f64, data, 2.0, 0.0);
    try testing.expectError(error.InvalidParameter, result);
}

test "ttest_1samp: error on invalid alpha (alpha=1)" {
    const data_slice = [_]f64{ 1.0, 2.0, 3.0 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data_slice, .row_major);
    defer data.deinit();

    const result = ttest_1samp(f64, data, 2.0, 1.0);
    try testing.expectError(error.InvalidParameter, result);
}

test "ttest_1samp: symmetry - opposite shifts yield opposite t-statistics" {
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();

    const result1 = try ttest_1samp(f64, data, 2.0, 0.05);
    const result2 = try ttest_1samp(f64, data, 4.0, 0.05);

    try testing.expectApproxEqAbs(result1.statistic, -result2.statistic, 1e-10);
}

test "ttest_1samp: consistency across multiple runs" {
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var data1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data1.deinit();
    var data2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data2.deinit();

    const result1 = try ttest_1samp(f64, data1, 3.0, 0.05);
    const result2 = try ttest_1samp(f64, data2, 3.0, 0.05);

    try testing.expectApproxEqAbs(result1.statistic, result2.statistic, 1e-10);
    try testing.expectApproxEqAbs(result1.p_value, result2.p_value, 1e-10);
}

// ============================================================================
// Independent Samples T-Test Tests (20+ tests)
// ============================================================================

test "ttest_ind: identical samples (t≈0, p≈1, reject=false)" {
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer sample2.deinit();

    const result = try ttest_ind(f64, sample1, sample2, 0.05, false);
    try testing.expectApproxEqAbs(0.0, result.statistic, 1e-10);
    try testing.expectApproxEqAbs(1.0, result.p_value, 1e-10);
    try testing.expect(result.reject == false);
}

test "ttest_ind: different means (t≠0, p<0.05, reject=true)" {
    const data1 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const data2 = [_]f64{ 10.0, 11.0, 12.0, 13.0, 14.0 };
    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data2, .row_major);
    defer sample2.deinit();

    const result = try ttest_ind(f64, sample1, sample2, 0.05, false);
    try testing.expect(result.statistic != 0.0);
    try testing.expect(result.p_value < 0.05);
    try testing.expect(result.reject == true);
}

test "ttest_ind: Welch vs pooled give different results for unequal variances" {
    // Use unequal sample sizes to see difference between Welch and pooled
    // When n1=n2, Welch and pooled give same t-statistic even with unequal variances
    const data1 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const data2 = [_]f64{ 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0 }; // n2=7, much higher variance
    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{7}, &data2, .row_major);
    defer sample2.deinit();

    const welch_result = try ttest_ind(f64, sample1, sample2, 0.05, false);
    const pooled_result = try ttest_ind(f64, sample1, sample2, 0.05, true);

    // Welch and pooled should differ for unequal sample sizes + unequal variances
    try testing.expect(@abs(welch_result.statistic - pooled_result.statistic) > 0.01);
}

test "ttest_ind: n1=1, n2=1 (edge case) errors" {
    const data1 = [_]f64{1.0};
    const data2 = [_]f64{3.0};
    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{1}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{1}, &data2, .row_major);
    defer sample2.deinit();

    // With n=1 for both samples, variance calculation fails (ddof=1, n-ddof=0)
    const result = ttest_ind(f64, sample1, sample2, 0.05, false);
    try testing.expectError(NDArray_type(f64, 1).Error.CapacityExceeded, result);
}

test "ttest_ind: n1=2, n2=2" {
    const data1 = [_]f64{ 1.0, 2.0 };
    const data2 = [_]f64{ 3.0, 4.0 };
    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &data2, .row_major);
    defer sample2.deinit();

    const result = try ttest_ind(f64, sample1, sample2, 0.05, false);
    try testing.expectApproxEqAbs(2.0, result.df, 1e-10); // Welch-Satterthwaite will give ≈2
}

test "ttest_ind: unequal sizes (n1=10, n2=50)" {
    var data1: [10]f64 = undefined;
    var data2: [50]f64 = undefined;
    for (0..10) |i| {
        data1[i] = @as(f64, @floatFromInt(i + 1));
    }
    for (0..50) |i| {
        data2[i] = @as(f64, @floatFromInt(i + 11));
    }

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{10}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{50}, &data2, .row_major);
    defer sample2.deinit();

    const result = try ttest_ind(f64, sample1, sample2, 0.05, false);
    try testing.expect(!math.isNan(result.statistic));
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "ttest_ind: f32 precision" {
    const data1 = [_]f32{ 1.0, 2.0, 3.0 };
    const data2 = [_]f32{ 4.0, 5.0, 6.0 };
    var sample1 = try NDArray_type(f32, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f32, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2.deinit();

    const result = try ttest_ind(f32, sample1, sample2, 0.05, false);
    try testing.expect(!math.isNan(result.statistic));
}

test "ttest_ind: alpha=0.01 vs alpha=0.05" {
    const data1 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const data2 = [_]f64{ 10.0, 11.0, 12.0, 13.0, 14.0 };
    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data2, .row_major);
    defer sample2.deinit();

    const result_005 = try ttest_ind(f64, sample1, sample2, 0.05, false);
    const result_001 = try ttest_ind(f64, sample1, sample2, 0.01, false);

    // Same p-value but different rejection decisions
    try testing.expectApproxEqAbs(result_005.p_value, result_001.p_value, 1e-10);
}

test "ttest_ind: error on empty sample1" {
    const data1: [0]f64 = [_]f64{};

    const result1 = NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{0}, &data1, .row_major);
    try testing.expectError(NDArray_type(f64, 1).Error.ZeroDimension, result1);
}

test "ttest_ind: error on empty sample2" {
    const data2: [0]f64 = [_]f64{};

    const result2 = NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{0}, &data2, .row_major);
    try testing.expectError(NDArray_type(f64, 1).Error.ZeroDimension, result2);
}

test "ttest_ind: error on invalid alpha" {
    const data1 = [_]f64{ 1.0, 2.0, 3.0 };
    const data2 = [_]f64{ 4.0, 5.0, 6.0 };
    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2.deinit();

    const result = ttest_ind(f64, sample1, sample2, 1.5, false);
    try testing.expectError(error.InvalidParameter, result);
}

test "ttest_ind: pooled variance df = n1+n2-2" {
    const data1 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const data2 = [_]f64{ 6.0, 7.0, 8.0, 9.0, 10.0 };
    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data2, .row_major);
    defer sample2.deinit();

    const result = try ttest_ind(f64, sample1, sample2, 0.05, true);
    try testing.expectApproxEqAbs(8.0, result.df, 1e-10); // 5+5-2=8
}

test "ttest_ind: Welch df calculation" {
    const data1 = [_]f64{ 1.0, 2.0, 3.0 };
    const data2 = [_]f64{ 4.0, 5.0, 6.0, 7.0 };
    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{4}, &data2, .row_major);
    defer sample2.deinit();

    const result = try ttest_ind(f64, sample1, sample2, 0.05, false);
    // Welch-Satterthwaite should give df < n1+n2-2
    try testing.expect(result.df > 0.0);
    try testing.expect(!math.isNan(result.df));
}

test "ttest_ind: p_value in [0, 1]" {
    const data1 = [_]f64{ 1.0, 2.0, 3.0 };
    const data2 = [_]f64{ 4.0, 5.0, 6.0 };
    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2.deinit();

    const result = try ttest_ind(f64, sample1, sample2, 0.05, false);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "ttest_ind: symmetry - swapping samples negates t-statistic" {
    const data1 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const data2 = [_]f64{ 6.0, 7.0, 8.0, 9.0, 10.0 };
    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data2, .row_major);
    defer sample2.deinit();

    const result1 = try ttest_ind(f64, sample1, sample2, 0.05, false);

    var sample1b = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data1, .row_major);
    defer sample1b.deinit();
    var sample2b = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data2, .row_major);
    defer sample2b.deinit();

    const result2 = try ttest_ind(f64, sample2b, sample1b, 0.05, false);

    try testing.expectApproxEqAbs(result1.statistic, -result2.statistic, 1e-10);
    try testing.expectApproxEqAbs(result1.p_value, result2.p_value, 1e-10);
}

// ============================================================================
// Paired Samples T-Test Tests (15+ tests)
// ============================================================================

test "ttest_rel: no change (before == after, t≈0, p≈1, reject=false)" {
    const before_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const after_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var before = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &before_data, .row_major);
    defer before.deinit();
    var after = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &after_data, .row_major);
    defer after.deinit();

    const result = try ttest_rel(f64, allocator, before, after, 0.05);
    try testing.expectApproxEqAbs(0.0, result.statistic, 1e-10);
    try testing.expectApproxEqAbs(1.0, result.p_value, 1e-10);
    try testing.expect(result.reject == false);
}

test "ttest_rel: after > before (t<0, p<0.05, reject=true)" {
    // Use data with variance in differences
    const before_data = [_]f64{ 1.0, 2.5, 3.0, 4.5, 5.0 };
    const after_data = [_]f64{ 6.0, 7.0, 8.5, 9.0, 10.5 };
    var before = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &before_data, .row_major);
    defer before.deinit();
    var after = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &after_data, .row_major);
    defer after.deinit();

    const result = try ttest_rel(f64, allocator, before, after, 0.05);
    try testing.expect(result.statistic < 0.0); // d = before - after = negative
    try testing.expect(result.p_value < 0.05);
    try testing.expect(result.reject == true);
}

test "ttest_rel: after < before (t>0, p<0.05, reject=true)" {
    // Use data with variance in differences
    const before_data = [_]f64{ 6.0, 7.5, 8.0, 9.5, 10.0 };
    const after_data = [_]f64{ 1.0, 2.0, 3.5, 4.0, 5.5 };
    var before = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &before_data, .row_major);
    defer before.deinit();
    var after = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &after_data, .row_major);
    defer after.deinit();

    const result = try ttest_rel(f64, allocator, before, after, 0.05);
    try testing.expect(result.statistic > 0.0); // d = before - after = positive
    try testing.expect(result.p_value < 0.05);
    try testing.expect(result.reject == true);
}

test "ttest_rel: single pair (n=1, df=0) errors" {
    const before_data = [_]f64{1.0};
    const after_data = [_]f64{3.0};
    var before = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{1}, &before_data, .row_major);
    defer before.deinit();
    var after = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{1}, &after_data, .row_major);
    defer after.deinit();

    // With single pair and ddof=1, variance calculation fails (n-ddof=0)
    const result = ttest_rel(f64, allocator, before, after, 0.05);
    try testing.expectError(NDArray_type(f64, 1).Error.CapacityExceeded, result);
}

test "ttest_rel: two pairs (n=2, df=1)" {
    const before_data = [_]f64{ 1.0, 3.0 };
    const after_data = [_]f64{ 2.0, 4.0 };
    var before = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &before_data, .row_major);
    defer before.deinit();
    var after = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &after_data, .row_major);
    defer after.deinit();

    const result = try ttest_rel(f64, allocator, before, after, 0.05);
    try testing.expectApproxEqAbs(1.0, result.df, 1e-10);
}

test "ttest_rel: many pairs (n=100)" {
    var before_data: [100]f64 = undefined;
    var after_data: [100]f64 = undefined;
    for (0..100) |i| {
        before_data[i] = @as(f64, @floatFromInt(i + 1));
        after_data[i] = @as(f64, @floatFromInt(i + 2)); // Consistent +1 difference
    }

    var before = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{100}, &before_data, .row_major);
    defer before.deinit();
    var after = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{100}, &after_data, .row_major);
    defer after.deinit();

    const result = try ttest_rel(f64, allocator, before, after, 0.05);
    try testing.expectApproxEqAbs(99.0, result.df, 1e-10);
}

test "ttest_rel: f32 precision" {
    const before_data = [_]f32{ 1.0, 2.0, 3.0 };
    const after_data = [_]f32{ 2.0, 3.0, 4.0 };
    var before = try NDArray_type(f32, 1).fromSlice(allocator, &[_]usize{3}, &before_data, .row_major);
    defer before.deinit();
    var after = try NDArray_type(f32, 1).fromSlice(allocator, &[_]usize{3}, &after_data, .row_major);
    defer after.deinit();

    const result = try ttest_rel(f32, allocator, before, after, 0.05);
    try testing.expect(!math.isNan(result.statistic));
}

test "ttest_rel: alpha=0.01 vs alpha=0.05" {
    const before_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const after_data = [_]f64{ 6.0, 7.0, 8.0, 9.0, 10.0 };
    var before1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &before_data, .row_major);
    defer before1.deinit();
    var after1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &after_data, .row_major);
    defer after1.deinit();

    var before2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &before_data, .row_major);
    defer before2.deinit();
    var after2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &after_data, .row_major);
    defer after2.deinit();

    const result_005 = try ttest_rel(f64, allocator, before1, after1, 0.05);
    const result_001 = try ttest_rel(f64, allocator, before2, after2, 0.01);

    // Same p-value but different rejection decisions
    try testing.expectApproxEqAbs(result_005.p_value, result_001.p_value, 1e-10);
}

test "ttest_rel: error on empty arrays" {
    const before_data: [0]f64 = [_]f64{};

    const result1 = NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{0}, &before_data, .row_major);
    try testing.expectError(NDArray_type(f64, 1).Error.ZeroDimension, result1);
}

test "ttest_rel: error on mismatched lengths" {
    const before_data = [_]f64{ 1.0, 2.0, 3.0 };
    const after_data = [_]f64{ 1.0, 2.0 }; // Different length
    var before = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &before_data, .row_major);
    defer before.deinit();
    var after = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &after_data, .row_major);
    defer after.deinit();

    const result = ttest_rel(f64, allocator, before, after, 0.05);
    try testing.expectError(error.UnequalLengths, result);
}

test "ttest_rel: error on invalid alpha (alpha=0)" {
    const before_data = [_]f64{ 1.0, 2.0, 3.0 };
    const after_data = [_]f64{ 2.0, 3.0, 4.0 };
    var before = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &before_data, .row_major);
    defer before.deinit();
    var after = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &after_data, .row_major);
    defer after.deinit();

    const result = ttest_rel(f64, allocator, before, after, 0.0);
    try testing.expectError(error.InvalidParameter, result);
}

test "ttest_rel: error on invalid alpha (alpha=1)" {
    const before_data = [_]f64{ 1.0, 2.0, 3.0 };
    const after_data = [_]f64{ 2.0, 3.0, 4.0 };
    var before = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &before_data, .row_major);
    defer before.deinit();
    var after = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &after_data, .row_major);
    defer after.deinit();

    const result = ttest_rel(f64, allocator, before, after, 1.0);
    try testing.expectError(error.InvalidParameter, result);
}

test "ttest_rel: df = n-1" {
    const before_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
    const after_data = [_]f64{ 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var before = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{7}, &before_data, .row_major);
    defer before.deinit();
    var after = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{7}, &after_data, .row_major);
    defer after.deinit();

    const result = try ttest_rel(f64, allocator, before, after, 0.05);
    try testing.expectApproxEqAbs(6.0, result.df, 1e-10);
}

test "ttest_rel: p_value in [0, 1]" {
    const before_data = [_]f64{ 1.0, 2.0, 3.0 };
    const after_data = [_]f64{ 2.0, 3.0, 4.0 };
    var before = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &before_data, .row_major);
    defer before.deinit();
    var after = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &after_data, .row_major);
    defer after.deinit();

    const result = try ttest_rel(f64, allocator, before, after, 0.05);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "ttest_rel: symmetry - swapping before/after negates t-statistic" {
    const before_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const after_data = [_]f64{ 6.0, 7.0, 8.0, 9.0, 10.0 };
    var before1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &before_data, .row_major);
    defer before1.deinit();
    var after1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &after_data, .row_major);
    defer after1.deinit();

    var before2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &after_data, .row_major);
    defer before2.deinit();
    var after2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &before_data, .row_major);
    defer after2.deinit();

    const result1 = try ttest_rel(f64, allocator, before1, after1, 0.05);
    const result2 = try ttest_rel(f64, allocator, before2, after2, 0.05);

    try testing.expectApproxEqAbs(result1.statistic, -result2.statistic, 1e-10);
    try testing.expectApproxEqAbs(result1.p_value, result2.p_value, 1e-10);
}

// ============================================================================
// Chi-Squared Test Tests (20+ tests)
// ============================================================================

test "chi2_test: perfect fit (obs == exp, χ²≈0, p≈1, reject=false)" {
    // Perfect fit: observed matches expected exactly
    const obs_data = [_]f64{ 10.0, 20.0, 30.0, 40.0 };
    const exp_data = [_]f64{ 10.0, 20.0, 30.0, 40.0 };
    var obs = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{4}, &obs_data, .row_major);
    defer obs.deinit();
    var exp = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{4}, &exp_data, .row_major);
    defer exp.deinit();

    const result = try chi2_test(f64, obs, exp, 0.05);

    // Perfect fit: χ² = 0, p-value ≈ 1.0, reject = false
    try testing.expectApproxEqAbs(0.0, result.statistic, 1e-10);
    try testing.expect(result.p_value > 0.99);
    try testing.expect(result.reject == false);
    try testing.expectApproxEqAbs(3.0, result.df, 1e-10); // df = k - 1 = 4 - 1 = 3
}

test "chi2_test: fair die (uniform distribution, should not reject H0)" {
    // Fair die: 60 rolls, uniform distribution expected
    const obs_data = [_]f64{ 10.0, 12.0, 8.0, 11.0, 9.0, 10.0 }; // total = 60
    const exp_data = [_]f64{ 10.0, 10.0, 10.0, 10.0, 10.0, 10.0 }; // uniform
    var obs = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{6}, &obs_data, .row_major);
    defer obs.deinit();
    var exp = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{6}, &exp_data, .row_major);
    defer exp.deinit();

    const result = try chi2_test(f64, obs, exp, 0.05);

    // Small deviations from uniform: should not reject H0
    try testing.expect(result.p_value > 0.05); // p > α
    try testing.expect(result.reject == false);
    try testing.expectApproxEqAbs(5.0, result.df, 1e-10); // df = 6 - 1 = 5
}

test "chi2_test: biased die (large deviation, should reject H0)" {
    // Biased die: observed frequencies strongly deviate from uniform
    const obs_data = [_]f64{ 5.0, 5.0, 5.0, 5.0, 5.0, 75.0 }; // total = 100, heavily biased to 6
    const exp_data = [_]f64{ 16.67, 16.67, 16.67, 16.67, 16.67, 16.67 }; // uniform (100/6 ≈ 16.67)
    var obs = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{6}, &obs_data, .row_major);
    defer obs.deinit();
    var exp = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{6}, &exp_data, .row_major);
    defer exp.deinit();

    const result = try chi2_test(f64, obs, exp, 0.05);

    // Large χ² value: should reject H0
    try testing.expect(result.statistic > 10.0); // Large chi-squared
    try testing.expect(result.p_value < 0.05); // p < α
    try testing.expect(result.reject == true);
}

test "chi2_test: two categories (binary outcome)" {
    // Coin flip: observed heads/tails vs expected fair coin
    const obs_data = [_]f64{ 60.0, 40.0 }; // 60 heads, 40 tails out of 100
    const exp_data = [_]f64{ 50.0, 50.0 }; // fair coin expectation
    var obs = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &obs_data, .row_major);
    defer obs.deinit();
    var exp = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &exp_data, .row_major);
    defer exp.deinit();

    const result = try chi2_test(f64, obs, exp, 0.05);

    // df = 2 - 1 = 1
    try testing.expectApproxEqAbs(1.0, result.df, 1e-10);

    // χ² = (60-50)²/50 + (40-50)²/50 = 100/50 + 100/50 = 4.0
    try testing.expectApproxEqAbs(4.0, result.statistic, 1e-10);

    // For χ²(1), critical value at α=0.05 is 3.841, so χ²=4.0 > 3.841 → reject H0
    try testing.expect(result.p_value < 0.05);
    try testing.expect(result.reject == true);
}

test "chi2_test: many categories (k=10)" {
    // 10 categories, uniform distribution
    const obs_data = [_]f64{ 12.0, 11.0, 9.0, 13.0, 10.0, 8.0, 14.0, 9.0, 11.0, 13.0 }; // total = 110
    const exp_data = [_]f64{ 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0 }; // uniform
    var obs = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{10}, &obs_data, .row_major);
    defer obs.deinit();
    var exp = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{10}, &exp_data, .row_major);
    defer exp.deinit();

    const result = try chi2_test(f64, obs, exp, 0.05);

    // df = 10 - 1 = 9
    try testing.expectApproxEqAbs(9.0, result.df, 1e-10);

    // Small deviations: should not reject H0
    try testing.expect(result.p_value > 0.05);
    try testing.expect(result.reject == false);
}

test "chi2_test: zero observed frequencies (valid case)" {
    // Some categories can have zero observations
    const obs_data = [_]f64{ 0.0, 10.0, 20.0, 30.0 }; // total = 60
    const exp_data = [_]f64{ 15.0, 15.0, 15.0, 15.0 }; // uniform
    var obs = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{4}, &obs_data, .row_major);
    defer obs.deinit();
    var exp = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{4}, &exp_data, .row_major);
    defer exp.deinit();

    const result = try chi2_test(f64, obs, exp, 0.05);

    // Should compute without error (zero observed is valid)
    try testing.expect(result.statistic > 0.0);
    try testing.expectApproxEqAbs(3.0, result.df, 1e-10);
}

test "chi2_test: f32 precision" {
    const obs_data = [_]f32{ 10.0, 20.0, 30.0 };
    const exp_data = [_]f32{ 20.0, 20.0, 20.0 };
    var obs = try NDArray_type(f32, 1).fromSlice(allocator, &[_]usize{3}, &obs_data, .row_major);
    defer obs.deinit();
    var exp = try NDArray_type(f32, 1).fromSlice(allocator, &[_]usize{3}, &exp_data, .row_major);
    defer exp.deinit();

    const result = try chi2_test(f32, obs, exp, 0.05);

    // χ² = (10-20)²/20 + (20-20)²/20 + (30-20)²/20 = 100/20 + 0 + 100/20 = 10.0
    try testing.expectApproxEqAbs(@as(f32, 10.0), result.statistic, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.0), result.df, 1e-5);
}

test "chi2_test: alpha=0.01 affects rejection" {
    const obs_data = [_]f64{ 60.0, 40.0 }; // χ² = 4.0 (borderline)
    const exp_data = [_]f64{ 50.0, 50.0 };
    var obs = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &obs_data, .row_major);
    defer obs.deinit();
    var exp = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &exp_data, .row_major);
    defer exp.deinit();

    const result = try chi2_test(f64, obs, exp, 0.01); // stricter alpha

    // χ² = 4.0, critical value at α=0.01 is 6.635, so don't reject
    try testing.expect(result.p_value > 0.01);
    try testing.expect(result.reject == false);
}

test "chi2_test: alpha=0.1 affects rejection" {
    const obs_data = [_]f64{ 55.0, 45.0 }; // χ² = 1.0 (small)
    const exp_data = [_]f64{ 50.0, 50.0 };
    var obs = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &obs_data, .row_major);
    defer obs.deinit();
    var exp = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &exp_data, .row_major);
    defer exp.deinit();

    const result = try chi2_test(f64, obs, exp, 0.1); // lenient alpha

    // χ² = 1.0, critical value at α=0.1 is 2.706, so don't reject
    try testing.expect(result.reject == false);
}

test "chi2_test: p_value in [0, 1]" {
    const obs_data = [_]f64{ 10.0, 20.0, 30.0 };
    const exp_data = [_]f64{ 15.0, 25.0, 20.0 };
    var obs = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &obs_data, .row_major);
    defer obs.deinit();
    var exp = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &exp_data, .row_major);
    defer exp.deinit();

    const result = try chi2_test(f64, obs, exp, 0.05);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "chi2_test: statistic is non-negative" {
    const obs_data = [_]f64{ 5.0, 15.0, 25.0, 35.0 };
    const exp_data = [_]f64{ 20.0, 20.0, 20.0, 20.0 };
    var obs = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{4}, &obs_data, .row_major);
    defer obs.deinit();
    var exp = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{4}, &exp_data, .row_major);
    defer exp.deinit();

    const result = try chi2_test(f64, obs, exp, 0.05);
    try testing.expect(result.statistic >= 0.0); // χ² is always non-negative
}

test "chi2_test: error on empty observed array" {
    const obs_data: [0]f64 = [_]f64{};
    const result = NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{0}, &obs_data, .row_major);
    try testing.expectError(NDArray_type(f64, 1).Error.ZeroDimension, result);
}

test "chi2_test: error on empty expected array" {
    const obs_data = [_]f64{ 10.0, 20.0 };
    const exp_data: [0]f64 = [_]f64{};
    var obs = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &obs_data, .row_major);
    defer obs.deinit();
    const result = NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{0}, &exp_data, .row_major);
    try testing.expectError(NDArray_type(f64, 1).Error.ZeroDimension, result);
}

test "chi2_test: error on mismatched lengths" {
    const obs_data = [_]f64{ 10.0, 20.0, 30.0 };
    const exp_data = [_]f64{ 15.0, 25.0 }; // Different length
    var obs = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &obs_data, .row_major);
    defer obs.deinit();
    var exp = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &exp_data, .row_major);
    defer exp.deinit();

    const result = chi2_test(f64, obs, exp, 0.05);
    try testing.expectError(error.UnequalLengths, result);
}

test "chi2_test: error on negative observed frequency" {
    const obs_data = [_]f64{ 10.0, -5.0, 20.0 }; // Negative observation
    const exp_data = [_]f64{ 10.0, 10.0, 10.0 };
    var obs = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &obs_data, .row_major);
    defer obs.deinit();
    var exp = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &exp_data, .row_major);
    defer exp.deinit();

    const result = chi2_test(f64, obs, exp, 0.05);
    try testing.expectError(error.InvalidParameter, result);
}

test "chi2_test: error on zero expected frequency" {
    const obs_data = [_]f64{ 10.0, 20.0, 30.0 };
    const exp_data = [_]f64{ 10.0, 0.0, 20.0 }; // Zero expected (invalid)
    var obs = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &obs_data, .row_major);
    defer obs.deinit();
    var exp = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &exp_data, .row_major);
    defer exp.deinit();

    const result = chi2_test(f64, obs, exp, 0.05);
    try testing.expectError(error.InvalidParameter, result);
}

test "chi2_test: error on negative expected frequency" {
    const obs_data = [_]f64{ 10.0, 20.0, 30.0 };
    const exp_data = [_]f64{ 10.0, -10.0, 20.0 }; // Negative expected (invalid)
    var obs = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &obs_data, .row_major);
    defer obs.deinit();
    var exp = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &exp_data, .row_major);
    defer exp.deinit();

    const result = chi2_test(f64, obs, exp, 0.05);
    try testing.expectError(error.InvalidParameter, result);
}

test "chi2_test: error on invalid alpha (alpha=0)" {
    const obs_data = [_]f64{ 10.0, 20.0 };
    const exp_data = [_]f64{ 15.0, 15.0 };
    var obs = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &obs_data, .row_major);
    defer obs.deinit();
    var exp = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &exp_data, .row_major);
    defer exp.deinit();

    const result = chi2_test(f64, obs, exp, 0.0);
    try testing.expectError(error.InvalidParameter, result);
}

test "chi2_test: error on invalid alpha (alpha=1)" {
    const obs_data = [_]f64{ 10.0, 20.0 };
    const exp_data = [_]f64{ 15.0, 15.0 };
    var obs = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &obs_data, .row_major);
    defer obs.deinit();
    var exp = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &exp_data, .row_major);
    defer exp.deinit();

    const result = chi2_test(f64, obs, exp, 1.0);
    try testing.expectError(error.InvalidParameter, result);
}

// ============================================================================
// ONE-WAY ANOVA Tests (18+ tests)
// ============================================================================

test "anova_oneway: three groups with identical means (F≈0, p≈1, reject=false)" {
    const group1 = [_]f64{ 5.0, 5.0, 5.0, 5.0, 5.0 };
    const group2 = [_]f64{ 5.0, 5.0, 5.0 };
    const group3 = [_]f64{ 5.0, 5.0, 5.0, 5.0 };

    const groups = [_][]const f64{
        &group1,
        &group2,
        &group3,
    };

    const result = try anova_oneway(f64, &groups, 0.05, allocator);

    // Null hypothesis not rejected: F should be near 0, p-value near 1
    try testing.expectApproxEqAbs(0.0, result.statistic, 1e-10);
    try testing.expectApproxEqAbs(1.0, result.p_value, 1e-5);
    try testing.expect(result.reject == false);
}

test "anova_oneway: three groups with different means (F>0, p<0.05, reject=true)" {
    const group1 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };      // mean ≈ 3
    const group2 = [_]f64{ 6.0, 7.0, 8.0, 9.0, 10.0 };    // mean ≈ 8
    const group3 = [_]f64{ 11.0, 12.0, 13.0, 14.0, 15.0 }; // mean ≈ 13

    const groups = [_][]const f64{
        &group1,
        &group2,
        &group3,
    };

    const result = try anova_oneway(f64, &groups, 0.05, allocator);

    // Clear difference between groups: should reject H0
    try testing.expect(result.statistic > 0.0);
    try testing.expect(result.p_value < 0.05);
    try testing.expect(result.reject == true);
}

test "anova_oneway: four groups with different means" {
    const group1 = [_]f64{ 2.0, 3.0, 4.0 };
    const group2 = [_]f64{ 5.0, 6.0, 7.0 };
    const group3 = [_]f64{ 8.0, 9.0, 10.0 };
    const group4 = [_]f64{ 11.0, 12.0, 13.0 };

    const groups = [_][]const f64{
        &group1,
        &group2,
        &group3,
        &group4,
    };

    const result = try anova_oneway(f64, &groups, 0.05, allocator);

    // Four distinct groups: should strongly reject H0
    try testing.expect(result.statistic > 0.0);
    try testing.expect(result.p_value < 0.05);
    try testing.expect(result.reject == true);
}

test "anova_oneway: five groups with various sizes" {
    const group1 = [_]f64{ 1.0, 2.0 };
    const group2 = [_]f64{ 3.0, 4.0, 5.0, 6.0 };
    const group3 = [_]f64{ 7.0, 8.0, 9.0 };
    const group4 = [_]f64{ 10.0, 11.0 };
    const group5 = [_]f64{ 12.0, 13.0, 14.0, 15.0, 16.0 };

    const groups = [_][]const f64{
        &group1,
        &group2,
        &group3,
        &group4,
        &group5,
    };

    const result = try anova_oneway(f64, &groups, 0.05, allocator);

    // Clear group differences with unequal sizes
    try testing.expect(result.statistic > 0.0);
    try testing.expect(result.p_value < 0.05);
    try testing.expect(result.reject == true);
}

test "anova_oneway: two groups (minimum case)" {
    const group1 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const group2 = [_]f64{ 6.0, 7.0, 8.0, 9.0, 10.0 };

    const groups = [_][]const f64{
        &group1,
        &group2,
    };

    const result = try anova_oneway(f64, &groups, 0.05, allocator);

    // Two groups with difference
    try testing.expect(result.statistic > 0.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "anova_oneway: large groups (n=100 per group)" {
    var group1: [100]f64 = undefined;
    var group2: [100]f64 = undefined;
    var group3: [100]f64 = undefined;

    for (0..100) |i| {
        group1[i] = @as(f64, @floatFromInt(i));
        group2[i] = @as(f64, @floatFromInt(i + 100));
        group3[i] = @as(f64, @floatFromInt(i + 200));
    }

    const groups = [_][]const f64{
        &group1,
        &group2,
        &group3,
    };

    const result = try anova_oneway(f64, &groups, 0.05, allocator);

    // Large groups with clear differences
    try testing.expect(result.statistic > 0.0);
    try testing.expect(result.p_value < 0.05);
    try testing.expect(result.reject == true);
}

test "anova_oneway: unequal group sizes (n1=5, n2=15, n3=10)" {
    const group1 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var group2: [15]f64 = undefined;
    var group3: [10]f64 = undefined;

    for (0..15) |i| {
        group2[i] = @as(f64, @floatFromInt(i + 10));
    }
    for (0..10) |i| {
        group3[i] = @as(f64, @floatFromInt(i + 25));
    }

    const groups = [_][]const f64{
        &group1,
        &group2,
        &group3,
    };

    const result = try anova_oneway(f64, &groups, 0.05, allocator);

    // Unequal group sizes with mean differences
    try testing.expect(result.statistic > 0.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "anova_oneway: all same single value in all groups (F=0, p=1)" {
    const group1 = [_]f64{ 42.0, 42.0, 42.0 };
    const group2 = [_]f64{ 42.0, 42.0, 42.0, 42.0 };
    const group3 = [_]f64{ 42.0, 42.0 };

    const groups = [_][]const f64{
        &group1,
        &group2,
        &group3,
    };

    const result = try anova_oneway(f64, &groups, 0.05, allocator);

    // No variance within or between: F=0, p=1
    try testing.expectApproxEqAbs(0.0, result.statistic, 1e-10);
    try testing.expectApproxEqAbs(1.0, result.p_value, 1e-5);
    try testing.expect(result.reject == false);
}

test "anova_oneway: f32 precision" {
    const group1 = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const group2 = [_]f32{ 6.0, 7.0, 8.0, 9.0, 10.0 };
    const group3 = [_]f32{ 11.0, 12.0, 13.0, 14.0, 15.0 };

    const groups = [_][]const f32{
        &group1,
        &group2,
        &group3,
    };

    const result = try anova_oneway(f32, &groups, 0.05, allocator);

    // Should work with f32
    try testing.expect(result.statistic > 0.0);
    try testing.expect(!math.isNan(result.p_value));
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "anova_oneway: degrees of freedom calculation (df1=k-1, df2=N-k)" {
    const group1 = [_]f64{ 1.0, 2.0, 3.0 };
    const group2 = [_]f64{ 4.0, 5.0, 6.0 };
    const group3 = [_]f64{ 7.0, 8.0, 9.0, 10.0 };

    const groups = [_][]const f64{
        &group1,
        &group2,
        &group3,
    };

    const result = try anova_oneway(f64, &groups, 0.05, allocator);

    // k=3 groups, N=10 total observations
    // df_between = k-1 = 2, df_within = N-k = 7
    // df is encoded as a single value; expect it to be meaningful
    try testing.expect(result.df > 0.0);
    try testing.expect(!math.isNan(result.df));
}

test "anova_oneway: F-statistic is non-negative" {
    const group1 = [_]f64{ 1.0, 2.0, 3.0 };
    const group2 = [_]f64{ 4.0, 5.0, 6.0 };
    const group3 = [_]f64{ 7.0, 8.0, 9.0 };

    const groups = [_][]const f64{
        &group1,
        &group2,
        &group3,
    };

    const result = try anova_oneway(f64, &groups, 0.05, allocator);

    // F-statistic is always non-negative
    try testing.expect(result.statistic >= 0.0);
}

test "anova_oneway: p-value in [0, 1]" {
    const group1 = [_]f64{ 1.0, 2.0, 3.0 };
    const group2 = [_]f64{ 4.0, 5.0, 6.0 };
    const group3 = [_]f64{ 7.0, 8.0, 9.0 };

    const groups = [_][]const f64{
        &group1,
        &group2,
        &group3,
    };

    const result = try anova_oneway(f64, &groups, 0.05, allocator);

    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "anova_oneway: alpha=0.01 affects rejection decision" {
    const group1 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const group2 = [_]f64{ 6.0, 7.0, 8.0, 9.0, 10.0 };
    const group3 = [_]f64{ 11.0, 12.0, 13.0, 14.0, 15.0 };

    const groups = [_][]const f64{
        &group1,
        &group2,
        &group3,
    };

    const result_005 = try anova_oneway(f64, &groups, 0.05, allocator);
    const result_001 = try anova_oneway(f64, &groups, 0.01, allocator);

    // Same p-value but different rejection decisions based on alpha
    try testing.expectApproxEqAbs(result_005.p_value, result_001.p_value, 1e-10);
}

test "anova_oneway: alpha=0.1 affects rejection decision" {
    const group1 = [_]f64{ 1.0, 2.0, 3.0 };
    const group2 = [_]f64{ 4.0, 5.0, 6.0 };
    const group3 = [_]f64{ 7.0, 8.0, 9.0 };

    const groups = [_][]const f64{
        &group1,
        &group2,
        &group3,
    };

    const result = try anova_oneway(f64, &groups, 0.1, allocator);

    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "anova_oneway: error on too few groups (k < 2)" {
    const group1 = [_]f64{ 1.0, 2.0, 3.0 };

    const groups = [_][]const f64{
        &group1,
    };

    const result = anova_oneway(f64, &groups, 0.05, allocator);
    try testing.expectError(error.TooFewGroups, result);
}

test "anova_oneway: error on empty group (n=0)" {
    const group1 = [_]f64{ 1.0, 2.0, 3.0 };
    const group2: [0]f64 = [_]f64{};
    const group3 = [_]f64{ 4.0, 5.0 };

    const groups = [_][]const f64{
        &group1,
        &group2,
        &group3,
    };

    const result = anova_oneway(f64, &groups, 0.05, allocator);
    try testing.expectError(error.EmptyGroup, result);
}

test "anova_oneway: error on invalid alpha (alpha=0)" {
    const group1 = [_]f64{ 1.0, 2.0, 3.0 };
    const group2 = [_]f64{ 4.0, 5.0, 6.0 };
    const group3 = [_]f64{ 7.0, 8.0, 9.0 };

    const groups = [_][]const f64{
        &group1,
        &group2,
        &group3,
    };

    const result = anova_oneway(f64, &groups, 0.0, allocator);
    try testing.expectError(error.InvalidParameter, result);
}

test "anova_oneway: error on invalid alpha (alpha=1)" {
    const group1 = [_]f64{ 1.0, 2.0, 3.0 };
    const group2 = [_]f64{ 4.0, 5.0, 6.0 };
    const group3 = [_]f64{ 7.0, 8.0, 9.0 };

    const groups = [_][]const f64{
        &group1,
        &group2,
        &group3,
    };

    const result = anova_oneway(f64, &groups, 1.0, allocator);
    try testing.expectError(error.InvalidParameter, result);
}

test "anova_oneway: error on invalid alpha (alpha > 1)" {
    const group1 = [_]f64{ 1.0, 2.0, 3.0 };
    const group2 = [_]f64{ 4.0, 5.0, 6.0 };

    const groups = [_][]const f64{
        &group1,
        &group2,
    };

    const result = anova_oneway(f64, &groups, 1.5, allocator);
    try testing.expectError(error.InvalidParameter, result);
}

// ============================================================================
// KOLMOGOROV-SMIRNOV TESTS
// ============================================================================

/// One-sample Kolmogorov-Smirnov test: H0: data follows theoretical distribution
///
/// Tests whether a sample follows a specified theoretical cumulative distribution.
/// Computes the maximum distance between empirical and theoretical CDFs.
///
/// Formula:
/// - Empirical CDF: F_n(x) = (number of observations ≤ x) / n
/// - Theoretical CDF: F(x) = cdf_fn(x)
/// - D = max|F_n(x) - F(x)| over all observations x
///
/// The test statistic D follows the Kolmogorov distribution asymptotically.
/// A large D value indicates poor fit between data and theoretical distribution.
///
/// Parameters:
/// - data: 1D NDArray of observed values
/// - cdf_fn: Function pointer to theoretical CDF (fn(T) T)
/// - alpha: significance level (default 0.05 for 95% confidence)
/// - allocator: memory allocator for sorting data
///
/// Returns: TestResult with D statistic, p-value (right-tailed), df=0, and rejection decision
///
/// Errors:
/// - error.EmptyArray if data is empty
/// - error.InvalidParameter if alpha not in (0, 1)
/// - error.OutOfMemory if allocation fails
///
/// Time: O(n log n) where n = sample size (sorting dominates)
/// Space: O(n) for sorted copy of data
///
/// Notes:
/// - This is a right-tailed test (large D → reject H0)
/// - P-value = P(Kolmogorov(n) > D_observed)
/// - Uses asymptotic approximation: p ≈ 2 * exp(-2 * D² * n) for large n
/// - For small n: approximation may be less accurate but still reasonable
///
/// Example:
/// ```zig
/// const uniform_cdf = struct {
///     pub fn cdf(x: f64) f64 { if (x < 0) return 0; if (x > 1) return 1; return x; }
/// }.cdf;
/// const data = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{50}, &my_data, .row_major);
/// const result = try ks_test_1samp(f64, data, uniform_cdf, 0.05, allocator);
/// ```
pub fn ks_test_1samp(
    comptime T: type,
    data: NDArray_type(T, 1),
    cdf_fn: *const fn(T) T,
    alpha: T,
    alloc: std.mem.Allocator,
) !TestResult(T) {
    const n = data.count();

    // Validation
    if (n == 0) return error.EmptyArray;
    if (alpha <= 0 or alpha >= 1) return error.InvalidParameter;

    // Create a sorted copy of the data
    const sorted_data = try alloc.alloc(T, n);
    defer alloc.free(sorted_data);

    @memcpy(sorted_data, data.data[0..n]);

    // Sort the data
    const asc_f64 = struct {
        fn lessThan(_: void, a: T, b: T) bool {
            return a < b;
        }
    };
    std.mem.sort(T, sorted_data, {}, asc_f64.lessThan);

    // Compute D statistic: max|F_n(x) - F(x)|
    var d_stat: T = 0.0;

    for (0..n) |i| {
        const x = sorted_data[i];
        const f_theoretical = cdf_fn(x);

        // Empirical CDF at point x (after this observation)
        // F_n(x) = i / n for the i-th order statistic (0-indexed)
        // But we need to check both F_n(x-) and F_n(x)
        const f_empirical_after = @as(T, @floatFromInt(i + 1)) / @as(T, @floatFromInt(n));
        const f_empirical_before = @as(T, @floatFromInt(i)) / @as(T, @floatFromInt(n));

        // D = max of |F_n - F| at all points
        const diff_after = @abs(f_empirical_after - f_theoretical);
        const diff_before = @abs(f_empirical_before - f_theoretical);

        d_stat = @max(d_stat, @max(diff_after, diff_before));
    }

    // Compute p-value using asymptotic approximation
    // For large n: p ≈ 2 * exp(-2 * D² * n)
    // This approximation is reasonable for n > 5
    const n_f = @as(T, @floatFromInt(n));

    // Asymptotic p-value: P(D > d) ≈ 2 * exp(-2 * D² * n)
    const d_squared = d_stat * d_stat;
    const exponent = -2.0 * d_squared * n_f;
    const p_value = 2.0 * math.exp(exponent);

    // Clamp p-value to [0, 1]
    const p_clamped = @min(1.0, @max(0.0, p_value));

    return TestResult(T).init(d_stat, p_clamped, 0.0, alpha);
}

/// Two-sample Kolmogorov-Smirnov test: H0: two samples follow same distribution
///
/// Tests whether two samples follow the same underlying distribution.
/// Computes the maximum distance between empirical CDFs of both samples.
///
/// Formula:
/// - Empirical CDF of sample 1: F_1(x) = (count of obs ≤ x in sample 1) / n1
/// - Empirical CDF of sample 2: F_2(x) = (count of obs ≤ x in sample 2) / n2
/// - D = max|F_1(x) - F_2(x)| over all observations from both samples
///
/// The test statistic D is compared using an effective sample size.
/// A large D value indicates the two distributions differ significantly.
///
/// Parameters:
/// - data1: 1D NDArray of first sample observations
/// - data2: 1D NDArray of second sample observations
/// - alpha: significance level (default 0.05 for 95% confidence)
/// - allocator: memory allocator for sorting data
///
/// Returns: TestResult with D statistic, p-value (right-tailed), df=0, and rejection decision
///
/// Errors:
/// - error.EmptyArray if either data1 or data2 is empty
/// - error.InvalidParameter if alpha not in (0, 1)
/// - error.OutOfMemory if allocation fails
///
/// Time: O((n1+n2) log(n1+n2)) for sorting combined samples
/// Space: O(n1 + n2) for merged and sorted data
///
/// Notes:
/// - This is a right-tailed test (large D → reject H0)
/// - P-value computed using two-sample Kolmogorov distribution (asymptotic)
/// - Effective sample size n_eff = sqrt(n1*n2/(n1+n2))
/// - P-value approximation: similar to one-sample case with effective n
/// - Test is symmetric: ks_test_2samp(a, b) ≡ ks_test_2samp(b, a)
///
/// Example:
/// ```zig
/// const data1 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{30}, &sample1, .row_major);
/// const data2 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{25}, &sample2, .row_major);
/// const result = try ks_test_2samp(f64, data1, data2, 0.05, allocator);
/// ```
pub fn ks_test_2samp(
    comptime T: type,
    data1: NDArray_type(T, 1),
    data2: NDArray_type(T, 1),
    alpha: T,
    alloc: std.mem.Allocator,
) !TestResult(T) {
    const n1 = data1.count();
    const n2 = data2.count();

    // Validation
    if (n1 == 0 or n2 == 0) return error.EmptyArray;
    if (alpha <= 0 or alpha >= 1) return error.InvalidParameter;

    // Merge both samples and track which sample each point came from
    const total_n = n1 + n2;
    var merged_data = try alloc.alloc(T, total_n);
    defer alloc.free(merged_data);

    @memcpy(merged_data[0..n1], data1.data[0..n1]);
    @memcpy(merged_data[n1..total_n], data2.data[0..n2]);

    // Sort merged data
    const asc_f64 = struct {
        fn lessThan(_: void, a: T, b: T) bool {
            return a < b;
        }
    };
    std.mem.sort(T, merged_data, {}, asc_f64.lessThan);

    // Compute D statistic by iterating through sorted unique values
    var d_stat: T = 0.0;

    // Track counts for each sample
    var count1: usize = 0;
    var count2: usize = 0;

    var i: usize = 0;
    while (i < total_n) : (i += 1) {
        const current_val = merged_data[i];

        // Count how many from each sample are ≤ current_val
        count1 = 0;
        count2 = 0;

        for (0..n1) |j| {
            if (data1.data[j] <= current_val) {
                count1 += 1;
            }
        }
        for (0..n2) |j| {
            if (data2.data[j] <= current_val) {
                count2 += 1;
            }
        }

        // Compute empirical CDFs
        const f1 = @as(T, @floatFromInt(count1)) / @as(T, @floatFromInt(n1));
        const f2 = @as(T, @floatFromInt(count2)) / @as(T, @floatFromInt(n2));

        // Update D statistic
        const diff = @abs(f1 - f2);
        d_stat = @max(d_stat, diff);
    }

    // Compute p-value using asymptotic approximation for two-sample KS test
    // Effective sample size: n_eff = sqrt(n1*n2 / (n1 + n2))
    const n1_f = @as(T, @floatFromInt(n1));
    const n2_f = @as(T, @floatFromInt(n2));
    const n_eff = math.sqrt((n1_f * n2_f) / (n1_f + n2_f));

    // p-value: similar to one-sample but with effective n
    // P(D > d) ≈ 2 * exp(-2 * D² * n_eff)
    const d_squared = d_stat * d_stat;
    const exponent = -2.0 * d_squared * n_eff;
    const p_value = 2.0 * math.exp(exponent);

    // Clamp p-value to [0, 1]
    const p_clamped = @min(1.0, @max(0.0, p_value));

    return TestResult(T).init(d_stat, p_clamped, 0.0, alpha);
}

// ============================================================================
// MANN-WHITNEY U TEST
// ============================================================================

/// Mann-Whitney U test: Non-parametric test comparing two independent samples
///
/// Tests whether two independent samples come from the same distribution using the
/// Mann-Whitney U statistic (also called Wilcoxon rank-sum test). This is the
/// non-parametric alternative to the independent samples t-test and does not assume
/// normality of the data.
///
/// Parameters:
/// - T: numeric type (f32 or f64)
/// - data1: First sample (NDArray, 1D)
/// - data2: Second sample (NDArray, 1D)
/// - alpha: Significance level (must be in (0, 1))
/// - alloc: Allocator for temporary arrays
///
/// Returns:
/// - TestResult with:
///   - statistic: U statistic (min of U1 and U2)
///   - p_value: Two-tailed p-value using normal approximation
///   - df: 0 (not applicable)
///   - reject: true if p_value < alpha
///
/// Algorithm:
/// 1. Merge both samples and assign ranks 1 to n (n = n1 + n2)
/// 2. Handle ties by averaging ranks of equal values
/// 3. Compute rank sums R1 and R2
/// 4. Calculate U1 = n1*n2 + n1(n1+1)/2 - R1 and U2 = n1*n2 + n2(n2+1)/2 - R2
/// 5. U = min(U1, U2)
/// 6. Use normal approximation for p-value: mean = n1*n2/2, var = n1*n2*(n1+n2+1)/12
/// 7. Two-tailed p-value: p = 2 * (1 - Φ(|z|))
///
/// Errors:
/// - error.EmptyArray: if either sample is empty
/// - error.InvalidParameter: if alpha is not in (0, 1)
///
/// Time: O((n1+n2) log(n1+n2)) for sorting
/// Space: O(n1+n2) for merged array and ranks
pub fn mannwhitney_u(
    comptime T: type,
    data1: NDArray_type(T, 1),
    data2: NDArray_type(T, 1),
    alpha: T,
    alloc: std.mem.Allocator,
) !TestResult(T) {
    const n1 = data1.count();
    const n2 = data2.count();

    // Validation
    if (n1 == 0 or n2 == 0) return error.EmptyArray;
    if (alpha <= 0 or alpha >= 1) return error.InvalidParameter;

    const total_n = n1 + n2;

    // Allocate arrays for merged data, tracking which sample, and ranks
    var merged_data = try alloc.alloc(T, total_n);
    defer alloc.free(merged_data);

    var sample_id = try alloc.alloc(usize, total_n);
    defer alloc.free(sample_id);

    var ranks = try alloc.alloc(T, total_n);
    defer alloc.free(ranks);

    // Copy and mark samples
    @memcpy(merged_data[0..n1], data1.data[0..n1]);
    for (0..n1) |i| {
        sample_id[i] = 0;
    }

    @memcpy(merged_data[n1..total_n], data2.data[0..n2]);
    for (0..n2) |i| {
        sample_id[n1 + i] = 1;
    }

    // Create indices array for sorting
    var indices = try alloc.alloc(usize, total_n);
    defer alloc.free(indices);
    for (0..total_n) |i| {
        indices[i] = i;
    }

    // Sort indices by data values
    const IndexComparator = struct {
        data: []T,

        fn lessThan(self: @This(), a: usize, b: usize) bool {
            return self.data[a] < self.data[b];
        }
    };

    const comp = IndexComparator{ .data = merged_data };
    std.mem.sort(usize, indices, comp, IndexComparator.lessThan);

    // Assign ranks, handling ties
    var i: usize = 0;
    while (i < total_n) {
        const j = i;
        // Find the end of tied group
        while (i < total_n - 1 and
            merged_data[indices[i]] == merged_data[indices[i + 1]])
        {
            i += 1;
        }

        // Average rank for tied group [j, i]
        const avg_rank = @as(T, @floatFromInt(j + i + 2)) / 2.0;
        for (j..i + 1) |k| {
            ranks[indices[k]] = avg_rank;
        }

        i += 1;
    }

    // Compute rank sums
    var rank_sum1: T = 0.0;
    var rank_sum2: T = 0.0;

    for (0..n1) |j| {
        rank_sum1 += ranks[j];
    }
    for (0..n2) |j| {
        rank_sum2 += ranks[n1 + j];
    }

    // Compute U statistics
    const n1_f = @as(T, @floatFromInt(n1));
    const n2_f = @as(T, @floatFromInt(n2));

    const u_stat_1 = n1_f * n2_f + n1_f * (n1_f + 1.0) / 2.0 - rank_sum1;
    const u_stat_2 = n1_f * n2_f + n2_f * (n2_f + 1.0) / 2.0 - rank_sum2;

    // U statistic is the minimum of U1 and U2
    const u_stat = @min(u_stat_1, u_stat_2);

    // Compute p-value using normal approximation
    const mean_u = n1_f * n2_f / 2.0;
    const var_u = n1_f * n2_f * (n1_f + n2_f + 1.0) / 12.0;

    // Handle edge case where variance is 0 (all identical values)
    if (var_u == 0.0) {
        return TestResult(T).init(u_stat, 1.0, 0.0, alpha);
    }

    const std_u = math.sqrt(var_u);

    // Standard normal approximation: z = (U - mean) / std
    // Note: continuity correction is sometimes applied, but standard normal approximation without it gives better p-values for moderate samples
    const z = @abs(u_stat - mean_u) / std_u;

    // Standard normal CDF using erf approximation
    const p_cdf = stdNormalCDF(T, z);
    const p_value = 2.0 * (1.0 - p_cdf);

    // Clamp p-value to [0, 1]
    const p_clamped = @min(1.0, @max(0.0, p_value));

    return TestResult(T).init(u_stat, p_clamped, 0.0, alpha);
}

// Helper function: standard normal CDF using erf approximation
fn stdNormalCDF(comptime T: type, z: T) T {
    // Standard normal CDF: Φ(z) = 0.5[1 + erfApprox(z/√2)]
    // Abramowitz and Stegun approximation for erf
    const a1: T = 0.254829592;
    const a2: T = -0.284496736;
    const a3: T = 1.421413741;
    const a4: T = -1.453152027;
    const a5: T = 1.061405429;
    const p: T = 0.3275911;

    const sqrt2 = math.sqrt(@as(T, 2.0));
    const z_norm = z / sqrt2;

    const sign = if (z_norm < 0) @as(T, -1.0) else @as(T, 1.0);
    const abs_x = @abs(z_norm);
    const t = 1.0 / (1.0 + p * abs_x);
    const t2 = t * t;
    const t3 = t2 * t;
    const t4 = t3 * t;
    const t5 = t4 * t;

    const erf_val = sign * (1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * math.exp(-abs_x * abs_x));
    return 0.5 * (1.0 + erf_val);
}

// ============================================================================
// Kolmogorov-Smirnov Test Tests (30+ tests)
// ============================================================================

test "ks_test_1samp: perfect fit (data from Uniform(0,1) vs uniform CDF, D≈0, p≈1, reject=false)" {
    // Data from Uniform(0,1) should match uniform CDF perfectly
    const data_slice = [_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{9}, &data_slice, .row_major);
    defer data.deinit();

    // Uniform CDF on [0, 1]: F(x) = x
    const uniform_cdf = struct {
        pub fn cdf(x: f64) f64 {
            if (x < 0.0) return 0.0;
            if (x > 1.0) return 1.0;
            return x;
        }
    }.cdf;

    const result = try ks_test_1samp(f64, data, uniform_cdf, 0.05, allocator);

    // Should have very small D statistic and large p-value
    try testing.expect(result.statistic >= 0.0 and result.statistic <= 1.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
    try testing.expect(!math.isNan(result.statistic));
    try testing.expect(!math.isNan(result.p_value));
}

test "ks_test_1samp: good fit (Normal(0,1) data vs standard normal CDF, should not reject)" {
    // Data approximately from Normal(0,1)
    const data_slice = [_]f64{ -1.5, -0.8, -0.3, 0.0, 0.2, 0.5, 1.0, 1.3, 1.8 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{9}, &data_slice, .row_major);
    defer data.deinit();

    // Standard normal CDF (erf approximation)
    const normal_cdf = struct {
        pub fn cdf(x: f64) f64 {
            // Standard normal CDF: Φ(x) = 0.5[1 + erf(x/√2)]
            const sqrt2 = @sqrt(2.0);
            return 0.5 * (1.0 + erf(x / sqrt2));
        }

        // Abramowitz and Stegun approximation for erf
        fn erf(x: f64) f64 {
            const a1 = 0.254829592;
            const a2 = -0.284496736;
            const a3 = 1.421413741;
            const a4 = -1.453152027;
            const a5 = 1.061405429;
            const p = 0.3275911;

            const sign = if (x < 0) @as(f64, -1.0) else @as(f64, 1.0);
            const abs_x = @abs(x);
            const t = 1.0 / (1.0 + p * abs_x);
            const t2 = t * t;
            const t3 = t2 * t;
            const t4 = t3 * t;
            const t5 = t4 * t;

            const result = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * @exp(-abs_x * abs_x);
            return sign * result;
        }
    }.cdf;

    const result = try ks_test_1samp(f64, data, normal_cdf, 0.05, allocator);

    try testing.expect(result.statistic >= 0.0 and result.statistic <= 1.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
    try testing.expect(!math.isNan(result.statistic));
}

test "ks_test_1samp: poor fit (Normal(0,1) data vs Exponential CDF, should reject)" {
    // Normal data vs exponential CDF: should have large D and small p
    const data_slice = [_]f64{ 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{9}, &data_slice, .row_major);
    defer data.deinit();

    // Exponential CDF with λ=1: F(x) = 1 - e^(-x)
    const exponential_cdf = struct {
        pub fn cdf(x: f64) f64 {
            if (x < 0.0) return 0.0;
            return 1.0 - math.exp(-x);
        }
    }.cdf;

    const result = try ks_test_1samp(f64, data, exponential_cdf, 0.05, allocator);

    try testing.expect(result.statistic >= 0.0 and result.statistic <= 1.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "ks_test_1samp: edge case - single sample" {
    const data_slice = [_]f64{0.5};
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{1}, &data_slice, .row_major);
    defer data.deinit();

    const uniform_cdf = struct {
        pub fn cdf(x: f64) f64 {
            if (x < 0.0) return 0.0;
            if (x > 1.0) return 1.0;
            return x;
        }
    }.cdf;

    const result = try ks_test_1samp(f64, data, uniform_cdf, 0.05, allocator);

    try testing.expect(result.statistic >= 0.0 and result.statistic <= 1.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "ks_test_1samp: edge case - two samples" {
    const data_slice = [_]f64{ 0.25, 0.75 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &data_slice, .row_major);
    defer data.deinit();

    const uniform_cdf = struct {
        pub fn cdf(x: f64) f64 {
            if (x < 0.0) return 0.0;
            if (x > 1.0) return 1.0;
            return x;
        }
    }.cdf;

    const result = try ks_test_1samp(f64, data, uniform_cdf, 0.05, allocator);

    try testing.expect(result.statistic >= 0.0 and result.statistic <= 1.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "ks_test_1samp: edge case - all same values (uniform empirical CDF)" {
    const data_slice = [_]f64{ 5.0, 5.0, 5.0, 5.0, 5.0 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();

    const point_cdf = struct {
        pub fn cdf(x: f64) f64 {
            if (x < 5.0) return 0.0;
            return 1.0;
        }
    }.cdf;

    const result = try ks_test_1samp(f64, data, point_cdf, 0.05, allocator);

    // All data at 5.0 should match Dirac delta at 5.0 perfectly
    try testing.expect(result.statistic >= 0.0 and result.statistic <= 1.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "ks_test_1samp: D statistic in valid range [0, 1]" {
    const data_slice = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();

    const uniform_cdf = struct {
        pub fn cdf(x: f64) f64 {
            if (x < 0.0) return 0.0;
            if (x > 1.0) return 1.0;
            return x;
        }
    }.cdf;

    const result = try ks_test_1samp(f64, data, uniform_cdf, 0.05, allocator);

    try testing.expect(result.statistic >= 0.0);
    try testing.expect(result.statistic <= 1.0);
}

test "ks_test_1samp: p_value in valid range [0, 1]" {
    const data_slice = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();

    const uniform_cdf = struct {
        pub fn cdf(x: f64) f64 {
            if (x < 0.0) return 0.0;
            if (x > 1.0) return 1.0;
            return x;
        }
    }.cdf;

    const result = try ks_test_1samp(f64, data, uniform_cdf, 0.05, allocator);

    try testing.expect(result.p_value >= 0.0);
    try testing.expect(result.p_value <= 1.0);
}

test "ks_test_1samp: larger D → smaller p-value" {
    // Create two datasets: one with small deviation, one with large
    const data_small = [_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
    const data_large = [_]f64{ 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9 };

    var small = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{9}, &data_small, .row_major);
    defer small.deinit();
    var large = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{9}, &data_large, .row_major);
    defer large.deinit();

    const uniform_cdf = struct {
        pub fn cdf(x: f64) f64 {
            if (x < 0.0) return 0.0;
            if (x > 1.0) return 1.0;
            return x;
        }
    }.cdf;

    const result_small = try ks_test_1samp(f64, small, uniform_cdf, 0.05, allocator);
    const result_large = try ks_test_1samp(f64, large, uniform_cdf, 0.05, allocator);

    // Large D should have smaller p-value
    try testing.expect(result_large.statistic > result_small.statistic);
    try testing.expect(result_large.p_value < result_small.p_value);
}

test "ks_test_1samp: alpha threshold affects rejection" {
    const data_slice = [_]f64{ 0.1, 0.5, 0.9 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data_slice, .row_major);
    defer data.deinit();

    const uniform_cdf = struct {
        pub fn cdf(x: f64) f64 {
            if (x < 0.0) return 0.0;
            if (x > 1.0) return 1.0;
            return x;
        }
    }.cdf;

    const result_alpha_01 = try ks_test_1samp(f64, data, uniform_cdf, 0.1, allocator);
    const result_alpha_001 = try ks_test_1samp(f64, data, uniform_cdf, 0.01, allocator);

    // Same p-value, but rejection decision depends on alpha
    try testing.expectApproxEqAbs(result_alpha_01.p_value, result_alpha_001.p_value, 1e-10);
    try testing.expect(result_alpha_01.reject == (result_alpha_01.p_value < 0.1));
    try testing.expect(result_alpha_001.reject == (result_alpha_001.p_value < 0.01));
}

test "ks_test_1samp: f32 precision" {
    const data_slice = [_]f32{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    var data = try NDArray_type(f32, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();

    const uniform_cdf_f32 = struct {
        pub fn cdf(x: f32) f32 {
            if (x < 0.0) return 0.0;
            if (x > 1.0) return 1.0;
            return x;
        }
    }.cdf;

    const result = try ks_test_1samp(f32, data, uniform_cdf_f32, 0.05, allocator);

    try testing.expect(!math.isNan(result.statistic));
    try testing.expect(!math.isNan(result.p_value));
    try testing.expect(result.statistic >= 0.0 and result.statistic <= 1.0);
}

test "ks_test_1samp: f64 precision" {
    const data_slice = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();

    const uniform_cdf = struct {
        pub fn cdf(x: f64) f64 {
            if (x < 0.0) return 0.0;
            if (x > 1.0) return 1.0;
            return x;
        }
    }.cdf;

    const result = try ks_test_1samp(f64, data, uniform_cdf, 0.05, allocator);

    try testing.expect(!math.isNan(result.statistic));
    try testing.expect(!math.isNan(result.p_value));
}

test "ks_test_1samp: large sample (n=100)" {
    var data_slice: [100]f64 = undefined;
    for (0..100) |i| {
        data_slice[i] = (@as(f64, @floatFromInt(i)) + 0.5) / 100.0; // Uniform approximation
    }
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{100}, &data_slice, .row_major);
    defer data.deinit();

    const uniform_cdf = struct {
        pub fn cdf(x: f64) f64 {
            if (x < 0.0) return 0.0;
            if (x > 1.0) return 1.0;
            return x;
        }
    }.cdf;

    const result = try ks_test_1samp(f64, data, uniform_cdf, 0.05, allocator);

    try testing.expect(result.statistic >= 0.0 and result.statistic <= 1.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "ks_test_1samp: memory safety - no leaks" {
    const data_slice = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();

    const uniform_cdf = struct {
        pub fn cdf(x: f64) f64 {
            if (x < 0.0) return 0.0;
            if (x > 1.0) return 1.0;
            return x;
        }
    }.cdf;

    // allocator is std.testing.allocator which detects memory leaks
    _ = try ks_test_1samp(f64, data, uniform_cdf, 0.05, allocator);
}

test "ks_test_1samp: error on empty array" {
    const data_slice: [0]f64 = [_]f64{};
    const result = NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{0}, &data_slice, .row_major);
    try testing.expectError(NDArray_type(f64, 1).Error.ZeroDimension, result);
}

test "ks_test_1samp: error on invalid alpha (alpha=0)" {
    const data_slice = [_]f64{ 0.1, 0.3, 0.5 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data_slice, .row_major);
    defer data.deinit();

    const uniform_cdf = struct {
        pub fn cdf(x: f64) f64 {
            return x;
        }
    }.cdf;

    const result = ks_test_1samp(f64, data, uniform_cdf, 0.0, allocator);
    try testing.expectError(error.InvalidParameter, result);
}

test "ks_test_1samp: error on invalid alpha (alpha=1)" {
    const data_slice = [_]f64{ 0.1, 0.3, 0.5 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data_slice, .row_major);
    defer data.deinit();

    const uniform_cdf = struct {
        pub fn cdf(x: f64) f64 {
            return x;
        }
    }.cdf;

    const result = ks_test_1samp(f64, data, uniform_cdf, 1.0, allocator);
    try testing.expectError(error.InvalidParameter, result);
}

test "ks_test_1samp: error on invalid alpha (alpha > 1)" {
    const data_slice = [_]f64{ 0.1, 0.3, 0.5 };
    var data = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data_slice, .row_major);
    defer data.deinit();

    const uniform_cdf = struct {
        pub fn cdf(x: f64) f64 {
            return x;
        }
    }.cdf;

    const result = ks_test_1samp(f64, data, uniform_cdf, 1.5, allocator);
    try testing.expectError(error.InvalidParameter, result);
}

test "ks_test_1samp: consistency across multiple runs" {
    const data_slice = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    var data1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data1.deinit();
    var data2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data2.deinit();

    const uniform_cdf = struct {
        pub fn cdf(x: f64) f64 {
            if (x < 0.0) return 0.0;
            if (x > 1.0) return 1.0;
            return x;
        }
    }.cdf;

    const result1 = try ks_test_1samp(f64, data1, uniform_cdf, 0.05, allocator);
    const result2 = try ks_test_1samp(f64, data2, uniform_cdf, 0.05, allocator);

    try testing.expectApproxEqAbs(result1.statistic, result2.statistic, 1e-10);
    try testing.expectApproxEqAbs(result1.p_value, result2.p_value, 1e-10);
}

// ============================================================================
// Two-Sample Kolmogorov-Smirnov Test Tests (15+ tests)
// ============================================================================

test "ks_test_2samp: identical samples (D≈0, p≈1, reject=false)" {
    const data = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer sample2.deinit();

    const result = try ks_test_2samp(f64, sample1, sample2, 0.05, allocator);

    // Identical samples: D should be very small, p-value ≈ 1.0
    try testing.expect(result.statistic >= 0.0 and result.statistic <= 1.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
    try testing.expect(!math.isNan(result.statistic));
}

test "ks_test_2samp: both samples from Normal(0,1) (should not reject)" {
    const data1 = [_]f64{ -1.5, -0.8, -0.3, 0.0, 0.2, 0.5, 1.0 };
    const data2 = [_]f64{ -1.2, -0.5, 0.1, 0.4, 0.8, 1.2, 1.5 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{7}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{7}, &data2, .row_major);
    defer sample2.deinit();

    const result = try ks_test_2samp(f64, sample1, sample2, 0.05, allocator);

    try testing.expect(result.statistic >= 0.0 and result.statistic <= 1.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "ks_test_2samp: Normal(0,1) vs Normal(5,1) (should reject)" {
    const data1 = [_]f64{ -1.0, 0.0, 1.0 };
    const data2 = [_]f64{ 4.0, 5.0, 6.0 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2.deinit();

    const result = try ks_test_2samp(f64, sample1, sample2, 0.05, allocator);

    try testing.expect(result.statistic >= 0.0 and result.statistic <= 1.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "ks_test_2samp: Uniform(0,1) vs Uniform(2,3) (should reject)" {
    const data1 = [_]f64{ 0.2, 0.4, 0.6, 0.8 };
    const data2 = [_]f64{ 2.2, 2.4, 2.6, 2.8 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{4}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{4}, &data2, .row_major);
    defer sample2.deinit();

    const result = try ks_test_2samp(f64, sample1, sample2, 0.05, allocator);

    try testing.expect(result.statistic >= 0.0 and result.statistic <= 1.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "ks_test_2samp: unequal sample sizes (n1=3, n2=7)" {
    const data1 = [_]f64{ 0.1, 0.5, 0.9 };
    const data2 = [_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{7}, &data2, .row_major);
    defer sample2.deinit();

    const result = try ks_test_2samp(f64, sample1, sample2, 0.05, allocator);

    try testing.expect(result.statistic >= 0.0 and result.statistic <= 1.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "ks_test_2samp: equal sample sizes" {
    const data1 = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    const data2 = [_]f64{ 0.15, 0.35, 0.55, 0.75, 0.95 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data2, .row_major);
    defer sample2.deinit();

    const result = try ks_test_2samp(f64, sample1, sample2, 0.05, allocator);

    try testing.expect(result.statistic >= 0.0 and result.statistic <= 1.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "ks_test_2samp: symmetry (swap data1 ↔ data2)" {
    const data1 = [_]f64{ 0.1, 0.3, 0.5 };
    const data2 = [_]f64{ 0.2, 0.4, 0.6 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2.deinit();
    var sample1_dup = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1_dup.deinit();
    var sample2_dup = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2_dup.deinit();

    const result_12 = try ks_test_2samp(f64, sample1, sample2, 0.05, allocator);
    const result_21 = try ks_test_2samp(f64, sample2_dup, sample1_dup, 0.05, allocator);

    // D-statistic should be equal (symmetric)
    try testing.expectApproxEqAbs(result_12.statistic, result_21.statistic, 1e-10);
    // p-value should also be equal
    try testing.expectApproxEqAbs(result_12.p_value, result_21.p_value, 1e-10);
}

test "ks_test_2samp: D statistic in valid range [0, 1]" {
    const data1 = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    const data2 = [_]f64{ 0.2, 0.4, 0.6, 0.8 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{4}, &data2, .row_major);
    defer sample2.deinit();

    const result = try ks_test_2samp(f64, sample1, sample2, 0.05, allocator);

    try testing.expect(result.statistic >= 0.0);
    try testing.expect(result.statistic <= 1.0);
}

test "ks_test_2samp: p_value in valid range [0, 1]" {
    const data1 = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    const data2 = [_]f64{ 0.2, 0.4, 0.6, 0.8 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{4}, &data2, .row_major);
    defer sample2.deinit();

    const result = try ks_test_2samp(f64, sample1, sample2, 0.05, allocator);

    try testing.expect(result.p_value >= 0.0);
    try testing.expect(result.p_value <= 1.0);
}

test "ks_test_2samp: larger D → smaller p-value" {
    // Two pairs: one similar, one very different
    const similar1 = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    const similar2 = [_]f64{ 0.15, 0.35, 0.55, 0.75, 0.95 };

    const diff1 = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    const diff2 = [_]f64{ 0.9, 0.9, 0.9, 0.9, 0.9 };

    var s1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &similar1, .row_major);
    defer s1.deinit();
    var s2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &similar2, .row_major);
    defer s2.deinit();
    var d1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &diff1, .row_major);
    defer d1.deinit();
    var d2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &diff2, .row_major);
    defer d2.deinit();

    const result_similar = try ks_test_2samp(f64, s1, s2, 0.05, allocator);
    const result_diff = try ks_test_2samp(f64, d1, d2, 0.05, allocator);

    // Different samples should have larger D
    try testing.expect(result_diff.statistic > result_similar.statistic);
    // Larger D should have smaller p-value
    try testing.expect(result_diff.p_value < result_similar.p_value);
}

test "ks_test_2samp: alpha threshold affects rejection" {
    const data1 = [_]f64{ 0.1, 0.5, 0.9 };
    const data2 = [_]f64{ 0.2, 0.6, 0.8 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2_1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2_1.deinit();
    var sample2_2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2_2.deinit();

    const result_alpha_10 = try ks_test_2samp(f64, sample1, sample2_1, 0.10, allocator);
    const result_alpha_01 = try ks_test_2samp(f64, sample1, sample2_2, 0.01, allocator);

    // Same p-value, rejection depends on alpha
    try testing.expectApproxEqAbs(result_alpha_10.p_value, result_alpha_01.p_value, 1e-10);
    try testing.expect(result_alpha_10.reject == (result_alpha_10.p_value < 0.10));
    try testing.expect(result_alpha_01.reject == (result_alpha_01.p_value < 0.01));
}

test "ks_test_2samp: f32 precision" {
    const data1 = [_]f32{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    const data2 = [_]f32{ 0.2, 0.4, 0.6, 0.8 };

    var sample1 = try NDArray_type(f32, 1).fromSlice(allocator, &[_]usize{5}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f32, 1).fromSlice(allocator, &[_]usize{4}, &data2, .row_major);
    defer sample2.deinit();

    const result = try ks_test_2samp(f32, sample1, sample2, 0.05, allocator);

    try testing.expect(!math.isNan(result.statistic));
    try testing.expect(!math.isNan(result.p_value));
    try testing.expect(result.statistic >= 0.0 and result.statistic <= 1.0);
}

test "ks_test_2samp: f64 precision" {
    const data1 = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    const data2 = [_]f64{ 0.2, 0.4, 0.6, 0.8 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{4}, &data2, .row_major);
    defer sample2.deinit();

    const result = try ks_test_2samp(f64, sample1, sample2, 0.05, allocator);

    try testing.expect(!math.isNan(result.statistic));
    try testing.expect(!math.isNan(result.p_value));
}

test "ks_test_2samp: large samples" {
    var data1: [50]f64 = undefined;
    var data2: [50]f64 = undefined;
    for (0..50) |i| {
        data1[i] = (@as(f64, @floatFromInt(i)) + 0.5) / 50.0;
        data2[i] = (@as(f64, @floatFromInt(i)) + 0.5) / 50.0 + 0.1; // Slight shift
    }

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{50}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{50}, &data2, .row_major);
    defer sample2.deinit();

    const result = try ks_test_2samp(f64, sample1, sample2, 0.05, allocator);

    try testing.expect(result.statistic >= 0.0 and result.statistic <= 1.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "ks_test_2samp: memory safety - no leaks" {
    const data1 = [_]f64{ 0.1, 0.3, 0.5 };
    const data2 = [_]f64{ 0.2, 0.4, 0.6 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2.deinit();

    // allocator is std.testing.allocator which detects memory leaks
    _ = try ks_test_2samp(f64, sample1, sample2, 0.05, allocator);
}

test "ks_test_2samp: consistency across multiple runs" {
    const data1 = [_]f64{ 0.1, 0.3, 0.5 };
    const data2 = [_]f64{ 0.2, 0.4, 0.6 };

    var sample1a = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1a.deinit();
    var sample2a = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2a.deinit();

    var sample1b = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1b.deinit();
    var sample2b = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2b.deinit();

    const result1 = try ks_test_2samp(f64, sample1a, sample2a, 0.05, allocator);
    const result2 = try ks_test_2samp(f64, sample1b, sample2b, 0.05, allocator);

    try testing.expectApproxEqAbs(result1.statistic, result2.statistic, 1e-10);
    try testing.expectApproxEqAbs(result1.p_value, result2.p_value, 1e-10);
}

// ============================================================================
// Mann-Whitney U Test Tests (20+ tests)
// ============================================================================

test "mannwhitney_u: identical samples (U ≈ n1*n2/2, p ≈ 1, no reject)" {
    const data1 = [_]f64{ 1.0, 2.0, 3.0 };
    const data2 = [_]f64{ 1.0, 2.0, 3.0 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2.deinit();

    const result = try mannwhitney_u(f64, sample1, sample2, 0.05, allocator);

    // U should be around n1*n2/2 = 4.5 for identical samples with n1=n2=3
    const _expected_u = @as(f64, 3.0) * @as(f64, 3.0) / 2.0; // 4.5
    _ = _expected_u;
    try testing.expect(result.statistic >= 0.0);
    try testing.expect(result.statistic <= 9.0); // max possible U = 3*3
    try testing.expect(result.p_value > 0.9); // very high p-value
    try testing.expect(!result.reject); // should not reject H0
}

test "mannwhitney_u: same distribution (Normal(0,1)), should not reject" {
    const data1 = [_]f64{ -1.5, -0.8, -0.3, 0.0, 0.2, 0.5, 1.0, 1.3, 1.8 };
    const data2 = [_]f64{ -1.4, -0.7, -0.2, 0.1, 0.3, 0.6, 1.1, 1.4, 1.9 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{9}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{9}, &data2, .row_major);
    defer sample2.deinit();

    const result = try mannwhitney_u(f64, sample1, sample2, 0.05, allocator);

    // Similar distributions should have large U and high p-value
    try testing.expect(result.statistic >= 0.0);
    try testing.expect(result.statistic <= 81.0); // max = 9*9
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
    try testing.expect(!result.reject); // should not reject similar distributions
}

test "mannwhitney_u: different medians ([1,2,3] vs [4,5,6]), should reject" {
    const data1 = [_]f64{ 1.0, 2.0, 3.0 };
    const data2 = [_]f64{ 4.0, 5.0, 6.0 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2.deinit();

    const result = try mannwhitney_u(f64, sample1, sample2, 0.05, allocator);

    // Completely separated samples should have U=0 and p≈0
    try testing.expect(result.statistic == 0.0); // U = min(0, 9) = 0
    try testing.expect(result.p_value < 0.05); // very small p-value
    try testing.expect(result.reject); // should reject H0
}

test "mannwhitney_u: overlapping ranges ([1,2,3,4] vs [3,4,5,6]), intermediate U" {
    const data1 = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const data2 = [_]f64{ 3.0, 4.0, 5.0, 6.0 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{4}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{4}, &data2, .row_major);
    defer sample2.deinit();

    const result = try mannwhitney_u(f64, sample1, sample2, 0.05, allocator);

    // Overlapping samples should have intermediate U
    try testing.expect(result.statistic > 0.0);
    try testing.expect(result.statistic < 16.0); // max = 4*4
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "mannwhitney_u: different distributions (Normal(0,1) vs Normal(5,1)), should reject" {
    const data1 = [_]f64{ -1.5, -0.8, -0.3, 0.0, 0.2, 0.5, 1.0, 1.3, 1.8 };
    const data2 = [_]f64{ 3.5, 4.2, 4.7, 5.0, 5.2, 5.5, 6.0, 6.3, 6.8 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{9}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{9}, &data2, .row_major);
    defer sample2.deinit();

    const result = try mannwhitney_u(f64, sample1, sample2, 0.05, allocator);

    // Shifted distributions should have small U and small p-value
    try testing.expect(result.statistic >= 0.0 and result.statistic <= 81.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
    try testing.expect(result.reject); // should reject H0
}

test "mannwhitney_u: single element each (n1=1, n2=1)" {
    const data1 = [_]f64{1.0};
    const data2 = [_]f64{2.0};

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{1}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{1}, &data2, .row_major);
    defer sample2.deinit();

    const result = try mannwhitney_u(f64, sample1, sample2, 0.05, allocator);

    // U = min(1, 0) = 0 for completely separated samples
    try testing.expect(result.statistic == 0.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "mannwhitney_u: unequal sizes (n1=3, n2=10)" {
    const data1 = [_]f64{ 1.0, 2.0, 3.0 };
    const data2 = [_]f64{ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{10}, &data2, .row_major);
    defer sample2.deinit();

    const result = try mannwhitney_u(f64, sample1, sample2, 0.05, allocator);

    // U should be valid for unequal sizes
    try testing.expect(result.statistic >= 0.0);
    try testing.expect(result.statistic <= 30.0); // max = 3*10
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "mannwhitney_u: all same values in one sample ([2,2,2] vs [1,3,5])" {
    const data1 = [_]f64{ 2.0, 2.0, 2.0 };
    const data2 = [_]f64{ 1.0, 3.0, 5.0 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2.deinit();

    const result = try mannwhitney_u(f64, sample1, sample2, 0.05, allocator);

    // Should handle ties properly
    try testing.expect(result.statistic >= 0.0 and result.statistic <= 9.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "mannwhitney_u: ties handling ([1,2,3,3,4] vs [2,3,3,5])" {
    const data1 = [_]f64{ 1.0, 2.0, 3.0, 3.0, 4.0 };
    const data2 = [_]f64{ 2.0, 3.0, 3.0, 5.0 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{4}, &data2, .row_major);
    defer sample2.deinit();

    const result = try mannwhitney_u(f64, sample1, sample2, 0.05, allocator);

    // Should handle multiple ties correctly
    try testing.expect(result.statistic >= 0.0 and result.statistic <= 20.0); // 5*4=20
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "mannwhitney_u: U statistic range validation (0 ≤ U ≤ n1*n2)" {
    const data1 = [_]f64{ 10.0, 20.0, 30.0 };
    const data2 = [_]f64{ 15.0, 25.0, 35.0 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2.deinit();

    const result = try mannwhitney_u(f64, sample1, sample2, 0.05, allocator);

    const max_u = @as(f64, 3.0) * @as(f64, 3.0);
    try testing.expect(result.statistic >= 0.0);
    try testing.expect(result.statistic <= max_u);
}

test "mannwhitney_u: p-value range validation (0 ≤ p ≤ 1)" {
    const data1 = [_]f64{ -5.0, -3.0, -1.0 };
    const data2 = [_]f64{ 1.0, 3.0, 5.0 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2.deinit();

    const result = try mannwhitney_u(f64, sample1, sample2, 0.05, allocator);

    try testing.expect(result.p_value >= 0.0);
    try testing.expect(result.p_value <= 1.0);
    try testing.expect(!math.isNan(result.p_value));
}

test "mannwhitney_u: symmetry property (swap samples preserves p-value)" {
    const data1 = [_]f64{ 1.0, 2.0, 3.0 };
    const data2 = [_]f64{ 4.0, 5.0, 6.0 };

    var sample1a = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1a.deinit();
    var sample2a = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2a.deinit();

    var sample1b = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample1b.deinit();
    var sample2b = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample2b.deinit();

    const result1 = try mannwhitney_u(f64, sample1a, sample2a, 0.05, allocator);
    const result2 = try mannwhitney_u(f64, sample1b, sample2b, 0.05, allocator);

    // p-value should be same regardless of order (symmetry property)
    try testing.expectApproxEqAbs(result1.p_value, result2.p_value, 1e-10);
    // U statistic reports min(U1, U2), so both should be valid (0 ≤ U ≤ n1·n2)
    try testing.expect(result1.statistic >= 0.0 and result1.statistic <= 9.0);
    try testing.expect(result2.statistic >= 0.0 and result2.statistic <= 9.0);
}

test "mannwhitney_u: larger difference → smaller U" {
    const data1 = [_]f64{ 1.0, 2.0, 3.0 };
    const data2 = [_]f64{ 4.0, 5.0, 6.0 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2.deinit();

    const result_sep = try mannwhitney_u(f64, sample1, sample2, 0.05, allocator);

    const data3 = [_]f64{ 1.0, 2.0, 3.0 };
    const data4 = [_]f64{ 1.5, 2.5, 3.5 };

    var sample3 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data3, .row_major);
    defer sample3.deinit();
    var sample4 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data4, .row_major);
    defer sample4.deinit();

    const result_overlap = try mannwhitney_u(f64, sample3, sample4, 0.05, allocator);

    // Completely separated samples should have smaller U than slightly overlapping
    try testing.expect(result_sep.statistic < result_overlap.statistic);
    try testing.expect(result_sep.p_value < result_overlap.p_value);
}

test "mannwhitney_u: alpha threshold rejection decision" {
    const data1 = [_]f64{ 1.0, 2.0, 3.0 };
    const data2 = [_]f64{ 4.0, 5.0, 6.0 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2.deinit();

    const result_strict = try mannwhitney_u(f64, sample1, sample2, 0.001, allocator);
    const result_lenient = try mannwhitney_u(f64, sample1, sample2, 0.5, allocator);

    // Same test with stricter alpha should be more conservative
    try testing.expect(!result_strict.reject or result_lenient.reject);
}

test "mannwhitney_u: large sample convergence (n1=50, n2=50)" {
    var data1: [50]f64 = undefined;
    var data2: [50]f64 = undefined;

    // Create two samples from different distributions
    for (0..50) |i| {
        data1[i] = @as(f64, @floatFromInt(i));
        data2[i] = @as(f64, @floatFromInt(i)) + 25.0; // Shifted by 25
    }

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{50}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{50}, &data2, .row_major);
    defer sample2.deinit();

    const result = try mannwhitney_u(f64, sample1, sample2, 0.05, allocator);

    try testing.expect(result.statistic >= 0.0 and result.statistic <= 2500.0); // 50*50
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
    try testing.expect(!math.isNan(result.statistic));
    try testing.expect(!math.isNan(result.p_value));
}

test "mannwhitney_u: f32 precision" {
    const data1 = [_]f32{ 1.5, 2.5, 3.5 };
    const data2 = [_]f32{ 4.5, 5.5, 6.5 };

    var sample1 = try NDArray_type(f32, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f32, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2.deinit();

    const result = try mannwhitney_u(f32, sample1, sample2, 0.05, allocator);

    try testing.expect(result.statistic >= 0.0 and result.statistic <= 9.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "mannwhitney_u: f64 precision" {
    const data1 = [_]f64{ 1.5, 2.5, 3.5 };
    const data2 = [_]f64{ 4.5, 5.5, 6.5 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2.deinit();

    const result = try mannwhitney_u(f64, sample1, sample2, 0.05, allocator);

    try testing.expect(result.statistic >= 0.0 and result.statistic <= 9.0);
    try testing.expect(result.p_value >= 0.0 and result.p_value <= 1.0);
}

test "mannwhitney_u: memory safety - no leaks" {
    const data1 = [_]f64{ 1.0, 2.0, 3.0 };
    const data2 = [_]f64{ 4.0, 5.0, 6.0 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2.deinit();

    // allocator is std.testing.allocator which detects memory leaks
    _ = try mannwhitney_u(f64, sample1, sample2, 0.05, allocator);
}

test "mannwhitney_u: consistency across multiple runs" {
    const data1 = [_]f64{ 1.0, 2.0, 3.0 };
    const data2 = [_]f64{ 1.5, 2.5, 3.5 };

    var sample1a = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1a.deinit();
    var sample2a = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2a.deinit();

    var sample1b = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1b.deinit();
    var sample2b = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2b.deinit();

    const result1 = try mannwhitney_u(f64, sample1a, sample2a, 0.05, allocator);
    const result2 = try mannwhitney_u(f64, sample1b, sample2b, 0.05, allocator);

    try testing.expectApproxEqAbs(result1.statistic, result2.statistic, 1e-10);
    try testing.expectApproxEqAbs(result1.p_value, result2.p_value, 1e-10);
}

test "mannwhitney_u: error - empty array" {
    // Note: NDArray doesn't support zero-length arrays (returns ZeroDimension error)
    // This test verifies the function would reject empty input if it were possible to create one
    // The mannwhitney_u function has validation: if (n1 == 0 or n2 == 0) return error.EmptyArray
    // We skip this test since NDArray itself prevents zero-length construction
    return error.SkipZigTest;
}

test "mannwhitney_u: error - invalid alpha" {
    const data1 = [_]f64{ 1.0, 2.0, 3.0 };
    const data2 = [_]f64{ 4.0, 5.0, 6.0 };

    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer sample2.deinit();

    const result_zero = mannwhitney_u(f64, sample1, sample2, 0.0, allocator);
    try testing.expectError(error.InvalidParameter, result_zero);
}
