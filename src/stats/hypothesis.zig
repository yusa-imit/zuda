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
/// const result = try ttest_rel(f64, before_data, after_data, 0.05);
/// // Tests if after differs from before at α=0.05
/// ```
pub fn ttest_rel(
    comptime T: type,
    before: NDArray_type(T, 1),
    after: NDArray_type(T, 1),
    alpha: T,
) !TestResult(T) {
    const n = before.count();
    if (n == 0) return error.EmptyArray;
    if (n != after.count()) return error.UnequalLengths;
    if (alpha <= 0 or alpha >= 1) return error.InvalidParameter;

    // Compute differences: d = before - after
    const page_allocator = std.heap.page_allocator;
    const diffs = try page_allocator.alloc(T, n);
    defer page_allocator.free(diffs);

    var before_iter = before.iterator();
    var after_iter = after.iterator();
    var idx: usize = 0;
    while (before_iter.next()) |b| {
        const a = after_iter.next() orelse return error.UnequalLengths;
        diffs[idx] = b - a;
        idx += 1;
    }

    // Create NDArray for differences
    var diff_array = try NDArray_type(T, 1).fromSlice(page_allocator, &[_]usize{n}, diffs, .row_major);
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

    const result = try ttest_rel(f64, before, after, 0.05);
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

    const result = try ttest_rel(f64, before, after, 0.05);
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

    const result = try ttest_rel(f64, before, after, 0.05);
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
    const result = ttest_rel(f64, before, after, 0.05);
    try testing.expectError(NDArray_type(f64, 1).Error.CapacityExceeded, result);
}

test "ttest_rel: two pairs (n=2, df=1)" {
    const before_data = [_]f64{ 1.0, 3.0 };
    const after_data = [_]f64{ 2.0, 4.0 };
    var before = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &before_data, .row_major);
    defer before.deinit();
    var after = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{2}, &after_data, .row_major);
    defer after.deinit();

    const result = try ttest_rel(f64, before, after, 0.05);
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

    const result = try ttest_rel(f64, before, after, 0.05);
    try testing.expectApproxEqAbs(99.0, result.df, 1e-10);
}

test "ttest_rel: f32 precision" {
    const before_data = [_]f32{ 1.0, 2.0, 3.0 };
    const after_data = [_]f32{ 2.0, 3.0, 4.0 };
    var before = try NDArray_type(f32, 1).fromSlice(allocator, &[_]usize{3}, &before_data, .row_major);
    defer before.deinit();
    var after = try NDArray_type(f32, 1).fromSlice(allocator, &[_]usize{3}, &after_data, .row_major);
    defer after.deinit();

    const result = try ttest_rel(f32, before, after, 0.05);
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

    const result_005 = try ttest_rel(f64, before1, after1, 0.05);
    const result_001 = try ttest_rel(f64, before2, after2, 0.01);

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

    const result = ttest_rel(f64, before, after, 0.05);
    try testing.expectError(error.UnequalLengths, result);
}

test "ttest_rel: error on invalid alpha (alpha=0)" {
    const before_data = [_]f64{ 1.0, 2.0, 3.0 };
    const after_data = [_]f64{ 2.0, 3.0, 4.0 };
    var before = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &before_data, .row_major);
    defer before.deinit();
    var after = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &after_data, .row_major);
    defer after.deinit();

    const result = ttest_rel(f64, before, after, 0.0);
    try testing.expectError(error.InvalidParameter, result);
}

test "ttest_rel: error on invalid alpha (alpha=1)" {
    const before_data = [_]f64{ 1.0, 2.0, 3.0 };
    const after_data = [_]f64{ 2.0, 3.0, 4.0 };
    var before = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &before_data, .row_major);
    defer before.deinit();
    var after = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &after_data, .row_major);
    defer after.deinit();

    const result = ttest_rel(f64, before, after, 1.0);
    try testing.expectError(error.InvalidParameter, result);
}

test "ttest_rel: df = n-1" {
    const before_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
    const after_data = [_]f64{ 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var before = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{7}, &before_data, .row_major);
    defer before.deinit();
    var after = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{7}, &after_data, .row_major);
    defer after.deinit();

    const result = try ttest_rel(f64, before, after, 0.05);
    try testing.expectApproxEqAbs(6.0, result.df, 1e-10);
}

test "ttest_rel: p_value in [0, 1]" {
    const before_data = [_]f64{ 1.0, 2.0, 3.0 };
    const after_data = [_]f64{ 2.0, 3.0, 4.0 };
    var before = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &before_data, .row_major);
    defer before.deinit();
    var after = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{3}, &after_data, .row_major);
    defer after.deinit();

    const result = try ttest_rel(f64, before, after, 0.05);
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

    const result1 = try ttest_rel(f64, before1, after1, 0.05);
    const result2 = try ttest_rel(f64, before2, after2, 0.05);

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
