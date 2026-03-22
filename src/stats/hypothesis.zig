//! Hypothesis Testing — Statistical tests for comparing means and proportions
//!
//! This module provides classical hypothesis testing procedures for comparing
//! population means, including Student's t-tests (one-sample, independent samples, paired).
//!
//! ## Supported Tests
//! - `ttest_1samp` — One-sample t-test (H0: sample mean = μ)
//! - `ttest_ind` — Independent samples t-test (H0: μ₁ = μ₂)
//! - `ttest_rel` — Paired samples t-test (H0: μ_diff = 0)
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
//!
//! ## Use Cases
//! - Comparing sample mean to population mean
//! - Comparing means of two independent groups
//! - Comparing paired observations (before/after, matched controls)
//! - Hypothesis testing with small to moderate sample sizes
//! - Confidence intervals for mean differences

const std = @import("std");
const math = std.math;
const testing = std.testing;

// Import dependencies
const descriptive = @import("descriptive.zig");
const NDArray_type = @import("../ndarray/ndarray.zig").NDArray;
const StudentT_Distribution = @import("distributions/student_t.zig").StudentT;

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
    const data1 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const data2 = [_]f64{ 5.0, 10.0, 15.0, 20.0, 25.0 }; // Much higher variance
    var sample1 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data1, .row_major);
    defer sample1.deinit();
    var sample2 = try NDArray_type(f64, 1).fromSlice(allocator, &[_]usize{5}, &data2, .row_major);
    defer sample2.deinit();

    const welch_result = try ttest_ind(f64, sample1, sample2, 0.05, false);
    const pooled_result = try ttest_ind(f64, sample1, sample2, 0.05, true);

    // Welch and pooled should differ for unequal variances
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
    const before_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const after_data = [_]f64{ 6.0, 7.0, 8.0, 9.0, 10.0 };
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
    const before_data = [_]f64{ 6.0, 7.0, 8.0, 9.0, 10.0 };
    const after_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
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
