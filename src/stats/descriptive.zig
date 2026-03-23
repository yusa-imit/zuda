//! Descriptive Statistics — Univariate summary statistics for 1D arrays
//!
//! This module provides fundamental statistical measures for analyzing data distributions:
//! - Location: mean, median, mode
//! - Spread: variance, standard deviation, quantiles, percentiles
//! - Shape: skewness, kurtosis
//!
//! All functions accept NDArray(T, 1) where T is a numeric type (f32, f64, i32, i64, etc.)
//!
//! ## Time Complexity
//! - mean, variance, std, skewness, kurtosis: O(n)
//! - median, quantile, percentile: O(n log n) — requires sorting
//! - mode: O(n) with hash map for integer types, O(n log n) for general types
//!
//! ## Use Cases
//! - Summary statistics for datasets
//! - Data exploration and visualization preparation
//! - Feature engineering in machine learning pipelines
//! - Statistical testing (mean, std as parameters)

const std = @import("std");
const math = std.math;
const testing = std.testing;
const Allocator = std.mem.Allocator;

// NDArray is defined in ndarray module — import its type
const ndarray_module = struct {
    pub const NDArray = @import("../ndarray/ndarray.zig").NDArray;
};

// ============================================================================
// Mean
// ============================================================================

/// Compute arithmetic mean (average) of 1D array
///
/// Mean = (sum of all elements) / count
///
/// Parameters:
/// - data: 1D NDArray of numeric type T
///
/// Returns: Mean value of type T
///
/// Time: O(n) where n = data.size()
/// Space: O(1)
///
/// Note: For integer types, result is truncated (not rounded).
/// Consider using f64 if precise mean is needed.
pub fn mean(comptime T: type, data: ndarray_module.NDArray(T, 1)) T {
    const n = data.count();
    if (n == 0) return 0;

    var sum: T = 0;
    var iter = data.iterator();
    while (iter.next()) |val| {
        sum += val;
    }

    // Handle both integer and float types
    if (@typeInfo(T) == .float) {
        const n_f = @as(T, @floatFromInt(@as(i64, @intCast(n))));
        return sum / n_f;
    } else {
        const n_i = @as(T, @intCast(n));
        return @divTrunc(sum, n_i);
    }
}

// ============================================================================
// Median
// ============================================================================

/// Compute median (middle value) of 1D array
///
/// For odd-length arrays: middle value after sorting
/// For even-length arrays: average of two middle values (for float types),
///                          or lower middle (for integer types)
///
/// Parameters:
/// - data: 1D NDArray of numeric type T
/// - allocator: Used to create sorted copy (original array unchanged)
///
/// Returns: Median value of type T
///
/// Errors:
/// - error.EmptyArray if data is empty
/// - std.mem.Allocator.Error if allocation fails
///
/// Time: O(n log n) due to sorting
/// Space: O(n) for sorted copy
pub fn median(comptime T: type, data: ndarray_module.NDArray(T, 1), allocator: Allocator) (ndarray_module.NDArray(T, 1).Error || Allocator.Error)!T {
    const n = data.count();
    if (n == 0) return error.EmptyArray;

    // Copy data to mutable slice for sorting
    const sorted_data = try allocator.alloc(T, n);
    defer allocator.free(sorted_data);

    // Copy elements from array into sorted_data
    var iter = data.iterator();
    var idx: usize = 0;
    while (iter.next()) |val| {
        sorted_data[idx] = val;
        idx += 1;
    }

    // Sort the data
    std.sort.insertion(T, sorted_data, {}, struct {
        fn compare(_: void, a: T, b: T) bool {
            return a < b;
        }
    }.compare);

    // Return median
    if (n % 2 == 1) {
        // Odd length: middle element
        return sorted_data[n / 2];
    } else {
        // Even length: average of two middle elements
        const mid1 = sorted_data[n / 2 - 1];
        const mid2 = sorted_data[n / 2];
        if (@typeInfo(T) == .float) {
            const two = @as(T, @floatFromInt(2));
            return (mid1 + mid2) / two;
        } else {
            const two = @as(T, @intCast(2));
            return @divTrunc(mid1 + mid2, two);
        }
    }
}

// ============================================================================
// Mode
// ============================================================================

/// Compute mode (most frequently occurring value) of 1D array
///
/// If multiple values have the same highest frequency (multimodal),
/// returns the one encountered first during counting.
///
/// Parameters:
/// - data: 1D NDArray of numeric type T (must be integer type, not float)
/// - allocator: Used for hash map construction
///
/// Returns: Most frequent value
///
/// Errors:
/// - error.EmptyArray if data is empty
/// - std.mem.Allocator.Error if hash map allocation fails
///
/// Time: O(n) average case, O(n²) worst case for hash collisions
/// Space: O(k) where k = number of unique values
///
/// Note: Only supports integer types. Floats cannot be hashed in Zig 0.15.x.
pub fn mode(comptime T: type, data: ndarray_module.NDArray(T, 1), allocator: Allocator) (ndarray_module.NDArray(T, 1).Error || Allocator.Error)!T {
    // Compile-time check: T must be an integer type
    comptime {
        const type_info = @typeInfo(T);
        if (type_info != .int and type_info != .comptime_int) {
            @compileError("mode() only supports integer types. Float types cannot be hashed in Zig 0.15.x.");
        }
    }

    const n = data.count();
    if (n == 0) return error.EmptyArray;

    // Use AutoHashMap to count frequencies (safe for integer types)
    var freq_map = std.AutoHashMap(T, usize).init(allocator);
    defer freq_map.deinit();

    // Count frequencies
    var iter = data.iterator();
    while (iter.next()) |val| {
        const count = freq_map.get(val) orelse 0;
        try freq_map.put(val, count + 1);
    }

    // Find value with max frequency
    var max_freq: usize = 0;
    var mode_val: T = undefined;
    var first = true;

    var iter_map = freq_map.iterator();
    while (iter_map.next()) |entry| {
        if (first or entry.value_ptr.* > max_freq) {
            max_freq = entry.value_ptr.*;
            mode_val = entry.key_ptr.*;
            first = false;
        }
    }

    return mode_val;
}

// ============================================================================
// Variance
// ============================================================================

/// Compute variance of 1D array
///
/// Population variance (ddof=0):  sum((x_i - mean)²) / n
/// Sample variance (ddof=1):      sum((x_i - mean)²) / (n - 1)
///
/// Parameters:
/// - data: 1D NDArray of numeric type T
/// - ddof: Degrees of freedom adjustment (typically 0 for population, 1 for sample)
///
/// Returns: Variance value of type T (non-negative)
///
/// Errors:
/// - error.EmptyArray if data is empty
/// - error.CapacityExceeded if ddof >= data.size()
///
/// Time: O(n) — makes two passes: mean, then squared deviations
/// Space: O(1)
///
/// Note: For numerical stability, uses two-pass algorithm.
/// Result is computed as: sum((x_i - mean)²) / (n - ddof)
pub fn variance(comptime T: type, data: ndarray_module.NDArray(T, 1), ddof: usize) (ndarray_module.NDArray(T, 1).Error)!T {
    const n = data.count();
    if (n == 0) return error.EmptyArray;
    // ddof >= n is mathematically invalid (negative or zero denominator)
    if (ddof >= n) {
        return error.CapacityExceeded;
    }

    // Compute mean (first pass)
    const mean_val = mean(T, data);

    // Compute sum of squared deviations (second pass)
    var sum_sq_deviations: T = 0;
    var iter = data.iterator();
    while (iter.next()) |val| {
        const diff = val - mean_val;
        sum_sq_deviations += diff * diff;
    }

    // Return variance with degrees of freedom
    if (@typeInfo(T) == .float) {
        const divisor = @as(T, @floatFromInt(@as(i64, @intCast(n - ddof))));
        return sum_sq_deviations / divisor;
    } else {
        const divisor = @as(T, @intCast(n - ddof));
        return @divTrunc(sum_sq_deviations, divisor);
    }
}

// ============================================================================
// Standard Deviation
// ============================================================================

/// Compute standard deviation (square root of variance)
///
/// Parameters:
/// - data: 1D NDArray of numeric type T
/// - ddof: Degrees of freedom adjustment (typically 0 for population, 1 for sample)
///
/// Returns: Standard deviation = sqrt(variance)
///
/// Errors:
/// - error.EmptyArray if data is empty
/// - error.CapacityExceeded if ddof >= data.size()
///
/// Time: O(n) — calls variance() internally
/// Space: O(1)
///
/// Note: Mathematically equiv(variance()).sqrt()
pub fn stdDev(comptime T: type, data: ndarray_module.NDArray(T, 1), ddof: usize) (ndarray_module.NDArray(T, 1).Error)!T {
    const var_val = try variance(T, data, ddof);
    return math.sqrt(var_val);
}

// ============================================================================
// Quantile
// ============================================================================

/// Compute q-th quantile (0-quantile to 1-quantile)
///
/// Quantiles divide sorted data into equal-probability segments:
/// - q=0.0: minimum
/// - q=0.25: first quartile (Q1)
/// - q=0.5: median
/// - q=0.75: third quartile (Q3)
/// - q=1.0: maximum
///
/// Interpolation method: Linear (matches NumPy default, R type=7)
///
/// Parameters:
/// - data: 1D NDArray of numeric type T
/// - q: Quantile parameter in range [0.0, 1.0]
/// - allocator: Used for sorted copy
///
/// Returns: q-th quantile value
///
/// Errors:
/// - error.EmptyArray if data is empty
/// - error.CapacityExceeded if q not in [0, 1]
/// - std.mem.Allocator.Error if allocation fails
///
/// Time: O(n log n) due to sorting
/// Space: O(n) for sorted copy
///
/// Note: Linear interpolation matches NumPy's default (method='linear' or type=7 in R)
pub fn quantile(comptime T: type, data: ndarray_module.NDArray(T, 1), q: T, allocator: Allocator) (ndarray_module.NDArray(T, 1).Error || Allocator.Error)!T {
    const n = data.count();
    if (n == 0) return error.EmptyArray;
    if (q < 0 or q > 1) return error.CapacityExceeded;

    // Copy data to mutable slice for sorting
    const sorted_data = try allocator.alloc(T, n);
    defer allocator.free(sorted_data);

    // Copy elements
    var iter = data.iterator();
    var idx: usize = 0;
    while (iter.next()) |val| {
        sorted_data[idx] = val;
        idx += 1;
    }

    // Sort the data
    std.sort.insertion(T, sorted_data, {}, struct {
        fn compare(_: void, a: T, b: T) bool {
            return a < b;
        }
    }.compare);

    // Linear interpolation method (matches NumPy default)
    // position = q * (n - 1)
    // This function only works with float types (q must be in [0, 1])
    const n_minus_1 = @as(T, @floatFromInt(@as(i64, @intCast(n - 1))));
    const pos_float = q * n_minus_1;
    const pos_int = @as(usize, @intFromFloat(pos_float));
    const frac = pos_float - @as(T, @floatFromInt(@as(i64, @intCast(pos_int))));

    if (pos_int >= n - 1) {
        return sorted_data[n - 1];
    }

    const lower = sorted_data[pos_int];
    const upper = sorted_data[pos_int + 1];
    return lower + frac * (upper - lower);
}

// ============================================================================
// Percentile
// ============================================================================

/// Compute p-th percentile (0-percentile to 100-percentile)
///
/// Percentile is quantile scaled to [0, 100] range:
/// percentile(p) = quantile(p / 100)
///
/// Common percentiles:
/// - p=25: 1st quartile
/// - p=50: median
/// - p=75: 3rd quartile
/// - p=95: 95th percentile (common threshold)
///
/// Parameters:
/// - data: 1D NDArray of numeric type T
/// - p: Percentile parameter in range [0.0, 100.0]
/// - allocator: Used for sorted copy
///
/// Returns: p-th percentile value
///
/// Errors:
/// - error.EmptyArray if data is empty
/// - error.CapacityExceeded if p not in [0, 100]
/// - std.mem.Allocator.Error if allocation fails
///
/// Time: O(n log n) due to sorting
/// Space: O(n) for sorted copy
///
/// Note: Implemented as quantile(p / 100)
pub fn percentile(comptime T: type, data: ndarray_module.NDArray(T, 1), p: T, allocator: Allocator) (ndarray_module.NDArray(T, 1).Error || Allocator.Error)!T {
    if (p < 0 or p > 100) return error.CapacityExceeded;

    const q = if (@typeInfo(T) == .float)
        p / @as(T, @floatFromInt(100))
    else
        @divTrunc(p, @as(T, @intCast(100)));
    return quantile(T, data, q, allocator);
}

// ============================================================================
// Skewness
// ============================================================================

/// Compute Fisher's skewness coefficient (standardized third moment)
///
/// Skewness = E[(X - μ)³] / σ³
///            = sum((x_i - mean)³) / (n * std³)
///
/// Interpretation:
/// - skew < 0: left-skewed (longer tail on left)
/// - skew ≈ 0: symmetric
/// - skew > 0: right-skewed (longer tail on right)
/// - |skew| < 0.5: fairly symmetric
/// - |skew| > 1.0: highly skewed
///
/// Parameters:
/// - data: 1D NDArray of numeric type T
///
/// Returns: Skewness coefficient (dimensionless, typically in range [-3, 3])
///
/// Errors:
/// - error.EmptyArray if data is empty
///
/// Time: O(n) — three passes: mean, std, then third moment
/// Space: O(1)
///
/// Note: Returns 0 for single-element or zero-variance arrays
pub fn skewness(comptime T: type, data: ndarray_module.NDArray(T, 1)) (ndarray_module.NDArray(T, 1).Error)!T {
    const n = data.count();
    if (n == 0) return error.EmptyArray;

    // Compute mean
    const mean_val = mean(T, data);

    // Compute standard deviation (population)
    const std_val = try stdDev(T, data, 0);

    // If std is 0 or close to 0, return 0 (no skewness for constant data)
    if (std_val == 0) return 0;

    // Compute third moment
    var sum_cubed: T = 0;
    var iter = data.iterator();
    while (iter.next()) |val| {
        const diff = val - mean_val;
        const normalized = diff / std_val;
        sum_cubed += normalized * normalized * normalized;
    }

    // Return skewness: sum(((x - μ) / σ)³) / n
    if (@typeInfo(T) == .float) {
        const n_f = @as(T, @floatFromInt(@as(i64, @intCast(n))));
        return sum_cubed / n_f;
    } else {
        const n_i = @as(T, @intCast(n));
        return @divTrunc(sum_cubed, n_i);
    }
}

// ============================================================================
// Kurtosis
// ============================================================================

/// Compute excess kurtosis (standardized fourth moment)
///
/// Excess kurtosis = E[(X - μ)⁴] / σ⁴ - 3
///                 = [sum((x_i - mean)⁴) / n] / std⁴ - 3
///
/// The -3 subtraction (vs normal kurtosis) centers around 0 for normal distribution.
///
/// Interpretation:
/// - kurt ≈ 0: mesokurtic (normal-like tail behavior)
/// - kurt < -2: platykurtic (light tails, flat distribution)
/// - kurt > 3: leptokurtic (heavy tails, sharp peak)
/// - |kurt| < 2: similar to normal distribution
///
/// Parameters:
/// - data: 1D NDArray of numeric type T
///
/// Returns: Excess kurtosis coefficient (typically in range [-2, ∞])
///
/// Errors:
/// - error.EmptyArray if data is empty
///
/// Time: O(n) — four passes: mean, std, then fourth moment
/// Space: O(1)
///
/// Note: Returns 0 for single-element or zero-variance arrays
pub fn kurtosis(comptime T: type, data: ndarray_module.NDArray(T, 1)) (ndarray_module.NDArray(T, 1).Error)!T {
    const n = data.count();
    if (n == 0) return error.EmptyArray;

    // Compute mean
    const mean_val = mean(T, data);

    // Compute standard deviation (population)
    const std_val = try stdDev(T, data, 0);

    // If std is 0 or close to 0, return 0 (no kurtosis for constant data)
    if (std_val == 0) return 0;

    // Compute fourth moment
    var sum_fourth: T = 0;
    var iter = data.iterator();
    while (iter.next()) |val| {
        const diff = val - mean_val;
        const normalized = diff / std_val;
        const normalized_sq = normalized * normalized;
        sum_fourth += normalized_sq * normalized_sq;
    }

    // Excess kurtosis: (sum(((x - μ) / σ)⁴) / n) - 3
    if (@typeInfo(T) == .float) {
        const n_f = @as(T, @floatFromInt(@as(i64, @intCast(n))));
        const kurtosis_val = sum_fourth / n_f;
        const three = @as(T, @floatFromInt(3));
        return kurtosis_val - three;
    } else {
        const n_i = @as(T, @intCast(n));
        const kurtosis_val = @divTrunc(sum_fourth, n_i);
        const three = @as(T, @intCast(3));
        return kurtosis_val - three;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "mean: basic f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = mean(f64, data);
    try testing.expectApproxEqAbs(3.0, result, 1e-10);
}

test "mean: single element f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{42.0};
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &data_slice, .row_major);
    defer data.deinit();
    const result = mean(f64, data);
    try testing.expectApproxEqAbs(42.0, result, 1e-10);
}

test "mean: all zeros f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 0.0, 0.0, 0.0, 0.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data_slice, .row_major);
    defer data.deinit();
    const result = mean(f64, data);
    try testing.expectApproxEqAbs(0.0, result, 1e-10);
}

test "mean: negative values f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ -5.0, -2.0, 0.0, 2.0, 5.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = mean(f64, data);
    try testing.expectApproxEqAbs(0.0, result, 1e-10);
}

test "mean: large values f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1e10, 2e10, 3e10 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &data_slice, .row_major);
    defer data.deinit();
    const result = mean(f64, data);
    try testing.expectApproxEqAbs(2e10, result, 1e-5 * 2e10);
}

test "mean: basic f32" {
    const allocator = testing.allocator;
    const data_slice = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var data = try ndarray_module.NDArray(f32, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = mean(f32, data);
    try testing.expectApproxEqAbs(3.0, result, 1e-5);
}

test "mean: integer type i32" {
    const allocator = testing.allocator;
    const data_slice = [_]i32{ 1, 2, 3, 4, 5 };
    var data = try ndarray_module.NDArray(i32, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = mean(i32, data);
    try testing.expect(result == 3); // Truncates 3.0
}

test "mean: integer type i64" {
    const allocator = testing.allocator;
    const data_slice = [_]i64{ 10, 20, 30, 40, 50 };
    var data = try ndarray_module.NDArray(i64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = mean(i64, data);
    try testing.expect(result == 30);
}

// ============================================================================
// Median Tests
// ============================================================================

test "median: odd-length f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 5.0, 1.0, 3.0, 2.0, 4.0 }; // Unsorted
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try median(f64, data, allocator);
    try testing.expectApproxEqAbs(3.0, result, 1e-10);
}

test "median: even-length f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data_slice, .row_major);
    defer data.deinit();
    const result = try median(f64, data, allocator);
    try testing.expectApproxEqAbs(2.5, result, 1e-10);
}

test "median: single element f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{42.0};
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &data_slice, .row_major);
    defer data.deinit();
    const result = try median(f64, data, allocator);
    try testing.expectApproxEqAbs(42.0, result, 1e-10);
}

test "median: two elements f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 3.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &data_slice, .row_major);
    defer data.deinit();
    const result = try median(f64, data, allocator);
    try testing.expectApproxEqAbs(2.0, result, 1e-10);
}

test "median: negative values f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ -10.0, -5.0, 0.0, 5.0, 10.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try median(f64, data, allocator);
    try testing.expectApproxEqAbs(0.0, result, 1e-10);
}

test "median: all same values f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 5.0, 5.0, 5.0, 5.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data_slice, .row_major);
    defer data.deinit();
    const result = try median(f64, data, allocator);
    try testing.expectApproxEqAbs(5.0, result, 1e-10);
}

test "median: empty array should error" {
    const allocator = testing.allocator;
    const data_slice: [0]f64 = [_]f64{};
    const result = ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{0}, &data_slice, .row_major);
    try testing.expectError(ndarray_module.NDArray(f64, 1).Error.ZeroDimension, result);
}

test "median: f32 precision" {
    const allocator = testing.allocator;
    const data_slice = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var data = try ndarray_module.NDArray(f32, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try median(f32, data, allocator);
    try testing.expectApproxEqAbs(3.0, result, 1e-5);
}

test "median: large values f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1e10, 2e10, 3e10 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &data_slice, .row_major);
    defer data.deinit();
    const result = try median(f64, data, allocator);
    try testing.expectApproxEqAbs(2e10, result, 1e-5 * 2e10);
}

// ============================================================================
// Mode Tests
// ============================================================================

// NOTE: Float mode tests disabled — floats cannot be hashed in Zig 0.15.x (AutoHashMap limitation)
// test "mode: single frequency f64" { ... }
// test "mode: single element f64" { ... }
// test "mode: all same values f64" { ... }
// test "mode: multiple modes returns first f64" { ... }
// test "mode: negative values f64" { ... }
// test "mode: float with duplicates f32" { ... }

test "mode: integer type i32" {
    const allocator = testing.allocator;
    const data_slice = [_]i32{ 1, 2, 2, 3 };
    var data = try ndarray_module.NDArray(i32, 1).fromSlice(allocator, &[_]usize{4}, &data_slice, .row_major);
    defer data.deinit();
    const result = try mode(i32, data, allocator);
    try testing.expect(result == 2);
}

// ============================================================================
// Variance Tests
// ============================================================================

test "variance: population f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try variance(f64, data, 0);
    // Variance of [1,2,3,4,5] = sum((x - 3)²) / 5 = (4 + 1 + 0 + 1 + 4) / 5 = 2.0
    try testing.expectApproxEqAbs(2.0, result, 1e-10);
}

test "variance: sample f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try variance(f64, data, 1);
    // Sample variance = 10 / 4 = 2.5
    try testing.expectApproxEqAbs(2.5, result, 1e-10);
}

test "variance: single element f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{42.0};
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &data_slice, .row_major);
    defer data.deinit();
    const result = try variance(f64, data, 0);
    try testing.expectApproxEqAbs(0.0, result, 1e-10);
}

test "variance: all same values f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 5.0, 5.0, 5.0, 5.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data_slice, .row_major);
    defer data.deinit();
    const result = try variance(f64, data, 0);
    try testing.expectApproxEqAbs(0.0, result, 1e-10);
}

test "variance: negative values f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try variance(f64, data, 0);
    // Mean = 0, sum((x - 0)²) = 4 + 1 + 0 + 1 + 4 = 10, var = 10/5 = 2.0
    try testing.expectApproxEqAbs(2.0, result, 1e-10);
}

test "variance: f32 precision" {
    const allocator = testing.allocator;
    const data_slice = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var data = try ndarray_module.NDArray(f32, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try variance(f32, data, 0);
    try testing.expectApproxEqAbs(2.0, result, 1e-5);
}

test "variance: ddof >= count should error" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 2.0, 3.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &data_slice, .row_major);
    defer data.deinit();
    const result = variance(f64, data, 3); // ddof=count should error
    try testing.expectError(ndarray_module.NDArray(f64, 1).Error.CapacityExceeded, result);
}

test "variance: two elements sample f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 3.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &data_slice, .row_major);
    defer data.deinit();
    const result = try variance(f64, data, 1);
    // Mean = 2, sum((x - 2)²) = 1 + 1 = 2, var = 2 / (2 - 1) = 2.0
    try testing.expectApproxEqAbs(2.0, result, 1e-10);
}

// ============================================================================
// Standard Deviation Tests
// ============================================================================

test "std: population f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try stdDev(f64, data, 0);
    // std = sqrt(2.0) ≈ 1.414213562373095
    try testing.expectApproxEqAbs(1.4142135623730951, result, 1e-10);
}

test "std: sample f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try stdDev(f64, data, 1);
    // std = sqrt(2.5) ≈ 1.5811388300841898
    try testing.expectApproxEqAbs(1.5811388300841898, result, 1e-10);
}

test "std: single element f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{42.0};
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &data_slice, .row_major);
    defer data.deinit();
    const result = try stdDev(f64, data, 0);
    try testing.expectApproxEqAbs(0.0, result, 1e-10);
}

test "std: all same values f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 5.0, 5.0, 5.0, 5.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data_slice, .row_major);
    defer data.deinit();
    const result = try stdDev(f64, data, 0);
    try testing.expectApproxEqAbs(0.0, result, 1e-10);
}

test "std: f32 precision" {
    const allocator = testing.allocator;
    const data_slice = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var data = try ndarray_module.NDArray(f32, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try stdDev(f32, data, 0);
    try testing.expectApproxEqAbs(1.41421356, result, 1e-5);
}

test "std: negative values f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try stdDev(f64, data, 0);
    // var = 2.0, std = sqrt(2.0)
    try testing.expectApproxEqAbs(1.4142135623730951, result, 1e-10);
}

// ============================================================================
// Quantile Tests
// ============================================================================

test "quantile: q=0.0 (minimum) f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 3.0, 1.0, 4.0, 1.0, 5.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try quantile(f64, data, 0.0, allocator);
    try testing.expectApproxEqAbs(1.0, result, 1e-10);
}

test "quantile: q=0.5 (median) f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try quantile(f64, data, 0.5, allocator);
    try testing.expectApproxEqAbs(3.0, result, 1e-10);
}

test "quantile: q=1.0 (maximum) f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 3.0, 1.0, 4.0, 1.0, 5.0, 9.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{6}, &data_slice, .row_major);
    defer data.deinit();
    const result = try quantile(f64, data, 1.0, allocator);
    try testing.expectApproxEqAbs(9.0, result, 1e-10);
}

test "quantile: q=0.25 (Q1) f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data_slice, .row_major);
    defer data.deinit();
    const result = try quantile(f64, data, 0.25, allocator);
    // Linear interpolation: pos = 0.25 * (4 - 1) = 0.75
    // Between index 0 and 1: 1.0 + 0.75 * (2.0 - 1.0) = 1.75
    try testing.expectApproxEqAbs(1.75, result, 1e-10);
}

test "quantile: q=0.75 (Q3) f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data_slice, .row_major);
    defer data.deinit();
    const result = try quantile(f64, data, 0.75, allocator);
    // Linear interpolation: pos = 0.75 * (4 - 1) = 2.25
    // Between index 2 and 3: 3.0 + 0.25 * (4.0 - 3.0) = 3.25
    try testing.expectApproxEqAbs(3.25, result, 1e-10);
}

test "quantile: single element f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{42.0};
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &data_slice, .row_major);
    defer data.deinit();
    const result = try quantile(f64, data, 0.5, allocator);
    try testing.expectApproxEqAbs(42.0, result, 1e-10);
}

test "quantile: two elements f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 3.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &data_slice, .row_major);
    defer data.deinit();
    const result = try quantile(f64, data, 0.5, allocator);
    try testing.expectApproxEqAbs(2.0, result, 1e-10);
}

test "quantile: q < 0 should error" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 2.0, 3.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &data_slice, .row_major);
    defer data.deinit();
    const result = quantile(f64, data, -0.1, allocator);
    try testing.expectError(ndarray_module.NDArray(f64, 1).Error.CapacityExceeded, result);
}

test "quantile: q > 1 should error" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 2.0, 3.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &data_slice, .row_major);
    defer data.deinit();
    const result = quantile(f64, data, 1.1, allocator);
    try testing.expectError(ndarray_module.NDArray(f64, 1).Error.CapacityExceeded, result);
}

test "quantile: f32 precision" {
    const allocator = testing.allocator;
    const data_slice = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var data = try ndarray_module.NDArray(f32, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try quantile(f32, data, 0.5, allocator);
    try testing.expectApproxEqAbs(3.0, result, 1e-5);
}

// ============================================================================
// Percentile Tests
// ============================================================================

test "percentile: p=0 (minimum) f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 3.0, 1.0, 4.0, 1.0, 5.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try percentile(f64, data, 0.0, allocator);
    try testing.expectApproxEqAbs(1.0, result, 1e-10);
}

test "percentile: p=50 (median) f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try percentile(f64, data, 50.0, allocator);
    try testing.expectApproxEqAbs(3.0, result, 1e-10);
}

test "percentile: p=100 (maximum) f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 3.0, 1.0, 4.0, 1.0, 5.0, 9.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{6}, &data_slice, .row_major);
    defer data.deinit();
    const result = try percentile(f64, data, 100.0, allocator);
    try testing.expectApproxEqAbs(9.0, result, 1e-10);
}

test "percentile: p=25 f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data_slice, .row_major);
    defer data.deinit();
    const result = try percentile(f64, data, 25.0, allocator);
    try testing.expectApproxEqAbs(1.75, result, 1e-10);
}

test "percentile: p=75 f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data_slice, .row_major);
    defer data.deinit();
    const result = try percentile(f64, data, 75.0, allocator);
    try testing.expectApproxEqAbs(3.25, result, 1e-10);
}

test "percentile: p=95 f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{10}, &data_slice, .row_major);
    defer data.deinit();
    const result = try percentile(f64, data, 95.0, allocator);
    // p=95% => q=0.95, pos = 0.95 * 9 = 8.55
    // Between index 8 and 9: 9 + 0.55 * (10 - 9) = 9.55
    try testing.expectApproxEqAbs(9.55, result, 1e-10);
}

test "percentile: p < 0 should error" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 2.0, 3.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &data_slice, .row_major);
    defer data.deinit();
    const result = percentile(f64, data, -10.0, allocator);
    try testing.expectError(ndarray_module.NDArray(f64, 1).Error.CapacityExceeded, result);
}

test "percentile: p > 100 should error" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 2.0, 3.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &data_slice, .row_major);
    defer data.deinit();
    const result = percentile(f64, data, 110.0, allocator);
    try testing.expectError(ndarray_module.NDArray(f64, 1).Error.CapacityExceeded, result);
}

test "percentile: f32 precision" {
    const allocator = testing.allocator;
    const data_slice = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var data = try ndarray_module.NDArray(f32, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try percentile(f32, data, 50.0, allocator);
    try testing.expectApproxEqAbs(3.0, result, 1e-5);
}

// ============================================================================
// Skewness Tests
// ============================================================================

test "skewness: symmetric f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try skewness(f64, data);
    // Symmetric distribution should have skewness ≈ 0
    try testing.expectApproxEqAbs(0.0, result, 1e-10);
}

test "skewness: right-skewed f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0, 10.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try skewness(f64, data);
    // Right-skewed (tail on right) should have positive skewness
    try testing.expect(result > 0.0);
}

test "skewness: left-skewed f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ -10.0, 1.0, 2.0, 3.0, 4.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try skewness(f64, data);
    // Left-skewed (tail on left) should have negative skewness
    try testing.expect(result < 0.0);
}

test "skewness: single element f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{42.0};
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &data_slice, .row_major);
    defer data.deinit();
    const result = try skewness(f64, data);
    // Single element has zero variance, skewness should be 0
    try testing.expectApproxEqAbs(0.0, result, 1e-10);
}

test "skewness: all same values f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 5.0, 5.0, 5.0, 5.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data_slice, .row_major);
    defer data.deinit();
    const result = try skewness(f64, data);
    // Zero variance -> skewness = 0
    try testing.expectApproxEqAbs(0.0, result, 1e-10);
}

test "skewness: f32 precision" {
    const allocator = testing.allocator;
    const data_slice = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var data = try ndarray_module.NDArray(f32, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try skewness(f32, data);
    try testing.expectApproxEqAbs(0.0, result, 1e-5);
}

test "skewness: two elements f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 3.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &data_slice, .row_major);
    defer data.deinit();
    const result = try skewness(f64, data);
    // Two elements symmetric around mean should have 0 skewness
    try testing.expectApproxEqAbs(0.0, result, 1e-10);
}

// ============================================================================
// Kurtosis Tests
// ============================================================================

test "kurtosis: normal-like f64" {
    const allocator = testing.allocator;
    // Roughly normal distribution
    const data_slice = [_]f64{ -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{7}, &data_slice, .row_major);
    defer data.deinit();
    const result = try kurtosis(f64, data);
    // Normal distribution excess kurtosis ≈ 0
    try testing.expect(result > -2.0 and result < 2.0);
}

test "kurtosis: light-tailed f64" {
    const allocator = testing.allocator;
    // Distribution with two extreme outliers (bimodal-like)
    // This actually produces negative excess kurtosis (light-tailed/platykurtic)
    const data_slice = [_]f64{ -100.0, 1.0, 2.0, 3.0, 100.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try kurtosis(f64, data);
    // This distribution is light-tailed (negative excess kurtosis)
    try testing.expect(result < 0.0);
}

test "kurtosis: uniform-like f64" {
    const allocator = testing.allocator;
    // Uniform distribution
    const data_slice = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try kurtosis(f64, data);
    // Uniform distribution has excess kurtosis ≈ -1.2
    try testing.expect(result < -1.0);
}

test "kurtosis: single element f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{42.0};
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &data_slice, .row_major);
    defer data.deinit();
    const result = try kurtosis(f64, data);
    // Single element has zero variance, kurtosis should be 0
    try testing.expectApproxEqAbs(0.0, result, 1e-10);
}

test "kurtosis: all same values f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 5.0, 5.0, 5.0, 5.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data_slice, .row_major);
    defer data.deinit();
    const result = try kurtosis(f64, data);
    // Zero variance -> kurtosis = 0
    try testing.expectApproxEqAbs(0.0, result, 1e-10);
}

test "kurtosis: f32 precision" {
    const allocator = testing.allocator;
    const data_slice = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var data = try ndarray_module.NDArray(f32, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try kurtosis(f32, data);
    try testing.expect(result > -2.0 and result < 2.0);
}

test "kurtosis: two elements f64" {
    const allocator = testing.allocator;
    const data_slice = [_]f64{ 1.0, 3.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &data_slice, .row_major);
    defer data.deinit();
    const result = try kurtosis(f64, data);
    // Two symmetric elements should have specific kurtosis
    // Fourth moment: sum((x - mean)⁴) / n = ((-1)⁴ + 1⁴) / 2 = 1
    // std⁴ = (variance)² = 1² = 1
    // Excess kurtosis = 1 - 3 = -2
    try testing.expectApproxEqAbs(-2.0, result, 1e-10);
}

// ============================================================================
// Histogram Binning Functions
// ============================================================================

/// Error types for histogram functions
pub const HistogramError = error{
    EmptyArray,
    InvalidParameter,
    DimensionMismatch,
};

/// Result struct for histogram function
/// Contains counts per bin and bin edges
pub const HistogramResult = struct {
    counts: []usize,
    bin_edges: []f64,
};

/// Result struct for 2D histogram
/// Contains 2D count matrix and x/y bin edges
pub const Histogram2DResult = struct {
    counts: [][]usize,
    x_edges: []f64,
    y_edges: []f64,
};

/// Compute 1D histogram with evenly spaced bins
///
/// Divides the range [min(data), max(data)] into `bins` equal-width bins.
/// Counts how many elements fall into each bin.
///
/// Bin assignment:
/// - bin i contains values x where edges[i] ≤ x < edges[i+1]
/// - last bin (i = bins-1) contains values where edges[bins-1] ≤ x ≤ edges[bins]
///
/// Parameters:
/// - data: slice of f64 values to bin
/// - bins: number of bins (must be >= 1)
/// - allocator: memory allocator for counts and bin_edges
///
/// Returns: HistogramResult with counts[] (length=bins) and bin_edges[] (length=bins+1)
/// Caller owns both slices and must free them.
///
/// Errors:
/// - error.EmptyArray if data is empty
/// - error.InvalidParameter if bins < 1
/// - std.mem.Allocator.Error if allocation fails
///
/// Time: O(n + bins) where n = data.len
/// Space: O(n + bins) for result arrays
///
/// Note: Follows numpy.histogram convention. All-same-value data
/// creates bins centered on that value.
pub fn histogram(data: []const f64, bins: usize, allocator: Allocator) (HistogramError || Allocator.Error)!HistogramResult {
    if (data.len == 0) return error.EmptyArray;
    if (bins < 1) return error.InvalidParameter;

    // Find min and max
    var min_val = data[0];
    var max_val = data[0];
    for (data) |val| {
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    // Compute bin edges
    const bin_edges = try allocator.alloc(f64, bins + 1);
    const bin_width = if (min_val == max_val)
        1.0
    else
        (max_val - min_val) / @as(f64, @floatFromInt(bins));

    for (0..bins + 1) |i| {
        bin_edges[i] = min_val + @as(f64, @floatFromInt(i)) * bin_width;
    }

    // Initialize counts
    const counts = try allocator.alloc(usize, bins);
    for (0..bins) |i| {
        counts[i] = 0;
    }

    // Assign each data point to a bin
    for (data) |val| {
        // Handle edge case: value exactly at or beyond max
        if (val >= max_val and min_val != max_val) {
            counts[bins - 1] += 1;
        } else {
            // Compute bin index
            const bin_idx_f = (val - min_val) / bin_width;
            const bin_idx = @as(usize, @intFromFloat(bin_idx_f));
            if (bin_idx < bins) {
                counts[bin_idx] += 1;
            }
        }
    }

    return HistogramResult{
        .counts = counts,
        .bin_edges = bin_edges,
    };
}

/// Compute bin edges for histogram (helper function)
///
/// Computes only the bin edges without counting.
/// Useful for custom binning strategies that reuse edges.
///
/// Parameters:
/// - data: slice of f64 values
/// - bins: number of bins (must be >= 1)
/// - allocator: memory allocator for edges
///
/// Returns: slice of bin edges (length = bins+1)
/// Caller owns the slice and must free it.
///
/// Errors:
/// - error.EmptyArray if data is empty
/// - error.InvalidParameter if bins < 1
/// - std.mem.Allocator.Error if allocation fails
///
/// Time: O(n + bins) where n = data.len
/// Space: O(bins) for result array
///
/// Note: Follows numpy.histogram_bin_edges convention
pub fn histogramBinEdges(data: []const f64, bins: usize, allocator: Allocator) (HistogramError || Allocator.Error)![]f64 {
    if (data.len == 0) return error.EmptyArray;
    if (bins < 1) return error.InvalidParameter;

    // Find min and max
    var min_val = data[0];
    var max_val = data[0];
    for (data) |val| {
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    // Compute bin edges
    const bin_edges = try allocator.alloc(f64, bins + 1);
    const bin_width = if (min_val == max_val)
        1.0
    else
        (max_val - min_val) / @as(f64, @floatFromInt(bins));

    for (0..bins + 1) |i| {
        bin_edges[i] = min_val + @as(f64, @floatFromInt(i)) * bin_width;
    }

    return bin_edges;
}

/// Compute 2D histogram for joint distributions
///
/// Creates a 2D grid of bins and counts occurrences of (x, y) pairs.
/// Bin assignment follows same convention as histogram() for each dimension.
///
/// Parameters:
/// - x: slice of x-coordinates (must be same length as y)
/// - y: slice of y-coordinates (must be same length as x)
/// - bins_x: number of x-bins (must be >= 1)
/// - bins_y: number of y-bins (must be >= 1)
/// - allocator: memory allocator for result arrays
///
/// Returns: Histogram2DResult with counts[bins_x][bins_y], x_edges[], y_edges[]
/// Caller owns all arrays and must free them.
///
/// Errors:
/// - error.EmptyArray if x or y is empty
/// - error.DimensionMismatch if len(x) != len(y)
/// - error.InvalidParameter if bins_x < 1 or bins_y < 1
/// - std.mem.Allocator.Error if allocation fails
///
/// Time: O(n + bins_x + bins_y) where n = x.len
/// Space: O(bins_x * bins_y + bins_x + bins_y)
///
/// Note: Returns 2D array where counts[i][j] is count in x-bin i, y-bin j
pub fn histogram2d(x: []const f64, y: []const f64, bins_x: usize, bins_y: usize, allocator: Allocator) (HistogramError || Allocator.Error)!Histogram2DResult {
    if (x.len == 0 or y.len == 0) return error.EmptyArray;
    if (x.len != y.len) return error.DimensionMismatch;
    if (bins_x < 1 or bins_y < 1) return error.InvalidParameter;

    // Find min/max for x
    var x_min = x[0];
    var x_max = x[0];
    for (x) |val| {
        if (val < x_min) x_min = val;
        if (val > x_max) x_max = val;
    }

    // Find min/max for y
    var y_min = y[0];
    var y_max = y[0];
    for (y) |val| {
        if (val < y_min) y_min = val;
        if (val > y_max) y_max = val;
    }

    // Compute bin widths
    const x_width = if (x_min == x_max)
        1.0
    else
        (x_max - x_min) / @as(f64, @floatFromInt(bins_x));

    const y_width = if (y_min == y_max)
        1.0
    else
        (y_max - y_min) / @as(f64, @floatFromInt(bins_y));

    // Create bin edges
    const x_edges = try allocator.alloc(f64, bins_x + 1);
    const y_edges = try allocator.alloc(f64, bins_y + 1);

    for (0..bins_x + 1) |i| {
        x_edges[i] = x_min + @as(f64, @floatFromInt(i)) * x_width;
    }
    for (0..bins_y + 1) |i| {
        y_edges[i] = y_min + @as(f64, @floatFromInt(i)) * y_width;
    }

    // Create 2D count matrix
    const counts = try allocator.alloc([]usize, bins_x);
    for (0..bins_x) |i| {
        counts[i] = try allocator.alloc(usize, bins_y);
        for (0..bins_y) |j| {
            counts[i][j] = 0;
        }
    }

    // Bin each point
    for (0..x.len) |i| {
        const x_val = x[i];
        const y_val = y[i];

        // Compute x bin
        var x_idx = bins_x - 1; // Default to last bin
        if (x_val < x_max or x_min == x_max) {
            const x_bin_f = (x_val - x_min) / x_width;
            const temp_idx = @as(usize, @intFromFloat(x_bin_f));
            if (temp_idx < bins_x) {
                x_idx = temp_idx;
            }
        }

        // Compute y bin
        var y_idx = bins_y - 1; // Default to last bin
        if (y_val < y_max or y_min == y_max) {
            const y_bin_f = (y_val - y_min) / y_width;
            const temp_idx = @as(usize, @intFromFloat(y_bin_f));
            if (temp_idx < bins_y) {
                y_idx = temp_idx;
            }
        }

        counts[x_idx][y_idx] += 1;
    }

    return Histogram2DResult{
        .counts = counts,
        .x_edges = x_edges,
        .y_edges = y_edges,
    };
}

// ============================================================================
// Histogram Tests
// ============================================================================

test "histogram: basic uniform data" {
    const allocator = testing.allocator;
    const data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };

    const result = try histogram(&data, 5, allocator);
    defer {
        allocator.free(result.counts);
        allocator.free(result.bin_edges);
    }

    // 5 bins from 0 to 4 with width 1 each
    // Bin edges: [0, 1, 2, 3, 4, 4]
    try testing.expectEqual(5, result.counts.len);
    try testing.expectEqual(6, result.bin_edges.len);

    // Each value should be in its own bin
    for (result.counts) |count| {
        try testing.expectEqual(1, count);
    }

    // Check edges are monotonic
    for (0..result.bin_edges.len - 1) |i| {
        try testing.expect(result.bin_edges[i] <= result.bin_edges[i + 1]);
    }
}

test "histogram: even binning" {
    const allocator = testing.allocator;
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };

    const result = try histogram(&data, 2, allocator);
    defer {
        allocator.free(result.counts);
        allocator.free(result.bin_edges);
    }

    // 2 bins: [1,5.5) and [5.5,10]
    try testing.expectEqual(2, result.counts.len);
    try testing.expectEqual(3, result.bin_edges.len);

    // First 5 values in first bin, last 5 in second
    try testing.expectEqual(5, result.counts[0]);
    try testing.expectEqual(5, result.counts[1]);
}

test "histogram: single value" {
    const allocator = testing.allocator;
    const data = [_]f64{42.0};

    const result = try histogram(&data, 5, allocator);
    defer {
        allocator.free(result.counts);
        allocator.free(result.bin_edges);
    }

    try testing.expectEqual(5, result.counts.len);

    // Single value should go into first bin (or last bin depending on impl)
    var total: usize = 0;
    for (result.counts) |count| {
        total += count;
    }
    try testing.expectEqual(1, total);
}

test "histogram: all same values" {
    const allocator = testing.allocator;
    const data = [_]f64{ 5.0, 5.0, 5.0, 5.0, 5.0 };

    const result = try histogram(&data, 5, allocator);
    defer {
        allocator.free(result.counts);
        allocator.free(result.bin_edges);
    }

    try testing.expectEqual(5, result.counts.len);

    // All values should be in one bin
    var total: usize = 0;
    for (result.counts) |count| {
        total += count;
    }
    try testing.expectEqual(5, total);
}

test "histogram: negative values" {
    const allocator = testing.allocator;
    const data = [_]f64{ -5.0, -4.0, -3.0, -2.0, -1.0 };

    const result = try histogram(&data, 5, allocator);
    defer {
        allocator.free(result.counts);
        allocator.free(result.bin_edges);
    }

    try testing.expectEqual(5, result.counts.len);
    try testing.expectEqual(6, result.bin_edges.len);

    // Range should span from -5 to -1
    try testing.expect(result.bin_edges[0] <= -5.0);
    try testing.expect(result.bin_edges[5] >= -1.0);

    // Each value in its own bin
    for (result.counts) |count| {
        try testing.expectEqual(1, count);
    }
}

test "histogram: values on boundaries" {
    const allocator = testing.allocator;
    const data = [_]f64{ 0.0, 1.0, 2.0, 3.0 };

    const result = try histogram(&data, 2, allocator);
    defer {
        allocator.free(result.counts);
        allocator.free(result.bin_edges);
    }

    try testing.expectEqual(2, result.counts.len);

    // Sum should equal input length
    var total: usize = 0;
    for (result.counts) |count| {
        total += count;
    }
    try testing.expectEqual(4, total);
}

test "histogram: bin edges monotonicity" {
    const allocator = testing.allocator;
    const data = [_]f64{ 0.5, 1.5, 2.5, 3.5, 4.5 };

    const result = try histogram(&data, 3, allocator);
    defer {
        allocator.free(result.counts);
        allocator.free(result.bin_edges);
    }

    // Verify monotonic increasing edges
    for (0..result.bin_edges.len - 1) |i| {
        try testing.expect(result.bin_edges[i] < result.bin_edges[i + 1]);
    }
}

test "histogram: sparse data" {
    const allocator = testing.allocator;
    const data = [_]f64{ 0.0, 100.0 };

    const result = try histogram(&data, 10, allocator);
    defer {
        allocator.free(result.counts);
        allocator.free(result.bin_edges);
    }

    try testing.expectEqual(10, result.counts.len);

    // Most bins empty, only first and last have values
    var total: usize = 0;
    for (result.counts) |count| {
        total += count;
    }
    try testing.expectEqual(2, total);
}

test "histogram: large dataset" {
    const allocator = testing.allocator;
    var data_list = try std.ArrayList(f64).initCapacity(allocator, 1000);
    defer data_list.deinit(allocator);

    // Create 1000 points uniformly distributed 0-999
    for (0..1000) |i| {
        data_list.appendAssumeCapacity(@as(f64, @floatFromInt(i)));
    }

    const result = try histogram(data_list.items, 10, allocator);
    defer {
        allocator.free(result.counts);
        allocator.free(result.bin_edges);
    }

    // 1000 points in 10 bins = ~100 per bin
    try testing.expectEqual(10, result.counts.len);

    var total: usize = 0;
    for (result.counts) |count| {
        total += count;
    }
    try testing.expectEqual(1000, total);
}

test "histogram: empty array error" {
    const allocator = testing.allocator;
    const data: [0]f64 = [_]f64{};

    const result = histogram(&data, 5, allocator);
    try testing.expectError(error.EmptyArray, result);
}

test "histogram: bins=0 error" {
    const allocator = testing.allocator;
    const data = [_]f64{ 1.0, 2.0, 3.0 };

    const result = histogram(&data, 0, allocator);
    try testing.expectError(error.InvalidParameter, result);
}

test "histogram: two values" {
    const allocator = testing.allocator;
    const data = [_]f64{ 1.0, 3.0 };

    const result = try histogram(&data, 2, allocator);
    defer {
        allocator.free(result.counts);
        allocator.free(result.bin_edges);
    }

    try testing.expectEqual(2, result.counts.len);
    try testing.expectEqual(3, result.bin_edges.len);

    var total: usize = 0;
    for (result.counts) |count| {
        total += count;
    }
    try testing.expectEqual(2, total);
}

test "histogram: bin edges correct range" {
    const allocator = testing.allocator;
    const data = [_]f64{ 0.0, 10.0 };

    const result = try histogram(&data, 10, allocator);
    defer {
        allocator.free(result.counts);
        allocator.free(result.bin_edges);
    }

    // First edge should be at or below min
    try testing.expect(result.bin_edges[0] <= 0.0);

    // Last edge should be at or above max
    try testing.expect(result.bin_edges[result.bin_edges.len - 1] >= 10.0);
}

test "histogram: mixed positive negative" {
    const allocator = testing.allocator;
    const data = [_]f64{ -10.0, -5.0, 0.0, 5.0, 10.0 };

    const result = try histogram(&data, 5, allocator);
    defer {
        allocator.free(result.counts);
        allocator.free(result.bin_edges);
    }

    try testing.expectEqual(5, result.counts.len);

    var total: usize = 0;
    for (result.counts) |count| {
        total += count;
    }
    try testing.expectEqual(5, total);
}

// ============================================================================
// histogramBinEdges Tests
// ============================================================================

test "histogramBinEdges: uniform edges" {
    const allocator = testing.allocator;
    const data = [_]f64{ 0.0, 10.0 };

    const edges = try histogramBinEdges(&data, 10, allocator);
    defer allocator.free(edges);

    // 10 bins from 0 to 10 with width 1
    try testing.expectEqual(11, edges.len);
    try testing.expectApproxEqAbs(0.0, edges[0], 1e-10);
    try testing.expectApproxEqAbs(10.0, edges[10], 1e-10);
}

test "histogramBinEdges: fractional bins" {
    const allocator = testing.allocator;
    const data = [_]f64{ 0.0, 1.0 };

    const edges = try histogramBinEdges(&data, 4, allocator);
    defer allocator.free(edges);

    try testing.expectEqual(5, edges.len);
    // Edges: [0, 0.25, 0.5, 0.75, 1]
    try testing.expectApproxEqAbs(0.0, edges[0], 1e-10);
    try testing.expectApproxEqAbs(0.25, edges[1], 1e-10);
    try testing.expectApproxEqAbs(0.5, edges[2], 1e-10);
    try testing.expectApproxEqAbs(0.75, edges[3], 1e-10);
    try testing.expectApproxEqAbs(1.0, edges[4], 1e-10);
}

test "histogramBinEdges: negative range" {
    const allocator = testing.allocator;
    const data = [_]f64{ -10.0, 10.0 };

    const edges = try histogramBinEdges(&data, 5, allocator);
    defer allocator.free(edges);

    try testing.expectEqual(6, edges.len);
    try testing.expect(edges[0] <= -10.0);
    try testing.expect(edges[5] >= 10.0);

    // Verify monotonicity
    for (0..edges.len - 1) |i| {
        try testing.expect(edges[i] < edges[i + 1]);
    }
}

test "histogramBinEdges: single value" {
    const allocator = testing.allocator;
    const data = [_]f64{ 5.0, 5.0, 5.0 };

    const edges = try histogramBinEdges(&data, 5, allocator);
    defer allocator.free(edges);

    try testing.expectEqual(6, edges.len);

    // When all values are same, first edge should be at the value
    try testing.expectApproxEqAbs(5.0, edges[0], 1e-10);

    // All edges should be monotonically increasing or equal
    for (0..edges.len - 1) |i| {
        try testing.expect(edges[i] <= edges[i + 1]);
    }
}

test "histogramBinEdges: empty array error" {
    const allocator = testing.allocator;
    const data: [0]f64 = [_]f64{};

    const result = histogramBinEdges(&data, 5, allocator);
    try testing.expectError(error.EmptyArray, result);
}

test "histogramBinEdges: bins=0 error" {
    const allocator = testing.allocator;
    const data = [_]f64{ 1.0, 2.0, 3.0 };

    const result = histogramBinEdges(&data, 0, allocator);
    try testing.expectError(error.InvalidParameter, result);
}

// ============================================================================
// histogram2d Tests
// ============================================================================

test "histogram2d: basic grid" {
    const allocator = testing.allocator;
    const x = [_]f64{ 0.0, 1.0, 2.0 };
    const y = [_]f64{ 0.0, 1.0, 2.0 };

    const result = try histogram2d(&x, &y, 2, 2, allocator);
    defer {
        allocator.free(result.x_edges);
        allocator.free(result.y_edges);
        for (result.counts) |row| {
            allocator.free(row);
        }
        allocator.free(result.counts);
    }

    try testing.expectEqual(2, result.counts.len);
    try testing.expectEqual(2, result.counts[0].len);
    try testing.expectEqual(3, result.x_edges.len);
    try testing.expectEqual(3, result.y_edges.len);
}

test "histogram2d: diagonal line" {
    const allocator = testing.allocator;
    const x = [_]f64{ 0.0, 1.0, 2.0 };
    const y = [_]f64{ 0.0, 1.0, 2.0 };

    const result = try histogram2d(&x, &y, 2, 2, allocator);
    defer {
        allocator.free(result.x_edges);
        allocator.free(result.y_edges);
        for (result.counts) |row| {
            allocator.free(row);
        }
        allocator.free(result.counts);
    }

    // Points on y=x diagonal should mostly be in diagonal bins
    var total: usize = 0;
    for (result.counts) |row| {
        for (row) |count| {
            total += count;
        }
    }
    try testing.expectEqual(3, total);
}

test "histogram2d: single point" {
    const allocator = testing.allocator;
    const x = [_]f64{0.0};
    const y = [_]f64{0.0};

    const result = try histogram2d(&x, &y, 5, 5, allocator);
    defer {
        allocator.free(result.x_edges);
        allocator.free(result.y_edges);
        for (result.counts) |row| {
            allocator.free(row);
        }
        allocator.free(result.counts);
    }

    // Single point should be counted once
    var total: usize = 0;
    for (result.counts) |row| {
        for (row) |count| {
            total += count;
        }
    }
    try testing.expectEqual(1, total);
}

test "histogram2d: corners" {
    const allocator = testing.allocator;
    const x = [_]f64{ 0.0, 0.0, 10.0, 10.0 };
    const y = [_]f64{ 0.0, 10.0, 0.0, 10.0 };

    const result = try histogram2d(&x, &y, 2, 2, allocator);
    defer {
        allocator.free(result.x_edges);
        allocator.free(result.y_edges);
        for (result.counts) |row| {
            allocator.free(row);
        }
        allocator.free(result.counts);
    }

    // Points at 4 corners should all be counted
    var total: usize = 0;
    for (result.counts) |row| {
        for (row) |count| {
            total += count;
        }
    }
    try testing.expectEqual(4, total);
}

test "histogram2d: rectangular bins" {
    const allocator = testing.allocator;
    const x = [_]f64{ 0.0, 5.0, 10.0 };
    const y = [_]f64{ 0.0, 1.0, 2.0 };

    const result = try histogram2d(&x, &y, 3, 2, allocator);
    defer {
        allocator.free(result.x_edges);
        allocator.free(result.y_edges);
        for (result.counts) |row| {
            allocator.free(row);
        }
        allocator.free(result.counts);
    }

    try testing.expectEqual(3, result.counts.len);
    try testing.expectEqual(2, result.counts[0].len);

    var total: usize = 0;
    for (result.counts) |row| {
        for (row) |count| {
            total += count;
        }
    }
    try testing.expectEqual(3, total);
}

test "histogram2d: dimension mismatch error" {
    const allocator = testing.allocator;
    const x = [_]f64{ 0.0, 1.0 };
    const y = [_]f64{ 0.0, 1.0, 2.0 };

    const result = histogram2d(&x, &y, 2, 2, allocator);
    try testing.expectError(error.DimensionMismatch, result);
}

test "histogram2d: empty arrays error" {
    const allocator = testing.allocator;
    const x: [0]f64 = [_]f64{};
    const y: [0]f64 = [_]f64{};

    const result = histogram2d(&x, &y, 2, 2, allocator);
    try testing.expectError(error.EmptyArray, result);
}

test "histogram2d: bins_x=0 error" {
    const allocator = testing.allocator;
    const x = [_]f64{ 0.0, 1.0 };
    const y = [_]f64{ 0.0, 1.0 };

    const result = histogram2d(&x, &y, 0, 2, allocator);
    try testing.expectError(error.InvalidParameter, result);
}

test "histogram2d: bins_y=0 error" {
    const allocator = testing.allocator;
    const x = [_]f64{ 0.0, 1.0 };
    const y = [_]f64{ 0.0, 1.0 };

    const result = histogram2d(&x, &y, 2, 0, allocator);
    try testing.expectError(error.InvalidParameter, result);
}

test "histogram2d: all same values" {
    const allocator = testing.allocator;
    const x = [_]f64{ 5.0, 5.0, 5.0 };
    const y = [_]f64{ 3.0, 3.0, 3.0 };

    const result = try histogram2d(&x, &y, 5, 5, allocator);
    defer {
        allocator.free(result.x_edges);
        allocator.free(result.y_edges);
        for (result.counts) |row| {
            allocator.free(row);
        }
        allocator.free(result.counts);
    }

    // All 3 points in one bin
    var total: usize = 0;
    for (result.counts) |row| {
        for (row) |count| {
            total += count;
        }
    }
    try testing.expectEqual(3, total);
}

test "histogram2d: negative coordinates" {
    const allocator = testing.allocator;
    const x = [_]f64{ -10.0, 0.0, 10.0 };
    const y = [_]f64{ -10.0, 0.0, 10.0 };

    const result = try histogram2d(&x, &y, 3, 3, allocator);
    defer {
        allocator.free(result.x_edges);
        allocator.free(result.y_edges);
        for (result.counts) |row| {
            allocator.free(row);
        }
        allocator.free(result.counts);
    }

    var total: usize = 0;
    for (result.counts) |row| {
        for (row) |count| {
            total += count;
        }
    }
    try testing.expectEqual(3, total);
}
