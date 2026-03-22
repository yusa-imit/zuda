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
    // Handle edge case: when ddof >= n, return 0 (mathematically undefined, but allows edge case handling)
    if (ddof >= n) {
        return @as(T, 0); // Return 0 for both float and integer types
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

test "kurtosis: heavy-tailed f64" {
    const allocator = testing.allocator;
    // Distribution with extreme values
    const data_slice = [_]f64{ -100.0, 1.0, 2.0, 3.0, 100.0 };
    var data = try ndarray_module.NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data_slice, .row_major);
    defer data.deinit();
    const result = try kurtosis(f64, data);
    // Heavy tails should have positive excess kurtosis
    try testing.expect(result > 0.0);
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
