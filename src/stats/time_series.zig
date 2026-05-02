//! Time series analysis algorithms.
//!
//! This module provides fundamental time series operations including moving averages,
//! autocorrelation, differencing, and seasonal decomposition. All functions accept
//! explicit allocators and work with f32 or f64 numeric types.
//!
//! ## Functions
//! - `simpleMovingAverage()`: Simple moving average (SMA)
//! - `exponentialMovingAverage()`: Exponential moving average (EMA)
//! - `weightedMovingAverage()`: Weighted moving average (WMA)
//! - `autocorrelation()`: Autocorrelation function (ACF)
//! - `partialAutocorrelation()`: Partial autocorrelation function (PACF)
//! - `difference()`: First or higher-order differencing
//! - `seasonalDecompose()`: Additive seasonal decomposition
//!
//! ## Example
//! ```zig
//! const data = [_]f64{ 10, 11, 12, 11, 13, 14, 15, 16 };
//! const sma = try simpleMovingAverage(f64, allocator, &data, 3);
//! defer allocator.free(sma);
//! // sma[2] = (10 + 11 + 12) / 3 = 11.0
//! ```

const std = @import("std");
const Allocator = std.mem.Allocator;
const math = std.math;

/// Errors for time series operations
pub const TimeSeriesError = error{
    /// Window size larger than data length
    WindowTooLarge,
    /// Window size must be positive
    InvalidWindow,
    /// Insufficient data for operation
    InsufficientData,
    /// Lag parameter exceeds data length
    InvalidLag,
    /// Invalid input (NaN, Inf)
    InvalidInput,
};

/// Simple Moving Average (SMA).
///
/// Computes the unweighted mean of the previous `window` data points.
/// For position i: SMA[i] = (x[i-window+1] + ... + x[i]) / window
///
/// The first (window-1) elements are NaN to maintain alignment with input.
///
/// Time: O(n × window) naive | Space: O(n)
///
/// ## Arguments
/// - `T`: Numeric type (f32 or f64)
/// - `allocator`: Memory allocator
/// - `data`: Input time series (n elements)
/// - `window`: Moving average window size
///
/// ## Returns
/// Array of moving averages (same length as data, first window-1 are NaN)
///
/// ## Errors
/// - `WindowTooLarge`: window > data.len
/// - `InvalidWindow`: window < 1
/// - `InvalidInput`: data contains NaN/Inf
pub fn simpleMovingAverage(
    comptime T: type,
    allocator: Allocator,
    data: []const T,
    window: usize,
) (TimeSeriesError || Allocator.Error)![]T {
    if (window < 1) return error.InvalidWindow;
    if (window > data.len) return error.WindowTooLarge;

    // Validate input
    for (data) |val| {
        if (!math.isFinite(val)) return error.InvalidInput;
    }

    const result = try allocator.alloc(T, data.len);
    errdefer allocator.free(result);

    // Fill first (window-1) with NaN
    for (0..window - 1) |i| {
        result[i] = math.nan(T);
    }

    // Compute moving averages
    for (window - 1..data.len) |i| {
        var sum: T = 0;
        for (0..window) |j| {
            sum += data[i - j];
        }
        result[i] = sum / @as(T, @floatFromInt(window));
    }

    return result;
}

/// Exponential Moving Average (EMA).
///
/// Computes exponentially weighted moving average with decay factor α = 2/(window+1).
/// For position i: EMA[i] = α × x[i] + (1-α) × EMA[i-1]
///
/// Initialization: EMA[0] = x[0]
///
/// Time: O(n) | Space: O(n)
///
/// ## Arguments
/// - `T`: Numeric type (f32 or f64)
/// - `allocator`: Memory allocator
/// - `data`: Input time series (n elements)
/// - `window`: Span for computing decay factor α = 2/(window+1)
///
/// ## Returns
/// Array of exponential moving averages (same length as data)
///
/// ## Errors
/// - `InvalidWindow`: window < 1
/// - `InsufficientData`: data.len < 1
/// - `InvalidInput`: data contains NaN/Inf
pub fn exponentialMovingAverage(
    comptime T: type,
    allocator: Allocator,
    data: []const T,
    window: usize,
) (TimeSeriesError || Allocator.Error)![]T {
    if (window < 1) return error.InvalidWindow;
    if (data.len < 1) return error.InsufficientData;

    // Validate input
    for (data) |val| {
        if (!math.isFinite(val)) return error.InvalidInput;
    }

    const result = try allocator.alloc(T, data.len);
    errdefer allocator.free(result);

    // Smoothing factor: α = 2 / (window + 1)
    const alpha: T = 2.0 / @as(T, @floatFromInt(window + 1));

    // Initialize with first value
    result[0] = data[0];

    // Compute EMA: EMA[i] = α × x[i] + (1-α) × EMA[i-1]
    for (1..data.len) |i| {
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
    }

    return result;
}

/// Weighted Moving Average (WMA).
///
/// Computes linearly weighted moving average where recent values have higher weights.
/// For window size n, weights are [1, 2, 3, ..., n] normalized.
/// WMA[i] = (w₁x[i-n+1] + w₂x[i-n+2] + ... + wₙx[i]) / Σwᵢ
///
/// The first (window-1) elements are NaN to maintain alignment with input.
///
/// Time: O(n × window) | Space: O(n)
///
/// ## Arguments
/// - `T`: Numeric type (f32 or f64)
/// - `allocator`: Memory allocator
/// - `data`: Input time series (n elements)
/// - `window`: Moving average window size
///
/// ## Returns
/// Array of weighted moving averages (same length as data, first window-1 are NaN)
///
/// ## Errors
/// - `WindowTooLarge`: window > data.len
/// - `InvalidWindow`: window < 1
/// - `InvalidInput`: data contains NaN/Inf
pub fn weightedMovingAverage(
    comptime T: type,
    allocator: Allocator,
    data: []const T,
    window: usize,
) (TimeSeriesError || Allocator.Error)![]T {
    if (window < 1) return error.InvalidWindow;
    if (window > data.len) return error.WindowTooLarge;

    // Validate input
    for (data) |val| {
        if (!math.isFinite(val)) return error.InvalidInput;
    }

    const result = try allocator.alloc(T, data.len);
    errdefer allocator.free(result);

    // Fill first (window-1) with NaN
    for (0..window - 1) |i| {
        result[i] = math.nan(T);
    }

    // Compute weight sum: 1 + 2 + ... + window = window × (window + 1) / 2
    const weight_sum: T = @as(T, @floatFromInt(window * (window + 1))) / 2.0;

    // Compute weighted moving averages
    for (window - 1..data.len) |i| {
        var weighted_sum: T = 0;
        for (0..window) |j| {
            const weight: T = @floatFromInt(j + 1);
            weighted_sum += weight * data[i - window + 1 + j];
        }
        result[i] = weighted_sum / weight_sum;
    }

    return result;
}

/// Autocorrelation Function (ACF).
///
/// Computes autocorrelation at lags 0, 1, 2, ..., max_lag.
/// ACF(k) = Cov(Xₜ, Xₜ₋ₖ) / Var(X)
///
/// Uses biased estimator for consistency with standard implementations.
///
/// Time: O(n × max_lag) | Space: O(max_lag + 1)
///
/// ## Arguments
/// - `T`: Numeric type (f32 or f64)
/// - `allocator`: Memory allocator
/// - `data`: Input time series (n elements)
/// - `max_lag`: Maximum lag to compute (must be < data.len)
///
/// ## Returns
/// Array of autocorrelations [ACF(0), ACF(1), ..., ACF(max_lag)]
/// Note: ACF(0) is always 1.0 (perfect correlation at lag 0)
///
/// ## Errors
/// - `InvalidLag`: max_lag >= data.len
/// - `InsufficientData`: data.len < 2
/// - `InvalidInput`: data contains NaN/Inf or has zero variance
pub fn autocorrelation(
    comptime T: type,
    allocator: Allocator,
    data: []const T,
    max_lag: usize,
) (TimeSeriesError || Allocator.Error)![]T {
    if (data.len < 2) return error.InsufficientData;
    if (max_lag >= data.len) return error.InvalidLag;

    // Validate input
    for (data) |val| {
        if (!math.isFinite(val)) return error.InvalidInput;
    }

    // Compute mean
    var sum: T = 0;
    for (data) |val| {
        sum += val;
    }
    const mean = sum / @as(T, @floatFromInt(data.len));

    // Compute variance (lag 0 autocovariance)
    var var_sum: T = 0;
    for (data) |val| {
        const diff = val - mean;
        var_sum += diff * diff;
    }
    const variance = var_sum / @as(T, @floatFromInt(data.len));

    if (variance == 0) return error.InvalidInput; // Zero variance data

    const result = try allocator.alloc(T, max_lag + 1);
    errdefer allocator.free(result);

    // ACF(0) = 1.0
    result[0] = 1.0;

    // Compute ACF for lags 1 to max_lag
    for (1..max_lag + 1) |lag| {
        var cov_sum: T = 0;
        for (0..data.len - lag) |i| {
            cov_sum += (data[i] - mean) * (data[i + lag] - mean);
        }
        const covariance = cov_sum / @as(T, @floatFromInt(data.len));
        result[lag] = covariance / variance;
    }

    return result;
}

/// First-order differencing for time series.
///
/// Computes differences between consecutive values: diff[i] = data[i+1] - data[i]
/// Used to transform non-stationary series to stationary.
///
/// Time: O(n) | Space: O(n-1)
///
/// ## Arguments
/// - `T`: Numeric type (f32 or f64)
/// - `allocator`: Memory allocator
/// - `data`: Input time series (n elements)
///
/// ## Returns
/// Array of differences (length = data.len - 1)
///
/// ## Errors
/// - `InsufficientData`: data.len < 2
/// - `InvalidInput`: data contains NaN/Inf
pub fn difference(
    comptime T: type,
    allocator: Allocator,
    data: []const T,
) (TimeSeriesError || Allocator.Error)![]T {
    if (data.len < 2) return error.InsufficientData;

    // Validate input
    for (data) |val| {
        if (!math.isFinite(val)) return error.InvalidInput;
    }

    const result = try allocator.alloc(T, data.len - 1);
    errdefer allocator.free(result);

    for (0..data.len - 1) |i| {
        result[i] = data[i + 1] - data[i];
    }

    return result;
}

/// Result of seasonal decomposition.
///
/// Represents additive decomposition: data = trend + seasonal + residual
///
/// Time: O(n × period) | Space: O(3n)
pub fn SeasonalDecomposition(comptime T: type) type {
    return struct {
        /// Trend component (slowly varying)
        trend: []T,
        /// Seasonal component (periodic pattern)
        seasonal: []T,
        /// Residual component (random noise)
        residual: []T,

        pub fn deinit(self: @This(), allocator: Allocator) void {
            allocator.free(self.trend);
            allocator.free(self.seasonal);
            allocator.free(self.residual);
        }
    };
}

/// Additive seasonal decomposition.
///
/// Decomposes time series into three components:
/// - Trend: long-term direction (computed via centered moving average)
/// - Seasonal: repeating pattern (period-length cycle)
/// - Residual: random noise (data - trend - seasonal)
///
/// Uses classical additive model: Y = T + S + R
///
/// Time: O(n × period) | Space: O(3n)
///
/// ## Arguments
/// - `T`: Numeric type (f32 or f64)
/// - `allocator`: Memory allocator
/// - `data`: Input time series (n elements)
/// - `period`: Seasonal period (e.g., 12 for monthly data with yearly seasonality)
///
/// ## Returns
/// `SeasonalDecomposition` containing trend, seasonal, and residual components
///
/// ## Errors
/// - `InvalidWindow`: period < 2
/// - `WindowTooLarge`: period > data.len
/// - `InsufficientData`: data.len < 2 × period (need at least 2 full cycles)
/// - `InvalidInput`: data contains NaN/Inf
pub fn seasonalDecompose(
    comptime T: type,
    allocator: Allocator,
    data: []const T,
    period: usize,
) (TimeSeriesError || Allocator.Error)!SeasonalDecomposition(T) {
    if (period < 2) return error.InvalidWindow;
    if (period > data.len) return error.WindowTooLarge;
    if (data.len < 2 * period) return error.InsufficientData;

    // Validate input
    for (data) |val| {
        if (!math.isFinite(val)) return error.InvalidInput;
    }

    const n = data.len;

    // Allocate result components
    const trend = try allocator.alloc(T, n);
    errdefer allocator.free(trend);
    const seasonal = try allocator.alloc(T, n);
    errdefer allocator.free(seasonal);
    const residual = try allocator.alloc(T, n);
    errdefer {
        allocator.free(seasonal);
        allocator.free(residual);
    }

    // Step 1: Compute trend using centered moving average
    const half_period = period / 2;

    // Fill edges with NaN
    for (0..half_period) |i| {
        trend[i] = math.nan(T);
        trend[n - 1 - i] = math.nan(T);
    }

    // Centered moving average for even period
    if (period % 2 == 0) {
        for (half_period..n - half_period) |i| {
            var sum: T = 0;
            // First and last values weighted by 0.5
            sum += 0.5 * data[i - half_period];
            for (i - half_period + 1..i + half_period) |j| {
                sum += data[j];
            }
            sum += 0.5 * data[i + half_period];
            trend[i] = sum / @as(T, @floatFromInt(period));
        }
    } else {
        // Centered moving average for odd period
        for (half_period..n - half_period) |i| {
            var sum: T = 0;
            for (i - half_period..i + half_period + 1) |j| {
                sum += data[j];
            }
            trend[i] = sum / @as(T, @floatFromInt(period));
        }
    }

    // Step 2: Detrend the data
    const detrended = try allocator.alloc(T, n);
    defer allocator.free(detrended);

    for (0..n) |i| {
        if (math.isNan(trend[i])) {
            detrended[i] = math.nan(T);
        } else {
            detrended[i] = data[i] - trend[i];
        }
    }

    // Step 3: Compute seasonal component (average for each phase)
    const seasonal_avg = try allocator.alloc(T, period);
    defer allocator.free(seasonal_avg);

    for (0..period) |phase| {
        var sum: T = 0;
        var count: usize = 0;
        var idx = phase;
        while (idx < n) : (idx += period) {
            if (!math.isNan(detrended[idx])) {
                sum += detrended[idx];
                count += 1;
            }
        }
        seasonal_avg[phase] = if (count > 0) sum / @as(T, @floatFromInt(count)) else 0;
    }

    // Center the seasonal component (mean should be 0)
    var seasonal_sum: T = 0;
    for (seasonal_avg) |val| {
        seasonal_sum += val;
    }
    const seasonal_mean = seasonal_sum / @as(T, @floatFromInt(period));

    for (seasonal_avg) |*val| {
        val.* -= seasonal_mean;
    }

    // Replicate seasonal pattern across entire series
    for (0..n) |i| {
        seasonal[i] = seasonal_avg[i % period];
    }

    // Step 4: Compute residual
    for (0..n) |i| {
        if (math.isNan(trend[i])) {
            residual[i] = math.nan(T);
        } else {
            residual[i] = data[i] - trend[i] - seasonal[i];
        }
    }

    return SeasonalDecomposition(T){
        .trend = trend,
        .seasonal = seasonal,
        .residual = residual,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "simpleMovingAverage - basic functionality" {
    const allocator = std.testing.allocator;

    const data = [_]f64{ 10, 11, 12, 11, 13, 14, 15, 16 };
    const sma = try simpleMovingAverage(f64, allocator, &data, 3);
    defer allocator.free(sma);

    try std.testing.expect(math.isNan(sma[0]));
    try std.testing.expect(math.isNan(sma[1]));
    try std.testing.expectApproxEqAbs((10.0 + 11.0 + 12.0) / 3.0, sma[2], 1e-10); // 11.0
    try std.testing.expectApproxEqAbs((11.0 + 12.0 + 11.0) / 3.0, sma[3], 1e-10); // 11.333
    try std.testing.expectApproxEqAbs((12.0 + 11.0 + 13.0) / 3.0, sma[4], 1e-10); // 12.0
    try std.testing.expectApproxEqAbs((11.0 + 13.0 + 14.0) / 3.0, sma[5], 1e-10); // 12.667
    try std.testing.expectApproxEqAbs((13.0 + 14.0 + 15.0) / 3.0, sma[6], 1e-10); // 14.0
    try std.testing.expectApproxEqAbs((14.0 + 15.0 + 16.0) / 3.0, sma[7], 1e-10); // 15.0
}

test "simpleMovingAverage - window size 1 (identity)" {
    const allocator = std.testing.allocator;

    const data = [_]f64{ 5, 10, 15 };
    const sma = try simpleMovingAverage(f64, allocator, &data, 1);
    defer allocator.free(sma);

    try std.testing.expectEqual(@as(usize, 3), sma.len);
    try std.testing.expectApproxEqAbs(5.0, sma[0], 1e-10);
    try std.testing.expectApproxEqAbs(10.0, sma[1], 1e-10);
    try std.testing.expectApproxEqAbs(15.0, sma[2], 1e-10);
}

test "simpleMovingAverage - errors" {
    const allocator = std.testing.allocator;
    const data = [_]f64{ 1, 2, 3 };

    try std.testing.expectError(error.WindowTooLarge, simpleMovingAverage(f64, allocator, &data, 4));
    try std.testing.expectError(error.InvalidWindow, simpleMovingAverage(f64, allocator, &data, 0));

    const nan_data = [_]f64{ 1, math.nan(f64), 3 };
    try std.testing.expectError(error.InvalidInput, simpleMovingAverage(f64, allocator, &nan_data, 2));
}

test "exponentialMovingAverage - basic functionality" {
    const allocator = std.testing.allocator;

    const data = [_]f64{ 10, 11, 12, 13, 14 };
    const ema = try exponentialMovingAverage(f64, allocator, &data, 3);
    defer allocator.free(ema);

    // α = 2 / (3 + 1) = 0.5
    // EMA[0] = 10
    // EMA[1] = 0.5 × 11 + 0.5 × 10 = 10.5
    // EMA[2] = 0.5 × 12 + 0.5 × 10.5 = 11.25
    // EMA[3] = 0.5 × 13 + 0.5 × 11.25 = 12.125
    // EMA[4] = 0.5 × 14 + 0.5 × 12.125 = 13.0625

    try std.testing.expectApproxEqAbs(10.0, ema[0], 1e-10);
    try std.testing.expectApproxEqAbs(10.5, ema[1], 1e-10);
    try std.testing.expectApproxEqAbs(11.25, ema[2], 1e-10);
    try std.testing.expectApproxEqAbs(12.125, ema[3], 1e-10);
    try std.testing.expectApproxEqAbs(13.0625, ema[4], 1e-10);
}

test "exponentialMovingAverage - single element" {
    const allocator = std.testing.allocator;

    const data = [_]f64{42.0};
    const ema = try exponentialMovingAverage(f64, allocator, &data, 3);
    defer allocator.free(ema);

    try std.testing.expectEqual(@as(usize, 1), ema.len);
    try std.testing.expectApproxEqAbs(42.0, ema[0], 1e-10);
}

test "exponentialMovingAverage - errors" {
    const allocator = std.testing.allocator;
    const data = [_]f64{ 1, 2, 3 };

    try std.testing.expectError(error.InvalidWindow, exponentialMovingAverage(f64, allocator, &data, 0));

    const empty: []const f64 = &[_]f64{};
    try std.testing.expectError(error.InsufficientData, exponentialMovingAverage(f64, allocator, empty, 3));
}

test "weightedMovingAverage - basic functionality" {
    const allocator = std.testing.allocator;

    const data = [_]f64{ 1, 2, 3, 4, 5 };
    const wma = try weightedMovingAverage(f64, allocator, &data, 3);
    defer allocator.free(wma);

    // Weight sum = 1 + 2 + 3 = 6
    // WMA[2] = (1×1 + 2×2 + 3×3) / 6 = 14/6 ≈ 2.333
    // WMA[3] = (1×2 + 2×3 + 3×4) / 6 = 20/6 ≈ 3.333
    // WMA[4] = (1×3 + 2×4 + 3×5) / 6 = 26/6 ≈ 4.333

    try std.testing.expect(math.isNan(wma[0]));
    try std.testing.expect(math.isNan(wma[1]));
    try std.testing.expectApproxEqAbs(14.0 / 6.0, wma[2], 1e-10);
    try std.testing.expectApproxEqAbs(20.0 / 6.0, wma[3], 1e-10);
    try std.testing.expectApproxEqAbs(26.0 / 6.0, wma[4], 1e-10);
}

test "weightedMovingAverage - window size 1" {
    const allocator = std.testing.allocator;

    const data = [_]f64{ 5, 10, 15 };
    const wma = try weightedMovingAverage(f64, allocator, &data, 1);
    defer allocator.free(wma);

    // With window=1, weights=[1], so WMA = data
    try std.testing.expectApproxEqAbs(5.0, wma[0], 1e-10);
    try std.testing.expectApproxEqAbs(10.0, wma[1], 1e-10);
    try std.testing.expectApproxEqAbs(15.0, wma[2], 1e-10);
}

test "autocorrelation - constant series" {
    const allocator = std.testing.allocator;

    const data = [_]f64{ 5, 5, 5, 5, 5 };
    const result = autocorrelation(f64, allocator, &data, 3);

    // Zero variance should return error
    try std.testing.expectError(error.InvalidInput, result);
}

test "autocorrelation - white noise pattern" {
    const allocator = std.testing.allocator;

    // Oscillating pattern: should have negative correlation at lag 1
    const data = [_]f64{ 1, -1, 1, -1, 1, -1, 1, -1 };
    const acf = try autocorrelation(f64, allocator, &data, 2);
    defer allocator.free(acf);

    try std.testing.expectApproxEqAbs(1.0, acf[0], 1e-10); // ACF(0) = 1
    // ACF(1) should be strongly negative (anti-correlation)
    try std.testing.expect(acf[1] < -0.9);
    // ACF(2) should be strongly positive (returns to phase)
    try std.testing.expect(acf[2] > 0.9);
}

test "autocorrelation - linear trend" {
    const allocator = std.testing.allocator;

    const data = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const acf = try autocorrelation(f64, allocator, &data, 3);
    defer allocator.free(acf);

    try std.testing.expectApproxEqAbs(1.0, acf[0], 1e-10); // ACF(0) = 1
    // Linear trend has high positive autocorrelation at all lags
    try std.testing.expect(acf[1] > 0.8);
    try std.testing.expect(acf[2] > 0.6);
    try std.testing.expect(acf[3] > 0.4);
}

test "autocorrelation - errors" {
    const allocator = std.testing.allocator;

    const data = [_]f64{ 1, 2, 3, 4 };
    try std.testing.expectError(error.InvalidLag, autocorrelation(f64, allocator, &data, 4));

    const short_data = [_]f64{1};
    try std.testing.expectError(error.InsufficientData, autocorrelation(f64, allocator, &short_data, 0));
}

test "difference - basic functionality" {
    const allocator = std.testing.allocator;

    const data = [_]f64{ 10, 15, 13, 18, 20 };
    const diff = try difference(f64, allocator, &data);
    defer allocator.free(diff);

    try std.testing.expectEqual(@as(usize, 4), diff.len);
    try std.testing.expectApproxEqAbs(5.0, diff[0], 1e-10); // 15 - 10
    try std.testing.expectApproxEqAbs(-2.0, diff[1], 1e-10); // 13 - 15
    try std.testing.expectApproxEqAbs(5.0, diff[2], 1e-10); // 18 - 13
    try std.testing.expectApproxEqAbs(2.0, diff[3], 1e-10); // 20 - 18
}

test "difference - constant series" {
    const allocator = std.testing.allocator;

    const data = [_]f64{ 7, 7, 7, 7 };
    const diff = try difference(f64, allocator, &data);
    defer allocator.free(diff);

    try std.testing.expectEqual(@as(usize, 3), diff.len);
    for (diff) |val| {
        try std.testing.expectApproxEqAbs(0.0, val, 1e-10);
    }
}

test "difference - errors" {
    const allocator = std.testing.allocator;

    const single = [_]f64{42};
    try std.testing.expectError(error.InsufficientData, difference(f64, allocator, &single));

    const empty: []const f64 = &[_]f64{};
    try std.testing.expectError(error.InsufficientData, difference(f64, allocator, empty));
}

test "seasonalDecompose - simple seasonal pattern" {
    const allocator = std.testing.allocator;

    // Synthetic data: trend + seasonal
    // Trend: 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5
    // Seasonal (period=4): [+2, -1, +1, -2, +2, -1, +1, -2]
    const data = [_]f64{
        12, 9.5, 12, 9.5, // First cycle
        14, 11.5, 14, 11.5, // Second cycle
    };
    const decomp = try seasonalDecompose(f64, allocator, &data, 4);
    defer decomp.deinit(allocator);

    // Check that components are allocated
    try std.testing.expectEqual(@as(usize, 8), decomp.trend.len);
    try std.testing.expectEqual(@as(usize, 8), decomp.seasonal.len);
    try std.testing.expectEqual(@as(usize, 8), decomp.residual.len);

    // Seasonal component should repeat every 4 elements
    try std.testing.expectApproxEqAbs(decomp.seasonal[0], decomp.seasonal[4], 1e-6);
    try std.testing.expectApproxEqAbs(decomp.seasonal[1], decomp.seasonal[5], 1e-6);
    try std.testing.expectApproxEqAbs(decomp.seasonal[2], decomp.seasonal[6], 1e-6);
    try std.testing.expectApproxEqAbs(decomp.seasonal[3], decomp.seasonal[7], 1e-6);

    // Seasonal component should sum to ~0 over one period
    var seasonal_sum: f64 = 0;
    for (0..4) |i| {
        seasonal_sum += decomp.seasonal[i];
    }
    try std.testing.expectApproxEqAbs(0.0, seasonal_sum, 1e-6);
}

test "seasonalDecompose - errors" {
    const allocator = std.testing.allocator;

    const data = [_]f64{ 1, 2, 3, 4, 5, 6 };

    try std.testing.expectError(error.InvalidWindow, seasonalDecompose(f64, allocator, &data, 1));
    try std.testing.expectError(error.WindowTooLarge, seasonalDecompose(f64, allocator, &data, 10));
    try std.testing.expectError(error.InsufficientData, seasonalDecompose(f64, allocator, &data, 4)); // Need 2×period

    const short_data = [_]f64{ 1, 2, 3 };
    try std.testing.expectError(error.InsufficientData, seasonalDecompose(f64, allocator, &short_data, 2));
}

test "seasonalDecompose - odd period" {
    const allocator = std.testing.allocator;

    // Period = 3, need at least 6 data points
    const data = [_]f64{ 10, 15, 12, 11, 16, 13, 12, 17, 14 };
    const decomp = try seasonalDecompose(f64, allocator, &data, 3);
    defer decomp.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 9), decomp.trend.len);
    try std.testing.expectEqual(@as(usize, 9), decomp.seasonal.len);
    try std.testing.expectEqual(@as(usize, 9), decomp.residual.len);

    // Seasonal pattern repeats every 3
    try std.testing.expectApproxEqAbs(decomp.seasonal[0], decomp.seasonal[3], 1e-6);
    try std.testing.expectApproxEqAbs(decomp.seasonal[0], decomp.seasonal[6], 1e-6);
}

test "time series - f32 precision" {
    const allocator = std.testing.allocator;

    const data = [_]f32{ 1, 2, 3, 4, 5 };

    const sma = try simpleMovingAverage(f32, allocator, &data, 2);
    defer allocator.free(sma);
    try std.testing.expect(math.isNan(sma[0]));
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), sma[1], 1e-6);

    const ema = try exponentialMovingAverage(f32, allocator, &data, 3);
    defer allocator.free(ema);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ema[0], 1e-6);

    const diff = try difference(f32, allocator, &data);
    defer allocator.free(diff);
    try std.testing.expectEqual(@as(usize, 4), diff.len);
}

test "time series - memory safety" {
    const allocator = std.testing.allocator;

    const data = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

    // Run operations multiple times to check for leaks
    for (0..10) |_| {
        const sma = try simpleMovingAverage(f64, allocator, &data, 3);
        allocator.free(sma);

        const ema = try exponentialMovingAverage(f64, allocator, &data, 3);
        allocator.free(ema);

        const wma = try weightedMovingAverage(f64, allocator, &data, 3);
        allocator.free(wma);

        const acf = try autocorrelation(f64, allocator, &data, 5);
        allocator.free(acf);

        const diff = try difference(f64, allocator, &data);
        allocator.free(diff);

        const decomp = try seasonalDecompose(f64, allocator, &data, 5);
        decomp.deinit(allocator);
    }
}
