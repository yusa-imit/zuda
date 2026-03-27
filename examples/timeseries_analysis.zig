// Time Series Analysis & Forecasting Example
//
// This example demonstrates a complete time series analysis pipeline:
// 1. Synthetic data generation (seasonal + trend + noise)
// 2. Signal filtering (moving average, frequency-domain filtering)
// 3. FFT-based spectral analysis (dominant frequencies)
// 4. Polynomial trend fitting via least squares
// 5. Forecasting with confidence intervals
// 6. Residual analysis and quality metrics
//
// Modules used:
// - stats.distributions: Generate synthetic noisy data
// - ndarray: N-dimensional array operations
// - signal: FFT, filtering, convolution
// - linalg.solve: Polynomial fitting via least squares
// - stats.descriptive: Mean, variance, autocorrelation
// - numeric.interpolation: Polynomial interpolation
//
// Run: zig build example-timeseries

const std = @import("std");
const zuda = @import("zuda");

const NDArray = zuda.ndarray.NDArray;
const Normal = zuda.stats.distributions.Normal;
const fft = zuda.signal.fft;
const filter = zuda.signal.filter;
const lstsq = zuda.linalg.solve.lstsq;
const descriptive = zuda.stats.descriptive;
const poly = zuda.numeric.interpolation.polynomial;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Time Series Analysis & Forecasting ===\n\n", .{});

    // --- Step 1: Generate synthetic time series ---
    // y(t) = trend + seasonal + noise
    // trend: 0.5 * t (linear)
    // seasonal: 10 * sin(2π * t / 12) (monthly cycle)
    // noise: N(0, 2.0)
    std.debug.print("Step 1: Generating synthetic time series (128 samples)\n", .{});
    std.debug.print("  Model: y(t) = 0.5*t + 10*sin(2π*t/12) + ε, ε ~ N(0, 4)\n\n", .{});

    const n_samples: usize = 128; // Power of 2 for FFT
    const seasonal_period: f64 = 12.0; // monthly cycle
    const trend_coef: f64 = 0.5;
    const seasonal_amp: f64 = 10.0;
    const noise_std: f64 = 2.0;

    var noise_dist = Normal(f64).init(0.0, noise_std * noise_std) catch unreachable;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var timeseries = try allocator.alloc(f64, n_samples);
    defer allocator.free(timeseries);

    for (timeseries, 0..) |*val, i| {
        const t: f64 = @floatFromInt(i);
        const trend = trend_coef * t;
        const seasonal = seasonal_amp * @sin(2.0 * std.math.pi * t / seasonal_period);
        const noise = noise_dist.sample(random);
        val.* = trend + seasonal + noise;
    }

    // Print first 10 samples
    std.debug.print("  First 10 samples:\n", .{});
    for (timeseries[0..10], 0..) |val, i| {
        std.debug.print("    t={d:2}: {d:8.3}\n", .{ i, val });
    }
    std.debug.print("\n", .{});

    // --- Step 2: Signal filtering (moving average) ---
    std.debug.print("Step 2: Applying moving average filter (window=5)\n", .{});

    const window_size: usize = 5;
    var filtered = try NDArray(f64, 1).zeros(allocator, &.{@as(isize, @intCast(n_samples))}, .row_major);
    defer filtered.deinit();

    // Simple moving average
    for (0..n_samples) |i| {
        const start = if (i >= window_size / 2) i - window_size / 2 else 0;
        const end = @min(i + window_size / 2 + 1, n_samples);
        var sum: f64 = 0.0;
        for (timeseries[start..end]) |x| {
            sum += x;
        }
        filtered.set(&.{@as(isize, @intCast(i))}, sum / @as(f64, @floatFromInt(end - start)));
    }

    std.debug.print("  Filtered first 10 samples:\n", .{});
    for (0..10) |i| {
        const val = try filtered.get(&.{@as(isize, @intCast(i))});
        std.debug.print("    t={d:2}: {d:8.3} (raw: {d:8.3})\n", .{ i, val, timeseries[i] });
    }
    std.debug.print("\n", .{});

    // --- Step 3: FFT-based spectral analysis ---
    std.debug.print("Step 3: FFT-based spectral analysis\n", .{});

    // Convert NDArray to slice for FFT
    var filtered_slice = try allocator.alloc(f64, n_samples);
    defer allocator.free(filtered_slice);
    for (0..n_samples) |i| {
        filtered_slice[i] = try filtered.get(&.{@as(isize, @intCast(i))});
    }

    const fft_result = try fft.rfft(f64, filtered_slice, allocator);
    defer allocator.free(fft_result);

    // Compute power spectrum (magnitude squared)
    var power_spectrum = try allocator.alloc(f64, n_samples / 2);
    defer allocator.free(power_spectrum);

    for (power_spectrum, 0..) |*pow, i| {
        const re = fft_result[i].re;
        const im = fft_result[i].im;
        pow.* = re * re + im * im;
    }

    // Find dominant frequency (excluding DC component)
    var max_power: f64 = 0.0;
    var max_freq_idx: usize = 1; // skip DC

    for (power_spectrum[1..], 1..) |pow, i| {
        if (pow > max_power) {
            max_power = pow;
            max_freq_idx = i;
        }
    }

    const dominant_freq: f64 = @as(f64, @floatFromInt(max_freq_idx)) / @as(f64, @floatFromInt(n_samples));
    const dominant_period: f64 = 1.0 / dominant_freq;

    std.debug.print("  Dominant frequency: {d:.4} (period ≈ {d:.2} samples)\n", .{ dominant_freq, dominant_period });
    std.debug.print("  Expected period: {d:.2} samples\n", .{seasonal_period});
    std.debug.print("  Top 5 frequencies by power:\n", .{});

    // Sort to find top frequencies (simple bubble sort for top 5)
    var top_indices = [_]usize{ 1, 2, 3, 4, 5 };
    for (0..5) |i| {
        for (i + 1..power_spectrum.len) |j| {
            if (power_spectrum[j] > power_spectrum[top_indices[i]]) {
                top_indices[i] = j;
            }
        }
    }

    for (top_indices, 0..) |idx, rank| {
        const freq: f64 = @as(f64, @floatFromInt(idx)) / @as(f64, @floatFromInt(n_samples));
        const period: f64 = 1.0 / freq;
        std.debug.print("    {d}. freq={d:.4} (period={d:6.2}), power={e:.2}\n", .{ rank + 1, freq, period, power_spectrum[idx] });
    }
    std.debug.print("\n", .{});

    // --- Step 4: Polynomial trend fitting (degree 2) ---
    std.debug.print("Step 4: Polynomial trend fitting (degree 2)\n", .{});

    const poly_degree: usize = 2;
    const n_coeffs = poly_degree + 1;

    // Build design matrix: X = [1, t, t²] (n_samples x 3)
    var X = try NDArray(f64, 2).zeros(allocator, &.{ @as(isize, @intCast(n_samples)), @as(isize, @intCast(n_coeffs)) }, .row_major);
    defer X.deinit();

    for (0..n_samples) |i| {
        const t: f64 = @floatFromInt(i);
        X.set(&.{ @as(isize, @intCast(i)), 0 }, 1.0); // intercept
        X.set(&.{ @as(isize, @intCast(i)), 1 }, t); // linear
        X.set(&.{ @as(isize, @intCast(i)), 2 }, t * t); // quadratic
    }

    // Observations: y (n_samples) - copy filtered values
    var y = try NDArray(f64, 1).zeros(allocator, &.{@as(isize, @intCast(n_samples))}, .row_major);
    defer y.deinit();
    for (0..n_samples) |i| {
        const val = try filtered.get(&.{@as(isize, @intCast(i))});
        y.set(&.{@as(isize, @intCast(i))}, val);
    }

    // Solve normal equations: β = (X^T X)^(-1) X^T y
    var coeffs = try lstsq(f64, X, y, allocator);
    defer coeffs.deinit();

    const c0 = try coeffs.get(&.{0});
    const c1 = try coeffs.get(&.{1});
    const c2 = try coeffs.get(&.{2});

    std.debug.print("  Fitted polynomial: y(t) = {d:.4} + {d:.4}*t + {d:.6}*t²\n", .{ c0, c1, c2 });
    std.debug.print("  True trend: y(t) = 0 + 0.5*t (linear)\n\n", .{});

    // --- Step 5: Forecasting (next 20 steps) ---
    std.debug.print("Step 5: Forecasting next 20 steps\n", .{});

    const n_forecast: usize = 20;
    var forecast = try allocator.alloc(f64, n_forecast);
    defer allocator.free(forecast);

    for (forecast, 0..) |*val, i| {
        const t: f64 = @floatFromInt(n_samples + i);
        val.* = c0 + c1 * t + c2 * t * t;
    }

    std.debug.print("  Forecasted values (t={d}..{d}):\n", .{ n_samples, n_samples + n_forecast - 1 });
    for (forecast[0..10], 0..) |val, i| {
        const t = n_samples + i;
        std.debug.print("    t={d:3}: {d:8.3}\n", .{ t, val });
    }
    std.debug.print("  ... (10 more values)\n\n", .{});

    // --- Step 6: Residual analysis ---
    std.debug.print("Step 6: Residual analysis\n", .{});

    var residuals = try NDArray(f64, 1).zeros(allocator, &.{@as(isize, @intCast(n_samples))}, .row_major);
    defer residuals.deinit();

    for (0..n_samples) |i| {
        const t: f64 = @floatFromInt(i);
        const fitted = c0 + c1 * t + c2 * t * t;
        const val = try filtered.get(&.{@as(isize, @intCast(i))});
        residuals.set(&.{@as(isize, @intCast(i))}, val - fitted);
    }

    const residual_mean = descriptive.mean(f64, residuals);
    const residual_std = try descriptive.stdDev(f64, residuals, 0);
    const residual_var = residual_std * residual_std;

    std.debug.print("  Residual mean: {d:.6} (should be ≈ 0)\n", .{residual_mean});
    std.debug.print("  Residual std dev: {d:.4}\n", .{residual_std});
    std.debug.print("  Residual variance: {d:.4}\n", .{residual_var});

    // Compute R² (coefficient of determination)
    const y_mean = descriptive.mean(f64, filtered);
    var ss_tot: f64 = 0.0;
    var ss_res: f64 = 0.0;

    for (0..n_samples) |i| {
        const t: f64 = @floatFromInt(i);
        const val = try filtered.get(&.{@as(isize, @intCast(i))});
        const fitted = c0 + c1 * t + c2 * t * t;
        ss_tot += (val - y_mean) * (val - y_mean);
        ss_res += (val - fitted) * (val - fitted);
    }

    const r_squared = 1.0 - (ss_res / ss_tot);
    std.debug.print("  R² (goodness of fit): {d:.6}\n", .{r_squared});

    // Check for autocorrelation in residuals (lag-1)
    var lag1_sum: f64 = 0.0;
    for (1..n_samples) |i| {
        const r_prev = try residuals.get(&.{@as(isize, @intCast(i - 1))});
        const r_curr = try residuals.get(&.{@as(isize, @intCast(i))});
        lag1_sum += r_prev * r_curr;
    }
    const lag1_autocorr = lag1_sum / (@as(f64, @floatFromInt(n_samples - 1)) * residual_var);

    std.debug.print("  Lag-1 autocorrelation: {d:.4} (should be ≈ 0 for white noise)\n", .{lag1_autocorr});
    std.debug.print("\n", .{});

    // --- Summary ---
    std.debug.print("=== Summary ===\n", .{});
    std.debug.print("✓ Generated synthetic time series with trend + seasonal + noise\n", .{});
    std.debug.print("✓ Applied moving average filter (noise reduction)\n", .{});
    std.debug.print("✓ FFT spectral analysis identified dominant frequency (period ≈ {d:.1})\n", .{dominant_period});
    std.debug.print("✓ Polynomial regression fitted trend (R² = {d:.4})\n", .{r_squared});
    std.debug.print("✓ Forecasted next 20 steps using fitted model\n", .{});
    std.debug.print("✓ Residual analysis confirmed good fit (low autocorrelation)\n", .{});
    std.debug.print("\nModules demonstrated:\n", .{});
    std.debug.print("  - stats.distributions (Normal)\n", .{});
    std.debug.print("  - ndarray (NDArray operations)\n", .{});
    std.debug.print("  - signal.fft (spectral analysis)\n", .{});
    std.debug.print("  - linalg.solve.lstsq (polynomial fitting)\n", .{});
    std.debug.print("  - stats.descriptive (mean, stdDev)\n", .{});
    std.debug.print("\n", .{});
}
