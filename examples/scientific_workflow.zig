//! Scientific Computing Workflow Example
//!
//! This example demonstrates a realistic scientific computing workflow using zuda:
//! 1. Generate synthetic dataset (using stats distributions)
//! 2. Perform linear regression (using linalg solvers)
//! 3. Analyze residuals (using stats descriptive)
//! 4. Compute FFT of residuals to detect periodicities (using signal processing)
//!
//! This showcases cross-module integration: stats, linalg, ndarray, signal

const std = @import("std");
const zuda = @import("zuda");

// Module aliases for readability
const NDArray = zuda.ndarray.NDArray;
const linalg = zuda.linalg;
const stats = zuda.stats;
const signal = zuda.signal;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Scientific Computing Workflow Example ===\n\n", .{});

    // Step 1: Generate synthetic dataset
    // Model: y = 2.5x + 1.3 + noise
    std.debug.print("Step 1: Generating synthetic data (y = 2.5x + 1.3 + noise)\n", .{});

    const n: usize = 128; // Power of 2 for FFT
    const x_data = try allocator.alloc(f64, n);
    defer allocator.free(x_data);

    const y_data = try allocator.alloc(f64, n);
    defer allocator.free(y_data);

    // Generate x values: 0.0 to 10.0
    for (x_data, 0..) |*x, i| {
        x.* = @as(f64, @floatFromInt(i)) / 10.0;
    }

    // Generate y values with Gaussian noise
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const normal = try stats.distributions.Normal(f64).init(0.0, 0.5);

    for (y_data, 0..) |*y, i| {
        const x = x_data[i];
        const true_y = 2.5 * x + 1.3;
        const noise = normal.sample(rng);
        y.* = true_y + noise;
    }

    std.debug.print("  Generated {} data points\n", .{n});
    std.debug.print("  First 5: x=[{d:.2}, {d:.2}, {d:.2}, {d:.2}, {d:.2}] y=[{d:.2}, {d:.2}, {d:.2}, {d:.2}, {d:.2}]\n\n",
        .{x_data[0], x_data[1], x_data[2], x_data[3], x_data[4],
          y_data[0], y_data[1], y_data[2], y_data[3], y_data[4]});

    // Step 2: Perform linear regression using least squares
    std.debug.print("Step 2: Performing linear regression (least squares fit)\n", .{});

    // Build design matrix: [1, x] for each row (intercept + slope)
    var A = try NDArray(f64, 2).zeros(allocator, &.{n, 2}, .row_major);
    defer A.deinit();

    for (0..n) |i| {
        A.set(&.{@intCast(i), 0}, 1.0); // intercept column
        A.set(&.{@intCast(i), 1}, x_data[i]); // slope column
    }

    // Target vector
    var b = try NDArray(f64, 1).fromSlice(allocator, &.{n}, y_data, .row_major);
    defer b.deinit();

    // Solve least squares: min ||Ax - b||₂
    const solve_module = zuda.linalg.solve;
    var result = try solve_module.lstsq(f64, A, b, allocator);
    defer result.deinit();

    const intercept = try result.get(&.{0});
    const slope = try result.get(&.{1});

    std.debug.print("  Fitted model: y = {d:.3}x + {d:.3}\n", .{slope, intercept});
    std.debug.print("  True model:   y = 2.500x + 1.300\n", .{});
    std.debug.print("  Error: slope={d:.3}, intercept={d:.3}\n\n",
        .{@abs(slope - 2.5), @abs(intercept - 1.3)});

    // Step 3: Analyze residuals
    std.debug.print("Step 3: Analyzing residuals\n", .{});

    const residuals_data = try allocator.alloc(f64, n);
    defer allocator.free(residuals_data);

    for (residuals_data, 0..) |*r, i| {
        const predicted = intercept + slope * x_data[i];
        r.* = y_data[i] - predicted;
    }

    // Convert to NDArray for stats functions
    var residuals = try NDArray(f64, 1).fromSlice(allocator, &.{n}, residuals_data, .row_major);
    defer residuals.deinit();

    const descriptive = zuda.stats.descriptive;
    const mean_residual = descriptive.mean(f64, residuals);
    const std_residual = try descriptive.stdDev(f64, residuals, 0);

    std.debug.print("  Mean residual: {d:.6} (should be ~0)\n", .{mean_residual});
    std.debug.print("  Std residual:  {d:.3} (should match noise std=0.5)\n\n", .{std_residual});

    // Step 4: FFT analysis of residuals (detect periodicities)
    std.debug.print("Step 4: FFT analysis of residuals (detect hidden periodicities)\n", .{});

    const fft_module = zuda.signal.fft;
    const fft_result = try fft_module.rfft(f64, residuals_data, allocator);
    defer allocator.free(fft_result);

    // Compute magnitude spectrum
    std.debug.print("  FFT computed: {} frequency bins\n", .{fft_result.len});
    std.debug.print("  First 5 magnitudes: ", .{});
    for (0..@min(5, fft_result.len)) |i| {
        const magnitude = @sqrt(fft_result[i].re * fft_result[i].re +
                               fft_result[i].im * fft_result[i].im);
        std.debug.print("{d:.2} ", .{magnitude});
    }
    std.debug.print("\n\n", .{});

    // Summary
    std.debug.print("=== Workflow Complete ===\n", .{});
    std.debug.print("Successfully demonstrated:\n", .{});
    std.debug.print("  ✓ stats.distributions (Normal) for data generation\n", .{});
    std.debug.print("  ✓ ndarray (NDArray) for matrix operations\n", .{});
    std.debug.print("  ✓ linalg.solve (lstsq) for linear regression\n", .{});
    std.debug.print("  ✓ stats.descriptive (mean, stdDev) for residual analysis\n", .{});
    std.debug.print("  ✓ signal.fft (rfft) for frequency analysis\n", .{});
}
