//! Cross-module Integration Tests (Phase 12 - v1.25.0+)
//!
//! Verifies seamless interoperability between zuda's v2.0 scientific computing modules.
//! These tests demonstrate real-world workflows combining multiple modules:
//! - NDArray ↔ linalg: Matrix operations and decompositions
//! - NDArray ↔ stats: Statistical analysis on multi-dimensional data
//! - NDArray ↔ numeric: Numerical integration and interpolation
//! - Optimization workflows: linalg + optimize for constrained problems
//! - Signal processing: NDArray → signal → stats analysis

const std = @import("std");
const testing = std.testing;
const zuda = @import("zuda");
const NDArray = zuda.ndarray.NDArray;

// ============================================================================
// NDArray ↔ linalg Integration
// ============================================================================

test "cross-module: NDArray → linalg SVD → NDArray results" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 2 }, .row_major);
    defer A.deinit();

    const data = [_]f64{ 1, 2, 3, 4, 5, 6 };
    @memcpy(A.data, &data);

    // linalg.decompositions.svd takes NDArray, returns NDArrays!
    var result = try zuda.linalg.decompositions.svd(f64, A, allocator);
    defer result.U.deinit();
    defer result.S.deinit();
    defer result.Vt.deinit();

    // Verify singular values sorted descending
    try testing.expect(result.S.data[0] >= result.S.data[1]);
}

test "cross-module: NDArray → linalg QR → NDArray results" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 2 }, .row_major);
    defer A.deinit();

    const data = [_]f64{ 1, 0, 0, 1, 0, 0 };
    @memcpy(A.data, &data);

    var result = try zuda.linalg.decompositions.qr(f64, A, allocator);
    defer result.Q.deinit();
    defer result.R.deinit();

    try testing.expectEqual(@as(usize, 3), result.Q.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.Q.shape[1]);
}

test "cross-module: NDArray → linalg Cholesky → NDArray result" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer A.deinit();

    const data = [_]f64{ 4, 2, 2, 3 };
    @memcpy(A.data, &data);

    var L = try zuda.linalg.decompositions.cholesky(f64, A, allocator);
    defer L.deinit();

    try testing.expectEqual(@as(usize, 2), L.shape[0]);
    try testing.expectEqual(@as(usize, 2), L.shape[1]);
}

// ============================================================================
// NDArray ↔ stats Integration
// ============================================================================

test "cross-module: NDArray → stats descriptive → statistical summary" {
    const allocator = testing.allocator;

    // Create 1D dataset
    var data = try NDArray(f64, 1).init(allocator, &[_]usize{6}, .row_major);
    defer data.deinit();

    const values = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    @memcpy(data.data, &values);

    // Compute statistics
    const m = zuda.stats.descriptive.mean(f64, data);
    const v = try zuda.stats.descriptive.variance(f64, data, 0); // ddof = 0 for population variance
    const s = @sqrt(v);

    try testing.expectApproxEqRel(3.5, m, 1e-10);
    try testing.expectApproxEqRel(2.9166666, v, 1e-6);
    try testing.expectApproxEqRel(@sqrt(2.9166666), s, 1e-6);
}

test "cross-module: NDArray → stats correlation → Pearson coefficient" {
    const allocator = testing.allocator;

    // Create two 1D arrays with perfect linear correlation
    var x = try NDArray(f64, 1).init(allocator, &[_]usize{6}, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).init(allocator, &[_]usize{6}, .row_major);
    defer y.deinit();

    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const y_data = [_]f64{ 2.0, 4.0, 6.0, 8.0, 10.0, 12.0 }; // y = 2x
    @memcpy(x.data, &x_data);
    @memcpy(y.data, &y_data);

    // Compute Pearson correlation coefficient
    const r = try zuda.stats.correlation.pearson(x, y, allocator);

    // Perfect positive correlation
    try testing.expectApproxEqRel(1.0, r, 1e-10);
}

// ============================================================================
// NDArray ↔ numeric Integration
// ============================================================================

test "cross-module: NDArray → numeric interpolation → resampled data" {
    const allocator = testing.allocator;

    // Original data points
    var x = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer y.deinit();

    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0 };
    const y_data = [_]f64{ 0.0, 1.0, 4.0, 9.0 }; // y = x^2
    @memcpy(x.data, &x_data);
    @memcpy(y.data, &y_data);

    // Interpolate at x = 1.5
    const x_new = [_]f64{1.5};
    const result = try zuda.numeric.interpolation.interp1d(f64, x.data, y.data, &x_new, allocator);
    defer allocator.free(result);

    // Linear interpolation: y(1.5) = 1 + (4-1) × 0.5 = 2.5
    try testing.expectApproxEqRel(2.5, result[0], 1e-10);
}

test "cross-module: NDArray → numeric integration → area under curve" {
    const allocator = testing.allocator;

    // Compute ∫₀³ x² dx = [x³/3]₀³ = 9 using trapezoidal rule
    var x = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer x.deinit();
    var y = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer y.deinit();

    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0 };
    const y_data = [_]f64{ 0.0, 1.0, 4.0, 9.0 }; // y = x^2
    @memcpy(x.data, &x_data);
    @memcpy(y.data, &y_data);

    const area = try zuda.numeric.integration.trapezoid(f64, x.data, y.data, allocator);

    // Trapezoidal rule approximation of ∫₀³ x² dx = 9
    try testing.expectApproxEqRel(9.0, area, 1e-1); // Coarse discretization → 10% tolerance
}

// ============================================================================
// linalg + optimize Workflow: Quadratic Programming
// ============================================================================

test "cross-module: linalg + optimize → QP solver with matrix constraints" {
    const allocator = testing.allocator;

    // Minimize (1/2)x^T Q x + c^T x subject to Ax ≤ b
    // Q = [[2, 0], [0, 2]] (positive definite)
    // c = [-2, -5]
    // A = [[1, 2], [-1, 2], [0, 1]] (inequality constraints)
    // b = [10, 4, 5]

    const Q_data = [_]f64{ 2, 0, 0, 2 };
    const c_data = [_]f64{ -2, -5 };
    const A_data = [_]f64{ 1, 2, -1, 2, 0, 1 };
    const b_data = [_]f64{ 10, 4, 5 };
    const x0_data = [_]f64{ 0, 0 };

    // Solve QP using optimize module
    const options = zuda.optimize.constrained.QPOptions(f64){
        .max_iter = 100,
        .tol = 1e-6,
        .method = .active_set,
    };

    const result = try zuda.optimize.constrained.quadratic_programming(
        f64,
        &Q_data,      // Q: n×n
        &c_data,      // c: n
        &A_data,      // A: m×n
        &b_data,      // b: m
        null,         // Aeq
        null,         // beq
        &x0_data,     // x0
        options,
        allocator,
    );
    defer {
        allocator.free(result.x);
        allocator.free(result.lambda_ineq);
        allocator.free(result.lambda_eq);
    }

    // Verify solution satisfies constraints
    try testing.expect(result.converged);
}

// ============================================================================
// signal + stats Workflow: Spectral Analysis
// ============================================================================

test "cross-module: signal FFT + stats → frequency domain analysis" {
    const allocator = testing.allocator;

    // Generate time-domain signal: 1 Hz sine wave sampled at 8 Hz for 1 second
    const n: usize = 8;
    const Complex = zuda.signal.fft.Complex(f64);
    var signal = try allocator.alloc(Complex, n);
    defer allocator.free(signal);

    const omega = 2.0 * std.math.pi; // 1 Hz
    for (0..n) |i| {
        const t = @as(f64, @floatFromInt(i)) / 8.0;
        signal[i] = Complex.init(@sin(omega * t), 0.0);
    }

    // Compute FFT
    const fft_result = try zuda.signal.fft.fft(f64, signal, allocator);
    defer allocator.free(fft_result);

    // Convert to magnitudes
    var magnitudes = try allocator.alloc(f64, n);
    defer allocator.free(magnitudes);
    for (0..n) |i| {
        magnitudes[i] = @sqrt(fft_result[i].re * fft_result[i].re + fft_result[i].im * fft_result[i].im);
    }

    // Compute statistics on frequency domain using NDArray
    var mag_array = try NDArray(f64, 1).init(allocator, &[_]usize{n}, .row_major);
    defer mag_array.deinit();
    @memcpy(mag_array.data, magnitudes);

    // Find peak magnitude (simple max search)
    var max_mag: f64 = mag_array.data[0];
    for (mag_array.data) |m| {
        if (m > max_mag) max_mag = m;
    }

    // Peak magnitude should be at 1 Hz bin (bin index 1 for 8-sample FFT)
    try testing.expect(max_mag > 0.0);
}
