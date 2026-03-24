//! Window Functions for Signal Processing
//!
//! This module provides common window functions used to reduce spectral leakage
//! in frequency domain analysis. Windows are applied element-wise to time-domain
//! signals before applying FFT.
//!
//! ## Supported Windows
//! - `hamming` — Hamming window (α = 0.54, β = 0.46)
//! - `hann` — Hann (Hanning) window (raised cosine)
//! - `blackman` — Blackman window (better sidelobe suppression)
//! - `bartlett` — Bartlett (triangular) window
//! - `kaiser` — Kaiser window (configurable via beta parameter)
//!
//! ## Time Complexity
//! - All windows: O(n) for n samples
//!
//! ## Space Complexity
//! - All windows: O(n) for output buffer
//!
//! ## Use Cases
//! - Spectral analysis (reduce frequency leakage)
//! - FIR filter design
//! - Audio processing (overlap-add, STFT)

const std = @import("std");
const math = std.math;
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Hamming window: w[n] = α - β·cos(2πn/(N-1))
/// where α = 0.54, β = 0.46
///
/// Properties:
/// - Main lobe width: 8π/N
/// - Side lobe attenuation: -43 dB
/// - Good frequency resolution with moderate leakage reduction
///
/// Time: O(n) | Space: O(n)
///
/// Parameters:
/// - T: Floating-point type (f32 or f64)
/// - n: Window length
/// - allocator: Memory allocator for result
///
/// Returns: Array of window coefficients w[0..n-1]
pub fn hamming(comptime T: type, n: usize, allocator: Allocator) Allocator.Error![]T {
    if (n == 0) return try allocator.alloc(T, 0);
    if (n == 1) {
        const result = try allocator.alloc(T, 1);
        result[0] = 1.0;
        return result;
    }

    const result = try allocator.alloc(T, n);
    errdefer allocator.free(result);

    const alpha: T = 0.54;
    const beta: T = 0.46;
    const n_minus_1: T = @floatFromInt(n - 1);

    for (0..n) |i| {
        const fi: T = @floatFromInt(i);
        const angle = 2.0 * math.pi * fi / n_minus_1;
        result[i] = alpha - beta * @cos(angle);
    }

    return result;
}

/// Hann window: w[n] = 0.5 * (1 - cos(2πn/(N-1)))
/// Also known as Hanning window (raised cosine)
///
/// Properties:
/// - Main lobe width: 8π/N
/// - Side lobe attenuation: -31 dB
/// - Smooth window with zero endpoints
///
/// Time: O(n) | Space: O(n)
///
/// Parameters:
/// - T: Floating-point type (f32 or f64)
/// - n: Window length
/// - allocator: Memory allocator for result
///
/// Returns: Array of window coefficients w[0..n-1]
pub fn hann(comptime T: type, n: usize, allocator: Allocator) Allocator.Error![]T {
    if (n == 0) return try allocator.alloc(T, 0);
    if (n == 1) {
        const result = try allocator.alloc(T, 1);
        result[0] = 1.0;
        return result;
    }

    const result = try allocator.alloc(T, n);
    errdefer allocator.free(result);

    const n_minus_1: T = @floatFromInt(n - 1);

    for (0..n) |i| {
        const fi: T = @floatFromInt(i);
        const angle = 2.0 * math.pi * fi / n_minus_1;
        result[i] = 0.5 * (1.0 - @cos(angle));
    }

    return result;
}

/// Blackman window: w[n] = α₀ - α₁·cos(2πn/(N-1)) + α₂·cos(4πn/(N-1))
/// where α₀ = 0.42, α₁ = 0.5, α₂ = 0.08
///
/// Properties:
/// - Main lobe width: 12π/N
/// - Side lobe attenuation: -58 dB
/// - Better sidelobe suppression than Hamming/Hann
///
/// Time: O(n) | Space: O(n)
///
/// Parameters:
/// - T: Floating-point type (f32 or f64)
/// - n: Window length
/// - allocator: Memory allocator for result
///
/// Returns: Array of window coefficients w[0..n-1]
pub fn blackman(comptime T: type, n: usize, allocator: Allocator) Allocator.Error![]T {
    if (n == 0) return try allocator.alloc(T, 0);
    if (n == 1) {
        const result = try allocator.alloc(T, 1);
        result[0] = 1.0;
        return result;
    }

    const result = try allocator.alloc(T, n);
    errdefer allocator.free(result);

    const a0: T = 0.42;
    const a1: T = 0.5;
    const a2: T = 0.08;
    const n_minus_1: T = @floatFromInt(n - 1);

    for (0..n) |i| {
        const fi: T = @floatFromInt(i);
        const angle1 = 2.0 * math.pi * fi / n_minus_1;
        const angle2 = 4.0 * math.pi * fi / n_minus_1;
        result[i] = a0 - a1 * @cos(angle1) + a2 * @cos(angle2);
    }

    return result;
}

/// Bartlett window: w[n] = 1 - |n - (N-1)/2| / ((N-1)/2)
/// Triangular window with zero endpoints
///
/// Properties:
/// - Main lobe width: 8π/N
/// - Side lobe attenuation: -26 dB
/// - Simple triangular shape
///
/// Time: O(n) | Space: O(n)
///
/// Parameters:
/// - T: Floating-point type (f32 or f64)
/// - n: Window length
/// - allocator: Memory allocator for result
///
/// Returns: Array of window coefficients w[0..n-1]
pub fn bartlett(comptime T: type, n: usize, allocator: Allocator) Allocator.Error![]T {
    if (n == 0) return try allocator.alloc(T, 0);
    if (n == 1) {
        const result = try allocator.alloc(T, 1);
        result[0] = 1.0;
        return result;
    }

    const result = try allocator.alloc(T, n);
    errdefer allocator.free(result);

    const n_minus_1: T = @floatFromInt(n - 1);
    const half_n_minus_1 = n_minus_1 / 2.0;

    for (0..n) |i| {
        const fi: T = @floatFromInt(i);
        result[i] = 1.0 - @abs(fi - half_n_minus_1) / half_n_minus_1;
    }

    return result;
}

/// Kaiser window: w[n] = I₀(β√(1 - ((n - (N-1)/2) / ((N-1)/2))²)) / I₀(β)
/// where I₀ is the modified Bessel function of the first kind
///
/// Properties:
/// - Configurable tradeoff between main lobe width and sidelobe level
/// - β = 0: rectangular window
/// - β = 5: similar to Hamming
/// - β = 8.6: similar to Blackman
/// - Higher β: better sidelobe suppression, wider main lobe
///
/// Time: O(n) | Space: O(n)
///
/// Parameters:
/// - T: Floating-point type (f32 or f64)
/// - n: Window length
/// - beta: Shape parameter (typically 0-20)
/// - allocator: Memory allocator for result
///
/// Returns: Array of window coefficients w[0..n-1]
pub fn kaiser(comptime T: type, n: usize, beta: T, allocator: Allocator) Allocator.Error![]T {
    if (n == 0) return try allocator.alloc(T, 0);
    if (n == 1) {
        const result = try allocator.alloc(T, 1);
        result[0] = 1.0;
        return result;
    }

    const result = try allocator.alloc(T, n);
    errdefer allocator.free(result);

    const bessel_beta = besselI0(T, beta);
    const n_minus_1: T = @floatFromInt(n - 1);
    const half_n_minus_1 = n_minus_1 / 2.0;

    for (0..n) |i| {
        const fi: T = @floatFromInt(i);
        const x = (fi - half_n_minus_1) / half_n_minus_1;
        const arg = beta * @sqrt(1.0 - x * x);
        result[i] = besselI0(T, arg) / bessel_beta;
    }

    return result;
}

/// Modified Bessel function of the first kind, order 0: I₀(x)
/// Uses series expansion for accurate computation
///
/// Time: O(k) where k is number of terms (typically ~20-50)
fn besselI0(comptime T: type, x: T) T {
    if (x == 0.0) return 1.0;

    // Series expansion: I₀(x) = Σ (x²/4)^k / (k!)²
    var sum: T = 1.0;
    var term: T = 1.0;
    const x_squared_over_4 = (x * x) / 4.0;

    var k: usize = 1;
    while (k < 50) : (k += 1) {
        const fk: T = @floatFromInt(k);
        term *= x_squared_over_4 / (fk * fk);
        sum += term;

        // Convergence check
        if (term < sum * 1e-12) break;
    }

    return sum;
}

// ============================================================================
// Tests
// ============================================================================

test "hamming: basic properties" {
    const window = try hamming(f64, 10, testing.allocator);
    defer testing.allocator.free(window);

    // Check length
    try testing.expectEqual(10, window.len);

    // Check symmetry
    try testing.expectApproxEqAbs(window[0], window[9], 1e-10);
    try testing.expectApproxEqAbs(window[1], window[8], 1e-10);
    try testing.expectApproxEqAbs(window[2], window[7], 1e-10);

    // Check peak at center
    try testing.expect(window[4] > window[0]);
    try testing.expect(window[4] > window[1]);
    try testing.expect(window[5] > window[0]);

    // Check endpoints are non-zero (Hamming characteristic)
    try testing.expect(window[0] > 0.0);
    try testing.expect(window[9] > 0.0);
}

test "hamming: edge cases" {
    // Empty window
    const empty = try hamming(f64, 0, testing.allocator);
    defer testing.allocator.free(empty);
    try testing.expectEqual(0, empty.len);

    // Single element
    const single = try hamming(f64, 1, testing.allocator);
    defer testing.allocator.free(single);
    try testing.expectEqual(1, single.len);
    try testing.expectApproxEqAbs(1.0, single[0], 1e-10);
}

test "hamming: coefficients" {
    const window = try hamming(f64, 5, testing.allocator);
    defer testing.allocator.free(window);

    // Known values for n=5 Hamming window
    try testing.expectApproxEqAbs(0.08, window[0], 1e-10);
    try testing.expectApproxEqAbs(0.54, window[1], 1e-10);
    try testing.expectApproxEqAbs(1.0, window[2], 1e-10);
    try testing.expectApproxEqAbs(0.54, window[3], 1e-10);
    try testing.expectApproxEqAbs(0.08, window[4], 1e-10);
}

test "hann: basic properties" {
    const window = try hann(f64, 10, testing.allocator);
    defer testing.allocator.free(window);

    // Check length
    try testing.expectEqual(10, window.len);

    // Check symmetry
    try testing.expectApproxEqAbs(window[0], window[9], 1e-10);
    try testing.expectApproxEqAbs(window[1], window[8], 1e-10);

    // Check endpoints are zero (Hann characteristic)
    try testing.expectApproxEqAbs(0.0, window[0], 1e-10);
    try testing.expectApproxEqAbs(0.0, window[9], 1e-10);

    // Check peak at center
    try testing.expect(window[4] > window[1]);
    try testing.expect(window[5] > window[1]);
}

test "hann: edge cases" {
    // Empty window
    const empty = try hann(f64, 0, testing.allocator);
    defer testing.allocator.free(empty);
    try testing.expectEqual(0, empty.len);

    // Single element
    const single = try hann(f64, 1, testing.allocator);
    defer testing.allocator.free(single);
    try testing.expectEqual(1, single.len);
    try testing.expectApproxEqAbs(1.0, single[0], 1e-10);
}

test "hann: coefficients" {
    const window = try hann(f64, 5, testing.allocator);
    defer testing.allocator.free(window);

    // Known values for n=5 Hann window
    try testing.expectApproxEqAbs(0.0, window[0], 1e-10);
    try testing.expectApproxEqAbs(0.5, window[1], 1e-10);
    try testing.expectApproxEqAbs(1.0, window[2], 1e-10);
    try testing.expectApproxEqAbs(0.5, window[3], 1e-10);
    try testing.expectApproxEqAbs(0.0, window[4], 1e-10);
}

test "blackman: basic properties" {
    const window = try blackman(f64, 10, testing.allocator);
    defer testing.allocator.free(window);

    // Check length
    try testing.expectEqual(10, window.len);

    // Check symmetry
    try testing.expectApproxEqAbs(window[0], window[9], 1e-10);
    try testing.expectApproxEqAbs(window[1], window[8], 1e-10);

    // Check endpoints are near zero
    try testing.expect(window[0] < 0.01);
    try testing.expect(window[9] < 0.01);

    // Check peak at center
    try testing.expect(window[4] > window[0]);
    try testing.expect(window[4] > window[1]);
}

test "blackman: edge cases" {
    // Empty window
    const empty = try blackman(f64, 0, testing.allocator);
    defer testing.allocator.free(empty);
    try testing.expectEqual(0, empty.len);

    // Single element
    const single = try blackman(f64, 1, testing.allocator);
    defer testing.allocator.free(single);
    try testing.expectEqual(1, single.len);
    try testing.expectApproxEqAbs(1.0, single[0], 1e-10);
}

test "bartlett: basic properties" {
    const window = try bartlett(f64, 10, testing.allocator);
    defer testing.allocator.free(window);

    // Check length
    try testing.expectEqual(10, window.len);

    // Check symmetry
    try testing.expectApproxEqAbs(window[0], window[9], 1e-10);
    try testing.expectApproxEqAbs(window[1], window[8], 1e-10);

    // Check endpoints are zero (triangular window)
    try testing.expectApproxEqAbs(0.0, window[0], 1e-10);
    try testing.expectApproxEqAbs(0.0, window[9], 1e-10);

    // Check linearity in first half
    const delta1 = window[2] - window[1];
    const delta2 = window[3] - window[2];
    try testing.expectApproxEqAbs(delta1, delta2, 1e-10);
}

test "bartlett: edge cases" {
    // Empty window
    const empty = try bartlett(f64, 0, testing.allocator);
    defer testing.allocator.free(empty);
    try testing.expectEqual(0, empty.len);

    // Single element
    const single = try bartlett(f64, 1, testing.allocator);
    defer testing.allocator.free(single);
    try testing.expectEqual(1, single.len);
    try testing.expectApproxEqAbs(1.0, single[0], 1e-10);
}

test "bartlett: coefficients" {
    const window = try bartlett(f64, 5, testing.allocator);
    defer testing.allocator.free(window);

    // Known values for n=5 Bartlett window
    try testing.expectApproxEqAbs(0.0, window[0], 1e-10);
    try testing.expectApproxEqAbs(0.5, window[1], 1e-10);
    try testing.expectApproxEqAbs(1.0, window[2], 1e-10);
    try testing.expectApproxEqAbs(0.5, window[3], 1e-10);
    try testing.expectApproxEqAbs(0.0, window[4], 1e-10);
}

test "kaiser: basic properties" {
    const window = try kaiser(f64, 10, 5.0, testing.allocator);
    defer testing.allocator.free(window);

    // Check length
    try testing.expectEqual(10, window.len);

    // Check symmetry
    try testing.expectApproxEqAbs(window[0], window[9], 1e-10);
    try testing.expectApproxEqAbs(window[1], window[8], 1e-10);

    // Check endpoints are non-zero but small
    try testing.expect(window[0] > 0.0);
    try testing.expect(window[0] < 0.5);

    // Check peak at center
    try testing.expect(window[4] > window[0]);
    try testing.expect(window[4] > window[1]);
}

test "kaiser: edge cases" {
    // Empty window
    const empty = try kaiser(f64, 0, 5.0, testing.allocator);
    defer testing.allocator.free(empty);
    try testing.expectEqual(0, empty.len);

    // Single element
    const single = try kaiser(f64, 1, 5.0, testing.allocator);
    defer testing.allocator.free(single);
    try testing.expectEqual(1, single.len);
    try testing.expectApproxEqAbs(1.0, single[0], 1e-10);
}

test "kaiser: beta parameter effects" {
    const window_low = try kaiser(f64, 10, 1.0, testing.allocator);
    defer testing.allocator.free(window_low);

    const window_high = try kaiser(f64, 10, 10.0, testing.allocator);
    defer testing.allocator.free(window_high);

    // Higher beta should have narrower main lobe (lower endpoints)
    try testing.expect(window_high[0] < window_low[0]);
    try testing.expect(window_high[1] < window_low[1]);

    // Both should have peak near 1.0 (within 6% due to discrete sampling)
    try testing.expectApproxEqAbs(1.0, window_low[4], 0.06);
    try testing.expectApproxEqAbs(1.0, window_high[4], 0.06);
}

test "besselI0: known values" {
    try testing.expectApproxEqAbs(1.0, besselI0(f64, 0.0), 1e-10);
    try testing.expectApproxEqAbs(1.2661, besselI0(f64, 1.0), 1e-4);
    try testing.expectApproxEqAbs(2.2796, besselI0(f64, 2.0), 1e-4);
    try testing.expectApproxEqAbs(27.2399, besselI0(f64, 5.0), 1e-3);
}

test "windows: all produce normalized output" {
    const n = 100;

    {
        const window = try hamming(f64, n, testing.allocator);
        defer testing.allocator.free(window);
        // Check max value is close to 1.0 (within 2%)
        var max: f64 = 0.0;
        for (window) |w| max = @max(max, w);
        try testing.expectApproxEqAbs(1.0, max, 0.02);
    }

    {
        const window = try hann(f64, n, testing.allocator);
        defer testing.allocator.free(window);
        var max: f64 = 0.0;
        for (window) |w| max = @max(max, w);
        try testing.expectApproxEqAbs(1.0, max, 0.02);
    }

    {
        const window = try blackman(f64, n, testing.allocator);
        defer testing.allocator.free(window);
        var max: f64 = 0.0;
        for (window) |w| max = @max(max, w);
        try testing.expectApproxEqAbs(1.0, max, 0.02);
    }

    {
        const window = try bartlett(f64, n, testing.allocator);
        defer testing.allocator.free(window);
        var max: f64 = 0.0;
        for (window) |w| max = @max(max, w);
        try testing.expectApproxEqAbs(1.0, max, 0.02);
    }

    {
        const window = try kaiser(f64, n, 5.0, testing.allocator);
        defer testing.allocator.free(window);
        var max: f64 = 0.0;
        for (window) |w| max = @max(max, w);
        try testing.expectApproxEqAbs(1.0, max, 0.02);
    }
}

test "windows: f32 precision" {
    const n = 10;

    {
        const window = try hamming(f32, n, testing.allocator);
        defer testing.allocator.free(window);
        try testing.expectEqual(n, window.len);
        try testing.expectApproxEqAbs(@as(f32, 0.08), window[0], 1e-5);
    }

    {
        const window = try hann(f32, n, testing.allocator);
        defer testing.allocator.free(window);
        try testing.expectEqual(n, window.len);
        try testing.expectApproxEqAbs(@as(f32, 0.0), window[0], 1e-5);
    }

    {
        const window = try blackman(f32, n, testing.allocator);
        defer testing.allocator.free(window);
        try testing.expectEqual(n, window.len);
        try testing.expect(window[0] < 0.01);
    }

    {
        const window = try bartlett(f32, n, testing.allocator);
        defer testing.allocator.free(window);
        try testing.expectEqual(n, window.len);
        try testing.expectApproxEqAbs(@as(f32, 0.0), window[0], 1e-5);
    }

    {
        const window = try kaiser(f32, n, 5.0, testing.allocator);
        defer testing.allocator.free(window);
        try testing.expectEqual(n, window.len);
        try testing.expect(window[0] > 0.0);
    }
}
