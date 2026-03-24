//! Convolution and Cross-Correlation — Time and Frequency Domain
//!
//! This module provides implementations of convolution and cross-correlation
//! in both the time domain (direct computation) and frequency domain (FFT-based fast).
//!
//! ## Supported Operations
//! - `convolve` — Linear convolution via direct method
//! - `correlate` — Cross-correlation via direct method
//! - `fftconvolve` — Linear convolution via FFT (for large signals)
//!
//! ## Time Complexity
//! - convolve: O((a.len + b.len)²) for direct convolution
//! - correlate: O((a.len + b.len)²) for direct correlation
//! - fftconvolve: O(n log n) where n = a.len + b.len - 1 (rounded to next power of 2)
//!
//! ## Space Complexity
//! - convolve: O(a.len + b.len - 1) for output + O(1) auxiliary
//! - correlate: O(a.len + b.len - 1) for output + O(1) auxiliary
//! - fftconvolve: O(n) for FFT computation where n = next power of 2 >= output length
//!
//! ## Mathematical Definitions
//!
//! **Linear Convolution**:
//! ```
//! y[n] = sum_{k=0}^{∞} a[k] * b[n-k]
//! Output length: a.len + b.len - 1
//! ```
//!
//! **Cross-Correlation**:
//! ```
//! r[n] = sum_{k=0}^{∞} a[k] * b[k+n]
//! Output length: a.len + b.len - 1
//! Note: correlate(a, b) = convolve(a, reverse(b))
//! ```
//!
//! ## Properties
//! - Commutativity: convolve(a, b) = convolve(b, a)
//! - Associativity: convolve(convolve(a, b), c) = convolve(a, convolve(b, c))
//! - Impulse property: convolve(x, [1, 0, 0, ...]) = x (zero-padded)
//! - FFT equivalence: fftconvolve(a, b) ≈ convolve(a, b) (within floating-point precision)
//!
//! ## Use Cases
//! - Discrete Linear Time-Invariant (LTI) system simulation
//! - Filter application: convolve(signal, impulse_response)
//! - Pattern matching and signal detection via correlation
//! - Polynomial multiplication via convolution
//! - Image filtering via 2D convolution
//!
//! ## References
//! - Oppenheim, A. V., Schafer, R. W., & Buck, J. R. (2010). "Discrete-Time Signal Processing"
//! - Smith, S. W. (1997). "The Scientist and Engineer's Guide to Digital Signal Processing"

const std = @import("std");
const math = std.math;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const fft_mod = @import("fft.zig");

/// Compute the linear convolution of two real-valued signals via direct method
///
/// Performs time-domain convolution of sequences a and b.
/// Equivalent to polynomial multiplication.
///
/// Parameters:
/// - T: Floating-point type (f32 or f64)
/// - allocator: Memory allocator for output
/// - a: First input sequence (caller owns, can be freed after call)
/// - b: Second input sequence (caller owns, can be freed after call)
///
/// Returns: Linear convolution result with length a.len + b.len - 1
///          (caller owns, must call allocator.free)
///
/// Errors:
/// - error.OutOfMemory if allocation fails
///
/// Time: O((a.len + b.len)²)
/// Space: O(a.len + b.len)
///
/// Properties:
/// - Result is commutative: convolve(a, b) = convolve(b, a)
/// - Result[k] = sum_{i=0}^{k} a[i] * b[k-i] (zero-padded)
/// - Impulse response: convolve(x, [1, 0, ...]) returns x with zeros appended
///
/// Example:
/// ```zig
/// const allocator = std.testing.allocator;
/// const a = [_]f64{ 1, 2, 3 };
/// const b = [_]f64{ 1, 1 };
/// const result = try convolve(f64, allocator, a[0..], b[0..]);
/// defer allocator.free(result);
/// // result ≈ [1, 3, 5, 3]
/// ```
pub fn convolve(comptime T: type, allocator: Allocator, a: []const T, b: []const T) Allocator.Error![]T {
    const output_len = if (a.len == 0 or b.len == 0) 0 else a.len + b.len - 1;
    const output = try allocator.alloc(T, output_len);
    errdefer allocator.free(output);

    if (output_len == 0) return output;

    // Initialize output to zero
    @memset(output, 0);

    // Direct convolution: y[n] = sum_{k=0}^{n} a[k] * b[n-k]
    for (0..a.len) |i| {
        for (0..b.len) |j| {
            output[i + j] += a[i] * b[j];
        }
    }

    return output;
}

/// Compute the cross-correlation of two real-valued signals via direct method
///
/// Cross-correlation measures the similarity between two sequences as a function of lag.
/// Definition: correlate(a, b)[n] = sum_{k=0}^{∞} a[k] * b[k+n]
///
/// Parameters:
/// - T: Floating-point type (f32 or f64)
/// - allocator: Memory allocator for output
/// - a: First input sequence (reference signal)
/// - b: Second input sequence (comparison signal)
///
/// Returns: Cross-correlation result with length a.len + b.len - 1
///          (caller owns, must call allocator.free)
///
/// Errors:
/// - error.OutOfMemory if allocation fails
///
/// Time: O((a.len + b.len)²)
/// Space: O(a.len + b.len)
///
/// Properties:
/// - Autocorrelation: correlate(x, x)[0] is the signal energy
/// - Not commutative: correlate(a, b) ≠ correlate(b, a) in general
/// - Relationship: correlate(a, b) = convolve(a, reverse(b))
/// - Peak location indicates delay of maximum similarity
///
/// Example:
/// ```zig
/// const allocator = std.testing.allocator;
/// const signal = [_]f64{ 1, 2, 3, 4 };
/// const corr = try correlate(f64, allocator, signal[0..], signal[0..]);
/// defer allocator.free(corr);
/// // corr[0] is autocorrelation peak at zero lag
/// ```
pub fn correlate(comptime T: type, allocator: Allocator, a: []const T, b: []const T) Allocator.Error![]T {
    const output_len = if (a.len == 0 or b.len == 0) 0 else a.len + b.len - 1;
    const output = try allocator.alloc(T, output_len);
    errdefer allocator.free(output);

    if (output_len == 0) return output;

    // Initialize output to zero
    @memset(output, 0);

    // Cross-correlation: r[n] = sum_{k=0}^{∞} a[k] * b[k+n]
    // For lag n, we sum where a[k] is defined and b[k+n] is defined
    for (0..output_len) |n| {
        var sum: T = 0.0;
        for (0..a.len) |k| {
            const b_idx = k + n;
            if (b_idx < b.len) {
                sum += a[k] * b[b_idx];
            }
        }
        output[n] = sum;
    }

    return output;
}

/// Compute linear convolution via FFT (fast method for large signals)
///
/// Uses the FFT to perform convolution in the frequency domain.
/// Theoretically: convolve(a, b) = ifft(fft(a) .* fft(b))
/// Inputs are zero-padded to the next power of 2 >= a.len + b.len - 1.
///
/// Parameters:
/// - T: Floating-point type (f32 or f64)
/// - allocator: Memory allocator for intermediate and output
/// - a: First input sequence
/// - b: Second input sequence
///
/// Returns: Linear convolution result with length a.len + b.len - 1
///          (caller owns, must call allocator.free)
///
/// Errors:
/// - error.OutOfMemory if allocation fails
/// - error.InvalidLength (propagated from FFT if internal constraint violated)
///
/// Time: O(n log n) where n = next power of 2 >= a.len + b.len - 1
/// Space: O(n) for FFT buffers
///
/// Properties:
/// - Output should match convolve() within floating-point precision
/// - More efficient than direct convolution for large signals (typically > 1000 samples)
/// - Faster FFT requires input length to be power of 2 (automatic padding)
/// - Numerical precision: results differ from direct method due to floating-point rounding
///
/// Example:
/// ```zig
/// const allocator = std.testing.allocator;
/// const a = [_]f64{ 1, 2, 3 };
/// const b = [_]f64{ 1, 1 };
/// const result = try fftconvolve(f64, allocator, a[0..], b[0..]);
/// defer allocator.free(result);
/// // result ≈ [1, 3, 5, 3] (same as direct convolve)
/// ```
pub fn fftconvolve(comptime T: type, allocator: Allocator, a: []const T, b: []const T) (Allocator.Error || error{InvalidLength})![]T {
    const output_len = if (a.len == 0 or b.len == 0) 0 else a.len + b.len - 1;

    if (output_len == 0) {
        return try allocator.alloc(T, 0);
    }

    // Find next power of 2 >= output_len
    var fft_len: usize = 1;
    while (fft_len < output_len) {
        fft_len *= 2;
    }

    // Allocate complex buffers for FFT
    const Complex = fft_mod.Complex(T);
    const a_fft = try allocator.alloc(Complex, fft_len);
    defer allocator.free(a_fft);
    const b_fft = try allocator.alloc(Complex, fft_len);
    defer allocator.free(b_fft);

    // Pad a to fft_len with zeros and convert to complex
    @memset(a_fft, Complex.init(0, 0));
    for (0..a.len) |i| {
        a_fft[i] = Complex.init(a[i], 0);
    }

    // Pad b to fft_len with zeros and convert to complex
    @memset(b_fft, Complex.init(0, 0));
    for (0..b.len) |i| {
        b_fft[i] = Complex.init(b[i], 0);
    }

    // Compute FFTs
    const a_spectrum = try fft_mod.fft(T, a_fft, allocator);
    defer allocator.free(a_spectrum);
    const b_spectrum = try fft_mod.fft(T, b_fft, allocator);
    defer allocator.free(b_spectrum);

    // Pointwise multiplication in frequency domain
    const product = try allocator.alloc(Complex, fft_len);
    defer allocator.free(product);
    for (0..fft_len) |i| {
        product[i] = a_spectrum[i].mul(b_spectrum[i]);
    }

    // Inverse FFT
    const time_domain = try fft_mod.ifft(T, product, allocator);
    defer allocator.free(time_domain);

    // Extract real part and trim to output_len
    const output = try allocator.alloc(T, output_len);
    for (0..output_len) |i| {
        output[i] = time_domain[i].re;
    }

    return output;
}

// ============================================================================
// TESTS
// ============================================================================

test "convolve empty arrays returns empty result" {
    const allocator = testing.allocator;
    var a: [0]f64 = .{};
    var b: [0]f64 = .{};
    const result = try convolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    try testing.expectEqual(result.len, 0);
}

test "convolve with one empty array returns empty result" {
    const allocator = testing.allocator;
    var a: [0]f64 = .{};
    var b = [_]f64{ 1, 2, 3 };
    const result = try convolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    try testing.expectEqual(result.len, 0);
}

test "convolve single element with single element" {
    const allocator = testing.allocator;
    var a = [_]f64{2};
    var b = [_]f64{3};
    const result = try convolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    try testing.expectEqual(result.len, 1);
    try testing.expectApproxEqAbs(result[0], 6.0, 1e-10);
}

test "convolve impulse response returns original signal (zero-padded)" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 1, 2, 3 };
    var impulse = [_]f64{1};
    const result = try convolve(f64, allocator, signal[0..], impulse[0..]);
    defer allocator.free(result);
    try testing.expectEqual(result.len, 3);
    for (signal, result) |orig, res| {
        try testing.expectApproxEqAbs(orig, res, 1e-10);
    }
}

test "convolve known sequence [1,2,3] * [1,1]" {
    const allocator = testing.allocator;
    var a = [_]f64{ 1, 2, 3 };
    var b = [_]f64{ 1, 1 };
    const result = try convolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    try testing.expectEqual(result.len, 4);
    const expected = [_]f64{ 1, 3, 5, 3 };
    for (expected, result) |exp, res| {
        try testing.expectApproxEqAbs(exp, res, 1e-10);
    }
}

test "convolve commutative property" {
    const allocator = testing.allocator;
    var a = [_]f64{ 1, 2, 3 };
    var b = [_]f64{ 4, 5 };
    const result1 = try convolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result1);
    const result2 = try convolve(f64, allocator, b[0..], a[0..]);
    defer allocator.free(result2);
    try testing.expectEqual(result1.len, result2.len);
    for (result1, result2) |r1, r2| {
        try testing.expectApproxEqAbs(r1, r2, 1e-10);
    }
}

test "convolve output length equals a.len + b.len - 1" {
    const allocator = testing.allocator;
    var a = [_]f64{ 1, 2, 3, 4, 5 };
    var b = [_]f64{ 1, 2 };
    const result = try convolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    try testing.expectEqual(result.len, a.len + b.len - 1);
}

test "convolve of two ones has triangular shape" {
    const allocator = testing.allocator;
    var a = [_]f64{ 1, 1, 1 };
    var b = [_]f64{ 1, 1 };
    const result = try convolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    // Expected: [1, 2, 2, 1] (triangular pattern)
    const expected = [_]f64{ 1, 2, 2, 1 };
    try testing.expectEqual(result.len, expected.len);
    for (expected, result) |exp, res| {
        try testing.expectApproxEqAbs(exp, res, 1e-10);
    }
}

test "convolve with negative values" {
    const allocator = testing.allocator;
    var a = [_]f64{ 1, -2, 3 };
    var b = [_]f64{ 2, -1 };
    const result = try convolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    // a[0]*b[0] = 2
    // a[0]*b[1] + a[1]*b[0] = -1 + (-4) = -5
    // a[1]*b[1] + a[2]*b[0] = 2 + 6 = 8
    // a[2]*b[1] = -3
    const expected = [_]f64{ 2, -5, 8, -3 };
    try testing.expectEqual(result.len, expected.len);
    for (expected, result) |exp, res| {
        try testing.expectApproxEqAbs(exp, res, 1e-10);
    }
}

test "convolve f32 type" {
    const allocator = testing.allocator;
    var a = [_]f32{ 1, 2 };
    var b = [_]f32{ 3, 4 };
    const result = try convolve(f32, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    const expected = [_]f32{ 3, 10, 8 };
    try testing.expectEqual(result.len, expected.len);
    for (expected, result) |exp, res| {
        try testing.expectApproxEqAbs(exp, res, 1e-5);
    }
}

test "convolve large arrays" {
    const allocator = testing.allocator;
    const n = 100;
    var a = try allocator.alloc(f64, n);
    defer allocator.free(a);
    var b = try allocator.alloc(f64, n);
    defer allocator.free(b);

    // Fill with pattern
    for (0..n) |i| {
        const i_f: f64 = @floatFromInt(i);
        a[i] = @sin(i_f * 0.1);
        b[i] = @cos(i_f * 0.1);
    }

    const result = try convolve(f64, allocator, a, b);
    defer allocator.free(result);
    try testing.expectEqual(result.len, 2 * n - 1);
}

test "correlate empty arrays returns empty result" {
    const allocator = testing.allocator;
    var a: [0]f64 = .{};
    var b: [0]f64 = .{};
    const result = try correlate(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    try testing.expectEqual(result.len, 0);
}

test "correlate with one empty array returns empty result" {
    const allocator = testing.allocator;
    var a: [0]f64 = .{};
    var b = [_]f64{ 1, 2, 3 };
    const result = try correlate(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    try testing.expectEqual(result.len, 0);
}

test "correlate single elements" {
    const allocator = testing.allocator;
    var a = [_]f64{2};
    var b = [_]f64{3};
    const result = try correlate(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    try testing.expectEqual(result.len, 1);
    // correlate(a, b)[0] = a[0] * b[0] = 6
    try testing.expectApproxEqAbs(result[0], 6.0, 1e-10);
}

test "correlate autocorrelation peak at zero lag" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 1, 2, 3, 4 };
    const result = try correlate(f64, allocator, signal[0..], signal[0..]);
    defer allocator.free(result);
    // Autocorrelation at lag 0: signal energy
    const expected_peak = 1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0;
    try testing.expectApproxEqAbs(result[0], expected_peak, 1e-10);
}

test "correlate known sequence [1,2,3] with [2,3,4]" {
    const allocator = testing.allocator;
    var a = [_]f64{ 1, 2, 3 };
    var b = [_]f64{ 2, 3, 4 };
    const result = try correlate(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    // output_len = a.len + b.len - 1 = 3 + 3 - 1 = 5
    // lag 0: 1*2 + 2*3 + 3*4 = 2 + 6 + 12 = 20
    // lag 1: 1*3 + 2*4 = 3 + 8 = 11
    // lag 2: 1*4 = 4
    // lag 3: (b doesn't have index 5, so nothing from a) = 0
    // lag 4: nothing = 0
    const expected = [_]f64{ 20, 11, 4, 0, 0 };
    try testing.expectEqual(result.len, expected.len);
    for (expected, result) |exp, res| {
        try testing.expectApproxEqAbs(exp, res, 1e-10);
    }
}

test "correlate output length equals a.len + b.len - 1" {
    const allocator = testing.allocator;
    var a = [_]f64{ 1, 2, 3, 4 };
    var b = [_]f64{ 5, 6 };
    const result = try correlate(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    try testing.expectEqual(result.len, a.len + b.len - 1);
}

test "correlate with negative values" {
    const allocator = testing.allocator;
    var a = [_]f64{ 1, -1, 1 };
    var b = [_]f64{ 1, 1 };
    const result = try correlate(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    // lag 0: 1*1 + (-1)*1 = 1 - 1 = 0
    // lag 1: 1*1 = 1
    // lag 2: (-1)*? (b doesn't have index 3) + 1*? = 0
    // lag 3: 1*? (b doesn't have index 4) = 0
    // Wait, output_len = 3 + 2 - 1 = 4
    // lag 0: a[0]*b[0] + a[1]*b[1] = 1*1 + (-1)*1 = 0
    // lag 1: a[0]*b[1] = 1
    // lag 2: nothing (b[2] doesn't exist) = 0
    // lag 3: nothing = 0
    const expected = [_]f64{ 0, 1, 0, 0 };
    try testing.expectEqual(result.len, expected.len);
    for (expected, result) |exp, res| {
        try testing.expectApproxEqAbs(exp, res, 1e-10);
    }
}

test "correlate f32 type" {
    const allocator = testing.allocator;
    var a = [_]f32{ 1, 2 };
    var b = [_]f32{ 3, 4 };
    const result = try correlate(f32, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    try testing.expectEqual(result.len, 3);
    // lag 0: 1*3 + 2*4 = 11
    // lag 1: 1*4 = 4
    // lag 2: nothing = 0
    const expected = [_]f32{ 11, 4, 0 };
    for (expected, result) |exp, res| {
        try testing.expectApproxEqAbs(exp, res, 1e-5);
    }
}

test "correlate large arrays" {
    const allocator = testing.allocator;
    const n = 50;
    var a = try allocator.alloc(f64, n);
    defer allocator.free(a);
    var b = try allocator.alloc(f64, n);
    defer allocator.free(b);

    // Fill with pattern
    for (0..n) |i| {
        const i_f: f64 = @floatFromInt(i);
        a[i] = @sin(i_f * 0.1);
        b[i] = @cos(i_f * 0.15);
    }

    const result = try correlate(f64, allocator, a, b);
    defer allocator.free(result);
    try testing.expectEqual(result.len, 2 * n - 1);
}

test "fftconvolve empty arrays returns empty result" {
    const allocator = testing.allocator;
    var a: [0]f64 = .{};
    var b: [0]f64 = .{};
    const result = try fftconvolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    try testing.expectEqual(result.len, 0);
}

test "fftconvolve with one empty array returns empty result" {
    const allocator = testing.allocator;
    var a: [0]f64 = .{};
    var b = [_]f64{ 1, 2, 3 };
    const result = try fftconvolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    try testing.expectEqual(result.len, 0);
}

test "fftconvolve single element equals direct convolve" {
    const allocator = testing.allocator;
    var a = [_]f64{2};
    var b = [_]f64{3};
    const direct = try convolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(direct);
    const fft_based = try fftconvolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(fft_based);
    try testing.expectEqual(direct.len, fft_based.len);
    for (direct, fft_based) |d, f| {
        try testing.expectApproxEqAbs(d, f, 1e-9);
    }
}

test "fftconvolve known sequence matches direct convolve" {
    const allocator = testing.allocator;
    var a = [_]f64{ 1, 2, 3 };
    var b = [_]f64{ 1, 1 };
    const direct = try convolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(direct);
    const fft_based = try fftconvolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(fft_based);
    try testing.expectEqual(direct.len, fft_based.len);
    for (direct, fft_based) |d, f| {
        try testing.expectApproxEqAbs(d, f, 1e-8);
    }
}

test "fftconvolve commutative property" {
    const allocator = testing.allocator;
    var a = [_]f64{ 1, 2, 3 };
    var b = [_]f64{ 4, 5 };
    const result1 = try fftconvolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result1);
    const result2 = try fftconvolve(f64, allocator, b[0..], a[0..]);
    defer allocator.free(result2);
    try testing.expectEqual(result1.len, result2.len);
    for (result1, result2) |r1, r2| {
        try testing.expectApproxEqAbs(r1, r2, 1e-8);
    }
}

test "fftconvolve output length equals a.len + b.len - 1" {
    const allocator = testing.allocator;
    var a = [_]f64{ 1, 2, 3, 4, 5 };
    var b = [_]f64{ 1, 2 };
    const result = try fftconvolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    try testing.expectEqual(result.len, a.len + b.len - 1);
}

test "fftconvolve triangular pattern" {
    const allocator = testing.allocator;
    var a = [_]f64{ 1, 1, 1 };
    var b = [_]f64{ 1, 1 };
    const result = try fftconvolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    const expected = [_]f64{ 1, 2, 2, 1 };
    try testing.expectEqual(result.len, expected.len);
    for (expected, result) |exp, res| {
        try testing.expectApproxEqAbs(exp, res, 1e-8);
    }
}

test "fftconvolve with negative values matches direct" {
    const allocator = testing.allocator;
    var a = [_]f64{ 1, -2, 3 };
    var b = [_]f64{ 2, -1 };
    const direct = try convolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(direct);
    const fft_based = try fftconvolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(fft_based);
    try testing.expectEqual(direct.len, fft_based.len);
    for (direct, fft_based) |d, f| {
        try testing.expectApproxEqAbs(d, f, 1e-8);
    }
}

test "fftconvolve f32 type" {
    const allocator = testing.allocator;
    var a = [_]f32{ 1, 2 };
    var b = [_]f32{ 3, 4 };
    const result = try fftconvolve(f32, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    const expected = [_]f32{ 3, 10, 8 };
    try testing.expectEqual(result.len, expected.len);
    for (expected, result) |exp, res| {
        try testing.expectApproxEqAbs(exp, res, 1e-5);
    }
}

test "fftconvolve large arrays (100 samples)" {
    const allocator = testing.allocator;
    const n = 100;
    var a = try allocator.alloc(f64, n);
    defer allocator.free(a);
    var b = try allocator.alloc(f64, 16);
    defer allocator.free(b);

    // Fill with pattern
    for (0..n) |i| {
        const i_f: f64 = @floatFromInt(i);
        a[i] = @sin(i_f * 0.1);
    }
    for (0..16) |i| {
        const i_f: f64 = @floatFromInt(i);
        b[i] = @exp(-i_f * 0.1);
    }

    const direct = try convolve(f64, allocator, a, b);
    defer allocator.free(direct);
    const fft_based = try fftconvolve(f64, allocator, a, b);
    defer allocator.free(fft_based);
    try testing.expectEqual(direct.len, fft_based.len);
    for (direct, fft_based) |d, f| {
        try testing.expectApproxEqAbs(d, f, 1e-6);
    }
}

test "fftconvolve vs convolve match on random data" {
    const allocator = testing.allocator;
    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();

    var a = try allocator.alloc(f64, 32);
    defer allocator.free(a);
    var b = try allocator.alloc(f64, 32);
    defer allocator.free(b);

    for (0..32) |i| {
        a[i] = random.float(f64) * 2 - 1;
        b[i] = random.float(f64) * 2 - 1;
    }

    const direct = try convolve(f64, allocator, a, b);
    defer allocator.free(direct);
    const fft_based = try fftconvolve(f64, allocator, a, b);
    defer allocator.free(fft_based);

    try testing.expectEqual(direct.len, fft_based.len);
    for (direct, fft_based) |d, f| {
        try testing.expectApproxEqAbs(d, f, 1e-7);
    }
}

test "correlate autocorrelation decreases as lag increases" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 1, 2, 3, 4 };
    const result = try correlate(f64, allocator, signal[0..], signal[0..]);
    defer allocator.free(result);

    // For autocorrelation, lag 0 gives maximum (energy)
    var expected_peak: f64 = 0;
    for (signal) |s| {
        expected_peak += s * s;
    }
    try testing.expectApproxEqAbs(result[0], expected_peak, 1e-10);

    // Autocorrelation should monotonically decrease as lag increases
    // (in the non-negative lag side)
    for (1..signal.len) |lag| {
        try testing.expect(result[lag] <= result[lag - 1]);
    }
}

test "no memory leaks in convolve" {
    const allocator = testing.allocator;
    for (0..10) |_| {
        var a = [_]f64{ 1, 2, 3 };
        var b = [_]f64{ 4, 5 };
        const result = try convolve(f64, allocator, a[0..], b[0..]);
        allocator.free(result);
    }
}

test "no memory leaks in correlate" {
    const allocator = testing.allocator;
    for (0..10) |_| {
        var a = [_]f64{ 1, 2, 3 };
        var b = [_]f64{ 4, 5 };
        const result = try correlate(f64, allocator, a[0..], b[0..]);
        allocator.free(result);
    }
}

test "no memory leaks in fftconvolve" {
    const allocator = testing.allocator;
    for (0..10) |_| {
        var a = [_]f64{ 1, 2, 3 };
        var b = [_]f64{ 4, 5 };
        const result = try fftconvolve(f64, allocator, a[0..], b[0..]);
        allocator.free(result);
    }
}

test "convolve with zeros" {
    const allocator = testing.allocator;
    var a = [_]f64{ 1, 2 };
    var b = [_]f64{ 0, 1 };
    const result = try convolve(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    // a[0]*b[0] = 1*0 = 0
    // a[0]*b[1] + a[1]*b[0] = 1*1 + 2*0 = 1
    // a[1]*b[1] = 2*1 = 2
    const expected = [_]f64{ 0, 1, 2 };
    try testing.expectEqual(result.len, expected.len);
    for (expected, result) |exp, res| {
        try testing.expectApproxEqAbs(exp, res, 1e-10);
    }
}

test "correlate with constant signal" {
    const allocator = testing.allocator;
    var a = [_]f64{ 2, 2, 2 };
    var b = [_]f64{ 3, 3 };
    const result = try correlate(f64, allocator, a[0..], b[0..]);
    defer allocator.free(result);
    // output_len = 3 + 2 - 1 = 4
    // lag 0: 2*3 + 2*3 = 12
    // lag 1: 2*3 = 6
    // lag 2: 0 (b doesn't have index 4)
    // lag 3: 0
    const expected = [_]f64{ 12, 6, 0, 0 };
    try testing.expectEqual(result.len, expected.len);
    for (expected, result) |exp, res| {
        try testing.expectApproxEqAbs(exp, res, 1e-10);
    }
}
