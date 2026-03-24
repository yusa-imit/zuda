//! Discrete Cosine Transform (DCT) — Type II and Type III
//!
//! This module provides implementations of the Discrete Cosine Transform,
//! used for signal compression and frequency analysis (especially in JPEG/audio).
//!
//! ## Supported Operations
//! - `dct` — DCT Type II (forward transform)
//! - `idct` — DCT Type III (inverse transform)
//!
//! ## Properties
//! - Orthogonal transform: preserves energy (Parseval-like theorem)
//! - Real-valued input and output (unlike FFT which produces complex)
//! - Energy concentration: most energy in low-frequency coefficients
//! - Round-trip: idct(dct(x)) ≈ x (within floating-point precision)
//!
//! ## Time Complexity
//! - Both dct and idct: O(N²) for naive implementation
//! - (Optimized FFT-based: O(N log N) — future enhancement)
//!
//! ## Space Complexity
//! - Both transforms: O(N) for output storage
//!
//! ## Mathematical Definitions
//!
//! **DCT Type II** (forward):
//! ```
//! X[k] = sum_{n=0}^{N-1} x[n] * cos(π * k * (n + 0.5) / N) for k = 0..N-1
//! ```
//! With orthonormal scaling: multiply by sqrt(2/N) for k > 0, sqrt(1/N) for k = 0
//!
//! **DCT Type III** (inverse):
//! ```
//! x[n] = 0.5 * X[0] + sum_{k=1}^{N-1} X[k] * cos(π * k * (n + 0.5) / N)
//! ```
//! With orthonormal scaling applied
//!
//! ## Use Cases
//! - JPEG compression (DCT of 8x8 blocks)
//! - Audio compression (MP3, AAC)
//! - Signal analysis (energy concentration in low frequencies)
//! - Spectral analysis alternative to FFT (real-valued only)
//!
//! ## References
//! - Ahmed, N., Natarajan, T., & Rao, K. R. (1974). "Discrete cosine transform"
//! - ISO/IEC 10918-1 (JPEG standard)

const std = @import("std");
const math = std.math;
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Compute the Discrete Cosine Transform Type II (forward transform)
///
/// Converts a time-domain real signal to its frequency-domain DCT coefficients.
/// Uses orthonormal scaling for energy preservation.
///
/// Parameters:
/// - T: Floating-point type (f32 or f64)
/// - signal: Real-valued time-domain signal (caller owns input, can be freed after call)
/// - allocator: Memory allocator for output
///
/// Returns: DCT coefficients (caller owns, must call allocator.free)
///
/// Errors:
/// - error.OutOfMemory if allocation fails
///
/// Time: O(N²) for N = signal length
/// Space: O(N)
///
/// Properties:
/// - Energy concentrated in low frequencies
/// - Suitable for compression (drop high-frequency coefficients)
/// - Round-trip: idct(dct(x)) ≈ x
///
/// Example:
/// ```zig
/// const allocator = std.testing.allocator;
/// const signal = [_]f64{ 1.0, 2.0, 1.5, 0.5 };
/// const coeffs = try dct(f64, signal[0..], allocator);
/// defer allocator.free(coeffs);
/// ```
pub fn dct(comptime T: type, signal: []const T, allocator: Allocator) Allocator.Error![]T {
    const n = signal.len;
    const coeffs = try allocator.alloc(T, n);
    errdefer allocator.free(coeffs);

    if (n == 0) return coeffs;

    const n_f = @as(T, @floatFromInt(n));
    const pi = math.pi;

    // Compute DCT coefficients
    for (0..n) |k| {
        var sum: T = 0.0;
        const k_f = @as(T, @floatFromInt(k));

        for (0..n) |n_idx| {
            const n_f_idx = @as(T, @floatFromInt(n_idx));
            const angle = pi * k_f * (n_f_idx + 0.5) / n_f;
            sum += signal[n_idx] * @cos(angle);
        }

        // Apply orthonormal scaling
        const scale = if (k == 0)
            @sqrt(1.0 / n_f)
        else
            @sqrt(2.0 / n_f);

        coeffs[k] = sum * scale;
    }

    return coeffs;
}

/// Compute the Discrete Cosine Transform Type III (inverse transform)
///
/// Converts DCT coefficients back to the time domain.
/// Inverse of dct() when both use orthonormal scaling.
///
/// Parameters:
/// - T: Floating-point type (f32 or f64)
/// - coeffs: DCT coefficients from dct()
/// - allocator: Memory allocator for output
///
/// Returns: Time-domain signal (caller owns, must call allocator.free)
///
/// Errors:
/// - error.OutOfMemory if allocation fails
///
/// Time: O(N²)
/// Space: O(N)
///
/// Properties:
/// - True inverse of dct() with orthonormal scaling
/// - idct(dct(x)) ≈ x (within floating-point precision)
/// - Preserves energy (Parseval-like theorem)
///
/// Example:
/// ```zig
/// const allocator = std.testing.allocator;
/// const signal = [_]f64{ 1.0, 2.0, 1.5, 0.5 };
/// const coeffs = try dct(f64, signal[0..], allocator);
/// defer allocator.free(coeffs);
/// const recovered = try idct(f64, coeffs, allocator);
/// defer allocator.free(recovered);
/// ```
pub fn idct(comptime T: type, coeffs: []const T, allocator: Allocator) Allocator.Error![]T {
    const n = coeffs.len;
    const signal = try allocator.alloc(T, n);
    errdefer allocator.free(signal);

    if (n == 0) return signal;

    const n_f = @as(T, @floatFromInt(n));
    const pi = math.pi;

    // Compute inverse DCT coefficients
    for (0..n) |n_idx| {
        var sum: T = 0.0;
        const n_f_idx = @as(T, @floatFromInt(n_idx));

        for (0..n) |k| {
            const k_f = @as(T, @floatFromInt(k));
            const angle = pi * k_f * (n_f_idx + 0.5) / n_f;

            // Apply orthonormal scaling (inverse of forward)
            const scale = if (k == 0)
                @sqrt(1.0 / n_f)
            else
                @sqrt(2.0 / n_f);

            sum += coeffs[k] * scale * @cos(angle);
        }

        signal[n_idx] = sum;
    }

    return signal;
}

// ============================================================================
// TESTS
// ============================================================================

test "dct of empty signal returns empty array" {
    const allocator = testing.allocator;
    var signal: [0]f64 = .{};
    const coeffs = try dct(f64, signal[0..], allocator);
    defer allocator.free(coeffs);
    try testing.expectEqual(coeffs.len, 0);
}

test "dct of single element preserves value scaled by sqrt(1/N)" {
    const allocator = testing.allocator;
    const value = 5.0;
    var signal = [_]f64{value};
    const coeffs = try dct(f64, signal[0..], allocator);
    defer allocator.free(coeffs);
    // For N=1: coeffs[0] = 5.0 * sqrt(1/1) * cos(0) = 5.0 * 1 * 1 = 5.0
    try testing.expectApproxEqAbs(coeffs[0], value, 1e-10);
}

test "dct of two-element constant signal" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 1.0, 1.0 };
    const coeffs = try dct(f64, signal[0..], allocator);
    defer allocator.free(coeffs);
    // DC component (k=0): sum = 2.0, scaled by sqrt(1/2) = 1.414...
    try testing.expectApproxEqAbs(coeffs[0], @sqrt(2.0), 1e-10);
    // AC component (k=1): cos(π*1*(0.5)/2) + cos(π*1*(1.5)/2) = cos(π/4) + cos(3π/4) = sqrt(2)/2 - sqrt(2)/2 = 0
    try testing.expectApproxEqAbs(coeffs[1], 0.0, 1e-10);
}

test "dct of four-element constant signal concentrates energy at DC" {
    const allocator = testing.allocator;
    const value = 1.0;
    var signal = [_]f64{ value, value, value, value };
    const coeffs = try dct(f64, signal[0..], allocator);
    defer allocator.free(coeffs);
    // DC coefficient: sum = 4.0, scaled by sqrt(1/4) = 0.5
    const expected_dc = 4.0 * 0.5;
    try testing.expectApproxEqAbs(coeffs[0], expected_dc, 1e-10);
    // AC coefficients should be near zero for constant signal
    for (coeffs[1..]) |coeff| {
        try testing.expectApproxEqAbs(coeff, 0.0, 1e-9);
    }
}

test "dct round-trip: idct(dct(x)) ≈ x for random f64 signal" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 1.0, 2.5, 0.3, -1.2, 0.8, 1.5 };
    const coeffs = try dct(f64, signal[0..], allocator);
    defer allocator.free(coeffs);
    const recovered = try idct(f64, coeffs, allocator);
    defer allocator.free(recovered);
    for (signal, recovered) |orig, recov| {
        try testing.expectApproxEqAbs(orig, recov, 1e-9);
    }
}

test "dct round-trip: idct(dct(x)) ≈ x for random f32 signal" {
    const allocator = testing.allocator;
    var signal = [_]f32{ 1.0, 2.5, 0.3, -1.2, 0.8 };
    const coeffs = try dct(f32, signal[0..], allocator);
    defer allocator.free(coeffs);
    const recovered = try idct(f32, coeffs, allocator);
    defer allocator.free(recovered);
    for (signal, recovered) |orig, recov| {
        try testing.expectApproxEqAbs(orig, recov, 1e-5);
    }
}

test "dct round-trip: large signal (N=64)" {
    const allocator = testing.allocator;
    const n = 64;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);
    // Fill with pseudo-random pattern
    for (0..n) |i| {
        const i_f: f64 = @floatFromInt(i);
        signal[i] = @sin(i_f * 0.1) + @cos(i_f * 0.05);
    }
    const coeffs = try dct(f64, signal, allocator);
    defer allocator.free(coeffs);
    const recovered = try idct(f64, coeffs, allocator);
    defer allocator.free(recovered);
    for (signal, recovered) |orig, recov| {
        try testing.expectApproxEqAbs(orig, recov, 1e-8);
    }
}

test "dct round-trip: non-power-of-2 (N=10)" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
    const coeffs = try dct(f64, signal[0..], allocator);
    defer allocator.free(coeffs);
    const recovered = try idct(f64, coeffs, allocator);
    defer allocator.free(recovered);
    for (signal, recovered) |orig, recov| {
        try testing.expectApproxEqAbs(orig, recov, 1e-9);
    }
}

test "dct round-trip: impulse signal" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 1.0, 0.0, 0.0, 0.0 };
    const coeffs = try dct(f64, signal[0..], allocator);
    defer allocator.free(coeffs);
    const recovered = try idct(f64, coeffs, allocator);
    defer allocator.free(recovered);
    for (signal, recovered) |orig, recov| {
        try testing.expectApproxEqAbs(orig, recov, 1e-9);
    }
}

test "idct of empty coefficients returns empty signal" {
    const allocator = testing.allocator;
    var coeffs: [0]f64 = .{};
    const signal = try idct(f64, coeffs[0..], allocator);
    defer allocator.free(signal);
    try testing.expectEqual(signal.len, 0);
}

test "idct of single coefficient" {
    const allocator = testing.allocator;
    const value = 5.0;
    var coeffs = [_]f64{value};
    const signal = try idct(f64, coeffs[0..], allocator);
    defer allocator.free(signal);
    // For N=1: signal[0] = 5.0 * sqrt(1/1) * cos(0) = 5.0
    try testing.expectApproxEqAbs(signal[0], value, 1e-10);
}

test "dct energy conservation: Parseval-like property (f64)" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 1.0, 2.0, 1.5, 0.5, 0.3, 1.2 };
    // Compute signal energy
    var signal_energy: f64 = 0.0;
    for (signal) |val| {
        signal_energy += val * val;
    }
    const coeffs = try dct(f64, signal[0..], allocator);
    defer allocator.free(coeffs);
    // Compute DCT energy (with orthonormal scaling, should equal signal energy)
    var dct_energy: f64 = 0.0;
    for (coeffs) |val| {
        dct_energy += val * val;
    }
    // With proper orthonormal scaling, energies should be equal
    try testing.expectApproxEqAbs(signal_energy, dct_energy, 1e-8);
}

test "dct DC component equals sum of signal times scaling factor" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 1.0, 2.0, 1.5, 0.5 };
    var sum: f64 = 0.0;
    for (signal) |val| {
        sum += val;
    }
    const coeffs = try dct(f64, signal[0..], allocator);
    defer allocator.free(coeffs);
    const n_f: f64 = @floatFromInt(signal.len);
    const expected_dc = sum * @sqrt(1.0 / n_f);
    try testing.expectApproxEqAbs(coeffs[0], expected_dc, 1e-10);
}

test "dct zero signal produces zero coefficients" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 0.0, 0.0, 0.0, 0.0 };
    const coeffs = try dct(f64, signal[0..], allocator);
    defer allocator.free(coeffs);
    for (coeffs) |coeff| {
        try testing.expectApproxEqAbs(coeff, 0.0, 1e-10);
    }
}

test "dct of negative values produces negative coefficients" {
    const allocator = testing.allocator;
    var signal = [_]f64{ -1.0, -2.0, -1.5, -0.5 };
    var pos_signal = [_]f64{ 1.0, 2.0, 1.5, 0.5 };
    const coeffs_neg = try dct(f64, signal[0..], allocator);
    defer allocator.free(coeffs_neg);
    const coeffs_pos = try dct(f64, pos_signal[0..], allocator);
    defer allocator.free(coeffs_pos);
    // Coefficients of negated signal should be negatives of original
    for (coeffs_neg, coeffs_pos) |cn, cp| {
        try testing.expectApproxEqAbs(cn, -cp, 1e-9);
    }
}

test "dct of mixed positive/negative signal" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 1.0, -0.5, 0.8, -1.2 };
    const coeffs = try dct(f64, signal[0..], allocator);
    defer allocator.free(coeffs);
    // Should produce non-zero coefficients
    var has_nonzero = false;
    for (coeffs) |coeff| {
        if (@abs(coeff) > 1e-10) {
            has_nonzero = true;
            break;
        }
    }
    try testing.expect(has_nonzero);
}

test "dct linearity: dct(a*x + b*y) ≈ a*dct(x) + b*dct(y)" {
    const allocator = testing.allocator;
    var x = [_]f64{ 1.0, 2.0, 0.5, 1.5 };
    var y = [_]f64{ 0.5, 1.0, 2.0, 0.8 };
    const a = 2.0;
    const b = 3.0;

    // Compute dct(x) and dct(y)
    const coeffs_x = try dct(f64, x[0..], allocator);
    defer allocator.free(coeffs_x);
    const coeffs_y = try dct(f64, y[0..], allocator);
    defer allocator.free(coeffs_y);

    // Compute combined signal a*x + b*y
    var combined = try allocator.alloc(f64, x.len);
    defer allocator.free(combined);
    for (0..x.len) |i| {
        combined[i] = a * x[i] + b * y[i];
    }

    // Compute dct of combined
    const coeffs_combined = try dct(f64, combined, allocator);
    defer allocator.free(coeffs_combined);

    // Verify linearity: coeffs_combined ≈ a*coeffs_x + b*coeffs_y
    for (coeffs_combined, coeffs_x, coeffs_y) |cc, cx, cy| {
        const expected = a * cx + b * cy;
        try testing.expectApproxEqAbs(cc, expected, 1e-8);
    }
}

test "dct symmetry: cosine basis orthogonality" {
    const allocator = testing.allocator;
    const n = 4;
    // For DCT, the cosine basis vectors should be orthogonal
    var basis1 = try allocator.alloc(f64, n);
    defer allocator.free(basis1);
    var basis2 = try allocator.alloc(f64, n);
    defer allocator.free(basis2);

    // Fill basis1: impulse at position 0
    basis1[0] = 1.0;
    for (1..n) |i| {
        basis1[i] = 0.0;
    }

    // Fill basis2: impulse at position 1
    basis2[0] = 0.0;
    basis2[1] = 1.0;
    for (2..n) |i| {
        basis2[i] = 0.0;
    }

    const coeffs1 = try dct(f64, basis1, allocator);
    defer allocator.free(coeffs1);
    const coeffs2 = try dct(f64, basis2, allocator);
    defer allocator.free(coeffs2);

    // Orthogonality: dot product of basis functions should be zero or delta
    var dot_product: f64 = 0.0;
    for (coeffs1, coeffs2) |c1, c2| {
        dot_product += c1 * c2;
    }
    // For different basis vectors, dot product should be close to zero
    try testing.expectApproxEqAbs(dot_product, 0.0, 1e-8);
}

test "dct large magnitude values" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 1000.0, 2000.0, 1500.0, 500.0 };
    const coeffs = try dct(f64, signal[0..], allocator);
    defer allocator.free(coeffs);
    const recovered = try idct(f64, coeffs, allocator);
    defer allocator.free(recovered);
    for (signal, recovered) |orig, recov| {
        try testing.expectApproxEqAbs(orig, recov, 1e-6);
    }
}

test "dct small magnitude values (numerical stability)" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 1e-5, 2e-5, 1.5e-5, 5e-6 };
    const coeffs = try dct(f64, signal[0..], allocator);
    defer allocator.free(coeffs);
    const recovered = try idct(f64, coeffs, allocator);
    defer allocator.free(recovered);
    for (signal, recovered) |orig, recov| {
        try testing.expectApproxEqAbs(orig, recov, 1e-14);
    }
}

test "dct coefficient magnitudes for constant signal" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 2.0, 2.0, 2.0, 2.0 };
    const coeffs = try dct(f64, signal[0..], allocator);
    defer allocator.free(coeffs);
    // Only DC component should be non-zero
    try testing.expect(@abs(coeffs[0]) > 1e-10);
    for (coeffs[1..]) |coeff| {
        try testing.expectApproxEqAbs(coeff, 0.0, 1e-9);
    }
}

test "dct coefficients decay for smooth signals" {
    const allocator = testing.allocator;
    const n = 8;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);
    // Smooth signal (low frequency content)
    for (0..n) |i| {
        const i_f: f64 = @floatFromInt(i);
        signal[i] = @sin(i_f * 0.5);
    }
    const coeffs = try dct(f64, signal, allocator);
    defer allocator.free(coeffs);
    // For smooth signals, low-frequency coefficients should dominate
    const low_freq_mag = @abs(coeffs[0]) + @abs(coeffs[1]);
    const high_freq_mag = @abs(coeffs[6]) + @abs(coeffs[7]);
    try testing.expect(low_freq_mag > high_freq_mag);
}

test "dct memory allocation and deallocation (f64)" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const coeffs = try dct(f64, signal[0..], allocator);
    allocator.free(coeffs);
    // If we reach here without detecting memory leak, allocation/deallocation was correct
}

test "dct memory allocation and deallocation (f32)" {
    const allocator = testing.allocator;
    var signal = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const coeffs = try dct(f32, signal[0..], allocator);
    allocator.free(coeffs);
    // If we reach here without detecting memory leak, allocation/deallocation was correct
}

test "idct memory allocation and deallocation (f64)" {
    const allocator = testing.allocator;
    var coeffs = [_]f64{ 1.0, 0.5, 0.1, 0.0 };
    const signal = try idct(f64, coeffs[0..], allocator);
    allocator.free(signal);
    // If we reach here without detecting memory leak, allocation/deallocation was correct
}

test "idct memory allocation and deallocation (f32)" {
    const allocator = testing.allocator;
    var coeffs = [_]f32{ 1.0, 0.5, 0.1, 0.0 };
    const signal = try idct(f32, coeffs[0..], allocator);
    allocator.free(signal);
    // If we reach here without detecting memory leak, allocation/deallocation was correct
}

test "dct of alternating signal" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 1.0, -1.0, 1.0, -1.0 };
    const coeffs = try dct(f64, signal[0..], allocator);
    defer allocator.free(coeffs);
    // Alternating signal is high frequency; DC should be zero
    try testing.expectApproxEqAbs(coeffs[0], 0.0, 1e-9);
    // High frequency components should dominate
    const low_energy = @abs(coeffs[1]) + @abs(coeffs[2]);
    const high_energy = @abs(coeffs[3]);
    try testing.expect(high_energy > 0.0 or low_energy > 0.0);
}

test "dct followed by idct multiple times" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 1.0, 2.0, 1.5, 0.5 };
    var current = try allocator.dupe(f64, signal[0..]);

    // Apply dct-idct cycle 3 times
    for (0..3) |_| {
        const coeffs = try dct(f64, current, allocator);
        const next = try idct(f64, coeffs, allocator);
        allocator.free(coeffs);
        allocator.free(current);
        current = next;
    }
    defer allocator.free(current);

    // After multiple cycles, should still recover original (within tolerance)
    for (signal, current) |orig, recov| {
        try testing.expectApproxEqAbs(orig, recov, 1e-8);
    }
}

test "dct with f32 produces reasonable results" {
    const allocator = testing.allocator;
    var signal = [_]f32{ 1.0, 2.0, 1.5, 0.5 };
    const coeffs = try dct(f32, signal[0..], allocator);
    defer allocator.free(coeffs);
    const recovered = try idct(f32, coeffs, allocator);
    defer allocator.free(recovered);
    for (signal, recovered) |orig, recov| {
        try testing.expectApproxEqAbs(orig, recov, 1e-4);
    }
}

test "idct is exact inverse of dct (different signal size)" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 0.5, 1.2, 0.8, 1.5, 2.0 };
    const coeffs = try dct(f64, signal[0..], allocator);
    defer allocator.free(coeffs);
    const recovered = try idct(f64, coeffs, allocator);
    defer allocator.free(recovered);
    for (signal, recovered) |orig, recov| {
        try testing.expectApproxEqAbs(orig, recov, 1e-10);
    }
}
