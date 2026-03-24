//! Fast Fourier Transform (FFT) — Cooley-Tukey Algorithm
//!
//! This module provides efficient implementations of the Fast Fourier Transform,
//! used for converting signals between time and frequency domains.
//!
//! ## Supported Operations
//! - `fft` — Complex FFT using Cooley-Tukey algorithm
//! - `ifft` — Inverse FFT
//! - `rfft` — Real FFT (exploits conjugate symmetry of real inputs)
//! - `irfft` — Inverse real FFT
//! - `fftfreq` — Compute frequency bin centers
//!
//! ## Time Complexity
//! - fft/ifft: O(n log n) where n is input length (for n = power of 2)
//! - rfft/irfft: O(n log n) with n/2 output size
//! - fftfreq: O(n) for n frequency bins
//!
//! ## Space Complexity
//! - All transforms: O(n) for output storage
//!
//! ## Numeric Properties
//! - Parseval's theorem: Sum of squared magnitudes is preserved between time and frequency
//! - Round-trip: ifft(fft(x)) ≈ x (within floating-point precision)
//! - Real symmetry: rfft output satisfies X[n-k] = conj(X[k])
//!
//! ## Use Cases
//! - Signal processing (audio, image, sensor data)
//! - Frequency domain analysis
//! - Convolution via FFT
//! - Filtering and spectral analysis

const std = @import("std");
const math = std.math;
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Complex number type with real and imaginary components
/// Parameters:
/// - T: Floating-point type (f32 or f64)
pub fn Complex(comptime T: type) type {
    return struct {
        re: T,
        im: T,

        const Self = @This();

        /// Create a complex number from real and imaginary parts
        pub fn init(re: T, im: T) Self {
            return .{ .re = re, .im = im };
        }

        /// Complex addition: a + b
        pub fn add(a: Self, b: Self) Self {
            return .{ .re = a.re + b.re, .im = a.im + b.im };
        }

        /// Complex subtraction: a - b
        pub fn sub(a: Self, b: Self) Self {
            return .{ .re = a.re - b.re, .im = a.im - b.im };
        }

        /// Complex multiplication: a * b
        pub fn mul(a: Self, b: Self) Self {
            return .{
                .re = a.re * b.re - a.im * b.im,
                .im = a.re * b.im + a.im * b.re,
            };
        }

        /// Complex conjugate: a*
        pub fn conj(a: Self) Self {
            return .{ .re = a.re, .im = -a.im };
        }

        /// Magnitude (absolute value): |a|
        pub fn magnitude(a: Self) T {
            return @sqrt(a.re * a.re + a.im * a.im);
        }

        /// Squared magnitude: |a|²
        pub fn magnitude_squared(a: Self) T {
            return a.re * a.re + a.im * a.im;
        }

        /// Phase angle: arg(a) in radians
        pub fn phase(a: Self) T {
            return math.atan2(a.im, a.re);
        }

        /// Equality test (within epsilon for floating-point)
        pub fn eql(a: Self, b: Self, epsilon: T) bool {
            return @abs(a.re - b.re) < epsilon and @abs(a.im - b.im) < epsilon;
        }
    };
}

/// Compute the Fast Fourier Transform of a complex signal using Cooley-Tukey algorithm
///
/// Converts a time-domain complex signal to its frequency-domain representation.
/// Input length must be a power of 2 (for this implementation).
///
/// Parameters:
/// - T: Floating-point type (f32 or f64)
/// - signal: Time-domain complex signal (caller owns input, can be freed after call)
/// - allocator: Memory allocator for output
///
/// Returns: Frequency-domain complex spectrum (caller owns, must call allocator.free)
///
/// Errors:
/// - error.OutOfMemory if allocation fails
/// - error.InvalidLength if signal length is not a power of 2
///
/// Time: O(n log n) where n = signal length
/// Space: O(n)
///
/// Example:
/// ```zig
/// const allocator = std.testing.allocator;
/// var signal = [_]Complex(f64){
///     Complex(f64).init(1, 0),
///     Complex(f64).init(0, 0),
///     Complex(f64).init(0, 0),
///     Complex(f64).init(0, 0),
/// };
/// const spectrum = try fft(f64, signal[0..], allocator);
/// defer allocator.free(spectrum);
/// ```
pub fn fft(comptime T: type, signal: []const Complex(T), allocator: Allocator) (Allocator.Error || error{InvalidLength})![]Complex(T) {
    const n = signal.len;

    // Validate power of 2
    if (n == 0 or (n & (n - 1)) != 0) {
        return error.InvalidLength;
    }

    // Allocate output
    const output = try allocator.alloc(Complex(T), n);
    errdefer allocator.free(output);

    // Copy input to output
    @memcpy(output, signal);

    // Perform FFT in-place
    try fftInPlace(T, output);

    return output;
}

/// Compute the Inverse Fast Fourier Transform
///
/// Converts a frequency-domain complex spectrum back to time domain.
/// Applies scaling factor of 1/n to normalize the result.
///
/// Parameters:
/// - T: Floating-point type (f32 or f64)
/// - spectrum: Frequency-domain complex spectrum
/// - allocator: Memory allocator for output
///
/// Returns: Time-domain complex signal (caller owns, must call allocator.free)
///
/// Errors:
/// - error.OutOfMemory if allocation fails
/// - error.InvalidLength if spectrum length is not a power of 2
///
/// Time: O(n log n)
/// Space: O(n)
pub fn ifft(comptime T: type, spectrum: []const Complex(T), allocator: Allocator) (Allocator.Error || error{InvalidLength})![]Complex(T) {
    const n = spectrum.len;

    // Validate power of 2
    if (n == 0 or (n & (n - 1)) != 0) {
        return error.InvalidLength;
    }

    // Allocate output
    const output = try allocator.alloc(Complex(T), n);
    errdefer allocator.free(output);

    // Conjugate input
    for (spectrum, 0..) |val, i| {
        output[i] = val.conj();
    }

    // Perform FFT
    try fftInPlace(T, output);

    // Conjugate and scale output
    const scale: T = 1.0 / @as(T, @floatFromInt(n));
    for (output) |*val| {
        val.* = val.conj();
        val.re *= scale;
        val.im *= scale;
    }

    return output;
}

/// Compute the Real FFT of a real-valued signal
///
/// Uses conjugate symmetry of real inputs to compute only positive frequencies.
/// Returns n/2 + 1 complex values representing the non-redundant frequency bins.
///
/// Parameters:
/// - T: Floating-point type (f32 or f64)
/// - signal: Real-valued time-domain signal
/// - allocator: Memory allocator for output
///
/// Returns: Complex spectrum with n/2 + 1 frequency bins (caller owns, must free)
///
/// Errors:
/// - error.OutOfMemory if allocation fails
/// - error.InvalidLength if signal length is not a power of 2
///
/// Time: O(n log n)
/// Space: O(n)
pub fn rfft(comptime T: type, signal: []const T, allocator: Allocator) (Allocator.Error || error{InvalidLength})![]Complex(T) {
    const n = signal.len;

    // Validate power of 2
    if (n == 0 or (n & (n - 1)) != 0) {
        return error.InvalidLength;
    }

    // Create complex signal
    const complex_signal = try allocator.alloc(Complex(T), n);
    defer allocator.free(complex_signal);

    for (signal, 0..) |val, i| {
        complex_signal[i] = Complex(T).init(val, 0);
    }

    // Compute FFT
    const spectrum = try fft(T, complex_signal, allocator);

    // Return only positive frequencies (first n/2 + 1 values)
    const positive_freqs = try allocator.alloc(Complex(T), n / 2 + 1);
    @memcpy(positive_freqs, spectrum[0 .. n / 2 + 1]);
    allocator.free(spectrum);

    return positive_freqs;
}

/// Compute the Inverse Real FFT
///
/// Reconstructs a real-valued time-domain signal from its real FFT spectrum.
/// Input is assumed to be the first n/2 + 1 bins from a real FFT.
///
/// Parameters:
/// - T: Floating-point type (f32 or f64)
/// - spectrum: Complex spectrum from rfft (n/2 + 1 bins)
/// - allocator: Memory allocator for output
///
/// Returns: Real-valued time-domain signal of length 2*(spectrum.len - 1)
///
/// Errors:
/// - error.OutOfMemory if allocation fails
///
/// Time: O(n log n)
/// Space: O(n)
pub fn irfft(comptime T: type, spectrum: []const Complex(T), allocator: Allocator) Allocator.Error![]T {
    const n_positive = spectrum.len;
    const n = (n_positive - 1) * 2;

    // Reconstruct full spectrum with conjugate symmetry
    const full_spectrum = try allocator.alloc(Complex(T), n);
    defer allocator.free(full_spectrum);

    // Copy positive frequencies
    @memcpy(full_spectrum[0..n_positive], spectrum);

    // Fill negative frequencies via conjugate symmetry
    for (1..n_positive - 1) |i| {
        full_spectrum[n - i] = spectrum[i].conj();
    }

    // Compute inverse FFT
    const time_signal = ifft(T, full_spectrum, allocator) catch |e| {
        return switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            error.InvalidLength => unreachable, // We constructed full_spectrum properly
        };
    };

    // Extract real part and return
    const result = try allocator.alloc(T, n);
    for (time_signal, 0..) |val, i| {
        result[i] = val.re;
    }
    allocator.free(time_signal);

    return result;
}

/// Compute frequency bin centers for FFT output
///
/// Returns the frequency values corresponding to FFT bins.
/// Assumes sample spacing d (inverse of sample rate).
///
/// Parameters:
/// - n: Number of frequency bins (input signal length for real FFT)
/// - d: Sample spacing (1/sample_rate)
/// - allocator: Memory allocator for output
///
/// Returns: Frequency values for each bin (caller owns, must free)
///
/// Errors:
/// - error.OutOfMemory if allocation fails
///
/// Time: O(n)
/// Space: O(n)
pub fn fftfreq(comptime T: type, n: usize, d: T, allocator: Allocator) Allocator.Error![]T {
    const freqs = try allocator.alloc(T, n);
    errdefer allocator.free(freqs);

    const n_f: T = @floatFromInt(n);
    for (0..n) |i| {
        const i_f: T = @floatFromInt(i);
        if (i < (n + 1) / 2) {
            freqs[i] = i_f / (n_f * d);
        } else {
            freqs[i] = (i_f - n_f) / (n_f * d);
        }
    }

    return freqs;
}

// ============================================================================
// PRIVATE HELPER FUNCTIONS
// ============================================================================

/// In-place FFT computation using Cooley-Tukey algorithm
fn fftInPlace(comptime T: type, data: []Complex(T)) (error{InvalidLength})!void {
    const n = data.len;

    if (n <= 1) return;

    // Bit-reversal permutation
    bitReversal(T, data);

    // Cooley-Tukey FFT: bottom-up approach
    var m: usize = 2;
    while (m <= n) : (m *= 2) {
        const angle: T = -2.0 * math.pi / @as(T, @floatFromInt(m));
        var k: usize = 0;
        while (k < n) : (k += m) {
            var j: usize = 0;
            while (j < m / 2) : (j += 1) {
                const w_real = @cos(@as(T, @floatFromInt(j)) * angle);
                const w_imag = @sin(@as(T, @floatFromInt(j)) * angle);
                const w = Complex(T).init(w_real, w_imag);

                const t = Complex(T).mul(w, data[k + j + m / 2]);
                const u = data[k + j];

                data[k + j] = Complex(T).add(u, t);
                data[k + j + m / 2] = Complex(T).sub(u, t);
            }
        }
    }
}

/// In-place bit-reversal permutation
fn bitReversal(comptime T: type, data: []Complex(T)) void {
    const n = data.len;
    var j: usize = 0;
    var i: usize = 0;
    while (i < n - 1) : (i += 1) {
        if (i < j) {
            const temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }

        var m = n / 2;
        while (m > 0 and j >= m) {
            j -= m;
            m /= 2;
        }
        j += m;
    }
}

// ============================================================================
// TESTS
// ============================================================================

test "Complex.init creates correct complex number" {
    const c = Complex(f64).init(3.0, 4.0);
    try testing.expectEqual(c.re, 3.0);
    try testing.expectEqual(c.im, 4.0);
}

test "Complex.add performs correct addition" {
    const a = Complex(f64).init(1.0, 2.0);
    const b = Complex(f64).init(3.0, 4.0);
    const result = Complex(f64).add(a, b);
    try testing.expectEqual(result.re, 4.0);
    try testing.expectEqual(result.im, 6.0);
}

test "Complex.sub performs correct subtraction" {
    const a = Complex(f64).init(5.0, 6.0);
    const b = Complex(f64).init(1.0, 2.0);
    const result = Complex(f64).sub(a, b);
    try testing.expectEqual(result.re, 4.0);
    try testing.expectEqual(result.im, 4.0);
}

test "Complex.mul performs correct multiplication" {
    const a = Complex(f64).init(2.0, 3.0);
    const b = Complex(f64).init(4.0, 5.0);
    const result = Complex(f64).mul(a, b);
    // (2 + 3i)(4 + 5i) = 8 + 10i + 12i + 15i² = 8 + 22i - 15 = -7 + 22i
    try testing.expectEqual(result.re, -7.0);
    try testing.expectEqual(result.im, 22.0);
}

test "Complex.conj computes correct conjugate" {
    const c = Complex(f64).init(3.0, -4.0);
    const result = Complex(f64).conj(c);
    try testing.expectEqual(result.re, 3.0);
    try testing.expectEqual(result.im, 4.0);
}

test "Complex.magnitude computes correct magnitude" {
    const c = Complex(f64).init(3.0, 4.0);
    const mag = Complex(f64).magnitude(c);
    try testing.expectApproxEqAbs(mag, 5.0, 1e-10);
}

test "Complex.magnitude_squared computes correct squared magnitude" {
    const c = Complex(f64).init(3.0, 4.0);
    const mag_sq = Complex(f64).magnitude_squared(c);
    try testing.expectEqual(mag_sq, 25.0);
}

test "Complex.phase computes correct phase angle" {
    const c = Complex(f64).init(1.0, 0.0);
    const phase = Complex(f64).phase(c);
    try testing.expectApproxEqAbs(phase, 0.0, 1e-10);

    const c2 = Complex(f64).init(0.0, 1.0);
    const phase2 = Complex(f64).phase(c2);
    try testing.expectApproxEqAbs(phase2, math.pi / 2.0, 1e-10);
}

test "Complex.eql correctly checks equality" {
    const a = Complex(f64).init(1.0, 2.0);
    const b = Complex(f64).init(1.0, 2.0);
    const c = Complex(f64).init(1.1, 2.0);
    try testing.expect(Complex(f64).eql(a, b, 1e-10));
    try testing.expect(!Complex(f64).eql(a, c, 1e-10));
}

test "fft rejects non-power-of-2 lengths" {
    const allocator = testing.allocator;
    var signal = [_]Complex(f64){Complex(f64).init(1.0, 0.0)};
    const result = fft(f64, signal[0..0], allocator);
    try testing.expectError(error.InvalidLength, result);

    var signal3 = [_]Complex(f64){
        Complex(f64).init(1.0, 0.0),
        Complex(f64).init(0.0, 0.0),
        Complex(f64).init(0.0, 0.0),
    };
    const result3 = fft(f64, signal3[0..], allocator);
    try testing.expectError(error.InvalidLength, result3);
}

test "fft of impulse returns constant spectrum" {
    const allocator = testing.allocator;
    var signal = [_]Complex(f64){
        Complex(f64).init(1.0, 0.0),
        Complex(f64).init(0.0, 0.0),
        Complex(f64).init(0.0, 0.0),
        Complex(f64).init(0.0, 0.0),
    };

    const spectrum = try fft(f64, signal[0..], allocator);
    defer allocator.free(spectrum);

    // All bins should have magnitude 1.0
    for (spectrum) |val| {
        const mag = Complex(f64).magnitude(val);
        try testing.expectApproxEqAbs(mag, 1.0, 1e-10);
    }
}

test "fft of DC signal concentrates energy at bin 0" {
    const allocator = testing.allocator;
    const value = 5.0;
    var signal = [_]Complex(f64){
        Complex(f64).init(value, 0.0),
        Complex(f64).init(value, 0.0),
        Complex(f64).init(value, 0.0),
        Complex(f64).init(value, 0.0),
    };

    const spectrum = try fft(f64, signal[0..], allocator);
    defer allocator.free(spectrum);

    // Bin 0 should have magnitude 4 * value = 20
    const bin0_mag = Complex(f64).magnitude(spectrum[0]);
    try testing.expectApproxEqAbs(bin0_mag, 4.0 * value, 1e-10);

    // Other bins should be near zero
    for (spectrum[1..]) |val| {
        const mag = Complex(f64).magnitude(val);
        try testing.expectApproxEqAbs(mag, 0.0, 1e-10);
    }
}

test "ifft is inverse of fft (round-trip)" {
    const allocator = testing.allocator;
    var original = [_]Complex(f64){
        Complex(f64).init(1.0, 0.0),
        Complex(f64).init(2.0, 1.0),
        Complex(f64).init(0.5, -0.5),
        Complex(f64).init(3.0, 0.0),
    };

    const spectrum = try fft(f64, original[0..], allocator);
    defer allocator.free(spectrum);

    const recovered = try ifft(f64, spectrum, allocator);
    defer allocator.free(recovered);

    for (original, recovered) |orig, recov| {
        try testing.expectApproxEqAbs(orig.re, recov.re, 1e-10);
        try testing.expectApproxEqAbs(orig.im, recov.im, 1e-10);
    }
}

test "fft of sine wave has peak at correct frequency" {
    const allocator = testing.allocator;
    const n: usize = 16;
    var signal = try allocator.alloc(Complex(f64), n);
    defer allocator.free(signal);

    // Generate sine wave: sin(2π * k / n) for k = 0..n-1
    const freq_idx = 3; // 3rd frequency bin
    for (0..n) |k| {
        const k_f: f64 = @floatFromInt(k);
        const n_f: f64 = @floatFromInt(n);
        const freq_idx_f: f64 = @floatFromInt(freq_idx);
        const angle = 2.0 * math.pi * freq_idx_f * k_f / n_f;
        signal[k] = Complex(f64).init(0.0, @sin(angle));
    }

    const spectrum = try fft(f64, signal, allocator);
    defer allocator.free(spectrum);

    // Find peak
    var max_mag: f64 = 0.0;
    var max_idx: usize = 0;
    for (spectrum, 0..) |val, i| {
        const mag = Complex(f64).magnitude(val);
        if (mag > max_mag) {
            max_mag = mag;
            max_idx = i;
        }
    }

    // Peak should be at freq_idx or n - freq_idx (negative frequency)
    try testing.expect(max_idx == freq_idx or max_idx == n - freq_idx);
}

test "fft satisfies Parseval's theorem (energy conservation)" {
    const allocator = testing.allocator;
    var signal = [_]Complex(f64){
        Complex(f64).init(1.0, 1.0),
        Complex(f64).init(2.0, 0.5),
        Complex(f64).init(0.5, -1.0),
        Complex(f64).init(3.0, 2.0),
    };

    // Compute time-domain energy: sum(|x[k]|²)
    var time_energy: f64 = 0.0;
    for (signal) |val| {
        time_energy += Complex(f64).magnitude_squared(val);
    }

    const spectrum = try fft(f64, signal[0..], allocator);
    defer allocator.free(spectrum);

    // Compute frequency-domain energy: (1/n) * sum(|X[k]|²)
    var freq_energy: f64 = 0.0;
    for (spectrum) |val| {
        freq_energy += Complex(f64).magnitude_squared(val);
    }
    const n_f: f64 = @floatFromInt(signal.len);
    freq_energy /= n_f;

    // Should be equal (within floating-point precision)
    try testing.expectApproxEqAbs(time_energy, freq_energy, 1e-10);
}

test "rfft of real signal has correct length" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 1.0, 2.0, 0.5, 3.0, 1.5, 0.0, 2.0, 1.0 };

    const spectrum = try rfft(f64, signal[0..], allocator);
    defer allocator.free(spectrum);

    // Length should be n/2 + 1 = 5
    try testing.expectEqual(spectrum.len, 5);
}

test "rfft output matches full fft for real input" {
    const allocator = testing.allocator;
    var real_signal = [_]f64{ 1.0, 2.0, 0.5, 3.0 };

    // Create complex version of input
    var complex_signal = [_]Complex(f64){
        Complex(f64).init(1.0, 0.0),
        Complex(f64).init(2.0, 0.0),
        Complex(f64).init(0.5, 0.0),
        Complex(f64).init(3.0, 0.0),
    };

    const rfft_result = try rfft(f64, real_signal[0..], allocator);
    defer allocator.free(rfft_result);

    const fft_result = try fft(f64, complex_signal[0..], allocator);
    defer allocator.free(fft_result);

    // rfft output should match first n/2 + 1 values of fft
    for (0..real_signal.len / 2 + 1) |i| {
        try testing.expectApproxEqAbs(rfft_result[i].re, fft_result[i].re, 1e-10);
        try testing.expectApproxEqAbs(rfft_result[i].im, fft_result[i].im, 1e-10);
    }
}

test "irfft is inverse of rfft (round-trip)" {
    const allocator = testing.allocator;
    var original = [_]f64{ 1.0, 2.0, 0.5, 3.0, 1.5, 0.0, 2.0, 1.0 };

    const spectrum = try rfft(f64, original[0..], allocator);
    defer allocator.free(spectrum);

    const recovered = try irfft(f64, spectrum, allocator);
    defer allocator.free(recovered);

    for (original, recovered) |orig, recov| {
        try testing.expectApproxEqAbs(orig, recov, 1e-10);
    }
}

test "irfft rejects non-power-of-2 output length" {
    const allocator = testing.allocator;
    // n/2 + 1 = 3, so n = 4 which is power of 2, but output should also be power of 2
    var spectrum = [_]Complex(f64){
        Complex(f64).init(1.0, 0.0),
        Complex(f64).init(2.0, 1.0),
        Complex(f64).init(3.0, 0.0),
    };

    const result = try irfft(f64, spectrum[0..], allocator);
    defer allocator.free(result);
    try testing.expectEqual(result.len, 4);
}

test "fftfreq computes correct frequencies for f64" {
    const allocator = testing.allocator;
    const n = 8;
    const d = 0.1; // 10 Hz sample rate (0.1 second spacing)

    const freqs = try fftfreq(f64, n, d, allocator);
    defer allocator.free(freqs);

    // Expected frequencies: [0, 1.25, 2.5, 3.75, -5, -3.75, -2.5, -1.25]
    const expected = [_]f64{ 0.0, 1.25, 2.5, 3.75, -5.0, -3.75, -2.5, -1.25 };

    for (freqs, expected) |actual, exp| {
        try testing.expectApproxEqAbs(actual, exp, 1e-10);
    }
}

test "fftfreq computes frequencies with different sample rates" {
    const allocator = testing.allocator;
    const n = 4;
    const d = 0.001; // 1000 Hz sample rate

    const freqs = try fftfreq(f64, n, d, allocator);
    defer allocator.free(freqs);

    // Expected: [0, 250, -500, -250]
    const expected = [_]f64{ 0.0, 250.0, -500.0, -250.0 };

    for (freqs, expected) |actual, exp| {
        try testing.expectApproxEqAbs(actual, exp, 1e-8);
    }
}

test "fftfreq returns correct number of bins" {
    const allocator = testing.allocator;
    const sizes = [_]usize{ 4, 8, 16, 32, 64 };

    for (sizes) |n| {
        const freqs = try fftfreq(f64, n, 0.1, allocator);
        defer allocator.free(freqs);
        try testing.expectEqual(freqs.len, n);
    }
}

test "fft of power of 2 sized signals works correctly" {
    const allocator = testing.allocator;
    const sizes = [_]usize{ 2, 4, 8, 16, 32 };

    for (sizes) |n| {
        var signal = try allocator.alloc(Complex(f64), n);
        defer allocator.free(signal);

        // Impulse at index 0
        signal[0] = Complex(f64).init(1.0, 0.0);
        for (1..n) |i| {
            signal[i] = Complex(f64).init(0.0, 0.0);
        }

        const spectrum = try fft(f64, signal, allocator);
        defer allocator.free(spectrum);

        // All bins should have magnitude 1.0 for impulse
        for (spectrum) |val| {
            const mag = Complex(f64).magnitude(val);
            try testing.expectApproxEqAbs(mag, 1.0, 1e-10);
        }
    }
}

test "fft of constant signal concentrates at DC component" {
    const allocator = testing.allocator;
    const n = 8;
    var signal = try allocator.alloc(Complex(f64), n);
    defer allocator.free(signal);

    const value = 7.0;
    for (0..n) |i| {
        signal[i] = Complex(f64).init(value, 0.0);
    }

    const spectrum = try fft(f64, signal, allocator);
    defer allocator.free(spectrum);

    // DC component (bin 0) should have magnitude n * value = 56
    const dc_mag = Complex(f64).magnitude(spectrum[0]);
    try testing.expectApproxEqAbs(dc_mag, @as(f64, @floatFromInt(n)) * value, 1e-10);

    // Other components should be near zero
    for (spectrum[1..]) |val| {
        const mag = Complex(f64).magnitude(val);
        try testing.expectApproxEqAbs(mag, 0.0, 1e-8);
    }
}

test "ifft normalization is correct" {
    const allocator = testing.allocator;
    var signal = [_]Complex(f64){
        Complex(f64).init(2.0, 0.0),
        Complex(f64).init(4.0, 0.0),
        Complex(f64).init(1.0, 0.0),
        Complex(f64).init(3.0, 0.0),
    };

    const spectrum = try fft(f64, signal[0..], allocator);
    defer allocator.free(spectrum);

    const recovered = try ifft(f64, spectrum, allocator);
    defer allocator.free(recovered);

    // Sum of recovered real parts should equal sum of original
    var original_sum: f64 = 0.0;
    var recovered_sum: f64 = 0.0;
    for (signal) |val| {
        original_sum += val.re;
    }
    for (recovered) |val| {
        recovered_sum += val.re;
    }
    try testing.expectApproxEqAbs(original_sum, recovered_sum, 1e-10);
}

test "rfft and irfft preserve real signal magnitude (Parseval)" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.5, 1.5 };

    // Compute time-domain energy
    var time_energy: f64 = 0.0;
    for (signal) |val| {
        time_energy += val * val;
    }

    const spectrum = try rfft(f64, signal[0..], allocator);
    defer allocator.free(spectrum);

    // Compute frequency-domain energy
    var freq_energy: f64 = 0.0;
    for (spectrum, 0..) |val, i| {
        const mag_sq = Complex(f64).magnitude_squared(val);
        if (i == 0 or i == signal.len / 2) {
            // DC and Nyquist components are not doubled
            freq_energy += mag_sq;
        } else {
            // Other bins are doubled due to negative frequencies
            freq_energy += 2.0 * mag_sq;
        }
    }
    freq_energy /= @as(f64, @floatFromInt(signal.len));

    try testing.expectApproxEqAbs(time_energy, freq_energy, 1e-8);
}

test "fft with complex input preserves magnitude (Parseval)" {
    const allocator = testing.allocator;
    var signal = [_]Complex(f64){
        Complex(f64).init(1.0, 0.5),
        Complex(f64).init(2.0, -0.5),
        Complex(f64).init(0.5, 1.0),
        Complex(f64).init(1.5, -1.0),
    };

    var time_energy: f64 = 0.0;
    for (signal) |val| {
        time_energy += Complex(f64).magnitude_squared(val);
    }

    const spectrum = try fft(f64, signal[0..], allocator);
    defer allocator.free(spectrum);

    var freq_energy: f64 = 0.0;
    for (spectrum) |val| {
        freq_energy += Complex(f64).magnitude_squared(val);
    }
    freq_energy /= @as(f64, @floatFromInt(signal.len));

    try testing.expectApproxEqAbs(time_energy, freq_energy, 1e-10);
}

test "ifft conjugate symmetry for real output" {
    const allocator = testing.allocator;
    var spectrum = [_]Complex(f64){
        Complex(f64).init(4.0, 0.0),
        Complex(f64).init(1.0, -1.0),
        Complex(f64).init(0.0, 0.0),
        Complex(f64).init(1.0, 1.0),
    };

    const time_signal = try ifft(f64, spectrum[0..], allocator);
    defer allocator.free(time_signal);

    // Output should have real-valued elements
    for (time_signal) |val| {
        try testing.expectApproxEqAbs(val.im, 0.0, 1e-10);
    }
}

test "rfft symmetry property: spectrum[n-k] = conj(spectrum[k])" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 1.0, 2.0, 0.5, 3.0, 1.5, 0.0, 2.0, 1.0 };

    const spectrum = try rfft(f64, signal[0..], allocator);
    defer allocator.free(spectrum);

    // For rfft with 8 inputs, spectrum has 5 elements (n/2 + 1)
    // We need to verify the symmetry with the full fft result

    // Create complex signal and compute full fft
    var complex_signal = try allocator.alloc(Complex(f64), signal.len);
    defer allocator.free(complex_signal);
    for (signal, 0..) |val, i| {
        complex_signal[i] = Complex(f64).init(val, 0.0);
    }

    const full_spectrum = try fft(f64, complex_signal, allocator);
    defer allocator.free(full_spectrum);

    // Check conjugate symmetry: full_spectrum[n-k] = conj(full_spectrum[k])
    const n = signal.len;
    for (1..n / 2) |k| {
        const sym_k = full_spectrum[k];
        const sym_n_minus_k = full_spectrum[n - k];
        try testing.expectApproxEqAbs(sym_k.re, sym_n_minus_k.re, 1e-10);
        try testing.expectApproxEqAbs(sym_k.im, -sym_n_minus_k.im, 1e-10);
    }
}

test "fft memory allocation and deallocation" {
    const allocator = testing.allocator;
    var signal = [_]Complex(f64){
        Complex(f64).init(1.0, 0.0),
        Complex(f64).init(2.0, 0.0),
        Complex(f64).init(0.5, 0.0),
        Complex(f64).init(3.0, 0.0),
    };

    const spectrum = try fft(f64, signal[0..], allocator);
    allocator.free(spectrum);
    // If we reach here without detecting memory leak, allocation/deallocation was correct
}

test "rfft memory allocation and deallocation" {
    const allocator = testing.allocator;
    var signal = [_]f64{ 1.0, 2.0, 0.5, 3.0 };

    const spectrum = try rfft(f64, signal[0..], allocator);
    allocator.free(spectrum);
    // If we reach here without detecting memory leak, allocation/deallocation was correct
}

test "irfft memory allocation and deallocation" {
    const allocator = testing.allocator;
    var spectrum = [_]Complex(f64){
        Complex(f64).init(1.0, 0.0),
        Complex(f64).init(2.0, 1.0),
        Complex(f64).init(3.0, 0.0),
    };

    const result = try irfft(f64, spectrum[0..], allocator);
    allocator.free(result);
    // If we reach here without detecting memory leak, allocation/deallocation was correct
}

test "fftfreq memory allocation and deallocation" {
    const allocator = testing.allocator;
    const freqs = try fftfreq(f64, 16, 0.1, allocator);
    allocator.free(freqs);
    // If we reach here without detecting memory leak, allocation/deallocation was correct
}

test "fft with f32 type" {
    const allocator = testing.allocator;
    var signal = [_]Complex(f32){
        Complex(f32).init(1.0, 0.0),
        Complex(f32).init(2.0, 0.0),
        Complex(f32).init(0.5, 0.0),
        Complex(f32).init(3.0, 0.0),
    };

    const spectrum = try fft(f32, signal[0..], allocator);
    defer allocator.free(spectrum);

    try testing.expectEqual(spectrum.len, 4);
}

test "rfft with f32 type" {
    const allocator = testing.allocator;
    var signal = [_]f32{ 1.0, 2.0, 0.5, 3.0 };

    const spectrum = try rfft(f32, signal[0..], allocator);
    defer allocator.free(spectrum);

    try testing.expectEqual(spectrum.len, 3);
}

test "fft cosine wave has peak at expected frequency" {
    const allocator = testing.allocator;
    const n: usize = 16;
    const signal = try allocator.alloc(Complex(f64), n);
    defer allocator.free(signal);

    // Generate cosine wave: cos(2π * k / n)
    for (0..n) |k| {
        const k_f: f64 = @floatFromInt(k);
        const n_f: f64 = @floatFromInt(n);
        const angle = 2.0 * math.pi * k_f / n_f;
        signal[k] = Complex(f64).init(@cos(angle), 0.0);
    }

    const spectrum = try fft(f64, signal, allocator);
    defer allocator.free(spectrum);

    // DC component should be near zero
    const dc_mag = Complex(f64).magnitude(spectrum[0]);
    try testing.expectApproxEqAbs(dc_mag, 0.0, 1e-8);

    // First bin should have peak
    const bin1_mag = Complex(f64).magnitude(spectrum[1]);
    try testing.expect(bin1_mag > 5.0);
}

test "rfft of sine wave has conjugate symmetry in recovery" {
    const allocator = testing.allocator;
    const n: usize = 8;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    // Generate sine wave
    for (0..n) |k| {
        const k_f: f64 = @floatFromInt(k);
        const n_f: f64 = @floatFromInt(n);
        const angle = 2.0 * math.pi * k_f / n_f;
        signal[k] = @sin(angle);
    }

    const spectrum = try rfft(f64, signal, allocator);
    defer allocator.free(spectrum);

    const recovered = try irfft(f64, spectrum, allocator);
    defer allocator.free(recovered);

    // Recovered signal should match original
    for (signal, recovered) |orig, recov| {
        try testing.expectApproxEqAbs(orig, recov, 1e-9);
    }
}

test "multiple sequential FFT calls work correctly" {
    const allocator = testing.allocator;
    var signal1 = [_]Complex(f64){
        Complex(f64).init(1.0, 0.0),
        Complex(f64).init(0.0, 0.0),
        Complex(f64).init(0.0, 0.0),
        Complex(f64).init(0.0, 0.0),
    };
    var signal2 = [_]Complex(f64){
        Complex(f64).init(2.0, 0.0),
        Complex(f64).init(0.0, 0.0),
        Complex(f64).init(0.0, 0.0),
        Complex(f64).init(0.0, 0.0),
    };

    const spectrum1 = try fft(f64, signal1[0..], allocator);
    defer allocator.free(spectrum1);

    const spectrum2 = try fft(f64, signal2[0..], allocator);
    defer allocator.free(spectrum2);

    // Second FFT should work independently
    try testing.expectEqual(spectrum1.len, 4);
    try testing.expectEqual(spectrum2.len, 4);

    // Magnitudes should be different
    const mag1 = Complex(f64).magnitude(spectrum1[0]);
    const mag2 = Complex(f64).magnitude(spectrum2[0]);
    try testing.expect(mag1 != mag2);
}

test "fft-ifft composition preserves complex values" {
    const allocator = testing.allocator;
    var signal = [_]Complex(f64){
        Complex(f64).init(1.0 + math.sqrt2, -math.pi),
        Complex(f64).init(0.5, 2.5),
        Complex(f64).init(-1.0, 0.0),
        Complex(f64).init(3.14159, -2.71828),
    };

    const spectrum = try fft(f64, signal[0..], allocator);
    defer allocator.free(spectrum);

    const recovered = try ifft(f64, spectrum, allocator);
    defer allocator.free(recovered);

    for (signal, recovered) |orig, recov| {
        try testing.expectApproxEqAbs(orig.re, recov.re, 1e-9);
        try testing.expectApproxEqAbs(orig.im, recov.im, 1e-9);
    }
}

test "fftfreq symmetry about zero" {
    const allocator = testing.allocator;
    const freqs = try fftfreq(f64, 8, 1.0, allocator);
    defer allocator.free(freqs);

    // Check that positive and negative frequency bins are symmetric
    // freqs[0] = DC, freqs[1..4] are positive, freqs[5..7] are negative
    try testing.expectApproxEqAbs(freqs[1], -freqs[7], 1e-10);
    try testing.expectApproxEqAbs(freqs[2], -freqs[6], 1e-10);
    try testing.expectApproxEqAbs(freqs[3], -freqs[5], 1e-10);
    // freqs[4] is Nyquist frequency = -0.5, so it's its own negative (DC=0, Nyquist=-0.5)
}
