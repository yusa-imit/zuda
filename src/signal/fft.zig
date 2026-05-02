const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Complex number type for FFT operations
pub fn Complex(comptime T: type) type {
    return struct {
        real: T,
        imag: T,

        const Self = @This();

        pub fn init(real: T, imag: T) Self {
            return .{ .real = real, .imag = imag };
        }

        pub fn add(self: Self, other: Self) Self {
            return .{
                .real = self.real + other.real,
                .imag = self.imag + other.imag,
            };
        }

        pub fn sub(self: Self, other: Self) Self {
            return .{
                .real = self.real - other.real,
                .imag = self.imag - other.imag,
            };
        }

        pub fn mul(self: Self, other: Self) Self {
            return .{
                .real = self.real * other.real - self.imag * other.imag,
                .imag = self.real * other.imag + self.imag * other.real,
            };
        }

        pub fn magnitude(self: Self) T {
            return @sqrt(self.real * self.real + self.imag * self.imag);
        }

        pub fn magnitude_squared(self: Self) T {
            return self.real * self.real + self.imag * self.imag;
        }

        pub fn phase(self: Self) T {
            return math.atan2(self.imag, self.real);
        }

        pub fn eql(self: Self, other: Self, epsilon: T) bool {
            return @abs(self.real - other.real) <= epsilon and
                @abs(self.imag - other.imag) <= epsilon;
        }
    };
}

pub const FFTError = error{
    InvalidSize,
    NotPowerOfTwo,
};

/// Compute the Fast Fourier Transform using Cooley-Tukey algorithm
///
/// Time: O(n log n) where n is the input length
/// Space: O(n) for output array
///
/// The input size must be a power of 2.
/// Returns frequency domain representation of the input signal.
pub fn fft(comptime T: type, allocator: Allocator, input: []const Complex(T)) ![]Complex(T) {
    const n = input.len;

    if (n == 0) return error.InvalidSize;
    if (!isPowerOfTwo(n)) return error.NotPowerOfTwo;

    var output = try allocator.alloc(Complex(T), n);
    errdefer allocator.free(output);

    // Copy input to output
    @memcpy(output, input);

    // Bit-reversal permutation
    bitReversePermutation(Complex(T), output);

    // Cooley-Tukey FFT
    var size: usize = 2;
    while (size <= n) : (size *= 2) {
        const half_size = size / 2;
        const theta = -2.0 * math.pi / @as(T, @floatFromInt(size));

        var k: usize = 0;
        while (k < n) : (k += size) {
            var j: usize = 0;
            while (j < half_size) : (j += 1) {
                const angle = theta * @as(T, @floatFromInt(j));
                const twiddle = Complex(T).init(@cos(angle), @sin(angle));

                const t = twiddle.mul(output[k + j + half_size]);
                const u = output[k + j];

                output[k + j] = u.add(t);
                output[k + j + half_size] = u.sub(t);
            }
        }
    }

    return output;
}

/// Compute the Inverse Fast Fourier Transform
///
/// Time: O(n log n) where n is the input length
/// Space: O(n) for output array
///
/// The input size must be a power of 2.
/// Returns time domain representation of the frequency spectrum.
pub fn ifft(comptime T: type, allocator: Allocator, input: []const Complex(T)) ![]Complex(T) {
    const n = input.len;

    if (n == 0) return error.InvalidSize;
    if (!isPowerOfTwo(n)) return error.NotPowerOfTwo;

    // Conjugate input
    var conjugated = try allocator.alloc(Complex(T), n);
    defer allocator.free(conjugated);

    for (input, 0..) |val, i| {
        conjugated[i] = Complex(T).init(val.real, -val.imag);
    }

    // Compute FFT of conjugated input
    const result = try fft(T, allocator, conjugated);
    errdefer allocator.free(result);

    // Conjugate and scale result
    const scale = 1.0 / @as(T, @floatFromInt(n));
    for (result) |*val| {
        val.real *= scale;
        val.imag = -val.imag * scale;
    }

    return result;
}

/// Compute FFT for real-valued input
///
/// Time: O(n log n) where n is the input length
/// Space: O(n) for output array
///
/// The input size must be a power of 2.
/// Returns n/2 + 1 complex frequencies (exploiting conjugate symmetry).
pub fn rfft(comptime T: type, allocator: Allocator, input: []const T) ![]Complex(T) {
    const n = input.len;

    if (n == 0) return error.InvalidSize;
    if (!isPowerOfTwo(n)) return error.NotPowerOfTwo;

    // Convert real input to complex
    var complex_input = try allocator.alloc(Complex(T), n);
    defer allocator.free(complex_input);

    for (input, 0..) |val, i| {
        complex_input[i] = Complex(T).init(val, 0.0);
    }

    // Compute full FFT
    const full_fft = try fft(T, allocator, complex_input);
    errdefer allocator.free(full_fft);

    // Return only first half (due to conjugate symmetry)
    const output_len = n / 2 + 1;
    const output = try allocator.alloc(Complex(T), output_len);
    @memcpy(output, full_fft[0..output_len]);

    allocator.free(full_fft);
    return output;
}

/// Compute inverse FFT for real-valued output
///
/// Time: O(n log n)
/// Space: O(n) for output array
///
/// Input should be n/2 + 1 complex frequencies from rfft.
/// Returns n real values.
pub fn irfft(comptime T: type, allocator: Allocator, input: []const Complex(T), n: usize) ![]T {
    if (n == 0) return error.InvalidSize;
    if (!isPowerOfTwo(n)) return error.NotPowerOfTwo;
    if (input.len != n / 2 + 1) return error.InvalidSize;

    // Reconstruct full spectrum using conjugate symmetry
    var full_spectrum = try allocator.alloc(Complex(T), n);
    defer allocator.free(full_spectrum);

    // Copy positive frequencies
    @memcpy(full_spectrum[0..input.len], input);

    // Fill negative frequencies (conjugate symmetry)
    var i: usize = 1;
    while (i < n / 2) : (i += 1) {
        full_spectrum[n - i] = Complex(T).init(input[i].real, -input[i].imag);
    }

    // Compute inverse FFT
    const complex_output = try ifft(T, allocator, full_spectrum);
    defer allocator.free(complex_output);

    // Extract real part
    var output = try allocator.alloc(T, n);
    for (complex_output, 0..) |val, idx| {
        output[idx] = val.real;
    }

    return output;
}

/// Compute power spectral density (magnitude squared of FFT)
///
/// Time: O(n log n)
/// Space: O(n) for output array
///
/// Returns |FFT(x)|^2 for each frequency bin.
pub fn powerSpectrum(comptime T: type, allocator: Allocator, input: []const T) ![]T {
    const freq = try rfft(T, allocator, input);
    defer allocator.free(freq);

    var power = try allocator.alloc(T, freq.len);
    for (freq, 0..) |val, i| {
        const mag = val.magnitude();
        power[i] = mag * mag;
    }

    return power;
}

/// Compute frequency bins for FFT output
///
/// Time: O(n)
/// Space: O(n) for output array
///
/// Returns array of frequencies corresponding to FFT bins.
/// sample_rate: samples per second
pub fn fftFreq(comptime T: type, allocator: Allocator, n: usize, sample_rate: T) ![]T {
    if (n == 0) return error.InvalidSize;

    var freqs = try allocator.alloc(T, n);
    const spacing = sample_rate / @as(T, @floatFromInt(n));

    for (0..n) |i| {
        if (i <= (n - 1) / 2) {
            // Positive frequencies: 0, 1, ..., floor((n-1)/2)
            freqs[i] = @as(T, @floatFromInt(i)) * spacing;
        } else {
            // Negative frequencies (including Nyquist for even n)
            freqs[i] = @as(T, @floatFromInt(@as(i64, @intCast(i)) - @as(i64, @intCast(n)))) * spacing;
        }
    }

    return freqs;
}

/// Compute frequency bins for rfft output
///
/// Time: O(n)
/// Space: O(n) for output array
///
/// Returns array of positive frequencies for real FFT.
pub fn rfftFreq(comptime T: type, allocator: Allocator, n: usize, sample_rate: T) ![]T {
    if (n == 0) return error.InvalidSize;

    const output_len = n / 2 + 1;
    var freqs = try allocator.alloc(T, output_len);
    const spacing = sample_rate / @as(T, @floatFromInt(n));

    for (0..output_len) |i| {
        freqs[i] = @as(T, @floatFromInt(i)) * spacing;
    }

    return freqs;
}

// Helper functions

fn isPowerOfTwo(n: usize) bool {
    return n > 0 and (n & (n - 1)) == 0;
}

fn bitReversePermutation(comptime T: type, data: []T) void {
    const n = data.len;
    const bits: u8 = @intCast(@ctz(@as(usize, n)));

    var i: usize = 0;
    while (i < n) : (i += 1) {
        const j = reverseBits(i, bits);
        if (i < j) {
            const temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
}

fn reverseBits(x: usize, num_bits: u8) usize {
    var result: usize = 0;
    var val = x;
    var bits = num_bits;

    while (bits > 0) : (bits -= 1) {
        result <<= 1;
        result |= val & 1;
        val >>= 1;
    }

    return result;
}

// Tests

test "Complex number operations" {
    const C = Complex(f64);

    const a = C.init(3.0, 4.0);
    const b = C.init(1.0, 2.0);

    const sum = a.add(b);
    try testing.expectApproxEqAbs(4.0, sum.real, 1e-10);
    try testing.expectApproxEqAbs(6.0, sum.imag, 1e-10);

    const diff = a.sub(b);
    try testing.expectApproxEqAbs(2.0, diff.real, 1e-10);
    try testing.expectApproxEqAbs(2.0, diff.imag, 1e-10);

    const prod = a.mul(b);
    try testing.expectApproxEqAbs(-5.0, prod.real, 1e-10);
    try testing.expectApproxEqAbs(10.0, prod.imag, 1e-10);

    const mag = a.magnitude();
    try testing.expectApproxEqAbs(5.0, mag, 1e-10);
}

test "FFT: impulse signal" {
    const allocator = testing.allocator;
    const C = Complex(f64);

    // Impulse at t=0 should have flat spectrum
    var input = [_]C{
        C.init(1.0, 0.0),
        C.init(0.0, 0.0),
        C.init(0.0, 0.0),
        C.init(0.0, 0.0),
    };

    const output = try fft(f64, allocator, &input);
    defer allocator.free(output);

    // All frequency bins should have magnitude 1.0
    for (output) |val| {
        try testing.expectApproxEqAbs(1.0, val.magnitude(), 1e-10);
    }
}

test "FFT: DC signal" {
    const allocator = testing.allocator;
    const C = Complex(f64);

    // Constant signal should have all energy at DC (bin 0)
    var input = [_]C{
        C.init(1.0, 0.0),
        C.init(1.0, 0.0),
        C.init(1.0, 0.0),
        C.init(1.0, 0.0),
    };

    const output = try fft(f64, allocator, &input);
    defer allocator.free(output);

    // DC component should be 4.0
    try testing.expectApproxEqAbs(4.0, output[0].real, 1e-10);
    try testing.expectApproxEqAbs(0.0, output[0].imag, 1e-10);

    // Other bins should be zero
    for (output[1..]) |val| {
        try testing.expectApproxEqAbs(0.0, val.magnitude(), 1e-10);
    }
}

test "FFT/IFFT roundtrip" {
    const allocator = testing.allocator;
    const C = Complex(f64);

    var input = [_]C{
        C.init(1.0, 0.5),
        C.init(2.0, -1.0),
        C.init(3.0, 0.0),
        C.init(0.5, 2.0),
    };

    const freq = try fft(f64, allocator, &input);
    defer allocator.free(freq);

    const reconstructed = try ifft(f64, allocator, freq);
    defer allocator.free(reconstructed);

    for (input, reconstructed) |orig, recon| {
        try testing.expectApproxEqAbs(orig.real, recon.real, 1e-10);
        try testing.expectApproxEqAbs(orig.imag, recon.imag, 1e-10);
    }
}

test "RFFT: real sine wave" {
    const allocator = testing.allocator;

    // 1 Hz sine wave, sampled at 8 Hz for 1 second
    const n = 8;
    var input: [n]f64 = undefined;
    for (0..n) |i| {
        const t = @as(f64, @floatFromInt(i)) / 8.0;
        input[i] = @sin(2.0 * math.pi * 1.0 * t);
    }

    const output = try rfft(f64, allocator, &input);
    defer allocator.free(output);

    try testing.expectEqual(5, output.len); // n/2 + 1 = 5

    // Peak should be at bin 1 (1 Hz)
    const mag1 = output[1].magnitude();
    try testing.expect(mag1 > 3.0); // Significant energy at 1 Hz

    // DC should be near zero
    try testing.expectApproxEqAbs(0.0, output[0].magnitude(), 1e-10);
}

test "RFFT/IRFFT roundtrip" {
    const allocator = testing.allocator;

    var input = [_]f64{ 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0 };

    const freq = try rfft(f64, allocator, &input);
    defer allocator.free(freq);

    const reconstructed = try irfft(f64, allocator, freq, input.len);
    defer allocator.free(reconstructed);

    for (input, reconstructed) |orig, recon| {
        try testing.expectApproxEqAbs(orig, recon, 1e-10);
    }
}

test "FFT: power of two requirement" {
    const allocator = testing.allocator;
    const C = Complex(f64);

    var input = [_]C{
        C.init(1.0, 0.0),
        C.init(2.0, 0.0),
        C.init(3.0, 0.0),
    };

    try testing.expectError(error.NotPowerOfTwo, fft(f64, allocator, &input));
}

test "FFT: empty input" {
    const allocator = testing.allocator;
    const C = Complex(f64);

    var input: [0]C = .{};

    try testing.expectError(error.InvalidSize, fft(f64, allocator, &input));
}

test "Power spectrum" {
    const allocator = testing.allocator;

    // Simple signal
    var input = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

    const power = try powerSpectrum(f64, allocator, &input);
    defer allocator.free(power);

    try testing.expectEqual(5, power.len); // n/2 + 1

    // Power should be non-negative
    for (power) |p| {
        try testing.expect(p >= 0.0);
    }
}

test "FFT frequency bins" {
    const allocator = testing.allocator;

    const freqs = try fftFreq(f64, allocator, 8, 8.0);
    defer allocator.free(freqs);

    try testing.expectEqual(8, freqs.len);
    try testing.expectApproxEqAbs(0.0, freqs[0], 1e-10);
    try testing.expectApproxEqAbs(1.0, freqs[1], 1e-10);
    try testing.expectApproxEqAbs(-4.0, freqs[4], 1e-10); // Nyquist
}

test "RFFT frequency bins" {
    const allocator = testing.allocator;

    const freqs = try rfftFreq(f64, allocator, 8, 8.0);
    defer allocator.free(freqs);

    try testing.expectEqual(5, freqs.len); // n/2 + 1
    try testing.expectApproxEqAbs(0.0, freqs[0], 1e-10);
    try testing.expectApproxEqAbs(1.0, freqs[1], 1e-10);
    try testing.expectApproxEqAbs(4.0, freqs[4], 1e-10); // Nyquist
}

test "FFT: f32 precision" {
    const allocator = testing.allocator;
    const C = Complex(f32);

    var input = [_]C{
        C.init(1.0, 0.0),
        C.init(0.0, 0.0),
        C.init(0.0, 0.0),
        C.init(0.0, 0.0),
    };

    const output = try fft(f32, allocator, &input);
    defer allocator.free(output);

    for (output) |val| {
        try testing.expectApproxEqAbs(@as(f32, 1.0), val.magnitude(), 1e-6);
    }
}

test "FFT: memory safety" {
    const allocator = testing.allocator;
    const C = Complex(f64);

    var input = [_]C{
        C.init(1.0, 0.5),
        C.init(2.0, -1.0),
        C.init(3.0, 0.0),
        C.init(0.5, 2.0),
        C.init(1.5, 1.0),
        C.init(2.5, -0.5),
        C.init(0.0, 1.5),
        C.init(3.5, -1.5),
    };

    // Run 10 iterations to detect memory leaks
    for (0..10) |_| {
        const freq = try fft(f64, allocator, &input);
        const reconstructed = try ifft(f64, allocator, freq);
        allocator.free(freq);
        allocator.free(reconstructed);
    }
}

test "RFFT: memory safety" {
    const allocator = testing.allocator;

    var input = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

    // Run 10 iterations to detect memory leaks
    for (0..10) |_| {
        const freq = try rfft(f64, allocator, &input);
        const reconstructed = try irfft(f64, allocator, freq, input.len);
        allocator.free(freq);
        allocator.free(reconstructed);
    }
}
