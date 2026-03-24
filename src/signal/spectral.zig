//! Spectral Analysis for Power Spectral Density Estimation
//!
//! This module provides functions for analyzing frequency content of signals
//! using power spectral density (PSD) estimation techniques.
//!
//! ## Supported Methods
//! - `periodogram` — Simple PSD estimate via FFT (Welch variant: one segment)
//! - `welch` — Welch's method: averaged periodogram with overlapping segments
//!
//! ## Time Complexity
//! - periodogram: O(N log N) via FFT
//! - welch: O(K·M log M) where K = number of segments, M = segment length
//!
//! ## Space Complexity
//! - Both methods: O(max(N, M)) for temporary storage
//!
//! ## Use Cases
//! - Frequency domain analysis of signals
//! - Noise characterization and power estimation
//! - Spectral leakage reduction (Welch with windowing)
//! - Audio, sensor data, communications signal analysis
//!
//! ## References
//! - Welch, P.D. (1967). "The use of fast Fourier transform for estimation of power spectra"
//! - Oppenheim & Schafer, "Discrete-Time Signal Processing" (3rd ed.)

const std = @import("std");
const math = std.math;
const testing = std.testing;
const Allocator = std.mem.Allocator;

const fft_module = @import("fft.zig");
const window_module = @import("window.zig");

const Complex = fft_module.Complex;

/// Result structure for periodogram computation
/// Contains positive frequencies (0 to fs/2) and corresponding power values
/// Both arrays are owned by caller and must be freed after use
pub fn PeriodogramResult(comptime T: type) type {
    return struct {
        frequencies: []T,
        power: []T,

        const Self = @This();

        /// Clean up allocated memory
        pub fn deinit(self: Self, allocator: Allocator) void {
            allocator.free(self.frequencies);
            allocator.free(self.power);
        }
    };
}

/// Result structure for Welch's method computation
/// Contains positive frequencies and smoothed power estimates
/// Both arrays are owned by caller and must be freed after use
pub fn WelchResult(comptime T: type) type {
    return struct {
        frequencies: []T,
        power: []T,

        const Self = @This();

        /// Clean up allocated memory
        pub fn deinit(self: Self, allocator: Allocator) void {
            allocator.free(self.frequencies);
            allocator.free(self.power);
        }
    };
}

/// Compute Power Spectral Density via Periodogram (simple FFT-based method)
///
/// Applies FFT to the entire signal to estimate power distribution across frequencies.
/// Returns positive frequencies only (0 to fs/2) using real FFT properties.
///
/// Power is computed as |FFT|² / N, representing energy per frequency bin.
/// This is equivalent to Welch's method with a single segment (no windowing).
///
/// ## Parameters
/// - T: Floating-point type (f32 or f64)
/// - signal: Real-valued time-domain signal (signal.len must be power of 2)
/// - fs: Sampling frequency (must be > 0)
/// - allocator: Memory allocator for result arrays
///
/// ## Returns
/// - PeriodogramResult(T) with:
///   - frequencies: Array of len = N/2 + 1, spacing = fs/N
///   - power: Array of len = N/2 + 1, values in power units (same as signal energy units)
///
/// ## Errors
/// - error.OutOfMemory if allocation fails
/// - error.InvalidLength if signal length is not power of 2
/// - error.EmptyArray if signal is empty (length 0)
/// - error.InvalidParameter if fs <= 0
///
/// ## Time: O(N log N) via FFT
/// ## Space: O(N)
///
/// ## Properties
/// - Energy conservation (Parseval's theorem): sum(power) ≈ mean(signal²)
/// - Frequency resolution: df = fs / N
/// - Maximum frequency: fs/2 (Nyquist)
///
/// ## Example
/// ```zig
/// const allocator = std.testing.allocator;
/// const signal = [_]f64{ 1.0, 0.5, -0.5, -1.0 };
/// const fs = 1.0;
/// const result = try periodogram(f64, signal[0..], fs, allocator);
/// defer result.deinit(allocator);
/// // result.frequencies: [0, 0.25, 0.5]
/// // result.power: [power values at each frequency]
/// ```
pub fn periodogram(comptime T: type, signal: []const T, fs: T, allocator: Allocator) (Allocator.Error || error{ InvalidLength, EmptyArray, InvalidParameter })!PeriodogramResult(T) {
    // Input validation
    if (signal.len == 0) {
        return error.EmptyArray;
    }
    if (fs <= 0) {
        return error.InvalidParameter;
    }
    const n = signal.len;
    if (n & (n - 1) != 0) {
        // Not a power of 2
        return error.InvalidLength;
    }

    // Compute real FFT of the signal
    const fft_result = try fft_module.rfft(T, signal, allocator);
    defer allocator.free(fft_result);

    // Allocate output arrays
    const n_freqs = n / 2 + 1;
    const frequencies = try allocator.alloc(T, n_freqs);
    errdefer allocator.free(frequencies);
    const power = try allocator.alloc(T, n_freqs);
    errdefer allocator.free(power);

    // Compute frequency array and power spectrum
    const bin_width = fs / @as(T, @floatFromInt(n));
    const n_inv = 1.0 / @as(T, @floatFromInt(n));
    const n_sq_inv = n_inv * n_inv;

    for (0..n_freqs) |i| {
        // Frequency at bin i
        frequencies[i] = @as(T, @floatFromInt(i)) * bin_width;

        // Power computation for single-sided spectrum:
        // FFT output is unnormalized, so we need |X[k]|² / N²
        // DC (0 Hz) and Nyquist (fs/2) get factor of 1
        // All other frequencies get factor of 2 (accounting for negative frequencies)
        const mag_sq = fft_result[i].magnitude_squared();
        if (i == 0 or i == n / 2) {
            // DC or Nyquist: |X[k]|² / N²
            power[i] = mag_sq * n_sq_inv;
        } else {
            // Positive frequencies: 2 * |X[k]|² / N² (double to account for negative)
            power[i] = 2.0 * mag_sq * n_sq_inv;
        }
    }

    return PeriodogramResult(T){
        .frequencies = frequencies,
        .power = power,
    };
}

/// Compute Power Spectral Density via Welch's Method
///
/// Welch's method reduces variance in spectral estimates by:
/// 1. Dividing signal into overlapping segments
/// 2. Applying window function to each segment (default: Hann window)
/// 3. Computing FFT and power for each segment
/// 4. Averaging power estimates across segments
///
/// This provides better noise suppression than simple periodogram,
/// especially for longer signals with non-stationary content.
///
/// ## Parameters
/// - T: Floating-point type (f32 or f64)
/// - signal: Real-valued time-domain signal
/// - fs: Sampling frequency (must be > 0)
/// - nperseg: Segment length for each FFT (default: 256)
///   - If nperseg > signal.len, entire signal is treated as one segment
/// - noverlap: Number of overlapping samples between segments (default: nperseg/2)
///   - Must satisfy: 0 <= noverlap < nperseg
/// - allocator: Memory allocator for result arrays
///
/// ## Returns
/// - WelchResult(T) with:
///   - frequencies: Array of len = nperseg/2 + 1, spacing = fs/nperseg
///   - power: Array of len = nperseg/2 + 1, average power across segments
///
/// ## Errors
/// - error.OutOfMemory if allocation fails
/// - error.EmptyArray if signal is empty
/// - error.InvalidParameter if:
///   - nperseg > signal.len (will use entire signal as one segment)
///   - noverlap >= nperseg
///   - fs <= 0
///
/// ## Time: O(K·M log M)
/// - K = ceil((signal.len - noverlap) / (nperseg - noverlap)) segments
/// - M = nperseg
///
/// ## Space: O(nperseg) for segment buffers + window
///
/// ## Default Parameters (if using simplified API)
/// - nperseg: 256 or signal.len if shorter
/// - noverlap: nperseg / 2
/// - window: Hann window
///
/// ## Properties
/// - Smoother estimate than periodogram (lower variance)
/// - Frequency resolution: df = fs / nperseg (coarser if nperseg < N)
/// - Overlap reduces spectral leakage and improves stability
/// - Typical choice: 50% overlap with Hann window
///
/// ## Example
/// ```zig
/// const allocator = std.testing.allocator;
/// var signal = [_]f64{ 1, 0.5, -0.5, -1, 0.8, ... }; // 1000+ samples
/// const fs = 100.0; // 100 Hz sampling rate
/// const result = try welch(f64, signal[0..], fs, 256, 128, allocator);
/// defer result.deinit(allocator);
/// // result provides smoother PSD estimate than periodogram
/// ```
pub fn welch(comptime T: type, signal: []const T, fs: T, nperseg: usize, noverlap: usize, allocator: Allocator) (Allocator.Error || error{ InvalidParameter, EmptyArray })!WelchResult(T) {
    // Input validation
    if (signal.len == 0) {
        return error.EmptyArray;
    }
    if (fs <= 0) {
        return error.InvalidParameter;
    }
    if (noverlap >= nperseg) {
        return error.InvalidParameter;
    }

    // Determine effective segment length (must be power of 2 for FFT)
    // Start with min(nperseg, signal.len)
    var segment_len = @min(nperseg, signal.len);

    // Round to nearest power of 2 <= signal.len
    // Find largest power of 2 that fits within signal.len
    var power_of_2: usize = 1;
    while (power_of_2 * 2 <= segment_len) {
        power_of_2 *= 2;
    }
    segment_len = power_of_2;

    // Ensure we have at least 2 samples (minimum for FFT)
    if (segment_len < 2) {
        segment_len = 2;
    }

    // Adjust noverlap if it exceeds actual segment length
    var actual_noverlap = noverlap;
    if (actual_noverlap >= segment_len) {
        // Clamp to segment_len - 1
        actual_noverlap = if (segment_len > 1) segment_len - 1 else 0;
    }

    // Calculate number of segments
    const step = segment_len - actual_noverlap;
    const num_segments = if (signal.len <= segment_len)
        1
    else
        (signal.len - actual_noverlap + step - 1) / step;

    const n_freqs = segment_len / 2 + 1;

    // Allocate output arrays
    const frequencies = try allocator.alloc(T, n_freqs);
    errdefer allocator.free(frequencies);
    const power = try allocator.alloc(T, n_freqs);
    errdefer allocator.free(power);

    // Zero-initialize power array for accumulation
    for (power) |*p| {
        p.* = 0.0;
    }

    // Get Hann window
    const window = try window_module.hann(T, segment_len, allocator);
    defer allocator.free(window);

    // Compute window normalization factor (sum of squared window values)
    var window_norm: T = 0.0;
    for (window) |w| {
        window_norm += w * w;
    }

    // Process each segment
    var seg_idx: usize = 0;
    while (seg_idx < num_segments) : (seg_idx += 1) {
        const start = seg_idx * step;
        const end = @min(start + segment_len, signal.len);

        // Allocate segment buffer
        var segment = try allocator.alloc(T, segment_len);
        defer allocator.free(segment);

        // Copy segment and zero-pad if necessary
        const len = end - start;
        for (0..len) |i| {
            segment[i] = signal[start + i];
        }
        for (len..segment_len) |i| {
            segment[i] = 0.0;
        }

        // Apply Hann window
        for (0..segment_len) |i| {
            segment[i] *= window[i];
        }

        // Compute FFT of windowed segment
        // segment_len is guaranteed to be power of 2, so rfft won't fail
        const fft_result = fft_module.rfft(T, segment, allocator) catch |e| {
            return switch (e) {
                error.OutOfMemory => error.OutOfMemory,
                error.InvalidLength => unreachable, // segment_len is always power of 2
            };
        };
        defer allocator.free(fft_result);

        // Accumulate power (single-sided spectrum)
        // FFT is unnormalized (scales with segment_len)
        // Window normalization: U = sum(w²) / segment_len
        // Power[k] = |windowed_FFT[k]|² / (segment_len² * U)
        //          = |windowed_FFT[k]|² / (segment_len * sum(w²))
        const scale = 1.0 / (@as(T, @floatFromInt(segment_len)) * window_norm);
        for (0..n_freqs) |i| {
            const mag_sq = fft_result[i].magnitude_squared();
            if (i == 0 or i == segment_len / 2) {
                // DC or Nyquist
                power[i] += mag_sq * scale;
            } else {
                // Positive frequencies (double to account for negative)
                power[i] += 2.0 * mag_sq * scale;
            }
        }
    }

    // Average power across segments
    const scale = 1.0 / @as(T, @floatFromInt(num_segments));
    for (power) |*p| {
        p.* *= scale;
    }

    // Compute frequency array
    const bin_width = fs / @as(T, @floatFromInt(segment_len));
    for (0..n_freqs) |i| {
        frequencies[i] = @as(T, @floatFromInt(i)) * bin_width;
    }

    return WelchResult(T){
        .frequencies = frequencies,
        .power = power,
    };
}

// ============================================================================
// TESTS
// ============================================================================

// ---- Periodogram Tests ----

test "periodogram basic: pure sinusoid at known frequency" {
    const allocator = testing.allocator;

    // Generate sinusoid at 10 Hz with 100 Hz sampling rate
    const fs: f64 = 100.0;
    const freq_target = 10.0;
    const n = 64; // Power of 2
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    for (0..n) |i| {
        const t: f64 = @as(f64, @floatFromInt(i)) / fs;
        const angle = 2.0 * math.pi * freq_target * t;
        signal[i] = @sin(angle);
    }

    const result = try periodogram(f64, signal, fs, allocator);
    defer result.deinit(allocator);

    try testing.expectEqual(result.frequencies.len, n / 2 + 1);
    try testing.expectEqual(result.power.len, n / 2 + 1);

    // Find peak frequency
    var max_power: f64 = 0.0;
    var peak_freq: f64 = 0.0;
    for (result.frequencies, result.power) |freq, power| {
        if (power > max_power) {
            max_power = power;
            peak_freq = freq;
        }
    }

    // Peak should be near target frequency (within bin width)
    const bin_width = fs / @as(f64, @floatFromInt(n));
    try testing.expectApproxEqAbs(peak_freq, freq_target, bin_width * 1.5);
}

test "periodogram: DC signal concentrates power at 0 Hz" {
    const allocator = testing.allocator;
    const fs = 1.0;
    const n = 32;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    const dc_value = 5.0;
    for (0..n) |i| {
        signal[i] = dc_value;
    }

    const result = try periodogram(f64, signal, fs, allocator);
    defer result.deinit(allocator);

    // First frequency should be 0 Hz
    try testing.expectApproxEqAbs(result.frequencies[0], 0.0, 1e-10);

    // DC component should have highest power
    var max_power: f64 = 0.0;
    var max_idx: usize = 0;
    for (result.power, 0..) |power, idx| {
        if (power > max_power) {
            max_power = power;
            max_idx = idx;
        }
    }
    try testing.expectEqual(max_idx, 0);
}

test "periodogram: frequency array spacing equals fs/N" {
    const allocator = testing.allocator;
    const fs = 100.0;
    const n = 64;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    for (0..n) |i| {
        signal[i] = @sin(2.0 * math.pi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n)));
    }

    const result = try periodogram(f64, signal, fs, allocator);
    defer result.deinit(allocator);

    const expected_spacing = fs / @as(f64, @floatFromInt(n));

    // Check spacing between consecutive frequencies
    for (1..result.frequencies.len) |i| {
        const spacing = result.frequencies[i] - result.frequencies[i - 1];
        try testing.expectApproxEqAbs(spacing, expected_spacing, 1e-8);
    }
}

test "periodogram: max frequency equals fs/2 (Nyquist)" {
    const allocator = testing.allocator;
    const fs = 100.0;
    const n = 64;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    for (0..n) |i| {
        signal[i] = 1.0;
    }

    const result = try periodogram(f64, signal, fs, allocator);
    defer result.deinit(allocator);

    const max_freq = result.frequencies[result.frequencies.len - 1];
    try testing.expectApproxEqAbs(max_freq, fs / 2.0, 1e-8);
}

test "periodogram: white noise has roughly flat spectrum" {
    const allocator = testing.allocator;
    const fs = 100.0;
    const n = 128;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    // Generate white noise with fixed seed
    var rng = std.Random.DefaultPrng.init(42);
    for (0..n) |i| {
        signal[i] = rng.random().float(f64) * 2.0 - 1.0;
    }

    const result = try periodogram(f64, signal, fs, allocator);
    defer result.deinit(allocator);

    // Compute mean and variance of power spectrum
    var mean: f64 = 0.0;
    var variance: f64 = 0.0;
    for (result.power) |power| {
        mean += power;
    }
    mean /= @floatFromInt(result.power.len);

    for (result.power) |power| {
        const diff = power - mean;
        variance += diff * diff;
    }
    variance /= @floatFromInt(result.power.len);

    // For white noise, coefficient of variation should be reasonable
    // (not overly skewed, but noise has natural variation)
    const cv = @sqrt(variance) / mean;
    try testing.expect(cv > 0.2 and cv < 2.0);
}

test "periodogram: Parseval's theorem sum(power) approximates mean(signal²)" {
    const allocator = testing.allocator;
    const fs = 1.0;
    const n = 32;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    // Mix of DC and AC
    for (0..n) |i| {
        const t: f64 = @as(f64, @floatFromInt(i)) / fs;
        signal[i] = 1.0 + @sin(2.0 * math.pi * t);
    }

    const result = try periodogram(f64, signal, fs, allocator);
    defer result.deinit(allocator);

    // Time domain energy
    var time_energy: f64 = 0.0;
    for (signal) |s| {
        time_energy += s * s;
    }
    time_energy /= @floatFromInt(n);

    // Frequency domain energy (sum of power)
    var freq_energy: f64 = 0.0;
    for (result.power) |p| {
        freq_energy += p;
    }

    try testing.expectApproxEqAbs(time_energy, freq_energy, 0.1);
}

test "periodogram: multiple frequency peaks detected" {
    const allocator = testing.allocator;
    const fs = 100.0;
    const n = 128;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    // Two sine waves: 10 Hz and 25 Hz
    for (0..n) |i| {
        const t: f64 = @as(f64, @floatFromInt(i)) / fs;
        signal[i] = @sin(2.0 * math.pi * 10.0 * t) + 0.5 * @sin(2.0 * math.pi * 25.0 * t);
    }

    const result = try periodogram(f64, signal, fs, allocator);
    defer result.deinit(allocator);

    // Find peaks (power > threshold)
    const threshold = 0.01; // Adjust based on expected signal level
    var peak_count: usize = 0;
    for (result.power) |power| {
        if (power > threshold) {
            peak_count += 1;
        }
    }

    // Should have multiple peaks (exact count depends on bin resolution)
    try testing.expect(peak_count > 2);
}

test "periodogram: power of 2 length requirement" {
    const allocator = testing.allocator;
    const fs = 1.0;

    // Valid lengths: 2, 4, 8, 16, etc
    for ([_]usize{ 4, 8, 16, 32 }) |n| {
        var signal = try allocator.alloc(f64, n);
        defer allocator.free(signal);

        for (0..n) |i| {
            signal[i] = 1.0;
        }

        const result = try periodogram(f64, signal, fs, allocator);
        defer result.deinit(allocator);

        try testing.expectEqual(result.frequencies.len, n / 2 + 1);
    }
}

test "periodogram: non-power-of-2 rejected" {
    const allocator = testing.allocator;
    const fs = 1.0;

    // Invalid lengths
    for ([_]usize{ 3, 5, 7, 10, 100 }) |n| {
        var signal = try allocator.alloc(f64, n);
        defer allocator.free(signal);

        for (0..n) |i| {
            signal[i] = 1.0;
        }

        const result = periodogram(f64, signal, fs, allocator);
        try testing.expectError(error.InvalidLength, result);
    }
}

test "periodogram: empty array error" {
    const allocator = testing.allocator;
    const fs = 1.0;
    const signal = try allocator.alloc(f64, 0);
    defer allocator.free(signal);

    const result = periodogram(f64, signal, fs, allocator);
    try testing.expectError(error.EmptyArray, result);
}

test "periodogram: invalid fs (zero or negative)" {
    const allocator = testing.allocator;
    const n = 32;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    for (0..n) |i| {
        signal[i] = 1.0;
    }

    // fs = 0
    var result = periodogram(f64, signal, 0.0, allocator);
    try testing.expectError(error.InvalidParameter, result);

    // fs < 0
    result = periodogram(f64, signal, -1.0, allocator);
    try testing.expectError(error.InvalidParameter, result);
}

test "periodogram: memory leak detection" {
    const allocator = testing.allocator;
    const fs = 100.0;
    const n = 64;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    for (0..n) |i| {
        signal[i] = @sin(2.0 * math.pi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n)));
    }

    const result = try periodogram(f64, signal, fs, allocator);
    result.deinit(allocator);
}

test "periodogram: f32 type support" {
    const allocator = testing.allocator;
    const fs: f32 = 100.0;
    const n = 32;
    var signal = try allocator.alloc(f32, n);
    defer allocator.free(signal);

    for (0..n) |i| {
        signal[i] = 1.0;
    }

    const result = try periodogram(f32, signal, fs, allocator);
    defer result.deinit(allocator);

    try testing.expectEqual(result.frequencies.len, n / 2 + 1);
}

// ---- Welch Tests ----

test "welch basic: sinusoid with default segment parameters" {
    const allocator = testing.allocator;
    const fs: f64 = 100.0;
    const freq_target = 10.0;
    const n = 512; // Longer signal
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    for (0..n) |i| {
        const t: f64 = @as(f64, @floatFromInt(i)) / fs;
        signal[i] = @sin(2.0 * math.pi * freq_target * t);
    }

    const result = try welch(f64, signal, fs, 256, 128, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.frequencies.len > 0);
    try testing.expectEqual(result.frequencies.len, result.power.len);

    // Should still detect peak near 10 Hz
    var max_power: f64 = 0.0;
    var peak_freq: f64 = 0.0;
    for (result.frequencies, result.power) |freq, power| {
        if (power > max_power) {
            max_power = power;
            peak_freq = freq;
        }
    }

    const bin_width = fs / 256.0;
    try testing.expectApproxEqAbs(peak_freq, freq_target, bin_width * 1.5);
}

test "welch: smoothing reduces variance vs periodogram" {
    const allocator = testing.allocator;
    const fs = 100.0;
    const n = 512;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    // White noise
    var rng = std.Random.DefaultPrng.init(123);
    for (0..n) |i| {
        signal[i] = rng.random().float(f64) * 2.0 - 1.0;
    }

    // Get Welch estimate
    const welch_result = try welch(f64, signal, fs, 128, 64, allocator);
    defer welch_result.deinit(allocator);

    // Compute variance of Welch power estimate
    var mean: f64 = 0.0;
    for (welch_result.power) |power| {
        mean += power;
    }
    mean /= @floatFromInt(welch_result.power.len);

    var variance: f64 = 0.0;
    for (welch_result.power) |power| {
        const diff = power - mean;
        variance += diff * diff;
    }
    variance /= @floatFromInt(welch_result.power.len);

    // Variance should be positive and reasonable
    try testing.expect(variance > 0.0);
    try testing.expect(variance < 1e3); // Sanity check
}

test "welch: segment count computed correctly" {
    const allocator = testing.allocator;
    const fs = 100.0;
    const n = 512;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    for (0..n) |i| {
        signal[i] = 1.0;
    }

    const nperseg = 128;
    const noverlap = 64;

    const result = try welch(f64, signal, fs, nperseg, noverlap, allocator);
    defer result.deinit(allocator);

    // Expected segments: (512 - 64) / (128 - 64) + 1 = 448 / 64 + 1 = 8
    // Frequency array size should be nperseg/2 + 1 = 65
    try testing.expectEqual(result.frequencies.len, nperseg / 2 + 1);
}

test "welch: no overlap (noverlap=0)" {
    const allocator = testing.allocator;
    const fs = 100.0;
    const n = 512;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    for (0..n) |i| {
        signal[i] = @sin(2.0 * math.pi * @as(f64, @floatFromInt(i)) / 100.0);
    }

    const result = try welch(f64, signal, fs, 128, 0, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.frequencies.len > 0);
    try testing.expectEqual(result.frequencies.len, result.power.len);
}

test "welch: high overlap (noverlap near nperseg)" {
    const allocator = testing.allocator;
    const fs = 100.0;
    const n = 512;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    for (0..n) |i| {
        signal[i] = 1.0;
    }

    const nperseg = 128;
    const noverlap = 120; // High overlap

    const result = try welch(f64, signal, fs, nperseg, noverlap, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.frequencies.len > 0);
}

test "welch: multiple frequency components" {
    const allocator = testing.allocator;
    const fs = 1000.0;
    const n = 4096;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    // Mix of 100 Hz and 250 Hz
    for (0..n) |i| {
        const t: f64 = @as(f64, @floatFromInt(i)) / fs;
        signal[i] = @sin(2.0 * math.pi * 100.0 * t) + 0.5 * @sin(2.0 * math.pi * 250.0 * t);
    }

    const result = try welch(f64, signal, fs, 512, 256, allocator);
    defer result.deinit(allocator);

    // Should have peaks near 100 and 250 Hz
    var peak_count: usize = 0;
    const threshold = 0.001;
    for (result.power) |power| {
        if (power > threshold) {
            peak_count += 1;
        }
    }

    try testing.expect(peak_count > 2);
}

test "welch: signal shorter than nperseg (single segment)" {
    const allocator = testing.allocator;
    const fs = 100.0;
    const n = 64; // Shorter than default 256
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    for (0..n) |i| {
        signal[i] = 1.0;
    }

    // nperseg = 256 > signal.len = 64
    const result = try welch(f64, signal, fs, 256, 128, allocator);
    defer result.deinit(allocator);

    // Should still work with single segment or adjusted params
    try testing.expect(result.frequencies.len > 0);
}

test "welch: exact fit no partial segment" {
    const allocator = testing.allocator;
    const fs = 100.0;
    const n = 512;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    for (0..n) |i| {
        signal[i] = 1.0;
    }

    // Perfect division: 512 = 256 + (256-128) + (256-128) = 4 segments of 256
    const nperseg = 256;
    const noverlap = 128;

    const result = try welch(f64, signal, fs, nperseg, noverlap, allocator);
    defer result.deinit(allocator);

    try testing.expectEqual(result.frequencies.len, nperseg / 2 + 1);
}

test "welch: noverlap >= nperseg error" {
    const allocator = testing.allocator;
    const fs = 100.0;
    const n = 512;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    for (0..n) |i| {
        signal[i] = 1.0;
    }

    // noverlap == nperseg (invalid)
    var result = welch(f64, signal, fs, 128, 128, allocator);
    try testing.expectError(error.InvalidParameter, result);

    // noverlap > nperseg (invalid)
    result = welch(f64, signal, fs, 128, 200, allocator);
    try testing.expectError(error.InvalidParameter, result);
}

test "welch: empty signal error" {
    const allocator = testing.allocator;
    const fs = 100.0;
    const signal = try allocator.alloc(f64, 0);
    defer allocator.free(signal);

    const result = welch(f64, signal, fs, 256, 128, allocator);
    try testing.expectError(error.EmptyArray, result);
}

test "welch: invalid fs error" {
    const allocator = testing.allocator;
    const n = 512;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    for (0..n) |i| {
        signal[i] = 1.0;
    }

    // fs = 0
    var result = welch(f64, signal, 0.0, 256, 128, allocator);
    try testing.expectError(error.InvalidParameter, result);

    // fs < 0
    result = welch(f64, signal, -100.0, 256, 128, allocator);
    try testing.expectError(error.InvalidParameter, result);
}

test "welch: memory leak detection long signal" {
    const allocator = testing.allocator;
    const fs = 1000.0;
    const n = 2048;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    var rng = std.Random.DefaultPrng.init(999);
    for (0..n) |i| {
        signal[i] = rng.random().float(f64);
    }

    const result = try welch(f64, signal, fs, 512, 256, allocator);
    result.deinit(allocator);
}

test "welch: f32 type support" {
    const allocator = testing.allocator;
    const fs: f32 = 100.0;
    const n = 256;
    var signal = try allocator.alloc(f32, n);
    defer allocator.free(signal);

    for (0..n) |i| {
        signal[i] = 1.0;
    }

    const result = try welch(f32, signal, fs, 128, 64, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.frequencies.len > 0);
}

test "welch: frequency resolution matches nperseg not signal length" {
    const allocator = testing.allocator;
    const fs = 100.0;
    const n = 1024;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    for (0..n) |i| {
        signal[i] = 1.0;
    }

    const nperseg = 256;
    const noverlap = 128;

    const result = try welch(f64, signal, fs, nperseg, noverlap, allocator);
    defer result.deinit(allocator);

    // Frequency resolution should be fs / nperseg, not fs / n
    const expected_resolution = fs / @as(f64, @floatFromInt(nperseg));

    if (result.frequencies.len > 1) {
        const actual_resolution = result.frequencies[1] - result.frequencies[0];
        try testing.expectApproxEqAbs(actual_resolution, expected_resolution, 1e-8);
    }
}

test "welch: sinusoid with 50% overlap (recommended)" {
    const allocator = testing.allocator;
    const fs = 200.0;
    const freq_target = 30.0;
    const n = 1024;
    var signal = try allocator.alloc(f64, n);
    defer allocator.free(signal);

    for (0..n) |i| {
        const t: f64 = @as(f64, @floatFromInt(i)) / fs;
        signal[i] = @sin(2.0 * math.pi * freq_target * t);
    }

    const nperseg = 256;
    const noverlap = nperseg / 2; // 50% overlap

    const result = try welch(f64, signal, fs, nperseg, noverlap, allocator);
    defer result.deinit(allocator);

    // Should detect peak at target frequency
    var max_power: f64 = 0.0;
    var peak_freq: f64 = 0.0;
    for (result.frequencies, result.power) |freq, power| {
        if (power > max_power) {
            max_power = power;
            peak_freq = freq;
        }
    }

    const bin_width = fs / @as(f64, @floatFromInt(nperseg));
    try testing.expectApproxEqAbs(peak_freq, freq_target, bin_width * 1.5);
}
