# Signal Processing — FFT, Filtering, and Transforms

## Overview

The `signal` module provides tools for digital signal processing, including Fast Fourier Transform (FFT), convolution, windowing functions, and spectral analysis. It's designed for audio processing, time-series analysis, and frequency-domain computations.

## Module Structure

```zig
const signal = zuda.signal;

// FFT and transforms
const fft = signal.fft;

// Convolution
const conv = signal.convolution;

// Window functions
const window = signal.windows;
```

## Fast Fourier Transform (FFT)

### Complex FFT

Transforms time-domain signal to frequency-domain.

```zig
const std = @import("std");
const zuda = @import("zuda");
const Complex = std.math.Complex;

// Time-domain signal (must be power of 2 length)
var signal_data = try std.ArrayList(Complex(f64)).initCapacity(allocator, 1024);
defer signal_data.deinit(allocator);

// Generate test signal: sin(2π * 10Hz * t) at 1000 Hz sample rate
const sample_rate = 1000.0;
const freq = 10.0;
for (0..1024) |i| {
    const t = @as(f64, @floatFromInt(i)) / sample_rate;
    const val = @sin(2.0 * std.math.pi * freq * t);
    signal_data.appendAssumeCapacity(Complex(f64).init(val, 0));
}

// Forward FFT
const spectrum = try fft.fft(f64, signal_data.items, allocator);
defer allocator.free(spectrum);

// spectrum[k] = Σ signal[n] * exp(-2πi * k * n / N)
// spectrum.len == signal.len

// Frequency bins
for (spectrum, 0..) |coef, k| {
    const freq_bin = @as(f64, @floatFromInt(k)) * sample_rate / @as(f64, @floatFromInt(spectrum.len));
    const magnitude = coef.magnitude();
    std.debug.print("Freq {d:.2} Hz: {d:.4}\n", .{freq_bin, magnitude});
}
```

### Inverse FFT

Transforms frequency-domain back to time-domain.

```zig
// Inverse FFT (reconstructs original signal)
const reconstructed = try fft.ifft(f64, spectrum, allocator);
defer allocator.free(reconstructed);

// Verify: reconstructed ≈ original (within numerical precision)
for (signal_data.items, 0..) |original, i| {
    const diff = original.re - reconstructed[i].re;
    std.debug.assert(@abs(diff) < 1e-10);
}
```

### Real FFT (Optimized)

For real-valued signals, uses symmetry to compute only half the spectrum.

```zig
// Real signal (no imaginary part)
var real_signal = try std.ArrayList(f64).initCapacity(allocator, 1024);
defer real_signal.deinit(allocator);

for (0..1024) |i| {
    const t = @as(f64, @floatFromInt(i)) / sample_rate;
    const val = @sin(2.0 * std.math.pi * 10.0 * t);
    real_signal.appendAssumeCapacity(val);
}

// Real FFT (returns N/2+1 complex coefficients)
const spectrum_real = try fft.rfft(f64, real_signal.items, allocator);
defer allocator.free(spectrum_real);

// spectrum_real.len == 513 (1024/2 + 1)
// Positive frequencies only; negative frequencies are conjugate symmetric
```

### Inverse Real FFT

```zig
// Inverse real FFT
const reconstructed_real = try fft.irfft(f64, spectrum_real, 1024, allocator);
defer allocator.free(reconstructed_real);

// reconstructed_real.len == 1024 (original length)
```

**Performance**: Real FFT is ~2x faster than complex FFT for real signals.

## Power Spectral Density

Estimate power distribution across frequencies.

```zig
// Periodogram (simple PSD estimate)
const psd = try fft.periodogram(f64, real_signal.items, sample_rate, allocator);
defer allocator.free(psd.power);
defer allocator.free(psd.freqs);

// psd.power[k] = |FFT[k]|² / N
// psd.freqs[k] = k * sample_rate / N

for (psd.freqs, 0..) |freq_val, k| {
    std.debug.print("Freq {d:.2} Hz: Power {d:.4}\n", .{freq_val, psd.power[k]});
}
```

## Windowing Functions

Reduce spectral leakage in FFT by tapering signal edges.

### Available Windows

```zig
// Rectangular (no windowing)
const rect = try window.rectangular(f64, 1024, allocator);
defer allocator.free(rect);

// Hann (raised cosine)
const hann = try window.hann(f64, 1024, allocator);
defer allocator.free(hann);

// Hamming
const hamming = try window.hamming(f64, 1024, allocator);
defer allocator.free(hamming);

// Blackman
const blackman = try window.blackman(f64, 1024, allocator);
defer allocator.free(blackman);

// Bartlett (triangular)
const bartlett = try window.bartlett(f64, 1024, allocator);
defer allocator.free(bartlett);
```

### Applying Window

```zig
// Apply window before FFT
const hann_window = try window.hann(f64, signal.len, allocator);
defer allocator.free(hann_window);

for (signal.items, 0..) |*val, i| {
    val.re *= hann_window[i];
    val.im *= hann_window[i];
}

const windowed_spectrum = try fft.fft(f64, signal.items, allocator);
defer allocator.free(windowed_spectrum);
```

**Window Selection**:
- **Rectangular**: No attenuation (default)
- **Hann**: Good frequency resolution, moderate sidelobe suppression
- **Hamming**: Similar to Hann, slightly better sidelobe suppression
- **Blackman**: Best sidelobe suppression, wider main lobe
- **Bartlett**: Linear taper, simple

## Convolution

### Linear Convolution

```zig
const signal_a = [_]f64{1, 2, 3, 4};
const signal_b = [_]f64{0.5, 1, 0.5};

// Convolve: output.len = a.len + b.len - 1
const result = try conv.convolve(f64, &signal_a, &signal_b, allocator);
defer allocator.free(result);

// result.len == 6
// Applications: filtering, smoothing, feature detection
```

### Circular Convolution

```zig
// Periodic convolution (output.len = a.len)
const result_circ = try conv.convolve_circular(f64, &signal_a, &signal_b, allocator);
defer allocator.free(result_circ);

// Equivalent to FFT-based: IFFT(FFT(a) * FFT(b))
```

### FFT-Based Convolution

Fast convolution for large signals using FFT.

```zig
// For large signals, FFT convolution is O(N log N) vs O(N²) for direct
const large_signal = try std.ArrayList(f64).initCapacity(allocator, 10000);
defer large_signal.deinit(allocator);
// ... fill signal ...

const kernel = [_]f64{0.2, 0.2, 0.2, 0.2, 0.2};  // Moving average

const result_fft = try conv.fftconvolve(f64, large_signal.items, &kernel, allocator);
defer allocator.free(result_fft);

// Automatically uses FFT when beneficial
```

## Common Applications

### Frequency Analysis

Identify dominant frequencies in a signal.

```zig
// Generate signal with multiple frequencies
var signal_multi = try std.ArrayList(f64).initCapacity(allocator, 1024);
defer signal_multi.deinit(allocator);

const fs = 1000.0;  // Sample rate
for (0..1024) |i| {
    const t = @as(f64, @floatFromInt(i)) / fs;
    // 50 Hz + 120 Hz components
    const val = @sin(2.0 * std.math.pi * 50.0 * t) +
                0.5 * @sin(2.0 * std.math.pi * 120.0 * t);
    signal_multi.appendAssumeCapacity(val);
}

// Apply window and compute FFT
const hann_win = try window.hann(f64, 1024, allocator);
defer allocator.free(hann_win);

for (signal_multi.items, 0..) |*val, i| {
    val.* *= hann_win[i];
}

const spectrum_multi = try fft.rfft(f64, signal_multi.items, allocator);
defer allocator.free(spectrum_multi);

// Find peaks
var max_mag: f64 = 0;
var max_idx: usize = 0;
for (spectrum_multi, 0..) |coef, k| {
    const mag = coef.magnitude();
    if (mag > max_mag) {
        max_mag = mag;
        max_idx = k;
    }
}

const dominant_freq = @as(f64, @floatFromInt(max_idx)) * fs / 1024.0;
std.debug.print("Dominant frequency: {d:.2} Hz\n", .{dominant_freq});
```

### Low-Pass Filter (Smoothing)

Remove high-frequency noise.

```zig
// Moving average filter (simple low-pass)
const window_size = 5;
var kernel = try std.ArrayList(f64).initCapacity(allocator, window_size);
defer kernel.deinit(allocator);

for (0..window_size) |_| {
    kernel.appendAssumeCapacity(1.0 / @as(f64, @floatFromInt(window_size)));
}

const smoothed = try conv.convolve(f64, noisy_signal, kernel.items, allocator);
defer allocator.free(smoothed);
```

### High-Pass Filter

Remove low-frequency trends.

```zig
// High-pass = Original - Low-pass
const low_pass = try conv.convolve(f64, signal, low_pass_kernel, allocator);
defer allocator.free(low_pass);

var high_pass = try std.ArrayList(f64).initCapacity(allocator, signal.len);
defer high_pass.deinit(allocator);

for (signal, 0..) |val, i| {
    if (i < low_pass.len) {
        high_pass.appendAssumeCapacity(val - low_pass[i]);
    }
}
```

### Spectrogram (Time-Frequency Analysis)

Visualize how frequency content changes over time.

```zig
// Short-Time Fourier Transform (STFT)
const win_len = 256;
const hop_len = 128;  // 50% overlap

var spectrogram = std.ArrayList([]Complex(f64)).init(allocator);
defer {
    for (spectrogram.items) |frame| {
        allocator.free(frame);
    }
    spectrogram.deinit();
}

const hann_stft = try window.hann(f64, win_len, allocator);
defer allocator.free(hann_stft);

var offset: usize = 0;
while (offset + win_len <= signal.len) : (offset += hop_len) {
    // Extract and window frame
    var frame = try std.ArrayList(Complex(f64)).initCapacity(allocator, win_len);
    for (0..win_len) |i| {
        const val = signal[offset + i] * hann_stft[i];
        frame.appendAssumeCapacity(Complex(f64).init(val, 0));
    }

    // Compute FFT of frame
    const frame_spectrum = try fft.fft(f64, frame.items, allocator);
    try spectrogram.append(frame_spectrum);
}

// spectrogram[time_idx][freq_idx] = complex coefficient
// Can visualize as heatmap: time × frequency → magnitude
```

### Cross-Correlation

Measure similarity between signals at different lags.

```zig
const signal1 = [_]f64{1, 2, 3, 4, 5};
const signal2 = [_]f64{0, 1, 2, 3, 4};  // Delayed version

// Cross-correlation
const xcorr = try conv.correlate(f64, &signal1, &signal2, allocator);
defer allocator.free(xcorr);

// Find lag with maximum correlation
var max_corr: f64 = -std.math.inf(f64);
var best_lag: isize = 0;
for (xcorr, 0..) |val, i| {
    if (val > max_corr) {
        max_corr = val;
        best_lag = @as(isize, @intCast(i)) - @as(isize, @intCast(signal2.len)) + 1;
    }
}

std.debug.print("Best lag: {} samples\n", .{best_lag});
```

### Auto-Correlation

Detect periodicity in a signal.

```zig
// Auto-correlation = cross-correlation with itself
const autocorr = try conv.correlate(f64, signal, signal, allocator);
defer allocator.free(autocorr);

// Peaks in auto-correlation indicate period
```

## FFT Properties and Tricks

### Parsevals Theorem

Energy in time domain equals energy in frequency domain.

```zig
// Time-domain energy
var energy_time: f64 = 0;
for (signal.items) |val| {
    energy_time += val.re * val.re + val.im * val.im;
}

// Frequency-domain energy
var energy_freq: f64 = 0;
for (spectrum) |coef| {
    energy_freq += coef.re * coef.re + coef.im * coef.im;
}
energy_freq /= @as(f64, @floatFromInt(spectrum.len));

// energy_time ≈ energy_freq (within numerical precision)
```

### Zero-Padding

Increase frequency resolution by padding with zeros.

```zig
var padded_signal = try std.ArrayList(Complex(f64)).initCapacity(allocator, 2048);
defer padded_signal.deinit(allocator);

// Original 1024 samples
for (signal.items) |val| {
    try padded_signal.append(val);
}

// Pad to 2048 with zeros
for (0..1024) |_| {
    try padded_signal.append(Complex(f64).init(0, 0));
}

// FFT gives 2x frequency resolution
const padded_spectrum = try fft.fft(f64, padded_signal.items, allocator);
defer allocator.free(padded_spectrum);
```

### Shift Theorem

Time shift ↔ Phase shift in frequency domain.

```zig
// Shift signal by k samples in frequency domain
const k = 10;
for (spectrum, 0..) |*coef, n| {
    const phase = -2.0 * std.math.pi * @as(f64, @floatFromInt(k * n)) / @as(f64, @floatFromInt(spectrum.len));
    const shift_factor = Complex(f64).init(@cos(phase), @sin(phase));
    coef.* = coef.mul(shift_factor);
}

const shifted = try fft.ifft(f64, spectrum, allocator);
defer allocator.free(shifted);
```

## Performance Tips

1. **Use power-of-2 lengths**: FFT is fastest for N = 2^k (pad with zeros if needed)
2. **Use real FFT for real signals**: 2x speedup over complex FFT
3. **Apply window only once**: Cache windowed signal if computing multiple FFTs
4. **Use FFT convolution for large kernels**: O(N log N) vs O(N²) for direct convolution
5. **Exploit symmetry**: Real signals have conjugate-symmetric spectra; only need half

## Error Handling

```zig
const spectrum = fft.fft(f64, signal, allocator) catch |err| switch (err) {
    error.InvalidLength => {
        std.debug.print("Signal length must be power of 2\n", .{});
        return;
    },
    error.EmptyArray => {
        std.debug.print("Signal cannot be empty\n", .{});
        return;
    },
    else => return err,
};

const conv_result = conv.convolve(f64, a, b, allocator) catch |err| switch (err) {
    error.InsufficientMemory => {
        std.debug.print("Not enough memory for convolution\n", .{});
        return;
    },
    else => return err,
};
```

## See Also

- [NDArray Guide](ndarray.md) — Array operations for signal data
- [Statistics Guide](stats.md) — Statistical analysis of signals
- [NumPy Compatibility](../NUMPY_COMPATIBILITY.md) — NumPy signal/fft → zuda mapping
