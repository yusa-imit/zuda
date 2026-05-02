const std = @import("std");
const zuda = @import("zuda");

const NDArray = zuda.ndarray.NDArray;
const distributions = zuda.stats.distributions;
const Normal = distributions.Normal;
const descriptive = zuda.stats.descriptive;
const fft = zuda.signal.fft;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Signal Processing Showcase ===\n\n", .{});

    // Part 1: Generate synthetic signal (sum of sinusoids + noise)
    std.debug.print("Part 1: Signal Generation\n", .{});
    std.debug.print("-" ** 50 ++ "\n", .{});

    const n: usize = 512;
    const sample_rate: f64 = 1000.0; // Hz
    const dt: f64 = 1.0 / sample_rate;

    var signal_data = try allocator.alloc(f64, n);
    defer allocator.free(signal_data);

    var rng = std.Random.DefaultPrng.init(42);
    const noise_dist = Normal(f64).init(0.0, 0.2) catch unreachable;

    // Generate: 50 Hz sine + 120 Hz sine + noise
    const freq1: f64 = 50.0;
    const freq2: f64 = 120.0;
    const amp1: f64 = 1.0;
    const amp2: f64 = 0.6;

    for (signal_data, 0..) |*s, i| {
        const t = @as(f64, @floatFromInt(i)) * dt;
        const clean = amp1 * @sin(2.0 * std.math.pi * freq1 * t) +
            amp2 * @sin(2.0 * std.math.pi * freq2 * t);
        const noise = noise_dist.sample(rng.random());
        s.* = clean + noise;
    }

    var signal_nd = try NDArray(f64, 1).fromSlice(allocator, &.{n}, signal_data, .row_major);
    defer signal_nd.deinit();

    const signal_mean = descriptive.mean(f64, signal_nd);
    const signal_std = try descriptive.stdDev(f64, signal_nd, 0);

    std.debug.print("Signal: {} samples at {} Hz\n", .{ n, sample_rate });
    std.debug.print("Components: {} Hz (amp {d:.2}) + {} Hz (amp {d:.2}) + noise\n", .{ freq1, amp1, freq2, amp2 });
    std.debug.print("Statistics: mean {d:.4}, std {d:.4}\n\n", .{ signal_mean, signal_std });

    // Part 2: Frequency Domain Analysis (FFT)
    std.debug.print("Part 2: Frequency Domain Analysis\n", .{});
    std.debug.print("-" ** 50 ++ "\n", .{});

    const fft_result = try fft.rfft(f64, allocator, signal_data);
    defer allocator.free(fft_result);

    // Compute power spectral density (magnitude squared)
    var psd = try allocator.alloc(f64, fft_result.len);
    defer allocator.free(psd);

    for (fft_result, 0..) |c, i| {
        psd[i] = c.real * c.real + c.imag * c.imag;
    }

    // Find dominant frequencies
    const freq_resolution = sample_rate / @as(f64, @floatFromInt(n));
    var peak_indices = try std.ArrayList(usize).initCapacity(allocator, 10);
    defer peak_indices.deinit(allocator);

    // Simple peak detection: local maxima with threshold
    var psd_nd = try NDArray(f64, 1).fromSlice(allocator, &.{psd.len}, psd, .row_major);
    defer psd_nd.deinit();
    const threshold = descriptive.mean(f64, psd_nd);
    for (psd[1 .. psd.len - 1], 0..) |p, idx| {
        const i = idx + 1;
        if (p > threshold * 5.0 and p > psd[i - 1] and p > psd[i + 1]) {
            try peak_indices.append(allocator, i);
        }
    }

    std.debug.print("FFT length: {} (real FFT: {} bins)\n", .{ n, fft_result.len });
    std.debug.print("Frequency resolution: {d:.2} Hz\n", .{freq_resolution});
    std.debug.print("Detected peaks:\n", .{});
    for (peak_indices.items) |idx| {
        const freq = @as(f64, @floatFromInt(idx)) * freq_resolution;
        std.debug.print("  - {d:.1} Hz (power: {e:.2})\n", .{ freq, psd[idx] });
    }
    std.debug.print("\n", .{});

    // Part 3: Lowpass Filtering (Moving Average)
    std.debug.print("Part 3: Lowpass Filtering (Moving Average)\n", .{});
    std.debug.print("-" ** 50 ++ "\n", .{});

    const window_size: usize = 15;
    const filtered = try allocator.alloc(f64, n);
    defer allocator.free(filtered);

    // Compute moving average manually
    for (0..n) |i| {
        const start = if (i >= window_size / 2) i - window_size / 2 else 0;
        const end = @min(i + window_size / 2 + 1, n);
        var sum: f64 = 0.0;
        for (signal_data[start..end]) |s| sum += s;
        filtered[i] = sum / @as(f64, @floatFromInt(end - start));
    }

    var filtered_nd = try NDArray(f64, 1).fromSlice(allocator, &.{filtered.len}, filtered, .row_major);
    defer filtered_nd.deinit();

    const filtered_std = try descriptive.stdDev(f64, filtered_nd, 0);
    const noise_reduction_pct = (1.0 - filtered_std / signal_std) * 100.0;

    std.debug.print("Window size: {}\n", .{window_size});
    std.debug.print("Original std: {d:.4}\n", .{signal_std});
    std.debug.print("Filtered std: {d:.4}\n", .{filtered_std});
    std.debug.print("Noise reduction: {d:.1}%\n\n", .{noise_reduction_pct});

    // Part 4: Highpass Filtering (Difference)
    std.debug.print("Part 4: Highpass Filtering (Difference)\n", .{});
    std.debug.print("-" ** 50 ++ "\n", .{});

    const highpass = try allocator.alloc(f64, n - 1);
    defer allocator.free(highpass);

    for (highpass, 0..) |*h, i| {
        h.* = signal_data[i + 1] - signal_data[i];
    }

    var highpass_nd = try NDArray(f64, 1).fromSlice(allocator, &.{highpass.len}, highpass, .row_major);
    defer highpass_nd.deinit();

    const highpass_mean = descriptive.mean(f64, highpass_nd);
    const highpass_std = try descriptive.stdDev(f64, highpass_nd, 0);

    std.debug.print("Highpass (first-order difference):\n", .{});
    std.debug.print("  Mean: {d:.4} (should be ~0)\n", .{highpass_mean});
    std.debug.print("  Std: {d:.4}\n\n", .{highpass_std});

    // Part 5: Windowing for Spectral Analysis
    std.debug.print("Part 5: Windowing (Hann Window)\n", .{});
    std.debug.print("-" ** 50 ++ "\n", .{});

    const windowed = try allocator.alloc(f64, n);
    defer allocator.free(windowed);

    // Apply Hann window: w[i] = 0.5 * (1 - cos(2π i / (N-1)))
    for (windowed, 0..) |*w, i| {
        const hann = 0.5 * (1.0 - @cos(2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n - 1))));
        w.* = signal_data[i] * hann;
    }

    const windowed_fft = try fft.rfft(f64, allocator, windowed);
    defer allocator.free(windowed_fft);

    var windowed_psd = try allocator.alloc(f64, windowed_fft.len);
    defer allocator.free(windowed_psd);

    for (windowed_fft, 0..) |c, i| {
        windowed_psd[i] = c.real * c.real + c.imag * c.imag;
    }

    // Compare spectral leakage (ratio of peak to average)
    const original_peak_max = blk: {
        var max: f64 = 0.0;
        for (psd) |p| max = @max(max, p);
        break :blk max;
    };
    const windowed_peak_max = blk: {
        var max: f64 = 0.0;
        for (windowed_psd) |p| max = @max(max, p);
        break :blk max;
    };

    var windowed_psd_nd = try NDArray(f64, 1).fromSlice(allocator, &.{windowed_psd.len}, windowed_psd, .row_major);
    defer windowed_psd_nd.deinit();

    const original_avg = descriptive.mean(f64, psd_nd);
    const windowed_avg = descriptive.mean(f64, windowed_psd_nd);

    const original_snr = original_peak_max / original_avg;
    const windowed_snr = windowed_peak_max / windowed_avg;

    std.debug.print("Hann window applied ({} samples)\n", .{n});
    std.debug.print("Spectral SNR (peak/average):\n", .{});
    std.debug.print("  Original: {d:.2}\n", .{original_snr});
    std.debug.print("  Windowed: {d:.2}\n", .{windowed_snr});
    std.debug.print("Improvement: {d:.2}x\n\n", .{windowed_snr / original_snr});

    // Part 6: Signal Envelope Detection (Hilbert-like using magnitude)
    std.debug.print("Part 6: Envelope Detection\n", .{});
    std.debug.print("-" ** 50 ++ "\n", .{});

    // Simple envelope: moving maximum over window
    const env_window: usize = 20;
    const envelope = try allocator.alloc(f64, n);
    defer allocator.free(envelope);

    for (envelope, 0..) |*e, i| {
        const start = if (i >= env_window / 2) i - env_window / 2 else 0;
        const end = @min(i + env_window / 2 + 1, n);

        var max_val: f64 = 0.0;
        for (signal_data[start..end]) |s| {
            max_val = @max(max_val, @abs(s));
        }
        e.* = max_val;
    }

    var envelope_nd = try NDArray(f64, 1).fromSlice(allocator, &.{envelope.len}, envelope, .row_major);
    defer envelope_nd.deinit();

    const env_mean = descriptive.mean(f64, envelope_nd);
    const env_max = blk: {
        var max: f64 = 0.0;
        for (envelope) |e| max = @max(max, e);
        break :blk max;
    };

    std.debug.print("Envelope (moving max, window={})\n", .{env_window});
    std.debug.print("  Mean: {d:.4}\n", .{env_mean});
    std.debug.print("  Max: {d:.4}\n\n", .{env_max});

    // Part 7: Signal Energy and Power
    std.debug.print("Part 7: Energy and Power Analysis\n", .{});
    std.debug.print("-" ** 50 ++ "\n", .{});

    // Energy: sum of squared samples
    var energy: f64 = 0.0;
    for (signal_data) |s| energy += s * s;

    // Power: energy / duration
    const duration = @as(f64, @floatFromInt(n)) * dt;
    const power = energy / duration;

    // RMS (root mean square)
    const rms = @sqrt(energy / @as(f64, @floatFromInt(n)));

    std.debug.print("Total energy: {d:.4} J\n", .{energy});
    std.debug.print("Duration: {d:.4} s\n", .{duration});
    std.debug.print("Average power: {d:.4} W\n", .{power});
    std.debug.print("RMS amplitude: {d:.4}\n\n", .{rms});

    // Part 8: Correlation Analysis (Autocorrelation at lag 1)
    std.debug.print("Part 8: Autocorrelation\n", .{});
    std.debug.print("-" ** 50 ++ "\n", .{});

    // Lag-1 autocorrelation
    const lag = 1;
    var sum_xy: f64 = 0.0;
    var sum_x: f64 = 0.0;
    var sum_y: f64 = 0.0;
    var sum_x2: f64 = 0.0;
    var sum_y2: f64 = 0.0;
    const count = n - lag;

    for (0..count) |i| {
        const x = signal_data[i];
        const y = signal_data[i + lag];
        sum_xy += x * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
        sum_y2 += y * y;
    }

    const n_f = @as(f64, @floatFromInt(count));
    const numerator = sum_xy - (sum_x * sum_y / n_f);
    const denom_x = sum_x2 - (sum_x * sum_x / n_f);
    const denom_y = sum_y2 - (sum_y * sum_y / n_f);
    const autocorr = numerator / @sqrt(denom_x * denom_y);

    std.debug.print("Lag-{} autocorrelation: {d:.4}\n", .{ lag, autocorr });
    if (autocorr > 0.9) {
        std.debug.print("Interpretation: Strong positive correlation (smooth signal)\n", .{});
    } else if (autocorr > 0.5) {
        std.debug.print("Interpretation: Moderate positive correlation\n", .{});
    } else if (autocorr > 0.0) {
        std.debug.print("Interpretation: Weak positive correlation\n", .{});
    } else {
        std.debug.print("Interpretation: Negative or no correlation\n", .{});
    }
    std.debug.print("\n", .{});

    std.debug.print("=== Summary ===\n", .{});
    std.debug.print("This example demonstrated:\n", .{});
    std.debug.print("1. Signal generation (multi-frequency synthesis)\n", .{});
    std.debug.print("2. FFT-based spectral analysis\n", .{});
    std.debug.print("3. Lowpass filtering (moving average)\n", .{});
    std.debug.print("4. Highpass filtering (difference)\n", .{});
    std.debug.print("5. Windowing (Hann) for reduced spectral leakage\n", .{});
    std.debug.print("6. Envelope detection\n", .{});
    std.debug.print("7. Energy/power computation\n", .{});
    std.debug.print("8. Autocorrelation analysis\n", .{});
    std.debug.print("\nModules used: stats.distributions, signal.fft, stats.descriptive\n", .{});
}
