const std = @import("std");
const zuda = @import("zuda");

const NDArray = zuda.ndarray.NDArray;
const descriptive = zuda.stats.descriptive;
const distributions = zuda.stats.distributions;

/// Anomaly Detection Example
///
/// Demonstrates:
/// 1. Statistical anomaly detection methods (Z-score, MAD, IQR)
/// 2. Time series smoothing and filtering
/// 3. Robust statistics for outlier handling
/// 4. Multivariate anomaly detection (Mahalanobis distance)
///
/// Use Case: Network traffic monitoring, sensor fault detection, fraud detection
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Anomaly Detection Demonstration ===\n\n", .{});

    // Part 1: Z-Score Method (assumes Gaussian distribution)
    std.debug.print("Part 1: Z-Score Anomaly Detection\n", .{});
    std.debug.print("----------------------------------\n", .{});

    const n_samples = 100;
    var data_list = try std.ArrayList(f64).initCapacity(allocator, n_samples);
    defer data_list.deinit(allocator);

    // Generate normal data with some outliers
    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    const normal_dist = try distributions.Normal(f64).init(50.0, 5.0); // mean=50, std=5

    for (0..n_samples) |i| {
        const value: f64 = if (i % 20 == 0 and i > 0) blk: {
            // Inject outliers: 5% of data points
            const outlier: f64 = if (random.boolean()) 80.0 else 20.0;
            break :blk outlier;
        } else blk: {
            break :blk normal_dist.sample(random);
        };
        data_list.appendAssumeCapacity(value);
    }

    // Compute Z-scores
    const data_slice = data_list.items;
    var data_nd = try NDArray(f64, 1).fromSlice(allocator, &.{n_samples}, data_slice, .row_major);
    defer data_nd.deinit();
    const mean_val = descriptive.mean(f64, data_nd);
    const std_val = try descriptive.stdDev(f64, data_nd, 0); // population std

    var z_scores = try std.ArrayList(f64).initCapacity(allocator, n_samples);
    defer z_scores.deinit(allocator);

    var anomaly_count_zscore: usize = 0;
    const z_threshold = 3.0; // 3-sigma rule

    for (data_slice) |value| {
        const z = @abs((value - mean_val) / std_val);
        z_scores.appendAssumeCapacity(z);
        if (z > z_threshold) {
            anomaly_count_zscore += 1;
        }
    }

    std.debug.print("Dataset: {d} samples\n", .{n_samples});
    std.debug.print("Mean: {d:.2}, Std: {d:.2}\n", .{ mean_val, std_val });
    std.debug.print("Z-score threshold: {d:.1} (3-sigma)\n", .{z_threshold});
    std.debug.print("Anomalies detected: {d}/{d} ({d:.1}%)\n\n", .{ anomaly_count_zscore, n_samples, @as(f64, @floatFromInt(anomaly_count_zscore)) / @as(f64, @floatFromInt(n_samples)) * 100.0 });

    // Part 2: MAD Method (robust to outliers)
    std.debug.print("Part 2: MAD (Median Absolute Deviation) Method\n", .{});
    std.debug.print("-----------------------------------------------\n", .{});

    // MAD is more robust than std for outlier detection
    const median_val = try descriptive.median(f64, data_nd, allocator);

    // Compute absolute deviations from median
    var abs_deviations = try std.ArrayList(f64).initCapacity(allocator, n_samples);
    defer abs_deviations.deinit(allocator);

    for (data_slice) |value| {
        abs_deviations.appendAssumeCapacity(@abs(value - median_val));
    }

    // MAD = median of absolute deviations
    var abs_deviations_nd = try NDArray(f64, 1).fromSlice(allocator, &.{n_samples}, abs_deviations.items, .row_major);
    defer abs_deviations_nd.deinit();
    const mad = try descriptive.median(f64, abs_deviations_nd, allocator);

    // Modified Z-score using MAD (constant 1.4826 makes it consistent with std for normal dist)
    const mad_threshold = 3.5; // Common threshold for modified Z-score
    var anomaly_count_mad: usize = 0;

    for (data_slice) |value| {
        const modified_z = 0.6745 * @abs(value - median_val) / mad; // 0.6745 ≈ 1/1.4826
        if (modified_z > mad_threshold) {
            anomaly_count_mad += 1;
        }
    }

    std.debug.print("Median: {d:.2}, MAD: {d:.2}\n", .{ median_val, mad });
    std.debug.print("Modified Z-score threshold: {d:.1}\n", .{mad_threshold});
    std.debug.print("Anomalies detected: {d}/{d} ({d:.1}%)\n\n", .{ anomaly_count_mad, n_samples, @as(f64, @floatFromInt(anomaly_count_mad)) / @as(f64, @floatFromInt(n_samples)) * 100.0 });

    // Part 3: IQR Method (Interquartile Range)
    std.debug.print("Part 3: IQR (Interquartile Range) Method\n", .{});
    std.debug.print("-----------------------------------------\n", .{});

    // IQR method: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are outliers
    const q1 = try descriptive.quantile(f64, data_nd, 0.25, allocator);
    const q3 = try descriptive.quantile(f64, data_nd, 0.75, allocator);
    const iqr = q3 - q1;

    const lower_bound = q1 - 1.5 * iqr;
    const upper_bound = q3 + 1.5 * iqr;

    var anomaly_count_iqr: usize = 0;
    for (data_slice) |value| {
        if (value < lower_bound or value > upper_bound) {
            anomaly_count_iqr += 1;
        }
    }

    std.debug.print("Q1: {d:.2}, Q3: {d:.2}, IQR: {d:.2}\n", .{ q1, q3, iqr });
    std.debug.print("Bounds: [{d:.2}, {d:.2}]\n", .{ lower_bound, upper_bound });
    std.debug.print("Anomalies detected: {d}/{d} ({d:.1}%)\n\n", .{ anomaly_count_iqr, n_samples, @as(f64, @floatFromInt(anomaly_count_iqr)) / @as(f64, @floatFromInt(n_samples)) * 100.0 });

    // Part 4: Time Series Anomaly Detection with Smoothing
    std.debug.print("Part 4: Time Series Anomaly Detection\n", .{});
    std.debug.print("--------------------------------------\n", .{});

    // Generate time series with trend + noise + anomalies
    const n_ts = 200;
    var ts_data = try std.ArrayList(f64).initCapacity(allocator, n_ts);
    defer ts_data.deinit(allocator);
    for (0..n_ts) |i| {
        const t = @as(f64, @floatFromInt(i));
        // Trend: linear increase
        const trend = 100.0 + 0.5 * t;
        // Seasonal: sine wave
        const seasonal = 10.0 * @sin(2.0 * std.math.pi * t / 20.0);
        // Noise: small random fluctuation
        const noise = normal_dist.sample(random) - 50.0; // center noise around 0

        var value = trend + seasonal + noise;

        // Inject anomalies (spikes)
        if (i == 50 or i == 100 or i == 150) {
            value += 50.0; // Positive spike
        } else if (i == 75 or i == 125) {
            value -= 40.0; // Negative spike
        }

        ts_data.appendAssumeCapacity(value);
    }

    // Simple moving average for smoothing (window = 10)
    const window_size = 10;
    var smoothed = try std.ArrayList(f64).initCapacity(allocator, n_ts);
    defer smoothed.deinit(allocator);

    for (0..n_ts) |i| {
        const start = if (i >= window_size / 2) i - window_size / 2 else 0;
        const end = @min(i + window_size / 2 + 1, n_ts);

        var sum: f64 = 0.0;
        for (start..end) |j| {
            sum += ts_data.items[j];
        }
        smoothed.appendAssumeCapacity(sum / @as(f64, @floatFromInt(end - start)));
    }

    // Detect anomalies as large deviations from smoothed trend
    var residuals = try std.ArrayList(f64).initCapacity(allocator, n_ts);
    defer residuals.deinit(allocator);

    for (0..n_ts) |i| {
        residuals.appendAssumeCapacity(ts_data.items[i] - smoothed.items[i]);
    }

    var residuals_nd = try NDArray(f64, 1).fromSlice(allocator, &.{n_ts}, residuals.items, .row_major);
    defer residuals_nd.deinit();
    const residual_std = try descriptive.stdDev(f64, residuals_nd, 0);
    const residual_threshold = 3.0 * residual_std;

    var ts_anomalies = try std.ArrayList(usize).initCapacity(allocator, 10);
    defer ts_anomalies.deinit(allocator);

    for (residuals.items, 0..) |residual, i| {
        if (@abs(residual) > residual_threshold) {
            try ts_anomalies.append(allocator, i);
        }
    }

    std.debug.print("Time series length: {d} samples\n", .{n_ts});
    std.debug.print("Smoothing window: {d} samples\n", .{window_size});
    std.debug.print("Residual std: {d:.2}\n", .{residual_std});
    std.debug.print("Threshold: {d:.2} (3-sigma)\n", .{residual_threshold});
    std.debug.print("Anomalies detected at indices: ", .{});
    for (ts_anomalies.items, 0..) |idx, i| {
        std.debug.print("{d}", .{idx});
        if (i < ts_anomalies.items.len - 1) std.debug.print(", ", .{});
    }
    std.debug.print("\n", .{});
    std.debug.print("Expected anomalies at: 50, 75, 100, 125, 150\n\n", .{});

    // Part 5: Multivariate Anomaly Detection (Simplified Mahalanobis Distance)
    std.debug.print("Part 5: Multivariate Anomaly Detection\n", .{});
    std.debug.print("---------------------------------------\n", .{});

    // Generate 2D data (two features) with correlation
    const n_multi = 50;
    var feature1 = try std.ArrayList(f64).initCapacity(allocator, n_multi);
    defer feature1.deinit(allocator);
    var feature2 = try std.ArrayList(f64).initCapacity(allocator, n_multi);
    defer feature2.deinit(allocator);

    const normal_x = try distributions.Normal(f64).init(0.0, 1.0);
    const normal_y = try distributions.Normal(f64).init(0.0, 1.0);

    for (0..n_multi) |i| {
        const x = normal_x.sample(random);
        // y is correlated with x (rho ≈ 0.8)
        const y = 0.8 * x + 0.6 * normal_y.sample(random);

        if (i == 10) {
            // Inject outlier far from the distribution
            feature1.appendAssumeCapacity(4.0);
            feature2.appendAssumeCapacity(-3.0);
        } else {
            feature1.appendAssumeCapacity(x);
            feature2.appendAssumeCapacity(y);
        }
    }

    // Compute means
    var feature1_nd = try NDArray(f64, 1).fromSlice(allocator, &.{n_multi}, feature1.items, .row_major);
    defer feature1_nd.deinit();
    var feature2_nd = try NDArray(f64, 1).fromSlice(allocator, &.{n_multi}, feature2.items, .row_major);
    defer feature2_nd.deinit();

    const mean_x = descriptive.mean(f64, feature1_nd);
    const mean_y = descriptive.mean(f64, feature2_nd);

    // Compute standard deviations
    const std_x = try descriptive.stdDev(f64, feature1_nd, 0);
    const std_y = try descriptive.stdDev(f64, feature2_nd, 0);

    // Compute correlation coefficient
    const corr = try zuda.stats.correlation.pearson(feature1_nd, feature2_nd, allocator);

    // Simplified Mahalanobis distance (assuming diagonal + correlation)
    // For full implementation, need inverse covariance matrix via linalg
    // Here we use standardized distance with correlation correction

    var multi_anomalies = try std.ArrayList(usize).initCapacity(allocator, 5);
    defer multi_anomalies.deinit(allocator);

    const mahalanobis_threshold = 3.0; // Chi-square approximation for 2 DOF

    for (0..n_multi) |i| {
        const z_x = (feature1.items[i] - mean_x) / std_x;
        const z_y = (feature2.items[i] - mean_y) / std_y;

        // Simplified Mahalanobis: sqrt((z_x^2 + z_y^2 - 2*rho*z_x*z_y) / (1 - rho^2))
        const numerator = z_x * z_x + z_y * z_y - 2.0 * corr * z_x * z_y;
        const denominator = 1.0 - corr * corr;
        const mahal_dist = @sqrt(numerator / denominator);

        if (mahal_dist > mahalanobis_threshold) {
            try multi_anomalies.append(allocator, i);
        }
    }

    std.debug.print("Feature 1: mean={d:.2}, std={d:.2}\n", .{ mean_x, std_x });
    std.debug.print("Feature 2: mean={d:.2}, std={d:.2}\n", .{ mean_y, std_y });
    std.debug.print("Correlation: {d:.3}\n", .{corr});
    std.debug.print("Mahalanobis threshold: {d:.1}\n", .{mahalanobis_threshold});
    std.debug.print("Anomalies detected at indices: ", .{});
    for (multi_anomalies.items, 0..) |idx, i| {
        std.debug.print("{d}", .{idx});
        if (i < multi_anomalies.items.len - 1) std.debug.print(", ", .{});
    }
    std.debug.print("\n", .{});
    std.debug.print("Expected anomaly at: 10\n\n", .{});

    // Summary
    std.debug.print("=== Summary ===\n", .{});
    std.debug.print("Methods compared:\n", .{});
    std.debug.print("  1. Z-score:     {d}/{d} anomalies (parametric, assumes normal)\n", .{ anomaly_count_zscore, n_samples });
    std.debug.print("  2. MAD:         {d}/{d} anomalies (robust, non-parametric)\n", .{ anomaly_count_mad, n_samples });
    std.debug.print("  3. IQR:         {d}/{d} anomalies (robust, quartile-based)\n", .{ anomaly_count_iqr, n_samples });
    std.debug.print("  4. Time Series: {d} anomalies (trend-based, smoothing)\n", .{ts_anomalies.items.len});
    std.debug.print("  5. Multivariate: {d} anomalies (Mahalanobis, correlation-aware)\n", .{multi_anomalies.items.len});
    std.debug.print("\nBest practices:\n", .{});
    std.debug.print("  - Z-score: Fast, works well for Gaussian data\n", .{});
    std.debug.print("  - MAD: More robust to outliers than Z-score\n", .{});
    std.debug.print("  - IQR: Simple, interpretable, non-parametric\n", .{});
    std.debug.print("  - Time Series: Use smoothing to detect deviations from trend\n", .{});
    std.debug.print("  - Multivariate: Accounts for feature correlations\n", .{});
}
