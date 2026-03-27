//! Machine Learning Pipeline Example
//!
//! This example demonstrates a complete supervised learning pipeline using zuda:
//! 1. Data generation (synthetic dataset)
//! 2. Train/test split
//! 3. Feature normalization (standardization)
//! 4. Model training (linear regression via least squares)
//! 5. Prediction on test set
//! 6. Model evaluation (R², RMSE, MAE)
//! 7. Residual analysis
//!
//! Build and run:
//! ```bash
//! zig build example-ml-pipeline
//! ```

const std = @import("std");
const zuda = @import("zuda");

const NDArray = zuda.ndarray.NDArray;
const descriptive = zuda.stats.descriptive;
const correlation = zuda.stats.correlation;
const distributions = zuda.stats.distributions;
const linalg = zuda.linalg;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Machine Learning Pipeline with zuda ===\n\n", .{});

    // ========================================================================
    // Step 1: Generate synthetic dataset
    // ========================================================================
    std.debug.print("Step 1: Generating synthetic dataset...\n", .{});

    const n_samples: usize = 200;
    const n_features: usize = 3;

    // Generate features: X ~ N(0, 1)
    var rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const random = rng.random();

    const X_data = try allocator.alloc(f64, n_samples * n_features);
    defer allocator.free(X_data);

    const normal = try distributions.Normal(f64).init(0.0, 1.0);
    for (X_data) |*val| {
        val.* = normal.sample(random);
    }

    var X = try NDArray(f64, 2).fromSlice(allocator, &.{ n_samples, n_features }, X_data, .row_major);
    defer X.deinit();

    // Generate target: y = 2*x1 - 3*x2 + 0.5*x3 + 1 + noise
    const true_weights = [_]f64{ 2.0, -3.0, 0.5 };
    const true_intercept: f64 = 1.0;

    var y_data = try allocator.alloc(f64, n_samples);
    defer allocator.free(y_data);

    for (0..n_samples) |i| {
        var sum: f64 = true_intercept;
        for (0..n_features) |j| {
            const x_val = try X.get(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) });
            sum += true_weights[j] * x_val;
        }
        // Add Gaussian noise
        const noise = normal.sample(random) * 0.5;
        y_data[i] = sum + noise;
    }

    var y = try NDArray(f64, 1).fromSlice(allocator, &.{n_samples}, y_data, .row_major);
    defer y.deinit();

    std.debug.print("  Generated {d} samples with {d} features\n", .{ n_samples, n_features });
    std.debug.print("  True model: y = {d:.2}*x1 + {d:.2}*x2 + {d:.2}*x3 + {d:.2}\n\n",
        .{ true_weights[0], true_weights[1], true_weights[2], true_intercept });

    // ========================================================================
    // Step 2: Train/test split (80/20)
    // ========================================================================
    std.debug.print("Step 2: Splitting data into train/test sets...\n", .{});

    const n_train = (n_samples * 80) / 100;
    const n_test = n_samples - n_train;

    var X_train_data = try allocator.alloc(f64, n_train * n_features);
    defer allocator.free(X_train_data);
    var X_test_data = try allocator.alloc(f64, n_test * n_features);
    defer allocator.free(X_test_data);

    for (0..n_train) |i| {
        for (0..n_features) |j| {
            X_train_data[i * n_features + j] = try X.get(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) });
        }
    }

    for (0..n_test) |i| {
        for (0..n_features) |j| {
            X_test_data[i * n_features + j] = try X.get(&.{ @as(isize, @intCast(n_train + i)), @as(isize, @intCast(j)) });
        }
    }

    var X_train = try NDArray(f64, 2).fromSlice(allocator, &.{ n_train, n_features }, X_train_data, .row_major);
    defer X_train.deinit();
    var X_test = try NDArray(f64, 2).fromSlice(allocator, &.{ n_test, n_features }, X_test_data, .row_major);
    defer X_test.deinit();

    var y_train_data = try allocator.alloc(f64, n_train);
    defer allocator.free(y_train_data);
    var y_test_data = try allocator.alloc(f64, n_test);
    defer allocator.free(y_test_data);

    for (0..n_train) |i| {
        y_train_data[i] = try y.get(&.{@as(isize, @intCast(i))});
    }
    for (0..n_test) |i| {
        y_test_data[i] = try y.get(&.{@as(isize, @intCast(n_train + i))});
    }

    var y_train = try NDArray(f64, 1).fromSlice(allocator, &.{n_train}, y_train_data, .row_major);
    defer y_train.deinit();
    var y_test = try NDArray(f64, 1).fromSlice(allocator, &.{n_test}, y_test_data, .row_major);
    defer y_test.deinit();

    std.debug.print("  Train set: {d} samples\n", .{n_train});
    std.debug.print("  Test set:  {d} samples\n\n", .{n_test});

    // ========================================================================
    // Step 3: Feature standardization (Z-score normalization)
    // ========================================================================
    std.debug.print("Step 3: Standardizing features...\n", .{});

    // Compute mean and std for each feature on training set
    var feature_means = try allocator.alloc(f64, n_features);
    defer allocator.free(feature_means);
    var feature_stds = try allocator.alloc(f64, n_features);
    defer allocator.free(feature_stds);

    for (0..n_features) |j| {
        var col_data = try allocator.alloc(f64, n_train);
        defer allocator.free(col_data);

        for (0..n_train) |i| {
            col_data[i] = try X_train.get(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) });
        }

        var col_array = try NDArray(f64, 1).fromSlice(allocator, &.{n_train}, col_data, .row_major);
        defer col_array.deinit();

        feature_means[j] = descriptive.mean(f64, col_array);
        feature_stds[j] = try descriptive.stdDev(f64, col_array, 0);
    }

    // Standardize train and test sets using training statistics
    for (0..n_train) |i| {
        for (0..n_features) |j| {
            const val = try X_train.get(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) });
            const standardized = (val - feature_means[j]) / feature_stds[j];
            X_train.set(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) }, standardized);
        }
    }

    for (0..n_test) |i| {
        for (0..n_features) |j| {
            const val = try X_test.get(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) });
            const standardized = (val - feature_means[j]) / feature_stds[j];
            X_test.set(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) }, standardized);
        }
    }

    std.debug.print("  Features standardized (mean=0, std=1)\n\n", .{});

    // ========================================================================
    // Step 4: Model training (Ordinary Least Squares)
    // ========================================================================
    std.debug.print("Step 4: Training linear regression model...\n", .{});

    // Add intercept column to X (column of ones)
    var X_train_aug_data = try allocator.alloc(f64, n_train * (n_features + 1));
    defer allocator.free(X_train_aug_data);

    for (0..n_train) |i| {
        X_train_aug_data[i * (n_features + 1)] = 1.0; // intercept
        for (0..n_features) |j| {
            X_train_aug_data[i * (n_features + 1) + j + 1] =
                try X_train.get(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) });
        }
    }

    var X_train_aug = try NDArray(f64, 2).fromSlice(allocator,
        &.{ n_train, n_features + 1 }, X_train_aug_data, .row_major);
    defer X_train_aug.deinit();

    // Solve normal equations: (X^T X) β = X^T y
    var coeffs = try linalg.solve.lstsq(f64, X_train_aug, y_train, allocator);
    defer coeffs.deinit();

    std.debug.print("  Learned coefficients:\n", .{});
    std.debug.print("    Intercept: {d:.4}\n", .{try coeffs.get(&.{0})});
    for (0..n_features) |j| {
        std.debug.print("    Feature {d}: {d:.4}\n", .{ j + 1, try coeffs.get(&.{@as(isize, @intCast(j + 1))}) });
    }
    std.debug.print("\n", .{});

    // ========================================================================
    // Step 5: Prediction on test set
    // ========================================================================
    std.debug.print("Step 5: Making predictions on test set...\n", .{});

    var X_test_aug_data = try allocator.alloc(f64, n_test * (n_features + 1));
    defer allocator.free(X_test_aug_data);

    for (0..n_test) |i| {
        X_test_aug_data[i * (n_features + 1)] = 1.0;
        for (0..n_features) |j| {
            X_test_aug_data[i * (n_features + 1) + j + 1] =
                try X_test.get(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) });
        }
    }

    var X_test_aug = try NDArray(f64, 2).fromSlice(allocator,
        &.{ n_test, n_features + 1 }, X_test_aug_data, .row_major);
    defer X_test_aug.deinit();

    // y_pred = X @ coeffs
    var y_pred_data = try allocator.alloc(f64, n_test);
    defer allocator.free(y_pred_data);

    for (0..n_test) |i| {
        var pred: f64 = 0.0;
        for (0..(n_features + 1)) |j| {
            pred += try X_test_aug.get(&.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) }) *
                    try coeffs.get(&.{@as(isize, @intCast(j))});
        }
        y_pred_data[i] = pred;
    }

    var y_pred = try NDArray(f64, 1).fromSlice(allocator, &.{n_test}, y_pred_data, .row_major);
    defer y_pred.deinit();

    std.debug.print("  Predictions generated for {d} test samples\n\n", .{n_test});

    // ========================================================================
    // Step 6: Model evaluation
    // ========================================================================
    std.debug.print("Step 6: Evaluating model performance...\n", .{});

    // Compute R² score
    const y_test_mean = descriptive.mean(f64, y_test);

    var ss_tot: f64 = 0.0;
    var ss_res: f64 = 0.0;
    for (0..n_test) |i| {
        const y_true = y_test_data[i];
        const y_hat = y_pred_data[i];
        ss_tot += (y_true - y_test_mean) * (y_true - y_test_mean);
        ss_res += (y_true - y_hat) * (y_true - y_hat);
    }

    const r_squared = 1.0 - (ss_res / ss_tot);

    // Compute RMSE and MAE
    var mse: f64 = 0.0;
    var mae: f64 = 0.0;
    for (0..n_test) |i| {
        const err = y_test_data[i] - y_pred_data[i];
        mse += err * err;
        mae += @abs(err);
    }
    mse /= @as(f64, @floatFromInt(n_test));
    mae /= @as(f64, @floatFromInt(n_test));
    const rmse = @sqrt(mse);

    std.debug.print("  R² score:  {d:.4} (1.0 = perfect fit)\n", .{r_squared});
    std.debug.print("  RMSE:      {d:.4}\n", .{rmse});
    std.debug.print("  MAE:       {d:.4}\n\n", .{mae});

    // ========================================================================
    // Step 7: Residual analysis
    // ========================================================================
    std.debug.print("Step 7: Analyzing residuals...\n", .{});

    var residuals_data = try allocator.alloc(f64, n_test);
    defer allocator.free(residuals_data);

    for (0..n_test) |i| {
        residuals_data[i] = y_test_data[i] - y_pred_data[i];
    }

    var residuals = try NDArray(f64, 1).fromSlice(allocator, &.{n_test}, residuals_data, .row_major);
    defer residuals.deinit();

    const residual_mean = descriptive.mean(f64, residuals);
    const residual_std = try descriptive.stdDev(f64, residuals, 0);

    std.debug.print("  Residual mean: {d:.6} (should be ~0)\n", .{residual_mean});
    std.debug.print("  Residual std:  {d:.4}\n", .{residual_std});

    // Check for normality (simple skewness check)
    var sum_cubed_dev: f64 = 0.0;
    for (residuals_data) |r| {
        const dev = r - residual_mean;
        sum_cubed_dev += dev * dev * dev;
    }
    const skewness = sum_cubed_dev / (@as(f64, @floatFromInt(n_test)) * residual_std * residual_std * residual_std);

    std.debug.print("  Residual skewness: {d:.4} (|skew| < 0.5 suggests normality)\n", .{skewness});

    std.debug.print("\n=== Pipeline Complete ===\n", .{});
    std.debug.print("The model successfully learned the underlying relationship!\n", .{});
}
