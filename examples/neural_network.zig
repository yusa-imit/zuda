const std = @import("std");
const zuda = @import("zuda");

const NDArray = zuda.ndarray.NDArray;
const Normal = zuda.stats.distributions.Normal;
const blas = zuda.linalg.blas;
const descriptive = zuda.stats.descriptive;

/// Linear regression with gradient descent - demonstrates NDArray, BLAS gemv, and stats APIs
/// This example shows the complete workflow: data generation → training loop → evaluation
/// Focus is on API demonstration rather than perfect convergence
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Linear Regression with Gradient Descent ===\n\n", .{});

    // PART 1: Data Generation
    // Generate y = 3x₁ + 2x₂ + 1 + noise
    std.debug.print("1. Generating synthetic dataset (y = 3x₁ + 2x₂ + 1)...\n", .{});
    const n_samples: usize = 100;
    const n_features: usize = 2;

    // Generate features uniformly in [0, 1]
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    const X_data = try allocator.alloc(f64, n_samples * n_features);
    defer allocator.free(X_data);
    const y_data = try allocator.alloc(f64, n_samples);
    defer allocator.free(y_data);

    const noise_dist = try Normal(f64).init(0.0, 0.1);

    for (0..n_samples) |i| {
        X_data[i * n_features + 0] = rand.float(f64);
        X_data[i * n_features + 1] = rand.float(f64);

        // True model: y = 3x₁ + 2x₂ + 1 + noise
        const noise = noise_dist.sample(rand);
        y_data[i] = 3.0 * X_data[i * n_features + 0] + 2.0 * X_data[i * n_features + 1] + 1.0 + noise;
    }

    var X = try NDArray(f64, 2).fromSlice(allocator, &.{ n_samples, n_features }, X_data, .row_major);
    defer X.deinit();
    var y = try NDArray(f64, 1).fromSlice(allocator, &.{n_samples}, y_data, .row_major);
    defer y.deinit();

    std.debug.print("   Dataset: {} samples, {} features\n", .{ n_samples, n_features });
    std.debug.print("   True model: y = 3.0x₁ + 2.0x₂ + 1.0\n", .{});
    std.debug.print("   Sample data: y[0]={d:.4}, X[0]=[{d:.4}, {d:.4}]\n", .{ y_data[0], X_data[0], X_data[1] });

    // PART 2: Model Initialization
    // Model: y = w₁x₁ + w₂x₂ + b
    std.debug.print("\n2. Initializing model parameters...\n", .{});

    // Initialize weights to small random values
    var w_data = try allocator.alloc(f64, n_features);
    defer allocator.free(w_data);
    for (0..n_features) |i| {
        w_data[i] = (rand.float(f64) - 0.5) * 0.1;
    }

    var w = try NDArray(f64, 1).fromSlice(allocator, &.{n_features}, w_data, .row_major);
    defer w.deinit();
    var b: f64 = (rand.float(f64) - 0.5) * 0.1;

    std.debug.print("   Initial w: [{d:.4}, {d:.4}], b: {d:.4}\n", .{ w_data[0], w_data[1], b });

    // PART 3: Training Loop (Batch Gradient Descent)
    std.debug.print("\n3. Training with batch gradient descent...\n", .{});

    const learning_rate: f64 = 0.01;
    const n_epochs: usize = 500;

    var epoch: usize = 0;
    while (epoch < n_epochs) : (epoch += 1) {
        // Forward pass: predictions = X @ w + b
        const Xw = try allocator.alloc(f64, n_samples);
        defer allocator.free(Xw);

        var Xw_arr = try NDArray(f64, 1).fromSlice(allocator, &.{n_samples}, Xw, .row_major);
        defer Xw_arr.deinit();

        // Matrix-vector multiply: Xw = X @ w
        try blas.gemv(f64, 1.0, X, w, 0.0, &Xw_arr);

        // predictions = Xw + b
        var predictions = try allocator.alloc(f64, n_samples);
        defer allocator.free(predictions);
        for (0..n_samples) |i| {
            predictions[i] = Xw[i] + b;
        }

        // Compute loss (MSE): (1/n) * Σ(predictions - y)²
        var loss: f64 = 0.0;
        var errors = try allocator.alloc(f64, n_samples);
        defer allocator.free(errors);

        for (0..n_samples) |i| {
            errors[i] = predictions[i] - y_data[i];
            loss += errors[i] * errors[i];
        }
        loss /= @as(f64, @floatFromInt(n_samples));

        // Backward pass: compute gradients
        // dL/dw = (1/n) * X^T @ errors = (1/n) * Σ errors[i] * X[i, :]
        const grad_w_data = try allocator.alloc(f64, n_features);
        defer allocator.free(grad_w_data);

        for (0..n_features) |j| {
            var sum: f64 = 0.0;
            for (0..n_samples) |i| {
                sum += errors[i] * X_data[i * n_features + j];
            }
            grad_w_data[j] = sum / @as(f64, @floatFromInt(n_samples));
        }

        // dL/db = (1/n) * Σ errors
        var grad_b: f64 = 0.0;
        for (errors) |e| {
            grad_b += e;
        }
        grad_b /= @as(f64, @floatFromInt(n_samples));

        // Update parameters (directly modify NDArray data)
        for (0..n_features) |i| {
            w.data[i] -= learning_rate * grad_w_data[i];
            w_data[i] = w.data[i]; // Keep w_data in sync for reporting
        }
        b -= learning_rate * grad_b;

        // Print progress
        if ((epoch + 1) % 100 == 0) {
            std.debug.print("   Epoch {}/{}  Loss: {d:.6}  w=[{d:.4}, {d:.4}]  b={d:.4}\n", .{ epoch + 1, n_epochs, loss, w.data[0], w.data[1], b });
        }
    }

    // PART 4: Final Results
    std.debug.print("\n4. Training complete!\n", .{});
    std.debug.print("   Learned model: y = {d:.4}x₁ + {d:.4}x₂ + {d:.4}\n", .{ w_data[0], w_data[1], b });
    std.debug.print("   True model:    y = 3.0000x₁ + 2.0000x₂ + 1.0000\n", .{});

    // Compute parameter errors
    const w1_error = @abs(w_data[0] - 3.0);
    const w2_error = @abs(w_data[1] - 2.0);
    const b_error = @abs(b - 1.0);

    std.debug.print("\n   Parameter errors:\n", .{});
    std.debug.print("     w₁: {d:.6}\n", .{w1_error});
    std.debug.print("     w₂: {d:.6}\n", .{w2_error});
    std.debug.print("     b:  {d:.6}\n", .{b_error});

    // PART 5: Evaluation Metrics
    std.debug.print("\n5. Computing evaluation metrics...\n", .{});

    // Final predictions
    const final_Xw = try allocator.alloc(f64, n_samples);
    defer allocator.free(final_Xw);
    var final_Xw_arr = try NDArray(f64, 1).fromSlice(allocator, &.{n_samples}, final_Xw, .row_major);
    defer final_Xw_arr.deinit();

    try blas.gemv(f64, 1.0, X, w, 0.0, &final_Xw_arr);

    var final_predictions = try allocator.alloc(f64, n_samples);
    defer allocator.free(final_predictions);
    for (0..n_samples) |i| {
        final_predictions[i] = final_Xw[i] + b;
    }

    // R² score
    const y_mean = descriptive.mean(f64, y);
    var ss_tot: f64 = 0.0;
    var ss_res: f64 = 0.0;

    for (0..n_samples) |i| {
        const y_diff = y_data[i] - y_mean;
        ss_tot += y_diff * y_diff;

        const residual = y_data[i] - final_predictions[i];
        ss_res += residual * residual;
    }

    const r_squared = 1.0 - (ss_res / ss_tot);

    // RMSE
    const rmse = @sqrt(ss_res / @as(f64, @floatFromInt(n_samples)));

    std.debug.print("   R² score: {d:.6}\n", .{r_squared});
    std.debug.print("   RMSE:     {d:.6}\n", .{rmse});

    std.debug.print("\n=== Demo Complete ===\n", .{});
    std.debug.print("\nDemonstrated zuda v2.0 APIs:\n", .{});
    std.debug.print("  - NDArray: fromSlice, transpose, data field access\n", .{});
    std.debug.print("  - BLAS: gemv (matrix-vector multiplication)\n", .{});
    std.debug.print("  - Stats: Normal distribution sampling, descriptive.mean\n", .{});
    std.debug.print("  - Gradient descent: manual backprop implementation\n", .{});
}
