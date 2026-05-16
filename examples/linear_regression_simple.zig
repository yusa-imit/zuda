const std = @import("std");
const zuda = @import("zuda");

/// Simple Linear Regression Example
///
/// Demonstrates using zuda's stats module to perform ordinary least squares
/// regression (y = mx + b) with synthetic data.
///
/// This example:
/// 1. Creates synthetic data with known slope and intercept + noise
/// 2. Uses stats.regression.polyfit() for least squares fitting
/// 3. Displays the fitted coefficients, R² score, and prediction errors
/// 4. Shows how to make predictions on new data points
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll("=== Linear Regression with zuda ===\n\n");

    // Generate synthetic data: y = 2.5x + 1.0 + noise
    const true_slope = 2.5;
    const true_intercept = 1.0;
    const n_points = 50;

    var x_data = try allocator.alloc(f64, n_points);
    defer allocator.free(x_data);
    var y_data = try allocator.alloc(f64, n_points);
    defer allocator.free(y_data);

    // Generate data points
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (0..n_points) |i| {
        x_data[i] = @as(f64, @floatFromInt(i)) / 10.0; // x from 0 to 4.9
        const noise = (random.float(f64) - 0.5) * 0.5; // noise ±0.25
        y_data[i] = true_slope * x_data[i] + true_intercept + noise;
    }

    try stdout.print("Generated {} data points\n", .{n_points});
    try stdout.print("True model: y = {d:.2}x + {d:.2}\n\n", .{ true_slope, true_intercept });

    // Fit linear model using polyfit (degree=1 for linear)
    const result = try zuda.stats.regression.polyfit(f64, allocator, x_data, y_data, 1);
    defer result.deinit(allocator);

    // result.coefficients[0] = intercept, result.coefficients[1] = slope
    const fitted_intercept = result.coefficients[0];
    const fitted_slope = result.coefficients[1];

    try stdout.print("Fitted model: y = {d:.4}x + {d:.4}\n", .{ fitted_slope, fitted_intercept });
    try stdout.print("Error: slope = {d:.4}, intercept = {d:.4}\n", .{
        @abs(fitted_slope - true_slope),
        @abs(fitted_intercept - true_intercept),
    });
    try stdout.print("R² score: {d:.6} (closer to 1.0 = better fit)\n\n", .{result.r_squared});

    // Make predictions on new data
    const new_x = [_]f64{ 0.0, 2.5, 5.0 };
    try stdout.writeAll("Predictions:\n");
    for (new_x) |x| {
        const y_pred_val = fitted_slope * x + fitted_intercept;
        const y_true = true_slope * x + true_intercept;
        try stdout.print("  x = {d:.1}: predicted = {d:.4}, true (no noise) = {d:.4}\n",
            .{ x, y_pred_val, y_true });
    }

    try stdout.writeAll("\n✓ Example complete!\n");
}
