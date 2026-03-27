const std = @import("std");
const zuda = @import("zuda");

const NDArray = zuda.ndarray.NDArray;
const descriptive = zuda.stats.descriptive;
const distributions = zuda.stats.distributions;

/// Kalman Filter Example (Simplified 1D Implementation)
///
/// Demonstrates:
/// 1. State estimation for dynamic systems
/// 2. Probabilistic modeling with Gaussian noise
/// 3. Prediction-update cycle
/// 4. Statistical analysis of estimation quality
///
/// System: 1D position tracking with constant velocity
///   State: [position, velocity]
///   Process: position(k+1) = position(k) + velocity(k)*dt
///           velocity(k+1) = velocity(k)
///   Measurement: observe position only (with noise)
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Kalman Filter Demonstration (1D Tracking) ===\n\n", .{});

    // Part 1: System Configuration
    std.debug.print("Part 1: System Setup\n", .{});
    std.debug.print("--------------------\n", .{});

    const dt = 0.1; // Time step: 100ms
    const n_steps = 100;

    // Process noise (uncertainty in motion model)
    const q_pos = 0.01; // Position process noise variance
    const q_vel = 0.01; // Velocity process noise variance

    // Measurement noise (sensor uncertainty)
    const r_pos = 0.5; // Position measurement noise variance

    std.debug.print("Time step dt: {d:.3}s\n", .{dt});
    std.debug.print("Process noise: pos={d:.3}, vel={d:.3}\n", .{ q_pos, q_vel });
    std.debug.print("Measurement noise: {d:.2}\n\n", .{r_pos});

    // Part 2: Generate True Trajectory
    std.debug.print("Part 2: Generate True Trajectory\n", .{});
    std.debug.print("---------------------------------\n", .{});

    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();

    // True initial state
    var true_pos: f64 = 0.0;
    var true_vel: f64 = 1.0; // 1 m/s constant velocity

    var true_positions = try std.ArrayList(f64).initCapacity(allocator, n_steps);
    defer true_positions.deinit(allocator);
    var measurements = try std.ArrayList(f64).initCapacity(allocator, n_steps);
    defer measurements.deinit(allocator);

    const process_noise_pos = try distributions.Normal(f64).init(0.0, @sqrt(q_pos));
    const process_noise_vel = try distributions.Normal(f64).init(0.0, @sqrt(q_vel));
    const measurement_noise = try distributions.Normal(f64).init(0.0, @sqrt(r_pos));

    // Generate trajectory
    for (0..n_steps) |_| {
        // True dynamics with process noise
        true_pos = true_pos + dt * true_vel + process_noise_pos.sample(random);
        true_vel = true_vel + process_noise_vel.sample(random);

        true_positions.appendAssumeCapacity(true_pos);

        // Noisy measurement of position
        const z = true_pos + measurement_noise.sample(random);
        measurements.appendAssumeCapacity(z);
    }

    std.debug.print("Generated {d} time steps\n", .{n_steps});
    std.debug.print("True initial: pos={d:.2}, vel={d:.2}\n", .{ 0.0, 1.0 });
    std.debug.print("Sample measurements: {d:.2}, {d:.2}, {d:.2}, ...\n\n", .{ measurements.items[0], measurements.items[1], measurements.items[2] });

    // Part 3: Kalman Filter
    std.debug.print("Part 3: Kalman Filter Estimation\n", .{});
    std.debug.print("----------------------------------\n", .{});

    // State: [position, velocity]
    var x_pos = measurements.items[0]; // Initialize from first measurement
    var x_vel: f64 = 0.0; // Assume stationary initially

    // Covariance matrix P (2x2) - diagonal for simplicity
    var p_00: f64 = 1.0; // Position variance
    var p_01: f64 = 0.0; // Position-velocity covariance
    var p_10: f64 = 0.0; // Velocity-position covariance
    var p_11: f64 = 1.0; // Velocity variance

    var estimated_positions = try std.ArrayList(f64).initCapacity(allocator, n_steps);
    defer estimated_positions.deinit(allocator);
    var estimated_velocities = try std.ArrayList(f64).initCapacity(allocator, n_steps);
    defer estimated_velocities.deinit(allocator);

    estimated_positions.appendAssumeCapacity(x_pos);
    estimated_velocities.appendAssumeCapacity(x_vel);

    // Kalman filter loop
    for (1..n_steps) |k| {
        // PREDICTION STEP
        // State prediction: x_minus = F * x
        const x_pos_minus = x_pos + dt * x_vel;
        const x_vel_minus = x_vel;

        // Covariance prediction: P_minus = F * P * F^T + Q
        // F = [[1, dt], [0, 1]]
        const p_00_minus = p_00 + dt * (p_01 + p_10) + dt * dt * p_11 + q_pos;
        const p_01_minus = p_01 + dt * p_11;
        const p_10_minus = p_10 + dt * p_11;
        const p_11_minus = p_11 + q_vel;

        // UPDATE STEP
        // Innovation: y = z - H * x_minus (H = [1, 0])
        const z = measurements.items[k];
        const innovation = z - x_pos_minus;

        // Innovation covariance: S = H * P_minus * H^T + R
        const S = p_00_minus + r_pos;

        // Kalman gain: K = P_minus * H^T * S^(-1)
        const k_0 = p_00_minus / S; // Gain for position
        const k_1 = p_10_minus / S; // Gain for velocity

        // State update: x = x_minus + K * innovation
        x_pos = x_pos_minus + k_0 * innovation;
        x_vel = x_vel_minus + k_1 * innovation;

        // Covariance update: P = (I - K*H) * P_minus
        // I - K*H = [[1-k_0, 0], [-k_1, 1]]
        p_00 = (1.0 - k_0) * p_00_minus;
        p_01 = (1.0 - k_0) * p_01_minus;
        p_10 = p_10_minus - k_1 * p_00_minus;
        p_11 = p_11_minus - k_1 * p_01_minus;

        estimated_positions.appendAssumeCapacity(x_pos);
        estimated_velocities.appendAssumeCapacity(x_vel);
    }

    std.debug.print("Kalman filter completed {d} iterations\n", .{n_steps});
    std.debug.print("Final estimate: pos={d:.2}, vel={d:.2}\n", .{ x_pos, x_vel });
    std.debug.print("Final uncertainty: σ_pos={d:.4}, σ_vel={d:.4}\n\n", .{ @sqrt(p_00), @sqrt(p_11) });

    // Part 4: Performance Evaluation
    std.debug.print("Part 4: Evaluation\n", .{});
    std.debug.print("------------------\n", .{});

    // Position RMSE
    var pos_errors = try std.ArrayList(f64).initCapacity(allocator, n_steps);
    defer pos_errors.deinit(allocator);
    for (0..n_steps) |i| {
        const err = estimated_positions.items[i] - true_positions.items[i];
        pos_errors.appendAssumeCapacity(err * err);
    }
    var pos_errors_nd = try NDArray(f64, 1).fromSlice(allocator, &.{n_steps}, pos_errors.items, .row_major);
    defer pos_errors_nd.deinit();
    const pos_rmse = @sqrt(descriptive.mean(f64, pos_errors_nd));

    // Measurement RMSE (for comparison)
    var meas_errors = try std.ArrayList(f64).initCapacity(allocator, n_steps);
    defer meas_errors.deinit(allocator);
    for (0..n_steps) |i| {
        const err = measurements.items[i] - true_positions.items[i];
        meas_errors.appendAssumeCapacity(err * err);
    }
    var meas_errors_nd = try NDArray(f64, 1).fromSlice(allocator, &.{n_steps}, meas_errors.items, .row_major);
    defer meas_errors_nd.deinit();
    const measurement_rmse = @sqrt(descriptive.mean(f64, meas_errors_nd));

    std.debug.print("Position RMSE:\n", .{});
    std.debug.print("  Raw measurements: {d:.3}m\n", .{measurement_rmse});
    std.debug.print("  Kalman estimates: {d:.3}m\n", .{pos_rmse});
    std.debug.print("  Improvement: {d:.1}%\n\n", .{(1.0 - pos_rmse / measurement_rmse) * 100.0});

    // Part 5: Visualization
    std.debug.print("Part 5: Trajectory Visualization (first 50 steps)\n", .{});
    std.debug.print("--------------------------------------------------\n", .{});

    const plot_steps = @min(50, n_steps);
    const plot_height = 20;

    // Find range
    var min_pos: f64 = true_positions.items[0];
    var max_pos: f64 = true_positions.items[0];
    for (0..plot_steps) |i| {
        min_pos = @min(min_pos, @min(true_positions.items[i], @min(measurements.items[i], estimated_positions.items[i])));
        max_pos = @max(max_pos, @max(true_positions.items[i], @max(measurements.items[i], estimated_positions.items[i])));
    }

    const pos_range = max_pos - min_pos;
    const scale = (plot_height - 1) / pos_range;

    // Create plot grid
    var plot = try allocator.alloc([]u8, plot_height);
    defer {
        for (plot) |row| allocator.free(row);
        allocator.free(plot);
    }

    for (0..plot_height) |i| {
        plot[i] = try allocator.alloc(u8, plot_steps);
        @memset(plot[i], ' ');
    }

    // Plot true trajectory
    for (0..plot_steps) |i| {
        const y = @as(usize, @intFromFloat((true_positions.items[i] - min_pos) * scale));
        const y_inv = plot_height - 1 - @min(y, plot_height - 1);
        plot[y_inv][i] = '*';
    }

    // Plot measurements
    for (0..plot_steps) |i| {
        const y = @as(usize, @intFromFloat((measurements.items[i] - min_pos) * scale));
        const y_inv = plot_height - 1 - @min(y, plot_height - 1);
        if (plot[y_inv][i] == ' ') {
            plot[y_inv][i] = '.';
        }
    }

    // Plot Kalman estimates
    for (0..plot_steps) |i| {
        const y = @as(usize, @intFromFloat((estimated_positions.items[i] - min_pos) * scale));
        const y_inv = plot_height - 1 - @min(y, plot_height - 1);
        if (plot[y_inv][i] == ' ') {
            plot[y_inv][i] = 'K';
        } else if (plot[y_inv][i] == '*') {
            plot[y_inv][i] = '@'; // Perfect match
        }
    }

    std.debug.print("Legend: * = true, . = measured, K = Kalman, @ = exact\n", .{});
    std.debug.print("Range: {d:.2} to {d:.2}m\n\n", .{ min_pos, max_pos });

    for (plot, 0..) |row, i| {
        const pos_val = max_pos - @as(f64, @floatFromInt(i)) * (pos_range / @as(f64, @floatFromInt(plot_height - 1)));
        std.debug.print("{d:>5.1} |{s}\n", .{ pos_val, row });
    }

    std.debug.print("      +{s}\n", .{"-" ** plot_steps});
    std.debug.print("       Time (steps)\n\n", .{});

    std.debug.print("=== Summary ===\n", .{});
    std.debug.print("Kalman filter successfully tracked 1D motion,\n", .{});
    std.debug.print("reducing position error by {d:.1}% compared to raw measurements.\n", .{(1.0 - pos_rmse / measurement_rmse) * 100.0});
    std.debug.print("Velocity was inferred despite no direct measurements.\n", .{});
}
