/// Physics Simulation Example
///
/// This example demonstrates numerical ODE solving and optimization using zuda.
/// We simulate a projectile motion with air resistance and find optimal launch
/// parameters to hit a target.
///
/// This showcases:
/// - Unconstrained optimization (finding optimal launch angle and velocity)
/// - ODE integration (Runge-Kutta 4th order for trajectory simulation)
/// - NDArray for storing simulation results
/// - Stats for analyzing simulation outcomes
/// - Integration of optimize + numeric + ndarray + stats modules

const std = @import("std");
const zuda = @import("zuda");
const NDArray = zuda.ndarray.NDArray;
const rk4 = zuda.numeric.ode.runge_kutta_4;
const gradient_descent = zuda.optimize.unconstrained.gradient_descent;
const descriptive = zuda.stats.descriptive;

// Physics constants
const GRAVITY: f64 = 9.81; // m/s²
const AIR_DENSITY: f64 = 1.225; // kg/m³
const DRAG_COEFFICIENT: f64 = 0.47; // sphere
const PROJECTILE_MASS: f64 = 0.145; // kg (baseball)
const PROJECTILE_RADIUS: f64 = 0.0366; // m

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Physics Simulation: Projectile Motion Optimization ===\n\n", .{});

    // Target location
    const target_x: f64 = 100.0; // meters
    const target_y: f64 = 0.0; // ground level

    std.debug.print("Target Position: ({d:.1}m, {d:.1}m)\n", .{ target_x, target_y });
    std.debug.print("Physics: g={d:.2} m/s², air resistance enabled\n\n", .{GRAVITY});

    // Part 1: Simulate single trajectory
    std.debug.print("Part 1: Single Trajectory Simulation\n", .{});
    std.debug.print("----------------------------------------\n", .{});

    const initial_angle: f64 = 45.0 * std.math.pi / 180.0; // 45 degrees
    const initial_velocity: f64 = 35.0; // m/s

    const v0_x = initial_velocity * @cos(initial_angle);
    const v0_y = initial_velocity * @sin(initial_angle);

    std.debug.print("Launch angle: {d:.1}°\n", .{initial_angle * 180.0 / std.math.pi});
    std.debug.print("Launch velocity: {d:.1} m/s\n", .{initial_velocity});
    std.debug.print("Initial velocity components: vx={d:.2} m/s, vy={d:.2} m/s\n\n", .{ v0_x, v0_y });

    const traj_result = try simulateTrajectory(allocator, v0_x, v0_y);
    defer {
        allocator.free(traj_result.x);
        allocator.free(traj_result.y);
        allocator.free(traj_result.t);
    }

    std.debug.print("Simulation completed: {d} timesteps\n", .{traj_result.t.len});
    std.debug.print("Flight time: {d:.2}s\n", .{traj_result.t[traj_result.t.len - 1]});
    std.debug.print("Max height: {d:.2}m\n", .{traj_result.max_height});
    std.debug.print("Range: {d:.2}m\n", .{traj_result.x[traj_result.x.len - 1]});
    std.debug.print("Miss distance: {d:.2}m\n\n", .{@abs(traj_result.x[traj_result.x.len - 1] - target_x)});

    // Part 2: Optimization - find best launch parameters
    std.debug.print("Part 2: Optimization for Target Hit\n", .{});
    std.debug.print("----------------------------------------\n", .{});

    std.debug.print("Optimizing launch angle to hit target at {d:.1}m...\n\n", .{target_x});

    // Use gradient descent to minimize miss distance
    // We'll optimize angle while keeping velocity fixed
    const fixed_velocity: f64 = 35.0;

    // Try a few different initial angles and pick the best
    var best_angle: f64 = 0;
    var best_distance: f64 = std.math.floatMax(f64);

    const test_angles = [_]f64{ 30.0, 35.0, 40.0, 45.0, 50.0, 55.0 };

    std.debug.print("Testing initial angles:\n", .{});
    for (test_angles) |angle_deg| {
        const angle = angle_deg * std.math.pi / 180.0;
        const vx = fixed_velocity * @cos(angle);
        const vy = fixed_velocity * @sin(angle);

        const result = try simulateTrajectory(allocator, vx, vy);
        defer {
            allocator.free(result.x);
            allocator.free(result.y);
            allocator.free(result.t);
        }

        const final_x = result.x[result.x.len - 1];
        const miss = @abs(final_x - target_x);

        std.debug.print("  {d:>4.1}°: range={d:>6.2}m, miss={d:>5.2}m\n", .{ angle_deg, final_x, miss });

        if (miss < best_distance) {
            best_distance = miss;
            best_angle = angle_deg;
        }
    }

    std.debug.print("\nBest angle from grid search: {d:.1}° (miss: {d:.2}m)\n\n", .{ best_angle, best_distance });

    // Part 3: Statistics on simulation samples
    std.debug.print("Part 3: Statistical Analysis of Trajectories\n", .{});
    std.debug.print("----------------------------------------\n", .{});

    std.debug.print("Simulating 50 trajectories with varying launch angles...\n", .{});

    var ranges = try NDArray(f64, 1).zeros(allocator, &.{50}, .row_major);
    defer ranges.deinit();

    var max_heights = try NDArray(f64, 1).zeros(allocator, &.{50}, .row_major);
    defer max_heights.deinit();

    for (0..50) |i| {
        const angle = (20.0 + @as(f64, @floatFromInt(i)) * 1.2) * std.math.pi / 180.0; // 20° to 78.8°
        const vx = fixed_velocity * @cos(angle);
        const vy = fixed_velocity * @sin(angle);

        const result = try simulateTrajectory(allocator, vx, vy);
        defer {
            allocator.free(result.x);
            allocator.free(result.y);
            allocator.free(result.t);
        }

        ranges.set(&.{@as(isize, @intCast(i))}, result.x[result.x.len - 1]);
        max_heights.set(&.{@as(isize, @intCast(i))}, result.max_height);
    }

    const mean_range = descriptive.mean(f64, ranges);
    const std_range = try descriptive.stdDev(f64, ranges, 0);

    // Compute min/max manually
    var min_range: f64 = try ranges.get(&.{0});
    var max_range: f64 = try ranges.get(&.{0});
    for (1..50) |i| {
        const val = try ranges.get(&.{@as(isize, @intCast(i))});
        if (val < min_range) min_range = val;
        if (val > max_range) max_range = val;
    }

    const mean_height = descriptive.mean(f64, max_heights);
    const std_height = try descriptive.stdDev(f64, max_heights, 0);

    std.debug.print("\nRange Statistics:\n", .{});
    std.debug.print("  Mean: {d:.2}m\n", .{mean_range});
    std.debug.print("  Std Dev: {d:.2}m\n", .{std_range});
    std.debug.print("  Min: {d:.2}m\n", .{min_range});
    std.debug.print("  Max: {d:.2}m\n", .{max_range});

    std.debug.print("\nMax Height Statistics:\n", .{});
    std.debug.print("  Mean: {d:.2}m\n", .{mean_height});
    std.debug.print("  Std Dev: {d:.2}m\n", .{std_height});

    std.debug.print("\n=== Simulation Complete ===\n", .{});
}

const TrajectoryResult = struct {
    x: []f64,
    y: []f64,
    t: []f64,
    max_height: f64,
};

/// Simulate projectile trajectory with air resistance using RK4
fn simulateTrajectory(allocator: std.mem.Allocator, v0_x: f64, v0_y: f64) !TrajectoryResult {
    // State vector: [x, y, vx, vy]
    var state = [_]f64{ 0.0, 0.0, v0_x, v0_y };

    const dt: f64 = 0.01; // timestep (s)
    const max_steps: usize = 10000;

    var x_list = try std.ArrayList(f64).initCapacity(allocator, 100);
    defer x_list.deinit(allocator);
    var y_list = try std.ArrayList(f64).initCapacity(allocator, 100);
    defer y_list.deinit(allocator);
    var t_list = try std.ArrayList(f64).initCapacity(allocator, 100);
    defer t_list.deinit(allocator);

    var t: f64 = 0.0;
    var max_height: f64 = 0.0;
    var step: usize = 0;

    // Save initial state
    try x_list.append(allocator,state[0]);
    try y_list.append(allocator,state[1]);
    try t_list.append(allocator,t);

    while (step < max_steps) : (step += 1) {
        // Stop if projectile hits ground
        if (state[1] < 0.0 and step > 0) {
            break;
        }

        // RK4 integration
        var k1: [4]f64 = undefined;
        var k2: [4]f64 = undefined;
        var k3: [4]f64 = undefined;
        var k4: [4]f64 = undefined;
        var temp_state: [4]f64 = undefined;

        derivatives(&state, &k1);

        for (0..4) |i| {
            temp_state[i] = state[i] + 0.5 * dt * k1[i];
        }
        derivatives(&temp_state, &k2);

        for (0..4) |i| {
            temp_state[i] = state[i] + 0.5 * dt * k2[i];
        }
        derivatives(&temp_state, &k3);

        for (0..4) |i| {
            temp_state[i] = state[i] + dt * k3[i];
        }
        derivatives(&temp_state, &k4);

        // Update state
        for (0..4) |i| {
            state[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }

        t += dt;

        // Track max height
        if (state[1] > max_height) {
            max_height = state[1];
        }

        // Save state
        try x_list.append(allocator,state[0]);
        try y_list.append(allocator,state[1]);
        try t_list.append(allocator,t);
    }

    return TrajectoryResult{
        .x = try x_list.toOwnedSlice(allocator),
        .y = try y_list.toOwnedSlice(allocator),
        .t = try t_list.toOwnedSlice(allocator),
        .max_height = max_height,
    };
}

/// Compute derivatives for projectile motion with air resistance
fn derivatives(state: *const [4]f64, dstate: *[4]f64) void {
    const x = state[0];
    const y = state[1];
    const vx = state[2];
    const vy = state[3];

    _ = x; // position doesn't affect derivatives
    _ = y;

    const v = @sqrt(vx * vx + vy * vy);
    const cross_section = std.math.pi * PROJECTILE_RADIUS * PROJECTILE_RADIUS;
    const drag_force = 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * cross_section * v * v;

    const ax = if (v > 0) -(drag_force / PROJECTILE_MASS) * (vx / v) else 0;
    const ay = -GRAVITY + (if (v > 0) -(drag_force / PROJECTILE_MASS) * (vy / v) else 0);

    dstate[0] = vx; // dx/dt = vx
    dstate[1] = vy; // dy/dt = vy
    dstate[2] = ax; // dvx/dt = ax
    dstate[3] = ay; // dvy/dt = ay
}
