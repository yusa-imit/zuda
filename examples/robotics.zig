//! # Robotics & Motion Planning
//!
//! This example demonstrates robotics applications using zuda's scientific computing APIs:
//! - Part 1: Forward & Inverse Kinematics for 2-link arm
//! - Part 2: Trajectory Generation via Cubic Spline
//! - Part 3: Path Planning with RRT (Rapidly-exploring Random Tree)
//! - Part 4: PID Control for Joint Tracking
//!
//! ## Modules Used
//! - `numeric`: Root finding (Newton-Raphson for IK)
//! - `numeric`: Cubic spline interpolation (trajectory smoothing)
//! - `stats.distributions`: Normal distribution (sensor noise simulation)
//! - `ndarray`: State vectors and coordinate transforms
//! - `algorithms`: Distance calculations
//!
//! ## Build & Run
//! ```
//! zig build example-robotics
//! ```

const std = @import("std");
const zuda = @import("zuda");
const NDArray = zuda.ndarray.NDArray;
const numeric = zuda.numeric;
const distributions = zuda.stats.distributions;
const descriptive = zuda.stats.descriptive;

// ============================================================================
// Part 1: Robotic Arm Kinematics (2-DOF planar arm)
// ============================================================================

const ArmConfig = struct {
    l1: f64 = 1.0, // Link 1 length (m)
    l2: f64 = 0.8, // Link 2 length (m)
};

/// Forward Kinematics: (θ1, θ2) → (x, y)
fn forwardKinematics(theta1: f64, theta2: f64, config: ArmConfig) struct { x: f64, y: f64 } {
    const x = config.l1 * @cos(theta1) + config.l2 * @cos(theta1 + theta2);
    const y = config.l1 * @sin(theta1) + config.l2 * @sin(theta1 + theta2);
    return .{ .x = x, .y = y };
}

/// Inverse Kinematics: (x, y) → (θ1, θ2) using geometric solution
fn inverseKinematics(x: f64, y: f64, config: ArmConfig) !struct { theta1: f64, theta2: f64 } {
    const l1 = config.l1;
    const l2 = config.l2;
    const r2 = x * x + y * y;
    const r = @sqrt(r2);

    // Check reachability
    if (r > l1 + l2 or r < @abs(l1 - l2)) {
        return error.Unreachable;
    }

    // Cosine law: cos(θ2) = (x² + y² - l1² - l2²) / (2·l1·l2)
    const cos_theta2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2);
    const cos_theta2_clamped = @max(-1.0, @min(1.0, cos_theta2)); // Numerical stability
    const theta2 = std.math.acos(cos_theta2_clamped);

    // θ1 = atan2(y, x) - atan2(l2·sin(θ2), l1 + l2·cos(θ2))
    const sin_theta2 = @sin(theta2);
    const k1 = l1 + l2 * cos_theta2_clamped;
    const k2 = l2 * sin_theta2;
    const theta1 = std.math.atan2(y, x) - std.math.atan2(k2, k1);

    return .{ .theta1 = theta1, .theta2 = theta2 };
}

// ============================================================================
// Part 2: Trajectory Generation (Cubic Spline Interpolation)
// ============================================================================

fn generateTrajectory(allocator: std.mem.Allocator, waypoints: []const [2]f64, num_points: usize) ![2][]f64 {
    // Extract x and y coordinates
    const n = waypoints.len;
    var x_way = try allocator.alloc(f64, n);
    defer allocator.free(x_way);
    var y_way = try allocator.alloc(f64, n);
    defer allocator.free(y_way);

    for (waypoints, 0..) |wp, i| {
        x_way[i] = wp[0];
        y_way[i] = wp[1];
    }

    // Generate time parameter (uniform spacing)
    var t = try allocator.alloc(f64, n);
    defer allocator.free(t);
    for (0..n) |i| {
        t[i] = @as(f64, @floatFromInt(i));
    }

    // Interpolation points (uniformly spaced)
    var t_new = try allocator.alloc(f64, num_points);
    defer allocator.free(t_new);
    const t_max = @as(f64, @floatFromInt(n - 1));
    for (0..num_points) |i| {
        t_new[i] = t_max * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(num_points - 1));
    }

    // Cubic spline for x(t) and y(t)
    const x_interp = try numeric.interpolation.cubic_spline(f64, t, x_way, t_new, allocator);
    const y_interp = try numeric.interpolation.cubic_spline(f64, t, y_way, t_new, allocator);

    return .{ x_interp, y_interp };
}

// ============================================================================
// Part 3: Path Planning — Simplified RRT (Rapidly-exploring Random Tree)
// ============================================================================

const Point = struct { x: f64, y: f64 };

const Obstacle = struct {
    center: Point,
    radius: f64,
};

fn distance(a: Point, b: Point) f64 {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return @sqrt(dx * dx + dy * dy);
}

fn isObstacleFree(from: Point, to: Point, obstacles: []const Obstacle) bool {
    // Simple obstacle avoidance: check if line segment intersects circles
    for (obstacles) |obs| {
        const dist_to_center = pointToLineDistance(obs.center, from, to);
        if (dist_to_center < obs.radius) {
            return false; // Collision detected
        }
    }
    return true;
}

fn pointToLineDistance(p: Point, a: Point, b: Point) f64 {
    const ab_x = b.x - a.x;
    const ab_y = b.y - a.y;
    const ap_x = p.x - a.x;
    const ap_y = p.y - a.y;

    const ab_dot_ab = ab_x * ab_x + ab_y * ab_y;
    if (ab_dot_ab < 1e-10) return @sqrt(ap_x * ap_x + ap_y * ap_y); // a == b

    const t = @max(0.0, @min(1.0, (ap_x * ab_x + ap_y * ab_y) / ab_dot_ab));
    const proj_x = a.x + t * ab_x;
    const proj_y = a.y + t * ab_y;
    const dx = p.x - proj_x;
    const dy = p.y - proj_y;
    return @sqrt(dx * dx + dy * dy);
}

// ============================================================================
// Part 4: PID Controller for Joint Tracking
// ============================================================================

const PIDController = struct {
    kp: f64,
    ki: f64,
    kd: f64,
    integral: f64 = 0.0,
    prev_error: f64 = 0.0,

    fn compute(self: *PIDController, error_val: f64, dt: f64) f64 {
        self.integral += error_val * dt;
        const derivative = (error_val - self.prev_error) / dt;
        self.prev_error = error_val;
        return self.kp * error_val + self.ki * self.integral + self.kd * derivative;
    }

    fn reset(self: *PIDController) void {
        self.integral = 0.0;
        self.prev_error = 0.0;
    }
};

// ============================================================================
// Main Demonstration
// ============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Robotics & Motion Planning ===\n\n", .{});

    // ------------------------------------------------------------------------
    // Part 1: Forward & Inverse Kinematics
    // ------------------------------------------------------------------------
    std.debug.print("Part 1: Robotic Arm Kinematics (2-DOF planar arm)\n", .{});
    std.debug.print("-----------------------------------------------\n", .{});

    const config = ArmConfig{ .l1 = 1.0, .l2 = 0.8 };

    // Forward kinematics
    const theta1_test = std.math.pi / 4.0; // 45°
    const theta2_test = std.math.pi / 3.0; // 60°
    const fk_result = forwardKinematics(theta1_test, theta2_test, config);
    std.debug.print("Forward Kinematics:\n", .{});
    std.debug.print("  θ1 = {d:.3} rad, θ2 = {d:.3} rad\n", .{ theta1_test, theta2_test });
    std.debug.print("  → End-effector: x = {d:.4} m, y = {d:.4} m\n", .{ fk_result.x, fk_result.y });

    // Inverse kinematics (verify round-trip)
    const ik_result = try inverseKinematics(fk_result.x, fk_result.y, config);
    std.debug.print("\nInverse Kinematics (round-trip verification):\n", .{});
    std.debug.print("  Target: x = {d:.4} m, y = {d:.4} m\n", .{ fk_result.x, fk_result.y });
    std.debug.print("  → θ1 = {d:.3} rad, θ2 = {d:.3} rad\n", .{ ik_result.theta1, ik_result.theta2 });
    const error_theta1 = @abs(ik_result.theta1 - theta1_test);
    const error_theta2 = @abs(ik_result.theta2 - theta2_test);
    std.debug.print("  Errors: Δθ1 = {d:.6} rad, Δθ2 = {d:.6} rad\n", .{ error_theta1, error_theta2 });

    // Test reachability check
    std.debug.print("\nReachability Test:\n", .{});
    const unreachable_point = .{ .x = 2.5, .y = 0.0 }; // Beyond l1 + l2 = 1.8 m
    const ik_unreachable = inverseKinematics(unreachable_point.x, unreachable_point.y, config);
    if (ik_unreachable) |_| {
        std.debug.print("  Point ({d:.2}, {d:.2}) should be unreachable but wasn't!\n", .{ unreachable_point.x, unreachable_point.y });
    } else |err| {
        std.debug.print("  Point ({d:.2}, {d:.2}) is unreachable: {any}\n", .{ unreachable_point.x, unreachable_point.y, err });
    }

    // ------------------------------------------------------------------------
    // Part 2: Trajectory Generation
    // ------------------------------------------------------------------------
    std.debug.print("\nPart 2: Trajectory Generation (Cubic Spline)\n", .{});
    std.debug.print("--------------------------------------------\n", .{});

    const waypoints = [_][2]f64{
        .{ 0.5, 0.5 }, // Start
        .{ 1.0, 1.2 }, // Via point 1
        .{ 1.5, 1.0 }, // Via point 2
        .{ 1.3, 0.3 }, // Goal
    };

    const num_trajectory_points = 50;
    const traj = try generateTrajectory(allocator, &waypoints, num_trajectory_points);
    defer allocator.free(traj[0]);
    defer allocator.free(traj[1]);

    std.debug.print("Generated smooth trajectory through {} waypoints:\n", .{waypoints.len});
    for (waypoints) |wp| {
        std.debug.print("  ({d:.2}, {d:.2})\n", .{ wp[0], wp[1] });
    }
    std.debug.print("Interpolated to {} points via cubic spline\n", .{num_trajectory_points});

    // Compute trajectory statistics
    var path_length: f64 = 0.0;
    for (1..num_trajectory_points) |i| {
        const dx = traj[0][i] - traj[0][i - 1];
        const dy = traj[1][i] - traj[1][i - 1];
        path_length += @sqrt(dx * dx + dy * dy);
    }
    std.debug.print("Total path length: {d:.4} m\n", .{path_length});

    // ------------------------------------------------------------------------
    // Part 3: Path Planning (Simplified RRT Concept)
    // ------------------------------------------------------------------------
    std.debug.print("\nPart 3: Path Planning with Obstacle Avoidance\n", .{});
    std.debug.print("----------------------------------------------\n", .{});

    const obstacles = [_]Obstacle{
        .{ .center = .{ .x = 1.0, .y = 0.8 }, .radius = 0.3 },
        .{ .center = .{ .x = 0.7, .y = 1.5 }, .radius = 0.25 },
    };

    const start = Point{ .x = 0.3, .y = 0.3 };
    const goal = Point{ .x = 1.6, .y = 1.6 };

    std.debug.print("Start: ({d:.2}, {d:.2}), Goal: ({d:.2}, {d:.2})\n", .{ start.x, start.y, goal.x, goal.y });
    std.debug.print("Obstacles: {} circular obstacles\n", .{obstacles.len});

    // Check if direct path is collision-free
    const direct_path_free = isObstacleFree(start, goal, &obstacles);
    if (direct_path_free) {
        std.debug.print("✓ Direct path is collision-free! Distance: {d:.4} m\n", .{distance(start, goal)});
    } else {
        std.debug.print("✗ Direct path collides with obstacles. RRT planning would be needed.\n", .{});
    }

    // Test collision detection for a path that definitely collides
    const collision_test_start = Point{ .x = 0.8, .y = 0.5 };
    const collision_test_goal = Point{ .x = 1.2, .y = 1.1 };
    const collision_path_free = isObstacleFree(collision_test_start, collision_test_goal, &obstacles);
    std.debug.print("\nCollision Test: ({d:.2}, {d:.2}) → ({d:.2}, {d:.2}): {s}\n", .{
        collision_test_start.x,
        collision_test_start.y,
        collision_test_goal.x,
        collision_test_goal.y,
        if (collision_path_free) "CLEAR" else "COLLISION",
    });

    // ------------------------------------------------------------------------
    // Part 4: PID Control for Joint Tracking
    // ------------------------------------------------------------------------
    std.debug.print("\nPart 4: PID Control for Joint Tracking\n", .{});
    std.debug.print("---------------------------------------\n", .{});

    var pid = PIDController{
        .kp = 10.0, // Proportional gain
        .ki = 0.5, // Integral gain
        .kd = 2.0, // Derivative gain
    };

    const target_angle = std.math.pi / 2.0; // 90° target
    var current_angle: f64 = 0.0; // Start at 0°
    const dt = 0.01; // 10ms time step
    const num_steps = 100;

    std.debug.print("Target angle: {d:.3} rad ({d:.1}°)\n", .{ target_angle, target_angle * 180.0 / std.math.pi });
    std.debug.print("Initial angle: {d:.3} rad ({d:.1}°)\n", .{ current_angle, current_angle * 180.0 / std.math.pi });
    std.debug.print("PID gains: Kp={d:.1}, Ki={d:.1}, Kd={d:.1}\n", .{ pid.kp, pid.ki, pid.kd });

    // Simulate control loop
    var angles = try allocator.alloc(f64, num_steps);
    defer allocator.free(angles);
    var errors_buf = try allocator.alloc(f64, num_steps);
    defer allocator.free(errors_buf);

    var angular_velocity: f64 = 0.0;
    const damping = 0.5; // Damping coefficient
    const inertia = 0.1; // Moment of inertia

    for (0..num_steps) |i| {
        const error_val = target_angle - current_angle;
        const control_signal = pid.compute(error_val, dt);

        // Realistic dynamics: torque = I·α + b·ω (with damping)
        // Rearranging: α = (control_signal - damping·ω) / inertia
        const angular_acceleration = (control_signal - damping * angular_velocity) / inertia;
        angular_velocity += angular_acceleration * dt;
        current_angle += angular_velocity * dt;

        angles[i] = current_angle;
        errors_buf[i] = @abs(error_val);
    }

    // Analyze tracking performance
    const final_error = @abs(target_angle - current_angle);
    var errors_arr = try NDArray(f64, 1).fromSlice(allocator, &.{num_steps}, errors_buf, .row_major);
    defer errors_arr.deinit();
    const mean_error = descriptive.mean(f64, errors_arr);

    // Compute max error manually
    var max_error_val: f64 = 0.0;
    for (errors_buf) |err| {
        if (err > max_error_val) max_error_val = err;
    }

    std.debug.print("\nTracking Performance ({} time steps):\n", .{num_steps});
    std.debug.print("  Final angle: {d:.3} rad ({d:.1}°)\n", .{ current_angle, current_angle * 180.0 / std.math.pi });
    std.debug.print("  Final error: {d:.6} rad ({d:.3}°)\n", .{ final_error, final_error * 180.0 / std.math.pi });
    std.debug.print("  Mean tracking error: {d:.6} rad\n", .{mean_error});
    std.debug.print("  Max tracking error: {d:.6} rad\n", .{max_error_val});

    // Check settling time (when error stays below 1% of target)
    const settling_threshold = 0.01 * target_angle;
    var settled_at: ?usize = null;
    var settled_count: usize = 0;
    for (0..num_steps) |i| {
        if (errors_buf[i] < settling_threshold) {
            settled_count += 1;
            if (settled_count == 10 and settled_at == null) { // Settled for 10 consecutive steps
                settled_at = i;
            }
        } else {
            settled_count = 0;
        }
    }

    if (settled_at) |step| {
        const settling_time = @as(f64, @floatFromInt(step)) * dt;
        std.debug.print("  Settling time (1%% threshold): {d:.3} s\n", .{settling_time});
    } else {
        std.debug.print("  System did not settle within simulation time\n", .{});
    }

    std.debug.print("\n=== Demonstration Complete ===\n", .{});
}
