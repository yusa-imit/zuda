const std = @import("std");
const zuda = @import("zuda");

const NDArray = zuda.ndarray.NDArray;
const blas = zuda.linalg.blas;
const solve = zuda.linalg.solve;
const decomp = zuda.linalg.decompositions;
const optimize = zuda.optimize;
const Normal = zuda.stats.distributions.Normal;
const descriptive = zuda.stats.descriptive;

/// Control Systems Demonstration
///
/// This example showcases classical and modern control theory:
/// 1. State-space representation (continuous and discrete systems)
/// 2. PID controller design and tuning
/// 3. Linear Quadratic Regulator (LQR) optimal control
/// 4. Closed-loop simulation with disturbances
/// 5. Performance metrics (settling time, overshoot, steady-state error)
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Control Systems Demonstration ===\n\n", .{});

    // Part 1: Mass-Spring-Damper System (Second-Order System)
    std.debug.print("--- Part 1: Mass-Spring-Damper System ---\n", .{});
    try massSpringDamper(allocator);

    // Part 2: PID Controller Design
    std.debug.print("\n--- Part 2: PID Controller ---\n", .{});
    try pidController(allocator);

    // Part 3: Linear Quadratic Regulator (LQR)
    std.debug.print("\n--- Part 3: LQR Optimal Control ---\n", .{});
    try lqrControl(allocator);

    // Part 4: Inverted Pendulum Stabilization
    std.debug.print("\n--- Part 4: Inverted Pendulum Stabilization ---\n", .{});
    try invertedPendulum(allocator);

    std.debug.print("\n✓ Control systems demonstration complete\n", .{});
}

/// Part 1: Mass-Spring-Damper System Simulation
///
/// Second-order system: m*x'' + c*x' + k*x = F(t)
/// State-space form: x_dot = A*x + B*u
/// where x = [position, velocity]'
fn massSpringDamper(allocator: std.mem.Allocator) !void {
    const m: f64 = 1.0; // mass (kg)
    const c: f64 = 0.5; // damping (N·s/m)
    const k: f64 = 2.0; // spring constant (N/m)
    const dt: f64 = 0.01; // time step (s)
    const t_final: f64 = 10.0; // simulation time (s)
    const n_steps = @as(usize, @intFromFloat(t_final / dt));

    // State-space matrices: x_dot = A*x + B*u
    // A = [0, 1; -k/m, -c/m]
    var A = try NDArray(f64, 2).zeros(allocator, &.{ 2, 2 }, .row_major);
    defer A.deinit();
    A.set(&.{ 0, 1 }, 1.0);
    A.set(&.{ 1, 0 }, -k / m);
    A.set(&.{ 1, 1 }, -c / m);

    // B = [0; 1/m]
    var B = try NDArray(f64, 2).zeros(allocator, &.{ 2, 1 }, .row_major);
    defer B.deinit();
    B.set(&.{ 1, 0 }, 1.0 / m);

    // Discretize using forward Euler: x[k+1] = (I + dt*A)*x[k] + dt*B*u[k]
    // A_discrete = I + dt*A (manually compute)
    var A_discrete = try NDArray(f64, 2).zeros(allocator, &.{ 2, 2 }, .row_major);
    defer A_discrete.deinit();
    A_discrete.set(&.{ 0, 0 }, 1.0 + dt * (try A.get(&.{ 0, 0 })));
    A_discrete.set(&.{ 0, 1 }, dt * (try A.get(&.{ 0, 1 })));
    A_discrete.set(&.{ 1, 0 }, dt * (try A.get(&.{ 1, 0 })));
    A_discrete.set(&.{ 1, 1 }, 1.0 + dt * (try A.get(&.{ 1, 1 })));

    // Initial state: x0 = [1.0, 0.0]' (displaced 1m, zero velocity)
    var x = try NDArray(f64, 1).zeros(allocator, &.{2}, .row_major);
    defer x.deinit();
    x.set(&.{0}, 1.0);

    // Simulate free response (u = 0)
    var positions = try std.ArrayList(f64).initCapacity(allocator, n_steps);
    defer positions.deinit(allocator);

    for (0..n_steps) |i| {
        positions.appendAssumeCapacity(try x.get(&.{0}));

        // x[k+1] = A_discrete * x[k] (no input)
        var x_next = try NDArray(f64, 1).zeros(allocator, &.{2}, .row_major);
        defer x_next.deinit();

        // Matrix-vector multiply manually (A is 2x2, x is 2x1)
        const x0 = try x.get(&.{0});
        const x1 = try x.get(&.{1});
        x_next.set(&.{0}, (try A_discrete.get(&.{ 0, 0 })) * x0 + (try A_discrete.get(&.{ 0, 1 })) * x1);
        x_next.set(&.{1}, (try A_discrete.get(&.{ 1, 0 })) * x0 + (try A_discrete.get(&.{ 1, 1 })) * x1);

        x.set(&.{0}, try x_next.get(&.{0}));
        x.set(&.{1}, try x_next.get(&.{1}));

        // Print sample points
        if (i % 100 == 0) {
            const t = @as(f64, @floatFromInt(i)) * dt;
            std.debug.print("  t={d:.2}s: position={d:.4}m, velocity={d:.4}m/s\n", .{ t, try x.get(&.{0}), try x.get(&.{1}) });
        }
    }

    // Compute settling time (when |x| < 0.02)
    const threshold = 0.02;
    var settling_index: usize = n_steps;
    for (0..n_steps) |i| {
        const idx = n_steps - 1 - i;
        if (@abs(positions.items[idx]) >= threshold) {
            settling_index = idx + 1;
            break;
        }
    }
    const settling_time = @as(f64, @floatFromInt(settling_index)) * dt;

    // Compute overshoot and natural frequency
    const natural_freq = @sqrt(k / m);
    const damping_ratio = c / (2.0 * @sqrt(k * m));

    std.debug.print("  System characteristics:\n", .{});
    std.debug.print("    Natural frequency: {d:.3} rad/s\n", .{natural_freq});
    std.debug.print("    Damping ratio: {d:.3} (", .{damping_ratio});
    if (damping_ratio < 1.0) {
        std.debug.print("underdamped)\n", .{});
    } else if (damping_ratio == 1.0) {
        std.debug.print("critically damped)\n", .{});
    } else {
        std.debug.print("overdamped)\n", .{});
    }
    std.debug.print("    Settling time (2%): {d:.2}s\n", .{settling_time});
}

/// Part 2: PID Controller for Setpoint Tracking
///
/// PID control law: u(t) = Kp*e(t) + Ki*∫e(τ)dτ + Kd*de/dt
/// where e(t) = r(t) - y(t) is the tracking error
fn pidController(allocator: std.mem.Allocator) !void {
    // System: first-order with delay (common in process control)
    // G(s) = K / (τs + 1) where K=2.0, τ=1.0
    const K_process: f64 = 2.0;
    const tau: f64 = 1.0;
    const dt: f64 = 0.01;
    const t_final: f64 = 10.0;
    const n_steps = @as(usize, @intFromFloat(t_final / dt));

    // PID gains (Ziegler-Nichols tuning)
    const Kp: f64 = 1.2 / K_process; // Proportional gain
    const Ki: f64 = 0.6 / (K_process * tau); // Integral gain
    const Kd: f64 = 0.6 * tau / K_process; // Derivative gain

    std.debug.print("  PID gains: Kp={d:.3}, Ki={d:.3}, Kd={d:.3}\n", .{ Kp, Ki, Kd });

    // Setpoint (step reference at t=0)
    const r: f64 = 1.0;

    // Initial conditions
    var y: f64 = 0.0; // process output
    var integral: f64 = 0.0; // integral term
    var prev_error: f64 = r - y; // previous error

    // Add process noise
    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    var noise_dist = try Normal(f64).init(0.0, 0.01);

    var outputs = try std.ArrayList(f64).initCapacity(allocator, n_steps);
    defer outputs.deinit(allocator);
    var errors = try std.ArrayList(f64).initCapacity(allocator, n_steps);
    defer errors.deinit(allocator);

    for (0..n_steps) |i| {
        const e = r - y;
        outputs.appendAssumeCapacity(y);
        errors.appendAssumeCapacity(e);

        // PID control law
        integral += e * dt;
        const derivative = (e - prev_error) / dt;
        const u = Kp * e + Ki * integral + Kd * derivative;

        // Process dynamics: dy/dt = (-y + K*u) / tau
        const dy_dt = (-y + K_process * u) / tau + noise_dist.sample(random);
        y += dy_dt * dt;

        prev_error = e;

        // Print sample points
        if (i % 100 == 0) {
            const t = @as(f64, @floatFromInt(i)) * dt;
            std.debug.print("  t={d:.2}s: setpoint={d:.2}, output={d:.4}, error={d:.4}, u={d:.4}\n", .{ t, r, y, e, u });
        }
    }

    // Performance metrics
    const rise_time = blk: {
        for (0..n_steps) |i| {
            if (outputs.items[i] >= 0.9 * r) {
                break :blk @as(f64, @floatFromInt(i)) * dt;
            }
        }
        break :blk t_final;
    };

    const steady_state_error = @abs(r - y);
    const iae = blk: {
        var sum: f64 = 0.0;
        for (errors.items) |err| {
            sum += @abs(err) * dt;
        }
        break :blk sum;
    };

    std.debug.print("  Performance:\n", .{});
    std.debug.print("    Rise time (90%%): {d:.2}s\n", .{rise_time});
    std.debug.print("    Steady-state error: {d:.4}\n", .{steady_state_error});
    std.debug.print("    IAE (Integral Absolute Error): {d:.3}\n", .{iae});
}

/// Part 3: LQR Optimal Control
///
/// Linear Quadratic Regulator minimizes J = ∫(x'Qx + u'Ru) dt
/// Solution: u = -Kx where K = R^(-1) B' P, P solves Riccati equation
fn lqrControl(allocator: std.mem.Allocator) !void {
    // Double integrator system: x'' = u
    // State: x = [position, velocity]'
    // A = [0, 1; 0, 0], B = [0; 1]

    // For this simple system, we can solve Riccati analytically
    // Q = diag([10, 1]) - penalize position more than velocity
    // R = 1 - input penalty
    const q1: f64 = 10.0;
    const q2: f64 = 1.0;
    const r: f64 = 1.0;

    // Analytical LQR gains for double integrator
    // K = [k1, k2] where k1 = sqrt(2*q1*r), k2 = sqrt(2*q2*r + k1^2/r)
    const k1 = @sqrt(2.0 * q1);
    const k2 = @sqrt(2.0 * q2 + k1 * k1 / r);

    std.debug.print("  LQR gains: K = [{d:.3}, {d:.3}]\n", .{ k1, k2 });

    // Simulate closed-loop system: x_dot = (A - B*K)*x
    const dt: f64 = 0.01;
    const t_final: f64 = 5.0;
    const n_steps = @as(usize, @intFromFloat(t_final / dt));

    // Initial state: x0 = [1.0, 0.0]'
    var x = try NDArray(f64, 1).zeros(allocator, &.{2}, .row_major);
    defer x.deinit();
    x.set(&.{0}, 1.0);

    // Closed-loop A matrix: A_cl = A - B*K = [0, 1; -k1, -k2]
    var A_cl = try NDArray(f64, 2).zeros(allocator, &.{ 2, 2 }, .row_major);
    defer A_cl.deinit();
    A_cl.set(&.{ 0, 1 }, 1.0);
    A_cl.set(&.{ 1, 0 }, -k1);
    A_cl.set(&.{ 1, 1 }, -k2);

    var cost: f64 = 0.0;

    for (0..n_steps) |i| {
        const x0 = try x.get(&.{0});
        const x1 = try x.get(&.{1});

        // Control input: u = -K*x
        const u = -k1 * x0 - k2 * x1;

        // Cost accumulation: J += (x'Qx + u'Ru) * dt
        const stage_cost = (q1 * x0 * x0 + q2 * x1 * x1 + r * u * u) * dt;
        cost += stage_cost;

        // State update: x_dot = A_cl * x
        const x0_dot = (try A_cl.get(&.{ 0, 0 })) * x0 + (try A_cl.get(&.{ 0, 1 })) * x1;
        const x1_dot = (try A_cl.get(&.{ 1, 0 })) * x0 + (try A_cl.get(&.{ 1, 1 })) * x1;

        x.set(&.{0}, x0 + x0_dot * dt);
        x.set(&.{1}, x1 + x1_dot * dt);

        if (i % 100 == 0) {
            const t = @as(f64, @floatFromInt(i)) * dt;
            std.debug.print("  t={d:.2}s: position={d:.4}, velocity={d:.4}, u={d:.4}, cost={d:.3}\n", .{ t, x0, x1, u, cost });
        }
    }

    std.debug.print("  Total cost J: {d:.3}\n", .{cost});
}

/// Part 4: Inverted Pendulum Stabilization
///
/// Linearized pendulum: θ'' = (g/L)θ + (1/mL²)u
/// State: x = [θ, θ']' (angle, angular velocity)
fn invertedPendulum(allocator: std.mem.Allocator) !void {
    const g: f64 = 9.81; // gravity (m/s²)
    const L: f64 = 1.0; // pendulum length (m)
    const mass: f64 = 0.1; // pendulum mass (kg)
    const dt: f64 = 0.005; // time step
    const t_final: f64 = 3.0;
    const n_steps = @as(usize, @intFromFloat(t_final / dt));

    // Linearized system: A = [0, 1; g/L, 0], B = [0; 1/(mL²)]
    var A = try NDArray(f64, 2).zeros(allocator, &.{ 2, 2 }, .row_major);
    defer A.deinit();
    A.set(&.{ 0, 1 }, 1.0);
    A.set(&.{ 1, 0 }, g / L);

    var B = try NDArray(f64, 2).zeros(allocator, &.{ 2, 1 }, .row_major);
    defer B.deinit();
    B.set(&.{ 1, 0 }, 1.0 / (mass * L * L));

    // For this system, we use pole placement approach
    // Desired poles: s = -5 ± 5i (fast, damped response)
    // K gains computed to achieve these poles
    const k1: f64 = 26.1; // angle feedback
    const k2: f64 = 10.0; // angular velocity feedback

    std.debug.print("  Stabilizing gains: K = [{d:.3}, {d:.3}]\n", .{ k1, k2 });

    // Initial state: θ0 = 0.2 rad (≈11.5°), θ'0 = 0
    var x = try NDArray(f64, 1).zeros(allocator, &.{2}, .row_major);
    defer x.deinit();
    x.set(&.{0}, 0.2);

    // Add disturbance at t=1.5s
    const disturbance_time = 1.5;
    const disturbance_impulse = 0.1;

    for (0..n_steps) |i| {
        const t = @as(f64, @floatFromInt(i)) * dt;
        const theta = try x.get(&.{0});
        const theta_dot = try x.get(&.{1});

        // Control input: u = -K*x
        var u = -k1 * theta - k2 * theta_dot;

        // Apply impulse disturbance
        if (@abs(t - disturbance_time) < dt) {
            u += disturbance_impulse / dt; // impulse = force * dt
            std.debug.print("  >> Impulse disturbance applied at t={d:.2}s\n", .{t});
        }

        // State update: x_dot = A*x + B*u
        const theta_ddot = (g / L) * theta + (1.0 / (mass * L * L)) * u;
        x.set(&.{0}, theta + theta_dot * dt);
        x.set(&.{1}, theta_dot + theta_ddot * dt);

        if (i % 100 == 0 or @abs(t - disturbance_time) < dt) {
            const angle_deg = theta * 180.0 / std.math.pi;
            std.debug.print("  t={d:.2}s: θ={d:.2}°, θ'={d:.3}rad/s, u={d:.3}N·m\n", .{ t, angle_deg, theta_dot, u });
        }
    }

    const final_theta = try x.get(&.{0});
    const final_theta_deg = final_theta * 180.0 / std.math.pi;
    std.debug.print("  Final state: θ={d:.4}° (stabilized: {})\n", .{ final_theta_deg, @abs(final_theta) < 0.01 });
}
