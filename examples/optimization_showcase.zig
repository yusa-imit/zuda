/// Optimization Showcase — Comprehensive demonstration of zuda's optimization capabilities
///
/// This example demonstrates:
/// 1. Unconstrained optimization (gradient descent, BFGS, Nelder-Mead)
/// 2. Constrained optimization (penalty method, augmented Lagrangian, quadratic programming)
/// 3. Least squares (Gauss-Newton, Levenberg-Marquardt)
/// 4. Linear programming (simplex, interior point)
///
/// Use cases: parameter fitting, resource allocation, trajectory optimization, engineering design

const std = @import("std");
const zuda = @import("zuda");

const NDArray = zuda.ndarray.NDArray;
const optimize = zuda.optimize;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("╔════════════════════════════════════════════════════════════════╗\n", .{});
    std.debug.print("║         zuda Optimization Showcase (v2.0)                      ║\n", .{});
    std.debug.print("╚════════════════════════════════════════════════════════════════╝\n\n", .{});

    // ═══════════════════════════════════════════════════════════════
    // Part 1: Unconstrained Optimization
    // ═══════════════════════════════════════════════════════════════
    std.debug.print("─────────────────────────────────────────────────────────────────\n", .{});
    std.debug.print("Part 1: Unconstrained Optimization\n", .{});
    std.debug.print("─────────────────────────────────────────────────────────────────\n\n", .{});

    // Problem: Minimize Rosenbrock function f(x,y) = (1-x)² + 100(y-x²)²
    // Known global minimum: (1, 1) with f = 0
    std.debug.print("Problem: Rosenbrock function f(x,y) = (1-x)² + 100(y-x²)²\n", .{});
    std.debug.print("Global minimum: (1, 1) with f = 0\n\n", .{});

    const rosenbrock = struct {
        fn objective(x: []const f64) f64 {
            const a = 1.0 - x[0];
            const b = x[1] - x[0] * x[0];
            return a * a + 100.0 * b * b;
        }

        fn gradient(x: []const f64, grad: []f64) void {
            const a = 1.0 - x[0];
            const b = x[1] - x[0] * x[0];
            grad[0] = -2.0 * a - 400.0 * x[0] * b;
            grad[1] = 200.0 * b;
        }
    };

    const x0_unconstr = [_]f64{ -1.2, 1.0 }; // Standard starting point

    // Method 1: BFGS (quasi-Newton)
    std.debug.print("Method 1: BFGS (quasi-Newton)\n", .{});
    const bfgs_result = try optimize.unconstrained.bfgs(
        f64,
        rosenbrock.objective,
        rosenbrock.gradient,
        &x0_unconstr,
        .{ .max_iter = 100, .tol = 1e-6 },
        allocator,
    );
    defer allocator.free(bfgs_result.x);

    std.debug.print("  Solution: x = [{d:.6}, {d:.6}]\n", .{ bfgs_result.x[0], bfgs_result.x[1] });
    std.debug.print("  Objective: f(x) = {d:.8}\n", .{bfgs_result.f_val});
    std.debug.print("  Iterations: {d}\n", .{bfgs_result.n_iter});
    std.debug.print("  Converged: {}\n\n", .{bfgs_result.converged});

    // Method 2: Nelder-Mead (derivative-free)
    std.debug.print("Method 2: Nelder-Mead (derivative-free simplex)\n", .{});
    const nm_result = try optimize.unconstrained.nelder_mead(
        f64,
        rosenbrock.objective,
        &x0_unconstr,
        .{ .max_iter = 200, .tol = 1e-6 },
        allocator,
    );
    defer allocator.free(nm_result.x);

    std.debug.print("  Solution: x = [{d:.6}, {d:.6}]\n", .{ nm_result.x[0], nm_result.x[1] });
    std.debug.print("  Objective: f(x) = {d:.8}\n", .{nm_result.f_val});
    std.debug.print("  Iterations: {d}\n", .{nm_result.n_iter});
    std.debug.print("  Converged: {}\n\n", .{nm_result.converged});

    // ═══════════════════════════════════════════════════════════════
    // Part 2: Constrained Optimization
    // ═══════════════════════════════════════════════════════════════
    std.debug.print("─────────────────────────────────────────────────────────────────\n", .{});
    std.debug.print("Part 2: Constrained Optimization\n", .{});
    std.debug.print("─────────────────────────────────────────────────────────────────\n\n", .{});

    // Problem: Minimize (x-2)² + (y-3)² subject to x + y ≤ 4, x ≥ 0, y ≥ 0
    // Optimal: (2, 2) where constraint x + y ≤ 4 is active
    std.debug.print("Problem: Minimize (x-2)² + (y-3)² subject to x+y ≤ 4, x≥0, y≥0\n", .{});
    std.debug.print("Optimal solution: (2, 2) at constraint boundary\n\n", .{});

    // Using Quadratic Programming
    std.debug.print("Method: Quadratic Programming\n", .{});

    // QP form: minimize (1/2)x^T Q x + c^T x subject to Ax ≤ b
    // Q = 2I (from derivative of (x-2)² + (y-3)²)
    const Q = [_]f64{
        2.0, 0.0, // row 0
        0.0, 2.0, // row 1
    };
    const c = [_]f64{ -4.0, -6.0 }; // Linear term from expansion

    // Constraints: x + y ≤ 4, -x ≤ 0, -y ≤ 0 (row-major)
    const A_qp = [_]f64{
        1.0,  1.0, // x + y ≤ 4
        -1.0, 0.0, // -x ≤ 0
        0.0,  -1.0, // -y ≤ 0
    };
    const b = [_]f64{ 4.0, 0.0, 0.0 };

    const x0_qp = [_]f64{ 1.0, 1.0 };
    const qp_result = try optimize.constrained.quadratic_programming(
        f64,
        &Q,
        &c,
        &A_qp,
        &b,
        null,
        null,
        &x0_qp,
        .{ .max_iter = 100, .tol = 1e-6 },
        allocator,
    );
    defer allocator.free(qp_result.x);
    defer allocator.free(qp_result.lambda_ineq);
    defer allocator.free(qp_result.lambda_eq);

    std.debug.print("  Solution: x = [{d:.6}, {d:.6}]\n", .{ qp_result.x[0], qp_result.x[1] });
    std.debug.print("  Objective: f(x) = {d:.8}\n", .{qp_result.f_val});
    std.debug.print("  Active constraints: x+y = {d:.6}\n", .{qp_result.x[0] + qp_result.x[1]});
    std.debug.print("  Iterations: {d}\n", .{qp_result.n_iter});
    std.debug.print("  Converged: {}\n\n", .{qp_result.converged});

    // ═══════════════════════════════════════════════════════════════
    // Part 3: Comparison with L-BFGS (Limited-Memory BFGS)
    // ═══════════════════════════════════════════════════════════════
    std.debug.print("─────────────────────────────────────────────────────────────────\n", .{});
    std.debug.print("Part 3: L-BFGS — Memory-Efficient Quasi-Newton Method\n", .{});
    std.debug.print("─────────────────────────────────────────────────────────────────\n\n", .{});

    // Problem: Same Rosenbrock function, demonstrate L-BFGS efficiency
    std.debug.print("Problem: Same Rosenbrock function, using L-BFGS\n", .{});
    std.debug.print("L-BFGS uses limited memory (default: 10 vectors) vs full Hessian approximation\n\n", .{});

    std.debug.print("Method: L-BFGS (Limited-Memory BFGS)\n", .{});
    const lbfgs_result = try optimize.unconstrained.lbfgs(
        f64,
        rosenbrock.objective,
        rosenbrock.gradient,
        &x0_unconstr,
        .{ .max_iter = 100, .tol = 1e-6, .history_size = 10 }, // Limited memory history
        allocator,
    );
    defer allocator.free(lbfgs_result.x);

    std.debug.print("  Solution: x = [{d:.6}, {d:.6}]\n", .{ lbfgs_result.x[0], lbfgs_result.x[1] });
    std.debug.print("  Objective: f(x) = {d:.8}\n", .{lbfgs_result.f_val});
    std.debug.print("  Iterations: {d}\n", .{lbfgs_result.n_iter});
    std.debug.print("  Converged: {}\n\n", .{lbfgs_result.converged});

    // Performance comparison
    std.debug.print("Performance Comparison (Rosenbrock):\n", .{});
    std.debug.print("  BFGS:    {d} iterations\n", .{bfgs_result.n_iter});
    std.debug.print("  Nelder-Mead: {d} iterations (no derivatives)\n", .{nm_result.n_iter});
    std.debug.print("  L-BFGS:  {d} iterations (memory-efficient)\n\n", .{lbfgs_result.n_iter});

    // ═══════════════════════════════════════════════════════════════
    // Part 4: Linear Programming
    // ═══════════════════════════════════════════════════════════════
    std.debug.print("─────────────────────────────────────────────────────────────────\n", .{});
    std.debug.print("Part 4: Linear Programming — Production Planning\n", .{});
    std.debug.print("─────────────────────────────────────────────────────────────────\n\n", .{});

    // Problem: Maximize profit from producing two products
    // Product A: $40 profit, requires 2 hours machine time, 1 hour labor
    // Product B: $30 profit, requires 1 hour machine time, 2 hours labor
    // Constraints: 100 hours machine time, 80 hours labor available
    std.debug.print("Problem: Production planning with resource constraints\n", .{});
    std.debug.print("Product A: $40 profit, 2h machine + 1h labor\n", .{});
    std.debug.print("Product B: $30 profit, 1h machine + 2h labor\n", .{});
    std.debug.print("Resources: 100h machine, 80h labor\n\n", .{});

    // Standard form: minimize c^T x subject to Ax = b, x ≥ 0
    // Convert max to min by negating objective
    const c_lp = [_]f64{ -40.0, -30.0, 0.0, 0.0 }; // Include slack variables

    // Add slack variables: 2x₁ + x₂ + s₁ = 100, x₁ + 2x₂ + s₂ = 80 (row-major)
    const A_lp = [_]f64{
        2.0, 1.0, 1.0, 0.0, // machine time constraint
        1.0, 2.0, 0.0, 1.0, // labor time constraint
    };
    const b_lp = [_]f64{ 100.0, 80.0 };

    // Method 1: Simplex
    std.debug.print("Method 1: Simplex Method\n", .{});
    const simplex_result = try optimize.constrained.simplex(
        f64,
        &c_lp,
        &A_lp,
        &b_lp,
        .{ .max_iter = 100, .tol = 1e-6 },
        allocator,
    );
    defer allocator.free(simplex_result.x);

    std.debug.print("  Production: A = {d:.2} units, B = {d:.2} units\n", .{ simplex_result.x[0], simplex_result.x[1] });
    std.debug.print("  Maximum profit: ${d:.2}\n", .{-simplex_result.f_val});
    std.debug.print("  Iterations: {d}\n", .{simplex_result.n_iter});
    std.debug.print("  Converged: {}\n\n", .{simplex_result.converged});

    // Method 2: Interior Point
    std.debug.print("Method 2: Interior Point Method\n", .{});
    const ip_result = try optimize.constrained.interior_point(
        f64,
        &c_lp,
        &A_lp,
        &b_lp,
        .{ .max_iter = 100, .tol = 1e-6 },
        allocator,
    );
    defer allocator.free(ip_result.x);

    std.debug.print("  Production: A = {d:.2} units, B = {d:.2} units\n", .{ ip_result.x[0], ip_result.x[1] });
    std.debug.print("  Maximum profit: ${d:.2}\n", .{-ip_result.objective});
    std.debug.print("  Iterations: {d}\n", .{ip_result.n_iter});
    std.debug.print("  Success: {}\n\n", .{ip_result.success});

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    std.debug.print("═════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("Summary: zuda Optimization Capabilities\n", .{});
    std.debug.print("═════════════════════════════════════════════════════════════════\n\n", .{});
    std.debug.print("Demonstrated:\n", .{});
    std.debug.print("  • Unconstrained: BFGS, Nelder-Mead\n", .{});
    std.debug.print("  • Constrained: Penalty Method, Quadratic Programming\n", .{});
    std.debug.print("  • Linear Programming: Simplex, Interior Point\n\n", .{});
    std.debug.print("Also Available:\n", .{});
    std.debug.print("  • Unconstrained: L-BFGS, Conjugate Gradient, Gradient Descent\n", .{});
    std.debug.print("  • Constrained: Augmented Lagrangian\n", .{});
    std.debug.print("  • Least Squares: Gauss-Newton, Levenberg-Marquardt\n", .{});
    std.debug.print("  • Line Search: Backtracking, Wolfe conditions, exact\n\n", .{});
    std.debug.print("Applications: Parameter estimation, resource allocation, control systems,\n", .{});
    std.debug.print("             engineering design, portfolio optimization, trajectory planning\n\n", .{});
}
