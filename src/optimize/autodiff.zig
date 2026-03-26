// Auto-differentiation (Forward-mode) using Dual Numbers
//
// Forward-mode automatic differentiation computes derivatives alongside function values
// using dual numbers: x + εx' where ε² = 0 (nilpotent infinitesimal).
//
// For f(x + εx'), we get f(x) + εf'(x)x' via chain rule propagation.
// This allows exact derivative computation (no numerical approximation error).
//
// Time complexity: O(n × cost(f)) for gradient of f: ℝⁿ → ℝ
// Space complexity: O(n) for storing dual parts

const std = @import("std");
const math = std.math;

/// Dual number: value + derivative
/// Represents x + εx' where ε² = 0
///
/// Time: O(1) for all operations
/// Space: O(1)
pub fn Dual(comptime T: type) type {
    comptime {
        if (T != f32 and T != f64) {
            @compileError("Dual only supports f32 and f64");
        }
    }

    return struct {
        const Self = @This();

        /// Real part (function value)
        value: T,
        /// Dual part (derivative)
        derivative: T,

        /// Create a constant (derivative = 0)
        pub fn constant(v: T) Self {
            return .{ .value = v, .derivative = 0 };
        }

        /// Create a variable (derivative = 1)
        pub fn variable(v: T) Self {
            return .{ .value = v, .derivative = 1 };
        }

        /// Create dual with explicit derivative
        pub fn init(v: T, d: T) Self {
            return .{ .value = v, .derivative = d };
        }

        // Arithmetic operations (chain rule automatic)

        /// Addition: (a + εa') + (b + εb') = (a+b) + ε(a'+b')
        pub fn add(a: Self, b: Self) Self {
            return .{
                .value = a.value + b.value,
                .derivative = a.derivative + b.derivative,
            };
        }

        /// Subtraction: (a + εa') - (b + εb') = (a-b) + ε(a'-b')
        pub fn sub(a: Self, b: Self) Self {
            return .{
                .value = a.value - b.value,
                .derivative = a.derivative - b.derivative,
            };
        }

        /// Multiplication: (a + εa')(b + εb') = ab + ε(a'b + ab')
        pub fn mul(a: Self, b: Self) Self {
            return .{
                .value = a.value * b.value,
                .derivative = a.derivative * b.value + a.value * b.derivative,
            };
        }

        /// Division: (a + εa') / (b + εb') = a/b + ε((a'b - ab')/b²)
        pub fn div(a: Self, b: Self) Self {
            const inv_b_sq = 1.0 / (b.value * b.value);
            return .{
                .value = a.value / b.value,
                .derivative = (a.derivative * b.value - a.value * b.derivative) * inv_b_sq,
            };
        }

        /// Negation: -(a + εa') = -a + ε(-a')
        pub fn neg(a: Self) Self {
            return .{
                .value = -a.value,
                .derivative = -a.derivative,
            };
        }

        /// Scalar multiplication: k(a + εa') = ka + ε(ka')
        pub fn scale(a: Self, k: T) Self {
            return .{
                .value = a.value * k,
                .derivative = a.derivative * k,
            };
        }

        // Elementary functions (using chain rule)

        /// Square: f(x) = x², f'(x) = 2x
        pub fn square(a: Self) Self {
            return .{
                .value = a.value * a.value,
                .derivative = 2 * a.value * a.derivative,
            };
        }

        /// Power: f(x) = xⁿ, f'(x) = nxⁿ⁻¹
        pub fn pow(a: Self, n: T) Self {
            const val_pow = math.pow(T, a.value, n);
            return .{
                .value = val_pow,
                .derivative = n * math.pow(T, a.value, n - 1) * a.derivative,
            };
        }

        /// Square root: f(x) = √x, f'(x) = 1/(2√x)
        pub fn sqrt(a: Self) Self {
            const val_sqrt = @sqrt(a.value);
            return .{
                .value = val_sqrt,
                .derivative = a.derivative / (2 * val_sqrt),
            };
        }

        /// Exponential: f(x) = eˣ, f'(x) = eˣ
        pub fn exp(a: Self) Self {
            const val_exp = @exp(a.value);
            return .{
                .value = val_exp,
                .derivative = a.derivative * val_exp,
            };
        }

        /// Natural log: f(x) = ln(x), f'(x) = 1/x
        pub fn log(a: Self) Self {
            return .{
                .value = @log(a.value),
                .derivative = a.derivative / a.value,
            };
        }

        /// Sine: f(x) = sin(x), f'(x) = cos(x)
        pub fn sin(a: Self) Self {
            return .{
                .value = @sin(a.value),
                .derivative = a.derivative * @cos(a.value),
            };
        }

        /// Cosine: f(x) = cos(x), f'(x) = -sin(x)
        pub fn cos(a: Self) Self {
            return .{
                .value = @cos(a.value),
                .derivative = -a.derivative * @sin(a.value),
            };
        }

        /// Tangent: f(x) = tan(x), f'(x) = sec²(x) = 1/cos²(x)
        pub fn tan(a: Self) Self {
            const cos_val = @cos(a.value);
            return .{
                .value = @tan(a.value),
                .derivative = a.derivative / (cos_val * cos_val),
            };
        }

        /// Absolute value: f(x) = |x|, f'(x) = sign(x)
        pub fn abs(a: Self) Self {
            const sign: T = if (a.value >= 0) 1 else -1;
            return .{
                .value = @abs(a.value),
                .derivative = sign * a.derivative,
            };
        }
    };
}

/// Compute gradient of f: ℝⁿ → ℝ at point x
///
/// Uses forward-mode AD: evaluates f(x + εeᵢ) for each basis vector eᵢ
/// to extract ∂f/∂xᵢ from the dual part.
///
/// Time: O(n × cost(f))
/// Space: O(n) for gradient array
///
/// Returns: caller owns the gradient array
pub fn gradient(
    comptime T: type,
    f: *const fn ([]const Dual(T)) Dual(T),
    x: []const T,
    allocator: std.mem.Allocator,
) ![]T {
    const n = x.len;
    const grad = try allocator.alloc(T, n);
    errdefer allocator.free(grad);

    // Create dual array for function evaluation
    const x_dual = try allocator.alloc(Dual(T), n);
    defer allocator.free(x_dual);

    // Compute ∂f/∂xᵢ for each variable
    for (0..n) |i| {
        // Set up dual numbers: xⱼ = constant, xᵢ = variable
        for (0..n) |j| {
            if (j == i) {
                x_dual[j] = Dual(T).variable(x[j]);
            } else {
                x_dual[j] = Dual(T).constant(x[j]);
            }
        }

        // Evaluate function — derivative wrt xᵢ is in dual part
        const result = f(x_dual);
        grad[i] = result.derivative;
    }

    return grad;
}

/// Compute Jacobian of f: ℝⁿ → ℝᵐ at point x
///
/// Jacobian J[i,j] = ∂fᵢ/∂xⱼ
/// Uses forward-mode AD: one pass per input dimension
///
/// Time: O(n × m × cost(f))
/// Space: O(m×n) for Jacobian matrix
///
/// Returns: caller owns the Jacobian (row-major: m rows × n cols)
pub fn jacobian(
    comptime T: type,
    f: *const fn ([]const Dual(T), []Dual(T)) void,
    x: []const T,
    m: usize,
    allocator: std.mem.Allocator,
) ![]T {
    const n = x.len;
    const J = try allocator.alloc(T, m * n);
    errdefer allocator.free(J);

    // Create dual arrays
    const x_dual = try allocator.alloc(Dual(T), n);
    defer allocator.free(x_dual);
    const y_dual = try allocator.alloc(Dual(T), m);
    defer allocator.free(y_dual);

    // Compute column j of Jacobian (∂f/∂xⱼ)
    for (0..n) |j| {
        // Set up dual numbers: xₖ = constant except xⱼ = variable
        for (0..n) |k| {
            if (k == j) {
                x_dual[k] = Dual(T).variable(x[k]);
            } else {
                x_dual[k] = Dual(T).constant(x[k]);
            }
        }

        // Evaluate function
        f(x_dual, y_dual);

        // Extract column j: J[i,j] = derivative of fᵢ wrt xⱼ
        for (0..m) |i| {
            J[i * n + j] = y_dual[i].derivative;
        }
    }

    return J;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "Dual: constant has zero derivative" {
    const D = Dual(f64);
    const c = D.constant(5.0);
    try testing.expectEqual(5.0, c.value);
    try testing.expectEqual(0.0, c.derivative);
}

test "Dual: variable has derivative 1" {
    const D = Dual(f64);
    const v = D.variable(3.0);
    try testing.expectEqual(3.0, v.value);
    try testing.expectEqual(1.0, v.derivative);
}

test "Dual: addition chain rule" {
    const D = Dual(f64);
    const a = D.init(2.0, 3.0);
    const b = D.init(5.0, 7.0);
    const c = a.add(b);
    try testing.expectEqual(7.0, c.value);
    try testing.expectEqual(10.0, c.derivative);
}

test "Dual: subtraction chain rule" {
    const D = Dual(f64);
    const a = D.init(10.0, 3.0);
    const b = D.init(4.0, 1.0);
    const c = a.sub(b);
    try testing.expectEqual(6.0, c.value);
    try testing.expectEqual(2.0, c.derivative);
}

test "Dual: multiplication chain rule" {
    const D = Dual(f64);
    const a = D.init(3.0, 1.0); // x at 3, dx/dx = 1
    const b = D.init(5.0, 0.0); // constant 5
    const c = a.mul(b); // 3x * 5 = 15x, derivative = 5
    try testing.expectEqual(15.0, c.value);
    try testing.expectEqual(5.0, c.derivative);
}

test "Dual: division chain rule" {
    const D = Dual(f64);
    const a = D.init(10.0, 1.0); // x at 10
    const b = D.init(2.0, 0.0); // constant 2
    const c = a.div(b); // x/2, derivative = 1/2
    try testing.expectEqual(5.0, c.value);
    try testing.expectEqual(0.5, c.derivative);
}

test "Dual: negation" {
    const D = Dual(f64);
    const a = D.init(3.0, 2.0);
    const b = a.neg();
    try testing.expectEqual(-3.0, b.value);
    try testing.expectEqual(-2.0, b.derivative);
}

test "Dual: scalar multiplication" {
    const D = Dual(f64);
    const a = D.init(2.0, 1.0);
    const b = a.scale(3.0);
    try testing.expectEqual(6.0, b.value);
    try testing.expectEqual(3.0, b.derivative);
}

test "Dual: square x²" {
    const D = Dual(f64);
    const x = D.variable(3.0); // x at 3, derivative = 2x = 6
    const y = x.square();
    try testing.expectEqual(9.0, y.value);
    try testing.expectEqual(6.0, y.derivative);
}

test "Dual: power x³" {
    const D = Dual(f64);
    const x = D.variable(2.0); // x³ at x=2: f=8, f'=3x²=12
    const y = x.pow(3.0);
    try testing.expectApproxEqAbs(8.0, y.value, 1e-10);
    try testing.expectApproxEqAbs(12.0, y.derivative, 1e-10);
}

test "Dual: sqrt" {
    const D = Dual(f64);
    const x = D.variable(4.0); // √x at x=4: f=2, f'=1/(2√4)=0.25
    const y = x.sqrt();
    try testing.expectApproxEqAbs(2.0, y.value, 1e-10);
    try testing.expectApproxEqAbs(0.25, y.derivative, 1e-10);
}

test "Dual: exp" {
    const D = Dual(f64);
    const x = D.variable(0.0); // eˣ at x=0: f=1, f'=1
    const y = x.exp();
    try testing.expectApproxEqAbs(1.0, y.value, 1e-10);
    try testing.expectApproxEqAbs(1.0, y.derivative, 1e-10);
}

test "Dual: log" {
    const D = Dual(f64);
    const x = D.variable(1.0); // ln(x) at x=1: f=0, f'=1/x=1
    const y = x.log();
    try testing.expectApproxEqAbs(0.0, y.value, 1e-10);
    try testing.expectApproxEqAbs(1.0, y.derivative, 1e-10);
}

test "Dual: sin" {
    const D = Dual(f64);
    const x = D.variable(0.0); // sin(x) at x=0: f=0, f'=cos(0)=1
    const y = x.sin();
    try testing.expectApproxEqAbs(0.0, y.value, 1e-10);
    try testing.expectApproxEqAbs(1.0, y.derivative, 1e-10);
}

test "Dual: cos" {
    const D = Dual(f64);
    const x = D.variable(0.0); // cos(x) at x=0: f=1, f'=-sin(0)=0
    const y = x.cos();
    try testing.expectApproxEqAbs(1.0, y.value, 1e-10);
    try testing.expectApproxEqAbs(0.0, y.derivative, 1e-10);
}

test "Dual: tan" {
    const D = Dual(f64);
    const x = D.variable(0.0); // tan(x) at x=0: f=0, f'=sec²(0)=1
    const y = x.tan();
    try testing.expectApproxEqAbs(0.0, y.value, 1e-10);
    try testing.expectApproxEqAbs(1.0, y.derivative, 1e-10);
}

test "Dual: abs positive" {
    const D = Dual(f64);
    const x = D.variable(3.0);
    const y = x.abs();
    try testing.expectEqual(3.0, y.value);
    try testing.expectEqual(1.0, y.derivative);
}

test "Dual: abs negative" {
    const D = Dual(f64);
    const x = D.variable(-3.0);
    const y = x.abs();
    try testing.expectEqual(3.0, y.value);
    try testing.expectEqual(-1.0, y.derivative);
}

test "Dual: composite function (x² + 2x)" {
    const D = Dual(f64);
    const x = D.variable(3.0);
    // f(x) = x² + 2x, f'(x) = 2x + 2
    const x_sq = x.square();
    const two_x = x.scale(2.0);
    const y = x_sq.add(two_x);
    try testing.expectEqual(15.0, y.value); // 9 + 6
    try testing.expectEqual(8.0, y.derivative); // 6 + 2
}

test "gradient: simple quadratic f(x) = x₁² + x₂²" {
    const D = Dual(f64);

    const QuadFn = struct {
        fn eval(x: []const D) D {
            return x[0].square().add(x[1].square());
        }
    };

    const x = [_]f64{ 2.0, 3.0 };
    const grad = try gradient(f64, QuadFn.eval, &x, testing.allocator);
    defer testing.allocator.free(grad);

    // ∇f = [2x₁, 2x₂] = [4, 6]
    try testing.expectApproxEqAbs(4.0, grad[0], 1e-8);
    try testing.expectApproxEqAbs(6.0, grad[1], 1e-8);
}

test "gradient: linear function f(x) = 3x₁ + 5x₂" {
    const D = Dual(f64);

    const LinearFn = struct {
        fn eval(x: []const D) D {
            const a = x[0].scale(3.0);
            const b = x[1].scale(5.0);
            return a.add(b);
        }
    };

    const x = [_]f64{ 1.0, 2.0 };
    const grad = try gradient(f64, LinearFn.eval, &x, testing.allocator);
    defer testing.allocator.free(grad);

    // ∇f = [3, 5]
    try testing.expectApproxEqAbs(3.0, grad[0], 1e-8);
    try testing.expectApproxEqAbs(5.0, grad[1], 1e-8);
}

test "gradient: Rosenbrock function" {
    const D = Dual(f64);

    const Rosenbrock = struct {
        fn eval(x: []const D) D {
            // f(x,y) = (1-x)² + 100(y-x²)²
            const one = D.constant(1.0);
            const hundred = D.constant(100.0);

            const term1 = one.sub(x[0]).square();
            const term2 = x[1].sub(x[0].square()).square().mul(hundred);
            return term1.add(term2);
        }
    };

    const x = [_]f64{ 0.5, 1.0 };
    const grad = try gradient(f64, Rosenbrock.eval, &x, testing.allocator);
    defer testing.allocator.free(grad);

    // At (0.5, 1.0): ∇f = [-2(1-x) - 400x(y-x²), 200(y-x²)]
    // = [-2(0.5) - 400(0.5)(1-0.25), 200(1-0.25)]
    // = [-1 - 150, 150] = [-151, 150]
    try testing.expectApproxEqAbs(-151.0, grad[0], 1e-8);
    try testing.expectApproxEqAbs(150.0, grad[1], 1e-8);
}

test "jacobian: linear map f(x) = Ax" {
    const D = Dual(f64);

    const LinearMap = struct {
        fn eval(x: []const D, y: []D) void {
            // A = [[2, 3], [4, 5]]
            y[0] = x[0].scale(2.0).add(x[1].scale(3.0));
            y[1] = x[0].scale(4.0).add(x[1].scale(5.0));
        }
    };

    const x = [_]f64{ 1.0, 1.0 };
    const J = try jacobian(f64, LinearMap.eval, &x, 2, testing.allocator);
    defer testing.allocator.free(J);

    // J = [[2, 3], [4, 5]]
    try testing.expectApproxEqAbs(2.0, J[0], 1e-8); // J[0,0]
    try testing.expectApproxEqAbs(3.0, J[1], 1e-8); // J[0,1]
    try testing.expectApproxEqAbs(4.0, J[2], 1e-8); // J[1,0]
    try testing.expectApproxEqAbs(5.0, J[3], 1e-8); // J[1,1]
}

test "jacobian: nonlinear map" {
    const D = Dual(f64);

    const NonlinearMap = struct {
        fn eval(x: []const D, y: []D) void {
            // f₁(x,y) = x²
            // f₂(x,y) = xy
            y[0] = x[0].square();
            y[1] = x[0].mul(x[1]);
        }
    };

    const x = [_]f64{ 2.0, 3.0 };
    const J = try jacobian(f64, NonlinearMap.eval, &x, 2, testing.allocator);
    defer testing.allocator.free(J);

    // J = [[2x, 0], [y, x]] at (2,3) = [[4, 0], [3, 2]]
    try testing.expectApproxEqAbs(4.0, J[0], 1e-8); // J[0,0]
    try testing.expectApproxEqAbs(0.0, J[1], 1e-8); // J[0,1]
    try testing.expectApproxEqAbs(3.0, J[2], 1e-8); // J[1,0]
    try testing.expectApproxEqAbs(2.0, J[3], 1e-8); // J[1,1]
}

test "gradient: memory safety (no leaks)" {
    const D = Dual(f64);

    const SimpleFn = struct {
        fn eval(x: []const D) D {
            return x[0].square();
        }
    };

    const x = [_]f64{2.0};
    const grad = try gradient(f64, SimpleFn.eval, &x, testing.allocator);
    defer testing.allocator.free(grad);
}

test "jacobian: memory safety (no leaks)" {
    const D = Dual(f64);

    const SimpleFn = struct {
        fn eval(x: []const D, y: []D) void {
            y[0] = x[0].square();
        }
    };

    const x = [_]f64{2.0};
    const J = try jacobian(f64, SimpleFn.eval, &x, 1, testing.allocator);
    defer testing.allocator.free(J);
}

test "Dual: f32 type support" {
    const D = Dual(f32);
    const x = D.variable(2.0);
    const y = x.square();
    try testing.expectApproxEqAbs(@as(f32, 4.0), y.value, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 4.0), y.derivative, 1e-6);
}
