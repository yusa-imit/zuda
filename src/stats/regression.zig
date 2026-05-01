//! Linear regression and curve fitting algorithms.
//!
//! This module provides ordinary least squares (OLS) regression, polynomial fitting,
//! and regression diagnostics. All functions accept explicit allocators and work with
//! f32 or f64 numeric types.
//!
//! ## Functions
//! - `ols()`: Ordinary Least Squares linear regression
//! - `polyfit()`: Polynomial curve fitting
//! - `polyval()`: Evaluate polynomial at given points
//! - `rSquared()`: Coefficient of determination (R²)
//! - `residuals()`: Calculate residuals (y - ŷ)
//!
//! ## Example
//! ```zig
//! const x = [_]f64{ 1, 2, 3, 4, 5 };
//! const y = [_]f64{ 2.1, 3.9, 6.2, 7.8, 10.1 };
//! const result = try ols(f64, allocator, &x, &y);
//! defer result.deinit(allocator);
//! // result.slope ≈ 2.0, result.intercept ≈ 0.1
//! ```

const std = @import("std");
const Allocator = std.mem.Allocator;
const linalg = @import("../linalg/solve.zig");

/// Errors for regression operations
pub const RegressionError = error{
    /// Input arrays have different lengths
    DimensionMismatch,
    /// Insufficient data points for regression
    InsufficientData,
    /// Polynomial degree >= number of data points
    DegreeTooHigh,
    /// Singular matrix (cannot invert)
    SingularMatrix,
    /// Invalid input (NaN, Inf)
    InvalidInput,
};

/// Result of ordinary least squares (OLS) regression.
/// Represents the linear model: y = slope * x + intercept
///
/// Time: O(n) to compute | Space: O(1)
pub fn OLSResult(comptime T: type) type {
    return struct {
        /// Slope of the regression line (β₁)
        slope: T,
        /// Y-intercept of the regression line (β₀)
        intercept: T,
        /// Coefficient of determination (R²) ∈ [0, 1]
        r_squared: T,
        /// Standard error of the estimate
        std_error: T,
        /// Predicted values (ŷ)
        predictions: []T,

        pub fn deinit(self: @This(), allocator: Allocator) void {
            allocator.free(self.predictions);
        }
    };
}

/// Ordinary Least Squares (OLS) linear regression.
///
/// Fits a line y = β₀ + β₁x that minimizes the sum of squared residuals:
///   min Σ(yᵢ - β₀ - β₁xᵢ)²
///
/// Uses the closed-form solution:
///   β₁ = Σ((xᵢ - x̄)(yᵢ - ȳ)) / Σ((xᵢ - x̄)²)
///   β₀ = ȳ - β₁x̄
///
/// Time: O(n) | Space: O(n) for predictions
///
/// ## Arguments
/// - `T`: Numeric type (f32 or f64)
/// - `allocator`: Memory allocator
/// - `x`: Independent variable (n elements)
/// - `y`: Dependent variable (n elements)
///
/// ## Returns
/// `OLSResult` containing slope, intercept, R², standard error, and predictions
///
/// ## Errors
/// - `DimensionMismatch`: x and y have different lengths
/// - `InsufficientData`: fewer than 2 data points
/// - `InvalidInput`: x or y contains NaN/Inf
/// - `SingularMatrix`: all x values are identical (no variance)
pub fn ols(comptime T: type, allocator: Allocator, x: []const T, y: []const T) (RegressionError || Allocator.Error)!OLSResult(T) {
    if (x.len != y.len) return error.DimensionMismatch;
    if (x.len < 2) return error.InsufficientData;

    const n: T = @floatFromInt(x.len);

    // Check for invalid inputs
    for (x, y) |xi, yi| {
        if (!std.math.isFinite(xi) or !std.math.isFinite(yi)) {
            return error.InvalidInput;
        }
    }

    // Compute means
    var sum_x: T = 0;
    var sum_y: T = 0;
    for (x, y) |xi, yi| {
        sum_x += xi;
        sum_y += yi;
    }
    const mean_x = sum_x / n;
    const mean_y = sum_y / n;

    // Compute covariance and variance
    var cov_xy: T = 0; // Σ((xᵢ - x̄)(yᵢ - ȳ))
    var var_x: T = 0; // Σ((xᵢ - x̄)²)
    for (x, y) |xi, yi| {
        const dx = xi - mean_x;
        const dy = yi - mean_y;
        cov_xy += dx * dy;
        var_x += dx * dx;
    }

    // Check for zero variance (all x values identical)
    if (@abs(var_x) < std.math.floatEps(T) * @abs(mean_x) * n) {
        return error.SingularMatrix;
    }

    // Compute slope and intercept
    const slope = cov_xy / var_x;
    const intercept = mean_y - slope * mean_x;

    // Compute predictions and residuals
    const predictions = try allocator.alloc(T, x.len);
    errdefer allocator.free(predictions);

    var ss_res: T = 0; // Sum of squared residuals
    var ss_tot: T = 0; // Total sum of squares
    for (x, y, 0..) |xi, yi, i| {
        const y_pred = slope * xi + intercept;
        predictions[i] = y_pred;
        const residual = yi - y_pred;
        ss_res += residual * residual;
        const deviation = yi - mean_y;
        ss_tot += deviation * deviation;
    }

    // Compute R² and standard error
    const r_squared = if (ss_tot > 0) 1.0 - (ss_res / ss_tot) else 1.0;
    const std_error = if (x.len > 2) @sqrt(ss_res / @as(T, @floatFromInt(x.len - 2))) else 0;

    return OLSResult(T){
        .slope = slope,
        .intercept = intercept,
        .r_squared = r_squared,
        .std_error = std_error,
        .predictions = predictions,
    };
}

/// Result of polynomial regression.
/// Represents the polynomial: y = c₀ + c₁x + c₂x² + ... + cₙxⁿ
///
/// Time: O(degree × n) to compute | Space: O(n) for predictions
pub fn PolyFitResult(comptime T: type) type {
    return struct {
        /// Polynomial coefficients [c₀, c₁, ..., cₙ] (ascending order)
        coefficients: []T,
        /// Coefficient of determination (R²) ∈ [0, 1]
        r_squared: T,
        /// Predicted values (ŷ)
        predictions: []T,

        pub fn deinit(self: @This(), allocator: Allocator) void {
            allocator.free(self.coefficients);
            allocator.free(self.predictions);
        }
    };
}

/// Polynomial curve fitting using least squares.
///
/// Fits a polynomial of specified degree that minimizes the sum of squared residuals:
///   min Σ(yᵢ - (c₀ + c₁xᵢ + c₂xᵢ² + ... + cₙxᵢⁿ))²
///
/// Uses the normal equations: (XᵀX)c = Xᵀy
/// where X is the Vandermonde matrix [1, x, x², ..., xⁿ]
///
/// Time: O(degree² × n + degree³) | Space: O(degree² + degree × n)
///
/// ## Arguments
/// - `T`: Numeric type (f32 or f64)
/// - `allocator`: Memory allocator
/// - `x`: Independent variable (n elements)
/// - `y`: Dependent variable (n elements)
/// - `degree`: Polynomial degree (0 = constant, 1 = linear, 2 = quadratic, etc.)
///
/// ## Returns
/// `PolyFitResult` containing coefficients, R², and predictions
///
/// ## Errors
/// - `DimensionMismatch`: x and y have different lengths
/// - `InsufficientData`: fewer than degree+1 data points
/// - `DegreeTooHigh`: degree >= number of data points
/// - `InvalidInput`: x or y contains NaN/Inf
/// - `SingularMatrix`: Vandermonde matrix is singular (e.g., duplicate x values)
pub fn polyfit(comptime T: type, allocator: Allocator, x: []const T, y: []const T, degree: usize) (RegressionError || Allocator.Error)!PolyFitResult(T) {
    if (x.len != y.len) return error.DimensionMismatch;
    if (x.len < degree + 1) return error.InsufficientData;
    if (degree >= x.len) return error.DegreeTooHigh;

    // Check for invalid inputs
    for (x, y) |xi, yi| {
        if (!std.math.isFinite(xi) or !std.math.isFinite(yi)) {
            return error.InvalidInput;
        }
    }

    const n = x.len;
    const m = degree + 1;

    // Build Vandermonde matrix X: n × m matrix where X[i,j] = x[i]^j
    const X = try allocator.alloc(T, n * m);
    defer allocator.free(X);

    for (0..n) |i| {
        var pow: T = 1;
        for (0..m) |j| {
            X[i * m + j] = pow;
            pow *= x[i];
        }
    }

    // Compute normal equations: (XᵀX)c = Xᵀy
    // XᵀX is m × m
    const XtX = try allocator.alloc(T, m * m);
    defer allocator.free(XtX);

    // XᵀX[i,j] = Σ X[k,i] * X[k,j]
    for (0..m) |i| {
        for (0..m) |j| {
            var sum: T = 0;
            for (0..n) |k| {
                sum += X[k * m + i] * X[k * m + j];
            }
            XtX[i * m + j] = sum;
        }
    }

    // Compute Xᵀy: m × 1 vector
    const Xty = try allocator.alloc(T, m);
    defer allocator.free(Xty);

    for (0..m) |i| {
        var sum: T = 0;
        for (0..n) |k| {
            sum += X[k * m + i] * y[k];
        }
        Xty[i] = sum;
    }

    // Solve (XᵀX)c = Xᵀy for coefficients c
    // Use Gaussian elimination with partial pivoting
    const coefficients = try gaussianElimination(T, allocator, XtX, Xty, m);
    errdefer allocator.free(coefficients);

    // Compute predictions and R²
    const predictions = try allocator.alloc(T, n);
    errdefer allocator.free(predictions);

    var mean_y: T = 0;
    for (y) |yi| {
        mean_y += yi;
    }
    mean_y /= @as(T, @floatFromInt(n));

    var ss_res: T = 0; // Sum of squared residuals
    var ss_tot: T = 0; // Total sum of squares
    for (0..n) |i| {
        const y_pred = polyval(T, coefficients, x[i]);
        predictions[i] = y_pred;
        const residual = y[i] - y_pred;
        ss_res += residual * residual;
        const deviation = y[i] - mean_y;
        ss_tot += deviation * deviation;
    }

    const r_squared = if (ss_tot > 0) 1.0 - (ss_res / ss_tot) else 1.0;

    return PolyFitResult(T){
        .coefficients = coefficients,
        .r_squared = r_squared,
        .predictions = predictions,
    };
}

/// Evaluate polynomial at given points.
///
/// Computes p(x) = c₀ + c₁x + c₂x² + ... + cₙxⁿ using Horner's method:
///   p(x) = c₀ + x(c₁ + x(c₂ + ... + x·cₙ))
///
/// Time: O(degree) per evaluation | Space: O(1)
///
/// ## Arguments
/// - `T`: Numeric type (f32 or f64)
/// - `coefficients`: Polynomial coefficients [c₀, c₁, ..., cₙ] (ascending order)
/// - `x`: Point at which to evaluate the polynomial
///
/// ## Returns
/// Value of the polynomial at x
pub fn polyval(comptime T: type, coefficients: []const T, x: T) T {
    if (coefficients.len == 0) return 0;

    // Horner's method: start from highest degree and work down
    var result = coefficients[coefficients.len - 1];
    var i: usize = coefficients.len - 1;
    while (i > 0) : (i -= 1) {
        result = result * x + coefficients[i - 1];
    }
    return result;
}

/// Calculate coefficient of determination (R²).
///
/// R² measures the proportion of variance in y explained by the model:
///   R² = 1 - (SS_res / SS_tot)
/// where:
///   SS_res = Σ(yᵢ - ŷᵢ)²  (sum of squared residuals)
///   SS_tot = Σ(yᵢ - ȳ)²   (total sum of squares)
///
/// R² ∈ [0, 1] for typical models, where:
///   1.0 = perfect fit
///   0.0 = model no better than mean
///   <0  = model worse than mean (possible for non-linear models)
///
/// Time: O(n) | Space: O(1)
///
/// ## Arguments
/// - `T`: Numeric type (f32 or f64)
/// - `y_true`: Actual values (n elements)
/// - `y_pred`: Predicted values (n elements)
///
/// ## Returns
/// R² statistic
///
/// ## Errors
/// - `DimensionMismatch`: y_true and y_pred have different lengths
/// - `InsufficientData`: fewer than 2 data points
/// - `InvalidInput`: y_true or y_pred contains NaN/Inf
pub fn rSquared(comptime T: type, y_true: []const T, y_pred: []const T) RegressionError!T {
    if (y_true.len != y_pred.len) return error.DimensionMismatch;
    if (y_true.len < 2) return error.InsufficientData;

    // Check for invalid inputs
    for (y_true, y_pred) |yt, yp| {
        if (!std.math.isFinite(yt) or !std.math.isFinite(yp)) {
            return error.InvalidInput;
        }
    }

    // Compute mean of y_true
    var mean: T = 0;
    for (y_true) |yt| {
        mean += yt;
    }
    mean /= @as(T, @floatFromInt(y_true.len));

    // Compute SS_res and SS_tot
    var ss_res: T = 0;
    var ss_tot: T = 0;
    for (y_true, y_pred) |yt, yp| {
        const residual = yt - yp;
        ss_res += residual * residual;
        const deviation = yt - mean;
        ss_tot += deviation * deviation;
    }

    // Avoid division by zero
    if (ss_tot == 0) return 1.0; // Perfect fit (all y values identical)

    return 1.0 - (ss_res / ss_tot);
}

/// Calculate residuals (prediction errors).
///
/// Residuals are the differences between actual and predicted values:
///   rᵢ = yᵢ - ŷᵢ
///
/// Positive residuals indicate underestimation; negative indicate overestimation.
///
/// Time: O(n) | Space: O(n)
///
/// ## Arguments
/// - `T`: Numeric type (f32 or f64)
/// - `allocator`: Memory allocator
/// - `y_true`: Actual values (n elements)
/// - `y_pred`: Predicted values (n elements)
///
/// ## Returns
/// Residual array (n elements)
///
/// ## Errors
/// - `DimensionMismatch`: y_true and y_pred have different lengths
/// - `InvalidInput`: y_true or y_pred contains NaN/Inf
pub fn residuals(comptime T: type, allocator: Allocator, y_true: []const T, y_pred: []const T) (RegressionError || Allocator.Error)![]T {
    if (y_true.len != y_pred.len) return error.DimensionMismatch;

    // Check for invalid inputs
    for (y_true, y_pred) |yt, yp| {
        if (!std.math.isFinite(yt) or !std.math.isFinite(yp)) {
            return error.InvalidInput;
        }
    }

    const result = try allocator.alloc(T, y_true.len);
    for (y_true, y_pred, 0..) |yt, yp, i| {
        result[i] = yt - yp;
    }
    return result;
}

// Helper: Gaussian elimination with partial pivoting for solving Ax = b
fn gaussianElimination(comptime T: type, allocator: Allocator, A: []const T, b: []const T, n: usize) (RegressionError || Allocator.Error)![]T {
    // Create augmented matrix [A | b]
    const aug = try allocator.alloc(T, n * (n + 1));
    defer allocator.free(aug);

    for (0..n) |i| {
        for (0..n) |j| {
            aug[i * (n + 1) + j] = A[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for (0..n) |k| {
        // Find pivot
        var max_row = k;
        var max_val = @abs(aug[k * (n + 1) + k]);
        for (k + 1..n) |i| {
            const val = @abs(aug[i * (n + 1) + k]);
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }

        // Check for singular matrix
        if (max_val < std.math.floatEps(T) * 1000) {
            return error.SingularMatrix;
        }

        // Swap rows
        if (max_row != k) {
            for (0..n + 1) |j| {
                const tmp = aug[k * (n + 1) + j];
                aug[k * (n + 1) + j] = aug[max_row * (n + 1) + j];
                aug[max_row * (n + 1) + j] = tmp;
            }
        }

        // Eliminate below
        for (k + 1..n) |i| {
            const factor = aug[i * (n + 1) + k] / aug[k * (n + 1) + k];
            for (k..n + 1) |j| {
                aug[i * (n + 1) + j] -= factor * aug[k * (n + 1) + j];
            }
        }
    }

    // Back substitution
    const x = try allocator.alloc(T, n);
    errdefer allocator.free(x);

    var i: usize = n;
    while (i > 0) {
        i -= 1;
        var sum: T = aug[i * (n + 1) + n];
        for (i + 1..n) |j| {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / aug[i * (n + 1) + i];
    }

    return x;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const expect = testing.expect;
const expectEqual = testing.expectEqual;
const expectApproxEqRel = testing.expectApproxEqRel;
const expectError = testing.expectError;

test "ols: perfect linear fit y = 2x + 1" {
    const x = [_]f64{ 1, 2, 3, 4, 5 };
    const y = [_]f64{ 3, 5, 7, 9, 11 };
    const result = try ols(f64, testing.allocator, &x, &y);
    defer result.deinit(testing.allocator);

    try expectApproxEqRel(2.0, result.slope, 1e-10);
    try expectApproxEqRel(1.0, result.intercept, 1e-10);
    try expectApproxEqRel(1.0, result.r_squared, 1e-10);
}

test "ols: noisy linear data" {
    const x = [_]f64{ 1, 2, 3, 4, 5 };
    const y = [_]f64{ 2.1, 3.9, 6.2, 7.8, 10.1 };
    const result = try ols(f64, testing.allocator, &x, &y);
    defer result.deinit(testing.allocator);

    try expectApproxEqRel(2.0, result.slope, 0.05); // slope ≈ 2.0
    try expectApproxEqRel(0.1, result.intercept, 0.5); // intercept ≈ 0.0
    try expect(result.r_squared > 0.99); // High R² for nearly linear data
}

test "ols: horizontal line (zero slope)" {
    const x = [_]f64{ 1, 2, 3, 4 };
    const y = [_]f64{ 5, 5, 5, 5 };
    const result = try ols(f64, testing.allocator, &x, &y);
    defer result.deinit(testing.allocator);

    try expectApproxEqRel(0.0, result.slope, 1e-10);
    try expectApproxEqRel(5.0, result.intercept, 1e-10);
    try expectApproxEqRel(1.0, result.r_squared, 1e-10); // Perfect fit
}

test "ols: negative slope" {
    const x = [_]f64{ 1, 2, 3, 4 };
    const y = [_]f64{ 10, 7, 4, 1 };
    const result = try ols(f64, testing.allocator, &x, &y);
    defer result.deinit(testing.allocator);

    try expectApproxEqRel(-3.0, result.slope, 1e-10);
    try expectApproxEqRel(13.0, result.intercept, 1e-10);
    try expectApproxEqRel(1.0, result.r_squared, 1e-10);
}

test "ols: dimension mismatch" {
    const x = [_]f64{ 1, 2, 3 };
    const y = [_]f64{ 1, 2 };
    try expectError(error.DimensionMismatch, ols(f64, testing.allocator, &x, &y));
}

test "ols: insufficient data" {
    const x = [_]f64{1};
    const y = [_]f64{2};
    try expectError(error.InsufficientData, ols(f64, testing.allocator, &x, &y));
}

test "ols: singular matrix (all x identical)" {
    const x = [_]f64{ 3, 3, 3, 3 };
    const y = [_]f64{ 1, 2, 3, 4 };
    try expectError(error.SingularMatrix, ols(f64, testing.allocator, &x, &y));
}

test "ols: f32 precision" {
    const x = [_]f32{ 1, 2, 3 };
    const y = [_]f32{ 2, 4, 6 };
    const result = try ols(f32, testing.allocator, &x, &y);
    defer result.deinit(testing.allocator);

    try expectApproxEqRel(@as(f32, 2.0), result.slope, 1e-6);
    try expectApproxEqRel(@as(f32, 0.0), result.intercept, 1e-6);
}

test "polyfit: degree 0 (constant)" {
    const x = [_]f64{ 1, 2, 3, 4 };
    const y = [_]f64{ 5, 5, 5, 5 };
    const result = try polyfit(f64, testing.allocator, &x, &y, 0);
    defer result.deinit(testing.allocator);

    try expectEqual(@as(usize, 1), result.coefficients.len);
    try expectApproxEqRel(5.0, result.coefficients[0], 1e-10);
    try expectApproxEqRel(1.0, result.r_squared, 1e-10);
}

test "polyfit: degree 1 (linear)" {
    const x = [_]f64{ 1, 2, 3, 4 };
    const y = [_]f64{ 3, 5, 7, 9 };
    const result = try polyfit(f64, testing.allocator, &x, &y, 1);
    defer result.deinit(testing.allocator);

    try expectEqual(@as(usize, 2), result.coefficients.len);
    try expectApproxEqRel(1.0, result.coefficients[0], 1e-10); // intercept
    try expectApproxEqRel(2.0, result.coefficients[1], 1e-10); // slope
    try expectApproxEqRel(1.0, result.r_squared, 1e-10);
}

test "polyfit: degree 2 (quadratic)" {
    const x = [_]f64{ -2, -1, 0, 1, 2 };
    const y = [_]f64{ 4, 1, 0, 1, 4 }; // y = x²
    const result = try polyfit(f64, testing.allocator, &x, &y, 2);
    defer result.deinit(testing.allocator);

    try expectEqual(@as(usize, 3), result.coefficients.len);
    try expectApproxEqRel(0.0, result.coefficients[0], 1e-10); // constant
    try expectApproxEqRel(0.0, result.coefficients[1], 1e-10); // linear
    try expectApproxEqRel(1.0, result.coefficients[2], 1e-10); // quadratic
    try expectApproxEqRel(1.0, result.r_squared, 1e-10);
}

test "polyfit: degree 3 (cubic)" {
    const x = [_]f64{ -1, 0, 1, 2, 3 };
    const y = [_]f64{ -1, 0, 1, 8, 27 }; // y = x³
    const result = try polyfit(f64, testing.allocator, &x, &y, 3);
    defer result.deinit(testing.allocator);

    try expectEqual(@as(usize, 4), result.coefficients.len);
    try expectApproxEqRel(0.0, result.coefficients[0], 1e-8);
    try expectApproxEqRel(0.0, result.coefficients[1], 1e-8);
    try expectApproxEqRel(0.0, result.coefficients[2], 1e-8);
    try expectApproxEqRel(1.0, result.coefficients[3], 1e-8);
    try expect(result.r_squared > 0.999);
}

test "polyfit: dimension mismatch" {
    const x = [_]f64{ 1, 2, 3 };
    const y = [_]f64{ 1, 2 };
    try expectError(error.DimensionMismatch, polyfit(f64, testing.allocator, &x, &y, 1));
}

test "polyfit: insufficient data" {
    const x = [_]f64{ 1, 2 };
    const y = [_]f64{ 1, 2 };
    try expectError(error.InsufficientData, polyfit(f64, testing.allocator, &x, &y, 2)); // need 3 points for degree 2
}

test "polyfit: degree too high" {
    const x = [_]f64{ 1, 2, 3 };
    const y = [_]f64{ 1, 2, 3 };
    try expectError(error.DegreeTooHigh, polyfit(f64, testing.allocator, &x, &y, 3)); // degree >= n
}

test "polyval: constant polynomial" {
    const coeffs = [_]f64{5.0};
    try expectApproxEqRel(5.0, polyval(f64, &coeffs, 10.0), 1e-10);
}

test "polyval: linear polynomial" {
    const coeffs = [_]f64{ 1.0, 2.0 }; // 1 + 2x
    try expectApproxEqRel(5.0, polyval(f64, &coeffs, 2.0), 1e-10); // 1 + 2*2 = 5
    try expectApproxEqRel(11.0, polyval(f64, &coeffs, 5.0), 1e-10); // 1 + 2*5 = 11
}

test "polyval: quadratic polynomial" {
    const coeffs = [_]f64{ 0.0, 0.0, 1.0 }; // x²
    try expectApproxEqRel(4.0, polyval(f64, &coeffs, 2.0), 1e-10);
    try expectApproxEqRel(9.0, polyval(f64, &coeffs, 3.0), 1e-10);
}

test "polyval: cubic polynomial" {
    const coeffs = [_]f64{ 1.0, 2.0, 3.0, 4.0 }; // 1 + 2x + 3x² + 4x³
    const x: f64 = 2.0;
    const expected = 1 + 2 * 2 + 3 * 4 + 4 * 8; // 1 + 4 + 12 + 32 = 49
    try expectApproxEqRel(expected, polyval(f64, &coeffs, x), 1e-10);
}

test "rSquared: perfect fit" {
    const y_true = [_]f64{ 1, 2, 3, 4, 5 };
    const y_pred = [_]f64{ 1, 2, 3, 4, 5 };
    const r2 = try rSquared(f64, &y_true, &y_pred);
    try expectApproxEqRel(1.0, r2, 1e-10);
}

test "rSquared: worst fit (mean)" {
    const y_true = [_]f64{ 1, 2, 3, 4, 5 };
    const mean: f64 = 3.0;
    const y_pred = [_]f64{ mean, mean, mean, mean, mean };
    const r2 = try rSquared(f64, &y_true, &y_pred);
    try expectApproxEqRel(0.0, r2, 1e-10);
}

test "rSquared: good fit" {
    const y_true = [_]f64{ 1, 2, 3, 4, 5 };
    const y_pred = [_]f64{ 1.1, 2.0, 2.9, 4.0, 5.1 };
    const r2 = try rSquared(f64, &y_true, &y_pred);
    try expect(r2 > 0.95); // Very high R²
}

test "rSquared: dimension mismatch" {
    const y_true = [_]f64{ 1, 2, 3 };
    const y_pred = [_]f64{ 1, 2 };
    try expectError(error.DimensionMismatch, rSquared(f64, &y_true, &y_pred));
}

test "rSquared: insufficient data" {
    const y_true = [_]f64{1};
    const y_pred = [_]f64{1};
    try expectError(error.InsufficientData, rSquared(f64, &y_true, &y_pred));
}

test "residuals: zero residuals" {
    const y_true = [_]f64{ 1, 2, 3 };
    const y_pred = [_]f64{ 1, 2, 3 };
    const res = try residuals(f64, testing.allocator, &y_true, &y_pred);
    defer testing.allocator.free(res);

    for (res) |r| {
        try expectApproxEqRel(0.0, r, 1e-10);
    }
}

test "residuals: positive and negative" {
    const y_true = [_]f64{ 1, 2, 3, 4 };
    const y_pred = [_]f64{ 1.5, 2.0, 2.5, 4.5 };
    const res = try residuals(f64, testing.allocator, &y_true, &y_pred);
    defer testing.allocator.free(res);

    try expectApproxEqRel(-0.5, res[0], 1e-10);
    try expectApproxEqRel(0.0, res[1], 1e-10);
    try expectApproxEqRel(0.5, res[2], 1e-10);
    try expectApproxEqRel(-0.5, res[3], 1e-10);
}

test "residuals: dimension mismatch" {
    const y_true = [_]f64{ 1, 2, 3 };
    const y_pred = [_]f64{ 1, 2 };
    try expectError(error.DimensionMismatch, residuals(f64, testing.allocator, &y_true, &y_pred));
}

test "ols: memory safety (10 iterations)" {
    for (0..10) |_| {
        const x = [_]f64{ 1, 2, 3, 4, 5 };
        const y = [_]f64{ 2, 4, 6, 8, 10 };
        const result = try ols(f64, testing.allocator, &x, &y);
        result.deinit(testing.allocator);
    }
}

test "polyfit: memory safety (10 iterations)" {
    for (0..10) |_| {
        const x = [_]f64{ 1, 2, 3, 4 };
        const y = [_]f64{ 1, 4, 9, 16 };
        const result = try polyfit(f64, testing.allocator, &x, &y, 2);
        result.deinit(testing.allocator);
    }
}

test "residuals: memory safety (10 iterations)" {
    for (0..10) |_| {
        const y_true = [_]f64{ 1, 2, 3 };
        const y_pred = [_]f64{ 1.1, 2.0, 2.9 };
        const res = try residuals(f64, testing.allocator, &y_true, &y_pred);
        testing.allocator.free(res);
    }
}
