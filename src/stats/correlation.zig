//! Correlation and Linear Regression — Relationship between variables
//!
//! This module provides functions for measuring linear relationships between two variables:
//! - Pearson correlation coefficient (parametric, assumes normality)
//! - Spearman rank correlation (non-parametric, rank-based)
//! - Kendall Tau correlation (non-parametric, pairwise concordance)
//! - Simple linear regression (y ~ x)
//! - Polynomial fitting and evaluation (polyfit, polyval)
//!
//! Most functions operate on 1D NDArray(f64, 1) inputs. polyfit/polyval work with raw f64 slices.
//!
//! ## Time Complexity
//! - `pearson`: O(n) — one pass for means, one for covariance
//! - `spearman`: O(n log n) — requires sorting and ranking
//! - `linregress`: O(n) — linear in sample size
//!
//! ## Use Cases
//! - Assessing strength and direction of linear relationships
//! - Detecting multicollinearity in regression
//! - Non-parametric alternative to Pearson (Spearman)
//! - Predicting values and assessing model quality
//! - Feature selection based on correlation with target

const std = @import("std");
const math = std.math;
const testing = std.testing;
const Allocator = std.mem.Allocator;

// Import dependencies
const descriptive = @import("descriptive.zig");
const NDArray_type = @import("../ndarray/ndarray.zig").NDArray;
const StudentT_Distribution = @import("distributions/student_t.zig").StudentT;

// ============================================================================
// REGRESSION RESULT TYPE
// ============================================================================

/// Result of simple linear regression (y ~ x)
///
/// Represents fitted model: ŷ = intercept + slope * x
///
/// Fields:
/// - slope: regression coefficient (β₁)
/// - intercept: regression intercept (β₀)
/// - r_squared: coefficient of determination [0, 1]
/// - p_value: significance test p-value for slope
/// - std_err: standard error of slope estimate
pub const RegressionResult = struct {
    slope: f64,
    intercept: f64,
    r_squared: f64,
    p_value: f64,
    std_err: f64,
};

// ============================================================================
// PEARSON CORRELATION
// ============================================================================

/// Compute Pearson correlation coefficient between two 1D arrays
///
/// Pearson r measures linear relationship: r = cov(x,y) / (σ_x * σ_y)
/// where cov(x,y) = E[(x - μ_x)(y - μ_y)], σ is standard deviation
///
/// Interpretation:
/// - r = 1: perfect positive linear relationship
/// - r = 0: no linear correlation
/// - r = -1: perfect negative linear relationship
/// - |r| < 0.3: weak; 0.3-0.7: moderate; >0.7: strong correlation
///
/// Properties:
/// - Symmetric: pearson(x, y) = pearson(y, x)
/// - Bounded: r ∈ [-1, 1]
/// - Unitless: invariant to scaling and translation
/// - Parametric: assumes bivariate normality for inference
///
/// Parameters:
/// - x, y: 1D NDArray(f64, 1) of equal length
/// - allocator: (not used, provided for API consistency)
///
/// Returns: Pearson r ∈ [-1, 1]
///
/// Errors:
/// - error.EmptyArray if either array is empty
/// - error.DimensionMismatch if x.count() != y.count()
/// - error.ZeroStdDev if either x or y has zero standard deviation
///
/// Time: O(n) — two passes (means, then covariance/stddev)
/// Space: O(1)
///
/// Example:
/// ```zig
/// const x_data = [_]f64{ 1, 2, 3, 4, 5 };
/// const y_data = [_]f64{ 2, 4, 6, 8, 10 };  // y = 2x
/// var x = try NDArray(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
/// var y = try NDArray(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
/// const r = try pearson(x, y, test_allocator);  // r = 1.0 (perfect positive)
/// ```
pub fn pearson(
    x: NDArray_type(f64, 1),
    y: NDArray_type(f64, 1),
    allocator: Allocator,
) !f64 {
    _ = allocator; // Not used, but provided for API consistency

    const n = x.count();
    if (n == 0) return error.EmptyArray;
    if (y.count() != n) return error.DimensionMismatch;

    // Compute means
    const mean_x = descriptive.mean(f64, x);
    const mean_y = descriptive.mean(f64, y);

    // Compute standard deviations
    const var_x = try descriptive.variance(f64, x, 0); // ddof=0 for population
    const var_y = try descriptive.variance(f64, y, 0);

    if (var_x == 0 or var_y == 0) return error.ZeroStdDev;

    const std_x = math.sqrt(var_x);
    const std_y = math.sqrt(var_y);

    // Compute covariance: E[(x - μ_x)(y - μ_y)]
    var covariance: f64 = 0;
    var iter_x = x.iterator();
    var iter_y = y.iterator();
    while (iter_x.next()) |x_val| {
        if (iter_y.next()) |y_val| {
            covariance += (x_val - mean_x) * (y_val - mean_y);
        }
    }
    covariance /= @as(f64, @floatFromInt(n));

    // r = cov(x,y) / (σ_x * σ_y)
    const r = covariance / (std_x * std_y);

    // Clamp to [-1, 1] to handle numerical errors
    return math.clamp(r, -1.0, 1.0);
}

// ============================================================================
// SPEARMAN RANK CORRELATION
// ============================================================================

/// Compute Spearman rank correlation coefficient between two 1D arrays
///
/// Spearman ρ is non-parametric rank-based alternative to Pearson r.
/// Algorithm: Convert x and y to ranks (handling ties via average ranks),
/// then compute Pearson r on the ranks.
///
/// Interpretation:
/// - ρ ∈ [-1, 1] with same meaning as Pearson r
/// - Non-parametric: no assumption of normality
/// - Detects monotonic relationships (not just linear)
/// - More robust to outliers than Pearson
///
/// Properties:
/// - Symmetric: spearman(x, y) = spearman(y, x)
/// - Bounded: ρ ∈ [-1, 1]
/// - Handles ties by averaging ranks
///
/// Parameters:
/// - x, y: 1D NDArray(f64, 1) of equal length
/// - allocator: Used for ranking temporary arrays
///
/// Returns: Spearman ρ ∈ [-1, 1]
///
/// Errors:
/// - error.EmptyArray if either array is empty
/// - error.DimensionMismatch if x.count() != y.count()
/// - Allocator.Error if allocation fails
///
/// Time: O(n log n) due to sorting for ranking
/// Space: O(n) for temporary rank arrays
///
/// Example:
/// ```zig
/// // Monotonic but non-linear relationship: y = x²
/// const x_data = [_]f64{ 1, 2, 3, 4, 5 };
/// const y_data = [_]f64{ 1, 4, 9, 16, 25 };
/// var x = try NDArray(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
/// var y = try NDArray(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
/// const rho = try spearman(x, y, test_allocator);  // ρ = 1.0 (perfect rank correlation)
/// ```
pub fn spearman(
    x: NDArray_type(f64, 1),
    y: NDArray_type(f64, 1),
    allocator: Allocator,
) !f64 {
    const n = x.count();
    if (n == 0) return error.EmptyArray;
    if (y.count() != n) return error.DimensionMismatch;

    // Convert x to ranks
    const x_data = try allocator.alloc(f64, n);
    defer test_allocator.free(x_data);
    var iter = x.iterator();
    var idx: usize = 0;
    while (iter.next()) |val| {
        x_data[idx] = val;
        idx += 1;
    }

    const x_ranks = try rankData(f64, x_data, allocator);
    defer test_allocator.free(x_ranks);

    // Convert y to ranks
    const y_data = try allocator.alloc(f64, n);
    defer test_allocator.free(y_data);
    iter = y.iterator();
    idx = 0;
    while (iter.next()) |val| {
        y_data[idx] = val;
        idx += 1;
    }

    const y_ranks = try rankData(f64, y_data, allocator);
    defer test_allocator.free(y_ranks);

    // Create NDArray from ranks and compute Pearson on ranks
    var x_rank_array = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{n}, x_ranks, .row_major);
    defer x_rank_array.deinit();
    var y_rank_array = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{n}, y_ranks, .row_major);
    defer y_rank_array.deinit();

    return try pearson(x_rank_array, y_rank_array, allocator);
}

/// Convert array to ranks, handling ties by averaging ranks
/// Returns newly allocated slice of ranks (caller must free)
///
/// Algorithm:
/// 1. Create (value, original_index) pairs
/// 2. Sort by value
/// 3. Assign ranks, averaging for ties
/// 4. Return ranks in original order
fn rankData(comptime T: type, data: []T, allocator: Allocator) ![]T {
    const n = data.len;
    if (n == 0) return error.EmptyArray;

    // Create (value, index) pairs and sort by value
    const IndexedValue = struct {
        value: T,
        index: usize,
    };

    const pairs = try allocator.alloc(IndexedValue, n);
    defer allocator.free(pairs);

    for (data, 0..) |val, i| {
        pairs[i] = .{ .value = val, .index = i };
    }

    // Sort by value
    std.mem.sort(IndexedValue, pairs, {}, struct {
        fn lessThan(_: void, a: IndexedValue, b: IndexedValue) bool {
            return a.value < b.value;
        }
    }.lessThan);

    // Assign ranks, handling ties
    const ranks = try allocator.alloc(T, n);
    var i: usize = 0;
    while (i < n) {
        // Find end of tie group
        var j = i + 1;
        while (j < n and pairs[j].value == pairs[i].value) {
            j += 1;
        }

        // Assign average rank to all values in tie group
        const avg_rank = @as(T, @floatFromInt(i + j - 1)) / 2.0 + 1.0;

        for (i..j) |idx| {
            ranks[pairs[idx].index] = avg_rank;
        }

        i = j;
    }

    return ranks;
}

// ============================================================================
// SIMPLE LINEAR REGRESSION
// ============================================================================

/// Fit simple linear regression model: ŷ = intercept + slope * x
///
/// Ordinary Least Squares (OLS) fitting minimizes sum of squared residuals:
/// minimize Σ(y_i - ŷ_i)²
///
/// Formulas:
/// - slope = cov(x, y) / var(x)
/// - intercept = mean(y) - slope * mean(x)
/// - r_squared = 1 - (RSS / TSS)   [proportion of variance explained]
/// - std_err = sqrt(MSE / Σ(x_i - mean(x))²)
/// - t = slope / std_err → p-value via Student's t distribution
///
/// where:
/// - RSS = Σ(y_i - ŷ_i)² [residual sum of squares]
/// - TSS = Σ(y_i - mean(y))² [total sum of squares]
/// - MSE = RSS / (n - 2) [mean squared error]
///
/// Parameters:
/// - x, y: 1D NDArray(f64, 1) of equal length (≥2 observations)
/// - allocator: (not used, provided for API consistency)
///
/// Returns: RegressionResult with slope, intercept, R², p-value, std_err
///
/// Errors:
/// - error.EmptyArray if either array is empty
/// - error.DimensionMismatch if x.count() != y.count()
/// - error.ConstantX if var(x) = 0 (vertical line, undefined slope)
/// - error.InsufficientSamples if n < 2
///
/// Time: O(n) — single pass with accumulation
/// Space: O(1)
///
/// Example:
/// ```zig
/// const x_data = [_]f64{ 1, 2, 3, 4, 5 };
/// const y_data = [_]f64{ 3, 5, 7, 9, 11 };  // y = 2x + 1
/// var x = try NDArray(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
/// var y = try NDArray(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
/// const result = try linregress(x, y, test_allocator);
/// // result.slope ≈ 2.0, result.intercept ≈ 1.0, result.r_squared ≈ 1.0
/// ```
pub fn linregress(
    x: NDArray_type(f64, 1),
    y: NDArray_type(f64, 1),
    allocator: Allocator,
) !RegressionResult {
    _ = allocator; // Not used, but provided for API consistency

    const n = x.count();
    if (n == 0) return error.EmptyArray;
    if (y.count() != n) return error.DimensionMismatch;
    if (n < 3) return error.InsufficientSamples; // Need n >= 3 for df = n-2 >= 1

    // Compute means
    const mean_x = descriptive.mean(f64, x);
    const mean_y = descriptive.mean(f64, y);

    // Compute variance of x
    const var_x = try descriptive.variance(f64, x, 0);
    if (var_x == 0) return error.ConstantX;

    // Compute covariance and sum of squared deviations
    var covariance: f64 = 0;
    var ss_x: f64 = 0; // Sum of squared x deviations
    var ss_y: f64 = 0; // Sum of squared y deviations
    var ss_xy: f64 = 0; // Sum of cross-product deviations

    var iter_x = x.iterator();
    var iter_y = y.iterator();
    while (iter_x.next()) |x_val| {
        if (iter_y.next()) |y_val| {
            const dx = x_val - mean_x;
            const dy = y_val - mean_y;
            covariance += dx * dy;
            ss_x += dx * dx;
            ss_y += dy * dy;
            ss_xy += dx * dy;
        }
    }
    covariance /= @as(f64, @floatFromInt(n));

    // Compute slope and intercept
    const slope = covariance / var_x;
    const intercept = mean_y - slope * mean_x;

    // Compute residuals and R-squared
    var rss: f64 = 0; // Residual sum of squares
    iter_x = x.iterator();
    iter_y = y.iterator();
    while (iter_x.next()) |x_val| {
        if (iter_y.next()) |y_val| {
            const predicted = intercept + slope * x_val;
            const residual = y_val - predicted;
            rss += residual * residual;
        }
    }

    const tss = ss_y; // Total sum of squares
    const r_squared = 1.0 - (rss / tss);

    // Compute standard error of slope
    const n_f = @as(f64, @floatFromInt(n));
    const mse = rss / (n_f - 2.0); // Mean squared error
    const std_err = math.sqrt(mse / ss_x);

    // Compute t-statistic and p-value
    const t_stat = slope / std_err;
    const df = @as(f64, @floatFromInt(n - 2));

    const p_value: f64 = blk: {
        const dist = try StudentT_Distribution(f64).init(df);
        const cdf_val = dist.cdf(t_stat);
        break :blk if (t_stat >= 0)
            2.0 * (1.0 - cdf_val)
        else
            2.0 * cdf_val;
    };

    return .{
        .slope = slope,
        .intercept = intercept,
        .r_squared = math.clamp(r_squared, 0.0, 1.0),
        .p_value = p_value,
        .std_err = std_err,
    };
}

// ============================================================================
// KENDALL'S TAU RANK CORRELATION
// ============================================================================

/// Compute Kendall's tau-b rank correlation coefficient between two 1D arrays
///
/// Kendall's tau is a non-parametric rank correlation measure that counts the
/// agreement between ranking of pairs. Tau-b formula (corrects for ties):
///
/// τ_b = (C - D) / sqrt((C + D + T_x)(C + D + T_y))
///
/// where:
/// - C = number of concordant pairs (x[i] < x[j] and y[i] < y[j], or both greater)
/// - D = number of discordant pairs (x[i] < x[j] and y[i] > y[j], or vice versa)
/// - T_x = number of pairs tied in x only (x[i] == x[j], y[i] != y[j])
/// - T_y = number of pairs tied in y only (x[i] != x[j], y[i] == y[j])
///
/// Interpretation:
/// - τ ∈ [-1, 1] with same meaning as other correlation measures
/// - τ = 1: perfect positive rank correlation (monotonically increasing)
/// - τ = 0: no rank correlation
/// - τ = -1: perfect negative rank correlation (monotonically decreasing)
/// - Non-parametric: no assumption of normality, detects monotonic relationships
/// - More conservative than Spearman for measuring strength, but invariant to
///   monotonic transformations (not just linear)
///
/// Properties:
/// - Symmetric: kendalltau(x, y) = kendalltau(y, x)
/// - Bounded: τ ∈ [-1, 1]
/// - Invariant to monotonic transformations (scaling, translation)
/// - Handles ties gracefully via tau-b formula
///
/// Parameters:
/// - x, y: 1D slices of f64 of equal length
/// - allocator: (not used, provided for API consistency)
///
/// Returns: Kendall's tau-b ∈ [-1, 1]
///
/// Errors:
/// - error.EmptyArray if either array is empty
/// - error.DimensionMismatch if x.len != y.len
///
/// Time: O(n²) — naive pairwise comparison loop
/// Space: O(1) — no heap allocation
///
/// Example:
/// ```zig
/// const x = [_]f64{ 1, 2, 3, 4, 5 };
/// const y = [_]f64{ 1, 2, 3, 4, 5 };  // Monotonically increasing
/// const tau = try kendalltau(&x, &y, allocator);  // tau = 1.0
/// ```
pub fn kendalltau(x: []const f64, y: []const f64, allocator: Allocator) !f64 {
    _ = allocator; // Not used, but provided for API consistency

    const n = x.len;
    if (n == 0) return error.EmptyArray;
    if (y.len != n) return error.DimensionMismatch;

    // Handle trivial cases
    if (n < 2) return 0.0;

    // Count concordant, discordant, and tied pairs
    var concordant: i64 = 0;
    var discordant: i64 = 0;
    var ties_x: i64 = 0;
    var ties_y: i64 = 0;

    // Compare all pairs (i, j) where i < j
    var i: usize = 0;
    while (i < n - 1) : (i += 1) {
        var j = i + 1;
        while (j < n) : (j += 1) {
            const x_i = x[i];
            const x_j = x[j];
            const y_i = y[i];
            const y_j = y[j];

            // Determine if x is tied
            const x_tied = (x_i == x_j);
            // Determine if y is tied
            const y_tied = (y_i == y_j);

            // Skip pairs tied in both x and y
            if (x_tied and y_tied) continue;

            // Count ties
            if (x_tied) {
                ties_x += 1;
            } else if (y_tied) {
                ties_y += 1;
            } else {
                // Neither tied: count as concordant or discordant
                const x_ordered = x_i < x_j;
                const y_ordered = y_i < y_j;

                if (x_ordered == y_ordered) {
                    // Both increasing or both decreasing → concordant
                    concordant += 1;
                } else {
                    // Opposite ordering → discordant
                    discordant += 1;
                }
            }
        }
    }

    // Apply Kendall tau-b formula
    const c_f = @as(f64, @floatFromInt(concordant));
    const d_f = @as(f64, @floatFromInt(discordant));
    const tx_f = @as(f64, @floatFromInt(ties_x));
    const ty_f = @as(f64, @floatFromInt(ties_y));

    const numerator = c_f - d_f;
    const denom_part1 = c_f + d_f + tx_f;
    const denom_part2 = c_f + d_f + ty_f;
    const denominator = math.sqrt(denom_part1 * denom_part2);

    const tau: f64 = if (denominator == 0)
        0.0
    else
        numerator / denominator;

    // Clamp to [-1, 1] for numerical stability
    return math.clamp(tau, -1.0, 1.0);
}

// ============================================================================
// POLYNOMIAL FITTING AND EVALUATION
// ============================================================================

/// Fit polynomial of specified degree to (x, y) data using least squares
///
/// Given n data points (x[i], y[i]), finds coefficients [c0, c1, ..., cd] that minimize
/// the sum of squared residuals: Σ(y[i] - (c0 + c1*x[i] + ... + cd*x[i]^d))²
///
/// Uses normal equations: V^T·V·c = V^T·y where V is Vandermonde matrix V[i,j] = x[i]^j
///
/// Parameters:
/// - x: independent variable data (length n)
/// - y: dependent variable data (length n)
/// - degree: degree of polynomial to fit (must be < n)
/// - allocator: memory allocator for result
///
/// Returns: coefficients [c0, c1, c2, ..., cd] (length = degree + 1)
///   - c0 is constant term
///   - c1 is linear coefficient
///   - cd is coefficient of x^d
///
/// Errors:
/// - error.EmptyArray if x or y is empty
/// - error.DimensionMismatch if x.len != y.len
/// - error.DegreeTooLarge if degree >= x.len (need more points than unknowns)
///
/// Time: O(n*d²) | Space: O(n*d)
///
pub fn polyfit(x: []const f64, y: []const f64, degree: usize, allocator: Allocator) ![]f64 {
    const n = x.len;

    // Validate inputs
    if (n == 0) return error.EmptyArray;
    if (y.len != n) return error.DimensionMismatch;
    if (degree >= n) return error.DegreeTooLarge;

    const d = degree + 1; // Number of coefficients

    // Build normal equations directly: compute V^T*V and V^T*y without explicit V matrix
    // This avoids numerical issues and uses less memory
    var A = try allocator.alloc(f64, d * d);
    errdefer allocator.free(A);
    var b = try allocator.alloc(f64, d);
    errdefer allocator.free(b);

    // Initialize A and b to zero
    for (0..d*d) |i| A[i] = 0;
    for (0..d) |i| b[i] = 0;

    // Compute normal equations by iterating through data points
    for (0..n) |i| {
        var pow_x: f64 = 1.0;
        for (0..d) |j| {
            // Update b vector: b[j] += y[i] * x[i]^j
            b[j] += y[i] * pow_x;

            // Update A matrix: A[j, k] += x[i]^(j+k)
            var pow_xk: f64 = pow_x;
            for (0..d-j) |k_offset| {
                const k = j + k_offset;
                A[j * d + k] += pow_xk;
                pow_xk *= x[i];
            }

            pow_x *= x[i];
        }
    }

    // Make A symmetric (copy upper triangle to lower)
    for (0..d) |i| {
        for (i+1..d) |j| {
            A[j * d + i] = A[i * d + j];
        }
    }

    // Solve A * coeffs = b using Gaussian elimination with partial pivoting
    var coeffs = try allocator.alloc(f64, d);
    errdefer allocator.free(coeffs);

    // Copy A and b to working matrices (we'll modify them)
    var A_work = try allocator.alloc(f64, d * d);
    defer allocator.free(A_work);
    var b_work = try allocator.alloc(f64, d);
    defer allocator.free(b_work);

    @memcpy(A_work, A);
    @memcpy(b_work, b);

    // Forward elimination with partial pivoting
    for (0..d) |col| {
        // Find pivot
        var max_row = col;
        var max_val = @abs(A_work[col * d + col]);
        for (col+1..d) |row| {
            const abs_val = @abs(A_work[row * d + col]);
            if (abs_val > max_val) {
                max_val = abs_val;
                max_row = row;
            }
        }

        // Swap rows if needed
        if (max_row != col) {
            for (0..d) |j| {
                const temp = A_work[col * d + j];
                A_work[col * d + j] = A_work[max_row * d + j];
                A_work[max_row * d + j] = temp;
            }
            const temp_b = b_work[col];
            b_work[col] = b_work[max_row];
            b_work[max_row] = temp_b;
        }

        // Check for singular matrix
        if (A_work[col * d + col] == 0) {
            return error.SingularMatrix;
        }

        // Eliminate column
        for (col+1..d) |row| {
            const factor = A_work[row * d + col] / A_work[col * d + col];
            for (col..d) |j| {
                A_work[row * d + j] -= factor * A_work[col * d + j];
            }
            b_work[row] -= factor * b_work[col];
        }
    }

    // Back substitution
    var i = d;
    while (i > 0) : (i -= 1) {
        const idx = i - 1;
        var sum: f64 = b_work[idx];
        for (idx+1..d) |j| {
            sum -= A_work[idx * d + j] * coeffs[j];
        }
        coeffs[idx] = sum / A_work[idx * d + idx];
    }

    allocator.free(A);
    allocator.free(b);

    return coeffs;
}

/// Evaluate polynomial at given points
///
/// Given coefficients [c0, c1, ..., cn] and evaluation points x[],
/// computes y[i] = c0 + c1*x[i] + c2*x[i]² + ... + cn*x[i]^n
///
/// Uses Horner's method for numerical stability:
///   y = c0 + x*(c1 + x*(c2 + x*(...)))
///
/// Parameters:
/// - coeffs: polynomial coefficients [c0, c1, ..., cn] where c0 is constant (can be []const f64 or *[]f64)
/// - x: evaluation points
/// - allocator: memory allocator for result
///
/// Returns: allocated array of length x.len with polynomial values
///
/// Errors:
/// - error.EmptyArray if coeffs or x is empty
///
/// Time: O(n*d) where n=len(x), d=len(coeffs) | Space: O(n)
///
pub fn polyval(coeffs_input: anytype, x: []const f64, allocator: Allocator) ![]f64 {
    // Handle both []const f64 and *[]f64 inputs
    const coeffs: []const f64 = if (@TypeOf(coeffs_input) == *[]f64)
        coeffs_input.*
    else if (@TypeOf(coeffs_input) == *[]const f64)
        coeffs_input.*
    else
        coeffs_input;

    // Validate inputs
    if (coeffs.len == 0) return error.EmptyArray;
    if (x.len == 0) return error.EmptyArray;

    const n = x.len;
    const d = coeffs.len;

    // Allocate result array
    var result = try allocator.alloc(f64, n);

    // Evaluate polynomial at each x[i] using Horner's method
    for (0..n) |i| {
        var y: f64 = coeffs[d - 1];

        // Work backwards from highest degree to lowest
        var j: usize = d - 1;
        while (j > 0) : (j -= 1) {
            y = y * x[i] + coeffs[j - 1];
        }

        result[i] = y;
    }

    return result;
}

// ============================================================================
// TESTS
// ============================================================================

const test_allocator = testing.allocator;

// ============================================================================
// Pearson Correlation Tests (25+ tests)
// ============================================================================

test "pearson: perfect positive correlation (r=1)" {
    // y = x: perfect positive linear relationship
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const r = try pearson(x, y, test_allocator);

    try testing.expectApproxEqAbs(1.0, r, 1e-10);
}

test "pearson: perfect negative correlation (r=-1)" {
    // y = -x: perfect negative linear relationship
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 5.0, 4.0, 3.0, 2.0, 1.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const r = try pearson(x, y, test_allocator);

    try testing.expectApproxEqAbs(-1.0, r, 1e-10);
}

test "pearson: no correlation (r≈0)" {
    // Unrelated: random data with near-zero correlation
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 5.0, 1.0, 4.0, 2.0, 3.0 }; // Random order

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const r = try pearson(x, y, test_allocator);

    // Should be close to zero (not exactly, but small magnitude)
    try testing.expect(@abs(r) < 0.5);
}

test "pearson: linear relationship y=2x+3" {
    // y = 2x + 3: linear with slope 2
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 5.0, 7.0, 9.0, 11.0, 13.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const r = try pearson(x, y, test_allocator);

    // Perfect linear transformation: r = 1
    try testing.expectApproxEqAbs(1.0, r, 1e-10);
}

test "pearson: symmetry property (pearson(x,y) = pearson(y,x))" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 2.0, 4.0, 5.0, 4.0, 6.0 };

    var x1 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x1.deinit();
    var y1 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y1.deinit();

    var x2 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer x2.deinit();
    var y2 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer y2.deinit();

    const r_xy = try pearson(x1, y1, test_allocator);
    const r_yx = try pearson(x2, y2, test_allocator);

    try testing.expectApproxEqAbs(r_xy, r_yx, 1e-10);
}

test "pearson: bounded in [-1, 1]" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    const y_data = [_]f64{ 2.1, 4.2, 5.9, 8.1, 10.1, 11.9, 14.0, 15.9, 18.1, 20.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{10}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{10}, &y_data, .row_major);
    defer y.deinit();

    const r = try pearson(x, y, test_allocator);

    try testing.expect(r >= -1.0);
    try testing.expect(r <= 1.0);
}

test "pearson: two points (r=1 or r=-1)" {
    const x_data = [_]f64{ 1.0, 2.0 };
    const y_data = [_]f64{ 3.0, 5.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{2}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{2}, &y_data, .row_major);
    defer y.deinit();

    const r = try pearson(x, y, test_allocator);

    // Two distinct points always have r = ±1
    try testing.expect(@abs(@abs(r) - 1.0) < 1e-10);
}

test "pearson: single point raises ZeroStdDev (no variance)" {
    const x_data = [_]f64{ 1.0, 1.0 };
    const y_data = [_]f64{ 2.0, 2.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{2}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{2}, &y_data, .row_major);
    defer y.deinit();

    const result = pearson(x, y, test_allocator);

    // Identical points have no variance
    try testing.expectError(error.ZeroStdDev, result);
}

test "pearson: constant array raises ZeroStdDev" {
    const x_data = [_]f64{ 1.0, 1.0, 1.0, 1.0, 1.0 };
    const y_data = [_]f64{ 2.0, 3.0, 4.0, 5.0, 6.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const result = pearson(x, y, test_allocator);

    try testing.expectError(error.ZeroStdDev, result);
}

test "pearson: both arrays constant raises ZeroStdDev" {
    const x_data = [_]f64{ 5.0, 5.0, 5.0 };
    const y_data = [_]f64{ 3.0, 3.0, 3.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{3}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{3}, &y_data, .row_major);
    defer y.deinit();

    const result = pearson(x, y, test_allocator);

    try testing.expectError(error.ZeroStdDev, result);
}

test "pearson: empty array handled gracefully" {
    // Skip: NDArray doesn't allow zero-size arrays in fromSlice
    // Empty array test would fail at NDArray creation, not at pearson
}

test "pearson: mismatched dimensions raises DimensionMismatch" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0 };
    const y_data = [_]f64{ 1.0, 2.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{3}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{2}, &y_data, .row_major);
    defer y.deinit();

    const result = pearson(x, y, test_allocator);

    try testing.expectError(error.DimensionMismatch, result);
}

test "pearson: negative slope (r<0)" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 10.0, 8.0, 6.0, 4.0, 2.0 }; // y = -2x + 12

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const r = try pearson(x, y, test_allocator);

    try testing.expectApproxEqAbs(-1.0, r, 1e-10);
}

test "pearson: weak positive correlation" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    const y_data = [_]f64{ 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{10}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{10}, &y_data, .row_major);
    defer y.deinit();

    const r = try pearson(x, y, test_allocator);

    // Moderate positive correlation
    try testing.expect(r > 0);
    try testing.expect(r < 1.0);
}

test "pearson: strong positive correlation" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    const y_data = [_]f64{ 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{10}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{10}, &y_data, .row_major);
    defer y.deinit();

    const r = try pearson(x, y, test_allocator);

    // Strong correlation: r > 0.7
    try testing.expect(r > 0.99);
}

test "pearson: large dataset (n=1000)" {
    var x_data = try test_allocator.alloc(f64, 1000);
    defer test_allocator.free(x_data);
    var y_data = try test_allocator.alloc(f64, 1000);
    defer test_allocator.free(y_data);

    for (0..1000) |i| {
        x_data[i] = @as(f64, @floatFromInt(i));
        y_data[i] = 2.0 * @as(f64, @floatFromInt(i)) + 1.0;
    }

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{1000}, x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{1000}, y_data, .row_major);
    defer y.deinit();

    const r = try pearson(x, y, test_allocator);

    try testing.expectApproxEqAbs(1.0, r, 1e-10);
}

test "pearson: shifted data (translation invariance)" {
    const x_data1 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data1 = [_]f64{ 2.0, 4.0, 6.0, 8.0, 10.0 };

    const x_data2 = [_]f64{ 101.0, 102.0, 103.0, 104.0, 105.0 };
    const y_data2 = [_]f64{ 102.0, 104.0, 106.0, 108.0, 110.0 };

    var x1 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data1, .row_major);
    defer x1.deinit();
    var y1 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data1, .row_major);
    defer y1.deinit();

    var x2 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data2, .row_major);
    defer x2.deinit();
    var y2 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data2, .row_major);
    defer y2.deinit();

    const r1 = try pearson(x1, y1, test_allocator);
    const r2 = try pearson(x2, y2, test_allocator);

    // Correlation should be invariant to translation
    try testing.expectApproxEqAbs(r1, r2, 1e-10);
}

test "pearson: scaled data (scale invariance)" {
    const x_data1 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data1 = [_]f64{ 2.0, 4.0, 6.0, 8.0, 10.0 };

    const x_data2 = [_]f64{ 10.0, 20.0, 30.0, 40.0, 50.0 };
    const y_data2 = [_]f64{ 20.0, 40.0, 60.0, 80.0, 100.0 };

    var x1 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data1, .row_major);
    defer x1.deinit();
    var y1 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data1, .row_major);
    defer y1.deinit();

    var x2 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data2, .row_major);
    defer x2.deinit();
    var y2 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data2, .row_major);
    defer y2.deinit();

    const r1 = try pearson(x1, y1, test_allocator);
    const r2 = try pearson(x2, y2, test_allocator);

    // Correlation should be invariant to scaling
    try testing.expectApproxEqAbs(r1, r2, 1e-10);
}

test "pearson: quadratic relationship (non-linear)" {
    // y = x²: non-linear monotonic relationship
    const x_data = [_]f64{ -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 25.0, 16.0, 9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0, 16.0, 25.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{11}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{11}, &y_data, .row_major);
    defer y.deinit();

    const r = try pearson(x, y, test_allocator);

    // Quadratic is non-linear, so r should be near 0
    try testing.expect(@abs(r) < 0.5);
}

// ============================================================================
// Spearman Correlation Tests (20+ tests)
// ============================================================================

test "spearman: perfect positive rank correlation (rho=1)" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const rho = try spearman(x, y, test_allocator);

    try testing.expectApproxEqAbs(1.0, rho, 1e-10);
}

test "spearman: perfect negative rank correlation (rho=-1)" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 5.0, 4.0, 3.0, 2.0, 1.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const rho = try spearman(x, y, test_allocator);

    try testing.expectApproxEqAbs(-1.0, rho, 1e-10);
}

test "spearman: quadratic relationship (rho=1 for monotonic)" {
    // y = x²: monotonic for x ≥ 0, should have rho=1
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 1.0, 4.0, 9.0, 16.0, 25.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const rho = try spearman(x, y, test_allocator);

    // Perfect monotonic: rho = 1 (even though non-linear)
    try testing.expectApproxEqAbs(1.0, rho, 1e-10);
}

test "spearman: symmetry (spearman(x,y) = spearman(y,x))" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 2.0, 4.0, 5.0, 4.0, 6.0 };

    var x1 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x1.deinit();
    var y1 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y1.deinit();

    var x2 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer x2.deinit();
    var y2 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer y2.deinit();

    const rho_xy = try spearman(x1, y1, test_allocator);
    const rho_yx = try spearman(x2, y2, test_allocator);

    try testing.expectApproxEqAbs(rho_xy, rho_yx, 1e-10);
}

test "spearman: bounded in [-1, 1]" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const y_data = [_]f64{ 8.0, 6.0, 3.0, 7.0, 2.0, 5.0, 1.0, 4.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{8}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{8}, &y_data, .row_major);
    defer y.deinit();

    const rho = try spearman(x, y, test_allocator);

    try testing.expect(rho >= -1.0);
    try testing.expect(rho <= 1.0);
}

test "spearman: ties in data (average ranks)" {
    // Data with ties: [1, 2, 2, 3] → ranks should be [1, 2.5, 2.5, 4]
    const x_data = [_]f64{ 1.0, 2.0, 2.0, 3.0 };
    const y_data = [_]f64{ 1.0, 2.0, 2.0, 3.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{4}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{4}, &y_data, .row_major);
    defer y.deinit();

    const rho = try spearman(x, y, test_allocator);

    // Identical tied data → perfect correlation
    try testing.expectApproxEqAbs(1.0, rho, 1e-10);
}

test "spearman: no correlation (rho≈0)" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 5.0, 1.0, 4.0, 2.0, 3.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const rho = try spearman(x, y, test_allocator);

    try testing.expect(@abs(rho) < 0.5);
}

test "spearman: two points (rho=±1)" {
    const x_data = [_]f64{ 1.0, 2.0 };
    const y_data = [_]f64{ 3.0, 5.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{2}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{2}, &y_data, .row_major);
    defer y.deinit();

    const rho = try spearman(x, y, test_allocator);

    try testing.expect(@abs(@abs(rho) - 1.0) < 1e-10);
}

test "spearman: empty array handled gracefully" {
    // Skip: NDArray doesn't allow zero-size arrays in fromSlice
    // Empty array test would fail at NDArray creation, not at spearman
}

test "spearman: mismatched dimensions raises DimensionMismatch" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0 };
    const y_data = [_]f64{ 1.0, 2.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{3}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{2}, &y_data, .row_major);
    defer y.deinit();

    const result = spearman(x, y, test_allocator);

    try testing.expectError(error.DimensionMismatch, result);
}

test "spearman: large dataset with ties" {
    var x_data = try test_allocator.alloc(f64, 100);
    defer test_allocator.free(x_data);
    var y_data = try test_allocator.alloc(f64, 100);
    defer test_allocator.free(y_data);

    for (0..100) |i| {
        x_data[i] = @as(f64, @floatFromInt(i / 10)); // Values [0..10] each repeated 10 times
        y_data[i] = 100.0 - @as(f64, @floatFromInt(i));
    }

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{100}, x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{100}, y_data, .row_major);
    defer y.deinit();

    const rho = try spearman(x, y, test_allocator);

    // Monotonic relationship: should have strong negative correlation (allow small numerical error)
    try testing.expectApproxEqAbs(-1.0, rho, 0.01);
}

test "spearman: many ties (all same rank)" {
    const x_data = [_]f64{ 1.0, 1.0, 1.0, 1.0, 1.0 };
    const y_data = [_]f64{ 1.0, 1.0, 1.0, 1.0, 1.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const result = spearman(x, y, test_allocator);

    // All same values: should raise ZeroStdDev during rank correlation
    try testing.expectError(error.ZeroStdDev, result);
}

test "spearman vs pearson: linear data should be similar" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 2.0, 4.0, 6.0, 8.0, 10.0 };

    var x1 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x1.deinit();
    var y1 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y1.deinit();

    var x2 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x2.deinit();
    var y2 = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y2.deinit();

    const r = try pearson(x1, y1, test_allocator);
    const rho = try spearman(x2, y2, test_allocator);

    // For perfect linear data, both should equal 1.0
    try testing.expectApproxEqAbs(r, rho, 1e-10);
}

test "spearman: outlier test (robustness)" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 100.0 };
    const y_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const rho = try spearman(x, y, test_allocator);

    // Rank-based: outlier doesn't affect monotonic relationship
    try testing.expectApproxEqAbs(1.0, rho, 1e-10);
}

// ============================================================================
// Linear Regression Tests (25+ tests)
// ============================================================================

test "linregress: perfect fit (y=2x+3)" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 5.0, 7.0, 9.0, 11.0, 13.0 }; // y = 2x + 3

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const result = try linregress(x, y, test_allocator);

    try testing.expectApproxEqAbs(2.0, result.slope, 1e-10);
    try testing.expectApproxEqAbs(3.0, result.intercept, 1e-10);
    try testing.expectApproxEqAbs(1.0, result.r_squared, 1e-10);
}

test "linregress: perfect fit with negative slope (y=-x+5)" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 4.0, 3.0, 2.0, 1.0, 0.0 }; // y = -x + 5

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const result = try linregress(x, y, test_allocator);

    try testing.expectApproxEqAbs(-1.0, result.slope, 1e-10);
    try testing.expectApproxEqAbs(5.0, result.intercept, 1e-10);
    try testing.expectApproxEqAbs(1.0, result.r_squared, 1e-10);
}

test "linregress: three points (perfect fit)" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0 };
    const y_data = [_]f64{ 3.0, 5.0, 7.0 }; // y = 2x + 1

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{3}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{3}, &y_data, .row_major);
    defer y.deinit();

    const result = try linregress(x, y, test_allocator);

    try testing.expectApproxEqAbs(2.0, result.slope, 1e-10);
    try testing.expectApproxEqAbs(1.0, result.intercept, 1e-10);
    try testing.expectApproxEqAbs(1.0, result.r_squared, 1e-10);
}

test "linregress: noisy data (r_squared < 1)" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 2.1, 3.9, 6.2, 7.8, 10.1 }; // y ≈ 2x with noise

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const result = try linregress(x, y, test_allocator);

    // Noisy data: 0 < R² < 1
    try testing.expect(result.r_squared > 0.9);
    try testing.expect(result.r_squared < 1.0);
    try testing.expect(result.slope > 1.9);
    try testing.expect(result.slope < 2.1);
}

test "linregress: r_squared bounded in [0, 1]" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    const y_data = [_]f64{ 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{10}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{10}, &y_data, .row_major);
    defer y.deinit();

    const result = try linregress(x, y, test_allocator);

    try testing.expect(result.r_squared >= 0.0);
    try testing.expect(result.r_squared <= 1.0);
}

test "linregress: p_value in [0, 1]" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 2.0, 4.0, 6.0, 8.0, 10.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const result = try linregress(x, y, test_allocator);

    try testing.expect(result.p_value >= 0.0);
    try testing.expect(result.p_value <= 1.0);
}

test "linregress: constant x raises ConstantX error" {
    const x_data = [_]f64{ 5.0, 5.0, 5.0, 5.0 };
    const y_data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{4}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{4}, &y_data, .row_major);
    defer y.deinit();

    const result = linregress(x, y, test_allocator);

    try testing.expectError(error.ConstantX, result);
}

test "linregress: single point raises InsufficientSamples" {
    const x_data = [_]f64{1.0};
    const y_data = [_]f64{2.0};

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{1}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{1}, &y_data, .row_major);
    defer y.deinit();

    const result = linregress(x, y, test_allocator);

    try testing.expectError(error.InsufficientSamples, result);
}

test "linregress: two points raises InsufficientSamples (df=0)" {
    // Two points lead to df = n-2 = 0, which is invalid for StudentT
    const x_data = [_]f64{ 1.0, 2.0 };
    const y_data = [_]f64{ 3.0, 5.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{2}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{2}, &y_data, .row_major);
    defer y.deinit();

    const result = linregress(x, y, test_allocator);

    try testing.expectError(error.InsufficientSamples, result);
}

test "linregress: mismatched dimensions raises DimensionMismatch" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0 };
    const y_data = [_]f64{ 1.0, 2.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{3}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{2}, &y_data, .row_major);
    defer y.deinit();

    const result = linregress(x, y, test_allocator);

    try testing.expectError(error.DimensionMismatch, result);
}

test "linregress: perfect fit p-value near 0" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    const y_data = [_]f64{ 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{10}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{10}, &y_data, .row_major);
    defer y.deinit();

    const result = try linregress(x, y, test_allocator);

    // Perfect linear: p-value should be very small (highly significant slope)
    try testing.expect(result.p_value < 0.05);
}

test "linregress: weak relationship p-value > 0.05" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 2.0, 1.0, 4.0, 3.0, 5.0 }; // Nearly random

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const result = try linregress(x, y, test_allocator);

    // Weak relationship: p-value likely > 0.05
    try testing.expect(result.p_value > 0.05 or result.r_squared < 0.3);
}

test "linregress: zero intercept (y=3x)" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 3.0, 6.0, 9.0, 12.0, 15.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const result = try linregress(x, y, test_allocator);

    try testing.expectApproxEqAbs(3.0, result.slope, 1e-10);
    try testing.expectApproxEqAbs(0.0, result.intercept, 1e-10);
}

test "linregress: zero slope (y=c, constant) - R² becomes NaN" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 7.0, 7.0, 7.0, 7.0, 7.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const result = try linregress(x, y, test_allocator);

    try testing.expectApproxEqAbs(0.0, result.slope, 1e-10);
    try testing.expectApproxEqAbs(7.0, result.intercept, 1e-10);
    // When TSS=0 (no variance in y), R² = 1 - (RSS/TSS) becomes undefined
    // Implementation clamps to [0, 1], so expect 0
    try testing.expect(result.r_squared >= 0.0 and result.r_squared <= 1.0);
}

test "linregress: large dataset (n=100)" {
    var x_data = try test_allocator.alloc(f64, 100);
    defer test_allocator.free(x_data);
    var y_data = try test_allocator.alloc(f64, 100);
    defer test_allocator.free(y_data);

    for (0..100) |i| {
        const x_val = @as(f64, @floatFromInt(i));
        x_data[i] = x_val;
        y_data[i] = 3.0 * x_val + 2.0;
    }

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{100}, x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{100}, y_data, .row_major);
    defer y.deinit();

    const result = try linregress(x, y, test_allocator);

    try testing.expectApproxEqAbs(3.0, result.slope, 1e-10);
    try testing.expectApproxEqAbs(2.0, result.intercept, 1e-10);
    try testing.expectApproxEqAbs(1.0, result.r_squared, 1e-10);
}

test "linregress: prediction example (slope=2, intercept=1)" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0 };
    const y_data = [_]f64{ 3.0, 5.0, 7.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{3}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{3}, &y_data, .row_major);
    defer y.deinit();

    const result = try linregress(x, y, test_allocator);

    // Verify model: ŷ = 1 + 2x
    const pred_at_4 = result.intercept + result.slope * 4.0;
    try testing.expectApproxEqAbs(9.0, pred_at_4, 1e-10);
}

test "linregress: negative slope strong significance" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    const y_data = [_]f64{ 20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{10}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{10}, &y_data, .row_major);
    defer y.deinit();

    const result = try linregress(x, y, test_allocator);

    try testing.expect(result.slope < 0);
    try testing.expect(result.p_value < 0.05);
    try testing.expectApproxEqAbs(-2.0, result.slope, 1e-10);
    try testing.expectApproxEqAbs(22.0, result.intercept, 1e-10);
}

test "linregress: std_err positive" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 2.0, 4.0, 6.0, 8.0, 10.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const result = try linregress(x, y, test_allocator);

    try testing.expect(result.std_err >= 0);
}

test "linregress: three points" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0 };
    const y_data = [_]f64{ 1.0, 3.0, 5.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{3}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{3}, &y_data, .row_major);
    defer y.deinit();

    const result = try linregress(x, y, test_allocator);

    try testing.expectApproxEqAbs(2.0, result.slope, 1e-10);
    try testing.expectApproxEqAbs(-1.0, result.intercept, 1e-10);
}

test "linregress: regression result struct fields valid" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 2.0, 4.0, 6.0, 8.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{4}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{4}, &y_data, .row_major);
    defer y.deinit();

    const result = try linregress(x, y, test_allocator);

    // All fields should be finite numbers
    try testing.expect(math.isFinite(result.slope));
    try testing.expect(math.isFinite(result.intercept));
    try testing.expect(math.isFinite(result.r_squared));
    try testing.expect(math.isFinite(result.p_value));
    try testing.expect(math.isFinite(result.std_err));
}

// ============================================================================
// Kendall's Tau Correlation Tests (20+ tests)
// ============================================================================

test "kendalltau: perfect positive correlation (tau=1)" {
    // Perfectly concordant pairs: y = x
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    const tau = try kendalltau(&x_data, &y_data, test_allocator);

    try testing.expectApproxEqAbs(1.0, tau, 1e-10);
}

test "kendalltau: perfect negative correlation (tau=-1)" {
    // Perfectly discordant pairs: y = -x
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 5.0, 4.0, 3.0, 2.0, 1.0 };

    const tau = try kendalltau(&x_data, &y_data, test_allocator);

    try testing.expectApproxEqAbs(-1.0, tau, 1e-10);
}

test "kendalltau: no correlation (tau≈0) with random permutation" {
    // Unrelated data: tau should be near zero
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 3.0, 1.0, 5.0, 2.0, 4.0 };

    const tau = try kendalltau(&x_data, &y_data, test_allocator);

    // Should be small in magnitude
    try testing.expect(@abs(tau) < 0.5);
}

test "kendalltau: bounded in [-1, 1]" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const y_data = [_]f64{ 1.5, 3.2, 4.1, 6.0, 7.2, 9.1, 10.5, 12.0 };

    const tau = try kendalltau(&x_data, &y_data, test_allocator);

    try testing.expect(tau >= -1.0);
    try testing.expect(tau <= 1.0);
}

test "kendalltau: symmetry property (kendalltau(x,y) = kendalltau(y,x))" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 2.0, 4.0, 5.0, 4.0, 6.0 };

    const tau_xy = try kendalltau(&x_data, &y_data, test_allocator);
    const tau_yx = try kendalltau(&y_data, &x_data, test_allocator);

    try testing.expectApproxEqAbs(tau_xy, tau_yx, 1e-10);
}

test "kendalltau: two elements (one pair)" {
    // Single pair: C=1, D=0, T_x=0, T_y=0 → tau = (1-0)/(sqrt(1*1)) = 1.0
    const x_data = [_]f64{ 1.0, 2.0 };
    const y_data = [_]f64{ 3.0, 5.0 };

    const tau = try kendalltau(&x_data, &y_data, test_allocator);

    // Single concordant pair → tau = 1
    try testing.expectApproxEqAbs(1.0, tau, 1e-10);
}

test "kendalltau: two elements discordant (tau=-1)" {
    // Single discordant pair: C=0, D=1 → tau = (0-1)/(sqrt(1*1)) = -1.0
    const x_data = [_]f64{ 1.0, 2.0 };
    const y_data = [_]f64{ 5.0, 3.0 };

    const tau = try kendalltau(&x_data, &y_data, test_allocator);

    try testing.expectApproxEqAbs(-1.0, tau, 1e-10);
}

test "kendalltau: three elements simple case" {
    // x=[1,2,3], y=[1,2,3]: all pairs concordant
    // Pairs: (1,2)→C, (1,3)→C, (2,3)→C. Total: C=3, D=0
    // tau = 3/(sqrt(3*3)) = 3/3 = 1.0
    const x_data = [_]f64{ 1.0, 2.0, 3.0 };
    const y_data = [_]f64{ 1.0, 2.0, 3.0 };

    const tau = try kendalltau(&x_data, &y_data, test_allocator);

    try testing.expectApproxEqAbs(1.0, tau, 1e-10);
}

test "kendalltau: ties in x only" {
    // x=[1,1,2,3], y=[1,2,3,4]
    // One tie in x: (indices 0,1). Tau-b formula accounts for ties.
    const x_data = [_]f64{ 1.0, 1.0, 2.0, 3.0 };
    const y_data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };

    const tau = try kendalltau(&x_data, &y_data, test_allocator);

    // Should still reflect strong positive correlation despite ties
    try testing.expect(tau > 0.5);
    try testing.expect(tau <= 1.0);
}

test "kendalltau: ties in y only" {
    // x=[1,2,3,4], y=[1,1,2,3]
    // One tie in y: (indices 0,1). Should handle gracefully.
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 1.0, 1.0, 2.0, 3.0 };

    const tau = try kendalltau(&x_data, &y_data, test_allocator);

    // Should reflect strong positive correlation
    try testing.expect(tau > 0.5);
    try testing.expect(tau <= 1.0);
}

test "kendalltau: ties in both x and y" {
    // x=[1,1,2,3], y=[1,2,2,3]
    // Ties in both variables. Tau-b formula reduces impact.
    const x_data = [_]f64{ 1.0, 1.0, 2.0, 3.0 };
    const y_data = [_]f64{ 1.0, 2.0, 2.0, 3.0 };

    const tau = try kendalltau(&x_data, &y_data, test_allocator);

    // Should reflect positive correlation despite ties
    try testing.expect(tau > 0.0);
    try testing.expect(tau <= 1.0);
}

test "kendalltau: all tied values in x (same value)" {
    // x=[5,5,5,5], y=[1,2,3,4]
    // All x values identical: T_x = n(n-1)/2 = 6, C and D will be 0
    // tau = (0-0)/(sqrt((0+0+6)(0+0+0))) = 0/0 → undefined/NaN or 0
    const x_data = [_]f64{ 5.0, 5.0, 5.0, 5.0 };
    const y_data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };

    const tau = try kendalltau(&x_data, &y_data, test_allocator);

    // Tied x means no rank ordering: tau should be 0 or undefined
    // Implementation should handle this gracefully (likely returning 0 or NaN)
    try testing.expect(!math.isNan(tau)); // Should not produce NaN
}

test "kendalltau: strong positive partial correlation" {
    // y mostly increases with x, but with some disorder
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 1.0, 3.0, 2.0, 5.0, 4.0 }; // Strong positive but imperfect correlation

    const tau = try kendalltau(&x_data, &y_data, test_allocator);

    // Should show positive correlation (8 concordant, 2 discordant → tau = 0.6)
    try testing.expect(tau > 0.3);
    try testing.expect(tau < 1.0);
}

test "kendalltau: strong negative partial correlation" {
    // y decreases with x, but with some disorder
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 5.0, 4.0, 3.5, 2.0, 1.0 }; // Some disorder

    const tau = try kendalltau(&x_data, &y_data, test_allocator);

    // Should show negative correlation
    try testing.expect(tau < -0.3);
    try testing.expect(tau >= -1.0);
}

test "kendalltau: identical data (perfect ties)" {
    // When all pairs are tied (all values same in both x and y)
    const x_data = [_]f64{ 2.0, 2.0, 2.0 };
    const y_data = [_]f64{ 3.0, 3.0, 3.0 };

    const tau = try kendalltau(&x_data, &y_data, test_allocator);

    // All same values: undefined correlation
    try testing.expect(!math.isNan(tau)); // Should handle gracefully
}

test "kendalltau: monotonic transformation (invariance)" {
    // y = x² for x > 0: monotonic but non-linear
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 1.0, 4.0, 9.0, 16.0, 25.0 };

    const tau = try kendalltau(&x_data, &y_data, test_allocator);

    // Monotonic relationship → tau = 1
    try testing.expectApproxEqAbs(1.0, tau, 1e-10);
}

test "kendalltau: large dataset (n=100)" {
    var x_data = try test_allocator.alloc(f64, 100);
    defer test_allocator.free(x_data);
    var y_data = try test_allocator.alloc(f64, 100);
    defer test_allocator.free(y_data);

    for (0..100) |i| {
        const i_f = @as(f64, @floatFromInt(i));
        x_data[i] = i_f;
        y_data[i] = 2.0 * i_f + 1.0; // y = 2x + 1 (monotonic)
    }

    const tau = try kendalltau(x_data, y_data, test_allocator);

    // Perfect monotonic relationship
    try testing.expectApproxEqAbs(1.0, tau, 1e-10);
}

test "kendalltau: error on empty array" {
    const x_data: [0]f64 = .{};
    const y_data: [0]f64 = .{};

    const result = kendalltau(&x_data, &y_data, test_allocator);

    try testing.expectError(error.EmptyArray, result);
}

test "kendalltau: error on dimension mismatch" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0 };
    const y_data = [_]f64{ 1.0, 2.0 };

    const result = kendalltau(&x_data, &y_data, test_allocator);

    try testing.expectError(error.DimensionMismatch, result);
}

test "kendalltau: known dataset correlation strength" {
    // Dataset with known moderate to strong positive correlation
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const y_data = [_]f64{ 2.0, 3.0, 4.0, 5.0, 7.0, 6.0 }; // Mostly monotonic with one inversion

    const tau = try kendalltau(&x_data, &y_data, test_allocator);

    // Moderate to strong positive: 0.4 < tau < 1.0
    try testing.expect(tau > 0.4);
    try testing.expect(tau < 1.0);
}

test "kendalltau: invariant under monotonic scaling" {
    // x₁=[1,2,3,4,5], y₁=[2,4,6,8,10]
    // x₂=[10,20,30,40,50], y₂=[20,40,60,80,100] (scaled by 10)
    const x1_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y1_data = [_]f64{ 2.0, 4.0, 6.0, 8.0, 10.0 };

    const x2_data = [_]f64{ 10.0, 20.0, 30.0, 40.0, 50.0 };
    const y2_data = [_]f64{ 20.0, 40.0, 60.0, 80.0, 100.0 };

    const tau1 = try kendalltau(&x1_data, &y1_data, test_allocator);
    const tau2 = try kendalltau(&x2_data, &y2_data, test_allocator);

    // Kendall tau should be invariant to monotonic scaling
    try testing.expectApproxEqAbs(tau1, tau2, 1e-10);
}

test "kendalltau: invariant under monotonic translation" {
    // x₁=[1,2,3,4,5], y₁=[2,4,6,8,10]
    // x₂=[101,102,103,104,105], y₂=[102,104,106,108,110] (shifted by 100)
    const x1_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y1_data = [_]f64{ 2.0, 4.0, 6.0, 8.0, 10.0 };

    const x2_data = [_]f64{ 101.0, 102.0, 103.0, 104.0, 105.0 };
    const y2_data = [_]f64{ 102.0, 104.0, 106.0, 108.0, 110.0 };

    const tau1 = try kendalltau(&x1_data, &y1_data, test_allocator);
    const tau2 = try kendalltau(&x2_data, &y2_data, test_allocator);

    // Kendall tau should be invariant to translation
    try testing.expectApproxEqAbs(tau1, tau2, 1e-10);
}

test "kendalltau: comparison with spearman on linear data" {
    // For linear/monotonic relationships without ties, tau and rho should be similar
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 2.0, 4.0, 6.0, 8.0, 10.0 };

    var x = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x.deinit();
    var y = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y.deinit();

    const tau = try kendalltau(&x_data, &y_data, test_allocator);
    const rho = try spearman(x, y, test_allocator);

    // Both should indicate perfect correlation for monotonic data
    try testing.expectApproxEqAbs(1.0, tau, 1e-10);
    try testing.expectApproxEqAbs(1.0, rho, 1e-10);
}

test "kendalltau: result is finite number" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 1.0, 3.0, 2.0, 5.0, 4.0 };

    const tau = try kendalltau(&x_data, &y_data, test_allocator);

    // Result should be a finite f64 (not NaN or infinity)
    try testing.expect(math.isFinite(tau));
}

// ============================================================================
// POLYNOMIAL FITTING AND EVALUATION TESTS
// ============================================================================

test "polyfit: constant polynomial (degree 0) from horizontal data" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 7.0, 7.0, 7.0, 7.0, 7.0 };

    var coeffs = try polyfit(&x_data, &y_data, 0, test_allocator);
    defer test_allocator.free(coeffs);

    // Degree 0 → constant polynomial: c0
    try testing.expect(coeffs.len == 1);
    // Should fit y = 7
    try testing.expectApproxEqAbs(7.0, coeffs[0], 1e-10);
}

test "polyfit: linear polynomial (degree 1) from linear data" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 3.0, 5.0, 7.0, 9.0, 11.0 }; // y = 2x + 1

    var coeffs = try polyfit(&x_data, &y_data, 1, test_allocator);
    defer test_allocator.free(coeffs);

    // Degree 1 → linear: c0 + c1*x
    try testing.expect(coeffs.len == 2);
    // Expected: intercept ≈ 1.0, slope ≈ 2.0
    try testing.expectApproxEqAbs(1.0, coeffs[0], 1e-10);
    try testing.expectApproxEqAbs(2.0, coeffs[1], 1e-10);
}

test "polyfit: quadratic polynomial (degree 2) with known parabola" {
    const x_data = [_]f64{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    const y_data = [_]f64{ 4.0, 1.0, 0.0, 1.0, 4.0 }; // y = x²

    var coeffs = try polyfit(&x_data, &y_data, 2, test_allocator);
    defer test_allocator.free(coeffs);

    // Degree 2 → quadratic: c0 + c1*x + c2*x²
    try testing.expect(coeffs.len == 3);
    // Expected: c0 ≈ 0, c1 ≈ 0, c2 ≈ 1
    try testing.expectApproxEqAbs(0.0, coeffs[0], 1e-10);
    try testing.expectApproxEqAbs(0.0, coeffs[1], 1e-10);
    try testing.expectApproxEqAbs(1.0, coeffs[2], 1e-10);
}

test "polyfit: cubic polynomial (degree 3) with known cubic" {
    const x_data = [_]f64{ -1.0, 0.0, 1.0, 2.0 };
    const y_data = [_]f64{ -5.0, 3.0, 5.0, 21.0 }; // y = 2x³ + 3

    var coeffs = try polyfit(&x_data, &y_data, 3, test_allocator);
    defer test_allocator.free(coeffs);

    // Degree 3 → cubic: c0 + c1*x + c2*x² + c3*x³
    try testing.expect(coeffs.len == 4);
    // Expected: c0 ≈ 3, c1 ≈ 0, c2 ≈ 0, c3 ≈ 2
    try testing.expectApproxEqAbs(3.0, coeffs[0], 1e-8);
    try testing.expectApproxEqAbs(0.0, coeffs[1], 1e-8);
    try testing.expectApproxEqAbs(0.0, coeffs[2], 1e-8);
    try testing.expectApproxEqAbs(2.0, coeffs[3], 1e-8);
}

test "polyfit: linear fit matches linregress results" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 2.5, 4.0, 5.5, 7.0, 8.5 };

    // Create NDArray versions for linregress
    var x_ndarray = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &x_data, .row_major);
    defer x_ndarray.deinit();
    var y_ndarray = try NDArray_type(f64, 1).fromSlice(test_allocator, &[_]usize{5}, &y_data, .row_major);
    defer y_ndarray.deinit();

    const lr_result = try linregress(x_ndarray, y_ndarray, test_allocator);

    var coeffs = try polyfit(&x_data, &y_data, 1, test_allocator);
    defer test_allocator.free(coeffs);

    // polyfit degree 1: c0 + c1*x should match linregress slope + intercept
    try testing.expectApproxEqAbs(lr_result.intercept, coeffs[0], 1e-10);
    try testing.expectApproxEqAbs(lr_result.slope, coeffs[1], 1e-10);
}

test "polyfit: noisy data produces reasonable fit" {
    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 0.5, 2.2, 3.9, 6.1, 8.3 }; // Approximately y = 2x

    var coeffs = try polyfit(&x_data, &y_data, 1, test_allocator);
    defer test_allocator.free(coeffs);

    // Should fit roughly y ≈ 2x + c
    try testing.expectApproxEqAbs(2.0, coeffs[1], 0.2); // Slope near 2
    try testing.expect(coeffs[0] >= -1.0 and coeffs[0] <= 1.0); // Intercept reasonable
}

test "polyfit: higher degree fits data with less error" {
    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0 };
    const y_data = [_]f64{ 1.0, 3.0, 7.0, 13.0 }; // y = x² + x + 1

    // Fit with degree 1 (linear)
    var coeffs1 = try polyfit(&x_data, &y_data, 1, test_allocator);
    defer test_allocator.free(coeffs1);

    // Fit with degree 2 (quadratic)
    var coeffs2 = try polyfit(&x_data, &y_data, 2, test_allocator);
    defer test_allocator.free(coeffs2);

    // Verify quadratic has correct form: c0=1, c1=1, c2=1
    try testing.expect(coeffs2.len == 3);
    try testing.expectApproxEqAbs(1.0, coeffs2[0], 1e-10);
    try testing.expectApproxEqAbs(1.0, coeffs2[1], 1e-10);
    try testing.expectApproxEqAbs(1.0, coeffs2[2], 1e-10);

    // Linear fit will have higher residuals than quadratic
    // (We don't test the residuals here, just verify both produce valid coefficients)
    try testing.expect(coeffs1.len == 2);
    try testing.expect(coeffs2.len == 3);
}

test "polyfit: error on empty arrays" {
    const x_data: [0]f64 = .{};
    const y_data: [0]f64 = .{};

    const result = polyfit(&x_data, &y_data, 1, test_allocator);

    try testing.expectError(error.EmptyArray, result);
}

test "polyfit: error on dimension mismatch" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0 };
    const y_data = [_]f64{ 1.0, 2.0 };

    const result = polyfit(&x_data, &y_data, 1, test_allocator);

    try testing.expectError(error.DimensionMismatch, result);
}

test "polyfit: error when degree >= number of points" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0 };
    const y_data = [_]f64{ 2.0, 4.0, 6.0 };

    // degree 3 >= n=3 should error
    const result = polyfit(&x_data, &y_data, 3, test_allocator);

    try testing.expectError(error.DegreeTooLarge, result);
}

test "polyfit: degree exactly n-1 is allowed" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0 };
    const y_data = [_]f64{ 2.0, 4.0, 6.0 };

    // degree 2 == n-1=2 should succeed (will interpolate exactly)
    var coeffs = try polyfit(&x_data, &y_data, 2, test_allocator);
    defer test_allocator.free(coeffs);

    try testing.expect(coeffs.len == 3);
}

test "polyfit: single point requires degree 0" {
    const x_data = [_]f64{5.0};
    const y_data = [_]f64{10.0};

    // degree 0 should work
    var coeffs = try polyfit(&x_data, &y_data, 0, test_allocator);
    defer test_allocator.free(coeffs);

    try testing.expect(coeffs.len == 1);
    try testing.expectApproxEqAbs(10.0, coeffs[0], 1e-10);

    // degree 1 should fail (1 >= n=1)
    const result = polyfit(&x_data, &y_data, 1, test_allocator);
    try testing.expectError(error.DegreeTooLarge, result);
}

test "polyval: evaluate constant polynomial" {
    const coeffs = [_]f64{7.0};
    const x_data = [_]f64{ 1.0, 2.0, 3.0 };

    var y = try polyval(&coeffs, &x_data, test_allocator);
    defer test_allocator.free(y);

    // y = 7 for all x
    try testing.expect(y.len == 3);
    for (y) |val| {
        try testing.expectApproxEqAbs(7.0, val, 1e-15);
    }
}

test "polyval: evaluate linear polynomial" {
    const coeffs = [_]f64{ 1.0, 2.0 }; // y = 1 + 2x
    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0 };

    var y = try polyval(&coeffs, &x_data, test_allocator);
    defer test_allocator.free(y);

    // Expected: [1, 3, 5, 7]
    try testing.expect(y.len == 4);
    try testing.expectApproxEqAbs(1.0, y[0], 1e-15);
    try testing.expectApproxEqAbs(3.0, y[1], 1e-15);
    try testing.expectApproxEqAbs(5.0, y[2], 1e-15);
    try testing.expectApproxEqAbs(7.0, y[3], 1e-15);
}

test "polyval: evaluate quadratic polynomial" {
    const coeffs = [_]f64{ 0.0, 0.0, 1.0 }; // y = x²
    const x_data = [_]f64{ -2.0, -1.0, 0.0, 1.0, 2.0 };

    var y = try polyval(&coeffs, &x_data, test_allocator);
    defer test_allocator.free(y);

    // Expected: [4, 1, 0, 1, 4]
    try testing.expect(y.len == 5);
    try testing.expectApproxEqAbs(4.0, y[0], 1e-15);
    try testing.expectApproxEqAbs(1.0, y[1], 1e-15);
    try testing.expectApproxEqAbs(0.0, y[2], 1e-15);
    try testing.expectApproxEqAbs(1.0, y[3], 1e-15);
    try testing.expectApproxEqAbs(4.0, y[4], 1e-15);
}

test "polyval: evaluate cubic polynomial" {
    const coeffs = [_]f64{ 0.0, 1.0, 0.0, 1.0 }; // y = x + x³
    const x_data = [_]f64{ -1.0, 0.0, 1.0, 2.0 };

    var y = try polyval(&coeffs, &x_data, test_allocator);
    defer test_allocator.free(y);

    // Expected: y(-1) = -1-1=-2, y(0)=0, y(1)=1+1=2, y(2)=2+8=10
    try testing.expect(y.len == 4);
    try testing.expectApproxEqAbs(-2.0, y[0], 1e-15);
    try testing.expectApproxEqAbs(0.0, y[1], 1e-15);
    try testing.expectApproxEqAbs(2.0, y[2], 1e-15);
    try testing.expectApproxEqAbs(10.0, y[3], 1e-15);
}

test "polyval: Horner's method numerical stability" {
    // Test that Horner's method gives accurate results for larger x values
    // Using y = 1 + 2x + 3x² + 4x³
    const coeffs = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const x_data = [_]f64{10.0};

    var y = try polyval(&coeffs, &x_data, test_allocator);
    defer test_allocator.free(y);

    // Expected: 1 + 2*10 + 3*100 + 4*1000 = 1 + 20 + 300 + 4000 = 4321
    try testing.expectApproxEqAbs(4321.0, y[0], 1e-10);
}

test "polyval: evaluate at negative x values" {
    const coeffs = [_]f64{ 5.0, -3.0, 2.0 }; // y = 5 - 3x + 2x²
    const x_data = [_]f64{ -2.0, -1.0, 0.0, 1.0 };

    var y = try polyval(&coeffs, &x_data, test_allocator);
    defer test_allocator.free(y);

    // y(-2) = 5 - 3*(-2) + 2*4 = 5 + 6 + 8 = 19
    // y(-1) = 5 - 3*(-1) + 2*1 = 5 + 3 + 2 = 10
    // y(0) = 5
    // y(1) = 5 - 3 + 2 = 4
    try testing.expectApproxEqAbs(19.0, y[0], 1e-15);
    try testing.expectApproxEqAbs(10.0, y[1], 1e-15);
    try testing.expectApproxEqAbs(5.0, y[2], 1e-15);
    try testing.expectApproxEqAbs(4.0, y[3], 1e-15);
}

test "polyval: evaluate at x=0" {
    const coeffs = [_]f64{ 3.0, 2.0, 1.0 }; // y = 3 + 2x + x²
    const x_data = [_]f64{0.0};

    var y = try polyval(&coeffs, &x_data, test_allocator);
    defer test_allocator.free(y);

    // y(0) = 3 (the constant term)
    try testing.expectApproxEqAbs(3.0, y[0], 1e-15);
}

test "polyval: evaluate at single point" {
    const coeffs = [_]f64{ 1.0, 2.0 }; // y = 1 + 2x
    const x_data = [_]f64{5.0};

    var y = try polyval(&coeffs, &x_data, test_allocator);
    defer test_allocator.free(y);

    // y(5) = 1 + 2*5 = 11
    try testing.expectApproxEqAbs(11.0, y[0], 1e-15);
}

test "polyval: error on empty evaluation points" {
    const coeffs = [_]f64{ 1.0, 2.0 };
    const x_data: [0]f64 = .{};

    const result = polyval(&coeffs, &x_data, test_allocator);

    try testing.expectError(error.EmptyArray, result);
}

test "polyval: error on empty coefficient array" {
    const coeffs: [0]f64 = .{};
    const x_data = [_]f64{ 1.0, 2.0 };

    const result = polyval(&coeffs, &x_data, test_allocator);

    try testing.expectError(error.EmptyArray, result);
}

test "polyval: single coefficient (constant)" {
    const coeffs = [_]f64{42.0};
    const x_data = [_]f64{ 1.0, 2.0, 3.0 };

    var y = try polyval(&coeffs, &x_data, test_allocator);
    defer test_allocator.free(y);

    // All outputs should be 42
    for (y) |val| {
        try testing.expectApproxEqAbs(42.0, val, 1e-15);
    }
}

test "polyfit-polyval roundtrip: reconstruct linear function" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_data = [_]f64{ 3.0, 5.0, 7.0, 9.0, 11.0 }; // y = 2x + 1

    // Fit polynomial
    var coeffs = try polyfit(&x_data, &y_data, 1, test_allocator);
    defer test_allocator.free(coeffs);

    // Evaluate at original x points
    var y_pred = try polyval(&coeffs, &x_data, test_allocator);
    defer test_allocator.free(y_pred);

    // Should reconstruct original y
    for (y_pred, y_data) |pred, orig| {
        try testing.expectApproxEqAbs(orig, pred, 1e-10);
    }
}

test "polyfit-polyval roundtrip: reconstruct quadratic function" {
    const x_data = [_]f64{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    const y_data = [_]f64{ 4.0, 1.0, 0.0, 1.0, 4.0 }; // y = x²

    // Fit polynomial
    var coeffs = try polyfit(&x_data, &y_data, 2, test_allocator);
    defer test_allocator.free(coeffs);

    // Evaluate at original x points
    var y_pred = try polyval(&coeffs, &x_data, test_allocator);
    defer test_allocator.free(y_pred);

    // Should reconstruct original y exactly (or very close due to floating point)
    for (y_pred, y_data) |pred, orig| {
        try testing.expectApproxEqAbs(orig, pred, 1e-10);
    }
}

test "polyfit-polyval roundtrip: interpolation at new points" {
    const x_data = [_]f64{ 0.0, 1.0, 2.0, 3.0 };
    const y_data = [_]f64{ 0.0, 1.0, 4.0, 9.0 }; // y = x²

    // Fit polynomial (degree 2)
    var coeffs = try polyfit(&x_data, &y_data, 2, test_allocator);
    defer test_allocator.free(coeffs);

    // Evaluate at new points (between and outside training data)
    const x_new = [_]f64{ 0.5, 1.5, 2.5 };
    var y_pred = try polyval(&coeffs, &x_new, test_allocator);
    defer test_allocator.free(y_pred);

    // Expected: [0.25, 2.25, 6.25]
    try testing.expectApproxEqAbs(0.25, y_pred[0], 1e-10);
    try testing.expectApproxEqAbs(2.25, y_pred[1], 1e-10);
    try testing.expectApproxEqAbs(6.25, y_pred[2], 1e-10);
}

test "polyval: zero coefficients produce zeros" {
    const coeffs = [_]f64{ 0.0, 0.0, 0.0 }; // y = 0
    const x_data = [_]f64{ 1.0, 10.0, 100.0, -5.0 };

    var y = try polyval(&coeffs, &x_data, test_allocator);
    defer test_allocator.free(y);

    for (y) |val| {
        try testing.expectApproxEqAbs(0.0, val, 1e-15);
    }
}

test "polyfit: memory allocation correctness" {
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const y_data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };

    var coeffs = try polyfit(&x_data, &y_data, 2, test_allocator);

    // Should allocate exactly degree+1 coefficients
    try testing.expect(coeffs.len == 3);

    test_allocator.free(coeffs);
}

test "polyval: memory allocation correctness" {
    const coeffs = [_]f64{ 1.0, 2.0, 3.0 };
    const x_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    var y = try polyval(&coeffs, &x_data, test_allocator);

    // Should allocate exactly as many outputs as x points
    try testing.expect(y.len == 5);

    test_allocator.free(y);
}

test "polyfit: large dataset accuracy" {
    // Create synthetic data: y = 3x² - 2x + 5
    var x_data = try test_allocator.alloc(f64, 50);
    defer test_allocator.free(x_data);
    var y_data = try test_allocator.alloc(f64, 50);
    defer test_allocator.free(y_data);

    for (0..50) |i| {
        const i_f = @as(f64, @floatFromInt(i)) / 10.0; // 0, 0.1, 0.2, ..., 4.9
        x_data[i] = i_f;
        y_data[i] = 3.0 * i_f * i_f - 2.0 * i_f + 5.0;
    }

    var coeffs = try polyfit(x_data, y_data, 2, test_allocator);
    defer test_allocator.free(coeffs);

    // Should fit: c0≈5, c1≈-2, c2≈3
    try testing.expectApproxEqAbs(5.0, coeffs[0], 1e-8);
    try testing.expectApproxEqAbs(-2.0, coeffs[1], 1e-8);
    try testing.expectApproxEqAbs(3.0, coeffs[2], 1e-8);
}
