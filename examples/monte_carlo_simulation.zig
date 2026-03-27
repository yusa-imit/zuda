const std = @import("std");
const zuda = @import("zuda");

const NDArray = zuda.ndarray.NDArray;
const Normal = zuda.stats.distributions.Normal;
const Uniform = zuda.stats.distributions.Uniform;
const descriptive = zuda.stats.descriptive;
const integration = zuda.numeric.integration;

/// Monte Carlo Simulation Examples
/// Demonstrates:
/// 1. Estimating π using random sampling
/// 2. Numerical integration via Monte Carlo
/// 3. European option pricing (Black-Scholes)
/// 4. Confidence intervals and convergence analysis

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Monte Carlo Simulation Examples ===\n\n", .{});

    // Part 1: Estimate π using circle sampling
    try estimatePi(allocator);

    // Part 2: Monte Carlo integration
    try monteCarloIntegration(allocator);

    // Part 3: European option pricing
    try optionPricing(allocator);

    // Part 4: Convergence analysis
    try convergenceAnalysis(allocator);
}

/// Estimate π by sampling random points in unit square
/// Points inside quarter circle: x² + y² ≤ 1
/// π/4 ≈ (points inside) / (total points)
fn estimatePi(allocator: std.mem.Allocator) !void {
    std.debug.print("--- Part 1: Estimating π ---\n", .{});

    const n_samples: usize = 1000000;
    var uniform = try Uniform(f64).init(0.0, 1.0);
    var rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));

    var inside: usize = 0;
    var i: usize = 0;
    while (i < n_samples) : (i += 1) {
        const x = uniform.sample(rng.random());
        const y = uniform.sample(rng.random());
        if (x * x + y * y <= 1.0) {
            inside += 1;
        }
    }

    const pi_estimate = 4.0 * @as(f64, @floatFromInt(inside)) / @as(f64, @floatFromInt(n_samples));
    const error_pct = @abs(pi_estimate - std.math.pi) / std.math.pi * 100.0;

    std.debug.print("Samples: {}\n", .{n_samples});
    std.debug.print("Points inside circle: {}\n", .{inside});
    std.debug.print("π estimate: {d:.6}\n", .{pi_estimate});
    std.debug.print("True π: {d:.6}\n", .{std.math.pi});
    std.debug.print("Error: {d:.4}%\n\n", .{error_pct});

    // Compute confidence interval (CLT: std error ≈ sqrt(p(1-p)/n))
    const p = @as(f64, @floatFromInt(inside)) / @as(f64, @floatFromInt(n_samples));
    const std_error = @sqrt(p * (1.0 - p) / @as(f64, @floatFromInt(n_samples)));
    const ci_width = 1.96 * std_error * 4.0; // 95% CI for π estimate

    std.debug.print("95% Confidence Interval: [{d:.6}, {d:.6}]\n\n", .{ pi_estimate - ci_width, pi_estimate + ci_width });

    _ = allocator;
}

/// Monte Carlo integration: estimate ∫₀¹ f(x) dx
/// We use f(x) = sqrt(1 - x²) (quarter circle, exact = π/4)
fn monteCarloIntegration(allocator: std.mem.Allocator) !void {
    std.debug.print("--- Part 2: Monte Carlo Integration ---\n", .{});
    std.debug.print("Function: f(x) = sqrt(1 - x²) on [0, 1]\n", .{});
    std.debug.print("Exact integral: π/4 = {d:.6}\n\n", .{std.math.pi / 4.0});

    const n_samples: usize = 100000;
    var uniform = try Uniform(f64).init(0.0, 1.0);
    var rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));

    // Sample function values at random points
    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    var i: usize = 0;
    while (i < n_samples) : (i += 1) {
        const x = uniform.sample(rng.random());
        const fx = @sqrt(@max(0.0, 1.0 - x * x)); // sqrt(1 - x²)
        sum += fx;
        sum_sq += fx * fx;
    }

    // Estimator: (b-a) * mean(f(x_i))
    const interval_width = 1.0;
    const mean = sum / @as(f64, @floatFromInt(n_samples));
    const integral_estimate = interval_width * mean;

    // Variance: Var[estimate] = (b-a)² * Var[f(X)] / n
    const variance = (sum_sq / @as(f64, @floatFromInt(n_samples)) - mean * mean);
    const std_error = @sqrt(variance / @as(f64, @floatFromInt(n_samples)));

    std.debug.print("Samples: {}\n", .{n_samples});
    std.debug.print("Integral estimate: {d:.6}\n", .{integral_estimate});
    std.debug.print("Standard error: {d:.6}\n", .{std_error});
    std.debug.print("Error vs exact: {d:.6}\n\n", .{@abs(integral_estimate - std.math.pi / 4.0)});

    _ = allocator;
}

/// European call option pricing via Monte Carlo
/// Uses geometric Brownian motion: dS = μS dt + σS dW
/// Exact solution (Black-Scholes): C = S₀Φ(d₁) - Ke^(-rT)Φ(d₂)
fn optionPricing(allocator: std.mem.Allocator) !void {
    std.debug.print("--- Part 3: European Call Option Pricing ---\n", .{});

    // Option parameters
    const S0: f64 = 100.0; // Initial stock price
    const K: f64 = 105.0; // Strike price
    const T: f64 = 1.0; // Time to maturity (years)
    const r: f64 = 0.05; // Risk-free rate
    const sigma: f64 = 0.2; // Volatility
    const n_simulations: usize = 100000;

    std.debug.print("Parameters:\n", .{});
    std.debug.print("  S₀ = {d:.2} (initial price)\n", .{S0});
    std.debug.print("  K  = {d:.2} (strike price)\n", .{K});
    std.debug.print("  T  = {d:.2} years\n", .{T});
    std.debug.print("  r  = {d:.3} (risk-free rate)\n", .{r});
    std.debug.print("  σ  = {d:.3} (volatility)\n\n", .{sigma});

    var normal = try Normal(f64).init(0.0, 1.0);
    var rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp() + 42));

    // Simulate terminal stock prices: S_T = S₀ exp((r - σ²/2)T + σ√T Z)
    const drift = (r - 0.5 * sigma * sigma) * T;
    const diffusion = sigma * @sqrt(T);

    var payoff_sum: f64 = 0.0;
    var payoff_sq_sum: f64 = 0.0;

    var i: usize = 0;
    while (i < n_simulations) : (i += 1) {
        const Z = normal.sample(rng.random());
        const ST = S0 * @exp(drift + diffusion * Z);
        const payoff = @max(0.0, ST - K); // max(S_T - K, 0)
        payoff_sum += payoff;
        payoff_sq_sum += payoff * payoff;
    }

    // Discount expected payoff to present value
    const discount_factor = @exp(-r * T);
    const mean_payoff = payoff_sum / @as(f64, @floatFromInt(n_simulations));
    const option_price = discount_factor * mean_payoff;

    // Standard error
    const payoff_variance = payoff_sq_sum / @as(f64, @floatFromInt(n_simulations)) - mean_payoff * mean_payoff;
    const std_error = discount_factor * @sqrt(payoff_variance / @as(f64, @floatFromInt(n_simulations)));

    std.debug.print("Monte Carlo Results:\n", .{});
    std.debug.print("  Simulations: {}\n", .{n_simulations});
    std.debug.print("  Option price: ${d:.4}\n", .{option_price});
    std.debug.print("  Standard error: ${d:.4}\n", .{std_error});
    std.debug.print("  95% CI: [${d:.4}, ${d:.4}]\n\n", .{ option_price - 1.96 * std_error, option_price + 1.96 * std_error });

    // Compare with Black-Scholes formula
    const bs_price = try blackScholesCall(S0, K, T, r, sigma);
    std.debug.print("Black-Scholes price: ${d:.4}\n", .{bs_price});
    std.debug.print("Difference: ${d:.4}\n\n", .{@abs(option_price - bs_price)});

    _ = allocator;
}

/// Black-Scholes formula for European call option
fn blackScholesCall(S0: f64, K: f64, T: f64, r: f64, sigma: f64) !f64 {
    const d1 = (@log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * @sqrt(T));
    const d2 = d1 - sigma * @sqrt(T);

    // Standard normal CDF approximation (error < 1e-7)
    const Nd1 = normalCDF(d1);
    const Nd2 = normalCDF(d2);

    const call_price = S0 * Nd1 - K * @exp(-r * T) * Nd2;
    return call_price;
}

/// Approximation of standard normal CDF using error function
fn normalCDF(x: f64) f64 {
    return 0.5 * (1.0 + erf(x / @sqrt(2.0)));
}

/// Error function approximation (Abramowitz and Stegun)
fn erf(x: f64) f64 {
    const a1: f64 = 0.254829592;
    const a2: f64 = -0.284496736;
    const a3: f64 = 1.421413741;
    const a4: f64 = -1.453152027;
    const a5: f64 = 1.061405429;
    const p: f64 = 0.3275911;

    const sign = if (x < 0) @as(f64, -1.0) else @as(f64, 1.0);
    const abs_x = @abs(x);

    const t = 1.0 / (1.0 + p * abs_x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * @exp(-abs_x * abs_x);

    return sign * y;
}

/// Convergence analysis: how estimate improves with sample size
fn convergenceAnalysis(allocator: std.mem.Allocator) !void {
    std.debug.print("--- Part 4: Convergence Analysis ---\n", .{});
    std.debug.print("Estimating π with increasing sample sizes\n\n", .{});

    const sample_sizes = [_]usize{ 100, 1000, 10000, 100000, 1000000 };

    var uniform = try Uniform(f64).init(0.0, 1.0);
    var rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));

    std.debug.print("{s:>12} {s:>12} {s:>12} {s:>12}\n", .{ "Samples", "π Estimate", "Error", "Std Error" });
    std.debug.print("{s:-<12} {s:-<12} {s:-<12} {s:-<12}\n", .{ "", "", "", "" });

    for (sample_sizes) |n| {
        var inside: usize = 0;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const x = uniform.sample(rng.random());
            const y = uniform.sample(rng.random());
            if (x * x + y * y <= 1.0) {
                inside += 1;
            }
        }

        const pi_estimate = 4.0 * @as(f64, @floatFromInt(inside)) / @as(f64, @floatFromInt(n));
        const error_val = @abs(pi_estimate - std.math.pi);

        // Theoretical standard error
        const p = @as(f64, @floatFromInt(inside)) / @as(f64, @floatFromInt(n));
        const std_error = 4.0 * @sqrt(p * (1.0 - p) / @as(f64, @floatFromInt(n)));

        std.debug.print("{d:>12} {d:>12.6} {d:>12.6} {d:>12.6}\n", .{ n, pi_estimate, error_val, std_error });
    }

    std.debug.print("\nNote: Error scales as O(1/√n) for Monte Carlo methods\n", .{});

    _ = allocator;
}
