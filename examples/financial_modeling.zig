const std = @import("std");
const zuda = @import("zuda");

const NDArray = zuda.ndarray.NDArray;
const distributions = zuda.stats.distributions;
const descriptive = zuda.stats.descriptive;

/// Demonstrates financial modeling using zuda's scientific computing capabilities:
/// - Options pricing via Monte Carlo (Black-Scholes model)
/// - Value at Risk (VaR) calculation
/// - Risk metrics and performance analysis
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== FINANCIAL MODELING WITH ZUDA ===\n\n", .{});

    // Part 1: Monte Carlo Options Pricing (European Call Option)
    std.debug.print("Part 1: Monte Carlo Options Pricing\n", .{});
    std.debug.print("-------------------------------------\n", .{});
    std.debug.print("European Call Option (Black-Scholes Model)\n", .{});
    std.debug.print("Parameters:\n", .{});
    std.debug.print("  - Spot price: $100\n", .{});
    std.debug.print("  - Strike price: $105\n", .{});
    std.debug.print("  - Time to maturity: 1 year\n", .{});
    std.debug.print("  - Risk-free rate: 5%\n", .{});
    std.debug.print("  - Volatility: 20%\n", .{});
    std.debug.print("  - Simulations: 10,000\n\n", .{});

    const S0: f64 = 100.0; // Spot price
    const K: f64 = 105.0; // Strike price
    const T: f64 = 1.0; // Time to maturity (years)
    const r: f64 = 0.05; // Risk-free rate
    const sigma: f64 = 0.20; // Volatility
    const n_sims: usize = 10000;

    // Generate standard normal random variables
    const normal_dist = try distributions.Normal(f64).init(0.0, 1.0);
    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    var payoffs = try allocator.alloc(f64, n_sims);
    defer allocator.free(payoffs);

    // Monte Carlo simulation: S_T = S_0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
    for (0..n_sims) |i| {
        const Z = normal_dist.sample(random);
        const drift = (r - 0.5 * sigma * sigma) * T;
        const diffusion = sigma * @sqrt(T) * Z;
        const S_T = S0 * @exp(drift + diffusion);
        const payoff = @max(S_T - K, 0.0); // Call option payoff
        payoffs[i] = payoff;
    }

    // Discount payoffs to present value
    const discount_factor = @exp(-r * T);
    for (payoffs) |*payoff| {
        payoff.* *= discount_factor;
    }

    // Convert to NDArray for statistical analysis
    var payoffs_array = try NDArray(f64, 1).fromSlice(allocator, &.{n_sims}, payoffs, .row_major);
    defer payoffs_array.deinit();

    const option_price = descriptive.mean(f64, payoffs_array);
    const option_std = try descriptive.stdDev(f64, payoffs_array, 0);
    const std_error = option_std / @sqrt(@as(f64, @floatFromInt(n_sims)));

    std.debug.print("Results:\n", .{});
    std.debug.print("  Option Price: ${d:.4}\n", .{option_price});
    std.debug.print("  Standard Error: ${d:.4}\n", .{std_error});
    std.debug.print("  95% Confidence Interval: [${d:.4}, ${d:.4}]\n\n", .{
        option_price - 1.96 * std_error,
        option_price + 1.96 * std_error,
    });

    // Part 2: Portfolio Risk Analysis
    std.debug.print("Part 2: Portfolio Risk Analysis\n", .{});
    std.debug.print("--------------------------------\n", .{});
    std.debug.print("Portfolio: $1,000,000\n", .{});
    std.debug.print("Expected Annual Return: 10%\n", .{});
    std.debug.print("Annual Volatility: 15%\n", .{});
    std.debug.print("Time Horizon: 1 day\n\n", .{});

    // Portfolio parameters
    const portfolio_value: f64 = 1_000_000.0;
    const annual_return: f64 = 0.10;
    const annual_volatility: f64 = 0.15;
    const portfolio_mean = annual_return / 252.0; // Daily return (252 trading days)
    const portfolio_std = annual_volatility / @sqrt(252.0); // Daily std dev

    // Simulate daily returns
    const n_days: usize = 1000;
    var daily_returns = try allocator.alloc(f64, n_days);
    defer allocator.free(daily_returns);

    const return_dist = try distributions.Normal(f64).init(portfolio_mean, portfolio_std);

    for (0..n_days) |i| {
        daily_returns[i] = return_dist.sample(random);
    }

    // Convert to NDArray for analysis
    var returns_array = try NDArray(f64, 1).fromSlice(allocator, &.{n_days}, daily_returns, .row_major);
    defer returns_array.deinit();

    const mean_return = descriptive.mean(f64, returns_array);
    const std_return = try descriptive.stdDev(f64, returns_array, 0);

    std.debug.print("Simulated Statistics:\n", .{});
    std.debug.print("  Mean Daily Return: {d:.4}% (expected: {d:.4}%)\n", .{
        mean_return * 100.0,
        portfolio_mean * 100.0,
    });
    std.debug.print("  Daily Volatility: {d:.4}% (expected: {d:.4}%)\n\n", .{
        std_return * 100.0,
        portfolio_std * 100.0,
    });

    // Part 3: Value at Risk (VaR)
    std.debug.print("Part 3: Value at Risk (VaR)\n", .{});
    std.debug.print("-----------------------------\n", .{});

    // Sort returns to find percentiles
    std.mem.sort(f64, daily_returns, {}, std.sort.asc(f64));

    const confidence_level: f64 = 0.95;
    const var_index = @as(usize, @intFromFloat(@floor((1.0 - confidence_level) * @as(f64, @floatFromInt(n_days)))));

    const var_return = daily_returns[var_index];
    const var_amount = -var_return * portfolio_value; // Negative sign for loss

    std.debug.print("VaR (95%%, 1-day):\n", .{});
    std.debug.print("  Amount at Risk: ${d:.2}\n", .{var_amount});
    std.debug.print("  Return Threshold (5th percentile): {d:.4}%\n", .{var_return * 100.0});

    // Calculate Conditional VaR (Expected Shortfall)
    var cvar_sum: f64 = 0.0;
    for (daily_returns[0..var_index]) |ret| {
        cvar_sum += ret;
    }
    const cvar_return = cvar_sum / @as(f64, @floatFromInt(var_index));
    const cvar_amount = -cvar_return * portfolio_value;

    std.debug.print("\nConditional VaR (CVaR / Expected Shortfall):\n", .{});
    std.debug.print("  Expected Loss (worst 5%%): ${d:.2}\n", .{cvar_amount});
    std.debug.print("  Average Return (worst 5%%): {d:.4}%\n\n", .{cvar_return * 100.0});

    // Part 4: Risk-Adjusted Performance Metrics
    std.debug.print("Part 4: Risk-Adjusted Performance\n", .{});
    std.debug.print("-----------------------------------\n", .{});

    // Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Std Dev
    const risk_free_daily = r / 252.0; // Daily risk-free rate
    const sharpe_ratio = (portfolio_mean - risk_free_daily) / portfolio_std;
    const sharpe_annualized = sharpe_ratio * @sqrt(252.0); // Annualized Sharpe

    std.debug.print("Sharpe Ratio (Daily): {d:.4}\n", .{sharpe_ratio});
    std.debug.print("Sharpe Ratio (Annualized): {d:.4}\n", .{sharpe_annualized});

    // Information Ratio (assuming benchmark return = risk-free rate)
    const benchmark_return = risk_free_daily;
    const active_return = portfolio_mean - benchmark_return;
    const tracking_error = portfolio_std; // Simplified (should be std of active returns)
    const information_ratio = active_return / tracking_error;

    std.debug.print("Information Ratio: {d:.4}\n", .{information_ratio});

    // Maximum Drawdown (from simulated returns)
    var cumulative: f64 = 1.0;
    var peak: f64 = 1.0;
    var max_drawdown: f64 = 0.0;

    for (daily_returns) |ret| {
        cumulative *= (1.0 + ret);

        if (cumulative > peak) {
            peak = cumulative;
        }

        const drawdown = (peak - cumulative) / peak;
        if (drawdown > max_drawdown) {
            max_drawdown = drawdown;
        }
    }

    std.debug.print("Maximum Drawdown: {d:.2}%\n", .{max_drawdown * 100.0});

    // Calmar Ratio = Annualized Return / Maximum Drawdown
    const calmar_ratio = annual_return / max_drawdown;
    std.debug.print("Calmar Ratio: {d:.4}\n\n", .{calmar_ratio});

    // Part 5: Downside Risk Metrics
    std.debug.print("Part 5: Downside Risk Metrics\n", .{});
    std.debug.print("-------------------------------\n", .{});

    // Sortino Ratio (downside deviation)
    var downside_sum: f64 = 0.0;
    var downside_count: usize = 0;
    for (daily_returns) |ret| {
        if (ret < risk_free_daily) {
            const deviation = ret - risk_free_daily;
            downside_sum += deviation * deviation;
            downside_count += 1;
        }
    }
    const downside_std = @sqrt(downside_sum / @as(f64, @floatFromInt(downside_count)));
    const sortino_ratio = (portfolio_mean - risk_free_daily) / downside_std;
    const sortino_annualized = sortino_ratio * @sqrt(252.0);

    std.debug.print("Sortino Ratio (Daily): {d:.4}\n", .{sortino_ratio});
    std.debug.print("Sortino Ratio (Annualized): {d:.4}\n", .{sortino_annualized});
    std.debug.print("Downside Volatility: {d:.4}%\n\n", .{downside_std * 100.0});

    std.debug.print("=== SUMMARY ===\n", .{});
    std.debug.print("This example demonstrates:\n", .{});
    std.debug.print("1. Monte Carlo simulation for derivatives pricing (Black-Scholes)\n", .{});
    std.debug.print("2. Risk measurement via Value at Risk (VaR) and CVaR\n", .{});
    std.debug.print("3. Performance metrics: Sharpe ratio, Information ratio, Sortino ratio\n", .{});
    std.debug.print("4. Downside risk analysis and maximum drawdown calculation\n", .{});
    std.debug.print("\nKey zuda modules used:\n", .{});
    std.debug.print("- stats.distributions.Normal: Random sampling for simulations\n", .{});
    std.debug.print("- stats.descriptive: Mean, std dev, percentiles\n", .{});
    std.debug.print("- ndarray.NDArray: Array operations for statistical analysis\n", .{});
}
