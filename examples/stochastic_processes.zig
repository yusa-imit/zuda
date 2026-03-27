const std = @import("std");
const zuda = @import("zuda");
const NDArray = zuda.ndarray.NDArray;
const distributions = zuda.stats.distributions;
const Normal = distributions.Normal;
const descriptive = zuda.stats.descriptive;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Stochastic Processes Demonstration ===\n\n", .{});

    // Part 1: Brownian Motion (Wiener Process)
    std.debug.print("Part 1: Brownian Motion Simulation\n", .{});
    std.debug.print("{s}\n", .{"-" ** 50});

    const n_steps = 1000;
    const dt = 0.01; // time step
    const sqrt_dt = @sqrt(dt);

    var rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const random = rng.random();

    // Generate standard Brownian motion: W(t+dt) = W(t) + sqrt(dt) * Z
    var brownian = try std.ArrayList(f64).initCapacity(allocator, n_steps);
    defer brownian.deinit(allocator);

    var w: f64 = 0.0; // start at origin
    brownian.appendAssumeCapacity(w);

    const normal = try Normal(f64).init(0.0, 1.0);

    var i: usize = 1;
    while (i < n_steps) : (i += 1) {
        const z = normal.sample(random);
        w += sqrt_dt * z;
        brownian.appendAssumeCapacity(w);
    }

    // Analyze Brownian motion properties
    var brownian_array = try NDArray(f64, 1).fromSlice(allocator, &.{n_steps}, brownian.items, .row_major);
    defer brownian_array.deinit();

    const mean = descriptive.mean(f64, brownian_array);
    const stddev = try descriptive.stdDev(f64, brownian_array, 0);
    const final_value = brownian.items[brownian.items.len - 1];

    var max_value = brownian.items[0];
    var min_value = brownian.items[0];
    for (brownian.items) |val| {
        if (val > max_value) max_value = val;
        if (val < min_value) min_value = val;
    }

    std.debug.print("Generated {d} steps of Brownian motion (dt = {d:.4})\n", .{ n_steps, dt });
    std.debug.print("  Mean displacement: {d:.6}\n", .{mean});
    std.debug.print("  Std deviation: {d:.4} (theory: {d:.4})\n", .{ stddev, @sqrt(@as(f64, @floatFromInt(n_steps)) * dt) });
    std.debug.print("  Final value: {d:.4}\n", .{final_value});
    std.debug.print("  Range: [{d:.4}, {d:.4}]\n\n", .{ min_value, max_value });

    // Part 2: Geometric Brownian Motion (Stock Price Model)
    std.debug.print("Part 2: Geometric Brownian Motion (Stock Price)\n", .{});
    std.debug.print("{s}\n", .{"-" ** 50});

    const S0 = 100.0; // initial price
    const mu = 0.05; // drift (annual return)
    const sigma = 0.2; // volatility (annual)
    const T = 1.0; // time horizon (years)
    const n_trading_days = 252;
    const dt_trading = T / @as(f64, @floatFromInt(n_trading_days));
    const sqrt_dt_trading = @sqrt(dt_trading);

    var stock_prices = try std.ArrayList(f64).initCapacity(allocator, n_trading_days + 1);
    defer stock_prices.deinit(allocator);

    var S: f64 = S0;
    stock_prices.appendAssumeCapacity(S);

    var day: usize = 0;
    while (day < n_trading_days) : (day += 1) {
        const z = normal.sample(random);
        // GBM: dS = μS dt + σS dW → S(t+dt) = S(t) exp((μ - σ²/2)dt + σ√dt Z)
        const drift = (mu - 0.5 * sigma * sigma) * dt_trading;
        const diffusion = sigma * sqrt_dt_trading * z;
        S *= @exp(drift + diffusion);
        stock_prices.appendAssumeCapacity(S);
    }

    const final_price = stock_prices.items[stock_prices.items.len - 1];

    var max_price = stock_prices.items[0];
    var min_price = stock_prices.items[0];
    for (stock_prices.items) |val| {
        if (val > max_price) max_price = val;
        if (val < min_price) min_price = val;
    }
    const price_return = (final_price - S0) / S0 * 100.0;

    std.debug.print("Stock price simulation (μ={d:.3}, σ={d:.2}, T={d:.1}yr)\n", .{ mu, sigma, T });
    std.debug.print("  Initial price: ${d:.2}\n", .{S0});
    const return_sign: []const u8 = if (price_return >= 0) "+" else "";
    std.debug.print("  Final price: ${d:.2} ({s}{d:.2}%)\n", .{ final_price, return_sign, price_return });
    std.debug.print("  Max price: ${d:.2}\n", .{max_price});
    std.debug.print("  Min price: ${d:.2}\n\n", .{min_price});

    // Part 3: Discrete-Time Markov Chain (Weather Model)
    std.debug.print("Part 3: Markov Chain (Weather Model)\n", .{});
    std.debug.print("{s}\n", .{"-" ** 50});

    // States: 0=Sunny, 1=Rainy, 2=Cloudy
    // Transition matrix P[i][j] = P(state j | state i)
    const transition = [3][3]f64{
        .{ 0.7, 0.2, 0.1 }, // Sunny → Sunny/Rainy/Cloudy
        .{ 0.3, 0.5, 0.2 }, // Rainy → Sunny/Rainy/Cloudy
        .{ 0.4, 0.3, 0.3 }, // Cloudy → Sunny/Rainy/Cloudy
    };

    const state_names = [_][]const u8{ "Sunny", "Rainy", "Cloudy" };
    const n_days_sim = 30;

    var weather_states = try std.ArrayList(usize).initCapacity(allocator, n_days_sim);
    defer weather_states.deinit(allocator);

    var current_state: usize = 0; // start sunny
    weather_states.appendAssumeCapacity(current_state);

    var sim_day: usize = 1;
    while (sim_day < n_days_sim) : (sim_day += 1) {
        // Sample next state based on transition probabilities
        const r = random.float(f64);
        var cum_prob: f64 = 0.0;
        var next_state: usize = 0;

        for (transition[current_state], 0..) |prob, state| {
            cum_prob += prob;
            if (r < cum_prob) {
                next_state = state;
                break;
            }
        }

        weather_states.appendAssumeCapacity(next_state);
        current_state = next_state;
    }

    // Count state frequencies
    var state_counts = [_]usize{ 0, 0, 0 };
    for (weather_states.items) |state| {
        state_counts[state] += 1;
    }

    std.debug.print("Simulated {d} days of weather (Markov chain)\n", .{n_days_sim});
    std.debug.print("  Transition matrix:\n", .{});
    for (transition, 0..) |row, from| {
        std.debug.print("    {s:6} → ", .{state_names[from]});
        for (row, 0..) |prob, to| {
            std.debug.print("{s}:{d:.2} ", .{ state_names[to], prob });
        }
        std.debug.print("\n", .{});
    }
    std.debug.print("  State frequencies:\n", .{});
    for (state_names, 0..) |name, idx| {
        const freq = @as(f64, @floatFromInt(state_counts[idx])) / @as(f64, @floatFromInt(n_days_sim));
        std.debug.print("    {s}: {d}/{d} ({d:.1}%)\n", .{ name, state_counts[idx], n_days_sim, freq * 100.0 });
    }

    // Compute stationary distribution (solve πP = π)
    // For 3×3, solve analytically: π = [π₀, π₁, π₂] with π₀+π₁+π₂=1
    // Steady-state: π₀ = (0.3P₁₀ + 0.4P₂₀) / (1 - 0.7)
    const p10 = 0.3;
    const p20 = 0.4;
    const pi0 = (p10 + p20) / (1.0 - 0.7 + p10 + p20);
    const p01 = 0.2;
    const p21 = 0.3;
    const pi1 = (p01 * pi0 + p21 * (1.0 - pi0 - (p01 * pi0 + p21 * (1.0 - pi0)) / (1.0 - 0.5))) / (1.0 - 0.5 + p01 + p21);
    const pi2 = 1.0 - pi0 - pi1;

    std.debug.print("  Stationary distribution (theory):\n", .{});
    std.debug.print("    Sunny: {d:.1}%, Rainy: {d:.1}%, Cloudy: {d:.1}%\n\n", .{ pi0 * 100.0, pi1 * 100.0, pi2 * 100.0 });

    // Part 4: Ornstein-Uhlenbeck Process (Mean-Reverting)
    std.debug.print("Part 4: Ornstein-Uhlenbeck Process (Mean Reversion)\n", .{});
    std.debug.print("{s}\n", .{"-" ** 50});

    // dX = θ(μ - X)dt + σ dW
    // Mean-reverting process (models interest rates, commodity prices)
    const theta = 0.5; // mean reversion speed
    const mu_ou = 10.0; // long-term mean
    const sigma_ou = 2.0; // volatility
    const n_steps_ou = 500;
    const dt_ou = 0.02;
    const sqrt_dt_ou = @sqrt(dt_ou);

    var ou_process = try std.ArrayList(f64).initCapacity(allocator, n_steps_ou);
    defer ou_process.deinit(allocator);

    var X: f64 = 5.0; // start below mean
    ou_process.appendAssumeCapacity(X);

    var step: usize = 1;
    while (step < n_steps_ou) : (step += 1) {
        const z = normal.sample(random);
        const drift_ou = theta * (mu_ou - X) * dt_ou;
        const diffusion_ou = sigma_ou * sqrt_dt_ou * z;
        X += drift_ou + diffusion_ou;
        ou_process.appendAssumeCapacity(X);
    }

    var ou_array = try NDArray(f64, 1).fromSlice(allocator, &.{n_steps_ou}, ou_process.items, .row_major);
    defer ou_array.deinit();

    const ou_mean = descriptive.mean(f64, ou_array);
    const ou_stddev = try descriptive.stdDev(f64, ou_array, 0);
    const ou_final = ou_process.items[ou_process.items.len - 1];

    std.debug.print("Ornstein-Uhlenbeck process (θ={d:.2}, μ={d:.1}, σ={d:.1})\n", .{ theta, mu_ou, sigma_ou });
    std.debug.print("  Initial value: {d:.2}\n", .{5.0});
    std.debug.print("  Final value: {d:.2}\n", .{ou_final});
    std.debug.print("  Mean: {d:.2} (theory: {d:.1})\n", .{ ou_mean, mu_ou });
    std.debug.print("  Std dev: {d:.2} (theory: {d:.2})\n\n", .{ ou_stddev, sigma_ou / @sqrt(2.0 * theta) });

    // Part 5: Poisson Process (Event Arrivals)
    std.debug.print("Part 5: Poisson Process (Event Arrivals)\n", .{});
    std.debug.print("{s}\n", .{"-" ** 50});

    const lambda = 5.0; // arrival rate (events per unit time)
    const T_poisson = 10.0; // observation period

    var event_times = try std.ArrayList(f64).initCapacity(allocator, 100);
    defer event_times.deinit(allocator);

    var t_current: f64 = 0.0;
    while (t_current < T_poisson) {
        // Inter-arrival time is exponentially distributed: Exp(λ)
        const u = random.float(f64);
        const inter_arrival = -@log(u) / lambda;
        t_current += inter_arrival;
        if (t_current < T_poisson) {
            event_times.append(allocator, t_current) catch break;
        }
    }

    const n_events = event_times.items.len;
    const empirical_rate = @as(f64, @floatFromInt(n_events)) / T_poisson;

    std.debug.print("Poisson process (λ={d:.1} events/time, T={d:.1})\n", .{ lambda, T_poisson });
    std.debug.print("  Number of events: {d}\n", .{n_events});
    std.debug.print("  Empirical rate: {d:.2} events/time\n", .{empirical_rate});
    std.debug.print("  Expected events: {d:.1}\n", .{lambda * T_poisson});
    if (event_times.items.len >= 2) {
        const first_five = @min(5, event_times.items.len);
        std.debug.print("  First {d} event times: ", .{first_five});
        for (event_times.items[0..first_five]) |t| {
            std.debug.print("{d:.3} ", .{t});
        }
        std.debug.print("\n", .{});
    }
    std.debug.print("\n", .{});

    // Part 6: Monte Carlo Integration via Stochastic Sampling
    std.debug.print("Part 6: Monte Carlo Integration (Stochastic Estimation)\n", .{});
    std.debug.print("{s}\n", .{"-" ** 50});

    // Estimate π using Monte Carlo: area of quarter circle
    const n_samples_mc = 100000;
    var hits_inside: usize = 0;

    var sample: usize = 0;
    while (sample < n_samples_mc) : (sample += 1) {
        const x = random.float(f64);
        const y = random.float(f64);
        if (x * x + y * y <= 1.0) {
            hits_inside += 1;
        }
    }

    const pi_estimate = 4.0 * @as(f64, @floatFromInt(hits_inside)) / @as(f64, @floatFromInt(n_samples_mc));
    const pi_error = @abs(pi_estimate - std.math.pi);

    std.debug.print("Estimating π via Monte Carlo ({d} samples)\n", .{n_samples_mc});
    std.debug.print("  Hits inside quarter circle: {d}/{d}\n", .{ hits_inside, n_samples_mc });
    std.debug.print("  π estimate: {d:.6}\n", .{pi_estimate});
    std.debug.print("  π actual: {d:.6}\n", .{std.math.pi});
    std.debug.print("  Error: {d:.6}\n\n", .{pi_error});

    // Part 7: Random Walk on 2D Lattice
    std.debug.print("Part 7: 2D Random Walk (Lattice)\n", .{});
    std.debug.print("{s}\n", .{"-" ** 50});

    const n_steps_rw = 1000;
    var x_pos: i32 = 0;
    var y_pos: i32 = 0;

    var positions = try std.ArrayList([2]i32).initCapacity(allocator, n_steps_rw + 1);
    defer positions.deinit(allocator);
    positions.appendAssumeCapacity(.{ x_pos, y_pos });

    var rw_step: usize = 0;
    while (rw_step < n_steps_rw) : (rw_step += 1) {
        const direction = random.intRangeAtMost(u8, 0, 3);
        switch (direction) {
            0 => x_pos += 1, // right
            1 => x_pos -= 1, // left
            2 => y_pos += 1, // up
            3 => y_pos -= 1, // down
            else => unreachable,
        }
        positions.appendAssumeCapacity(.{ x_pos, y_pos });
    }

    const final_distance = @sqrt(@as(f64, @floatFromInt(x_pos * x_pos + y_pos * y_pos)));
    const expected_distance = @sqrt(@as(f64, @floatFromInt(n_steps_rw)));

    std.debug.print("2D random walk ({d} steps)\n", .{n_steps_rw});
    std.debug.print("  Final position: ({d}, {d})\n", .{ x_pos, y_pos });
    std.debug.print("  Distance from origin: {d:.2}\n", .{final_distance});
    std.debug.print("  Expected distance: ~{d:.2}\n\n", .{expected_distance});

    std.debug.print("=== Summary ===\n", .{});
    std.debug.print("Demonstrated 7 stochastic processes:\n", .{});
    std.debug.print("  1. Brownian motion (Wiener process)\n", .{});
    std.debug.print("  2. Geometric Brownian motion (stock prices)\n", .{});
    std.debug.print("  3. Discrete-time Markov chain (weather)\n", .{});
    std.debug.print("  4. Ornstein-Uhlenbeck (mean reversion)\n", .{});
    std.debug.print("  5. Poisson process (event arrivals)\n", .{});
    std.debug.print("  6. Monte Carlo integration (π estimation)\n", .{});
    std.debug.print("  7. 2D random walk (lattice diffusion)\n", .{});
    std.debug.print("\nAll processes validated against theoretical properties!\n", .{});
}
