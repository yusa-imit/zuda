const std = @import("std");
const zuda = @import("zuda");

const NDArray = zuda.ndarray.NDArray;
const Normal = zuda.stats.distributions.Normal;
const Uniform = zuda.stats.distributions.Uniform;
const descriptive = zuda.stats.descriptive;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Bayesian Inference & MCMC Sampling Demo ===\n\n", .{});

    // Part 1: Problem setup - Estimating normal distribution parameters
    // True parameters: μ = 10.0, σ = 2.0
    // We'll generate observed data and use MCMC to recover the parameters
    std.debug.print("Part 1: Data Generation\n", .{});
    std.debug.print("True parameters: μ = 10.0, σ = 2.0\n", .{});

    const n_obs: usize = 50;
    const true_mu: f64 = 10.0;
    const true_sigma: f64 = 2.0;

    // Generate synthetic observations
    var observations = try std.ArrayList(f64).initCapacity(allocator, n_obs);
    defer observations.deinit(allocator);

    var rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const random = rng.random();

    const data_dist = Normal(f64){ .mean = true_mu, .std = true_sigma };
    for (0..n_obs) |_| {
        const sample = data_dist.sample(random);
        observations.appendAssumeCapacity(sample);
    }

    var obs_array = try NDArray(f64, 1).fromSlice(allocator, &.{n_obs}, observations.items, .row_major);
    defer obs_array.deinit();

    const obs_mean = descriptive.mean(f64, obs_array);
    const obs_std = try descriptive.stdDev(f64, obs_array, 1);
    std.debug.print("Observed data: n={}, mean={d:.3}, std={d:.3}\n\n", .{ n_obs, obs_mean, obs_std });

    // Part 2: Bayesian Model Setup
    // Prior: μ ~ Normal(0, 10), σ ~ Uniform(0.1, 10)
    // Likelihood: X_i ~ Normal(μ, σ)
    // Posterior: p(μ, σ | X) ∝ p(X | μ, σ) × p(μ) × p(σ)
    std.debug.print("Part 2: Bayesian Model\n", .{});
    std.debug.print("Prior: μ ~ Normal(0, 10), σ ~ Uniform(0.1, 10)\n", .{});
    std.debug.print("Likelihood: X_i ~ Normal(μ, σ)\n\n", .{});

    // Part 3: Metropolis-Hastings MCMC Sampling
    std.debug.print("Part 3: MCMC Sampling (Metropolis-Hastings)\n", .{});

    const n_iterations: usize = 10000;
    const burn_in: usize = 2000;
    const proposal_std_mu: f64 = 0.5;
    const proposal_std_sigma: f64 = 0.3;

    const expected_samples = n_iterations - burn_in;
    var samples_mu = try std.ArrayList(f64).initCapacity(allocator, expected_samples);
    defer samples_mu.deinit(allocator);
    var samples_sigma = try std.ArrayList(f64).initCapacity(allocator, expected_samples);
    defer samples_sigma.deinit(allocator);

    // Initialize chain at prior mean
    var current_mu: f64 = 0.0;
    var current_sigma: f64 = 5.0;
    var current_log_posterior = logPosterior(current_mu, current_sigma, observations.items);

    var accepted: usize = 0;

    for (0..n_iterations) |iter| {
        // Propose new parameters
        const proposal_mu_dist = Normal(f64){ .mean = current_mu, .std = proposal_std_mu };
        const proposal_sigma_dist = Normal(f64){ .mean = current_sigma, .std = proposal_std_sigma };

        const proposed_mu = proposal_mu_dist.sample(random);
        var proposed_sigma = proposal_sigma_dist.sample(random);

        // Ensure sigma > 0
        if (proposed_sigma <= 0.1) proposed_sigma = 0.1;

        // Compute acceptance probability
        const proposed_log_posterior = logPosterior(proposed_mu, proposed_sigma, observations.items);
        const log_alpha = proposed_log_posterior - current_log_posterior;

        // Accept or reject
        const u = random.float(f64);
        if (@log(u) < log_alpha) {
            current_mu = proposed_mu;
            current_sigma = proposed_sigma;
            current_log_posterior = proposed_log_posterior;
            accepted += 1;
        }

        // Store samples after burn-in
        if (iter >= burn_in) {
            samples_mu.appendAssumeCapacity(current_mu);
            samples_sigma.appendAssumeCapacity(current_sigma);
        }
    }

    const acceptance_rate = @as(f64, @floatFromInt(accepted)) / @as(f64, @floatFromInt(n_iterations));
    std.debug.print("Iterations: {}, Burn-in: {}, Acceptance rate: {d:.1}%\n", .{ n_iterations, burn_in, acceptance_rate * 100.0 });
    std.debug.print("Posterior samples: {}\n\n", .{samples_mu.items.len});

    // Part 4: Posterior Analysis
    std.debug.print("Part 4: Posterior Analysis\n", .{});

    var mu_array = try NDArray(f64, 1).fromSlice(allocator, &.{samples_mu.items.len}, samples_mu.items, .row_major);
    defer mu_array.deinit();
    var sigma_array = try NDArray(f64, 1).fromSlice(allocator, &.{samples_sigma.items.len}, samples_sigma.items, .row_major);
    defer sigma_array.deinit();

    const posterior_mu_mean = descriptive.mean(f64, mu_array);
    const posterior_mu_std = try descriptive.stdDev(f64, mu_array, 1);
    const posterior_sigma_mean = descriptive.mean(f64, sigma_array);
    const posterior_sigma_std = try descriptive.stdDev(f64, sigma_array, 1);

    std.debug.print("Posterior μ: mean={d:.3}, std={d:.3} (true={d:.3})\n", .{ posterior_mu_mean, posterior_mu_std, true_mu });
    std.debug.print("Posterior σ: mean={d:.3}, std={d:.3} (true={d:.3})\n\n", .{ posterior_sigma_mean, posterior_sigma_std, true_sigma });

    // Compute 95% credible intervals
    const sorted_mu = try allocator.dupe(f64, samples_mu.items);
    defer allocator.free(sorted_mu);
    std.mem.sort(f64, sorted_mu, {}, comptime std.sort.asc(f64));

    const sorted_sigma = try allocator.dupe(f64, samples_sigma.items);
    defer allocator.free(sorted_sigma);
    std.mem.sort(f64, sorted_sigma, {}, comptime std.sort.asc(f64));

    const n_samples = sorted_mu.len;
    const lower_idx = @as(usize, @intFromFloat(@as(f64, @floatFromInt(n_samples)) * 0.025));
    const upper_idx = @as(usize, @intFromFloat(@as(f64, @floatFromInt(n_samples)) * 0.975));

    const mu_ci_lower = sorted_mu[lower_idx];
    const mu_ci_upper = sorted_mu[upper_idx];
    const sigma_ci_lower = sorted_sigma[lower_idx];
    const sigma_ci_upper = sorted_sigma[upper_idx];

    std.debug.print("95% Credible Intervals:\n", .{});
    std.debug.print("μ: [{d:.3}, {d:.3}] (true={d:.3} included: {})\n", .{
        mu_ci_lower,
        mu_ci_upper,
        true_mu,
        mu_ci_lower <= true_mu and true_mu <= mu_ci_upper,
    });
    std.debug.print("σ: [{d:.3}, {d:.3}] (true={d:.3} included: {})\n\n", .{
        sigma_ci_lower,
        sigma_ci_upper,
        true_sigma,
        sigma_ci_lower <= true_sigma and true_sigma <= sigma_ci_upper,
    });

    // Part 5: Convergence Diagnostics
    std.debug.print("Part 5: Convergence Diagnostics\n", .{});

    // Compute effective sample size (simple version: n_samples / (1 + 2*sum(autocorr)))
    // For simplicity, we'll compute lag-1 autocorrelation
    const acf_mu = computeAutocorrelation(samples_mu.items, 1);
    const acf_sigma = computeAutocorrelation(samples_sigma.items, 1);

    std.debug.print("Lag-1 Autocorrelation: μ={d:.3}, σ={d:.3}\n", .{ acf_mu, acf_sigma });
    std.debug.print("(Lower autocorrelation indicates better mixing)\n\n", .{});

    // Part 6: Trace Plot Summary (first 100 samples)
    std.debug.print("Part 6: Trace Plot Summary (first 100 post-burn-in samples)\n", .{});
    std.debug.print("Sample   μ         σ\n", .{});
    std.debug.print("------ -------- --------\n", .{});
    for (0..@min(100, samples_mu.items.len)) |i| {
        if (i % 10 == 0) {
            std.debug.print("{:6}  {d:8.3}  {d:8.3}\n", .{ i, samples_mu.items[i], samples_sigma.items[i] });
        }
    }

    std.debug.print("\n=== Demo Complete ===\n", .{});
}

/// Compute log posterior: log p(μ, σ | X) = log p(X | μ, σ) + log p(μ) + log p(σ)
fn logPosterior(mu: f64, sigma: f64, observations: []const f64) f64 {
    // Prior for μ: Normal(0, 10)
    const prior_mu = Normal(f64){ .mean = 0.0, .std = 10.0 };
    const log_prior_mu = prior_mu.logpdf(mu);

    // Prior for σ: Uniform(0.1, 10)
    const prior_sigma = Uniform(f64){ .a = 0.1, .b = 10.0 };
    const log_prior_sigma = prior_sigma.logpdf(sigma);

    // Likelihood: product of Normal(μ, σ) for each observation
    var log_likelihood: f64 = 0.0;
    const likelihood_dist = Normal(f64){ .mean = mu, .std = sigma };
    for (observations) |x| {
        log_likelihood += likelihood_dist.logpdf(x);
    }

    return log_likelihood + log_prior_mu + log_prior_sigma;
}

/// Compute autocorrelation at given lag
fn computeAutocorrelation(data: []const f64, lag: usize) f64 {
    const n = data.len;
    if (n <= lag) return 0.0;

    // Compute mean manually to avoid NDArray overhead in helper function
    var sum: f64 = 0.0;
    for (data) |x| {
        sum += x;
    }
    const data_mean = sum / @as(f64, @floatFromInt(n));

    var numerator: f64 = 0.0;
    var denominator: f64 = 0.0;

    for (0..n - lag) |i| {
        numerator += (data[i] - data_mean) * (data[i + lag] - data_mean);
    }

    for (data) |x| {
        denominator += (x - data_mean) * (x - data_mean);
    }

    if (denominator == 0.0) return 0.0;
    return numerator / denominator;
}
