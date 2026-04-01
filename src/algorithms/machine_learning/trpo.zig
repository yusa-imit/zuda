const std = @import("std");
const Allocator = std.mem.Allocator;

/// Trust Region Policy Optimization (TRPO)
///
/// Policy gradient method with KL divergence constraint for stable updates.
///
/// Algorithm Overview:
/// - Policy: Stochastic policy π(a|s) parameterized by θ
/// - Advantage: Generalized Advantage Estimation (GAE) for variance reduction
/// - Update: Maximize surrogate objective subject to KL(π_old || π_new) ≤ δ
/// - Optimization: Conjugate gradient + line search for natural gradient
/// - Trust region: Hard constraint on policy change (δ = 0.01 typical)
///
/// Key Features:
/// - Monotonic improvement guarantee (theory)
/// - Natural policy gradient via Fisher information matrix
/// - Backtracking line search for constraint satisfaction
/// - GAE for bias-variance tradeoff
///
/// Time complexity: O(n × m × max_iter) per update, where n = states, m = actions
/// Space complexity: O(n × m) for policy and value function
///
/// Use cases:
/// - Continuous control: robotics, locomotion, manipulation
/// - Stable training: guaranteed monotonic improvement
/// - Research baseline: foundation for PPO
/// - Safety-critical: hard constraint on policy change
///
/// Trade-offs:
/// - vs PPO: More stable (hard constraint), but slower (CG iterations)
/// - vs A2C: Sample efficient (multi-epoch), but complex optimization
/// - vs REINFORCE: Much lower variance (critic + GAE), better convergence
///
pub fn TRPO(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        num_states: usize,
        num_actions: usize,

        // Policy network: log probabilities for each (state, action)
        log_policy: []T, // [num_states * num_actions]

        // Value function: V(s)
        value_fn: []T, // [num_states]

        // Hyperparameters
        gamma: T, // Discount factor
        lambda: T, // GAE lambda (bias-variance tradeoff)
        delta: T, // KL divergence constraint
        damping: T, // Damping for conjugate gradient
        max_kl: T, // Maximum KL before backtrack
        alpha: T, // Value function learning rate
        cg_iters: usize, // Conjugate gradient iterations
        max_backtracks: usize, // Line search backtracks

        /// Configuration for TRPO
        pub const Config = struct {
            gamma: T = 0.99, // Discount factor
            lambda: T = 0.95, // GAE lambda
            delta: T = 0.01, // KL constraint
            damping: T = 0.1, // CG damping
            max_kl: T = 0.01, // Max KL for line search
            alpha: T = 0.01, // Value learning rate
            cg_iters: usize = 10, // CG iterations
            max_backtracks: usize = 10, // Line search steps
        };

        /// Initialize TRPO agent
        ///
        /// Time: O(n × m)
        /// Space: O(n × m)
        pub fn init(allocator: Allocator, num_states: usize, num_actions: usize, config: Config) !Self {
            if (num_states == 0) return error.InvalidStateSpace;
            if (num_actions == 0) return error.InvalidActionSpace;

            const policy_size = num_states * num_actions;
            const log_policy = try allocator.alloc(T, policy_size);
            errdefer allocator.free(log_policy);

            const value_fn = try allocator.alloc(T, num_states);
            errdefer allocator.free(value_fn);

            // Initialize uniform policy: log(1/m) for each action
            const log_uniform = @log(@as(T, 1.0) / @as(T, @floatFromInt(num_actions)));
            @memset(log_policy, log_uniform);

            // Initialize value function to zero
            @memset(value_fn, 0);

            return Self{
                .allocator = allocator,
                .num_states = num_states,
                .num_actions = num_actions,
                .log_policy = log_policy,
                .value_fn = value_fn,
                .gamma = config.gamma,
                .lambda = config.lambda,
                .delta = config.delta,
                .damping = config.damping,
                .max_kl = config.max_kl,
                .alpha = config.alpha,
                .cg_iters = config.cg_iters,
                .max_backtracks = config.max_backtracks,
            };
        }

        /// Free resources
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.log_policy);
            self.allocator.free(self.value_fn);
        }

        /// Select action using current policy (stochastic)
        ///
        /// Time: O(m)
        /// Space: O(1)
        pub fn selectAction(self: *const Self, state: usize, rng: std.Random) !usize {
            if (state >= self.num_states) return error.InvalidState;

            const offset = state * self.num_actions;
            const log_probs = self.log_policy[offset..offset + self.num_actions];

            // Convert log probabilities to probabilities
            var max_log: T = log_probs[0];
            for (log_probs[1..]) |lp| {
                if (lp > max_log) max_log = lp;
            }

            // Numerically stable softmax
            var sum: T = 0;
            var probs: [32]T = undefined; // Stack allocation for common case
            var heap_probs: ?[]T = null;
            const probs_slice = if (self.num_actions <= 32)
                probs[0..self.num_actions]
            else blk: {
                heap_probs = try self.allocator.alloc(T, self.num_actions);
                break :blk heap_probs.?;
            };
            defer if (heap_probs) |hp| self.allocator.free(hp);

            for (log_probs, 0..) |lp, i| {
                const prob = @exp(lp - max_log);
                probs_slice[i] = prob;
                sum += prob;
            }

            // Sample from categorical distribution
            const r = rng.float(T);
            var cumsum: T = 0;
            for (probs_slice, 0..) |p, i| {
                cumsum += p / sum;
                if (r < cumsum) return i;
            }

            return self.num_actions - 1;
        }

        /// Select greedy action (argmax policy)
        ///
        /// Time: O(m)
        /// Space: O(1)
        pub fn selectGreedyAction(self: *const Self, state: usize) !usize {
            if (state >= self.num_states) return error.InvalidState;

            const offset = state * self.num_actions;
            const log_probs = self.log_policy[offset..offset + self.num_actions];

            var best_action: usize = 0;
            var best_log_prob = log_probs[0];

            for (log_probs[1..], 1..) |lp, a| {
                if (lp > best_log_prob) {
                    best_log_prob = lp;
                    best_action = a;
                }
            }

            return best_action;
        }

        /// Experience tuple for trajectory collection
        pub const Experience = struct {
            state: usize,
            action: usize,
            reward: T,
            next_state: usize,
            done: bool,
        };

        /// Update policy using trajectory with TRPO
        ///
        /// Steps:
        /// 1. Compute GAE advantages
        /// 2. Compute policy gradient
        /// 3. Compute natural gradient via conjugate gradient
        /// 4. Line search with KL constraint
        /// 5. Update value function
        ///
        /// Time: O(K × m × cg_iters) where K = trajectory length
        /// Space: O(K)
        pub fn update(self: *Self, experiences: []const Experience) !void {
            if (experiences.len == 0) return;

            // 1. Compute advantages using GAE
            const advantages = try self.allocator.alloc(T, experiences.len);
            defer self.allocator.free(advantages);

            const returns = try self.allocator.alloc(T, experiences.len);
            defer self.allocator.free(returns);

            try self.computeGAE(experiences, advantages, returns);

            // 2. Normalize advantages (mean=0, std=1)
            const adv_mean = self.mean(advantages);
            const adv_std = self.stdDev(advantages, adv_mean);

            if (adv_std > 1e-8) {
                for (advantages) |*adv| {
                    adv.* = (adv.* - adv_mean) / adv_std;
                }
            }

            // 3. Compute policy gradient
            const grad = try self.allocator.alloc(T, self.log_policy.len);
            defer self.allocator.free(grad);
            @memset(grad, 0);

            for (experiences, 0..) |exp, i| {
                const idx = exp.state * self.num_actions + exp.action;
                grad[idx] += advantages[i];
            }

            // 4. Compute natural gradient using conjugate gradient
            const nat_grad = try self.allocator.alloc(T, self.log_policy.len);
            defer self.allocator.free(nat_grad);

            try self.conjugateGradient(experiences, grad, nat_grad);

            // 5. Line search with KL constraint
            const old_policy = try self.allocator.alloc(T, self.log_policy.len);
            defer self.allocator.free(old_policy);
            @memcpy(old_policy, self.log_policy);

            var step_size: T = 1.0;
            var backtrack: usize = 0;

            while (backtrack < self.max_backtracks) : (backtrack += 1) {
                // Try update: θ_new = θ_old + step_size × nat_grad
                for (self.log_policy, 0..) |*lp, i| {
                    lp.* = old_policy[i] + step_size * nat_grad[i];
                }

                // Normalize to valid log probabilities
                self.normalizePolicy();

                // Check KL constraint
                const kl = try self.computeKL(experiences, old_policy);

                if (kl <= self.max_kl) {
                    // Accept update
                    break;
                }

                // Backtrack
                step_size *= 0.5;
            } else {
                // Failed to satisfy constraint, revert
                @memcpy(self.log_policy, old_policy);
            }

            // 6. Update value function using TD learning
            for (experiences, 0..) |exp, i| {
                const td_error = returns[i] - self.value_fn[exp.state];
                self.value_fn[exp.state] += self.alpha * td_error;
            }
        }

        /// Compute Generalized Advantage Estimation (GAE)
        ///
        /// GAE(λ): Â_t = Σ(γλ)^l δ_{t+l}
        /// where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        ///
        /// Time: O(K)
        /// Space: O(1)
        fn computeGAE(self: *const Self, experiences: []const Experience, advantages: []T, returns: []T) !void {
            var gae: T = 0;

            // Compute backwards for efficiency
            var i: usize = experiences.len;
            while (i > 0) {
                i -= 1;

                const exp = experiences[i];
                const value = self.value_fn[exp.state];
                const next_value = if (exp.done) 0 else self.value_fn[exp.next_state];

                // TD error: δ = r + γV(s') - V(s)
                const td_error = exp.reward + self.gamma * next_value - value;

                // GAE: Â = δ + γλÂ_{t+1}
                gae = td_error + self.gamma * self.lambda * gae;
                advantages[i] = gae;

                // Return: R = Â + V(s)
                returns[i] = gae + value;

                if (exp.done) gae = 0;
            }
        }

        /// Conjugate gradient to solve Ax = b where A = Fisher information matrix
        ///
        /// Fisher matrix: F = E[∇log π(a|s) ∇log π(a|s)ᵀ]
        /// Approximately solves: F × x = g (natural gradient)
        ///
        /// Time: O(cg_iters × K × m)
        /// Space: O(m²)
        fn conjugateGradient(self: *const Self, experiences: []const Experience, grad: []const T, result: []T) !void {
            @memset(result, 0);

            const residual = try self.allocator.alloc(T, grad.len);
            defer self.allocator.free(residual);
            @memcpy(residual, grad);

            const direction = try self.allocator.alloc(T, grad.len);
            defer self.allocator.free(direction);
            @memcpy(direction, residual);

            const fisher_dir = try self.allocator.alloc(T, grad.len);
            defer self.allocator.free(fisher_dir);

            var iter: usize = 0;
            while (iter < self.cg_iters) : (iter += 1) {
                // Compute Fisher-vector product: Fv
                try self.fisherVectorProduct(experiences, direction, fisher_dir);

                // Add damping: Fv + damping × v
                for (fisher_dir, 0..) |*fv, i| {
                    fv.* += self.damping * direction[i];
                }

                // α = rᵀr / dᵀFd
                var r_dot_r: T = 0;
                var d_dot_fd: T = 0;
                for (residual) |r| r_dot_r += r * r;
                for (direction, 0..) |d, i| d_dot_fd += d * fisher_dir[i];

                if (@abs(d_dot_fd) < 1e-10) break;
                const alpha = r_dot_r / d_dot_fd;

                // x = x + α × d
                for (result, 0..) |*x, i| {
                    x.* += alpha * direction[i];
                }

                // r = r - α × Fd
                for (residual, 0..) |*r, i| {
                    r.* -= alpha * fisher_dir[i];
                }

                // β = r_new / r_old
                var r_new_dot: T = 0;
                for (residual) |r| r_new_dot += r * r;

                if (r_new_dot < 1e-10) break;
                const beta = r_new_dot / r_dot_r;

                // d = r + β × d
                for (direction, 0..) |*d, i| {
                    d.* = residual[i] + beta * d.*;
                }
            }
        }

        /// Compute Fisher information matrix - vector product
        ///
        /// F × v ≈ E[∇log π(a|s) (∇log π(a|s)ᵀ × v)]
        ///
        /// Time: O(K × m)
        /// Space: O(m)
        fn fisherVectorProduct(self: *const Self, experiences: []const Experience, vec: []const T, result: []T) !void {
            @memset(result, 0);

            for (experiences) |exp| {
                const offset = exp.state * self.num_actions;
                const log_probs = self.log_policy[offset..offset + self.num_actions];

                // Compute probabilities
                var max_lp = log_probs[0];
                for (log_probs[1..]) |lp| {
                    if (lp > max_lp) max_lp = lp;
                }

                var sum: T = 0;
                var i: usize = 0;
                while (i < self.num_actions) : (i += 1) {
                    sum += @exp(log_probs[i] - max_lp);
                }

                // ∇log π(a|s) = e_a - π(·|s) (one-hot minus distribution)
                // (∇log π)ᵀ × v for each action
                var dot: T = 0;
                i = 0;
                while (i < self.num_actions) : (i += 1) {
                    const prob = @exp(log_probs[i] - max_lp) / sum;
                    const indicator: T = if (i == exp.action) 1 else 0;
                    const grad_component = indicator - prob;
                    dot += grad_component * vec[offset + i];
                }

                // F × v = E[∇log π × dot]
                i = 0;
                while (i < self.num_actions) : (i += 1) {
                    const prob = @exp(log_probs[i] - max_lp) / sum;
                    const indicator: T = if (i == exp.action) 1 else 0;
                    const grad_component = indicator - prob;
                    result[offset + i] += grad_component * dot;
                }
            }

            // Average over trajectory
            const scale = 1.0 / @as(T, @floatFromInt(experiences.len));
            for (result) |*r| {
                r.* *= scale;
            }
        }

        /// Compute KL divergence: KL(π_old || π_new)
        ///
        /// KL = Σ π_old(a|s) log(π_old(a|s) / π_new(a|s))
        ///
        /// Time: O(K × m)
        /// Space: O(1)
        fn computeKL(self: *const Self, experiences: []const Experience, old_policy: []const T) !T {
            var kl: T = 0;
            var count: usize = 0;

            for (experiences) |exp| {
                const offset = exp.state * self.num_actions;
                const old_log_probs = old_policy[offset..offset + self.num_actions];
                const new_log_probs = self.log_policy[offset..offset + self.num_actions];

                // Compute old probabilities
                var max_old = old_log_probs[0];
                for (old_log_probs[1..]) |lp| {
                    if (lp > max_old) max_old = lp;
                }

                var sum_old: T = 0;
                for (old_log_probs) |lp| {
                    sum_old += @exp(lp - max_old);
                }

                // Compute new probabilities
                var max_new = new_log_probs[0];
                for (new_log_probs[1..]) |lp| {
                    if (lp > max_new) max_new = lp;
                }

                var sum_new: T = 0;
                for (new_log_probs) |lp| {
                    sum_new += @exp(lp - max_new);
                }

                // KL = Σ p_old × log(p_old / p_new)
                for (0..self.num_actions) |a| {
                    const p_old = @exp(old_log_probs[a] - max_old) / sum_old;
                    const p_new = @exp(new_log_probs[a] - max_new) / sum_new;

                    if (p_old > 1e-10 and p_new > 1e-10) {
                        kl += p_old * @log(p_old / p_new);
                    }
                }

                count += 1;
            }

            return if (count > 0) kl / @as(T, @floatFromInt(count)) else 0;
        }

        /// Normalize policy to valid log probabilities
        ///
        /// Time: O(n × m)
        /// Space: O(1)
        fn normalizePolicy(self: *Self) void {
            var s: usize = 0;
            while (s < self.num_states) : (s += 1) {
                const offset = s * self.num_actions;
                const log_probs = self.log_policy[offset..offset + self.num_actions];

                // Log-sum-exp for numerical stability
                var max_lp = log_probs[0];
                for (log_probs[1..]) |lp| {
                    if (lp > max_lp) max_lp = lp;
                }

                var sum: T = 0;
                for (log_probs) |lp| {
                    sum += @exp(lp - max_lp);
                }

                const log_sum = max_lp + @log(sum);
                for (log_probs) |*lp| {
                    lp.* -= log_sum;
                }
            }
        }

        /// Compute mean of values
        fn mean(self: *const Self, values: []const T) T {
            _ = self;
            if (values.len == 0) return 0;
            var sum: T = 0;
            for (values) |v| sum += v;
            return sum / @as(T, @floatFromInt(values.len));
        }

        /// Compute standard deviation
        fn stdDev(self: *const Self, values: []const T, mean_val: T) T {
            _ = self;
            if (values.len == 0) return 0;
            var sum_sq: T = 0;
            for (values) |v| {
                const diff = v - mean_val;
                sum_sq += diff * diff;
            }
            return @sqrt(sum_sq / @as(T, @floatFromInt(values.len)));
        }

        /// Get policy probabilities for a state
        ///
        /// Time: O(m)
        /// Space: O(1)
        pub fn getPolicyProbs(self: *const Self, state: usize, probs: []T) !void {
            if (state >= self.num_states) return error.InvalidState;
            if (probs.len != self.num_actions) return error.InvalidBuffer;

            const offset = state * self.num_actions;
            const log_probs = self.log_policy[offset..offset + self.num_actions];

            var max_lp = log_probs[0];
            for (log_probs[1..]) |lp| {
                if (lp > max_lp) max_lp = lp;
            }

            var sum: T = 0;
            for (log_probs, 0..) |lp, i| {
                probs[i] = @exp(lp - max_lp);
                sum += probs[i];
            }

            for (probs) |*p| {
                p.* /= sum;
            }
        }

        /// Get value function estimate for a state
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn getValue(self: *const Self, state: usize) !T {
            if (state >= self.num_states) return error.InvalidState;
            return self.value_fn[state];
        }

        /// Reset agent to initial state
        ///
        /// Time: O(n × m)
        /// Space: O(1)
        pub fn reset(self: *Self) void {
            const log_uniform = @log(@as(T, 1.0) / @as(T, @floatFromInt(self.num_actions)));
            @memset(self.log_policy, log_uniform);
            @memset(self.value_fn, 0);
        }
    };
}

// ============================= Tests =============================

const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectError = testing.expectError;

test "TRPO: initialization" {
    const allocator = testing.allocator;

    var agent = try TRPO(f64).init(allocator, 4, 2, .{});
    defer agent.deinit();

    try expectEqual(@as(usize, 4), agent.num_states);
    try expectEqual(@as(usize, 2), agent.num_actions);
    try expectEqual(@as(usize, 8), agent.log_policy.len);
    try expectEqual(@as(usize, 4), agent.value_fn.len);
}

test "TRPO: uniform initial policy" {
    const allocator = testing.allocator;

    var agent = try TRPO(f64).init(allocator, 3, 2, .{});
    defer agent.deinit();

    // Check uniform distribution
    var probs: [2]f64 = undefined;
    try agent.getPolicyProbs(0, &probs);

    try expectApproxEqAbs(@as(f64, 0.5), probs[0], 1e-6);
    try expectApproxEqAbs(@as(f64, 0.5), probs[1], 1e-6);
}

test "TRPO: action selection (stochastic)" {
    const allocator = testing.allocator;

    var agent = try TRPO(f64).init(allocator, 2, 3, .{});
    defer agent.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const action = try agent.selectAction(0, rng);
    try testing.expect(action < 3);
}

test "TRPO: greedy action selection" {
    const allocator = testing.allocator;

    var agent = try TRPO(f64).init(allocator, 2, 3, .{});
    defer agent.deinit();

    // Modify policy to prefer action 1
    agent.log_policy[0 * 3 + 0] = @log(0.2);
    agent.log_policy[0 * 3 + 1] = @log(0.7);
    agent.log_policy[0 * 3 + 2] = @log(0.1);
    agent.normalizePolicy();

    const action = try agent.selectGreedyAction(0);
    try expectEqual(@as(usize, 1), action);
}

test "TRPO: experience storage" {
    const allocator = testing.allocator;

    var agent = try TRPO(f64).init(allocator, 4, 2, .{});
    defer agent.deinit();

    const experiences = [_]TRPO(f64).Experience{
        .{ .state = 0, .action = 0, .reward = 1.0, .next_state = 1, .done = false },
        .{ .state = 1, .action = 1, .reward = 0.5, .next_state = 2, .done = false },
        .{ .state = 2, .action = 0, .reward = -0.5, .next_state = 3, .done = false },
        .{ .state = 3, .action = 1, .reward = 1.0, .next_state = 0, .done = true },
    };

    try agent.update(&experiences);

    // Policy should have changed from uniform
    var probs: [2]f64 = undefined;
    try agent.getPolicyProbs(0, &probs);

    // Not checking exact values, just that it's not uniform anymore
    try testing.expect(@abs(probs[0] - 0.5) > 0.01 or @abs(probs[1] - 0.5) > 0.01);
}

test "TRPO: GAE computation" {
    const allocator = testing.allocator;

    var agent = try TRPO(f64).init(allocator, 4, 2, .{ .gamma = 0.9, .lambda = 0.95 });
    defer agent.deinit();

    // Set some value estimates
    agent.value_fn[0] = 0.5;
    agent.value_fn[1] = 0.7;
    agent.value_fn[2] = 0.3;
    agent.value_fn[3] = 0.1;

    const experiences = [_]TRPO(f64).Experience{
        .{ .state = 0, .action = 0, .reward = 1.0, .next_state = 1, .done = false },
        .{ .state = 1, .action = 1, .reward = 0.5, .next_state = 2, .done = false },
        .{ .state = 2, .action = 0, .reward = -0.5, .next_state = 3, .done = true },
    };

    const advantages = try allocator.alloc(f64, 3);
    defer allocator.free(advantages);

    const returns = try allocator.alloc(f64, 3);
    defer allocator.free(returns);

    try agent.computeGAE(&experiences, advantages, returns);

    // Advantages should be computed
    try testing.expect(advantages[0] != 0 or advantages[1] != 0 or advantages[2] != 0);
}

test "TRPO: KL divergence computation" {
    const allocator = testing.allocator;

    var agent = try TRPO(f64).init(allocator, 2, 2, .{});
    defer agent.deinit();

    const old_policy = try allocator.alloc(f64, 4);
    defer allocator.free(old_policy);
    @memcpy(old_policy, agent.log_policy);

    const experiences = [_]TRPO(f64).Experience{
        .{ .state = 0, .action = 0, .reward = 1.0, .next_state = 1, .done = false },
        .{ .state = 1, .action = 1, .reward = 0.5, .next_state = 0, .done = true },
    };

    // KL with itself should be ~0
    const kl_same = try agent.computeKL(&experiences, old_policy);
    try expectApproxEqAbs(@as(f64, 0), kl_same, 1e-6);

    // Change policy
    agent.log_policy[0] = @log(0.9);
    agent.log_policy[1] = @log(0.1);
    agent.normalizePolicy();

    // KL should be positive now
    const kl_diff = try agent.computeKL(&experiences, old_policy);
    try testing.expect(kl_diff > 0);
}

test "TRPO: value function update" {
    const allocator = testing.allocator;

    var agent = try TRPO(f64).init(allocator, 4, 2, .{ .alpha = 0.1 });
    defer agent.deinit();

    const initial_value = agent.value_fn[0];

    const experiences = [_]TRPO(f64).Experience{
        .{ .state = 0, .action = 0, .reward = 1.0, .next_state = 1, .done = false },
        .{ .state = 1, .action = 1, .reward = 0.5, .next_state = 2, .done = false },
    };

    try agent.update(&experiences);

    // Value function should have changed
    try testing.expect(agent.value_fn[0] != initial_value);
}

test "TRPO: terminal state handling" {
    const allocator = testing.allocator;

    var agent = try TRPO(f64).init(allocator, 3, 2, .{});
    defer agent.deinit();

    const experiences = [_]TRPO(f64).Experience{
        .{ .state = 0, .action = 0, .reward = 1.0, .next_state = 1, .done = false },
        .{ .state = 1, .action = 1, .reward = 0.5, .next_state = 2, .done = true },
    };

    try agent.update(&experiences);

    // Should complete without error
}

test "TRPO: policy improvement on simple chain" {
    const allocator = testing.allocator;

    var agent = try TRPO(f64).init(allocator, 2, 2, .{
        .gamma = 0.9,
        .lambda = 0.95,
        .alpha = 0.1,
        .delta = 0.01,
        .cg_iters = 5,
    });
    defer agent.deinit();

    // Simple 2-state MDP: S0 -> S1 -> terminal
    // Action 0 gives reward 1, action 1 gives reward 0
    var iter: usize = 0;
    while (iter < 10) : (iter += 1) {
        const experiences = [_]TRPO(f64).Experience{
            .{ .state = 0, .action = 0, .reward = 1.0, .next_state = 1, .done = false },
            .{ .state = 1, .action = 0, .reward = 1.0, .next_state = 0, .done = true },
        };

        try agent.update(&experiences);
    }

    // Should prefer action 0 in state 0
    const action = try agent.selectGreedyAction(0);
    try expectEqual(@as(usize, 0), action);
}

test "TRPO: reset functionality" {
    const allocator = testing.allocator;

    var agent = try TRPO(f64).init(allocator, 4, 2, .{});
    defer agent.deinit();

    // Modify state
    agent.log_policy[0] = 0.5;
    agent.value_fn[0] = 1.0;

    // Reset
    agent.reset();

    // Should be back to uniform policy
    var probs: [2]f64 = undefined;
    try agent.getPolicyProbs(0, &probs);

    try expectApproxEqAbs(@as(f64, 0.5), probs[0], 1e-6);
    try expectApproxEqAbs(@as(f64, 0), agent.value_fn[0], 1e-6);
}

test "TRPO: f32 support" {
    const allocator = testing.allocator;

    var agent = try TRPO(f32).init(allocator, 3, 2, .{});
    defer agent.deinit();

    try expectEqual(@as(usize, 3), agent.num_states);
    try expectEqual(@as(usize, 2), agent.num_actions);
}

test "TRPO: large state-action space" {
    const allocator = testing.allocator;

    var agent = try TRPO(f64).init(allocator, 20, 5, .{});
    defer agent.deinit();

    try expectEqual(@as(usize, 20), agent.num_states);
    try expectEqual(@as(usize, 5), agent.num_actions);

    const action = try agent.selectGreedyAction(10);
    try testing.expect(action < 5);
}

test "TRPO: invalid state error" {
    const allocator = testing.allocator;

    var agent = try TRPO(f64).init(allocator, 3, 2, .{});
    defer agent.deinit();

    try expectError(error.InvalidState, agent.selectGreedyAction(3));
}

test "TRPO: invalid action space error" {
    const allocator = testing.allocator;

    try expectError(error.InvalidActionSpace, TRPO(f64).init(allocator, 3, 0, .{}));
}

test "TRPO: invalid state space error" {
    const allocator = testing.allocator;

    try expectError(error.InvalidStateSpace, TRPO(f64).init(allocator, 0, 2, .{}));
}

test "TRPO: memory safety with multiple updates" {
    const allocator = testing.allocator;

    var agent = try TRPO(f64).init(allocator, 4, 2, .{});
    defer agent.deinit();

    var iter: usize = 0;
    while (iter < 5) : (iter += 1) {
        const experiences = [_]TRPO(f64).Experience{
            .{ .state = 0, .action = 0, .reward = 1.0, .next_state = 1, .done = false },
            .{ .state = 1, .action = 1, .reward = 0.5, .next_state = 2, .done = false },
            .{ .state = 2, .action = 0, .reward = -0.5, .next_state = 3, .done = false },
            .{ .state = 3, .action = 1, .reward = 1.0, .next_state = 0, .done = true },
        };

        try agent.update(&experiences);
    }
}
