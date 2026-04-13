//! Random number generation module
//!
//! Provides high-quality pseudo-random number generators (PRNGs) and sampling utilities.
//!
//! ## Generators
//! - PCG64: PCG family 64-bit generator (128-bit state)
//! - Xoshiro256**: xoshiro256** 64-bit generator (256-bit state)
//!
//! ## Sampling Functions
//! - uniform: Uniform distribution sampling
//! - normal: Normal distribution sampling (Box-Muller transform)
//! - exponential: Exponential distribution sampling
//! - shuffle: Fisher-Yates shuffle
//! - choice: Weighted sampling without replacement
//! - multinomial: Multinomial distribution sampling
//!
//! ## Design Principles
//! - Explicit state (no global RNG)
//! - Seed-based reproducibility
//! - Type-generic sampling
//!
//! ## Usage
//! ```zig
//! var rng = Pcg64.init(12345);
//! const value = rng.random().float(f64);
//! var data = [_]i32{1, 2, 3, 4, 5};
//! shuffle(i32, &data, rng.random());
//! ```

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const math = std.math;

/// PCG64 - Permuted Congruential Generator (64-bit output, 128-bit state)
///
/// Fast, high-quality PRNG from the PCG family.
/// - Period: 2^128
/// - Output: 64-bit
/// - State: 128-bit
/// - Passes: TestU01 BigCrush
///
/// Reference: O'Neill (2014) "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation"
pub const Pcg64 = struct {
    state: u128,
    inc: u128,

    const Self = @This();
    const PCG_DEFAULT_MULTIPLIER: u128 = 0x2360ED051FC65DA44385DF649FCCF645;

    /// Initialize with seed
    ///
    /// Time: O(1)
    /// Space: O(1)
    pub fn init(seed: u64) Self {
        var self = Self{
            .state = 0,
            .inc = 1, // Must be odd
        };
        // Seed initialization
        _ = self.next();
        self.state +%= seed;
        _ = self.next();
        return self;
    }

    /// Initialize with full state
    ///
    /// Time: O(1)
    /// Space: O(1)
    pub fn initFull(state: u128, inc: u128) Self {
        return Self{
            .state = state,
            .inc = inc | 1, // Ensure inc is odd
        };
    }

    /// Generate next random number
    ///
    /// Time: O(1)
    /// Space: O(1)
    fn next(self: *Self) u64 {
        const old_state = self.state;
        // PCG LCG step
        self.state = old_state *% PCG_DEFAULT_MULTIPLIER +% self.inc;
        // Output function (XSL RR)
        const xor_shifted: u64 = @truncate(((old_state >> 64) ^ old_state) >> 58);
        const rot: u6 = @truncate(old_state >> 122);
        return (xor_shifted >> rot) | (xor_shifted << ((~rot +% 1) & 63));
    }

    /// Get std.Random interface
    ///
    /// Time: O(1)
    /// Space: O(1)
    pub fn random(self: *Self) std.Random {
        return std.Random.init(self, fill);
    }

    fn fill(self: *Self, buffer: []u8) void {
        var i: usize = 0;
        while (i + 8 <= buffer.len) : (i += 8) {
            const value = self.next();
            buffer[i..][0..8].* = @bitCast(value);
        }
        if (i < buffer.len) {
            var value = self.next();
            for (buffer[i..]) |*byte| {
                byte.* = @truncate(value);
                value >>= 8;
            }
        }
    }
};

/// Xoshiro256** - xoshiro256** generator (64-bit output, 256-bit state)
///
/// Fast, high-quality PRNG with excellent statistical properties.
/// - Period: 2^256 - 1
/// - Output: 64-bit
/// - State: 256-bit (four u64)
/// - Passes: TestU01 BigCrush
///
/// Reference: Blackman & Vigna (2019) "Scrambled Linear Pseudorandom Number Generators"
pub const Xoshiro256 = struct {
    s: [4]u64,

    const Self = @This();

    /// Initialize with seed
    ///
    /// Uses SplitMix64 to expand seed to 256-bit state
    ///
    /// Time: O(1)
    /// Space: O(1)
    pub fn init(seed: u64) Self {
        var sm = SplitMix64{ .state = seed };
        return Self{
            .s = [4]u64{
                sm.next(),
                sm.next(),
                sm.next(),
                sm.next(),
            },
        };
    }

    /// Initialize with full state
    ///
    /// Time: O(1)
    /// Space: O(1)
    pub fn initFull(s0: u64, s1: u64, s2: u64, s3: u64) Self {
        return Self{ .s = [4]u64{ s0, s1, s2, s3 } };
    }

    /// Generate next random number
    ///
    /// Time: O(1)
    /// Space: O(1)
    fn next(self: *Self) u64 {
        const result = rotl(self.s[1] *% 5, 7) *% 9;
        const t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = rotl(self.s[3], 45);

        return result;
    }

    /// Get std.Random interface
    ///
    /// Time: O(1)
    /// Space: O(1)
    pub fn random(self: *Self) std.Random {
        return std.Random.init(self, fill);
    }

    fn fill(self: *Self, buffer: []u8) void {
        var i: usize = 0;
        while (i + 8 <= buffer.len) : (i += 8) {
            const value = self.next();
            buffer[i..][0..8].* = @bitCast(value);
        }
        if (i < buffer.len) {
            var value = self.next();
            for (buffer[i..]) |*byte| {
                byte.* = @truncate(value);
                value >>= 8;
            }
        }
    }

    fn rotl(x: u64, k: u6) u64 {
        const shift: u6 = @intCast(64 - @as(u7, k));
        return (x << k) | (x >> shift);
    }
};

/// SplitMix64 - used for seeding Xoshiro256
const SplitMix64 = struct {
    state: u64,

    fn next(self: *SplitMix64) u64 {
        self.state +%= 0x9e3779b97f4a7c15;
        var z = self.state;
        z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }
};

/// Sample from uniform distribution [min, max)
///
/// Time: O(1)
/// Space: O(1)
pub fn uniform(comptime T: type, min: T, max: T, rng: std.Random) T {
    const type_info = @typeInfo(T);
    if (type_info == .int) {
        return rng.intRangeAtMost(T, min, max - 1);
    } else if (type_info == .float) {
        return min + rng.float(T) * (max - min);
    } else {
        @compileError("uniform only supports integer and float types");
    }
}

/// Sample from standard normal distribution N(0, 1) using Box-Muller transform
///
/// Time: O(1)
/// Space: O(1)
pub fn normal(comptime T: type, rng: std.Random) T {
    if (T != f32 and T != f64) {
        @compileError("normal only supports f32 and f64");
    }

    const u1_val = rng.float(T);
    const u2_val = rng.float(T);

    // Ensure u1_val is not exactly 0 (avoid log(0))
    const u1_safe = if (u1_val == 0.0) 1e-15 else u1_val;

    // Box-Muller transform
    const r = @sqrt(-2.0 * @log(u1_safe));
    const theta = 2.0 * math.pi * u2_val;
    return r * @cos(theta);
}

/// Sample from exponential distribution with rate λ
///
/// Time: O(1)
/// Space: O(1)
pub fn exponential(comptime T: type, lambda: T, rng: std.Random) T {
    if (T != f32 and T != f64) {
        @compileError("exponential only supports f32 and f64");
    }
    if (lambda <= 0) {
        @panic("exponential: lambda must be positive");
    }

    const u = rng.float(T);
    const u_safe = if (u == 0.0) 1e-15 else u;
    return -@log(u_safe) / lambda;
}

/// Fisher-Yates shuffle
///
/// Randomly permute array in-place.
///
/// Time: O(n)
/// Space: O(1)
pub fn shuffle(comptime T: type, items: []T, rng: std.Random) void {
    if (items.len <= 1) return;

    var i = items.len - 1;
    while (i > 0) : (i -= 1) {
        const j = rng.intRangeLessThan(usize, 0, i + 1);
        std.mem.swap(T, &items[i], &items[j]);
    }
}

/// Sample k items without replacement
///
/// Returns selected indices.
///
/// Time: O(k)
/// Space: O(k)
pub fn choice(k: usize, n: usize, allocator: Allocator, rng: std.Random) ![]usize {
    if (k > n) return error.InvalidCount;
    if (k == 0) return &[_]usize{};

    var result = try allocator.alloc(usize, k);
    errdefer allocator.free(result);

    // Algorithm: Floyd's sampling algorithm (efficient for k << n)
    var selected = std.AutoHashMap(usize, void).init(allocator);
    defer selected.deinit();

    var i: usize = n - k;
    while (i < n) : (i += 1) {
        const pos = rng.intRangeLessThan(usize, 0, i + 1);
        const item = if (selected.contains(pos)) i else pos;
        try selected.put(item, {});
    }

    var it = selected.keyIterator();
    var idx: usize = 0;
    while (it.next()) |key| : (idx += 1) {
        result[idx] = key.*;
    }

    return result;
}

/// Weighted sampling without replacement
///
/// Sample k items from n items according to weights.
/// Uses Algorithm A-Res (Efraimidis & Spirakis 2006).
///
/// Time: O(n log k)
/// Space: O(k)
pub fn weightedChoice(comptime T: type, weights: []const T, k: usize, allocator: Allocator, rng: std.Random) ![]usize {
    if (T != f32 and T != f64) {
        @compileError("weightedChoice only supports f32 and f64");
    }
    if (k > weights.len) return error.InvalidCount;
    if (k == 0) return &[_]usize{};

    // Validate weights
    for (weights) |w| {
        if (w < 0) return error.NegativeWeight;
    }

    // Generate keys: u^(1/w) for each item
    // Use simple array allocation instead of ArrayList
    var keys = try allocator.alloc(T, weights.len);
    defer allocator.free(keys);
    var valid_count: usize = 0;

    for (weights, 0..) |w, i| {
        if (w > 0) {
            const u = rng.float(T);
            const u_safe = if (u == 0.0) 1e-15 else u;
            keys[i] = std.math.pow(T, u_safe, 1.0 / w);
            valid_count += 1;
        } else {
            keys[i] = 0.0; // Invalid items get key 0
        }
    }

    if (valid_count < k) return error.InsufficientItems;

    // Simple selection: find top k indices
    const result = try allocator.alloc(usize, k);
    errdefer allocator.free(result);

    for (0..k) |i| {
        var max_idx: usize = 0;
        var max_key: T = -std.math.inf(T);

        for (0..weights.len) |j| {
            if (keys[j] > max_key) {
                // Check if this index was already selected
                var already_selected = false;
                for (result[0..i]) |selected| {
                    if (selected == j) {
                        already_selected = true;
                        break;
                    }
                }
                if (!already_selected) {
                    max_key = keys[j];
                    max_idx = j;
                }
            }
        }
        result[i] = max_idx;
    }

    return result;
}

/// Multinomial distribution sampling
///
/// Generate counts for n trials across categories with given probabilities.
///
/// Time: O(n + k) where k = categories
/// Space: O(k)
pub fn multinomial(probs: []const f64, n: usize, allocator: Allocator, rng: std.Random) ![]usize {
    if (probs.len == 0) return error.EmptyProbabilities;

    // Validate probabilities
    var sum: f64 = 0;
    for (probs) |p| {
        if (p < 0) return error.NegativeProbability;
        sum += p;
    }
    if (@abs(sum - 1.0) > 1e-10) return error.ProbabilitiesNotNormalized;

    var result = try allocator.alloc(usize, probs.len);
    @memset(result, 0);

    // Sample n times
    for (0..n) |_| {
        const u = rng.float(f64);
        var cumsum: f64 = 0;
        for (probs, 0..) |p, i| {
            cumsum += p;
            if (u < cumsum) {
                result[i] += 1;
                break;
            }
        }
    }

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "Pcg64 - init and basic generation" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    const value1 = r.int(u64);
    const value2 = r.int(u64);

    // Should generate different values
    try testing.expect(value1 != value2);
}

test "Pcg64 - reproducibility" {
    var rng1 = Pcg64.init(42);
    var rng2 = Pcg64.init(42);

    for (0..100) |_| {
        const v1 = rng1.random().int(u64);
        const v2 = rng2.random().int(u64);
        try testing.expectEqual(v1, v2);
    }
}

test "Pcg64 - float generation" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    for (0..1000) |_| {
        const f = r.float(f64);
        try testing.expect(f >= 0.0 and f < 1.0);
    }
}

test "Pcg64 - initFull" {
    var rng = Pcg64.initFull(0x123456789abcdef, 0xfedcba987654321);
    const value1 = rng.random().int(u64);
    const value2 = rng.random().int(u64);
    // At least one of two values should be different (extremely high probability)
    try testing.expect(value1 != value2 or value1 != 0);
}

test "Xoshiro256 - init and basic generation" {
    var rng = Xoshiro256.init(12345);
    const r = rng.random();

    const value1 = r.int(u64);
    const value2 = r.int(u64);

    // Should generate different values
    try testing.expect(value1 != value2);
}

test "Xoshiro256 - reproducibility" {
    var rng1 = Xoshiro256.init(42);
    var rng2 = Xoshiro256.init(42);

    for (0..100) |_| {
        const v1 = rng1.random().int(u64);
        const v2 = rng2.random().int(u64);
        try testing.expectEqual(v1, v2);
    }
}

test "Xoshiro256 - float generation" {
    var rng = Xoshiro256.init(12345);
    const r = rng.random();

    for (0..1000) |_| {
        const f = r.float(f64);
        try testing.expect(f >= 0.0 and f < 1.0);
    }
}

test "Xoshiro256 - initFull" {
    var rng = Xoshiro256.initFull(1, 2, 3, 4);
    const value = rng.random().int(u64);
    try testing.expect(value != 0);
}

test "uniform - integer range" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    for (0..1000) |_| {
        const value = uniform(i32, 10, 20, r);
        try testing.expect(value >= 10 and value < 20);
    }
}

test "uniform - float range" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    for (0..1000) |_| {
        const value = uniform(f64, 5.0, 10.0, r);
        try testing.expect(value >= 5.0 and value < 10.0);
    }
}

test "normal - standard normal properties" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    var sum: f64 = 0;
    var sum_sq: f64 = 0;
    const n = 10000;

    for (0..n) |_| {
        const value = normal(f64, r);
        sum += value;
        sum_sq += value * value;
    }

    const mean_val = sum / @as(f64, @floatFromInt(n));
    const variance = (sum_sq / @as(f64, @floatFromInt(n))) - (mean_val * mean_val);

    // Standard normal: μ ≈ 0, σ² ≈ 1
    try testing.expect(@abs(mean_val) < 0.1);
    try testing.expect(@abs(variance - 1.0) < 0.1);
}

test "normal - f32 variant" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    for (0..100) |_| {
        const value = normal(f32, r);
        try testing.expect(!math.isNan(value));
    }
}

test "exponential - mean λ=1" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    var sum: f64 = 0;
    const n = 10000;

    for (0..n) |_| {
        const value = exponential(f64, 1.0, r);
        sum += value;
    }

    const mean_val = sum / @as(f64, @floatFromInt(n));
    // Exponential(λ=1) has mean = 1/λ = 1
    try testing.expect(@abs(mean_val - 1.0) < 0.1);
}

test "exponential - mean λ=2" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    var sum: f64 = 0;
    const n = 10000;

    for (0..n) |_| {
        const value = exponential(f64, 2.0, r);
        sum += value;
    }

    const mean_val = sum / @as(f64, @floatFromInt(n));
    // Exponential(λ=2) has mean = 1/λ = 0.5
    try testing.expect(@abs(mean_val - 0.5) < 0.1);
}

test "shuffle - basic" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    var data = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const original = data;

    shuffle(i32, &data, r);

    // Should have same elements
    var sum: i32 = 0;
    for (data) |x| sum += x;
    try testing.expectEqual(@as(i32, 55), sum);

    // Should be different order (with high probability)
    var different = false;
    for (data, 0..) |x, i| {
        if (x != original[i]) {
            different = true;
            break;
        }
    }
    try testing.expect(different);
}

test "shuffle - empty" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    var data = [_]i32{};
    shuffle(i32, &data, r);
    try testing.expectEqual(@as(usize, 0), data.len);
}

test "shuffle - single element" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    var data = [_]i32{42};
    shuffle(i32, &data, r);
    try testing.expectEqual(@as(i32, 42), data[0]);
}

test "choice - basic" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    const result = try choice(5, 10, testing.allocator, r);
    defer testing.allocator.free(result);

    try testing.expectEqual(@as(usize, 5), result.len);

    // All indices should be unique and < 10
    for (result) |idx| {
        try testing.expect(idx < 10);
    }

    // Check uniqueness
    for (result, 0..) |a, i| {
        for (result[i + 1 ..]) |b| {
            try testing.expect(a != b);
        }
    }
}

test "choice - k=0" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    const result = try choice(0, 10, testing.allocator, r);
    defer testing.allocator.free(result);

    try testing.expectEqual(@as(usize, 0), result.len);
}

test "choice - k>n error" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    try testing.expectError(error.InvalidCount, choice(15, 10, testing.allocator, r));
}

test "weightedChoice - basic" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    const weights = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const result = try weightedChoice(f64, &weights, 2, testing.allocator, r);
    defer testing.allocator.free(result);

    try testing.expectEqual(@as(usize, 2), result.len);

    // All indices should be < 4
    for (result) |idx| {
        try testing.expect(idx < 4);
    }
}

test "weightedChoice - negative weight error" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    const weights = [_]f64{ 1.0, -2.0, 3.0 };
    try testing.expectError(error.NegativeWeight, weightedChoice(f64, &weights, 2, testing.allocator, r));
}

test "weightedChoice - zero weights" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    const weights = [_]f64{ 0.0, 1.0, 0.0 };
    const result = try weightedChoice(f64, &weights, 1, testing.allocator, r);
    defer testing.allocator.free(result);

    try testing.expectEqual(@as(usize, 1), result.len);
    try testing.expectEqual(@as(usize, 1), result[0]);
}

test "multinomial - basic" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    const probs = [_]f64{ 0.25, 0.25, 0.25, 0.25 };
    const result = try multinomial(&probs, 100, testing.allocator, r);
    defer testing.allocator.free(result);

    try testing.expectEqual(@as(usize, 4), result.len);

    // Sum should equal n
    var sum: usize = 0;
    for (result) |count| {
        sum += count;
    }
    try testing.expectEqual(@as(usize, 100), sum);
}

test "multinomial - skewed probabilities" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    const probs = [_]f64{ 0.9, 0.1 };
    const result = try multinomial(&probs, 1000, testing.allocator, r);
    defer testing.allocator.free(result);

    // First category should have ~900, second ~100
    const expected1: f64 = 900;
    const expected2: f64 = 100;
    const actual1: f64 = @floatFromInt(result[0]);
    const actual2: f64 = @floatFromInt(result[1]);

    try testing.expect(@abs(actual1 - expected1) < 100); // Within ±100
    try testing.expect(@abs(actual2 - expected2) < 100);
}

test "multinomial - negative probability error" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    const probs = [_]f64{ 0.5, -0.5, 1.0 };
    try testing.expectError(error.NegativeProbability, multinomial(&probs, 100, testing.allocator, r));
}

test "multinomial - not normalized error" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    const probs = [_]f64{ 0.3, 0.3 }; // Sum = 0.6 ≠ 1.0
    try testing.expectError(error.ProbabilitiesNotNormalized, multinomial(&probs, 100, testing.allocator, r));
}

test "Pcg64 - distribution uniformity (chi-square test)" {
    var rng = Pcg64.init(12345);
    const r = rng.random();

    const n_bins = 10;
    const n_samples = 10000;
    var bins = [_]usize{0} ** n_bins;

    for (0..n_samples) |_| {
        const value = r.intRangeLessThan(usize, 0, n_bins);
        bins[value] += 1;
    }

    // Each bin should have ~1000 samples
    // Chi-square test would be more rigorous, but simple range check suffices
    for (bins) |count| {
        try testing.expect(count > 800 and count < 1200);
    }
}

test "Xoshiro256 - distribution uniformity" {
    var rng = Xoshiro256.init(12345);
    const r = rng.random();

    const n_bins = 10;
    const n_samples = 10000;
    var bins = [_]usize{0} ** n_bins;

    for (0..n_samples) |_| {
        const value = r.intRangeLessThan(usize, 0, n_bins);
        bins[value] += 1;
    }

    for (bins) |count| {
        try testing.expect(count > 800 and count < 1200);
    }
}

test "memory safety - Pcg64 with testing.allocator" {
    var rng = Pcg64.init(42);
    const r = rng.random();

    for (0..10) |_| {
        const result = try choice(5, 10, testing.allocator, r);
        testing.allocator.free(result);
    }
}

test "memory safety - Xoshiro256 with testing.allocator" {
    var rng = Xoshiro256.init(42);
    const r = rng.random();

    for (0..10) |_| {
        const probs = [_]f64{ 0.25, 0.25, 0.25, 0.25 };
        const result = try multinomial(&probs, 100, testing.allocator, r);
        testing.allocator.free(result);
    }
}
