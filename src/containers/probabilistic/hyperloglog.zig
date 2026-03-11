const std = @import("std");
const testing = std.testing;

/// HyperLogLog - cardinality estimation (count distinct elements)
///
/// Estimates the number of distinct elements in a multiset with logarithmic space.
/// Uses p bits for bucket selection, stores maximum leading zeros in each bucket.
///
/// Standard error: 1.04/√(2^p) for p precision bits (m = 2^p buckets)
/// - p=10 (1024 buckets): ~3.2% error, ~1KB memory
/// - p=14 (16384 buckets): ~0.81% error, ~16KB memory
/// - p=16 (65536 buckets): ~0.4% error, ~64KB memory
///
/// Operations:
/// - add(item): update sketch with new item, O(1)
/// - count(): estimate cardinality, O(m) where m = 2^p
/// - merge(other): combine two HLL sketches, O(m)
///
/// Consumer: zoltraak (Redis PFCOUNT), unique visitor counting
pub fn HyperLogLog(
    comptime T: type,
    comptime Context: type,
    comptime hashFn: fn (ctx: Context, key: T) u64,
) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        registers: []u8, // m = 2^p registers, each stores max leading zeros
        p: u5, // Precision (4 ≤ p ≤ 18)
        m: usize, // Number of registers (2^p)
        ctx: Context,

        // Alpha correction factor for bias correction (depends on m)
        // For large m, α_m ≈ 0.7213/(1 + 1.079/m)
        fn alphaM(m: usize) f64 {
            return switch (m) {
                16 => 0.673,
                32 => 0.697,
                64 => 0.709,
                else => 0.7213 / (1.0 + 1.079 / @as(f64, @floatFromInt(m))),
            };
        }

        /// Time: O(1) | Space: O(2^p)
        /// p: precision bits (4 ≤ p ≤ 18), higher = more accurate but more memory
        pub fn init(allocator: std.mem.Allocator, p: u5, ctx: Context) !Self {
            if (p < 4 or p > 18) return error.InvalidPrecision;

            const m: usize = @as(usize, 1) << p; // 2^p
            const registers = try allocator.alloc(u8, m);
            @memset(registers, 0);

            return Self{
                .allocator = allocator,
                .registers = registers,
                .p = p,
                .m = m,
                .ctx = ctx,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.registers);
            self.* = undefined;
        }

        /// Count leading zeros in a 64-bit value after the first p bits
        /// Returns number of leading zeros + 1 (1-indexed, 0 means hash was 0)
        fn rho(w: u64, p: u5) u8 {
            if (w == 0) return @intCast(@as(u8, 65) - @as(u8, p)); // All zeros
            // Count leading zeros in the remaining (64 - p) bits
            const leading = @clz(w << p);
            return @intCast(leading + 1);
        }

        /// Add an item to the HyperLogLog sketch
        /// Time: O(1) | Space: O(1)
        pub fn add(self: *Self, item: T) void {
            const hash = hashFn(self.ctx, item);

            // Use first p bits as bucket index
            const shift_amount: u6 = @intCast(@as(u8, 64) - @as(u8, self.p));
            const j = hash >> shift_amount;

            // Count leading zeros in remaining bits + 1
            const leading_zeros = rho(hash, self.p);

            // Update register with maximum
            self.registers[j] = @max(self.registers[j], leading_zeros);
        }

        /// Estimate cardinality (number of distinct elements)
        /// Time: O(m) where m = 2^p | Space: O(1)
        pub fn count(self: *const Self) u64 {
            // Harmonic mean of 2^M[j] for all registers
            var sum: f64 = 0.0;
            var zeros: usize = 0;

            for (self.registers) |reg| {
                if (reg == 0) {
                    zeros += 1;
                }
                sum += std.math.pow(f64, 2.0, -@as(f64, @floatFromInt(reg)));
            }

            const alpha = alphaM(self.m);
            const m_float = @as(f64, @floatFromInt(self.m));
            var estimate = alpha * m_float * m_float / sum;

            // Small range correction (LinearCounting)
            if (estimate <= 2.5 * m_float) {
                if (zeros != 0) {
                    const zeros_float = @as(f64, @floatFromInt(zeros));
                    estimate = m_float * @log(m_float / zeros_float);
                }
            }
            // Large range correction (for hash collisions beyond 2^32)
            else if (estimate > (1.0 / 30.0) * std.math.pow(f64, 2.0, 32.0)) {
                estimate = -std.math.pow(f64, 2.0, 32.0) * @log(1.0 - estimate / std.math.pow(f64, 2.0, 32.0));
            }

            return @intFromFloat(@max(0, @round(estimate)));
        }

        /// Merge another HyperLogLog with same precision
        /// Time: O(m) | Space: O(1)
        pub fn merge(self: *Self, other: *const Self) error{IncompatibleHLL}!void {
            if (self.p != other.p) {
                return error.IncompatibleHLL;
            }

            for (self.registers, other.registers) |*self_reg, other_reg| {
                self_reg.* = @max(self_reg.*, other_reg);
            }
        }

        /// Clear all registers
        /// Time: O(m) | Space: O(1)
        pub fn clear(self: *Self) void {
            @memset(self.registers, 0);
        }

        /// Memory usage in bytes
        /// Time: O(1) | Space: O(1)
        pub fn memoryUsage(self: *const Self) usize {
            return self.registers.len;
        }
    };
}

// Default hash function for integers
pub fn defaultHashInt(comptime T: type) fn (void, T) u64 {
    return struct {
        fn hash(_: void, key: T) u64 {
            var hasher = std.hash.Wyhash.init(0);
            const bytes = std.mem.asBytes(&key);
            hasher.update(bytes);
            return hasher.final();
        }
    }.hash;
}

// Default hash function for slices
pub fn defaultHashSlice(comptime T: type) fn (void, []const T) u64 {
    return struct {
        fn hash(_: void, key: []const T) u64 {
            var hasher = std.hash.Wyhash.init(0);
            for (key) |item| {
                const bytes = std.mem.asBytes(&item);
                hasher.update(bytes);
            }
            return hasher.final();
        }
    }.hash;
}

// Tests
test "HyperLogLog - basic operations" {
    const HLL = HyperLogLog(u32, void, defaultHashInt(u32));
    var hll = try HLL.init(testing.allocator, 10, {});
    defer hll.deinit();

    // Empty HLL
    try testing.expectEqual(@as(u64, 0), hll.count());

    // Add distinct items
    for (0..100) |i| {
        hll.add(@intCast(i));
    }

    const estimate = hll.count();
    // With p=10 (~3.2% error), estimate should be close to 100
    // Allow ±10% error for 100 items
    try testing.expect(estimate >= 90);
    try testing.expect(estimate <= 110);
}

test "HyperLogLog - cardinality estimation accuracy" {
    const HLL = HyperLogLog(u32, void, defaultHashInt(u32));
    var hll = try HLL.init(testing.allocator, 14, {}); // p=14, ~0.81% error
    defer hll.deinit();

    const true_cardinality: usize = 10000;

    // Add 10k distinct items
    for (0..true_cardinality) |i| {
        hll.add(@intCast(i));
    }

    const estimate = hll.count();
    const true_card_f64 = @as(f64, @floatFromInt(true_cardinality));
    const estimate_f64 = @as(f64, @floatFromInt(estimate));
    const error_rate = @abs(estimate_f64 - true_card_f64) / true_card_f64;

    // With p=14, standard error is ~0.81%, allow up to 5% for test stability
    try testing.expect(error_rate < 0.05);
}

test "HyperLogLog - duplicates" {
    const HLL = HyperLogLog(u32, void, defaultHashInt(u32));
    var hll = try HLL.init(testing.allocator, 12, {});
    defer hll.deinit();

    // Add 100 distinct items, each 10 times
    for (0..100) |i| {
        for (0..10) |_| {
            hll.add(@intCast(i));
        }
    }

    const estimate = hll.count();
    // Should estimate ~100 distinct items, not 1000 total
    try testing.expect(estimate >= 90);
    try testing.expect(estimate <= 110);
}

test "HyperLogLog - clear" {
    const HLL = HyperLogLog(u32, void, defaultHashInt(u32));
    var hll = try HLL.init(testing.allocator, 10, {});
    defer hll.deinit();

    for (0..100) |i| {
        hll.add(@intCast(i));
    }
    try testing.expect(hll.count() > 0);

    hll.clear();
    try testing.expectEqual(@as(u64, 0), hll.count());
}

test "HyperLogLog - string keys" {
    const HLL = HyperLogLog([]const u8, void, defaultHashSlice(u8));
    var hll = try HLL.init(testing.allocator, 10, {});
    defer hll.deinit();

    const words = [_][]const u8{ "apple", "banana", "cherry", "date", "elderberry" };

    // Add each word 10 times
    for (words) |word| {
        for (0..10) |_| {
            hll.add(word);
        }
    }

    const estimate = hll.count();
    // Should estimate ~5 distinct strings
    try testing.expect(estimate >= 4);
    try testing.expect(estimate <= 6);
}

test "HyperLogLog - merge" {
    const HLL = HyperLogLog(u32, void, defaultHashInt(u32));
    var hll1 = try HLL.init(testing.allocator, 10, {});
    defer hll1.deinit();
    var hll2 = try HLL.init(testing.allocator, 10, {});
    defer hll2.deinit();

    // hll1 has items 0-49
    for (0..50) |i| {
        hll1.add(@intCast(i));
    }

    // hll2 has items 25-74 (overlap with hll1)
    for (25..75) |i| {
        hll2.add(@intCast(i));
    }

    try hll1.merge(&hll2);

    const estimate = hll1.count();
    // Combined cardinality should be ~75 (0-74)
    try testing.expect(estimate >= 65);
    try testing.expect(estimate <= 85);
}

test "HyperLogLog - incompatible merge" {
    const HLL = HyperLogLog(u32, void, defaultHashInt(u32));
    var hll1 = try HLL.init(testing.allocator, 10, {});
    defer hll1.deinit();
    var hll2 = try HLL.init(testing.allocator, 12, {}); // Different precision
    defer hll2.deinit();

    try testing.expectError(error.IncompatibleHLL, hll1.merge(&hll2));
}

test "HyperLogLog - large cardinality" {
    const HLL = HyperLogLog(u32, void, defaultHashInt(u32));
    var hll = try HLL.init(testing.allocator, 14, {});
    defer hll.deinit();

    const true_cardinality: usize = 100000;

    // Add 100k distinct items
    for (0..true_cardinality) |i| {
        hll.add(@intCast(i));
    }

    const estimate = hll.count();
    const true_card_f64 = @as(f64, @floatFromInt(true_cardinality));
    const estimate_f64 = @as(f64, @floatFromInt(estimate));
    const error_rate = @abs(estimate_f64 - true_card_f64) / true_card_f64;

    // With p=14, allow up to 5% error
    try testing.expect(error_rate < 0.05);
}

test "HyperLogLog - small cardinality correction" {
    const HLL = HyperLogLog(u32, void, defaultHashInt(u32));
    var hll = try HLL.init(testing.allocator, 10, {});
    defer hll.deinit();

    // Very small cardinality (< 2.5 * m)
    for (0..10) |i| {
        hll.add(@intCast(i));
    }

    const estimate = hll.count();
    // LinearCounting correction should improve small range accuracy
    try testing.expect(estimate >= 8);
    try testing.expect(estimate <= 12);
}

test "HyperLogLog - edge cases" {
    const HLL = HyperLogLog(u32, void, defaultHashInt(u32));

    // Invalid precision
    try testing.expectError(error.InvalidPrecision, HLL.init(testing.allocator, 3, {})); // Too small
    try testing.expectError(error.InvalidPrecision, HLL.init(testing.allocator, 19, {})); // Too large

    // Minimal precision
    var hll_min = try HLL.init(testing.allocator, 4, {});
    defer hll_min.deinit();
    try testing.expectEqual(@as(usize, 16), hll_min.m); // 2^4 = 16

    // Maximal precision
    var hll_max = try HLL.init(testing.allocator, 18, {});
    defer hll_max.deinit();
    try testing.expectEqual(@as(usize, 262144), hll_max.m); // 2^18 = 262144
}

test "HyperLogLog - memory usage" {
    const HLL = HyperLogLog(u32, void, defaultHashInt(u32));

    // p=10 → 2^10 = 1024 bytes
    var hll10 = try HLL.init(testing.allocator, 10, {});
    defer hll10.deinit();
    try testing.expectEqual(@as(usize, 1024), hll10.memoryUsage());

    // p=14 → 2^14 = 16384 bytes
    var hll14 = try HLL.init(testing.allocator, 14, {});
    defer hll14.deinit();
    try testing.expectEqual(@as(usize, 16384), hll14.memoryUsage());
}
