const std = @import("std");
const testing = std.testing;

/// Count-Min Sketch - space-efficient frequency estimation with one-sided error
///
/// A Count-Min Sketch approximates item frequencies using a 2D array of counters (d x w)
/// with d independent hash functions. Never underestimates counts (one-sided error).
///
/// Operations:
/// - add(item, count): increment counters for item by count
/// - estimate(item): return estimated frequency (may overestimate)
/// - merge(other): add frequencies from another compatible sketch
///
/// Error bounds:
/// - Pr[estimate(item) > freq(item) + ε*N] ≤ δ
/// - where N = total count, ε = error rate, δ = failure probability
/// - Optimal parameters: w = ⌈e/ε⌉, d = ⌈ln(1/δ)⌉
///
/// Consumer: streaming analytics, network traffic monitoring, heavy hitters
pub fn CountMinSketch(
    comptime T: type,
    comptime Context: type,
    comptime hashFn: fn (ctx: Context, key: T, seed: u64) u64,
) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        table: [][]u64, // d x w table of counters
        d: usize, // Number of hash functions (depth)
        w: usize, // Width of each row
        total: u64, // Total count (N)
        ctx: Context,

        /// Time: O(1) | Space: O(d*w)
        /// d: depth (number of hash functions), w: width (number of counters per row)
        pub fn init(allocator: std.mem.Allocator, d: usize, w: usize, ctx: Context) !Self {
            if (d == 0 or w == 0) return error.InvalidDimensions;

            const table = try allocator.alloc([]u64, d);
            errdefer allocator.free(table);

            for (table, 0..) |*row, i| {
                row.* = allocator.alloc(u64, w) catch |err| {
                    // Free previously allocated rows
                    for (table[0..i]) |prev_row| {
                        allocator.free(prev_row);
                    }
                    allocator.free(table);
                    return err;
                };
                @memset(row.*, 0);
            }

            return Self{
                .allocator = allocator,
                .table = table,
                .d = d,
                .w = w,
                .total = 0,
                .ctx = ctx,
            };
        }

        /// Initialize with error bounds
        /// Time: O(1) | Space: O(d*w)
        /// epsilon: error rate (0 < ε < 1), delta: failure probability (0 < δ < 1)
        /// Formula: w = ⌈e/ε⌉ ≈ ⌈2.718/ε⌉, d = ⌈ln(1/δ)⌉
        pub fn initWithErrorBounds(
            allocator: std.mem.Allocator,
            epsilon: f64,
            delta: f64,
            ctx: Context,
        ) !Self {
            if (epsilon <= 0.0 or epsilon >= 1.0) return error.InvalidEpsilon;
            if (delta <= 0.0 or delta >= 1.0) return error.InvalidDelta;

            const e = 2.718281828459045; // Euler's number
            const w_float = e / epsilon;
            const w = @max(1, @as(usize, @intFromFloat(@ceil(w_float))));

            const d_float = @log(1.0 / delta);
            const d = @max(1, @as(usize, @intFromFloat(@ceil(d_float))));

            return Self.init(allocator, d, w, ctx);
        }

        /// Frees all allocated memory. Invalidates all iterators.
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            for (self.table) |row| {
                self.allocator.free(row);
            }
            self.allocator.free(self.table);
            self.* = undefined;
        }

        /// Add an item with a count (increment frequency)
        /// Time: O(d) | Space: O(1)
        pub fn add(self: *Self, item: T, count: u64) void {
            for (self.table, 0..) |row, i| {
                const hash = hashFn(self.ctx, item, i);
                const index = hash % self.w;
                row[index] += count;
            }
            self.total += count;
        }

        /// Estimate the frequency of an item
        /// Time: O(d) | Space: O(1)
        /// Returns minimum counter value across all hash functions
        /// May overestimate but never underestimates
        pub fn estimate(self: *const Self, item: T) u64 {
            var min_count: u64 = std.math.maxInt(u64);
            for (self.table, 0..) |row, i| {
                const hash = hashFn(self.ctx, item, i);
                const index = hash % self.w;
                min_count = @min(min_count, row[index]);
            }
            return min_count;
        }

        /// Clear all counters
        /// Time: O(d*w) | Space: O(1)
        pub fn clear(self: *Self) void {
            for (self.table) |row| {
                @memset(row, 0);
            }
            self.total = 0;
        }

        /// Merge another Count-Min Sketch with same dimensions
        /// Time: O(d*w) | Space: O(1)
        pub fn merge(self: *Self, other: *const Self) error{IncompatibleSketches}!void {
            if (self.d != other.d or self.w != other.w) {
                return error.IncompatibleSketches;
            }

            for (self.table, other.table) |self_row, other_row| {
                for (self_row, other_row) |*self_count, other_count| {
                    self_count.* += other_count;
                }
            }
            self.total += other.total;
        }

        /// Total count across all items
        /// Time: O(1) | Space: O(1)
        pub fn totalCount(self: *const Self) u64 {
            return self.total;
        }

        /// Estimate error for a specific item
        /// Time: O(d) | Space: O(1)
        /// Returns upper bound: estimate - actual ≤ ε*N with probability 1-δ
        pub fn estimateError(self: *const Self, epsilon: f64) u64 {
            const error_bound = epsilon * @as(f64, @floatFromInt(self.total));
            return @as(u64, @intFromFloat(@ceil(error_bound)));
        }

        /// Validate internal invariants
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: *const Self) void {
            std.debug.assert(self.d > 0);
            std.debug.assert(self.w > 0);
            std.debug.assert(self.table.len == self.d);
            for (self.table) |row| {
                std.debug.assert(row.len == self.w);
            }
        }
    };
}

// Default hash function for integers
/// Default hash function for integers.
/// Time: O(1) | Space: O(1)
pub fn defaultHashInt(comptime T: type) fn (void, T, u64) u64 {
    return struct {
        fn hash(_: void, key: T, seed: u64) u64 {
            var hasher = std.hash.Wyhash.init(seed);
            const bytes = std.mem.asBytes(&key);
            hasher.update(bytes);
            return hasher.final();
        }
    }.hash;
}

// Default hash function for slices
/// Default hash function for slices.
/// Time: O(n) | Space: O(1)
pub fn defaultHashSlice(comptime T: type) fn (void, []const T, u64) u64 {
    return struct {
        fn hash(_: void, key: []const T, seed: u64) u64 {
            var hasher = std.hash.Wyhash.init(seed);
            for (key) |item| {
                const bytes = std.mem.asBytes(&item);
                hasher.update(bytes);
            }
            return hasher.final();
        }
    }.hash;
}

// Tests
test "CountMinSketch - basic operations" {
    const Sketch = CountMinSketch(u32, void, defaultHashInt(u32));
    var sketch = try Sketch.init(testing.allocator, 4, 1000, {});
    defer sketch.deinit();

    // Empty sketch
    try testing.expectEqual(@as(u64, 0), sketch.estimate(42));
    try testing.expectEqual(@as(u64, 0), sketch.totalCount());

    // Add items
    sketch.add(42, 5);
    sketch.add(100, 10);
    sketch.add(42, 3); // Add 3 more to 42

    // Estimates should match or exceed actual counts
    const est_42 = sketch.estimate(42);
    const est_100 = sketch.estimate(100);
    try testing.expect(est_42 >= 8); // 5 + 3
    try testing.expect(est_100 >= 10);

    try testing.expectEqual(@as(u64, 18), sketch.totalCount());
}

test "CountMinSketch - error bounds" {
    const Sketch = CountMinSketch(u32, void, defaultHashInt(u32));
    // ε=0.01 (1% error), δ=0.01 (1% failure probability)
    var sketch = try Sketch.initWithErrorBounds(testing.allocator, 0.01, 0.01, {});
    defer sketch.deinit();

    // Add 1000 occurrences of item 1
    sketch.add(1, 1000);

    // Add 10000 total items (9000 other items)
    for (0..9000) |i| {
        sketch.add(@intCast(i + 2), 1);
    }

    const estimate = sketch.estimate(1);
    // With ε=0.01, error bound is 0.01 * 10000 = 100
    // So estimate should be in [1000, 1100] with high probability
    try testing.expect(estimate >= 1000);
    try testing.expect(estimate <= 1100); // May occasionally fail due to probabilistic nature
}

test "CountMinSketch - clear" {
    const Sketch = CountMinSketch(u32, void, defaultHashInt(u32));
    var sketch = try Sketch.init(testing.allocator, 4, 100, {});
    defer sketch.deinit();

    sketch.add(1, 10);
    sketch.add(2, 20);
    try testing.expect(sketch.estimate(1) >= 10);

    sketch.clear();
    try testing.expectEqual(@as(u64, 0), sketch.estimate(1));
    try testing.expectEqual(@as(u64, 0), sketch.estimate(2));
    try testing.expectEqual(@as(u64, 0), sketch.totalCount());
}

test "CountMinSketch - string keys" {
    const Sketch = CountMinSketch([]const u8, void, defaultHashSlice(u8));
    var sketch = try Sketch.init(testing.allocator, 5, 1000, {});
    defer sketch.deinit();

    sketch.add("hello", 10);
    sketch.add("world", 20);
    sketch.add("hello", 5);

    const est_hello = sketch.estimate("hello");
    const est_world = sketch.estimate("world");
    try testing.expect(est_hello >= 15);
    try testing.expect(est_world >= 20);
}

test "CountMinSketch - merge" {
    const Sketch = CountMinSketch(u32, void, defaultHashInt(u32));
    var sketch1 = try Sketch.init(testing.allocator, 4, 100, {});
    defer sketch1.deinit();
    var sketch2 = try Sketch.init(testing.allocator, 4, 100, {});
    defer sketch2.deinit();

    sketch1.add(1, 10);
    sketch1.add(2, 20);
    sketch2.add(1, 5);
    sketch2.add(3, 15);

    try sketch1.merge(&sketch2);

    // After merge, sketch1 should have combined counts
    const est_1 = sketch1.estimate(1);
    const est_2 = sketch1.estimate(2);
    const est_3 = sketch1.estimate(3);
    try testing.expect(est_1 >= 15); // 10 + 5
    try testing.expect(est_2 >= 20);
    try testing.expect(est_3 >= 15);
    try testing.expectEqual(@as(u64, 50), sketch1.totalCount());
}

test "CountMinSketch - incompatible merge" {
    const Sketch = CountMinSketch(u32, void, defaultHashInt(u32));
    var sketch1 = try Sketch.init(testing.allocator, 4, 100, {});
    defer sketch1.deinit();
    var sketch2 = try Sketch.init(testing.allocator, 5, 100, {}); // Different depth
    defer sketch2.deinit();

    try testing.expectError(error.IncompatibleSketches, sketch1.merge(&sketch2));
}

test "CountMinSketch - one-sided error property" {
    const Sketch = CountMinSketch(u32, void, defaultHashInt(u32));
    var sketch = try Sketch.init(testing.allocator, 5, 100, {});
    defer sketch.deinit();

    const item: u32 = 42;
    const true_count: u64 = 100;
    sketch.add(item, true_count);

    // Add many other items to create potential hash collisions
    for (0..1000) |i| {
        sketch.add(@intCast(i), 1);
    }

    const estimate = sketch.estimate(item);
    // Count-Min Sketch never underestimates
    try testing.expect(estimate >= true_count);
}

test "CountMinSketch - stress test" {
    const Sketch = CountMinSketch(u32, void, defaultHashInt(u32));
    var sketch = try Sketch.initWithErrorBounds(testing.allocator, 0.001, 0.01, {});
    defer sketch.deinit();

    // Add 10k unique items, each with count 1
    for (0..10000) |i| {
        sketch.add(@intCast(i), 1);
    }

    // Verify first 100 items have reasonable estimates
    for (0..100) |i| {
        const estimate = sketch.estimate(@intCast(i));
        try testing.expect(estimate >= 1); // Never underestimates
        // With ε=0.001, error bound is 0.001 * 10000 = 10
        try testing.expect(estimate <= 11); // May occasionally exceed
    }
}

test "CountMinSketch - edge cases" {
    const Sketch = CountMinSketch(u32, void, defaultHashInt(u32));

    // Invalid parameters
    try testing.expectError(error.InvalidDimensions, Sketch.init(testing.allocator, 0, 100, {}));
    try testing.expectError(error.InvalidDimensions, Sketch.init(testing.allocator, 4, 0, {}));
    try testing.expectError(error.InvalidEpsilon, Sketch.initWithErrorBounds(testing.allocator, 0.0, 0.01, {}));
    try testing.expectError(error.InvalidEpsilon, Sketch.initWithErrorBounds(testing.allocator, 1.0, 0.01, {}));
    try testing.expectError(error.InvalidDelta, Sketch.initWithErrorBounds(testing.allocator, 0.01, 0.0, {}));
    try testing.expectError(error.InvalidDelta, Sketch.initWithErrorBounds(testing.allocator, 0.01, 1.0, {}));

    // Minimal sketch
    var sketch = try Sketch.init(testing.allocator, 1, 1, {});
    defer sketch.deinit();

    sketch.add(1, 5);
    try testing.expectEqual(@as(u64, 5), sketch.estimate(1));
}

test "CountMinSketch - estimate error" {
    const Sketch = CountMinSketch(u32, void, defaultHashInt(u32));
    var sketch = try Sketch.init(testing.allocator, 4, 100, {});
    defer sketch.deinit();

    sketch.add(1, 1000);
    try testing.expectEqual(@as(u64, 1000), sketch.totalCount());

    const error_bound = sketch.estimateError(0.01); // ε=0.01
    try testing.expectEqual(@as(u64, 10), error_bound); // 0.01 * 1000 = 10
}
