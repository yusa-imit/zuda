//! Count-Min Sketch - Frequency estimation with sublinear space
//!
//! Data structure for approximate frequency counting in streams.
//! Provides probabilistic guarantees: count(x) >= true_count(x) with high probability.
//!
//! Parameters:
//! - width (w): Determines accuracy (ε = e/w where e is Euler's number)
//! - depth (d): Determines confidence (δ = 1/e^d probability of large error)
//!
//! Typical usage: w=⌈e/ε⌉, d=⌈ln(1/δ)⌉
//! Example: ε=0.01, δ=0.01 → w≈272, d≈5
//!
//! Use cases:
//! - Network traffic monitoring (packet frequency)
//! - Database query optimization (selectivity estimation)
//! - Real-time analytics (event counting)
//! - Anomaly detection (frequency-based outliers)

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Count-Min Sketch for frequency estimation
///
/// Time: O(d) for update/query where d is depth
/// Space: O(w×d) for w width and d depth
pub fn CountMinSketch(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        width: usize,
        depth: usize,
        table: [][]u64,
        seeds: []u64, // Random seeds for hash functions

        /// Initialize Count-Min Sketch with given dimensions
        ///
        /// Time: O(w×d)
        /// Space: O(w×d)
        pub fn init(allocator: Allocator, width: usize, depth: usize) !Self {
            if (width == 0 or depth == 0) return error.InvalidDimensions;

            var table = try allocator.alloc([]u64, depth);
            errdefer allocator.free(table);

            var initialized: usize = 0;
            errdefer for (0..initialized) |i| {
                allocator.free(table[i]);
            };

            for (0..depth) |i| {
                table[i] = try allocator.alloc(u64, width);
                @memset(table[i], 0);
                initialized += 1;
            }

            var seeds = try allocator.alloc(u64, depth);
            errdefer allocator.free(seeds);

            // Initialize with deterministic but different seeds
            for (0..depth) |i| {
                seeds[i] = @as(u64, @intCast(i)) * 0x9e3779b97f4a7c15; // Golden ratio hash
            }

            return Self{
                .allocator = allocator,
                .width = width,
                .depth = depth,
                .table = table,
                .seeds = seeds,
            };
        }

        /// Clean up resources
        pub fn deinit(self: *Self) void {
            for (self.table) |row| {
                self.allocator.free(row);
            }
            self.allocator.free(self.table);
            self.allocator.free(self.seeds);
        }

        /// Hash function for row i
        fn hash(self: Self, item: T, row: usize) usize {
            const h = std.hash.Wyhash.hash(self.seeds[row], std.mem.asBytes(&item));
            return @as(usize, @intCast(h % self.width));
        }

        /// Update count for an item
        ///
        /// Time: O(d)
        /// Space: O(1)
        pub fn update(self: *Self, item: T, count: u64) void {
            for (0..self.depth) |i| {
                const pos = self.hash(item, i);
                self.table[i][pos] += count;
            }
        }

        /// Estimate count for an item (conservative estimate)
        ///
        /// Time: O(d)
        /// Space: O(1)
        pub fn estimate(self: Self, item: T) u64 {
            var min_count: u64 = std.math.maxInt(u64);
            for (0..self.depth) |i| {
                const pos = self.hash(item, i);
                const count = self.table[i][pos];
                min_count = @min(min_count, count);
            }
            return min_count;
        }

        /// Reset all counts to zero
        ///
        /// Time: O(w×d)
        /// Space: O(1)
        pub fn clear(self: *Self) void {
            for (self.table) |row| {
                @memset(row, 0);
            }
        }

        /// Merge another sketch into this one (element-wise addition)
        /// Both sketches must have same dimensions
        ///
        /// Time: O(w×d)
        /// Space: O(1)
        pub fn merge(self: *Self, other: Self) error{DimensionMismatch}!void {
            if (self.width != other.width or self.depth != other.depth) {
                return error.DimensionMismatch;
            }

            for (0..self.depth) |i| {
                for (0..self.width) |j| {
                    self.table[i][j] += other.table[i][j];
                }
            }
        }
    };
}

// Tests
test "CountMinSketch: basic operations" {
    const CMS = CountMinSketch(u32);
    var sketch = try CMS.init(std.testing.allocator, 100, 5);
    defer sketch.deinit();

    // Initially all counts should be 0
    try std.testing.expectEqual(@as(u64, 0), sketch.estimate(42));
    try std.testing.expectEqual(@as(u64, 0), sketch.estimate(100));

    // Update counts
    sketch.update(42, 10);
    sketch.update(100, 5);
    sketch.update(42, 3); // Additional updates

    // Estimates should be at least the true count
    const est42 = sketch.estimate(42);
    const est100 = sketch.estimate(100);
    try std.testing.expect(est42 >= 13); // 10 + 3
    try std.testing.expect(est100 >= 5);
}

test "CountMinSketch: frequency estimation" {
    const CMS = CountMinSketch([]const u8);
    var sketch = try CMS.init(std.testing.allocator, 200, 5);
    defer sketch.deinit();

    const items = [_][]const u8{ "apple", "banana", "apple", "cherry", "apple", "banana" };
    // True counts: apple=3, banana=2, cherry=1

    for (items) |item| {
        sketch.update(item, 1);
    }

    const apple_est = sketch.estimate("apple");
    const banana_est = sketch.estimate("banana");
    const cherry_est = sketch.estimate("cherry");
    const unknown_est = sketch.estimate("unknown");

    try std.testing.expect(apple_est >= 3);
    try std.testing.expect(banana_est >= 2);
    try std.testing.expect(cherry_est >= 1);
    try std.testing.expectEqual(@as(u64, 0), unknown_est);
}

test "CountMinSketch: clear operation" {
    const CMS = CountMinSketch(u32);
    var sketch = try CMS.init(std.testing.allocator, 100, 5);
    defer sketch.deinit();

    sketch.update(42, 100);
    try std.testing.expect(sketch.estimate(42) >= 100);

    sketch.clear();
    try std.testing.expectEqual(@as(u64, 0), sketch.estimate(42));
}

test "CountMinSketch: merge operation" {
    const CMS = CountMinSketch(u32);
    var sketch1 = try CMS.init(std.testing.allocator, 100, 5);
    defer sketch1.deinit();
    var sketch2 = try CMS.init(std.testing.allocator, 100, 5);
    defer sketch2.deinit();

    sketch1.update(42, 10);
    sketch2.update(42, 5);
    sketch2.update(100, 3);

    try sketch1.merge(sketch2);

    const est42 = sketch1.estimate(42);
    const est100 = sketch1.estimate(100);
    try std.testing.expect(est42 >= 15); // 10 + 5
    try std.testing.expect(est100 >= 3);
}

test "CountMinSketch: dimension mismatch error" {
    const CMS = CountMinSketch(u32);
    var sketch1 = try CMS.init(std.testing.allocator, 100, 5);
    defer sketch1.deinit();
    var sketch2 = try CMS.init(std.testing.allocator, 200, 5);
    defer sketch2.deinit();

    try std.testing.expectError(error.DimensionMismatch, sketch1.merge(sketch2));
}

test "CountMinSketch: invalid dimensions" {
    const CMS = CountMinSketch(u32);
    try std.testing.expectError(error.InvalidDimensions, CMS.init(std.testing.allocator, 0, 5));
    try std.testing.expectError(error.InvalidDimensions, CMS.init(std.testing.allocator, 100, 0));
}

test "CountMinSketch: stress test with many items" {
    const CMS = CountMinSketch(u64);
    var sketch = try CMS.init(std.testing.allocator, 1000, 5);
    defer sketch.deinit();

    // Insert 10000 items with known frequencies
    var i: u64 = 0;
    while (i < 10000) : (i += 1) {
        const item = i % 100; // 100 unique items, each appears 100 times
        sketch.update(item, 1);
    }

    // Verify estimates
    i = 0;
    while (i < 100) : (i += 1) {
        const est = sketch.estimate(i);
        try std.testing.expect(est >= 100);
        // With good parameters, overestimation should be small
        try std.testing.expect(est <= 200); // Within 2x
    }
}

test "CountMinSketch: memory safety" {
    const CMS = CountMinSketch(u32);
    var sketch = try CMS.init(std.testing.allocator, 100, 5);
    defer sketch.deinit();

    var i: u32 = 0;
    while (i < 1000) : (i += 1) {
        sketch.update(i, 1);
    }

    i = 0;
    while (i < 1000) : (i += 1) {
        _ = sketch.estimate(i);
    }
}
