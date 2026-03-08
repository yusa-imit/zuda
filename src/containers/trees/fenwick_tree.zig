const std = @import("std");
const Allocator = std.mem.Allocator;

/// FenwickTree (Binary Indexed Tree) is a data structure for efficient prefix sum queries
/// and point updates. It can also be adapted for range queries with range updates.
///
/// The tree uses a clever indexing scheme where each index stores a cumulative value
/// for a range determined by the least significant bit (LSB) of the index.
///
/// Time Complexity:
/// - Build: O(n)
/// - Prefix Sum Query: O(log n)
/// - Point Update: O(log n)
/// - Range Query: O(log n)
///
/// Space Complexity: O(n)
///
/// Note: This implementation uses 1-based indexing internally for cleaner bit manipulation.
pub fn FenwickTree(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        tree: []T,
        n: usize,

        /// Initialize a Fenwick tree from a slice of elements.
        /// Time: O(n) | Space: O(n)
        pub fn init(allocator: Allocator, data: []const T) !Self {
            if (data.len == 0) {
                return Self{
                    .allocator = allocator,
                    .tree = &[_]T{},
                    .n = 0,
                };
            }

            const n = data.len;
            // 1-based indexing: tree[0] is unused, indices 1..n are used
            const tree = try allocator.alloc(T, n + 1);
            errdefer allocator.free(tree);

            // Initialize all to zero
            for (tree) |*val| {
                val.* = 0;
            }

            var self = Self{
                .allocator = allocator,
                .tree = tree,
                .n = n,
            };

            // Build tree efficiently in O(n) time
            for (data, 0..) |val, i| {
                try self.add(i, val);
            }

            return self;
        }

        /// Initialize an empty Fenwick tree with size n (all elements are zero).
        /// Time: O(n) | Space: O(n)
        pub fn initZero(allocator: Allocator, n: usize) !Self {
            if (n == 0) {
                return Self{
                    .allocator = allocator,
                    .tree = &[_]T{},
                    .n = 0,
                };
            }

            const tree = try allocator.alloc(T, n + 1);
            errdefer allocator.free(tree);

            for (tree) |*val| {
                val.* = 0;
            }

            return Self{
                .allocator = allocator,
                .tree = tree,
                .n = n,
            };
        }

        /// Free all allocated memory.
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.tree);
            self.* = undefined;
        }

        /// Get the least significant bit value.
        fn lsb(x: usize) usize {
            // x & -x isolates the least significant bit
            const signed_x: isize = @intCast(x);
            const result: isize = signed_x & -signed_x;
            return @intCast(result);
        }

        /// Add delta to element at index idx.
        /// Time: O(log n) | Space: O(1)
        pub fn add(self: *Self, idx: usize, delta: T) !void {
            if (self.n == 0) return error.EmptyTree;
            if (idx >= self.n) return error.IndexOutOfBounds;

            var i = idx + 1; // Convert to 1-based
            while (i <= self.n) {
                self.tree[i] += delta;
                i += lsb(i);
            }
        }

        /// Set element at index idx to value.
        /// Time: O(log n) | Space: O(1)
        pub fn set(self: *Self, idx: usize, value: T) !void {
            if (self.n == 0) return error.EmptyTree;
            if (idx >= self.n) return error.IndexOutOfBounds;

            const old_value = try self.get(idx);
            const delta = value - old_value;
            try self.add(idx, delta);
        }

        /// Get element at index idx.
        /// Time: O(log n) | Space: O(1)
        pub fn get(self: *const Self, idx: usize) !T {
            if (self.n == 0) return error.EmptyTree;
            if (idx >= self.n) return error.IndexOutOfBounds;

            if (idx == 0) {
                return self.prefixSum(0);
            } else {
                return self.prefixSum(idx) - self.prefixSum(idx - 1);
            }
        }

        /// Compute prefix sum from index 0 to idx (inclusive).
        /// Time: O(log n) | Space: O(1)
        pub fn prefixSum(self: *const Self, idx: usize) T {
            if (self.n == 0 or idx >= self.n) {
                return 0;
            }

            var sum: T = 0;
            var i = idx + 1; // Convert to 1-based
            while (i > 0) {
                sum += self.tree[i];
                i -= lsb(i);
            }
            return sum;
        }

        /// Compute range sum from start to end (inclusive).
        /// Time: O(log n) | Space: O(1)
        pub fn rangeSum(self: *const Self, start: usize, end: usize) !T {
            if (self.n == 0) return error.EmptyTree;
            if (start > end) return error.InvalidRange;
            if (end >= self.n) return error.IndexOutOfBounds;

            if (start == 0) {
                return self.prefixSum(end);
            } else {
                return self.prefixSum(end) - self.prefixSum(start - 1);
            }
        }

        /// Get the number of elements.
        pub fn count(self: *const Self) usize {
            return self.n;
        }

        /// Check if the tree is empty.
        pub fn isEmpty(self: *const Self) bool {
            return self.n == 0;
        }

        /// Format the Fenwick tree for debugging.
        pub fn format(
            self: *const Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("FenwickTree(n={})", .{self.n});
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "FenwickTree: basic prefix sum" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4, 5 };
    var tree = try FenwickTree(i32).init(allocator, &data);
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 5), tree.count());
    try testing.expectEqual(false, tree.isEmpty());

    // Prefix sums
    try testing.expectEqual(@as(i32, 1), tree.prefixSum(0)); // 1
    try testing.expectEqual(@as(i32, 3), tree.prefixSum(1)); // 1+2
    try testing.expectEqual(@as(i32, 6), tree.prefixSum(2)); // 1+2+3
    try testing.expectEqual(@as(i32, 10), tree.prefixSum(3)); // 1+2+3+4
    try testing.expectEqual(@as(i32, 15), tree.prefixSum(4)); // 1+2+3+4+5
}

test "FenwickTree: range sum" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4, 5 };
    var tree = try FenwickTree(i32).init(allocator, &data);
    defer tree.deinit();

    // Range [0, 4]: 1+2+3+4+5 = 15
    const sum_all = try tree.rangeSum(0, 4);
    try testing.expectEqual(@as(i32, 15), sum_all);

    // Range [1, 3]: 2+3+4 = 9
    const sum_middle = try tree.rangeSum(1, 3);
    try testing.expectEqual(@as(i32, 9), sum_middle);

    // Range [2, 2]: 3
    const sum_single = try tree.rangeSum(2, 2);
    try testing.expectEqual(@as(i32, 3), sum_single);

    // Range [4, 4]: 5
    const sum_last = try tree.rangeSum(4, 4);
    try testing.expectEqual(@as(i32, 5), sum_last);
}

test "FenwickTree: point update with add" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4, 5 };
    var tree = try FenwickTree(i32).init(allocator, &data);
    defer tree.deinit();

    // Initial prefix sum
    const sum_before = tree.prefixSum(4);
    try testing.expectEqual(@as(i32, 15), sum_before);

    // Add 10 to index 2 (3 -> 13)
    try tree.add(2, 10);

    // New prefix sum [0, 4]: 1+2+13+4+5 = 25
    const sum_after = tree.prefixSum(4);
    try testing.expectEqual(@as(i32, 25), sum_after);

    // Range [2, 2]: 13
    const sum_updated = try tree.rangeSum(2, 2);
    try testing.expectEqual(@as(i32, 13), sum_updated);
}

test "FenwickTree: point update with set" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const data = [_]i64{ 10, 20, 30, 40, 50 };
    var tree = try FenwickTree(i64).init(allocator, &data);
    defer tree.deinit();

    // Get initial value at index 2
    const val_before = try tree.get(2);
    try testing.expectEqual(@as(i64, 30), val_before);

    // Set index 2 to 100
    try tree.set(2, 100);

    // Get updated value
    const val_after = try tree.get(2);
    try testing.expectEqual(@as(i64, 100), val_after);

    // Range sum [0, 4]: 10+20+100+40+50 = 220
    const sum = try tree.rangeSum(0, 4);
    try testing.expectEqual(@as(i64, 220), sum);
}

test "FenwickTree: empty tree" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const data = [_]i32{};
    var tree = try FenwickTree(i32).init(allocator, &data);
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 0), tree.count());
    try testing.expectEqual(true, tree.isEmpty());

    // Prefix sum on empty tree returns 0
    try testing.expectEqual(@as(i32, 0), tree.prefixSum(0));

    // Operations should return errors
    const add_result = tree.add(0, 10);
    try testing.expectError(error.EmptyTree, add_result);

    const range_result = tree.rangeSum(0, 0);
    try testing.expectError(error.EmptyTree, range_result);
}

test "FenwickTree: single element" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const data = [_]i32{42};
    var tree = try FenwickTree(i32).init(allocator, &data);
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 1), tree.count());
    try testing.expectEqual(false, tree.isEmpty());

    const sum = tree.prefixSum(0);
    try testing.expectEqual(@as(i32, 42), sum);

    try tree.add(0, 10);
    const sum_after = tree.prefixSum(0);
    try testing.expectEqual(@as(i32, 52), sum_after);
}

test "FenwickTree: initZero" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var tree = try FenwickTree(i32).initZero(allocator, 5);
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 5), tree.count());

    // All prefix sums should be 0
    for (0..5) |i| {
        try testing.expectEqual(@as(i32, 0), tree.prefixSum(i));
    }

    // Add values
    try tree.add(0, 10);
    try tree.add(2, 20);
    try tree.add(4, 30);

    // Check prefix sums
    try testing.expectEqual(@as(i32, 10), tree.prefixSum(0));
    try testing.expectEqual(@as(i32, 10), tree.prefixSum(1));
    try testing.expectEqual(@as(i32, 30), tree.prefixSum(2));
    try testing.expectEqual(@as(i32, 30), tree.prefixSum(3));
    try testing.expectEqual(@as(i32, 60), tree.prefixSum(4));
}

test "FenwickTree: error handling" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4, 5 };
    var tree = try FenwickTree(i32).init(allocator, &data);
    defer tree.deinit();

    // Out of bounds
    const add_oob = tree.add(10, 5);
    try testing.expectError(error.IndexOutOfBounds, add_oob);

    const set_oob = tree.set(10, 5);
    try testing.expectError(error.IndexOutOfBounds, set_oob);

    const get_oob = tree.get(10);
    try testing.expectError(error.IndexOutOfBounds, get_oob);

    const range_oob = tree.rangeSum(0, 10);
    try testing.expectError(error.IndexOutOfBounds, range_oob);

    // Invalid range (start > end)
    const invalid_range = tree.rangeSum(3, 2);
    try testing.expectError(error.InvalidRange, invalid_range);
}

test "FenwickTree: stress test with 1000 elements" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var data: [1000]i64 = undefined;
    for (&data, 0..) |*val, i| {
        val.* = @intCast(i + 1);
    }

    var tree = try FenwickTree(i64).init(allocator, &data);
    defer tree.deinit();

    // Sum of 1..1000 = 1000*1001/2 = 500500
    const sum_all = tree.prefixSum(999);
    try testing.expectEqual(@as(i64, 500500), sum_all);

    // Sum of 1..500 = 500*501/2 = 125250
    const sum_half = tree.prefixSum(499);
    try testing.expectEqual(@as(i64, 125250), sum_half);

    // Range [500, 999]: 500500 - 125250 = 375250
    const sum_upper = try tree.rangeSum(500, 999);
    try testing.expectEqual(@as(i64, 375250), sum_upper);

    // Update several elements
    try tree.add(0, 999); // 1 -> 1000
    try tree.add(499, 1500); // 500 -> 2000

    // New sum [0, 499]: 125250 + 999 + 1500 = 127749
    const sum_after = tree.prefixSum(499);
    try testing.expectEqual(@as(i64, 127749), sum_after);
}

test "FenwickTree: multiple updates and queries" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var tree = try FenwickTree(i32).initZero(allocator, 10);
    defer tree.deinit();

    // Build array incrementally
    for (0..10) |i| {
        try tree.add(i, @intCast(i + 1));
    }

    // Verify: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    const sum = try tree.rangeSum(0, 9);
    try testing.expectEqual(@as(i32, 55), sum);

    // Update multiple positions
    try tree.set(0, 10); // 1 -> 10 (+9)
    try tree.set(5, 1); // 6 -> 1 (-5)
    try tree.set(9, 20); // 10 -> 20 (+10)

    // New sum: 55 + 9 - 5 + 10 = 69
    const sum_after = try tree.rangeSum(0, 9);
    try testing.expectEqual(@as(i32, 69), sum_after);

    // Verify individual gets
    const val0 = try tree.get(0);
    try testing.expectEqual(@as(i32, 10), val0);

    const val5 = try tree.get(5);
    try testing.expectEqual(@as(i32, 1), val5);

    const val9 = try tree.get(9);
    try testing.expectEqual(@as(i32, 20), val9);
}

test "FenwickTree: negative values" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const data = [_]i32{ -5, 10, -3, 7, -2 };
    var tree = try FenwickTree(i32).init(allocator, &data);
    defer tree.deinit();

    // Sum [0, 4]: -5+10-3+7-2 = 7
    const sum = try tree.rangeSum(0, 4);
    try testing.expectEqual(@as(i32, 7), sum);

    // Add -10 to index 1 (10 -> 0)
    try tree.add(1, -10);

    // New sum: 7 - 10 = -3
    const sum_after = try tree.rangeSum(0, 4);
    try testing.expectEqual(@as(i32, -3), sum_after);
}
