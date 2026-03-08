const std = @import("std");
const Allocator = std.mem.Allocator;

/// LazySegmentTree is a segment tree with lazy propagation support.
/// It allows efficient range updates in addition to range queries.
/// Lazy propagation defers updates until necessary, maintaining O(log n) complexity.
///
/// Time Complexity:
/// - Build: O(n)
/// - Range Query: O(log n)
/// - Range Update: O(log n)
/// - Point Query: O(log n)
///
/// Space Complexity: O(n)
pub fn LazySegmentTree(
    comptime T: type,
    comptime Context: type,
    comptime combineFn: fn (ctx: Context, a: T, b: T) T,
    comptime applyFn: fn (ctx: Context, value: T, lazy: T, range_size: usize) T,
    comptime composeFn: fn (ctx: Context, old_lazy: T, new_lazy: T) T,
) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        tree: []T,
        lazy: []T,
        n: usize,
        ctx: Context,
        default_value: T,

        /// Initialize a lazy segment tree from a slice of elements.
        /// default_value: identity element for combineFn (e.g., 0 for sum, max_int for min)
        /// Time: O(n) | Space: O(n)
        pub fn init(allocator: Allocator, data: []const T, ctx: Context, default_value: T) !Self {
            if (data.len == 0) {
                return Self{
                    .allocator = allocator,
                    .tree = &[_]T{},
                    .lazy = &[_]T{},
                    .n = 0,
                    .ctx = ctx,
                    .default_value = default_value,
                };
            }

            const n = data.len;
            const tree_size = 4 * n;
            const tree = try allocator.alloc(T, tree_size);
            errdefer allocator.free(tree);

            const lazy = try allocator.alloc(T, tree_size);
            errdefer allocator.free(lazy);

            // Initialize lazy array with default values
            for (lazy) |*val| {
                val.* = default_value;
            }

            var self = Self{
                .allocator = allocator,
                .tree = tree,
                .lazy = lazy,
                .n = n,
                .ctx = ctx,
                .default_value = default_value,
            };

            self.build(data, 0, 0, n - 1);
            return self;
        }

        /// Free all allocated memory.
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.tree);
            self.allocator.free(self.lazy);
            self.* = undefined;
        }

        fn build(self: *Self, data: []const T, node: usize, start: usize, end: usize) void {
            if (start == end) {
                self.tree[node] = data[start];
            } else {
                const mid = start + (end - start) / 2;
                const left_child = 2 * node + 1;
                const right_child = 2 * node + 2;

                self.build(data, left_child, start, mid);
                self.build(data, right_child, mid + 1, end);

                self.tree[node] = combineFn(self.ctx, self.tree[left_child], self.tree[right_child]);
            }
        }

        /// Push down lazy value to children.
        fn push(self: *Self, node: usize, start: usize, end: usize) void {
            if (std.meta.eql(self.lazy[node], self.default_value)) {
                return; // No pending update
            }

            const range_size = end - start + 1;
            self.tree[node] = applyFn(self.ctx, self.tree[node], self.lazy[node], range_size);

            if (start != end) {
                const left_child = 2 * node + 1;
                const right_child = 2 * node + 2;
                self.lazy[left_child] = composeFn(self.ctx, self.lazy[left_child], self.lazy[node]);
                self.lazy[right_child] = composeFn(self.ctx, self.lazy[right_child], self.lazy[node]);
            }

            self.lazy[node] = self.default_value;
        }

        /// Query the result of combining elements in range [query_start, query_end].
        /// Time: O(log n) | Space: O(1)
        pub fn query(self: *Self, query_start: usize, query_end: usize) !T {
            if (self.n == 0) return error.EmptyTree;
            if (query_start > query_end) return error.InvalidRange;
            if (query_end >= self.n) return error.IndexOutOfBounds;

            return self.queryRecursive(0, 0, self.n - 1, query_start, query_end);
        }

        fn queryRecursive(self: *Self, node: usize, node_start: usize, node_end: usize, query_start: usize, query_end: usize) T {
            self.push(node, node_start, node_end);

            // Complete overlap
            if (query_start <= node_start and node_end <= query_end) {
                return self.tree[node];
            }

            const mid = node_start + (node_end - node_start) / 2;
            const left_child = 2 * node + 1;
            const right_child = 2 * node + 2;

            // Partial overlap
            if (query_end <= mid) {
                return self.queryRecursive(left_child, node_start, mid, query_start, query_end);
            } else if (query_start > mid) {
                return self.queryRecursive(right_child, mid + 1, node_end, query_start, query_end);
            } else {
                const left_result = self.queryRecursive(left_child, node_start, mid, query_start, query_end);
                const right_result = self.queryRecursive(right_child, mid + 1, node_end, query_start, query_end);
                return combineFn(self.ctx, left_result, right_result);
            }
        }

        /// Update a range [update_start, update_end] with lazy value.
        /// Time: O(log n) | Space: O(1)
        pub fn updateRange(self: *Self, update_start: usize, update_end: usize, lazy_value: T) !void {
            if (self.n == 0) return error.EmptyTree;
            if (update_start > update_end) return error.InvalidRange;
            if (update_end >= self.n) return error.IndexOutOfBounds;

            self.updateRangeRecursive(0, 0, self.n - 1, update_start, update_end, lazy_value);
        }

        fn updateRangeRecursive(self: *Self, node: usize, node_start: usize, node_end: usize, update_start: usize, update_end: usize, lazy_value: T) void {
            self.push(node, node_start, node_end);

            // No overlap
            if (update_end < node_start or update_start > node_end) {
                return;
            }

            // Complete overlap
            if (update_start <= node_start and node_end <= update_end) {
                self.lazy[node] = composeFn(self.ctx, self.lazy[node], lazy_value);
                self.push(node, node_start, node_end);
                return;
            }

            // Partial overlap
            const mid = node_start + (node_end - node_start) / 2;
            const left_child = 2 * node + 1;
            const right_child = 2 * node + 2;

            self.updateRangeRecursive(left_child, node_start, mid, update_start, update_end, lazy_value);
            self.updateRangeRecursive(right_child, mid + 1, node_end, update_start, update_end, lazy_value);

            // Recompute parent
            self.push(left_child, node_start, mid);
            self.push(right_child, mid + 1, node_end);
            self.tree[node] = combineFn(self.ctx, self.tree[left_child], self.tree[right_child]);
        }

        /// Get the number of elements in the original data.
        pub fn count(self: *const Self) usize {
            return self.n;
        }

        /// Check if the tree is empty.
        pub fn isEmpty(self: *const Self) bool {
            return self.n == 0;
        }

        /// Format the lazy segment tree for debugging.
        pub fn format(
            self: *const Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("LazySegmentTree(n={}, tree_size={})", .{ self.n, self.tree.len });
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "LazySegmentTree: range sum with range add updates" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const SumContext = struct {};
    const combineFn = struct {
        fn f(_: SumContext, a: i64, b: i64) i64 {
            return a + b;
        }
    }.f;
    const applyFn = struct {
        fn f(_: SumContext, value: i64, lazy: i64, range_size: usize) i64 {
            return value + lazy * @as(i64, @intCast(range_size));
        }
    }.f;
    const composeFn = struct {
        fn f(_: SumContext, old_lazy: i64, new_lazy: i64) i64 {
            return old_lazy + new_lazy;
        }
    }.f;

    const Tree = LazySegmentTree(i64, SumContext, combineFn, applyFn, composeFn);

    const data = [_]i64{ 1, 2, 3, 4, 5 };
    var tree = try Tree.init(allocator, &data, .{}, 0);
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 5), tree.count());
    try testing.expectEqual(false, tree.isEmpty());

    // Initial sum [0, 4]: 1+2+3+4+5 = 15
    const sum_before = try tree.query(0, 4);
    try testing.expectEqual(@as(i64, 15), sum_before);

    // Add 10 to range [1, 3]: [1, 12, 13, 14, 5]
    try tree.updateRange(1, 3, 10);

    // New sum [0, 4]: 1+12+13+14+5 = 45
    const sum_after = try tree.query(0, 4);
    try testing.expectEqual(@as(i64, 45), sum_after);

    // Query [1, 3]: 12+13+14 = 39
    const sum_middle = try tree.query(1, 3);
    try testing.expectEqual(@as(i64, 39), sum_middle);
}

test "LazySegmentTree: range min with range set updates" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const MinContext = struct {};
    const combineFn = struct {
        fn f(_: MinContext, a: i32, b: i32) i32 {
            return @min(a, b);
        }
    }.f;
    const applyFn = struct {
        fn f(_: MinContext, value: i32, lazy: i32, range_size: usize) i32 {
            _ = value;
            _ = range_size;
            return lazy; // Set operation
        }
    }.f;
    const composeFn = struct {
        fn f(_: MinContext, old_lazy: i32, new_lazy: i32) i32 {
            _ = old_lazy;
            return new_lazy; // Later set overwrites
        }
    }.f;

    const Tree = LazySegmentTree(i32, MinContext, combineFn, applyFn, composeFn);

    const data = [_]i32{ 5, 2, 8, 1, 9, 3 };
    var tree = try Tree.init(allocator, &data, .{}, std.math.maxInt(i32));
    defer tree.deinit();

    // Initial min [0, 5]: 1
    const min_before = try tree.query(0, 5);
    try testing.expectEqual(@as(i32, 1), min_before);

    // Set range [2, 4] to 10: [5, 2, 10, 10, 10, 3]
    try tree.updateRange(2, 4, 10);

    // New min [0, 5]: 2
    const min_after = try tree.query(0, 5);
    try testing.expectEqual(@as(i32, 2), min_after);

    // Min [2, 4]: 10
    const min_middle = try tree.query(2, 4);
    try testing.expectEqual(@as(i32, 10), min_middle);
}

test "LazySegmentTree: multiple range updates" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const SumContext = struct {};
    const combineFn = struct {
        fn f(_: SumContext, a: i64, b: i64) i64 {
            return a + b;
        }
    }.f;
    const applyFn = struct {
        fn f(_: SumContext, value: i64, lazy: i64, range_size: usize) i64 {
            return value + lazy * @as(i64, @intCast(range_size));
        }
    }.f;
    const composeFn = struct {
        fn f(_: SumContext, old_lazy: i64, new_lazy: i64) i64 {
            return old_lazy + new_lazy;
        }
    }.f;

    const Tree = LazySegmentTree(i64, SumContext, combineFn, applyFn, composeFn);

    const data = [_]i64{ 1, 1, 1, 1, 1 };
    var tree = try Tree.init(allocator, &data, .{}, 0);
    defer tree.deinit();

    // Add 5 to [0, 2]: [6, 6, 6, 1, 1]
    try tree.updateRange(0, 2, 5);

    // Add 3 to [2, 4]: [6, 6, 9, 4, 4]
    try tree.updateRange(2, 4, 3);

    // Query [0, 4]: 6+6+9+4+4 = 29
    const sum = try tree.query(0, 4);
    try testing.expectEqual(@as(i64, 29), sum);

    // Query [2, 2]: 9
    const sum_overlap = try tree.query(2, 2);
    try testing.expectEqual(@as(i64, 9), sum_overlap);
}

test "LazySegmentTree: empty tree" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const SumContext = struct {};
    const combineFn = struct {
        fn f(_: SumContext, a: i64, b: i64) i64 {
            return a + b;
        }
    }.f;
    const applyFn = struct {
        fn f(_: SumContext, value: i64, lazy: i64, range_size: usize) i64 {
            return value + lazy * @as(i64, @intCast(range_size));
        }
    }.f;
    const composeFn = struct {
        fn f(_: SumContext, old_lazy: i64, new_lazy: i64) i64 {
            return old_lazy + new_lazy;
        }
    }.f;

    const Tree = LazySegmentTree(i64, SumContext, combineFn, applyFn, composeFn);

    const data = [_]i64{};
    var tree = try Tree.init(allocator, &data, .{}, 0);
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 0), tree.count());
    try testing.expectEqual(true, tree.isEmpty());

    const query_result = tree.query(0, 0);
    try testing.expectError(error.EmptyTree, query_result);

    const update_result = tree.updateRange(0, 0, 42);
    try testing.expectError(error.EmptyTree, update_result);
}

test "LazySegmentTree: single element" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const SumContext = struct {};
    const combineFn = struct {
        fn f(_: SumContext, a: i64, b: i64) i64 {
            return a + b;
        }
    }.f;
    const applyFn = struct {
        fn f(_: SumContext, value: i64, lazy: i64, range_size: usize) i64 {
            return value + lazy * @as(i64, @intCast(range_size));
        }
    }.f;
    const composeFn = struct {
        fn f(_: SumContext, old_lazy: i64, new_lazy: i64) i64 {
            return old_lazy + new_lazy;
        }
    }.f;

    const Tree = LazySegmentTree(i64, SumContext, combineFn, applyFn, composeFn);

    const data = [_]i64{42};
    var tree = try Tree.init(allocator, &data, .{}, 0);
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 1), tree.count());

    const sum_before = try tree.query(0, 0);
    try testing.expectEqual(@as(i64, 42), sum_before);

    try tree.updateRange(0, 0, 10);
    const sum_after = try tree.query(0, 0);
    try testing.expectEqual(@as(i64, 52), sum_after);
}

test "LazySegmentTree: error handling" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const SumContext = struct {};
    const combineFn = struct {
        fn f(_: SumContext, a: i64, b: i64) i64 {
            return a + b;
        }
    }.f;
    const applyFn = struct {
        fn f(_: SumContext, value: i64, lazy: i64, range_size: usize) i64 {
            return value + lazy * @as(i64, @intCast(range_size));
        }
    }.f;
    const composeFn = struct {
        fn f(_: SumContext, old_lazy: i64, new_lazy: i64) i64 {
            return old_lazy + new_lazy;
        }
    }.f;

    const Tree = LazySegmentTree(i64, SumContext, combineFn, applyFn, composeFn);

    const data = [_]i64{ 1, 2, 3, 4, 5 };
    var tree = try Tree.init(allocator, &data, .{}, 0);
    defer tree.deinit();

    // Invalid range (start > end)
    const invalid_query = tree.query(3, 2);
    try testing.expectError(error.InvalidRange, invalid_query);

    const invalid_update = tree.updateRange(3, 2, 10);
    try testing.expectError(error.InvalidRange, invalid_update);

    // Out of bounds
    const oob_query = tree.query(0, 10);
    try testing.expectError(error.IndexOutOfBounds, oob_query);

    const oob_update = tree.updateRange(0, 10, 5);
    try testing.expectError(error.IndexOutOfBounds, oob_update);
}

test "LazySegmentTree: overlapping range updates" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const SumContext = struct {};
    const combineFn = struct {
        fn f(_: SumContext, a: i64, b: i64) i64 {
            return a + b;
        }
    }.f;
    const applyFn = struct {
        fn f(_: SumContext, value: i64, lazy: i64, range_size: usize) i64 {
            return value + lazy * @as(i64, @intCast(range_size));
        }
    }.f;
    const composeFn = struct {
        fn f(_: SumContext, old_lazy: i64, new_lazy: i64) i64 {
            return old_lazy + new_lazy;
        }
    }.f;

    const Tree = LazySegmentTree(i64, SumContext, combineFn, applyFn, composeFn);

    var data: [10]i64 = undefined;
    for (&data, 0..) |*val, i| {
        val.* = @intCast(i + 1);
    }

    var tree = try Tree.init(allocator, &data, .{}, 0);
    defer tree.deinit();

    // Sum [0, 9]: 1+2+...+10 = 55
    const sum_before = try tree.query(0, 9);
    try testing.expectEqual(@as(i64, 55), sum_before);

    // Add 10 to [0, 4]: [11, 12, 13, 14, 15, 6, 7, 8, 9, 10]
    try tree.updateRange(0, 4, 10);

    // Add 5 to [3, 7]: [11, 12, 13, 19, 20, 11, 12, 13, 9, 10]
    try tree.updateRange(3, 7, 5);

    // Sum [0, 9]: 11+12+13+19+20+11+12+13+9+10 = 130
    const sum_after = try tree.query(0, 9);
    try testing.expectEqual(@as(i64, 130), sum_after);

    // Sum [3, 7]: 19+20+11+12+13 = 75
    const sum_overlap = try tree.query(3, 7);
    try testing.expectEqual(@as(i64, 75), sum_overlap);
}

test "LazySegmentTree: stress test with 1000 elements" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const SumContext = struct {};
    const combineFn = struct {
        fn f(_: SumContext, a: i64, b: i64) i64 {
            return a + b;
        }
    }.f;
    const applyFn = struct {
        fn f(_: SumContext, value: i64, lazy: i64, range_size: usize) i64 {
            return value + lazy * @as(i64, @intCast(range_size));
        }
    }.f;
    const composeFn = struct {
        fn f(_: SumContext, old_lazy: i64, new_lazy: i64) i64 {
            return old_lazy + new_lazy;
        }
    }.f;

    const Tree = LazySegmentTree(i64, SumContext, combineFn, applyFn, composeFn);

    var data: [1000]i64 = undefined;
    for (&data) |*val| {
        val.* = 1;
    }

    var tree = try Tree.init(allocator, &data, .{}, 0);
    defer tree.deinit();

    // Initial sum: 1000
    const sum_before = try tree.query(0, 999);
    try testing.expectEqual(@as(i64, 1000), sum_before);

    // Add 5 to all elements
    try tree.updateRange(0, 999, 5);
    const sum_after = try tree.query(0, 999);
    try testing.expectEqual(@as(i64, 6000), sum_after);

    // Add 10 to first half
    try tree.updateRange(0, 499, 10);
    const sum_half = try tree.query(0, 499);
    try testing.expectEqual(@as(i64, 8000), sum_half); // 500 * 16

    // Total sum: 500*16 + 500*6 = 8000 + 3000 = 11000
    const sum_total = try tree.query(0, 999);
    try testing.expectEqual(@as(i64, 11000), sum_total);
}

test "LazySegmentTree: range max with range add" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const MaxContext = struct {};
    const combineFn = struct {
        fn f(_: MaxContext, a: i32, b: i32) i32 {
            return @max(a, b);
        }
    }.f;
    const applyFn = struct {
        fn f(_: MaxContext, value: i32, lazy: i32, range_size: usize) i32 {
            _ = range_size;
            return value + lazy;
        }
    }.f;
    const composeFn = struct {
        fn f(_: MaxContext, old_lazy: i32, new_lazy: i32) i32 {
            return old_lazy + new_lazy;
        }
    }.f;

    const Tree = LazySegmentTree(i32, MaxContext, combineFn, applyFn, composeFn);

    const data = [_]i32{ 1, 5, 3, 2, 4 };
    var tree = try Tree.init(allocator, &data, .{}, 0);
    defer tree.deinit();

    // Max [0, 4]: 5
    const max_before = try tree.query(0, 4);
    try testing.expectEqual(@as(i32, 5), max_before);

    // Add 10 to [2, 4]: [1, 5, 13, 12, 14]
    try tree.updateRange(2, 4, 10);

    // Max [0, 4]: 14
    const max_after = try tree.query(0, 4);
    try testing.expectEqual(@as(i32, 14), max_after);

    // Max [0, 1]: 5
    const max_left = try tree.query(0, 1);
    try testing.expectEqual(@as(i32, 5), max_left);
}
