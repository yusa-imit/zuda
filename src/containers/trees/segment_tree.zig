const std = @import("std");
const Allocator = std.mem.Allocator;

/// SegmentTree is a binary tree used for storing intervals or segments.
/// It allows querying which segments contain a given point efficiently.
/// Commonly used for range query problems with static data.
///
/// Time Complexity:
/// - Build: O(n)
/// - Query: O(log n + k) where k is the number of matches
/// - Point Update: O(log n)
/// - Range Update: Not supported (use LazySegmentTree for that)
///
/// Space Complexity: O(n)
pub fn SegmentTree(comptime T: type, comptime Context: type, comptime combineFn: fn (ctx: Context, a: T, b: T) T) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        tree: []T,
        n: usize,
        ctx: Context,

        /// Initialize a segment tree from a slice of elements.
        /// Time: O(n) | Space: O(n)
        pub fn init(allocator: Allocator, data: []const T, ctx: Context) !Self {
            if (data.len == 0) {
                return Self{
                    .allocator = allocator,
                    .tree = &[_]T{},
                    .n = 0,
                    .ctx = ctx,
                };
            }

            const n = data.len;
            const tree_size = 4 * n; // Upper bound for tree size
            const tree = try allocator.alloc(T, tree_size);
            errdefer allocator.free(tree);

            var self = Self{
                .allocator = allocator,
                .tree = tree,
                .n = n,
                .ctx = ctx,
            };

            self.build(data, 0, 0, n - 1);
            return self;
        }

        /// Free all allocated memory.
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.tree);
            self.* = undefined;
        }

        /// Build the segment tree recursively.
        /// node: current node index in tree
        /// start, end: range [start, end] in the original data
        fn build(self: *Self, data: []const T, node: usize, start: usize, end: usize) void {
            if (start == end) {
                // Leaf node
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

        /// Query the result of combining elements in range [query_start, query_end].
        /// Time: O(log n) | Space: O(1)
        pub fn query(self: *const Self, query_start: usize, query_end: usize) !T {
            if (self.n == 0) return error.EmptyTree;
            if (query_start > query_end) return error.InvalidRange;
            if (query_end >= self.n) return error.IndexOutOfBounds;

            return self.queryRecursive(0, 0, self.n - 1, query_start, query_end);
        }

        fn queryRecursive(self: *const Self, node: usize, node_start: usize, node_end: usize, query_start: usize, query_end: usize) T {
            // Complete overlap
            if (query_start <= node_start and node_end <= query_end) {
                return self.tree[node];
            }

            const mid = node_start + (node_end - node_start) / 2;
            const left_child = 2 * node + 1;
            const right_child = 2 * node + 2;

            // Partial overlap
            if (query_end <= mid) {
                // Query is entirely in left subtree
                return self.queryRecursive(left_child, node_start, mid, query_start, query_end);
            } else if (query_start > mid) {
                // Query is entirely in right subtree
                return self.queryRecursive(right_child, mid + 1, node_end, query_start, query_end);
            } else {
                // Query spans both subtrees
                const left_result = self.queryRecursive(left_child, node_start, mid, query_start, query_end);
                const right_result = self.queryRecursive(right_child, mid + 1, node_end, query_start, query_end);
                return combineFn(self.ctx, left_result, right_result);
            }
        }

        /// Update a single element at index idx to value.
        /// Time: O(log n) | Space: O(1)
        pub fn update(self: *Self, idx: usize, value: T) !void {
            if (self.n == 0) return error.EmptyTree;
            if (idx >= self.n) return error.IndexOutOfBounds;

            self.updateRecursive(0, 0, self.n - 1, idx, value);
        }

        fn updateRecursive(self: *Self, node: usize, start: usize, end: usize, idx: usize, value: T) void {
            if (start == end) {
                // Leaf node
                self.tree[node] = value;
            } else {
                const mid = start + (end - start) / 2;
                const left_child = 2 * node + 1;
                const right_child = 2 * node + 2;

                if (idx <= mid) {
                    self.updateRecursive(left_child, start, mid, idx, value);
                } else {
                    self.updateRecursive(right_child, mid + 1, end, idx, value);
                }

                self.tree[node] = combineFn(self.ctx, self.tree[left_child], self.tree[right_child]);
            }
        }

        /// Get the number of elements in the original data.
        pub fn count(self: *const Self) usize {
            return self.n;
        }

        /// Check if the tree is empty.
        pub fn isEmpty(self: *const Self) bool {
            return self.n == 0;
        }

        /// Validate the segment tree structure (checks internal consistency).
        /// Time: O(n) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            if (self.n == 0) return;
            try self.validateRecursive(0, 0, self.n - 1);
        }

        fn validateRecursive(self: *const Self, node: usize, start: usize, end: usize) !void {
            if (start > end) return error.TreeInvariant;

            if (start == end) {
                // Leaf node - no further validation needed
                return;
            }

            const mid = start + (end - start) / 2;
            const left_child = 2 * node + 1;
            const right_child = 2 * node + 2;

            if (left_child >= self.tree.len or right_child >= self.tree.len) {
                return error.TreeInvariant;
            }

            // Validate that parent is combination of children
            const expected = combineFn(self.ctx, self.tree[left_child], self.tree[right_child]);
            if (!std.meta.eql(self.tree[node], expected)) {
                return error.TreeInvariant;
            }

            try self.validateRecursive(left_child, start, mid);
            try self.validateRecursive(right_child, mid + 1, end);
        }

        /// Format the segment tree for debugging.
        pub fn format(
            self: *const Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("SegmentTree(n={}, tree_size={})", .{ self.n, self.tree.len });
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "SegmentTree: basic sum query" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const SumContext = struct {};
    const sumFn = struct {
        fn f(_: SumContext, a: i32, b: i32) i32 {
            return a + b;
        }
    }.f;

    const data = [_]i32{ 1, 3, 5, 7, 9, 11 };
    var tree = try SegmentTree(i32, SumContext, sumFn).init(allocator, &data, .{});
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 6), tree.count());
    try testing.expectEqual(false, tree.isEmpty());

    // Query entire range: 1+3+5+7+9+11 = 36
    const sum_all = try tree.query(0, 5);
    try testing.expectEqual(@as(i32, 36), sum_all);

    // Query [1, 3]: 3+5+7 = 15
    const sum_middle = try tree.query(1, 3);
    try testing.expectEqual(@as(i32, 15), sum_middle);

    // Query single element [2, 2]: 5
    const sum_single = try tree.query(2, 2);
    try testing.expectEqual(@as(i32, 5), sum_single);

    try tree.validate();
}

test "SegmentTree: min query" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const MinContext = struct {};
    const minFn = struct {
        fn f(_: MinContext, a: i32, b: i32) i32 {
            return @min(a, b);
        }
    }.f;

    const data = [_]i32{ 5, 2, 8, 1, 9, 3 };
    var tree = try SegmentTree(i32, MinContext, minFn).init(allocator, &data, .{});
    defer tree.deinit();

    // Query entire range: min = 1
    const min_all = try tree.query(0, 5);
    try testing.expectEqual(@as(i32, 1), min_all);

    // Query [0, 2]: min(5, 2, 8) = 2
    const min_left = try tree.query(0, 2);
    try testing.expectEqual(@as(i32, 2), min_left);

    // Query [3, 5]: min(1, 9, 3) = 1
    const min_right = try tree.query(3, 5);
    try testing.expectEqual(@as(i32, 1), min_right);

    try tree.validate();
}

test "SegmentTree: max query" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const MaxContext = struct {};
    const maxFn = struct {
        fn f(_: MaxContext, a: i32, b: i32) i32 {
            return @max(a, b);
        }
    }.f;

    const data = [_]i32{ 5, 2, 8, 1, 9, 3 };
    var tree = try SegmentTree(i32, MaxContext, maxFn).init(allocator, &data, .{});
    defer tree.deinit();

    // Query entire range: max = 9
    const max_all = try tree.query(0, 5);
    try testing.expectEqual(@as(i32, 9), max_all);

    // Query [0, 2]: max(5, 2, 8) = 8
    const max_left = try tree.query(0, 2);
    try testing.expectEqual(@as(i32, 8), max_left);

    // Query [4, 5]: max(9, 3) = 9
    const max_right = try tree.query(4, 5);
    try testing.expectEqual(@as(i32, 9), max_right);

    try tree.validate();
}

test "SegmentTree: point update" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const SumContext = struct {};
    const sumFn = struct {
        fn f(_: SumContext, a: i32, b: i32) i32 {
            return a + b;
        }
    }.f;

    const data = [_]i32{ 1, 2, 3, 4, 5 };
    var tree = try SegmentTree(i32, SumContext, sumFn).init(allocator, &data, .{});
    defer tree.deinit();

    // Initial sum [0, 4]: 1+2+3+4+5 = 15
    const sum_before = try tree.query(0, 4);
    try testing.expectEqual(@as(i32, 15), sum_before);

    // Update index 2: 3 -> 10
    try tree.update(2, 10);

    // New sum [0, 4]: 1+2+10+4+5 = 22
    const sum_after = try tree.query(0, 4);
    try testing.expectEqual(@as(i32, 22), sum_after);

    // Query [2, 2]: should be 10
    const sum_single = try tree.query(2, 2);
    try testing.expectEqual(@as(i32, 10), sum_single);

    try tree.validate();
}

test "SegmentTree: empty tree" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const SumContext = struct {};
    const sumFn = struct {
        fn f(_: SumContext, a: i32, b: i32) i32 {
            return a + b;
        }
    }.f;

    const data = [_]i32{};
    var tree = try SegmentTree(i32, SumContext, sumFn).init(allocator, &data, .{});
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 0), tree.count());
    try testing.expectEqual(true, tree.isEmpty());

    // Operations on empty tree should return error
    const query_result = tree.query(0, 0);
    try testing.expectError(error.EmptyTree, query_result);

    const update_result = tree.update(0, 42);
    try testing.expectError(error.EmptyTree, update_result);

    try tree.validate();
}

test "SegmentTree: single element" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const SumContext = struct {};
    const sumFn = struct {
        fn f(_: SumContext, a: i32, b: i32) i32 {
            return a + b;
        }
    }.f;

    const data = [_]i32{42};
    var tree = try SegmentTree(i32, SumContext, sumFn).init(allocator, &data, .{});
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 1), tree.count());
    try testing.expectEqual(false, tree.isEmpty());

    const sum = try tree.query(0, 0);
    try testing.expectEqual(@as(i32, 42), sum);

    try tree.update(0, 100);
    const new_sum = try tree.query(0, 0);
    try testing.expectEqual(@as(i32, 100), new_sum);

    try tree.validate();
}

test "SegmentTree: error handling" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const SumContext = struct {};
    const sumFn = struct {
        fn f(_: SumContext, a: i32, b: i32) i32 {
            return a + b;
        }
    }.f;

    const data = [_]i32{ 1, 2, 3, 4, 5 };
    var tree = try SegmentTree(i32, SumContext, sumFn).init(allocator, &data, .{});
    defer tree.deinit();

    // Invalid range (start > end)
    const invalid_range = tree.query(3, 2);
    try testing.expectError(error.InvalidRange, invalid_range);

    // Out of bounds
    const out_of_bounds = tree.query(0, 10);
    try testing.expectError(error.IndexOutOfBounds, out_of_bounds);

    // Update out of bounds
    const update_oob = tree.update(10, 42);
    try testing.expectError(error.IndexOutOfBounds, update_oob);
}

test "SegmentTree: stress test with 1000 elements" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const SumContext = struct {};
    const sumFn = struct {
        fn f(_: SumContext, a: i64, b: i64) i64 {
            return a + b;
        }
    }.f;

    var data: [1000]i64 = undefined;
    for (&data, 0..) |*val, i| {
        val.* = @intCast(i + 1);
    }

    var tree = try SegmentTree(i64, SumContext, sumFn).init(allocator, &data, .{});
    defer tree.deinit();

    // Sum of 1..1000 = 1000*1001/2 = 500500
    const sum_all = try tree.query(0, 999);
    try testing.expectEqual(@as(i64, 500500), sum_all);

    // Sum of 1..500 = 500*501/2 = 125250
    const sum_half = try tree.query(0, 499);
    try testing.expectEqual(@as(i64, 125250), sum_half);

    // Update several elements
    try tree.update(0, 1000); // 1 -> 1000 (+999)
    try tree.update(499, 2000); // 500 -> 2000 (+1500)

    // New sum [0, 499]: 125250 + 999 + 1500 = 127749
    const sum_after = try tree.query(0, 499);
    try testing.expectEqual(@as(i64, 127749), sum_after);

    try tree.validate();
}

test "SegmentTree: GCD query" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const GcdContext = struct {};
    const gcdFn = struct {
        fn gcd(a: i32, b: i32) i32 {
            var x = a;
            var y = b;
            while (y != 0) {
                const temp = y;
                y = @mod(x, y);
                x = temp;
            }
            return x;
        }

        fn f(_: GcdContext, a: i32, b: i32) i32 {
            return gcd(a, b);
        }
    }.f;

    const data = [_]i32{ 12, 18, 24, 30, 36 };
    var tree = try SegmentTree(i32, GcdContext, gcdFn).init(allocator, &data, .{});
    defer tree.deinit();

    // GCD of all elements: gcd(12,18,24,30,36) = 6
    const gcd_all = try tree.query(0, 4);
    try testing.expectEqual(@as(i32, 6), gcd_all);

    // GCD of [0,1]: gcd(12,18) = 6
    const gcd_pair = try tree.query(0, 1);
    try testing.expectEqual(@as(i32, 6), gcd_pair);

    try tree.validate();
}

test "SegmentTree: multiple updates" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const MinContext = struct {};
    const minFn = struct {
        fn f(_: MinContext, a: i32, b: i32) i32 {
            return @min(a, b);
        }
    }.f;

    var data = [_]i32{ 10, 20, 30, 40, 50 };
    var tree = try SegmentTree(i32, MinContext, minFn).init(allocator, &data, .{});
    defer tree.deinit();

    // Initial min: 10
    const min_before = try tree.query(0, 4);
    try testing.expectEqual(@as(i32, 10), min_before);

    // Update to make 5 the new minimum
    try tree.update(2, 5);
    const min_after1 = try tree.query(0, 4);
    try testing.expectEqual(@as(i32, 5), min_after1);

    // Update to make 3 the new minimum
    try tree.update(4, 3);
    const min_after2 = try tree.query(0, 4);
    try testing.expectEqual(@as(i32, 3), min_after2);

    // Update back to large value
    try tree.update(4, 100);
    const min_after3 = try tree.query(0, 4);
    try testing.expectEqual(@as(i32, 5), min_after3);

    try tree.validate();
}
