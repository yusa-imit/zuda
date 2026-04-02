const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Wavelet Tree — Space-efficient data structure for range queries on sequences
///
/// A Wavelet Tree is a balanced binary tree that stores a sequence and supports:
/// - rank(c, i): Count occurrences of c in [0, i)
/// - select(c, k): Find position of k-th occurrence of c
/// - access(i): Get element at position i
/// - rangeCount(l, r, c): Count occurrences of c in [l, r)
/// - rangeQuantile(l, r, k): Find k-th smallest element in [l, r)
///
/// Time complexity:
/// - Construction: O(n log σ) where σ is alphabet size
/// - rank/select/access: O(log σ)
/// - rangeCount/rangeQuantile: O(log σ)
///
/// Space complexity: O(n log σ) bits (can be reduced to n log σ + o(n log σ))
///
/// Use cases:
/// - Compressed suffix arrays
/// - Range counting/quantile queries
/// - Document retrieval
/// - Sequence indexing with small alphabets
pub fn WaveletTree(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Bitmap for left/right branching decisions
        const Bitmap = std.DynamicBitSet;

        /// Node in the wavelet tree
        const Node = struct {
            bitmap: Bitmap,
            left: ?*Node,
            right: ?*Node,
            min_val: T,
            max_val: T,

            fn deinit(self: *Node, allocator: Allocator) void {
                if (self.left) |left| {
                    left.deinit(allocator);
                    allocator.destroy(left);
                }
                if (self.right) |right| {
                    right.deinit(allocator);
                    allocator.destroy(right);
                }
                self.bitmap.deinit();
            }
        };

        allocator: Allocator,
        root: ?*Node,
        length: usize,
        alphabet_size: usize,

        /// Initialize an empty wavelet tree
        /// Time: O(1)
        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
                .root = null,
                .length = 0,
                .alphabet_size = 0,
            };
        }

        /// Build wavelet tree from a sequence
        /// Time: O(n log σ) where n is sequence length, σ is alphabet size
        /// Space: O(n log σ)
        pub fn build(allocator: Allocator, sequence: []const T) !Self {
            if (sequence.len == 0) {
                return init(allocator);
            }

            // Find alphabet range
            var min_val = sequence[0];
            var max_val = sequence[0];
            for (sequence) |val| {
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }

            const alphabet_size = @as(usize, @intCast(@as(i64, max_val) - @as(i64, min_val) + 1));

            var self = Self{
                .allocator = allocator,
                .root = null,
                .length = sequence.len,
                .alphabet_size = alphabet_size,
            };

            self.root = try self.buildNode(sequence, min_val, max_val);
            return self;
        }

        /// Recursively build a wavelet tree node
        fn buildNode(self: *Self, sequence: []const T, min_val: T, max_val: T) !*Node {
            const node = try self.allocator.create(Node);
            errdefer self.allocator.destroy(node);

            node.* = .{
                .bitmap = try Bitmap.initEmpty(self.allocator, sequence.len),
                .left = null,
                .right = null,
                .min_val = min_val,
                .max_val = max_val,
            };
            errdefer node.bitmap.deinit();

            // Leaf node
            if (min_val == max_val) {
                return node;
            }

            // Split value (midpoint)
            const mid = min_val + @divFloor(max_val - min_val, 2);

            // Build left and right sequences
            var left_seq = std.ArrayList(T).init(self.allocator);
            defer left_seq.deinit();
            var right_seq = std.ArrayList(T).init(self.allocator);
            defer right_seq.deinit();

            for (sequence, 0..) |val, i| {
                if (val <= mid) {
                    try left_seq.append(val);
                    // bitmap[i] = 0 for left (already 0 by default)
                } else {
                    try right_seq.append(val);
                    node.bitmap.set(i);
                }
            }

            // Recursively build children
            if (left_seq.items.len > 0) {
                node.left = try self.buildNode(left_seq.items, min_val, mid);
            }
            if (right_seq.items.len > 0) {
                node.right = try self.buildNode(right_seq.items, mid + 1, max_val);
            }

            return node;
        }

        /// Free all memory
        pub fn deinit(self: *Self) void {
            if (self.root) |root| {
                root.deinit(self.allocator);
                self.allocator.destroy(root);
            }
        }

        /// Access element at position i
        /// Time: O(log σ)
        pub fn access(self: *const Self, i: usize) !T {
            if (i >= self.length) return error.IndexOutOfBounds;
            if (self.root == null) return error.EmptyTree;

            var node = self.root.?;
            var pos = i;

            while (node.min_val != node.max_val) {
                const goes_right = node.bitmap.isSet(pos);
                if (goes_right) {
                    // Count how many elements before pos went right
                    pos = self.rank1(node, pos);
                    node = node.right orelse return error.InvalidState;
                } else {
                    // Count how many elements before pos went left
                    pos = self.rank0(node, pos);
                    node = node.left orelse return error.InvalidState;
                }
            }

            return node.min_val;
        }

        /// Count occurrences of c in [0, i)
        /// Time: O(log σ)
        pub fn rank(self: *const Self, c: T, i: usize) !usize {
            if (i > self.length) return error.IndexOutOfBounds;
            if (i == 0) return 0;
            if (self.root == null) return 0;

            return self.rankNode(self.root.?, c, i);
        }

        fn rankNode(self: *const Self, node: *Node, c: T, i: usize) usize {
            if (c < node.min_val or c > node.max_val) return 0;
            if (node.min_val == node.max_val) return i;

            const mid = node.min_val + @divFloor(node.max_val - node.min_val, 2);

            if (c <= mid) {
                const left_i = self.rank0(node, i);
                return if (node.left) |left| self.rankNode(left, c, left_i) else 0;
            } else {
                const right_i = self.rank1(node, i);
                return if (node.right) |right| self.rankNode(right, c, right_i) else 0;
            }
        }

        /// Find position of k-th occurrence of c (0-indexed)
        /// Time: O(log σ)
        pub fn select(self: *const Self, c: T, k: usize) !usize {
            if (self.root == null) return error.NotFound;
            return self.selectNode(self.root.?, c, k);
        }

        fn selectNode(self: *const Self, node: *Node, c: T, k: usize) error{NotFound}!usize {
            if (c < node.min_val or c > node.max_val) return error.NotFound;
            if (node.min_val == node.max_val) {
                // At leaf, k-th occurrence is just position k
                if (k >= node.bitmap.capacity()) return error.NotFound;
                return k;
            }

            const mid = node.min_val + @divFloor(node.max_val - node.min_val, 2);

            if (c <= mid) {
                if (node.left) |left| {
                    const left_pos = try self.selectNode(left, c, k);
                    // Map back to original position using select0
                    return self.select0(node, left_pos);
                }
            } else {
                if (node.right) |right| {
                    const right_pos = try self.selectNode(right, c, k);
                    // Map back to original position using select1
                    return self.select1(node, right_pos);
                }
            }

            return error.NotFound;
        }

        /// Count occurrences of c in [l, r)
        /// Time: O(log σ)
        pub fn rangeCount(self: *const Self, l: usize, r: usize, c: T) !usize {
            if (l >= r or r > self.length) return error.InvalidRange;
            const rank_r = try self.rank(c, r);
            const rank_l = try self.rank(c, l);
            return rank_r - rank_l;
        }

        /// Find k-th smallest element in [l, r) (0-indexed)
        /// Time: O(log σ)
        pub fn rangeQuantile(self: *const Self, l: usize, r: usize, k: usize) !T {
            if (l >= r or r > self.length) return error.InvalidRange;
            if (self.root == null) return error.EmptyTree;
            if (k >= (r - l)) return error.IndexOutOfBounds;

            return self.quantileNode(self.root.?, l, r, k);
        }

        fn quantileNode(self: *const Self, node: *Node, l: usize, r: usize, k: usize) error{ InvalidState, IndexOutOfBounds }!T {
            if (node.min_val == node.max_val) return node.min_val;

            const left_count = self.rank0(node, r) - self.rank0(node, l);

            if (k < left_count) {
                // k-th smallest is in left subtree
                const new_l = self.rank0(node, l);
                const new_r = self.rank0(node, r);
                return self.quantileNode(node.left orelse return error.InvalidState, new_l, new_r, k);
            } else {
                // k-th smallest is in right subtree
                const new_l = self.rank1(node, l);
                const new_r = self.rank1(node, r);
                return self.quantileNode(node.right orelse return error.InvalidState, new_l, new_r, k - left_count);
            }
        }

        /// Count 0s (left branches) in bitmap[0..i)
        fn rank0(self: *const Self, node: *Node, i: usize) usize {
            _ = self;
            return i - node.bitmap.count();
        }

        /// Count 1s (right branches) in bitmap[0..i)
        fn rank1(self: *const Self, node: *Node, i: usize) usize {
            _ = self;
            var count: usize = 0;
            var idx: usize = 0;
            while (idx < i) : (idx += 1) {
                if (node.bitmap.isSet(idx)) count += 1;
            }
            return count;
        }

        /// Find position of k-th 0 (0-indexed)
        fn select0(self: *const Self, node: *Node, k: usize) usize {
            _ = self;
            var count: usize = 0;
            var idx: usize = 0;
            while (idx < node.bitmap.capacity()) : (idx += 1) {
                if (!node.bitmap.isSet(idx)) {
                    if (count == k) return idx;
                    count += 1;
                }
            }
            return node.bitmap.capacity();
        }

        /// Find position of k-th 1 (0-indexed)
        fn select1(self: *const Self, node: *Node, k: usize) usize {
            _ = self;
            var count: usize = 0;
            var idx: usize = 0;
            while (idx < node.bitmap.capacity()) : (idx += 1) {
                if (node.bitmap.isSet(idx)) {
                    if (count == k) return idx;
                    count += 1;
                }
            }
            return node.bitmap.capacity();
        }

        /// Get the length of the sequence
        pub fn len(self: *const Self) usize {
            return self.length;
        }

        /// Check if tree is empty
        pub fn isEmpty(self: *const Self) bool {
            return self.length == 0;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "WaveletTree: init and deinit" {
    var wt = WaveletTree(u32).init(testing.allocator);
    defer wt.deinit();

    try testing.expect(wt.isEmpty());
    try testing.expectEqual(@as(usize, 0), wt.len());
}

test "WaveletTree: build from sequence" {
    const sequence = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var wt = try WaveletTree(u32).build(testing.allocator, &sequence);
    defer wt.deinit();

    try testing.expectEqual(@as(usize, 10), wt.len());
    try testing.expect(!wt.isEmpty());
}

test "WaveletTree: access elements" {
    const sequence = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var wt = try WaveletTree(u32).build(testing.allocator, &sequence);
    defer wt.deinit();

    try testing.expectEqual(@as(u32, 3), try wt.access(0));
    try testing.expectEqual(@as(u32, 1), try wt.access(1));
    try testing.expectEqual(@as(u32, 4), try wt.access(2));
    try testing.expectEqual(@as(u32, 1), try wt.access(3));
    try testing.expectEqual(@as(u32, 5), try wt.access(4));
    try testing.expectEqual(@as(u32, 9), try wt.access(5));
    try testing.expectEqual(@as(u32, 2), try wt.access(6));
    try testing.expectEqual(@as(u32, 6), try wt.access(7));
    try testing.expectEqual(@as(u32, 5), try wt.access(8));
    try testing.expectEqual(@as(u32, 3), try wt.access(9));
}

test "WaveletTree: access out of bounds" {
    const sequence = [_]u32{ 1, 2, 3 };
    var wt = try WaveletTree(u32).build(testing.allocator, &sequence);
    defer wt.deinit();

    try testing.expectError(error.IndexOutOfBounds, wt.access(3));
    try testing.expectError(error.IndexOutOfBounds, wt.access(100));
}

test "WaveletTree: rank queries" {
    const sequence = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var wt = try WaveletTree(u32).build(testing.allocator, &sequence);
    defer wt.deinit();

    // rank(1, i) counts 1s before position i
    try testing.expectEqual(@as(usize, 0), try wt.rank(1, 0));
    try testing.expectEqual(@as(usize, 0), try wt.rank(1, 1));
    try testing.expectEqual(@as(usize, 1), try wt.rank(1, 2));
    try testing.expectEqual(@as(usize, 1), try wt.rank(1, 3));
    try testing.expectEqual(@as(usize, 2), try wt.rank(1, 4));
    try testing.expectEqual(@as(usize, 2), try wt.rank(1, 10));

    // rank(5, i) counts 5s before position i
    try testing.expectEqual(@as(usize, 0), try wt.rank(5, 4));
    try testing.expectEqual(@as(usize, 1), try wt.rank(5, 5));
    try testing.expectEqual(@as(usize, 1), try wt.rank(5, 8));
    try testing.expectEqual(@as(usize, 2), try wt.rank(5, 9));
}

test "WaveletTree: rank with non-existent element" {
    const sequence = [_]u32{ 1, 2, 3, 4, 5 };
    var wt = try WaveletTree(u32).build(testing.allocator, &sequence);
    defer wt.deinit();

    try testing.expectEqual(@as(usize, 0), try wt.rank(10, 5));
    try testing.expectEqual(@as(usize, 0), try wt.rank(0, 5));
}

test "WaveletTree: select queries" {
    const sequence = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var wt = try WaveletTree(u32).build(testing.allocator, &sequence);
    defer wt.deinit();

    // select(1, 0) finds position of first 1
    try testing.expectEqual(@as(usize, 1), try wt.select(1, 0));
    // select(1, 1) finds position of second 1
    try testing.expectEqual(@as(usize, 3), try wt.select(1, 1));

    // select(5, 0) finds position of first 5
    try testing.expectEqual(@as(usize, 4), try wt.select(5, 0));
    // select(5, 1) finds position of second 5
    try testing.expectEqual(@as(usize, 8), try wt.select(5, 1));

    // select(3, 0) finds position of first 3
    try testing.expectEqual(@as(usize, 0), try wt.select(3, 0));
}

test "WaveletTree: select not found" {
    const sequence = [_]u32{ 1, 2, 3 };
    var wt = try WaveletTree(u32).build(testing.allocator, &sequence);
    defer wt.deinit();

    try testing.expectError(error.NotFound, wt.select(1, 5));
    try testing.expectError(error.NotFound, wt.select(10, 0));
}

test "WaveletTree: range count queries" {
    const sequence = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var wt = try WaveletTree(u32).build(testing.allocator, &sequence);
    defer wt.deinit();

    // Count 1s in [0, 10)
    try testing.expectEqual(@as(usize, 2), try wt.rangeCount(0, 10, 1));
    // Count 1s in [0, 2)
    try testing.expectEqual(@as(usize, 1), try wt.rangeCount(0, 2, 1));
    // Count 1s in [2, 4)
    try testing.expectEqual(@as(usize, 1), try wt.rangeCount(2, 4, 1));
    // Count 5s in [0, 10)
    try testing.expectEqual(@as(usize, 2), try wt.rangeCount(0, 10, 5));
    // Count 5s in [4, 9)
    try testing.expectEqual(@as(usize, 2), try wt.rangeCount(4, 9, 5));
}

test "WaveletTree: range quantile queries" {
    const sequence = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var wt = try WaveletTree(u32).build(testing.allocator, &sequence);
    defer wt.deinit();

    // 0-th smallest (minimum) in [0, 10)
    try testing.expectEqual(@as(u32, 1), try wt.rangeQuantile(0, 10, 0));
    // 1-st smallest in [0, 10)
    try testing.expectEqual(@as(u32, 1), try wt.rangeQuantile(0, 10, 1));
    // 2-nd smallest in [0, 10)
    try testing.expectEqual(@as(u32, 2), try wt.rangeQuantile(0, 10, 2));
    // 9-th smallest (maximum) in [0, 10)
    try testing.expectEqual(@as(u32, 9), try wt.rangeQuantile(0, 10, 9));

    // Median in [0, 10) (5-th smallest, 0-indexed)
    try testing.expectEqual(@as(u32, 4), try wt.rangeQuantile(0, 10, 5));
}

test "WaveletTree: range quantile in subrange" {
    const sequence = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var wt = try WaveletTree(u32).build(testing.allocator, &sequence);
    defer wt.deinit();

    // 0-th smallest in [2, 7) = [4, 1, 5, 9, 2]
    try testing.expectEqual(@as(u32, 1), try wt.rangeQuantile(2, 7, 0));
    // 2-nd smallest in [2, 7)
    try testing.expectEqual(@as(u32, 4), try wt.rangeQuantile(2, 7, 2));
    // 4-th smallest (max) in [2, 7)
    try testing.expectEqual(@as(u32, 9), try wt.rangeQuantile(2, 7, 4));
}

test "WaveletTree: single element" {
    const sequence = [_]u32{42};
    var wt = try WaveletTree(u32).build(testing.allocator, &sequence);
    defer wt.deinit();

    try testing.expectEqual(@as(usize, 1), wt.len());
    try testing.expectEqual(@as(u32, 42), try wt.access(0));
    try testing.expectEqual(@as(usize, 1), try wt.rank(42, 1));
    try testing.expectEqual(@as(usize, 0), try wt.select(42, 0));
}

test "WaveletTree: all same elements" {
    const sequence = [_]u32{ 7, 7, 7, 7, 7 };
    var wt = try WaveletTree(u32).build(testing.allocator, &sequence);
    defer wt.deinit();

    for (0..5) |i| {
        try testing.expectEqual(@as(u32, 7), try wt.access(i));
    }
    try testing.expectEqual(@as(usize, 5), try wt.rank(7, 5));
    try testing.expectEqual(@as(usize, 0), try wt.select(7, 0));
    try testing.expectEqual(@as(usize, 4), try wt.select(7, 4));
}

test "WaveletTree: large alphabet" {
    const sequence = [_]u32{ 100, 200, 50, 150, 75, 125, 175, 25 };
    var wt = try WaveletTree(u32).build(testing.allocator, &sequence);
    defer wt.deinit();

    try testing.expectEqual(@as(u32, 100), try wt.access(0));
    try testing.expectEqual(@as(u32, 25), try wt.access(7));
    try testing.expectEqual(@as(u32, 25), try wt.rangeQuantile(0, 8, 0)); // min
    try testing.expectEqual(@as(u32, 200), try wt.rangeQuantile(0, 8, 7)); // max
}

test "WaveletTree: memory safety with testing.allocator" {
    const sequence = [_]u32{ 5, 2, 8, 2, 9, 1, 4, 7, 3, 6 };
    var wt = try WaveletTree(u32).build(testing.allocator, &sequence);
    defer wt.deinit();

    _ = try wt.access(5);
    _ = try wt.rank(2, 8);
    _ = try wt.select(9, 0);
    _ = try wt.rangeCount(2, 7, 2);
    _ = try wt.rangeQuantile(0, 10, 4);
}

test "WaveletTree: i8 support" {
    const sequence = [_]i8{ -3, 1, -4, 1, 5, -9, 2, -6, 5, -3 };
    var wt = try WaveletTree(i8).build(testing.allocator, &sequence);
    defer wt.deinit();

    try testing.expectEqual(@as(i8, -3), try wt.access(0));
    try testing.expectEqual(@as(i8, -9), try wt.access(5));
    try testing.expectEqual(@as(usize, 2), try wt.rank(1, 10));
    try testing.expectEqual(@as(i8, -9), try wt.rangeQuantile(0, 10, 0)); // min
    try testing.expectEqual(@as(i8, 5), try wt.rangeQuantile(0, 10, 9)); // max
}
