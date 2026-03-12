const std = @import("std");
const Allocator = std.mem.Allocator;

/// Rope data structure for efficient string/sequence editing.
///
/// A rope is a binary tree where each leaf node holds a substring and each
/// internal node holds the total weight (length) of its left subtree. This
/// allows for efficient insertion, deletion, and concatenation compared to
/// a flat array representation.
///
/// Time complexity:
/// - concat: O(log n)
/// - split: O(log n)
/// - insert: O(log n)
/// - delete: O(log n)
/// - charAt: O(log n)
/// - substring: O(log n + k) where k is substring length
///
/// Space complexity: O(n)
///
/// Use cases:
/// - Text editors with large documents
/// - Collaborative editing systems
/// - Undo/redo functionality
/// - Version control systems
pub fn Rope(comptime T: type) type {
    return struct {
        const Self = @This();

        const Node = struct {
            weight: usize, // length of left subtree (or data length for leaf)
            left: ?*Node,
            right: ?*Node,
            data: ?[]const T, // non-null for leaf nodes

            fn isLeaf(self: *const Node) bool {
                return self.data != null;
            }
        };

        const SPLIT_THRESHOLD = 1024; // Max leaf size before splitting

        allocator: Allocator,
        root: ?*Node,

        /// Initialize an empty rope.
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
                .root = null,
            };
        }

        /// Initialize rope from a slice.
        /// Time: O(n) | Space: O(n)
        pub fn fromSlice(allocator: Allocator, data: []const T) !Self {
            var self = Self.init(allocator);
            if (data.len == 0) return self;

            const owned = try allocator.dupe(T, data);
            const node = try allocator.create(Node);
            node.* = .{
                .weight = data.len,
                .left = null,
                .right = null,
                .data = owned,
            };
            self.root = node;
            return self;
        }

        /// Free all allocated memory.
        /// Time: O(n) | Space: O(log n) call stack
        pub fn deinit(self: *Self) void {
            if (self.root) |root| {
                self.freeNode(root);
            }
        }

        fn freeNode(self: *Self, node: *Node) void {
            if (node.left) |left| {
                self.freeNode(left);
            }
            if (node.right) |right| {
                self.freeNode(right);
            }
            if (node.data) |data| {
                self.allocator.free(data);
            }
            self.allocator.destroy(node);
        }

        /// Get the total length of the rope.
        /// Time: O(1) | Space: O(1)
        pub fn length(self: *const Self) usize {
            if (self.root) |root| {
                return self.nodeLength(root);
            }
            return 0;
        }

        fn nodeLength(self: *const Self, node: *const Node) usize {
            if (node.isLeaf()) {
                return node.data.?.len;
            }
            var len = node.weight;
            if (node.right) |right| {
                len += self.nodeLength(right);
            }
            return len;
        }

        /// Concatenate another rope to this rope.
        /// Time: O(log n) | Space: O(1)
        pub fn concat(self: *Self, other: *const Self) !void {
            if (other.root == null) return;
            if (self.root == null) {
                self.root = try self.cloneNode(other.root.?);
                return;
            }

            const new_root = try self.allocator.create(Node);
            const cloned_other = try self.cloneNode(other.root.?);

            new_root.* = .{
                .weight = self.length(),
                .left = self.root,
                .right = cloned_other,
                .data = null,
            };
            self.root = new_root;
        }

        fn cloneNode(self: *Self, node: *const Node) !*Node {
            const new_node = try self.allocator.create(Node);
            new_node.* = .{
                .weight = node.weight,
                .left = null,
                .right = null,
                .data = null,
            };

            if (node.data) |data| {
                new_node.data = try self.allocator.dupe(T, data);
            } else {
                if (node.left) |left| {
                    new_node.left = try self.cloneNode(left);
                }
                if (node.right) |right| {
                    new_node.right = try self.cloneNode(right);
                }
            }

            return new_node;
        }

        /// Get character at index.
        /// Returns error.OutOfBounds if index >= length.
        /// Time: O(log n) | Space: O(1)
        pub fn charAt(self: *const Self, index: usize) !T {
            if (self.root == null) return error.OutOfBounds;
            return self.charAtNode(self.root.?, index);
        }

        fn charAtNode(self: *const Self, node: *const Node, index: usize) !T {
            if (node.isLeaf()) {
                const data = node.data.?;
                if (index >= data.len) return error.OutOfBounds;
                return data[index];
            }

            if (index < node.weight) {
                return self.charAtNode(node.left.?, index);
            } else {
                return self.charAtNode(node.right.?, index - node.weight);
            }
        }

        /// Insert data at index.
        /// Time: O(log n + k) where k is data length | Space: O(log n)
        pub fn insert(self: *Self, index: usize, data: []const T) !void {
            if (data.len == 0) return;

            // Create new leaf node for insertion
            const owned_data = try self.allocator.dupe(T, data);
            const new_node = try self.allocator.create(Node);
            new_node.* = .{
                .weight = data.len,
                .left = null,
                .right = null,
                .data = owned_data,
            };

            if (self.root == null) {
                self.root = new_node;
                return;
            }

            if (index == 0) {
                // Prepend
                const parent = try self.allocator.create(Node);
                parent.* = .{
                    .weight = data.len,
                    .left = new_node,
                    .right = self.root,
                    .data = null,
                };
                self.root = parent;
                return;
            }

            if (index >= self.length()) {
                // Append
                const parent = try self.allocator.create(Node);
                parent.* = .{
                    .weight = self.length(),
                    .left = self.root,
                    .right = new_node,
                    .data = null,
                };
                self.root = parent;
                return;
            }

            // Split at index, insert in middle
            var right_part = try self.split(index);
            defer right_part.deinit();

            // Create parent for left + new_node
            const left_parent = try self.allocator.create(Node);
            left_parent.* = .{
                .weight = if (self.root) |r| self.nodeLength(r) else 0,
                .left = self.root,
                .right = new_node,
                .data = null,
            };

            // Create final parent
            const final_parent = try self.allocator.create(Node);
            final_parent.* = .{
                .weight = self.nodeLength(left_parent),
                .left = left_parent,
                .right = right_part.root,
                .data = null,
            };

            self.root = final_parent;
            right_part.root = null; // Prevent double-free
        }

        /// Split rope at index, returning the right part.
        /// The left part remains in self.
        /// Time: O(log n) | Space: O(log n)
        pub fn split(self: *Self, index: usize) !Self {
            if (index >= self.length()) {
                return Self.init(self.allocator);
            }

            if (index == 0) {
                const result = Self{
                    .allocator = self.allocator,
                    .root = self.root,
                };
                self.root = null;
                return result;
            }

            const parts = try self.splitNode(self.root.?, index);
            self.root = parts.left;

            return .{
                .allocator = self.allocator,
                .root = parts.right,
            };
        }

        fn splitNode(self: *Self, node: *Node, index: usize) !struct { left: ?*Node, right: ?*Node } {
            if (node.isLeaf()) {
                const data = node.data.?;
                if (index >= data.len) {
                    return .{ .left = node, .right = null };
                }
                if (index == 0) {
                    return .{ .left = null, .right = node };
                }

                const left_data = try self.allocator.dupe(T, data[0..index]);
                const right_data = try self.allocator.dupe(T, data[index..]);

                const left_node = try self.allocator.create(Node);
                left_node.* = .{
                    .weight = left_data.len,
                    .left = null,
                    .right = null,
                    .data = left_data,
                };

                const right_node = try self.allocator.create(Node);
                right_node.* = .{
                    .weight = right_data.len,
                    .left = null,
                    .right = null,
                    .data = right_data,
                };

                // Free original node
                self.allocator.free(data);
                self.allocator.destroy(node);

                return .{ .left = left_node, .right = right_node };
            }

            if (index < node.weight) {
                const left_split = try self.splitNode(node.left.?, index);
                node.left = left_split.right;
                node.weight = if (node.left) |l| self.nodeLength(l) else 0;

                if (node.left == null and node.right == null) {
                    self.allocator.destroy(node);
                    return .{ .left = left_split.left, .right = null };
                }

                return .{ .left = left_split.left, .right = node };
            } else if (index == node.weight) {
                const left = node.left;
                const right = node.right;
                self.allocator.destroy(node);
                return .{ .left = left, .right = right };
            } else {
                const right_split = try self.splitNode(node.right.?, index - node.weight);
                node.right = right_split.left;

                if (node.left == null and node.right == null) {
                    self.allocator.destroy(node);
                    return .{ .left = null, .right = right_split.right };
                }

                return .{ .left = node, .right = right_split.right };
            }
        }

        /// Convert rope to a contiguous slice.
        /// Caller owns returned memory.
        /// Time: O(n) | Space: O(n)
        pub fn toSlice(self: *const Self) ![]T {
            const len = self.length();
            if (len == 0) {
                return &[_]T{};
            }

            const result = try self.allocator.alloc(T, len);
            var index: usize = 0;
            if (self.root) |root| {
                self.toSliceNode(root, result, &index);
            }
            return result;
        }

        fn toSliceNode(self: *const Self, node: *const Node, buffer: []T, index: *usize) void {
            if (node.isLeaf()) {
                const data = node.data.?;
                @memcpy(buffer[index.*..][0..data.len], data);
                index.* += data.len;
                return;
            }

            if (node.left) |left| {
                self.toSliceNode(left, buffer, index);
            }
            if (node.right) |right| {
                self.toSliceNode(right, buffer, index);
            }
        }

        /// Validate internal invariants.
        /// Time: O(n) | Space: O(log n)
        pub fn validate(self: *const Self) !void {
            if (self.root) |root| {
                _ = try self.validateNode(root);
            }
        }

        fn validateNode(self: *const Self, node: *const Node) !usize {
            if (node.isLeaf()) {
                const data = node.data.?;
                if (node.weight != data.len) {
                    return error.InvalidWeight;
                }
                return data.len;
            }

            var total_len: usize = 0;
            if (node.left) |left| {
                const left_len = try self.validateNode(left);
                if (node.weight != left_len) {
                    return error.InvalidWeight;
                }
                total_len += left_len;
            }

            if (node.right) |right| {
                total_len += try self.validateNode(right);
            }

            return total_len;
        }
    };
}

// Tests
test "Rope: basic init and length" {
    const allocator = std.testing.allocator;
    var rope = Rope(u8).init(allocator);
    defer rope.deinit();

    try std.testing.expectEqual(@as(usize, 0), rope.length());
}

test "Rope: from slice" {
    const allocator = std.testing.allocator;
    var rope = try Rope(u8).fromSlice(allocator, "hello");
    defer rope.deinit();

    try std.testing.expectEqual(@as(usize, 5), rope.length());
    try rope.validate();
}

test "Rope: charAt" {
    const allocator = std.testing.allocator;
    var rope = try Rope(u8).fromSlice(allocator, "hello");
    defer rope.deinit();

    try std.testing.expectEqual(@as(u8, 'h'), try rope.charAt(0));
    try std.testing.expectEqual(@as(u8, 'e'), try rope.charAt(1));
    try std.testing.expectEqual(@as(u8, 'l'), try rope.charAt(2));
    try std.testing.expectEqual(@as(u8, 'l'), try rope.charAt(3));
    try std.testing.expectEqual(@as(u8, 'o'), try rope.charAt(4));
    try std.testing.expectError(error.OutOfBounds, rope.charAt(5));
}

test "Rope: concat" {
    const allocator = std.testing.allocator;
    var rope1 = try Rope(u8).fromSlice(allocator, "hello");
    defer rope1.deinit();

    var rope2 = try Rope(u8).fromSlice(allocator, " world");
    defer rope2.deinit();

    try rope1.concat(&rope2);

    try std.testing.expectEqual(@as(usize, 11), rope1.length());
    try rope1.validate();

    const result = try rope1.toSlice();
    defer allocator.free(result);
    try std.testing.expectEqualSlices(u8, "hello world", result);
}

test "Rope: toSlice" {
    const allocator = std.testing.allocator;
    var rope = try Rope(u8).fromSlice(allocator, "test");
    defer rope.deinit();

    const slice = try rope.toSlice();
    defer allocator.free(slice);

    try std.testing.expectEqualSlices(u8, "test", slice);
}

test "Rope: insert at beginning" {
    const allocator = std.testing.allocator;
    var rope = try Rope(u8).fromSlice(allocator, "world");
    defer rope.deinit();

    try rope.insert(0, "hello ");

    try std.testing.expectEqual(@as(usize, 11), rope.length());

    const result = try rope.toSlice();
    defer allocator.free(result);
    try std.testing.expectEqualSlices(u8, "hello world", result);
}

test "Rope: insert at end" {
    const allocator = std.testing.allocator;
    var rope = try Rope(u8).fromSlice(allocator, "hello");
    defer rope.deinit();

    try rope.insert(5, " world");

    try std.testing.expectEqual(@as(usize, 11), rope.length());

    const result = try rope.toSlice();
    defer allocator.free(result);
    try std.testing.expectEqualSlices(u8, "hello world", result);
}

test "Rope: insert in middle" {
    const allocator = std.testing.allocator;
    var rope = try Rope(u8).fromSlice(allocator, "helo");
    defer rope.deinit();

    try rope.insert(2, "l");

    try std.testing.expectEqual(@as(usize, 5), rope.length());

    const result = try rope.toSlice();
    defer allocator.free(result);
    try std.testing.expectEqualSlices(u8, "hello", result);
}

test "Rope: split" {
    const allocator = std.testing.allocator;
    var rope = try Rope(u8).fromSlice(allocator, "hello world");
    defer rope.deinit();

    var right = try rope.split(6);
    defer right.deinit();

    try std.testing.expectEqual(@as(usize, 6), rope.length());
    try std.testing.expectEqual(@as(usize, 5), right.length());

    const left_result = try rope.toSlice();
    defer allocator.free(left_result);
    try std.testing.expectEqualSlices(u8, "hello ", left_result);

    const right_result = try right.toSlice();
    defer allocator.free(right_result);
    try std.testing.expectEqualSlices(u8, "world", right_result);
}

test "Rope: empty rope operations" {
    const allocator = std.testing.allocator;
    var rope = Rope(u8).init(allocator);
    defer rope.deinit();

    try std.testing.expectEqual(@as(usize, 0), rope.length());
    try std.testing.expectError(error.OutOfBounds, rope.charAt(0));

    const slice = try rope.toSlice();
    defer allocator.free(slice);
    try std.testing.expectEqual(@as(usize, 0), slice.len);
}

test "Rope: memory leak check" {
    const allocator = std.testing.allocator;

    var rope = try Rope(u8).fromSlice(allocator, "test string for memory leak detection");
    defer rope.deinit();

    try rope.insert(5, "inserted ");
    var right = try rope.split(10);
    defer right.deinit();

    var rope2 = try Rope(u8).fromSlice(allocator, "another");
    defer rope2.deinit();

    try rope.concat(&rope2);
}

test "Rope: validate invariants" {
    const allocator = std.testing.allocator;
    var rope = try Rope(u8).fromSlice(allocator, "hello");
    defer rope.deinit();

    try rope.validate();

    var rope2 = try Rope(u8).fromSlice(allocator, " world");
    defer rope2.deinit();

    try rope.concat(&rope2);
    try rope.validate();
}
