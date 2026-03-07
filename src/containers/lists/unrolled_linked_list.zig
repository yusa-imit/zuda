const std = @import("std");
const testing = std.testing;

/// UnrolledLinkedList — cache-friendly linked list storing multiple elements per node
///
/// Each node contains a fixed-size array of up to N elements, reducing memory overhead
/// and improving cache locality compared to traditional linked lists. Nodes are split
/// when full and merged when under-utilized.
///
/// Performance characteristics:
/// - Insert: O(n) worst-case, O(1) amortized at ends
/// - Remove: O(n) worst-case, O(1) amortized at ends
/// - Get: O(n/N) where N is elements per node
/// - Iteration: O(n) with better cache locality than singly-linked list
///
/// Type parameters:
/// - T: element type
/// - N: comptime elements per node (default 16 for balance of memory and cache)
pub fn UnrolledLinkedList(comptime T: type, comptime N: usize) type {
    if (N == 0) @compileError("UnrolledLinkedList: N must be at least 1");

    return struct {
        const Self = @This();

        /// Node in the unrolled linked list
        const Node = struct {
            elements: [N]T = undefined,
            count: usize = 0, // number of valid elements in this node
            next: ?*Node = null,

            fn isFull(self: *const Node) bool {
                return self.count == N;
            }

            fn isEmpty(self: *const Node) bool {
                return self.count == 0;
            }

            fn isUnderHalf(self: *const Node) bool {
                return self.count < N / 2;
            }

            /// Insert element at position within this node
            fn insertAt(self: *Node, index: usize, value: T) void {
                std.debug.assert(index <= self.count);
                std.debug.assert(self.count < N);

                // Shift elements to make room
                if (index < self.count) {
                    var i = self.count;
                    while (i > index) : (i -= 1) {
                        self.elements[i] = self.elements[i - 1];
                    }
                }

                self.elements[index] = value;
                self.count += 1;
            }

            /// Remove element at position within this node
            fn removeAt(self: *Node, index: usize) T {
                std.debug.assert(index < self.count);

                const value = self.elements[index];

                // Shift elements to fill gap
                var i = index;
                while (i < self.count - 1) : (i += 1) {
                    self.elements[i] = self.elements[i + 1];
                }

                self.count -= 1;
                return value;
            }
        };

        pub const Iterator = struct {
            current_node: ?*Node,
            index: usize,

            /// Returns next element, or null if exhausted
            /// Time: O(1) amortized | Space: O(1)
            pub fn next(self: *Iterator) ?T {
                while (self.current_node) |node| {
                    if (self.index < node.count) {
                        const value = node.elements[self.index];
                        self.index += 1;
                        return value;
                    } else {
                        self.current_node = node.next;
                        self.index = 0;
                    }
                }
                return null;
            }
        };

        allocator: std.mem.Allocator,
        head: ?*Node = null,
        tail: ?*Node = null,
        length: usize = 0,

        // -- Lifecycle --

        /// Initialize empty unrolled linked list
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
            };
        }

        /// Free all allocated memory
        /// Time: O(n/N) | Space: O(1)
        pub fn deinit(self: *Self) void {
            var current = self.head;
            while (current) |node| {
                const next_node = node.next;
                self.allocator.destroy(node);
                current = next_node;
            }
            self.* = undefined;
        }

        /// Create a deep copy of this list
        /// Time: O(n) | Space: O(n)
        pub fn clone(self: *const Self) !Self {
            var new_list = Self.init(self.allocator);
            errdefer new_list.deinit();

            var it = self.iterator();
            while (it.next()) |value| {
                try new_list.append(value);
            }

            return new_list;
        }

        // -- Capacity --

        /// Returns the total number of elements
        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.length;
        }

        /// Returns true if the list is empty
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.length == 0;
        }

        // -- Modification --

        /// Append element to the end of the list
        /// Time: O(1) amortized | Space: O(1) amortized
        pub fn append(self: *Self, value: T) !void {
            if (self.tail) |tail_node| {
                if (tail_node.isFull()) {
                    // Need to create a new node
                    const new_node = try self.allocator.create(Node);
                    new_node.* = .{};
                    new_node.insertAt(0, value);

                    tail_node.next = new_node;
                    self.tail = new_node;
                } else {
                    // Append to existing tail node
                    tail_node.insertAt(tail_node.count, value);
                }
            } else {
                // First element
                const new_node = try self.allocator.create(Node);
                new_node.* = .{};
                new_node.insertAt(0, value);

                self.head = new_node;
                self.tail = new_node;
            }

            self.length += 1;
        }

        /// Prepend element to the beginning of the list
        /// Time: O(1) amortized | Space: O(1) amortized
        pub fn prepend(self: *Self, value: T) !void {
            if (self.head) |head_node| {
                if (head_node.isFull()) {
                    // Need to create a new node
                    const new_node = try self.allocator.create(Node);
                    new_node.* = .{};
                    new_node.insertAt(0, value);
                    new_node.next = head_node;

                    self.head = new_node;
                } else {
                    // Prepend to existing head node
                    head_node.insertAt(0, value);
                }
            } else {
                // First element
                const new_node = try self.allocator.create(Node);
                new_node.* = .{};
                new_node.insertAt(0, value);

                self.head = new_node;
                self.tail = new_node;
            }

            self.length += 1;
        }

        /// Insert element at the specified index
        /// Time: O(n/N) average | Space: O(1) amortized
        pub fn insertAt(self: *Self, index: usize, value: T) !void {
            if (index > self.length) return error.IndexOutOfBounds;
            if (index == 0) return self.prepend(value);
            if (index == self.length) return self.append(value);

            // Find the node and local index
            var current_node = self.head;
            var remaining = index;

            while (current_node) |node| {
                if (remaining <= node.count) {
                    // Insert in this node
                    if (node.isFull()) {
                        // Need to split the node
                        try self.splitNode(node);
                        // Retry insertion (tail recursion emulation)
                        return self.insertAt(index, value);
                    } else {
                        node.insertAt(remaining, value);
                        self.length += 1;
                        return;
                    }
                }
                remaining -= node.count;
                current_node = node.next;
            }

            unreachable; // Should never reach here
        }

        /// Remove and return element at the specified index
        /// Time: O(n/N) average | Space: O(1)
        pub fn removeAt(self: *Self, index: usize) !T {
            if (index >= self.length) return error.IndexOutOfBounds;
            if (self.isEmpty()) return error.ListEmpty;

            // Find the node and local index
            var prev_node: ?*Node = null;
            var current_node = self.head;
            var remaining = index;

            while (current_node) |node| {
                if (remaining < node.count) {
                    const value = node.removeAt(remaining);
                    self.length -= 1;

                    // Check if node should be merged or removed
                    if (node.isEmpty()) {
                        // Remove empty node
                        if (prev_node) |prev| {
                            prev.next = node.next;
                        } else {
                            self.head = node.next;
                        }

                        if (self.tail == node) {
                            self.tail = prev_node;
                        }

                        self.allocator.destroy(node);
                    } else if (node.isUnderHalf() and node.next != null) {
                        // Try to merge with next node
                        self.tryMergeNodes(prev_node, node) catch {};
                    }

                    return value;
                }

                remaining -= node.count;
                prev_node = node;
                current_node = node.next;
            }

            unreachable; // Should never reach here
        }

        /// Remove and return the first element
        /// Time: O(1) amortized | Space: O(1)
        pub fn popFirst(self: *Self) !T {
            return self.removeAt(0);
        }

        /// Remove and return the last element
        /// Time: O(n/N) | Space: O(1)
        pub fn popLast(self: *Self) !T {
            if (self.isEmpty()) return error.ListEmpty;
            return self.removeAt(self.length - 1);
        }

        /// Remove all elements from the list
        /// Time: O(n/N) | Space: O(1)
        pub fn clear(self: *Self) void {
            var current = self.head;
            while (current) |node| {
                const next_node = node.next;
                self.allocator.destroy(node);
                current = next_node;
            }

            self.head = null;
            self.tail = null;
            self.length = 0;
        }

        // -- Lookup --

        /// Get element at the specified index
        /// Time: O(n/N) average | Space: O(1)
        pub fn get(self: *const Self, index: usize) !T {
            if (index >= self.length) return error.IndexOutOfBounds;

            var current_node = self.head;
            var remaining = index;

            while (current_node) |node| {
                if (remaining < node.count) {
                    return node.elements[remaining];
                }
                remaining -= node.count;
                current_node = node.next;
            }

            unreachable;
        }

        /// Get pointer to element at the specified index
        /// Time: O(n/N) average | Space: O(1)
        pub fn getPtr(self: *Self, index: usize) !*T {
            if (index >= self.length) return error.IndexOutOfBounds;

            var current_node = self.head;
            var remaining = index;

            while (current_node) |node| {
                if (remaining < node.count) {
                    return &node.elements[remaining];
                }
                remaining -= node.count;
                current_node = node.next;
            }

            unreachable;
        }

        // -- Iteration --

        /// Returns an iterator over the list
        /// Time: O(1) | Space: O(1)
        pub fn iterator(self: *const Self) Iterator {
            return .{
                .current_node = self.head,
                .index = 0,
            };
        }

        // -- Bulk Operations --

        /// Create list from a slice
        /// Time: O(n) | Space: O(n)
        pub fn fromSlice(allocator: std.mem.Allocator, slice: []const T) !Self {
            var list = Self.init(allocator);
            errdefer list.deinit();

            for (slice) |value| {
                try list.append(value);
            }

            return list;
        }

        /// Convert list to a newly allocated slice
        /// Time: O(n) | Space: O(n)
        pub fn toSlice(self: *const Self, allocator: std.mem.Allocator) ![]T {
            const slice = try allocator.alloc(T, self.length);
            errdefer allocator.free(slice);

            var i: usize = 0;
            var it = self.iterator();
            while (it.next()) |value| {
                slice[i] = value;
                i += 1;
            }

            return slice;
        }

        // -- Debug & Validation --

        /// Format the list for debugging
        pub fn format(
            self: *const Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;

            try writer.writeAll("UnrolledLinkedList(");
            try writer.print("len={d}, nodes=", .{self.length});

            var node_count: usize = 0;
            var current = self.head;
            try writer.writeAll("[");
            while (current) |node| {
                if (node_count > 0) try writer.writeAll(", ");
                try writer.print("{d}", .{node.count});
                current = node.next;
                node_count += 1;
            }
            try writer.writeAll("])");
        }

        /// Validate internal invariants
        /// Time: O(n/N) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            if (self.head == null) {
                if (self.tail != null) return error.TailWithoutHead;
                if (self.length != 0) return error.LengthMismatch;
                return;
            }

            // Count actual elements
            var actual_length: usize = 0;
            var node_count: usize = 0;
            var current = self.head;
            var prev: ?*Node = null;

            while (current) |node| {
                // No empty nodes should exist in the middle
                if (node.isEmpty() and node != self.head) {
                    return error.EmptyNodeInList;
                }

                // Count should be within bounds
                if (node.count > N) return error.NodeCountExceedsCapacity;

                actual_length += node.count;
                node_count += 1;

                // Check tail pointer
                if (node.next == null and self.tail != node) {
                    return error.TailPointerIncorrect;
                }

                prev = node;
                current = node.next;
            }

            if (actual_length != self.length) return error.LengthMismatch;
        }

        // -- Private Helpers --

        /// Split a full node into two nodes
        fn splitNode(self: *Self, node: *Node) !void {
            std.debug.assert(node.isFull());

            const new_node = try self.allocator.create(Node);
            errdefer self.allocator.destroy(new_node);

            new_node.* = .{};

            // Move half of elements to new node
            const split_point = N / 2;
            const move_count = N - split_point;

            var i: usize = 0;
            while (i < move_count) : (i += 1) {
                new_node.elements[i] = node.elements[split_point + i];
            }
            new_node.count = move_count;
            node.count = split_point;

            // Link nodes
            new_node.next = node.next;
            node.next = new_node;

            // Update tail if necessary
            if (self.tail == node) {
                self.tail = new_node;
            }
        }

        /// Try to merge node with its next node if combined size allows
        fn tryMergeNodes(self: *Self, prev_node: ?*Node, node: *Node) !void {
            const next_node = node.next orelse return;

            if (node.count + next_node.count <= N) {
                // Merge next_node into node
                var i: usize = 0;
                while (i < next_node.count) : (i += 1) {
                    node.elements[node.count + i] = next_node.elements[i];
                }
                node.count += next_node.count;
                node.next = next_node.next;

                // Update tail if necessary
                if (self.tail == next_node) {
                    self.tail = node;
                }

                self.allocator.destroy(next_node);
            }

            _ = prev_node; // Not used in current implementation
        }
    };
}

// -- Tests --

test "UnrolledLinkedList: basic append and get" {
    var list = UnrolledLinkedList(i32, 4).init(testing.allocator);
    defer list.deinit();

    try list.append(10);
    try list.append(20);
    try list.append(30);

    try testing.expectEqual(@as(usize, 3), list.count());
    try testing.expectEqual(@as(i32, 10), try list.get(0));
    try testing.expectEqual(@as(i32, 20), try list.get(1));
    try testing.expectEqual(@as(i32, 30), try list.get(2));

    try list.validate();
}

test "UnrolledLinkedList: prepend" {
    var list = UnrolledLinkedList(i32, 4).init(testing.allocator);
    defer list.deinit();

    try list.prepend(30);
    try list.prepend(20);
    try list.prepend(10);

    try testing.expectEqual(@as(usize, 3), list.count());
    try testing.expectEqual(@as(i32, 10), try list.get(0));
    try testing.expectEqual(@as(i32, 20), try list.get(1));
    try testing.expectEqual(@as(i32, 30), try list.get(2));

    try list.validate();
}

test "UnrolledLinkedList: node splitting on full" {
    var list = UnrolledLinkedList(i32, 4).init(testing.allocator);
    defer list.deinit();

    // Fill first node to capacity
    try list.append(1);
    try list.append(2);
    try list.append(3);
    try list.append(4);

    // This should create a new node
    try list.append(5);

    try testing.expectEqual(@as(usize, 5), list.count());
    try testing.expectEqual(@as(i32, 1), try list.get(0));
    try testing.expectEqual(@as(i32, 5), try list.get(4));

    try list.validate();
}

test "UnrolledLinkedList: insertAt middle" {
    var list = UnrolledLinkedList(i32, 4).init(testing.allocator);
    defer list.deinit();

    try list.append(10);
    try list.append(30);
    try list.insertAt(1, 20);

    try testing.expectEqual(@as(usize, 3), list.count());
    try testing.expectEqual(@as(i32, 10), try list.get(0));
    try testing.expectEqual(@as(i32, 20), try list.get(1));
    try testing.expectEqual(@as(i32, 30), try list.get(2));

    try list.validate();
}

test "UnrolledLinkedList: removeAt" {
    var list = UnrolledLinkedList(i32, 4).init(testing.allocator);
    defer list.deinit();

    try list.append(10);
    try list.append(20);
    try list.append(30);
    try list.append(40);

    const removed = try list.removeAt(1);
    try testing.expectEqual(@as(i32, 20), removed);
    try testing.expectEqual(@as(usize, 3), list.count());
    try testing.expectEqual(@as(i32, 10), try list.get(0));
    try testing.expectEqual(@as(i32, 30), try list.get(1));
    try testing.expectEqual(@as(i32, 40), try list.get(2));

    try list.validate();
}

test "UnrolledLinkedList: popFirst and popLast" {
    var list = UnrolledLinkedList(i32, 4).init(testing.allocator);
    defer list.deinit();

    try list.append(10);
    try list.append(20);
    try list.append(30);

    try testing.expectEqual(@as(i32, 10), try list.popFirst());
    try testing.expectEqual(@as(i32, 30), try list.popLast());
    try testing.expectEqual(@as(usize, 1), list.count());
    try testing.expectEqual(@as(i32, 20), try list.get(0));

    try list.validate();
}

test "UnrolledLinkedList: iterator" {
    var list = UnrolledLinkedList(i32, 4).init(testing.allocator);
    defer list.deinit();

    try list.append(10);
    try list.append(20);
    try list.append(30);
    try list.append(40);
    try list.append(50);

    var expected: i32 = 10;
    var it = list.iterator();
    while (it.next()) |value| {
        try testing.expectEqual(expected, value);
        expected += 10;
    }

    try testing.expectEqual(@as(i32, 60), expected);
}

test "UnrolledLinkedList: clear" {
    var list = UnrolledLinkedList(i32, 4).init(testing.allocator);
    defer list.deinit();

    try list.append(10);
    try list.append(20);
    try list.append(30);

    list.clear();

    try testing.expectEqual(@as(usize, 0), list.count());
    try testing.expect(list.isEmpty());

    try list.validate();
}

test "UnrolledLinkedList: fromSlice and toSlice" {
    const data = [_]i32{ 1, 2, 3, 4, 5 };
    var list = try UnrolledLinkedList(i32, 4).fromSlice(testing.allocator, &data);
    defer list.deinit();

    try testing.expectEqual(@as(usize, 5), list.count());

    const slice = try list.toSlice(testing.allocator);
    defer testing.allocator.free(slice);

    try testing.expectEqualSlices(i32, &data, slice);

    try list.validate();
}

test "UnrolledLinkedList: clone" {
    var original = UnrolledLinkedList(i32, 4).init(testing.allocator);
    defer original.deinit();

    try original.append(10);
    try original.append(20);
    try original.append(30);

    var cloned = try original.clone();
    defer cloned.deinit();

    try testing.expectEqual(original.count(), cloned.count());
    try testing.expectEqual(@as(i32, 10), try cloned.get(0));
    try testing.expectEqual(@as(i32, 20), try cloned.get(1));
    try testing.expectEqual(@as(i32, 30), try cloned.get(2));

    try cloned.validate();
}

test "UnrolledLinkedList: stress test with node splitting and merging" {
    var list = UnrolledLinkedList(i32, 8).init(testing.allocator);
    defer list.deinit();

    // Insert many elements to trigger multiple splits
    var i: i32 = 0;
    while (i < 100) : (i += 1) {
        try list.append(i);
    }

    try testing.expectEqual(@as(usize, 100), list.count());

    // Verify all elements
    i = 0;
    while (i < 100) : (i += 1) {
        try testing.expectEqual(i, try list.get(@intCast(i)));
    }

    // Remove elements to trigger merges
    while (list.count() > 10) {
        _ = try list.removeAt(list.count() / 2);
    }

    try testing.expectEqual(@as(usize, 10), list.count());

    try list.validate();
}

test "UnrolledLinkedList: memory leak check" {
    var list = UnrolledLinkedList(i32, 4).init(testing.allocator);
    defer list.deinit();

    var i: i32 = 0;
    while (i < 50) : (i += 1) {
        try list.append(i);
    }

    while (!list.isEmpty()) {
        _ = try list.popFirst();
    }

    try list.validate();
}

test "UnrolledLinkedList: edge case - single element" {
    var list = UnrolledLinkedList(i32, 4).init(testing.allocator);
    defer list.deinit();

    try list.append(42);
    try testing.expectEqual(@as(usize, 1), list.count());
    try testing.expectEqual(@as(i32, 42), try list.get(0));

    const removed = try list.popFirst();
    try testing.expectEqual(@as(i32, 42), removed);
    try testing.expect(list.isEmpty());

    try list.validate();
}

test "UnrolledLinkedList: error cases" {
    var list = UnrolledLinkedList(i32, 4).init(testing.allocator);
    defer list.deinit();

    // Get out of bounds
    try testing.expectError(error.IndexOutOfBounds, list.get(0));

    try list.append(10);

    // Remove out of bounds
    try testing.expectError(error.IndexOutOfBounds, list.removeAt(5));

    // Insert out of bounds
    try testing.expectError(error.IndexOutOfBounds, list.insertAt(10, 99));

    try list.validate();
}
