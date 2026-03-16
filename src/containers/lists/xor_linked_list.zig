//! XOR Linked List - Memory-efficient doubly linked list
//!
//! An XOR linked list uses the XOR of previous and next pointers to save memory.
//! Instead of storing two pointers per node (prev and next), each node stores
//! only one pointer: the XOR of prev and next addresses.
//!
//! Space complexity: O(n) with half the pointer overhead of standard doubly linked list
//! Traversal requires keeping track of the previous node to compute the next node.
//!
//! Note: This is primarily useful in memory-constrained environments. For most use cases,
//! a standard doubly linked list is more practical due to simpler implementation and
//! better cache locality.

const std = @import("std");

/// XOR Linked List implementation
/// Time: insert/remove O(1) at known position, search O(n)
/// Space: O(n) with half the pointer overhead of doubly linked list
pub fn XorLinkedList(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const Node = struct {
            data: T,
            // XOR of addresses of previous and next nodes
            xor_ptr: usize,

            fn init(data: T) Node {
                return .{
                    .data = data,
                    .xor_ptr = 0,
                };
            }
        };

        pub const Iterator = struct {
            list: *const Self,
            current: ?*Node,
            prev: ?*Node,

            /// Time: O(1) | Space: O(1)
            pub fn next(self: *Iterator) ?T {
                const curr = self.current orelse return null;
                const result = curr.data;

                // Compute next node: next = curr.xor_ptr XOR prev
                const prev_addr = if (self.prev) |p| @intFromPtr(p) else 0;
                const next_addr = curr.xor_ptr ^ prev_addr;

                self.prev = self.current;
                self.current = if (next_addr != 0) @ptrFromInt(next_addr) else null;

                return result;
            }
        };

        allocator: std.mem.Allocator,
        head: ?*Node,
        tail: ?*Node,
        len: usize,

        // -- Lifecycle --

        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .head = null,
                .tail = null,
                .len = 0,
            };
        }

        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            var prev: ?*Node = null;
            var curr = self.head;

            while (curr) |node| {
                const prev_addr = if (prev) |p| @intFromPtr(p) else 0;
                const next_addr = node.xor_ptr ^ prev_addr;
                const next_node = if (next_addr != 0) @as(?*Node, @ptrFromInt(next_addr)) else null;

                self.allocator.destroy(node);

                prev = curr;
                curr = next_node;
            }

            self.head = null;
            self.tail = null;
            self.len = 0;
        }

        // -- Capacity --

        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.len;
        }

        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.len == 0;
        }

        // -- Modification --

        /// Insert at the front of the list
        /// Time: O(1) | Space: O(1) amortized
        pub fn pushFront(self: *Self, data: T) !void {
            const new_node = try self.allocator.create(Node);
            new_node.* = Node.init(data);

            if (self.head) |old_head| {
                // new_node.xor_ptr = 0 XOR old_head = old_head
                new_node.xor_ptr = @intFromPtr(old_head);

                // Update old_head.xor_ptr: old_prev XOR old_next becomes new_node XOR old_next
                // Since old_prev was null (0), old_head.xor_ptr was just old_next
                // New value: new_node XOR old_next = new_node XOR (0 XOR old_next) = new_node XOR old_head.xor_ptr
                old_head.xor_ptr ^= @intFromPtr(new_node);

                self.head = new_node;
            } else {
                // Empty list
                self.head = new_node;
                self.tail = new_node;
            }

            self.len += 1;
        }

        /// Insert at the back of the list
        /// Time: O(1) | Space: O(1) amortized
        pub fn pushBack(self: *Self, data: T) !void {
            const new_node = try self.allocator.create(Node);
            new_node.* = Node.init(data);

            if (self.tail) |old_tail| {
                // new_node.xor_ptr = old_tail XOR 0 = old_tail
                new_node.xor_ptr = @intFromPtr(old_tail);

                // Update old_tail.xor_ptr: old_prev XOR old_next becomes old_prev XOR new_node
                // Since old_next was null (0), old_tail.xor_ptr was just old_prev
                // New value: old_prev XOR new_node = old_tail.xor_ptr XOR new_node
                old_tail.xor_ptr ^= @intFromPtr(new_node);

                self.tail = new_node;
            } else {
                // Empty list
                self.head = new_node;
                self.tail = new_node;
            }

            self.len += 1;
        }

        /// Remove from the front of the list
        /// Time: O(1) | Space: O(1)
        pub fn popFront(self: *Self) ?T {
            const node = self.head orelse return null;
            const result = node.data;

            // Get next node: head.xor_ptr XOR 0 = next
            const next_addr = node.xor_ptr;
            const next_node = if (next_addr != 0) @as(?*Node, @ptrFromInt(next_addr)) else null;

            if (next_node) |next| {
                // Update next.xor_ptr: old_next.xor_ptr was head XOR next_next
                // New value should be 0 XOR next_next = next_next
                // So: next.xor_ptr XOR head = next_next
                next.xor_ptr ^= @intFromPtr(node);

                self.head = next;
            } else {
                // List becomes empty
                self.head = null;
                self.tail = null;
            }

            self.allocator.destroy(node);
            self.len -= 1;

            return result;
        }

        /// Remove from the back of the list
        /// Time: O(1) | Space: O(1)
        pub fn popBack(self: *Self) ?T {
            const node = self.tail orelse return null;
            const result = node.data;

            // Get previous node: tail.xor_ptr XOR 0 = prev
            const prev_addr = node.xor_ptr;
            const prev_node = if (prev_addr != 0) @as(?*Node, @ptrFromInt(prev_addr)) else null;

            if (prev_node) |prev| {
                // Update prev.xor_ptr: old_prev.xor_ptr was prev_prev XOR tail
                // New value should be prev_prev XOR 0 = prev_prev
                // So: prev.xor_ptr XOR tail = prev_prev
                prev.xor_ptr ^= @intFromPtr(node);

                self.tail = prev;
            } else {
                // List becomes empty
                self.head = null;
                self.tail = null;
            }

            self.allocator.destroy(node);
            self.len -= 1;

            return result;
        }

        // -- Iteration --

        /// Time: O(1) to create iterator, O(n) to traverse all | Space: O(1)
        pub fn iterator(self: *const Self) Iterator {
            return .{
                .list = self,
                .current = self.head,
                .prev = null,
            };
        }

        // -- Debug --

        /// Validate internal invariants
        /// Time: O(n) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            if (self.len == 0) {
                if (self.head != null or self.tail != null) {
                    return error.InvalidEmptyState;
                }
                return;
            }

            if (self.head == null or self.tail == null) {
                return error.InvalidNonEmptyState;
            }

            // Count nodes by traversing
            var node_count: usize = 0;
            var prev: ?*Node = null;
            var curr = self.head;

            while (curr) |node| : (node_count += 1) {
                const prev_addr = if (prev) |p| @intFromPtr(p) else 0;
                const next_addr = node.xor_ptr ^ prev_addr;
                const next_node = if (next_addr != 0) @as(?*Node, @ptrFromInt(next_addr)) else null;

                prev = curr;
                curr = next_node;
            }

            if (node_count != self.len) {
                return error.LengthMismatch;
            }
        }
    };
}

// -- Tests --

test "XorLinkedList: basic operations" {
    const allocator = std.testing.allocator;

    var list = XorLinkedList(i32).init(allocator);
    defer list.deinit();

    try std.testing.expect(list.isEmpty());
    try std.testing.expectEqual(@as(usize, 0), list.count());

    // Push front
    try list.pushFront(10);
    try list.pushFront(20);
    try list.pushFront(30);

    try std.testing.expectEqual(@as(usize, 3), list.count());
    try std.testing.expect(!list.isEmpty());

    // Pop front should give: 30, 20, 10
    try std.testing.expectEqual(@as(i32, 30), list.popFront().?);
    try std.testing.expectEqual(@as(i32, 20), list.popFront().?);
    try std.testing.expectEqual(@as(i32, 10), list.popFront().?);
    try std.testing.expectEqual(@as(?i32, null), list.popFront());

    try std.testing.expect(list.isEmpty());
}

test "XorLinkedList: push/pop back" {
    const allocator = std.testing.allocator;

    var list = XorLinkedList(i32).init(allocator);
    defer list.deinit();

    // Push back
    try list.pushBack(1);
    try list.pushBack(2);
    try list.pushBack(3);

    try std.testing.expectEqual(@as(usize, 3), list.count());

    // Pop back should give: 3, 2, 1
    try std.testing.expectEqual(@as(i32, 3), list.popBack().?);
    try std.testing.expectEqual(@as(i32, 2), list.popBack().?);
    try std.testing.expectEqual(@as(i32, 1), list.popBack().?);
    try std.testing.expectEqual(@as(?i32, null), list.popBack());
}

test "XorLinkedList: mixed operations" {
    const allocator = std.testing.allocator;

    var list = XorLinkedList(i32).init(allocator);
    defer list.deinit();

    try list.pushFront(2);
    try list.pushBack(3);
    try list.pushFront(1);
    try list.pushBack(4);

    // List should be: 1 <-> 2 <-> 3 <-> 4

    try std.testing.expectEqual(@as(usize, 4), list.count());

    try std.testing.expectEqual(@as(i32, 1), list.popFront().?);
    try std.testing.expectEqual(@as(i32, 4), list.popBack().?);
    try std.testing.expectEqual(@as(i32, 2), list.popFront().?);
    try std.testing.expectEqual(@as(i32, 3), list.popBack().?);

    try std.testing.expect(list.isEmpty());
}

test "XorLinkedList: iterator" {
    const allocator = std.testing.allocator;

    var list = XorLinkedList(i32).init(allocator);
    defer list.deinit();

    try list.pushBack(1);
    try list.pushBack(2);
    try list.pushBack(3);
    try list.pushBack(4);

    var it = list.iterator();
    try std.testing.expectEqual(@as(i32, 1), it.next().?);
    try std.testing.expectEqual(@as(i32, 2), it.next().?);
    try std.testing.expectEqual(@as(i32, 3), it.next().?);
    try std.testing.expectEqual(@as(i32, 4), it.next().?);
    try std.testing.expectEqual(@as(?i32, null), it.next());
}

test "XorLinkedList: validate" {
    const allocator = std.testing.allocator;

    var list = XorLinkedList(i32).init(allocator);
    defer list.deinit();

    try std.testing.expectEqual(0, list.len);
    try list.validate();

    try list.pushBack(1);
    try list.pushBack(2);
    try list.pushBack(3);
    try std.testing.expectEqual(3, list.len);

    try list.validate();

    const first = list.popFront();
    try std.testing.expectEqual(1, first);
    try std.testing.expectEqual(2, list.len);
    try list.validate();

    const last = list.popBack();
    try std.testing.expectEqual(3, last);
    try std.testing.expectEqual(1, list.len);
    try list.validate();

    const only = list.popFront();
    try std.testing.expectEqual(2, only);
    try std.testing.expectEqual(0, list.len);
    try list.validate();
}

test "XorLinkedList: stress test" {
    const allocator = std.testing.allocator;

    var list = XorLinkedList(usize).init(allocator);
    defer list.deinit();

    // Insert 1000 elements
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        if (i % 2 == 0) {
            try list.pushBack(i);
        } else {
            try list.pushFront(i);
        }
    }

    try std.testing.expectEqual(@as(usize, 1000), list.count());
    try list.validate();

    // Remove all elements
    while (!list.isEmpty()) {
        if (list.count() % 2 == 0) {
            _ = list.popFront();
        } else {
            _ = list.popBack();
        }
    }

    try std.testing.expect(list.isEmpty());
    try list.validate();
}

test "XorLinkedList: memory leak detection" {
    const allocator = std.testing.allocator;

    var list = XorLinkedList(i32).init(allocator);
    defer list.deinit();

    try list.pushBack(1);
    try list.pushBack(2);
    try list.pushBack(3);
    try list.pushBack(4);
    try list.pushBack(5);
    try std.testing.expectEqual(5, list.len);

    // Verify all values are present
    var iter = list.iterator();
    var expected: i32 = 1;
    while (iter.next()) |value| {
        try std.testing.expectEqual(expected, value);
        expected += 1;
    }
    try std.testing.expectEqual(6, expected);

    // deinit will be called by defer, which should free all nodes
}

test "XorLinkedList: single element" {
    const allocator = std.testing.allocator;

    var list = XorLinkedList(i32).init(allocator);
    defer list.deinit();

    try list.pushFront(42);
    try std.testing.expectEqual(@as(usize, 1), list.count());
    try list.validate();

    try std.testing.expectEqual(@as(i32, 42), list.popBack().?);
    try std.testing.expect(list.isEmpty());
    try list.validate();
}
