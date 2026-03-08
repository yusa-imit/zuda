const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// PairingHeap is a heap data structure with simple implementation and competitive performance.
/// It uses a multi-way tree structure with lazy merging and two-pass pairing for delete-min.
///
/// Time complexity:
/// - insert: O(1)
/// - findMin: O(1)
/// - extractMin: O(log n) amortized
/// - decreaseKey: O(log n) amortized (conjectured, proven O(2^sqrt(log log n)))
/// - merge: O(1)
///
/// Space: O(n)
///
/// Use cases:
/// - Priority queues with frequent insertions and merges
/// - Graph algorithms (Dijkstra, Prim)
/// - Simpler alternative to Fibonacci heap with similar performance
///
pub fn PairingHeap(comptime T: type, comptime Context: type, comptime lessThan: fn (ctx: Context, a: T, b: T) bool) type {
    return struct {
        const Self = @This();

        const Node = struct {
            data: T,
            parent: ?*Node = null,
            leftmost_child: ?*Node = null,
            next_sibling: ?*Node = null,
            prev_sibling: ?*Node = null, // For efficient removal

            fn init(data: T) Node {
                return .{
                    .data = data,
                };
            }

            fn addChild(parent_node: *Node, child: *Node) void {
                child.parent = parent_node;
                child.next_sibling = parent_node.leftmost_child;
                child.prev_sibling = null;
                if (parent_node.leftmost_child) |old_child| {
                    old_child.prev_sibling = child;
                }
                parent_node.leftmost_child = child;
            }

            fn removeFromSiblings(node: *Node) void {
                if (node.prev_sibling) |prev| {
                    prev.next_sibling = node.next_sibling;
                } else if (node.parent) |parent| {
                    parent.leftmost_child = node.next_sibling;
                }
                if (node.next_sibling) |next| {
                    next.prev_sibling = node.prev_sibling;
                }
                node.prev_sibling = null;
                node.next_sibling = null;
                node.parent = null;
            }
        };

        allocator: Allocator,
        context: Context,
        root: ?*Node = null,
        size: usize = 0,

        // -- Lifecycle --

        /// Initialize an empty pairing heap.
        pub fn init(allocator: Allocator, context: Context) Self {
            return .{
                .allocator = allocator,
                .context = context,
            };
        }

        /// Deinitialize the heap and free all nodes.
        pub fn deinit(self: *Self) void {
            if (self.root) |r| {
                self.freeNode(r);
            }
            self.* = undefined;
        }

        fn freeNode(self: *Self, node: *Node) void {
            var child = node.leftmost_child;
            while (child) |c| {
                const next = c.next_sibling;
                self.freeNode(c);
                child = next;
            }
            self.allocator.destroy(node);
        }

        /// Clone the heap.
        pub fn clone(self: *const Self) !Self {
            var new = Self.init(self.allocator, self.context);
            if (self.root) |r| {
                new.root = try self.cloneNode(r);
                new.size = self.size;
            }
            return new;
        }

        fn cloneNode(self: *const Self, node: *const Node) !*Node {
            const new_node = try self.allocator.create(Node);
            new_node.* = Node.init(node.data);

            var child = node.leftmost_child;
            var prev_sibling: ?*Node = null;
            while (child) |c| {
                const new_child = try self.cloneNode(c);
                new_child.parent = new_node;
                if (prev_sibling) |prev| {
                    prev.next_sibling = new_child;
                    new_child.prev_sibling = prev;
                } else {
                    new_node.leftmost_child = new_child;
                }
                prev_sibling = new_child;
                child = c.next_sibling;
            }

            return new_node;
        }

        // -- Capacity --

        /// Return the number of elements in the heap.
        pub fn count(self: *const Self) usize {
            return self.size;
        }

        /// Check if the heap is empty.
        pub fn isEmpty(self: *const Self) bool {
            return self.size == 0;
        }

        // -- Modification --

        /// Insert a new element into the heap.
        /// Time: O(1) | Space: O(1)
        pub fn insert(self: *Self, data: T) !*Node {
            const node = try self.allocator.create(Node);
            node.* = Node.init(data);
            self.root = self.meld(self.root, node);
            self.size += 1;
            return node;
        }

        /// Remove and return the minimum element.
        /// Time: O(log n) amortized | Space: O(1)
        pub fn extractMin(self: *Self) ?T {
            const min_node = self.root orelse return null;
            const data = min_node.data;

            // Two-pass pairing for children
            self.root = self.combineSiblings(min_node.leftmost_child);
            self.allocator.destroy(min_node);
            self.size -= 1;

            return data;
        }

        /// Decrease the value of a node.
        /// Time: O(log n) amortized | Space: O(1)
        pub fn decreaseKey(self: *Self, node: *Node, new_data: T) void {
            node.data = new_data;
            if (node.parent) |_| {
                node.removeFromSiblings();
                self.root = self.meld(self.root, node);
            }
        }

        /// Merge another heap into this one.
        /// Time: O(1) | Space: O(1)
        pub fn merge(self: *Self, other: *Self) void {
            self.root = self.meld(self.root, other.root);
            self.size += other.size;
            other.root = null;
            other.size = 0;
        }

        // -- Lookup --

        /// Return the minimum element without removing it.
        /// Time: O(1) | Space: O(1)
        pub fn findMin(self: *const Self) ?T {
            return if (self.root) |r| r.data else null;
        }

        // -- Private Helpers --

        /// Meld two nodes, returning the new root.
        fn meld(self: *Self, a: ?*Node, b: ?*Node) ?*Node {
            if (a == null) return b;
            if (b == null) return a;

            const node_a = a.?;
            const node_b = b.?;

            if (lessThan(self.context, node_a.data, node_b.data)) {
                Node.addChild(node_a, node_b);
                return node_a;
            } else {
                Node.addChild(node_b, node_a);
                return node_b;
            }
        }

        /// Two-pass pairing algorithm for combining siblings.
        /// First pass: pair adjacent siblings left-to-right.
        /// Second pass: meld pairs right-to-left.
        fn combineSiblings(self: *Self, first_child: ?*Node) ?*Node {
            if (first_child == null) return null;

            // First pass: pair adjacent siblings
            var pairs = std.ArrayList(?*Node).init(self.allocator);
            defer pairs.deinit();

            var current = first_child;
            while (current) |c| {
                const next = c.next_sibling;
                c.parent = null;
                c.prev_sibling = null;
                c.next_sibling = null;

                if (next) |n| {
                    const next_next = n.next_sibling;
                    n.parent = null;
                    n.prev_sibling = null;
                    n.next_sibling = null;

                    pairs.append(self.meld(c, n)) catch unreachable;
                    current = next_next;
                } else {
                    pairs.append(c) catch unreachable;
                    current = null;
                }
            }

            // Second pass: meld pairs right-to-left
            var result = pairs.pop();
            while (pairs.items.len > 0) {
                const pair = pairs.pop();
                result = self.meld(result, pair);
            }

            return result;
        }

        // -- Debug --

        /// Validate heap invariants.
        pub fn validate(self: *const Self) !void {
            var actual_size: usize = 0;
            if (self.root) |r| {
                try self.validateNode(r, null, &actual_size);
            }
            if (actual_size != self.size) {
                return error.InvalidSize;
            }
        }

        fn validateNode(self: *const Self, node: *const Node, parent: ?*const Node, size: *usize) !void {
            size.* += 1;

            // Parent pointer check
            if (node.parent != parent) {
                return error.InvalidParent;
            }

            // Heap property: parent <= children
            if (parent) |p| {
                if (lessThan(self.context, node.data, p.data)) {
                    return error.HeapPropertyViolation;
                }
            }

            // Validate children
            var child = node.leftmost_child;
            var prev: ?*const Node = null;
            while (child) |c| {
                if (c.prev_sibling != prev) {
                    return error.InvalidSiblingLinks;
                }
                try self.validateNode(c, node, size);
                prev = c;
                child = c.next_sibling;
            }
        }

        /// Format the heap for debugging.
        pub fn format(self: *const Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;
            try writer.print("PairingHeap(size={})", .{self.size});
        }
    };
}

// -- Tests --

fn testLessThan(_: void, a: i32, b: i32) bool {
    return a < b;
}

test "PairingHeap: basic operations" {
    var heap = PairingHeap(i32, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    try testing.expect(heap.isEmpty());
    try testing.expectEqual(@as(usize, 0), heap.count());
    try testing.expectEqual(@as(?i32, null), heap.findMin());

    _ = try heap.insert(5);
    try testing.expect(!heap.isEmpty());
    try testing.expectEqual(@as(usize, 1), heap.count());
    try testing.expectEqual(@as(?i32, 5), heap.findMin());
    try heap.validate();

    _ = try heap.insert(3);
    try testing.expectEqual(@as(?i32, 3), heap.findMin());
    try heap.validate();

    _ = try heap.insert(7);
    try testing.expectEqual(@as(?i32, 3), heap.findMin());
    try heap.validate();

    try testing.expectEqual(@as(?i32, 3), heap.extractMin());
    try testing.expectEqual(@as(?i32, 5), heap.findMin());
    try heap.validate();

    try testing.expectEqual(@as(?i32, 5), heap.extractMin());
    try testing.expectEqual(@as(?i32, 7), heap.findMin());
    try heap.validate();

    try testing.expectEqual(@as(?i32, 7), heap.extractMin());
    try testing.expect(heap.isEmpty());
}

test "PairingHeap: decreaseKey" {
    var heap = PairingHeap(i32, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    const n1 = try heap.insert(10);
    const n2 = try heap.insert(20);
    const n3 = try heap.insert(30);

    try testing.expectEqual(@as(?i32, 10), heap.findMin());

    heap.decreaseKey(n3, 5);
    try testing.expectEqual(@as(?i32, 5), heap.findMin());
    try heap.validate();

    heap.decreaseKey(n2, 3);
    try testing.expectEqual(@as(?i32, 3), heap.findMin());
    try heap.validate();

    heap.decreaseKey(n1, 1);
    try testing.expectEqual(@as(?i32, 1), heap.findMin());
    try heap.validate();

    try testing.expectEqual(@as(?i32, 1), heap.extractMin());
    try testing.expectEqual(@as(?i32, 3), heap.extractMin());
    try testing.expectEqual(@as(?i32, 5), heap.extractMin());
    try testing.expect(heap.isEmpty());
}

test "PairingHeap: merge" {
    var heap1 = PairingHeap(i32, void, testLessThan).init(testing.allocator, {});
    defer heap1.deinit();

    var heap2 = PairingHeap(i32, void, testLessThan).init(testing.allocator, {});
    defer heap2.deinit();

    _ = try heap1.insert(1);
    _ = try heap1.insert(3);
    _ = try heap1.insert(5);

    _ = try heap2.insert(2);
    _ = try heap2.insert(4);
    _ = try heap2.insert(6);

    heap1.merge(&heap2);
    try testing.expectEqual(@as(usize, 6), heap1.count());
    try testing.expectEqual(@as(usize, 0), heap2.count());
    try heap1.validate();

    try testing.expectEqual(@as(?i32, 1), heap1.extractMin());
    try testing.expectEqual(@as(?i32, 2), heap1.extractMin());
    try testing.expectEqual(@as(?i32, 3), heap1.extractMin());
    try testing.expectEqual(@as(?i32, 4), heap1.extractMin());
    try testing.expectEqual(@as(?i32, 5), heap1.extractMin());
    try testing.expectEqual(@as(?i32, 6), heap1.extractMin());
    try testing.expect(heap1.isEmpty());
}

test "PairingHeap: sorted insertion" {
    var heap = PairingHeap(i32, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    const n = 100;
    var i: i32 = 0;
    while (i < n) : (i += 1) {
        _ = try heap.insert(i);
    }

    try testing.expectEqual(@as(usize, n), heap.count());
    try heap.validate();

    i = 0;
    while (i < n) : (i += 1) {
        try testing.expectEqual(@as(?i32, i), heap.extractMin());
    }
    try testing.expect(heap.isEmpty());
}

test "PairingHeap: reverse sorted insertion" {
    var heap = PairingHeap(i32, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    const n = 100;
    var i: i32 = n - 1;
    while (i >= 0) : (i -= 1) {
        _ = try heap.insert(i);
    }

    try testing.expectEqual(@as(usize, n), heap.count());
    try heap.validate();

    i = 0;
    while (i < n) : (i += 1) {
        try testing.expectEqual(@as(?i32, i), heap.extractMin());
    }
    try testing.expect(heap.isEmpty());
}

test "PairingHeap: random insertion" {
    var heap = PairingHeap(i32, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const n = 1000;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const val = random.intRangeAtMost(i32, 0, 9999);
        _ = try heap.insert(val);
    }

    try testing.expectEqual(@as(usize, n), heap.count());
    try heap.validate();

    var prev: i32 = -1;
    while (!heap.isEmpty()) {
        const val = heap.extractMin().?;
        try testing.expect(val >= prev);
        prev = val;
    }
}

test "PairingHeap: clone" {
    var heap = PairingHeap(i32, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    _ = try heap.insert(5);
    _ = try heap.insert(3);
    _ = try heap.insert(7);

    var cloned = try heap.clone();
    defer cloned.deinit();

    try testing.expectEqual(heap.count(), cloned.count());
    try testing.expectEqual(heap.findMin(), cloned.findMin());
    try cloned.validate();

    try testing.expectEqual(@as(?i32, 3), heap.extractMin());
    try testing.expectEqual(@as(?i32, 3), cloned.extractMin());

    try testing.expectEqual(@as(?i32, 5), heap.extractMin());
    try testing.expectEqual(@as(?i32, 5), cloned.extractMin());

    try testing.expectEqual(@as(?i32, 7), heap.extractMin());
    try testing.expectEqual(@as(?i32, 7), cloned.extractMin());
}

test "PairingHeap: stress test with decreaseKey" {
    var heap = PairingHeap(i32, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    var prng = std.Random.DefaultPrng.init(1337);
    const random = prng.random();

    var nodes = std.ArrayList(*PairingHeap(i32, void, testLessThan).Node).init(testing.allocator);
    defer nodes.deinit();

    const n = 500;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const val = random.intRangeAtMost(i32, 0, 9999);
        const node = try heap.insert(val);
        try nodes.append(node);
    }

    // Random decreaseKey operations
    i = 0;
    while (i < 100) : (i += 1) {
        const idx = random.uintLessThan(usize, nodes.items.len);
        const node = nodes.items[idx];
        const new_val = random.intRangeAtMost(i32, 0, node.data);
        heap.decreaseKey(node, new_val);
    }

    try heap.validate();

    var prev: i32 = -1;
    while (!heap.isEmpty()) {
        const val = heap.extractMin().?;
        try testing.expect(val >= prev);
        prev = val;
    }
}

test "PairingHeap: empty operations" {
    var heap = PairingHeap(i32, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    try testing.expectEqual(@as(?i32, null), heap.extractMin());
    try testing.expectEqual(@as(?i32, null), heap.findMin());
    try heap.validate();
}

test "PairingHeap: single element" {
    var heap = PairingHeap(i32, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    _ = try heap.insert(42);
    try testing.expectEqual(@as(?i32, 42), heap.findMin());
    try heap.validate();

    try testing.expectEqual(@as(?i32, 42), heap.extractMin());
    try testing.expect(heap.isEmpty());
}

test "PairingHeap: duplicates" {
    var heap = PairingHeap(i32, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    _ = try heap.insert(5);
    _ = try heap.insert(5);
    _ = try heap.insert(5);

    try testing.expectEqual(@as(usize, 3), heap.count());
    try heap.validate();

    try testing.expectEqual(@as(?i32, 5), heap.extractMin());
    try testing.expectEqual(@as(?i32, 5), heap.extractMin());
    try testing.expectEqual(@as(?i32, 5), heap.extractMin());
    try testing.expect(heap.isEmpty());
}

test "PairingHeap: merge empty heaps" {
    var heap1 = PairingHeap(i32, void, testLessThan).init(testing.allocator, {});
    defer heap1.deinit();

    var heap2 = PairingHeap(i32, void, testLessThan).init(testing.allocator, {});
    defer heap2.deinit();

    heap1.merge(&heap2);
    try testing.expect(heap1.isEmpty());
    try testing.expect(heap2.isEmpty());
}

test "PairingHeap: merge with empty" {
    var heap1 = PairingHeap(i32, void, testLessThan).init(testing.allocator, {});
    defer heap1.deinit();

    var heap2 = PairingHeap(i32, void, testLessThan).init(testing.allocator, {});
    defer heap2.deinit();

    _ = try heap1.insert(5);
    _ = try heap1.insert(3);

    heap1.merge(&heap2);
    try testing.expectEqual(@as(usize, 2), heap1.count());
    try testing.expectEqual(@as(?i32, 3), heap1.findMin());
}

test "PairingHeap: memory leak detection" {
    var heap = PairingHeap(i32, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        _ = try heap.insert(@intCast(i));
    }

    while (!heap.isEmpty()) {
        _ = heap.extractMin();
    }
}

test "PairingHeap: power of two insertion" {
    var heap = PairingHeap(i32, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    var i: usize = 0;
    while (i < 16) : (i += 1) {
        _ = try heap.insert(@intCast(i));
    }

    try testing.expectEqual(@as(usize, 16), heap.count());
    try heap.validate();

    i = 0;
    while (i < 16) : (i += 1) {
        try testing.expectEqual(@as(?i32, @intCast(i)), heap.extractMin());
    }
}
