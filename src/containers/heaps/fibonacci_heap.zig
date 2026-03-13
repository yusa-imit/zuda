const std = @import("std");
const Allocator = std.mem.Allocator;

/// Fibonacci Heap - A heap data structure with efficient amortized operations.
///
/// Features:
/// - O(1) amortized insert, findMin, merge, decreaseKey
/// - O(log n) amortized extractMin
/// - Lazy consolidation for efficient batch operations
/// - Mark-based cascading cuts for balancing
///
/// Use cases:
/// - Priority queues for graph algorithms (Dijkstra, Prim)
/// - Situations requiring frequent decrease-key operations
/// - Batch operations with lazy evaluation
///
/// Type Parameters:
/// - T: Element type (must support comparison via Context)
/// - Context: Comparison context type
/// - compareFn: Function to compare two elements (returns std.math.Order)
pub fn FibonacciHeap(
    comptime T: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: T, b: T) std.math.Order,
) type {
    return struct {
        const Self = @This();

        const Node = struct {
            value: T,
            degree: usize,
            marked: bool,
            parent: ?*Node,
            child: ?*Node,
            prev: *Node,
            next: *Node,

            fn init(value: T) Node {
                var node = Node{
                    .value = value,
                    .degree = 0,
                    .marked = false,
                    .parent = null,
                    .child = null,
                    .prev = undefined,
                    .next = undefined,
                };
                node.prev = &node;
                node.next = &node;
                return node;
            }
        };

        allocator: Allocator,
        context: Context,
        min_node: ?*Node,
        node_count: usize,

        // -- Lifecycle --

        /// Initialize an empty Fibonacci heap.
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator, context: Context) Self {
            return Self{
                .allocator = allocator,
                .context = context,
                .min_node = null,
                .node_count = 0,
            };
        }

        /// Destroy the heap and free all nodes.
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.min_node) |min| {
                self.destroyTree(min);
            }
        }

        fn destroyTree(self: *Self, start: *Node) void {
            var current = start;
            const first = start;
            var first_visit = true;

            while (first_visit or current != first) {
                first_visit = false;
                const next = current.next;

                // Recursively destroy children
                if (current.child) |child| {
                    self.destroyTree(child);
                }

                self.allocator.destroy(current);
                current = next;
            }
        }

        // -- Capacity --

        /// Get the number of elements in the heap.
        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.node_count;
        }

        /// Check if the heap is empty.
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.node_count == 0;
        }

        // -- Modification --

        /// Insert a new element into the heap.
        /// Time: O(1) amortized | Space: O(1)
        pub fn insert(self: *Self, value: T) !void {
            const node = try self.allocator.create(Node);
            node.* = Node.init(value);

            if (self.min_node == null) {
                self.min_node = node;
            } else {
                self.insertIntoRootList(node);
                const min = self.min_node.?;
                if (compareFn(self.context, node.value, min.value) == .lt) {
                    self.min_node = node;
                }
            }

            self.node_count += 1;
        }

        /// Remove and return the minimum element.
        /// Time: O(log n) amortized | Space: O(1)
        pub fn extractMin(self: *Self) ?T {
            const min = self.min_node orelse return null;
            const value = min.value;

            // Add all children to root list
            if (min.child) |child| {
                var current = child;
                const first = child;
                var first_visit = true;

                while (first_visit or current != first) {
                    first_visit = false;
                    const next = current.next;
                    current.parent = null;
                    current = next;
                }

                // Merge child list with root list
                self.mergeIntoRootList(child);
            }

            // Remove min from root list
            if (min.next == min) {
                // Last node
                self.min_node = null;
            } else {
                min.prev.next = min.next;
                min.next.prev = min.prev;
                self.min_node = min.next;
                self.consolidate();
            }

            self.allocator.destroy(min);
            self.node_count -= 1;

            return value;
        }

        /// Decrease the value of a specific element.
        /// Time: O(1) amortized | Space: O(1)
        ///
        /// NOTE: This is a simplified version. In practice, you'd need to maintain
        /// external references to nodes to use this efficiently.
        pub fn decreaseKey(self: *Self, node: *Node, new_value: T) !void {
            if (compareFn(self.context, new_value, node.value) != .lt) {
                return error.NewValueNotSmaller;
            }

            node.value = new_value;
            const parent = node.parent;

            if (parent != null and compareFn(self.context, node.value, parent.?.value) == .lt) {
                self.cut(node, parent.?);
                self.cascadingCut(parent.?);
            }

            const min = self.min_node.?;
            if (compareFn(self.context, node.value, min.value) == .lt) {
                self.min_node = node;
            }
        }

        // -- Lookup --

        /// Get the minimum element without removing it.
        /// Time: O(1) | Space: O(1)
        pub fn peekMin(self: *const Self) ?T {
            if (self.min_node) |min| {
                return min.value;
            }
            return null;
        }

        // -- Merge --

        /// Merge another Fibonacci heap into this one.
        /// Time: O(1) | Space: O(1)
        pub fn merge(self: *Self, other: *Self) void {
            if (other.min_node == null) return;

            if (self.min_node == null) {
                self.min_node = other.min_node;
            } else {
                self.mergeIntoRootList(other.min_node.?);
                const min = self.min_node.?;
                const other_min = other.min_node.?;
                if (compareFn(self.context, other_min.value, min.value) == .lt) {
                    self.min_node = other_min;
                }
            }

            self.node_count += other.node_count;
            other.min_node = null;
            other.node_count = 0;
        }

        // -- Debug --

        /// Validate heap invariants.
        /// Time: O(n) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            if (self.min_node == null) {
                if (self.node_count != 0) {
                    return error.InvalidNodeCount;
                }
                return;
            }

            var node_count: usize = 0;
            const min = self.min_node.?;

            // Check root list
            var current = min;
            const first = min;
            var first_visit = true;

            while (first_visit or current != first) {
                first_visit = false;
                node_count += 1;
                node_count += try self.validateTree(current);

                // Check that min_node is indeed minimum
                if (compareFn(self.context, current.value, min.value) == .lt) {
                    return error.InvalidMinNode;
                }

                current = current.next;
            }

            if (node_count != self.node_count) {
                return error.InvalidNodeCount;
            }
        }

        fn validateTree(self: *const Self, node: *const Node) !usize {
            var child_count: usize = 0;

            if (node.child) |child| {
                var current = child;
                const first = child;
                var first_visit = true;

                while (first_visit or current != first) {
                    first_visit = false;

                    // Check parent pointer
                    if (current.parent != node) {
                        return error.InvalidParentPointer;
                    }

                    // Check heap property
                    if (compareFn(self.context, current.value, node.value) == .lt) {
                        return error.HeapPropertyViolation;
                    }

                    child_count += 1;
                    child_count += try self.validateTree(current);
                    current = current.next;
                }
            }

            return child_count;
        }

        // -- Private Helpers --

        fn insertIntoRootList(self: *Self, node: *Node) void {
            const min = self.min_node.?;
            node.prev = min.prev;
            node.next = min;
            min.prev.next = node;
            min.prev = node;
            node.parent = null;
        }

        fn mergeIntoRootList(self: *Self, list: *Node) void {
            const min = self.min_node.?;
            const list_last = list.prev;

            // Connect list to min
            min.prev.next = list;
            list.prev = min.prev;

            // Connect list end to min
            list_last.next = min;
            min.prev = list_last;
        }

        fn consolidate(self: *Self) void {
            if (self.min_node == null) return;

            // Max degree bound: D(n) = O(log n)
            // For Fibonacci heap, max degree is approximately log_φ(n) where φ ≈ 1.618
            const max_degree = @as(usize, @intFromFloat(@ceil(@log2(@as(f64, @floatFromInt(self.node_count))) * 1.5))) + 1;

            var degree_table = self.allocator.alloc(?*Node, max_degree) catch return;
            defer self.allocator.free(degree_table);

            @memset(degree_table, null);

            // Collect root list nodes
            var roots: std.ArrayList(*Node) = .{};
            defer roots.deinit(self.allocator);

            const min = self.min_node.?;
            var current = min;
            const first = min;
            var first_visit = true;

            while (first_visit or current != first) {
                first_visit = false;
                roots.append(self.allocator, current) catch return;
                current = current.next;
            }

            // Consolidate trees
            for (roots.items) |root| {
                var node = root;
                var degree = node.degree;

                while (degree_table[degree] != null) {
                    var other = degree_table[degree].?;
                    if (compareFn(self.context, other.value, node.value) == .lt) {
                        const temp = node;
                        node = other;
                        other = temp;
                    }

                    self.link(other, node);
                    degree_table[degree] = null;
                    degree += 1;
                }

                degree_table[degree] = node;
            }

            // Rebuild root list and find new minimum
            self.min_node = null;
            for (degree_table) |maybe_node| {
                if (maybe_node) |node| {
                    if (self.min_node == null) {
                        node.prev = node;
                        node.next = node;
                        self.min_node = node;
                    } else {
                        self.insertIntoRootList(node);
                        const current_min = self.min_node.?;
                        if (compareFn(self.context, node.value, current_min.value) == .lt) {
                            self.min_node = node;
                        }
                    }
                }
            }
        }

        fn link(self: *Self, child: *Node, parent: *Node) void {
            _ = self;
            // Remove child from root list
            child.prev.next = child.next;
            child.next.prev = child.prev;

            // Make child a child of parent
            if (parent.child == null) {
                parent.child = child;
                child.prev = child;
                child.next = child;
            } else {
                const first_child = parent.child.?;
                child.prev = first_child.prev;
                child.next = first_child;
                first_child.prev.next = child;
                first_child.prev = child;
            }

            child.parent = parent;
            parent.degree += 1;
            child.marked = false;
        }

        fn cut(self: *Self, node: *Node, parent: *Node) void {
            // Remove node from parent's child list
            if (node.next == node) {
                parent.child = null;
            } else {
                if (parent.child == node) {
                    parent.child = node.next;
                }
                node.prev.next = node.next;
                node.next.prev = node.prev;
            }

            parent.degree -= 1;
            node.parent = null;
            node.marked = false;

            // Add node to root list
            self.insertIntoRootList(node);
        }

        fn cascadingCut(self: *Self, node: *Node) void {
            const parent = node.parent orelse return;

            if (!node.marked) {
                node.marked = true;
            } else {
                self.cut(node, parent);
                self.cascadingCut(parent);
            }
        }
    };
}

// -- Tests --

test "FibonacciHeap: basic insert and extract" {
    const IntHeap = FibonacciHeap(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);

    var heap = IntHeap.init(std.testing.allocator, {});
    defer heap.deinit();

    try heap.insert(5);
    try heap.insert(3);
    try heap.insert(7);
    try heap.insert(1);

    try std.testing.expectEqual(@as(usize, 4), heap.count());
    try std.testing.expectEqual(@as(i32, 1), heap.peekMin().?);

    try std.testing.expectEqual(@as(i32, 1), heap.extractMin().?);
    try std.testing.expectEqual(@as(i32, 3), heap.extractMin().?);
    try std.testing.expectEqual(@as(i32, 5), heap.extractMin().?);
    try std.testing.expectEqual(@as(i32, 7), heap.extractMin().?);

    try std.testing.expect(heap.isEmpty());
    try std.testing.expectEqual(@as(?i32, null), heap.extractMin());
}

test "FibonacciHeap: empty heap" {
    const IntHeap = FibonacciHeap(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);

    var heap = IntHeap.init(std.testing.allocator, {});
    defer heap.deinit();

    try std.testing.expect(heap.isEmpty());
    try std.testing.expectEqual(@as(?i32, null), heap.peekMin());
    try std.testing.expectEqual(@as(?i32, null), heap.extractMin());
}

test "FibonacciHeap: single element" {
    const IntHeap = FibonacciHeap(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);

    var heap = IntHeap.init(std.testing.allocator, {});
    defer heap.deinit();

    try heap.insert(42);
    try std.testing.expectEqual(@as(usize, 1), heap.count());
    try std.testing.expectEqual(@as(i32, 42), heap.peekMin().?);
    try std.testing.expectEqual(@as(i32, 42), heap.extractMin().?);
    try std.testing.expect(heap.isEmpty());
}

test "FibonacciHeap: duplicate values" {
    const IntHeap = FibonacciHeap(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);

    var heap = IntHeap.init(std.testing.allocator, {});
    defer heap.deinit();

    try heap.insert(5);
    try heap.insert(5);
    try heap.insert(3);
    try heap.insert(3);

    try std.testing.expectEqual(@as(i32, 3), heap.extractMin().?);
    try std.testing.expectEqual(@as(i32, 3), heap.extractMin().?);
    try std.testing.expectEqual(@as(i32, 5), heap.extractMin().?);
    try std.testing.expectEqual(@as(i32, 5), heap.extractMin().?);
}

test "FibonacciHeap: merge two heaps" {
    const IntHeap = FibonacciHeap(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);

    var heap1 = IntHeap.init(std.testing.allocator, {});
    defer heap1.deinit();

    var heap2 = IntHeap.init(std.testing.allocator, {});
    defer heap2.deinit();

    try heap1.insert(5);
    try heap1.insert(10);
    try heap2.insert(3);
    try heap2.insert(7);

    heap1.merge(&heap2);

    try std.testing.expectEqual(@as(usize, 4), heap1.count());
    try std.testing.expectEqual(@as(usize, 0), heap2.count());

    try std.testing.expectEqual(@as(i32, 3), heap1.extractMin().?);
    try std.testing.expectEqual(@as(i32, 5), heap1.extractMin().?);
    try std.testing.expectEqual(@as(i32, 7), heap1.extractMin().?);
    try std.testing.expectEqual(@as(i32, 10), heap1.extractMin().?);
}

test "FibonacciHeap: merge with empty heap" {
    const IntHeap = FibonacciHeap(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);

    var heap1 = IntHeap.init(std.testing.allocator, {});
    defer heap1.deinit();

    var heap2 = IntHeap.init(std.testing.allocator, {});
    defer heap2.deinit();

    try heap1.insert(5);
    heap1.merge(&heap2);

    try std.testing.expectEqual(@as(usize, 1), heap1.count());
    try std.testing.expectEqual(@as(i32, 5), heap1.extractMin().?);
}

test "FibonacciHeap: stress test with 1000 insertions" {
    const IntHeap = FibonacciHeap(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);

    var heap = IntHeap.init(std.testing.allocator, {});
    defer heap.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Insert 1000 random values
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        const value = random.intRangeAtMost(i32, 0, 10000);
        try heap.insert(value);
    }

    try std.testing.expectEqual(@as(usize, 1000), heap.count());

    // Extract all values and verify they're in ascending order
    var prev = heap.extractMin().?;
    i = 1;
    while (i < 1000) : (i += 1) {
        const current = heap.extractMin().?;
        try std.testing.expect(prev <= current);
        prev = current;
    }

    try std.testing.expect(heap.isEmpty());
}

test "FibonacciHeap: memory leak detection" {
    const IntHeap = FibonacciHeap(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);

    var heap = IntHeap.init(std.testing.allocator, {});
    defer heap.deinit();

    try heap.insert(1);
    try heap.insert(2);
    try heap.insert(3);
    _ = heap.extractMin();
    _ = heap.extractMin();
}

test "FibonacciHeap: validate invariants after insertions" {
    const IntHeap = FibonacciHeap(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);

    var heap = IntHeap.init(std.testing.allocator, {});
    defer heap.deinit();

    try heap.insert(10);
    try heap.validate();

    try heap.insert(5);
    try heap.validate();

    try heap.insert(15);
    try heap.validate();

    try heap.insert(3);
    try heap.validate();
}

test "FibonacciHeap: validate invariants after extractions" {
    const IntHeap = FibonacciHeap(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);

    var heap = IntHeap.init(std.testing.allocator, {});
    defer heap.deinit();

    try heap.insert(10);
    try heap.insert(5);
    try heap.insert(15);
    try heap.insert(3);
    try heap.insert(20);

    _ = heap.extractMin();
    try heap.validate();

    _ = heap.extractMin();
    try heap.validate();

    _ = heap.extractMin();
    try heap.validate();
}

test "FibonacciHeap: descending order insertion" {
    const IntHeap = FibonacciHeap(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);

    var heap = IntHeap.init(std.testing.allocator, {});
    defer heap.deinit();

    try heap.insert(100);
    try heap.insert(90);
    try heap.insert(80);
    try heap.insert(70);
    try heap.insert(60);

    try std.testing.expectEqual(@as(i32, 60), heap.extractMin().?);
    try std.testing.expectEqual(@as(i32, 70), heap.extractMin().?);
    try std.testing.expectEqual(@as(i32, 80), heap.extractMin().?);
}

test "FibonacciHeap: ascending order insertion" {
    const IntHeap = FibonacciHeap(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);

    var heap = IntHeap.init(std.testing.allocator, {});
    defer heap.deinit();

    try heap.insert(1);
    try heap.insert(2);
    try heap.insert(3);
    try heap.insert(4);
    try heap.insert(5);

    try std.testing.expectEqual(@as(i32, 1), heap.extractMin().?);
    try std.testing.expectEqual(@as(i32, 2), heap.extractMin().?);
    try std.testing.expectEqual(@as(i32, 3), heap.extractMin().?);
}

test "FibonacciHeap: negative values" {
    const IntHeap = FibonacciHeap(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);

    var heap = IntHeap.init(std.testing.allocator, {});
    defer heap.deinit();

    try heap.insert(-5);
    try heap.insert(0);
    try heap.insert(-10);
    try heap.insert(5);

    try std.testing.expectEqual(@as(i32, -10), heap.extractMin().?);
    try std.testing.expectEqual(@as(i32, -5), heap.extractMin().?);
    try std.testing.expectEqual(@as(i32, 0), heap.extractMin().?);
    try std.testing.expectEqual(@as(i32, 5), heap.extractMin().?);
}

test "FibonacciHeap: max heap with custom comparator" {
    const IntHeap = FibonacciHeap(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(b, a); // Reversed for max heap
        }
    }.cmp);

    var heap = IntHeap.init(std.testing.allocator, {});
    defer heap.deinit();

    try heap.insert(5);
    try heap.insert(3);
    try heap.insert(7);
    try heap.insert(1);

    try std.testing.expectEqual(@as(i32, 7), heap.extractMin().?); // Max element
    try std.testing.expectEqual(@as(i32, 5), heap.extractMin().?);
    try std.testing.expectEqual(@as(i32, 3), heap.extractMin().?);
    try std.testing.expectEqual(@as(i32, 1), heap.extractMin().?);
}

test "FibonacciHeap: large merge operation" {
    const IntHeap = FibonacciHeap(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);

    var heap1 = IntHeap.init(std.testing.allocator, {});
    defer heap1.deinit();

    var heap2 = IntHeap.init(std.testing.allocator, {});
    defer heap2.deinit();

    // Insert 100 elements in each heap
    var i: i32 = 0;
    while (i < 100) : (i += 1) {
        try heap1.insert(i * 2);
        try heap2.insert(i * 2 + 1);
    }

    heap1.merge(&heap2);

    try std.testing.expectEqual(@as(usize, 200), heap1.count());

    // Extract first 10 and verify order
    i = 0;
    while (i < 10) : (i += 1) {
        try std.testing.expectEqual(i, heap1.extractMin().?);
    }
}

test "FibonacciHeap: interleaved insert and extract" {
    const IntHeap = FibonacciHeap(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);

    var heap = IntHeap.init(std.testing.allocator, {});
    defer heap.deinit();

    try heap.insert(10);
    try heap.insert(5);
    try std.testing.expectEqual(@as(i32, 5), heap.extractMin().?);

    try heap.insert(15);
    try heap.insert(3);
    try std.testing.expectEqual(@as(i32, 3), heap.extractMin().?);

    try heap.insert(8);
    try std.testing.expectEqual(@as(i32, 8), heap.extractMin().?);
    try std.testing.expectEqual(@as(i32, 10), heap.extractMin().?);
    try std.testing.expectEqual(@as(i32, 15), heap.extractMin().?);

    try std.testing.expect(heap.isEmpty());
}
