const std = @import("std");
const Allocator = std.mem.Allocator;

/// Binomial Heap - A heap data structure using binomial trees.
///
/// Features:
/// - O(log n) insert, extractMin, merge
/// - O(1) findMin
/// - Each binomial tree has 2^k nodes at order k
/// - Forest of binomial trees with at most one tree of each order
///
/// Use cases:
/// - Priority queues with frequent merge operations
/// - Simpler implementation than Fibonacci heap
/// - Guaranteed O(log n) worst-case (not amortized) operations
///
/// Type Parameters:
/// - T: Element type (must support comparison via Context)
/// - Context: Comparison context type
/// - compareFn: Function to compare two elements (returns std.math.Order)
pub fn BinomialHeap(
    comptime T: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: T, b: T) std.math.Order,
) type {
    return struct {
        const Self = @This();

        const Node = struct {
            value: T,
            degree: usize,
            parent: ?*Node,
            child: ?*Node,
            sibling: ?*Node,

            fn init(value: T) Node {
                return Node{
                    .value = value,
                    .degree = 0,
                    .parent = null,
                    .child = null,
                    .sibling = null,
                };
            }
        };

        allocator: Allocator,
        context: Context,
        head: ?*Node,
        min_node: ?*Node,
        node_count: usize,

        // -- Lifecycle --

        /// Initialize an empty binomial heap.
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator, context: Context) Self {
            return Self{
                .allocator = allocator,
                .context = context,
                .head = null,
                .min_node = null,
                .node_count = 0,
            };
        }

        /// Destroy the heap and free all nodes.
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            var current = self.head;
            while (current) |node| {
                const next = node.sibling;
                self.destroyTree(node);
                current = next;
            }
        }

        fn destroyTree(self: *Self, node: *Node) void {
            var child = node.child;
            while (child) |c| {
                const next_sibling = c.sibling;
                self.destroyTree(c);
                child = next_sibling;
            }
            self.allocator.destroy(node);
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
        /// Time: O(log n) | Space: O(1)
        pub fn insert(self: *Self, value: T) !void {
            const node = try self.allocator.create(Node);
            node.* = Node.init(value);

            // Create a temporary heap with just this node
            var temp = Self.init(self.allocator, self.context);
            temp.head = node;
            temp.min_node = node;
            temp.node_count = 1;

            // Merge into current heap
            self.mergeHeap(&temp);
        }

        /// Remove and return the minimum element.
        /// Time: O(log n) | Space: O(1)
        pub fn extractMin(self: *Self) ?T {
            if (self.min_node == null) return null;

            const min = self.min_node.?;
            const value = min.value;

            // Remove min from root list
            if (self.head == min) {
                self.head = min.sibling;
            } else {
                var prev = self.head;
                while (prev) |p| {
                    if (p.sibling == min) {
                        p.sibling = min.sibling;
                        break;
                    }
                    prev = p.sibling;
                }
            }

            // Create a new heap from min's children
            var temp = Self.init(self.allocator, self.context);
            if (min.child) |first_child| {
                // Reverse the child list to maintain proper order
                temp.head = self.reverseList(first_child);

                // Clear parent pointers and count nodes
                var child = temp.head;
                while (child) |c| {
                    c.parent = null;
                    temp.node_count += 1;
                    child = c.sibling;
                }
            }

            self.allocator.destroy(min);
            self.node_count -= 1;

            // Merge the children back into the heap
            self.mergeHeap(&temp);

            // Find new minimum
            self.updateMin();

            return value;
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

        /// Merge another binomial heap into this one.
        /// Time: O(log n) | Space: O(1)
        pub fn merge(self: *Self, other: *Self) void {
            self.mergeHeap(other);
        }

        // -- Debug --

        /// Validate heap invariants.
        /// Time: O(n) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            if (self.head == null) {
                if (self.node_count != 0) {
                    return error.InvalidNodeCount;
                }
                return;
            }

            var node_count: usize = 0;
            var prev_degree: ?usize = null;

            // Check root list
            var current = self.head;
            while (current) |node| {
                // Check degree ordering
                if (prev_degree) |pd| {
                    if (node.degree <= pd) {
                        return error.InvalidDegreeOrder;
                    }
                }
                prev_degree = node.degree;

                // Check parent pointer
                if (node.parent != null) {
                    return error.InvalidParentPointer;
                }

                // Validate tree
                node_count += 1;
                node_count += try self.validateTree(node);

                current = node.sibling;
            }

            if (node_count != self.node_count) {
                return error.InvalidNodeCount;
            }

            // Check that min_node is correct
            if (self.min_node) |min| {
                var check = self.head;
                var found = false;
                while (check) |c| {
                    if (c == min) found = true;
                    if (compareFn(self.context, c.value, min.value) == .lt) {
                        return error.InvalidMinNode;
                    }
                    check = c.sibling;
                }
                if (!found) return error.InvalidMinNode;
            }
        }

        fn validateTree(self: *const Self, node: *const Node) !usize {
            var child_count: usize = 0;
            var expected_degree: usize = node.degree;

            var child = node.child;
            while (child) |c| {
                // Check parent pointer
                if (c.parent != node) {
                    return error.InvalidParentPointer;
                }

                // Check heap property
                if (compareFn(self.context, c.value, node.value) == .lt) {
                    return error.HeapPropertyViolation;
                }

                // Check degree ordering (children should be in decreasing degree order)
                expected_degree -= 1;
                if (c.degree != expected_degree) {
                    return error.InvalidChildDegree;
                }

                child_count += 1;
                child_count += try self.validateTree(c);
                child = c.sibling;
            }

            // Binomial tree of degree k has exactly 2^k nodes
            const expected_count = (@as(usize, 1) << @intCast(node.degree)) - 1;
            if (child_count != expected_count) {
                return error.InvalidTreeSize;
            }

            return child_count;
        }

        // -- Private Helpers --

        fn mergeHeap(self: *Self, other: *Self) void {
            if (other.head == null) return;

            if (self.head == null) {
                self.head = other.head;
                self.min_node = other.min_node;
                self.node_count = other.node_count;
            } else {
                // Merge root lists
                self.head = self.mergeRootLists(self.head, other.head);
                self.node_count += other.node_count;

                // Consolidate trees with same degree
                self.consolidate();

                // Update minimum
                self.updateMin();
            }

            other.head = null;
            other.min_node = null;
            other.node_count = 0;
        }

        fn mergeRootLists(self: *Self, h1: ?*Node, h2: ?*Node) ?*Node {
            _ = self;
            if (h1 == null) return h2;
            if (h2 == null) return h1;

            var head: ?*Node = null;
            var tail: ?*Node = null;
            var list1 = h1;
            var list2 = h2;

            // Merge by degree (ascending order)
            while (list1 != null or list2 != null) {
                var next: *Node = undefined;

                if (list1 == null) {
                    next = list2.?;
                    list2 = list2.?.sibling;
                } else if (list2 == null) {
                    next = list1.?;
                    list1 = list1.?.sibling;
                } else if (list1.?.degree <= list2.?.degree) {
                    next = list1.?;
                    list1 = list1.?.sibling;
                } else {
                    next = list2.?;
                    list2 = list2.?.sibling;
                }

                if (tail == null) {
                    head = next;
                } else {
                    tail.?.sibling = next;
                }
                tail = next;
            }

            if (tail) |t| {
                t.sibling = null;
            }

            return head;
        }

        fn consolidate(self: *Self) void {
            if (self.head == null) return;

            var prev: ?*Node = null;
            var current = self.head;
            var next = if (current) |c| c.sibling else null;

            while (current) |curr| {
                if (next == null or curr.degree != next.?.degree) {
                    // No merge needed
                    prev = curr;
                    current = next;
                    next = if (current) |c| c.sibling else null;
                } else if (next.?.sibling != null and next.?.sibling.?.degree == curr.degree) {
                    // Three consecutive trees with same degree - skip merge
                    prev = curr;
                    current = next;
                    next = if (current) |c| c.sibling else null;
                } else {
                    // Merge curr and next
                    if (compareFn(self.context, curr.value, next.?.value) == .lt or
                        compareFn(self.context, curr.value, next.?.value) == .eq)
                    {
                        // curr becomes parent of next
                        curr.sibling = next.?.sibling;
                        self.linkTrees(next.?, curr);
                        next = curr.sibling;
                    } else {
                        // next becomes parent of curr
                        if (prev == null) {
                            self.head = next;
                        } else {
                            prev.?.sibling = next;
                        }
                        self.linkTrees(curr, next.?);
                        current = next;
                        next = if (current) |c| c.sibling else null;
                    }
                }
            }
        }

        fn linkTrees(self: *Self, child: *Node, parent: *Node) void {
            _ = self;
            child.parent = parent;
            child.sibling = parent.child;
            parent.child = child;
            parent.degree += 1;
        }

        fn updateMin(self: *Self) void {
            self.min_node = null;
            var current = self.head;

            while (current) |node| {
                if (self.min_node == null or
                    compareFn(self.context, node.value, self.min_node.?.value) == .lt)
                {
                    self.min_node = node;
                }
                current = node.sibling;
            }
        }

        fn reverseList(self: *Self, head: *Node) ?*Node {
            _ = self;
            var prev: ?*Node = null;
            var current: ?*Node = head;

            while (current) |curr| {
                const next = curr.sibling;
                curr.sibling = prev;
                prev = curr;
                current = next;
            }

            return prev;
        }
    };
}

// -- Tests --

test "BinomialHeap: basic insert and extract" {
    const IntHeap = BinomialHeap(i32, void, struct {
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

test "BinomialHeap: empty heap" {
    const IntHeap = BinomialHeap(i32, void, struct {
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

test "BinomialHeap: single element" {
    const IntHeap = BinomialHeap(i32, void, struct {
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

test "BinomialHeap: duplicate values" {
    const IntHeap = BinomialHeap(i32, void, struct {
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

test "BinomialHeap: merge two heaps" {
    const IntHeap = BinomialHeap(i32, void, struct {
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

test "BinomialHeap: merge with empty heap" {
    const IntHeap = BinomialHeap(i32, void, struct {
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

test "BinomialHeap: stress test with 1000 insertions" {
    const IntHeap = BinomialHeap(i32, void, struct {
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

test "BinomialHeap: memory leak detection" {
    const IntHeap = BinomialHeap(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);

    var heap = IntHeap.init(std.testing.allocator, {});
    defer heap.deinit();

    try heap.insert(1);
    try heap.insert(2);
    try heap.insert(3);
    try std.testing.expectEqual(@as(usize, 3), heap.count());

    const min1 = heap.extractMin();
    try std.testing.expectEqual(@as(?i32, 1), min1);
    try std.testing.expectEqual(@as(usize, 2), heap.count());

    const min2 = heap.extractMin();
    try std.testing.expectEqual(@as(?i32, 2), min2);
    try std.testing.expectEqual(@as(usize, 1), heap.count());
}

test "BinomialHeap: validate invariants after insertions" {
    const IntHeap = BinomialHeap(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);

    var heap = IntHeap.init(std.testing.allocator, {});
    defer heap.deinit();

    try heap.insert(10);
    try heap.validate();
    try std.testing.expectEqual(@as(usize, 1), heap.count());
    try std.testing.expectEqual(@as(?i32, 10), heap.peekMin());

    try heap.insert(5);
    try heap.validate();
    try std.testing.expectEqual(@as(usize, 2), heap.count());
    try std.testing.expectEqual(@as(?i32, 5), heap.peekMin());

    try heap.insert(15);
    try heap.validate();
    try std.testing.expectEqual(@as(usize, 3), heap.count());
    try std.testing.expectEqual(@as(?i32, 5), heap.peekMin());

    try heap.insert(3);
    try heap.validate();
    try std.testing.expectEqual(@as(usize, 4), heap.count());
    try std.testing.expectEqual(@as(?i32, 3), heap.peekMin());
}

test "BinomialHeap: validate invariants after extractions" {
    const IntHeap = BinomialHeap(i32, void, struct {
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
    try std.testing.expectEqual(@as(usize, 5), heap.count());

    const min1 = heap.extractMin();
    try std.testing.expectEqual(@as(?i32, 3), min1);
    try heap.validate();
    try std.testing.expectEqual(@as(usize, 4), heap.count());

    const min2 = heap.extractMin();
    try std.testing.expectEqual(@as(?i32, 5), min2);
    try heap.validate();
    try std.testing.expectEqual(@as(usize, 3), heap.count());

    const min3 = heap.extractMin();
    try std.testing.expectEqual(@as(?i32, 10), min3);
    try heap.validate();
    try std.testing.expectEqual(@as(usize, 2), heap.count());
}

test "BinomialHeap: descending order insertion" {
    const IntHeap = BinomialHeap(i32, void, struct {
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

test "BinomialHeap: ascending order insertion" {
    const IntHeap = BinomialHeap(i32, void, struct {
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

test "BinomialHeap: negative values" {
    const IntHeap = BinomialHeap(i32, void, struct {
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

test "BinomialHeap: max heap with custom comparator" {
    const IntHeap = BinomialHeap(i32, void, struct {
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

test "BinomialHeap: large merge operation" {
    const IntHeap = BinomialHeap(i32, void, struct {
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

test "BinomialHeap: interleaved insert and extract" {
    const IntHeap = BinomialHeap(i32, void, struct {
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

test "BinomialHeap: power of 2 insertions" {
    const IntHeap = BinomialHeap(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);

    var heap = IntHeap.init(std.testing.allocator, {});
    defer heap.deinit();

    // Insert exactly 16 elements (power of 2)
    var i: i32 = 16;
    while (i > 0) : (i -= 1) {
        try heap.insert(i);
    }

    try heap.validate();
    try std.testing.expectEqual(@as(usize, 16), heap.count());

    // Should have a single binomial tree of degree 4
    try std.testing.expectEqual(@as(usize, 4), heap.head.?.degree);
}
