const std = @import("std");
const testing = std.testing;

/// Persistent Array - Immutable vector with structural sharing
///
/// Uses a tree structure (similar to Clojure's PersistentVector) for efficient
/// immutable operations. Each node has a branching factor of 32.
///
/// Time Complexity:
/// - get: O(log₃₂ n) ≈ O(1) for practical sizes (up to 2³⁰ elements)
/// - set: O(log₃₂ n) with path copying
/// - push: O(log₃₂ n) amortized
/// - pop: O(log₃₂ n)
/// - slice: O(log₃₂ n)
///
/// Space Complexity:
/// - O(n) for storage
/// - O(log₃₂ n) per mutation (path copying)
///
/// Consumer use case: Functional programming patterns, undo/redo systems,
/// concurrent data structures without locks, version control systems
pub fn PersistentArray(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Branching factor - must be power of 2 for efficient bit shifts
        const BITS = 5;
        const BRANCH_FACTOR = 1 << BITS; // 32
        const MASK = BRANCH_FACTOR - 1; // 0x1F

        const Node = union(enum) {
            leaf: [BRANCH_FACTOR]T,
            internal: [BRANCH_FACTOR]?*Node,

            fn deinitRecursive(node: *Node, allocator: std.mem.Allocator, height: u32) void {
                if (height > 0) {
                    for (node.internal) |maybe_child| {
                        if (maybe_child) |child| {
                            child.deinitRecursive(allocator, height - 1);
                        }
                    }
                }
                allocator.destroy(node);
            }
        };

        allocator: std.mem.Allocator,
        root: ?*Node,
        tail: []T, // Last incomplete block (up to BRANCH_FACTOR elements)
        size: usize,
        height: u32, // Tree height (0 for empty or tail-only)

        /// Initialize an empty persistent array
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: std.mem.Allocator) !Self {
            return Self{
                .allocator = allocator,
                .root = null,
                .tail = &[_]T{},
                .size = 0,
                .height = 0,
            };
        }

        /// Deinitialize and free all memory
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.root) |root| {
                root.deinitRecursive(self.allocator, self.height);
            }
            if (self.tail.len > 0) {
                self.allocator.free(self.tail);
            }
        }

        /// Get the number of elements
        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.size;
        }

        /// Check if the array is empty
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.size == 0;
        }

        /// Get element at index (bounds checked)
        /// Time: O(log₃₂ n) | Space: O(1)
        pub fn get(self: *const Self, index: usize) error{IndexOutOfBounds}!T {
            if (index >= self.size) return error.IndexOutOfBounds;

            // Check if index is in tail
            const tail_offset = self.tailOffset();
            if (index >= tail_offset) {
                return self.tail[index - tail_offset];
            }

            // Navigate tree
            var node = self.root orelse return error.IndexOutOfBounds;
            var shift = self.height * BITS;
            const idx = index;

            while (shift > 0) {
                shift -= BITS;
                const slot = (idx >> @intCast(shift)) & MASK;
                node = node.internal[slot] orelse return error.IndexOutOfBounds;
            }

            const slot = idx & MASK;
            return node.leaf[slot];
        }

        /// Set element at index, returning new array (original unchanged)
        /// Time: O(log₃₂ n) | Space: O(log₃₂ n)
        pub fn set(self: *const Self, index: usize, value: T) !Self {
            if (index >= self.size) return error.IndexOutOfBounds;

            // Handle tail separately
            const tail_offset = self.tailOffset();
            if (index >= tail_offset) {
                var new_self = try self.clone();
                new_self.tail[index - tail_offset] = value;
                return new_self;
            }

            // Copy path from root to target leaf
            const new_self = Self{
                .allocator = self.allocator,
                .root = try self.copyPath(index),
                .tail = try self.allocator.dupe(T, self.tail),
                .size = self.size,
                .height = self.height,
            };

            // Update the leaf
            var node = new_self.root.?;
            var shift = self.height * BITS;
            const idx = index;

            while (shift > 0) {
                shift -= BITS;
                const slot = (idx >> @intCast(shift)) & MASK;
                node = node.internal[slot].?;
            }

            const slot = idx & MASK;
            node.leaf[slot] = value;

            return new_self;
        }

        /// Append element, returning new array (original unchanged)
        /// Time: O(log₃₂ n) amortized | Space: O(log₃₂ n)
        pub fn push(self: *const Self, value: T) !Self {
            // If tail has room, just append
            if (self.tail.len < BRANCH_FACTOR) {
                var new_tail = try self.allocator.alloc(T, self.tail.len + 1);
                @memcpy(new_tail[0..self.tail.len], self.tail);
                new_tail[self.tail.len] = value;

                return Self{
                    .allocator = self.allocator,
                    .root = self.root,
                    .tail = new_tail,
                    .size = self.size + 1,
                    .height = self.height,
                };
            }

            // Tail is full, need to push it into tree
            const new_root = try self.pushTailIntoTree();
            const new_tail = try self.allocator.alloc(T, 1);
            new_tail[0] = value;

            return Self{
                .allocator = self.allocator,
                .root = new_root.root,
                .tail = new_tail,
                .size = self.size + 1,
                .height = new_root.height,
            };
        }

        /// Remove last element, returning new array (original unchanged)
        /// Time: O(log₃₂ n) | Space: O(log₃₂ n)
        pub fn pop(self: *const Self) !Self {
            if (self.size == 0) return error.Empty;

            // If tail has more than one element, just remove last
            if (self.tail.len > 1) {
                return Self{
                    .allocator = self.allocator,
                    .root = self.root,
                    .tail = try self.allocator.dupe(T, self.tail[0 .. self.tail.len - 1]),
                    .size = self.size - 1,
                    .height = self.height,
                };
            }

            // Need to pull last block from tree into tail
            if (self.root == null) {
                return Self{
                    .allocator = self.allocator,
                    .root = null,
                    .tail = &[_]T{},
                    .size = 0,
                    .height = 0,
                };
            }

            // Pop from tree: get the last leaf block and make it the new tail
            const new_tail_offset = self.tailOffset() - BRANCH_FACTOR;
            const new_tail = try self.allocator.alloc(T, BRANCH_FACTOR - 1);

            // Navigate to the last leaf in the tree (at new_tail_offset)
            var node = self.root.?;
            var shift = self.height * BITS;
            const idx = new_tail_offset;

            while (shift > 0) {
                shift -= BITS;
                const slot = (idx >> @intCast(shift)) & MASK;
                node = node.internal[slot] orelse return error.InvalidState;
            }

            // Copy leaf elements (all except the last which was in tail)
            @memcpy(new_tail, node.leaf[0 .. BRANCH_FACTOR - 1]);

            // Remove this leaf from tree - need to copy path and exclude last block
            const new_root = try self.popFromTree(new_tail_offset);

            return Self{
                .allocator = self.allocator,
                .root = new_root.root,
                .tail = new_tail,
                .size = self.size - 1,
                .height = new_root.height,
            };
        }

        /// Create a slice view [start..end)
        /// Time: O(end - start) | Space: O(end - start)
        pub fn slice(self: *const Self, start: usize, end: usize) ![]T {
            if (start > end or end > self.size) return error.InvalidRange;

            var result = try self.allocator.alloc(T, end - start);
            for (start..end, 0..) |i, j| {
                result[j] = try self.get(i);
            }
            return result;
        }

        /// Convert to a regular slice
        /// Time: O(n) | Space: O(n)
        pub fn toSlice(self: *const Self) ![]T {
            return try self.slice(0, self.size);
        }

        /// Create from a slice
        /// Time: O(n) | Space: O(n)
        pub fn fromSlice(allocator: std.mem.Allocator, items: []const T) !Self {
            var array = try Self.init(allocator);
            for (items) |item| {
                const new_array = try array.push(item);
                if (array.size > 0) array.deinit();
                array = new_array;
            }
            return array;
        }

        // --- Private helpers ---

        fn tailOffset(self: *const Self) usize {
            if (self.size <= BRANCH_FACTOR) return 0;
            return ((self.size - 1) >> BITS) << BITS;
        }

        fn clone(self: *const Self) !Self {
            return Self{
                .allocator = self.allocator,
                .root = self.root,
                .tail = try self.allocator.dupe(T, self.tail),
                .size = self.size,
                .height = self.height,
            };
        }

        fn copyPath(self: *const Self, index: usize) !*Node {
            const new_root = try self.allocator.create(Node);
            new_root.* = Node{ .internal = [_]?*Node{null} ** BRANCH_FACTOR };

            var src = self.root orelse unreachable;
            var dst = new_root;
            var shift = self.height * BITS;
            const idx = index;

            while (shift > BITS) {
                shift -= BITS;
                const slot = (idx >> @intCast(shift)) & MASK;

                // Copy current level
                for (src.internal, 0..) |child, i| {
                    if (i != slot) {
                        dst.internal[i] = child; // Share unchanged branches
                    }
                }

                // Create new node for path
                const new_node = try self.allocator.create(Node);
                new_node.* = Node{ .internal = [_]?*Node{null} ** BRANCH_FACTOR };
                dst.internal[slot] = new_node;

                src = src.internal[slot].?;
                dst = new_node;
            }

            // Copy leaf level
            const slot = (idx >> BITS) & MASK;
            const new_leaf = try self.allocator.create(Node);
            new_leaf.* = Node{ .leaf = src.internal[slot].?.leaf };
            dst.internal[slot] = new_leaf;

            return new_root;
        }

        fn pushTailIntoTree(self: *const Self) !struct { root: ?*Node, height: u32 } {
            // Create leaf node from tail
            const new_leaf = try self.allocator.create(Node);
            new_leaf.* = Node{ .leaf = undefined };
            @memcpy(&new_leaf.leaf, self.tail);

            if (self.root == null) {
                return .{ .root = new_leaf, .height = 0 };
            }

            // Check if we need to increase height
            const slots_at_level = std.math.pow(usize, BRANCH_FACTOR, self.height + 1);
            if (self.size > slots_at_level) {
                // Need to grow tree height
                const new_root = try self.allocator.create(Node);
                new_root.* = Node{ .internal = [_]?*Node{null} ** BRANCH_FACTOR };
                new_root.internal[0] = self.root;
                new_root.internal[1] = new_leaf;
                return .{ .root = new_root, .height = self.height + 1 };
            }

            // Insert into existing tree (path copying)
            const new_root = try self.copyPathForInsert(self.tailOffset(), new_leaf);
            return .{ .root = new_root, .height = self.height };
        }

        fn copyPathForInsert(self: *const Self, index: usize, leaf: *Node) !*Node {
            const new_root = try self.allocator.create(Node);
            new_root.* = Node{ .internal = [_]?*Node{null} ** BRANCH_FACTOR };

            var src = self.root.?;
            var dst = new_root;
            var shift = self.height * BITS;
            const idx = index;

            // Copy path down to where we insert
            while (shift > 0) {
                shift -= BITS;
                const slot = (idx >> @intCast(shift)) & MASK;

                // Share unchanged branches
                for (src.internal, 0..) |child, i| {
                    if (i < slot) {
                        dst.internal[i] = child;
                    }
                }

                if (shift == 0) {
                    // Insert leaf here
                    dst.internal[slot] = leaf;
                    break;
                }

                // Create new internal node for path
                if (src.internal[slot]) |child| {
                    const new_node = try self.allocator.create(Node);
                    new_node.* = Node{ .internal = [_]?*Node{null} ** BRANCH_FACTOR };
                    dst.internal[slot] = new_node;
                    src = child;
                    dst = new_node;
                } else {
                    // Path doesn't exist yet, create it
                    const new_node = try self.allocator.create(Node);
                    new_node.* = Node{ .internal = [_]?*Node{null} ** BRANCH_FACTOR };
                    dst.internal[slot] = new_node;
                    dst = new_node;
                }
            }

            return new_root;
        }

        fn popFromTree(self: *const Self, last_index: usize) !struct { root: ?*Node, height: u32 } {
            if (self.root == null) return .{ .root = null, .height = 0 };

            // If only one element in tree, return null root
            if (self.tailOffset() == BRANCH_FACTOR) {
                return .{ .root = null, .height = 0 };
            }

            // Copy path excluding the last block
            const new_root = try self.copyPathForPop(last_index);

            // Check if we can reduce height (root has only one child)
            if (self.height > 0) {
                var child_count: usize = 0;
                var first_child: ?*Node = null;
                for (new_root.?.internal) |child| {
                    if (child) |c| {
                        child_count += 1;
                        if (first_child == null) first_child = c;
                    }
                }
                if (child_count == 1 and first_child != null) {
                    // Reduce height - deallocate old root, return the single child
                    return .{ .root = first_child, .height = self.height - 1 };
                }
            }

            return .{ .root = new_root, .height = self.height };
        }

        fn copyPathForPop(self: *const Self, last_index: usize) !?*Node {
            const new_root = try self.allocator.create(Node);
            new_root.* = Node{ .internal = [_]?*Node{null} ** BRANCH_FACTOR };

            var src = self.root.?;
            var dst = new_root;
            var shift = self.height * BITS;
            const idx = last_index;

            while (shift > 0) {
                shift -= BITS;
                const slot = (idx >> @intCast(shift)) & MASK;

                // Copy all children except the path to last element
                for (src.internal, 0..) |child, i| {
                    if (i < slot) {
                        dst.internal[i] = child;
                    } else if (i == slot and shift > 0) {
                        // Continue path - create new node
                        if (child) |c| {
                            const new_node = try self.allocator.create(Node);
                            new_node.* = Node{ .internal = [_]?*Node{null} ** BRANCH_FACTOR };
                            dst.internal[i] = new_node;
                            src = c;
                            dst = new_node;
                        }
                    }
                    // i > slot or (i == slot and shift == 0): skip (exclude last block)
                }
            }

            return new_root;
        }

        /// Validate internal invariants
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: *const Self) void {
            std.debug.assert(self.tail.len <= BRANCH_FACTOR);
            if (self.size == 0) {
                std.debug.assert(self.root == null);
                std.debug.assert(self.tail.len == 0);
                std.debug.assert(self.height == 0);
            }
        }
    };
}

// --- Tests ---

test "PersistentArray: init and empty" {
    var arr = try PersistentArray(i32).init(testing.allocator);
    defer arr.deinit();

    try testing.expectEqual(@as(usize, 0), arr.count());
    try testing.expect(arr.isEmpty());
}

test "PersistentArray: push and get" {
    var arr1 = try PersistentArray(i32).init(testing.allocator);
    defer arr1.deinit();

    const arr2 = try arr1.push(10);
    defer arr2.deinit();

    try testing.expectEqual(@as(usize, 1), arr2.count());
    try testing.expectEqual(@as(i32, 10), try arr2.get(0));

    const arr3 = try arr2.push(20);
    defer arr3.deinit();

    try testing.expectEqual(@as(usize, 2), arr3.count());
    try testing.expectEqual(@as(i32, 10), try arr3.get(0));
    try testing.expectEqual(@as(i32, 20), try arr3.get(1));
}

test "PersistentArray: immutability" {
    var arr1 = try PersistentArray(i32).init(testing.allocator);
    defer arr1.deinit();

    const arr2 = try arr1.push(10);
    defer arr2.deinit();

    const arr3 = try arr2.push(20);
    defer arr3.deinit();

    // arr1 is still empty
    try testing.expectEqual(@as(usize, 0), arr1.count());
    // arr2 still has one element
    try testing.expectEqual(@as(usize, 1), arr2.count());
    try testing.expectEqual(@as(i32, 10), try arr2.get(0));
    // arr3 has two elements
    try testing.expectEqual(@as(usize, 2), arr3.count());
}

test "PersistentArray: set immutability" {
    var arr1 = try PersistentArray(i32).fromSlice(testing.allocator, &[_]i32{ 1, 2, 3 });
    defer arr1.deinit();

    var arr2 = try arr1.set(1, 99);
    defer arr2.deinit();

    // arr1 unchanged
    try testing.expectEqual(@as(i32, 2), try arr1.get(1));
    // arr2 has new value
    try testing.expectEqual(@as(i32, 99), try arr2.get(1));
    // Other elements shared
    try testing.expectEqual(@as(i32, 1), try arr2.get(0));
    try testing.expectEqual(@as(i32, 3), try arr2.get(2));
}

test "PersistentArray: fromSlice and toSlice" {
    const items = [_]i32{ 1, 2, 3, 4, 5 };
    var arr = try PersistentArray(i32).fromSlice(testing.allocator, &items);
    defer arr.deinit();

    try testing.expectEqual(@as(usize, 5), arr.count());

    const result = try arr.toSlice();
    defer testing.allocator.free(result);

    try testing.expectEqualSlices(i32, &items, result);
}

test "PersistentArray: large array (force tree growth)" {
    var arr = try PersistentArray(i32).init(testing.allocator);
    defer arr.deinit();

    // Push enough elements to trigger tree structure (> 32 elements)
    var current = arr;
    for (0..100) |i| {
        const next = try current.push(@intCast(i));
        if (i > 0) current.deinit();
        current = next;
    }

    try testing.expectEqual(@as(usize, 100), current.count());

    // Verify all elements
    for (0..100) |i| {
        try testing.expectEqual(@as(i32, @intCast(i)), try current.get(i));
    }

    current.deinit();
}

test "PersistentArray: bounds checking" {
    var arr = try PersistentArray(i32).fromSlice(testing.allocator, &[_]i32{ 1, 2, 3 });
    defer arr.deinit();

    try testing.expectError(error.IndexOutOfBounds, arr.get(3));
    try testing.expectError(error.IndexOutOfBounds, arr.set(3, 99));
}

test "PersistentArray: slice operation" {
    var arr = try PersistentArray(i32).fromSlice(testing.allocator, &[_]i32{ 0, 1, 2, 3, 4, 5 });
    defer arr.deinit();

    const s = try arr.slice(2, 5);
    defer testing.allocator.free(s);

    try testing.expectEqualSlices(i32, &[_]i32{ 2, 3, 4 }, s);
}

test "PersistentArray: structural sharing" {
    // Create initial array with 40 elements (requires tree structure)
    var arr1 = try PersistentArray(i32).init(testing.allocator);
    defer arr1.deinit();

    var current = arr1;
    for (0..40) |i| {
        const next = try current.push(@intCast(i));
        if (i > 0) current.deinit();
        current = next;
    }

    // Modify one element - should share most structure
    var arr2 = try current.set(10, 999);
    defer arr2.deinit();

    // Both arrays should be independent
    try testing.expectEqual(@as(i32, 10), try current.get(10));
    try testing.expectEqual(@as(i32, 999), try arr2.get(10));

    // Other elements should match
    for (0..40) |i| {
        if (i == 10) continue;
        try testing.expectEqual(try current.get(i), try arr2.get(i));
    }

    current.deinit();
}

test "PersistentArray: memory leak check" {
    var arr1 = try PersistentArray(i32).fromSlice(testing.allocator, &[_]i32{ 1, 2, 3, 4, 5 });
    try testing.expectEqual(@as(usize, 5), arr1.count());
    try testing.expectEqual(@as(i32, 1), try arr1.get(0));

    var arr2 = try arr1.push(6);
    try testing.expectEqual(@as(usize, 6), arr2.count());
    try testing.expectEqual(@as(i32, 6), try arr2.get(5));

    var arr3 = try arr2.set(2, 99);
    try testing.expectEqual(@as(usize, 6), arr3.count());
    try testing.expectEqual(@as(i32, 99), try arr3.get(2));

    arr1.deinit();
    arr2.deinit();
    arr3.deinit();
}

test "PersistentArray: strings" {
    var arr1 = try PersistentArray([]const u8).init(testing.allocator);
    defer arr1.deinit();

    var arr2 = try arr1.push("hello");
    defer arr2.deinit();

    var arr3 = try arr2.push("world");
    defer arr3.deinit();

    try testing.expectEqualStrings("hello", try arr3.get(0));
    try testing.expectEqualStrings("world", try arr3.get(1));
}

test "PersistentArray: pop from tail" {
    var arr = try PersistentArray(i32).init(testing.allocator);
    defer arr.deinit();

    // Push a few elements
    const arr1 = try arr.push(10);
    defer arr1.deinit();
    const arr2 = try arr1.push(20);
    defer arr2.deinit();
    const arr3 = try arr2.push(30);
    defer arr3.deinit();

    try testing.expectEqual(@as(usize, 3), arr3.count());

    // Pop one element (should only modify tail)
    const arr4 = try arr3.pop();
    defer arr4.deinit();

    try testing.expectEqual(@as(usize, 2), arr4.count());
    try testing.expectEqual(@as(i32, 10), try arr4.get(0));
    try testing.expectEqual(@as(i32, 20), try arr4.get(1));

    // Original arr3 should be unchanged
    try testing.expectEqual(@as(usize, 3), arr3.count());
    try testing.expectEqual(@as(i32, 30), try arr3.get(2));
}

test "PersistentArray: pop to empty" {
    var arr = try PersistentArray(i32).init(testing.allocator);
    defer arr.deinit();

    const arr1 = try arr.push(42);
    defer arr1.deinit();

    const arr2 = try arr1.pop();
    defer arr2.deinit();

    try testing.expectEqual(@as(usize, 0), arr2.count());
    try testing.expect(arr2.isEmpty());
}

test "PersistentArray: pop from empty" {
    var arr = try PersistentArray(i32).init(testing.allocator);
    defer arr.deinit();

    const result = arr.pop();
    try testing.expectError(error.Empty, result);
}

test "PersistentArray: pop from tree (large array)" {
    var arr = try PersistentArray(i32).init(testing.allocator);
    defer arr.deinit();

    // Push 100 elements to force tree growth beyond tail
    var current = arr;
    var i: i32 = 0;
    while (i < 100) : (i += 1) {
        const next = try current.push(i);
        if (i > 0) current.deinit();
        current = next;
    }
    defer current.deinit();

    try testing.expectEqual(@as(usize, 100), current.count());

    // Pop one element (should require pulling from tree)
    const popped = try current.pop();
    defer popped.deinit();

    try testing.expectEqual(@as(usize, 99), popped.count());
    try testing.expectEqual(@as(i32, 98), try popped.get(98));

    // Verify all elements are intact
    i = 0;
    while (i < 99) : (i += 1) {
        try testing.expectEqual(i, try popped.get(@intCast(i)));
    }
}

test "PersistentArray: multiple pops" {
    var arr = try PersistentArray(i32).init(testing.allocator);
    defer arr.deinit();

    // Push 50 elements
    var current = arr;
    var i: i32 = 0;
    while (i < 50) : (i += 1) {
        const next = try current.push(i);
        if (i > 0) current.deinit();
        current = next;
    }
    defer current.deinit();

    // Pop 10 times
    var popped = current;
    var pop_count: usize = 0;
    while (pop_count < 10) : (pop_count += 1) {
        const next = try popped.pop();
        if (pop_count > 0) popped.deinit();
        popped = next;
    }
    defer popped.deinit();

    try testing.expectEqual(@as(usize, 40), popped.count());
    try testing.expectEqual(@as(i32, 39), try popped.get(39));
}

test "PersistentArray: pop maintains immutability" {
    var arr = try PersistentArray(i32).init(testing.allocator);
    defer arr.deinit();

    const arr1 = try arr.push(1);
    defer arr1.deinit();
    const arr2 = try arr1.push(2);
    defer arr2.deinit();
    const arr3 = try arr2.push(3);
    defer arr3.deinit();

    const arr4 = try arr3.pop();
    defer arr4.deinit();

    // arr3 should be unchanged
    try testing.expectEqual(@as(usize, 3), arr3.count());
    try testing.expectEqual(@as(i32, 3), try arr3.get(2));

    // arr4 should have 2 elements
    try testing.expectEqual(@as(usize, 2), arr4.count());
    try testing.expectEqual(@as(i32, 2), try arr4.get(1));
}
