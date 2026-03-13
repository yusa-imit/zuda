const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// DaryHeap is a generalized d-ary heap where each node has d children.
/// Array-backed implementation with optimal cache performance when d=4.
///
/// Time complexity:
/// - insert: O(log_d n)
/// - findMin: O(1)
/// - extractMin: O(d * log_d n)
/// - decreaseKey: O(log_d n)
/// - heapify: O(n)
///
/// Space: O(n)
///
/// Use cases:
/// - Priority queues with cache-conscious performance (d=4 optimal)
/// - Embedded systems with fixed-capacity variant
/// - Applications requiring tunable heap branching factor
///
/// Note: Binary heap is a special case with d=2.
/// Larger d reduces tree height but increases comparison cost in extractMin.
/// d=4 often provides the best practical performance due to cache effects.
///
pub fn DaryHeap(
    comptime T: type,
    comptime d: comptime_int,
    comptime Context: type,
    comptime lessThan: fn (ctx: Context, a: T, b: T) bool,
) type {
    if (d < 2) @compileError("d must be at least 2");

    return struct {
        const Self = @This();

        /// Entry type for tracking original indices (used for decreaseKey)
        pub const Entry = struct {
            data: T,
            index: usize, // Position in the heap array
        };

        allocator: Allocator,
        context: Context,
        items: std.ArrayList(T),

        // -- Lifecycle --

        /// Initialize an empty d-ary heap.
        pub fn init(allocator: Allocator, context: Context) Self {
            return .{
                .allocator = allocator,
                .context = context,
                .items = .{},
            };
        }

        /// Initialize a heap from a slice (heapify).
        /// Time: O(n) | Space: O(n)
        pub fn initFromSlice(allocator: Allocator, context: Context, slice: []const T) !Self {
            var self = Self.init(allocator, context);
            try self.items.appendSlice(allocator, slice);

            // Bottom-up heapify
            if (self.items.items.len > 1) {
                var i: isize = @intCast(self.parentIndex(self.items.items.len - 1));
                while (i >= 0) : (i -= 1) {
                    self.siftDown(@intCast(i));
                }
            }

            return self;
        }

        /// Deinitialize the heap.
        pub fn deinit(self: *Self) void {
            self.items.deinit(self.allocator);
            self.* = undefined;
        }

        /// Clone the heap.
        pub fn clone(self: *const Self) !Self {
            var new = Self.init(self.allocator, self.context);
            try new.items.appendSlice(self.items.items);
            return new;
        }

        // -- Capacity --

        /// Return the number of elements in the heap.
        pub fn count(self: *const Self) usize {
            return self.items.items.len;
        }

        /// Check if the heap is empty.
        pub fn isEmpty(self: *const Self) bool {
            return self.items.items.len == 0;
        }

        /// Reserve capacity for at least `additional` more elements.
        pub fn ensureTotalCapacity(self: *Self, new_capacity: usize) !void {
            try self.items.ensureTotalCapacity(new_capacity);
        }

        // -- Modification --

        /// Insert a new element into the heap.
        /// Time: O(log_d n) | Space: O(1) amortized
        pub fn insert(self: *Self, data: T) !void {
            try self.items.append(self.allocator, data);
            self.siftUp(self.items.items.len - 1);
        }

        /// Remove and return the minimum element.
        /// Time: O(d * log_d n) | Space: O(1)
        pub fn extractMin(self: *Self) ?T {
            if (self.items.items.len == 0) return null;

            const min = self.items.items[0];

            if (self.items.items.len == 1) {
                _ = self.items.pop();
                return min;
            }

            self.items.items[0] = self.items.pop();
            self.siftDown(0);

            return min;
        }

        /// Remove element at given index.
        /// Time: O(d * log_d n) | Space: O(1)
        pub fn remove(self: *Self, index: usize) ?T {
            if (index >= self.items.items.len) return null;

            const data = self.items.items[index];

            if (index == self.items.items.len - 1) {
                _ = self.items.pop();
                return data;
            }

            const last = self.items.pop();
            const old = self.items.items[index];
            self.items.items[index] = last;

            // Determine whether to sift up or down
            if (lessThan(self.context, last, old)) {
                self.siftUp(index);
            } else {
                self.siftDown(index);
            }

            return data;
        }

        /// Update the value at a given index and restore heap property.
        /// Time: O(log_d n) | Space: O(1)
        pub fn update(self: *Self, index: usize, new_data: T) void {
            if (index >= self.items.items.len) return;

            const old = self.items.items[index];
            self.items.items[index] = new_data;

            if (lessThan(self.context, new_data, old)) {
                self.siftUp(index);
            } else if (lessThan(self.context, old, new_data)) {
                self.siftDown(index);
            }
        }

        /// Clear all elements from the heap.
        pub fn clear(self: *Self) void {
            self.items.clearRetainingCapacity();
        }

        // -- Lookup --

        /// Return the minimum element without removing it.
        /// Time: O(1) | Space: O(1)
        pub fn peekMin(self: *const Self) ?T {
            return if (self.items.items.len > 0) self.items.items[0] else null;
        }

        // -- Bulk Operations --

        /// Convert heap to a sorted slice (heap sort).
        /// Time: O(n * d * log_d n) | Space: O(n)
        pub fn toSortedSlice(self: *Self, allocator: Allocator) ![]T {
            var sorted: std.ArrayList(T) = .{};
            errdefer sorted.deinit(allocator);

            var temp = try self.clone();
            defer temp.deinit();

            while (temp.extractMin()) |min| {
                try sorted.append(allocator, min);
            }

            return sorted.toOwnedSlice(allocator);
        }

        // -- Private Helpers --

        fn parentIndex(self: *const Self, index: usize) usize {
            _ = self;
            if (index == 0) return 0;
            return (index - 1) / d;
        }

        fn firstChildIndex(self: *const Self, index: usize) usize {
            _ = self;
            return d * index + 1;
        }

        fn siftUp(self: *Self, start_index: usize) void {
            var index = start_index;
            const item = self.items.items[index];

            while (index > 0) {
                const parent = self.parentIndex(index);
                if (!lessThan(self.context, item, self.items.items[parent])) {
                    break;
                }
                self.items.items[index] = self.items.items[parent];
                index = parent;
            }

            self.items.items[index] = item;
        }

        fn siftDown(self: *Self, start_index: usize) void {
            var index = start_index;
            const item = self.items.items[index];
            const half = self.items.items.len / d;

            while (index < half) {
                var min_child_index = self.firstChildIndex(index);
                const child_end = @min(min_child_index + d, self.items.items.len);

                // Find minimum among children
                var min_child = self.items.items[min_child_index];
                var i = min_child_index + 1;
                while (i < child_end) : (i += 1) {
                    if (lessThan(self.context, self.items.items[i], min_child)) {
                        min_child = self.items.items[i];
                        min_child_index = i;
                    }
                }

                if (!lessThan(self.context, min_child, item)) {
                    break;
                }

                self.items.items[index] = min_child;
                index = min_child_index;
            }

            self.items.items[index] = item;
        }

        // -- Debug --

        /// Validate heap invariants.
        pub fn validate(self: *const Self) !void {
            var i: usize = 0;
            while (i < self.items.items.len) : (i += 1) {
                const first_child = self.firstChildIndex(i);
                var child_idx = first_child;
                while (child_idx < first_child + d and child_idx < self.items.items.len) : (child_idx += 1) {
                    if (lessThan(self.context, self.items.items[child_idx], self.items.items[i])) {
                        return error.HeapPropertyViolation;
                    }
                }
            }
        }

        /// Format the heap for debugging.
        pub fn format(self: *const Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;
            try writer.print("DaryHeap(d={}, size={})", .{ d, self.items.items.len });
        }
    };
}

// -- Tests --

fn testLessThan(_: void, a: i32, b: i32) bool {
    return a < b;
}

test "DaryHeap(2): basic operations (binary heap)" {
    var heap = DaryHeap(i32, 2, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    try testing.expect(heap.isEmpty());
    try testing.expectEqual(@as(usize, 0), heap.count());
    try testing.expectEqual(@as(?i32, null), heap.peekMin());

    try heap.insert(5);
    try testing.expect(!heap.isEmpty());
    try testing.expectEqual(@as(usize, 1), heap.count());
    try testing.expectEqual(@as(?i32, 5), heap.peekMin());
    try heap.validate();

    try heap.insert(3);
    try testing.expectEqual(@as(?i32, 3), heap.peekMin());
    try heap.validate();

    try heap.insert(7);
    try testing.expectEqual(@as(?i32, 3), heap.peekMin());
    try heap.validate();

    try testing.expectEqual(@as(?i32, 3), heap.extractMin());
    try testing.expectEqual(@as(?i32, 5), heap.peekMin());
    try heap.validate();

    try testing.expectEqual(@as(?i32, 5), heap.extractMin());
    try testing.expectEqual(@as(?i32, 7), heap.peekMin());
    try heap.validate();

    try testing.expectEqual(@as(?i32, 7), heap.extractMin());
    try testing.expect(heap.isEmpty());
}

test "DaryHeap(4): basic operations (quaternary heap)" {
    var heap = DaryHeap(i32, 4, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    try heap.insert(5);
    try heap.insert(3);
    try heap.insert(7);
    try heap.insert(1);
    try heap.insert(9);
    try heap.insert(2);

    try testing.expectEqual(@as(?i32, 1), heap.peekMin());
    try heap.validate();

    try testing.expectEqual(@as(?i32, 1), heap.extractMin());
    try testing.expectEqual(@as(?i32, 2), heap.extractMin());
    try testing.expectEqual(@as(?i32, 3), heap.extractMin());
    try testing.expectEqual(@as(?i32, 5), heap.extractMin());
    try testing.expectEqual(@as(?i32, 7), heap.extractMin());
    try testing.expectEqual(@as(?i32, 9), heap.extractMin());
    try testing.expect(heap.isEmpty());
}

test "DaryHeap(8): basic operations (octary heap)" {
    var heap = DaryHeap(i32, 8, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    try heap.insert(50);
    try heap.insert(30);
    try heap.insert(70);
    try heap.insert(10);
    try heap.insert(90);
    try heap.insert(20);

    try testing.expectEqual(@as(?i32, 10), heap.peekMin());
    try heap.validate();
}

test "DaryHeap(4): heapify from slice" {
    const data = [_]i32{ 5, 3, 7, 1, 9, 2, 4, 6, 8 };
    var heap = try DaryHeap(i32, 4, void, testLessThan).initFromSlice(testing.allocator, {}, &data);
    defer heap.deinit();

    try testing.expectEqual(@as(usize, 9), heap.count());
    try testing.expectEqual(@as(?i32, 1), heap.peekMin());
    try heap.validate();

    var prev: i32 = -1;
    while (heap.extractMin()) |val| {
        try testing.expect(val >= prev);
        prev = val;
    }
}

test "DaryHeap(3): sorted insertion (ternary heap)" {
    var heap = DaryHeap(i32, 3, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    const n = 100;
    var i: i32 = 0;
    while (i < n) : (i += 1) {
        try heap.insert(i);
    }

    try testing.expectEqual(@as(usize, n), heap.count());
    try heap.validate();

    i = 0;
    while (i < n) : (i += 1) {
        try testing.expectEqual(@as(?i32, i), heap.extractMin());
    }
    try testing.expect(heap.isEmpty());
}

test "DaryHeap(4): reverse sorted insertion" {
    var heap = DaryHeap(i32, 4, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    const n = 100;
    var i: i32 = n - 1;
    while (i >= 0) : (i -= 1) {
        try heap.insert(i);
    }

    try testing.expectEqual(@as(usize, n), heap.count());
    try heap.validate();

    i = 0;
    while (i < n) : (i += 1) {
        try testing.expectEqual(@as(?i32, i), heap.extractMin());
    }
    try testing.expect(heap.isEmpty());
}

test "DaryHeap(4): random insertion" {
    var heap = DaryHeap(i32, 4, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const n = 1000;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const val = random.intRangeAtMost(i32, 0, 9999);
        try heap.insert(val);
    }

    try testing.expectEqual(@as(usize, n), heap.count());
    try heap.validate();

    var prev: i32 = -1;
    while (heap.extractMin()) |val| {
        try testing.expect(val >= prev);
        prev = val;
    }
}

test "DaryHeap(4): clone" {
    var heap = DaryHeap(i32, 4, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    try heap.insert(5);
    try heap.insert(3);
    try heap.insert(7);

    var cloned = try heap.clone();
    defer cloned.deinit();

    try testing.expectEqual(heap.count(), cloned.count());
    try testing.expectEqual(heap.peekMin(), cloned.peekMin());
    try cloned.validate();

    try testing.expectEqual(@as(?i32, 3), heap.extractMin());
    try testing.expectEqual(@as(?i32, 3), cloned.extractMin());

    try testing.expectEqual(@as(?i32, 5), heap.extractMin());
    try testing.expectEqual(@as(?i32, 5), cloned.extractMin());

    try testing.expectEqual(@as(?i32, 7), heap.extractMin());
    try testing.expectEqual(@as(?i32, 7), cloned.extractMin());
}

test "DaryHeap(4): update" {
    var heap = DaryHeap(i32, 4, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    try heap.insert(10);
    try heap.insert(20);
    try heap.insert(30);
    try heap.insert(40);

    try testing.expectEqual(@as(?i32, 10), heap.peekMin());

    heap.update(0, 50); // Update min to 50
    try heap.validate();
    try testing.expectEqual(@as(?i32, 20), heap.peekMin());

    heap.update(heap.count() - 1, 5); // Update last to 5
    try heap.validate();
    try testing.expectEqual(@as(?i32, 5), heap.peekMin());
}

test "DaryHeap(4): remove" {
    var heap = DaryHeap(i32, 4, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    try heap.insert(10);
    try heap.insert(20);
    try heap.insert(30);
    try heap.insert(40);
    try heap.insert(50);

    try testing.expectEqual(@as(usize, 5), heap.count());

    _ = heap.remove(2); // Remove element at index 2
    try testing.expectEqual(@as(usize, 4), heap.count());
    try heap.validate();

    _ = heap.remove(0); // Remove min
    try testing.expectEqual(@as(usize, 3), heap.count());
    try heap.validate();
}

test "DaryHeap(4): clear" {
    var heap = DaryHeap(i32, 4, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    try heap.insert(5);
    try heap.insert(3);
    try heap.insert(7);

    heap.clear();
    try testing.expect(heap.isEmpty());
    try testing.expectEqual(@as(?i32, null), heap.peekMin());
}

test "DaryHeap(4): toSortedSlice" {
    var heap = DaryHeap(i32, 4, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    try heap.insert(5);
    try heap.insert(3);
    try heap.insert(7);
    try heap.insert(1);
    try heap.insert(9);

    const sorted = try heap.toSortedSlice(testing.allocator);
    defer testing.allocator.free(sorted);

    try testing.expectEqualSlices(i32, &[_]i32{ 1, 3, 5, 7, 9 }, sorted);
}

test "DaryHeap(4): empty operations" {
    var heap = DaryHeap(i32, 4, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    try testing.expectEqual(@as(?i32, null), heap.extractMin());
    try testing.expectEqual(@as(?i32, null), heap.peekMin());
    try heap.validate();
}

test "DaryHeap(4): single element" {
    var heap = DaryHeap(i32, 4, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    try heap.insert(42);
    try testing.expectEqual(@as(?i32, 42), heap.peekMin());
    try heap.validate();

    try testing.expectEqual(@as(?i32, 42), heap.extractMin());
    try testing.expect(heap.isEmpty());
}

test "DaryHeap(4): duplicates" {
    var heap = DaryHeap(i32, 4, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    try heap.insert(5);
    try heap.insert(5);
    try heap.insert(5);

    try testing.expectEqual(@as(usize, 3), heap.count());
    try heap.validate();

    try testing.expectEqual(@as(?i32, 5), heap.extractMin());
    try testing.expectEqual(@as(?i32, 5), heap.extractMin());
    try testing.expectEqual(@as(?i32, 5), heap.extractMin());
    try testing.expect(heap.isEmpty());
}

test "DaryHeap(4): memory leak detection" {
    var heap = DaryHeap(i32, 4, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try heap.insert(@intCast(i));
    }

    while (!heap.isEmpty()) {
        _ = heap.extractMin();
    }
}

test "DaryHeap(4): capacity reservation" {
    var heap = DaryHeap(i32, 4, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    try heap.ensureTotalCapacity(100);

    var i: usize = 0;
    while (i < 50) : (i += 1) {
        try heap.insert(@intCast(i));
    }

    try testing.expectEqual(@as(usize, 50), heap.count());
}

test "DaryHeap(16): high branching factor" {
    var heap = DaryHeap(i32, 16, void, testLessThan).init(testing.allocator, {});
    defer heap.deinit();

    var prng = std.Random.DefaultPrng.init(123);
    const random = prng.random();

    const n = 500;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const val = random.intRangeAtMost(i32, 0, 9999);
        try heap.insert(val);
    }

    try heap.validate();

    var prev: i32 = -1;
    while (heap.extractMin()) |val| {
        try testing.expect(val >= prev);
        prev = val;
    }
}
