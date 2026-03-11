const std = @import("std");
const Allocator = std.mem.Allocator;

/// MergeSort - Stable divide-and-conquer sorting algorithm
///
/// Time Complexity:
///   Best:    O(n log n)
///   Average: O(n log n)
///   Worst:   O(n log n)
///
/// Space Complexity: O(n) - requires auxiliary buffer for merging
///
/// Characteristics:
/// - Stable: preserves relative order of equal elements
/// - Not in-place: requires auxiliary space
/// - Divide-and-conquer: recursively splits and merges
/// - Predictable: always O(n log n) regardless of input
///
/// Note: This is the standard top-down merge sort with auxiliary buffer.
/// For "in-place" variants with O(1) space but worse constants, see block-based merging.
pub fn MergeSort(comptime T: type, comptime Context: type, comptime lessThanFn: fn (ctx: Context, a: T, b: T) bool) type {
    return struct {
        allocator: Allocator,
        context: Context,

        const Self = @This();

        /// Initialize merge sort instance
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator, context: Context) Self {
            return .{
                .allocator = allocator,
                .context = context,
            };
        }

        /// Sort a slice using merge sort
        /// Time: O(n log n) | Space: O(n)
        pub fn sort(self: *Self, items: []T) !void {
            if (items.len <= 1) return;

            // Allocate auxiliary buffer
            const aux = try self.allocator.alloc(T, items.len);
            defer self.allocator.free(aux);

            self.sortRange(items, aux, 0, items.len);
        }

        /// Recursively sort a range
        fn sortRange(self: *Self, items: []T, aux: []T, left: usize, right: usize) void {
            if (right - left <= 1) return;

            const mid = left + (right - left) / 2;
            self.sortRange(items, aux, left, mid);
            self.sortRange(items, aux, mid, right);
            self.merge(items, aux, left, mid, right);
        }

        /// Merge two sorted ranges
        fn merge(self: *Self, items: []T, aux: []T, left: usize, mid: usize, right: usize) void {
            // Copy to auxiliary array
            var i: usize = left;
            while (i < right) : (i += 1) {
                aux[i] = items[i];
            }

            // Merge back to original array
            var l = left;
            var r = mid;
            i = left;

            while (l < mid and r < right) {
                if (lessThanFn(self.context, aux[l], aux[r])) {
                    items[i] = aux[l];
                    l += 1;
                } else {
                    items[i] = aux[r];
                    r += 1;
                }
                i += 1;
            }

            // Copy remaining left elements
            while (l < mid) : (l += 1) {
                items[i] = aux[l];
                i += 1;
            }

            // Right elements are already in place
        }
    };
}

/// Bottom-up iterative merge sort (no recursion)
/// Time: O(n log n) | Space: O(n)
/// Better cache locality than top-down in some cases
pub fn MergeSortBottomUp(comptime T: type, comptime Context: type, comptime lessThanFn: fn (ctx: Context, a: T, b: T) bool) type {
    return struct {
        allocator: Allocator,
        context: Context,

        const Self = @This();

        pub fn init(allocator: Allocator, context: Context) Self {
            return .{
                .allocator = allocator,
                .context = context,
            };
        }

        /// Sort using bottom-up merge sort
        /// Time: O(n log n) | Space: O(n)
        pub fn sort(self: *Self, items: []T) !void {
            if (items.len <= 1) return;

            const aux = try self.allocator.alloc(T, items.len);
            defer self.allocator.free(aux);

            var width: usize = 1;
            while (width < items.len) : (width *= 2) {
                var left: usize = 0;
                while (left < items.len) : (left += 2 * width) {
                    const mid = @min(left + width, items.len);
                    const right = @min(left + 2 * width, items.len);
                    if (mid < right) {
                        self.merge(items, aux, left, mid, right);
                    }
                }
            }
        }

        fn merge(self: *Self, items: []T, aux: []T, left: usize, mid: usize, right: usize) void {
            // Copy to auxiliary array
            var i: usize = left;
            while (i < right) : (i += 1) {
                aux[i] = items[i];
            }

            // Merge back
            var l = left;
            var r = mid;
            i = left;

            while (l < mid and r < right) {
                if (lessThanFn(self.context, aux[l], aux[r])) {
                    items[i] = aux[l];
                    l += 1;
                } else {
                    items[i] = aux[r];
                    r += 1;
                }
                i += 1;
            }

            while (l < mid) : (l += 1) {
                items[i] = aux[l];
                i += 1;
            }
        }
    };
}

/// Natural merge sort - exploits existing runs in data
/// Time: O(n log n) worst, O(n) best (already sorted) | Space: O(n)
/// Adaptive variant that performs well on partially sorted data
pub fn NaturalMergeSort(comptime T: type, comptime Context: type, comptime lessThanFn: fn (ctx: Context, a: T, b: T) bool) type {
    return struct {
        allocator: Allocator,
        context: Context,

        const Self = @This();

        pub fn init(allocator: Allocator, context: Context) Self {
            return .{
                .allocator = allocator,
                .context = context,
            };
        }

        /// Sort using natural merge sort (exploits existing order)
        /// Time: O(n) best, O(n log n) worst | Space: O(n)
        pub fn sort(self: *Self, items: []T) !void {
            if (items.len <= 1) return;

            const aux = try self.allocator.alloc(T, items.len);
            defer self.allocator.free(aux);

            while (true) {
                const runs = try self.findRuns(items);
                defer runs.deinit(self.allocator);

                if (runs.items.len <= 1) break; // Already sorted

                self.mergeRuns(items, aux, runs.items);
            }
        }

        const Run = struct { start: usize, end: usize };

        fn findRuns(self: *Self, items: []T) !std.ArrayListUnmanaged(Run) {
            var runs = std.ArrayListUnmanaged(Run){};

            var i: usize = 0;
            while (i < items.len) {
                const start = i;
                i += 1;

                // Find end of ascending run
                while (i < items.len and !lessThanFn(self.context, items[i], items[i - 1])) {
                    i += 1;
                }

                try runs.append(self.allocator, .{ .start = start, .end = i });
            }

            return runs;
        }

        fn mergeRuns(self: *Self, items: []T, aux: []T, runs: []const Run) void {
            var i: usize = 0;
            while (i + 1 < runs.len) : (i += 2) {
                self.merge(items, aux, runs[i].start, runs[i].end, runs[i + 1].end);
            }
        }

        fn merge(self: *Self, items: []T, aux: []T, left: usize, mid: usize, right: usize) void {
            var i: usize = left;
            while (i < right) : (i += 1) {
                aux[i] = items[i];
            }

            var l = left;
            var r = mid;
            i = left;

            while (l < mid and r < right) {
                if (lessThanFn(self.context, aux[l], aux[r])) {
                    items[i] = aux[l];
                    l += 1;
                } else {
                    items[i] = aux[r];
                    r += 1;
                }
                i += 1;
            }

            while (l < mid) : (l += 1) {
                items[i] = aux[l];
                i += 1;
            }
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

fn lessThanU32(ctx: void, a: u32, b: u32) bool {
    _ = ctx;
    return a < b;
}

fn lessThanI32(ctx: void, a: i32, b: i32) bool {
    _ = ctx;
    return a < b;
}

test "MergeSort: empty array" {
    var sorter = MergeSort(u32, void, lessThanU32).init(testing.allocator, {});
    var arr = [_]u32{};
    try sorter.sort(&arr);
    try testing.expectEqual(0, arr.len);
}

test "MergeSort: single element" {
    var sorter = MergeSort(u32, void, lessThanU32).init(testing.allocator, {});
    var arr = [_]u32{42};
    try sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{42}, &arr);
}

test "MergeSort: already sorted" {
    var sorter = MergeSort(u32, void, lessThanU32).init(testing.allocator, {});
    var arr = [_]u32{ 1, 2, 3, 4, 5 };
    try sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3, 4, 5 }, &arr);
}

test "MergeSort: reverse sorted" {
    var sorter = MergeSort(u32, void, lessThanU32).init(testing.allocator, {});
    var arr = [_]u32{ 5, 4, 3, 2, 1 };
    try sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3, 4, 5 }, &arr);
}

test "MergeSort: random order" {
    var sorter = MergeSort(u32, void, lessThanU32).init(testing.allocator, {});
    var arr = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    try sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 1, 2, 3, 3, 4, 5, 5, 6, 9 }, &arr);
}

test "MergeSort: duplicates" {
    var sorter = MergeSort(u32, void, lessThanU32).init(testing.allocator, {});
    var arr = [_]u32{ 3, 3, 3, 1, 1, 2, 2, 2 };
    try sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 1, 2, 2, 2, 3, 3, 3 }, &arr);
}

test "MergeSort: signed integers" {
    var sorter = MergeSort(i32, void, lessThanI32).init(testing.allocator, {});
    var arr = [_]i32{ -5, 3, -2, 0, 8, -10 };
    try sorter.sort(&arr);
    try testing.expectEqualSlices(i32, &[_]i32{ -10, -5, -2, 0, 3, 8 }, &arr);
}

test "MergeSort: stability check" {
    const Item = struct {
        key: u32,
        value: u32,

        fn lessThan(ctx: void, a: @This(), b: @This()) bool {
            _ = ctx;
            return a.key < b.key;
        }
    };

    var sorter = MergeSort(Item, void, Item.lessThan).init(testing.allocator, {});
    var arr = [_]Item{
        .{ .key = 3, .value = 1 },
        .{ .key = 1, .value = 2 },
        .{ .key = 3, .value = 3 },
        .{ .key = 1, .value = 4 },
    };

    try sorter.sort(&arr);

    // Check sorted by key
    try testing.expectEqual(1, arr[0].key);
    try testing.expectEqual(1, arr[1].key);
    try testing.expectEqual(3, arr[2].key);
    try testing.expectEqual(3, arr[3].key);

    // Check stability: items with same key preserve original order
    try testing.expectEqual(2, arr[0].value); // first key=1
    try testing.expectEqual(4, arr[1].value); // second key=1
    try testing.expectEqual(1, arr[2].value); // first key=3
    try testing.expectEqual(3, arr[3].value); // second key=3
}

test "MergeSort: stress test (10k elements)" {
    var sorter = MergeSort(u32, void, lessThanU32).init(testing.allocator, {});

    const n = 10_000;
    var arr = try testing.allocator.alloc(u32, n);
    defer testing.allocator.free(arr);

    // Fill with pseudo-random values
    var seed: u32 = 42;
    for (arr, 0..) |*item, i| {
        seed = seed *% 1103515245 +% 12345;
        item.* = @as(u32, @intCast(i)) ^ seed;
    }

    try sorter.sort(arr);

    // Verify sorted
    for (arr[0 .. arr.len - 1], 1..) |item, i| {
        try testing.expect(item <= arr[i]);
    }
}

// ============================================================================
// Bottom-Up MergeSort Tests
// ============================================================================

test "MergeSortBottomUp: basic functionality" {
    var sorter = MergeSortBottomUp(u32, void, lessThanU32).init(testing.allocator, {});
    var arr = [_]u32{ 5, 2, 8, 1, 9, 3 };
    try sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3, 5, 8, 9 }, &arr);
}

test "MergeSortBottomUp: empty and single" {
    var sorter = MergeSortBottomUp(u32, void, lessThanU32).init(testing.allocator, {});

    var empty = [_]u32{};
    try sorter.sort(&empty);
    try testing.expectEqual(0, empty.len);

    var single = [_]u32{42};
    try sorter.sort(&single);
    try testing.expectEqualSlices(u32, &[_]u32{42}, &single);
}

test "MergeSortBottomUp: power of two size" {
    var sorter = MergeSortBottomUp(u32, void, lessThanU32).init(testing.allocator, {});
    var arr = [_]u32{ 8, 7, 6, 5, 4, 3, 2, 1 }; // size = 8 = 2^3
    try sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 }, &arr);
}

test "MergeSortBottomUp: non-power of two size" {
    var sorter = MergeSortBottomUp(u32, void, lessThanU32).init(testing.allocator, {});
    var arr = [_]u32{ 5, 4, 3, 2, 1, 0 }; // size = 6
    try sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 2, 3, 4, 5 }, &arr);
}

// ============================================================================
// Natural MergeSort Tests
// ============================================================================

test "NaturalMergeSort: already sorted (best case)" {
    var sorter = NaturalMergeSort(u32, void, lessThanU32).init(testing.allocator, {});
    var arr = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 }, &arr);
}

test "NaturalMergeSort: partially sorted runs" {
    var sorter = NaturalMergeSort(u32, void, lessThanU32).init(testing.allocator, {});
    var arr = [_]u32{ 1, 3, 5, 2, 4, 6 }; // two runs: [1,3,5], [2,4,6]
    try sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3, 4, 5, 6 }, &arr);
}

test "NaturalMergeSort: reverse sorted (worst case)" {
    var sorter = NaturalMergeSort(u32, void, lessThanU32).init(testing.allocator, {});
    var arr = [_]u32{ 8, 7, 6, 5, 4, 3, 2, 1 };
    try sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 }, &arr);
}

test "NaturalMergeSort: mixed runs" {
    var sorter = NaturalMergeSort(u32, void, lessThanU32).init(testing.allocator, {});
    var arr = [_]u32{ 3, 5, 7, 1, 2, 4, 6, 8 }; // runs: [3,5,7], [1,2,4,6,8]
    try sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 }, &arr);
}
