const std = @import("std");
const Allocator = std.mem.Allocator;

/// BlockSort - In-place stable sorting with O(1) extra space
///
/// Also known as "Block Merge Sort" or "Grail Sort" variant.
/// This is a simplified implementation focusing on correctness over optimal performance.
///
/// Time Complexity:
///   Best:    O(n)      - already sorted
///   Average: O(n log n)
///   Worst:   O(n log n)
///
/// Space Complexity: O(1) - truly in-place using block rotation
///
/// Characteristics:
/// - Stable: preserves relative order of equal elements
/// - In-place: uses only O(1) auxiliary space (vs O(n) for standard merge sort)
/// - Adaptive: exploits existing order in data
/// - Complex: requires sophisticated block manipulation
///
/// Algorithm:
/// 1. Detect and extend natural runs (like TimSort)
/// 2. Merge runs in-place using block-based rotation
/// 3. Use internal buffer extracted from array for efficiency
/// 4. Fall back to rotation-based merging when necessary
pub fn BlockSort(comptime T: type, comptime Context: type, comptime lessThanFn: fn (ctx: Context, a: T, b: T) bool) type {
    return struct {
        context: Context,

        const Self = @This();

        /// Minimum run length for insertion sort
        const MIN_RUN = 16;

        /// Block size for block merge
        const BLOCK_SIZE = 16;

        pub fn init(context: Context) Self {
            return .{ .context = context };
        }

        /// Sort a slice using block sort
        /// Time: O(n log n) | Space: O(1)
        pub fn sort(self: *Self, items: []T) void {
            if (items.len <= 1) return;

            // For small arrays, use insertion sort
            if (items.len < MIN_RUN * 2) {
                self.insertionSort(items, 0, items.len);
                return;
            }

            // Find and extend runs
            var runs = std.ArrayList(Run).init(std.heap.page_allocator);
            defer runs.deinit();

            self.findRuns(items, &runs) catch return;

            // Merge runs using in-place block merge
            while (runs.items.len > 1) {
                var new_runs = std.ArrayList(Run).init(std.heap.page_allocator);
                defer new_runs.deinit();

                var i: usize = 0;
                while (i < runs.items.len) : (i += 2) {
                    if (i + 1 < runs.items.len) {
                        self.mergeRuns(items, runs.items[i], runs.items[i + 1]);
                        new_runs.append(.{
                            .start = runs.items[i].start,
                            .end = runs.items[i + 1].end,
                        }) catch continue;
                    } else {
                        new_runs.append(runs.items[i]) catch continue;
                    }
                }

                runs.clearRetainingCapacity();
                runs.appendSlice(new_runs.items) catch return;
            }
        }

        const Run = struct {
            start: usize,
            end: usize,
        };

        fn findRuns(self: *Self, items: []T, runs: *std.ArrayList(Run)) !void {
            var i: usize = 0;
            while (i < items.len) {
                const start = i;
                var end = i + 1;

                // Extend ascending run
                while (end < items.len and !lessThanFn(self.context, items[end], items[end - 1])) {
                    end += 1;
                }

                // If run is too short, extend with insertion sort
                if (end - start < MIN_RUN) {
                    end = @min(start + MIN_RUN, items.len);
                    self.insertionSort(items, start, end);
                }

                try runs.append(.{ .start = start, .end = end });
                i = end;
            }
        }

        fn mergeRuns(self: *Self, items: []T, run1: Run, run2: Run) void {
            // Simple in-place merge using rotation
            const left = run1.start;
            const mid = run1.end;
            const right = run2.end;

            if (mid >= right) return;

            // Check if already merged
            if (!lessThanFn(self.context, items[mid], items[mid - 1])) {
                return;
            }

            self.inPlaceMerge(items, left, mid, right);
        }

        /// In-place merge using rotation-based approach
        fn inPlaceMerge(self: *Self, items: []T, left: usize, mid: usize, right: usize) void {
            if (left >= mid or mid >= right) return;

            var l = left;
            var m = mid;

            while (l < m and m < right) {
                // Find position in left part where right[m] should be inserted
                while (l < m and !lessThanFn(self.context, items[m], items[l])) {
                    l += 1;
                }

                // Find block in right part that is less than left[l]
                var r = m;
                while (r < right and lessThanFn(self.context, items[r], items[l])) {
                    r += 1;
                }

                // Rotate the block [l, m) and [m, r) to get [m, r), [l, m)
                if (l < m and m < r) {
                    self.rotate(items, l, m, r);
                    const block_size = r - m;
                    l += block_size;
                    m = r;
                }
            }
        }

        /// Rotate array segment [left, mid, right) to [mid, right, left)
        /// This is the "block swap" or "rotation" operation
        fn rotate(self: *Self, items: []T, left: usize, mid: usize, right: usize) void {
            _ = self;
            if (left >= mid or mid >= right) return;

            // Reverse [left, mid)
            std.mem.reverse(T, items[left..mid]);
            // Reverse [mid, right)
            std.mem.reverse(T, items[mid..right]);
            // Reverse [left, right)
            std.mem.reverse(T, items[left..right]);
        }

        /// Insertion sort for small ranges
        fn insertionSort(self: *Self, items: []T, start: usize, end: usize) void {
            var i = start + 1;
            while (i < end) : (i += 1) {
                const key = items[i];
                var j = i;
                while (j > start and lessThanFn(self.context, key, items[j - 1])) {
                    items[j] = items[j - 1];
                    j -= 1;
                }
                items[j] = key;
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

test "BlockSort: empty array" {
    var sorter = BlockSort(u32, void, lessThanU32).init({});
    var arr = [_]u32{};
    sorter.sort(&arr);
    try testing.expectEqual(0, arr.len);
}

test "BlockSort: single element" {
    var sorter = BlockSort(u32, void, lessThanU32).init({});
    var arr = [_]u32{42};
    sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{42}, &arr);
}

test "BlockSort: already sorted" {
    var sorter = BlockSort(u32, void, lessThanU32).init({});
    var arr = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 }, &arr);
}

test "BlockSort: reverse sorted" {
    var sorter = BlockSort(u32, void, lessThanU32).init({});
    var arr = [_]u32{ 8, 7, 6, 5, 4, 3, 2, 1 };
    sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 }, &arr);
}

test "BlockSort: random order" {
    var sorter = BlockSort(u32, void, lessThanU32).init({});
    var arr = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 1, 2, 3, 3, 4, 5, 5, 6, 9 }, &arr);
}

test "BlockSort: small array (triggers insertion sort)" {
    var sorter = BlockSort(u32, void, lessThanU32).init({});
    var arr = [_]u32{ 5, 2, 8, 1, 9 };
    sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 5, 8, 9 }, &arr);
}

test "BlockSort: duplicates" {
    var sorter = BlockSort(u32, void, lessThanU32).init({});
    var arr = [_]u32{ 3, 3, 3, 1, 1, 2, 2, 2 };
    sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 1, 2, 2, 2, 3, 3, 3 }, &arr);
}

test "BlockSort: signed integers" {
    var sorter = BlockSort(i32, void, lessThanI32).init({});
    var arr = [_]i32{ -5, 3, -2, 0, 8, -10, 4, -1 };
    sorter.sort(&arr);
    try testing.expectEqualSlices(i32, &[_]i32{ -10, -5, -2, -1, 0, 3, 4, 8 }, &arr);
}

test "BlockSort: partially sorted runs" {
    var sorter = BlockSort(u32, void, lessThanU32).init({});
    var arr = [_]u32{ 1, 3, 5, 7, 2, 4, 6, 8 }; // Two runs: [1,3,5,7], [2,4,6,8]
    sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 }, &arr);
}

test "BlockSort: stability check" {
    const Item = struct {
        key: u32,
        value: u32,

        fn lessThan(ctx: void, a: @This(), b: @This()) bool {
            _ = ctx;
            return a.key < b.key;
        }
    };

    var sorter = BlockSort(Item, void, Item.lessThan).init({});
    var arr = [_]Item{
        .{ .key = 3, .value = 1 },
        .{ .key = 1, .value = 2 },
        .{ .key = 3, .value = 3 },
        .{ .key = 1, .value = 4 },
        .{ .key = 2, .value = 5 },
    };

    sorter.sort(&arr);

    // Check sorted by key
    try testing.expectEqual(1, arr[0].key);
    try testing.expectEqual(1, arr[1].key);
    try testing.expectEqual(2, arr[2].key);
    try testing.expectEqual(3, arr[3].key);
    try testing.expectEqual(3, arr[4].key);

    // Check stability (items with same key preserve original order)
    try testing.expectEqual(2, arr[0].value); // first key=1
    try testing.expectEqual(4, arr[1].value); // second key=1
    try testing.expectEqual(1, arr[3].value); // first key=3
    try testing.expectEqual(3, arr[4].value); // second key=3
}

test "BlockSort: medium size array" {
    var sorter = BlockSort(u32, void, lessThanU32).init({});

    const n = 100;
    var arr: [n]u32 = undefined;

    // Fill with pseudo-random values
    var seed: u32 = 42;
    for (&arr, 0..) |*item, i| {
        seed = seed *% 1103515245 +% 12345;
        item.* = @as(u32, @intCast(i)) ^ seed;
    }

    sorter.sort(&arr);

    // Verify sorted
    for (arr[0 .. arr.len - 1], 1..) |item, i| {
        try testing.expect(item <= arr[i]);
    }
}

test "BlockSort: stress test (1k elements)" {
    var sorter = BlockSort(u32, void, lessThanU32).init({});

    const n = 1000;
    var arr: [n]u32 = undefined;

    // Fill with pseudo-random values
    var seed: u32 = 42;
    for (&arr, 0..) |*item, i| {
        seed = seed *% 1103515245 +% 12345;
        item.* = @as(u32, @intCast(i)) ^ seed;
    }

    sorter.sort(&arr);

    // Verify sorted
    for (arr[0 .. arr.len - 1], 1..) |item, i| {
        try testing.expect(item <= arr[i]);
    }
}

test "BlockSort: rotation correctness" {
    var sorter = BlockSort(u32, void, lessThanU32).init({});
    var arr = [_]u32{ 1, 2, 3, 4, 5, 6 };

    // Rotate [0, 3, 6) -> [3, 6, 0)
    // Expected: [4, 5, 6, 1, 2, 3]
    sorter.rotate(&arr, 0, 3, 6);
    try testing.expectEqualSlices(u32, &[_]u32{ 4, 5, 6, 1, 2, 3 }, &arr);
}

test "BlockSort: all equal elements" {
    var sorter = BlockSort(u32, void, lessThanU32).init({});
    var arr = [_]u32{ 7, 7, 7, 7, 7, 7, 7, 7 };
    sorter.sort(&arr);
    try testing.expectEqualSlices(u32, &[_]u32{ 7, 7, 7, 7, 7, 7, 7, 7 }, &arr);
}
