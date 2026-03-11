const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// TimSort - Hybrid stable sorting algorithm combining merge sort and insertion sort.
///
/// TimSort is an adaptive, stable sorting algorithm derived from merge sort and insertion sort.
/// It was designed to perform well on many kinds of real-world data. It finds runs of consecutive
/// ordered elements (runs), possibly extends them to a minimum size using insertion sort, and then
/// merges these runs using a merge sort variant.
///
/// Key features:
/// - Stable: preserves relative order of equal elements
/// - Adaptive: O(n) on already-sorted or reverse-sorted data
/// - Optimized for real-world data with natural ordering patterns
///
/// Time Complexity: O(n log n) worst case, O(n) best case
/// Space Complexity: O(n) auxiliary space for merge buffer
///
/// Generic parameters:
/// - T: Element type
/// - Context: Context type for comparison
/// - compareFn: Comparison function (a, b) -> std.math.Order
pub fn TimSort(
    comptime T: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: T, b: T) std.math.Order,
) type {
    return struct {
        const Self = @This();

        // Minimum size of a run - runs smaller than this are extended using insertion sort
        const MIN_MERGE: usize = 32;

        // Maximum size of run stack
        const MAX_MERGE_PENDING: usize = 85; // log2(max_usize) + some margin

        /// A run is a sequence of consecutive ordered elements
        const Run = struct {
            start: usize,
            len: usize,
        };

        /// Sort a slice in-place using TimSort.
        ///
        /// Time: O(n log n) worst, O(n) best | Space: O(n)
        pub fn sort(allocator: Allocator, items: []T, context: Context) !void {
            if (items.len < 2) return;

            // For small arrays, just use insertion sort
            if (items.len < 64) {
                insertionSort(items, 0, items.len, context);
                return;
            }

            const min_run = minRunLength(items.len);

            var run_stack: [MAX_MERGE_PENDING]Run = undefined;
            var run_count: usize = 0;

            var pos: usize = 0;
            while (pos < items.len) {
                // Find next run
                var run_len = countRunAndMakeAscending(items, pos, context);

                // If run is too short, extend it to min_run using insertion sort
                if (run_len < min_run) {
                    const force_len = @min(min_run, items.len - pos);
                    insertionSort(items, pos, pos + force_len, context);
                    run_len = force_len;
                }

                // Push run onto stack
                run_stack[run_count] = .{ .start = pos, .len = run_len };
                run_count += 1;
                pos += run_len;

                // Merge runs on stack to maintain invariants
                try mergeCollapse(allocator, items, run_stack[0..run_count], &run_count, context);
            }

            // Force merge all remaining runs
            try mergeForceCollapse(allocator, items, run_stack[0..run_count], &run_count, context);
        }

        /// Compute minimum run length for merge sort.
        /// Returns value k where 32 <= k <= 64 and n/k is close to a power of 2.
        fn minRunLength(n: usize) usize {
            var nn = n;
            var r: usize = 0;

            while (nn >= MIN_MERGE) {
                r |= nn & 1;
                nn >>= 1;
            }

            return nn + r;
        }

        /// Count length of run starting at pos and make it ascending if it's descending.
        /// Returns length of the run.
        fn countRunAndMakeAscending(items: []T, pos: usize, context: Context) usize {
            if (pos + 1 >= items.len) return 1;

            var run_end = pos + 1;

            // Strictly descending?
            if (compareFn(context, items[run_end], items[pos]) == .lt) {
                run_end += 1;
                while (run_end < items.len and compareFn(context, items[run_end], items[run_end - 1]) == .lt) {
                    run_end += 1;
                }
                // Reverse the run
                reverseRange(items, pos, run_end);
            } else {
                // Ascending or equal
                run_end += 1;
                while (run_end < items.len and compareFn(context, items[run_end], items[run_end - 1]) != .lt) {
                    run_end += 1;
                }
            }

            return run_end - pos;
        }

        /// Reverse a range of elements in-place.
        fn reverseRange(items: []T, start: usize, end: usize) void {
            var left = start;
            var right = end - 1;

            while (left < right) {
                std.mem.swap(T, &items[left], &items[right]);
                left += 1;
                right -= 1;
            }
        }

        /// Insertion sort for small runs.
        /// Sorts items[start..end] in-place.
        fn insertionSort(items: []T, start: usize, end: usize, context: Context) void {
            var i = start + 1;
            while (i < end) : (i += 1) {
                const pivot = items[i];
                var j = i;

                // Shift elements right until we find the insertion point
                while (j > start and compareFn(context, items[j - 1], pivot) == .gt) {
                    items[j] = items[j - 1];
                    j -= 1;
                }

                items[j] = pivot;
            }
        }

        /// Merge pending runs to maintain invariants.
        /// Invariants:
        ///   1. run_len[i-2] > run_len[i-1] + run_len[i]
        ///   2. run_len[i-1] > run_len[i]
        fn mergeCollapse(
            allocator: Allocator,
            items: []T,
            runs: []Run,
            run_count: *usize,
            context: Context,
        ) !void {
            while (run_count.* > 1) {
                const n = run_count.*;

                // Check if we need to merge
                if (n >= 3) {
                    const len_i = runs[n - 1].len;
                    const len_i1 = runs[n - 2].len;
                    const len_i2 = runs[n - 3].len;

                    // Invariant 1 violated?
                    if (len_i2 <= len_i1 + len_i) {
                        // Merge smaller of (i-2, i-1) with i
                        if (len_i2 < len_i) {
                            try mergeAt(allocator, items, runs, n - 3, context);
                        } else {
                            try mergeAt(allocator, items, runs, n - 2, context);
                        }
                        run_count.* -= 1;
                        continue;
                    }
                }

                if (n >= 2) {
                    const len_i = runs[n - 1].len;
                    const len_i1 = runs[n - 2].len;

                    // Invariant 2 violated?
                    if (len_i1 <= len_i) {
                        try mergeAt(allocator, items, runs, n - 2, context);
                        run_count.* -= 1;
                        continue;
                    }
                }

                break;
            }
        }

        /// Force merge all remaining runs.
        fn mergeForceCollapse(
            allocator: Allocator,
            items: []T,
            runs: []Run,
            run_count: *usize,
            context: Context,
        ) !void {
            while (run_count.* > 1) {
                const n = run_count.*;

                if (n >= 3 and runs[n - 3].len < runs[n - 1].len) {
                    try mergeAt(allocator, items, runs, n - 3, context);
                } else {
                    try mergeAt(allocator, items, runs, n - 2, context);
                }

                run_count.* -= 1;
            }
        }

        /// Merge run at index i with run at index i+1.
        /// Updates run[i] to represent the merged run and shifts subsequent runs.
        fn mergeAt(
            allocator: Allocator,
            items: []T,
            runs: []Run,
            i: usize,
            context: Context,
        ) !void {
            const run1 = runs[i];
            const run2 = runs[i + 1];

            // Merge run1 and run2
            try mergeRuns(allocator, items, run1.start, run1.len, run2.len, context);

            // Update run stack
            runs[i].len = run1.len + run2.len;

            // Shift subsequent runs down
            if (i + 2 < runs.len) {
                var j = i + 1;
                while (j + 1 < runs.len) : (j += 1) {
                    runs[j] = runs[j + 1];
                }
            }
        }

        /// Merge two consecutive runs.
        /// items[start..start+len1] and items[start+len1..start+len1+len2]
        fn mergeRuns(
            allocator: Allocator,
            items: []T,
            start: usize,
            len1: usize,
            len2: usize,
            context: Context,
        ) !void {
            // Allocate temporary buffer for merge
            const temp = try allocator.alloc(T, len1);
            defer allocator.free(temp);

            // Copy first run to temp
            @memcpy(temp, items[start .. start + len1]);

            var i: usize = 0; // Index in temp
            var j: usize = start + len1; // Index in second run
            var k: usize = start; // Index in result

            const end2 = start + len1 + len2;

            // Merge temp and second run back into items
            while (i < len1 and j < end2) {
                if (compareFn(context, temp[i], items[j]) != .gt) {
                    items[k] = temp[i];
                    i += 1;
                } else {
                    items[k] = items[j];
                    j += 1;
                }
                k += 1;
            }

            // Copy remaining elements from temp
            while (i < len1) : (i += 1) {
                items[k] = temp[i];
                k += 1;
            }

            // No need to copy remaining from second run - they're already in place
        }
    };
}

/// Convenience wrapper for sorting with default comparison
pub fn sort(comptime T: type, allocator: Allocator, items: []T) !void {
    const Context = struct {
        pub fn compare(_: @This(), a: T, b: T) std.math.Order {
            return std.math.order(a, b);
        }
    };
    const Sorter = TimSort(T, Context, Context.compare);
    return Sorter.sort(allocator, items, .{});
}

// ============================================================================
// Tests
// ============================================================================

test "TimSort: empty array" {
    const allocator = testing.allocator;
    var items: [0]i32 = undefined;
    try sort(i32, allocator, &items);
}

test "TimSort: single element" {
    const allocator = testing.allocator;
    var items = [_]i32{42};
    try sort(i32, allocator, &items);
    try testing.expectEqual(@as(i32, 42), items[0]);
}

test "TimSort: already sorted" {
    const allocator = testing.allocator;
    var items = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    try sort(i32, allocator, &items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "TimSort: reverse sorted" {
    const allocator = testing.allocator;
    var items = [_]i32{ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    try sort(i32, allocator, &items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "TimSort: random array" {
    const allocator = testing.allocator;
    var items = [_]i32{ 3, 7, 1, 9, 2, 8, 4, 6, 5, 10 };
    try sort(i32, allocator, &items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "TimSort: with duplicates" {
    const allocator = testing.allocator;
    var items = [_]i32{ 5, 2, 8, 2, 9, 1, 5, 5, 3, 2 };
    try sort(i32, allocator, &items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "TimSort: stability check" {
    const allocator = testing.allocator;

    const Item = struct {
        key: i32,
        value: u32,
    };

    const Context = struct {
        pub fn compare(_: @This(), a: Item, b: Item) std.math.Order {
            return std.math.order(a.key, b.key);
        }
    };

    var items = [_]Item{
        .{ .key = 3, .value = 1 },
        .{ .key = 1, .value = 2 },
        .{ .key = 3, .value = 3 },
        .{ .key = 1, .value = 4 },
        .{ .key = 2, .value = 5 },
    };

    const Sorter = TimSort(Item, Context, Context.compare);
    try Sorter.sort(allocator, &items, .{});

    // Check sorted by key
    try testing.expectEqual(@as(i32, 1), items[0].key);
    try testing.expectEqual(@as(i32, 1), items[1].key);
    try testing.expectEqual(@as(i32, 2), items[2].key);
    try testing.expectEqual(@as(i32, 3), items[3].key);
    try testing.expectEqual(@as(i32, 3), items[4].key);

    // Check stability: items with same key maintain original order
    try testing.expectEqual(@as(u32, 2), items[0].value); // 1 came second
    try testing.expectEqual(@as(u32, 4), items[1].value); // 1 came fourth
    try testing.expectEqual(@as(u32, 1), items[3].value); // 3 came first
    try testing.expectEqual(@as(u32, 3), items[4].value); // 3 came third
}

test "TimSort: large array" {
    const allocator = testing.allocator;

    const items = try allocator.alloc(i32, 1000);
    defer allocator.free(items);

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (items) |*item| {
        item.* = random.intRangeAtMost(i32, -1000, 1000);
    }

    try sort(i32, allocator, items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "TimSort: alternating pattern" {
    const allocator = testing.allocator;
    var items = [_]i32{ 1, 3, 2, 4, 3, 5, 4, 6, 5, 7 };
    try sort(i32, allocator, &items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "TimSort: many duplicates" {
    const allocator = testing.allocator;
    var items = [_]i32{ 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };
    try sort(i32, allocator, &items);

    for (items) |item| {
        try testing.expectEqual(@as(i32, 5), item);
    }
}

test "TimSort: stress test with runs" {
    const allocator = testing.allocator;

    const items = try allocator.alloc(i32, 10000);
    defer allocator.free(items);

    // Create data with natural runs (ascending and descending sequences)
    var pos: usize = 0;
    var value: i32 = 0;
    while (pos < items.len) {
        const run_len = @min(100, items.len - pos);
        const ascending = (pos / 100) % 2 == 0;

        for (0..run_len) |i| {
            items[pos + i] = if (ascending) value else -value;
            value += 1;
        }

        pos += run_len;
    }

    try sort(i32, allocator, items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}
