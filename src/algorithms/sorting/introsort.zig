const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// IntroSort - Hybrid introspective sorting algorithm.
///
/// IntroSort (Introspective Sort) is a hybrid sorting algorithm that provides both fast
/// average performance and optimal worst-case performance. It begins with quicksort and
/// switches to heapsort when recursion depth exceeds a level based on log(n). This prevents
/// quicksort's O(n²) worst case. For small arrays, it uses insertion sort.
///
/// Key features:
/// - Fast average case like quicksort: O(n log n)
/// - Optimal worst case like heapsort: O(n log n)
/// - No auxiliary space required (in-place)
/// - Unstable (does not preserve relative order of equal elements)
///
/// Time Complexity: O(n log n) worst case, O(n log n) average
/// Space Complexity: O(log n) for recursion stack
///
/// Generic parameters:
/// - T: Element type
/// - Context: Context type for comparison
/// - compareFn: Comparison function (a, b) -> std.math.Order
pub fn IntroSort(
    comptime T: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: T, b: T) std.math.Order,
) type {
    return struct {
        const Self = @This();

        // Threshold for switching to insertion sort
        const INSERTION_THRESHOLD: usize = 16;

        /// Sort a slice in-place using IntroSort.
        ///
        /// Time: O(n log n) worst case | Space: O(log n) stack
        pub fn sort(items: []T, context: Context) void {
            if (items.len < 2) return;

            const max_depth = 2 * std.math.log2_int(usize, items.len);
            introsortLoop(items, 0, items.len, max_depth, context);
            insertionSort(items, 0, items.len, context);
        }

        /// Main introsort loop with depth limit.
        fn introsortLoop(items: []T, start: usize, end: usize, depth_limit: usize, context: Context) void {
            while (end - start > INSERTION_THRESHOLD) {
                if (depth_limit == 0) {
                    // Depth limit reached, switch to heapsort
                    heapsort(items, start, end, context);
                    return;
                }

                // Partition using median-of-three pivot
                const pivot = partition(items, start, end, context);

                // Recurse on smaller partition first to limit stack depth
                const left_size = pivot - start;
                const right_size = end - pivot - 1;

                if (left_size < right_size) {
                    introsortLoop(items, start, pivot, depth_limit - 1, context);
                    // Tail recursion elimination: continue with right partition
                    return introsortLoop(items, pivot + 1, end, depth_limit - 1, context);
                } else {
                    introsortLoop(items, pivot + 1, end, depth_limit - 1, context);
                    // Tail recursion elimination: continue with left partition
                    return introsortLoop(items, start, pivot, depth_limit - 1, context);
                }
            }
        }

        /// Partition using median-of-three pivot selection.
        fn partition(items: []T, start: usize, end: usize, context: Context) usize {
            const mid = start + (end - start) / 2;

            // Median-of-three: sort start, mid, end-1
            if (compareFn(context, items[mid], items[start]) == .lt) {
                std.mem.swap(T, &items[start], &items[mid]);
            }
            if (compareFn(context, items[end - 1], items[start]) == .lt) {
                std.mem.swap(T, &items[start], &items[end - 1]);
            }
            if (compareFn(context, items[mid], items[end - 1]) == .lt) {
                std.mem.swap(T, &items[mid], &items[end - 1]);
            }

            // Use mid as pivot, move it to end-1
            std.mem.swap(T, &items[mid], &items[end - 1]);
            const pivot = items[end - 1];

            var i = start;
            var j = end - 1;

            while (true) {
                // Move i right while items[i] < pivot
                i += 1;
                while (i < j and compareFn(context, items[i], pivot) == .lt) : (i += 1) {}

                // Move j left while items[j] > pivot
                j -= 1;
                while (i < j and compareFn(context, items[j], pivot) == .gt) : (j -= 1) {}

                if (i >= j) break;

                std.mem.swap(T, &items[i], &items[j]);
            }

            // Place pivot in final position
            std.mem.swap(T, &items[i], &items[end - 1]);
            return i;
        }

        /// Heapsort for when recursion depth limit is reached.
        fn heapsort(items: []T, start: usize, end: usize, context: Context) void {
            const n = end - start;

            // Build max heap
            var i: usize = n / 2;
            while (i > 0) : (i -= 1) {
                siftDown(items, start, i - 1, n, context);
            }

            // Extract elements from heap
            i = n;
            while (i > 1) : (i -= 1) {
                std.mem.swap(T, &items[start], &items[start + i - 1]);
                siftDown(items, start, 0, i - 1, context);
            }
        }

        /// Sift down for heapsort.
        fn siftDown(items: []T, start: usize, root: usize, heap_size: usize, context: Context) void {
            var current = root;

            while (2 * current + 1 < heap_size) {
                var child = 2 * current + 1;

                // Choose larger child
                if (child + 1 < heap_size and
                    compareFn(context, items[start + child], items[start + child + 1]) == .lt) {
                    child += 1;
                }

                // If parent is larger than largest child, done
                if (compareFn(context, items[start + current], items[start + child]) != .lt) {
                    break;
                }

                std.mem.swap(T, &items[start + current], &items[start + child]);
                current = child;
            }
        }

        /// Insertion sort for small arrays and final cleanup.
        fn insertionSort(items: []T, start: usize, end: usize, context: Context) void {
            var i = start + 1;
            while (i < end) : (i += 1) {
                const key = items[i];
                var j = i;

                while (j > start and compareFn(context, items[j - 1], key) == .gt) : (j -= 1) {
                    items[j] = items[j - 1];
                }

                items[j] = key;
            }
        }
    };
}

/// Convenience function for sorting with default comparison.
pub fn sort(comptime T: type, items: []T, context: anytype) void {
    const Context = @TypeOf(context);
    const Sorter = IntroSort(T, Context, struct {
        fn compare(ctx: Context, a: T, b: T) std.math.Order {
            _ = ctx;
            return std.math.order(a, b);
        }
    }.compare);
    Sorter.sort(items, context);
}

// ============================================================================
// Tests
// ============================================================================

test "IntroSort - empty array" {
    var items: [0]i32 = undefined;
    const Sorter = IntroSort(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);
    Sorter.sort(&items, {});
}

test "IntroSort - single element" {
    var items = [_]i32{42};
    const Sorter = IntroSort(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);
    Sorter.sort(&items, {});
    try testing.expectEqual(@as(i32, 42), items[0]);
}

test "IntroSort - already sorted" {
    var items = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const Sorter = IntroSort(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);
    Sorter.sort(&items, {});

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "IntroSort - reverse sorted" {
    var items = [_]i32{ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    const Sorter = IntroSort(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);
    Sorter.sort(&items, {});

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "IntroSort - random data" {
    var items = [_]i32{ 3, 7, 1, 9, 2, 5, 8, 4, 6, 10 };
    const Sorter = IntroSort(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);
    Sorter.sort(&items, {});

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "IntroSort - duplicates" {
    var items = [_]i32{ 5, 2, 8, 2, 9, 1, 5, 5, 3, 2 };
    const Sorter = IntroSort(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);
    Sorter.sort(&items, {});

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "IntroSort - all same" {
    var items = [_]i32{ 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 };
    const Sorter = IntroSort(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);
    Sorter.sort(&items, {});

    for (items) |item| {
        try testing.expectEqual(@as(i32, 7), item);
    }
}

test "IntroSort - large array (triggers heapsort)" {
    const allocator = testing.allocator;

    // Create array large enough to trigger depth limit
    const n = 1000;
    var items = try allocator.alloc(i32, n);
    defer allocator.free(items);

    // Fill with reverse sorted data to trigger worst case
    for (0..n) |i| {
        items[i] = @intCast(n - i);
    }

    const Sorter = IntroSort(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);
    Sorter.sort(items, {});

    // Verify sorted
    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "IntroSort - custom comparator (descending)" {
    var items = [_]i32{ 3, 7, 1, 9, 2, 5, 8, 4, 6, 10 };
    const Sorter = IntroSort(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(b, a); // Reversed for descending
        }
    }.cmp);
    Sorter.sort(&items, {});

    // Should be sorted in descending order
    for (0..items.len - 1) |i| {
        try testing.expect(items[i] >= items[i + 1]);
    }
}

test "IntroSort - strings" {
    var items = [_][]const u8{ "zebra", "apple", "mango", "banana", "cherry" };
    const Sorter = IntroSort([]const u8, void, struct {
        fn cmp(_: void, a: []const u8, b: []const u8) std.math.Order {
            return std.mem.order(u8, a, b);
        }
    }.cmp);
    Sorter.sort(&items, {});

    try testing.expectEqualStrings("apple", items[0]);
    try testing.expectEqualStrings("banana", items[1]);
    try testing.expectEqualStrings("cherry", items[2]);
    try testing.expectEqualStrings("mango", items[3]);
    try testing.expectEqualStrings("zebra", items[4]);
}

test "IntroSort - small array (insertion sort path)" {
    var items = [_]i32{ 5, 2, 8, 1, 9 };
    const Sorter = IntroSort(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);
    Sorter.sort(&items, {});

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "IntroSort - stress test" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const n = 10000;
    var items = try allocator.alloc(i32, n);
    defer allocator.free(items);

    // Fill with random data
    for (0..n) |i| {
        items[i] = random.intRangeAtMost(i32, -1000, 1000);
    }

    const Sorter = IntroSort(i32, void, struct {
        fn cmp(_: void, a: i32, b: i32) std.math.Order {
            return std.math.order(a, b);
        }
    }.cmp);
    Sorter.sort(items, {});

    // Verify sorted
    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}
