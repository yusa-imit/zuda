const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Order = std.math.Order;
const Random = std.Random;

/// QuickSort - Classic divide-and-conquer sorting algorithm.
///
/// QuickSort is one of the fastest general-purpose sorting algorithms in practice.
/// It works by selecting a 'pivot' element and partitioning the array around it,
/// such that elements smaller than the pivot come before it and larger elements
/// come after it. This process is then recursively applied to the sub-arrays.
///
/// Key features:
/// - Unstable: does not preserve relative order of equal elements
/// - In-place: O(log n) space for recursion stack (worst case O(n))
/// - Cache-friendly: good locality of reference
/// - Adaptive: can be optimized for many data patterns
///
/// Time Complexity: O(n log n) average, O(n²) worst case (rare with good pivot selection)
/// Space Complexity: O(log n) average recursion depth, O(n) worst case
///
/// Generic parameters:
/// - T: Element type
/// - Context: Context type for comparison
/// - compareFn: Comparison function (ctx, a, b) -> std.math.Order
pub fn QuickSort(
    comptime T: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: T, b: T) Order,
) type {
    return struct {
        const Self = @This();

        /// Threshold for switching to insertion sort for small subarrays
        const INSERTION_SORT_THRESHOLD: usize = 16;

        /// Sort a slice in-place using classic QuickSort with Hoare partition scheme.
        ///
        /// Uses median-of-three pivot selection and insertion sort for small subarrays.
        ///
        /// Time: O(n log n) average, O(n²) worst | Space: O(log n)
        pub fn sort(items: []T, context: Context) void {
            if (items.len < 2) return;
            quicksortClassic(items, 0, items.len - 1, context);
        }

        /// Sort using 3-way partitioning (Dijkstra's Dutch National Flag algorithm).
        ///
        /// Efficient for arrays with many duplicate elements.
        /// Partitions into three regions: <pivot, =pivot, >pivot
        ///
        /// Time: O(n log n) average, O(n) best (all equal) | Space: O(log n)
        pub fn sort3Way(items: []T, context: Context) void {
            if (items.len < 2) return;
            quicksort3Way(items, 0, items.len - 1, context);
        }

        /// Sort using dual-pivot partitioning (similar to Java's Arrays.sort).
        ///
        /// Uses two pivots to partition into three segments.
        /// Often faster than single-pivot quicksort in practice.
        ///
        /// Time: O(n log n) average | Space: O(log n)
        pub fn sortDualPivot(items: []T, context: Context) void {
            if (items.len < 2) return;
            quicksortDualPivot(items, 0, items.len - 1, context);
        }

        // --- Classic QuickSort with Hoare partition ---

        fn quicksortClassic(items: []T, left: usize, right: usize, context: Context) void {
            if (left >= right) return;

            // Use insertion sort for small subarrays
            if (right - left + 1 <= INSERTION_SORT_THRESHOLD) {
                insertionSort(items, left, right + 1, context);
                return;
            }

            const pivot_idx = partitionLomuto(items, left, right, context);
            // Lomuto partition puts pivot in final position
            // Recurse on [left..pivot_idx-1] and [pivot_idx+1..right]
            if (pivot_idx > left) {
                quicksortClassic(items, left, pivot_idx - 1, context);
            }
            if (pivot_idx < right) {
                quicksortClassic(items, pivot_idx + 1, right, context);
            }
        }

        fn partitionLomuto(items: []T, left: usize, right: usize, context: Context) usize {
            // Median-of-three pivot selection for better performance
            const mid = left + (right - left) / 2;
            medianOfThree(items, left, mid, right, context);
            // Move pivot to end
            std.mem.swap(T, &items[mid], &items[right]);
            const pivot = items[right];

            var i: usize = left;
            var j: usize = left;

            while (j < right) : (j += 1) {
                if (compareFn(context, items[j], pivot) != .gt) {
                    std.mem.swap(T, &items[i], &items[j]);
                    i += 1;
                }
            }

            // Place pivot in its final position
            std.mem.swap(T, &items[i], &items[right]);
            return i;
        }

        // --- 3-Way QuickSort (Dutch National Flag) ---

        fn quicksort3Way(items: []T, left: usize, right: usize, context: Context) void {
            if (right <= left) return;

            if (right - left + 1 <= INSERTION_SORT_THRESHOLD) {
                insertionSort(items, left, right + 1, context);
                return;
            }

            // Partition into [left..lt-1], [lt..gt], [gt+1..right]
            const result = partition3Way(items, left, right, context);
            const lt = result.lt;
            const gt = result.gt;

            if (lt > left) quicksort3Way(items, left, lt - 1, context);
            if (gt < right) quicksort3Way(items, gt + 1, right, context);
        }

        const Partition3WayResult = struct {
            lt: usize, // First element equal to pivot
            gt: usize, // Last element equal to pivot
        };

        fn partition3Way(items: []T, left: usize, right: usize, context: Context) Partition3WayResult {
            const pivot = items[left];
            var lt: usize = left; // items[left..lt-1] < pivot
            var i: usize = left + 1; // items[lt..i-1] == pivot
            var gt: usize = right; // items[gt+1..right] > pivot

            while (i <= gt) {
                const cmp = compareFn(context, items[i], pivot);
                if (cmp == .lt) {
                    std.mem.swap(T, &items[lt], &items[i]);
                    lt += 1;
                    i += 1;
                } else if (cmp == .gt) {
                    std.mem.swap(T, &items[i], &items[gt]);
                    gt -= 1;
                } else {
                    i += 1;
                }
            }

            return .{ .lt = lt, .gt = gt };
        }

        // --- Dual-Pivot QuickSort ---

        fn quicksortDualPivot(items: []T, left: usize, right: usize, context: Context) void {
            if (right <= left) return;

            if (right - left + 1 <= INSERTION_SORT_THRESHOLD) {
                insertionSort(items, left, right + 1, context);
                return;
            }

            const result = partitionDualPivot(items, left, right, context);
            const lp = result.lp;
            const rp = result.rp;

            if (lp > left) quicksortDualPivot(items, left, lp - 1, context);
            if (rp > lp + 1 and rp <= right) quicksortDualPivot(items, lp + 1, rp - 1, context);
            if (rp < right) quicksortDualPivot(items, rp + 1, right, context);
        }

        const PartitionDualPivotResult = struct {
            lp: usize, // Left pivot position
            rp: usize, // Right pivot position
        };

        fn partitionDualPivot(items: []T, left: usize, right: usize, context: Context) PartitionDualPivotResult {
            // Ensure items[left] <= items[right]
            if (compareFn(context, items[left], items[right]) == .gt) {
                std.mem.swap(T, &items[left], &items[right]);
            }

            const pivot1 = items[left];
            const pivot2 = items[right];

            var i: usize = left + 1;
            var lt: usize = left + 1; // items[left+1..lt-1] < pivot1
            var gt: usize = right - 1; // items[gt+1..right-1] > pivot2

            while (i <= gt) {
                const cmp1 = compareFn(context, items[i], pivot1);
                if (cmp1 == .lt) {
                    std.mem.swap(T, &items[lt], &items[i]);
                    lt += 1;
                    i += 1;
                } else {
                    const cmp2 = compareFn(context, items[i], pivot2);
                    if (cmp2 == .gt) {
                        std.mem.swap(T, &items[i], &items[gt]);
                        gt -= 1;
                    } else {
                        i += 1;
                    }
                }
            }

            // Move pivots to their final positions
            lt -= 1;
            gt += 1;
            std.mem.swap(T, &items[left], &items[lt]);
            std.mem.swap(T, &items[right], &items[gt]);

            return .{ .lp = lt, .rp = gt };
        }

        // --- Helper functions ---

        fn medianOfThree(items: []T, a: usize, b: usize, c: usize, context: Context) void {
            if (compareFn(context, items[b], items[a]) == .lt) std.mem.swap(T, &items[a], &items[b]);
            if (compareFn(context, items[c], items[b]) == .lt) std.mem.swap(T, &items[b], &items[c]);
            if (compareFn(context, items[b], items[a]) == .lt) std.mem.swap(T, &items[a], &items[b]);
        }

        fn insertionSort(items: []T, start: usize, end: usize, context: Context) void {
            var i: usize = start + 1;
            while (i < end) : (i += 1) {
                const key = items[i];
                var j: usize = i;
                while (j > start and compareFn(context, items[j - 1], key) == .gt) : (j -= 1) {
                    items[j] = items[j - 1];
                }
                items[j] = key;
            }
        }
    };
}

/// Convenience function for sorting with default comparison.
pub fn sort(comptime T: type, items: []T) void {
    const S = struct {
        fn compare(_: void, a: T, b: T) Order {
            if (a < b) return .lt;
            if (a > b) return .gt;
            return .eq;
        }
    };
    QuickSort(T, void, S.compare).sort(items, {});
}

/// Convenience function for 3-way sorting with default comparison.
pub fn sort3Way(comptime T: type, items: []T) void {
    const S = struct {
        fn compare(_: void, a: T, b: T) Order {
            if (a < b) return .lt;
            if (a > b) return .gt;
            return .eq;
        }
    };
    QuickSort(T, void, S.compare).sort3Way(items, {});
}

/// Convenience function for dual-pivot sorting with default comparison.
pub fn sortDualPivot(comptime T: type, items: []T) void {
    const S = struct {
        fn compare(_: void, a: T, b: T) Order {
            if (a < b) return .lt;
            if (a > b) return .gt;
            return .eq;
        }
    };
    QuickSort(T, void, S.compare).sortDualPivot(items, {});
}

// --- Tests ---

fn compareU32(_: void, a: u32, b: u32) Order {
    if (a < b) return .lt;
    if (a > b) return .gt;
    return .eq;
}

test "QuickSort - basic sorting" {
    var arr = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    QuickSort(u32, void, compareU32).sort(&arr, {});
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 1, 2, 3, 3, 4, 5, 5, 6, 9 }, &arr);
}

test "QuickSort - already sorted" {
    var arr = [_]u32{ 1, 2, 3, 4, 5 };
    QuickSort(u32, void, compareU32).sort(&arr, {});
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3, 4, 5 }, &arr);
}

test "QuickSort - reverse sorted" {
    var arr = [_]u32{ 5, 4, 3, 2, 1 };
    QuickSort(u32, void, compareU32).sort(&arr, {});
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3, 4, 5 }, &arr);
}

test "QuickSort - single element" {
    var arr = [_]u32{42};
    QuickSort(u32, void, compareU32).sort(&arr, {});
    try testing.expectEqualSlices(u32, &[_]u32{42}, &arr);
}

test "QuickSort - two elements" {
    var arr = [_]u32{ 2, 1 };
    QuickSort(u32, void, compareU32).sort(&arr, {});
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2 }, &arr);
}

test "QuickSort - all equal" {
    var arr = [_]u32{ 5, 5, 5, 5, 5 };
    QuickSort(u32, void, compareU32).sort(&arr, {});
    try testing.expectEqualSlices(u32, &[_]u32{ 5, 5, 5, 5, 5 }, &arr);
}

test "QuickSort - large array" {
    var arr: [1000]u32 = undefined;
    var prng = Random.DefaultPrng.init(42);
    const random = prng.random();
    for (&arr) |*val| {
        val.* = random.int(u32);
    }

    QuickSort(u32, void, compareU32).sort(&arr, {});

    // Verify sorted
    var i: usize = 1;
    while (i < arr.len) : (i += 1) {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}

test "QuickSort - with duplicates" {
    var arr = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9 };
    QuickSort(u32, void, compareU32).sort(&arr, {});
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 7, 8, 9, 9, 9 }, &arr);
}

test "QuickSort - 3-way basic" {
    var arr = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    QuickSort(u32, void, compareU32).sort3Way(&arr, {});
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 1, 2, 3, 3, 4, 5, 5, 6, 9 }, &arr);
}

test "QuickSort - 3-way many duplicates" {
    var arr = [_]u32{ 3, 1, 3, 1, 3, 2, 2, 3, 1, 2 };
    QuickSort(u32, void, compareU32).sort3Way(&arr, {});
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 1, 1, 2, 2, 2, 3, 3, 3, 3 }, &arr);
}

test "QuickSort - 3-way all equal" {
    var arr = [_]u32{ 7, 7, 7, 7, 7, 7, 7 };
    QuickSort(u32, void, compareU32).sort3Way(&arr, {});
    try testing.expectEqualSlices(u32, &[_]u32{ 7, 7, 7, 7, 7, 7, 7 }, &arr);
}

test "QuickSort - dual-pivot basic" {
    var arr = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    QuickSort(u32, void, compareU32).sortDualPivot(&arr, {});
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 1, 2, 3, 3, 4, 5, 5, 6, 9 }, &arr);
}

test "QuickSort - dual-pivot large array" {
    var arr: [500]u32 = undefined;
    var prng = Random.DefaultPrng.init(12345);
    const random = prng.random();
    for (&arr) |*val| {
        val.* = random.int(u32) % 100;
    }

    QuickSort(u32, void, compareU32).sortDualPivot(&arr, {});

    // Verify sorted
    var i: usize = 1;
    while (i < arr.len) : (i += 1) {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}

test "QuickSort - convenience functions" {
    var arr1 = [_]u32{ 5, 2, 8, 1, 9 };
    sort(u32, &arr1);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 5, 8, 9 }, &arr1);

    var arr2 = [_]u32{ 5, 2, 8, 1, 9 };
    sort3Way(u32, &arr2);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 5, 8, 9 }, &arr2);

    var arr3 = [_]u32{ 5, 2, 8, 1, 9 };
    sortDualPivot(u32, &arr3);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 5, 8, 9 }, &arr3);
}

test "QuickSort - f64 support" {
    var arr = [_]f64{ 3.14, 1.41, 2.71, 0.58, 1.61 };
    const S = struct {
        fn compare(_: void, a: f64, b: f64) Order {
            if (a < b) return .lt;
            if (a > b) return .gt;
            return .eq;
        }
    };
    QuickSort(f64, void, S.compare).sort(&arr, {});

    // Verify sorted
    var i: usize = 1;
    while (i < arr.len) : (i += 1) {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}

test "QuickSort - stress test" {
    var arr: [10000]u32 = undefined;
    var prng = Random.DefaultPrng.init(54321);
    const random = prng.random();
    for (&arr) |*val| {
        val.* = random.int(u32);
    }

    QuickSort(u32, void, compareU32).sort(&arr, {});

    // Verify sorted
    var i: usize = 1;
    while (i < arr.len) : (i += 1) {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}

test "QuickSort - 3-way stress test" {
    var arr: [10000]u32 = undefined;
    var prng = Random.DefaultPrng.init(99999);
    const random = prng.random();
    for (&arr) |*val| {
        val.* = random.int(u32) % 50; // Many duplicates
    }

    QuickSort(u32, void, compareU32).sort3Way(&arr, {});

    // Verify sorted
    var i: usize = 1;
    while (i < arr.len) : (i += 1) {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}

test "QuickSort - dual-pivot stress test" {
    var arr: [10000]u32 = undefined;
    var prng = Random.DefaultPrng.init(11111);
    const random = prng.random();
    for (&arr) |*val| {
        val.* = random.int(u32);
    }

    QuickSort(u32, void, compareU32).sortDualPivot(&arr, {});

    // Verify sorted
    var i: usize = 1;
    while (i < arr.len) : (i += 1) {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}

test "QuickSort - custom context" {
    const S = struct {
        reverse: bool,

        fn compare(self: @This(), a: u32, b: u32) Order {
            const cmp = if (a < b) Order.lt else if (a > b) Order.gt else Order.eq;
            if (self.reverse) {
                return switch (cmp) {
                    .lt => .gt,
                    .gt => .lt,
                    .eq => .eq,
                };
            }
            return cmp;
        }
    };

    var arr = [_]u32{ 3, 1, 4, 1, 5 };
    const ctx = S{ .reverse = true };
    QuickSort(u32, S, S.compare).sort(&arr, ctx);
    try testing.expectEqualSlices(u32, &[_]u32{ 5, 4, 3, 1, 1 }, &arr);
}
