const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Shell Sort — gap-based insertion sort with diminishing increments
///
/// Shell sort improves insertion sort by comparing elements separated by a gap,
/// gradually reducing the gap until it becomes 1 (standard insertion sort).
/// The gap sequence significantly affects performance.
///
/// Algorithm:
/// 1. Start with a large gap (e.g., n/2)
/// 2. Perform gapped insertion sort (compare elements gap positions apart)
/// 3. Reduce gap (e.g., gap = gap/2) and repeat
/// 4. Continue until gap = 1 (final pass is standard insertion sort)
///
/// Time complexity:
/// - Best: O(n log n) with optimal gap sequence (Sedgewick, Tokuda)
/// - Average: O(n^1.5) with Shell's original sequence, O(n^(4/3)) with Knuth
/// - Worst: O(n^2) with Shell's sequence, O(n^(3/2)) with Knuth
/// Space complexity: O(1) — in-place sorting
///
/// Stability: Unstable (relative order of equal elements not preserved)
///
/// Gap sequences (from best to worst performance):
/// - Sedgewick (1986): 1, 5, 19, 41, 109, ... → O(n^(4/3))
/// - Tokuda (1992): 1, 4, 9, 20, 46, ... → O(n^(4/3))
/// - Knuth (1973): 1, 4, 13, 40, 121, ... (3k+1) → O(n^(3/2))
/// - Shell (1959): n/2, n/4, ..., 1 → O(n^2)
///
/// Use cases:
/// - Medium-sized datasets (1K-100K elements)
/// - Systems with limited memory (in-place, no allocation)
/// - When simple implementation is preferred over optimal O(n log n)
/// - Embedded systems, real-time constraints
/// - Educational purposes (understanding gap-based sorting)

/// Gap sequence types
pub const GapSequence = enum {
    /// Shell's original sequence: n/2, n/4, ..., 1
    /// Simple but worst performance: O(n^2)
    shell,

    /// Knuth's sequence: (3^k - 1) / 2 = 1, 4, 13, 40, 121, 364, ...
    /// Good balance: O(n^(3/2)) worst case
    knuth,

    /// Sedgewick (1986): Best theoretical performance O(n^(4/3))
    /// Sequence: 1, 5, 19, 41, 109, 209, 505, ...
    sedgewick,

    /// Tokuda (1992): Similar to Sedgewick, empirically good
    /// Sequence: ceil((9/4)^k / 5) = 1, 4, 9, 20, 46, 103, ...
    tokuda,
};

/// Shell sort with configurable gap sequence
///
/// Time: O(n^1.5) to O(n^2) depending on gap sequence
/// Space: O(1) — in-place
pub fn shellSort(comptime T: type, items: []T, sequence: GapSequence) void {
    if (items.len <= 1) return;

    switch (sequence) {
        .shell => shellSortShell(T, items),
        .knuth => shellSortKnuth(T, items),
        .sedgewick => shellSortSedgewick(T, items),
        .tokuda => shellSortTokuda(T, items),
    }
}

/// Shell sort with custom comparator and gap sequence
///
/// Time: O(n^1.5) to O(n^2) depending on gap sequence
/// Space: O(1) — in-place
pub fn shellSortBy(
    comptime T: type,
    items: []T,
    sequence: GapSequence,
    comptime lessThanFn: fn (T, T) bool,
) void {
    if (items.len <= 1) return;

    switch (sequence) {
        .shell => shellSortShellBy(T, items, lessThanFn),
        .knuth => shellSortKnuthBy(T, items, lessThanFn),
        .sedgewick => shellSortSedgewickBy(T, items, lessThanFn),
        .tokuda => shellSortTokudaBy(T, items, lessThanFn),
    }
}

/// Shell's original sequence: n/2, n/4, ..., 1
/// Time: O(n^2) worst case
/// Space: O(1)
fn shellSortShell(comptime T: type, items: []T) void {
    const n = items.len;
    var gap: usize = n / 2;

    while (gap > 0) : (gap /= 2) {
        // Gapped insertion sort
        var i: usize = gap;
        while (i < n) : (i += 1) {
            const temp = items[i];
            var j = i;

            // Shift elements gap positions apart
            while (j >= gap and items[j - gap] > temp) {
                items[j] = items[j - gap];
                j -= gap;
            }
            items[j] = temp;
        }
    }
}

fn shellSortShellBy(
    comptime T: type,
    items: []T,
    comptime lessThanFn: fn (T, T) bool,
) void {
    const n = items.len;
    var gap: usize = n / 2;

    while (gap > 0) : (gap /= 2) {
        var i: usize = gap;
        while (i < n) : (i += 1) {
            const temp = items[i];
            var j = i;

            while (j >= gap and lessThanFn(temp, items[j - gap])) {
                items[j] = items[j - gap];
                j -= gap;
            }
            items[j] = temp;
        }
    }
}

/// Knuth's sequence: 1, 4, 13, 40, 121, ... (3^k - 1) / 2
/// Time: O(n^(3/2)) worst case
/// Space: O(1)
fn shellSortKnuth(comptime T: type, items: []T) void {
    const n = items.len;

    // Find starting gap: largest value in sequence <= n/3
    var gap: usize = 1;
    while (gap < n / 3) {
        gap = gap * 3 + 1; // 1, 4, 13, 40, 121, ...
    }

    while (gap > 0) {
        // Gapped insertion sort
        var i: usize = gap;
        while (i < n) : (i += 1) {
            const temp = items[i];
            var j = i;

            while (j >= gap and items[j - gap] > temp) {
                items[j] = items[j - gap];
                j -= gap;
            }
            items[j] = temp;
        }

        gap /= 3;
    }
}

fn shellSortKnuthBy(
    comptime T: type,
    items: []T,
    comptime lessThanFn: fn (T, T) bool,
) void {
    const n = items.len;
    var gap: usize = 1;
    while (gap < n / 3) gap = gap * 3 + 1;

    while (gap > 0) {
        var i: usize = gap;
        while (i < n) : (i += 1) {
            const temp = items[i];
            var j = i;

            while (j >= gap and lessThanFn(temp, items[j - gap])) {
                items[j] = items[j - gap];
                j -= gap;
            }
            items[j] = temp;
        }
        gap /= 3;
    }
}

/// Sedgewick's 1986 sequence: 1, 5, 19, 41, 109, 209, ...
/// Formula: 9*4^k - 9*2^k + 1 for k even, 4^(k+2) - 3*2^(k+2) + 1 for k odd
/// Time: O(n^(4/3)) worst case
/// Space: O(1)
fn shellSortSedgewick(comptime T: type, items: []T) void {
    const n = items.len;

    // Generate Sedgewick sequence up to n
    var gaps: [32]usize = undefined;
    var gap_count: usize = 0;
    var k: usize = 0;

    while (true) : (k += 1) {
        // Calculate Sedgewick sequence with overflow protection
        const gap = blk: {
            if (k % 2 == 0) {
                const k_half = k / 2;
                if (k_half > 10) break :blk n; // Prevent overflow for large k
                const pow4 = std.math.pow(usize, 4, k_half);
                const pow2 = std.math.pow(usize, 2, k_half);
                if (pow4 < 9 or pow2 < 9) break :blk n;
                break :blk 9 * pow4 - 9 * pow2 + 1;
            } else {
                const k_half = (k + 1) / 2;
                const k_plus = (k + 3) / 2;
                if (k_half > 10 or k_plus > 15) break :blk n; // Prevent overflow
                const pow4 = std.math.pow(usize, 4, k_half);
                const pow2 = std.math.pow(usize, 2, k_plus);
                if (pow2 > pow4 / 3) break :blk n; // Check for underflow
                break :blk pow4 - 3 * pow2 + 1;
            }
        };

        if (gap >= n) break;
        gaps[gap_count] = gap;
        gap_count += 1;
        if (gap_count >= gaps.len) break;
    }

    // Sort using gaps in descending order
    if (gap_count == 0) {
        gaps[0] = 1;
        gap_count = 1;
    }

    var idx: usize = gap_count;
    while (idx > 0) {
        idx -= 1;
        const gap = gaps[idx];

        var i: usize = gap;
        while (i < n) : (i += 1) {
            const temp = items[i];
            var j = i;

            while (j >= gap and items[j - gap] > temp) {
                items[j] = items[j - gap];
                j -= gap;
            }
            items[j] = temp;
        }
    }
}

fn shellSortSedgewickBy(
    comptime T: type,
    items: []T,
    comptime lessThanFn: fn (T, T) bool,
) void {
    const n = items.len;
    var gaps: [32]usize = undefined;
    var gap_count: usize = 0;
    var k: usize = 0;

    while (true) : (k += 1) {
        // Calculate Sedgewick sequence with overflow protection
        const gap = blk: {
            if (k % 2 == 0) {
                const k_half = k / 2;
                if (k_half > 10) break :blk n; // Prevent overflow for large k
                const pow4 = std.math.pow(usize, 4, k_half);
                const pow2 = std.math.pow(usize, 2, k_half);
                if (pow4 < 9 or pow2 < 9) break :blk n;
                break :blk 9 * pow4 - 9 * pow2 + 1;
            } else {
                const k_half = (k + 1) / 2;
                const k_plus = (k + 3) / 2;
                if (k_half > 10 or k_plus > 15) break :blk n; // Prevent overflow
                const pow4 = std.math.pow(usize, 4, k_half);
                const pow2 = std.math.pow(usize, 2, k_plus);
                if (pow2 > pow4 / 3) break :blk n; // Check for underflow
                break :blk pow4 - 3 * pow2 + 1;
            }
        };

        if (gap >= n) break;
        gaps[gap_count] = gap;
        gap_count += 1;
        if (gap_count >= gaps.len) break;
    }

    if (gap_count == 0) {
        gaps[0] = 1;
        gap_count = 1;
    }

    var idx: usize = gap_count;
    while (idx > 0) {
        idx -= 1;
        const gap = gaps[idx];

        var i: usize = gap;
        while (i < n) : (i += 1) {
            const temp = items[i];
            var j = i;

            while (j >= gap and lessThanFn(temp, items[j - gap])) {
                items[j] = items[j - gap];
                j -= gap;
            }
            items[j] = temp;
        }
    }
}

/// Tokuda's 1992 sequence: 1, 4, 9, 20, 46, 103, ...
/// Formula: ceil((9/4)^k / 5)
/// Time: O(n^(4/3)) empirically
/// Space: O(1)
fn shellSortTokuda(comptime T: type, items: []T) void {
    const n = items.len;

    // Generate Tokuda sequence up to n
    var gaps: [32]usize = undefined;
    var gap_count: usize = 0;
    var gap: usize = 1;

    while (gap < n) {
        gaps[gap_count] = gap;
        gap_count += 1;
        if (gap_count >= gaps.len) break;

        // Next gap: ceil((9/4)^k / 5)
        // Approximate with: gap = (gap * 9 + 3) / 4
        gap = (gap * 9 + 4) / 4;
    }

    if (gap_count == 0) {
        gaps[0] = 1;
        gap_count = 1;
    }

    // Sort using gaps in descending order
    var idx: usize = gap_count;
    while (idx > 0) {
        idx -= 1;
        const gap_val = gaps[idx];

        var i: usize = gap_val;
        while (i < n) : (i += 1) {
            const temp = items[i];
            var j = i;

            while (j >= gap_val and items[j - gap_val] > temp) {
                items[j] = items[j - gap_val];
                j -= gap_val;
            }
            items[j] = temp;
        }
    }
}

fn shellSortTokudaBy(
    comptime T: type,
    items: []T,
    comptime lessThanFn: fn (T, T) bool,
) void {
    const n = items.len;
    var gaps: [32]usize = undefined;
    var gap_count: usize = 0;
    var gap: usize = 1;

    while (gap < n) {
        gaps[gap_count] = gap;
        gap_count += 1;
        if (gap_count >= gaps.len) break;
        gap = (gap * 9 + 4) / 4;
    }

    if (gap_count == 0) {
        gaps[0] = 1;
        gap_count = 1;
    }

    var idx: usize = gap_count;
    while (idx > 0) {
        idx -= 1;
        const gap_val = gaps[idx];

        var i: usize = gap_val;
        while (i < n) : (i += 1) {
            const temp = items[i];
            var j = i;

            while (j >= gap_val and lessThanFn(temp, items[j - gap_val])) {
                items[j] = items[j - gap_val];
                j -= gap_val;
            }
            items[j] = temp;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "shellsort: basic sorting with Shell sequence" {
    var arr = [_]i32{ 5, 2, 8, 1, 9, 3 };
    shellSort(i32, &arr, .shell);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 5, 8, 9 }, &arr);
}

test "shellsort: basic sorting with Knuth sequence" {
    var arr = [_]i32{ 5, 2, 8, 1, 9, 3 };
    shellSort(i32, &arr, .knuth);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 5, 8, 9 }, &arr);
}

test "shellsort: basic sorting with Sedgewick sequence" {
    var arr = [_]i32{ 5, 2, 8, 1, 9, 3 };
    shellSort(i32, &arr, .sedgewick);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 5, 8, 9 }, &arr);
}

test "shellsort: basic sorting with Tokuda sequence" {
    var arr = [_]i32{ 5, 2, 8, 1, 9, 3 };
    shellSort(i32, &arr, .tokuda);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 5, 8, 9 }, &arr);
}

test "shellsort: empty array" {
    var arr = [_]i32{};
    shellSort(i32, &arr, .knuth);
    try testing.expectEqualSlices(i32, &[_]i32{}, &arr);
}

test "shellsort: single element" {
    var arr = [_]i32{42};
    shellSort(i32, &arr, .knuth);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}

test "shellsort: already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    shellSort(i32, &arr, .knuth);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "shellsort: reverse sorted" {
    var arr = [_]i32{ 9, 7, 5, 3, 1 };
    shellSort(i32, &arr, .knuth);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 3, 5, 7, 9 }, &arr);
}

test "shellsort: duplicates" {
    var arr = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    shellSort(i32, &arr, .knuth);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 3, 3, 4, 5, 5, 6, 9 }, &arr);
}

test "shellsort: all same values" {
    var arr = [_]i32{ 7, 7, 7, 7, 7 };
    shellSort(i32, &arr, .knuth);
    try testing.expectEqualSlices(i32, &[_]i32{ 7, 7, 7, 7, 7 }, &arr);
}

test "shellsort: negative numbers" {
    var arr = [_]i32{ -5, 2, -8, 1, -9, 3 };
    shellSort(i32, &arr, .knuth);
    try testing.expectEqualSlices(i32, &[_]i32{ -9, -8, -5, 1, 2, 3 }, &arr);
}

test "shellsort: large array with Knuth" {
    var arr: [100]i32 = undefined;
    for (&arr, 0..) |*val, i| {
        val.* = @intCast(100 - i);
    }

    shellSort(i32, &arr, .knuth);

    for (arr, 1..) |val, i| {
        try testing.expectEqual(@as(i32, @intCast(i)), val);
    }
}

test "shellsort: large array with Sedgewick" {
    var arr: [100]i32 = undefined;
    for (&arr, 0..) |*val, i| {
        val.* = @intCast(100 - i);
    }

    shellSort(i32, &arr, .sedgewick);

    for (arr, 1..) |val, i| {
        try testing.expectEqual(@as(i32, @intCast(i)), val);
    }
}

test "shellsort: large array with Tokuda" {
    var arr: [100]i32 = undefined;
    for (&arr, 0..) |*val, i| {
        val.* = @intCast(100 - i);
    }

    shellSort(i32, &arr, .tokuda);

    for (arr, 1..) |val, i| {
        try testing.expectEqual(@as(i32, @intCast(i)), val);
    }
}

test "shellsort: custom comparator descending" {
    const desc = struct {
        fn lessThan(a: i32, b: i32) bool {
            return a > b; // Reverse comparison for descending
        }
    }.lessThan;

    var arr = [_]i32{ 5, 2, 8, 1, 9, 3 };
    shellSortBy(i32, &arr, .knuth, desc);
    try testing.expectEqualSlices(i32, &[_]i32{ 9, 8, 5, 3, 2, 1 }, &arr);
}

test "shellsort: floats with Knuth" {
    var arr = [_]f64{ 3.14, 1.41, 2.71, 0.57, 4.66 };
    shellSort(f64, &arr, .knuth);

    try testing.expectApproxEqAbs(0.57, arr[0], 0.01);
    try testing.expectApproxEqAbs(1.41, arr[1], 0.01);
    try testing.expectApproxEqAbs(2.71, arr[2], 0.01);
    try testing.expectApproxEqAbs(3.14, arr[3], 0.01);
    try testing.expectApproxEqAbs(4.66, arr[4], 0.01);
}

test "shellsort: u8 characters" {
    var arr = [_]u8{ 'z', 'a', 'x', 'c', 'b' };
    shellSort(u8, &arr, .knuth);
    try testing.expectEqualSlices(u8, "abcxz", &arr);
}

test "shellsort: sequence comparison on same data" {
    // All sequences should produce the same sorted result
    const original = [_]i32{ 9, 3, 7, 1, 5, 8, 2, 6, 4 };

    var arr_shell = original;
    var arr_knuth = original;
    var arr_sedgewick = original;
    var arr_tokuda = original;

    shellSort(i32, &arr_shell, .shell);
    shellSort(i32, &arr_knuth, .knuth);
    shellSort(i32, &arr_sedgewick, .sedgewick);
    shellSort(i32, &arr_tokuda, .tokuda);

    const expected = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    try testing.expectEqualSlices(i32, &expected, &arr_shell);
    try testing.expectEqualSlices(i32, &expected, &arr_knuth);
    try testing.expectEqualSlices(i32, &expected, &arr_sedgewick);
    try testing.expectEqualSlices(i32, &expected, &arr_tokuda);
}

test "shellsort: stress test with random-like data" {
    var arr: [50]i32 = undefined;
    // Pseudo-random pattern
    for (&arr, 0..) |*val, i| {
        val.* = @intCast((i * 17 + 13) % 50);
    }

    shellSort(i32, &arr, .knuth);

    // Verify sorted
    for (arr[1..], 0..) |val, i| {
        try testing.expect(arr[i] <= val);
    }
}
