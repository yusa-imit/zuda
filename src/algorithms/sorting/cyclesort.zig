const std = @import("std");
const testing = std.testing;
const Order = std.math.Order;

/// Cycle Sort - In-place sorting with minimal write operations
///
/// Cycle Sort is a comparison-based sorting algorithm that is particularly efficient
/// when write operations are expensive (e.g., flash memory, EEPROM). It minimizes
/// the number of memory writes by ensuring each value is written to its final position
/// at most once.
///
/// Key characteristics:
/// - In-place: O(1) extra space
/// - Unstable: Does not preserve relative order of equal elements
/// - Optimal writes: At most n-1 writes to the array (theoretical minimum for comparison sorts)
/// - Non-adaptive: Always performs O(n²) comparisons regardless of input order
///
/// Algorithm:
/// For each position, find the correct final position of the element by counting
/// how many elements are smaller. If not already in place, swap into correct position
/// and continue the cycle until returning to the starting position.
///
/// Time Complexity:
/// - Best/Average/Worst: O(n²) comparisons
/// - Writes: O(n) - at most n-1 writes regardless of input
///
/// Space Complexity: O(1) - in-place sorting
///
/// Use Cases:
/// - Systems with expensive write operations (flash memory, EEPROM, SSD)
/// - Embedded systems where write endurance matters
/// - Educational purposes (understanding optimal write algorithms)
/// - Situations where minimizing writes is more critical than minimizing comparisons

/// Sorts an array in-place using cycle sort with a custom comparison function
///
/// Time: O(n²) comparisons, O(n) writes
/// Space: O(1) in-place
///
/// Example:
/// ```zig
/// var arr = [_]i32{ 5, 2, 9, 1, 7 };
/// cycleSort(i32, &arr, struct {
///     fn lessThan(a: i32, b: i32) bool {
///         return a < b;
///     }
/// }.lessThan);
/// // arr is now [1, 2, 5, 7, 9]
/// ```
pub fn cycleSort(comptime T: type, arr: []T, comptime lessThan: fn (T, T) bool) void {
    if (arr.len <= 1) return;

    const n = arr.len;

    // Process each cycle starting at position cycle_start
    var cycle_start: usize = 0;
    while (cycle_start < n - 1) : (cycle_start += 1) {
        var item = arr[cycle_start];

        // Find position where we need to put the item
        var pos = cycle_start;
        var i = cycle_start + 1;
        while (i < n) : (i += 1) {
            if (lessThan(arr[i], item)) {
                pos += 1;
            }
        }

        // If item is already in correct position, continue to next cycle
        if (pos == cycle_start) {
            continue;
        }

        // Skip duplicates - place item after duplicates
        // Use comparison: if not less and not greater, they are equal
        while (!lessThan(item, arr[pos]) and !lessThan(arr[pos], item)) {
            pos += 1;
        }

        // Put item to its right position (first write)
        const temp = arr[pos];
        arr[pos] = item;
        item = temp;

        // Rotate rest of the cycle
        while (pos != cycle_start) {
            // Find position where we need to put the item
            pos = cycle_start;
            i = cycle_start + 1;
            while (i < n) : (i += 1) {
                if (lessThan(arr[i], item)) {
                    pos += 1;
                }
            }

            // Skip duplicates
            while (!lessThan(item, arr[pos]) and !lessThan(arr[pos], item)) {
                pos += 1;
            }

            // Put item to its right position
            const temp2 = arr[pos];
            arr[pos] = item;
            item = temp2;
        }
    }
}

/// Sorts an array in ascending order using cycle sort
///
/// Time: O(n²) comparisons, O(n) writes
/// Space: O(1) in-place
///
/// Example:
/// ```zig
/// var arr = [_]i32{ 5, 2, 9, 1, 7 };
/// cycleSortAsc(i32, &arr);
/// // arr is now [1, 2, 5, 7, 9]
/// ```
pub fn cycleSortAsc(comptime T: type, arr: []T) void {
    cycleSort(T, arr, struct {
        fn lessThan(a: T, b: T) bool {
            return a < b;
        }
    }.lessThan);
}

/// Sorts an array in descending order using cycle sort
///
/// Time: O(n²) comparisons, O(n) writes
/// Space: O(1) in-place
///
/// Example:
/// ```zig
/// var arr = [_]i32{ 5, 2, 9, 1, 7 };
/// cycleSortDesc(i32, &arr);
/// // arr is now [9, 7, 5, 2, 1]
/// ```
pub fn cycleSortDesc(comptime T: type, arr: []T) void {
    cycleSort(T, arr, struct {
        fn lessThan(a: T, b: T) bool {
            return a > b;
        }
    }.lessThan);
}

/// Sorts using a custom comparison function that returns Order
///
/// Time: O(n²) comparisons, O(n) writes
/// Space: O(1) in-place
///
/// Example:
/// ```zig
/// const Point = struct { x: i32, y: i32 };
/// var points = [_]Point{ .{ .x = 3, .y = 4 }, .{ .x = 1, .y = 2 } };
/// sortBy(Point, &points, struct {
///     fn compare(a: Point, b: Point) Order {
///         return std.math.order(a.x, b.x);
///     }
/// }.compare);
/// ```
pub fn sortBy(comptime T: type, arr: []T, comptime compareFn: fn (T, T) Order) void {
    cycleSort(T, arr, struct {
        fn lessThan(a: T, b: T) bool {
            return compareFn(a, b) == .lt;
        }
    }.lessThan);
}

/// Counts the number of write operations performed during cycle sort
///
/// This is useful for analyzing the efficiency of cycle sort and verifying
/// that it achieves the theoretical minimum number of writes.
///
/// Returns: Number of writes to the array (at most n-1)
///
/// Example:
/// ```zig
/// var arr = [_]i32{ 5, 2, 9, 1, 7 };
/// const writes = countWrites(i32, &arr, asc);
/// // writes will be at most arr.len - 1
/// ```
pub fn countWrites(comptime T: type, arr: []T, comptime lessThan: fn (T, T) bool) usize {
    if (arr.len <= 1) return 0;

    const n = arr.len;
    var write_count: usize = 0;

    // Process each cycle starting at position cycle_start
    var cycle_start: usize = 0;
    while (cycle_start < n - 1) : (cycle_start += 1) {
        var item = arr[cycle_start];

        // Find position where we need to put the item
        var pos = cycle_start;
        var i = cycle_start + 1;
        while (i < n) : (i += 1) {
            if (lessThan(arr[i], item)) {
                pos += 1;
            }
        }

        // If item is already in correct position, continue to next cycle
        if (pos == cycle_start) {
            continue;
        }

        // Skip duplicates
        while (!lessThan(item, arr[pos]) and !lessThan(arr[pos], item)) {
            pos += 1;
        }

        // Put item to its right position (first write)
        const temp = arr[pos];
        arr[pos] = item;
        item = temp;
        write_count += 1;

        // Rotate rest of the cycle
        while (pos != cycle_start) {
            // Find position where we need to put the item
            pos = cycle_start;
            i = cycle_start + 1;
            while (i < n) : (i += 1) {
                if (lessThan(arr[i], item)) {
                    pos += 1;
                }
            }

            // Skip duplicates
            while (!lessThan(item, arr[pos]) and !lessThan(arr[pos], item)) {
                pos += 1;
            }

            // Put item to its right position
            const temp2 = arr[pos];
            arr[pos] = item;
            item = temp2;
            write_count += 1;
        }
    }

    return write_count;
}

// Helper comparison functions
fn asc(comptime T: type) fn (T, T) bool {
    return struct {
        fn lessThan(a: T, b: T) bool {
            return a < b;
        }
    }.lessThan;
}

fn desc(comptime T: type) fn (T, T) bool {
    return struct {
        fn lessThan(a: T, b: T) bool {
            return a > b;
        }
    }.lessThan;
}

// ===== Tests =====

test "cycle sort - basic ascending" {
    var arr = [_]i32{ 5, 2, 9, 1, 7, 3 };
    cycleSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 5, 7, 9 }, &arr);
}

test "cycle sort - basic descending" {
    var arr = [_]i32{ 5, 2, 9, 1, 7, 3 };
    cycleSortDesc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 9, 7, 5, 3, 2, 1 }, &arr);
}

test "cycle sort - with duplicates" {
    var arr = [_]i32{ 5, 2, 9, 2, 7, 5, 1 };
    cycleSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 2, 5, 5, 7, 9 }, &arr);
}

test "cycle sort - edge case: empty" {
    var arr = [_]i32{};
    cycleSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{}, &arr);
}

test "cycle sort - edge case: single element" {
    var arr = [_]i32{42};
    cycleSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}

test "cycle sort - edge case: two elements" {
    var arr = [_]i32{ 2, 1 };
    cycleSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2 }, &arr);
}

test "cycle sort - already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    cycleSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "cycle sort - reverse sorted" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    cycleSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "cycle sort - all equal elements" {
    var arr = [_]i32{ 7, 7, 7, 7, 7 };
    cycleSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 7, 7, 7, 7, 7 }, &arr);
}

test "cycle sort - custom comparison with sortBy" {
    const Point = struct {
        x: i32,
        y: i32,
    };
    var points = [_]Point{
        .{ .x = 3, .y = 4 },
        .{ .x = 1, .y = 2 },
        .{ .x = 5, .y = 1 },
    };
    sortBy(Point, &points, struct {
        fn compare(a: Point, b: Point) Order {
            return std.math.order(a.x, b.x);
        }
    }.compare);
    try testing.expectEqual(@as(i32, 1), points[0].x);
    try testing.expectEqual(@as(i32, 3), points[1].x);
    try testing.expectEqual(@as(i32, 5), points[2].x);
}

test "cycle sort - write count for sorted array" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    const writes = countWrites(i32, &arr, asc(i32));
    // Already sorted - should have 0 writes
    try testing.expectEqual(@as(usize, 0), writes);
}

test "cycle sort - write count for reverse sorted" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    const writes = countWrites(i32, &arr, asc(i32));
    // Reverse sorted - will need swaps, but at most n-1
    try testing.expect(writes <= arr.len - 1);
    // Actual number of writes for this specific reverse sorted array
    try testing.expectEqual(@as(usize, 4), writes);
}

test "cycle sort - write count maximum" {
    var arr = [_]i32{ 5, 2, 9, 1, 7, 3 };
    const writes = countWrites(i32, &arr, asc(i32));
    // At most n-1 writes
    try testing.expect(writes <= arr.len - 1);
}

test "cycle sort - optimal writes property" {
    // Cycle sort should minimize writes
    var arr = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const writes = countWrites(i32, &arr, asc(i32));
    // Each element should be written at most once to its final position
    try testing.expect(writes <= arr.len);
}

test "cycle sort - large array with duplicates" {
    var arr: [50]i32 = undefined;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    for (&arr) |*val| {
        val.* = @mod(random.int(i32), 20); // Values 0-19 with many duplicates
    }

    cycleSortAsc(i32, &arr);

    // Verify sorted
    var i: usize = 1;
    while (i < arr.len) : (i += 1) {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}

test "cycle sort - floating point f32" {
    var arr = [_]f32{ 5.5, 2.2, 9.9, 1.1, 7.7 };
    cycleSortAsc(f32, &arr);
    try testing.expectEqualSlices(f32, &[_]f32{ 1.1, 2.2, 5.5, 7.7, 9.9 }, &arr);
}

test "cycle sort - floating point f64" {
    var arr = [_]f64{ 5.5, 2.2, 9.9, 1.1, 7.7 };
    cycleSortAsc(f64, &arr);
    try testing.expectEqualSlices(f64, &[_]f64{ 1.1, 2.2, 5.5, 7.7, 9.9 }, &arr);
}

test "cycle sort - negative numbers" {
    var arr = [_]i32{ -5, 2, -9, 1, -7, 3 };
    cycleSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ -9, -7, -5, 1, 2, 3 }, &arr);
}

test "cycle sort - unsigned integers" {
    var arr = [_]u8{ 200, 150, 100, 250, 50 };
    cycleSortAsc(u8, &arr);
    try testing.expectEqualSlices(u8, &[_]u8{ 50, 100, 150, 200, 250 }, &arr);
}

test "cycle sort - stress test with random data" {
    var arr: [100]i32 = undefined;
    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();
    for (&arr) |*val| {
        val.* = random.int(i32);
    }

    cycleSortAsc(i32, &arr);

    // Verify sorted
    var i: usize = 1;
    while (i < arr.len) : (i += 1) {
        try testing.expect(arr[i - 1] <= arr[i]);
    }

    // Count writes to verify optimality
    var arr2: [100]i32 = undefined;
    for (&arr2, 0..) |*val, idx| {
        val.* = arr[idx];
    }
    // Shuffle arr2 to create unsorted version
    for (&arr2) |*val| {
        val.* = random.int(i32);
    }
    const writes = countWrites(i32, &arr2, asc(i32));
    try testing.expect(writes <= arr2.len);
}
