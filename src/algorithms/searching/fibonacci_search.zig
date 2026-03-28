//! Fibonacci Search Algorithm
//!
//! Fibonacci search is a comparison-based technique that uses Fibonacci numbers
//! to search a sorted array. It's similar to binary search but divides the array
//! using Fibonacci numbers instead of halving.
//!
//! Advantages over binary search:
//! - No division operations (only addition/subtraction)
//! - Better for very large arrays on systems where division is expensive
//! - Can be adapted for unevenly sized data structures
//!
//! Time Complexity:
//! - Best case: O(1)
//! - Average case: O(log n)
//! - Worst case: O(log n)
//!
//! Space Complexity: O(1)
//!
//! Use cases:
//! - Systems where division is expensive (embedded systems, older hardware)
//! - When array size is close to a Fibonacci number
//! - As an alternative to binary search with different access patterns

const std = @import("std");
const math = std.math;

/// Fibonacci search for finding a target value in a sorted array.
/// Returns the index of the target if found, null otherwise.
///
/// Time: O(log n) | Space: O(1)
///
/// Example:
/// ```zig
/// const arr = [_]i32{ 1, 3, 5, 7, 9, 11, 13, 15 };
/// const idx = fibonacciSearch(i32, &arr, 7, {}, std.sort.asc(i32));
/// // idx == 3
/// ```
pub fn fibonacciSearch(
    comptime T: type,
    arr: []const T,
    target: T,
    context: anytype,
    comptime compareFn: fn (@TypeOf(context), T, T) std.math.Order,
) ?usize {
    const n = arr.len;
    if (n == 0) return null;

    // Initialize Fibonacci numbers
    var fib_m2: usize = 0; // (m-2)'th Fibonacci number
    var fib_m1: usize = 1; // (m-1)'th Fibonacci number
    var fib_m: usize = fib_m2 + fib_m1; // m'th Fibonacci number

    // Find smallest Fibonacci number >= n
    while (fib_m < n) {
        fib_m2 = fib_m1;
        fib_m1 = fib_m;
        fib_m = fib_m2 + fib_m1;
    }

    // Marks the eliminated range from front
    var offset: usize = 0;

    // While there are elements to inspect
    while (fib_m > 1) {
        // Check if fib_m2 is a valid index
        const i = @min(offset + fib_m2, n - 1);

        const order = compareFn(context, arr[i], target);

        if (order == .lt) {
            // Target is in right subarray
            fib_m = fib_m1;
            fib_m1 = fib_m2;
            fib_m2 = fib_m - fib_m1;
            offset = i;
        } else if (order == .gt) {
            // Target is in left subarray
            fib_m = fib_m2;
            fib_m1 = fib_m1 - fib_m2;
            fib_m2 = fib_m - fib_m1;
        } else {
            // Found target
            return i;
        }
    }

    // Check the last element
    if (fib_m1 == 1 and offset + 1 < n) {
        if (compareFn(context, arr[offset + 1], target) == .eq) {
            return offset + 1;
        }
    }

    return null;
}

/// Helper function to compute the k'th Fibonacci number.
/// Used internally for educational purposes and testing.
///
/// Time: O(k) | Space: O(1)
fn fibonacci(k: usize) usize {
    if (k == 0) return 0;
    if (k == 1) return 1;

    var a: usize = 0;
    var b: usize = 1;
    var i: usize = 2;

    while (i <= k) : (i += 1) {
        const temp = a + b;
        a = b;
        b = temp;
    }

    return b;
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "fibonacciSearch: basic functionality" {
    const arr = [_]i32{ 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21 };

    // Found cases
    try testing.expectEqual(@as(?usize, 0), fibonacciSearch(i32, &arr, 1, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, 3), fibonacciSearch(i32, &arr, 7, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, 5), fibonacciSearch(i32, &arr, 11, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, 10), fibonacciSearch(i32, &arr, 21, {}, std.sort.asc(i32)));

    // Not found cases
    try testing.expectEqual(@as(?usize, null), fibonacciSearch(i32, &arr, 0, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, null), fibonacciSearch(i32, &arr, 8, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, null), fibonacciSearch(i32, &arr, 22, {}, std.sort.asc(i32)));
}

test "fibonacciSearch: edge cases" {
    // Empty array
    const empty = [_]i32{};
    try testing.expectEqual(@as(?usize, null), fibonacciSearch(i32, &empty, 5, {}, std.sort.asc(i32)));

    // Single element - found
    const single = [_]i32{42};
    try testing.expectEqual(@as(?usize, 0), fibonacciSearch(i32, &single, 42, {}, std.sort.asc(i32)));

    // Single element - not found
    try testing.expectEqual(@as(?usize, null), fibonacciSearch(i32, &single, 10, {}, std.sort.asc(i32)));

    // Two elements
    const two = [_]i32{ 1, 2 };
    try testing.expectEqual(@as(?usize, 0), fibonacciSearch(i32, &two, 1, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, 1), fibonacciSearch(i32, &two, 2, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, null), fibonacciSearch(i32, &two, 3, {}, std.sort.asc(i32)));
}

test "fibonacciSearch: Fibonacci-sized arrays" {
    // Array of size 8 (Fibonacci number)
    const fib8 = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try testing.expectEqual(@as(?usize, 0), fibonacciSearch(i32, &fib8, 1, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, 4), fibonacciSearch(i32, &fib8, 5, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, 7), fibonacciSearch(i32, &fib8, 8, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, null), fibonacciSearch(i32, &fib8, 9, {}, std.sort.asc(i32)));

    // Array of size 13 (Fibonacci number)
    const fib13 = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
    try testing.expectEqual(@as(?usize, 0), fibonacciSearch(i32, &fib13, 1, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, 6), fibonacciSearch(i32, &fib13, 7, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, 12), fibonacciSearch(i32, &fib13, 13, {}, std.sort.asc(i32)));
}

test "fibonacciSearch: duplicates" {
    const arr = [_]i32{ 1, 2, 2, 2, 3, 4, 5 };

    // Should find one of the duplicates
    const idx = fibonacciSearch(i32, &arr, 2, {}, std.sort.asc(i32)).?;
    try testing.expect(idx >= 1 and idx <= 3);
    try testing.expectEqual(@as(i32, 2), arr[idx]);
}

test "fibonacciSearch: large array" {
    var arr: [10000]i32 = undefined;
    for (&arr, 0..) |*val, i| {
        val.* = @intCast(i * 2); // 0, 2, 4, 6, ..., 19998
    }

    // Test various positions
    try testing.expectEqual(@as(?usize, 0), fibonacciSearch(i32, &arr, 0, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, 50), fibonacciSearch(i32, &arr, 100, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, 5000), fibonacciSearch(i32, &arr, 10000, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, 9999), fibonacciSearch(i32, &arr, 19998, {}, std.sort.asc(i32)));

    // Not found (odd numbers)
    try testing.expectEqual(@as(?usize, null), fibonacciSearch(i32, &arr, 1, {}, std.sort.asc(i32)));
    try testing.expectEqual(@as(?usize, null), fibonacciSearch(i32, &arr, 101, {}, std.sort.asc(i32)));
}

test "fibonacciSearch: descending order" {
    const arr = [_]i32{ 20, 18, 15, 12, 10, 8, 5, 3, 1 };

    try testing.expectEqual(@as(?usize, 0), fibonacciSearch(i32, &arr, 20, {}, std.sort.desc(i32)));
    try testing.expectEqual(@as(?usize, 4), fibonacciSearch(i32, &arr, 10, {}, std.sort.desc(i32)));
    try testing.expectEqual(@as(?usize, 8), fibonacciSearch(i32, &arr, 1, {}, std.sort.desc(i32)));

    try testing.expectEqual(@as(?usize, null), fibonacciSearch(i32, &arr, 0, {}, std.sort.desc(i32)));
    try testing.expectEqual(@as(?usize, null), fibonacciSearch(i32, &arr, 21, {}, std.sort.desc(i32)));
}

test "fibonacciSearch: float type" {
    const arr = [_]f64{ -5.5, -2.3, 0.0, 1.1, 3.14, 7.5, 10.0 };

    try testing.expectEqual(@as(?usize, 2), fibonacciSearch(f64, &arr, 0.0, {}, std.sort.asc(f64)));
    try testing.expectEqual(@as(?usize, 4), fibonacciSearch(f64, &arr, 3.14, {}, std.sort.asc(f64)));
    try testing.expectEqual(@as(?usize, null), fibonacciSearch(f64, &arr, 2.0, {}, std.sort.asc(f64)));
}

test "fibonacciSearch: comparison with binary search behavior" {
    const arr = [_]i32{ 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29 };

    // Both should find the same elements
    for (arr) |val| {
        const result = fibonacciSearch(i32, &arr, val, {}, std.sort.asc(i32));
        try testing.expect(result != null);
        try testing.expectEqual(val, arr[result.?]);
    }

    // Both should fail to find the same non-existent elements
    const not_found = [_]i32{ 0, 2, 4, 6, 8, 10, 30 };
    for (not_found) |val| {
        try testing.expectEqual(@as(?usize, null), fibonacciSearch(i32, &arr, val, {}, std.sort.asc(i32)));
    }
}

test "fibonacci helper: correctness" {
    // First 15 Fibonacci numbers
    try testing.expectEqual(@as(usize, 0), fibonacci(0));
    try testing.expectEqual(@as(usize, 1), fibonacci(1));
    try testing.expectEqual(@as(usize, 1), fibonacci(2));
    try testing.expectEqual(@as(usize, 2), fibonacci(3));
    try testing.expectEqual(@as(usize, 3), fibonacci(4));
    try testing.expectEqual(@as(usize, 5), fibonacci(5));
    try testing.expectEqual(@as(usize, 8), fibonacci(6));
    try testing.expectEqual(@as(usize, 13), fibonacci(7));
    try testing.expectEqual(@as(usize, 21), fibonacci(8));
    try testing.expectEqual(@as(usize, 34), fibonacci(9));
    try testing.expectEqual(@as(usize, 55), fibonacci(10));
}

test "fibonacciSearch: stress test with various array sizes" {
    // Test with different array sizes around Fibonacci numbers
    const sizes = [_]usize{ 7, 8, 9, 12, 13, 14, 20, 21, 22 };

    for (sizes) |size| {
        const arr = try testing.allocator.alloc(i32, size);
        defer testing.allocator.free(arr);

        for (arr, 0..) |*val, i| {
            val.* = @intCast(i);
        }

        // Test finding each element
        for (arr, 0..) |val, expected_idx| {
            const result = fibonacciSearch(i32, arr, val, {}, std.sort.asc(i32));
            try testing.expectEqual(@as(?usize, expected_idx), result);
        }

        // Test not found
        try testing.expectEqual(@as(?usize, null), fibonacciSearch(i32, arr, -1, {}, std.sort.asc(i32)));
        try testing.expectEqual(@as(?usize, null), fibonacciSearch(i32, arr, @intCast(size), {}, std.sort.asc(i32)));
    }
}
