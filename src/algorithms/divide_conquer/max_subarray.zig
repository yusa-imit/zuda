/// Maximum Subarray problem using divide-and-conquer and Kadane's algorithm.
///
/// Find the contiguous subarray with the largest sum. This is a classic problem
/// with both divide-and-conquer O(n log n) and dynamic programming O(n) solutions.
///
/// ## Algorithms
///
/// 1. **Kadane's Algorithm** (DP): O(n) time, O(1) space - optimal for most cases
/// 2. **Divide-and-Conquer**: O(n log n) time, O(log n) space - demonstrates D&C paradigm
///
/// ## Use Cases
///
/// - Stock market analysis (maximum profit period)
/// - Data analysis (best performing time window)
/// - Signal processing (strongest signal period)
///
const std = @import("std");
const testing = std.testing;

/// Result of maximum subarray computation.
pub fn Result(comptime T: type) type {
    return struct {
        start: usize,
        end: usize, // exclusive
        sum: T,
    };
}

/// Find maximum subarray using Kadane's algorithm (optimal).
///
/// Time: O(n)
/// Space: O(1)
///
/// ## Example
///
/// ```zig
/// const arr = [_]i32{ -2, 1, -3, 4, -1, 2, 1, -5, 4 };
/// const result = kadane(i32, &arr); // [4, -1, 2, 1] = 6
/// ```
pub fn kadane(comptime T: type, arr: []const T) error{EmptyArray}!Result(T) {
    if (arr.len == 0) return error.EmptyArray;

    var max_sum = arr[0];
    var current_sum = arr[0];
    var start: usize = 0;
    var end: usize = 1;
    var temp_start: usize = 0;

    for (arr[1..], 1..) |val, i| {
        if (current_sum < 0) {
            current_sum = val;
            temp_start = i;
        } else {
            current_sum += val;
        }

        if (current_sum > max_sum) {
            max_sum = current_sum;
            start = temp_start;
            end = i + 1;
        }
    }

    return Result(T){
        .start = start,
        .end = end,
        .sum = max_sum,
    };
}

/// Find maximum subarray using divide-and-conquer.
///
/// Divides array in half, finds:
/// 1. Max subarray in left half
/// 2. Max subarray in right half
/// 3. Max subarray crossing midpoint
///
/// Time: O(n log n)
/// Space: O(log n) for recursion stack
pub fn divideConquer(comptime T: type, arr: []const T) error{EmptyArray}!Result(T) {
    if (arr.len == 0) return error.EmptyArray;
    return divideConquerImpl(T, arr, 0, arr.len);
}

fn divideConquerImpl(comptime T: type, arr: []const T, left: usize, right: usize) Result(T) {
    // Base case: single element
    if (right - left == 1) {
        return Result(T){
            .start = left,
            .end = right,
            .sum = arr[left],
        };
    }

    const mid = left + (right - left) / 2;

    // Recursively find max in left and right halves
    const left_result = divideConquerImpl(T, arr, left, mid);
    const right_result = divideConquerImpl(T, arr, mid, right);

    // Find max crossing midpoint
    const cross_result = maxCrossingSubarray(T, arr, left, mid, right);

    // Return the maximum of three
    if (left_result.sum >= right_result.sum and left_result.sum >= cross_result.sum) {
        return left_result;
    } else if (right_result.sum >= left_result.sum and right_result.sum >= cross_result.sum) {
        return right_result;
    } else {
        return cross_result;
    }
}

fn maxCrossingSubarray(comptime T: type, arr: []const T, left: usize, mid: usize, right: usize) Result(T) {
    // Find max sum in left half (ending at mid-1)
    var left_sum = arr[mid - 1];
    var sum = arr[mid - 1];
    var max_left = mid - 1;

    if (mid > left + 1) {
        var i = mid - 1;
        while (i > left) : (i -= 1) {
            sum += arr[i - 1];
            if (sum > left_sum) {
                left_sum = sum;
                max_left = i - 1;
            }
        }
    }

    // Find max sum in right half (starting at mid)
    var right_sum = arr[mid];
    sum = arr[mid];
    var max_right = mid + 1;

    for (arr[mid + 1 .. right], mid + 1..) |val, i| {
        sum += val;
        if (sum > right_sum) {
            right_sum = sum;
            max_right = i + 1;
        }
    }

    return Result(T){
        .start = max_left,
        .end = max_right,
        .sum = left_sum + right_sum,
    };
}

/// Find maximum sum of any subarray (sum only, no indices).
///
/// Time: O(n)
/// Space: O(1)
pub fn maxSum(comptime T: type, arr: []const T) error{EmptyArray}!T {
    return (try kadane(T, arr)).sum;
}

/// Find maximum product subarray (variant of maximum subarray).
///
/// Handles negative numbers which can become large when multiplied.
/// Tracks both max and min products at each position.
///
/// Time: O(n)
/// Space: O(1)
pub fn maxProduct(comptime T: type, arr: []const T) error{EmptyArray}!Result(T) {
    if (arr.len == 0) return error.EmptyArray;

    var max_prod = arr[0];
    var current_max = arr[0];
    var current_min = arr[0];
    var start: usize = 0;
    var end: usize = 1;
    var temp_start: usize = 0;

    for (arr[1..], 1..) |val, i| {
        if (val < 0) {
            // Swap max and min when multiplying by negative
            const temp = current_max;
            current_max = current_min;
            current_min = temp;
        }

        // Max/min product ending at current position
        current_max = @max(val, current_max * val);
        current_min = @min(val, current_min * val);

        if (current_max == val) {
            temp_start = i;
        }

        if (current_max > max_prod) {
            max_prod = current_max;
            start = temp_start;
            end = i + 1;
        }
    }

    return Result(T){
        .start = start,
        .end = end,
        .sum = max_prod, // Using 'sum' field for product
    };
}

// ============================================================================
// Tests
// ============================================================================

test "kadane - basic" {
    const arr = [_]i32{ -2, 1, -3, 4, -1, 2, 1, -5, 4 };
    const result = try kadane(i32, &arr);

    try testing.expectEqual(@as(i32, 6), result.sum); // [4, -1, 2, 1]
    try testing.expectEqual(@as(usize, 3), result.start);
    try testing.expectEqual(@as(usize, 7), result.end);
}

test "kadane - all negative" {
    const arr = [_]i32{ -5, -2, -8, -1, -4 };
    const result = try kadane(i32, &arr);

    try testing.expectEqual(@as(i32, -1), result.sum); // Least negative
}

test "kadane - all positive" {
    const arr = [_]i32{ 1, 2, 3, 4, 5 };
    const result = try kadane(i32, &arr);

    try testing.expectEqual(@as(i32, 15), result.sum); // Entire array
    try testing.expectEqual(@as(usize, 0), result.start);
    try testing.expectEqual(@as(usize, 5), result.end);
}

test "kadane - single element" {
    const arr = [_]i32{42};
    const result = try kadane(i32, &arr);

    try testing.expectEqual(@as(i32, 42), result.sum);
    try testing.expectEqual(@as(usize, 0), result.start);
    try testing.expectEqual(@as(usize, 1), result.end);
}

test "kadane - empty array" {
    const arr = [_]i32{};
    try testing.expectError(error.EmptyArray, kadane(i32, &arr));
}

test "kadane - with zeros" {
    const arr = [_]i32{ -1, 0, -2, 0, -3 };
    const result = try kadane(i32, &arr);

    try testing.expectEqual(@as(i32, 0), result.sum);
}

test "divideConquer - basic" {
    const arr = [_]i32{ -2, 1, -3, 4, -1, 2, 1, -5, 4 };
    const result = try divideConquer(i32, &arr);

    try testing.expectEqual(@as(i32, 6), result.sum);
    try testing.expectEqual(@as(usize, 3), result.start);
    try testing.expectEqual(@as(usize, 7), result.end);
}

test "divideConquer - all negative" {
    const arr = [_]i32{ -5, -2, -8, -1, -4 };
    const result = try divideConquer(i32, &arr);

    try testing.expectEqual(@as(i32, -1), result.sum);
}

test "divideConquer - single element" {
    const arr = [_]i32{42};
    const result = try divideConquer(i32, &arr);

    try testing.expectEqual(@as(i32, 42), result.sum);
}

test "divideConquer vs kadane - consistency" {
    const arr = [_]i32{ 5, -3, 5, -1, 3, -2, 8, -7, 4 };
    const kadane_result = try kadane(i32, &arr);
    const dc_result = try divideConquer(i32, &arr);

    // Both should find same maximum sum
    try testing.expectEqual(kadane_result.sum, dc_result.sum);
}

test "maxSum - convenience function" {
    const arr = [_]i32{ -2, 1, -3, 4, -1, 2, 1, -5, 4 };
    try testing.expectEqual(@as(i32, 6), try maxSum(i32, &arr));
}

test "maxProduct - basic" {
    const arr = [_]i32{ 2, 3, -2, 4 };
    const result = try maxProduct(i32, &arr);

    try testing.expectEqual(@as(i32, 6), result.sum); // [2, 3]
    try testing.expectEqual(@as(usize, 0), result.start);
    try testing.expectEqual(@as(usize, 2), result.end);
}

test "maxProduct - with negatives" {
    const arr = [_]i32{ -2, 3, -4 };
    const result = try maxProduct(i32, &arr);

    try testing.expectEqual(@as(i32, 24), result.sum); // [-2, 3, -4] = 24
}

test "maxProduct - single negative" {
    const arr = [_]i32{ 2, -5, 3 };
    const result = try maxProduct(i32, &arr);

    try testing.expectEqual(@as(i32, 3), result.sum); // Just [3]
}

test "maxProduct - all positive" {
    const arr = [_]i32{ 2, 3, 4 };
    const result = try maxProduct(i32, &arr);

    try testing.expectEqual(@as(i32, 24), result.sum); // [2, 3, 4]
}

test "maxProduct - with zero" {
    const arr = [_]i32{ 2, 3, 0, 4, 5 };
    const result = try maxProduct(i32, &arr);

    try testing.expectEqual(@as(i32, 20), result.sum); // [4, 5]
}

test "kadane - large array" {
    const allocator = testing.allocator;
    const n = 1000;
    var arr = try allocator.alloc(i32, n);
    defer allocator.free(arr);

    // Alternating pattern: 10, -1, 10, -1, ...
    for (0..n) |i| {
        arr[i] = if (i % 2 == 0) 10 else -1;
    }

    const result = try kadane(i32, arr);
    try testing.expect(result.sum > 0);
}

test "maxProduct - floats" {
    const arr = [_]f64{ 2.5, 3.0, -2.0, 4.0 };
    const result = try maxProduct(f64, &arr);

    try testing.expectApproxEqAbs(@as(f64, 7.5), result.sum, 0.0001);
}
