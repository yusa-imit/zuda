const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Maximum Sum Subarray (Kadane's Algorithm)
///
/// Finds the contiguous subarray with the maximum sum.
/// Classic dynamic programming problem with multiple variants.
///
/// Reference: Jay Kadane (1984), LeetCode #53 (Maximum Subarray)
///
/// Key Insight:
/// At each position, decide whether to:
/// - Extend current subarray (add current element)
/// - Start new subarray (current element alone)
/// Choice: max(current_sum + arr[i], arr[i])
///
/// Time Complexity:
/// - maxSum: O(n) single pass
/// - maxSumWithIndices: O(n) single pass with tracking
/// - maxSumTable: O(n) with explicit DP table
/// - maxCircularSum: O(n) two linear passes
/// - maxSumK: O(n) for fixed k subarrays
/// - countSumGreaterThan: O(n²) check all subarrays
///
/// Space Complexity:
/// - maxSum: O(1) two variables
/// - maxSumTable: O(n) DP table
/// - maxCircularSum: O(1) constant space
/// - maxSumK: O(n) DP table
///
/// Use Cases:
/// - Stock trading (maximize profit over time period)
/// - Signal processing (find peak signal interval)
/// - Resource allocation (maximize value in contiguous window)
/// - Array analytics (best contiguous segment)

/// Result containing maximum sum and subarray indices
pub fn SubarrayResult(comptime T: type) type {
    return struct {
        sum: T,
        start: usize,
        end: usize,

        pub fn length(self: @This()) usize {
            return self.end - self.start + 1;
        }
    };
}

/// Find maximum sum of any contiguous subarray
/// Time: O(n), Space: O(1)
///
/// Algorithm (Kadane's):
/// - Track current_sum (ending at position i)
/// - Track max_sum (best seen so far)
/// - At each step: current_sum = max(arr[i], current_sum + arr[i])
///
/// Example:
/// Input: [-2, 1, -3, 4, -1, 2, 1, -5, 4]
/// Output: 6 (subarray [4, -1, 2, 1])
pub fn maxSum(comptime T: type, arr: []const T) !T {
    if (arr.len == 0) return error.EmptyArray;

    var max_sum = arr[0];
    var current_sum = arr[0];

    for (arr[1..]) |val| {
        current_sum = @max(val, current_sum + val);
        max_sum = @max(max_sum, current_sum);
    }

    return max_sum;
}

/// Find maximum sum with start/end indices
/// Time: O(n), Space: O(1)
///
/// Tracks indices when extending vs starting new subarray
pub fn maxSumWithIndices(comptime T: type, arr: []const T) !SubarrayResult(T) {
    if (arr.len == 0) return error.EmptyArray;

    var result = SubarrayResult(T){
        .sum = arr[0],
        .start = 0,
        .end = 0,
    };

    var current_sum = arr[0];
    var current_start: usize = 0;

    for (arr[1..], 1..) |val, i| {
        // Decide: extend current or start new
        if (val > current_sum + val) {
            // Starting new subarray is better
            current_sum = val;
            current_start = i;
        } else {
            // Extending current subarray
            current_sum = current_sum + val;
        }

        // Update best result
        if (current_sum > result.sum) {
            result.sum = current_sum;
            result.start = current_start;
            result.end = i;
        }
    }

    return result;
}

/// Maximum sum with explicit DP table (for clarity/reconstruction)
/// Time: O(n), Space: O(n)
///
/// dp[i] = maximum sum of subarray ending at index i
/// dp[i] = max(arr[i], dp[i-1] + arr[i])
pub fn maxSumTable(comptime T: type, allocator: Allocator, arr: []const T) !struct { sum: T, table: []T } {
    if (arr.len == 0) return error.EmptyArray;

    const dp = try allocator.alloc(T, arr.len);
    errdefer allocator.free(dp);

    dp[0] = arr[0];
    var max_sum = dp[0];

    for (arr[1..], 1..) |val, i| {
        dp[i] = @max(val, dp[i - 1] + val);
        max_sum = @max(max_sum, dp[i]);
    }

    return .{ .sum = max_sum, .table = dp };
}

/// Maximum sum in circular array (wrap around allowed)
/// Time: O(n), Space: O(1)
///
/// Two cases:
/// 1. Maximum subarray does NOT wrap → standard Kadane
/// 2. Maximum subarray DOES wrap → total_sum - minimum_subarray
/// Return max of both cases
///
/// Example:
/// Input: [5, -3, 5]
/// Case 1: 5 + (-3) + 5 = 7 (no wrap)
/// Case 2: 5 + 5 = 10 (wrap around, exclude -3)
/// Output: 10
pub fn maxCircularSum(comptime T: type, arr: []const T) !T {
    if (arr.len == 0) return error.EmptyArray;
    if (arr.len == 1) return arr[0];

    // Case 1: Standard maximum subarray (no wrap)
    const max_kadane = try maxSum(T, arr);

    // Case 2: Maximum with wrap = total - minimum subarray
    var total: T = 0;
    for (arr) |val| {
        total += val;
    }

    // Find minimum subarray (negate values, find max, negate result)
    var min_sum = arr[0];
    var current_sum = arr[0];
    for (arr[1..]) |val| {
        current_sum = @min(val, current_sum + val);
        min_sum = @min(min_sum, current_sum);
    }

    // If all elements are negative, max_kadane will be the least negative
    // In this case, wrapping doesn't help (would give empty array)
    if (total == min_sum) {
        return max_kadane;
    }

    const max_wrap = total - min_sum;
    return @max(max_kadane, max_wrap);
}

/// Find maximum sum when array is partitioned into exactly k non-empty subarrays
/// Time: O(n²×k), Space: O(n×k)
///
/// This partitions the array into k contiguous parts (no gaps allowed).
/// dp[i][j] = maximum sum partitioning first i elements into exactly j subarrays
///
/// Note: This is a PARTITION problem (array divided into k parts with no gaps).
/// NOT a selection problem (picking k best subarrays with gaps allowed).
///
/// Example:
/// Input: [-2, 1, -3, 4, -1, 2, 1, -5, 4], k=1
/// Partition: entire array = sum = 1
pub fn maxSumKSubarrays(comptime T: type, allocator: Allocator, arr: []const T, k: usize) !T {
    if (arr.len == 0) return error.EmptyArray;
    if (k == 0 or k > arr.len) return error.InvalidK;

    const n = arr.len;

    // dp[i][j] = max sum using first i elements with exactly j subarrays
    const dp = try allocator.alloc([]T, n + 1);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }

    for (dp) |*row| {
        row.* = try allocator.alloc(T, k + 1);
        @memset(row.*, std.math.minInt(T) / 2); // Use min/2 to avoid overflow
    }

    // Base case: 0 elements, 0 subarrays = 0
    dp[0][0] = 0;

    // Compute prefix sums for range sum queries
    const prefix = try allocator.alloc(T, n + 1);
    defer allocator.free(prefix);
    prefix[0] = 0;
    for (1..n + 1) |i| {
        prefix[i] = prefix[i - 1] + arr[i - 1];
    }

    // Fill DP table
    for (1..n + 1) |i| {
        for (1..@min(i, k) + 1) |j| {
            // Try all possible ending positions for j-th subarray
            for (j - 1..i) |start| {
                // j-th subarray is arr[start..i]
                // Need exactly j-1 subarrays in first start elements
                if (dp[start][j - 1] == std.math.minInt(T) / 2) continue;

                const subarray_sum = prefix[i] - prefix[start];
                const candidate = dp[start][j - 1] + subarray_sum;
                dp[i][j] = @max(dp[i][j], candidate);
            }
        }
    }

    return dp[n][k];
}

/// Count number of subarrays with sum greater than threshold
/// Time: O(n²), Space: O(1)
pub fn countSumGreaterThan(comptime T: type, arr: []const T, threshold: T) usize {
    if (arr.len == 0) return 0;

    var count: usize = 0;
    for (0..arr.len) |i| {
        var sum: T = 0;
        for (i..arr.len) |j| {
            sum += arr[j];
            if (sum > threshold) count += 1;
        }
    }
    return count;
}

/// Find minimum sum of any contiguous subarray (dual problem)
/// Time: O(n), Space: O(1)
pub fn minSum(comptime T: type, arr: []const T) !T {
    if (arr.len == 0) return error.EmptyArray;

    var min_sum = arr[0];
    var current_sum = arr[0];

    for (arr[1..]) |val| {
        current_sum = @min(val, current_sum + val);
        min_sum = @min(min_sum, current_sum);
    }

    return min_sum;
}

// ============================================================================
// Tests
// ============================================================================

test "max sum - basic" {
    const arr = [_]i32{ -2, 1, -3, 4, -1, 2, 1, -5, 4 };
    const result = try maxSum(i32, &arr);
    try testing.expectEqual(@as(i32, 6), result); // [4, -1, 2, 1]
}

test "max sum - all positive" {
    const arr = [_]i32{ 1, 2, 3, 4, 5 };
    const result = try maxSum(i32, &arr);
    try testing.expectEqual(@as(i32, 15), result); // entire array
}

test "max sum - all negative" {
    const arr = [_]i32{ -5, -2, -8, -1, -4 };
    const result = try maxSum(i32, &arr);
    try testing.expectEqual(@as(i32, -1), result); // least negative element
}

test "max sum - single element" {
    const arr = [_]i32{42};
    const result = try maxSum(i32, &arr);
    try testing.expectEqual(@as(i32, 42), result);
}

test "max sum - mixed" {
    const arr = [_]i32{ 5, -3, 5 };
    const result = try maxSum(i32, &arr);
    try testing.expectEqual(@as(i32, 7), result); // entire array
}

test "max sum - empty array" {
    const arr = [_]i32{};
    try testing.expectError(error.EmptyArray, maxSum(i32, &arr));
}

test "max sum with indices - basic" {
    const arr = [_]i32{ -2, 1, -3, 4, -1, 2, 1, -5, 4 };
    const result = try maxSumWithIndices(i32, &arr);
    try testing.expectEqual(@as(i32, 6), result.sum);
    try testing.expectEqual(@as(usize, 3), result.start); // index 3 = value 4
    try testing.expectEqual(@as(usize, 6), result.end); // index 6 = value 1
    try testing.expectEqual(@as(usize, 4), result.length());
}

test "max sum with indices - all positive" {
    const arr = [_]i32{ 1, 2, 3, 4, 5 };
    const result = try maxSumWithIndices(i32, &arr);
    try testing.expectEqual(@as(i32, 15), result.sum);
    try testing.expectEqual(@as(usize, 0), result.start);
    try testing.expectEqual(@as(usize, 4), result.end);
}

test "max sum with indices - all negative" {
    const arr = [_]i32{ -5, -2, -8, -1, -4 };
    const result = try maxSumWithIndices(i32, &arr);
    try testing.expectEqual(@as(i32, -1), result.sum);
    try testing.expectEqual(@as(usize, 3), result.start);
    try testing.expectEqual(@as(usize, 3), result.end);
}

test "max sum with indices - at end" {
    const arr = [_]i32{ -1, -2, 5 };
    const result = try maxSumWithIndices(i32, &arr);
    try testing.expectEqual(@as(i32, 5), result.sum);
    try testing.expectEqual(@as(usize, 2), result.start);
    try testing.expectEqual(@as(usize, 2), result.end);
}

test "max sum table" {
    const arr = [_]i32{ -2, 1, -3, 4, -1, 2, 1, -5, 4 };
    const result = try maxSumTable(i32, testing.allocator, &arr);
    defer testing.allocator.free(result.table);

    try testing.expectEqual(@as(i32, 6), result.sum);

    // Verify DP table values
    try testing.expectEqual(@as(i32, -2), result.table[0]); // [-2]
    try testing.expectEqual(@as(i32, 1), result.table[1]); // [1]
    try testing.expectEqual(@as(i32, -2), result.table[2]); // [1,-3]
    try testing.expectEqual(@as(i32, 4), result.table[3]); // [4]
    try testing.expectEqual(@as(i32, 3), result.table[4]); // [4,-1]
    try testing.expectEqual(@as(i32, 5), result.table[5]); // [4,-1,2]
    try testing.expectEqual(@as(i32, 6), result.table[6]); // [4,-1,2,1]
}

test "max sum circular - wrap around better" {
    const arr = [_]i32{ 5, -3, 5 };
    const result = try maxCircularSum(i32, &arr);
    try testing.expectEqual(@as(i32, 10), result); // [5] + [5] wrapping
}

test "max sum circular - no wrap better" {
    const arr = [_]i32{ 1, 2, 3, 4, 5 };
    const result = try maxCircularSum(i32, &arr);
    try testing.expectEqual(@as(i32, 15), result); // entire array
}

test "max sum circular - all negative" {
    const arr = [_]i32{ -5, -2, -8, -1, -4 };
    const result = try maxCircularSum(i32, &arr);
    try testing.expectEqual(@as(i32, -1), result); // least negative
}

test "max sum circular - single element" {
    const arr = [_]i32{42};
    const result = try maxCircularSum(i32, &arr);
    try testing.expectEqual(@as(i32, 42), result);
}

test "max sum circular - larger wrap" {
    const arr = [_]i32{ 8, -1, -5, -9, 4, 6 };
    const result = try maxCircularSum(i32, &arr);
    try testing.expectEqual(@as(i32, 18), result); // [8] + [4,6] wrapping
}

test "max sum k subarrays - k=1" {
    const arr = [_]i32{ -2, 1, -3, 4, -1, 2, 1, -5, 4 };
    const result = try maxSumKSubarrays(i32, testing.allocator, &arr, 1);
    // k=1 means entire array as one partition = sum of all elements
    try testing.expectEqual(@as(i32, 1), result); // -2+1-3+4-1+2+1-5+4 = 1
}

test "max sum k subarrays - k=2" {
    const arr = [_]i32{ 1, 2, 3, 4, 5 };
    const result = try maxSumKSubarrays(i32, testing.allocator, &arr, 2);
    try testing.expectEqual(@as(i32, 15), result); // split anywhere, sum is same
}

test "max sum k subarrays - k=3" {
    const arr = [_]i32{ -1, 4, -2, 5, -3, 2 };
    const result = try maxSumKSubarrays(i32, testing.allocator, &arr, 3);
    // Partition into 3 parts: best is [-1,4], [-2,5], [-3,2]
    // = 3 + 3 + (-1) = 5
    try testing.expectEqual(@as(i32, 5), result);
}

test "max sum k subarrays - k equals array length" {
    const arr = [_]i32{ 1, -2, 3 };
    const result = try maxSumKSubarrays(i32, testing.allocator, &arr, 3);
    try testing.expectEqual(@as(i32, 2), result); // [1], [-2], [3] = 2
}

test "max sum k subarrays - invalid k" {
    const arr = [_]i32{ 1, 2, 3 };
    try testing.expectError(error.InvalidK, maxSumKSubarrays(i32, testing.allocator, &arr, 0));
    try testing.expectError(error.InvalidK, maxSumKSubarrays(i32, testing.allocator, &arr, 4));
}

test "count sum greater than - basic" {
    const arr = [_]i32{ 1, 2, 3, 4, 5 };
    const count = countSumGreaterThan(i32, &arr, 5);
    // Subarrays: [1,2,3]=6, [1,2,3,4]=10, [1,2,3,4,5]=15, [2,3,4]=9, [2,3,4,5]=14,
    //            [3,4]=7, [3,4,5]=12, [4,5]=9, [1,2,3,4,5]=15
    // Count subarrays with sum > 5: many
    try testing.expect(count > 0);
}

test "count sum greater than - none" {
    const arr = [_]i32{ 1, 1, 1 };
    const count = countSumGreaterThan(i32, &arr, 100);
    try testing.expectEqual(@as(usize, 0), count);
}

test "count sum greater than - threshold zero" {
    const arr = [_]i32{ -1, 2, -3, 4 };
    const count = countSumGreaterThan(i32, &arr, 0);
    try testing.expect(count > 0); // [2], [4], [2,-3,4], etc.
}

test "min sum - basic" {
    const arr = [_]i32{ 2, -1, 3, -4, 5, -1 };
    const result = try minSum(i32, &arr);
    try testing.expectEqual(@as(i32, -4), result); // [-4]
}

test "min sum - all positive" {
    const arr = [_]i32{ 1, 2, 3, 4, 5 };
    const result = try minSum(i32, &arr);
    try testing.expectEqual(@as(i32, 1), result); // [1]
}

test "min sum - all negative" {
    const arr = [_]i32{ -5, -2, -8, -1, -4 };
    const result = try minSum(i32, &arr);
    try testing.expectEqual(@as(i32, -20), result); // entire array
}

test "min sum - contiguous negative" {
    const arr = [_]i32{ 5, -3, -2, -1, 4 };
    const result = try minSum(i32, &arr);
    try testing.expectEqual(@as(i32, -6), result); // [-3,-2,-1]
}

test "type generic - f64" {
    const arr = [_]f64{ -2.5, 1.5, -3.0, 4.5, -1.0, 2.0, 1.0, -5.5, 4.0 };
    const result = try maxSum(f64, &arr);
    try testing.expectApproxEqAbs(@as(f64, 6.5), result, 1e-10); // [4.5, -1.0, 2.0, 1.0]
}

test "large array - stress test" {
    const allocator = testing.allocator;
    const arr = try allocator.alloc(i32, 1000);
    defer allocator.free(arr);

    // Pattern: 100 positive, 50 negative, repeat
    for (arr, 0..) |*val, i| {
        val.* = if (i % 150 < 100) @as(i32, @intCast(i % 10 + 1)) else -@as(i32, @intCast(i % 5 + 1));
    }

    const result = try maxSum(i32, arr);
    try testing.expect(result > 0); // Should find positive subarray
}

test "memory safety - allocator cycles" {
    const arr = [_]i32{ 1, -2, 3, -4, 5 };
    for (0..100) |_| {
        const result = try maxSumTable(i32, testing.allocator, &arr);
        testing.allocator.free(result.table);
    }
}
