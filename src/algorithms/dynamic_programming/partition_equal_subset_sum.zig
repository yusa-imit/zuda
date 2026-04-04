//! Partition Equal Subset Sum - Dynamic Programming
//!
//! Determines if an array can be partitioned into two subsets with equal sum.
//! Classic DP problem, variation of Subset Sum and 0/1 Knapsack.
//!
//! **Problem**: Given a set of positive integers, partition into two subsets S1 and S2
//! such that sum(S1) = sum(S2).
//!
//! **Algorithm**: Bottom-up DP table
//! - If total sum is odd → impossible
//! - If total sum is even → reduce to subset sum problem with target = sum/2
//! - DP recurrence: dp[i][j] = can we make sum j using first i elements?
//!   * dp[i][j] = dp[i-1][j] (exclude nums[i-1]) OR dp[i-1][j-nums[i-1]] (include nums[i-1])
//!
//! **Time Complexity**: O(n × sum) where n = array length, sum = total sum
//! **Space Complexity**: O(sum) with space optimization (rolling array)
//!
//! **Use Cases**:
//! - Load balancing (distribute tasks into two equal groups)
//! - Resource allocation (split resources evenly)
//! - Team formation (divide players into balanced teams)
//! - Data partitioning (split dataset into equal-size chunks)

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Determines if array can be partitioned into two equal-sum subsets.
/// Uses dynamic programming with space optimization.
///
/// Time: O(n × sum) | Space: O(sum)
///
/// Returns true if partition exists, false otherwise.
///
/// Example:
/// ```zig
/// const nums = [_]i32{1, 5, 11, 5};
/// const result = try canPartition(i32, &nums);
/// // result = true (partitions: {1,5,5} and {11})
/// ```
pub fn canPartition(comptime T: type, nums: []const T) bool {
    if (nums.len == 0) return true;
    if (nums.len == 1) return false;

    // Calculate total sum
    var total: i64 = 0;
    for (nums) |num| {
        if (num < 0) return false; // negative numbers not allowed
        total += @as(i64, @intCast(num));
    }

    // If sum is odd, cannot partition equally
    if (total % 2 != 0) return false;

    const target = @divTrunc(total, 2);
    return subsetSumExists(T, nums, @intCast(target));
}

/// Finds actual partition if it exists.
/// Returns two ArrayLists containing the partition elements.
///
/// Time: O(n × sum) | Space: O(n × sum)
///
/// Returns error.NoPartition if no equal partition exists.
/// Caller owns returned ArrayLists and must deinit them.
///
/// Example:
/// ```zig
/// var result = try findPartition(i32, allocator, &nums);
/// defer result.subset1.deinit();
/// defer result.subset2.deinit();
/// ```
pub fn findPartition(comptime T: type, allocator: Allocator, nums: []const T) !struct {
    subset1: std.ArrayList(T),
    subset2: std.ArrayList(T),
} {
    if (nums.len == 0) {
        return .{
            .subset1 = std.ArrayList(T).init(allocator),
            .subset2 = std.ArrayList(T).init(allocator),
        };
    }
    if (nums.len == 1) return error.NoPartition;

    // Calculate total sum
    var total: i64 = 0;
    for (nums) |num| {
        if (num < 0) return error.NegativeNumber;
        total += @as(i64, @intCast(num));
    }

    if (total % 2 != 0) return error.NoPartition;

    const target = @divTrunc(total, 2);
    const path = try subsetSumPath(T, allocator, nums, @intCast(target));
    defer allocator.free(path);

    // Build subsets from path
    var subset1 = std.ArrayList(T).init(allocator);
    errdefer subset1.deinit();
    var subset2 = std.ArrayList(T).init(allocator);
    errdefer subset2.deinit();

    for (nums, 0..) |num, i| {
        if (path[i]) {
            try subset1.append(num);
        } else {
            try subset2.append(num);
        }
    }

    return .{ .subset1 = subset1, .subset2 = subset2 };
}

/// Optimized version: returns just boolean with O(sum) space.
/// Uses rolling DP array (current row only).
///
/// Time: O(n × sum) | Space: O(sum)
fn subsetSumExists(comptime T: type, nums: []const T, target: usize) bool {
    // DP array: dp[j] = can we make sum j?
    var dp = [_]bool{false} ** 100001; // max sum constraint
    if (target > 100000) return false;

    dp[0] = true; // base case: sum 0 is always possible (empty subset)

    // Process each number
    for (nums) |num| {
        const n = @as(usize, @intCast(num));
        if (n > target) continue;

        // Traverse backwards to avoid using same element twice
        var j = target;
        while (j >= n) : (j -= 1) {
            if (dp[j - n]) {
                dp[j] = true;
            }
            if (j == n) break;
        }

        if (dp[target]) return true; // early exit
    }

    return dp[target];
}

/// Finds subset that sums to target and returns boolean array indicating inclusion.
/// Time: O(n × sum) | Space: O(n × sum)
fn subsetSumPath(comptime T: type, allocator: Allocator, nums: []const T, target: usize) ![]bool {
    const n = nums.len;
    
    // DP table: dp[i][j] = can we make sum j using first i elements?
    const dp = try allocator.alloc([]bool, n + 1);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }

    for (0..n + 1) |i| {
        dp[i] = try allocator.alloc(bool, target + 1);
        @memset(dp[i], false);
        dp[i][0] = true; // base case
    }

    // Fill DP table
    for (1..n + 1) |i| {
        const num = @as(usize, @intCast(nums[i - 1]));
        for (0..target + 1) |j| {
            // Exclude current number
            dp[i][j] = dp[i - 1][j];
            
            // Include current number if possible
            if (j >= num and dp[i - 1][j - num]) {
                dp[i][j] = true;
            }
        }
    }

    if (!dp[n][target]) return error.NoPartition;

    // Backtrack to find actual subset
    const path = try allocator.alloc(bool, n);
    @memset(path, false);

    var j = target;
    var i = n;
    while (i > 0 and j > 0) {
        // If dp[i][j] came from including nums[i-1]
        const num = @as(usize, @intCast(nums[i - 1]));
        if (j >= num and dp[i - 1][j - num] and !dp[i - 1][j]) {
            path[i - 1] = true;
            j -= num;
        }
        i -= 1;
    }

    return path;
}

/// Variant: Can partition into K equal-sum subsets?
/// More general problem, NP-complete.
///
/// Time: O(k × 2^n) | Space: O(2^n)
///
/// Uses backtracking with memoization.
pub fn canPartitionKSubsets(comptime T: type, allocator: Allocator, nums: []const T, k: usize) !bool {
    if (k == 1) return true;
    if (nums.len < k) return false;

    var total: i64 = 0;
    for (nums) |num| {
        if (num < 0) return error.NegativeNumber;
        total += @as(i64, @intCast(num));
    }

    if (total % @as(i64, @intCast(k)) != 0) return false;
    const target = @divTrunc(total, @as(i64, @intCast(k)));

    // Sort descending for better pruning
    const sorted = try allocator.dupe(T, nums);
    defer allocator.free(sorted);
    std.mem.sort(T, sorted, {}, std.sort.desc(T));

    // Early exit if largest element exceeds target
    if (@as(i64, @intCast(sorted[0])) > target) return false;

    // Backtracking with visited bitmask
    const visited = try allocator.alloc(bool, nums.len);
    defer allocator.free(visited);
    @memset(visited, false);

    return canPartitionKSubsetsHelper(T, sorted, visited, 0, k, 0, @intCast(target));
}

fn canPartitionKSubsetsHelper(
    comptime T: type,
    nums: []const T,
    visited: []bool,
    start_idx: usize,
    k: usize,
    current_sum: i64,
    target: i64,
) bool {
    if (k == 1) return true; // Last subset is automatically valid
    if (current_sum == target) {
        // Found one subset, look for next
        return canPartitionKSubsetsHelper(T, nums, visited, 0, k - 1, 0, target);
    }

    for (start_idx..nums.len) |i| {
        if (visited[i]) continue;
        const num = @as(i64, @intCast(nums[i]));
        if (current_sum + num > target) continue;

        visited[i] = true;
        if (canPartitionKSubsetsHelper(T, nums, visited, i + 1, k, current_sum + num, target)) {
            return true;
        }
        visited[i] = false;

        // Pruning: if adding first element fails, no point trying others
        if (current_sum == 0) break;
    }

    return false;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const expect = testing.expect;
const expectEqual = testing.expectEqual;

test "Partition Equal Subset Sum - basic examples" {
    // Example 1: [1,5,11,5] -> {1,5,5} and {11}
    {
        const nums = [_]i32{ 1, 5, 11, 5 };
        try expect(canPartition(i32, &nums));
    }

    // Example 2: [1,2,3,5] -> cannot partition (sum=11, odd)
    {
        const nums = [_]i32{ 1, 2, 3, 5 };
        try expect(!canPartition(i32, &nums));
    }

    // Example 3: [1,2,3,4] -> {1,4} and {2,3}
    {
        const nums = [_]i32{ 1, 2, 3, 4 };
        try expect(canPartition(i32, &nums));
    }
}

test "Partition Equal Subset Sum - edge cases" {
    // Empty array
    {
        const nums = [_]i32{};
        try expect(canPartition(i32, &nums));
    }

    // Single element
    {
        const nums = [_]i32{1};
        try expect(!canPartition(i32, &nums));
    }

    // Two equal elements
    {
        const nums = [_]i32{ 5, 5 };
        try expect(canPartition(i32, &nums));
    }

    // Two unequal elements
    {
        const nums = [_]i32{ 3, 7 };
        try expect(!canPartition(i32, &nums));
    }
}

test "Partition Equal Subset Sum - odd sum" {
    // Odd total sum -> impossible
    const nums = [_]i32{ 1, 2, 3, 4, 5 };
    try expect(!canPartition(i32, &nums));
}

test "Partition Equal Subset Sum - large values" {
    // Large values but valid partition
    {
        const nums = [_]i32{ 100, 100, 100, 100 };
        try expect(canPartition(i32, &nums));
    }

    // Large unbalanced
    {
        const nums = [_]i32{ 1, 1, 1, 1000 };
        try expect(!canPartition(i32, &nums));
    }
}

test "Partition Equal Subset Sum - findPartition" {
    const allocator = testing.allocator;

    // Example: [1,5,11,5]
    const nums = [_]i32{ 1, 5, 11, 5 };
    var result = try findPartition(i32, allocator, &nums);
    defer result.subset1.deinit();
    defer result.subset2.deinit();

    // Calculate sums
    var sum1: i32 = 0;
    for (result.subset1.items) |x| sum1 += x;
    var sum2: i32 = 0;
    for (result.subset2.items) |x| sum2 += x;

    try expectEqual(@as(i32, 11), sum1);
    try expectEqual(@as(i32, 11), sum2);
}

test "Partition Equal Subset Sum - findPartition impossible" {
    const allocator = testing.allocator;

    // Odd sum -> error
    const nums = [_]i32{ 1, 2, 3, 5 };
    try testing.expectError(error.NoPartition, findPartition(i32, allocator, &nums));
}

test "Partition Equal Subset Sum - findPartition empty" {
    const allocator = testing.allocator;

    const nums = [_]i32{};
    var result = try findPartition(i32, allocator, &nums);
    defer result.subset1.deinit();
    defer result.subset2.deinit();

    try expectEqual(@as(usize, 0), result.subset1.items.len);
    try expectEqual(@as(usize, 0), result.subset2.items.len);
}

test "Partition Equal Subset Sum - all equal elements" {
    // [5,5,5,5] -> {5,5} and {5,5}
    const nums = [_]i32{ 5, 5, 5, 5 };
    try expect(canPartition(i32, &nums));
}

test "Partition Equal Subset Sum - multiple solutions" {
    // [1,1,2,2] -> {1,2} and {1,2} or {2} and {2} (multiple valid partitions)
    const nums = [_]i32{ 1, 1, 2, 2 };
    try expect(canPartition(i32, &nums));
}

test "Partition Equal Subset Sum - large array" {
    const allocator = testing.allocator;
    const n = 50;
    const nums = try allocator.alloc(i32, n);
    defer allocator.free(nums);

    // Fill with values 1..50 (sum = 1275, target = 637.5 -> odd, impossible)
    for (0..n) |i| {
        nums[i] = @intCast(i + 1);
    }

    try expect(!canPartition(i32, nums));
}

test "Partition Equal Subset Sum - negative numbers error" {
    const allocator = testing.allocator;

    const nums = [_]i32{ -1, 2, 3 };
    try expect(!canPartition(i32, &nums));
    try testing.expectError(error.NegativeNumber, findPartition(i32, allocator, &nums));
}

test "Partition Equal Subset Sum - K subsets basic" {
    const allocator = testing.allocator;

    // [4,3,2,3,5,2,1] can be partitioned into 4 subsets of sum 5 each
    {
        const nums = [_]i32{ 4, 3, 2, 3, 5, 2, 1 };
        const result = try canPartitionKSubsets(i32, allocator, &nums, 4);
        try expect(result);
    }

    // [1,2,3,4] can be partitioned into 2 subsets of sum 5 each
    {
        const nums = [_]i32{ 1, 2, 3, 4 };
        const result = try canPartitionKSubsets(i32, allocator, &nums, 2);
        try expect(result);
    }
}

test "Partition Equal Subset Sum - K subsets impossible" {
    const allocator = testing.allocator;

    // [1,1,1,1,1,1] cannot be partitioned into 4 equal subsets (sum=6, not divisible)
    {
        const nums = [_]i32{ 1, 1, 1, 1, 1, 1 };
        const result = try canPartitionKSubsets(i32, allocator, &nums, 4);
        try expect(!result);
    }

    // K=1 always true
    {
        const nums = [_]i32{ 1, 2, 3 };
        const result = try canPartitionKSubsets(i32, allocator, &nums, 1);
        try expect(result);
    }
}

test "Partition Equal Subset Sum - f64 support" {
    // Note: For floating point, we'd typically scale to integers
    // This test uses integer-representable floats
    const nums = [_]f64{ 1.0, 5.0, 11.0, 5.0 };
    try expect(canPartition(f64, &nums));
}

test "Partition Equal Subset Sum - memory safety" {
    const allocator = testing.allocator;

    const nums = [_]i32{ 1, 2, 3, 4 };
    var result = try findPartition(i32, allocator, &nums);
    defer result.subset1.deinit();
    defer result.subset2.deinit();

    // Verify no leaks (testing.allocator will catch this)
    try expect(result.subset1.items.len > 0);
    try expect(result.subset2.items.len > 0);
}

test "Partition Equal Subset Sum - three elements" {
    // [1,2,3] -> sum=6, target=3 -> {3} and {1,2}
    const nums = [_]i32{ 1, 2, 3 };
    try expect(canPartition(i32, &nums));
}
