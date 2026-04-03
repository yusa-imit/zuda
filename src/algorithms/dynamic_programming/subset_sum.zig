const std = @import("std");
const Allocator = std.mem.Allocator;

/// Subset Sum Problem — Dynamic Programming Solutions
///
/// The subset sum problem asks: given a set of integers, is there a non-empty subset
/// whose sum equals a target value? This is a classic NP-complete problem that can be
/// solved efficiently using dynamic programming for small target values.
///
/// Applications:
/// - Resource allocation (scheduling with time/budget constraints)
/// - Partition problems (dividing items into equal groups)
/// - Cryptography (knapsack-based encryption schemes)
/// - Load balancing (distributing work across processors)
/// - Financial planning (meeting exact budget targets)
///
/// Reference: Cormen et al., "Introduction to Algorithms" (2009), Section 35.5

/// Check if a subset exists that sums to the target value.
///
/// Uses bottom-up dynamic programming with O(n*target) time and O(target) space.
/// The algorithm builds a boolean DP table where dp[j] indicates whether sum j is achievable.
///
/// **Algorithm**:
/// - Initialize: dp[0] = true (empty subset has sum 0)
/// - For each number x in the set:
///   - For each sum j from target down to x:
///     - If dp[j-x] is true, mark dp[j] as true (include x in subset)
/// - Result: dp[target] indicates if target sum is achievable
///
/// **Time**: O(n × target) where n = number of elements
/// **Space**: O(target) for DP array (space-optimized from O(n × target))
///
/// **Example**:
/// ```zig
/// const set = [_]i32{3, 34, 4, 12, 5, 2};
/// const result = try canPartition(i32, allocator, &set, 9);
/// // Returns true (subset: {4, 5} or {3, 4, 2})
/// ```
pub fn canPartition(comptime T: type, allocator: Allocator, set: []const T, target: T) !bool {
    if (set.len == 0) return false;
    if (target < 0) return false;
    if (target == 0) return true;

    const target_usize = @as(usize, @intCast(target));

    // DP array: dp[j] = true if sum j is achievable
    var dp = try allocator.alloc(bool, target_usize + 1);
    defer allocator.free(dp);

    @memset(dp, false);
    dp[0] = true; // Base case: empty subset has sum 0

    // Process each element
    for (set) |num| {
        if (num < 0 or num > target) continue; // Skip negative or too-large values

        const num_usize = @as(usize, @intCast(num));

        // Traverse from target down to num (reverse order to avoid using same element twice)
        var j = target_usize;
        while (j >= num_usize) : (j -= 1) {
            if (dp[j - num_usize]) {
                dp[j] = true;
            }
            if (j == 0) break; // Prevent underflow
        }
    }

    return dp[target_usize];
}

/// Find one subset that sums to the target value (if exists).
///
/// Returns an ArrayList containing the indices of elements in the subset.
/// The subset is reconstructed by backtracking through the DP table.
///
/// **Time**: O(n × target) for DP + O(n) for backtracking
/// **Space**: O(n × target) for DP table (needs full 2D table for reconstruction)
///
/// **Example**:
/// ```zig
/// const set = [_]i32{3, 34, 4, 12, 5, 2};
/// const subset = try findSubset(i32, allocator, &set, 9);
/// defer subset.deinit();
/// // Returns indices like [0, 2, 5] for subset {3, 4, 2}
/// ```
pub fn findSubset(comptime T: type, allocator: Allocator, set: []const T, target: T) !?std.ArrayList(usize) {
    if (set.len == 0) return null;
    if (target < 0) return null;
    if (target == 0) {
        return std.ArrayList(usize).init(allocator); // Empty subset
    }

    const n = set.len;
    const target_usize = @as(usize, @intCast(target));

    // DP table: dp[i][j] = true if subset of first i elements can sum to j
    var dp = try allocator.alloc([]bool, n + 1);
    defer {
        for (dp) |row| {
            allocator.free(row);
        }
        allocator.free(dp);
    }

    for (dp) |*row| {
        row.* = try allocator.alloc(bool, target_usize + 1);
        @memset(row.*, false);
    }

    // Base case: sum 0 is achievable with empty subset
    for (0..n + 1) |i| {
        dp[i][0] = true;
    }

    // Fill DP table
    for (1..n + 1) |i| {
        const num = set[i - 1];
        if (num < 0) continue; // Skip negative values

        for (0..target_usize + 1) |j| {
            // Option 1: Don't include current element
            dp[i][j] = dp[i - 1][j];

            // Option 2: Include current element (if it fits)
            if (!dp[i][j] and num <= @as(T, @intCast(j))) {
                const num_usize = @as(usize, @intCast(num));
                if (j >= num_usize) {
                    dp[i][j] = dp[i - 1][j - num_usize];
                }
            }
        }
    }

    // Check if target is achievable
    if (!dp[n][target_usize]) {
        return null; // No subset exists
    }

    // Backtrack to find the subset
    var result = std.ArrayList(usize).init(allocator);
    errdefer result.deinit();

    var i = n;
    var j = target_usize;

    while (i > 0 and j > 0) {
        // If dp[i][j] != dp[i-1][j], then set[i-1] is included
        if (dp[i][j] and !dp[i - 1][j]) {
            try result.append(i - 1); // Add index to result
            const num = set[i - 1];
            const num_usize = @as(usize, @intCast(num));
            j -= num_usize;
        }
        i -= 1;
    }

    return result;
}

/// Count the number of distinct subsets that sum to the target value.
///
/// This variant counts all possible ways to form the target sum (order-independent).
///
/// **Time**: O(n × target)
/// **Space**: O(target) for DP array
///
/// **Example**:
/// ```zig
/// const set = [_]i32{1, 2, 3, 3};
/// const count = try countSubsets(i32, allocator, &set, 6);
/// // Returns 3: subsets {1,2,3}, {1,2,3}, {3,3}
/// ```
pub fn countSubsets(comptime T: type, allocator: Allocator, set: []const T, target: T) !usize {
    if (set.len == 0) return 0;
    if (target < 0) return 0;
    if (target == 0) return 1; // Empty subset

    const target_usize = @as(usize, @intCast(target));

    // DP array: dp[j] = number of ways to achieve sum j
    var dp = try allocator.alloc(usize, target_usize + 1);
    defer allocator.free(dp);

    @memset(dp, 0);
    dp[0] = 1; // Base case: one way to achieve sum 0 (empty subset)

    // Process each element
    for (set) |num| {
        if (num < 0 or num > target) continue;

        const num_usize = @as(usize, @intCast(num));

        // Traverse from target down to num
        var j = target_usize;
        while (j >= num_usize) : (j -= 1) {
            dp[j] += dp[j - num_usize];
            if (j == 0) break;
        }
    }

    return dp[target_usize];
}

/// Check if the set can be partitioned into two subsets with equal sum.
///
/// This is a special case of subset sum where target = total_sum / 2.
/// Only possible if total sum is even.
///
/// **Time**: O(n × sum/2)
/// **Space**: O(sum/2)
///
/// **Example**:
/// ```zig
/// const set = [_]i32{1, 5, 11, 5};
/// const result = try canPartitionEqual(i32, allocator, &set);
/// // Returns true: {1, 5, 5} and {11} both sum to 11
/// ```
pub fn canPartitionEqual(comptime T: type, allocator: Allocator, set: []const T) !bool {
    if (set.len == 0) return false;

    // Calculate total sum
    var total: T = 0;
    for (set) |num| {
        total += num;
    }

    // If total is odd, can't partition into equal sums
    if (@rem(total, 2) != 0) return false;

    const target = @divTrunc(total, 2);
    return try canPartition(T, allocator, set, target);
}

/// Find the minimum subset sum difference when partitioning into two subsets.
///
/// This finds two subsets that minimize |sum(S1) - sum(S2)|.
///
/// **Time**: O(n × total_sum)
/// **Space**: O(total_sum)
///
/// **Example**:
/// ```zig
/// const set = [_]i32{1, 6, 11, 5};
/// const diff = try minSubsetSumDiff(i32, allocator, &set);
/// // Returns 1: {1, 5, 6} sum to 12, {11} sums to 11, diff = 1
/// ```
pub fn minSubsetSumDiff(comptime T: type, allocator: Allocator, set: []const T) !T {
    if (set.len == 0) return 0;

    // Calculate total sum
    var total: T = 0;
    for (set) |num| {
        total += num;
    }

    const target_usize = @as(usize, @intCast(total));

    // DP array: dp[j] = true if sum j is achievable
    var dp = try allocator.alloc(bool, target_usize + 1);
    defer allocator.free(dp);

    @memset(dp, false);
    dp[0] = true;

    // Fill DP table
    for (set) |num| {
        if (num < 0 or num > total) continue;

        const num_usize = @as(usize, @intCast(num));

        var j = target_usize;
        while (j >= num_usize) : (j -= 1) {
            if (dp[j - num_usize]) {
                dp[j] = true;
            }
            if (j == 0) break;
        }
    }

    // Find the largest sum <= total/2 that is achievable
    var min_diff = total;
    const half = @divTrunc(total, 2);

    var sum1: T = 0;
    while (sum1 <= half) : (sum1 += 1) {
        const sum1_usize = @as(usize, @intCast(sum1));
        if (dp[sum1_usize]) {
            const sum2 = total - sum1;
            const diff = if (sum2 > sum1) sum2 - sum1 else sum1 - sum2;
            if (diff < min_diff) {
                min_diff = diff;
            }
        }
    }

    return min_diff;
}

// ============================================================================
// Tests
// ============================================================================

test "canPartition: basic subset sum exists" {
    const allocator = std.testing.allocator;
    const set = [_]i32{ 3, 34, 4, 12, 5, 2 };

    // Target 9 can be achieved: {4, 5} or {3, 4, 2}
    try std.testing.expect(try canPartition(i32, allocator, &set, 9));
}

test "canPartition: subset sum does not exist" {
    const allocator = std.testing.allocator;
    const set = [_]i32{ 3, 34, 4, 12, 5, 2 };

    // Target 30 cannot be achieved
    try std.testing.expect(!try canPartition(i32, allocator, &set, 30));
}

test "canPartition: target 0" {
    const allocator = std.testing.allocator;
    const set = [_]i32{ 1, 2, 3 };

    // Empty subset has sum 0
    try std.testing.expect(try canPartition(i32, allocator, &set, 0));
}

test "canPartition: single element matching target" {
    const allocator = std.testing.allocator;
    const set = [_]i32{7};

    try std.testing.expect(try canPartition(i32, allocator, &set, 7));
    try std.testing.expect(!try canPartition(i32, allocator, &set, 5));
}

test "canPartition: empty set" {
    const allocator = std.testing.allocator;
    const set = [_]i32{};

    try std.testing.expect(!try canPartition(i32, allocator, &set, 5));
}

test "canPartition: negative target" {
    const allocator = std.testing.allocator;
    const set = [_]i32{ 1, 2, 3 };

    try std.testing.expect(!try canPartition(i32, allocator, &set, -5));
}

test "findSubset: basic reconstruction" {
    const allocator = std.testing.allocator;
    const set = [_]i32{ 3, 34, 4, 12, 5, 2 };

    const subset = (try findSubset(i32, allocator, &set, 9)) orelse {
        try std.testing.expect(false); // Should find a subset
        return;
    };
    defer subset.deinit();

    // Verify sum equals target
    var sum: i32 = 0;
    for (subset.items) |idx| {
        sum += set[idx];
    }
    try std.testing.expectEqual(@as(i32, 9), sum);
}

test "findSubset: no subset exists" {
    const allocator = std.testing.allocator;
    const set = [_]i32{ 3, 34, 4, 12, 5, 2 };

    const subset = try findSubset(i32, allocator, &set, 100);
    try std.testing.expect(subset == null);
}

test "findSubset: empty subset for target 0" {
    const allocator = std.testing.allocator;
    const set = [_]i32{ 1, 2, 3 };

    const subset = (try findSubset(i32, allocator, &set, 0)) orelse {
        try std.testing.expect(false);
        return;
    };
    defer subset.deinit();

    try std.testing.expectEqual(@as(usize, 0), subset.items.len);
}

test "countSubsets: basic counting" {
    const allocator = std.testing.allocator;
    const set = [_]i32{ 2, 3, 5, 6, 8, 10 };

    // Target 10: {10}, {2, 8}, {2, 3, 5} = 3 subsets
    const count = try countSubsets(i32, allocator, &set, 10);
    try std.testing.expectEqual(@as(usize, 3), count);
}

test "countSubsets: with duplicates" {
    const allocator = std.testing.allocator;
    const set = [_]i32{ 1, 1, 2, 2 };

    // Target 3: {1, 2}, {1, 2} = 2 subsets (duplicates counted separately)
    const count = try countSubsets(i32, allocator, &set, 3);
    try std.testing.expect(count >= 2); // At least 2 ways
}

test "countSubsets: target 0" {
    const allocator = std.testing.allocator;
    const set = [_]i32{ 1, 2, 3 };

    // Only empty subset
    const count = try countSubsets(i32, allocator, &set, 0);
    try std.testing.expectEqual(@as(usize, 1), count);
}

test "countSubsets: no subsets exist" {
    const allocator = std.testing.allocator;
    const set = [_]i32{ 2, 4, 6 };

    // Target 5 cannot be achieved (all even numbers)
    const count = try countSubsets(i32, allocator, &set, 5);
    try std.testing.expectEqual(@as(usize, 0), count);
}

test "canPartitionEqual: basic equal partition" {
    const allocator = std.testing.allocator;
    const set = [_]i32{ 1, 5, 11, 5 };

    // {1, 5, 5} and {11} both sum to 11
    try std.testing.expect(try canPartitionEqual(i32, allocator, &set));
}

test "canPartitionEqual: cannot partition" {
    const allocator = std.testing.allocator;
    const set = [_]i32{ 1, 2, 3, 5 };

    // Total sum = 11 (odd), cannot partition equally
    try std.testing.expect(!try canPartitionEqual(i32, allocator, &set));
}

test "canPartitionEqual: empty set" {
    const allocator = std.testing.allocator;
    const set = [_]i32{};

    try std.testing.expect(!try canPartitionEqual(i32, allocator, &set));
}

test "minSubsetSumDiff: basic minimum difference" {
    const allocator = std.testing.allocator;
    const set = [_]i32{ 1, 6, 11, 5 };

    // {1, 5, 6} = 12, {11} = 11, diff = 1
    const diff = try minSubsetSumDiff(i32, allocator, &set);
    try std.testing.expectEqual(@as(i32, 1), diff);
}

test "minSubsetSumDiff: perfect partition" {
    const allocator = std.testing.allocator;
    const set = [_]i32{ 1, 5, 11, 5 };

    // {1, 5, 5} = 11, {11} = 11, diff = 0
    const diff = try minSubsetSumDiff(i32, allocator, &set);
    try std.testing.expectEqual(@as(i32, 0), diff);
}

test "minSubsetSumDiff: single element" {
    const allocator = std.testing.allocator;
    const set = [_]i32{10};

    // One subset gets all, other gets nothing
    const diff = try minSubsetSumDiff(i32, allocator, &set);
    try std.testing.expectEqual(@as(i32, 10), diff);
}

test "canPartition: large dataset" {
    const allocator = std.testing.allocator;
    var set = [_]i32{0} ** 50;

    // Fill with values 1 to 50
    for (0..50) |i| {
        set[i] = @intCast(i + 1);
    }

    // Total sum = 1275, half = 637 or 638
    // Many subsets can sum to 637
    try std.testing.expect(try canPartition(i32, allocator, &set, 637));
}

test "findSubset: large target" {
    const allocator = std.testing.allocator;
    const set = [_]i32{ 10, 20, 30, 40, 50, 100 };

    const subset = (try findSubset(i32, allocator, &set, 150)) orelse {
        try std.testing.expect(false);
        return;
    };
    defer subset.deinit();

    var sum: i32 = 0;
    for (subset.items) |idx| {
        sum += set[idx];
    }
    try std.testing.expectEqual(@as(i32, 150), sum);
}

test "subset sum: memory safety" {
    const allocator = std.testing.allocator;
    const set = [_]i32{ 1, 2, 3, 4, 5 };

    // Test all functions for memory leaks
    _ = try canPartition(i32, allocator, &set, 7);
    _ = try countSubsets(i32, allocator, &set, 7);
    _ = try canPartitionEqual(i32, allocator, &set);
    _ = try minSubsetSumDiff(i32, allocator, &set);

    const subset = try findSubset(i32, allocator, &set, 7);
    if (subset) |s| {
        s.deinit();
    }
}
