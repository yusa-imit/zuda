const std = @import("std");
const Allocator = std.mem.Allocator;

/// Target Sum Problem — Dynamic Programming Solutions
///
/// The target sum problem asks: given an array of non-negative integers and a target sum,
/// assign + or - signs to each number such that the sum equals the target. Count how many
/// different ways this can be done.
///
/// Mathematical Reduction to Subset Sum:
/// - Let P = sum of numbers assigned +, N = sum of numbers assigned -
/// - We want: P - N = target
/// - We know: P + N = total_sum (all elements)
/// - Solving: P = (target + total_sum) / 2
/// - Problem reduces to: count subsets with sum = P
///
/// Applications:
/// - Expression evaluation (parenthesization with +/-)
/// - Portfolio optimization (long/short positions)
/// - Resource allocation (positive/negative contributions)
/// - Game theory (scoring with gains/losses)
/// - Statistical modeling (signed feature combinations)
///
/// Reference: LeetCode #494 (Target Sum), classic DP counting problem
///
/// Time complexity: O(n × sum) where n = array length, sum = total of all elements
/// Space complexity: O(sum) with space optimization, O(n × sum) with 2D DP

/// Count the number of ways to assign +/- signs to reach the target sum.
///
/// Uses bottom-up DP with space optimization. The algorithm reduces the problem to
/// subset sum counting: find subsets with sum = (target + total_sum) / 2.
///
/// **Algorithm**:
/// - Calculate total_sum and validate feasibility
/// - Reduce to subset sum with target_sum = (target + total_sum) / 2
/// - Use DP table: dp[j] = number of ways to sum to j
/// - For each number x, update dp backward to avoid reusing same element
///
/// **Time**: O(n × sum) where n = array length, sum = total of all elements
/// **Space**: O(sum) with rolling array optimization
///
/// **Example**:
/// ```zig
/// const allocator = std.heap.page_allocator;
/// const nums = [_]i32{ 1, 1, 1, 1, 1 };
/// const ways = try findTargetSumWays(i32, allocator, &nums, 3);
/// // Result: 5 ways to assign +/- to get 3
/// // (+1+1+1+1-1), (+1+1+1-1+1), (+1+1-1+1+1), (+1-1+1+1+1), (-1+1+1+1+1)
/// ```
///
/// **Errors**:
/// - `error.InvalidTarget` if (target + total_sum) is odd or target exceeds total_sum
/// - `error.OutOfMemory` if allocation fails
pub fn findTargetSumWays(comptime T: type, allocator: Allocator, nums: []const T, target: T) !usize {
    if (nums.len == 0) return if (target == 0) 1 else 0;

    // Calculate total sum
    var total_sum: T = 0;
    for (nums) |num| {
        total_sum += num;
    }

    // Check feasibility: target must be within [-total_sum, total_sum]
    if (@abs(target) > total_sum) return error.InvalidTarget;

    // Check parity: (target + total_sum) must be even
    if (@mod(target + total_sum, 2) != 0) return error.InvalidTarget;

    // Reduce to subset sum: find subsets with sum = (target + total_sum) / 2
    const target_sum = @divTrunc(target + total_sum, 2);
    if (target_sum < 0) return error.InvalidTarget;

    // DP table: dp[j] = number of ways to sum to j
    const dp_size = @as(usize, @intCast(target_sum + 1));
    const dp = try allocator.alloc(usize, dp_size);
    defer allocator.free(dp);
    @memset(dp, 0);
    dp[0] = 1; // Base case: one way to sum to 0 (empty subset)

    // For each number in the array
    for (nums) |num| {
        const num_usize = @as(usize, @intCast(num));
        // Update DP backward to avoid reusing same element
        var j: usize = dp_size - 1;
        while (j >= num_usize) : (j -= 1) {
            dp[j] += dp[j - num_usize];
            if (j == 0) break; // Prevent underflow
        }
    }

    return dp[dp_size - 1];
}

/// Count ways with detailed 2D DP table (for educational purposes).
///
/// Returns the full DP table showing number of ways at each step. Less space-efficient
/// than the 1D rolling array but useful for understanding the algorithm.
///
/// **Time**: O(n × sum)
/// **Space**: O(n × sum)
///
/// **Example**:
/// ```zig
/// const allocator = std.heap.page_allocator;
/// const nums = [_]i32{ 1, 2, 3 };
/// const result = try findTargetSumWaysTable(i32, allocator, &nums, 0);
/// defer allocator.free(result.table);
/// // result.ways = 2 (ways to get 0: +1+2-3, -1-2+3)
/// // result.table shows DP table for each step
/// ```
pub fn findTargetSumWaysTable(comptime T: type, allocator: Allocator, nums: []const T, target: T) !struct { ways: usize, table: []usize } {
    if (nums.len == 0) return .{ .ways = if (target == 0) 1 else 0, .table = try allocator.alloc(usize, 0) };

    var total_sum: T = 0;
    for (nums) |num| {
        total_sum += num;
    }

    if (@abs(target) > total_sum) return error.InvalidTarget;
    if (@mod(target + total_sum, 2) != 0) return error.InvalidTarget;

    const target_sum = @divTrunc(target + total_sum, 2);
    if (target_sum < 0) return error.InvalidTarget;

    const dp_size = @as(usize, @intCast(target_sum + 1));
    const table = try allocator.alloc(usize, dp_size * nums.len);
    @memset(table, 0);

    // Initialize first row
    table[0] = 1;

    // Fill DP table
    for (nums, 0..) |num, i| {
        const num_usize = @as(usize, @intCast(num));
        const row_offset = i * dp_size;
        const prev_offset = if (i > 0) (i - 1) * dp_size else 0;

        for (0..dp_size) |j| {
            // Copy from previous row
            if (i > 0) {
                table[row_offset + j] = table[prev_offset + j];
            }
            // Add contribution from current number
            if (j >= num_usize) {
                const prev_val = if (i > 0) table[prev_offset + j - num_usize] else if (j == num_usize) 1 else 0;
                table[row_offset + j] += prev_val;
            }
        }
    }

    const last_row = (nums.len - 1) * dp_size;
    return .{ .ways = table[last_row + dp_size - 1], .table = table };
}

/// Count ways with memoization (top-down DP).
///
/// Uses recursive approach with memoization map. This is more intuitive but less
/// space-efficient than bottom-up DP due to recursion stack and hash map overhead.
///
/// **Time**: O(n × sum)
/// **Space**: O(n × sum) for memoization map + O(n) recursion stack
///
/// **Example**:
/// ```zig
/// const allocator = std.heap.page_allocator;
/// const nums = [_]i32{ 1, 1, 1, 1, 1 };
/// const ways = try findTargetSumWaysMemo(i32, allocator, &nums, 3);
/// // Result: 5 ways
/// ```
pub fn findTargetSumWaysMemo(comptime T: type, allocator: Allocator, nums: []const T, target: T) !usize {
    const State = struct {
        index: usize,
        current_sum: T,

        pub fn hash(self: @This()) u64 {
            var hasher = std.hash.Wyhash.init(0);
            std.hash.autoHash(&hasher, self.index);
            std.hash.autoHash(&hasher, self.current_sum);
            return hasher.final();
        }

        pub fn eql(self: @This(), other: @This()) bool {
            return self.index == other.index and self.current_sum == other.current_sum;
        }
    };

    var memo = std.HashMap(State, usize, struct {
        pub fn hash(_: @This(), s: State) u64 {
            return s.hash();
        }
        pub fn eql(_: @This(), a: State, b: State) bool {
            return a.eql(b);
        }
    }, std.hash_map.default_max_load_percentage).init(allocator);
    defer memo.deinit();

    const Helper = struct {
        fn solve(m: *std.HashMap(State, usize, anytype, std.hash_map.default_max_load_percentage), n: []const T, idx: usize, sum: T, tgt: T) !usize {
            if (idx == n.len) {
                return if (sum == tgt) 1 else 0;
            }

            const state = State{ .index = idx, .current_sum = sum };
            if (m.get(state)) |cached| {
                return cached;
            }

            const add = try solve(m, n, idx + 1, sum + n[idx], tgt);
            const sub = try solve(m, n, idx + 1, sum - n[idx], tgt);
            const result = add + sub;

            try m.put(state, result);
            return result;
        }
    };

    return Helper.solve(&memo, nums, 0, 0, target);
}

/// Get one valid assignment of +/- signs (if exists).
///
/// Returns an ArrayList of booleans where true = +, false = -.
/// Returns null if no valid assignment exists.
///
/// **Time**: O(n × sum)
/// **Space**: O(sum) for DP + O(n) for result
///
/// **Example**:
/// ```zig
/// const allocator = std.heap.page_allocator;
/// const nums = [_]i32{ 1, 2, 3 };
/// const assignment = try getTargetSumAssignment(i32, allocator, &nums, 0);
/// defer if (assignment) |a| a.deinit();
/// // assignment might be [true, true, false] representing +1+2-3=0
/// ```
pub fn getTargetSumAssignment(comptime T: type, allocator: Allocator, nums: []const T, target: T) !?std.ArrayList(bool) {
    if (nums.len == 0) return if (target == 0) std.ArrayList(bool).init(allocator) else null;

    var total_sum: T = 0;
    for (nums) |num| {
        total_sum += num;
    }

    if (@abs(target) > total_sum) return null;
    if (@mod(target + total_sum, 2) != 0) return null;

    const target_sum = @divTrunc(target + total_sum, 2);
    if (target_sum < 0) return null;

    // Build DP table to check feasibility
    const dp_size = @as(usize, @intCast(target_sum + 1));
    const dp = try allocator.alloc(bool, dp_size);
    defer allocator.free(dp);
    @memset(dp, false);
    dp[0] = true;

    for (nums) |num| {
        const num_usize = @as(usize, @intCast(num));
        var j: usize = dp_size - 1;
        while (j >= num_usize) : (j -= 1) {
            if (dp[j - num_usize]) {
                dp[j] = true;
            }
            if (j == 0) break;
        }
    }

    if (!dp[dp_size - 1]) return null;

    // Backtrack to find assignment
    var assignment = std.ArrayList(bool).init(allocator);
    errdefer assignment.deinit();

    var remaining = @as(usize, @intCast(target_sum));
    var i: usize = nums.len;
    while (i > 0) {
        i -= 1;
        const num_usize = @as(usize, @intCast(nums[i]));
        if (remaining >= num_usize) {
            // Try including this number in positive subset
            @memset(dp, false);
            dp[0] = true;
            for (0..i) |k| {
                const n = @as(usize, @intCast(nums[k]));
                var j: usize = dp_size - 1;
                while (j >= n) : (j -= 1) {
                    if (dp[j - n]) {
                        dp[j] = true;
                    }
                    if (j == 0) break;
                }
            }

            if (dp[remaining - num_usize]) {
                try assignment.append(true); // Positive
                remaining -= num_usize;
            } else {
                try assignment.append(false); // Negative
            }
        } else {
            try assignment.append(false); // Must be negative
        }
    }

    // Reverse to get correct order
    std.mem.reverse(bool, assignment.items);
    return assignment;
}

// =============================================================================
// Tests
// =============================================================================

test "findTargetSumWays: basic example" {
    const allocator = std.testing.allocator;
    const nums = [_]i32{ 1, 1, 1, 1, 1 };
    const ways = try findTargetSumWays(i32, allocator, &nums, 3);
    try std.testing.expectEqual(@as(usize, 5), ways);
}

test "findTargetSumWays: zero target" {
    const allocator = std.testing.allocator;
    const nums = [_]i32{ 1, 2, 3 };
    const ways = try findTargetSumWays(i32, allocator, &nums, 0);
    try std.testing.expectEqual(@as(usize, 2), ways); // +1+2-3=0, -1-2+3=0
}

test "findTargetSumWays: single element" {
    const allocator = std.testing.allocator;
    const nums = [_]i32{5};
    const ways_pos = try findTargetSumWays(i32, allocator, &nums, 5);
    const ways_neg = try findTargetSumWays(i32, allocator, &nums, -5);
    try std.testing.expectEqual(@as(usize, 1), ways_pos);
    try std.testing.expectEqual(@as(usize, 1), ways_neg);
}

test "findTargetSumWays: two elements" {
    const allocator = std.testing.allocator;
    const nums = [_]i32{ 1, 1 };
    const ways = try findTargetSumWays(i32, allocator, &nums, 0);
    try std.testing.expectEqual(@as(usize, 2), ways); // +1-1=0, -1+1=0
}

test "findTargetSumWays: empty array" {
    const allocator = std.testing.allocator;
    const nums = [_]i32{};
    const ways_zero = try findTargetSumWays(i32, allocator, &nums, 0);
    const ways_nonzero = try findTargetSumWays(i32, allocator, &nums, 5);
    try std.testing.expectEqual(@as(usize, 1), ways_zero); // Empty sum = 0
    try std.testing.expectEqual(@as(usize, 0), ways_nonzero);
}

test "findTargetSumWays: invalid target (too large)" {
    const allocator = std.testing.allocator;
    const nums = [_]i32{ 1, 2, 3 };
    const result = findTargetSumWays(i32, allocator, &nums, 10);
    try std.testing.expectError(error.InvalidTarget, result);
}

test "findTargetSumWays: invalid target (wrong parity)" {
    const allocator = std.testing.allocator;
    const nums = [_]i32{ 2, 2, 2 }; // Sum = 6
    const result = findTargetSumWays(i32, allocator, &nums, 1); // 1+6=7 (odd)
    try std.testing.expectError(error.InvalidTarget, result);
}

test "findTargetSumWays: all zeros" {
    const allocator = std.testing.allocator;
    const nums = [_]i32{ 0, 0, 0 };
    const ways = try findTargetSumWays(i32, allocator, &nums, 0);
    try std.testing.expectEqual(@as(usize, 8), ways); // 2^3 ways (each 0 can be +/-)
}

test "findTargetSumWays: with zeros" {
    const allocator = std.testing.allocator;
    const nums = [_]i32{ 0, 1 };
    const ways = try findTargetSumWays(i32, allocator, &nums, 1);
    try std.testing.expectEqual(@as(usize, 2), ways); // +0+1=1, -0+1=1
}

test "findTargetSumWays: large array" {
    const allocator = std.testing.allocator;
    var nums: [20]i32 = undefined;
    for (&nums, 0..) |*n, i| {
        n.* = @intCast(i % 5 + 1);
    }
    const ways = try findTargetSumWays(i32, allocator, &nums, 10);
    try std.testing.expect(ways > 0);
}

test "findTargetSumWaysTable: basic example" {
    const allocator = std.testing.allocator;
    const nums = [_]i32{ 1, 1, 1 };
    const result = try findTargetSumWaysTable(i32, allocator, &nums, 1);
    defer allocator.free(result.table);
    try std.testing.expectEqual(@as(usize, 3), result.ways); // +1+1-1, +1-1+1, -1+1+1
    try std.testing.expectEqual(@as(usize, 6), result.table.len); // 3 rows × 2 cols
}

test "findTargetSumWaysTable: consistency with 1D" {
    const allocator = std.testing.allocator;
    const nums = [_]i32{ 1, 2, 3, 4 };
    const ways_1d = try findTargetSumWays(i32, allocator, &nums, 0);
    const result_2d = try findTargetSumWaysTable(i32, allocator, &nums, 0);
    defer allocator.free(result_2d.table);
    try std.testing.expectEqual(ways_1d, result_2d.ways);
}

test "findTargetSumWaysMemo: basic example" {
    const allocator = std.testing.allocator;
    const nums = [_]i32{ 1, 1, 1, 1, 1 };
    const ways = try findTargetSumWaysMemo(i32, allocator, &nums, 3);
    try std.testing.expectEqual(@as(usize, 5), ways);
}

test "findTargetSumWaysMemo: consistency with DP" {
    const allocator = std.testing.allocator;
    const nums = [_]i32{ 1, 2, 3, 4, 5 };
    const ways_dp = try findTargetSumWays(i32, allocator, &nums, 3);
    const ways_memo = try findTargetSumWaysMemo(i32, allocator, &nums, 3);
    try std.testing.expectEqual(ways_dp, ways_memo);
}

test "getTargetSumAssignment: valid assignment" {
    const allocator = std.testing.allocator;
    const nums = [_]i32{ 1, 2, 3 };
    const assignment = try getTargetSumAssignment(i32, allocator, &nums, 0);
    defer if (assignment) |a| a.deinit();

    try std.testing.expect(assignment != null);
    if (assignment) |a| {
        try std.testing.expectEqual(@as(usize, 3), a.items.len);
        // Verify the assignment sums to 0
        var sum: i32 = 0;
        for (a.items, 0..) |sign, i| {
            sum += if (sign) nums[i] else -nums[i];
        }
        try std.testing.expectEqual(@as(i32, 0), sum);
    }
}

test "getTargetSumAssignment: no valid assignment" {
    const allocator = std.testing.allocator;
    const nums = [_]i32{ 1, 2, 3 };
    const assignment = try getTargetSumAssignment(i32, allocator, &nums, 10);
    defer if (assignment) |a| a.deinit();
    try std.testing.expect(assignment == null);
}

test "getTargetSumAssignment: single element" {
    const allocator = std.testing.allocator;
    const nums = [_]i32{5};
    const assignment = try getTargetSumAssignment(i32, allocator, &nums, 5);
    defer if (assignment) |a| a.deinit();

    try std.testing.expect(assignment != null);
    if (assignment) |a| {
        try std.testing.expectEqual(@as(usize, 1), a.items.len);
        try std.testing.expectEqual(true, a.items[0]); // +5
    }
}

test "Target Sum: i64 support" {
    const allocator = std.testing.allocator;
    const nums = [_]i64{ 1, 2, 3, 4, 5 };
    const ways = try findTargetSumWays(i64, allocator, &nums, 3);
    try std.testing.expect(ways > 0);
}

test "Target Sum: memory safety" {
    const allocator = std.testing.allocator;
    const nums = [_]i32{ 1, 2, 3, 4, 5 };

    for (0..10) |_| {
        const ways = try findTargetSumWays(i32, allocator, &nums, 3);
        _ = ways;

        const result = try findTargetSumWaysTable(i32, allocator, &nums, 1);
        allocator.free(result.table);

        const ways_memo = try findTargetSumWaysMemo(i32, allocator, &nums, 1);
        _ = ways_memo;

        const assignment = try getTargetSumAssignment(i32, allocator, &nums, 3);
        if (assignment) |a| a.deinit();
    }
}
