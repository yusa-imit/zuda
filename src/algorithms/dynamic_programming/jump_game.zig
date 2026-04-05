//! Jump Game - Classic DP greedy problem
//!
//! Given an array where each element represents maximum jump length from that position,
//! determine if you can reach the last index, find minimum jumps needed, or count ways.
//!
//! **Algorithm**: Dynamic programming with greedy optimization
//!
//! **Key Features**:
//! - canJump(): Check if last index is reachable - O(n) time, O(1) space (greedy)
//! - canJumpDP(): DP variant - O(n²) time, O(n) space
//! - minJumps(): Minimum jumps to reach end - O(n²) DP or O(n) BFS
//! - minJumpsGreedy(): Greedy variant - O(n) time, O(1) space
//! - countWays(): Count distinct paths to reach end - O(n²) time, O(n) space
//! - jumpPath(): Actual sequence of indices taken - O(n) space for path
//!
//! **Time Complexity**:
//! - canJump: O(n) greedy
//! - minJumps: O(n²) DP, O(n) greedy
//! - countWays: O(n²)
//!
//! **Space Complexity**: O(1) for greedy, O(n) for DP variants
//!
//! **Use Cases**:
//! - Game pathfinding (board games, platformers)
//! - Network routing (minimum hops with capacity constraints)
//! - Compiler optimization (instruction scheduling)
//! - Resource allocation (step-by-step planning)
//!
//! **Reference**: LeetCode #55 (Jump Game), #45 (Jump Game II), classic greedy/DP problem

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Check if you can reach the last index (greedy approach)
/// Time: O(n) | Space: O(1)
pub fn canJump(comptime T: type, nums: []const T) !bool {
    if (nums.len == 0) return error.EmptyArray;
    if (nums.len == 1) return true;

    var max_reach: usize = 0;
    for (nums, 0..) |num, i| {
        if (i > max_reach) return false; // Can't reach this position
        const reach = i + @as(usize, @intCast(num));
        max_reach = @max(max_reach, reach);
        if (max_reach >= nums.len - 1) return true;
    }
    return max_reach >= nums.len - 1;
}

/// Check if you can reach the last index (DP approach)
/// Time: O(n²) | Space: O(n)
pub fn canJumpDP(comptime T: type, nums: []const T, allocator: Allocator) !bool {
    if (nums.len == 0) return error.EmptyArray;
    if (nums.len == 1) return true;

    const n = nums.len;
    var dp = try allocator.alloc(bool, n);
    defer allocator.free(dp);

    @memset(dp, false);
    dp[0] = true;

    for (0..n) |i| {
        if (!dp[i]) continue;
        const jump_len = @as(usize, @intCast(nums[i]));
        const max_j = @min(i + jump_len, n - 1);
        for (i + 1..max_j + 1) |j| {
            dp[j] = true;
            if (j == n - 1) return true;
        }
    }

    return dp[n - 1];
}

/// Find minimum number of jumps to reach the last index (DP approach)
/// Returns error.Unreachable if cannot reach end
/// Time: O(n²) | Space: O(n)
pub fn minJumps(comptime T: type, nums: []const T, allocator: Allocator) !usize {
    if (nums.len == 0) return error.EmptyArray;
    if (nums.len == 1) return 0;

    const n = nums.len;
    var dp = try allocator.alloc(?usize, n);
    defer allocator.free(dp);

    @memset(dp, null);
    dp[0] = 0;

    for (0..n) |i| {
        if (dp[i] == null) continue;
        const jump_len = @as(usize, @intCast(nums[i]));
        const max_j = @min(i + jump_len, n - 1);
        for (i + 1..max_j + 1) |j| {
            if (dp[j] == null or dp[j].? > dp[i].? + 1) {
                dp[j] = dp[i].? + 1;
            }
        }
    }

    return dp[n - 1] orelse error.Unreachable;
}

/// Find minimum number of jumps (greedy BFS approach - optimal)
/// Returns error.Unreachable if cannot reach end
/// Time: O(n) | Space: O(1)
pub fn minJumpsGreedy(comptime T: type, nums: []const T) !usize {
    if (nums.len == 0) return error.EmptyArray;
    if (nums.len == 1) return 0;

    const n = nums.len;
    var jumps: usize = 0;
    var current_end: usize = 0;
    var farthest: usize = 0;

    for (0..n - 1) |i| {
        const reach = i + @as(usize, @intCast(nums[i]));
        farthest = @max(farthest, reach);

        if (i == current_end) {
            jumps += 1;
            current_end = farthest;
            if (current_end >= n - 1) return jumps;
        }
    }

    if (current_end >= n - 1) return jumps;
    return error.Unreachable;
}

/// Count number of distinct ways to reach the last index
/// Time: O(n²) | Space: O(n)
pub fn countWays(comptime T: type, nums: []const T, allocator: Allocator) !usize {
    if (nums.len == 0) return error.EmptyArray;
    if (nums.len == 1) return 1;

    const n = nums.len;
    var dp = try allocator.alloc(usize, n);
    defer allocator.free(dp);

    @memset(dp, 0);
    dp[0] = 1;

    for (0..n) |i| {
        if (dp[i] == 0) continue;
        const jump_len = @as(usize, @intCast(nums[i]));
        const max_j = @min(i + jump_len, n - 1);
        for (i + 1..max_j + 1) |j| {
            dp[j] += dp[i];
        }
    }

    return dp[n - 1];
}

pub const JumpPath = struct {
    indices: []usize,
    jumps: usize,

    pub fn deinit(self: JumpPath, allocator: Allocator) void {
        allocator.free(self.indices);
    }
};

/// Get actual jump path (sequence of indices)
/// Returns error.Unreachable if cannot reach end
/// Time: O(n²) | Space: O(n)
pub fn jumpPath(comptime T: type, nums: []const T, allocator: Allocator) !JumpPath {
    if (nums.len == 0) return error.EmptyArray;
    if (nums.len == 1) {
        const path = try allocator.alloc(usize, 1);
        path[0] = 0;
        return JumpPath{ .indices = path, .jumps = 0 };
    }

    const n = nums.len;
    var dp = try allocator.alloc(?usize, n);
    defer allocator.free(dp);
    var parent = try allocator.alloc(?usize, n);
    defer allocator.free(parent);

    @memset(dp, null);
    @memset(parent, null);
    dp[0] = 0;

    for (0..n) |i| {
        if (dp[i] == null) continue;
        const jump_len = @as(usize, @intCast(nums[i]));
        const max_j = @min(i + jump_len, n - 1);
        for (i + 1..max_j + 1) |j| {
            if (dp[j] == null or dp[j].? > dp[i].? + 1) {
                dp[j] = dp[i].? + 1;
                parent[j] = i;
            }
        }
    }

    if (dp[n - 1] == null) return error.Unreachable;

    const jumps = dp[n - 1].?;
    var path = try allocator.alloc(usize, jumps + 1);
    errdefer allocator.free(path);

    var idx: usize = n - 1;
    var pos: usize = jumps;
    while (true) {
        path[pos] = idx;
        if (idx == 0) break;
        idx = parent[idx].?;
        pos -= 1;
    }

    return JumpPath{ .indices = path, .jumps = jumps };
}

test "jump game - can reach end (greedy)" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Can reach end
    try testing.expect(try canJump(u32, &[_]u32{ 2, 3, 1, 1, 4 }));
    try testing.expect(try canJump(u32, &[_]u32{ 3, 2, 1, 0, 4 })); // Can jump over 0

    // Cannot reach end
    try testing.expect(!try canJump(u32, &[_]u32{ 1, 0, 1, 0 }));

    // Edge cases
    try testing.expect(try canJump(u32, &[_]u32{0})); // Single element
    try testing.expect(try canJump(u32, &[_]u32{ 1, 1, 1, 1 })); // All reachable

    // Empty array should error
    try testing.expectError(error.EmptyArray, canJump(u32, &[_]u32{}));

    _ = allocator;
}

test "jump game - can reach end (DP)" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Can reach end
    try testing.expect(try canJumpDP(u32, &[_]u32{ 2, 3, 1, 1, 4 }, allocator));
    try testing.expect(try canJumpDP(u32, &[_]u32{ 3, 2, 1, 0, 4 }, allocator));

    // Cannot reach end
    try testing.expect(!try canJumpDP(u32, &[_]u32{ 1, 0, 1, 0 }, allocator));

    // Edge cases
    try testing.expect(try canJumpDP(u32, &[_]u32{0}, allocator));
}

test "jump game - minimum jumps (DP)" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Basic examples
    try testing.expectEqual(@as(usize, 2), try minJumps(u32, &[_]u32{ 2, 3, 1, 1, 4 }, allocator));
    try testing.expectEqual(@as(usize, 2), try minJumps(u32, &[_]u32{ 2, 3, 0, 1, 4 }, allocator));

    // Single element
    try testing.expectEqual(@as(usize, 0), try minJumps(u32, &[_]u32{0}, allocator));

    // Unreachable
    try testing.expectError(error.Unreachable, minJumps(u32, &[_]u32{ 1, 0, 1, 0 }, allocator));
}

test "jump game - minimum jumps (greedy)" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Basic examples
    try testing.expectEqual(@as(usize, 2), try minJumpsGreedy(u32, &[_]u32{ 2, 3, 1, 1, 4 }));
    try testing.expectEqual(@as(usize, 2), try minJumpsGreedy(u32, &[_]u32{ 2, 3, 0, 1, 4 }));

    // Large jump
    try testing.expectEqual(@as(usize, 1), try minJumpsGreedy(u32, &[_]u32{ 10, 1, 1, 1, 1, 1 }));

    // All ones
    try testing.expectEqual(@as(usize, 4), try minJumpsGreedy(u32, &[_]u32{ 1, 1, 1, 1, 1 }));

    // Single element
    try testing.expectEqual(@as(usize, 0), try minJumpsGreedy(u32, &[_]u32{0}));

    // Unreachable
    try testing.expectError(error.Unreachable, minJumpsGreedy(u32, &[_]u32{ 1, 0, 1, 0 }));

    _ = allocator;
}

test "jump game - count ways" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Basic examples
    try testing.expectEqual(@as(usize, 3), try countWays(u32, &[_]u32{ 2, 3, 1, 1, 4 }, allocator));
    // From 0: jump to 1 or 2
    // From 1: can reach 2,3,4 (paths: 0→1→2, 0→1→3, 0→1→4)
    // From 2: can reach 3 (paths: 0→2→3)
    // Total: 0→1→4, 0→1→3→4, 0→2→3→4 = 3 ways

    // All ones (only one way - step by step)
    try testing.expectEqual(@as(usize, 1), try countWays(u32, &[_]u32{ 1, 1, 1, 1 }, allocator));

    // Two elements
    try testing.expectEqual(@as(usize, 1), try countWays(u32, &[_]u32{ 1, 0 }, allocator));

    // Unreachable returns 0
    try testing.expectEqual(@as(usize, 0), try countWays(u32, &[_]u32{ 1, 0, 1, 0 }, allocator));
}

test "jump game - jump path" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Basic path
    {
        const result = try jumpPath(u32, &[_]u32{ 2, 3, 1, 1, 4 }, allocator);
        defer result.deinit(allocator);
        try testing.expectEqual(@as(usize, 2), result.jumps);
        try testing.expectEqual(@as(usize, 3), result.indices.len);
        try testing.expectEqual(@as(usize, 0), result.indices[0]);
        try testing.expectEqual(@as(usize, 4), result.indices[2]);
    }

    // Single element
    {
        const result = try jumpPath(u32, &[_]u32{5}, allocator);
        defer result.deinit(allocator);
        try testing.expectEqual(@as(usize, 0), result.jumps);
        try testing.expectEqual(@as(usize, 1), result.indices.len);
        try testing.expectEqual(@as(usize, 0), result.indices[0]);
    }

    // Unreachable
    try testing.expectError(error.Unreachable, jumpPath(u32, &[_]u32{ 1, 0, 1, 0 }, allocator));
}

test "jump game - greedy vs DP consistency" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const test_cases = [_][]const u32{
        &[_]u32{ 2, 3, 1, 1, 4 },
        &[_]u32{ 2, 3, 0, 1, 4 },
        &[_]u32{ 1, 1, 1, 1, 1 },
        &[_]u32{ 5, 1, 1, 1, 1 },
    };

    for (test_cases) |nums| {
        const greedy = try minJumpsGreedy(u32, nums);
        const dp = try minJumps(u32, nums, allocator);
        try testing.expectEqual(greedy, dp);
    }
}

test "jump game - large array" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // All ones - should take n-1 jumps
    var nums = try allocator.alloc(u32, 100);
    defer allocator.free(nums);
    @memset(nums, 1);

    try testing.expectEqual(@as(usize, 99), try minJumpsGreedy(u32, nums));
    try testing.expectEqual(@as(usize, 1), try countWays(u32, nums, allocator));

    // All 2s - multiple paths
    @memset(nums, 2);
    const ways = try countWays(u32, nums, allocator);
    try testing.expect(ways > 1);
}

test "jump game - i64 support" {
    const testing = std.testing;
    const allocator = testing.allocator;

    try testing.expect(try canJump(i64, &[_]i64{ 2, 3, 1, 1, 4 }));
    try testing.expectEqual(@as(usize, 2), try minJumpsGreedy(i64, &[_]i64{ 2, 3, 1, 1, 4 }));

    _ = allocator;
}

test "jump game - memory safety" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const nums = [_]u32{ 2, 3, 1, 1, 4 };

    // Run multiple times to check for leaks
    for (0..10) |_| {
        _ = try canJumpDP(u32, &nums, allocator);
        _ = try minJumps(u32, &nums, allocator);
        _ = try countWays(u32, &nums, allocator);

        const path = try jumpPath(u32, &nums, allocator);
        path.deinit(allocator);
    }
}
