//! Longest Arithmetic Progression (LAP)
//!
//! Find the longest arithmetic subsequence (not necessarily contiguous) in an array.
//! An arithmetic sequence has constant difference between consecutive elements.
//!
//! Example: [1, 7, 10, 15, 27, 29] → [1, 15, 29] (length 3, diff 14)
//!
//! Time Complexity: O(n²) for all variants
//! Space Complexity: O(n²) for DP table
//!
//! Algorithm:
//! - Use DP with 2D table: dp[i][d] = length of longest AP ending at i with difference d
//! - For each pair (i, j) where j > i, compute diff = arr[j] - arr[i]
//! - Update dp[j][diff] = dp[i][diff] + 1 (extend AP ending at i)
//! - Track maximum length across all states
//!
//! Use Cases:
//! - Pattern detection in sequences (time series, signal processing)
//! - Numerical analysis (finding linear trends in data)
//! - Competitive programming (LeetCode #1027, #873)
//! - Educational (understanding 2D DP with HashMap)

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Result structure containing arithmetic progression details
pub fn APResult(comptime T: type) type {
    return struct {
        length: usize,
        difference: T,
        start_index: usize,
        indices: ?[]usize = null, // optional path reconstruction

        pub fn deinit(self: @This(), allocator: Allocator) void {
            if (self.indices) |idx| {
                allocator.free(idx);
            }
        }
    };
}

/// Find length of longest arithmetic progression in array
///
/// Time: O(n²), Space: O(n²)
///
/// Example:
/// ```zig
/// const arr = [_]i32{1, 7, 10, 15, 27, 29};
/// const len = try longestArithmeticProgression(i32, &arr, allocator);
/// // len == 3 ([1, 15, 29] with diff 14)
/// ```
pub fn longestArithmeticProgression(comptime T: type, arr: []const T, allocator: Allocator) !usize {
    if (arr.len == 0) return error.EmptyArray;
    if (arr.len <= 2) return arr.len;

    const n = arr.len;
    
    // dp[i] = HashMap where key=diff, value=length of AP ending at i with that diff
    var dp = try allocator.alloc(std.AutoHashMap(T, usize), n);
    defer {
        for (dp) |*map| {
            map.deinit();
        }
        allocator.free(dp);
    }
    
    for (dp) |*map| {
        map.* = std.AutoHashMap(T, usize).init(allocator);
    }
    
    var max_length: usize = 2; // minimum AP length is 2
    
    // For each ending position j
    for (1..n) |j| {
        // For each starting position i < j
        for (0..j) |i| {
            const diff = arr[j] - arr[i];
            
            // Check if there's an AP ending at i with this difference
            const prev_len = dp[i].get(diff) orelse 1;
            const new_len = prev_len + 1;
            
            try dp[j].put(diff, new_len);
            max_length = @max(max_length, new_len);
        }
    }
    
    return max_length;
}

/// Find longest AP with detailed result (length, difference, start position)
///
/// Time: O(n²), Space: O(n²)
pub fn longestAPWithDetails(comptime T: type, arr: []const T, allocator: Allocator) !APResult(T) {
    if (arr.len == 0) return error.EmptyArray;
    if (arr.len == 1) return APResult(T){ .length = 1, .difference = 0, .start_index = 0 };
    if (arr.len == 2) return APResult(T){ .length = 2, .difference = arr[1] - arr[0], .start_index = 0 };

    const n = arr.len;
    
    var dp = try allocator.alloc(std.AutoHashMap(T, usize), n);
    defer {
        for (dp) |*map| {
            map.deinit();
        }
        allocator.free(dp);
    }
    
    for (dp) |*map| {
        map.* = std.AutoHashMap(T, usize).init(allocator);
    }
    
    var result = APResult(T){ .length = 2, .difference = arr[1] - arr[0], .start_index = 0 };
    
    for (1..n) |j| {
        for (0..j) |i| {
            const diff = arr[j] - arr[i];
            const prev_len = dp[i].get(diff) orelse 1;
            const new_len = prev_len + 1;
            
            try dp[j].put(diff, new_len);
            
            if (new_len > result.length) {
                result.length = new_len;
                result.difference = diff;
                result.start_index = i;
            }
        }
    }
    
    return result;
}

/// Find longest AP and return the actual sequence indices
///
/// Time: O(n²), Space: O(n²)
pub fn longestAPWithPath(comptime T: type, arr: []const T, allocator: Allocator) !APResult(T) {
    if (arr.len == 0) return error.EmptyArray;
    if (arr.len == 1) {
        const indices = try allocator.alloc(usize, 1);
        indices[0] = 0;
        return APResult(T){ .length = 1, .difference = 0, .start_index = 0, .indices = indices };
    }
    if (arr.len == 2) {
        const indices = try allocator.alloc(usize, 2);
        indices[0] = 0;
        indices[1] = 1;
        return APResult(T){ .length = 2, .difference = arr[1] - arr[0], .start_index = 0, .indices = indices };
    }

    const n = arr.len;

    // Define the entry type for clarity
    const DPEntry = struct { len: usize, prev: usize };

    // Store both length and previous index for path reconstruction
    var dp = try allocator.alloc(std.AutoHashMap(T, DPEntry), n);
    defer {
        for (dp) |*map| {
            map.deinit();
        }
        allocator.free(dp);
    }

    for (dp) |*map| {
        map.* = std.AutoHashMap(T, DPEntry).init(allocator);
    }
    
    var max_length: usize = 2;
    var best_end: usize = 1;
    var best_diff: T = arr[1] - arr[0];
    
    for (1..n) |j| {
        for (0..j) |i| {
            const diff = arr[j] - arr[i];

            const entry = dp[i].get(diff) orelse DPEntry{ .len = 1, .prev = i };
            const new_len = entry.len + 1;

            try dp[j].put(diff, DPEntry{ .len = new_len, .prev = i });

            if (new_len > max_length) {
                max_length = new_len;
                best_end = j;
                best_diff = diff;
            }
        }
    }
    
    // Reconstruct path
    var path = try std.ArrayList(usize).initCapacity(allocator, max_length);
    defer path.deinit(allocator);

    var current = best_end;
    path.appendAssumeCapacity(current);
    
    while (dp[current].get(best_diff)) |entry| {
        if (entry.len <= 1) break;
        current = entry.prev;
        path.appendAssumeCapacity(current);
    }
    
    // Reverse to get correct order
    std.mem.reverse(usize, path.items);

    const indices = try allocator.alloc(usize, path.items.len);
    @memcpy(indices, path.items);
    
    return APResult(T){ 
        .length = max_length, 
        .difference = best_diff, 
        .start_index = indices[0], 
        .indices = indices 
    };
}

/// Count number of arithmetic progressions of length k
///
/// Time: O(n² × k), Space: O(n²)
pub fn countAPsOfLength(comptime T: type, arr: []const T, k: usize, allocator: Allocator) !usize {
    if (k == 0 or arr.len < k) return 0;
    if (k == 1) return arr.len;
    if (k == 2) return (arr.len * (arr.len - 1)) / 2;

    const n = arr.len;

    // Use a simpler 3D DP approach: dp[len][i][diff] = count of APs of length len ending at i with difference diff
    // We'll iterate through lengths and keep only the previous length in memory

    var dp = try allocator.alloc(std.AutoHashMap(T, usize), n);
    defer {
        for (dp) |*map| {
            map.deinit();
        }
        allocator.free(dp);
    }

    for (dp) |*map| {
        map.* = std.AutoHashMap(T, usize).init(allocator);
    }

    // Initialize: All pairs form APs of length 2
    for (1..n) |j| {
        for (0..j) |i| {
            const diff = arr[j] - arr[i];
            const count = dp[j].get(diff) orelse 0;
            try dp[j].put(diff, count + 1);
        }
    }

    if (k == 2) {
        var total: usize = 0;
        for (dp) |*map| {
            var it = map.valueIterator();
            while (it.next()) |count| {
                total += count.*;
            }
        }
        return total;
    }

    // For each length from 3 to k
    var curr_len: usize = 3;
    while (curr_len <= k) : (curr_len += 1) {
        var new_dp = try allocator.alloc(std.AutoHashMap(T, usize), n);
        for (new_dp) |*map| {
            map.* = std.AutoHashMap(T, usize).init(allocator);
        }

        for (1..n) |j| {
            for (0..j) |i| {
                const diff = arr[j] - arr[i];
                // If there's an AP of length curr_len-1 ending at i with this diff,
                // we can extend it to j
                if (dp[i].get(diff)) |count| {
                    const new_count = new_dp[j].get(diff) orelse 0;
                    try new_dp[j].put(diff, new_count + count);
                }
            }
        }

        // Free old dp and move new_dp to dp
        for (dp) |*map| {
            map.deinit();
        }
        allocator.free(dp);
        dp = new_dp;
    }

    var total: usize = 0;
    for (dp) |*map| {
        var it = map.valueIterator();
        while (it.next()) |count| {
            total += count.*;
        }
    }

    return total;
}

// Tests
test "LAP: basic example" {
    const arr = [_]i32{ 1, 7, 10, 15, 27, 29 };
    const len = try longestArithmeticProgression(i32, &arr, testing.allocator);
    try testing.expectEqual(@as(usize, 3), len); // [1, 15, 29]
}

test "LAP: all equal elements" {
    const arr = [_]i32{ 5, 5, 5, 5, 5 };
    const len = try longestArithmeticProgression(i32, &arr, testing.allocator);
    try testing.expectEqual(@as(usize, 5), len); // all elements with diff 0
}

test "LAP: consecutive integers" {
    const arr = [_]i32{ 1, 2, 3, 4, 5 };
    const len = try longestArithmeticProgression(i32, &arr, testing.allocator);
    try testing.expectEqual(@as(usize, 5), len); // entire array with diff 1
}

test "LAP: no progression longer than 2" {
    const arr = [_]i32{ 1, 2, 4, 8, 16 }; // powers of 2
    const len = try longestArithmeticProgression(i32, &arr, testing.allocator);
    try testing.expectEqual(@as(usize, 2), len);
}

test "LAP: single element" {
    const arr = [_]i32{42};
    const len = try longestArithmeticProgression(i32, &arr, testing.allocator);
    try testing.expectEqual(@as(usize, 1), len);
}

test "LAP: two elements" {
    const arr = [_]i32{ 3, 7 };
    const len = try longestArithmeticProgression(i32, &arr, testing.allocator);
    try testing.expectEqual(@as(usize, 2), len);
}

test "LAP: empty array error" {
    const arr = [_]i32{};
    try testing.expectError(error.EmptyArray, longestArithmeticProgression(i32, &arr, testing.allocator));
}

test "LAP: with details" {
    const arr = [_]i32{ 3, 6, 9, 12 };
    var result = try longestAPWithDetails(i32, &arr, testing.allocator);
    defer result.deinit(testing.allocator);
    
    try testing.expectEqual(@as(usize, 4), result.length);
    try testing.expectEqual(@as(i32, 3), result.difference);
}

test "LAP: with details - multiple APs" {
    const arr = [_]i32{ 1, 7, 10, 13, 14, 19 };
    var result = try longestAPWithDetails(i32, &arr, testing.allocator);
    defer result.deinit(testing.allocator);
    
    // [1, 7, 13, 19] with diff 6, length 4
    try testing.expectEqual(@as(usize, 4), result.length);
    try testing.expectEqual(@as(i32, 6), result.difference);
}

test "LAP: path reconstruction" {
    const arr = [_]i32{ 3, 6, 9, 12 };
    var result = try longestAPWithPath(i32, &arr, testing.allocator);
    defer result.deinit(testing.allocator);
    
    try testing.expectEqual(@as(usize, 4), result.length);
    try testing.expectEqual(@as(i32, 3), result.difference);
    try testing.expect(result.indices != null);
    
    // Verify path: indices should be [0, 1, 2, 3]
    const indices = result.indices.?;
    try testing.expectEqual(@as(usize, 4), indices.len);
    try testing.expectEqual(@as(usize, 0), indices[0]);
    try testing.expectEqual(@as(usize, 1), indices[1]);
    try testing.expectEqual(@as(usize, 2), indices[2]);
    try testing.expectEqual(@as(usize, 3), indices[3]);
}

test "LAP: path reconstruction - complex" {
    const arr = [_]i32{ 1, 7, 10, 15, 27, 29 };
    var result = try longestAPWithPath(i32, &arr, testing.allocator);
    defer result.deinit(testing.allocator);
    
    try testing.expectEqual(@as(usize, 3), result.length);
    const indices = result.indices.?;
    try testing.expectEqual(@as(usize, 3), indices.len);
    
    // Verify it's an arithmetic progression
    const val1 = arr[indices[0]];
    const val2 = arr[indices[1]];
    const val3 = arr[indices[2]];
    try testing.expectEqual(val2 - val1, val3 - val2);
}

test "LAP: negative numbers" {
    const arr = [_]i32{ -5, -2, 1, 4, 7 };
    const len = try longestArithmeticProgression(i32, &arr, testing.allocator);
    try testing.expectEqual(@as(usize, 5), len); // entire array with diff 3
}

test "LAP: large array" {
    var arr: [100]i32 = undefined;
    for (&arr, 0..) |*val, i| {
        val.* = @intCast(i * 2); // 0, 2, 4, 6, ..., 198
    }
    const len = try longestArithmeticProgression(i32, &arr, testing.allocator);
    try testing.expectEqual(@as(usize, 100), len); // entire array with diff 2
}

test "LAP: count APs of length 3" {
    const arr = [_]i32{ 1, 2, 3, 4, 5 };
    const count = try countAPsOfLength(i32, &arr, 3, testing.allocator);
    // [1,2,3], [2,3,4], [3,4,5], [1,3,5]
    try testing.expectEqual(@as(usize, 4), count);
}

test "LAP: count APs of length 2" {
    const arr = [_]i32{ 1, 2, 3, 4 };
    const count = try countAPsOfLength(i32, &arr, 2, testing.allocator);
    // C(4,2) = 6 pairs
    try testing.expectEqual(@as(usize, 6), count);
}

test "LAP: count APs of length k > n" {
    const arr = [_]i32{ 1, 2, 3 };
    const count = try countAPsOfLength(i32, &arr, 5, testing.allocator);
    try testing.expectEqual(@as(usize, 0), count);
}

test "LAP: i64 support" {
    const arr = [_]i64{ 100, 200, 300, 400, 500 };
    const len = try longestArithmeticProgression(i64, &arr, testing.allocator);
    try testing.expectEqual(@as(usize, 5), len); // entire array with diff 100
}

test "LAP: memory safety" {
    const arr = [_]i32{ 1, 4, 7, 10, 13 };
    for (0..10) |_| {
        _ = try longestArithmeticProgression(i32, &arr, testing.allocator);
        
        var result = try longestAPWithDetails(i32, &arr, testing.allocator);
        result.deinit(testing.allocator);
        
        var path_result = try longestAPWithPath(i32, &arr, testing.allocator);
        path_result.deinit(testing.allocator);
    }
}
