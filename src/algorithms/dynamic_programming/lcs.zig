const std = @import("std");
const Allocator = std.mem.Allocator;

/// Longest Common Subsequence (LCS)
///
/// Provides algorithms for finding the longest common subsequence between two sequences.

/// Find LCS length using dynamic programming
/// Time: O(m*n) | Space: O(m*n)
pub fn length(comptime T: type, allocator: Allocator, a: []const T, b: []const T) !usize {
    if (a.len == 0 or b.len == 0) return 0;

    const m = a.len;
    const n = b.len;

    // Allocate DP table
    const dp = try allocator.alloc([]usize, m + 1);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }

    for (0..m + 1) |i| {
        dp[i] = try allocator.alloc(usize, n + 1);
        @memset(dp[i], 0);
    }

    for (1..m + 1) |i| {
        for (1..n + 1) |j| {
            if (a[i - 1] == b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = @max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }

    return dp[m][n];
}

/// Find LCS length with space optimization
/// Time: O(m*n) | Space: O(min(m,n))
pub fn lengthOptimized(comptime T: type, allocator: Allocator, a: []const T, b: []const T) !usize {
    if (a.len == 0 or b.len == 0) return 0;

    // Use shorter sequence for space optimization
    const shorter = if (a.len <= b.len) a else b;
    const longer = if (a.len <= b.len) b else a;

    const n = shorter.len;

    var prev = try allocator.alloc(usize, n + 1);
    defer allocator.free(prev);
    var curr = try allocator.alloc(usize, n + 1);
    defer allocator.free(curr);

    @memset(prev, 0);
    @memset(curr, 0);

    for (longer) |long_char| {
        for (shorter, 1..) |short_char, j| {
            if (long_char == short_char) {
                curr[j] = prev[j - 1] + 1;
            } else {
                curr[j] = @max(prev[j], curr[j - 1]);
            }
        }
        std.mem.swap([]usize, &prev, &curr);
        @memset(curr, 0);
    }

    return prev[n];
}

/// Find the actual LCS sequence
/// Time: O(m*n) | Space: O(m*n)
pub fn findSequence(comptime T: type, allocator: Allocator, a: []const T, b: []const T) ![]T {
    if (a.len == 0 or b.len == 0) return try allocator.alloc(T, 0);

    const m = a.len;
    const n = b.len;

    // Build DP table
    const dp = try allocator.alloc([]usize, m + 1);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }

    for (0..m + 1) |i| {
        dp[i] = try allocator.alloc(usize, n + 1);
        @memset(dp[i], 0);
    }

    for (1..m + 1) |i| {
        for (1..n + 1) |j| {
            if (a[i - 1] == b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = @max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }

    // Reconstruct LCS
    const lcs_len = dp[m][n];
    const result = try allocator.alloc(T, lcs_len);

    var i = m;
    var j = n;
    var idx = lcs_len;

    while (i > 0 and j > 0) {
        if (a[i - 1] == b[j - 1]) {
            idx -= 1;
            result[idx] = a[i - 1];
            i -= 1;
            j -= 1;
        } else if (dp[i - 1][j] > dp[i][j - 1]) {
            i -= 1;
        } else {
            j -= 1;
        }
    }

    return result;
}

/// Find all LCS sequences (there may be multiple)
/// Time: O(m*n + k) where k is number of LCS | Space: O(m*n)
pub fn findAllSequences(comptime T: type, allocator: Allocator, a: []const T, b: []const T) ![][]T {
    if (a.len == 0 or b.len == 0) {
        const result = try allocator.alloc([]T, 1);
        result[0] = try allocator.alloc(T, 0);
        return result;
    }

    const m = a.len;
    const n = b.len;

    // Build DP table
    const dp = try allocator.alloc([]usize, m + 1);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }

    for (0..m + 1) |i| {
        dp[i] = try allocator.alloc(usize, n + 1);
        @memset(dp[i], 0);
    }

    for (1..m + 1) |i| {
        for (1..n + 1) |j| {
            if (a[i - 1] == b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = @max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }

    // Reconstruct all LCS using backtracking
    var results: std.ArrayList([]T) = .{};
    defer results.deinit(allocator);

    const lcs_len = dp[m][n];
    const current = try allocator.alloc(T, lcs_len);
    defer allocator.free(current);

    try backtrackLCS(T, allocator, a, b, dp, m, n, current, lcs_len, &results);

    return results.toOwnedSlice(allocator);
}

fn backtrackLCS(
    comptime T: type,
    allocator: Allocator,
    a: []const T,
    b: []const T,
    dp: [][]usize,
    i: usize,
    j: usize,
    current: []T,
    idx: usize,
    results: *std.ArrayList([]T),
) !void {
    if (i == 0 or j == 0) {
        if (idx == 0) {
            const copy = try allocator.alloc(T, current.len);
            @memcpy(copy, current);
            try results.append(allocator, copy);
        }
        return;
    }

    if (a[i - 1] == b[j - 1]) {
        current[idx - 1] = a[i - 1];
        try backtrackLCS(T, allocator, a, b, dp, i - 1, j - 1, current, idx - 1, results);
    } else {
        const go_up = dp[i - 1][j];
        const go_left = dp[i][j - 1];

        if (go_up >= go_left) {
            try backtrackLCS(T, allocator, a, b, dp, i - 1, j, current, idx, results);
        }
        if (go_left >= go_up) {
            try backtrackLCS(T, allocator, a, b, dp, i, j - 1, current, idx, results);
        }
    }
}

/// Compute LCS for strings (convenience wrapper)
/// Time: O(m*n) | Space: O(m*n)
pub fn lengthString(allocator: Allocator, a: []const u8, b: []const u8) !usize {
    return length(u8, allocator, a, b);
}

/// Find LCS for strings
/// Time: O(m*n) | Space: O(m*n)
pub fn findSequenceString(allocator: Allocator, a: []const u8, b: []const u8) ![]u8 {
    return findSequence(u8, allocator, a, b);
}

// ============================================================================
// Tests
// ============================================================================

test "LCS: empty sequences" {
    const a: []const i32 = &[_]i32{};
    const b = [_]i32{ 1, 2, 3 };

    try std.testing.expectEqual(0, try length(i32, std.testing.allocator, a, &b));
    try std.testing.expectEqual(0, try length(i32, std.testing.allocator, &b, a));
    try std.testing.expectEqual(0, try lengthOptimized(i32, std.testing.allocator, a, &b));

    const seq = try findSequence(i32, std.testing.allocator, a, &b);
    defer std.testing.allocator.free(seq);
    try std.testing.expectEqual(0, seq.len);
}

test "LCS: identical sequences" {
    const items = [_]i32{ 1, 2, 3, 4, 5 };

    try std.testing.expectEqual(5, try length(i32, std.testing.allocator, &items, &items));
    try std.testing.expectEqual(5, try lengthOptimized(i32, std.testing.allocator, &items, &items));

    const seq = try findSequence(i32, std.testing.allocator, &items, &items);
    defer std.testing.allocator.free(seq);
    try std.testing.expectEqual(5, seq.len);
    try std.testing.expectEqualSlices(i32, &items, seq);
}

test "LCS: no common elements" {
    const a = [_]i32{ 1, 2, 3 };
    const b = [_]i32{ 4, 5, 6 };

    try std.testing.expectEqual(0, try length(i32, std.testing.allocator, &a, &b));
    try std.testing.expectEqual(0, try lengthOptimized(i32, std.testing.allocator, &a, &b));

    const seq = try findSequence(i32, std.testing.allocator, &a, &b);
    defer std.testing.allocator.free(seq);
    try std.testing.expectEqual(0, seq.len);
}

test "LCS: simple case" {
    const a = [_]i32{ 1, 2, 3, 4, 5 };
    const b = [_]i32{ 2, 4, 5 };

    try std.testing.expectEqual(3, try length(i32, std.testing.allocator, &a, &b));
    try std.testing.expectEqual(3, try lengthOptimized(i32, std.testing.allocator, &a, &b));

    const seq = try findSequence(i32, std.testing.allocator, &a, &b);
    defer std.testing.allocator.free(seq);
    try std.testing.expectEqual(3, seq.len);
    try std.testing.expectEqualSlices(i32, &b, seq);
}

test "LCS: classic example ABCDGH vs AEDFHR" {
    const a = "ABCDGH";
    const b = "AEDFHR";

    // LCS: ADH (length 3)
    try std.testing.expectEqual(3, try lengthString(std.testing.allocator, a, b));
    try std.testing.expectEqual(3, try lengthOptimized(u8, std.testing.allocator, a, b));

    const seq = try findSequenceString(std.testing.allocator, a, b);
    defer std.testing.allocator.free(seq);
    try std.testing.expectEqual(3, seq.len);
    try std.testing.expectEqualStrings("ADH", seq);
}

test "LCS: AGGTAB vs GXTXAYB" {
    const a = "AGGTAB";
    const b = "GXTXAYB";

    // LCS: GTAB (length 4)
    try std.testing.expectEqual(4, try lengthString(std.testing.allocator, a, b));
    try std.testing.expectEqual(4, try lengthOptimized(u8, std.testing.allocator, a, b));

    const seq = try findSequenceString(std.testing.allocator, a, b);
    defer std.testing.allocator.free(seq);
    try std.testing.expectEqual(4, seq.len);
    try std.testing.expectEqualStrings("GTAB", seq);
}

test "LCS: all common with different order" {
    const a = [_]i32{ 1, 2, 3, 4 };
    const b = [_]i32{ 4, 3, 2, 1 };

    // LCS: any single element (length 1)
    try std.testing.expectEqual(1, try length(i32, std.testing.allocator, &a, &b));
    try std.testing.expectEqual(1, try lengthOptimized(i32, std.testing.allocator, &a, &b));
}

test "LCS: multiple LCS" {
    const a = "ABCBDAB";
    const b = "BDCABA";

    // Multiple LCS: BCAB, BCBA, BDAB (all length 4)
    try std.testing.expectEqual(4, try lengthString(std.testing.allocator, a, b));

    const all_seqs = try findAllSequences(u8, std.testing.allocator, a, b);
    defer {
        for (all_seqs) |seq| std.testing.allocator.free(seq);
        std.testing.allocator.free(all_seqs);
    }

    try std.testing.expect(all_seqs.len >= 1);
    for (all_seqs) |seq| {
        try std.testing.expectEqual(4, seq.len);
    }
}

test "LCS: long sequences" {
    const allocator = std.testing.allocator;
    const n = 200;
    const a = try allocator.alloc(i32, n);
    defer allocator.free(a);
    const b = try allocator.alloc(i32, n);
    defer allocator.free(b);

    for (0..n) |i| {
        a[i] = @intCast(i);
        b[i] = @intCast(i);
    }

    // Identical sequences, LCS = n
    try std.testing.expectEqual(n, try length(i32, allocator, a, b));
    try std.testing.expectEqual(n, try lengthOptimized(i32, allocator, a, b));

    const seq = try findSequence(i32, allocator, a, b);
    defer allocator.free(seq);
    try std.testing.expectEqual(n, seq.len);
}

test "LCS: space optimization correctness" {
    const a = "ABCDEFGHIJ";
    const b = "ACEGI";

    const len1 = try lengthString(std.testing.allocator, a, b);
    const len2 = try lengthOptimized(u8, std.testing.allocator, a, b);

    try std.testing.expectEqual(len1, len2);
}

test "LCS: with duplicates" {
    const a = [_]i32{ 1, 2, 2, 3, 4 };
    const b = [_]i32{ 2, 2, 3, 3, 4 };

    // LCS: 2, 2, 3, 4 (length 4)
    try std.testing.expectEqual(4, try length(i32, std.testing.allocator, &a, &b));
    try std.testing.expectEqual(4, try lengthOptimized(i32, std.testing.allocator, &a, &b));

    const seq = try findSequence(i32, std.testing.allocator, &a, &b);
    defer std.testing.allocator.free(seq);
    try std.testing.expectEqual(4, seq.len);
}

test "LCS: different lengths" {
    const a = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const b = [_]i32{ 2, 4, 6, 8 };

    try std.testing.expectEqual(4, try length(i32, std.testing.allocator, &a, &b));
    try std.testing.expectEqual(4, try lengthOptimized(i32, std.testing.allocator, &a, &b));

    const seq = try findSequence(i32, std.testing.allocator, &a, &b);
    defer std.testing.allocator.free(seq);
    try std.testing.expectEqual(4, seq.len);
    try std.testing.expectEqualSlices(i32, &b, seq);
}

test "LCS: random sequences" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const n = 50;
    const a = try allocator.alloc(i32, n);
    defer allocator.free(a);
    const b = try allocator.alloc(i32, n);
    defer allocator.free(b);

    for (0..n) |i| {
        a[i] = random.intRangeAtMost(i32, 0, 20);
        b[i] = random.intRangeAtMost(i32, 0, 20);
    }

    const len1 = try length(i32, allocator, a, b);
    const len2 = try lengthOptimized(i32, allocator, a, b);

    // Both algorithms should give the same result
    try std.testing.expectEqual(len1, len2);

    const seq = try findSequence(i32, allocator, a, b);
    defer allocator.free(seq);
    try std.testing.expectEqual(len1, seq.len);
}
