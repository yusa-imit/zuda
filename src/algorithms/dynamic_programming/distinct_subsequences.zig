const std = @import("std");
const Allocator = std.mem.Allocator;

/// Distinct Subsequences — Count number of distinct subsequences of s that equal t
///
/// Problem: Given strings s and t, count how many distinct subsequences of s match t.
/// A subsequence is a sequence derived by deleting some (or none) characters without
/// changing the order of the remaining characters.
///
/// Example: s = "rabbbit", t = "rabbit"
/// Result: 3 (rabb[b]it, rab[b]bit, ra[b]bbit)
///
/// Algorithm:
/// - Dynamic Programming with 2D table: dp[i][j] = number of ways to match t[0..j) using s[0..i)
/// - Recurrence relation:
///   * If s[i-1] == t[j-1]: dp[i][j] = dp[i-1][j-1] (use s[i-1]) + dp[i-1][j] (skip s[i-1])
///   * Else: dp[i][j] = dp[i-1][j] (can only skip s[i-1])
/// - Base cases:
///   * dp[i][0] = 1 for all i (empty t matches in one way — select nothing)
///   * dp[0][j] = 0 for j > 0 (can't match non-empty t with empty s)
///
/// Time complexity: O(n×m) where n = len(s), m = len(t)
/// Space complexity: O(n×m) for full table, O(m) for optimized rolling buffer
///
/// Use cases:
/// - String matching analysis (how many ways to form pattern)
/// - Text analysis (subsequence counting)
/// - DNA sequence analysis (count matching patterns)
/// - Pattern recognition in sequential data

/// Count distinct subsequences of s that match t using full DP table
///
/// Time: O(n×m) where n = len(s), m = len(t)
/// Space: O(n×m) for DP table
pub fn countSubsequences(allocator: Allocator, s: []const u8, t: []const u8) !u64 {
    const n = s.len;
    const m = t.len;

    // Base case: empty pattern matches in one way
    if (m == 0) return 1;
    // Empty source can't match non-empty pattern
    if (n == 0) return 0;
    // Pattern longer than source — impossible
    if (m > n) return 0;

    // DP table: dp[i][j] = ways to match t[0..j) using s[0..i)
    var dp = try allocator.alloc([]u64, n + 1);
    defer allocator.free(dp);

    for (0..n + 1) |i| {
        dp[i] = try allocator.alloc(u64, m + 1);
    }
    defer {
        for (0..n + 1) |i| {
            allocator.free(dp[i]);
        }
    }

    // Initialize: empty pattern matches in one way
    for (0..n + 1) |i| {
        dp[i][0] = 1;
    }

    // Empty source can't match non-empty pattern
    for (1..m + 1) |j| {
        dp[0][j] = 0;
    }

    // Fill DP table
    for (1..n + 1) |i| {
        for (1..m + 1) |j| {
            // Can always skip current character in s
            dp[i][j] = dp[i - 1][j];

            // If characters match, can also use current character
            if (s[i - 1] == t[j - 1]) {
                dp[i][j] += dp[i - 1][j - 1];
            }
        }
    }

    return dp[n][m];
}

/// Count distinct subsequences using space-optimized rolling buffer
///
/// Only keeps previous row, reducing space from O(n×m) to O(m).
/// Must iterate in reverse order to avoid overwriting needed values.
///
/// Time: O(n×m)
/// Space: O(m)
pub fn countSubsequencesOptimized(allocator: Allocator, s: []const u8, t: []const u8) !u64 {
    const n = s.len;
    const m = t.len;

    if (m == 0) return 1;
    if (n == 0) return 0;
    if (m > n) return 0;

    // Only need one row: dp[j] = ways to match t[0..j) so far
    var dp = try allocator.alloc(u64, m + 1);
    defer allocator.free(dp);

    // Base case: empty pattern
    dp[0] = 1;
    for (1..m + 1) |j| {
        dp[j] = 0;
    }

    // Process each character in s
    for (0..n) |i| {
        // Must iterate backwards to avoid overwriting values we need
        var j: usize = m;
        while (j > 0) : (j -= 1) {
            if (s[i] == t[j - 1]) {
                dp[j] += dp[j - 1];
            }
        }
    }

    return dp[m];
}

/// Find all distinct subsequences (for demonstration/debugging)
///
/// Returns list of all distinct ways to select indices from s to form t.
/// Each result is a list of indices in s.
///
/// Time: O(n×m + k) where k = number of distinct subsequences
/// Space: O(n×m + k×m)
pub fn allSubsequences(allocator: Allocator, s: []const u8, t: []const u8) !std.ArrayList(std.ArrayList(usize)) {
    const n = s.len;
    const m = t.len;

    var result = std.ArrayList(std.ArrayList(usize)).init(allocator);
    errdefer {
        for (result.items) |seq| {
            seq.deinit();
        }
        result.deinit();
    }

    if (m == 0) {
        // Empty pattern matches with empty selection
        try result.append(std.ArrayList(usize).init(allocator));
        return result;
    }

    if (n == 0 or m > n) {
        return result; // No matches possible
    }

    // Backtrack through DP table to find all solutions
    // Store state as: (i, j, current_path)
    const State = struct {
        i: usize,
        j: usize,
        path: std.ArrayList(usize),
    };

    var stack = std.ArrayList(State).init(allocator);
    defer {
        for (stack.items) |*state| {
            state.path.deinit();
        }
        stack.deinit();
    }

    // Start from end
    try stack.append(.{
        .i = n,
        .j = m,
        .path = std.ArrayList(usize).init(allocator),
    });

    while (stack.items.len > 0) {
        var state = stack.pop();

        // Reached start — found a complete match
        if (state.j == 0) {
            // Reverse path (we built it backwards)
            std.mem.reverse(usize, state.path.items);
            try result.append(state.path);
            continue;
        }

        // Can't continue
        if (state.i == 0) {
            state.path.deinit();
            continue;
        }

        // Try skipping s[i-1]
        const skip_state = State{
            .i = state.i - 1,
            .j = state.j,
            .path = try state.path.clone(),
        };
        try stack.append(skip_state);

        // Try using s[i-1] if it matches t[j-1]
        if (s[state.i - 1] == t[state.j - 1]) {
            try state.path.append(state.i - 1);
            state.i -= 1;
            state.j -= 1;
            try stack.append(state);
        } else {
            state.path.deinit();
        }
    }

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "distinct subsequences - basic example" {
    const allocator = std.testing.allocator;

    // "rabbbit" has 3 distinct subsequences matching "rabbit"
    // rabb[b]it, rab[b]bit, ra[b]bbit
    const count = try countSubsequences(allocator, "rabbbit", "rabbit");
    try std.testing.expectEqual(@as(u64, 3), count);
}

test "distinct subsequences - empty pattern" {
    const allocator = std.testing.allocator;

    // Empty pattern matches any string in exactly one way
    const count = try countSubsequences(allocator, "hello", "");
    try std.testing.expectEqual(@as(u64, 1), count);
}

test "distinct subsequences - empty source" {
    const allocator = std.testing.allocator;

    // Empty source can't match non-empty pattern
    const count = try countSubsequences(allocator, "", "a");
    try std.testing.expectEqual(@as(u64, 0), count);
}

test "distinct subsequences - no match" {
    const allocator = std.testing.allocator;

    // No way to form "xyz" from "abc"
    const count = try countSubsequences(allocator, "abc", "xyz");
    try std.testing.expectEqual(@as(u64, 0), count);
}

test "distinct subsequences - single character" {
    const allocator = std.testing.allocator;

    // "aaa" has 3 ways to form "a"
    const count = try countSubsequences(allocator, "aaa", "a");
    try std.testing.expectEqual(@as(u64, 3), count);
}

test "distinct subsequences - full match" {
    const allocator = std.testing.allocator;

    // Exact match has exactly one way
    const count = try countSubsequences(allocator, "abc", "abc");
    try std.testing.expectEqual(@as(u64, 1), count);
}

test "distinct subsequences - pattern longer than source" {
    const allocator = std.testing.allocator;

    // Pattern longer than source — impossible
    const count = try countSubsequences(allocator, "ab", "abc");
    try std.testing.expectEqual(@as(u64, 0), count);
}

test "distinct subsequences - repeated characters" {
    const allocator = std.testing.allocator;

    // "babgbag" has 5 ways to form "bag"
    // b[a]bg[b]ag, b[a]bgb[a]g, b[a]bgbag, ba[b]g[b]ag, ba[b]gbag
    const count = try countSubsequences(allocator, "babgbag", "bag");
    try std.testing.expectEqual(@as(u64, 5), count);
}

test "distinct subsequences - large example" {
    const allocator = std.testing.allocator;

    // Larger string with multiple matches
    const s = "aabbbcccc";
    const t = "abc";
    // a's: 2, b's: 3, c's: 4 → 2 × 3 × 4 = 24 ways
    const count = try countSubsequences(allocator, s, t);
    try std.testing.expectEqual(@as(u64, 24), count);
}

test "distinct subsequences - optimized variant" {
    const allocator = std.testing.allocator;

    // Test space-optimized version
    const count = try countSubsequencesOptimized(allocator, "rabbbit", "rabbit");
    try std.testing.expectEqual(@as(u64, 3), count);
}

test "distinct subsequences - optimized empty cases" {
    const allocator = std.testing.allocator;

    try std.testing.expectEqual(@as(u64, 1), try countSubsequencesOptimized(allocator, "hello", ""));
    try std.testing.expectEqual(@as(u64, 0), try countSubsequencesOptimized(allocator, "", "a"));
    try std.testing.expectEqual(@as(u64, 0), try countSubsequencesOptimized(allocator, "ab", "abc"));
}

test "distinct subsequences - optimized large" {
    const allocator = std.testing.allocator;

    const s = "aabbbcccc";
    const t = "abc";
    const count = try countSubsequencesOptimized(allocator, s, t);
    try std.testing.expectEqual(@as(u64, 24), count);
}

test "distinct subsequences - all subsequences basic" {
    const allocator = std.testing.allocator;

    var sequences = try allSubsequences(allocator, "abc", "ac");
    defer {
        for (sequences.items) |seq| {
            seq.deinit();
        }
        sequences.deinit();
    }

    // Should find exactly 1 way: indices [0, 2]
    try std.testing.expectEqual(@as(usize, 1), sequences.items.len);
    try std.testing.expectEqual(@as(usize, 2), sequences.items[0].items.len);
    try std.testing.expectEqual(@as(usize, 0), sequences.items[0].items[0]);
    try std.testing.expectEqual(@as(usize, 2), sequences.items[0].items[1]);
}

test "distinct subsequences - all subsequences empty" {
    const allocator = std.testing.allocator;

    var sequences = try allSubsequences(allocator, "abc", "");
    defer {
        for (sequences.items) |seq| {
            seq.deinit();
        }
        sequences.deinit();
    }

    // Empty pattern matches with empty selection
    try std.testing.expectEqual(@as(usize, 1), sequences.items.len);
    try std.testing.expectEqual(@as(usize, 0), sequences.items[0].items.len);
}

test "distinct subsequences - all subsequences no match" {
    const allocator = std.testing.allocator;

    var sequences = try allSubsequences(allocator, "abc", "xyz");
    defer {
        for (sequences.items) |seq| {
            seq.deinit();
        }
        sequences.deinit();
    }

    // No matches
    try std.testing.expectEqual(@as(usize, 0), sequences.items.len);
}

test "distinct subsequences - consistency check" {
    const allocator = std.testing.allocator;

    // Verify both methods produce same count
    const test_cases = [_]struct { s: []const u8, t: []const u8 }{
        .{ .s = "rabbbit", .t = "rabbit" },
        .{ .s = "babgbag", .t = "bag" },
        .{ .s = "aabbbcccc", .t = "abc" },
        .{ .s = "abc", .t = "abc" },
        .{ .s = "hello", .t = "hlo" },
    };

    for (test_cases) |tc| {
        const count1 = try countSubsequences(allocator, tc.s, tc.t);
        const count2 = try countSubsequencesOptimized(allocator, tc.s, tc.t);
        try std.testing.expectEqual(count1, count2);
    }
}

test "distinct subsequences - memory safety" {
    const allocator = std.testing.allocator;

    // Test multiple allocations and deallocations
    for (0..10) |_| {
        _ = try countSubsequences(allocator, "rabbbit", "rabbit");
        _ = try countSubsequencesOptimized(allocator, "babgbag", "bag");

        var sequences = try allSubsequences(allocator, "abc", "ac");
        for (sequences.items) |seq| {
            seq.deinit();
        }
        sequences.deinit();
    }
}
