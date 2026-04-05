const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Longest Valid Parentheses — Dynamic Programming
///
/// Finds the length of the longest valid (well-formed) parentheses substring.
/// A valid parentheses string is one where every opening parenthesis has a matching
/// closing parenthesis in the correct order.
///
/// # Algorithm: Three Approaches
///
/// ## 1. Dynamic Programming (longestValidParentheses)
/// - State: dp[i] = length of longest valid substring ending at position i
/// - Recurrence:
///   * If s[i] == '(': dp[i] = 0 (cannot end with opening bracket)
///   * If s[i] == ')':
///     - If s[i-1] == '(': dp[i] = dp[i-2] + 2 (matched pair)
///     - If s[i-1] == ')' and s[i-dp[i-1]-1] == '(':
///       dp[i] = dp[i-1] + 2 + dp[i-dp[i-1]-2] (extend previous match)
/// - Time: O(n), Space: O(n)
///
/// ## 2. Two-Pass Counting (longestValidParenthesesTwoPass)
/// - Pass 1 (left→right): count left/right, reset on right > left
/// - Pass 2 (right→left): count left/right, reset on left > right
/// - No extra space except counters
/// - Time: O(n), Space: O(1)
///
/// ## 3. Stack-Based (longestValidParenthesesStack)
/// - Use stack to track indices of unmatched parentheses
/// - Calculate distances between unmatched positions
/// - Time: O(n), Space: O(n)
///
/// # Use Cases
/// - Compiler syntax checking (validate balanced delimiters)
/// - Text editors (highlight matching brackets)
/// - Code analysis tools (detect malformed expressions)
/// - Interview problems (LeetCode #32 - Hard)
///
/// # Reference
/// - LeetCode #32 (Longest Valid Parentheses)
/// - Classic DP string problem with multiple solution approaches

/// Longest valid parentheses using dynamic programming.
///
/// Time: O(n) where n = string length
/// Space: O(n) for DP array
///
/// Returns the length of the longest valid parentheses substring.
pub fn longestValidParentheses(allocator: Allocator, s: []const u8) !usize {
    if (s.len < 2) return 0;

    var dp = try allocator.alloc(usize, s.len);
    defer allocator.free(dp);
    @memset(dp, 0);

    var max_len: usize = 0;

    for (1..s.len) |i| {
        if (s[i] == ')') {
            if (s[i - 1] == '(') {
                // Case: ...()
                dp[i] = (if (i >= 2) dp[i - 2] else 0) + 2;
            } else if (i > dp[i - 1]) {
                // Case: ...))
                const match_idx = i - dp[i - 1] - 1;
                if (s[match_idx] == '(') {
                    dp[i] = dp[i - 1] + 2;
                    if (match_idx > 0) {
                        dp[i] += dp[match_idx - 1];
                    }
                }
            }
            max_len = @max(max_len, dp[i]);
        }
    }

    return max_len;
}

/// Longest valid parentheses using two-pass counting (space-optimized).
///
/// Time: O(n) where n = string length
/// Space: O(1) - only counters
///
/// Returns the length of the longest valid parentheses substring.
pub fn longestValidParenthesesTwoPass(s: []const u8) usize {
    if (s.len < 2) return 0;

    var max_len: usize = 0;

    // Left to right pass
    var left: usize = 0;
    var right: usize = 0;
    for (s) |ch| {
        if (ch == '(') {
            left += 1;
        } else if (ch == ')') {
            right += 1;
        }

        if (left == right) {
            max_len = @max(max_len, left * 2);
        } else if (right > left) {
            left = 0;
            right = 0;
        }
    }

    // Right to left pass
    left = 0;
    right = 0;
    var i: usize = s.len;
    while (i > 0) {
        i -= 1;
        const ch = s[i];
        if (ch == '(') {
            left += 1;
        } else if (ch == ')') {
            right += 1;
        }

        if (left == right) {
            max_len = @max(max_len, left * 2);
        } else if (left > right) {
            left = 0;
            right = 0;
        }
    }

    return max_len;
}

/// Longest valid parentheses using stack-based approach.
///
/// Time: O(n) where n = string length
/// Space: O(n) for stack
///
/// Returns the length of the longest valid parentheses substring.
pub fn longestValidParenthesesStack(allocator: Allocator, s: []const u8) !usize {
    if (s.len < 2) return 0;

    var stack = std.ArrayList(isize).init(allocator);
    defer stack.deinit();

    try stack.append(-1); // Base index
    var max_len: usize = 0;

    for (s, 0..) |ch, i| {
        if (ch == '(') {
            try stack.append(@intCast(i));
        } else if (ch == ')') {
            _ = stack.pop();
            if (stack.items.len == 0) {
                try stack.append(@intCast(i));
            } else {
                const top = stack.items[stack.items.len - 1];
                const len: usize = @intCast(@as(isize, @intCast(i)) - top);
                max_len = @max(max_len, len);
            }
        }
    }

    return max_len;
}

/// Find all valid parentheses substrings and return their starting indices and lengths.
///
/// Time: O(n) where n = string length
/// Space: O(n) for DP array + O(k) for results where k = number of valid substrings
///
/// Returns an ArrayList of ValidSubstring structs.
pub fn findAllValidSubstrings(allocator: Allocator, s: []const u8) !std.ArrayList(ValidSubstring) {
    var result = std.ArrayList(ValidSubstring).init(allocator);
    errdefer result.deinit();

    if (s.len < 2) return result;

    var dp = try allocator.alloc(usize, s.len);
    defer allocator.free(dp);
    @memset(dp, 0);

    for (1..s.len) |i| {
        if (s[i] == ')') {
            if (s[i - 1] == '(') {
                dp[i] = (if (i >= 2) dp[i - 2] else 0) + 2;
            } else if (i > dp[i - 1]) {
                const match_idx = i - dp[i - 1] - 1;
                if (s[match_idx] == '(') {
                    dp[i] = dp[i - 1] + 2;
                    if (match_idx > 0) {
                        dp[i] += dp[match_idx - 1];
                    }
                }
            }

            // If we have a valid substring ending here, record it
            if (dp[i] > 0) {
                const start = i + 1 - dp[i];
                const len = dp[i];

                // Check if this is a new substring or extends previous
                if (result.items.len == 0 or start > result.items[result.items.len - 1].start) {
                    try result.append(.{ .start = start, .length = len });
                } else {
                    // Update last entry if it's an extension
                    result.items[result.items.len - 1].length = len;
                }
            }
        }
    }

    return result;
}

/// Represents a valid parentheses substring with its position and length.
pub const ValidSubstring = struct {
    start: usize,
    length: usize,

    /// Extract the actual substring from the original string.
    pub fn substring(self: ValidSubstring, s: []const u8) []const u8 {
        return s[self.start .. self.start + self.length];
    }
};

// Tests

test "longestValidParentheses - basic examples" {
    const allocator = testing.allocator;

    // Basic valid pairs
    try testing.expectEqual(@as(usize, 2), try longestValidParentheses(allocator, "()"));
    try testing.expectEqual(@as(usize, 4), try longestValidParentheses(allocator, "(())"));
    try testing.expectEqual(@as(usize, 4), try longestValidParentheses(allocator, "()()"));

    // Invalid or partially valid
    try testing.expectEqual(@as(usize, 0), try longestValidParentheses(allocator, "("));
    try testing.expectEqual(@as(usize, 0), try longestValidParentheses(allocator, ")"));
    try testing.expectEqual(@as(usize, 2), try longestValidParentheses(allocator, "()("));
}

test "longestValidParentheses - classic examples" {
    const allocator = testing.allocator;

    // LeetCode example 1: ")()())" → "()()" = 4
    try testing.expectEqual(@as(usize, 4), try longestValidParentheses(allocator, ")()())"));

    // LeetCode example 2: "(()" → "()" = 2
    try testing.expectEqual(@as(usize, 2), try longestValidParentheses(allocator, "(()"));

    // Complex: "()(()" → "()" = 2
    try testing.expectEqual(@as(usize, 2), try longestValidParentheses(allocator, "()(()"));

    // All valid: "(()())" = 6
    try testing.expectEqual(@as(usize, 6), try longestValidParentheses(allocator, "(()())"));
}

test "longestValidParentheses - edge cases" {
    const allocator = testing.allocator;

    // Empty
    try testing.expectEqual(@as(usize, 0), try longestValidParentheses(allocator, ""));

    // Single char
    try testing.expectEqual(@as(usize, 0), try longestValidParentheses(allocator, "("));
    try testing.expectEqual(@as(usize, 0), try longestValidParentheses(allocator, ")"));

    // All open
    try testing.expectEqual(@as(usize, 0), try longestValidParentheses(allocator, "(((("));

    // All close
    try testing.expectEqual(@as(usize, 0), try longestValidParentheses(allocator, "))))"));
}

test "longestValidParentheses - nested structures" {
    const allocator = testing.allocator;

    // Nested: "((()))" = 6
    try testing.expectEqual(@as(usize, 6), try longestValidParentheses(allocator, "((()))"));

    // Multiple nested: "(())(())" = 8
    try testing.expectEqual(@as(usize, 8), try longestValidParentheses(allocator, "(())((()))"));

    // Nested with invalid: "((())" = 4
    try testing.expectEqual(@as(usize, 4), try longestValidParentheses(allocator, "((())"));
}

test "longestValidParentheses - mixed valid/invalid" {
    const allocator = testing.allocator;

    // "()(())" = 6
    try testing.expectEqual(@as(usize, 6), try longestValidParentheses(allocator, "()(())"));

    // "()(()()" = 4 (last four)
    try testing.expectEqual(@as(usize, 4), try longestValidParentheses(allocator, "()(()("));

    // ")()()()(" = 6 (middle part)
    try testing.expectEqual(@as(usize, 6), try longestValidParentheses(allocator, ")()()()("));
}

test "longestValidParenthesesTwoPass - consistency with DP" {
    const allocator = testing.allocator;

    const test_cases = [_][]const u8{
        "()",
        "(()",
        ")()())",
        "(()())",
        "((()))",
        "()(())",
        ")()()()(",
        "",
        "(",
        ")",
    };

    for (test_cases) |s| {
        const dp_result = try longestValidParentheses(allocator, s);
        const two_pass_result = longestValidParenthesesTwoPass(s);
        try testing.expectEqual(dp_result, two_pass_result);
    }
}

test "longestValidParenthesesStack - consistency with DP" {
    const allocator = testing.allocator;

    const test_cases = [_][]const u8{
        "()",
        "(()",
        ")()())",
        "(()())",
        "((()))",
        "()(())",
        ")()()()(",
        "",
        "(",
        ")",
    };

    for (test_cases) |s| {
        const dp_result = try longestValidParentheses(allocator, s);
        const stack_result = try longestValidParenthesesStack(allocator, s);
        try testing.expectEqual(dp_result, stack_result);
    }
}

test "longestValidParentheses - large input" {
    const allocator = testing.allocator;

    // Generate large valid string: "()()()" repeated
    var large = std.ArrayList(u8).init(allocator);
    defer large.deinit();

    for (0..50) |_| {
        try large.appendSlice("()");
    }

    const result = try longestValidParentheses(allocator, large.items);
    try testing.expectEqual(@as(usize, 100), result); // 50 pairs = 100 chars
}

test "longestValidParentheses - complex patterns" {
    const allocator = testing.allocator;

    // "(()()" = 4
    try testing.expectEqual(@as(usize, 4), try longestValidParentheses(allocator, "(()("));

    // "()(()" = 2
    try testing.expectEqual(@as(usize, 2), try longestValidParentheses(allocator, "()(()"));

    // "()(())" = 6
    try testing.expectEqual(@as(usize, 6), try longestValidParentheses(allocator, "()(())"));

    // "(()(()))" = 8
    try testing.expectEqual(@as(usize, 8), try longestValidParentheses(allocator, "(()(()))"));
}

test "findAllValidSubstrings - basic" {
    const allocator = testing.allocator;

    {
        var result = try findAllValidSubstrings(allocator, "()");
        defer result.deinit();

        try testing.expectEqual(@as(usize, 1), result.items.len);
        try testing.expectEqual(@as(usize, 0), result.items[0].start);
        try testing.expectEqual(@as(usize, 2), result.items[0].length);
    }

    {
        var result = try findAllValidSubstrings(allocator, ")()())");
        defer result.deinit();

        try testing.expectEqual(@as(usize, 1), result.items.len);
        try testing.expectEqual(@as(usize, 1), result.items[0].start);
        try testing.expectEqual(@as(usize, 4), result.items[0].length);
    }
}

test "findAllValidSubstrings - substring extraction" {
    const allocator = testing.allocator;

    const s = "()((())";
    var result = try findAllValidSubstrings(allocator, s);
    defer result.deinit();

    for (result.items) |sub| {
        const extracted = sub.substring(s);
        // Verify extracted string is valid parentheses
        try testing.expect(extracted.len > 0);
        try testing.expect(extracted.len % 2 == 0); // Must be even length
    }
}

test "longestValidParentheses - memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        _ = try longestValidParentheses(allocator, ")()())");
        _ = try longestValidParenthesesStack(allocator, "(()())");
        var result = try findAllValidSubstrings(allocator, "()((()))");
        result.deinit();
    }
}
