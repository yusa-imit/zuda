const std = @import("std");
const Allocator = std.mem.Allocator;

/// Boolean Parenthesization Problem
///
/// Given a boolean expression with symbols (T/F) and operators (&, |, ^),
/// count the number of ways to parenthesize it to evaluate to True or False.
///
/// Applications: Expression evaluation, compiler optimization, Boolean algebra.

/// Count Ways to Parenthesize: Count ways to get True/False result
/// Time: O(n³) | Space: O(n²) where n = number of symbols
///
/// Returns number of ways to parenthesize expression to get True.
/// Uses bottom-up DP with memoization for both True and False counts.
///
/// Expression format: Alternating symbols (T/F) and operators (&/|/^)
/// - Symbols: 'T' (true), 'F' (false)
/// - Operators: '&' (AND), '|' (OR), '^' (XOR)
///
/// Example:
/// ```zig
/// const ways = try countWaysToTrue("T|F&T", allocator); // 2 ways
/// // (T|(F&T)) = T|F = T ✓
/// // ((T|F)&T) = T&T = T ✓
/// ```
pub fn countWaysToTrue(expr: []const u8, allocator: Allocator) !usize {
    const result = try countWays(expr, allocator);
    return result.true_count;
}

/// Count Ways Result: Both True and False counts
pub const CountResult = struct {
    true_count: usize,
    false_count: usize,
};

/// Count Ways to Parenthesize (both True and False)
/// Time: O(n³) | Space: O(n²)
///
/// Returns counts for both True and False results.
///
/// Example:
/// ```zig
/// const result = try countWays("T^F", allocator);
/// // result.true_count = 1: (T^F) = T
/// // result.false_count = 0
/// ```
pub fn countWays(expr: []const u8, allocator: Allocator) !CountResult {
    if (expr.len == 0) return error.EmptyExpression;
    if (expr.len % 2 == 0) return error.InvalidExpression; // Must be odd length (alternating)

    // Validate expression format
    for (expr, 0..) |c, i| {
        if (i % 2 == 0) {
            // Even positions: symbols
            if (c != 'T' and c != 'F') return error.InvalidSymbol;
        } else {
            // Odd positions: operators
            if (c != '&' and c != '|' and c != '^') return error.InvalidOperator;
        }
    }

    const n = (expr.len + 1) / 2; // Number of symbols

    // Extract symbols and operators
    const symbols = try allocator.alloc(bool, n);
    defer allocator.free(symbols);
    const operators = try allocator.alloc(u8, n - 1);
    defer allocator.free(operators);

    for (0..n) |i| {
        symbols[i] = (expr[i * 2] == 'T');
    }
    for (0..n - 1) |i| {
        operators[i] = expr[i * 2 + 1];
    }

    // dp_true[i][j] = ways to get true from symbols[i..j+1]
    // dp_false[i][j] = ways to get false from symbols[i..j+1]
    const dp_true = try allocator.alloc([]usize, n);
    defer allocator.free(dp_true);
    const dp_false = try allocator.alloc([]usize, n);
    defer allocator.free(dp_false);

    for (0..n) |i| {
        dp_true[i] = try allocator.alloc(usize, n);
        dp_false[i] = try allocator.alloc(usize, n);
        @memset(dp_true[i], 0);
        @memset(dp_false[i], 0);
    }
    defer {
        for (0..n) |i| {
            allocator.free(dp_true[i]);
            allocator.free(dp_false[i]);
        }
    }

    // Base case: single symbols
    for (0..n) |i| {
        if (symbols[i]) {
            dp_true[i][i] = 1;
            dp_false[i][i] = 0;
        } else {
            dp_true[i][i] = 0;
            dp_false[i][i] = 1;
        }
    }

    // Fill DP table for subexpressions of increasing length
    for (1..n) |length| {
        var i: usize = 0;
        while (i + length < n) : (i += 1) {
            const j = i + length;

            // Try all split points
            for (i..j) |k| {
                const op = operators[k];

                const left_true = dp_true[i][k];
                const left_false = dp_false[i][k];
                const right_true = dp_true[k + 1][j];
                const right_false = dp_false[k + 1][j];

                if (op == '&') {
                    // AND: T & T = T, all else = F
                    dp_true[i][j] += left_true * right_true;
                    dp_false[i][j] += left_true * right_false + left_false * right_true + left_false * right_false;
                } else if (op == '|') {
                    // OR: F | F = F, all else = T
                    dp_true[i][j] += left_true * right_true + left_true * right_false + left_false * right_true;
                    dp_false[i][j] += left_false * right_false;
                } else if (op == '^') {
                    // XOR: same = F, different = T
                    dp_true[i][j] += left_true * right_false + left_false * right_true;
                    dp_false[i][j] += left_true * right_true + left_false * right_false;
                }
            }
        }
    }

    return CountResult{
        .true_count = dp_true[0][n - 1],
        .false_count = dp_false[0][n - 1],
    };
}

/// Count Ways to False: Count ways to get False result
/// Time: O(n³) | Space: O(n²)
///
/// Example:
/// ```zig
/// const ways = try countWaysToFalse("T&F", allocator); // 1 way
/// ```
pub fn countWaysToFalse(expr: []const u8, allocator: Allocator) !usize {
    const result = try countWays(expr, allocator);
    return result.false_count;
}

/// Total Parenthesizations: Count all possible ways (True + False)
/// Time: O(n³) | Space: O(n²)
///
/// Example:
/// ```zig
/// const total = try totalParenthesizations("T|F", allocator); // 1 way total
/// ```
pub fn totalParenthesizations(expr: []const u8, allocator: Allocator) !usize {
    const result = try countWays(expr, allocator);
    return result.true_count + result.false_count;
}

// ============================================================================
// Tests
// ============================================================================

test "boolean parenthesization: single symbol true" {
    const allocator = std.testing.allocator;
    const result = try countWays("T", allocator);
    try std.testing.expectEqual(@as(usize, 1), result.true_count);
    try std.testing.expectEqual(@as(usize, 0), result.false_count);
}

test "boolean parenthesization: single symbol false" {
    const allocator = std.testing.allocator;
    const result = try countWays("F", allocator);
    try std.testing.expectEqual(@as(usize, 0), result.true_count);
    try std.testing.expectEqual(@as(usize, 1), result.false_count);
}

test "boolean parenthesization: T&F" {
    const allocator = std.testing.allocator;
    const result = try countWays("T&F", allocator);
    try std.testing.expectEqual(@as(usize, 0), result.true_count);
    try std.testing.expectEqual(@as(usize, 1), result.false_count);
}

test "boolean parenthesization: T|F" {
    const allocator = std.testing.allocator;
    const result = try countWays("T|F", allocator);
    try std.testing.expectEqual(@as(usize, 1), result.true_count);
    try std.testing.expectEqual(@as(usize, 0), result.false_count);
}

test "boolean parenthesization: T^F (XOR)" {
    const allocator = std.testing.allocator;
    const result = try countWays("T^F", allocator);
    try std.testing.expectEqual(@as(usize, 1), result.true_count);
    try std.testing.expectEqual(@as(usize, 0), result.false_count);
}

test "boolean parenthesization: T^T (XOR)" {
    const allocator = std.testing.allocator;
    const result = try countWays("T^T", allocator);
    try std.testing.expectEqual(@as(usize, 0), result.true_count);
    try std.testing.expectEqual(@as(usize, 1), result.false_count);
}

test "boolean parenthesization: T|F&T (classic example)" {
    const allocator = std.testing.allocator;
    const result = try countWays("T|F&T", allocator);
    // (T|(F&T)) = T|F = T ✓
    // ((T|F)&T) = T&T = T ✓
    try std.testing.expectEqual(@as(usize, 2), result.true_count);
}

test "boolean parenthesization: T&F|T" {
    const allocator = std.testing.allocator;
    const result = try countWays("T&F|T", allocator);
    // (T&(F|T)) = T&T = T ✓
    // ((T&F)|T) = F|T = T ✓
    try std.testing.expectEqual(@as(usize, 2), result.true_count);
}

test "boolean parenthesization: T|F^T" {
    const allocator = std.testing.allocator;
    const result = try countWays("T|F^T", allocator);
    // (T|(F^T)) = T|T = T ✓
    // ((T|F)^T) = T^T = F
    try std.testing.expectEqual(@as(usize, 1), result.true_count);
    try std.testing.expectEqual(@as(usize, 1), result.false_count);
}

test "boolean parenthesization: complex expression" {
    const allocator = std.testing.allocator;
    const result = try countWays("T&T|F^T", allocator);
    try std.testing.expect(result.true_count > 0);
    try std.testing.expect(result.false_count > 0);
}

test "boolean parenthesization: all AND" {
    const allocator = std.testing.allocator;
    const result = try countWays("T&T&T", allocator);
    try std.testing.expectEqual(@as(usize, 2), result.true_count); // 2 ways to parenthesize
}

test "boolean parenthesization: all OR" {
    const allocator = std.testing.allocator;
    const result = try countWays("F|F|F", allocator);
    try std.testing.expectEqual(@as(usize, 0), result.true_count);
    try std.testing.expectEqual(@as(usize, 2), result.false_count); // 2 ways to parenthesize
}

test "boolean parenthesization: countWaysToTrue helper" {
    const allocator = std.testing.allocator;
    const ways = try countWaysToTrue("T|F&T", allocator);
    try std.testing.expectEqual(@as(usize, 2), ways);
}

test "boolean parenthesization: countWaysToFalse helper" {
    const allocator = std.testing.allocator;
    const ways = try countWaysToFalse("T&F", allocator);
    try std.testing.expectEqual(@as(usize, 1), ways);
}

test "boolean parenthesization: totalParenthesizations" {
    const allocator = std.testing.allocator;
    const total = try totalParenthesizations("T|F^T", allocator);
    try std.testing.expectEqual(@as(usize, 2), total); // 1 true + 1 false
}

test "boolean parenthesization: large expression" {
    const allocator = std.testing.allocator;
    const result = try countWays("T&F|T^F&T|F^T", allocator);
    try std.testing.expect(result.true_count > 0);
    try std.testing.expect(result.false_count > 0);
}

test "boolean parenthesization: empty expression error" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.EmptyExpression, countWays("", allocator));
}

test "boolean parenthesization: invalid expression (even length)" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.InvalidExpression, countWays("T&", allocator));
}

test "boolean parenthesization: invalid symbol" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.InvalidSymbol, countWays("X|F", allocator));
}

test "boolean parenthesization: invalid operator" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.InvalidOperator, countWays("T+F", allocator));
}

test "boolean parenthesization: memory safety" {
    const allocator = std.testing.allocator;
    
    // Test multiple expressions to ensure no leaks
    _ = try countWays("T", allocator);
    _ = try countWays("T&F", allocator);
    _ = try countWays("T|F&T", allocator);
    _ = try countWays("T&F|T^F", allocator);
}
