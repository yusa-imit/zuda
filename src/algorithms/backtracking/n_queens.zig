const std = @import("std");
const ArrayList = std.ArrayList;

/// N-Queens problem: Place N queens on an NxN chessboard such that no two queens attack each other.
/// A queen can attack any piece in the same row, column, or diagonal.
///
/// Time: O(N!) worst case - explores all permutations with pruning
/// Space: O(N) for recursion stack + board state
pub fn solveNQueens(allocator: std.mem.Allocator, n: usize) !ArrayList([]const u8) {
    var solutions = ArrayList([]const u8).init(allocator);
    errdefer {
        for (solutions.items) |sol| allocator.free(sol);
        solutions.deinit();
    }

    if (n == 0) return solutions;

    const board = try allocator.alloc(i32, n);
    defer allocator.free(board);
    @memset(board, -1);

    try backtrack(allocator, &solutions, board, 0, n);
    return solutions;
}

fn backtrack(
    allocator: std.mem.Allocator,
    solutions: *ArrayList([]const u8),
    board: []i32,
    row: usize,
    n: usize,
) !void {
    if (row == n) {
        // Found a solution
        const solution = try boardToString(allocator, board, n);
        try solutions.append(solution);
        return;
    }

    for (0..n) |col| {
        if (isSafe(board, row, col, n)) {
            board[row] = @intCast(col);
            try backtrack(allocator, solutions, board, row + 1, n);
            board[row] = -1; // Backtrack
        }
    }
}

fn isSafe(board: []const i32, row: usize, col: usize, _: usize) bool {
    // Check all previously placed queens
    for (0..row) |i| {
        const placed_col: usize = @intCast(board[i]);

        // Same column
        if (placed_col == col) return false;

        // Same diagonal
        const row_diff = row - i;
        const col_diff = if (col > placed_col) col - placed_col else placed_col - col;
        if (row_diff == col_diff) return false;
    }
    return true;
}

fn boardToString(allocator: std.mem.Allocator, board: []const i32, n: usize) ![]const u8 {
    const size = n * (n + 1); // n chars per row + newline
    var buffer = try allocator.alloc(u8, size);
    var pos: usize = 0;

    for (0..n) |row| {
        const col: usize = @intCast(board[row]);
        for (0..n) |c| {
            buffer[pos] = if (c == col) 'Q' else '.';
            pos += 1;
        }
        buffer[pos] = '\n';
        pos += 1;
    }

    return buffer;
}

/// Count all solutions without generating boards (more efficient)
///
/// Time: O(N!) worst case
/// Space: O(N) for recursion stack
pub fn countNQueens(n: usize) u64 {
    if (n == 0) return 0;

    var board = [_]i32{-1} ** 64;
    return countBacktrack(board[0..n], 0, n);
}

fn countBacktrack(board: []i32, row: usize, n: usize) u64 {
    if (row == n) return 1;

    var count: u64 = 0;
    for (0..n) |col| {
        if (isSafe(board, row, col, n)) {
            board[row] = @intCast(col);
            count += countBacktrack(board, row + 1, n);
            board[row] = -1;
        }
    }
    return count;
}

test "N-Queens: 4x4 board has 2 solutions" {
    const allocator = std.testing.allocator;
    var solutions = try solveNQueens(allocator, 4);
    defer {
        for (solutions.items) |sol| allocator.free(sol);
        solutions.deinit();
    }

    try std.testing.expectEqual(@as(usize, 2), solutions.items.len);

    // Verify first solution format
    const first = solutions.items[0];
    var lines: usize = 0;
    for (first) |c| {
        if (c == '\n') lines += 1;
    }
    try std.testing.expectEqual(@as(usize, 4), lines);
}

test "N-Queens: count solutions efficiently" {
    try std.testing.expectEqual(@as(u64, 0), countNQueens(0));
    try std.testing.expectEqual(@as(u64, 1), countNQueens(1));
    try std.testing.expectEqual(@as(u64, 0), countNQueens(2));
    try std.testing.expectEqual(@as(u64, 0), countNQueens(3));
    try std.testing.expectEqual(@as(u64, 2), countNQueens(4));
    try std.testing.expectEqual(@as(u64, 10), countNQueens(5));
    try std.testing.expectEqual(@as(u64, 4), countNQueens(6));
    try std.testing.expectEqual(@as(u64, 40), countNQueens(7));
    try std.testing.expectEqual(@as(u64, 92), countNQueens(8));
}

test "N-Queens: 1x1 board" {
    const allocator = std.testing.allocator;
    var solutions = try solveNQueens(allocator, 1);
    defer {
        for (solutions.items) |sol| allocator.free(sol);
        solutions.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), solutions.items.len);
    try std.testing.expectEqualStrings("Q\n", solutions.items[0]);
}

test "N-Queens: 2x2 and 3x3 have no solutions" {
    const allocator = std.testing.allocator;

    var solutions2 = try solveNQueens(allocator, 2);
    defer solutions2.deinit();
    try std.testing.expectEqual(@as(usize, 0), solutions2.items.len);

    var solutions3 = try solveNQueens(allocator, 3);
    defer solutions3.deinit();
    try std.testing.expectEqual(@as(usize, 0), solutions3.items.len);
}

test "N-Queens: verify 8x8 classic solution count" {
    try std.testing.expectEqual(@as(u64, 92), countNQueens(8));
}
