const std = @import("std");

/// Sudoku solver using backtracking.
/// Board is represented as [9][9]u8 where 0 represents empty cells.
///
/// Time: O(9^(n*n)) worst case where n=9, but heavy pruning in practice
/// Space: O(n*n) for recursion stack
pub fn solveSudoku(board: *[9][9]u8) bool {
    return backtrack(board, 0, 0);
}

fn backtrack(board: *[9][9]u8, row: usize, col: usize) bool {
    // Find next empty cell
    var curr_row = row;
    var curr_col = col;

    while (curr_row < 9) : (curr_row += 1) {
        while (curr_col < 9) : (curr_col += 1) {
            if (board[curr_row][curr_col] == 0) {
                // Try digits 1-9
                for (1..10) |digit| {
                    const d: u8 = @intCast(digit);
                    if (isValid(board, curr_row, curr_col, d)) {
                        board[curr_row][curr_col] = d;

                        // Move to next cell
                        const next_col = if (curr_col + 1 < 9) curr_col + 1 else 0;
                        const next_row = if (curr_col + 1 < 9) curr_row else curr_row + 1;

                        if (backtrack(board, next_row, next_col)) {
                            return true;
                        }

                        // Backtrack
                        board[curr_row][curr_col] = 0;
                    }
                }
                return false; // No valid digit found
            }
        }
        curr_col = 0;
    }

    return true; // All cells filled
}

fn isValid(board: *const [9][9]u8, row: usize, col: usize, digit: u8) bool {
    // Check row
    for (0..9) |c| {
        if (board[row][c] == digit) return false;
    }

    // Check column
    for (0..9) |r| {
        if (board[r][col] == digit) return false;
    }

    // Check 3x3 box
    const box_row = (row / 3) * 3;
    const box_col = (col / 3) * 3;
    for (0..3) |r| {
        for (0..3) |c| {
            if (board[box_row + r][box_col + c] == digit) return false;
        }
    }

    return true;
}

/// Validate a completed or partial Sudoku board
///
/// Time: O(1) - fixed 9x9 board
/// Space: O(1)
pub fn isValidSudoku(board: *const [9][9]u8) bool {
    // Check rows
    for (0..9) |row| {
        var seen = [_]bool{false} ** 10;
        for (0..9) |col| {
            const digit = board[row][col];
            if (digit != 0) {
                if (seen[digit]) return false;
                seen[digit] = true;
            }
        }
    }

    // Check columns
    for (0..9) |col| {
        var seen = [_]bool{false} ** 10;
        for (0..9) |row| {
            const digit = board[row][col];
            if (digit != 0) {
                if (seen[digit]) return false;
                seen[digit] = true;
            }
        }
    }

    // Check 3x3 boxes
    for (0..3) |box_row_idx| {
        for (0..3) |box_col_idx| {
            var seen = [_]bool{false} ** 10;
            const box_row = box_row_idx * 3;
            const box_col = box_col_idx * 3;
            for (0..3) |r| {
                for (0..3) |c| {
                    const digit = board[box_row + r][box_col + c];
                    if (digit != 0) {
                        if (seen[digit]) return false;
                        seen[digit] = true;
                    }
                }
            }
        }
    }

    return true;
}

test "Sudoku: solve easy puzzle" {
    var board = [_][9]u8{
        [_]u8{ 5, 3, 0, 0, 7, 0, 0, 0, 0 },
        [_]u8{ 6, 0, 0, 1, 9, 5, 0, 0, 0 },
        [_]u8{ 0, 9, 8, 0, 0, 0, 0, 6, 0 },
        [_]u8{ 8, 0, 0, 0, 6, 0, 0, 0, 3 },
        [_]u8{ 4, 0, 0, 8, 0, 3, 0, 0, 1 },
        [_]u8{ 7, 0, 0, 0, 2, 0, 0, 0, 6 },
        [_]u8{ 0, 6, 0, 0, 0, 0, 2, 8, 0 },
        [_]u8{ 0, 0, 0, 4, 1, 9, 0, 0, 5 },
        [_]u8{ 0, 0, 0, 0, 8, 0, 0, 7, 9 },
    };

    const solved = solveSudoku(&board);
    try std.testing.expect(solved);
    try std.testing.expect(isValidSudoku(&board));

    // Verify some known solution values
    try std.testing.expectEqual(@as(u8, 5), board[0][0]);
    try std.testing.expectEqual(@as(u8, 3), board[0][1]);
    try std.testing.expectEqual(@as(u8, 9), board[8][8]);
}

test "Sudoku: validate valid board" {
    const board = [_][9]u8{
        [_]u8{ 5, 3, 4, 6, 7, 8, 9, 1, 2 },
        [_]u8{ 6, 7, 2, 1, 9, 5, 3, 4, 8 },
        [_]u8{ 1, 9, 8, 3, 4, 2, 5, 6, 7 },
        [_]u8{ 8, 5, 9, 7, 6, 1, 4, 2, 3 },
        [_]u8{ 4, 2, 6, 8, 5, 3, 7, 9, 1 },
        [_]u8{ 7, 1, 3, 9, 2, 4, 8, 5, 6 },
        [_]u8{ 9, 6, 1, 5, 3, 7, 2, 8, 4 },
        [_]u8{ 2, 8, 7, 4, 1, 9, 6, 3, 5 },
        [_]u8{ 3, 4, 5, 2, 8, 6, 1, 7, 9 },
    };

    try std.testing.expect(isValidSudoku(&board));
}

test "Sudoku: detect invalid row" {
    const board = [_][9]u8{
        [_]u8{ 5, 5, 0, 0, 0, 0, 0, 0, 0 }, // duplicate 5 in row
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    };

    try std.testing.expect(!isValidSudoku(&board));
}

test "Sudoku: detect invalid column" {
    const board = [_][9]u8{
        [_]u8{ 5, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 5, 0, 0, 0, 0, 0, 0, 0, 0 }, // duplicate 5 in column
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    };

    try std.testing.expect(!isValidSudoku(&board));
}

test "Sudoku: detect invalid 3x3 box" {
    const board = [_][9]u8{
        [_]u8{ 5, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 5, 0, 0, 0, 0, 0, 0, 0 }, // duplicate 5 in box
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    };

    try std.testing.expect(!isValidSudoku(&board));
}

test "Sudoku: unsolvable puzzle returns false" {
    var board = [_][9]u8{
        [_]u8{ 5, 5, 0, 0, 0, 0, 0, 0, 0 }, // invalid initial state
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    };

    const solved = solveSudoku(&board);
    try std.testing.expect(!solved);
}
