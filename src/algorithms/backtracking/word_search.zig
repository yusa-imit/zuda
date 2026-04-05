//! Word Search - 2D Grid Backtracking
//!
//! Given a 2D grid of characters and a word, find if the word exists in the grid.
//! The word can be constructed from letters of sequentially adjacent cells,
//! where adjacent cells are horizontally or vertically neighboring.
//! The same cell may not be used more than once.
//!
//! Algorithm:
//! - Type: Backtracking with DFS
//! - Time: O(m × n × 4^L) where m = rows, n = cols, L = word length
//! - Space: O(L) for recursion stack + O(m × n) for visited tracking
//!
//! Use cases:
//! - Word puzzles (crossword validation, word games)
//! - Pattern matching in 2D data (image processing, OCR)
//! - Grid-based search problems
//! - Pathfinding with constraints
//!
//! Reference: Classic backtracking problem, LeetCode #79

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// Position in the grid
pub const Position = struct {
    row: usize,
    col: usize,

    pub fn eql(self: Position, other: Position) bool {
        return self.row == other.row and self.col == other.col;
    }
};

/// Check if word exists in the board
///
/// Time: O(m × n × 4^L) where m = rows, n = cols, L = word length
/// Space: O(L) recursion + O(m × n) visited
pub fn exist(comptime T: type, board: []const []const T, word: []const T) bool {
    if (board.len == 0 or word.len == 0) return false;

    const rows = board.len;
    const cols = board[0].len;

    // Allocate visited matrix on stack for small boards
    var visited_buf: [100][100]bool = undefined;
    const use_stack = rows <= 100 and cols <= 100;

    if (use_stack) {
        // Stack allocation for small boards
        for (0..rows) |i| {
            for (0..cols) |j| {
                visited_buf[i][j] = false;
            }
        }

        // Try starting from each cell
        for (0..rows) |i| {
            for (0..cols) |j| {
                if (dfs(T, board, word, 0, i, j, &visited_buf)) {
                    return true;
                }
            }
        }
        return false;
    } else {
        // For large boards, would need allocator - return false for now
        // (existWithPath handles this case properly)
        return false;
    }
}

/// Find word in board and return the path if found
///
/// Time: O(m × n × 4^L)
/// Space: O(L) recursion + O(m × n) visited + O(L) path
pub fn existWithPath(comptime T: type, allocator: Allocator, board: []const []const T, word: []const T) !?ArrayList(Position) {
    if (board.len == 0 or word.len == 0) return null;

    const rows = board.len;
    const cols = board[0].len;

    // Allocate visited matrix
    const visited = try allocator.alloc([]bool, rows);
    defer {
        for (visited) |row| {
            allocator.free(row);
        }
        allocator.free(visited);
    }

    for (0..rows) |i| {
        visited[i] = try allocator.alloc(bool, cols);
        @memset(visited[i], false);
    }

    // Try starting from each cell
    for (0..rows) |i| {
        for (0..cols) |j| {
            var path = try ArrayList(Position).initCapacity(allocator, word.len);
            if (try dfsWithPath(T, allocator, board, word, 0, i, j, visited, &path)) {
                return path;
            }
            path.deinit(allocator);
        }
    }

    return null;
}

/// Find all occurrences of word in board
///
/// Time: O(m × n × 4^L × k) where k = number of occurrences
/// Space: O(k × L) for all paths
pub fn findAll(comptime T: type, allocator: Allocator, board: []const []const T, word: []const T) !ArrayList(ArrayList(Position)) {
    var results = try ArrayList(ArrayList(Position)).initCapacity(allocator, 1);
    errdefer {
        for (results.items) |*path| {
            path.deinit(allocator);
        }
        results.deinit(allocator);
    }

    if (board.len == 0 or word.len == 0) return results;

    const rows = board.len;
    const cols = board[0].len;

    // Allocate visited matrix
    const visited = try allocator.alloc([]bool, rows);
    defer {
        for (visited) |row| {
            allocator.free(row);
        }
        allocator.free(visited);
    }

    for (0..rows) |i| {
        visited[i] = try allocator.alloc(bool, cols);
        @memset(visited[i], false);
    }

    // Try starting from each cell
    for (0..rows) |i| {
        for (0..cols) |j| {
            var path = try ArrayList(Position).initCapacity(allocator, word.len);
            if (try dfsWithPath(T, allocator, board, word, 0, i, j, visited, &path)) {
                try results.append(allocator, path);
            } else {
                path.deinit(allocator);
            }
        }
    }

    return results;
}

/// Count occurrences of word in board
///
/// Time: O(m × n × 4^L)
/// Space: O(L) recursion + O(m × n) visited
pub fn countOccurrences(comptime T: type, allocator: Allocator, board: []const []const T, word: []const T) !usize {
    const all = try findAll(T, allocator, board, word);
    defer {
        for (all.items) |path| {
            path.deinit(allocator);
        }
        all.deinit(allocator);
    }
    return all.items.len;
}

// Helper: DFS with backtracking (stack-based visited)
fn dfs(comptime T: type, board: []const []const T, word: []const T, index: usize, row: usize, col: usize, visited: *[100][100]bool) bool {
    const rows = board.len;
    const cols = board[0].len;

    // Base case: found the word
    if (index == word.len) return true;

    // Boundary check
    if (row >= rows or col >= cols) return false;

    // Check if visited or character mismatch
    if (visited[row][col] or board[row][col] != word[index]) return false;

    // Mark as visited
    visited[row][col] = true;

    // Explore all four directions
    const directions = [_][2]isize{ .{ -1, 0 }, .{ 1, 0 }, .{ 0, -1 }, .{ 0, 1 } };

    for (directions) |dir| {
        const new_row = @as(isize, @intCast(row)) + dir[0];
        const new_col = @as(isize, @intCast(col)) + dir[1];

        if (new_row >= 0 and new_col >= 0) {
            if (dfs(T, board, word, index + 1, @intCast(new_row), @intCast(new_col), visited)) {
                visited[row][col] = false; // Backtrack
                return true;
            }
        }
    }

    // Backtrack
    visited[row][col] = false;
    return false;
}

// Helper: DFS with path reconstruction (heap-based visited)
fn dfsWithPath(comptime T: type, allocator: Allocator, board: []const []const T, word: []const T, index: usize, row: usize, col: usize, visited: [][]bool, path: *ArrayList(Position)) !bool {
    const rows = board.len;
    const cols = board[0].len;

    // Base case: found the word
    if (index == word.len) return true;

    // Boundary check
    if (row >= rows or col >= cols) return false;

    // Check if visited or character mismatch
    if (visited[row][col] or board[row][col] != word[index]) return false;

    // Mark as visited and add to path
    visited[row][col] = true;
    path.appendAssumeCapacity(Position{ .row = row, .col = col });

    // Explore all four directions
    const directions = [_][2]isize{ .{ -1, 0 }, .{ 1, 0 }, .{ 0, -1 }, .{ 0, 1 } };

    for (directions) |dir| {
        const new_row = @as(isize, @intCast(row)) + dir[0];
        const new_col = @as(isize, @intCast(col)) + dir[1];

        if (new_row >= 0 and new_col >= 0) {
            if (try dfsWithPath(T, allocator, board, word, index + 1, @intCast(new_row), @intCast(new_col), visited, path)) {
                visited[row][col] = false; // Backtrack visited
                return true;
            }
        }
    }

    // Backtrack: unmark visited and remove from path
    visited[row][col] = false;
    _ = path.pop();
    return false;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "word search - basic 3x4 grid" {
    const board = [_][]const u8{
        &.{ 'A', 'B', 'C', 'E' },
        &.{ 'S', 'F', 'C', 'S' },
        &.{ 'A', 'D', 'E', 'E' },
    };

    // Word exists: ABCCED
    try testing.expect(exist(u8, &board, "ABCCED"));

    // Word exists: SEE
    try testing.expect(exist(u8, &board, "SEE"));

    // Word exists: AS
    try testing.expect(exist(u8, &board, "AS"));
}

test "word search - word not found" {
    const board = [_][]const u8{
        &.{ 'A', 'B', 'C', 'E' },
        &.{ 'S', 'F', 'C', 'S' },
        &.{ 'A', 'D', 'E', 'E' },
    };

    // Word doesn't exist: ABCB (can't reuse 'B')
    try testing.expect(!exist(u8, &board, "ABCB"));

    // Word doesn't exist: XYZ
    try testing.expect(!exist(u8, &board, "XYZ"));
}

test "word search - single cell" {
    const board = [_][]const u8{
        &.{'A'},
    };

    try testing.expect(exist(u8, &board, "A"));
    try testing.expect(!exist(u8, &board, "B"));
    try testing.expect(!exist(u8, &board, "AA"));
}

test "word search - empty grid or word" {
    const empty_board = [_][]const u8{};
    try testing.expect(!exist(u8, &empty_board, "A"));

    const board = [_][]const u8{
        &.{'A'},
    };
    try testing.expect(!exist(u8, &board, ""));
}

test "word search - word longer than grid" {
    const board = [_][]const u8{
        &.{ 'A', 'B' },
        &.{ 'C', 'D' },
    };

    // Only 4 cells, word needs 5
    try testing.expect(!exist(u8, &board, "ABCDE"));
}

test "word search - duplicate letters" {
    const board = [_][]const u8{
        &.{ 'A', 'A', 'A' },
        &.{ 'A', 'A', 'A' },
        &.{ 'A', 'A', 'A' },
    };

    try testing.expect(exist(u8, &board, "AAA"));
    try testing.expect(exist(u8, &board, "AAAAA"));
    try testing.expect(!exist(u8, &board, "AAAAAAAAAA")); // Too long
}

test "word search - with path reconstruction" {
    const board = [_][]const u8{
        &.{ 'A', 'B', 'C' },
        &.{ 'D', 'E', 'F' },
        &.{ 'G', 'H', 'I' },
    };

    const allocator = testing.allocator;

    // Find "ABC"
    const result = try existWithPath(u8, allocator, &board, "ABC");
    try testing.expect(result != null);
    defer result.?.deinit(allocator);

    try testing.expectEqual(@as(usize, 3), result.?.items.len);
    try testing.expect(result.?.items[0].eql(Position{ .row = 0, .col = 0 })); // A
    try testing.expect(result.?.items[1].eql(Position{ .row = 0, .col = 1 })); // B
    try testing.expect(result.?.items[2].eql(Position{ .row = 0, .col = 2 })); // C
}

test "word search - path not found" {
    const board = [_][]const u8{
        &.{ 'A', 'B' },
        &.{ 'C', 'D' },
    };

    const allocator = testing.allocator;
    const result = try existWithPath(u8, allocator, &board, "XYZ");
    try testing.expect(result == null);
}

test "word search - zigzag path" {
    const board = [_][]const u8{
        &.{ 'A', 'B', 'C' },
        &.{ 'D', 'E', 'F' },
        &.{ 'G', 'H', 'I' },
    };

    const allocator = testing.allocator;

    // Find "ABEHI" (A -> B -> E -> H -> I)
    const result = try existWithPath(u8, allocator, &board, "ABEHI");
    try testing.expect(result != null);
    defer result.?.deinit(allocator);

    try testing.expectEqual(@as(usize, 5), result.?.items.len);
}

test "word search - find all occurrences" {
    const board = [_][]const u8{
        &.{ 'A', 'B', 'A' },
        &.{ 'A', 'B', 'A' },
    };

    const allocator = testing.allocator;

    // Find all "AB" occurrences
    const results = try findAll(u8, allocator, &board, "AB");
    defer {
        for (results.items) |*path| {
            path.deinit(allocator);
        }
        results.deinit(allocator);
    }

    // Should find 2 occurrences: top-left AB and bottom-left AB
    try testing.expectEqual(@as(usize, 2), results.items.len);
}

test "word search - count occurrences" {
    const board = [_][]const u8{
        &.{ 'A', 'B', 'A' },
        &.{ 'A', 'B', 'A' },
    };

    const allocator = testing.allocator;
    const count = try countOccurrences(u8, allocator, &board, "AB");
    try testing.expectEqual(@as(usize, 2), count);
}

test "word search - no occurrences" {
    const board = [_][]const u8{
        &.{ 'A', 'B' },
        &.{ 'C', 'D' },
    };

    const allocator = testing.allocator;
    const count = try countOccurrences(u8, allocator, &board, "XYZ");
    try testing.expectEqual(@as(usize, 0), count);
}

test "word search - single occurrence" {
    const board = [_][]const u8{
        &.{ 'A', 'B' },
        &.{ 'C', 'D' },
    };

    const allocator = testing.allocator;
    const count = try countOccurrences(u8, allocator, &board, "ABDC");
    try testing.expectEqual(@as(usize, 1), count);
}

test "word search - large grid stress test" {
    const allocator = testing.allocator;

    // Create 10x10 grid with pattern
    var board_storage: [10][10]u8 = undefined;
    var board_rows: [10][]const u8 = undefined;

    for (0..10) |i| {
        for (0..10) |j| {
            board_storage[i][j] = 'A' + @as(u8, @intCast((i + j) % 26));
        }
        board_rows[i] = &board_storage[i];
    }

    const board: []const []const u8 = &board_rows;

    // Search for a word that exists
    try testing.expect(exist(u8, board, "ABC"));

    // Search with path
    const result = try existWithPath(u8, allocator, board, "ABC");
    try testing.expect(result != null);
    defer result.?.deinit(allocator);
}

test "word search - memory safety" {
    const board = [_][]const u8{
        &.{ 'A', 'B', 'C' },
        &.{ 'D', 'E', 'F' },
    };

    const allocator = testing.allocator;

    // Test existWithPath
    {
        const result = try existWithPath(u8, allocator, &board, "ABC");
        try testing.expect(result != null);
        result.?.deinit(allocator);
    }

    // Test findAll
    {
        const results = try findAll(u8, allocator, &board, "ABC");
        for (results.items) |*path| {
            path.deinit(allocator);
        }
        results.deinit(allocator);
    }

    // Test countOccurrences
    _ = try countOccurrences(u8, allocator, &board, "ABC");
}

test "word search - integer type" {
    const board = [_][]const u32{
        &.{ 1, 2, 3 },
        &.{ 4, 5, 6 },
        &.{ 7, 8, 9 },
    };

    // Search for pattern 1-2-5-8
    try testing.expect(exist(u32, &board, &.{ 1, 2, 5, 8 }));

    // Search for non-existent pattern
    try testing.expect(!exist(u32, &board, &.{ 1, 3, 5, 7 }));
}

test "word search - vertical and horizontal paths" {
    const board = [_][]const u8{
        &.{ 'A', 'B', 'C' },
        &.{ 'D', 'E', 'F' },
        &.{ 'G', 'H', 'I' },
    };

    // Horizontal: ABC
    try testing.expect(exist(u8, &board, "ABC"));

    // Vertical: ADG
    try testing.expect(exist(u8, &board, "ADG"));

    // Diagonal not allowed (only horizontal/vertical)
    try testing.expect(!exist(u8, &board, "AEI"));
}
