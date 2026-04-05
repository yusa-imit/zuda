/// Knight's Tour — Find a sequence of moves for a knight to visit every square on a chessboard.
///
/// The Knight's Tour problem asks: can a knight visit all squares on an n×n board exactly once?
/// This is a classic backtracking problem with applications in:
/// - Chess AI and puzzles
/// - Graph theory (Hamiltonian paths on knight graphs)
/// - Algorithm education (demonstrates backtracking techniques)
///
/// ## Algorithm
///
/// Uses backtracking with Warnsdorff's heuristic for optimization:
/// 1. Start from a given position
/// 2. Try all 8 possible knight moves
/// 3. Choose the move to the square with the fewest onward moves (Warnsdorff's rule)
/// 4. Mark the square as visited
/// 5. Recursively visit the next square
/// 6. If no valid move exists, backtrack
///
/// ## Time Complexity
///
/// - **Worst case**: O(8^(n²)) — tries all move combinations
/// - **With Warnsdorff's heuristic**: O(n²) — dramatically reduces backtracking
///
/// ## Space Complexity
///
/// - O(n²) for the board + O(n²) recursion depth
///
/// ## References
///
/// - Warnsdorff's rule (1823) — heuristic that prioritizes moves to less-accessible squares
/// - De Jaenisch (1862) — first complete analysis of the knight's tour

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Position on the chessboard.
pub const Position = struct {
    row: isize,
    col: isize,

    pub fn eql(self: Position, other: Position) bool {
        return self.row == other.row and self.col == other.col;
    }
};

/// Result of finding a knight's tour.
pub const TourResult = struct {
    /// The sequence of positions visited (length = n*n for a complete tour).
    path: []Position,
    /// Whether a complete tour was found.
    found: bool,

    pub fn deinit(self: *TourResult, allocator: Allocator) void {
        allocator.free(self.path);
    }
};

/// Knight move offsets (L-shaped moves: 2 in one direction, 1 in perpendicular).
const knight_moves = [_][2]isize{
    .{ -2, -1 }, .{ -2, 1 }, .{ -1, -2 }, .{ -1, 2 },
    .{ 1, -2 }, .{ 1, 2 }, .{ 2, -1 }, .{ 2, 1 },
};

/// Find a knight's tour on an n×n board starting from (start_row, start_col).
///
/// Uses backtracking with Warnsdorff's heuristic (prioritize moves to less-accessible squares).
///
/// ## Parameters
///
/// - `allocator`: Memory allocator for result path
/// - `n`: Board size (n×n)
/// - `start_row`: Starting row (0-indexed)
/// - `start_col`: Starting column (0-indexed)
///
/// ## Returns
///
/// `TourResult` with:
/// - `path`: Sequence of positions (length n² if complete tour found)
/// - `found`: true if complete tour exists
///
/// ## Errors
///
/// - `OutOfMemory`: Allocation failure
///
/// ## Time Complexity
///
/// - O(n²) with Warnsdorff's heuristic (average case)
/// - O(8^(n²)) worst case without heuristic
///
/// ## Space Complexity
///
/// - O(n²) for board and path storage
///
/// ## Example
///
/// ```zig
/// var result = try knightsTour(allocator, 8, 0, 0);
/// defer result.deinit(allocator);
/// if (result.found) {
///     // result.path contains the full tour (64 positions for 8×8 board)
/// }
/// ```
pub fn knightsTour(allocator: Allocator, n: usize, start_row: usize, start_col: usize) !TourResult {
    if (n == 0) return error.InvalidBoardSize;
    if (start_row >= n or start_col >= n) return error.InvalidStartPosition;

    // Initialize board (all squares unvisited = -1).
    var board = try allocator.alloc([]isize, n);
    defer allocator.free(board);
    for (board) |*row| {
        row.* = try allocator.alloc(isize, n);
    }
    defer for (board) |row| allocator.free(row);
    for (board) |row| {
        for (row) |*cell| cell.* = -1;
    }

    // Path to store the tour.
    var path = try allocator.alloc(Position, n * n);
    errdefer allocator.free(path);

    // Start the tour.
    board[start_row][start_col] = 0;
    path[0] = .{ .row = @intCast(start_row), .col = @intCast(start_col) };

    const found = try solveKnightsTour(board, n, @intCast(start_row), @intCast(start_col), 1, path);

    return TourResult{
        .path = path,
        .found = found,
    };
}

/// Recursive backtracking solver with Warnsdorff's heuristic.
fn solveKnightsTour(
    board: [][]isize,
    n: usize,
    row: isize,
    col: isize,
    move_count: usize,
    path: []Position,
) !bool {
    // Base case: all squares visited.
    if (move_count == n * n) return true;

    // Try all 8 possible knight moves, prioritized by Warnsdorff's heuristic.
    var moves: [8]struct { pos: Position, accessibility: usize } = undefined;
    var valid_count: usize = 0;

    for (knight_moves) |offset| {
        const new_row = row + offset[0];
        const new_col = col + offset[1];

        if (isValid(board, n, new_row, new_col)) {
            const accessibility = countAccessibleSquares(board, n, new_row, new_col);
            moves[valid_count] = .{
                .pos = .{ .row = new_row, .col = new_col },
                .accessibility = accessibility,
            };
            valid_count += 1;
        }
    }

    // Sort moves by accessibility (Warnsdorff's rule: prefer less-accessible squares).
    std.mem.sort(@TypeOf(moves[0]), moves[0..valid_count], {}, struct {
        fn lessThan(_: void, a: @TypeOf(moves[0]), b: @TypeOf(moves[0])) bool {
            return a.accessibility < b.accessibility;
        }
    }.lessThan);

    // Try each move in priority order.
    for (moves[0..valid_count]) |move| {
        const new_row = move.pos.row;
        const new_col = move.pos.col;

        // Make move.
        board[@intCast(new_row)][@intCast(new_col)] = @intCast(move_count);
        path[move_count] = move.pos;

        // Recurse.
        if (try solveKnightsTour(board, n, new_row, new_col, move_count + 1, path)) {
            return true;
        }

        // Backtrack.
        board[@intCast(new_row)][@intCast(new_col)] = -1;
    }

    return false;
}

/// Check if a position is valid and unvisited.
fn isValid(board: [][]isize, n: usize, row: isize, col: isize) bool {
    return row >= 0 and row < n and col >= 0 and col < n and
        board[@intCast(row)][@intCast(col)] == -1;
}

/// Count how many unvisited squares are reachable from (row, col).
/// Used for Warnsdorff's heuristic.
fn countAccessibleSquares(board: [][]isize, n: usize, row: isize, col: isize) usize {
    var count: usize = 0;
    for (knight_moves) |offset| {
        const new_row = row + offset[0];
        const new_col = col + offset[1];
        if (isValid(board, n, new_row, new_col)) {
            count += 1;
        }
    }
    return count;
}

/// Count total number of valid knight's tours from a starting position.
///
/// **Warning**: Extremely slow for n > 5 (exponential time without heuristic).
///
/// ## Time Complexity
///
/// - O(8^(n²)) — explores all possible paths
///
/// ## Example
///
/// ```zig
/// const count = try countTours(allocator, 5, 0, 0); // Small board only!
/// ```
pub fn countTours(allocator: Allocator, n: usize, start_row: usize, start_col: usize) !usize {
    if (n == 0) return error.InvalidBoardSize;
    if (start_row >= n or start_col >= n) return error.InvalidStartPosition;

    var board = try allocator.alloc([]isize, n);
    defer allocator.free(board);
    for (board) |*row| {
        row.* = try allocator.alloc(isize, n);
    }
    defer for (board) |row| allocator.free(row);
    for (board) |row| {
        for (row) |*cell| cell.* = -1;
    }

    board[start_row][start_col] = 0;
    return countToursRecursive(board, n, @intCast(start_row), @intCast(start_col), 1);
}

fn countToursRecursive(board: [][]isize, n: usize, row: isize, col: isize, move_count: usize) usize {
    if (move_count == n * n) return 1;

    var total: usize = 0;
    for (knight_moves) |offset| {
        const new_row = row + offset[0];
        const new_col = col + offset[1];

        if (isValid(board, n, new_row, new_col)) {
            board[@intCast(new_row)][@intCast(new_col)] = @intCast(move_count);
            total += countToursRecursive(board, n, new_row, new_col, move_count + 1);
            board[@intCast(new_row)][@intCast(new_col)] = -1;
        }
    }
    return total;
}

/// Check if a given path is a valid knight's tour.
pub fn isValidTour(n: usize, path: []const Position) bool {
    if (path.len != n * n) return false;

    // Check all squares visited exactly once.
    var visited = std.AutoHashMap(Position, void).init(std.heap.page_allocator);
    defer visited.deinit();

    for (path) |pos| {
        if (pos.row < 0 or pos.row >= n or pos.col < 0 or pos.col >= n) return false;
        if (visited.contains(pos)) return false;
        visited.put(pos, {}) catch return false;
    }

    // Check consecutive positions are valid knight moves.
    for (path[0 .. path.len - 1], path[1..]) |curr, next| {
        const dr = @abs(next.row - curr.row);
        const dc = @abs(next.col - curr.col);
        if (!((dr == 2 and dc == 1) or (dr == 1 and dc == 2))) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "knight's tour: 5×5 board from (0,0)" {
    const allocator = testing.allocator;
    var result = try knightsTour(allocator, 5, 0, 0);
    defer result.deinit(allocator);

    try testing.expect(result.found);
    try testing.expectEqual(@as(usize, 25), result.path.len);
    try testing.expect(isValidTour(5, result.path));
}

test "knight's tour: 6×6 board from (0,0)" {
    const allocator = testing.allocator;
    var result = try knightsTour(allocator, 6, 0, 0);
    defer result.deinit(allocator);

    try testing.expect(result.found);
    try testing.expectEqual(@as(usize, 36), result.path.len);
    try testing.expect(isValidTour(6, result.path));
}

test "knight's tour: 8×8 board from (0,0)" {
    const allocator = testing.allocator;
    var result = try knightsTour(allocator, 8, 0, 0);
    defer result.deinit(allocator);

    try testing.expect(result.found);
    try testing.expectEqual(@as(usize, 64), result.path.len);
    try testing.expect(isValidTour(8, result.path));
    try testing.expectEqual(@as(isize, 0), result.path[0].row);
    try testing.expectEqual(@as(isize, 0), result.path[0].col);
}

test "knight's tour: 8×8 board from (3,4)" {
    const allocator = testing.allocator;
    var result = try knightsTour(allocator, 8, 3, 4);
    defer result.deinit(allocator);

    try testing.expect(result.found);
    try testing.expectEqual(@as(usize, 64), result.path.len);
    try testing.expect(isValidTour(8, result.path));
    try testing.expectEqual(@as(isize, 3), result.path[0].row);
    try testing.expectEqual(@as(isize, 4), result.path[0].col);
}

test "knight's tour: different start positions on 6×6" {
    const allocator = testing.allocator;

    const starts = [_]Position{
        .{ .row = 0, .col = 0 },
        .{ .row = 2, .col = 2 },
        .{ .row = 5, .col = 5 },
    };

    for (starts) |start| {
        var result = try knightsTour(allocator, 6, @intCast(start.row), @intCast(start.col));
        defer result.deinit(allocator);

        try testing.expect(result.found);
        try testing.expectEqual(@as(usize, 36), result.path.len);
        try testing.expect(isValidTour(6, result.path));
        try testing.expect(result.path[0].eql(start));
    }
}

test "knight's tour: invalid board size" {
    const allocator = testing.allocator;
    try testing.expectError(error.InvalidBoardSize, knightsTour(allocator, 0, 0, 0));
}

test "knight's tour: invalid start position" {
    const allocator = testing.allocator;
    try testing.expectError(error.InvalidStartPosition, knightsTour(allocator, 8, 8, 0));
    try testing.expectError(error.InvalidStartPosition, knightsTour(allocator, 8, 0, 8));
    try testing.expectError(error.InvalidStartPosition, knightsTour(allocator, 8, 10, 10));
}

test "knight's tour: path validation - valid tour" {
    const path = [_]Position{
        .{ .row = 0, .col = 0 }, .{ .row = 2, .col = 1 }, .{ .row = 0, .col = 2 },
        .{ .row = 1, .col = 0 }, .{ .row = 2, .col = 2 }, .{ .row = 0, .col = 1 },
        .{ .row = 1, .col = 2 }, .{ .row = 2, .col = 0 }, .{ .row = 1, .col = 1 },
    };
    try testing.expect(isValidTour(3, &path));
}

test "knight's tour: path validation - wrong length" {
    const path = [_]Position{
        .{ .row = 0, .col = 0 }, .{ .row = 2, .col = 1 },
    };
    try testing.expect(!isValidTour(3, &path));
}

test "knight's tour: path validation - duplicate position" {
    const path = [_]Position{
        .{ .row = 0, .col = 0 }, .{ .row = 2, .col = 1 }, .{ .row = 0, .col = 0 },
        .{ .row = 1, .col = 0 }, .{ .row = 2, .col = 2 }, .{ .row = 0, .col = 1 },
        .{ .row = 1, .col = 2 }, .{ .row = 2, .col = 0 }, .{ .row = 1, .col = 1 },
    };
    try testing.expect(!isValidTour(3, &path));
}

test "knight's tour: path validation - invalid move" {
    const path = [_]Position{
        .{ .row = 0, .col = 0 }, .{ .row = 1, .col = 1 }, // Not a knight move!
        .{ .row = 0, .col = 2 },
        .{ .row = 1, .col = 0 }, .{ .row = 2, .col = 2 }, .{ .row = 0, .col = 1 },
        .{ .row = 1, .col = 2 }, .{ .row = 2, .col = 0 }, .{ .row = 2, .col = 1 },
    };
    try testing.expect(!isValidTour(3, &path));
}

test "knight's tour: path validation - out of bounds" {
    const path = [_]Position{
        .{ .row = 0, .col = 0 }, .{ .row = 2, .col = 1 }, .{ .row = 0, .col = 5 }, // Out of bounds!
        .{ .row = 1, .col = 0 }, .{ .row = 2, .col = 2 }, .{ .row = 0, .col = 1 },
        .{ .row = 1, .col = 2 }, .{ .row = 2, .col = 0 }, .{ .row = 1, .col = 1 },
    };
    try testing.expect(!isValidTour(3, &path));
}

test "knight's tour: count tours on 5×5 (slow)" {
    const allocator = testing.allocator;
    const count = try countTours(allocator, 5, 0, 0);
    try testing.expect(count > 0); // At least one tour exists
}

test "knight's tour: Warnsdorff's heuristic effectiveness" {
    const allocator = testing.allocator;

    // Warnsdorff's heuristic should find a solution quickly for 8×8.
    var result = try knightsTour(allocator, 8, 0, 0);
    defer result.deinit(allocator);

    try testing.expect(result.found);
    try testing.expectEqual(@as(usize, 64), result.path.len);
}

test "knight's tour: memory safety" {
    const allocator = testing.allocator;

    // Multiple allocations and deallocations.
    for (0..10) |_| {
        var result = try knightsTour(allocator, 5, 0, 0);
        result.deinit(allocator);
    }
}
