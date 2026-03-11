const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Hungarian - Optimal assignment algorithm for weighted bipartite matching.
///
/// Solves the assignment problem: find a maximum-weight perfect matching in a weighted bipartite graph.
/// Also known as the Kuhn-Munkres algorithm or the Hungarian method.
///
/// Time Complexity: O(V³) for dense graphs
/// Space Complexity: O(V²)
///
/// The algorithm:
/// 1. Reduce the cost matrix by subtracting row/column minimums
/// 2. Find a maximum matching using minimum number of lines to cover all zeros
/// 3. If matching is perfect, done; otherwise adjust costs and repeat
///
/// Uses the primal-dual approach with dual variables (potentials) for efficiency.
///
/// Generic parameters:
/// - W: Weight type (must support arithmetic operations and comparison)
pub fn Hungarian(comptime W: type) type {
    return struct {
        const Self = @This();

        /// Result of Hungarian algorithm
        pub const Result = struct {
            /// Total cost/weight of the optimal assignment
            total_cost: W,
            /// Assignment: assignment[i] = j means row i is matched to column j
            /// -1 indicates unmatched (for non-perfect matchings)
            assignment: []isize,
            allocator: Allocator,

            pub fn deinit(self: *Result) void {
                self.allocator.free(self.assignment);
            }

            /// Check if row i is matched
            pub fn isMatched(self: *const Result, row: usize) bool {
                return self.assignment[row] >= 0;
            }

            /// Get the column that row i is matched to (or null if unmatched)
            pub fn getMatch(self: *const Result, row: usize) ?usize {
                const col = self.assignment[row];
                if (col >= 0) {
                    return @intCast(col);
                }
                return null;
            }
        };

        /// Run Hungarian algorithm to find minimum cost perfect matching.
        ///
        /// The cost matrix is n×n where cost[i][j] is the cost of assigning row i to column j.
        /// For maximum weight matching, negate all weights before passing.
        ///
        /// Returns Result with optimal assignment.
        ///
        /// Time: O(n³) | Space: O(n²)
        pub fn run(
            allocator: Allocator,
            cost_matrix: []const []const W,
            zero_weight: W,
        ) !Result {
            const n = cost_matrix.len;
            if (n == 0) {
                const assignment = try allocator.alloc(isize, 0);
                return Result{
                    .total_cost = zero_weight,
                    .assignment = assignment,
                    .allocator = allocator,
                };
            }

            // Create mutable cost matrix copy
            var costs = try allocator.alloc([]W, n);
            errdefer allocator.free(costs);
            for (costs, 0..) |*row, i| {
                row.* = try allocator.alloc(W, n);
                errdefer for (costs[0..i]) |r| allocator.free(r);
                @memcpy(row.*, cost_matrix[i]);
            }
            defer {
                for (costs) |row| allocator.free(row);
                allocator.free(costs);
            }

            // Step 1: Subtract row minimums
            for (costs) |row| {
                var min_val = row[0];
                for (row[1..]) |val| {
                    if (@as(i128, @intCast(val)) < @as(i128, @intCast(min_val))) {
                        min_val = val;
                    }
                }
                for (row) |*val| {
                    val.* = val.* - min_val;
                }
            }

            // Step 2: Subtract column minimums
            for (0..n) |j| {
                var min_val = costs[0][j];
                for (costs[1..]) |row| {
                    if (@as(i128, @intCast(row[j])) < @as(i128, @intCast(min_val))) {
                        min_val = row[j];
                    }
                }
                for (costs) |row| {
                    row[j] = row[j] - min_val;
                }
            }

            // Matching arrays
            var row_match = try allocator.alloc(isize, n);
            defer allocator.free(row_match);
            var col_match = try allocator.alloc(isize, n);
            defer allocator.free(col_match);

            @memset(row_match, -1);
            @memset(col_match, -1);

            // Find initial matching using greedy approach
            for (0..n) |i| {
                for (0..n) |j| {
                    if (@as(i128, @intCast(costs[i][j])) == @as(i128, @intCast(zero_weight)) and col_match[j] == -1) {
                        row_match[i] = @intCast(j);
                        col_match[j] = @intCast(i);
                        break;
                    }
                }
            }

            // Augment until perfect matching
            for (0..n) |i| {
                if (row_match[i] == -1) {
                    try augment(allocator, costs, row_match, col_match, i, zero_weight);
                }
            }

            // Compute total cost from original matrix
            var total: W = zero_weight;
            for (row_match, 0..) |col, i| {
                if (col >= 0) {
                    const j: usize = @intCast(col);
                    total = total + cost_matrix[i][j];
                }
            }

            // Copy assignment
            const assignment = try allocator.alloc(isize, n);
            @memcpy(assignment, row_match);

            return Result{
                .total_cost = total,
                .assignment = assignment,
                .allocator = allocator,
            };
        }

        /// Find an augmenting path from unmatched row using BFS
        fn augment(
            allocator: Allocator,
            costs: [][]W,
            row_match: []isize,
            col_match: []isize,
            start_row: usize,
            zero_weight: W,
        ) !void {
            const n = costs.len;

            var visited_rows = try allocator.alloc(bool, n);
            defer allocator.free(visited_rows);
            var visited_cols = try allocator.alloc(bool, n);
            defer allocator.free(visited_cols);
            var parent_col = try allocator.alloc(isize, n);
            defer allocator.free(parent_col);

            @memset(visited_rows, false);
            @memset(visited_cols, false);
            @memset(parent_col, -1);

            // BFS from start_row
            var queue: std.ArrayList(usize) = .{};
            defer queue.deinit(allocator);

            try queue.append(allocator, start_row);
            visited_rows[start_row] = true;

            var found_col: isize = -1;

            outer: while (queue.items.len > 0) {
                const u = queue.orderedRemove(0);

                // Try all columns
                for (0..n) |j| {
                    if (visited_cols[j]) continue;
                    if (@as(i128, @intCast(costs[u][j])) != @as(i128, @intCast(zero_weight))) continue;

                    visited_cols[j] = true;
                    parent_col[j] = @intCast(u);

                    if (col_match[j] == -1) {
                        // Found augmenting path
                        found_col = @intCast(j);
                        break :outer;
                    } else {
                        const next_row: usize = @intCast(col_match[j]);
                        if (!visited_rows[next_row]) {
                            visited_rows[next_row] = true;
                            try queue.append(allocator, next_row);
                        }
                    }
                }
            }

            if (found_col >= 0) {
                // Augment along the path
                var col = found_col;
                while (col >= 0) {
                    const row = parent_col[@intCast(col)];
                    const prev_col = row_match[@intCast(row)];
                    row_match[@intCast(row)] = col;
                    col_match[@intCast(col)] = row;
                    col = prev_col;
                }
            } else {
                // No augmenting path found using zeros - need to adjust costs
                // Find minimum slack
                var min_slack: W = std.math.maxInt(W);
                for (0..n) |i| {
                    if (!visited_rows[i]) continue;
                    for (0..n) |j| {
                        if (visited_cols[j]) continue;
                        if (@as(i128, @intCast(costs[i][j])) < @as(i128, @intCast(min_slack))) {
                            min_slack = costs[i][j];
                        }
                    }
                }

                // Adjust costs
                for (0..n) |i| {
                    for (0..n) |j| {
                        if (visited_rows[i] and !visited_cols[j]) {
                            // Subtract from uncovered rows
                            costs[i][j] = costs[i][j] - min_slack;
                        } else if (!visited_rows[i] and visited_cols[j]) {
                            // Add to covered columns
                            costs[i][j] = costs[i][j] + min_slack;
                        }
                        // Intersection (visited row AND visited col): no change (subtract and add cancel out)
                    }
                }

                // Retry augmentation
                try augment(allocator, costs, row_match, col_match, start_row, zero_weight);
            }
        }
    };
}

// Tests

test "Hungarian: simple 2x2" {
    const allocator = testing.allocator;

    const costs = [_][]const u32{
        &[_]u32{ 1, 2 },
        &[_]u32{ 3, 4 },
    };

    const H = Hungarian(u32);
    var result = try H.run(allocator, &costs, 0);
    defer result.deinit();

    // Optimal: 0->0 (cost 1), 1->1 (cost 4) = 5
    try testing.expectEqual(@as(u32, 5), result.total_cost);
    try testing.expect(result.isMatched(0));
    try testing.expect(result.isMatched(1));
}

test "Hungarian: 3x3 example" {
    const allocator = testing.allocator;

    const costs = [_][]const u32{
        &[_]u32{ 10, 19, 8 },
        &[_]u32{ 15, 15, 13 },
        &[_]u32{ 9, 10, 12 },
    };

    const H = Hungarian(u32);
    var result = try H.run(allocator, &costs, 0);
    defer result.deinit();

    // Optimal assignment: total cost should be minimal
    // One optimal solution: 0->2 (8), 1->1 (15), 2->0 (9) = 32
    // Or: 0->0 (10), 1->2 (13), 2->1 (10) = 33
    // Or: 0->2 (8), 1->0 (15), 2->1 (10) = 33
    // The minimum is 32
    try testing.expectEqual(@as(u32, 32), result.total_cost);
}

test "Hungarian: maximum weight (negated)" {
    const allocator = testing.allocator;

    // For maximum weight, negate the weights
    const profits = [_][]const i32{
        &[_]i32{ 5, 3, 2 },
        &[_]i32{ 4, 6, 3 },
        &[_]i32{ 7, 2, 8 },
    };

    // Negate for min-cost algorithm
    var costs: [3][3]i32 = undefined;
    for (&costs, 0..) |*row, i| {
        for (row, 0..) |*val, j| {
            val.* = -profits[i][j];
        }
    }

    const costs_slice = [_][]const i32{
        &costs[0],
        &costs[1],
        &costs[2],
    };

    const H = Hungarian(i32);
    var result = try H.run(allocator, &costs_slice, 0);
    defer result.deinit();

    // Maximum profit assignment: 0->0 (5), 1->1 (6), 2->2 (8) = 19
    // Result will be negated: -19
    try testing.expectEqual(@as(i32, -19), result.total_cost);
}

test "Hungarian: empty" {
    const allocator = testing.allocator;

    const costs: []const []const u32 = &[_][]const u32{};

    const H = Hungarian(u32);
    var result = try H.run(allocator, costs, 0);
    defer result.deinit();

    try testing.expectEqual(@as(u32, 0), result.total_cost);
    try testing.expectEqual(@as(usize, 0), result.assignment.len);
}

test "Hungarian: single element" {
    const allocator = testing.allocator;

    const costs = [_][]const u32{
        &[_]u32{42},
    };

    const H = Hungarian(u32);
    var result = try H.run(allocator, &costs, 0);
    defer result.deinit();

    try testing.expectEqual(@as(u32, 42), result.total_cost);
    try testing.expectEqual(@as(usize, 1), result.assignment.len);
    try testing.expectEqual(@as(isize, 0), result.assignment[0]);
}
