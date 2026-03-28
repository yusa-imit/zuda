const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Edmonds-Karp algorithm for computing maximum flow in a flow network.
/// Uses BFS to find shortest augmenting paths (in terms of number of edges).
///
/// Time: O(V × E²) where V = vertices, E = edges
/// Space: O(V) for BFS queue
///
/// **Algorithm**: Variation of Ford-Fulkerson that uses BFS instead of DFS.
/// This guarantees polynomial time complexity.
///
/// **Properties**:
/// - Finds shortest augmenting path (fewest edges) in each iteration
/// - Guaranteed O(V × E²) time complexity (vs. potentially exponential for Ford-Fulkerson)
/// - Each augmentation increases shortest path length by at least 1
/// - At most O(V × E) augmenting paths
///
/// **Use cases**: Network flow when guaranteed polynomial time is required
pub fn maxFlow(comptime T: type, allocator: Allocator, capacity: []const []const T, source: usize, sink: usize) !T {
    if (capacity.len == 0) return 0;
    const n = capacity.len;
    if (source >= n or sink >= n) return error.InvalidVertex;
    if (source == sink) return 0;

    // Create residual graph
    var residual = try allocator.alloc([]T, n);
    errdefer allocator.free(residual);
    for (residual, 0..) |*row, i| {
        row.* = try allocator.alloc(T, n);
        @memcpy(row.*, capacity[i]);
    }
    defer {
        for (residual) |row| allocator.free(row);
        allocator.free(residual);
    }

    const parent = try allocator.alloc(?usize, n);
    defer allocator.free(parent);

    var total_flow: T = 0;

    // While there exists an augmenting path (using BFS)
    while (try bfs(T, allocator, residual, source, sink, parent)) {
        // Find bottleneck capacity along the path
        var path_flow: T = std.math.maxInt(T);
        var v = sink;
        while (parent[v]) |u| {
            path_flow = @min(path_flow, residual[u][v]);
            v = u;
        }

        // Update residual capacities
        v = sink;
        while (parent[v]) |u| {
            residual[u][v] -= path_flow;
            residual[v][u] += path_flow;
            v = u;
        }

        total_flow += path_flow;
    }

    return total_flow;
}

/// BFS to find shortest augmenting path. Returns true if path exists.
fn bfs(comptime T: type, allocator: Allocator, residual: [][]T, source: usize, sink: usize, parent: []?usize) !bool {
    const n = residual.len;
    var visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);
    @memset(visited, false);
    for (parent) |*p| p.* = null;

    var queue = std.ArrayList(usize).init(allocator);
    defer queue.deinit();

    try queue.append(source);
    visited[source] = true;

    var read_idx: usize = 0;
    while (read_idx < queue.items.len) : (read_idx += 1) {
        const u = queue.items[read_idx];

        for (residual[u], 0..) |cap, v| {
            if (!visited[v] and cap > 0) {
                visited[v] = true;
                parent[v] = u;
                try queue.append(v);

                if (v == sink) return true; // Early exit when sink reached
            }
        }
    }

    return false; // No path to sink
}

/// Get the actual flow values on each edge after computing max flow.
/// Returns a matrix where result[u][v] = flow from u to v.
///
/// Time: O(V²) to compute flow from residual graph
/// Space: O(V²) for flow matrix
pub fn getFlowMatrix(comptime T: type, allocator: Allocator, capacity: []const []const T, source: usize, sink: usize) ![][]T {
    if (capacity.len == 0) return &[_][]T{};
    const n = capacity.len;

    // First compute max flow
    var residual = try allocator.alloc([]T, n);
    errdefer allocator.free(residual);
    for (residual, 0..) |*row, i| {
        row.* = try allocator.alloc(T, n);
        @memcpy(row.*, capacity[i]);
    }
    defer {
        for (residual) |row| allocator.free(row);
        allocator.free(residual);
    }

    const parent = try allocator.alloc(?usize, n);
    defer allocator.free(parent);

    while (try bfs(T, allocator, residual, source, sink, parent)) {
        var path_flow: T = std.math.maxInt(T);
        var v = sink;
        while (parent[v]) |u| {
            path_flow = @min(path_flow, residual[u][v]);
            v = u;
        }

        v = sink;
        while (parent[v]) |u| {
            residual[u][v] -= path_flow;
            residual[v][u] += path_flow;
            v = u;
        }
    }

    // Flow = capacity - residual capacity
    const flow = try allocator.alloc([]T, n);
    errdefer {
        for (flow) |row| allocator.free(row);
        allocator.free(flow);
    }

    for (flow, 0..) |*row, i| {
        row.* = try allocator.alloc(T, n);
        for (row.*, 0..) |*f, j| {
            f.* = capacity[i][j] - residual[i][j];
        }
    }

    return flow;
}

// ============================================================================
// Tests
// ============================================================================

test "Edmonds-Karp: basic max flow" {
    const allocator = testing.allocator;

    var capacity = [_][4]u32{
        .{ 0, 10, 5, 0 },
        .{ 0, 0, 0, 10 },
        .{ 0, 0, 0, 5 },
        .{ 0, 0, 0, 0 },
    };
    var capacity_ptrs: [4][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const flow = try maxFlow(u32, allocator, &capacity_ptrs, 0, 3);
    try testing.expectEqual(@as(u32, 15), flow);
}

test "Edmonds-Karp: single edge" {
    const allocator = testing.allocator;

    var capacity = [_][2]u32{
        .{ 0, 10 },
        .{ 0, 0 },
    };
    var capacity_ptrs: [2][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const flow = try maxFlow(u32, allocator, &capacity_ptrs, 0, 1);
    try testing.expectEqual(@as(u32, 10), flow);
}

test "Edmonds-Karp: no path" {
    const allocator = testing.allocator;

    var capacity = [_][3]u32{
        .{ 0, 10, 0 },
        .{ 0, 0, 0 },
        .{ 0, 0, 0 },
    };
    var capacity_ptrs: [3][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const flow = try maxFlow(u32, allocator, &capacity_ptrs, 0, 2);
    try testing.expectEqual(@as(u32, 0), flow);
}

test "Edmonds-Karp: complex network" {
    const allocator = testing.allocator;

    // 6-vertex network with multiple paths
    var capacity = [_][6]u32{
        .{ 0, 16, 13, 0, 0, 0 }, // s (0)
        .{ 0, 0, 10, 12, 0, 0 }, // 1
        .{ 0, 4, 0, 0, 14, 0 }, // 2
        .{ 0, 0, 9, 0, 0, 20 }, // 3
        .{ 0, 0, 0, 7, 0, 4 }, // 4
        .{ 0, 0, 0, 0, 0, 0 }, // t (5)
    };
    var capacity_ptrs: [6][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const flow = try maxFlow(u32, allocator, &capacity_ptrs, 0, 5);
    try testing.expectEqual(@as(u32, 23), flow); // Known max flow for this graph
}

test "Edmonds-Karp: bottleneck detection" {
    const allocator = testing.allocator;

    var capacity = [_][4]u32{
        .{ 0, 100, 0, 0 },
        .{ 0, 0, 1, 0 }, // Bottleneck at 1->2
        .{ 0, 0, 0, 100 },
        .{ 0, 0, 0, 0 },
    };
    var capacity_ptrs: [4][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const flow = try maxFlow(u32, allocator, &capacity_ptrs, 0, 3);
    try testing.expectEqual(@as(u32, 1), flow);
}

test "Edmonds-Karp: parallel edges" {
    const allocator = testing.allocator;

    // Multiple edges between same vertices (represented as sum)
    var capacity = [_][3]u32{
        .{ 0, 15, 10 }, // s to 1 and 2
        .{ 0, 0, 5 }, // 1 to 2
        .{ 0, 0, 0 }, // 2 is sink
    };
    var capacity_ptrs: [3][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const flow = try maxFlow(u32, allocator, &capacity_ptrs, 0, 2);
    try testing.expectEqual(@as(u32, 25), flow);
}

test "Edmonds-Karp: f64 capacities" {
    const allocator = testing.allocator;

    var capacity = [_][3]f64{
        .{ 0.0, 7.5, 5.5 },
        .{ 0.0, 0.0, 3.3 },
        .{ 0.0, 0.0, 0.0 },
    };
    var capacity_ptrs: [3][]const f64 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const flow = try maxFlow(f64, allocator, &capacity_ptrs, 0, 2);
    try testing.expectApproxEqAbs(@as(f64, 10.8), flow, 1e-6);
}

test "Edmonds-Karp: source equals sink" {
    const allocator = testing.allocator;

    var capacity = [_][2]u32{
        .{ 0, 10 },
        .{ 0, 0 },
    };
    var capacity_ptrs: [2][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const flow = try maxFlow(u32, allocator, &capacity_ptrs, 0, 0);
    try testing.expectEqual(@as(u32, 0), flow);
}

test "Edmonds-Karp: empty graph" {
    const allocator = testing.allocator;
    const capacity: []const []const u32 = &[_][]const u32{};
    const flow = try maxFlow(u32, allocator, capacity, 0, 0);
    try testing.expectEqual(@as(u32, 0), flow);
}

test "getFlowMatrix: basic flow values" {
    const allocator = testing.allocator;

    var capacity = [_][3]u32{
        .{ 0, 10, 5 },
        .{ 0, 0, 10 },
        .{ 0, 0, 0 },
    };
    var capacity_ptrs: [3][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const flow_matrix = try getFlowMatrix(u32, allocator, &capacity_ptrs, 0, 2);
    defer {
        for (flow_matrix) |row| allocator.free(row);
        allocator.free(flow_matrix);
    }

    // Verify flow conservation (except at source/sink)
    var flow_in: u32 = 0;
    var flow_out: u32 = 0;
    for (flow_matrix, 0..) |row, i| {
        for (row, 0..) |f, j| {
            if (i == 1) flow_out += f;
            if (j == 1) flow_in += f;
        }
    }
    try testing.expectEqual(flow_in, flow_out); // Flow conservation at vertex 1
}

test "getFlowMatrix: verify capacity constraints" {
    const allocator = testing.allocator;

    var capacity = [_][4]u32{
        .{ 0, 10, 5, 0 },
        .{ 0, 0, 0, 10 },
        .{ 0, 0, 0, 5 },
        .{ 0, 0, 0, 0 },
    };
    var capacity_ptrs: [4][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const flow_matrix = try getFlowMatrix(u32, allocator, &capacity_ptrs, 0, 3);
    defer {
        for (flow_matrix) |row| allocator.free(row);
        allocator.free(flow_matrix);
    }

    // Verify flow doesn't exceed capacity
    for (flow_matrix, 0..) |row, i| {
        for (row, 0..) |f, j| {
            try testing.expect(f <= capacity[i][j]);
        }
    }
}
