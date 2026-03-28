const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Ford-Fulkerson algorithm for computing maximum flow in a flow network.
/// Uses DFS to find augmenting paths.
///
/// Time: O(E × max_flow) where E = number of edges, max_flow = maximum flow value
/// Space: O(V) for DFS stack
///
/// **Algorithm**: Repeatedly find augmenting paths from source to sink using DFS,
/// and augment flow along these paths until no more paths exist.
///
/// **Properties**:
/// - Works on directed graphs with non-negative capacities
/// - Final flow satisfies capacity constraints and flow conservation
/// - Max flow = Min cut (by max-flow min-cut theorem)
///
/// **Use cases**: Network capacity analysis, bipartite matching, circulation with demands
pub fn maxFlow(comptime T: type, allocator: Allocator, capacity: []const []const T, source: usize, sink: usize) !T {
    if (capacity.len == 0) return 0;
    const n = capacity.len;
    if (source >= n or sink >= n) return error.InvalidVertex;
    if (source == sink) return 0;

    // Create residual graph (mutable copy of capacity)
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

    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);
    const parent = try allocator.alloc(?usize, n);
    defer allocator.free(parent);

    var total_flow: T = 0;

    // While there exists an augmenting path from source to sink
    while (true) {
        // Reset visited and parent arrays
        @memset(visited, false);
        for (parent) |*p| p.* = null;

        // Find augmenting path using DFS
        const path_flow = try dfs(T, residual, source, sink, visited, parent, std.math.maxInt(T));
        if (path_flow == 0) break; // No more augmenting paths

        // Update residual capacities along the path
        var v = sink;
        while (parent[v]) |u| {
            residual[u][v] -= path_flow;
            residual[v][u] += path_flow; // Add reverse edge
            v = u;
        }

        total_flow += path_flow;
    }

    return total_flow;
}

/// DFS helper to find augmenting path and return bottleneck capacity.
fn dfs(comptime T: type, residual: [][]T, u: usize, sink: usize, visited: []bool, parent: []?usize, flow: T) !T {
    if (u == sink) return flow;
    visited[u] = true;

    for (residual[u], 0..) |cap, v| {
        if (!visited[v] and cap > 0) {
            const min_flow = @min(flow, cap);
            const path_flow = try dfs(T, residual, v, sink, visited, parent, min_flow);
            if (path_flow > 0) {
                parent[v] = u;
                return path_flow;
            }
        }
    }

    return 0;
}

/// Compute minimum cut from maximum flow.
/// Returns a list of vertices in the source side of the cut.
///
/// Time: O(V + E) for DFS traversal of residual graph
/// Space: O(V) for visited array and result list
pub fn minCut(comptime T: type, allocator: Allocator, capacity: []const []const T, source: usize, sink: usize) ![]usize {
    if (capacity.len == 0) return &[_]usize{};
    const n = capacity.len;
    if (source >= n or sink >= n) return error.InvalidVertex;

    // First compute max flow to get residual graph
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

    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);
    const parent = try allocator.alloc(?usize, n);
    defer allocator.free(parent);

    // Run Ford-Fulkerson to saturation
    while (true) {
        @memset(visited, false);
        for (parent) |*p| p.* = null;
        const path_flow = try dfs(T, residual, source, sink, visited, parent, std.math.maxInt(T));
        if (path_flow == 0) break;

        var v = sink;
        while (parent[v]) |u| {
            residual[u][v] -= path_flow;
            residual[v][u] += path_flow;
            v = u;
        }
    }

    // Find all vertices reachable from source in residual graph
    @memset(visited, false);
    var stack = std.ArrayList(usize).init(allocator);
    defer stack.deinit();
    try stack.append(source);
    visited[source] = true;

    while (stack.items.len > 0) {
        const u = stack.pop();
        for (residual[u], 0..) |cap, v| {
            if (!visited[v] and cap > 0) {
                visited[v] = true;
                try stack.append(v);
            }
        }
    }

    // Collect vertices in source side of cut
    var result = std.ArrayList(usize).init(allocator);
    errdefer result.deinit();
    for (visited, 0..) |vis, i| {
        if (vis) try result.append(i);
    }

    return result.toOwnedSlice();
}

// ============================================================================
// Tests
// ============================================================================

test "Ford-Fulkerson: basic max flow" {
    const allocator = testing.allocator;

    // Simple graph: s -> 1 -> t
    //              s -> 2 -> t
    // Capacities: s-1: 10, 1-t: 10, s-2: 5, 2-t: 5
    var capacity = [_][4]u32{
        .{ 0, 10, 5, 0 }, // s (0)
        .{ 0, 0, 0, 10 }, // 1
        .{ 0, 0, 0, 5 }, // 2
        .{ 0, 0, 0, 0 }, // t (3)
    };

    var capacity_ptrs: [4][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const flow = try maxFlow(u32, allocator, &capacity_ptrs, 0, 3);
    try testing.expectEqual(@as(u32, 15), flow); // 10 + 5 = 15
}

test "Ford-Fulkerson: single edge" {
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

test "Ford-Fulkerson: no path" {
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

test "Ford-Fulkerson: bottleneck" {
    const allocator = testing.allocator;

    // s -> 1 -> 2 -> t with middle edge as bottleneck
    var capacity = [_][4]u32{
        .{ 0, 100, 0, 0 }, // s
        .{ 0, 0, 10, 0 }, // 1 (bottleneck: 1->2 = 10)
        .{ 0, 0, 0, 100 }, // 2
        .{ 0, 0, 0, 0 }, // t
    };
    var capacity_ptrs: [4][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const flow = try maxFlow(u32, allocator, &capacity_ptrs, 0, 3);
    try testing.expectEqual(@as(u32, 10), flow); // Limited by bottleneck
}

test "Ford-Fulkerson: multiple paths" {
    const allocator = testing.allocator;

    // Diamond graph with multiple paths
    var capacity = [_][4]u32{
        .{ 0, 10, 10, 0 }, // s
        .{ 0, 0, 0, 10 }, // 1
        .{ 0, 0, 0, 10 }, // 2
        .{ 0, 0, 0, 0 }, // t
    };
    var capacity_ptrs: [4][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const flow = try maxFlow(u32, allocator, &capacity_ptrs, 0, 3);
    try testing.expectEqual(@as(u32, 20), flow); // Both paths contribute
}

test "Ford-Fulkerson: f64 capacities" {
    const allocator = testing.allocator;

    var capacity = [_][3]f64{
        .{ 0.0, 5.5, 3.3 },
        .{ 0.0, 0.0, 2.2 },
        .{ 0.0, 0.0, 0.0 },
    };
    var capacity_ptrs: [3][]const f64 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const flow = try maxFlow(f64, allocator, &capacity_ptrs, 0, 2);
    try testing.expectApproxEqAbs(@as(f64, 5.5), flow, 1e-6);
}

test "Ford-Fulkerson: source equals sink" {
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

test "Ford-Fulkerson: invalid vertex" {
    const allocator = testing.allocator;

    var capacity = [_][2]u32{
        .{ 0, 10 },
        .{ 0, 0 },
    };
    var capacity_ptrs: [2][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    try testing.expectError(error.InvalidVertex, maxFlow(u32, allocator, &capacity_ptrs, 0, 5));
}

test "Ford-Fulkerson: empty graph" {
    const allocator = testing.allocator;
    const capacity: []const []const u32 = &[_][]const u32{};
    const flow = try maxFlow(u32, allocator, capacity, 0, 0);
    try testing.expectEqual(@as(u32, 0), flow);
}

test "Min-Cut: basic cut" {
    const allocator = testing.allocator;

    var capacity = [_][4]u32{
        .{ 0, 10, 5, 0 },
        .{ 0, 0, 0, 10 },
        .{ 0, 0, 0, 5 },
        .{ 0, 0, 0, 0 },
    };
    var capacity_ptrs: [4][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const cut = try minCut(u32, allocator, &capacity_ptrs, 0, 3);
    defer allocator.free(cut);

    // Source side should contain at least the source vertex
    try testing.expect(cut.len > 0);
    try testing.expect(std.mem.indexOfScalar(usize, cut, 0) != null);
}

test "Min-Cut: single edge cut" {
    const allocator = testing.allocator;

    var capacity = [_][2]u32{
        .{ 0, 10 },
        .{ 0, 0 },
    };
    var capacity_ptrs: [2][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const cut = try minCut(u32, allocator, &capacity_ptrs, 0, 1);
    defer allocator.free(cut);

    try testing.expectEqual(@as(usize, 1), cut.len);
    try testing.expectEqual(@as(usize, 0), cut[0]);
}

test "Min-Cut: no path results in source only" {
    const allocator = testing.allocator;

    var capacity = [_][3]u32{
        .{ 0, 10, 0 },
        .{ 0, 0, 0 },
        .{ 0, 0, 0 },
    };
    var capacity_ptrs: [3][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const cut = try minCut(u32, allocator, &capacity_ptrs, 0, 2);
    defer allocator.free(cut);

    // Only source and vertex 1 (reachable from source) should be in cut
    try testing.expect(cut.len >= 1);
    try testing.expect(std.mem.indexOfScalar(usize, cut, 0) != null);
}
