const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Dinic's algorithm for computing maximum flow in a flow network.
/// Uses level graph construction and blocking flow computation.
///
/// Time: O(V² × E) general case, O(E × √V) for unit capacity networks
/// Space: O(V + E) for level graph and adjacency lists
///
/// **Algorithm**:
/// 1. Construct level graph using BFS (shortest path distances from source)
/// 2. Find blocking flow in level graph using DFS
/// 3. Repeat until no augmenting path exists
///
/// **Properties**:
/// - Faster than Edmonds-Karp for dense graphs
/// - Each phase increases minimum distance to sink
/// - At most O(V) phases
/// - Optimal for bipartite matching: O(E × √V)
///
/// **Use cases**: Maximum bipartite matching, min-cost max-flow, dense networks
pub fn maxFlow(comptime T: type, allocator: Allocator, capacity: []const []const T, source: usize, sink: usize) !T {
    if (capacity.len == 0) return 0;
    const n = capacity.len;
    if (source >= n or sink >= n) return error.InvalidVertex;
    if (source == sink) return 0;

    // Create residual graph
    const residual = try allocator.alloc([]T, n);
    errdefer allocator.free(residual);
    for (residual, 0..) |*row, i| {
        row.* = try allocator.alloc(T, n);
        @memcpy(row.*, capacity[i]);
    }
    defer {
        for (residual) |row| allocator.free(row);
        allocator.free(residual);
    }

    const level = try allocator.alloc(?usize, n);
    defer allocator.free(level);
    const iter = try allocator.alloc(usize, n); // Iterator for DFS
    defer allocator.free(iter);

    var total_flow: T = 0;

    // Repeat while level graph can be constructed
    while (try buildLevelGraph(T, allocator, residual, source, sink, level)) {
        @memset(iter, 0);

        // Find blocking flow
        while (true) {
            const flow = try sendFlow(T, residual, source, sink, level, iter, std.math.maxInt(T));
            if (flow == 0) break;
            total_flow += flow;
        }
    }

    return total_flow;
}

/// Build level graph using BFS. Returns true if sink is reachable.
fn buildLevelGraph(comptime T: type, allocator: Allocator, residual: [][]T, source: usize, sink: usize, level: []?usize) !bool {
    for (level) |*l| l.* = null;
    level[source] = 0;

    var queue = std.ArrayList(usize).init(allocator);
    defer queue.deinit();
    try queue.append(source);

    var read_idx: usize = 0;
    while (read_idx < queue.items.len) : (read_idx += 1) {
        const u = queue.items[read_idx];

        for (residual[u], 0..) |cap, v| {
            if (level[v] == null and cap > 0) {
                level[v] = level[u].? + 1;
                try queue.append(v);
            }
        }
    }

    return level[sink] != null;
}

/// Send flow using DFS on level graph (finds blocking flow).
fn sendFlow(comptime T: type, residual: [][]T, u: usize, sink: usize, level: []?usize, iter: []usize, flow: T) !T {
    if (u == sink) return flow;

    const n = residual.len;
    while (iter[u] < n) : (iter[u] += 1) {
        const v = iter[u];
        if (level[v]) |lv| {
            if (lv == level[u].? + 1 and residual[u][v] > 0) {
                const min_flow = @min(flow, residual[u][v]);
                const pushed = try sendFlow(T, residual, v, sink, level, iter, min_flow);

                if (pushed > 0) {
                    residual[u][v] -= pushed;
                    residual[v][u] += pushed;
                    return pushed;
                }
            }
        }
    }

    return 0;
}

/// Compute maximum matching in bipartite graph.
/// Left partition: [0, left_size), Right partition: [left_size, left_size + right_size)
/// edges[i] = list of right vertices connected to left vertex i.
///
/// Time: O(E × √V) using Dinic's algorithm
/// Space: O(V²) for capacity matrix
///
/// Returns number of matched edges.
pub fn maxBipartiteMatching(allocator: Allocator, left_size: usize, right_size: usize, edges: []const []const usize) !usize {
    const n = left_size + right_size + 2; // +2 for source and sink
    const source = n - 2;
    const sink = n - 1;

    // Build capacity matrix
    var capacity = try allocator.alloc([]u32, n);
    errdefer allocator.free(capacity);
    for (capacity) |*row| {
        row.* = try allocator.alloc(u32, n);
        @memset(row.*, 0);
    }
    defer {
        for (capacity) |row| allocator.free(row);
        allocator.free(capacity);
    }

    // Source to left partition (capacity 1)
    for (0..left_size) |i| {
        capacity[source][i] = 1;
    }

    // Left to right edges (capacity 1)
    for (edges, 0..) |adj, i| {
        for (adj) |j| {
            if (j < right_size) {
                capacity[i][left_size + j] = 1;
            }
        }
    }

    // Right partition to sink (capacity 1)
    for (0..right_size) |j| {
        capacity[left_size + j][sink] = 1;
    }

    // Convert capacity to slices
    var capacity_ptrs = try allocator.alloc([]const u32, n);
    defer allocator.free(capacity_ptrs);
    for (capacity, 0..) |row, i| {
        capacity_ptrs[i] = row;
    }

    const flow = try maxFlow(u32, allocator, capacity_ptrs, source, sink);
    return flow;
}

// ============================================================================
// Tests
// ============================================================================

test "Dinic: basic max flow" {
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

test "Dinic: single edge" {
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

test "Dinic: no path" {
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

test "Dinic: complex network (same as Edmonds-Karp)" {
    const allocator = testing.allocator;

    var capacity = [_][6]u32{
        .{ 0, 16, 13, 0, 0, 0 },
        .{ 0, 0, 10, 12, 0, 0 },
        .{ 0, 4, 0, 0, 14, 0 },
        .{ 0, 0, 9, 0, 0, 20 },
        .{ 0, 0, 0, 7, 0, 4 },
        .{ 0, 0, 0, 0, 0, 0 },
    };
    var capacity_ptrs: [6][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const flow = try maxFlow(u32, allocator, &capacity_ptrs, 0, 5);
    try testing.expectEqual(@as(u32, 23), flow);
}

test "Dinic: bottleneck" {
    const allocator = testing.allocator;

    var capacity = [_][4]u32{
        .{ 0, 100, 0, 0 },
        .{ 0, 0, 1, 0 },
        .{ 0, 0, 0, 100 },
        .{ 0, 0, 0, 0 },
    };
    var capacity_ptrs: [4][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const flow = try maxFlow(u32, allocator, &capacity_ptrs, 0, 3);
    try testing.expectEqual(@as(u32, 1), flow);
}

test "Dinic: unit capacity network" {
    const allocator = testing.allocator;

    // Unit capacities (optimal for Dinic: O(E√V))
    var capacity = [_][5]u32{
        .{ 0, 1, 1, 0, 0 },
        .{ 0, 0, 0, 1, 0 },
        .{ 0, 0, 0, 1, 0 },
        .{ 0, 0, 0, 0, 1 },
        .{ 0, 0, 0, 0, 0 },
    };
    var capacity_ptrs: [5][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const flow = try maxFlow(u32, allocator, &capacity_ptrs, 0, 4);
    try testing.expectEqual(@as(u32, 2), flow);
}

test "Dinic: dense graph" {
    const allocator = testing.allocator;

    // Dense 4-vertex graph
    var capacity = [_][4]u32{
        .{ 0, 10, 10, 10 },
        .{ 0, 0, 10, 10 },
        .{ 0, 0, 0, 10 },
        .{ 0, 0, 0, 0 },
    };
    var capacity_ptrs: [4][]const u32 = undefined;
    for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;

    const flow = try maxFlow(u32, allocator, &capacity_ptrs, 0, 3);
    try testing.expectEqual(@as(u32, 30), flow);
}

test "Dinic: f64 capacities" {
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

test "Dinic: empty graph" {
    const allocator = testing.allocator;
    const capacity: []const []const u32 = &[_][]const u32{};
    const flow = try maxFlow(u32, allocator, capacity, 0, 0);
    try testing.expectEqual(@as(u32, 0), flow);
}

test "maxBipartiteMatching: simple matching" {
    const allocator = testing.allocator;

    // Left: {0, 1}, Right: {0, 1}
    // Edges: 0->0, 0->1, 1->1
    var edges = [_][]const usize{
        &[_]usize{ 0, 1 }, // Left 0 connects to right 0, 1
        &[_]usize{1}, // Left 1 connects to right 1
    };

    const matching = try maxBipartiteMatching(allocator, 2, 2, &edges);
    try testing.expectEqual(@as(usize, 2), matching); // Perfect matching
}

test "maxBipartiteMatching: incomplete matching" {
    const allocator = testing.allocator;

    // Left: {0, 1, 2}, Right: {0, 1}
    // Edges: 0->0, 1->0, 2->1
    var edges = [_][]const usize{
        &[_]usize{0},
        &[_]usize{0},
        &[_]usize{1},
    };

    const matching = try maxBipartiteMatching(allocator, 3, 2, &edges);
    try testing.expectEqual(@as(usize, 2), matching); // Can't match all left vertices
}

test "maxBipartiteMatching: no edges" {
    const allocator = testing.allocator;

    var edges = [_][]const usize{
        &[_]usize{},
        &[_]usize{},
    };

    const matching = try maxBipartiteMatching(allocator, 2, 2, &edges);
    try testing.expectEqual(@as(usize, 0), matching);
}

test "maxBipartiteMatching: complete bipartite" {
    const allocator = testing.allocator;

    // K_{2,3} - complete bipartite graph
    var edges = [_][]const usize{
        &[_]usize{ 0, 1, 2 },
        &[_]usize{ 0, 1, 2 },
    };

    const matching = try maxBipartiteMatching(allocator, 2, 3, &edges);
    try testing.expectEqual(@as(usize, 2), matching); // Limited by smaller partition
}
