const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

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

    const residual = try residual_init(T, allocator, capacity);
    defer residual_free(T, allocator, residual);

    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);

    const parent = try allocator.alloc(?usize, n);
    defer allocator.free(parent);

    return saturate(T, residual, visited, parent, source, sink);
}

/// Copy `capacity` into a freshly allocated residual matrix. Frees everything on failure.
fn residual_init(comptime T: type, allocator: Allocator, capacity: []const []const T) ![][]T {
    const residual = try allocator.alloc([]T, capacity.len);
    var filled: usize = 0;
    errdefer residual_free_rows(T, allocator, residual, filled);

    while (filled < capacity.len) : (filled += 1) {
        residual[filled] = try allocator.alloc(T, capacity.len);
        @memcpy(residual[filled], capacity[filled]);
    }
    return residual;
}

fn residual_free(comptime T: type, allocator: Allocator, residual: [][]T) void {
    residual_free_rows(T, allocator, residual, residual.len);
}

/// Free the first `filled` rows (the only initialized ones), then the row array.
fn residual_free_rows(comptime T: type, allocator: Allocator, residual: [][]T, filled: usize) void {
    for (residual[0..filled]) |row| allocator.free(row);
    allocator.free(residual);
}

/// Augment along DFS paths until none remain; leaves `residual` saturated. Returns the flow.
fn saturate(
    comptime T: type,
    residual: [][]T,
    visited: []bool,
    parent: []?usize,
    source: usize,
    sink: usize,
) !T {
    assert(source != sink);
    assert(visited.len == residual.len);
    assert(parent.len == residual.len);

    // Floats have no maxInt; an infinite bottleneck is capped by the first edge taken.
    const unbounded: T = switch (@typeInfo(T)) {
        .float => std.math.inf(T),
        else => std.math.maxInt(T),
    };
    var total_flow: T = 0;

    // While there exists an augmenting path from source to sink
    while (true) {
        @memset(visited, false);
        for (parent) |*p| p.* = null;

        const path_flow = try dfs(T, residual, source, sink, visited, parent, unbounded);
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

    const residual = try residual_init(T, allocator, capacity);
    defer residual_free(T, allocator, residual);

    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);

    const parent = try allocator.alloc(?usize, n);
    defer allocator.free(parent);

    // Run Ford-Fulkerson to saturation; source == sink has no flow and cuts nothing.
    if (source != sink) _ = try saturate(T, residual, visited, parent, source, sink);

    // Find all vertices reachable from source in residual graph
    @memset(visited, false);
    var stack: std.ArrayList(usize) = .empty;
    defer stack.deinit(allocator);

    try stack.append(allocator, source);
    visited[source] = true;

    while (stack.pop()) |u| {
        for (residual[u], 0..) |cap, v| {
            if (!visited[v] and cap > 0) {
                visited[v] = true;
                try stack.append(allocator, v);
            }
        }
    }

    // Collect vertices in source side of cut
    var result: std.ArrayList(usize) = .empty;
    errdefer result.deinit(allocator);

    for (visited, 0..) |vis, i| {
        if (vis) try result.append(allocator, i);
    }

    return result.toOwnedSlice(allocator);
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
    try testing.expect(std.mem.findScalar(usize, cut, 0) != null);
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
    try testing.expect(std.mem.findScalar(usize, cut, 0) != null);
}

test "Ford-Fulkerson: allocation failure at every step leaks nothing" {
    try testing.checkAllAllocationFailures(testing.allocator, flow_and_cut_once, .{});
}

fn flow_and_cut_once(gpa: Allocator) !void {
    const rows = [_][4]u32{
        .{ 0, 10, 5, 0 },
        .{ 0, 0, 0, 10 },
        .{ 0, 0, 0, 5 },
        .{ 0, 0, 0, 0 },
    };
    var capacity: [4][]const u32 = undefined;
    for (&rows, 0..) |*row, i| capacity[i] = row;

    try testing.expectEqual(@as(u32, 15), try maxFlow(u32, gpa, &capacity, 0, 3));

    const cut = try minCut(u32, gpa, &capacity, 0, 3);
    defer gpa.free(cut);

    try testing.expectEqualSlices(usize, &[_]usize{0}, cut);
}

test "Ford-Fulkerson: f64 bottleneck is the smallest capacity on the path" {
    const rows = [_][3]f64{
        .{ 0, 2.5, 0 },
        .{ 0, 0, 1.5 },
        .{ 0, 0, 0 },
    };
    var capacity: [3][]const f64 = undefined;
    for (&rows, 0..) |*row, i| capacity[i] = row;

    try testing.expectEqual(@as(f64, 1.5), try maxFlow(f64, testing.allocator, &capacity, 0, 2));
}
