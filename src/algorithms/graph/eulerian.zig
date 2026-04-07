const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;

/// Eulerian Path and Circuit Algorithms
///
/// An Eulerian path visits every edge exactly once. An Eulerian circuit is an Eulerian path
/// that starts and ends at the same vertex.
///
/// **Existence Conditions**:
/// - **Eulerian Circuit**: All vertices have even degree (for undirected graphs)
/// - **Eulerian Path**: Exactly 0 or 2 vertices have odd degree (for undirected graphs)
/// - **Directed Graphs**: Similar conditions with in-degree and out-degree
///
/// **Algorithm**: Hierholzer's algorithm with modifications
/// - Build adjacency list
/// - Check degree conditions
/// - Perform DFS with edge removal
/// - Reconstruct path from stack
///
/// **Use Cases**:
/// - Route planning (Chinese Postman Problem)
/// - DNA sequence assembly (de Bruijn graphs)
/// - Network traversal (visiting all edges)
/// - Mathematical puzzles (Seven Bridges of Königsberg)
/// - Maze solving algorithms
///
/// **Time Complexity**: O(V + E) where V = vertices, E = edges
/// **Space Complexity**: O(V + E) for adjacency list and path storage

/// Graph type for Eulerian path algorithms
pub const GraphType = enum {
    undirected,
    directed,
};

/// Result type for Eulerian path queries
pub const EulerianType = enum {
    none, // No Eulerian path or circuit
    path, // Eulerian path exists (not a circuit)
    circuit, // Eulerian circuit exists
};

/// Check if a graph has an Eulerian path or circuit
/// Time: O(V + E) | Space: O(V)
pub fn hasEulerianPath(
    comptime T: type,
    allocator: Allocator,
    edges: []const [2]T,
    graph_type: GraphType,
) !EulerianType {
    if (edges.len == 0) return .none;

    // Build degree map
    var degree = AutoHashMap(T, isize).init(allocator);
    defer degree.deinit();

    if (graph_type == .undirected) {
        // For undirected: count degree of each vertex
        for (edges) |edge| {
            const u = edge[0];
            const v = edge[1];

            const du = degree.get(u) orelse 0;
            const dv = degree.get(v) orelse 0;

            try degree.put(u, du + 1);
            if (u != v) { // Avoid counting self-loop twice
                try degree.put(v, dv + 1);
            }
        }

        // Count vertices with odd degree
        var odd_count: usize = 0;
        var it = degree.iterator();
        while (it.next()) |entry| {
            if (@mod(entry.value_ptr.*, 2) == 1) {
                odd_count += 1;
            }
        }

        // Eulerian circuit: all even degrees
        if (odd_count == 0) return .circuit;
        // Eulerian path: exactly 2 odd degrees
        if (odd_count == 2) return .path;
        // No Eulerian path
        return .none;
    } else {
        // For directed: check in-degree and out-degree
        for (edges) |edge| {
            const u = edge[0];
            const v = edge[1];

            const du = degree.get(u) orelse 0;
            const dv = degree.get(v) orelse 0;

            try degree.put(u, du + 1); // out-degree
            try degree.put(v, dv - 1); // in-degree (negative)
        }

        // Count imbalance
        var start_count: usize = 0; // out-degree > in-degree
        var end_count: usize = 0; // in-degree > out-degree
        var it = degree.iterator();
        while (it.next()) |entry| {
            const diff = entry.value_ptr.*;
            if (diff > 0) {
                if (diff == 1) {
                    start_count += 1;
                } else {
                    return .none; // Too much imbalance
                }
            } else if (diff < 0) {
                if (diff == -1) {
                    end_count += 1;
                } else {
                    return .none; // Too much imbalance
                }
            }
        }

        // Eulerian circuit: all balanced
        if (start_count == 0 and end_count == 0) return .circuit;
        // Eulerian path: one start, one end
        if (start_count == 1 and end_count == 1) return .path;
        // No Eulerian path
        return .none;
    }
}

/// Find an Eulerian path or circuit using Hierholzer's algorithm
/// Time: O(V + E) | Space: O(V + E)
pub fn findEulerianPath(
    comptime T: type,
    allocator: Allocator,
    edges: []const [2]T,
    graph_type: GraphType,
) !?ArrayList(T) {
    if (edges.len == 0) return null;

    // Check if Eulerian path exists
    const euler_type = try hasEulerianPath(T, allocator, edges, graph_type);
    if (euler_type == .none) return null;

    // Build adjacency list (as ArrayList of neighbors)
    var adj = AutoHashMap(T, ArrayList(T)).init(allocator);
    defer {
        var it = adj.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit();
        }
        adj.deinit();
    }

    // Build degree map to find start vertex
    var degree = AutoHashMap(T, isize).init(allocator);
    defer degree.deinit();

    for (edges) |edge| {
        const u = edge[0];
        const v = edge[1];

        // Add edge to adjacency list
        var entry = try adj.getOrPut(u);
        if (!entry.found_existing) {
            entry.value_ptr.* = ArrayList(T).init(allocator);
        }
        try entry.value_ptr.append(v);

        if (graph_type == .undirected and u != v) {
            // Add reverse edge for undirected graph
            var entry2 = try adj.getOrPut(v);
            if (!entry2.found_existing) {
                entry2.value_ptr.* = ArrayList(T).init(allocator);
            }
            try entry2.value_ptr.append(u);
        }

        // Track degrees
        if (graph_type == .undirected) {
            const du = degree.get(u) orelse 0;
            const dv = degree.get(v) orelse 0;
            try degree.put(u, du + 1);
            if (u != v) {
                try degree.put(v, dv + 1);
            }
        } else {
            const du = degree.get(u) orelse 0;
            const dv = degree.get(v) orelse 0;
            try degree.put(u, du + 1);
            try degree.put(v, dv - 1);
        }
    }

    // Find start vertex
    var start: T = edges[0][0];
    if (euler_type == .path) {
        // Find vertex with odd degree (undirected) or out-degree > in-degree (directed)
        var it = degree.iterator();
        while (it.next()) |entry| {
            if (graph_type == .undirected) {
                if (@mod(entry.value_ptr.*, 2) == 1) {
                    start = entry.key_ptr.*;
                    break;
                }
            } else {
                if (entry.value_ptr.* > 0) {
                    start = entry.key_ptr.*;
                    break;
                }
            }
        }
    }

    // Hierholzer's algorithm
    var stack = ArrayList(T).init(allocator);
    defer stack.deinit();

    var path = ArrayList(T).init(allocator);
    errdefer path.deinit();

    try stack.append(start);

    while (stack.items.len > 0) {
        const v = stack.items[stack.items.len - 1];
        var neighbors = adj.getPtr(v);

        if (neighbors != null and neighbors.?.items.len > 0) {
            // Follow an edge
            const u = neighbors.?.pop();
            try stack.append(u);

            // Remove reverse edge if undirected
            if (graph_type == .undirected) {
                if (adj.getPtr(u)) |rev_neighbors| {
                    // Find and remove the edge back to v
                    for (rev_neighbors.items, 0..) |neighbor, i| {
                        if (std.meta.eql(neighbor, v)) {
                            _ = rev_neighbors.swapRemove(i);
                            break;
                        }
                    }
                }
            }
        } else {
            // No more edges, add to path
            _ = stack.pop();
            try path.append(v);
        }
    }

    // Reverse path (Hierholzer's produces reversed order)
    std.mem.reverse(T, path.items);

    return path;
}

/// Count the number of edges in the Eulerian path (should equal number of edges in graph)
/// Time: O(1) | Space: O(1)
pub fn pathLength(comptime T: type, path: []const T) usize {
    if (path.len == 0) return 0;
    return path.len - 1; // Number of edges = vertices - 1
}

/// Verify that a path is a valid Eulerian path for the given edges
/// Time: O(E) | Space: O(E)
pub fn isValidEulerianPath(
    comptime T: type,
    allocator: Allocator,
    path: []const T,
    edges: []const [2]T,
    graph_type: GraphType,
) !bool {
    if (path.len == 0) return edges.len == 0;
    if (path.len - 1 != edges.len) return false;

    // Build edge multiset from original edges
    var edge_set = AutoHashMap([2]T, usize).init(allocator);
    defer edge_set.deinit();

    for (edges) |edge| {
        const count = edge_set.get(edge) orelse 0;
        try edge_set.put(edge, count + 1);

        if (graph_type == .undirected) {
            // Also add reversed edge
            const rev_edge = [2]T{ edge[1], edge[0] };
            const rev_count = edge_set.get(rev_edge) orelse 0;
            try edge_set.put(rev_edge, rev_count + 1);
        }
    }

    // Check each edge in path exists in original edges
    for (0..path.len - 1) |i| {
        const edge = [2]T{ path[i], path[i + 1] };
        const count = edge_set.get(edge) orelse return false;
        if (count == 0) return false;
        try edge_set.put(edge, count - 1);
    }

    // Check all edges were used
    var it = edge_set.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.* != 0) return false;
    }

    return true;
}

// ============================================================================
// Tests
// ============================================================================

test "eulerian - basic undirected circuit" {
    const allocator = std.testing.allocator;

    // Triangle: 0-1-2-0 (all vertices have degree 2)
    const edges = [_][2]u32{
        .{ 0, 1 },
        .{ 1, 2 },
        .{ 2, 0 },
    };

    const result = try hasEulerianPath(u32, allocator, &edges, .undirected);
    try std.testing.expectEqual(EulerianType.circuit, result);
}

test "eulerian - basic undirected path" {
    const allocator = std.testing.allocator;

    // Line: 0-1-2 (vertices 0 and 2 have degree 1, vertex 1 has degree 2)
    const edges = [_][2]u32{
        .{ 0, 1 },
        .{ 1, 2 },
    };

    const result = try hasEulerianPath(u32, allocator, &edges, .undirected);
    try std.testing.expectEqual(EulerianType.path, result);
}

test "eulerian - no path undirected" {
    const allocator = std.testing.allocator;

    // K4 graph has all vertices with degree 3 (odd)
    const edges = [_][2]u32{
        .{ 0, 1 },
        .{ 0, 2 },
        .{ 0, 3 },
        .{ 1, 2 },
        .{ 1, 3 },
        .{ 2, 3 },
    };

    const result = try hasEulerianPath(u32, allocator, &edges, .undirected);
    try std.testing.expectEqual(EulerianType.none, result);
}

test "eulerian - directed circuit" {
    const allocator = std.testing.allocator;

    // Directed triangle: 0→1→2→0
    const edges = [_][2]u32{
        .{ 0, 1 },
        .{ 1, 2 },
        .{ 2, 0 },
    };

    const result = try hasEulerianPath(u32, allocator, &edges, .directed);
    try std.testing.expectEqual(EulerianType.circuit, result);
}

test "eulerian - directed path" {
    const allocator = std.testing.allocator;

    // Directed line: 0→1→2
    const edges = [_][2]u32{
        .{ 0, 1 },
        .{ 1, 2 },
    };

    const result = try hasEulerianPath(u32, allocator, &edges, .directed);
    try std.testing.expectEqual(EulerianType.path, result);
}

test "eulerian - find path undirected" {
    const allocator = std.testing.allocator;

    // Triangle: 0-1-2-0
    const edges = [_][2]u32{
        .{ 0, 1 },
        .{ 1, 2 },
        .{ 2, 0 },
    };

    const path = try findEulerianPath(u32, allocator, &edges, .undirected);
    try std.testing.expect(path != null);
    defer path.?.deinit();

    try std.testing.expectEqual(@as(usize, 4), path.?.items.len); // 4 vertices for 3 edges
    try std.testing.expectEqual(@as(usize, 3), pathLength(u32, path.?.items)); // 3 edges
}

test "eulerian - find path directed" {
    const allocator = std.testing.allocator;

    // Directed path: 0→1→2→3
    const edges = [_][2]u32{
        .{ 0, 1 },
        .{ 1, 2 },
        .{ 2, 3 },
    };

    const path = try findEulerianPath(u32, allocator, &edges, .directed);
    try std.testing.expect(path != null);
    defer path.?.deinit();

    try std.testing.expectEqual(@as(usize, 4), path.?.items.len);
    try std.testing.expectEqual(@as(u32, 0), path.?.items[0]);
    try std.testing.expectEqual(@as(u32, 3), path.?.items[3]);
}

test "eulerian - verify path" {
    const allocator = std.testing.allocator;

    const edges = [_][2]u32{
        .{ 0, 1 },
        .{ 1, 2 },
        .{ 2, 0 },
    };

    const path = try findEulerianPath(u32, allocator, &edges, .undirected);
    try std.testing.expect(path != null);
    defer path.?.deinit();

    const valid = try isValidEulerianPath(u32, allocator, path.?.items, &edges, .undirected);
    try std.testing.expect(valid);
}

test "eulerian - square undirected" {
    const allocator = std.testing.allocator;

    // Square: 0-1-2-3-0 (all vertices have degree 2)
    const edges = [_][2]u32{
        .{ 0, 1 },
        .{ 1, 2 },
        .{ 2, 3 },
        .{ 3, 0 },
    };

    const result = try hasEulerianPath(u32, allocator, &edges, .undirected);
    try std.testing.expectEqual(EulerianType.circuit, result);

    const path = try findEulerianPath(u32, allocator, &edges, .undirected);
    try std.testing.expect(path != null);
    defer path.?.deinit();

    const valid = try isValidEulerianPath(u32, allocator, path.?.items, &edges, .undirected);
    try std.testing.expect(valid);
}

test "eulerian - complete graph K3" {
    const allocator = std.testing.allocator;

    // K3: all vertices connected (all have degree 2)
    const edges = [_][2]u32{
        .{ 0, 1 },
        .{ 1, 2 },
        .{ 2, 0 },
    };

    const result = try hasEulerianPath(u32, allocator, &edges, .undirected);
    try std.testing.expectEqual(EulerianType.circuit, result);
}

test "eulerian - self loop" {
    const allocator = std.testing.allocator;

    // Self loop: 0-0
    const edges = [_][2]u32{
        .{ 0, 0 },
    };

    const result = try hasEulerianPath(u32, allocator, &edges, .undirected);
    try std.testing.expectEqual(EulerianType.circuit, result);

    const path = try findEulerianPath(u32, allocator, &edges, .undirected);
    try std.testing.expect(path != null);
    defer path.?.deinit();

    try std.testing.expectEqual(@as(usize, 2), path.?.items.len); // 0 -> 0
}

test "eulerian - empty graph" {
    const allocator = std.testing.allocator;

    const edges = [_][2]u32{};

    const result = try hasEulerianPath(u32, allocator, &edges, .undirected);
    try std.testing.expectEqual(EulerianType.none, result);

    const path = try findEulerianPath(u32, allocator, &edges, .undirected);
    try std.testing.expect(path == null);
}

test "eulerian - single edge" {
    const allocator = std.testing.allocator;

    const edges = [_][2]u32{
        .{ 0, 1 },
    };

    const result = try hasEulerianPath(u32, allocator, &edges, .undirected);
    try std.testing.expectEqual(EulerianType.path, result);

    const path = try findEulerianPath(u32, allocator, &edges, .undirected);
    try std.testing.expect(path != null);
    defer path.?.deinit();

    try std.testing.expectEqual(@as(usize, 2), path.?.items.len);
}

test "eulerian - complex undirected" {
    const allocator = std.testing.allocator;

    // Graph with Eulerian circuit: pentagon with all diagonals
    // Each vertex has degree 4 (connected to 4 others)
    const edges = [_][2]u32{
        .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 4 }, .{ 4, 0 }, // pentagon
        .{ 0, 2 }, .{ 1, 3 }, .{ 2, 4 }, .{ 3, 0 }, .{ 4, 1 }, // diagonals
    };

    const result = try hasEulerianPath(u32, allocator, &edges, .undirected);
    try std.testing.expectEqual(EulerianType.circuit, result);

    const path = try findEulerianPath(u32, allocator, &edges, .undirected);
    try std.testing.expect(path != null);
    defer path.?.deinit();

    try std.testing.expectEqual(@as(usize, 11), path.?.items.len); // 10 edges + 1
    const valid = try isValidEulerianPath(u32, allocator, path.?.items, &edges, .undirected);
    try std.testing.expect(valid);
}

test "eulerian - directed complex" {
    const allocator = std.testing.allocator;

    // Directed graph with circuit: 0→1→2→0 and 0→3→2
    const edges = [_][2]u32{
        .{ 0, 1 },
        .{ 1, 2 },
        .{ 2, 0 },
        .{ 0, 3 },
        .{ 3, 2 },
    };

    // Check degrees: 0(out:2, in:1), 1(out:1, in:1), 2(out:1, in:2), 3(out:1, in:1)
    // Not balanced → should be path
    const result = try hasEulerianPath(u32, allocator, &edges, .directed);
    try std.testing.expectEqual(EulerianType.path, result);
}

test "eulerian - large undirected path" {
    const allocator = std.testing.allocator;

    // Build a long path: 0-1-2-...-99
    var edges = try allocator.alloc([2]u32, 99);
    defer allocator.free(edges);

    for (0..99) |i| {
        edges[i] = .{ @intCast(i), @intCast(i + 1) };
    }

    const result = try hasEulerianPath(u32, allocator, edges, .undirected);
    try std.testing.expectEqual(EulerianType.path, result);

    const path = try findEulerianPath(u32, allocator, edges, .undirected);
    try std.testing.expect(path != null);
    defer path.?.deinit();

    try std.testing.expectEqual(@as(usize, 100), path.?.items.len);
}

test "eulerian - memory safety" {
    const allocator = std.testing.allocator;

    const edges = [_][2]u32{
        .{ 0, 1 },
        .{ 1, 2 },
        .{ 2, 0 },
    };

    // Multiple iterations to detect leaks
    for (0..10) |_| {
        _ = try hasEulerianPath(u32, allocator, &edges, .undirected);
        const path = try findEulerianPath(u32, allocator, &edges, .undirected);
        if (path) |p| {
            _ = try isValidEulerianPath(u32, allocator, p.items, &edges, .undirected);
            p.deinit();
        }
    }
}
