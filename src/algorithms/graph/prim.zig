const std = @import("std");
const testing = std.testing;

/// Prim's Minimum Spanning Tree algorithm.
///
/// Computes a minimum spanning tree (MST) of an undirected, weighted graph by
/// growing a single tree from an arbitrary starting vertex using a priority queue.
///
/// Time: O(E log V) with binary heap priority queue
/// Space: O(V + E) for adjacency list representation and priority queue
///
/// Generic over:
/// - V: Vertex type (must be hashable and comparable)
/// - W: Weight type (must be comparable, e.g., i32, f64)
///
/// Algorithm:
/// 1. Initialize: Start from arbitrary vertex, mark as visited
/// 2. Add all edges from current vertex to priority queue
/// 3. Extract minimum edge from queue
/// 4. If edge leads to unvisited vertex:
///    - Add edge to MST
///    - Mark target vertex as visited
///    - Add all edges from target to queue
/// 5. Repeat until all vertices visited or queue empty
///
/// Difference from Kruskal:
/// - Prim: Grows a single tree (vertex-centric, needs adjacency list)
/// - Kruskal: Merges forests (edge-centric, works with edge list)
///
/// Note: If the graph is disconnected, returns MST of the component containing start_vertex.
pub fn Prim(comptime V: type, comptime W: type) type {
    return struct {
        const Self = @This();

        pub const Edge = struct {
            from: V,
            to: V,
            weight: W,
        };

        pub const Neighbor = struct {
            neighbor: V,
            weight: W,
        };

        pub const Result = struct {
            edges: []Edge,
            total_weight: W,

            pub fn deinit(self: *Result, allocator: std.mem.Allocator) void {
                allocator.free(self.edges);
            }
        };

        /// Priority queue entry for Prim's algorithm
        const QueueEntry = struct {
            vertex: V,
            from: V,
            weight: W,

            fn lessThan(_: void, a: QueueEntry, b: QueueEntry) std.math.Order {
                return switch (@typeInfo(W)) {
                    .int, .comptime_int => if (a.weight < b.weight) .lt else if (a.weight > b.weight) .gt else .eq,
                    .float, .comptime_float => if (a.weight < b.weight) .lt else if (a.weight > b.weight) .gt else .eq,
                    else => @compileError("Weight type must be numeric (int or float)"),
                };
            }
        };

        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .allocator = allocator };
        }

        /// Compute the minimum spanning tree using Prim's algorithm.
        ///
        /// Time: O(E log V) — each edge is added/removed from priority queue once
        /// Space: O(V + E) — adjacency list and priority queue
        ///
        /// Parameters:
        /// - adjacency_list: Map from vertex to list of (neighbor, weight) pairs
        /// - start_vertex: Starting vertex (arbitrary choice, affects only the tree structure)
        ///
        /// Returns:
        /// - Result containing MST edges and total weight
        ///
        /// Note: If the graph is disconnected, returns MST of only the component containing start_vertex.
        pub fn run(
            self: Self,
            adjacency_list: *const std.AutoHashMap(V, std.ArrayList(Neighbor)),
            start_vertex: V,
        ) !Result {
            if (adjacency_list.count() == 0) {
                return Result{
                    .edges = try self.allocator.alloc(Edge, 0),
                    .total_weight = 0,
                };
            }

            // Check if start vertex exists in graph
            if (!adjacency_list.contains(start_vertex)) {
                return error.StartVertexNotFound;
            }

            // Track visited vertices
            var visited = std.AutoHashMap(V, void).init(self.allocator);
            defer visited.deinit();

            // Priority queue for edges (min-heap by weight)
            var pq = std.PriorityQueue(QueueEntry, void, QueueEntry.lessThan).init(self.allocator, {});
            defer pq.deinit();

            // MST edges
            var mst_edges: std.ArrayList(Edge) = .{};
            errdefer mst_edges.deinit(self.allocator);

            var total_weight: W = 0;

            // Start from the given vertex
            try visited.put(start_vertex, {});

            // Add all edges from start vertex to priority queue
            if (adjacency_list.get(start_vertex)) |neighbors| {
                for (neighbors.items) |neighbor_info| {
                    try pq.add(.{
                        .vertex = neighbor_info.neighbor,
                        .from = start_vertex,
                        .weight = neighbor_info.weight,
                    });
                }
            }

            // Grow MST
            while (pq.removeOrNull()) |entry| {
                // Skip if vertex already visited
                if (visited.contains(entry.vertex)) {
                    continue;
                }

                // Add edge to MST
                try mst_edges.append(self.allocator, .{
                    .from = entry.from,
                    .to = entry.vertex,
                    .weight = entry.weight,
                });

                total_weight = switch (@typeInfo(W)) {
                    .int, .comptime_int => total_weight + entry.weight,
                    .float, .comptime_float => total_weight + entry.weight,
                    else => unreachable,
                };

                // Mark vertex as visited
                try visited.put(entry.vertex, {});

                // Add edges from newly visited vertex
                if (adjacency_list.get(entry.vertex)) |neighbors| {
                    for (neighbors.items) |neighbor_info| {
                        if (!visited.contains(neighbor_info.neighbor)) {
                            try pq.add(.{
                                .vertex = neighbor_info.neighbor,
                                .from = entry.vertex,
                                .weight = neighbor_info.weight,
                            });
                        }
                    }
                }
            }

            return Result{
                .edges = try mst_edges.toOwnedSlice(self.allocator),
                .total_weight = total_weight,
            };
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "Prim: basic MST" {
    const P = Prim(u32, i32);
    var prim = P.init(testing.allocator);

    // Triangle graph:
    //     1
    //    / \
    //  5/   \2
    //  /     \
    // 0-------2
    //     3
    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(P.Neighbor)).init(testing.allocator);
    defer {
        var iter = adjacency_list.valueIterator();
        while (iter.next()) |list| {
            list.deinit(testing.allocator);
        }
        adjacency_list.deinit();
    }

    // Vertex 0
    var v0_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v0_neighbors.append(testing.allocator, .{ .neighbor = 1, .weight = 5 });
    try v0_neighbors.append(testing.allocator, .{ .neighbor = 2, .weight = 3 });
    try adjacency_list.put(0, v0_neighbors);

    // Vertex 1
    var v1_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v1_neighbors.append(testing.allocator, .{ .neighbor = 0, .weight = 5 });
    try v1_neighbors.append(testing.allocator, .{ .neighbor = 2, .weight = 2 });
    try adjacency_list.put(1, v1_neighbors);

    // Vertex 2
    var v2_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v2_neighbors.append(testing.allocator, .{ .neighbor = 0, .weight = 3 });
    try v2_neighbors.append(testing.allocator, .{ .neighbor = 1, .weight = 2 });
    try adjacency_list.put(2, v2_neighbors);

    var result = try prim.run(&adjacency_list, 0);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), result.edges.len); // V-1 = 3-1 = 2
    try testing.expectEqual(@as(i32, 5), result.total_weight); // 2 + 3 = 5
}

test "Prim: single vertex" {
    const P = Prim(u32, i32);
    var prim = P.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(P.Neighbor)).init(testing.allocator);
    defer adjacency_list.deinit();

    // Single vertex with no edges
    var v0_neighbors: std.ArrayList(P.Neighbor) = .{};
    defer v0_neighbors.deinit(testing.allocator);
    try adjacency_list.put(0, v0_neighbors);

    var result = try prim.run(&adjacency_list, 0);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 0), result.edges.len);
    try testing.expectEqual(@as(i32, 0), result.total_weight);
}

test "Prim: two vertices" {
    const P = Prim(u32, i32);
    var prim = P.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(P.Neighbor)).init(testing.allocator);
    defer {
        var iter = adjacency_list.valueIterator();
        while (iter.next()) |list| {
            list.deinit(testing.allocator);
        }
        adjacency_list.deinit();
    }

    // 0 -- 10 -- 1
    var v0_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v0_neighbors.append(testing.allocator, .{ .neighbor = 1, .weight = 10 });
    try adjacency_list.put(0, v0_neighbors);

    var v1_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v1_neighbors.append(testing.allocator, .{ .neighbor = 0, .weight = 10 });
    try adjacency_list.put(1, v1_neighbors);

    var result = try prim.run(&adjacency_list, 0);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), result.edges.len);
    try testing.expectEqual(@as(i32, 10), result.total_weight);
}

test "Prim: complete graph K4" {
    const P = Prim(u32, i32);
    var prim = P.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(P.Neighbor)).init(testing.allocator);
    defer {
        var iter = adjacency_list.valueIterator();
        while (iter.next()) |list| {
            list.deinit(testing.allocator);
        }
        adjacency_list.deinit();
    }

    // Complete graph on 4 vertices with same weights as Kruskal test
    // Vertex 0
    var v0_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v0_neighbors.append(testing.allocator, .{ .neighbor = 1, .weight = 1 });
    try v0_neighbors.append(testing.allocator, .{ .neighbor = 2, .weight = 4 });
    try v0_neighbors.append(testing.allocator, .{ .neighbor = 3, .weight = 3 });
    try adjacency_list.put(0, v0_neighbors);

    // Vertex 1
    var v1_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v1_neighbors.append(testing.allocator, .{ .neighbor = 0, .weight = 1 });
    try v1_neighbors.append(testing.allocator, .{ .neighbor = 2, .weight = 2 });
    try v1_neighbors.append(testing.allocator, .{ .neighbor = 3, .weight = 5 });
    try adjacency_list.put(1, v1_neighbors);

    // Vertex 2
    var v2_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v2_neighbors.append(testing.allocator, .{ .neighbor = 0, .weight = 4 });
    try v2_neighbors.append(testing.allocator, .{ .neighbor = 1, .weight = 2 });
    try v2_neighbors.append(testing.allocator, .{ .neighbor = 3, .weight = 6 });
    try adjacency_list.put(2, v2_neighbors);

    // Vertex 3
    var v3_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v3_neighbors.append(testing.allocator, .{ .neighbor = 0, .weight = 3 });
    try v3_neighbors.append(testing.allocator, .{ .neighbor = 1, .weight = 5 });
    try v3_neighbors.append(testing.allocator, .{ .neighbor = 2, .weight = 6 });
    try adjacency_list.put(3, v3_neighbors);

    var result = try prim.run(&adjacency_list, 0);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 3), result.edges.len); // V-1 = 4-1 = 3
    try testing.expectEqual(@as(i32, 6), result.total_weight); // Same as Kruskal: 1+2+3 = 6
}

test "Prim: zero weight edges" {
    const P = Prim(u32, i32);
    var prim = P.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(P.Neighbor)).init(testing.allocator);
    defer {
        var iter = adjacency_list.valueIterator();
        while (iter.next()) |list| {
            list.deinit(testing.allocator);
        }
        adjacency_list.deinit();
    }

    // 0 -0- 1 -0- 2 -1- 3
    var v0_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v0_neighbors.append(testing.allocator, .{ .neighbor = 1, .weight = 0 });
    try adjacency_list.put(0, v0_neighbors);

    var v1_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v1_neighbors.append(testing.allocator, .{ .neighbor = 0, .weight = 0 });
    try v1_neighbors.append(testing.allocator, .{ .neighbor = 2, .weight = 0 });
    try adjacency_list.put(1, v1_neighbors);

    var v2_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v2_neighbors.append(testing.allocator, .{ .neighbor = 1, .weight = 0 });
    try v2_neighbors.append(testing.allocator, .{ .neighbor = 3, .weight = 1 });
    try adjacency_list.put(2, v2_neighbors);

    var v3_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v3_neighbors.append(testing.allocator, .{ .neighbor = 2, .weight = 1 });
    try adjacency_list.put(3, v3_neighbors);

    var result = try prim.run(&adjacency_list, 0);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 3), result.edges.len);
    try testing.expectEqual(@as(i32, 1), result.total_weight); // 0+0+1 = 1
}

test "Prim: negative weights" {
    const P = Prim(i32, i32);
    var prim = P.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(i32, std.ArrayList(P.Neighbor)).init(testing.allocator);
    defer {
        var iter = adjacency_list.valueIterator();
        while (iter.next()) |list| {
            list.deinit(testing.allocator);
        }
        adjacency_list.deinit();
    }

    // Triangle with negative weight
    var v0_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v0_neighbors.append(testing.allocator, .{ .neighbor = 1, .weight = -5 });
    try v0_neighbors.append(testing.allocator, .{ .neighbor = 2, .weight = 2 });
    try adjacency_list.put(0, v0_neighbors);

    var v1_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v1_neighbors.append(testing.allocator, .{ .neighbor = 0, .weight = -5 });
    try v1_neighbors.append(testing.allocator, .{ .neighbor = 2, .weight = 3 });
    try adjacency_list.put(1, v1_neighbors);

    var v2_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v2_neighbors.append(testing.allocator, .{ .neighbor = 0, .weight = 2 });
    try v2_neighbors.append(testing.allocator, .{ .neighbor = 1, .weight = 3 });
    try adjacency_list.put(2, v2_neighbors);

    var result = try prim.run(&adjacency_list, 0);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), result.edges.len);
    try testing.expectEqual(@as(i32, -3), result.total_weight); // -5 + 2 = -3
}

test "Prim: floating point weights" {
    const P = Prim(u32, f64);
    var prim = P.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(P.Neighbor)).init(testing.allocator);
    defer {
        var iter = adjacency_list.valueIterator();
        while (iter.next()) |list| {
            list.deinit(testing.allocator);
        }
        adjacency_list.deinit();
    }

    var v0_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v0_neighbors.append(testing.allocator, .{ .neighbor = 1, .weight = 1.5 });
    try v0_neighbors.append(testing.allocator, .{ .neighbor = 2, .weight = 3.7 });
    try adjacency_list.put(0, v0_neighbors);

    var v1_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v1_neighbors.append(testing.allocator, .{ .neighbor = 0, .weight = 1.5 });
    try v1_neighbors.append(testing.allocator, .{ .neighbor = 2, .weight = 2.3 });
    try adjacency_list.put(1, v1_neighbors);

    var v2_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v2_neighbors.append(testing.allocator, .{ .neighbor = 0, .weight = 3.7 });
    try v2_neighbors.append(testing.allocator, .{ .neighbor = 1, .weight = 2.3 });
    try adjacency_list.put(2, v2_neighbors);

    var result = try prim.run(&adjacency_list, 0);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), result.edges.len);
    try testing.expectApproxEqAbs(@as(f64, 3.8), result.total_weight, 0.001); // 1.5 + 2.3 = 3.8
}

test "Prim: start from different vertex" {
    const P = Prim(u32, i32);
    var prim = P.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(P.Neighbor)).init(testing.allocator);
    defer {
        var iter = adjacency_list.valueIterator();
        while (iter.next()) |list| {
            list.deinit(testing.allocator);
        }
        adjacency_list.deinit();
    }

    // Linear chain: 0-5-1-2-2-3
    var v0_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v0_neighbors.append(testing.allocator, .{ .neighbor = 1, .weight = 5 });
    try adjacency_list.put(0, v0_neighbors);

    var v1_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v1_neighbors.append(testing.allocator, .{ .neighbor = 0, .weight = 5 });
    try v1_neighbors.append(testing.allocator, .{ .neighbor = 2, .weight = 2 });
    try adjacency_list.put(1, v1_neighbors);

    var v2_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v2_neighbors.append(testing.allocator, .{ .neighbor = 1, .weight = 2 });
    try v2_neighbors.append(testing.allocator, .{ .neighbor = 3, .weight = 3 });
    try adjacency_list.put(2, v2_neighbors);

    var v3_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v3_neighbors.append(testing.allocator, .{ .neighbor = 2, .weight = 3 });
    try adjacency_list.put(3, v3_neighbors);

    // Starting from vertex 0
    var result1 = try prim.run(&adjacency_list, 0);
    defer result1.deinit(testing.allocator);

    // Starting from vertex 3
    var result2 = try prim.run(&adjacency_list, 3);
    defer result2.deinit(testing.allocator);

    // Both should produce same total weight (MST is unique by edge weights)
    try testing.expectEqual(@as(usize, 3), result1.edges.len);
    try testing.expectEqual(@as(usize, 3), result2.edges.len);
    try testing.expectEqual(result1.total_weight, result2.total_weight);
    try testing.expectEqual(@as(i32, 10), result1.total_weight); // 5+2+3 = 10
}

test "Prim: disconnected graph" {
    const P = Prim(u32, i32);
    var prim = P.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(P.Neighbor)).init(testing.allocator);
    defer {
        var iter = adjacency_list.valueIterator();
        while (iter.next()) |list| {
            list.deinit(testing.allocator);
        }
        adjacency_list.deinit();
    }

    // Component 1: 0-1 (weight 5)
    var v0_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v0_neighbors.append(testing.allocator, .{ .neighbor = 1, .weight = 5 });
    try adjacency_list.put(0, v0_neighbors);

    var v1_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v1_neighbors.append(testing.allocator, .{ .neighbor = 0, .weight = 5 });
    try adjacency_list.put(1, v1_neighbors);

    // Component 2: 2-3 (weight 3)
    var v2_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v2_neighbors.append(testing.allocator, .{ .neighbor = 3, .weight = 3 });
    try adjacency_list.put(2, v2_neighbors);

    var v3_neighbors: std.ArrayList(P.Neighbor) = .{};
    try v3_neighbors.append(testing.allocator, .{ .neighbor = 2, .weight = 3 });
    try adjacency_list.put(3, v3_neighbors);

    // Prim from vertex 0 should only cover component 1
    var result = try prim.run(&adjacency_list, 0);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), result.edges.len); // Only 0-1 edge
    try testing.expectEqual(@as(i32, 5), result.total_weight);
}

test "Prim: stress test (100 vertices chain)" {
    const P = Prim(u32, i32);
    var prim = P.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(P.Neighbor)).init(testing.allocator);
    defer {
        var iter = adjacency_list.valueIterator();
        while (iter.next()) |list| {
            list.deinit(testing.allocator);
        }
        adjacency_list.deinit();
    }

    // Create chain: 0-1-2-...-99
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        var neighbors: std.ArrayList(P.Neighbor) = .{};
        if (i > 0) {
            try neighbors.append(testing.allocator, .{ .neighbor = i - 1, .weight = @intCast(i) });
        }
        if (i < 99) {
            try neighbors.append(testing.allocator, .{ .neighbor = i + 1, .weight = @intCast(i + 1) });
        }
        try adjacency_list.put(i, neighbors);
    }

    var result = try prim.run(&adjacency_list, 0);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 99), result.edges.len); // V-1 = 100-1 = 99

    // Total weight: 1+2+3+...+99 = 99*100/2 = 4950
    const expected_weight: i32 = (99 * 100) / 2;
    try testing.expectEqual(expected_weight, result.total_weight);
}
