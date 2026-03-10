const std = @import("std");
const testing = std.testing;

/// Borůvka's Minimum Spanning Tree algorithm.
///
/// Computes a minimum spanning tree (MST) of an undirected, weighted graph using a
/// parallel-friendly approach that adds multiple edges simultaneously in rounds.
///
/// Time: O(E log V) — at most log V rounds, each round processes all edges
/// Space: O(V + E) for storing component membership and result edges
///
/// Generic over:
/// - V: Vertex type (must be hashable and comparable)
/// - W: Weight type (must be comparable, e.g., i32, f64)
///
/// Algorithm:
/// 1. Initialize: Each vertex is its own component
/// 2. Repeat until one component remains (or no edges to add):
///    a. For each component, find the minimum-weight edge leaving it
///    b. Add all such edges to MST (merging components)
/// 3. Return MST edges
///
/// Difference from Kruskal and Prim:
/// - Borůvka: Adds multiple edges per round (parallelizable, oldest MST algorithm - 1926)
/// - Kruskal: Adds edges one-by-one in sorted order (edge-centric)
/// - Prim: Grows a single tree (vertex-centric)
///
/// Note: If the graph is disconnected, returns a minimum spanning forest.
pub fn Boruvka(comptime V: type, comptime W: type) type {
    return struct {
        const Self = @This();

        pub const Edge = struct {
            from: V,
            to: V,
            weight: W,
        };

        pub const Result = struct {
            edges: []Edge,
            total_weight: W,

            pub fn deinit(self: *Result, allocator: std.mem.Allocator) void {
                allocator.free(self.edges);
            }
        };

        /// Internal union-find structure for component tracking.
        const UnionFind = struct {
            parent: std.AutoHashMap(V, V),
            rank: std.AutoHashMap(V, usize),
            allocator: std.mem.Allocator,

            fn init(allocator: std.mem.Allocator) UnionFind {
                return .{
                    .parent = std.AutoHashMap(V, V).init(allocator),
                    .rank = std.AutoHashMap(V, usize).init(allocator),
                    .allocator = allocator,
                };
            }

            fn deinit(self: *UnionFind) void {
                self.parent.deinit();
                self.rank.deinit();
            }

            fn makeSet(self: *UnionFind, v: V) !void {
                if (!self.parent.contains(v)) {
                    try self.parent.put(v, v);
                    try self.rank.put(v, 0);
                }
            }

            fn find(self: *UnionFind, v: V) !V {
                const parent = self.parent.get(v) orelse return error.VertexNotFound;
                if (std.meta.eql(parent, v)) {
                    return v;
                }
                // Path compression
                const root = try self.find(parent);
                try self.parent.put(v, root);
                return root;
            }

            fn unite(self: *UnionFind, a: V, b: V) !bool {
                const root_a = try self.find(a);
                const root_b = try self.find(b);

                if (std.meta.eql(root_a, root_b)) {
                    return false; // Already in same component
                }

                // Union by rank
                const rank_a = self.rank.get(root_a) orelse 0;
                const rank_b = self.rank.get(root_b) orelse 0;

                if (rank_a < rank_b) {
                    try self.parent.put(root_a, root_b);
                } else if (rank_a > rank_b) {
                    try self.parent.put(root_b, root_a);
                } else {
                    try self.parent.put(root_b, root_a);
                    try self.rank.put(root_a, rank_a + 1);
                }

                return true;
            }
        };

        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .allocator = allocator };
        }

        /// Compute the minimum spanning tree from a list of edges.
        ///
        /// Time: O(E log V) — at most log V rounds of O(E) work each
        /// Space: O(V + E) — union-find structures and result storage
        ///
        /// Parameters:
        /// - edges: All edges in the graph (undirected, so each edge appears once)
        ///
        /// Returns:
        /// - Result containing MST edges and total weight
        ///
        /// Note: If the graph is disconnected, returns a minimum spanning forest.
        pub fn run(self: Self, edges: []const Edge) !Result {
            if (edges.len == 0) {
                return Result{
                    .edges = try self.allocator.alloc(Edge, 0),
                    .total_weight = 0,
                };
            }

            // Initialize union-find with all vertices
            var uf = UnionFind.init(self.allocator);
            defer uf.deinit();

            // Collect unique vertices
            var vertices = std.AutoHashMap(V, void).init(self.allocator);
            defer vertices.deinit();
            for (edges) |edge| {
                try vertices.put(edge.from, {});
                try vertices.put(edge.to, {});
            }

            // Make sets for all vertices
            var vertex_iter = vertices.keyIterator();
            while (vertex_iter.next()) |v| {
                try uf.makeSet(v.*);
            }

            // MST edges collected across all rounds
            var mst_edges: std.ArrayList(Edge) = .{};
            errdefer mst_edges.deinit(self.allocator);

            var total_weight: W = 0;
            const num_vertices = vertices.count();

            // Repeat until we have V-1 edges (one tree) or no edges to add
            while (mst_edges.items.len < num_vertices - 1) {
                // For each component, find the minimum-weight edge leaving it
                var cheapest = std.AutoHashMap(V, Edge).init(self.allocator);
                defer cheapest.deinit();

                // Find cheapest outgoing edge for each component
                for (edges) |edge| {
                    const comp_from = try uf.find(edge.from);
                    const comp_to = try uf.find(edge.to);

                    // Skip edges within same component
                    if (std.meta.eql(comp_from, comp_to)) {
                        continue;
                    }

                    // Check if this is cheaper for comp_from
                    if (cheapest.get(comp_from)) |current| {
                        if (compareWeights(W, edge.weight, current.weight) == .lt) {
                            try cheapest.put(comp_from, edge);
                        }
                    } else {
                        try cheapest.put(comp_from, edge);
                    }

                    // Check if this is cheaper for comp_to
                    if (cheapest.get(comp_to)) |current| {
                        if (compareWeights(W, edge.weight, current.weight) == .lt) {
                            try cheapest.put(comp_to, edge);
                        }
                    } else {
                        try cheapest.put(comp_to, edge);
                    }
                }

                // If no edges found, we're done (disconnected components)
                if (cheapest.count() == 0) {
                    break;
                }

                // Add all cheapest edges and merge components
                var cheapest_iter = cheapest.valueIterator();
                var added_this_round: usize = 0;

                while (cheapest_iter.next()) |edge_ptr| {
                    const edge = edge_ptr.*;
                    // Try to unite (may fail if already united by another edge this round)
                    if (try uf.unite(edge.from, edge.to)) {
                        try mst_edges.append(self.allocator, edge);
                        total_weight = switch (@typeInfo(W)) {
                            .int, .comptime_int => total_weight + edge.weight,
                            .float, .comptime_float => total_weight + edge.weight,
                            else => unreachable,
                        };
                        added_this_round += 1;
                    }
                }

                // If no edges were added, we're done
                if (added_this_round == 0) {
                    break;
                }
            }

            return Result{
                .edges = try mst_edges.toOwnedSlice(self.allocator),
                .total_weight = total_weight,
            };
        }

        /// Compare two weights of type W.
        fn compareWeights(comptime WeightType: type, a: WeightType, b: WeightType) std.math.Order {
            return switch (@typeInfo(WeightType)) {
                .int, .comptime_int => if (a < b) .lt else if (a > b) .gt else .eq,
                .float, .comptime_float => if (a < b) .lt else if (a > b) .gt else .eq,
                else => @compileError("Weight type must be numeric (int or float)"),
            };
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "Boruvka: basic MST" {
    const B = Boruvka(u32, i32);
    var boruvka = B.init(testing.allocator);

    // Triangle graph:
    //     1
    //    / \
    //  5/   \2
    //  /     \
    // 0-------2
    //     3
    const edges = [_]B.Edge{
        .{ .from = 0, .to = 1, .weight = 5 },
        .{ .from = 1, .to = 2, .weight = 2 },
        .{ .from = 0, .to = 2, .weight = 3 },
    };

    var result = try boruvka.run(&edges);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), result.edges.len); // V-1 = 3-1 = 2
    try testing.expectEqual(@as(i32, 5), result.total_weight); // 2 + 3 = 5

    // MST should contain edges (1,2,2) and (0,2,3)
    var has_1_2 = false;
    var has_0_2 = false;
    for (result.edges) |edge| {
        if ((edge.from == 1 and edge.to == 2) or (edge.from == 2 and edge.to == 1)) {
            has_1_2 = true;
            try testing.expectEqual(@as(i32, 2), edge.weight);
        }
        if ((edge.from == 0 and edge.to == 2) or (edge.from == 2 and edge.to == 0)) {
            has_0_2 = true;
            try testing.expectEqual(@as(i32, 3), edge.weight);
        }
    }
    try testing.expect(has_1_2);
    try testing.expect(has_0_2);
}

test "Boruvka: single vertex" {
    const B = Boruvka(u32, i32);
    var boruvka = B.init(testing.allocator);

    const edges = [_]B.Edge{};
    var result = try boruvka.run(&edges);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 0), result.edges.len);
    try testing.expectEqual(@as(i32, 0), result.total_weight);
}

test "Boruvka: two vertices" {
    const B = Boruvka(u32, i32);
    var boruvka = B.init(testing.allocator);

    const edges = [_]B.Edge{
        .{ .from = 0, .to = 1, .weight = 10 },
    };

    var result = try boruvka.run(&edges);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), result.edges.len);
    try testing.expectEqual(@as(i32, 10), result.total_weight);
    try testing.expectEqual(@as(i32, 10), result.edges[0].weight);
}

test "Boruvka: disconnected graph (forest)" {
    const B = Boruvka(u32, i32);
    var boruvka = B.init(testing.allocator);

    // Two disconnected components:
    // Component 1: 0-1 (weight 5)
    // Component 2: 2-3 (weight 3)
    const edges = [_]B.Edge{
        .{ .from = 0, .to = 1, .weight = 5 },
        .{ .from = 2, .to = 3, .weight = 3 },
    };

    var result = try boruvka.run(&edges);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), result.edges.len);
    try testing.expectEqual(@as(i32, 8), result.total_weight); // 5 + 3
}

test "Boruvka: complete graph K4" {
    const B = Boruvka(u32, i32);
    var boruvka = B.init(testing.allocator);

    // Complete graph on 4 vertices with distinct weights
    const edges = [_]B.Edge{
        .{ .from = 0, .to = 1, .weight = 1 },
        .{ .from = 0, .to = 2, .weight = 4 },
        .{ .from = 0, .to = 3, .weight = 3 },
        .{ .from = 1, .to = 2, .weight = 2 },
        .{ .from = 1, .to = 3, .weight = 5 },
        .{ .from = 2, .to = 3, .weight = 6 },
    };

    var result = try boruvka.run(&edges);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 3), result.edges.len); // V-1 = 4-1 = 3
    try testing.expectEqual(@as(i32, 6), result.total_weight); // 1+2+3 = 6 (MST: edges (0,1,1), (1,2,2), (0,3,3))
}

test "Boruvka: parallel edges (multi-graph)" {
    const B = Boruvka(u32, i32);
    var boruvka = B.init(testing.allocator);

    // Multiple edges between same vertices (should pick lightest)
    const edges = [_]B.Edge{
        .{ .from = 0, .to = 1, .weight = 10 },
        .{ .from = 0, .to = 1, .weight = 5 }, // lighter parallel edge
        .{ .from = 1, .to = 2, .weight = 3 },
    };

    var result = try boruvka.run(&edges);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), result.edges.len);
    try testing.expectEqual(@as(i32, 8), result.total_weight); // 5 + 3 = 8 (picks lighter edge)
}

test "Boruvka: zero weight edges" {
    const B = Boruvka(u32, i32);
    var boruvka = B.init(testing.allocator);

    const edges = [_]B.Edge{
        .{ .from = 0, .to = 1, .weight = 0 },
        .{ .from = 1, .to = 2, .weight = 0 },
        .{ .from = 2, .to = 3, .weight = 1 },
    };

    var result = try boruvka.run(&edges);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 3), result.edges.len);
    try testing.expectEqual(@as(i32, 1), result.total_weight); // 0+0+1 = 1
}

test "Boruvka: negative weights" {
    const B = Boruvka(i32, i32);
    var boruvka = B.init(testing.allocator);

    const edges = [_]B.Edge{
        .{ .from = 0, .to = 1, .weight = -5 },
        .{ .from = 1, .to = 2, .weight = 3 },
        .{ .from = 0, .to = 2, .weight = 2 },
    };

    var result = try boruvka.run(&edges);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), result.edges.len);
    try testing.expectEqual(@as(i32, -3), result.total_weight); // -5 + 2 = -3
}

test "Boruvka: floating point weights" {
    const B = Boruvka(u32, f64);
    var boruvka = B.init(testing.allocator);

    const edges = [_]B.Edge{
        .{ .from = 0, .to = 1, .weight = 1.5 },
        .{ .from = 1, .to = 2, .weight = 2.3 },
        .{ .from = 0, .to = 2, .weight = 3.7 },
    };

    var result = try boruvka.run(&edges);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), result.edges.len);
    try testing.expectApproxEqAbs(@as(f64, 3.8), result.total_weight, 0.001); // 1.5 + 2.3 = 3.8
}

test "Boruvka: stress test (100 vertices)" {
    const B = Boruvka(u32, i32);
    var boruvka = B.init(testing.allocator);

    // Create a simple chain: 0-1-2-3-...-99
    var edges: std.ArrayList(B.Edge) = .{};
    defer edges.deinit(testing.allocator);

    var i: u32 = 0;
    while (i < 99) : (i += 1) {
        try edges.append(testing.allocator, .{ .from = i, .to = i + 1, .weight = @intCast(i + 1) });
    }

    // Add some extra edges with higher weights (should not be included in MST)
    try edges.append(testing.allocator, .{ .from = 0, .to = 50, .weight = 1000 });
    try edges.append(testing.allocator, .{ .from = 25, .to = 75, .weight = 2000 });

    var result = try boruvka.run(edges.items);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 99), result.edges.len); // V-1 = 100-1 = 99

    // Total weight should be 1+2+3+...+99 = 99*100/2 = 4950
    const expected_weight: i32 = (99 * 100) / 2;
    try testing.expectEqual(expected_weight, result.total_weight);
}

test "Boruvka: multiple rounds behavior" {
    const B = Boruvka(u32, i32);
    var boruvka = B.init(testing.allocator);

    // Graph that requires multiple rounds:
    // Round 1: Connects (0-1), (2-3), (4-5), (6-7)
    // Round 2: Connects pairs into larger components
    // Round 3: Merges into final tree
    const edges = [_]B.Edge{
        // First layer: 4 components of 2 vertices each
        .{ .from = 0, .to = 1, .weight = 1 },
        .{ .from = 2, .to = 3, .weight = 1 },
        .{ .from = 4, .to = 5, .weight = 1 },
        .{ .from = 6, .to = 7, .weight = 1 },
        // Second layer: connect pairs
        .{ .from = 1, .to = 2, .weight = 2 },
        .{ .from = 5, .to = 6, .weight = 2 },
        // Third layer: connect halves
        .{ .from = 3, .to = 4, .weight = 3 },
    };

    var result = try boruvka.run(&edges);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 7), result.edges.len); // V-1 = 8-1 = 7
    try testing.expectEqual(@as(i32, 11), result.total_weight); // 1+1+1+1+2+2+3 = 11
}
