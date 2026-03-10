const std = @import("std");
const testing = std.testing;

/// Kruskal's Minimum Spanning Tree algorithm.
///
/// Computes a minimum spanning tree (MST) of an undirected, weighted graph using the
/// greedy edge-addition approach with union-find for cycle detection.
///
/// Time: O(E log E) for sorting edges + O(E α(V)) for union-find ≈ O(E log E)
/// Space: O(V + E) for storing parent/rank arrays and result edges
///
/// Generic over:
/// - V: Vertex type (must be hashable and comparable)
/// - W: Weight type (must be comparable, e.g., i32, f64)
///
/// Algorithm:
/// 1. Sort all edges by weight (ascending)
/// 2. Initialize union-find with all vertices as separate sets
/// 3. For each edge in sorted order:
///    - If endpoints are in different sets (no cycle): add edge to MST, union sets
///    - Otherwise: skip edge
/// 4. Return MST edges (exactly V-1 edges for a connected graph)
///
/// Note: If the graph is disconnected, returns a minimum spanning forest.
pub fn Kruskal(comptime V: type, comptime W: type) type {
    return struct {
        const Self = @This();

        pub const Edge = struct {
            from: V,
            to: V,
            weight: W,

            pub fn lessThan(_: void, a: Edge, b: Edge) bool {
                return switch (@typeInfo(W)) {
                    .int => a.weight < b.weight,
                    .float => a.weight < b.weight,
                    .comptime_int => a.weight < b.weight,
                    .comptime_float => a.weight < b.weight,
                    else => @compileError("Weight type must be numeric (int or float)"),
                };
            }
        };

        pub const Result = struct {
            edges: []Edge,
            total_weight: W,

            pub fn deinit(self: *Result, allocator: std.mem.Allocator) void {
                allocator.free(self.edges);
            }
        };

        /// Internal union-find structure for cycle detection.
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
                    return false; // Already in same set (cycle would be formed)
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
        /// Time: O(E log E) — dominated by edge sorting
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

            // Sort edges by weight
            const sorted_edges = try self.allocator.alloc(Edge, edges.len);
            defer self.allocator.free(sorted_edges);
            @memcpy(sorted_edges, edges);
            std.mem.sort(Edge, sorted_edges, {}, Edge.lessThan);

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

            // Greedily add edges
            var mst_edges: std.ArrayList(Edge) = .{};
            errdefer mst_edges.deinit(self.allocator);

            var total_weight: W = 0;

            for (sorted_edges) |edge| {
                if (try uf.unite(edge.from, edge.to)) {
                    try mst_edges.append(self.allocator, edge);
                    total_weight = switch (@typeInfo(W)) {
                        .int, .comptime_int => total_weight + edge.weight,
                        .float, .comptime_float => total_weight + edge.weight,
                        else => unreachable,
                    };
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

test "Kruskal: basic MST" {
    const K = Kruskal(u32, i32);
    var kruskal = K.init(testing.allocator);

    // Triangle graph:
    //     1
    //    / \
    //  5/   \2
    //  /     \
    // 0-------2
    //     3
    const edges = [_]K.Edge{
        .{ .from = 0, .to = 1, .weight = 5 },
        .{ .from = 1, .to = 2, .weight = 2 },
        .{ .from = 0, .to = 2, .weight = 3 },
    };

    var result = try kruskal.run(&edges);
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

test "Kruskal: single vertex" {
    const K = Kruskal(u32, i32);
    var kruskal = K.init(testing.allocator);

    const edges = [_]K.Edge{};
    var result = try kruskal.run(&edges);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 0), result.edges.len);
    try testing.expectEqual(@as(i32, 0), result.total_weight);
}

test "Kruskal: two vertices" {
    const K = Kruskal(u32, i32);
    var kruskal = K.init(testing.allocator);

    const edges = [_]K.Edge{
        .{ .from = 0, .to = 1, .weight = 10 },
    };

    var result = try kruskal.run(&edges);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), result.edges.len);
    try testing.expectEqual(@as(i32, 10), result.total_weight);
    try testing.expectEqual(@as(i32, 10), result.edges[0].weight);
}

test "Kruskal: disconnected graph (forest)" {
    const K = Kruskal(u32, i32);
    var kruskal = K.init(testing.allocator);

    // Two disconnected components:
    // Component 1: 0-1 (weight 5)
    // Component 2: 2-3 (weight 3)
    const edges = [_]K.Edge{
        .{ .from = 0, .to = 1, .weight = 5 },
        .{ .from = 2, .to = 3, .weight = 3 },
    };

    var result = try kruskal.run(&edges);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), result.edges.len);
    try testing.expectEqual(@as(i32, 8), result.total_weight); // 5 + 3
}

test "Kruskal: complete graph K4" {
    const K = Kruskal(u32, i32);
    var kruskal = K.init(testing.allocator);

    // Complete graph on 4 vertices with distinct weights
    const edges = [_]K.Edge{
        .{ .from = 0, .to = 1, .weight = 1 },
        .{ .from = 0, .to = 2, .weight = 4 },
        .{ .from = 0, .to = 3, .weight = 3 },
        .{ .from = 1, .to = 2, .weight = 2 },
        .{ .from = 1, .to = 3, .weight = 5 },
        .{ .from = 2, .to = 3, .weight = 6 },
    };

    var result = try kruskal.run(&edges);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 3), result.edges.len); // V-1 = 4-1 = 3
    try testing.expectEqual(@as(i32, 6), result.total_weight); // 1+2+3 = 6 (MST: edges (0,1,1), (1,2,2), (0,3,3))
}

test "Kruskal: parallel edges (multi-graph)" {
    const K = Kruskal(u32, i32);
    var kruskal = K.init(testing.allocator);

    // Multiple edges between same vertices (should pick lightest)
    const edges = [_]K.Edge{
        .{ .from = 0, .to = 1, .weight = 10 },
        .{ .from = 0, .to = 1, .weight = 5 }, // lighter parallel edge
        .{ .from = 1, .to = 2, .weight = 3 },
    };

    var result = try kruskal.run(&edges);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), result.edges.len);
    try testing.expectEqual(@as(i32, 8), result.total_weight); // 5 + 3 = 8 (picks lighter edge)
}

test "Kruskal: zero weight edges" {
    const K = Kruskal(u32, i32);
    var kruskal = K.init(testing.allocator);

    const edges = [_]K.Edge{
        .{ .from = 0, .to = 1, .weight = 0 },
        .{ .from = 1, .to = 2, .weight = 0 },
        .{ .from = 2, .to = 3, .weight = 1 },
    };

    var result = try kruskal.run(&edges);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 3), result.edges.len);
    try testing.expectEqual(@as(i32, 1), result.total_weight); // 0+0+1 = 1
}

test "Kruskal: negative weights" {
    const K = Kruskal(i32, i32);
    var kruskal = K.init(testing.allocator);

    const edges = [_]K.Edge{
        .{ .from = 0, .to = 1, .weight = -5 },
        .{ .from = 1, .to = 2, .weight = 3 },
        .{ .from = 0, .to = 2, .weight = 2 },
    };

    var result = try kruskal.run(&edges);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), result.edges.len);
    try testing.expectEqual(@as(i32, -3), result.total_weight); // -5 + 2 = -3
}

test "Kruskal: floating point weights" {
    const K = Kruskal(u32, f64);
    var kruskal = K.init(testing.allocator);

    const edges = [_]K.Edge{
        .{ .from = 0, .to = 1, .weight = 1.5 },
        .{ .from = 1, .to = 2, .weight = 2.3 },
        .{ .from = 0, .to = 2, .weight = 3.7 },
    };

    var result = try kruskal.run(&edges);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), result.edges.len);
    try testing.expectApproxEqAbs(@as(f64, 3.8), result.total_weight, 0.001); // 1.5 + 2.3 = 3.8
}

test "Kruskal: stress test (100 vertices)" {
    const K = Kruskal(u32, i32);
    var kruskal = K.init(testing.allocator);

    // Create a simple chain: 0-1-2-3-...-99
    var edges: std.ArrayList(K.Edge) = .{};
    defer edges.deinit(testing.allocator);

    var i: u32 = 0;
    while (i < 99) : (i += 1) {
        try edges.append(testing.allocator, .{ .from = i, .to = i + 1, .weight = @intCast(i + 1) });
    }

    // Add some extra edges with higher weights (should not be included in MST)
    try edges.append(testing.allocator, .{ .from = 0, .to = 50, .weight = 1000 });
    try edges.append(testing.allocator, .{ .from = 25, .to = 75, .weight = 2000 });

    var result = try kruskal.run(edges.items);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 99), result.edges.len); // V-1 = 100-1 = 99

    // Total weight should be 1+2+3+...+99 = 99*100/2 = 4950
    const expected_weight: i32 = (99 * 100) / 2;
    try testing.expectEqual(expected_weight, result.total_weight);
}

// Note: String vertex types ([]const u8) require special handling with StringHashMap
// and are not demonstrated in these tests. Use integer or custom hashable types instead.
