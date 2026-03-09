const std = @import("std");
const Allocator = std.mem.Allocator;

/// Compressed Sparse Row (CSR) graph representation.
/// An immutable, cache-friendly format optimal for sparse graphs.
///
/// Structure:
/// - rowOffsets[v] contains the starting index in edges array for vertex v's neighbors
/// - rowOffsets[v+1] - rowOffsets[v] gives the degree of vertex v
/// - edges array contains all neighbor vertex IDs in consecutive blocks
/// - edgeWeights (optional) contains corresponding edge weights
///
/// Time complexity:
/// - Construction: O(V + E)
/// - Neighbor access: O(1) to get the neighbor slice
/// - Space: O(V + E)
///
/// Best for: read-heavy graph algorithms (BFS, DFS, PageRank, shortest paths)
/// where the graph structure doesn't change after construction.
pub fn CompressedSparseRow(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Iterator over neighbors of a vertex
        pub const Iterator = struct {
            neighbors: []const usize,
            weights: ?[]const T,
            index: usize,

            pub const Entry = struct {
                vertex: usize,
                weight: ?T,
            };

            /// Returns the next neighbor and its weight (if weighted)
            /// Time: O(1)
            pub fn next(self: *Iterator) ?Entry {
                if (self.index >= self.neighbors.len) return null;
                const vertex = self.neighbors[self.index];
                const weight = if (self.weights) |w| w[self.index] else null;
                self.index += 1;
                return Entry{ .vertex = vertex, .weight = weight };
            }
        };

        allocator: Allocator,
        /// Number of vertices in the graph
        vertex_count: usize,
        /// Number of edges in the graph
        edge_count: usize,
        /// rowOffsets[i] is the start index in edges array for vertex i
        /// rowOffsets has length vertex_count + 1
        row_offsets: []usize,
        /// Flattened array of all neighbor vertex IDs
        edges: []usize,
        /// Optional edge weights (same length as edges)
        edge_weights: ?[]T,
        /// Whether the graph is directed
        directed: bool,

        /// Initialize an empty CSR graph
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator, directed: bool) Self {
            return .{
                .allocator = allocator,
                .vertex_count = 0,
                .edge_count = 0,
                .row_offsets = &[_]usize{},
                .edges = &[_]usize{},
                .edge_weights = null,
                .directed = directed,
            };
        }

        /// Deinitialize and free all memory
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.row_offsets.len > 0) {
                self.allocator.free(self.row_offsets);
            }
            if (self.edges.len > 0) {
                self.allocator.free(self.edges);
            }
            if (self.edge_weights) |weights| {
                self.allocator.free(weights);
            }
            self.* = undefined;
        }

        /// Build CSR from edge list
        /// Time: O(V + E) | Space: O(V + E)
        pub fn fromEdges(
            allocator: Allocator,
            vertex_count: usize,
            edges_list: []const Edge(T),
            directed: bool,
        ) !Self {
            if (vertex_count == 0) {
                return Self.init(allocator, directed);
            }

            // Count out-degree for each vertex
            const degrees = try allocator.alloc(usize, vertex_count);
            defer allocator.free(degrees);
            @memset(degrees, 0);

            for (edges_list) |edge| {
                if (edge.from >= vertex_count or edge.to >= vertex_count) {
                    return error.VertexOutOfBounds;
                }
                degrees[edge.from] += 1;
                if (!directed and edge.from != edge.to) {
                    degrees[edge.to] += 1;
                }
            }

            // Build row offsets (prefix sum of degrees)
            const row_offsets = try allocator.alloc(usize, vertex_count + 1);
            errdefer allocator.free(row_offsets);

            row_offsets[0] = 0;
            for (0..vertex_count) |i| {
                row_offsets[i + 1] = row_offsets[i] + degrees[i];
            }

            const total_edges = row_offsets[vertex_count];
            const edge_array = try allocator.alloc(usize, total_edges);
            errdefer allocator.free(edge_array);

            const has_weights = edges_list.len > 0 and edges_list[0].weight != null;
            const weight_array = if (has_weights) try allocator.alloc(T, total_edges) else null;
            errdefer if (weight_array) |w| allocator.free(w);

            // Fill edges and weights
            // Use degrees array as temporary write position tracker
            @memset(degrees, 0);

            for (edges_list) |edge| {
                const from_pos = row_offsets[edge.from] + degrees[edge.from];
                edge_array[from_pos] = edge.to;
                if (weight_array) |weights| {
                    weights[from_pos] = edge.weight orelse @as(T, 0);
                }
                degrees[edge.from] += 1;

                if (!directed and edge.from != edge.to) {
                    const to_pos = row_offsets[edge.to] + degrees[edge.to];
                    edge_array[to_pos] = edge.from;
                    if (weight_array) |weights| {
                        weights[to_pos] = edge.weight orelse @as(T, 0);
                    }
                    degrees[edge.to] += 1;
                }
            }

            return .{
                .allocator = allocator,
                .vertex_count = vertex_count,
                .edge_count = edges_list.len,
                .row_offsets = row_offsets,
                .edges = edge_array,
                .edge_weights = weight_array,
                .directed = directed,
            };
        }

        /// Build CSR from adjacency list representation
        /// Time: O(V + E) | Space: O(V + E)
        pub fn fromAdjacencyList(
            allocator: Allocator,
            adj_list: []const []const usize,
            weights: ?[]const []const T,
            directed: bool,
        ) !Self {
            const vertex_count = adj_list.len;
            if (vertex_count == 0) {
                return Self.init(allocator, directed);
            }

            // Calculate total edges
            var total_edges: usize = 0;
            for (adj_list) |neighbors| {
                total_edges += neighbors.len;
            }

            // Build row offsets
            const row_offsets = try allocator.alloc(usize, vertex_count + 1);
            errdefer allocator.free(row_offsets);

            row_offsets[0] = 0;
            for (0..vertex_count) |i| {
                row_offsets[i + 1] = row_offsets[i] + adj_list[i].len;
            }

            // Allocate edges array
            const edge_array = try allocator.alloc(usize, total_edges);
            errdefer allocator.free(edge_array);

            // Allocate weights if provided
            const weight_array = if (weights != null) try allocator.alloc(T, total_edges) else null;
            errdefer if (weight_array) |w| allocator.free(w);

            // Fill edges and weights
            var edge_idx: usize = 0;
            for (adj_list, 0..) |neighbors, v| {
                @memcpy(edge_array[edge_idx .. edge_idx + neighbors.len], neighbors);
                if (weight_array) |w| {
                    if (weights) |wlist| {
                        @memcpy(w[edge_idx .. edge_idx + neighbors.len], wlist[v]);
                    }
                }
                edge_idx += neighbors.len;
            }

            return .{
                .allocator = allocator,
                .vertex_count = vertex_count,
                .edge_count = total_edges,
                .row_offsets = row_offsets,
                .edges = edge_array,
                .edge_weights = weight_array,
                .directed = directed,
            };
        }

        /// Get the number of vertices
        /// Time: O(1) | Space: O(1)
        pub fn vertexCount(self: *const Self) usize {
            return self.vertex_count;
        }

        /// Get the number of edges
        /// Time: O(1) | Space: O(1)
        pub fn edgeCount(self: *const Self) usize {
            return self.edge_count;
        }

        /// Check if the graph is empty
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.vertex_count == 0;
        }

        /// Check if an edge exists from u to v
        /// Time: O(degree(u)) | Space: O(1)
        pub fn hasEdge(self: *const Self, from: usize, to: usize) bool {
            if (from >= self.vertex_count) return false;
            const neighbors = self.getNeighbors(from);
            for (neighbors) |neighbor| {
                if (neighbor == to) return true;
            }
            return false;
        }

        /// Get the weight of edge from u to v
        /// Returns null if edge doesn't exist or graph is unweighted
        /// Time: O(degree(u)) | Space: O(1)
        pub fn getEdgeWeight(self: *const Self, from: usize, to: usize) ?T {
            if (from >= self.vertex_count) return null;
            const neighbors = self.getNeighbors(from);
            for (neighbors, 0..) |neighbor, i| {
                if (neighbor == to) {
                    if (self.edge_weights) |weights| {
                        const offset = self.row_offsets[from];
                        return weights[offset + i];
                    }
                    return null;
                }
            }
            return null;
        }

        /// Get neighbors of a vertex as a slice
        /// Time: O(1) | Space: O(1)
        pub fn getNeighbors(self: *const Self, vertex: usize) []const usize {
            if (vertex >= self.vertex_count) return &[_]usize{};
            const start = self.row_offsets[vertex];
            const end = self.row_offsets[vertex + 1];
            return self.edges[start..end];
        }

        /// Get out-degree of a vertex
        /// Time: O(1) | Space: O(1)
        pub fn outDegree(self: *const Self, vertex: usize) usize {
            if (vertex >= self.vertex_count) return 0;
            return self.row_offsets[vertex + 1] - self.row_offsets[vertex];
        }

        /// Get in-degree of a vertex (requires scanning all edges for directed graphs)
        /// Time: O(E) | Space: O(1)
        pub fn inDegree(self: *const Self, vertex: usize) usize {
            if (vertex >= self.vertex_count) return 0;
            if (!self.directed) return self.outDegree(vertex);

            var count: usize = 0;
            for (self.edges) |neighbor| {
                if (neighbor == vertex) count += 1;
            }
            return count;
        }

        /// Create an iterator over neighbors of a vertex
        /// Time: O(1) | Space: O(1)
        pub fn iterator(self: *const Self, vertex: usize) Iterator {
            if (vertex >= self.vertex_count) {
                return .{
                    .neighbors = &[_]usize{},
                    .weights = null,
                    .index = 0,
                };
            }

            const start = self.row_offsets[vertex];
            const end = self.row_offsets[vertex + 1];
            const neighbors = self.edges[start..end];
            const weights = if (self.edge_weights) |w| w[start..end] else null;

            return .{
                .neighbors = neighbors,
                .weights = weights,
                .index = 0,
            };
        }

        /// Validate graph invariants
        /// Time: O(V + E) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            // Check row_offsets length
            if (self.row_offsets.len != self.vertex_count + 1) {
                return error.InvalidRowOffsets;
            }

            // Check row_offsets are non-decreasing
            for (0..self.vertex_count) |i| {
                if (self.row_offsets[i] > self.row_offsets[i + 1]) {
                    return error.InvalidRowOffsets;
                }
            }

            // Check total edges matches row_offsets
            if (self.row_offsets[self.vertex_count] != self.edges.len) {
                return error.EdgeCountMismatch;
            }

            // Check weights array length if present
            if (self.edge_weights) |weights| {
                if (weights.len != self.edges.len) {
                    return error.WeightCountMismatch;
                }
            }

            // Check all edge vertices are in bounds
            for (self.edges) |vertex| {
                if (vertex >= self.vertex_count) {
                    return error.VertexOutOfBounds;
                }
            }
        }

        /// Clone the CSR graph
        /// Time: O(V + E) | Space: O(V + E)
        pub fn clone(self: *const Self) !Self {
            const new_row_offsets = try self.allocator.dupe(usize, self.row_offsets);
            errdefer self.allocator.free(new_row_offsets);

            const new_edges = try self.allocator.dupe(usize, self.edges);
            errdefer self.allocator.free(new_edges);

            const new_weights = if (self.edge_weights) |w|
                try self.allocator.dupe(T, w)
            else
                null;
            errdefer if (new_weights) |w| self.allocator.free(w);

            return .{
                .allocator = self.allocator,
                .vertex_count = self.vertex_count,
                .edge_count = self.edge_count,
                .row_offsets = new_row_offsets,
                .edges = new_edges,
                .edge_weights = new_weights,
                .directed = self.directed,
            };
        }
    };
}

/// Edge representation for building CSR from edge list
pub fn Edge(comptime T: type) type {
    return struct {
        from: usize,
        to: usize,
        weight: ?T,
    };
}

// ============================================================================
// TESTS
// ============================================================================

test "CSR: basic construction from edges" {
    const allocator = std.testing.allocator;

    const edges = [_]Edge(i32){
        .{ .from = 0, .to = 1, .weight = 10 },
        .{ .from = 0, .to = 2, .weight = 20 },
        .{ .from = 1, .to = 2, .weight = 30 },
    };

    var csr = try CompressedSparseRow(i32).fromEdges(allocator, 3, &edges, true);
    defer csr.deinit();

    try std.testing.expectEqual(@as(usize, 3), csr.vertexCount());
    try std.testing.expectEqual(@as(usize, 3), csr.edgeCount());
    try std.testing.expect(csr.hasEdge(0, 1));
    try std.testing.expect(csr.hasEdge(0, 2));
    try std.testing.expect(csr.hasEdge(1, 2));
    try std.testing.expect(!csr.hasEdge(1, 0));

    try csr.validate();
}

test "CSR: undirected graph construction" {
    const allocator = std.testing.allocator;

    const edges = [_]Edge(i32){
        .{ .from = 0, .to = 1, .weight = 10 },
        .{ .from = 1, .to = 2, .weight = 20 },
    };

    var csr = try CompressedSparseRow(i32).fromEdges(allocator, 3, &edges, false);
    defer csr.deinit();

    // In undirected graph, edges are bidirectional
    try std.testing.expect(csr.hasEdge(0, 1));
    try std.testing.expect(csr.hasEdge(1, 0));
    try std.testing.expect(csr.hasEdge(1, 2));
    try std.testing.expect(csr.hasEdge(2, 1));

    try csr.validate();
}

test "CSR: weighted edges" {
    const allocator = std.testing.allocator;

    const edges = [_]Edge(f32){
        .{ .from = 0, .to = 1, .weight = 1.5 },
        .{ .from = 0, .to = 2, .weight = 2.5 },
        .{ .from = 1, .to = 2, .weight = 3.5 },
    };

    var csr = try CompressedSparseRow(f32).fromEdges(allocator, 3, &edges, true);
    defer csr.deinit();

    try std.testing.expectEqual(@as(?f32, 1.5), csr.getEdgeWeight(0, 1));
    try std.testing.expectEqual(@as(?f32, 2.5), csr.getEdgeWeight(0, 2));
    try std.testing.expectEqual(@as(?f32, 3.5), csr.getEdgeWeight(1, 2));
    try std.testing.expectEqual(@as(?f32, null), csr.getEdgeWeight(1, 0));

    try csr.validate();
}

test "CSR: neighbor iteration" {
    const allocator = std.testing.allocator;

    const edges = [_]Edge(i32){
        .{ .from = 0, .to = 1, .weight = 10 },
        .{ .from = 0, .to = 2, .weight = 20 },
        .{ .from = 0, .to = 3, .weight = 30 },
    };

    var csr = try CompressedSparseRow(i32).fromEdges(allocator, 4, &edges, true);
    defer csr.deinit();

    var iter = csr.iterator(0);
    var count: usize = 0;
    while (iter.next()) |entry| {
        try std.testing.expect(entry.vertex >= 1 and entry.vertex <= 3);
        try std.testing.expect(entry.weight != null);
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), count);

    try csr.validate();
}

test "CSR: degrees" {
    const allocator = std.testing.allocator;

    const edges = [_]Edge(i32){
        .{ .from = 0, .to = 1, .weight = null },
        .{ .from = 0, .to = 2, .weight = null },
        .{ .from = 1, .to = 2, .weight = null },
        .{ .from = 2, .to = 0, .weight = null },
    };

    var csr = try CompressedSparseRow(i32).fromEdges(allocator, 3, &edges, true);
    defer csr.deinit();

    try std.testing.expectEqual(@as(usize, 2), csr.outDegree(0));
    try std.testing.expectEqual(@as(usize, 1), csr.outDegree(1));
    try std.testing.expectEqual(@as(usize, 1), csr.outDegree(2));

    try std.testing.expectEqual(@as(usize, 1), csr.inDegree(0));
    try std.testing.expectEqual(@as(usize, 2), csr.inDegree(2));

    try csr.validate();
}

test "CSR: from adjacency list" {
    const allocator = std.testing.allocator;

    const adj0 = [_]usize{ 1, 2 };
    const adj1 = [_]usize{2};
    const adj2 = [_]usize{};
    const adj_list = [_][]const usize{ &adj0, &adj1, &adj2 };

    var csr = try CompressedSparseRow(i32).fromAdjacencyList(allocator, &adj_list, null, true);
    defer csr.deinit();

    try std.testing.expectEqual(@as(usize, 3), csr.vertexCount());
    try std.testing.expect(csr.hasEdge(0, 1));
    try std.testing.expect(csr.hasEdge(0, 2));
    try std.testing.expect(csr.hasEdge(1, 2));
    try std.testing.expect(!csr.hasEdge(2, 0));

    try csr.validate();
}

test "CSR: empty graph" {
    const allocator = std.testing.allocator;

    var csr = CompressedSparseRow(i32).init(allocator, true);
    defer csr.deinit();

    try std.testing.expect(csr.isEmpty());
    try std.testing.expectEqual(@as(usize, 0), csr.vertexCount());
    try std.testing.expectEqual(@as(usize, 0), csr.edgeCount());
    try std.testing.expect(!csr.hasEdge(0, 1));

    try csr.validate();
}

test "CSR: self-loop" {
    const allocator = std.testing.allocator;

    const edges = [_]Edge(i32){
        .{ .from = 0, .to = 0, .weight = 5 },
        .{ .from = 0, .to = 1, .weight = 10 },
    };

    var csr = try CompressedSparseRow(i32).fromEdges(allocator, 2, &edges, true);
    defer csr.deinit();

    try std.testing.expect(csr.hasEdge(0, 0));
    try std.testing.expectEqual(@as(?i32, 5), csr.getEdgeWeight(0, 0));
    try std.testing.expectEqual(@as(usize, 2), csr.outDegree(0));

    try csr.validate();
}

test "CSR: clone" {
    const allocator = std.testing.allocator;

    const edges = [_]Edge(i32){
        .{ .from = 0, .to = 1, .weight = 10 },
        .{ .from = 1, .to = 2, .weight = 20 },
    };

    var csr = try CompressedSparseRow(i32).fromEdges(allocator, 3, &edges, true);
    defer csr.deinit();

    var cloned = try csr.clone();
    defer cloned.deinit();

    try std.testing.expectEqual(csr.vertexCount(), cloned.vertexCount());
    try std.testing.expectEqual(csr.edgeCount(), cloned.edgeCount());
    try std.testing.expect(cloned.hasEdge(0, 1));
    try std.testing.expect(cloned.hasEdge(1, 2));
    try std.testing.expectEqual(@as(?i32, 10), cloned.getEdgeWeight(0, 1));

    try cloned.validate();
}

test "CSR: stress test" {
    const allocator = std.testing.allocator;

    var edge_list = std.ArrayList(Edge(i32)).init(allocator);
    defer edge_list.deinit();

    // Create a dense graph: 100 vertices, each connected to next 10
    for (0..100) |i| {
        for (1..11) |j| {
            const target = (i + j) % 100;
            try edge_list.append(.{
                .from = i,
                .to = target,
                .weight = @intCast(i * 100 + target),
            });
        }
    }

    var csr = try CompressedSparseRow(i32).fromEdges(
        allocator,
        100,
        edge_list.items,
        true,
    );
    defer csr.deinit();

    try std.testing.expectEqual(@as(usize, 100), csr.vertexCount());
    try std.testing.expectEqual(@as(usize, 1000), csr.edgeCount());

    // Verify some edges
    try std.testing.expect(csr.hasEdge(0, 1));
    try std.testing.expect(csr.hasEdge(50, 55));
    try std.testing.expect(!csr.hasEdge(0, 50));

    try csr.validate();
}
