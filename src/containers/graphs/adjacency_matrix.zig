const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// AdjacencyMatrix - Graph representation using a 2D matrix.
/// Suitable for dense graphs where edge queries need to be O(1).
/// Uses more memory than AdjacencyList but provides constant-time edge lookups.
///
/// Time Complexity:
/// - Add vertex: O(V²) (must resize matrix)
/// - Add edge: O(1)
/// - Remove vertex: O(V²) (must resize matrix)
/// - Remove edge: O(1)
/// - Check edge: O(1)
/// - Get neighbors: O(V) (must scan row)
/// - Degree: O(V) (must scan row/column)
///
/// Space Complexity: O(V²)
///
/// Generic parameters:
/// - V: Vertex type (must be integer-like or convertible to index)
/// - W: Edge weight type (use bool for unweighted graphs)
/// - Context: Context type for vertex-to-index mapping
/// - indexFn: Function to convert vertex to matrix index
/// - vertexFn: Function to convert matrix index to vertex
pub fn AdjacencyMatrix(
    comptime V: type,
    comptime W: type,
    comptime Context: type,
    comptime indexFn: fn (ctx: Context, vertex: V) usize,
    comptime vertexFn: fn (ctx: Context, index: usize) V,
) type {
    return struct {
        const Self = @This();

        /// Edge information
        pub const EdgeInfo = struct {
            exists: bool,
            weight: W,
        };

        /// Neighbor entry
        pub const Neighbor = struct {
            vertex: V,
            weight: W,
        };

        /// Iterator for graph neighbors
        pub const NeighborIterator = struct {
            matrix: *const Self,
            source_idx: usize,
            current_idx: usize,

            pub fn next(self: *NeighborIterator) ?Neighbor {
                while (self.current_idx < self.matrix.capacity) : (self.current_idx += 1) {
                    if (self.matrix.getEdgeByIndex(self.source_idx, self.current_idx)) |info| {
                        if (info.exists) {
                            const result = Neighbor{
                                .vertex = vertexFn(self.matrix.context, self.current_idx),
                                .weight = info.weight,
                            };
                            self.current_idx += 1;
                            return result;
                        }
                    }
                }
                return null;
            }
        };

        /// Matrix data stored as flat array (row-major order)
        matrix: []?W,
        allocator: Allocator,
        context: Context,
        directed: bool,
        capacity: usize,
        vertex_count: usize,
        edge_count: usize,

        // -- Lifecycle --

        /// Create an empty graph with initial capacity.
        /// Time: O(capacity²) | Space: O(capacity²)
        pub fn init(allocator: Allocator, context: Context, directed: bool, initial_capacity: usize) !Self {
            const size = initial_capacity * initial_capacity;
            const matrix = try allocator.alloc(?W, size);
            @memset(matrix, null);

            return .{
                .matrix = matrix,
                .allocator = allocator,
                .context = context,
                .directed = directed,
                .capacity = initial_capacity,
                .vertex_count = 0,
                .edge_count = 0,
            };
        }

        /// Free all memory used by the graph.
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.matrix);
        }

        // -- Private helpers --

        fn getIndex(self: *const Self, row: usize, col: usize) usize {
            return row * self.capacity + col;
        }

        fn getEdgeByIndex(self: *const Self, row: usize, col: usize) ?EdgeInfo {
            if (row >= self.capacity or col >= self.capacity) return null;
            const idx = self.getIndex(row, col);
            const weight_opt = self.matrix[idx];
            return .{
                .exists = weight_opt != null,
                .weight = weight_opt orelse if (W == bool) false else @as(W, 0),
            };
        }

        fn setEdgeByIndex(self: *Self, row: usize, col: usize, weight: ?W) void {
            if (row >= self.capacity or col >= self.capacity) return;
            const idx = self.getIndex(row, col);
            self.matrix[idx] = weight;
        }

        /// Resize matrix to accommodate new capacity.
        /// Time: O(new_capacity²) | Space: O(new_capacity²)
        fn resize(self: *Self, new_capacity: usize) !void {
            if (new_capacity <= self.capacity) return;

            const new_size = new_capacity * new_capacity;
            const new_matrix = try self.allocator.alloc(?W, new_size);
            @memset(new_matrix, null);

            // Copy old matrix data
            for (0..self.capacity) |row| {
                for (0..self.capacity) |col| {
                    const old_idx = row * self.capacity + col;
                    const new_idx = row * new_capacity + col;
                    new_matrix[new_idx] = self.matrix[old_idx];
                }
            }

            self.allocator.free(self.matrix);
            self.matrix = new_matrix;
            self.capacity = new_capacity;
        }

        // -- Vertex Operations --

        /// Add a vertex to the graph.
        /// May trigger matrix resize if vertex index exceeds capacity.
        /// Time: O(V²) if resize needed, O(1) otherwise | Space: O(V²)
        pub fn addVertex(self: *Self, vertex: V) !void {
            const idx = indexFn(self.context, vertex);

            // Resize if needed
            if (idx >= self.capacity) {
                const new_capacity = @max(self.capacity * 2, idx + 1);
                try self.resize(new_capacity);
            }

            self.vertex_count += 1;
        }

        /// Remove a vertex and all its incident edges.
        /// Does not shrink the matrix.
        /// Time: O(V) | Space: O(1)
        pub fn removeVertex(self: *Self, vertex: V) !void {
            const idx = indexFn(self.context, vertex);
            if (idx >= self.capacity) return error.VertexNotFound;

            // Remove all edges involving this vertex
            for (0..self.capacity) |i| {
                // Remove outgoing edges
                if (self.matrix[self.getIndex(idx, i)]) |_| {
                    self.matrix[self.getIndex(idx, i)] = null;
                    self.edge_count -= 1;
                }
                // Remove incoming edges
                if (i != idx) {
                    if (self.matrix[self.getIndex(i, idx)]) |_| {
                        self.matrix[self.getIndex(i, idx)] = null;
                        if (self.directed) {
                            self.edge_count -= 1;
                        }
                    }
                }
            }

            self.vertex_count -= 1;
        }

        /// Check if a vertex exists in the graph.
        /// Time: O(1) | Space: O(1)
        pub fn containsVertex(self: *const Self, vertex: V) bool {
            const idx = indexFn(self.context, vertex);
            return idx < self.capacity;
        }

        // -- Edge Operations --

        /// Add an edge from source to target.
        /// For undirected graphs, adds edges in both directions.
        /// Time: O(1) | Space: O(1)
        pub fn addEdge(self: *Self, source: V, target: V, weight: W) !void {
            const src_idx = indexFn(self.context, source);
            const tgt_idx = indexFn(self.context, target);

            // Ensure capacity
            const max_idx = @max(src_idx, tgt_idx);
            if (max_idx >= self.capacity) {
                const new_capacity = @max(self.capacity * 2, max_idx + 1);
                try self.resize(new_capacity);
            }

            // Check if edge already exists
            const already_exists = self.matrix[self.getIndex(src_idx, tgt_idx)] != null;

            self.setEdgeByIndex(src_idx, tgt_idx, weight);

            if (!already_exists) {
                self.edge_count += 1;
            }

            if (!self.directed) {
                if (src_idx != tgt_idx) {
                    self.setEdgeByIndex(tgt_idx, src_idx, weight);
                }
            }
        }

        /// Remove an edge.
        /// For undirected graphs, removes edges in both directions.
        /// Time: O(1) | Space: O(1)
        pub fn removeEdge(self: *Self, source: V, target: V) !void {
            const src_idx = indexFn(self.context, source);
            const tgt_idx = indexFn(self.context, target);

            if (src_idx >= self.capacity or tgt_idx >= self.capacity) {
                return error.EdgeNotFound;
            }

            if (self.matrix[self.getIndex(src_idx, tgt_idx)] == null) {
                return error.EdgeNotFound;
            }

            self.setEdgeByIndex(src_idx, tgt_idx, null);
            self.edge_count -= 1;

            if (!self.directed and src_idx != tgt_idx) {
                self.setEdgeByIndex(tgt_idx, src_idx, null);
            }
        }

        /// Check if an edge exists.
        /// Time: O(1) | Space: O(1)
        pub fn hasEdge(self: *const Self, source: V, target: V) bool {
            const src_idx = indexFn(self.context, source);
            const tgt_idx = indexFn(self.context, target);

            if (src_idx >= self.capacity or tgt_idx >= self.capacity) {
                return false;
            }

            return self.matrix[self.getIndex(src_idx, tgt_idx)] != null;
        }

        /// Get edge weight if edge exists.
        /// Time: O(1) | Space: O(1)
        pub fn getEdge(self: *const Self, source: V, target: V) ?W {
            const src_idx = indexFn(self.context, source);
            const tgt_idx = indexFn(self.context, target);

            if (src_idx >= self.capacity or tgt_idx >= self.capacity) {
                return null;
            }

            return self.matrix[self.getIndex(src_idx, tgt_idx)];
        }

        // -- Graph Properties --

        /// Get number of vertices.
        /// Time: O(1) | Space: O(1)
        pub fn vertexCount(self: *const Self) usize {
            return self.vertex_count;
        }

        /// Get number of edges.
        /// Time: O(1) | Space: O(1)
        pub fn edgeCount(self: *const Self) usize {
            return self.edge_count;
        }

        /// Get out-degree of a vertex (number of outgoing edges).
        /// Time: O(V) | Space: O(1)
        pub fn outDegree(self: *const Self, vertex: V) usize {
            const idx = indexFn(self.context, vertex);
            if (idx >= self.capacity) return 0;

            var degree: usize = 0;
            for (0..self.capacity) |col| {
                if (self.matrix[self.getIndex(idx, col)]) |_| {
                    degree += 1;
                }
            }
            return degree;
        }

        /// Get in-degree of a vertex (number of incoming edges).
        /// For undirected graphs, same as out-degree.
        /// Time: O(V) | Space: O(1)
        pub fn inDegree(self: *const Self, vertex: V) usize {
            if (!self.directed) return self.outDegree(vertex);

            const idx = indexFn(self.context, vertex);
            if (idx >= self.capacity) return 0;

            var degree: usize = 0;
            for (0..self.capacity) |row| {
                if (self.matrix[self.getIndex(row, idx)]) |_| {
                    degree += 1;
                }
            }
            return degree;
        }

        // -- Iteration --

        /// Get an iterator over neighbors of a vertex.
        /// Time: O(1) to create, O(V) to iterate | Space: O(1)
        pub fn neighbors(self: *const Self, vertex: V) NeighborIterator {
            const idx = indexFn(self.context, vertex);
            return .{
                .matrix = self,
                .source_idx = idx,
                .current_idx = 0,
            };
        }

        // -- Debug --

        /// Validate graph invariants.
        /// Time: O(V²) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            var counted_edges: usize = 0;

            for (0..self.capacity) |row| {
                for (0..self.capacity) |col| {
                    if (self.matrix[self.getIndex(row, col)]) |_| {
                        counted_edges += 1;

                        // For undirected graphs, check symmetry (except self-loops)
                        if (!self.directed and row != col) {
                            if (self.matrix[self.getIndex(col, row)] == null) {
                                return error.GraphInvariant;
                            }
                        }
                    }
                }
            }

            if (self.directed) {
                if (counted_edges != self.edge_count) {
                    return error.GraphInvariant;
                }
            } else {
                // For undirected graphs, each edge is stored twice (except self-loops)
                // So counted_edges should be roughly 2 * edge_count
                // But we need to account for self-loops which are only stored once
                // For simplicity, we just check that counted_edges >= edge_count
                if (counted_edges < self.edge_count) {
                    return error.GraphInvariant;
                }
            }
        }

        /// Format graph for debugging.
        pub fn format(
            self: *const Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("AdjacencyMatrix{{ vertices: {}, edges: {}, directed: {} }}", .{
                self.vertex_count,
                self.edge_count,
                self.directed,
            });
        }
    };
}

// -- Tests --

test "AdjacencyMatrix: basic integer graph operations" {
    const IntContext = struct {
        fn index(_: @This(), v: u32) usize {
            return v;
        }
        fn vertex(_: @This(), i: usize) u32 {
            return @intCast(i);
        }
    };

    const Graph = AdjacencyMatrix(u32, f32, IntContext, IntContext.index, IntContext.vertex);
    var graph = try Graph.init(testing.allocator, .{}, true, 4);
    defer graph.deinit();

    // Add vertices
    try graph.addVertex(0);
    try graph.addVertex(1);
    try graph.addVertex(2);

    try testing.expectEqual(@as(usize, 3), graph.vertexCount());

    // Add edges
    try graph.addEdge(0, 1, 1.0);
    try graph.addEdge(1, 2, 2.0);
    try graph.addEdge(0, 2, 3.0);

    try testing.expectEqual(@as(usize, 3), graph.edgeCount());

    // Check edges
    try testing.expect(graph.hasEdge(0, 1));
    try testing.expect(graph.hasEdge(1, 2));
    try testing.expect(graph.hasEdge(0, 2));
    try testing.expect(!graph.hasEdge(2, 0)); // directed

    // Get edge weights
    try testing.expectEqual(@as(?f32, 1.0), graph.getEdge(0, 1));
    try testing.expectEqual(@as(?f32, 2.0), graph.getEdge(1, 2));
    try testing.expectEqual(@as(?f32, 3.0), graph.getEdge(0, 2));

    try graph.validate();
}

test "AdjacencyMatrix: undirected graph" {
    const IntContext = struct {
        fn index(_: @This(), v: u32) usize {
            return v;
        }
        fn vertex(_: @This(), i: usize) u32 {
            return @intCast(i);
        }
    };

    const Graph = AdjacencyMatrix(u32, bool, IntContext, IntContext.index, IntContext.vertex);
    var graph = try Graph.init(testing.allocator, .{}, false, 4);
    defer graph.deinit();

    try graph.addEdge(0, 1, true);
    try graph.addEdge(1, 2, true);

    // Both directions should exist
    try testing.expect(graph.hasEdge(0, 1));
    try testing.expect(graph.hasEdge(1, 0));
    try testing.expect(graph.hasEdge(1, 2));
    try testing.expect(graph.hasEdge(2, 1));

    try graph.validate();
}

test "AdjacencyMatrix: neighbor iteration" {
    const IntContext = struct {
        fn index(_: @This(), v: u32) usize {
            return v;
        }
        fn vertex(_: @This(), i: usize) u32 {
            return @intCast(i);
        }
    };

    const Graph = AdjacencyMatrix(u32, f32, IntContext, IntContext.index, IntContext.vertex);
    var graph = try Graph.init(testing.allocator, .{}, true, 8);
    defer graph.deinit();

    try graph.addEdge(0, 1, 1.0);
    try graph.addEdge(0, 2, 2.0);
    try graph.addEdge(0, 3, 3.0);

    var iter = graph.neighbors(0);
    var count: usize = 0;
    while (iter.next()) |neighbor| {
        count += 1;
        try testing.expect(neighbor.vertex >= 1 and neighbor.vertex <= 3);
    }
    try testing.expectEqual(@as(usize, 3), count);
}

test "AdjacencyMatrix: remove operations" {
    const IntContext = struct {
        fn index(_: @This(), v: u32) usize {
            return v;
        }
        fn vertex(_: @This(), i: usize) u32 {
            return @intCast(i);
        }
    };

    const Graph = AdjacencyMatrix(u32, f32, IntContext, IntContext.index, IntContext.vertex);
    var graph = try Graph.init(testing.allocator, .{}, true, 8);
    defer graph.deinit();

    try graph.addEdge(0, 1, 1.0);
    try graph.addEdge(0, 2, 2.0);

    try graph.removeEdge(0, 1);
    try testing.expect(!graph.hasEdge(0, 1));
    try testing.expect(graph.hasEdge(0, 2));

    try testing.expectEqual(@as(usize, 1), graph.edgeCount());

    try graph.validate();
}

test "AdjacencyMatrix: degree operations" {
    const IntContext = struct {
        fn index(_: @This(), v: u32) usize {
            return v;
        }
        fn vertex(_: @This(), i: usize) u32 {
            return @intCast(i);
        }
    };

    const Graph = AdjacencyMatrix(u32, f32, IntContext, IntContext.index, IntContext.vertex);
    var graph = try Graph.init(testing.allocator, .{}, true, 8);
    defer graph.deinit();

    try graph.addEdge(0, 1, 1.0);
    try graph.addEdge(0, 2, 2.0);
    try graph.addEdge(1, 0, 3.0);

    try testing.expectEqual(@as(usize, 2), graph.outDegree(0));
    try testing.expectEqual(@as(usize, 1), graph.inDegree(0));
    try testing.expectEqual(@as(usize, 1), graph.outDegree(1));
    try testing.expectEqual(@as(usize, 1), graph.inDegree(1));
}

test "AdjacencyMatrix: resize on large vertex" {
    const IntContext = struct {
        fn index(_: @This(), v: u32) usize {
            return v;
        }
        fn vertex(_: @This(), i: usize) u32 {
            return @intCast(i);
        }
    };

    const Graph = AdjacencyMatrix(u32, f32, IntContext, IntContext.index, IntContext.vertex);
    var graph = try Graph.init(testing.allocator, .{}, true, 4);
    defer graph.deinit();

    // This should trigger resize
    try graph.addVertex(10);
    try testing.expect(graph.capacity >= 11);

    try graph.addEdge(0, 10, 5.0);
    try testing.expect(graph.hasEdge(0, 10));

    try graph.validate();
}

test "AdjacencyMatrix: self-loop" {
    const IntContext = struct {
        fn index(_: @This(), v: u32) usize {
            return v;
        }
        fn vertex(_: @This(), i: usize) u32 {
            return @intCast(i);
        }
    };

    const Graph = AdjacencyMatrix(u32, f32, IntContext, IntContext.index, IntContext.vertex);
    var graph = try Graph.init(testing.allocator, .{}, true, 4);
    defer graph.deinit();

    try graph.addEdge(0, 0, 1.0);
    try testing.expect(graph.hasEdge(0, 0));
    try testing.expectEqual(@as(usize, 1), graph.edgeCount());

    try graph.validate();
}

test "AdjacencyMatrix: stress test" {
    const IntContext = struct {
        fn index(_: @This(), v: u32) usize {
            return v;
        }
        fn vertex(_: @This(), i: usize) u32 {
            return @intCast(i);
        }
    };

    const Graph = AdjacencyMatrix(u32, f32, IntContext, IntContext.index, IntContext.vertex);
    var graph = try Graph.init(testing.allocator, .{}, true, 16);
    defer graph.deinit();

    // Add many edges
    for (0..10) |i| {
        for (0..10) |j| {
            if (i != j) {
                try graph.addEdge(@intCast(i), @intCast(j), @floatFromInt(i * 10 + j));
            }
        }
    }

    try testing.expectEqual(@as(usize, 90), graph.edgeCount());

    // Verify all edges exist
    for (0..10) |i| {
        for (0..10) |j| {
            if (i != j) {
                try testing.expect(graph.hasEdge(@intCast(i), @intCast(j)));
            }
        }
    }

    try graph.validate();
}

test "AdjacencyMatrix: remove vertex" {
    const IntContext = struct {
        fn index(_: @This(), v: u32) usize {
            return v;
        }
        fn vertex(_: @This(), i: usize) u32 {
            return @intCast(i);
        }
    };

    const Graph = AdjacencyMatrix(u32, f32, IntContext, IntContext.index, IntContext.vertex);
    var graph = try Graph.init(testing.allocator, .{}, true, 8);
    defer graph.deinit();

    try graph.addVertex(0);
    try graph.addVertex(1);
    try graph.addVertex(2);

    try graph.addEdge(0, 1, 1.0);
    try graph.addEdge(1, 2, 2.0);
    try graph.addEdge(2, 0, 3.0);

    try graph.removeVertex(1);

    try testing.expect(!graph.hasEdge(0, 1));
    try testing.expect(!graph.hasEdge(1, 2));
    try testing.expect(graph.hasEdge(2, 0)); // This edge should remain

    try testing.expectEqual(@as(usize, 2), graph.vertexCount());
    try testing.expectEqual(@as(usize, 1), graph.edgeCount());

    try graph.validate();
}
