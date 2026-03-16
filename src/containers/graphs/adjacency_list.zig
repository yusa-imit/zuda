const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// AdjacencyList - Graph representation using adjacency lists.
/// Space-efficient for sparse graphs, supports weighted and unweighted edges.
///
/// Time Complexity:
/// - Add vertex: O(1) amortized
/// - Add edge: O(1) amortized
/// - Remove vertex: O(V + E) where V is vertices, E is edges
/// - Remove edge: O(deg(v)) where deg(v) is degree of vertex v
/// - Check edge: O(deg(v))
/// - Get neighbors: O(1) access, O(deg(v)) iteration
/// - Degree: O(1) for out-degree, O(V) for in-degree
///
/// Space Complexity: O(V + E)
///
/// Generic parameters:
/// - V: Vertex type (must be hashable if using hash-based storage)
/// - W: Edge weight type (use void for unweighted graphs)
/// - Context: Context type for hashing/comparing vertices
/// - hashFn: Hash function for vertices
/// - eqlFn: Equality function for vertices
pub fn AdjacencyList(
    comptime V: type,
    comptime W: type,
    comptime Context: type,
    comptime _: fn (ctx: Context, key: V) u64, // hashFn - used by HashMap internally
    comptime eqlFn: fn (ctx: Context, a: V, b: V) bool,
) type {
    return struct {
        const Self = @This();

        /// Edge representation
        pub const Edge = struct {
            target: V,
            weight: W,
        };

        /// Adjacency information for a vertex
        pub const Adjacency = struct {
            edges: std.ArrayList(Edge),
            allocator: Allocator,

            fn init(allocator: Allocator) Adjacency {
                return .{
                    .edges = std.ArrayList(Edge){},
                    .allocator = allocator,
                };
            }

            fn deinit(self: *Adjacency) void {
                self.edges.deinit(self.allocator);
            }
        };

        adjacencies: std.HashMap(V, Adjacency, Context, std.hash_map.default_max_load_percentage),
        allocator: Allocator,
        context: Context,
        directed: bool,
        vertex_count: usize,
        edge_count: usize,

        // -- Lifecycle --

        /// Create an empty graph.
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator, context: Context, directed: bool) Self {
            return .{
                .adjacencies = std.HashMap(V, Adjacency, Context, std.hash_map.default_max_load_percentage).init(allocator),
                .allocator = allocator,
                .context = context,
                .directed = directed,
                .vertex_count = 0,
                .edge_count = 0,
            };
        }

        /// Free all memory used by the graph.
        /// Time: O(V + E) | Space: O(1)
        pub fn deinit(self: *Self) void {
            var it = self.adjacencies.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.deinit();
            }
            self.adjacencies.deinit();
        }

        // -- Vertex Operations --

        /// Add a vertex to the graph.
        /// Returns error.VertexExists if vertex already exists.
        /// Time: O(1) amortized | Space: O(1)
        pub fn addVertex(self: *Self, vertex: V) !void {
            const result = try self.adjacencies.getOrPut(vertex);
            if (result.found_existing) {
                return error.VertexExists;
            }
            result.value_ptr.* = Adjacency.init(self.allocator);
            self.vertex_count += 1;
        }

        /// Remove a vertex and all its incident edges.
        /// Time: O(V + E) | Space: O(1)
        pub fn removeVertex(self: *Self, vertex: V) !void {
            // Get and remove the vertex's adjacency list
            var adjacency = self.adjacencies.fetchRemove(vertex) orelse return error.VertexNotFound;
            const out_degree = adjacency.value.edges.items.len;
            adjacency.value.deinit();
            self.vertex_count -= 1;
            self.edge_count -= out_degree;

            // Remove all edges pointing to this vertex
            var it = self.adjacencies.iterator();
            while (it.next()) |entry| {
                const edges = &entry.value_ptr.edges;
                var i: usize = 0;
                while (i < edges.items.len) {
                    if (eqlFn(self.context, edges.items[i].target, vertex)) {
                        _ = edges.swapRemove(i);
                        if (self.directed) {
                            self.edge_count -= 1;
                        }
                    } else {
                        i += 1;
                    }
                }
            }
        }

        /// Check if a vertex exists in the graph.
        /// Time: O(1) average | Space: O(1)
        pub fn containsVertex(self: *const Self, vertex: V) bool {
            return self.adjacencies.contains(vertex);
        }

        // -- Edge Operations --

        /// Add an edge from source to target.
        /// For undirected graphs, adds edges in both directions.
        /// Time: O(1) amortized | Space: O(1)
        pub fn addEdge(self: *Self, source: V, target: V, weight: W) !void {
            // Ensure both vertices exist
            if (!self.adjacencies.contains(source)) {
                try self.addVertex(source);
            }
            if (!self.adjacencies.contains(target)) {
                try self.addVertex(target);
            }

            // Add edge from source to target
            var source_adj = self.adjacencies.getPtr(source).?;
            try source_adj.edges.append(source_adj.allocator, Edge{ .target = target, .weight = weight });
            self.edge_count += 1;

            // For undirected graphs, add reverse edge
            if (!self.directed) {
                var target_adj = self.adjacencies.getPtr(target).?;
                try target_adj.edges.append(target_adj.allocator, Edge{ .target = source, .weight = weight });
            }
        }

        /// Remove an edge from source to target.
        /// For undirected graphs, removes edges in both directions.
        /// Time: O(deg(source)) | Space: O(1)
        pub fn removeEdge(self: *Self, source: V, target: V) !void {
            var source_adj = self.adjacencies.getPtr(source) orelse return error.VertexNotFound;

            // Find and remove edge
            var found = false;
            for (source_adj.edges.items, 0..) |edge, i| {
                if (eqlFn(self.context, edge.target, target)) {
                    _ = source_adj.edges.swapRemove(i);
                    found = true;
                    self.edge_count -= 1;
                    break;
                }
            }

            if (!found) {
                return error.EdgeNotFound;
            }

            // For undirected graphs, remove reverse edge
            if (!self.directed) {
                var target_adj = self.adjacencies.getPtr(target) orelse return error.VertexNotFound;
                for (target_adj.edges.items, 0..) |edge, i| {
                    if (eqlFn(self.context, edge.target, source)) {
                        _ = target_adj.edges.swapRemove(i);
                        break;
                    }
                }
            }
        }

        /// Check if an edge exists from source to target.
        /// Time: O(deg(source)) | Space: O(1)
        pub fn containsEdge(self: *const Self, source: V, target: V) bool {
            const source_adj = self.adjacencies.getPtr(source) orelse return false;
            for (source_adj.edges.items) |edge| {
                if (eqlFn(self.context, edge.target, target)) {
                    return true;
                }
            }
            return false;
        }

        /// Get edge weight from source to target.
        /// Time: O(deg(source)) | Space: O(1)
        pub fn getEdgeWeight(self: *const Self, source: V, target: V) ?W {
            const source_adj = self.adjacencies.getPtr(source) orelse return null;
            for (source_adj.edges.items) |edge| {
                if (eqlFn(self.context, edge.target, target)) {
                    return edge.weight;
                }
            }
            return null;
        }

        // -- Query Operations --

        /// Get the list of edges from a vertex.
        /// Returns a slice (points into internal storage).
        /// Time: O(1) | Space: O(1)
        pub fn getNeighbors(self: *const Self, vertex: V) ?[]const Edge {
            const adjacency = self.adjacencies.getPtr(vertex) orelse return null;
            return adjacency.edges.items;
        }

        /// Get the out-degree of a vertex (number of outgoing edges).
        /// Time: O(1) | Space: O(1)
        pub fn outDegree(self: *const Self, vertex: V) usize {
            const adjacency = self.adjacencies.getPtr(vertex) orelse return 0;
            return adjacency.edges.items.len;
        }

        /// Get the in-degree of a vertex (number of incoming edges).
        /// For undirected graphs, returns the same as out-degree.
        /// Time: O(V + E) | Space: O(1)
        pub fn inDegree(self: *const Self, vertex: V) usize {
            if (!self.directed) {
                return self.outDegree(vertex);
            }

            var count: usize = 0;
            var it = self.adjacencies.iterator();
            while (it.next()) |entry| {
                for (entry.value_ptr.edges.items) |edge| {
                    if (eqlFn(self.context, edge.target, vertex)) {
                        count += 1;
                    }
                }
            }
            return count;
        }

        // -- Capacity --

        /// Returns the number of vertices in the graph.
        /// Time: O(1) | Space: O(1)
        pub fn vertexCount(self: *const Self) usize {
            return self.vertex_count;
        }

        /// Returns the number of edges in the graph.
        /// For undirected graphs, each undirected edge is counted once.
        /// Time: O(1) | Space: O(1)
        pub fn edgeCount(self: *const Self) usize {
            return self.edge_count;
        }

        /// Returns true if the graph has no vertices.
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.vertex_count == 0;
        }

        // -- Iteration --

        /// Iterator over all vertices in the graph.
        pub const VertexIterator = struct {
            it: std.HashMap(V, Adjacency, Context, std.hash_map.default_max_load_percentage).KeyIterator,

            /// Returns next element or null when exhausted.
            /// Time: O(1) amortized | Space: O(1)
            pub fn next(self: *VertexIterator) ?V {
                const ptr = self.it.next() orelse return null;
                return ptr.*;
            }
        };

        /// Get an iterator over all vertices.
        /// Time: O(1) | Space: O(1)
        pub fn vertexIterator(self: *const Self) VertexIterator {
            return .{ .it = self.adjacencies.keyIterator() };
        }

        // -- Debug & Validation --

        /// Validate the graph structure invariants.
        /// Time: O(V + E) | Space: O(V)
        pub fn validate(self: *const Self) !void {
            var vertex_set = std.AutoHashMap(V, void).init(self.allocator);
            defer vertex_set.deinit();

            // Collect all vertices
            var it = self.adjacencies.keyIterator();
            while (it.next()) |vertex| {
                try vertex_set.put(vertex.*, {});
            }

            // Validate vertex count
            if (vertex_set.count() != self.vertex_count) {
                return error.InvalidVertexCount;
            }

            // Validate edges
            var total_edges: usize = 0;
            var adj_it = self.adjacencies.iterator();
            while (adj_it.next()) |entry| {
                for (entry.value_ptr.edges.items) |edge| {
                    // Edge target must be a valid vertex
                    if (!vertex_set.contains(edge.target)) {
                        return error.InvalidEdgeTarget;
                    }
                    total_edges += 1;
                }
            }

            // Validate edge count
            const expected_count = if (self.directed) self.edge_count else self.edge_count * 2;
            if (total_edges != expected_count) {
                return error.InvalidEdgeCount;
            }
        }

        /// Format the graph for debugging.
        /// Time: O(V + E) | Space: O(1)
        pub fn format(self: *const Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;
            try writer.print("AdjacencyList({s}, vertices={}, edges={})", .{
                if (self.directed) "directed" else "undirected",
                self.vertex_count,
                self.edge_count,
            });
        }
    };
}

// -- Helper for common integer vertex graphs --

/// Creates a graph type for integer vertices.
/// Time: O(1) | Space: O(1)
pub fn IntGraph(comptime W: type) type {
    const Context = struct {
        /// Computes hash for the key.
        /// Time: O(1) | Space: O(1)
        pub fn hash(_: @This(), key: u32) u64 {
            return std.hash.Wyhash.hash(0, std.mem.asBytes(&key));
        }
        /// Checks equality of two keys.
        /// Time: O(1) | Space: O(1)
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };
    return AdjacencyList(u32, W, Context, Context.hash, Context.eql);
}

// -- Tests --

test "AdjacencyList: basic directed graph" {
    var graph = IntGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addVertex(1);
    try graph.addVertex(2);
    try graph.addVertex(3);

    try testing.expectEqual(@as(usize, 3), graph.vertexCount());
    try testing.expect(graph.containsVertex(1));
    try testing.expect(graph.containsVertex(2));
    try testing.expect(graph.containsVertex(3));
    try testing.expect(!graph.containsVertex(4));

    try graph.validate();
}

test "AdjacencyList: add edges" {
    var graph = IntGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addEdge(1, 2, {});
    try graph.addEdge(2, 3, {});
    try graph.addEdge(1, 3, {});

    try testing.expectEqual(@as(usize, 3), graph.vertexCount());
    try testing.expectEqual(@as(usize, 3), graph.edgeCount());

    try testing.expect(graph.containsEdge(1, 2));
    try testing.expect(graph.containsEdge(2, 3));
    try testing.expect(graph.containsEdge(1, 3));
    try testing.expect(!graph.containsEdge(2, 1));
    try testing.expect(!graph.containsEdge(3, 1));

    try graph.validate();
}

test "AdjacencyList: weighted edges" {
    var graph = IntGraph(i32).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addEdge(1, 2, 10);
    try graph.addEdge(2, 3, 20);
    try graph.addEdge(1, 3, 5);

    try testing.expectEqual(@as(?i32, 10), graph.getEdgeWeight(1, 2));
    try testing.expectEqual(@as(?i32, 20), graph.getEdgeWeight(2, 3));
    try testing.expectEqual(@as(?i32, 5), graph.getEdgeWeight(1, 3));
    try testing.expectEqual(@as(?i32, null), graph.getEdgeWeight(2, 1));

    try graph.validate();
}

test "AdjacencyList: undirected graph" {
    var graph = IntGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try graph.addEdge(1, 2, {});
    try graph.addEdge(2, 3, {});

    try testing.expectEqual(@as(usize, 3), graph.vertexCount());
    try testing.expectEqual(@as(usize, 2), graph.edgeCount());

    // Undirected edges should work in both directions
    try testing.expect(graph.containsEdge(1, 2));
    try testing.expect(graph.containsEdge(2, 1));
    try testing.expect(graph.containsEdge(2, 3));
    try testing.expect(graph.containsEdge(3, 2));

    try graph.validate();
}

test "AdjacencyList: remove edge" {
    var graph = IntGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addEdge(1, 2, {});
    try graph.addEdge(2, 3, {});
    try graph.addEdge(1, 3, {});

    try graph.removeEdge(1, 2);

    try testing.expectEqual(@as(usize, 2), graph.edgeCount());
    try testing.expect(!graph.containsEdge(1, 2));
    try testing.expect(graph.containsEdge(2, 3));
    try testing.expect(graph.containsEdge(1, 3));

    try graph.validate();
}

test "AdjacencyList: remove vertex" {
    var graph = IntGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addEdge(1, 2, {});
    try graph.addEdge(2, 3, {});
    try graph.addEdge(1, 3, {});
    try graph.addEdge(3, 2, {});

    try graph.removeVertex(2);

    try testing.expectEqual(@as(usize, 2), graph.vertexCount());
    try testing.expectEqual(@as(usize, 1), graph.edgeCount());
    try testing.expect(!graph.containsVertex(2));
    try testing.expect(graph.containsEdge(1, 3));
    try testing.expect(!graph.containsEdge(1, 2));
    try testing.expect(!graph.containsEdge(3, 2));

    try graph.validate();
}

test "AdjacencyList: degree" {
    var graph = IntGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addEdge(1, 2, {});
    try graph.addEdge(1, 3, {});
    try graph.addEdge(2, 3, {});

    try testing.expectEqual(@as(usize, 2), graph.outDegree(1));
    try testing.expectEqual(@as(usize, 1), graph.outDegree(2));
    try testing.expectEqual(@as(usize, 0), graph.outDegree(3));

    try testing.expectEqual(@as(usize, 0), graph.inDegree(1));
    try testing.expectEqual(@as(usize, 1), graph.inDegree(2));
    try testing.expectEqual(@as(usize, 2), graph.inDegree(3));

    try graph.validate();
}

test "AdjacencyList: neighbors" {
    var graph = IntGraph(i32).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addEdge(1, 2, 10);
    try graph.addEdge(1, 3, 20);
    try graph.addEdge(1, 4, 30);

    const neighbors = graph.getNeighbors(1).?;
    try testing.expectEqual(@as(usize, 3), neighbors.len);

    // Check all targets are present (order may vary)
    var found = [_]bool{false} ** 3;
    for (neighbors) |edge| {
        if (edge.target == 2 and edge.weight == 10) found[0] = true;
        if (edge.target == 3 and edge.weight == 20) found[1] = true;
        if (edge.target == 4 and edge.weight == 30) found[2] = true;
    }
    try testing.expect(found[0] and found[1] and found[2]);

    try graph.validate();
}

test "AdjacencyList: vertex iterator" {
    var graph = IntGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addVertex(1);
    try graph.addVertex(2);
    try graph.addVertex(3);

    var count: usize = 0;
    var it = graph.vertexIterator();
    while (it.next()) |_| {
        count += 1;
    }

    try testing.expectEqual(@as(usize, 3), count);
    try graph.validate();
}

test "AdjacencyList: empty graph" {
    var graph = IntGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try testing.expect(graph.isEmpty());
    try testing.expectEqual(@as(usize, 0), graph.vertexCount());
    try testing.expectEqual(@as(usize, 0), graph.edgeCount());

    try graph.validate();
}

test "AdjacencyList: memory leak check" {
    var graph = IntGraph(i32).init(testing.allocator, .{}, true);
    defer graph.deinit();

    // Add many vertices and edges
    for (0..100) |i| {
        const v = @as(u32, @intCast(i));
        try graph.addVertex(v);
        if (i > 0) {
            try graph.addEdge(v, v - 1, @intCast(i * 10));
        }
    }

    try testing.expectEqual(@as(usize, 100), graph.vertexCount());
    try testing.expectEqual(@as(usize, 99), graph.edgeCount());

    try graph.validate();
}

test "AdjacencyList: error cases" {
    var graph = IntGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addVertex(1);

    // Duplicate vertex
    try testing.expectError(error.VertexExists, graph.addVertex(1));

    // Remove non-existent vertex
    try testing.expectError(error.VertexNotFound, graph.removeVertex(999));

    // Remove non-existent edge
    try graph.addVertex(2);
    try testing.expectError(error.EdgeNotFound, graph.removeEdge(1, 2));

    try graph.validate();
}
