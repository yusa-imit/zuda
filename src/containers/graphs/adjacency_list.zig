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

// -- Convenience Constructor Tests --

/// Context for i32 vertices with auto-hash and equality.
/// Time: O(1) | Space: O(1)
const I32Context = struct {
    pub fn hash(_: @This(), key: i32) u64 {
        return std.hash.Wyhash.hash(0, std.mem.asBytes(&key));
    }

    pub fn eql(_: @This(), a: i32, b: i32) bool {
        return a == b;
    }
};

/// Context for string vertices with Wyhash and memcmp equality.
/// Time: O(1) hash, O(n) eql | Space: O(1)
const StringContext = struct {
    pub fn hash(_: @This(), key: []const u8) u64 {
        return std.hash.Wyhash.hash(0, key);
    }

    pub fn eql(_: @This(), a: []const u8, b: []const u8) bool {
        return std.mem.eql(u8, a, b);
    }
};

/// Creates an AdjacencyList with auto-context for i32 vertices (directed).
/// Time: O(1) | Space: O(1)
pub fn IntDirectedGraph(comptime W: type) type {
    return AdjacencyList(i32, W, I32Context, I32Context.hash, I32Context.eql);
}

/// Creates an AdjacencyList with auto-context for i32 vertices (undirected).
/// Time: O(1) | Space: O(1)
pub fn IntUndirectedGraph(comptime W: type) type {
    return AdjacencyList(i32, W, I32Context, I32Context.hash, I32Context.eql);
}

/// Creates an AdjacencyList with string context for []const u8 vertices (directed).
/// Time: O(1) | Space: O(1)
pub fn StringDirectedGraph(comptime W: type) type {
    return AdjacencyList([]const u8, W, StringContext, StringContext.hash, StringContext.eql);
}

/// Creates an AdjacencyList with string context for []const u8 vertices (undirected).
/// Time: O(1) | Space: O(1)
pub fn StringUndirectedGraph(comptime W: type) type {
    return AdjacencyList([]const u8, W, StringContext, StringContext.hash, StringContext.eql);
}

test "initDirected: i32 vertices unweighted" {
    var graph = IntDirectedGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addVertex(10);
    try graph.addVertex(20);
    try graph.addVertex(30);

    try testing.expectEqual(@as(usize, 3), graph.vertexCount());
    try testing.expect(graph.containsVertex(10));
    try testing.expect(graph.containsVertex(20));
    try testing.expect(graph.containsVertex(30));
    try testing.expect(!graph.containsVertex(99));

    try graph.validate();
}

test "initDirected: i32 vertices weighted" {
    var graph = IntDirectedGraph(i64).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addEdge(1, 2, 100);
    try graph.addEdge(2, 3, 200);
    try graph.addEdge(1, 3, 50);

    try testing.expectEqual(@as(usize, 3), graph.vertexCount());
    try testing.expectEqual(@as(usize, 3), graph.edgeCount());

    try testing.expectEqual(@as(?i64, 100), graph.getEdgeWeight(1, 2));
    try testing.expectEqual(@as(?i64, 200), graph.getEdgeWeight(2, 3));
    try testing.expectEqual(@as(?i64, 50), graph.getEdgeWeight(1, 3));
    try testing.expectEqual(@as(?i64, null), graph.getEdgeWeight(2, 1));

    try graph.validate();
}

test "initDirected: i32 vertices directed semantics" {
    var graph = IntDirectedGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addEdge(5, 10, {});
    try graph.addEdge(10, 15, {});

    // Directed: edges are one-way only
    try testing.expect(graph.containsEdge(5, 10));
    try testing.expect(!graph.containsEdge(10, 5));
    try testing.expect(graph.containsEdge(10, 15));
    try testing.expect(!graph.containsEdge(15, 10));

    try graph.validate();
}

test "initUndirected: i32 vertices unweighted" {
    var graph = IntUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try graph.addVertex(100);
    try graph.addVertex(200);
    try graph.addVertex(300);

    try testing.expectEqual(@as(usize, 3), graph.vertexCount());
    try testing.expect(graph.containsVertex(100));
    try testing.expect(graph.containsVertex(200));
    try testing.expect(graph.containsVertex(300));

    try graph.validate();
}

test "initUndirected: i32 vertices weighted" {
    var graph = IntUndirectedGraph(f32).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try graph.addEdge(1, 2, 1.5);
    try graph.addEdge(2, 3, 2.5);
    try graph.addEdge(1, 3, 3.5);

    try testing.expectEqual(@as(usize, 3), graph.vertexCount());
    try testing.expectEqual(@as(usize, 3), graph.edgeCount());

    try testing.expectEqual(@as(?f32, 1.5), graph.getEdgeWeight(1, 2));
    try testing.expectEqual(@as(?f32, 1.5), graph.getEdgeWeight(2, 1));
    try testing.expectEqual(@as(?f32, 2.5), graph.getEdgeWeight(2, 3));
    try testing.expectEqual(@as(?f32, 2.5), graph.getEdgeWeight(3, 2));

    try graph.validate();
}

test "initUndirected: i32 vertices bidirectional edges" {
    var graph = IntUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try graph.addEdge(7, 8, {});
    try graph.addEdge(8, 9, {});

    // Undirected: edges work both ways
    try testing.expect(graph.containsEdge(7, 8));
    try testing.expect(graph.containsEdge(8, 7));
    try testing.expect(graph.containsEdge(8, 9));
    try testing.expect(graph.containsEdge(9, 8));

    try graph.validate();
}

test "initDirected: string vertices unweighted" {
    var graph = StringDirectedGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addVertex("alpha");
    try graph.addVertex("beta");
    try graph.addVertex("gamma");

    try testing.expectEqual(@as(usize, 3), graph.vertexCount());
    try testing.expect(graph.containsVertex("alpha"));
    try testing.expect(graph.containsVertex("beta"));
    try testing.expect(graph.containsVertex("gamma"));
    try testing.expect(!graph.containsVertex("delta"));
}

test "initDirected: string vertices weighted" {
    var graph = StringDirectedGraph(i32).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addEdge("start", "mid", 10);
    try graph.addEdge("mid", "end", 20);
    try graph.addEdge("start", "end", 15);

    try testing.expectEqual(@as(usize, 3), graph.vertexCount());
    try testing.expectEqual(@as(usize, 3), graph.edgeCount());

    try testing.expectEqual(@as(?i32, 10), graph.getEdgeWeight("start", "mid"));
    try testing.expectEqual(@as(?i32, 20), graph.getEdgeWeight("mid", "end"));
    try testing.expectEqual(@as(?i32, 15), graph.getEdgeWeight("start", "end"));
    try testing.expectEqual(@as(?i32, null), graph.getEdgeWeight("mid", "start"));

}

test "initDirected: string vertices directed semantics" {
    var graph = StringDirectedGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addEdge("alice", "bob", {});
    try graph.addEdge("bob", "charlie", {});

    // Directed: one-way edges
    try testing.expect(graph.containsEdge("alice", "bob"));
    try testing.expect(!graph.containsEdge("bob", "alice"));
    try testing.expect(graph.containsEdge("bob", "charlie"));
    try testing.expect(!graph.containsEdge("charlie", "bob"));

}

test "initUndirected: string vertices unweighted" {
    var graph = StringUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try graph.addVertex("node1");
    try graph.addVertex("node2");
    try graph.addVertex("node3");

    try testing.expectEqual(@as(usize, 3), graph.vertexCount());
    try testing.expect(graph.containsVertex("node1"));
    try testing.expect(graph.containsVertex("node2"));
    try testing.expect(graph.containsVertex("node3"));

}

test "initUndirected: string vertices weighted" {
    var graph = StringUndirectedGraph(f64).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try graph.addEdge("city_a", "city_b", 5.5);
    try graph.addEdge("city_b", "city_c", 3.2);
    try graph.addEdge("city_a", "city_c", 7.1);

    try testing.expectEqual(@as(usize, 3), graph.vertexCount());
    try testing.expectEqual(@as(usize, 3), graph.edgeCount());

    try testing.expectEqual(@as(?f64, 5.5), graph.getEdgeWeight("city_a", "city_b"));
    try testing.expectEqual(@as(?f64, 5.5), graph.getEdgeWeight("city_b", "city_a"));
    try testing.expectEqual(@as(?f64, 3.2), graph.getEdgeWeight("city_b", "city_c"));
    try testing.expectEqual(@as(?f64, 3.2), graph.getEdgeWeight("city_c", "city_b"));

}

test "initUndirected: string vertices bidirectional edges" {
    var graph = StringUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try graph.addEdge("x", "y", {});
    try graph.addEdge("y", "z", {});

    // Undirected: edges work in both directions
    try testing.expect(graph.containsEdge("x", "y"));
    try testing.expect(graph.containsEdge("y", "x"));
    try testing.expect(graph.containsEdge("y", "z"));
    try testing.expect(graph.containsEdge("z", "y"));

}

test "initDirected: i32 remove vertex" {
    var graph = IntDirectedGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addEdge(1, 2, {});
    try graph.addEdge(2, 3, {});
    try graph.addEdge(1, 3, {});

    try graph.removeVertex(2);

    try testing.expectEqual(@as(usize, 2), graph.vertexCount());
    try testing.expect(!graph.containsVertex(2));
    try testing.expect(graph.containsVertex(1));
    try testing.expect(graph.containsVertex(3));
    try testing.expect(graph.containsEdge(1, 3));
    try testing.expect(!graph.containsEdge(1, 2));
    try testing.expect(!graph.containsEdge(2, 3));

    try graph.validate();
}

test "initDirected: string remove vertex" {
    var graph = StringDirectedGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addEdge("a", "b", {});
    try graph.addEdge("b", "c", {});
    try graph.addEdge("a", "c", {});

    try graph.removeVertex("b");

    try testing.expectEqual(@as(usize, 2), graph.vertexCount());
    try testing.expect(!graph.containsVertex("b"));
    try testing.expect(graph.containsVertex("a"));
    try testing.expect(graph.containsVertex("c"));
    try testing.expect(graph.containsEdge("a", "c"));

}

test "initUndirected: i32 remove vertex" {
    var graph = IntUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try graph.addEdge(10, 20, {});
    try graph.addEdge(20, 30, {});
    try graph.addEdge(10, 30, {});

    try graph.removeVertex(20);

    try testing.expectEqual(@as(usize, 2), graph.vertexCount());
    try testing.expect(!graph.containsVertex(20));
    try testing.expect(graph.containsVertex(10));
    try testing.expect(graph.containsVertex(30));
    try testing.expect(graph.containsEdge(10, 30));
    try testing.expect(graph.containsEdge(30, 10));

    try graph.validate();
}

test "initUndirected: string remove vertex" {
    var graph = StringUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try graph.addEdge("foo", "bar", {});
    try graph.addEdge("bar", "baz", {});
    try graph.addEdge("foo", "baz", {});

    try graph.removeVertex("bar");

    try testing.expectEqual(@as(usize, 2), graph.vertexCount());
    try testing.expect(!graph.containsVertex("bar"));
    try testing.expect(graph.containsVertex("foo"));
    try testing.expect(graph.containsVertex("baz"));
    try testing.expect(graph.containsEdge("foo", "baz"));
    try testing.expect(graph.containsEdge("baz", "foo"));

}

test "initDirected: i32 remove edge" {
    var graph = IntDirectedGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addEdge(5, 6, {});
    try graph.addEdge(6, 7, {});
    try graph.addEdge(5, 7, {});

    try graph.removeEdge(5, 6);

    try testing.expectEqual(@as(usize, 3), graph.vertexCount());
    try testing.expectEqual(@as(usize, 2), graph.edgeCount());
    try testing.expect(!graph.containsEdge(5, 6));
    try testing.expect(graph.containsEdge(6, 7));
    try testing.expect(graph.containsEdge(5, 7));

    try graph.validate();
}

test "initDirected: string remove edge" {
    var graph = StringDirectedGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addEdge("u", "v", {});
    try graph.addEdge("v", "w", {});
    try graph.addEdge("u", "w", {});

    try graph.removeEdge("u", "v");

    try testing.expectEqual(@as(usize, 3), graph.vertexCount());
    try testing.expectEqual(@as(usize, 2), graph.edgeCount());
    try testing.expect(!graph.containsEdge("u", "v"));
    try testing.expect(graph.containsEdge("v", "w"));
    try testing.expect(graph.containsEdge("u", "w"));

}

test "initUndirected: i32 remove edge" {
    var graph = IntUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try graph.addEdge(11, 12, {});
    try graph.addEdge(12, 13, {});
    try graph.addEdge(11, 13, {});

    try graph.removeEdge(11, 12);

    try testing.expectEqual(@as(usize, 3), graph.vertexCount());
    try testing.expectEqual(@as(usize, 2), graph.edgeCount());
    try testing.expect(!graph.containsEdge(11, 12));
    try testing.expect(!graph.containsEdge(12, 11));
    try testing.expect(graph.containsEdge(12, 13));

    try graph.validate();
}

test "initUndirected: string remove edge" {
    var graph = StringUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try graph.addEdge("p", "q", {});
    try graph.addEdge("q", "r", {});
    try graph.addEdge("p", "r", {});

    try graph.removeEdge("p", "q");

    try testing.expectEqual(@as(usize, 3), graph.vertexCount());
    try testing.expectEqual(@as(usize, 2), graph.edgeCount());
    try testing.expect(!graph.containsEdge("p", "q"));
    try testing.expect(!graph.containsEdge("q", "p"));
    try testing.expect(graph.containsEdge("q", "r"));

}

test "initDirected: i32 neighbors and degree" {
    var graph = IntDirectedGraph(i32).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addEdge(1, 2, 10);
    try graph.addEdge(1, 3, 20);
    try graph.addEdge(2, 3, 30);

    try testing.expectEqual(@as(usize, 2), graph.outDegree(1));
    try testing.expectEqual(@as(usize, 1), graph.outDegree(2));
    try testing.expectEqual(@as(usize, 0), graph.outDegree(3));

    try testing.expectEqual(@as(usize, 0), graph.inDegree(1));
    try testing.expectEqual(@as(usize, 1), graph.inDegree(2));
    try testing.expectEqual(@as(usize, 2), graph.inDegree(3));

    const neighbors_1 = graph.getNeighbors(1).?;
    try testing.expectEqual(@as(usize, 2), neighbors_1.len);

    try graph.validate();
}

test "initDirected: string neighbors and degree" {
    var graph = StringDirectedGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addEdge("root", "left", {});
    try graph.addEdge("root", "right", {});
    try graph.addEdge("left", "leaf", {});

    try testing.expectEqual(@as(usize, 2), graph.outDegree("root"));
    try testing.expectEqual(@as(usize, 1), graph.outDegree("left"));
    try testing.expectEqual(@as(usize, 0), graph.outDegree("right"));
    try testing.expectEqual(@as(usize, 0), graph.outDegree("leaf"));

    try testing.expectEqual(@as(usize, 1), graph.inDegree("left"));
    try testing.expectEqual(@as(usize, 1), graph.inDegree("right"));
    try testing.expectEqual(@as(usize, 1), graph.inDegree("leaf"));

}

test "initUndirected: i32 neighbors and degree" {
    var graph = IntUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try graph.addEdge(1, 2, {});
    try graph.addEdge(1, 3, {});
    try graph.addEdge(2, 3, {});

    // In undirected graphs, inDegree = outDegree
    try testing.expectEqual(@as(usize, 2), graph.outDegree(1));
    try testing.expectEqual(@as(usize, 2), graph.inDegree(1));
    try testing.expectEqual(@as(usize, 2), graph.outDegree(2));
    try testing.expectEqual(@as(usize, 2), graph.inDegree(2));
    try testing.expectEqual(@as(usize, 2), graph.outDegree(3));
    try testing.expectEqual(@as(usize, 2), graph.inDegree(3));

    try graph.validate();
}

test "initUndirected: string neighbors and degree" {
    var graph = StringUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try graph.addEdge("a", "b", {});
    try graph.addEdge("b", "c", {});
    try graph.addEdge("c", "a", {});

    // All have degree 2 (complete triangle)
    try testing.expectEqual(@as(usize, 2), graph.outDegree("a"));
    try testing.expectEqual(@as(usize, 2), graph.inDegree("a"));
    try testing.expectEqual(@as(usize, 2), graph.outDegree("b"));
    try testing.expectEqual(@as(usize, 2), graph.inDegree("b"));
    try testing.expectEqual(@as(usize, 2), graph.outDegree("c"));
    try testing.expectEqual(@as(usize, 2), graph.inDegree("c"));

}

test "initDirected: i32 vertex iterator" {
    var graph = IntDirectedGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addVertex(42);
    try graph.addVertex(43);
    try graph.addVertex(44);

    var count: usize = 0;
    var it = graph.vertexIterator();
    while (it.next()) |_| {
        count += 1;
    }

    try testing.expectEqual(@as(usize, 3), count);
    try graph.validate();
}

test "initDirected: string vertex iterator" {
    var graph = StringDirectedGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addVertex("first");
    try graph.addVertex("second");
    try graph.addVertex("third");

    var count: usize = 0;
    var it = graph.vertexIterator();
    while (it.next()) |_| {
        count += 1;
    }

    try testing.expectEqual(@as(usize, 3), count);
}

test "initUndirected: i32 vertex iterator" {
    var graph = IntUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try graph.addVertex(50);
    try graph.addVertex(51);

    var count: usize = 0;
    var it = graph.vertexIterator();
    while (it.next()) |_| {
        count += 1;
    }

    try testing.expectEqual(@as(usize, 2), count);
    try graph.validate();
}

test "initUndirected: string vertex iterator" {
    var graph = StringUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try graph.addVertex("north");
    try graph.addVertex("south");

    var count: usize = 0;
    var it = graph.vertexIterator();
    while (it.next()) |_| {
        count += 1;
    }

    try testing.expectEqual(@as(usize, 2), count);
}

test "initDirected: i32 empty graph" {
    var graph = IntDirectedGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try testing.expect(graph.isEmpty());
    try testing.expectEqual(@as(usize, 0), graph.vertexCount());
    try testing.expectEqual(@as(usize, 0), graph.edgeCount());

    try graph.validate();
}

test "initDirected: string empty graph" {
    var graph = StringDirectedGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try testing.expect(graph.isEmpty());
    try testing.expectEqual(@as(usize, 0), graph.vertexCount());
    try testing.expectEqual(@as(usize, 0), graph.edgeCount());

}

test "initUndirected: i32 empty graph" {
    var graph = IntUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try testing.expect(graph.isEmpty());
    try testing.expectEqual(@as(usize, 0), graph.vertexCount());
    try testing.expectEqual(@as(usize, 0), graph.edgeCount());

    try graph.validate();
}

test "initUndirected: string empty graph" {
    var graph = StringUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try testing.expect(graph.isEmpty());
    try testing.expectEqual(@as(usize, 0), graph.vertexCount());
    try testing.expectEqual(@as(usize, 0), graph.edgeCount());

}

test "initDirected: i32 duplicate vertex error" {
    var graph = IntDirectedGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addVertex(1);
    try testing.expectError(error.VertexExists, graph.addVertex(1));

    try graph.validate();
}

test "initDirected: string duplicate vertex error" {
    var graph = StringDirectedGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addVertex("dup");
    try testing.expectError(error.VertexExists, graph.addVertex("dup"));

}

test "initUndirected: i32 duplicate vertex error" {
    var graph = IntUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try graph.addVertex(99);
    try testing.expectError(error.VertexExists, graph.addVertex(99));

    try graph.validate();
}

test "initUndirected: string duplicate vertex error" {
    var graph = StringUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try graph.addVertex("same");
    try testing.expectError(error.VertexExists, graph.addVertex("same"));

}

test "initDirected: i32 remove nonexistent vertex error" {
    var graph = IntDirectedGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try testing.expectError(error.VertexNotFound, graph.removeVertex(999));

    try graph.validate();
}

test "initDirected: string remove nonexistent vertex error" {
    var graph = StringDirectedGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try testing.expectError(error.VertexNotFound, graph.removeVertex("missing"));

}

test "initUndirected: i32 remove nonexistent vertex error" {
    var graph = IntUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try testing.expectError(error.VertexNotFound, graph.removeVertex(111));

    try graph.validate();
}

test "initUndirected: string remove nonexistent vertex error" {
    var graph = StringUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try testing.expectError(error.VertexNotFound, graph.removeVertex("absent"));

}

test "initDirected: i32 remove nonexistent edge error" {
    var graph = IntDirectedGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addVertex(1);
    try graph.addVertex(2);
    try testing.expectError(error.EdgeNotFound, graph.removeEdge(1, 2));

    try graph.validate();
}

test "initDirected: string remove nonexistent edge error" {
    var graph = StringDirectedGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addVertex("x");
    try graph.addVertex("y");
    try testing.expectError(error.EdgeNotFound, graph.removeEdge("x", "y"));

}

test "initUndirected: i32 remove nonexistent edge error" {
    var graph = IntUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try graph.addVertex(5);
    try graph.addVertex(6);
    try testing.expectError(error.EdgeNotFound, graph.removeEdge(5, 6));

    try graph.validate();
}

test "initUndirected: string remove nonexistent edge error" {
    var graph = StringUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    try graph.addVertex("src");
    try graph.addVertex("dst");
    try testing.expectError(error.EdgeNotFound, graph.removeEdge("src", "dst"));

}

test "initDirected: i32 large graph memory leak check" {
    var graph = IntDirectedGraph(i64).init(testing.allocator, .{}, true);
    defer graph.deinit();

    for (0..100) |i| {
        const v = @as(i32, @intCast(i));
        try graph.addVertex(v);
        if (i > 0) {
            try graph.addEdge(v - 1, v, @as(i64, @intCast(i * 10)));
        }
    }

    try testing.expectEqual(@as(usize, 100), graph.vertexCount());
    try testing.expectEqual(@as(usize, 99), graph.edgeCount());

    try graph.validate();
}

test "initDirected: string large graph memory leak check" {
    var graph = StringDirectedGraph(i32).init(testing.allocator, .{}, true);
    defer graph.deinit();

    var vertices: std.ArrayList([]const u8) = .{};
    defer vertices.deinit(testing.allocator);
    defer for (vertices.items) |v| testing.allocator.free(v);

    for (0..50) |i| {
        var buffer: [20]u8 = undefined;
        const label = try std.fmt.bufPrint(&buffer, "v_{d}", .{i});
        const owned_label = try testing.allocator.dupe(u8, label);
        try vertices.append(testing.allocator, owned_label);
        try graph.addVertex(owned_label);
        if (i > 0) {
            try graph.addEdge(vertices.items[i - 1], owned_label, @as(i32, @intCast(i)));
        }
    }

    try testing.expectEqual(@as(usize, 50), graph.vertexCount());
    try testing.expectEqual(@as(usize, 49), graph.edgeCount());
}

test "initUndirected: i32 large graph memory leak check" {
    var graph = IntUndirectedGraph(f32).init(testing.allocator, .{}, false);
    defer graph.deinit();

    for (0..80) |i| {
        const v = @as(i32, @intCast(i));
        try graph.addVertex(v);
        if (i > 0) {
            try graph.addEdge(v - 1, v, @as(f32, @floatFromInt(i)) * 1.5);
        }
    }

    try testing.expectEqual(@as(usize, 80), graph.vertexCount());
    try testing.expectEqual(@as(usize, 79), graph.edgeCount());

    try graph.validate();
}

test "initUndirected: string large graph memory leak check" {
    var graph = StringUndirectedGraph(void).init(testing.allocator, .{}, false);
    defer graph.deinit();

    var vertices: std.ArrayList([]const u8) = .{};
    defer vertices.deinit(testing.allocator);
    defer for (vertices.items) |v| testing.allocator.free(v);

    for (0..60) |i| {
        var buffer: [20]u8 = undefined;
        const label = try std.fmt.bufPrint(&buffer, "node_{d}", .{i});
        const owned_label = try testing.allocator.dupe(u8, label);
        try vertices.append(testing.allocator, owned_label);
        try graph.addVertex(owned_label);
        if (i > 0) {
            try graph.addEdge(vertices.items[i - 1], owned_label, {});
        }
    }

    try testing.expectEqual(@as(usize, 60), graph.vertexCount());
    try testing.expectEqual(@as(usize, 59), graph.edgeCount());
}
