const std = @import("std");
const Allocator = std.mem.Allocator;

/// Edge List graph representation.
/// The simplest graph representation: just a list of edges.
///
/// Best for:
/// - Sparse graphs where edge iteration is primary operation
/// - Algorithms that process all edges (Kruskal's MST, edge-centric algorithms)
/// - Memory-constrained environments
/// - When graph structure changes frequently
///
/// Time complexity:
/// - Add edge: O(1) amortized
/// - Remove edge: O(E)
/// - Has edge: O(E)
/// - Neighbor iteration: O(E)
/// - Get all edges: O(1)
///
/// Space complexity: O(E)
pub fn EdgeList(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Single edge in the graph
        pub const Edge = struct {
            from: usize,
            to: usize,
            weight: ?T,

            pub fn init(from: usize, to: usize, weight: ?T) Edge {
                return .{ .from = from, .to = to, .weight = weight };
            }
        };

        /// Iterator over edges
        pub const Iterator = struct {
            edges: []const Edge,
            index: usize,

            /// Get next edge
            /// Time: O(1)
            pub fn next(self: *Iterator) ?Edge {
                if (self.index >= self.edges.len) return null;
                const edge = self.edges[self.index];
                self.index += 1;
                return edge;
            }
        };

        /// Iterator over neighbors of a vertex
        pub const NeighborIterator = struct {
            edges: []const Edge,
            vertex: usize,
            index: usize,
            directed: bool,

            pub const Entry = struct {
                vertex: usize,
                weight: ?T,
            };

            /// Get next neighbor
            /// Time: O(1) per call, O(E) total
            pub fn next(self: *NeighborIterator) ?Entry {
                while (self.index < self.edges.len) {
                    const edge = self.edges[self.index];
                    self.index += 1;

                    if (edge.from == self.vertex) {
                        return Entry{ .vertex = edge.to, .weight = edge.weight };
                    }
                    if (!self.directed and edge.to == self.vertex) {
                        return Entry{ .vertex = edge.from, .weight = edge.weight };
                    }
                }
                return null;
            }
        };

        allocator: Allocator,
        edges: std.ArrayList(Edge),
        vertex_count: usize,
        directed: bool,

        /// Initialize an empty edge list
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator, vertex_count: usize, directed: bool) Self {
            return .{
                .allocator = allocator,
                .edges = std.ArrayList(Edge).init(allocator),
                .vertex_count = vertex_count,
                .directed = directed,
            };
        }

        /// Deinitialize and free all memory
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.edges.deinit();
            self.* = undefined;
        }

        /// Get number of vertices
        /// Time: O(1) | Space: O(1)
        pub fn vertexCount(self: *const Self) usize {
            return self.vertex_count;
        }

        /// Get number of edges
        /// Time: O(1) | Space: O(1)
        pub fn edgeCount(self: *const Self) usize {
            return self.edges.items.len;
        }

        /// Check if graph is empty
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.edges.items.len == 0;
        }

        /// Add an edge from u to v with optional weight
        /// Time: O(1) amortized | Space: O(1) amortized
        pub fn addEdge(self: *Self, from: usize, to: usize, weight: ?T) !void {
            if (from >= self.vertex_count or to >= self.vertex_count) {
                return error.VertexOutOfBounds;
            }
            try self.edges.append(Edge.init(from, to, weight));
        }

        /// Remove first occurrence of edge from u to v
        /// Returns true if edge was found and removed
        /// Time: O(E) | Space: O(1)
        pub fn removeEdge(self: *Self, from: usize, to: usize) bool {
            for (self.edges.items, 0..) |edge, i| {
                if (edge.from == from and edge.to == to) {
                    _ = self.edges.swapRemove(i);
                    return true;
                }
            }
            return false;
        }

        /// Remove all edges from u to v
        /// Returns number of edges removed
        /// Time: O(E) | Space: O(1)
        pub fn removeAllEdges(self: *Self, from: usize, to: usize) usize {
            var removed: usize = 0;
            var i: usize = 0;
            while (i < self.edges.items.len) {
                if (self.edges.items[i].from == from and self.edges.items[i].to == to) {
                    _ = self.edges.swapRemove(i);
                    removed += 1;
                } else {
                    i += 1;
                }
            }
            return removed;
        }

        /// Check if edge exists from u to v
        /// Time: O(E) | Space: O(1)
        pub fn hasEdge(self: *const Self, from: usize, to: usize) bool {
            for (self.edges.items) |edge| {
                if (edge.from == from and edge.to == to) {
                    return true;
                }
            }
            return false;
        }

        /// Get weight of edge from u to v
        /// Returns null if edge doesn't exist
        /// Time: O(E) | Space: O(1)
        pub fn getEdgeWeight(self: *const Self, from: usize, to: usize) ?T {
            for (self.edges.items) |edge| {
                if (edge.from == from and edge.to == to) {
                    return edge.weight;
                }
            }
            return null;
        }

        /// Update weight of edge from u to v
        /// Returns true if edge was found and updated
        /// Time: O(E) | Space: O(1)
        pub fn setEdgeWeight(self: *Self, from: usize, to: usize, weight: ?T) bool {
            for (self.edges.items) |*edge| {
                if (edge.from == from and edge.to == to) {
                    edge.weight = weight;
                    return true;
                }
            }
            return false;
        }

        /// Get all edges as a slice
        /// Time: O(1) | Space: O(1)
        pub fn getEdges(self: *const Self) []const Edge {
            return self.edges.items;
        }

        /// Get out-degree of a vertex
        /// Time: O(E) | Space: O(1)
        pub fn outDegree(self: *const Self, vertex: usize) usize {
            if (vertex >= self.vertex_count) return 0;
            var count: usize = 0;
            for (self.edges.items) |edge| {
                if (edge.from == vertex) count += 1;
            }
            return count;
        }

        /// Get in-degree of a vertex
        /// Time: O(E) | Space: O(1)
        pub fn inDegree(self: *const Self, vertex: usize) usize {
            if (vertex >= self.vertex_count) return 0;
            var count: usize = 0;
            for (self.edges.items) |edge| {
                if (edge.to == vertex) count += 1;
            }
            return count;
        }

        /// Get degree of a vertex (for undirected graphs)
        /// Time: O(E) | Space: O(1)
        pub fn degree(self: *const Self, vertex: usize) usize {
            if (self.directed) {
                return self.outDegree(vertex) + self.inDegree(vertex);
            }
            if (vertex >= self.vertex_count) return 0;
            var count: usize = 0;
            for (self.edges.items) |edge| {
                if (edge.from == vertex) count += 1;
                if (!self.directed and edge.to == vertex and edge.from != vertex) {
                    count += 1;
                }
            }
            return count;
        }

        /// Create an iterator over all edges
        /// Time: O(1) | Space: O(1)
        pub fn iterator(self: *const Self) Iterator {
            return .{
                .edges = self.edges.items,
                .index = 0,
            };
        }

        /// Create an iterator over neighbors of a vertex
        /// Time: O(1) | Space: O(1)
        pub fn neighbors(self: *const Self, vertex: usize) NeighborIterator {
            return .{
                .edges = self.edges.items,
                .vertex = vertex,
                .index = 0,
                .directed = self.directed,
            };
        }

        /// Clear all edges
        /// Time: O(1) | Space: O(1)
        pub fn clear(self: *Self) void {
            self.edges.clearRetainingCapacity();
        }

        /// Reserve capacity for edges
        /// Time: O(n) | Space: O(n)
        pub fn ensureCapacity(self: *Self, capacity: usize) !void {
            try self.edges.ensureTotalCapacity(capacity);
        }

        /// Sort edges by (from, to) ordering
        /// Useful for improving cache locality in edge iteration
        /// Time: O(E log E) | Space: O(log E)
        pub fn sortEdges(self: *Self) void {
            const lessThan = struct {
                fn lessThan(_: void, a: Edge, b: Edge) bool {
                    if (a.from != b.from) return a.from < b.from;
                    return a.to < b.to;
                }
            }.lessThan;
            std.mem.sort(Edge, self.edges.items, {}, lessThan);
        }

        /// Clone the edge list
        /// Time: O(E) | Space: O(E)
        pub fn clone(self: *const Self) !Self {
            const new_edges = try self.edges.clone();
            return .{
                .allocator = self.allocator,
                .edges = new_edges,
                .vertex_count = self.vertex_count,
                .directed = self.directed,
            };
        }

        /// Reverse the direction of all edges
        /// Time: O(E) | Space: O(1)
        pub fn reverse(self: *Self) void {
            for (self.edges.items) |*edge| {
                const tmp = edge.from;
                edge.from = edge.to;
                edge.to = tmp;
            }
        }

        /// Get transpose (reverse) of the graph
        /// Time: O(E) | Space: O(E)
        pub fn transpose(self: *const Self) !Self {
            var result = try self.clone();
            result.reverse();
            return result;
        }

        /// Validate graph invariants
        /// Time: O(E) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            for (self.edges.items) |edge| {
                if (edge.from >= self.vertex_count) {
                    return error.VertexOutOfBounds;
                }
                if (edge.to >= self.vertex_count) {
                    return error.VertexOutOfBounds;
                }
            }
        }
    };
}

// ============================================================================
// TESTS
// ============================================================================

test "EdgeList: basic operations" {
    const allocator = std.testing.allocator;

    var graph = EdgeList(i32).init(allocator, 4, true);
    defer graph.deinit();

    try graph.addEdge(0, 1, 10);
    try graph.addEdge(0, 2, 20);
    try graph.addEdge(1, 3, 30);

    try std.testing.expectEqual(@as(usize, 4), graph.vertexCount());
    try std.testing.expectEqual(@as(usize, 3), graph.edgeCount());
    try std.testing.expect(!graph.isEmpty());

    try std.testing.expect(graph.hasEdge(0, 1));
    try std.testing.expect(graph.hasEdge(0, 2));
    try std.testing.expect(graph.hasEdge(1, 3));
    try std.testing.expect(!graph.hasEdge(1, 0));

    try graph.validate();
}

test "EdgeList: weighted edges" {
    const allocator = std.testing.allocator;

    var graph = EdgeList(f32).init(allocator, 3, true);
    defer graph.deinit();

    try graph.addEdge(0, 1, 1.5);
    try graph.addEdge(1, 2, 2.5);

    try std.testing.expectEqual(@as(?f32, 1.5), graph.getEdgeWeight(0, 1));
    try std.testing.expectEqual(@as(?f32, 2.5), graph.getEdgeWeight(1, 2));
    try std.testing.expectEqual(@as(?f32, null), graph.getEdgeWeight(1, 0));

    try std.testing.expect(graph.setEdgeWeight(0, 1, 3.5));
    try std.testing.expectEqual(@as(?f32, 3.5), graph.getEdgeWeight(0, 1));

    try graph.validate();
}

test "EdgeList: edge removal" {
    const allocator = std.testing.allocator;

    var graph = EdgeList(i32).init(allocator, 3, true);
    defer graph.deinit();

    try graph.addEdge(0, 1, 10);
    try graph.addEdge(0, 2, 20);
    try graph.addEdge(1, 2, 30);

    try std.testing.expect(graph.removeEdge(0, 1));
    try std.testing.expect(!graph.hasEdge(0, 1));
    try std.testing.expect(graph.hasEdge(0, 2));
    try std.testing.expectEqual(@as(usize, 2), graph.edgeCount());

    try std.testing.expect(!graph.removeEdge(0, 1));

    try graph.validate();
}

test "EdgeList: remove all edges" {
    const allocator = std.testing.allocator;

    var graph = EdgeList(i32).init(allocator, 3, true);
    defer graph.deinit();

    try graph.addEdge(0, 1, 10);
    try graph.addEdge(0, 1, 20);
    try graph.addEdge(0, 1, 30);
    try graph.addEdge(0, 2, 40);

    const removed = graph.removeAllEdges(0, 1);
    try std.testing.expectEqual(@as(usize, 3), removed);
    try std.testing.expect(!graph.hasEdge(0, 1));
    try std.testing.expect(graph.hasEdge(0, 2));
    try std.testing.expectEqual(@as(usize, 1), graph.edgeCount());

    try graph.validate();
}

test "EdgeList: degrees" {
    const allocator = std.testing.allocator;

    var graph = EdgeList(i32).init(allocator, 4, true);
    defer graph.deinit();

    try graph.addEdge(0, 1, null);
    try graph.addEdge(0, 2, null);
    try graph.addEdge(1, 2, null);
    try graph.addEdge(2, 3, null);

    try std.testing.expectEqual(@as(usize, 2), graph.outDegree(0));
    try std.testing.expectEqual(@as(usize, 1), graph.outDegree(1));
    try std.testing.expectEqual(@as(usize, 1), graph.outDegree(2));

    try std.testing.expectEqual(@as(usize, 0), graph.inDegree(0));
    try std.testing.expectEqual(@as(usize, 2), graph.inDegree(2));
    try std.testing.expectEqual(@as(usize, 1), graph.inDegree(3));

    try graph.validate();
}

test "EdgeList: undirected graph degrees" {
    const allocator = std.testing.allocator;

    var graph = EdgeList(i32).init(allocator, 3, false);
    defer graph.deinit();

    try graph.addEdge(0, 1, null);
    try graph.addEdge(1, 2, null);

    try std.testing.expectEqual(@as(usize, 1), graph.degree(0));
    try std.testing.expectEqual(@as(usize, 2), graph.degree(1));
    try std.testing.expectEqual(@as(usize, 1), graph.degree(2));

    try graph.validate();
}

test "EdgeList: edge iteration" {
    const allocator = std.testing.allocator;

    var graph = EdgeList(i32).init(allocator, 3, true);
    defer graph.deinit();

    try graph.addEdge(0, 1, 10);
    try graph.addEdge(1, 2, 20);
    try graph.addEdge(0, 2, 30);

    var iter = graph.iterator();
    var count: usize = 0;
    while (iter.next()) |_| {
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), count);

    try graph.validate();
}

test "EdgeList: neighbor iteration" {
    const allocator = std.testing.allocator;

    var graph = EdgeList(i32).init(allocator, 4, true);
    defer graph.deinit();

    try graph.addEdge(0, 1, 10);
    try graph.addEdge(0, 2, 20);
    try graph.addEdge(0, 3, 30);

    var iter = graph.neighbors(0);
    var count: usize = 0;
    while (iter.next()) |entry| {
        try std.testing.expect(entry.vertex >= 1 and entry.vertex <= 3);
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), count);

    try graph.validate();
}

test "EdgeList: sort edges" {
    const allocator = std.testing.allocator;

    var graph = EdgeList(i32).init(allocator, 4, true);
    defer graph.deinit();

    try graph.addEdge(2, 1, null);
    try graph.addEdge(0, 2, null);
    try graph.addEdge(1, 3, null);
    try graph.addEdge(0, 1, null);

    graph.sortEdges();

    const edges = graph.getEdges();
    try std.testing.expectEqual(@as(usize, 0), edges[0].from);
    try std.testing.expectEqual(@as(usize, 1), edges[0].to);
    try std.testing.expectEqual(@as(usize, 0), edges[1].from);
    try std.testing.expectEqual(@as(usize, 2), edges[1].to);

    try graph.validate();
}

test "EdgeList: reverse and transpose" {
    const allocator = std.testing.allocator;

    var graph = EdgeList(i32).init(allocator, 3, true);
    defer graph.deinit();

    try graph.addEdge(0, 1, 10);
    try graph.addEdge(1, 2, 20);

    graph.reverse();
    try std.testing.expect(graph.hasEdge(1, 0));
    try std.testing.expect(graph.hasEdge(2, 1));
    try std.testing.expect(!graph.hasEdge(0, 1));

    // Test transpose (creates new graph)
    var graph2 = EdgeList(i32).init(allocator, 3, true);
    defer graph2.deinit();
    try graph2.addEdge(0, 1, 10);
    try graph2.addEdge(1, 2, 20);

    var transposed = try graph2.transpose();
    defer transposed.deinit();

    try std.testing.expect(transposed.hasEdge(1, 0));
    try std.testing.expect(transposed.hasEdge(2, 1));
    try std.testing.expect(graph2.hasEdge(0, 1)); // Original unchanged

    try transposed.validate();
}

test "EdgeList: clone" {
    const allocator = std.testing.allocator;

    var graph = EdgeList(i32).init(allocator, 3, true);
    defer graph.deinit();

    try graph.addEdge(0, 1, 10);
    try graph.addEdge(1, 2, 20);

    var cloned = try graph.clone();
    defer cloned.deinit();

    try std.testing.expectEqual(graph.vertexCount(), cloned.vertexCount());
    try std.testing.expectEqual(graph.edgeCount(), cloned.edgeCount());
    try std.testing.expect(cloned.hasEdge(0, 1));
    try std.testing.expect(cloned.hasEdge(1, 2));

    try cloned.validate();
}

test "EdgeList: empty graph" {
    const allocator = std.testing.allocator;

    var graph = EdgeList(i32).init(allocator, 5, true);
    defer graph.deinit();

    try std.testing.expect(graph.isEmpty());
    try std.testing.expectEqual(@as(usize, 5), graph.vertexCount());
    try std.testing.expectEqual(@as(usize, 0), graph.edgeCount());

    try graph.validate();
}

test "EdgeList: clear" {
    const allocator = std.testing.allocator;

    var graph = EdgeList(i32).init(allocator, 3, true);
    defer graph.deinit();

    try graph.addEdge(0, 1, 10);
    try graph.addEdge(1, 2, 20);

    graph.clear();
    try std.testing.expect(graph.isEmpty());
    try std.testing.expectEqual(@as(usize, 0), graph.edgeCount());
    try std.testing.expectEqual(@as(usize, 3), graph.vertexCount());

    try graph.validate();
}

test "EdgeList: self-loop" {
    const allocator = std.testing.allocator;

    var graph = EdgeList(i32).init(allocator, 2, true);
    defer graph.deinit();

    try graph.addEdge(0, 0, 5);
    try graph.addEdge(0, 1, 10);

    try std.testing.expect(graph.hasEdge(0, 0));
    try std.testing.expectEqual(@as(?i32, 5), graph.getEdgeWeight(0, 0));
    try std.testing.expectEqual(@as(usize, 1), graph.outDegree(0));
    try std.testing.expectEqual(@as(usize, 1), graph.inDegree(0));

    try graph.validate();
}

test "EdgeList: stress test" {
    const allocator = std.testing.allocator;

    var graph = EdgeList(i32).init(allocator, 100, true);
    defer graph.deinit();

    // Add 1000 edges
    for (0..1000) |i| {
        const from = i % 100;
        const to = (i * 7) % 100;
        try graph.addEdge(from, to, @intCast(i));
    }

    try std.testing.expectEqual(@as(usize, 100), graph.vertexCount());
    try std.testing.expectEqual(@as(usize, 1000), graph.edgeCount());

    graph.sortEdges();
    try graph.validate();
}
