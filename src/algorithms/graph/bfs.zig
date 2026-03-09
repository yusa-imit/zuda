const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// BFS - Breadth-First Search
///
/// Traverses a graph in breadth-first order, visiting all vertices reachable from the start vertex.
/// Computes shortest paths (in terms of edge count) from start to all reachable vertices.
///
/// Time Complexity: O(V + E) where V is vertices, E is edges
/// Space Complexity: O(V) for visited set and queue
///
/// Generic parameters:
/// - V: Vertex type (must be hashable)
/// - Context: Context type for hashing/comparing vertices (must have .hash and .eql methods)
pub fn BFS(
    comptime V: type,
    comptime Context: type,
) type {
    return struct {
        const Self = @This();
        const HashMapContext = struct {
            user_ctx: Context,

            pub fn hash(ctx: @This(), key: V) u64 {
                return ctx.user_ctx.hash(key);
            }

            pub fn eql(ctx: @This(), a: V, b: V) bool {
                return ctx.user_ctx.eql(a, b);
            }
        };

        /// BFS result containing traversal information
        pub const Result = struct {
            /// Distance from start vertex to each vertex (in edge count)
            distances: std.HashMap(V, usize, HashMapContext, std.hash_map.default_max_load_percentage),
            /// Parent pointers for path reconstruction
            parents: std.HashMap(V, V, HashMapContext, std.hash_map.default_max_load_percentage),
            /// Visit order (sequence of vertices in BFS order)
            visit_order: std.ArrayList(V),
            allocator: Allocator,
            context: Context,

            pub fn deinit(self: *Result) void {
                self.distances.deinit();
                self.parents.deinit();
                self.visit_order.deinit(self.allocator);
            }

            /// Get the distance to a vertex from the start vertex.
            /// Returns null if the vertex is not reachable.
            pub fn getDistance(self: *const Result, vertex: V) ?usize {
                return self.distances.get(vertex);
            }

            /// Get the parent of a vertex in the BFS tree.
            /// Returns null if the vertex has no parent (is the start vertex or unreachable).
            pub fn getParent(self: *const Result, vertex: V) ?V {
                return self.parents.get(vertex);
            }

            /// Reconstruct the path from start to target.
            /// Returns null if target is not reachable from start.
            /// Caller owns the returned slice.
            pub fn getPath(self: *const Result, target: V) !?[]V {
                if (!self.distances.contains(target)) {
                    return null;
                }

                var path: std.ArrayList(V) = .{};
                errdefer path.deinit(self.allocator);

                var current = target;
                while (true) {
                    try path.append(self.allocator, current);
                    const parent = self.parents.get(current) orelse break;
                    current = parent;
                }

                // Reverse to get path from start to target
                std.mem.reverse(V, path.items);
                return try path.toOwnedSlice(self.allocator);
            }
        };

        /// Run BFS from a start vertex on a graph.
        ///
        /// Graph type must provide:
        /// - `getNeighbors(vertex: V) -> ?[]const Edge` where Edge has `.target: V`
        ///
        /// Time: O(V + E) | Space: O(V)
        pub fn run(
            allocator: Allocator,
            graph: anytype,
            start: V,
            context: Context,
        ) !Result {
            const hm_ctx = HashMapContext{ .user_ctx = context };

            var distances = std.HashMap(V, usize, HashMapContext, std.hash_map.default_max_load_percentage).initContext(allocator, hm_ctx);
            errdefer distances.deinit();

            var parents = std.HashMap(V, V, HashMapContext, std.hash_map.default_max_load_percentage).initContext(allocator, hm_ctx);
            errdefer parents.deinit();

            var visit_order: std.ArrayList(V) = .{};
            errdefer visit_order.deinit(allocator);

            var queue: std.ArrayList(V) = .{};
            defer queue.deinit(allocator);

            // Initialize start vertex
            try distances.put(start, 0);
            try queue.append(allocator, start);
            try visit_order.append(allocator, start);

            // BFS main loop
            var read_idx: usize = 0;
            while (read_idx < queue.items.len) : (read_idx += 1) {
                const current = queue.items[read_idx];
                const current_dist = distances.get(current).?;

                // Get neighbors
                const neighbors = graph.getNeighbors(current) orelse continue;

                for (neighbors) |edge| {
                    const neighbor = edge.target;

                    // Skip if already visited
                    if (distances.contains(neighbor)) continue;

                    // Mark as visited with distance
                    try distances.put(neighbor, current_dist + 1);
                    try parents.put(neighbor, current);
                    try queue.append(allocator, neighbor);
                    try visit_order.append(allocator, neighbor);
                }
            }

            return Result{
                .distances = distances,
                .parents = parents,
                .visit_order = visit_order,
                .allocator = allocator,
                .context = context,
            };
        }

        /// Run BFS from start until goal is found (early termination).
        /// Returns the result with partial traversal information.
        /// If goal is not reachable, traverses the entire reachable component.
        ///
        /// Time: O(V + E) worst case, O(shortest path length) best case | Space: O(V)
        pub fn runToGoal(
            allocator: Allocator,
            graph: anytype,
            start: V,
            goal: V,
            context: Context,
        ) !Result {
            const hm_ctx = HashMapContext{ .user_ctx = context };

            var distances = std.HashMap(V, usize, HashMapContext, std.hash_map.default_max_load_percentage).initContext(allocator, hm_ctx);
            errdefer distances.deinit();

            var parents = std.HashMap(V, V, HashMapContext, std.hash_map.default_max_load_percentage).initContext(allocator, hm_ctx);
            errdefer parents.deinit();

            var visit_order: std.ArrayList(V) = .{};
            errdefer visit_order.deinit(allocator);

            var queue: std.ArrayList(V) = .{};
            defer queue.deinit(allocator);

            // Initialize start vertex
            try distances.put(start, 0);
            try queue.append(allocator, start);
            try visit_order.append(allocator, start);

            // Early exit if start == goal
            if (context.eql(start, goal)) {
                return Result{
                    .distances = distances,
                    .parents = parents,
                    .visit_order = visit_order,
                    .allocator = allocator,
                    .context = context,
                };
            }

            // BFS main loop with early termination
            var read_idx: usize = 0;
            while (read_idx < queue.items.len) : (read_idx += 1) {
                const current = queue.items[read_idx];
                const current_dist = distances.get(current).?;

                // Get neighbors
                const neighbors = graph.getNeighbors(current) orelse continue;

                for (neighbors) |edge| {
                    const neighbor = edge.target;

                    // Skip if already visited
                    if (distances.contains(neighbor)) continue;

                    // Mark as visited
                    try distances.put(neighbor, current_dist + 1);
                    try parents.put(neighbor, current);
                    try queue.append(allocator, neighbor);
                    try visit_order.append(allocator, neighbor);

                    // Early termination if goal found
                    if (context.eql(neighbor, goal)) {
                        return Result{
                            .distances = distances,
                            .parents = parents,
                            .visit_order = visit_order,
                            .allocator = allocator,
                            .context = context,
                        };
                    }
                }
            }

            // Goal not reachable
            return Result{
                .distances = distances,
                .parents = parents,
                .visit_order = visit_order,
                .allocator = allocator,
                .context = context,
            };
        }
    };
}

// -- Tests --

const AdjacencyList = @import("../../containers/graphs/adjacency_list.zig").AdjacencyList;

fn IntGraph(comptime W: type) type {
    const Context = struct {
        pub fn hash(_: @This(), key: i32) u64 {
            return @as(u64, @bitCast(@as(i64, key)));
        }
        pub fn eql(_: @This(), a: i32, b: i32) bool {
            return a == b;
        }
    };
    return AdjacencyList(i32, W, Context, Context.hash, Context.eql);
}

const IntContext = struct {
    pub fn hash(_: @This(), key: i32) u64 {
        return @as(u64, @bitCast(@as(i64, key)));
    }

    pub fn eql(_: @This(), a: i32, b: i32) bool {
        return a == b;
    }
};

const IntBFS = BFS(i32, IntContext);

test "BFS: basic traversal" {
    var graph = IntGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    // Create a simple graph: 1 -> 2 -> 3, 1 -> 4
    try graph.addEdge(1, 2, {});
    try graph.addEdge(2, 3, {});
    try graph.addEdge(1, 4, {});

    var result = try IntBFS.run(testing.allocator, &graph, 1, IntContext{});
    defer result.deinit();

    // Check distances
    try testing.expectEqual(@as(?usize, 0), result.getDistance(1));
    try testing.expectEqual(@as(?usize, 1), result.getDistance(2));
    try testing.expectEqual(@as(?usize, 2), result.getDistance(3));
    try testing.expectEqual(@as(?usize, 1), result.getDistance(4));

    // Check parents
    try testing.expectEqual(@as(?i32, null), result.getParent(1)); // Start has no parent
    try testing.expectEqual(@as(?i32, 1), result.getParent(2));
    try testing.expectEqual(@as(?i32, 2), result.getParent(3));
    try testing.expectEqual(@as(?i32, 1), result.getParent(4));

    // Check visit order (BFS should visit 1, then {2,4} in some order, then 3)
    try testing.expectEqual(@as(usize, 4), result.visit_order.items.len);
    try testing.expectEqual(@as(i32, 1), result.visit_order.items[0]);
}

test "BFS: path reconstruction" {
    var graph = IntGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    // Create a path: 1 -> 2 -> 3 -> 4
    try graph.addEdge(1, 2, {});
    try graph.addEdge(2, 3, {});
    try graph.addEdge(3, 4, {});

    var result = try IntBFS.run(testing.allocator, &graph, 1, IntContext{});
    defer result.deinit();

    const path = (try result.getPath(4)) orelse {
        try testing.expect(false); // Path should exist
        return;
    };
    defer testing.allocator.free(path);

    try testing.expectEqual(@as(usize, 4), path.len);
    try testing.expectEqual(@as(i32, 1), path[0]);
    try testing.expectEqual(@as(i32, 2), path[1]);
    try testing.expectEqual(@as(i32, 3), path[2]);
    try testing.expectEqual(@as(i32, 4), path[3]);
}

test "BFS: disconnected graph" {
    var graph = IntGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    // Two disconnected components: 1 -> 2, 3 -> 4
    try graph.addEdge(1, 2, {});
    try graph.addEdge(3, 4, {});

    var result = try IntBFS.run(testing.allocator, &graph, 1, IntContext{});
    defer result.deinit();

    // Should only reach component containing 1
    try testing.expectEqual(@as(?usize, 0), result.getDistance(1));
    try testing.expectEqual(@as(?usize, 1), result.getDistance(2));
    try testing.expectEqual(@as(?usize, null), result.getDistance(3)); // Unreachable
    try testing.expectEqual(@as(?usize, null), result.getDistance(4)); // Unreachable

    const path_to_3 = try result.getPath(3);
    try testing.expectEqual(@as(?[]i32, null), path_to_3); // No path
}

test "BFS: early termination" {
    var graph = IntGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    // Create a larger graph: 1 -> {2, 3}, 2 -> 4, 3 -> {5, 6}
    try graph.addEdge(1, 2, {});
    try graph.addEdge(1, 3, {});
    try graph.addEdge(2, 4, {});
    try graph.addEdge(3, 5, {});
    try graph.addEdge(3, 6, {});

    var result = try IntBFS.runToGoal(testing.allocator, &graph, 1, 4, IntContext{});
    defer result.deinit();

    // Should find path to 4
    try testing.expectEqual(@as(?usize, 2), result.getDistance(4));

    const path = (try result.getPath(4)) orelse {
        try testing.expect(false);
        return;
    };
    defer testing.allocator.free(path);

    try testing.expectEqual(@as(usize, 3), path.len);
    try testing.expectEqual(@as(i32, 1), path[0]);
    try testing.expectEqual(@as(i32, 2), path[1]);
    try testing.expectEqual(@as(i32, 4), path[2]);
}

test "BFS: cycle handling" {
    var graph = IntGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    // Create a cycle: 1 -> 2 -> 3 -> 1
    try graph.addEdge(1, 2, {});
    try graph.addEdge(2, 3, {});
    try graph.addEdge(3, 1, {});

    var result = try IntBFS.run(testing.allocator, &graph, 1, IntContext{});
    defer result.deinit();

    // All vertices should be reachable
    try testing.expectEqual(@as(?usize, 0), result.getDistance(1));
    try testing.expectEqual(@as(?usize, 1), result.getDistance(2));
    try testing.expectEqual(@as(?usize, 2), result.getDistance(3));

    // Should visit each vertex exactly once
    try testing.expectEqual(@as(usize, 3), result.visit_order.items.len);
}

test "BFS: undirected graph" {
    var graph = IntGraph(void).init(testing.allocator, .{}, false); // Undirected
    defer graph.deinit();

    // Create undirected edges: 1 -- 2 -- 3
    try graph.addEdge(1, 2, {});
    try graph.addEdge(2, 3, {});

    // BFS from vertex 2 should reach both 1 and 3
    var result = try IntBFS.run(testing.allocator, &graph, 2, IntContext{});
    defer result.deinit();

    try testing.expectEqual(@as(?usize, 0), result.getDistance(2));
    try testing.expectEqual(@as(?usize, 1), result.getDistance(1));
    try testing.expectEqual(@as(?usize, 1), result.getDistance(3));
}

test "BFS: single vertex" {
    var graph = IntGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    try graph.addVertex(1);

    var result = try IntBFS.run(testing.allocator, &graph, 1, IntContext{});
    defer result.deinit();

    try testing.expectEqual(@as(?usize, 0), result.getDistance(1));
    try testing.expectEqual(@as(usize, 1), result.visit_order.items.len);
}

test "BFS: self-loop" {
    var graph = IntGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    // Vertex with self-loop: 1 -> 1
    try graph.addEdge(1, 1, {});

    var result = try IntBFS.run(testing.allocator, &graph, 1, IntContext{});
    defer result.deinit();

    // Self-loop should not cause infinite loop
    try testing.expectEqual(@as(?usize, 0), result.getDistance(1));
    try testing.expectEqual(@as(usize, 1), result.visit_order.items.len);
}

test "BFS: stress test" {
    var graph = IntGraph(void).init(testing.allocator, .{}, true);
    defer graph.deinit();

    // Create a chain: 0 -> 1 -> 2 -> ... -> 999
    const n = 1000;
    var i: i32 = 0;
    while (i < n - 1) : (i += 1) {
        try graph.addEdge(i, i + 1, {});
    }

    var result = try IntBFS.run(testing.allocator, &graph, 0, IntContext{});
    defer result.deinit();

    // Check distances are correct
    i = 0;
    while (i < n) : (i += 1) {
        try testing.expectEqual(@as(?usize, @intCast(i)), result.getDistance(i));
    }

    // Check path to last vertex
    const path = (try result.getPath(n - 1)) orelse {
        try testing.expect(false);
        return;
    };
    defer testing.allocator.free(path);

    try testing.expectEqual(@as(usize, n), path.len);
    i = 0;
    while (i < n) : (i += 1) {
        try testing.expectEqual(i, path[@intCast(i)]);
    }
}
