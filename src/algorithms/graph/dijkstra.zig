const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Dijkstra - Single-source shortest paths for non-negative edge weights.
///
/// Computes shortest paths from a start vertex to all reachable vertices
/// using Dijkstra's algorithm with a priority queue (min-heap).
///
/// Time Complexity: O((V + E) log V) with binary heap, O(E + V log V) with Fibonacci heap
/// Space Complexity: O(V) for distance map, parent map, and priority queue
///
/// Constraints:
/// - All edge weights must be non-negative
/// - For negative weights, use Bellman-Ford instead
///
/// Generic parameters:
/// - V: Vertex type (must be hashable)
/// - W: Weight type (must support comparison and addition)
/// - Context: Context type for hashing/comparing vertices (must have .hash and .eql methods)
pub fn Dijkstra(
    comptime V: type,
    comptime W: type,
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

        /// Priority queue entry for Dijkstra
        const QueueEntry = struct {
            vertex: V,
            distance: W,
        };

        /// Comparison function for min-heap (smallest distance first)
        fn compareDistance(_: void, a: QueueEntry, b: QueueEntry) std.math.Order {
            return std.math.order(a.distance, b.distance);
        }

        /// Dijkstra result containing shortest path information
        pub const Result = struct {
            /// Distance from start vertex to each vertex
            distances: std.HashMap(V, W, HashMapContext, std.hash_map.default_max_load_percentage),
            /// Parent pointers for path reconstruction
            parents: std.HashMap(V, V, HashMapContext, std.hash_map.default_max_load_percentage),
            allocator: Allocator,
            context: Context,

            pub fn deinit(self: *Result) void {
                self.distances.deinit();
                self.parents.deinit();
            }

            /// Get the distance to a vertex from the start vertex.
            /// Returns null if the vertex is not reachable.
            pub fn getDistance(self: *const Result, vertex: V) ?W {
                return self.distances.get(vertex);
            }

            /// Get the parent of a vertex in the shortest path tree.
            /// Returns null if the vertex has no parent (is the start vertex or unreachable).
            pub fn getParent(self: *const Result, vertex: V) ?V {
                return self.parents.get(vertex);
            }

            /// Reconstruct the shortest path from start to target.
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

        /// Run Dijkstra's algorithm from a start vertex on a weighted graph.
        ///
        /// Graph type must provide:
        /// - `getNeighbors(vertex: V) -> ?[]const Edge` where Edge has `.target: V` and `.weight: W`
        ///
        /// Weight type W must support:
        /// - Comparison operators (for std.math.order)
        /// - Addition operator (+)
        ///
        /// Returns error if graph contains negative edge weights.
        ///
        /// Time: O((V + E) log V) with binary heap | Space: O(V)
        pub fn run(
            allocator: Allocator,
            graph: anytype,
            start: V,
            context: Context,
            zero_weight: W,
        ) !Result {
            var distances = std.HashMap(V, W, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            errdefer distances.deinit();

            var parents = std.HashMap(V, V, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            errdefer parents.deinit();

            // Priority queue (min-heap) for vertices
            var pq = std.PriorityQueue(QueueEntry, void, compareDistance).init(allocator, {});
            defer pq.deinit();

            // Initialize start vertex
            try distances.put(start, zero_weight);
            try pq.add(.{ .vertex = start, .distance = zero_weight });

            while (pq.removeOrNull()) |entry| {
                const u = entry.vertex;
                const dist_u = entry.distance;

                // Skip if we've already processed this vertex with a better distance
                if (distances.get(u)) |current_dist| {
                    if (std.math.order(dist_u, current_dist) == .gt) {
                        continue;
                    }
                }

                // Process neighbors
                const neighbors = graph.getNeighbors(u) orelse continue;
                for (neighbors) |edge| {
                    const v = edge.target;
                    const weight = edge.weight;

                    // Check for negative weights
                    if (std.math.order(weight, zero_weight) == .lt) {
                        return error.NegativeWeight;
                    }

                    const alt = dist_u + weight;

                    // Relaxation step
                    const current_dist_v = distances.get(v);
                    if (current_dist_v == null or std.math.order(alt, current_dist_v.?) == .lt) {
                        try distances.put(v, alt);
                        try parents.put(v, u);
                        try pq.add(.{ .vertex = v, .distance = alt });
                    }
                }
            }

            return Result{
                .distances = distances,
                .parents = parents,
                .allocator = allocator,
                .context = context,
            };
        }

        /// Run Dijkstra's algorithm but stop early when target is reached.
        ///
        /// Time: O((V + E) log V) worst case, often faster | Space: O(V)
        pub fn runToGoal(
            allocator: Allocator,
            graph: anytype,
            start: V,
            target: V,
            context: Context,
            zero_weight: W,
        ) !Result {
            var distances = std.HashMap(V, W, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            errdefer distances.deinit();

            var parents = std.HashMap(V, V, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            errdefer parents.deinit();

            var pq = std.PriorityQueue(QueueEntry, void, compareDistance).init(allocator, {});
            defer pq.deinit();

            try distances.put(start, zero_weight);
            try pq.add(.{ .vertex = start, .distance = zero_weight });

            while (pq.removeOrNull()) |entry| {
                const u = entry.vertex;
                const dist_u = entry.distance;

                // Early termination if we reached the target
                if (context.eql(u, target)) {
                    break;
                }

                if (distances.get(u)) |current_dist| {
                    if (std.math.order(dist_u, current_dist) == .gt) {
                        continue;
                    }
                }

                const neighbors = graph.getNeighbors(u) orelse continue;
                for (neighbors) |edge| {
                    const v = edge.target;
                    const weight = edge.weight;

                    if (std.math.order(weight, zero_weight) == .lt) {
                        return error.NegativeWeight;
                    }

                    const alt = dist_u + weight;
                    const current_dist_v = distances.get(v);
                    if (current_dist_v == null or std.math.order(alt, current_dist_v.?) == .lt) {
                        try distances.put(v, alt);
                        try parents.put(v, u);
                        try pq.add(.{ .vertex = v, .distance = alt });
                    }
                }
            }

            return Result{
                .distances = distances,
                .parents = parents,
                .allocator = allocator,
                .context = context,
            };
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const IntContext = struct {
    pub fn hash(_: @This(), key: u32) u64 {
        return key;
    }
    pub fn eql(_: @This(), a: u32, b: u32) bool {
        return a == b;
    }
};

const TestGraph = struct {
    edges: std.AutoHashMap(u32, std.ArrayList(Edge)),
    allocator: Allocator,

    const Edge = struct {
        target: u32,
        weight: i32,
    };

    fn init(allocator: Allocator) TestGraph {
        return .{
            .edges = std.AutoHashMap(u32, std.ArrayList(Edge)).init(allocator),
            .allocator = allocator,
        };
    }

    fn deinit(self: *TestGraph) void {
        var it = self.edges.valueIterator();
        while (it.next()) |list| {
            list.deinit(self.allocator);
        }
        self.edges.deinit();
    }

    fn addEdge(self: *TestGraph, from: u32, to: u32, weight: i32) !void {
        const entry = try self.edges.getOrPut(from);
        if (!entry.found_existing) {
            entry.value_ptr.* = .{};
        }
        try entry.value_ptr.append(self.allocator, .{ .target = to, .weight = weight });
    }

    fn getNeighbors(self: *const TestGraph, vertex: u32) ?[]const Edge {
        const list = self.edges.get(vertex) orelse return null;
        return list.items;
    }
};

test "Dijkstra: basic shortest paths" {
    var graph = TestGraph.init(testing.allocator);
    defer graph.deinit();

    // Graph:
    //   0 --5--> 1 --2--> 2
    //   |        |
    //   3        4
    //   |        |
    //   v        v
    //   3 --1--> 4
    try graph.addEdge(0, 1, 5);
    try graph.addEdge(0, 3, 3);
    try graph.addEdge(1, 2, 2);
    try graph.addEdge(1, 4, 4);
    try graph.addEdge(3, 4, 1);

    const Algo = Dijkstra(u32, i32, IntContext);
    var result = try Algo.run(testing.allocator, &graph, 0, .{}, 0);
    defer result.deinit();

    // Check distances
    try testing.expectEqual(@as(i32, 0), result.getDistance(0).?);
    try testing.expectEqual(@as(i32, 5), result.getDistance(1).?);
    try testing.expectEqual(@as(i32, 7), result.getDistance(2).?);
    try testing.expectEqual(@as(i32, 3), result.getDistance(3).?);
    try testing.expectEqual(@as(i32, 4), result.getDistance(4).?);

    // Check path to vertex 2
    const path2 = (try result.getPath(2)).?;
    defer testing.allocator.free(path2);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 2 }, path2);

    // Check path to vertex 4
    const path4 = (try result.getPath(4)).?;
    defer testing.allocator.free(path4);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 3, 4 }, path4);
}

test "Dijkstra: single vertex" {
    var graph = TestGraph.init(testing.allocator);
    defer graph.deinit();

    const Algo = Dijkstra(u32, i32, IntContext);
    var result = try Algo.run(testing.allocator, &graph, 0, .{}, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 0), result.getDistance(0).?);
    try testing.expectEqual(@as(?i32, null), result.getDistance(1));
}

test "Dijkstra: unreachable vertices" {
    var graph = TestGraph.init(testing.allocator);
    defer graph.deinit();

    try graph.addEdge(0, 1, 1);
    try graph.addEdge(2, 3, 1);

    const Algo = Dijkstra(u32, i32, IntContext);
    var result = try Algo.run(testing.allocator, &graph, 0, .{}, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 0), result.getDistance(0).?);
    try testing.expectEqual(@as(i32, 1), result.getDistance(1).?);
    try testing.expectEqual(@as(?i32, null), result.getDistance(2));
    try testing.expectEqual(@as(?i32, null), result.getDistance(3));
}

test "Dijkstra: early termination with runToGoal" {
    var graph = TestGraph.init(testing.allocator);
    defer graph.deinit();

    // 0 -> 1 -> 2 -> 3 -> 4 -> 5
    try graph.addEdge(0, 1, 1);
    try graph.addEdge(1, 2, 1);
    try graph.addEdge(2, 3, 1);
    try graph.addEdge(3, 4, 1);
    try graph.addEdge(4, 5, 1);

    const Algo = Dijkstra(u32, i32, IntContext);
    var result = try Algo.runToGoal(testing.allocator, &graph, 0, 3, .{}, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 0), result.getDistance(0).?);
    try testing.expectEqual(@as(i32, 1), result.getDistance(1).?);
    try testing.expectEqual(@as(i32, 2), result.getDistance(2).?);
    try testing.expectEqual(@as(i32, 3), result.getDistance(3).?);

    const path = (try result.getPath(3)).?;
    defer testing.allocator.free(path);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 2, 3 }, path);
}

test "Dijkstra: negative weight detection" {
    var graph = TestGraph.init(testing.allocator);
    defer graph.deinit();

    try graph.addEdge(0, 1, 5);
    try graph.addEdge(1, 2, -3); // Negative weight

    const Algo = Dijkstra(u32, i32, IntContext);
    const result = Algo.run(testing.allocator, &graph, 0, .{}, 0);

    try testing.expectError(error.NegativeWeight, result);
}

test "Dijkstra: multiple paths - chooses shortest" {
    var graph = TestGraph.init(testing.allocator);
    defer graph.deinit();

    // Two paths from 0 to 3:
    // 0 -> 1 -> 3 (cost 10)
    // 0 -> 2 -> 3 (cost 8)
    try graph.addEdge(0, 1, 5);
    try graph.addEdge(0, 2, 6);
    try graph.addEdge(1, 3, 5);
    try graph.addEdge(2, 3, 2);

    const Algo = Dijkstra(u32, i32, IntContext);
    var result = try Algo.run(testing.allocator, &graph, 0, .{}, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 8), result.getDistance(3).?);

    const path = (try result.getPath(3)).?;
    defer testing.allocator.free(path);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 2, 3 }, path);
}

test "Dijkstra: zero-weight edges" {
    var graph = TestGraph.init(testing.allocator);
    defer graph.deinit();

    try graph.addEdge(0, 1, 0);
    try graph.addEdge(1, 2, 0);
    try graph.addEdge(2, 3, 5);

    const Algo = Dijkstra(u32, i32, IntContext);
    var result = try Algo.run(testing.allocator, &graph, 0, .{}, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 0), result.getDistance(0).?);
    try testing.expectEqual(@as(i32, 0), result.getDistance(1).?);
    try testing.expectEqual(@as(i32, 0), result.getDistance(2).?);
    try testing.expectEqual(@as(i32, 5), result.getDistance(3).?);
}

test "Dijkstra: self-loop handling" {
    var graph = TestGraph.init(testing.allocator);
    defer graph.deinit();

    try graph.addEdge(0, 0, 10); // Self-loop
    try graph.addEdge(0, 1, 5);

    const Algo = Dijkstra(u32, i32, IntContext);
    var result = try Algo.run(testing.allocator, &graph, 0, .{}, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 0), result.getDistance(0).?);
    try testing.expectEqual(@as(i32, 5), result.getDistance(1).?);
}

test "Dijkstra: stress test - large linear graph" {
    var graph = TestGraph.init(testing.allocator);
    defer graph.deinit();

    const n = 1000;
    var i: u32 = 0;
    while (i < n - 1) : (i += 1) {
        try graph.addEdge(i, i + 1, 1);
    }

    const Algo = Dijkstra(u32, i32, IntContext);
    var result = try Algo.run(testing.allocator, &graph, 0, .{}, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 0), result.getDistance(0).?);
    try testing.expectEqual(@as(i32, 500), result.getDistance(500).?);
    try testing.expectEqual(@as(i32, 999), result.getDistance(999).?);

    const path = (try result.getPath(999)).?;
    defer testing.allocator.free(path);
    try testing.expectEqual(@as(usize, 1000), path.len);
}

test "Dijkstra: floating point weights" {
    const FloatGraph = struct {
        edges: std.AutoHashMap(u32, std.ArrayList(Edge)),
        allocator: Allocator,

        const Edge = struct {
            target: u32,
            weight: f64,
        };

        fn init(allocator: Allocator) @This() {
            return .{
                .edges = std.AutoHashMap(u32, std.ArrayList(Edge)).init(allocator),
                .allocator = allocator,
            };
        }

        fn deinit(self: *@This()) void {
            var it = self.edges.valueIterator();
            while (it.next()) |list| {
                list.deinit(self.allocator);
            }
            self.edges.deinit();
        }

        fn addEdge(self: *@This(), from: u32, to: u32, weight: f64) !void {
            const entry = try self.edges.getOrPut(from);
            if (!entry.found_existing) {
                entry.value_ptr.* = .{};
            }
            try entry.value_ptr.append(self.allocator, .{ .target = to, .weight = weight });
        }

        fn getNeighbors(self: *const @This(), vertex: u32) ?[]const Edge {
            const list = self.edges.get(vertex) orelse return null;
            return list.items;
        }
    };

    var graph = FloatGraph.init(testing.allocator);
    defer graph.deinit();

    try graph.addEdge(0, 1, 1.5);
    try graph.addEdge(0, 2, 2.3);
    try graph.addEdge(1, 3, 0.7);
    try graph.addEdge(2, 3, 0.1);

    const Algo = Dijkstra(u32, f64, IntContext);
    var result = try Algo.run(testing.allocator, &graph, 0, .{}, 0.0);
    defer result.deinit();

    try testing.expectEqual(@as(f64, 0.0), result.getDistance(0).?);
    try testing.expectEqual(@as(f64, 1.5), result.getDistance(1).?);
    try testing.expectEqual(@as(f64, 2.3), result.getDistance(2).?);
    try testing.expectApproxEqAbs(@as(f64, 2.2), result.getDistance(3).?, 0.01);
}
