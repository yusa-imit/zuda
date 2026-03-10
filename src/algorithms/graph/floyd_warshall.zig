const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Floyd-Warshall - All-pairs shortest paths algorithm.
///
/// Computes shortest paths between all pairs of vertices using dynamic programming.
/// Works with negative edge weights but detects negative cycles.
///
/// Time Complexity: O(V³) where V is the number of vertices
/// Space Complexity: O(V²) for distance and parent matrices
///
/// Advantages:
/// - Simple implementation using dynamic programming
/// - Works with negative edge weights
/// - Detects negative cycles
/// - Computes all-pairs shortest paths in one pass
///
/// Disadvantages:
/// - O(V³) time complexity makes it impractical for large graphs
/// - Requires dense representation (O(V²) space)
/// - For sparse graphs, running Dijkstra V times may be faster
///
/// Use cases:
/// - Small to medium graphs (V < ~500)
/// - Need shortest paths between all vertex pairs
/// - Graph has negative edge weights
/// - Transitive closure computation
///
/// Generic parameters:
/// - V: Vertex type (must be hashable and equality comparable)
/// - W: Weight type (must support comparison, addition, and have a max value)
/// - Context: Context type for hashing/comparing vertices
pub fn FloydWarshall(
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

        /// Edge representation for Floyd-Warshall algorithm
        pub const Edge = struct {
            source: V,
            target: V,
            weight: W,
        };

        /// Result containing all-pairs shortest path information
        pub const Result = struct {
            /// Distance matrix: distances[i][j] = shortest distance from vertex i to j
            distances: std.AutoHashMap(V, std.AutoHashMap(V, W)),
            /// Parent matrix: parents[i][j] = parent of j on shortest path from i to j
            parents: std.AutoHashMap(V, std.AutoHashMap(V, V)),
            /// Ordered list of vertices (for indexing)
            vertices: []V,
            /// Whether a negative cycle was detected
            has_negative_cycle: bool,
            allocator: Allocator,
            context: Context,

            pub fn deinit(self: *Result) void {
                // Free distance maps
                var dist_iter = self.distances.valueIterator();
                while (dist_iter.next()) |inner_map| {
                    inner_map.deinit();
                }
                self.distances.deinit();

                // Free parent maps
                var parent_iter = self.parents.valueIterator();
                while (parent_iter.next()) |inner_map| {
                    inner_map.deinit();
                }
                self.parents.deinit();

                self.allocator.free(self.vertices);
            }

            /// Get the shortest distance from source to target.
            /// Returns null if no path exists or if there's a negative cycle affecting the path.
            pub fn getDistance(self: *const Result, source: V, target: V) ?W {
                const source_map = self.distances.get(source) orelse return null;
                return source_map.get(target);
            }

            /// Get the parent of target on the shortest path from source.
            /// Returns null if target has no parent (is unreachable or is the source itself).
            pub fn getParent(self: *const Result, source: V, target: V) ?V {
                const parent_map = self.parents.get(source) orelse return null;
                return parent_map.get(target);
            }

            /// Reconstruct the shortest path from source to target.
            /// Returns null if target is not reachable from source or if there's a negative cycle.
            /// Caller owns the returned slice.
            ///
            /// Time: O(V) where V is the path length
            /// Space: O(V) for the path array
            pub fn getPath(self: *const Result, source: V, target: V) !?[]V {
                // Check if path exists
                const source_dist_map = self.distances.get(source) orelse return null;
                _ = source_dist_map.get(target) orelse return null;

                var path: std.ArrayList(V) = .{};
                errdefer path.deinit(self.allocator);

                // Reconstruct path backwards
                var current = target;
                try path.append(self.allocator, current);

                const eql_ctx = HashMapContext{ .user_ctx = self.context };
                while (!eql_ctx.eql(current, source)) {
                    const parent = self.getParent(source, current) orelse {
                        // No parent found - path is incomplete
                        path.deinit(self.allocator);
                        return null;
                    };
                    try path.append(self.allocator, parent);
                    current = parent;
                }

                // Reverse to get source -> target order
                std.mem.reverse(V, path.items);
                return try path.toOwnedSlice(self.allocator);
            }

            /// Check if there is a path from source to target.
            pub fn hasPath(self: *const Result, source: V, target: V) bool {
                return self.getDistance(source, target) != null;
            }
        };

        /// Run Floyd-Warshall algorithm on a graph.
        ///
        /// Graph is represented as an edge list. Each edge is a tuple (source, target, weight).
        ///
        /// Time: O(V³) where V is the number of vertices
        /// Space: O(V²) for distance and parent matrices
        ///
        /// Returns Result containing all-pairs shortest paths, or error if allocation fails.
        pub fn run(
            allocator: Allocator,
            edges: []const Edge,
            vertices: []const V,
            context: Context,
        ) !Result {
            const hash_ctx = HashMapContext{ .user_ctx = context };

            // Initialize result
            var result = Result{
                .distances = std.AutoHashMap(V, std.AutoHashMap(V, W)).init(allocator),
                .parents = std.AutoHashMap(V, std.AutoHashMap(V, V)).init(allocator),
                .vertices = try allocator.dupe(V, vertices),
                .has_negative_cycle = false,
                .allocator = allocator,
                .context = context,
            };
            errdefer {
                var dist_iter = result.distances.valueIterator();
                while (dist_iter.next()) |inner_map| {
                    inner_map.deinit();
                }
                result.distances.deinit();

                var parent_iter = result.parents.valueIterator();
                while (parent_iter.next()) |inner_map| {
                    inner_map.deinit();
                }
                result.parents.deinit();

                allocator.free(result.vertices);
            }

            // Initialize distance and parent matrices
            for (vertices) |u| {
                var dist_map = std.AutoHashMap(V, W).init(allocator);
                errdefer dist_map.deinit();

                var parent_map = std.AutoHashMap(V, V).init(allocator);
                errdefer parent_map.deinit();

                for (vertices) |v| {
                    if (hash_ctx.eql(u, v)) {
                        // Distance from vertex to itself is 0
                        try dist_map.put(v, @as(W, 0));
                    }
                    // Other distances remain unset (infinite)
                }

                try result.distances.put(u, dist_map);
                try result.parents.put(u, parent_map);
            }

            // Add edge weights
            for (edges) |edge| {
                var dist_map = result.distances.getPtr(edge.source) orelse continue;
                try dist_map.put(edge.target, edge.weight);

                var parent_map = result.parents.getPtr(edge.source) orelse continue;
                try parent_map.put(edge.target, edge.source);
            }

            // Floyd-Warshall main loop: try all intermediate vertices
            for (vertices) |k| {
                for (vertices) |i| {
                    const dist_i = result.distances.getPtr(i) orelse continue;
                    const parent_i = result.parents.getPtr(i) orelse continue;

                    const dist_ik = dist_i.get(k) orelse continue;

                    for (vertices) |j| {
                        const dist_k = result.distances.get(k) orelse continue;
                        const dist_kj = dist_k.get(j) orelse continue;

                        // Calculate distance via k
                        const new_dist = dist_ik + dist_kj;

                        // Update if shorter path found
                        const current_dist = dist_i.get(j);
                        if (current_dist == null or new_dist < current_dist.?) {
                            try dist_i.put(j, new_dist);
                            const parent_k = result.parents.get(k) orelse continue;
                            const parent_kj = parent_k.get(j) orelse continue;
                            try parent_i.put(j, parent_kj);
                        }
                    }
                }
            }

            // Check for negative cycles (diagonal entries < 0)
            for (vertices) |v| {
                const dist_map = result.distances.get(v) orelse continue;
                const self_dist = dist_map.get(v) orelse continue;
                if (self_dist < 0) {
                    result.has_negative_cycle = true;
                    break;
                }
            }

            return result;
        }
    };
}

// ===== Tests =====

test "FloydWarshall - basic shortest paths" {
    const allocator = testing.allocator;

    // Simple graph:
    //   A --1--> B --2--> C
    //   |        |        ^
    //   3        1        |
    //   v        v        |
    //   D --1--> E --2----+
    const Context = struct {
        pub fn hash(_: @This(), v: u8) u64 {
            return v;
        }
        pub fn eql(_: @This(), a: u8, b: u8) bool {
            return a == b;
        }
    };
    const FW = FloydWarshall(u8, i32, Context);
    const edges = [_]FW.Edge{
        .{ .source = 'A', .target = 'B', .weight = 1 },
        .{ .source = 'A', .target = 'D', .weight = 3 },
        .{ .source = 'B', .target = 'C', .weight = 2 },
        .{ .source = 'B', .target = 'E', .weight = 1 },
        .{ .source = 'D', .target = 'E', .weight = 1 },
        .{ .source = 'E', .target = 'C', .weight = 2 },
    };
    const vertices = [_]u8{ 'A', 'B', 'C', 'D', 'E' };

    var result = try FW.run(
        allocator,
        &edges,
        &vertices,
        .{},
    );
    defer result.deinit();

    // Check A -> C: A -> B -> C = 1 + 2 = 3
    try testing.expectEqual(@as(i32, 3), result.getDistance('A', 'C').?);

    // Check A -> E: A -> B -> E = 1 + 1 = 2 (shorter than A -> D -> E = 3 + 1 = 4)
    try testing.expectEqual(@as(i32, 2), result.getDistance('A', 'E').?);

    // Check D -> C: D -> E -> C = 1 + 2 = 3
    try testing.expectEqual(@as(i32, 3), result.getDistance('D', 'C').?);

    // Check self-distances
    try testing.expectEqual(@as(i32, 0), result.getDistance('A', 'A').?);
    try testing.expectEqual(@as(i32, 0), result.getDistance('C', 'C').?);

    // Check no negative cycle
    try testing.expect(!result.has_negative_cycle);
}

test "FloydWarshall - negative weights" {
    const allocator = testing.allocator;

    // Graph with negative weights (but no negative cycle):
    //   A --1--> B
    //   |        |
    //   |       -3
    //   |        v
    //   +---5--> C
    const Context = struct {
        pub fn hash(_: @This(), v: u8) u64 {
            return v;
        }
        pub fn eql(_: @This(), a: u8, b: u8) bool {
            return a == b;
        }
    };
    const FW = FloydWarshall(u8, i32, Context);
    const edges = [_]FW.Edge{
        .{ .source = 'A', .target = 'B', .weight = 1 },
        .{ .source = 'A', .target = 'C', .weight = 5 },
        .{ .source = 'B', .target = 'C', .weight = -3 },
    };
    const vertices = [_]u8{ 'A', 'B', 'C' };

    var result = try FW.run(
        allocator,
        &edges,
        &vertices,
        .{},
    );
    defer result.deinit();

    // A -> C via B is shorter: 1 + (-3) = -2 < 5
    try testing.expectEqual(@as(i32, -2), result.getDistance('A', 'C').?);

    // No negative cycle
    try testing.expect(!result.has_negative_cycle);
}

test "FloydWarshall - negative cycle detection" {
    const allocator = testing.allocator;

    // Graph with negative cycle:
    //   A --> B --> C
    //   ^           |
    //   |          -5
    //   +----2------+
    // Cycle A -> B -> C -> A has total weight: 1 + 1 + (-5) = -3
    const Context = struct {
        pub fn hash(_: @This(), v: u8) u64 {
            return v;
        }
        pub fn eql(_: @This(), a: u8, b: u8) bool {
            return a == b;
        }
    };
    const FW = FloydWarshall(u8, i32, Context);
    const edges = [_]FW.Edge{
        .{ .source = 'A', .target = 'B', .weight = 1 },
        .{ .source = 'B', .target = 'C', .weight = 1 },
        .{ .source = 'C', .target = 'A', .weight = -5 },
    };
    const vertices = [_]u8{ 'A', 'B', 'C' };

    var result = try FW.run(
        allocator,
        &edges,
        &vertices,
        .{},
    );
    defer result.deinit();

    // Should detect negative cycle
    try testing.expect(result.has_negative_cycle);
}

test "FloydWarshall - disconnected graph" {
    const allocator = testing.allocator;

    // Disconnected graph:
    //   A --> B    C --> D
    const Context = struct {
        pub fn hash(_: @This(), v: u8) u64 {
            return v;
        }
        pub fn eql(_: @This(), a: u8, b: u8) bool {
            return a == b;
        }
    };
    const FW = FloydWarshall(u8, i32, Context);
    const edges = [_]FW.Edge{
        .{ .source = 'A', .target = 'B', .weight = 1 },
        .{ .source = 'C', .target = 'D', .weight = 2 },
    };
    const vertices = [_]u8{ 'A', 'B', 'C', 'D' };

    var result = try FW.run(
        allocator,
        &edges,
        &vertices,
        .{},
    );
    defer result.deinit();

    // Check paths within components
    try testing.expectEqual(@as(i32, 1), result.getDistance('A', 'B').?);
    try testing.expectEqual(@as(i32, 2), result.getDistance('C', 'D').?);

    // Check no paths between components
    try testing.expectEqual(@as(?i32, null), result.getDistance('A', 'C'));
    try testing.expectEqual(@as(?i32, null), result.getDistance('B', 'D'));

    // Check self-distances
    try testing.expectEqual(@as(i32, 0), result.getDistance('A', 'A').?);
    try testing.expectEqual(@as(i32, 0), result.getDistance('D', 'D').?);

    try testing.expect(!result.has_negative_cycle);
}

test "FloydWarshall - path reconstruction" {
    const allocator = testing.allocator;

    // Simple path: A -> B -> C -> D
    const Context = struct {
        pub fn hash(_: @This(), v: u8) u64 {
            return v;
        }
        pub fn eql(_: @This(), a: u8, b: u8) bool {
            return a == b;
        }
    };
    const FW = FloydWarshall(u8, i32, Context);
    const edges = [_]FW.Edge{
        .{ .source = 'A', .target = 'B', .weight = 1 },
        .{ .source = 'B', .target = 'C', .weight = 2 },
        .{ .source = 'C', .target = 'D', .weight = 3 },
        .{ .source = 'A', .target = 'C', .weight = 10 }, // Longer alternative
    };
    const vertices = [_]u8{ 'A', 'B', 'C', 'D' };

    var result = try FW.run(
        allocator,
        &edges,
        &vertices,
        .{},
    );
    defer result.deinit();

    // Get path A -> D
    const path = (try result.getPath('A', 'D')).?;
    defer allocator.free(path);

    try testing.expectEqual(@as(usize, 4), path.len);
    try testing.expectEqual(@as(u8, 'A'), path[0]);
    try testing.expectEqual(@as(u8, 'B'), path[1]);
    try testing.expectEqual(@as(u8, 'C'), path[2]);
    try testing.expectEqual(@as(u8, 'D'), path[3]);

    // Verify distance
    try testing.expectEqual(@as(i32, 6), result.getDistance('A', 'D').?);
}

test "FloydWarshall - single vertex" {
    const allocator = testing.allocator;

    const Context = struct {
        pub fn hash(_: @This(), v: u8) u64 {
            return v;
        }
        pub fn eql(_: @This(), a: u8, b: u8) bool {
            return a == b;
        }
    };
    const FW = FloydWarshall(u8, i32, Context);
    const edges = [_]FW.Edge{};
    const vertices = [_]u8{'A'};

    var result = try FW.run(
        allocator,
        &edges,
        &vertices,
        .{},
    );
    defer result.deinit();

    // Distance to self should be 0
    try testing.expectEqual(@as(i32, 0), result.getDistance('A', 'A').?);
    try testing.expect(!result.has_negative_cycle);
}

test "FloydWarshall - complete graph" {
    const allocator = testing.allocator;

    // Complete graph with 4 vertices
    const Context = struct {
        pub fn hash(_: @This(), v: u8) u64 {
            return v;
        }
        pub fn eql(_: @This(), a: u8, b: u8) bool {
            return a == b;
        }
    };
    const FW = FloydWarshall(u8, i32, Context);
    const edges = [_]FW.Edge{
        .{ .source = 'A', .target = 'B', .weight = 1 },
        .{ .source = 'A', .target = 'C', .weight = 4 },
        .{ .source = 'A', .target = 'D', .weight = 7 },
        .{ .source = 'B', .target = 'C', .weight = 2 },
        .{ .source = 'B', .target = 'D', .weight = 5 },
        .{ .source = 'C', .target = 'D', .weight = 1 },
    };
    const vertices = [_]u8{ 'A', 'B', 'C', 'D' };

    var result = try FW.run(
        allocator,
        &edges,
        &vertices,
        .{},
    );
    defer result.deinit();

    // Direct paths
    try testing.expectEqual(@as(i32, 1), result.getDistance('A', 'B').?);
    try testing.expectEqual(@as(i32, 2), result.getDistance('B', 'C').?);

    // Multi-hop paths
    // A -> D: should be A -> B -> C -> D = 1 + 2 + 1 = 4 (shorter than direct 7)
    try testing.expectEqual(@as(i32, 4), result.getDistance('A', 'D').?);

    // A -> C: should be A -> B -> C = 1 + 2 = 3 (shorter than direct 4)
    try testing.expectEqual(@as(i32, 3), result.getDistance('A', 'C').?);

    try testing.expect(!result.has_negative_cycle);
}
