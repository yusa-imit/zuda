const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// BellmanFord - Single-source shortest paths with negative edge weights.
///
/// Computes shortest paths from a start vertex to all reachable vertices
/// using the Bellman-Ford algorithm. Unlike Dijkstra, this algorithm can
/// handle negative edge weights and detects negative cycles.
///
/// Time Complexity: O(V * E)
/// Space Complexity: O(V) for distance and parent maps
///
/// Features:
/// - Handles negative edge weights correctly
/// - Detects negative cycles reachable from the start vertex
/// - Reports vertices involved in negative cycles
///
/// Generic parameters:
/// - V: Vertex type (must be hashable)
/// - W: Weight type (must support comparison, addition, and subtraction)
/// - Context: Context type for hashing/comparing vertices (must have .hash and .eql methods)
pub fn BellmanFord(
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

        /// Edge representation for Bellman-Ford
        pub const Edge = struct {
            from: V,
            to: V,
            weight: W,
        };

        /// BellmanFord result containing shortest path information
        pub const Result = struct {
            /// Distance from start vertex to each vertex
            distances: std.HashMap(V, W, HashMapContext, std.hash_map.default_max_load_percentage),
            /// Parent pointers for path reconstruction
            parents: std.HashMap(V, V, HashMapContext, std.hash_map.default_max_load_percentage),
            /// Vertices involved in negative cycles (if any)
            negative_cycle_vertices: ?[]V,
            allocator: Allocator,
            context: Context,

            pub fn deinit(self: *Result) void {
                self.distances.deinit();
                self.parents.deinit();
                if (self.negative_cycle_vertices) |vertices| {
                    self.allocator.free(vertices);
                }
            }

            /// Returns true if a negative cycle was detected.
            pub fn hasNegativeCycle(self: *const Result) bool {
                return self.negative_cycle_vertices != null;
            }

            /// Get the distance to a vertex from the start vertex.
            /// Returns null if the vertex is not reachable or affected by a negative cycle.
            pub fn getDistance(self: *const Result, vertex: V) ?W {
                return self.distances.get(vertex);
            }

            /// Get the parent of a vertex in the shortest path tree.
            /// Returns null if the vertex has no parent (is the start vertex or unreachable).
            pub fn getParent(self: *const Result, vertex: V) ?V {
                return self.parents.get(vertex);
            }

            /// Reconstruct the shortest path from start to target.
            /// Returns null if target is not reachable from start or affected by a negative cycle.
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

        /// Run Bellman-Ford algorithm from a start vertex.
        ///
        /// Time: O(V * E)
        /// Space: O(V)
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - edges: All edges in the graph
        /// - vertices: All vertices in the graph (needed to initialize distances)
        /// - start: Starting vertex
        /// - context: Context for hashing and comparing vertices
        /// - max_weight: Maximum weight value (used as infinity)
        ///
        /// Returns: Result containing distances, parents, and negative cycle info
        pub fn run(
            allocator: Allocator,
            edges: []const Edge,
            vertices: []const V,
            start: V,
            context: Context,
            max_weight: W,
        ) !Result {
            const hash_ctx = HashMapContext{ .user_ctx = context };

            var distances = std.HashMap(V, W, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            errdefer distances.deinit();

            var parents = std.HashMap(V, V, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            errdefer parents.deinit();

            // Initialize distances
            distances.ctx = hash_ctx;
            parents.ctx = hash_ctx;

            for (vertices) |v| {
                try distances.put(v, max_weight);
            }
            try distances.put(start, @as(W, 0));

            // Relax edges |V| - 1 times
            for (0..vertices.len - 1) |_| {
                var relaxed = false;
                for (edges) |edge| {
                    const from_dist = distances.get(edge.from) orelse max_weight;
                    if (from_dist == max_weight) continue;

                    const to_dist = distances.get(edge.to) orelse max_weight;
                    const new_dist = from_dist + edge.weight;

                    if (new_dist < to_dist) {
                        try distances.put(edge.to, new_dist);
                        try parents.put(edge.to, edge.from);
                        relaxed = true;
                    }
                }
                // Early termination if no relaxation occurred
                if (!relaxed) break;
            }

            // Check for negative cycles
            var negative_cycle_vertices: ?[]V = null;
            var affected_vertices: std.ArrayList(V) = .{};
            defer affected_vertices.deinit(allocator);

            for (edges) |edge| {
                const from_dist = distances.get(edge.from) orelse max_weight;
                if (from_dist == max_weight) continue;

                const to_dist = distances.get(edge.to) orelse max_weight;
                const new_dist = from_dist + edge.weight;

                if (new_dist < to_dist) {
                    // Negative cycle detected
                    try affected_vertices.append(allocator, edge.to);
                }
            }

            if (affected_vertices.items.len > 0) {
                negative_cycle_vertices = try affected_vertices.toOwnedSlice(allocator);
            }

            return Result{
                .distances = distances,
                .parents = parents,
                .negative_cycle_vertices = negative_cycle_vertices,
                .allocator = allocator,
                .context = context,
            };
        }
    };
}

// --- Tests ---

test "BellmanFord: simple graph with positive weights" {
    const IntContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    const BF = BellmanFord(u32, i32, IntContext);
    const allocator = testing.allocator;
    const ctx = IntContext{};

    const edges = [_]BF.Edge{
        .{ .from = 0, .to = 1, .weight = 4 },
        .{ .from = 0, .to = 2, .weight = 1 },
        .{ .from = 2, .to = 1, .weight = 2 },
        .{ .from = 1, .to = 3, .weight = 1 },
        .{ .from = 2, .to = 3, .weight = 5 },
    };
    const vertices = [_]u32{ 0, 1, 2, 3 };

    var result = try BF.run(allocator, &edges, &vertices, 0, ctx, std.math.maxInt(i32));
    defer result.deinit();

    try testing.expect(!result.hasNegativeCycle());
    try testing.expectEqual(@as(i32, 0), result.getDistance(0).?);
    try testing.expectEqual(@as(i32, 3), result.getDistance(1).?);
    try testing.expectEqual(@as(i32, 1), result.getDistance(2).?);
    try testing.expectEqual(@as(i32, 4), result.getDistance(3).?);
}

test "BellmanFord: graph with negative weights" {
    const IntContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    const BF = BellmanFord(u32, i32, IntContext);
    const allocator = testing.allocator;
    const ctx = IntContext{};

    const edges = [_]BF.Edge{
        .{ .from = 0, .to = 1, .weight = 5 },
        .{ .from = 0, .to = 2, .weight = 3 },
        .{ .from = 1, .to = 2, .weight = -2 },
        .{ .from = 2, .to = 3, .weight = 2 },
        .{ .from = 1, .to = 3, .weight = 4 },
    };
    const vertices = [_]u32{ 0, 1, 2, 3 };

    var result = try BF.run(allocator, &edges, &vertices, 0, ctx, std.math.maxInt(i32));
    defer result.deinit();

    try testing.expect(!result.hasNegativeCycle());
    try testing.expectEqual(@as(i32, 0), result.getDistance(0).?);
    try testing.expectEqual(@as(i32, 5), result.getDistance(1).?);
    try testing.expectEqual(@as(i32, 3), result.getDistance(2).?);
    try testing.expectEqual(@as(i32, 5), result.getDistance(3).?);
}

test "BellmanFord: negative cycle detection" {
    const IntContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    const BF = BellmanFord(u32, i32, IntContext);
    const allocator = testing.allocator;
    const ctx = IntContext{};

    // Graph with negative cycle: 0 -> 1 -> 2 -> 1 (cycle weight = -1)
    const edges = [_]BF.Edge{
        .{ .from = 0, .to = 1, .weight = 1 },
        .{ .from = 1, .to = 2, .weight = 2 },
        .{ .from = 2, .to = 1, .weight = -4 }, // Creates negative cycle
    };
    const vertices = [_]u32{ 0, 1, 2 };

    var result = try BF.run(allocator, &edges, &vertices, 0, ctx, std.math.maxInt(i32));
    defer result.deinit();

    try testing.expect(result.hasNegativeCycle());
    try testing.expect(result.negative_cycle_vertices.?.len > 0);
}

test "BellmanFord: disconnected graph" {
    const IntContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    const BF = BellmanFord(u32, i32, IntContext);
    const allocator = testing.allocator;
    const ctx = IntContext{};

    const edges = [_]BF.Edge{
        .{ .from = 0, .to = 1, .weight = 1 },
        .{ .from = 2, .to = 3, .weight = 1 }, // Disconnected component
    };
    const vertices = [_]u32{ 0, 1, 2, 3 };

    var result = try BF.run(allocator, &edges, &vertices, 0, ctx, std.math.maxInt(i32));
    defer result.deinit();

    try testing.expect(!result.hasNegativeCycle());
    try testing.expectEqual(@as(i32, 0), result.getDistance(0).?);
    try testing.expectEqual(@as(i32, 1), result.getDistance(1).?);
    // Vertices 2 and 3 are unreachable from 0
    try testing.expectEqual(@as(i32, std.math.maxInt(i32)), result.getDistance(2).?);
    try testing.expectEqual(@as(i32, std.math.maxInt(i32)), result.getDistance(3).?);
}

test "BellmanFord: single vertex" {
    const IntContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    const BF = BellmanFord(u32, i32, IntContext);
    const allocator = testing.allocator;
    const ctx = IntContext{};

    const edges = [_]BF.Edge{};
    const vertices = [_]u32{0};

    var result = try BF.run(allocator, &edges, &vertices, 0, ctx, std.math.maxInt(i32));
    defer result.deinit();

    try testing.expect(!result.hasNegativeCycle());
    try testing.expectEqual(@as(i32, 0), result.getDistance(0).?);
}

test "BellmanFord: path reconstruction" {
    const IntContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    const BF = BellmanFord(u32, i32, IntContext);
    const allocator = testing.allocator;
    const ctx = IntContext{};

    const edges = [_]BF.Edge{
        .{ .from = 0, .to = 1, .weight = 4 },
        .{ .from = 0, .to = 2, .weight = 1 },
        .{ .from = 2, .to = 1, .weight = 2 },
        .{ .from = 1, .to = 3, .weight = 1 },
        .{ .from = 2, .to = 3, .weight = 5 },
    };
    const vertices = [_]u32{ 0, 1, 2, 3 };

    var result = try BF.run(allocator, &edges, &vertices, 0, ctx, std.math.maxInt(i32));
    defer result.deinit();

    // Path to vertex 3 should be: 0 -> 2 -> 1 -> 3
    const path = (try result.getPath(3)).?;
    defer allocator.free(path);

    try testing.expectEqual(@as(usize, 4), path.len);
    try testing.expectEqual(@as(u32, 0), path[0]);
    try testing.expectEqual(@as(u32, 2), path[1]);
    try testing.expectEqual(@as(u32, 1), path[2]);
    try testing.expectEqual(@as(u32, 3), path[3]);
}

test "BellmanFord: self-loop with negative weight" {
    const IntContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    const BF = BellmanFord(u32, i32, IntContext);
    const allocator = testing.allocator;
    const ctx = IntContext{};

    // Self-loop with negative weight creates a negative cycle
    const edges = [_]BF.Edge{
        .{ .from = 0, .to = 1, .weight = 1 },
        .{ .from = 1, .to = 1, .weight = -1 }, // Negative self-loop
    };
    const vertices = [_]u32{ 0, 1 };

    var result = try BF.run(allocator, &edges, &vertices, 0, ctx, std.math.maxInt(i32));
    defer result.deinit();

    try testing.expect(result.hasNegativeCycle());
    try testing.expect(result.negative_cycle_vertices.?.len > 0);
}
