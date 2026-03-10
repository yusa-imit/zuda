const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Johnson - All-pairs shortest paths for sparse graphs.
///
/// Computes shortest paths between all pairs of vertices using Johnson's algorithm.
/// This algorithm is efficient for sparse graphs and can handle negative edge weights
/// (but not negative cycles).
///
/// Algorithm:
/// 1. Add a new source vertex s connected to all vertices with 0-weight edges
/// 2. Run Bellman-Ford from s to compute h(v) values
/// 3. Reweight edges: w'(u,v) = w(u,v) + h(u) - h(v) (all non-negative)
/// 4. Run Dijkstra from each vertex using reweighted graph
/// 5. Adjust distances: d(u,v) = d'(u,v) + h(v) - h(u)
///
/// Time Complexity: O(V²log V + VE) for sparse graphs (better than Floyd-Warshall's O(V³))
/// Space Complexity: O(V²) for distance matrix
///
/// Features:
/// - Handles negative edge weights correctly
/// - Detects negative cycles
/// - More efficient than Floyd-Warshall for sparse graphs
/// - Uses Bellman-Ford + Dijkstra combination
///
/// Generic parameters:
/// - V: Vertex type (must be hashable)
/// - W: Weight type (must support comparison, addition, and subtraction)
/// - Context: Context type for hashing/comparing vertices (must have .hash and .eql methods)
pub fn Johnson(
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

        /// Edge representation for Johnson's algorithm
        pub const Edge = struct {
            from: V,
            to: V,
            weight: W,
        };

        /// Johnson result containing all-pairs shortest path information
        pub const Result = struct {
            /// Distance matrix: distances.get(u).get(v) = shortest distance from u to v
            distances: std.HashMap(V, std.HashMap(V, W, HashMapContext, std.hash_map.default_max_load_percentage), HashMapContext, std.hash_map.default_max_load_percentage),
            /// Parent matrix for path reconstruction
            parents: std.HashMap(V, std.HashMap(V, ?V, HashMapContext, std.hash_map.default_max_load_percentage), HashMapContext, std.hash_map.default_max_load_percentage),
            /// True if a negative cycle was detected
            has_negative_cycle: bool,
            allocator: Allocator,
            context: Context,

            pub fn deinit(self: *Result) void {
                var dist_it = self.distances.valueIterator();
                while (dist_it.next()) |inner_map| {
                    inner_map.deinit();
                }
                self.distances.deinit();

                var parent_it = self.parents.valueIterator();
                while (parent_it.next()) |inner_map| {
                    inner_map.deinit();
                }
                self.parents.deinit();
            }

            /// Returns true if a negative cycle was detected.
            pub fn hasNegativeCycle(self: *const Result) bool {
                return self.has_negative_cycle;
            }

            /// Get the shortest distance from source to target.
            /// Returns null if target is not reachable from source.
            pub fn getDistance(self: *const Result, source: V, target: V) ?W {
                const inner = self.distances.get(source) orelse return null;
                return inner.get(target);
            }

            /// Reconstruct the shortest path from source to target.
            /// Returns null if target is not reachable from source.
            /// Caller owns the returned slice.
            pub fn getPath(self: *const Result, source: V, target: V) !?[]V {
                const parent_map = self.parents.get(source) orelse return null;

                // Check if target is reachable
                if (!parent_map.contains(target)) {
                    return null;
                }

                var path: std.ArrayList(V) = .{};
                errdefer path.deinit(self.allocator);

                var current = target;
                while (true) {
                    try path.append(self.allocator, current);
                    const parent = parent_map.get(current).? orelse break;
                    current = parent;
                }

                // Reverse to get path from source to target
                std.mem.reverse(V, path.items);
                return try path.toOwnedSlice(self.allocator);
            }
        };

        /// Run Johnson's algorithm on a graph.
        ///
        /// Time: O(V²log V + VE)
        /// Space: O(V²)
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - edges: All edges in the graph
        /// - vertices: All vertices in the graph
        /// - context: Context for hashing and comparing vertices
        /// - zero_weight: Zero value for weight type (e.g., 0 for integers, 0.0 for floats)
        /// - max_weight: Maximum weight value (used as infinity)
        ///
        /// Returns: Result containing all-pairs distances and parent pointers
        pub fn run(
            allocator: Allocator,
            edges: []const Edge,
            vertices: []const V,
            context: Context,
            zero_weight: W,
            max_weight: W,
        ) !Result {
            const hash_ctx = HashMapContext{ .user_ctx = context };

            // Step 1: Add a new source vertex (we'll use a separate edge list for Bellman-Ford)
            // We need to create edges from new source (represented by null internally) to all vertices
            var bf_edges: std.ArrayList(BFEdge) = .{};
            defer bf_edges.deinit(allocator);

            // Add original edges
            for (edges) |edge| {
                try bf_edges.append(allocator, .{
                    .from = .{ .original = edge.from },
                    .to = .{ .original = edge.to },
                    .weight = edge.weight,
                });
            }

            // Add edges from new source to all vertices with weight 0
            for (vertices) |v| {
                try bf_edges.append(allocator, .{
                    .from = .new_source,
                    .to = .{ .original = v },
                    .weight = zero_weight,
                });
            }

            // Create vertex list including new source
            var bf_vertices: std.ArrayList(ExtendedVertex) = .{};
            defer bf_vertices.deinit(allocator);
            try bf_vertices.append(allocator, .new_source);
            for (vertices) |v| {
                try bf_vertices.append(allocator, .{ .original = v });
            }

            // Step 2: Run Bellman-Ford from new source
            var h_values = std.HashMap(V, W, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            defer h_values.deinit();
            h_values.ctx = hash_ctx;

            var bf_result = try runBellmanFordInternal(
                allocator,
                bf_edges.items,
                bf_vertices.items,
                context,
                zero_weight,
                max_weight,
            );
            defer {
                bf_result.distances.deinit();
            }

            // Check for negative cycles
            if (bf_result.has_negative_cycle) {
                return Result{
                    .distances = std.HashMap(V, std.HashMap(V, W, HashMapContext, std.hash_map.default_max_load_percentage), HashMapContext, std.hash_map.default_max_load_percentage).init(allocator),
                    .parents = std.HashMap(V, std.HashMap(V, ?V, HashMapContext, std.hash_map.default_max_load_percentage), HashMapContext, std.hash_map.default_max_load_percentage).init(allocator),
                    .has_negative_cycle = true,
                    .allocator = allocator,
                    .context = context,
                };
            }

            // Extract h values for original vertices
            for (vertices) |v| {
                const h = bf_result.distances.get(.{ .original = v }) orelse max_weight;
                try h_values.put(v, h);
            }

            // Step 3: Build adjacency list with reweighted edges
            var reweighted_graph: std.HashMap(V, std.ArrayList(DijkstraEdge), HashMapContext, std.hash_map.default_max_load_percentage) = std.HashMap(V, std.ArrayList(DijkstraEdge), HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            defer {
                var it = reweighted_graph.valueIterator();
                while (it.next()) |list| {
                    list.deinit(allocator);
                }
                reweighted_graph.deinit();
            }
            reweighted_graph.ctx = hash_ctx;

            for (edges) |edge| {
                const h_from = h_values.get(edge.from) orelse max_weight;
                const h_to = h_values.get(edge.to) orelse max_weight;

                // Reweight: w'(u,v) = w(u,v) + h(u) - h(v)
                const reweighted = edge.weight + h_from - h_to;

                const entry = try reweighted_graph.getOrPut(edge.from);
                if (!entry.found_existing) {
                    entry.value_ptr.* = .{};
                }
                try entry.value_ptr.append(allocator, .{
                    .target = edge.to,
                    .weight = reweighted,
                });
            }

            // Step 4: Run Dijkstra from each vertex
            var distances = std.HashMap(V, std.HashMap(V, W, HashMapContext, std.hash_map.default_max_load_percentage), HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            errdefer {
                var it = distances.valueIterator();
                while (it.next()) |inner_map| {
                    inner_map.deinit();
                }
                distances.deinit();
            }
            distances.ctx = hash_ctx;

            var parents = std.HashMap(V, std.HashMap(V, ?V, HashMapContext, std.hash_map.default_max_load_percentage), HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            errdefer {
                var it = parents.valueIterator();
                while (it.next()) |inner_map| {
                    inner_map.deinit();
                }
                parents.deinit();
            }
            parents.ctx = hash_ctx;

            for (vertices) |source| {
                var dijkstra_result = try runDijkstraInternal(
                    allocator,
                    &reweighted_graph,
                    source,
                    context,
                    zero_weight,
                );
                defer {
                    dijkstra_result.distances.deinit();
                    dijkstra_result.parents.deinit();
                }

                // Create distance map for this source
                var source_distances = std.HashMap(V, W, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
                source_distances.ctx = hash_ctx;

                var source_parents = std.HashMap(V, ?V, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
                source_parents.ctx = hash_ctx;

                // Adjust distances: d(u,v) = d'(u,v) + h(v) - h(u)
                const h_source = h_values.get(source) orelse max_weight;
                for (vertices) |target| {
                    const h_target = h_values.get(target) orelse max_weight;

                    if (dijkstra_result.distances.get(target)) |dist_reweighted| {
                        const original_dist = dist_reweighted + h_target - h_source;
                        try source_distances.put(target, original_dist);

                        // Copy parent information
                        const parent = dijkstra_result.parents.get(target) orelse null;
                        try source_parents.put(target, parent);
                    }
                }

                try distances.put(source, source_distances);
                try parents.put(source, source_parents);
            }

            return Result{
                .distances = distances,
                .parents = parents,
                .has_negative_cycle = false,
                .allocator = allocator,
                .context = context,
            };
        }

        // ============================================================================
        // Internal Helpers
        // ============================================================================

        /// Extended vertex type that includes a new source vertex
        const ExtendedVertex = union(enum) {
            new_source: void,
            original: V,
        };

        const BFEdge = struct {
            from: ExtendedVertex,
            to: ExtendedVertex,
            weight: W,
        };

        const ExtendedContext = struct {
            user_ctx: Context,

            pub fn hash(ctx: @This(), key: ExtendedVertex) u64 {
                return switch (key) {
                    .new_source => 0,
                    .original => |v| ctx.user_ctx.hash(v),
                };
            }

            pub fn eql(ctx: @This(), a: ExtendedVertex, b: ExtendedVertex) bool {
                if (@as(std.meta.Tag(ExtendedVertex), a) != @as(std.meta.Tag(ExtendedVertex), b)) {
                    return false;
                }
                return switch (a) {
                    .new_source => true,
                    .original => |va| ctx.user_ctx.eql(va, b.original),
                };
            }
        };

        const BFResult = struct {
            distances: std.HashMap(ExtendedVertex, W, ExtendedHashMapContext, std.hash_map.default_max_load_percentage),
            has_negative_cycle: bool,
        };

        const ExtendedHashMapContext = struct {
            ctx: ExtendedContext,

            pub fn hash(self: @This(), key: ExtendedVertex) u64 {
                return self.ctx.hash(key);
            }

            pub fn eql(self: @This(), a: ExtendedVertex, b: ExtendedVertex) bool {
                return self.ctx.eql(a, b);
            }
        };

        /// Internal Bellman-Ford implementation for extended vertices
        fn runBellmanFordInternal(
            allocator: Allocator,
            edges: []const BFEdge,
            vertices: []const ExtendedVertex,
            context: Context,
            zero_weight: W,
            max_weight: W,
        ) !BFResult {
            const ext_ctx = ExtendedContext{ .user_ctx = context };
            const hash_ctx = ExtendedHashMapContext{ .ctx = ext_ctx };

            var distances = std.HashMap(ExtendedVertex, W, ExtendedHashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            distances.ctx = hash_ctx;

            // Initialize distances
            for (vertices) |v| {
                try distances.put(v, max_weight);
            }
            try distances.put(.new_source, zero_weight);

            // Relax edges |V| - 1 times
            for (0..vertices.len - 1) |_| {
                var relaxed = false;
                for (edges) |edge| {
                    const from_dist = distances.get(edge.from) orelse max_weight;
                    if (std.math.order(from_dist, max_weight) == .eq) continue;

                    const to_dist = distances.get(edge.to) orelse max_weight;
                    const new_dist = from_dist + edge.weight;

                    if (std.math.order(new_dist, to_dist) == .lt) {
                        try distances.put(edge.to, new_dist);
                        relaxed = true;
                    }
                }
                if (!relaxed) break;
            }

            // Check for negative cycles
            var has_negative_cycle = false;
            for (edges) |edge| {
                const from_dist = distances.get(edge.from) orelse max_weight;
                if (std.math.order(from_dist, max_weight) == .eq) continue;

                const to_dist = distances.get(edge.to) orelse max_weight;
                const new_dist = from_dist + edge.weight;

                if (std.math.order(new_dist, to_dist) == .lt) {
                    has_negative_cycle = true;
                    break;
                }
            }

            return BFResult{
                .distances = distances,
                .has_negative_cycle = has_negative_cycle,
            };
        }

        const DijkstraEdge = struct {
            target: V,
            weight: W,
        };

        const DijkstraResult = struct {
            distances: std.HashMap(V, W, HashMapContext, std.hash_map.default_max_load_percentage),
            parents: std.HashMap(V, ?V, HashMapContext, std.hash_map.default_max_load_percentage),
        };

        const QueueEntry = struct {
            vertex: V,
            distance: W,
        };

        fn compareDistance(_: void, a: QueueEntry, b: QueueEntry) std.math.Order {
            return std.math.order(a.distance, b.distance);
        }

        /// Internal Dijkstra implementation
        fn runDijkstraInternal(
            allocator: Allocator,
            graph: *const std.HashMap(V, std.ArrayList(DijkstraEdge), HashMapContext, std.hash_map.default_max_load_percentage),
            start: V,
            context: Context,
            zero_weight: W,
        ) !DijkstraResult {
            const hash_ctx = HashMapContext{ .user_ctx = context };

            var distances = std.HashMap(V, W, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            distances.ctx = hash_ctx;

            var parents = std.HashMap(V, ?V, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            parents.ctx = hash_ctx;

            var pq = std.PriorityQueue(QueueEntry, void, compareDistance).init(allocator, {});
            defer pq.deinit();

            try distances.put(start, zero_weight);
            try parents.put(start, null);
            try pq.add(.{ .vertex = start, .distance = zero_weight });

            while (pq.removeOrNull()) |entry| {
                const u = entry.vertex;
                const dist_u = entry.distance;

                if (distances.get(u)) |current_dist| {
                    if (std.math.order(dist_u, current_dist) == .gt) {
                        continue;
                    }
                }

                const neighbors = graph.get(u) orelse continue;
                for (neighbors.items) |edge| {
                    const v = edge.target;
                    const weight = edge.weight;
                    const alt = dist_u + weight;

                    const current_dist_v = distances.get(v);
                    if (current_dist_v == null or std.math.order(alt, current_dist_v.?) == .lt) {
                        try distances.put(v, alt);
                        try parents.put(v, u);
                        try pq.add(.{ .vertex = v, .distance = alt });
                    }
                }
            }

            return DijkstraResult{
                .distances = distances,
                .parents = parents,
            };
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "Johnson: simple graph with positive weights" {
    const IntContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    const J = Johnson(u32, i32, IntContext);
    const allocator = testing.allocator;
    const ctx = IntContext{};

    const edges = [_]J.Edge{
        .{ .from = 0, .to = 1, .weight = 4 },
        .{ .from = 0, .to = 2, .weight = 1 },
        .{ .from = 2, .to = 1, .weight = 2 },
        .{ .from = 1, .to = 3, .weight = 1 },
        .{ .from = 2, .to = 3, .weight = 5 },
    };
    const vertices = [_]u32{ 0, 1, 2, 3 };

    var result = try J.run(allocator, &edges, &vertices, ctx, 0, std.math.maxInt(i32));
    defer result.deinit();

    try testing.expect(!result.hasNegativeCycle());

    // Check distances from vertex 0
    try testing.expectEqual(@as(i32, 0), result.getDistance(0, 0).?);
    try testing.expectEqual(@as(i32, 3), result.getDistance(0, 1).?);
    try testing.expectEqual(@as(i32, 1), result.getDistance(0, 2).?);
    try testing.expectEqual(@as(i32, 4), result.getDistance(0, 3).?);

    // Check distances from vertex 1
    try testing.expectEqual(@as(i32, 1), result.getDistance(1, 3).?);
}

test "Johnson: graph with negative weights" {
    const IntContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    const J = Johnson(u32, i32, IntContext);
    const allocator = testing.allocator;
    const ctx = IntContext{};

    const edges = [_]J.Edge{
        .{ .from = 0, .to = 1, .weight = 5 },
        .{ .from = 0, .to = 2, .weight = 3 },
        .{ .from = 1, .to = 2, .weight = -2 },
        .{ .from = 2, .to = 3, .weight = 2 },
        .{ .from = 1, .to = 3, .weight = 4 },
    };
    const vertices = [_]u32{ 0, 1, 2, 3 };

    var result = try J.run(allocator, &edges, &vertices, ctx, 0, std.math.maxInt(i32));
    defer result.deinit();

    try testing.expect(!result.hasNegativeCycle());

    // Check distances from vertex 0
    try testing.expectEqual(@as(i32, 0), result.getDistance(0, 0).?);
    try testing.expectEqual(@as(i32, 5), result.getDistance(0, 1).?);
    try testing.expectEqual(@as(i32, 3), result.getDistance(0, 2).?);
    try testing.expectEqual(@as(i32, 5), result.getDistance(0, 3).?);
}

test "Johnson: negative cycle detection" {
    const IntContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    const J = Johnson(u32, i32, IntContext);
    const allocator = testing.allocator;
    const ctx = IntContext{};

    // Graph with negative cycle: 0 -> 1 -> 2 -> 1 (cycle weight = -1)
    const edges = [_]J.Edge{
        .{ .from = 0, .to = 1, .weight = 1 },
        .{ .from = 1, .to = 2, .weight = 2 },
        .{ .from = 2, .to = 1, .weight = -4 },
    };
    const vertices = [_]u32{ 0, 1, 2 };

    var result = try J.run(allocator, &edges, &vertices, ctx, 0, std.math.maxInt(i32));
    defer result.deinit();

    try testing.expect(result.hasNegativeCycle());
}

test "Johnson: disconnected graph" {
    const IntContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    const J = Johnson(u32, i32, IntContext);
    const allocator = testing.allocator;
    const ctx = IntContext{};

    const edges = [_]J.Edge{
        .{ .from = 0, .to = 1, .weight = 1 },
        .{ .from = 2, .to = 3, .weight = 1 },
    };
    const vertices = [_]u32{ 0, 1, 2, 3 };

    var result = try J.run(allocator, &edges, &vertices, ctx, 0, std.math.maxInt(i32));
    defer result.deinit();

    try testing.expect(!result.hasNegativeCycle());

    // Distances within connected components
    try testing.expectEqual(@as(i32, 0), result.getDistance(0, 0).?);
    try testing.expectEqual(@as(i32, 1), result.getDistance(0, 1).?);
    try testing.expectEqual(@as(i32, 0), result.getDistance(2, 2).?);
    try testing.expectEqual(@as(i32, 1), result.getDistance(2, 3).?);

    // Unreachable pairs
    try testing.expectEqual(@as(?i32, null), result.getDistance(0, 2));
    try testing.expectEqual(@as(?i32, null), result.getDistance(0, 3));
}

test "Johnson: single vertex" {
    const IntContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    const J = Johnson(u32, i32, IntContext);
    const allocator = testing.allocator;
    const ctx = IntContext{};

    const edges = [_]J.Edge{};
    const vertices = [_]u32{0};

    var result = try J.run(allocator, &edges, &vertices, ctx, 0, std.math.maxInt(i32));
    defer result.deinit();

    try testing.expect(!result.hasNegativeCycle());
    try testing.expectEqual(@as(i32, 0), result.getDistance(0, 0).?);
}

test "Johnson: path reconstruction" {
    const IntContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    const J = Johnson(u32, i32, IntContext);
    const allocator = testing.allocator;
    const ctx = IntContext{};

    const edges = [_]J.Edge{
        .{ .from = 0, .to = 1, .weight = 4 },
        .{ .from = 0, .to = 2, .weight = 1 },
        .{ .from = 2, .to = 1, .weight = 2 },
        .{ .from = 1, .to = 3, .weight = 1 },
        .{ .from = 2, .to = 3, .weight = 5 },
    };
    const vertices = [_]u32{ 0, 1, 2, 3 };

    var result = try J.run(allocator, &edges, &vertices, ctx, 0, std.math.maxInt(i32));
    defer result.deinit();

    // Path from 0 to 3 should be: 0 -> 2 -> 1 -> 3
    const path = (try result.getPath(0, 3)).?;
    defer allocator.free(path);

    try testing.expectEqual(@as(usize, 4), path.len);
    try testing.expectEqual(@as(u32, 0), path[0]);
    try testing.expectEqual(@as(u32, 2), path[1]);
    try testing.expectEqual(@as(u32, 1), path[2]);
    try testing.expectEqual(@as(u32, 3), path[3]);
}

test "Johnson: complex graph with negative weights (no cycle)" {
    const IntContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    const J = Johnson(u32, i32, IntContext);
    const allocator = testing.allocator;
    const ctx = IntContext{};

    // Same graph as "graph with negative weights" test
    const edges = [_]J.Edge{
        .{ .from = 0, .to = 1, .weight = 5 },
        .{ .from = 0, .to = 2, .weight = 3 },
        .{ .from = 1, .to = 2, .weight = -2 },
        .{ .from = 2, .to = 3, .weight = 2 },
        .{ .from = 1, .to = 3, .weight = 4 },
    };
    const vertices = [_]u32{ 0, 1, 2, 3 };

    var result = try J.run(allocator, &edges, &vertices, ctx, 0, std.math.maxInt(i32));
    defer result.deinit();

    try testing.expect(!result.hasNegativeCycle());

    // Verify all-pairs distances
    // From vertex 0
    try testing.expectEqual(@as(i32, 0), result.getDistance(0, 0).?);
    try testing.expectEqual(@as(i32, 5), result.getDistance(0, 1).?);
    try testing.expectEqual(@as(i32, 3), result.getDistance(0, 2).?);
    try testing.expectEqual(@as(i32, 5), result.getDistance(0, 3).?);

    // From vertex 1
    try testing.expectEqual(@as(i32, 0), result.getDistance(1, 1).?);
    try testing.expectEqual(@as(i32, -2), result.getDistance(1, 2).?);
    try testing.expectEqual(@as(i32, 0), result.getDistance(1, 3).?); // min(1->3=4, 1->2->3=-2+2=0)

    // From vertex 2
    try testing.expectEqual(@as(i32, 0), result.getDistance(2, 2).?);
    try testing.expectEqual(@as(i32, 2), result.getDistance(2, 3).?);
}

test "Johnson: sparse vs dense graph performance characteristics" {
    const IntContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    const J = Johnson(u32, i32, IntContext);
    const allocator = testing.allocator;
    const ctx = IntContext{};

    // Small sparse graph (chain)
    const n = 10;
    var edges: std.ArrayList(J.Edge) = .{};
    defer edges.deinit(allocator);

    var i: u32 = 0;
    while (i < n - 1) : (i += 1) {
        try edges.append(allocator, .{ .from = i, .to = i + 1, .weight = 1 });
    }

    var vertices: [n]u32 = undefined;
    for (&vertices, 0..) |*v, idx| {
        v.* = @intCast(idx);
    }

    var result = try J.run(allocator, edges.items, &vertices, ctx, 0, std.math.maxInt(i32));
    defer result.deinit();

    // Verify correct distances
    try testing.expectEqual(@as(i32, 0), result.getDistance(0, 0).?);
    try testing.expectEqual(@as(i32, 5), result.getDistance(0, 5).?);
    try testing.expectEqual(@as(i32, 9), result.getDistance(0, 9).?);
}
