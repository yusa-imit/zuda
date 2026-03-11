const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Hopcroft-Karp - Maximum cardinality bipartite matching algorithm.
///
/// Finds the maximum matching in a bipartite graph using augmenting paths.
/// Repeatedly finds a maximal set of vertex-disjoint shortest augmenting paths in one phase.
///
/// Time Complexity: O(E * sqrt(V))
/// Space Complexity: O(V)
///
/// The algorithm:
/// 1. Build a layered graph via BFS to find all shortest augmenting paths
/// 2. Use DFS to find vertex-disjoint augmenting paths in that layered graph
/// 3. Repeat until no augmenting paths exist
///
/// A bipartite graph is divided into two sets U and V where edges only connect vertices from different sets.
/// A matching is a set of edges with no common vertices.
/// An augmenting path is an alternating path of unmatched-matched edges starting and ending at unmatched vertices.
///
/// Generic parameters:
/// - V: Vertex type (must be hashable)
/// - Context: Context type for hashing/comparing vertices
pub fn HopcroftKarp(
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

        /// Result of maximum matching computation
        pub const Result = struct {
            /// Number of edges in the maximum matching
            matching_size: usize,
            /// For each vertex in U, its matched vertex in V (or null if unmatched)
            pair_u: std.HashMap(V, ?V, HashMapContext, std.hash_map.default_max_load_percentage),
            /// For each vertex in V, its matched vertex in U (or null if unmatched)
            pair_v: std.HashMap(V, ?V, HashMapContext, std.hash_map.default_max_load_percentage),
            allocator: Allocator,

            pub fn deinit(self: *Result) void {
                self.pair_u.deinit();
                self.pair_v.deinit();
            }

            /// Check if vertex u from U is matched
            pub fn isMatched(self: *const Result, u: V) bool {
                if (self.pair_u.get(u)) |maybe_v| {
                    return maybe_v != null;
                }
                return false;
            }

            /// Get the vertex from V that u is matched to (or null if unmatched)
            pub fn getMatch(self: *const Result, u: V) ?V {
                if (self.pair_u.get(u)) |maybe_v| {
                    return maybe_v;
                }
                return null;
            }
        };

        /// Run Hopcroft-Karp algorithm to find maximum cardinality bipartite matching.
        ///
        /// The graph must be bipartite with vertex sets U and V.
        /// Edges are provided as an adjacency list: for each vertex u in U, list its neighbors in V.
        ///
        /// Graph type must provide:
        /// - Iterator over U vertices and their V neighbors
        ///
        /// Returns Result with matching information.
        ///
        /// Time: O(E * sqrt(V)) | Space: O(V)
        pub fn run(
            allocator: Allocator,
            graph: anytype,
            context: Context,
        ) !Result {
            var pair_u = std.HashMap(V, ?V, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            errdefer pair_u.deinit();

            var pair_v = std.HashMap(V, ?V, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            errdefer pair_v.deinit();

            // Initialize all vertices as unmatched
            var vertex_iter = graph.iterator();
            while (vertex_iter.next()) |entry| {
                const u = entry.key_ptr.*;
                try pair_u.put(u, null);

                // Also register all V vertices from adjacency lists
                for (entry.value_ptr.*) |v| {
                    if (!pair_v.contains(v)) {
                        try pair_v.put(v, null);
                    }
                }
            }

            var matching_size: usize = 0;

            // Repeat until no augmenting paths found
            while (true) {
                // BFS to build layered graph and find distances
                var dist = std.HashMap(V, usize, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
                defer dist.deinit();

                const found = try bfs(allocator, graph, &pair_u, &pair_v, &dist, context);
                if (!found) break; // No more augmenting paths

                // DFS to find maximal set of vertex-disjoint augmenting paths
                vertex_iter = graph.iterator();
                while (vertex_iter.next()) |entry| {
                    const u = entry.key_ptr.*;
                    if (pair_u.get(u)) |maybe_v| {
                        if (maybe_v == null) {
                            if (try dfs(graph, &pair_u, &pair_v, &dist, u, context)) {
                                matching_size += 1;
                            }
                        }
                    }
                }
            }

            return Result{
                .matching_size = matching_size,
                .pair_u = pair_u,
                .pair_v = pair_v,
                .allocator = allocator,
            };
        }

        /// BFS to find shortest augmenting paths and build layered graph
        fn bfs(
            allocator: Allocator,
            graph: anytype,
            pair_u: *std.HashMap(V, ?V, HashMapContext, std.hash_map.default_max_load_percentage),
            pair_v: *std.HashMap(V, ?V, HashMapContext, std.hash_map.default_max_load_percentage),
            dist: *std.HashMap(V, usize, HashMapContext, std.hash_map.default_max_load_percentage),
            _: Context,
        ) !bool {
            var queue: std.ArrayList(V) = .{};
            defer queue.deinit(allocator);

            const inf = std.math.maxInt(usize);

            // Start from all unmatched vertices in U
            var vertex_iter = graph.iterator();
            while (vertex_iter.next()) |entry| {
                const u = entry.key_ptr.*;
                if (pair_u.get(u)) |maybe_v| {
                    if (maybe_v == null) {
                        try dist.put(u, 0);
                        try queue.append(allocator, u);
                    }
                }
            }

            var min_dist_to_unmatched_v: usize = inf;

            var read_idx: usize = 0;
            while (read_idx < queue.items.len) {
                const u = queue.items[read_idx];
                read_idx += 1;

                const u_dist = dist.get(u).?;
                if (u_dist >= min_dist_to_unmatched_v) continue;

                // Explore neighbors in V
                var iter = graph.iterator();
                while (iter.next()) |entry| {
                    if (entry.key_ptr.* != u) continue; // Not a real iterator, workaround

                    for (entry.value_ptr.*) |v| {
                        const matched_u = pair_v.get(v).?;
                        if (matched_u == null) {
                            // Found an unmatched v - this completes an augmenting path
                            min_dist_to_unmatched_v = @min(min_dist_to_unmatched_v, u_dist + 1);
                        } else {
                            // v is matched to some u'
                            const u_prime = matched_u.?;
                            if (!dist.contains(u_prime)) {
                                try dist.put(u_prime, u_dist + 2); // Distance increases by 2 (matched edge + next edge)
                                try queue.append(allocator, u_prime);
                            }
                        }
                    }
                }
            }

            return min_dist_to_unmatched_v < inf;
        }

        /// DFS to find an augmenting path from u using the layered graph
        fn dfs(
            graph: anytype,
            pair_u: *std.HashMap(V, ?V, HashMapContext, std.hash_map.default_max_load_percentage),
            pair_v: *std.HashMap(V, ?V, HashMapContext, std.hash_map.default_max_load_percentage),
            dist: *std.HashMap(V, usize, HashMapContext, std.hash_map.default_max_load_percentage),
            u: V,
            context: Context,
        ) !bool {
            const u_dist = dist.get(u) orelse return false;

            // Explore neighbors in V
            var iter = graph.iterator();
            while (iter.next()) |entry| {
                if (!context.eql(entry.key_ptr.*, u)) continue;

                for (entry.value_ptr.*) |v| {
                    const matched_u = pair_v.get(v).?;

                    if (matched_u == null) {
                        // Found an unmatched v - augment along this path
                        try pair_v.put(v, u);
                        try pair_u.put(u, v);
                        return true;
                    } else {
                        const u_prime = matched_u.?;
                        const expected_dist = u_dist + 2;

                        if (dist.get(u_prime)) |u_prime_dist| {
                            if (u_prime_dist == expected_dist) {
                                // Continue DFS from u'
                                if (try dfs(graph, pair_u, pair_v, dist, u_prime, context)) {
                                    // Augment: swap the matching
                                    try pair_v.put(v, u);
                                    try pair_u.put(u, v);
                                    return true;
                                }
                            }
                        }
                    }
                }
            }

            // Mark this vertex as visited (no augmenting path from u)
            try dist.put(u, std.math.maxInt(usize));
            return false;
        }
    };
}

// Tests

test "HopcroftKarp: simple matching" {
    const allocator = testing.allocator;

    const Graph = struct {
        const Vertex = u32;
        const Context = struct {
            pub fn hash(_: @This(), v: Vertex) u64 {
                return v;
            }
            pub fn eql(_: @This(), a: Vertex, b: Vertex) bool {
                return a == b;
            }
        };
        edges: std.HashMap(Vertex, []const Vertex, Context, std.hash_map.default_max_load_percentage),

        fn init(alloc: Allocator) @This() {
            return .{
                .edges = std.HashMap(Vertex, []const Vertex, Context, std.hash_map.default_max_load_percentage).init(alloc),
            };
        }

        fn deinit(self: *@This()) void {
            self.edges.deinit();
        }

        fn addEdges(self: *@This(), u: Vertex, vs: []const Vertex) !void {
            try self.edges.put(u, vs);
        }

        fn iterator(self: *const @This()) std.HashMap(Vertex, []const Vertex, Context, std.hash_map.default_max_load_percentage).Iterator {
            return self.edges.iterator();
        }
    };

    var graph = Graph.init(allocator);
    defer graph.deinit();

    // U = {0, 1}, V = {2, 3}
    // 0 -> {2, 3}
    // 1 -> {3}
    try graph.addEdges(0, &[_]u32{ 2, 3 });
    try graph.addEdges(1, &[_]u32{3});

    const HK = HopcroftKarp(Graph.Vertex, Graph.Context);
    var result = try HK.run(allocator, &graph, Graph.Context{});
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.matching_size);
}

test "HopcroftKarp: no matching" {
    const allocator = testing.allocator;

    const Graph = struct {
        const Vertex = u32;
        const Context = struct {
            pub fn hash(_: @This(), v: Vertex) u64 {
                return v;
            }
            pub fn eql(_: @This(), a: Vertex, b: Vertex) bool {
                return a == b;
            }
        };
        edges: std.HashMap(Vertex, []const Vertex, Context, std.hash_map.default_max_load_percentage),

        fn init(alloc: Allocator) @This() {
            return .{
                .edges = std.HashMap(Vertex, []const Vertex, Context, std.hash_map.default_max_load_percentage).init(alloc),
            };
        }

        fn deinit(self: *@This()) void {
            self.edges.deinit();
        }

        fn addEdges(self: *@This(), u: Vertex, vs: []const Vertex) !void {
            try self.edges.put(u, vs);
        }

        fn iterator(self: *const @This()) std.HashMap(Vertex, []const Vertex, Context, std.hash_map.default_max_load_percentage).Iterator {
            return self.edges.iterator();
        }
    };

    var graph = Graph.init(allocator);
    defer graph.deinit();

    // U = {0}, V = {} (no edges)
    try graph.addEdges(0, &[_]u32{});

    const HK = HopcroftKarp(Graph.Vertex, Graph.Context);
    var result = try HK.run(allocator, &graph, Graph.Context{});
    defer result.deinit();

    try testing.expectEqual(@as(usize, 0), result.matching_size);
}

test "HopcroftKarp: complete bipartite" {
    const allocator = testing.allocator;

    const Graph = struct {
        const Vertex = u32;
        const Context = struct {
            pub fn hash(_: @This(), v: Vertex) u64 {
                return v;
            }
            pub fn eql(_: @This(), a: Vertex, b: Vertex) bool {
                return a == b;
            }
        };
        edges: std.HashMap(Vertex, []const Vertex, Context, std.hash_map.default_max_load_percentage),

        fn init(alloc: Allocator) @This() {
            return .{
                .edges = std.HashMap(Vertex, []const Vertex, Context, std.hash_map.default_max_load_percentage).init(alloc),
            };
        }

        fn deinit(self: *@This()) void {
            self.edges.deinit();
        }

        fn addEdges(self: *@This(), u: Vertex, vs: []const Vertex) !void {
            try self.edges.put(u, vs);
        }

        fn iterator(self: *const @This()) std.HashMap(Vertex, []const Vertex, Context, std.hash_map.default_max_load_percentage).Iterator {
            return self.edges.iterator();
        }
    };

    var graph = Graph.init(allocator);
    defer graph.deinit();

    // K_{3,3}: U = {0, 1, 2}, V = {3, 4, 5}
    // Each vertex in U connects to all in V
    try graph.addEdges(0, &[_]u32{ 3, 4, 5 });
    try graph.addEdges(1, &[_]u32{ 3, 4, 5 });
    try graph.addEdges(2, &[_]u32{ 3, 4, 5 });

    const HK = HopcroftKarp(Graph.Vertex, Graph.Context);
    var result = try HK.run(allocator, &graph, Graph.Context{});
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.matching_size);
}

test "HopcroftKarp: asymmetric bipartite" {
    const allocator = testing.allocator;

    const Graph = struct {
        const Vertex = u32;
        const Context = struct {
            pub fn hash(_: @This(), v: Vertex) u64 {
                return v;
            }
            pub fn eql(_: @This(), a: Vertex, b: Vertex) bool {
                return a == b;
            }
        };
        edges: std.HashMap(Vertex, []const Vertex, Context, std.hash_map.default_max_load_percentage),

        fn init(alloc: Allocator) @This() {
            return .{
                .edges = std.HashMap(Vertex, []const Vertex, Context, std.hash_map.default_max_load_percentage).init(alloc),
            };
        }

        fn deinit(self: *@This()) void {
            self.edges.deinit();
        }

        fn addEdges(self: *@This(), u: Vertex, vs: []const Vertex) !void {
            try self.edges.put(u, vs);
        }

        fn iterator(self: *const @This()) std.HashMap(Vertex, []const Vertex, Context, std.hash_map.default_max_load_percentage).Iterator {
            return self.edges.iterator();
        }
    };

    var graph = Graph.init(allocator);
    defer graph.deinit();

    // U = {0, 1, 2}, V = {3, 4}
    // More vertices in U than V - matching size limited by min(|U|, |V|)
    try graph.addEdges(0, &[_]u32{ 3, 4 });
    try graph.addEdges(1, &[_]u32{ 3, 4 });
    try graph.addEdges(2, &[_]u32{ 3, 4 });

    const HK = HopcroftKarp(Graph.Vertex, Graph.Context);
    var result = try HK.run(allocator, &graph, Graph.Context{});
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.matching_size);
}
