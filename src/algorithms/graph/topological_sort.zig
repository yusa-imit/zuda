const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Topological Sort - Orders vertices in a directed acyclic graph (DAG) such that
/// for every directed edge (u, v), vertex u comes before v in the ordering.
///
/// Uses Kahn's algorithm (BFS-based approach with in-degree tracking).
///
/// Time Complexity: O(V + E) where V is vertices, E is edges
/// Space Complexity: O(V) for in-degree map and queue
///
/// Generic parameters:
/// - V: Vertex type (must be hashable)
/// - Context: Context type for hashing/comparing vertices (must have .hash and .eql methods)
///
/// Returns null if the graph contains a cycle (not a DAG).
pub fn TopologicalSort(
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

        /// Result of topological sort
        pub const Result = struct {
            /// Topologically sorted vertices (empty if cycle detected)
            order: std.ArrayList(V),
            /// True if sort succeeded (graph is a DAG), false if cycle detected
            success: bool,
            /// If cycle detected, this contains vertices involved in a cycle
            cycle_vertices: ?std.ArrayList(V),
            allocator: Allocator,

            pub fn deinit(self: *Result) void {
                self.order.deinit(self.allocator);
                if (self.cycle_vertices) |*cv| {
                    cv.deinit(self.allocator);
                }
            }
        };

        /// Perform topological sort using Kahn's algorithm.
        ///
        /// Time: O(V + E) | Space: O(V)
        ///
        /// The graph parameter should be an AdjacencyList.
        pub fn sort(
            allocator: Allocator,
            graph: anytype,
            context: Context,
        ) !Result {
            const hash_ctx = HashMapContext{ .user_ctx = context };

            // Calculate in-degrees for all vertices
            var in_degree = std.HashMap(V, usize, HashMapContext, std.hash_map.default_max_load_percentage).initContext(allocator, hash_ctx);
            defer in_degree.deinit();

            // Initialize in-degree map
            var vertices_it = graph.adjacencies.iterator();
            while (vertices_it.next()) |entry| {
                try in_degree.put(entry.key_ptr.*, 0);
            }

            // Count incoming edges for each vertex
            vertices_it = graph.adjacencies.iterator();
            while (vertices_it.next()) |entry| {
                const vertex = entry.key_ptr.*;
                const neighbors = graph.getNeighbors(vertex);
                if (neighbors) |edges| {
                    for (edges) |edge| {
                        const neighbor_deg = in_degree.getPtr(edge.target) orelse continue;
                        neighbor_deg.* += 1;
                    }
                }
            }

            // Queue for vertices with in-degree 0
            var queue: std.ArrayList(V) = .{};
            defer queue.deinit(allocator);

            var in_degree_it = in_degree.iterator();
            while (in_degree_it.next()) |entry| {
                if (entry.value_ptr.* == 0) {
                    try queue.append(allocator, entry.key_ptr.*);
                }
            }

            // Process vertices in topological order
            var result: std.ArrayList(V) = .{};
            errdefer result.deinit(allocator);

            while (queue.items.len > 0) {
                const current = queue.orderedRemove(0);
                try result.append(allocator, current);

                // Reduce in-degree of neighbors
                const neighbors = graph.getNeighbors(current);
                if (neighbors) |edges| {
                    for (edges) |edge| {
                        const entry = in_degree.getPtr(edge.target) orelse continue;
                        entry.* -= 1;
                        if (entry.* == 0) {
                            try queue.append(allocator, edge.target);
                        }
                    }
                }
            }

            // Check if all vertices were processed (if not, there's a cycle)
            const total_vertices = graph.vertexCount();
            if (result.items.len < total_vertices) {
                // Cycle detected - collect remaining vertices
                var cycle_vertices: std.ArrayList(V) = .{};
                errdefer cycle_vertices.deinit(allocator);

                in_degree_it = in_degree.iterator();
                while (in_degree_it.next()) |entry| {
                    if (entry.value_ptr.* > 0) {
                        try cycle_vertices.append(allocator, entry.key_ptr.*);
                    }
                }

                return Result{
                    .order = std.ArrayList(V){},
                    .success = false,
                    .cycle_vertices = cycle_vertices,
                    .allocator = allocator,
                };
            }

            return Result{
                .order = result,
                .success = true,
                .cycle_vertices = null,
                .allocator = allocator,
            };
        }

        /// Perform topological sort using DFS-based approach.
        ///
        /// Time: O(V + E) | Space: O(V)
        ///
        /// This is an alternative implementation that uses depth-first search
        /// and produces a reverse post-order traversal.
        pub fn sortDFS(
            allocator: Allocator,
            graph: anytype,
            context: Context,
        ) !Result {
            const hash_ctx = HashMapContext{ .user_ctx = context };

            var visited = std.HashMap(V, void, HashMapContext, std.hash_map.default_max_load_percentage).initContext(allocator, hash_ctx);
            defer visited.deinit();

            var rec_stack = std.HashMap(V, void, HashMapContext, std.hash_map.default_max_load_percentage).initContext(allocator, hash_ctx);
            defer rec_stack.deinit();

            var result: std.ArrayList(V) = .{};
            errdefer result.deinit(allocator);

            // Visit all vertices
            var vertices_it = graph.adjacencies.iterator();
            while (vertices_it.next()) |entry| {
                const vertex = entry.key_ptr.*;
                if (!visited.contains(vertex)) {
                    const has_cycle = try dfsVisit(
                        allocator,
                        graph,
                        vertex,
                        &visited,
                        &rec_stack,
                        &result,
                        hash_ctx,
                    );
                    if (has_cycle) {
                        // Cycle detected
                        return Result{
                            .order = std.ArrayList(V){},
                            .success = false,
                            .cycle_vertices = null,
                            .allocator = allocator,
                        };
                    }
                }
            }

            // Reverse the result (DFS gives reverse topological order)
            std.mem.reverse(V, result.items);

            return Result{
                .order = result,
                .success = true,
                .cycle_vertices = null,
                .allocator = allocator,
            };
        }

        fn dfsVisit(
            allocator: Allocator,
            graph: anytype,
            vertex: V,
            visited: *std.HashMap(V, void, HashMapContext, std.hash_map.default_max_load_percentage),
            rec_stack: *std.HashMap(V, void, HashMapContext, std.hash_map.default_max_load_percentage),
            result: *std.ArrayList(V),
            hash_ctx: HashMapContext,
        ) !bool {
            try visited.put(vertex, {});
            try rec_stack.put(vertex, {});

            const neighbors = graph.getNeighbors(vertex);
            if (neighbors) |edges| {
                for (edges) |edge| {
                    const neighbor = edge.target;
                    if (rec_stack.contains(neighbor)) {
                        // Cycle detected (back edge)
                        return true;
                    }
                    if (!visited.contains(neighbor)) {
                        const has_cycle = try dfsVisit(
                            allocator,
                            graph,
                            neighbor,
                            visited,
                            rec_stack,
                            result,
                            hash_ctx,
                        );
                        if (has_cycle) return true;
                    }
                }
            }

            _ = rec_stack.remove(vertex);
            try result.append(allocator, vertex);
            return false;
        }
    };
}

// --------------------------------- Tests ---------------------------------

const AdjacencyList = @import("../../containers/graphs/adjacency_list.zig").AdjacencyList;

fn IntGraph(comptime W: type) type {
    const Context = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return @as(u64, key);
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };
    return AdjacencyList(u32, W, Context, Context.hash, Context.eql);
}

const IntContext = struct {
    pub fn hash(_: @This(), key: u32) u64 {
        return @as(u64, key);
    }

    pub fn eql(_: @This(), a: u32, b: u32) bool {
        return a == b;
    }
};

const IntTopoSort = TopologicalSort(u32, IntContext);

test "TopologicalSort: simple DAG (Kahn)" {
    const allocator = testing.allocator;

    var graph = IntGraph(void).init(allocator, .{}, true); // directed
    defer graph.deinit();

    // DAG: 0 -> 1 -> 3
    //      |         ^
    //      v         |
    //      2 --------+
    try graph.addVertex(0);
    try graph.addVertex(1);
    try graph.addVertex(2);
    try graph.addVertex(3);
    try graph.addEdge(0, 1, {});
    try graph.addEdge(0, 2, {});
    try graph.addEdge(1, 3, {});
    try graph.addEdge(2, 3, {});

    var result = try IntTopoSort.sort(allocator, &graph, .{});
    defer result.deinit();

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 4), result.order.items.len);

    // Valid topological orders: [0, 1, 2, 3] or [0, 2, 1, 3]
    try testing.expectEqual(@as(u32, 0), result.order.items[0]);
    try testing.expectEqual(@as(u32, 3), result.order.items[3]);

    // Verify 1 comes before 3 and 2 comes before 3
    var idx1: usize = 0;
    var idx2: usize = 0;
    var idx3: usize = 0;
    for (result.order.items, 0..) |v, i| {
        if (v == 1) idx1 = i;
        if (v == 2) idx2 = i;
        if (v == 3) idx3 = i;
    }
    try testing.expect(idx1 < idx3);
    try testing.expect(idx2 < idx3);
}

test "TopologicalSort: cycle detection (Kahn)" {
    const allocator = testing.allocator;
    var graph = IntGraph(void).init(allocator, .{}, true); // directed
    defer graph.deinit();

    // Cycle: 0 -> 1 -> 2 -> 0
    try graph.addVertex(0);
    try graph.addVertex(1);
    try graph.addVertex(2);
    try graph.addEdge(0, 1, {});
    try graph.addEdge(1, 2, {});
    try graph.addEdge(2, 0, {}); // creates cycle

    var result = try IntTopoSort.sort(allocator, &graph, .{});
    defer result.deinit();

    try testing.expect(!result.success);
    try testing.expectEqual(@as(usize, 0), result.order.items.len);
    try testing.expect(result.cycle_vertices != null);
    try testing.expectEqual(@as(usize, 3), result.cycle_vertices.?.items.len);
}

test "TopologicalSort: simple DAG (DFS)" {
    const allocator = testing.allocator;
    var graph = IntGraph(void).init(allocator, .{}, true); // directed
    defer graph.deinit();

    // DAG: 0 -> 1 -> 3
    //      |         ^
    //      v         |
    //      2 --------+
    try graph.addVertex(0);
    try graph.addVertex(1);
    try graph.addVertex(2);
    try graph.addVertex(3);
    try graph.addEdge(0, 1, {});
    try graph.addEdge(0, 2, {});
    try graph.addEdge(1, 3, {});
    try graph.addEdge(2, 3, {});

    var result = try IntTopoSort.sortDFS(allocator, &graph, .{});
    defer result.deinit();

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 4), result.order.items.len);

    // Verify 0 comes first and 3 comes last
    try testing.expectEqual(@as(u32, 0), result.order.items[0]);
    try testing.expectEqual(@as(u32, 3), result.order.items[3]);

    // Verify 1 comes before 3 and 2 comes before 3
    var idx1: usize = 0;
    var idx2: usize = 0;
    var idx3: usize = 0;
    for (result.order.items, 0..) |v, i| {
        if (v == 1) idx1 = i;
        if (v == 2) idx2 = i;
        if (v == 3) idx3 = i;
    }
    try testing.expect(idx1 < idx3);
    try testing.expect(idx2 < idx3);
}

test "TopologicalSort: cycle detection (DFS)" {
    const allocator = testing.allocator;
    var graph = IntGraph(void).init(allocator, .{}, true); // directed
    defer graph.deinit();

    // Cycle: 0 -> 1 -> 2 -> 0
    try graph.addVertex(0);
    try graph.addVertex(1);
    try graph.addVertex(2);
    try graph.addEdge(0, 1, {});
    try graph.addEdge(1, 2, {});
    try graph.addEdge(2, 0, {}); // creates cycle

    var result = try IntTopoSort.sortDFS(allocator, &graph, .{});
    defer result.deinit();

    try testing.expect(!result.success);
    try testing.expectEqual(@as(usize, 0), result.order.items.len);
}

test "TopologicalSort: single vertex" {
    const allocator = testing.allocator;
    var graph = IntGraph(void).init(allocator, .{}, true);
    defer graph.deinit();

    try graph.addVertex(42);

    var result = try IntTopoSort.sort(allocator, &graph, .{});
    defer result.deinit();

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 1), result.order.items.len);
    try testing.expectEqual(@as(u32, 42), result.order.items[0]);
}

test "TopologicalSort: linear chain" {
    const allocator = testing.allocator;
    var graph = IntGraph(void).init(allocator, .{}, true);
    defer graph.deinit();

    // Linear: 0 -> 1 -> 2 -> 3 -> 4
    try graph.addVertex(0);
    try graph.addVertex(1);
    try graph.addVertex(2);
    try graph.addVertex(3);
    try graph.addVertex(4);
    try graph.addEdge(0, 1, {});
    try graph.addEdge(1, 2, {});
    try graph.addEdge(2, 3, {});
    try graph.addEdge(3, 4, {});

    var result = try IntTopoSort.sort(allocator, &graph, .{});
    defer result.deinit();

    try testing.expect(result.success);
    try testing.expectEqual(@as(usize, 5), result.order.items.len);

    // Should be in exact order: [0, 1, 2, 3, 4]
    for (result.order.items, 0..) |v, i| {
        try testing.expectEqual(@as(u32, @intCast(i)), v);
    }
}
