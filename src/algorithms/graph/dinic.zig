const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Dinic - Maximum flow algorithm using level graphs and blocking flows.
///
/// Computes the maximum flow from a source vertex to a sink vertex in a flow network.
/// Uses BFS to build level graphs and DFS to find blocking flows, achieving O(V²E) time complexity.
///
/// Time Complexity: O(V²E) where V is vertices, E is edges
/// Space Complexity: O(V + E) for residual graph and level graph
///
/// Dinic's algorithm is faster than Edmonds-Karp for many practical cases, especially on
/// networks with specific structures (e.g., unit capacity networks achieve O(E * min(V^(2/3), E^(1/2)))).
///
/// The algorithm works by:
/// 1. Building a level graph using BFS (distances from source)
/// 2. Finding blocking flows using DFS (paths that increase flow until no more augmenting paths exist at current level)
/// 3. Repeating until sink is unreachable in level graph
///
/// Generic parameters:
/// - V: Vertex type (must be hashable)
/// - C: Capacity type (must support comparison, addition, subtraction, zero value)
/// - Context: Context type for hashing/comparing vertices
pub fn Dinic(
    comptime V: type,
    comptime C: type,
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

        /// Residual edge (used internally for residual graph)
        const ResidualEdge = struct {
            target: V,
            capacity: C,
            rev_index: usize, // Index of reverse edge in adjacency list
        };

        const AdjacencyList = std.HashMap(V, std.ArrayList(ResidualEdge), HashMapContext, std.hash_map.default_max_load_percentage);
        const LevelMap = std.HashMap(V, i32, HashMapContext, std.hash_map.default_max_load_percentage);
        const IteratorMap = std.HashMap(V, usize, HashMapContext, std.hash_map.default_max_load_percentage);

        /// Result of max flow computation
        pub const Result = struct {
            /// Maximum flow value from source to sink
            max_flow: C,

            pub fn deinit(_: *Result) void {
                // No allocations in result for now
            }
        };

        /// Run Dinic's algorithm to find maximum flow.
        ///
        /// Graph type must provide:
        /// - `adjacencies: HashMap(V, ...)` where each value has `.edges.items: []Edge`
        /// - Edge must have `.target: V` and `.weight: C` (capacity)
        ///
        /// Capacity type C must support:
        /// - Comparison operators
        /// - Addition operator (+)
        /// - Subtraction operator (-)
        /// - Must be numeric
        ///
        /// Returns error if source equals sink.
        ///
        /// Time: O(V²E) | Space: O(V + E)
        pub fn run(
            allocator: Allocator,
            graph: anytype,
            source: V,
            sink: V,
            context: Context,
            zero_capacity: C,
        ) !Result {
            if (context.eql(source, sink)) {
                return error.SourceEqualsSink;
            }

            // Build residual graph
            var residual = AdjacencyList.init(allocator);
            defer {
                var it = residual.iterator();
                while (it.next()) |entry| {
                    entry.value_ptr.deinit(allocator);
                }
                residual.deinit();
            }

            // Initialize residual graph from input graph
            try initResidualGraph(allocator, graph, &residual, zero_capacity);

            // Level map for BFS
            var level = LevelMap.init(allocator);
            defer level.deinit();

            // Iterator map for DFS (current edge index per vertex)
            var iter = IteratorMap.init(allocator);
            defer iter.deinit();

            var max_flow = zero_capacity;

            // While sink is reachable in level graph
            while (try buildLevelGraph(allocator, &residual, source, sink, &level, zero_capacity, context)) {
                iter.clearRetainingCapacity();

                // Find blocking flows
                while (true) {
                    const inf_capacity = if (@typeInfo(C) == .int)
                        std.math.maxInt(C)
                    else if (@typeInfo(C) == .float)
                        std.math.inf(C)
                    else
                        @compileError("Capacity type must be integer or float");

                    const flow = try sendFlow(allocator, &residual, &level, &iter, source, sink, inf_capacity, zero_capacity, context);
                    if (std.math.order(flow, zero_capacity) == .eq) break;
                    max_flow = max_flow + flow;
                }
            }

            return Result{
                .max_flow = max_flow,
            };
        }

        /// Initialize residual graph from input graph
        fn initResidualGraph(
            allocator: Allocator,
            graph: anytype,
            residual: *AdjacencyList,
            zero_capacity: C,
        ) !void {
            // Iterate over all vertices in the graph
            var vertex_iter = graph.adjacencies.iterator();
            while (vertex_iter.next()) |entry| {
                const u = entry.key_ptr.*;
                const neighbors = entry.value_ptr.edges.items;

                const result_u = try residual.getOrPut(u);
                if (!result_u.found_existing) {
                    result_u.value_ptr.* = .{};
                }

                for (neighbors) |edge| {
                    const v = edge.target;

                    // Ensure target vertex exists in residual graph
                    const result_v = try residual.getOrPut(v);
                    if (!result_v.found_existing) {
                        result_v.value_ptr.* = .{};
                    }

                    // Forward edge index
                    const forward_idx = result_u.value_ptr.items.len;
                    // Reverse edge index (will be next)
                    const reverse_idx = result_v.value_ptr.items.len;

                    // Forward edge (original capacity)
                    try result_u.value_ptr.append(allocator, .{
                        .target = v,
                        .capacity = edge.weight,
                        .rev_index = reverse_idx,
                    });

                    // Backward edge (zero capacity initially)
                    try result_v.value_ptr.append(allocator, .{
                        .target = u,
                        .capacity = zero_capacity,
                        .rev_index = forward_idx,
                    });
                }
            }
        }

        /// Build level graph using BFS from source.
        /// Returns true if sink is reachable.
        /// Time: O(V + E) | Space: O(V)
        fn buildLevelGraph(
            allocator: Allocator,
            residual: *AdjacencyList,
            source: V,
            sink: V,
            level: *LevelMap,
            zero_capacity: C,
            _: Context,
        ) !bool {
            level.clearRetainingCapacity();

            var queue: std.ArrayList(V) = .{};
            defer queue.deinit(allocator);

            try level.put(source, 0);
            try queue.append(allocator, source);

            var read_idx: usize = 0;
            while (read_idx < queue.items.len) : (read_idx += 1) {
                const u = queue.items[read_idx];
                const u_level = level.get(u).?;

                if (residual.get(u)) |neighbors| {
                    for (neighbors.items) |edge| {
                        if (std.math.order(edge.capacity, zero_capacity) == .gt and !level.contains(edge.target)) {
                            try level.put(edge.target, u_level + 1);
                            try queue.append(allocator, edge.target);
                        }
                    }
                }
            }

            return level.contains(sink);
        }

        /// Find blocking flow using DFS.
        /// Time: O(VE) per phase | Space: O(V)
        fn sendFlow(
            allocator: Allocator,
            residual: *AdjacencyList,
            level: *LevelMap,
            iter: *IteratorMap,
            u: V,
            sink: V,
            flow: C,
            zero_capacity: C,
            context: Context,
        ) error{OutOfMemory}!C {
            if (context.eql(u, sink)) return flow;

            const u_level = level.get(u) orelse return zero_capacity;

            // Get current iterator position for u
            const iter_pos = iter.get(u) orelse 0;

            if (residual.getPtr(u)) |neighbors| {
                var i = iter_pos;
                while (i < neighbors.items.len) : (i += 1) {
                    const edge = &neighbors.items[i];

                    if (std.math.order(edge.capacity, zero_capacity) != .gt) continue;

                    const v_level = level.get(edge.target) orelse continue;
                    if (v_level != u_level + 1) continue;

                    // Calculate minimum flow we can push
                    const min_flow = if (std.math.order(flow, edge.capacity) == .lt) flow else edge.capacity;

                    const pushed = try sendFlow(allocator, residual, level, iter, edge.target, sink, min_flow, zero_capacity, context);
                    if (std.math.order(pushed, zero_capacity) == .gt) {
                        // Update residual capacities
                        edge.capacity = edge.capacity - pushed;

                        // Update reverse edge
                        if (residual.getPtr(edge.target)) |rev_neighbors| {
                            rev_neighbors.items[edge.rev_index].capacity = rev_neighbors.items[edge.rev_index].capacity + pushed;
                        }

                        // Update iterator position
                        try iter.put(u, i);
                        return pushed;
                    }
                }

                // Mark all edges as explored
                try iter.put(u, neighbors.items.len);
            }

            return zero_capacity;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const TestContext = struct {
    pub fn hash(_: TestContext, key: u32) u64 {
        return key;
    }

    pub fn eql(_: TestContext, a: u32, b: u32) bool {
        return a == b;
    }
};

const GraphAdjList = @import("../../containers/graphs/adjacency_list.zig").AdjacencyList;
const Graph = GraphAdjList(u32, i32, TestContext, TestContext.hash, TestContext.eql);

test "Dinic: basic max flow (0->5 in example graph)" {
    const allocator = testing.allocator;
    const ctx = TestContext{};

    // Same graph as EdmondsKarp test:
    //     10        10
    //  0 ----> 1 ----> 5
    //  |       |       ^
    //  | 10    | 2     | 10
    //  v       v       |
    //  2 ----> 3 ----> 4
    //     4        9
    //
    // Max flow from 0 to 5 should be 14
    var graph = Graph.init(allocator, ctx, true);
    defer graph.deinit();

    try graph.addVertex(0);
    try graph.addVertex(1);
    try graph.addVertex(2);
    try graph.addVertex(3);
    try graph.addVertex(4);
    try graph.addVertex(5);

    try graph.addEdge(0, 1, 10);
    try graph.addEdge(0, 2, 10);
    try graph.addEdge(1, 3, 2);
    try graph.addEdge(1, 5, 10);
    try graph.addEdge(2, 3, 4);
    try graph.addEdge(3, 4, 9);
    try graph.addEdge(4, 5, 10);

    const D = Dinic(u32, i32, TestContext);
    var result = try D.run(allocator, graph, 0, 5, ctx, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 14), result.max_flow);
}

test "Dinic: single edge" {
    const allocator = testing.allocator;
    const ctx = TestContext{};

    // 0 --10--> 1
    var graph = Graph.init(allocator, ctx, true);
    defer graph.deinit();

    try graph.addVertex(0);
    try graph.addVertex(1);
    try graph.addEdge(0, 1, 10);

    const D = Dinic(u32, i32, TestContext);
    var result = try D.run(allocator, graph, 0, 1, ctx, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 10), result.max_flow);
}

test "Dinic: no path from source to sink" {
    const allocator = testing.allocator;
    const ctx = TestContext{};

    // 0 --10--> 1    2 --5--> 3
    // Disconnected: no path from 0 to 3
    var graph = Graph.init(allocator, ctx, true);
    defer graph.deinit();

    try graph.addVertex(0);
    try graph.addVertex(1);
    try graph.addVertex(2);
    try graph.addVertex(3);
    try graph.addEdge(0, 1, 10);
    try graph.addEdge(2, 3, 5);

    const D = Dinic(u32, i32, TestContext);
    var result = try D.run(allocator, graph, 0, 3, ctx, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 0), result.max_flow);
}

test "Dinic: diamond graph" {
    const allocator = testing.allocator;
    const ctx = TestContext{};

    //      10
    //   0 ----> 1
    //   |       |
    // 10|       | 10
    //   v       v
    //   2 ----> 3
    //      10
    var graph = Graph.init(allocator, ctx, true);
    defer graph.deinit();

    try graph.addVertex(0);
    try graph.addVertex(1);
    try graph.addVertex(2);
    try graph.addVertex(3);

    try graph.addEdge(0, 1, 10);
    try graph.addEdge(0, 2, 10);
    try graph.addEdge(1, 3, 10);
    try graph.addEdge(2, 3, 10);

    const D = Dinic(u32, i32, TestContext);
    var result = try D.run(allocator, graph, 0, 3, ctx, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 20), result.max_flow);
}

test "Dinic: unit capacity network (faster convergence)" {
    const allocator = testing.allocator;
    const ctx = TestContext{};

    // Unit capacity network - where Dinic shines
    // Linear chain: 0 -> 1 -> 2 -> 3
    var graph = Graph.init(allocator, ctx, true);
    defer graph.deinit();

    try graph.addVertex(0);
    try graph.addVertex(1);
    try graph.addVertex(2);
    try graph.addVertex(3);

    try graph.addEdge(0, 1, 1);
    try graph.addEdge(1, 2, 1);
    try graph.addEdge(2, 3, 1);

    const D = Dinic(u32, i32, TestContext);
    var result = try D.run(allocator, graph, 0, 3, ctx, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 1), result.max_flow);
}

test "Dinic: source equals sink (error)" {
    const allocator = testing.allocator;
    const ctx = TestContext{};

    var graph = Graph.init(allocator, ctx, true);
    defer graph.deinit();

    try graph.addVertex(0);

    const D = Dinic(u32, i32, TestContext);
    const result = D.run(allocator, graph, 0, 0, ctx, 0);
    try testing.expectError(error.SourceEqualsSink, result);
}

test "Dinic: parallel edges" {
    const allocator = testing.allocator;
    const ctx = TestContext{};

    // Two parallel edges from 0 to 1: capacity 10 and 5
    var graph = Graph.init(allocator, ctx, true);
    defer graph.deinit();

    try graph.addVertex(0);
    try graph.addVertex(1);
    try graph.addEdge(0, 1, 10);
    try graph.addEdge(0, 1, 5);

    const D = Dinic(u32, i32, TestContext);
    var result = try D.run(allocator, graph, 0, 1, ctx, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 15), result.max_flow);
}
