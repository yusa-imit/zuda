const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Edmonds-Karp - Maximum flow algorithm using Ford-Fulkerson with BFS.
///
/// Computes the maximum flow from a source vertex to a sink vertex in a flow network.
/// Uses BFS to find augmenting paths, ensuring O(VE²) time complexity.
///
/// Time Complexity: O(VE²) where V is vertices, E is edges
/// Space Complexity: O(V + E) for residual graph and BFS structures
///
/// The algorithm works on a flow network where:
/// - Each edge has a capacity
/// - Flow must satisfy capacity constraints and flow conservation
/// - Maximum flow is found by repeatedly finding augmenting paths using BFS
///
/// Generic parameters:
/// - V: Vertex type (must be hashable)
/// - C: Capacity type (must support comparison, addition, subtraction)
/// - Context: Context type for hashing/comparing vertices
pub fn EdmondsKarp(
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

        /// Edge in the flow network
        pub const Edge = struct {
            target: V,
            capacity: C,
        };

        /// Residual edge (used internally for residual graph)
        const ResidualEdge = struct {
            target: V,
            capacity: C,
            is_reverse: bool,
        };

        /// Result of max flow computation
        pub const Result = struct {
            /// Maximum flow value from source to sink
            max_flow: C,
            /// Flow on each edge (source, target) -> flow
            flows: std.HashMap(EdgeKey, C, EdgeKeyContext, std.hash_map.default_max_load_percentage),
            /// Minimum cut vertices (source side)
            min_cut_source_side: std.HashMap(V, void, HashMapContext, std.hash_map.default_max_load_percentage),
            allocator: Allocator,

            const EdgeKey = struct {
                source: V,
                target: V,
            };

            const EdgeKeyContext = struct {
                user_ctx: Context,

                pub fn hash(ctx: @This(), key: EdgeKey) u64 {
                    const h1 = ctx.user_ctx.hash(key.source);
                    const h2 = ctx.user_ctx.hash(key.target);
                    return h1 ^ (h2 +% 0x9e3779b9 +% (h1 << 6) +% (h1 >> 2));
                }

                pub fn eql(ctx: @This(), a: EdgeKey, b: EdgeKey) bool {
                    return ctx.user_ctx.eql(a.source, b.source) and ctx.user_ctx.eql(a.target, b.target);
                }
            };

            pub fn deinit(self: *Result) void {
                self.flows.deinit();
                self.min_cut_source_side.deinit();
            }

            /// Get the flow on edge (u, v).
            /// Returns zero_capacity if edge doesn't exist.
            pub fn getFlow(self: *const Result, source: V, target: V, zero_capacity: C) C {
                return self.flows.get(.{ .source = source, .target = target }) orelse zero_capacity;
            }

            /// Check if vertex is on source side of minimum cut.
            pub fn isInMinCut(self: *const Result, vertex: V) bool {
                return self.min_cut_source_side.contains(vertex);
            }
        };

        /// Run Edmonds-Karp algorithm to find maximum flow.
        ///
        /// Graph type must provide:
        /// - `getNeighbors(vertex: V) -> ?[]const Edge` where Edge has `.target: V` and `.capacity: C`
        ///
        /// Capacity type C must support:
        /// - Comparison operators (for std.math.order)
        /// - Addition operator (+)
        /// - Subtraction operator (-)
        /// - Must be numeric or numeric-like
        ///
        /// Returns error if source equals sink.
        ///
        /// Time: O(VE²) | Space: O(V + E)
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
            var residual = std.HashMap(V, std.ArrayList(ResidualEdge), HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            defer {
                var it = residual.iterator();
                while (it.next()) |entry| {
                    entry.value_ptr.deinit(allocator);
                }
                residual.deinit();
            }

            // Initialize residual graph from input graph
            try initResidualGraph(allocator, graph, &residual, context, zero_capacity);

            // Track flow on each edge
            var flows = std.HashMap(Result.EdgeKey, C, Result.EdgeKeyContext, std.hash_map.default_max_load_percentage).init(allocator);
            errdefer flows.deinit();

            var max_flow = zero_capacity;

            // Find augmenting paths using BFS
            while (true) {
                const path_result = try findAugmentingPath(allocator, &residual, source, sink, context, zero_capacity);
                if (path_result.path == null) break;

                defer allocator.free(path_result.path.?);
                const path = path_result.path.?;
                const bottleneck = path_result.bottleneck;

                // Update flows along the path
                for (0..path.len - 1) |i| {
                    const u = path[i];
                    const v = path[i + 1];

                    // Update flow map
                    const key = Result.EdgeKey{ .source = u, .target = v };
                    const current_flow = flows.get(key) orelse zero_capacity;
                    try flows.put(key, current_flow + bottleneck);

                    // Update residual graph
                    try updateResidualGraph(allocator, &residual, u, v, bottleneck, context, zero_capacity);
                }

                max_flow = max_flow + bottleneck;
            }

            // Compute minimum cut (vertices reachable from source in residual graph)
            var min_cut = std.HashMap(V, void, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            errdefer min_cut.deinit();
            try computeMinCut(allocator, &residual, source, &min_cut, context);

            return Result{
                .max_flow = max_flow,
                .flows = flows,
                .min_cut_source_side = min_cut,
                .allocator = allocator,
            };
        }

        /// Initialize residual graph from input graph
        fn initResidualGraph(
            allocator: Allocator,
            graph: anytype,
            residual: *std.HashMap(V, std.ArrayList(ResidualEdge), HashMapContext, std.hash_map.default_max_load_percentage),
            _: Context,
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

                    // Forward edge (original capacity)
                    try result_u.value_ptr.append(allocator, .{
                        .target = v,
                        .capacity = edge.weight,
                        .is_reverse = false,
                    });

                    // Backward edge (zero capacity initially)
                    const result_v = try residual.getOrPut(v);
                    if (!result_v.found_existing) {
                        result_v.value_ptr.* = .{};
                    }
                    try result_v.value_ptr.append(allocator, .{
                        .target = u,
                        .capacity = zero_capacity,
                        .is_reverse = true,
                    });
                }
            }
        }

        /// Find augmenting path using BFS
        const PathResult = struct {
            path: ?[]V,
            bottleneck: C,
        };

        fn findAugmentingPath(
            allocator: Allocator,
            residual: *std.HashMap(V, std.ArrayList(ResidualEdge), HashMapContext, std.hash_map.default_max_load_percentage),
            source: V,
            sink: V,
            context: Context,
            zero_capacity: C,
        ) !PathResult {
            var visited = std.HashMap(V, void, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            defer visited.deinit();

            var parent = std.HashMap(V, V, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            defer parent.deinit();

            var capacity_to = std.HashMap(V, C, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            defer capacity_to.deinit();

            var queue: std.ArrayList(V) = .{};
            defer queue.deinit(allocator);

            try queue.append(allocator, source);
            try visited.put(source, {});
            // Use a very large capacity value for source (acts as infinity)
            // For integers use maxInt, for floats use inf
            const inf_capacity = if (@typeInfo(C) == .int)
                std.math.maxInt(C)
            else if (@typeInfo(C) == .float)
                std.math.inf(C)
            else
                @compileError("Capacity type must be integer or float");
            try capacity_to.put(source, inf_capacity);

            var found_sink = false;

            while (queue.items.len > 0) {
                const u = queue.orderedRemove(0);

                if (context.eql(u, sink)) {
                    found_sink = true;
                    break;
                }

                const neighbors = residual.get(u) orelse continue;
                for (neighbors.items) |edge| {
                    const v = edge.target;

                    // Only consider edges with remaining capacity
                    if (std.math.order(edge.capacity, zero_capacity) == .gt and !visited.contains(v)) {
                        try visited.put(v, {});
                        try parent.put(v, u);

                        const parent_capacity = capacity_to.get(u).?;
                        const path_capacity = if (std.math.order(parent_capacity, edge.capacity) == .lt)
                            parent_capacity
                        else
                            edge.capacity;
                        try capacity_to.put(v, path_capacity);

                        try queue.append(allocator, v);
                    }
                }
            }

            if (!found_sink) {
                return PathResult{ .path = null, .bottleneck = zero_capacity };
            }

            // Reconstruct path
            var path: std.ArrayList(V) = .{};
            errdefer path.deinit(allocator);

            var current = sink;
            while (true) {
                try path.append(allocator, current);
                if (context.eql(current, source)) break;
                current = parent.get(current).?;
            }

            std.mem.reverse(V, path.items);
            const bottleneck = capacity_to.get(sink).?;

            return PathResult{
                .path = try path.toOwnedSlice(allocator),
                .bottleneck = bottleneck,
            };
        }

        /// Update residual graph after augmenting flow
        fn updateResidualGraph(
            allocator: Allocator,
            residual: *std.HashMap(V, std.ArrayList(ResidualEdge), HashMapContext, std.hash_map.default_max_load_percentage),
            u: V,
            v: V,
            flow: C,
            context: Context,
            zero_capacity: C,
        ) !void {
            // Decrease forward edge capacity
            if (residual.getPtr(u)) |edges| {
                for (edges.items, 0..) |*edge, i| {
                    if (context.eql(edge.target, v)) {
                        edge.capacity = edge.capacity - flow;
                        if (std.math.order(edge.capacity, zero_capacity) == .eq) {
                            _ = edges.swapRemove(i);
                        }
                        break;
                    }
                }
            }

            // Increase backward edge capacity
            const result_v = try residual.getOrPut(v);
            if (!result_v.found_existing) {
                result_v.value_ptr.* = .{};
            }

            var found = false;
            for (result_v.value_ptr.items) |*edge| {
                if (context.eql(edge.target, u)) {
                    edge.capacity = edge.capacity + flow;
                    found = true;
                    break;
                }
            }

            if (!found) {
                try result_v.value_ptr.append(allocator, .{
                    .target = u,
                    .capacity = flow,
                    .is_reverse = true,
                });
            }
        }

        /// Compute minimum cut (vertices reachable from source in residual graph)
        fn computeMinCut(
            allocator: Allocator,
            residual: *std.HashMap(V, std.ArrayList(ResidualEdge), HashMapContext, std.hash_map.default_max_load_percentage),
            source: V,
            min_cut: *std.HashMap(V, void, HashMapContext, std.hash_map.default_max_load_percentage),
            _: Context,
        ) !void {
            var visited = std.HashMap(V, void, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            defer visited.deinit();

            var stack: std.ArrayList(V) = .{};
            defer stack.deinit(allocator);

            try stack.append(allocator, source);
            try visited.put(source, {});

            while (stack.items.len > 0) {
                const u = stack.pop().?;
                try min_cut.put(u, {});

                const neighbors = residual.get(u) orelse continue;
                for (neighbors.items) |edge| {
                    const v = edge.target;
                    if (!visited.contains(v)) {
                        try visited.put(v, {});
                        try stack.append(allocator, v);
                    }
                }
            }
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

const AdjacencyList = @import("../../containers/graphs/adjacency_list.zig").AdjacencyList;
const Graph = AdjacencyList(u32, i32, TestContext, TestContext.hash, TestContext.eql);

test "EdmondsKarp: basic max flow (0->5 in example graph)" {
    const allocator = testing.allocator;
    const ctx = TestContext{};

    // Build graph:
    //     10        10
    //  0 ----> 1 ----> 5
    //  |       |       ^
    //  | 10    | 2     | 10
    //  v       v       |
    //  2 ----> 3 ----> 4
    //     4        9
    //
    // Max flow from 0 to 5 should be 14
    // - 0→1→5: 8 units
    // - 0→1→3→4→5: 2 units
    // - 0→2→3→4→5: 4 units
    // Total: 14
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

    const EK = EdmondsKarp(u32, i32, TestContext);
    var result = try EK.run(allocator, graph, 0, 5, ctx, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 14), result.max_flow);

    // Verify source is in min cut
    try testing.expect(result.isInMinCut(0));
}

test "EdmondsKarp: single edge" {
    const allocator = testing.allocator;
    const ctx = TestContext{};

    // 0 --10--> 1
    var graph = Graph.init(allocator, ctx, true);
    defer graph.deinit();

    try graph.addVertex(0);
    try graph.addVertex(1);
    try graph.addEdge(0, 1, 10);

    const EK = EdmondsKarp(u32, i32, TestContext);
    var result = try EK.run(allocator, graph, 0, 1, ctx, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 10), result.max_flow);
    try testing.expectEqual(@as(i32, 10), result.getFlow(0, 1, 0));
}

test "EdmondsKarp: no path from source to sink" {
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

    const EK = EdmondsKarp(u32, i32, TestContext);
    var result = try EK.run(allocator, graph, 0, 3, ctx, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 0), result.max_flow);
}

test "EdmondsKarp: parallel edges" {
    const allocator = testing.allocator;
    const ctx = TestContext{};

    // Two parallel edges from 0 to 1: capacity 10 and 5
    var graph = Graph.init(allocator, ctx, true);
    defer graph.deinit();

    try graph.addVertex(0);
    try graph.addVertex(1);
    try graph.addEdge(0, 1, 10);
    try graph.addEdge(0, 1, 5);

    const EK = EdmondsKarp(u32, i32, TestContext);
    var result = try EK.run(allocator, graph, 0, 1, ctx, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 15), result.max_flow);
}

test "EdmondsKarp: zero capacity edges" {
    const allocator = testing.allocator;
    const ctx = TestContext{};

    // 0 --0--> 1
    var graph = Graph.init(allocator, ctx, true);
    defer graph.deinit();

    try graph.addVertex(0);
    try graph.addVertex(1);
    try graph.addEdge(0, 1, 0);

    const EK = EdmondsKarp(u32, i32, TestContext);
    var result = try EK.run(allocator, graph, 0, 1, ctx, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 0), result.max_flow);
}

test "EdmondsKarp: floating point capacities" {
    const allocator = testing.allocator;
    const FContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };
    const ctx = FContext{};

    const FGraph = AdjacencyList(u32, f64, FContext, FContext.hash, FContext.eql);
    var graph = FGraph.init(allocator, ctx, true);
    defer graph.deinit();

    try graph.addVertex(0);
    try graph.addVertex(1);
    try graph.addEdge(0, 1, 10.5);

    const EK = EdmondsKarp(u32, f64, FContext);
    var result = try EK.run(allocator, graph, 0, 1, ctx, 0.0);
    defer result.deinit();

    try testing.expectApproxEqAbs(@as(f64, 10.5), result.max_flow, 0.001);
}

test "EdmondsKarp: multiple sources (not directly supported, use super-source)" {
    const allocator = testing.allocator;
    const ctx = TestContext{};

    // Create a super-source connected to multiple sources
    //       S
    //      / \
    //    10   15
    //    /     \
    //   0       1
    //    \     /
    //    5   10
    //     \ /
    //      2
    var graph = Graph.init(allocator, ctx, true);
    defer graph.deinit();

    try graph.addVertex(100); // Super-source S
    try graph.addVertex(0);
    try graph.addVertex(1);
    try graph.addVertex(2);

    try graph.addEdge(100, 0, 10);
    try graph.addEdge(100, 1, 15);
    try graph.addEdge(0, 2, 5);
    try graph.addEdge(1, 2, 10);

    const EK = EdmondsKarp(u32, i32, TestContext);
    var result = try EK.run(allocator, graph, 100, 2, ctx, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 15), result.max_flow);
}

test "EdmondsKarp: diamond graph" {
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

    const EK = EdmondsKarp(u32, i32, TestContext);
    var result = try EK.run(allocator, graph, 0, 3, ctx, 0);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 20), result.max_flow);
}

test "EdmondsKarp: source equals sink (error)" {
    const allocator = testing.allocator;
    const ctx = TestContext{};

    var graph = Graph.init(allocator, ctx, true);
    defer graph.deinit();

    try graph.addVertex(0);

    const EK = EdmondsKarp(u32, i32, TestContext);
    const result = EK.run(allocator, graph, 0, 0, ctx, 0);
    try testing.expectError(error.SourceEqualsSink, result);
}

test "EdmondsKarp: stress test (grid graph 10x10)" {
    const allocator = testing.allocator;
    const ctx = TestContext{};

    var graph = Graph.init(allocator, ctx, true);
    defer graph.deinit();

    // Create a 10x10 grid with unit capacities
    const n = 10;
    for (0..n) |i| {
        for (0..n) |j| {
            const node = @as(u32, @intCast(i * n + j));

            // Right edge
            if (j + 1 < n) {
                const right = @as(u32, @intCast(i * n + (j + 1)));
                try graph.addEdge(node, right, 1);
            }

            // Down edge
            if (i + 1 < n) {
                const down = @as(u32, @intCast((i + 1) * n + j));
                try graph.addEdge(node, down, 1);
            }
        }
    }

    const EK = EdmondsKarp(u32, i32, TestContext);
    var result = try EK.run(allocator, graph, 0, n * n - 1, ctx, 0);
    defer result.deinit();

    // Max flow in grid from top-left to bottom-right with unit capacities
    // Should be 2 (2 edge-disjoint paths)
    try testing.expectEqual(@as(i32, 2), result.max_flow);
}
