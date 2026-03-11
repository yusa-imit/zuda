const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Push-Relabel - Maximum flow algorithm using preflow-push approach.
///
/// Computes the maximum flow from a source vertex to a sink vertex in a flow network.
/// Uses local push and relabel operations rather than finding augmenting paths globally.
/// Achieves O(V²E) time with FIFO selection, O(V³) with basic implementation.
///
/// Time Complexity: O(V³) basic, O(V²E) with FIFO
/// Space Complexity: O(V + E) for residual graph and vertex attributes
///
/// The algorithm maintains:
/// - Preflow: flow that may violate flow conservation at non-source/sink vertices
/// - Height function: labels guiding push operations
/// - Excess: amount of excess flow at each vertex
///
/// Operations:
/// - Push: send excess flow from vertex to lower neighbor
/// - Relabel: increase vertex height to enable new pushes
///
/// Generic parameters:
/// - V: Vertex type (must be hashable)
/// - C: Capacity type (must support comparison, addition, subtraction)
/// - Context: Context type for hashing/comparing vertices
pub fn PushRelabel(
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

        /// Residual edge (used internally)
        const ResidualEdge = struct {
            target: V,
            capacity: C,
            rev_index: usize,
        };

        /// Vertex attributes for push-relabel
        const VertexData = struct {
            height: usize,
            excess: C,
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

        /// Run Push-Relabel algorithm to find maximum flow.
        ///
        /// Graph type must provide:
        /// - `adjacencies` field with iterator() returning vertices and edges
        ///
        /// Capacity type C must support:
        /// - Comparison operators
        /// - Addition operator (+)
        /// - Subtraction operator (-)
        /// - Must be numeric or numeric-like
        ///
        /// Returns error if source equals sink.
        ///
        /// Time: O(V³) | Space: O(V + E)
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

            // Initialize residual graph
            try initResidualGraph(allocator, graph, &residual, context, zero_capacity);

            // Vertex data: height and excess
            var vertex_data = std.HashMap(V, VertexData, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            defer vertex_data.deinit();

            // Initialize heights and excesses
            var vertex_iter = residual.keyIterator();
            while (vertex_iter.next()) |v_ptr| {
                const v = v_ptr.*;
                const height: usize = if (context.eql(v, source)) residual.count() else 0;
                try vertex_data.put(v, .{
                    .height = height,
                    .excess = zero_capacity,
                });
            }

            // Saturate edges from source
            if (residual.getPtr(source)) |source_edges| {
                for (source_edges.items) |*edge| {
                    if (@as(i64, @intCast(edge.capacity)) > @as(i64, @intCast(zero_capacity))) {
                        const flow = edge.capacity;

                        // Update excess at target
                        if (vertex_data.getPtr(edge.target)) |target_data| {
                            target_data.excess = target_data.excess + flow;
                        }

                        // Update residual capacities
                        const old_cap = edge.capacity;
                        edge.capacity = zero_capacity;

                        // Update reverse edge
                        if (residual.getPtr(edge.target)) |target_edges| {
                            target_edges.items[edge.rev_index].capacity =
                                target_edges.items[edge.rev_index].capacity + old_cap;
                        }
                    }
                }
            }

            // Active vertices queue (FIFO for better complexity)
            var active: std.ArrayList(V) = .{};
            defer active.deinit(allocator);

            // Add all vertices with excess (except source and sink)
            var vd_iter = vertex_data.iterator();
            while (vd_iter.next()) |entry| {
                const v = entry.key_ptr.*;
                if (!context.eql(v, source) and !context.eql(v, sink)) {
                    if (@as(i64, @intCast(entry.value_ptr.excess)) > @as(i64, @intCast(zero_capacity))) {
                        try active.append(allocator, v);
                    }
                }
            }

            // Main loop: process active vertices
            while (active.items.len > 0) {
                const u = active.orderedRemove(0);
                try discharge(allocator, &residual, &vertex_data, &active, u, sink, context, zero_capacity);
            }

            // Compute max flow (sum of excess at sink)
            const max_flow = if (vertex_data.get(sink)) |sink_data| sink_data.excess else zero_capacity;

            // Reconstruct flows
            var flows = std.HashMap(Result.EdgeKey, C, Result.EdgeKeyContext, std.hash_map.default_max_load_percentage).init(allocator);
            errdefer flows.deinit();
            try reconstructFlows(allocator, graph, &residual, &flows, context, zero_capacity);

            // Compute minimum cut
            var min_cut = std.HashMap(V, void, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            errdefer min_cut.deinit();
            try computeMinCut(allocator, &residual, source, &min_cut, context, zero_capacity);

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

                    const result_v = try residual.getOrPut(v);
                    if (!result_v.found_existing) {
                        result_v.value_ptr.* = .{};
                    }

                    // Add forward edge and remember its index
                    const fwd_idx = result_u.value_ptr.items.len;
                    const rev_idx = result_v.value_ptr.items.len;

                    try result_u.value_ptr.append(allocator, .{
                        .target = v,
                        .capacity = edge.capacity,
                        .rev_index = rev_idx,
                    });

                    try result_v.value_ptr.append(allocator, .{
                        .target = u,
                        .capacity = zero_capacity,
                        .rev_index = fwd_idx,
                    });
                }
            }
        }

        /// Push operation: send flow from u to v
        fn push(
            residual: *std.HashMap(V, std.ArrayList(ResidualEdge), HashMapContext, std.hash_map.default_max_load_percentage),
            vertex_data: *std.HashMap(V, VertexData, HashMapContext, std.hash_map.default_max_load_percentage),
            u: V,
            edge_idx: usize,
            _: C,
        ) void {
            const u_edges = residual.getPtr(u).?;
            const edge = &u_edges.items[edge_idx];
            const v = edge.target;

            const u_data = vertex_data.getPtr(u).?;
            const v_data = vertex_data.getPtr(v).?;

            // Compute flow to push (min of excess and capacity)
            const flow = blk: {
                if (@as(i64, @intCast(u_data.excess)) < @as(i64, @intCast(edge.capacity))) {
                    break :blk u_data.excess;
                } else {
                    break :blk edge.capacity;
                }
            };

            // Update excesses
            u_data.excess = u_data.excess - flow;
            v_data.excess = v_data.excess + flow;

            // Update residual capacities
            edge.capacity = edge.capacity - flow;

            // Update reverse edge
            const v_edges = residual.getPtr(v).?;
            v_edges.items[edge.rev_index].capacity =
                v_edges.items[edge.rev_index].capacity + flow;
        }

        /// Relabel operation: increase height of vertex
        fn relabel(
            residual: *std.HashMap(V, std.ArrayList(ResidualEdge), HashMapContext, std.hash_map.default_max_load_percentage),
            vertex_data: *std.HashMap(V, VertexData, HashMapContext, std.hash_map.default_max_load_percentage),
            u: V,
            zero_capacity: C,
        ) void {
            var min_height: usize = std.math.maxInt(usize);

            if (residual.get(u)) |edges| {
                for (edges.items) |edge| {
                    if (@as(i64, @intCast(edge.capacity)) > @as(i64, @intCast(zero_capacity))) {
                        if (vertex_data.get(edge.target)) |neighbor_data| {
                            if (neighbor_data.height < min_height) {
                                min_height = neighbor_data.height;
                            }
                        }
                    }
                }
            }

            const u_data = vertex_data.getPtr(u).?;
            if (min_height < std.math.maxInt(usize)) {
                u_data.height = min_height + 1;
            } else {
                // No outgoing residual edges: set height to n (unreachable from sink)
                u_data.height = vertex_data.count();
            }
        }

        /// Discharge: process vertex until excess is zero
        fn discharge(
            allocator: Allocator,
            residual: *std.HashMap(V, std.ArrayList(ResidualEdge), HashMapContext, std.hash_map.default_max_load_percentage),
            vertex_data: *std.HashMap(V, VertexData, HashMapContext, std.hash_map.default_max_load_percentage),
            active: *std.ArrayList(V),
            u: V,
            sink: V,
            context: Context,
            zero_capacity: C,
        ) !void {
            var current_edge: usize = 0;
            const max_height = 2 * vertex_data.count(); // Height bound: prevents infinite loop

            while (true) {
                const u_data = vertex_data.get(u).?;
                if (@as(i64, @intCast(u_data.excess)) <= @as(i64, @intCast(zero_capacity))) return;

                // If height exceeds bound, vertex can't reach sink - stop processing
                if (u_data.height >= max_height) return;

                const edges_ptr = residual.getPtr(u) orelse return;
                if (edges_ptr.items.len == 0) return;

                // Try to push on current edge
                if (current_edge < edges_ptr.items.len) {
                    const edge = edges_ptr.items[current_edge];

                    if (@as(i64, @intCast(edge.capacity)) > @as(i64, @intCast(zero_capacity))) {
                        const u_height = vertex_data.get(u).?.height;
                        if (vertex_data.get(edge.target)) |v_data| {
                            if (u_height == v_data.height + 1) {
                                // Admissible edge: push
                                const v = edge.target;
                                const old_v_excess = v_data.excess;

                                push(residual, vertex_data, u, current_edge, zero_capacity);

                                // Add v to active queue if it became active
                                const new_v_excess = vertex_data.get(v).?.excess;
                                if (@as(i64, @intCast(old_v_excess)) <= @as(i64, @intCast(zero_capacity)) and
                                    @as(i64, @intCast(new_v_excess)) > @as(i64, @intCast(zero_capacity)) and
                                    !context.eql(v, sink))
                                {
                                    try active.append(allocator, v);
                                }

                                // Check if excess is now zero
                                const current_excess = vertex_data.get(u).?.excess;
                                if (@as(i64, @intCast(current_excess)) <= @as(i64, @intCast(zero_capacity))) {
                                    return;
                                }

                                // Continue with next edge
                                continue;
                            }
                        }
                    }

                    // Move to next edge
                    current_edge += 1;
                } else {
                    // Scanned all edges without finding admissible one: relabel
                    relabel(residual, vertex_data, u, zero_capacity);
                    current_edge = 0; // Reset to first edge after relabel
                }
            }
        }

        /// Reconstruct actual edge flows from residual graph
        fn reconstructFlows(
            allocator: Allocator,
            graph: anytype,
            residual: *std.HashMap(V, std.ArrayList(ResidualEdge), HashMapContext, std.hash_map.default_max_load_percentage),
            flows: *std.HashMap(Result.EdgeKey, C, Result.EdgeKeyContext, std.hash_map.default_max_load_percentage),
            context: Context,
            zero_capacity: C,
        ) !void {
            _ = allocator;

            var vertex_iter = graph.adjacencies.iterator();
            while (vertex_iter.next()) |entry| {
                const u = entry.key_ptr.*;
                const neighbors = entry.value_ptr.edges.items;

                for (neighbors) |edge| {
                    const v = edge.target;
                    const original_capacity = edge.capacity;

                    // Find residual capacity
                    var residual_capacity = zero_capacity;
                    if (residual.get(u)) |edges| {
                        for (edges.items) |res_edge| {
                            if (context.eql(res_edge.target, v)) {
                                residual_capacity = res_edge.capacity;
                                break;
                            }
                        }
                    }

                    // Flow = original - residual
                    const flow = original_capacity - residual_capacity;
                    if (@as(i64, @intCast(flow)) > @as(i64, @intCast(zero_capacity))) {
                        try flows.put(.{ .source = u, .target = v }, flow);
                    }
                }
            }
        }

        /// Compute minimum cut
        fn computeMinCut(
            allocator: Allocator,
            residual: *std.HashMap(V, std.ArrayList(ResidualEdge), HashMapContext, std.hash_map.default_max_load_percentage),
            source: V,
            min_cut: *std.HashMap(V, void, HashMapContext, std.hash_map.default_max_load_percentage),
            _: Context,
            zero_capacity: C,
        ) !void {
            var queue: std.ArrayList(V) = .{};
            defer queue.deinit(allocator);

            try min_cut.put(source, {});
            try queue.append(allocator, source);

            var read_idx: usize = 0;
            while (read_idx < queue.items.len) {
                const u = queue.items[read_idx];
                read_idx += 1;

                if (residual.get(u)) |edges| {
                    for (edges.items) |edge| {
                        if (@as(i64, @intCast(edge.capacity)) > @as(i64, @intCast(zero_capacity)) and !min_cut.contains(edge.target)) {
                            try min_cut.put(edge.target, {});
                            try queue.append(allocator, edge.target);
                        }
                    }
                }
            }
        }
    };
}

// Tests

test "PushRelabel: basic max flow" {
    const allocator = testing.allocator;

    const TestGraph = struct {
        const Vertex = u32;
        const Edge = PushRelabel(Vertex, u32, Context).Edge;
        const Context = struct {
            pub fn hash(_: @This(), v: Vertex) u64 {
                return v;
            }
            pub fn eql(_: @This(), a: Vertex, b: Vertex) bool {
                return a == b;
            }
        };
        const Adjacency = struct {
            edges: std.ArrayList(Edge),
        };
        adjacencies: std.HashMap(Vertex, Adjacency, Context, std.hash_map.default_max_load_percentage),

        fn init(alloc: Allocator) @This() {
            return .{
                .adjacencies = std.HashMap(Vertex, Adjacency, Context, std.hash_map.default_max_load_percentage).init(alloc),
            };
        }
        fn deinit(self: *@This()) void {
            var it = self.adjacencies.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.edges.deinit(self.adjacencies.allocator);
            }
            self.adjacencies.deinit();
        }
        fn addEdge(self: *@This(), from: Vertex, to: Vertex, capacity: u32) !void {
            const result = try self.adjacencies.getOrPut(from);
            if (!result.found_existing) {
                result.value_ptr.* = .{ .edges = .{} };
            }
            try result.value_ptr.edges.append(self.adjacencies.allocator, .{ .target = to, .capacity = capacity });
            const result_to = try self.adjacencies.getOrPut(to);
            if (!result_to.found_existing) {
                result_to.value_ptr.* = .{ .edges = .{} };
            }
        }
    };

    var graph = TestGraph.init(allocator);
    defer graph.deinit();

    // Simple flow network
    try graph.addEdge(0, 1, 10);
    try graph.addEdge(0, 2, 2);
    try graph.addEdge(1, 3, 10);
    try graph.addEdge(1, 4, 8);
    try graph.addEdge(2, 4, 10);
    try graph.addEdge(4, 3, 10);

    const PR = PushRelabel(TestGraph.Vertex, u32, TestGraph.Context);
    var result = try PR.run(allocator, graph, 0, 3, TestGraph.Context{}, 0);
    defer result.deinit();

    try testing.expectEqual(@as(u32, 12), result.max_flow);
}

test "PushRelabel: single edge" {
    const allocator = testing.allocator;

    const TestGraph = struct {
        const Vertex = u32;
        const Edge = PushRelabel(Vertex, u32, Context).Edge;
        const Context = struct {
            pub fn hash(_: @This(), v: Vertex) u64 {
                return v;
            }
            pub fn eql(_: @This(), a: Vertex, b: Vertex) bool {
                return a == b;
            }
        };
        const Adjacency = struct {
            edges: std.ArrayList(Edge),
        };
        adjacencies: std.HashMap(Vertex, Adjacency, Context, std.hash_map.default_max_load_percentage),

        fn init(alloc: Allocator) @This() {
            return .{
                .adjacencies = std.HashMap(Vertex, Adjacency, Context, std.hash_map.default_max_load_percentage).init(alloc),
            };
        }
        fn deinit(self: *@This()) void {
            var it = self.adjacencies.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.edges.deinit(self.adjacencies.allocator);
            }
            self.adjacencies.deinit();
        }
        fn addEdge(self: *@This(), from: Vertex, to: Vertex, capacity: u32) !void {
            const result = try self.adjacencies.getOrPut(from);
            if (!result.found_existing) {
                result.value_ptr.* = .{ .edges = .{} };
            }
            try result.value_ptr.edges.append(self.adjacencies.allocator, .{ .target = to, .capacity = capacity });
            const result_to = try self.adjacencies.getOrPut(to);
            if (!result_to.found_existing) {
                result_to.value_ptr.* = .{ .edges = .{} };
            }
        }
    };

    var graph = TestGraph.init(allocator);
    defer graph.deinit();

    try graph.addEdge(0, 1, 42);

    const PR = PushRelabel(TestGraph.Vertex, u32, TestGraph.Context);
    var result = try PR.run(allocator, graph, 0, 1, TestGraph.Context{}, 0);
    defer result.deinit();

    try testing.expectEqual(@as(u32, 42), result.max_flow);
}

test "PushRelabel: no path" {
    const allocator = testing.allocator;

    const TestGraph = struct {
        const Vertex = u32;
        const Edge = PushRelabel(Vertex, u32, Context).Edge;
        const Context = struct {
            pub fn hash(_: @This(), v: Vertex) u64 {
                return v;
            }
            pub fn eql(_: @This(), a: Vertex, b: Vertex) bool {
                return a == b;
            }
        };
        const Adjacency = struct {
            edges: std.ArrayList(Edge),
        };
        adjacencies: std.HashMap(Vertex, Adjacency, Context, std.hash_map.default_max_load_percentage),

        fn init(alloc: Allocator) @This() {
            return .{
                .adjacencies = std.HashMap(Vertex, Adjacency, Context, std.hash_map.default_max_load_percentage).init(alloc),
            };
        }
        fn deinit(self: *@This()) void {
            var it = self.adjacencies.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.edges.deinit(self.adjacencies.allocator);
            }
            self.adjacencies.deinit();
        }
        fn addEdge(self: *@This(), from: Vertex, to: Vertex, capacity: u32) !void {
            const result = try self.adjacencies.getOrPut(from);
            if (!result.found_existing) {
                result.value_ptr.* = .{ .edges = .{} };
            }
            try result.value_ptr.edges.append(self.adjacencies.allocator, .{ .target = to, .capacity = capacity });
            const result_to = try self.adjacencies.getOrPut(to);
            if (!result_to.found_existing) {
                result_to.value_ptr.* = .{ .edges = .{} };
            }
        }
    };

    var graph = TestGraph.init(allocator);
    defer graph.deinit();

    try graph.addEdge(0, 1, 10);
    try graph.addEdge(2, 3, 10);

    const PR = PushRelabel(TestGraph.Vertex, u32, TestGraph.Context);
    var result = try PR.run(allocator, graph, 0, 3, TestGraph.Context{}, 0);
    defer result.deinit();

    try testing.expectEqual(@as(u32, 0), result.max_flow);
}

test "PushRelabel: source equals sink" {
    const allocator = testing.allocator;

    const TestGraph = struct {
        const Vertex = u32;
        const Edge = PushRelabel(Vertex, u32, Context).Edge;
        const Context = struct {
            pub fn hash(_: @This(), v: Vertex) u64 {
                return v;
            }
            pub fn eql(_: @This(), a: Vertex, b: Vertex) bool {
                return a == b;
            }
        };
        const Adjacency = struct {
            edges: std.ArrayList(Edge),
        };
        adjacencies: std.HashMap(Vertex, Adjacency, Context, std.hash_map.default_max_load_percentage),

        fn init(alloc: Allocator) @This() {
            return .{
                .adjacencies = std.HashMap(Vertex, Adjacency, Context, std.hash_map.default_max_load_percentage).init(alloc),
            };
        }
        fn deinit(self: *@This()) void {
            var it = self.adjacencies.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.edges.deinit(self.adjacencies.allocator);
            }
            self.adjacencies.deinit();
        }
    };

    var graph = TestGraph.init(allocator);
    defer graph.deinit();

    const PR = PushRelabel(TestGraph.Vertex, u32, TestGraph.Context);
    const result = PR.run(allocator, graph, 0, 0, TestGraph.Context{}, 0);
    try testing.expectError(error.SourceEqualsSink, result);
}

test "PushRelabel: complex network" {
    const allocator = testing.allocator;

    const TestGraph = struct {
        const Vertex = u32;
        const Edge = PushRelabel(Vertex, u32, Context).Edge;
        const Context = struct {
            pub fn hash(_: @This(), v: Vertex) u64 {
                return v;
            }
            pub fn eql(_: @This(), a: Vertex, b: Vertex) bool {
                return a == b;
            }
        };
        const Adjacency = struct {
            edges: std.ArrayList(Edge),
        };
        adjacencies: std.HashMap(Vertex, Adjacency, Context, std.hash_map.default_max_load_percentage),

        fn init(alloc: Allocator) @This() {
            return .{
                .adjacencies = std.HashMap(Vertex, Adjacency, Context, std.hash_map.default_max_load_percentage).init(alloc),
            };
        }
        fn deinit(self: *@This()) void {
            var it = self.adjacencies.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.edges.deinit(self.adjacencies.allocator);
            }
            self.adjacencies.deinit();
        }
        fn addEdge(self: *@This(), from: Vertex, to: Vertex, capacity: u32) !void {
            const result = try self.adjacencies.getOrPut(from);
            if (!result.found_existing) {
                result.value_ptr.* = .{ .edges = .{} };
            }
            try result.value_ptr.edges.append(self.adjacencies.allocator, .{ .target = to, .capacity = capacity });
            const result_to = try self.adjacencies.getOrPut(to);
            if (!result_to.found_existing) {
                result_to.value_ptr.* = .{ .edges = .{} };
            }
        }
    };

    var graph = TestGraph.init(allocator);
    defer graph.deinit();

    try graph.addEdge(0, 1, 10);
    try graph.addEdge(0, 2, 10);
    try graph.addEdge(1, 5, 10);
    try graph.addEdge(1, 3, 10);
    try graph.addEdge(2, 3, 10);
    try graph.addEdge(3, 4, 10);
    try graph.addEdge(4, 5, 10);

    const PR = PushRelabel(TestGraph.Vertex, u32, TestGraph.Context);
    var result = try PR.run(allocator, graph, 0, 5, TestGraph.Context{}, 0);
    defer result.deinit();

    try testing.expectEqual(@as(u32, 20), result.max_flow);
}

test "PushRelabel: minimum cut" {
    const allocator = testing.allocator;

    const TestGraph = struct {
        const Vertex = u32;
        const Edge = PushRelabel(Vertex, u32, Context).Edge;
        const Context = struct {
            pub fn hash(_: @This(), v: Vertex) u64 {
                return v;
            }
            pub fn eql(_: @This(), a: Vertex, b: Vertex) bool {
                return a == b;
            }
        };
        const Adjacency = struct {
            edges: std.ArrayList(Edge),
        };
        adjacencies: std.HashMap(Vertex, Adjacency, Context, std.hash_map.default_max_load_percentage),

        fn init(alloc: Allocator) @This() {
            return .{
                .adjacencies = std.HashMap(Vertex, Adjacency, Context, std.hash_map.default_max_load_percentage).init(alloc),
            };
        }
        fn deinit(self: *@This()) void {
            var it = self.adjacencies.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.edges.deinit(self.adjacencies.allocator);
            }
            self.adjacencies.deinit();
        }
        fn addEdge(self: *@This(), from: Vertex, to: Vertex, capacity: u32) !void {
            const result = try self.adjacencies.getOrPut(from);
            if (!result.found_existing) {
                result.value_ptr.* = .{ .edges = .{} };
            }
            try result.value_ptr.edges.append(self.adjacencies.allocator, .{ .target = to, .capacity = capacity });
            const result_to = try self.adjacencies.getOrPut(to);
            if (!result_to.found_existing) {
                result_to.value_ptr.* = .{ .edges = .{} };
            }
        }
    };

    var graph = TestGraph.init(allocator);
    defer graph.deinit();

    try graph.addEdge(0, 1, 1);
    try graph.addEdge(1, 2, 100);

    const PR = PushRelabel(TestGraph.Vertex, u32, TestGraph.Context);
    var result = try PR.run(allocator, graph, 0, 2, TestGraph.Context{}, 0);
    defer result.deinit();

    try testing.expectEqual(@as(u32, 1), result.max_flow);
    try testing.expect(result.isInMinCut(0));
    try testing.expect(!result.isInMinCut(1));
    try testing.expect(!result.isInMinCut(2));
}
