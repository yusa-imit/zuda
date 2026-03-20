const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// DFS - Depth-First Search
///
/// Traverses a graph in depth-first order, visiting vertices and exploring as far as possible
/// along each branch before backtracking. Provides discovery/finish times for advanced graph
/// algorithms (topological sort, SCC, cycle detection).
///
/// Time Complexity: O(V + E) where V is vertices, E is edges
/// Space Complexity: O(V) for visited set and recursion/stack (iterative uses explicit stack)
///
/// Generic parameters:
/// - V: Vertex type (must be hashable)
/// - Context: Context type for hashing/comparing vertices (must have .hash and .eql methods)
pub fn DFS(
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

        /// Vertex state during DFS traversal
        pub const State = enum {
            unvisited, // Not yet discovered
            visiting, // Currently on the stack (discovered but not finished)
            visited, // Finished (all descendants processed)
        };

        /// Edge classification in DFS tree
        pub const EdgeType = enum {
            tree, // Edge in the DFS tree (parent -> child)
            back, // Edge to an ancestor (indicates cycle in directed graphs)
            forward, // Edge to a descendant (not in DFS tree)
            cross, // Edge to neither ancestor nor descendant
        };

        /// DFS result containing traversal information
        pub const Result = struct {
            /// Discovery time for each vertex (when first visited)
            discovery: std.HashMap(V, usize, HashMapContext, std.hash_map.default_max_load_percentage),
            /// Finish time for each vertex (when all descendants processed)
            finish: std.HashMap(V, usize, HashMapContext, std.hash_map.default_max_load_percentage),
            /// Parent pointers for DFS tree
            parents: std.HashMap(V, V, HashMapContext, std.hash_map.default_max_load_percentage),
            /// Visit order (sequence of vertices in DFS discovery order)
            visit_order: std.ArrayList(V),
            /// Finish order (sequence of vertices in reverse topological order for DAGs)
            finish_order: std.ArrayList(V),
            /// Whether a cycle (back edge) was detected during traversal
            has_cycle: bool,
            allocator: Allocator,
            context: Context,

            pub fn deinit(self: *Result) void {
                self.discovery.deinit();
                self.finish.deinit();
                self.parents.deinit();
                self.visit_order.deinit(self.allocator);
                self.finish_order.deinit(self.allocator);
            }

            /// Get the discovery time of a vertex.
            /// Returns null if the vertex is not reachable.
            pub fn getDiscovery(self: *const Result, vertex: V) ?usize {
                return self.discovery.get(vertex);
            }

            /// Get the finish time of a vertex.
            /// Returns null if the vertex is not reachable.
            pub fn getFinish(self: *const Result, vertex: V) ?usize {
                return self.finish.get(vertex);
            }

            /// Get the parent of a vertex in the DFS tree.
            /// Returns null if the vertex has no parent (is the start vertex or unreachable).
            pub fn getParent(self: *const Result, vertex: V) ?V {
                return self.parents.get(vertex);
            }

            /// Reconstruct the path from start to target in the DFS tree.
            /// Returns null if target is not reachable from start.
            /// Caller owns the returned slice.
            pub fn getPath(self: *const Result, target: V) !?[]V {
                if (!self.discovery.contains(target)) {
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

            /// Check if there's a back edge (cycle) in the traversal.
            /// For directed graphs, presence of back edge indicates a cycle.
            /// Time: O(1) | Space: O(1)
            pub fn hasCycle(self: *const Result) bool {
                return self.has_cycle;
            }
        };

        /// Stack frame for iterative DFS
        const Frame = struct {
            vertex: V,
            neighbors: []const Edge,
            neighbor_idx: usize,

            const Edge = struct {
                target: V,
            };
        };

        /// Run DFS from a start vertex on a graph (iterative version).
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

            var discovery = std.HashMap(V, usize, HashMapContext, std.hash_map.default_max_load_percentage).initContext(allocator, hm_ctx);
            errdefer discovery.deinit();

            var finish = std.HashMap(V, usize, HashMapContext, std.hash_map.default_max_load_percentage).initContext(allocator, hm_ctx);
            errdefer finish.deinit();

            var parents = std.HashMap(V, V, HashMapContext, std.hash_map.default_max_load_percentage).initContext(allocator, hm_ctx);
            errdefer parents.deinit();

            var visit_order: std.ArrayList(V) = .{};
            errdefer visit_order.deinit(allocator);

            var finish_order: std.ArrayList(V) = .{};
            errdefer finish_order.deinit(allocator);

            var states = std.HashMap(V, State, HashMapContext, std.hash_map.default_max_load_percentage).initContext(allocator, hm_ctx);
            defer states.deinit();

            var time: usize = 0;
            var has_cycle = false;

            // Explicit stack for iterative DFS
            const StackFrame = struct { v: V, first_visit: bool };
            var stack: std.ArrayList(StackFrame) = .{};
            defer stack.deinit(allocator);

            try stack.append(allocator, .{ .v = start, .first_visit = true });

            while (stack.items.len > 0) {
                const frame = stack.items[stack.items.len - 1];
                stack.items.len -= 1; // Manual pop
                const current = frame.v;

                if (frame.first_visit) {
                    // First time visiting this vertex
                    const state = states.get(current) orelse State.unvisited;
                    if (state != State.unvisited) continue;

                    // Mark as visiting and record discovery time
                    try states.put(current, State.visiting);
                    time += 1;
                    try discovery.put(current, time);
                    try visit_order.append(allocator, current);

                    // Push finish marker
                    try stack.append(allocator, .{ .v = current, .first_visit = false });

                    // Push all unvisited neighbors
                    const neighbors = graph.getNeighbors(current) orelse continue;
                    var i: usize = neighbors.len;
                    while (i > 0) {
                        i -= 1;
                        const neighbor = neighbors[i].target;
                        const neighbor_state = states.get(neighbor) orelse State.unvisited;

                        if (neighbor_state == State.unvisited) {
                            try parents.put(neighbor, current);
                            try stack.append(allocator, .{ .v = neighbor, .first_visit = true });
                        } else if (neighbor_state == State.visiting) {
                            // Back edge detected (cycle in directed graph)
                            has_cycle = true;
                        }
                    }
                } else {
                    // Finish visiting this vertex
                    try states.put(current, State.visited);
                    time += 1;
                    try finish.put(current, time);
                    try finish_order.append(allocator, current);
                }
            }

            return Result{
                .discovery = discovery,
                .finish = finish,
                .parents = parents,
                .visit_order = visit_order,
                .finish_order = finish_order,
                .has_cycle = has_cycle,
                .allocator = allocator,
                .context = context,
            };
        }

        /// Run DFS from start until goal is found (early termination).
        /// Returns the result with partial traversal information.
        /// If goal is not reachable, traverses the entire reachable component.
        ///
        /// Time: O(V + E) worst case, O(path length) best case | Space: O(V)
        pub fn runToGoal(
            allocator: Allocator,
            graph: anytype,
            start: V,
            goal: V,
            context: Context,
        ) !Result {
            const hm_ctx = HashMapContext{ .user_ctx = context };

            var discovery = std.HashMap(V, usize, HashMapContext, std.hash_map.default_max_load_percentage).initContext(allocator, hm_ctx);
            errdefer discovery.deinit();

            var finish = std.HashMap(V, usize, HashMapContext, std.hash_map.default_max_load_percentage).initContext(allocator, hm_ctx);
            errdefer finish.deinit();

            var parents = std.HashMap(V, V, HashMapContext, std.hash_map.default_max_load_percentage).initContext(allocator, hm_ctx);
            errdefer parents.deinit();

            var visit_order: std.ArrayList(V) = .{};
            errdefer visit_order.deinit(allocator);

            var finish_order: std.ArrayList(V) = .{};
            errdefer finish_order.deinit(allocator);

            var states = std.HashMap(V, State, HashMapContext, std.hash_map.default_max_load_percentage).initContext(allocator, hm_ctx);
            defer states.deinit();

            var time: usize = 0;
            var has_cycle = false;

            // Early exit if start == goal
            if (context.eql(start, goal)) {
                time += 1;
                try discovery.put(start, time);
                try visit_order.append(allocator, start);
                time += 1;
                try finish.put(start, time);
                try finish_order.append(allocator, start);

                return Result{
                    .discovery = discovery,
                    .finish = finish,
                    .parents = parents,
                    .visit_order = visit_order,
                    .finish_order = finish_order,
                    .has_cycle = false,
                    .allocator = allocator,
                    .context = context,
                };
            }

            // Explicit stack for iterative DFS
            const StackFrame = struct { v: V, first_visit: bool };
            var stack: std.ArrayList(StackFrame) = .{};
            defer stack.deinit(allocator);

            try stack.append(allocator, .{ .v = start, .first_visit = true });

            var found_goal = false;

            while (stack.items.len > 0) {
                const frame = stack.items[stack.items.len - 1];
                stack.items.len -= 1; // Manual pop
                const current = frame.v;

                if (frame.first_visit) {
                    const state = states.get(current) orelse State.unvisited;
                    if (state != State.unvisited) continue;

                    try states.put(current, State.visiting);
                    time += 1;
                    try discovery.put(current, time);
                    try visit_order.append(allocator, current);

                    // Check if we found the goal
                    if (context.eql(current, goal)) {
                        found_goal = true;
                        // Mark as finished immediately
                        try states.put(current, State.visited);
                        time += 1;
                        try finish.put(current, time);
                        try finish_order.append(allocator, current);
                        break;
                    }

                    try stack.append(allocator, .{ .v = current, .first_visit = false });

                    const neighbors = graph.getNeighbors(current) orelse continue;
                    var i: usize = neighbors.len;
                    while (i > 0) {
                        i -= 1;
                        const neighbor = neighbors[i].target;
                        const neighbor_state = states.get(neighbor) orelse State.unvisited;

                        if (neighbor_state == State.unvisited) {
                            try parents.put(neighbor, current);
                            try stack.append(allocator, .{ .v = neighbor, .first_visit = true });
                        } else if (neighbor_state == State.visiting) {
                            // Back edge detected (cycle in directed graph)
                            has_cycle = true;
                        }
                    }
                } else {
                    try states.put(current, State.visited);
                    time += 1;
                    try finish.put(current, time);
                    try finish_order.append(allocator, current);
                }
            }

            // If goal was found, finish any remaining vertices on the stack
            if (found_goal) {
                while (stack.items.len > 0) {
                    const frame = stack.items[stack.items.len - 1];
                    stack.items.len -= 1; // Manual pop
                    if (!frame.first_visit) {
                        const state = states.get(frame.v) orelse State.unvisited;
                        if (state == State.visiting) {
                            try states.put(frame.v, State.visited);
                            time += 1;
                            try finish.put(frame.v, time);
                            try finish_order.append(allocator, frame.v);
                        }
                    }
                }
            }

            return Result{
                .discovery = discovery,
                .finish = finish,
                .parents = parents,
                .visit_order = visit_order,
                .finish_order = finish_order,
                .has_cycle = has_cycle,
                .allocator = allocator,
                .context = context,
            };
        }

        /// Run DFS on all vertices in the graph (for disconnected graphs).
        /// Performs DFS from each unvisited vertex, producing a DFS forest.
        ///
        /// Graph type must provide:
        /// - `getNeighbors(vertex: V) -> ?[]const Edge` where Edge has `.target: V`
        /// - `getAllVertices() -> []const V`
        ///
        /// Time: O(V + E) | Space: O(V)
        pub fn runAll(
            allocator: Allocator,
            graph: anytype,
            context: Context,
        ) !Result {
            const hm_ctx = HashMapContext{ .user_ctx = context };

            var discovery = std.HashMap(V, usize, HashMapContext, std.hash_map.default_max_load_percentage).initContext(allocator, hm_ctx);
            errdefer discovery.deinit();

            var finish = std.HashMap(V, usize, HashMapContext, std.hash_map.default_max_load_percentage).initContext(allocator, hm_ctx);
            errdefer finish.deinit();

            var parents = std.HashMap(V, V, HashMapContext, std.hash_map.default_max_load_percentage).initContext(allocator, hm_ctx);
            errdefer parents.deinit();

            var visit_order: std.ArrayList(V) = .{};
            errdefer visit_order.deinit(allocator);

            var finish_order: std.ArrayList(V) = .{};
            errdefer finish_order.deinit(allocator);

            var states = std.HashMap(V, State, HashMapContext, std.hash_map.default_max_load_percentage).initContext(allocator, hm_ctx);
            defer states.deinit();

            var time: usize = 0;
            var has_cycle = false;

            const vertices = graph.getAllVertices();

            for (vertices) |start| {
                const state = states.get(start) orelse State.unvisited;
                if (state != State.unvisited) continue;

                const StackFrame = struct { v: V, first_visit: bool };
                var stack: std.ArrayList(StackFrame) = .{};
                defer stack.deinit(allocator);

                try stack.append(allocator, .{ .v = start, .first_visit = true });

                while (stack.items.len > 0) {
                    const frame = stack.items[stack.items.len - 1];
                    stack.items.len -= 1; // Manual pop
                    const current = frame.v;

                    if (frame.first_visit) {
                        const curr_state = states.get(current) orelse State.unvisited;
                        if (curr_state != State.unvisited) continue;

                        try states.put(current, State.visiting);
                        time += 1;
                        try discovery.put(current, time);
                        try visit_order.append(allocator, current);

                        try stack.append(allocator, .{ .v = current, .first_visit = false });

                        const neighbors = graph.getNeighbors(current) orelse continue;
                        var i: usize = neighbors.len;
                        while (i > 0) {
                            i -= 1;
                            const neighbor = neighbors[i].target;
                            const neighbor_state = states.get(neighbor) orelse State.unvisited;

                            if (neighbor_state == State.unvisited) {
                                try parents.put(neighbor, current);
                                try stack.append(allocator, .{ .v = neighbor, .first_visit = true });
                            } else if (neighbor_state == State.visiting) {
                                // Back edge detected (cycle in directed graph)
                                has_cycle = true;
                            }
                        }
                    } else {
                        try states.put(current, State.visited);
                        time += 1;
                        try finish.put(current, time);
                        try finish_order.append(allocator, current);
                    }
                }
            }

            return Result{
                .discovery = discovery,
                .finish = finish,
                .parents = parents,
                .visit_order = visit_order,
                .finish_order = finish_order,
                .has_cycle = has_cycle,
                .allocator = allocator,
                .context = context,
            };
        }
    };
}

// ============================================================================
// TESTS
// ============================================================================

test "DFS: basic traversal" {
    const TestGraph = struct {
        edges: std.AutoHashMap(u32, std.ArrayList(u32)),

        pub fn getNeighbors(self: *const @This(), vertex: u32) ?[]const Edge {
            const list = self.edges.get(vertex) orelse return null;
            // Cast []u32 to []Edge (same layout since Edge is just {target: u32})
            return @ptrCast(list.items);
        }

        const Edge = struct { target: u32 };
    };

    const TestContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    var graph: TestGraph = .{ .edges = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator) };
    defer graph.edges.deinit();
    defer {
        var it = graph.edges.valueIterator();
        while (it.next()) |list| list.deinit(testing.allocator);
    }

    // Graph: 1 -> 2 -> 3
    //        |         ^
    //        +-> 4 ----+
    var list1 = std.ArrayList(u32){};
    try list1.append(testing.allocator, 2);
    try list1.append(testing.allocator, 4);
    try graph.edges.put(1, list1);

    var list2 = std.ArrayList(u32){};
    try list2.append(testing.allocator, 3);
    try graph.edges.put(2, list2);

    var list4 = std.ArrayList(u32){};
    try list4.append(testing.allocator, 3);
    try graph.edges.put(4, list4);

    const dfs_impl = DFS(u32, TestContext);
    var result = try dfs_impl.run(testing.allocator, &graph, 1, .{});
    defer result.deinit();

    // All vertices should be reachable
    try testing.expect(result.getDiscovery(1) != null);
    try testing.expect(result.getDiscovery(2) != null);
    try testing.expect(result.getDiscovery(3) != null);
    try testing.expect(result.getDiscovery(4) != null);

    // Discovery time of start should be 1
    try testing.expectEqual(@as(usize, 1), result.getDiscovery(1).?);

    // Parent of 1 should be null (root)
    try testing.expectEqual(@as(?u32, null), result.getParent(1));

    // Visit order should contain all vertices
    try testing.expectEqual(@as(usize, 4), result.visit_order.items.len);
    try testing.expectEqual(@as(usize, 4), result.finish_order.items.len);
}

test "DFS: path reconstruction" {
    const TestGraph = struct {
        edges: std.AutoHashMap(u32, std.ArrayList(u32)),

        pub fn getNeighbors(self: *const @This(), vertex: u32) ?[]const Edge {
            const list = self.edges.get(vertex) orelse return null;
            return @ptrCast(list.items);
        }

        const Edge = struct { target: u32 };
    };

    const TestContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    var graph: TestGraph = .{ .edges = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator) };
    defer graph.edges.deinit();
    defer {
        var it = graph.edges.valueIterator();
        while (it.next()) |list| list.deinit(testing.allocator);
    }

    // Linear chain: 1 -> 2 -> 3 -> 4
    var list1 = std.ArrayList(u32){};
    try list1.append(testing.allocator, 2);
    try graph.edges.put(1, list1);

    var list2 = std.ArrayList(u32){};
    try list2.append(testing.allocator, 3);
    try graph.edges.put(2, list2);

    var list3 = std.ArrayList(u32){};
    try list3.append(testing.allocator, 4);
    try graph.edges.put(3, list3);

    const dfs_impl = DFS(u32, TestContext);
    var result = try dfs_impl.run(testing.allocator, &graph, 1, .{});
    defer result.deinit();

    const path = (try result.getPath(4)).?;
    defer testing.allocator.free(path);

    try testing.expectEqual(@as(usize, 4), path.len);
    try testing.expectEqual(@as(u32, 1), path[0]);
    try testing.expectEqual(@as(u32, 2), path[1]);
    try testing.expectEqual(@as(u32, 3), path[2]);
    try testing.expectEqual(@as(u32, 4), path[3]);
}

test "DFS: disconnected graph" {
    const TestGraph = struct {
        edges: std.AutoHashMap(u32, std.ArrayList(u32)),

        pub fn getNeighbors(self: *const @This(), vertex: u32) ?[]const Edge {
            const list = self.edges.get(vertex) orelse return null;
            return @ptrCast(list.items);
        }

        const Edge = struct { target: u32 };
    };

    const TestContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    var graph: TestGraph = .{ .edges = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator) };
    defer graph.edges.deinit();
    defer {
        var it = graph.edges.valueIterator();
        while (it.next()) |list| list.deinit(testing.allocator);
    }

    // Two disconnected components: 1 -> 2 and 3 -> 4
    var list1 = std.ArrayList(u32){};
    try list1.append(testing.allocator, 2);
    try graph.edges.put(1, list1);

    var list3 = std.ArrayList(u32){};
    try list3.append(testing.allocator, 4);
    try graph.edges.put(3, list3);

    const dfs_impl = DFS(u32, TestContext);
    var result = try dfs_impl.run(testing.allocator, &graph, 1, .{});
    defer result.deinit();

    // Only vertices 1 and 2 should be reachable from 1
    try testing.expect(result.getDiscovery(1) != null);
    try testing.expect(result.getDiscovery(2) != null);
    try testing.expectEqual(@as(?usize, null), result.getDiscovery(3));
    try testing.expectEqual(@as(?usize, null), result.getDiscovery(4));

    try testing.expectEqual(@as(usize, 2), result.visit_order.items.len);
}

test "DFS: early termination (runToGoal)" {
    const TestGraph = struct {
        edges: std.AutoHashMap(u32, std.ArrayList(u32)),

        pub fn getNeighbors(self: *const @This(), vertex: u32) ?[]const Edge {
            const list = self.edges.get(vertex) orelse return null;
            return @ptrCast(list.items);
        }

        const Edge = struct { target: u32 };
    };

    const TestContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    var graph: TestGraph = .{ .edges = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator) };
    defer graph.edges.deinit();
    defer {
        var it = graph.edges.valueIterator();
        while (it.next()) |list| list.deinit(testing.allocator);
    }

    // Graph: 1 -> 2 -> 3 -> 4 -> 5
    var list1 = std.ArrayList(u32){};
    try list1.append(testing.allocator, 2);
    try graph.edges.put(1, list1);

    var list2 = std.ArrayList(u32){};
    try list2.append(testing.allocator, 3);
    try graph.edges.put(2, list2);

    var list3 = std.ArrayList(u32){};
    try list3.append(testing.allocator, 4);
    try graph.edges.put(3, list3);

    var list4 = std.ArrayList(u32){};
    try list4.append(testing.allocator, 5);
    try graph.edges.put(4, list4);

    const dfs_impl = DFS(u32, TestContext);
    var result = try dfs_impl.runToGoal(testing.allocator, &graph, 1, 3, .{});
    defer result.deinit();

    // Should find path to 3
    try testing.expect(result.getDiscovery(3) != null);

    const path = (try result.getPath(3)).?;
    defer testing.allocator.free(path);

    try testing.expectEqual(@as(usize, 3), path.len);
    try testing.expectEqual(@as(u32, 1), path[0]);
    try testing.expectEqual(@as(u32, 2), path[1]);
    try testing.expectEqual(@as(u32, 3), path[2]);
}

test "DFS: cycle detection (back edge)" {
    const TestGraph = struct {
        edges: std.AutoHashMap(u32, std.ArrayList(u32)),

        pub fn getNeighbors(self: *const @This(), vertex: u32) ?[]const Edge {
            const list = self.edges.get(vertex) orelse return null;
            return @ptrCast(list.items);
        }

        const Edge = struct { target: u32 };
    };

    const TestContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    var graph: TestGraph = .{ .edges = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator) };
    defer graph.edges.deinit();
    defer {
        var it = graph.edges.valueIterator();
        while (it.next()) |list| list.deinit(testing.allocator);
    }

    // Graph with cycle: 1 -> 2 -> 3 -> 1
    var list1 = std.ArrayList(u32){};
    try list1.append(testing.allocator, 2);
    try graph.edges.put(1, list1);

    var list2 = std.ArrayList(u32){};
    try list2.append(testing.allocator, 3);
    try graph.edges.put(2, list2);

    var list3 = std.ArrayList(u32){};
    try list3.append(testing.allocator, 1);
    try graph.edges.put(3, list3);

    const dfs_impl = DFS(u32, TestContext);
    var result = try dfs_impl.run(testing.allocator, &graph, 1, .{});
    defer result.deinit();

    // All vertices should be visited
    try testing.expect(result.getDiscovery(1) != null);
    try testing.expect(result.getDiscovery(2) != null);
    try testing.expect(result.getDiscovery(3) != null);

    // Verify discovery times are before finish times
    try testing.expect(result.getDiscovery(1).? < result.getFinish(1).?);
    try testing.expect(result.getDiscovery(2).? < result.getFinish(2).?);
    try testing.expect(result.getDiscovery(3).? < result.getFinish(3).?);

    // Verify cycle was detected
    try testing.expect(result.hasCycle());
}

test "DFS: acyclic graph (DAG)" {
    const TestGraph = struct {
        edges: std.AutoHashMap(u32, std.ArrayList(u32)),

        pub fn getNeighbors(self: *const @This(), vertex: u32) ?[]const Edge {
            const list = self.edges.get(vertex) orelse return null;
            return @ptrCast(list.items);
        }

        const Edge = struct { target: u32 };
    };

    const TestContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    var graph: TestGraph = .{ .edges = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator) };
    defer graph.edges.deinit();
    defer {
        var it = graph.edges.valueIterator();
        while (it.next()) |list| list.deinit(testing.allocator);
    }

    // DAG: 1 -> 2 -> 4
    //      |    |
    //      v    v
    //      3 -> 4
    var list1 = std.ArrayList(u32){};
    try list1.append(testing.allocator, 2);
    try list1.append(testing.allocator, 3);
    try graph.edges.put(1, list1);

    var list2 = std.ArrayList(u32){};
    try list2.append(testing.allocator, 4);
    try graph.edges.put(2, list2);

    var list3 = std.ArrayList(u32){};
    try list3.append(testing.allocator, 4);
    try graph.edges.put(3, list3);

    const dfs_impl = DFS(u32, TestContext);
    var result = try dfs_impl.run(testing.allocator, &graph, 1, .{});
    defer result.deinit();

    // No cycle should be detected in DAG
    try testing.expect(!result.hasCycle());
}

test "DFS: single vertex" {
    const TestGraph = struct {
        edges: std.AutoHashMap(u32, std.ArrayList(u32)),

        pub fn getNeighbors(self: *const @This(), vertex: u32) ?[]const Edge {
            const list = self.edges.get(vertex) orelse return null;
            return @ptrCast(list.items);
        }

        const Edge = struct { target: u32 };
    };

    const TestContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    var graph: TestGraph = .{ .edges = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator) };
    defer graph.edges.deinit();
    defer {
        var it = graph.edges.valueIterator();
        while (it.next()) |list| list.deinit(testing.allocator);
    }

    const list1 = std.ArrayList(u32){};
    try graph.edges.put(1, list1);

    const dfs_impl = DFS(u32, TestContext);
    var result = try dfs_impl.run(testing.allocator, &graph, 1, .{});
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.visit_order.items.len);
    try testing.expectEqual(@as(u32, 1), result.visit_order.items[0]);
    try testing.expectEqual(@as(usize, 1), result.getDiscovery(1).?);
    try testing.expectEqual(@as(usize, 2), result.getFinish(1).?);
}

test "DFS: runAll on disconnected graph" {
    const TestGraph = struct {
        edges: std.AutoHashMap(u32, std.ArrayList(u32)),
        vertices: []const u32,

        pub fn getNeighbors(self: *const @This(), vertex: u32) ?[]const Edge {
            const list = self.edges.get(vertex) orelse return null;
            return @ptrCast(list.items);
        }

        pub fn getAllVertices(self: *const @This()) []const u32 {
            return self.vertices;
        }

        const Edge = struct { target: u32 };
    };

    const TestContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    var graph: TestGraph = .{
        .edges = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator),
        .vertices = &[_]u32{ 1, 2, 3, 4 },
    };
    defer graph.edges.deinit();
    defer {
        var it = graph.edges.valueIterator();
        while (it.next()) |list| list.deinit(testing.allocator);
    }

    // Two components: 1 -> 2 and 3 -> 4
    var list1 = std.ArrayList(u32){};
    try list1.append(testing.allocator, 2);
    try graph.edges.put(1, list1);

    var list3 = std.ArrayList(u32){};
    try list3.append(testing.allocator, 4);
    try graph.edges.put(3, list3);

    const dfs_impl = DFS(u32, TestContext);
    var result = try dfs_impl.runAll(testing.allocator, &graph, .{});
    defer result.deinit();

    // All vertices should be visited
    try testing.expect(result.getDiscovery(1) != null);
    try testing.expect(result.getDiscovery(2) != null);
    try testing.expect(result.getDiscovery(3) != null);
    try testing.expect(result.getDiscovery(4) != null);

    try testing.expectEqual(@as(usize, 4), result.visit_order.items.len);
}

test "DFS: stress test (large chain)" {
    const TestGraph = struct {
        edges: std.AutoHashMap(u32, std.ArrayList(u32)),

        pub fn getNeighbors(self: *const @This(), vertex: u32) ?[]const Edge {
            const list = self.edges.get(vertex) orelse return null;
            return @ptrCast(list.items);
        }

        const Edge = struct { target: u32 };
    };

    const TestContext = struct {
        pub fn hash(_: @This(), key: u32) u64 {
            return key;
        }
        pub fn eql(_: @This(), a: u32, b: u32) bool {
            return a == b;
        }
    };

    var graph: TestGraph = .{ .edges = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator) };
    defer graph.edges.deinit();
    defer {
        var it = graph.edges.valueIterator();
        while (it.next()) |list| list.deinit(testing.allocator);
    }

    // Build a chain: 0 -> 1 -> 2 -> ... -> 999
    const chain_length = 1000;
    var i: u32 = 0;
    while (i < chain_length - 1) : (i += 1) {
        var list = std.ArrayList(u32){};
        try list.append(testing.allocator, i + 1);
        try graph.edges.put(i, list);
    }

    const dfs_impl = DFS(u32, TestContext);
    var result = try dfs_impl.run(testing.allocator, &graph, 0, .{});
    defer result.deinit();

    try testing.expectEqual(@as(usize, chain_length), result.visit_order.items.len);
    try testing.expectEqual(@as(usize, 1), result.getDiscovery(0).?);
    try testing.expect(result.getDiscovery(chain_length - 1) != null);

    const path = (try result.getPath(chain_length - 1)).?;
    defer testing.allocator.free(path);

    try testing.expectEqual(@as(usize, chain_length), path.len);
    try testing.expectEqual(@as(u32, 0), path[0]);
    try testing.expectEqual(@as(u32, chain_length - 1), path[chain_length - 1]);
}
