const std = @import("std");
const testing = std.testing;

/// Tarjan's algorithm for finding bridges (cut edges) in an undirected graph.
///
/// A bridge is an edge whose removal increases the number of connected components.
/// Bridges are critical edges in network reliability analysis.
///
/// Time: O(V + E) — single DFS traversal
/// Space: O(V) — recursion stack and metadata arrays
///
/// Generic over:
/// - V: Vertex type (must be hashable and comparable)
///
/// Algorithm (DFS with low-link values):
/// 1. DFS traversal with discovery time tracking
/// 2. Track low-link values (lowest discovery time reachable via back edges)
/// 3. For each edge (u, v):
///    - If low[v] > discovery[u]: edge (u, v) is a bridge
///    - (No back edge from v's subtree reaches u or above)
///
/// Consumer use cases:
/// - Network reliability analysis (single point of failure detection)
/// - Circuit design (critical connections)
/// - Road network planning (bottleneck identification)
pub fn Bridges(comptime V: type) type {
    return struct {
        const Self = @This();

        pub const Edge = struct {
            u: V,
            v: V,

            pub fn eql(self: Edge, other: Edge) bool {
                return (std.meta.eql(self.u, other.u) and std.meta.eql(self.v, other.v)) or
                    (std.meta.eql(self.u, other.v) and std.meta.eql(self.v, other.u));
            }
        };

        pub const Result = struct {
            bridges: []Edge,

            pub fn deinit(self: *Result, allocator: std.mem.Allocator) void {
                allocator.free(self.bridges);
            }
        };

        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .allocator = allocator };
        }

        /// Find all bridges in an undirected graph.
        ///
        /// Time: O(V + E) — single DFS traversal
        /// Space: O(V) — recursion stack and metadata
        ///
        /// Parameters:
        /// - adjacency_list: Map from vertex to list of adjacent neighbors (undirected)
        ///
        /// Returns:
        /// - Result containing list of bridges (edges)
        ///
        /// Note: Graph must be undirected (if (u,v) exists, (v,u) must also exist)
        pub fn run(
            self: Self,
            adjacency_list: *const std.AutoHashMap(V, std.ArrayList(V)),
        ) !Result {
            if (adjacency_list.count() == 0) {
                return Result{ .bridges = try self.allocator.alloc(Edge, 0) };
            }

            var context = Context(V).init(self.allocator);
            defer context.deinit();

            // Run DFS from all unvisited vertices
            var vertex_iter = adjacency_list.keyIterator();
            while (vertex_iter.next()) |vertex| {
                if (!context.visited.contains(vertex.*)) {
                    try context.dfs(vertex.*, vertex.*, adjacency_list);
                }
            }

            return Result{
                .bridges = try context.bridges.toOwnedSlice(self.allocator),
            };
        }
    };
}

/// Internal context for bridge-finding algorithm
fn Context(comptime V: type) type {
    return struct {
        const CtxSelf = @This();

        allocator: std.mem.Allocator,
        visited: std.AutoHashMap(V, void),
        discovery: std.AutoHashMap(V, usize),
        low_link: std.AutoHashMap(V, usize),
        time: usize,
        bridges: std.ArrayList(Bridges(V).Edge),

        fn init(allocator: std.mem.Allocator) CtxSelf {
            return .{
                .allocator = allocator,
                .visited = std.AutoHashMap(V, void).init(allocator),
                .discovery = std.AutoHashMap(V, usize).init(allocator),
                .low_link = std.AutoHashMap(V, usize).init(allocator),
                .time = 0,
                .bridges = .{},
            };
        }

        fn deinit(self: *CtxSelf) void {
            self.visited.deinit();
            self.discovery.deinit();
            self.low_link.deinit();
            self.bridges.deinit(self.allocator);
        }

        fn dfs(
            self: *CtxSelf,
            u: V,
            parent: V,
            adjacency_list: *const std.AutoHashMap(V, std.ArrayList(V)),
        ) !void {
            try self.visited.put(u, {});
            self.time += 1;
            try self.discovery.put(u, self.time);
            try self.low_link.put(u, self.time);

            const neighbors = adjacency_list.get(u) orelse return;

            for (neighbors.items) |v| {
                if (!self.visited.contains(v)) {
                    // Tree edge: recurse on unvisited neighbor
                    try self.dfs(v, u, adjacency_list);

                    // Update low-link: can we reach an earlier vertex via v?
                    const v_low = self.low_link.get(v).?;
                    const u_low = self.low_link.get(u).?;
                    try self.low_link.put(u, @min(u_low, v_low));

                    // Bridge condition: v cannot reach u or above via back edges
                    const u_disc = self.discovery.get(u).?;
                    if (v_low > u_disc) {
                        try self.bridges.append(self.allocator, .{ .u = u, .v = v });
                    }
                } else if (!std.meta.eql(v, parent)) {
                    // Back edge (not parent): update low-link
                    const v_disc = self.discovery.get(v).?;
                    const u_low = self.low_link.get(u).?;
                    try self.low_link.put(u, @min(u_low, v_disc));
                }
                // Ignore edge to parent (avoids treating undirected edge as back edge)
            }
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "Bridges: empty graph" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();

    var finder = Bridges(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.bridges.len);
}

test "Bridges: single vertex" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();
    defer {
        var iter = graph.valueIterator();
        while (iter.next()) |list| list.deinit(allocator);
    }

    try graph.put(0, .{});

    var finder = Bridges(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.bridges.len);
}

test "Bridges: simple bridge (0-1)" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();
    defer {
        var iter = graph.valueIterator();
        while (iter.next()) |list| list.deinit(allocator);
    }

    // 0 -- 1 (simple bridge)
    var list0: std.ArrayList(u32) = .{};
    try list0.append(allocator, 1);
    try graph.put(0, list0);

    var list1: std.ArrayList(u32) = .{};
    try list1.append(allocator, 0);
    try graph.put(1, list1);

    var finder = Bridges(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), result.bridges.len);
    const bridge = result.bridges[0];
    try testing.expect(bridge.eql(.{ .u = 0, .v = 1 }) or bridge.eql(.{ .u = 1, .v = 0 }));
}

test "Bridges: triangle (no bridges)" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();
    defer {
        var iter = graph.valueIterator();
        while (iter.next()) |list| list.deinit(allocator);
    }

    // Triangle: 0 -- 1
    //           |    |
    //           2 ---+
    // No bridges (all edges are in a cycle)
    var list0: std.ArrayList(u32) = .{};
    try list0.append(allocator, 1);
    try list0.append(allocator, 2);
    try graph.put(0, list0);

    var list1: std.ArrayList(u32) = .{};
    try list1.append(allocator, 0);
    try list1.append(allocator, 2);
    try graph.put(1, list1);

    var list2: std.ArrayList(u32) = .{};
    try list2.append(allocator, 0);
    try list2.append(allocator, 1);
    try graph.put(2, list2);

    var finder = Bridges(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.bridges.len);
}

test "Bridges: square with diagonal (no bridges)" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();
    defer {
        var iter = graph.valueIterator();
        while (iter.next()) |list| list.deinit(allocator);
    }

    // 0 -- 1
    // | \  |
    // |  \ |
    // 3 -- 2
    // No bridges (diagonal provides alternate path)
    var list0: std.ArrayList(u32) = .{};
    try list0.append(allocator, 1);
    try list0.append(allocator, 2);
    try list0.append(allocator, 3);
    try graph.put(0, list0);

    var list1: std.ArrayList(u32) = .{};
    try list1.append(allocator, 0);
    try list1.append(allocator, 2);
    try graph.put(1, list1);

    var list2: std.ArrayList(u32) = .{};
    try list2.append(allocator, 0);
    try list2.append(allocator, 1);
    try list2.append(allocator, 3);
    try graph.put(2, list2);

    var list3: std.ArrayList(u32) = .{};
    try list3.append(allocator, 0);
    try list3.append(allocator, 2);
    try graph.put(3, list3);

    var finder = Bridges(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.bridges.len);
}

test "Bridges: chain (all edges are bridges)" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();
    defer {
        var iter = graph.valueIterator();
        while (iter.next()) |list| list.deinit(allocator);
    }

    // 0 -- 1 -- 2 -- 3
    // All edges are bridges
    var list0: std.ArrayList(u32) = .{};
    try list0.append(allocator, 1);
    try graph.put(0, list0);

    var list1: std.ArrayList(u32) = .{};
    try list1.append(allocator, 0);
    try list1.append(allocator, 2);
    try graph.put(1, list1);

    var list2: std.ArrayList(u32) = .{};
    try list2.append(allocator, 1);
    try list2.append(allocator, 3);
    try graph.put(2, list2);

    var list3: std.ArrayList(u32) = .{};
    try list3.append(allocator, 2);
    try graph.put(3, list3);

    var finder = Bridges(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 3), result.bridges.len);
}

test "Bridges: cycle with tail" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();
    defer {
        var iter = graph.valueIterator();
        while (iter.next()) |list| list.deinit(allocator);
    }

    //     0
    //     |
    //     1 -- 2
    //     |    |
    //     4 -- 3
    // Bridge: 0-1 (tail connecting to cycle)
    var list0: std.ArrayList(u32) = .{};
    try list0.append(allocator, 1);
    try graph.put(0, list0);

    var list1: std.ArrayList(u32) = .{};
    try list1.append(allocator, 0);
    try list1.append(allocator, 2);
    try list1.append(allocator, 4);
    try graph.put(1, list1);

    var list2: std.ArrayList(u32) = .{};
    try list2.append(allocator, 1);
    try list2.append(allocator, 3);
    try graph.put(2, list2);

    var list3: std.ArrayList(u32) = .{};
    try list3.append(allocator, 2);
    try list3.append(allocator, 4);
    try graph.put(3, list3);

    var list4: std.ArrayList(u32) = .{};
    try list4.append(allocator, 1);
    try list4.append(allocator, 3);
    try graph.put(4, list4);

    var finder = Bridges(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), result.bridges.len);
    const bridge = result.bridges[0];
    try testing.expect(bridge.eql(.{ .u = 0, .v = 1 }) or bridge.eql(.{ .u = 1, .v = 0 }));
}

test "Bridges: disconnected components" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();
    defer {
        var iter = graph.valueIterator();
        while (iter.next()) |list| list.deinit(allocator);
    }

    // Component 1: 0 -- 1 (bridge)
    // Component 2: 2 -- 3 -- 4 (all bridges)
    var list0: std.ArrayList(u32) = .{};
    try list0.append(allocator, 1);
    try graph.put(0, list0);

    var list1: std.ArrayList(u32) = .{};
    try list1.append(allocator, 0);
    try graph.put(1, list1);

    var list2: std.ArrayList(u32) = .{};
    try list2.append(allocator, 3);
    try graph.put(2, list2);

    var list3: std.ArrayList(u32) = .{};
    try list3.append(allocator, 2);
    try list3.append(allocator, 4);
    try graph.put(3, list3);

    var list4: std.ArrayList(u32) = .{};
    try list4.append(allocator, 3);
    try graph.put(4, list4);

    var finder = Bridges(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 3), result.bridges.len);
}

test "Bridges: complex network" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();
    defer {
        var iter = graph.valueIterator();
        while (iter.next()) |list| list.deinit(allocator);
    }

    // Complex:  0 -- 1 -- 2
    //           |    |    |
    //           3 -- 4    5 -- 6
    // Bridges: 1-2 (connects biconnected component to chain 2-5-6)
    //          2-5 (connects to chain)
    //          5-6 (chain edge)
    var list0: std.ArrayList(u32) = .{};
    try list0.append(allocator, 1);
    try list0.append(allocator, 3);
    try graph.put(0, list0);

    var list1: std.ArrayList(u32) = .{};
    try list1.append(allocator, 0);
    try list1.append(allocator, 2);
    try list1.append(allocator, 4);
    try graph.put(1, list1);

    var list2: std.ArrayList(u32) = .{};
    try list2.append(allocator, 1);
    try list2.append(allocator, 5);
    try graph.put(2, list2);

    var list3: std.ArrayList(u32) = .{};
    try list3.append(allocator, 0);
    try list3.append(allocator, 4);
    try graph.put(3, list3);

    var list4: std.ArrayList(u32) = .{};
    try list4.append(allocator, 1);
    try list4.append(allocator, 3);
    try graph.put(4, list4);

    var list5: std.ArrayList(u32) = .{};
    try list5.append(allocator, 2);
    try list5.append(allocator, 6);
    try graph.put(5, list5);

    var list6: std.ArrayList(u32) = .{};
    try list6.append(allocator, 5);
    try graph.put(6, list6);

    var finder = Bridges(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 3), result.bridges.len);
}

test "Bridges: self-loop" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();
    defer {
        var iter = graph.valueIterator();
        while (iter.next()) |list| list.deinit(allocator);
    }

    // 0 -- 1 (with self-loop on 1)
    // Self-loop is not a bridge
    var list0: std.ArrayList(u32) = .{};
    try list0.append(allocator, 1);
    try graph.put(0, list0);

    var list1: std.ArrayList(u32) = .{};
    try list1.append(allocator, 0);
    try list1.append(allocator, 1); // self-loop
    try graph.put(1, list1);

    var finder = Bridges(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    // 0-1 is a bridge (self-loop doesn't prevent it)
    try testing.expectEqual(@as(usize, 1), result.bridges.len);
}

test "Bridges: stress test (large chain)" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();
    defer {
        var iter = graph.valueIterator();
        while (iter.next()) |list| list.deinit(allocator);
    }

    // Chain: 0 -- 1 -- 2 -- ... -- 99
    const n = 100;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        var list: std.ArrayList(u32) = .{};
        if (i > 0) try list.append(allocator, i - 1);
        if (i < n - 1) try list.append(allocator, i + 1);
        try graph.put(i, list);
    }

    var finder = Bridges(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, n - 1), result.bridges.len);
}
