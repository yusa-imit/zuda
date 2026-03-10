const std = @import("std");
const testing = std.testing;

/// Tarjan's algorithm for finding articulation points (cut vertices) in an undirected graph.
///
/// An articulation point is a vertex whose removal increases the number of connected components.
/// Critical for network reliability analysis.
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
/// 3. For each vertex u:
///    - Root: is articulation point if it has ≥2 children in DFS tree
///    - Non-root: is articulation point if any child v satisfies low[v] ≥ discovery[u]
///      (v's subtree cannot reach u's ancestors without going through u)
///
/// Consumer use cases:
/// - Network reliability analysis (single point of failure detection)
/// - Social network analysis (key influencers/connectors)
/// - Transportation networks (critical hubs)
pub fn ArticulationPoints(comptime V: type) type {
    return struct {
        const Self = @This();

        pub const Result = struct {
            articulation_points: []V,

            pub fn deinit(self: *Result, allocator: std.mem.Allocator) void {
                allocator.free(self.articulation_points);
            }
        };

        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .allocator = allocator };
        }

        /// Find all articulation points in an undirected graph.
        ///
        /// Time: O(V + E) — single DFS traversal
        /// Space: O(V) — recursion stack and metadata
        ///
        /// Parameters:
        /// - adjacency_list: Map from vertex to list of adjacent neighbors (undirected)
        ///
        /// Returns:
        /// - Result containing list of articulation points (vertices)
        ///
        /// Note: Graph must be undirected (if (u,v) exists, (v,u) must also exist)
        pub fn run(
            self: Self,
            adjacency_list: *const std.AutoHashMap(V, std.ArrayList(V)),
        ) !Result {
            if (adjacency_list.count() == 0) {
                return Result{ .articulation_points = try self.allocator.alloc(V, 0) };
            }

            var context = Context(V).init(self.allocator);
            defer context.deinit();

            // Run DFS from all unvisited vertices
            var vertex_iter = adjacency_list.keyIterator();
            while (vertex_iter.next()) |vertex| {
                if (!context.visited.contains(vertex.*)) {
                    try context.dfs(vertex.*, vertex.*, adjacency_list, true);
                }
            }

            return Result{
                .articulation_points = try context.articulation_points.toOwnedSlice(self.allocator),
            };
        }
    };
}

/// Internal context for articulation point finding algorithm
fn Context(comptime V: type) type {
    return struct {
        const CtxSelf = @This();

        allocator: std.mem.Allocator,
        visited: std.AutoHashMap(V, void),
        discovery: std.AutoHashMap(V, usize),
        low_link: std.AutoHashMap(V, usize),
        time: usize,
        articulation_points: std.ArrayList(V),

        fn init(allocator: std.mem.Allocator) CtxSelf {
            return .{
                .allocator = allocator,
                .visited = std.AutoHashMap(V, void).init(allocator),
                .discovery = std.AutoHashMap(V, usize).init(allocator),
                .low_link = std.AutoHashMap(V, usize).init(allocator),
                .time = 0,
                .articulation_points = .{},
            };
        }

        fn deinit(self: *CtxSelf) void {
            self.visited.deinit();
            self.discovery.deinit();
            self.low_link.deinit();
            self.articulation_points.deinit(self.allocator);
        }

        fn dfs(
            self: *CtxSelf,
            u: V,
            parent: V,
            adjacency_list: *const std.AutoHashMap(V, std.ArrayList(V)),
            is_root: bool,
        ) !void {
            try self.visited.put(u, {});
            self.time += 1;
            try self.discovery.put(u, self.time);
            try self.low_link.put(u, self.time);

            const neighbors = adjacency_list.get(u) orelse return;

            var children: usize = 0;

            for (neighbors.items) |v| {
                if (!self.visited.contains(v)) {
                    // Tree edge: recurse on unvisited neighbor
                    children += 1;
                    try self.dfs(v, u, adjacency_list, false);

                    // Update low-link: can we reach an earlier vertex via v?
                    const v_low = self.low_link.get(v).?;
                    const u_low = self.low_link.get(u).?;
                    try self.low_link.put(u, @min(u_low, v_low));

                    // Articulation point condition (non-root):
                    // v cannot reach u or above via back edges
                    if (!is_root) {
                        const u_disc = self.discovery.get(u).?;
                        if (v_low >= u_disc) {
                            // u is an articulation point (removing u disconnects v's subtree)
                            // Check if already added (can be identified multiple times)
                            var already_added = false;
                            for (self.articulation_points.items) |ap| {
                                if (std.meta.eql(ap, u)) {
                                    already_added = true;
                                    break;
                                }
                            }
                            if (!already_added) {
                                try self.articulation_points.append(self.allocator, u);
                            }
                        }
                    }
                } else if (!std.meta.eql(v, parent)) {
                    // Back edge (not parent): update low-link
                    const v_disc = self.discovery.get(v).?;
                    const u_low = self.low_link.get(u).?;
                    try self.low_link.put(u, @min(u_low, v_disc));
                }
                // Ignore edge to parent (avoids treating undirected edge as back edge)
            }

            // Root articulation point condition: ≥2 children in DFS tree
            if (is_root and children >= 2) {
                try self.articulation_points.append(self.allocator, u);
            }
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "ArticulationPoints: empty graph" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();

    var finder = ArticulationPoints(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.articulation_points.len);
}

test "ArticulationPoints: single vertex" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();
    defer {
        var iter = graph.valueIterator();
        while (iter.next()) |list| list.deinit(allocator);
    }

    try graph.put(0, .{});

    var finder = ArticulationPoints(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.articulation_points.len);
}

test "ArticulationPoints: simple bridge (0-1-2)" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();
    defer {
        var iter = graph.valueIterator();
        while (iter.next()) |list| list.deinit(allocator);
    }

    // 0 -- 1 -- 2
    // Articulation point: 1 (removing 1 disconnects 0 and 2)
    var list0: std.ArrayList(u32) = .{};
    try list0.append(allocator, 1);
    try graph.put(0, list0);

    var list1: std.ArrayList(u32) = .{};
    try list1.append(allocator, 0);
    try list1.append(allocator, 2);
    try graph.put(1, list1);

    var list2: std.ArrayList(u32) = .{};
    try list2.append(allocator, 1);
    try graph.put(2, list2);

    var finder = ArticulationPoints(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), result.articulation_points.len);
    try testing.expectEqual(@as(u32, 1), result.articulation_points[0]);
}

test "ArticulationPoints: triangle (no articulation points)" {
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
    // No articulation points (all vertices have alternate paths)
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

    var finder = ArticulationPoints(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), result.articulation_points.len);
}

test "ArticulationPoints: star graph (center is articulation point)" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();
    defer {
        var iter = graph.valueIterator();
        while (iter.next()) |list| list.deinit(allocator);
    }

    //     1
    //     |
    // 2 - 0 - 3
    //     |
    //     4
    // Articulation point: 0 (center - removing it disconnects all leaves)
    var list0: std.ArrayList(u32) = .{};
    try list0.append(allocator, 1);
    try list0.append(allocator, 2);
    try list0.append(allocator, 3);
    try list0.append(allocator, 4);
    try graph.put(0, list0);

    var list1: std.ArrayList(u32) = .{};
    try list1.append(allocator, 0);
    try graph.put(1, list1);

    var list2: std.ArrayList(u32) = .{};
    try list2.append(allocator, 0);
    try graph.put(2, list2);

    var list3: std.ArrayList(u32) = .{};
    try list3.append(allocator, 0);
    try graph.put(3, list3);

    var list4: std.ArrayList(u32) = .{};
    try list4.append(allocator, 0);
    try graph.put(4, list4);

    var finder = ArticulationPoints(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), result.articulation_points.len);
    try testing.expectEqual(@as(u32, 0), result.articulation_points[0]);
}

test "ArticulationPoints: chain (all internal vertices are articulation points)" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();
    defer {
        var iter = graph.valueIterator();
        while (iter.next()) |list| list.deinit(allocator);
    }

    // 0 -- 1 -- 2 -- 3
    // Articulation points: 1, 2 (internal vertices)
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

    var finder = ArticulationPoints(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 2), result.articulation_points.len);
    // Order may vary, just check both are present
    var has_1 = false;
    var has_2 = false;
    for (result.articulation_points) |ap| {
        if (ap == 1) has_1 = true;
        if (ap == 2) has_2 = true;
    }
    try testing.expect(has_1 and has_2);
}

test "ArticulationPoints: cycle with tail" {
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
    // Articulation point: 1 (connects tail 0 to cycle)
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

    var finder = ArticulationPoints(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), result.articulation_points.len);
    try testing.expectEqual(@as(u32, 1), result.articulation_points[0]);
}

test "ArticulationPoints: two cycles connected by bridge" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();
    defer {
        var iter = graph.valueIterator();
        while (iter.next()) |list| list.deinit(allocator);
    }

    // Cycle 1: 0-1-2-0
    // Bridge: 2-3
    // Cycle 2: 3-4-5-3
    // Articulation points: 2, 3 (endpoints of bridge)
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
    try list2.append(allocator, 3);
    try graph.put(2, list2);

    var list3: std.ArrayList(u32) = .{};
    try list3.append(allocator, 2);
    try list3.append(allocator, 4);
    try list3.append(allocator, 5);
    try graph.put(3, list3);

    var list4: std.ArrayList(u32) = .{};
    try list4.append(allocator, 3);
    try list4.append(allocator, 5);
    try graph.put(4, list4);

    var list5: std.ArrayList(u32) = .{};
    try list5.append(allocator, 3);
    try list5.append(allocator, 4);
    try graph.put(5, list5);

    var finder = ArticulationPoints(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 2), result.articulation_points.len);
    // Order may vary, just check both are present
    var has_2 = false;
    var has_3 = false;
    for (result.articulation_points) |ap| {
        if (ap == 2) has_2 = true;
        if (ap == 3) has_3 = true;
    }
    try testing.expect(has_2 and has_3);
}

test "ArticulationPoints: disconnected components" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();
    defer {
        var iter = graph.valueIterator();
        while (iter.next()) |list| list.deinit(allocator);
    }

    // Component 1: 0 -- 1 (no articulation points)
    // Component 2: 2 -- 3 -- 4 (3 is articulation point)
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

    var finder = ArticulationPoints(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), result.articulation_points.len);
    try testing.expectEqual(@as(u32, 3), result.articulation_points[0]);
}

test "ArticulationPoints: complex network" {
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
    // Articulation points: 1 (separates {0,3,4} from {2,5,6}),
    //                      2 (separates biconnected component from tail 5-6),
    //                      5 (separates everything from 6)
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

    var finder = ArticulationPoints(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 3), result.articulation_points.len);
    // Order may vary, check 1, 2, and 5 are present
    var has_1 = false;
    var has_2 = false;
    var has_5 = false;
    for (result.articulation_points) |ap| {
        if (ap == 1) has_1 = true;
        if (ap == 2) has_2 = true;
        if (ap == 5) has_5 = true;
    }
    try testing.expect(has_1 and has_2 and has_5);
}

test "ArticulationPoints: self-loop" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();
    defer {
        var iter = graph.valueIterator();
        while (iter.next()) |list| list.deinit(allocator);
    }

    // 0 -- 1 -- 2 (with self-loop on 1)
    // Articulation point: 1 (self-loop doesn't change connectivity)
    var list0: std.ArrayList(u32) = .{};
    try list0.append(allocator, 1);
    try graph.put(0, list0);

    var list1: std.ArrayList(u32) = .{};
    try list1.append(allocator, 0);
    try list1.append(allocator, 1); // self-loop
    try list1.append(allocator, 2);
    try graph.put(1, list1);

    var list2: std.ArrayList(u32) = .{};
    try list2.append(allocator, 1);
    try graph.put(2, list2);

    var finder = ArticulationPoints(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    try testing.expectEqual(@as(usize, 1), result.articulation_points.len);
    try testing.expectEqual(@as(u32, 1), result.articulation_points[0]);
}

test "ArticulationPoints: stress test (large chain)" {
    const allocator = testing.allocator;
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer graph.deinit();
    defer {
        var iter = graph.valueIterator();
        while (iter.next()) |list| list.deinit(allocator);
    }

    // Chain: 0 -- 1 -- 2 -- ... -- 99
    // Articulation points: 1, 2, ..., 98 (all internal vertices)
    const n = 100;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        var list: std.ArrayList(u32) = .{};
        if (i > 0) try list.append(allocator, i - 1);
        if (i < n - 1) try list.append(allocator, i + 1);
        try graph.put(i, list);
    }

    var finder = ArticulationPoints(u32).init(allocator);
    var result = try finder.run(&graph);
    defer result.deinit(allocator);

    // Internal vertices: 1 to n-2 inclusive = n-2 articulation points
    try testing.expectEqual(@as(usize, n - 2), result.articulation_points.len);
}
