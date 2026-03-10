const std = @import("std");
const testing = std.testing;

/// Kosaraju's algorithm for finding Strongly Connected Components (SCCs) in a directed graph.
///
/// A strongly connected component is a maximal set of vertices where every vertex is
/// reachable from every other vertex in the set.
///
/// Time: O(V + E) — two DFS passes (original + transposed graph)
/// Space: O(V + E) — transposed graph storage
///
/// Generic over:
/// - V: Vertex type (must be hashable and comparable)
///
/// Algorithm (two-pass DFS):
/// 1. First DFS on original graph: compute finish times for all vertices
/// 2. Create transposed graph (reverse all edges)
/// 3. Second DFS on transposed graph: visit vertices in decreasing finish time order
/// 4. Each DFS tree in step 3 is one SCC
///
/// Difference from Tarjan:
/// - Kosaraju: Two DFS passes on original and transposed graph (offline algorithm)
/// - Tarjan: Single DFS pass with stack (online algorithm)
///
/// Both have O(V + E) time. Kosaraju is conceptually simpler and easier to parallelize.
pub fn KosarajuSCC(comptime V: type) type {
    return struct {
        const Self = @This();

        pub const Component = []V;

        pub const Result = struct {
            components: []Component,

            pub fn deinit(self: *Result, allocator: std.mem.Allocator) void {
                for (self.components) |component| {
                    allocator.free(component);
                }
                allocator.free(self.components);
            }
        };

        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .allocator = allocator };
        }

        /// Find all strongly connected components in a directed graph.
        ///
        /// Time: O(V + E) — two DFS passes
        /// Space: O(V + E) — transposed graph storage
        ///
        /// Parameters:
        /// - adjacency_list: Map from vertex to list of outgoing neighbors
        ///
        /// Returns:
        /// - Result containing list of SCCs (each SCC is a list of vertices)
        ///
        /// Note: Components are in topological order of the condensation graph.
        pub fn run(
            self: Self,
            adjacency_list: *const std.AutoHashMap(V, std.ArrayList(V)),
        ) !Result {
            if (adjacency_list.count() == 0) {
                return Result{ .components = try self.allocator.alloc(Component, 0) };
            }

            // Phase 1: First DFS to compute finish times
            var visited = std.AutoHashMap(V, void).init(self.allocator);
            defer visited.deinit();

            var finish_stack: std.ArrayList(V) = .{};
            defer finish_stack.deinit(self.allocator);

            var vertex_iter = adjacency_list.keyIterator();
            while (vertex_iter.next()) |vertex| {
                if (!visited.contains(vertex.*)) {
                    try self.dfsFinish(vertex.*, adjacency_list, &visited, &finish_stack);
                }
            }

            // Phase 2: Create transposed graph (reverse all edges)
            var transposed = std.AutoHashMap(V, std.ArrayList(V)).init(self.allocator);
            defer {
                var iter = transposed.valueIterator();
                while (iter.next()) |list| {
                    list.deinit(self.allocator);
                }
                transposed.deinit();
            }

            // Initialize empty adjacency lists for all vertices
            vertex_iter = adjacency_list.keyIterator();
            while (vertex_iter.next()) |vertex| {
                const empty_list: std.ArrayList(V) = .{};
                try transposed.put(vertex.*, empty_list);
            }

            // Add reversed edges
            var edge_iter = adjacency_list.iterator();
            while (edge_iter.next()) |entry| {
                const from = entry.key_ptr.*;
                const neighbors = entry.value_ptr.*;

                for (neighbors.items) |to| {
                    // Ensure vertex exists in transposed graph
                    if (!transposed.contains(to)) {
                        const new_list: std.ArrayList(V) = .{};
                        try transposed.put(to, new_list);
                    }
                    var to_list = transposed.getPtr(to).?;
                    try to_list.append(self.allocator, from);
                }
            }

            // Phase 3: Second DFS on transposed graph in reverse finish order
            visited.clearRetainingCapacity();

            var components: std.ArrayList(Component) = .{};
            defer {
                if (components.items.len > 0) {
                    for (components.items) |component| {
                        self.allocator.free(component);
                    }
                }
                components.deinit(self.allocator);
            }

            // Process vertices in decreasing finish time (reverse stack order)
            var i: usize = finish_stack.items.len;
            while (i > 0) {
                i -= 1;
                const vertex = finish_stack.items[i];

                if (!visited.contains(vertex)) {
                    var component: std.ArrayList(V) = .{};
                    errdefer component.deinit(self.allocator);

                    try self.dfsCollect(vertex, &transposed, &visited, &component);
                    try components.append(self.allocator, try component.toOwnedSlice(self.allocator));
                }
            }

            return Result{
                .components = try components.toOwnedSlice(self.allocator),
            };
        }

        /// First DFS pass: compute finish times (post-order traversal)
        fn dfsFinish(
            self: Self,
            v: V,
            adjacency_list: *const std.AutoHashMap(V, std.ArrayList(V)),
            visited: *std.AutoHashMap(V, void),
            finish_stack: *std.ArrayList(V),
        ) !void {
            try visited.put(v, {});

            if (adjacency_list.get(v)) |neighbors| {
                for (neighbors.items) |w| {
                    if (!visited.contains(w)) {
                        try self.dfsFinish(w, adjacency_list, visited, finish_stack);
                    }
                }
            }

            // Push to finish stack after visiting all descendants
            try finish_stack.append(self.allocator, v);
        }

        /// Second DFS pass: collect vertices in the same SCC
        fn dfsCollect(
            self: Self,
            v: V,
            transposed: *std.AutoHashMap(V, std.ArrayList(V)),
            visited: *std.AutoHashMap(V, void),
            component: *std.ArrayList(V),
        ) !void {
            try visited.put(v, {});
            try component.append(self.allocator, v);

            if (transposed.get(v)) |neighbors| {
                for (neighbors.items) |w| {
                    if (!visited.contains(w)) {
                        try self.dfsCollect(w, transposed, visited, component);
                    }
                }
            }
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "KosarajuSCC: single vertex" {
    const T = KosarajuSCC(u32);
    var kosaraju = T.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator);
    defer adjacency_list.deinit();

    var v0_neighbors: std.ArrayList(u32) = .{};
    defer v0_neighbors.deinit(testing.allocator);
    try adjacency_list.put(0, v0_neighbors);

    var result = try kosaraju.run(&adjacency_list);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), result.components.len);
    try testing.expectEqual(@as(usize, 1), result.components[0].len);
    try testing.expectEqual(@as(u32, 0), result.components[0][0]);
}

test "KosarajuSCC: two vertices, no edges" {
    const T = KosarajuSCC(u32);
    var kosaraju = T.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator);
    defer {
        var iter = adjacency_list.valueIterator();
        while (iter.next()) |list| {
            list.deinit(testing.allocator);
        }
        adjacency_list.deinit();
    }

    const v0_neighbors: std.ArrayList(u32) = .{};
    try adjacency_list.put(0, v0_neighbors);

    const v1_neighbors: std.ArrayList(u32) = .{};
    try adjacency_list.put(1, v1_neighbors);

    var result = try kosaraju.run(&adjacency_list);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), result.components.len);
    // Each vertex is its own SCC
    for (result.components) |component| {
        try testing.expectEqual(@as(usize, 1), component.len);
    }
}

test "KosarajuSCC: simple cycle" {
    const T = KosarajuSCC(u32);
    var kosaraju = T.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator);
    defer {
        var iter = adjacency_list.valueIterator();
        while (iter.next()) |list| {
            list.deinit(testing.allocator);
        }
        adjacency_list.deinit();
    }

    // Cycle: 0 -> 1 -> 2 -> 0
    var v0_neighbors: std.ArrayList(u32) = .{};
    try v0_neighbors.append(testing.allocator, 1);
    try adjacency_list.put(0, v0_neighbors);

    var v1_neighbors: std.ArrayList(u32) = .{};
    try v1_neighbors.append(testing.allocator, 2);
    try adjacency_list.put(1, v1_neighbors);

    var v2_neighbors: std.ArrayList(u32) = .{};
    try v2_neighbors.append(testing.allocator, 0);
    try adjacency_list.put(2, v2_neighbors);

    var result = try kosaraju.run(&adjacency_list);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), result.components.len);
    try testing.expectEqual(@as(usize, 3), result.components[0].len);

    // All vertices should be in the same SCC
    var vertices_in_scc = std.AutoHashMap(u32, void).init(testing.allocator);
    defer vertices_in_scc.deinit();
    for (result.components[0]) |v| {
        try vertices_in_scc.put(v, {});
    }
    try testing.expect(vertices_in_scc.contains(0));
    try testing.expect(vertices_in_scc.contains(1));
    try testing.expect(vertices_in_scc.contains(2));
}

test "KosarajuSCC: linear chain (DAG)" {
    const T = KosarajuSCC(u32);
    var kosaraju = T.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator);
    defer {
        var iter = adjacency_list.valueIterator();
        while (iter.next()) |list| {
            list.deinit(testing.allocator);
        }
        adjacency_list.deinit();
    }

    // Chain: 0 -> 1 -> 2 -> 3 (no cycles)
    var v0_neighbors: std.ArrayList(u32) = .{};
    try v0_neighbors.append(testing.allocator, 1);
    try adjacency_list.put(0, v0_neighbors);

    var v1_neighbors: std.ArrayList(u32) = .{};
    try v1_neighbors.append(testing.allocator, 2);
    try adjacency_list.put(1, v1_neighbors);

    var v2_neighbors: std.ArrayList(u32) = .{};
    try v2_neighbors.append(testing.allocator, 3);
    try adjacency_list.put(2, v2_neighbors);

    const v3_neighbors: std.ArrayList(u32) = .{};
    try adjacency_list.put(3, v3_neighbors);

    var result = try kosaraju.run(&adjacency_list);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 4), result.components.len);
    // Each vertex is its own SCC (no cycles)
    for (result.components) |component| {
        try testing.expectEqual(@as(usize, 1), component.len);
    }
}

test "KosarajuSCC: two separate cycles" {
    const T = KosarajuSCC(u32);
    var kosaraju = T.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator);
    defer {
        var iter = adjacency_list.valueIterator();
        while (iter.next()) |list| {
            list.deinit(testing.allocator);
        }
        adjacency_list.deinit();
    }

    // Cycle 1: 0 -> 1 -> 0
    var v0_neighbors: std.ArrayList(u32) = .{};
    try v0_neighbors.append(testing.allocator, 1);
    try adjacency_list.put(0, v0_neighbors);

    var v1_neighbors: std.ArrayList(u32) = .{};
    try v1_neighbors.append(testing.allocator, 0);
    try adjacency_list.put(1, v1_neighbors);

    // Cycle 2: 2 -> 3 -> 2
    var v2_neighbors: std.ArrayList(u32) = .{};
    try v2_neighbors.append(testing.allocator, 3);
    try adjacency_list.put(2, v2_neighbors);

    var v3_neighbors: std.ArrayList(u32) = .{};
    try v3_neighbors.append(testing.allocator, 2);
    try adjacency_list.put(3, v3_neighbors);

    var result = try kosaraju.run(&adjacency_list);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), result.components.len);
    // Each cycle is one SCC
    for (result.components) |component| {
        try testing.expectEqual(@as(usize, 2), component.len);
    }
}

test "KosarajuSCC: complex graph" {
    const T = KosarajuSCC(u32);
    var kosaraju = T.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator);
    defer {
        var iter = adjacency_list.valueIterator();
        while (iter.next()) |list| {
            list.deinit(testing.allocator);
        }
        adjacency_list.deinit();
    }

    // Graph with multiple SCCs:
    // SCC1: 0 <-> 1
    // SCC2: 2 <-> 3 <-> 4 (triangle)
    // SCC3: 5 (alone)
    // Edges between SCCs: 1 -> 2, 4 -> 5

    var v0_neighbors: std.ArrayList(u32) = .{};
    try v0_neighbors.append(testing.allocator, 1);
    try adjacency_list.put(0, v0_neighbors);

    var v1_neighbors: std.ArrayList(u32) = .{};
    try v1_neighbors.append(testing.allocator, 0);
    try v1_neighbors.append(testing.allocator, 2);
    try adjacency_list.put(1, v1_neighbors);

    var v2_neighbors: std.ArrayList(u32) = .{};
    try v2_neighbors.append(testing.allocator, 3);
    try adjacency_list.put(2, v2_neighbors);

    var v3_neighbors: std.ArrayList(u32) = .{};
    try v3_neighbors.append(testing.allocator, 4);
    try adjacency_list.put(3, v3_neighbors);

    var v4_neighbors: std.ArrayList(u32) = .{};
    try v4_neighbors.append(testing.allocator, 2);
    try v4_neighbors.append(testing.allocator, 5);
    try adjacency_list.put(4, v4_neighbors);

    const v5_neighbors: std.ArrayList(u32) = .{};
    try adjacency_list.put(5, v5_neighbors);

    var result = try kosaraju.run(&adjacency_list);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 3), result.components.len);

    // Find SCC sizes
    var sizes: std.ArrayList(usize) = .{};
    defer sizes.deinit(testing.allocator);
    for (result.components) |component| {
        try sizes.append(testing.allocator, component.len);
    }

    // Sort sizes for comparison
    std.mem.sort(usize, sizes.items, {}, std.sort.asc(usize));

    try testing.expectEqual(@as(usize, 1), sizes.items[0]); // SCC3: {5}
    try testing.expectEqual(@as(usize, 2), sizes.items[1]); // SCC1: {0, 1}
    try testing.expectEqual(@as(usize, 3), sizes.items[2]); // SCC2: {2, 3, 4}
}

test "KosarajuSCC: self-loop" {
    const T = KosarajuSCC(u32);
    var kosaraju = T.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator);
    defer {
        var iter = adjacency_list.valueIterator();
        while (iter.next()) |list| {
            list.deinit(testing.allocator);
        }
        adjacency_list.deinit();
    }

    // Vertex 0 has a self-loop
    var v0_neighbors: std.ArrayList(u32) = .{};
    try v0_neighbors.append(testing.allocator, 0);
    try adjacency_list.put(0, v0_neighbors);

    var result = try kosaraju.run(&adjacency_list);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), result.components.len);
    try testing.expectEqual(@as(usize, 1), result.components[0].len);
    try testing.expectEqual(@as(u32, 0), result.components[0][0]);
}

test "KosarajuSCC: stress test (100 vertices chain)" {
    const T = KosarajuSCC(u32);
    var kosaraju = T.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator);
    defer {
        var iter = adjacency_list.valueIterator();
        while (iter.next()) |list| {
            list.deinit(testing.allocator);
        }
        adjacency_list.deinit();
    }

    // Chain: 0 -> 1 -> 2 -> ... -> 99 (no cycles, all separate SCCs)
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        var neighbors: std.ArrayList(u32) = .{};
        if (i < 99) {
            try neighbors.append(testing.allocator, i + 1);
        }
        try adjacency_list.put(i, neighbors);
    }

    var result = try kosaraju.run(&adjacency_list);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 100), result.components.len);
    // Each vertex is its own SCC
    for (result.components) |component| {
        try testing.expectEqual(@as(usize, 1), component.len);
    }
}

test "KosarajuSCC: stress test (100 vertex cycle)" {
    const T = KosarajuSCC(u32);
    var kosaraju = T.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator);
    defer {
        var iter = adjacency_list.valueIterator();
        while (iter.next()) |list| {
            list.deinit(testing.allocator);
        }
        adjacency_list.deinit();
    }

    // Cycle: 0 -> 1 -> 2 -> ... -> 99 -> 0 (one giant SCC)
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        var neighbors: std.ArrayList(u32) = .{};
        try neighbors.append(testing.allocator, (i + 1) % 100);
        try adjacency_list.put(i, neighbors);
    }

    var result = try kosaraju.run(&adjacency_list);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), result.components.len);
    try testing.expectEqual(@as(usize, 100), result.components[0].len);
}

test "KosarajuSCC: vertices only in edges (not explicit)" {
    const T = KosarajuSCC(u32);
    var kosaraju = T.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator);
    defer {
        var iter = adjacency_list.valueIterator();
        while (iter.next()) |list| {
            list.deinit(testing.allocator);
        }
        adjacency_list.deinit();
    }

    // Graph: 0 -> 1, 1 -> 2
    // Vertex 2 exists only as a target, not in key set initially
    var v0_neighbors: std.ArrayList(u32) = .{};
    try v0_neighbors.append(testing.allocator, 1);
    try adjacency_list.put(0, v0_neighbors);

    var v1_neighbors: std.ArrayList(u32) = .{};
    try v1_neighbors.append(testing.allocator, 2);
    try adjacency_list.put(1, v1_neighbors);

    var result = try kosaraju.run(&adjacency_list);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 3), result.components.len);
    // Each vertex is its own SCC (no cycles)
    for (result.components) |component| {
        try testing.expectEqual(@as(usize, 1), component.len);
    }
}
