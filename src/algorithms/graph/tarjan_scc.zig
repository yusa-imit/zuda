const std = @import("std");
const testing = std.testing;

/// Tarjan's algorithm for finding Strongly Connected Components (SCCs) in a directed graph.
///
/// A strongly connected component is a maximal set of vertices where every vertex is
/// reachable from every other vertex in the set.
///
/// Time: O(V + E) — single DFS traversal
/// Space: O(V) — stack and metadata arrays
///
/// Generic over:
/// - V: Vertex type (must be hashable and comparable)
///
/// Algorithm (single-pass DFS with stack):
/// 1. DFS traversal with discovery time tracking
/// 2. Maintain stack of vertices in current path
/// 3. Track low-link values (lowest discovery time reachable)
/// 4. When vertex is a root of SCC (discovery == low-link):
///    - Pop stack until vertex is reached
///    - All popped vertices form one SCC
///
/// Difference from Kosaraju:
/// - Tarjan: Single DFS pass with stack (online algorithm)
/// - Kosaraju: Two DFS passes on original and transposed graph
///
/// Both have O(V + E) time, but Tarjan is often preferred for its single-pass nature.
pub fn TarjanSCC(comptime V: type) type {
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
        /// Time: O(V + E) — single DFS traversal
        /// Space: O(V) — recursion stack and metadata
        ///
        /// Parameters:
        /// - adjacency_list: Map from vertex to list of outgoing neighbors
        ///
        /// Returns:
        /// - Result containing list of SCCs (each SCC is a list of vertices)
        ///
        /// Note: Components are in reverse topological order of the condensation graph.
        pub fn run(
            self: Self,
            adjacency_list: *const std.AutoHashMap(V, std.ArrayList(V)),
        ) !Result {
            if (adjacency_list.count() == 0) {
                return Result{ .components = try self.allocator.alloc(Component, 0) };
            }

            var context = Context(V).init(self.allocator);
            defer context.deinit();

            // Run DFS from all unvisited vertices
            var vertex_iter = adjacency_list.keyIterator();
            while (vertex_iter.next()) |vertex| {
                if (!context.visited.contains(vertex.*)) {
                    try context.strongconnect(vertex.*, adjacency_list);
                }
            }

            return Result{
                .components = try context.components.toOwnedSlice(self.allocator),
            };
        }
    };
}

/// Internal context for Tarjan's algorithm
fn Context(comptime V: type) type {
    return struct {
        const CtxSelf = @This();

        allocator: std.mem.Allocator,
        visited: std.AutoHashMap(V, void),
        discovery: std.AutoHashMap(V, usize),
        low_link: std.AutoHashMap(V, usize),
        on_stack: std.AutoHashMap(V, void),
        stack: std.ArrayList(V),
        components: std.ArrayList([]V),
        time: usize,

        fn init(allocator: std.mem.Allocator) CtxSelf {
            return .{
                .allocator = allocator,
                .visited = std.AutoHashMap(V, void).init(allocator),
                .discovery = std.AutoHashMap(V, usize).init(allocator),
                .low_link = std.AutoHashMap(V, usize).init(allocator),
                .on_stack = std.AutoHashMap(V, void).init(allocator),
                .stack = .{},
                .components = .{},
                .time = 0,
            };
        }

        fn deinit(self: *CtxSelf) void {
            self.visited.deinit();
            self.discovery.deinit();
            self.low_link.deinit();
            self.on_stack.deinit();
            self.stack.deinit(self.allocator);
            self.components.deinit(self.allocator);
        }

        fn strongconnect(
            self: *CtxSelf,
            v: V,
            adjacency_list: *const std.AutoHashMap(V, std.ArrayList(V)),
        ) !void {
            // Set discovery time and low-link value
            try self.discovery.put(v, self.time);
            try self.low_link.put(v, self.time);
            self.time += 1;

            // Mark as visited and push to stack
            try self.visited.put(v, {});
            try self.stack.append(self.allocator, v);
            try self.on_stack.put(v, {});

            // Explore neighbors
            if (adjacency_list.get(v)) |neighbors| {
                for (neighbors.items) |w| {
                    if (!self.visited.contains(w)) {
                        // Recurse on unvisited neighbor
                        try self.strongconnect(w, adjacency_list);

                        // Update low-link value
                        const v_low = self.low_link.get(v) orelse unreachable;
                        const w_low = self.low_link.get(w) orelse unreachable;
                        try self.low_link.put(v, @min(v_low, w_low));
                    } else if (self.on_stack.contains(w)) {
                        // w is in current SCC, update low-link
                        const v_low = self.low_link.get(v) orelse unreachable;
                        const w_disc = self.discovery.get(w) orelse unreachable;
                        try self.low_link.put(v, @min(v_low, w_disc));
                    }
                }
            }

            // If v is a root of SCC, pop the stack to form component
            const v_disc = self.discovery.get(v) orelse unreachable;
            const v_low = self.low_link.get(v) orelse unreachable;

            if (v_disc == v_low) {
                var component: std.ArrayList(V) = .{};
                errdefer component.deinit(self.allocator);

                while (true) {
                    const w = self.stack.pop() orelse unreachable;
                    _ = self.on_stack.remove(w);
                    try component.append(self.allocator, w);

                    if (std.meta.eql(w, v)) {
                        break;
                    }
                }

                try self.components.append(self.allocator, try component.toOwnedSlice(self.allocator));
            }
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "TarjanSCC: single vertex" {
    const T = TarjanSCC(u32);
    var tarjan = T.init(testing.allocator);

    var adjacency_list = std.AutoHashMap(u32, std.ArrayList(u32)).init(testing.allocator);
    defer adjacency_list.deinit();

    var v0_neighbors: std.ArrayList(u32) = .{};
    defer v0_neighbors.deinit(testing.allocator);
    try adjacency_list.put(0, v0_neighbors);

    var result = try tarjan.run(&adjacency_list);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), result.components.len);
    try testing.expectEqual(@as(usize, 1), result.components[0].len);
    try testing.expectEqual(@as(u32, 0), result.components[0][0]);
}

test "TarjanSCC: two vertices, no edges" {
    const T = TarjanSCC(u32);
    var tarjan = T.init(testing.allocator);

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

    var result = try tarjan.run(&adjacency_list);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), result.components.len);
    // Each vertex is its own SCC
    for (result.components) |component| {
        try testing.expectEqual(@as(usize, 1), component.len);
    }
}

test "TarjanSCC: simple cycle" {
    const T = TarjanSCC(u32);
    var tarjan = T.init(testing.allocator);

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

    var result = try tarjan.run(&adjacency_list);
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

test "TarjanSCC: linear chain (DAG)" {
    const T = TarjanSCC(u32);
    var tarjan = T.init(testing.allocator);

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

    var result = try tarjan.run(&adjacency_list);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 4), result.components.len);
    // Each vertex is its own SCC (no cycles)
    for (result.components) |component| {
        try testing.expectEqual(@as(usize, 1), component.len);
    }
}

test "TarjanSCC: two separate cycles" {
    const T = TarjanSCC(u32);
    var tarjan = T.init(testing.allocator);

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

    var result = try tarjan.run(&adjacency_list);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), result.components.len);
    // Each cycle is one SCC
    for (result.components) |component| {
        try testing.expectEqual(@as(usize, 2), component.len);
    }
}

test "TarjanSCC: complex graph" {
    const T = TarjanSCC(u32);
    var tarjan = T.init(testing.allocator);

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

    var result = try tarjan.run(&adjacency_list);
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

test "TarjanSCC: self-loop" {
    const T = TarjanSCC(u32);
    var tarjan = T.init(testing.allocator);

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

    var result = try tarjan.run(&adjacency_list);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), result.components.len);
    try testing.expectEqual(@as(usize, 1), result.components[0].len);
    try testing.expectEqual(@as(u32, 0), result.components[0][0]);
}

test "TarjanSCC: stress test (100 vertices chain)" {
    const T = TarjanSCC(u32);
    var tarjan = T.init(testing.allocator);

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

    var result = try tarjan.run(&adjacency_list);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 100), result.components.len);
    // Each vertex is its own SCC
    for (result.components) |component| {
        try testing.expectEqual(@as(usize, 1), component.len);
    }
}

test "TarjanSCC: stress test (100 vertex cycle)" {
    const T = TarjanSCC(u32);
    var tarjan = T.init(testing.allocator);

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

    var result = try tarjan.run(&adjacency_list);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), result.components.len);
    try testing.expectEqual(@as(usize, 100), result.components[0].len);
}
