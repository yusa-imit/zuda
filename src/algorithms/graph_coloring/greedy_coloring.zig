const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// Greedy Graph Coloring using First-Fit strategy
/// Assigns the smallest available color to each vertex in order
///
/// Time: O(V²) where V = number of vertices
/// Space: O(V) for color assignments
///
/// Parameters:
///   - T: Vertex ID type (must be integer type)
///   - allocator: Memory allocator
///   - adj_list: Adjacency list representation (adj_list[u] = neighbors of u)
///
/// Returns:
///   - ArrayList of colors (colors[u] = color of vertex u, 0-indexed)
///
/// Example:
///   ```zig
///   var graph = try ArrayList(ArrayList(u32)).initCapacity(allocator, 3);
///   defer graph.deinit();
///   // Add edges...
///   const coloring = try greedyColoring(u32, allocator, graph);
///   defer coloring.deinit();
///   ```
pub fn greedyColoring(comptime T: type, allocator: Allocator, adj_list: ArrayList(ArrayList(T))) !ArrayList(usize) {
    const n = adj_list.items.len;
    var colors = try ArrayList(usize).initCapacity(allocator, n);
    errdefer colors.deinit();

    // Initialize all vertices as uncolored (max value sentinel)
    for (0..n) |_| {
        colors.appendAssumeCapacity(std.math.maxInt(usize));
    }

    // Color vertices in order 0, 1, 2, ..., n-1
    for (0..n) |u| {
        // Track which colors are used by neighbors
        var used_colors = std.AutoHashMap(usize, void).init(allocator);
        defer used_colors.deinit();

        for (adj_list.items[u].items) |v_raw| {
            const v: usize = @intCast(v_raw);
            if (colors.items[v] != std.math.maxInt(usize)) {
                try used_colors.put(colors.items[v], {});
            }
        }

        // Find smallest unused color
        var color: usize = 0;
        while (used_colors.contains(color)) : (color += 1) {}
        colors.items[u] = color;
    }

    return colors;
}

/// Welsh-Powell Graph Coloring Algorithm
/// Colors vertices in decreasing order of degree
///
/// Time: O(V² + E) where V = vertices, E = edges
/// Space: O(V) for color assignments + O(V) for degree sorting
///
/// Generally produces better colorings than simple greedy approach
pub fn welshPowell(comptime T: type, allocator: Allocator, adj_list: ArrayList(ArrayList(T))) !ArrayList(usize) {
    const n = adj_list.items.len;

    // Compute degrees
    var degrees = try ArrayList(struct { vertex: usize, degree: usize }).initCapacity(allocator, n);
    defer degrees.deinit();

    for (adj_list.items, 0..) |neighbors, u| {
        degrees.appendAssumeCapacity(.{ .vertex = u, .degree = neighbors.items.len });
    }

    // Sort vertices by degree (descending)
    std.mem.sort(@TypeOf(degrees.items[0]), degrees.items, {}, struct {
        fn lessThan(_: void, a: @TypeOf(degrees.items[0]), b: @TypeOf(degrees.items[0])) bool {
            return a.degree > b.degree; // Descending order
        }
    }.lessThan);

    // Initialize colors
    var colors = try ArrayList(usize).initCapacity(allocator, n);
    errdefer colors.deinit();
    for (0..n) |_| {
        colors.appendAssumeCapacity(std.math.maxInt(usize));
    }

    // Color vertices in degree order
    for (degrees.items) |item| {
        const u = item.vertex;

        // Track which colors are used by neighbors
        var used_colors = std.AutoHashMap(usize, void).init(allocator);
        defer used_colors.deinit();

        for (adj_list.items[u].items) |v_raw| {
            const v: usize = @intCast(v_raw);
            if (colors.items[v] != std.math.maxInt(usize)) {
                try used_colors.put(colors.items[v], {});
            }
        }

        // Find smallest unused color
        var color: usize = 0;
        while (used_colors.contains(color)) : (color += 1) {}
        colors.items[u] = color;
    }

    return colors;
}

/// DSatur (Degree of Saturation) Graph Coloring Algorithm
/// Colors vertices in order of highest saturation degree (number of different colors used by neighbors)
///
/// Time: O(V² + E) where V = vertices, E = edges
/// Space: O(V) for color assignments + O(V) for saturation tracking
///
/// Often produces optimal or near-optimal colorings
pub fn dsatur(comptime T: type, allocator: Allocator, adj_list: ArrayList(ArrayList(T))) !ArrayList(usize) {
    const n = adj_list.items.len;

    // Initialize colors
    var colors = try ArrayList(usize).initCapacity(allocator, n);
    errdefer colors.deinit();
    for (0..n) |_| {
        colors.appendAssumeCapacity(std.math.maxInt(usize));
    }

    // Track saturation degree (number of different colors used by neighbors)
    var saturation = try ArrayList(usize).initCapacity(allocator, n);
    defer saturation.deinit();
    for (0..n) |_| {
        saturation.appendAssumeCapacity(0);
    }

    // Track uncolored vertices
    var uncolored = std.AutoHashMap(usize, void).init(allocator);
    defer uncolored.deinit();
    for (0..n) |u| {
        try uncolored.put(u, {});
    }

    // Color the vertex with highest degree first
    var max_degree: usize = 0;
    var first_vertex: usize = 0;
    for (adj_list.items, 0..) |neighbors, u| {
        if (neighbors.items.len > max_degree) {
            max_degree = neighbors.items.len;
            first_vertex = u;
        }
    }
    colors.items[first_vertex] = 0;
    _ = uncolored.remove(first_vertex);

    // Update saturation of neighbors
    for (adj_list.items[first_vertex].items) |v_raw| {
        const v: usize = @intCast(v_raw);
        saturation.items[v] = 1;
    }

    // Color remaining vertices
    while (uncolored.count() > 0) {
        // Find vertex with highest saturation (break ties by degree)
        var max_sat: usize = 0;
        var max_deg: usize = 0;
        var next_vertex: usize = 0;

        var it = uncolored.keyIterator();
        while (it.next()) |u_ptr| {
            const u = u_ptr.*;
            const sat = saturation.items[u];
            const deg = adj_list.items[u].items.len;

            if (sat > max_sat or (sat == max_sat and deg > max_deg)) {
                max_sat = sat;
                max_deg = deg;
                next_vertex = u;
            }
        }

        // Color the selected vertex
        var used_colors = std.AutoHashMap(usize, void).init(allocator);
        defer used_colors.deinit();

        for (adj_list.items[next_vertex].items) |v_raw| {
            const v: usize = @intCast(v_raw);
            if (colors.items[v] != std.math.maxInt(usize)) {
                try used_colors.put(colors.items[v], {});
            }
        }

        // Find smallest unused color
        var color: usize = 0;
        while (used_colors.contains(color)) : (color += 1) {}
        colors.items[next_vertex] = color;
        _ = uncolored.remove(next_vertex);

        // Update saturation of uncolored neighbors
        for (adj_list.items[next_vertex].items) |v_raw| {
            const v: usize = @intCast(v_raw);
            if (colors.items[v] == std.math.maxInt(usize)) {
                // Count distinct colors used by v's neighbors
                var neighbor_colors = std.AutoHashMap(usize, void).init(allocator);
                defer neighbor_colors.deinit();

                for (adj_list.items[v].items) |w_raw| {
                    const w: usize = @intCast(w_raw);
                    if (colors.items[w] != std.math.maxInt(usize)) {
                        try neighbor_colors.put(colors.items[w], {});
                    }
                }
                saturation.items[v] = neighbor_colors.count();
            }
        }
    }

    return colors;
}

/// Computes the chromatic number (minimum number of colors) from a coloring
///
/// Time: O(V)
/// Space: O(1)
pub fn chromaticNumber(colors: ArrayList(usize)) usize {
    if (colors.items.len == 0) return 0;

    var max_color: usize = 0;
    for (colors.items) |c| {
        if (c != std.math.maxInt(usize) and c > max_color) {
            max_color = c;
        }
    }
    return max_color + 1; // Colors are 0-indexed
}

/// Validates a graph coloring
/// Returns true if the coloring is valid (no adjacent vertices have the same color)
///
/// Time: O(V + E)
/// Space: O(1)
pub fn isValidColoring(comptime T: type, adj_list: ArrayList(ArrayList(T)), colors: ArrayList(usize)) bool {
    if (adj_list.items.len != colors.items.len) return false;

    for (adj_list.items, 0..) |neighbors, u| {
        const u_color = colors.items[u];
        if (u_color == std.math.maxInt(usize)) continue; // Uncolored vertex

        for (neighbors.items) |v_raw| {
            const v: usize = @intCast(v_raw);
            if (colors.items[v] == u_color) return false; // Adjacent vertices with same color
        }
    }
    return true;
}

// ============================================================================
// TESTS
// ============================================================================

test "greedy coloring - simple triangle" {
    const allocator = std.testing.allocator;

    // Triangle graph: 0-1-2-0
    var graph = try ArrayList(ArrayList(u32)).initCapacity(allocator, 3);
    defer {
        for (graph.items) |*list| list.deinit();
        graph.deinit();
    }

    var list0 = ArrayList(u32).init(allocator);
    try list0.append(1);
    try list0.append(2);
    try graph.append(list0);

    var list1 = ArrayList(u32).init(allocator);
    try list1.append(0);
    try list1.append(2);
    try graph.append(list1);

    var list2 = ArrayList(u32).init(allocator);
    try list2.append(0);
    try list2.append(1);
    try graph.append(list2);

    const coloring = try greedyColoring(u32, allocator, graph);
    defer coloring.deinit();

    try std.testing.expect(isValidColoring(u32, graph, coloring));
    try std.testing.expectEqual(@as(usize, 3), chromaticNumber(coloring));
}

test "greedy coloring - bipartite graph" {
    const allocator = std.testing.allocator;

    // Complete bipartite K(2,2): 0-2, 0-3, 1-2, 1-3
    var graph = try ArrayList(ArrayList(u32)).initCapacity(allocator, 4);
    defer {
        for (graph.items) |*list| list.deinit();
        graph.deinit();
    }

    var list0 = ArrayList(u32).init(allocator);
    try list0.append(2);
    try list0.append(3);
    try graph.append(list0);

    var list1 = ArrayList(u32).init(allocator);
    try list1.append(2);
    try list1.append(3);
    try graph.append(list1);

    var list2 = ArrayList(u32).init(allocator);
    try list2.append(0);
    try list2.append(1);
    try graph.append(list2);

    var list3 = ArrayList(u32).init(allocator);
    try list3.append(0);
    try list3.append(1);
    try graph.append(list3);

    const coloring = try greedyColoring(u32, allocator, graph);
    defer coloring.deinit();

    try std.testing.expect(isValidColoring(u32, graph, coloring));
    try std.testing.expectEqual(@as(usize, 2), chromaticNumber(coloring));
}

test "welsh-powell - improved coloring" {
    const allocator = std.testing.allocator;

    // Graph where Welsh-Powell outperforms simple greedy
    // 0-1, 0-2, 1-3, 2-3, 3-4
    var graph = try ArrayList(ArrayList(u32)).initCapacity(allocator, 5);
    defer {
        for (graph.items) |*list| list.deinit();
        graph.deinit();
    }

    var list0 = ArrayList(u32).init(allocator);
    try list0.append(1);
    try list0.append(2);
    try graph.append(list0);

    var list1 = ArrayList(u32).init(allocator);
    try list1.append(0);
    try list1.append(3);
    try graph.append(list1);

    var list2 = ArrayList(u32).init(allocator);
    try list2.append(0);
    try list2.append(3);
    try graph.append(list2);

    var list3 = ArrayList(u32).init(allocator);
    try list3.append(1);
    try list3.append(2);
    try list3.append(4);
    try graph.append(list3);

    var list4 = ArrayList(u32).init(allocator);
    try list4.append(3);
    try graph.append(list4);

    const coloring = try welshPowell(u32, allocator, graph);
    defer coloring.deinit();

    try std.testing.expect(isValidColoring(u32, graph, coloring));
    // Should use at most 3 colors (optimal)
    try std.testing.expect(chromaticNumber(coloring) <= 3);
}

test "dsatur - optimal coloring" {
    const allocator = std.testing.allocator;

    // Same graph as welsh-powell test
    var graph = try ArrayList(ArrayList(u32)).initCapacity(allocator, 5);
    defer {
        for (graph.items) |*list| list.deinit();
        graph.deinit();
    }

    var list0 = ArrayList(u32).init(allocator);
    try list0.append(1);
    try list0.append(2);
    try graph.append(list0);

    var list1 = ArrayList(u32).init(allocator);
    try list1.append(0);
    try list1.append(3);
    try graph.append(list1);

    var list2 = ArrayList(u32).init(allocator);
    try list2.append(0);
    try list2.append(3);
    try graph.append(list2);

    var list3 = ArrayList(u32).init(allocator);
    try list3.append(1);
    try list3.append(2);
    try list3.append(4);
    try graph.append(list3);

    var list4 = ArrayList(u32).init(allocator);
    try list4.append(3);
    try graph.append(list4);

    const coloring = try dsatur(u32, allocator, graph);
    defer coloring.deinit();

    try std.testing.expect(isValidColoring(u32, graph, coloring));
    try std.testing.expectEqual(@as(usize, 3), chromaticNumber(coloring));
}

test "dsatur - complete graph K5" {
    const allocator = std.testing.allocator;

    // Complete graph K5 requires 5 colors
    var graph = try ArrayList(ArrayList(u32)).initCapacity(allocator, 5);
    defer {
        for (graph.items) |*list| list.deinit();
        graph.deinit();
    }

    for (0..5) |i| {
        var list = ArrayList(u32).init(allocator);
        for (0..5) |j| {
            if (i != j) try list.append(@intCast(j));
        }
        try graph.append(list);
    }

    const coloring = try dsatur(u32, allocator, graph);
    defer coloring.deinit();

    try std.testing.expect(isValidColoring(u32, graph, coloring));
    try std.testing.expectEqual(@as(usize, 5), chromaticNumber(coloring));
}

test "chromatic number - empty graph" {
    const allocator = std.testing.allocator;
    var colors = ArrayList(usize).init(allocator);
    defer colors.deinit();

    try std.testing.expectEqual(@as(usize, 0), chromaticNumber(colors));
}

test "chromatic number - single color" {
    const allocator = std.testing.allocator;
    var colors = try ArrayList(usize).initCapacity(allocator, 3);
    defer colors.deinit();

    colors.appendAssumeCapacity(0);
    colors.appendAssumeCapacity(0);
    colors.appendAssumeCapacity(0);

    try std.testing.expectEqual(@as(usize, 1), chromaticNumber(colors));
}

test "isValidColoring - invalid coloring" {
    const allocator = std.testing.allocator;

    // Simple edge 0-1
    var graph = try ArrayList(ArrayList(u32)).initCapacity(allocator, 2);
    defer {
        for (graph.items) |*list| list.deinit();
        graph.deinit();
    }

    var list0 = ArrayList(u32).init(allocator);
    try list0.append(1);
    try graph.append(list0);

    var list1 = ArrayList(u32).init(allocator);
    try list1.append(0);
    try graph.append(list1);

    // Invalid: both vertices have the same color
    var colors = try ArrayList(usize).initCapacity(allocator, 2);
    defer colors.deinit();
    colors.appendAssumeCapacity(0);
    colors.appendAssumeCapacity(0);

    try std.testing.expect(!isValidColoring(u32, graph, colors));
}

test "greedy coloring - empty graph" {
    const allocator = std.testing.allocator;

    var graph = ArrayList(ArrayList(u32)).init(allocator);
    defer graph.deinit();

    const coloring = try greedyColoring(u32, allocator, graph);
    defer coloring.deinit();

    try std.testing.expectEqual(@as(usize, 0), coloring.items.len);
}

test "greedy coloring - single vertex" {
    const allocator = std.testing.allocator;

    var graph = try ArrayList(ArrayList(u32)).initCapacity(allocator, 1);
    defer {
        for (graph.items) |*list| list.deinit();
        graph.deinit();
    }

    const list0 = ArrayList(u32).init(allocator);
    try graph.append(list0);

    const coloring = try greedyColoring(u32, allocator, graph);
    defer coloring.deinit();

    try std.testing.expectEqual(@as(usize, 1), coloring.items.len);
    try std.testing.expectEqual(@as(usize, 0), coloring.items[0]);
    try std.testing.expectEqual(@as(usize, 1), chromaticNumber(coloring));
}

test "welsh-powell - petersen graph" {
    const allocator = std.testing.allocator;

    // Petersen graph (10 vertices, 15 edges, chromatic number = 3)
    var graph = try ArrayList(ArrayList(u32)).initCapacity(allocator, 10);
    defer {
        for (graph.items) |*list| list.deinit();
        graph.deinit();
    }

    // Outer pentagon: 0-1-2-3-4-0
    // Inner star: 5-7-9-6-8-5
    // Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    const edges = [_][2]u32{
        .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 4 }, .{ 4, 0 }, // Outer
        .{ 5, 7 }, .{ 7, 9 }, .{ 9, 6 }, .{ 6, 8 }, .{ 8, 5 }, // Inner
        .{ 0, 5 }, .{ 1, 6 }, .{ 2, 7 }, .{ 3, 8 }, .{ 4, 9 }, // Spokes
    };

    for (0..10) |_| {
        try graph.append(ArrayList(u32).init(allocator));
    }

    for (edges) |edge| {
        try graph.items[edge[0]].append(edge[1]);
        try graph.items[edge[1]].append(edge[0]);
    }

    const coloring = try welshPowell(u32, allocator, graph);
    defer coloring.deinit();

    try std.testing.expect(isValidColoring(u32, graph, coloring));
    try std.testing.expectEqual(@as(usize, 3), chromaticNumber(coloring));
}

test "dsatur - cycle graph" {
    const allocator = std.testing.allocator;

    // Cycle C6: 0-1-2-3-4-5-0 (chromatic number = 2)
    var graph = try ArrayList(ArrayList(u32)).initCapacity(allocator, 6);
    defer {
        for (graph.items) |*list| list.deinit();
        graph.deinit();
    }

    for (0..6) |i| {
        var list = ArrayList(u32).init(allocator);
        try list.append(@intCast((i + 1) % 6));
        try list.append(@intCast((i + 5) % 6));
        try graph.append(list);
    }

    const coloring = try dsatur(u32, allocator, graph);
    defer coloring.deinit();

    try std.testing.expect(isValidColoring(u32, graph, coloring));
    try std.testing.expectEqual(@as(usize, 2), chromaticNumber(coloring));
}
