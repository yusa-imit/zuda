//! Greedy graph coloring heuristics: first-fit, Welsh-Powell and DSatur.
//!
//! Input is an adjacency list (`adj_list[u]` holds the neighbours of `u`, ids below the vertex
//! count). Results are unmanaged `ArrayList(usize)` colors, 0-indexed, owned by the caller and
//! released with `deinit(gpa)`. Every function allocates only its scratch and result and frees
//! all of it on failure.

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
///   var graph = try ArrayList(ArrayList(u32)).initCapacity(gpa, 3);
///   defer graph.deinit(gpa);
///   // Add edges...
///   var coloring = try greedyColoring(u32, allocator, graph);
///   defer coloring.deinit(gpa);
///   ```
pub fn greedyColoring(comptime T: type, gpa: Allocator, adj_list: ArrayList(ArrayList(T))) !ArrayList(usize) {
    const n = adj_list.items.len;
    var colors = try ArrayList(usize).initCapacity(gpa, n);
    errdefer colors.deinit(gpa);

    // Initialize all vertices as uncolored (max value sentinel)
    for (0..n) |_| {
        colors.appendAssumeCapacity(std.math.maxInt(usize));
    }

    // Color vertices in order 0, 1, 2, ..., n-1
    for (0..n) |u| {
        // Track which colors are used by neighbors
        var used_colors = std.AutoHashMap(usize, void).init(gpa);
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
pub fn welshPowell(comptime T: type, gpa: Allocator, adj_list: ArrayList(ArrayList(T))) !ArrayList(usize) {
    const n = adj_list.items.len;

    // Compute degrees
    var degrees = try ArrayList(struct { vertex: usize, degree: usize }).initCapacity(gpa, n);
    defer degrees.deinit(gpa);

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
    var colors = try ArrayList(usize).initCapacity(gpa, n);
    errdefer colors.deinit(gpa);
    for (0..n) |_| {
        colors.appendAssumeCapacity(std.math.maxInt(usize));
    }

    // Color vertices in degree order
    for (degrees.items) |item| {
        const u = item.vertex;

        // Track which colors are used by neighbors
        var used_colors = std.AutoHashMap(usize, void).init(gpa);
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
pub fn dsatur(comptime T: type, gpa: Allocator, adj_list: ArrayList(ArrayList(T))) !ArrayList(usize) {
    const n = adj_list.items.len;

    // Initialize colors
    var colors = try ArrayList(usize).initCapacity(gpa, n);
    errdefer colors.deinit(gpa);
    for (0..n) |_| {
        colors.appendAssumeCapacity(std.math.maxInt(usize));
    }

    // Track saturation degree (number of different colors used by neighbors)
    var saturation = try ArrayList(usize).initCapacity(gpa, n);
    defer saturation.deinit(gpa);
    for (0..n) |_| {
        saturation.appendAssumeCapacity(0);
    }

    // An empty graph has no vertex to seed the coloring with.
    if (n == 0) return colors;

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

    // Update saturation of neighbors
    for (adj_list.items[first_vertex].items) |v_raw| {
        const v: usize = @intCast(v_raw);
        saturation.items[v] = 1;
    }

    // Color the remaining n - 1 vertices; scanning 0..n in index order keeps ties deterministic.
    for (1..n) |_| {
        // Find the uncolored vertex with highest saturation (break ties by degree). Selecting
        // on "no candidate yet" rather than a zero threshold keeps isolated vertices reachable.
        var max_sat: usize = 0;
        var max_deg: usize = 0;
        var next_vertex: ?usize = null;

        for (0..n) |u| {
            if (colors.items[u] != std.math.maxInt(usize)) continue;
            const sat = saturation.items[u];
            const deg = adj_list.items[u].items.len;

            if (next_vertex == null or sat > max_sat or (sat == max_sat and deg > max_deg)) {
                max_sat = sat;
                max_deg = deg;
                next_vertex = u;
            }
        }
        const chosen = next_vertex.?; // Exactly one uncolored vertex disappears per iteration.

        // Color the selected vertex
        var used_colors = std.AutoHashMap(usize, void).init(gpa);
        defer used_colors.deinit();

        for (adj_list.items[chosen].items) |v_raw| {
            const v: usize = @intCast(v_raw);
            if (colors.items[v] != std.math.maxInt(usize)) {
                try used_colors.put(colors.items[v], {});
            }
        }

        // Find smallest unused color
        var color: usize = 0;
        while (used_colors.contains(color)) : (color += 1) {}
        colors.items[chosen] = color;

        // Update saturation of uncolored neighbors
        for (adj_list.items[chosen].items) |v_raw| {
            const v: usize = @intCast(v_raw);
            if (colors.items[v] == std.math.maxInt(usize)) {
                // Count distinct colors used by v's neighbors
                var neighbor_colors = std.AutoHashMap(usize, void).init(gpa);
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
    const gpa = std.testing.allocator;

    // Triangle graph: 0-1-2-0
    var graph = try ArrayList(ArrayList(u32)).initCapacity(gpa, 3);
    defer {
        for (graph.items) |*list| list.deinit(gpa);
        graph.deinit(gpa);
    }

    var list0: ArrayList(u32) = .empty;
    try list0.append(gpa, 1);
    try list0.append(gpa, 2);
    try graph.append(gpa, list0);

    var list1: ArrayList(u32) = .empty;
    try list1.append(gpa, 0);
    try list1.append(gpa, 2);
    try graph.append(gpa, list1);

    var list2: ArrayList(u32) = .empty;
    try list2.append(gpa, 0);
    try list2.append(gpa, 1);
    try graph.append(gpa, list2);

    var coloring = try greedyColoring(u32, gpa, graph);
    defer coloring.deinit(gpa);

    try std.testing.expect(isValidColoring(u32, graph, coloring));
    try std.testing.expectEqual(@as(usize, 3), chromaticNumber(coloring));
}

test "greedy coloring - bipartite graph" {
    const gpa = std.testing.allocator;

    // Complete bipartite K(2,2): 0-2, 0-3, 1-2, 1-3
    var graph = try ArrayList(ArrayList(u32)).initCapacity(gpa, 4);
    defer {
        for (graph.items) |*list| list.deinit(gpa);
        graph.deinit(gpa);
    }

    var list0: ArrayList(u32) = .empty;
    try list0.append(gpa, 2);
    try list0.append(gpa, 3);
    try graph.append(gpa, list0);

    var list1: ArrayList(u32) = .empty;
    try list1.append(gpa, 2);
    try list1.append(gpa, 3);
    try graph.append(gpa, list1);

    var list2: ArrayList(u32) = .empty;
    try list2.append(gpa, 0);
    try list2.append(gpa, 1);
    try graph.append(gpa, list2);

    var list3: ArrayList(u32) = .empty;
    try list3.append(gpa, 0);
    try list3.append(gpa, 1);
    try graph.append(gpa, list3);

    var coloring = try greedyColoring(u32, gpa, graph);
    defer coloring.deinit(gpa);

    try std.testing.expect(isValidColoring(u32, graph, coloring));
    try std.testing.expectEqual(@as(usize, 2), chromaticNumber(coloring));
}

test "welsh-powell - improved coloring" {
    const gpa = std.testing.allocator;

    // Graph where Welsh-Powell outperforms simple greedy
    // 0-1, 0-2, 1-3, 2-3, 3-4
    var graph = try ArrayList(ArrayList(u32)).initCapacity(gpa, 5);
    defer {
        for (graph.items) |*list| list.deinit(gpa);
        graph.deinit(gpa);
    }

    var list0: ArrayList(u32) = .empty;
    try list0.append(gpa, 1);
    try list0.append(gpa, 2);
    try graph.append(gpa, list0);

    var list1: ArrayList(u32) = .empty;
    try list1.append(gpa, 0);
    try list1.append(gpa, 3);
    try graph.append(gpa, list1);

    var list2: ArrayList(u32) = .empty;
    try list2.append(gpa, 0);
    try list2.append(gpa, 3);
    try graph.append(gpa, list2);

    var list3: ArrayList(u32) = .empty;
    try list3.append(gpa, 1);
    try list3.append(gpa, 2);
    try list3.append(gpa, 4);
    try graph.append(gpa, list3);

    var list4: ArrayList(u32) = .empty;
    try list4.append(gpa, 3);
    try graph.append(gpa, list4);

    var coloring = try welshPowell(u32, gpa, graph);
    defer coloring.deinit(gpa);

    try std.testing.expect(isValidColoring(u32, graph, coloring));
    // Should use at most 3 colors (optimal)
    try std.testing.expect(chromaticNumber(coloring) <= 3);
}

test "dsatur - optimal coloring" {
    const gpa = std.testing.allocator;

    // Same graph as welsh-powell test
    var graph = try ArrayList(ArrayList(u32)).initCapacity(gpa, 5);
    defer {
        for (graph.items) |*list| list.deinit(gpa);
        graph.deinit(gpa);
    }

    var list0: ArrayList(u32) = .empty;
    try list0.append(gpa, 1);
    try list0.append(gpa, 2);
    try graph.append(gpa, list0);

    var list1: ArrayList(u32) = .empty;
    try list1.append(gpa, 0);
    try list1.append(gpa, 3);
    try graph.append(gpa, list1);

    var list2: ArrayList(u32) = .empty;
    try list2.append(gpa, 0);
    try list2.append(gpa, 3);
    try graph.append(gpa, list2);

    var list3: ArrayList(u32) = .empty;
    try list3.append(gpa, 1);
    try list3.append(gpa, 2);
    try list3.append(gpa, 4);
    try graph.append(gpa, list3);

    var list4: ArrayList(u32) = .empty;
    try list4.append(gpa, 3);
    try graph.append(gpa, list4);

    // The graph is bipartite (cycle 0-1-3-2 plus pendant 4), so two colors are optimal.
    var coloring = try dsatur(u32, gpa, graph);
    defer coloring.deinit(gpa);

    try std.testing.expect(isValidColoring(u32, graph, coloring));
    try std.testing.expectEqual(@as(usize, 2), chromaticNumber(coloring));
}

test "dsatur - complete graph K5" {
    const gpa = std.testing.allocator;

    // Complete graph K5 requires 5 colors
    var graph = try ArrayList(ArrayList(u32)).initCapacity(gpa, 5);
    defer {
        for (graph.items) |*list| list.deinit(gpa);
        graph.deinit(gpa);
    }

    for (0..5) |i| {
        var list: ArrayList(u32) = .empty;
        for (0..5) |j| {
            if (i != j) try list.append(gpa, @intCast(j));
        }
        try graph.append(gpa, list);
    }

    var coloring = try dsatur(u32, gpa, graph);
    defer coloring.deinit(gpa);

    try std.testing.expect(isValidColoring(u32, graph, coloring));
    try std.testing.expectEqual(@as(usize, 5), chromaticNumber(coloring));
}

test "chromatic number - empty graph" {
    const gpa = std.testing.allocator;
    var colors: ArrayList(usize) = .empty;
    defer colors.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 0), chromaticNumber(colors));
}

test "chromatic number - single color" {
    const gpa = std.testing.allocator;
    var colors = try ArrayList(usize).initCapacity(gpa, 3);
    defer colors.deinit(gpa);

    colors.appendAssumeCapacity(0);
    colors.appendAssumeCapacity(0);
    colors.appendAssumeCapacity(0);

    try std.testing.expectEqual(@as(usize, 1), chromaticNumber(colors));
}

test "isValidColoring - invalid coloring" {
    const gpa = std.testing.allocator;

    // Simple edge 0-1
    var graph = try ArrayList(ArrayList(u32)).initCapacity(gpa, 2);
    defer {
        for (graph.items) |*list| list.deinit(gpa);
        graph.deinit(gpa);
    }

    var list0: ArrayList(u32) = .empty;
    try list0.append(gpa, 1);
    try graph.append(gpa, list0);

    var list1: ArrayList(u32) = .empty;
    try list1.append(gpa, 0);
    try graph.append(gpa, list1);

    // Invalid: both vertices have the same color
    var colors = try ArrayList(usize).initCapacity(gpa, 2);
    defer colors.deinit(gpa);
    colors.appendAssumeCapacity(0);
    colors.appendAssumeCapacity(0);

    try std.testing.expect(!isValidColoring(u32, graph, colors));
}

test "greedy coloring - empty graph" {
    const gpa = std.testing.allocator;

    var graph: ArrayList(ArrayList(u32)) = .empty;
    defer graph.deinit(gpa);

    var coloring = try greedyColoring(u32, gpa, graph);
    defer coloring.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 0), coloring.items.len);
}

test "greedy coloring - single vertex" {
    const gpa = std.testing.allocator;

    var graph = try ArrayList(ArrayList(u32)).initCapacity(gpa, 1);
    defer {
        for (graph.items) |*list| list.deinit(gpa);
        graph.deinit(gpa);
    }

    const list0 = ArrayList(u32).empty;
    try graph.append(gpa, list0);

    var coloring = try greedyColoring(u32, gpa, graph);
    defer coloring.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 1), coloring.items.len);
    try std.testing.expectEqual(@as(usize, 0), coloring.items[0]);
    try std.testing.expectEqual(@as(usize, 1), chromaticNumber(coloring));
}

test "welsh-powell - petersen graph" {
    const gpa = std.testing.allocator;

    // Petersen graph (10 vertices, 15 edges, chromatic number = 3)
    var graph = try ArrayList(ArrayList(u32)).initCapacity(gpa, 10);
    defer {
        for (graph.items) |*list| list.deinit(gpa);
        graph.deinit(gpa);
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
        try graph.append(gpa, ArrayList(u32).empty);
    }

    for (edges) |edge| {
        try graph.items[edge[0]].append(gpa, edge[1]);
        try graph.items[edge[1]].append(gpa, edge[0]);
    }

    var coloring = try welshPowell(u32, gpa, graph);
    defer coloring.deinit(gpa);

    try std.testing.expect(isValidColoring(u32, graph, coloring));
    try std.testing.expectEqual(@as(usize, 3), chromaticNumber(coloring));
}

test "dsatur - cycle graph" {
    const gpa = std.testing.allocator;

    // Cycle C6: 0-1-2-3-4-5-0 (chromatic number = 2)
    var graph = try ArrayList(ArrayList(u32)).initCapacity(gpa, 6);
    defer {
        for (graph.items) |*list| list.deinit(gpa);
        graph.deinit(gpa);
    }

    for (0..6) |i| {
        var list: ArrayList(u32) = .empty;
        try list.append(gpa, @intCast((i + 1) % 6));
        try list.append(gpa, @intCast((i + 5) % 6));
        try graph.append(gpa, list);
    }

    var coloring = try dsatur(u32, gpa, graph);
    defer coloring.deinit(gpa);

    try std.testing.expect(isValidColoring(u32, graph, coloring));
    try std.testing.expectEqual(@as(usize, 2), chromaticNumber(coloring));
}

test "dsatur - empty graph" {
    const gpa = std.testing.allocator;

    var graph: ArrayList(ArrayList(u32)) = .empty;
    defer graph.deinit(gpa);

    var coloring = try dsatur(u32, gpa, graph);
    defer coloring.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 0), coloring.items.len);
    try std.testing.expectEqual(@as(usize, 0), chromaticNumber(coloring));
}

test "welsh-powell - empty graph" {
    const gpa = std.testing.allocator;

    var graph: ArrayList(ArrayList(u32)) = .empty;
    defer graph.deinit(gpa);

    var coloring = try welshPowell(u32, gpa, graph);
    defer coloring.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 0), coloring.items.len);
}

fn color_triangle_with_every_strategy(gpa: Allocator) !void {
    // Fixture graph is built with the testing allocator so only the colorers can fail.
    var graph: ArrayList(ArrayList(u32)) = .empty;
    defer {
        for (graph.items) |*list| list.deinit(std.testing.allocator);
        graph.deinit(std.testing.allocator);
    }
    for (0..3) |u| {
        var list: ArrayList(u32) = .empty;
        errdefer list.deinit(std.testing.allocator);
        for (0..3) |v| {
            if (u != v) try list.append(std.testing.allocator, @intCast(v));
        }
        try graph.append(std.testing.allocator, list);
    }

    var greedy = try greedyColoring(u32, gpa, graph);
    defer greedy.deinit(gpa);
    var welsh = try welshPowell(u32, gpa, graph);
    defer welsh.deinit(gpa);
    var saturated = try dsatur(u32, gpa, graph);
    defer saturated.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 3), chromaticNumber(greedy));
    try std.testing.expectEqual(@as(usize, 3), chromaticNumber(welsh));
    try std.testing.expectEqual(@as(usize, 3), chromaticNumber(saturated));
}

test "coloring - no leaks when an allocation fails" {
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        color_triangle_with_every_strategy,
        .{},
    );
}

test "dsatur - isolated vertices terminate and get color zero" {
    const gpa = std.testing.allocator;

    // Vertex 0-1 share an edge; vertices 2 and 3 are isolated.
    var graph: ArrayList(ArrayList(u32)) = .empty;
    defer {
        for (graph.items) |*list| list.deinit(gpa);
        graph.deinit(gpa);
    }
    for (0..4) |u| {
        var list: ArrayList(u32) = .empty;
        errdefer list.deinit(gpa);
        if (u == 0) try list.append(gpa, 1);
        if (u == 1) try list.append(gpa, 0);
        try graph.append(gpa, list);
    }

    var coloring = try dsatur(u32, gpa, graph);
    defer coloring.deinit(gpa);

    try std.testing.expect(isValidColoring(u32, graph, coloring));
    try std.testing.expectEqual(@as(usize, 2), chromaticNumber(coloring));
    try std.testing.expectEqual(@as(usize, 0), coloring.items[2]);
    try std.testing.expectEqual(@as(usize, 0), coloring.items[3]);
}

test "dsatur - edgeless graph uses one color" {
    const gpa = std.testing.allocator;

    var graph: ArrayList(ArrayList(u32)) = .empty;
    defer {
        for (graph.items) |*list| list.deinit(gpa);
        graph.deinit(gpa);
    }
    for (0..2) |_| try graph.append(gpa, .empty);

    var coloring = try dsatur(u32, gpa, graph);
    defer coloring.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 1), chromaticNumber(coloring));
}
