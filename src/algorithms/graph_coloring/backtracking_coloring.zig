const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// M-Coloring Problem using Backtracking
/// Determines if a graph can be colored using at most m colors
///
/// Time: O(m^V) worst case (exponential - NP-complete)
/// Space: O(V) for recursion stack and color assignments
///
/// Parameters:
///   - T: Vertex ID type (must be integer type)
///   - allocator: Memory allocator
///   - adj_list: Adjacency list representation
///   - m: Maximum number of colors allowed
///
/// Returns:
///   - ArrayList of colors if possible, null otherwise
///
/// Example:
///   ```zig
///   const result = try mColoring(u32, allocator, graph, 3);
///   if (result) |coloring| {
///       defer coloring.deinit();
///       // Use coloring...
///   }
///   ```
pub fn mColoring(comptime T: type, allocator: Allocator, adj_list: ArrayList(ArrayList(T)), m: usize) !?ArrayList(usize) {
    const n = adj_list.items.len;
    if (n == 0) return ArrayList(usize).init(allocator);
    if (m == 0) return null;

    var colors = try ArrayList(usize).initCapacity(allocator, n);
    errdefer colors.deinit();

    // Initialize all vertices as uncolored
    for (0..n) |_| {
        colors.appendAssumeCapacity(std.math.maxInt(usize));
    }

    // Try to color all vertices
    if (try mColoringHelper(T, adj_list, m, 0, &colors)) {
        return colors;
    } else {
        colors.deinit();
        return null;
    }
}

/// Helper function for m-coloring backtracking
fn mColoringHelper(comptime T: type, adj_list: ArrayList(ArrayList(T)), m: usize, vertex: usize, colors: *ArrayList(usize)) !bool {
    const n = adj_list.items.len;

    // Base case: all vertices are colored
    if (vertex == n) return true;

    // Try all colors for current vertex
    for (0..m) |color| {
        if (isSafeColor(T, adj_list, vertex, color, colors)) {
            colors.items[vertex] = color;

            // Recursively color remaining vertices
            if (try mColoringHelper(T, adj_list, m, vertex + 1, colors)) {
                return true;
            }

            // Backtrack
            colors.items[vertex] = std.math.maxInt(usize);
        }
    }

    return false;
}

/// Checks if assigning color to vertex is safe (no conflicts with neighbors)
fn isSafeColor(comptime T: type, adj_list: ArrayList(ArrayList(T)), vertex: usize, color: usize, colors: *ArrayList(usize)) bool {
    for (adj_list.items[vertex].items) |neighbor_raw| {
        const neighbor: usize = @intCast(neighbor_raw);
        if (colors.items[neighbor] == color) return false;
    }
    return true;
}

/// Finds the chromatic number (minimum colors needed) using backtracking
/// Tries m=1, 2, 3, ... until a valid coloring is found
///
/// Time: O(V^(V+1)) worst case (very expensive)
/// Space: O(V)
///
/// Use with caution - only for small graphs (V < 15)
pub fn findChromaticNumber(comptime T: type, allocator: Allocator, adj_list: ArrayList(ArrayList(T))) !usize {
    const n = adj_list.items.len;
    if (n == 0) return 0;

    // Upper bound: use simple greedy to get an upper bound
    var max_degree: usize = 0;
    for (adj_list.items) |neighbors| {
        if (neighbors.items.len > max_degree) {
            max_degree = neighbors.items.len;
        }
    }
    const upper_bound = max_degree + 1;

    // Try colors from 1 to upper_bound
    for (1..upper_bound + 1) |m| {
        if (try mColoring(T, allocator, adj_list, m)) |coloring| {
            defer coloring.deinit();
            return m;
        }
    }

    return upper_bound; // Fallback
}

/// Hamiltonian Cycle-based Graph Coloring
/// Uses the fact that finding optimal coloring is related to finding cliques
///
/// Time: O(m^V) worst case
/// Space: O(V)
///
/// Returns true if the graph can be colored with exactly k colors
pub fn exactKColoring(comptime T: type, allocator: Allocator, adj_list: ArrayList(ArrayList(T)), k: usize) !bool {
    const result = try mColoring(T, allocator, adj_list, k);
    if (result) |coloring| {
        defer coloring.deinit();

        // Verify that all k colors are actually used
        var colors_used = std.AutoHashMap(usize, void).init(allocator);
        defer colors_used.deinit();

        for (coloring.items) |color| {
            if (color != std.math.maxInt(usize)) {
                try colors_used.put(color, {});
            }
        }

        return colors_used.count() == k;
    }
    return false;
}

/// Graph Coloring with Constraint Propagation
/// Uses forward checking to prune the search space
///
/// Time: O(m^V) worst case, but often much faster than naive backtracking
/// Space: O(V × m) for domain tracking
pub fn coloringWithPropagation(comptime T: type, allocator: Allocator, adj_list: ArrayList(ArrayList(T)), m: usize) !?ArrayList(usize) {
    const n = adj_list.items.len;
    if (n == 0) return ArrayList(usize).init(allocator);
    if (m == 0) return null;

    // Initialize color domains (possible colors for each vertex)
    var domains = try ArrayList(std.AutoHashMap(usize, void)).initCapacity(allocator, n);
    defer {
        for (domains.items) |*domain| domain.deinit();
        domains.deinit();
    }

    for (0..n) |_| {
        var domain = std.AutoHashMap(usize, void).init(allocator);
        for (0..m) |color| {
            try domain.put(color, {});
        }
        domains.appendAssumeCapacity(domain);
    }

    var colors = try ArrayList(usize).initCapacity(allocator, n);
    errdefer colors.deinit();
    for (0..n) |_| {
        colors.appendAssumeCapacity(std.math.maxInt(usize));
    }

    if (try propagateHelper(T, adj_list, 0, &colors, &domains)) {
        return colors;
    } else {
        colors.deinit();
        return null;
    }
}

fn propagateHelper(comptime T: type, adj_list: ArrayList(ArrayList(T)), vertex: usize, colors: *ArrayList(usize), domains: *ArrayList(std.AutoHashMap(usize, void))) !bool {
    const n = adj_list.items.len;
    if (vertex == n) return true;

    // Try each color in the domain
    var it = domains.items[vertex].keyIterator();
    while (it.next()) |color_ptr| {
        const color = color_ptr.*;

        if (isSafeColor(T, adj_list, vertex, color, colors)) {
            colors.items[vertex] = color;

            // Forward check: remove color from neighbors' domains
            var removed = ArrayList(struct { neighbor: usize, color: usize }).init(domains.items[0].allocator);
            defer removed.deinit();

            for (adj_list.items[vertex].items) |neighbor_raw| {
                const neighbor: usize = @intCast(neighbor_raw);
                if (colors.items[neighbor] == std.math.maxInt(usize)) {
                    if (domains.items[neighbor].remove(color)) {
                        try removed.append(.{ .neighbor = neighbor, .color = color });
                    }

                    // Check if domain became empty
                    if (domains.items[neighbor].count() == 0) {
                        // Restore domains
                        for (removed.items) |item| {
                            try domains.items[item.neighbor].put(item.color, {});
                        }
                        colors.items[vertex] = std.math.maxInt(usize);
                        continue; // Try next color
                    }
                }
            }

            // Recursively color remaining vertices
            if (try propagateHelper(T, adj_list, vertex + 1, colors, domains)) {
                return true;
            }

            // Backtrack: restore domains
            for (removed.items) |item| {
                try domains.items[item.neighbor].put(item.color, {});
            }
            colors.items[vertex] = std.math.maxInt(usize);
        }
    }

    return false;
}

// ============================================================================
// TESTS
// ============================================================================

test "m-coloring - triangle with 3 colors" {
    const allocator = std.testing.allocator;

    // Triangle: 0-1-2-0
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

    const result = try mColoring(u32, allocator, graph, 3);
    try std.testing.expect(result != null);

    if (result) |coloring| {
        defer coloring.deinit();
        // All three vertices should have different colors
        try std.testing.expect(coloring.items[0] != coloring.items[1]);
        try std.testing.expect(coloring.items[1] != coloring.items[2]);
        try std.testing.expect(coloring.items[0] != coloring.items[2]);
    }
}

test "m-coloring - triangle with 2 colors (impossible)" {
    const allocator = std.testing.allocator;

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

    const result = try mColoring(u32, allocator, graph, 2);
    try std.testing.expect(result == null);
}

test "m-coloring - bipartite graph with 2 colors" {
    const allocator = std.testing.allocator;

    // Complete bipartite K(2,2)
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

    const result = try mColoring(u32, allocator, graph, 2);
    try std.testing.expect(result != null);

    if (result) |coloring| {
        defer coloring.deinit();
        // Bipartite: partition {0,1} and {2,3}
        try std.testing.expectEqual(coloring.items[0], coloring.items[1]);
        try std.testing.expectEqual(coloring.items[2], coloring.items[3]);
        try std.testing.expect(coloring.items[0] != coloring.items[2]);
    }
}

test "find chromatic number - triangle" {
    const allocator = std.testing.allocator;

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

    const chi = try findChromaticNumber(u32, allocator, graph);
    try std.testing.expectEqual(@as(usize, 3), chi);
}

test "find chromatic number - bipartite" {
    const allocator = std.testing.allocator;

    // Path graph: 0-1-2-3 (bipartite, chromatic number = 2)
    var graph = try ArrayList(ArrayList(u32)).initCapacity(allocator, 4);
    defer {
        for (graph.items) |*list| list.deinit();
        graph.deinit();
    }

    var list0 = ArrayList(u32).init(allocator);
    try list0.append(1);
    try graph.append(list0);

    var list1 = ArrayList(u32).init(allocator);
    try list1.append(0);
    try list1.append(2);
    try graph.append(list1);

    var list2 = ArrayList(u32).init(allocator);
    try list2.append(1);
    try list2.append(3);
    try graph.append(list2);

    var list3 = ArrayList(u32).init(allocator);
    try list3.append(2);
    try graph.append(list3);

    const chi = try findChromaticNumber(u32, allocator, graph);
    try std.testing.expectEqual(@as(usize, 2), chi);
}

test "exact k-coloring - true case" {
    const allocator = std.testing.allocator;

    // Triangle requires exactly 3 colors
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

    try std.testing.expect(try exactKColoring(u32, allocator, graph, 3));
    try std.testing.expect(!try exactKColoring(u32, allocator, graph, 2));
}

test "coloring with propagation - triangle" {
    const allocator = std.testing.allocator;

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

    const result = try coloringWithPropagation(u32, allocator, graph, 3);
    try std.testing.expect(result != null);

    if (result) |coloring| {
        defer coloring.deinit();
        try std.testing.expect(coloring.items[0] != coloring.items[1]);
        try std.testing.expect(coloring.items[1] != coloring.items[2]);
        try std.testing.expect(coloring.items[0] != coloring.items[2]);
    }
}

test "coloring with propagation - impossible" {
    const allocator = std.testing.allocator;

    // Complete graph K4 with only 3 colors
    var graph = try ArrayList(ArrayList(u32)).initCapacity(allocator, 4);
    defer {
        for (graph.items) |*list| list.deinit();
        graph.deinit();
    }

    for (0..4) |i| {
        var list = ArrayList(u32).init(allocator);
        for (0..4) |j| {
            if (i != j) try list.append(@intCast(j));
        }
        try graph.append(list);
    }

    const result = try coloringWithPropagation(u32, allocator, graph, 3);
    try std.testing.expect(result == null);
}

test "m-coloring - empty graph" {
    const allocator = std.testing.allocator;

    var graph = ArrayList(ArrayList(u32)).init(allocator);
    defer graph.deinit();

    const result = try mColoring(u32, allocator, graph, 1);
    try std.testing.expect(result != null);

    if (result) |coloring| {
        defer coloring.deinit();
        try std.testing.expectEqual(@as(usize, 0), coloring.items.len);
    }
}

test "m-coloring - single vertex" {
    const allocator = std.testing.allocator;

    var graph = try ArrayList(ArrayList(u32)).initCapacity(allocator, 1);
    defer {
        for (graph.items) |*list| list.deinit();
        graph.deinit();
    }

    const list0 = ArrayList(u32).init(allocator);
    try graph.append(list0);

    const result = try mColoring(u32, allocator, graph, 1);
    try std.testing.expect(result != null);

    if (result) |coloring| {
        defer coloring.deinit();
        try std.testing.expectEqual(@as(usize, 1), coloring.items.len);
        try std.testing.expectEqual(@as(usize, 0), coloring.items[0]);
    }
}

test "find chromatic number - empty graph" {
    const allocator = std.testing.allocator;

    var graph = ArrayList(ArrayList(u32)).init(allocator);
    defer graph.deinit();

    const chi = try findChromaticNumber(u32, allocator, graph);
    try std.testing.expectEqual(@as(usize, 0), chi);
}

test "coloring with propagation - cycle C5" {
    const allocator = std.testing.allocator;

    // Odd cycle C5 requires 3 colors
    var graph = try ArrayList(ArrayList(u32)).initCapacity(allocator, 5);
    defer {
        for (graph.items) |*list| list.deinit();
        graph.deinit();
    }

    for (0..5) |i| {
        var list = ArrayList(u32).init(allocator);
        try list.append(@intCast((i + 1) % 5));
        try list.append(@intCast((i + 4) % 5));
        try graph.append(list);
    }

    const result = try coloringWithPropagation(u32, allocator, graph, 3);
    try std.testing.expect(result != null);

    if (result) |coloring| {
        defer coloring.deinit();
        // Verify no adjacent vertices have the same color
        for (graph.items, 0..) |neighbors, u| {
            for (neighbors.items) |v_raw| {
                const v: usize = @intCast(v_raw);
                try std.testing.expect(coloring.items[u] != coloring.items[v]);
            }
        }
    }
}
