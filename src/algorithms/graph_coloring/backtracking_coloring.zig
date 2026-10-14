//! Exact graph coloring by backtracking: m-coloring, chromatic number, exact k-coloring, and
//! m-coloring with forward checking.
//!
//! Graphs are adjacency lists (`adj_list.items[u]` = neighbors of `u`). Every search is an
//! explicit iterative depth-first walk over vertices `0..n`; its depth is bounded by `n`.
//! Self loops are ignored. Returned colorings are `ArrayList(usize)` owned by the caller,
//! who frees them with `deinit(gpa)`.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// Sentinel color of a vertex that has not been colored yet.
const uncolored: usize = std.math.maxInt(usize);

/// M-Coloring Problem using Backtracking
/// Determines if a graph can be colored using at most m colors
///
/// Time: O(m^V) worst case (exponential - NP-complete)
/// Space: O(V) for color assignments (the search is iterative)
///
/// Parameters:
///   - T: Vertex ID type (must be integer type)
///   - gpa: Memory allocator
///   - adj_list: Adjacency list representation
///   - m: Maximum number of colors allowed
///
/// Returns:
///   - ArrayList of colors (colors[u] in 0..m) if possible, null otherwise
///
/// Example:
///   ```zig
///   const result = try mColoring(u32, gpa, graph, 3);
///   if (result) |found| {
///       var coloring = found;
///       defer coloring.deinit(gpa);
///       // Use coloring...
///   }
///   ```
pub fn mColoring(
    comptime T: type,
    gpa: Allocator,
    adj_list: ArrayList(ArrayList(T)),
    m: usize,
) !?ArrayList(usize) {
    const n = adj_list.items.len;
    if (n == 0) return .empty;
    if (m == 0) return null;

    var colors = try ArrayList(usize).initCapacity(gpa, n);
    errdefer colors.deinit(gpa);

    colors.appendNTimesAssumeCapacity(uncolored, n);
    assert(colors.items.len == n);

    if (searchColoring(T, adj_list, m, colors.items)) return colors;

    colors.deinit(gpa);
    return null;
}

/// Iterative backtracking over vertices in index order. Fills `colors` (all `uncolored` on
/// entry) and returns true on success; returns false when no coloring with `m` colors exists.
/// Every step either advances one vertex or retreats one vertex and tries a strictly larger
/// color there, so the walk visits each node of the finite search tree at most once.
fn searchColoring(
    comptime T: type,
    adj_list: ArrayList(ArrayList(T)),
    m: usize,
    colors: []usize,
) bool {
    const n = colors.len;
    assert(n == adj_list.items.len);
    assert(m > 0);

    var vertex: usize = 0;
    var color_first: usize = 0;
    while (vertex < n) {
        assert(colors[vertex] == uncolored);

        var color_chosen: ?usize = null;
        for (color_first..m) |color| {
            if (isSafeColor(T, adj_list, vertex, color, colors)) {
                color_chosen = color;
                break;
            }
        }

        if (color_chosen) |color| {
            colors[vertex] = color;
            vertex += 1;
            color_first = 0;
        } else {
            if (vertex == 0) return false;
            vertex -= 1;
            color_first = colors[vertex] + 1;
            colors[vertex] = uncolored;
        }
    }

    assert(vertex == n);
    return true;
}

/// Checks if assigning color to vertex is safe (no conflicts with neighbors)
fn isSafeColor(
    comptime T: type,
    adj_list: ArrayList(ArrayList(T)),
    vertex: usize,
    color: usize,
    colors: []const usize,
) bool {
    assert(vertex < colors.len);
    assert(color != uncolored);

    for (adj_list.items[vertex].items) |neighbor_raw| {
        const neighbor: usize = @intCast(neighbor_raw);
        assert(neighbor < colors.len);
        if (neighbor != vertex and colors[neighbor] == color) return false;
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
pub fn findChromaticNumber(
    comptime T: type,
    gpa: Allocator,
    adj_list: ArrayList(ArrayList(T)),
) !usize {
    const n = adj_list.items.len;
    if (n == 0) return 0;

    // Greedy first-fit never needs more than max_degree + 1 colors, so the search ends there.
    var max_degree: usize = 0;
    for (adj_list.items) |neighbors| {
        if (neighbors.items.len > max_degree) {
            max_degree = neighbors.items.len;
        }
    }
    const upper_bound = max_degree + 1;
    assert(upper_bound >= 1);

    for (1..upper_bound + 1) |m| {
        if (try mColoring(T, gpa, adj_list, m)) |found| {
            var coloring = found;
            defer coloring.deinit(gpa);

            assert(coloring.items.len == n);
            return m;
        }
    }

    return upper_bound; // Fallback
}

/// Exact k-coloring check
/// A graph has a proper coloring that uses exactly k colors if and only if its chromatic
/// number is at most k and k is at most the vertex count: take a proper coloring with the
/// fewest colors and split color classes (each of size >= 2) until k classes exist.
///
/// Time: O(m^V) worst case
/// Space: O(V)
///
/// Returns true if the graph can be colored with exactly k colors
pub fn exactKColoring(
    comptime T: type,
    gpa: Allocator,
    adj_list: ArrayList(ArrayList(T)),
    k: usize,
) !bool {
    const n = adj_list.items.len;
    if (k > n) return false;

    const result = try mColoring(T, gpa, adj_list, k);
    if (result) |found| {
        var coloring = found;
        defer coloring.deinit(gpa);

        assert(coloring.items.len == n);
        return true;
    }
    assert(n > 0);
    return false;
}

/// Graph Coloring with Constraint Propagation
/// Uses forward checking to prune the search space: when a vertex takes a color, that color
/// is blocked for its neighbors, and the branch is abandoned as soon as an uncolored
/// neighbor has no color left.
///
/// Time: O(m^V) worst case, but often much faster than naive backtracking
/// Space: O(V × m) for domain tracking
pub fn coloringWithPropagation(
    comptime T: type,
    gpa: Allocator,
    adj_list: ArrayList(ArrayList(T)),
    m: usize,
) !?ArrayList(usize) {
    const n = adj_list.items.len;
    if (n == 0) return .empty;
    if (m == 0) return null;

    // Any graph is n-colorable, so a palette larger than n only adds search width.
    const palette = @min(m, n);
    assert(palette > 0);

    // blocked[v * palette + c] counts colored neighbors of v that hold color c.
    // Color c is in the domain of v exactly when that count is zero.
    var blocked = try ArrayList(u32).initCapacity(gpa, n * palette);
    defer blocked.deinit(gpa);

    blocked.appendNTimesAssumeCapacity(0, n * palette);

    var colors = try ArrayList(usize).initCapacity(gpa, n);
    errdefer colors.deinit(gpa);

    colors.appendNTimesAssumeCapacity(uncolored, n);

    if (searchPropagating(T, adj_list, palette, colors.items, blocked.items)) return colors;

    colors.deinit(gpa);
    return null;
}

/// Iterative backtracking with forward checking. `colors` is all `uncolored` and `blocked` all
/// zero on entry; on a false return both are back in that state.
fn searchPropagating(
    comptime T: type,
    adj_list: ArrayList(ArrayList(T)),
    palette: usize,
    colors: []usize,
    blocked: []u32,
) bool {
    const n = colors.len;
    assert(n == adj_list.items.len);
    assert(blocked.len == n * palette);

    var vertex: usize = 0;
    var color_first: usize = 0;
    while (vertex < n) {
        assert(colors[vertex] == uncolored);

        var color_chosen: ?usize = null;
        for (color_first..palette) |color| {
            if (blocked[vertex * palette + color] != 0) continue;

            blockColor(T, adj_list, palette, vertex, color, blocked);
            if (neighborsHaveDomain(T, adj_list, palette, vertex, colors, blocked)) {
                color_chosen = color;
                break;
            }
            unblockColor(T, adj_list, palette, vertex, color, blocked);
        }

        if (color_chosen) |color| {
            colors[vertex] = color;
            vertex += 1;
            color_first = 0;
        } else {
            if (vertex == 0) return false;
            vertex -= 1;
            const color_previous = colors[vertex];
            unblockColor(T, adj_list, palette, vertex, color_previous, blocked);
            colors[vertex] = uncolored;
            color_first = color_previous + 1;
        }
    }

    assert(vertex == n);
    return true;
}

/// Removes `color` from the domain of every neighbor of `vertex`.
fn blockColor(
    comptime T: type,
    adj_list: ArrayList(ArrayList(T)),
    palette: usize,
    vertex: usize,
    color: usize,
    blocked: []u32,
) void {
    assert(color < palette);
    assert(vertex < adj_list.items.len);

    for (adj_list.items[vertex].items) |neighbor_raw| {
        const neighbor: usize = @intCast(neighbor_raw);
        if (neighbor == vertex) continue;
        blocked[neighbor * palette + color] += 1;
    }
}

/// Exact inverse of `blockColor`.
fn unblockColor(
    comptime T: type,
    adj_list: ArrayList(ArrayList(T)),
    palette: usize,
    vertex: usize,
    color: usize,
    blocked: []u32,
) void {
    assert(color < palette);
    assert(vertex < adj_list.items.len);

    for (adj_list.items[vertex].items) |neighbor_raw| {
        const neighbor: usize = @intCast(neighbor_raw);
        if (neighbor == vertex) continue;
        assert(blocked[neighbor * palette + color] > 0);
        blocked[neighbor * palette + color] -= 1;
    }
}

/// Forward check: every uncolored neighbor of `vertex` must still have a free color.
fn neighborsHaveDomain(
    comptime T: type,
    adj_list: ArrayList(ArrayList(T)),
    palette: usize,
    vertex: usize,
    colors: []const usize,
    blocked: []const u32,
) bool {
    assert(vertex < colors.len);
    assert(blocked.len == colors.len * palette);

    for (adj_list.items[vertex].items) |neighbor_raw| {
        const neighbor: usize = @intCast(neighbor_raw);
        if (neighbor == vertex or colors[neighbor] != uncolored) continue;

        const row = blocked[neighbor * palette ..][0..palette];
        var has_free_color = false;
        for (row) |count| {
            if (count == 0) {
                has_free_color = true;
                break;
            }
        }
        if (!has_free_color) return false;
    }
    return true;
}

// ============================================================================
// TESTS
// ============================================================================

const triangle_edges = [_][2]u32{ .{ 0, 1 }, .{ 1, 2 }, .{ 0, 2 } };
const k4_edges = [_][2]u32{ .{ 0, 1 }, .{ 0, 2 }, .{ 0, 3 }, .{ 1, 2 }, .{ 1, 3 }, .{ 2, 3 } };
const c5_edges = [_][2]u32{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 4 }, .{ 4, 0 } };
const path4_edges = [_][2]u32{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 } };
const k22_edges = [_][2]u32{ .{ 0, 2 }, .{ 0, 3 }, .{ 1, 2 }, .{ 1, 3 } };

/// Builds an undirected adjacency list with `n` vertices from an edge list.
fn graph_init(gpa: Allocator, n: usize, edges: []const [2]u32) !ArrayList(ArrayList(u32)) {
    var graph = try ArrayList(ArrayList(u32)).initCapacity(gpa, n);
    errdefer graph_deinit(gpa, &graph);

    for (0..n) |_| graph.appendAssumeCapacity(.empty);
    for (edges) |edge| {
        try graph.items[edge[0]].append(gpa, edge[1]);
        try graph.items[edge[1]].append(gpa, edge[0]);
    }
    return graph;
}

fn graph_deinit(gpa: Allocator, graph: *ArrayList(ArrayList(u32))) void {
    for (graph.items) |*list| list.deinit(gpa);
    graph.deinit(gpa);
}

/// Fails unless `coloring` is proper for `graph` and every color is below `palette`.
fn expect_proper(
    graph: ArrayList(ArrayList(u32)),
    coloring: ArrayList(usize),
    palette: usize,
) !void {
    try std.testing.expectEqual(graph.items.len, coloring.items.len);
    for (graph.items, 0..) |neighbors, u| {
        try std.testing.expect(coloring.items[u] < palette);
        for (neighbors.items) |v| {
            try std.testing.expect(coloring.items[u] != coloring.items[v]);
        }
    }
}

test "m-coloring - triangle with 3 colors" {
    const gpa = std.testing.allocator;

    var graph = try graph_init(gpa, 3, &triangle_edges);
    defer graph_deinit(gpa, &graph);

    var coloring = (try mColoring(u32, gpa, graph, 3)) orelse return error.TestUnexpectedResult;
    defer coloring.deinit(gpa);

    try expect_proper(graph, coloring, 3);
    // All three vertices should have different colors
    try std.testing.expect(coloring.items[0] != coloring.items[1]);
    try std.testing.expect(coloring.items[1] != coloring.items[2]);
    try std.testing.expect(coloring.items[0] != coloring.items[2]);
}

test "m-coloring - triangle with 2 colors (impossible)" {
    const gpa = std.testing.allocator;

    var graph = try graph_init(gpa, 3, &triangle_edges);
    defer graph_deinit(gpa, &graph);

    const result = try mColoring(u32, gpa, graph, 2);
    try std.testing.expect(result == null);
}

test "m-coloring - bipartite graph with 2 colors" {
    const gpa = std.testing.allocator;

    // Complete bipartite K(2,2)
    var graph = try graph_init(gpa, 4, &k22_edges);
    defer graph_deinit(gpa, &graph);

    var coloring = (try mColoring(u32, gpa, graph, 2)) orelse return error.TestUnexpectedResult;
    defer coloring.deinit(gpa);

    // Bipartite: partition {0,1} and {2,3}
    try std.testing.expectEqual(coloring.items[0], coloring.items[1]);
    try std.testing.expectEqual(coloring.items[2], coloring.items[3]);
    try std.testing.expect(coloring.items[0] != coloring.items[2]);
}

test "m-coloring - zero colors" {
    const gpa = std.testing.allocator;

    var graph = try graph_init(gpa, 2, &.{});
    defer graph_deinit(gpa, &graph);

    const result = try mColoring(u32, gpa, graph, 0);
    try std.testing.expect(result == null);
}

test "m-coloring - odd cycle needs backtracking past first choices" {
    const gpa = std.testing.allocator;

    var graph = try graph_init(gpa, 5, &c5_edges);
    defer graph_deinit(gpa, &graph);

    const two = try mColoring(u32, gpa, graph, 2);
    try std.testing.expect(two == null);

    var coloring = (try mColoring(u32, gpa, graph, 3)) orelse return error.TestUnexpectedResult;
    defer coloring.deinit(gpa);

    try expect_proper(graph, coloring, 3);
}

test "find chromatic number - triangle" {
    const gpa = std.testing.allocator;

    var graph = try graph_init(gpa, 3, &triangle_edges);
    defer graph_deinit(gpa, &graph);

    const chi = try findChromaticNumber(u32, gpa, graph);
    try std.testing.expectEqual(@as(usize, 3), chi);
}

test "find chromatic number - bipartite" {
    const gpa = std.testing.allocator;

    // Path graph: 0-1-2-3 (bipartite, chromatic number = 2)
    var graph = try graph_init(gpa, 4, &path4_edges);
    defer graph_deinit(gpa, &graph);

    const chi = try findChromaticNumber(u32, gpa, graph);
    try std.testing.expectEqual(@as(usize, 2), chi);
}

test "find chromatic number - K4 and C5" {
    const gpa = std.testing.allocator;

    var k4 = try graph_init(gpa, 4, &k4_edges);
    defer graph_deinit(gpa, &k4);
    try std.testing.expectEqual(@as(usize, 4), try findChromaticNumber(u32, gpa, k4));

    var c5 = try graph_init(gpa, 5, &c5_edges);
    defer graph_deinit(gpa, &c5);
    try std.testing.expectEqual(@as(usize, 3), try findChromaticNumber(u32, gpa, c5));
}

test "exact k-coloring - true case" {
    const gpa = std.testing.allocator;

    // Triangle requires exactly 3 colors
    var graph = try graph_init(gpa, 3, &triangle_edges);
    defer graph_deinit(gpa, &graph);

    try std.testing.expect(try exactKColoring(u32, gpa, graph, 3));
    try std.testing.expect(!try exactKColoring(u32, gpa, graph, 2));
}

test "exact k-coloring - more colors than the chromatic number" {
    const gpa = std.testing.allocator;

    // Path on 4 vertices: chi = 2, and it has proper colorings using exactly 2, 3, or 4 colors
    // (split classes), but none using 1 color or 5 colors.
    var graph = try graph_init(gpa, 4, &path4_edges);
    defer graph_deinit(gpa, &graph);

    try std.testing.expect(!try exactKColoring(u32, gpa, graph, 0));
    try std.testing.expect(!try exactKColoring(u32, gpa, graph, 1));
    try std.testing.expect(try exactKColoring(u32, gpa, graph, 2));
    try std.testing.expect(try exactKColoring(u32, gpa, graph, 3));
    try std.testing.expect(try exactKColoring(u32, gpa, graph, 4));
    try std.testing.expect(!try exactKColoring(u32, gpa, graph, 5));
}

test "coloring with propagation - triangle" {
    const gpa = std.testing.allocator;

    var graph = try graph_init(gpa, 3, &triangle_edges);
    defer graph_deinit(gpa, &graph);

    var coloring = (try coloringWithPropagation(u32, gpa, graph, 3)) orelse
        return error.TestUnexpectedResult;
    defer coloring.deinit(gpa);

    try expect_proper(graph, coloring, 3);
    try std.testing.expect(coloring.items[0] != coloring.items[1]);
    try std.testing.expect(coloring.items[1] != coloring.items[2]);
    try std.testing.expect(coloring.items[0] != coloring.items[2]);
}

test "coloring with propagation - impossible" {
    const gpa = std.testing.allocator;

    // Complete graph K4 with only 3 colors
    var graph = try graph_init(gpa, 4, &k4_edges);
    defer graph_deinit(gpa, &graph);

    const result = try coloringWithPropagation(u32, gpa, graph, 3);
    try std.testing.expect(result == null);
}

test "coloring with propagation - zero colors and surplus colors" {
    const gpa = std.testing.allocator;

    var graph = try graph_init(gpa, 3, &triangle_edges);
    defer graph_deinit(gpa, &graph);

    const none = try coloringWithPropagation(u32, gpa, graph, 0);
    try std.testing.expect(none == null);

    // A palette far larger than the vertex count must not blow up the domain table.
    var coloring = (try coloringWithPropagation(u32, gpa, graph, 1_000_000)) orelse
        return error.TestUnexpectedResult;
    defer coloring.deinit(gpa);

    try expect_proper(graph, coloring, 3);
}

test "m-coloring - empty graph" {
    const gpa = std.testing.allocator;

    var graph: ArrayList(ArrayList(u32)) = .empty;
    defer graph.deinit(gpa);

    var coloring = (try mColoring(u32, gpa, graph, 1)) orelse return error.TestUnexpectedResult;
    defer coloring.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 0), coloring.items.len);
}

test "m-coloring - single vertex" {
    const gpa = std.testing.allocator;

    var graph = try graph_init(gpa, 1, &.{});
    defer graph_deinit(gpa, &graph);

    var coloring = (try mColoring(u32, gpa, graph, 1)) orelse return error.TestUnexpectedResult;
    defer coloring.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 1), coloring.items.len);
    try std.testing.expectEqual(@as(usize, 0), coloring.items[0]);
}

test "find chromatic number - empty graph" {
    const gpa = std.testing.allocator;

    var graph: ArrayList(ArrayList(u32)) = .empty;
    defer graph.deinit(gpa);

    const chi = try findChromaticNumber(u32, gpa, graph);
    try std.testing.expectEqual(@as(usize, 0), chi);
}

test "exact k-coloring - empty graph" {
    const gpa = std.testing.allocator;

    var graph: ArrayList(ArrayList(u32)) = .empty;
    defer graph.deinit(gpa);

    try std.testing.expect(try exactKColoring(u32, gpa, graph, 0));
    try std.testing.expect(!try exactKColoring(u32, gpa, graph, 1));
}

test "coloring with propagation - empty graph" {
    const gpa = std.testing.allocator;

    var graph: ArrayList(ArrayList(u32)) = .empty;
    defer graph.deinit(gpa);

    var coloring = (try coloringWithPropagation(u32, gpa, graph, 2)) orelse
        return error.TestUnexpectedResult;
    defer coloring.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 0), coloring.items.len);
}

test "coloring with propagation - cycle C5" {
    const gpa = std.testing.allocator;

    // Odd cycle C5 requires 3 colors
    var graph = try graph_init(gpa, 5, &c5_edges);
    defer graph_deinit(gpa, &graph);

    const two = try coloringWithPropagation(u32, gpa, graph, 2);
    try std.testing.expect(two == null);

    var coloring = (try coloringWithPropagation(u32, gpa, graph, 3)) orelse
        return error.TestUnexpectedResult;
    defer coloring.deinit(gpa);

    // Verify no adjacent vertices have the same color
    try expect_proper(graph, coloring, 3);
}

fn color_with_every_function(gpa: Allocator) !void {
    // The fixture graph uses the testing allocator so only the functions under test can fail.
    var graph = try graph_init(std.testing.allocator, 5, &c5_edges);
    defer graph_deinit(std.testing.allocator, &graph);

    var three = (try mColoring(u32, gpa, graph, 3)) orelse return error.TestUnexpectedResult;
    defer three.deinit(gpa);
    try expect_proper(graph, three, 3);

    const two = try mColoring(u32, gpa, graph, 2);
    try std.testing.expect(two == null);

    try std.testing.expectEqual(@as(usize, 3), try findChromaticNumber(u32, gpa, graph));
    try std.testing.expect(try exactKColoring(u32, gpa, graph, 4));
    try std.testing.expect(!try exactKColoring(u32, gpa, graph, 2));

    var pruned = (try coloringWithPropagation(u32, gpa, graph, 3)) orelse
        return error.TestUnexpectedResult;
    defer pruned.deinit(gpa);
    try expect_proper(graph, pruned, 3);

    const pruned_none = try coloringWithPropagation(u32, gpa, graph, 2);
    try std.testing.expect(pruned_none == null);
}

test "backtracking coloring - no leaks when an allocation fails" {
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        color_with_every_function,
        .{},
    );
}
