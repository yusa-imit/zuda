const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// Vertex Cover Approximation Algorithms
///
/// Vertex cover: a set of vertices such that every edge has at least one endpoint in the set.
/// Finding minimum vertex cover is NP-complete, but we can approximate it efficiently.

/// Computes a 2-approximation of minimum vertex cover using greedy edge selection.
/// Repeatedly picks an uncovered edge and adds both endpoints to the cover.
///
/// Time: O(V + E) where V = number of vertices, E = number of edges
/// Space: O(V) for the cover set
///
/// Returns: ArrayList of vertex indices in the cover (caller owns memory)
///
/// Algorithm:
/// 1. While there are uncovered edges:
///    - Pick an arbitrary uncovered edge (u, v)
///    - Add both u and v to the cover
///    - Mark all edges incident to u or v as covered
/// 2. Approximation ratio: |C| ≤ 2 × |OPT|
///
/// Example:
/// ```zig
/// var edges = ArrayList([2]usize).init(allocator);
/// try edges.append(.{0, 1});
/// try edges.append(.{1, 2});
/// try edges.append(.{2, 3});
/// var cover = try vertexCoverApprox(allocator, 4, edges.items);
/// defer cover.deinit();
/// // cover contains vertices that cover all edges (e.g., {1, 2})
/// ```
pub fn vertexCoverApprox(allocator: Allocator, num_vertices: usize, edges: [][2]usize) !ArrayList(usize) {
    var cover = ArrayList(usize).init(allocator);
    errdefer cover.deinit();

    if (edges.len == 0) return cover;

    // Track which vertices are in the cover
    var in_cover = try allocator.alloc(bool, num_vertices);
    defer allocator.free(in_cover);
    @memset(in_cover, false);

    // Track which edges are covered
    var edge_covered = try allocator.alloc(bool, edges.len);
    defer allocator.free(edge_covered);
    @memset(edge_covered, false);

    // Greedy: pick uncovered edges and add both endpoints
    for (edges, 0..) |edge, i| {
        if (!edge_covered[i]) {
            const u = edge[0];
            const v = edge[1];

            // Add both endpoints to cover
            if (!in_cover[u]) {
                try cover.append(u);
                in_cover[u] = true;
            }
            if (!in_cover[v]) {
                try cover.append(v);
                in_cover[v] = true;
            }

            // Mark all edges incident to u or v as covered
            for (edges, 0..) |e, j| {
                if (e[0] == u or e[0] == v or e[1] == u or e[1] == v) {
                    edge_covered[j] = true;
                }
            }
        }
    }

    return cover;
}

/// Computes vertex cover using a degree-based greedy heuristic (not guaranteed approximation).
/// Repeatedly picks the vertex with highest degree and adds it to the cover.
///
/// Time: O(V × E) for repeated degree calculations
/// Space: O(V) for the cover set
///
/// Returns: ArrayList of vertex indices in the cover (caller owns memory)
///
/// Note: This is a greedy heuristic with no theoretical approximation guarantee,
/// but often performs well in practice. Use vertexCoverApprox for guaranteed 2-approximation.
///
/// Example:
/// ```zig
/// var edges = ArrayList([2]usize).init(allocator);
/// try edges.append(.{0, 1});
/// try edges.append(.{0, 2});
/// try edges.append(.{1, 2});
/// var cover = try vertexCoverGreedy(allocator, 3, edges.items);
/// defer cover.deinit();
/// // cover likely contains vertex 0 or 1 (highest degree)
/// ```
pub fn vertexCoverGreedy(allocator: Allocator, num_vertices: usize, edges: [][2]usize) !ArrayList(usize) {
    var cover = ArrayList(usize).init(allocator);
    errdefer cover.deinit();

    if (edges.len == 0) return cover;

    // Track which edges are covered
    var edge_covered = try allocator.alloc(bool, edges.len);
    defer allocator.free(edge_covered);
    @memset(edge_covered, false);

    var uncovered_count = edges.len;

    while (uncovered_count > 0) {
        // Find vertex with maximum degree among uncovered edges
        var degrees = try allocator.alloc(usize, num_vertices);
        defer allocator.free(degrees);
        @memset(degrees, 0);

        for (edges, 0..) |edge, i| {
            if (!edge_covered[i]) {
                degrees[edge[0]] += 1;
                degrees[edge[1]] += 1;
            }
        }

        // Find vertex with max degree
        var max_degree: usize = 0;
        var max_vertex: usize = 0;
        for (degrees, 0..) |deg, v| {
            if (deg > max_degree) {
                max_degree = deg;
                max_vertex = v;
            }
        }

        if (max_degree == 0) break;

        // Add vertex to cover
        try cover.append(max_vertex);

        // Mark edges incident to this vertex as covered
        for (edges, 0..) |edge, i| {
            if (!edge_covered[i] and (edge[0] == max_vertex or edge[1] == max_vertex)) {
                edge_covered[i] = true;
                uncovered_count -= 1;
            }
        }
    }

    return cover;
}

/// Validates that the given vertex cover actually covers all edges.
///
/// Time: O(E) to check all edges
/// Space: O(1)
///
/// Returns: true if all edges are covered, false otherwise
pub fn isValidCover(edges: [][2]usize, cover: []const usize) bool {
    // Check each edge has at least one endpoint in cover
    for (edges) |edge| {
        var covered = false;
        for (cover) |v| {
            if (v == edge[0] or v == edge[1]) {
                covered = true;
                break;
            }
        }
        if (!covered) return false;
    }
    return true;
}

// ============================================================================
// Tests
// ============================================================================

test "vertex cover: empty graph" {
    const allocator = std.testing.allocator;
    const edges: [][2]usize = &.{};
    var cover = try vertexCoverApprox(allocator, 0, edges);
    defer cover.deinit();
    try std.testing.expectEqual(@as(usize, 0), cover.items.len);
}

test "vertex cover: single edge" {
    const allocator = std.testing.allocator;
    const edges = [_][2]usize{.{ 0, 1 }};
    var cover = try vertexCoverApprox(allocator, 2, &edges);
    defer cover.deinit();

    try std.testing.expect(cover.items.len >= 1 and cover.items.len <= 2);
    try std.testing.expect(isValidCover(&edges, cover.items));
}

test "vertex cover: triangle graph" {
    const allocator = std.testing.allocator;
    const edges = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 0 } };
    var cover = try vertexCoverApprox(allocator, 3, &edges);
    defer cover.deinit();

    // Triangle requires at least 2 vertices (OPT=2), approx gives ≤4
    try std.testing.expect(cover.items.len >= 2);
    try std.testing.expect(cover.items.len <= 4);
    try std.testing.expect(isValidCover(&edges, cover.items));
}

test "vertex cover: path graph" {
    const allocator = std.testing.allocator;
    // Path: 0-1-2-3-4
    const edges = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 4 } };
    var cover = try vertexCoverApprox(allocator, 5, &edges);
    defer cover.deinit();

    // Path of length 4 requires OPT=2 (e.g., {1,3}), approx gives ≤4
    try std.testing.expect(cover.items.len >= 2);
    try std.testing.expect(cover.items.len <= 4);
    try std.testing.expect(isValidCover(&edges, cover.items));
}

test "vertex cover: star graph" {
    const allocator = std.testing.allocator;
    // Star: center 0 connected to 1,2,3,4
    const edges = [_][2]usize{ .{ 0, 1 }, .{ 0, 2 }, .{ 0, 3 }, .{ 0, 4 } };
    var cover = try vertexCoverApprox(allocator, 5, &edges);
    defer cover.deinit();

    // Star requires OPT=1 (center), approx may give 2-4
    try std.testing.expect(cover.items.len >= 1);
    try std.testing.expect(isValidCover(&edges, cover.items));
}

test "vertex cover: complete graph K4" {
    const allocator = std.testing.allocator;
    const edges = [_][2]usize{
        .{ 0, 1 }, .{ 0, 2 }, .{ 0, 3 },
        .{ 1, 2 }, .{ 1, 3 }, .{ 2, 3 },
    };
    var cover = try vertexCoverApprox(allocator, 4, &edges);
    defer cover.deinit();

    // K4 requires OPT=3, approx gives ≤6
    try std.testing.expect(cover.items.len >= 3);
    try std.testing.expect(cover.items.len <= 6);
    try std.testing.expect(isValidCover(&edges, cover.items));
}

test "vertex cover: disconnected components" {
    const allocator = std.testing.allocator;
    // Two triangles: 0-1-2 and 3-4-5
    const edges = [_][2]usize{
        .{ 0, 1 }, .{ 1, 2 }, .{ 2, 0 },
        .{ 3, 4 }, .{ 4, 5 }, .{ 5, 3 },
    };
    var cover = try vertexCoverApprox(allocator, 6, &edges);
    defer cover.deinit();

    // Two triangles require OPT=4, approx gives ≤8
    try std.testing.expect(cover.items.len >= 4);
    try std.testing.expect(isValidCover(&edges, cover.items));
}

test "vertex cover: greedy heuristic - star" {
    const allocator = std.testing.allocator;
    // Star: center 0 connected to 1,2,3,4
    const edges = [_][2]usize{ .{ 0, 1 }, .{ 0, 2 }, .{ 0, 3 }, .{ 0, 4 } };
    var cover = try vertexCoverGreedy(allocator, 5, &edges);
    defer cover.deinit();

    // Greedy should pick center (highest degree) - optimal solution
    try std.testing.expect(cover.items.len >= 1);
    try std.testing.expect(isValidCover(&edges, cover.items));
}

test "vertex cover: greedy vs approx comparison" {
    const allocator = std.testing.allocator;
    const edges = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 4 } };

    var cover_approx = try vertexCoverApprox(allocator, 5, &edges);
    defer cover_approx.deinit();

    var cover_greedy = try vertexCoverGreedy(allocator, 5, &edges);
    defer cover_greedy.deinit();

    // Both should be valid covers
    try std.testing.expect(isValidCover(&edges, cover_approx.items));
    try std.testing.expect(isValidCover(&edges, cover_greedy.items));
}

test "vertex cover: large path (stress test)" {
    const allocator = std.testing.allocator;

    // Path of length 100: 0-1-2-...-100
    var edges = ArrayList([2]usize).init(allocator);
    defer edges.deinit();

    for (0..100) |i| {
        try edges.append(.{ i, i + 1 });
    }

    var cover = try vertexCoverApprox(allocator, 101, edges.items);
    defer cover.deinit();

    // Path of length 100 requires OPT=50, approx gives ≤100
    try std.testing.expect(cover.items.len >= 50);
    try std.testing.expect(cover.items.len <= 100);
    try std.testing.expect(isValidCover(edges.items, cover.items));
}

test "vertex cover: isValidCover correctly identifies invalid cover" {
    const edges = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 } };
    const invalid_cover = [_]usize{ 0, 3 }; // Missing vertex 1 or 2
    try std.testing.expect(!isValidCover(&edges, &invalid_cover));
}

test "vertex cover: isValidCover accepts minimal valid cover" {
    const edges = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 } };
    const valid_cover = [_]usize{ 1, 2 }; // Covers all edges
    try std.testing.expect(isValidCover(&edges, &valid_cover));
}
