//! Vertex cover approximation algorithms.
//!
//! Vertex cover: a set of vertices such that every edge has at least one endpoint in the set.
//! The minimum is NP-complete; `vertexCoverApprox` is a 2-approximation and
//! `vertexCoverGreedy` is a degree heuristic without a guarantee.
//!
//! Edges are `[2]usize` endpoint pairs and are user data: an endpoint at or above `num_vertices`
//! returns `error.InvalidEdge` before anything is allocated. Self-loops are allowed.
//!
//! Ownership: returned `ArrayList(usize)` values are owned by the caller, who releases them with
//! `deinit(gpa)` using the same allocator. On error nothing is leaked.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

pub const Error = error{InvalidEdge} || Allocator.Error;

/// Returns `error.InvalidEdge` when any endpoint is not below `num_vertices`.
fn edges_validate(num_vertices: usize, edges: []const [2]usize) error{InvalidEdge}!void {
    for (edges) |edge| {
        if (edge[0] >= num_vertices) return error.InvalidEdge;
        if (edge[1] >= num_vertices) return error.InvalidEdge;
    }
}

/// Computes a 2-approximation of minimum vertex cover using greedy edge selection.
/// Repeatedly picks an uncovered edge and adds both endpoints to the cover.
///
/// Time: O(V + E) where V = number of vertices, E = number of edges
/// Space: O(V) for the cover set
///
/// Returns: ArrayList of vertex indices in the cover (caller owns memory)
///
/// Algorithm:
/// 1. Scan the edges in order; an edge whose endpoints are both outside the cover is uncovered
///    - Add both endpoints to the cover (this also covers every edge incident to them)
/// 2. The picked edges form a matching and OPT needs one endpoint of each, so |C| ≤ 2 × |OPT|
///
/// Example:
/// ```zig
/// const edges = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 } };
/// var cover = try vertexCoverApprox(gpa, 4, &edges);
/// defer cover.deinit(gpa);
/// // cover contains vertices that cover all edges (here {0, 1, 2, 3})
/// ```
pub fn vertexCoverApprox(
    gpa: Allocator,
    num_vertices: usize,
    edges: []const [2]usize,
) Error!ArrayList(usize) {
    try edges_validate(num_vertices, edges);

    var cover: ArrayList(usize) = .empty;
    errdefer cover.deinit(gpa);

    if (edges.len == 0) return cover;

    // Track which vertices are in the cover
    const in_cover = try gpa.alloc(bool, num_vertices);
    defer gpa.free(in_cover);

    @memset(in_cover, false);

    // An edge is covered exactly when one of its endpoints is already in the cover.
    for (edges) |edge| {
        const u = edge[0];
        const v = edge[1];
        // Postcondition of `edges_validate` on this path.
        assert(u < num_vertices);
        assert(v < num_vertices);
        if (in_cover[u] or in_cover[v]) continue;

        // Add both endpoints to cover (a self-loop adds one vertex).
        try cover.append(gpa, u);
        in_cover[u] = true;
        if (v != u) {
            try cover.append(gpa, v);
            in_cover[v] = true;
        }
    }

    assert(cover.items.len <= num_vertices);
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
/// const edges = [_][2]usize{ .{ 0, 1 }, .{ 0, 2 }, .{ 1, 2 } };
/// var cover = try vertexCoverGreedy(gpa, 3, &edges);
/// defer cover.deinit(gpa);
/// // cover likely contains vertex 0 or 1 (highest degree)
/// ```
pub fn vertexCoverGreedy(
    gpa: Allocator,
    num_vertices: usize,
    edges: []const [2]usize,
) Error!ArrayList(usize) {
    try edges_validate(num_vertices, edges);

    var cover: ArrayList(usize) = .empty;
    errdefer cover.deinit(gpa);

    if (edges.len == 0) return cover;

    // Track which edges are covered
    const edge_covered = try gpa.alloc(bool, edges.len);
    defer gpa.free(edge_covered);

    @memset(edge_covered, false);

    const degrees = try gpa.alloc(usize, num_vertices);
    defer gpa.free(degrees);

    var uncovered_count = edges.len;

    // Every pass covers at least one edge, so `edges.len` bounds the loop.
    for (0..edges.len) |_| {
        if (uncovered_count == 0) break;

        // Find vertex with maximum degree among uncovered edges
        @memset(degrees, 0);
        for (edges, 0..) |edge, i| {
            if (edge_covered[i]) continue;
            // Postcondition of `edges_validate` on this path.
            assert(edge[0] < num_vertices);
            assert(edge[1] < num_vertices);
            degrees[edge[0]] += 1;
            degrees[edge[1]] += 1;
        }

        var max_degree: usize = 0;
        var max_vertex: usize = 0;
        for (degrees, 0..) |deg, v| {
            if (deg > max_degree) {
                max_degree = deg;
                max_vertex = v;
            }
        }
        // An uncovered edge contributes to the degree of its endpoints.
        assert(max_degree > 0);

        // Add vertex to cover
        try cover.append(gpa, max_vertex);

        // Mark edges incident to this vertex as covered
        for (edges, 0..) |edge, i| {
            if (!edge_covered[i] and (edge[0] == max_vertex or edge[1] == max_vertex)) {
                edge_covered[i] = true;
                uncovered_count -= 1;
            }
        }
    }

    assert(uncovered_count == 0);
    return cover;
}

/// Validates that the given vertex cover actually covers all edges.
///
/// Time: O(E) to check all edges
/// Space: O(1)
///
/// Returns: true if all edges are covered, false otherwise
pub fn isValidCover(edges: []const [2]usize, cover: []const usize) bool {
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
    const gpa = std.testing.allocator;
    const edges: []const [2]usize = &.{};
    var cover = try vertexCoverApprox(gpa, 0, edges);
    defer cover.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 0), cover.items.len);
}

test "vertex cover: single edge" {
    const gpa = std.testing.allocator;
    const edges = [_][2]usize{.{ 0, 1 }};
    var cover = try vertexCoverApprox(gpa, 2, &edges);
    defer cover.deinit(gpa);

    // The single edge is picked, so both endpoints enter the cover.
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, cover.items);
    try std.testing.expect(isValidCover(&edges, cover.items));
}

test "vertex cover: triangle graph" {
    const gpa = std.testing.allocator;
    const edges = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 0 } };
    var cover = try vertexCoverApprox(gpa, 3, &edges);
    defer cover.deinit(gpa);

    // Edge (0,1) is picked and covers the whole triangle: OPT=2, approx gives exactly 2.
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, cover.items);
    try std.testing.expect(isValidCover(&edges, cover.items));
}

test "vertex cover: path graph" {
    const gpa = std.testing.allocator;
    // Path: 0-1-2-3-4
    const edges = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 4 } };
    var cover = try vertexCoverApprox(gpa, 5, &edges);
    defer cover.deinit(gpa);

    // Matching edges (0,1) and (2,3) are picked: 4 vertices against OPT=2 ({1,3}), ratio 2.
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3 }, cover.items);
    try std.testing.expect(isValidCover(&edges, cover.items));
}

test "vertex cover: star graph" {
    const gpa = std.testing.allocator;
    // Star: center 0 connected to 1,2,3,4
    const edges = [_][2]usize{ .{ 0, 1 }, .{ 0, 2 }, .{ 0, 3 }, .{ 0, 4 } };
    var cover = try vertexCoverApprox(gpa, 5, &edges);
    defer cover.deinit(gpa);

    // Star requires OPT=1 (center); edge (0,1) is picked and covers every edge: 2 = 2 x OPT.
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, cover.items);
    try std.testing.expect(isValidCover(&edges, cover.items));
}

test "vertex cover: complete graph K4" {
    const gpa = std.testing.allocator;
    const edges = [_][2]usize{
        .{ 0, 1 }, .{ 0, 2 }, .{ 0, 3 },
        .{ 1, 2 }, .{ 1, 3 }, .{ 2, 3 },
    };
    var cover = try vertexCoverApprox(gpa, 4, &edges);
    defer cover.deinit(gpa);

    // K4 requires OPT=3; edges (0,1) and (2,3) are picked, giving 4 <= 2 x OPT.
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3 }, cover.items);
    try std.testing.expect(isValidCover(&edges, cover.items));
}

test "vertex cover: disconnected components" {
    const gpa = std.testing.allocator;
    // Two triangles: 0-1-2 and 3-4-5
    const edges = [_][2]usize{
        .{ 0, 1 }, .{ 1, 2 }, .{ 2, 0 },
        .{ 3, 4 }, .{ 4, 5 }, .{ 5, 3 },
    };
    var cover = try vertexCoverApprox(gpa, 6, &edges);
    defer cover.deinit(gpa);

    // Two triangles require OPT=4; each triangle contributes its first edge: 4 vertices.
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3, 4 }, cover.items);
    try std.testing.expect(isValidCover(&edges, cover.items));
}

test "vertex cover: greedy heuristic - star" {
    const gpa = std.testing.allocator;
    // Star: center 0 connected to 1,2,3,4
    const edges = [_][2]usize{ .{ 0, 1 }, .{ 0, 2 }, .{ 0, 3 }, .{ 0, 4 } };
    var cover = try vertexCoverGreedy(gpa, 5, &edges);
    defer cover.deinit(gpa);

    // Greedy should pick center (highest degree) - optimal solution
    try std.testing.expectEqualSlices(usize, &.{0}, cover.items);
    try std.testing.expect(isValidCover(&edges, cover.items));
}

test "vertex cover: greedy vs approx comparison" {
    const gpa = std.testing.allocator;
    const edges = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 4 } };

    var cover_approx = try vertexCoverApprox(gpa, 5, &edges);
    defer cover_approx.deinit(gpa);

    var cover_greedy = try vertexCoverGreedy(gpa, 5, &edges);
    defer cover_greedy.deinit(gpa);

    // Both should be valid covers
    try std.testing.expect(isValidCover(&edges, cover_approx.items));
    try std.testing.expect(isValidCover(&edges, cover_greedy.items));
}

test "vertex cover: large path (stress test)" {
    const gpa = std.testing.allocator;

    // Path of length 100: 0-1-2-...-100
    var edges: ArrayList([2]usize) = .empty;
    defer edges.deinit(gpa);

    for (0..100) |i| {
        try edges.append(gpa, .{ i, i + 1 });
    }

    var cover = try vertexCoverApprox(gpa, 101, edges.items);
    defer cover.deinit(gpa);

    // Path of length 100 requires OPT=50; the picked matching has 50 edges, so 100 vertices.
    try std.testing.expectEqual(@as(usize, 100), cover.items.len);
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

test "vertex cover: self-loop adds its vertex once" {
    const gpa = std.testing.allocator;
    const edges = [_][2]usize{ .{ 2, 2 }, .{ 0, 1 } };
    var cover = try vertexCoverApprox(gpa, 3, &edges);
    defer cover.deinit(gpa);

    try std.testing.expectEqualSlices(usize, &.{ 2, 0, 1 }, cover.items);
    try std.testing.expect(isValidCover(&edges, cover.items));

    var greedy = try vertexCoverGreedy(gpa, 3, &edges);
    defer greedy.deinit(gpa);

    try std.testing.expect(isValidCover(&edges, greedy.items));
}

test "vertex cover: greedy can beat the matching bound" {
    const gpa = std.testing.allocator;
    // Path 0-1-2-3-4: the minimum cover {1,3} has 2 vertices; greedy takes degree-2 vertex 1,
    // then the remaining edges (2,3),(3,4) have vertex 3 with degree 2.
    const edges = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 4 } };
    var cover = try vertexCoverGreedy(gpa, 5, &edges);
    defer cover.deinit(gpa);

    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, cover.items);
}

test "vertex cover: no leaks when an allocation fails" {
    const Runner = struct {
        fn run(gpa: Allocator) !void {
            const edges = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 4 } };
            var approx = try vertexCoverApprox(gpa, 5, &edges);
            defer approx.deinit(gpa);

            var greedy = try vertexCoverGreedy(gpa, 5, &edges);
            defer greedy.deinit(gpa);

            try std.testing.expect(isValidCover(&edges, approx.items));
            try std.testing.expect(isValidCover(&edges, greedy.items));
        }
    };

    try std.testing.checkAllAllocationFailures(std.testing.allocator, Runner.run, .{});
}

test "vertex cover: endpoint out of range is InvalidEdge in both algorithms" {
    const gpa = std.testing.allocator;
    // Boundary: endpoint == num_vertices is the first invalid index; the bad edge is last so
    // earlier valid edges must not leak.
    const edges = [_][2]usize{ .{ 0, 1 }, .{ 1, 3 } };
    try std.testing.expectError(error.InvalidEdge, vertexCoverApprox(gpa, 3, &edges));
    try std.testing.expectError(error.InvalidEdge, vertexCoverGreedy(gpa, 3, &edges));

    const first_endpoint = [_][2]usize{.{ 5, 0 }};
    try std.testing.expectError(error.InvalidEdge, vertexCoverApprox(gpa, 2, &first_endpoint));
    try std.testing.expectError(error.InvalidEdge, vertexCoverGreedy(gpa, 2, &first_endpoint));

    // The largest valid index still succeeds.
    const valid = [_][2]usize{.{ 2, 2 }};
    var cover = try vertexCoverApprox(gpa, 3, &valid);
    defer cover.deinit(gpa);
    try std.testing.expectEqualSlices(usize, &.{2}, cover.items);
}
