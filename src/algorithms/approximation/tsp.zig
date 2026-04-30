const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// Traveling Salesman Problem (TSP) Approximation Algorithms
///
/// TSP: given a complete graph with edge weights, find the shortest Hamiltonian cycle
/// (tour visiting all vertices exactly once).
/// Finding optimal TSP solution is NP-complete, but we can approximate for metric TSP.

pub const TspResult = struct {
    tour: ArrayList(usize),
    cost: f64,

    pub fn deinit(self: *TspResult) void {
        self.tour.deinit();
    }
};

/// Edge representation for MST construction
const Edge = struct {
    u: usize,
    v: usize,
    weight: f64,
};

/// Union-Find (Disjoint Set) for Kruskal's MST
const UnionFind = struct {
    parent: []usize,
    rank: []usize,
    allocator: Allocator,

    fn init(allocator: Allocator, n: usize) !UnionFind {
        const parent = try allocator.alloc(usize, n);
        errdefer allocator.free(parent);
        const rank = try allocator.alloc(usize, n);
        errdefer allocator.free(rank);

        for (parent, 0..) |*p, i| p.* = i;
        @memset(rank, 0);

        return UnionFind{
            .parent = parent,
            .rank = rank,
            .allocator = allocator,
        };
    }

    fn deinit(self: *UnionFind) void {
        self.allocator.free(self.parent);
        self.allocator.free(self.rank);
    }

    fn find(self: *UnionFind, x: usize) usize {
        if (self.parent[x] != x) {
            self.parent[x] = self.find(self.parent[x]); // Path compression
        }
        return self.parent[x];
    }

    fn unite(self: *UnionFind, x: usize, y: usize) void {
        const root_x = self.find(x);
        const root_y = self.find(y);
        if (root_x == root_y) return;

        // Union by rank
        if (self.rank[root_x] < self.rank[root_y]) {
            self.parent[root_x] = root_y;
        } else if (self.rank[root_x] > self.rank[root_y]) {
            self.parent[root_y] = root_x;
        } else {
            self.parent[root_y] = root_x;
            self.rank[root_x] += 1;
        }
    }
};

/// Computes Minimum Spanning Tree using Kruskal's algorithm.
///
/// Time: O(E log E) for sorting edges
/// Space: O(V + E)
fn minimumSpanningTree(
    allocator: Allocator,
    num_vertices: usize,
    dist_matrix: []const []const f64,
) !ArrayList(Edge) {
    var edges = ArrayList(Edge).init(allocator);
    defer edges.deinit();

    // Collect all edges
    for (0..num_vertices) |i| {
        for (i + 1..num_vertices) |j| {
            try edges.append(.{
                .u = i,
                .v = j,
                .weight = dist_matrix[i][j],
            });
        }
    }

    // Sort edges by weight
    const Context = struct {
        pub fn lessThan(_: @This(), a: Edge, b: Edge) bool {
            return a.weight < b.weight;
        }
    };
    std.mem.sort(Edge, edges.items, Context{}, Context.lessThan);

    // Kruskal's algorithm
    var uf = try UnionFind.init(allocator, num_vertices);
    defer uf.deinit();

    var mst = ArrayList(Edge).init(allocator);
    errdefer mst.deinit();

    for (edges.items) |edge| {
        if (uf.find(edge.u) != uf.find(edge.v)) {
            try mst.append(edge);
            uf.unite(edge.u, edge.v);
            if (mst.items.len == num_vertices - 1) break;
        }
    }

    return mst;
}

/// Builds adjacency list from MST edges.
fn mstToAdjList(
    allocator: Allocator,
    num_vertices: usize,
    mst: []const Edge,
) ![]ArrayList(usize) {
    var adj = try allocator.alloc(ArrayList(usize), num_vertices);
    for (adj) |*list| list.* = ArrayList(usize).init(allocator);

    for (mst) |edge| {
        try adj[edge.u].append(edge.v);
        try adj[edge.v].append(edge.u);
    }

    return adj;
}

/// DFS traversal to get preorder vertices (for MST-based tour).
fn dfsPreorder(
    adj: []const ArrayList(usize),
    start: usize,
    visited: []bool,
    tour: *ArrayList(usize),
) !void {
    visited[start] = true;
    try tour.append(start);

    for (adj[start].items) |neighbor| {
        if (!visited[neighbor]) {
            try dfsPreorder(adj, neighbor, visited, tour);
        }
    }
}

/// 2-approximation for metric TSP using MST.
///
/// Time: O(V² log V) for MST construction and tour building
/// Space: O(V²) for adjacency list and visited array
///
/// Returns: TspResult with tour and cost (caller owns memory)
///
/// Algorithm:
/// 1. Compute MST of the complete graph
/// 2. Perform DFS on MST to get preorder traversal
/// 3. Close the tour by returning to start
/// 4. Approximation ratio: cost(tour) ≤ 2 × cost(OPT) for metric TSP
///
/// Requires: Triangle inequality (metric distances)
///
/// Example:
/// ```zig
/// var dist = try allocator.alloc([]f64, 4);
/// // Fill distance matrix...
/// var result = try tspMst(allocator, 4, dist);
/// defer result.deinit();
/// // result.tour contains vertex order, result.cost is total distance
/// ```
pub fn tspMst(
    allocator: Allocator,
    num_vertices: usize,
    dist_matrix: []const []const f64,
) !TspResult {
    if (num_vertices == 0) {
        return TspResult{
            .tour = ArrayList(usize).init(allocator),
            .cost = 0.0,
        };
    }

    // Step 1: Compute MST
    var mst = try minimumSpanningTree(allocator, num_vertices, dist_matrix);
    defer mst.deinit();

    // Step 2: Build adjacency list from MST
    const adj = try mstToAdjList(allocator, num_vertices, mst.items);
    defer {
        for (adj) |*list| list.deinit();
        allocator.free(adj);
    }

    // Step 3: DFS preorder traversal
    var tour = ArrayList(usize).init(allocator);
    errdefer tour.deinit();

    const visited = try allocator.alloc(bool, num_vertices);
    defer allocator.free(visited);
    @memset(visited, false);

    try dfsPreorder(adj, 0, visited, &tour);

    // Step 4: Close the tour (return to start)
    try tour.append(0);

    // Step 5: Compute tour cost
    var cost: f64 = 0.0;
    for (0..tour.items.len - 1) |i| {
        cost += dist_matrix[tour.items[i]][tour.items[i + 1]];
    }

    return TspResult{
        .tour = tour,
        .cost = cost,
    };
}

/// Nearest-neighbor heuristic for TSP (no approximation guarantee).
///
/// Time: O(V²)
/// Space: O(V)
///
/// Returns: TspResult with tour and cost (caller owns memory)
///
/// Algorithm:
/// 1. Start at vertex 0
/// 2. Repeatedly visit the nearest unvisited vertex
/// 3. Return to start
///
/// Note: This is a greedy heuristic with no theoretical approximation guarantee,
/// but often performs well in practice. Use tspMst for guaranteed 2-approximation.
///
/// Example:
/// ```zig
/// var dist = try allocator.alloc([]f64, 4);
/// // Fill distance matrix...
/// var result = try tspNearestNeighbor(allocator, 4, dist);
/// defer result.deinit();
/// ```
pub fn tspNearestNeighbor(
    allocator: Allocator,
    num_vertices: usize,
    dist_matrix: []const []const f64,
) !TspResult {
    var tour = ArrayList(usize).init(allocator);
    errdefer tour.deinit();

    if (num_vertices == 0) {
        return TspResult{ .tour = tour, .cost = 0.0 };
    }

    var visited = try allocator.alloc(bool, num_vertices);
    defer allocator.free(visited);
    @memset(visited, false);

    // Start at vertex 0
    var current: usize = 0;
    try tour.append(current);
    visited[current] = true;

    // Visit nearest unvisited vertex
    for (1..num_vertices) |_| {
        var nearest: usize = 0;
        var min_dist: f64 = std.math.inf(f64);

        for (0..num_vertices) |v| {
            if (!visited[v] and dist_matrix[current][v] < min_dist) {
                nearest = v;
                min_dist = dist_matrix[current][v];
            }
        }

        current = nearest;
        try tour.append(current);
        visited[current] = true;
    }

    // Close the tour
    try tour.append(0);

    // Compute cost
    var cost: f64 = 0.0;
    for (0..tour.items.len - 1) |i| {
        cost += dist_matrix[tour.items[i]][tour.items[i + 1]];
    }

    return TspResult{
        .tour = tour,
        .cost = cost,
    };
}

/// Validates that the tour is a valid Hamiltonian cycle.
///
/// Time: O(V)
/// Space: O(V) for tracking visits
///
/// Returns: true if tour visits all vertices exactly once (plus return to start), false otherwise
pub fn isValidTour(allocator: Allocator, num_vertices: usize, tour: []const usize) !bool {
    if (tour.len != num_vertices + 1) return false;
    if (tour[0] != tour[tour.len - 1]) return false;

    var visited = try allocator.alloc(bool, num_vertices);
    defer allocator.free(visited);
    @memset(visited, false);

    // Check all vertices visited (except last = first)
    for (tour[0 .. tour.len - 1]) |v| {
        if (v >= num_vertices) return false;
        if (visited[v]) return false;
        visited[v] = true;
    }

    // Check all vertices visited
    for (visited) |v| {
        if (!v) return false;
    }

    return true;
}

/// Computes tour cost from distance matrix.
///
/// Time: O(V)
/// Space: O(1)
pub fn tourCost(dist_matrix: []const []const f64, tour: []const usize) f64 {
    var cost: f64 = 0.0;
    for (0..tour.len - 1) |i| {
        cost += dist_matrix[tour[i]][tour[i + 1]];
    }
    return cost;
}

// ============================================================================
// Tests
// ============================================================================

test "tsp: empty graph" {
    const allocator = std.testing.allocator;
    const dist: []const []const f64 = &.{};
    var result = try tspMst(allocator, 0, dist);
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 0), result.tour.items.len);
    try std.testing.expectEqual(@as(f64, 0.0), result.cost);
}

test "tsp: single vertex" {
    const allocator = std.testing.allocator;
    var dist = [_][]const f64{&[_]f64{0.0}};
    var result = try tspMst(allocator, 1, &dist);
    defer result.deinit();

    try std.testing.expect(try isValidTour(allocator, 1, result.tour.items));
    try std.testing.expectEqual(@as(f64, 0.0), result.cost);
}

test "tsp: triangle" {
    const allocator = std.testing.allocator;
    // Triangle with distances: 0-1: 1.0, 1-2: 2.0, 2-0: 3.0
    var dist = [_][]const f64{
        &[_]f64{ 0.0, 1.0, 3.0 },
        &[_]f64{ 1.0, 0.0, 2.0 },
        &[_]f64{ 3.0, 2.0, 0.0 },
    };
    var result = try tspMst(allocator, 3, &dist);
    defer result.deinit();

    try std.testing.expect(try isValidTour(allocator, 3, result.tour.items));
    try std.testing.expectEqual(@as(usize, 4), result.tour.items.len); // 3 + 1
    // Optimal tour: 0→1→2→0 cost = 1+2+3 = 6, MST-based gives ≤12
    try std.testing.expect(result.cost <= 12.0);
}

test "tsp: square (4 vertices)" {
    const allocator = std.testing.allocator;
    // Square with unit sides: 0-1-2-3 in a square
    var dist = [_][]const f64{
        &[_]f64{ 0.0, 1.0, 1.414, 1.0 }, // 0: adjacent to 1,3, diagonal to 2
        &[_]f64{ 1.0, 0.0, 1.0, 1.414 }, // 1: adjacent to 0,2, diagonal to 3
        &[_]f64{ 1.414, 1.0, 0.0, 1.0 }, // 2: adjacent to 1,3, diagonal to 0
        &[_]f64{ 1.0, 1.414, 1.0, 0.0 }, // 3: adjacent to 0,2, diagonal to 1
    };
    var result = try tspMst(allocator, 4, &dist);
    defer result.deinit();

    try std.testing.expect(try isValidTour(allocator, 4, result.tour.items));
    // Optimal tour: perimeter = 4.0, MST-based gives ≤8.0
    try std.testing.expect(result.cost <= 8.0);
}

test "tsp: complete graph K4" {
    const allocator = std.testing.allocator;
    var dist = [_][]const f64{
        &[_]f64{ 0.0, 10.0, 15.0, 20.0 },
        &[_]f64{ 10.0, 0.0, 35.0, 25.0 },
        &[_]f64{ 15.0, 35.0, 0.0, 30.0 },
        &[_]f64{ 20.0, 25.0, 30.0, 0.0 },
    };
    var result = try tspMst(allocator, 4, &dist);
    defer result.deinit();

    try std.testing.expect(try isValidTour(allocator, 4, result.tour.items));
    try std.testing.expect(result.cost > 0.0);
}

test "tsp: nearest neighbor heuristic" {
    const allocator = std.testing.allocator;
    var dist = [_][]const f64{
        &[_]f64{ 0.0, 1.0, 3.0 },
        &[_]f64{ 1.0, 0.0, 2.0 },
        &[_]f64{ 3.0, 2.0, 0.0 },
    };
    var result = try tspNearestNeighbor(allocator, 3, &dist);
    defer result.deinit();

    try std.testing.expect(try isValidTour(allocator, 3, result.tour.items));
    try std.testing.expect(result.cost > 0.0);
}

test "tsp: MST vs nearest neighbor comparison" {
    const allocator = std.testing.allocator;
    var dist = [_][]const f64{
        &[_]f64{ 0.0, 10.0, 15.0, 20.0 },
        &[_]f64{ 10.0, 0.0, 35.0, 25.0 },
        &[_]f64{ 15.0, 35.0, 0.0, 30.0 },
        &[_]f64{ 20.0, 25.0, 30.0, 0.0 },
    };

    var result_mst = try tspMst(allocator, 4, &dist);
    defer result_mst.deinit();

    var result_nn = try tspNearestNeighbor(allocator, 4, &dist);
    defer result_nn.deinit();

    // Both should be valid tours
    try std.testing.expect(try isValidTour(allocator, 4, result_mst.tour.items));
    try std.testing.expect(try isValidTour(allocator, 4, result_nn.tour.items));
}

test "tsp: large graph (stress test)" {
    const allocator = std.testing.allocator;

    // Create 10-vertex complete graph with random distances
    var dist_storage = ArrayList(ArrayList(f64)).init(allocator);
    defer {
        for (dist_storage.items) |*row| row.deinit();
        dist_storage.deinit();
    }

    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    for (0..10) |i| {
        var row = ArrayList(f64).init(allocator);
        for (0..10) |j| {
            if (i == j) {
                try row.append(0.0);
            } else {
                try row.append(random.float(f64) * 100.0);
            }
        }
        try dist_storage.append(row);
    }

    var dist_slices = try allocator.alloc([]const f64, 10);
    defer allocator.free(dist_slices);
    for (dist_storage.items, 0..) |row, i| {
        dist_slices[i] = row.items;
    }

    var result = try tspMst(allocator, 10, dist_slices);
    defer result.deinit();

    try std.testing.expect(try isValidTour(allocator, 10, result.tour.items));
    try std.testing.expectEqual(@as(usize, 11), result.tour.items.len);
}

test "tsp: isValidTour detects invalid tours" {
    const allocator = std.testing.allocator;
    // Missing vertex
    const tour1 = [_]usize{ 0, 1, 0 }; // Skips vertex 2
    try std.testing.expect(!try isValidTour(allocator, 3, &tour1));

    // Duplicate vertex
    const tour2 = [_]usize{ 0, 1, 1, 2, 0 };
    try std.testing.expect(!try isValidTour(allocator, 3, &tour2));

    // Doesn't return to start
    const tour3 = [_]usize{ 0, 1, 2 };
    try std.testing.expect(!try isValidTour(allocator, 3, &tour3));
}

test "tsp: isValidTour accepts valid tour" {
    const allocator = std.testing.allocator;
    const tour = [_]usize{ 0, 1, 2, 0 };
    try std.testing.expect(try isValidTour(allocator, 3, &tour));
}

test "tsp: tourCost computation" {
    var dist = [_][]const f64{
        &[_]f64{ 0.0, 10.0, 20.0 },
        &[_]f64{ 10.0, 0.0, 15.0 },
        &[_]f64{ 20.0, 15.0, 0.0 },
    };
    const tour = [_]usize{ 0, 1, 2, 0 };
    const cost = tourCost(&dist, &tour);
    try std.testing.expectEqual(@as(f64, 10.0 + 15.0 + 20.0), cost);
}
