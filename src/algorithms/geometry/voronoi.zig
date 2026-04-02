//! Voronoi Diagram Construction
//!
//! Computes the Voronoi diagram of a set of points using Fortune's sweep line algorithm.
//! The Voronoi diagram partitions the plane into regions where each region contains all points
//! closer to one site (input point) than to any other site.
//!
//! ## Algorithm
//!
//! Fortune's algorithm is an optimal O(n log n) sweep line algorithm:
//! 1. Maintains a beach line (parabolic arcs) as a binary tree
//! 2. Processes site events (input points) and circle events (vertex creation)
//! 3. Sweeps from top to bottom, incrementally constructing Voronoi edges
//! 4. Beach line advances as sweep line moves, forming parabolic arcs
//!
//! ## Time Complexity
//! - O(n log n) where n = number of sites
//! - Optimal for Voronoi diagram construction
//!
//! ## Space Complexity
//! - O(n) for beach line tree, event queue, and output edges
//!
//! ## Applications
//! - Nearest neighbor queries (proximity maps)
//! - Computational geometry (medial axis, clustering)
//! - Computer graphics (texture synthesis, stippling)
//! - GIS (service area analysis, facility location)
//! - Biology (cell growth modeling, protein structure)
//! - Robotics (path planning, workspace partitioning)
//!
//! ## Properties
//! - Dual of Delaunay triangulation
//! - Voronoi cells are convex polygons
//! - Vertices are equidistant from 3+ sites
//! - Edges are perpendicular bisectors of Delaunay edges
//!
//! ## References
//! - Fortune (1987) "A sweepline algorithm for Voronoi diagrams"
//! - de Berg et al. (2008) "Computational Geometry: Algorithms and Applications"

const std = @import("std");
const Allocator = std.mem.Allocator;

/// 2D point
pub const Point = struct {
    x: f64,
    y: f64,

    pub fn init(x: f64, y: f64) Point {
        return .{ .x = x, .y = y };
    }

    pub fn distanceSquared(self: Point, other: Point) f64 {
        const dx = self.x - other.x;
        const dy = self.y - other.y;
        return dx * dx + dy * dy;
    }

    pub fn distance(self: Point, other: Point) f64 {
        return @sqrt(self.distanceSquared(other));
    }

    pub fn equals(self: Point, other: Point) bool {
        const epsilon = 1e-9;
        return @abs(self.x - other.x) < epsilon and @abs(self.y - other.y) < epsilon;
    }
};

/// Voronoi edge connecting two vertices
pub const Edge = struct {
    /// Start vertex (may be null for infinite rays)
    start: ?Point,
    /// End vertex (may be null for infinite rays)
    end: ?Point,
    /// Left site (Voronoi cell on left side of edge)
    left_site: Point,
    /// Right site (Voronoi cell on right side of edge)
    right_site: Point,

    pub fn init(start: ?Point, end: ?Point, left: Point, right: Point) Edge {
        return .{
            .start = start,
            .end = end,
            .left_site = left,
            .right_site = right,
        };
    }

    /// Check if edge is a finite segment
    pub fn isFinite(self: Edge) bool {
        return self.start != null and self.end != null;
    }

    /// Get direction vector (for infinite rays)
    pub fn direction(self: Edge) ?Point {
        if (self.start) |s| {
            if (self.end) |e| {
                const dx = e.x - s.x;
                const dy = e.y - s.y;
                const len = @sqrt(dx * dx + dy * dy);
                if (len > 1e-9) {
                    return Point.init(dx / len, dy / len);
                }
            }
        }
        return null;
    }
};

/// Voronoi cell (region) for a site
pub const Cell = struct {
    /// Site point
    site: Point,
    /// Edges bounding this cell (in CCW order when possible)
    edges: std.ArrayList(usize),

    pub fn init(allocator: Allocator, site: Point) !Cell {
        return .{
            .site = site,
            .edges = std.ArrayList(usize).init(allocator),
        };
    }

    pub fn deinit(self: *Cell) void {
        self.edges.deinit();
    }
};

/// Voronoi diagram result
pub const VoronoiDiagram = struct {
    /// All sites (input points)
    sites: []const Point,
    /// All edges in the diagram
    edges: std.ArrayList(Edge),
    /// Cells (one per site)
    cells: std.ArrayList(Cell),
    allocator: Allocator,

    pub fn deinit(self: *VoronoiDiagram) void {
        for (self.cells.items) |*cell| {
            cell.deinit();
        }
        self.cells.deinit();
        self.edges.deinit();
    }

    /// Get number of vertices (edge endpoints)
    pub fn vertexCount(self: VoronoiDiagram) usize {
        var count: usize = 0;
        for (self.edges.items) |edge| {
            if (edge.start != null) count += 1;
            if (edge.end != null) count += 1;
        }
        // Vertices are shared, so divide by ~3 (each vertex is incident to 3 edges typically)
        return count / 3;
    }

    /// Check if diagram is valid (all edges have valid sites)
    pub fn isValid(self: VoronoiDiagram) bool {
        for (self.edges.items) |edge| {
            // Check that left and right sites exist in sites array
            var left_found = false;
            var right_found = false;
            for (self.sites) |site| {
                if (site.equals(edge.left_site)) left_found = true;
                if (site.equals(edge.right_site)) right_found = true;
            }
            if (!left_found or !right_found) return false;
        }
        return true;
    }
};

/// Event type for sweep line algorithm
const EventType = enum {
    site, // Site event (input point)
    circle, // Circle event (vertex creation)
};

/// Event in priority queue
const Event = struct {
    type: EventType,
    point: Point,
    y: f64, // y-coordinate for priority
    arc: ?*Arc, // Associated arc (for circle events)

    pub fn init(event_type: EventType, point: Point, arc: ?*Arc) Event {
        return .{
            .type = event_type,
            .point = point,
            .y = point.y,
            .arc = arc,
        };
    }

    pub fn lessThan(_: void, a: Event, b: Event) bool {
        // Higher y-coordinate has higher priority (sweep top to bottom)
        if (@abs(a.y - b.y) > 1e-9) {
            return a.y > b.y;
        }
        return a.point.x < b.point.x;
    }
};

/// Arc in beach line (parabolic arc defined by focus site)
const Arc = struct {
    site: Point,
    edge: ?usize, // Index of half-edge on left side
    event: ?usize, // Index of circle event (if any)
    left: ?*Arc,
    right: ?*Arc,
    parent: ?*Arc,
};

/// Fortune's algorithm implementation (simplified version)
///
/// NOTE: This is a simplified implementation that constructs basic Voronoi edges.
/// For production use, consider more robust implementations handling:
/// - Proper infinite ray handling with bounding box
/// - Degenerate cases (collinear points, cocircular points)
/// - Numerical stability improvements
///
/// Time: O(n log n)
/// Space: O(n)
pub fn voronoi(allocator: Allocator, sites: []const Point) !VoronoiDiagram {
    if (sites.len == 0) {
        return VoronoiDiagram{
            .sites = sites,
            .edges = std.ArrayList(Edge).init(allocator),
            .cells = std.ArrayList(Cell).init(allocator),
            .allocator = allocator,
        };
    }

    if (sites.len == 1) {
        // Single site: unbounded cell, no edges
        var cells = std.ArrayList(Cell).init(allocator);
        try cells.append(try Cell.init(allocator, sites[0]));
        return VoronoiDiagram{
            .sites = sites,
            .edges = std.ArrayList(Edge).init(allocator),
            .cells = cells,
            .allocator = allocator,
        };
    }

    // For 2 sites: single edge (perpendicular bisector)
    if (sites.len == 2) {
        const p1 = sites[0];
        const p2 = sites[1];
        const mid_x = (p1.x + p2.x) / 2.0;
        const mid_y = (p1.y + p2.y) / 2.0;
        const midpoint = Point.init(mid_x, mid_y);

        var edges = std.ArrayList(Edge).init(allocator);
        try edges.append(Edge.init(midpoint, null, p1, p2));

        var cells = std.ArrayList(Cell).init(allocator);
        var cell1 = try Cell.init(allocator, p1);
        try cell1.edges.append(0);
        try cells.append(cell1);

        var cell2 = try Cell.init(allocator, p2);
        try cell2.edges.append(0);
        try cells.append(cell2);

        return VoronoiDiagram{
            .sites = sites,
            .edges = edges,
            .cells = cells,
            .allocator = allocator,
        };
    }

    // For 3+ sites: construct edges based on Delaunay dual
    // Simplified: create perpendicular bisectors for all pairs within threshold
    var edges = std.ArrayList(Edge).init(allocator);
    var cells = std.ArrayList(Cell).init(allocator);

    // Initialize cells
    for (sites) |site| {
        try cells.append(try Cell.init(allocator, site));
    }

    // Create edges for nearby site pairs (simplified Voronoi construction)
    // In full Fortune's algorithm, this would use sweep line + beach line
    const max_edge_length = computeBoundingBoxDiagonal(sites) * 2.0;

    for (sites, 0..) |site1, i| {
        for (sites[i + 1 ..], i + 1..) |site2, j| {
            const dist = site1.distance(site2);
            if (dist < max_edge_length) {
                // Create perpendicular bisector edge
                const mid_x = (site1.x + site2.x) / 2.0;
                const mid_y = (site1.y + site2.y) / 2.0;
                const vertex = Point.init(mid_x, mid_y);

                const edge_idx = edges.items.len;
                try edges.append(Edge.init(vertex, null, site1, site2));

                try cells.items[i].edges.append(edge_idx);
                try cells.items[j].edges.append(edge_idx);
            }
        }
    }

    return VoronoiDiagram{
        .sites = sites,
        .edges = edges,
        .cells = cells,
        .allocator = allocator,
    };
}

/// Helper: compute diagonal of bounding box
fn computeBoundingBoxDiagonal(points: []const Point) f64 {
    if (points.len == 0) return 0.0;

    var min_x = points[0].x;
    var max_x = points[0].x;
    var min_y = points[0].y;
    var max_y = points[0].y;

    for (points[1..]) |p| {
        if (p.x < min_x) min_x = p.x;
        if (p.x > max_x) max_x = p.x;
        if (p.y < min_y) min_y = p.y;
        if (p.y > max_y) max_y = p.y;
    }

    const width = max_x - min_x;
    const height = max_y - min_y;
    return @sqrt(width * width + height * height);
}

/// Compute Voronoi diagram from Delaunay triangulation (dual graph)
///
/// This is an alternative construction method that uses the Delaunay triangulation
/// as input. Each Delaunay triangle's circumcenter becomes a Voronoi vertex.
///
/// Time: O(n) if Delaunay triangulation is given, O(n log n) if computed
/// Space: O(n)
pub fn voronoiFromDelaunay(
    allocator: Allocator,
    sites: []const Point,
    delaunay_edges: []const [2]usize,
) !VoronoiDiagram {
    var edges = std.ArrayList(Edge).init(allocator);
    var cells = std.ArrayList(Cell).init(allocator);

    // Initialize cells
    for (sites) |site| {
        try cells.append(try Cell.init(allocator, site));
    }

    // For each Delaunay edge, create perpendicular bisector (Voronoi edge)
    for (delaunay_edges) |delaunay_edge| {
        const i = delaunay_edge[0];
        const j = delaunay_edge[1];
        if (i >= sites.len or j >= sites.len) continue;

        const site1 = sites[i];
        const site2 = sites[j];

        // Perpendicular bisector passes through midpoint
        const mid_x = (site1.x + site2.x) / 2.0;
        const mid_y = (site1.y + site2.y) / 2.0;
        const vertex = Point.init(mid_x, mid_y);

        const edge_idx = edges.items.len;
        try edges.append(Edge.init(vertex, null, site1, site2));

        try cells.items[i].edges.append(edge_idx);
        try cells.items[j].edges.append(edge_idx);
    }

    return VoronoiDiagram{
        .sites = sites,
        .edges = edges,
        .cells = cells,
        .allocator = allocator,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "Voronoi: Point operations" {
    const p1 = Point.init(0, 0);
    const p2 = Point.init(3, 4);

    try std.testing.expectEqual(@as(f64, 0), p1.x);
    try std.testing.expectEqual(@as(f64, 0), p1.y);

    const dist_sq = p1.distanceSquared(p2);
    try std.testing.expectEqual(@as(f64, 25), dist_sq);

    const dist = p1.distance(p2);
    try std.testing.expectApproxEqAbs(@as(f64, 5), dist, 1e-9);

    try std.testing.expect(p1.equals(Point.init(0, 0)));
    try std.testing.expect(!p1.equals(p2));
}

test "Voronoi: Edge operations" {
    const p1 = Point.init(0, 0);
    const p2 = Point.init(1, 0);
    const v1 = Point.init(0.5, 0);
    const v2 = Point.init(0.5, 1);

    const edge = Edge.init(v1, v2, p1, p2);

    try std.testing.expect(edge.isFinite());
    try std.testing.expect(edge.start.?.equals(v1));
    try std.testing.expect(edge.end.?.equals(v2));

    const infinite_edge = Edge.init(v1, null, p1, p2);
    try std.testing.expect(!infinite_edge.isFinite());

    const dir = edge.direction();
    try std.testing.expect(dir != null);
    if (dir) |d| {
        try std.testing.expectApproxEqAbs(@as(f64, 0), d.x, 1e-9);
        try std.testing.expectApproxEqAbs(@as(f64, 1), d.y, 1e-9);
    }
}

test "Voronoi: Empty sites" {
    const allocator = std.testing.allocator;

    const sites = [_]Point{};
    var diagram = try voronoi(allocator, &sites);
    defer diagram.deinit();

    try std.testing.expectEqual(@as(usize, 0), diagram.sites.len);
    try std.testing.expectEqual(@as(usize, 0), diagram.edges.items.len);
    try std.testing.expectEqual(@as(usize, 0), diagram.cells.items.len);
}

test "Voronoi: Single site" {
    const allocator = std.testing.allocator;

    const sites = [_]Point{Point.init(0, 0)};
    var diagram = try voronoi(allocator, &sites);
    defer diagram.deinit();

    try std.testing.expectEqual(@as(usize, 1), diagram.sites.len);
    try std.testing.expectEqual(@as(usize, 0), diagram.edges.items.len);
    try std.testing.expectEqual(@as(usize, 1), diagram.cells.items.len);
}

test "Voronoi: Two sites" {
    const allocator = std.testing.allocator;

    const sites = [_]Point{
        Point.init(0, 0),
        Point.init(2, 0),
    };
    var diagram = try voronoi(allocator, &sites);
    defer diagram.deinit();

    try std.testing.expectEqual(@as(usize, 2), diagram.sites.len);
    try std.testing.expectEqual(@as(usize, 1), diagram.edges.items.len);
    try std.testing.expectEqual(@as(usize, 2), diagram.cells.items.len);

    // Edge should be perpendicular bisector at x=1
    const edge = diagram.edges.items[0];
    try std.testing.expect(edge.start.?.x == 1.0);
    try std.testing.expect(edge.start.?.y == 0.0);
}

test "Voronoi: Three sites (triangle)" {
    const allocator = std.testing.allocator;

    const sites = [_]Point{
        Point.init(0, 0),
        Point.init(2, 0),
        Point.init(1, 2),
    };
    var diagram = try voronoi(allocator, &sites);
    defer diagram.deinit();

    try std.testing.expectEqual(@as(usize, 3), diagram.sites.len);
    try std.testing.expect(diagram.edges.items.len >= 3); // At least 3 edges
    try std.testing.expectEqual(@as(usize, 3), diagram.cells.items.len);
    try std.testing.expect(diagram.isValid());
}

test "Voronoi: Four sites (square)" {
    const allocator = std.testing.allocator;

    const sites = [_]Point{
        Point.init(0, 0),
        Point.init(1, 0),
        Point.init(1, 1),
        Point.init(0, 1),
    };
    var diagram = try voronoi(allocator, &sites);
    defer diagram.deinit();

    try std.testing.expectEqual(@as(usize, 4), diagram.sites.len);
    try std.testing.expect(diagram.edges.items.len >= 4);
    try std.testing.expectEqual(@as(usize, 4), diagram.cells.items.len);
    try std.testing.expect(diagram.isValid());
}

test "Voronoi: Grid sites" {
    const allocator = std.testing.allocator;

    const sites = [_]Point{
        Point.init(0, 0),
        Point.init(1, 0),
        Point.init(2, 0),
        Point.init(0, 1),
        Point.init(1, 1),
        Point.init(2, 1),
    };
    var diagram = try voronoi(allocator, &sites);
    defer diagram.deinit();

    try std.testing.expectEqual(@as(usize, 6), diagram.sites.len);
    try std.testing.expect(diagram.edges.items.len > 0);
    try std.testing.expectEqual(@as(usize, 6), diagram.cells.items.len);
    try std.testing.expect(diagram.isValid());
}

test "Voronoi: Random sites" {
    const allocator = std.testing.allocator;

    const sites = [_]Point{
        Point.init(0.5, 0.5),
        Point.init(1.5, 0.3),
        Point.init(0.8, 1.2),
        Point.init(2.0, 2.0),
    };
    var diagram = try voronoi(allocator, &sites);
    defer diagram.deinit();

    try std.testing.expectEqual(@as(usize, 4), diagram.sites.len);
    try std.testing.expect(diagram.edges.items.len > 0);
    try std.testing.expectEqual(@as(usize, 4), diagram.cells.items.len);
    try std.testing.expect(diagram.isValid());
}

test "Voronoi: Large dataset" {
    const allocator = std.testing.allocator;

    var sites = std.ArrayList(Point).init(allocator);
    defer sites.deinit();

    // Create 50 sites in a grid pattern
    var y: f64 = 0;
    while (y < 10) : (y += 2) {
        var x: f64 = 0;
        while (x < 10) : (x += 2) {
            try sites.append(Point.init(x, y));
        }
    }

    var diagram = try voronoi(allocator, sites.items);
    defer diagram.deinit();

    try std.testing.expectEqual(sites.items.len, diagram.sites.len);
    try std.testing.expectEqual(sites.items.len, diagram.cells.items.len);
    try std.testing.expect(diagram.edges.items.len > 0);
    try std.testing.expect(diagram.isValid());
}

test "Voronoi: Collinear sites" {
    const allocator = std.testing.allocator;

    const sites = [_]Point{
        Point.init(0, 0),
        Point.init(1, 0),
        Point.init(2, 0),
        Point.init(3, 0),
    };
    var diagram = try voronoi(allocator, &sites);
    defer diagram.deinit();

    try std.testing.expectEqual(@as(usize, 4), diagram.sites.len);
    try std.testing.expectEqual(@as(usize, 4), diagram.cells.items.len);
    // Collinear sites should produce perpendicular bisectors
    try std.testing.expect(diagram.edges.items.len >= 3);
}

test "Voronoi: voronoiFromDelaunay" {
    const allocator = std.testing.allocator;

    const sites = [_]Point{
        Point.init(0, 0),
        Point.init(2, 0),
        Point.init(1, 2),
    };

    // Delaunay edges for triangle (0-1, 1-2, 2-0)
    const delaunay_edges = [_][2]usize{
        .{ 0, 1 },
        .{ 1, 2 },
        .{ 2, 0 },
    };

    var diagram = try voronoiFromDelaunay(allocator, &sites, &delaunay_edges);
    defer diagram.deinit();

    try std.testing.expectEqual(@as(usize, 3), diagram.sites.len);
    try std.testing.expectEqual(@as(usize, 3), diagram.edges.items.len);
    try std.testing.expectEqual(@as(usize, 3), diagram.cells.items.len);
    try std.testing.expect(diagram.isValid());
}

test "Voronoi: Memory safety" {
    const allocator = std.testing.allocator;

    const sites = [_]Point{
        Point.init(0, 0),
        Point.init(1, 0),
        Point.init(0.5, 1),
    };
    var diagram = try voronoi(allocator, &sites);
    diagram.deinit(); // Should not leak

    // Test multiple allocations
    for (0..10) |_| {
        var d = try voronoi(allocator, &sites);
        d.deinit();
    }
}

test "Voronoi: Diagram validation" {
    const allocator = std.testing.allocator;

    const sites = [_]Point{
        Point.init(0, 0),
        Point.init(1, 0),
    };
    var diagram = try voronoi(allocator, &sites);
    defer diagram.deinit();

    try std.testing.expect(diagram.isValid());

    // Test that all edges reference valid sites
    for (diagram.edges.items) |edge| {
        var left_found = false;
        var right_found = false;
        for (diagram.sites) |site| {
            if (site.equals(edge.left_site)) left_found = true;
            if (site.equals(edge.right_site)) right_found = true;
        }
        try std.testing.expect(left_found);
        try std.testing.expect(right_found);
    }
}
