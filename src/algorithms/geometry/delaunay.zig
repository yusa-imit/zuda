const std = @import("std");
const testing = std.testing;

/// Point in 2D space
pub fn Point(comptime T: type) type {
    return struct {
        x: T,
        y: T,

        const Self = @This();

        pub fn equals(self: Self, other: Self) bool {
            return self.x == other.x and self.y == other.y;
        }

        pub fn distanceSquared(self: Self, other: Self) T {
            const dx = self.x - other.x;
            const dy = self.y - other.y;
            return dx * dx + dy * dy;
        }
    };
}

/// Triangle defined by three vertex indices
pub const Triangle = struct {
    a: usize,
    b: usize,
    c: usize,

    pub fn equals(self: Triangle, other: Triangle) bool {
        return (self.a == other.a and self.b == other.b and self.c == other.c) or
            (self.a == other.b and self.b == other.c and self.c == other.a) or
            (self.a == other.c and self.b == other.a and self.c == other.b);
    }

    pub fn containsVertex(self: Triangle, vertex: usize) bool {
        return self.a == vertex or self.b == vertex or self.c == vertex;
    }

    pub fn hasEdge(self: Triangle, v1: usize, v2: usize) bool {
        return (self.a == v1 and self.b == v2) or (self.b == v1 and self.c == v2) or
            (self.c == v1 and self.a == v2) or
            (self.a == v2 and self.b == v1) or (self.b == v2 and self.c == v1) or
            (self.c == v2 and self.a == v1);
    }
};

/// Edge defined by two vertex indices
pub const Edge = struct {
    a: usize,
    b: usize,

    pub fn equals(self: Edge, other: Edge) bool {
        return (self.a == other.a and self.b == other.b) or
            (self.a == other.b and self.b == other.a);
    }
};

/// Delaunay Triangulation using Bowyer-Watson incremental algorithm
///
/// Time: O(n log n) expected, O(n²) worst case
/// Space: O(n) for triangulation
///
/// The Delaunay triangulation maximizes the minimum angle of all triangles,
/// avoiding skinny triangles. It has the empty circumcircle property: no point
/// lies inside the circumcircle of any triangle.
///
/// Applications:
/// - Mesh generation (finite element analysis, computer graphics)
/// - Terrain modeling and surface reconstruction
/// - Nearest neighbor and interpolation
/// - Voronoi diagram computation (dual structure)
pub fn DelaunayTriangulation(comptime T: type) type {
    return struct {
        allocator: std.mem.Allocator,
        points: std.ArrayList(Point(T)),
        triangles: std.ArrayList(Triangle),

        const Self = @This();

        /// Initialize triangulation
        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .points = std.ArrayList(Point(T)).init(allocator),
                .triangles = std.ArrayList(Triangle).init(allocator),
            };
        }

        /// Free resources
        pub fn deinit(self: *Self) void {
            self.points.deinit();
            self.triangles.deinit();
        }

        /// Compute Delaunay triangulation using Bowyer-Watson algorithm
        ///
        /// Time: O(n log n) expected, O(n²) worst case
        /// Space: O(n) for triangulation
        pub fn triangulate(self: *Self, points: []const Point(T)) !void {
            if (points.len < 3) return error.InsufficientPoints;

            // Clear previous state
            self.points.clearRetainingCapacity();
            self.triangles.clearRetainingCapacity();

            // Add all points
            try self.points.appendSlice(points);

            // Create super-triangle that contains all points
            const bounds = computeBounds(points);
            const super_a = Point(T){ .x = bounds.min_x - bounds.width * 2, .y = bounds.min_y - bounds.height };
            const super_b = Point(T){ .x = bounds.max_x + bounds.width * 2, .y = bounds.min_y - bounds.height };
            const super_c = Point(T){ .x = (bounds.min_x + bounds.max_x) / 2, .y = bounds.max_y + bounds.height * 3 };

            const super_a_idx = self.points.items.len;
            try self.points.append(super_a);
            const super_b_idx = self.points.items.len;
            try self.points.append(super_b);
            const super_c_idx = self.points.items.len;
            try self.points.append(super_c);

            try self.triangles.append(Triangle{ .a = super_a_idx, .b = super_b_idx, .c = super_c_idx });

            // Add each point one at a time
            for (0..points.len) |i| {
                try self.addPoint(i);
            }

            // Remove triangles that share vertices with super-triangle
            var j: usize = 0;
            while (j < self.triangles.items.len) {
                const tri = self.triangles.items[j];
                if (tri.containsVertex(super_a_idx) or
                    tri.containsVertex(super_b_idx) or
                    tri.containsVertex(super_c_idx))
                {
                    _ = self.triangles.swapRemove(j);
                } else {
                    j += 1;
                }
            }

            // Remove super-triangle vertices
            _ = self.points.pop();
            _ = self.points.pop();
            _ = self.points.pop();
        }

        /// Add a point to the triangulation
        fn addPoint(self: *Self, point_idx: usize) !void {
            const point = self.points.items[point_idx];
            var bad_triangles = std.ArrayList(Triangle).init(self.allocator);
            defer bad_triangles.deinit();

            // Find triangles whose circumcircle contains the point
            for (self.triangles.items) |tri| {
                if (try self.inCircumcircle(point, tri)) {
                    try bad_triangles.append(tri);
                }
            }

            // Find the boundary edges of the polygonal hole
            var polygon = std.ArrayList(Edge).init(self.allocator);
            defer polygon.deinit();

            for (bad_triangles.items) |bad_tri| {
                const edges = [_]Edge{
                    Edge{ .a = bad_tri.a, .b = bad_tri.b },
                    Edge{ .a = bad_tri.b, .b = bad_tri.c },
                    Edge{ .a = bad_tri.c, .b = bad_tri.a },
                };

                for (edges) |edge| {
                    var is_shared = false;
                    for (bad_triangles.items) |other_tri| {
                        if (bad_tri.equals(other_tri)) continue;
                        if (other_tri.hasEdge(edge.a, edge.b)) {
                            is_shared = true;
                            break;
                        }
                    }
                    if (!is_shared) {
                        try polygon.append(edge);
                    }
                }
            }

            // Remove bad triangles
            var i: usize = 0;
            while (i < self.triangles.items.len) {
                var is_bad = false;
                for (bad_triangles.items) |bad_tri| {
                    if (self.triangles.items[i].equals(bad_tri)) {
                        is_bad = true;
                        break;
                    }
                }
                if (is_bad) {
                    _ = self.triangles.swapRemove(i);
                } else {
                    i += 1;
                }
            }

            // Create new triangles from polygon edges to the new point
            for (polygon.items) |edge| {
                try self.triangles.append(Triangle{
                    .a = edge.a,
                    .b = edge.b,
                    .c = point_idx,
                });
            }
        }

        /// Check if a point is inside the circumcircle of a triangle
        fn inCircumcircle(self: *Self, point: Point(T), tri: Triangle) !bool {
            const pa = self.points.items[tri.a];
            const pb = self.points.items[tri.b];
            const pc = self.points.items[tri.c];

            const ax = pa.x - point.x;
            const ay = pa.y - point.y;
            const bx = pb.x - point.x;
            const by = pb.y - point.y;
            const cx = pc.x - point.x;
            const cy = pc.y - point.y;

            const det = (ax * ax + ay * ay) * (bx * cy - cx * by) -
                (bx * bx + by * by) * (ax * cy - cx * ay) +
                (cx * cx + cy * cy) * (ax * by - bx * ay);

            return det > 0;
        }

        /// Get triangulation result as slice
        pub fn getTriangles(self: *const Self) []const Triangle {
            return self.triangles.items;
        }

        /// Validate triangulation (for testing)
        pub fn validate(self: *const Self) !void {
            // Check that all triangles reference valid vertices
            for (self.triangles.items) |tri| {
                if (tri.a >= self.points.items.len or
                    tri.b >= self.points.items.len or
                    tri.c >= self.points.items.len)
                {
                    return error.InvalidTriangle;
                }
                if (tri.a == tri.b or tri.b == tri.c or tri.c == tri.a) {
                    return error.DegenerateTriangle;
                }
            }
        }
    };
}

const Bounds = struct {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    width: f64,
    height: f64,
};

fn computeBounds(points: []const Point(f64)) Bounds {
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

    return .{
        .min_x = min_x,
        .max_x = max_x,
        .min_y = min_y,
        .max_y = max_y,
        .width = max_x - min_x,
        .height = max_y - min_y,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "Delaunay: basic triangle" {
    var dt = DelaunayTriangulation(f64).init(testing.allocator);
    defer dt.deinit();

    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 1, .y = 0 },
        .{ .x = 0.5, .y = 1 },
    };

    try dt.triangulate(&points);
    try dt.validate();

    const triangles = dt.getTriangles();
    try testing.expectEqual(@as(usize, 1), triangles.len);
}

test "Delaunay: square (4 points)" {
    var dt = DelaunayTriangulation(f64).init(testing.allocator);
    defer dt.deinit();

    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 1, .y = 0 },
        .{ .x = 1, .y = 1 },
        .{ .x = 0, .y = 1 },
    };

    try dt.triangulate(&points);
    try dt.validate();

    const triangles = dt.getTriangles();
    try testing.expectEqual(@as(usize, 2), triangles.len);
}

test "Delaunay: pentagon (5 points)" {
    var dt = DelaunayTriangulation(f64).init(testing.allocator);
    defer dt.deinit();

    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 2, .y = 0 },
        .{ .x = 3, .y = 2 },
        .{ .x = 1, .y = 3 },
        .{ .x = -1, .y = 2 },
    };

    try dt.triangulate(&points);
    try dt.validate();

    const triangles = dt.getTriangles();
    try testing.expect(triangles.len >= 3);
}

test "Delaunay: random points (10 points)" {
    var dt = DelaunayTriangulation(f64).init(testing.allocator);
    defer dt.deinit();

    const points = [_]Point(f64){
        .{ .x = 0.1, .y = 0.2 },
        .{ .x = 0.8, .y = 0.1 },
        .{ .x = 0.9, .y = 0.9 },
        .{ .x = 0.2, .y = 0.8 },
        .{ .x = 0.5, .y = 0.5 },
        .{ .x = 0.3, .y = 0.4 },
        .{ .x = 0.7, .y = 0.6 },
        .{ .x = 0.4, .y = 0.7 },
        .{ .x = 0.6, .y = 0.3 },
        .{ .x = 0.15, .y = 0.55 },
    };

    try dt.triangulate(&points);
    try dt.validate();

    const triangles = dt.getTriangles();
    // For n points, expect roughly 2n - 5 triangles (Euler's formula)
    try testing.expect(triangles.len >= 10);
}

test "Delaunay: collinear points (should fail gracefully)" {
    var dt = DelaunayTriangulation(f64).init(testing.allocator);
    defer dt.deinit();

    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 1, .y = 0 },
        .{ .x = 2, .y = 0 },
    };

    // Collinear points should still produce a result (degenerate triangulation)
    try dt.triangulate(&points);
}

test "Delaunay: insufficient points" {
    var dt = DelaunayTriangulation(f64).init(testing.allocator);
    defer dt.deinit();

    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 1, .y = 0 },
    };

    try testing.expectError(error.InsufficientPoints, dt.triangulate(&points));
}

test "Delaunay: duplicate points" {
    var dt = DelaunayTriangulation(f64).init(testing.allocator);
    defer dt.deinit();

    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 1, .y = 0 },
        .{ .x = 0, .y = 1 },
        .{ .x = 0, .y = 0 }, // duplicate
    };

    try dt.triangulate(&points);
    try dt.validate();
}

test "Delaunay: large scale (50 points)" {
    var dt = DelaunayTriangulation(f64).init(testing.allocator);
    defer dt.deinit();

    var points = std.ArrayList(Point(f64)).init(testing.allocator);
    defer points.deinit();

    // Generate grid points
    var i: usize = 0;
    while (i < 50) : (i += 1) {
        const x = @as(f64, @floatFromInt(i % 10));
        const y = @as(f64, @floatFromInt(i / 10));
        try points.append(.{ .x = x, .y = y });
    }

    try dt.triangulate(points.items);
    try dt.validate();

    const triangles = dt.getTriangles();
    try testing.expect(triangles.len > 0);
}

test "Delaunay: f32 type support" {
    var dt = DelaunayTriangulation(f32).init(testing.allocator);
    defer dt.deinit();

    const points = [_]Point(f32){
        .{ .x = 0, .y = 0 },
        .{ .x = 1, .y = 0 },
        .{ .x = 0.5, .y = 1 },
    };

    try dt.triangulate(&points);
    try dt.validate();
}

test "Delaunay: triangle count (Euler's formula)" {
    var dt = DelaunayTriangulation(f64).init(testing.allocator);
    defer dt.deinit();

    // 10 points in general position
    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 3, .y = 0 },
        .{ .x = 6, .y = 0 },
        .{ .x = 0, .y = 3 },
        .{ .x = 3, .y = 3 },
        .{ .x = 6, .y = 3 },
        .{ .x = 0, .y = 6 },
        .{ .x = 3, .y = 6 },
        .{ .x = 6, .y = 6 },
        .{ .x = 1.5, .y = 1.5 },
    };

    try dt.triangulate(&points);
    try dt.validate();

    const triangles = dt.getTriangles();
    // For convex hull in general position: T = 2n - 2 - h, where h = hull size
    // For 10 points, expect around 12-16 triangles
    try testing.expect(triangles.len >= 10);
    try testing.expect(triangles.len <= 20);
}

test "Delaunay: Point equals" {
    const p1 = Point(f64){ .x = 1.0, .y = 2.0 };
    const p2 = Point(f64){ .x = 1.0, .y = 2.0 };
    const p3 = Point(f64){ .x = 1.0, .y = 3.0 };

    try testing.expect(p1.equals(p2));
    try testing.expect(!p1.equals(p3));
}

test "Delaunay: Triangle containsVertex" {
    const tri = Triangle{ .a = 0, .b = 1, .c = 2 };

    try testing.expect(tri.containsVertex(0));
    try testing.expect(tri.containsVertex(1));
    try testing.expect(tri.containsVertex(2));
    try testing.expect(!tri.containsVertex(3));
}

test "Delaunay: Triangle hasEdge" {
    const tri = Triangle{ .a = 0, .b = 1, .c = 2 };

    try testing.expect(tri.hasEdge(0, 1));
    try testing.expect(tri.hasEdge(1, 0));
    try testing.expect(tri.hasEdge(1, 2));
    try testing.expect(tri.hasEdge(2, 0));
    try testing.expect(!tri.hasEdge(0, 3));
}

test "Delaunay: Edge equals" {
    const e1 = Edge{ .a = 0, .b = 1 };
    const e2 = Edge{ .a = 1, .b = 0 };
    const e3 = Edge{ .a = 0, .b = 2 };

    try testing.expect(e1.equals(e2));
    try testing.expect(!e1.equals(e3));
}

test "Delaunay: memory safety" {
    var dt = DelaunayTriangulation(f64).init(testing.allocator);
    defer dt.deinit();

    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 1, .y = 0 },
        .{ .x = 0.5, .y = 1 },
    };

    try dt.triangulate(&points);
    // Memory should be properly freed by deinit
}
