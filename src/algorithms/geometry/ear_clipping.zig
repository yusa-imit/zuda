/// Polygon triangulation using ear clipping algorithm
///
/// Triangulates a simple polygon (no holes, no self-intersections) into a set of triangles.
/// Uses the ear clipping method: repeatedly finds and removes "ears" (triangles formed by
/// three consecutive vertices where no other vertices lie inside).
///
/// Time: O(n²) for basic implementation, O(n³) worst case
/// Space: O(n) for auxiliary data structures
///
/// Algorithm:
/// 1. Find all convex vertices (potential ear tips)
/// 2. For each convex vertex, check if the triangle formed is an ear (no vertices inside)
/// 3. Remove the ear, add triangle to result
/// 4. Update adjacent vertices and repeat until 3 vertices remain
///
/// Reference: "Computational Geometry: Algorithms and Applications" by de Berg et al.
const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// 2D point representation
pub fn Point(comptime T: type) type {
    return struct {
        x: T,
        y: T,

        pub fn equals(self: @This(), other: @This()) bool {
            const epsilon = switch (@typeInfo(T)) {
                .Float => @as(T, 1e-9),
                else => 0,
            };
            const dx = if (self.x > other.x) self.x - other.x else other.x - self.x;
            const dy = if (self.y > other.y) self.y - other.y else other.y - self.y;
            return dx <= epsilon and dy <= epsilon;
        }
    };
}

/// Triangle represented by three vertex indices
pub const Triangle = struct {
    i0: usize,
    i1: usize,
    i2: usize,
};

/// Triangulate a simple polygon using ear clipping
///
/// Time: O(n²) average, O(n³) worst case
/// Space: O(n)
///
/// Parameters:
/// - T: numeric type (i32, f32, f64)
/// - allocator: memory allocator for result
/// - polygon: vertices in counter-clockwise order
///
/// Returns: ArrayList of Triangle (indices into original polygon)
///
/// Note: Polygon must be simple (no holes, no self-intersections)
/// and vertices should be in counter-clockwise order.
pub fn triangulate(comptime T: type, allocator: Allocator, polygon: []const Point(T)) !std.ArrayList(Triangle) {
    if (polygon.len < 3) return error.DegeneratePolygon;

    var triangles = std.ArrayList(Triangle).init(allocator);
    errdefer triangles.deinit();

    // Handle trivial case
    if (polygon.len == 3) {
        try triangles.append(.{ .i0 = 0, .i1 = 1, .i2 = 2 });
        return triangles;
    }

    // Create a list of remaining vertex indices
    var remaining = try std.ArrayList(usize).initCapacity(allocator, polygon.len);
    defer remaining.deinit();
    for (0..polygon.len) |i| {
        try remaining.append(i);
    }

    // Ear clipping: remove ears until we have a triangle
    var iterations: usize = 0;
    const max_iterations = polygon.len * polygon.len; // Safety limit
    while (remaining.items.len > 3 and iterations < max_iterations) : (iterations += 1) {
        var ear_found = false;

        // Try to find an ear
        var i: usize = 0;
        while (i < remaining.items.len) : (i += 1) {
            const prev_idx = if (i == 0) remaining.items.len - 1 else i - 1;
            const next_idx = (i + 1) % remaining.items.len;

            const v_prev = remaining.items[prev_idx];
            const v_curr = remaining.items[i];
            const v_next = remaining.items[next_idx];

            // Check if this is an ear
            if (isEar(T, polygon, remaining.items, v_prev, v_curr, v_next)) {
                // Add triangle
                try triangles.append(.{
                    .i0 = v_prev,
                    .i1 = v_curr,
                    .i2 = v_next,
                });

                // Remove the ear tip
                _ = remaining.orderedRemove(i);
                ear_found = true;
                break;
            }
        }

        if (!ear_found) {
            // No ear found - polygon may be invalid or algorithm failed
            return error.TriangulationFailed;
        }
    }

    // Add the final triangle
    if (remaining.items.len == 3) {
        try triangles.append(.{
            .i0 = remaining.items[0],
            .i1 = remaining.items[1],
            .i2 = remaining.items[2],
        });
    }

    return triangles;
}

/// Check if a vertex forms an ear
///
/// An ear is a triangle (v_prev, v_curr, v_next) where:
/// 1. The triangle is convex at v_curr
/// 2. No other polygon vertices lie inside the triangle
fn isEar(comptime T: type, polygon: []const Point(T), remaining: []const usize, v_prev: usize, v_curr: usize, v_next: usize) bool {
    const p_prev = polygon[v_prev];
    const p_curr = polygon[v_curr];
    const p_next = polygon[v_next];

    // Check if the vertex is convex (positive cross product)
    if (!isConvex(T, p_prev, p_curr, p_next)) return false;

    // Check if any other vertex lies inside the triangle
    for (remaining) |idx| {
        if (idx == v_prev or idx == v_curr or idx == v_next) continue;

        const p = polygon[idx];
        if (pointInTriangle(T, p, p_prev, p_curr, p_next)) {
            return false;
        }
    }

    return true;
}

/// Check if a vertex is convex (positive turn from prev to curr to next)
fn isConvex(comptime T: type, p_prev: Point(T), p_curr: Point(T), p_next: Point(T)) bool {
    const cross = crossProduct(T, p_prev, p_curr, p_next);
    return switch (@typeInfo(T)) {
        .Float => cross > 1e-9,
        else => cross > 0,
    };
}

/// Compute cross product of vectors (p_curr - p_prev) and (p_next - p_curr)
fn crossProduct(comptime T: type, p_prev: Point(T), p_curr: Point(T), p_next: Point(T)) T {
    const dx1 = p_curr.x - p_prev.x;
    const dy1 = p_curr.y - p_prev.y;
    const dx2 = p_next.x - p_curr.x;
    const dy2 = p_next.y - p_curr.y;
    return dx1 * dy2 - dy1 * dx2;
}

/// Check if a point lies inside a triangle using barycentric coordinates
fn pointInTriangle(comptime T: type, p: Point(T), a: Point(T), b: Point(T), c: Point(T)) bool {
    // Compute barycentric coordinates
    const v0x = c.x - a.x;
    const v0y = c.y - a.y;
    const v1x = b.x - a.x;
    const v1y = b.y - a.y;
    const v2x = p.x - a.x;
    const v2y = p.y - a.y;

    const dot00 = v0x * v0x + v0y * v0y;
    const dot01 = v0x * v1x + v0y * v1y;
    const dot02 = v0x * v2x + v0y * v2y;
    const dot11 = v1x * v1x + v1y * v1y;
    const dot12 = v1x * v2x + v1y * v2y;

    const inv_denom = switch (@typeInfo(T)) {
        .Float => blk: {
            const denom = dot00 * dot11 - dot01 * dot01;
            if (@abs(denom) < 1e-9) return false;
            break :blk 1.0 / denom;
        },
        else => blk: {
            const denom = dot00 * dot11 - dot01 * dot01;
            if (denom == 0) return false;
            break :blk denom; // Will do integer division later
        },
    };

    const u = switch (@typeInfo(T)) {
        .Float => (dot11 * dot02 - dot01 * dot12) * inv_denom,
        else => @divTrunc(dot11 * dot02 - dot01 * dot12, inv_denom),
    };
    const v = switch (@typeInfo(T)) {
        .Float => (dot00 * dot12 - dot01 * dot02) * inv_denom,
        else => @divTrunc(dot00 * dot12 - dot01 * dot02, inv_denom),
    };

    const epsilon = switch (@typeInfo(T)) {
        .Float => @as(T, 1e-9),
        else => 0,
    };

    // Check if point is inside triangle
    return (u > epsilon) and (v > epsilon) and (u + v < 1.0 - epsilon);
}

/// Calculate total area of triangulated polygon
///
/// Time: O(n) where n = number of triangles
/// Space: O(1)
pub fn triangulationArea(comptime T: type, polygon: []const Point(T), triangles: []const Triangle) T {
    var total_area: T = 0;

    for (triangles) |tri| {
        const p0 = polygon[tri.i0];
        const p1 = polygon[tri.i1];
        const p2 = polygon[tri.i2];

        // Triangle area using cross product
        const area = switch (@typeInfo(T)) {
            .Float => @abs((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)) / 2.0,
            else => blk: {
                const cross = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
                const abs_cross = if (cross < 0) -cross else cross;
                break :blk @divTrunc(abs_cross, 2);
            },
        };
        total_area += area;
    }

    return total_area;
}

// ============================================================================
// Tests
// ============================================================================

test "triangulate simple square" {
    const allocator = testing.allocator;

    const square = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 10, .y = 0 },
        .{ .x = 10, .y = 10 },
        .{ .x = 0, .y = 10 },
    };

    var triangles = try triangulate(f64, allocator, &square);
    defer triangles.deinit();

    // Should produce 2 triangles (n-2 for n vertices)
    try testing.expectEqual(@as(usize, 2), triangles.items.len);

    // Verify area
    const area = triangulationArea(f64, &square, triangles.items);
    try testing.expectApproxEqAbs(@as(f64, 100.0), area, 1e-6);
}

test "triangulate triangle - trivial case" {
    const allocator = testing.allocator;

    const triangle = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 10, .y = 0 },
        .{ .x = 5, .y = 10 },
    };

    var triangles = try triangulate(f64, allocator, &triangle);
    defer triangles.deinit();

    // Should produce 1 triangle
    try testing.expectEqual(@as(usize, 1), triangles.items.len);
    try testing.expectEqual(Triangle{ .i0 = 0, .i1 = 1, .i2 = 2 }, triangles.items[0]);
}

test "triangulate pentagon" {
    const allocator = testing.allocator;

    const pentagon = [_]Point(f64){
        .{ .x = 5, .y = 0 },
        .{ .x = 10, .y = 3 },
        .{ .x = 8, .y = 9 },
        .{ .x = 2, .y = 9 },
        .{ .x = 0, .y = 3 },
    };

    var triangles = try triangulate(f64, allocator, &pentagon);
    defer triangles.deinit();

    // Should produce 3 triangles (5-2)
    try testing.expectEqual(@as(usize, 3), triangles.items.len);
}

test "triangulate L-shaped polygon" {
    const allocator = testing.allocator;

    const l_shape = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 10, .y = 0 },
        .{ .x = 10, .y = 5 },
        .{ .x = 5, .y = 5 },
        .{ .x = 5, .y = 10 },
        .{ .x = 0, .y = 10 },
    };

    var triangles = try triangulate(f64, allocator, &l_shape);
    defer triangles.deinit();

    // Should produce 4 triangles (6-2)
    try testing.expectEqual(@as(usize, 4), triangles.items.len);

    // Verify area: L-shape = 10×5 + 5×5 = 75
    const area = triangulationArea(f64, &l_shape, triangles.items);
    try testing.expectApproxEqAbs(@as(f64, 75.0), area, 1e-6);
}

test "triangulate concave hexagon" {
    const allocator = testing.allocator;

    const hexagon = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 4, .y = 0 },
        .{ .x = 6, .y = 2 },
        .{ .x = 4, .y = 4 },
        .{ .x = 0, .y = 4 },
        .{ .x = 2, .y = 2 },
    };

    var triangles = try triangulate(f64, allocator, &hexagon);
    defer triangles.deinit();

    // Should produce 4 triangles (6-2)
    try testing.expectEqual(@as(usize, 4), triangles.items.len);
}

test "degenerate polygon - too few vertices" {
    const allocator = testing.allocator;

    const line = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 10, .y = 10 },
    };

    const result = triangulate(f64, allocator, &line);
    try testing.expectError(error.DegeneratePolygon, result);
}

test "integer coordinates" {
    const allocator = testing.allocator;

    const square = [_]Point(i32){
        .{ .x = 0, .y = 0 },
        .{ .x = 10, .y = 0 },
        .{ .x = 10, .y = 10 },
        .{ .x = 0, .y = 10 },
    };

    var triangles = try triangulate(i32, allocator, &square);
    defer triangles.deinit();

    try testing.expectEqual(@as(usize, 2), triangles.items.len);

    const area = triangulationArea(i32, &square, triangles.items);
    try testing.expectEqual(@as(i32, 100), area);
}

test "f32 support" {
    const allocator = testing.allocator;

    const triangle = [_]Point(f32){
        .{ .x = 0, .y = 0 },
        .{ .x = 6, .y = 0 },
        .{ .x = 3, .y = 4 },
    };

    var triangles = try triangulate(f32, allocator, &triangle);
    defer triangles.deinit();

    try testing.expectEqual(@as(usize, 1), triangles.items.len);

    // Area = 0.5 * base * height = 0.5 * 6 * 4 = 12
    const area = triangulationArea(f32, &triangle, triangles.items);
    try testing.expectApproxEqAbs(@as(f32, 12.0), area, 1e-4);
}

test "triangle count formula" {
    const allocator = testing.allocator;

    // Test various polygon sizes
    const sizes = [_]usize{ 3, 4, 5, 6, 7, 8 };

    for (sizes) |n| {
        // Create a simple convex polygon (regular n-gon approximation)
        var polygon = try std.ArrayList(Point(f64)).initCapacity(allocator, n);
        defer polygon.deinit();

        const radius = 10.0;
        for (0..n) |i| {
            const angle = @as(f64, @floatFromInt(i)) * 2.0 * std.math.pi / @as(f64, @floatFromInt(n));
            try polygon.append(.{
                .x = radius * @cos(angle),
                .y = radius * @sin(angle),
            });
        }

        var triangles = try triangulate(f64, allocator, polygon.items);
        defer triangles.deinit();

        // Should always be n-2 triangles
        try testing.expectEqual(n - 2, triangles.items.len);
    }
}

test "large polygon - stress test" {
    const allocator = testing.allocator;

    // Create a larger polygon (20 vertices)
    const n = 20;
    var polygon = try std.ArrayList(Point(f64)).initCapacity(allocator, n);
    defer polygon.deinit();

    const radius = 100.0;
    for (0..n) |i| {
        const angle = @as(f64, @floatFromInt(i)) * 2.0 * std.math.pi / @as(f64, @floatFromInt(n));
        try polygon.append(.{
            .x = radius * @cos(angle),
            .y = radius * @sin(angle),
        });
    }

    var triangles = try triangulate(f64, allocator, polygon.items);
    defer triangles.deinit();

    try testing.expectEqual(n - 2, triangles.items.len);
}

test "verify triangle indices validity" {
    const allocator = testing.allocator;

    const pentagon = [_]Point(f64){
        .{ .x = 5, .y = 0 },
        .{ .x = 10, .y = 3 },
        .{ .x = 8, .y = 9 },
        .{ .x = 2, .y = 9 },
        .{ .x = 0, .y = 3 },
    };

    var triangles = try triangulate(f64, allocator, &pentagon);
    defer triangles.deinit();

    // All indices should be valid (< pentagon.len)
    for (triangles.items) |tri| {
        try testing.expect(tri.i0 < pentagon.len);
        try testing.expect(tri.i1 < pentagon.len);
        try testing.expect(tri.i2 < pentagon.len);

        // Indices should be distinct
        try testing.expect(tri.i0 != tri.i1);
        try testing.expect(tri.i1 != tri.i2);
        try testing.expect(tri.i2 != tri.i0);
    }
}

test "memory safety - proper cleanup" {
    const allocator = testing.allocator;

    const square = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 10, .y = 0 },
        .{ .x = 10, .y = 10 },
        .{ .x = 0, .y = 10 },
    };

    var triangles = try triangulate(f64, allocator, &square);
    defer triangles.deinit();

    // Just verify we can create and destroy without leaks
    try testing.expect(triangles.items.len > 0);
}
