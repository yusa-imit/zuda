const std = @import("std");
const Allocator = std.mem.Allocator;
const convex_hull = @import("convex_hull.zig");

/// Re-export Point type from convex_hull for consistency
pub const Point = convex_hull.Point;

/// Helper to compute Euclidean distance between two points
fn distanceTo(comptime T: type, p1: Point(T), p2: Point(T)) T {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    return @sqrt(dx * dx + dy * dy);
}

/// Helper to compute squared distance (avoids sqrt, faster for comparisons)
fn distanceSquaredTo(comptime T: type, p1: Point(T), p2: Point(T)) T {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

/// Result of diameter computation
pub fn DiameterResult(comptime T: type) type {
    return struct {
        p1: Point(T),
        p2: Point(T),
        distance: T,
    };
}

/// Result of width computation
pub fn WidthResult(comptime T: type) type {
    return struct {
        width: T,
        support1: Point(T), // Point on first supporting line
        support2: Point(T), // Point on second supporting line
    };
}

/// Result of minimum bounding rectangle computation
pub fn MinBoundingRectResult(comptime T: type) type {
    return struct {
        area: T,
        corners: [4]Point(T),
    };
}

/// Compute the diameter of a point set using rotating calipers.
/// The diameter is the maximum distance between any two points.
///
/// Time: O(n log n) — dominated by convex hull computation
/// Space: O(n) — for convex hull storage
///
/// Note: Points must form a valid convex polygon (use convex hull first for arbitrary point sets)
pub fn diameter(comptime T: type, allocator: Allocator, points: []const Point(T)) !DiameterResult(T) {
    if (points.len < 2) return error.InsufficientPoints;

    // Compute convex hull first
    const hull = try convex_hull.grahamScan(T, allocator, points);
    defer allocator.free(hull);

    if (hull.len < 2) return error.InsufficientPoints;
    if (hull.len == 2) {
        return DiameterResult(T){
            .p1 = hull[0],
            .p2 = hull[1],
            .distance = distanceTo(T, hull[0], hull[1]),
        };
    }

    return diameterOnConvexHull(T, hull);
}

/// Compute diameter on a pre-computed convex hull.
/// Hull must be in counter-clockwise order.
///
/// Time: O(n) where n = hull size
/// Space: O(1)
pub fn diameterOnConvexHull(comptime T: type, hull: []const Point(T)) DiameterResult(T) {
    const n = hull.len;
    var max_dist: T = 0;
    var p1_idx: usize = 0;
    var p2_idx: usize = 0;

    // Find initial antipodal pair (roughly opposite vertices)
    var j: usize = 1;
    while (area2(T, hull[n - 1], hull[0], hull[j % n]) > area2(T, hull[n - 1], hull[0], hull[(j + 1) % n])) {
        j += 1;
    }

    var i: usize = 0;
    var q = j;

    // Rotate calipers around hull
    while (i < n) {
        const dist_sq = distanceSquaredTo(T, hull[i], hull[q % n]);
        if (dist_sq > max_dist) {
            max_dist = dist_sq;
            p1_idx = i;
            p2_idx = q % n;
        }

        // Determine which caliper to rotate
        const next_i = (i + 1) % n;
        const next_q = (q + 1) % n;

        if (area2(T, hull[i], hull[next_i], hull[next_q]) > area2(T, hull[q % n], hull[next_q], hull[next_i])) {
            q += 1;
        } else {
            i += 1;
        }

        if (i >= n) break;
    }

    return DiameterResult(T){
        .p1 = hull[p1_idx],
        .p2 = hull[p2_idx],
        .distance = @sqrt(max_dist),
    };
}

/// Compute the width of a point set using rotating calipers.
/// Width is the minimum distance between parallel supporting lines.
///
/// Time: O(n log n) — dominated by convex hull computation
/// Space: O(n) — for convex hull storage
pub fn width(comptime T: type, allocator: Allocator, points: []const Point(T)) !WidthResult(T) {
    if (points.len < 2) return error.InsufficientPoints;

    const hull = try convex_hull.grahamScan(T, allocator, points);
    defer allocator.free(hull);

    if (hull.len < 2) return error.InsufficientPoints;
    if (hull.len == 2) {
        return WidthResult(T){
            .width = 0,
            .support1 = hull[0],
            .support2 = hull[1],
        };
    }

    return widthOnConvexHull(T, hull);
}

/// Compute width on a pre-computed convex hull.
///
/// Time: O(n) where n = hull size
/// Space: O(1)
pub fn widthOnConvexHull(comptime T: type, hull: []const Point(T)) WidthResult(T) {
    const n = hull.len;
    var min_width: T = std.math.inf(T);
    var support1_idx: usize = 0;
    var support2_idx: usize = 0;

    var j: usize = 1;
    var i: usize = 0;

    while (i < n) {
        const next_i = (i + 1) % n;

        // Rotate caliper at j to maximize distance from edge (i, next_i)
        while (perpDist(T, hull[i], hull[next_i], hull[(j + 1) % n]) > perpDist(T, hull[i], hull[next_i], hull[j % n])) {
            j = (j + 1) % n;
        }

        const dist = perpDist(T, hull[i], hull[next_i], hull[j % n]);
        if (dist < min_width) {
            min_width = dist;
            support1_idx = i;
            support2_idx = j % n;
        }

        i += 1;
    }

    return WidthResult(T){
        .width = min_width,
        .support1 = hull[support1_idx],
        .support2 = hull[support2_idx],
    };
}

/// Compute minimum-area bounding rectangle using rotating calipers.
///
/// Time: O(n log n) — dominated by convex hull computation
/// Space: O(n) — for convex hull storage
pub fn minBoundingRect(comptime T: type, allocator: Allocator, points: []const Point(T)) !MinBoundingRectResult(T) {
    if (points.len < 3) return error.InsufficientPoints;

    const hull = try convex_hull.grahamScan(T, allocator, points);
    defer allocator.free(hull);

    if (hull.len < 3) return error.InsufficientPoints;

    return minBoundingRectOnConvexHull(T, hull);
}

/// Compute minimum-area bounding rectangle on a pre-computed convex hull.
///
/// Time: O(n) where n = hull size
/// Space: O(1)
pub fn minBoundingRectOnConvexHull(comptime T: type, hull: []const Point(T)) MinBoundingRectResult(T) {
    const n = hull.len;
    var min_area: T = std.math.inf(T);
    var best_corners: [4]Point(T) = undefined;

    // For each edge of the hull
    for (hull, 0..) |_, i| {
        const next_i = (i + 1) % n;
        const edge_x = hull[next_i].x - hull[i].x;
        const edge_y = hull[next_i].y - hull[i].y;
        const edge_len = @sqrt(edge_x * edge_x + edge_y * edge_y);

        if (edge_len < std.math.floatEps(T)) continue;

        // Normalized edge direction
        const ux = edge_x / edge_len;
        const uy = edge_y / edge_len;

        // Perpendicular direction
        const vx = -uy;
        const vy = ux;

        // Project all points onto edge and perpendicular
        var min_u: T = std.math.inf(T);
        var max_u: T = -std.math.inf(T);
        var min_v: T = std.math.inf(T);
        var max_v: T = -std.math.inf(T);

        for (hull) |p| {
            const u = (p.x - hull[i].x) * ux + (p.y - hull[i].y) * uy;
            const v = (p.x - hull[i].x) * vx + (p.y - hull[i].y) * vy;

            min_u = @min(min_u, u);
            max_u = @max(max_u, u);
            min_v = @min(min_v, v);
            max_v = @max(max_v, v);
        }

        const width_u = max_u - min_u;
        const width_v = max_v - min_v;
        const area = width_u * width_v;

        if (area < min_area) {
            min_area = area;

            // Compute rectangle corners
            const base_x = hull[i].x;
            const base_y = hull[i].y;

            best_corners[0] = Point(T){ .x = base_x + min_u * ux + min_v * vx, .y = base_y + min_u * uy + min_v * vy };
            best_corners[1] = Point(T){ .x = base_x + max_u * ux + min_v * vx, .y = base_y + max_u * uy + min_v * vy };
            best_corners[2] = Point(T){ .x = base_x + max_u * ux + max_v * vx, .y = base_y + max_u * uy + max_v * vy };
            best_corners[3] = Point(T){ .x = base_x + min_u * ux + max_v * vx, .y = base_y + min_u * uy + max_v * vy };
        }
    }

    return MinBoundingRectResult(T){
        .area = min_area,
        .corners = best_corners,
    };
}

// ==================== Helper Functions ====================

/// Compute twice the signed area of triangle (p, q, r)
fn area2(comptime T: type, p: Point(T), q: Point(T), r: Point(T)) T {
    return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);
}

/// Compute perpendicular distance from point p to line defined by a-b
fn perpDist(comptime T: type, a: Point(T), b: Point(T), p: Point(T)) T {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = @sqrt(dx * dx + dy * dy);
    if (len < std.math.floatEps(T)) return 0;

    return @abs((p.x - a.x) * dy - (p.y - a.y) * dx) / len;
}

// ==================== Tests ====================

test "rotating_calipers: diameter of square" {
    const allocator = std.testing.allocator;

    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 1, .y = 0 },
        .{ .x = 1, .y = 1 },
        .{ .x = 0, .y = 1 },
    };

    const result = try diameter(f64, allocator, &points);

    // Diameter of unit square is sqrt(2) (diagonal)
    const expected = @sqrt(2.0);
    try std.testing.expectApproxEqAbs(expected, result.distance, 1e-10);
}

test "rotating_calipers: diameter of rectangle" {
    const allocator = std.testing.allocator;

    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 3, .y = 0 },
        .{ .x = 3, .y = 1 },
        .{ .x = 0, .y = 1 },
    };

    const result = try diameter(f64, allocator, &points);

    // Diameter of 3×1 rectangle is sqrt(10) (diagonal)
    const expected = @sqrt(10.0);
    try std.testing.expectApproxEqAbs(expected, result.distance, 1e-10);
}

test "rotating_calipers: diameter of triangle" {
    const allocator = std.testing.allocator;

    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 4, .y = 0 },
        .{ .x = 2, .y = 3 },
    };

    const result = try diameter(f64, allocator, &points);

    // Maximum distance is base of triangle
    try std.testing.expectApproxEqAbs(4.0, result.distance, 1e-10);
}

test "rotating_calipers: diameter of circle (approximation)" {
    const allocator = std.testing.allocator;

    // Circle with radius 5, approximated by 8 points
    const n = 8;
    var points: [n]Point(f64) = undefined;
    const radius = 5.0;

    for (0..n) |i| {
        const angle = 2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n));
        points[i] = Point(f64){
            .x = radius * @cos(angle),
            .y = radius * @sin(angle),
        };
    }

    const result = try diameter(f64, allocator, &points);

    // Diameter should be approximately 2*radius = 10
    try std.testing.expectApproxEqAbs(10.0, result.distance, 0.1);
}

test "rotating_calipers: width of square" {
    const allocator = std.testing.allocator;

    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 2, .y = 0 },
        .{ .x = 2, .y = 2 },
        .{ .x = 0, .y = 2 },
    };

    const result = try width(f64, allocator, &points);

    // Width of square is side length
    try std.testing.expectApproxEqAbs(2.0, result.width, 1e-10);
}

test "rotating_calipers: width of rectangle" {
    const allocator = std.testing.allocator;

    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 5, .y = 0 },
        .{ .x = 5, .y = 2 },
        .{ .x = 0, .y = 2 },
    };

    const result = try width(f64, allocator, &points);

    // Width of 5×2 rectangle is smaller dimension
    try std.testing.expectApproxEqAbs(2.0, result.width, 1e-10);
}

test "rotating_calipers: width of triangle" {
    const allocator = std.testing.allocator;

    // Equilateral triangle with side length 2
    const h = @sqrt(3.0);
    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 2, .y = 0 },
        .{ .x = 1, .y = h },
    };

    const result = try width(f64, allocator, &points);

    // Width (altitude) of equilateral triangle with side 2 is sqrt(3)
    try std.testing.expectApproxEqAbs(h, result.width, 1e-10);
}

test "rotating_calipers: minimum bounding rectangle of square" {
    const allocator = std.testing.allocator;

    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 2, .y = 0 },
        .{ .x = 2, .y = 2 },
        .{ .x = 0, .y = 2 },
    };

    const result = try minBoundingRect(f64, allocator, &points);

    // Area of 2×2 square
    try std.testing.expectApproxEqAbs(4.0, result.area, 1e-10);
}

test "rotating_calipers: minimum bounding rectangle of rotated square" {
    const allocator = std.testing.allocator;

    // Square rotated 45 degrees
    const s = @sqrt(2.0) / 2.0;
    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = s, .y = s },
        .{ .x = 0, .y = 2 * s },
        .{ .x = -s, .y = s },
    };

    const result = try minBoundingRect(f64, allocator, &points);

    // Area should be 1 (side length sqrt(2) * sqrt(2) / 2 on each side)
    try std.testing.expectApproxEqAbs(1.0, result.area, 1e-9);
}

test "rotating_calipers: minimum bounding rectangle of triangle" {
    const allocator = std.testing.allocator;

    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 4, .y = 0 },
        .{ .x = 2, .y = 3 },
    };

    const result = try minBoundingRect(f64, allocator, &points);

    // Area should be less than or equal to 4*3 = 12 (axis-aligned bounding box)
    try std.testing.expect(result.area <= 12.0 + 1e-10);
    try std.testing.expect(result.area > 0);
}

test "rotating_calipers: f32 support" {
    const allocator = std.testing.allocator;

    const points = [_]Point(f32){
        .{ .x = 0, .y = 0 },
        .{ .x = 1, .y = 0 },
        .{ .x = 1, .y = 1 },
        .{ .x = 0, .y = 1 },
    };

    const diam = try diameter(f32, allocator, &points);
    const w = try width(f32, allocator, &points);
    const rect = try minBoundingRect(f32, allocator, &points);

    try std.testing.expectApproxEqAbs(@as(f32, @sqrt(2.0)), diam.distance, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), w.width, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), rect.area, 1e-6);
}

test "rotating_calipers: error on insufficient points" {
    const allocator = std.testing.allocator;

    const points = [_]Point(f64){.{ .x = 0, .y = 0 }};

    try std.testing.expectError(error.InsufficientPoints, diameter(f64, allocator, &points));
    try std.testing.expectError(error.InsufficientPoints, width(f64, allocator, &points));
    try std.testing.expectError(error.InsufficientPoints, minBoundingRect(f64, allocator, &points));
}

test "rotating_calipers: two points" {
    const allocator = std.testing.allocator;

    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 3, .y = 4 },
    };

    const result = try diameter(f64, allocator, &points);

    // Distance between (0,0) and (3,4) is 5
    try std.testing.expectApproxEqAbs(5.0, result.distance, 1e-10);
}

test "rotating_calipers: large scale validation" {
    const allocator = std.testing.allocator;

    // Regular polygon with 100 points
    const n = 100;
    var points = try allocator.alloc(Point(f64), n);
    defer allocator.free(points);

    const radius = 10.0;
    for (0..n) |i| {
        const angle = 2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n));
        points[i] = Point(f64){
            .x = radius * @cos(angle),
            .y = radius * @sin(angle),
        };
    }

    const diam = try diameter(f64, allocator, points);
    const w = try width(f64, allocator, points);
    const rect = try minBoundingRect(f64, allocator, points);

    // Diameter should be approximately 2*radius
    try std.testing.expectApproxEqAbs(20.0, diam.distance, 0.1);

    // Width should be approximately 2*radius
    try std.testing.expectApproxEqAbs(20.0, w.width, 0.1);

    // Bounding rectangle area should be approximately (2*radius)^2
    try std.testing.expectApproxEqAbs(400.0, rect.area, 5.0);
}

test "rotating_calipers: collinear points" {
    const allocator = std.testing.allocator;

    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 1, .y = 1 },
        .{ .x = 2, .y = 2 },
        .{ .x = 3, .y = 3 },
    };

    const result = try diameter(f64, allocator, &points);

    // Diameter of collinear points is distance between endpoints
    const expected = @sqrt(18.0); // distance from (0,0) to (3,3)
    try std.testing.expectApproxEqAbs(expected, result.distance, 1e-10);
}

test "rotating_calipers: memory safety" {
    const allocator = std.testing.allocator;

    const points = [_]Point(f64){
        .{ .x = 0, .y = 0 },
        .{ .x = 1, .y = 0 },
        .{ .x = 0.5, .y = 1 },
    };

    // Multiple operations should not leak
    _ = try diameter(f64, allocator, &points);
    _ = try width(f64, allocator, &points);
    _ = try minBoundingRect(f64, allocator, &points);
}
