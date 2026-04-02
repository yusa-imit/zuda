//! Polygon Algorithms
//!
//! This module provides fundamental algorithms for polygon geometry including:
//! - Area calculation using the Shoelace formula
//! - Point-in-Polygon test using ray casting
//! - Perimeter calculation
//! - Centroid computation
//!
//! Reference: Computational Geometry: Algorithms and Applications (de Berg et al., 2008)

const std = @import("std");
const testing = std.testing;
const math = std.math;

/// 2D Point representation
pub const Point2D = struct {
    x: f64,
    y: f64,

    pub fn init(x: f64, y: f64) Point2D {
        return .{ .x = x, .y = y };
    }
};

/// Calculate the signed area of a polygon using the Shoelace formula.
/// The area is positive if vertices are in counter-clockwise order,
/// negative if clockwise.
///
/// Time: O(n) where n is the number of vertices
/// Space: O(1)
///
/// Example:
/// ```zig
/// const vertices = [_]Point2D{
///     Point2D.init(0, 0),
///     Point2D.init(4, 0),
///     Point2D.init(4, 3),
///     Point2D.init(0, 3),
/// };
/// const area = signedArea(&vertices); // Returns 12.0 (4×3)
/// ```
pub fn signedArea(vertices: []const Point2D) f64 {
    if (vertices.len < 3) return 0.0;

    var sum: f64 = 0.0;
    const n = vertices.len;

    for (0..n) |i| {
        const j = (i + 1) % n;
        sum += vertices[i].x * vertices[j].y;
        sum -= vertices[j].x * vertices[i].y;
    }

    return sum / 2.0;
}

/// Calculate the absolute area of a polygon.
///
/// Time: O(n) where n is the number of vertices
/// Space: O(1)
///
/// Example:
/// ```zig
/// const vertices = [_]Point2D{
///     Point2D.init(0, 0),
///     Point2D.init(4, 0),
///     Point2D.init(4, 3),
///     Point2D.init(0, 3),
/// };
/// const area = polygonArea(&vertices); // Returns 12.0
/// ```
pub fn polygonArea(vertices: []const Point2D) f64 {
    return @abs(signedArea(vertices));
}

/// Calculate the perimeter of a polygon.
///
/// Time: O(n) where n is the number of vertices
/// Space: O(1)
///
/// Example:
/// ```zig
/// const vertices = [_]Point2D{
///     Point2D.init(0, 0),
///     Point2D.init(3, 0),
///     Point2D.init(3, 4),
///     Point2D.init(0, 4),
/// };
/// const perim = perimeter(&vertices); // Returns 14.0
/// ```
pub fn perimeter(vertices: []const Point2D) f64 {
    if (vertices.len < 2) return 0.0;

    var sum: f64 = 0.0;
    const n = vertices.len;

    for (0..n) |i| {
        const j = (i + 1) % n;
        const dx = vertices[j].x - vertices[i].x;
        const dy = vertices[j].y - vertices[i].y;
        sum += @sqrt(dx * dx + dy * dy);
    }

    return sum;
}

/// Calculate the centroid (geometric center) of a polygon.
/// The polygon must be non-self-intersecting.
///
/// Time: O(n) where n is the number of vertices
/// Space: O(1)
///
/// Example:
/// ```zig
/// const vertices = [_]Point2D{
///     Point2D.init(0, 0),
///     Point2D.init(2, 0),
///     Point2D.init(2, 2),
///     Point2D.init(0, 2),
/// };
/// const c = centroid(&vertices); // Returns Point2D{.x=1, .y=1}
/// ```
pub fn centroid(vertices: []const Point2D) Point2D {
    if (vertices.len == 0) return Point2D.init(0, 0);
    if (vertices.len == 1) return vertices[0];
    if (vertices.len == 2) {
        return Point2D.init(
            (vertices[0].x + vertices[1].x) / 2.0,
            (vertices[0].y + vertices[1].y) / 2.0,
        );
    }

    var cx: f64 = 0.0;
    var cy: f64 = 0.0;
    var area: f64 = 0.0;
    const n = vertices.len;

    for (0..n) |i| {
        const j = (i + 1) % n;
        const cross = vertices[i].x * vertices[j].y - vertices[j].x * vertices[i].y;
        area += cross;
        cx += (vertices[i].x + vertices[j].x) * cross;
        cy += (vertices[i].y + vertices[j].y) * cross;
    }

    area /= 2.0;
    if (@abs(area) < 1e-10) {
        // Degenerate polygon, use arithmetic mean
        var sum_x: f64 = 0.0;
        var sum_y: f64 = 0.0;
        for (vertices) |v| {
            sum_x += v.x;
            sum_y += v.y;
        }
        return Point2D.init(sum_x / @as(f64, @floatFromInt(n)), sum_y / @as(f64, @floatFromInt(n)));
    }

    cx /= (6.0 * area);
    cy /= (6.0 * area);
    return Point2D.init(cx, cy);
}

/// Test if a point is inside a polygon using the ray casting algorithm.
/// The polygon must be closed (first and last points connected implicitly).
///
/// Time: O(n) where n is the number of vertices
/// Space: O(1)
///
/// Algorithm: Cast a horizontal ray from the point to infinity and count
/// how many times it crosses the polygon boundary. Odd = inside, even = outside.
///
/// Example:
/// ```zig
/// const square = [_]Point2D{
///     Point2D.init(0, 0),
///     Point2D.init(4, 0),
///     Point2D.init(4, 4),
///     Point2D.init(0, 4),
/// };
/// const inside = pointInPolygon(Point2D.init(2, 2), &square); // true
/// const outside = pointInPolygon(Point2D.init(5, 2), &square); // false
/// ```
pub fn pointInPolygon(point: Point2D, vertices: []const Point2D) bool {
    if (vertices.len < 3) return false;

    var inside = false;
    const n = vertices.len;

    for (0..n) |i| {
        const j = (i + 1) % n;
        const vi = vertices[i];
        const vj = vertices[j];

        // Check if point is on the edge boundary
        if (pointOnSegment(point, vi, vj)) return true;

        // Ray casting: count crossings
        const yi_above = vi.y > point.y;
        const yj_above = vj.y > point.y;

        if (yi_above != yj_above) {
            // Edge crosses the horizontal ray from point
            // Check if crossing is to the right of point
            const slope = (vj.x - vi.x) / (vj.y - vi.y);
            const x_intersect = vi.x + slope * (point.y - vi.y);

            if (point.x < x_intersect) {
                inside = !inside;
            }
        }
    }

    return inside;
}

/// Test if a point lies on a line segment.
///
/// Time: O(1)
/// Space: O(1)
fn pointOnSegment(point: Point2D, a: Point2D, b: Point2D) bool {
    const epsilon = 1e-10;

    // Check if point is collinear with segment
    const cross = (point.y - a.y) * (b.x - a.x) - (point.x - a.x) * (b.y - a.y);
    if (@abs(cross) > epsilon) return false;

    // Check if point is within segment bounds
    const dot = (point.x - a.x) * (b.x - a.x) + (point.y - a.y) * (b.y - a.y);
    const len_sq = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y);

    return dot >= -epsilon and dot <= len_sq + epsilon;
}

/// Check if a polygon is convex.
/// A polygon is convex if all interior angles are less than 180 degrees.
///
/// Time: O(n) where n is the number of vertices
/// Space: O(1)
///
/// Example:
/// ```zig
/// const square = [_]Point2D{
///     Point2D.init(0, 0),
///     Point2D.init(1, 0),
///     Point2D.init(1, 1),
///     Point2D.init(0, 1),
/// };
/// const is_convex = isConvex(&square); // true
/// ```
pub fn isConvex(vertices: []const Point2D) bool {
    if (vertices.len < 3) return false;

    var sign: ?bool = null;
    const n = vertices.len;

    for (0..n) |i| {
        const j = (i + 1) % n;
        const k = (i + 2) % n;

        const v1x = vertices[j].x - vertices[i].x;
        const v1y = vertices[j].y - vertices[i].y;
        const v2x = vertices[k].x - vertices[j].x;
        const v2y = vertices[k].y - vertices[j].y;

        // Cross product determines turn direction
        const cross = v1x * v2y - v1y * v2x;

        if (@abs(cross) > 1e-10) {
            const current_sign = cross > 0;
            if (sign == null) {
                sign = current_sign;
            } else if (sign.? != current_sign) {
                return false; // Direction changed, not convex
            }
        }
    }

    return true;
}

// ============================================================================
// Tests
// ============================================================================

test "signedArea - square counter-clockwise" {
    const square = [_]Point2D{
        Point2D.init(0, 0),
        Point2D.init(4, 0),
        Point2D.init(4, 3),
        Point2D.init(0, 3),
    };
    const area = signedArea(&square);
    try testing.expectApproxEqAbs(12.0, area, 1e-10);
}

test "signedArea - square clockwise" {
    const square = [_]Point2D{
        Point2D.init(0, 0),
        Point2D.init(0, 3),
        Point2D.init(4, 3),
        Point2D.init(4, 0),
    };
    const area = signedArea(&square);
    try testing.expectApproxEqAbs(-12.0, area, 1e-10);
}

test "signedArea - triangle" {
    const triangle = [_]Point2D{
        Point2D.init(0, 0),
        Point2D.init(4, 0),
        Point2D.init(2, 3),
    };
    const area = signedArea(&triangle);
    try testing.expectApproxEqAbs(6.0, area, 1e-10);
}

test "signedArea - degenerate cases" {
    const empty: [0]Point2D = .{};
    try testing.expectEqual(0.0, signedArea(&empty));

    const single = [_]Point2D{Point2D.init(1, 1)};
    try testing.expectEqual(0.0, signedArea(&single));

    const line = [_]Point2D{ Point2D.init(0, 0), Point2D.init(1, 1) };
    try testing.expectEqual(0.0, signedArea(&line));
}

test "polygonArea - various shapes" {
    const square = [_]Point2D{
        Point2D.init(0, 0),
        Point2D.init(4, 0),
        Point2D.init(4, 3),
        Point2D.init(0, 3),
    };
    try testing.expectApproxEqAbs(12.0, polygonArea(&square), 1e-10);

    const triangle = [_]Point2D{
        Point2D.init(0, 0),
        Point2D.init(6, 0),
        Point2D.init(3, 4),
    };
    try testing.expectApproxEqAbs(12.0, polygonArea(&triangle), 1e-10);
}

test "perimeter - square" {
    const square = [_]Point2D{
        Point2D.init(0, 0),
        Point2D.init(3, 0),
        Point2D.init(3, 4),
        Point2D.init(0, 4),
    };
    const perim = perimeter(&square);
    try testing.expectApproxEqAbs(14.0, perim, 1e-10);
}

test "perimeter - triangle" {
    const triangle = [_]Point2D{
        Point2D.init(0, 0),
        Point2D.init(3, 0),
        Point2D.init(0, 4),
    };
    const perim = perimeter(&triangle);
    try testing.expectApproxEqAbs(12.0, perim, 1e-10); // 3 + 4 + 5
}

test "perimeter - degenerate cases" {
    const empty: [0]Point2D = .{};
    try testing.expectEqual(0.0, perimeter(&empty));

    const single = [_]Point2D{Point2D.init(1, 1)};
    try testing.expectEqual(0.0, perimeter(&single));
}

test "centroid - square" {
    const square = [_]Point2D{
        Point2D.init(0, 0),
        Point2D.init(2, 0),
        Point2D.init(2, 2),
        Point2D.init(0, 2),
    };
    const c = centroid(&square);
    try testing.expectApproxEqAbs(1.0, c.x, 1e-10);
    try testing.expectApproxEqAbs(1.0, c.y, 1e-10);
}

test "centroid - triangle" {
    const triangle = [_]Point2D{
        Point2D.init(0, 0),
        Point2D.init(6, 0),
        Point2D.init(3, 6),
    };
    const c = centroid(&triangle);
    try testing.expectApproxEqAbs(3.0, c.x, 1e-10);
    try testing.expectApproxEqAbs(2.0, c.y, 1e-10);
}

test "centroid - degenerate cases" {
    const empty: [0]Point2D = .{};
    const c1 = centroid(&empty);
    try testing.expectEqual(0.0, c1.x);
    try testing.expectEqual(0.0, c1.y);

    const single = [_]Point2D{Point2D.init(3, 4)};
    const c2 = centroid(&single);
    try testing.expectEqual(3.0, c2.x);
    try testing.expectEqual(4.0, c2.y);

    const line = [_]Point2D{ Point2D.init(0, 0), Point2D.init(4, 4) };
    const c3 = centroid(&line);
    try testing.expectApproxEqAbs(2.0, c3.x, 1e-10);
    try testing.expectApproxEqAbs(2.0, c3.y, 1e-10);
}

test "pointInPolygon - square" {
    const square = [_]Point2D{
        Point2D.init(0, 0),
        Point2D.init(4, 0),
        Point2D.init(4, 4),
        Point2D.init(0, 4),
    };

    // Points inside
    try testing.expect(pointInPolygon(Point2D.init(2, 2), &square));
    try testing.expect(pointInPolygon(Point2D.init(1, 1), &square));
    try testing.expect(pointInPolygon(Point2D.init(3.5, 3.5), &square));

    // Points outside
    try testing.expect(!pointInPolygon(Point2D.init(5, 2), &square));
    try testing.expect(!pointInPolygon(Point2D.init(-1, 2), &square));
    try testing.expect(!pointInPolygon(Point2D.init(2, 5), &square));
    try testing.expect(!pointInPolygon(Point2D.init(2, -1), &square));

    // Points on boundary
    try testing.expect(pointInPolygon(Point2D.init(0, 0), &square));
    try testing.expect(pointInPolygon(Point2D.init(2, 0), &square));
    try testing.expect(pointInPolygon(Point2D.init(4, 2), &square));
}

test "pointInPolygon - triangle" {
    const triangle = [_]Point2D{
        Point2D.init(0, 0),
        Point2D.init(4, 0),
        Point2D.init(2, 3),
    };

    try testing.expect(pointInPolygon(Point2D.init(2, 1), &triangle));
    try testing.expect(pointInPolygon(Point2D.init(1, 0.5), &triangle));

    try testing.expect(!pointInPolygon(Point2D.init(0, 2), &triangle));
    try testing.expect(!pointInPolygon(Point2D.init(4, 2), &triangle));
}

test "pointInPolygon - concave polygon" {
    // L-shaped polygon: (0,0) → (3,0) → (3,2) → (1,2) → (1,3) → (0,3) → (0,0)
    const concave = [_]Point2D{
        Point2D.init(0, 0),
        Point2D.init(3, 0),
        Point2D.init(3, 2),
        Point2D.init(1, 2),
        Point2D.init(1, 3),
        Point2D.init(0, 3),
    };

    // Points definitely inside
    try testing.expect(pointInPolygon(Point2D.init(0.5, 0.5), &concave));
    try testing.expect(pointInPolygon(Point2D.init(2, 1), &concave));
    try testing.expect(pointInPolygon(Point2D.init(0.5, 2.5), &concave));

    // Point in the cutout region (outside)
    try testing.expect(!pointInPolygon(Point2D.init(2, 2.5), &concave));
    try testing.expect(!pointInPolygon(Point2D.init(2.5, 2.5), &concave));
}

test "pointInPolygon - degenerate cases" {
    const line = [_]Point2D{ Point2D.init(0, 0), Point2D.init(4, 0) };
    try testing.expect(!pointInPolygon(Point2D.init(2, 1), &line));

    const single = [_]Point2D{Point2D.init(1, 1)};
    try testing.expect(!pointInPolygon(Point2D.init(1, 1), &single));
}

test "isConvex - convex shapes" {
    const square = [_]Point2D{
        Point2D.init(0, 0),
        Point2D.init(1, 0),
        Point2D.init(1, 1),
        Point2D.init(0, 1),
    };
    try testing.expect(isConvex(&square));

    const triangle = [_]Point2D{
        Point2D.init(0, 0),
        Point2D.init(2, 0),
        Point2D.init(1, 2),
    };
    try testing.expect(isConvex(&triangle));
}

test "isConvex - concave shapes" {
    const concave = [_]Point2D{
        Point2D.init(0, 0),
        Point2D.init(4, 0),
        Point2D.init(4, 4),
        Point2D.init(2, 2), // Creates concave shape
        Point2D.init(0, 4),
    };
    try testing.expect(!isConvex(&concave));
}

test "isConvex - degenerate cases" {
    const line = [_]Point2D{ Point2D.init(0, 0), Point2D.init(1, 1) };
    try testing.expect(!isConvex(&line));

    const single = [_]Point2D{Point2D.init(1, 1)};
    try testing.expect(!isConvex(&single));
}

test "pointOnSegment - various cases" {
    const a = Point2D.init(0, 0);
    const b = Point2D.init(4, 4);

    // Point on segment
    try testing.expect(pointOnSegment(Point2D.init(2, 2), a, b));
    try testing.expect(pointOnSegment(Point2D.init(0, 0), a, b)); // Endpoint
    try testing.expect(pointOnSegment(Point2D.init(4, 4), a, b)); // Endpoint

    // Point on line but not segment
    try testing.expect(!pointOnSegment(Point2D.init(-1, -1), a, b));
    try testing.expect(!pointOnSegment(Point2D.init(5, 5), a, b));

    // Point not on line
    try testing.expect(!pointOnSegment(Point2D.init(2, 3), a, b));
}

test "polygon - integration test" {
    const pentagon = [_]Point2D{
        Point2D.init(0, 0),
        Point2D.init(4, 0),
        Point2D.init(5, 3),
        Point2D.init(2, 5),
        Point2D.init(-1, 3),
    };

    // Area should be positive (counter-clockwise)
    const area = polygonArea(&pentagon);
    try testing.expect(area > 0);

    // Centroid should be inside
    const c = centroid(&pentagon);
    try testing.expect(pointInPolygon(c, &pentagon));

    // Perimeter should be reasonable
    const perim = perimeter(&pentagon);
    try testing.expect(perim > 0);
}
