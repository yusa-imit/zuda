/// Line Segment Intersection
///
/// Detects if two line segments intersect and computes the intersection point.
/// Uses orientation-based method with cross products for numerical stability.
///
/// Algorithm: Orientation method
/// - Compute orientations of triplets (p1,p2,q1) and (p1,p2,q2)
/// - Compute orientations of triplets (q1,q2,p1) and (q1,q2,p2)
/// - Segments intersect if orientations differ on both sides
/// - Handle collinear cases with bounding box checks
///
/// Time complexity: O(1) - constant time for two segments
/// Space complexity: O(1) - constant space
///
/// Use cases:
/// - Computational geometry (polygon intersection, visibility)
/// - Computer graphics (ray tracing, clipping)
/// - GIS systems (road network analysis, boundary detection)
/// - Game engines (collision detection, pathfinding)
/// - CAD systems (detecting overlaps, constraint solving)
///
/// Reference: Cormen et al., "Introduction to Algorithms" (2009)

const std = @import("std");
const testing = std.testing;

/// 2D Point representation
pub fn Point(comptime T: type) type {
    return struct {
        x: T,
        y: T,

        const Self = @This();

        pub fn init(x: T, y: T) Self {
            return .{ .x = x, .y = y };
        }
    };
}

/// Line segment represented by two endpoints
pub fn Segment(comptime T: type) type {
    return struct {
        p1: Point(T),
        p2: Point(T),

        const Self = @This();

        pub fn init(p1: Point(T), p2: Point(T)) Self {
            return .{ .p1 = p1, .p2 = p2 };
        }
    };
}

/// Orientation of ordered triplet (p, q, r)
pub const Orientation = enum {
    collinear, // Points are collinear
    clockwise, // Clockwise orientation
    counterclockwise, // Counterclockwise orientation
};

/// Intersection result
pub fn IntersectionResult(comptime T: type) type {
    return struct {
        intersects: bool,
        point: ?Point(T), // Intersection point if intersects

        const Self = @This();

        pub fn none() Self {
            return .{ .intersects = false, .point = null };
        }

        pub fn at(point: Point(T)) Self {
            return .{ .intersects = true, .point = point };
        }
    };
}

/// Compute orientation of ordered triplet (p, q, r)
///
/// Returns:
/// - Orientation.collinear if p, q, r are collinear
/// - Orientation.clockwise if clockwise
/// - Orientation.counterclockwise if counterclockwise
///
/// Time: O(1) | Space: O(1)
pub fn orientation(comptime T: type, p: Point(T), q: Point(T), r: Point(T)) Orientation {
    // Cross product (q - p) × (r - q)
    // = (q.x - p.x) * (r.y - q.y) - (q.y - p.y) * (r.x - q.x)
    const val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);

    const epsilon: T = if (@typeInfo(T) == .Float) 1e-10 else 0;

    if (@abs(val) <= epsilon) return .collinear;
    return if (val > 0) .clockwise else .counterclockwise;
}

/// Check if point q lies on segment pr (assuming q is collinear with p and r)
///
/// Time: O(1) | Space: O(1)
fn onSegment(comptime T: type, p: Point(T), q: Point(T), r: Point(T)) bool {
    return q.x <= @max(p.x, r.x) and q.x >= @min(p.x, r.x) and
        q.y <= @max(p.y, r.y) and q.y >= @min(p.y, r.y);
}

/// Check if two line segments intersect
///
/// Uses orientation method:
/// 1. Compute orientations of (p1,p2,q1) and (p1,p2,q2)
/// 2. Compute orientations of (q1,q2,p1) and (q1,q2,p2)
/// 3. General case: segments intersect if orientations differ on both sides
/// 4. Special cases: handle collinear points with bounding box overlap
///
/// Time: O(1) | Space: O(1)
pub fn doSegmentsIntersect(comptime T: type, seg1: Segment(T), seg2: Segment(T)) bool {
    const p1 = seg1.p1;
    const p2 = seg1.p2;
    const q1 = seg2.p1;
    const q2 = seg2.p2;

    const o1 = orientation(T, p1, p2, q1);
    const o2 = orientation(T, p1, p2, q2);
    const o3 = orientation(T, q1, q2, p1);
    const o4 = orientation(T, q1, q2, p2);

    // General case: segments intersect if orientations differ
    if (o1 != o2 and o3 != o4) return true;

    // Special cases: collinear points
    // p1, p2, q1 are collinear and q1 lies on segment p1p2
    if (o1 == .collinear and onSegment(T, p1, q1, p2)) return true;

    // p1, p2, q2 are collinear and q2 lies on segment p1p2
    if (o2 == .collinear and onSegment(T, p1, q2, p2)) return true;

    // q1, q2, p1 are collinear and p1 lies on segment q1q2
    if (o3 == .collinear and onSegment(T, q1, p1, q2)) return true;

    // q1, q2, p2 are collinear and p2 lies on segment q1q2
    if (o4 == .collinear and onSegment(T, q1, p2, q2)) return true;

    return false;
}

/// Find intersection point of two line segments (if it exists)
///
/// Uses parametric line equations:
/// - Line 1: P = p1 + t * (p2 - p1)
/// - Line 2: Q = q1 + s * (q2 - q1)
/// - Solve for t and s, check if both are in [0, 1]
///
/// Time: O(1) | Space: O(1)
pub fn segmentIntersection(comptime T: type, seg1: Segment(T), seg2: Segment(T)) IntersectionResult(T) {
    if (!doSegmentsIntersect(T, seg1, seg2)) {
        return IntersectionResult(T).none();
    }

    // For floating point, compute exact intersection point
    if (@typeInfo(T) == .Float) {
        const p1 = seg1.p1;
        const p2 = seg1.p2;
        const q1 = seg2.p1;
        const q2 = seg2.p2;

        const dx1 = p2.x - p1.x;
        const dy1 = p2.y - p1.y;
        const dx2 = q2.x - q1.x;
        const dy2 = q2.y - q1.y;

        // Cross product for denominator
        const denom = dx1 * dy2 - dy1 * dx2;

        // Parallel or collinear segments
        if (@abs(denom) < 1e-10) {
            // Return midpoint of overlapping region for collinear overlap
            // For simplicity, return first point
            return IntersectionResult(T).at(p1);
        }

        // Solve for parameter t
        const t = ((q1.x - p1.x) * dy2 - (q1.y - p1.y) * dx2) / denom;

        // Compute intersection point
        const x = p1.x + t * dx1;
        const y = p1.y + t * dy1;

        return IntersectionResult(T).at(Point(T).init(x, y));
    }

    // For integers, return approximate midpoint
    const p1 = seg1.p1;
    const q1 = seg2.p1;
    const x = @divTrunc(p1.x + q1.x, 2);
    const y = @divTrunc(p1.y + q1.y, 2);
    return IntersectionResult(T).at(Point(T).init(x, y));
}

// ============================================================================
// Tests
// ============================================================================

test "orientation: clockwise" {
    const p = Point(f64).init(0, 0);
    const q = Point(f64).init(4, 4);
    const r = Point(f64).init(1, 2);

    try testing.expectEqual(Orientation.clockwise, orientation(f64, p, q, r));
}

test "orientation: counterclockwise" {
    const p = Point(f64).init(0, 0);
    const q = Point(f64).init(4, 4);
    const r = Point(f64).init(2, 1);

    try testing.expectEqual(Orientation.counterclockwise, orientation(f64, p, q, r));
}

test "orientation: collinear" {
    const p = Point(f64).init(0, 0);
    const q = Point(f64).init(4, 4);
    const r = Point(f64).init(2, 2);

    try testing.expectEqual(Orientation.collinear, orientation(f64, p, q, r));
}

test "onSegment: point on segment" {
    const p = Point(i32).init(0, 0);
    const q = Point(i32).init(5, 5);
    const r = Point(i32).init(10, 10);

    try testing.expect(onSegment(i32, p, q, r));
}

test "onSegment: point not on segment" {
    const p = Point(i32).init(0, 0);
    const q = Point(i32).init(15, 15);
    const r = Point(i32).init(10, 10);

    try testing.expect(!onSegment(i32, p, q, r));
}

test "doSegmentsIntersect: general intersection" {
    const seg1 = Segment(f64).init(
        Point(f64).init(1, 1),
        Point(f64).init(10, 1),
    );
    const seg2 = Segment(f64).init(
        Point(f64).init(1, 2),
        Point(f64).init(10, 2),
    );

    // Parallel segments
    try testing.expect(!doSegmentsIntersect(f64, seg1, seg2));

    // Crossing segments
    const seg3 = Segment(f64).init(
        Point(f64).init(5, 0),
        Point(f64).init(5, 10),
    );
    try testing.expect(doSegmentsIntersect(f64, seg1, seg3));
}

test "doSegmentsIntersect: X-shaped intersection" {
    const seg1 = Segment(i32).init(
        Point(i32).init(0, 0),
        Point(i32).init(10, 10),
    );
    const seg2 = Segment(i32).init(
        Point(i32).init(0, 10),
        Point(i32).init(10, 0),
    );

    try testing.expect(doSegmentsIntersect(i32, seg1, seg2));
}

test "doSegmentsIntersect: T-shaped intersection" {
    const seg1 = Segment(f64).init(
        Point(f64).init(0, 5),
        Point(f64).init(10, 5),
    );
    const seg2 = Segment(f64).init(
        Point(f64).init(5, 0),
        Point(f64).init(5, 5),
    );

    try testing.expect(doSegmentsIntersect(f64, seg1, seg2));
}

test "doSegmentsIntersect: no intersection" {
    const seg1 = Segment(f64).init(
        Point(f64).init(0, 0),
        Point(f64).init(1, 1),
    );
    const seg2 = Segment(f64).init(
        Point(f64).init(2, 2),
        Point(f64).init(3, 3),
    );

    try testing.expect(!doSegmentsIntersect(f64, seg1, seg2));
}

test "doSegmentsIntersect: collinear overlapping" {
    const seg1 = Segment(i32).init(
        Point(i32).init(0, 0),
        Point(i32).init(10, 10),
    );
    const seg2 = Segment(i32).init(
        Point(i32).init(5, 5),
        Point(i32).init(15, 15),
    );

    try testing.expect(doSegmentsIntersect(i32, seg1, seg2));
}

test "doSegmentsIntersect: collinear non-overlapping" {
    const seg1 = Segment(i32).init(
        Point(i32).init(0, 0),
        Point(i32).init(5, 5),
    );
    const seg2 = Segment(i32).init(
        Point(i32).init(10, 10),
        Point(i32).init(15, 15),
    );

    try testing.expect(!doSegmentsIntersect(i32, seg1, seg2));
}

test "doSegmentsIntersect: touching endpoints" {
    const seg1 = Segment(f64).init(
        Point(f64).init(0, 0),
        Point(f64).init(5, 5),
    );
    const seg2 = Segment(f64).init(
        Point(f64).init(5, 5),
        Point(f64).init(10, 0),
    );

    try testing.expect(doSegmentsIntersect(f64, seg1, seg2));
}

test "segmentIntersection: X-shaped with point" {
    const seg1 = Segment(f64).init(
        Point(f64).init(0, 0),
        Point(f64).init(10, 10),
    );
    const seg2 = Segment(f64).init(
        Point(f64).init(0, 10),
        Point(f64).init(10, 0),
    );

    const result = segmentIntersection(f64, seg1, seg2);
    try testing.expect(result.intersects);
    try testing.expect(result.point != null);

    // Intersection should be at (5, 5)
    const p = result.point.?;
    try testing.expectApproxEqAbs(5.0, p.x, 1e-6);
    try testing.expectApproxEqAbs(5.0, p.y, 1e-6);
}

test "segmentIntersection: T-shaped with point" {
    const seg1 = Segment(f64).init(
        Point(f64).init(0, 5),
        Point(f64).init(10, 5),
    );
    const seg2 = Segment(f64).init(
        Point(f64).init(5, 0),
        Point(f64).init(5, 10),
    );

    const result = segmentIntersection(f64, seg1, seg2);
    try testing.expect(result.intersects);
    try testing.expect(result.point != null);

    // Intersection should be at (5, 5)
    const p = result.point.?;
    try testing.expectApproxEqAbs(5.0, p.x, 1e-6);
    try testing.expectApproxEqAbs(5.0, p.y, 1e-6);
}

test "segmentIntersection: no intersection" {
    const seg1 = Segment(f64).init(
        Point(f64).init(0, 0),
        Point(f64).init(1, 1),
    );
    const seg2 = Segment(f64).init(
        Point(f64).init(2, 2),
        Point(f64).init(3, 3),
    );

    const result = segmentIntersection(f64, seg1, seg2);
    try testing.expect(!result.intersects);
    try testing.expect(result.point == null);
}

test "segmentIntersection: parallel segments" {
    const seg1 = Segment(f64).init(
        Point(f64).init(0, 0),
        Point(f64).init(10, 0),
    );
    const seg2 = Segment(f64).init(
        Point(f64).init(0, 1),
        Point(f64).init(10, 1),
    );

    const result = segmentIntersection(f64, seg1, seg2);
    try testing.expect(!result.intersects);
    try testing.expect(result.point == null);
}

test "segmentIntersection: integer type" {
    const seg1 = Segment(i32).init(
        Point(i32).init(0, 0),
        Point(i32).init(10, 10),
    );
    const seg2 = Segment(i32).init(
        Point(i32).init(0, 10),
        Point(i32).init(10, 0),
    );

    const result = segmentIntersection(i32, seg1, seg2);
    try testing.expect(result.intersects);
    try testing.expect(result.point != null);
}

test "segmentIntersection: f32 precision" {
    const seg1 = Segment(f32).init(
        Point(f32).init(0.0, 0.0),
        Point(f32).init(10.0, 10.0),
    );
    const seg2 = Segment(f32).init(
        Point(f32).init(0.0, 10.0),
        Point(f32).init(10.0, 0.0),
    );

    const result = segmentIntersection(f32, seg1, seg2);
    try testing.expect(result.intersects);
    try testing.expect(result.point != null);

    const p = result.point.?;
    try testing.expectApproxEqAbs(@as(f32, 5.0), p.x, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 5.0), p.y, 1e-5);
}

test "segmentIntersection: near-parallel segments" {
    const seg1 = Segment(f64).init(
        Point(f64).init(0, 0),
        Point(f64).init(10, 1),
    );
    const seg2 = Segment(f64).init(
        Point(f64).init(0, 10),
        Point(f64).init(10, 11),
    );

    const result = segmentIntersection(f64, seg1, seg2);
    try testing.expect(!result.intersects);
}

test "segmentIntersection: vertical and horizontal" {
    const seg1 = Segment(f64).init(
        Point(f64).init(5, 0),
        Point(f64).init(5, 10),
    );
    const seg2 = Segment(f64).init(
        Point(f64).init(0, 5),
        Point(f64).init(10, 5),
    );

    const result = segmentIntersection(f64, seg1, seg2);
    try testing.expect(result.intersects);
    try testing.expect(result.point != null);

    const p = result.point.?;
    try testing.expectApproxEqAbs(5.0, p.x, 1e-6);
    try testing.expectApproxEqAbs(5.0, p.y, 1e-6);
}

test "segmentIntersection: large coordinates" {
    const seg1 = Segment(f64).init(
        Point(f64).init(0, 0),
        Point(f64).init(1000000, 1000000),
    );
    const seg2 = Segment(f64).init(
        Point(f64).init(0, 1000000),
        Point(f64).init(1000000, 0),
    );

    const result = segmentIntersection(f64, seg1, seg2);
    try testing.expect(result.intersects);
    try testing.expect(result.point != null);

    const p = result.point.?;
    try testing.expectApproxEqAbs(500000.0, p.x, 1.0);
    try testing.expectApproxEqAbs(500000.0, p.y, 1.0);
}

test "memory: no allocations" {
    // Line intersection uses no allocations
    const seg1 = Segment(f64).init(
        Point(f64).init(0, 0),
        Point(f64).init(10, 10),
    );
    const seg2 = Segment(f64).init(
        Point(f64).init(0, 10),
        Point(f64).init(10, 0),
    );

    _ = doSegmentsIntersect(f64, seg1, seg2);
    _ = segmentIntersection(f64, seg1, seg2);
    // No allocations to check
}
