const std = @import("std");
const testing = std.testing;

/// A 2D point with generic numeric type
pub fn Point(comptime T: type) type {
    return struct {
        x: T,
        y: T,

        const Self = @This();

        pub fn init(x: T, y: T) Self {
            return .{ .x = x, .y = y };
        }

        pub fn add(self: Self, other: Self) Self {
            return .{ .x = self.x + other.x, .y = self.y + other.y };
        }

        pub fn sub(self: Self, other: Self) Self {
            return .{ .x = self.x - other.x, .y = self.y - other.y };
        }

        pub fn dot(self: Self, other: Self) T {
            return self.x * other.x + self.y * other.y;
        }

        pub fn scale(self: Self, scalar: T) Self {
            return .{ .x = self.x * scalar, .y = self.y * scalar };
        }

        pub fn distanceSquared(self: Self, other: Self) T {
            const dx = self.x - other.x;
            const dy = self.y - other.y;
            return dx * dx + dy * dy;
        }
    };
}

/// Axis-Aligned Bounding Box (AABB)
pub fn AABB(comptime T: type) type {
    return struct {
        min: Point(T),
        max: Point(T),

        const Self = @This();

        pub fn init(min: Point(T), max: Point(T)) Self {
            return .{ .min = min, .max = max };
        }

        /// Compute width of the bounding box
        pub fn width(self: Self) T {
            return self.max.x - self.min.x;
        }

        /// Compute height of the bounding box
        pub fn height(self: Self) T {
            return self.max.y - self.min.y;
        }

        /// Compute area of the bounding box
        pub fn area(self: Self) T {
            return self.width() * self.height();
        }

        /// Compute center point of the bounding box
        pub fn center(self: Self) Point(T) {
            return Point(T).init(
                (self.min.x + self.max.x) / 2,
                (self.min.y + self.max.y) / 2,
            );
        }

        /// Check if a point is inside the bounding box (inclusive)
        pub fn contains(self: Self, point: Point(T)) bool {
            return point.x >= self.min.x and point.x <= self.max.x and
                point.y >= self.min.y and point.y <= self.max.y;
        }

        /// Check if this AABB intersects with another AABB
        pub fn intersects(self: Self, other: Self) bool {
            return !(self.max.x < other.min.x or self.min.x > other.max.x or
                self.max.y < other.min.y or self.min.y > other.max.y);
        }

        /// Compute the intersection of two AABBs
        /// Returns null if they don't intersect
        pub fn intersection(self: Self, other: Self) ?Self {
            if (!self.intersects(other)) return null;
            return Self.init(
                Point(T).init(
                    @max(self.min.x, other.min.x),
                    @max(self.min.y, other.min.y),
                ),
                Point(T).init(
                    @min(self.max.x, other.max.x),
                    @min(self.max.y, other.max.y),
                ),
            );
        }

        /// Compute the union (smallest AABB containing both)
        pub fn unionWith(self: Self, other: Self) Self {
            return Self.init(
                Point(T).init(
                    @min(self.min.x, other.min.x),
                    @min(self.min.y, other.min.y),
                ),
                Point(T).init(
                    @max(self.max.x, other.max.x),
                    @max(self.max.y, other.max.y),
                ),
            );
        }
    };
}

/// Oriented Bounding Box (OBB)
pub fn OBB(comptime T: type) type {
    return struct {
        center: Point(T),
        half_extents: Point(T), // half-width and half-height
        rotation: T, // rotation angle in radians

        const Self = @This();

        pub fn init(center: Point(T), half_extents: Point(T), rotation: T) Self {
            return .{
                .center = center,
                .half_extents = half_extents,
                .rotation = rotation,
            };
        }

        /// Compute area of the bounding box
        pub fn area(self: Self) T {
            return (2 * self.half_extents.x) * (2 * self.half_extents.y);
        }

        /// Get the four corner vertices of the OBB
        /// Returns corners in counter-clockwise order starting from bottom-left
        pub fn getCorners(self: Self) [4]Point(T) {
            const cos_r = @cos(self.rotation);
            const sin_r = @sin(self.rotation);

            // Local corners (before rotation)
            const local_corners = [4]Point(T){
                Point(T).init(-self.half_extents.x, -self.half_extents.y),
                Point(T).init(self.half_extents.x, -self.half_extents.y),
                Point(T).init(self.half_extents.x, self.half_extents.y),
                Point(T).init(-self.half_extents.x, self.half_extents.y),
            };

            var corners: [4]Point(T) = undefined;
            for (local_corners, 0..) |local, i| {
                // Rotate and translate
                const rotated_x = local.x * cos_r - local.y * sin_r;
                const rotated_y = local.x * sin_r + local.y * cos_r;
                corners[i] = Point(T).init(
                    self.center.x + rotated_x,
                    self.center.y + rotated_y,
                );
            }
            return corners;
        }

        /// Convert OBB to AABB (axis-aligned bounding box containing the OBB)
        pub fn toAABB(self: Self) AABB(T) {
            const corners = self.getCorners();
            var min_x = corners[0].x;
            var max_x = corners[0].x;
            var min_y = corners[0].y;
            var max_y = corners[0].y;

            for (corners[1..]) |corner| {
                min_x = @min(min_x, corner.x);
                max_x = @max(max_x, corner.x);
                min_y = @min(min_y, corner.y);
                max_y = @max(max_y, corner.y);
            }

            return AABB(T).init(
                Point(T).init(min_x, min_y),
                Point(T).init(max_x, max_y),
            );
        }
    };
}

/// Compute axis-aligned bounding box for a set of points
/// Time: O(n) where n is the number of points
/// Space: O(1)
pub fn computeAABB(comptime T: type, points: []const Point(T)) !AABB(T) {
    if (points.len == 0) return error.EmptyInput;

    var min_x = points[0].x;
    var max_x = points[0].x;
    var min_y = points[0].y;
    var max_y = points[0].y;

    for (points[1..]) |point| {
        min_x = @min(min_x, point.x);
        max_x = @max(max_x, point.x);
        min_y = @min(min_y, point.y);
        max_y = @max(max_y, point.y);
    }

    return AABB(T).init(
        Point(T).init(min_x, min_y),
        Point(T).init(max_x, max_y),
    );
}

/// Compute minimum area oriented bounding box using rotating calipers
/// Time: O(n) where n is the number of convex hull vertices
/// Space: O(1)
///
/// Note: This assumes the input points form a convex hull in counter-clockwise order.
/// For arbitrary points, compute the convex hull first using Graham scan or similar.
pub fn computeMinimumOBB(comptime T: type, convex_hull: []const Point(T)) !OBB(T) {
    if (convex_hull.len < 3) return error.InsufficientPoints;

    const n = convex_hull.len;
    var min_area: T = std.math.inf(T);
    var best_obb: OBB(T) = undefined;

    // For each edge of the convex hull
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const j = (i + 1) % n;
        const edge = convex_hull[j].sub(convex_hull[i]);

        // Compute rotation angle for this edge (to align with x-axis)
        const angle = std.math.atan2(edge.y, edge.x);
        const cos_a = @cos(angle);
        const sin_a = @sin(angle);

        // Rotate all points and find AABB in rotated space
        var min_x: T = std.math.inf(T);
        var max_x: T = -std.math.inf(T);
        var min_y: T = std.math.inf(T);
        var max_y: T = -std.math.inf(T);

        for (convex_hull) |point| {
            const rotated_x = point.x * cos_a + point.y * sin_a;
            const rotated_y = -point.x * sin_a + point.y * cos_a;
            min_x = @min(min_x, rotated_x);
            max_x = @max(max_x, rotated_x);
            min_y = @min(min_y, rotated_y);
            max_y = @max(max_y, rotated_y);
        }

        const width = max_x - min_x;
        const height = max_y - min_y;
        const area = width * height;

        if (area < min_area) {
            min_area = area;

            // Compute center in rotated space
            const center_x_rot = (min_x + max_x) / 2;
            const center_y_rot = (min_y + max_y) / 2;

            // Rotate back to original space
            const center_x = center_x_rot * cos_a - center_y_rot * sin_a;
            const center_y = center_x_rot * sin_a + center_y_rot * cos_a;

            best_obb = OBB(T).init(
                Point(T).init(center_x, center_y),
                Point(T).init(width / 2, height / 2),
                angle,
            );
        }
    }

    return best_obb;
}

// ============================================================================
// Tests
// ============================================================================

test "AABB - basic properties" {
    const aabb = AABB(f64).init(
        Point(f64).init(0, 0),
        Point(f64).init(10, 5),
    );

    try testing.expectEqual(@as(f64, 10), aabb.width());
    try testing.expectEqual(@as(f64, 5), aabb.height());
    try testing.expectEqual(@as(f64, 50), aabb.area());

    const c = aabb.center();
    try testing.expectEqual(@as(f64, 5), c.x);
    try testing.expectEqual(@as(f64, 2.5), c.y);
}

test "AABB - contains point" {
    const aabb = AABB(f64).init(
        Point(f64).init(0, 0),
        Point(f64).init(10, 10),
    );

    try testing.expect(aabb.contains(Point(f64).init(5, 5)));
    try testing.expect(aabb.contains(Point(f64).init(0, 0)));
    try testing.expect(aabb.contains(Point(f64).init(10, 10)));
    try testing.expect(!aabb.contains(Point(f64).init(-1, 5)));
    try testing.expect(!aabb.contains(Point(f64).init(11, 5)));
}

test "AABB - intersection" {
    const aabb1 = AABB(f64).init(
        Point(f64).init(0, 0),
        Point(f64).init(10, 10),
    );
    const aabb2 = AABB(f64).init(
        Point(f64).init(5, 5),
        Point(f64).init(15, 15),
    );

    try testing.expect(aabb1.intersects(aabb2));

    const inter = aabb1.intersection(aabb2).?;
    try testing.expectEqual(@as(f64, 5), inter.min.x);
    try testing.expectEqual(@as(f64, 5), inter.min.y);
    try testing.expectEqual(@as(f64, 10), inter.max.x);
    try testing.expectEqual(@as(f64, 10), inter.max.y);
}

test "AABB - no intersection" {
    const aabb1 = AABB(f64).init(
        Point(f64).init(0, 0),
        Point(f64).init(5, 5),
    );
    const aabb2 = AABB(f64).init(
        Point(f64).init(10, 10),
        Point(f64).init(15, 15),
    );

    try testing.expect(!aabb1.intersects(aabb2));
    try testing.expect(aabb1.intersection(aabb2) == null);
}

test "AABB - union" {
    const aabb1 = AABB(f64).init(
        Point(f64).init(0, 0),
        Point(f64).init(5, 5),
    );
    const aabb2 = AABB(f64).init(
        Point(f64).init(3, 3),
        Point(f64).init(10, 8),
    );

    const union_box = aabb1.unionWith(aabb2);
    try testing.expectEqual(@as(f64, 0), union_box.min.x);
    try testing.expectEqual(@as(f64, 0), union_box.min.y);
    try testing.expectEqual(@as(f64, 10), union_box.max.x);
    try testing.expectEqual(@as(f64, 8), union_box.max.y);
}

test "computeAABB - basic" {
    const points = [_]Point(f64){
        Point(f64).init(1, 2),
        Point(f64).init(5, 8),
        Point(f64).init(3, 4),
        Point(f64).init(7, 1),
    };

    const aabb = try computeAABB(f64, &points);
    try testing.expectEqual(@as(f64, 1), aabb.min.x);
    try testing.expectEqual(@as(f64, 1), aabb.min.y);
    try testing.expectEqual(@as(f64, 7), aabb.max.x);
    try testing.expectEqual(@as(f64, 8), aabb.max.y);
}

test "computeAABB - empty input" {
    const points = [_]Point(f64){};
    try testing.expectError(error.EmptyInput, computeAABB(f64, &points));
}

test "computeAABB - single point" {
    const points = [_]Point(f64){Point(f64).init(3, 4)};
    const aabb = try computeAABB(f64, &points);
    try testing.expectEqual(@as(f64, 3), aabb.min.x);
    try testing.expectEqual(@as(f64, 4), aabb.min.y);
    try testing.expectEqual(@as(f64, 3), aabb.max.x);
    try testing.expectEqual(@as(f64, 4), aabb.max.y);
}

test "OBB - axis-aligned (rotation = 0)" {
    const obb = OBB(f64).init(
        Point(f64).init(5, 5),
        Point(f64).init(3, 2),
        0,
    );

    try testing.expectEqual(@as(f64, 24), obb.area());

    const corners = obb.getCorners();
    try testing.expectApproxEqAbs(@as(f64, 2), corners[0].x, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 3), corners[0].y, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 8), corners[1].x, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 3), corners[1].y, 1e-10);
}

test "OBB - rotated 45 degrees" {
    const obb = OBB(f64).init(
        Point(f64).init(0, 0),
        Point(f64).init(1, 1),
        std.math.pi / 4.0, // 45 degrees
    );

    try testing.expectEqual(@as(f64, 4), obb.area());

    const corners = obb.getCorners();
    // When rotated 45 degrees, a 2x2 square has corners at roughly (0, -√2), (√2, 0), (0, √2), (-√2, 0)
    const sqrt2 = @sqrt(2.0);
    try testing.expectApproxEqAbs(@as(f64, 0), corners[0].x, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, -sqrt2), corners[0].y, 1e-10);
}

test "OBB - toAABB" {
    const obb = OBB(f64).init(
        Point(f64).init(0, 0),
        Point(f64).init(2, 1),
        std.math.pi / 4.0,
    );

    const aabb = obb.toAABB();

    // For a 4x2 box rotated 45 degrees, the AABB should be larger
    // Center should still be at origin
    try testing.expectApproxEqAbs(@as(f64, 0), aabb.center().x, 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 0), aabb.center().y, 1e-10);
    // Width and height should be larger than the OBB half-extents
    try testing.expect(aabb.width() > 4);
    try testing.expect(aabb.height() > 2);
}

test "computeMinimumOBB - axis-aligned square" {
    // Square aligned with axes
    const points = [_]Point(f64){
        Point(f64).init(0, 0),
        Point(f64).init(4, 0),
        Point(f64).init(4, 4),
        Point(f64).init(0, 4),
    };

    const obb = try computeMinimumOBB(f64, &points);
    try testing.expectEqual(@as(f64, 16), obb.area());
    try testing.expectApproxEqAbs(@as(f64, 2), obb.center.x, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 2), obb.center.y, 1e-6);
}

test "computeMinimumOBB - rotated rectangle" {
    // Rectangle rotated 45 degrees (diamond shape when axis-aligned)
    // Vertices at (1,0), (0,1), (-1,0), (0,-1) form a square of area 2
    const points = [_]Point(f64){
        Point(f64).init(1, 0),
        Point(f64).init(0, 1),
        Point(f64).init(-1, 0),
        Point(f64).init(0, -1),
    };

    const obb = try computeMinimumOBB(f64, &points);
    // The minimum OBB should have area = 2 (sqrt(2) × sqrt(2))
    try testing.expectApproxEqAbs(@as(f64, 2), obb.area(), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0), obb.center.x, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0), obb.center.y, 1e-6);
}

test "computeMinimumOBB - triangle" {
    const points = [_]Point(f64){
        Point(f64).init(0, 0),
        Point(f64).init(4, 0),
        Point(f64).init(2, 3),
    };

    const obb = try computeMinimumOBB(f64, &points);
    // Should find the minimum bounding box
    try testing.expect(obb.area() > 0);
    try testing.expect(obb.area() < 20); // Less than the axis-aligned box would be
}

test "computeMinimumOBB - insufficient points" {
    const points = [_]Point(f64){
        Point(f64).init(0, 0),
        Point(f64).init(1, 1),
    };
    try testing.expectError(error.InsufficientPoints, computeMinimumOBB(f64, &points));
}

test "AABB - integer type" {
    const aabb = AABB(i32).init(
        Point(i32).init(0, 0),
        Point(i32).init(10, 10),
    );

    try testing.expectEqual(@as(i32, 10), aabb.width());
    try testing.expectEqual(@as(i32, 100), aabb.area());
    try testing.expect(aabb.contains(Point(i32).init(5, 5)));
}

test "computeAABB - large dataset" {
    var points: [100]Point(f64) = undefined;
    for (&points, 0..) |*p, i| {
        const fi: f64 = @floatFromInt(i);
        p.* = Point(f64).init(fi, fi * 2);
    }

    const aabb = try computeAABB(f64, &points);
    try testing.expectEqual(@as(f64, 0), aabb.min.x);
    try testing.expectEqual(@as(f64, 0), aabb.min.y);
    try testing.expectEqual(@as(f64, 99), aabb.max.x);
    try testing.expectEqual(@as(f64, 198), aabb.max.y);
}

test "AABB - f32 type support" {
    const aabb = AABB(f32).init(
        Point(f32).init(0, 0),
        Point(f32).init(5, 5),
    );

    try testing.expectEqual(@as(f32, 25), aabb.area());
    try testing.expect(aabb.contains(Point(f32).init(2.5, 2.5)));
}
