const std = @import("std");
const testing = std.testing;

/// A 2D point with x and y coordinates
pub fn Point(comptime T: type) type {
    return struct {
        x: T,
        y: T,

        const Self = @This();

        pub fn init(x: T, y: T) Self {
            return .{ .x = x, .y = y };
        }

        /// Compare points lexicographically (by x first, then by y)
        pub fn lessThan(a: Self, b: Self) bool {
            return a.x < b.x or (a.x == b.x and a.y < b.y);
        }

        /// Compute the cross product of vectors (b - a) and (c - a)
        /// Positive: counter-clockwise turn, Negative: clockwise turn, Zero: collinear
        pub fn crossProduct(a: Self, b: Self, c: Self) T {
            const dx1 = b.x - a.x;
            const dy1 = b.y - a.y;
            const dx2 = c.x - a.x;
            const dy2 = c.y - a.y;
            return dx1 * dy2 - dy1 * dx2;
        }
    };
}

/// Graham scan algorithm for computing the convex hull.
/// Returns the convex hull as a list of points in counter-clockwise order.
/// Time: O(n log n) where n is the number of points
/// Space: O(n) for the result
pub fn grahamScan(comptime T: type, allocator: std.mem.Allocator, points: []const Point(T)) ![]Point(T) {
    if (points.len < 3) {
        // Convex hull of < 3 points is just the points themselves
        const result = try allocator.alloc(Point(T), points.len);
        @memcpy(result, points);
        return result;
    }

    // Create a mutable copy and sort lexicographically
    const sorted = try allocator.alloc(Point(T), points.len);
    defer allocator.free(sorted);
    @memcpy(sorted, points);

    std.mem.sort(Point(T), sorted, {}, struct {
        fn lessThan(_: void, a: Point(T), b: Point(T)) bool {
            return Point(T).lessThan(a, b);
        }
    }.lessThan);

    // Build lower hull
    var lower: std.ArrayList(Point(T)) = .{};
    defer lower.deinit(allocator);

    for (sorted) |p| {
        while (lower.items.len >= 2) {
            const len = lower.items.len;
            const cross = Point(T).crossProduct(lower.items[len - 2], lower.items[len - 1], p);
            if (cross <= 0) {
                _ = lower.pop();
            } else {
                break;
            }
        }
        try lower.append(allocator, p);
    }

    // Build upper hull
    var upper: std.ArrayList(Point(T)) = .{};
    defer upper.deinit(allocator);

    var i: usize = sorted.len;
    while (i > 0) {
        i -= 1;
        const p = sorted[i];
        while (upper.items.len >= 2) {
            const len = upper.items.len;
            const cross = Point(T).crossProduct(upper.items[len - 2], upper.items[len - 1], p);
            if (cross <= 0) {
                _ = upper.pop();
            } else {
                break;
            }
        }
        try upper.append(allocator, p);
    }

    // Remove last point of each half because it's repeated at the beginning of the other half
    if (lower.items.len > 0) _ = lower.pop();
    if (upper.items.len > 0) _ = upper.pop();

    // Concatenate lower and upper hull
    const result = try allocator.alloc(Point(T), lower.items.len + upper.items.len);
    @memcpy(result[0..lower.items.len], lower.items);
    @memcpy(result[lower.items.len..], upper.items);

    return result;
}

/// Jarvis march (gift wrapping) algorithm for computing the convex hull.
/// More efficient than Graham scan when the hull has few points.
/// Time: O(n * h) where n is the number of points and h is the hull size
/// Space: O(h) for the result
pub fn jarvisMarch(comptime T: type, allocator: std.mem.Allocator, points: []const Point(T)) ![]Point(T) {
    if (points.len < 3) {
        const result = try allocator.alloc(Point(T), points.len);
        @memcpy(result, points);
        return result;
    }

    var hull: std.ArrayList(Point(T)) = .{};
    errdefer hull.deinit(allocator);

    // Start with leftmost point
    var leftmost: usize = 0;
    for (points, 0..) |p, i| {
        if (p.x < points[leftmost].x or (p.x == points[leftmost].x and p.y < points[leftmost].y)) {
            leftmost = i;
        }
    }

    var current = leftmost;
    while (true) {
        try hull.append(allocator, points[current]);

        // Find next point (most counter-clockwise from current)
        var next: usize = 0;
        for (points, 0..) |_, i| {
            if (i == current) continue;

            if (next == current) {
                next = i;
                continue;
            }

            const cross = Point(T).crossProduct(points[current], points[next], points[i]);
            if (cross > 0) {
                next = i;
            }
        }

        current = next;
        if (current == leftmost) break;
    }

    return hull.toOwnedSlice(allocator);
}

// ============================================================================
// Tests
// ============================================================================

test "Point - crossProduct" {
    const P = Point(i32);
    const a = P.init(0, 0);
    const b = P.init(1, 0);
    const c = P.init(1, 1);

    // Counter-clockwise turn
    try testing.expect(P.crossProduct(a, b, c) > 0);

    // Clockwise turn
    const d = P.init(1, -1);
    try testing.expect(P.crossProduct(a, b, d) < 0);

    // Collinear
    const e = P.init(2, 0);
    try testing.expectEqual(@as(i32, 0), P.crossProduct(a, b, e));
}

test "grahamScan - simple square" {
    const P = Point(i32);
    const points = [_]P{
        P.init(0, 0),
        P.init(1, 0),
        P.init(1, 1),
        P.init(0, 1),
    };

    const hull = try grahamScan(i32, testing.allocator, &points);
    defer testing.allocator.free(hull);

    try testing.expectEqual(@as(usize, 4), hull.len);
}

test "grahamScan - with interior points" {
    const P = Point(i32);
    const points = [_]P{
        P.init(0, 0),
        P.init(2, 0),
        P.init(2, 2),
        P.init(0, 2),
        P.init(1, 1), // interior point
    };

    const hull = try grahamScan(i32, testing.allocator, &points);
    defer testing.allocator.free(hull);

    // Hull should be the outer 4 points
    try testing.expectEqual(@as(usize, 4), hull.len);
}

test "grahamScan - triangle" {
    const P = Point(i32);
    const points = [_]P{
        P.init(0, 0),
        P.init(1, 0),
        P.init(0, 1),
    };

    const hull = try grahamScan(i32, testing.allocator, &points);
    defer testing.allocator.free(hull);

    try testing.expectEqual(@as(usize, 3), hull.len);
}

test "grahamScan - collinear points" {
    const P = Point(i32);
    const points = [_]P{
        P.init(0, 0),
        P.init(1, 0),
        P.init(2, 0),
        P.init(3, 0),
    };

    const hull = try grahamScan(i32, testing.allocator, &points);
    defer testing.allocator.free(hull);

    // Convex hull of collinear points is the two endpoints
    try testing.expectEqual(@as(usize, 2), hull.len);
}

test "jarvisMarch - simple square" {
    const P = Point(i32);
    const points = [_]P{
        P.init(0, 0),
        P.init(1, 0),
        P.init(1, 1),
        P.init(0, 1),
    };

    const hull = try jarvisMarch(i32, testing.allocator, &points);
    defer testing.allocator.free(hull);

    try testing.expectEqual(@as(usize, 4), hull.len);
}

test "jarvisMarch - with interior points" {
    const P = Point(i32);
    const points = [_]P{
        P.init(0, 0),
        P.init(2, 0),
        P.init(2, 2),
        P.init(0, 2),
        P.init(1, 1), // interior point
    };

    const hull = try jarvisMarch(i32, testing.allocator, &points);
    defer testing.allocator.free(hull);

    try testing.expectEqual(@as(usize, 4), hull.len);
}

test "jarvisMarch - triangle" {
    const P = Point(i32);
    const points = [_]P{
        P.init(0, 0),
        P.init(1, 0),
        P.init(0, 1),
    };

    const hull = try jarvisMarch(i32, testing.allocator, &points);
    defer testing.allocator.free(hull);

    try testing.expectEqual(@as(usize, 3), hull.len);
}

test "grahamScan - random points" {
    const P = Point(i32);
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var points: [100]P = undefined;
    for (&points) |*p| {
        p.* = P.init(random.intRangeAtMost(i32, -100, 100), random.intRangeAtMost(i32, -100, 100));
    }

    const hull = try grahamScan(i32, testing.allocator, &points);
    defer testing.allocator.free(hull);

    // Hull should have at least 3 points (unless all collinear)
    try testing.expect(hull.len >= 2);
    try testing.expect(hull.len <= points.len);
}

test "jarvisMarch - random points" {
    const P = Point(i32);
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var points: [100]P = undefined;
    for (&points) |*p| {
        p.* = P.init(random.intRangeAtMost(i32, -100, 100), random.intRangeAtMost(i32, -100, 100));
    }

    const hull = try jarvisMarch(i32, testing.allocator, &points);
    defer testing.allocator.free(hull);

    try testing.expect(hull.len >= 2);
    try testing.expect(hull.len <= points.len);
}

test "grahamScan vs jarvisMarch - consistency" {
    const P = Point(i32);
    const points = [_]P{
        P.init(0, 0),
        P.init(2, 0),
        P.init(2, 2),
        P.init(0, 2),
        P.init(1, 1),
        P.init(1, 0),
        P.init(0, 1),
    };

    const hull1 = try grahamScan(i32, testing.allocator, &points);
    defer testing.allocator.free(hull1);

    const hull2 = try jarvisMarch(i32, testing.allocator, &points);
    defer testing.allocator.free(hull2);

    // Both should produce the same number of hull points
    try testing.expectEqual(hull1.len, hull2.len);
}
