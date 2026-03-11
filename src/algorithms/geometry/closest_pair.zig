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

        /// Compute squared Euclidean distance between two points
        pub fn distanceSquared(a: Self, b: Self) T {
            const dx = a.x - b.x;
            const dy = a.y - b.y;
            return dx * dx + dy * dy;
        }

        /// Compare points by x-coordinate
        pub fn compareX(_: void, a: Self, b: Self) bool {
            return a.x < b.x or (a.x == b.x and a.y < b.y);
        }

        /// Compare points by y-coordinate
        pub fn compareY(_: void, a: Self, b: Self) bool {
            return a.y < b.y or (a.y == b.y and a.x < b.x);
        }
    };
}

/// Result of closest pair search
pub fn ClosestPairResult(comptime T: type) type {
    return struct {
        p1: Point(T),
        p2: Point(T),
        distance_squared: T,
    };
}

/// Brute force closest pair algorithm.
/// Time: O(n²)
/// Space: O(1)
pub fn bruteForce(comptime T: type, points: []const Point(T)) !ClosestPairResult(T) {
    if (points.len < 2) return error.InsufficientPoints;

    var min_dist_sq = Point(T).distanceSquared(points[0], points[1]);
    var result = ClosestPairResult(T){
        .p1 = points[0],
        .p2 = points[1],
        .distance_squared = min_dist_sq,
    };

    for (points, 0..) |p1, i| {
        for (points[i + 1 ..]) |p2| {
            const dist_sq = Point(T).distanceSquared(p1, p2);
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                result = .{
                    .p1 = p1,
                    .p2 = p2,
                    .distance_squared = dist_sq,
                };
            }
        }
    }

    return result;
}

/// Divide-and-conquer closest pair algorithm.
/// Time: O(n log² n) or O(n log n) with optimizations
/// Space: O(n) for the sorted arrays
pub fn closestPair(comptime T: type, allocator: std.mem.Allocator, points: []const Point(T)) !ClosestPairResult(T) {
    if (points.len < 2) return error.InsufficientPoints;
    if (points.len <= 3) return try bruteForce(T, points);

    // Create sorted copies
    const px = try allocator.alloc(Point(T), points.len);
    defer allocator.free(px);
    @memcpy(px, points);
    std.mem.sort(Point(T), px, {}, Point(T).compareX);

    const py = try allocator.alloc(Point(T), points.len);
    defer allocator.free(py);
    @memcpy(py, points);
    std.mem.sort(Point(T), py, {}, Point(T).compareY);

    return closestPairRecursive(T, allocator, px, py);
}

fn closestPairRecursive(comptime T: type, allocator: std.mem.Allocator, px: []const Point(T), py: []const Point(T)) !ClosestPairResult(T) {
    const n = px.len;

    // Base case: use brute force for small inputs
    if (n <= 3) {
        return try bruteForce(T, px);
    }

    // Divide
    const mid = n / 2;
    const mid_point = px[mid];

    // Split py into left and right halves
    const pyl = try allocator.alloc(Point(T), mid);
    defer allocator.free(pyl);
    const pyr = try allocator.alloc(Point(T), n - mid);
    defer allocator.free(pyr);

    var li: usize = 0;
    var ri: usize = 0;
    for (py) |p| {
        if (p.x < mid_point.x or (p.x == mid_point.x and p.y < mid_point.y)) {
            if (li < pyl.len) {
                pyl[li] = p;
                li += 1;
            }
        } else {
            if (ri < pyr.len) {
                pyr[ri] = p;
                ri += 1;
            }
        }
    }

    // Conquer
    const left_result = try closestPairRecursive(T, allocator, px[0..mid], pyl[0..li]);
    const right_result = try closestPairRecursive(T, allocator, px[mid..], pyr[0..ri]);

    // Find minimum of left and right results
    var min_result = if (left_result.distance_squared < right_result.distance_squared)
        left_result
    else
        right_result;

    // Build strip of points close to dividing line
    const strip = try allocator.alloc(Point(T), n);
    defer allocator.free(strip);
    var strip_size: usize = 0;

    for (py) |p| {
        const dx = p.x - mid_point.x;
        if (dx * dx < min_result.distance_squared) {
            strip[strip_size] = p;
            strip_size += 1;
        }
    }

    // Find closest points in strip
    const strip_result = try closestInStrip(T, strip[0..strip_size], min_result.distance_squared);
    if (strip_result) |sr| {
        if (sr.distance_squared < min_result.distance_squared) {
            min_result = sr;
        }
    }

    return min_result;
}

fn closestInStrip(comptime T: type, strip: []const Point(T), min_dist_sq: T) !?ClosestPairResult(T) {
    if (strip.len < 2) return null;

    var result: ?ClosestPairResult(T) = null;
    var current_min = min_dist_sq;

    for (strip, 0..) |p1, i| {
        var j = i + 1;
        while (j < strip.len) : (j += 1) {
            const p2 = strip[j];
            const dy = p2.y - p1.y;

            // Early termination: points are sorted by y, so if dy² >= current_min, no need to check further
            if (dy * dy >= current_min) break;

            const dist_sq = Point(T).distanceSquared(p1, p2);
            if (dist_sq < current_min) {
                current_min = dist_sq;
                result = .{
                    .p1 = p1,
                    .p2 = p2,
                    .distance_squared = dist_sq,
                };
            }
        }
    }

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "Point - distanceSquared" {
    const P = Point(i32);
    const p1 = P.init(0, 0);
    const p2 = P.init(3, 4);

    // 3² + 4² = 9 + 16 = 25
    try testing.expectEqual(@as(i32, 25), P.distanceSquared(p1, p2));
}

test "bruteForce - simple case" {
    const P = Point(i32);
    const points = [_]P{
        P.init(0, 0),
        P.init(1, 0),
        P.init(0, 1),
        P.init(10, 10),
    };

    const result = try bruteForce(i32, &points);

    // Closest pair should be (0,0) and (1,0) or (0,0) and (0,1), both distance 1
    try testing.expectEqual(@as(i32, 1), result.distance_squared);
}

test "bruteForce - single pair" {
    const P = Point(i32);
    const points = [_]P{
        P.init(0, 0),
        P.init(5, 5),
    };

    const result = try bruteForce(i32, &points);

    // 5² + 5² = 50
    try testing.expectEqual(@as(i32, 50), result.distance_squared);
}

test "bruteForce - insufficient points" {
    const P = Point(i32);
    const points = [_]P{P.init(0, 0)};

    try testing.expectError(error.InsufficientPoints, bruteForce(i32, &points));
}

test "closestPair - simple case" {
    const P = Point(i32);
    const points = [_]P{
        P.init(0, 0),
        P.init(1, 0),
        P.init(0, 1),
        P.init(10, 10),
    };

    const result = try closestPair(i32, testing.allocator, &points);

    try testing.expectEqual(@as(i32, 1), result.distance_squared);
}

test "closestPair - larger set" {
    const P = Point(i32);
    const points = [_]P{
        P.init(2, 3),
        P.init(12, 30),
        P.init(40, 50),
        P.init(5, 1),
        P.init(12, 10),
        P.init(3, 4),
    };

    const result = try closestPair(i32, testing.allocator, &points);

    // Closest pair is (2,3) and (3,4), distance² = 1 + 1 = 2
    try testing.expectEqual(@as(i32, 2), result.distance_squared);
}

test "closestPair - random points" {
    const P = Point(i32);
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var points: [50]P = undefined;
    for (&points) |*p| {
        p.* = P.init(random.intRangeAtMost(i32, -100, 100), random.intRangeAtMost(i32, -100, 100));
    }

    const result = try closestPair(i32, testing.allocator, &points);

    // Verify against brute force
    const brute_result = try bruteForce(i32, &points);
    try testing.expectEqual(brute_result.distance_squared, result.distance_squared);
}

test "closestPair vs bruteForce - consistency" {
    const P = Point(i32);
    const points = [_]P{
        P.init(0, 0),
        P.init(1, 0),
        P.init(0, 1),
        P.init(10, 10),
        P.init(5, 5),
        P.init(3, 3),
        P.init(7, 2),
    };

    const result1 = try closestPair(i32, testing.allocator, &points);
    const result2 = try bruteForce(i32, &points);

    try testing.expectEqual(result2.distance_squared, result1.distance_squared);
}

test "closestPair - vertical strip case" {
    const P = Point(i32);
    // Points designed to test the strip logic
    const points = [_]P{
        P.init(0, 0),
        P.init(0, 10),
        P.init(1, 5),
        P.init(2, 5),
        P.init(10, 0),
        P.init(10, 10),
    };

    const result = try closestPair(i32, testing.allocator, &points);

    // Closest pair is (1,5) and (2,5), distance² = 1
    try testing.expectEqual(@as(i32, 1), result.distance_squared);
}

test "closestPair - all same x-coordinate" {
    const P = Point(i32);
    const points = [_]P{
        P.init(5, 0),
        P.init(5, 1),
        P.init(5, 10),
        P.init(5, 20),
    };

    const result = try closestPair(i32, testing.allocator, &points);

    // Closest pair is (5,0) and (5,1), distance² = 1
    try testing.expectEqual(@as(i32, 1), result.distance_squared);
}

test "closestPair - all same y-coordinate" {
    const P = Point(i32);
    const points = [_]P{
        P.init(0, 5),
        P.init(1, 5),
        P.init(10, 5),
        P.init(20, 5),
    };

    const result = try closestPair(i32, testing.allocator, &points);

    // Closest pair is (0,5) and (1,5), distance² = 1
    try testing.expectEqual(@as(i32, 1), result.distance_squared);
}
