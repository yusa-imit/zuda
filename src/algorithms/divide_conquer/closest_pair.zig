/// Closest Pair of Points using divide-and-conquer.
///
/// Find the closest pair of points in 2D space using an efficient
/// divide-and-conquer algorithm.
///
/// ## Algorithm
///
/// 1. Sort points by x-coordinate
/// 2. Divide points into left and right halves
/// 3. Recursively find closest pairs in each half
/// 4. Find closest pair with one point in each half (crossing the dividing line)
/// 5. Return overall minimum
///
/// ## Complexity
///
/// - **Time**: O(n log n) with sorting, O(n log²n) without presorted input
/// - **Space**: O(n) for auxiliary arrays
///
/// ## Use Cases
///
/// - Collision detection in graphics
/// - Nearest neighbor search
/// - Computational geometry
/// - Clustering algorithms
///
const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// A 2D point.
pub const Point = struct {
    x: f64,
    y: f64,

    pub fn distanceTo(self: Point, other: Point) f64 {
        const dx = self.x - other.x;
        const dy = self.y - other.y;
        return @sqrt(dx * dx + dy * dy);
    }
};

/// Result of closest pair computation.
pub const ClosestPairResult = struct {
    p1: Point,
    p2: Point,
    distance: f64,
};

/// Find the closest pair of points using divide-and-conquer.
///
/// Time: O(n log n)
/// Space: O(n)
///
/// ## Example
///
/// ```zig
/// const points = [_]Point{
///     .{ .x = 0, .y = 0 },
///     .{ .x = 1, .y = 1 },
///     .{ .x = 5, .y = 5 },
/// };
/// const result = try closestPair(allocator, &points);
/// // result.distance ≈ 1.414 (between (0,0) and (1,1))
/// ```
pub fn closestPair(allocator: Allocator, points: []const Point) error{ OutOfMemory, TooFewPoints }!ClosestPairResult {
    if (points.len < 2) return error.TooFewPoints;

    // Copy and sort by x-coordinate
    const px = try allocator.alloc(Point, points.len);
    defer allocator.free(px);
    @memcpy(px, points);
    std.mem.sort(Point, px, {}, compareByX);

    // Sort by y-coordinate for strip operations
    const py = try allocator.alloc(Point, points.len);
    defer allocator.free(py);
    @memcpy(py, points);
    std.mem.sort(Point, py, {}, compareByY);

    return closestPairRec(allocator, px, py);
}

fn closestPairRec(allocator: Allocator, px: []const Point, py: []const Point) error{OutOfMemory}!ClosestPairResult {
    const n = px.len;

    // Base case: brute force for small n
    if (n <= 3) {
        return bruteForce(px);
    }

    const mid = n / 2;
    const mid_point = px[mid];

    // Divide py into left and right based on mid_point.x
    var py_left = try allocator.alloc(Point, py.len);
    defer allocator.free(py_left);
    var py_right = try allocator.alloc(Point, py.len);
    defer allocator.free(py_right);

    var left_count: usize = 0;
    var right_count: usize = 0;
    for (py) |p| {
        if (p.x <= mid_point.x) {
            py_left[left_count] = p;
            left_count += 1;
        } else {
            py_right[right_count] = p;
            right_count += 1;
        }
    }

    // Recursively find closest in each half
    const left_result = try closestPairRec(allocator, px[0..mid], py_left[0..left_count]);
    const right_result = try closestPairRec(allocator, px[mid..], py_right[0..right_count]);

    // Find the minimum of the two
    var min_result = if (left_result.distance < right_result.distance) left_result else right_result;

    // Find closest pair with one point in each half
    const strip_result = try closestInStrip(allocator, py, mid_point.x, min_result.distance);
    if (strip_result) |strip| {
        if (strip.distance < min_result.distance) {
            min_result = strip;
        }
    }

    return min_result;
}

fn bruteForce(points: []const Point) ClosestPairResult {
    var min_dist = std.math.floatMax(f64);
    var result = ClosestPairResult{
        .p1 = points[0],
        .p2 = points[1],
        .distance = min_dist,
    };

    for (0..points.len) |i| {
        for (i + 1..points.len) |j| {
            const dist = points[i].distanceTo(points[j]);
            if (dist < min_dist) {
                min_dist = dist;
                result = ClosestPairResult{
                    .p1 = points[i],
                    .p2 = points[j],
                    .distance = dist,
                };
            }
        }
    }

    return result;
}

fn closestInStrip(
    allocator: Allocator,
    py: []const Point,
    mid_x: f64,
    delta: f64,
) error{OutOfMemory}!?ClosestPairResult {
    // Build strip of points within delta of mid_x
    var strip = try allocator.alloc(Point, py.len);
    defer allocator.free(strip);

    var strip_size: usize = 0;
    for (py) |p| {
        if (@abs(p.x - mid_x) < delta) {
            strip[strip_size] = p;
            strip_size += 1;
        }
    }

    if (strip_size == 0) return null;

    var min_dist = delta;
    var result: ?ClosestPairResult = null;

    // Check points in strip (at most 7 points need to be checked per point)
    for (0..strip_size) |i| {
        var j = i + 1;
        while (j < strip_size and (strip[j].y - strip[i].y) < min_dist) : (j += 1) {
            const dist = strip[i].distanceTo(strip[j]);
            if (dist < min_dist) {
                min_dist = dist;
                result = ClosestPairResult{
                    .p1 = strip[i],
                    .p2 = strip[j],
                    .distance = dist,
                };
            }
        }
    }

    return result;
}

fn compareByX(_: void, a: Point, b: Point) bool {
    return a.x < b.x;
}

fn compareByY(_: void, a: Point, b: Point) bool {
    return a.y < b.y;
}

/// Find the closest pair using brute force (for reference/testing).
///
/// Time: O(n²)
/// Space: O(1)
pub fn closestPairBruteForce(points: []const Point) error{TooFewPoints}!ClosestPairResult {
    if (points.len < 2) return error.TooFewPoints;
    return bruteForce(points);
}

// ============================================================================
// Tests
// ============================================================================

test "closestPair - basic" {
    const allocator = testing.allocator;

    const points = [_]Point{
        .{ .x = 0, .y = 0 },
        .{ .x = 1, .y = 1 },
        .{ .x = 5, .y = 5 },
        .{ .x = 10, .y = 10 },
    };

    const result = try closestPair(allocator, &points);

    // Closest should be (0,0) and (1,1) with distance sqrt(2) ≈ 1.414
    try testing.expectApproxEqAbs(1.414, result.distance, 0.01);
}

test "closestPair - two points" {
    const allocator = testing.allocator;

    const points = [_]Point{
        .{ .x = 0, .y = 0 },
        .{ .x = 3, .y = 4 },
    };

    const result = try closestPair(allocator, &points);
    try testing.expectApproxEqAbs(5.0, result.distance, 0.01); // 3-4-5 triangle
}

test "closestPair - collinear points" {
    const allocator = testing.allocator;

    const points = [_]Point{
        .{ .x = 0, .y = 0 },
        .{ .x = 1, .y = 0 },
        .{ .x = 2, .y = 0 },
        .{ .x = 5, .y = 0 },
    };

    const result = try closestPair(allocator, &points);
    try testing.expectApproxEqAbs(1.0, result.distance, 0.01);
}

test "closestPair - grid pattern" {
    const allocator = testing.allocator;

    const points = [_]Point{
        .{ .x = 0, .y = 0 },
        .{ .x = 0, .y = 1 },
        .{ .x = 1, .y = 0 },
        .{ .x = 1, .y = 1 },
        .{ .x = 5, .y = 5 },
    };

    const result = try closestPair(allocator, &points);
    try testing.expectApproxEqAbs(1.0, result.distance, 0.01);
}

test "closestPair - same x coordinates" {
    const allocator = testing.allocator;

    const points = [_]Point{
        .{ .x = 1, .y = 0 },
        .{ .x = 1, .y = 3 },
        .{ .x = 1, .y = 5 },
        .{ .x = 10, .y = 0 },
    };

    const result = try closestPair(allocator, &points);
    try testing.expectApproxEqAbs(2.0, result.distance, 0.01); // (1,3) to (1,5)
}

test "closestPair - same y coordinates" {
    const allocator = testing.allocator;

    const points = [_]Point{
        .{ .x = 0, .y = 1 },
        .{ .x = 3, .y = 1 },
        .{ .x = 5, .y = 1 },
        .{ .x = 0, .y = 10 },
    };

    const result = try closestPair(allocator, &points);
    try testing.expectApproxEqAbs(2.0, result.distance, 0.01); // (3,1) to (5,1)
}

test "closestPair - large random set" {
    const allocator = testing.allocator;

    var prng = std.rand.DefaultPrng.init(42);
    const random = prng.random();

    const n = 100;
    var points = try allocator.alloc(Point, n);
    defer allocator.free(points);

    for (0..n) |i| {
        points[i] = Point{
            .x = random.float(f64) * 100,
            .y = random.float(f64) * 100,
        };
    }

    const result = try closestPair(allocator, points);

    // Verify with brute force
    const brute_result = try closestPairBruteForce(points);
    try testing.expectApproxEqAbs(brute_result.distance, result.distance, 0.001);
}

test "closestPair - too few points" {
    const allocator = testing.allocator;

    const points = [_]Point{.{ .x = 0, .y = 0 }};
    try testing.expectError(error.TooFewPoints, closestPair(allocator, &points));

    const empty: []const Point = &[_]Point{};
    try testing.expectError(error.TooFewPoints, closestPair(allocator, empty));
}

test "closestPair - duplicates" {
    const allocator = testing.allocator;

    const points = [_]Point{
        .{ .x = 0, .y = 0 },
        .{ .x = 0, .y = 0 }, // Duplicate
        .{ .x = 10, .y = 10 },
    };

    const result = try closestPair(allocator, &points);
    try testing.expectApproxEqAbs(0.0, result.distance, 0.01); // Same point
}

test "closestPairBruteForce - basic" {
    const points = [_]Point{
        .{ .x = 0, .y = 0 },
        .{ .x = 1, .y = 1 },
        .{ .x = 5, .y = 5 },
    };

    const result = try closestPairBruteForce(&points);
    try testing.expectApproxEqAbs(1.414, result.distance, 0.01);
}

test "Point.distanceTo" {
    const p1 = Point{ .x = 0, .y = 0 };
    const p2 = Point{ .x = 3, .y = 4 };

    try testing.expectApproxEqAbs(5.0, p1.distanceTo(p2), 0.01);
    try testing.expectApproxEqAbs(5.0, p2.distanceTo(p1), 0.01); // Symmetric
}

test "closestPair vs bruteForce - small sets" {
    const allocator = testing.allocator;

    const test_cases = [_][]const Point{
        &[_]Point{
            .{ .x = 0, .y = 0 },
            .{ .x = 1, .y = 1 },
            .{ .x = 2, .y = 2 },
        },
        &[_]Point{
            .{ .x = 5, .y = 10 },
            .{ .x = 3, .y = 7 },
            .{ .x = 8, .y = 2 },
            .{ .x = 1, .y = 15 },
        },
    };

    for (test_cases) |points| {
        const dc_result = try closestPair(allocator, points);
        const bf_result = try closestPairBruteForce(points);

        try testing.expectApproxEqAbs(bf_result.distance, dc_result.distance, 0.001);
    }
}
