//! Douglas-Peucker Algorithm for Polygon/Polyline Simplification
//!
//! This module implements the Ramer-Douglas-Peucker algorithm, which reduces
//! the number of points in a curve/polyline while preserving its shape.
//! The algorithm recursively divides the line and removes points that are
//! within a specified distance (epsilon) from the simplified line.
//!
//! Applications:
//! - GIS systems: Simplifying map features for different zoom levels
//! - Data compression: Reducing storage for GPS tracks
//! - Computer graphics: Level-of-detail rendering
//! - Data visualization: Reducing complexity while maintaining shape
//! - Cartography: Generalizing coastlines and boundaries
//!
//! Reference: Douglas, D. H., & Peucker, T. K. (1973).
//!           "Algorithms for the reduction of the number of points required
//!            to represent a digitized line or its caricature"
//!
//! Time: O(n²) worst case, O(n log n) average case
//! Space: O(n) for recursion stack

const std = @import("std");
const testing = std.testing;
const math = std.math;
const Allocator = std.mem.Allocator;

/// 2D Point representation
pub const Point = struct {
    x: f64,
    y: f64,

    pub fn init(x: f64, y: f64) Point {
        return .{ .x = x, .y = y };
    }

    /// Equality comparison with epsilon tolerance
    pub fn eql(self: Point, other: Point, epsilon: f64) bool {
        return @abs(self.x - other.x) < epsilon and @abs(self.y - other.y) < epsilon;
    }
};

/// Calculate perpendicular distance from point to line segment.
/// Uses the formula: |Ax + By + C| / sqrt(A² + B²)
/// where the line through p1 and p2 is: Ax + By + C = 0
///
/// Time: O(1)
/// Space: O(1)
pub fn perpendicularDistance(point: Point, line_start: Point, line_end: Point) f64 {
    const dx = line_end.x - line_start.x;
    const dy = line_end.y - line_start.y;

    // Handle degenerate case: line is actually a point
    const line_length_squared = dx * dx + dy * dy;
    if (line_length_squared == 0.0) {
        // Distance to a point is just Euclidean distance
        const px = point.x - line_start.x;
        const py = point.y - line_start.y;
        return @sqrt(px * px + py * py);
    }

    // Calculate perpendicular distance using cross product formula
    // |cross product| / |line length|
    const numerator = @abs((line_end.y - line_start.y) * point.x -
        (line_end.x - line_start.x) * point.y +
        line_end.x * line_start.y -
        line_end.y * line_start.x);

    const denominator = @sqrt(line_length_squared);

    return numerator / denominator;
}

/// Simplify a polyline using the Douglas-Peucker algorithm.
///
/// The algorithm works recursively:
/// 1. Find the point with maximum distance from the line segment (start, end)
/// 2. If max distance > epsilon, recursively simplify the two subsegments
/// 3. Otherwise, remove all intermediate points
///
/// Parameters:
///   - allocator: Memory allocator for the result array
///   - points: Input polyline points
///   - epsilon: Distance threshold for simplification (larger = more aggressive)
///
/// Returns:
///   - ArrayList of simplified points (caller must deinit)
///
/// Time: O(n²) worst case, O(n log n) average case
/// Space: O(n) for recursion and result
///
/// Example:
/// ```zig
/// var points = [_]Point{
///     Point.init(0, 0),
///     Point.init(1, 0.1),
///     Point.init(2, -0.1),
///     Point.init(3, 5),
///     Point.init(4, 6),
///     Point.init(5, 7),
///     Point.init(6, 8.1),
///     Point.init(7, 9),
///     Point.init(8, 9),
///     Point.init(9, 9),
/// };
/// var result = try simplify(allocator, &points, 1.0);
/// defer result.deinit();
/// // Result typically: [(0,0), (3,5), (7,9), (9,9)]
/// ```
pub fn simplify(allocator: Allocator, points: []const Point, epsilon: f64) !std.ArrayList(Point) {
    if (points.len <= 2) {
        // Base case: 2 or fewer points cannot be simplified
        var result = std.ArrayList(Point).init(allocator);
        try result.appendSlice(points);
        return result;
    }

    // Create a boolean array to mark which points to keep
    var keep = try allocator.alloc(bool, points.len);
    defer allocator.free(keep);
    @memset(keep, false);

    // Always keep first and last points
    keep[0] = true;
    keep[points.len - 1] = true;

    // Recursively simplify
    try simplifyRecursive(points, epsilon, 0, points.len - 1, keep);

    // Build result array with kept points
    var result = std.ArrayList(Point).init(allocator);
    for (points, 0..) |point, i| {
        if (keep[i]) {
            try result.append(point);
        }
    }

    return result;
}

/// Recursive helper function for Douglas-Peucker algorithm
fn simplifyRecursive(
    points: []const Point,
    epsilon: f64,
    start_idx: usize,
    end_idx: usize,
    keep: []bool,
) !void {
    if (end_idx <= start_idx + 1) {
        // No intermediate points
        return;
    }

    // Find the point with maximum distance from the line segment
    var max_dist: f64 = 0.0;
    var max_idx: usize = start_idx;

    const line_start = points[start_idx];
    const line_end = points[end_idx];

    for (start_idx + 1..end_idx) |i| {
        const dist = perpendicularDistance(points[i], line_start, line_end);
        if (dist > max_dist) {
            max_dist = dist;
            max_idx = i;
        }
    }

    // If max distance is greater than epsilon, recursively simplify
    if (max_dist > epsilon) {
        keep[max_idx] = true;
        try simplifyRecursive(points, epsilon, start_idx, max_idx, keep);
        try simplifyRecursive(points, epsilon, max_idx, end_idx, keep);
    }
    // Otherwise, all intermediate points are removed (keep remains false)
}

/// Calculate the compression ratio achieved by simplification
///
/// Returns a value between 0.0 and 1.0:
/// - 1.0 means no compression (all points kept)
/// - 0.5 means 50% of points removed
/// - 0.1 means 90% compression
pub fn compressionRatio(original_count: usize, simplified_count: usize) f64 {
    if (original_count == 0) return 1.0;
    return @as(f64, @floatFromInt(simplified_count)) / @as(f64, @floatFromInt(original_count));
}

/// Calculate the maximum deviation between original and simplified polylines.
/// This measures the quality of the simplification - smaller values indicate
/// better shape preservation.
///
/// Time: O(n × m) where n = original points, m = simplified points
/// Space: O(1)
pub fn maxDeviation(original: []const Point, simplified: []const Point) f64 {
    if (original.len == 0 or simplified.len < 2) return 0.0;

    var max_dev: f64 = 0.0;

    // For each original point, find minimum distance to simplified polyline
    for (original) |orig_point| {
        var min_dist = math.inf(f64);

        // Check distance to each segment in simplified polyline
        for (0..simplified.len - 1) |i| {
            const dist = perpendicularDistance(orig_point, simplified[i], simplified[i + 1]);
            min_dist = @min(min_dist, dist);
        }

        max_dev = @max(max_dev, min_dist);
    }

    return max_dev;
}

// ============================================================================
// Tests
// ============================================================================

test "perpendicular distance - point on line" {
    const p = Point.init(1, 1);
    const start = Point.init(0, 0);
    const end = Point.init(2, 2);

    const dist = perpendicularDistance(p, start, end);
    try testing.expectApproxEqAbs(0.0, dist, 1e-10);
}

test "perpendicular distance - point above line" {
    const p = Point.init(1, 2);
    const start = Point.init(0, 0);
    const end = Point.init(2, 0);

    const dist = perpendicularDistance(p, start, end);
    try testing.expectApproxEqAbs(2.0, dist, 1e-10);
}

test "perpendicular distance - degenerate line" {
    const p = Point.init(3, 4);
    const start = Point.init(0, 0);
    const end = Point.init(0, 0);

    const dist = perpendicularDistance(p, start, end);
    try testing.expectApproxEqAbs(5.0, dist, 1e-10); // sqrt(3^2 + 4^2)
}

test "simplify - straight line remains unchanged" {
    const allocator = testing.allocator;
    var points = [_]Point{
        Point.init(0, 0),
        Point.init(1, 1),
        Point.init(2, 2),
        Point.init(3, 3),
    };

    var result = try simplify(allocator, &points, 0.1);
    defer result.deinit();

    // All points are collinear within tolerance, so only endpoints kept
    try testing.expectEqual(@as(usize, 2), result.items.len);
    try testing.expect(result.items[0].eql(points[0], 1e-10));
    try testing.expect(result.items[1].eql(points[3], 1e-10));
}

test "simplify - single outlier kept" {
    const allocator = testing.allocator;
    var points = [_]Point{
        Point.init(0, 0),
        Point.init(1, 0),
        Point.init(2, 5), // Outlier
        Point.init(3, 0),
        Point.init(4, 0),
    };

    var result = try simplify(allocator, &points, 1.0);
    defer result.deinit();

    // Should keep outlier point
    try testing.expect(result.items.len >= 3);
    var found_outlier = false;
    for (result.items) |p| {
        if (p.eql(points[2], 1e-10)) {
            found_outlier = true;
            break;
        }
    }
    try testing.expect(found_outlier);
}

test "simplify - epsilon effect" {
    const allocator = testing.allocator;
    var points = [_]Point{
        Point.init(0, 0),
        Point.init(1, 0.1),
        Point.init(2, -0.1),
        Point.init(3, 0.1),
        Point.init(4, 0),
    };

    // Small epsilon: keep more points
    var result_small = try simplify(allocator, &points, 0.05);
    defer result_small.deinit();

    // Large epsilon: remove more points
    var result_large = try simplify(allocator, &points, 0.5);
    defer result_large.deinit();

    try testing.expect(result_small.items.len >= result_large.items.len);
}

test "simplify - zigzag pattern" {
    const allocator = testing.allocator;
    var points = [_]Point{
        Point.init(0, 0),
        Point.init(1, 1),
        Point.init(2, 0),
        Point.init(3, 1),
        Point.init(4, 0),
        Point.init(5, 1),
        Point.init(6, 0),
    };

    var result = try simplify(allocator, &points, 0.5);
    defer result.deinit();

    // Should significantly reduce points while maintaining overall zigzag shape
    try testing.expect(result.items.len < points.len);
    try testing.expect(result.items.len >= 2); // At least endpoints
}

test "simplify - minimal input (2 points)" {
    const allocator = testing.allocator;
    var points = [_]Point{
        Point.init(0, 0),
        Point.init(1, 1),
    };

    var result = try simplify(allocator, &points, 1.0);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.items.len);
}

test "simplify - single point" {
    const allocator = testing.allocator;
    var points = [_]Point{Point.init(0, 0)};

    var result = try simplify(allocator, &points, 1.0);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.items.len);
}

test "simplify - empty input" {
    const allocator = testing.allocator;
    var points = [_]Point{};

    var result = try simplify(allocator, &points, 1.0);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 0), result.items.len);
}

test "simplify - GPS track example" {
    const allocator = testing.allocator;
    // Simulate GPS track with noise
    var points = [_]Point{
        Point.init(0.0, 0.0),
        Point.init(1.0, 0.05), // Slight noise
        Point.init(2.0, -0.03), // Slight noise
        Point.init(3.0, 0.02), // Slight noise
        Point.init(4.0, 0.0),
        Point.init(5.0, 1.0), // Turn
        Point.init(6.0, 2.0),
        Point.init(7.0, 2.95), // Slight noise
        Point.init(8.0, 3.0),
    };

    var result = try simplify(allocator, &points, 0.1);
    defer result.deinit();

    // Should remove noisy points but keep significant features
    try testing.expect(result.items.len < points.len);
    try testing.expect(result.items.len >= 3); // At least endpoints and turn
}

test "compressionRatio - calculation" {
    try testing.expectApproxEqAbs(1.0, compressionRatio(10, 10), 1e-10);
    try testing.expectApproxEqAbs(0.5, compressionRatio(10, 5), 1e-10);
    try testing.expectApproxEqAbs(0.1, compressionRatio(100, 10), 1e-10);
    try testing.expectApproxEqAbs(1.0, compressionRatio(0, 0), 1e-10);
}

test "maxDeviation - measures simplification quality" {
    var original = [_]Point{
        Point.init(0, 0),
        Point.init(1, 0),
        Point.init(2, 0),
        Point.init(3, 0),
        Point.init(4, 0),
    };

    var simplified = [_]Point{
        Point.init(0, 0),
        Point.init(4, 0),
    };

    const dev = maxDeviation(&original, &simplified);
    try testing.expectApproxEqAbs(0.0, dev, 1e-10); // All points on line

    // Test with outlier
    var original_outlier = [_]Point{
        Point.init(0, 0),
        Point.init(1, 0),
        Point.init(2, 5), // Outlier
        Point.init(3, 0),
        Point.init(4, 0),
    };

    const dev_outlier = maxDeviation(&original_outlier, &simplified);
    try testing.expect(dev_outlier > 4.0); // Significant deviation from outlier
}

test "maxDeviation - empty input" {
    var original = [_]Point{};
    var simplified = [_]Point{};

    const dev = maxDeviation(&original, &simplified);
    try testing.expectEqual(@as(f64, 0.0), dev);
}

test "simplify - large scale test" {
    const allocator = testing.allocator;

    // Create a polyline with 1000 points
    var points = try allocator.alloc(Point, 1000);
    defer allocator.free(points);

    // Generate sinusoidal curve with noise
    for (0..1000) |i| {
        const x = @as(f64, @floatFromInt(i)) / 100.0;
        const y = @sin(x) + (@as(f64, @floatFromInt(i % 10)) - 5.0) * 0.01;
        points[i] = Point.init(x, y);
    }

    var result = try simplify(allocator, points, 0.1);
    defer result.deinit();

    // Should achieve significant compression
    try testing.expect(result.items.len < points.len);
    try testing.expect(result.items.len < 200); // At least 80% compression

    // First and last points should be preserved
    try testing.expect(result.items[0].eql(points[0], 1e-10));
    try testing.expect(result.items[result.items.len - 1].eql(points[points.len - 1], 1e-10));
}

test "simplify - memory safety" {
    const allocator = testing.allocator;
    var points = [_]Point{
        Point.init(0, 0),
        Point.init(1, 1),
        Point.init(2, 0),
    };

    var result = try simplify(allocator, &points, 0.5);
    defer result.deinit();

    // Should not leak memory
    try testing.expect(result.items.len > 0);
}
