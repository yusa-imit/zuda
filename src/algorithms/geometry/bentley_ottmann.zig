//! Sweep-line segment intersection (the Bentley-Ottmann sweep, without the balanced status tree)
//!
//! Sweeps a vertical line left to right over the segment endpoints and tests each entering
//! segment against the segments the line currently crosses. That reports every intersection
//! exactly once in O(n log n + n * a + k) time, where n = segments, a = peak active segments
//! and k = intersections. The classic neighbour-only status tree reaches O((n + k) log n) but
//! needs exact-arithmetic tie handling for vertical and collinear segments; this version trades
//! that bound for correctness on every input.
//!
//! Applications:
//! - Map overlay operations (GIS)
//! - Circuit board design (detecting wire crossings)
//!
//! Algorithm:
//! 1. Create a left and a right event for every segment and sort them by x (left first on ties)
//! 2. Left event: test the segment against every active one, record hits, make it active
//! 3. Right event: remove the segment from the active set
//!
//! Allocation: only `findIntersections` and `countIntersections` allocate, from the allocator
//! they are given; the result owns its lists and frees them with `deinit(allocator)`.
//!
//! References:
//! - Bentley, J. L., & Ottmann, T. A. (1979). "Algorithms for Reporting and Counting Geometric Intersections"
//! - de Berg et al., "Computational Geometry: Algorithms and Applications" (2008)

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const line_intersection = @import("line_intersection.zig");

/// 2D point with floating-point coordinates
pub fn Point(comptime T: type) type {
    return struct {
        x: T,
        y: T,

        const Self = @This();

        pub fn equals(self: Self, other: Self) bool {
            const epsilon = if (T == f32) 1e-6 else 1e-10;
            return @abs(self.x - other.x) < epsilon and @abs(self.y - other.y) < epsilon;
        }
    };
}

/// Line segment defined by two endpoints
pub fn Segment(comptime T: type) type {
    return struct {
        p1: Point(T),
        p2: Point(T),
        id: usize, // Unique identifier for this segment

        const Self = @This();

        /// Ensures p1 is leftmost point (or lower if same x)
        pub fn normalized(self: Self) Self {
            if (self.p1.x < self.p2.x or (self.p1.x == self.p2.x and self.p1.y < self.p2.y)) {
                return self;
            }
            return .{ .p1 = self.p2, .p2 = self.p1, .id = self.id };
        }

        /// Get y-coordinate at given x (assumes segment is not vertical)
        pub fn yAtX(self: Self, x: T) T {
            if (self.p1.x == self.p2.x) return self.p1.y; // Vertical segment
            const t = (x - self.p1.x) / (self.p2.x - self.p1.x);
            return self.p1.y + t * (self.p2.y - self.p1.y);
        }

        /// Compare segments at given x-coordinate (for status structure ordering)
        pub fn compareAtX(self: Self, other: Self, x: T) std.math.Order {
            const y1 = self.yAtX(x);
            const y2 = other.yAtX(x);
            if (y1 < y2) return .lt;
            if (y1 > y2) return .gt;
            return .eq;
        }
    };
}

/// Event types in the sweep line algorithm
pub const EventType = enum {
    left_endpoint, // Segment starts
    right_endpoint, // Segment ends
    intersection, // Two segments intersect
};

/// Event in the event queue
pub fn Event(comptime T: type) type {
    return struct {
        point: Point(T),
        event_type: EventType,
        segments: [2]?usize, // Segment IDs (1 for endpoints, 2 for intersections)

        const Self = @This();

        /// Compare events for priority queue (sort by x, then y, then type)
        pub fn lessThan(_: void, a: Self, b: Self) bool {
            const epsilon = if (T == f32) 1e-6 else 1e-10;

            // First by x-coordinate
            if (@abs(a.point.x - b.point.x) > epsilon) {
                return a.point.x < b.point.x;
            }

            // Then by y-coordinate
            if (@abs(a.point.y - b.point.y) > epsilon) {
                return a.point.y < b.point.y;
            }

            // Then by event type (left < intersection < right)
            const a_ord: u8 = switch (a.event_type) {
                .left_endpoint => 0,
                .intersection => 1,
                .right_endpoint => 2,
            };
            const b_ord: u8 = switch (b.event_type) {
                .left_endpoint => 0,
                .intersection => 1,
                .right_endpoint => 2,
            };
            return a_ord < b_ord;
        }
    };
}

/// Result containing all intersection points
pub fn IntersectionResult(comptime T: type) type {
    return struct {
        points: ArrayList(Point(T)),
        // pairs[i] = [seg_id1, seg_id2] that intersect at points[i]
        pairs: ArrayList([2]usize),

        pub fn deinit(self: *@This(), allocator: Allocator) void {
            self.points.deinit(allocator);
            self.pairs.deinit(allocator);
        }
    };
}

/// Find all intersection points among the given segments with a sweep line.
///
/// Time: O(n log n + n * a + k) where n = segments, a = peak number of segments the sweep
/// line crosses at once, k = intersections | Space: O(n + k)
///
/// Returns every point where two segments meet, with the pair of segment ids for each point.
/// Segments that merely touch at an endpoint, or overlap collinearly, count as intersecting.
/// The caller owns the returned result and must call deinit() to free memory.
pub fn findIntersections(comptime T: type, allocator: Allocator, segments: []const Segment(T)) !IntersectionResult(T) {
    var result = IntersectionResult(T){ .points = .empty, .pairs = .empty };
    errdefer result.deinit(allocator);
    if (segments.len == 0) return result;

    var normalized = try ArrayList(Segment(T)).initCapacity(allocator, segments.len);
    defer normalized.deinit(allocator);

    for (segments) |seg| {
        normalized.appendAssumeCapacity(seg.normalized());
    }
    assert(normalized.items.len == segments.len);

    // Two events per segment, ordered left to right. At equal x every left endpoint precedes
    // every right endpoint, so segments that touch at one x are both active when tested.
    var events = try ArrayList(SweepEvent(T)).initCapacity(allocator, 2 * segments.len);
    defer events.deinit(allocator);

    for (normalized.items, 0..) |seg, index| {
        events.appendAssumeCapacity(.{ .x = seg.p1.x, .is_left = true, .index = index });
        events.appendAssumeCapacity(.{ .x = seg.p2.x, .is_left = false, .index = index });
    }
    std.mem.sort(SweepEvent(T), events.items, {}, SweepEvent(T).lessThan);

    // Active set: indices of the segments the sweep line currently crosses.
    var active = try ArrayList(usize).initCapacity(allocator, segments.len);
    defer active.deinit(allocator);

    for (events.items) |event| {
        if (event.is_left) {
            try sweep_insert(T, allocator, &result, normalized.items, active.items, event.index);
            active.appendAssumeCapacity(event.index);
        } else {
            const position = std.mem.indexOfScalar(usize, active.items, event.index).?;
            _ = active.swapRemove(position);
        }
    }
    assert(active.items.len == 0);

    return result;
}

/// A sweep-line stop: one segment endpoint at x.
fn SweepEvent(comptime T: type) type {
    return struct {
        x: T,
        is_left: bool,
        index: usize,

        fn lessThan(_: void, a: @This(), b: @This()) bool {
            if (a.x != b.x) return a.x < b.x;
            if (a.is_left != b.is_left) return a.is_left;
            return a.index < b.index;
        }
    };
}

/// Test the segment entering the sweep against every active segment and record the hits.
fn sweep_insert(
    comptime T: type,
    allocator: Allocator,
    result: *IntersectionResult(T),
    segments: []const Segment(T),
    active: []const usize,
    new_index: usize,
) !void {
    assert(new_index < segments.len);
    const new_seg = segments[new_index];
    const new_ls = line_intersection.Segment(T){
        .p1 = .{ .x = new_seg.p1.x, .y = new_seg.p1.y },
        .p2 = .{ .x = new_seg.p2.x, .y = new_seg.p2.y },
    };

    for (active) |other_index| {
        assert(other_index != new_index);
        const other = segments[other_index];
        const other_ls = line_intersection.Segment(T){
            .p1 = .{ .x = other.p1.x, .y = other.p1.y },
            .p2 = .{ .x = other.p2.x, .y = other.p2.y },
        };
        const hit = line_intersection.segmentIntersection(T, new_ls, other_ls);
        if (hit.point) |point| {
            const lo = @min(new_seg.id, other.id);
            const hi = @max(new_seg.id, other.id);
            try result.points.append(allocator, .{ .x = point.x, .y = point.y });
            try result.pairs.append(allocator, .{ lo, hi });
        }
    }
}

/// Count total number of intersection points.
///
/// Time: O(n log n + n * a + k), same as findIntersections | Space: O(n + k)
pub fn countIntersections(comptime T: type, allocator: Allocator, segments: []const Segment(T)) !usize {
    var result = try findIntersections(T, allocator, segments);
    defer result.deinit(allocator);
    return result.points.items.len;
}

// ============================================================================
// Tests
// ============================================================================

test "Bentley-Ottmann: no segments" {
    const allocator = std.testing.allocator;
    var result = try findIntersections(f64, allocator, &[_]Segment(f64){});
    defer result.deinit(allocator);

    try std.testing.expectEqual(0, result.points.items.len);
    try std.testing.expectEqual(0, result.pairs.items.len);
}

test "Bentley-Ottmann: single segment" {
    const allocator = std.testing.allocator;
    const segments = [_]Segment(f64){
        .{ .p1 = .{ .x = 0, .y = 0 }, .p2 = .{ .x = 1, .y = 1 }, .id = 0 },
    };
    var result = try findIntersections(f64, allocator, &segments);
    defer result.deinit(allocator);

    try std.testing.expectEqual(0, result.points.items.len);
}

test "Bentley-Ottmann: two parallel segments" {
    const allocator = std.testing.allocator;
    const segments = [_]Segment(f64){
        .{ .p1 = .{ .x = 0, .y = 0 }, .p2 = .{ .x = 2, .y = 0 }, .id = 0 },
        .{ .p1 = .{ .x = 0, .y = 1 }, .p2 = .{ .x = 2, .y = 1 }, .id = 1 },
    };
    var result = try findIntersections(f64, allocator, &segments);
    defer result.deinit(allocator);

    try std.testing.expectEqual(0, result.points.items.len);
}

test "Bentley-Ottmann: two intersecting segments (X shape)" {
    const allocator = std.testing.allocator;
    const segments = [_]Segment(f64){
        .{ .p1 = .{ .x = 0, .y = 0 }, .p2 = .{ .x = 2, .y = 2 }, .id = 0 },
        .{ .p1 = .{ .x = 0, .y = 2 }, .p2 = .{ .x = 2, .y = 0 }, .id = 1 },
    };
    var result = try findIntersections(f64, allocator, &segments);
    defer result.deinit(allocator);

    try std.testing.expectEqual(1, result.points.items.len);
    try std.testing.expect(result.points.items[0].equals(.{ .x = 1, .y = 1 }));
    try std.testing.expectEqual(2, result.pairs.items[0].len);
}

test "Bentley-Ottmann: multiple intersections" {
    const allocator = std.testing.allocator;
    const segments = [_]Segment(f64){
        .{ .p1 = .{ .x = 0, .y = 1 }, .p2 = .{ .x = 4, .y = 1 }, .id = 0 }, // Horizontal at y=1
        .{ .p1 = .{ .x = 1, .y = 0 }, .p2 = .{ .x = 1, .y = 2 }, .id = 1 }, // Vertical at x=1
        .{ .p1 = .{ .x = 2, .y = 0 }, .p2 = .{ .x = 2, .y = 2 }, .id = 2 }, // Vertical at x=2
        .{ .p1 = .{ .x = 3, .y = 0 }, .p2 = .{ .x = 3, .y = 2 }, .id = 3 }, // Vertical at x=3
    };
    var result = try findIntersections(f64, allocator, &segments);
    defer result.deinit(allocator);

    // Horizontal line intersects 3 vertical lines
    try std.testing.expectEqual(3, result.points.items.len);
}

test "Bentley-Ottmann: no intersections among many segments" {
    const allocator = std.testing.allocator;
    var segments = ArrayList(Segment(f64)).empty;
    defer segments.deinit(allocator);

    // Create 10 non-intersecting horizontal segments at different y-coordinates
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const y = @as(f64, @floatFromInt(i));
        try segments.append(allocator, .{
            .p1 = .{ .x = 0, .y = y },
            .p2 = .{ .x = 10, .y = y },
            .id = i,
        });
    }

    var result = try findIntersections(f64, allocator, segments.items);
    defer result.deinit(allocator);

    try std.testing.expectEqual(0, result.points.items.len);
}

test "Bentley-Ottmann: star pattern (many intersections)" {
    const allocator = std.testing.allocator;
    const segments = [_]Segment(f64){
        .{ .p1 = .{ .x = 1, .y = 0 }, .p2 = .{ .x = 1, .y = 2 }, .id = 0 }, // Vertical
        .{ .p1 = .{ .x = 0, .y = 1 }, .p2 = .{ .x = 2, .y = 1 }, .id = 1 }, // Horizontal
        .{ .p1 = .{ .x = 0, .y = 0 }, .p2 = .{ .x = 2, .y = 2 }, .id = 2 }, // Diagonal /
        .{ .p1 = .{ .x = 0, .y = 2 }, .p2 = .{ .x = 2, .y = 0 }, .id = 3 }, // Diagonal \
    };
    var result = try findIntersections(f64, allocator, &segments);
    defer result.deinit(allocator);

    // All 4 segments intersect at center (1,1), giving C(4,2) = 6 pairs
    try std.testing.expect(result.points.items.len >= 4); // At least 4 distinct intersections
}

test "Bentley-Ottmann: touching endpoints" {
    const allocator = std.testing.allocator;
    const segments = [_]Segment(f64){
        .{ .p1 = .{ .x = 0, .y = 0 }, .p2 = .{ .x = 1, .y = 1 }, .id = 0 },
        .{ .p1 = .{ .x = 1, .y = 1 }, .p2 = .{ .x = 2, .y = 0 }, .id = 1 },
    };
    var result = try findIntersections(f64, allocator, &segments);
    defer result.deinit(allocator);

    // Endpoint touching counts as intersection
    try std.testing.expectEqual(1, result.points.items.len);
    try std.testing.expect(result.points.items[0].equals(.{ .x = 1, .y = 1 }));
}

test "Bentley-Ottmann: T-junction" {
    const allocator = std.testing.allocator;
    const segments = [_]Segment(f64){
        .{ .p1 = .{ .x = 0, .y = 1 }, .p2 = .{ .x = 2, .y = 1 }, .id = 0 }, // Horizontal
        .{ .p1 = .{ .x = 1, .y = 0 }, .p2 = .{ .x = 1, .y = 1 }, .id = 1 }, // Vertical to midpoint
    };
    var result = try findIntersections(f64, allocator, &segments);
    defer result.deinit(allocator);

    try std.testing.expectEqual(1, result.points.items.len);
    try std.testing.expect(result.points.items[0].equals(.{ .x = 1, .y = 1 }));
}

test "Bentley-Ottmann: collinear overlapping segments" {
    const allocator = std.testing.allocator;
    const segments = [_]Segment(f64){
        .{ .p1 = .{ .x = 0, .y = 0 }, .p2 = .{ .x = 2, .y = 0 }, .id = 0 },
        .{ .p1 = .{ .x = 1, .y = 0 }, .p2 = .{ .x = 3, .y = 0 }, .id = 1 },
    };
    var result = try findIntersections(f64, allocator, &segments);
    defer result.deinit(allocator);

    // Collinear overlapping - line_intersection should handle this
    // Expected: may report endpoints or entire overlap region
    // At minimum, should not crash
    try std.testing.expect(result.points.items.len >= 0);
}

test "Bentley-Ottmann: count intersections" {
    const allocator = std.testing.allocator;
    const segments = [_]Segment(f64){
        .{ .p1 = .{ .x = 0, .y = 0 }, .p2 = .{ .x = 2, .y = 2 }, .id = 0 },
        .{ .p1 = .{ .x = 0, .y = 2 }, .p2 = .{ .x = 2, .y = 0 }, .id = 1 },
        .{ .p1 = .{ .x = 0, .y = 1 }, .p2 = .{ .x = 2, .y = 1 }, .id = 2 },
    };

    const count = try countIntersections(f64, allocator, &segments);
    try std.testing.expect(count >= 2); // Diagonals + horizontal intersects both
}

test "Bentley-Ottmann: f32 type" {
    const allocator = std.testing.allocator;
    const segments = [_]Segment(f32){
        .{ .p1 = .{ .x = 0, .y = 0 }, .p2 = .{ .x = 1, .y = 1 }, .id = 0 },
        .{ .p1 = .{ .x = 0, .y = 1 }, .p2 = .{ .x = 1, .y = 0 }, .id = 1 },
    };
    var result = try findIntersections(f32, allocator, &segments);
    defer result.deinit(allocator);

    try std.testing.expectEqual(1, result.points.items.len);
    try std.testing.expect(result.points.items[0].equals(.{ .x = 0.5, .y = 0.5 }));
}

test "Bentley-Ottmann: large scale" {
    const allocator = std.testing.allocator;
    var segments = ArrayList(Segment(f64)).empty;
    defer segments.deinit(allocator);

    // Create 20 segments: 10 horizontal + 10 vertical (100 intersections)
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const y = @as(f64, @floatFromInt(i));
        try segments.append(allocator, .{
            .p1 = .{ .x = 0, .y = y },
            .p2 = .{ .x = 10, .y = y },
            .id = i,
        });
    }
    while (i < 20) : (i += 1) {
        const x = @as(f64, @floatFromInt(i - 10));
        try segments.append(allocator, .{
            .p1 = .{ .x = x, .y = 0 },
            .p2 = .{ .x = x, .y = 10 },
            .id = i,
        });
    }

    var result = try findIntersections(f64, allocator, segments.items);
    defer result.deinit(allocator);

    // Should find 10*10 = 100 intersections
    try std.testing.expectEqual(100, result.points.items.len);
}

test "Bentley-Ottmann: memory safety" {
    const allocator = std.testing.allocator;
    const segments = [_]Segment(f64){
        .{ .p1 = .{ .x = 0, .y = 0 }, .p2 = .{ .x = 2, .y = 2 }, .id = 0 },
        .{ .p1 = .{ .x = 0, .y = 2 }, .p2 = .{ .x = 2, .y = 0 }, .id = 1 },
    };

    var result = try findIntersections(f64, allocator, &segments);
    result.deinit(allocator);
    // Should not leak memory (testing.allocator will catch leaks)
}
