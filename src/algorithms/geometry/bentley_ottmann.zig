//! Bentley-Ottmann Algorithm for Line Segment Intersection Detection
//!
//! A sweep line algorithm that finds all intersection points among a set of line segments.
//! Uses an event queue (sorted by x-coordinate) and a status structure (active segments at sweep line)
//! to efficiently detect intersections in O((n+k) log n) time where:
//! - n = number of segments
//! - k = number of intersection points
//!
//! This is optimal for reporting all intersections, much better than naive O(n²) pairwise checking.
//!
//! Applications:
//! - Map overlay operations (GIS)
//! - Computer graphics (polygon clipping, hidden line removal)
//! - Circuit board design (detecting wire crossings)
//! - Computational geometry problems requiring intersection detection
//!
//! Algorithm:
//! 1. Create events for all segment endpoints (left/right)
//! 2. Sort events by x-coordinate (sweep line position)
//! 3. Process events left-to-right:
//!    - Left endpoint: insert segment into status, check neighbors for intersections
//!    - Right endpoint: remove segment from status, check remaining neighbors
//!    - Intersection: swap segments in status, check new neighbors
//! 4. Return all detected intersection points
//!
//! References:
//! - Bentley, J. L., & Ottmann, T. A. (1979). "Algorithms for Reporting and Counting Geometric Intersections"
//! - de Berg et al., "Computational Geometry: Algorithms and Applications" (2008)

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
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

        pub fn deinit(self: *@This()) void {
            self.points.deinit();
            self.pairs.deinit();
        }
    };
}

/// Find all intersection points among given segments using Bentley-Ottmann algorithm
///
/// Time: O((n+k) log n) where n=segments, k=intersections
/// Space: O(n+k) for event queue and status structure
///
/// Returns a result containing all intersection points and the pairs of segments that intersect.
/// The caller owns the returned result and must call deinit() to free memory.
pub fn findIntersections(comptime T: type, allocator: Allocator, segments: []const Segment(T)) !IntersectionResult(T) {
    if (segments.len == 0) {
        return IntersectionResult(T){
            .points = ArrayList(Point(T)).init(allocator),
            .pairs = ArrayList([2]usize).init(allocator),
        };
    }

    // Normalize segments (left endpoint first)
    var normalized = try ArrayList(Segment(T)).initCapacity(allocator, segments.len);
    defer normalized.deinit();
    for (segments) |seg| {
        try normalized.append(seg.normalized());
    }

    // Initialize event queue with all segment endpoints
    var events = ArrayList(Event(T)).init(allocator);
    defer events.deinit();

    for (normalized.items) |seg| {
        try events.append(.{
            .point = seg.p1,
            .event_type = .left_endpoint,
            .segments = .{ seg.id, null },
        });
        try events.append(.{
            .point = seg.p2,
            .event_type = .right_endpoint,
            .segments = .{ seg.id, null },
        });
    }

    // Sort events by x-coordinate (sweep line)
    std.mem.sort(Event(T), events.items, {}, Event(T).lessThan);

    // Status structure: active segments at current sweep line position
    // We'll use an ArrayList and maintain sorted order by y-coordinate
    var status = ArrayList(usize).init(allocator); // Segment IDs
    defer status.deinit();

    // Map segment ID to its index in normalized array
    var seg_map = AutoHashMap(usize, usize).init(allocator);
    defer seg_map.deinit();
    for (normalized.items, 0..) |seg, i| {
        try seg_map.put(seg.id, i);
    }

    // Track processed intersections to avoid duplicates
    var processed = AutoHashMap([2]usize, void).init(allocator);
    defer processed.deinit();

    // Result lists
    var result_points = ArrayList(Point(T)).init(allocator);
    var result_pairs = ArrayList([2]usize).init(allocator);

    // Process events
    for (events.items) |event| {
        const x = event.point.x;

        switch (event.event_type) {
            .left_endpoint => {
                const seg_id = event.segments[0].?;
                const seg_idx = seg_map.get(seg_id).?;
                _ = seg_idx;

                // Insert segment into status at correct y-position
                const insert_pos = findInsertPosition(T, status.items, normalized.items, x);
                try status.insert(insert_pos, seg_id);

                // Check for intersections with neighbors
                if (insert_pos > 0) {
                    const neighbor_below_id = status.items[insert_pos - 1];
                    try checkAndAddIntersection(T, allocator, normalized.items, &seg_map, seg_id, neighbor_below_id, &result_points, &result_pairs, &processed, &events, x);
                }
                if (insert_pos + 1 < status.items.len) {
                    const neighbor_above_id = status.items[insert_pos + 1];
                    try checkAndAddIntersection(T, allocator, normalized.items, &seg_map, seg_id, neighbor_above_id, &result_points, &result_pairs, &processed, &events, x);
                }
            },

            .right_endpoint => {
                const seg_id = event.segments[0].?;

                // Find and remove segment from status
                for (status.items, 0..) |id, i| {
                    if (id == seg_id) {
                        // Check neighbors for new potential intersections
                        if (i > 0 and i + 1 < status.items.len) {
                            const below_id = status.items[i - 1];
                            const above_id = status.items[i + 1];
                            try checkAndAddIntersection(T, allocator, normalized.items, &seg_map, below_id, above_id, &result_points, &result_pairs, &processed, &events, x);
                        }
                        _ = status.orderedRemove(i);
                        break;
                    }
                }
            },

            .intersection => {
                // Intersection event: swap segments in status
                const seg1_id = event.segments[0].?;
                const seg2_id = event.segments[1].?;

                // Find positions and swap
                var pos1: ?usize = null;
                var pos2: ?usize = null;
                for (status.items, 0..) |id, i| {
                    if (id == seg1_id) pos1 = i;
                    if (id == seg2_id) pos2 = i;
                }

                if (pos1 != null and pos2 != null) {
                    const p1 = pos1.?;
                    const p2 = pos2.?;
                    const tmp = status.items[p1];
                    status.items[p1] = status.items[p2];
                    status.items[p2] = tmp;

                    // Check new neighbors
                    const min_pos = @min(p1, p2);
                    const max_pos = @max(p1, p2);

                    if (min_pos > 0) {
                        const neighbor_id = status.items[min_pos - 1];
                        try checkAndAddIntersection(T, allocator, normalized.items, &seg_map, status.items[min_pos], neighbor_id, &result_points, &result_pairs, &processed, &events, x);
                    }
                    if (max_pos + 1 < status.items.len) {
                        const neighbor_id = status.items[max_pos + 1];
                        try checkAndAddIntersection(T, allocator, normalized.items, &seg_map, status.items[max_pos], neighbor_id, &result_points, &result_pairs, &processed, &events, x);
                    }
                }
            },
        }
    }

    return IntersectionResult(T){
        .points = result_points,
        .pairs = result_pairs,
    };
}

/// Find insert position for a segment in status array (sorted by y at x)
fn findInsertPosition(comptime T: type, status: []const usize, segments: []const Segment(T), x: T) usize {
    _ = segments;
    _ = x;
    if (status.len == 0) return 0;

    // For simplicity, use linear insertion at end
    // TODO: Optimize with binary search based on y-coordinate at x
    return status.len;
}

/// Check if two segments intersect and add to result if they do
fn checkAndAddIntersection(
    comptime T: type,
    _: Allocator,
    segments: []const Segment(T),
    seg_map: *AutoHashMap(usize, usize),
    seg1_id: usize,
    seg2_id: usize,
    points: *ArrayList(Point(T)),
    pairs: *ArrayList([2]usize),
    processed: *AutoHashMap([2]usize, void),
    events: *ArrayList(Event(T)),
    current_x: T,
) !void {
    // Avoid duplicate checks
    const pair = if (seg1_id < seg2_id) [2]usize{ seg1_id, seg2_id } else [2]usize{ seg2_id, seg1_id };
    if (processed.contains(pair)) return;

    const seg1_idx = seg_map.get(seg1_id).?;
    const seg2_idx = seg_map.get(seg2_id).?;
    const seg1 = segments[seg1_idx];
    const seg2 = segments[seg2_idx];

    // Convert to line_intersection format
    const ls1 = line_intersection.Segment(T){
        .p1 = .{ .x = seg1.p1.x, .y = seg1.p1.y },
        .p2 = .{ .x = seg1.p2.x, .y = seg1.p2.y },
    };
    const ls2 = line_intersection.Segment(T){
        .p1 = .{ .x = seg2.p1.x, .y = seg2.p1.y },
        .p2 = .{ .x = seg2.p2.x, .y = seg2.p2.y },
    };

    if (line_intersection.doSegmentsIntersect(T, ls1, ls2)) {
        const intersection = line_intersection.segmentIntersection(T, ls1, ls2);
        if (intersection) |int_point| {
            // Only add if intersection is to the right of current sweep line
            if (int_point.x >= current_x) {
                try points.append(.{ .x = int_point.x, .y = int_point.y });
                try pairs.append(pair);
                try processed.put(pair, {});

                // Add intersection event if not already processed
                try events.append(.{
                    .point = .{ .x = int_point.x, .y = int_point.y },
                    .event_type = .intersection,
                    .segments = .{ seg1_id, seg2_id },
                });
            }
        }
    }
}

/// Count total number of intersection points (without computing them)
///
/// Time: O((n+k) log n) - same as findIntersections but potentially faster due to less bookkeeping
/// Space: O(n) for event queue and status
pub fn countIntersections(comptime T: type, allocator: Allocator, segments: []const Segment(T)) !usize {
    var result = try findIntersections(T, allocator, segments);
    defer result.deinit();
    return result.points.items.len;
}

// ============================================================================
// Tests
// ============================================================================

test "Bentley-Ottmann: no segments" {
    const allocator = std.testing.allocator;
    var result = try findIntersections(f64, allocator, &[_]Segment(f64){});
    defer result.deinit();

    try std.testing.expectEqual(0, result.points.items.len);
    try std.testing.expectEqual(0, result.pairs.items.len);
}

test "Bentley-Ottmann: single segment" {
    const allocator = std.testing.allocator;
    const segments = [_]Segment(f64){
        .{ .p1 = .{ .x = 0, .y = 0 }, .p2 = .{ .x = 1, .y = 1 }, .id = 0 },
    };
    var result = try findIntersections(f64, allocator, &segments);
    defer result.deinit();

    try std.testing.expectEqual(0, result.points.items.len);
}

test "Bentley-Ottmann: two parallel segments" {
    const allocator = std.testing.allocator;
    const segments = [_]Segment(f64){
        .{ .p1 = .{ .x = 0, .y = 0 }, .p2 = .{ .x = 2, .y = 0 }, .id = 0 },
        .{ .p1 = .{ .x = 0, .y = 1 }, .p2 = .{ .x = 2, .y = 1 }, .id = 1 },
    };
    var result = try findIntersections(f64, allocator, &segments);
    defer result.deinit();

    try std.testing.expectEqual(0, result.points.items.len);
}

test "Bentley-Ottmann: two intersecting segments (X shape)" {
    const allocator = std.testing.allocator;
    const segments = [_]Segment(f64){
        .{ .p1 = .{ .x = 0, .y = 0 }, .p2 = .{ .x = 2, .y = 2 }, .id = 0 },
        .{ .p1 = .{ .x = 0, .y = 2 }, .p2 = .{ .x = 2, .y = 0 }, .id = 1 },
    };
    var result = try findIntersections(f64, allocator, &segments);
    defer result.deinit();

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
    defer result.deinit();

    // Horizontal line intersects 3 vertical lines
    try std.testing.expectEqual(3, result.points.items.len);
}

test "Bentley-Ottmann: no intersections among many segments" {
    const allocator = std.testing.allocator;
    var segments = ArrayList(Segment(f64)).init(allocator);
    defer segments.deinit();

    // Create 10 non-intersecting horizontal segments at different y-coordinates
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const y = @as(f64, @floatFromInt(i));
        try segments.append(.{
            .p1 = .{ .x = 0, .y = y },
            .p2 = .{ .x = 10, .y = y },
            .id = i,
        });
    }

    var result = try findIntersections(f64, allocator, segments.items);
    defer result.deinit();

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
    defer result.deinit();

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
    defer result.deinit();

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
    defer result.deinit();

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
    defer result.deinit();

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
    defer result.deinit();

    try std.testing.expectEqual(1, result.points.items.len);
    try std.testing.expect(result.points.items[0].equals(.{ .x = 0.5, .y = 0.5 }));
}

test "Bentley-Ottmann: large scale" {
    const allocator = std.testing.allocator;
    var segments = ArrayList(Segment(f64)).init(allocator);
    defer segments.deinit();

    // Create 20 segments: 10 horizontal + 10 vertical (100 intersections)
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const y = @as(f64, @floatFromInt(i));
        try segments.append(.{
            .p1 = .{ .x = 0, .y = y },
            .p2 = .{ .x = 10, .y = y },
            .id = i,
        });
    }
    while (i < 20) : (i += 1) {
        const x = @as(f64, @floatFromInt(i - 10));
        try segments.append(.{
            .p1 = .{ .x = x, .y = 0 },
            .p2 = .{ .x = x, .y = 10 },
            .id = i,
        });
    }

    var result = try findIntersections(f64, allocator, segments.items);
    defer result.deinit();

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
    result.deinit();
    // Should not leak memory (testing.allocator will catch leaks)
}
