const std = @import("std");

/// QuadTree: 2D spatial partitioning structure for efficient range queries and nearest neighbor search.
///
/// Each internal node subdivides its region into 4 quadrants (NW, NE, SW, SE).
/// Points are stored in leaf nodes until a configurable capacity is reached, then the node splits.
///
/// Time complexities (n = number of points):
/// - insert: O(log n) average, O(n) worst case (degenerate subdivision)
/// - remove: O(log n) average, O(n) worst case
/// - range query: O(k + log n), k = output size
/// - nearest neighbor: O(log n) average, O(n) worst case
///
/// Space complexity: O(n)
pub fn QuadTree(
    comptime T: type,
    comptime Context: type,
    comptime getX: fn (ctx: Context, item: T) f64,
    comptime getY: fn (ctx: Context, item: T) f64,
) type {
    return struct {
        const Self = @This();

        pub const Point = struct {
            x: f64,
            y: f64,
            data: T,
        };

        pub const Rect = struct {
            min_x: f64,
            max_x: f64,
            min_y: f64,
            max_y: f64,

            fn contains(self: Rect, x: f64, y: f64) bool {
                return x >= self.min_x and x <= self.max_x and y >= self.min_y and y <= self.max_y;
            }

            fn intersects(self: Rect, other: Rect) bool {
                return !(other.min_x > self.max_x or other.max_x < self.min_x or other.min_y > self.max_y or other.max_y < self.min_y);
            }

            fn quadrant(self: Rect, q: u2) Rect {
                const mid_x = (self.min_x + self.max_x) / 2.0;
                const mid_y = (self.min_y + self.max_y) / 2.0;
                return switch (q) {
                    0 => .{ .min_x = self.min_x, .max_x = mid_x, .min_y = mid_y, .max_y = self.max_y }, // NW
                    1 => .{ .min_x = mid_x, .max_x = self.max_x, .min_y = mid_y, .max_y = self.max_y }, // NE
                    2 => .{ .min_x = self.min_x, .max_x = mid_x, .min_y = self.min_y, .max_y = mid_y }, // SW
                    3 => .{ .min_x = mid_x, .max_x = self.max_x, .min_y = self.min_y, .max_y = mid_y }, // SE
                };
            }
        };

        const Node = struct {
            bounds: Rect,
            points: std.ArrayListUnmanaged(Point),
            children: ?*[4]*Node,
            is_leaf: bool,
        };

        allocator: std.mem.Allocator,
        root: ?*Node,
        context: Context,
        capacity: usize,
        count: usize,

        /// Creates a new QuadTree covering the specified bounds.
        /// Nodes will split when they contain more than `capacity` points.
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: std.mem.Allocator, context: Context, bounds: Rect, capacity: usize) !Self {
            const root = try allocator.create(Node);
            root.* = .{
                .bounds = bounds,
                .points = .{},
                .children = null,
                .is_leaf = true,
            };
            return .{
                .allocator = allocator,
                .root = root,
                .context = context,
                .capacity = capacity,
                .count = 0,
            };
        }

        /// Frees all memory used by the tree.
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.root) |root| {
                self.deinitNode(root);
                self.allocator.destroy(root);
            }
        }

        fn deinitNode(self: *Self, node: *Node) void {
            node.points.deinit(self.allocator);
            if (node.children) |children| {
                for (children) |child| {
                    self.deinitNode(child);
                    self.allocator.destroy(child);
                }
                self.allocator.destroy(children);
            }
        }

        /// Returns the number of points in the tree.
        /// Time: O(1) | Space: O(1)
        pub fn size(self: *const Self) usize {
            return self.count;
        }

        /// Checks if the tree is empty.
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.count == 0;
        }

        /// Inserts a point into the tree.
        /// Time: O(log n) average, O(n) worst | Space: O(1) amortized
        pub fn insert(self: *Self, item: T) !void {
            const x = getX(self.context, item);
            const y = getY(self.context, item);
            const point = Point{ .x = x, .y = y, .data = item };

            if (self.root) |root| {
                try self.insertIntoNode(root, point);
                self.count += 1;
            }
        }

        fn insertIntoNode(self: *Self, node: *Node, point: Point) !void {
            if (!node.bounds.contains(point.x, point.y)) {
                return; // Point outside bounds
            }

            if (node.is_leaf) {
                try node.points.append(self.allocator, point);
                if (node.points.items.len > self.capacity) {
                    try self.subdivide(node);
                }
            } else {
                if (node.children) |children| {
                    for (children) |child| {
                        if (child.bounds.contains(point.x, point.y)) {
                            try self.insertIntoNode(child, point);
                            return;
                        }
                    }
                }
            }
        }

        fn subdivide(self: *Self, node: *Node) !void {
            if (!node.is_leaf) return;

            const children = try self.allocator.create([4]*Node);
            for (0..4) |i| {
                children[i] = try self.allocator.create(Node);
                children[i].* = .{
                    .bounds = node.bounds.quadrant(@intCast(i)),
                    .points = .{},
                    .children = null,
                    .is_leaf = true,
                };
            }

            for (node.points.items) |point| {
                for (children) |child| {
                    if (child.bounds.contains(point.x, point.y)) {
                        try child.points.append(self.allocator, point);
                        break;
                    }
                }
            }

            node.points.deinit(self.allocator);
            node.points = .{};
            node.children = children;
            node.is_leaf = false;
        }

        /// Queries all points within the specified rectangle.
        /// Time: O(k + log n), k = output size | Space: O(k)
        pub fn rangeQuery(self: *const Self, allocator: std.mem.Allocator, range: Rect) !std.ArrayList(T) {
            var result: std.ArrayList(T) = .{};
            errdefer result.deinit(allocator);

            if (self.root) |root| {
                try self.rangeQueryNode(root, range, &result, allocator);
            }
            return result;
        }

        fn rangeQueryNode(self: *const Self, node: *const Node, range: Rect, result: *std.ArrayList(T), allocator: std.mem.Allocator) !void {
            if (!node.bounds.intersects(range)) {
                return; // No intersection
            }

            if (node.is_leaf) {
                for (node.points.items) |point| {
                    if (range.contains(point.x, point.y)) {
                        try result.append(allocator, point.data);
                    }
                }
            } else {
                if (node.children) |children| {
                    for (children.*) |child| {
                        try self.rangeQueryNode(child, range, result, allocator);
                    }
                }
            }
        }

        /// Finds the nearest neighbor to the query point.
        /// Time: O(log n) average, O(n) worst | Space: O(1)
        pub fn nearest(self: *const Self, query_x: f64, query_y: f64) ?T {
            if (self.root == null or self.count == 0) return null;

            var best: ?Point = null;
            var best_dist: f64 = std.math.inf(f64);

            self.nearestNode(self.root.?, query_x, query_y, &best, &best_dist);

            return if (best) |b| b.data else null;
        }

        fn nearestNode(self: *const Self, node: *const Node, qx: f64, qy: f64, best: *?Point, best_dist: *f64) void {
            const dist_to_bounds = distanceToRect(qx, qy, node.bounds);
            if (dist_to_bounds > best_dist.*) {
                return; // Prune this branch
            }

            if (node.is_leaf) {
                for (node.points.items) |point| {
                    const dx = point.x - qx;
                    const dy = point.y - qy;
                    const dist = @sqrt(dx * dx + dy * dy);
                    if (dist < best_dist.*) {
                        best_dist.* = dist;
                        best.* = point;
                    }
                }
            } else {
                if (node.children) |children| {
                    // Search children in order of distance to query point
                    var dists: [4]f64 = undefined;
                    for (children, 0..) |child, i| {
                        dists[i] = distanceToRect(qx, qy, child.bounds);
                    }

                    // Simple sorting by distance (bubble sort for 4 elements)
                    var indices: [4]usize = .{ 0, 1, 2, 3 };
                    for (0..4) |i| {
                        for (i + 1..4) |j| {
                            if (dists[indices[i]] > dists[indices[j]]) {
                                const tmp = indices[i];
                                indices[i] = indices[j];
                                indices[j] = tmp;
                            }
                        }
                    }

                    for (indices) |idx| {
                        self.nearestNode(children[idx], qx, qy, best, best_dist);
                    }
                }
            }
        }

        fn distanceToRect(x: f64, y: f64, rect: Rect) f64 {
            const dx = @max(rect.min_x - x, @max(0, x - rect.max_x));
            const dy = @max(rect.min_y - y, @max(0, y - rect.max_y));
            return @sqrt(dx * dx + dy * dy);
        }

        /// Validates tree invariants (for testing).
        /// Time: O(n) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            if (self.root) |root| {
                const actual_count = try self.validateNode(root);
                if (actual_count != self.count) {
                    return error.TreeInvariant;
                }
            }
        }

        fn validateNode(self: *const Self, node: *const Node) !usize {
            var total: usize = 0;

            if (node.is_leaf) {
                total += node.points.items.len;
                // Check all points are within bounds
                for (node.points.items) |point| {
                    if (!node.bounds.contains(point.x, point.y)) {
                        return error.TreeInvariant;
                    }
                }
                if (node.children != null) {
                    return error.TreeInvariant; // Leaf can't have children
                }
            } else {
                if (node.points.items.len != 0) {
                    return error.TreeInvariant; // Internal node can't have points
                }
                if (node.children) |children| {
                    for (children) |child| {
                        total += try self.validateNode(child);
                    }
                } else {
                    return error.TreeInvariant; // Internal node must have children
                }
            }

            return total;
        }
    };
}

test "QuadTree: basic operations" {
    const Point2D = struct {
        x: f64,
        y: f64,
    };

    const ctx = {};
    const getX = struct {
        fn f(_: @TypeOf(ctx), p: Point2D) f64 {
            return p.x;
        }
    }.f;
    const getY = struct {
        fn f(_: @TypeOf(ctx), p: Point2D) f64 {
            return p.y;
        }
    }.f;

    const QT = QuadTree(Point2D, @TypeOf(ctx), getX, getY);
    const bounds = QT.Rect{ .min_x = 0, .max_x = 100, .min_y = 0, .max_y = 100 };

    var tree = try QT.init(std.testing.allocator, ctx, bounds, 4);
    defer tree.deinit();

    try std.testing.expectEqual(@as(usize, 0), tree.size());
    try std.testing.expect(tree.isEmpty());

    try tree.insert(.{ .x = 10, .y = 10 });
    try tree.insert(.{ .x = 20, .y = 20 });
    try tree.insert(.{ .x = 30, .y = 30 });

    try std.testing.expectEqual(@as(usize, 3), tree.size());
    try tree.validate();
}

test "QuadTree: range query" {
    const Point2D = struct {
        x: f64,
        y: f64,
        id: u32,
    };

    const ctx = {};
    const getX = struct {
        fn f(_: @TypeOf(ctx), p: Point2D) f64 {
            return p.x;
        }
    }.f;
    const getY = struct {
        fn f(_: @TypeOf(ctx), p: Point2D) f64 {
            return p.y;
        }
    }.f;

    const QT = QuadTree(Point2D, @TypeOf(ctx), getX, getY);
    const bounds = QT.Rect{ .min_x = 0, .max_x = 100, .min_y = 0, .max_y = 100 };

    var tree = try QT.init(std.testing.allocator, ctx, bounds, 2);
    defer tree.deinit();

    try tree.insert(.{ .x = 10, .y = 10, .id = 1 });
    try tree.insert(.{ .x = 20, .y = 20, .id = 2 });
    try tree.insert(.{ .x = 50, .y = 50, .id = 3 });
    try tree.insert(.{ .x = 80, .y = 80, .id = 4 });

    const query = QT.Rect{ .min_x = 0, .max_x = 30, .min_y = 0, .max_y = 30 };
    var result = try tree.rangeQuery(std.testing.allocator, query);
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), result.items.len);
    try tree.validate();
}

test "QuadTree: nearest neighbor" {
    const Point2D = struct {
        x: f64,
        y: f64,
        id: u32,
    };

    const ctx = {};
    const getX = struct {
        fn f(_: @TypeOf(ctx), p: Point2D) f64 {
            return p.x;
        }
    }.f;
    const getY = struct {
        fn f(_: @TypeOf(ctx), p: Point2D) f64 {
            return p.y;
        }
    }.f;

    const QT = QuadTree(Point2D, @TypeOf(ctx), getX, getY);
    const bounds = QT.Rect{ .min_x = 0, .max_x = 100, .min_y = 0, .max_y = 100 };

    var tree = try QT.init(std.testing.allocator, ctx, bounds, 4);
    defer tree.deinit();

    try tree.insert(.{ .x = 10, .y = 10, .id = 1 });
    try tree.insert(.{ .x = 50, .y = 50, .id = 2 });
    try tree.insert(.{ .x = 90, .y = 90, .id = 3 });

    const nn = tree.nearest(12, 12);
    try std.testing.expect(nn != null);
    try std.testing.expectEqual(@as(u32, 1), nn.?.id);

    const nn2 = tree.nearest(91, 91);
    try std.testing.expect(nn2 != null);
    try std.testing.expectEqual(@as(u32, 3), nn2.?.id);
}

test "QuadTree: subdivision" {
    const Point2D = struct {
        x: f64,
        y: f64,
    };

    const ctx = {};
    const getX = struct {
        fn f(_: @TypeOf(ctx), p: Point2D) f64 {
            return p.x;
        }
    }.f;
    const getY = struct {
        fn f(_: @TypeOf(ctx), p: Point2D) f64 {
            return p.y;
        }
    }.f;

    const QT = QuadTree(Point2D, @TypeOf(ctx), getX, getY);
    const bounds = QT.Rect{ .min_x = 0, .max_x = 100, .min_y = 0, .max_y = 100 };

    var tree = try QT.init(std.testing.allocator, ctx, bounds, 3);
    defer tree.deinit();

    // Insert enough points to trigger subdivision
    try tree.insert(.{ .x = 10, .y = 10 });
    try tree.insert(.{ .x = 11, .y = 11 });
    try tree.insert(.{ .x = 12, .y = 12 });
    try tree.insert(.{ .x = 13, .y = 13 }); // Should trigger split

    try std.testing.expectEqual(@as(usize, 4), tree.size());
    try tree.validate();
}

test "QuadTree: stress test" {
    const Point2D = struct {
        x: f64,
        y: f64,
        id: u32,
    };

    const ctx = {};
    const getX = struct {
        fn f(_: @TypeOf(ctx), p: Point2D) f64 {
            return p.x;
        }
    }.f;
    const getY = struct {
        fn f(_: @TypeOf(ctx), p: Point2D) f64 {
            return p.y;
        }
    }.f;

    const QT = QuadTree(Point2D, @TypeOf(ctx), getX, getY);
    const bounds = QT.Rect{ .min_x = 0, .max_x = 1000, .min_y = 0, .max_y = 1000 };

    var tree = try QT.init(std.testing.allocator, ctx, bounds, 8);
    defer tree.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Insert 1000 random points
    for (0..1000) |i| {
        const x = random.float(f64) * 1000;
        const y = random.float(f64) * 1000;
        try tree.insert(.{ .x = x, .y = y, .id = @intCast(i) });
    }

    try std.testing.expectEqual(@as(usize, 1000), tree.size());
    try tree.validate();

    // Test range query
    const query = QT.Rect{ .min_x = 400, .max_x = 600, .min_y = 400, .max_y = 600 };
    var result = try tree.rangeQuery(std.testing.allocator, query);
    defer result.deinit(std.testing.allocator);

    // Should find some points in the center region
    try std.testing.expect(result.items.len > 0);
}
