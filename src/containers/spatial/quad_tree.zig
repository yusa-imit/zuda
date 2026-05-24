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

test "QuadTree: out of bounds insertion" {
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

    // Insert valid point
    try tree.insert(.{ .x = 50, .y = 50, .id = 1 });
    try std.testing.expectEqual(@as(usize, 1), tree.size());

    // Insert out-of-bounds points (should be silently ignored)
    try tree.insert(.{ .x = -10, .y = 50, .id = 2 });
    try tree.insert(.{ .x = 50, .y = 150, .id = 3 });
    try tree.insert(.{ .x = 200, .y = 200, .id = 4 });

    // Size should remain 1 (only valid point counted)
    try std.testing.expectEqual(@as(usize, 1), tree.size());
    try tree.validate();
}

test "QuadTree: boundary edge cases" {
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

    // Insert points exactly on boundaries
    try tree.insert(.{ .x = 0, .y = 0, .id = 1 }); // min corner
    try tree.insert(.{ .x = 100, .y = 100, .id = 2 }); // max corner
    try tree.insert(.{ .x = 0, .y = 100, .id = 3 }); // top-left corner
    try tree.insert(.{ .x = 100, .y = 0, .id = 4 }); // bottom-right corner
    try tree.insert(.{ .x = 50, .y = 0, .id = 5 }); // bottom edge midpoint
    try tree.insert(.{ .x = 50, .y = 100, .id = 6 }); // top edge midpoint

    try std.testing.expectEqual(@as(usize, 6), tree.size());
    try tree.validate();

    // Test range query that includes boundaries
    const query = QT.Rect{ .min_x = 0, .max_x = 50, .min_y = 0, .max_y = 50 };
    var result = try tree.rangeQuery(std.testing.allocator, query);
    defer result.deinit(std.testing.allocator);

    // Should include points on the boundary
    try std.testing.expect(result.items.len >= 2);
}

test "QuadTree: empty range query" {
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

    // Insert points in one region
    try tree.insert(.{ .x = 10, .y = 10, .id = 1 });
    try tree.insert(.{ .x = 20, .y = 20, .id = 2 });

    // Query a completely different region (no matches)
    const query = QT.Rect{ .min_x = 70, .max_x = 90, .min_y = 70, .max_y = 90 };
    var result = try tree.rangeQuery(std.testing.allocator, query);
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 0), result.items.len);
}

test "QuadTree: nearest neighbor on empty tree" {
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

    // Query nearest on empty tree
    const nn = tree.nearest(50, 50);
    try std.testing.expect(nn == null);
}

test "QuadTree: memory safety with multiple operations" {
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

    // Multiple allocations and deallocations to detect memory leaks
    for (0..10) |_| {
        var tree = try QT.init(std.testing.allocator, ctx, bounds, 3);
        defer tree.deinit();

        // Mix operations
        for (0..20) |i| {
            const x = @as(f64, @floatFromInt(i * 5));
            const y = @as(f64, @floatFromInt(i * 5));
            try tree.insert(.{ .x = x, .y = y, .id = @intCast(i) });
        }

        const query = QT.Rect{ .min_x = 20, .max_x = 60, .min_y = 20, .max_y = 60 };
        var result = try tree.rangeQuery(std.testing.allocator, query);
        result.deinit(std.testing.allocator);

        _ = tree.nearest(50, 50);
        try tree.validate();
    }
}

test "QuadTree: deep subdivision with capacity 1" {
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

    // capacity=1 forces subdivision after each point
    var tree = try QT.init(std.testing.allocator, ctx, bounds, 1);
    defer tree.deinit();

    // Insert 8 points in all four quadrants
    try tree.insert(.{ .x = 10, .y = 10, .id = 1 }); // SW
    try tree.insert(.{ .x = 90, .y = 10, .id = 2 }); // SE
    try tree.insert(.{ .x = 10, .y = 90, .id = 3 }); // NW
    try tree.insert(.{ .x = 90, .y = 90, .id = 4 }); // NE
    try tree.insert(.{ .x = 50, .y = 10, .id = 5 }); // S-center
    try tree.insert(.{ .x = 10, .y = 50, .id = 6 }); // W-center
    try tree.insert(.{ .x = 90, .y = 50, .id = 7 }); // E-center
    try tree.insert(.{ .x = 50, .y = 90, .id = 8 }); // N-center

    try std.testing.expectEqual(@as(usize, 8), tree.size());
    try tree.validate();
}

test "QuadTree: range query returns correct point IDs" {
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

    // Insert 4 points with distinct IDs
    try tree.insert(.{ .x = 10, .y = 10, .id = 1 });
    try tree.insert(.{ .x = 80, .y = 10, .id = 2 });
    try tree.insert(.{ .x = 10, .y = 80, .id = 3 });
    try tree.insert(.{ .x = 80, .y = 80, .id = 4 });

    // Query rect covering only southwest region (10,10)
    const query = QT.Rect{ .min_x = 0, .max_x = 50, .min_y = 0, .max_y = 50 };
    var result = try tree.rangeQuery(std.testing.allocator, query);
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 1), result.items.len);
    try std.testing.expectEqual(@as(u32, 1), result.items[0].id);

    try tree.validate();
}

test "QuadTree: full bounds range query returns all points" {
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

    // Insert 5 points at various coordinates
    try tree.insert(.{ .x = 10, .y = 10, .id = 1 });
    try tree.insert(.{ .x = 30, .y = 30, .id = 2 });
    try tree.insert(.{ .x = 50, .y = 50, .id = 3 });
    try tree.insert(.{ .x = 70, .y = 70, .id = 4 });
    try tree.insert(.{ .x = 90, .y = 90, .id = 5 });

    // Query entire bounds
    const query = QT.Rect{ .min_x = 0, .max_x = 100, .min_y = 0, .max_y = 100 };
    var result = try tree.rangeQuery(std.testing.allocator, query);
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 5), result.items.len);

    try tree.validate();
}

test "QuadTree: single point tree operations" {
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

    // Insert exactly 1 point
    try tree.insert(.{ .x = 40, .y = 60, .id = 7 });
    try std.testing.expectEqual(@as(usize, 1), tree.size());

    // Nearest neighbor at exact same location
    const nn = tree.nearest(40, 60);
    try std.testing.expect(nn != null);
    try std.testing.expectEqual(@as(u32, 7), nn.?.id);

    // Range query covering the point
    const query1 = QT.Rect{ .min_x = 30, .max_x = 50, .min_y = 50, .max_y = 70 };
    var result1 = try tree.rangeQuery(std.testing.allocator, query1);
    defer result1.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), result1.items.len);
    try std.testing.expectEqual(@as(u32, 7), result1.items[0].id);

    // Range query NOT covering the point
    const query2 = QT.Rect{ .min_x = 70, .max_x = 90, .min_y = 70, .max_y = 90 };
    var result2 = try tree.rangeQuery(std.testing.allocator, query2);
    defer result2.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), result2.items.len);

    try tree.validate();
}

test "QuadTree: nearest neighbor is geometrically closest" {
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

    // Insert 3 points
    try tree.insert(.{ .x = 10, .y = 10, .id = 1 });
    try tree.insert(.{ .x = 50, .y = 50, .id = 2 });
    try tree.insert(.{ .x = 90, .y = 90, .id = 3 });

    // Query (15, 15): closest is (10, 10) with distance ~7.07
    const nn1 = tree.nearest(15, 15);
    try std.testing.expect(nn1 != null);
    try std.testing.expectEqual(@as(u32, 1), nn1.?.id);

    // Query (85, 85): closest is (90, 90) with distance ~7.07
    const nn2 = tree.nearest(85, 85);
    try std.testing.expect(nn2 != null);
    try std.testing.expectEqual(@as(u32, 3), nn2.?.id);

    try tree.validate();
}

test "QuadTree: range query after forced subdivision" {
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

    // capacity=1 forces subdivision at every insert
    var tree = try QT.init(std.testing.allocator, ctx, bounds, 1);
    defer tree.deinit();

    // Insert 6 points, spreading across quadrants
    try tree.insert(.{ .x = 5, .y = 5, .id = 1 });     // SW
    try tree.insert(.{ .x = 5, .y = 95, .id = 2 });    // NW
    try tree.insert(.{ .x = 95, .y = 5, .id = 3 });    // SE
    try tree.insert(.{ .x = 95, .y = 95, .id = 4 });   // NE
    try tree.insert(.{ .x = 50, .y = 50, .id = 5 });   // Center
    try tree.insert(.{ .x = 25, .y = 25, .id = 6 });   // SW-ish

    // Query small rect [0, 30] × [0, 30] should include id=1 and id=6
    const query = QT.Rect{ .min_x = 0, .max_x = 30, .min_y = 0, .max_y = 30 };
    var result = try tree.rangeQuery(std.testing.allocator, query);
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), result.items.len);

    // Verify both id=1 and id=6 are present
    var found_1 = false;
    var found_6 = false;
    for (result.items) |point| {
        if (point.id == 1) found_1 = true;
        if (point.id == 6) found_6 = true;
    }
    try std.testing.expect(found_1);
    try std.testing.expect(found_6);

    try tree.validate();
}
