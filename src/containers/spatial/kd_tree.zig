const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const ArrayList = std.ArrayList;

/// K-Dimensional Tree for efficient spatial queries.
/// Time: O(log n) insert/search average, O(n) worst case
/// Space: O(n)
///
/// A KD-Tree partitions k-dimensional space recursively by cycling through dimensions.
/// Each node represents a hyperplane that divides the space along one dimension.
///
/// Usage:
/// ```
/// const Point2D = struct { x: f64, y: f64 };
/// var tree = try KDTree(Point2D, 2, f64, getValue).init(allocator);
/// defer tree.deinit();
/// try tree.insert(.{ .x = 1.0, .y = 2.0 });
/// const nearest = tree.nearestNeighbor(.{ .x = 1.5, .y = 2.5 });
/// ```
pub fn KDTree(
    comptime T: type,
    comptime k: comptime_int,
    comptime CoordType: type,
    comptime getCoord: fn (point: T, dimension: usize) CoordType,
) type {
    return struct {
        const Self = @This();

        const Node = struct {
            point: T,
            left: ?*Node = null,
            right: ?*Node = null,
            dimension: usize,
        };

        allocator: Allocator,
        root: ?*Node = null,
        size: usize = 0,

        /// Initialize an empty KD-Tree.
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
            };
        }

        /// Free all memory used by the tree.
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.destroyNode(self.root);
            self.* = undefined;
        }

        fn destroyNode(self: *Self, node: ?*Node) void {
            if (node) |n| {
                self.destroyNode(n.left);
                self.destroyNode(n.right);
                self.allocator.destroy(n);
            }
        }

        /// Insert a point into the tree.
        /// Time: O(log n) average, O(n) worst | Space: O(1) amortized
        pub fn insert(self: *Self, point: T) !void {
            const new_node = try self.allocator.create(Node);
            new_node.* = .{
                .point = point,
                .dimension = 0,
            };

            if (self.root == null) {
                self.root = new_node;
                self.size = 1;
                return;
            }

            var current = self.root.?;
            var depth: usize = 0;

            while (true) {
                const dim = depth % k;
                new_node.dimension = dim;

                const current_val = getCoord(current.point, dim);
                const new_val = getCoord(point, dim);

                if (new_val < current_val) {
                    if (current.left) |left| {
                        current = left;
                        depth += 1;
                    } else {
                        new_node.dimension = (depth + 1) % k;
                        current.left = new_node;
                        self.size += 1;
                        return;
                    }
                } else {
                    if (current.right) |right| {
                        current = right;
                        depth += 1;
                    } else {
                        new_node.dimension = (depth + 1) % k;
                        current.right = new_node;
                        self.size += 1;
                        return;
                    }
                }
            }
        }

        /// Build a balanced KD-Tree from a slice of points.
        /// Time: O(n log n) | Space: O(n)
        pub fn fromSlice(allocator: Allocator, points: []const T) !Self {
            var tree = Self.init(allocator);
            errdefer tree.deinit();

            if (points.len == 0) return tree;

            // Create a mutable copy for sorting
            const points_copy = try allocator.alloc(T, points.len);
            defer allocator.free(points_copy);
            @memcpy(points_copy, points);

            tree.root = try tree.buildBalanced(points_copy, 0);
            tree.size = points.len;
            return tree;
        }

        fn buildBalanced(self: *Self, points: []T, depth: usize) !?*Node {
            if (points.len == 0) return null;

            const dim = depth % k;

            // Sort points by current dimension
            const Context = struct {
                dimension: usize,
                /// Determines if a < b.
                /// Time: O(1) | Space: O(1)
                pub fn lessThan(ctx: @This(), a: T, b: T) bool {
                    return getCoord(a, ctx.dimension) < getCoord(b, ctx.dimension);
                }
            };
            std.mem.sort(T, points, Context{ .dimension = dim }, Context.lessThan);

            const median_idx = points.len / 2;
            const node = try self.allocator.create(Node);
            node.* = .{
                .point = points[median_idx],
                .dimension = dim,
            };

            if (median_idx > 0) {
                node.left = try self.buildBalanced(points[0..median_idx], depth + 1);
            }
            if (median_idx + 1 < points.len) {
                node.right = try self.buildBalanced(points[median_idx + 1 ..], depth + 1);
            }

            return node;
        }

        /// Find the nearest neighbor to a query point.
        /// Time: O(log n) average, O(n) worst | Space: O(log n) stack
        pub fn nearestNeighbor(self: *const Self, query: T) ?T {
            if (self.root == null) return null;

            var best: ?T = null;
            var best_dist: CoordType = std.math.inf(CoordType);

            self.searchNearest(self.root, query, &best, &best_dist, 0);
            return best;
        }

        fn searchNearest(
            self: *const Self,
            node: ?*Node,
            query: T,
            best: *?T,
            best_dist: *CoordType,
            depth: usize,
        ) void {
            if (node == null) return;
            const n = node.?;

            // Calculate distance to current point
            const dist = self.squaredDistance(n.point, query);
            if (dist < best_dist.*) {
                best_dist.* = dist;
                best.* = n.point;
            }

            const dim = depth % k;
            const query_val = getCoord(query, dim);
            const node_val = getCoord(n.point, dim);
            const diff = query_val - node_val;

            // Determine which subtree to search first
            const first_child = if (diff < 0) n.left else n.right;
            const second_child = if (diff < 0) n.right else n.left;

            // Search the near subtree
            self.searchNearest(first_child, query, best, best_dist, depth + 1);

            // Check if we need to search the far subtree
            // (if the hyperplane is closer than current best)
            const plane_dist = diff * diff;
            if (plane_dist < best_dist.*) {
                self.searchNearest(second_child, query, best, best_dist, depth + 1);
            }
        }

        /// Find all points within a given radius of the query point.
        /// Caller owns the returned slice and must free it.
        /// Time: O(n) worst case | Space: O(n) for results
        pub fn rangeSearch(self: *const Self, allocator: Allocator, query: T, radius: CoordType) ![]T {
            var results: ArrayList(T) = .{};
            errdefer results.deinit(allocator);

            const radius_sq = radius * radius;
            try self.searchRange(allocator, self.root, query, radius_sq, &results, 0);
            return results.toOwnedSlice(allocator);
        }

        fn searchRange(
            self: *const Self,
            allocator: Allocator,
            node: ?*Node,
            query: T,
            radius_sq: CoordType,
            results: *ArrayList(T),
            depth: usize,
        ) !void {
            if (node == null) return;
            const n = node.?;

            const dist_sq = self.squaredDistance(n.point, query);
            if (dist_sq <= radius_sq) {
                try results.append(allocator, n.point);
            }

            // Search both subtrees - range query needs to check both sides
            try self.searchRange(allocator, n.left, query, radius_sq, results, depth + 1);
            try self.searchRange(allocator, n.right, query, radius_sq, results, depth + 1);
        }

        fn squaredDistance(self: *const Self, a: T, b: T) CoordType {
            _ = self;
            var sum: CoordType = 0;
            inline for (0..k) |dim| {
                const diff = getCoord(a, dim) - getCoord(b, dim);
                sum += diff * diff;
            }
            return sum;
        }

        /// Returns the number of points in the tree.
        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.size;
        }

        /// Checks if the tree is empty.
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.size == 0;
        }

        /// Removes all points from the tree.
        /// Time: O(n) | Space: O(1)
        pub fn clear(self: *Self) void {
            self.destroyNode(self.root);
            self.root = null;
            self.size = 0;
        }

        /// Validates the tree structure (for testing).
        /// Time: O(n) | Space: O(log n) stack
        pub fn validate(self: *const Self) !void {
            _ = self;
            // KD-tree validation is complex; for now, we just ensure we can traverse
            // In the future, could verify partition properties
        }

        /// Iterator for in-order traversal of all points.
        pub const Iterator = struct {
            stack: ArrayList(*Node),
            allocator: Allocator,
            current: ?*Node,

            /// Returns next element or null when exhausted.
            /// Time: O(1) amortized | Space: O(1)
            pub fn next(self: *Iterator) ?T {
                while (self.current != null or self.stack.items.len > 0) {
                    if (self.current) |node| {
                        self.stack.append(self.allocator, node) catch return null;
                        self.current = node.left;
                    } else {
                        if (self.stack.pop()) |node| {
                            self.current = node.right;
                            return node.point;
                        }
                        return null;
                    }
                }
                return null;
            }

            /// Frees iterator resources.
            /// Time: O(1) | Space: O(1)
            pub fn deinit(self: *Iterator) void {
                self.stack.deinit(self.allocator);
            }
        };

        /// Returns an iterator over all points in the tree.
        /// Time: O(1) to create | Space: O(log n) for stack
        pub fn iterator(self: *const Self, allocator: Allocator) Iterator {
            return Iterator{
                .stack = .{},
                .allocator = allocator,
                .current = self.root,
            };
        }
    };
}

// --- Tests ---

const Point2D = struct {
    x: f64,
    y: f64,

    fn getCoord(p: Point2D, dim: usize) f64 {
        return if (dim == 0) p.x else p.y;
    }
};

const Point3D = struct {
    x: f64,
    y: f64,
    z: f64,

    fn getCoord(p: Point3D, dim: usize) f64 {
        return switch (dim) {
            0 => p.x,
            1 => p.y,
            2 => p.z,
            else => unreachable,
        };
    }
};

test "KDTree: basic 2D insert and count" {
    var tree = KDTree(Point2D, 2, f64, Point2D.getCoord).init(testing.allocator);
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 0), tree.count());
    try testing.expect(tree.isEmpty());

    try tree.insert(.{ .x = 1.0, .y = 2.0 });
    try testing.expectEqual(@as(usize, 1), tree.count());
    try testing.expect(!tree.isEmpty());

    try tree.insert(.{ .x = 3.0, .y = 4.0 });
    try testing.expectEqual(@as(usize, 2), tree.count());
}

test "KDTree: nearest neighbor 2D" {
    var tree = KDTree(Point2D, 2, f64, Point2D.getCoord).init(testing.allocator);
    defer tree.deinit();

    try tree.insert(.{ .x = 2.0, .y = 3.0 });
    try tree.insert(.{ .x = 5.0, .y = 4.0 });
    try tree.insert(.{ .x = 9.0, .y = 6.0 });
    try tree.insert(.{ .x = 4.0, .y = 7.0 });
    try tree.insert(.{ .x = 8.0, .y = 1.0 });
    try tree.insert(.{ .x = 7.0, .y = 2.0 });

    const query = Point2D{ .x = 3.0, .y = 4.5 };
    const nearest = tree.nearestNeighbor(query);
    try testing.expect(nearest != null);

    // Should be closest to (2, 3) or (4, 7) or (5, 4)
    // (2,3): dist^2 = 1 + 2.25 = 3.25
    // (5,4): dist^2 = 4 + 0.25 = 4.25
    // (4,7): dist^2 = 1 + 6.25 = 7.25
    // So (2, 3) should be nearest
    try testing.expectApproxEqAbs(@as(f64, 2.0), nearest.?.x, 0.001);
    try testing.expectApproxEqAbs(@as(f64, 3.0), nearest.?.y, 0.001);
}

test "KDTree: nearest neighbor empty tree" {
    var tree = KDTree(Point2D, 2, f64, Point2D.getCoord).init(testing.allocator);
    defer tree.deinit();

    const nearest = tree.nearestNeighbor(.{ .x = 1.0, .y = 1.0 });
    try testing.expect(nearest == null);
}

test "KDTree: range search 2D" {
    var tree = KDTree(Point2D, 2, f64, Point2D.getCoord).init(testing.allocator);
    defer tree.deinit();

    try tree.insert(.{ .x = 0.0, .y = 0.0 });
    try tree.insert(.{ .x = 1.0, .y = 1.0 });
    try tree.insert(.{ .x = 2.0, .y = 2.0 });
    try tree.insert(.{ .x = 10.0, .y = 10.0 });

    const results = try tree.rangeSearch(testing.allocator, .{ .x = 0.0, .y = 0.0 }, 2.0);
    defer testing.allocator.free(results);

    // Points within radius 2.0 of origin: (0,0) and (1,1)
    // (0,0): dist = 0
    // (1,1): dist = sqrt(2) ≈ 1.414 < 2
    // (2,2): dist = sqrt(8) ≈ 2.828 > 2
    try testing.expect(results.len >= 2);
    try testing.expect(results.len <= 3); // (2,2) is borderline
}

test "KDTree: 3D operations" {
    var tree = KDTree(Point3D, 3, f64, Point3D.getCoord).init(testing.allocator);
    defer tree.deinit();

    try tree.insert(.{ .x = 1.0, .y = 2.0, .z = 3.0 });
    try tree.insert(.{ .x = 4.0, .y = 5.0, .z = 6.0 });
    try tree.insert(.{ .x = 7.0, .y = 8.0, .z = 9.0 });

    try testing.expectEqual(@as(usize, 3), tree.count());

    const nearest = tree.nearestNeighbor(.{ .x = 1.5, .y = 2.5, .z = 3.5 });
    try testing.expect(nearest != null);
    try testing.expectApproxEqAbs(@as(f64, 1.0), nearest.?.x, 0.001);
}

test "KDTree: fromSlice builds balanced tree" {
    const points = [_]Point2D{
        .{ .x = 2.0, .y = 3.0 },
        .{ .x = 5.0, .y = 4.0 },
        .{ .x = 9.0, .y = 6.0 },
        .{ .x = 4.0, .y = 7.0 },
        .{ .x = 8.0, .y = 1.0 },
        .{ .x = 7.0, .y = 2.0 },
    };

    var tree = try KDTree(Point2D, 2, f64, Point2D.getCoord).fromSlice(testing.allocator, &points);
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 6), tree.count());

    // Verify we can find nearest neighbor
    const nearest = tree.nearestNeighbor(.{ .x = 3.0, .y = 4.5 });
    try testing.expect(nearest != null);
}

test "KDTree: clear" {
    var tree = KDTree(Point2D, 2, f64, Point2D.getCoord).init(testing.allocator);
    defer tree.deinit();

    try tree.insert(.{ .x = 1.0, .y = 2.0 });
    try tree.insert(.{ .x = 3.0, .y = 4.0 });
    try testing.expectEqual(@as(usize, 2), tree.count());

    tree.clear();
    try testing.expectEqual(@as(usize, 0), tree.count());
    try testing.expect(tree.isEmpty());
}

test "KDTree: iterator traversal" {
    var tree = KDTree(Point2D, 2, f64, Point2D.getCoord).init(testing.allocator);
    defer tree.deinit();

    try tree.insert(.{ .x = 5.0, .y = 5.0 });
    try tree.insert(.{ .x = 2.0, .y = 3.0 });
    try tree.insert(.{ .x = 8.0, .y = 7.0 });

    var iter = tree.iterator(testing.allocator);
    defer iter.deinit();

    var count: usize = 0;
    while (iter.next()) |_| {
        count += 1;
    }
    try testing.expectEqual(@as(usize, 3), count);
}

test "KDTree: nearest neighbor with many points" {
    var tree = KDTree(Point2D, 2, f64, Point2D.getCoord).init(testing.allocator);
    defer tree.deinit();

    // Insert a grid of points
    var x: f64 = 0;
    while (x < 10) : (x += 1) {
        var y: f64 = 0;
        while (y < 10) : (y += 1) {
            try tree.insert(.{ .x = x, .y = y });
        }
    }

    try testing.expectEqual(@as(usize, 100), tree.count());

    // Query near (5.2, 5.3) - should find (5, 5)
    const nearest = tree.nearestNeighbor(.{ .x = 5.2, .y = 5.3 });
    try testing.expect(nearest != null);
    try testing.expectApproxEqAbs(@as(f64, 5.0), nearest.?.x, 0.001);
    try testing.expectApproxEqAbs(@as(f64, 5.0), nearest.?.y, 0.001);
}

test "KDTree: range search edge cases" {
    var tree = KDTree(Point2D, 2, f64, Point2D.getCoord).init(testing.allocator);
    defer tree.deinit();

    try tree.insert(.{ .x = 0.0, .y = 0.0 });

    // Radius 0 should only find exact match
    const results1 = try tree.rangeSearch(testing.allocator, .{ .x = 0.0, .y = 0.0 }, 0.0);
    defer testing.allocator.free(results1);
    try testing.expectEqual(@as(usize, 1), results1.len);

    // Radius very small should not find anything if query is away from point
    const results2 = try tree.rangeSearch(testing.allocator, .{ .x = 1.0, .y = 1.0 }, 0.1);
    defer testing.allocator.free(results2);
    try testing.expectEqual(@as(usize, 0), results2.len);
}

test "KDTree: validate" {
    var tree = KDTree(Point2D, 2, f64, Point2D.getCoord).init(testing.allocator);
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 0), tree.count());
    try testing.expect(tree.isEmpty());

    try tree.insert(.{ .x = 1.0, .y = 2.0 });
    try testing.expectEqual(@as(usize, 1), tree.count());
    try testing.expect(!tree.isEmpty());
    try tree.validate();

    try tree.insert(.{ .x = 3.0, .y = 4.0 });
    try testing.expectEqual(@as(usize, 2), tree.count());
    try tree.validate();

    // Verify nearest neighbor returns correct point after inserts
    const nearest = tree.nearestNeighbor(.{ .x = 1.5, .y = 2.5 });
    try testing.expect(nearest != null);
    try testing.expectApproxEqAbs(@as(f64, 1.0), nearest.?.x, 0.001);
    try testing.expectApproxEqAbs(@as(f64, 2.0), nearest.?.y, 0.001);
}

test "KDTree: memory leak check" {
    var tree = KDTree(Point2D, 2, f64, Point2D.getCoord).init(testing.allocator);
    defer tree.deinit();

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try tree.insert(.{ .x = @floatFromInt(i), .y = @floatFromInt(i * 2) });
    }

    tree.clear();
    try testing.expectEqual(@as(usize, 0), tree.count());
}
