const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// OctTree - 3D spatial partitioning structure for efficient spatial queries.
/// Recursively subdivides 3D space into eight octants when capacity is exceeded.
///
/// Time Complexity:
/// - insert: O(log n) average, O(n) worst case (degenerate)
/// - remove: O(log n) average
/// - query (point): O(log n) average
/// - query (range): O(k + log n) where k is number of results
///
/// Space Complexity: O(n)
///
/// Generic parameters:
/// - T: Value type stored at each point
/// - capacity: Maximum points per node before subdivision (comptime)
pub fn OctTree(comptime T: type, comptime capacity: usize) type {
    return struct {
        const Self = @This();

        pub const Point3D = struct {
            x: f64,
            y: f64,
            z: f64,

            /// Checks if two objects are equal.
            /// Time: O(1) | Space: O(1)
            pub fn equals(self: Point3D, other: Point3D) bool {
                const epsilon = 1e-9;
                return @abs(self.x - other.x) < epsilon and
                    @abs(self.y - other.y) < epsilon and
                    @abs(self.z - other.z) < epsilon;
            }
        };

        pub const Entry = struct {
            point: Point3D,
            value: T,
        };

        pub const AABB = struct {
            min: Point3D,
            max: Point3D,

            /// Checks if a point is contained in the region.
            /// Time: O(1) | Space: O(1)
            pub fn contains(self: AABB, point: Point3D) bool {
                return point.x >= self.min.x and point.x <= self.max.x and
                    point.y >= self.min.y and point.y <= self.max.y and
                    point.z >= self.min.z and point.z <= self.max.z;
            }

            /// Checks if two regions intersect.
            /// Time: O(1) | Space: O(1)
            pub fn intersects(self: AABB, other: AABB) bool {
                return self.min.x <= other.max.x and self.max.x >= other.min.x and
                    self.min.y <= other.max.y and self.max.y >= other.min.y and
                    self.min.z <= other.max.z and self.max.z >= other.min.z;
            }

            /// Returns the center point of the region.
            /// Time: O(1) | Space: O(1)
            pub fn center(self: AABB) Point3D {
                return .{
                    .x = (self.min.x + self.max.x) / 2.0,
                    .y = (self.min.y + self.max.y) / 2.0,
                    .z = (self.min.z + self.max.z) / 2.0,
                };
            }
        };

        const Node = struct {
            boundary: AABB,
            points: [capacity]Entry,
            points_len: usize,
            // Eight octants: TNE, TNW, TSE, TSW, BNE, BNW, BSE, BSW
            // T=Top, B=Bottom, N=North, S=South, E=East, W=West
            children: ?[8]*Node,
            allocator: Allocator,

            fn init(allocator: Allocator, boundary: AABB) !*Node {
                const node = try allocator.create(Node);
                node.* = .{
                    .boundary = boundary,
                    .points = undefined,
                    .points_len = 0,
                    .children = null,
                    .allocator = allocator,
                };
                return node;
            }

            fn deinit(self: *Node) void {
                if (self.children) |children| {
                    for (children) |child| {
                        child.deinit();
                        self.allocator.destroy(child);
                    }
                }
            }

            fn subdivide(self: *Node) !void {
                const c = self.boundary.center();
                const min = self.boundary.min;
                const max = self.boundary.max;

                var children: [8]*Node = undefined;

                // Top octants (z > center.z)
                children[0] = try Node.init(self.allocator, .{
                    .min = .{ .x = c.x, .y = c.y, .z = c.z },
                    .max = .{ .x = max.x, .y = max.y, .z = max.z },
                }); // TNE
                children[1] = try Node.init(self.allocator, .{
                    .min = .{ .x = min.x, .y = c.y, .z = c.z },
                    .max = .{ .x = c.x, .y = max.y, .z = max.z },
                }); // TNW
                children[2] = try Node.init(self.allocator, .{
                    .min = .{ .x = c.x, .y = min.y, .z = c.z },
                    .max = .{ .x = max.x, .y = c.y, .z = max.z },
                }); // TSE
                children[3] = try Node.init(self.allocator, .{
                    .min = .{ .x = min.x, .y = min.y, .z = c.z },
                    .max = .{ .x = c.x, .y = c.y, .z = max.z },
                }); // TSW

                // Bottom octants (z <= center.z)
                children[4] = try Node.init(self.allocator, .{
                    .min = .{ .x = c.x, .y = c.y, .z = min.z },
                    .max = .{ .x = max.x, .y = max.y, .z = c.z },
                }); // BNE
                children[5] = try Node.init(self.allocator, .{
                    .min = .{ .x = min.x, .y = c.y, .z = min.z },
                    .max = .{ .x = c.x, .y = max.y, .z = c.z },
                }); // BNW
                children[6] = try Node.init(self.allocator, .{
                    .min = .{ .x = c.x, .y = min.y, .z = min.z },
                    .max = .{ .x = max.x, .y = c.y, .z = c.z },
                }); // BSE
                children[7] = try Node.init(self.allocator, .{
                    .min = .{ .x = min.x, .y = min.y, .z = min.z },
                    .max = .{ .x = c.x, .y = c.y, .z = c.z },
                }); // BSW

                self.children = children;
            }

            /// Time: O(log n) average, O(n) worst case
            fn insert(self: *Node, entry: Entry) !bool {
                if (!self.boundary.contains(entry.point)) {
                    return false;
                }

                if (self.children == null and self.points_len < capacity) {
                    self.points[self.points_len] = entry;
                    self.points_len += 1;
                    return true;
                }

                if (self.children == null) {
                    try self.subdivide();
                    // Redistribute existing points
                    const old_len = self.points_len;
                    const old_points = self.points;
                    self.points_len = 0;
                    for (old_points[0..old_len]) |p| {
                        var inserted = false;
                        for (self.children.?) |child| {
                            if (try child.insert(p)) {
                                inserted = true;
                                break;
                            }
                        }
                        if (!inserted) return error.SubdivisionFailed;
                    }
                }

                for (self.children.?) |child| {
                    if (try child.insert(entry)) {
                        return true;
                    }
                }

                return false;
            }

            /// Time: O(log n) average
            fn remove(self: *Node, point: Point3D) bool {
                if (!self.boundary.contains(point)) {
                    return false;
                }

                if (self.children == null) {
                    for (self.points[0..self.points_len], 0..) |entry, i| {
                        if (entry.point.equals(point)) {
                            // Swap with last element and decrement length
                            if (i < self.points_len - 1) {
                                self.points[i] = self.points[self.points_len - 1];
                            }
                            self.points_len -= 1;
                            return true;
                        }
                    }
                    return false;
                }

                for (self.children.?) |child| {
                    if (child.remove(point)) {
                        return true;
                    }
                }

                return false;
            }

            /// Time: O(log n) average
            fn get(self: *const Node, point: Point3D) ?T {
                if (!self.boundary.contains(point)) {
                    return null;
                }

                if (self.children == null) {
                    for (self.points[0..self.points_len]) |entry| {
                        if (entry.point.equals(point)) {
                            return entry.value;
                        }
                    }
                    return null;
                }

                for (self.children.?) |child| {
                    if (child.get(point)) |value| {
                        return value;
                    }
                }

                return null;
            }

            /// Time: O(k + log n) where k is number of results
            fn queryRange(self: *const Node, range: AABB, allocator: Allocator, results: *std.ArrayList(Entry)) !void {
                if (!self.boundary.intersects(range)) {
                    return;
                }

                if (self.children == null) {
                    for (self.points[0..self.points_len]) |entry| {
                        if (range.contains(entry.point)) {
                            try results.append(allocator, entry);
                        }
                    }
                    return;
                }

                for (self.children.?) |child| {
                    try child.queryRange(range, allocator, results);
                }
            }

            fn countPoints(self: *const Node) usize {
                if (self.children == null) {
                    return self.points_len;
                }

                var total: usize = 0;
                for (self.children.?) |child| {
                    total += child.countPoints();
                }
                return total;
            }

            fn height(self: *const Node) usize {
                if (self.children == null) {
                    return 1;
                }

                var max_height: usize = 0;
                for (self.children.?) |child| {
                    max_height = @max(max_height, child.height());
                }
                return max_height + 1;
            }

            fn validateInvariants(self: *const Node) !void {
                // Check boundary validity
                if (self.boundary.min.x > self.boundary.max.x or
                    self.boundary.min.y > self.boundary.max.y or
                    self.boundary.min.z > self.boundary.max.z)
                {
                    return error.InvalidBoundary;
                }

                // Check points are within boundary
                for (self.points[0..self.points_len]) |entry| {
                    if (!self.boundary.contains(entry.point)) {
                        return error.PointOutOfBounds;
                    }
                }

                // If subdivided, points should be empty
                if (self.children != null and self.points_len > 0) {
                    return error.SubdividedNodeHasPoints;
                }

                // Recursively validate children
                if (self.children) |children| {
                    for (children) |child| {
                        try child.validateInvariants();
                    }
                }
            }
        };

        pub const Iterator = struct {
            stack: std.ArrayList(*const Node),
            current_index: usize,

            /// Returns next element or null when exhausted.
            /// Time: O(1) amortized | Space: O(1)
            pub fn next(self: *Iterator, allocator: Allocator) ?Entry {
                while (self.stack.items.len > 0) {
                    const node = self.stack.items[self.stack.items.len - 1];

                    if (node.children == null) {
                        if (self.current_index < node.points_len) {
                            const entry = node.points[self.current_index];
                            self.current_index += 1;
                            return entry;
                        }
                        _ = self.stack.pop();
                        self.current_index = 0;
                        continue;
                    }

                    _ = self.stack.pop();
                    for (node.children.?) |child| {
                        self.stack.append(allocator, child) catch return null;
                    }
                }

                return null;
            }

            /// Frees iterator resources.
            /// Time: O(1) | Space: O(1)
            pub fn deinit(self: *Iterator, allocator: Allocator) void {
                self.stack.deinit(allocator);
            }
        };

        allocator: Allocator,
        root: ?*Node,
        boundary: AABB,
        node_count: usize,

        // -- Lifecycle --

        /// Initialize an empty OctTree with the specified boundary.
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator, boundary: AABB) Self {
            return .{
                .allocator = allocator,
                .root = null,
                .boundary = boundary,
                .node_count = 0,
            };
        }

        /// Free all memory used by the tree.
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.root) |root| {
                root.deinit();
                self.allocator.destroy(root);
            }
        }

        // -- Capacity --

        /// Returns the number of points in the tree.
        /// Time: O(n) | Space: O(1)
        pub fn count(self: *const Self) usize {
            if (self.root) |root| {
                return root.countPoints();
            }
            return 0;
        }

        /// Returns true if the tree is empty.
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.root == null;
        }

        /// Returns the height of the tree (number of levels).
        /// Time: O(n) | Space: O(1)
        pub fn getHeight(self: *const Self) usize {
            if (self.root) |root| {
                return root.height();
            }
            return 0;
        }

        // -- Modification --

        /// Insert a point with associated value into the tree.
        /// Time: O(log n) average, O(n) worst case | Space: O(1) amortized
        pub fn insert(self: *Self, point: Point3D, value: T) !void {
            if (!self.boundary.contains(point)) {
                return error.PointOutOfBounds;
            }

            if (self.root == null) {
                self.root = try Node.init(self.allocator, self.boundary);
                self.node_count += 1;
            }

            const entry = Entry{ .point = point, .value = value };
            _ = try self.root.?.insert(entry);
        }

        /// Remove a point from the tree.
        /// Time: O(log n) average | Space: O(1)
        pub fn remove(self: *Self, point: Point3D) bool {
            if (self.root) |root| {
                return root.remove(point);
            }
            return false;
        }

        // -- Lookup --

        /// Get the value associated with a point.
        /// Time: O(log n) average | Space: O(1)
        pub fn get(self: *const Self, point: Point3D) ?T {
            if (self.root) |root| {
                return root.get(point);
            }
            return null;
        }

        /// Check if a point exists in the tree.
        /// Time: O(log n) average | Space: O(1)
        pub fn contains(self: *const Self, point: Point3D) bool {
            return self.get(point) != null;
        }

        /// Query all points within a given AABB range.
        /// Time: O(k + log n) where k is number of results | Space: O(k)
        pub fn queryRange(self: *const Self, range: AABB) !std.ArrayList(Entry) {
            var results: std.ArrayList(Entry) = .{};
            results.clearAndFree(self.allocator);
            if (self.root) |root| {
                try root.queryRange(range, self.allocator, &results);
            }
            return results;
        }

        /// Query all points within a sphere centered at point with given radius.
        /// Time: O(k + log n) where k is number of results | Space: O(k)
        pub fn querySphere(self: *const Self, center: Point3D, radius: f64) !std.ArrayList(Entry) {
            const range = AABB{
                .min = .{ .x = center.x - radius, .y = center.y - radius, .z = center.z - radius },
                .max = .{ .x = center.x + radius, .y = center.y + radius, .z = center.z + radius },
            };

            var candidates = try self.queryRange(range);
            defer candidates.deinit(self.allocator);

            var results: std.ArrayList(Entry) = .{};
            results.clearAndFree(self.allocator);
            const radius_sq = radius * radius;

            for (candidates.items) |entry| {
                const dx = entry.point.x - center.x;
                const dy = entry.point.y - center.y;
                const dz = entry.point.z - center.z;
                const dist_sq = dx * dx + dy * dy + dz * dz;

                if (dist_sq <= radius_sq) {
                    try results.append(self.allocator, entry);
                }
            }

            return results;
        }

        // -- Iteration --

        /// Returns an iterator over all points in the tree.
        /// Time: O(1) to create, O(n) to exhaust | Space: O(log n)
        pub fn iterator(self: *const Self, allocator: Allocator) !Iterator {
            var stack: std.ArrayList(*const Node) = .{};
            stack.clearAndFree(allocator);
            if (self.root) |root| {
                try stack.append(allocator, root);
            }
            return .{
                .stack = stack,
                .current_index = 0,
            };
        }

        // -- Debug --

        /// Validate tree invariants. Returns error if invariants are violated.
        /// Time: O(n) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            if (self.root) |root| {
                try root.validateInvariants();
            }
        }

        /// Format the tree for debug printing.
        pub fn format(
            self: *const Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("OctTree{{ count={}, height={} }}", .{ self.count(), self.getHeight() });
        }
    };
}

// -- Tests --

test "OctTree - basic insert and get" {
    const allocator = testing.allocator;
    const Tree = OctTree(u32, 4);

    const boundary = Tree.AABB{
        .min = .{ .x = 0, .y = 0, .z = 0 },
        .max = .{ .x = 100, .y = 100, .z = 100 },
    };

    var tree = Tree.init(allocator, boundary);
    defer tree.deinit();

    try tree.insert(.{ .x = 10, .y = 10, .z = 10 }, 1);
    try tree.insert(.{ .x = 20, .y = 20, .z = 20 }, 2);
    try tree.insert(.{ .x = 30, .y = 30, .z = 30 }, 3);

    try testing.expectEqual(@as(?u32, 1), tree.get(.{ .x = 10, .y = 10, .z = 10 }));
    try testing.expectEqual(@as(?u32, 2), tree.get(.{ .x = 20, .y = 20, .z = 20 }));
    try testing.expectEqual(@as(?u32, 3), tree.get(.{ .x = 30, .y = 30, .z = 30 }));
    try testing.expectEqual(@as(?u32, null), tree.get(.{ .x = 40, .y = 40, .z = 40 }));

    try testing.expect(tree.contains(.{ .x = 10, .y = 10, .z = 10 }));
    try testing.expect(!tree.contains(.{ .x = 40, .y = 40, .z = 40 }));
}

test "OctTree - subdivision" {
    const allocator = testing.allocator;
    const Tree = OctTree(u32, 4);

    const boundary = Tree.AABB{
        .min = .{ .x = 0, .y = 0, .z = 0 },
        .max = .{ .x = 100, .y = 100, .z = 100 },
    };

    var tree = Tree.init(allocator, boundary);
    defer tree.deinit();

    // Insert more than capacity to trigger subdivision
    try tree.insert(.{ .x = 10, .y = 10, .z = 10 }, 1);
    try tree.insert(.{ .x = 20, .y = 20, .z = 20 }, 2);
    try tree.insert(.{ .x = 30, .y = 30, .z = 30 }, 3);
    try tree.insert(.{ .x = 40, .y = 40, .z = 40 }, 4);
    try tree.insert(.{ .x = 50, .y = 50, .z = 50 }, 5);

    try testing.expectEqual(@as(usize, 5), tree.count());
    try testing.expect(tree.getHeight() > 1);

    // All points should still be retrievable
    try testing.expectEqual(@as(?u32, 1), tree.get(.{ .x = 10, .y = 10, .z = 10 }));
    try testing.expectEqual(@as(?u32, 5), tree.get(.{ .x = 50, .y = 50, .z = 50 }));

    try tree.validate();
}

test "OctTree - remove" {
    const allocator = testing.allocator;
    const Tree = OctTree(u32, 4);

    const boundary = Tree.AABB{
        .min = .{ .x = 0, .y = 0, .z = 0 },
        .max = .{ .x = 100, .y = 100, .z = 100 },
    };

    var tree = Tree.init(allocator, boundary);
    defer tree.deinit();

    try tree.insert(.{ .x = 10, .y = 10, .z = 10 }, 1);
    try tree.insert(.{ .x = 20, .y = 20, .z = 20 }, 2);
    try tree.insert(.{ .x = 30, .y = 30, .z = 30 }, 3);

    try testing.expect(tree.remove(.{ .x = 20, .y = 20, .z = 20 }));
    try testing.expectEqual(@as(?u32, null), tree.get(.{ .x = 20, .y = 20, .z = 20 }));
    try testing.expectEqual(@as(usize, 2), tree.count());

    try testing.expect(!tree.remove(.{ .x = 40, .y = 40, .z = 40 }));
}

test "OctTree - range query" {
    const allocator = testing.allocator;
    const Tree = OctTree(u32, 4);

    const boundary = Tree.AABB{
        .min = .{ .x = 0, .y = 0, .z = 0 },
        .max = .{ .x = 100, .y = 100, .z = 100 },
    };

    var tree = Tree.init(allocator, boundary);
    defer tree.deinit();

    try tree.insert(.{ .x = 10, .y = 10, .z = 10 }, 1);
    try tree.insert(.{ .x = 20, .y = 20, .z = 20 }, 2);
    try tree.insert(.{ .x = 30, .y = 30, .z = 30 }, 3);
    try tree.insert(.{ .x = 80, .y = 80, .z = 80 }, 4);

    const query_range = Tree.AABB{
        .min = .{ .x = 0, .y = 0, .z = 0 },
        .max = .{ .x = 40, .y = 40, .z = 40 },
    };

    var results = try tree.queryRange(query_range);
    defer results.deinit(allocator);

    try testing.expectEqual(@as(usize, 3), results.items.len);
}

test "OctTree - sphere query" {
    const allocator = testing.allocator;
    const Tree = OctTree(u32, 4);

    const boundary = Tree.AABB{
        .min = .{ .x = 0, .y = 0, .z = 0 },
        .max = .{ .x = 100, .y = 100, .z = 100 },
    };

    var tree = Tree.init(allocator, boundary);
    defer tree.deinit();

    try tree.insert(.{ .x = 50, .y = 50, .z = 50 }, 1);
    try tree.insert(.{ .x = 55, .y = 55, .z = 55 }, 2);
    try tree.insert(.{ .x = 45, .y = 45, .z = 45 }, 3);
    try tree.insert(.{ .x = 90, .y = 90, .z = 90 }, 4);

    const center = Tree.Point3D{ .x = 50, .y = 50, .z = 50 };
    var results = try tree.querySphere(center, 10.0);
    defer results.deinit(allocator);

    try testing.expect(results.items.len >= 3);
    try testing.expect(results.items.len <= 4);
}

test "OctTree - iterator" {
    const allocator = testing.allocator;
    const Tree = OctTree(u32, 4);

    const boundary = Tree.AABB{
        .min = .{ .x = 0, .y = 0, .z = 0 },
        .max = .{ .x = 100, .y = 100, .z = 100 },
    };

    var tree = Tree.init(allocator, boundary);
    defer tree.deinit();

    try tree.insert(.{ .x = 10, .y = 10, .z = 10 }, 1);
    try tree.insert(.{ .x = 20, .y = 20, .z = 20 }, 2);
    try tree.insert(.{ .x = 30, .y = 30, .z = 30 }, 3);

    var iter = try tree.iterator(allocator);
    defer iter.deinit(allocator);

    var total: usize = 0;
    while (iter.next(allocator)) |_| {
        total += 1;
    }

    try testing.expectEqual(@as(usize, 3), total);
}

test "OctTree - out of bounds" {
    const allocator = testing.allocator;
    const Tree = OctTree(u32, 4);

    const boundary = Tree.AABB{
        .min = .{ .x = 0, .y = 0, .z = 0 },
        .max = .{ .x = 100, .y = 100, .z = 100 },
    };

    var tree = Tree.init(allocator, boundary);
    defer tree.deinit();

    try testing.expectError(error.PointOutOfBounds, tree.insert(.{ .x = 150, .y = 150, .z = 150 }, 1));
}

test "OctTree - stress test" {
    const allocator = testing.allocator;
    const Tree = OctTree(usize, 4);

    const boundary = Tree.AABB{
        .min = .{ .x = 0, .y = 0, .z = 0 },
        .max = .{ .x = 1000, .y = 1000, .z = 1000 },
    };

    var tree = Tree.init(allocator, boundary);
    defer tree.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const n = 1000;
    for (0..n) |i| {
        const x = random.float(f64) * 1000.0;
        const y = random.float(f64) * 1000.0;
        const z = random.float(f64) * 1000.0;
        try tree.insert(.{ .x = x, .y = y, .z = z }, i);
    }

    try testing.expectEqual(@as(usize, n), tree.count());
    try tree.validate();

    // Query a region
    const query_range = Tree.AABB{
        .min = .{ .x = 0, .y = 0, .z = 0 },
        .max = .{ .x = 500, .y = 500, .z = 500 },
    };
    var results = try tree.queryRange(query_range);
    defer results.deinit(allocator);

    try testing.expect(results.items.len > 0);
    try testing.expect(results.items.len < n);
}

test "OctTree - empty tree" {
    const allocator = testing.allocator;
    const Tree = OctTree(u32, 4);

    const boundary = Tree.AABB{
        .min = .{ .x = 0, .y = 0, .z = 0 },
        .max = .{ .x = 100, .y = 100, .z = 100 },
    };

    var tree = Tree.init(allocator, boundary);
    defer tree.deinit();

    try testing.expect(tree.isEmpty());
    try testing.expectEqual(@as(usize, 0), tree.count());
    try testing.expectEqual(@as(usize, 0), tree.getHeight());
    try testing.expectEqual(@as(?u32, null), tree.get(.{ .x = 50, .y = 50, .z = 50 }));

    try tree.validate();
}
