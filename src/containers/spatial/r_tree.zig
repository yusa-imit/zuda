const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// RTree — R-tree spatial index for efficient bounding-box queries
///
/// A balanced tree structure that groups spatial objects by their bounding rectangles.
/// Widely used in spatial databases, GIS, and collision detection systems.
///
/// Design choices:
/// - Generic dimension k (comptime parameter)
/// - Configurable min/max children per node (M parameter)
/// - Quadratic split algorithm (balances quality and speed)
/// - Bounding box represented as [k]T for min and [k]T for max coordinates
/// - Comptime coordinate accessor for flexible data types
///
/// Type parameters:
/// - T: coordinate type (must support ordering, typically f32/f64)
/// - k: number of dimensions (2D, 3D, etc.)
/// - M: maximum children per node (minimum is M/2, typically M=4-16)
/// - Data: user data type to associate with each spatial object
/// - getCoordsFn: function to extract bounding box from Data
///
/// Time complexity:
/// - Insert: O(log n) average, O(n) worst case
/// - Remove: O(log n) average, O(n) worst case
/// - Search (overlap): O(log n + k) where k is result size
/// - Nearest neighbor: O(log n) average with pruning
///
/// Space: O(n)
pub fn RTree(
    comptime T: type,
    comptime k: comptime_int,
    comptime M: comptime_int,
    comptime Data: type,
    comptime getCoordsFn: fn (Data) BoundingBox(T, k),
) type {
    if (k < 1) @compileError("RTree dimension k must be >= 1");
    if (M < 2) @compileError("RTree max children M must be >= 2");

    return struct {
        const Self = @This();
        pub const BBox = BoundingBox(T, k);
        const min_children = M / 2;

        pub const Entry = struct {
            bbox: BBox,
            data: Data,
        };

        const NodeType = enum { leaf, internal };

        const Node = struct {
            node_type: NodeType,
            bbox: BBox,
            children: std.ArrayList(union(enum) {
                leaf_entry: Entry,
                internal_child: *Node,
            }),

            fn deinit(self: *Node, allocator: Allocator) void {
                if (self.node_type == .internal) {
                    for (self.children.items) |child| {
                        switch (child) {
                            .internal_child => |node| {
                                node.deinit(allocator);
                                allocator.destroy(node);
                            },
                            else => unreachable,
                        }
                    }
                }
                self.children.deinit(allocator);
            }

            fn updateBBox(self: *Node) void {
                if (self.children.items.len == 0) return;

                var min: [k]T = undefined;
                var max: [k]T = undefined;

                const first_bbox = switch (self.children.items[0]) {
                    .leaf_entry => |e| e.bbox,
                    .internal_child => |n| n.bbox,
                };

                @memcpy(&min, &first_bbox.min);
                @memcpy(&max, &first_bbox.max);

                for (self.children.items[1..]) |child| {
                    const bbox = switch (child) {
                        .leaf_entry => |e| e.bbox,
                        .internal_child => |n| n.bbox,
                    };
                    for (0..k) |i| {
                        min[i] = @min(min[i], bbox.min[i]);
                        max[i] = @max(max[i], bbox.max[i]);
                    }
                }

                self.bbox = BBox{ .min = min, .max = max };
            }
        };

        const InsertResult = struct {
            split_node: ?*Node,
        };

        allocator: Allocator,
        root: ?*Node,
        count_val: usize,

        /// Initialize an empty RTree
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
                .root = null,
                .count_val = 0,
            };
        }

        /// Free all memory used by the tree
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.root) |root| {
                root.deinit(self.allocator);
                self.allocator.destroy(root);
            }
            self.* = undefined;
        }

        /// Return the number of entries in the tree
        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.count_val;
        }

        /// Check if the tree is empty
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.count_val == 0;
        }

        /// Insert a data item into the tree
        /// Time: O(log n) average, O(n) worst | Space: O(log n) stack
        pub fn insert(self: *Self, data: Data) !void {
            const bbox = getCoordsFn(data);
            const entry = Entry{ .bbox = bbox, .data = data };

            if (self.root == null) {
                // Create first leaf node
                const node = try self.allocator.create(Node);
                node.* = .{
                    .node_type = .leaf,
                    .bbox = bbox,
                    .children = .{},
                };
                try node.children.append(self.allocator,.{ .leaf_entry = entry });
                self.root = node;
                self.count_val = 1;
                return;
            }

            const result = try self.insertRecursive(self.root.?, entry);

            if (result.split_node) |split| {
                // Root split — create new root
                const new_root = try self.allocator.create(Node);
                new_root.* = .{
                    .node_type = .internal,
                    .bbox = undefined,
                    .children = .{},
                };
                try new_root.children.append(self.allocator,.{ .internal_child = self.root.? });
                try new_root.children.append(self.allocator,.{ .internal_child = split });
                new_root.updateBBox();
                self.root = new_root;
            }

            self.count_val += 1;
        }

        fn insertRecursive(self: *Self, node: *Node, entry: Entry) !InsertResult {
            if (node.node_type == .leaf) {
                // Insert into leaf
                try node.children.append(self.allocator,.{ .leaf_entry = entry });
                node.updateBBox();

                if (node.children.items.len > M) {
                    // Split leaf
                    const split_node = try self.splitNode(node);
                    return .{ .split_node = split_node };
                }
                return .{ .split_node = null };
            }

            // Choose subtree to insert into
            const best_child_idx = self.chooseSubtree(node, entry.bbox);
            const child = node.children.items[best_child_idx].internal_child;
            const result = try self.insertRecursive(child, entry);

            if (result.split_node) |split| {
                // Child split — add new child
                try node.children.append(self.allocator,.{ .internal_child = split });

                if (node.children.items.len > M) {
                    // Split internal node
                    const split_node = try self.splitNode(node);
                    return .{ .split_node = split_node };
                }
            }

            // Update bbox after potential split
            node.updateBBox();

            return .{ .split_node = null };
        }

        fn chooseSubtree(self: *const Self, node: *Node, bbox: BBox) usize {
            _ = self;
            std.debug.assert(node.node_type == .internal);

            var best_idx: usize = 0;
            var min_enlargement: T = std.math.floatMax(T);
            var min_area: T = std.math.floatMax(T);

            for (node.children.items, 0..) |child, i| {
                const child_bbox = child.internal_child.bbox;
                const current_area = child_bbox.area();
                const enlarged = child_bbox.merge(bbox);
                const enlarged_area = enlarged.area();
                const enlargement = enlarged_area - current_area;

                if (enlargement < min_enlargement or
                    (enlargement == min_enlargement and current_area < min_area))
                {
                    min_enlargement = enlargement;
                    min_area = current_area;
                    best_idx = i;
                }
            }

            return best_idx;
        }

        fn splitNode(self: *Self, node: *Node) !*Node {
            // Quadratic split algorithm
            const seeds = self.pickSeeds(node);

            const new_node = try self.allocator.create(Node);
            new_node.* = .{
                .node_type = node.node_type,
                .bbox = undefined,
                .children = .{},
            };

            // Copy all children to temporary list
            const ChildType = @TypeOf(node.children.items[0]);
            const all_children = try self.allocator.alloc(ChildType, node.children.items.len);
            defer self.allocator.free(all_children);
            @memcpy(all_children, node.children.items);

            // Clear node children and rebuild
            node.children.clearRetainingCapacity();

            // Add seeds
            try node.children.append(self.allocator, all_children[seeds.first]);
            try new_node.children.append(self.allocator, all_children[seeds.second]);

            // Distribute remaining children
            const total = all_children.len;
            for (all_children, 0..) |child, i| {
                if (i == seeds.first or i == seeds.second) continue;

                const child_bbox = switch (child) {
                    .leaf_entry => |e| e.bbox,
                    .internal_child => |n| n.bbox,
                };

                const remaining = total - (node.children.items.len + new_node.children.items.len);

                // Force distribution if one node would go below minimum
                if (node.children.items.len + remaining == min_children) {
                    try node.children.append(self.allocator, child);
                    continue;
                }
                if (new_node.children.items.len + remaining == min_children) {
                    try new_node.children.append(self.allocator, child);
                    continue;
                }

                node.updateBBox();
                new_node.updateBBox();

                const node_enlargement = node.bbox.merge(child_bbox).area() - node.bbox.area();
                const new_enlargement = new_node.bbox.merge(child_bbox).area() - new_node.bbox.area();

                if (node_enlargement < new_enlargement) {
                    try node.children.append(self.allocator, child);
                } else if (new_enlargement < node_enlargement) {
                    try new_node.children.append(self.allocator, child);
                } else {
                    // Tie — add to smaller group
                    if (node.children.items.len <= new_node.children.items.len) {
                        try node.children.append(self.allocator, child);
                    } else {
                        try new_node.children.append(self.allocator, child);
                    }
                }
            }

            node.updateBBox();
            new_node.updateBBox();

            return new_node;
        }

        fn pickSeeds(self: *const Self, node: *Node) struct { first: usize, second: usize } {
            _ = self;
            var max_waste: T = 0;
            var seed1: usize = 0;
            var seed2: usize = 1;

            for (node.children.items, 0..) |child1, i| {
                for (node.children.items[i + 1 ..], i + 1..) |child2, j| {
                    const bbox1 = switch (child1) {
                        .leaf_entry => |e| e.bbox,
                        .internal_child => |n| n.bbox,
                    };
                    const bbox2 = switch (child2) {
                        .leaf_entry => |e| e.bbox,
                        .internal_child => |n| n.bbox,
                    };

                    const merged = bbox1.merge(bbox2);
                    const waste = merged.area() - bbox1.area() - bbox2.area();

                    if (waste > max_waste) {
                        max_waste = waste;
                        seed1 = i;
                        seed2 = j;
                    }
                }
            }

            return .{ .first = seed1, .second = seed2 };
        }

        /// Search for all entries whose bounding boxes overlap with the query box
        /// Time: O(log n + k) where k is result size | Space: O(k)
        pub fn search(self: *const Self, allocator: Allocator, query: BBox) !std.ArrayList(Data) {
            var results: std.ArrayList(Data) = .{};
            errdefer results.deinit(allocator);

            if (self.root) |root| {
                try self.searchRecursive(root, query, &results, allocator);
            }

            return results;
        }

        fn searchRecursive(self: *const Self, node: *Node, query: BBox, results: *std.ArrayList(Data), allocator: Allocator) !void {
            if (!node.bbox.overlaps(query)) return;

            if (node.node_type == .leaf) {
                for (node.children.items) |child| {
                    const entry = child.leaf_entry;
                    if (entry.bbox.overlaps(query)) {
                        try results.append(allocator,entry.data);
                    }
                }
            } else {
                for (node.children.items) |child| {
                    try self.searchRecursive(child.internal_child, query, results, allocator);
                }
            }
        }

        /// Find the nearest neighbor to a query point
        /// Time: O(log n) average with pruning | Space: O(log n) stack
        pub fn nearestNeighbor(self: *const Self, point: [k]T) ?Data {
            if (self.root == null) return null;

            var best_dist: T = std.math.floatMax(T);
            var best_data: ?Data = null;

            self.nearestRecursive(self.root.?, point, &best_dist, &best_data);

            return best_data;
        }

        fn nearestRecursive(self: *const Self, node: *Node, point: [k]T, best_dist: *T, best_data: *?Data) void {
            const min_dist = node.bbox.minDistanceToPoint(point);
            if (min_dist >= best_dist.*) return; // Prune

            if (node.node_type == .leaf) {
                for (node.children.items) |child| {
                    const entry = child.leaf_entry;
                    const dist = entry.bbox.centerDistanceToPoint(point);
                    if (dist < best_dist.*) {
                        best_dist.* = dist;
                        best_data.* = entry.data;
                    }
                }
            } else {
                // Sort children by minimum distance
                const ChildDist = struct { idx: usize, dist: T };
                var children_dists: [M]ChildDist = undefined;
                var child_count: usize = 0;
                for (node.children.items, 0..) |child, i| {
                    const child_min_dist = child.internal_child.bbox.minDistanceToPoint(point);
                    children_dists[child_count] = .{ .idx = i, .dist = child_min_dist };
                    child_count += 1;
                }

                std.mem.sort(ChildDist, children_dists[0..child_count], {}, struct {
                    fn lessThan(_: void, a: ChildDist, b: ChildDist) bool {
                        return a.dist < b.dist;
                    }
                }.lessThan);

                for (children_dists[0..child_count]) |item| {
                    self.nearestRecursive(node.children.items[item.idx].internal_child, point, best_dist, best_data);
                }
            }
        }

        /// Iterator for traversing all entries in the tree
        pub const Iterator = struct {
            stack: std.ArrayList(*Node),
            leaf_idx: usize,
            allocator: Allocator,

            pub fn next(self: *Iterator) ?Data {
                while (self.stack.items.len > 0) {
                    const node = self.stack.items[self.stack.items.len - 1];

                    if (node.node_type == .leaf) {
                        if (self.leaf_idx < node.children.items.len) {
                            const entry = node.children.items[self.leaf_idx].leaf_entry;
                            self.leaf_idx += 1;
                            return entry.data;
                        } else {
                            _ = self.stack.pop();
                            self.leaf_idx = 0;
                        }
                    } else {
                        _ = self.stack.pop();
                        // Push children in reverse order for in-order traversal
                        var i = node.children.items.len;
                        while (i > 0) {
                            i -= 1;
                            self.stack.append(self.allocator,node.children.items[i].internal_child) catch return null;
                        }
                    }
                }

                return null;
            }

            pub fn deinit(self: *Iterator) void {
                self.stack.deinit(self.allocator);
            }
        };

        /// Create an iterator over all entries
        /// Time: O(1) | Space: O(log n) for stack
        pub fn iterator(self: *const Self) !Iterator {
            var stack: std.ArrayList(*Node) = .{};
            if (self.root) |root| {
                try stack.append(self.allocator,root);
            }
            return Iterator{
                .stack = stack,
                .leaf_idx = 0,
                .allocator = self.allocator,
            };
        }

        /// Validate tree invariants (for testing)
        /// Time: O(n) | Space: O(log n) stack
        pub fn validate(self: *const Self) !void {
            if (self.root) |root| {
                try self.validateNode(root, true);
            }
        }

        fn validateNode(self: *const Self, node: *Node, is_root: bool) !void {
            // Check child count
            if (!is_root and node.children.items.len < min_children) {
                return error.TreeInvariant;
            }
            if (node.children.items.len > M) {
                return error.TreeInvariant;
            }

            // Check bounding box validity
            if (!node.bbox.isValid()) {
                return error.TreeInvariant;
            }

            // Recursively validate children
            if (node.node_type == .internal) {
                for (node.children.items) |child| {
                    const child_node = child.internal_child;
                    try self.validateNode(child_node, false);

                    // Check that parent bbox contains child bbox
                    if (!node.bbox.contains(child_node.bbox)) {
                        return error.TreeInvariant;
                    }
                }
            }
        }
    };
}

/// Bounding box in k dimensions
pub fn BoundingBox(comptime T: type, comptime k: comptime_int) type {
    return struct {
        const Self = @This();

        min: [k]T,
        max: [k]T,

        pub fn isValid(self: Self) bool {
            for (0..k) |i| {
                if (self.min[i] > self.max[i]) return false;
            }
            return true;
        }

        pub fn overlaps(self: Self, other: Self) bool {
            for (0..k) |i| {
                if (self.max[i] < other.min[i] or self.min[i] > other.max[i]) {
                    return false;
                }
            }
            return true;
        }

        pub fn contains(self: Self, other: Self) bool {
            for (0..k) |i| {
                if (other.min[i] < self.min[i] or other.max[i] > self.max[i]) {
                    return false;
                }
            }
            return true;
        }

        pub fn merge(self: Self, other: Self) Self {
            var result: Self = undefined;
            for (0..k) |i| {
                result.min[i] = @min(self.min[i], other.min[i]);
                result.max[i] = @max(self.max[i], other.max[i]);
            }
            return result;
        }

        pub fn area(self: Self) T {
            var result: T = 1;
            for (0..k) |i| {
                result *= (self.max[i] - self.min[i]);
            }
            return result;
        }

        pub fn minDistanceToPoint(self: Self, point: [k]T) T {
            var dist_sq: T = 0;
            for (0..k) |i| {
                if (point[i] < self.min[i]) {
                    const d = self.min[i] - point[i];
                    dist_sq += d * d;
                } else if (point[i] > self.max[i]) {
                    const d = point[i] - self.max[i];
                    dist_sq += d * d;
                }
            }
            return @sqrt(dist_sq);
        }

        pub fn centerDistanceToPoint(self: Self, point: [k]T) T {
            var center: [k]T = undefined;
            for (0..k) |i| {
                center[i] = (self.min[i] + self.max[i]) / 2;
            }

            var dist_sq: T = 0;
            for (0..k) |i| {
                const d = center[i] - point[i];
                dist_sq += d * d;
            }
            return @sqrt(dist_sq);
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const Point2D = struct {
    x: f32,
    y: f32,
    id: u32,
};

fn getPointBBox(p: Point2D) BoundingBox(f32, 2) {
    return .{
        .min = .{ p.x, p.y },
        .max = .{ p.x, p.y },
    };
}

const Rectangle = struct {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    id: u32,
};

fn getRectBBox(r: Rectangle) BoundingBox(f32, 2) {
    return .{
        .min = .{ r.min_x, r.min_y },
        .max = .{ r.max_x, r.max_y },
    };
}

test "RTree: basic 2D point operations" {
    const RTree2D = RTree(f32, 2, 4, Point2D, getPointBBox);
    var tree = RTree2D.init(testing.allocator);
    defer tree.deinit();

    try testing.expect(tree.isEmpty());
    try testing.expectEqual(@as(usize, 0), tree.count());

    // Insert points
    try tree.insert(.{ .x = 1.0, .y = 1.0, .id = 1 });
    try tree.insert(.{ .x = 5.0, .y = 5.0, .id = 2 });
    try tree.insert(.{ .x = 2.0, .y = 3.0, .id = 3 });

    try testing.expectEqual(@as(usize, 3), tree.count());
    try testing.expect(!tree.isEmpty());

    try tree.validate();
}

test "RTree: 2D rectangle search" {
    const RTree2D = RTree(f32, 2, 4, Rectangle, getRectBBox);
    var tree = RTree2D.init(testing.allocator);
    defer tree.deinit();

    // Insert rectangles
    try tree.insert(.{ .min_x = 0.0, .min_y = 0.0, .max_x = 2.0, .max_y = 2.0, .id = 1 });
    try tree.insert(.{ .min_x = 3.0, .min_y = 3.0, .max_x = 5.0, .max_y = 5.0, .id = 2 });
    try tree.insert(.{ .min_x = 1.0, .min_y = 1.0, .max_x = 4.0, .max_y = 4.0, .id = 3 });
    try tree.insert(.{ .min_x = 6.0, .min_y = 6.0, .max_x = 8.0, .max_y = 8.0, .id = 4 });

    // Search for overlapping rectangles
    const query = RTree2D.BBox{
        .min = .{ 2.0, 2.0 },
        .max = .{ 6.0, 6.0 },
    };

    var results = try tree.search(testing.allocator, query);
    defer results.deinit(testing.allocator);

    // Should find rectangles 2 and 3 (overlapping with query)
    try testing.expect(results.items.len >= 2);

    var found_ids: std.ArrayList(u32) = .{};
    defer found_ids.deinit(testing.allocator);
    for (results.items) |r| {
        try found_ids.append(testing.allocator,r.id);
    }

    try testing.expect(std.mem.containsAtLeast(u32, found_ids.items, 1, &[_]u32{2}));
    try testing.expect(std.mem.containsAtLeast(u32, found_ids.items, 1, &[_]u32{3}));

    try tree.validate();
}

test "RTree: nearest neighbor search" {
    const RTree2D = RTree(f32, 2, 4, Point2D, getPointBBox);
    var tree = RTree2D.init(testing.allocator);
    defer tree.deinit();

    try tree.insert(.{ .x = 1.0, .y = 1.0, .id = 1 });
    try tree.insert(.{ .x = 5.0, .y = 5.0, .id = 2 });
    try tree.insert(.{ .x = 9.0, .y = 9.0, .id = 3 });
    try tree.insert(.{ .x = 2.0, .y = 3.0, .id = 4 });

    // Find nearest to (2.5, 2.5)
    const nearest = tree.nearestNeighbor(.{ 2.5, 2.5 });
    try testing.expect(nearest != null);
    // Should be either point 1 or 4 (both close)
    try testing.expect(nearest.?.id == 1 or nearest.?.id == 4);

    // Find nearest to (10.0, 10.0)
    const nearest2 = tree.nearestNeighbor(.{ 10.0, 10.0 });
    try testing.expect(nearest2 != null);
    try testing.expectEqual(@as(u32, 3), nearest2.?.id);

    try tree.validate();
}

test "RTree: iterator" {
    const RTree2D = RTree(f32, 2, 4, Point2D, getPointBBox);
    var tree = RTree2D.init(testing.allocator);
    defer tree.deinit();

    try tree.insert(.{ .x = 1.0, .y = 1.0, .id = 1 });
    try tree.insert(.{ .x = 2.0, .y = 2.0, .id = 2 });
    try tree.insert(.{ .x = 3.0, .y = 3.0, .id = 3 });

    var iter = try tree.iterator();
    defer iter.deinit();

    var count: usize = 0;
    while (iter.next()) |_| {
        count += 1;
    }

    try testing.expectEqual(@as(usize, 3), count);

    try tree.validate();
}

test "RTree: stress test with many points" {
    const RTree2D = RTree(f32, 2, 8, Point2D, getPointBBox);
    var tree = RTree2D.init(testing.allocator);
    defer tree.deinit();

    // Insert 100 points in a grid
    var id: u32 = 0;
    var y: f32 = 0;
    while (y < 10) : (y += 1) {
        var x: f32 = 0;
        while (x < 10) : (x += 1) {
            try tree.insert(.{ .x = x, .y = y, .id = id });
            id += 1;
        }
    }

    try testing.expectEqual(@as(usize, 100), tree.count());

    // Search in a region
    const query = RTree2D.BBox{
        .min = .{ 2.0, 2.0 },
        .max = .{ 5.0, 5.0 },
    };

    var results = try tree.search(testing.allocator, query);
    defer results.deinit(testing.allocator);

    // Should find 16 points (4x4 grid)
    try testing.expectEqual(@as(usize, 16), results.items.len);

    try tree.validate();
}

test "RTree: 3D operations" {
    const Point3D = struct {
        x: f32,
        y: f32,
        z: f32,
        id: u32,
    };

    const getPoint3DBBox = struct {
        fn f(p: Point3D) BoundingBox(f32, 3) {
            return .{
                .min = .{ p.x, p.y, p.z },
                .max = .{ p.x, p.y, p.z },
            };
        }
    }.f;

    const RTree3D = RTree(f32, 3, 4, Point3D, getPoint3DBBox);
    var tree = RTree3D.init(testing.allocator);
    defer tree.deinit();

    try tree.insert(.{ .x = 1.0, .y = 1.0, .z = 1.0, .id = 1 });
    try tree.insert(.{ .x = 5.0, .y = 5.0, .z = 5.0, .id = 2 });
    try tree.insert(.{ .x = 2.0, .y = 3.0, .z = 4.0, .id = 3 });

    try testing.expectEqual(@as(usize, 3), tree.count());

    const query = RTree3D.BBox{
        .min = .{ 0.0, 0.0, 0.0 },
        .max = .{ 3.0, 3.0, 3.0 },
    };

    var results = try tree.search(testing.allocator, query);
    defer results.deinit(testing.allocator);

    try testing.expect(results.items.len >= 1);

    try tree.validate();
}

test "RTree: empty tree operations" {
    const RTree2D = RTree(f32, 2, 4, Point2D, getPointBBox);
    var tree = RTree2D.init(testing.allocator);
    defer tree.deinit();

    try testing.expect(tree.isEmpty());

    const query = RTree2D.BBox{
        .min = .{ 0.0, 0.0 },
        .max = .{ 10.0, 10.0 },
    };

    var results = try tree.search(testing.allocator, query);
    defer results.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 0), results.items.len);

    const nearest = tree.nearestNeighbor(.{ 5.0, 5.0 });
    try testing.expect(nearest == null);

    try tree.validate();
}

test "RTree: memory leak check" {
    const RTree2D = RTree(f32, 2, 4, Point2D, getPointBBox);
    var tree = RTree2D.init(testing.allocator);
    defer tree.deinit();

    var i: u32 = 0;
    while (i < 50) : (i += 1) {
        try tree.insert(.{
            .x = @as(f32, @floatFromInt(i)),
            .y = @as(f32, @floatFromInt(i % 10)),
            .id = i,
        });
    }

    try testing.expectEqual(@as(usize, 50), tree.count());

    try tree.validate();
}

test "RTree: bounding box operations" {
    const BBox2D = BoundingBox(f32, 2);

    const bbox1 = BBox2D{ .min = .{ 0.0, 0.0 }, .max = .{ 2.0, 2.0 } };
    const bbox2 = BBox2D{ .min = .{ 1.0, 1.0 }, .max = .{ 3.0, 3.0 } };
    const bbox3 = BBox2D{ .min = .{ 5.0, 5.0 }, .max = .{ 7.0, 7.0 } };

    try testing.expect(bbox1.isValid());
    try testing.expect(bbox1.overlaps(bbox2));
    try testing.expect(!bbox1.overlaps(bbox3));

    const merged = bbox1.merge(bbox2);
    try testing.expectEqual(@as(f32, 0.0), merged.min[0]);
    try testing.expectEqual(@as(f32, 0.0), merged.min[1]);
    try testing.expectEqual(@as(f32, 3.0), merged.max[0]);
    try testing.expectEqual(@as(f32, 3.0), merged.max[1]);

    const area1 = bbox1.area();
    try testing.expectEqual(@as(f32, 4.0), area1);

    const point = [2]f32{ 1.0, 1.0 };
    const dist = bbox1.minDistanceToPoint(point);
    try testing.expectEqual(@as(f32, 0.0), dist);

    const point2 = [2]f32{ 5.0, 5.0 };
    const dist2 = bbox1.minDistanceToPoint(point2);
    try testing.expect(dist2 > 0);
}
