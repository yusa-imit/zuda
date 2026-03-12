const std = @import("std");
const Allocator = std.mem.Allocator;

/// Disjoint Set (Union-Find) data structure with path compression and union by rank.
///
/// Maintains a collection of disjoint sets, supporting efficient union and find operations.
/// Uses two key optimizations:
/// - Path compression: flatten tree structure during find operations
/// - Union by rank: attach smaller tree to root of larger tree
///
/// Time complexity:
/// - makeSet: O(1)
/// - find: O(α(n)) amortized (α is inverse Ackermann, effectively constant)
/// - union: O(α(n)) amortized
/// - connected: O(α(n)) amortized
///
/// Space complexity: O(n) where n is number of elements
///
/// Use cases:
/// - Kruskal's MST algorithm
/// - Connected components in graphs
/// - Image segmentation
/// - Network connectivity
/// - Detecting cycles in undirected graphs
pub fn DisjointSet(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Internal node structure
        const Node = struct {
            parent: usize,
            rank: usize,
            data: T,
        };

        allocator: Allocator,
        nodes: std.ArrayList(Node),
        map: std.AutoHashMap(T, usize),

        /// Initialize an empty disjoint set.
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
                .nodes = .{},
                .map = std.AutoHashMap(T, usize).init(allocator),
            };
        }

        /// Free all allocated memory.
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.nodes.deinit(self.allocator);
            self.map.deinit();
        }

        /// Create a new set containing a single element.
        /// Returns error if element already exists.
        /// Time: O(1) amortized | Space: O(1) amortized
        pub fn makeSet(self: *Self, data: T) !void {
            if (self.map.contains(data)) {
                return error.AlreadyExists;
            }

            const index = self.nodes.items.len;
            try self.nodes.append(self.allocator, .{
                .parent = index, // initially, element is its own parent
                .rank = 0,
                .data = data,
            });
            try self.map.put(data, index);
        }

        /// Find the representative (root) of the set containing the element.
        /// Applies path compression: all nodes on the path to root point directly to root.
        /// Time: O(α(n)) amortized | Space: O(1)
        pub fn find(self: *Self, data: T) !usize {
            const index = self.map.get(data) orelse return error.NotFound;
            return self.findByIndex(index);
        }

        fn findByIndex(self: *Self, index: usize) usize {
            if (self.nodes.items[index].parent != index) {
                // Path compression: make parent point directly to root
                self.nodes.items[index].parent = self.findByIndex(self.nodes.items[index].parent);
            }
            return self.nodes.items[index].parent;
        }

        /// Union two sets containing the given elements.
        /// Uses union by rank: attach smaller tree to root of larger tree.
        /// Returns true if sets were merged, false if already in same set.
        /// Time: O(α(n)) amortized | Space: O(1)
        pub fn unite(self: *Self, a: T, b: T) !bool {
            const root_a = try self.find(a);
            const root_b = try self.find(b);

            if (root_a == root_b) {
                return false; // already in same set
            }

            // Union by rank: attach smaller tree to larger
            if (self.nodes.items[root_a].rank < self.nodes.items[root_b].rank) {
                self.nodes.items[root_a].parent = root_b;
            } else if (self.nodes.items[root_a].rank > self.nodes.items[root_b].rank) {
                self.nodes.items[root_b].parent = root_a;
            } else {
                // Equal rank: arbitrary choice, increment rank of new root
                self.nodes.items[root_b].parent = root_a;
                self.nodes.items[root_a].rank += 1;
            }

            return true;
        }

        /// Check if two elements are in the same set.
        /// Time: O(α(n)) amortized | Space: O(1)
        pub fn connected(self: *Self, a: T, b: T) !bool {
            const root_a = try self.find(a);
            const root_b = try self.find(b);
            return root_a == root_b;
        }

        /// Get the number of elements in the disjoint set.
        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.nodes.items.len;
        }

        /// Get the number of disjoint sets.
        /// Time: O(n) | Space: O(1)
        pub fn numSets(self: *Self) usize {
            var sets: usize = 0;
            for (self.nodes.items, 0..) |node, i| {
                if (node.parent == i) {
                    sets += 1;
                }
            }
            return sets;
        }

        /// Validate internal invariants.
        /// - All roots have parent pointing to themselves
        /// - All ranks are non-negative
        /// - Map size equals node count
        /// Time: O(n) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            if (self.map.count() != self.nodes.items.len) {
                return error.InvalidState;
            }

            for (self.nodes.items, 0..) |node, i| {
                // Root nodes have parent = self
                if (node.parent == i) {
                    // Root node - this is valid
                } else {
                    // Non-root must have valid parent index
                    if (node.parent >= self.nodes.items.len) {
                        return error.InvalidParent;
                    }
                }
            }
        }
    };
}

// Tests
test "DisjointSet: basic operations" {
    var ds = DisjointSet(u32).init(std.testing.allocator);
    defer ds.deinit();

    try ds.makeSet(1);
    try ds.makeSet(2);
    try ds.makeSet(3);

    try std.testing.expectEqual(@as(usize, 3), ds.count());
    try std.testing.expectEqual(@as(usize, 3), ds.numSets());

    // Initially, all in separate sets
    try std.testing.expect(!try ds.connected(1, 2));
    try std.testing.expect(!try ds.connected(2, 3));
    try std.testing.expect(!try ds.connected(1, 3));
}

test "DisjointSet: union operations" {
    var ds = DisjointSet(u32).init(std.testing.allocator);
    defer ds.deinit();

    try ds.makeSet(1);
    try ds.makeSet(2);
    try ds.makeSet(3);
    try ds.makeSet(4);

    // Unite 1 and 2
    try std.testing.expect(try ds.unite(1, 2));
    try std.testing.expect(try ds.connected(1, 2));
    try std.testing.expectEqual(@as(usize, 3), ds.numSets());

    // Unite 3 and 4
    try std.testing.expect(try ds.unite(3, 4));
    try std.testing.expect(try ds.connected(3, 4));
    try std.testing.expectEqual(@as(usize, 2), ds.numSets());

    // Unite sets {1,2} and {3,4}
    try std.testing.expect(try ds.unite(2, 3));
    try std.testing.expect(try ds.connected(1, 4));
    try std.testing.expectEqual(@as(usize, 1), ds.numSets());
}

test "DisjointSet: duplicate union" {
    var ds = DisjointSet(u32).init(std.testing.allocator);
    defer ds.deinit();

    try ds.makeSet(1);
    try ds.makeSet(2);

    try std.testing.expect(try ds.unite(1, 2));
    try std.testing.expect(!try ds.unite(1, 2)); // already united
}

test "DisjointSet: path compression" {
    var ds = DisjointSet(u32).init(std.testing.allocator);
    defer ds.deinit();

    // Create a chain: 1-2-3-4-5
    for (1..6) |i| {
        try ds.makeSet(@intCast(i));
    }

    try std.testing.expect(try ds.unite(1, 2));
    try std.testing.expect(try ds.unite(2, 3));
    try std.testing.expect(try ds.unite(3, 4));
    try std.testing.expect(try ds.unite(4, 5));

    // All should be connected
    try std.testing.expect(try ds.connected(1, 5));
    try std.testing.expect(try ds.connected(2, 4));
    try std.testing.expect(try ds.connected(1, 3));

    // After find operations, path compression should flatten structure
    const root = try ds.find(1);
    try std.testing.expectEqual(root, try ds.find(5));
}

test "DisjointSet: not found errors" {
    var ds = DisjointSet(u32).init(std.testing.allocator);
    defer ds.deinit();

    try ds.makeSet(1);

    try std.testing.expectError(error.NotFound, ds.find(2));
    try std.testing.expectError(error.NotFound, ds.connected(1, 2));
    try std.testing.expectError(error.NotFound, ds.unite(1, 2));
}

test "DisjointSet: already exists" {
    var ds = DisjointSet(u32).init(std.testing.allocator);
    defer ds.deinit();

    try ds.makeSet(1);
    try std.testing.expectError(error.AlreadyExists, ds.makeSet(1));
}

test "DisjointSet: i64 type" {
    var ds = DisjointSet(i64).init(std.testing.allocator);
    defer ds.deinit();

    try ds.makeSet(-100);
    try ds.makeSet(0);
    try ds.makeSet(100);

    try std.testing.expect(try ds.unite(-100, 0));
    try std.testing.expect(try ds.connected(-100, 0));
    try std.testing.expect(!try ds.connected(-100, 100));
}

test "DisjointSet: validate invariants" {
    var ds = DisjointSet(u32).init(std.testing.allocator);
    defer ds.deinit();

    try ds.makeSet(1);
    try ds.makeSet(2);
    try ds.makeSet(3);

    try ds.validate();

    try std.testing.expect(try ds.unite(1, 2));
    try ds.validate();

    try std.testing.expect(try ds.unite(2, 3));
    try ds.validate();
}

test "DisjointSet: large scale" {
    var ds = DisjointSet(u32).init(std.testing.allocator);
    defer ds.deinit();

    const n = 1000;

    // Create 1000 elements
    for (0..n) |i| {
        try ds.makeSet(@intCast(i));
    }

    try std.testing.expectEqual(@as(usize, n), ds.count());
    try std.testing.expectEqual(@as(usize, n), ds.numSets());

    // Unite pairs: 0-1, 2-3, 4-5, ...
    for (0..n / 2) |i| {
        try std.testing.expect(try ds.unite(@intCast(i * 2), @intCast(i * 2 + 1)));
    }

    try std.testing.expectEqual(@as(usize, n / 2), ds.numSets());

    // Verify connectivity
    for (0..n / 2) |i| {
        try std.testing.expect(try ds.connected(@intCast(i * 2), @intCast(i * 2 + 1)));
    }

    try ds.validate();
}

test "DisjointSet: union by rank" {
    var ds = DisjointSet(u32).init(std.testing.allocator);
    defer ds.deinit();

    // Create tree with controlled ranks
    for (0..8) |i| {
        try ds.makeSet(@intCast(i));
    }

    // Build two balanced trees
    // Tree A: 0-1, 2-3, then (0,1)-(2,3)
    try std.testing.expect(try ds.unite(0, 1));
    try std.testing.expect(try ds.unite(2, 3));
    try std.testing.expect(try ds.unite(0, 2));

    // Tree B: 4-5, 6-7, then (4,5)-(6,7)
    try std.testing.expect(try ds.unite(4, 5));
    try std.testing.expect(try ds.unite(6, 7));
    try std.testing.expect(try ds.unite(4, 6));

    // Now merge the two balanced trees
    try std.testing.expect(try ds.unite(0, 4));

    // All should be connected
    for (0..8) |i| {
        for (0..8) |j| {
            try std.testing.expect(try ds.connected(@intCast(i), @intCast(j)));
        }
    }

    try std.testing.expectEqual(@as(usize, 1), ds.numSets());
    try ds.validate();
}

test "DisjointSet: empty set" {
    var ds = DisjointSet(u32).init(std.testing.allocator);
    defer ds.deinit();

    try std.testing.expectEqual(@as(usize, 0), ds.count());
    try std.testing.expectEqual(@as(usize, 0), ds.numSets());
    try ds.validate();
}

test "DisjointSet: single element" {
    var ds = DisjointSet(u32).init(std.testing.allocator);
    defer ds.deinit();

    try ds.makeSet(42);
    try std.testing.expectEqual(@as(usize, 1), ds.count());
    try std.testing.expectEqual(@as(usize, 1), ds.numSets());

    const root = try ds.find(42);
    try std.testing.expectEqual(root, try ds.find(42));

    try ds.validate();
}

test "DisjointSet: memory leak check" {
    var ds = DisjointSet(u32).init(std.testing.allocator);
    defer ds.deinit();

    for (0..100) |i| {
        try ds.makeSet(@intCast(i));
    }

    for (0..99) |i| {
        _ = try ds.unite(@intCast(i), @intCast(i + 1));
    }

    try ds.validate();
}
