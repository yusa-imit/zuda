const std = @import("std");
const Allocator = std.mem.Allocator;

/// Van Emde Boas Tree (vEB Tree)
/// A recursive data structure for maintaining a dynamic set of integers
/// from a bounded universe U = {0, 1, ..., u-1} where u = 2^k.
///
/// Operations: insert, remove, contains, minimum, maximum, predecessor, successor
/// Time complexity: O(log log u) for all operations
/// Space complexity: O(u) worst case, but can be optimized with lazy allocation
///
/// The tree is recursively defined:
/// - Base case (u=2): Store 2 bits for elements 0 and 1
/// - Recursive case: Store min, max, a summary vEB tree of size sqrt(u),
///   and sqrt(u) cluster vEB trees each of size sqrt(u)
///
/// For element x:
/// - high(x) = x / sqrt(u) — which cluster
/// - low(x) = x % sqrt(u) — position within cluster
pub fn VanEmdeBoasTree(comptime u: u32) type {
    // Validate universe size is a power of 2
    comptime {
        if (u == 0 or (u & (u - 1)) != 0) {
            @compileError("Universe size must be a power of 2");
        }
    }

    return struct {
        const Self = @This();
        const universe_size: u32 = u;
        const sqrt_u: u32 = blk: {
            var bits: u32 = 0;
            var n = u;
            while (n > 1) : (n >>= 1) {
                bits += 1;
            }
            break :blk @as(u32, 1) << (bits / 2);
        };

        // For base case (u=2), we only need 2 bits
        const base_case = (u == 2);

        allocator: Allocator,
        min: ?u32, // Minimum element (not stored in clusters)
        max: ?u32, // Maximum element
        summary: if (base_case) void else ?*VanEmdeBoasTree(sqrt_u), // Summary tree
        cluster: if (base_case) void else []?*VanEmdeBoasTree(sqrt_u), // Array of cluster trees

        /// Initialize an empty vEB tree
        /// Time: O(1)
        pub fn init(allocator: Allocator) !Self {
            if (base_case) {
                return Self{
                    .allocator = allocator,
                    .min = null,
                    .max = null,
                    .summary = {},
                    .cluster = {},
                };
            } else {
                const cluster = try allocator.alloc(?*VanEmdeBoasTree(sqrt_u), sqrt_u);
                @memset(cluster, null);
                return Self{
                    .allocator = allocator,
                    .min = null,
                    .max = null,
                    .summary = null,
                    .cluster = cluster,
                };
            }
        }

        /// Free all memory used by the tree
        /// Time: O(u)
        pub fn deinit(self: *Self) void {
            if (base_case) return;

            if (self.summary) |summary| {
                summary.deinit();
                self.allocator.destroy(summary);
            }

            for (self.cluster) |cluster_ptr| {
                if (cluster_ptr) |c| {
                    c.deinit();
                    self.allocator.destroy(c);
                }
            }

            self.allocator.free(self.cluster);
        }

        /// Check if the tree is empty
        /// Time: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.min == null;
        }

        /// Get the number of elements in the tree
        /// Time: O(u) — requires full traversal
        pub fn count(self: *const Self) usize {
            if (self.min == null) return 0;

            if (base_case) {
                var c: usize = 0;
                if (self.min != null) c += 1;
                if (self.max) |max_val| {
                    if (self.min) |min_val| {
                        if (max_val != min_val) c += 1;
                    }
                }
                return c;
            }

            var total: usize = 0;
            if (self.min) |min_val| {
                total += 1;
                if (self.max) |max_val| {
                    if (min_val != max_val) {
                        // Count elements in clusters
                        for (self.cluster) |cluster_ptr| {
                            if (cluster_ptr) |c| {
                                total += c.count();
                            }
                        }
                    }
                }
            }
            return total;
        }

        /// Get high index (cluster number)
        inline fn high(x: u32) u32 {
            return x / sqrt_u;
        }

        /// Get low index (position within cluster)
        inline fn low(x: u32) u32 {
            return x % sqrt_u;
        }

        /// Combine high and low indices
        inline fn index(h: u32, l: u32) u32 {
            return h * sqrt_u + l;
        }

        /// Insert an element into the tree
        /// Time: O(log log u)
        /// Returns true if the element was newly inserted, false if it already existed
        pub fn insert(self: *Self, x: u32) !bool {
            if (x >= u) return error.OutOfBounds;

            // Empty tree case
            if (self.min == null) {
                self.min = x;
                self.max = x;
                return true;
            }

            const min_val = self.min.?;
            const max_val = self.max.?;

            // Element already exists as min or max
            if (x == min_val or x == max_val) {
                return false;
            }

            // Base case
            if (base_case) {
                // We have min and max, and x is different from both
                // Update min/max accordingly
                if (x < min_val) {
                    self.min = x;
                } else if (x > max_val) {
                    self.max = x;
                }
                return true;
            }

            // If inserting smaller than min, swap
            var to_insert = x;
            if (x < min_val) {
                self.min = x;
                to_insert = min_val;
            }

            const h = high(to_insert);
            const l = low(to_insert);

            // Ensure cluster exists
            if (self.cluster[h] == null) {
                self.cluster[h] = try self.allocator.create(VanEmdeBoasTree(sqrt_u));
                self.cluster[h].?.* = try VanEmdeBoasTree(sqrt_u).init(self.allocator);

                // Update summary
                if (self.summary == null) {
                    self.summary = try self.allocator.create(VanEmdeBoasTree(sqrt_u));
                    self.summary.?.* = try VanEmdeBoasTree(sqrt_u).init(self.allocator);
                }
                _ = try self.summary.?.insert(h);
            }

            const inserted = try self.cluster[h].?.insert(l);

            // Update max if necessary
            if (to_insert > max_val) {
                self.max = to_insert;
            }

            return inserted;
        }

        /// Remove an element from the tree
        /// Time: O(log log u)
        /// Returns true if the element was found and removed, false otherwise
        pub fn remove(self: *Self, x: u32) !bool {
            if (x >= u) return error.OutOfBounds;
            if (self.min == null) return false;

            const min_val = self.min.?;
            const max_val = self.max.?;

            // Single element case
            if (min_val == max_val) {
                if (x == min_val) {
                    self.min = null;
                    self.max = null;
                    return true;
                }
                return false;
            }

            // Base case
            if (base_case) {
                if (x == min_val) {
                    self.min = max_val;
                    return true;
                } else if (x == max_val) {
                    self.max = min_val;
                    return true;
                }
                return false;
            }

            // Removing min: replace with next element
            var to_remove = x;
            if (x == min_val) {
                const first_cluster = self.summary.?.minimum() orelse {
                    // Only min exists, no other elements
                    if (x == max_val) {
                        self.min = null;
                        self.max = null;
                        return true;
                    }
                    return false;
                };
                const new_min_low = self.cluster[first_cluster].?.minimum().?;
                to_remove = index(first_cluster, new_min_low);
                self.min = to_remove;
            }

            const h = high(to_remove);
            const l = low(to_remove);

            if (self.cluster[h]) |cluster| {
                const removed = try cluster.remove(l);
                if (!removed) return false;

                // If cluster is now empty, remove from summary
                if (cluster.isEmpty()) {
                    cluster.deinit();
                    self.allocator.destroy(cluster);
                    self.cluster[h] = null;

                    if (self.summary) |summary| {
                        _ = try summary.remove(h);

                        // If summary is now empty, free it
                        if (summary.isEmpty()) {
                            summary.deinit();
                            self.allocator.destroy(summary);
                            self.summary = null;
                        }
                    }

                    // Update max if we removed it
                    if (to_remove == max_val) {
                        if (self.summary) |summary| {
                            if (summary.maximum()) |max_cluster| {
                                const new_max = index(max_cluster, self.cluster[max_cluster].?.maximum().?);
                                self.max = new_max;
                            } else {
                                self.max = self.min;
                            }
                        } else {
                            self.max = self.min;
                        }
                    }
                } else {
                    // Update max if we removed it
                    if (to_remove == max_val) {
                        const new_max = index(h, cluster.maximum().?);
                        self.max = new_max;
                    }
                }

                return true;
            }

            return false;
        }

        /// Check if an element exists in the tree
        /// Time: O(log log u)
        pub fn contains(self: *const Self, x: u32) bool {
            if (x >= u) return false;
            if (self.min == null) return false;

            const min_val = self.min.?;
            const max_val = self.max.?;

            if (x == min_val or x == max_val) return true;

            if (base_case) return false;

            const h = high(x);
            const l = low(x);

            if (self.cluster[h]) |cluster| {
                return cluster.contains(l);
            }

            return false;
        }

        /// Get the minimum element
        /// Time: O(1)
        pub fn minimum(self: *const Self) ?u32 {
            return self.min;
        }

        /// Get the maximum element
        /// Time: O(1)
        pub fn maximum(self: *const Self) ?u32 {
            return self.max;
        }

        /// Find the predecessor of x (largest element < x)
        /// Time: O(log log u)
        pub fn predecessor(self: *const Self, x: u32) ?u32 {
            if (x >= u or self.min == null) return null;

            const min_val = self.min.?;

            // Base case
            if (base_case) {
                if (x == 1 and min_val == 0) return 0;
                return null;
            }

            // If x is smaller than or equal to min, no predecessor
            if (x <= min_val) return null;

            const max_val = self.max.?;

            // If x is greater than max, predecessor is max
            if (x > max_val) return max_val;

            const h = high(x);
            const l = low(x);

            // Check within same cluster
            if (self.cluster[h]) |cluster| {
                if (cluster.minimum()) |cluster_min| {
                    if (l > cluster_min) {
                        if (cluster.predecessor(l)) |pred_low| {
                            return index(h, pred_low);
                        }
                    }
                }
            }

            // Look in previous cluster
            if (self.summary) |summary| {
                if (summary.predecessor(h)) |pred_cluster| {
                    const pred_low = self.cluster[pred_cluster].?.maximum().?;
                    return index(pred_cluster, pred_low);
                }
            }

            // Only min is smaller
            if (min_val < x) return min_val;

            return null;
        }

        /// Find the successor of x (smallest element > x)
        /// Time: O(log log u)
        pub fn successor(self: *const Self, x: u32) ?u32 {
            if (x >= u - 1 or self.min == null) return null;

            const min_val = self.min.?;

            // Base case
            if (base_case) {
                if (x == 0 and self.max.? == 1) return 1;
                return null;
            }

            // If x is smaller than min, successor is min
            if (x < min_val) return min_val;

            const max_val = self.max.?;

            // If x >= max, no successor
            if (x >= max_val) return null;

            const h = high(x);
            const l = low(x);

            // Check within same cluster
            if (self.cluster[h]) |cluster| {
                if (cluster.maximum()) |cluster_max| {
                    if (l < cluster_max) {
                        if (cluster.successor(l)) |succ_low| {
                            return index(h, succ_low);
                        }
                    }
                }
            }

            // Look in next cluster
            if (self.summary) |summary| {
                if (summary.successor(h)) |succ_cluster| {
                    const succ_low = self.cluster[succ_cluster].?.minimum().?;
                    return index(succ_cluster, succ_low);
                }
            }

            return null;
        }

        /// Clear all elements from the tree
        /// Time: O(u)
        pub fn clear(self: *Self) void {
            if (base_case) {
                self.min = null;
                self.max = null;
                return;
            }

            if (self.summary) |summary| {
                summary.deinit();
                self.allocator.destroy(summary);
                self.summary = null;
            }

            for (self.cluster, 0..) |cluster_ptr, i| {
                if (cluster_ptr) |c| {
                    c.deinit();
                    self.allocator.destroy(c);
                    self.cluster[i] = null;
                }
            }

            self.min = null;
            self.max = null;
        }

        /// Validate tree invariants (for testing)
        pub fn validate(self: *const Self) !void {
            if (self.min == null) {
                if (self.max != null) return error.TreeInvariant;
                if (!base_case and self.summary != null) return error.TreeInvariant;
                return;
            }

            const min_val = self.min.?;
            const max_val = self.max.?;

            if (min_val >= u or max_val >= u) return error.TreeInvariant;
            if (min_val > max_val) return error.TreeInvariant;

            if (!base_case) {
                // Validate summary and clusters
                for (self.cluster, 0..) |cluster_ptr, i| {
                    if (cluster_ptr) |cluster| {
                        try cluster.validate();

                        // Check summary contains this cluster index
                        if (self.summary) |summary| {
                            if (!summary.contains(@intCast(i))) return error.TreeInvariant;
                        }
                    }
                }

                // Validate summary
                if (self.summary) |summary| {
                    try summary.validate();
                }
            }
        }
    };
}

// =============================================================================
// Tests
// =============================================================================

test "VanEmdeBoasTree: init and deinit" {
    const VEB = VanEmdeBoasTree(16);
    var tree = try VEB.init(std.testing.allocator);
    defer tree.deinit();

    try std.testing.expect(tree.isEmpty());
    try std.testing.expectEqual(@as(?u32, null), tree.minimum());
    try std.testing.expectEqual(@as(?u32, null), tree.maximum());
}

test "VanEmdeBoasTree: insert and contains" {
    const VEB = VanEmdeBoasTree(16);
    var tree = try VEB.init(std.testing.allocator);
    defer tree.deinit();

    try std.testing.expect(try tree.insert(5));
    try std.testing.expect(tree.contains(5));
    try std.testing.expect(!tree.contains(3));
    try std.testing.expect(!tree.contains(7));

    try std.testing.expect(try tree.insert(3));
    try std.testing.expect(try tree.insert(7));
    try std.testing.expect(tree.contains(3));
    try std.testing.expect(tree.contains(5));
    try std.testing.expect(tree.contains(7));
}

test "VanEmdeBoasTree: duplicate insert" {
    const VEB = VanEmdeBoasTree(16);
    var tree = try VEB.init(std.testing.allocator);
    defer tree.deinit();

    try std.testing.expect(try tree.insert(5));
    try std.testing.expect(!try tree.insert(5)); // Duplicate
    try std.testing.expectEqual(@as(usize, 1), tree.count());
}

test "VanEmdeBoasTree: minimum and maximum" {
    const VEB = VanEmdeBoasTree(16);
    var tree = try VEB.init(std.testing.allocator);
    defer tree.deinit();

    _ = try tree.insert(5);
    _ = try tree.insert(3);
    _ = try tree.insert(7);
    _ = try tree.insert(1);
    _ = try tree.insert(9);

    try std.testing.expectEqual(@as(?u32, 1), tree.minimum());
    try std.testing.expectEqual(@as(?u32, 9), tree.maximum());
}

test "VanEmdeBoasTree: remove" {
    const VEB = VanEmdeBoasTree(16);
    var tree = try VEB.init(std.testing.allocator);
    defer tree.deinit();

    _ = try tree.insert(5);
    _ = try tree.insert(3);
    _ = try tree.insert(7);

    try std.testing.expect(try tree.remove(3));
    try std.testing.expect(!tree.contains(3));
    try std.testing.expect(tree.contains(5));
    try std.testing.expect(tree.contains(7));

    try std.testing.expect(!try tree.remove(3)); // Already removed
}

test "VanEmdeBoasTree: remove min and max" {
    const VEB = VanEmdeBoasTree(16);
    var tree = try VEB.init(std.testing.allocator);
    defer tree.deinit();

    _ = try tree.insert(1);
    _ = try tree.insert(5);
    _ = try tree.insert(9);

    try std.testing.expect(try tree.remove(1)); // Remove min
    try std.testing.expectEqual(@as(?u32, 5), tree.minimum());

    try std.testing.expect(try tree.remove(9)); // Remove max
    try std.testing.expectEqual(@as(?u32, 5), tree.maximum());

    try std.testing.expect(try tree.remove(5)); // Remove last element
    try std.testing.expect(tree.isEmpty());
}

test "VanEmdeBoasTree: predecessor" {
    const VEB = VanEmdeBoasTree(16);
    var tree = try VEB.init(std.testing.allocator);
    defer tree.deinit();

    _ = try tree.insert(1);
    _ = try tree.insert(5);
    _ = try tree.insert(9);
    _ = try tree.insert(13);

    try std.testing.expectEqual(@as(?u32, null), tree.predecessor(0));
    try std.testing.expectEqual(@as(?u32, null), tree.predecessor(1));
    try std.testing.expectEqual(@as(?u32, 1), tree.predecessor(2));
    try std.testing.expectEqual(@as(?u32, 1), tree.predecessor(5));
    try std.testing.expectEqual(@as(?u32, 5), tree.predecessor(6));
    try std.testing.expectEqual(@as(?u32, 5), tree.predecessor(9));
    try std.testing.expectEqual(@as(?u32, 9), tree.predecessor(13));
    try std.testing.expectEqual(@as(?u32, 13), tree.predecessor(15));
}

test "VanEmdeBoasTree: successor" {
    const VEB = VanEmdeBoasTree(16);
    var tree = try VEB.init(std.testing.allocator);
    defer tree.deinit();

    _ = try tree.insert(1);
    _ = try tree.insert(5);
    _ = try tree.insert(9);
    _ = try tree.insert(13);

    try std.testing.expectEqual(@as(?u32, 1), tree.successor(0));
    try std.testing.expectEqual(@as(?u32, 5), tree.successor(1));
    try std.testing.expectEqual(@as(?u32, 5), tree.successor(2));
    try std.testing.expectEqual(@as(?u32, 9), tree.successor(5));
    try std.testing.expectEqual(@as(?u32, 9), tree.successor(6));
    try std.testing.expectEqual(@as(?u32, 13), tree.successor(9));
    try std.testing.expectEqual(@as(?u32, null), tree.successor(13));
    try std.testing.expectEqual(@as(?u32, null), tree.successor(15));
}

test "VanEmdeBoasTree: clear" {
    const VEB = VanEmdeBoasTree(16);
    var tree = try VEB.init(std.testing.allocator);
    defer tree.deinit();

    _ = try tree.insert(1);
    _ = try tree.insert(5);
    _ = try tree.insert(9);

    tree.clear();
    try std.testing.expect(tree.isEmpty());
    try std.testing.expect(!tree.contains(1));
    try std.testing.expect(!tree.contains(5));
    try std.testing.expect(!tree.contains(9));
}

test "VanEmdeBoasTree: count" {
    const VEB = VanEmdeBoasTree(16);
    var tree = try VEB.init(std.testing.allocator);
    defer tree.deinit();

    try std.testing.expectEqual(@as(usize, 0), tree.count());

    _ = try tree.insert(1);
    try std.testing.expectEqual(@as(usize, 1), tree.count());

    _ = try tree.insert(5);
    _ = try tree.insert(9);
    try std.testing.expectEqual(@as(usize, 3), tree.count());

    _ = try tree.remove(5);
    try std.testing.expectEqual(@as(usize, 2), tree.count());
}

test "VanEmdeBoasTree: base case u=2" {
    const VEB = VanEmdeBoasTree(2);
    var tree = try VEB.init(std.testing.allocator);
    defer tree.deinit();

    try std.testing.expect(tree.isEmpty());

    try std.testing.expect(try tree.insert(0));
    try std.testing.expect(tree.contains(0));
    try std.testing.expect(!tree.contains(1));
    try std.testing.expectEqual(@as(?u32, 0), tree.minimum());
    try std.testing.expectEqual(@as(?u32, 0), tree.maximum());

    try std.testing.expect(try tree.insert(1));
    try std.testing.expect(tree.contains(0));
    try std.testing.expect(tree.contains(1));
    try std.testing.expectEqual(@as(?u32, 0), tree.minimum());
    try std.testing.expectEqual(@as(?u32, 1), tree.maximum());

    try std.testing.expect(try tree.remove(0));
    try std.testing.expect(!tree.contains(0));
    try std.testing.expect(tree.contains(1));
    try std.testing.expectEqual(@as(?u32, 1), tree.minimum());
    try std.testing.expectEqual(@as(?u32, 1), tree.maximum());
}

test "VanEmdeBoasTree: larger universe u=256" {
    const VEB = VanEmdeBoasTree(256);
    var tree = try VEB.init(std.testing.allocator);
    defer tree.deinit();

    // Insert some elements
    _ = try tree.insert(0);
    _ = try tree.insert(17);
    _ = try tree.insert(42);
    _ = try tree.insert(100);
    _ = try tree.insert(255);

    try std.testing.expectEqual(@as(?u32, 0), tree.minimum());
    try std.testing.expectEqual(@as(?u32, 255), tree.maximum());

    try std.testing.expectEqual(@as(?u32, null), tree.predecessor(0));
    try std.testing.expectEqual(@as(?u32, 0), tree.predecessor(17));
    try std.testing.expectEqual(@as(?u32, 17), tree.predecessor(42));
    try std.testing.expectEqual(@as(?u32, 100), tree.predecessor(255));

    try std.testing.expectEqual(@as(?u32, 17), tree.successor(0));
    try std.testing.expectEqual(@as(?u32, 42), tree.successor(17));
    try std.testing.expectEqual(@as(?u32, 100), tree.successor(42));
    try std.testing.expectEqual(@as(?u32, null), tree.successor(255));
}

test "VanEmdeBoasTree: stress test with random operations" {
    const VEB = VanEmdeBoasTree(64);
    var tree = try VEB.init(std.testing.allocator);
    defer tree.deinit();

    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    var expected = std.AutoHashMap(u32, void).init(std.testing.allocator);
    defer expected.deinit();

    // Random insertions and deletions
    for (0..100) |_| {
        const x = random.intRangeAtMost(u32, 0, 63);
        const should_insert = random.boolean();

        if (should_insert) {
            const inserted = try tree.insert(x);
            const was_new = try expected.fetchPut(x, {}) == null;
            try std.testing.expectEqual(was_new, inserted);
        } else {
            const removed = try tree.remove(x);
            const was_present = expected.remove(x);
            try std.testing.expectEqual(was_present, removed);
        }

        // Verify contains
        try std.testing.expectEqual(expected.contains(x), tree.contains(x));
    }

    // Verify count matches
    try std.testing.expectEqual(expected.count(), tree.count());

    // Validate tree invariants
    try tree.validate();
}

test "VanEmdeBoasTree: validate invariants" {
    const VEB = VanEmdeBoasTree(16);
    var tree = try VEB.init(std.testing.allocator);
    defer tree.deinit();

    try tree.validate(); // Empty tree

    _ = try tree.insert(5);
    try tree.validate();

    _ = try tree.insert(3);
    _ = try tree.insert(7);
    try tree.validate();

    _ = try tree.remove(3);
    try tree.validate();

    tree.clear();
    try tree.validate();
}

test "VanEmdeBoasTree: bounds checking" {
    const VEB = VanEmdeBoasTree(16);
    var tree = try VEB.init(std.testing.allocator);
    defer tree.deinit();

    try std.testing.expectError(error.OutOfBounds, tree.insert(16));
    try std.testing.expectError(error.OutOfBounds, tree.insert(100));
    try std.testing.expectError(error.OutOfBounds, tree.remove(16));
}
