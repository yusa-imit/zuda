const std = @import("std");
const Allocator = std.mem.Allocator;

/// FIFO (First In First Out) Cache Implementation
///
/// Evicts the oldest inserted item when capacity is reached.
/// Simpler than LRU (no reordering on access), lower overhead.
///
/// Time Complexity:
/// - get(): O(1) average
/// - put(): O(1) average
/// - evict(): O(1)
///
/// Space Complexity: O(capacity)
///
/// Use Cases:
/// - Simple page replacement
/// - Message queues with bounded size
/// - Ring buffers
/// - Scenarios where recency doesn't matter, only insertion order
pub fn FIFO(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();

        pub const Node = struct {
            key: K,
            value: V,
            next: ?*Node = null,
        };

        allocator: Allocator,
        capacity: usize,
        map: std.AutoHashMap(K, *Node),
        head: ?*Node, // Oldest (next to evict)
        tail: ?*Node, // Newest

        /// Initialize FIFO cache with given capacity
        /// Time: O(1) | Space: O(capacity)
        pub fn init(allocator: Allocator, capacity: usize) !Self {
            if (capacity == 0) return error.ZeroCapacity;
            return Self{
                .allocator = allocator,
                .capacity = capacity,
                .map = std.AutoHashMap(K, *Node).init(allocator),
                .head = null,
                .tail = null,
            };
        }

        /// Deallocate all nodes and hash map
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            var current = self.head;
            while (current) |node| {
                const next = node.next;
                self.allocator.destroy(node);
                current = next;
            }
            self.map.deinit();
        }

        /// Get value by key (no reordering)
        /// Returns null if key not found
        /// Time: O(1) average | Space: O(1)
        pub fn get(self: Self, key: K) ?V {
            const node = self.map.get(key) orelse return null;
            return node.value;
        }

        /// Put key-value pair, evicting oldest if at capacity
        /// Updates value if key exists (maintains insertion order)
        /// Time: O(1) average | Space: O(1)
        pub fn put(self: *Self, key: K, value: V) !void {
            if (self.map.get(key)) |node| {
                // Update existing (no reordering)
                node.value = value;
            } else {
                // Insert new
                if (self.map.count() >= self.capacity) {
                    // Evict oldest (head)
                    if (self.head) |oldest| {
                        _ = self.map.remove(oldest.key);
                        self.head = oldest.next;
                        if (self.head == null) {
                            self.tail = null;
                        }
                        self.allocator.destroy(oldest);
                    }
                }

                const node = try self.allocator.create(Node);
                node.* = .{ .key = key, .value = value, .next = null };

                if (self.tail) |t| {
                    t.next = node;
                } else {
                    self.head = node;
                }
                self.tail = node;

                try self.map.put(key, node);
            }
        }

        /// Remove key from cache
        /// Returns true if key existed
        /// Time: O(n) worst case (need to find predecessor)
        /// Space: O(1)
        pub fn remove(self: *Self, key: K) bool {
            const node = self.map.get(key) orelse return false;
            _ = self.map.remove(key);

            // Find predecessor
            if (self.head == node) {
                self.head = node.next;
                if (self.head == null) {
                    self.tail = null;
                }
            } else {
                var current = self.head;
                while (current) |curr| {
                    if (curr.next == node) {
                        curr.next = node.next;
                        if (node == self.tail) {
                            self.tail = curr;
                        }
                        break;
                    }
                    current = curr.next;
                }
            }

            self.allocator.destroy(node);
            return true;
        }

        /// Check if key exists
        /// Time: O(1) average | Space: O(1)
        pub fn contains(self: Self, key: K) bool {
            return self.map.contains(key);
        }

        /// Get current number of items
        /// Time: O(1) | Space: O(1)
        pub fn size(self: Self) usize {
            return self.map.count();
        }

        /// Clear all items
        /// Time: O(n) | Space: O(1)
        pub fn clear(self: *Self) void {
            var current = self.head;
            while (current) |node| {
                const next = node.next;
                self.allocator.destroy(node);
                current = next;
            }
            self.map.clearRetainingCapacity();
            self.head = null;
            self.tail = null;
        }

        /// Get oldest key (next to evict)
        /// Returns null if cache is empty
        /// Time: O(1) | Space: O(1)
        pub fn peekOldest(self: Self) ?K {
            return if (self.head) |node| node.key else null;
        }

        /// Get newest key
        /// Returns null if cache is empty
        /// Time: O(1) | Space: O(1)
        pub fn peekNewest(self: Self) ?K {
            return if (self.tail) |node| node.key else null;
        }
    };
}

// === Tests ===

test "FIFO: basic operations" {
    var cache = try FIFO(u32, []const u8).init(std.testing.allocator, 3);
    defer cache.deinit();

    try cache.put(1, "one");
    try cache.put(2, "two");
    try cache.put(3, "three");

    try std.testing.expectEqualStrings("one", cache.get(1).?);
    try std.testing.expectEqualStrings("two", cache.get(2).?);
    try std.testing.expectEqualStrings("three", cache.get(3).?);
}

test "FIFO: capacity enforcement" {
    var cache = try FIFO(u32, u32).init(std.testing.allocator, 2);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);
    try std.testing.expectEqual(2, cache.size());

    // Evicts key 1 (oldest)
    try cache.put(3, 30);
    try std.testing.expectEqual(2, cache.size());
    try std.testing.expectEqual(null, cache.get(1));
    try std.testing.expectEqual(20, cache.get(2).?);
    try std.testing.expectEqual(30, cache.get(3).?);
}

test "FIFO: no reordering on access" {
    var cache = try FIFO(u32, u32).init(std.testing.allocator, 3);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);
    try cache.put(3, 30);

    // Access key 1 multiple times (should NOT reorder)
    _ = cache.get(1);
    _ = cache.get(1);
    _ = cache.get(1);

    // Order unchanged: 1 (oldest) -> 2 -> 3 (newest)
    // Adding key 4 should still evict key 1
    try cache.put(4, 40);
    try std.testing.expectEqual(null, cache.get(1)); // Evicted despite recent access
    try std.testing.expectEqual(20, cache.get(2).?);
}

test "FIFO: update existing key" {
    var cache = try FIFO(u32, u32).init(std.testing.allocator, 2);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);

    // Update key 1 (maintains insertion order)
    try cache.put(1, 100);
    try std.testing.expectEqual(100, cache.get(1).?);
    try std.testing.expectEqual(2, cache.size());

    // Key 1 still oldest, so next insert evicts it
    try cache.put(3, 30);
    try std.testing.expectEqual(null, cache.get(1));
}

test "FIFO: remove operation" {
    var cache = try FIFO(u32, u32).init(std.testing.allocator, 3);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);
    try cache.put(3, 30);

    try std.testing.expect(cache.remove(2));
    try std.testing.expectEqual(2, cache.size());
    try std.testing.expectEqual(null, cache.get(2));
    try std.testing.expect(!cache.remove(2)); // Already removed
}

test "FIFO: clear operation" {
    var cache = try FIFO(u32, u32).init(std.testing.allocator, 3);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);
    cache.clear();

    try std.testing.expectEqual(0, cache.size());
    try std.testing.expectEqual(null, cache.get(1));
}

test "FIFO: peek operations" {
    var cache = try FIFO(u32, u32).init(std.testing.allocator, 3);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);
    try cache.put(3, 30);

    // Oldest is 1, newest is 3
    try std.testing.expectEqual(1, cache.peekOldest().?);
    try std.testing.expectEqual(3, cache.peekNewest().?);
}

test "FIFO: zero capacity error" {
    const result = FIFO(u32, u32).init(std.testing.allocator, 0);
    try std.testing.expectError(error.ZeroCapacity, result);
}

test "FIFO: single capacity" {
    var cache = try FIFO(u32, u32).init(std.testing.allocator, 1);
    defer cache.deinit();

    try cache.put(1, 10);
    try std.testing.expectEqual(10, cache.get(1).?);

    try cache.put(2, 20); // Evicts key 1
    try std.testing.expectEqual(null, cache.get(1));
    try std.testing.expectEqual(20, cache.get(2).?);
}

test "FIFO: large scale" {
    var cache = try FIFO(u32, u32).init(std.testing.allocator, 100);
    defer cache.deinit();

    // Insert 150 items
    var i: u32 = 0;
    while (i < 150) : (i += 1) {
        try cache.put(i, i * 10);
    }

    // Only last 100 should remain
    try std.testing.expectEqual(100, cache.size());
    try std.testing.expectEqual(null, cache.get(0)); // Evicted
    try std.testing.expectEqual(null, cache.get(49)); // Evicted
    try std.testing.expectEqual(500, cache.get(50).?); // Kept
    try std.testing.expectEqual(1490, cache.get(149).?); // Kept
}

test "FIFO: sequential eviction" {
    var cache = try FIFO(u32, u32).init(std.testing.allocator, 3);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);
    try cache.put(3, 30);

    // Evictions follow insertion order: 1, 2, 3
    try cache.put(4, 40); // Evicts 1
    try cache.put(5, 50); // Evicts 2
    try cache.put(6, 60); // Evicts 3

    try std.testing.expectEqual(null, cache.get(1));
    try std.testing.expectEqual(null, cache.get(2));
    try std.testing.expectEqual(null, cache.get(3));
    try std.testing.expectEqual(40, cache.get(4).?);
    try std.testing.expectEqual(50, cache.get(5).?);
    try std.testing.expectEqual(60, cache.get(6).?);
}
