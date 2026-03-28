const std = @import("std");
const Allocator = std.mem.Allocator;

/// LRU (Least Recently Used) Cache Implementation
///
/// Evicts the least recently used item when capacity is reached.
/// Combines HashMap (O(1) lookup) + Doubly Linked List (O(1) reordering).
///
/// Time Complexity:
/// - get(): O(1) average
/// - put(): O(1) average
/// - evict(): O(1)
///
/// Space Complexity: O(capacity) for storage + O(capacity) for access order tracking
///
/// Use Cases:
/// - Page replacement in operating systems
/// - Database buffer pools (e.g., silica buffer_pool.zig)
/// - Web server caching
/// - Memory-constrained systems needing temporal locality
pub fn LRU(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();

        pub const Node = struct {
            key: K,
            value: V,
            prev: ?*Node = null,
            next: ?*Node = null,
        };

        allocator: Allocator,
        capacity: usize,
        map: std.AutoHashMap(K, *Node),
        head: ?*Node, // Most recently used
        tail: ?*Node, // Least recently used

        /// Initialize LRU cache with given capacity
        /// Time: O(1) | Space: O(capacity) for pre-allocated hash map
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
        /// Time: O(n) where n is current size | Space: O(1)
        pub fn deinit(self: *Self) void {
            var current = self.head;
            while (current) |node| {
                const next = node.next;
                self.allocator.destroy(node);
                current = next;
            }
            self.map.deinit();
        }

        /// Get value by key, moving it to most recently used
        /// Returns null if key not found
        /// Time: O(1) average | Space: O(1)
        pub fn get(self: *Self, key: K) ?V {
            const node = self.map.get(key) orelse return null;
            self.moveToHead(node);
            return node.value;
        }

        /// Put key-value pair, evicting LRU if at capacity
        /// Updates value if key exists
        /// Time: O(1) average | Space: O(1)
        pub fn put(self: *Self, key: K, value: V) !void {
            if (self.map.get(key)) |node| {
                // Update existing
                node.value = value;
                self.moveToHead(node);
            } else {
                // Insert new
                const node = try self.allocator.create(Node);
                node.* = .{ .key = key, .value = value };

                if (self.map.count() >= self.capacity) {
                    // Evict LRU
                    if (self.tail) |lru| {
                        _ = self.map.remove(lru.key);
                        self.removeNode(lru);
                        self.allocator.destroy(lru);
                    }
                }

                try self.map.put(key, node);
                self.addToHead(node);
            }
        }

        /// Remove key from cache
        /// Returns true if key existed
        /// Time: O(1) average | Space: O(1)
        pub fn remove(self: *Self, key: K) bool {
            const node = self.map.get(key) orelse return false;
            _ = self.map.remove(key);
            self.removeNode(node);
            self.allocator.destroy(node);
            return true;
        }

        /// Check if key exists (without updating access order)
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

        /// Get least recently used key (without evicting)
        /// Returns null if cache is empty
        /// Time: O(1) | Space: O(1)
        pub fn peekLRU(self: Self) ?K {
            return if (self.tail) |node| node.key else null;
        }

        /// Get most recently used key
        /// Returns null if cache is empty
        /// Time: O(1) | Space: O(1)
        pub fn peekMRU(self: Self) ?K {
            return if (self.head) |node| node.key else null;
        }

        // === Internal helpers ===

        fn moveToHead(self: *Self, node: *Node) void {
            if (self.head == node) return; // Already at head
            self.removeNode(node);
            self.addToHead(node);
        }

        fn addToHead(self: *Self, node: *Node) void {
            node.next = self.head;
            node.prev = null;
            if (self.head) |h| {
                h.prev = node;
            }
            self.head = node;
            if (self.tail == null) {
                self.tail = node;
            }
        }

        fn removeNode(self: *Self, node: *Node) void {
            if (node.prev) |p| {
                p.next = node.next;
            } else {
                self.head = node.next;
            }
            if (node.next) |n| {
                n.prev = node.prev;
            } else {
                self.tail = node.prev;
            }
        }
    };
}

// === Tests ===

test "LRU: basic operations" {
    var cache = try LRU(u32, []const u8).init(std.testing.allocator, 3);
    defer cache.deinit();

    // Put and get
    try cache.put(1, "one");
    try cache.put(2, "two");
    try cache.put(3, "three");

    try std.testing.expectEqualStrings("one", cache.get(1).?);
    try std.testing.expectEqualStrings("two", cache.get(2).?);
    try std.testing.expectEqualStrings("three", cache.get(3).?);
}

test "LRU: capacity enforcement" {
    var cache = try LRU(u32, u32).init(std.testing.allocator, 2);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);
    try std.testing.expectEqual(2, cache.size());

    // Evicts key 1 (LRU)
    try cache.put(3, 30);
    try std.testing.expectEqual(2, cache.size());
    try std.testing.expectEqual(null, cache.get(1));
    try std.testing.expectEqual(20, cache.get(2).?);
    try std.testing.expectEqual(30, cache.get(3).?);
}

test "LRU: access order updates" {
    var cache = try LRU(u32, u32).init(std.testing.allocator, 3);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);
    try cache.put(3, 30);

    // Access key 1 (moves to head)
    _ = cache.get(1);

    // Now order is: 1 (MRU) -> 3 -> 2 (LRU)
    // Adding key 4 should evict key 2
    try cache.put(4, 40);
    try std.testing.expectEqual(null, cache.get(2));
    try std.testing.expectEqual(10, cache.get(1).?);
    try std.testing.expectEqual(30, cache.get(3).?);
    try std.testing.expectEqual(40, cache.get(4).?);
}

test "LRU: update existing key" {
    var cache = try LRU(u32, u32).init(std.testing.allocator, 2);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);

    // Update key 1
    try cache.put(1, 100);
    try std.testing.expectEqual(100, cache.get(1).?);
    try std.testing.expectEqual(2, cache.size()); // Size unchanged
}

test "LRU: remove operation" {
    var cache = try LRU(u32, u32).init(std.testing.allocator, 3);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);
    try cache.put(3, 30);

    try std.testing.expect(cache.remove(2));
    try std.testing.expectEqual(2, cache.size());
    try std.testing.expectEqual(null, cache.get(2));
    try std.testing.expect(!cache.remove(2)); // Already removed
}

test "LRU: clear operation" {
    var cache = try LRU(u32, u32).init(std.testing.allocator, 3);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);
    cache.clear();

    try std.testing.expectEqual(0, cache.size());
    try std.testing.expectEqual(null, cache.get(1));
}

test "LRU: peek operations" {
    var cache = try LRU(u32, u32).init(std.testing.allocator, 3);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);
    try cache.put(3, 30);

    // MRU is 3, LRU is 1
    try std.testing.expectEqual(3, cache.peekMRU().?);
    try std.testing.expectEqual(1, cache.peekLRU().?);

    // Access key 1
    _ = cache.get(1);

    // Now MRU is 1, LRU is 2
    try std.testing.expectEqual(1, cache.peekMRU().?);
    try std.testing.expectEqual(2, cache.peekLRU().?);
}

test "LRU: zero capacity error" {
    const result = LRU(u32, u32).init(std.testing.allocator, 0);
    try std.testing.expectError(error.ZeroCapacity, result);
}

test "LRU: single capacity" {
    var cache = try LRU(u32, u32).init(std.testing.allocator, 1);
    defer cache.deinit();

    try cache.put(1, 10);
    try std.testing.expectEqual(10, cache.get(1).?);

    try cache.put(2, 20); // Evicts key 1
    try std.testing.expectEqual(null, cache.get(1));
    try std.testing.expectEqual(20, cache.get(2).?);
}

test "LRU: large scale" {
    var cache = try LRU(u32, u32).init(std.testing.allocator, 100);
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

test "LRU: complex access pattern" {
    var cache = try LRU(u32, u32).init(std.testing.allocator, 3);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);
    try cache.put(3, 30);

    // Access pattern: 2, 1, 4 (evicts 3)
    _ = cache.get(2); // 2 -> 1 -> 3
    _ = cache.get(1); // 1 -> 2 -> 3
    try cache.put(4, 40); // 4 -> 1 -> 2 (3 evicted)

    try std.testing.expectEqual(null, cache.get(3));
    try std.testing.expectEqual(10, cache.get(1).?);
    try std.testing.expectEqual(20, cache.get(2).?);
    try std.testing.expectEqual(40, cache.get(4).?);
}
