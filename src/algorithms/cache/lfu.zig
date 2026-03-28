const std = @import("std");
const Allocator = std.mem.Allocator;

/// LFU (Least Frequently Used) Cache Implementation
///
/// Evicts the least frequently accessed item when capacity is reached.
/// Ties broken by least recently used (LRU within same frequency).
/// Uses HashMap + frequency buckets (doubly linked lists per frequency).
///
/// Time Complexity:
/// - get(): O(1) average
/// - put(): O(1) average
/// - evict(): O(1)
///
/// Space Complexity: O(capacity) for storage + O(capacity) for frequency tracking
///
/// Use Cases:
/// - Web caching with hot/cold data distinction
/// - Database query result caching (zoltraak expiry logic)
/// - Content Delivery Networks (CDN)
/// - Workloads with clear access frequency patterns
pub fn LFU(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();

        pub const Node = struct {
            key: K,
            value: V,
            freq: usize,
            prev: ?*Node = null,
            next: ?*Node = null,
        };

        pub const FreqList = struct {
            head: ?*Node = null,
            tail: ?*Node = null,
            size: usize = 0,
        };

        allocator: Allocator,
        capacity: usize,
        min_freq: usize,
        map: std.AutoHashMap(K, *Node),
        freq_map: std.AutoHashMap(usize, FreqList),

        /// Initialize LFU cache with given capacity
        /// Time: O(1) | Space: O(capacity)
        pub fn init(allocator: Allocator, capacity: usize) !Self {
            if (capacity == 0) return error.ZeroCapacity;
            return Self{
                .allocator = allocator,
                .capacity = capacity,
                .min_freq = 0,
                .map = std.AutoHashMap(K, *Node).init(allocator),
                .freq_map = std.AutoHashMap(usize, FreqList).init(allocator),
            };
        }

        /// Deallocate all nodes and hash maps
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            var freq_iter = self.freq_map.valueIterator();
            while (freq_iter.next()) |list| {
                var current = list.head;
                while (current) |node| {
                    const next = node.next;
                    self.allocator.destroy(node);
                    current = next;
                }
            }
            self.map.deinit();
            self.freq_map.deinit();
        }

        /// Get value by key, incrementing its frequency
        /// Returns null if key not found
        /// Time: O(1) average | Space: O(1)
        pub fn get(self: *Self, key: K) !?V {
            const node = self.map.get(key) orelse return null;
            try self.incrementFreq(node);
            return node.value;
        }

        /// Put key-value pair, evicting LFU if at capacity
        /// Updates value if key exists
        /// Time: O(1) average | Space: O(1)
        pub fn put(self: *Self, key: K, value: V) !void {
            if (self.map.get(key)) |node| {
                // Update existing
                node.value = value;
                try self.incrementFreq(node);
            } else {
                // Insert new
                if (self.map.count() >= self.capacity) {
                    // Evict LFU
                    try self.evictLFU();
                }

                const node = try self.allocator.create(Node);
                node.* = .{ .key = key, .value = value, .freq = 1 };

                try self.map.put(key, node);
                try self.addToFreqList(node, 1);
                self.min_freq = 1;
            }
        }

        /// Remove key from cache
        /// Returns true if key existed
        /// Time: O(1) average | Space: O(1)
        pub fn remove(self: *Self, key: K) bool {
            const node = self.map.get(key) orelse return false;
            _ = self.map.remove(key);
            self.removeFromFreqList(node);
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
            var freq_iter = self.freq_map.valueIterator();
            while (freq_iter.next()) |list| {
                var current = list.head;
                while (current) |node| {
                    const next = node.next;
                    self.allocator.destroy(node);
                    current = next;
                }
            }
            self.map.clearRetainingCapacity();
            self.freq_map.clearRetainingCapacity();
            self.min_freq = 0;
        }

        /// Get access frequency of a key
        /// Returns null if key not found
        /// Time: O(1) average | Space: O(1)
        pub fn getFrequency(self: Self, key: K) ?usize {
            const node = self.map.get(key) orelse return null;
            return node.freq;
        }

        // === Internal helpers ===

        fn incrementFreq(self: *Self, node: *Node) !void {
            const old_freq = node.freq;
            const new_freq = old_freq + 1;

            self.removeFromFreqList(node);
            node.freq = new_freq;
            try self.addToFreqList(node, new_freq);

            // Update min_freq if old list is now empty
            if (old_freq == self.min_freq) {
                const old_list = self.freq_map.get(old_freq);
                if (old_list == null or old_list.?.size == 0) {
                    self.min_freq = new_freq;
                }
            }
        }

        fn addToFreqList(self: *Self, node: *Node, freq: usize) !void {
            const entry = try self.freq_map.getOrPut(freq);
            if (!entry.found_existing) {
                entry.value_ptr.* = FreqList{};
            }
            const list = entry.value_ptr;

            node.next = list.head;
            node.prev = null;
            if (list.head) |h| {
                h.prev = node;
            }
            list.head = node;
            if (list.tail == null) {
                list.tail = node;
            }
            list.size += 1;
        }

        fn removeFromFreqList(self: *Self, node: *Node) void {
            var list = self.freq_map.getPtr(node.freq) orelse return;

            if (node.prev) |p| {
                p.next = node.next;
            } else {
                list.head = node.next;
            }
            if (node.next) |n| {
                n.prev = node.prev;
            } else {
                list.tail = node.prev;
            }
            list.size -= 1;
        }

        fn evictLFU(self: *Self) !void {
            const list = self.freq_map.getPtr(self.min_freq) orelse return;
            const lfu = list.tail orelse return; // LRU within min_freq

            _ = self.map.remove(lfu.key);
            self.removeFromFreqList(lfu);
            self.allocator.destroy(lfu);
        }
    };
}

// === Tests ===

test "LFU: basic operations" {
    var cache = try LFU(u32, []const u8).init(std.testing.allocator, 3);
    defer cache.deinit();

    try cache.put(1, "one");
    try cache.put(2, "two");
    try cache.put(3, "three");

    try std.testing.expectEqualStrings("one", (try cache.get(1)).?);
    try std.testing.expectEqualStrings("two", (try cache.get(2)).?);
    try std.testing.expectEqualStrings("three", (try cache.get(3)).?);
}

test "LFU: frequency tracking" {
    var cache = try LFU(u32, u32).init(std.testing.allocator, 3);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);
    try cache.put(3, 30);

    // Access key 1 three times
    _ = try cache.get(1);
    _ = try cache.get(1);
    _ = try cache.get(1);

    // Frequencies: 1 -> 4, 2 -> 1, 3 -> 1
    try std.testing.expectEqual(4, cache.getFrequency(1).?);
    try std.testing.expectEqual(1, cache.getFrequency(2).?);
    try std.testing.expectEqual(1, cache.getFrequency(3).?);
}

test "LFU: eviction by frequency" {
    var cache = try LFU(u32, u32).init(std.testing.allocator, 2);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);

    // Access key 1 (freq=2), key 2 (freq=1)
    _ = try cache.get(1);

    // Insert key 3, should evict key 2 (least frequent)
    try cache.put(3, 30);
    try std.testing.expectEqual(null, try cache.get(2));
    try std.testing.expectEqual(10, (try cache.get(1)).?);
    try std.testing.expectEqual(30, (try cache.get(3)).?);
}

test "LFU: LRU tie-breaking" {
    var cache = try LFU(u32, u32).init(std.testing.allocator, 3);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);
    try cache.put(3, 30);

    // All have freq=1, order is 3 (MRU) -> 2 -> 1 (LRU)
    // Adding key 4 should evict key 1 (LRU within freq=1)
    try cache.put(4, 40);
    try std.testing.expectEqual(null, try cache.get(1));
    try std.testing.expectEqual(20, (try cache.get(2)).?);
    try std.testing.expectEqual(30, (try cache.get(3)).?);
    try std.testing.expectEqual(40, (try cache.get(4)).?);
}

test "LFU: update existing key" {
    var cache = try LFU(u32, u32).init(std.testing.allocator, 2);
    defer cache.deinit();

    try cache.put(1, 10);
    _ = try cache.get(1); // freq=2

    // Update value, should increment frequency
    try cache.put(1, 100);
    try std.testing.expectEqual(100, (try cache.get(1)).?);
    try std.testing.expectEqual(4, cache.getFrequency(1).?); // put increments too
}

test "LFU: remove operation" {
    var cache = try LFU(u32, u32).init(std.testing.allocator, 3);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);
    try cache.put(3, 30);

    try std.testing.expect(cache.remove(2));
    try std.testing.expectEqual(2, cache.size());
    try std.testing.expectEqual(null, try cache.get(2));
    try std.testing.expect(!cache.remove(2)); // Already removed
}

test "LFU: clear operation" {
    var cache = try LFU(u32, u32).init(std.testing.allocator, 3);
    defer cache.deinit();

    try cache.put(1, 10);
    try cache.put(2, 20);
    _ = try cache.get(1);

    cache.clear();
    try std.testing.expectEqual(0, cache.size());
    try std.testing.expectEqual(null, try cache.get(1));
}

test "LFU: zero capacity error" {
    const result = LFU(u32, u32).init(std.testing.allocator, 0);
    try std.testing.expectError(error.ZeroCapacity, result);
}

test "LFU: single capacity" {
    var cache = try LFU(u32, u32).init(std.testing.allocator, 1);
    defer cache.deinit();

    try cache.put(1, 10);
    try std.testing.expectEqual(10, (try cache.get(1)).?);

    try cache.put(2, 20); // Evicts key 1
    try std.testing.expectEqual(null, try cache.get(1));
    try std.testing.expectEqual(20, (try cache.get(2)).?);
}

test "LFU: large scale" {
    var cache = try LFU(u32, u32).init(std.testing.allocator, 100);
    defer cache.deinit();

    // Insert 150 items
    var i: u32 = 0;
    while (i < 150) : (i += 1) {
        try cache.put(i, i * 10);
    }

    // Only last 100 should remain
    try std.testing.expectEqual(100, cache.size());
}

test "LFU: hot/cold data pattern" {
    var cache = try LFU(u32, u32).init(std.testing.allocator, 3);
    defer cache.deinit();

    try cache.put(1, 10); // hot
    try cache.put(2, 20); // cold
    try cache.put(3, 30); // cold

    // Make key 1 hot (freq=5)
    var i: usize = 0;
    while (i < 4) : (i += 1) {
        _ = try cache.get(1);
    }

    // Insert 4, 5, 6 - should evict cold keys 2, 3 but keep hot key 1
    try cache.put(4, 40);
    try cache.put(5, 50);
    try cache.put(6, 60);

    try std.testing.expectEqual(10, (try cache.get(1)).?); // Still present (hot)
    try std.testing.expectEqual(null, try cache.get(2)); // Evicted (cold)
    try std.testing.expectEqual(null, try cache.get(3)); // Evicted (cold)
}
