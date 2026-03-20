//! LRU Cache — Least Recently Used eviction cache with O(1) operations.
//!
//! Combines a hash map for O(1) lookup with a doubly-linked list for O(1) LRU ordering.
//! When capacity is exceeded, the least recently used entry is evicted.
//!
//! Key features:
//!   - O(1) get, put, remove operations
//!   - Optional eviction callback for dirty page flushing (silica buffer pool use case)
//!   - Generic over K, V with custom hash/eql functions
//!   - Thread-unsafe (external synchronization required for concurrent access)
//!
//! Consumer use cases:
//!   - silica: buffer pool LRU page cache (1237 LOC → simplified)
//!   - General: web request caching, DNS lookup cache, computation memoization

const std = @import("std");

/// LRU Cache with O(1) get/put/remove operations.
///
/// Parameters:
///   - K: Key type
///   - V: Value type
///   - Context: Hash/equality context (default: std.hash_map.AutoContext(K))
///   - evictFn: Optional callback when entry is evicted (for cleanup/flushing)
pub fn LRUCache(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime evictFn: ?fn (key: K, value: V) void,
) type {
    return struct {
        const Self = @This();

        /// Entry in the LRU list and hash map.
        pub const Entry = struct {
            key: K,
            value: V,
            prev: ?*Entry = null,
            next: ?*Entry = null,
            pin_count: usize = 0, // Reference count for pinning (buffer pool use case)
        };

        pub const Iterator = struct {
            current: ?*Entry,

            /// Returns the next entry in MRU → LRU order (head → tail).
            /// Time: O(1) | Space: O(1)
            pub fn next(self: *Iterator) ?struct { key: K, value: V } {
                const entry = self.current orelse return null;
                self.current = entry.next;
                return .{ .key = entry.key, .value = entry.value };
            }
        };

        allocator: std.mem.Allocator,
        capacity: usize,
        map: std.HashMap(K, *Entry, Context, std.hash_map.default_max_load_percentage),
        // Doubly-linked list: head = most recently used, tail = least recently used
        head: ?*Entry = null,
        tail: ?*Entry = null,

        /// Initialize an LRU cache with the given capacity.
        /// Time: O(1) | Space: O(capacity)
        pub fn init(allocator: std.mem.Allocator, capacity: usize) Self {
            return .{
                .allocator = allocator,
                .capacity = capacity,
                .map = std.HashMap(K, *Entry, Context, std.hash_map.default_max_load_percentage).init(allocator),
            };
        }

        /// Free all resources.
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            var it = self.map.valueIterator();
            while (it.next()) |entry_ptr| {
                self.allocator.destroy(entry_ptr.*);
            }
            self.map.deinit();
        }

        /// Number of entries in the cache.
        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.map.count();
        }

        /// Check if the cache is empty.
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.map.count() == 0;
        }

        /// Get a value by key, marking it as most recently used.
        /// Returns null if key not found.
        /// Time: O(1) | Space: O(1)
        pub fn get(self: *Self, key: K) ?V {
            const entry = self.map.get(key) orelse return null;
            self.moveToHead(entry);
            return entry.value;
        }

        /// Get a value by key without updating LRU order (peek).
        /// Returns null if key not found.
        /// Time: O(1) | Space: O(1)
        pub fn peek(self: *const Self, key: K) ?V {
            const entry = self.map.get(key) orelse return null;
            return entry.value;
        }

        /// Check if a key exists in the cache.
        /// Time: O(1) | Space: O(1)
        pub fn contains(self: *const Self, key: K) bool {
            return self.map.contains(key);
        }

        /// Pin an entry, preventing it from being evicted.
        /// Increments the pin count. Pinned entries (pin_count > 0) will not be evicted.
        /// Multiple pins stack — must call unpin() the same number of times.
        /// Returns error.KeyNotFound if key does not exist.
        /// Time: O(1) | Space: O(1)
        pub fn pin(self: *Self, key: K) !void {
            const entry = self.map.get(key) orelse return error.KeyNotFound;
            entry.pin_count += 1;
        }

        /// Unpin an entry, allowing it to be evicted again.
        /// Decrements the pin count. When pin_count reaches 0, entry can be evicted.
        /// Returns error.KeyNotFound if key does not exist.
        /// Returns error.NotPinned if entry is already unpinned (pin_count == 0).
        /// Time: O(1) | Space: O(1)
        pub fn unpin(self: *Self, key: K) !void {
            const entry = self.map.get(key) orelse return error.KeyNotFound;
            if (entry.pin_count == 0) return error.NotPinned;
            entry.pin_count -= 1;
        }

        /// Check if an entry is currently pinned (pin_count > 0).
        /// Returns false if key does not exist.
        /// Time: O(1) | Space: O(1)
        pub fn isPinned(self: *const Self, key: K) bool {
            const entry = self.map.get(key) orelse return false;
            return entry.pin_count > 0;
        }

        /// Insert or update a key-value pair.
        /// If key exists, updates value and marks as most recently used.
        /// If capacity is exceeded, evicts the least recently used entry.
        /// Returns the old value if key existed, null otherwise.
        /// Time: O(1) amortized | Space: O(1)
        pub fn put(self: *Self, key: K, value: V) !?V {
            // Update existing entry
            if (self.map.getPtr(key)) |entry_ptr| {
                const old_value = entry_ptr.*.value;
                entry_ptr.*.value = value;
                self.moveToHead(entry_ptr.*);
                return old_value;
            }

            // Evict LRU if at capacity
            if (self.map.count() >= self.capacity) {
                try self.evictLRU();
            }

            // Create new entry
            const entry = try self.allocator.create(Entry);
            entry.* = .{
                .key = key,
                .value = value,
            };

            try self.map.put(key, entry);
            self.addToHead(entry);
            return null;
        }

        /// Remove a key from the cache.
        /// Returns the removed value if key existed, null otherwise.
        /// Time: O(1) amortized | Space: O(1)
        pub fn remove(self: *Self, key: K) ?V {
            const entry = self.map.fetchRemove(key) orelse return null;
            self.removeFromList(entry.value);
            const value = entry.value.value;
            self.allocator.destroy(entry.value);
            return value;
        }

        /// Remove all entries from the cache.
        /// Time: O(n) | Space: O(1)
        pub fn clear(self: *Self) void {
            var it = self.map.valueIterator();
            while (it.next()) |entry_ptr| {
                self.allocator.destroy(entry_ptr.*);
            }
            self.map.clearRetainingCapacity();
            self.head = null;
            self.tail = null;
        }

        /// Iterate over entries in MRU → LRU order (head → tail).
        /// Time: O(1) setup, O(1) per next() | Space: O(1)
        pub fn iterator(self: *const Self) Iterator {
            return .{ .current = self.head };
        }

        /// Validate cache invariants (for testing).
        /// Time: O(n) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            const n = self.map.count();

            // Check list size matches map size
            var list_count: usize = 0;
            var node = self.head;
            while (node) |entry| : (node = entry.next) {
                list_count += 1;
                // Check map contains this entry
                const map_entry = self.map.get(entry.key) orelse return error.MissingInMap;
                if (map_entry != entry) return error.MapEntryMismatch;
                // Check prev/next consistency
                if (entry.next) |next| {
                    if (next.prev != entry) return error.InvalidPrevPointer;
                }
            }
            if (list_count != n) return error.ListSizeMismatch;

            // Check head/tail consistency
            if (n == 0) {
                if (self.head != null or self.tail != null) return error.EmptyListNotNull;
            } else {
                if (self.head == null or self.tail == null) return error.NonEmptyListNull;
                if (self.head.?.prev != null) return error.HeadHasPrev;
                if (self.tail.?.next != null) return error.TailHasNext;
            }
        }

        // ── Private helpers ──────────────────────────────────────────────

        /// Move an existing entry to the head (most recently used).
        fn moveToHead(self: *Self, entry: *Entry) void {
            if (self.head == entry) return; // Already at head
            self.removeFromList(entry);
            self.addToHead(entry);
        }

        /// Add a new entry to the head (most recently used).
        fn addToHead(self: *Self, entry: *Entry) void {
            entry.next = self.head;
            entry.prev = null;
            if (self.head) |head| {
                head.prev = entry;
            }
            self.head = entry;
            if (self.tail == null) {
                self.tail = entry;
            }
        }

        /// Remove an entry from the doubly-linked list.
        fn removeFromList(self: *Self, entry: *Entry) void {
            if (entry.prev) |prev| {
                prev.next = entry.next;
            } else {
                self.head = entry.next;
            }
            if (entry.next) |next| {
                next.prev = entry.prev;
            } else {
                self.tail = entry.prev;
            }
            entry.prev = null;
            entry.next = null;
        }

        /// Evict the least recently used unpinned entry.
        /// Walks from tail (LRU) toward head (MRU) to find first unpinned entry.
        /// Returns error.NoPinnableEntries if all entries are pinned.
        fn evictLRU(self: *Self) !void {
            // Find first unpinned entry from LRU end
            var candidate = self.tail;
            while (candidate) |entry| {
                if (entry.pin_count == 0) {
                    // Found unpinned entry, evict it
                    const key = entry.key;
                    const value = entry.value;

                    // Call eviction callback if provided
                    if (evictFn) |evict| {
                        evict(key, value);
                    }

                    _ = self.map.remove(key);
                    self.removeFromList(entry);
                    self.allocator.destroy(entry);
                    return;
                }
                candidate = entry.prev; // Move toward head (MRU)
            }

            // All entries are pinned
            return error.NoPinnableEntries;
        }
    };
}

// ── Tests ────────────────────────────────────────────────────────────

test "LRUCache: basic operations" {
    const Cache = LRUCache(u32, []const u8, std.hash_map.AutoContext(u32), null);
    var cache = Cache.init(std.testing.allocator, 3);
    defer cache.deinit();

    // Empty cache
    try std.testing.expectEqual(@as(usize, 0), cache.count());
    try std.testing.expect(cache.isEmpty());
    try std.testing.expectEqual(@as(?[]const u8, null), cache.get(1));
    try cache.validate();

    // Insert entries
    try std.testing.expectEqual(@as(?[]const u8, null), try cache.put(1, "one"));
    try std.testing.expectEqual(@as(?[]const u8, null), try cache.put(2, "two"));
    try std.testing.expectEqual(@as(?[]const u8, null), try cache.put(3, "three"));
    try std.testing.expectEqual(@as(usize, 3), cache.count());
    try cache.validate();

    // Get existing keys
    try std.testing.expectEqualStrings("one", cache.get(1).?);
    try std.testing.expectEqualStrings("two", cache.get(2).?);
    try std.testing.expectEqualStrings("three", cache.get(3).?);
    try cache.validate();

    // Update existing key
    try std.testing.expectEqualStrings("one", (try cache.put(1, "ONE")).?);
    try std.testing.expectEqualStrings("ONE", cache.get(1).?);
    try cache.validate();
}

test "LRUCache: eviction on capacity overflow" {
    const Cache = LRUCache(u32, u32, std.hash_map.AutoContext(u32), null);
    var cache = Cache.init(std.testing.allocator, 3);
    defer cache.deinit();

    // Fill to capacity
    _ = try cache.put(1, 10);
    _ = try cache.put(2, 20);
    _ = try cache.put(3, 30);
    try cache.validate();

    // Insert 4th item → evicts LRU (key 1)
    _ = try cache.put(4, 40);
    try std.testing.expectEqual(@as(usize, 3), cache.count());
    try std.testing.expectEqual(@as(?u32, null), cache.get(1)); // evicted
    try std.testing.expectEqual(@as(?u32, 20), cache.get(2));
    try std.testing.expectEqual(@as(?u32, 30), cache.get(3));
    try std.testing.expectEqual(@as(?u32, 40), cache.get(4));
    try cache.validate();

    // Access key 2 → moves to head
    _ = cache.get(2);
    // Insert 5th item → evicts new LRU (key 3)
    _ = try cache.put(5, 50);
    try std.testing.expectEqual(@as(?u32, null), cache.get(3)); // evicted
    try std.testing.expectEqual(@as(?u32, 20), cache.get(2));
    try std.testing.expectEqual(@as(?u32, 40), cache.get(4));
    try std.testing.expectEqual(@as(?u32, 50), cache.get(5));
    try cache.validate();
}

test "LRUCache: LRU ordering" {
    const Cache = LRUCache(u32, u32, std.hash_map.AutoContext(u32), null);
    var cache = Cache.init(std.testing.allocator, 4);
    defer cache.deinit();

    _ = try cache.put(1, 10);
    _ = try cache.put(2, 20);
    _ = try cache.put(3, 30);
    _ = try cache.put(4, 40);

    // MRU → LRU: 4, 3, 2, 1
    var it = cache.iterator();
    try std.testing.expectEqual(@as(u32, 4), it.next().?.key);
    try std.testing.expectEqual(@as(u32, 3), it.next().?.key);
    try std.testing.expectEqual(@as(u32, 2), it.next().?.key);
    try std.testing.expectEqual(@as(u32, 1), it.next().?.key);
    try std.testing.expect(it.next() == null);

    // Access key 1 → moves to head
    _ = cache.get(1);
    // MRU → LRU: 1, 4, 3, 2
    it = cache.iterator();
    try std.testing.expectEqual(@as(u32, 1), it.next().?.key);
    try std.testing.expectEqual(@as(u32, 4), it.next().?.key);
    try std.testing.expectEqual(@as(u32, 3), it.next().?.key);
    try std.testing.expectEqual(@as(u32, 2), it.next().?.key);
    try cache.validate();
}

test "LRUCache: remove" {
    const Cache = LRUCache(u32, u32, std.hash_map.AutoContext(u32), null);
    var cache = Cache.init(std.testing.allocator, 3);
    defer cache.deinit();

    _ = try cache.put(1, 10);
    _ = try cache.put(2, 20);
    _ = try cache.put(3, 30);

    // Remove middle entry
    try std.testing.expectEqual(@as(?u32, 20), cache.remove(2));
    try std.testing.expectEqual(@as(usize, 2), cache.count());
    try std.testing.expectEqual(@as(?u32, null), cache.get(2));
    try cache.validate();

    // Remove head
    try std.testing.expectEqual(@as(?u32, 30), cache.remove(3));
    try std.testing.expectEqual(@as(usize, 1), cache.count());
    try cache.validate();

    // Remove tail (last entry)
    try std.testing.expectEqual(@as(?u32, 10), cache.remove(1));
    try std.testing.expectEqual(@as(usize, 0), cache.count());
    try std.testing.expect(cache.isEmpty());
    try cache.validate();

    // Remove non-existent
    try std.testing.expectEqual(@as(?u32, null), cache.remove(99));
}

test "LRUCache: peek does not update LRU" {
    const Cache = LRUCache(u32, u32, std.hash_map.AutoContext(u32), null);
    var cache = Cache.init(std.testing.allocator, 3);
    defer cache.deinit();

    _ = try cache.put(1, 10);
    _ = try cache.put(2, 20);
    _ = try cache.put(3, 30);

    // Peek does not change order
    try std.testing.expectEqual(@as(?u32, 10), cache.peek(1));
    var it = cache.iterator();
    try std.testing.expectEqual(@as(u32, 3), it.next().?.key); // still MRU
    try std.testing.expectEqual(@as(u32, 2), it.next().?.key);
    try std.testing.expectEqual(@as(u32, 1), it.next().?.key); // still LRU

    // Get does change order
    _ = cache.get(1);
    it = cache.iterator();
    try std.testing.expectEqual(@as(u32, 1), it.next().?.key); // now MRU
    try cache.validate();
}

test "LRUCache: clear" {
    const Cache = LRUCache(u32, u32, std.hash_map.AutoContext(u32), null);
    var cache = Cache.init(std.testing.allocator, 3);
    defer cache.deinit();

    _ = try cache.put(1, 10);
    _ = try cache.put(2, 20);
    _ = try cache.put(3, 30);

    cache.clear();
    try std.testing.expectEqual(@as(usize, 0), cache.count());
    try std.testing.expect(cache.isEmpty());
    try std.testing.expectEqual(@as(?u32, null), cache.get(1));
    try cache.validate();

    // Can reuse after clear
    _ = try cache.put(4, 40);
    try std.testing.expectEqual(@as(?u32, 40), cache.get(4));
    try cache.validate();
}

test "LRUCache: single entry" {
    const Cache = LRUCache(u32, u32, std.hash_map.AutoContext(u32), null);
    var cache = Cache.init(std.testing.allocator, 1);
    defer cache.deinit();

    _ = try cache.put(1, 10);
    try std.testing.expectEqual(@as(?u32, 10), cache.get(1));
    try cache.validate();

    // Inserting 2nd evicts the only entry
    _ = try cache.put(2, 20);
    try std.testing.expectEqual(@as(?u32, null), cache.get(1));
    try std.testing.expectEqual(@as(?u32, 20), cache.get(2));
    try cache.validate();
}

test "LRUCache: eviction callback" {
    const EvictTracker = struct {
        var evicted_keys: std.ArrayList(u32) = undefined;
        fn evict(key: u32, _: u32) void {
            evicted_keys.append(std.testing.allocator, key) catch unreachable;
        }
    };

    EvictTracker.evicted_keys = .{};
    defer EvictTracker.evicted_keys.deinit(std.testing.allocator);

    const Cache = LRUCache(u32, u32, std.hash_map.AutoContext(u32), EvictTracker.evict);
    var cache = Cache.init(std.testing.allocator, 2);
    defer cache.deinit();

    _ = try cache.put(1, 10);
    _ = try cache.put(2, 20);
    _ = try cache.put(3, 30); // evicts key 1

    try std.testing.expectEqual(@as(usize, 1), EvictTracker.evicted_keys.items.len);
    try std.testing.expectEqual(@as(u32, 1), EvictTracker.evicted_keys.items[0]);

    _ = try cache.put(4, 40); // evicts key 2
    try std.testing.expectEqual(@as(usize, 2), EvictTracker.evicted_keys.items.len);
    try std.testing.expectEqual(@as(u32, 2), EvictTracker.evicted_keys.items[1]);
}

test "LRUCache: stress test" {
    const Cache = LRUCache(u32, u32, std.hash_map.AutoContext(u32), null);
    var cache = Cache.init(std.testing.allocator, 100);
    defer cache.deinit();

    // Insert 1000 entries (should evict 900)
    for (0..1000) |i| {
        _ = try cache.put(@intCast(i), @intCast(i * 2));
    }
    try std.testing.expectEqual(@as(usize, 100), cache.count());
    try cache.validate();

    // Most recent 100 should still be present
    for (900..1000) |i| {
        try std.testing.expectEqual(@as(?u32, @intCast(i * 2)), cache.get(@intCast(i)));
    }
    // Oldest 900 should be evicted
    for (0..900) |i| {
        try std.testing.expectEqual(@as(?u32, null), cache.get(@intCast(i)));
    }
    try cache.validate();
}

test "LRUCache: contains" {
    const Cache = LRUCache(u32, u32, std.hash_map.AutoContext(u32), null);
    var cache = Cache.init(std.testing.allocator, 3);
    defer cache.deinit();

    try std.testing.expect(!cache.contains(1));
    _ = try cache.put(1, 10);
    try std.testing.expect(cache.contains(1));
    try std.testing.expect(!cache.contains(2));
}

test "LRUCache: memory leak detection" {
    const Cache = LRUCache(u32, u32, std.hash_map.AutoContext(u32), null);
    var cache = Cache.init(std.testing.allocator, 10);
    defer cache.deinit();

    // Fill and evict multiple times
    for (0..100) |i| {
        _ = try cache.put(@intCast(i), @intCast(i));
    }

    // Verify final state: cache should have exactly 10 items
    try std.testing.expectEqual(@as(usize, 10), cache.count());

    // Verify most recent item is present
    try std.testing.expect(cache.contains(99));

    // Verify oldest items were evicted (LRU behavior)
    try std.testing.expect(!cache.contains(0));
    try std.testing.expect(!cache.contains(10));

    try cache.validate();
    // std.testing.allocator will detect leaks automatically
}

test "LRUCache: pin prevents eviction" {
    const Cache = LRUCache(u32, u32, std.hash_map.AutoContext(u32), null);
    var cache = Cache.init(std.testing.allocator, 3);
    defer cache.deinit();

    // Fill cache
    _ = try cache.put(1, 10);
    _ = try cache.put(2, 20);
    _ = try cache.put(3, 30);

    // Pin the LRU entry (key 1)
    try cache.pin(1);
    try std.testing.expect(cache.isPinned(1));
    try std.testing.expect(!cache.isPinned(2));

    // Insert 4th item → should evict key 2 (not 1, because 1 is pinned)
    _ = try cache.put(4, 40);
    try std.testing.expectEqual(@as(usize, 3), cache.count());
    try std.testing.expectEqual(@as(?u32, 10), cache.get(1)); // pinned, not evicted
    try std.testing.expectEqual(@as(?u32, null), cache.get(2)); // evicted
    try std.testing.expectEqual(@as(?u32, 30), cache.get(3));
    try std.testing.expectEqual(@as(?u32, 40), cache.get(4));

    // Unpin key 1
    try cache.unpin(1);
    try std.testing.expect(!cache.isPinned(1));

    // Now key 1 can be evicted
    _ = try cache.put(5, 50);
    try std.testing.expectEqual(@as(?u32, null), cache.get(1)); // now evicted
    try cache.validate();
}

test "LRUCache: multiple pins stack" {
    const Cache = LRUCache(u32, u32, std.hash_map.AutoContext(u32), null);
    var cache = Cache.init(std.testing.allocator, 2);
    defer cache.deinit();

    _ = try cache.put(1, 10);
    _ = try cache.put(2, 20);

    // Pin key 1 twice and key 2 once
    try cache.pin(1);
    try cache.pin(1);
    try cache.pin(2);
    try std.testing.expect(cache.isPinned(1));
    try std.testing.expect(cache.isPinned(2));

    // Try to insert 3rd item → fails because both entries are pinned
    const result1 = cache.put(3, 30);
    try std.testing.expectError(error.NoPinnableEntries, result1);

    // Unpin key 2 → now unpinned
    try cache.unpin(2);
    try std.testing.expect(!cache.isPinned(2));
    try std.testing.expect(cache.isPinned(1)); // still pinned x2

    // Now can evict key 2
    _ = try cache.put(3, 30);
    try std.testing.expectEqual(@as(?u32, null), cache.get(2)); // evicted
    try std.testing.expectEqual(@as(?u32, 10), cache.get(1)); // still present, pinned x2
    try std.testing.expectEqual(@as(?u32, 30), cache.get(3)); // newly inserted

    // Unpin key 1 once → still pinned
    try cache.unpin(1);
    try std.testing.expect(cache.isPinned(1));

    // Unpin again → now unpinned
    try cache.unpin(1);
    try std.testing.expect(!cache.isPinned(1));

    // Now both can be evicted
    _ = try cache.put(4, 40);
    try cache.validate();
}

test "LRUCache: pin non-existent key" {
    const Cache = LRUCache(u32, u32, std.hash_map.AutoContext(u32), null);
    var cache = Cache.init(std.testing.allocator, 3);
    defer cache.deinit();

    // Pin non-existent key
    try std.testing.expectError(error.KeyNotFound, cache.pin(99));
}

test "LRUCache: unpin non-existent key" {
    const Cache = LRUCache(u32, u32, std.hash_map.AutoContext(u32), null);
    var cache = Cache.init(std.testing.allocator, 3);
    defer cache.deinit();

    _ = try cache.put(1, 10);

    // Unpin non-existent key
    try std.testing.expectError(error.KeyNotFound, cache.unpin(99));
}

test "LRUCache: unpin already unpinned" {
    const Cache = LRUCache(u32, u32, std.hash_map.AutoContext(u32), null);
    var cache = Cache.init(std.testing.allocator, 3);
    defer cache.deinit();

    _ = try cache.put(1, 10);

    // Unpin unpinned entry
    try std.testing.expectError(error.NotPinned, cache.unpin(1));
}

test "LRUCache: all entries pinned scenario" {
    const Cache = LRUCache(u32, u32, std.hash_map.AutoContext(u32), null);
    var cache = Cache.init(std.testing.allocator, 2);
    defer cache.deinit();

    _ = try cache.put(1, 10);
    _ = try cache.put(2, 20);

    // Pin all entries
    try cache.pin(1);
    try cache.pin(2);

    // Try to insert → should fail because all entries are pinned
    const result = cache.put(3, 30);
    try std.testing.expectError(error.NoPinnableEntries, result);

    // Cache should still have original 2 entries
    try std.testing.expectEqual(@as(usize, 2), cache.count());
    try std.testing.expectEqual(@as(?u32, 10), cache.get(1));
    try std.testing.expectEqual(@as(?u32, 20), cache.get(2));
    try cache.validate();
}
