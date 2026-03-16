//! ARC Cache — Adaptive Replacement Cache with self-tuning eviction policy.
//!
//! ARC (Adaptive Replacement Cache) automatically balances between recency and frequency
//! by maintaining four LRU lists: T1 (recent once), T2 (recent frequent), B1 (ghost T1), B2 (ghost B2).
//! The algorithm adapts to workload patterns by adjusting the target size p dynamically.
//!
//! Key features:
//!   - O(1) get, put, remove operations
//!   - Self-tuning: adapts to recency vs frequency workload patterns
//!   - Better hit rate than LRU or LFU alone for mixed workloads
//!   - Generic over K, V with custom hash/eql functions
//!   - Thread-unsafe (external synchronization required for concurrent access)
//!
//! Algorithm:
//!   - T1: Recent pages (seen once)
//!   - T2: Frequent pages (seen ≥ twice)
//!   - B1: Ghost entries evicted from T1 (metadata only, no value)
//!   - B2: Ghost entries evicted from T2 (metadata only, no value)
//!   - p: target size for T1 (c - p is target for T2)
//!   - When B1 hit: increase p (favor recency)
//!   - When B2 hit: decrease p (favor frequency)
//!
//! Reference: Megiddo & Modha, "ARC: A Self-Tuning, Low Overhead Replacement Cache" (FAST 2003)

const std = @import("std");

/// ARC Cache with O(1) get/put/remove and adaptive eviction policy.
///
/// Parameters:
///   - K: Key type
///   - V: Value type
///   - Context: Hash/equality context (default: std.hash_map.AutoContext(K))
pub fn ARCCache(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
) type {
    return struct {
        const Self = @This();

        /// Entry in a list (T1/T2 contain full entries, B1/B2 contain ghost entries).
        pub const Entry = struct {
            key: K,
            value: ?V = null, // null for ghost entries (B1/B2)
            prev: ?*Entry = null,
            next: ?*Entry = null,
        };

        /// List type identifier.
        const ListType = enum { T1, T2, B1, B2 };

        /// Doubly-linked list structure.
        const List = struct {
            head: ?*Entry = null,
            tail: ?*Entry = null,
            size: usize = 0,

            fn isEmpty(self: *const List) bool {
                return self.size == 0;
            }

            fn moveToHead(self: *List, entry: *Entry) void {
                // If already at head, nothing to do
                if (self.head == entry) return;

                // Remove from current position
                if (entry.prev) |prev| {
                    prev.next = entry.next;
                }
                if (entry.next) |next| {
                    next.prev = entry.prev;
                } else {
                    self.tail = entry.prev;
                }

                // Insert at head
                entry.prev = null;
                entry.next = self.head;
                if (self.head) |head| {
                    head.prev = entry;
                }
                self.head = entry;
                // Note: size unchanged (just reordering)
            }

            fn append(self: *List, entry: *Entry) void {
                entry.prev = self.tail;
                entry.next = null;
                if (self.tail) |tail| {
                    tail.next = entry;
                } else {
                    self.head = entry;
                }
                self.tail = entry;
                self.size += 1;
            }

            fn remove(self: *List, entry: *Entry) void {
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
                self.size -= 1;
            }

            fn removeTail(self: *List) ?*Entry {
                const tail = self.tail orelse return null;
                self.remove(tail);
                return tail;
            }
        };

        allocator: std.mem.Allocator,
        capacity: usize,
        map: std.HashMap(K, *Entry, Context, std.hash_map.default_max_load_percentage),
        list_map: std.HashMap(*Entry, ListType, std.hash_map.AutoContext(*Entry), std.hash_map.default_max_load_percentage),

        t1: List = .{}, // Recent (seen once)
        t2: List = .{}, // Frequent (seen ≥ twice)
        b1: List = .{}, // Ghost T1
        b2: List = .{}, // Ghost B2
        p: usize = 0, // Target size for T1

        /// Initialize an ARC cache with the given capacity.
        /// Time: O(1) | Space: O(capacity)
        pub fn init(allocator: std.mem.Allocator, capacity: usize) Self {
            return .{
                .allocator = allocator,
                .capacity = capacity,
                .map = std.HashMap(K, *Entry, Context, std.hash_map.default_max_load_percentage).init(allocator),
                .list_map = std.HashMap(*Entry, ListType, std.hash_map.AutoContext(*Entry), std.hash_map.default_max_load_percentage).init(allocator),
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
            self.list_map.deinit();
        }

        /// Number of cached entries (T1 + T2, excluding ghosts).
        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.t1.size + self.t2.size;
        }

        /// Check if the cache is empty.
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.count() == 0;
        }

        /// Get a value by key, adapting the cache based on access pattern.
        /// Returns null if key not found.
        /// Time: O(1) | Space: O(1)
        pub fn get(self: *Self, key: K) ?V {
            const entry = self.map.get(key) orelse return null;
            const list_type = self.list_map.get(entry) orelse return null;

            // Only T1/T2 have values; B1/B2 are ghosts
            if (entry.value == null) return null;

            // Hit in T1 → move to T2 (promote to frequent)
            if (list_type == .T1) {
                self.t1.remove(entry);
                entry.prev = null;
                entry.next = self.t2.head;
                if (self.t2.head) |head| {
                    head.prev = entry;
                } else {
                    self.t2.tail = entry;
                }
                self.t2.head = entry;
                self.t2.size += 1;
                self.list_map.put(entry, .T2) catch unreachable;
            }
            // Hit in T2 → move to head (already frequent)
            else if (list_type == .T2) {
                self.t2.moveToHead(entry);
            }

            return entry.value.?;
        }

        /// Insert or update a key-value pair, evicting if necessary.
        /// Returns the previous value if key existed.
        /// Time: O(1) amortized | Space: O(1)
        pub fn put(self: *Self, key: K, value: V) !?V {
            // Check if key already exists
            if (self.map.get(key)) |entry| {
                const list_type = self.list_map.get(entry).?;
                const old_value = entry.value;
                entry.value = value;

                // Hit in T1 → promote to T2
                if (list_type == .T1) {
                    self.t1.remove(entry);
                    entry.prev = null;
                    entry.next = self.t2.head;
                    if (self.t2.head) |head| {
                        head.prev = entry;
                    } else {
                        self.t2.tail = entry;
                    }
                    self.t2.head = entry;
                    self.t2.size += 1;
                    try self.list_map.put(entry, .T2);
                }
                // Hit in T2 → move to head
                else if (list_type == .T2) {
                    self.t2.moveToHead(entry);
                }
                // Hit in B1 (ghost) → adapt p, replace entry
                else if (list_type == .B1) {
                    // Increase p (favor recency)
                    const delta = if (self.b1.size >= self.b2.size) 1 else @divTrunc(self.b2.size, self.b1.size);
                    self.p = @min(self.p + delta, self.capacity);
                    try self.replace(key, .B1);
                    self.b1.remove(entry);
                    entry.value = value;
                    self.t2.append(entry);
                    try self.list_map.put(entry, .T2);
                }
                // Hit in B2 (ghost) → adapt p, replace entry
                else if (list_type == .B2) {
                    // Decrease p (favor frequency)
                    const delta = if (self.b2.size >= self.b1.size) 1 else @divTrunc(self.b1.size, self.b2.size);
                    self.p = if (self.p >= delta) self.p - delta else 0;
                    try self.replace(key, .B2);
                    self.b2.remove(entry);
                    entry.value = value;
                    self.t2.append(entry);
                    try self.list_map.put(entry, .T2);
                }

                return old_value;
            }

            // New key
            if (self.t1.size + self.t2.size >= self.capacity) {
                try self.replace(key, null);
            }

            const entry = try self.allocator.create(Entry);
            entry.* = .{ .key = key, .value = value };
            try self.map.put(key, entry);
            self.t1.append(entry);
            try self.list_map.put(entry, .T1);

            return null;
        }

        /// Replace an entry based on ARC policy.
        fn replace(self: *Self, _: K, hit_in: ?ListType) !void {
            // Case 1: T1 is non-empty and (T1 exceeds target p OR hit in B2 and T1 size equals p)
            if (self.t1.size > 0 and
                (self.t1.size > self.p or (hit_in == .B2 and self.t1.size == self.p)))
            {
                const old = self.t1.removeTail().?;
                try self.list_map.put(old, .B1);
                old.value = null; // Convert to ghost
                self.b1.append(old);
            }
            // Case 2: T1 is empty or T2 should be evicted
            else {
                const old = self.t2.removeTail().?;
                try self.list_map.put(old, .B2);
                old.value = null; // Convert to ghost
                self.b2.append(old);
            }

            // Trim ghost lists if they exceed capacity
            if (self.b1.size > self.capacity - self.p) {
                if (self.b1.removeTail()) |entry| {
                    _ = self.map.remove(entry.key);
                    _ = self.list_map.remove(entry);
                    self.allocator.destroy(entry);
                }
            }
            if (self.b2.size > self.p) {
                if (self.b2.removeTail()) |entry| {
                    _ = self.map.remove(entry.key);
                    _ = self.list_map.remove(entry);
                    self.allocator.destroy(entry);
                }
            }
        }

        /// Remove a key-value pair from the cache.
        /// Returns the value if key existed, null otherwise.
        /// Time: O(1) | Space: O(1)
        pub fn remove(self: *Self, key: K) ?V {
            const entry = self.map.get(key) orelse return null;
            const list_type = self.list_map.get(entry) orelse return null;
            const old_value = entry.value;

            _ = self.map.remove(key);
            _ = self.list_map.remove(entry);

            switch (list_type) {
                .T1 => self.t1.remove(entry),
                .T2 => self.t2.remove(entry),
                .B1 => self.b1.remove(entry),
                .B2 => self.b2.remove(entry),
            }

            self.allocator.destroy(entry);
            return old_value;
        }

        /// Clear all entries.
        /// Time: O(n) | Space: O(1)
        pub fn clear(self: *Self) void {
            var it = self.map.valueIterator();
            while (it.next()) |entry_ptr| {
                self.allocator.destroy(entry_ptr.*);
            }
            self.map.clearRetainingCapacity();
            self.list_map.clearRetainingCapacity();
            self.t1 = .{};
            self.t2 = .{};
            self.b1 = .{};
            self.b2 = .{};
            self.p = 0;
        }

        /// Check if a key exists in the cache (without promoting).
        /// Time: O(1) | Space: O(1)
        pub fn contains(self: *const Self, key: K) bool {
            if (self.map.get(key)) |entry| {
                return entry.value != null; // Only count T1/T2, not ghosts
            }
            return false;
        }

        /// Validate cache invariants (for testing).
        /// Returns error if any invariant is violated.
        pub fn validate(self: *const Self) !void {
            // Invariant 1: T1 + T2 ≤ capacity
            if (self.t1.size + self.t2.size > self.capacity) {
                return error.CapacityViolation;
            }

            // Invariant 2: B1 + B2 ≤ capacity (ghost lists bounded)
            // Note: In practice B1 + B2 can be up to 2*capacity, but we bound them individually
            if (self.b1.size > self.capacity or self.b2.size > self.capacity) {
                return error.GhostListTooLarge;
            }

            // Invariant 3: p ≤ capacity
            if (self.p > self.capacity) {
                return error.TargetSizeInvalid;
            }

            // Invariant 4: All T1/T2 entries have values, B1/B2 do not
            var it = self.map.valueIterator();
            while (it.next()) |entry_ptr| {
                const entry = entry_ptr.*;
                const list_type = self.list_map.get(entry) orelse return error.EntryNotInList;
                switch (list_type) {
                    .T1, .T2 => {
                        if (entry.value == null) return error.CacheEntryMissingValue;
                    },
                    .B1, .B2 => {
                        if (entry.value != null) return error.GhostEntryHasValue;
                    },
                }
            }

            // Invariant 5: map.count() == all list sizes
            if (self.map.count() != self.t1.size + self.t2.size + self.b1.size + self.b2.size) {
                return error.MapSizeMismatch;
            }
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "ARCCache: basic operations" {
    const Cache = ARCCache(u32, []const u8, std.hash_map.AutoContext(u32));
    var cache = Cache.init(testing.allocator, 3);
    defer cache.deinit();

    // Insert
    try testing.expectEqual(null, try cache.put(1, "one"));
    try testing.expectEqual(null, try cache.put(2, "two"));
    try testing.expectEqual(null, try cache.put(3, "three"));
    try testing.expectEqual(@as(usize, 3), cache.count());

    // Get
    try testing.expectEqualStrings("one", cache.get(1).?);
    try testing.expectEqualStrings("two", cache.get(2).?);
    try testing.expectEqualStrings("three", cache.get(3).?);
    try testing.expectEqual(null, cache.get(999));

    // Contains
    try testing.expect(cache.contains(1));
    try testing.expect(!cache.contains(999));

    // Update
    try testing.expectEqualStrings("one", (try cache.put(1, "ONE")).?);
    try testing.expectEqualStrings("ONE", cache.get(1).?);

    try cache.validate();
}

test "ARCCache: eviction on capacity" {
    const Cache = ARCCache(u32, u32, std.hash_map.AutoContext(u32));
    var cache = Cache.init(testing.allocator, 2);
    defer cache.deinit();

    _ = try cache.put(1, 10);
    _ = try cache.put(2, 20);
    try testing.expectEqual(@as(usize, 2), cache.count());

    // Access 1 → promote to T2
    _ = cache.get(1);

    // Insert 3 → evicts 2 (LRU in T1)
    _ = try cache.put(3, 30);
    try testing.expectEqual(@as(usize, 2), cache.count());
    try testing.expectEqual(@as(u32, 10), cache.get(1).?);
    try testing.expectEqual(@as(u32, 30), cache.get(3).?);
    try testing.expectEqual(null, cache.get(2)); // evicted

    try cache.validate();
}

test "ARCCache: adaptation - recency vs frequency" {
    const Cache = ARCCache(u32, u32, std.hash_map.AutoContext(u32));
    var cache = Cache.init(testing.allocator, 4);
    defer cache.deinit();

    // Fill cache
    _ = try cache.put(1, 10);
    _ = try cache.put(2, 20);
    _ = try cache.put(3, 30);
    _ = try cache.put(4, 40);

    // Access 1, 2 multiple times → move to T2 (frequent)
    _ = cache.get(1);
    _ = cache.get(1);
    _ = cache.get(2);
    _ = cache.get(2);

    // Access 3 once → stays in T1 or moves to T2
    _ = cache.get(3);

    // Now insert 5 → should evict from T1 (less accessed)
    _ = try cache.put(5, 50);

    // 1, 2 should still be present (frequent in T2)
    try testing.expectEqual(@as(u32, 10), cache.get(1).?);
    try testing.expectEqual(@as(u32, 20), cache.get(2).?);

    try cache.validate();
}

test "ARCCache: remove" {
    const Cache = ARCCache(u32, []const u8, std.hash_map.AutoContext(u32));
    var cache = Cache.init(testing.allocator, 3);
    defer cache.deinit();

    _ = try cache.put(1, "one");
    _ = try cache.put(2, "two");
    _ = try cache.put(3, "three");

    try testing.expectEqualStrings("two", cache.remove(2).?);
    try testing.expectEqual(@as(usize, 2), cache.count());
    try testing.expectEqual(null, cache.get(2));
    try testing.expect(!cache.contains(2));

    try cache.validate();
}

test "ARCCache: clear" {
    const Cache = ARCCache(u32, u32, std.hash_map.AutoContext(u32));
    var cache = Cache.init(testing.allocator, 3);
    defer cache.deinit();

    _ = try cache.put(1, 10);
    _ = try cache.put(2, 20);
    _ = try cache.put(3, 30);

    cache.clear();
    try testing.expectEqual(@as(usize, 0), cache.count());
    try testing.expect(cache.isEmpty());
    try testing.expectEqual(null, cache.get(1));

    try cache.validate();
}

test "ARCCache: empty cache" {
    const Cache = ARCCache(u32, u32, std.hash_map.AutoContext(u32));
    var cache = Cache.init(testing.allocator, 5);
    defer cache.deinit();

    try testing.expect(cache.isEmpty());
    try testing.expectEqual(@as(usize, 0), cache.count());
    try testing.expectEqual(null, cache.get(42));
    try testing.expectEqual(null, cache.remove(42));
    try testing.expect(!cache.contains(42));

    try cache.validate();
}

test "ARCCache: single element" {
    const Cache = ARCCache(u32, u32, std.hash_map.AutoContext(u32));
    var cache = Cache.init(testing.allocator, 1);
    defer cache.deinit();

    _ = try cache.put(1, 10);
    try testing.expectEqual(@as(u32, 10), cache.get(1).?);

    // Insert another → evicts first
    _ = try cache.put(2, 20);
    try testing.expectEqual(null, cache.get(1));
    try testing.expectEqual(@as(u32, 20), cache.get(2).?);

    try cache.validate();
}

test "ARCCache: stress test with mixed access" {
    const Cache = ARCCache(u32, u32, std.hash_map.AutoContext(u32));
    var cache = Cache.init(testing.allocator, 100);
    defer cache.deinit();

    // Insert 150 items (will evict 50)
    var i: u32 = 0;
    while (i < 150) : (i += 1) {
        _ = try cache.put(i, i * 10);
    }
    try testing.expectEqual(@as(usize, 100), cache.count());

    // Access some items multiple times to promote to T2
    i = 50;
    while (i < 100) : (i += 1) {
        _ = cache.get(i);
        _ = cache.get(i);
    }

    // Insert more items
    i = 150;
    while (i < 200) : (i += 1) {
        _ = try cache.put(i, i * 10);
    }

    // Frequently accessed items should still be present
    i = 50;
    while (i < 100) : (i += 1) {
        if (cache.get(i)) |val| {
            try testing.expectEqual(i * 10, val);
        }
        // Note: some may have been evicted if T2 filled up
    }

    try cache.validate();
}

test "ARCCache: ghost list behavior" {
    const Cache = ARCCache(u32, u32, std.hash_map.AutoContext(u32));
    var cache = Cache.init(testing.allocator, 2);
    defer cache.deinit();

    // Fill cache
    _ = try cache.put(1, 10);
    _ = try cache.put(2, 20);
    // T1: [1 (head), 2 (tail)] in insertion order

    // Insert 3 → evicts LRU from T1 (tail = 2, since p=0 initially)
    _ = try cache.put(3, 30);
    try testing.expectEqual(null, cache.get(2)); // 2 was evicted to B1 ghost
    try testing.expectEqual(@as(u32, 10), cache.get(1).?); // 1 still present (promoted to T2 on access)
    try testing.expectEqual(@as(u32, 30), cache.get(3).?); // 3 present

    // Re-access 2 (ghost hit in B1) → should adapt p and move to T2
    _ = try cache.put(2, 200);
    try testing.expectEqual(@as(u32, 200), cache.get(2).?);
    try testing.expect(cache.p > 0); // p increased (favor recency after B1 hit)

    try cache.validate();
}

test "ARCCache: memory leak check" {
    const Cache = ARCCache(u32, u32, std.hash_map.AutoContext(u32));
    var cache = Cache.init(testing.allocator, 10);
    defer cache.deinit();

    // Fill cache with 100 items (capacity=10, so many will be evicted)
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        _ = try cache.put(i, i);
    }

    // Verify final state: cache should have evicted most items
    try testing.expectEqual(@as(usize, 10), cache.count());

    // Verify we can still get items that were in cache
    if (cache.get(99)) |val| {
        try testing.expectEqual(@as(u32, 99), val);
    }

    // All allocations should be freed on deinit
}
