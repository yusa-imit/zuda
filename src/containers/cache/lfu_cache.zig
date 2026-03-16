//! LFU Cache — Least Frequently Used eviction cache with O(1) operations.
//!
//! Tracks access frequency for each key and evicts the least frequently accessed entry.
//! Uses frequency buckets with doubly-linked lists for O(1) operations.
//! When multiple entries have the same frequency, evicts the least recently used (LRU tiebreaker).
//!
//! Key features:
//!   - O(1) get, put, remove operations
//!   - Frequency tracking with LRU tiebreaker for equal frequencies
//!   - Optional eviction callback for cleanup operations
//!   - Generic over K, V with custom hash/eql functions
//!   - Thread-unsafe (external synchronization required for concurrent access)
//!
//! Consumer use cases:
//!   - zoltraak: cache eviction for hot key detection
//!   - General: CDN caching, database query cache, API rate limiting

const std = @import("std");

/// LFU Cache with O(1) get/put/remove operations using frequency buckets.
///
/// Parameters:
///   - K: Key type
///   - V: Value type
///   - Context: Hash/equality context (default: std.hash_map.AutoContext(K))
///   - evictFn: Optional callback when entry is evicted (for cleanup)
pub fn LFUCache(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime evictFn: ?fn (key: K, value: V) void,
) type {
    return struct {
        const Self = @This();

        /// Entry in the LFU cache.
        pub const Entry = struct {
            key: K,
            value: V,
            freq: usize = 1, // Access frequency
            prev: ?*Entry = null, // Within frequency bucket
            next: ?*Entry = null,
        };

        /// Frequency bucket — doubly-linked list of entries with the same frequency.
        const FreqBucket = struct {
            freq: usize,
            head: ?*Entry = null,
            tail: ?*Entry = null,
            prev: ?*FreqBucket = null,
            next: ?*FreqBucket = null,
        };

        pub const Iterator = struct {
            current_bucket: ?*FreqBucket,
            current_entry: ?*Entry,

            /// Returns the next entry in frequency order (lowest → highest).
            /// Time: O(1) | Space: O(1)
            pub fn next(self: *Iterator) ?struct { key: K, value: V, freq: usize } {
                while (self.current_bucket != null) {
                    if (self.current_entry) |entry| {
                        self.current_entry = entry.next;
                        return .{ .key = entry.key, .value = entry.value, .freq = entry.freq };
                    }
                    // Move to next bucket
                    self.current_bucket = self.current_bucket.?.next;
                    if (self.current_bucket) |bucket| {
                        self.current_entry = bucket.head;
                    }
                }
                return null;
            }
        };

        allocator: std.mem.Allocator,
        capacity: usize,
        map: std.HashMap(K, *Entry, Context, std.hash_map.default_max_load_percentage),
        // Frequency bucket list: head = min frequency, tail = max frequency
        freq_head: ?*FreqBucket = null,
        min_freq: usize = 1, // Track minimum frequency for O(1) eviction

        /// Initialize an LFU cache with the given capacity.
        /// Time: O(1) | Space: O(capacity)
        pub fn init(allocator: std.mem.Allocator, capacity: usize) Self {
            return .{
                .allocator = allocator,
                .capacity = capacity,
                .map = std.HashMap(K, *Entry, Context, std.hash_map.default_max_load_percentage).init(allocator),
            };
        }

        /// Free all resources.
        /// Time: O(n + f) where f is number of unique frequencies | Space: O(1)
        pub fn deinit(self: *Self) void {
            // Free all entries
            var it = self.map.valueIterator();
            while (it.next()) |entry_ptr| {
                self.allocator.destroy(entry_ptr.*);
            }
            self.map.deinit();

            // Free all frequency buckets
            var bucket = self.freq_head;
            while (bucket) |b| {
                const next_bucket = b.next;
                self.allocator.destroy(b);
                bucket = next_bucket;
            }
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

        /// Get a value by key, incrementing its frequency.
        /// Returns null if key not found.
        /// Time: O(1) | Space: O(1)
        pub fn get(self: *Self, key: K) !?V {
            const entry = self.map.get(key) orelse return null;
            try self.incrementFreq(entry);
            return entry.value;
        }

        /// Get a value by key without updating frequency (peek).
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

        /// Get the frequency of a key.
        /// Returns null if key not found.
        /// Time: O(1) | Space: O(1)
        pub fn getFreq(self: *const Self, key: K) ?usize {
            const entry = self.map.get(key) orelse return null;
            return entry.freq;
        }

        /// Insert or update a key-value pair.
        /// If key exists, updates value and increments frequency.
        /// If capacity is exceeded, evicts the least frequently used entry.
        /// Returns the old value if key existed, null otherwise.
        /// Time: O(1) amortized | Space: O(1)
        pub fn put(self: *Self, key: K, value: V) !?V {
            // Update existing entry
            if (self.map.getPtr(key)) |entry_ptr| {
                const old_value = entry_ptr.*.value;
                entry_ptr.*.value = value;
                try self.incrementFreq(entry_ptr.*);
                return old_value;
            }

            // Evict LFU if at capacity
            if (self.map.count() >= self.capacity) {
                try self.evictLFU();
            }

            // Create new entry with freq = 1
            const entry = try self.allocator.create(Entry);
            entry.* = .{
                .key = key,
                .value = value,
                .freq = 1,
            };

            try self.map.put(key, entry);
            try self.addToFreqBucket(entry, 1);
            self.min_freq = 1; // New entry always has freq = 1
            return null;
        }

        /// Remove a key from the cache.
        /// Returns the removed value if key existed, null otherwise.
        /// Time: O(1) amortized | Space: O(1)
        pub fn remove(self: *Self, key: K) ?V {
            const entry = self.map.fetchRemove(key) orelse return null;
            const value = entry.value.value;
            self.removeFromFreqBucket(entry.value);
            self.allocator.destroy(entry.value);
            return value;
        }

        /// Clear all entries from the cache.
        /// Time: O(n) | Space: O(1)
        pub fn clear(self: *Self) void {
            var it = self.map.valueIterator();
            while (it.next()) |entry_ptr| {
                if (evictFn) |f| {
                    f(entry_ptr.*.key, entry_ptr.*.value);
                }
                self.allocator.destroy(entry_ptr.*);
            }
            self.map.clearRetainingCapacity();

            // Free frequency buckets
            var bucket = self.freq_head;
            while (bucket) |b| {
                const next_bucket = b.next;
                self.allocator.destroy(b);
                bucket = next_bucket;
            }
            self.freq_head = null;
            self.min_freq = 1;
        }

        /// Iterate over all entries in frequency order (lowest → highest).
        /// Time: O(1) | Space: O(1)
        pub fn iterator(self: *const Self) Iterator {
            const first_entry = if (self.freq_head) |bucket| bucket.head else null;
            return .{
                .current_bucket = self.freq_head,
                .current_entry = first_entry,
            };
        }

        /// Validate internal invariants.
        /// Time: O(n + f) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            // Check map count consistency
            const map_count = self.map.count();

            // Count entries via frequency buckets
            var bucket_count: usize = 0;
            var bucket = self.freq_head;
            var prev_freq: usize = 0;
            while (bucket) |b| {
                // Frequencies must be strictly increasing
                if (prev_freq > 0 and b.freq <= prev_freq) {
                    return error.InvalidFrequencyOrder;
                }
                prev_freq = b.freq;

                var entry = b.head;
                while (entry) |e| {
                    bucket_count += 1;
                    // Entry frequency must match bucket frequency
                    if (e.freq != b.freq) {
                        return error.FrequencyMismatch;
                    }
                    entry = e.next;
                }
                bucket = b.next;
            }

            if (map_count != bucket_count) {
                return error.CountMismatch;
            }

            // If cache is not empty, min_freq bucket must exist and have entries
            if (map_count > 0) {
                var found_min_freq = false;
                bucket = self.freq_head;
                while (bucket) |b| {
                    if (b.freq == self.min_freq and b.head != null) {
                        found_min_freq = true;
                        break;
                    }
                    bucket = b.next;
                }
                if (!found_min_freq) {
                    return error.InvalidMinFreq;
                }
            }
        }

        // -- Private Helpers --

        /// Evict the least frequently used entry (LRU tiebreaker).
        /// Time: O(1) | Space: O(1)
        fn evictLFU(self: *Self) !void {
            // Find the bucket with min_freq
            var bucket = self.freq_head;
            while (bucket) |b| {
                if (b.freq == self.min_freq) {
                    // Evict the tail (LRU within same frequency)
                    const victim = b.tail orelse return error.InvalidState;
                    if (evictFn) |f| {
                        f(victim.key, victim.value);
                    }
                    _ = self.map.remove(victim.key);
                    self.removeFromFreqBucket(victim);
                    self.allocator.destroy(victim);
                    return;
                }
                bucket = b.next;
            }
            return error.MinFreqBucketNotFound;
        }

        /// Increment the frequency of an entry and move it to the next frequency bucket.
        /// Time: O(1) amortized | Space: O(1)
        fn incrementFreq(self: *Self, entry: *Entry) !void {
            const old_freq = entry.freq;
            const new_freq = old_freq + 1;

            // Remove from old frequency bucket
            self.removeFromFreqBucket(entry);

            // Update min_freq if necessary
            if (old_freq == self.min_freq) {
                // Check if old bucket is now empty
                var bucket = self.freq_head;
                var old_bucket_empty = true;
                while (bucket) |b| {
                    if (b.freq == old_freq and b.head != null) {
                        old_bucket_empty = false;
                        break;
                    }
                    bucket = b.next;
                }
                if (old_bucket_empty) {
                    self.min_freq = new_freq;
                }
            }

            // Add to new frequency bucket
            entry.freq = new_freq;
            try self.addToFreqBucket(entry, new_freq);
        }

        /// Add an entry to the head of a frequency bucket (most recently used within that frequency).
        /// Time: O(1) amortized | Space: O(1)
        fn addToFreqBucket(self: *Self, entry: *Entry, freq: usize) !void {
            // Find or create the frequency bucket
            var bucket = self.freq_head;
            var prev_bucket: ?*FreqBucket = null;

            while (bucket) |b| {
                if (b.freq == freq) {
                    // Bucket exists, add entry to head
                    entry.next = b.head;
                    entry.prev = null;
                    if (b.head) |head| {
                        head.prev = entry;
                    } else {
                        b.tail = entry;
                    }
                    b.head = entry;
                    return;
                } else if (b.freq > freq) {
                    // Need to insert new bucket before this one
                    break;
                }
                prev_bucket = b;
                bucket = b.next;
            }

            // Create new bucket
            const new_bucket = try self.allocator.create(FreqBucket);
            new_bucket.* = .{
                .freq = freq,
                .head = entry,
                .tail = entry,
                .prev = prev_bucket,
                .next = bucket,
            };
            entry.prev = null;
            entry.next = null;

            // Link into bucket list
            if (prev_bucket) |pb| {
                pb.next = new_bucket;
            } else {
                self.freq_head = new_bucket;
            }
            if (bucket) |b| {
                b.prev = new_bucket;
            }
        }

        /// Remove an entry from its frequency bucket.
        /// Time: O(1) | Space: O(1)
        fn removeFromFreqBucket(self: *Self, entry: *Entry) void {
            const freq = entry.freq;

            // Find the bucket
            var bucket = self.freq_head;
            while (bucket) |b| {
                if (b.freq == freq) {
                    // Remove entry from bucket's doubly-linked list
                    if (entry.prev) |prev| {
                        prev.next = entry.next;
                    } else {
                        b.head = entry.next;
                    }

                    if (entry.next) |next| {
                        next.prev = entry.prev;
                    } else {
                        b.tail = entry.prev;
                    }

                    // If bucket is now empty, remove it from bucket list
                    if (b.head == null) {
                        if (b.prev) |prev| {
                            prev.next = b.next;
                        } else {
                            self.freq_head = b.next;
                        }
                        if (b.next) |next| {
                            next.prev = b.prev;
                        }
                        self.allocator.destroy(b);
                    }
                    return;
                }
                bucket = b.next;
            }
        }
    };
}

// -- Tests --

test "LFUCache: basic operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const Cache = LFUCache(i32, []const u8, std.hash_map.AutoContext(i32), null);
    var cache = Cache.init(allocator, 3);
    defer cache.deinit();

    // Insert entries
    try testing.expectEqual(@as(?[]const u8, null), try cache.put(1, "one"));
    try testing.expectEqual(@as(?[]const u8, null), try cache.put(2, "two"));
    try testing.expectEqual(@as(?[]const u8, null), try cache.put(3, "three"));
    try testing.expectEqual(@as(usize, 3), cache.count());

    // Get entries (increments frequency)
    try testing.expectEqualStrings("one", (try cache.get(1)).?);
    try testing.expectEqualStrings("two", (try cache.get(2)).?);
    try testing.expectEqualStrings("one", (try cache.get(1)).?); // freq(1) = 3

    // Validate invariants
    try cache.validate();
}

test "LFUCache: LFU eviction" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const Cache = LFUCache(i32, i32, std.hash_map.AutoContext(i32), null);
    var cache = Cache.init(allocator, 2);
    defer cache.deinit();

    _ = try cache.put(1, 100);
    _ = try cache.put(2, 200);

    // Access key 1 multiple times (freq(1) = 3, freq(2) = 1)
    _ = try cache.get(1);
    _ = try cache.get(1);

    // Insert key 3 — should evict key 2 (least frequently used)
    _ = try cache.put(3, 300);
    try testing.expectEqual(@as(?i32, null), try cache.get(2)); // Evicted
    try testing.expectEqual(@as(i32, 100), (try cache.get(1)).?);
    try testing.expectEqual(@as(i32, 300), (try cache.get(3)).?);

    try cache.validate();
}

test "LFUCache: LRU tiebreaker" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const Cache = LFUCache(i32, i32, std.hash_map.AutoContext(i32), null);
    var cache = Cache.init(allocator, 3);
    defer cache.deinit();

    // All entries have freq = 1
    _ = try cache.put(1, 100);
    _ = try cache.put(2, 200);
    _ = try cache.put(3, 300);

    // Insert key 4 — should evict key 1 (LRU tiebreaker)
    _ = try cache.put(4, 400);
    try testing.expectEqual(@as(?i32, null), try cache.get(1)); // Evicted (LRU)
    try testing.expectEqual(@as(i32, 200), (try cache.get(2)).?);
    try testing.expectEqual(@as(i32, 300), (try cache.get(3)).?);
    try testing.expectEqual(@as(i32, 400), (try cache.get(4)).?);

    try cache.validate();
}

test "LFUCache: update existing key" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const Cache = LFUCache(i32, []const u8, std.hash_map.AutoContext(i32), null);
    var cache = Cache.init(allocator, 3);
    defer cache.deinit();

    _ = try cache.put(1, "one");
    _ = try cache.get(1); // freq(1) = 2

    // Update key 1 — should increment frequency to 3
    const old = try cache.put(1, "ONE");
    try testing.expectEqualStrings("one", old.?);
    try testing.expectEqualStrings("ONE", (try cache.get(1)).?);
    try testing.expectEqual(@as(usize, 4), cache.getFreq(1).?); // freq = 4 (initial + get + put + get)

    try cache.validate();
}

test "LFUCache: remove" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const Cache = LFUCache(i32, i32, std.hash_map.AutoContext(i32), null);
    var cache = Cache.init(allocator, 3);
    defer cache.deinit();

    _ = try cache.put(1, 100);
    _ = try cache.put(2, 200);

    const removed = cache.remove(1);
    try testing.expectEqual(@as(i32, 100), removed.?);
    try testing.expectEqual(@as(?i32, null), try cache.get(1));
    try testing.expectEqual(@as(usize, 1), cache.count());

    try cache.validate();
}

test "LFUCache: clear" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const Cache = LFUCache(i32, i32, std.hash_map.AutoContext(i32), null);
    var cache = Cache.init(allocator, 3);
    defer cache.deinit();

    _ = try cache.put(1, 100);
    _ = try cache.put(2, 200);
    cache.clear();

    try testing.expectEqual(@as(usize, 0), cache.count());
    try testing.expect(cache.isEmpty());
}

test "LFUCache: peek does not update frequency" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const Cache = LFUCache(i32, i32, std.hash_map.AutoContext(i32), null);
    var cache = Cache.init(allocator, 2);
    defer cache.deinit();

    _ = try cache.put(1, 100);
    _ = try cache.put(2, 200);

    // Peek does not increment frequency
    try testing.expectEqual(@as(i32, 100), cache.peek(1).?);
    try testing.expectEqual(@as(usize, 1), cache.getFreq(1).?);

    _ = try cache.get(1); // freq(1) = 2
    try testing.expectEqual(@as(usize, 2), cache.getFreq(1).?);

    try cache.validate();
}

test "LFUCache: iterator" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const Cache = LFUCache(i32, i32, std.hash_map.AutoContext(i32), null);
    var cache = Cache.init(allocator, 5);
    defer cache.deinit();

    _ = try cache.put(1, 100);
    _ = try cache.put(2, 200);
    _ = try cache.put(3, 300);

    // Increment frequencies: freq(1) = 3, freq(2) = 2, freq(3) = 1
    _ = try cache.get(1);
    _ = try cache.get(1);
    _ = try cache.get(2);

    // Iterator should yield in frequency order (lowest → highest)
    var it = cache.iterator();
    var count: usize = 0;
    while (it.next()) |entry| {
        count += 1;
        if (count == 1) {
            try testing.expectEqual(@as(i32, 3), entry.key);
            try testing.expectEqual(@as(usize, 1), entry.freq);
        } else if (count == 2) {
            try testing.expectEqual(@as(i32, 2), entry.key);
            try testing.expectEqual(@as(usize, 2), entry.freq);
        } else if (count == 3) {
            try testing.expectEqual(@as(i32, 1), entry.key);
            try testing.expectEqual(@as(usize, 3), entry.freq);
        }
    }
    try testing.expectEqual(@as(usize, 3), count);
}

test "LFUCache: stress test" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const Cache = LFUCache(i32, i32, std.hash_map.AutoContext(i32), null);
    var cache = Cache.init(allocator, 100);
    defer cache.deinit();

    // Insert 200 entries (will evict 100)
    for (0..200) |i| {
        _ = try cache.put(@intCast(i), @intCast(i * 10));
    }
    try testing.expectEqual(@as(usize, 100), cache.count());

    // Access some entries to vary frequencies
    for (0..50) |i| {
        _ = try cache.get(@intCast(i + 100)); // freq = 2
    }

    // Validate invariants
    try cache.validate();
}

test "LFUCache: eviction callback" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const EvictTracker = struct {
        var evicted_count: usize = 0;
        var last_evicted_key: i32 = 0;

        fn callback(key: i32, _: i32) void {
            evicted_count += 1;
            last_evicted_key = key;
        }

        fn reset() void {
            evicted_count = 0;
            last_evicted_key = 0;
        }
    };

    const Cache = LFUCache(i32, i32, std.hash_map.AutoContext(i32), EvictTracker.callback);
    var cache = Cache.init(allocator, 2);
    defer cache.deinit();

    EvictTracker.reset();
    _ = try cache.put(1, 100);
    _ = try cache.put(2, 200);
    _ = try cache.put(3, 300); // Should evict key 1 or 2 (LRU tiebreaker)

    try testing.expectEqual(@as(usize, 1), EvictTracker.evicted_count);
    try testing.expectEqual(@as(i32, 1), EvictTracker.last_evicted_key); // LRU
}

test "LFUCache: memory leak check" {
    const testing = std.testing;

    const Cache = LFUCache(i32, i32, std.hash_map.AutoContext(i32), null);
    var cache = Cache.init(testing.allocator, 10);
    defer cache.deinit();

    // Fill cache with 100 items (capacity=10, so many will be evicted)
    for (0..100) |i| {
        _ = try cache.put(@intCast(i), @intCast(i * 2));
    }

    // Verify final state: cache should contain exactly 10 items
    try testing.expectEqual(@as(usize, 10), cache.count());

    // Verify items are still retrievable
    if (cache.get(99)) |val| {
        try testing.expectEqual(@as(i32, 99 * 2), val);
    }

    try cache.validate();
}
