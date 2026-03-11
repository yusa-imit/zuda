const std = @import("std");
const testing = std.testing;

/// Cuckoo Filter - space-efficient set membership test with deletion support
///
/// Improves on Bloom filters by supporting deletion while maintaining similar
/// space efficiency and false positive rates. Uses cuckoo hashing with fingerprints.
///
/// Structure:
/// - m buckets, each holding b entries (typically b=4)
/// - Each entry stores an f-bit fingerprint (typically f=8 or f=16)
/// - Item can be in one of two buckets: i1 = hash(x) or i2 = i1 ⊕ hash(fingerprint(x))
///
/// False positive rate: ≈ 2b/2^f for load factor α ≈ 0.95
/// - f=8, b=4: ~3.1% false positive rate
/// - f=16, b=4: ~0.012% false positive rate
///
/// Operations:
/// - add(item): insert fingerprint, O(1) expected
/// - contains(item): check if item might be in set, O(1)
/// - remove(item): delete fingerprint, O(1) expected
///
/// Consumer: caching systems, network deduplication, approximate sets with deletion
pub fn CuckooFilter(
    comptime T: type,
    comptime Context: type,
    comptime hashFn: fn (ctx: Context, key: T) u64,
    comptime fingerprintFn: fn (ctx: Context, key: T) u32,
) type {
    return struct {
        const Self = @This();
        const BUCKET_SIZE = 4; // Standard bucket size
        const MAX_KICKS = 500; // Maximum relocations before resize

        const Bucket = struct {
            fingerprints: [BUCKET_SIZE]u32,
            count: u8,

            fn init() Bucket {
                return .{ .fingerprints = [_]u32{0} ** BUCKET_SIZE, .count = 0 };
            }

            fn insert(self: *Bucket, fp: u32) bool {
                if (self.count >= BUCKET_SIZE) return false;
                self.fingerprints[self.count] = fp;
                self.count += 1;
                return true;
            }

            fn remove(self: *Bucket, fp: u32) bool {
                for (self.fingerprints[0..self.count], 0..) |stored_fp, i| {
                    if (stored_fp == fp) {
                        // Shift remaining entries
                        if (i + 1 < self.count) {
                            std.mem.copyForwards(
                                u32,
                                self.fingerprints[i .. self.count - 1],
                                self.fingerprints[i + 1 .. self.count],
                            );
                        }
                        self.count -= 1;
                        return true;
                    }
                }
                return false;
            }

            fn contains(self: *const Bucket, fp: u32) bool {
                for (self.fingerprints[0..self.count]) |stored_fp| {
                    if (stored_fp == fp) return true;
                }
                return false;
            }

            fn isFull(self: *const Bucket) bool {
                return self.count >= BUCKET_SIZE;
            }
        };

        allocator: std.mem.Allocator,
        buckets: []Bucket,
        num_items: usize,
        ctx: Context,

        /// Time: O(n) | Space: O(n*b*f) where n=buckets, b=bucket_size, f=fingerprint_bits
        pub fn init(allocator: std.mem.Allocator, num_buckets: usize, ctx: Context) !Self {
            if (num_buckets == 0) return error.InvalidBucketCount;

            const buckets = try allocator.alloc(Bucket, num_buckets);
            for (buckets) |*bucket| {
                bucket.* = Bucket.init();
            }

            return Self{
                .allocator = allocator,
                .buckets = buckets,
                .num_items = 0,
                .ctx = ctx,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.buckets);
            self.* = undefined;
        }

        fn indexHash(self: *const Self, item: T) usize {
            const hash = hashFn(self.ctx, item);
            return hash % self.buckets.len;
        }

        fn altIndex(self: *const Self, index: usize, fp: u32) usize {
            // Compute alternative index: i2 = i1 ⊕ hash(fingerprint)
            var hasher = std.hash.Wyhash.init(0);
            const fp_bytes = std.mem.asBytes(&fp);
            hasher.update(fp_bytes);
            const hash_fp = hasher.final();
            return (index ^ hash_fp) % self.buckets.len;
        }

        /// Add an item to the filter
        /// Time: O(1) expected | Space: O(1)
        /// May fail if filter is too full (returns error.FilterFull)
        pub fn add(self: *Self, item: T) !void {
            const fp = fingerprintFn(self.ctx, item);
            if (fp == 0) return error.InvalidFingerprint; // 0 fingerprint reserved for empty

            const i = self.indexHash(item);
            const alt_i = self.altIndex(i, fp);

            // Try to insert in first bucket
            if (self.buckets[i].insert(fp)) {
                self.num_items += 1;
                return;
            }

            // Try to insert in alternate bucket
            if (self.buckets[alt_i].insert(fp)) {
                self.num_items += 1;
                return;
            }

            // Both buckets full, relocate existing entries (cuckoo hashing)
            var current_fp = fp;
            var current_index = i;

            for (0..MAX_KICKS) |_| {
                // Pick random entry from bucket to evict
                const victim_idx = @as(usize, @intCast(std.crypto.random.intRangeAtMost(u8, 0, BUCKET_SIZE - 1)));
                const victim_fp = self.buckets[current_index].fingerprints[victim_idx];

                // Replace victim with current fingerprint
                self.buckets[current_index].fingerprints[victim_idx] = current_fp;

                // Try to insert victim into its alternate location
                current_fp = victim_fp;
                current_index = self.altIndex(current_index, victim_fp);

                if (!self.buckets[current_index].isFull()) {
                    _ = self.buckets[current_index].insert(current_fp);
                    self.num_items += 1;
                    return;
                }
            }

            return error.FilterFull; // Too many relocations
        }

        /// Check if item might be in the filter
        /// Time: O(1) | Space: O(1)
        /// Returns true if item MAY be present (possible false positive)
        /// Returns false if item is DEFINITELY NOT present (no false negatives)
        pub fn contains(self: *const Self, item: T) bool {
            const fp = fingerprintFn(self.ctx, item);
            if (fp == 0) return false;

            const idx1 = self.indexHash(item);
            if (self.buckets[idx1].contains(fp)) return true;

            const idx2 = self.altIndex(idx1, fp);
            return self.buckets[idx2].contains(fp);
        }

        /// Remove an item from the filter
        /// Time: O(1) | Space: O(1)
        /// Returns true if item was found and removed, false otherwise
        pub fn remove(self: *Self, item: T) bool {
            const fp = fingerprintFn(self.ctx, item);
            if (fp == 0) return false;

            const idx1 = self.indexHash(item);
            if (self.buckets[idx1].remove(fp)) {
                self.num_items -= 1;
                return true;
            }

            const idx2 = self.altIndex(idx1, fp);
            if (self.buckets[idx2].remove(fp)) {
                self.num_items -= 1;
                return true;
            }

            return false;
        }

        /// Clear all entries
        /// Time: O(n) | Space: O(1)
        pub fn clear(self: *Self) void {
            for (self.buckets) |*bucket| {
                bucket.* = Bucket.init();
            }
            self.num_items = 0;
        }

        /// Number of items in the filter
        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.num_items;
        }

        /// Load factor (fraction of occupied slots)
        /// Time: O(1) | Space: O(1)
        pub fn loadFactor(self: *const Self) f64 {
            const capacity = self.buckets.len * BUCKET_SIZE;
            return @as(f64, @floatFromInt(self.num_items)) / @as(f64, @floatFromInt(capacity));
        }
    };
}

// Default hash function for integers
pub fn defaultHashInt(comptime T: type) fn (void, T) u64 {
    return struct {
        fn hash(_: void, key: T) u64 {
            var hasher = std.hash.Wyhash.init(0);
            const bytes = std.mem.asBytes(&key);
            hasher.update(bytes);
            return hasher.final();
        }
    }.hash;
}

// Default fingerprint function (8-bit)
pub fn defaultFingerprintInt(comptime T: type) fn (void, T) u32 {
    return struct {
        fn fingerprint(_: void, key: T) u32 {
            var hasher = std.hash.Wyhash.init(1); // Different seed from hash
            const bytes = std.mem.asBytes(&key);
            hasher.update(bytes);
            const hash = hasher.final();
            // Use lower 8 bits, avoid 0
            const fp = @as(u32, @truncate(hash & 0xFF));
            return if (fp == 0) 1 else fp;
        }
    }.fingerprint;
}

// Default hash function for slices
pub fn defaultHashSlice(comptime T: type) fn (void, []const T) u64 {
    return struct {
        fn hash(_: void, key: []const T) u64 {
            var hasher = std.hash.Wyhash.init(0);
            for (key) |item| {
                const bytes = std.mem.asBytes(&item);
                hasher.update(bytes);
            }
            return hasher.final();
        }
    }.hash;
}

// Default fingerprint for slices
pub fn defaultFingerprintSlice(comptime T: type) fn (void, []const T) u32 {
    return struct {
        fn fingerprint(_: void, key: []const T) u32 {
            var hasher = std.hash.Wyhash.init(1);
            for (key) |item| {
                const bytes = std.mem.asBytes(&item);
                hasher.update(bytes);
            }
            const hash = hasher.final();
            const fp = @as(u32, @truncate(hash & 0xFF));
            return if (fp == 0) 1 else fp;
        }
    }.fingerprint;
}

// Tests
test "CuckooFilter - basic operations" {
    const Filter = CuckooFilter(u32, void, defaultHashInt(u32), defaultFingerprintInt(u32));
    var filter = try Filter.init(testing.allocator, 100, {});
    defer filter.deinit();

    // Empty filter
    try testing.expect(!filter.contains(42));
    try testing.expectEqual(@as(usize, 0), filter.count());

    // Add items
    try filter.add(42);
    try filter.add(100);
    try filter.add(200);

    // Check added items
    try testing.expect(filter.contains(42));
    try testing.expect(filter.contains(100));
    try testing.expect(filter.contains(200));
    try testing.expectEqual(@as(usize, 3), filter.count());

    // Remove item
    try testing.expect(filter.remove(100));
    try testing.expect(!filter.contains(100));
    try testing.expectEqual(@as(usize, 2), filter.count());
}

test "CuckooFilter - deletion" {
    const Filter = CuckooFilter(u32, void, defaultHashInt(u32), defaultFingerprintInt(u32));
    var filter = try Filter.init(testing.allocator, 100, {});
    defer filter.deinit();

    // Add and remove
    try filter.add(1);
    try filter.add(2);
    try filter.add(3);

    try testing.expect(filter.remove(2));
    try testing.expect(!filter.contains(2));
    try testing.expect(filter.contains(1));
    try testing.expect(filter.contains(3));

    // Remove non-existent
    try testing.expect(!filter.remove(999));
}

test "CuckooFilter - duplicates" {
    const Filter = CuckooFilter(u32, void, defaultHashInt(u32), defaultFingerprintInt(u32));
    var filter = try Filter.init(testing.allocator, 100, {});
    defer filter.deinit();

    // Add same item multiple times
    try filter.add(42);
    try filter.add(42);
    try filter.add(42);

    try testing.expectEqual(@as(usize, 3), filter.count());
    try testing.expect(filter.contains(42));

    // Remove once
    try testing.expect(filter.remove(42));
    try testing.expect(filter.contains(42)); // Still 2 copies
    try testing.expectEqual(@as(usize, 2), filter.count());

    // Remove remaining
    try testing.expect(filter.remove(42));
    try testing.expect(filter.remove(42));
    try testing.expect(!filter.contains(42));
    try testing.expectEqual(@as(usize, 0), filter.count());
}

test "CuckooFilter - clear" {
    const Filter = CuckooFilter(u32, void, defaultHashInt(u32), defaultFingerprintInt(u32));
    var filter = try Filter.init(testing.allocator, 100, {});
    defer filter.deinit();

    try filter.add(1);
    try filter.add(2);
    try filter.add(3);

    filter.clear();
    try testing.expect(!filter.contains(1));
    try testing.expect(!filter.contains(2));
    try testing.expect(!filter.contains(3));
    try testing.expectEqual(@as(usize, 0), filter.count());
}

test "CuckooFilter - string keys" {
    const Filter = CuckooFilter([]const u8, void, defaultHashSlice(u8), defaultFingerprintSlice(u8));
    var filter = try Filter.init(testing.allocator, 1000, {});
    defer filter.deinit();

    try filter.add("hello");
    try filter.add("world");
    try filter.add("cuckoo");

    try testing.expect(filter.contains("hello"));
    try testing.expect(filter.contains("world"));
    try testing.expect(filter.contains("cuckoo"));
    try testing.expect(!filter.contains("notfound"));

    try testing.expect(filter.remove("world"));
    try testing.expect(!filter.contains("world"));
}

test "CuckooFilter - load factor" {
    const Filter = CuckooFilter(u32, void, defaultHashInt(u32), defaultFingerprintInt(u32));
    var filter = try Filter.init(testing.allocator, 100, {}); // 100 buckets * 4 entries = 400 capacity
    defer filter.deinit();

    // Add 200 items (50% load)
    for (0..200) |i| {
        try filter.add(@intCast(i));
    }

    const lf = filter.loadFactor();
    try testing.expect(lf >= 0.49 and lf <= 0.51); // ~50%
}

test "CuckooFilter - high load stress" {
    const Filter = CuckooFilter(u32, void, defaultHashInt(u32), defaultFingerprintInt(u32));
    var filter = try Filter.init(testing.allocator, 100, {}); // 400 capacity
    defer filter.deinit();

    // Fill to ~90% capacity (360 items)
    for (0..360) |i| {
        filter.add(@intCast(i)) catch break; // May fail near capacity
    }

    // Verify most items are present
    var found: usize = 0;
    for (0..360) |i| {
        if (filter.contains(@intCast(i))) {
            found += 1;
        }
    }
    // Should have at least 95% of items
    try testing.expect(found >= 342); // 360 * 0.95
}

test "CuckooFilter - false positive rate" {
    const Filter = CuckooFilter(u32, void, defaultHashInt(u32), defaultFingerprintInt(u32));
    var filter = try Filter.init(testing.allocator, 1000, {});
    defer filter.deinit();

    // Add 1000 items
    for (0..1000) |i| {
        try filter.add(@intCast(i));
    }

    // Count false positives for items not in set
    var false_positives: usize = 0;
    for (1000..2000) |i| {
        if (filter.contains(@intCast(i))) {
            false_positives += 1;
        }
    }

    const fp_rate = @as(f64, @floatFromInt(false_positives)) / 1000.0;
    // With 8-bit fingerprints and bucket size 4, FP rate ≈ 2b/2^f ≈ 8/256 ≈ 3.1%
    // Allow up to 10% for test stability
    try testing.expect(fp_rate < 0.10);
}

test "CuckooFilter - edge cases" {
    const Filter = CuckooFilter(u32, void, defaultHashInt(u32), defaultFingerprintInt(u32));

    // Invalid bucket count
    try testing.expectError(error.InvalidBucketCount, Filter.init(testing.allocator, 0, {}));

    // Minimal filter
    var filter = try Filter.init(testing.allocator, 1, {});
    defer filter.deinit();

    try filter.add(1);
    try testing.expect(filter.contains(1));
    try testing.expect(filter.remove(1));
    try testing.expect(!filter.contains(1));
}
