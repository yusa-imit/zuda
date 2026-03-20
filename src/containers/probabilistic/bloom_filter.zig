const std = @import("std");
const testing = std.testing;

/// Bloom filter - space-efficient probabilistic set membership test
/// No false negatives, configurable false positive rate
///
/// A Bloom filter is a bit array of m bits with k independent hash functions.
/// - add(item): set k bits to 1
/// - contains(item): check if all k bits are 1
/// - False positives possible (may return true for non-members)
/// - False negatives impossible (never returns false for members)
///
/// False positive rate: p ≈ (1 - e^(-kn/m))^k
/// Optimal k: k = (m/n) * ln(2)
/// where m = bit array size, n = expected number of elements, k = hash count
///
/// Consumer: zoltraak (dedupe), web crawlers (URL visited check)
pub fn BloomFilter(
    comptime T: type,
    comptime Context: type,
    comptime hashFn: fn (ctx: Context, key: T, seed: u64) u64,
) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        bits: []u64, // Bit array stored as u64 words
        m: usize, // Total bits
        k: usize, // Number of hash functions
        count: usize, // Approximate element count
        ctx: Context,

        /// Time: O(1) | Space: O(m)
        /// m: bit array size, k: number of hash functions
        pub fn init(allocator: std.mem.Allocator, m: usize, k: usize, ctx: Context) !Self {
            if (m == 0) return error.InvalidSize;
            if (k == 0) return error.InvalidHashCount;

            const words = (m + 63) / 64; // Round up to nearest u64
            const bits = try allocator.alloc(u64, words);
            @memset(bits, 0);

            return Self{
                .allocator = allocator,
                .bits = bits,
                .m = m,
                .k = k,
                .count = 0,
                .ctx = ctx,
            };
        }

        /// Initialize with target false positive rate
        /// Time: O(1) | Space: O(m)
        /// n: expected number of elements, p: target false positive rate (0 < p < 1)
        /// Formula: m = -n*ln(p) / (ln(2))^2, k = (m/n)*ln(2)
        pub fn initWithFalsePositiveRate(
            allocator: std.mem.Allocator,
            n: usize,
            p: f64,
            ctx: Context,
        ) !Self {
            if (n == 0) return error.InvalidElementCount;
            if (p <= 0.0 or p >= 1.0) return error.InvalidFalsePositiveRate;

            const ln2 = @log(2.0);
            const m_float = -@as(f64, @floatFromInt(n)) * @log(p) / (ln2 * ln2);
            const m = @max(1, @as(usize, @intFromFloat(@ceil(m_float))));

            const k_float = (@as(f64, @floatFromInt(m)) / @as(f64, @floatFromInt(n))) * ln2;
            const k = @max(1, @as(usize, @intFromFloat(@round(k_float))));

            return Self.init(allocator, m, k, ctx);
        }

        /// Frees all allocated memory. Invalidates all iterators.
        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.bits);
            self.* = undefined;
        }

        /// Add an item to the filter
        /// Time: O(k) | Space: O(1)
        pub fn add(self: *Self, item: T) void {
            for (0..self.k) |i| {
                const hash = hashFn(self.ctx, item, i);
                const bit_index = hash % self.m;
                const word_index: usize = @intCast(bit_index / 64);
                const bit_offset = @as(u6, @intCast(bit_index % 64));
                self.bits[word_index] |= @as(u64, 1) << bit_offset;
            }
            self.count += 1;
        }

        /// Check if an item might be in the filter
        /// Time: O(k) | Space: O(1)
        /// Returns true if item MAY be in set (possible false positive)
        /// Returns false if item is DEFINITELY NOT in set (no false negatives)
        pub fn contains(self: *const Self, item: T) bool {
            for (0..self.k) |i| {
                const hash = hashFn(self.ctx, item, i);
                const bit_index = hash % self.m;
                const word_index: usize = @intCast(bit_index / 64);
                const bit_offset = @as(u6, @intCast(bit_index % 64));
                const bit_set = (self.bits[word_index] & (@as(u64, 1) << bit_offset)) != 0;
                if (!bit_set) return false;
            }
            return true;
        }

        /// Clear all bits (reset filter)
        /// Time: O(m/64) | Space: O(1)
        pub fn clear(self: *Self) void {
            @memset(self.bits, 0);
            self.count = 0;
        }

        /// Estimate current false positive rate
        /// Time: O(m/64) | Space: O(1)
        pub fn estimatedFalsePositiveRate(self: *const Self) f64 {
            if (self.count == 0) return 0.0;

            // Count set bits
            var set_bits: usize = 0;
            for (self.bits) |word| {
                set_bits += @popCount(word);
            }

            const fill_ratio = @as(f64, @floatFromInt(set_bits)) / @as(f64, @floatFromInt(self.m));
            const k_float = @as(f64, @floatFromInt(self.k));
            return std.math.pow(f64, fill_ratio, k_float);
        }

        /// Approximate number of elements added
        /// Time: O(1) | Space: O(1)
        pub fn approximateCount(self: *const Self) usize {
            return self.count;
        }

        /// Union (merge) two Bloom filters with same parameters
        /// Time: O(m/64) | Space: O(1)
        /// Both filters must have same m, k, and hash functions
        pub fn unionWith(self: *Self, other: *const Self) error{IncompatibleFilters}!void {
            if (self.m != other.m or self.k != other.k) {
                return error.IncompatibleFilters;
            }

            for (self.bits, other.bits) |*self_word, other_word| {
                self_word.* |= other_word;
            }
            // Approximate count (may overestimate due to overlap)
            self.count = self.count + other.count;
        }

        /// Intersection of two Bloom filters with same parameters
        /// Time: O(m/64) | Space: O(1)
        /// Result may have higher false positive rate than either input
        pub fn intersectionWith(self: *Self, other: *const Self) error{IncompatibleFilters}!void {
            if (self.m != other.m or self.k != other.k) {
                return error.IncompatibleFilters;
            }

            for (self.bits, other.bits) |*self_word, other_word| {
                self_word.* &= other_word;
            }
            // Approximate count (may underestimate)
            self.count = @min(self.count, other.count);
        }

        /// Validate internal invariants
        /// Time: O(1) | Space: O(1)
        pub fn validate(self: *const Self) void {
            const expected_words = (self.m + 63) / 64;
            std.debug.assert(self.bits.len == expected_words);
            std.debug.assert(self.m > 0);
            std.debug.assert(self.k > 0);
        }
    };
}

// Default hash function for integers
/// Default hash function for integers.
/// Time: O(1) | Space: O(1)
pub fn defaultHashInt(comptime T: type) fn (void, T, u64) u64 {
    return struct {
        fn hash(_: void, key: T, seed: u64) u64 {
            var hasher = std.hash.Wyhash.init(seed);
            const bytes = std.mem.asBytes(&key);
            hasher.update(bytes);
            return hasher.final();
        }
    }.hash;
}

// Default hash function for slices
/// Default hash function for slices.
/// Time: O(n) | Space: O(1)
pub fn defaultHashSlice(comptime T: type) fn (void, []const T, u64) u64 {
    return struct {
        fn hash(_: void, key: []const T, seed: u64) u64 {
            var hasher = std.hash.Wyhash.init(seed);
            for (key) |item| {
                const bytes = std.mem.asBytes(&item);
                hasher.update(bytes);
            }
            return hasher.final();
        }
    }.hash;
}

// Tests
test "BloomFilter - basic operations" {
    const Filter = BloomFilter(u32, void, defaultHashInt(u32));
    var filter = try Filter.init(testing.allocator, 1000, 3, {});
    defer filter.deinit();

    // Empty filter
    try testing.expect(!filter.contains(42));
    try testing.expect(!filter.contains(100));

    // Add items
    filter.add(42);
    filter.add(100);
    filter.add(200);

    // Check added items (no false negatives)
    try testing.expect(filter.contains(42));
    try testing.expect(filter.contains(100));
    try testing.expect(filter.contains(200));

    // Check non-existent items (may have false positives, but unlikely with 1000 bits)
    const non_existent = filter.contains(999);
    _ = non_existent; // May be true or false
}

test "BloomFilter - false positive rate" {
    const Filter = BloomFilter(u32, void, defaultHashInt(u32));
    var filter = try Filter.initWithFalsePositiveRate(testing.allocator, 100, 0.01, {});
    defer filter.deinit();

    // Add 100 items
    for (0..100) |i| {
        filter.add(@intCast(i));
    }

    // All added items must be found (no false negatives)
    for (0..100) |i| {
        try testing.expect(filter.contains(@intCast(i)));
    }

    // Count false positives for items not in set
    var false_positives: usize = 0;
    for (100..1000) |i| {
        if (filter.contains(@intCast(i))) {
            false_positives += 1;
        }
    }

    const fp_rate = @as(f64, @floatFromInt(false_positives)) / 900.0;
    // Should be close to target 0.01, allow some variance
    try testing.expect(fp_rate < 0.05); // Less than 5% false positive rate

    const estimated = filter.estimatedFalsePositiveRate();
    try testing.expect(estimated >= 0.0 and estimated <= 1.0);
}

test "BloomFilter - clear" {
    const Filter = BloomFilter(u32, void, defaultHashInt(u32));
    var filter = try Filter.init(testing.allocator, 1000, 3, {});
    defer filter.deinit();

    filter.add(1);
    filter.add(2);
    filter.add(3);
    try testing.expect(filter.contains(1));

    filter.clear();
    try testing.expect(!filter.contains(1));
    try testing.expect(!filter.contains(2));
    try testing.expect(!filter.contains(3));
    try testing.expectEqual(@as(usize, 0), filter.approximateCount());
}

test "BloomFilter - string keys" {
    const Filter = BloomFilter([]const u8, void, defaultHashSlice(u8));
    var filter = try Filter.init(testing.allocator, 10000, 4, {});
    defer filter.deinit();

    filter.add("hello");
    filter.add("world");
    filter.add("bloom");
    filter.add("filter");

    try testing.expect(filter.contains("hello"));
    try testing.expect(filter.contains("world"));
    try testing.expect(filter.contains("bloom"));
    try testing.expect(filter.contains("filter"));
    try testing.expect(!filter.contains("notfound"));
}

test "BloomFilter - union" {
    const Filter = BloomFilter(u32, void, defaultHashInt(u32));
    var filter1 = try Filter.init(testing.allocator, 1000, 3, {});
    defer filter1.deinit();
    var filter2 = try Filter.init(testing.allocator, 1000, 3, {});
    defer filter2.deinit();

    filter1.add(1);
    filter1.add(2);
    filter2.add(3);
    filter2.add(4);

    try filter1.unionWith(&filter2);

    // Filter1 should now contain all items
    try testing.expect(filter1.contains(1));
    try testing.expect(filter1.contains(2));
    try testing.expect(filter1.contains(3));
    try testing.expect(filter1.contains(4));
}

test "BloomFilter - intersection" {
    const Filter = BloomFilter(u32, void, defaultHashInt(u32));
    var filter1 = try Filter.init(testing.allocator, 1000, 3, {});
    defer filter1.deinit();
    var filter2 = try Filter.init(testing.allocator, 1000, 3, {});
    defer filter2.deinit();

    filter1.add(1);
    filter1.add(2);
    filter1.add(3);
    filter2.add(2);
    filter2.add(3);
    filter2.add(4);

    try filter1.intersectionWith(&filter2);

    // Filter1 should contain items present in both
    try testing.expect(filter1.contains(2));
    try testing.expect(filter1.contains(3));
    // Items unique to each filter may or may not be detected (depends on bit overlap)
}

test "BloomFilter - incompatible merge" {
    const Filter = BloomFilter(u32, void, defaultHashInt(u32));
    var filter1 = try Filter.init(testing.allocator, 1000, 3, {});
    defer filter1.deinit();
    var filter2 = try Filter.init(testing.allocator, 2000, 4, {}); // Different size and k
    defer filter2.deinit();

    try testing.expectError(error.IncompatibleFilters, filter1.unionWith(&filter2));
    try testing.expectError(error.IncompatibleFilters, filter1.intersectionWith(&filter2));
}

test "BloomFilter - stress test" {
    const Filter = BloomFilter(u32, void, defaultHashInt(u32));
    var filter = try Filter.initWithFalsePositiveRate(testing.allocator, 10000, 0.01, {});
    defer filter.deinit();

    // Add 10k items
    for (0..10000) |i| {
        filter.add(@intCast(i));
    }

    // Verify all added items are found
    for (0..10000) |i| {
        try testing.expect(filter.contains(@intCast(i)));
    }

    // Count false positives
    var false_positives: usize = 0;
    for (10000..20000) |i| {
        if (filter.contains(@intCast(i))) {
            false_positives += 1;
        }
    }

    const fp_rate = @as(f64, @floatFromInt(false_positives)) / 10000.0;
    // With 10k items at target 0.01 FP rate, should be under 5%
    try testing.expect(fp_rate < 0.05);
}

test "BloomFilter - empty and edge cases" {
    const Filter = BloomFilter(u32, void, defaultHashInt(u32));

    // Invalid parameters
    try testing.expectError(error.InvalidSize, Filter.init(testing.allocator, 0, 3, {}));
    try testing.expectError(error.InvalidHashCount, Filter.init(testing.allocator, 100, 0, {}));
    try testing.expectError(error.InvalidElementCount, Filter.initWithFalsePositiveRate(testing.allocator, 0, 0.01, {}));
    try testing.expectError(error.InvalidFalsePositiveRate, Filter.initWithFalsePositiveRate(testing.allocator, 100, 0.0, {}));
    try testing.expectError(error.InvalidFalsePositiveRate, Filter.initWithFalsePositiveRate(testing.allocator, 100, 1.0, {}));

    // Minimal filter
    var filter = try Filter.init(testing.allocator, 8, 1, {});
    defer filter.deinit();

    filter.add(1);
    try testing.expect(filter.contains(1));
}

test "BloomFilter - benchmark calculation verification" {
    // This test verifies that lookups can be timed independently
    // Simulating the benchmark fix: pre-populate filter, then time lookups only
    const Filter = BloomFilter(u64, void, defaultHashInt(u64));

    var filter = try Filter.init(testing.allocator, 100_000, 7, {});
    defer filter.deinit();

    // Pre-populate with 1000 items (setup, NOT timed)
    for (0..1000) |i| {
        filter.add(@intCast(i));
    }

    // Time 100K lookup operations
    var timer = try std.time.Timer.start();
    for (0..100_000) |i| {
        _ = filter.contains(@intCast(i % 2000)); // Mix of present and absent
    }
    const elapsed_ns = timer.read();

    // Verify we get meaningful timing
    try testing.expect(elapsed_ns > 0);

    // Verify ops/sec calculation works
    const ops_per_sec = @divFloor(100_000 * 1_000_000_000, elapsed_ns);
    try testing.expect(ops_per_sec > 0);

    // For a simple lookup, should be at least 1M ops/sec on modern hardware
    const million_ops_per_sec = @divFloor(ops_per_sec, 1_000_000);
    try testing.expect(million_ops_per_sec >= 1);
}
