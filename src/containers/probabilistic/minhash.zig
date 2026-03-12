//! MinHash — Locality-Sensitive Hashing for Jaccard similarity estimation.
//!
//! MinHash is a probabilistic technique for estimating the Jaccard similarity
//! between sets. It uses k independent hash functions to produce k minimum hash
//! values (signatures) for each set. The fraction of matching signatures approximates
//! the Jaccard similarity.
//!
//! Key properties:
//!   - E[|MinHash(A) ∩ MinHash(B)| / k] = |A ∩ B| / |A ∪ B| (Jaccard similarity)
//!   - Standard error: sqrt((J(1-J))/k) where J is Jaccard similarity
//!   - k=128: ~8.8% error, k=256: ~6.2% error, k=512: ~4.4% error
//!
//! Applications:
//!   - Document similarity and near-duplicate detection
//!   - Clustering large sets (LSH bucketing)
//!   - Web page deduplication, plagiarism detection
//!   - Recommendation systems (user/item similarity)
//!
//! Consumer use cases:
//!   - zoltraak: Similar key detection for cache optimization
//!   - General: Text similarity, data deduplication, clustering

const std = @import("std");

/// MinHash signature for efficient Jaccard similarity estimation.
///
/// Parameters:
///   - T: Element type (must be hashable)
///   - Context: Hash/equality context for T
///   - num_hashes: Number of hash functions (more = higher accuracy, more memory)
pub fn MinHash(
    comptime T: type,
    comptime Context: type,
    comptime num_hashes: usize,
) type {
    return struct {
        const Self = @This();

        /// Hash function parameters (a, b for universal hashing: h(x) = (ax + b) mod p)
        const HashParams = struct {
            a: u64,
            b: u64,
        };

        /// Signature: k minimum hash values.
        signature: [num_hashes]u64,
        params: [num_hashes]HashParams,
        ctx: Context,

        /// Initialize a MinHash with random hash function parameters.
        /// Time: O(k) | Space: O(k)
        pub fn init(seed: u64) Self {
            var rng = std.Random.DefaultPrng.init(seed);
            const random = rng.random();

            var self = Self{
                .signature = [_]u64{std.math.maxInt(u64)} ** num_hashes,
                .params = undefined,
                .ctx = if (@hasDecl(Context, "init")) Context.init() else .{},
            };

            // Generate random parameters for k hash functions
            for (&self.params) |*p| {
                p.a = random.int(u64) | 1; // Ensure odd (coprime with 2^64)
                p.b = random.int(u64);
            }

            return self;
        }

        /// Add an element to the MinHash signature.
        /// Updates signature[i] = min(signature[i], hash_i(element)).
        /// Time: O(k) | Space: O(1)
        pub fn add(self: *Self, element: T) void {
            const base_hash = if (@hasDecl(Context, "hash"))
                Context.hash(self.ctx, element)
            else
                std.hash_map.getAutoHashFn(T, Context)(self.ctx, element);

            for (self.params, 0..) |params, i| {
                const h = self.hashWithParams(base_hash, params);
                self.signature[i] = @min(self.signature[i], h);
            }
        }

        /// Estimate Jaccard similarity with another MinHash.
        /// Returns value in [0.0, 1.0] where 1.0 = identical sets, 0.0 = disjoint sets.
        /// Time: O(k) | Space: O(1)
        pub fn similarity(self: *const Self, other: *const Self) f64 {
            var matches: usize = 0;
            for (self.signature, other.signature) |s1, s2| {
                if (s1 == s2) matches += 1;
            }
            return @as(f64, @floatFromInt(matches)) / @as(f64, @floatFromInt(num_hashes));
        }

        /// Estimate Jaccard similarity from raw signatures.
        /// Time: O(k) | Space: O(1)
        pub fn similarityFromSignatures(sig1: [num_hashes]u64, sig2: [num_hashes]u64) f64 {
            var matches: usize = 0;
            for (sig1, sig2) |s1, s2| {
                if (s1 == s2) matches += 1;
            }
            return @as(f64, @floatFromInt(matches)) / @as(f64, @floatFromInt(num_hashes));
        }

        /// Merge another MinHash into this one (union of sets).
        /// signature[i] = min(signature[i], other.signature[i])
        /// Time: O(k) | Space: O(1)
        pub fn merge(self: *Self, other: *const Self) void {
            for (&self.signature, other.signature) |*s1, s2| {
                s1.* = @min(s1.*, s2);
            }
        }

        /// Clear the signature (reset to empty set).
        /// Time: O(k) | Space: O(1)
        pub fn clear(self: *Self) void {
            @memset(&self.signature, std.math.maxInt(u64));
        }

        /// Check if the signature is empty (no elements added).
        /// Time: O(k) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            for (self.signature) |s| {
                if (s != std.math.maxInt(u64)) return false;
            }
            return true;
        }

        /// Get a copy of the signature array.
        /// Time: O(k) | Space: O(k)
        pub fn getSignature(self: *const Self) [num_hashes]u64 {
            return self.signature;
        }

        /// Create a MinHash from multiple elements.
        /// Time: O(n * k) | Space: O(k)
        pub fn fromSlice(seed: u64, elements: []const T) Self {
            var mh = Self.init(seed);
            for (elements) |elem| {
                mh.add(elem);
            }
            return mh;
        }

        // ── Private helpers ──────────────────────────────────────────────

        /// Universal hash function: h(x) = (ax + b) mod 2^64
        fn hashWithParams(self: *const Self, base: u64, params: HashParams) u64 {
            _ = self;
            // Use wrapping arithmetic (natural mod 2^64)
            return base *% params.a +% params.b;
        }
    };
}

// ── Tests ────────────────────────────────────────────────────────────

test "MinHash: basic operations" {
    const MH = MinHash(u32, std.hash_map.AutoContext(u32), 128);
    var mh1 = MH.init(42);
    var mh2 = MH.init(42); // Same seed = same hash functions

    // Empty signatures
    try std.testing.expect(mh1.isEmpty());
    try std.testing.expect(mh2.isEmpty());

    // Add same elements
    mh1.add(1);
    mh1.add(2);
    mh1.add(3);

    mh2.add(1);
    mh2.add(2);
    mh2.add(3);

    // Identical sets → similarity = 1.0
    const sim = mh1.similarity(&mh2);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sim, 0.001);

    try std.testing.expect(!mh1.isEmpty());
}

test "MinHash: Jaccard similarity estimation" {
    const MH = MinHash(u32, std.hash_map.AutoContext(u32), 256);

    // Set A = {1, 2, 3, 4, 5}
    var mh_a = MH.init(42);
    for (1..6) |i| mh_a.add(@intCast(i));

    // Set B = {4, 5, 6, 7, 8}
    var mh_b = MH.init(42);
    for (4..9) |i| mh_b.add(@intCast(i));

    // Jaccard(A, B) = |{4, 5}| / |{1,2,3,4,5,6,7,8}| = 2/8 = 0.25
    const sim = mh_a.similarity(&mh_b);
    // Allow ±10% error (256 hashes should give ~6% standard error)
    try std.testing.expect(sim >= 0.15 and sim <= 0.35);
}

test "MinHash: disjoint sets" {
    const MH = MinHash(u32, std.hash_map.AutoContext(u32), 128);

    var mh_a = MH.init(42);
    for (1..11) |i| mh_a.add(@intCast(i)); // {1..10}

    var mh_b = MH.init(42);
    for (11..21) |i| mh_b.add(@intCast(i)); // {11..20}

    // Disjoint sets → similarity ≈ 0.0
    const sim = mh_a.similarity(&mh_b);
    try std.testing.expect(sim <= 0.1); // Allow some noise
}

test "MinHash: identical sets" {
    const MH = MinHash(u32, std.hash_map.AutoContext(u32), 128);

    var mh_a = MH.init(42);
    var mh_b = MH.init(42);

    for (1..51) |i| {
        mh_a.add(@intCast(i));
        mh_b.add(@intCast(i));
    }

    // Identical sets → similarity = 1.0
    const sim = mh_a.similarity(&mh_b);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sim, 0.001);
}

test "MinHash: order independence" {
    const MH = MinHash(u32, std.hash_map.AutoContext(u32), 128);

    var mh_a = MH.init(42);
    mh_a.add(1);
    mh_a.add(2);
    mh_a.add(3);

    var mh_b = MH.init(42);
    mh_b.add(3);
    mh_b.add(1);
    mh_b.add(2);

    // Order doesn't matter
    const sim = mh_a.similarity(&mh_b);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sim, 0.001);
}

test "MinHash: duplicate elements" {
    const MH = MinHash(u32, std.hash_map.AutoContext(u32), 128);

    var mh_a = MH.init(42);
    mh_a.add(1);
    mh_a.add(1); // duplicate
    mh_a.add(2);

    var mh_b = MH.init(42);
    mh_b.add(1);
    mh_b.add(2);

    // Duplicates don't affect signature (set semantics)
    const sim = mh_a.similarity(&mh_b);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sim, 0.001);
}

test "MinHash: merge (union)" {
    const MH = MinHash(u32, std.hash_map.AutoContext(u32), 128);

    var mh_a = MH.init(42);
    for (1..6) |i| mh_a.add(@intCast(i)); // {1..5}

    var mh_b = MH.init(42);
    for (4..9) |i| mh_b.add(@intCast(i)); // {4..8}

    var mh_union = MH.init(42);
    for (1..9) |i| mh_union.add(@intCast(i)); // {1..8}

    // Merge A into a copy
    var mh_merged = MH.init(42);
    for (1..6) |i| mh_merged.add(@intCast(i));
    mh_merged.merge(&mh_b);

    // Merged should be similar to union
    const sim = mh_merged.similarity(&mh_union);
    try std.testing.expect(sim >= 0.9); // High similarity expected
}

test "MinHash: clear" {
    const MH = MinHash(u32, std.hash_map.AutoContext(u32), 128);

    var mh = MH.init(42);
    mh.add(1);
    mh.add(2);
    try std.testing.expect(!mh.isEmpty());

    mh.clear();
    try std.testing.expect(mh.isEmpty());

    // Can reuse after clear
    mh.add(3);
    try std.testing.expect(!mh.isEmpty());
}

test "MinHash: fromSlice" {
    const MH = MinHash(u32, std.hash_map.AutoContext(u32), 128);

    const elements = [_]u32{ 1, 2, 3, 4, 5 };
    const mh = MH.fromSlice(42, &elements);

    var mh_manual = MH.init(42);
    for (elements) |e| mh_manual.add(e);

    const sim = mh.similarity(&mh_manual);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sim, 0.001);
}

test "MinHash: string sets" {
    const MH = MinHash([]const u8, std.hash_map.StringContext, 256);

    var mh_a = MH.init(42);
    mh_a.add("hello");
    mh_a.add("world");
    mh_a.add("foo");

    var mh_b = MH.init(42);
    mh_b.add("hello");
    mh_b.add("world");
    mh_b.add("bar");

    // Jaccard = |{hello, world}| / |{hello, world, foo, bar}| = 2/4 = 0.5
    const sim = mh_a.similarity(&mh_b);
    try std.testing.expect(sim >= 0.35 and sim <= 0.65); // ±15% error
}

test "MinHash: different seeds = different hash functions" {
    const MH = MinHash(u32, std.hash_map.AutoContext(u32), 128);

    var mh_a = MH.init(42);
    var mh_b = MH.init(99); // Different seed

    for (1..11) |i| {
        mh_a.add(@intCast(i));
        mh_b.add(@intCast(i));
    }

    // Same set, different hash functions → similarity likely < 1.0
    const sim = mh_a.similarity(&mh_b);
    // Signatures will differ due to different hash functions
    // (This is expected behavior — for comparison, use same seed)
    _ = sim; // Just verify it doesn't crash
}

test "MinHash: accuracy with varying k" {
    // k=64 → higher error
    const MH64 = MinHash(u32, std.hash_map.AutoContext(u32), 64);
    var mh64_a = MH64.init(42);
    var mh64_b = MH64.init(42);
    for (1..11) |i| mh64_a.add(@intCast(i)); // {1..10}
    for (6..16) |i| mh64_b.add(@intCast(i)); // {6..15}
    // Jaccard = 5/15 = 0.333
    const sim64 = mh64_a.similarity(&mh64_b);

    // k=512 → lower error
    const MH512 = MinHash(u32, std.hash_map.AutoContext(u32), 512);
    var mh512_a = MH512.init(42);
    var mh512_b = MH512.init(42);
    for (1..11) |i| mh512_a.add(@intCast(i));
    for (6..16) |i| mh512_b.add(@intCast(i));
    const sim512 = mh512_a.similarity(&mh512_b);

    // Both should be close to 0.333, but allow wider range for k=64
    try std.testing.expect(sim64 >= 0.2 and sim64 <= 0.5);
    try std.testing.expect(sim512 >= 0.25 and sim512 <= 0.42);
    // k=512 should generally be more accurate (closer to 0.333)
}

test "MinHash: stress test" {
    const MH = MinHash(u32, std.hash_map.AutoContext(u32), 128);

    var mh_a = MH.init(42);
    var mh_b = MH.init(42);

    // Large sets with 50% overlap
    for (0..1000) |i| mh_a.add(@intCast(i)); // {0..999}
    for (500..1500) |i| mh_b.add(@intCast(i)); // {500..1499}

    // Jaccard = 500 / 1500 = 0.333
    const sim = mh_a.similarity(&mh_b);
    try std.testing.expect(sim >= 0.25 and sim <= 0.42); // ±25% error allowance
}

test "MinHash: getSignature" {
    const MH = MinHash(u32, std.hash_map.AutoContext(u32), 128);

    var mh = MH.init(42);
    mh.add(1);
    mh.add(2);
    mh.add(3);

    const sig = mh.getSignature();
    try std.testing.expectEqual(@as(usize, 128), sig.len);

    // Signature should not be all maxInt (some elements added)
    var all_max = true;
    for (sig) |s| {
        if (s != std.math.maxInt(u64)) {
            all_max = false;
            break;
        }
    }
    try std.testing.expect(!all_max);
}

test "MinHash: similarityFromSignatures" {
    const MH = MinHash(u32, std.hash_map.AutoContext(u32), 128);

    var mh1 = MH.init(42);
    var mh2 = MH.init(42);

    for (1..11) |i| mh1.add(@intCast(i));
    for (1..11) |i| mh2.add(@intCast(i));

    const sig1 = mh1.getSignature();
    const sig2 = mh2.getSignature();

    const sim1 = mh1.similarity(&mh2);
    const sim2 = MH.similarityFromSignatures(sig1, sig2);

    try std.testing.expectApproxEqAbs(sim1, sim2, 0.001);
}
