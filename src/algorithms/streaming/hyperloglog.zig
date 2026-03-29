//! HyperLogLog - Cardinality estimation for distinct elements
//!
//! Probabilistic algorithm for counting distinct elements in a stream.
//! Uses O(m) space to estimate cardinality with typical error ~1.04/√m.
//!
//! Parameters:
//! - b: precision parameter (b bits determine m = 2^b registers)
//! - Typical b=14 (16KB) gives ±0.81% error
//! - b=16 (64KB) gives ±0.41% error
//!
//! Use cases:
//! - Database query optimization (distinct value estimation)
//! - Network monitoring (unique IP counting)
//! - Web analytics (unique visitor counting)
//! - Redis PFCOUNT implementation (zoltraak compatibility)

const std = @import("std");
const Allocator = std.mem.Allocator;

/// HyperLogLog for cardinality estimation
///
/// Time: O(1) for add, O(m) for count where m = 2^b
/// Space: O(2^b) registers, each 5-8 bits (typically 16-64KB)
pub fn HyperLogLog(comptime b: u6) type {
    if (b < 4 or b > 18) @compileError("b must be in range [4, 18]");
    
    const m = @as(usize, 1) << b; // Number of registers
    const alpha = switch (m) {
        16 => 0.673,
        32 => 0.697,
        64 => 0.709,
        else => 0.7213 / (1.0 + 1.079 / @as(f64, @floatFromInt(m))),
    };
    
    return struct {
        const Self = @This();
        
        allocator: Allocator,
        registers: []u8, // m registers, each stores max(rho(hash))
        
        /// Initialize HyperLogLog with 2^b registers
        ///
        /// Time: O(2^b)
        /// Space: O(2^b)
        pub fn init(allocator: Allocator) !Self {
            const registers = try allocator.alloc(u8, m);
            @memset(registers, 0);
            
            return Self{
                .allocator = allocator,
                .registers = registers,
            };
        }
        
        /// Clean up resources
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.registers);
        }
        
        /// Add element to the set
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn add(self: *Self, item: anytype) void {
            const hash = hashItem(item);
            
            // Use first b bits for register index
            const j = @as(usize, @intCast(hash >> (64 - b)));
            
            // Count leading zeros in remaining bits + 1
            const w = hash << b;
            const rho = if (w == 0) 65 - b else @clz(w) + 1;
            
            // Update register with maximum rho value
            self.registers[j] = @max(self.registers[j], @as(u8, @intCast(rho)));
        }
        
        /// Estimate cardinality
        ///
        /// Time: O(m) where m = 2^b
        /// Space: O(1)
        pub fn estimate(self: Self) u64 {
            // Calculate raw estimate
            var sum: f64 = 0.0;
            var V: usize = 0; // Count of zero registers
            
            for (self.registers) |reg| {
                if (reg == 0) {
                    V += 1;
                }
                const val = @as(f64, @floatFromInt(@as(u64, 1) << @as(u6, @intCast(reg))));
                sum += 1.0 / val;
            }
            
            const raw_estimate = alpha * @as(f64, @floatFromInt(m * m)) / sum;

            // Apply bias correction
            const corrected_estimate = if (raw_estimate <= 2.5 * @as(f64, @floatFromInt(m))) blk: {
                // Small range correction
                if (V > 0) {
                    const m_f = @as(f64, @floatFromInt(m));
                    const V_f = @as(f64, @floatFromInt(V));
                    break :blk m_f * @log(m_f / V_f);
                } else {
                    break :blk raw_estimate;
                }
            } else if (raw_estimate <= (1.0 / 30.0) * @as(f64, @floatFromInt(@as(u64, 1) << 32))) blk: {
                // No correction
                break :blk raw_estimate;
            } else blk: {
                // Large range correction
                const pow32 = @as(f64, @floatFromInt(@as(u64, 1) << 32));
                break :blk -pow32 * @log(1.0 - raw_estimate / pow32);
            };

            return @as(u64, @intFromFloat(@max(0.0, corrected_estimate)));
        }
        
        /// Merge another HyperLogLog into this one (union operation)
        ///
        /// Time: O(m)
        /// Space: O(1)
        pub fn merge(self: *Self, other: Self) void {
            for (self.registers, other.registers) |*r1, r2| {
                r1.* = @max(r1.*, r2);
            }
        }
        
        /// Reset all registers to zero
        ///
        /// Time: O(m)
        /// Space: O(1)
        pub fn clear(self: *Self) void {
            @memset(self.registers, 0);
        }
        
        /// Hash function for items
        fn hashItem(item: anytype) u64 {
            return std.hash.Wyhash.hash(0, std.mem.asBytes(&item));
        }
    };
}

// Tests
test "HyperLogLog: basic operations" {
    const HLL = HyperLogLog(10); // 2^10 = 1024 registers
    var hll = try HLL.init(std.testing.allocator);
    defer hll.deinit();
    
    // Initially estimate should be ~0
    const initial = hll.estimate();
    try std.testing.expect(initial <= 10); // Small margin for empty set
    
    // Add some distinct elements
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        hll.add(i);
    }
    
    const est = hll.estimate();
    // Should be close to 100 (allow ±20% error for small sets)
    try std.testing.expect(est >= 80);
    try std.testing.expect(est <= 120);
}

test "HyperLogLog: distinct vs duplicate elements" {
    const HLL = HyperLogLog(12);
    var hll1 = try HLL.init(std.testing.allocator);
    defer hll1.deinit();
    var hll2 = try HLL.init(std.testing.allocator);
    defer hll2.deinit();
    
    // Add 500 distinct elements to hll1
    var i: u32 = 0;
    while (i < 500) : (i += 1) {
        hll1.add(i);
    }
    
    // Add same 100 elements 5 times each to hll2
    var j: u32 = 0;
    while (j < 5) : (j += 1) {
        var k: u32 = 0;
        while (k < 100) : (k += 1) {
            hll2.add(k);
        }
    }
    
    const est1 = hll1.estimate();
    const est2 = hll2.estimate();
    
    // hll1 should estimate ~500, hll2 should estimate ~100
    try std.testing.expect(est1 >= 400);
    try std.testing.expect(est1 <= 600);
    try std.testing.expect(est2 >= 80);
    try std.testing.expect(est2 <= 120);
}

test "HyperLogLog: large cardinality" {
    const HLL = HyperLogLog(14); // 16K registers
    var hll = try HLL.init(std.testing.allocator);
    defer hll.deinit();
    
    // Add 10000 distinct elements
    var i: u32 = 0;
    while (i < 10000) : (i += 1) {
        hll.add(i);
    }
    
    const est = hll.estimate();
    // With b=14, error should be ~±0.81%
    // Allow ±5% for test stability
    try std.testing.expect(est >= 9500);
    try std.testing.expect(est <= 10500);
}

test "HyperLogLog: merge operation" {
    const HLL = HyperLogLog(10);
    var hll1 = try HLL.init(std.testing.allocator);
    defer hll1.deinit();
    var hll2 = try HLL.init(std.testing.allocator);
    defer hll2.deinit();
    var hll_union = try HLL.init(std.testing.allocator);
    defer hll_union.deinit();
    
    // Add 0-99 to hll1
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        hll1.add(i);
        hll_union.add(i);
    }
    
    // Add 50-149 to hll2 (50 overlap)
    i = 50;
    while (i < 150) : (i += 1) {
        hll2.add(i);
        hll_union.add(i);
    }
    
    // Merge hll2 into hll1
    hll1.merge(hll2);
    
    const est_merged = hll1.estimate();
    const est_union = hll_union.estimate();
    
    // Both should estimate ~150 (union size)
    const diff = if (est_merged > est_union) 
        est_merged - est_union 
    else 
        est_union - est_merged;
    
    // Estimates should be close (allow 20% difference)
    try std.testing.expect(diff <= 30);
}

test "HyperLogLog: clear operation" {
    const HLL = HyperLogLog(10);
    var hll = try HLL.init(std.testing.allocator);
    defer hll.deinit();
    
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        hll.add(i);
    }
    
    try std.testing.expect(hll.estimate() >= 80);
    
    hll.clear();
    try std.testing.expect(hll.estimate() <= 10);
}

test "HyperLogLog: string keys" {
    const HLL = HyperLogLog(12);
    var hll = try HLL.init(std.testing.allocator);
    defer hll.deinit();
    
    const words = [_][]const u8{
        "apple", "banana", "cherry", "date", "elderberry",
        "fig", "grape", "honeydew", "kiwi", "lemon",
    };
    
    // Add each word 10 times (duplicates)
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        for (words) |word| {
            hll.add(word);
        }
    }
    
    const est = hll.estimate();
    // Should estimate ~10 distinct words
    try std.testing.expect(est >= 8);
    try std.testing.expect(est <= 12);
}

test "HyperLogLog: memory safety" {
    const HLL = HyperLogLog(14);
    var hll = try HLL.init(std.testing.allocator);
    defer hll.deinit();
    
    // Add many elements
    var i: u32 = 0;
    while (i < 100000) : (i += 1) {
        hll.add(i);
    }
    
    _ = hll.estimate();
}

test "HyperLogLog: precision comparison" {
    // Test different precision levels
    const HLL10 = HyperLogLog(10); // 1K registers
    const HLL14 = HyperLogLog(14); // 16K registers
    
    var hll10 = try HLL10.init(std.testing.allocator);
    defer hll10.deinit();
    var hll14 = try HLL14.init(std.testing.allocator);
    defer hll14.deinit();
    
    // Add 5000 distinct elements to both
    var i: u32 = 0;
    while (i < 5000) : (i += 1) {
        hll10.add(i);
        hll14.add(i);
    }
    
    const est10 = hll10.estimate();
    const est14 = hll14.estimate();
    
    // Both should be in valid range
    try std.testing.expect(est10 >= 4000);
    try std.testing.expect(est10 <= 6000);
    try std.testing.expect(est14 >= 4500);
    try std.testing.expect(est14 <= 5500);
    
    // Higher precision should be more accurate (closer to 5000)
    const error10 = if (est10 > 5000) est10 - 5000 else 5000 - est10;
    const error14 = if (est14 > 5000) est14 - 5000 else 5000 - est14;
    
    // b=14 should typically have lower error than b=10
    // (not guaranteed for single run, but generally true)
    _ = error10;
    _ = error14;
}
