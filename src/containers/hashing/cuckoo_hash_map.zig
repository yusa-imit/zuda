const std = @import("std");
const testing = std.testing;

/// CuckooHashMap implements cuckoo hashing with two hash functions and two tables.
/// Provides O(1) worst-case lookup and amortized O(1) insertion with rehashing on cycles.
///
/// Key features:
/// - Two independent hash functions minimize collisions
/// - O(1) worst-case lookup (checks at most 2 positions)
/// - Automatic rehashing with new hash functions on insertion cycles
/// - Configurable max displacement before rehash triggers
///
/// Generic parameters:
/// - K: Key type (must be hashable)
/// - V: Value type
/// - Context: Hash context type (default: void)
/// - hash1Fn: First hash function
/// - hash2Fn: Second hash function
/// - eqlFn: Key equality function
pub fn CuckooHashMap(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime hash1Fn: fn (ctx: Context, key: K) u64,
    comptime hash2Fn: fn (ctx: Context, key: K) u64,
    comptime eqlFn: fn (ctx: Context, a: K, b: K) bool,
) type {
    return struct {
        const Self = @This();

        pub const Entry = struct {
            key: K,
            value: V,
        };

        const Slot = struct {
            entry: Entry,
            occupied: bool,
        };

        pub const Iterator = struct {
            map: *const Self,
            table_idx: usize, // 0 or 1
            slot_idx: usize,

            /// Time: O(1) amortized | Space: O(1)
            pub fn next(self: *Iterator) ?Entry {
                while (self.table_idx < 2) : (self.table_idx += 1) {
                    const table = if (self.table_idx == 0) self.map.table1 else self.map.table2;
                    while (self.slot_idx < table.len) : (self.slot_idx += 1) {
                        if (table[self.slot_idx].occupied) {
                            const entry = table[self.slot_idx].entry;
                            self.slot_idx += 1;
                            return entry;
                        }
                    }
                    self.slot_idx = 0;
                }
                return null;
            }
        };

        allocator: std.mem.Allocator,
        table1: []Slot,
        table2: []Slot,
        context: Context,
        len: usize,
        seed1: u64, // Seed for hash1
        seed2: u64, // Seed for hash2
        max_displacement: usize, // Max cuckoo kicks before rehash

        const DEFAULT_CAPACITY = 16;
        const MAX_DISPLACEMENT = 512;
        const LOAD_FACTOR_THRESHOLD = 0.5; // Rehash when 50% full

        /// Initialize an empty CuckooHashMap
        /// Time: O(1) | Space: O(capacity)
        pub fn init(allocator: std.mem.Allocator, context: Context) !Self {
            return initCapacity(allocator, context, DEFAULT_CAPACITY);
        }

        /// Initialize with specific capacity
        /// Time: O(capacity) | Space: O(capacity)
        pub fn initCapacity(allocator: std.mem.Allocator, context: Context, initial_capacity: usize) !Self {
            const cap = if (initial_capacity < 4) 4 else std.math.ceilPowerOfTwo(usize, initial_capacity) catch return error.CapacityTooLarge;

            const table1 = try allocator.alloc(Slot, cap);
            errdefer allocator.free(table1);
            const table2 = try allocator.alloc(Slot, cap);
            errdefer allocator.free(table2);

            @memset(table1, .{ .entry = undefined, .occupied = false });
            @memset(table2, .{ .entry = undefined, .occupied = false });

            var prng = std.Random.DefaultPrng.init(@as(u64, @intCast(std.time.timestamp())));
            const random = prng.random();

            return Self{
                .allocator = allocator,
                .table1 = table1,
                .table2 = table2,
                .context = context,
                .len = 0,
                .seed1 = random.int(u64),
                .seed2 = random.int(u64),
                .max_displacement = MAX_DISPLACEMENT,
            };
        }

        /// Free all memory
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.table1);
            self.allocator.free(self.table2);
            self.* = undefined;
        }

        /// Create a deep copy
        /// Time: O(capacity) | Space: O(capacity)
        pub fn clone(self: *const Self) !Self {
            const table1 = try self.allocator.alloc(Slot, self.table1.len);
            errdefer self.allocator.free(table1);
            const table2 = try self.allocator.alloc(Slot, self.table2.len);
            errdefer self.allocator.free(table2);

            @memcpy(table1, self.table1);
            @memcpy(table2, self.table2);

            return Self{
                .allocator = self.allocator,
                .table1 = table1,
                .table2 = table2,
                .context = self.context,
                .len = self.len,
                .seed1 = self.seed1,
                .seed2 = self.seed2,
                .max_displacement = self.max_displacement,
            };
        }

        /// Get number of entries
        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.len;
        }

        /// Check if map is empty
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.len == 0;
        }

        /// Get capacity (per table)
        /// Time: O(1) | Space: O(1)
        pub fn capacity(self: *const Self) usize {
            return self.table1.len;
        }

        fn hashToIndex(hash: u64, cap: usize) usize {
            return @as(usize, @intCast(hash & (cap - 1)));
        }

        fn getHash1(self: *const Self, key: K) u64 {
            return hash1Fn(self.context, key) ^ self.seed1;
        }

        fn getHash2(self: *const Self, key: K) u64 {
            return hash2Fn(self.context, key) ^ self.seed2;
        }

        /// Insert or update entry. Returns old value if key existed.
        /// Time: O(1) amortized | Space: O(1) amortized
        pub fn insert(self: *Self, key: K, value: V) !?V {
            // Check load factor and resize if needed
            const load_factor = @as(f64, @floatFromInt(self.len + 1)) / @as(f64, @floatFromInt(self.table1.len));
            if (load_factor > LOAD_FACTOR_THRESHOLD) {
                try self.resize(self.table1.len * 2);
            }

            return try self.insertInternal(key, value);
        }

        fn insertInternal(self: *Self, key: K, value: V) std.mem.Allocator.Error!?V {
            const idx1 = hashToIndex(self.getHash1(key), self.table1.len);
            const idx2 = hashToIndex(self.getHash2(key), self.table2.len);

            // Check if key already exists in either table
            if (self.table1[idx1].occupied and eqlFn(self.context, self.table1[idx1].entry.key, key)) {
                const old = self.table1[idx1].entry.value;
                self.table1[idx1].entry.value = value;
                return old;
            }
            if (self.table2[idx2].occupied and eqlFn(self.context, self.table2[idx2].entry.key, key)) {
                const old = self.table2[idx2].entry.value;
                self.table2[idx2].entry.value = value;
                return old;
            }

            // Insert new entry using cuckoo hashing
            var current_entry = Entry{ .key = key, .value = value };
            var current_table: u1 = 0; // 0 = table1, 1 = table2
            var displacement: usize = 0;

            while (displacement < self.max_displacement) : (displacement += 1) {
                const idx = if (current_table == 0)
                    hashToIndex(self.getHash1(current_entry.key), self.table1.len)
                else
                    hashToIndex(self.getHash2(current_entry.key), self.table2.len);

                const table = if (current_table == 0) &self.table1 else &self.table2;

                if (!table.*[idx].occupied) {
                    // Empty slot found, insert here
                    table.*[idx] = .{ .entry = current_entry, .occupied = true };
                    self.len += 1;
                    return null;
                }

                // Kick out existing entry and swap
                const temp = table.*[idx].entry;
                table.*[idx].entry = current_entry;
                current_entry = temp;
                current_table = 1 - current_table; // Switch table
            }

            // Max displacement reached, rehash with new seeds
            try self.rehash();
            return try self.insertInternal(key, value);
        }

        fn resize(self: *Self, new_capacity: usize) !void {
            const old_table1 = self.table1;
            const old_table2 = self.table2;
            defer self.allocator.free(old_table1);
            defer self.allocator.free(old_table2);

            const new_table1 = try self.allocator.alloc(Slot, new_capacity);
            errdefer self.allocator.free(new_table1);
            const new_table2 = try self.allocator.alloc(Slot, new_capacity);
            errdefer self.allocator.free(new_table2);

            @memset(new_table1, .{ .entry = undefined, .occupied = false });
            @memset(new_table2, .{ .entry = undefined, .occupied = false });

            self.table1 = new_table1;
            self.table2 = new_table2;
            const old_len = self.len;
            self.len = 0;

            // Reinsert all entries
            for (old_table1) |slot| {
                if (slot.occupied) {
                    _ = try self.insertInternal(slot.entry.key, slot.entry.value);
                }
            }
            for (old_table2) |slot| {
                if (slot.occupied) {
                    _ = try self.insertInternal(slot.entry.key, slot.entry.value);
                }
            }

            std.debug.assert(self.len == old_len);
        }

        fn rehash(self: *Self) !void {
            // Change hash seeds to get new hash functions
            var prng = std.Random.DefaultPrng.init(@as(u64, @intCast(std.time.nanoTimestamp())));
            const random = prng.random();
            self.seed1 = random.int(u64);
            self.seed2 = random.int(u64);

            // Reinsert all entries with new hash functions
            const old_table1 = self.table1;
            const old_table2 = self.table2;
            defer self.allocator.free(old_table1);
            defer self.allocator.free(old_table2);

            const new_table1 = try self.allocator.alloc(Slot, self.table1.len);
            errdefer self.allocator.free(new_table1);
            const new_table2 = try self.allocator.alloc(Slot, self.table2.len);
            errdefer self.allocator.free(new_table2);

            @memset(new_table1, .{ .entry = undefined, .occupied = false });
            @memset(new_table2, .{ .entry = undefined, .occupied = false });

            self.table1 = new_table1;
            self.table2 = new_table2;
            const old_len = self.len;
            self.len = 0;

            for (old_table1) |slot| {
                if (slot.occupied) {
                    _ = try self.insertInternal(slot.entry.key, slot.entry.value);
                }
            }
            for (old_table2) |slot| {
                if (slot.occupied) {
                    _ = try self.insertInternal(slot.entry.key, slot.entry.value);
                }
            }

            std.debug.assert(self.len == old_len);
        }

        /// Remove entry by key. Returns value if found.
        /// Time: O(1) worst-case | Space: O(1)
        pub fn remove(self: *Self, key: K) ?V {
            const idx1 = hashToIndex(self.getHash1(key), self.table1.len);
            const idx2 = hashToIndex(self.getHash2(key), self.table2.len);

            if (self.table1[idx1].occupied and eqlFn(self.context, self.table1[idx1].entry.key, key)) {
                const value = self.table1[idx1].entry.value;
                self.table1[idx1].occupied = false;
                self.len -= 1;
                return value;
            }

            if (self.table2[idx2].occupied and eqlFn(self.context, self.table2[idx2].entry.key, key)) {
                const value = self.table2[idx2].entry.value;
                self.table2[idx2].occupied = false;
                self.len -= 1;
                return value;
            }

            return null;
        }

        /// Lookup value by key
        /// Time: O(1) worst-case | Space: O(1)
        pub fn get(self: *const Self, key: K) ?V {
            const idx1 = hashToIndex(self.getHash1(key), self.table1.len);
            const idx2 = hashToIndex(self.getHash2(key), self.table2.len);

            if (self.table1[idx1].occupied and eqlFn(self.context, self.table1[idx1].entry.key, key)) {
                return self.table1[idx1].entry.value;
            }

            if (self.table2[idx2].occupied and eqlFn(self.context, self.table2[idx2].entry.key, key)) {
                return self.table2[idx2].entry.value;
            }

            return null;
        }

        /// Check if key exists
        /// Time: O(1) worst-case | Space: O(1)
        pub fn contains(self: *const Self, key: K) bool {
            return self.get(key) != null;
        }

        /// Get iterator over all entries
        /// Time: O(1) | Space: O(1)
        pub fn iterator(self: *const Self) Iterator {
            return Iterator{
                .map = self,
                .table_idx = 0,
                .slot_idx = 0,
            };
        }

        /// Clear all entries
        /// Time: O(capacity) | Space: O(1)
        pub fn clear(self: *Self) void {
            @memset(self.table1, .{ .entry = undefined, .occupied = false });
            @memset(self.table2, .{ .entry = undefined, .occupied = false });
            self.len = 0;
        }

        /// Validate internal invariants
        /// Time: O(n) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            var actual_count: usize = 0;

            // Check table1
            for (self.table1, 0..) |slot, i| {
                if (slot.occupied) {
                    actual_count += 1;
                    const idx1 = hashToIndex(self.getHash1(slot.entry.key), self.table1.len);
                    if (idx1 != i) {
                        return error.InvalidHashPosition;
                    }
                }
            }

            // Check table2
            for (self.table2, 0..) |slot, i| {
                if (slot.occupied) {
                    actual_count += 1;
                    const idx2 = hashToIndex(self.getHash2(slot.entry.key), self.table2.len);
                    if (idx2 != i) {
                        return error.InvalidHashPosition;
                    }
                }
            }

            if (actual_count != self.len) {
                return error.InvalidCount;
            }

            // Verify capacity is power of two
            if (!std.math.isPowerOfTwo(self.table1.len)) {
                return error.InvalidCapacity;
            }
            if (!std.math.isPowerOfTwo(self.table2.len)) {
                return error.InvalidCapacity;
            }

            // Verify tables have same capacity
            if (self.table1.len != self.table2.len) {
                return error.TableSizeMismatch;
            }
        }

        /// Format for debugging
        pub fn format(
            self: *const Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("CuckooHashMap{{ len={}, capacity={} }}", .{ self.len, self.table1.len });
        }
    };
}

// Convenience alias for common case
/// Creates hash map with automatic context.
/// Time: O(1) amortized | Space: O(n)
pub fn AutoCuckooHashMap(comptime K: type, comptime V: type) type {
    const Context = struct {
        pub fn hash1(_: @This(), key: K) u64 {
            return std.hash.Wyhash.hash(0, std.mem.asBytes(&key));
        }

        pub fn hash2(_: @This(), key: K) u64 {
            return std.hash.Wyhash.hash(1, std.mem.asBytes(&key));
        }

        pub fn eql(_: @This(), a: K, b: K) bool {
            return a == b;
        }
    };

    return CuckooHashMap(K, V, Context, Context.hash1, Context.hash2, Context.eql);
}

// --- Tests ---

test "CuckooHashMap: basic insert and get" {
    var map = try AutoCuckooHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    try testing.expect(map.isEmpty());
    try testing.expectEqual(@as(usize, 0), map.count());

    const old = try map.insert(1, 100);
    try testing.expectEqual(@as(?u32, null), old);
    try testing.expectEqual(@as(usize, 1), map.count());

    const value = map.get(1);
    try testing.expectEqual(@as(?u32, 100), value);
}

test "CuckooHashMap: update existing key" {
    var map = try AutoCuckooHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    _ = try map.insert(1, 100);
    const old = try map.insert(1, 200);
    try testing.expectEqual(@as(?u32, 100), old);
    try testing.expectEqual(@as(u32, 200), map.get(1).?);
    try testing.expectEqual(@as(usize, 1), map.count());
}

test "CuckooHashMap: remove" {
    var map = try AutoCuckooHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    _ = try map.insert(1, 100);
    _ = try map.insert(2, 200);
    try testing.expectEqual(@as(usize, 2), map.count());

    const removed = map.remove(1);
    try testing.expectEqual(@as(?u32, 100), removed);
    try testing.expectEqual(@as(usize, 1), map.count());
    try testing.expectEqual(@as(?u32, null), map.get(1));

    const not_found = map.remove(999);
    try testing.expectEqual(@as(?u32, null), not_found);
}

test "CuckooHashMap: contains" {
    var map = try AutoCuckooHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    try testing.expect(!map.contains(1));
    _ = try map.insert(1, 100);
    try testing.expect(map.contains(1));
    _ = map.remove(1);
    try testing.expect(!map.contains(1));
}

test "CuckooHashMap: iterator" {
    var map = try AutoCuckooHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    _ = try map.insert(1, 100);
    _ = try map.insert(2, 200);
    _ = try map.insert(3, 300);

    var sum: u32 = 0;
    var iter = map.iterator();
    while (iter.next()) |entry| {
        sum += entry.value;
    }
    try testing.expectEqual(@as(u32, 600), sum);
}

test "CuckooHashMap: clear" {
    var map = try AutoCuckooHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    _ = try map.insert(1, 100);
    _ = try map.insert(2, 200);
    try testing.expectEqual(@as(usize, 2), map.count());

    map.clear();
    try testing.expectEqual(@as(usize, 0), map.count());
    try testing.expect(map.isEmpty());
    try testing.expectEqual(@as(?u32, null), map.get(1));
}

test "CuckooHashMap: stress test with many insertions" {
    var map = try AutoCuckooHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    const n = 1000;
    for (0..n) |i| {
        const key = @as(u32, @intCast(i));
        _ = try map.insert(key, key * 2);
    }

    try testing.expectEqual(@as(usize, n), map.count());

    // Verify all values
    for (0..n) |i| {
        const key = @as(u32, @intCast(i));
        const value = map.get(key).?;
        try testing.expectEqual(key * 2, value);
    }

    // Validate invariants
    try map.validate();
}

test "CuckooHashMap: collision handling with sequential keys" {
    var map = try AutoCuckooHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    // Insert many sequential keys to test cuckoo hashing
    for (0..100) |i| {
        const key = @as(u32, @intCast(i));
        _ = try map.insert(key, key);
    }

    try testing.expectEqual(@as(usize, 100), map.count());

    for (0..100) |i| {
        const key = @as(u32, @intCast(i));
        try testing.expect(map.contains(key));
    }

    try map.validate();
}

test "CuckooHashMap: clone" {
    var map = try AutoCuckooHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    _ = try map.insert(1, 100);
    _ = try map.insert(2, 200);

    var cloned = try map.clone();
    defer cloned.deinit();

    try testing.expectEqual(map.count(), cloned.count());
    try testing.expectEqual(map.get(1).?, cloned.get(1).?);
    try testing.expectEqual(map.get(2).?, cloned.get(2).?);

    // Modify clone, original should be unchanged
    _ = try cloned.insert(3, 300);
    try testing.expectEqual(@as(usize, 2), map.count());
    try testing.expectEqual(@as(usize, 3), cloned.count());
}

test "CuckooHashMap: capacity growth" {
    var map = try AutoCuckooHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    const initial_capacity = map.capacity();

    // Insert enough to trigger resize
    for (0..100) |i| {
        const key = @as(u32, @intCast(i));
        _ = try map.insert(key, key);
    }

    try testing.expect(map.capacity() > initial_capacity);
    try testing.expectEqual(@as(usize, 100), map.count());
    try map.validate();
}

test "CuckooHashMap: memory leak test" {
    var map = try AutoCuckooHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    for (0..100) |i| {
        const key = @as(u32, @intCast(i));
        _ = try map.insert(key, key);
    }
    try testing.expectEqual(@as(usize, 100), map.count());

    for (0..50) |i| {
        const key = @as(u32, @intCast(i));
        _ = map.remove(key);
    }
    try testing.expectEqual(@as(usize, 50), map.count());

    // Verify remaining keys are correct
    for (50..100) |i| {
        const key = @as(u32, @intCast(i));
        try testing.expectEqual(@as(?u32, key), map.get(key));
    }
    // Allocator will detect leaks on deinit
}

test "CuckooHashMap: validate invariants" {
    var map = try AutoCuckooHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    try map.validate();
    try testing.expectEqual(@as(usize, 0), map.count());

    _ = try map.insert(1, 100);
    try map.validate();
    try testing.expectEqual(@as(usize, 1), map.count());
    try testing.expectEqual(@as(?u32, 100), map.get(1));

    _ = try map.insert(2, 200);
    try map.validate();
    try testing.expectEqual(@as(usize, 2), map.count());
    try testing.expectEqual(@as(?u32, 200), map.get(2));

    _ = map.remove(1);
    try map.validate();
    try testing.expectEqual(@as(usize, 1), map.count());
    try testing.expectEqual(@as(?u32, null), map.get(1));
    try testing.expectEqual(@as(?u32, 200), map.get(2));
}

test "CuckooHashMap: empty operations" {
    var map = try AutoCuckooHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    try testing.expectEqual(@as(?u32, null), map.get(1));
    try testing.expectEqual(@as(?u32, null), map.remove(1));
    try testing.expect(!map.contains(1));

    var iter = map.iterator();
    try testing.expectEqual(@as(?AutoCuckooHashMap(u32, u32).Entry, null), iter.next());
}

test "CuckooHashMap: O(1) worst-case lookup guarantee" {
    var map = try AutoCuckooHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    // Insert many items
    for (0..500) |i| {
        const key = @as(u32, @intCast(i));
        _ = try map.insert(key, key);
    }

    // Each lookup should check at most 2 positions (table1[hash1] and table2[hash2])
    // This is guaranteed by the cuckoo hashing algorithm
    for (0..500) |i| {
        const key = @as(u32, @intCast(i));
        const value = map.get(key);
        try testing.expectEqual(@as(?u32, key), value);
    }

    try map.validate();
}
