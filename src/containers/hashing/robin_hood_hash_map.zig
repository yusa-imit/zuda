const std = @import("std");
const testing = std.testing;

/// RobinHoodHashMap implements open addressing with Robin Hood hashing.
/// The Robin Hood heuristic reduces variance in probe lengths by "stealing from the rich
/// and giving to the poor" - entries with shorter probe distances yield their position
/// to entries with longer probe distances.
///
/// Key features:
/// - O(1) average-case lookup, insert, and delete
/// - Low variance in probe lengths leads to predictable performance
/// - Better cache locality than chaining or cuckoo hashing
/// - Backward shift deletion to maintain probe sequence invariants
///
/// Generic parameters:
/// - K: Key type (must be hashable)
/// - V: Value type
/// - Context: Hash context type (default: void)
/// - hashFn: Hash function
/// - eqlFn: Key equality function
pub fn RobinHoodHashMap(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime hashFn: fn (ctx: Context, key: K) u64,
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
            psl: u32, // Probe Sequence Length
            occupied: bool,
        };

        pub const Iterator = struct {
            map: *const Self,
            index: usize,

            /// Time: O(1) amortized | Space: O(1)
            pub fn next(self: *Iterator) ?Entry {
                while (self.index < self.map.slots.len) : (self.index += 1) {
                    if (self.map.slots[self.index].occupied) {
                        const entry = self.map.slots[self.index].entry;
                        self.index += 1;
                        return entry;
                    }
                }
                return null;
            }
        };

        allocator: std.mem.Allocator,
        slots: []Slot,
        context: Context,
        len: usize,
        seed: u64,

        const DEFAULT_CAPACITY = 16;
        const LOAD_FACTOR_THRESHOLD = 0.875; // 87.5% load factor

        /// Initialize an empty RobinHoodHashMap
        /// Time: O(1) | Space: O(capacity)
        pub fn init(allocator: std.mem.Allocator, context: Context) !Self {
            return initCapacity(allocator, context, DEFAULT_CAPACITY);
        }

        /// Initialize with specific capacity
        /// Time: O(capacity) | Space: O(capacity)
        pub fn initCapacity(allocator: std.mem.Allocator, context: Context, initial_capacity: usize) !Self {
            const cap = if (initial_capacity < 4) 4 else std.math.ceilPowerOfTwo(usize, initial_capacity) catch return error.CapacityTooLarge;

            const slots = try allocator.alloc(Slot, cap);
            @memset(slots, .{ .entry = undefined, .psl = 0, .occupied = false });

            var prng = std.Random.DefaultPrng.init(@as(u64, @intCast(std.time.timestamp())));
            const random = prng.random();

            return Self{
                .allocator = allocator,
                .slots = slots,
                .context = context,
                .len = 0,
                .seed = random.int(u64),
            };
        }

        /// Free all memory
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.slots);
            self.* = undefined;
        }

        /// Create a deep copy
        /// Time: O(capacity) | Space: O(capacity)
        pub fn clone(self: *const Self) !Self {
            const slots = try self.allocator.alloc(Slot, self.slots.len);
            @memcpy(slots, self.slots);

            return Self{
                .allocator = self.allocator,
                .slots = slots,
                .context = self.context,
                .len = self.len,
                .seed = self.seed,
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

        /// Get capacity
        /// Time: O(1) | Space: O(1)
        pub fn capacity(self: *const Self) usize {
            return self.slots.len;
        }

        fn getHash(self: *const Self, key: K) u64 {
            return hashFn(self.context, key) ^ self.seed;
        }

        fn desiredPosition(self: *const Self, key: K) usize {
            const hash = self.getHash(key);
            return @as(usize, @intCast(hash & (self.slots.len - 1)));
        }

        fn probeDistance(self: *const Self, slot_index: usize, desired: usize) u32 {
            if (slot_index >= desired) {
                return @as(u32, @intCast(slot_index - desired));
            } else {
                // Wrapped around
                return @as(u32, @intCast((self.slots.len - desired) + slot_index));
            }
        }

        /// Insert or update entry. Returns old value if key existed.
        /// Time: O(1) amortized | Space: O(1) amortized
        pub fn insert(self: *Self, key: K, value: V) !?V {
            // Check load factor and resize if needed
            const load_factor = @as(f64, @floatFromInt(self.len + 1)) / @as(f64, @floatFromInt(self.slots.len));
            if (load_factor > LOAD_FACTOR_THRESHOLD) {
                try self.resize(self.slots.len * 2);
            }

            return self.insertInternal(key, value);
        }

        fn insertInternal(self: *Self, key: K, value: V) ?V {
            var current_entry = Entry{ .key = key, .value = value };
            var desired = self.desiredPosition(key);
            var current_psl: u32 = 0;
            var pos = desired;

            while (true) {
                if (!self.slots[pos].occupied) {
                    // Empty slot, insert here
                    self.slots[pos] = .{
                        .entry = current_entry,
                        .psl = current_psl,
                        .occupied = true,
                    };
                    self.len += 1;
                    return null;
                }

                // Check if this is an update
                if (eqlFn(self.context, self.slots[pos].entry.key, current_entry.key)) {
                    const old_value = self.slots[pos].entry.value;
                    self.slots[pos].entry.value = current_entry.value;
                    return old_value;
                }

                // Robin Hood heuristic: if current entry has longer PSL, swap with existing
                if (current_psl > self.slots[pos].psl) {
                    // Swap entries
                    const temp_entry = self.slots[pos].entry;
                    const temp_psl = self.slots[pos].psl;
                    self.slots[pos] = .{
                        .entry = current_entry,
                        .psl = current_psl,
                        .occupied = true,
                    };
                    current_entry = temp_entry;
                    current_psl = temp_psl;
                    desired = self.desiredPosition(current_entry.key);
                }

                // Move to next slot
                pos = (pos + 1) & (self.slots.len - 1);
                current_psl += 1;
            }
        }

        fn resize(self: *Self, new_capacity: usize) !void {
            const old_slots = self.slots;
            defer self.allocator.free(old_slots);

            const new_slots = try self.allocator.alloc(Slot, new_capacity);
            @memset(new_slots, .{ .entry = undefined, .psl = 0, .occupied = false });

            self.slots = new_slots;
            const old_len = self.len;
            self.len = 0;

            // Reinsert all entries
            for (old_slots) |slot| {
                if (slot.occupied) {
                    _ = self.insertInternal(slot.entry.key, slot.entry.value);
                }
            }

            std.debug.assert(self.len == old_len);
        }

        /// Remove entry by key. Returns value if found.
        /// Time: O(1) amortized | Space: O(1)
        pub fn remove(self: *Self, key: K) ?V {
            const desired = self.desiredPosition(key);
            var pos = desired;
            var psl: u32 = 0;

            while (true) {
                if (!self.slots[pos].occupied) {
                    // Not found
                    return null;
                }

                // If PSL is less than current probe distance, key doesn't exist
                if (self.slots[pos].psl < psl) {
                    return null;
                }

                if (eqlFn(self.context, self.slots[pos].entry.key, key)) {
                    // Found it
                    const value = self.slots[pos].entry.value;
                    self.len -= 1;

                    // Backward shift deletion
                    var current = pos;
                    while (true) {
                        const next = (current + 1) & (self.slots.len - 1);

                        // Stop if next slot is empty or has PSL = 0
                        if (!self.slots[next].occupied or self.slots[next].psl == 0) {
                            self.slots[current].occupied = false;
                            break;
                        }

                        // Shift entry backward
                        self.slots[current] = .{
                            .entry = self.slots[next].entry,
                            .psl = self.slots[next].psl - 1,
                            .occupied = true,
                        };
                        current = next;
                    }

                    return value;
                }

                pos = (pos + 1) & (self.slots.len - 1);
                psl += 1;
            }
        }

        /// Lookup value by key
        /// Time: O(1) average-case | Space: O(1)
        pub fn get(self: *const Self, key: K) ?V {
            const desired = self.desiredPosition(key);
            var pos = desired;
            var psl: u32 = 0;

            while (true) {
                if (!self.slots[pos].occupied) {
                    return null;
                }

                // If PSL is less than current probe distance, key doesn't exist
                if (self.slots[pos].psl < psl) {
                    return null;
                }

                if (eqlFn(self.context, self.slots[pos].entry.key, key)) {
                    return self.slots[pos].entry.value;
                }

                pos = (pos + 1) & (self.slots.len - 1);
                psl += 1;
            }
        }

        /// Check if key exists
        /// Time: O(1) average-case | Space: O(1)
        pub fn contains(self: *const Self, key: K) bool {
            return self.get(key) != null;
        }

        /// Get iterator over all entries
        /// Time: O(1) | Space: O(1)
        pub fn iterator(self: *const Self) Iterator {
            return Iterator{
                .map = self,
                .index = 0,
            };
        }

        /// Clear all entries
        /// Time: O(capacity) | Space: O(1)
        pub fn clear(self: *Self) void {
            @memset(self.slots, .{ .entry = undefined, .psl = 0, .occupied = false });
            self.len = 0;
        }

        /// Validate internal invariants
        /// Time: O(n) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            var actual_count: usize = 0;

            for (self.slots, 0..) |slot, i| {
                if (slot.occupied) {
                    actual_count += 1;

                    // Verify PSL is correct
                    const desired = self.desiredPosition(slot.entry.key);
                    const expected_psl = self.probeDistance(i, desired);
                    if (slot.psl != expected_psl) {
                        return error.InvalidPSL;
                    }

                    // Verify entry can be found by lookup
                    const found = self.get(slot.entry.key);
                    if (found == null) {
                        return error.EntryNotFoundByLookup;
                    }
                }
            }

            if (actual_count != self.len) {
                return error.InvalidCount;
            }

            // Verify capacity is power of two
            if (!std.math.isPowerOfTwo(self.slots.len)) {
                return error.InvalidCapacity;
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
            try writer.print("RobinHoodHashMap{{ len={}, capacity={} }}", .{ self.len, self.slots.len });
        }
    };
}

// Default hash and equality context for common types
pub const AutoContext = struct {
    /// Computes hash for the key.
    /// Time: O(1) | Space: O(1)
    pub fn hash(ctx: @This(), key: anytype) u64 {
        _ = ctx;
        return std.hash.Wyhash.hash(0, std.mem.asBytes(&key));
    }

    /// Checks equality of two keys.
    /// Time: O(1) | Space: O(1)
    pub fn eql(ctx: @This(), a: anytype, b: @TypeOf(a)) bool {
        _ = ctx;
        return a == b;
    }
};

// Convenience alias for common case
/// Creates hash map with automatic context.
/// Time: O(1) amortized | Space: O(n)
pub fn AutoRobinHoodHashMap(comptime K: type, comptime V: type) type {
    return RobinHoodHashMap(K, V, AutoContext, AutoContext.hash, AutoContext.eql);
}

// --- Tests ---

test "RobinHoodHashMap: basic insert and get" {
    var map = try AutoRobinHoodHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    try testing.expect(map.isEmpty());
    try testing.expectEqual(@as(usize, 0), map.count());

    const old = try map.insert(1, 100);
    try testing.expectEqual(@as(?u32, null), old);
    try testing.expectEqual(@as(usize, 1), map.count());

    const value = map.get(1);
    try testing.expectEqual(@as(?u32, 100), value);
}

test "RobinHoodHashMap: update existing key" {
    var map = try AutoRobinHoodHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    _ = try map.insert(1, 100);
    const old = try map.insert(1, 200);
    try testing.expectEqual(@as(?u32, 100), old);
    try testing.expectEqual(@as(u32, 200), map.get(1).?);
    try testing.expectEqual(@as(usize, 1), map.count());
}

test "RobinHoodHashMap: remove" {
    var map = try AutoRobinHoodHashMap(u32, u32).init(testing.allocator, .{});
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

test "RobinHoodHashMap: contains" {
    var map = try AutoRobinHoodHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    try testing.expect(!map.contains(1));
    _ = try map.insert(1, 100);
    try testing.expect(map.contains(1));
    _ = map.remove(1);
    try testing.expect(!map.contains(1));
}

test "RobinHoodHashMap: iterator" {
    var map = try AutoRobinHoodHashMap(u32, u32).init(testing.allocator, .{});
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

test "RobinHoodHashMap: clear" {
    var map = try AutoRobinHoodHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    _ = try map.insert(1, 100);
    _ = try map.insert(2, 200);
    try testing.expectEqual(@as(usize, 2), map.count());

    map.clear();
    try testing.expectEqual(@as(usize, 0), map.count());
    try testing.expect(map.isEmpty());
    try testing.expectEqual(@as(?u32, null), map.get(1));
}

test "RobinHoodHashMap: stress test with many insertions" {
    var map = try AutoRobinHoodHashMap(u32, u32).init(testing.allocator, .{});
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

test "RobinHoodHashMap: Robin Hood heuristic reduces variance" {
    var map = try AutoRobinHoodHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    // Insert many items to trigger Robin Hood swaps
    for (0..200) |i| {
        const key = @as(u32, @intCast(i));
        _ = try map.insert(key, key);
    }

    // Calculate max PSL (should be relatively low due to Robin Hood)
    var max_psl: u32 = 0;
    for (map.slots) |slot| {
        if (slot.occupied and slot.psl > max_psl) {
            max_psl = slot.psl;
        }
    }

    // With Robin Hood heuristic, max PSL should be much smaller than without
    // For 200 items with ~87.5% load factor, expect max PSL < 20
    try testing.expect(max_psl < 20);
    try map.validate();
}

test "RobinHoodHashMap: clone" {
    var map = try AutoRobinHoodHashMap(u32, u32).init(testing.allocator, .{});
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

test "RobinHoodHashMap: capacity growth" {
    var map = try AutoRobinHoodHashMap(u32, u32).init(testing.allocator, .{});
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

test "RobinHoodHashMap: memory leak test" {
    var map = try AutoRobinHoodHashMap(u32, u32).init(testing.allocator, .{});
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

test "RobinHoodHashMap: validate invariants" {
    var map = try AutoRobinHoodHashMap(u32, u32).init(testing.allocator, .{});
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

test "RobinHoodHashMap: empty operations" {
    var map = try AutoRobinHoodHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    try testing.expectEqual(@as(?u32, null), map.get(1));
    try testing.expectEqual(@as(?u32, null), map.remove(1));
    try testing.expect(!map.contains(1));

    var iter = map.iterator();
    try testing.expectEqual(@as(?AutoRobinHoodHashMap(u32, u32).Entry, null), iter.next());
}

test "RobinHoodHashMap: backward shift deletion maintains invariants" {
    var map = try AutoRobinHoodHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    // Insert items that will create a chain
    for (0..20) |i| {
        const key = @as(u32, @intCast(i));
        _ = try map.insert(key, key);
    }

    try map.validate();

    // Remove items from the middle of chains
    for (0..10) |i| {
        const key = @as(u32, @intCast(i * 2));
        _ = map.remove(key);
    }

    // Invariants should still hold after backward shift deletion
    try map.validate();

    // Verify remaining items are still accessible
    for (0..10) |i| {
        const key = @as(u32, @intCast(i * 2 + 1));
        try testing.expect(map.contains(key));
    }
}

test "RobinHoodHashMap: high load factor performance" {
    var map = try AutoRobinHoodHashMap(u32, u32).init(testing.allocator, .{});
    defer map.deinit();

    // Insert up to 87.5% load factor
    const target_count = (map.capacity() * 7) / 8;
    for (0..target_count) |i| {
        const key = @as(u32, @intCast(i));
        _ = try map.insert(key, key);
    }

    try testing.expectEqual(target_count, map.count());

    // All lookups should still work efficiently
    for (0..target_count) |i| {
        const key = @as(u32, @intCast(i));
        try testing.expectEqual(@as(?u32, key), map.get(key));
    }

    try map.validate();
}
